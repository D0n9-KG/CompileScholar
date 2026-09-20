# OCTOPACK: INSTRUCTION TUNING CODE LARGE LANGUAGE MODELS

![](images/55653db8163f810701cce44b9e99a98eecb2203aeaeed30028bfe1818ba68b01.jpg)

<details>
<summary>natural_image</summary>

Purple cartoon octopus character wearing a Big Code cap and headset, using a laptop (no text or symbols on the robot or background)
</details>

Niklas Muennighoff

Qian Liu

Armel Zebaze

Qinkai Zheng

Binyuan Hui

Terry Yue Zhuo

Swayam Singh

Xiangru Tang

Leandro von Werra

Shayne Longpre

n.muennighoff@gmail.com

# ABSTRACT

Finetuning large language models (LLMs) on instructions leads to vast performance improvements on natural language tasks. We apply instruction tuning using code, leveraging the natural structure of Git commits, which pair code changes with human instructions. We compile COMMITPACK: 4 terabytes of Git commits across 350 programming languages. We benchmark COMMITPACK against other natural and synthetic code instructions (xP3x, Self-Instruct, OASST) on the 16B parameter StarCoder model, and achieve state-of-the-art performance among models not trained on OpenAI outputs, on the HumanEval Python benchmark (46.2% pass@1). We further introduce HUMANEVALPACK, expanding the HumanEval benchmark to a total of 3 coding tasks (Code Repair, Code Explanation, Code Synthesis) across 6 languages (Python, JavaScript, Java, Go, C++, Rust). Our models, OCTOCODER and OCTOGEEX, achieve the best performance across HUMANEVALPACK among all permissive models, demonstrating COMMITPACK's benefits in generalizing to a wider set of languages and natural coding tasks. Code, models and data are freely available at https://github.com/bigcode-project/octopack.

# 1) CommitPack

import numpy as np
import matplotlib.pyplot as plt
Code Before

\# generate sample data
x\_data = np.linspace(-5, 5, 20)
y\_data = np.random.normal(0.0, 1.0, x\_data.size)

plt.plot(x\_data, y\_data, 'o')
plt.show()

Change to sin() function with noise

Commit Message

import math
import numpy as np
import matplotlib.pyplot as plt

# generate sample data
x\_data = np.linspace(-math.pi, math.pi, 30)
y\_data = np.sin(x\_data) + np.random.normal(0.0, 0.1, x\_data.size)

plt.plot(x\_data, y\_data, 'o')
plt.show()

![](images/ca3ea60698b9e90c6ae0f0a37f48dda1b8ba81a35ed7b94c2e4fabcef4288cc1.jpg)

<details>
<summary>bar</summary>

2) HumanEvalPack
| Category | BLOOMZ | StarChat-β | CodeGeeX2 | StarCoder | OctoGeeX | OctoCoder | InstructCodeT5+ | WizardCoder | GPT4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fixing Code | 18.0 | 16.0 | 17.0 | 19.0 | 24.4 | 27.0 | 5.0 | 25.7 | 47.8 |
| Explaining Code | 10.0 | 22.9 | 3.0 | 2.0 | 24.5 | 24.5 | 7.0 | 27.5 | 52.1 |
| Synthesizing Code | 12.0 | 30.9 | 30.9 | 30.9 | 35.5 | 35.5 | 18.0 | 40.1 | 78.3 |
</details>

Figure 1: OCTOPACK Overview. 1) Sample from our 4TB dataset, COMMITPACK. 2) Performance of OCTOCODER, OCTOGEEX and other code models including non-permissive ones (WizardCoder, GPT-4) on HUMANEVALPACK spanning 3 coding tasks and 6 programming languages.

# 1 INTRODUCTION

Finetuning large language models (LLMs) on a variety of language tasks explained via instructions (instruction tuning) has been shown to improve model usability and general performance (Wei et al., 2022; Sanh et al., 2022; Min et al., 2022; Ouyang et al., 2022). The instruction tuning paradigm has also proven successful for models trained on visual (Liu et al., 2023a; Li et al., 2023a), audio (Zhang et al., 2023b) and multilingual (Muennighoff et al., 2022b; Wang et al., 2022b) data.

In this work, we instruction tune LLMs on the coding modality. While Code LLMs can already be indirectly instructed to generate desired code using code comments, this procedure is brittle and does not work when the desired output is natural language, such as explaining code. Explicit instructing tuning of Code LLMs may improve their steerability and enable their application to more tasks. Concurrently to our work, three instruction tuned Code LLMs have been proposed: PanGu-Coder2 (Shen et al., 2023), WizardCoder (Luo et al., 2023) and InstructCodeT5+ (Wang et al., 2023c). These models rely on more capable and closed models from the OpenAI API $^{1}$ to create their instruction training data. This approach is problematic as (1) closed-source APIs keep changing and have unpredictable availability (Pozzobon et al., 2023; Chen et al., 2023a), (2) it relies on the assumption that a more capable model exists (3) it can reinforce model hallucination (Gudibande et al., 2023) and (4), depending on legal interpretation, OpenAI’s terms of use $^{2}$ forbid such models: “...You may not...use output from the Services to develop models that compete with OpenAI...”. Thus, we consider models trained on OpenAI outputs not usable for commercial purposes in practice and classify them as non-permissive in this work.

We focus on more permissively licensed data and avoid using a closed-source model to generate synthetic data. We benchmark four popular sources of code instruction data: (1) xP3x (Muennighoff et al., 2022b), which contains data from common code benchmarks, (2) Self-Instruct (Wang et al., 2023a) data we create using a permissive Code LLM, (3) OASST (Köpf et al., 2023), which contains mostly natural language data and few code examples and (4) COMMITPACK, our new 4TB dataset of Git commits. Instruction tuning's primary purpose is to expand models' generalization abilities to a wide variety of tasks and settings. Thus, we extend the code synthesis benchmark, HumanEval (Chen et al., 2021; Zheng et al., 2023), to create HUMANEVALPACK: A code benchmark covering code synthesis, code repair, and code explanation across six programming languages.

Instruction tuning StarCoder (Li et al., 2023b) on a filtered variant of COMMITPACK and OASST leads to our best model, OCTOCODER, which surpasses all other openly licensed models (Figure 1), but falls short of the much larger GPT-4 (OpenAI, 2023). GPT-4 is close to maximum performance on the code synthesis variant, notably with a pass@1 score of $86.6\%$ on Python HumanEval. However, it performs significantly worse on the code fixing and explanation variants of HUMANEVALPACK, which we introduce. This suggests that the original HumanEval benchmark may soon cease to be useful due to models reaching close to the maximum performance. Our more challenging evaluation variants provide room for future LLMs to improve on the performance of the current state-of-the-art.

In summary, we contribute:

- COMMITPACK and COMMITPACKFT: 4TB of permissively licensed code commits across 350 programming languages for pretraining and a filtered 2GB variant containing high-quality code instructions used for finetuning   
- HUMANEVALPACK: A benchmark for Code LLM generalization, spanning three scenarios (Code Repair, Code Explanation, Code Synthesis) and 6 programming languages (Python, JavaScript, Java, Go, C++, Rust)   
- OCTOCODER and OCTOGEEX: The best permissive Code LLMs

# 2 COMMITPACK: CODE INSTRUCTION DATA

Prior work has shown that models can generalize to languages included in pretraining, but absent during instruction tuning (Muennighoff et al., 2022b). However, they also show that including such

![](images/d557c9a791093dcb4023596cd90eafc034a33eab28332df7425dff17a0365781.jpg)

<details>
<summary>bar_stacked</summary>

| Language | # Samples CommitPack |
| :--- | :--- |
| Markdown | 100000 |
| Python | 80000 |
| JavaScript | 70000 |
| Java | 30000 |
| JSON | 50000 |
| Ruby | 60000 |
| C | 10000 |
| YAML | 120000 |
| PHP | 40000 |
| C++ | 60000 |
| HTML | 30000 |
| XML | 20000 |
| Text | 15000 |
| Go | 5000 |
| Shell | 4000 |
| C# | 1500 |
| TypeScript | 800 |
| CSS | 700 |
| reStructuredText | 900 |
| Perl | 300 |
| Makefile | 150 |
| Swift | 60 |
| Scala | 70 |
| INI | 80 |
| Rust | 40 |
| JavaScript | 70 |
| SCSS | 80 |
| Dockerfile | 120 |
| HTML ERB | 150 |
| Nix | 20 |
| Haskell | 25 |
| gettextCatalog | 25 |
| Clojure | 35 |
| JSX | 35 |
| Lua | 15 |
| SQL | 35 |
| Kotlin | 35 |
| Groovy | 25 |
| AsciiDoc | 25 |
| Jupyter Notebook | 25 |
| Erlang | 25 |
| Tex | 25 |
| Less | 25 |
| Emacs Lisp | 25 |
| OCaml | 25 |
| Other (305 lang.) | 60000 |

| Category | Percentage (%) |
| :--- | :--- |
| New Features | -25.57 |
| User Interface | -0.88 |
| Testing & QA | -13.32 |
| Testing & QA Testing | -13.32 |
| Bug Fixes | -19.02 |
| Bug Fixes Performance Improvements | -19.02 |
| Bug Fixes | -19.02 |
| Bug Fixes Improvement | -19.02 |
| Bug Fixes Implementation | -19.02 |
| Bug Fixes Implementation Efficiency | -19.02 |
| Bug Fixes Implementation Efficiency Reduction | -19.02 |
| Bug Fixes Implementation Efficiency Reduction Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Efficiency Reduction Performance Improvements | (25.57) New Features
(25.57) New Features
(25.57) New Features
(25.57) New Features
(25.57) New Features
(25.57) New Features
(25.57) New Features
(25.57) New Features
(25.57) New Features
(25.57) New Features
(25.57) New Features
Deprecation (0.28%) 
Build System/Tooling (1.30%) 
Documentation (3.93%) 
Dependencies (5.38%) 
Configuration (4.61%) 
Release Management (4.14%) 
Formatting/Linting (0.40%) 
Refactoring/Code Cleanup (19.78%) 
(Refractoring/Code Cleanup) (19.78%)
</details>

Figure 2: Overview of COMMITPACK and COMMITPACKFT. Top: Language distribution of the full commit data (COMMITPACK) and the variant filtered for high-quality instructions (COMMITPACKFT). See Appendix C for the full distribution. Bottom: Task distribution of commits on the Python subset of COMMITPACKFT (59K samples) according to GPT-4. 

<table><tr><td></td><td colspan="3">Base dataset</td><td colspan="3">Subset</td></tr><tr><td>Dataset (↓)</td><td>Lang.</td><td>Samples</td><td>Code fraction</td><td>Lang.</td><td>Samples</td><td>Code fraction</td></tr><tr><td>xP3x</td><td>8</td><td>532,107,156</td><td>0.67%</td><td>8</td><td>5,000</td><td>100%</td></tr><tr><td>StarCoder Self-Instruct</td><td>12</td><td>5,003</td><td>100%</td><td>12</td><td>5,003</td><td>100%</td></tr><tr><td>OASST</td><td>49</td><td>161,443</td><td>0.9%</td><td>28</td><td>8,587</td><td>2.5%</td></tr><tr><td>COMMITPACKFT</td><td>277</td><td>742,273</td><td>100%</td><td>6</td><td>5,000</td><td>100%</td></tr></table>

Table 1: Statistics of code instruction data we consider. We display the number of programming languages, total samples, and fraction of samples that contain code for permissive instruction datasets. For finetuning on these datasets, we use small subsets with around 5,000 samples each.

languages during instruction tuning boosts their performance further. We hypothesize that code data exhibits the same behavior. To improve performance on code-related tasks, we thus construct a code instruction dataset leveraging the natural structure of Git commits.

COMMITPACK To create the dataset, we use commit metadata from the GitHub action dump on Google BigQuery. $^{3}$ We apply quality filters, filter for commercially friendly licenses, and discard commits that affect more than a single file to ensure commit messages are very specific and to avoid additional complexity from dealing with multiple files. We use the filtered metadata to scrape the affected code files prior to and after the commit from GitHub. This leads to almost 4 terabytes of data covering 350 programming languages (COMMITPACK). As instruction tuning does not require so much data (Zhou et al., 2023a; Touvron et al., 2023), we apply several strict filters to

reduce the dataset to 2 gigabytes and 277 languages (COMMITPACKFT). These include filtering for samples where the commit message has specific words in uppercase imperative form at the start (e.g. "Verify ..."), consists of multiple words, and does not contain external references. All filters are detailed in Appendix D. Figure 2 depicts the distribution of both datasets and the tasks contained in COMMITPACKFT. For instruction tuning our models, we select 5,000 random samples from COMMITPACKFT across the 6 programming languages that we evaluate on. In Appendix G, we also experiment with pretraining on the entirety of COMMITPACK.

Alternatives We consider three additional datasets for instruction tuning presented in Table 1. xP3x: xP3x is a large-scale collection of multilingual instruction data with around 532 million samples (Muennighoff et al., 2022b). We focus only on the code subset of xP3x, excluding Neural-CodeSearch (Li et al., 2019) which is not licensed permissively, and select 5,000 samples.
Self-Instruct: Using the Self-Instruct method (Wang et al., 2022a) and the StarCoder model (Li et al., 2023b), we create 5,003 synthetic instructions and corresponding answers.
OASST: OASST is a diverse dataset of multi-turn chat dialogues (Köpf et al., 2023). Only a few of the dialogues contain code. We reuse a filtered variant from prior work (Dettmers et al., 2023) and additionally filter out moralizing assistant answers (Appendix D) leading to 8,587 samples.

# 3 HUMANEvalPack: EVALUATING INSTRUCTION TUNED CODE MODELS

# HumanEvalPack

Languages: Python, JavaScript, Java, Go, C++, Rust

Metric: Pass@k

Subtasks: HumanEvalFix, HumanEvalExplain, HumanEvalSynthesize

Creation: Humans

![](images/956541be057e69382c9f7c3246f4d98fd97b75379abb29d6160f3be577283393.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Fix Code"] --> B["Explain Code"]
    B --> C["Synthesize Code"]

    subgraph Fix Code
        D["from typing import List"]
        E["def has_close_elements(numbers: List[float"], threshold: float) --> bool: for idx, elem in enumerate(numbers): for idx2, elem2 in enumerate(numbers): if idx != idx2: distance = elem - elem2 if distance < threshold: return True]
        F["return False"]
        G["def check(has_close_elements): assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2"], 0.3) == True]
        H["assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2"], 0.05) == False]
        I["assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0"], 0.95) == True]
        J["assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0"], 0.8) == False]
        K["assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0"], 0.1) == True]
        L["assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1"], 1.0) == True]
        M["assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1"], 0.5) == False]
    end

    subgraph Explain Code
        N["from typing import List"]
        O["def has_close_elements(numbers: List[float"], threshold: float) --> bool: for idx, elem in enumerate(numbers): for idx2, elem2 in enumerate(numbers): if idx != idx2: distance = abs(elem - elem2) if distance < threshold: return True]
        P["return False"]
        Q["Provide a concise natural language description of the function using at most 213 characters."] --> R["Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0"], 0.5) False]
        S[">>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0"], 0.3) True]
    end

    subgraph Synthesize Code
        T["Write a Python function `has_close_elements(numbers: List[float"], threshold: float) --> bool: '’' Check if in given list of numbers, are any two numbers closer to each other than given threshold.
>>> has_close_elements([1.0, 2.0, 3.0], 0.5) False
>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True]
        U["from typing import List"]
        V["def has_close_elements(numbers: List[float"], threshold: float) --> bool: '’' Check if in given list of numbers, are any two numbers closer to each other than given threshold.
>>> has_close_elements([1.0, 2.0, 3.0], 0.5) False
>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0).3) True]
        W["for idx, elem in enumerate(numbers): for idx2, elem2 in enumerate(numbers): if idx != idx2: distance = abs(elem - elem2) if distance < threshold: return True"]
        X["return False"]
    end
    style A fill:#f9f9f9,stroke:#333
    style B fill:#f9f9f9,stroke:#333
    style C fill:#f9f9f9,stroke:#333
    style D fill:#f9f9f9,stroke:#333
    style E fill:#f9f9f9,stroke:#333
    style F fill:#f9f9f9,stroke:#333
    style G fill:#f9f9f9,stroke:#333
    style H fill:#f9f9f9,stroke:#333
    style I fill:#f9f9f9,stroke:#333
    style J fill:#f9f9f9,stroke:#333
    style K fill:#f9f9f9,stroke:#333
    style L fill:#f9f9f9,stroke:#333
    style M fill:#f9f9f9,stroke:#333
    style N fill:#f9f9f9,stroke:#333
    style O fill:#f9f9f9,stroke:#333
    style P fill:#f9f9f9,stroke:#333
    style Q fill:#f9f9f9,stroke:#333
    style R fill:#f9f9f9,stroke:#333
    style S fill:#f9f9f9,stroke:#333
    style T fill:#f9f9f9,stroke:#333
    style U fill:#f9f9f9,stroke:#333
    style V fill:#f9f9f9,stroke:#333
    style W fill:#f9f9f9,stroke:#333
    style X fill:#f9f9f9,stroke:#333
    style Y fill:#f9f9f9,stroke:#333
    style Z fill:#f9f9f9,stroke:#333
```
</details>

Figure 3: HUMANEVALPACK overview. The first HumanEval problem is depicted across the three scenarios for Python. The bug for HUMANEVALFIX consists of a missing "abs" statement.

When instruction tuning LLMs using natural language (NL) data, the input is an NL instruction with optional NL context and the target output is the NL answer to the task (Wei et al., 2022). When instruction tuning with code (C) data, code may either appear only in the input alongside the NL instruction (NL+C→NL, e.g. code explanation), only in the output (NL→C, e.g. code synthesis), or in both input and output (NL+C→C, e.g. code modifications like bug fixing). While prior benchmarks commonly only cover variants of code synthesis, users may want to use models in all three scenarios. Thus, we expand the code synthesis benchmark HumanEval (Chen et al., 2021; Zheng et al., 2023) to cover all three input-output combinations for six languages (Figure 3).

HUMANEVALFIX (NL+C→C) Given an incorrect code function with a subtle bug and accompanying unit tests, the model is tasked to fix the function. We manually add a bug to each of the 164 HumanEval solutions across all 6 languages (984 total bugs). For a given sample, the bugs are as similar as possible across the 6 languages enabling meaningful comparison of scores across languages. Bugs are written such that the code still runs but produces an incorrect result leading to at least one unit test failing. Bug statistics and examples are in Appendix L. We also evaluate an easier variant of this task where instead of unit tests, models are provided with the correct function docstring as the source of truth to fix bugs, see Appendix K.

HUMANEVALEXPLAIN (NL+C→NL) Given a correct code function, the model is tasked to generate an explanation of the code. Subsequently, the same model is tasked to regenerate the code given only its own explanation. The second step allows us to score this task via code execution and measure pass@k (Chen et al., 2021) instead of evaluating the explanation itself using heuristic-based metrics like BLEU (Papineni et al., 2002) or ROUGE (Lin, 2004) which have major limitations (Reiter, 2018; Schluter, 2017; Eghbali & Pradel, 2022; Zhou et al., 2023b). To prevent models from copying the solution into the description, we remove any solution overlap of at least 20 characters from the description. We further enforce a character length limit on the model-generated explanation equivalent to the length of the docstring describing the function. This limit is specified in the prompt for the model. Note that the function docstring itself is never provided to the model for this task.

HUMANEVALSYNTHESIZE (NL→C) Given a natural language docstring or comment describing the desired code, the model is tasked to synthesize the correct code. This task corresponds to the original HumanEval benchmark (Chen et al., 2021). For instruction tuned models, we add an explicit instruction to the input explaining what the model should do. For models that have only gone through language model pretraining, we follow Chen et al. (2021) and provide the model with the function header and docstring to evaluate its completion of the function.

For all tasks we execute the code generations to compute performance using the pass@k metric (Chen et al., 2021): a problem is considered solved if any of k code generations passes every test case. We focus on the simplest version of pass@k, which is pass@1: the likelihood that the model solves a problem in a single attempt. Like Chen et al. (2021), we use a sampling temperature of 0.2 and $top_{p} = 0.95$ to estimate pass@1. We generate n = 20 samples, which is enough to get reliable pass@1 estimates (Li et al., 2023b). For GPT-4, we generate n = 1 samples. Using n = 1 instead of n = 20 for GPT-4 only changed scores from 75.0% to 75.2% pass@1 on HUMANEVALSYNTHESIZE Python while providing 20x cost savings.

Python HumanEval is the most widely used code benchmark and many training datasets have already been decontaminated for it (Kocetkov et al., 2022). By manually extending HumanEval, we ensure existing decontamination remains valid to enable fair evaluation. However, this may not hold for all models (e.g. GPT-4), thus results should be interpreted carefully.

# 4 OctoCoder: BEST COMMERCIALLY LICENSED CODE LLM

# 4.1 ABLATING INSTRUCTION DATA CHOICES

We instruction tune the pretrained StarCoder model (Li et al., 2023b) on different combinations of our instruction datasets ( $\S2$ ). We evaluate all models on the Python subset of HUMANEVALPACK as depicted in Figure 4. Similar to prior work (Taori et al., 2023), we format all instructions into a consistent schema to distinguish question and answer (see Figure 18).

COMMITPACKFT enables CodeLLMs to fix bugs COMMITPACKFT is critical for the performance boost on code repair (HUMANEVALFIX), where instruction tuning on only OASST or other variants results in a significantly lower score. This is likely due to COMMITPACKFT including around 20% of bug fixes among other code-related tasks (Figure 2).

Importance of samples with natural language targets The pretrained StarCoder model, as well as the Self-Instruct variant, perform poorly on code explanation (HUMANEVALEXPLAIN). This is because both models are only conditioned to write code instead of natural language. We find that to

![](images/da3f3d2397b2b5c7fd663e6f146846bcf7e4ad2c0e0a474fa1d0e89194545116.jpg)

<details>
<summary>bar</summary>

| Category | No instruction tuning (%) | Self-Instruct (%) | OASST (%) | Self-Instruct + OASST (%) | xP3x-Code + OASST (%) | CommitPackFT + OASST (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Code Fixing | 9 | 24 | 24 | 25 | 28 | 30 |
| Code Explanation | - | - | 35 | 29 | 28 | 35 |
| Code Synthesis | 34 | 44 | 47 | 46 | 45 | 46 |
| Average | 14 | 22 | 35 | 33 | 34 | 38 |
</details>

Figure 4: Comparing permissively licensed instruction datasets by instruction tuning StarCoder. Models are evaluated on the Python subset of HUMANEVALPACK.

perform well at explaining code, it is necessary to include samples with natural language as the target output during instruction tuning. Only relying on data with code as the target, such as the Self-Instruct data, will lead to models always outputting code even if the question requires a natural language output. Thus, we mix all other ablations with OASST, which contains many natural language targets. While the xP3x subset also contains samples with natural language output, many of its target outputs are short, which leads to models with a bias for short answers. This is impractical for the explanation task leading to the comparatively low score of mixing xP3x with OASST.

COMMITPACKFT+OASST yields best performance All instruction datasets provide similar boosts for code synthesis (HUMANEVALSYNTHESIZE), which has been the focus of all prior work on code instruction models (Wang et al., 2023c; Luo et al., 2023; Muennighoff et al., 2022b). We achieve the best average score by instruction tuning on COMMITPACKFT mixed with our filtered OASST data yielding an absolute $23\%$ improvement over StarCoder. Thus, we select COMMITPACKFT+OASST for our final model dubbed OCTOCODER. Using the same data, we also instruction tune the 6 billion parameter CodeGeeX2 (Zheng et al., 2023) to create OCTOGEEX. Training hyperparameters for both models are in Appendix P.

# 4.2 COMPARING WITH OTHER MODELS

We benchmark OCTOCODER and OCTOGEEX with state-of-the-art Code LLMs on HUMANEVALPACK in Table 2. For all models, we use the prompt put forward by the model creators if applicable or else a simple intuitive prompt, see Appendix Q.

OCTOCODER performs best among permissive models OCTOCODER has the highest average score across all three evaluation scenarios among all permissive models. With just 6 billion parameters, OCTOGEEX is the smallest model benchmarked, but still outperforms all prior permissive Code LLMs. GPT-4 (OpenAI, 2023) performs best among all models benchmarked with a significant margin. However, GPT-4 is closed-source and likely much larger than all other models evaluated.

Instruction tuning generalizes to unseen programming languages Trained primarily on natural language, not code, BLOOMZ (Muennighoff et al., 2022b) performs worse than other models despite having 176 billion parameters. Go and Rust are not contained in BLOOMZ's instruction data, yet it performs much better than the random baseline of 0.0 for these two languages across most tasks. This confirms our hypothesis that models are capable of generalizing instructions to programming languages only seen at pretraining, similar to crosslingual generalization for natural languages (Muennighoff et al., 2022b). To improve programming language generalization further, we tune OCTOCODER and OCTOGEEX on many languages from COMMITPACKFT, and this generalization improvement is reflected in the performance on HUMANEVALPACK's new languages.

Pretraining weight correlates with programming language performance after instruction tuning. Prior work has shown that the performance on natural languages after instruction tuning is correlated with the weight of these languages during pretraining (Muennighoff et al., 2022b). The more weight during pretraining, the better the performance after instruction tuning. We find the same to be

<table><tr><td>Model (↓)</td><td>Python</td><td>JavaScript</td><td>Java</td><td>Go</td><td>C++</td><td>Rust</td><td>Avg.</td></tr></table>

HUMANEVALFIX 

<table><tr><td colspan="8">Non-permissive models</td></tr><tr><td>InstructCodeT5+†</td><td>2.7</td><td>1.2</td><td>4.3</td><td>2.1</td><td>0.2</td><td>0.5</td><td>1.8</td></tr><tr><td>WizardCoder†</td><td>31.8</td><td>29.5</td><td>30.7</td><td>30.4</td><td>18.7</td><td>13.0</td><td>25.7</td></tr><tr><td>GPT-4</td><td>47.0</td><td>48.2</td><td>50.0</td><td>50.6</td><td>47.6</td><td>43.3</td><td>47.8</td></tr><tr><td colspan="8">Permissive models</td></tr><tr><td>BLOOMZ</td><td>16.6</td><td>15.5</td><td>15.2</td><td>16.4</td><td>6.7</td><td>5.7</td><td>12.5</td></tr><tr><td>StarChat-β</td><td>18.1</td><td>18.1</td><td>24.1</td><td>18.1</td><td>8.2</td><td>3.6</td><td>11.2</td></tr><tr><td>CodeGeeX2*</td><td>15.9</td><td>14.7</td><td>18.0</td><td>13.6</td><td>4.3</td><td>6.1</td><td>12.1</td></tr><tr><td>StarCoder</td><td>8.7</td><td>15.7</td><td>13.3</td><td>20.1</td><td>15.6</td><td>6.7</td><td>13.4</td></tr><tr><td>OCTOGEEX*</td><td>28.1</td><td>27.7</td><td>30.4</td><td>27.6</td><td>22.9</td><td>9.6</td><td>24.4</td></tr><tr><td>OCTOCODER</td><td>30.4</td><td>28.4</td><td>30.6</td><td>30.2</td><td>26.1</td><td>16.5</td><td>27.0</td></tr></table>

HUMAN EVALE EXPLAIN 

<table><tr><td colspan="8">Non-permissive models</td></tr><tr><td>InstructCodeT5+†</td><td>20.8</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.1</td><td>0.0</td><td>3.5</td></tr><tr><td>WizardCoder†</td><td>32.5</td><td>33.0</td><td>27.4</td><td>26.7</td><td>28.2</td><td>16.9</td><td>27.5</td></tr><tr><td>GPT-4</td><td>64.6</td><td>57.3</td><td>51.2</td><td>58.5</td><td>38.4</td><td>42.7</td><td>52.1</td></tr><tr><td colspan="8">Permissive models</td></tr><tr><td>BLOOMZ</td><td>14.7</td><td>8.8</td><td>12.1</td><td>8.5</td><td>0.6</td><td>0.0</td><td>7.5</td></tr><tr><td>StarChat-β</td><td>25.4</td><td>21.5</td><td>24.5</td><td>18.4</td><td>17.6</td><td>13.2</td><td>20.1</td></tr><tr><td>CodeGeeX2*</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>StarCoder</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>OCTOGEEX*</td><td>30.4</td><td>24.0</td><td>24.7</td><td>21.7</td><td>21.0</td><td>15.9</td><td>22.9</td></tr><tr><td>OCTOCODER</td><td>35.1</td><td>24.5</td><td>27.3</td><td>21.1</td><td>24.1</td><td>14.8</td><td>24.5</td></tr></table>

HUMANEVALSYNTHESIZE 

<table><tr><td colspan="8">Non-permissive models</td></tr><tr><td>InstructCodeT5+†</td><td>37.0</td><td>18.9</td><td>17.4</td><td>9.5</td><td>19.8</td><td>0.3</td><td>17.1</td></tr><tr><td>WizardCoder†</td><td>57.3</td><td>49.5</td><td>36.1</td><td>36.4</td><td>40.9</td><td>20.2</td><td>40.1</td></tr><tr><td>GPT-4</td><td>86.6</td><td>82.9</td><td>81.7</td><td>72.6</td><td>78.7</td><td>67.1</td><td>78.3</td></tr><tr><td colspan="8">Permissive models</td></tr><tr><td>BLOOMZ</td><td>15.6</td><td>14.8</td><td>18.4</td><td>8.4</td><td>6.5</td><td>5.5</td><td>11.5</td></tr><tr><td>StarChat-β</td><td>33.5</td><td>31.4</td><td>26.7</td><td>25.5</td><td>26.6</td><td>14.0</td><td>26.3</td></tr><tr><td>CodeGeeX2*</td><td>35.9</td><td>32.2</td><td>30.8</td><td>22.5</td><td>29.3</td><td>18.1</td><td>28.1</td></tr><tr><td>StarCoder</td><td>33.6</td><td>30.8</td><td>30.2</td><td>17.6</td><td>31.6</td><td>21.8</td><td>27.6</td></tr><tr><td>OCTOGEEX*</td><td>44.7</td><td>33.8</td><td>36.9</td><td>21.9</td><td>32.3</td><td>15.7</td><td>30.9</td></tr><tr><td>OCTOCODER</td><td>46.2</td><td>39.2</td><td>38.2</td><td>30.4</td><td>35.6</td><td>23.4</td><td>35.5</td></tr></table>

Table 2: Zero-shot pass@1 (%) performance across HUMANEVALPACK. InstructCodeT5+, WizardCoder, StarChat- $\beta$ , StarCoder and OCTOCODER have 16B parameters. CodeGeeX2 and OCTOGEEX have 6B parameters. BLOOMZ has 176B parameters. In this work, we call models "permissive" if weights are freely accessible and usable for commercial purposes. \*: Commercial license available after submitting a form. †: Trained on data that may not be used "to develop models that compete with OpenAI" thus we classify them as non-permissive in this work (see §1).

the case for programming languages. Python, Java, and JavaScript collectively make up around 30% of the pretraining data of StarCoder (Li et al., 2023b). After instruction tuning StarCoder to produce OCTOCODER, we see the best performance among these three languages, especially for HUMANEVALSYNTHESIZE. OCTOCODER performs weakest on Rust, which is the lowest resource language of StarCoder among the languages we benchmark (1.2% of pretraining data).

Models struggle with small targeted changes HUMANEVALFIX is the most challenging task for most models. They commonly regenerate the buggy function without making any change (e.g. WizardCoder in Figure 34) or they introduce new bugs (e.g. GPT-4 in Figure 33). We analyze model performance by bug type in Appendix M and find bugs that require removing excess code are the most challenging. OCTOCODER performs comparatively well across all languages. Instruction tuning on COMMITPACKFT has likely taught OCTOCODER to make small, targeted changes to fix bugs.

Models struggle switching between code and text Some models fail at HUMANEVALEXPLAIN, as they do not generate natural language explanations. We manually inspect explanations for the first ten samples of the Python split and disqualify a model if none of them are explanations. This is the case for StarCoder and CodeGeeX2, which generate code instead of natural language explanations. BLOOMZ and InstructCodeT5+ also occasionally generate code. Other models exclusively generate natural language explanations, not containing any code for inspected samples.

Models struggle adhering to a specified output length HUMANEVALEXPLAIN instructs models to fit their explanation within a given character limit ( $\S3$ ). Current models appear to have no understanding of how many characters they are generating. They commonly write very short and thus underspecified explanations (e.g. BLOOMZ in Figure 35) or excessively long explanations that end up being cut off (e.g. StarChat- $\beta$ in Figure 38). Future work could investigate how to enable models to be aware of their generated output length to improve HUMANEVALEXPLAIN performance.

HumanEval code synthesis is close to saturation Pure code synthesis on HUMANEVALSYN-THESIZE is the easiest task for all models. With a pass rate of 86.6% for a single solution, GPT-4 is close to fully saturating the Python subset. GPT-4 was originally found to score 67% on Python HumanEval (OpenAI, 2023) and 81% in later work (Bubeck et al., 2023). Our score for GPT-4 is significantly higher, possibly due to improvements made to the API by OpenAI, contamination of HumanEval in GPT-4 training, or slightly different prompting and evaluation. An example of our prompt is depicted in Figure 3 (right). We perform very careful evaluation to ensure every generation is correctly processed. We reproduce the HumanEval score of WizardCoder (Luo et al., 2023; Xu et al., 2023a) and find it to also perform well across other languages. For BLOOMZ and InstructCodeT5+ our evaluation leads to a higher Python score than they reported, likely because of our more careful processing of generations. OCTOCODER has the highest performance for every language among permissively licensed models. With a pass@1 of 46.2% on the original Python split, OCTOCODER improves by a relative 38% over its base model, StarCoder.

# 5 RELATED WORK

# 5.1 CODE MODELS

There has been extensive work on code models tailored to a specific coding task, such as code summarization (Iyer et al., 2016; Ahmad et al., 2020; Zhang et al., 2022a; Shi et al., 2022) or code editing (Drain et al., 2021; Zhang et al., 2022c; He et al., 2022; Zhang et al., 2022b; Wei et al., 2023; Prenner & Robbes, 2023; Fakhoury et al., 2023; Skreta et al., 2023) (also see work on edit models more generally (Reid & Neubig, 2022; Schick et al., 2022; Dwivedi-Yu et al., 2022; Raheja et al., 2023)). These works use task-specific heuristics that limit the applicability of their methods to other tasks. In contrast, we aim to build models applicable to all kinds of tasks related to code and beyond.

Through large-scale pretraining more generally applicable code models have been developed (Nijkamp et al., 2022; 2023; Xu et al., 2022a; Christopoulou et al., 2022; Gunasekar et al., 2023; Li et al., 2023b; Bui et al., 2023; Scao et al., 2022a;b). However, these models only continue code making them hard to use for tasks such as explaining code with natural language (HUMANEVALEXPLAIN). Teaching them to follow human instructions is critical to make them applicable to diverse tasks.

# 5.2 INSTRUCTION MODELS

Training models to follow instructions has led to new capabilities in text (Ouyang et al., 2022; Wang et al., 2022b; Chung et al., 2022) and visual modalities (Xu et al., 2023b; OpenAI, 2023). Prior work has shown its benefits for traditional language tasks (Wei et al., 2022; Longpre et al., 2023a; Iyer et al., 2022), multilingual tasks (Muennighoff et al., 2022b; 2024; Yong et al., 2022; Üstün et al., 2024), and dialog (Köpf et al., 2023; Bai et al., 2022; Ganguli et al., 2022). For coding applications, PanGu-Coder2 (Shen et al., 2023), WizardCoder (Luo et al., 2023) and InstructCodeT5+ (Wang et al., 2023c) are recent models trained with coding instructions. However, they all use the CodeAlpaca dataset (Chaudhary, 2023), which is synthetically generated from OpenAI models. Using data from powerful closed-source models provides a strong advantage, but limits the model use and has other limitations highlighted in §1. CoEditor (Wei et al., 2023) proposes an “auto-editing” task, trained on 1650 python commit history repositories. Our work expands this to more general coding tasks via instructions, more languages, and orders of magnitude more commit data.

# 5.3 CODE BENCHMARKS

Many code synthesis benchmarks have been proposed (Wang et al., 2022d;c; Yu et al., 2023; Lai et al., 2023; Du et al., 2023). HumanEval (Chen et al., 2021; Liu et al., 2023b) has emerged as the standard for this task. Prior work has extended HumanEval to new programming languages via automatic translation mechanisms (Athiwaratkun et al., 2022; Cassano et al., 2023; Orlanski et al., 2023). These approaches are error-prone and only translate tests, not the actual solutions, which are needed for tasks like code explanation. Thus, we rely only on humans to create all parts of HUMANEVALPACK including test cases, correct solutions, buggy solutions, and other metadata across 6 languages.

Code repair is commonly evaluated on Quixbugs (Lin et al., 2017; Prenner & Robbes, 2021; Ye et al., 2021; Xia & Zhang, 2023; Jiang et al., 2023; Sobania et al., 2023) or Python bugs (He et al., 2022; Bradley et al., 2023). The latter does not support code execution, which limits its utility. While Quixbugs supports execution with unit tests, it only contains 40 samples in Python and Java. Further, the problems in Quixbugs are generic functions, such as bucket sort. This makes them easy to solve and hard to decontaminate training data for. Our benchmark, HUMANEVALFIX, contains 164 buggy functions for six languages with solutions and unit tests. Further, our coding problems, derived from HumanEval, are very specific, such as keeping track of a bank account balance (see Figure 14).

Prior work on evaluating code explanations (Lu et al., 2021; Cui et al., 2022) has relied on metrics such as METEOR (Banerjee & Lavie, 2005) or BLEU (Papineni et al., 2002). By chaining code explanation with code synthesis, we can evaluate this task using the execution-based pass@k metric overcoming the major limitations of BLEU and other heuristics-based metrics (Reiter, 2018).

Large-scale benchmarking has proven useful in many areas of natural language processing (Wang et al., 2019; Kiela et al., 2021; Srivastava et al., 2022; Muennighoff et al., 2022a). By producing 18 scores (6 languages across 3 tasks) for 9 models, we take a step towards large-scale benchmarking of code models. However, we lack many models capable of generating code (Black et al., 2021; Fried et al., 2022; Black et al., 2022; Wang & Komatsuzaki, 2021; Biderman et al., 2023b). Future work may consider more models or extending HUMANEVALPACK to new languages or tasks, such as code efficiency (Madaan et al., 2023a; Yetistiren et al., 2022) or code classification (Khan et al., 2023).

# 6 CONCLUSION

This work studies training and evaluation of Code LLMs that follow instructions. We introduce COMMITPACK, a 4TB dataset of Git commits covering 350 programming languages. We filter this large-scale dataset to create COMMITPACKFT, 2GB of high-quality code with commit messages that assimilate instructions. To enable a comprehensive evaluation of instruction code models, we construct HUMANEVALPACK, a human-written benchmark covering 3 different tasks for 6 programming languages. We ablate several instruction datasets and find that COMMITPACKFT combined with natural language data leads to the best performance. While our models, OCTOCODER and OCTOGEEX, are the best permissively licensed Code LLMs available, they are outperformed by closed-source models such as GPT-4. In addition to improving the instruction tuning paradigm, future work should consider training more capable base models.

# ACKNOWLEDGEMENTS

We thank Hugging Face for providing compute instances. We are extremely grateful to Rodrigo Garcia for the Rust translations, Dimitry Ageev and Calum Bird for help with GPT-4 evaluation, Loubna Ben Allal for help on evaluation, Arjun Guha for insightful discussions on chaining evaluation tasks to avoid evaluating with BLEU, Lewis Tunstall for help on the OASST data, Victor Sanh and Nadav Timor for discussions, Jiaxi Yang for logo editing and domain classification prompting design, Ghosal et al. (2023); Zeng et al. (2023) for design inspiration, Harm de Vries for feedback and all members of BigCode for general support. Finally, we thank every programmer who takes the time to write informative commit messages.

# REFERENCES

Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. A transformer-based approach for source code summarization. arXiv preprint arXiv:2005.00653, 2020.   
Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. Santacoder: don't reach for the stars! arXiv preprint arXiv:2301.03988, 2023.   
Ben Athiwaratkun, Sanjay Krishna Gouda, Zijian Wang, Xiaopeng Li, Yuchen Tian, Ming Tan, Wasi Uddin Ahmad, Shiqi Wang, Qing Sun, Mingyue Shang, et al. Multi-lingual evaluation of code generation models. arXiv preprint arXiv:2210.14868, 2022.   
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.   
Hannah McLean Babe, Sydney Nguyen, Yangtian Zi, Arjun Guha, Molly Q Feldman, and Carolyn Jane Anderson. Studenteval: A benchmark of student-written prompts for large language models of code. arXiv preprint arXiv:2306.04556, 2023.   
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022. URL https://arxiv.org/abs/2204.05862.   
Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pp. 65–72, 2005.   
Antonio Valerio Miceli Barone and Rico Sennrich. A parallel corpus of python functions and documentation strings for automated code documentation and code generation. arXiv preprint arXiv:1707.02275, 2017.   
Mohammad Bavarian, Heewoo Jun, Nikolas A. Tezak, John Schulman, Christine McLeavey, Jerry Tworek, and Mark Chen. Efficient training of language models to fill in the middle. arXiv preprint arXiv:2207.14255, 2022.   
Loubna Ben Allal, Niklas Muennighoff, Logesh Kumar Umapathi, Ben Lipkin, and Leandro von Werra. A framework for the evaluation of code generation models. https://github.com/bigcode-project/bigcode-evaluation-harness, 2022.   
Stella Biderman, USVSN Sai Prashanth, Lintang Sutawika, Hailey Schoelkopf, Quentin Anthony, Shivanshu Purohit, and Edward Raf. Emergent and predictable memorization in large language models. arXiv preprint arXiv:2304.11158, 2023a.   
Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023b.

Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. Gpt-neo: Large scale autoregressive language modeling with mesh-tensorflow. If you use this software, please cite it using these metadata, 58, 2021.   
Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. Gpt-neox-20b: An open-source autoregressive language model. arXiv preprint arXiv:2204.06745, 2022.   
Herbie Bradley, Honglu Fan, Harry Saini, Reshinth Adithyan, Shivanshu Purohit, and Joel Lehman. Diff models - a new way to edit code. CarperAI Blog, Jan 2023. URL https://carper.ai/diff-model/.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Conference on Neural Information Processing Systems (NeurIPS), 2020. URL https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.   
Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.   
Nghi DQ Bui, Hung Le, Yue Wang, Junnan Li, Akhilesh Deepak Gotmare, and Steven CH Hoi. Codetf: One-stop transformer library for state-of-the-art code llm. arXiv preprint arXiv:2306.00029, 2023.   
Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, et al. Multipl-e: a scalable and polyglot approach to benchmarking neural code generation. IEEE Transactions on Software Engineering, 2023.   
Sahil Chaudhary. Code alpaca: An instruction-following llama model for code generation. https://github.com/sahil280114/codealpaca, 2023.   
Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. Codet: Code generation with generated tests. arXiv preprint arXiv:2207.10397, 2022.   
Lingjiao Chen, Matei Zaharia, and James Zou. How is chatgpt's behavior changing over time?, 2023a.   
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.   
Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023b.   
Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. arXiv preprint arXiv:2304.05128, 2023c.   
Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.   
Fenia Christopoulou, Gerasimos Lampouras, Milan Gritta, Guchun Zhang, Yinpeng Guo, Zhongqi Li, Qi Zhang, Meng Xiao, Bo Shen, Lin Li, et al. Pangu-coder: Program synthesis with function-level language modeling. arXiv preprint arXiv:2207.11280, 2022.   
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022. URL https://arxiv.org/abs/2210.11416.   
Haotian Cui, Chenglong Wang, Junjie Huang, Jeevana Priya Inala, Todd Mytkowicz, Bo Wang, Jianfeng Gao, and Nan Duan. Codeexp: Explanatory code document generation. arXiv preprint arXiv:2211.15395, 2022.

Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.   
Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.   
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023.   
Kaustubh D Dhole, Varun Gangal, Sebastian Gehrmann, Aadesh Gupta, Zhenhao Li, Saad Mahamood, Abinaya Mahendiran, Simon Mille, Ashish Srivastava, Samson Tan, et al. Nl-augmenter: A framework for task-sensitive natural language augmentation. arXiv preprint arXiv:2112.02721, 2021.   
Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, and Bing Xiang. Cocomic: Code completion by jointly modeling in-file and cross-file context. arXiv preprint arXiv:2212.10007, 2022.   
Yihong Dong, Xue Jiang, Zhi Jin, and Ge Li. Self-collaboration code generation via chatgpt. arXiv preprint arXiv:2304.07590, 2023.   
Dawn Drain, Colin B Clement, Guillermo Serrato, and Neel Sundaresan. Deepdebug: Fixing python bugs using stack traces, backtranslation, and code skeletons. arXiv preprint arXiv:2105.09352, 2021.   
Xueying Du, Mingwei Liu, Kaixin Wang, Hanlin Wang, Junwei Liu, Yixuan Chen, Jiayi Feng, Chaofeng Sha, Xin Peng, and Yiling Lou. Classeval: A manually-crafted benchmark for evaluating llms on class-level code generation. arXiv preprint arXiv:2308.01861, 2023.   
Jane Dwivedi-Yu, Timo Schick, Zhengbao Jiang, Maria Lomeli, Patrick Lewis, Gautier Izacard, Edouard Grave, Sebastian Riedel, and Fabio Petroni. Editeval: An instruction-based benchmark for text improvements. arXiv preprint arXiv:2209.13331, 2022.   
Aryaz Eghbali and Michael Pradel. Crystalbleu: precisely and efficiently measuring the similarity of code. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, pp. 1–12, 2022.   
Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization, 2024.   
Sarah Fakhoury, Saikat Chakraborty, Madan Musuvathi, and Shuvendu K Lahiri. Towards generating functionally correct code edits from natural language issue descriptions. arXiv preprint arXiv:2304.03816, 2023.   
Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. Incoder: A generative model for code infilling and synthesis. arXiv preprint arXiv:2204.05999, 2022.   
Jinlan Fu, See-Kiong Ng, Zhengbao Jiang, and Pengfei Liu. Gptscore: Evaluate as you desire. arXiv preprint arXiv:2302.04166, 2023.   
Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858, 2022.   
Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 2021. URL https://doi.org/10.5281/zenodo.5371628.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pp. 10764–10799. PMLR, 2023.   
Deepanway Ghosal, Yew Ken Chia, Navonil Majumder, and Soujanya Poria. Flacuna: Unleashing the problem solving power of vicuna using flan fine-tuning. arXiv preprint arXiv:2307.02053, 2023.   
Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738, 2023.   
Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, A. Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Daniel Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hanna Hajishirzi. Olmo: Accelerating the science of language models. 2024. URL https://api.semanticscholar.org/CorpusID:267365485.   
Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. The false promise of imitating proprietary llms. arXiv preprint arXiv:2305.15717, 2023.   
Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023.   
Jingxuan He, Luca Beurer-Kellner, and Martin Vechev. On distribution shift in learning-based bug detectors. In International Conference on Machine Learning, pp. 8559–8580. PMLR, 2022.   
Vincent J Hellendoorn, Charles Sutton, Rishabh Singh, Petros Maniatis, and David Bieber. Global relational models of source code. In International conference on learning representations, 2019.   
Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938, 2021.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.   
Yi Hu, Haotong Yang, Zhouchen Lin, and Muhan Zhang. Code prompting: a neural symbolic method for complex reasoning in large language models. arXiv preprint arXiv:2305.18507, 2023.   
Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, and Luke Zettlemoyer. Summarizing source code using a neural attention model. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2073–2083, 2016.   
Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Daniel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Koura, Xian Li, Brian O'Horo, Gabriel Pereyra, Jeff Wang, Christopher Dewan, Asli Celikyilmaz, Luke Zettlemoyer, and Ves Stoyanov. Opt-iml: Scaling language model instruction meta learning through the lens of generalization. arXiv preprint arXiv:2212.12017, 2022. URL https://arxiv.org/abs/2212.12017.   
Mingi Jeon, Seung-Yeop Baik, Joonghyuk Hahn, Yo-Sub Han, and Sang-Ki Ko. Deep Learning-based Code Complexity Prediction. openreview, 2022.   
Nan Jiang, Kevin Liu, Thibaud Lutellier, and Lin Tan. Impact of code language models on automated program repair. arXiv preprint arXiv:2302.05020, 2023.

Tae-Hwan Jung. Commitbert: Commit message generation using pre-trained programming language model. arXiv preprint arXiv:2105.14242, 2021.   
Mohammad Abdullah Matin Khan, M Saiful Bari, Xuan Long Do, Weishi Wang, Md Rizwan Parvez, and Shafiq Joty. xcodeeval: A large scale multilingual multitask benchmark for code understanding, generation, translation and retrieval. arXiv preprint arXiv:2303.03004, 2023.   
Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Casey A Fitzpatrick, Peter Bull, Greg Lipstein, Tony Nelli, Ron Zhu, et al. The hateful memes challenge: Competition report. In NeurIPS 2020 Competition and Demonstration Track, pp. 344–360. PMLR, 2021.   
Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, et al. The stack: 3 tb of permissively licensed source code. arXiv preprint arXiv:2211.15533, 2022.   
Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, et al. Openassistant conversations–democratizing large language model alignment. arXiv preprint arXiv:2304.07327, 2023.   
Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. Ds-1000: A natural and reliable benchmark for data science code generation. In International Conference on Machine Learning, pp. 18319–18345. PMLR, 2023.   
Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro Von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, et al. The bigscience roots corpus: A 1.6 tb composite multilingual dataset. Advances in Neural Information Processing Systems, 35:31809–31826, 2022.   
Joel Lehman, Jonathan Gordon, Shawn Jain, Kamal Ndousse, Cathy Yeh, and Kenneth O Stanley. Evolution through large models. arXiv preprint arXiv:2206.08896, 2022.   
Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023a.   
Hongyu Li, Seohyun Kim, and Satish Chandra. Neural code search evaluation dataset. arXiv preprint arXiv:1908.09804, 2019.   
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023b.   
Xueyang Li, Shangqing Liu, Ruitao Feng, Guozhu Meng, Xiaofei Xie, Kai Chen, and Yang Liu. Transrepair: Context-aware program repair for compilation errors. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, pp. 1–13, 2022a.   
Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022b.   
Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004.   
Derrick Lin, James Koppel, Angela Chen, and Armando Solar-Lezama. Quixbugs: A multi-lingual program repair benchmark set based on the quixey challenge. In Proceedings Companion of the 2017 ACM SIGPLAN international conference on systems, programming, languages, and applications: software for humanity, pp. 55–56, 2017.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023a.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. arXiv preprint arXiv:2305.01210, 2023b.   
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172, 2023c.   
Tianyang Liu, Canwen Xu, and Julian McAuley. Repobench: Benchmarking repository-level code auto-completion systems. arXiv preprint arXiv:2306.03091, 2023d.   
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. Gpteval: Nlg evaluation using gpt-4 with better human alignment. arXiv preprint arXiv:2303.16634, 2023e.   
Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. The flan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023a.   
Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barrett Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, et al. A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity. arXiv preprint arXiv:2305.13169, 2023b.   
Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al. Codexglue: A machine learning benchmark dataset for code understanding and generation. arXiv preprint arXiv:2102.04664, 2021.   
Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. arXiv preprint arXiv:2306.08568, 2023.   
Aman Madaan, Alexander Shypula, Uri Alon, Milad Hashemi, Parthasarathy Ranganathan, Yiming Yang, Graham Neubig, and Amir Yazdanbakhsh. Learning performance-improving code edits. arXiv preprint arXiv:2302.07867, 2023a.   
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023b.   
Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. MetaICL: Learning to learn in context. Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2022. URL https://arxiv.org/abs/2110.15943.   
Martin Monperrus, Matias Martinez, He Ye, Fernanda Madeiral, Thomas Durieux, and Zhongxing Yu. Megadiff: A dataset of 600k java source code changes categorized by diff size. arXiv preprint arXiv:2108.04631, 2021.   
Niklas Muennighoff. Sgpt: Gpt sentence embeddings for semantic search. arXiv preprint arXiv:2202.08904, 2022.   
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316, 2022a. doi: 10.48550/ARXIV.2210.07316. URL https://arxiv.org/abs/2210.07316.   
Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786, 2022b.   
Niklas Muennighoff, Alexander M Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language models. arXiv preprint arXiv:2305.16264, 2023.   
Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. Generative representational instruction tuning, 2024.

Ansong Ni, Srini Iyer, Dragomir Radev, Veselin Stoyanov, Wen-tau Yih, Sida Wang, and Xi Victoria Lin. Lever: Learning to verify language-to-code generation with execution. In International Conference on Machine Learning, pp. 26106–26128. PMLR, 2023.   
Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474, 2022.   
Erik Nijkamp, Hiroaki Hayashi, Caiming Xiong, Silvio Savarese, and Yingbo Zhou. Codegen2: Lessons for training llms on programming and natural languages. arXiv preprint arXiv:2305.02309, 2023.   
Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114, 2021. URL https://openreview.net/forum?id=iedYJm92o0a.   
OpenAI. Gpt-4 technical report, 2023.   
Gabriel Orlanski, Kefan Xiao, Xavier Garcia, Jeffrey Hui, Joshua Howland, Jonathan Malmaud, Jacob Austin, Rishah Singh, and Michele Catasta. Measuring the impact of programming language distribution. arXiv preprint arXiv:2302.01973, 2023.   
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Conference on Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2203.02155.   
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318, 2002.   
Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.   
Ethan Perez, Douwe Kiela, and Kyunghyun Cho. True few-shot learning with language models. Advances in Neural Information Processing Systems, 34:11054–11070, 2021.   
Luiza Amador Pozzobon, Beyza Ermis, Patrick Lewis, and Sara Hooker. On the challenges of using black-box apis for toxicity evaluation in research. In ICLR 2023 Workshop on Trustworthy and Reliable Large-Scale Machine Learning Models, 2023.   
Julian Aron Prenner and Romain Robbes. Automatic program repair with openai's codex: Evaluating quixbugs. arXiv preprint arXiv:2111.03922, 2021.   
Julian Aron Prenner and Romain Robbes. Runbugrun—an executable dataset for automated program repair. arXiv preprint arXiv:2304.01102, 2023.   
Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021.   
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.   
Vipul Raheja, Dhruv Kumar, Ryan Koo, and Dongyeop Kang. Coedit: Text editing by task-specific instruction tuning. arXiv preprint arXiv:2305.09857, 2023.   
Machel Reid and Graham Neubig. Learning to model editing processes. arXiv preprint arXiv:2205.12374, 2022.   
Ehud Reiter. A structured review of the validity of bleu. Computational Linguistics, 44(3):393–401, 2018.

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. International Conference on Learning Representations (ICLR), 2022. URL https://openreview.net/forum?id=9Vrb9D0WI4.   
Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022a.   
Teven Le Scao, Thomas Wang, Daniel Hesslow, Lucile Saulnier, Stas Bekman, M Saiful Bari, Stella Bideman, Hady Elsahar, Niklas Muennighoff, Jason Phang, et al. What language model to train if you have one million gpu hours? arXiv preprint arXiv:2210.15424, 2022b.   
Timo Schick, Jane Dwivedi-Yu, Zhengbao Jiang, Fabio Petroni, Patrick Lewis, Gautier Izacard, Qingfei You, Christoforos Nalmpantis, Edouard Grave, and Sebastian Riedel. Peer: A collaborative language model. arXiv preprint arXiv:2208.11663, 2022.   
Natalie Schluter. The limits of automatic summarisation according to rouge. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics, pp. 41–45. Association for Computational Linguistics, 2017.   
Noam M. Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019.   
Bo Shen, Jiaxin Zhang, Taihong Chen, Daoguang Zan, Bing Geng, An Fu, Muhan Zeng, Ailun Yu, Jichuan Ji, Jingyang Zhao, Yuenan Guo, and Qianxiang Wang. Pangu-coder2: Boosting large language models for code with ranking feedback, 2023.   
Ensheng Shi, Yanlin Wang, Lun Du, Junjie Chen, Shi Han, Hongyu Zhang, Dongmei Zhang, and Hongbin Sun. On the evaluation of neural code summarization. In Proceedings of the 44th International Conference on Software Engineering, pp. 1597–1608, 2022.   
Disha Shrivastava, Denis Kocetkov, Harm de Vries, Dzmitry Bahdanau, and Torsten Scholak. Repofusion: Training code models to understand your repository. arXiv preprint arXiv:2306.10998, 2023a.   
Disha Shrivastava, Hugo Larochelle, and Daniel Tarlow. Repository-level prompt generation for large language models of code. In International Conference on Machine Learning, pp. 31693–31715. PMLR, 2023b.   
Shivalika Singh, Freddie Vargus, Daniel Dsouza, Börje F Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, et al. Aya dataset: An open-access collection for multilingual instruction tuning. arXiv preprint arXiv:2402.06619, 2024.   
Marta Skreta, Naruki Yoshikawa, Sebastian Arellano-Rubach, Zhi Ji, Lasse Bjørn Kristensen, Kourosh Darvish, Alán Aspuru-Guzik, Florian Shkurti, and Animesh Garg. Errors are useful prompts: Instruction guided task programming with verifier-assisted iterative prompting. arXiv preprint arXiv:2303.14100, 2023.   
Dominik Sobania, Martin Briesch, Carol Hanna, and Justyna Petke. An analysis of the automatic bug fixing performance of chatgpt. arXiv preprint arXiv:2301.08653, 2023.   
Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Raghavi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, A. Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Daniel Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hanna Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an open corpus of three trillion tokens for language model pretraining research. 2024. URL https://api.semanticscholar.org/CorpusID:267364861.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models, 2022. URL https://arxiv.org/abs/2206.04615.   
Simeng Sun, Kalpesh Krishna, Andrew Mattarella-Micke, and Mohit Iyyer. Do long-range language models actually use long-range context? ArXiv, abs/2109.09115, 2021. URL https://api.semanticscholar.org/CorpusID:237572264.   
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford\_alpaca, 2023.   
Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085, 2022.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
Lewis Tunstall, Nathan Lambert, Nazneen Rajani, Edward Beeching, Teven Le Scao, Leandro von Werra, Sheon Han, Philipp Schmid, and Alexander Rush. Creating a coding assistant with starcoder. Hugging Face Blog, 2023. https://huggingface.co/blog/starchat.   
Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D'souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, et al. Aya model: An instruction finetuned open-access multilingual language model. arXiv preprint arXiv:2402.07827, 2024.   
Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. SuperGLUE: A stickier benchmark for general-purpose language understanding systems. Conference on Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1905.00537.   
Ben Wang and Aran Komatsuzaki. Gpt-j-6b: A 6 billion parameter autoregressive language model, 2021.   
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations (ICLR), 2023a. URL https://openreview.net/forum?id=1PL1NIMMrw.   
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560, 2022a.   
Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705, 2022b.   
Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A Smith, Iz Beltagy, et al. How far can camels go? exploring the state of instruction tuning on open resources. arXiv preprint arXiv:2306.04751, 2023b.   
Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv:2305.07922, 2023c.   
Zhiruo Wang, Grace Cuenca, Shuyan Zhou, Frank F Xu, and Graham Neubig. Mconala: a benchmark for code generation from multiple natural languages. arXiv preprint arXiv:2203.08388, 2022c.

Zhiruo Wang, Shuyan Zhou, Daniel Fried, and Graham Neubig. Execution-based evaluation for open-domain code generation. arXiv preprint arXiv:2212.10481, 2022d.   
Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. International Conference on Learning Representations (ICLR), 2022. URL https://openreview.net/forum?id=gEZrGCozdqR.   
Jiayi Wei, Greg Durrett, and Isil Dillig. Coeditor: Leveraging contextual changes for multi-round code auto-editing. arXiv preprint arXiv:2305.18584, 2023.   
Minghao Wu and Alham Fikri Aji. Style over substance: Evaluation biases for large language models. arXiv preprint arXiv:2307.03025, 2023.   
Chunqiu Steven Xia and Lingming Zhang. Conversational automated program repair. arXiv preprint arXiv:2301.13246, 2023.   
Mengzhou Xia, Mikel Artetxe, Chunting Zhou, Xi Victoria Lin, Ramakanth Pasunuru, Danqi Chen, Luke Zettlemoyer, and Ves Stoyanov. Training trajectories of language models across scales. arXiv preprint arXiv:2212.09803, 2022.   
Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023a.   
Frank F Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn. A systematic evaluation of large language models of code. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming, pp. 1–10, 2022a.   
Shengbin Xu, Yuan Yao, Feng Xu, Tianxiao Gu, and Hanghang Tong. Combining code context and fine-grained code difference for commit message generation. In Proceedings of the 13th Asia-Pacific Symposium on Internetware, pp. 242–251, 2022b.   
Zhiyang Xu, Ying Shen, and Lifu Huang. Multiinstruct: Improving multi-modal zero-shot learning via instruction tuning, 2023b.   
Michihiro Yasunaga and Percy Liang. Break-it-fix-it: Unsupervised learning for program repair. In International Conference on Machine Learning, pp. 11941–11952. PMLR, 2021.   
He Ye, Matias Martinez, Thomas Durieux, and Martin Monperrus. A comprehensive study of automatic program repair on the quixbugs benchmark. Journal of Systems and Software, 171:110825, 2021.   
Burak Yetistiren, Isik Ozsoy, and Eray Tuzun. Assessing the quality of github copilot's code generation. In Proceedings of the 18th International Conference on Predictive Models and Data Analytics in Software Engineering, pp. 62–71, 2022.   
Pengcheng Yin, Bowen Deng, Edgar Chen, Bogdan Vasilescu, and Graham Neubig. Learning to mine aligned code and natural language pairs from stack overflow. In International Conference on Mining Software Repositories, MSR, pp. 476–486. ACM, 2018. doi: https://doi.org/10.1145/3196398.3196408.   
Zheng-Xin Yong, Hailey Schoelkopf, Niklas Muennighoff, Alham Fikri Aji, David Ifeoluwa Adelani, Khalid Almubarak, M Saiful Bari, Lintang Sutawika, Jungo Kasai, Ahmed Baruwa, et al. Bloom+1: Adding language support to bloom for zero-shot prompting. arXiv preprint arXiv:2212.09535, 2022.   
Hao Yu, Bo Shen, Dezhi Ran, Jiaxin Zhang, Qi Zhang, Yuchi Ma, Guangtai Liang, Ying Li, Tao Xie, and Qianxiang Wang. Codereval: A benchmark of pragmatic code generation with generative pre-trained models. arXiv preprint arXiv:2302.00288, 2023.   
Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, and Tao Kong. What matters in training a gpt4-style language model with multimodal inputs? arXiv preprint arXiv:2307.02469, 2023.

Chunyan Zhang, Junchao Wang, Qinglei Zhou, Ting Xu, Ke Tang, Hairen Gui, and Fudong Liu. A survey of automatic source code summarization. Symmetry, 14(3):471, 2022a.   
Fengji Zhang, Bei Chen, Yue Zhang, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. Repocoder: Repository-level code completion through iterative retrieval and generation. arXiv preprint arXiv:2303.12570, 2023a.   
Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023b.   
Jialu Zhang, José Cambronero, Sumit Gulwani, Vu Le, Ruzica Piskac, Gustavo Soares, and Gust Verbruggen. Repairing bugs in python assignments using large language models. arXiv preprint arXiv:2209.14876, 2022b.   
Jiyang Zhang, Sheena Panthaplackel, Pengyu Nie, Junyi Jessy Li, and Milos Gligoric. Coditt5: Pretraining for source code and natural language editing. In 37th IEEE/ACM International Conference on Automated Software Engineering, pp. 1–12, 2022c.   
Tianyi Zhang, Tao Yu, Tatsunori Hashimoto, Mike Lewis, Wen-tau Yih, Daniel Fried, and Sida Wang. Coder reviewer reranking for code generation. In International Conference on Machine Learning, pp. 41832–41846. PMLR, 2023c.   
Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang, Lei Shen, Andi Wang, Yang Li, et al. Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x. arXiv preprint arXiv:2303.17568, 2023.   
Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. arXiv preprint arXiv:2305.11206, 2023a.   
Shuyan Zhou, Uri Alon, Sumit Agarwal, and Graham Neubig. Codebertscore: Evaluating code generation with pretrained models of code. arXiv preprint arXiv:2302.05527, 2023b.   
Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910, 2022.   
Ming Zhu, Aneesh Jain, Karthik Suresh, Roshan Ravindran, Sindhu Tipirneni, and Chandan K Reddy. Xlcost: A benchmark dataset for cross-lingual code intelligence. arXiv preprint arXiv:2206.08474, 2022.   
Terry Yue Zhuo. Large language models are state-of-the-art evaluators of code generation. arXiv preprint arXiv:2304.14317, 2023.   
Terry Yue Zhuo, Armel Zebaze, Nitchakarn Suppattarachai, Leandro von Werra, Harm de Vries, Qian Liu, and Niklas Muennighoff. Astraios: Parameter-efficient instruction tuning code large language models. arXiv preprint arXiv:2401.00788, 2024.

# APPENDIX

# Contents

A Contributions 22

B Artifacts 22

C COMMITPACK and COMMITPACKFT Languages 23

D Dataset Creation 29

E Comparing Data Before and After Filtering 32

F Comparing COMMITPACK and The Stack 32

G Pretraining on COMMITPACK 33

H HUMANEVALPACK Statistics 33

I Full Instruction Data Ablations 34

J Line Diff Format for Fixing Code 34

K Results on HUMANEVALFIXDOCS 37

L HUMANEVALFIX Bug Types 37

M Performance Breakdown by HUMANEVALFIX Bug Type 40

N HUMANEVALEXPLAIN with Fill-In-The-Middle 40

O HUMANEVALEXPLAIN BLEU and METEOR comparison 41

P Hyperparameters 41

Q Prompts 41

R Examples 45

R.1 OCTOCODER 45

R.2 GPT-4 48

R.3 WizardCoder 53

R.4 BLOOMZ 54

R.5 StarCoder 54

R.6 InstructCodeT5+ 56

R.7 StarChat- $\beta$ 56

R.8 Diff Codegen 58

S Limitations and Future Work 58

T Version Control 59

U OCTOBADPACK 60

# A CONTRIBUTIONS

Niklas Muennighoff created COMMITPACK and HUMANEVALPACK, wrote most of the paper and led the project. Qian Liu devised many quality filters, ran SantaCoder ablations, investigated early training decisions and helped edit the paper. Armel Zebaze created the Self-Instruct data and ran numerous ablations. Niklas Muennighoff, Armel Zebaze and Qinkai Zheng created and evaluated OCTOCODER and OCTOGEEX. Binyuan Hui pretrained SantaCoder, made major contributions to the presentation and helped edit the paper. Terry Yue Zhuo ran GPT-4 evaluations and helped edit the paper. Xiangru Tang provided help on several experiments for evaluation and helped edit the paper. Leandro von Werra provided early guidance, suggested many quality filters and added the commit data to StarCoder pretraining. Niklas Muennighoff, Qian Liu, Binyuan Hui, Swayam Singh and Shayne Longpre conducted the data analysis. Shayne Longpre advised the project and made large contributions to the paper.

# B ARTIFACTS

<table><tr><td>Model</td><td>Public Link</td></tr><tr><td colspan="2">Other models</td></tr><tr><td>Diff Codegen 2B (Bradley et al., 2023)InstructCodeT5+ (Wang et al., 2023c)BLOOMZ (Muennighoff et al., 2022b)StarChat-β (Tunstall et al., 2023)CodeGeeX2 (Zheng et al., 2023)SantaCoder (Allal et al., 2023)StarCoder (Li et al., 2023b)WizardCoder (Luo et al., 2023)GPT-4 (OpenAI, 2023)</td><td>https://hf.co/CarperAI/diff-codegen-2b-v2https://hf.co/Salesforce/instructcodet5p-16bhttps://hf.co/bigscience/bloomzhttps://hf.co/HuggingFaceH4/starchat-betahttps://github.com/THUDM/CodeGeeX2https://hf.co/bigcode/santacoderhttps://hf.co/bigcode/starcoderhttps://hf.co/WizardLM/WizardCoder-15B-V1.0https://openai.com/gpt-4</td></tr><tr><td colspan="2">Data Ablations (Appendix I) - Data</td></tr><tr><td>Filtered xP3x codeStarCoder Self-InstructFiltered OASSTManual selection (Appendix I)</td><td>https://hf.co/datasets/bigcode/xp3x-octopackhttps://hf.co/datasets/codeparrot/self-instruct-starcoderhttps://hf.co/datasets/bigcode/oasst-octopackhttps://hf.co/datasets/bigcode/co-manual</td></tr><tr><td colspan="2">Data Ablations (Appendix I) - Models</td></tr><tr><td>Self-Instruct (SI)OASST (O)SI + OxP3x + OCOMMITPACKFT + O (Formatting)COMMITPACKFT + O (Target loss)COMMITPACKFT + O (Manual)COMMITPACKFT + xP3x + OCOMMITPACKFT + xP3x + SI + O</td><td>https://hf.co/bigcode/starcoder-shttps://hf.co/bigcode/starcoder-ohttps://hf.co/bigcode/starcoder-sohttps://hf.co/bigcode/starcoder-xohttps://hf.co/bigcode/starcoder-co-formathttps://hf.co/bigcode/starcoder-co-targethttps://hf.co/bigcode/starcoder-co-manualhttps://hf.co/bigcode/starcoder-cxohttps://hf.co/bigcode/starcoder-cxso</td></tr><tr><td colspan="2">SantaCoder ablations (Appendix G, Appendix J)</td></tr><tr><td>Commit format PretrainingCommit format FinetuningLine diff format Finetuning</td><td>https://hf.co/bigcode/santacoderpackhttps://hf.co/bigcode/santacoder-cfhttps://hf.co/bigcode/santacoder-ldf</td></tr><tr><td colspan="2">Other datasets</td></tr><tr><td>COMMITPACK Metadata</td><td>https://hf.co/datasets/bigcode/commitpackmeta</td></tr><tr><td colspan="2">Main artifacts</td></tr><tr><td>COMMITPACKCOMMITPACKFTHUMANIEVALPACKOCTOGEEXOCTOCODER</td><td>https://hf.co/datasets/bigcode/commitpackhttps://hf.co/datasets/bigcode/commitpackfthttps://hf.co/datasets/bigcode/humanevalpackhttps://hf.co/bigcode/octogeexhttps://hf.co/bigcode/octocoder</td></tr></table>

Table 3: Used and produced artifacts.

C COMMITPACK AND COMMITPACKFT LANGUAGES 

<table><tr><td rowspan="2">Language (↓)</td><td colspan="3">COMMITPACK</td><td colspan="3">COMMITPACKFT</td></tr><tr><td>MB</td><td>Samples</td><td>% (MB)</td><td>MB</td><td>Samples</td><td>% (MB)</td></tr><tr><td>Total</td><td>3709175.78</td><td>57700105</td><td>100.0</td><td>1545.02</td><td>702062</td><td>100.0</td></tr><tr><td>json</td><td>583293.82</td><td>3495038</td><td>15.73</td><td>86.74</td><td>39777</td><td>5.61</td></tr><tr><td>xml</td><td>279208.68</td><td>1923159</td><td>7.53</td><td>23.68</td><td>9337</td><td>1.53</td></tr><tr><td>text</td><td>270662.6</td><td>1389525</td><td>7.3</td><td>66.66</td><td>46588</td><td>4.31</td></tr><tr><td>javascript</td><td>262824.84</td><td>5401937</td><td>7.09</td><td>125.01</td><td>52989</td><td>8.09</td></tr><tr><td>objective-c++</td><td>239009.3</td><td>32227</td><td>6.44</td><td>0.38</td><td>86</td><td>0.02</td></tr><tr><td>python</td><td>234311.56</td><td>6189601</td><td>6.32</td><td>132.68</td><td>56025</td><td>8.59</td></tr><tr><td>c</td><td>200876.8</td><td>2779478</td><td>5.42</td><td>21.08</td><td>8506</td><td>1.36</td></tr><tr><td>c++</td><td>186585.26</td><td>2402294</td><td>5.03</td><td>14.14</td><td>4992</td><td>0.92</td></tr><tr><td>markdown</td><td>171849.95</td><td>7645354</td><td>4.63</td><td>131.15</td><td>62518</td><td>8.49</td></tr><tr><td>java</td><td>127103.45</td><td>3744377</td><td>3.43</td><td>56.28</td><td>20635</td><td>3.64</td></tr><tr><td>html</td><td>105305.28</td><td>2366841</td><td>2.84</td><td>48.42</td><td>20214</td><td>3.13</td></tr><tr><td>yaml</td><td>100466.64</td><td>2592787</td><td>2.71</td><td>190.88</td><td>114320</td><td>12.35</td></tr><tr><td>go</td><td>86444.62</td><td>1183612</td><td>2.33</td><td>12.13</td><td>5004</td><td>0.79</td></tr><tr><td>csv</td><td>82946.19</td><td>79268</td><td>2.24</td><td>0.53</td><td>375</td><td>0.03</td></tr><tr><td>php</td><td>74961.64</td><td>2555419</td><td>2.02</td><td>60.22</td><td>24791</td><td>3.9</td></tr><tr><td>jupyter-notebook</td><td>66854.08</td><td>94000</td><td>1.8</td><td>0.1</td><td>48</td><td>0.01</td></tr><tr><td>gettext-catalog</td><td>62296.88</td><td>168327</td><td>1.68</td><td>0.13</td><td>72</td><td>0.01</td></tr><tr><td>sql</td><td>56802.76</td><td>132772</td><td>1.53</td><td>3.74</td><td>2069</td><td>0.24</td></tr><tr><td>unity3d-asset</td><td>39535.01</td><td>17867</td><td>1.07</td><td>0.16</td><td>101</td><td>0.01</td></tr><tr><td>typescript</td><td>39254.8</td><td>572136</td><td>1.06</td><td>14.28</td><td>5868</td><td>0.92</td></tr><tr><td>owl</td><td>36435.46</td><td>7458</td><td>0.98</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>ruby</td><td>35830.74</td><td>2928702</td><td>0.97</td><td>195.29</td><td>69413</td><td>12.64</td></tr><tr><td>c#</td><td>33669.65</td><td>923157</td><td>0.91</td><td>26.84</td><td>9346</td><td>1.74</td></tr><tr><td>nix</td><td>33547.92</td><td>221281</td><td>0.9</td><td>3.84</td><td>1593</td><td>0.25</td></tr><tr><td>shell</td><td>25109.95</td><td>1017977</td><td>0.68</td><td>66.86</td><td>31217</td><td>4.33</td></tr><tr><td>perl</td><td>21148.93</td><td>374266</td><td>0.57</td><td>4.99</td><td>2288</td><td>0.32</td></tr><tr><td>tex</td><td>17471.11</td><td>89283</td><td>0.47</td><td>0.56</td><td>307</td><td>0.04</td></tr><tr><td>css</td><td>16306.63</td><td>548818</td><td>0.44</td><td>9.36</td><td>5049</td><td>0.61</td></tr><tr><td>restructuredtext</td><td>15613.89</td><td>494037</td><td>0.42</td><td>15.73</td><td>6560</td><td>1.02</td></tr><tr><td>rust</td><td>15011.3</td><td>296214</td><td>0.4</td><td>7.24</td><td>2996</td><td>0.47</td></tr><tr><td>groff</td><td>12020.19</td><td>32923</td><td>0.32</td><td>0.4</td><td>192</td><td>0.03</td></tr><tr><td>ini</td><td>8375.16</td><td>297100</td><td>0.23</td><td>21.04</td><td>11360</td><td>1.36</td></tr><tr><td>scala</td><td>8325.96</td><td>316064</td><td>0.22</td><td>11.18</td><td>5040</td><td>0.72</td></tr><tr><td>coffeescrit</td><td>6795.14</td><td>292446</td><td>0.18</td><td>16.96</td><td>5513</td><td>1.1</td></tr><tr><td>haskell</td><td>6306.12</td><td>217325</td><td>0.17</td><td>3.31</td><td>1389</td><td>0.21</td></tr><tr><td>swift</td><td>5902.72</td><td>319289</td><td>0.16</td><td>16.27</td><td>4849</td><td>1.05</td></tr><tr><td>lua</td><td>5763.12</td><td>139091</td><td>0.16</td><td>1.85</td><td>920</td><td>0.12</td></tr><tr><td>svg</td><td>5645.44</td><td>27095</td><td>0.15</td><td>0.25</td><td>169</td><td>0.02</td></tr><tr><td>gas</td><td>5585.38</td><td>15121</td><td>0.15</td><td>0.34</td><td>193</td><td>0.02</td></tr><tr><td>ocaml</td><td>5355.4</td><td>81360</td><td>0.14</td><td>0.7</td><td>333</td><td>0.05</td></tr><tr><td>erlang</td><td>5043.32</td><td>93685</td><td>0.14</td><td>1.19</td><td>480</td><td>0.08</td></tr><tr><td>makefile</td><td>4238.51</td><td>343379</td><td>0.11</td><td>2.53</td><td>960</td><td>0.16</td></tr><tr><td>asciidoc</td><td>4138.59</td><td>96671</td><td>0.11</td><td>1.86</td><td>523</td><td>0.12</td></tr><tr><td>emacs-lisp</td><td>3988.65</td><td>83228</td><td>0.11</td><td>1.97</td><td>1015</td><td>0.13</td></tr><tr><td>scss</td><td>3944.94</td><td>288190</td><td>0.11</td><td>13.21</td><td>6829</td><td>0.86</td></tr><tr><td>clojure</td><td>3523.41</td><td>158674</td><td>0.09</td><td>5.07</td><td>2403</td><td>0.33</td></tr><tr><td>org</td><td>3126.22</td><td>30198</td><td>0.08</td><td>0.27</td><td>136</td><td>0.02</td></tr><tr><td>common-lisp</td><td>2954.9</td><td>74628</td><td>0.08</td><td>1.45</td><td>778</td><td>0.09</td></tr><tr><td>diff</td><td>2586.05</td><td>21021</td><td>0.07</td><td>1.48</td><td>680</td><td>0.1</td></tr><tr><td>groovy</td><td>2569.14</td><td>110057</td><td>0.07</td><td>4.17</td><td>1486</td><td>0.27</td></tr><tr><td>html+erb</td><td>2450.68</td><td>225379</td><td>0.07</td><td>23.1</td><td>10910</td><td>1.5</td></tr><tr><td>nesc</td><td>2439.56</td><td>473</td><td>0.07</td><td>0.02</td><td>7</td><td>0.0</td></tr><tr><td>dart</td><td>2395.8</td><td>56873</td><td>0.06</td><td>1.96</td><td>765</td><td>0.13</td></tr><tr><td>powershell</td><td>2289.28</td><td>55381</td><td>0.06</td><td>2.06</td><td>991</td><td>0.13</td></tr><tr><td>f#</td><td>2289.24</td><td>66840</td><td>0.06</td><td>0.66</td><td>254</td><td>0.04</td></tr><tr><td>dm</td><td>2223.14</td><td>55584</td><td>0.06</td><td>0.15</td><td>16</td><td>0.01</td></tr><tr><td>kotlin</td><td>2219.25</td><td>124266</td><td>0.06</td><td>5.37</td><td>2214</td><td>0.35</td></tr><tr><td>pascal</td><td>2194.68</td><td>42511</td><td>0.06</td><td>0.05</td><td>25</td><td>0.0</td></tr><tr><td>jsx</td><td>2124.74</td><td>139148</td><td>0.06</td><td>5.5</td><td>2199</td><td>0.36</td></tr><tr><td>viml</td><td>1948.21</td><td>74062</td><td>0.05</td><td>1.96</td><td>1063</td><td>0.13</td></tr><tr><td>actionscript</td><td>1844.15</td><td>28819</td><td>0.05</td><td>0.12</td><td>49</td><td>0.01</td></tr><tr><td>cython</td><td>1736.59</td><td>25927</td><td>0.05</td><td>0.31</td><td>123</td><td>0.02</td></tr><tr><td>turtle</td><td>1698.95</td><td>3882</td><td>0.05</td><td>0.05</td><td>21</td><td>0.0</td></tr><tr><td>less</td><td>1616.56</td><td>88634</td><td>0.04</td><td>3.72</td><td>1360</td><td>0.24</td></tr><tr><td>mathematica</td><td>1475.04</td><td>925</td><td>0.04</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>xslt</td><td>1441.46</td><td>27956</td><td>0.04</td><td>0.26</td><td>99</td><td>0.02</td></tr><tr><td>scheme</td><td>1249.24</td><td>30546</td><td>0.03</td><td>0.42</td><td>213</td><td>0.03</td></tr><tr><td>perl6</td><td>1223.16</td><td>12167</td><td>0.03</td><td>0.27</td><td>122</td><td>0.02</td></tr><tr><td>edn</td><td>1186.94</td><td>2289</td><td>0.03</td><td>0.09</td><td>48</td><td>0.01</td></tr><tr><td>fortran</td><td>1178.55</td><td>13463</td><td>0.03</td><td>0.14</td><td>70</td><td>0.01</td></tr><tr><td>java-server-pages</td><td>1173.07</td><td>53574</td><td>0.03</td><td>0.45</td><td>173</td><td>0.03</td></tr><tr><td>standard-ml</td><td>1133.48</td><td>20097</td><td>0.03</td><td>0.15</td><td>72</td><td>0.01</td></tr><tr><td>cmake</td><td>1132.07</td><td>58446</td><td>0.03</td><td>2.27</td><td>981</td><td>0.15</td></tr><tr><td>json5</td><td>1108.2</td><td>1827</td><td>0.03</td><td>0.08</td><td>33</td><td>0.01</td></tr><tr><td>vala</td><td>1104.51</td><td>14822</td><td>0.03</td><td>0.12</td><td>50</td><td>0.01</td></tr><tr><td>vue</td><td>1093.8</td><td>68967</td><td>0.03</td><td>1.38</td><td>587</td><td>0.09</td></tr><tr><td>freemarker</td><td>1032.33</td><td>36216</td><td>0.03</td><td>1.03</td><td>510</td><td>0.07</td></tr><tr><td>graphql</td><td>1004.84</td><td>2009</td><td>0.03</td><td>0.03</td><td>17</td><td>0.0</td></tr><tr><td>twig</td><td>958.96</td><td>39588</td><td>0.03</td><td>3.96</td><td>1610</td><td>0.26</td></tr><tr><td>tcl</td><td>869.83</td><td>16407</td><td>0.02</td><td>0.29</td><td>103</td><td>0.02</td></tr><tr><td>pod</td><td>859.02</td><td>14922</td><td>0.02</td><td>0.15</td><td>54</td><td>0.01</td></tr><tr><td>dockerfile</td><td>849.73</td><td>259379</td><td>0.02</td><td>0.1</td><td>39</td><td>0.01</td></tr><tr><td>yacc</td><td>845.7</td><td>8230</td><td>0.02</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>postscript</td><td>800.73</td><td>903</td><td>0.02</td><td>0.02</td><td>9</td><td>0.0</td></tr><tr><td>racket</td><td>796.64</td><td>16615</td><td>0.02</td><td>0.2</td><td>117</td><td>0.01</td></tr><tr><td>eagle</td><td>785.68</td><td>2237</td><td>0.02</td><td>0.01</td><td>4</td><td>0.0</td></tr><tr><td>haxe</td><td>772.9</td><td>28447</td><td>0.02</td><td>0.34</td><td>174</td><td>0.02</td></tr><tr><td>julia</td><td>752.07</td><td>22695</td><td>0.02</td><td>0.31</td><td>180</td><td>0.02</td></tr><tr><td>handlebars</td><td>740.82</td><td>49842</td><td>0.02</td><td>3.29</td><td>1429</td><td>0.21</td></tr><tr><td>smarty</td><td>720.94</td><td>41065</td><td>0.02</td><td>1.59</td><td>737</td><td>0.1</td></tr><tr><td>visual-basic</td><td>681.52</td><td>10511</td><td>0.02</td><td>0.15</td><td>48</td><td>0.01</td></tr><tr><td>literate-haskell</td><td>673.74</td><td>10729</td><td>0.02</td><td>0.02</td><td>7</td><td>0.0</td></tr><tr><td>smalltalk</td><td>665.89</td><td>11741</td><td>0.02</td><td>0.46</td><td>284</td><td>0.03</td></tr><tr><td>isabelle</td><td>655.82</td><td>8359</td><td>0.02</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>nimrod</td><td>652.86</td><td>12023</td><td>0.02</td><td>0.24</td><td>67</td><td>0.02</td></tr><tr><td>zig</td><td>621.38</td><td>4290</td><td>0.02</td><td>0.01</td><td>4</td><td>0.0</td></tr><tr><td>m4</td><td>603.58</td><td>12465</td><td>0.02</td><td>0.26</td><td>101</td><td>0.02</td></tr><tr><td>max</td><td>603.56</td><td>2259</td><td>0.02</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>elixir</td><td>558.12</td><td>35473</td><td>0.02</td><td>2.35</td><td>1150</td><td>0.15</td></tr><tr><td>mako</td><td>543.01</td><td>8943</td><td>0.01</td><td>0.76</td><td>170</td><td>0.05</td></tr><tr><td>arduino</td><td>534.18</td><td>32350</td><td>0.01</td><td>0.46</td><td>225</td><td>0.03</td></tr><tr><td>jade</td><td>531.4</td><td>46993</td><td>0.01</td><td>2.35</td><td>1119</td><td>0.15</td></tr><tr><td>haml</td><td>502.01</td><td>74792</td><td>0.01</td><td>10.74</td><td>4415</td><td>0.7</td></tr><tr><td>elm</td><td>481.97</td><td>18542</td><td>0.01</td><td>0.62</td><td>265</td><td>0.04</td></tr><tr><td>purebasic</td><td>474.28</td><td>36</td><td>0.01</td><td>0.02</td><td>5</td><td>0.0</td></tr><tr><td>coldfusion</td><td>470.78</td><td>9263</td><td>0.01</td><td>0.02</td><td>9</td><td>0.0</td></tr><tr><td>lean</td><td>470.03</td><td>7507</td><td>0.01</td><td>0.02</td><td>3</td><td>0.0</td></tr><tr><td>r</td><td>454.32</td><td>12858</td><td>0.01</td><td>0.23</td><td>121</td><td>0.01</td></tr><tr><td>cuda</td><td>437.67</td><td>11450</td><td>0.01</td><td>0.07</td><td>25</td><td>0.0</td></tr><tr><td>textile</td><td>425.12</td><td>18491</td><td>0.01</td><td>0.18</td><td>61</td><td>0.01</td></tr><tr><td>robotframework</td><td>421.61</td><td>9211</td><td>0.01</td><td>0.21</td><td>85</td><td>0.01</td></tr><tr><td>abap</td><td>409.62</td><td>1955</td><td>0.01</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>rdoc</td><td>397.03</td><td>38760</td><td>0.01</td><td>0.55</td><td>270</td><td>0.04</td></tr><tr><td>llvm</td><td>382.2</td><td>10727</td><td>0.01</td><td>1.6</td><td>780</td><td>0.1</td></tr><tr><td>ada</td><td>380.7</td><td>13258</td><td>0.01</td><td>0.73</td><td>265</td><td>0.05</td></tr><tr><td>batchfile</td><td>372.16</td><td>43674</td><td>0.01</td><td>2.98</td><td>1466</td><td>0.19</td></tr><tr><td>qml</td><td>361.45</td><td>19360</td><td>0.01</td><td>0.94</td><td>368</td><td>0.06</td></tr><tr><td>jasmin</td><td>359.82</td><td>4782</td><td>0.01</td><td>0.05</td><td>9</td><td>0.0</td></tr><tr><td>assembly</td><td>343.62</td><td>8126</td><td>0.01</td><td>0.17</td><td>105</td><td>0.01</td></tr><tr><td>g-code</td><td>334.96</td><td>3690</td><td>0.01</td><td>0.04</td><td>7</td><td>0.0</td></tr><tr><td>cucumber</td><td>331.38</td><td>26677</td><td>0.01</td><td>2.59</td><td>976</td><td>0.17</td></tr><tr><td>html+php</td><td>323.35</td><td>18381</td><td>0.01</td><td>0.33</td><td>150</td><td>0.02</td></tr><tr><td>kicad</td><td>321.94</td><td>759</td><td>0.01</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>api-blueprint</td><td>317.85</td><td>4765</td><td>0.01</td><td>0.06</td><td>23</td><td>0.0</td></tr><tr><td>eiffel</td><td>311.48</td><td>373</td><td>0.01</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>toml</td><td>292.68</td><td>63517</td><td>0.01</td><td>5.58</td><td>3424</td><td>0.36</td></tr><tr><td>modelica</td><td>284.62</td><td>2611</td><td>0.01</td><td>0.04</td><td>15</td><td>0.0</td></tr><tr><td>bitbake</td><td>277.58</td><td>43239</td><td>0.01</td><td>4.46</td><td>1308</td><td>0.29</td></tr><tr><td>lex</td><td>275.96</td><td>705</td><td>0.01</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>stylus</td><td>273.06</td><td>21967</td><td>0.01</td><td>0.95</td><td>480</td><td>0.06</td></tr><tr><td>protocol-buffer</td><td>254.12</td><td>9202</td><td>0.01</td><td>0.52</td><td>181</td><td>0.03</td></tr><tr><td>unknown</td><td>252.23</td><td>30570</td><td>0.01</td><td>3.05</td><td>1597</td><td>0.2</td></tr><tr><td>nit</td><td>244.54</td><td>4951</td><td>0.01</td><td>0.02</td><td>3</td><td>0.0</td></tr><tr><td>factor</td><td>241.19</td><td>15378</td><td>0.01</td><td>0.36</td><td>113</td><td>0.02</td></tr><tr><td>xs</td><td>239.04</td><td>3215</td><td>0.01</td><td>0.02</td><td>7</td><td>0.0</td></tr><tr><td>sass</td><td>230.65</td><td>23144</td><td>0.01</td><td>1.36</td><td>705</td><td>0.09</td></tr><tr><td>pir</td><td>230.2</td><td>6231</td><td>0.01</td><td>0.08</td><td>23</td><td>0.01</td></tr><tr><td>html+django</td><td>217.04</td><td>10535</td><td>0.01</td><td>0.85</td><td>399</td><td>0.06</td></tr><tr><td>mediawiki</td><td>214.32</td><td>10188</td><td>0.01</td><td>0.08</td><td>33</td><td>0.01</td></tr><tr><td>logos</td><td>212.3</td><td>1733</td><td>0.01</td><td>0.04</td><td>19</td><td>0.0</td></tr><tr><td>genshi</td><td>209.3</td><td>956</td><td>0.01</td><td>0.02</td><td>3</td><td>0.0</td></tr><tr><td>coldfusion-cfc</td><td>208.16</td><td>4410</td><td>0.01</td><td>0.05</td><td>20</td><td>0.0</td></tr><tr><td>xtend</td><td>179.54</td><td>7775</td><td>0.0</td><td>0.13</td><td>55</td><td>0.01</td></tr><tr><td>sqf</td><td>168.66</td><td>7778</td><td>0.0</td><td>0.09</td><td>45</td><td>0.01</td></tr><tr><td>vhdl</td><td>155.95</td><td>2185</td><td>0.0</td><td>0.02</td><td>5</td><td>0.0</td></tr><tr><td>antlr</td><td>143.55</td><td>3651</td><td>0.0</td><td>0.03</td><td>15</td><td>0.0</td></tr><tr><td>systemverilog</td><td>140.19</td><td>3944</td><td>0.0</td><td>0.08</td><td>35</td><td>0.01</td></tr><tr><td>hcl</td><td>136.75</td><td>13379</td><td>0.0</td><td>0.91</td><td>421</td><td>0.06</td></tr><tr><td>asp</td><td>136.1</td><td>4286</td><td>0.0</td><td>0.09</td><td>22</td><td>0.01</td></tr><tr><td>nsis</td><td>129.12</td><td>4048</td><td>0.0</td><td>0.06</td><td>15</td><td>0.0</td></tr><tr><td>inform-7</td><td>120.19</td><td>184</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>slim</td><td>119.04</td><td>18726</td><td>0.0</td><td>2.06</td><td>1052</td><td>0.13</td></tr><tr><td>groovy-server-pages</td><td>117.37</td><td>6695</td><td>0.0</td><td>0.07</td><td>25</td><td>0.0</td></tr><tr><td>ceylon</td><td>116.14</td><td>7256</td><td>0.0</td><td>0.1</td><td>49</td><td>0.01</td></tr><tr><td>fish</td><td>111.28</td><td>15351</td><td>0.0</td><td>1.33</td><td>813</td><td>0.09</td></tr><tr><td>processing</td><td>108.58</td><td>5912</td><td>0.0</td><td>0.07</td><td>35</td><td>0.0</td></tr><tr><td>component-pascal</td><td>105.5</td><td>43</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>lasso</td><td>104.17</td><td>67</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>glsl</td><td>99.49</td><td>9478</td><td>0.0</td><td>0.34</td><td>164</td><td>0.02</td></tr><tr><td>saltstack</td><td>98.2</td><td>12314</td><td>0.0</td><td>1.41</td><td>617</td><td>0.09</td></tr><tr><td>xbase</td><td>94.42</td><td>1670</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>autohotkey</td><td>94.22</td><td>1452</td><td>0.0</td><td>0.02</td><td>15</td><td>0.0</td></tr><tr><td>liquid</td><td>93.79</td><td>2651</td><td>0.0</td><td>0.09</td><td>30</td><td>0.01</td></tr><tr><td>purescript</td><td>92.41</td><td>5024</td><td>0.0</td><td>0.17</td><td>80</td><td>0.01</td></tr><tr><td>agda</td><td>92.06</td><td>4956</td><td>0.0</td><td>0.02</td><td>10</td><td>0.0</td></tr><tr><td>inno-setup</td><td>91.36</td><td>3014</td><td>0.0</td><td>0.06</td><td>16</td><td>0.0</td></tr><tr><td>oz</td><td>90.48</td><td>1551</td><td>0.0</td><td>0.03</td><td>8</td><td>0.0</td></tr><tr><td>chapel</td><td>89.62</td><td>26447</td><td>0.0</td><td>0.04</td><td>20</td><td>0.0</td></tr><tr><td>arc</td><td>87.21</td><td>758</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>opencl</td><td>86.43</td><td>2489</td><td>0.0</td><td>0.05</td><td>23</td><td>0.0</td></tr><tr><td>graphviz-dot</td><td>85.8</td><td>1525</td><td>0.0</td><td>0.07</td><td>35</td><td>0.0</td></tr><tr><td>pawn</td><td>85.42</td><td>580</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>jsoniq</td><td>75.15</td><td>1343</td><td>0.0</td><td>0.01</td><td>6</td><td>0.0</td></tr><tr><td>bluespec</td><td>72.38</td><td>2500</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>smali</td><td>71.38</td><td>174</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>krl</td><td>69.87</td><td>1879</td><td>0.0</td><td>0.02</td><td>4</td><td>0.0</td></tr><tr><td>maple</td><td>68.28</td><td>1311</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>unrealscript</td><td>67.67</td><td>585</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>ooc</td><td>63.19</td><td>3416</td><td>0.0</td><td>0.04</td><td>15</td><td>0.0</td></tr><tr><td>pure-data</td><td>62.62</td><td>603</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>xquery</td><td>61.96</td><td>2237</td><td>0.0</td><td>0.08</td><td>39</td><td>0.01</td></tr><tr><td>dcl</td><td>59.64</td><td>833</td><td>0.0</td><td>0.04</td><td>19</td><td>0.0</td></tr><tr><td>moonscript</td><td>59.21</td><td>1951</td><td>0.0</td><td>0.02</td><td>10</td><td>0.0</td></tr><tr><td>awk</td><td>57.18</td><td>2206</td><td>0.0</td><td>0.1</td><td>52</td><td>0.01</td></tr><tr><td>pike</td><td>52.87</td><td>1262</td><td>0.0</td><td>0.02</td><td>6</td><td>0.0</td></tr><tr><td>livescript</td><td>51.23</td><td>5194</td><td>0.0</td><td>0.13</td><td>63</td><td>0.01</td></tr><tr><td>solidity</td><td>50.86</td><td>3689</td><td>0.0</td><td>0.08</td><td>37</td><td>0.01</td></tr><tr><td>monkey</td><td>48.26</td><td>1367</td><td>0.0</td><td>0.02</td><td>4</td><td>0.0</td></tr><tr><td>jsonld</td><td>48.01</td><td>462</td><td>0.0</td><td>0.02</td><td>6</td><td>0.0</td></tr><tr><td>zephir</td><td>42.68</td><td>1265</td><td>0.0</td><td>0.02</td><td>4</td><td>0.0</td></tr><tr><td>crystal</td><td>41.92</td><td>4217</td><td>0.0</td><td>0.35</td><td>182</td><td>0.02</td></tr><tr><td>rhtml</td><td>41.02</td><td>4551</td><td>0.0</td><td>0.35</td><td>135</td><td>0.02</td></tr><tr><td>stata</td><td>40.68</td><td>1344</td><td>0.0</td><td>0.02</td><td>10</td><td>0.0</td></tr><tr><td>idris</td><td>39.9</td><td>3025</td><td>0.0</td><td>0.13</td><td>38</td><td>0.01</td></tr><tr><td>raml</td><td>39.39</td><td>948</td><td>0.0</td><td>0.03</td><td>9</td><td>0.0</td></tr><tr><td>openscad</td><td>37.73</td><td>2178</td><td>0.0</td><td>0.05</td><td>21</td><td>0.0</td></tr><tr><td>red</td><td>35.26</td><td>1108</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>c2hs-haskell</td><td>34.47</td><td>1021</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>cycript</td><td>33.96</td><td>197</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>applescript</td><td>33.51</td><td>1304</td><td>0.0</td><td>0.04</td><td>19</td><td>0.0</td></tr><tr><td>mupad</td><td>32.49</td><td>178</td><td>0.0</td><td>0.02</td><td>4</td><td>0.0</td></tr><tr><td>literate-agda</td><td>31.38</td><td>567</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>boo</td><td>31.17</td><td>26289</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>sourcepawn</td><td>29.53</td><td>717</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>qmake</td><td>29.51</td><td>3632</td><td>0.0</td><td>0.32</td><td>140</td><td>0.02</td></tr><tr><td>ragel-in-ruby-host</td><td>28.3</td><td>888</td><td>0.0</td><td>0.01</td><td>4</td><td>0.0</td></tr><tr><td>io</td><td>27.95</td><td>1247</td><td>0.0</td><td>0.01</td><td>4</td><td>0.0</td></tr><tr><td>desktop</td><td>27.65</td><td>5021</td><td>0.0</td><td>0.36</td><td>186</td><td>0.02</td></tr><tr><td>propeller-spin</td><td>26.77</td><td>625</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>thrift</td><td>26.75</td><td>1007</td><td>0.0</td><td>0.08</td><td>28</td><td>0.01</td></tr><tr><td>volt</td><td>25.05</td><td>1660</td><td>0.0</td><td>0.02</td><td>9</td><td>0.0</td></tr><tr><td>xproc</td><td>24.21</td><td>914</td><td>0.0</td><td>0.02</td><td>3</td><td>0.0</td></tr><tr><td>igor-pro</td><td>23.75</td><td>388</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>lolcode</td><td>23.74</td><td>24861</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>html+eex</td><td>21.41</td><td>2100</td><td>0.0</td><td>0.29</td><td>135</td><td>0.02</td></tr><tr><td>logtalk</td><td>20.43</td><td>1035</td><td>0.0</td><td>0.06</td><td>21</td><td>0.0</td></tr><tr><td>mirah</td><td>20.1</td><td>706</td><td>0.0</td><td>0.04</td><td>16</td><td>0.0</td></tr><tr><td>gnuplot</td><td>19.68</td><td>889</td><td>0.0</td><td>0.03</td><td>17</td><td>0.0</td></tr><tr><td>literate-coffeescript</td><td>19.02</td><td>1041</td><td>0.0</td><td>0.05</td><td>19</td><td>0.0</td></tr><tr><td>jflex</td><td>18.61</td><td>555</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>emberscript</td><td>18.39</td><td>1024</td><td>0.0</td><td>0.02</td><td>7</td><td>0.0</td></tr><tr><td>cobol</td><td>17.0</td><td>24953</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>yang</td><td>16.94</td><td>597</td><td>0.0</td><td>0.02</td><td>6</td><td>0.0</td></tr><tr><td>rebol</td><td>16.47</td><td>239</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>linker-script</td><td>16.08</td><td>1604</td><td>0.0</td><td>0.08</td><td>37</td><td>0.01</td></tr><tr><td>cartocss</td><td>15.92</td><td>555</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>urweb</td><td>13.07</td><td>304</td><td>0.0</td><td>0.02</td><td>6</td><td>0.0</td></tr><tr><td>rmarkdown</td><td>13.03</td><td>750</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>darcs-patch</td><td>13.01</td><td>80</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>csound</td><td>12.85</td><td>229</td><td>0.0</td><td>0.01</td><td>4</td><td>0.0</td></tr><tr><td>squirrel</td><td>12.84</td><td>531</td><td>0.0</td><td>0.01</td><td>4</td><td>0.0</td></tr><tr><td>apl</td><td>12.56</td><td>586</td><td>0.0</td><td>0.02</td><td>7</td><td>0.0</td></tr><tr><td>hlsl</td><td>12.17</td><td>1529</td><td>0.0</td><td>0.03</td><td>11</td><td>0.0</td></tr><tr><td>latte</td><td>11.89</td><td>1380</td><td>0.0</td><td>0.02</td><td>7</td><td>0.0</td></tr><tr><td>pony</td><td>11.84</td><td>624</td><td>0.0</td><td>0.05</td><td>16</td><td>0.0</td></tr><tr><td>ioke</td><td>10.86</td><td>373</td><td>0.0</td><td>0.04</td><td>25</td><td>0.0</td></tr><tr><td>hy</td><td>10.51</td><td>879</td><td>0.0</td><td>0.04</td><td>12</td><td>0.0</td></tr><tr><td>uno</td><td>10.36</td><td>628</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>pan</td><td>10.34</td><td>637</td><td>0.0</td><td>0.05</td><td>23</td><td>0.0</td></tr><tr><td>xojo</td><td>10.31</td><td>642</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>papyrus</td><td>10.26</td><td>130</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>stan</td><td>10.25</td><td>540</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>slash</td><td>9.9</td><td>640</td><td>0.0</td><td>0.01</td><td>4</td><td>0.0</td></tr><tr><td>supercollider</td><td>9.8</td><td>318</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>vcl</td><td>9.46</td><td>747</td><td>0.0</td><td>0.04</td><td>18</td><td>0.0</td></tr><tr><td>smt</td><td>9.03</td><td>117</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>glyph</td><td>8.95</td><td>7</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>wisp</td><td>8.74</td><td>262</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>renpy</td><td>8.3</td><td>421</td><td>0.0</td><td>0.02</td><td>3</td><td>0.0</td></tr><tr><td>clips</td><td>7.73</td><td>450</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>dns-zone</td><td>7.56</td><td>54</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>sas</td><td>7.54</td><td>269</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>rouge</td><td>7.2</td><td>396</td><td>0.0</td><td>0.1</td><td>41</td><td>0.01</td></tr><tr><td>ec</td><td>7.03</td><td>94</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>dylan</td><td>6.82</td><td>280</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>tcsh</td><td>6.52</td><td>748</td><td>0.0</td><td>0.02</td><td>10</td><td>0.0</td></tr><tr><td>aspectj</td><td>6.33</td><td>451</td><td>0.0</td><td>0.02</td><td>8</td><td>0.0</td></tr><tr><td>netlogo</td><td>6.3</td><td>140</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>gap</td><td>6.1</td><td>46</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>fancy</td><td>5.95</td><td>675</td><td>0.0</td><td>0.02</td><td>8</td><td>0.0</td></tr><tr><td>coq</td><td>5.74</td><td>80</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>click</td><td>5.74</td><td>9</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>capn-proto</td><td>5.64</td><td>330</td><td>0.0</td><td>0.04</td><td>12</td><td>0.0</td></tr><tr><td>flux</td><td>5.57</td><td>47</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>forth</td><td>5.51</td><td>265</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>ats</td><td>5.42</td><td>383</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>netlinx</td><td>5.17</td><td>144</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>clean</td><td>5.07</td><td>171</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>parrot-assembly</td><td>4.66</td><td>227</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>alloy</td><td>4.64</td><td>203</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>lfe</td><td>4.58</td><td>287</td><td>0.0</td><td>0.02</td><td>6</td><td>0.0</td></tr><tr><td>gdscript</td><td>4.49</td><td>460</td><td>0.0</td><td>0.03</td><td>9</td><td>0.0</td></tr><tr><td>augeas</td><td>4.44</td><td>395</td><td>0.0</td><td>0.04</td><td>13</td><td>0.0</td></tr><tr><td>sparql</td><td>4.4</td><td>1036</td><td>0.0</td><td>0.04</td><td>23</td><td>0.0</td></tr><tr><td>lilypond</td><td>4.31</td><td>265</td><td>0.0</td><td>0.01</td><td>6</td><td>0.0</td></tr><tr><td>scilab</td><td>4.09</td><td>375</td><td>0.0</td><td>0.02</td><td>10</td><td>0.0</td></tr><tr><td>autoit</td><td>4.06</td><td>279</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>myghty</td><td>3.86</td><td>105</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>blitzmax</td><td>3.74</td><td>220</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>creole</td><td>3.42</td><td>337</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>harbour</td><td>3.34</td><td>107</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>piglatin</td><td>3.17</td><td>513</td><td>0.0</td><td>0.02</td><td>11</td><td>0.0</td></tr><tr><td>opa</td><td>3.16</td><td>211</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>sage</td><td>3.03</td><td>414</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>ston</td><td>2.85</td><td>414</td><td>0.0</td><td>0.01</td><td>6</td><td>0.0</td></tr><tr><td>maxscript</td><td>2.8</td><td>47</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>lsl</td><td>2.68</td><td>74</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>gentoo-ebuild</td><td>2.58</td><td>601</td><td>0.0</td><td>0.06</td><td>16</td><td>0.0</td></tr><tr><td>nu</td><td>2.38</td><td>170</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>bro</td><td>2.34</td><td>333</td><td>0.0</td><td>0.01</td><td>3</td><td>0.0</td></tr><tr><td>xc</td><td>2.02</td><td>88</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>j</td><td>1.81</td><td>142</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>metal</td><td>1.72</td><td>151</td><td>0.0</td><td>0.02</td><td>4</td><td>0.0</td></tr><tr><td>mms</td><td>1.54</td><td>91</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>webidl</td><td>1.51</td><td>96</td><td>0.0</td><td>0.05</td><td>6</td><td>0.0</td></tr><tr><td>tea</td><td>1.47</td><td>29</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>redcode</td><td>1.27</td><td>149</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>shen</td><td>1.2</td><td>71</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>pov-ray-sdl</td><td>1.14</td><td>104</td><td>0.0</td><td>0.01</td><td>5</td><td>0.0</td></tr><tr><td>x10</td><td>1.01</td><td>33</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>brainfuck</td><td>0.96</td><td>167</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>ninja</td><td>0.95</td><td>187</td><td>0.0</td><td>0.03</td><td>14</td><td>0.0</td></tr><tr><td>golo</td><td>0.9</td><td>115</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>webassembly</td><td>0.86</td><td>83</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>self</td><td>0.82</td><td>15</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>labview</td><td>0.81</td><td>61</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>octave</td><td>0.8</td><td>12</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>pogoscript</td><td>0.8</td><td>74</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>d</td><td>0.8</td><td>20</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>http</td><td>0.74</td><td>140</td><td>0.0</td><td>0.03</td><td>19</td><td>0.0</td></tr><tr><td>ecl</td><td>0.66</td><td>48</td><td>0.0</td><td>0.01</td><td>4</td><td>0.0</td></tr><tr><td>chuck</td><td>0.58</td><td>99</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>gosu</td><td>0.52</td><td>60</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>parrot</td><td>0.52</td><td>17</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>opal</td><td>0.47</td><td>69</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>objective-j</td><td>0.46</td><td>37</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>kit</td><td>0.41</td><td>48</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>gams</td><td>0.38</td><td>18</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>prolog</td><td>0.28</td><td>35</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>clarion</td><td>0.27</td><td>13</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>mask</td><td>0.25</td><td>37</td><td>0.0</td><td>0.01</td><td>4</td><td>0.0</td></tr><tr><td>brightscript</td><td>0.24</td><td>28</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>scaml</td><td>0.18</td><td>31</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>matlab</td><td>0.16</td><td>29</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>idl</td><td>0.15</td><td>1</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>ags-script</td><td>0.12</td><td>31</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>lookml</td><td>0.12</td><td>10</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>apacheconf</td><td>0.11</td><td>59</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>oxygen</td><td>0.1</td><td>9</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>txl</td><td>0.1</td><td>3</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>gf</td><td>0.09</td><td>39</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>renderscript</td><td>0.06</td><td>54</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>mtml</td><td>0.05</td><td>13</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>unified-parallel-c</td><td>0.05</td><td>6</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>dogescript</td><td>0.04</td><td>10</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>gentoo-eclass</td><td>0.04</td><td>6</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>zimpl</td><td>0.04</td><td>7</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>irc-log</td><td>0.04</td><td>9</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>fantom</td><td>0.03</td><td>11</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>numpy</td><td>0.03</td><td>1</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>cirru</td><td>0.02</td><td>4</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>xpages</td><td>0.02</td><td>7</td><td>0.0</td><td>0.01</td><td>1</td><td>0.0</td></tr><tr><td>nginx</td><td>0.02</td><td>6</td><td>0.0</td><td>0.01</td><td>2</td><td>0.0</td></tr><tr><td>objdump</td><td>0.02</td><td>1</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>python-traceback</td><td>0.02</td><td>10</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>realbasic</td><td>0.01</td><td>1</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>befunge</td><td>0.01</td><td>2</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>bison</td><td>0.01</td><td>1</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>m</td><td>0.01</td><td>1</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>omgrofl</td><td>0.01</td><td>1</td><td>0.0</td><td>0</td><td>0</td><td>0.0</td></tr></table>

Table 4: Programming language distribution of COMMITPACK and COMMITPACKFT. Shortcuts: MB=Megabytes, owl=web-ontology-language, pir=parrot-internal-representation, dcl=digital-command-language, mms=module-management-system, gf=grammatical-framework

# D DATASET CREATION

COMMITPACK We use the GitHub archive available on GCP which contains metadata from GitHub commits up to 2016. $^{4}$ It contains around 3TB of GitHub activity data for more than 2.8 million GitHub repositories including more than 145 million unique commits, over 2 billion different file paths and the contents of the latest revision for 163 million files. $^{5}$ We apply the filters in Table 5 to this dataset. The resulting dataset containing only metadata is uploaded at https://hf.co/datasets/bigcode/commitpackmeta. As the activity dataset only contains commit ids without the actual code changes, we scrape the code from GitHub. We use the metadata and the GitHub API to scrape the changed file prior and after the respective commit. Some repositories referenced in the activity data are no longer accessible, thus we discard them. This results in COMMITPACK with approximately 4 terabytes uploaded at https://hf.co/datasets/bigcode/commitpack.

<table><tr><td>Description</td><td>Details</td></tr><tr><td>License</td><td>Only keep samples licensed as MIT, Artistic-2.0, ISC, CC0-1.0, EPL-1.0, MPL-2.0, Apache-2.0, BSD-3-Clause, AGPL-3.0, LGPL-2.1, BSD-2-Clause or without license.</td></tr><tr><td>Length</td><td>Only keep code where the commit message has at least 5 and at most 10,000 characters</td></tr><tr><td>Noise</td><td>Remove code where the lowercased commit message is any of &#x27;add files via upload&#x27;, &quot;can&#x27;t you see i&#x27;m updating the time?&quot;, &#x27;commit&#x27;, &#x27;create readme.md&#x27;, &#x27;dummy&#x27;, &#x27;first commit&#x27;, &#x27;heartbeat update&#x27;, &#x27;initial commit&#x27;, &#x27;mirroring from micro.blog.&#x27;, &#x27;no message&#x27;, &#x27;pi push&#x27;, &#x27;readme&#x27;, &#x27;update&#x27;, &#x27;updates&#x27;, &#x27;update _config.yaml&#x27;, &#x27;update index.html&#x27;, &#x27;update readme.md&#x27;, &#x27;update readme&#x27;, &#x27;updated readme&#x27;, &#x27;update log&#x27;, &#x27;update data.js&#x27;, &#x27;update data.json&#x27;, &#x27;update data.js&#x27;, &#x27;pi push&#x27; or starts with &#x27;merge&#x27;</td></tr><tr><td>Single file</td><td>Remove samples that contain changes across multiple files</td></tr><tr><td>Opt-out</td><td>Remove samples from repositories owned by users that opted out of The Stack (Kocetkov et al., 2022)</td></tr></table>

Table 5: COMMITPACK filters.

COMMITPACKFT Prior work has shown the importance of careful data filtering to maintain quality (Yin et al., 2018; Dhole et al., 2021; Laurençon et al., 2022; Longpre et al., 2023b; Singh et al., 2024). To create a smaller version focused on commits that resemble high-quality instructions, we further filter COMMITPACK to create COMMITPACKFT using the steps outlined in Table 6. We also checked for any contamination with HumanEval (Chen et al., 2021) but did not find any solution or docstring present in COMMITPACKFT. This is likely because our commit data only goes up to 2016, which is several years prior to the release of HumanEval. Our filters reduce the dataset by a factor of around 1000 resulting in close to 2 gigabytes across 277 languages. To gain a deeper understanding of the rich content within COMMITPACKFT, we analyze commits on its Python subset (56K samples). We first collect the most prevalent commit domain by prompting GPT-4 with: "I'd like to know the main types of commits on Github and aim to cover as comprehensively as possible.". Subsequently, we use GPT-4 to classify each sample using the prompt in Figure 5. The task distribution is visualized in Figure 2.

<table><tr><td>Description</td><td>Details</td></tr><tr><td>Length</td><td>Remove samples where the before code has more than 50,000 characters</td></tr><tr><td>Length</td><td>Remove samples where the after code has 0 characters</td></tr><tr><td>Difference</td><td>Remove samples where the before and after code are the same (e.g. file name changes)</td></tr><tr><td>Difference</td><td>Remove samples that contain a hashtag (to avoid references to issues)</td></tr><tr><td>Extension</td><td>Remove samples where the filename of the code after has an atypical extension for the programming language (e.g. only keep &#x27;.py&#x27; for Python)</td></tr><tr><td>Filename</td><td>Remove samples where the filename is contained in the commit message (as we do not use the filename in finetuning)</td></tr><tr><td>Length</td><td>Only keep samples where the commit message has more than 10 and less than 1000 characters</td></tr><tr><td>Words</td><td>Only keep samples where the commit message can be split into more than 4 and less than 1000 space-separated words</td></tr><tr><td>Clean</td><td>Remove any appearances of &#x27;[skip ci]&#x27;, &#x27;[ci skip]&#x27;, sequences at the beginning or end that are in brackets, sequences at the beginning that end with&#x27;:&#x27; and strip whitespace at the beginning or end</td></tr><tr><td>Capitalized</td><td>Only keep samples where the message starts with an uppercase letter</td></tr><tr><td>Tokens</td><td>Only keep samples where the concatenation of the code before, a special token and the code after has at least 50 tokens and at most 768 tokens according to the StarCoder tokenizer</td></tr><tr><td>Instructions</td><td>Only keep samples where the lowercased commit message starts with any of the words in Table 7</td></tr><tr><td>Noise</td><td>Remove samples where the lowercased commit message contains any of &#x27;auto commit&#x27;, &#x27;update contributing&#x27;, &#x27;&lt;?xml&#x27;, &#x27;merge branch&#x27;, &#x27;merge pull request&#x27;, &#x27;signed-off-by&#x27;, &quot;fix that bug where things didn&#x27;t work but now they should&quot;, &quot;put the thingie in the thingie&quot;, &quot;add a beter commit message&quot;, &quot;code review&quot;, &quot;/codereview&quot;, &quot;work in progress&quot;, &quot;wip&quot;, &quot;https://&quot;, &quot;http://&quot;, &quot;l leetcode&quot;, &quot;cdpcp&quot;, &quot;i&quot;, &quot;i&#x27;ve&quot;, &quot;i&#x27;m&quot; or both &quot;thanks to&quot; and &quot;for&quot;</td></tr><tr><td>Regex</td><td>Remove samples where the lowercased commit message has a match for any of the regular expressions (?:v)?\d+\. \d+\. \d+(?=$|\S),^[a-f0-9]+(?:-[a-f0-9]+)*$, ([a-f0-9]{40}), issue\s*\d+,bug\s*\d+ or feature\s*\d+</td></tr><tr><td>Downsample</td><td>With 90% probability remove samples where the commit message starts with &quot;Bump&quot;, &quot;Set version&quot; or &quot;Update version&quot;</td></tr></table>

Table 6: COMMITPACKFT filters applied to COMMITPACK. With the commit message we refer to the commit message subject only, not the body.

"abort', 'accelerate', 'access', 'accumulate', 'add', 'address', 'adjust', 'advance', 'align', 'allot', 'allow', 'amplify', 'annotate', 'append', 'apply', 'archive', 'arrange', 'attach', 'augment', 'automate', 'backup', 'boost', 'break', 'bring', 'brush up', 'build', 'bump', 'call', 'change', 'check', 'choose', 'clarify', 'clean', 'clear', 'clone', 'comment', 'complete', 'compress', 'concatenate', 'configure', 'connect', 'consolidate', 'convert', 'copy', 'correct', 'cover', 'create', 'customize', 'cut', 'deal with', 'debug', 'decipher', 'declare', 'decommission', 'decomplexify', 'decompress', 'decrease', 'decrypt', 'define', 'delete', 'deploy', 'designate', 'destroy', 'detach', 'determine', 'develop', 'diminish', 'disable', 'discard', 'disentangle', 'dismantle', 'divide', 'document', 'downgrade', 'drop', 'duplicate', 'edit', 'embed', 'emphasize', 'enable', 'encrypt', 'enforce', 'enhance', 'enlarge', 'enumerate', 'eradicate', 'escalate', 'establish', 'exclude', 'exit', 'expand', 'expedite', 'expire', 'extend', 'facilitate', 'fix', 'format', 'gather', 'generalize', 'halt', 'handle', 'hasten', 'hide', 'implement', 'improve', 'include', 'increase', 'increment', 'indent', 'index', 'inflate', 'initialize', 'insert', 'install', 'integrate', 'interpolate', 'interrupt', 'introduce', 'isolate', 'join', 'kill', 'leverage', 'load', 'magnify', 'maintain', 'make', 'manage', 'mark', 'mask', 'mend', 'merge', 'migrate', 'modify', 'monitor', 'move', 'multiply', 'normalize', 'optimize', 'orchestrate', 'order', 'package', 'paraphrase', 'paste', 'patch', 'plug', 'prepare', 'prepend', 'print', 'provision', 'purge', 'put', 'quit', 'raise', 'read', 'reannotate', 'rearrange', 'rebase', 'reboot', 'rebuild', 'recommend', 'recompile', 'reconfigure', 'reconnect', 'rectify', 'redact', 'redefine', 'reduce', 'refactor', 'reformat', 'refresh', 'reimplement', 'reinforce', 'relocate', 'remove', 'rename', 'reorder', 'reorganize', 'repackage', 'repair', 'rephrase', 'replace', 'reposition', 'reschedule', 'reset', 'reshape', 'resolve', 'restructure', 'return', 'revert', 'revise', 'revoke', 'reword', 'rework', 'rewrite', 'rollback', 'save', 'scale', 'scrub', 'secure', 'select', 'send', 'set', 'settle', 'simplify', 'solve', 'sort', "speed up", 'split', "stabilize", "standardize", "stipulate", 'stop', "store", "streamline", "strengthen", "structure", "substitute", "subtract", "support", "swap", "switch", "synchronize", "tackle", "tag", "terminate", "test", "throw", "tidy", "transform", "transpose", "trim", "troubleshoot", "truncate", "tweak", "unblock", "uncover", "undo", "unify", "uninstall", "unplug", "unpublish", "unravel", "unstage", "unsync", "untangle", "unwind", "update", "upgrade", "use", "validate", "verify", "watch", "watermark", "whitelist", "withdraw", "work", "write"

Table 7: Commit message starting words allowed in COMMITPACKFT.

Please categorize the following commit message, which may fall into more than one category.

\### Category

Bug fixes, New features, Refactoring/code cleanup, Documentation, Testing, User interface, Dependencies, Configuration, Build system/tooling, Performance improvements, Formatting/Linting, Security, Technical debt repayment, Release management, Accessibility, Deprecation, Logging/Instrumentation, Internationalization

\### Commit Message

Add the blacklist checking to the bulk

\### Classification

Bug fixes, New features

\### Commit Message

{COMMIT\_MESSAGE}

\### Classification

Figure 5: GPT-4 1-shot prompt for classifying commits in COMMITPACKFT.

xP3x We use a subset of xP3x (Muennighoff et al., 2022b) focusing on code datasets consisting of APPS (Hendrycks et al., 2021), CodeContests (Li et al., 2022b), Jupyter Code Pairs, $^{6}$ MBPP (Austin et al., 2021), XLCoST (Zhu et al., 2022), Code Complex (Jeon et al., 2022), Docstring Corpus (Barone & Sennrich, 2017), Great Code (Hellendoorn et al., 2019) and State Changes. $^{7}$

OASST We reuse a filtered variant of OASST (Köpf et al., 2023) from prior work (Dettmers et al., 2023) and apply additional filters to remove responses that refuse to comply with the user request. To compute the programming languages and code fraction for OASST depicted in Table 1, we count all responses containing e.g.\`\`\`python or\`\`\`py for the Python programming language. There are code samples that are not enclosed in backticks or do not specify the language, thus we are likely underestimating the actual fraction of code data for OASST in Table 1.

# E COMPARING DATA BEFORE AND AFTER FILTERING

In Table 8 we compare word statistics prior to and after filtering COMMITPACK to create COMMITPACKFT. The mean commit subject and message length increases suggesting that messages are more informative in COMMITPACKFT. The code lengths decrease significantly as we limit the number of allowed tokens in the filters in Table 6. This is intended, as we would like to maximize the amount of training signal per token. The code before and after the commit are usually largely the same. By filtering for short samples, we ensure that there are more differences between the code before and after, thus making the model learn faster. The percentage of code changed between pre- and post-commit is $77.6 / 59.1 = 1.31$ (a $31\%$ increase) as opposed to $3269.8 / 3269.9 = 1.007$ (a $0.7\%$ increase). Thus, the filtered data carries significantly more signal per token with fewer repetitions of the code prior to the commit.

<table><tr><td>Metric</td><td>Before Filter</td><td>After Filter</td><td>Difference</td></tr><tr><td>Subject Length (words)</td><td>5.7±0.02</td><td>6.9±0.01</td><td>+1.28</td></tr><tr><td>Message Length (words)</td><td>8.7±0.06</td><td>9.9±0.05</td><td>+1.34</td></tr><tr><td>Pre-Commit Code Length (words)</td><td>3269.9±298.8</td><td>59.1±0.19</td><td>-3210.9</td></tr><tr><td>Post-Commit Code Length (words)</td><td>3269.8±299.5</td><td>77.6±0.23</td><td>-3214.2</td></tr></table>

Table 8: The effect of data filters on subject, message, and code lengths. We compare differences in word statistics of COMMITPACK and COMMITPACKFT.

# F COMPARING COMMITPACK AND THE STACK

In Table 9 we provide statistics on repositories and usernames of COMMITPACK and The Stack (Kocetkov et al., 2022). COMMITPACK contains a total of 1,934,255 repositories. Around half (49.3%) of them are also in The Stack. However, The Stack only provides the raw code files of these repositories from some fixed point in time. COMMITPACK contains the changes made to the code files in the form of commits. Thus, the same code file may appear multiple times in COMMITPACK for each change that was made to it. Therefore, The Stack only contains 3 terabytes of data, while COMMITPACK contains close to 4.

<table><tr><td>Statistic (↓)</td><td>COMMITPACK</td><td>The Stack 1.2</td><td>Shared</td><td>Shared (%)</td></tr><tr><td>Repositories</td><td>1,934,255</td><td>18,712,378</td><td>954,135</td><td>49.3%</td></tr><tr><td>Usernames</td><td>825,885</td><td>6,434,196</td><td>663,050</td><td>80.3%</td></tr></table>

Table 9: Overlap in repositories and usernames of COMMITPACK and The Stack.

# G PRETRAINING ON COMMITPACK

Due to the scale of COMMITPACK, it is also adequate as a large-scale pretraining dataset. We have included parts of COMMITPACK during the pretraining of StarCoder (Li et al., 2023b) in the format of <commit\_before>code\_before<commit\_msg>message<commit\_after>code\_after. We also pretrain a new model, named SANTACODERPACK, with the same architecture as SantaCoder (Allal et al., 2023) on COMMITPACK using this format. We filter COMMITPACK for our six evaluation languages and samples that fit within 8192 tokens leaving us a total of 35B tokens. Following prior work (Muennighoff et al., 2023), we train on this data repeated close to 4 times for a total of 131B tokens taking 14 days. Detailed hyperparameters are in Appendix P.

In Table 10, we benchmark StarCoder and SANTACODERPACK on HUMANEVALFIX using the above-detailed commit format. We find that the commit format leads to very strong performance for StarCoder often surpassing the instruction tuned OCTOCODER from Table 2. However, this pretraining format is not suitable for HUMANEVALEXPLAIN limiting its universality. For SANTACODERPACK, we find performance comparable to SantaCoder, including checkpoints at 131B and 236B tokens. SANTACODERPACK performs slightly worse on Python than SantaCoder. We hypothesize that this discrepancy is due to a multilingual tax, as SANTACODERPACK needs to accommodate three additional coding languages (Go, C++ and Rust). SantaCoder has thus more capacity allocated to Python, JavaScript, and Java.

SANTACODERPACK may also be bottlenecked by its small model size of 1.1B parameters. More research into what exactly happens during pretraining (Xia et al., 2022; Biderman et al., 2023a) and how to unify pretraining and instruction tuning are needed. Prior work has also found that including raw code data during pretraining benefits some natural language tasks (Muennighoff et al., 2023; Soldaini et al., 2024; Groeneveld et al., 2024). Future work may consider the effects of including code commit data on natural language tasks.

<table><tr><td>Model (↓)</td><td>Python</td><td>JavaScript</td><td>Java</td><td>Go</td><td>C++</td><td>Rust</td><td>Avg.</td></tr><tr><td>SantaCoder (131B tokens) Instruct Format</td><td>6.5</td><td>4.2</td><td>2.9</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SantaCoder (236B tokens) Instruct Format</td><td>7.1</td><td>4.2</td><td>1.8</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SANTACODERPACK (131B tokens) Commit Format</td><td>3.2</td><td>4.9</td><td>1.8</td><td>3.6</td><td>4.2</td><td>1.7</td><td>3.3</td></tr><tr><td>StarCoder Commit Format</td><td>32.7</td><td>33.6</td><td>33.0</td><td>31.9</td><td>31.6</td><td>20.2</td><td>30.5</td></tr></table>

Table 10: Zero-shot pass@1 (%) performance on HUMANEVALFIX of pretraining experiments.

# H HUMAN EVAL PACK STATISTICS

Table 11 displays statistics of HUMANEVALPACK. Docstrings are largely the same across languages leading to similar statistics except for Rust. As Rust is already a very verbose language as seen by its maximum solution length in Table 11, we do not include examples of how to call the function at the end of its docstrings (see Python docstrings with examples in e.g. Figure 11). Rust also has type annotations for every function so providing these examples is not as needed as it is for e.g. JavaScript which lacks type annotations.

<table><tr><td>Statistic (↓)</td><td>Python</td><td>JavaScript</td><td>Java</td><td>Go</td><td>C++</td><td>Rust</td></tr><tr><td>Docstring Avg. Length (chars)</td><td>354</td><td>352</td><td>357</td><td>365</td><td>352</td><td>231</td></tr><tr><td>Docstring Min. Length (chars)</td><td>56</td><td>56</td><td>56</td><td>56</td><td>56</td><td>23</td></tr><tr><td>Docstring Max. Length (chars)</td><td>1207</td><td>1207</td><td>1211</td><td>1207</td><td>1207</td><td>1067</td></tr><tr><td>Solution Avg. Length (chars)</td><td>181</td><td>234</td><td>259</td><td>354</td><td>295</td><td>339</td></tr><tr><td>Solution Min. Length (chars)</td><td>16</td><td>19</td><td>18</td><td>29</td><td>17</td><td>17</td></tr><tr><td>Solution Max. Length (chars)</td><td>864</td><td>1325</td><td>1399</td><td>1333</td><td>1144</td><td>2157</td></tr></table>

Table 11: Statistics of HUMANEVALPACK computed across the 164 samples for each language.

# I FULL INSTRUCTION DATA ABLATIONS

We provide tabular results of the ablations from Figure 4 in Table 12. We try some additional mixtures, however, none of them perform better than COMMITPACKFT + OASST. We experiment with changing the formatting to be <commit\_before>old code<commit\_msg>message<commit\_after>new code for COMMITPACKFT and <commit\_before><commit\_msg>input<commit\_after>output for OASST referred to as the "Formatting" ablation. We hypothesized that aligning the formatting during instruction tuning with the commit format that we used during pretraining (Appendix G) would improve performance. While it seems to improve performance for HUMANEVALFIX compared to our default formatting (see Figure 18), it reduces performance on the other tasks leading to a worse average score of 35.3 in Table 12. "Target Loss" refers to an ablation where we mask loss for inputs as is commonly done during instruction tuning (Muennighoff et al., 2022b). While this leads to the best performance on HUMANEVALSYNTHESIZE, its average performance is worse compared to COMMITPACKFT + OASST, where the loss is computed over the full sequence. We also perform an ablation where we manually select 1178 high-quality samples (725 from OASST and 89, 61, 86, 72, 70 and 75 from COMMITPACKFT for Python, JavaScript, Java, Go, C++ and Rust, respectively). However, this manual selection did not outperform random selection for OCTOCODER. It performed better for OCTOGEEX, however, hence we used it for OCTOGEEX. We hypothesize that our models could achieve significantly better performance by further improving the quality of the instruction data beyond. This may necessitate very careful human selection of samples and manual editing of the data to ensure a uniform style in the outputs. We leave such explorations to future work.

<table><tr><td rowspan="2">Instruction Tuning Dataset (↓)</td><td colspan="3">HUMANEVALPACK Python</td><td rowspan="2">Average</td></tr><tr><td>Fix</td><td>Explain</td><td>Synthesize</td></tr><tr><td>Without instruction tuning</td><td>8.7</td><td>0.0</td><td>33.6</td><td>14.1</td></tr><tr><td>Self-Instruct (SI)</td><td>23.6</td><td>0.6</td><td>43.0</td><td>22.2</td></tr><tr><td>OASST</td><td>23.1</td><td>34.5</td><td>46.4</td><td>34.7</td></tr><tr><td>SI + OASST</td><td>24.9</td><td>28.7</td><td>46.2</td><td>33.3</td></tr><tr><td>xP3x + OASST</td><td>28.4</td><td>28.4</td><td>45.0</td><td>33.9</td></tr><tr><td>COMMITPACKFT + OASST</td><td>30.4</td><td>35.1</td><td>46.2</td><td>37.2</td></tr><tr><td>COMMITPACKFT + OASST (Formatting)</td><td>31.1</td><td>28.9</td><td>45.8</td><td>35.3</td></tr><tr><td>COMMITPACKFT + OASST (Target loss)</td><td>29.8</td><td>31.2</td><td>47.8</td><td>36.3</td></tr><tr><td>COMMITPACKFT + OASST (Manual)</td><td>27.2</td><td>29.6</td><td>45.8</td><td>34.2</td></tr><tr><td>COMMITPACKFT + xP3x + OASST</td><td>30.9</td><td>29.5</td><td>45.9</td><td>35.4</td></tr><tr><td>COMMITPACKFT + SI + xP3x + OASST</td><td>31.4</td><td>33.8</td><td>46.0</td><td>37.1</td></tr></table>

Table 12: Zero-shot pass@1 (%) performance across the Python split of HUMANEVALPACK for StarCoder instruction tuning data ablations.

# J LINE DIFF FORMAT FOR FIXING CODE

We finetune SantaCoder to experiment with different formatting strategies for fixing bugs comparing full code generation and code diff generation. When fixing a code bug, usually only a small part of the code needs to change. Only generating the code diff corresponding to the necessary change can make inference significantly more efficient by avoiding repeated characters in the output generation. We finetune SantaCoder on the Python, Java and JavaScript subset of COMMITPACKFT. We exclude other languages as SantaCoder has only been pretrained on these three languages (Allal et al., 2023).

Commit Format For full code generation, we reuse the format that we employed for commits in StarCoder pretraining from Appendix G: <commit\_before>code\_before<commit\_msg>message<commit\_after>code\_after. However, SantaCoder has not seen this format during pretraining and does not have special tokens like StarCoder for the delimiters. Thus, for SantaCoder e.g. <commit\_before> is tokenized as ['<', 'commit', '\_', 'before', '>'].

Unified diff format For code diff generation, a simple solution is using the unified diff format, $^{8}$ which is a standard way to display changes between code files in a compact and readable format (Lehman et al., 2022; Jung, 2021; Xu et al., 2022b; Monperrus et al., 2021). We depict an example of this format in Figure 6. However, the unified diff format still requires the model to output several unchanged lines below and after the actual modification. Thus, its efficiency gains are limited and there is still unnecessary duplication of the input.

```python
from typing import List    from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(numbers)
    :
    if idx != idx2:
    distance = elem - elem2
    if distance < threshold:
    return True    return True

return False    return False

@@ -4,7 +4,7 @@
    for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(numbers):
    if idx != idx2:
    -    distance = elem - elem2
+    distance = abs(elem - elem2)
    if distance < threshold:
    return True 
```  
Figure 6: The first problem from the HUMANEVALFIX Python split and the necessary change to fix the bug in unified diff format. Top: Code with and without the bug from Figure 11. Bottom: Necessary change to fix the bug in unified diff format.

```txt
- 7 distance = elem - elem2
+ 7 distance = abs(elem - elem2) 
```  
Figure 7: The line diff format for the problem from Figure 6.

Line diff format To address the inefficiencies of the unified diff format, we propose the line diff format for representing code differences. There are two requirements for our format: (1) The diff can be unambiguously applied to the code before the commit to generate the code after the commit, and (2) the code diff should be as short as possible to maximize efficiency by avoiding the inclusion of unchanged code. In Figure 7, we show how our format addresses these. The line diff format keeps track of each change sequentially line-by-line to ensure the code can be correctly modified. By focusing only on the lines that change, we reduce the number of characters in the diff by 70% compared to the unified diff representation in Figure 6.

Both the unified diff format and our line diff format require the model to predict line numbers. This is very challenging when training on raw code as models need to count and keep track of line numbers. To simplify line number prediction, we automatically add line numbers to the raw code in the finetuning dataset for the line diff format. This allows the model to simply copy the line number into the output simplifying the diff generation. However, it diminishes efficiency slightly by adding additional input tokens that the model needs to process.

As summarized in Table 13, finetuning SantaCoder using the line diff format significantly improves performance compared to prior finetuning on HUMANEVALFIX across all languages. It also outperforms finetuning using the commit format, which only provides gains on JavaScript and Java compared to no finetuning. However, finetuning on the diff format may converge slower than the commit format as the diff format significantly differs from the raw code seen during pretraining.

Figures 8, 9, 10 show line diff generations of our model. A limitation of our current line diff implementation is that it does not handle code insertion well. The inserted lines may change the line numbers of all following lines, which can result in problems when applying the diff. Further, the diff format is not useful for HUMANEVALEXPLAIN and HUMANEVALSYNTHESIZE. Future work could consider training models that can both be instructed to use the line diff format, such as for HUMANEVALFIX, but also explain or synthesize code without producing a diff.

<table><tr><td>Model</td><td>Python</td><td>JavaScript</td><td>Java</td></tr><tr><td>SantaCoder</td><td>7.1</td><td>4.2</td><td>1.8</td></tr><tr><td>SantaCoder + Commit format finetuning</td><td>3.8</td><td>5.3</td><td>9.2</td></tr><tr><td>SantaCoder + Line diff format finetuning</td><td>9.9</td><td>9.7</td><td>10.0</td></tr></table>

Table 13: Zero-shot pass@1 (%) performance on HUMANEVALFIX of SantaCoder formatting experiments.

```javascript
- 3 let depth = 0, max_depth = 0;
+ 3 let depth = 0, max_depth = 1;
- 12 return max_depth;
+ 12 return max_depth - 1;
- 14 return paren_string.split('_')
- 15 .filter(x => x != ''')
- 16 .map(x => parseParenGroup(x));
- 17 }
+ 14 let paren_list = paren_string.split('_');
+ 15 let nested parens = paren_list.map(x => parseParenGroup(x));
+ 16 return nested parens.reduce((prev, curr) => {
+ 17 if (prev == 0) {
+ 18 return curr;
+ 19 } else {
+ 20 return curr - 1;
+ 21 }
+ 22 });
+ 23 } 
```  
Figure 8: A line diff generation of our model on a JavaScript HUMANEVALFIX problem.

```javascript
- 18 if (current_depth < 0) {
+ 18 if (current_depth < 0 && current_string.length() > 0) { 
```  
Figure 9: A line diff generation of our model on a Java HUMANEVALFIX problem.

```python
- 2 for i, l1 in enumerate(l):
- 3 for j in range(i, len(l)):
+ 2 for i in range(0, len(l)):
+ 3 for j in range(i+1, len(l)): 
```  
Figure 10: A line diff generation of our model on a Python HUMANEVALFIX problem.

# K RESULTS ON HUMAN EVALFIXDOCS

The default version of HUMANEVALFIX does not include docstrings, but only provides the unit tests to the model alongside the buggy function. An alternative is providing docstrings as the source of ground truth for the model to fix the buggy function. Solving from docstrings is generally easier for models than from tests, as models can also solve it via pure code synthesis without looking at the buggy function at all. We provide results of some models on this variant in Table 14. For StarCoder, we distinguish two prompting formats: An instruction to fix bugs like in Figure 3 or the commit format it has seen during pretraining (Appendix G). OCTOCODER performs strongly on this variant. However, directly using StarCoder with the commit format from pretraining (Appendix G) is slightly better. This is in line with the commit format from pretraining also performing slightly better on HUMANEVALFIX in Table 10 compared to OCTOCODER in Table 2. Diff Codegen 2B (Bradley et al., 2023) performs poorly as its predicted code diffs are often irrelevant to the actual bug, see Figure 39.

<table><tr><td>Model</td><td>Python</td><td>JavaScript</td><td>Java</td><td>Go</td><td>C++</td><td>Rust</td><td>Avg.</td></tr><tr><td colspan="8">Non-permissive models</td></tr><tr><td>GPT-4</td><td>88.4</td><td>80.5</td><td>82.9</td><td>81.1</td><td>82.3</td><td>68.9</td><td>80.7</td></tr><tr><td colspan="8">Permissive Models</td></tr><tr><td>Diff Codegen 2B</td><td>0.0</td><td>0.1</td><td>0.0</td><td>0.3</td><td>0.0</td><td>0.2</td><td>0.1</td></tr><tr><td>StarCoder Commit Format</td><td>58.8</td><td>49.2</td><td>43.9</td><td>55.2</td><td>51.5</td><td>41.8</td><td>50.1</td></tr><tr><td>StarCoder Instruct Format</td><td>41.7</td><td>30.7</td><td>44.3</td><td>34.5</td><td>28.7</td><td>14.0</td><td>26.5</td></tr><tr><td>OCTOCODER</td><td>53.8</td><td>48.1</td><td>54.3</td><td>54.9</td><td>49.2</td><td>32.1</td><td>48.7</td></tr></table>

Table 14: Zero-shot pass@1 (%) performance on HUMANEVALFIXDOCS.

# L HUMANEvalFIX BUG TYPES

Table 15 contains an overview of bugs that were manually added by one of the authors to HumanEval solutions for the construction of HUMANEVALFIX. Figures 11-16 contain an example of each type from the Python split. The bug type for each problem is the same across all programming languages in HUMANEVALFIX, but for a few samples it affects a different part of the solution due to the code solutions not being perfectly parallel across languages.

<table><tr><td>Bug type</td><td>Subtype</td><td>Explanation</td><td>Example</td><td>Count</td></tr><tr><td>Missing logic</td><td></td><td>Misses code needed to solve the problem</td><td>Figure 11</td><td>33</td></tr><tr><td>Excess logic</td><td></td><td>Contains excess code leading to mistakes</td><td>Figure 12</td><td>31</td></tr><tr><td rowspan="4">Wrong logic</td><td>Value misuse</td><td>An incorrect value is used</td><td>Figure 13</td><td>44</td></tr><tr><td>Operator misuse</td><td>An incorrect operator is used</td><td>Figure 14</td><td>25</td></tr><tr><td>Variable misuse</td><td>An incorrect variable is used</td><td>Figure 15</td><td>23</td></tr><tr><td>Function misuse</td><td>An incorrect function is used</td><td>Figure 16</td><td>8</td></tr><tr><td>Total</td><td></td><td></td><td></td><td>164</td></tr></table>

Table 15: HUMANEVALFIX bug types.

from typing import List   
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers,
    are any two numbers closer to
    each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0,
    3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0,
    4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(
    numbers):
    if idx != idx2:
    distance = abs(elem -
    elem2)
    if distance < threshold:
    return True

return False 
```

from typing import List   
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers,
    are any two numbers closer to
    each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0,
    3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0,
    4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(
    numbers):
    if idx != idx2:
    distance = elem - elem2
    if distance < threshold:
    return True

return False 
```

Figure 11: Missing logic bug example. The buggy code (right) misses the 'abs' statement.   
```python
def truncate_number(number: float) ->
    float:
    """ Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

Return the decimal part of the number.
>>> truncate_number(3.5)
0.5
"""
return number % 1.0 
```

def truncate\_number(number: float) -> float:   
```python
""" Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals
(leftover part always smaller than 1).

Return the decimal part of the number.
>>> truncate_number(3.5)
0.5

"""
return number % 1.0 + 1.0 
```  
Figure 12: Excess logic bug example. The buggy code (right) incorrectly adds 1 to the result.

from typing import List, Tuple   
```python
def sum_product(numbers: List[int]) ->
    Tuple[int, int]:
    """ For a given list of integers,
    return a tuple consisting of a
    sum and a product of all the
    integers in a list.
    Empty sum should be equal to 0 and
    empty product should be equal to
    1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
    sum_value = 0
    prod_value = 1

    for n in numbers:
    sum_value += n
    prod_value *= n
    return sum_value, prod_value 
```

from typing import List, Tuple   
```python
def sum_product(numbers: List[int]) ->
    Tuple[int, int]:
    """ For a given list of integers,
    return a tuple consisting of a
    sum and a product of all the
    integers in a list.
    Empty sum should be equal to 0 and
    empty product should be equal to
    1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
    sum_value = 0
    prod_value = 0

    for n in numbers:
    sum_value += n
    prod_value *= n
    return sum_value, prod_value 
```  
Figure 13: Value misuse bug example. The buggy code (right) incorrectly initializes the product to 0.

```python
from typing import List
def below_zero(operations: List[int]) ->
    bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with zero balance. Your task is to detect if at any point the balance of account falls below zero, and at that point function should return True. Otherwise it should return False.
>>> below_zero([1, 2, 3])
False
>>> below_zero([1, 2, -4, 5])
True
"""
balance = 0

for op in operations:
    balance += op
    if balance < 0:
    return True

return False

from typing import List
def below_zero(operations: List[int]) ->
    bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with zero balance. Your task is to detect if at any point the balance of account falls below zero, and at that point function should return True. Otherwise it should return False.
>>> below_zero([1, 2, 3])
False
>>> below_zero([1, 2, -4, 6])
True
"""
balance = 0

for op in operations:
    balance += op
    if balance == 0:
    return True

return False 
```  
Figure 14: Operator misuse bug example. The buggy code (right) incorrectly checks for equality with 0.

```python
from typing import List
def mean_absolute_deviation(numbers: List[ float]) -> float:
    """ For a given list of input numbers,
    calculate Mean Absolute Deviation
around the mean of this dataset.
Mean Absolute Deviation is the average absolute difference between each element and a centerpoint (mean in this case):
MAD = average | x - x_mean |
>>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
1.0
"""
mean = sum(numbers) / len(numbers)
return sum(abs(x - mean) for x in numbers) / len(numbers)

from typing import List
def mean_absolute_deviation(numbers: List[ float]) -> float:
    """ For a given list of input numbers,
    calculate Mean Absolute Deviation
around the mean of this dataset.
Mean Absolute Deviation is the average absolute difference between each element and a centerpoint (mean in this case):
MAD = average | x - x_mean |
>>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])

1.0
""" mean = sum(numbers) / len(numbers)
return sum(abs(x - mean) for x in numbers) / mean 
```  
Figure 15: Variable misuse bug example. The buggy code (right) incorrectly divides by the mean.

```python
def flip_case(string: str) -> str:
    """ For a given string, flip lowercase characters to uppercase and uppercase to lowercase.
>>> flip_case('Hello')
'hELLO'
"""
return string.swapcase()
def flip_case(string: str) -> str:
    """ For a given string, flip lowercase characters to uppercase and uppercase to lowercase.
>>> flip_case('Hello')
'hELLO'
"""
return string.lower() 
```  
Figure 16: Function misuse bug example. The buggy code (right) incorrectly uses the 'lower()' function.

# M PERFORMANCE BREAKDOWN BY HUMAN EVALFIX BUG TYPE

All bugs in HUMANEVALFIX are categorized into bug types as described in Appendix L. In Table 16, we break down the HUMANEVALFIX performance of select models from Table 2 by bug type. We find that models struggle most with bugs that require removing excess logic (e.g. Figure 12). WizardCoder is only able to solve 11% of excess logic bugs while solving about four times more bugs that relate to value misuse. The performance of OCTOGEEX and OCTOCODER is more stable than WizardCoder across the different bug types, possibly due to the diversity of COMMITPACKFT as displayed in Figure 2. GPT-4 performs best across all bug types.

<table><tr><td>Bug type</td><td>Subtype</td><td>OCTOGEEX</td><td>OCTOCODER</td><td>WizardCoder</td><td>GPT-4</td></tr><tr><td>Missing logic</td><td></td><td>24.2</td><td>24.4</td><td>31.2</td><td>45.5</td></tr><tr><td>Excess logic</td><td></td><td>16.3</td><td>16.9</td><td>11.0</td><td>38.7</td></tr><tr><td rowspan="4">Wrong logic</td><td>Value misuse</td><td>33.2</td><td>34.7</td><td>45.1</td><td>50.0</td></tr><tr><td>Operator misuse</td><td>32.8</td><td>42.0</td><td>34.4</td><td>56.0</td></tr><tr><td>Variable misuse</td><td>35.7</td><td>33.7</td><td>30.4</td><td>43.5</td></tr><tr><td>Function misuse</td><td>25.0</td><td>37.5</td><td>37.5</td><td>50.0</td></tr><tr><td colspan="2">Overall</td><td>28.1</td><td>30.4</td><td>31.8</td><td>47.0</td></tr></table>

Table 16: Breakdown of HUMANEVALFIX Python pass@1 (%) performance by bug type for select models. Statistics for each bug type are in Table 15.

# N HUMAN EVALE EXPLAIN WITH FILL-IN-THE-MIDDLE

In Table 2, all models are prompted with the same instruction to provide an explanation (but using slightly different formats, see Appendix Q). For StarCoder, we can alternatively prompt it with the Fill-in-the-Middle (FIM) technique (Bavarian et al., 2022), which it already knows from pretraining (Li et al., 2023b). To do so, we provide it with a prompt as shown in Figure 17 to generate the docstring. While a docstring is not necessarily an explanation, it can be similar. The results in Table 17 show that it performs significantly better with this prompting strategy than in Table 2. However, it still falls short of OCTOCODER and other models likely due to the imperfect approximation of an explanation.

```python
< fim_prefix> from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ < fim_suffix>
    """
    for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(numbers):
    if idx != idx2:
    distance = abs(elem - elem2)
    if distance < threshold:
    return True

return False < fim_middle> 
```

Figure 17: FIM prompt example for StarCoder. 

<table><tr><td>Model</td><td>Python</td><td>JavaScript</td><td>Java</td><td>Go</td><td>C++</td><td>Rust</td><td>Avg.</td></tr><tr><td>StarCoder FIM</td><td>19.4</td><td>17.6</td><td>16.3</td><td>11.8</td><td>17.9</td><td>16.7</td><td>16.6</td></tr></table>

Table 17: Performance of StarCoder on HUMANEVALEXPLAIN with FIM.

# O HUMANEVALEXPLAIN BLEU AND METEOR COMPARISON

By default, we use pass@k to evaluate HUMANEVALEXPLAIN ( $\S3$ ). In Table 18 we compare it with BLEU (Papineni et al., 2002) and METEOR (Banerjee & Lavie, 2005). While our pass@k formulation does not require a ground truth explanation, BLEU and METEOR do. This can be a limiting factor. For this evaluation, we use the function docstrings as the ground-truth explanation to compute the BLEU and METEOR scores. We compute BLEU and METEOR for each of the n = 20 generations (3) and select the highest score. The scores are then averaged across the 164 samples for each language. Rust scores are the highest which is likely due to Rust docstrings containing no example function calls (Appendix H).

<table><tr><td>Metric (↓)</td><td>Python</td><td>JavaScript</td><td>Java</td><td>Go</td><td>C++</td><td>Rust</td><td>Avg.</td></tr><tr><td>pass@1</td><td>35.1</td><td>24.5</td><td>27.3</td><td>21.1</td><td>24.1</td><td>14.8</td><td>24.5</td></tr><tr><td>BLEU-1</td><td>7.1</td><td>7.0</td><td>6.3</td><td>6.1</td><td>6.4</td><td>8.1</td><td>6.8</td></tr><tr><td>BLEU-2/3/4</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>METEOR</td><td>7.8</td><td>7.6</td><td>7.2</td><td>7.4</td><td>7.0</td><td>9.4</td><td>7.7</td></tr></table>

Table 18: Comparison of different metrics for HUMANEVALEXPLAIN on explanations by OCTOCODER. Pass@1 is computed with respect to a generated solution given the explanation (§3) while BLEU and METEOR are computed by comparing the explanation with the docstring.

# P HYPERPARAMETERS

StarCoder finetuning (OCTOCODER) For all experiments finetuning StarCoder, we use a learning rate of 5e-4 with a cosine schedule and linear warmup. We use a batch size of 32 and train for up to one epoch, as we did not observe benefits from more steps. OCTOCODER was trained for 35 steps with a sequence length of 2048 and packing corresponding to 2.2 million total finetuning tokens. We use LoRA (Hu et al., 2021) as we did not observe a significant difference from full finetuning. Follow-up work has further investigated this choice (Zhuo et al., 2024).

CodeGeeX finetuning (OCTOGEEX) To create OCTOGEEX, we finetune CodeGeeX2 for 35 steps with a batch size of 48 and a learning rate of 5e-5 largely following the OCTOCODER setup.

SantaCoder finetuning For all experiments finetuning SantaCoder, we use a learning rate of 5e-5 with a cosine schedule and linear warmup. We finetune SantaCoder using a batch size of 64 for up to 200,000 steps.

SantaCoder pretraining (SANTACODERPACK) We follow the setup from Allal et al. (2023) to pretrain on COMMITPACK except for using a sequence length of 8192 and the StarCoder tokenizer, which has special tokens for the commit format delimiters (see Appendix G). SANTACODERPACK utilizes Multi Query Attention (MQA) (Shazeer, 2019) but removes Fill-in-the-Middle (FIM) (Bavarian et al., 2022). We conducted pretraining on 32 A100 GPUs, totaling 250k training steps, with a global batch size of 64. Other hyperparameter settings follow SantaCoder, including using Adam with $\beta_{1} = 0.9$ , $\beta_{2} = 0.95$ , $\epsilon = 10^{-8}$ , and a weight decay of 0.1. The learning rate is set to $2 \times 10^{-4}$ and follows a cosine decay after warming up for $2\%$ of the training steps.

# Q PROMPTS

The prompting format can significantly impact performance. In the spirit of true few-shot learning (Perez et al., 2021) we do not optimize prompts and go with the format provided by the respective model authors or the most intuitive format if none is provided. For each task, we define an instruction, an optional context and an optional function start (Table 19). The function start is provided to make sure the model directly completes the function without having to search for the function in the model output. These three parts are then combined in slightly different ways for each model (Figures 18-24). We implement our evaluation using open-source frameworks (Ben Allal et al., 2022; Gao et al., 2021).

<table><tr><td colspan="2">HUMANEVALFIX</td></tr><tr><td>Instruction</td><td>Fix bugs in has_close_elements.</td></tr><tr><td>Context</td><td>from typing import Listdef has_close_elements(numbers: List[float], threshold: float) -&gt; bool:for idx, elem in enumerate(numbers):for idx2, elem2 in enumerate(numbers):if idx != idx2:distance = elem - elem2if distance &lt; threshold:return Truereturn False</td></tr><tr><td>Function start</td><td>from typing import Listdef has_close_elements(numbers: List[float], threshold: float) -&gt; bool:</td></tr><tr><td colspan="2">HUMANEVALEXPLAIN</td></tr><tr><td>Instruction (Describe)</td><td>Provide a concise natural language description of the code using at most 213 characters.</td></tr><tr><td>Context (Describe)</td><td>from typing import Listdef has_close_elements(numbers: List[float], threshold: float) -&gt; bool:for idx, elem in enumerate(numbers):for idx2, elem2 in enumerate(numbers):if idx != idx2:distance = abs(elem - elem2)if distance &lt; threshold:return Truereturn False</td></tr><tr><td>Instruction (Synthesize)</td><td>Write functional code in Python according to the description.</td></tr><tr><td>Context (Synthesize)</td><td>{Description generated by the model}</td></tr><tr><td>Function start (Synthesize)</td><td>from typing import Listdef has_close_elements(numbers: List[float], threshold: float) -&gt; bool:</td></tr><tr><td colspan="2">HUMANEVALSYNTHESIZE</td></tr><tr><td>Instruction</td><td>Write a Python function ‘has_close_elements(numbers: List[float], threshold: float) -&gt; bool‘ to solve the following problem:Check if in given list of numbers, are any two numbers closer to each other than given threshold.»&gt; has_close_elements([1.0, 2.0, 3.0], 0.5)False»&gt; has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)True</td></tr><tr><td>Function start</td><td>from typing import Listdef has_close_elements(numbers: List[float], threshold: float) -&gt; bool:&quot;&quot;&quot; Check if in given list of numbers, are any two numbers closer to each other than given threshold.»&gt; has_close_elements([1.0, 2.0, 3.0], 0.5)False»&gt; has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)True&quot;&quot;&quot;</td></tr></table>

Table 19: Instructions and function examples used. If no function start or no context is present, that part is not added to the prompt (and the preceding newline is also removed).

```txt
Question: {instruction}
{context}
Answer:
{function_start} 
```  
Figure 18: OCTOCODER and OCTOGEEX prompting format

```txt
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
{context}
### Response:
{function_start} 
```  
Figure 19: WizardCoder prompting format from their codebase. $^{9}$

```txt
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
{context}
### Response:{function_start} 
```  
Figure 20: InstructCodeT5+ decoder prompting format from their codebase. $^{10}$ The encoder receives the same prompt without the function start.

```txt
<|systeml>
<|endl>
<|userl>
{instruction}
{context}<|endl>
<|assistant|
{function_start} 
```  
Figure 21: StarChat- $\beta$ prompting format from their documentation. $^{11}$

```jsonl
{context}
{instruction}
{function_start} 
```  
Figure 22: Default prompting format (StarCoder, BLOOMZ). Used for Figure 3.

```txt
{context}
{instruction} 
```  
Figure 23: GPT-4 prompting format. Same as Figure 22 but excluding the function start, as we found the model would not continue from the provided function start, likely due to the chat mode.

```txt
{context}
{instruction}
Start your code with:
{func_start} 
```  
Figure 24: GPT-4 prompting format for HUMANEVALEXPLAIN (Synthesize). We include the function start for the synthesize part of HUMANEVALEXPLAIN in case the function name is not included in the code description generated by GPT-4.

```txt
<NME> {filename}
<BEF> {context}
<MSG> {instruction}
<DFF> 
```  
Figure 25: Prompting format for Diff Codegen 2B evaluated in Appendix K.

# R EXAMPLES

# R.1 OCTOCODER

Input:   
Question: Fix bugs in sum\_product.   
```c
#include<stdio.h>
#include<vector>
using namespace std;
#include<algorithm>
#include<math.h>
#include<stdlib.h>
vector<int> sum_product(vector<int> numbers) {
    int sum=0,product=0;
    for (int i=0;i(numbers.size();i++)
    {
    sum+=numbers[i];
    product*=numbers[i];
    }
    return {sum,product};
} 
```

Answer:   
```c
#include<stdio.h>
#include<vector>
using namespace std;
#include<algorithm>
#include<math.h>
#include<stdlib.h>
vector<int> sum_product(vector<int> numbers) { 
```

OctoCoder:   
```javascript
int sum=0,product=1;
for (int i=0;i<numbers.size();i++)
{
    sum+=numbers[i];
    product*=numbers[i];
}
return {sum,product}; 
```  
Figure 26: Correct solution from OCTOCODER to a HUMANEVALFIX C++ problem. The corresponding Python problem is depicted in Figure 13.

# Input:

Question: Write a Java function ‘public String decodeShift(String s)’ to solve the following problem: takes as input string encoded with encodeShift function. Returns decoded string.

Answer:   
```java
import java.util.*;
import java.lang.*;

class Solution {
    /**
    returns encoded string by shifting every character by 5 in the alphabet.
    */
    public String encodeShift(String s) {
    StringBuilder sb = new StringBuilder();
    for (char ch : s.toCharArray()) {
    sb.append((char) ('a' + ((ch + 5 - 'a') % 26)));
    }
    return sb.toString();
    }

    /**
    takes as input string encoded with encodeShift function. Returns decoded string.
    */
    public String decodeShift(String s) { 
```

# OctoCoder:

```txt
StringBuilder sb = new StringBuilder();
for (char ch : s.toCharArray()) {
    sb.append((char) ('a' + ((ch - 'a' - 5) % 26)));
}
return sb.toString();
} 
```  
Figure 27: Incorrect solution from OCTOCODER to a HUMANEVALSYNTHESIZE Java problem. The correct solution would be to add 21 to ch instead of subtracting 5 from it.

# Input:

Question: Provide a concise natural language description of the code using at most 671 characters.

```go
func FruitDistribution(s string, n int) int {
    split := strings.Split(s, " _
    for _, i := range split {
    atoi, err := strconv.Atoi(i)
    if err != nil {
    continue
    }
    n = n - atoi
    }
    return n
} 
```

Answer:

# OctoCoder:

The function FruitDistribution takes two arguments, a string s and an integer n. The function splits the string s into a slice of strings using the space character as the delimiter. The function iterates over the slice and converts each string to an integer using the strconv.Atoi function. If the conversion is successful, the integer is subtracted from n. The function returns the value of n after all conversions have been performed.

# Input (without previous input):

Question: Write functional code in Go according to the description.

The function FruitDistribution takes two arguments, a string s and an integer n. The function splits the string s into a slice of strings using the space character as the delimiter. The function iterates over the slice and converts each string to an integer using the strconv.Atoi function. If the conversion is successful, the integer is subtracted from n. The function returns the value of n after all conversions have been performed.

Answer:

```txt
func FruitDistribution(s string, n int) int { 
```

# OctoCoder:

```autohotkey
fruits := strings.Split(s, "_")
for _, fruit := range fruits {
    if num, err := strconv.Atoi(fruit); err == nil {
    n -= num
    }
}
return n 
```

Figure 28: Correct solution from OCTOCODER to a HUMANEVALEXPLAIN Go problem. Given its own description OCTOCODER writes a correct solution to the problem. Its solution is more compact and efficient than the original solution.

# R.2 GPT-4

User:   
```txt
Write a Go function `func RescaleToUnit(numbers []float64) []float64` to solve the following problem: Given a list of numbers (of at least two elements), apply a linear transform to that list, such that the smallest number will become 0 and the largest will become 1
>>> RescaleToUnit([1.0, 2.0, 3.0, 4.0, 5.0])
[0.0, 0.25, 0.5, 0.75, 1.0] 
```  
GPT-4:

Here is a Go function that solves the problem:   
```go
package main

import (
    "fmt"
    "math"
)

func RescaleToUnit(numbers []float64) []float64 {
    min := numbers[0]
    max := numbers[0]

    // Find the min and max values in the list
    for _, number := range numbers {
    if number < min {
    min = number
    }
    if number > max {
    max = number
    }
    }

    // Apply the linear transform to each number in the list
    for i, number := range numbers {
    numbers[i] = (number - min) / (max - min)
    }

    return numbers
}

func main() {
    fmt.Println(RescaleToUnit([]float64{1.0, 2.0, 3.0, 4.0, 5.0})
} 
```

This function first finds the minimum and maximum values in the list of numbers. Then it applies the linear transformation to each number in the list. The transformation is ‘(number - min) / (max - min)’, which scales the number such that ‘min’ becomes 0 and ‘max’ becomes 1. The transformed numbers are returned in a new list.

Figure 29: Incorrect solution from GPT-4 to a HUMANEVALSYNTHESIZE Go problem. GPT-4 includes an import that it does not use (math) which leads to a compilation error in Golang and thus the function does not pass the test cases.

User:

Provide a concise natural language description of the code using at most 246 characters.

```python
def truncate_number(number: float) -> float:
    return number % 1.0 
```

GPT-4:

The code defines a function that takes a float as input and returns its fractional part by using the modulus operator with 1.0.

User (without previous conversation history):

The code defines a function that takes a float as input and returns its fractional part by using the modulus operator with 1.0.

Write functional code in Python according to the description.

Start your code with:

```txt
def truncate_number(number: float) -> float: 
```

GPT-4:

```python
def truncate_number(number: float) -> float:
    return number % 1.0 
```  
Figure 30: Correct solution from GPT-4 to a HUMANEVALEXPLAIN Python problem. GPT-4 writes a description that is significantly shorter than the docstring and thus within the character limit. It then generates the perfect solution from its own description. The docstring for this problem can be seen in Figure 12.

User:   
```python
fn has_close_elements(numbers:Vec<f32>, threshold: f32) -> bool{
    for i in 0..numbers.len() {
    for j in 1..numbers.len() {

    if i != j {
    let distance:f32 = numbers[i] - numbers[j];

    if distance < threshold{
    return true;
    }
    }
    }
    }
    return false;
}

def check(has_close_elements):
    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False

check(has_close_elements) 
```

Fix bugs in has\_close\_elements.

# GPT-4:

The function has\_close\_elements is supposed to check if there are any two elements in the vector that are closer than a given threshold. However, there are a few issues with the current implementation:

1. The distance calculation is not correct. It should be the absolute difference between the two numbers, not just the difference.   
2. The second loop should start from $\mathrm{i} + 1$ , not 1. This is because we only need to compare each number with the numbers after it, not the numbers before it.

Here is the corrected function:

```txt
fn has_close_elements(numbers:Vec<f32>, threshold: f32) -> bool {
    for i in 0..numbers.len() {
    for j in i+1..numbers.len() {
    let distance:f32 = (numbers[i] - numbers[j]).abs();
    if distance < threshold{
    return true;
    }
    }
    }
    return false;
} 
```

This function will now correctly check if there are any two elements in the vector that are closer than the given threshold.

Figure 31: Correct solution from GPT-4 to a HUMANEVALFIX Rust problem. GPT-4 correctly fixes the bug by adding the necessary 'abs' statement to compute the absolute value. It additionally improves the efficiency of the function by starting the inner loop from 'i+1' instead of 1 like in the original solution. As some problems in HUMANEVALPACK are not perfectly efficient, evaluating the ability of models to improve the efficiency of the solutions is another interesting benchmarking direction (Yetistiren et al., 2022).

User:   
```python
def car_race_collision(n: int):
    return n**3

def check(car_race_collision):
    assert car_race_collision(2) == 4
    assert car_race_collision(3) == 9
    assert car_race_collision(4) == 16
    assert car_race_collision(8) == 64
    assert car_race_collision(10) == 100

check(car_race_collision) 
```

Fix bugs in car\_race\_collision.

# GPT-4:

The function car\_race\_collision seems to be working fine as it is. It takes an integer 'n' as input and returns the cube of 'n'. There don't seem to be any bugs in this function.

However, if the function is supposed to calculate something specific related to a car race collision and it's not doing that correctly, we would need more information about the expected behavior to fix it.

Figure 32: Incorrect solution from GPT-4 to a HUMANEVALFIX Python problem. GPT-4 does not catch that the test cases expect the function to square, not cube n. OCTOCODER correctly modifies the 3 to 2 in 6 out of n = 20 generations.

User:   
```python
def rounded_avg(n, m):
    if m < n:
    return -1
    summation = 0
    for i in range(n, m+1):
    summation += i
    return bin(round(summation/(m - n)))

def check(rounded_avg):
    # Check some simple cases
    assert rounded_avg(1, 5) == "0b11"
    assert rounded_avg(7, 13) == "0b1010"
    assert rounded_avg(964, 977) == "0b1111001010"
    assert rounded_avg(996, 997) == "0b1111100100"
    assert rounded_avg(560, 851) == "0b1011000010"
    assert rounded_avg(185, 546) == "0b101101110"
    assert rounded_avg(362, 496) == "0b110101101"
    assert rounded_avg(350, 902) == "0b1001110010"
    assert rounded_avg(197, 233) == "0b11010111"

    # Check some edge cases that are easy to work out by hand.
    assert rounded_avg(7, 5) == -1
    assert rounded_avg(5, 1) == -1
    assert rounded_avg(5, 5) == "0b101"

check(rounded_avg) 
```

Fix bugs in rounded\_avg.

# GPT-4:

The function rounded\_avg is supposed to calculate the average of all numbers between n and m (inclusive), round it, and then return the binary representation of the result.

There are several bugs in the function:

1. The function should return -1 if m < n, but it should also return -1 if m == n because in this case, the denominator of the average calculation would be 0, which would cause a division by zero error.   
2. The function calculates the average by summing all numbers between n and m and then dividing by $(m - n)$ . However, this is incorrect because the number of integers between n and m (inclusive) is $(m - n + 1)$ , not $(m - n)$ .   
3. The function uses the bin function to convert the result to binary. However, the bin function returns a string that starts with '0b' to indicate that it is a binary number. If the function is supposed to return the binary representation as an integer, it should remove this prefix.

Here is the corrected function:

```python
def rounded_avg(n, m):
    if m <= n:
    return -1
    summation = 0
    for i in range(n, m+1):
    summation += i
    average = round(summation / (m - n + 1))
    return int(bin(average)[2:]) 
```

This function now correctly calculates the average, rounds it, and returns the binary representation as an integer. If n is greater than or equal to m, it returns -1.

Figure 33: Incorrect solution from GPT-4 to a HUMANEVALFIX Python problem. The bug consists of a "+1" missing after "m-n". GPT-4 correctly fixes that, however, it introduces two new bugs: Retuning "-1" when m=n and removing the "0b", both of which evidently lead to the provided unit tests failing.

# R.3 WIZARDCODER

Input:

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: Fix bugs in bf.   
```python
def bf (planet1, planet2):
    planet_names = ("Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune")
    if planet1 not in planet_names or planet2 not in planet_names or planet1 == planet2:
    return ()
    planet1_index = planet_names.index (planet1)
    planet2_index = planet_names.index (planet2)
    if planet1_index < planet2_index:
    return (planet_names[planet1_index + 1: planet2_index])
    else:
    return (planet_names[planet2_index + 1: planet1_index])

def check (bf):
    # Check some simple cases
    assert bf("Jupiter", "Neptune") == ("Saturn", "Uranus"), "First_test_error:_" + str(len (bf("Jupiter", "Neptune"))
    assert bf("Earth", "Mercury") == ("Venus"), "Second_test_error:_" + str(bf("Earth", "Mercury"))
    assert bf("Mercury", "Uranus") == ("Venus", "Earth", "Mars", "Jupiter", "Saturn"), "Third_test_error:_" + str(bf("Mercury", "Uranus"))
    assert bf("Neptune", "Venus") == ("Earth", "Mars", "Jupiter", "Saturn", "Uranus"), "Fourth_test_error:_" + str(bf("Neptune", "Venus"))

    # Check some edge cases that are easy to work out by hand.
    assert bf("Earth", "Earth") == ()
    assert bf("Mars", "Earth") == ()
    assert bf("Jupiter", "Makemake") == ()
check (bf)

### Response:
def bf (planet1, planet2): 
```

WizardCoder:   
```python
planet_names = ("Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune")
if planet1 not in planet_names or planet2 not in planet_names or planet1 == planet2:
    return ()
planet1_index = planet_names.index(planet1)
planet2_index = planet_names.index(planet2)
if planet1_index < planet2_index:
    return (planet_names[planet1_index + 1: planet2_index])
else:
    return (planet_names[planet2_index + 1 : planet1_index]) 
```  
Figure 34: Incorrect solution from WizardCoder to a HUMANEVALFIX Python problem. The bug consists of "Jupiter" being misspelled as "Jupyter" in the function. WizardCoder returns the same function without making a single change leaving the bug. This is a particularly difficult bug that only GPT-4 and OCTOCODER (only for 1 out of the n=20 generations) are able to solve.

# R.4 BLOOMZ

Input:   
```javascript
const generateIntegers = (a, b) => {
    if (a > b) {
    let tmp = a;
    a = b;
    b = tmp;
    }
    let y = []
    for (let i = a; i <= b; i++) {
    if (i == 2 || i == 4 || i == 6 || i == 8) { y.push(i) }
    }
    return y
} 
```

Provide a concise natural language description of the code using at most 218 characters.

BLOOMZ:   
```txt
Generate a list of integers from a to b. 
```  
Input (without previous input):

```txt
Generate a list of integers from a to b.
Write functional code in JavaScript according to the description.
const generateIntegers = (a, b) => { 
```

BLOOMZ:   
```txt
let integers = [];
for (let i = a; i <= b; i++) {
    integers.push(i);
}
return integers;
} 
```  
Figure 35: Incorrect solution from BLOOMZ to a HUMANEVALEXPLAIN JavaScript problem. BLOOMZ misses in its description that the function generates only the even digits between a and b, in ascending order. Thus, its synthesized function based on its own description simply collects all integers between a and b. BLOOMZ likely left out this part due to its bias for short and concise answers. However, this bias backfires for this task.

# R.5 STARCODER

Input:   
```java
import java.util.*;
import java.lang.*;

class Solution {
    public Map<String, Integer> histogram(String test) {
    Map<String, Integer> dict1 = new HashMap<>();
    List<String> list1 = Arrays.asList(test.split("_"));
    int t = 0;
    for (String i : list1) {
    if (Collections.frequency(list1, i) > t && !i.isEmpty()) {
    t = Collections.frequency(list1, i);
    }
    }
    if (t > 0) {
    for (String i : list1) {
    if (Collections.frequency(list1, i) == t) {
    dict1.put(i, t);
    }
    }
    return dict1;
    }
} 
```

Provide a concise natural language description of the code using at most 423 characters.

StarCoder:   
```java
public class Solution {
    public Map<String, Integer> histogram(String test) {
    int t = 0;
    }
    if (t > 0) { 
```

Write functional code in Java according to the description.

```java
import java.util.*;
import java.lang.*;

class Solution {
    public Map<String, Integer> histogram(String test) { 
```  
StarCoder:

```txt
Map<String, Integer> map = new HashMap<String, Integer>();
int t = 0;
for (int i = 0; i < test.length(); i++) {
    if (test.charAt(i) == 't') {
    t++;
    }
}
map.put("t", t);
return map;
} 
```

Figure 36: Incorrect solution from StarCoder to a HUMANEVALEXPLAIN Java problem. Star-Coder is not capable of generating natural language descriptions and instead copies the solution. However, our post-processing removes substrings from the solution ( $\S3$ ) leading to a useless description. Given this description it generates an irrelevant function. As StarCoder cannot generate natural language explanations, we disqualify it from HUMANEVALEXPLAIN setting its score to 0 in Table 2.

# R.6 INSTRUCTCODET5+

Input:

Below is an instruction that describes a task. Write a response that appropriately completes the request.

\### Instruction: Fix bugs in has\_close\_elements.

from typing import List   
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(numbers):
    if idx != idx2:
    distance = elem - elem2
    if distance < threshold:
    return True

return False

def check (has_close_elements):
    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False

check (has_close_elements) 
```

\### Response:from typing import List

def has\_close\_elements(numbers: List[float], threshold: float) -> bool:   
InstructCodeT5+:   
```python
for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(numbers):
    if idx != idx2:
    distance = elem - elem2
    if distance < threshold:
    return True

return False 
```  
Figure 37: Incorrect solution from InstructCodeT5+ to a HUMANEVALFIX Python problem. InstructCodeT5+ simply repeated the function without adding the missing "abs" statement to turn the distance into an absolute value. This is one of the simplest bugs that most other models are able to fix. For InstructCodeT5+ the prompt starts directly after "Response:" without a newline (Figure 20).

# R.7 STARCHAT- $\beta$

Input:   
```cpp
#include<stdio.h>
#include<math.h>
#include<vector>
#include<string>
#include<algorithm>
using namespace std;
#include<stdlib.h>
int specialFilter(vector<int> nums) {
    int num=0;
    for (int i=0;i<nums.size();i++)
    if (nums[i]>10)
    {
    string w=to_string(nums[i]);
    if (w[0]%2==1 and w[w.length()-1]%2==1) num+=1;
    }
    return num;
} 
```

Provide a concise natural language description of the code using at most 296 characters.

StarChat-β:

This code is a solution to the "Special Array" problem on LeetCode. It uses C++ and includes several standard libraries, including vector, string, and algorithm.
The code starts by declaring some variables and importing the necessary libraries. It then defines a specialFilter function that takes

Input (without previous input):

This code is a solution to the "Special Array" problem on LeetCode. It uses C++ and includes several standard libraries, including vector, string, and algorithm.
The code starts by declaring some variables and importing the necessary libraries. It then defines a specialFilter function that takes
Write functional code in C++ according to the description.

```cpp
#include<stdio.h>
#include<math.h>
#include<vector>
#include<string>
#include<algorithm>
using namespace std;
#include<stdlib.h>
int specialFilter(vector<int> nums) { 
```  
StarChat- $\beta$ :

```txt
int n = nums.size();
int sum = 0;
for(int i = 0; i < n; i++) {
    sum += pow(nums[i], 2);
}
int target = (int)sqrt(sum);
vector<int> res;
for(int i = 0; i < n; i++) {
    if(nums[i] > target) {
    res.push_back(nums[i]);
    }
}
return res;
} 
```

Figure 38: Incorrect solution from StarChat- $\beta$ to a HUMANEVALEXPLAIN C++ problem. StarChat- $\beta$ has a bias for excessively long answers. StarChat- $\beta$ 's description includes irrelevant parts like "..problem on LeetCode..includes several standard libraries.." (despite the problem not existing on LeetCode). As the description exceeds the 296-character limit for the code, it is cut off at the end.

# R.8 DIFF CODEGEN

Input:   
```python
<NME> truncate_number.py
<BEF> def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    return number % 1.0 + 1.0
<MSG> Fix bugs in truncate_number.
<DFF> 
```

Diff Codegen 2B   
```diff
@@ -4,7 +4,7 @@ def truncate_number(number: float) -> float:
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

- Return the decimal part of the number.
+ Return 
```  
Figure 39: Incorrect solution from Diff Codegen 2B to a HUMANEVALFIXDOCS Python problem. Diff Codegen 2B suggests an irrelevant diff modifying parts of the docstring. The model commonly outputs diffs that modify the docstring or an import statement and rarely addresses the actual bug.

# S LIMITATIONS AND FUTURE WORK

Model Execution A promising avenue for improving performance on HUMANEVALFIX is letting the model execute the given code or its own generated code and inspect its output (Chen et al., 2022; 2023c; Yasunaga & Liang, 2021; Li et al., 2022a; Gao et al., 2023; Dong et al., 2023; Zhang et al., 2023c; Madaan et al., 2023b; Ni et al., 2023; Gou et al., 2023; Hu et al., 2023; Taylor et al., 2022; Nye et al., 2021). This could allow the model to discover which unit tests are failing and for what reason. The model could then simply iterate on the function until all unit tests are passing. We leave explorations of this strategy to improve performance on HUMANEVALPACK to future work.

Multi-file changes For the creation of COMMITPACK, we have filtered out any commits that affect multiple files to ensure commits are very specific and account for the fact that most current models are only capable of operating on a single file. Allowing models to take multiple files as input and modify multiple files given a single instruction is a promising direction for future work. There is active research on using repository-level context (Ding et al., 2022; Shrivastava et al., 2023a;b; Zhang et al., 2023a; Liu et al., 2023d) and the necessary long context windows (Dai et al., 2019; Press et al., 2021; Sun et al., 2021; Dao et al., 2022; Peng et al., 2023; Liu et al., 2023c; Chen et al., 2023b).

Length-awareness Current Code LLMs including OCTOCODER struggle with awareness about the length of their generated output. For HUMANEVALEXPLAIN, we instruct the models to limit their output to a given number of characters. While it is trivial for humans to count characters and adhere to the limit, all models tested frequently generate far too many characters. Prior work has shown that human raters are biased towards preferring longer texts (Wu & Aji, 2023) regardless of content. All models evaluated are instruction tuned on text that was at least indirectly assessed by human raters, hence they may be biased towards generating longer texts even if it means including literary bloat.

Better evaluation Evaluating code instruction models is challenging for several reasons: (1) Prompting: The prompt can significantly impact the performance of large language mod-

els (Brown et al., 2020; Zhou et al., 2022; Muennighoff, 2022; Babe et al., 2023). To ensure fair evaluation we use the prompting format put forth by the respective authors of the models and a simple intuitive prompt for models without a canonical prompt (see Appendix Q). However, this may put models without a canonical prompt recommendation (e.g. BLOOMZ, GPT-4) at a slight disadvantage. OCTOCODER and OCTOGEEX perform best when prompted using the same format we use during training (Figure 18) and we recommend always using this format at inference.

(2) Processing: Models may accidentally impair otherwise correct code by e.g. including a natural language explanation in their output. We largely circumvent this issue through the use of strict stopping criteria and careful postprocessing (e.g. for GPT-4 we check if it has enclosed the code in backticks, and if so, extract only the inner part of the backticks discarding its explanations).   
(3) Execution: When executing code to compute pass@k, it is important that the generated code matches the installed programming language version. Models may inadvertently use expressions from a different version (e.g. they may use the Python 2 syntax of print "hi", which would fail in a Python 3 environment). In our evaluation, we did not find this to be a problem, however, as models become more capable, it may make sense to specify the version. Future prompts may include the version (e.g. "use JDK 1.18.0") or provide models with an execution environment that has the exact version installed that will be used for evaluation.   
(4) Comprehensiveness: Executing code can only reflect functional correctness lacking a comprehensive understanding of quality. Compared to execution-based evaluation, the human judgment of code quality can be considered more comprehensive as humans can consider factors beyond correctness. Directly hiring human annotators can be inefficient and expensive, and therefore researchers have explored approaches to automate human-aligned evaluation via LLMs (Fu et al., 2023; Liu et al., 2023e; Zhuo, 2023). However, recent work (Wang et al., 2023b) suggests LLM-based evaluation can be biased towards certain contexts. Future work on automating the human-aligned evaluation of instruction tuned Code LLMs while avoiding such bias is needed.

Reward Models Our commit datasets, COMMITPACK and COMMITPACKFT, also lend themselves well for learning human preferences (Ethayarajh et al., 2024; Rafailov et al., 2024). The changed code after a commit generally represents a human-preferred version of the code (else the code would not have been modified). Thus, one could train a reward model that given the code before and after a commit, learns that the code afterward is better. Similar to prior work (Christiano et al., 2017; Ouyang et al., 2022), this reward model could then be used to guide a language model to generate code that is preferred by humans.

# T VERSION CONTROL

V1 → V2:

- Added Appendix O on HUMANEVALEXPLAIN metrics   
- Added Appendix N on Fill-in-the-Middle   
- Expanded the motivation for limiting the number of tokens in COMMITPACKFT in Appendix E   
- Fixed the StarCoder HUMANEVALFIXDOCS in Appendix K thanks to Abhijeet Awasthi   
- Specified the programming languages of CommitPackFT (CommitPack has 350 languages while CommitPackFT has 277 after filtering)   
• Made small writing improvements throughout

# U OctoBadPack

![](images/0afd4928e3bfc597bfe5c2cee1c2cfc592ba11e7de6755b63b70dc73b3b0b61b.jpg)  
Figure 40: OCTOPACK (left) and her evil brother OCTOBADPACK (right).