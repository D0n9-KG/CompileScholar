# Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models

Peijie Liu $^{1}$ Fengli Xu $^{1}$ Yong Li $^{1}$

# Abstract

Chain-of-Thought (CoT) technique has proven effective in improving the performance of large language models (LLMs) on complex reasoning tasks. However, the performance gains are inconsistent across different tasks, and the underlying mechanism remains a long-standing research question. In this work, we make a preliminary observation that the monotonicity of token probability distributions may be correlated with the gains achieved through CoT reasoning. Leveraging this insight, we propose two indicators based on the token probability distribution to assess CoT effectiveness across different tasks. By combining instance-level indicators with logistic regression model, we introduce Dynamic CoT, a method that dynamically select between CoT and direct answer. Furthermore, we extend Dynamic CoT to closed-source models by transferring decision strategies learned from open-source models. Our indicators for assessing CoT effectiveness achieve an accuracy of 89.2%, and Dynamic CoT reduces token consumption by more than 35% while maintaining high accuracy. Overall, our work offers a novel perspective on the underlying mechanisms of CoT reasoning and provides a framework for its more efficient deployment. The code can be found at https://github.com/tsinghua-fib-lab/Token\_Signature.

# 1. Introduction

Chain-of-Thought (CoT) prompting (Wei et al., 2022) has become a widely adopted technique for enhancing the reasoning capabilities of large language models(LLMs). By $^{1}$ Department of Electronic Engineering, BNRist, Tsinghua University, China. Correspondence to: Fengli Xu <fenglixu@tsinghua.edu.cn>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

incorporating examples of CoT reasoning in a few-shot prompt (Wei et al., 2022), CoT can be effectively triggered, and the ability of LLMs to solve various complex problems is improved by decomposing the problem step by step (Wang et al., 2022a), while also providing detailed and interpretable explanations (Lanham et al., 2023). Inspired by CoT reasoning, OpenAI (2024a) has introduced the concept of test-time scaling (Xu et al., 2025), which suggests that the reasoning capabilities of LLMs can be enhanced with more time spent thinking (test-time compute). For many problems such as mathematical word problems and symbolic reasoning, CoT is generally considered to be an effective method (Chae et al., 2024) (Qi et al., 2024).

However, recent studies have shown that CoT prompting does not consistently improve performance across all tasks, and its effectiveness varies depending on the problem domain. As illustrated in Figure 1, CoT's performance varies across different task categories. Moreover, in symbolic tasks, which are considered to be tasks where CoT is generally effective (Sprague et al., 2024), such as ContextHub-abductive and ContextHub-deductive (Hua et al., 2024), the significance of CoT gain also varies. In non-mathematical fields, for example, CoT has been found to be less effective (Kambhampati et al., 2024) and may even result in negative performance outcomes (Wang et al., 2024). Sprague et al. (2024) conducted a meta-analysis of CoT-related studies and experiments (Sprague et al., 2024), revealing that CoT is predominantly effective for mathematical and symbolic reasoning tasks, with limited or no improvements for other types of problems. However, this analysis (Sprague et al., 2024) focused solely on thematic trends and did not fully explore the underlying mechanisms driving CoT's effectiveness. While the effectiveness of CoT across different problems and models can be generally inferred from the task category, it remains inconsistent and lacks a definitive measure of effectiveness. Consequently, our work is motivated by two main goals: to explore the underlying mechanisms of CoT reasoning and to develop a task-level method for evaluating its effectiveness.

In this paper, we look at a novel perspective of the LLM decoding process. Instead of focusing on the most probable next token, we look at the token probability distribution and

![](images/ed297050d7269fe2ffa90b96f33dbd87a4a740c6348db2f16a51ccea1c612e10.jpg)

<details>
<summary>bar_line</summary>

| Category | Dataset | CoT Accuracy | CoT Gain |
| :--- | :--- | :--- | :--- |
| Mathematical | GSM8K | 70 | 58 |
| Mathematical | MultiArith | 85 | 56 |
| Mathematical | FOLIO | 54 | 5 |
| Mathematical | CH_a | 37 | 5 |
| Mathematical | CH_d | 52 | 8 |
| Symbolic | Arc_chall | 81 | 4 |
| Symbolic | Arc_easy | 88 | 2 |
| Knowledge | GPQA | 28 | -2 |
| Knowledge | MuSR | 49 | -2 |
| Knowledge | LSAT | 46 | -1 |
| Soft Reasoning | CSQA | 71 | 1 |
| Soft Reasoning | PIQA | 81 | 1 |
| Commonsense | SIQA | 66 | 1 |
| Commonsense | StrategyQA | 75 | -3 |
| Statistical | GSM8K (error bars) | 12 | 12 |
| Statistical | MultiArith (error bars) | 30 | 30 |
| Statistical | FOLIO (error bars) | 48 | 48 |
| Statistical | CH_a (error bars) | 32 | 5 |
| Statistical | CH_d (error bars) | 44 | 8 |
| Knowledge | Arc_chall (error bars) | 76 | 4 |
| Knowledge | Arc_easy (error bars) | 87 | 2 |
| Soft Reasoning | GPQA (error bars) | 30 | -2 |
| Soft Reasoning | MuSR (error bars) | 51 | -2 |
| Soft Reasoning | LSAT (error bars) | 46 | -1 |
| Commonsense | CSQA (error bars) | 70 | 1 |
| Commonsense | PIQA (error bars) | 80 | 1 |
| Commonsense | SIQA (error bars) | 65 | 1 |
| Commonsense | StrategyQA (error bars) | 78 | -3 |
The chart displays the error bars for each category on the Y-axis labeled 'Accuracy'. Error bars are shown as vertical lines between the bars.
</details>

Figure 1: Average zero-shot CoT accuracy, direct answer (DA) accuracy, and CoT Gain across five benchmark categories for four open-source models. The benchmarks highlighted in green indicate that CoT is significantly greater than 0 (p<0.05). The categories include: Mathematical, Symbolic, Knowledge, Soft Reasoning, and Commonsense. The results show that CoT does not consistently lead to performance gain. Additionally, CoT's performance varies across different question types and task categories, and even within the same category, its effectiveness is inconsistent (with varying significance levels). This highlights that the utility of CoT cannot be solely determined by the question category.

how it changes as the number of tokens scales, which is what we defined as Token Signature. Specifically, we use standard prompts (i.e., questions only) to elicit responses and observe that the probability distribution of the initial token in the model's greedy decoding path is highly variable and closely correlated with CoT gain. Leveraging this insight, we develop two indicators based on the token probability distribution and Spearman Correlation (SC) (Wissler, 1905): Instance SC and Aggregated SC. These indicators quantify CoT effectiveness at the benchmark level. Secondly, we apply Instance-level SC to individual instances across different models and combine a small number of benchmark samples for classification, which allows us to dynamically select between CoT and direct answer. We refer to this approach as Dynamic CoT. Finally, we determine the best answer type (CoT or direct answer) by ensembling the results of a small open-source model at the question level, which is then transferred to a larger closed-source model for evaluation.

We test the effectiveness of our method on 12 well-known benchmarks, including GSM8K (Cobbe et al., 2021), Multi-Arith (Roy & Roth, 2016), CommonsenseQA (Talmor et al., 2018), and LSAT (Zhong et al., 2023), etc. We demonstrate its generalization across four closed-source models and two open-source models. At the benchmark level, we introduce two indicators that effectively predict the applicability of CoT. The positive and negative values of these indicators are closely correlated with CoT gains. Specifically, Instance SC achieves a prediction accuracy of $69.6\%$ , while Aggregated SC reaches $89.2\%$ . At the question level, the accuracy of our method, Dynamic CoT, is nearly identical to the highest performance between CoT and direct answers. Compared to CoT, Dynamic CoT reduces token consumption by 39.1%. In the transfer experiment, our method still maintains high accuracy and reduces token consumption by 35.8%.

Our main contributions are as follows:

- We introduce the concept of Token Signature to study the CoT gain across different question types based on the decoded token probability distribution.   
- We propose two token probability distribution indicators, Instance SC and Aggregated SC, to assess whether a benchmark is suitable for CoT at the benchmark level.   
- We design Dynamic CoT at the instance level to enable the reasonable selection of either CoT or direct answer.

# 2. Preliminary Token-level Analysis

In this section, we present our initial observation on token probability distribution in open-source language models. We begin by conducting a token-level probability analysis using the publicly available Mistral-7B-Instruct model. The model is prompted with question-only inputs (i.e., without CoT trigger or direct answer trigger) and decoded using a greedy strategy. Figure 3 illustrates the probability distribution of the initially generated tokens across the four benchmarks.

Our preliminary result reveals distinct patterns across differ-

![](images/7d9b4855f66cf74452c444cf935263ffc88ec44b8cba67a2571c808dcfd888af.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Question"] --> B["LLMs"]
    B --> C{Question}
    C --> D["To select the most correct option from the options provided. (A): to dry a table that was under the sea (B): to dry a table thats wet under the sea"]
    C --> E{To select the most correct option from the options provided. (A): to dry a table that was under the sea (B): to dry a table thats wet under the sea}
    B --> F["The correct answer is (E): slow down . The phrase ..."]
    B --> G["The correct option is (B): to dry a table that was under the sea"]
    B --> H["The correct option is (A): to dry a table that was under the sea"]
    I["CoT Answer"] --> J["Spearman Correlation"]
    J --> K["Direct answer"]
    K --> L["0.58"]
    M["DA Answer"] --> N["Spearman Correlation"]
    N --> O["Direct answer"]
    O --> P["0.58"]
    Q["To determine the correct option, ... the most correct option is:\n\n(A): to dry a table that was under the sea"] --> R["DA Answer"]
    S["Let's break it down step by step:., ... the correct answer is:\n\n(C): go slowly"] --> T["Do not make it down step"]
```
</details>

Figure 2: Illustration of the proposed method for analyzing CoT features through decoding. We calculate spearman correlation from token probability distribution obtained via greedy decoding of the standard prompt. This indicator reflects model confidence in answering a question and guides whether to introduce CoT reasoning after the standard prompt or to directly respond with a trigger prompt.

![](images/d4aad29f15ff7512ed671748f267b9e1c0dbbff1fcc337ce4b62fc722103f1d6.jpg)

<details>
<summary>line</summary>

| Token Index | Mean Probability |
| ----------- | ---------------- |
| 0           | 0.8              |
| 5           | 0.9              |
| 10          | 0.95             |
| 15          | 0.9              |
| 20          | 0.85             |
| 25          | 0.9              |
| 30          | 0.9              |
| 35          | 0.9              |
| 40          | 0.9              |
| 45          | 0.9              |
| 50          | 0.9              |
</details>

![](images/041d8006cfd69533fc0f5eb0004e92809c414f398a751cc0eaa5b4815d96e1ed.jpg)

<details>
<summary>scatter</summary>

| Token Index | Mean Probability |
| ----------- | ---------------- |
| 0           | 0.7              |
| 5           | 0.8              |
| 10          | 0.9              |
| 15          | 0.9              |
| 20          | 0.9              |
| 25          | 0.9              |
| 30          | 0.9              |
| 35          | 0.9              |
| 40          | 0.9              |
| 45          | 0.9              |
| 50          | 0.9              |
</details>

![](images/d50e9c0acbac7537a68ac367e37ad28a0c2b2523c73beaefe3d87fa30b5d95ea.jpg)

<details>
<summary>line</summary>

| Token Index | Mean Probability |
| ----------- | ---------------- |
| 0           | 0.8              |
| 5           | 0.9              |
| 10          | 0.85             |
| 15          | 0.8              |
| 20          | 0.75             |
| 25          | 0.78             |
| 30          | 0.77             |
| 35          | 0.76             |
| 40          | 0.75             |
| 45          | 0.74             |
| 50          | 0.73             |
</details>

![](images/368a146a993a4ba01812754365becf3aad5827870b25ae19e34b485e40798f73.jpg)

<details>
<summary>line</summary>

| Token Index | Mean Probability |
| ----------- | ---------------- |
| 0           | 0.9              |
| 5           | 0.85             |
| 10          | 0.8              |
| 15          | 0.75             |
| 20          | 0.7              |
| 25          | 0.65             |
| 30          | 0.6              |
| 35          | 0.55             |
| 40          | 0.5              |
| 45          | 0.45             |
| 50          | 0.4              |
</details>

Figure 3: Probability distributions of the first 50 tokens generated along the trigger-free greedy decoding path for four benchmarks: MultiArith, GSM8K, SIQA and PIQA. The observed trends suggest a potential correlation between token probability distributions and CoT gain.

ent benchmarks. Notably, for benchmarks such as GSM8K and MultiArith, where CoT reasoning significantly enhances performance, the token probability distribution exhibits an increasing trend—indicating that later tokens are assigned higher probabilities, reflecting greater model confidence. In contrast, for benchmarks such as PIQA and SIQA, where CoT has almost no benefit, the probability distributions display a decreasing trend, suggesting a decline in model confidence as decoding progresses. Based on these observations, we propose the following hypothesis:

The probability distribution of large language models along the decoding path is potentially correlated with the CoT gain across different question types.

This insight motivates the approach introduced in the next section, where we leverage token probability distributions to characterize the features of CoT reasoning.

# 3. Our Approach

In this section, we introduce a novel perspective on the decoding process of LLMs. Rather than focusing solely on the most probable next token, we analyze the token probability distribution and its evolution over the decoding trajectory, which is what we define as Token Signature. We propose two key indicators to evaluate the effectiveness of CoT reasoning at the benchmark granularity. Next, we design an instance-granularity approach for dynamically selecting CoT. Finally, we develop a mechanism to adapt our method to closed-source models.

# 3.1. Token Signature

In Section 2, we preliminarily find that the trends of the probability distribution of tokens in different benchmarks are different. To capture this different feature, we introduce Spearman Correlation (Wissler, 1905) under standard prompt to measure the correlation between token probability and sequence order. A schematic diagram of the instance-level calculation of the token signature is shown in Figure 2.

For spearman correlation $(\rho_{i})$ (Wissler, 1905), that is, given two ranked variables $X = \{x_{1}, x_{2}, \ldots, x_{n}\}$ and

$Y = \{y_{1}, y_{2}, \ldots, y_{n}\}$ , the $\rho_{i}$ is defined as:

$$
\text { Spearman } (X, Y) = 1 - \frac {6 \sum d _ {i} ^ {2}}{n (n ^ {2} - 1)}, \tag {1}
$$

where $d_{i} = R(x_{i}) - R(y_{i})$ is the difference in ranks for each pair $(x_{i}, y_{i})$ , and n is the number of observations.

Instance SC The Instance SC measures the monotonic relationship between the token probabilities and their sequence order within an individual response. For each question $q_{i}$ , we extract the probability sequence of the first 50 tokens (typically covering 28% of the entire response):

$$
P _ {i} = \{p _ {i, 1}, p _ {i, 2}, \dots , p _ {i, 5 0} \}, \tag {2}
$$

where $p_{i,t}$ represents the model's softmax probability for the $t$ -th token in the generated response to question $q_i$ . We compute the spearman correlation between $P_i$ and its corresponding token index sequence $T = \{1, 2, \dots, 50\}$ :

$$
\rho_ {i} = \text { Spearman } (P _ {i}, T). \tag {3}
$$

Finally, the Instance SC is defined as the mean Spearman correlation across all test instances in a given benchmark:

$$
\text { Instance } \mathbf {S C} = \frac {1}{N} \sum_ {i = 1} ^ {N} \rho_ {i} \tag {4}
$$

where $N$ is the total number of questions in the benchmark.

Aggregated SC The Aggregated SC provides a benchmark-wide measure of token probability trends. Instead of computing Spearman Correlation per instance, we first compute the mean probability of each token index across all responses:

$$
\bar {P} _ {t} = \frac {1}{N} \sum_ {i = 1} ^ {N} p _ {i, t}, \quad t \in \{1, 2, \dots , 5 0 \}, \tag {5}
$$

where $\bar{P}_{t}$ represents the average probability assigned to the t-th token across all responses in the benchmark. We then compute the Spearman Correlation between the aggregated probability sequence $\bar{P}=\{\bar{P}_{1},\bar{P}_{2},\ldots,\bar{P}_{50}\}$ and the token index sequence $T=\{1,2,\ldots,50\}$ :

$$
\text { Aggregated   } \mathbf {S C} = \text { Spearman } (\bar {P}, T). \tag {6}
$$

We use the two metrics, Instance SC and Aggregated SC, to predict the effectiveness of CoT on a specific benchmark. The significance of CoT is categorized into three levels: positive, none, and negative. The prediction results are determined as follows:

$$
\text { Pred\_Significance } = \left\{ \begin{array}{l l} \text { positive }, & \text { if   indicator } > 0, \\ \text { none / negative }, & \text { if   indicator } \leq 0. \end{array} \right. \tag {7}
$$

# 3.2. Dynamic CoT

We also propose a classification-based approach that leverages Instance-level SC to adaptively apply CoT reasoning. Given the variability in LLM's training data, we do not simply use zero as the threshold for instance-level classification. Instead, we introduce a logistic regression model trained on a small sample (50 instances) per benchmark, using instance-level SC as input and assigning labels. This trained model is then used to classify the remaining instances.

Specifically, the test label represents the better prompt to choose. When we selected the test set, we did not consider the case where CoT and DA had the same answer. For question $q_{i}$ , $y_{i}$ be the binary label, where:

$$
y _ {i} = \left\{ \begin{array}{l l} 1, & \text { if   answer   of   CoT   is   correct }, \\ 0, & \text { if   answer   of   DA   is   correct }. \end{array} \right. \tag {8}
$$

Combining the Instance-level SC and labels of equation (3), we train the logistic regression model:

$$
P (y _ {i} = 1 \mid \rho_ {i}) = \frac {1}{1 + e ^ {- (w \rho_ {i} + b)}}, \tag {9}
$$

where $w$ and $b$ are the learned parameters, weight, and bias. We select the binary cross-entropy loss as a loss function:

$$
L = - \frac {1}{N} \sum_ {i = 1} ^ {N} \left[ y _ {i} \log \hat {y} _ {i} + (1 - y _ {i}) \log (1 - \hat {y} _ {i}) \right], \tag {10}
$$

During classification, if the predicted probability satisfies:

$$
P (y _ {i} = 1 \mid \rho_ {i}) > 0. 5, \tag {11}
$$

then we classify $q_{i}$ as requiring CoT (i.e., $y_{i} = 1$ ). Otherwise, the model generates a direct answer without CoT.

# 3.3. Transfer to Closed-source Model

Most closed-source models do not provide token probability outputs, making it challenging to apply our method. To address this limitation, we propose a voting mechanism to transfer our method.

We first evaluate the Dynamic CoT selection strategy in multiple open-source models to obtain multiple predicted $P_{i}$ for the question. We aggregate predictions from multiple open-source models using a voting mechanism. Specifically, the final label $Y_{i}$ for the instance $q_{i}$ is computed as:

$$
Y _ {i} = \mathbb {I} \left(\frac {1}{M} \sum_ {m = 1} ^ {M} P _ {i} ^ {(m)} > 0. 5\right), \tag {12}
$$

where M is the total number of open-source models. $\mathbb{I}(\cdot)$ is the indicator function, which outputs 1 if the condition holds and 0 otherwise.

CoT is used only when $Y_{i}$ is 1, otherwise it is a direct answer. Based on the above voting results, the Dynamic CoT is then transferred to a closed-source model and then tested.

# 4. Experimental Setup

In this section, we will introduce the following aspects: model, benchmark, and prompt used in the experiment and how to evaluate the accuracy of the experiment.

# 4.1. Base Models

We conduct experiments primarily on four widely used open-source models and two popular closed-source models. The four open-source models include: Llama-3.2-3B-Instruct (Dubey et al., 2024), Phi-3.5-mini-instruct (Abdin et al., 2024), Llama-3.1-8B-Instruct (Dubey et al., 2024) and Mistral-7B-Instruct-v0.3 (Jiang et al., 2023). The two closed-source models are GPT-4o-mini and GPT-4o (OpenAI, 2024b).

# 4.2. Benchmark

For benchmarks, we refer to the categories outlined in Sprague et al. (2024)'s work. We focus on five types of benchmarks: Mathematical, Symbolic, Knowledge, Soft Reasoning, and Commonsense. The benchmarks used are categorized as follows:

• Mathematical: GSM8K (Cobbe et al., 2021), Multi-Arith (Roy & Roth, 2016)   
- Symbolic: FOLIO (Han et al., 2022), ContextHub (Hua et al., 2024)   
- Knowledge: ARC (Clark et al., 2018), GPQA (Rein et al., 2023)   
- Soft Reasoning: MuSR (Sprague et al., 2023), AGIEval LSAT (Zhong et al., 2023)   
- Commonsense: CommonsenseQA (Talmor et al., 2018), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), StrategyQA (Geva et al., 2021)

The answer formats mainly include two types: short-answer and multiple-choice. We introduce the specific benchmark details in the Appendix A.

# 4.3. Prompt Settings

We utilize two primary types of prompts: zero-shot CoT prompt (Kojima et al., 2022), and zero-shot direct answer(DA) prompt. For the zero-shot CoT prompt, we employ the phrase “Let’s think step by step” (Kojima et al., 2022) as the CoT trigger. Additionally, we designed a direct answer prompt tailored to different benchmarks to ensure the model adhered to the provided instructions. Detailed descriptions of the prompts can be found in the Appendix A.

# 4.4. Evaluation

Answer Extract To extract answers from the model's response, we employ distinct strategies tailored to different question types. For short-answer mathematical reasoning questions, we select the final numerical value in the model's output as the answer, adhering to a widely accepted protocol for evaluating language models (Ivison et al., 2023; Wang et al., 2023). For multiple-choice questions, we identify the first letter of the option provided in the direct response as the answer. In the case of the CoT response, we append the prompt “So the best answer letter choice is” to extend the response, subsequently extracting the corresponding letter option as the answer, and then matching them with the correct answer.

Answer Accuracy We evaluate the accuracy of the answers by comparing the extracted answers with the correct ones. For each evaluation, we calculate the accuracy as:

$$
\mathrm{Acc} = N _ {\text { correct }} / N, \tag {13}
$$

where $N_{correct}$ denotes the number of correctly answered questions, and N represents the total number of questions.

Significance Judgment To assess the significance of CoT gain on a benchmark, we perform a two-tailed Z test. The null hypothesis assumes no significant difference between DA Acc $(p_{1})$ and CoT Acc $(p_{2})$ , i.e., $p_{1} = p_{2}$ . The alternative hypothesis tests whether the difference $p_{2} - p_{1}$ is significantly different, i.e., $p_{2} \neq p_{1}$ . The detailed calculation process is provided in Appendix A.

# 5. Results

# 5.1. CoT Effectiveness at the Benchmark Level

In this section, we present the results of using two Token Signature indicators (Instance SC and Aggregated SC) to predict the effectiveness of CoT reasoning across different benchmarks.

We evaluate CoT reasoning and direct answer performance on 12 benchmarks using 4 open-source models, with all results summarized in Table 7. Additionally, we compute the values of Instance SC and Aggregated SC for each benchmark and record their corresponding CoT gain in Table 1. Table 1 presents the results of the Llama-3.2-3B-Instruct model. The values of Instance SC and Aggregated SC exhibit a strong correlation with CoT gain, where their signs (positive or negative) align closely with the effectiveness of CoT reasoning. For benchmarks with significant CoT gains, such as GSM8K, MultiArith, FOLIO, and CH\_d, both Instance SC and Aggregated SC are positive. Conversely, for benchmarks with minimal or negative CoT gains, such as MuSR, LSAT, SIQA, and StrategyQA, both indicators are negative. A similar trend can be observed in Table 8.

Table 1: Instance SC, Aggregated SC, and CoT Gain across benchmarks on Llama-3.2-3B-Instruct. SC indicators (Instance SC/Aggregated SC) effectively predict CoT effectiveness at the benchmark granularity. Significance indicates the significance of CoT gain determined using the Z test (positive/none/negative). For more complete information, see Table 7 and 8. 

<table><tr><td></td><td>Instance SC</td><td>Aggregated SC</td><td>CoT Gain</td><td>Significance</td></tr><tr><td>GSM8K</td><td>0.0450</td><td>0.2080</td><td>63.76</td><td>positive</td></tr><tr><td>MultiArith</td><td>0.1619</td><td>0.2904</td><td>67.83</td><td>positive</td></tr><tr><td>FOLIO</td><td>0.1869</td><td>0.0488</td><td>6.31</td><td>positive</td></tr><tr><td>CH_a</td><td>0.0720</td><td>-0.032</td><td>1.67</td><td>none</td></tr><tr><td>CH_d</td><td>0.0900</td><td>0.2670</td><td>19.95</td><td>positive</td></tr><tr><td>Arc_chall</td><td>-0.0513</td><td>-0.4597</td><td>4.01</td><td>positive</td></tr><tr><td>Arc_easy</td><td>-0.0552</td><td>-0.6061</td><td>0.93</td><td>none</td></tr><tr><td>GPQA</td><td>0.0038</td><td>-0.1042</td><td>-10.71</td><td>negative</td></tr><tr><td>MuSR</td><td>-0.0840</td><td>-0.4016</td><td>-6.09</td><td>negative</td></tr><tr><td>LSAT</td><td>-0.0661</td><td>-0.4180</td><td>-0.99</td><td>none</td></tr><tr><td>CSQA</td><td>-0.1102</td><td>-0.3424</td><td>1.64</td><td>none</td></tr><tr><td>PIQA</td><td>-0.2698</td><td>-0.5340</td><td>1.74</td><td>none</td></tr><tr><td>SIQA</td><td>-0.2698</td><td>-0.7639</td><td>0.36</td><td>none</td></tr><tr><td>StrategyQA</td><td>-0.0082</td><td>-0.3917</td><td>-13.1</td><td>negative</td></tr></table>

We hypothesize that when these indicators are negative during the model's reasoning process, it suggests a high degree of uncertainty, which leads to the inadaptability of CoT reasoning. This implies that CoT is less effective in scenarios where token probability distributions indicate lower confidence in sequential reasoning.

Notably, for symbolic benchmarks such as CH\_d, CoT exhibits a negative effect on Mistral-7B-Instruct-v0.3 and Phi-3.5-mini-instruct, with their corresponding Aggregated SC values also being negative. Conversely, CoT has a positive effect on Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct, where their Aggregated SC values are positive.

We further evaluate the accuracy of Instance SC and Aggregated SC in predicting the effectiveness of CoT across four models. To classify the impact of CoT, we define three categories using Z test CoT gain: positive significance (CoT improves performance), no significance (CoT has negligible impact), and negative significance (CoT reduces performance). The prediction is considered accurate if the indicator is greater than 0 when CoT has a positive gain, or less than 0 when CoT has no gain or a negative gain, which indicates that CoT is ineffective or even harmful. The detailed accuracy results are reported in Table 2.

We observe that both indicators demonstrate high accuracy in predicting CoT effectiveness. On average, Instance SC achieves 69.6% accuracy, while Aggregated SC performs even better, reaching 89.2% accuracy. These results further reinforce the potential correlation between token probability distribution and CoT gain, while also validating the predictive power of our proposed indicators. Compared with Instance SC, Aggregated SC has better prediction ability, which may be attributed to the divergence of instance-granular token probability distribution.

Table 2: Prediction accuracy of two indicators for zero-shot CoT effectiveness across four open-source models. 

<table><tr><td></td><td>Instance SC</td><td>Aggregated SC</td></tr><tr><td>Llama-3.2-3B</td><td>78.6</td><td>92.9</td></tr><tr><td>Mistral-7B</td><td>50.0</td><td>85.7</td></tr><tr><td>Phi-3.5-mini</td><td>78.6</td><td>85.7</td></tr><tr><td>Llama-3.1-8B</td><td>71.4</td><td>92.9</td></tr></table>

# 5.2. Dynamic CoT

In Section 3, we introduce our Dynamic CoT, which integrates Instance-level SC with a logistic regression model to explore the impact of question granularity and dynamically select CoT and DA. The experimental results for Dynamic CoT across four open-source models are presented in Table 9, while the average performance of Dynamic CoT on these models is summarized in Table 3.

Table 3: Average performance impact of Dynamic CoT across benchmarks on four open-source models. Gap denotes the relative difference between Dynamic CoT and the better of CoT and DA. Dynamic CoT consistently achieves the higher accuracy between CoT and direct answers on most benchmarks.

<table><tr><td></td><td>CoT Acc</td><td>DA Acc</td><td>Dynamic CoT</td><td>Gap</td></tr><tr><td>GSM8K</td><td>70.28</td><td>12.32</td><td>69.29</td><td>-1.4</td></tr><tr><td>MultiArith</td><td>85.59</td><td>30.21</td><td>85.00</td><td>-1.0</td></tr><tr><td>FOLIO</td><td>53.93</td><td>49.09</td><td>54.05</td><td>0.2</td></tr><tr><td>CH_a</td><td>37.02</td><td>31.95</td><td>35.29</td><td>-4.7</td></tr><tr><td>CH_d</td><td>51.73</td><td>44.20</td><td>53.06</td><td>3.6</td></tr><tr><td>Arc_chall</td><td>80.55</td><td>76.60</td><td>81.20</td><td>0.8</td></tr><tr><td>Arc_easy</td><td>88.21</td><td>86.89</td><td>88.65</td><td>0.5</td></tr><tr><td>GPQA</td><td>27.40</td><td>30.19</td><td>28.52</td><td>-5.5</td></tr><tr><td>MuSR</td><td>48.81</td><td>51.42</td><td>50.99</td><td>-0.8</td></tr><tr><td>LSAT</td><td>45.10</td><td>46.61</td><td>45.99</td><td>-1.3</td></tr><tr><td>CSQA</td><td>71.05</td><td>70.35</td><td>71.54</td><td>1.7</td></tr><tr><td>PIQA</td><td>80.93</td><td>80.73</td><td>82.31</td><td>1.7</td></tr><tr><td>SIQA</td><td>66.35</td><td>66.15</td><td>66.89</td><td>0.8</td></tr><tr><td>StrategyQA</td><td>75.02</td><td>78.27</td><td>79.54</td><td>1.6</td></tr></table>

In Table 9 and 10, we can see that the performance of Dynamic CoT basically reaches the highest performance among all CoT and all direct answers. Specifically, in a series of 56 experimental setups across four models, Dynamic CoT achieves the highest performance in 46.4% of

the cases within CoT, direct answer, and Dynamic CoT comparison, and ranks among the top two methods in 92.8% of these experiments. Additionally, Table 3 demonstrates that Dynamic CoT achieve the best average performance across eight benchmarks and the second-best performance across six others. These results demonstrate that our method is highly effective at achieving classification-level performance on a question-specific granularity.

![](images/bcfaf90d88a1437b92244a04dd0acb0caa34fdcf618c93015deebaf24c3afcde.jpg)

<details>
<summary>line</summary>

| Category     | Dynamic CoT | CoT  |
| ------------ | ----------- | ---- |
| GSM8K        | 280         | 280  |
| MultiArith   | 150         | 150  |
| FOLIO        | 230         | 260  |
| CH_a         | 150         | 230  |
| CH_d         | 160         | 220  |
| Arc_chall    | 240         | 230  |
| Arc_easy     | 180         | 220  |
| GPQA         | 170         | 650  |
| MuSR         | 80          | 310  |
| LSAT         | 170         | 350  |
| CSQA         | 150         | 210  |
| PIQA         | 90          | 190  |
| SIQA         | 170         | 190  |
| StrategyQA   | 50          | 190  |
</details>

Figure 4: Comparison of token consumption between Dynamic CoT and All CoT in multiple open-source models of multiple benchmarks

Furthermore, Figure 4 presents the average token consumption across 14 benchmarks for both CoT and Dynamic CoT. The findings indicate that Dynamic CoT uses fewer tokens than CoT on eight benchmarks, with significant reductions observed in GPQA, MuSR, PIQA, and StrategyQA. Overall, Dynamic CoT reduces token consumption by an average of 104 tokens, representing a 39.1% reduction compared to CoT.

In summary, our Dynamic CoT method has been effectively applied to open-source models, enabling dynamic selection between CoT and direct answers while maintaining high accuracy. Furthermore, our approach significantly reduces unnecessary token generation compared to the direct use of CoT, particularly in benchmarks where CoT is less effective.

# 5.3. Model Transfer

For LLMs that cannot be deployed locally, the token probability distribution may be inaccessible, and the evaluation cost tends to be relatively high. As a result, a transfer strategy is required to adapt our method to closed-source models. So we propose an approach to integrate results from smaller open-source models, which are then applied to the closed-source model. The details of this method are outlined in Section 3.

Table 11 presents the experimental results for each benchmark on GPT-4o-mini and GPT-4o. The results in Table 4 indicate that Dynamic CoT performs well, ranking first Table 4: The cross-model transfer effect of Dynamic CoT from open-source to closed-source models across benchmarks. Gap denotes the relative difference between Dynamic CoT and the better of CoT and DA. Experimental results demonstrate significant performance improvements on closed-source models and multiple benchmarks.

<table><tr><td></td><td>CoT Acc</td><td>DA Acc</td><td>Dynamic CoT</td><td>Gap</td></tr><tr><td>GSM8K</td><td>84.76</td><td>42.46</td><td>84.76</td><td>0.0</td></tr><tr><td>MultiArith</td><td>93.92</td><td>96.42</td><td>93.92</td><td>-0.5</td></tr><tr><td>FOLIO</td><td>72.14</td><td>65.91</td><td>72.14</td><td>0.0</td></tr><tr><td>CH_a</td><td>58.17</td><td>44.19</td><td>58.17</td><td>0.0</td></tr><tr><td>CH_d</td><td>57.17</td><td>48.34</td><td>57.17</td><td>0.0</td></tr><tr><td>Arc_chall</td><td>94.11</td><td>92.19</td><td>94.11</td><td>0.0</td></tr><tr><td>Arc_easy</td><td>94.30</td><td>94.28</td><td>94.30</td><td>0.0</td></tr><tr><td>GPQA</td><td>51.68</td><td>48.11</td><td>47.88</td><td>-7.4</td></tr><tr><td>MuSR</td><td>60.78</td><td>57.61</td><td>57.61</td><td>-5.2</td></tr><tr><td>LSAT</td><td>68.93</td><td>68.34</td><td>68.59</td><td>-0.5</td></tr><tr><td>CSQA</td><td>82.76</td><td>83.29</td><td>82.72</td><td>-0.7</td></tr><tr><td>PIQA</td><td>89.97</td><td>91.70</td><td>90.29</td><td>-1.5</td></tr><tr><td>SIQA</td><td>77.49</td><td>77.05</td><td>77.46</td><td>0.0</td></tr><tr><td>StrategyQA</td><td>61.77</td><td>54.72</td><td>54.76</td><td>-11.3</td></tr></table>

or second in the majority of experiments across the three conditions. As shown in Figures 4 and 5, compared to the native Dynamic CoT, our method exhibits a slight decrease in classification performance. In benchmarks such as GSM8K, MultiArith, and FOLIO, Dynamic CoT exhibits performance equivalent to CoT. This outcome arises from the voting process, where the method ultimately selects CoT for these benchmarks. Notably, these benchmarks show relatively high gains from CoT, highlighting that our approach effectively identifies when CoT is appropriate at the benchmark level.

![](images/1e24808aaabb1f6c1262c0228e0053a628f7ea32de866df7aa8c20c6143a6664.jpg)

<details>
<summary>line</summary>

| Model       | Dynamic CoT(transfer) | CoT  |
|-------------|------------------------|------|
| GSM8K       | 320                    | 320  |
| MultiArith  | 190                    | 190  |
| FOLIO       | 450                    | 450  |
| CH_a        | 480                    | 480  |
| CH_d        | 360                    | 360  |
| Arc_chall   | 310                    | 310  |
| Arc_easy    | 290                    | 290  |
| GPQA        | 10                     | 780  |
| MuSR        | 10                     | 520  |
| LSAT        | 290                    | 590  |
| CSQA        | 260                    | 260  |
| PIQA        | 180                    | 240  |
| SIQA        | 240                    | 240  |
| StrategyQA  | 10                     | 240  |
</details>

Figure 5: Average token consumption of the Dynamic CoT approach after transferring the approach from open-source to closed-source. A comparison of token consumption between Dynamic CoT (transfer) and All CoT across multiple models and benchmarks.   
Figure 5 presents the token consumption for each bench-

mark on the GPT model. Results show no change in token consumption for benchmarks such as GSM8K, MultiArith, and FOLIO, while a decrease is observed for benchmarks like GPQA, MuSR, and LSAT. On average, our method consumes 134 fewer tokens than all CoT methods, resulting in a 35.8% reduction.

Overall, after transferring using the voting method, Dynamic CoT retains high accuracy, although its fine-grained classification ability experiences a slight reduction. Additionally, this approach consumes fewer tokens compared to all CoT methods.

# 6. Analysis

# 6.1. Impact of SC Threshold

To examine the effect of token count on SC-based prediction performance, we conduct experiments using the top n tokens, where $n \in \{10, 20, 50, 100, 200\}$ . We compare the accuracy of both Instance SC and Aggregated SC under these settings. The results are presented in Figure 6. Notably, using the first 50 tokens yields the highest prediction performance, 69.6% for Instance SC and 89.3% for Aggregated SC.

Based on these findings, we empirically select 50 as the default token threshold in our experiments. We observe that smaller n values fail to capture sufficient correlation trends, while larger values suffer from sparsity issues due to shorter decoding paths in many samples. Therefore, using 50 tokens provides a balanced trade-off between signal strength and data availability.

![](images/c7e1e6651d94ce1748f61984f0d4c3d3b372110fc46037bea6af14003f329857.jpg)

<details>
<summary>line</summary>

| SC Threshold | Aggregated_SC | Instance_SC |
| ------------ | ------------- | ----------- |
| 0            | 40            | 30          |
| 50           | 90            | 70          |
| 100          | 80            | 65          |
| 200          | 55            | 65          |
</details>

Figure 6: Prediction accuracy of Instance SC and Aggregated SC under varying token thresholds. The best performance is achieved when using the top 50 tokens.

# 6.2. Impact of Decoding Strategies

Decoding strategy is a critical factor influencing the outputs of large language models. We systematically evaluate the impact of different decoding strategies, focusing primarily on temperature and Top-K sampling, using the Llama-3.2-3B-Instruct model. We perform a sensitivity analysis across various samples and compare the prediction accuracy of two CoT effectiveness indicators at the benchmark granularity under these varying decoding configurations. The results, presented in Table 12, indicate that our method consistently maintains strong predictive power and robustness despite changes in decoding strategies. Notably, across different task types, Token Signature exhibits similar predictive characteristics for assessing the effectiveness of chain-of-thought reasoning.

# 6.3. Impact of CoT Prompt

We investigate the impact of alternative prompt settings, such as few-shot CoT prompting. We conduct experiments using two indicators, with detailed results presented in Table 13. The prediction rates for each indicator are statistically summarized in Table 5. The experimental results show that our method still has good performance under few-shot CoT. Furthermore, Token Signature effectively predicts the gains achieved by few-shot CoT. Overall, the results demonstrate strong robustness.

Table 5: Prediction accuracy of Instance SC and Aggregated SC for few-shot CoT effectiveness across four open-source models. 

<table><tr><td></td><td>Instance SC</td><td>Aggregated SC</td></tr><tr><td>Llama-3.2-3B</td><td>92.9</td><td>92.9</td></tr><tr><td>Mistral-7B</td><td>50.0</td><td>85.7</td></tr><tr><td>Phi-3.5-mini</td><td>57.1</td><td>78.6</td></tr><tr><td>Llama-3.1-8B</td><td>71.4</td><td>92.9</td></tr></table>

# 6.4. Intuitive Theoretical Analysis

Token probability has been shown to reflect a language model's confidence in its outputs (Farquhar et al., 2024). CoT prompting can improve performance on tasks with inherently sequential structures by enabling a deeper, token-expensive search process (Li et al., 2024). However, for tasks that are not intrinsically sequential, CoT may have adverse effects due to the accumulation of reasoning errors, also known as the snowball effect (Gan et al., 2025).

Mathematical reasoning tasks, such as those found in GSM8K, require a strict step-by-step process, where operations (e.g., arithmetic calculations, logical deductions) must be executed in a well-defined order. The solution space for such problems is highly constrained, and the generation of intermediate steps must adhere to deterministic rules. In this context, CoT acts as a structured reasoning scaffold, effectively enhancing the model's confidence and improving solution accuracy.

Conversely, in tasks such as commonsense reasoning, the solution space is more diverse and less rule-governed. When the model's initial confidence is low, CoT may lead it down

an incorrect reasoning path, with each subsequent step compounding earlier errors. This error propagation amplified by CoT can degrade performance.

Token Signature as Early Predictor We leverage the Spearman Correlation indicators between token probabilities and correct reasoning paths as a proxy for the model's uncertainty at the onset of the reasoning process. A high SC indicates that the model's internal token confidence is well-aligned with successful reasoning trajectories, suggesting that CoT is likely to yield performance gains. Thus, SC serves as an effective indicator for predicting the utility of CoT across different task types.

# 7. Related Work

# 7.1. Chain-of-Thought Reasoning

CoT reasoning enhances LLMs by generating intermediate reasoning steps, thereby improving both interpretability and performance on complex tasks. Few-shot CoT, first introduced by (Wei et al., 2022), enables CoT reasoning with only a few examples, significantly boosting performance in tasks such as arithmetic and symbolic reasoning. Kojima et al. (2022) proposed zero-shot CoT, which initiates the reasoning process using the prompt “Let’s think step by step”, enabling CoT reasoning without requiring labeled examples. In addition, numerous CoT variants have emerged, including Auto-CoT (Zhang et al., 2022), ToT (Yao et al., 2024), and Coconut (Hao et al., 2024), each designed to further enhance the generalization and effectiveness of CoT reasoning across various domains. In addition, CoT is also widely used (Shao et al., 2024; Shang et al., 2024b; Chen et al., 2024; Meng et al., 2025).

However, recent studies have demonstrated that CoT reasoning exhibits inconsistent performance across different benchmarks (Sprague et al., 2024). While CoT significantly enhances mathematical and symbolic reasoning tasks, its impact on commonsense reasoning and factual question answering remains limited (Kojima et al., 2022). CoT is particularly effective for structured, decomposable tasks but offers minimal improvement in tasks that require external knowledge or lack explicit reasoning steps. Furthermore, Liu et al. (2024) argues that CoT should not be indiscriminately applied to all tasks, as it significantly degrades the performance of LLMs in scenarios where excessive reasoning-akin to overthinking-detrimentally affects human performance. Although various techniques have been proposed to mitigate CoT's instability across tasks, such as Self-Consistency (Wang et al., 2022b), Program-of-Thought (Chen et al., 2022), Division-of-Thoughts (Shao et al., 2025), and Synergy-of-Thoughts (Shang et al., 2024a) the underlying principles governing CoT remain an open research question actively explored by the community.

# 7.2. Decoding for Large Language Models

The decoding process plays a crucial role in LLMs. Popular algorithms such as greedy decoding, temperature sampling (Ficler & Goldberg, 2017), top-k sampling (Radford et al., 2019), and diverse beam search (Vijayakumar et al., 2018) are often employed to enhance the quality of generated responses. In addition, Li et al. (2022) proposed a new decoding strategy called Contrastive Decoding to enhance output quality in open-ended text generation tasks. Shi et al. (2023) introduced Context-Aware Decoding, which focuses on reducing hallucination during the generation process. Recent study (Wang & Zhou, 2024) has shown that the CoT path can be spontaneously generated in the decoding path. By leveraging the information in the decoding process, the high confidence of the answer produced by the model can be used to find the CoT path without explicit prompt words (Wang & Zhou, 2024). While most existing studies primarily rely on next-token prediction, there has been limited exploration from the perspective of the overall token probability distribution. This gap is addressed in our work.

# 8. Conclusion

In this paper, we propose a novel perspective on LLM decoding by focusing on the token probability distribution rather than the most probable next token to analyze CoT reasoning. We introduce the concept of Token Signature to explore the correlation between the token probability distribution and CoT gains. Based on this insight, we develop two metrics, Instance SC and Aggregated SC, which effectively quantify CoT effectiveness at both the instance and benchmark levels. We further design Dynamic CoT, combining instance-level SC, to dynamically select between CoT and direct answers.

Extensive experiments across 12 widely used benchmarks validate the effectiveness of our approach on both open-source and closed-source models. At the benchmark level, our metrics predict the applicability of CoT with high accuracy, achieving 69.6% for Instance SC and 89.2% for Aggregated SC. At the question level, Dynamic CoT achieves performance comparable to the best results between CoT and direct answers while reducing token consumption by 39.1%. In transfer experiments, we observe that Dynamic CoT maintains high accuracy and further reduces token consumption by 35.8%.

Overall, we leverage token signature to assess CoT effectiveness and introduce a powerful mechanism for dynamically selecting the most effective answer strategy, while minimizing computational cost. Future work can explore extending the token signature concept, providing deeper insights into the CoT mechanism, and contributing to the development of more efficient large language models.

# Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under 23IAA02114, 62472241, and Beijing National Research Center for Information Science and Technology.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Abdin, M., Aneja, J., Awadalla, H., Awadallah, A., Awan, A. A., Bach, N., Bahree, A., Bakhtiari, A., Bao, J., Behl, H., et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.   
Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.   
Chae, H., Kim, Y., Kim, S., Ong, K. T.-i., Kwak, B.-w., Kim, M., Kim, S., Kwon, T., Chung, J., Yu, Y., et al. Language models as compilers: Simulating pseudocode execution improves algorithmic reasoning in language models. arXiv preprint arXiv:2404.02575, 2024.   
Chen, L., Xu, F., Li, N., Han, Z., Wang, M., Li, Y., and Hui, P. Large language model-driven meta-structure discovery in heterogeneous information network. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 307–318, 2024.   
Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.   
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.   
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.   
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.   
Ficler, J. and Goldberg, Y. Controlling linguistic style aspects in neural language generation. arXiv preprint arXiv:1707.02633, 2017.   
Gan, Z., Liao, Y., and Liu, Y. Rethinking external slow-thinking: From snowball errors to probability of correct reasoning. arXiv preprint arXiv:2501.15602, 2025.   
Geva, M., Khashabi, D., Segal, E., Khot, T., Roth, D., and Berant, J. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346–361, 2021.   
Han, S., Schoelkopf, H., Zhao, Y., Qi, Z., Riddell, M., Zhou, W., Coady, J., Peng, D., Qiao, Y., Benson, L., et al. Folio: Natural language reasoning with first-order logic. arXiv preprint arXiv:2209.00840, 2022.   
Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., and Tian, Y. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.   
Hua, W., Zhu, K., Li, L., Fan, L., Lin, S., Jin, M., Xue, H., Li, Z., Wang, J., and Zhang, Y. Disentangling logic: The role of context in large language model reasoning capabilities. arXiv preprint arXiv:2406.02787, 2024.   
Ivison, H., Wang, Y., Pyatkin, V., Lambert, N., Peters, M., Dasigi, P., Jang, J., Wadden, D., Smith, N. A., Beltagy, I., et al. Camels in a changing climate: Enhancing lm adaptation with tulu 2. arXiv preprint arXiv:2311.10702, 2023.   
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.   
Kambhampati, S., Valmeekam, K., Guan, L., Verma, M., Stechly, K., Bhambri, S., Saldyt, L., and Murthy, A. Llms can't plan, but can help planning in llm-modulo frameworks. arXiv preprint arXiv:2402.01817, 2024.   
Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.   
Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., Hernandez, D., Li, D., Durmus, E., Hubinger, E., Kernion, J., et al. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702, 2023.

Li, X. L., Holtzman, A., Fried, D., Liang, P., Eisner, J., Hashimoto, T., Zettlemoyer, L., and Lewis, M. Contrastive decoding: Open-ended text generation as optimization. arXiv preprint arXiv:2210.15097, 2022.   
Li, Z., Liu, H., Zhou, D., and Ma, T. Chain of thought empowers transformers to solve inherently serial problems. arXiv preprint arXiv:2402.12875, 1, 2024.   
Liu, R., Geng, J., Wu, A. J., Sucholutsky, I., Lombrozo, T., and Griffiths, T. L. Mind your step (by step): Chain-of-thought can reduce performance on tasks where thinking makes humans worse. arXiv preprint arXiv:2410.21333, 2024.   
Meng, F., Ding, J., Gong, J., Yang, C., Chen, H., Wang, Z., Lu, H., and Li, Y. Tuning language models for robust prediction of diverse user behaviors. arXiv preprint arXiv:2505.17682, 2025.   
OpenAI. Learning to reason with llms, 2024a.
URL https://openai.com/index/learning-to-reason-with-llms/.   
OpenAI. Hello gpt-4o, 2024b. URL https://openai.com/index/hello-gpt-4o/.   
Qi, Z., Ma, M., Xu, J., Zhang, L. L., Yang, F., and Yang, M. Mutual reasoning makes smaller llms stronger problem-solvers. arXiv preprint arXiv:2408.06195, 2024.   
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.   
Roy, S. and Roth, D. Solving general arithmetic word problems. arXiv preprint arXiv:1608.01413, 2016.   
Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.   
Shang, Y., Li, Y., Xu, F., and Li, Y. Synergy-of-thoughts: Eliciting efficient reasoning in hybrid language models. arXiv preprint arXiv:2402.02563, 2024a.   
Shang, Y., Li, Y., Zhao, K., Ma, L., Liu, J., Xu, F., and Li, Y. Agentsquare: Automatic llm agent search in modular design space. arXiv preprint arXiv:2410.06153, 2024b.   
Shao, C., Xu, F., Fan, B., Ding, J., Yuan, Y., Wang, M., and Li, Y. Chain-of-planned-behaviour workflow elicits few-shot mobility generation in llms. arXiv preprint arXiv:2402.09836, 2024.

Shao, C., Hu, X., Lin, Y., and Xu, F. Division-of-thoughts: Harnessing hybrid language model synergy for efficient on-device agents. In Proceedings of the ACM on Web Conference 2025, pp. 1822–1833, 2025.   
Shi, W., Han, X., Lewis, M., Tsvetkov, Y., Zettlemoyer, L., and Yih, S. W.-t. Trusting your evidence: Hallucinate less with context-aware decoding. arXiv preprint arXiv:2305.14739, 2023.   
Sprague, Z., Ye, X., Bostrom, K., Chaudhuri, S., and Durrett, G. Musr: Testing the limits of chain-of-thought with multistep soft reasoning. arXiv preprint arXiv:2310.16049, 2023.   
Sprague, Z., Yin, F., Rodriguez, J. D., Jiang, D., Wadhwa, M., Singhal, P., Zhao, X., Ye, X., Mahowald, K., and Durrett, G. To cot or not to cot? chain-of-thought helps mainly on math and symbolic reasoning. arXiv preprint arXiv:2409.12183, 2024.   
Talmor, A., Herzig, J., Lourie, N., and Berant, J. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937, 2018.   
Vijayakumar, A., Cogswell, M., Selvaraju, R., Sun, Q., Lee, S., Crandall, D., and Batra, D. Diverse beam search for improved description of complex scenes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.   
Wang, B., Min, S., Deng, X., Shen, J., Wu, Y., Zettlemoyer, L., and Sun, H. Towards understanding chain-of-thought prompting: An empirical study of what matters. arXiv preprint arXiv:2212.10001, 2022a.   
Wang, X. and Zhou, D. Chain-of-thought reasoning without prompting. arXiv preprint arXiv:2402.10200, 2024.   
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022b.   
Wang, Y., Ivison, H., Dasigi, P., Hessel, J., Khot, T., Chandu, K., Wadden, D., MacMillan, K., Smith, N. A., Beltagy, I., et al. How far can camels go? exploring the state of instruction tuning on open resources. Advances in Neural Information Processing Systems, 36:74764–74786, 2023.   
Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
Wissler, C. The spearman correlation formula. Science, 22(558):309–311, 1905.   
Xu, F., Hao, Q., Zong, Z., Wang, J., Zhang, Y., Wang, J., Lan, X., Gong, J., Ouyang, T., Meng, F., et al. Towards large reasoning models: A survey of reinforced reasoning with large language models. arXiv preprint arXiv:2501.09686, 2025.   
Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024.   
Zhang, Z., Zhang, A., Li, M., and Smola, A. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493, 2022.   
Zhong, W., Cui, R., Guo, Y., Liang, Y., Lu, S., Wang, Y., Saied, A., Chen, W., and Duan, N. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.

# A. Implementation Details

Benchmark We use 12 widely used benchmarks, which are described in Table 6. Among them, We use the abductive and deductive data of level = 2 in ContextHub, represented by CH\_a and CH\_b. We use the challenge and easy data in ARC, represented by Arc\_chall and Arc\_easy.

Table 6: Introduction to the benchmark used in the paper. 

<table><tr><td></td><td>Category</td><td>Answer Format</td><td>Number</td><td>Brief Description</td></tr><tr><td>GSM8K(Cobbe et al., 2021)</td><td>Mathematical</td><td>Short Answer</td><td>1319</td><td>A dataset containing high-quality and diverse elementary school math word problems, designed to evaluate the mathematical reasoning ability in the model.</td></tr><tr><td>MultiArith(Roy &amp; Roth, 2016)</td><td>Mathematical</td><td>Short Answer</td><td>600</td><td>A benchmark focused on multi-step arithmetic reasoning tasks that require models to solve math word problems involving basic operations.</td></tr><tr><td>FOLIO(Han et al., 2022)</td><td>Symbolic</td><td>True/False</td><td>1204</td><td>A dataset designed to test models on reasoning with first-order logic statements and deriving logical conclusions.</td></tr><tr><td>CH_a(Hua et al., 2024)</td><td>Symbolic</td><td>True/False</td><td>2400</td><td>A benchmark for testing abductive reasoning, where models must infer the most plausible explanation given a scenario.</td></tr><tr><td>CH_d(Hua et al., 2024)</td><td>Symbolic</td><td>True/False</td><td>2400</td><td>A benchmark focused on deductive reasoning tasks, requiring models to derive conclusions logically based on premises.</td></tr><tr><td>Arc_chall(Clark et al., 2018)</td><td>Knowledge</td><td>Multiple choice</td><td>1172</td><td>A challenging subset of the AI2 Reasoning Challenge (ARC) that contains difficult science questions for models to solve.</td></tr><tr><td>Arc_easy(Clark et al., 2018)</td><td>Knowledge</td><td>Multiple choice</td><td>2376</td><td>An easier subset of the AI2 Reasoning Challenge (ARC) designed to evaluate basic science knowledge and reasoning.</td></tr><tr><td>GPQA(Rein et al., 2023)</td><td>Knowledge</td><td>Multiple choice</td><td>448</td><td>The General Physics Question Answering benchmark assesses a model&#x27;s ability to answer questions about fundamental physics concepts.</td></tr><tr><td>MuSR (Sprague et al., 2023)</td><td>Soft Reasoning</td><td>Multiple choice</td><td>756</td><td>A benchmark for evaluating Multistep Symbolic Reasoning, where models must solve tasks involving symbolic manipulation.</td></tr><tr><td>LSAT(Zhong et al., 2023)</td><td>Soft Reasoning</td><td>Multiple choice</td><td>1009</td><td>Based on the Law School Admission Test, this benchmark tests logical reasoning and reading comprehension.</td></tr><tr><td>CSQA(Talmor et al., 2018)</td><td>Commonsense</td><td>Multiple choice</td><td>1221</td><td>A benchmark to evaluate models&#x27; commonsense reasoning by answering questions that require real-world understanding.</td></tr><tr><td>PIQA(Bisk et al., 2020)</td><td>Commonsense</td><td>Multiple choice</td><td>1838</td><td>The Physical Interaction Question Answering benchmark focuses on models&#x27; understanding of everyday physical commonsense.</td></tr><tr><td>SIQA (Sap et al., 2019)</td><td>Commonsense</td><td>Multiple choice</td><td>1954</td><td>Social Interaction Question Answering, testing models&#x27; ability to reason about social situations and motivations.</td></tr><tr><td>StrategyQA(Geva et al., 2021)</td><td>Commonsense</td><td>True/False</td><td>1508</td><td>A benchmark designed for multi-step reasoning tasks where models must strategically reason to answer open-ended questions.</td></tr></table>

Detailed Prompt Setting For standard prompts, we carefully craft prompts for all benchmarks. For benchmarks with a True/False answer format, we restructure them as multiple-choice questions. For Zero-shot CoT prompts, we use the phrase “Let’s think step by step” (Kojima et al., 2022) to trigger the CoT reasoning process. For direct answer prompts, we meticulously design them to ensure the model adheres to instructions. For benchmarks requiring short answers, we use the directive: “Your answer must not include any reasoning step. You must only write your numerical answer directly. You only output ‘The answer is <answer>’ where <answer> is the numerical answer to the problem,” as the DA trigger. For multiple-choice benchmarks, we employ: “Your answer must not include any reasoning. You must write your answer directly. Write the answer in the following format: ‘Answer: <Your Answer Letter Choice>’” as the DA trigger.

The open-source models utilized in our work are all instruction fine-tuned, and therefore, we employ different placeholders to encapsulate prompt tokens for each model. For Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct (Dubey et al., 2024), we adopt ‘<|begin\_of\_text|><|start\_header\_id|>user<|end\_header\_id|>’ and ‘<|eot\_id|><|start\_header\_id|>assistant<|end\_header\_id|’. For Phi-3.5-mini-instruct (Abdin et al., 2024), we adopt ‘<|user|>’ and ‘<|end|><|assistant|>’. For Mistral-7B-Instruct-v0.3 (Jiang et al., 2023), we adopt ‘[INST]’ and ‘[/INST]’.

Detailed Experimental Parameter Setting We aim to maintain consistent experimental parameters across different models for the same benchmark. In general, we set do\_sample = False or temperature = 0 to ensure greedy decoding. For standard prompt experiments, the maximum number of generated tokens is set to 50 (max\_tokens=50); for CoT experiments, it is set to 1024 (max\_tokens=1024); and for direct answer experiments, it is set to 32 (max\_tokens=32). Minor adjustments are made in some experiments as needed.

Significance Judgment We use a two-tailed Z test to assess the significance of the difference between CoT Acc and DA Acc. For a specific benchmark, let $p_{1}$ and $p_{2}$ represent the accuracy rates under DA and CoT, respectively, and let $n_{1}$ and $n_{2}$ represent the sample sizes, both of which are equal to N, as detailed in the Table 6. The null hypothesis assumes no significant difference between the two values, i.e., $p_{1} = p_{2}$ . The alternative hypothesis tests whether the difference $p_{2} - p_{1}$ is significantly different, i.e., $p_{2} \neq p_{1}$ . The Z test statistic is calculated using the following formula:

$$
Z = \frac {(p _ {2} - p _ {1})}{\sqrt {\frac {p _ {0} (1 - p _ {0})}{n _ {1}} + \frac {p _ {0} (1 - p _ {0})}{n _ {2}}}},
$$

where $p_0$ is the pooled proportion, computed as:

$$
p _ {0} = \frac {p _ {1} n _ {1} + p _ {2} n _ {2}}{n _ {1} + n _ {2}}.
$$

The p-value for a two-sided test is calculated as follows:

$$
p = 2 \cdot P (z > | Z |).
$$

Under the null hypothesis, the test statistic Z follows a standard normal distribution. If the absolute value of the calculated Z-score exceeds the critical value associated with the desired significance level (typically 1.96 for a 95% confidence level) or if the p-value is less than 0.05, we reject the null hypothesis. This indicates that the observed difference is statistically significant. Specifically, if $|Z|$ surpasses the threshold for a 95% confidence level or p < 0.05, we conclude that $p_{2}$ represents a statistically significant change from $p_{1}$ . Conversely, if these conditions are not met, we fail to reject the null hypothesis, suggesting that there is no significant improvement in CoT. Specifically, the significance of the result is determined as follows:

$$
\text { Significance } = \left\{ \begin{array}{l l} \text { positive }, & \text { if   } Z > 0 \text {   and   } p <   0. 0 5, \\ \text { none }, & \text { if   } Z = 0 \text {   or   } p \geq 0. 0 5, \\ \text { negative }, & \text { if   } Z <   0 \text {   and   } p <   0. 0 5. \end{array} \right.
$$

Compute Resources We deploy four open-source models for inference on an A100 GPU with 80 GB RAM. Each experiment for each benchmark takes from a few minutes to a few hours, depending on the number of questions and the experiment type (Standard/CoT/Direct answer). For experiments on closed-source models, we use the official API interface of OpenAI $^{1}$ .

# B. Supplementary experimental results

This section presents the supplementary experimental results from the paper. Specifically, Table 7 provides the CoT accuracy, direct answer accuracy, CoT gain, and significance judgment across 12 benchmarks for 4 open-source models and 2 closed-source models. Table 8 displays the experimental results for Instance SC and Aggregated SC on three models not covered in the main text. Table 9 and Table 10 outline the performance of Dynamic CoT on four models, comparing it to both CoT and direct answer approaches. Finally, Table 11 details the transfer of Dynamic CoT to a closed-source model and compares its performance with CoT and direct answer.

Table 7: Zero-shot CoT, direct answer accuracy and significance judgment on different benchmarks and different models 

<table><tr><td></td><td>Model</td><td>CoT Acc</td><td>DA Acc</td><td>CoT Gain</td><td>Z Statistic</td><td>p Value</td><td>Significance</td></tr><tr><td rowspan="6">GSM8K</td><td>Llama-3.2-3B-Instruct</td><td>72.10</td><td>8.34</td><td>63.76</td><td>33.39</td><td>0.000</td><td>positive</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>49.58</td><td>8.34</td><td>41.24</td><td>23.35</td><td>0.000</td><td>positive</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>78.24</td><td>18.35</td><td>59.89</td><td>30.78</td><td>0.000</td><td>positive</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>81.20</td><td>14.25</td><td>66.95</td><td>34.42</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o-mini</td><td>84.38</td><td>31.01</td><td>53.37</td><td>27.74</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o</td><td>85.14</td><td>53.90</td><td>64.83</td><td>17.43</td><td>0.000</td><td>positive</td></tr><tr><td rowspan="6">MultiArith</td><td>Llama-3.2-3B-Instruct</td><td>88.50</td><td>20.67</td><td>67.83</td><td>23.60</td><td>0.000</td><td>positive</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>77.67</td><td>14.33</td><td>63.34</td><td>22.01</td><td>0.000</td><td>positive</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>85.17</td><td>44.33</td><td>40.84</td><td>14.81</td><td>0.000</td><td>positive</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>91.00</td><td>41.50</td><td>49.50</td><td>18.13</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o-mini</td><td>94.33</td><td>94.83</td><td>-0.5</td><td>-0.38</td><td>0.702</td><td>none</td></tr><tr><td>GPT-4o</td><td>93.50</td><td>98.00</td><td>-4.5</td><td>-3.86</td><td>0.000</td><td>negative</td></tr><tr><td rowspan="6">FOLIO</td><td>Llama-3.2-3B-Instruct</td><td>48.92</td><td>42.61</td><td>6.31</td><td>3.11</td><td>0.002</td><td>positive</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>49.42</td><td>51.00</td><td>-1.58</td><td>-0.78</td><td>0.438</td><td>none</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>59.14</td><td>53.32</td><td>5.82</td><td>2.88</td><td>0.004</td><td>positive</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>58.22</td><td>49.42</td><td>8.80</td><td>4.33</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o-mini</td><td>69.19</td><td>62.21</td><td>6.98</td><td>3.61</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o</td><td>75.08</td><td>69.60</td><td>5.48</td><td>3.00</td><td>0.003</td><td>positive</td></tr><tr><td rowspan="6">CH_a</td><td>Llama-3.2-3B-Instruct</td><td>35.21</td><td>33.54</td><td>1.67</td><td>1.22</td><td>0.223</td><td>none</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>32.37</td><td>30.17</td><td>2.20</td><td>1.64</td><td>0.100</td><td>none</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>40.29</td><td>33.58</td><td>6.71</td><td>4.82</td><td>0.000</td><td>positive</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>40.21</td><td>30.50</td><td>9.71</td><td>7.03</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o-mini</td><td>56.92</td><td>46.25</td><td>10.67</td><td>7.40</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o</td><td>59.42</td><td>42.13</td><td>17.29</td><td>11.98</td><td>0.000</td><td>positive</td></tr><tr><td rowspan="6">CH_d</td><td>Llama-3.2-3B-Instruct</td><td>39.87</td><td>19.92</td><td>19.95</td><td>15.10</td><td>0.0</td><td>positive</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>55.92</td><td>59.04</td><td>-3.12</td><td>-2.19</td><td>0.029</td><td>negative</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>53.25</td><td>60.04</td><td>-6.79</td><td>-4.75</td><td>0.000</td><td>negative</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>57.87</td><td>37.79</td><td>20.08</td><td>13.92</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o-mini</td><td>53.92</td><td>43.67</td><td>10.25</td><td>7.10</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o</td><td>60.42</td><td>53.00</td><td>7.42</td><td>5.19</td><td>0.000</td><td>positive</td></tr><tr><td rowspan="6">Arc_chall</td><td>Llama-3.2-3B-Instruct</td><td>73.89</td><td>69.88</td><td>4.01</td><td>2.16</td><td>0.031</td><td>positive</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>76.45</td><td>73.63</td><td>2.82</td><td>1.58</td><td>0.115</td><td>none</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>86.52</td><td>83.36</td><td>3.16</td><td>2.14</td><td>0.032</td><td>positive</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>85.32</td><td>79.52</td><td>5.80</td><td>3.69</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o-mini</td><td>93.77</td><td>90.61</td><td>3.16</td><td>2.85</td><td>0.004</td><td>positive</td></tr><tr><td>GPT-4o</td><td>94.45</td><td>93.77</td><td>0.68</td><td>0.70</td><td>0.484</td><td>none</td></tr><tr><td rowspan="6">Arc_easy</td><td>Llama-3.2-3B-Instruct</td><td>84.39</td><td>83.46</td><td>0.93</td><td>0.87</td><td>0.383</td><td>none</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>86.99</td><td>84.68</td><td>2.31</td><td>2.28</td><td>0.022</td><td>positive</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>91.71</td><td>90.78</td><td>0.93</td><td>1.13</td><td>0.257</td><td>none</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>89.73</td><td>88.64</td><td>1.09</td><td>1.21</td><td>0.226</td><td>none</td></tr><tr><td>GPT-4o-mini</td><td>94.07</td><td>94.07</td><td>0.00</td><td>0.0</td><td>1.0</td><td>none</td></tr><tr><td>GPT-4o</td><td>94.53</td><td>94.49</td><td>0.04</td><td>0.06</td><td>0.952</td><td>none</td></tr></table>

Continued on next page

Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models 

<table><tr><td></td><td>Model</td><td>CoT Acc</td><td>DA Acc</td><td>CoT Gain</td><td>Z Statistic</td><td>p Value</td><td>Significance</td></tr><tr><td rowspan="6">GPQA</td><td>Llama-3.2-3B-Instruct</td><td>27.01</td><td>37.72</td><td>-10.71</td><td>-3.43</td><td>0.001</td><td>negative</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>30.58</td><td>34.60</td><td>-4.02</td><td>-1.28</td><td>0.199</td><td>none</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>23.21</td><td>16.29</td><td>6.92</td><td>2.60</td><td>0.009</td><td>positive</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>28.79</td><td>32.14</td><td>-3.35</td><td>-1.09</td><td>0.276</td><td>none</td></tr><tr><td>GPT-4o-mini</td><td>46.88</td><td>42.19</td><td>4.69</td><td>-1.41</td><td>0.157</td><td>none</td></tr><tr><td>GPT-4o</td><td>61.16</td><td>49.33</td><td>11.83</td><td>3.56</td><td>0.000</td><td>positive</td></tr><tr><td rowspan="6">MuSR</td><td>Llama-3.2-3B-Instruct</td><td>44.84</td><td>50.93</td><td>-6.09</td><td>-2.37</td><td>0.018</td><td>negative</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>45.90</td><td>47.35</td><td>-1.45</td><td>-0.57</td><td>0.572</td><td>none</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>54.37</td><td>55.69</td><td>-1.32</td><td>-0.52</td><td>0.606</td><td>none</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>50.13</td><td>51.72</td><td>-1.59</td><td>-0.62</td><td>0.536</td><td>none</td></tr><tr><td>GPT-4o-mini</td><td>58.60</td><td>55.82</td><td>2.78</td><td>1.09</td><td>0.275</td><td>none</td></tr><tr><td>GPT-4o</td><td>62.96</td><td>59.39</td><td>3.57</td><td>1.42</td><td>0.154</td><td>none</td></tr><tr><td rowspan="6">LSAT</td><td>Llama-3.2-3B-Instruct</td><td>39.74</td><td>40.73</td><td>-0.99</td><td>-0.45</td><td>0.650</td><td>none</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>45.00</td><td>46.58</td><td>-1.58</td><td>-0.71</td><td>0.473</td><td>none</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>47.97</td><td>48.07</td><td>-0.10</td><td>-0.045</td><td>0.964</td><td>none</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>47.67</td><td>51.04</td><td>-3.37</td><td>-1.51</td><td>0.130</td><td>none</td></tr><tr><td>GPT-4o-mini</td><td>61.94</td><td>64.02</td><td>-2.08</td><td>-0.97</td><td>0.333</td><td>none</td></tr><tr><td>GPT-4o</td><td>75.92</td><td>72.65</td><td>3.27</td><td>1.68</td><td>0.093</td><td>none</td></tr><tr><td rowspan="6">CSQA</td><td>Llama-3.2-3B-Instruct</td><td>67.24</td><td>65.60</td><td>1.64</td><td>0.86</td><td>0.391</td><td>none</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>70.93</td><td>69.12</td><td>1.81</td><td>0.98</td><td>0.329</td><td>none</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>71.66</td><td>73.38</td><td>-1.72</td><td>-0.95</td><td>0.341</td><td>none</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>74.37</td><td>73.30</td><td>1.07</td><td>0.60</td><td>0.548</td><td>none</td></tr><tr><td>GPT-4o-mini</td><td>81.41</td><td>81.90</td><td>-0.49</td><td>-0.31</td><td>0.754</td><td>none</td></tr><tr><td>GPT-4o</td><td>84.11</td><td>84.68</td><td>-0.57</td><td>-0.39</td><td>0.698</td><td>none</td></tr><tr><td rowspan="6">PIQA</td><td>Llama-3.2-3B-Instruct</td><td>77.42</td><td>75.68</td><td>1.74</td><td>1.24</td><td>0.213</td><td>none</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>80.20</td><td>81.77</td><td>-1.57</td><td>-1.21</td><td>0.225</td><td>none</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>81.99</td><td>83.41</td><td>-1.42</td><td>-1.14</td><td>0.255</td><td>none</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>84.11</td><td>82.05</td><td>2.06</td><td>1.67</td><td>0.096</td><td>none</td></tr><tr><td>GPT-4o-mini</td><td>88.85</td><td>89.77</td><td>-0.92</td><td>-0.90</td><td>0.367</td><td>none</td></tr><tr><td>GPT-4o</td><td>91.08</td><td>93.63</td><td>-2.55</td><td>-2.91</td><td>0.004</td><td>negative</td></tr><tr><td rowspan="6">SIQA</td><td>Llama-3.2-3B-Instruct</td><td>62.69</td><td>62.33</td><td>0.36</td><td>0.23</td><td>0.816</td><td>none</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>62.64</td><td>63.56</td><td>-0.92</td><td>-0.60</td><td>0.551</td><td>none</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>70.57</td><td>70.01</td><td>0.56</td><td>0.38</td><td>0.702</td><td>none</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>69.50</td><td>68.68</td><td>0.82</td><td>0.55</td><td>0.579</td><td>none</td></tr><tr><td>GPT-4o-mini</td><td>77.23</td><td>76.00</td><td>1.23</td><td>0.91</td><td>0.364</td><td>none</td></tr><tr><td>GPT-4o</td><td>77.74</td><td>78.10</td><td>-0.36</td><td>-0.27</td><td>0.786</td><td>none</td></tr><tr><td rowspan="6">StrategyQA</td><td>Llama-3.2-3B-Instruct</td><td>66.59</td><td>79.69</td><td>-13.1</td><td>-10.00</td><td>0.000</td><td>negative</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>86.16</td><td>82.93</td><td>3.23</td><td>3.02</td><td>0.002</td><td>positive</td></tr><tr><td>Phi-3.5-mini-instruct</td><td>75.46</td><td>73.89</td><td>1.57</td><td>1.22</td><td>0.222</td><td>none</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>71.88</td><td>76.55</td><td>-4.67</td><td>-3.61</td><td>0.000</td><td>negative</td></tr><tr><td>GPT-4o-mini</td><td>62.88</td><td>54.19</td><td>8.69</td><td>5.97</td><td>0.000</td><td>positive</td></tr><tr><td>GPT-4o</td><td>60.66</td><td>55.24</td><td>5.42</td><td>3.72</td><td>0.000</td><td>positive</td></tr></table>

Table 8: Instance SC, Aggregated SC, CoT Gain and CoT Gain Significance across benchmarks on Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct, Llama-3.1-8B-Instruct.   
Mistral-7B-Instruct-v0.3 

<table><tr><td></td><td>GSM8K</td><td>MultiArith</td><td>FOLIO</td><td>CH_a</td><td>CH_d</td><td>Arc_chall</td><td>Arc_easy</td></tr><tr><td>Instance SC</td><td>0.1193</td><td>0.0837</td><td>0.1925</td><td>-0.1410</td><td>-0.035</td><td>-0.1479</td><td>-0.1218</td></tr><tr><td>Aggregated SC</td><td>0.4285</td><td>0.2781</td><td>-0.0889</td><td>-0.4540</td><td>-0.509</td><td>-0.6940</td><td>-0.6640</td></tr><tr><td>CoT Gain</td><td>41.24</td><td>63.34</td><td>-1.58</td><td>2.20</td><td>-3.12</td><td>2.82</td><td>2.31</td></tr><tr><td>Significance</td><td>positive</td><td>positive</td><td>none</td><td>none</td><td>negative</td><td>none</td><td>positive</td></tr><tr><td></td><td>GPQA</td><td>MuSR</td><td>LSAT</td><td>CSQA</td><td>PIQA</td><td>SIQA</td><td>StrategyQA</td></tr><tr><td>Instance SC</td><td>0.0298</td><td>0.3173</td><td>0.1771</td><td>-0.0488</td><td>-0.0690</td><td>0.1708</td><td>-0.1409</td></tr><tr><td>Aggregated SC</td><td>-0.4995</td><td>-0.4520</td><td>-0.5528</td><td>-0.4241</td><td>-0.6690</td><td>-0.7730</td><td>-0.7609</td></tr><tr><td>CoT Gain</td><td>-4.02</td><td>-1.45</td><td>-1.58</td><td>1.81</td><td>-1.57</td><td>-0.92</td><td>3.23</td></tr><tr><td>Significance</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td><td>positive</td></tr></table>

Phi-3.5-mini-instruct 

<table><tr><td></td><td>GSM8K</td><td>MultiArith</td><td>FOLIO</td><td>CH_a</td><td>CH_d</td><td>Arc_chall</td><td>Arc_easy</td></tr><tr><td>Instance SC</td><td>0.3292</td><td>0.2699</td><td>0.0437</td><td>0.0230</td><td>0.0098</td><td>-0.1908</td><td>-0.2007</td></tr><tr><td>Aggregated SC</td><td>0.6560</td><td>0.6160</td><td>0.2386</td><td>0.1460</td><td>-0.0385</td><td>-0.7586</td><td>-0.7571</td></tr><tr><td>CoT Gain</td><td>59.89</td><td>40.84</td><td>5.82</td><td>10.71</td><td>-6.79</td><td>3.16</td><td>0.93</td></tr><tr><td>Significance</td><td>positive</td><td>positive</td><td>positive</td><td>positive</td><td>negative</td><td>positive</td><td>none</td></tr><tr><td></td><td>GPQA</td><td>MuSR</td><td>LSAT</td><td>CSQA</td><td>PIQA</td><td>SIQA</td><td>StrategyQA</td></tr><tr><td>Instance SC</td><td>0.0015</td><td>-0.0534</td><td>0.0126</td><td>-0.2399</td><td>-0.0839</td><td>-0.0734</td><td>-0.1905</td></tr><tr><td>Aggregated SC</td><td>-0.3629</td><td>-0.4933</td><td>-0.4341</td><td>-0.8655</td><td>-0.6396</td><td>-0.5340</td><td>-0.7711</td></tr><tr><td>CoT Gain</td><td>6.92</td><td>-1.32</td><td>-0.10</td><td>-1.72</td><td>-1.42</td><td>0.56</td><td>1.57</td></tr><tr><td>Significance</td><td>positive</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td></tr></table>

Llama-3.1-8B-Instruct 

<table><tr><td></td><td>GSM8K</td><td>MultiArith</td><td>FOLIO</td><td>CH_a</td><td>CH_d</td><td>Arc_chall</td><td>Arc_easy</td></tr><tr><td>Instance SC</td><td>0.0015</td><td>0.1144</td><td>0.3419</td><td>0.236</td><td>0.094</td><td>0.2619</td><td>0.4176</td></tr><tr><td>Aggregated SC</td><td>0.2469</td><td>0.3061</td><td>0.5058</td><td>0.107</td><td>0.042</td><td>-0.5972</td><td>-0.5449</td></tr><tr><td>CoT Gain</td><td>66.95</td><td>49.50</td><td>8.80</td><td>9.71</td><td>20.08</td><td>5.80</td><td>1.09</td></tr><tr><td>Significance</td><td>positive</td><td>positive</td><td>positive</td><td>positive</td><td>positive</td><td>positive</td><td>none</td></tr><tr><td></td><td>GPQA</td><td>MuSR</td><td>LSAT</td><td>CSQA</td><td>PIQA</td><td>SIQA</td><td>StrategyQA</td></tr><tr><td>Instance SC</td><td>0.3395</td><td>-0.0182</td><td>0.1033</td><td>-0.0156</td><td>0.4174</td><td>-0.0351</td><td>-0.1669</td></tr><tr><td>Aggregated SC</td><td>-0.3422</td><td>-0.3472</td><td>-0.2613</td><td>-0.4229</td><td>-0.1907</td><td>-0.6066</td><td>-0.1714</td></tr><tr><td>CoT Gain</td><td>-3.35</td><td>-1.59</td><td>-3.37</td><td>1.07</td><td>2.06</td><td>0.82</td><td>-4.67</td></tr><tr><td>Significance</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td><td>negative</td></tr></table>

Table 9: Accuracy and token consumption of CoT, DA and Dynamic CoT experiments on Llama-3.2-3B-Instruct and Mistral-7B-Instruct-v0.3   
Llama-3.2-3B-Instruct 

<table><tr><td></td><td>CoT Acc</td><td>DA Acc</td><td>Dynamic CoT</td><td>CoT Tokens</td><td>DA Tokens</td><td>Dynamic CoT Tokens</td></tr><tr><td>GSM8K</td><td>72.10</td><td>8.34</td><td>71.24</td><td>217.82</td><td>5.09</td><td>218.11</td></tr><tr><td>MultiArith</td><td>88.50</td><td>20.67</td><td>87.64</td><td>114.38</td><td>4.59</td><td>114.90</td></tr><tr><td>FOLIO</td><td>48.92</td><td>42.61</td><td>48.27</td><td>330.34</td><td>3.88</td><td>329.58</td></tr><tr><td>CH_a</td><td>35.21</td><td>33.54</td><td>32.26</td><td>250.42</td><td>3.53</td><td>45.862</td></tr><tr><td>CH_d</td><td>39.87</td><td>19.92</td><td>39.02</td><td>256.69</td><td>3.89</td><td>256.17</td></tr><tr><td>Arc_chall</td><td>73.89</td><td>69.88</td><td>74.51</td><td>238.37</td><td>4.03</td><td>238.07</td></tr><tr><td>Arc_easy</td><td>84.39</td><td>83.46</td><td>84.31</td><td>231.37</td><td>4.07</td><td>84.76</td></tr><tr><td>GPQA</td><td>27.01</td><td>37.72</td><td>34.42</td><td>729.99</td><td>4.03</td><td>4.03</td></tr><tr><td>MuSR</td><td>44.84</td><td>50.93</td><td>50.14</td><td>170.98</td><td>4.06</td><td>4.04</td></tr><tr><td>LSAT</td><td>39.74</td><td>40.73</td><td>40.77</td><td>386.84</td><td>4.01</td><td>128.73</td></tr><tr><td>CSQA</td><td>67.24</td><td>65.60</td><td>66.27</td><td>186.32</td><td>5.32</td><td>73.43</td></tr><tr><td>PIQA</td><td>77.42</td><td>75.68</td><td>77.68</td><td>178.08</td><td>4.94</td><td>178.47</td></tr><tr><td>SIQA</td><td>62.69</td><td>62.33</td><td>62.76</td><td>166.38</td><td>4.64</td><td>165.96</td></tr><tr><td>StrategyQA</td><td>66.59</td><td>79.69</td><td>79.87</td><td>212.67</td><td>4.28</td><td>4.28</td></tr></table>

Mistral-7B-Instruct-v0.3 

<table><tr><td></td><td>CoT Acc</td><td>DA Acc</td><td>Dynamic CoT</td><td>CoT Tokens</td><td>DA Tokens</td><td>Dynamic CoT Tokens</td></tr><tr><td>GSM8K</td><td>49.58</td><td>8.34</td><td>47.83</td><td>241.32</td><td>8.60</td><td>242.04</td></tr><tr><td>MultiArith</td><td>77.67</td><td>14.33</td><td>76.00</td><td>157.25</td><td>7.49</td><td>158.24</td></tr><tr><td>FOLIO</td><td>49.42</td><td>51.00</td><td>50.69</td><td>113.75</td><td>7.75</td><td>7.77</td></tr><tr><td>CH_a</td><td>32.37</td><td>30.17</td><td>29.66</td><td>129.33</td><td>11.56</td><td>11.60</td></tr><tr><td>CH_d</td><td>55.92</td><td>59.04</td><td>55.87</td><td>125.95</td><td>9.95</td><td>125.77</td></tr><tr><td>Arc_chall</td><td>76.45</td><td>73.63</td><td>77.01</td><td>144.55</td><td>11.52</td><td>143.40</td></tr><tr><td>Arc_easy</td><td>86.99</td><td>84.68</td><td>87.49</td><td>119.36</td><td>9.41</td><td>118.47</td></tr><tr><td>GPQA</td><td>30.58</td><td>34.60</td><td>30.90</td><td>477.52</td><td>18.54</td><td>17.82</td></tr><tr><td>MuSR</td><td>45.90</td><td>47.35</td><td>46.32</td><td>220.77</td><td>9.69</td><td>65.74</td></tr><tr><td>LSAT</td><td>45.00</td><td>46.58</td><td>45.46</td><td>238.42</td><td>16.22</td><td>108.43</td></tr><tr><td>CSQA</td><td>70.93</td><td>69.12</td><td>71.31</td><td>112.47</td><td>5.67</td><td>110.77</td></tr><tr><td>PIQA</td><td>80.20</td><td>81.77</td><td>82.33</td><td>108.49</td><td>15.20</td><td>15.40</td></tr><tr><td>SIQA</td><td>62.64</td><td>63.56</td><td>63.97</td><td>111.61</td><td>8.28</td><td>31.24</td></tr><tr><td>StrategyQA</td><td>86.16</td><td>82.93</td><td>86.83</td><td>98.02</td><td>6.07</td><td>97.67</td></tr></table>

Table 10: Accuracy and token consumption of CoT, DA and Dynamic CoT experiments on Phi-3.5-mini-instruct and Llama-3.1-8B-Instruct   
Phi-3.5-mini-instruct 

<table><tr><td></td><td>CoT Acc</td><td>DA Acc</td><td>Dynamic CoT</td><td>CoT Tokens</td><td>DA Tokens</td><td>Dynamic CoT Tokens</td></tr><tr><td>GSM8K</td><td>78.24</td><td>18.35</td><td>77.54</td><td>444.11</td><td>13.78</td><td>445.44</td></tr><tr><td>MultiArith</td><td>85.17</td><td>44.33</td><td>85.27</td><td>217.34</td><td>12.89</td><td>213.23</td></tr><tr><td>FOLIO</td><td>59.14</td><td>53.32</td><td>59.19</td><td>282.26</td><td>4.03</td><td>282.03</td></tr><tr><td>CH_a</td><td>40.29</td><td>33.58</td><td>39.66</td><td>262.88</td><td>4.74</td><td>262.86</td></tr><tr><td>CH_d</td><td>53.25</td><td>60.04</td><td>60.17</td><td>258.51</td><td>6.16</td><td>14.58</td></tr><tr><td>Arc_chall</td><td>86.52</td><td>83.36</td><td>87.25</td><td>329.93</td><td>4.69</td><td>330.22</td></tr><tr><td>Arc_easy</td><td>91.71</td><td>90.78</td><td>92.30</td><td>320.30</td><td>4.68</td><td>320.26</td></tr><tr><td>GPQA</td><td>23.21</td><td>16.29</td><td>19.10</td><td>661.04</td><td>5.86</td><td>670.99</td></tr><tr><td>MuSR</td><td>53.37</td><td>55.69</td><td>56.09</td><td>543.52</td><td>4.18</td><td>214.08</td></tr><tr><td>LSAT</td><td>47.97</td><td>48.07</td><td>47.24</td><td>426.53</td><td>5.17</td><td>425.43</td></tr><tr><td>CSQA</td><td>71.66</td><td>73.38</td><td>73.70</td><td>306.96</td><td>4.32</td><td>184.70</td></tr><tr><td>PIQA</td><td>81.99</td><td>83.41</td><td>84.17</td><td>308.52</td><td>4.08</td><td>4.08</td></tr><tr><td>SIQA</td><td>70.57</td><td>70.01</td><td>70.01</td><td>308.74</td><td>4.97</td><td>308.19</td></tr><tr><td>StrategyQA</td><td>75.46</td><td>73.89</td><td>74.64</td><td>272.98</td><td>4.05</td><td>51.60</td></tr></table>

Llama-3.1-8B-Instruct 

<table><tr><td></td><td>CoT Acc</td><td>DA Acc</td><td>Dynamic CoT</td><td>CoT Tokens</td><td>DA Tokens</td><td>Dynamic CoT Tokens</td></tr><tr><td>GSM8K</td><td>81.20</td><td>14.25</td><td>80.54</td><td>218.91</td><td>11.51</td><td>218.14</td></tr><tr><td>MultiArith</td><td>91.00</td><td>41.50</td><td>91.09</td><td>118.16</td><td>7.01</td><td>117.93</td></tr><tr><td>FOLIO</td><td>58.22</td><td>49.42</td><td>58.06</td><td>323.22</td><td>4.56</td><td>322.66</td></tr><tr><td>CH_a</td><td>40.21</td><td>30.50</td><td>39.57</td><td>291.01</td><td>4.00</td><td>290.79</td></tr><tr><td>CH_d</td><td>57.87</td><td>37.79</td><td>57.19</td><td>242.38</td><td>4.00</td><td>241.49</td></tr><tr><td>Arc_chall</td><td>85.32</td><td>79.52</td><td>86.01</td><td>265.21</td><td>4.03</td><td>265.32</td></tr><tr><td>Arc_easy</td><td>89.73</td><td>88.64</td><td>90.50</td><td>230.45</td><td>4.07</td><td>194.21</td></tr><tr><td>GPQA</td><td>28.79</td><td>32.14</td><td>29.65</td><td>767.19</td><td>7.78</td><td>7.80</td></tr><tr><td>MuSR</td><td>50.13</td><td>51.72</td><td>51.42</td><td>299.08</td><td>4.00</td><td>4.00</td></tr><tr><td>LSAT</td><td>47.67</td><td>51.04</td><td>50.47</td><td>353.13</td><td>3.98</td><td>34.14</td></tr><tr><td>CSQA</td><td>74.37</td><td>73.30</td><td>74.89</td><td>224.82</td><td>4.04</td><td>224.50</td></tr><tr><td>PIQA</td><td>84.11</td><td>82.05</td><td>85.07</td><td>168.42</td><td>4.02</td><td>144.47</td></tr><tr><td>SIQA</td><td>69.50</td><td>68.68</td><td>69.80</td><td>198.46</td><td>4.01</td><td>185.58</td></tr><tr><td>StrategyQA</td><td>71.88</td><td>76.55</td><td>76.83</td><td>189.51</td><td>4.31</td><td>4.28</td></tr></table>

Table 11: Accuracy and token consumption of CoT, DA and Dynamic CoT(transfer) experiments on closed-source models   
GPT-4o-mini 

<table><tr><td></td><td>CoT Acc</td><td>DA Acc</td><td>Dynamic CoT</td><td>CoT Tokens</td><td>DA Tokens</td><td>Dynamic CoT Tokens</td></tr><tr><td>GSM8K</td><td>84.38</td><td>31.01</td><td>84.38</td><td>314.94</td><td>5.71</td><td>314.94</td></tr><tr><td>MultiArith</td><td>94.33</td><td>94.83</td><td>94.33</td><td>193.42</td><td>5.14</td><td>193.42</td></tr><tr><td>FOLIO</td><td>69.19</td><td>62.21</td><td>69.19</td><td>360.00</td><td>3.00</td><td>359.30</td></tr><tr><td>CH_a</td><td>56.92</td><td>46.25</td><td>56.92</td><td>360.33</td><td>3.01</td><td>360.16</td></tr><tr><td>CH_d</td><td>53.92</td><td>43.67</td><td>53.92</td><td>296.18</td><td>3.01</td><td>296.18</td></tr><tr><td>Arc_chall</td><td>93.77</td><td>90.61</td><td>93.77</td><td>255.05</td><td>3.06</td><td>255.05</td></tr><tr><td>Arc_easy</td><td>94.07</td><td>94.07</td><td>94.07</td><td>231.77</td><td>3.06</td><td>231.61</td></tr><tr><td>GPQA</td><td>46.88</td><td>42.19</td><td>46.43</td><td>667.46</td><td>3.58</td><td>9.75</td></tr><tr><td>MuSR</td><td>58.60</td><td>55.82</td><td>55.82</td><td>406.48</td><td>2.98</td><td>4.16</td></tr><tr><td>LSAT</td><td>61.94</td><td>64.02</td><td>63.33</td><td>488.74</td><td>3.01</td><td>234.38</td></tr><tr><td>CSQA</td><td>81.41</td><td>81.90</td><td>81.41</td><td>212.58</td><td>3.00</td><td>212.19</td></tr><tr><td>PIQA</td><td>88.85</td><td>89.77</td><td>88.79</td><td>208.04</td><td>2.99</td><td>150.36</td></tr><tr><td>SIQA</td><td>77.23</td><td>76.00</td><td>77.23</td><td>233.87</td><td>3.02</td><td>232.90</td></tr><tr><td>StrategyQA</td><td>62.88</td><td>54.19</td><td>54.24</td><td>206.40</td><td>3.00</td><td>3.17</td></tr></table>

GPT-4o 

<table><tr><td></td><td>CoT Acc</td><td>DA Acc</td><td>Dynamic CoT</td><td>CoT Tokens</td><td>DA Tokens</td><td>Dynamic CoT Tokens</td></tr><tr><td>GSM8K</td><td>85.14</td><td>53.90</td><td>85.14</td><td>314.56</td><td>5.86</td><td>314.56</td></tr><tr><td>MultiArith</td><td>93.50</td><td>98.00</td><td>93.50</td><td>185.56</td><td>5.54</td><td>185.56</td></tr><tr><td>FOLIO</td><td>75.08</td><td>69.60</td><td>75.08</td><td>523.60</td><td>3.00</td><td>522.48</td></tr><tr><td>CH_a</td><td>59.42</td><td>42.13</td><td>59.42</td><td>599.96</td><td>3.00</td><td>599.70</td></tr><tr><td>CH_d</td><td>60.42</td><td>53.00</td><td>60.42</td><td>425.53</td><td>2.98</td><td>425.53</td></tr><tr><td>Arc_chall</td><td>94.45</td><td>93.77</td><td>94.45</td><td>372.28</td><td>3.03</td><td>372.28</td></tr><tr><td>Arc_easy</td><td>94.53</td><td>94.49</td><td>94.53</td><td>345.25</td><td>3.05</td><td>344.99</td></tr><tr><td>GPQA</td><td>61.16</td><td>49.33</td><td>49.33</td><td>883.89</td><td>3.77</td><td>10.78</td></tr><tr><td>MuSR</td><td>62.96</td><td>59.39</td><td>59.39</td><td>629.73</td><td>3.01</td><td>4.96</td></tr><tr><td>LSAT</td><td>75.92</td><td>72.65</td><td>73.84</td><td>690.96</td><td>3.04</td><td>341.13</td></tr><tr><td>CSQA</td><td>84.11</td><td>84.68</td><td>84.03</td><td>300.50</td><td>3.00</td><td>299.94</td></tr><tr><td>PIQA</td><td>91.08</td><td>93.63</td><td>91.78</td><td>273.49</td><td>2.96</td><td>197.62</td></tr><tr><td>SIQA</td><td>77.74</td><td>78.10</td><td>77.69</td><td>247.44</td><td>3.02</td><td>246.60</td></tr><tr><td>StrategyQA</td><td>60.66</td><td>55.24</td><td>55.28</td><td>267.89</td><td>2.99</td><td>3.19</td></tr></table>

Table 12: Prediction accuracy of two CoT effectiveness indicators (Instance SC and aggregated SC) using Llama-3.2-3B-Instruct with modified decoding strategies 

<table><tr><td rowspan="2">Strategies</td><td colspan="3">temperature = 0.3</td><td colspan="3">temperature = 0.7</td><td colspan="3">temperature = 0.9</td></tr><tr><td>topk=5</td><td>topk=10</td><td>topk=20</td><td>topk=5</td><td>topk=10</td><td>topk=20</td><td>topk=5</td><td>topk=10</td><td>topk=20</td></tr><tr><td>Agg_accuracy(%)</td><td>92.86</td><td>92.86</td><td>85.71</td><td>85.71</td><td>85.71</td><td>85.71</td><td>85.71</td><td>85.71</td><td>85.71</td></tr><tr><td>Ins_accuracy(%)</td><td>85.71</td><td>85.71</td><td>78.57</td><td>92.86</td><td>92.86</td><td>92.86</td><td>92.86</td><td>92.86</td><td>92.86</td></tr></table>

Table 13: Experiments based on few-shot CoT: Instance SC, Aggregated SC, CoT Gain and CoT Gain Significance across benchmarks on Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct, Llama-3.1-8B-Instruct.   
Llama-3.2-3B-Instruct 

<table><tr><td></td><td>GSM8K</td><td>MultiArith</td><td>FOLIO</td><td>CH_a</td><td>CH_d</td><td>Arc_chall</td><td>Arc_easy</td></tr><tr><td>Instance SC</td><td>0.0450</td><td>0.1619</td><td>0.1869</td><td>0.0720</td><td>0.0900</td><td>-0.0513</td><td>-0.0552</td></tr><tr><td>Aggregated SC</td><td>0.2080</td><td>0.2904</td><td>0.0488</td><td>-0.032</td><td>0.2670</td><td>-0.4597</td><td>-0.6061</td></tr><tr><td>CoT Gain</td><td>67.25</td><td>72.16</td><td>6.31</td><td>7.54</td><td>26.7</td><td>-1.36</td><td>-3.28</td></tr><tr><td>Significance</td><td>positive</td><td>positive</td><td>positive</td><td>positive</td><td>positive</td><td>none</td><td>negative</td></tr><tr><td></td><td>GPQA</td><td>MuSR</td><td>LSAT</td><td>CSQA</td><td>PIQA</td><td>SIQA</td><td>StrategyQA</td></tr><tr><td>Instance SC</td><td>0.0038</td><td>-0.0840</td><td>-0.0661</td><td>-0.1102</td><td>-0.2698</td><td>-0.2698</td><td>-0.0082</td></tr><tr><td>Aggregated SC</td><td>-0.1042</td><td>-0.4016</td><td>-0.4180</td><td>-0.3424</td><td>-0.5340</td><td>-0.7639</td><td>-0.3917</td></tr><tr><td>CoT Gain</td><td>-7.59</td><td>-5.69</td><td>-4.46</td><td>0.82</td><td>0.44</td><td>2.77</td><td>-6.46</td></tr><tr><td>Significance</td><td>negative</td><td>negative</td><td>negative</td><td>none</td><td>none</td><td>none</td><td>negative</td></tr></table>

Mistral-7B-Instruct-v0.3 

<table><tr><td></td><td>GSM8K</td><td>MultiArith</td><td>FOLIO</td><td>CH_a</td><td>CH_d</td><td>Arc_chall</td><td>Arc_easy</td></tr><tr><td>Instance SC</td><td>0.1193</td><td>0.0837</td><td>0.1925</td><td>-0.1410</td><td>-0.035</td><td>-0.1479</td><td>-0.1218</td></tr><tr><td>Aggregated SC</td><td>0.4285</td><td>0.2781</td><td>-0.0889</td><td>-0.4540</td><td>-0.509</td><td>-0.6940</td><td>-0.6640</td></tr><tr><td>CoT Gain</td><td>40.86</td><td>59.34</td><td>-2.58</td><td>2.83</td><td>-1.79</td><td>2.22</td><td>0.34</td></tr><tr><td>Significance</td><td>positive</td><td>positive</td><td>none</td><td>positive</td><td>none</td><td>none</td><td>none</td></tr><tr><td></td><td>GPQA</td><td>MuSR</td><td>LSAT</td><td>CSQA</td><td>PIQA</td><td>SIQA</td><td>StrategyQA</td></tr><tr><td>Instance SC</td><td>0.0298</td><td>0.3173</td><td>0.1771</td><td>-0.0488</td><td>-0.0690</td><td>0.1708</td><td>-0.1409</td></tr><tr><td>Aggregated SC</td><td>-0.4995</td><td>-0.4520</td><td>-0.5528</td><td>-0.4241</td><td>-0.6690</td><td>-0.7730</td><td>-0.7609</td></tr><tr><td>CoT Gain</td><td>-5.36</td><td>-0.79</td><td>-0.99</td><td>-0.73</td><td>-3.04</td><td>-2.10</td><td>5.28</td></tr><tr><td>Significance</td><td>none</td><td>none</td><td>none</td><td>none</td><td>negative</td><td>none</td><td>positive</td></tr></table>

Phi-3.5-mini-instruct 

<table><tr><td></td><td>GSM8K</td><td>MultiArith</td><td>FOLIO</td><td>CH_a</td><td>CH_d</td><td>Arc_chall</td><td>Arc_easy</td></tr><tr><td>Instance SC</td><td>0.3292</td><td>0.2699</td><td>0.0437</td><td>0.0230</td><td>0.0098</td><td>-0.1908</td><td>-0.2007</td></tr><tr><td>Aggregated SC</td><td>0.6560</td><td>0.6160</td><td>0.2386</td><td>0.1460</td><td>-0.0385</td><td>-0.7586</td><td>-0.7571</td></tr><tr><td>CoT Gain</td><td>59.36</td><td>43.50</td><td>0.25</td><td>1.63</td><td>-0.33</td><td>-1.88</td><td>0.55</td></tr><tr><td>Significance</td><td>positive</td><td>positive</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td></tr><tr><td></td><td>GPQA</td><td>MuSR</td><td>LSAT</td><td>CSQA</td><td>PIQA</td><td>SIQA</td><td>StrategyQA</td></tr><tr><td>Instance SC</td><td>0.0015</td><td>-0.0534</td><td>0.0126</td><td>-0.2399</td><td>-0.0839</td><td>-0.0734</td><td>-0.1905</td></tr><tr><td>Aggregated SC</td><td>-0.3629</td><td>-0.4933</td><td>-0.4341</td><td>-0.8655</td><td>-0.6396</td><td>-0.5340</td><td>-0.7711</td></tr><tr><td>CoT Gain</td><td>4.69</td><td>1.06</td><td>0.49</td><td>0.66</td><td>-3.37</td><td>-1.07</td><td>4.58</td></tr><tr><td>Significance</td><td>none</td><td>none</td><td>none</td><td>none</td><td>negative</td><td>none</td><td>positive</td></tr></table>

Llama-3.1-8B-Instruct 

<table><tr><td></td><td>GSM8K</td><td>MultiArith</td><td>FOLIO</td><td>CH_a</td><td>CH_d</td><td>Arc_chall</td><td>Arc_easy</td></tr><tr><td>Instance SC</td><td>0.0015</td><td>0.1144</td><td>0.3419</td><td>0.236</td><td>0.094</td><td>0.2619</td><td>0.4176</td></tr><tr><td>Aggregated SC</td><td>0.2469</td><td>0.3061</td><td>0.5058</td><td>0.107</td><td>0.042</td><td>-0.5972</td><td>-0.5449</td></tr><tr><td>CoT Gain</td><td>66.11</td><td>55.17</td><td>9.13</td><td>13.54</td><td>22.46</td><td>4.01</td><td>0.16</td></tr><tr><td>Significance</td><td>positive</td><td>positive</td><td>positive</td><td>positive</td><td>positive</td><td>positive</td><td>none</td></tr><tr><td></td><td>GPQA</td><td>MuSR</td><td>LSAT</td><td>CSQA</td><td>PIQA</td><td>SIQA</td><td>StrategyQA</td></tr><tr><td>Instance SC</td><td>0.3395</td><td>-0.0182</td><td>0.1033</td><td>-0.0156</td><td>0.4174</td><td>-0.0351</td><td>-0.1669</td></tr><tr><td>Aggregated SC</td><td>-0.3422</td><td>-0.3472</td><td>-0.2613</td><td>-0.4229</td><td>-0.1907</td><td>-0.6066</td><td>-0.1714</td></tr><tr><td>CoT Gain</td><td>3.13</td><td>-2.12</td><td>0.40</td><td>0.25</td><td>0.27</td><td>-0.41</td><td>-2.18</td></tr><tr><td>Significance</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td></tr></table>