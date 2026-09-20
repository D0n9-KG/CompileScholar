# Putting People in LLMs' Shoes: Generating Better Answers via Question Rewriter

Junhao Chen, Bowen Wang\*, Zhouqiang Jiang, Yuta Nakashima

Osaka University, Japan
{junhao, zhouqiang}@is.ids.osaka-u.ac.jp, {wang, n-yuta}@ids.osaka-u.ac.jp

# Abstract

Large Language Models (LLMs) have demonstrated significant capabilities, particularly in the domain of question answering (QA). However, their effectiveness in QA is often undermined by the vagueness of user questions. To address this issue, we introduce single-round instance-level prompt optimization, referred to as question rewriter. By enhancing the intelligibility of human questions for black-box LLMs, our question rewriter improves the quality of generated answers. The rewriter is optimized using direct preference optimization based on feedback collected from automatic criteria for evaluating generated answers; therefore, its training does not require costly human annotations. The experiments across multiple black-box LLMs and long-form question answering (LFQA) datasets demonstrate the efficacy of our method. This paper provides a practical framework for training question rewriters and sets a precedent for future explorations in prompt optimization within LFQA tasks. Code is available at https://github.com/3244we/Question-Rewriter.

# Introduction

Large language models (LLMs) have incorporated extensive world knowledge through learning vast publicly available corpora (Roberts, Raffel, and Shazeer 2020). It becomes increasingly common for people to seek knowledge from LLMs, especially in fields such as medicine and law (Atallah et al. 2023; Harrington 2023; Wang et al. 2024). However, a near-paradoxical issue arises: People ask questions to get knowledge, while lack of knowledge often leads to poorly formulated or vague questions, hindering LLMs from providing precise answers (Kim et al. 2023; Zhang et al. 2024). Fine-tuning can enhance LLMs' ability to understand vague questions, but most popular LLMs are black-box models, and their parameters are inaccessible. Thus, a step of transforming user questions into a format that LLMs can understand better, known as question rewriting, is crucial for question answering (QA).

Question rewriting is closely related to prompt optimization. A prompt is an input to LLMs that guides them in generating a specific response, including a question (possibly with some instructions), a conversation history, etc. (Liu et al. 2023). Question rewriting is prompt optimization solely for questions. Previous work on prompt optimization primarily focused on optimizing task-level prompts. They decompose prompts into task-level instructions and instance-level inputs, optimizing task-level instructions for better performance across all instances of the task (Fernando et al. 2023; Guo et al. 2023; Kong et al. 2024).

![](images/8ffa0e0c7ea0c5f0b45eab1f7b353a47a7393b331c42552d520a073b9a497697.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Ask Directly"] --> B["Human"]
    B --> C["Original Question"]
    C --> D["Black-box LLM"]
    D --> E["Original Answer"]
    F["Multi-round Method"] --> G["Original Question"]
    G --> H["or"]
    H --> I["Rewritten Question"]
    I --> J["Black-box LLM"]
    J --> K["Better Answer"]
    L["Our Method"] --> M["Original Question"]
    M --> N["Rewriter"]
    N --> O["Rewritten Question"]
    O --> P["Black-box LLM"]
    P --> Q["Better Answer"]
    R["What causes random chest pains that come and go?"] --> C
    S["It could be stress or muscle strain."] --> D
    T["What are the possible causes of intermittent chest pain?"] --> M
    U["Intermittent chest pain can be caused by various factors, including angina, gastroesophageal reflux disease (GERD), musculoskeletal issues, or anxiety."] --> Q
```
</details>

Figure 1: The original questions posed by the user are difficult for black-box LLMs to understand, resulting in poor answers. However, when the questions are rewritten by the rewriter, they become easier for LLMs to comprehend, leading to better answers.

Recent studies have shown that directly optimizing the prompts at the instance level offers more flexibility in prompt editing tailored for a given prompt(Lin et al. 2024) and can lead to better responses (Srivastava et al. 2023). By obtaining feedback from humans or LLMs, they iteratively refine a given prompt, which requires multi-round interactions. In addition, previous prompt optimization is mainly applied to arithmetic reasoning (Cobbe et al. 2021) and short-form question answering (SFQA) (Kwiatkowski et al. 2019), where the latter involves answers in few words. These tasks do not necessarily cover real-world QA scenarios.

This paper proposes single-round instance-level prompt optimization, referred to as question rewriter, aiming at optimizing questions for long-form question answering (LFQA), which is closer to real-world QA scenarios (Bhat et al. 2023). The question rewriter serves as an intermediary between users and black-box LLMs to generate answers, as shown in Figure 1. When a user submits a question, our question rewriter scutches it to contextualizes the question for black-box LLMs to generate a more accurate answer.

The key to our method lies in obtaining supervising signals for optimizing the question rewriter. Different from arithmetic reasoning and SFQA, there is no unique best answer for LFQA questions; therefore, obtaining the optimal rewritten question as the ground truth to train a question rewriter is not trivial (Radford and Narasimhan 2018). Our method, in contrast, assumes the presence of automatic criteria to evaluate generated answers, which are typically provided in LFQA datasets, and uses them to identify better and worse rewritten questions. With such supervising signals, we propose to use direct preference optimization (DPO) (Rafailov et al. 2023) to train our question rewriter. Thanks to this design choice, our method does not necessitate costly human interactions used in reinforcement learning from human feedback (Bai et al. 2022) and a differentiable reward model in proximal policy optimization (Schulman et al. 2017).

Contribution. Our question rewriter is single-round prompt optimization without human interventions, which has not been explored so far. We experimentally show across multiple datasets and LLMs for answer generation that, with optimization by automatic evaluation criteria, the question rewriter can generate questions that end up with better answers. Intiguingly, our analysis implies that the question rewriter learns to generate non-leading and concise questions in a professional tone, which aligns with our intuitions when engineering a prompt.

# Related Work

Early work for prompt optimization focused on white-box models (Shin et al. 2020; Shi et al. 2023; Li and Liang 2021; Lester, Al-Rfou, and Constant 2021; Zhong, Friedman, and Chen 2021). Due to the prevalent nature of black-box models such as GPT-3 (Patel et al. 2023) and Claude (Anthropic 2024), the following work targeted at these black-box models. Most works decomposed prompts into task-level (i.e., instructions) and instance-level (i.e., specific queries) and optimizing only task-level instructions. Some work assumed that input (i.e., text embeddings) and output (i.e., logits) are accessible and leveraged them to optimize prompts (Sun et al. 2022b,a; Chai et al. 2022). Other recent work has attempted to remove this assumption. Prasad et al. (2023) and Pryzant et al. (2023) evaluate task-level instructions with small edits (e.g., replacing some phrases with their synonyms) and find better ones step by step. Evolutionary algorithms (Fernando et al. 2023; Guo et al. 2023), reinforce learning (Diao et al. 2023; Kong et al. 2024), and planning-based methods (Wang et al. 2023) have also been adopted.

Some work fully utilized the inherent capabilities of LLMs to refine prompts. Zhou et al. (2023) leverages an LLM to generate and refine the prompts iteratively, and Yang et al. (2023) provides the prompt optimization trajectory to an LLM, allowing for discovering inherent patterns and optimizing the prompt progressively. Other notable efforts, such as InstructZero (Chen et al. 2023) and INSTINCT (Lin et al. 2023), have transformed black-box optimization into an iterative optimization problem with white-box LLMs.

![](images/a7e07053e29bbd6d34383239fc99777a7f2d3cb399e8bd992790988f44126cd1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original Questions (LFQA Dataset)"] --> B["Initial Rewriter"]
    B --> C["Rewritten Questions"]
    C --> D["Generate LLM"]
    D --> E["Rewritten Answers"]
    E --> F["Judge Method"]
    F --> G["Better and Worse Question Pairs"]
    G --> H["Direct Preference Optimization"]
    H --> I["Optimized Rewriter"]
    I --> B
    B -.->|Same| J["Initial Rewriter"]
```
</details>

Figure 2: Pipeline of our method.

All these works are task-level prompt optimization. Recent studies have shown that instance-level prompt optimization results in better performance by offering more specific prompts (Lin et al. 2024; Srivastava et al. 2023). These works iteratively refine a prompt at the instance level by obtaining feedback from humans or ChatGPT. We also adopt the instance-level approach, but unlike the other instance-level method, ours does not require feedback from LLMs or humans for multi-round optimization. We instead use preference optimization (Rafailov et al. 2023) for our question rewriter that can optimize the questions at once without any feedback nor iterative refinement. We evaluate our method over LFQA tasks, whereas previous works mainly use arithmetic reasoning (Cobbe et al. 2021) and SFQA (Kwiatkowski et al. 2019). LFQA is closer to real-world QA scenarios (Bhat et al. 2023) and can highlight differences in the generated text.

# Method

Our question rewriter R learns to rewrite questions so that an LLM can give a better answer for a rewritten question. We design our method under the assumption that the goodness of the answer to a certain question is automatically judgeable. With this assumption, we can sample rewritten questions and the corresponding answers using LLMs, contrasting them to learn desirable questions.

Figure 2 shows the pipeline of our method. Let $\mathcal{D} = \{(q, a)\}$ denote a training dataset of pairs of question $q$ and answer $a$ , with an associated set $\mathcal{C} = \{c\}$ of automatic evaluation criteria $c$ . Firstly, our pipeline rewrites questions for $q \in \mathcal{D}$ . Then, $c \in \mathcal{C}$ evaluates the rewritten questions to make a set $\mathcal{P} = \{(\hat{q}, \check{q})\}$ of pairs of a better question $\hat{q}$ and a worse question $\check{q}$ . Finally, we use DPO (Rafailov et al. 2023) to train $R$ with $\mathcal{P}$ .

# Sampling Rewritten Questions

We use a pre-trained LLM $R_0$ to sample rewritten questions without fine-tuning as it offers sufficient capability for initial rewriting solely through prompt engineering. We use

top-p sampling (Radford et al. 2019) to generate K different rewritten questions $\mathcal{Q}(q)=\{r_{k}(q)|k=1,\ldots,K\}$ of $q\in D$ , where $r_{k}(q)$ is the k-th rewritten question for q, with the predefined prompt t, i.e., t equals:

Rewriting question to make it more understandable, just give me the rewritten question without any other word:

followed by $q$ .

# Making Better and Worse Question Pairs

Datasets for LFQA typically provide methods for evaluating generated answers. For instance, some datasets (Manes et al. 2024) are inspired by FActScore (Min et al. 2023) and annotate the facts required to answer each question, allowing LLMs to assess whether the corresponding facts are implied by or contradict the generated answers to derive scores for comprehensiveness and precision. Other datasets (Lin, Hilton, and Evans 2022) offer extensive binary annotations used to train classifiers to determine whether answers conform to certain attributes like truthfulness. Additionally, some datasets $^{1}$ are in the form of preference datasets, which provide pairs of samples, where one is better than the other. Such datasets can be used to train reward models to evaluate whether answers align with human preferences. We can use these automatic evaluation criteria as C to evaluate rewritten questions. Such automatic evaluation criteria substitute the human feedback typically used in previous methods (Rafailov et al. 2023) to make P.

Let L denote a pre-trained LLM for answer generation. For a question-answer pair $(q, a) \in \mathcal{D}$ , we generate answers for all $q' \in \mathcal{Q}(q)$ as $a' = L(q')$ . We also generate the answer to the original question q as $\tilde{a} = L(q)$ , which serves as the baseline to judge the goodness of rewritten questions.

To make better-worse pairs, we first identify $q' \in \mathcal{Q}(q)$ that gives better answers and worse answers, collectively denoted by $\mathcal{Q}_{+}(q)$ and $\mathcal{Q}_{-}(q)$ , respectively. Observing that criterion $c \in C$ is often numerical $^{2}$ , we judge $q'$ is better if $a'$ is larger than or equal to $\tilde{a}$ in terms of all criteria and $a'$ is larger than $\tilde{a}$ at least one criterion, $^{3}$ i.e.,

$$
\begin{array}{l} \mathcal {Q} _ {+} (q) = \left\{q ^ {\prime} \in \mathcal {Q} (q) | \forall_ {c \in \mathcal {C}} c \left(a ^ {\prime}\right) \geq c (\tilde {a}), \right. \\ \exists_ {c \in \mathcal {C}} c (a ^ {\prime}) > c (\tilde {a}) \}. \tag {1} \\ \end{array}
$$

$\mathcal{Q}_{-}(q)$ is defined in the opposite way, i.e., $a'$ should be always worse than or equal to $\tilde{a}$ and $a'$ should be worse than $\tilde{a}$ for at least one criterion.

A better and worse question pair is created by picking one rewritten question from $\mathcal{Q}_{+}(q)$ and the other from $\mathcal{Q}_{-}(q)$ . As we wish to train a model R to generate good questions, we rank rewritten questions in $Q_{+}$ according to a certain

composition of all $c \in C,^{4}$ and use the top $N_{+}$ questions. The set of chosen better questions is denoted by $\mathcal{Q}_{+}^{\star}(q)$ . On the other hand, to avoid following DPO training only with easy negatives, we randomly choose $N_{-}$ questions in $\mathcal{Q}_{-}(q)$ and pair each of them with $\hat{q} \in \mathcal{Q}_{+}^{\star}(q)$ . Formally, letting S denote randomly sampled $N_{-}$ questions from $\mathcal{Q}_{-}(q)$ without replacement, the set $\mathcal{P}(q)$ of better and worse question pairs for q is given by:

$$
\mathcal {P} (q) = \{(\hat {q}, \check {q}) | \hat {q} \in \mathcal {Q} _ {+} ^ {\star} (q), \check {q} \in \mathcal {S} (\mathcal {Q} _ {-} (q)) \}. \tag {2}
$$

$\mathcal{P}(q)$ contains $N_{+} \times N_{-}$ pairs when $|\mathcal{Q}_{+}(q)| \geq N_{+}$ and $|\mathcal{Q}_{-}(q)| \geq N_{-}$ ; otherwise, $|\mathcal{P}(q)|$ is smaller. The comparison of the different sampling combination $^{5}$ for $\mathcal{P}(q)$ can be found in Table 6.

# Optimizing Question Rewriter

Training our question rewriter R is costly when it requires human feedback or a reward model that learns the human feedback. Fortunately, LFQA tasks typically offer automatic criteria to evaluate the goodness of generated answers. We can use the criteria to (indirectly) evaluate rewritten questions through evaluating their answers.

Let $P_{R}(q^{\prime}|t,q)$ denote the average probability of tokens in $q^{\prime}$ given the predefined prompt t and the original question q with R, given by:

$$
P _ {R} (q ^ {\prime}) = \frac {1}{T} \sum_ {k = 1} ^ {K} p _ {R} (w _ {k} | t, q, w _ {1: k - 1}), \tag {3}
$$

where K is the length of $q'$ ; $p_{R}(w_{t}|t, q, w_{1:k-1})$ is the probability of token $w_{t}$ given t, q, and a set $w_{1:k-1}$ of tokens generated by the $(k-1)$ -th step (i.e., $q' = w_{1:K}$ ). $P_{R_{0}}(q')$ is defined likewise for the initial question rewriter $R_{0}$ . DPO's training loss is given by:

$$
L = - \mathbb {E} \left[ \log \sigma \left(\beta \log \frac {P _ {R} (\hat {q})}{P _ {R _ {0}} (\hat {q})} - \beta \log \frac {P _ {R} (\check {q})}{P _ {R _ {0}} (\check {q})}\right) \right] \tag {4}
$$

where $\sigma$ is sigmoid, and $\beta$ is a hyperparameter that controls how much $R$ deviates from $R_0$ , and the expectation is computed over $q \sim \mathcal{D}$ and $(\hat{q}, \check{q}) \sim \mathcal{P}(q)$ .

To mitigate the risk of overfitting, we use dropout in the model. Also, the original LFQA dataset is divided into three parts: training, validation, and testing. R is trained on the training set (i.e., D) for one epoch, and we select the best model that most prefer $\hat{q}$ 's to $\check{q}$ 's. Specifically, we define preference score PS as

$$
\mathrm{PS} = \mathbb {E} \left[ \mathbf {1} [ P _ {R} (\hat {q} | t, q) > P _ {R} (\check {q} | t, q) ] \right], \tag {5}
$$

where $1[\cdot]$ gives 1 if the given condition is satisfied, and otherwise 0; the expectation is computed for all the q from the validation set and $(\hat{q},\check{q})\sim\mathcal{P}(q)$ .

<table><tr><td>Dataset</td><td>Training</td><td>Validation</td><td>Testing</td><td>Total</td></tr><tr><td>K-QA</td><td>101</td><td>50</td><td>50</td><td>201</td></tr><tr><td>TruthfulQA</td><td>407</td><td>205</td><td>205</td><td>817</td></tr><tr><td>OASST1QA</td><td>1,000</td><td>93</td><td>93</td><td>1,186</td></tr></table>

Table 1: Statistics of LFQA datasets used to evaluate our method. Columns for Training, Validation, and Testing give the numbers of samples in respective dataset splits.

# Experiments

# Experimental Setup

Dataset We evaluate three distinct LFQA datasets, each equipped with automated evaluation criteria.

K-QA (Manes et al. 2024), sourced from the medical domain, is designed to evaluate the factual comprehensiveness and precision of answers through metrics $S_{comp}$ and $S_{cont}$ , employing a FActScore type method (Min et al. 2023). To combine these two criteria for ranking rewritten questions in $\mathcal{Q} + (q)$ , we first use $S_{cont}$ to rank them, and then use $S_{comp}$ if $S_{cont}$ is the same for multiple questions.

TruthfulQA (Lin, Hilton, and Evans 2022), covering multiple domains including health and law, assesses the truthfulness ( $S_{\text{truth}}$ ) and informativeness ( $S_{\text{info}}$ ) of answers. The evaluation criteria are implemented as binary classifiers. We use the probabilities for positive classes (truthful for $S_{\text{truth}}$ and informative for $S_{\text{info}}$ ). An overall score ( $S_{\text{overall}}$ ) is computed as the product of these scores. For better rewritten pair ranking, we use $S_{\text{overall}}$ .

OASST1QA, derived from the multi-turn dialogue alignment dataset OASST1 $^{6}$ , incorporates a criterion $S_{pref}$ that measures human preference for answers using a pre-trained reward model. This dataset provides a single criterion for evaluation (i.e., $|C| = 1$ ), so we directly use $S_{pref}$ for ranking better rewritten questions.

More details about these datasets and their evaluation criteria can be found in the appendix. Table 1 summarizes the statistics on the datasets.

LLMs The base model of our question rewriter R (and $R_{0}$ is Llama3-8B-instruct, and the answer generation model L is also Llama3-8B-instruct because it is one of the most powerful but small LLMs. R is fine-tuned with our method, while L is frozen. Subsequently, we evaluate the generalizability of R on multiple answer generation LLMs, including Llama3-8B-instruct $^{7}$ , mistral-7B-instruct-v0.2 $^{8}$ , zephyr-7B-beta $^{9}$ , gemma-1.1-7B-it $^{10}$ , gpt-3.5-turbo-1106, and gpt-4o-2024-05-13. They will be referred to as Llama3-8B, Mistral-7B-v0.2, Zephyr-7B-beta, Gemma-1.1-7B, GPT-3.5, and GPT-4o, respectively. It is worth noting that we only use L as Llama3-8B-instruct to build P for training R, and then test the generalizability of R on other models.

Hyperparameters We borrowed open-source code for DPO training over all three datasets $^{11}$ , which also provides the code for supervised fine-tuning of automatic criteria $S_{truth}$ and $S_{info}$ for TruthfulQA. During DPO training, we set the dropout rate to 0.8, the training batch size to 32, and the testing batch size to 64, maintaining all other parameters at their default settings in the source code. For sampling rewritten questions, we use top-p sampling, where the cumulative probability for top-p sampling is set to 0.999, and the temperature of $R_{0}$ is 1, to ensure diversity. We sample 100 unique rewritten questions for each of the original questions and terminate the sampling after 10,000 attempts. $N_{+}$ and $N_{-}$ are defaulted to (10, 20), (5, 10), and (4, 5) in K-QA, TQA, and OQA respectively. When multiplied by the number of samples in the corresponding training sets, they are around 20,000. The maximum token length is set to 512 during feedback collection and testing. During testing, to ensure reproducibility, we generate answers using greedy sampling.

Device All our testing and training, except for the DPO training of OASST1QA, are conducted on a system equipped with four NVIDIA A100-PCIE-40GB. Due to the extensive length of OASST1QA, we only used samples whose question plus the prompt t and rewritten questions $q'$ for question rewriting is less than or equal to 512 tokens and conducted the DPO training on a system with four NVIDIA A100-80GB-PCIe.

Baselines In our experiments across different datasets and models, we compare our method with both the original questions and the initial Llama3-8B-instruct rewriter (without fine-tuning). To demonstrate the effectiveness of our approach, we also compare it with the widely used task-level prompting method, Zero-Shot Chain-of-Thought (Zero-Shot CoT) (Kojima et al. 2022). Other instance-level methods, such as PRoMPTed, require multiple rounds of interactions with humans or LLMs to obtain feedback and iteratively modify the prompt or question during inference, which is extremely costly in our QA scenarios. Therefore, we only perform comparisons with PRoMPTed and other question rewriting methods on the K-QA dataset and Llama3-8B-instruct.

# Result across models and datasets

Table 2 summarizes our experimental results over three LFQA datasets. Our method demonstrates superior performance in most combinations of LLMs and datasets.

For the K-QA dataset, our method consistently shows the highest $S_{comp}$ scores across all models, especially with GPT-4o, where the improvement is most significant. Furthermore, it achieves the lowest $S_{cont}$ with half of the models and the second lowest in the rest. Notably, our method trained an effective question rewriter using only 151 samples (i.e., training set plus validation set), implying that our method requires only a small number of annotated samples in a real-world QA scenario. Table 3 shows an example from K-QA,

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">K-QA</td><td colspan="3">TruthfulQA</td><td>OASST1QA</td></tr><tr><td> $S_{\text{comp}} \uparrow$ </td><td> $S_{\text{cont}} \downarrow$ </td><td> $S_{\text{truth}} \uparrow$ </td><td> $S_{\text{info}} \uparrow$ </td><td> $S_{\text{overall}} \uparrow$ </td><td> $S_{\text{pref}} \uparrow$ </td></tr><tr><td rowspan="4">Llama-3-8B</td><td>Original</td><td>0.4573</td><td>0.4400</td><td>0.7683</td><td>0.9664</td><td>0.7397</td><td>0.8654</td></tr><tr><td>Zero-Shot CoT</td><td>0.4579</td><td>0.4000</td><td>0.7476</td><td>0.9306</td><td>0.6938</td><td>0.8838</td></tr><tr><td>Initial Rewriter</td><td>0.4262</td><td>0.5000</td><td>0.7914</td><td>0.9564</td><td>0.7566</td><td>0.8748</td></tr><tr><td>Ours</td><td>0.4600</td><td>0.4000</td><td>0.8059</td><td>0.9668</td><td>0.7789</td><td>0.9104</td></tr><tr><td rowspan="4">Mistral-7B-v0.2</td><td>Original</td><td>0.4374</td><td>0.2200</td><td>0.8364</td><td>0.9834</td><td>0.8227</td><td>0.8281</td></tr><tr><td>Zero-Shot CoT</td><td>0.4428</td><td>0.2800</td><td>0.8423</td><td>0.9737</td><td>0.8199</td><td>0.8908</td></tr><tr><td>Initial Rewriter</td><td>0.4177</td><td>0.3400</td><td>0.7916</td><td>0.9689</td><td>0.7670</td><td>0.8381</td></tr><tr><td>Ours</td><td>0.4899</td><td>0.2600</td><td>0.8474</td><td>0.9788</td><td>0.8296</td><td>0.8762</td></tr><tr><td rowspan="4">Zephyr-7B-beta</td><td>Original</td><td>0.4396</td><td>0.3400</td><td>0.7644</td><td>0.9826</td><td>0.7518</td><td>0.6369</td></tr><tr><td>Zero-Shot CoT</td><td>0.4333</td><td>0.3200</td><td>0.7081</td><td>0.9705</td><td>0.6867</td><td>0.7606</td></tr><tr><td>Initial Rewriter</td><td>0.4666</td><td>0.2200</td><td>0.7353</td><td>0.9723</td><td>0.7167</td><td>0.6417</td></tr><tr><td>Ours</td><td>0.4702</td><td>0.2600</td><td>0.7709</td><td>0.9775</td><td>0.7528</td><td>0.7768</td></tr><tr><td rowspan="4">Gemma-1.1-7B</td><td>Original</td><td>0.4010</td><td>0.5400</td><td>0.6780</td><td>0.9716</td><td>0.6554</td><td>0.7428</td></tr><tr><td>Zero-Shot CoT</td><td>0.4516</td><td>0.5000</td><td>0.7216</td><td>0.9454</td><td>0.6752</td><td>0.8632</td></tr><tr><td>Initial Rewriter</td><td>0.4928</td><td>0.4400</td><td>0.6415</td><td>0.9617</td><td>0.6124</td><td>0.7955</td></tr><tr><td>Ours</td><td>0.4956</td><td>0.2200</td><td>0.7224</td><td>0.9558</td><td>0.6888</td><td>0.9034</td></tr><tr><td rowspan="4">GPT-3.5-turbo</td><td>Original</td><td>0.4909</td><td>0.3200</td><td>0.7451</td><td>0.9804</td><td>0.7303</td><td>0.7294</td></tr><tr><td>Zero-Shot CoT</td><td>0.4748</td><td>0.1600</td><td>0.7413</td><td>0.9781</td><td>0.7237</td><td>0.8222</td></tr><tr><td>Initial Rewriter</td><td>0.4454</td><td>0.3000</td><td>0.7325</td><td>0.9768</td><td>0.7164</td><td>0.7353</td></tr><tr><td>Ours</td><td>0.4978</td><td>0.2800</td><td>0.7574</td><td>0.9682</td><td>0.7309</td><td>0.8994</td></tr><tr><td rowspan="4">GPT-4o</td><td>Original</td><td>0.5167</td><td>0.2800</td><td>0.8812</td><td>0.9790</td><td>0.8631</td><td>0.8532</td></tr><tr><td>Zero-Shot CoT</td><td>0.4903</td><td>0.3000</td><td>0.8739</td><td>0.9611</td><td>0.8400</td><td>0.8471</td></tr><tr><td>Initial Rewriter</td><td>0.5255</td><td>0.2400</td><td>0.8593</td><td>0.9683</td><td>0.8329</td><td>0.8461</td></tr><tr><td>Ours</td><td>0.6253</td><td>0.2400</td><td>0.8880</td><td>0.9722</td><td>0.8641</td><td>0.9100</td></tr></table>

Table 2: Comparison of different question rewriting methods across multiple datasets and LLMs for answer generation. We use automatic evaluation criteria associated with each dataset. Bold indicates the best method, and underline indicates the second best.

in which $S_{comp}$ increases and $S_{cont}$ decreases after rewriting the original question.

For the TruthfulQA dataset, all methods generally reduce the informativeness (i.e., $S_{info}$ ) of the answers, and only ours gains the truthfulness score (i.e., $S_{truth}$ ). This is typical behavior in the TruthfulQA dataset as these two criteria exhibit a trade-off relationship (Lin, Hilton, and Evans 2022). Our method can increase $S_{overall}$ , while the others reduce it.

For OASST1QA, our method outperforms others except Mistral-7B-v0.2, where Zero-Shot CoT performs best.

![](images/ca5a8294d8a025a9d1495210c45a1b8324a46d5d6b9496a0596d7b8f79927aff.jpg)

<details>
<summary>line</summary>

| k     | S_comp | S_cont |
|-------|--------|--------|
| 5k    | 0.46   | 0.38   |
| 10k   | 0.46   | 0.32   |
| 20k   | 0.46   | 0.48   |
| 30k   | 0.46   | 0.58   |
| 40k   | 0.38   | 0.48   |

| oasst1QA | S_pref ↑ |
| ------- | -------- |
| 5k      | 0.88     |
| 10k     | 0.88     |
| 20k     | 0.91     |
| 30k     | 0.85     |
| 40k     | 0.86     |
</details>

Figure 3: Evaluating the impact of $N_{+}$ and $N_{-}$ on the performance over K-QA and OASST1QA.

Overall, our method excels in all metrics and all datasets, not only on the Llama3-8B-instruct used for training R but also on the other LLMs, demonstrating their generalizability to answer generation models. Thanks to this generalizability, the rewriter R can be deployed without adjusting to individual (black-box) answer generation models.

# Impact of $N_{+}$ and $N_{-}$

The number of better and worse question pairs $\mathcal{P} = \cup_{q \in \mathcal{D}} \mathcal{P}(q)$ is determined by $N_{+}, N_{-}$ , and $|D|$ (i.e., $|P| = N_{+} \times N_{-} \times |D|$ ), where the choice of $N_{+}$ and $N_{-}$ is rather arbitrary. To explore the impact of $N_{+}$ and $N_{-}$ on the performance, we evaluated our method with varying $N_{+}$ and $N_{-}$ over K-QA OASST1QA with Llama-3-8B $^{12}$ . The results are summarized in Figure 3.

For the K-QA dataset, finding the optimal values of $N_{+}$ and $N_{-}$ is not easy since multiple criteria are involved, but we would say $|P|$ between 5k and 20k works well. Beyond 20k, $S_{comp}$ starts to decrease, and $S_{cont}$ spikes. In

<table><tr><td>Original Question</td><td>How do I know if it&#x27;s a canker sore versus herpes?</td></tr><tr><td>Original Answer $S_{comp}$  ↑: 2/6 $S_{cont}$  ↓: 1</td><td>Canker sores and cold sores (caused by herpes simplex virus) are both types of oral lesions but they have distinct characteristics.Canker Sore:Location:Canker sores typically occur on the inside of the mouth, tongue, lips or throat.Cold Sore (Herpes Simplex Virus):Location:Cold sores typically occur on the lips, around the mouth, or on the nose. ...</td></tr><tr><td>Rewritten Question</td><td>How can I distinguish between a canker sore and a herpes outbreak based on their symptoms and characteristics?</td></tr><tr><td>Rewritten Answer $S_{comp}$  ↑: 3/6 $S_{cont}$  ↓: 0</td><td>Canker Sores (Aphthous Ulcers):Location:Typically occur on the inside of the mouth, tongue, lips, or throat.Herpes Simplex Virus (HSV) Outbreaks:Location:Can occur anywhere on the body but most commonly on the mouth, lips, or genital area.Appearance:Small, fluid-filled blisters that eventually crust over and form a scab. ...</td></tr></table>

Table 3: An example of original and rewritten questions in K-QA with their answers. Text highlighted in green is consistent with the must-to-have fact, whereas red contradicts the must-to-have fact. Omitted text in answers, represented by “...”, are irrelevant to any must-to-have fact.

OASST1QA, the performance increases along with $|P|$ increases by 20k, then it decreases. The performance drops when $|P|$ larger than 20k are attributed to overfitting during DPO training. These results highlight the necessity of adjusting $N_{+}$ and $N_{-}$ for each dataset.

# Comparison with Other Methods

To validate the superiority of our method, we compared it with other question rewriting methods on the K-QA dataset and Llama3-8B-instruct. In addition to the PRoMPTed method mentioned earlier, we considered two other closely related methods. The first method employs a transformer model to rewrite ill-formed questions into well-structured ones, while it does not evaluate the impact on answer quality (Chu et al. 2020). We replicated this method using T5-Flan-base and achieved better performance on their original dataset. The second method, PRewrite (Kong et al. 2024), is an RL-based task-level prompt optimization method that uses RLHF, requiring a differentiable reward function. However, some LFQA datasets, such as K-QA, lack differentiable rewards. To address this, we implemented DPO, a variant of RLHF, to replicate it. We used prompts from it:

Rewrite the following instruction via rephrasing and/or adding specific requirements. Add instructions which would be helpful to solve the problem correctly. Output the new instruction only.

to optimize the original task instruction: “provide the answer:”, and then appended the optimized instruction to the original questions to obtain answers. The optimized instruction can be found in the appendix.

As shown in Table 4, only our method demonstrates improvements in both metrics. While the PRoMPTed method

<table><tr><td>Metric</td><td>Orig.</td><td>Ours</td><td>PRewrite</td><td>T5</td><td>PRoMPTed</td></tr><tr><td> $S_{\text{comp}} \uparrow$ </td><td>0.4573</td><td>0.4600</td><td>0.4409</td><td>0.4160</td><td>0.5012</td></tr><tr><td> $S_{\text{cont}} \downarrow$ </td><td>0.4400</td><td>0.4000</td><td>0.3600</td><td>0.3600</td><td>0.5400</td></tr></table>

Table 4: Comparison of different question rewriting methods. Bold indicates the optimization method is better than the original results. Orig. indicates the original results. T5 indicates the replicated result of ill-formed questions rewriting method

improves $S_{comp}$ , it significantly compromises $S_{cont}$ , highlighting the superiority of our method in balancing these competing objectives. This further confirms that our method outperforms other question rewriting methods.

# Cross-Dataset Generalizability

We explored the performance of rewriters trained on Llama3-8B-instruct across different datasets. Table 5 shows that rewriters trained in the K-QA and TruthfulQA datasets can optimize the generated answers on the OASST1QA dataset, but each fails on one metric in their respective datasets. In contrast, rewriters trained on the OASST1QA dataset are almost ineffective on the other two datasets. This suggests that training on more complex LFQA datasets, which include multiple automatic evaluation criteria, yields rewriters with better generalizability.

# Discussion

Our question rewriters can significantly improve the representation of questions, making them more likely to obtain higher-quality answers with LLMs. To quantitatively analyze how attributes impact the evaluation criteria of gener-

<table><tr><td rowspan="2">Rw.</td><td colspan="2">K-QA</td><td colspan="3">TruthfulQA</td><td>OQA</td></tr><tr><td> $S_{\text{comp}}$ </td><td> $S_{\text{cont}}$ </td><td> $S_{\text{truth}}$ </td><td> $S_{\text{info}}$ </td><td> $S_{\text{overall}}$ </td><td> $S_{\text{pref}}$ </td></tr><tr><td>Orig.</td><td>0.4573</td><td>0.4400</td><td>0.7683</td><td>0.9664</td><td>0.7397</td><td>0.8654</td></tr><tr><td>Rw-K</td><td>0.4600</td><td>0.4000</td><td>0.7834</td><td>0.9535</td><td>0.7454</td><td>0.8759</td></tr><tr><td>Rw-T</td><td>0.4104</td><td>0.2800</td><td>0.8059</td><td>0.9668</td><td>0.7789</td><td>0.8839</td></tr><tr><td>Rw-O</td><td>0.4510</td><td>0.4200</td><td>0.7622</td><td>0.9373</td><td>0.7155</td><td>0.9104</td></tr></table>

Table 5: Performance of rewrites(Rw.) across datasets on Llama3-8B-instruct: OQA represents OASST1QA, Rw-K, Rw-T, and Rw-O represent rewriters trained on K-QA, TruthfulQA and OASST1QA, respectively. Bold indicates the rewriter performing better than the corresponding original for this LLM. Among all the metrics, except for $S_{cont}$ , which is better when lower, all other metrics are better when higher.

ated answers, we study 50 original questions in KQA's test set and their rewritten versions, resulting in 100 questions in total. We adopt 10 attributes: non-leadingness, word choice, tone, conciseness, neutrality, grammar and spelling, structure, politeness, clarity, and emotion, which are identified by an LLM. For each attribute and each question, we use GPT-4o to assign a score ranging from 1 to 5 to the 100 questions. The definitions of attributes and the prompt templates are available in the extended version appendix.

To explore the important attributes that determine evaluation criteria $S_{comp}$ and $S_{cont}$ , we use a random forest regressor that takes the attribute scores as input and predicts either $S_{comp}$ or $S_{cont}$ of each question. We train the regressors with these 100 questions and use the predictions again for them. $^{13}$ The regressors yielded $R^{2}$ values of 0.56 and 0.55 for $S_{comp}$ and $S_{cont}$ , respectively, demonstrating that the attribute scores are significantly correlated with the criteria. The random forest regressors provide feature importance, which, in our case, corresponds to the importance of each attribute.

In addition to the attribute importance, we also examine whether each attribute has a positive or negative impact on the evaluation criteria. To this end, we define the impact by:

$$
I _ {l a} = \hat {S} _ {l a} - \check {S} _ {l a} \tag {6}
$$

where $l \in \{comp, cont\}$ and a is one of the 10 attributes; $\hat{S}_{la}$ and $\check{S}_{la}$ are the averages of evaluation criterion $S_{l}$ of questions whose attribute score for a are among the top-50 and bottom-50, respectively. Specifically, $\hat{S}_{la}$ is given by:

$$
\hat {S} _ {l a} = \frac {1}{5 0} \sum_ {S _ {l} \in \hat {\mathcal {S}} _ {l a}} S _ {l}, \tag {7}
$$

where $\hat{S}_{la}$ is the set of scores $S_{l}$ of questions whose attribute scores are among top 50. $\check{S}_{la}$ is defined likewise. A higher $\hat{S}_{la}$ , for instance, means that the attribute a is positively correlated with $S_{l}$ .

![](images/dcdbd9f85e9ebef40910a4fc26b687bfbbb67b6cf26e68e231e938a259bb97e4.jpg)

<details>
<summary>bar</summary>

| Category | Importance | Impact |
| -------- | ---------- | ------ |
| NL       | 0.20       | 0.12   |
| WC       | 0.14       | 0.10   |
| Tone     | 0.12       | 0.19   |
| Conc.    | 0.12       | 0.12   |
| Neut.    | 0.11       | 0.11   |
| G&S      | 0.11       | 0.08   |
| Strt.    | 0.10       | 0.10   |
| Pol.     | 0.08       | 0.02   |
| Clar.    | 0.03       | -0.05  |
| Emo.     | 0.03       | 0.05   |
</details>

(a) $S_{comp}$

![](images/00c360fc8d2c8a7cef74e4b4550ca0ae5d9453e1fd2bc6acf8ee9ea0bd5a315c.jpg)

<details>
<summary>bar</summary>

| Category | Importance | Impact |
| -------- | ---------- | ------ |
| NL       | 0.16       | -1     |
| Conc.    | 0.16       | 0      |
| Pol.     | 0.12       | 0      |
| G&S      | 0.12       | 0      |
| Neut.    | 0.10       | 0      |
| Str.     | 0.10       | 0      |
| WC       | 0.08       | 0      |
| Tone     | 0.04       | 0      |
| Clar.    | 0.02       | 0      |
| Emo.     | 0.02       | 0      |
</details>

(b) $S_{cont}$   
Figure 4: The importance and impact of attributes conciseness (Conc.), structure (Strt.), word choice (WC), emotion (Emo.), non-leadingness (NL), grammar and spelling (G&S), neutrality (Neut.), tone, clarity (Clar.), and politeness (Pol.). The bar plots are important, while the line plots are impact.

As shown in Figure 4, for $S_{comp}$ , non-leadingness, word choice, and tone are the most contributing attributes to regression, while for $S_{comp}$ , non-leadingness, conciseness, and politeness are important. Intriguingly, no-leadingness for example, which means a question does not give some implication of a certain answer, is the most contributing attribute for both criteria. It is not straightforward to interpret this result, but being or not being leading may give some cues about attributes to be used for regression. At least, these attributes are somehow correlated with the evaluation criteria, and so the impact for these attributes can be meaningful.

As for the impact, we can see tone gives a positive impact to $S_{comp}$ , while non-leadingness and conciseness are negatively correlated with $S_{cont}$ . A higher attribute score for tone means the question is written in a formal language and a professional manner. We can reason that a formal language triggers expert knowledge encompassed in a LLM, which is likely to be written also in a formal language with proper wordings. Meanwhile, being non-leading and concise leads to lower $S_{cont}$ , which is preferable. These results also make much sense; extra text in a question can lead to knowledge that is still relevant to the extra text but irrelevant to the question. Overall, our importance and impact analysis unveils that our question rewriter learns to generate professional, non-leading, and concise questions, which align with our intuitions, solely through supervision by $S_{comp}$ and $S_{cont}$ .

# Conclusions and Future Work

This paper proposes single-round instance-level question optimization for LFQA tasks, coined question rewriter. We employ DPO to optimize the question rewriter with automatic evaluation criteria. Our experimental results demonstrated that our question rewriter can generate questions that give better answers in terms of the automatic evaluation criteria. Meanwhile, although our method demonstrates some degree of cross-domain generalizability, it still has limitations in performance. Therefore, exploring completely domain-agnostic methods would be an interesting direction for future research.

# Acknowledgments

This work was supported by World Premier International Research Center Initiative (WPI), MEXT, Japan. This work is also supported by JST ACT-X Grant Number JPM-JAX24C8 and JSPS KAKENHI No. 24K20795.

# References

Anthropic. 2024. The Claude 3 Model Family: Opus, Sonnet, Haiku.   
Atallah, S.; Banda, N.; Banda, A.; and Roeck, N. 2023. How large language models including generative pre-trained transformer (GPT) 3 and 4 will impact medicine and surgery. Techniques in Coloproctology, 27(8): 609–614.   
Bai, Y.; Jones, A.; Ndousse, K.; Askell, A.; Chen, A.; Das-Sarma, N.; Drain, D.; Fort, S.; Ganguli, D.; Henighan, T.; Joseph, N.; Kadavath, S.; Kernion, J.; Conerly, T.; Showk, S. E.; Elhage, N.; Hatfield-Dodds, Z.; Hernandez, D.; Hume, T.; Johnston, S.; Kravec, S.; Lovitt, L.; Nanda, N.; Olsson, C.; Amodei, D.; Brown, T. B.; Clark, J.; McCandlish, S.; Olah, C.; Mann, B.; and Kaplan, J. 2022. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback. CoRR, abs/2204.05862.   
Bhat, M. M.; Meng, R.; Liu, Y.; Zhou, Y.; and Yavuz, S. 2023. Investigating Answerability of LLMs for Long-Form Question Answering. CoRR, abs/2309.08210.   
Chai, Y.; Wang, S.; Sun, Y.; Tian, H.; Wu, H.; and Wang, H. 2022. Clip-Tuning: Towards Derivative-free Prompt Learning with a Mixture of Rewards. In Goldberg, Y.; Kozareva, Z.; and Zhang, Y., eds., Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, 108–117. Association for Computational Linguistics.   
Chen, L.; Chen, J.; Goldstein, T.; Huang, H.; and Zhou, T. 2023. InstructZero: Efficient Instruction Optimization for Black-Box Large Language Models. CoRR, abs/2306.03082.   
Chu, Z.; Chen, M.; Chen, J.; Wang, M.; Gimpel, K.; Faruqui, M.; and Si, X. 2020. How to ask better questions? a large-scale multi-domain dataset for rewriting ill-formed questions. In Proceedings of the AAAI Conference on Artificial Intelligence, 7586–7593.   
Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; Hesse, C.; and Schulman, J. 2021. Training Verifiers to Solve Math Word Problems. CoRR, abs/2110.14168.   
Diao, S.; Huang, Z.; Xu, R.; Li, X.; Lin, Y.; Zhou, X.; and Zhang, T. 2023. Black-Box Prompt Learning for Pre-trained Language Models. Trans. Mach. Learn. Res., 2023.   
Fernando, C.; Banarse, D.; Michalewski, H.; Osindero, S.; and Rocktäschel, T. 2023. Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution. CoRR, abs/2309.16797.   
Guo, Q.; Wang, R.; Guo, J.; Li, B.; Song, K.; Tan, X.; Liu, G.; Bian, J.; and Yang, Y. 2023. Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers. CoRR, abs/2309.08532.

Harrington, S. A. 2023. The Case for Large Language Model Optimism in Legal Research from a Law & Technology Librarian. Available at SSRN.

Kim, G.; Kim, S.; Jeon, B.; Park, J.; and Kang, J. 2023. Tree of Clarifications: Answering Ambiguous Questions with Retrieval-Augmented Large Language Models. In Bouamor, H.; Pino, J.; and Bali, K., eds., Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 996–1009. Singapore: Association for Computational Linguistics.

Kojima, T.; Gu, S. S.; Reid, M.; Matsuo, Y.; and Iwasawa, Y. 2022. Large Language Models are Zero-Shot Reasoners. In Koyejo, S.; Mohamed, S.; Agarwal, A.; Belgrave, D.; Cho, K.; and Oh, A., eds., Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Kong, W.; Hombaiah, S. A.; Zhang, M.; Mei, Q.; and Bendersky, M. 2024. PRewrite: Prompt Rewriting with Reinforcement Learning. CoRR, abs/2401.08189.

Kwiatkowski, T.; Palomaki, J.; Redfield, O.; Collins, M.; Parikh, A. P.; Alberti, C.; Epstein, D.; Polosukhin, I.; Devlin, J.; Lee, K.; Toutanova, K.; Jones, L.; Kelcey, M.; Chang, M.; Dai, A. M.; Uszkoreit, J.; Le, Q.; and Petrov, S. 2019. Natural Questions: a Benchmark for Question Answering Research. Trans. Assoc. Comput. Linguistics, 7: 452–466.

Lester, B.; Al-Rfou, R.; and Constant, N. 2021. The Power of Scale for Parameter-Efficient Prompt Tuning. In Moens, M.; Huang, X.; Specia, L.; and Yih, S. W., eds., Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, 3045–3059. Association for Computational Linguistics.

Li, X. L.; and Liang, P. 2021. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Zong, C.; Xia, F.; Li, W.; and Navigli, R., eds., Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, 4582–4597. Association for Computational Linguistics.

Lin, S.; Hilton, J.; and Evans, O. 2022. TruthfulQA: Measuring How Models Mimic Human Falsehoods. In Muresan, S.; Nakov, P.; and Villavicencio, A., eds., Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, 3214–3252. Association for Computational Linguistics.

Lin, X.; Dai, Z.; Verma, A.; Ng, S.; Jaillet, P.; and Low, B. K. H. 2024. Prompt Optimization with Human Feedback. CoRR, abs/2405.17346.

Lin, X.; Wu, Z.; Dai, Z.; Hu, W.; Shu, Y.; Ng, S.; Jaillet, P.; and Low, B. K. H. 2023. Use Your INSTINCT: INSTRUCTION optimization usIng Neural bandits Coupled with Transformers. CoRR, abs/2310.02905.

Liu, P.; Yuan, W.; Fu, J.; Jiang, Z.; Hayashi, H.; and Neubig, G. 2023. Pre-train, Prompt, and Predict: A Systematic Sur-

vey of Prompting Methods in Natural Language Processing. ACM Comput. Surv., 55(9): 195:1–195:35.   
Manes, I.; Ronn, N.; Cohen, D.; Ber, R. I.; Horowitz-Kugler, Z.; and Stanovsky, G. 2024. K-QA: A Real-World Medical Q&A Benchmark. CoRR, abs/2401.14493.   
Min, S.; Krishna, K.; Lyu, X.; Lewis, M.; Yih, W.; Koh, P. W.; Iyyer, M.; Zettlemoyer, L.; and Hajishirzi, H. 2023. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. In Bouamor, H.; Pino, J.; and Bali, K., eds., Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, 12076–12100. Association for Computational Linguistics.   
Patel, A.; Li, B.; Rasooli, M. S.; Constant, N.; Raffel, C.; and Callison-Burch, C. 2023. Bidirectional Language Models Are Also Few-shot Learners. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.   
Prasad, A.; Hase, P.; Zhou, X.; and Bansal, M. 2023. GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models. In Vlachos, A.; and Augenstein, I., eds., Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2023, Dubrovnik, Croatia, May 2-6, 2023, 3827–3846. Association for Computational Linguistics.   
Pryzant, R.; Iter, D.; Li, J.; Lee, Y. T.; Zhu, C.; and Zeng, M. 2023. Automatic Prompt Optimization with "Gradient Descent" and Beam Search. In Bouamor, H.; Pino, J.; and Bali, K., eds., Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, 7957–7968. Association for Computational Linguistics.   
Radford, A.; and Narasimhan, K. 2018. Improving Language Understanding by Generative Pre-Training.   
Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.; et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8): 9.   
Rafailov, R.; Sharma, A.; Mitchell, E.; Manning, C. D.; Ermon, S.; and Finn, C. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. In Oh, A.; Naumann, T.; Globerson, A.; Saenko, K.; Hardt, M.; and Levine, S., eds., Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.   
Roberts, A.; Raffel, C.; and Shazeer, N. 2020. How Much Knowledge Can You Pack Into the Parameters of a Language Model? In Webber, B.; Cohn, T.; He, Y.; and Liu, Y., eds., Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, 5418–5426. Association for Computational Linguistics.   
Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal Policy Optimization Algorithms. CoRR, abs/1707.06347.   
Shi, W.; Han, X.; Gonen, H.; Holtzman, A.; Tsvetkov, Y.; and Zettlemoyer, L. 2023. Toward Human Readable Prompt

Tuning: Kubrick's The Shining is a good movie, and a good prompt too? In Bouamor, H.; Pino, J.; and Bali, K., eds., Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, 10994–11005. Association for Computational Linguistics.   
Shin, T.; Razeghi, Y.; IV, R. L. L.; Wallace, E.; and Singh, S. 2020. AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. In Webber, B.; Cohn, T.; He, Y.; and Liu, Y., eds., Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, 4222–4235. Association for Computational Linguistics.   
Srivastava, S.; Huang, C.; Fan, W.; and Yao, Z. 2023. Instance Needs More Care: Rewriting Prompts for Instances Yields Better Zero-Shot Performance. CoRR, abs/2310.02107.   
Sun, T.; He, Z.; Qian, H.; Huang, X.; and Qiu, X. 2022a. BBTv2: Pure Black-Box Optimization Can Be Comparable to Gradient Descent for Few-Shot Learning. CoRR, abs/2205.11200.   
Sun, T.; Shao, Y.; Qian, H.; Huang, X.; and Qiu, X. 2022b. Black-Box Tuning for Language-Model-as-a-Service. In Chaudhuri, K.; Jegelka, S.; Song, L.; Szepesvári, C.; Niu, G.; and Sabato, S., eds., International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, 20841–20855. PMLR.   
Wang, B.; Chang, J.; Qian, Y.; Chen, G.; Chen, J.; Jiang, Z.; Zhang, J.; Nakashima, Y.; and Nagahara, H. 2024. Di-ReCT: Diagnostic Reasoning for Clinical Notes via Large Language Models. arXiv preprint arXiv:2408.01933.   
Wang, X.; Li, C.; Wang, Z.; Bai, F.; Luo, H.; Zhang, J.; Jojic, N.; Xing, E. P.; and Hu, Z. 2023. PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization. CoRR, abs/2310.16427.   
Yang, C.; Wang, X.; Lu, Y.; Liu, H.; Le, Q. V.; Zhou, D.; and Chen, X. 2023. Large Language Models as Optimizers. CoRR, abs/2309.03409.   
Zhang, T.; Qin, P.; Deng, Y.; Huang, C.; Lei, W.; Liu, J.; Jin, D.; Liang, H.; and Chua, T. 2024. CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs in Large Language Models. CoRR, abs/2405.12063.   
Zhong, Z.; Friedman, D.; and Chen, D. 2021. Factual Probing Is [MASK]: Learning vs. Learning to Recall. In Toutanova, K.; Rumshisky, A.; Zettlemoyer, L.; Hakkani-Tür, D.; Beltagy, I.; Bethard, S.; Cotterell, R.; Chakraborty, T.; and Zhou, Y., eds., Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, 5017–5033. Association for Computational Linguistics.   
Zhou, Y.; Muresanu, A. I.; Han, Z.; Paster, K.; Pitis, S.; Chan, H.; and Ba, J. 2023. Large Language Models are Human-Level Prompt Engineers. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

# Comparison for Sampling Combination

We test the performance of rewriters trained by $\mathcal{P}(q)$ sampled by different sampling combination on the K-QA dataset. There are four sampling combinations in total, where $Q_{+}$ can be either the top $N_{+}$ questions or randomly sampled, and $Q_{-}$ can be either the bottom $N_{-}$ questions or randomly sampled. Table 6 shows that rewriters trained by the Best-Random combination, which is the combination we currently use, perform the best, failing to optimize only one metric in one model. The Random-Random combination is effective across most models and metrics. The other three compositions all have 3 or 4 metrics worse than the corresponding original.

# Dataset

We conduct experiments on three LFQA datasets that have automated evaluation criteria.

# K-QA

A medical LFQA dataset (Manes et al. 2024) collected from a medical platform featuring questions from real-world patients in which questions are carefully curated to be ansared independently without additional patient history. The evaluation method for this dataset is inspired by the FActScore (Min et al. 2023). Thus, in this dataset, each question is associated with must-have facts collected by professional annotators. These facts are then used to evaluate two metrics $S_{comp}$ and $)S_{cont}$ , considering the comprehensiveness and of the answers, respectively, defined as follows:

$$
S _ {\text { comp }} (a) = \frac {| \{x \in \text { Must\_Have } \mid a \text { entails } x \} |}{| \text { Must\_Have } |}
$$

$$
S _ {\text { cont }} (a) = | \{x \in \text { Must\_Have } \mid a \text { contradicts } x \} |,
$$

where Must\_Have represents the must-have facts for each question, a entails x indicates that the answer entails the corresponding fact, and a contradicts x indicates that the answer contradicts the corresponding fact. The entailment and contradiction are determined by LLM. The original study used GPT-4 for automatic evaluation. Due to time and cost constraints, we chose to use Llama-3-8B-instruct for automatic evaluation. The average accuracy of judgments across the dataset is 70%.

# TruthfulQA

An LFQA data (Lin, Hilton, and Evans 2022) set is created to automatically assess the truthfulness and informativeness of the answers generated by LLMs. It spans 38 categories, including health, law, finance, and politics. In order to train the automatic evaluators, it includes truthful, untruthful, informatives, and uninformative answers for each question. After fine-tuning LLMs by specific templates, we can get automatic evaluators that can assess the truthfulness and informativeness of new answers to questions in this dataset. Upon inputting a question and an answer, the evaluators output a "yes" or "no", which is used to calculate the scores for truthfulness or informativeness using the following formula:

$$
S = \frac {p _ {\mathrm{yes}}}{p _ {\mathrm{yes}} + p _ {\mathrm{no}}}
$$

The original research use a GPT-3 base evaluation model. However, due to the deprecation of the corresponding API, we have transitioned to using Llama-3-8B-instruct as the base model. The batch size we use is 32, and the learning rate we use is $5 \times 10^{-6}$ . To ensure the evaluator evaluates according to the original research standards with an accuracy exceeding 90%, we conduct training for 2 epochs on truthfulness and 1 epoch on informativeness. We achieve accuracies of 93% and 94%, respectively. To prevent any potential information leakage during the evaluation of constructing the preference dataset, extra automatic evaluators are fine-tuned exclusively on the training set question and used to evaluate the metrics for answers to rewritten questions corresponding to the training set questions.

# OASST1QA

An LFQA dataset derived from the multi-turn dialogue alignment dataset, $OASST1^{14}$ . OASST1 has several different responses along with human-annotated rankings in each dialog context. The dataset $oasst1\_pairwise^{15}$ selects the highest and lowest ranked responses from each dialogue context to construct a preference dataset. The pretrained reward model $^{16}$ we use is a binary classification model fine-tuned by oasst1\_pairwise. It can output the probability of whether a response is the highest-ranked response in a given dialogue context and the probability can be regarded as a measure of human preference, where a higher score indicates a greater preference.

We randomly sample 1000 single-round English QA pairs from the training set of oasst1\_pairwise to serve as our training dataset. We use half of the single-round English dialog from the validation set of oasst1\_pairwise as our preference dataset's validation set and the remaining half for final evaluation. We remeasure the discriminator, and its accuracy in the new testset is $75\%$ .

# Statistic

In both KQA and TruthfulQA, approximately 50% of the questions are randomly selected as the training set, with 25% of the questions serving as the validation set and test set. The selection method for OASST1QA is as described previously. Detailed statistics are in Table 1.

# Optimized Task Instruction of K-QA

The optimized task instruction of K-QA obtained after migrating our method to task level is as follows:

```txt
**To Solve the Problem Correctly:**\n\nTo provide the correct answer, please follow these steps:\n\n1.**Read the question 
```

<table><tr><td>Metric</td><td>Method</td><td>Original</td><td>Best-Random</td><td>Random-Worst</td><td>Best-Worst</td><td>Random-Random</td></tr><tr><td rowspan="6"> $S_{comp} \uparrow$ </td><td>Llama-3-8B</td><td>0.4573</td><td>0.4600</td><td>0.4276</td><td>0.4026</td><td>0.4564</td></tr><tr><td>Mistral-7B-v0.2</td><td>0.4374</td><td>0.4899</td><td>0.4643</td><td>0.4304</td><td>0.4746</td></tr><tr><td>Zephyr-7B-beta</td><td>0.4396</td><td>0.4702</td><td>0.4607</td><td>0.4586</td><td>0.4650</td></tr><tr><td>Gemma-1.1-7B</td><td>0.4010</td><td>0.4956</td><td>0.4304</td><td>0.4065</td><td>0.4655</td></tr><tr><td>GPT-3.5-turbo</td><td>0.4909</td><td>0.4978</td><td>0.4905</td><td>0.4598</td><td>0.5302</td></tr><tr><td>GPT-4o</td><td>0.5167</td><td>0.6253</td><td>0.5759</td><td>0.5382</td><td>0.5520</td></tr><tr><td rowspan="6"> $S_{cont} \downarrow$ </td><td>Llama-3-8B</td><td>0.4400</td><td>0.4000</td><td>0.4000</td><td>0.4000</td><td>0.4000</td></tr><tr><td>Mistral-7B-v0.2</td><td>0.2200</td><td>0.2600</td><td>0.2200</td><td>0.2400</td><td>0.3400</td></tr><tr><td>Zephyr-7B-beta</td><td>0.3400</td><td>0.2600</td><td>0.2800</td><td>0.3000</td><td>0.2600</td></tr><tr><td>Gemma-1.1-7B</td><td>0.5400</td><td>0.2200</td><td>0.3800</td><td>0.4000</td><td>0.3400</td></tr><tr><td>GPT-3.5-turbo</td><td>0.3200</td><td>0.2800</td><td>0.3000</td><td>0.3000</td><td>0.2600</td></tr><tr><td>GPT-4o</td><td>0.2800</td><td>0.2400</td><td>0.2000</td><td>0.2200</td><td>0.3200</td></tr></table>

Table 6: Ablation study of the different sampling combination for $\mathcal{P}(q)$ . Bold indicates the rewriter trained by the $\mathcal{P}(q)$ sampled by this combination performing better than the corresponding original for this LLM.

carefully\*\*: Make sure you understand the question and its requirements. Take a moment to re-read the question to ensure you grasp the context and what is being asked.\n2.\*\*Use the provided information\*\*: Refer to the given data, charts, or graphs to help you answer the question. Make sure to accurately interpret the information and use it to support your answer.\n3.\*\*Calculate or derive the answer\*\*: Use mathematical formulas, calculations, or logical deductions to arrive at the correct answer. Double-check your work to ensure accuracy and precision.\n4.\*\*Write your answer clearly and concisely\*\*: Use clear and concise language to express your answer. Avoid using jargon or technical terms unless absolutely necessary. Make sure your answer is easy to read and understand.\n5.\*\*Check your work\*\*: Verify your answer by re-checking your calculations, formulas, or logical deductions. Ensure that your answer is accurate, complete, and relevant to the question.\n\n\*\*Additional Tips:\*\*\n\n\* Use a calculator or spreadsheet to help with calculations, if necessary.\n\* Break down complex problems into smaller, manageable parts to make them easier to solve.\n\* Use visual aids, such as diagrams or flowcharts, to help illustrate your answer.\n\* Check for errors or inconsistencies in your work to ensure accuracy and precision.\n\nBy following these steps and tips, you

will be well on your way to providing the correct answer and solving the problem correctly.

# Question Evaluation Templates

In this section, we present the templates used to evaluate question attributes. There are a total of ten attributes assessed, namely non-leading, word choice, tone, conciseness, neutrality, grammar and spelling, structure, politeness, clarity, and emotion. We provide detailed temples in Table 7, Table 8 and Table 9.

<table><tr><td>Attributes</td><td>Templates</td></tr><tr><td>Word Choice</td><td>Evaluate the word choice of the following question based on the specified criteria and provide an explanation for your rating.\n\n\n- **1 point - Very Poor:** The question uses non-professional or colloquial language.\n- **2 points - Poor:** The question uses some non-professional language or terminology.\n- **3 points - Moderate:** The question uses mostly professional language but may include some informal terms.\n- **4 points - Good:** The question uses professional and precise language.\n- **5 points - Excellent:** The question uses highly professional and precise language, appropriate for the context.\n\n**Question:** GIVEN QUESTION\n**Rating:** {1-5points}\n**Reasoning:**\n**Example:**\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n**Rating:** 5 points\n**Reasoning:**\n- The word choice is highly professional and precise, suitable for a medical inquiry, making it appropriate and respectful for the context.</td></tr><tr><td>Tone</td><td>Rate the tone of the following question based on the specified criteria and provide an explanation for your rating.\n\n\n- **1 point - Very Unprofessional Tone:** The question uses informal language, including slang or overly casual expressions that are inappropriate for a professional setting.\n\n- **2 points - Unprofessional Tone:** The question is somewhat informal, possibly including casual language that may not be suitable for all professional contexts.\n\n- **3 points - Moderately Professional Tone:** The question uses generally professional language but may include slight informal elements that could be polished further.\n\n- **4 points - Quite Professional Tone:** The question maintains a professional tone, using appropriate language for the context and avoiding casual terms.\n\n- **5 points - Very Professional Tone:** The question exemplifies a highly professional tone, using formal language that is perfectly suited for any professional setting.\n\n**Question:** GIVEN QUESTION\n**Rating:** {1-5 points}\n**Reasoning:**\n**Example:**\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n**Rating:** 5 points\n**Reasoning:**\n- The question maintains a highly professional tone, using formal and respectful language suitable for a medical inquiry. The use of "please" adds a courteous touch, enhancing the professional quality of the communication.</td></tr><tr><td>Conciseness</td><td>Evaluate the conciseness of the following question based on the specified criteria and provide an explanation for your rating.\n\n\n- **1 point - Very Verbose:** The question is verbose, containing much irrelevant information that obscures the main point.\n- **2 points - Verbose:** The question includes unnecessary details that detract from the main point, making it less concise.\n- **3 points - Moderately Concise:** The question is generally concise, though it contains a few extraneous details.\n- **4 points - Quite Concise:** The question is concise and to the point, with no unnecessary information.\n- **5 points - Extremely Concise:** The question is extremely concise, containing only essential information needed to answer the question effectively.\n\n**Question:** GIVEN QUESTION\n**Rating:** {1-5 points}\n**Reasoning:**\n**Example:**\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n**Rating:** 5 points\n**Reasoning:**\n- The question is extremely concise, focusing solely on the essential information required. It directly asks for detailed options for oral medication without adding any superfluous details, ensuring the query is clear and straightforward.\n\nPlease rate the given question using this template.</td></tr><tr><td>Non-leading</td><td>Evaluate whether the following question is non-leading based on the specified criteria and provide an explanation for your rating.\n\n- **1 point - Highly Leading:** The question is leading, suggesting a particular answer or outcome.\n- **2 points - Somewhat Leading:** The question is somewhat leading, with language that hints at a specific answer.\n- **3 points - Moderately Non-leading:** The question is mostly non-leading, with only slight suggestions towards an answer.\n- **4 points - Quite Non-leading:** The question is non-leading, asking for information without suggesting an answer.\n- **5 points - Completely Non-leading:** The question is entirely non-leading, purely seeking information without any implied bias towards certain answers.\n\n**Question:** GIVEN QUESTION\n**Rating:** {1-5 points}\n**Reasoning:**\n**Example:**\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n**Rating:** 5 points\n**Reasoning:**\n- The question does not suggest any particular answer or bias, simply requesting information, making it completely non-leading.\n\nPlease rate the given question using these templates.</td></tr><tr><td>Politeness</td><td>Evaluate the politeness of the following question based on the specified criteria and provide an explanation for your rating.\n\n\n- **1 point - Very Impolite:** The question is rude or disrespectful, lacking basic courtesy.\n- **2 points - Impolite:** The question may come across as brusque or somewhat disrespectful.\n- **3 points - Moderately Polite:** The question is polite but may lack warmth or additional elements of courteous language.\n- **4 points - Quite Polite:** The question is polite and respectful, using appropriate expressions of courtesy.\n- **5 points - Very Polite:** The question is very polite and courteous, including expressions that show respect and consideration.\n\n\n**Question:** GIVEN QUESTION\n\n**Rating:** {1-5 points}\n\n**Reasoning:**\n\n**Example:**\n\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n\n**Rating:** 5 points\n\n**Reasoning:**\n- The use of "please" shows a high level of politeness and respect, making the question very courteous and considerate in tone.</td></tr><tr><td>Grammar and Spelling</td><td>Evaluate the grammar and spelling of the following question based on the specified criteria and provide an explanation for your rating.\n\n\n- **1 point - Very Poor:** The question contains numerous grammatical and spelling errors.\n- **2 points - Poor:** The question has several grammatical and spelling errors.\n- **3 points - Moderate:** The question has occasional grammatical or spelling errors.\n- **4 points - Good:** The question has minor or no grammatical and spelling errors.\n- **5 points - Excellent:** The question is free from grammatical and spelling errors.\n\n\n**Question:** GIVEN QUESTION\n\n**Rating:** {1-5 points}\n\n**Reasoning:**\n\n**Example:**\n\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n\n**Rating:** 5 points\n\n**Reasoning:**\n- The question is free from grammatical and spelling errors, demonstrating excellent use of language.</td></tr><tr><td>Structure</td><td>Evaluate the structure of the following question based on the specified criteria and provide an explanation for your rating.\n\n\n- **1 point - Very Poorly Structured:** The question is poorly structured and difficult to follow, lacking logical sequence.\n- **2 points - Poorly Structured:** The question has some structure, but it lacks effective organization and logical flow.\n- **3 points - Moderately Well-Structured:** The question is reasonably well-structured but could be improved for better understanding and flow.\n- **4 points - Well-Structured:** The question is well-organized, making it easy to follow and understand.\n- **5 points - Excellently Structured:** The question is excellently structured, enhancing quick and clear comprehension with a logical and coherent sequence.\n\n\n**Question:** GIVEN QUESTION\n\n**Rating:** {1-5 points}\n\n**Reasoning:**\n\n**Example:**\n\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n\n**Rating:** 5 points\n\n**Reasoning:**\n- The question is excellently structured, presenting a clear and direct inquiry. The logical progression from a general request for information to specific details about oral medication options makes it easy for the responder to understand and address.</td></tr><tr><td>Neutrality</td><td>Evaluate the neutrality of the following question based on the specified criteria and provide an explanation for your rating.\n\n\n- **1 point - Very Biased:** The question is heavily biased, containing strong subjective opinions and positions.\n- **2 points - Biased:** The question shows some bias, indicating a leaning towards a particular view or opinion.\n- **3 points - Moderately Neutral:** The question is generally neutral but includes minor subjective elements.\n- **4 points - Quite Neutral:** The question is largely neutral, with very little subjective language.\n- **5 points - Completely Neutral:** The question is entirely neutral, free of any subjective opinions or biases.\n\n\n**Question:** GIVEN QUESTION\n\n**Rating:** {1-5 points}\n\n**Reasoning:**\n\n**Example:**\n\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n\n**Rating:** 5 points\n\n**Reasoning:**\n- The question is presented without any subjective bias, focusing solely on gathering factual information about oral medication options for treating scabies, making it completely neutral.</td></tr><tr><td>Emotion</td><td>Evaluate the emotional tone of the following question based on the specified criteria and provide an explanation for your rating.\n\n\n- **1 point - Highly Emotional:** The question uses highly emotional, exaggerative, or sensational language.\n- **2 points - Emotional:** The question displays some emotional language, though not excessive.\n- **3 points - Moderately Emotional:** The question contains a balance of emotional and neutral language.\n- **4 points - Minimally Emotional:** The question uses minimal emotional language, maintaining an objective tone.\n- **5 points - Emotionally Neutral:** The question is emotionally neutral, using precise and factual language.\n\n\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n\n**Rating:** {1-5 points}\n\n**Reasoning:**\n\n**Example:**\n\n**Question:** GIVEN QUESTION\n\n**Rating:** 5 points\n\n**Reasoning:**\n- The question is posed in a straightforward and factual manner, devoid of any emotional language, making it emotionally neutral.</td></tr><tr><td>Clarity</td><td>Evaluate the clarity of the following question based on the specified criteria and provide an explanation for your rating.\n\n\n- **1 point - Very Unclear:** The question is ambiguous and difficult to understand.\n- **2 points - Unclear:** The question is partially clear but could be misunderstood.\n- **3 points - Moderately Clear:** The question is understandable but could benefit from further clarification.\n- **4 points - Quite Clear:** The question is clear and easy to understand.\n- **5 points - Very Clear:** The question is completely clear, with no ambiguity.\n\n\n**Question:** Can you please provide detailed information on the oral medication options available for treating scabies?\n\n**Rating:** {1-5 points}\n\n**Reasoning:**\n\n**Example:**\n\n**Question:** GIVEN QUESTION\n\n**Rating:** 5 points\n\n**Reasoning:**\n- The question is structured and phrased clearly, making it easy to understand and directly addressing the need for information on oral medication options.</td></tr></table>

Table 7: Examples of Question Evaluation Template 1

Table 8: Examples of Question Evaluation Template 2

Table 9: Examples of Question Evaluation Template 3