# TWO-STAGE LLM FINE-TUNING WITH LESS SPECIALIZATION AND MORE GENERALIZATION

# Yihan Wang\*

Department of Computer Science

UCLA

wangyihan617@gmail.com

# Si Si

Google

sisidaisy@google.com

# Daliang Li

Google

daliangli@google.com

# Michal Lukasik

Google

mlukasik@google.com

# Felix Yu

Google

felixyu@google.com

# Cho-Jui Hsieh

Department of Computer Science
UCLA

chohsieh@cs.ucla.edu

# Inderjit S Dhillon

Google

isd@google.com

# Sanjiv Kumar

Google

sanjivk@google.com

# ABSTRACT

Pretrained large language models (LLMs) are general purpose problem solvers applicable to a diverse set of tasks with prompts. They can be further improved towards a specific task by fine-tuning on a specialized dataset. However, fine-tuning usually makes the model narrowly specialized on this dataset with reduced general in-context learning performances, which is undesirable whenever the fine-tuned model needs to handle additional tasks where no fine-tuning data is available. In this work, we first demonstrate that fine-tuning on a single task indeed decreases LLMs' general in-context learning performance. We discover one important cause of such forgetting, format specialization, where the model overfits to the format of the fine-tuned task. We further show that format specialization happens at the very beginning of fine-tuning. To solve this problem, we propose Prompt Tuning with MOdel Tuning (ProMoT), a simple yet effective two-stage fine-tuning framework that reduces format specialization and improves generalization. ProMoT offloads task-specific format learning into additional and removable parameters by first doing prompt tuning and then fine-tuning the model itself with this soft prompt attached. With experiments on several fine-tuning tasks and 8 in-context evaluation tasks, we show that ProMoT achieves comparable performance on fine-tuned tasks to standard fine-tuning, but with much less loss of in-context learning performances across a board range of out-of-domain evaluation tasks. More importantly, ProMoT can even enhance generalization on in-context learning tasks that are semantically related to the fine-tuned task, e.g. ProMoT on En-Fr translation significantly improves performance on other language pairs, and ProMoT on NLI improves performance on summarization. Experiments also show that ProMoT can improve the generalization performance of multi-task training.

# 1 INTRODUCTION

Natural language processing (NLP) has recently been revolutionized by scaling up transformer based large language models (LLMs) together with large-scale pretraining (Vaswani et al., 2017; Devlin et al., 2019; Raffel et al., 2020a; Brown et al., 2020; Rae et al., 2021; Chowdhery et al., 2022; Smith et al., 2022; Touvron et al., 2023). In addition to improved downstream performances, these pretrained LLMs can perform a broad array of unforeseen tasks when provided with a prompt. This in-context

learning capability allows users to flexibly re-purpose LLMs for specific tasks with a minimum amount of supervised data, making it extremely convenient for fast prototyping and experimentation, especially in the low data regime.

However, even the largest and most advanced LLMs leave a lot to be improved. Grounding and eliminating hallucinations (Maynez et al., 2020), reasoning and logical clarity (Creswell & Shanahan, 2022), mathematics (Brown et al., 2020; Noorbakhsh et al., 2021) are just a few examples where LLMs still lag behind the best human performances, or in some cases, the fine-tuned performances of the same model.

The most common practice to improve a pretrained model is to fine-tune it on a specialized task or several tasks. However, fine-tuning on LLM usually causes over-specialization to the fine-tuning tasks, and harm the model's pre-existing generalization ability on unseen tasks via in-context learning. As we show later, an mT5 model finetuned on a single task loses its few-shot performance on unseen tasks within one thousand steps of fine-tuning. When faced with hundreds of downstream tasks and even unknown tasks, we expect to have a single fine-tuned model that is both superior on supervised fine-tuned tasks and general unseen tasks. Thus, it becomes very important to develop new techniques for finetuning that prevent over-specialization of these fine-tuned models only to a few tasks.

<table><tr><td>Ground-Truth Output</td><td>Mercedes’ Lewis Hamilton took the outright championship lead for the first time this season with a dominant victory in the Italian Grand Prix.</td></tr><tr><td>Pretrained mT5</td><td>Hamilton won the British Grand Prix.</td></tr><tr><td>Fine-tuned mT5 on RTE</td><td>True</td></tr><tr><td>Fine-tuned mT5 with ProMoT (Ours) on RTE</td><td>Lewis Hamilton won the Italian Grand Prix.</td></tr></table>

Table 1: Output comparison of pretrained and fine-tuned mT5 models vs. fine-tuned with ProMoT on the RTE binary classification NLI dataset, performing in-context 1-shot summarization.

In this work, we discover that the loss of general in-context learning abilities during fine-tuning is, to a large extent, caused by format specialization, which makes model overfitting to the specific task format. For example, an mT5 (Xue et al., 2020) model learns in the output space with only “True” and “False” if we fine-tune it on a binary classification dataset, losing its ability to flexibly generate different output styles according to the in-context prompts of other tasks. We show that format specialization tends to happen at the very beginning of fine-tuning, before the model fully learns the semantic content of the task.

Based on these observations, we propose a simple solution to alleviate format specialization: PROmpt Tuning with MOdel Tuning (ProMoT), which off-loads format learning to a small amount of task-specific parameters that are external to the model. ProMoT is a two-stage fine-tuning process. At the first stage, we freeze the pretrained model and tune a small set of additional parameters, where we find adding soft prompt before the input (Lester et al., 2021) is a good choice. At the second stage, we freeze the additional parameters and tune the main model. Since format information is learned first, it mostly enters the small set of additional parameters. At inference time, we can decide whether to remove the additional parameters depending on whether the incoming task share the same format as the fine-tuned task.

Our experiments show that ProMoT significantly alleviates specialization during fine-tuning, while boosting generalization on semantically related tasks with different formats. For example, fine-tuning the model only on an NLI binary classification dataset, a mT5 XXL model consistently obtains improved in-context learning performance on summarization compared with the pretrained model, possibly due to improved grounding learned from NLI. See Table 1 for a concrete example. With ProMoT, we can obtain models with both better supervised performance compared to pretrained models and better general in-context learning performance compared to standard finetuning.

To summarize, our contributions are 4-fold:

- We show empirically that general in-context learning capabilities decrease during single-task fine-tuning for T5 models. We identify format specialization as one of the important causes which mostly happens at the beginning of fine-tuning.   
- We propose a novel 2-stage fine-tuning framework: PROmpt Tuning with MOdel Tuning (ProMoT) to reduce format specialization during fine-tuning.

- Experiments on 10+ NLP tasks show that ProMoT significantly reduces specialization of fine-tuned models compared to standard fine-tuning, while reaching similar supervised performance. The reduction in specialization opens up opportunities to enhance generalization across very dissimilar tasks when they share some semantic aspects.   
- ProMoT can be combined with many existing fine-tuning and parameter-efficient fine-tuning methods. We show examples where ProMoT is combined with multi-task fine-tuning and fine-tuning with 1-shot prompts to further boost the generalization on unseen tasks.

# 2 RELATED WORK

Pretrained LLMs are general problem solvers with in-context prompts (Raffel et al., 2020b; Xue et al., 2020; Radford et al., 2018; Chowdhery et al., 2022; Min et al., 2022; Touvron et al., 2023). Zhai et al. (2023) evaluates the catastrophic forgetting in multimodal language model fine-tuning, which is limited to image classification tasks. Chan et al. (2022); Gao et al. (2020) study the effect of pretraining data distribution on in-context learning on image recognition tasks, where the tension between in-context learning tasks and fine-tuning tasks is discussed. They propose changing the data distribution to ease such tension, which could be difficult for generative NLP tasks. ProMoT is an orthogonal method that does not require changes in data distribution.

In a recent study, Ramasesh et al. (2022) found that as model size increases, the model becomes less prone to catastrophic forgetting. However such studies are mostly focused on tasks of similar format, e.g. a sequence of different classification tasks. In this work we explore vastly different tasks, e.g. classification v.s. long form generation where the format itself is critical.

Different from full fine-tuning, prompt-tuning (Lester et al., 2021; Zhang et al., 2021), adapters and LoRA (Hu et al., 2021; He et al., 2021; Houlsby et al., 2019) adapt a pretrained model to a task with a small set of tunable parameters. Parameter-efficient methods like these largely leave the pretrained model intact, which can preserve the pre-existing in-context learning abilities. However, they also miss the opportunity to further improve the pretrained model with a small, high quality dataset that generalizes beyond the fine-tuned task. Besides, these parameter-efficient methods also underperform fine-tuning on the supervised task in many cases, as shown in (Lester et al., 2021; Liu et al., 2021) and in our results.

Another line of work uses multi-task fine-tuning to improve generalization on unseen in-context learning tasks. Wei et al. (2021a); Chung et al. (2022) fine-tune PaLM and T5 on large-scale multitask datasets with diverse natural language prompts, improving the zero- and few-shot performance on unseen tasks. Min et al. (2021) incorporate the in-context learning objective into fine-tuning on multitask datasets with few-shot prompts. This approach relies on multi-task training to generalize, while orthogonally, ProMoT improves the generalization of each single fine-tuning task, whether used in a multi-task setting or not. ProMoT can indeed be combined with multi-task training to obtain better generalization as we demonstrate in Sec. 5.4. In addition, such approaches often require human engineered instructions or prompts for each task to partly alleviate format specialization, while ProMoT uses prompt tuning, which has two advantages: 1) ProMoT does not require the elaborate trial and error of prompt engineering as it optimizes the soft prompts with data. 2) Soft prompts are more effective at absorbing the format compared to natural language prompts, as shown in Table 6.

# 3 FORMAT SPECIALIZATION IN FINE-TUNING CAUSES THE LOSS OF IN-CONTEXT LEARNING CAPABILITIES

In this section, we first show empirically with an mT5 XXL model that 1) in-context learning abilities are lost during fine-tuning; 2) format specialization is an important cause for such loss; 3) format specialization happens at the very beginning of fine-tuning.

# 3.1 LOSS OF IN-CONTEXT LEARNING CAPABILITIES DURING FINE-TUNING

In this subsection, we first show that the in-context learning performance usually drops significantly after standard fine-tuning.

In our experiments, we fine-tune a pretrained mT5 XXL model (13B parameters) (Xue et al., 2020) on the Recognizing Textual Entailment (RTE) dataset (Wang et al., 2019). In RTE tasks, the model is required to predict “True” or “False” for whether the two given sentences are entailed. We fine-tune

the mT5 model with default hyper-parameters and input/output template used in PaLM (Chowdhery et al., 2022).

![](images/128b5c596f80077788d6b41694c7668252665a5335154b8496ca956c3164d5ed.jpg)

<details>
<summary>line</summary>

| Fine-tune steps | RTE   | 1-shot triviaQA | 1-shot web_questions |
| --------------- | ----- | ---------------- | -------------------- |
| 0               | 0     | 85               | 45                   |
| 50              | 50    | 75               | 40                   |
| 100             | 50    | 70               | 35                   |
| 150             | 50    | 60               | 30                   |
| 200             | 80    | 50               | 25                   |
| 250             | 85    | 40               | 20                   |
| 300             | 85    | 15               | 10                   |
| 350             | 85    | 15               | 5                    |
| 400             | 85    | 15               | 2                    |
| 450             | 90    | 10               | 1                    |
| 500             | 90    | 5                | 0                    |
</details>

Figure 1: Loss of in-context learning abilities during fine-tuning. We show the learning curve of a model being fine-tuned on RTE dataset while being tested on 1-shot QA datasets. Left axis: Accuracy on RTE. Right axis: Exact match rate on 1-shot QA.

![](images/f12f52843239caaf6d9b2044286c675cd3d7dd49a3441a93ef4975c1f5b6b4fe.jpg)

<details>
<summary>bar_line</summary>

| Fine-tune steps | Accuracy | Ratio of True/False |
| --------------- | -------- | ------------------- |
| 0               | 0        | 0.0                 |
| 50              | 50       | 0.2                 |
| 100             | 50       | 0.4                 |
| 150             | 50       | 0.6                 |
| 200             | 80       | 0.8                 |
| 250             | 80       | 0.9                 |
| 300             | 90       | 1.0                 |
| 350             | 90       | 1.0                 |
| 400             | 95       | 1.0                 |
| 450             | 95       | 1.0                 |
| 500             | 100      | 1.0                 |
</details>

Figure 2: Format specialization in fine-tuning: showing the frequency of "True/False" style outputs when evaluated on 1-shot TriviaQA. The model is being fine-tuned on RTE. Left axis: Accuracy on RTE. Right axis: Ratio of True/False.

We want to see whether the model lost its in-context learning abilities on unseen task during fine-tuning. Therefore, we evaluate the fine-tuned model with two 1-shot QA tasks, TriviaQA (Joshi et al., 2017) and web\_questions (Berant et al., 2013). The results are illustrated in Figure 1, where we can see that when the accuracy on RTE dataset increases with fine-tuning, performance on few-shot QA tasks drops drastically. This phenomenon is general and not a result of specific fine-tuning or evaluation tasks (more results in Section 5.3).

# 3.2 FORMAT SPECIALIZATION

Why are the in-context learning abilities of an LLM so easily lost after a few hundred steps of fine-tuning? A natural hypothesis is that due to the homogeneity of output formats in fine-tuning datasets, the model quickly specializes to this task format and learns to follow it no matter what the input sequence is. This leads to the loss of in-context learning abilities on other tasks that do not share the same format. here by “format” we refer to the common characteristics of the sequences in fine-tuning task as a subset of all possible sequences, such as the language used, typical input/output lengths and styles, special tokens or punctuation, upper/lower case styles etc. For example, the output format of RTE is a set of two labels, “True” or “False”, among all possible sequences of tokens of various lengths. Since all data points share the same format in single-task fine-tuning, the model receives a strong gradient signal that the output should follow this format, thus its in-context learning performance on other tasks with different formats will drop, even when they share important semantic similarities with the fine-tuned task.

To verify this hypothesis, we evaluate the RTE fine-tuned mT5 model on 1-shot TriviaQA task and count the percentage of outputs which are “True” or “False”. Figure 2 shows that as the fine-tuning proceeds, the model outputs more “True” or ‘False’ even with a 1-shot prompted input from TriviaQA. In particular, after 300 fine-tuning steps, 90% of the output becomes “True” or “False”. The same phenomenon happens on other in-context learning tasks. With a 1-shot WMT16 En-De translation prompt, after 500 steps of RTE fine-tuning, more than 99% of the output becomes “True” or “False”. This indicates that format specialization is a possible reason for the loss of general in-context learning capabilities during fine-tuning.

# 3.3 FORMAT LEARNING HAPPENS FIRST DURING STANDARD FINE-TUNING

Next, we show experimental evidence that format learning happens first during standard fine-tuning. This is not surprising as the overwhelming majority of fine-tuning data points have very similar formats, causing a gradient signal that dominates over others, more nuanced elements such as the semantic content of the task.

More concretely, for the RTE dataset, the “format” refers to the fact that the output $\in \{True, False\}$ , while the semantic content refers to the correlation between the input sequence and the output label.

![](images/3622a5e3977d560838f5ad0c772b549fe8573c38ac2c3dcc41aba3f0c284192e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Model: frozen"] --> B["Soft prompt tuning by gradient descent"]
    B --> C["Input"]
    D["Model tuning by gradient descent"] --> E["Soft prompt: frozen"]
    E --> F["Input"]
    G["Evaluation"] --> H["Model: 2-stage finetuned"]
    I["OR"] --> J["Input"]
    I --> K["Trained prompt"]
    I --> L["Input"]
```
</details>

Figure 4: Overview of ProMoT, our two-stage fine-tuning strategy. We run prompt tuning at Stage 1 and model fine-tuning with the trained prompt at Stage 2. Green denotes trainable parameters and blue means frozen.

We isolate format learning from semantic learning by creating a randomized RTE dataset where the output labels are randomly shuffled, thus are no longer correlated with the input sequences. The gradients of format learning, $g_{format}$ , are then given by the gradients on the randomized RTE dataset. By comparing with the full gradient g on the original RTE we can detect when format learning happens during fine-tuning. We compute the gradients on the same batches of inputs for the two different settings. Figure 3 and Figure 7 in Appendix show that at the very beginning of fine-tuning (step 0), the full gradient g is highly aligned with the format-only gradient $g_{format}$ , signified by $\cos(\langle g_{0}, g_{\text{format},0} \rangle) \approx 1$ . Since randomized RTE and original RTE share the format information only and contain totally different semantic content, this alignment implies that the model is mostly learning the format. After 400 fine-tuning steps, this alignment disappears where the cosine similarity drops to around $0.2^{1}$ , when the True/False ratio reaches nearly 100%.

# 4 PROPOSED METHOD: PROMPT TUNING WITH MODEL TUNING (PROMOT)

The observations from Section 3 inspire us to decouple format learning from fine-tuning, in order to alleviate specialization to the fine-tuned task and preserve general in-context learning abilities. The key idea is to offload format learning to a separate small set of parameters during early fine-tuning, and allow the model's own parameter changes afterwards to focus more on the semantic content of the task. We propose a two-stage fine-tuning strategy called ProMoT, illustrated in Figure 4. At the first stage, ProMoT uses prompt tuning to capture the format in a trainable soft prompt while the model itself is frozen. At the second stage, ProMoT freezes the learned soft prompt and fine-tunes the model itself to focus on semantic skills that might be more transferable.

Stage 1: Prompt Tuning. Here we use a continuous trainable prompt (soft prompt) (Lester et al., 2021) prepended before the embedded inputs as the separate small set of tunable parameters. The soft prompt for a given fine-tuned task $P_{e} \in \mathbb{R}^{p \times e}$ is a small set of free parameters taking the form of a few trainable embeddings, where $p$ is the prompt length and $e$ is the embedding size. Given an input sequence, prompt tuning first embeds it with the text embedding layer of the pretrained model, and then prepends it with the soft trainable prompt. The soft prompt is then optimized to reduce the loss while the pretrained model is frozen. As indicated in Section 3.3, fine-tuning first learns the format. We expect that by prompt tuning first, the soft prompt will learn the format. Although it is not guaranteed that

![](images/502720b628ae71ebcd6fe883adf4d2338f8b0e9ebb3c1b41ba179b61b4e3d3e2.jpg)

<details>
<summary>bar</summary>

| Modules       | cos(< g₀, g_format, 0 >) | cos(< g₄₀₀, g_format, 400 >) |
| ------------- | ------------------------ | ----------------------------- |
| MLP kernel    | 0.98                     | 0.10                          |
| Query         | 0.96                     | 0.08                          |
| Key           | 0.99                     | 0.05                          |
| Value         | 0.99                     | 0.20                          |
</details>

Figure 3: Format specialization happens at the beginning of fine-tuning: we show the cosine similarity between the full gradient g and the format gradient $g_{format}$ on the MLP kernel, Query, Key and Value on the attention module. The g and $g_{format}$ are much better aligned at the start of training, compared to at 400 steps. Comparison between more steps can be found in Figure 7 (Appendix).

the soft prompt only learns the format, the small capacity can prevent the soft prompt from learning all semantic skills in most realistic NLP tasks, as demonstrated by the performance gap between prompt tuning and standard fine-tuning.

Stage 2: Fine-tuning with trained prompt. After prompt-tuning, we expect the trained prompt now storing most of the format information. We then freeze the soft prompt and fine-tune the pretrained model. Importantly, as shown in Figure 4, the soft prompt is still prepended before the input during this stage, forcing the model to learn things not captured already by the soft prompt.

Other parameter-efficient and fine-tuning methods. ProMoT is a general framework that can be combined with different parameter-efficient tuning and fine-tuning techniques in respective stages. Conceptually, the prompt-tuning at the first stage can be replaced by other commonly used parameter-efficient methods such as LoRA Hu et al. (2021). However, empirically we found prompt-tuning is much better than LoRA on absorbing format information in early fine-tuning. More discussions can be found in Appendix C.7. For fine-tuning methods, we show examples to combine ProMoT with multi-task fine-tuning (Section 5.4) and 1-shot in-context learning prompt (Section 5.3, Section 5.4). Training with 1-shot prompt is introduced by Min et al. (2021) in a multi-task training setting.

Evaluation. After the two-stage fine-tuning, we obtain a fine-tuned model checkpoint and a trained soft prompt for a specific fine-tuning target task. We expect the soft prompt stores most of the format information, and we only use this prompt during inference when the inference task has the same format as the fine-tuned target task. Otherwise, we remove the learned prompt and simply feed the original input into the fine-tuned model.

# 5 EXPERIMENTS

# 5.1 SETTINGS

Datasets. We use RTE (Wang et al., 2019; Bentivogli et al., 2009) and WMT14 En-Fr (Bojar et al., 2014) as two fine-tuning tasks in our main experiments. They are selected as examples of classification (RTE) and generative tasks (WMT14 En-Fr translation). Experiments on additional fine-tuning tasks including SNLI (Bowman et al., 2015) and OpenbookQA (Mihaylov et al., 2018) can be found in Appendix C.

We use 8 tasks unseen during fine-tuning to evaluate the model's generalization abilities. The 8 evaluation tasks are chosen to represent four types of tasks:

- Natural language inference: CB (De Marneff et al., 2019) and WiC (Pilehvar & Camacho-Collados, 2018) from superGLUE (Wang et al., 2019)   
- Closed book QA: TriviaQA (Joshi et al., 2017), web\_questions (Berant et al., 2013)   
• Translation: WMT16 En-Ro, WMT16 En-De (Bojar et al., 2016)   
- Summarization: XSum (Narayan et al., 2018), WikiLingua (Ladhak et al., 2020)

For each evaluation task, we use 1-shot and 4-shots prompts and task templates from PaLM (Chowdhery et al., 2022) as described in the Appendix 7.

Metrics. We report accuracy for classification tasks, exact match ratio for QA tasks, BLEU score (Papineni et al., 2002) for translation tasks and Rouge-2 score (Lin, 2004) for summarization tasks. We evaluate the model on development set for superGLUE sub-tasks (RTE, CB and WiC) and on test set for all other tasks. Besides per-task performance, we also report the normalized average (Norm. Avg.) performance on all evaluation tasks by averaging the performances normalized to [0,100], following the "normalized preferred metric" in BIG-bench (Srivastava et al., 2022) and Chung et al. (2022).

Models. We primarily use mT5 (Xue et al., 2020) XXL model (Raffel et al., 2020b) in our main experiments, which is pretrained on multi-lingual corpus and contains 13B parameters. This is to accommodate multi-lingual scenarios among our training and evaluation tasks. To show the effectiveness of our method on different pretraining corpus, model sizes and architectures, we also include experiments on mT5 XL, T5.1.1 XXL and PaLM 8b in Appendix C. T5 based models

are shown to have meaningful few-shot performance as shown in Chung et al. (2022). We do not consider FLAN-T5 (Chung et al., 2022) as a base model in our experiments because it has already been fine-tuned on a large amount of supervised datasets, including our evaluation datasets. More experimental details can be found in Appendix B.

Comparing methods. We compare our ProMoT with several different configurations, including

- Pretrained model: We evaluate the pretrained model on all tasks without any fine-tuning.   
- Standard fine-tuning: Fine-tune the pretrained model without trainable prompts. We also include a multi-task version in Section 5.4 which is commonly used to boost model generalization on unseen tasks.   
- Prompt tuning: Tune the trainable prompt with pretrained model frozen. As the model is fixed, prompt tuning will not change the pretrained model's performance on in-context learning tasks comparing when the prompt is removed.   
- Our proposed method: ProMoT: Our proposed two-stage fine-tuning strategy.   
- Our proposed method: ProMoT+1-shot: To further boost in-context learning performance, we prepend a 1-shot example to the input in Figure 4 during training.

# 5.2 SUPERVISED PERFORMANCE ON FINE-TUNING TASKS

We first show that ProMoT training can achieve similar or even better performance on fine-tuning tasks compared to standard fine-tuning. We apply three different fine-tuning methods on four different tasks and report the result in Table 2. We report the best performance within the same number of fine-tuning steps (See Appendix B for more details). ProMoT outperforms standard fine-tuning on supervised performance on 3 out of 4 fine-tuning target tasks and outperforms prompt-tuning on 4 out of 4 tasks. Therefore the improved in-context learning performance on unseen tasks (better generalization ability), as will be demonstrated in the next few sections, comes without sacrificing the fine-tune task's performance.

<table><tr><td></td><td>Prompt tuning</td><td>Standard Fine-tuning</td><td>ProMoT (Ours)</td></tr><tr><td>RTE</td><td>91.34</td><td> $\underline{92.06}$ </td><td> $\underline{92.78}$ </td></tr><tr><td>WMT14 En-Fr</td><td>39.28</td><td> $\underline{41.80}$ </td><td> $\underline{41.30}$ </td></tr><tr><td>SNLI</td><td>88.53</td><td> $\underline{88.91}$ </td><td> $\underline{89.62}$ </td></tr><tr><td>OpenbookQA</td><td>73.60</td><td> $\underline{77.2}$ </td><td> $\underline{81.6}$ </td></tr></table>

Table 2: Comparison of supervised performances of a mT5 XXL model on fine-tuning target tasks. We use 0-shot in fine-tuning tasks. We report accuracy for RTE, SNLI and OpenbookQA, and BLEU score for WMT14 En-Fr.

# 5.3 GENERALIZATION WITH SINGLE TASK FINE-TUNING

<table><tr><td></td><td></td><td colspan="2">Pretrained</td><td colspan="2">Standard Fine-tuning</td><td colspan="2">ProMoT (Ours)</td><td colspan="2">ProMoT + 1-shot (Ours)</td></tr><tr><td>Fine-tuning</td><td>RTE</td><td colspan="2">47.653</td><td colspan="2">92.06</td><td colspan="2">92.78</td><td colspan="2">93.86</td></tr><tr><td rowspan="10">Evaluation</td><td></td><td>1-shot</td><td>4-shots</td><td>1-shot</td><td>4-shots</td><td>1-shot</td><td>4-shots</td><td>1-shot</td><td>4-shots</td></tr><tr><td>Norm. Avg.</td><td>17.52</td><td>18.75</td><td>15.43(-2.10)</td><td>16.56(-2.19)</td><td>20.10(+2.58)</td><td>21.24(+2.49)</td><td>22.26(+4.74)</td><td>22.33(+3.58)</td></tr><tr><td>CB</td><td>46.43</td><td>51.79</td><td>73.21</td><td>82.14</td><td>66.07</td><td>67.86</td><td>83.93</td><td>82.14</td></tr><tr><td>WiC</td><td>49.69</td><td>49.69</td><td>50.00</td><td>50.16</td><td>51.41</td><td>53.61</td><td>51.25</td><td>50.63</td></tr><tr><td>triviaQA</td><td>17.58</td><td>19.02</td><td>0.15</td><td>0.11</td><td>17.64</td><td>18.66</td><td>17.82</td><td>19.62</td></tr><tr><td>web_questions</td><td>9.70</td><td>13.04</td><td>0.05</td><td>0.05</td><td>11.07</td><td>13.19</td><td>10.14</td><td>12.11</td></tr><tr><td>WMT16_ende</td><td>3.97</td><td>8.83</td><td>0.00</td><td>0.00</td><td>2.02</td><td>3.69</td><td>2.26</td><td>4.89</td></tr><tr><td>WMT16_enro</td><td>1.82</td><td>3.92</td><td>0.00</td><td>0.00</td><td>0.70</td><td>0.96</td><td>0.87</td><td>1.87</td></tr><tr><td>XSum</td><td>6.41</td><td>2.35</td><td>0.00</td><td>0.00</td><td>7.02</td><td>7.01</td><td>6.94</td><td>3.93</td></tr><tr><td>WikiLingua/en</td><td>4.59</td><td>1.33</td><td>0.00</td><td>0.00</td><td>4.84</td><td>4.90</td><td>4.87</td><td>3.43</td></tr></table>

Table 3: Performances of a mT5 XXL model finetuned on RTE and evaluated on 8 different tasks to verify the generalization ability. The accuracy on fine-tuned task (RTE) is in the first row. We compare the Norm. Avg. (normalized average performance) with pretrained model and report the relative difference, where red denotes decreased performance and blue denotes increased performance. CB and WiC are also NLI tasks, very similar to RTE.

In this section, we evaluate and compare the few-shot performance on unseen tasks after fine-tuning. We show the evaluation results of fine-tuning on RTE and WMT14 En-Fr in Table 3 and Table 4,

respectively. Experiments on additional fine-tuning tasks SNLI/OpenbookQA and additional base models including mT5 XL, T5.1.1 XXL and PaLM 8b can be found in Appendix C.

<table><tr><td></td><td></td><td colspan="2">Pretrained</td><td colspan="2">Standard Fine-tuning</td><td colspan="2">ProMoT (Ours)</td><td colspan="2">ProMoT + 1-shot (Ours)</td></tr><tr><td>Fine-tuning</td><td>WMT14 En-Fr</td><td colspan="2">1.98</td><td colspan="2">41.80</td><td colspan="2">41.30</td><td colspan="2">41.19</td></tr><tr><td rowspan="10">Evaluation</td><td></td><td>1-shot</td><td>4-shots</td><td>1-shot</td><td>4-shots</td><td>1-shot</td><td>4-shots</td><td>1-shot</td><td>4-shots</td></tr><tr><td>Norm. Avg.</td><td>17.52</td><td>18.75</td><td>9.15(-8.37)</td><td>11.67(-7.07)</td><td>18.87(+1.35)</td><td>20.64(+1.89)</td><td>19.91(+2.39)</td><td>21.99(+3.24)</td></tr><tr><td>CB</td><td>46.43</td><td>51.79</td><td>16.07</td><td>32.14</td><td>41.07</td><td>57.14</td><td>41.07</td><td>53.57</td></tr><tr><td>WiC</td><td>49.69</td><td>49.69</td><td>50.63</td><td>49.06</td><td>50.16</td><td>50.31</td><td>49.84</td><td>50.63</td></tr><tr><td>triviaQA</td><td>17.58</td><td>19.02</td><td>3.20</td><td>3.15</td><td>13.63</td><td>15.20</td><td>16.93</td><td>18.19</td></tr><tr><td>web_questions</td><td>9.70</td><td>13.04</td><td>0.89</td><td>6.15</td><td>9.40</td><td>7.92</td><td>10.14</td><td>12.01</td></tr><tr><td>WMT16_ende</td><td>3.97</td><td>8.83</td><td>0.81</td><td>0.18</td><td>15.52</td><td>15.55</td><td>16.14</td><td>15.63</td></tr><tr><td>WMT16_enro</td><td>1.82</td><td>3.92</td><td>1.53</td><td>0.42</td><td>18.54</td><td>17.80</td><td>17.57</td><td>16.81</td></tr><tr><td>XSum</td><td>6.41</td><td>2.35</td><td>0.05</td><td>1.86</td><td>1.49</td><td>0.65</td><td>3.41</td><td>4.36</td></tr><tr><td>WikiLingua/en</td><td>4.59</td><td>1.33</td><td>0.03</td><td>0.43</td><td>1.14</td><td>0.52</td><td>4.22</td><td>4.73</td></tr></table>

Table 4: Performances of a mT5 XXL model finetuned on WMT14 En-Fr and evaluated on 8 few-shot tasks to verify the generalization ability. BLEU on the fine-tuned task is in the first row. We compare the Norm. Avg. (normalized average performance) with pretrained model and report the relative difference, where red denotes decreased performance and blue denotes increased performance. WMT16 En-De and En-Ro are translation tasks with different language pairs from WMT14 En-Fr.

From both tables, we first observe that the model's in-context learning performance drops significantly after standard fine-tuning. In particular, the few-shot learning performances drop to near zero for 6 over 8 tasks in Table 3, with the only exceptions being CB and WiC where they share the same format as the RTE fine-tuning task.

On the contrary, ProMoT reduces the loss of the in-context learning performance on unseen few-shot evaluation tasks, and even boosts some evaluation tasks that are semantically related to the fine-tuning task but with totally different task formats, resulting in an increasing in-context learning performance on average. In Table 3, ProMoT on the binary NLI dataset dataset consistently improves few-shot performances on two summarization tasks beyond the pretrained model. In Table 4, ProMoT training on English-French translation substantially improves few-shot performance on other language translation pairs such as English to German and Romanian. This cross-task generalization across different task formats are infeasible with previous fine-tuning techniques. Text examples from standard fine-tuning and ProMoT can be found in Appendix C.9. The improvement with less specialization and more generalization can be further boosted when we combine ProMoT with 1-shot prompt to incorporate in-context learning objective during fine-tuning.

It is however not surprising that even ProMoT cannot completely eliminate specialization and may still negatively influence some unseen in-context learning tasks compared to the pretrained model, depending on the characteristics of the fine-tuning task. In the next section, we show that a multi-task setup further improves the already strong generalization of ProMoT.

# 5.4 MORE GENERALIZATION WITH MULTITASK TRAINING

Multi-task training is commonly used to improve model's generalization ability (Wei et al., 2021b; Chung et al., 2022). As a general fine-tuning framework, ProMoT can be combined with multi-tasking and achieves better generalization compared to standard multi-task fine-tuning.

We apply multi-task ProMoT training on mixed RTE and WMT14 En-Fr translation dataset. At the prompt-tuning stage, we train a soft prompt for each task. At the fine-tuning stage, we mix different tasks and prepend the corresponding soft task prompt to each training example. We keep other configurations the same as Section 5.3 and report the results in Table 5. We compare multi-task ProMoT with standard multi-task fine-tuning. The results show that Multi-task ProMoT significantly outperforms standard multi-task fine-tuning on enhancing generalization with larger improvement on average on unseen 1-shot evaluation tasks. Similar to the single task setting, adding 1-shot prompt before each training input in the fine-tuning stage further boosts the performance of both multi-task fine-tuning and multi-task ProMoT.

<table><tr><td></td><td></td><td>Pretrained</td><td>multi-task FT</td><td>Multi ProMoT</td><td>multi-task FT + 1-shot</td><td>Multi-ProMoT + 1-shot</td></tr><tr><td rowspan="2">Multi-task Fine-tuning</td><td>RTE</td><td>47.65</td><td>90.25</td><td>91.34</td><td>91.70</td><td>93.14</td></tr><tr><td>WMT14 En-Fr</td><td>1.982</td><td>41.34</td><td>40.73</td><td>40.87</td><td>40.55</td></tr><tr><td rowspan="9">Evaluation</td><td>Norm. Avg.</td><td>17.52</td><td>20.06(+2.54)</td><td>25.88(+8.35)</td><td>22.62(+5.10)</td><td>26.17(+8.65)</td></tr><tr><td>CB</td><td>46.43</td><td>80.36</td><td>83.93</td><td>87.50</td><td>85.71</td></tr><tr><td>WiC</td><td>49.69</td><td>51.10</td><td>51.41</td><td>53.29</td><td>52.04</td></tr><tr><td>TriviaQA</td><td>17.58</td><td>15.76</td><td>16.99</td><td>16.53</td><td>17.18</td></tr><tr><td>Web_questions</td><td>9.70</td><td>9.70</td><td>10.04</td><td>9.40</td><td>10.38</td></tr><tr><td>WMT16 En-De</td><td>3.97</td><td>0.88</td><td>18.83</td><td>2.50</td><td>17.57</td></tr><tr><td>WMT16 En-Ro</td><td>1.82</td><td>1.52</td><td>18.41</td><td>5.62</td><td>18.57</td></tr><tr><td>XSum</td><td>6.41</td><td>0.44</td><td>4.50</td><td>1.82</td><td>4.32</td></tr><tr><td>WikiLingua/en</td><td>4.59</td><td>0.72</td><td>2.89</td><td>4.33</td><td>3.56</td></tr></table>

Table 5: Comparison of multi-task training on a mixed dataset of RTE and WMT14 En-Fr. We compare the evaluation results of pretrained mT5 model, standard multi-task fine-tuning (FT) and multitask (Multi) ProMoT training. We compare the Norm. Avg. (normalized average performance) with pretrained model and report the relative difference, where blue denotes increased performance.

<table><tr><td></td><td></td><td>Joint Fine-tuning</td><td>Fine-tuning + 1-shot</td><td>Fine-tuning with random prompt</td><td>ProMoT + 1-shot (Ours)</td></tr><tr><td>Fine-tuning</td><td>RTE</td><td>90.97</td><td>90.97</td><td>92.06</td><td>93.86</td></tr><tr><td rowspan="8">Evaluation</td><td>CB</td><td>83.93</td><td>78.57</td><td>83.93</td><td>83.93</td></tr><tr><td>WiC</td><td>50.47</td><td>51.41</td><td>51.72</td><td>51.25</td></tr><tr><td>TriviaQA</td><td>0.75</td><td>0.03</td><td>0.83</td><td>17.82</td></tr><tr><td>web_questions</td><td>0.64</td><td>0.00</td><td>0.30</td><td>10.14</td></tr><tr><td>WMT16_ende</td><td>0.00</td><td>0.00</td><td>0.00</td><td>2.26</td></tr><tr><td>WMT16_enro</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.87</td></tr><tr><td>XSum</td><td>0.00</td><td>0.00</td><td>0.00</td><td>6.94</td></tr><tr><td>WikiLingua/en</td><td>0.00</td><td>0.00</td><td>0.00</td><td>4.87</td></tr></table>

Table 6: The ablation study results. Joint fine-tuning: fine-tuning the soft prompt and the main model together. Fine-tuning + 1-shot: standard fine-tuning with a 1-shot natural language prompt attached to every input sequence. Fine-tuning with random prompt: fine-tuning with a fixed soft prompt randomly initialized with uniform distribution. ProMoT + 1-shot: ProMoT is applied with an attached 1-shot natural language prompt before each training input.

# 5.5 ABLATION STUDY

We conduct several ablation studies in Table 6. First, instead of fine-tuning in a two-stage process, we consider “jointly fine-tuning” both the soft prompt and the model parameters in one stage. As shown in Table 6, this method still results in specialization and severe loss of in-context learning abilities. Thus the benefit of ProMoT comes from its two-stage nature instead of merely adding more learnable parameters (soft prompt). In addition, fine-tuning the models with a fixed random soft prompt does not help - as it does not help to remove format specialization. Another important baseline is to fine-tune the model with natural language prompts in place instead of soft prompts, which also capture the format to some extent. In a 1-shot scenario, this approach is still far worse compared to ProMoT, showing that learned soft prompts work better than natural language prompts in reducing format specialization in fine-tuning.

# 6 CONCLUSIONS AND LIMITATIONS

In this paper, we identify format specialization as one important cause of the loss of general in-context learning abilities during LLM fine-tuning, which tends to happen at the beginning of fine-tuning. We are motivated to develop ProMoT, a simple yet effective two-stage fine-tuning framework that utilizes soft trainable prompts to absorb task-specific formats before model fine-tuning. Experiments on a diverse set of NLP tasks show that ProMoT reduces format specialization and results in surprising generalization across very different tasks, making it a promising method to build general-purpose capabilities into LLMs with small fine-tuning datasets. Although we have shown the effectiveness of ProMoT in our main paper, there is no theoretical guarantee on how much format specialization can be absorbed by the soft prompt during the first stage of ProMoT. Besides, our experiments are done with models smaller than 15B due to limited computation resources. It can be interesting to test ProMoT on larger models.

# ACKNOWLEDGMENTS

We thank the reviewers for their invaluable feedbacks. The work is supported in part by NSF 2008173, 2048280, 2331966, ONR N00014-23-1-2300:P00001, ARL 20230936 and Cisco.

# REFERENCES

Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. The fifth pascal recognizing textual entailment challenge. In TAC, 2009.   
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from question-answer pairs. In Proceedings of the 2013 conference on empirical methods in natural language processing, pp. 1533–1544, 2013.   
Ond rej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Matthias Huck, Antonio Jimeno Yepes, Philipp Koehn, Varvara Logacheva, Christof Monz, Matteo Negri, Aurelie Neveol, Mariana Neves, Martin Popel, Matt Post, Raphael Rubino, Carolina Scarton, Lucia Specia, Marco Turchi, Karin Verspoor, and Marcos Zampieri. Findings of the 2016 conference on machine translation. In Proceedings of the First Conference on Machine Translation, pp. 131–198, Berlin, Germany, August 2016. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/W/W16/W16-2301.   
Ondrej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, Radu Soricut, Lucia Specia, and Ale s Tamchyna. Findings of the 2014 workshop on statistical machine translation. In Proceedings of the Ninth Workshop on Statistical Machine Translation, pp. 12–58, Baltimore, Maryland, USA, June 2014. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/W/W14/W14-3302.   
Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2015.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in NeurIPS, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.   
Stephanie C. Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya Singh, Pierre H. Richemond, Jay McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers, 2022. URL https://arxiv.org/abs/2205.05055.   
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022. URL https://arxiv.org/abs/2204.02311.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.   
Antonia Creswell and Murray Shanahan. Faithful reasoning using large language models, 2022. URL https://arxiv.org/abs/2208.14271.   
Marie-Catherine De Marneff, Mandy Simons, and Judith Tonhauser. The commitmentbank: Investigating projection in naturally occurring discourse. proceedings of Sinn und Bedeutung 23, 2019.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL, pp. 4171–4186, 2019. URL https://aclanthology.org/N19-1423.   
Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. arXiv preprint arXiv:2012.15723, 2020.   
Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig. Towards a unified view of parameter-efficient transfer learning. arXiv preprint arXiv:2110.04366, 2021.   
Yun He, Steven Zheng, Yi Tay, Jai Gupta, Yu Du, Vamsi Aribandi, Zhe Zhao, YaGuang Li, Zhao Chen, Donald Metzler, et al. Hyperprompt: Prompt-based task-conditioning of transformers. In International Conference on Machine Learning, pp. 8678–8690. PMLR, 2022.   
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pp. 2790–2799. PMLR, 2019.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.   
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.   
Faisal Ladhak, Esin Durmus, Claire Cardie, and Kathleen McKeown. WikiLingua: A new benchmark dataset for cross-lingual abstractive summarization. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4034–4048, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.360. URL https://aclanthology.org/2020.findings-emnlp.360.   
Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 3045–3059, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.243. URL https://aclanthology.org/2021.emnlp-main.243.   
Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74–81, 2004.   
Xiao Liu, Kaixuan Ji, Yicheng Fu, Zhengxiao Du, Zhilin Yang, and Jie Tang. P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks. arXiv preprint arXiv:2110.07602, 2021.   
Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. On faithfulness and factuality in abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 1906–1919, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.173. URL https://aclanthology.org/2020.acl-main.173.   
Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.

Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn in context. arXiv preprint arXiv:2110.15943, 2021.   
Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837, 2022.   
Shashi Narayan, Shay B Cohen, and Mirella Lapata. Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. arXiv preprint arXiv:1808.08745, 2018.   
Kimia Noorbakhsh, Modar Sulaiman, Mahdi Sharifi, Kallol Roy, and Pooyan Jamshidi. Pretrained language models are symbolic mathematics solvers too! arXiv preprint arXiv:2110.03501, 2021.   
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp. 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P02-1040.   
Mohammad Taher Pilehvar and José Camacho-Collados. Wic: 10,000 example pairs for evaluating context-sensitive representations. arXiv preprint arXiv:1808.09121, 6, 2018.   
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.   
Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher, 2021. URL https://arxiv.org/abs/2112.11446.   
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR, 21(140):1–67, 2020a. URL http://jmlr.org/papers/v21/20-074.html.   
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140):1–67, 2020b.   
Vinay Venkatesh Ramasesh, Aitor Lewkowycz, and Ethan Dyer. Effect of scale on catastrophic forgetting in neural networks. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=GhVS8\_yPeEa.   
Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, Elton Zhang, Rewon Child, Reza Yazdani Aminabadi, Julie Bernauer, Xia Song, Mohammad Shoeybi, Yuxiong He, Michael Houston, Saurabh Tiwary, and Bryan Catanzaro. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model, 2022. URL https://arxiv.org/abs/2201.11990.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in NeurIPS, 2017. URL https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.   
Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32, 2019.   
Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. CoRR, abs/2109.01652, 2021a. URL https://arxiv.org/abs/2109.01652.   
Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021b.   
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mt5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934, 2020.   
Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. Investigating the catastrophic forgetting in multimodal large language models. arXiv preprint arXiv:2309.10313, 2023.   
Ningyu Zhang, Luoqiu Li, Xiang Chen, Shumin Deng, Zhen Bi, Chuanqi Tan, Fei Huang, and Huajun Chen. Differentiable prompt makes pre-trained language models better few-shot learners. arXiv preprint arXiv:2108.13161, 2021.

# A BROADER IMPACTS

In our work, we propose a method to improve general-purpose language models with fine-tuning datasets. The improved general-purpose language model may be used in malicious applications such as generating disinformation. To mitigate the potential negative impacts, we can add watermark or deploy AI-generated text classifiers before releasing the model.

# B EXPERIMENT DETAILS

# B.1 INPUT TEMPLATE USED IN EXPERIMENTS

In Table 7, we list the natural language input template used in our experiments for each task. The 

<table><tr><td>Task</td><td>Template</td></tr><tr><td>RTE</td><td>[premise] question: [hypothesis] Is it true or false? answer: {True, False}</td></tr><tr><td>CB &amp; SNLI</td><td>[premise] question: [hypothesis] Is it true or false or neither? answer: {True, False, Neither}</td></tr><tr><td>WiC</td><td>[sentence1] [sentence2] question: The word [word] is used in the same way in the two sentences.Is it true or False? answer: {True, False}</td></tr><tr><td>OpenbookQA</td><td>Q: [question] A) [option A] B) [option B] C) [option C] D) [option D] A:</td></tr><tr><td>QA</td><td>Q: [question] A:</td></tr><tr><td>Translation</td><td>Translate [source language] to [target language]: [sentence 1]</td></tr><tr><td>Summarization</td><td>Article: [article] One sentence summary:</td></tr></table>

Table 7: Input template for each task

example shown in Table 1 is from ID 41141109 in XSum dataset.

# B.2 OUTPUT POST-PROCESSING

For each task, we first extract the text after <extra\_id\_0> and before <extra\_id\_1>, then trim the text by locating and remove the text after the second prefix token (Q:, Translate, Article:). For classification tasks including RTE, CB and WiC, we check whether the first output token is True or False.

# B.3 DATASET AND MODELS

We list the statistics of all datasets used in the paper in Table 8. All the datasets and models can be used in research context.

# B.4 HYPER-PARAMETERS

For all mT5 models, we fine-tune with learning rate 0.001, drop rate 0.1 and label smoothing 0.1, following the default settings for T5 models (Raffel et al., 2020b). For all prompt tuning experiments, we use learning rate 0.2 and prompt length 100. For all tasks except summarization tasks, we choose the model input sequence length larger than the input length in datasets. For summarization, we

<table><tr><td>Dataset</td><td>Version</td><td>Training</td><td>Validation</td><td>Test</td></tr><tr><td>RTE</td><td>v102</td><td>2,490</td><td>277</td><td>3,000</td></tr><tr><td>CB</td><td>v102</td><td>250</td><td>56</td><td>250</td></tr><tr><td>WiC</td><td>v102</td><td>5,428</td><td>638</td><td>1,400</td></tr><tr><td>WMT14 En-Fr</td><td>v003</td><td>15,786,979</td><td>3,000</td><td>3,003</td></tr><tr><td>WMT16 En-De</td><td>v003</td><td>4,548,885</td><td>2,169</td><td>2,999</td></tr><tr><td>WMT16 En-Ro</td><td>v003</td><td>610,320</td><td>1,999</td><td>1,999</td></tr><tr><td>TriviaQA</td><td>rc.nocontext:1.1.0</td><td>138,384</td><td>18,669</td><td>17,210</td></tr><tr><td>Web Questions</td><td>1.0.0</td><td>3,778</td><td>-</td><td>2,032</td></tr><tr><td>XSum</td><td>1.1.0</td><td>203,577</td><td>11,305</td><td>11,301</td></tr><tr><td>WikiLingua/en</td><td>gem/wiki_lingua_english_en</td><td>99,020</td><td>13,823</td><td>28,614</td></tr></table>

Table 8: Version number, sizes of training, validation, and testing splits for each dataset used.

<table><tr><td>Fine-tuning Datasets</td><td>Prompt Tuning + 1-shot</td><td>ProMoT + 1-shot</td></tr><tr><td>RTE</td><td>92.78</td><td>93.86</td></tr><tr><td>WMT14 En-Fr</td><td>39.41</td><td>41.19</td></tr></table>

Table 9: Performances of prompt tuning + 1-shot and ProMoT + 1-shot on fine-tuning tasks.

cut each input to 1024 tokens. We use Adafactor optimizer and batch size 64 without data-packing across all experiments. In inference, we use beam search to decode the outputs with width 4. More experimental settings are provided in the appendix. For ProMoT tuning, at stage 1 we run prompt tuning for 5000 steps and save a checkpoint every 1000 steps, then select the prompt checkpoint with the best performance on target task. At stage 2, we freeze the trained prompt and fine-tune the model for 1000 steps, checkpointing every 100 steps. We pick the model checkpoint with highest performance on the fine-tuned task as our final checkpoint. For comparison, we run prompt tuning and standard fine-tuning for 5000 and 1000 training steps respectively and report the performance of the best checkpoint. We explore fine-tuning with more steps in Appendix C.3.

In ablation study in Section 5.5, we include an experiment to jointly fine-tune soft prompt and pretrained model. In this experiment, we finetune the model and prompt for 1000 steps with the same learning rate 0.001, following the setting in (He et al., 2022).

# B.5 HARDWARE AND IMPLEMENTATION

All the experiments are implemented based on the original prompt tuning $^{2}$ and T5x code base $^{3}$ . All experiments are run on a cluster of 64 parallel TPUs. Time cost for different experiments varies, however, all training experiments can be finished within 1 day.

# C ADDITIONAL EXPERIMENT RESULTS

# C.1 ADDITIONAL RESULTS ON SINGLE TASK FINE-TUNING

As complementary results of Table 3 and 4, we list and compare the performance of prompt tuning + 1-shot in Table 9. We also provide experiments on SNLI and OpenbookQA datasets in Table 10. Without fine-tuning, pretrained mT5 failed to output “A”, “B”, “C”, “D” for multi-choice QA in 0-shot openbookQA dataset, which results in a zero accuracy. We can see that the additional experiments are consistent with our main experiments that ProMoT can achieve similar supervised performance on fine-tuning tasks with less forgetting and even better performance on general in-context learning tasks.

<table><tr><td>Datasets</td><td>Pretrained</td><td>Prompt-tuning</td><td>Standard Fine-tuning on SNLI</td><td>ProMoT (Ours) on SNLI</td><td>Standard Fine-tuning on OpenbookQA</td><td>ProMoT (Ours) on OpenbookQA</td></tr><tr><td>SNLI</td><td>1.32</td><td>88.53</td><td>88.91</td><td>89.62</td><td>-</td><td>-</td></tr><tr><td>OpenbookQA</td><td>0.00</td><td>73.60</td><td>-</td><td>-</td><td>77.2</td><td>81.6</td></tr><tr><td>Norm. Average</td><td>17.52</td><td>17.52</td><td>16.20(-1.32)</td><td>19.83(+2.31)</td><td>0.00(-17.52)</td><td>17.40(-0.12)</td></tr><tr><td>CB</td><td>46.43</td><td>46.43</td><td>69.64</td><td>62.5</td><td>0.00</td><td>41.07</td></tr><tr><td>WiC</td><td>49.69</td><td>49.69</td><td>53.29</td><td>51.25</td><td>0.00</td><td>50.0</td></tr><tr><td>triviaQA</td><td>17.58</td><td>17.58</td><td>4.54</td><td>20.56</td><td>0.05</td><td>21.10</td></tr><tr><td>web_questions</td><td>9.70</td><td>9.70</td><td>2.12</td><td>9.94</td><td>0.00</td><td>10.53</td></tr><tr><td>WMT16_ende</td><td>3.97</td><td>3.97</td><td>0.00</td><td>2.48</td><td>0.00</td><td>2.80</td></tr><tr><td>WMT16_enro</td><td>1.82</td><td>1.82</td><td>0.00</td><td>0.90</td><td>0.00</td><td>1.04</td></tr><tr><td>XSum</td><td>6.41</td><td>6.41</td><td>0.00</td><td>6.58</td><td>0.00</td><td>7.48</td></tr><tr><td>WikiLingua/en</td><td>4.59</td><td>4.59</td><td>0.00</td><td>4.39</td><td>0.00</td><td>5.20</td></tr></table>

Table 10: Performance of a mT5 XXL model finetuned on SNLI and OpenbookQA and evaluated on 8 1-shot tasks. The accuracy on fine-tuned tasks are in the first two rows. Prompt-tuning doesn't modify pretrained model parameters and has the same in-context performance as pretrained model.

# C.2 4-SHOT EVALUATION RESULTS OF MULTI-TASK TRAINING

As an additional result to Table 5, in Table 11 we provide the comparison between the pretrained model, multi-task standard fine-tuning and multi-task ProMoT.

<table><tr><td></td><td></td><td>Pretrained</td><td>multi-task FT</td><td>Multi ProMoT</td></tr><tr><td rowspan="2">Multi-task Fine-tuning</td><td>RTE</td><td>47.65</td><td>90.25</td><td>91.34</td></tr><tr><td>WMT14 En-Fr</td><td>1.982</td><td>41.34</td><td>40.73</td></tr><tr><td rowspan="9">Evaluation</td><td>Norm. Avg.</td><td>18.75</td><td>20.81(+2.06)</td><td>26.31(+7.56)</td></tr><tr><td>CB</td><td>51.79</td><td>82.14</td><td>78.57</td></tr><tr><td>WiC</td><td>49.69</td><td>52.82</td><td>52.50</td></tr><tr><td>TriviaQA</td><td>19.02</td><td>17.75</td><td>22.26</td></tr><tr><td>Web_questions</td><td>13.04</td><td>12.25</td><td>12.50</td></tr><tr><td>WMT16 En-De</td><td>8.83</td><td>0.24</td><td>18.84</td></tr><tr><td>WMT16 En-Ro</td><td>3.92</td><td>0.54</td><td>18.81</td></tr><tr><td>XSum</td><td>2.35</td><td>0.34</td><td>5.04</td></tr><tr><td>WikiLingua/en</td><td>1.33</td><td>0.39</td><td>1.92</td></tr></table>

Table 11: Comparison of multi-task training on a mixed dataset of RTE and WMT14 En-Fr. We compare the 4-shot evaluation results of pretrained mT5 model, standard multi-task fine-tuning (FT) and multitask (Multi) ProMoT training. We compare the Norm. Avg. (normalized average performance) with pretrained model and report the relative difference, where blue denotes increased performance.

# C.3 TRAINING MORE STEPS: TRADE-OFF BETWEEN FINE-TUNING TARGET TASK AND IN-CONTEXT LEARNING ABILITIES

In Section 5.3, we report the results of the best checkpoints within 1000 steps of fine-tuning. With a longer training period, we can see a more clear trade-off between the performance on fine-tuning target task and the performance on in-context learning abilities. Here we show the long-term trade-off between fine-tuning target task and in-context learning evaluation tasks by scattering the performance of different checkpoints within 20000 steps fine-tuning. In Figure 5, and 6, we plot the trade-off on classification and translation tasks, respectively.

![](images/bebfb4e2808e267a17ec04c2478d79688cd297da4f4ac6f3bf1ee202d734e472.jpg)

<details>
<summary>scatter</summary>

| BLEU Score | Traditional Fine-Tuning | ProMoT |
| ---------- | ------------------------ | ------ |
| 41         | -                        | 43     |
| 42         | 36                       | 38     |
| 43         | 35                       | 39     |
| 44         | 10                       | 12     |
</details>

![](images/aa2084da62f0b69e78c5110134972142cf35893237d9cd2e47e633a9a1864a4e.jpg)

<details>
<summary>scatter</summary>

| Method               | BLEU Score | Avg. BLEU Score |
| -------------------- | ---------- | --------------- |
| Traditional Fine-Tuning | 41.0       | 13.0            |
| Traditional Fine-Tuning | 41.5       | 1.0             |
| Traditional Fine-Tuning | 42.0       | 6.0             |
| Traditional Fine-Tuning | 42.5       | 2.5             |
| Traditional Fine-Tuning | 43.0       | 3.5             |
| Traditional Fine-Tuning | 43.5       | 2.0             |
| Traditional Fine-Tuning | 44.0       | 0.5             |
| Traditional Fine-Tuning | 44.5       | 0.5             |
| ProMoT               | 41.0       | 13.0            |
| ProMoT               | 42.0       | 6.0             |
| ProMoT               | 42.5       | 2.5             |
| ProMoT               | 43.0       | 3.5             |
| ProMoT               | 43.5       | 2.5             |
| ProMoT               | 44.0       | 2.0             |
| ProMoT               | 44.5       | 0.5             |
</details>

Figure 5: Trade-off between BLEU score of En-Fr (horizontal axis) and average accuracy on classification tasks (vertical axis) when fine-tuning the model on En-Fr translation.   
Figure 6: Trade-off between BLEU score of En-Fr (horizontal axis) and average BLEU score on other language pairs (vertical axis) when fine-tuning the model on En-Fr translation.

As we can see from the figures, datapoints for ProMoT is higher than standard fine-tuning on the figures, which implies that with the same performance on fine-tuning target task, forgetting is alleviated with ProMoT fine-tuning.

# C.4 ADDITIONAL EXPERIMENTS ON T5 XXL

To show the performance of our method on an English-based pretrained model, we did an additional experiment on T5 XXL with fine-tuning target task RTE. The result is shown in Table 12. The results are consistent with our main experiments on the mT5 XXL model.

<table><tr><td>Datasets</td><td>Pretrained</td><td>Prompt-tuning</td><td>Standard Fine-tuning</td><td>ProMoT (Ours)</td></tr><tr><td>RTE</td><td>-</td><td>91.7</td><td>93.5</td><td>93.14</td></tr><tr><td>Norm. Average</td><td>19.75</td><td>19.75</td><td>14.07(-5.68)</td><td>22.23(+2.49)</td></tr><tr><td>CB</td><td>55.36</td><td>55.36</td><td>62.50</td><td>73.21</td></tr><tr><td>WiC</td><td>49.84</td><td>49.84</td><td>50.00</td><td>50.78</td></tr><tr><td>triviaQA</td><td>34.15</td><td>34.15</td><td>0.02</td><td>33.86</td></tr><tr><td>web_questions</td><td>16.04</td><td>16.04</td><td>0.00</td><td>15.95</td></tr><tr><td>WMT16_ende</td><td>0.13</td><td>0.13</td><td>0.00</td><td>0.02</td></tr><tr><td>WMT16_enro</td><td>0.06</td><td>0.06</td><td>0.00</td><td>0.01</td></tr><tr><td>XSum</td><td>1.26</td><td>1.26</td><td>0.00</td><td>1.79</td></tr><tr><td>WikiLingua/en</td><td>1.12</td><td>1.12</td><td>0</td><td>2.25</td></tr></table>

Table 12: Performance of a T5.1.1 XXL model finetuned on RTE and evaluated on 8 1-shot tasks. The accuracy on fine-tuned task (RTE) is in the first row. Prompt-tuning doesn't modify pretrained model parameters and has the same in-context performance as pretrained model.

# C.5 ADDITIONAL EXPERIMENTS ON MT5 XL

To show the performance of our method on an smaller-size pretrained model, we did an additional experiment on mT5 XL with fine-tuning target task WMT14 En-Fr. The result is shown in Table 13. The results are consistent with our main experiments on the mT5 XXL model.

<table><tr><td>Datasets</td><td>Pretrained</td><td>Prompt-tuning</td><td>Standard Fine-tuning</td><td>ProMoT (Ours)</td></tr><tr><td>WMT14 En-Fr</td><td>-</td><td>32.47</td><td>35.84</td><td>36.46</td></tr><tr><td>Norm. Average</td><td>13.59</td><td>13.59</td><td>8.61(-4.98)</td><td>14.40(+0.81)</td></tr><tr><td>CB</td><td>26.79</td><td>26.79</td><td>21.43</td><td>28.57</td></tr><tr><td>WiC</td><td>50.0</td><td>50.0</td><td>44.20</td><td>51.10</td></tr><tr><td>triviaQA</td><td>12.13</td><td>12.13</td><td>0.83</td><td>8.81</td></tr><tr><td>web_questions</td><td>6.59</td><td>6.59</td><td>0.44</td><td>5.31</td></tr><tr><td>WMT16_ende</td><td>2.56</td><td>2.56</td><td>0.63</td><td>7.69</td></tr><tr><td>WMT16_enro</td><td>1.52</td><td>1.52</td><td>1.20</td><td>10.40</td></tr><tr><td>XSum</td><td>4.26</td><td>4.26</td><td>0.05</td><td>1.37</td></tr><tr><td>WikiLingua/en</td><td>4.88</td><td>4.88</td><td>0.06</td><td>1.94</td></tr></table>

Table 13: Performance of a mT5 XL model finetuned on WMT14 En-Fr and evaluated on 8 1-shot tasks. The BLEU score on fine-tuned task (WMT14 En-Fr) is in the first row. Prompt-tuning doesn't modify pretrained model parameters and has the same in-context performance as pretrained model.

# C.6 ADDITIONAL EXPERIMENTS ON PALM 8B

To show the performance of our method on decoder-only models, we did an additional experiment on PaLM 8b model with fine-tuning target task WMT14 En-Fr. We use prompt length 50 and learning rate 0.3 in prompt-tuning and default fine-tuning hyperparameters in fine-tuning. The result is shown in Table 14. The results are consistent with our main experiments on mT5, where ProMoT can achieve similar supervised performance on fine-tuning tasks with less forgetting on general in-context learning tasks.

# C.7 USING LORA IN THE FIRST STAGE

As we have discussed in Section 4, conceptually we can use any parameter-efficient method at the first ProMoT fine-tuning stage to absorb the task format information. Here we did experiments to

<table><tr><td>Datasets</td><td>Pretrained</td><td>Prompt-tuning</td><td>Standard Fine-tuning</td><td>ProMoT (Ours)</td></tr><tr><td>WMT14 En-Fr</td><td>-</td><td>13.62</td><td>33.04</td><td>32.02</td></tr><tr><td>Norm. Average</td><td>26.09</td><td>26.09</td><td>17.80(-8.29)</td><td>22.37(-3.72)</td></tr><tr><td>CB</td><td>46.43</td><td>46.43</td><td>32.14</td><td>33.93</td></tr><tr><td>WiC</td><td>49.69</td><td>49.69</td><td>49.06</td><td>49.69</td></tr><tr><td>triviaQA</td><td>44.69</td><td>44.69</td><td>37.09</td><td>42.11</td></tr><tr><td>web_questions</td><td>13.02</td><td>13.02</td><td>11.91</td><td>13.01</td></tr><tr><td>WMT16_ende</td><td>23.85</td><td>23.85</td><td>3.77</td><td>19.33</td></tr><tr><td>WMT16_enro</td><td>19.89</td><td>19.89</td><td>4.02</td><td>13.18</td></tr><tr><td>XSum</td><td>5.57</td><td>5.57</td><td>2.29</td><td>2.9</td></tr><tr><td>WikiLingua/en</td><td>5.59</td><td>5.59</td><td>3.14</td><td>4.77</td></tr></table>

Table 14: Performance of a PaLM 8b model finetuned on WMT14 En-Fr and evaluated on 8 1-shot tasks. The BLEU score on fine-tuned task (WMT14 En-Fr) is in the first row. Prompt-tuning doesn't modify pretrained model parameters and has the same in-context performance as pretrained model.

![](images/76265f9b7482ab3c3a4dc8c83c265680d78b871cd33c2250b6d21ffc75eb1333.jpg)

<details>
<summary>bar</summary>

| Layers       | cos(<g₀, g_format,₀>) | cos(<g₁₀₀, g_format,₁₀₀>) | cos(<g₂₀₀, g_format,₂₀₀>) | cos(<g₃₀₀, g_format,₃₀₀>) | cos(<g₄₀₀, g_format,₄₀₀>) |
| ------------ | ---------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| MLP kernel   | 0.98                   | 0.85                       | 0.87                       | 0.76                       | 0.10                       |
| Query        | 0.96                   | 0.67                       | 0.58                       | 0.56                       | 0.08                       |
| Key          | 0.99                   | 0.82                       | 0.80                       | 0.73                       | 0.05                       |
| Value        | 0.99                   | 0.95                       | 0.96                       | 0.74                       | 0.20                       |
</details>

Figure 7: Cosine similarity between the full gradient g and the format gradient $g_{format}$ on different parts of the last decoder layer. We collect and show the cosine value for gradients on MLP kernel, Query, Key and Value on the attention module.

compare LoRA and prompt-tuning (used in our ProMoT main experiments) in the first fine-tuning stage. We report the results in Table 15. As we can see from the table, ProMoT with prompt-tuning is significantly better than ProMoT with LoRA, in both supervised fine-tuning task and unseen 1-shot evaluation tasks. This might partially due to better alignment of soft prompt between format description in natural language corpus.

<table><tr><td>Datasets</td><td>Pretrained</td><td>Standard Fine-tuning</td><td>ProMoT with LoRA (r=2)</td><td>ProMoT with LoRA (r=4)</td><td>ProMoT</td></tr><tr><td>WMT14 En-Fr</td><td>-</td><td>41.80</td><td>39.09</td><td>39.97</td><td>41.19</td></tr><tr><td>CB</td><td>46.43</td><td>16.07</td><td>26.78</td><td>23.21</td><td>41.07</td></tr><tr><td>WiC</td><td>49.69</td><td>50.63</td><td>53.13</td><td>53.25</td><td>50.16</td></tr><tr><td>triviaQA</td><td>17.58</td><td>0.03</td><td>4.98</td><td>5.19</td><td>13.63</td></tr><tr><td>web_questions</td><td>9.70</td><td>0.05</td><td>3.30</td><td>3.84</td><td>9.40</td></tr><tr><td>WMT16_ende</td><td>3.97</td><td>0.67</td><td>1.23</td><td>1.87</td><td>15.52</td></tr><tr><td>WMT16_enro</td><td>1.82</td><td>0.91</td><td>2.06</td><td>2.89</td><td>18.54</td></tr><tr><td>XSum</td><td>6.41</td><td>0.030</td><td>0.17</td><td>0.35</td><td>1.49</td></tr><tr><td>WikiLingua/en</td><td>4.59</td><td>0.05</td><td>0.64</td><td>0.57</td><td>1.14</td></tr></table>

Table 15: Performance of a mT5 XXL model finetuned on WMT14 En-Fr and evaluated on 8 1-shot tasks. In this experiment we use LoRA at the first ProMoT stage instead of prompt-tuning. $r$ is the rank of LoRA's low-rank update matrices. The BLEU score on fine-tuned task (WMT14 En-Fr) is in the first row. Prompt-tuning doesn't modify pretrained model parameters and has the same in-context performance as pretrained model.

# C.8 PLOTTING MORE STEPS FOR FIGURE 3

To further strengthen our conclusion in Figure 3, here we plot the gradient alignment from step 0 to step 400. As we can see from the figure, gradient alignment drops significantly after 300 steps which is matched with Figure 2 where the true and false ratio increases before 300 steps and then remains stable.

# C.9 QUALITATIVE RESULTS ON FINE-TUNING WMT14 EN-FR TASK

In Table 1 we show an example from fine-tuning task RTE. Here we show examples for fine-tuning task WMT14 En-Fr translation on different unseen few-shot tasks. We compare the outputs from ground-truth targets, pretrained mT5, fine-tuned mT5 on WMT14 En-Fr and ProMoT mT5 on WMT14 En-Fr. The outputs are generated with a 1-shot example. As we can see from the examples, standard fine-tuning on WMT14 En-Fr will 1) make the model overfit its format and tend to output French; and 2) model tends to repeat its input which is similar to translation task. ProMoT alleviates this specialization on fine-tuning task and has better generalization.

# - WMT16 En-De

- Target: Danach war der Mann, der sich nach Angaben seines Anwalts mittlerweile wieder auf freiem Fußbefindet, in eine größere Zelle verlegt worden.   
- Pretrained: Danach wurde der Mann in eine größere.   
- Fine-tune: L'homme, qui, selon une déclaration de son avocat, a depuis été libéré, a ensuite été transféré dans une cellule plus grande.   
- ProMoT: Danach wurde der Mann, der mittlerweile freigelassen wurde, in eine größere Zelle verlegt.

# - WebQuestions

- Target: Milwaukee   
– Pretrained: Milwaukee, Wisconsin   
- Fine-tune: Where is harley davidson corporate headquarters? A: Milwaukee, Wisconsin Q: what movies has scarlett johansson in? A: Girl with a Pearl Earring Q: where is harley davidson corporate headquarters? A: Milwaukee, Wisconsin Q: where is harley davidson corporate headquarters? ...   
- ProMoT: Milwaukee, Wisconsin

# - WikiLingua/en

- Target: Ensure that you have never activated a free trial of Amazon Prime on your account in the past. Repeat the steps above to locate products that are fulfilled by Amazon. Click the button that says, ...   
- Pretrained: Click “Start my Free Trial.” Sign up with Amazon Prime. Submit your order.   
- Fine-tune: Si vous avez besoin d'une aide supplémentaire, n'hésitez pas à communiquer avec l'équipe d'Amazon Prime. Vous pouvez vous inscrire à l'Amazon Prime et vous inscrire à l'Amazon Prime. Vous pouvez vous inscrire ...   
- ProMoT: Click “Start my Free Trial.” Sign up with Amazon Prime. Enter your credit card details or use one of your saved payment methods. Submit your order. If you do not return, you will be charged \$99 for a year membership to Amazon Prime at the end of your trial period