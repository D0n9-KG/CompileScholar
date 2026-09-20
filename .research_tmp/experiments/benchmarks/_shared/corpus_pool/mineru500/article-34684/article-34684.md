# Can Watermarking Large Language Models Prevent Copyrighted Text Generation and Hide Training Data?

Michael-Andrei Panaitescu-Liess $^{§}$ , Zora Che $^{§}$ , Bang An $^{§}$ , Yuancheng Xu $^{§}$ , Pankayaraj Pathmanathan $^{§}$ , Souradip Chakraborty $^{§}$ , Sicheng Zhu $^{§}$ , Tom Goldstein $^{§}$ , Furong Huang $^{§,¶}$

$^{§}$ University of Maryland, College Park ¶ Capital One {mpanaite, zche, bangan, ycxu, pan, schakra3, sczhu, tomg, furongh}@umd.edu

# Abstract

Large Language Models (LLMs) have demonstrated impressive capabilities in generating diverse and contextually rich text. However, concerns regarding copyright infringement arise as LLMs may inadvertently produce copyrighted material. In this paper, we first investigate the effectiveness of watermarking LLMs as a deterrent against the generation of copyrighted texts. Through theoretical analysis and empirical evaluation, we demonstrate that incorporating watermarks into LLMs significantly reduces the likelihood of generating copyrighted content, thereby addressing a critical concern in the deployment of LLMs. However, we also find that watermarking can have unintended consequences on Membership Inference Attacks (MIAs), which aim to discern whether a sample was part of the pretraining dataset and may be used to detect copyright violations. Surprisingly, we find that watermarking adversely affects the success rate of MIAs, complicating the task of detecting copyrighted text in the pretraining dataset. These results reveal the complex interplay between different regulatory measures, which may impact each other in unforeseen ways. Finally, we propose an adaptive technique to improve the success rate of a recent MIA under watermarking. Our findings underscore the importance of developing adaptive methods to study critical problems in LLMs with potential legal implications.

# Introduction

In recent years, Large Language Models (LLMs) have pushed the frontiers of natural language processing by facilitating sophisticated tasks like text generation, translation, and summarization. With their impressive performance, LLMs are increasingly integrated into various applications, including virtual assistants, chatbots, content generation, and education. However, the widespread usage of LLMs brings forth serious concerns regarding potential copyright infringements. Addressing these challenges is critical for the ethical and legal deployment of LLMs.

Copyright infringement involves unauthorized usage of copyrighted content, which violates the intellectual property rights of copyright owners, potentially undermining content creators' ability to fund their work, and affecting the diversity of creative outputs in society. Additionally, violators can face legal consequences, including lawsuits and financial penalties. For LLMs, copyright infringement can occur through (1) generation of copyrighted content during deployment and (2) illegal usage of copyrighted works during training. Ensuring the absence of copyrighted content in the vast training datasets of LLMs is challenging. Moreover, legal debates around generative AI copyright infringement vary by region, complicating compliance further.

Current lawsuits against AI companies for unauthorized use of copyrighted content (e.g., Andersen v. Stability AI Ltd, NYT v. OpenAI) highlight the urgent need for methods to address these challenges. In this paper, we focus on studying the effects of watermarking LLMs on two critical issues: (1) preventing the generation of copyrighted content, and (2) detecting copyrighted content in training data. We show that watermarking can significantly impact both the generation of copyrighted text and the detection of copyrighted content in training data.

Firstly, we observe that current LLM output watermarking techniques can significantly reduce the probability of LLMs generating copyrighted content, by tens of orders of magnitude. Our empirical results focus on two recent watermarking methods: UMD (Kirchenbauer et al. 2023) and Unigram-Watermark (Zhao et al. 2023). Both methods split the vocabulary into two sets (green and red) and bias the model towards selecting tokens from the green set by altering the logits distribution, thereby embedding a detectable signal. We provide both empirical and theoretical results to support our findings.

Secondly, we demonstrate that watermarking techniques can decrease the success rate of Membership Inference Attacks (MIAs), which aim to detect whether a piece of copyrighted text was part of the training dataset. Since MIAs exploit the model's output, their performance can suffer under watermarking due to changes in the probability distribution of output tokens. Our comprehensive empirical study, including 5 recent MIAs and 5 LLMs, shows that the AUC of detection methods can be reduced by up to $16\%$ in the presence of watermarks.

![](images/757ab95de28f9c174677ebcbe6c9e09c41258c36f440b5067adf70ad44645c76.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Robot"] -->|Generate| B["Document"]
    B --> C["Document"]
    C --> D["Document"]
    D --> E["Document"]
    E --> F["Document"]
    F --> G["Document"]
    G --> H["Document"]
    H --> I["Document"]
    I --> J["Watermark"]
    J --> K["Generate"]
    K --> L["Document"]
    L --> M["Document"]
    M --> N["Document"]
    N --> O["Document"]
    O --> P["Document"]
    P --> Q["10^20x less"]
```
</details>

Figure 1: Illustration of the effect of LLM watermarking on generation of copyrighted content. We observe that a moderately strong watermark ( $\delta = 10$ ) can make it more than $10^{20}$ times less likely for Llama-30B to generate copyrighted content.

enhance the success rate of a recent MIA (Shi et al. 2023) in detecting copyright violations under watermarking. This method applies a correction to the model's output to account for the perturbations introduced by watermarks. By incorporating knowledge about the watermarking scheme, we improve the detection performance for pretraining data, counteracting the obfuscation caused by watermarking. Our contribution underscores the importance of continuously developing adaptive attack methodologies to keep pace with advances in defense mechanisms.

The rest of the paper is organized as follows. In the “Related Work” section, we review prior research on LLM watermarking and copyright. The “Setup and Notations” section formally introduces the problems we study. We then present our first two contributions and introduce the adaptive version of the Min-K% Prob membership inference attack in the following three sections. Finally, we provide concluding remarks in the last section. Additional experiments, theoretical results, and a discussion on the limitations of our work are included in the appendix

# Related Work

Watermarks for LLMs. Language model watermarking techniques embed identifiable markers into output text to detect AI-generated content. Recent strategies incorporate watermarks during the decoding phase of language models (Zhao et al. 2023; Kirchenbauer et al. 2023). Aaronson (2023) develops the Gumbel watermark, which employs traceable pseudo-random sampling for generating subsequent tokens. Kirchenbauer et al. (2023) splits the vocabulary into red and green lists according to preceding tokens, biasing the generation towards green tokens. Zhao et al. (2023) employs a fixed grouping strategy to develop a robust watermark with theoretical guarantees. Liu et al. (2024) proposes to generate watermark logits based on the preceding tokens' semantics rather than their token IDs to boost the robustness. Kuditipudi et al. (2023) and Christ, Gunn, and Zamir (2023) explore watermark methods that do not change the output textual distribution.

Copyright. Copyright protection in the age of AI has gained importance, as discussed by Ren et al. (2024). Vyas, Kakade, and Barak (2023) addresses content protection through near access-freeness (NAF) and developed learning algorithms for generative models to ensure compliance under NAF conditions. Prior works focus on training algorithms to prevent copyrighted text generation (Vyas, Kakade, and Barak 2023; Chu, Song, and Yang 2024), whereas our work emphasizes lightweight, inference-time algorithms. Other works have studied copyright in machine learning from a legal perspective. Hacohen et al. (2024) utilizes a generative model to determine the generic characteristics of works to aid in defining the scope of copyright. Elkin-Koren et al. (2023) demonstrates that copying does not necessarily constitute copyright infringement and argues that existing detection methods may detract from the foundational purposes of copyright law.

Additionally, we include a discussion on memorization and membership inference in the appendix.

# Setup and Notations

# Definitions

Let $D$ be a training dataset, $C$ be all the copyrighted texts, and $C_D$ be all the copyrighted texts that are part of $D$ . We give definitions for the following setups.

Verbatim Memorization of Copyrighted Content. For a fixed $k \in N$ , Carlini et al. (2022b) defines a string s as being memorized by a model if s is extractable with a prompt p of length k using greedy decoding and the concatenation $p \oplus s \in D$ . We adopt a similar definition for verbatim memorization of copyrighted content but employ a continuous metric to measure it. Specifically, we measure verbatim memorization of a text $c \in C$ using the perplexity of the model on the copyrighted text $c_{p}$ when given the prefix p as a prompt (where $c_{p}$ represents the text c after removing its prefix p). Note that for $c_{p} = c_{p}^{(1)} \oplus c_{p}^{(2)} \oplus \cdots \oplus c_{p}^{(n)}$ we compute the perplexity using the following formula $\text{perplexity}(c_{p}|p) = \left(\prod_{i=1}^{n} \mathbb{P}(c_{p}^{(i)}|p \oplus c_{p}^{(0)} \oplus c_{p}^{(1)} \oplus \cdots \oplus c_{p}^{(i-1)})^{-\frac{1}{n}}\right)$ , where $c_{p}^{(0)}$ is the empty string. In our experiments, p is either an empty string or the first 10, 20, or 100 tokens of c. Lower perplexity thereby indicate higher levels of memorization.

MIAs for Copyrighted Training Data Detection. MIAs are privacy attacks aiming to detect whether a sample was part of the training set. We define an MIA for copyrighted data as a binary classifier $A(\cdot)$ , which ideally outputs $A(x) = 1, \forall x \in C_{D}$ and $A(x) = 0, \forall x \in C - C_{D}$ . In practice, $A(\cdot)$ is defined by thresholding a metric (e.g., perplexity), i.e., $A(x) = 1, \forall x$ such that perplexity $(x) < t$ and 0, otherwise. Since the threshold t needs to be set, prior work (Shi et al. 2023) uses AUC (Area Under the ROC Curve) as an evaluation metric which is independent of t. Note that we employ the same metric in our experiments.

LLM Watermarking. Watermarking LLMs consists of introducing signals during its training or inference that are difficult to detect by humans without the knowledge of a watermark key but can be detected using an algorithm if the key is known. We focus our paper on recent methods that employ logits distribution changes as a way of inserting watermark signals during the decoding process (Kirchenbauer et al. 2023; Zhao et al. 2023).

# MIAs

Current MIAs for detecting training data rely on thresholding various heuristics that capture differences in output probabilities for each token between data included in the training set and data that was not. Below, we present an overview of these heuristics.

Perplexity. This metric distinguishes between data used to train the model (members) and data that was not (non-members), as members are generally expected to have lower perplexity.

Smaller Ref, Lowercase and Zlib (Carlini et al. 2021). Smaller Ref is defined as the ratio of the log-perplexity of the target LLM on a sample to the log-perplexity of a smaller reference LLM on the same sample. Lowercase represents the ratio of the log-perplexity of the target LLM on the original sample to the log-perplexity of the LLM on the lowercase version of the sample. Zlib is defined as the ratio of the log-perplexity of the target LLM on a sample to the zlib entropy of the same sample.

Min-K% Prob (Shi et al. 2023). This heuristic computes the average of the minimum $K\%$ token probabilities outputted by the LLM on the sample. Note that this method requires tuning $K$ , so in all our experiments we chose the best result over $K\% \in \{5\%, 10\%, 20\%, 30\%, 40\%, 50\%, 60\%\}$ .

# LLM Watermarking Methods

UMD (Kirchenbauer et al. 2023) splits the vocabulary into two sets (green and red) and biases the model towards the green tokens by altering the logit distribution. The hash of the previous token's ID serves as a seed for a pseudorandom number generator used to split the vocabulary into these two groups. For a “hard” watermark, the model is forced not to sample from the red list at all. For a “soft” watermark, a positive bias $\delta$ is added to the logits of the green tokens before sampling. We focus our empirical evaluation on “soft” watermarks as they are more suitable for LLM deployment due to their smaller impact on the quality of the generated text.

Unigram-Watermark (Zhao et al. 2023) employs a similar approach of splitting the vocabulary into two sets and biasing the model towards one of the two sets. However, the split remains consistent throughout the generation. This choice is made to provide a provable improvement against paraphrasing attacks (Krishna et al. 2024).

<table><tr><td></td><td></td><td colspan="2">Llama-30B</td><td colspan="2">Llama-13B</td></tr><tr><td></td><td>P.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td></tr><tr><td rowspan="3">UMD</td><td>0</td><td>3.3</td><td>31.2</td><td>4.9</td><td>34.3</td></tr><tr><td>10</td><td>2.8</td><td>28.7</td><td>3.5</td><td>31.9</td></tr><tr><td>20</td><td>2.4</td><td>30.1</td><td>3.5</td><td>33.4</td></tr><tr><td rowspan="3">Unigram</td><td>0</td><td>4.1</td><td>34.1</td><td>5.0</td><td>36.6</td></tr><tr><td>10</td><td>3.0</td><td>31.7</td><td>4.0</td><td>34.3</td></tr><tr><td>20</td><td>2.4</td><td>31.5</td><td>3.4</td><td>34.0</td></tr></table>

Table 1: Measuring the reduction in verbatim memorization of training texts on WikiMIA-32. We report the relative increase in both the minimum and average perplexity between the watermarked and unwatermarked models, where larger values correspond to less memorization. Note that “P” stands for “prompt length”.

# Watermarking LLMs Prevents Copyrighted Text Generation

In this section, we study the effect of LLM watermarking techniques on verbatim memorization. We discuss the their implications for preventing copyrighted text generation.

Datasets. We consider 4 versions of the WikiMIA benchmark (Shi et al. 2023) with 32, 64, 128, and 256 words in each sample and only consider the samples that were very likely part of the training set of all the models we consider (labeled as 1 in Shi et al. (2023)). We consider these subsets as a proxy for text that was used in the training set, and the model may be prone to verbatim memorization. From now on, we refer to this subset as the “training samples” or “training texts”. Similarly, we consider BookMIA dataset (Shi et al. 2023), which contains samples from copyrighted books.

Metric. We measure the relative increase in perplexity on the generation of training samples by the watermarked model compared to the original model. We report the increase in both the minimum and average perplexity over the training samples. Note that a large increase in perplexity corresponds to a large decrease in the probability of generating that specific sample, as shown later in this section. When computing the perplexity, we prompt the model with an empty string, the first 10, and the first 20 tokens of the targeted training sample, respectively. In the BookMIA dataset, we designate the initial 100 or 256 tokens as the prompt. This is because each BookMIA sample contains 512 words, which is larger than the sample size in WikiMIA.

Models. We conduct our empirical evaluation on 5 recent LLMs: Llama-30B (Touvron et al. 2023), GPT-NeoX-20B (Black et al. 2022), Llama-13B (Touvron et al. 2023), Pythia-2.8B (Biderman et al. 2023) and OPT-2.7B (Zhang et al. 2022).

![](images/1d1edcacd327eb12a71154d3856bf7001bf80e2009d1309e0c5e2f9243f5f5b3.jpg)

<details>
<summary>line</summary>

| Watermark Strength | Train Samples (given 20 tokens) | Train Samples (given 10 tokens) | Train Samples (given 0 tokens) | Free Samples (given 0 tokens) |
| ------------------ | -------------------------------- | -------------------------------- | ------------------------------- | ------------------------------ |
| 2                  | 0                                | 0                                | 0                               | 0                              |
| 4                  | 0                                | 0                                | 0                               | 0                              |
| 6                  | 0                                | 0                                | 0                               | 0                              |
| 8                  | 0                                | 0                                | 0                               | 0                              |
| 10                 | 0                                | 0                                | 0                               | 0                              |
| 12                 | 100                              | 100                              | 100                             | 0                              |
| 14                 | 300                              | 300                              | 300                             | 0                              |
| 16                 | 900                              | 850                              | 850                             | 0                              |
</details>

![](images/7f88fcab0a87b4198b6ddac511f66e8a6963c25ebc763207a20658babc4e99a2.jpg)

<details>
<summary>line</summary>

| Watermark Strength | Relative Increase in Perplexity Due to Watermarking |
| ------------------ | ---------------------------------------------------- |
| 2                  | 0.0                                                  |
| 4                  | 0.0                                                  |
| 6                  | 0.0                                                  |
| 8                  | 0.5                                                  |
| 10                 | 3.0                                                  |
| 12                 | 7.0                                                  |
| 14                 | 16.0                                                 |
| 16                 | 33.0                                                 |
</details>

Figure 2: We study how the watermark strength (under the UMD scheme) affects the average and the minimum perplexity of training samples from WikiMIA-32, as well as the quality of generated text.

# Empirical Evaluation

In Table 1, we show the increase in perplexity on the training samples when the model is watermarked relative to the unwatermarked model. We observe that for Llama-30B, Unigram-Watermark induces a relative increase of 4.1 in the minimum and 34.1 in the average perplexity. Note that a relative increase of 4.1 in perplexity for a sample makes it more than $4.3 \times 10^{22}$ times less likely to be generated. This is based on a sample with only 32 tokens, which is likely a lower bound since the number of tokens is typically larger than the number of words. We observe consistent results over several models and prompt lengths. For all experiments, unless otherwise specified, we use a fixed strength parameter $\delta = 10$ for watermark methods and a fixed percentage of 50% green tokens. All the results are averaged over 5 runs with different seeds for the watermark methods. We include additional results on WikiMIA-64, WikiMIA-128 and WikiMIA-256 in Tables 7, 8 and 9, respectively, in the appendix. We observe that our findings are consistent across models and splits of WikiMIA. Finally, we include the complete version of Table 1 in the appendix (Table 6), which shows results for additional models and random logit perturbations with the same strength as the watermarking methods. Overall, the additional results are consistent with our previous findings.

In Figure 2, as well as Figure 5 from the appendix, we study the influence of the strength of the watermark $\delta$ on the relative increase in both the minimum and average perplexity on the WikiMIA-32 training samples. In this experiment, we also consider a baseline of generating text freely to study the impact of watermarks on the quality of text relative to the impact on training samples' generation (here, perplexity is computed by an unwatermarked model). All the results are averaged over 5 runs with different seeds for the watermark methods. In the case of free generation, we generate 100 samples for 5 different watermarking seeds and average the results. The length of the generated samples is up to 42 tokens, which is approximately 32 words in the benchmark (on a token-to-word ratio of 4:3). The results show an exponential increase in the perplexity of the training samples with the increase in watermark strength, while the generation quality is affected at a slower rate. This suggests that even if there is a trade-off between protecting the generation of text memorized verbatim and generating high-quality text, finding a suitable watermark strength for each particular application is possible. Examples of generated samples at varying watermark strengths are provided in the appendix.

<table><tr><td></td><td></td><td colspan="2">Llama-30B</td><td colspan="2">Llama-13B</td></tr><tr><td></td><td>P.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td></tr><tr><td rowspan="4">UMD</td><td>0</td><td>1.5</td><td>33.7</td><td>2.4</td><td>41.2</td></tr><tr><td>10</td><td>1.5</td><td>33.6</td><td>2.3</td><td>41.0</td></tr><tr><td>20</td><td>1.4</td><td>33.5</td><td>2.3</td><td>40.8</td></tr><tr><td>100</td><td>1.3</td><td>32.9</td><td>1.9</td><td>40.3</td></tr><tr><td rowspan="4">Unigram</td><td>0</td><td>1.6</td><td>36.4</td><td>2.4</td><td>44.5</td></tr><tr><td>10</td><td>1.6</td><td>36.3</td><td>2.4</td><td>44.3</td></tr><tr><td>20</td><td>1.5</td><td>36.1</td><td>2.3</td><td>44.2</td></tr><tr><td>100</td><td>1.4</td><td>35.5</td><td>1.8</td><td>43.6</td></tr></table>

Table 2: Measuring the reduction in verbatim memorization of training texts on BookMIA. We report the relative increase in both the minimum and average perplexity between the watermarked and unwatermarked models, where larger values correspond to less memorization. Note that “P.” stands for “prompt length”.

Approximate Memorization. Informally, we consider a training sample approximately memorized by a model if, given its prefix, it is possible to generate a completion that is similar enough to the ground truth completion. In our experiments, we use models fine-tuned on a subset of BookMIA (details provided in the appendix) and we consider Normalized Edit Similarity (referred to as edit similarity from now on) and BLEU score as similarity measures, as in (Ippolito et al. 2023). Note that we consider both word-level and token-level variants for the BLEU score. The range for each metric is between 0 and 1, where values close to 1 represent similar texts. In all experiments, since all the samples are 512 words long, we consider the first 256 words as the prefix and the last 256 words as the ground truth completion. We present the results for edit similarity with the UMD watermark in Figure 3, and the complete results—using all metrics and including the Unigram watermark—in Figure 7,

averaged over 20 runs with different random seeds. Note that the duplication factor (shown on x-axis) represents the number of times the target copyrighted text is duplicated. We observe that for high levels of memorization, a strong watermark significantly reduces the similarity between the generated completion and the ground truth (copyrighted) one.

![](images/2c851ba229b32e96a5e89c40275d66550a5e7ec7f2b0efc5fc222fa7a174114a.jpg)

<details>
<summary>line</summary>

| Duplication factor | w/o watermark | w/ watermark (strength = 2) | w/ watermark (strength = 5) | w/ watermark (strength = 10) |
| ------------------ | ------------- | --------------------------- | --------------------------- | ---------------------------- |
| 0                  | 0.25          | 0.25                        | 0.25                        | 0.25                         |
| 10                 | 0.25          | 0.25                        | 0.25                        | 0.25                         |
| 20                 | 0.25          | 0.25                        | 0.25                        | 0.25                         |
| 50                 | 0.7           | 0.65                        | 0.28                        | 0.25                         |
</details>

Figure 3: Edit similarity between the generated completion and the ground truth when considering different watermark strengths and memorization levels.

Takeaways. Watermarking significantly increases the perplexity of generating training texts, reducing verbatim memorization likelihood. This is achieved with only a moderate impact on the overall quality of generated text. This suggests that watermark strength can be effectively tailored to balance verbatim memorization and text quality for specific applications. Finally, we believe that our findings on WikiMIA—which does not necessarily contain copyrighted data—directly extend to the generation of copyrighted text verbatim, as this constitutes a form of verbatim memorization of the training data. To confirm, we run similar experiments on a dataset containing copyrighted data (BookMIA) and include the results in the Table 2. Additionally, we consider finetuning Llama-7B (Touvron et al. 2023) on BookMIA while controlling memorization by duplicating training samples. Detailed information about this experiment is provided in the appendix.

# Impact of Watermarking on Pretraining Data Detection

Datasets. We revisit the WikiMIA benchmark as discussed in the previous section. We consider the full datasets, rather than the subset of samples that were part of the training for models we study. Additionally, we consider the BookMIA benchmark, which contains copyrighted texts.

Metrics. We follow the prior work (Shi et al. 2023; Duarte et al. 2024) and report the AUC and AUC drop to study the detection performance of the MIAs. Note that this metric has the advantage of not having to tune the threshold for the detection classifier.

![](images/92afcc92e99823ea1b967f46348af60a98503cdc5cdbfa9d9ef6230d3efccba3.jpg)

<details>
<summary>line</summary>

| Watermark Strength | PPL   | Lowercase | Zlib  | Min-K% Prob |
| ------------------ | ----- | --------- | ----- | ----------- |
| 2                  | 0.7   | 1.2       | 0.5   | 0.7         |
| 4                  | 1.0   | 2.3       | 0.6   | 1.1         |
| 6                  | 1.1   | 2.7       | 0.5   | 1.1         |
| 8                  | 1.3   | 3.4       | 0.6   | 1.3         |
| 10                 | 1.4   | 4.2       | 0.7   | 1.2         |
| 12                 | 2.2   | 6.5       | 1.3   | 1.7         |
</details>

Figure 4: AUC drop due to watermarking for each MIA when varying the strength of the watermark.

Models. We conduct experiments on the same LLMs as in the previous section. Additionally, for the Smaller Ref method that requires a smaller reference model along with the target LLM, we consider Llama-7B, Neo-125M, Pythia-70M, and OPT-350M as references.

# Empirical Evaluation

In Table 4, we show the AUC for the unwatermarked and watermarked models using the UMD scheme, as well as the drop between the two. We observe that watermarking reduces the AUC (drop shown in bold in the table) by up to $14.2\%$ across 4 detection methods and 5 LLMs. All the experiments on watermarked models are run with 5 different seeds and we report the mean and standard deviation of the results. We also report the AUC drop, which is computed by the difference between the AUC for the unwatermarked model and the mean AUC over the 5 runs for the watermarked model. Additionally, while the experiments from Table 4 are conducted on WikiMIA-256, we observe similar trends for WikiMIA-32, WikiMIA-64, and WikiMIA-128 in the appendix. We also study the impact of the watermark's strength on the AUC drop for Llama-30B in Figure 4 and for the other models in Figure 6 from the appendix. Note that we considered WikiMIA-256 for these experiments. We observe that higher watermark strengths generally induce larger AUC drops.

In addition to the 4 detection methods, we also consider Smaller Ref attack, which we include in Table 13 of the appendix. We consider different variations, including an unwatermarked reference model and a watermarked one with a similar strength but a different seed or with both strength and seed changed in comparison to the watermarked target model. The baseline is an unwatermarked model with an unwatermarked reference model. We observe the AUC drops in all scenarios (up to 16.4%), which is consistent with our previous findings.

We also experiment with several percentages of green

<table><tr><td></td><td>Llama-30B</td><td>Llama-13B</td></tr><tr><td rowspan="3">PPL</td><td>85.4%</td><td>68.2%</td></tr><tr><td>84.7 ± 1.4%</td><td>67.6 ± 2.5%</td></tr><tr><td>0.7%</td><td>0.6%</td></tr><tr><td rowspan="3">Lowercase</td><td>87.9%</td><td>77.6%</td></tr><tr><td>80.9 ± 3.1%</td><td>67.2 ± 4.0%</td></tr><tr><td>7.0%</td><td>10.4%</td></tr><tr><td rowspan="3">Zlib</td><td>82.5%</td><td>62.5%</td></tr><tr><td>77.8 ± 1.2%</td><td>57.1 ± 2.0%</td></tr><tr><td>4.7%</td><td>5.4%</td></tr><tr><td rowspan="3">Min-K% Prob</td><td>85.1%</td><td>70.2%</td></tr><tr><td>85.0 ± 1.0%</td><td>68.5 ± 0.1%</td></tr><tr><td>0.1%</td><td>1.7%</td></tr></table>

Table 3: AUC of each MIA for the unwatermarked (top of each cell), watermarked models (middle of each cell), and the drop between the two (bottom of each cell) on BookMIA using UMD scheme.

tokens for a fixed watermark strength of $\delta = 10$ . We show the results in Table 14 of the appendix. We observe that for all models, in at least 80% of the cases all of the attacks' AUCs are negatively affected (positive drop value), suggesting that, in general, finding a watermarking scheme that reduces the success rates of the current MIAs is not a difficult task. Note that the experiments are run on WikiMIA for UMD scheme and the results are averaged over 5 watermark seeds.

Takeaways. Watermarking can significantly reduce the success of membership inference attacks (MIAs), with AUC drops up to 16.4%. By varying the percentage of green tokens as well as the watermark's strength, we observe that watermarking schemes can be easily tuned to negatively impact the detection success rates of MIAs. Finally, we conduct experiments on the BookMIA dataset and observe results consistent with our previous findings. These results are included in Table 3.

# Improving Detection Performance with Adaptive Min-K% Prob

This section demonstrates how an informed, adaptive attacker can improve the success rate of a recent MIA, Min-K% Prob. Our main idea is that an attacker with knowledge of the watermarking technique (including green-red token lists and watermark's strength $\delta$ ) can readjust token probabilities. This is possible even without additional information about the logit distribution, relying solely on the probability of each token from the target sample given the preceding tokens. Our approach relies on two key assumptions. First, knowledge of the watermarking scheme, which aligns with assumptions made in prior work on public watermark detection (Kirchenbauer et al. 2023). Second, access to the probability of each token in a sample, given the previous tokens—an assumption also made by the Min-K% Prob method (Shi et al. 2023).

Threat model. (1) The attacker's goal is to infer whether specific samples are part of the training set or not. In our setting, the attacker is not malicious, as the goal is to detect copyright violations. (2) Regarding the attacker's knowledge, we assume the attacker knows the watermarking method and its parameters (green and red lists, and the watermark strength), which aligns with the assumption made in prior work by Kirchenbauer et al. (2023) for public watermark detection. (3) As for the attacker's capabilities, we assume they can access the probabilities for each token in the given samples, similar to what a copyright auditor may have access to. This also mirrors the assumption made by Shi et al. (2023) in the context of training data detection.

Our method described in Algorithm 1 is based on the observation that if the denominator of softmax function (i.e., $\sum_{i} e^{z_i}$ , where $z_i$ is the logit for the $i$ -th vocabulary) does not vary significantly when generating samples with the watermarked model (and similarly for the unwatermarked model), then we can readjust the probabilities of the green tokens by "removing" the bias $\delta$ . More precisely, assuming the approximation for the denominator of softmax is good, then the probability for each token $t_i$ in an unwatermarked model will be around $\frac{e^{L_i}}{c}$ , where $L_i$ is the logit corresponding to the token $t_i$ and $c$ is a constant. However, for a watermarked model, if the token $t_i$ is green, then the probability would be approximated by $\frac{e^{L_i + \delta}}{d}$ , where $d$ is again a constant, while in the case $t_i$ is red the probability will be around $\frac{e^{L_i}}{d}$ . To compensate for the bias introduced by watermarking, we divide the probability of green tokens by $e^\delta$ and this way we end up with probabilities that are just a scaled (by $\frac{c}{d}$ ) version of the probabilities from the unwatermarked model. The scaling factor will not affect the orders between the samples when computing the average of the minimum K% log-probabilities as long as the tested sentences are approximately the same length, which is an assumption made by Shi et al. (2023) as well.

Despite the strong assumption we assumed regarding the approximation of the denominator, empirical results show that our method effectively improves the success rate of Min-K% under watermarking. We show results for two LLMs in Table 5 and include the complete results for 5 LLMs in Table 17 from the appendix. We observe that our method improves over the baseline in 95% of the cases, and the increase is as high as 4.8% (averaged over 5 runs).

Finally, we also consider adaptive versions of the Lower-case and Zlib methods. Our findings show that these adaptive methods outperform the baselines in at least 80% of cases. Detailed results are provided in the appendix.

Takeaways. We demonstrate that an adaptive attacker can leverage the knowledge of a watermarking scheme to increase the success rates of recent MIAs.

<table><tr><td></td><td>Llama-30B</td><td>NeoX-20B</td><td>Llama-13B</td><td>Pythia-2.8B</td><td>OPT-2.7B</td></tr><tr><td rowspan="3">PPL</td><td>72.0%</td><td>71.3%</td><td>71.2%</td><td>67.8%</td><td>60.5%</td></tr><tr><td> $70.6 \pm 1.9\%$ </td><td> $64.7 \pm 2.3\%$ </td><td> $70.0 \pm 2.6\%$ </td><td> $64.4 \pm 1.9\%$ </td><td> $54.9 \pm 2.2\%$ </td></tr><tr><td>1.4%</td><td>6.6%</td><td>1.2%</td><td>3.4%</td><td>5.6%</td></tr><tr><td rowspan="3">Lowercase</td><td>68.1%</td><td>68.2%</td><td>65.5%</td><td>62.9%</td><td>58.9%</td></tr><tr><td> $63.8 \pm 4.5\%$ </td><td> $55.4 \pm 5.5\%$ </td><td> $61.6 \pm 3.8\%$ </td><td> $58.7 \pm 3.2\%$ </td><td> $49.7 \pm 2.9\%$ </td></tr><tr><td>4.3%</td><td>14.2%</td><td>3.9%</td><td>4.2%</td><td>9.2%</td></tr><tr><td rowspan="3">Zlib</td><td>72.7%</td><td>73.2%</td><td>73.1%</td><td>69.2%</td><td>62.7%</td></tr><tr><td> $72.0 \pm 1.6\%$ </td><td> $66.6 \pm 2.0\%$ </td><td> $71.6 \pm 2.3\%$ </td><td> $66.1 \pm 1.2\%$ </td><td> $58.1 \pm 1.8\%$ </td></tr><tr><td>0.7%</td><td>6.6%</td><td>1.5%</td><td>3.1%</td><td>4.6%</td></tr><tr><td rowspan="3">Min-K% Prob</td><td>71.8%</td><td>78.0%</td><td>72.9%</td><td>71.0%</td><td>65.5%</td></tr><tr><td> $70.5 \pm 1.8\%$ </td><td> $76.2 \pm 2.1\%$ </td><td> $70.4 \pm 3.2\%$ </td><td> $69.5 \pm 1.6\%$ </td><td> $63.1 \pm 3.4\%$ </td></tr><tr><td>1.3%</td><td>1.8%</td><td>2.5%</td><td>1.5%</td><td>2.4%</td></tr></table>

Table 4: AUC of each MIA for the unwatermarked (top of each cell), watermarked models (middle of each cell), and the drop between the two (bottom of each cell) on WikiMIA-256 using UMD scheme.

<table><tr><td colspan="2">Algorithm 1: Adaptive Min-K% Prob</td></tr><tr><td colspan="2">Require: Tokenized target sample  $t = t_1 \oplus t_2 \oplus ... \oplus t_n$ , access to the probability of the target (watermarked) LLM  $f$  to generate  $t_i$  given the  $i - 1$  previous tokens and  $t_0$  (empty string)  $f(t_i|t_0 \oplus t_1 \oplus ... \oplus t_{i-1})$  (similar assumption as Min-K% Prob algorithm),  $K$ , we assume we know the watermarking scheme (e.g., for public watermark detection purposes), i.e. we know the green and red lists as well as  $\delta$ .</td></tr><tr><td colspan="2">Output: Adjusted average of the minimum  $K\%$  token probabilities when generating  $t_1 \oplus t_2 \oplus ... \oplus t_n$ adj_prob ← {} ▷ The set of adjusted probabilitiesfor  $i \in 1,2,\ldots,n$  do $p_f(t_i) \leftarrow f(t_i|t_0 \oplus t_1 \oplus ... \oplus t_{i-1})$  ▷ Probability of  $t_i$  when the model is watermarkedif  $t_i$  is green thenadj_prob ← adj_prob  $\cup \{ \frac{p_f(t_i)}{e^\delta} \}$  ▷ Adjust the probability if the token is greenelseadj_prob ← adj_prob  $\cup \{ p_f(t_i) \}$ end</td></tr><tr><td colspan="2">end $k = floor(n \cdot K\%)$  ▷ Find the number of token probabilities to keepadj_k_prob ← min_k(adj_prob) ▷ Select the minimum  $k$  probabilitiesreturn mean(log(adj_k_prob)) ▷ Return the mean of the minimum  $k$  log-probabilities</td></tr></table>

<table><tr><td></td><td></td><td>Llama-30B</td><td>Llama-13B</td></tr><tr><td rowspan="2">WikiMIA 32</td><td>Not adapt.</td><td>66.2%</td><td>64.5%</td></tr><tr><td>Adapt.</td><td>68.5%</td><td>66.3%</td></tr><tr><td rowspan="2">WikiMIA 64</td><td>Not adapt.</td><td>64.4%</td><td>62.8%</td></tr><tr><td>Adapt.</td><td>67.3%</td><td>64.9%</td></tr><tr><td rowspan="2">WikiMIA 128</td><td>Not adapt.</td><td>70.0%</td><td>68.9%</td></tr><tr><td>Adapt.</td><td>73.1%</td><td>71.0%</td></tr><tr><td rowspan="2">WikiMIA 256</td><td>Not adapt.</td><td>70.5%</td><td>70.4%</td></tr><tr><td>Adapt.</td><td>71.3%</td><td>72.4%</td></tr></table>

Table 5: We show the AUC of Min-%K Prob (referred as “Not adapt.”) and our method (referred as “Adapt.”) when using UMD watermarking scheme. We highlight the cases when our method improves over the baseline.

# Conclusion and Discussion

Watermarking LLMs has unintended consequences on methods towards copyright protection. Our experiments demonstrate that while watermarking may be a promising solution to prevent copyrighted text generation, watermarking also complicates membership inference attacks that may be employed to detect copyright abuses. Watermarking can be a double-edged sword for copyright regulators since it promotes compliance during generation time, while making training time copyright violations harder to detect. We hope our work furthers the discussion around watermarking and copyright issues for LLMs.

# Acknowledgements

Panaitescu-Liess, Che, An, Xu, Pathmanathan, Chakraborty, Zhu, and Huang are supported by DARPA Transfer from Imprecise and Abstract Models to Autonomous Technologies

(TIAMAT) 80321, National Science Foundation NSF-IIS-2147276 FAI, DOD-AFOSR-Air Force Office of Scientific Research under award number FA9550-23-1-0048, Adobe, Capital One and JP Morgan faculty fellowships.

# References

Aaronson, S. 2023. Simons institute talk on watermarking of large language models.   
Bentley, J. W.; Gibney, D.; Hoppenworth, G.; and Jha, S. K. 2020. Quantifying membership inference vulnerability via generalization gap and other model metrics. arXiv preprint arXiv:2009.05669.   
Biderman, S.; Schoelkopf, H.; Anthony, Q. G.; Bradley, H.; O'Brien, K.; Hallahan, E.; Khan, M. A.; Purohit, S.; Prashanth, U. S.; Raff, E.; et al. 2023. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, 2397–2430. PMLR.   
Black, S.; Biderman, S.; Hallahan, E.; Anthony, Q.; Gao, L.; Golding, L.; He, H.; Leahy, C.; McDonell, K.; Phang, J.; et al. 2022. Gpt-neox-20b: An open-source autoregressive language model. arXiv preprint arXiv:2204.06745.   
Carlini, N.; Chien, S.; Nasr, M.; Song, S.; Terzis, A.; and Tramer, F. 2022a. Membership inference attacks from first principles. In 2022 IEEE Symposium on Security and Privacy (SP), 1897–1914. IEEE.   
Carlini, N.; Hayes, J.; Nasr, M.; Jagielski, M.; Sehwag, V.; Tramer, F.; Balle, B.; Ippolito, D.; and Wallace, E. 2023. Extracting training data from diffusion models. In 32nd USENIX Security Symposium (USENIX Security 23), 5253–5270.   
Carlini, N.; Ippolito, D.; Jagielski, M.; Lee, K.; Tramer, F.; and Zhang, C. 2022b. Quantifying memorization across neural language models. arXiv preprint arXiv:2202.07646.   
Carlini, N.; Liu, C.; Erlingsson, Ú.; Kos, J.; and Song, D. 2019. The secret sharer: Evaluating and testing unintended memorization in neural networks. In 28th USENIX security symposium (USENIX security 19), 267–284.   
Carlini, N.; Tramer, F.; Wallace, E.; Jagielski, M.; Herbert-Voss, A.; Lee, K.; Roberts, A.; Brown, T.; Song, D.; Erlingsson, U.; et al. 2021. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), 2633–2650.   
Christ, M.; Gunn, S.; and Zamir, O. 2023. Undetectable watermarks for language models. arXiv preprint arXiv:2306.09194.   
Chu, T.; Song, Z.; and Yang, C. 2024. How to Protect Copyright Data in Optimization of Large Language Models? In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 17871–17879.   
Das, D.; Zhang, J.; and Tramèr, F. 2024. Blind Baselines Beat Membership Inference Attacks for Foundation Models. arXiv preprint arXiv:2406.16201.   
Duarte, A. V.; Zhao, X.; Oliveira, A. L.; and Li, L. 2024. DE-COP: Detecting Copyrighted Content in Language Models Training Data. arXiv preprint arXiv:2402.09910.

Elkin-Koren, N.; Hacohen, U.; Livni, R.; and Moran, S. 2023. Can Copyright be Reduced to Privacy? arXiv preprint arXiv:2305.14822.   
Hacohen, U.; Haviv, A.; Sarfaty, S.; Friedman, B.; Elkin-Koren, N.; Livni, R.; and Bermano, A. H. 2024. Not All Similarities Are Created Equal: Leveraging Data-Driven Biases to Inform GenAI Copyright Disputes. arXiv preprint arXiv:2403.17691.   
Hans, A.; Wen, Y.; Jain, N.; Kirchenbauer, J.; Kazemi, H.; Singhania, P.; Singh, S.; Somepalli, G.; Geiping, J.; Bhatele, A.; et al. 2024. Be like a Goldfish, Don’t Memorize! Mitigating Memorization in Generative LLMs. arXiv preprint arXiv:2406.10209.   
Ippolito, D.; Tramèr, F.; Nasr, M.; Zhang, C.; Jagielski, M.; Lee, K.; Choquette-Choo, C. A.; and Carlini, N. 2023. Preventing generation of verbatim memorization in language models gives a false sense of privacy. In Proceedings of the 16th International Natural Language Generation Conference, 28–53. Association for Computational Linguistics.   
Kandpal, N.; Wallace, E.; and Raffel, C. 2022. Deduplicating training data mitigates privacy risks in language models. In International Conference on Machine Learning, 10697–10707. PMLR.   
Karamolegkou, A.; Li, J.; Zhou, L.; and Søgaard, A. 2023. Copyright violations and large language models. arXiv preprint arXiv:2310.13771.   
Kirchenbauer, J.; Geiping, J.; Wen, Y.; Katz, J.; Miers, I.; and Goldstein, T. 2023. A watermark for large language models. In International Conference on Machine Learning, 17061–17084. PMLR.   
Krishna, K.; Song, Y.; Karpinska, M.; Wieting, J.; and Iyyer, M. 2024. Paraphrasing evades detectors of ai-generated text, but retrieval is an effective defense. Advances in Neural Information Processing Systems, 36.   
Kuditipudi, R.; Thickstun, J.; Hashimoto, T.; and Liang, P. 2023. Robust distortion-free watermarks for language models. arXiv preprint arXiv:2307.15593.   
Lee, K.; Ippolito, D.; Nystrom, A.; Zhang, C.; Eck, D.; Callison-Burch, C.; and Carlini, N. 2021. Deduplicating training data makes language models better. arXiv preprint arXiv:2107.06499.   
Liu, A.; Pan, L.; Hu, X.; Meng, S.; and Wen, L. 2024. A Semantic Invariant Robust Watermark for Large Language Models. In The Twelfth International Conference on Learning Representations.   
Mattern, J.; Mireshghallah, F.; Jin, Z.; Schoelkopf, B.; Sachan, M.; and Berg-Kirkpatrick, T. 2023. Membership Inference Attacks against Language Models via Neighbourhood Comparison. In Rogers, A.; Boyd-Graber, J.; and Okazaki, N., eds., Findings of the Association for Computational Linguistics: ACL 2023, 11330–11343. Toronto, Canada: Association for Computational Linguistics.   
Mireshghallah, F.; Goyal, K.; Uniyal, A.; Berg-Kirkpatrick, T.; and Shokri, R. 2022. Quantifying privacy risks of masked language models using membership inference attacks. arXiv preprint arXiv:2203.03929.

Oren, Y.; Meister, N.; Chatterji, N.; Ladhak, F.; and Hashimoto, T. B. 2023. Proving test set contamination in black box language models. arXiv preprint arXiv:2310.17623.   
Ren, J.; Xu, H.; He, P.; Cui, Y.; Zeng, S.; Zhang, J.; Wen, H.; Ding, J.; Liu, H.; Chang, Y.; et al. 2024. Copyright Protection in Generative AI: A Technical Perspective. arXiv preprint arXiv:2402.02333.   
Sablayrolles, A.; Douze, M.; Schmid, C.; Ollivier, Y.; and Jégou, H. 2019. White-box vs black-box: Bayes optimal strategies for membership inference. In International Conference on Machine Learning, 5558–5567. PMLR.   
Shejwalkar, V.; Inan, H. A.; Houmansadr, A.; and Sim, R. 2021. Membership inference attacks against nlp classification models. In NeurIPS 2021 Workshop Privacy in Machine Learning.   
Shi, W.; Ajith, A.; Xia, M.; Huang, Y.; Liu, D.; Blevins, T.; Chen, D.; and Zettlemoyer, L. 2023. Detecting pre-training data from large language models. arXiv preprint arXiv:2310.16789.   
Shokri, R.; Stronati, M.; Song, C.; and Shmatikov, V. 2017. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP), 3–18. IEEE.   
Somepalli, G.; Singla, V.; Goldblum, M.; Geiping, J.; and Goldstein, T. 2023a. Diffusion art or digital forgery? investigating data replication in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6048–6058.   
Somepalli, G.; Singla, V.; Goldblum, M.; Geiping, J.; and Goldstein, T. 2023b. Understanding and mitigating copying in diffusion models. Advances in Neural Information Processing Systems, 36: 47783–47803.   
Song, C.; and Shmatikov, V. 2019. Auditing data provenance in text-generation models. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 196–206.   
Tirumala, K.; Markosyan, A.; Zettlemoyer, L.; and Aghajanyan, A. 2022. Memorization without overfitting: Analyzing the training dynamics of large language models. Advances in Neural Information Processing Systems, 35:38274–38290.   
Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; Rodriguez, A.; Joulin, A.; Grave, E.; and Lample, G. 2023. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971.   
Vyas, N.; Kakade, S. M.; and Barak, B. 2023. On provable copyright protection for generative models. In International Conference on Machine Learning, 35277–35299. PMLR.   
Wen, Y.; Liu, Y.; Chen, C.; and Lyu, L. 2024. Detecting, Explaining, and Mitigating Memorization in Diffusion Models. In The Twelfth International Conference on Learning Representations.   
Yeom, S.; Giacomelli, I.; Fredrikson, M.; and Jha, S. 2018. Privacy risk in machine learning: Analyzing the connection

to overfitting. In 2018 IEEE 31st computer security foundations symposium (CSF), 268–282. IEEE.

Zhang, J.; Sun, J.; Yeats, E.; Ouyang, Y.; Kuo, M.; Zhang, J.; Yang, H.; and Li, H. 2024. Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models. arXiv preprint arXiv:2404.02936.

Zhang, S.; Roller, S.; Goyal, N.; Artetxe, M.; Chen, M.; Chen, S.; Dewan, C.; Diab, M.; Li, X.; Lin, X. V.; et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

Zhao, X.; Ananth, P.; Li, L.; and Wang, Y.-X. 2023. Provable robust watermarking for ai-generated text. arXiv preprint arXiv:2306.17439.

# Appendix

# Additional experiments on verbatim memorization on WikiMIA

Table 6: Measuring the reduction in verbatim memorization of training texts on WikiMIA-32. We report the relative increase in both the minimum and average perplexity between the watermarked and unwatermarked models, where larger values correspond to less memorization. Note that “P.” stands for “prompt length”. 

<table><tr><td></td><td></td><td colspan="2">Llama-30B</td><td colspan="2">NeoX-20B</td><td colspan="2">Llama-13B</td><td colspan="2">Pythia-2.8B</td><td colspan="2">OPT-2.7B</td></tr><tr><td></td><td>P.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td></tr><tr><td rowspan="3">UMD</td><td>0</td><td>3.3</td><td>31.2</td><td>3.7</td><td>52.1</td><td>4.9</td><td>34.3</td><td>11.4</td><td>61.3</td><td>10.4</td><td>64.5</td></tr><tr><td>10</td><td>2.8</td><td>28.7</td><td>2.2</td><td>52.1</td><td>3.5</td><td>31.9</td><td>8.8</td><td>63.7</td><td>8.3</td><td>67.7</td></tr><tr><td>20</td><td>2.4</td><td>30.1</td><td>1.8</td><td>66.0</td><td>3.5</td><td>33.4</td><td>5.0</td><td>74.0</td><td>7.0</td><td>84.4</td></tr><tr><td rowspan="3">Unigram</td><td>0</td><td>4.1</td><td>34.1</td><td>4.4</td><td>54.1</td><td>5.0</td><td>36.6</td><td>14.3</td><td>74.5</td><td>11.5</td><td>66.1</td></tr><tr><td>10</td><td>3.0</td><td>31.7</td><td>2.8</td><td>52.5</td><td>4.0</td><td>34.3</td><td>11.8</td><td>73.6</td><td>9.8</td><td>70.2</td></tr><tr><td>20</td><td>2.4</td><td>31.5</td><td>2.0</td><td>56.4</td><td>3.4</td><td>34.0</td><td>6.6</td><td>79.1</td><td>5.8</td><td>81.4</td></tr><tr><td rowspan="3">Random</td><td>0</td><td>4.0</td><td>34.3</td><td>4.9</td><td>51.1</td><td>5.5</td><td>34.7</td><td>8.0</td><td>62.3</td><td>7.4</td><td>60.6</td></tr><tr><td>10</td><td>2.6</td><td>31.4</td><td>3.1</td><td>51.4</td><td>3.6</td><td>31.7</td><td>5.5</td><td>62.9</td><td>6.3</td><td>64.0</td></tr><tr><td>20</td><td>2.1</td><td>31.8</td><td>1.2</td><td>59.3</td><td>2.8</td><td>32.9</td><td>4.7</td><td>78.6</td><td>3.9</td><td>73.7</td></tr></table>

Table 7: Measuring the reduction in verbatim memorization of training texts on WikiMIA-64. We report the relative increase in both the minimum and average perplexity between the watermarked and unwatermarked models, where larger values correspond to less memorization. Note that “P.” stands for “prompt length”. 

<table><tr><td></td><td></td><td colspan="2">Llama-30B</td><td colspan="2">NeoX-20B</td><td colspan="2">Llama-13B</td><td colspan="2">Pythia-2.8B</td><td colspan="2">OPT-2.7B</td></tr><tr><td></td><td>P.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td></tr><tr><td rowspan="3">UMD</td><td>0</td><td>4.9</td><td>27.6</td><td>4.2</td><td>42.8</td><td>6.7</td><td>30.5</td><td>15.2</td><td>50.4</td><td>14.9</td><td>51.7</td></tr><tr><td>10</td><td>4.3</td><td>26.2</td><td>3.7</td><td>41.3</td><td>6.1</td><td>29.1</td><td>15.6</td><td>49.2</td><td>14.4</td><td>51.4</td></tr><tr><td>20</td><td>3.9</td><td>26.4</td><td>3.6</td><td>43.1</td><td>5.8</td><td>29.3</td><td>14.0</td><td>50.3</td><td>12.5</td><td>52.7</td></tr><tr><td rowspan="3">Unigram</td><td>0</td><td>5.0</td><td>28.1</td><td>4.3</td><td>45.3</td><td>6.7</td><td>30.9</td><td>17.6</td><td>62.3</td><td>16.0</td><td>53.7</td></tr><tr><td>10</td><td>3.8</td><td>26.9</td><td>3.4</td><td>43.6</td><td>5.3</td><td>29.7</td><td>16.2</td><td>60.6</td><td>17.1</td><td>53.0</td></tr><tr><td>20</td><td>3.2</td><td>26.9</td><td>3.1</td><td>44.2</td><td>4.4</td><td>29.7</td><td>13.6</td><td>60.9</td><td>11.7</td><td>53.6</td></tr></table>

Table 8: Measuring the reduction in verbatim memorization of training texts on WikiMIA-128. We report the relative increase in both the minimum and average perplexity between the watermarked and unwatermarked models, where larger values correspond to less memorization. Note that “P.” stands for “prompt length”. 

<table><tr><td></td><td></td><td colspan="2">Llama-30B</td><td colspan="2">NeoX-20B</td><td colspan="2">Llama-13B</td><td colspan="2">Pythia-2.8B</td><td colspan="2">OPT-2.7B</td></tr><tr><td></td><td>P.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td></tr><tr><td rowspan="3">UMD</td><td>0</td><td>5.7</td><td>25.3</td><td>4.6</td><td>39.5</td><td>7.6</td><td>28.0</td><td>23.1</td><td>45.3</td><td>18.6</td><td>48.1</td></tr><tr><td>10</td><td>5.3</td><td>24.4</td><td>4.3</td><td>38.9</td><td>7.2</td><td>27.1</td><td>23.6</td><td>44.7</td><td>19.1</td><td>47.6</td></tr><tr><td>20</td><td>5.2</td><td>24.5</td><td>4.3</td><td>39.3</td><td>6.8</td><td>27.2</td><td>23.0</td><td>44.7</td><td>17.5</td><td>47.8</td></tr><tr><td rowspan="3">Unigram</td><td>0</td><td>4.5</td><td>25.6</td><td>5.9</td><td>42.9</td><td>6.4</td><td>28.2</td><td>17.6</td><td>54.9</td><td>19.6</td><td>50.0</td></tr><tr><td>10</td><td>3.9</td><td>25.0</td><td>5.3</td><td>42.0</td><td>5.7</td><td>27.6</td><td>15.8</td><td>53.6</td><td>18.7</td><td>49.9</td></tr><tr><td>20</td><td>3.6</td><td>25.2</td><td>5.1</td><td>42.1</td><td>5.3</td><td>27.7</td><td>15.1</td><td>53.6</td><td>18.0</td><td>49.9</td></tr></table>

Table 9: Measuring the reduction in verbatim memorization of training texts on WikiMIA-256. We report the relative increase in both the minimum and average perplexity between the watermarked and unwatermarked models, where larger values correspond to less memorization. Note that “P.” stands for “prompt length”. 

<table><tr><td></td><td></td><td colspan="2">Llama-30B</td><td colspan="2">NeoX-20B</td><td colspan="2">Llama-13B</td><td colspan="2">Pythia-2.8B</td><td colspan="2">OPT-2.7B</td></tr><tr><td></td><td>P.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td><td>Min.</td><td>Avg.</td></tr><tr><td rowspan="3">UMD</td><td>0</td><td>7.5</td><td>23.9</td><td>15.4</td><td>37.8</td><td>13.0</td><td>26.3</td><td>31.2</td><td>45.4</td><td>27.3</td><td>46.3</td></tr><tr><td>10</td><td>7.3</td><td>23.4</td><td>15.5</td><td>37.6</td><td>12.5</td><td>25.8</td><td>30.9</td><td>45.2</td><td>27.5</td><td>46.0</td></tr><tr><td>20</td><td>7.2</td><td>23.5</td><td>16.1</td><td>37.6</td><td>12.6</td><td>25.9</td><td>30.4</td><td>45.0</td><td>27.6</td><td>46.2</td></tr><tr><td rowspan="3">Unigram</td><td>0</td><td>7.4</td><td>24.4</td><td>21.0</td><td>42.4</td><td>13.9</td><td>26.8</td><td>36.9</td><td>54.3</td><td>28.8</td><td>46.3</td></tr><tr><td>10</td><td>7.1</td><td>24.1</td><td>21.2</td><td>41.9</td><td>13.7</td><td>26.5</td><td>35.4</td><td>53.7</td><td>28.2</td><td>46.0</td></tr><tr><td>20</td><td>6.7</td><td>24.2</td><td>21.5</td><td>41.8</td><td>13.7</td><td>26.5</td><td>34.6</td><td>53.4</td><td>29.3</td><td>45.9</td></tr></table>

![](images/59898bdb71a777c6550a22ac46bd5bc4ad5fa30cc1d6a027cbdfaf8aecbade11.jpg)

<details>
<summary>line</summary>

| Watermark Strength | Train Samples (given 20 tokens) | Train Samples (given 10 tokens) | Train Samples (given 0 tokens) | Free Samples (given 0 tokens) |
| ------------------ | -------------------------------- | -------------------------------- | ------------------------------- | ------------------------------ |
| 2                  | 0                                | 0                                | 0                               | 0                              |
| 4                  | 0                                | 0                                | 0                               | 0                              |
| 6                  | 0                                | 0                                | 0                               | 0                              |
| 8                  | 0                                | 0                                | 0                               | 0                              |
| 10                 | 50                               | 50                               | 50                              | 0                              |
| 12                 | 100                              | 100                              | 100                             | 0                              |
| 14                 | 300                              | 300                              | 300                             | 0                              |
| 16                 | 1100                             | 1050                             | 1050                            | 0                              |
</details>

![](images/5b4d54553760a0f418c8a8414a7456d39a3612ca5995abc99fd2af43330182ee.jpg)

<details>
<summary>line</summary>

| Watermark Strength | Blue Line | Orange Line | Green Line | Gray Dashed Line |
| ------------------ | --------- | ----------- | ---------- | ---------------- |
| 2                  | 0         | 0           | 0          | 0                |
| 4                  | 0         | 0           | 0          | 0                |
| 6                  | 0         | 0           | 0          | 2                |
| 8                  | 2         | 1           | 1          | 1                |
| 10                 | 5         | 3           | 2          | 2                |
| 12                 | 10        | 7           | 5          | 2                |
| 14                 | 20        | 15          | 12         | 2                |
| 16                 | 40        | 30          | 25         | 3                |
</details>

Figure 5: We study how the watermark strength (under the Unigram scheme) affects the average and the minimum perplexity of training samples from WikiMIA-32, as well as the quality of generated text.

# Additional experiments on pretraining data detection on WikiMIA

Table 10: AUC of each MIA for the unwatermarked (top of each cell), watermarked models (middle of each cell) and the drop between the two (bottom of each cell) on WikiMIA-128 using UMD scheme. 

<table><tr><td></td><td>Llama-30B</td><td>NeoX-20B</td><td>Llama-13B</td><td>Pythia-2.8B</td><td>OPT-2.7B</td></tr><tr><td rowspan="3">PPL</td><td>70.3%</td><td>70.6%</td><td>67.7%</td><td>62.8%</td><td>60.0%</td></tr><tr><td>66.3 ± 2.2%</td><td>63.6 ± 2.4%</td><td>63.4 ± 2.6%</td><td>61.4 ± 2.3%</td><td>55.1 ± 1.6%</td></tr><tr><td>4.0%</td><td>7.0%</td><td>4.3%</td><td>1.4%</td><td>4.9%</td></tr><tr><td rowspan="3">Lowercase</td><td>59.1%</td><td>68.0%</td><td>60.6%</td><td>59.4%</td><td>57.1%</td></tr><tr><td>55.9 ± 2.9%</td><td>58.2 ± 3.4%</td><td>55.1 ± 3.0%</td><td>55.7 ± 1.6%</td><td>49.2 ± 4.5%</td></tr><tr><td>3.2%</td><td>9.2%</td><td>5.5%</td><td>3.7%</td><td>7.9%</td></tr><tr><td rowspan="3">Zlib</td><td>71.8%</td><td>72.3%</td><td>69.6%</td><td>64.9%</td><td>62.3%</td></tr><tr><td>68.6 ± 2.3%</td><td>66.3 ± 2.1%</td><td>65.8 ± 2.7%</td><td>63.9 ± 1.9%</td><td>58.9 ± 1.3%</td></tr><tr><td>3.2%</td><td>6.0%</td><td>3.8%</td><td>1.0%</td><td>3.4%</td></tr><tr><td rowspan="3">Min-K% Prob</td><td>73.8%</td><td>76.4%</td><td>71.5%</td><td>66.8%</td><td>64.3%</td></tr><tr><td>70.0 ± 1.5%</td><td>72.8 ± 2.3%</td><td>68.9 ± 2.2%</td><td>64.8 ± 1.4%</td><td>59.2 ± 2.4%</td></tr><tr><td>3.8%</td><td>3.6%</td><td>2.6%</td><td>2.0%</td><td>5.1%</td></tr></table>

Table 11: AUC of each MIA for the unwatermarked (top of each cell), watermarked models (middle of each cell) and the drop between the two (bottom of each cell) on WikiMIA-64 using UMD scheme. 

<table><tr><td></td><td>Llama-30B</td><td>NeoX-20B</td><td>Llama-13B</td><td>Pythia-2.8B</td><td>OPT-2.7B</td></tr><tr><td rowspan="3">PPL</td><td>66.1%</td><td>66.6%</td><td>63.6%</td><td>58.4%</td><td>55.1%</td></tr><tr><td> $60.7 \pm 3.4\%$ </td><td> $60.1 \pm 3.2\%$ </td><td> $58.0 \pm 3.7\%$ </td><td> $58.7 \pm 1.7\%$ </td><td> $52.2 \pm 2.1\%$ </td></tr><tr><td>5.4%</td><td>6.5%</td><td>5.6%</td><td>-0.3%</td><td>2.9%</td></tr><tr><td rowspan="3">Lowercase</td><td>61.8%</td><td>66.4%</td><td>62.0%</td><td>57.7%</td><td>56.6%</td></tr><tr><td> $54.8 \pm 1.7\%$ </td><td> $56.8 \pm 3.8\%$ </td><td> $53.8 \pm 1.1\%$ </td><td> $54.5 \pm 1.0\%$ </td><td> $51.4 \pm 3.1\%$ </td></tr><tr><td>7.0%</td><td>9.6%</td><td>8.2%</td><td>3.2%</td><td>5.2%</td></tr><tr><td rowspan="3">Zlib</td><td>67.4%</td><td>68.1%</td><td>65.3%</td><td>60.5%</td><td>57.7%</td></tr><tr><td> $62.4 \pm 3.3\%$ </td><td> $62.0 \pm 2.6\%$ </td><td> $59.9 \pm 3.6\%$ </td><td> $60.9 \pm 1.8\%$ </td><td> $55.5 \pm 1.5\%$ </td></tr><tr><td>5.0%</td><td>6.1%</td><td>4.9%</td><td>5.4%</td><td>2.2%</td></tr><tr><td rowspan="3">Min-K% Prob</td><td>68.4%</td><td>72.8%</td><td>65.9%</td><td>61.2%</td><td>58.0%</td></tr><tr><td> $64.4 \pm 2.9\%$ </td><td> $67.7 \pm 3.3\%$ </td><td> $62.8 \pm 3.4\%$ </td><td> $59.8 \pm 0.7\%$ </td><td> $55.3 \pm 2.3\%$ </td></tr><tr><td>4.0%</td><td>5.1%</td><td>3.1%</td><td>1.4%</td><td>2.7%</td></tr></table>

Table 12: AUC of each MIA for the unwatermarked (top of each cell), watermarked models (middle of each cell) and the drop between the two (bottom of each cell) on WikiMIA-32 using UMD scheme. 

<table><tr><td></td><td>Llama-30B</td><td>NeoX-20B</td><td>Llama-13B</td><td>Pythia-2.8B</td><td>OPT-2.7B</td></tr><tr><td rowspan="3">PPL</td><td>69.4%</td><td>69.0%</td><td>67.5%</td><td>61.3%</td><td>58.2%</td></tr><tr><td> $63.6 \pm 5.2\%$ </td><td> $62.7 \pm 3.5\%$ </td><td> $61.4 \pm 5.7\%$ </td><td> $60.8 \pm 2.3\%$ </td><td> $55.2 \pm 2.1\%$ </td></tr><tr><td>5.5%</td><td>6.3%</td><td>6.1%</td><td>0.5%</td><td>3.0%</td></tr><tr><td rowspan="3">Lowercase</td><td>64.1%</td><td>68.2%</td><td>63.9%</td><td>60.9%</td><td>59.2%</td></tr><tr><td> $54.9 \pm 1.8\%$ </td><td> $59.4 \pm 4.8\%$ </td><td> $54.2 \pm 1.8\%$ </td><td> $55.5 \pm 1.6\%$ </td><td> $52.1 \pm 3.9\%$ </td></tr><tr><td>9.2%</td><td>8.8%</td><td>9.7%</td><td>0.6%</td><td>2.8%</td></tr><tr><td rowspan="3">Zlib</td><td>69.8%</td><td>69.2%</td><td>67.8%</td><td>62.1%</td><td>59.4%</td></tr><tr><td> $64.4 \pm 4.7\%$ </td><td> $63.2 \pm 2.8\%$ </td><td> $62.3 \pm 5.1\%$ </td><td> $61.5 \pm 1.9\%$ </td><td> $56.6 \pm 1.6\%$ </td></tr><tr><td>5.4%</td><td>6.0%</td><td>5.5%</td><td>0.6%</td><td>2.8%</td></tr><tr><td rowspan="3">Min-K% Prob</td><td>70.1%</td><td>72.1%</td><td>67.9%</td><td>61.8%</td><td>59.2%</td></tr><tr><td> $66.2 \pm 4.2\%$ </td><td> $67.1 \pm 4.2\%$ </td><td> $64.5 \pm 4.1\%$ </td><td> $61.0 \pm 1.5\%$ </td><td> $55.8 \pm 2.3\%$ </td></tr><tr><td>3.9%</td><td>5.0%</td><td>3.4%</td><td>0.8%</td><td>3.4%</td></tr></table>

Table 13: Results for Smaller Ref attack on WikiMIA-256. The first two rows represent the pair of target and smaller reference model, “No model w.” row represents the baseline AUC of a unwatermarked target LLM and unwatermarked reference model, the other three “double rows” correspond to different variations of the reference model and each cell contains the AUC followed by the AUC drop in comparison to the baseline. 

<table><tr><td></td><td>Llama-30B</td><td>NeoX-20B</td><td>Llama-13B</td><td>Pythia-2.8B</td><td>OPT-2.7B</td></tr><tr><td></td><td>Llama-7B</td><td>Neo-125M</td><td>Llama-7B</td><td>Pythia-70M</td><td>OPT-350M</td></tr><tr><td>No model w.</td><td>74.7%</td><td>70.2%</td><td>70.5%</td><td>63.6%</td><td>64.4%</td></tr><tr><td>Ref. not w.</td><td>69.7 ± 3.3%5.0%</td><td>61.0 ± 1.8%9.2%</td><td>66.3 ± 4.6%4.2%</td><td>61.6 ± 2.0%2.0%</td><td>53.2 ± 3.4%11.2%</td></tr><tr><td>Ref. diff. seed</td><td>61.7 ± 4.4%13.0%</td><td>55.5 ± 3.4%15.0%</td><td>54.1 ± 4.4%16.4%</td><td>58.3 ± 2.4%5.3%</td><td>51.3 ± 4.3%13.1%</td></tr><tr><td>Ref. diff. str.</td><td>73.7 ± 2.6%1.0%</td><td>61.0 ± 3.2%9.2%</td><td>68.8 ± 4.8%1.7%</td><td>62.5 ± 1.2%1.1%</td><td>57.3 ± 3.6%7.1%</td></tr></table>

![](images/a4526fdd1b658353ca958f2184a91fa47a9879490fb0c122de90b229fb6363e2.jpg)

<details>
<summary>line</summary>

| Watermark Strength | PPL  | Lowercase | Zlib | Min-K% Prob |
| ------------------ | ---- | --------- | ---- | ----------- |
| 2                  | 1.0  | 2.5       | 1.0  | 0.0         |
| 4                  | 3.0  | 6.0       | 3.0  | 0.5         |
| 6                  | 4.0  | 9.0       | 4.5  | 1.0         |
| 8                  | 5.5  | 11.5      | 5.5  | 1.5         |
| 10                 | 6.5  | 12.5      | 6.5  | 1.8         |
| 12                 | 7.5  | 13.5      | 7.5  | 2.0         |
</details>

![](images/46682e3f6438150bf6a2568785e3ff0cc37eef77d44694a0cf05c5e3cb87a2d4.jpg)

<details>
<summary>line</summary>

| Watermark Strength | Orange Line | Red Line | Green Line | Blue Line |
| ------------------ | ----------- | -------- | ---------- | --------- |
| 2                  | 1.5         | 0.1      | 0.5        | 0.7       |
| 4                  | 2.9         | 1.3      | 0.7        | 0.6       |
| 6                  | 3.0         | 1.5      | 0.5        | 0.5       |
| 8                  | 3.4         | 2.2      | 0.8        | 0.6       |
| 10                 | 4.0         | 2.5      | 1.5        | 1.2       |
| 12                 | 4.5         | 2.6      | 2.2        | 2.1       |
</details>

![](images/d1d94c29968c61e227749b0139827eb4ada42210fddb0dfea76c4a2b8247f3b9.jpg)

<details>
<summary>line</summary>

| Watermark Strength | Orange Line | Blue Line | Green Line | Red Line |
| ------------------ | ----------- | --------- | ---------- | -------- |
| 2                  | 0.8         | 0.3       | 0.3        | 0.3      |
| 4                  | 2.5         | 1.0       | 0.7        | -0.2     |
| 6                  | 3.2         | 2.0       | 1.8        | 0.1      |
| 8                  | 4.0         | 2.7       | 2.5        | 0.8      |
| 10                 | 4.2         | 3.4       | 3.1        | 1.5      |
| 12                 | 4.2         | 4.0       | 3.7        | 1.6      |
</details>

![](images/08618d8a74da1dd210c5451eb89f5fb75c5b9956adb6df08f432c14fdb290d05.jpg)

<details>
<summary>line</summary>

| Watermark Strength | Orange Line | Blue Line | Green Line | Red Line |
| ------------------ | ----------- | --------- | ---------- | -------- |
| 2                  | 1.5         | 1.8       | 1.0        | -0.5     |
| 4                  | 4.5         | 3.5       | 2.0        | 0.5      |
| 6                  | 7.0         | 4.5       | 3.5        | 2.0      |
| 8                  | 8.5         | 5.0       | 4.5        | 2.5      |
| 10                 | 9.5         | 5.5       | 4.8        | 2.8      |
| 12                 | 10.0        | 6.0       | 5.5        | 3.0      |
</details>

Figure 6: AUC drop due to watermarking for each MIA when varying the strength of the watermark.

Table 14: We show the AUC drop when we vary the percentage of green tokens between 30% and 70%. We bold the scenarios when a specific percentage value induces AUC drops for all the attacks. 

<table><tr><td colspan="2"></td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td></tr><tr><td rowspan="4">Llama-30B</td><td>PPL</td><td>0.71</td><td>-0.10</td><td>1.40</td><td>2.14</td><td>1.65</td></tr><tr><td>Lowercase</td><td>0.27</td><td>2.60</td><td>4.24</td><td>2.45</td><td>3.15</td></tr><tr><td>Zlib</td><td>0.58</td><td>-0.18</td><td>0.66</td><td>1.10</td><td>0.52</td></tr><tr><td>Min-K% Prob</td><td>1.38</td><td>0.03</td><td>1.28</td><td>1.64</td><td>1.45</td></tr><tr><td rowspan="4">NeoX-20B</td><td>PPL</td><td>5.56</td><td>5.65</td><td>6.64</td><td>6.92</td><td>4.96</td></tr><tr><td>Lowercase</td><td>9.84</td><td>11.12</td><td>12.79</td><td>11.94</td><td>9.78</td></tr><tr><td>Zlib</td><td>6.68</td><td>5.94</td><td>6.54</td><td>6.51</td><td>4.71</td></tr><tr><td>Min-K% Prob</td><td>0.45</td><td>0.88</td><td>1.83</td><td>1.26</td><td>3.84</td></tr><tr><td rowspan="4">Llama-13B</td><td>PPL</td><td>0.19</td><td>-0.86</td><td>1.22</td><td>1.84</td><td>1.39</td></tr><tr><td>Lowercase</td><td>0.49</td><td>2.45</td><td>3.93</td><td>1.54</td><td>1.65</td></tr><tr><td>Zlib</td><td>1.31</td><td>0.28</td><td>1.51</td><td>2.00</td><td>1.29</td></tr><tr><td>Min-K% Prob</td><td>2.81</td><td>1.21</td><td>2.45</td><td>2.70</td><td>2.78</td></tr><tr><td rowspan="4">Pythia-2.8B</td><td>PPL</td><td>4.65</td><td>4.49</td><td>3.39</td><td>4.42</td><td>3.66</td></tr><tr><td>Lowercase</td><td>4.91</td><td>6.23</td><td>4.18</td><td>5.33</td><td>7.22</td></tr><tr><td>Zlib</td><td>5.37</td><td>3.97</td><td>3.07</td><td>3.17</td><td>2.20</td></tr><tr><td>Min-K% Prob</td><td>1.10</td><td>1.21</td><td>1.44</td><td>3.42</td><td>4.71</td></tr><tr><td rowspan="4">OPT-2.7B</td><td>PPL</td><td>5.45</td><td>5.39</td><td>5.55</td><td>5.18</td><td>5.76</td></tr><tr><td>Lowercase</td><td>7.71</td><td>9.91</td><td>9.23</td><td>9.83</td><td>7.28</td></tr><tr><td>Zlib</td><td>3.35</td><td>4.12</td><td>4.57</td><td>4.17</td><td>4.11</td></tr><tr><td>Min-K% Prob</td><td>2.30</td><td>2.07</td><td>2.40</td><td>3.90</td><td>5.00</td></tr></table>

# Additional experiments on BookMIA

In this section, we conduct experiments using models finetuned on a subset of BookMIA, which we refer to as BookMIA-2. To build BookMIA-2, we first select only the samples that were not part of the training set of any model that we consider (labeled as 0 by (Shi et al. 2023)). Then, we randomly select half of them as finetuning data (referred to as seen samples) and keep the other half as unseen samples. Note that for BookMIA-2, there would not be a distribution difference between the seen and unseen samples (Das, Zhang, and Tramèr 2024). We also consider duplicating a sample from the training set of BookMIA-2 to have more fine-grained control over the memorization of that sample. We use Llama-7B in all the experiments from this section.

Verbatim Memorization. We study verbatim memorization on BookMIA-2 by measuring the relative increase in perplexity on the generation of the duplicated sample by the watermarked model compared to the original model, as well as the ratio between the probability of generating the duplicated sample by the original model to the watermarked model, which we refer to as probability reduction factor. We run each experiment with 20 seeds and report the average perplexity and the minimum probability reduction factor. We consider both the UMD and Unigram watermarking methods with several strengths (2, 5, and 10) and prompt the model with an empty string, as well as with the first 10, 20, and 100 words from the training sample. Additionally, we consider several duplication factors (the number of times one randomly chosen target sample appears in the dataset): 1, 10, 20, and 50. We show the results in Table 15. We observe that even in high memorization cases (duplication factor of 50), as long as the watermark is strong enough, the probability of generating the memorized sample decreases by almost 200 orders of magnitude, making it very unlikely to be generated.

Table 15: Average relative increase in perplexity and minimum probability reduction factor for generating the memorized target sample from BookMIA-2. Note that “S.”, “P.”, and “D.” stand for watermark method’s strength, prompt length, and duplication factor, respectively. 

<table><tr><td></td><td></td><td></td><td colspan="2">D = 1</td><td colspan="2">D = 10</td><td colspan="2">D = 20</td><td colspan="2">D = 50</td></tr><tr><td></td><td>S.</td><td>P.</td><td>PPL.</td><td>Prob.</td><td>PPL.</td><td>Prob.</td><td>PPL.</td><td>Prob.</td><td>PPL.</td><td>Prob.</td></tr><tr><td rowspan="12">UMD</td><td rowspan="4">2</td><td>0</td><td>0.32</td><td> $3.1 \times 10^{70}$ </td><td>0.28</td><td> $2.9 \times 10^{45}$ </td><td>0.17</td><td> $1.1 \times 10^{19}$ </td><td>0.007</td><td> $4.4 \times 10^0$ </td></tr><tr><td>10</td><td>0.32</td><td> $1.9 \times 10^{69}$ </td><td>0.28</td><td> $1.2 \times 10^{44}$ </td><td>0.17</td><td> $1.3 \times 10^{18}$ </td><td>0.005</td><td> $5.1 \times 10^0$ </td></tr><tr><td>20</td><td>0.32</td><td> $1.7 \times 10^{67}$ </td><td>0.28</td><td> $2.9 \times 10^{42}$ </td><td>0.17</td><td> $4.8 \times 10^{17}$ </td><td>0.005</td><td> $5.0 \times 10^0$ </td></tr><tr><td>100</td><td>0.32</td><td> $3.7 \times 10^{57}$ </td><td>0.28</td><td> $1.6 \times 10^{34}$ </td><td>0.16</td><td> $3.3 \times 10^{10}$ </td><td>0.005</td><td> $4.2 \times 10^0$ </td></tr><tr><td rowspan="4">5</td><td>0</td><td>2.93</td><td> $1.1 \times 10^{366}$ </td><td>2.58</td><td> $1.8 \times 10^{261}$ </td><td>1.53</td><td> $1.6 \times 10^{122}$ </td><td>0.10</td><td> $1.8 \times 10^{18}$ </td></tr><tr><td>10</td><td>2.92</td><td> $7.6 \times 10^{360}$ </td><td>2.56</td><td> $6.9 \times 10^{257}$ </td><td>1.51</td><td> $8.9 \times 10^{99}$ </td><td>0.09</td><td> $1.1 \times 10^{16}$ </td></tr><tr><td>20</td><td>2.91</td><td> $1.2 \times 10^{354}$ </td><td>2.55</td><td> $2.6 \times 10^{254}$ </td><td>1.50</td><td> $3.0 \times 10^{98}$ </td><td>0.09</td><td> $4.3 \times 10^{15}$ </td></tr><tr><td>100</td><td>2.89</td><td> $4.9 \times 10^{313}$ </td><td>2.53</td><td> $4.2 \times 10^{213}$ </td><td>1.45</td><td> $6.0 \times 10^{71}$ </td><td>0.09</td><td> $6.7 \times 10^{13}$ </td></tr><tr><td rowspan="4">10</td><td>0</td><td>38.6</td><td> $3.2 \times 10^{1028}$ </td><td>33.0</td><td> $6.7 \times 10^{883}$ </td><td>18.6</td><td> $1.9 \times 10^{508}$ </td><td>1.86</td><td> $7.1 \times 10^{245}$ </td></tr><tr><td>10</td><td>38.5</td><td> $1.1 \times 10^{1006}$ </td><td>32.8</td><td> $9.2 \times 10^{869}$ </td><td>18.4</td><td> $2.5 \times 10^{495}$ </td><td>1.78</td><td> $4.7 \times 10^{226}$ </td></tr><tr><td>20</td><td>38.4</td><td> $2.4 \times 10^{982}$ </td><td>32.7</td><td> $3.7 \times 10^{851}$ </td><td>18.2</td><td> $3.8 \times 10^{482}$ </td><td>1.76</td><td> $4.6 \times 10^{223}$ </td></tr><tr><td>100</td><td>38.0</td><td> $2.8 \times 10^{860}$ </td><td>32.3</td><td> $6.6 \times 10^{731}$ </td><td>17.7</td><td> $1.0 \times 10^{388}$ </td><td>1.70</td><td> $3.8 \times 10^{199}$ </td></tr><tr><td rowspan="12">Unigram</td><td rowspan="4">2</td><td>0</td><td>0.32</td><td> $5.2 \times 10^{63}$ </td><td>0.29</td><td> $6.8 \times 10^{59}$ </td><td>0.17</td><td> $1.2 \times 10^{14}$ </td><td>0.008</td><td> $2.9 \times 10^0$ </td></tr><tr><td>10</td><td>0.32</td><td> $4.7 \times 10^{62}$ </td><td>0.29</td><td> $3.6 \times 10^{58}$ </td><td>0.17</td><td> $4.8 \times 10^{14}$ </td><td>0.005</td><td> $5.9 \times 10^0$ </td></tr><tr><td>20</td><td>0.32</td><td> $7.9 \times 10^{61}$ </td><td>0.29</td><td> $9.8 \times 10^{56}$ </td><td>0.17</td><td> $3.1 \times 10^{14}$ </td><td>0.005</td><td> $5.8 \times 10^0$ </td></tr><tr><td>100</td><td>0.31</td><td> $1.2 \times 10^{50}$ </td><td>0.28</td><td> $5.2 \times 10^{46}$ </td><td>0.16</td><td> $2.7 \times 10^7$ </td><td>0.005</td><td> $5.1 \times 10^0$ </td></tr><tr><td rowspan="4">5</td><td>0</td><td>2.88</td><td> $3.4 \times 10^{304}$ </td><td>2.56</td><td> $1.2 \times 10^{290}$ </td><td>1.53</td><td> $1.1 \times 10^{79}$ </td><td>0.11</td><td> $2.2 \times 10^{22}$ </td></tr><tr><td>10</td><td>2.87</td><td> $1.6 \times 10^{300}$ </td><td>2.56</td><td> $1.9 \times 10^{286}$ </td><td>1.52</td><td> $8.6 \times 10^{77}$ </td><td>0.10</td><td> $5.9 \times 10^{16}$ </td></tr><tr><td>20</td><td>2.87</td><td> $8.2 \times 10^{295}$ </td><td>2.55</td><td> $3.1 \times 10^{282}$ </td><td>1.50</td><td> $6.2 \times 10^{76}$ </td><td>0.09</td><td> $4.6 \times 10^{16}$ </td></tr><tr><td>100</td><td>2.82</td><td> $2.5 \times 10^{261}$ </td><td>2.50</td><td> $6.6 \times 10^{239}$ </td><td>1.45</td><td> $1.0 \times 10^{47}$ </td><td>0.09</td><td> $4.1 \times 10^{14}$ </td></tr><tr><td rowspan="4">10</td><td>0</td><td>37.6</td><td> $2.4 \times 10^{834}$ </td><td>32.3</td><td> $2.2 \times 10^{811}$ </td><td>18.5</td><td> $1.8 \times 10^{425}$ </td><td>1.87</td><td> $2.6 \times 10^{268}$ </td></tr><tr><td>10</td><td>37.7</td><td> $3.5 \times 10^{824}$ </td><td>32.3</td><td> $7.1 \times 10^{799}$ </td><td>18.3</td><td> $1.3 \times 10^{419}$ </td><td>1.80</td><td> $5.7 \times 10^{251}$ </td></tr><tr><td>20</td><td>37.6</td><td> $4.9 \times 10^{812}$ </td><td>32.2</td><td> $2.3 \times 10^{788}$ </td><td>18.1</td><td> $6.1 \times 10^{405}$ </td><td>1.78</td><td> $1.3 \times 10^{248}$ </td></tr><tr><td>100</td><td>36.9</td><td> $2.4 \times 10^{707}$ </td><td>31.5</td><td> $9.5 \times 10^{684}$ </td><td>17.4</td><td> $1.5 \times 10^{323}$ </td><td>1.72</td><td> $1.0 \times 10^{212}$ </td></tr></table>

MIA. We also study the effect of the watermark on the effectiveness of MIAs for copyrighted training data detection (on BookMIA-2, without duplicated samples). We show the results in Table 16 and observe that watermarking negatively affects MIAs' success rate, which is consistent with our previous findings (from Section ). Finally, we also run our adaptive method from Section and observe an improvement of $0.9\%$ over Min-K% Prob.

![](images/09ddf53c4368a7833695ada5b442c758b021814546ec1387bb6e2ba841e5a105.jpg)

<details>
<summary>line</summary>

| Duplication factor | w/o watermark | w/ watermark (strength = 2) | w/ watermark (strength = 5) | w/ watermark (strength = 10) |
| ------------------ | ------------- | --------------------------- | --------------------------- | ---------------------------- |
| 0                  | 0.25          | 0.25                        | 0.25                        | 0.25                         |
| 10                 | 0.25          | 0.25                        | 0.25                        | 0.25                         |
| 20                 | 0.25          | 0.25                        | 0.25                        | 0.25                         |
| 50                 | 0.7           | 0.64                        | 0.28                        | 0.25                         |
</details>

![](images/7f408b359dd9d6e07a3a5474e93ee8fa5c38e4d378b193087480209818152efa.jpg)

<details>
<summary>line</summary>

| Duplication factor | w/o watermark | w/ watermark (strength = 2) | w/ watermark (strength = 5) | w/ watermark (strength = 10) |
| ------------------ | ------------- | --------------------------- | --------------------------- | ---------------------------- |
| 0                  | 0.25          | 0.25                        | 0.25                        | 0.25                         |
| 10                 | 0.25          | 0.25                        | 0.25                        | 0.25                         |
| 20                 | 0.25          | 0.25                        | 0.25                        | 0.25                         |
| 50                 | 0.7           | 0.6                         | 0.3                         | 0.25                         |
</details>

![](images/d8995e91f356497f32ba51f52ac446816f59319a59f39a739d270a2daf81f5d6.jpg)

<details>
<summary>line</summary>

| Duplication factor | w/o watermark | w/ watermark (strength = 2) | w/ watermark (strength = 5) | w/ watermark (strength = 10) |
| ------------------ | ------------- | --------------------------- | --------------------------- | ---------------------------- |
| 0                  | 0.03          | 0.02                        | 0.01                        | 0.01                         |
| 10                 | 0.04          | 0.03                        | 0.02                        | 0.01                         |
| 20                 | 0.07          | 0.06                        | 0.03                        | 0.02                         |
| 50                 | 0.65          | 0.58                        | 0.10                        | 0.03                         |
</details>

![](images/cf9b467d21ce02b5f394db97b48c67a23a6cc94dbf1726f89aac35ee71b15f4a.jpg)

<details>
<summary>line</summary>

| Duplication factor | w/o watermark | w/ watermark (strength = 2) | w/ watermark (strength = 5) | w/ watermark (strength = 10) |
| ------------------ | ------------- | --------------------------- | --------------------------- | ---------------------------- |
| 0                  | 0.03          | 0.02                        | 0.01                        | 0.01                         |
| 10                 | 0.03          | 0.03                        | 0.02                        | 0.01                         |
| 20                 | 0.08          | 0.07                        | 0.04                        | 0.01                         |
| 50                 | 0.65          | 0.52                        | 0.15                        | 0.03                         |
</details>

![](images/1a28ecd0f8e4df710bea4d2cb9dbd7e76c4e8e28a60466231aee3fb19d5d09b4.jpg)

<details>
<summary>line</summary>

| Duplication factor | w/o watermark | w/ watermark (strength = 2) | w/ watermark (strength = 5) | w/ watermark (strength = 10) |
| ------------------ | ------------- | --------------------------- | --------------------------- | ---------------------------- |
| 0                  | 0.03          | 0.03                        | 0.02                        | 0.01                         |
| 10                 | 0.04          | 0.04                        | 0.03                        | 0.02                         |
| 20                 | 0.08          | 0.07                        | 0.05                        | 0.03                         |
| 50                 | 0.65          | 0.58                        | 0.11                        | 0.03                         |
</details>

![](images/12db215d32ff0bc99099ad0b5f5ccbf992215d219ec3d291b8262d70f09ee22a.jpg)

<details>
<summary>line</summary>

| Duplication factor | w/o watermark | w/ watermark (strength = 2) | w/ watermark (strength = 5) | w/ watermark (strength = 10) |
| ------------------ | ------------- | --------------------------- | --------------------------- | ---------------------------- |
| 0                  | 0.03          | 0.02                        | 0.01                        | 0.01                         |
| 10                 | 0.03          | 0.03                        | 0.02                        | 0.01                         |
| 20                 | 0.08          | 0.07                        | 0.04                        | 0.01                         |
| 50                 | 0.65          | 0.52                        | 0.16                        | 0.03                         |
</details>

Figure 7: Edit similarity (top), word-level BLEU score (middle), and token-level BLEU score (bottom) between the generated completion and the ground truth when considering different watermark strengths on BookMIA-2.

# Theoretical analysis

Notations and assumptions. We assume that the set of all copyrighted texts $C_D$ that were part of the training data has $m$ elements $\{s_1, s_2, ..., s_m\}$ . Also, we assume that each copyrighted text has a fixed length $n$ , and they are independent from each other.

Theorem 1 For an LLM watermarked using a “hard” UMD scheme with a percentage of $\gamma$ green tokens, then the probability of generating a copyrighted text is lower than $m \cdot \gamma^{n}$ .

Proof. Given one sample $s = t_{1} \oplus t_{2} \oplus \ldots \oplus t_{n} \in C_{D}$ . For a “hard” watermarking scheme, the probability $P(s)$ of generating s is smaller than the probability of each token $t_{i}$ to be on a green list. So, $P(s) < \gamma^{n}$ . The probability of not generating any $s_{j} \in C_{D}$ is $P(\neg s_{1} \land \neg s_{2} \land \ldots \land \neg s_{m}) = \prod_{i=\overline{1,m}}(1 - P(s_{i})) > (1 - \gamma^{n})^{m} > 1 - m\gamma^{n}$ . Note that we used Bernoulli’s inequality at the end. So, the probability of generating at least one copyrighted text is lower than $1 - (1 - m \cdot \gamma^{n})$ and hence lower than $m \cdot \gamma^{n}$ .

Table 16: AUC of each MIA for the unwatermarked (top of each cell), watermarked models (middle of each cell), and the drop between the two (bottom of each cell) on BookMIA-2 (without any duplicated samples) using the UMD scheme with a strength of 10. We average the results over 5 runs with different seeds. 

<table><tr><td></td><td>Llama-7B (fine-tuned)</td></tr><tr><td rowspan="3">PPL</td><td>58.5 ± 0.0%</td></tr><tr><td>56.6 ± 0.0%</td></tr><tr><td>1.9%</td></tr><tr><td rowspan="3">Lowercase</td><td>59.8 ± 0.1%</td></tr><tr><td>52.9 ± 0.3%</td></tr><tr><td>6.9%</td></tr><tr><td rowspan="3">Zlib</td><td>59.7 ± 0.0%</td></tr><tr><td>56.1 ± 0.1%</td></tr><tr><td>3.6%</td></tr><tr><td rowspan="3">Min-K% Prob</td><td>58.5 ± 0.0%</td></tr><tr><td>57.1 ± 0.2%</td></tr><tr><td>1.4%</td></tr></table>

Example. Let's consider a "hard" UMD watermarking scheme with $\gamma = 0.5$ . Let's assume each copyrighted text is 100 tokens, the model was trained on a dataset containing $10^{9}$ copyrighted texts. The probability to generate a copyrighted text is $< 10^{9} \cdot 0.5^{100} = \frac{10^{9}}{2^{100}} = \frac{1000^{3}}{1024^{10}} < \frac{1000^{3}}{1000^{10}} = 1000^{-7} = 10^{-21}$ and hence very low.

Theorem 2 Let f be a LLM and $f_{W}$ its watermarked version with a “soft” UMD scheme and let $\epsilon \in (0, \frac{1}{4})$ . Let $s = t_{1} \oplus t_{2} \oplus \ldots \oplus t_{n} \in C_{D}$ be a copyrighted sample. We consider $\gamma = 0.5$ . We denote the output of the softmax layer of f for generating the token $t_{i}$ as $\frac{a_{i}}{d_{i} + a_{i}}$ and in the case of $f_{W}$ , we denote it by $\frac{a_{i} \cdot e^{\delta}}{b_{i}^{\prime} + c_{i}^{\prime} \cdot e^{\delta} + a_{i} \cdot e^{\delta}}$ (if $t_{i}$ is on the green list) and $\frac{a_{i}}{b_{i}^{\prime\prime} + c_{i}^{\prime\prime} \cdot e^{\delta} + a_{i}}$ (if $t_{i}$ is on the red list), where $a_{i}$ is the exponential of the logit value corresponding to the token $t_{i}$ and $b_{i}^{\prime}, b_{i}^{\prime\prime}$ and $c_{i}^{\prime}, c_{i}^{\prime\prime}$ are the sum of the exponentials of the logits corresponding to other tokens that are on the red list and green list, respectively. We assume that $\frac{x}{a_{i}} < M = \frac{1 - 4\epsilon}{1 + 4\epsilon}$ , for all $x \in \{d_{i}, b_{i}^{\prime}, b_{i}^{\prime\prime}, c_{i}^{\prime}, c_{i}^{\prime\prime}\}$ which would restrict f to be relatively confident in its predictions for each token $t_{i}$ . Then, we can always find a $\delta$ (strength) for the watermarking scheme such that the probability of generating s is reduced by at least $(1 + \frac{2\epsilon}{2\epsilon + 1})^{n}$ times in comparison to the case of the unwatermarked model.

Proof. First, we observe that the probability of generating the token $t_{i}$ by the unwatermarked model is $\frac{a_{i}}{d_{i}+a_{i}} = \frac{1}{\frac{d_{i}}{a_{i}}+1} > \frac{1}{M+1} = 1/2 + 2\epsilon$ .

We observe that since there is a finite number of $\frac{x}{a_i}'s$ and they are all positive, then it exist a lower bound for $\frac{x}{a_i}$ (let's denote it by $m > 0$ ). Since $\gamma = 0.5$ , the probability of $t_i$ being a green token is $\frac{1}{2}$ and hence the probability of the watermarked model to generate $t_i$ is $\frac{1}{2} \frac{a_i \cdot e^\delta}{b_i' + c_i' \cdot e^\delta + a_i \cdot e^\delta} + \frac{1}{2} \frac{a_i}{b_i'' + c_i'' \cdot e^\delta + a_i} < \frac{1}{2} + \frac{1}{2} \frac{a_i}{b_i'' + c_i'' \cdot e^\delta + a_i} = \frac{1}{2} + \frac{1}{2} \frac{1}{\frac{b_i''}{a_i} + \frac{c_i''}{a_i} \cdot e^\delta + 1} \leq \frac{1}{2} + \frac{1}{2} \frac{1}{m \cdot (e^\delta + 1) + 1}$ . We pick $\delta > \log \left( \frac{1 - 2\epsilon(m + 1)}{2\epsilon m} \right)$ and we observe that $\frac{1}{2} + \frac{1}{2} \frac{1}{m \cdot (e^\delta + 1) + 1} < \frac{1}{2} + \frac{1}{2} \frac{1}{m \cdot (\frac{1 - 2\epsilon(m + 1)}{2\epsilon m} + 1) + 1} = \frac{1}{2} + \frac{1}{2} \frac{1}{m \cdot (\frac{1 - 2\epsilon}{2\epsilon m}) + 1} = \frac{1}{2} + \epsilon$ .

So, by combining the two observations above, we conclude that the probability of generating $t_{i}$ is reduced by at least $\frac{\frac{1}{2}+2\epsilon}{\frac{1}{2}+\epsilon}=1+\frac{2\epsilon}{2\epsilon+1}$ times. Therefore, since there are n tokens in s, the probability of generating s is reduced by at least $(1+\frac{2\epsilon}{2\epsilon+1})^{n}$ times.

Observation. Since the probability is reduced by at least $(1 + \frac{2\epsilon}{2\epsilon+1})^{n}$ times in Theorem 2 then the probability of generating s is lower than $(\frac{2\epsilon+1}{4\epsilon+1})^{n}$ (as the maximum probability of generating with the unwatermarked model is 1). Hence, as in Theorem 1, we observe that the probability of generating a copyrighted text is lower than $m \cdot (\frac{2\epsilon+1}{4\epsilon+1})^{n}$ .

Takeaways. Our theoretical analysis demonstrates that watermarking significantly reduces the probability of generating copyrighted text verbatim. For both a “hard” and “soft” UMD scheme, the upper bound for the likelihood of producing copyrighted content decreases exponentially with the length of the copyrighted texts.

# Metrics used in MIAs

In this section we provide short formulas for several metrics used by MIAs such as Smaller Ref, Lowercase and Zlib (Carlini et al. 2021).

Let $f$ be the model the MIA is applied to, $g$ be a smaller model, $x$ be a sample (string), log be the natural logarithm function, perplexity\_h(x) be a function that computes the perplexity of a model $h$ on $x$ , lowercase be a function that maps a string to its lowercased version, and zlib(x) be a function that computes the zlib entropy of a string $x$ .

Smaller Ref uses the metric $\frac{\log(\text{perplexity\_f}(x))}{\log(\text{perplexity\_g}(x))}$ .

Lowercase uses the metric $\frac{\log(\text{perplexity\_f}(x))}{\log(\text{perplexity\_f}(\text{lowercase}(x)))}$ .

Zlib uses the metric $\frac{\log(\text{perplexity\_f}(x))}{\text{zlib}(x)}$ .

# Adaptive Min-K% Prob

Table 17: We show the AUC of Min-%K Prob (referred as “Not adapt.”) and our method (referred as “Adapt.”) when using UMD watermarking scheme. We highlight the cases when our method improves over the baseline. 

<table><tr><td></td><td></td><td>Llama-30B</td><td>NeoX-20B</td><td>Llama-13B</td><td>Pythia-2.8B</td><td>OPT-2.7B</td></tr><tr><td rowspan="2">WikiMIA 32</td><td>Not adapt.</td><td>66.2%</td><td>67.1%</td><td>64.5%</td><td>61.0%</td><td>55.7%</td></tr><tr><td>Adapt.</td><td>68.5%</td><td>71.3%</td><td>66.3%</td><td>61.0%</td><td>59.1%</td></tr><tr><td rowspan="2">WikiMIA 64</td><td>Not adapt.</td><td>64.4%</td><td>67.7%</td><td>62.8%</td><td>59.8%</td><td>55.3%</td></tr><tr><td>Adapt.</td><td>67.3%</td><td>72.0%</td><td>64.9%</td><td>60.6%</td><td>57.4%</td></tr><tr><td rowspan="2">WikiMIA 128</td><td>Not adapt.</td><td>70.0%</td><td>73.0%</td><td>68.9%</td><td>64.8%</td><td>59.2%</td></tr><tr><td>Adapt.</td><td>73.1%</td><td>75.9%</td><td>71.0%</td><td>66.4%</td><td>64.0%</td></tr><tr><td rowspan="2">WikiMIA 256</td><td>Not adapt.</td><td>70.5%</td><td>76.2%</td><td>70.4%</td><td>69.5%</td><td>63.1%</td></tr><tr><td>Adapt.</td><td>71.3%</td><td>78.2%</td><td>72.4%</td><td>70.7%</td><td>66.2%</td></tr></table>

# Adaptive Zlib and Adaptive Lowercase

We have also adapted Zlib, as well as Lowercase. In these adaptations, we use a method similar to Adaptive Min-K% Prob to approximate pre-watermarked probabilities, then we apply Zlib / Lowercase. The results are presented in Tables 18 and 19, where each double-cell shows the AUC score for the non-adaptive attack on top and the adaptive version on the bottom. We observe that adaptation improves the baseline in over 80% of cases (for Zlib) and over 90% of cases (for Lowercase). However, these adaptive methods appear slightly less effective compared to Min-K% Prob. We hypothesize that this may be due to them using all token probabilities rather than the minimum K% subset, as in Min-K% Prob, which may increase error accumulation due to our token probability approximation.

Table 18: We show the AUC of Zlib (referred as “Not adapt.”) and our method (referred as “Adapt.”) when using UMD watermarking scheme. We highlight the cases when our method improves over the baseline. 

<table><tr><td></td><td></td><td>Llama-30B</td><td>NeoX-20B</td><td>Llama-13B</td></tr><tr><td rowspan="2">WikiMIA 32</td><td>Not adapt.</td><td>64.4%</td><td>63.2%</td><td>62.3%</td></tr><tr><td>Adapt.</td><td>66.8%</td><td>64.2%</td><td>64.9%</td></tr><tr><td rowspan="2">WikiMIA 64</td><td>Not adapt.</td><td>62.4%</td><td>62.0%</td><td>59.9%</td></tr><tr><td>Adapt.</td><td>66.8%</td><td>63.6%</td><td>64.8%</td></tr><tr><td rowspan="2">WikiMIA 128</td><td>Not adapt.</td><td>68.6%</td><td>66.3%</td><td>65.8%</td></tr><tr><td>Adapt.</td><td>71.4%</td><td>67.3%</td><td>69.1%</td></tr><tr><td rowspan="2">WikiMIA 256</td><td>Not adapt.</td><td>72.0%</td><td>66.6%</td><td>71.6%</td></tr><tr><td>Adapt.</td><td>71.3%</td><td>67.1%</td><td>71.1%</td></tr></table>

Table 19: We show the AUC of Lowercase (referred as “Not adapt.”) and our method (referred as “Adapt.”) when using UMD watermarking scheme. We highlight the cases when our method improves over the baseline. 

<table><tr><td></td><td></td><td>Llama-30B</td><td>NeoX-20B</td><td>Llama-13B</td></tr><tr><td rowspan="2">WikiMIA 32</td><td>Not adapt.</td><td>54.9%</td><td>59.4%</td><td>54.2%</td></tr><tr><td>Adapt.</td><td>57.4%</td><td>62.0%</td><td>56.9%</td></tr><tr><td rowspan="2">WikiMIA 64</td><td>Not adapt.</td><td>54.8%</td><td>56.8%</td><td>53.8%</td></tr><tr><td>Adapt.</td><td>56.7%</td><td>59.4%</td><td>56.0%</td></tr><tr><td rowspan="2">WikiMIA 128</td><td>Not adapt.</td><td>55.9%</td><td>58.2%</td><td>55.1%</td></tr><tr><td>Adapt.</td><td>57.2%</td><td>59.4%</td><td>56.0%</td></tr><tr><td rowspan="2">WikiMIA 256</td><td>Not adapt.</td><td>63.8%</td><td>55.4%</td><td>61.6%</td></tr><tr><td>Adapt.</td><td>63.5%</td><td>58.1%</td><td>62.3%</td></tr></table>

# Examples of generated samples

Below we provide text samples generated from the unwatermarked model and from models with various UMD watermark strengths. All examples are generated from an empty string (free generation). These examples have relative perplexity increases that match or even exceed the corresponding averages in Figure 2.

w/o watermark: “Nonviolent communication, also called compassionate communication, is a way of relating to others based on values of human dignity, honesty, and empathy ...”

w/ watermark (strength = 2): “The Latifa Hospital was established in Dubai, United Arab Emirates in March 1983 to provide advanced health care within Dubai ...”

w/ watermark (strength = 4): “We all make New Year’s Resolutions and all of them start with “getting fitter” :) If it happens to be in your list, let me know ...”

w/ watermark (strength = 6): “I’ll be the first to say that MTS doesn’t always achieve what it sets out to do at first look but you can see the software is reaching its potential ...”

w/ watermark (strength = 8): “United Nations : Member states of UNO climate change conference agreed on Thursday that progress had to accelerate in implementing ...”

w/ watermark (strength = 10): “Like our Facebook page to see when the weekend is set and to find out about local motorcycling news, events and ...”

w/ watermark (strength = 12): “Your child will learn time management as well: it will learn how much time to spend doing each assignment and studying for tests to ensure they get them done ...”

w/ watermark (strength = 14): “Grant is a shy four to six year old beagle boy with one blind eye and lacking much vision in both eyes. His family had been unable to ...”

w/ watermark (strength = 16): “We are leading online doctor shopping solution company in Noida. It offers the best web solutions with goal orientated expertise in medical sector website development and marketing ...”

# Extended Related Work

Memorization. One cause of copyright issues is that machine learning models may memorize training data. Prior studies have observed that LLMs can memorize private information in training data, such as phone numbers and addresses (Karamolegkou et al. 2023; Carlini et al. 2019, 2021; Lee et al. 2021), leading to significant privacy and security concerns. To measure memorization, Carlini et al. (2021) proposes eidetic memorization, defining a string as memorized if it was present in the training data and it can be reproduced by a prompt. This definition, along with variations like exact and perfect memorization, has been widely adopted in subsequent studies (Tirumala et al. 2022; Kandpal, Wallace, and Raffel 2022). Carlini et al. (2022b) quantitatively measures memorization in LLMs as the fraction of extractable training data and finds that memorization significantly grows as model size scales and training examples are duplicated. To minimize memorization, Lee et al. (2021) and Kandpal, Wallace, and Raffel (2022) propose deduplicating training data, which also improves accuracy. Hans et al.

(2024) proposes the Goldfish Loss as a training-time defense against verbatim memorization. Ippolito et al. (2023) proposes an inference time defense that perfectly prevents all verbatim memorization. However, it cannot prevent the leakage of training data due to the existence of many “style-transfer” prompts, suggesting it is a challenging open problem. Unlike the methods that we are studying in this paper, Ippolito et al. (2023) requires access to a complete set of copyrighted texts that the model was trained on. Memorization in the image domain has also been studied from various angles (Somepalli et al. 2023a,b; Carlini et al. 2023; Wen et al. 2024).

Membership Inference. As a proxy for measuring memorization, membership inference attacks (MIAs) predict whether or not a particular example was used to train the model (Shokri et al. 2017; Yeom et al. 2018; Bentley et al. 2020). Most membership inference attacks rely only on the model's loss since the model is more likely to overfit an example if it is in the training data (Sablayrolles et al. 2019). Carlini et al. (2022a) trains shadow models to predict whether an example is from the training data. In the NLP domain, many works have focused on masked language models (Mireshghallah et al. 2022) and fine-tuning data detection (Song and Shmatikov 2019; Shejwalkar et al. 2021). Recently, Shi et al. (2023) studies pretraining data inference and introduced a detection method based on the hypothesis that unseen examples are likely to contain outlier words with low probabilities under the LLM. Zhang et al. (2024) approaches pretraining data detection by measuring how sharply peaked the likelihood is around the inputs. Duarte et al. (2024) proposes detecting copyrighted content in training data by probing the LLM with multiple-choice questions, whose options include both verbatim text and their paraphrases. Other methods include testing perplexity differences (Mattern et al. 2023) and providing provable guarantees of test set contamination without access to pretraining data or model weights (Oren et al. 2023).

# Computing Infrastructure

All of our experiments were run on either three Nvidia RTX A6000 or four Nvidia RTX A5000 GPUs, using 128 GB of memory. We used the Transformers library (version 4.35.2) and PyTorch (version 2.1.0).

# Limitations & Discussion

We emphasize that our claim is not that any watermarking method inherently prevents the generation of verbatim copyrighted text. Instead, we demonstrate that popular watermarking methods can unintentionally produce this effect. We believe that a distortion-free watermark, such as the one proposed by Kuditipudi et al. (2023), would not have this unintended effect. Our proposed method for improving MIAs' success rate on watermarked models makes strong assumptions on the watermarking scheme, which may not always be satisfied despite empirical improvements in our experiments. Our observations on the deterioration of MIAs' success suggests that for copyright violation auditing, an unwatermarked model or the watermarking scheme may be needed. We encourage the community to further refine adaptive methods to ensure robust copyright protection and data privacy, and consider the interactions of different methods on downstream legal concerns.