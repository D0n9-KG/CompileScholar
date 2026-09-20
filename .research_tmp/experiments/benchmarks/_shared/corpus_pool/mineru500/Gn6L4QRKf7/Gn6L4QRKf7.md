# On the Power of Context-Enhanced Learning in LLMs

Xingyu Zhu $^{1*}$ Abhishek Panigrahi $^{1*}$ Sanjeev Arora $^{1}$

# Abstract

We formalize a new concept for LLMs, context-enhanced learning. It involves standard gradient-based learning on text except that the context is enhanced with additional data on which no autoregressive gradients are computed. This setting is a gradient-based analog of usual in-context learning (ICL) and appears in some recent works.

Using a multi-step reasoning task, we prove in a simplified setting that context-enhanced learning can be exponentially more sample-efficient than standard learning when the model is capable of ICL. At a mechanistic level, we find that the benefit of context-enhancement arises from a more accurate gradient learning signal. We also experimentally demonstrate that it appears hard to detect or recover learning materials that were used in the context during training. This may have implications for data security as well as copyright.

# 1. Introduction

Pre-trained LLMs (Brown et al., 2020; Touvron et al., 2023; Team et al., 2023) show strong capability to learn new material at inference time, for instance via in-context-learning (ICL). There is also emerging evidence that gradient-based learning on a piece of text (say, math Q&A) can be enhanced if additional helpful text is placed in the context, even though no auto-regressive loss is computed on this helpful text (Liao et al., 2024; Zou et al., 2024; Choi et al., 2025). Such strategies have also been shown to benefit pre-training as prepending source URLs to documents can enhance the model's training efficiency and memorization capacity (Allen-Zhu & Li, 2024; Gao et al., 2025).

In this paper we seek to formally study this phenomenon, whereby LLMs' gradient-based learning is enhanced via \* Equal contribution with theoretical framework and setting up the problem, XZ undertook most of the experiments. 1Princeton Language and Intelligence, Princeton University. Correspondence to: <{xingyu.zhu, ap34}@princeton.edu>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

placement of additional helpful material in the context, but without actual auto-regressive gradient updates on this material. We will call this form of learning context-enhanced learning. Because the material used for context-enhancement can evolve over the course of training, this approach naturally aligns with the idea of a curriculum.

Context-enhanced learning intuitively mirrors how humans learn: when solving problems, they refer to textbooks or demonstrations for guidance, yet they do not seek to memorize these resources per se. An analogous concept Learning using Privileged Information (or LUPI), has been well-studied in the context of kernel SVMs (Vapnik & Vashist, 2009) and classification models. Our work adapts this concept for LLMs, and surfaces the following questions:

Q1: Even though autoregressive loss is computed on the same set of tokens, can context-enhanced learning be significantly more powerful than usual auto-regressive learning that has no additional in-context materials? If so, can we theoretically characterize and understand the mechanism behind such improvement?

Q2: Do models need a certain capability level to benefit from context-enhanced learning? This is a natural question, since leveraging in-context information (e.g., ICL) likely requires a minimum capability level or model size (Brown et al., 2020; Wei et al., 2022).

Q3: Is context-enhanced learning a viable way to use privileged/private information during learning? Providing such privileged information in the context could conceivably enhance the model's learning, but since no auto-regressive gradient updates happen on the privileged/private information, there might be a lower risk of leakage of such information via API calls.

Paper Overview: Section 2.1 formally defines context-enhanced learning. To allow rigorous understanding of the power of context-enhanced learning, Section 2.2 introduces a multi-step reasoning task called Multi-layer translation. This is a synthetic setting involving $d + 1$ languages $L_{1}, L_{2}, \ldots, L_{d+1}$ over finite alphabets. For each i, there is a simple phrasebook that describes how to translate from $L_{i}$ to $L_{i+1}$ , and the mapping from $L_{1}$ to $L_{d+1}$ is a sequential application of the set of phrasebooks.

The goal is to learn how to translate text from $L_{1}$ into $L_{d + 1}$

without explicitly writing down intermediate steps. The learner is provided excerpts from these phrasebooks as helpful information in the context during training, but allowed no auto-regressive gradient updates on these tokens.

If we train with auto-regressive loss on translation output conditioning on the phrasebooks' excerpts and the input, a model with certain ICL capacity level may quickly learn the translation task by leveraging the in-context phrasebooks. However, this learning could be brittle, that the model becomes reliant on having the phrasebooks' excerpts in context. This reliance can be weaned off by use of probabilistic dropout on phrasebooks tokens in context. Intuitively, this curriculum forces the model to not only read phrasebooks' excerpts, but also gradually internalize the phrasebooks' contents. Over time, the model's ability to translate from $L_{1}$ to $L_{d+1}$ will become robust to the dropout of phrasebooks' excerpts, and eventually, their complete removal.

Experiments show that this training strategy indeed works when the learner is a pre-trained LLM that is capable of ICL (but fails when LLM is incapable of ICL). Even when training with 20% dropout rate, the model can perfectly translate strings from $L_{1}$ to $L_{d+1}$ without any phrasebooks' excerpts at test time. The rest of the paper is structured as follows:

\- Section 3 details our experiments and the findings sketched above. Experiments show that an ICL-capable model follows an intuitive sequential processing of the phrasebooks provided in-context, whereby transformer layers approach stages of translation in an intuitive way; e.g., $L_{3} \to L_{4}$ is done after $L_{2} \to L_{3}$ (Section 3.3).

\- Section 4, shows that after context-enhanced learning, the output probabilities of the model reveal little about the phrasebooks rules that were seen during training.

\- In Section 5, we propose a theoretical framework using a surrogate/simplified model that represents an ideal LLM for the translation task (Section 5.1). This framework shows an exponential gap in sample complexity depending on whether the model is trained with or without in-context information on phrasebooks (Sections 5.2 and 5.3). Experiments reveal that the mechanism behind the increased sample efficiency of context-enhanced learning is an improved gradient signal, measured by gradient prediction accuracy (Section 5.4).

# 2. Setup

# 2.1. Context-Enhanced Learning

Let $X$ be the space of all possible text strings and let $\mathcal{Y}$ be the space of all possible distributions over texts. Let $g$ be a language task mapping inputs $x \in X_g \subset X$ to a distribution $Y \in \mathcal{Y}$ . Let $f_{\theta}: X \to \mathcal{Y}$ be a general autoregressive language model. We characterize $f_{\theta}$ 's capability on task $g$ as follows:

Definition 2.1 (g-capable model, informal). A language model $f_{\theta}$ is g-capable for a language task g if $f_{\theta}$ is close to g, as measured by a suitable metric on $X_{g}$ .

Vanilla supervised fine-tuning (SFT) aims to create a g-capable model by minimizing auto-regressive loss $\ell_{auto}$ on a supervised dataset $D_{g} = \{(x_{i}, y_{i})\}_{i=1}^{N}$ , where the label $y_{i}$ for each $x_{i} \in X_{g}$ is sampled from $g(x_{i})$ .

Context-enhanced learning involves augmenting the supervision with additional curriculum-text that depends on the task g, input x, and training step t. We denote curriculum-text as $\text{CURR}_{g}(x,t)$ , which could be anything (helpful explanations, excerpts from textbooks, worked-out examples, etc.).

# Algorithm 1 Context-Enhanced Learning

In contrast to standard SFT, it relies on curriculum-text in context on which no auto-regressive loss is computed.

Input: Supervised dataset $D_{g}$ , curriculum-text $CURR_{g}$ , initialization $\theta$ , total steps T

for t = 1 to T do
    Sample $(x, y) \sim D_{g}$ Compute loss $l \leftarrow \ell_{\text{auto}} \left( f_{\theta}([CURR_{g}(x, t), x, y]), y \right)$ .
    Update parameters $\theta$ with gradient $\nabla_{\theta}l$ end for
Return $\theta$

On sample $(x,y)$ drawn from the supervised dataset, we use auto-regressive loss for model's prediction on y conditioned on $[CURR_{g}(x,t),x]$ to train our models. Note that no loss is computed for curriculum-text tokens. We denote this loss as $\ell_{\mathrm{auto}}(f_{\theta}([CURR_{g}(x,t),x,y]),y)$ .

# 2.2. Multi-level Translation (MLT)

To study the power of context-enhanced learning, we introduce a multi-step translation task that is easy to learn with a straightforward curriculum in the context, but very difficult to learn with just input-output examples.

The task is inspired by encryption methods $^{1}$ such as the Feistel cipher (Knudsen, 1993). The multi-level translation (MLT) task involves a bijective mapping from strings to strings that is a composition of 2d simpler bijections, each involving simple shift by 1, or transforming bigrams (2-tuples of characters) via a bijection. The depth-d translation can be described by $O(d)$ bits. But we will show that learning the task only from input-output pairs would require $e^{\Omega(d)}$ sample complexity in the SQ-learning framework (Kearns, 1998) (see Theorem 5.4).

Table 1. Important notations for defining MLT 

<table><tr><td>d</td><td>Depth of translation task</td></tr><tr><td>n</td><td>Number of characters in each alphabet</td></tr><tr><td>A</td><td>An alphabet set</td></tr><tr><td>π</td><td>A phrasebook between 2-tuples in two alphabets</td></tr><tr><td>Π</td><td>A set of phrasebooks {πi} defining a translation task</td></tr><tr><td>MLTπ</td><td>Translation task with a set of phrasebooks Π</td></tr><tr><td>MLT(d,n)</td><td>Family of translation tasks of depth d and n characters</td></tr></table>

Concretely, let $A_{1},\ldots,A_{d+1}$ be $d+1$ alphabets all of the same size with n characters. For every consecutive pair of alphabets $A_{i}$ and $A_{i+1}$ , we fix a phrasebook $\pi_{i}:A_{i}^{2}\to A_{i+1}^{2}$ as a bijective mapping from 2-tuples in $A_{i}$ to 2-tuples in $A_{i+1}$ . Each phrasebook $\pi_{i}$ can be represented by a binary stochastic matrix $\text{Matrix}(\pi_{i})$ with rules represented as one-hot columns (see Definition G.4).

![](images/a5f437637953a08f5a8e9d80f4ead0f6f21a0bb5cc7e3788f43db8cb0592a5ab.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph s1
        A1["A"] --> B1["B"]
        B1 --> C1["C"]
        C1 --> D1["D"]
        D1 --> E1["E"]
        E1 --> F1["F"]
        F1 --> G1["G"]
        G1 --> H1["H"]
    end
    subgraph s2
        S1["B"] --> T1["T"]
        T1 --> U1["π1"]
        U1 --> V1["π1"]
        V1 --> W1["π1"]
        W1 --> X1["π1"]
        X1 --> Y1["g"]
        Y1 --> Z1["c"]
    end
    subgraph s3
        S2["b"] --> T2["T"]
        T2 --> U2["π2"]
        U2 --> V2["π2"]
        V2 --> W2["π2"]
        W2 --> X2["π2"]
        X2 --> Y2["π2"]
        Y2 --> Z2["ζ"]
        Z2 --> AA["μ"]
    end
    style s1 fill:#f9f,stroke:#333
    style s2 fill:#bbf,stroke:#333
    style s3 fill:#bfb,stroke:#333
    style T1 fill:#ffb,stroke:#333
    style T2 fill:#ffb,stroke:#333
    style U1 fill:#ffb,stroke:#333
    style U2 fill:#ffb,stroke:#333
    style V1 fill:#ffb,stroke:#333
    style V2 fill:#ffb,stroke:#333
    style W1 fill:#ffb,stroke:#333
    style W2 fill:#ffb,stroke:#333
    style X1 fill:#ffb,stroke:#333
    style X2 fill:#ffb,stroke:#333
    style Y1 fill:#ffb,stroke:#333
    style Y2 fill:#ffb,stroke:#333
    style Z1 fill:#ffb,stroke:#333
    style AA fill:#ffb,stroke:#333
```
</details>

Figure 1. Illustration of an MLT(2,8) instance with input sequence of 8 tokens. Input sequence $s_1$ went through 2 translation steps to $s_3$ . Each output character depends on 4 input characters, for example, character $\mu$ is derived from $c, e$ in $s_2$ , which, in turn, are computed from A, B, C, H in $s_1$ .

The input of the translation process is an even-length sequence, referred to as $s_{1} \in A_{1}^{L}$ where L is the sequence length. The translation process modifies $s_{1}$ recursively. For every $i \in [d]$ , $s_{i} \in A_{i}^{L}$ will be transformed to $s_{i+1} \in A_{i+1}^{L}$ using phrasebook $\pi_{i}$ through the following 2 sub-processes:

- Circular shift: The characters in $\boldsymbol{s}_i \in A_i^L$ are shifted by 1 character leftward (and wrapped around to the end if necessary) to give sequence $\tilde{\boldsymbol{s}}_i \in A_i^L$ . Formally, for each $j \in [1, L]$ we have $\tilde{\boldsymbol{s}}_{i,j} = \boldsymbol{s}_{i,(j+1)\%L}$ .   
- Translate: Using the phrasebook $\pi_i: A_i^2 \to A_{i+1}^2$ , we translate 2-tuples (bigrams) of consecutive characters in sequence $\tilde{s}_i$ to create $s_{i+1}$ . That is, for every odd $j \in [1, L]$ , $(s_{i+1,j}, s_{i+1,j+1}) = \pi_i(\tilde{s}_{i,j}, \tilde{s}_{i,j+1})$ .

We denote the mapping from $s_{i}$ to $s_{i+1}$ as $s_{i+1} = T_{\pi_{i}}(s_{i})$ . The d-step translation is defined as the composition $s_{d+1} = T_{\pi_{d}} \circ T_{\pi_{d-1}} \circ \cdots \circ T_{\pi_{1}}(s_{1})$ which converts the input sequence $s_{1}$ to $s_{d+1}$ through d translation steps. Please see Figure 1 for a visual illustration with d = 2, n = 8.

We denote $\Pi=\{\pi_{i}\}_{i=1}^{d}$ the set of phrasebooks of all levels, and denote $\mathbf{MLT}_{\Pi}:s_{1}\mapsto\mathbf{MLT}_{\Pi}(s_{1}):=T_{\pi_{d}}\circ\cdots\circ T_{\pi_{1}}(s_{1})$ the mapping from input $s_{1}$ to output $s_{d+1}$ using $\Pi$ . We use $\mathbf{MLT}(d,n)$ to refer to the family of translation tasks involving d step and n characters in each alphabet.

We note that $\mathbf{MLT}(d,n)$ has two key properties: 1. Once the phrasebooks are fixed, the translation task defines a bijection between input and output strings since Circular shift and Translate are invertible (see Lemma E.1). 2. Each character in the output string depends on 2d characters from the input text string (see caption of Figure 1), making learning from input-output pairs very difficult (Theorem 5.4).

For each phrasebook $\pi_{i}$ , we compose its textual representation $\mathrm{STR}(\pi_i)$ to be of the form ... a b -> C D; e d -> B A; ... which lists (insensitive of ordering) phrasebook rules between 2-tuples in the previous alphabet and the next alphabet. Moreover, we denote the concatenation $[\mathrm{STR}(\pi_1),\ldots ,\mathrm{STR}(\pi_d)]$ as $\mathrm{STR}(\Pi)$ , and will be used to define curriculum-text.

# 2.3. Needed: Curriculum without Explicit CoT

To teach the model a particular translation task $MLT_{\Pi}$ from input-output pairs of the form $(s_{1}, MLT_{\Pi}(s_{1}))$ , we can train it with relevant sections of the phrasebooks STR ( $\Pi$ ) in context as curriculum, but at test time it would not have access to the phrasebook so it is important not to teach it explicit chain-of-thought (CoT) containing in-context information. (Another consideration is data privacy, with the phrasebook being considered privileged information.) However, a dual use of CoT is to provide the model extra compute at inference time (Goyal et al., 2023), which is needed here since the translation task has d stages. To facilitate such silent computation we teach the model to output a fixed number of <THINK> tokens, sometimes referred to as silent CoT or internalized CoT.

# 2.4. ICL-capability for MLT(d, n)

To learn from books, one needs to know how to read. The analogous notion under study here is whether context-enhanced learning requires capability to sort-of “understand” the in-context material (Q2). In the context of MLT, we formalize such capability as being able to achieve low loss on the translation task when provided with the relevant phrasebook sections in context while allowing silent CoT.

Definition 2.2 (MLT(d, n)-ICL-capability, informal). A language model $f_{\theta}$ is MLT(d, n)-ICL-capable if for any set of phrasebooks $\Pi$ in MLT(d, n), $f_{\theta}([STR(\Pi), s_1])$ is close to $s_{d+1} = MLT_{\Pi}(s_1)$ disregarding the <THINK> tokens, when measured by a discrepancy metric over all valid input strings $s_1$ .

# 3. Experiments and Observations

In this section, we fix a set of phrasebooks $\Pi^{*}$ and study context-enhanced learning on $MLT_{\Pi^{*}}$ .

We first introduce the preparation of an MLT(d, n)-ICL-capable model. We then introduce a context-enhanced learning curriculum involving random dropping of phrasebook rules in context. We then present empirical evidence for significant sample efficiency of context-enhanced learning. We conclude the section with mechanistic insights into context-enhanced learning concerning internal representations and evolution of parameters.

We use the Llama 3.2-3B instruction-tuned model (Dubey et al., 2024) as the base model and fix d = 5 with n = 8 or 10. Detailed configurations are available in Appendix B.5.

# 3.1. Experimental Setup

# (i) Preparing an MLT(d, n)-ICL-Capable Model:

The Llama 3.2B model is $\mathbf{MLT}(d,n)$ -ICL-capable as it has not seen the task during training. To make it ICL-capable for our purpose, we use SFT on other random translation tasks with random phrasebooks $\Pi_1,\ldots ,\Pi_M$ , following common CoT internalization pipeline (Deng et al., 2024; Pfau et al., 2024; Hao et al., 2024). We use one training example per set of phrasebooks to prevent memorization of specific phrasebooks. At the end of training, given input $[\mathrm{STR}(\Pi),s_1]$ for any $s_1$ and $\Pi$ the model can generate <THINK>, . . ., $\mathbf{MLT}_{\Pi}(s_1)$ correctly. Details on the first stage of training are described in Appendix B.4.

(ii) Setting up context-enhanced learning for $MLT_{\Pi^{*}}$ : We use the $MLT(d,n)$ -ICL-capable model above as initialization and train for $MLT_{\Pi^{*}}$ . Supervised dataset $D_{\Pi^{*}}$ is curated with input-label pairs of the form $(s_{1}, [<THINK>, \ldots, MLT_{\Pi^{*}}(s_{1})])$ , where $s_{1}$ is a random string sampled from $A_{1}$ , with length between 20 and 40.

We define curriculum-text $\mathrm{CURR}_{\Pi^{*}}(s_{1},t)$ using excerpts from phrasebooks $\mathrm{STR}(\Pi^{*})$ (selected based on $s_{1}$ ) with random dropout of rules (parameterized by training step t). We explore the following curriculum and study their impact:

- No Context (vanilla SFT): Empty curriculum-text.   
- Fixed Dropout: A simple strategy independent of step $t$ ; given $s_1$ , only curate rules in $\Pi^*$ used in the translation of $s_1$ , then randomly drop $20\%$ of the curated rules.   
- Annealing Dropout: A better strategy: for $s_1$ , select the necessary rules from $\Pi^*$ plus 25% unused rules. Apply random dropout on these rules, increasing linearly from 0% to 100% over the first 60% of training, then maintain 100%.   
- No Dropout (ablation): Given $s_1$ , always provide all rules in $\Pi^*$ used in the translation of $s_1$ in curriculum-text.

\- Wrong Context (ablation): Equivalent to Annealing Dropout but the rules in the curriculum are incorrect.

# 3.2. Experiment Results

To check the sample efficiency benefit of context-enhanced learning (Q1), we construct supervised datasets $D_{\Pi^{*}}$ with $10^{4}$ to $10^{6}$ unique samples and train the models for one epoch on each. $^{2}$ We report the next-token prediction accuracy on the final answer tokens (ignoring thought tokens) for held-out samples when conditioning on no curriculum-text (100% dropped-out) and compare against the supervised dataset size. To check the necessity of proper ICL capability (Q2), we ablate with Annealing Dropout but starting from non-MLT(d, n)-ICL-Capable 3B base model.

![](images/3247d8a5fd58bfdf33763ecc8bdfc9324a43c13029c4160522d0588dd048ae9d.jpg)

<details>
<summary>text_image</summary>

No Context (baseline)
Fixed Dropout
Annealing Dropout
No Dropout (ablation)
Wrong Context (ablation)
No ICL (ablation)
</details>

![](images/5bef9ea5236559cf336a88ad28ffaf5ae0df7c22b08701b5698463ad19e703b0.jpg)

<details>
<summary>line</summary>

| Number of Samples |d_N* | d=5, n=8 - Random Guess | d=5, n=8 - Model 1 | d=5, n=10 - Random Guess | d=5, n=10 - Model 2 |
| ----------------- | -------- | ------------------------ | ------------------- | ------------------------- | ----------------------- |
| 10^4              | 0.15     | 0.15                     | 0.15                | 0.15                      | 0.15                    |
| 10^5              | 0.95     | 0.95                     | 0.95                | 0.95                      | 0.95                    |
| 10^6              | 0.95     | 0.95                     | 0.95                | 0.95                      | 0.95                    |
</details>

Figure 2. Context-enhanced learning of MLT $_{\Pi^{*}}$ when n = 8 and n = 10. Annealing Dropout learns the fastest followed by Fixed Dropout, requiring 10x less samples compared to vanilla SFT. Ablations show the necessity of correct context, proper dropout, and sufficient ICL capability as a good initialization. Note that there are multiple ablations overlapping around random guesses.

Figure 2 demonstrates the significant sample efficiency of context-enhanced learning. Moreover, models trained with subsets of phrasebooks and just 20% dropout give perfect heldout test-time accuracy with 100% dropout rate. Thus they are able to effectively use phrasebook rules from subsets of the phrasebook that did not co-occur in the same training sample. Clearly, the model has learned the phrasebook atomically, and can combine the rules as needed at test time.

In an ablation (Appendix C), we show that context-enhanced learning only internalizes the rules whose dropout from curriculum-text leads to an increase in loss on training data.

The experiment results can be summarized as follows:

(i) Context-enhanced learning from an ICL-capable model greatly improves training sample efficiency.   
(ii) The phrasebook rules are internalized atomically, and only when missing them incurs an increased loss.

![](images/0eafb8c2fa6954a6c52ad515e3e6d693011a18b4102595b9f3aeb3a969b96e84.jpg)

Figure 3. Evidence for sequential processing in a MLT(5,8)-ICL-capable model. Each entry is the $l_{2}$ norm between the latent representation pre and post-perturbing $\pi_{i}$ (after certain layer at certain token position). Perturbing later phrasebooks in the context changes representations in the later layers, suggesting that later translation steps happens in later layers. To rule out the possibility that the affected depth is only dependent on position of the perturbation instead of the semantic content, we conduct the same set of experiment except that we only perturb rules that are not used when translation $s_{1}$ , which yields negligible representation difference (see comparison in Figure 8).   
![](images/2a17c5311bcf683c9de361021e251f5c699b1540c786962783e52b90d4f79740.jpg)  
Figure 4. Starting from the MLT(5,8)-ICL-capable model $f_{\theta}$ evaluated in Figure 3, we fix a set of phrasebooks $\Pi^{*}$ , train with Annealing Dropout curriculum on 100k samples, and obtain an MLT $_{\Pi^{*}}$ -capable model $f_{\theta^{*}}$ . We construct “stitched models” by selectively substituting layers from layer $L_{start}$ to layer $L_{end}$ in $f_{\theta}$ by $f_{\theta^{*}}$ and evaluate on MLT $_{\Pi^{*}}$ with certain phrasebooks dropped from context. Bright colors close to the diagonal reflects that knowledge required to compensate the dropped phrasebook in context can be localized in a small subset of layers in $f_{\theta^{*}}$ . It is particularly worth noting that for any level i, the end of the group of layers responsible for storing $\pi_{i}^{*}$ matches the start of the layers responsible for reading $\pi_{i}$ (see Figure 3) before context-enhanced learning.

# 3.3. Mechanistic Insights: Layer by Layer Mapping of the Translation Task

To understand the behavior of context-enhanced learning, we probe into the hidden representations and weights of models before and after context-enhanced learning.

# Sequential Processing in ICL-capable model

First, we look into how phrasebooks in curriculum-text are used by an MLT(d, n)-ICL-capable model during the translation task via the following controlled experiment. Fix a random set of phrasebooks $\Pi = \{\pi_{1}, \ldots, \pi_{d}\}$ and an input sequence $s_{1}$ , we pass $[\mathrm{STR}(\Pi), s_{1}, <\mathrm{THINK}>, \ldots, s_{d+1}]$ into the ICL-capable model, and record the model's hidden representations (output of each transformer block) for the tokens $<\mathrm{THINK}>, \ldots, s_{d+1}$ .

Next, we apply independent perturbations to each phrasebook $\pi_i$ in the curriculum-text and measure the change in the model's representations. That is, for each level $1 \leq i \leq d$ , we randomly sample 10 rules in $\pi_i$ used in the translation of $s_1$ and replace their image by other randomly chosen tuples. We denote the perturbed phrasebook $\mathrm{STR}(\hat{\pi}_i)$ , and compute the norm difference between the model's representations at tokens $<\mathrm{THINK}>, \ldots, s_{d+1}$ before and after replacing

STR $(\pi_{i})$ with STR $(\hat{\pi}_{i})$ . The first layer showing a significant difference in its output representation is identified as the layer where the model begins processing the phrasebook.

In Figure 3, we show that an ICL-capable model processes phrasebooks in curriculum-text sequentially, with earlier phrasebooks read by earlier layers. Formal details are in Appendix B.2 and additional experiments are in Appendix C.2.

# Localized Storage after Context-Enhanced Learning

Here, we verify whether a similar sequential pattern for internalizing phrasebooks is used in a MLT $_{\Pi^{*}}$ -capable model. We denote the ICL-capable model as $f_{\theta}$ and its post context-enhanced learning counterpart as $f_{\theta^{*}}$ . As $f_{\theta^{*}}$ 's capability no longer depends on textual representation of the phrasebooks, our analysis focuses on the model's parameters.

We assess the importance of each layer in model $f_{\theta^{*}}$ for each phrasebook by constructing “stitched” models and measuring their output behavior. For every $1 \leq L_{start} \leq L_{end} \leq 28$ (total layers in Llama3.2 3B), we replace layers $L_{start}$ to $L_{end}$ of the ICL model $f_{\theta}$ with corresponding layers from $f_{\theta^{*}}$ to create a stitched model. This strategy has been used in prior works on localizing information after SFT (Gong et al., 2022; Panigrahi et al., 2023; Wei et al., 2024).

For each level $1 \leq i \leq d$ , we evaluate the stitched models on $MLT_{\Pi^{*}}$ using the following in-context information: $[\mathrm{STR}(\pi_{1}^{*}), \cdots, \mathrm{STR}(\pi_{i-1}^{*}), \mathrm{STR}(\pi_{i+1}^{*}), \cdots, \mathrm{STR}(\pi_{d}^{*})]$ . Note here the textual representation of $\pi_{i}^{*}$ has been dropped. If a stitched model shows high accuracy on inputs that use $\pi_{i}^{*}$ for translation but contain only the above in-context information, then the layers selected from $f_{\theta^{*}}$ to create the stitched model can be deemed responsible for storing information on $\pi_{i}^{*}$ in their parameter space.

Figure 4 demonstrates that information of all phrasebooks can be localized to a few mutually disjoint layers in $f_{\theta^{*}}$ . Moreover, the end of the group of layers where $\pi_{i}^{*}$ is localized in $f_{\theta^{*}}$ marks the start of the layers that begin processing level i phrasebook in the ICL-capable model $f_{\theta}$ (e.g. compare the role of layer 17 in Figures 3 and 4). Formal details and additional experiments are in Appendices B.3 and C.2.

This suggests that instead of storing phrasebooks as a single chunk in its parameters, the model re-learns each translation step locally to compensate for missing information when rules are dropped. Thus, we conjecture that context-enhanced learning leverages curriculum-text to improve training by localizing learning in the parameter space. We build a surrogate model in Section 5 to show how such localized learning can prove beneficial for faster training.

# 4. Curriculum-text: Detectable from Queries?

Context-enhanced learning for MLT uses the phrasebooks in its curriculum-text. Can rules of the phrasebooks be recovered post-hoc querying the model (Q3) using likelihood-based methods for detecting training data?

We take an MLT $_{\Pi^{*}}$ -capable model (with $d = 5, n = 10$ ) trained with context-enhanced learning from Section 3.2. Let STR ( $\Pi^{*}$ ) denote the concatenation of all phrasebooks. For each rule in STR ( $\Pi^{*}$ ), which are of the form “a b -> C D” $^{3}$ , we measure the probability of the model generating the ground truth tokens “C D” after observing “a b ->”. We compute (1) the fraction of cases where the model’s top-1 prediction matches the ground truth (greedy decoding) and (2) the probability of generating ground truth tokens via random sampling (with temperature set as 1). These are standard tests on textual description, and we would expect high probabilities for the correct tokens.

We also test two adversarial strategies with additional token filtering $^{4}$ in generation: (i) setting probability of <THINK> tokens to zero (ii) when querying for a rule in $\pi_{i}$ with output alphabet $A_{i+1}$ , setting probability of all tokens outside $A_{i+1}$ to zero $^{5}$ . Note that this corresponds to a strong adversary which knows the alphabet set of the intermediate phrasebooks, but not the rules.

Table 2. Recovery Success Rate (Rounded to 2 decimals) 

<table><tr><td rowspan="2">Token Filter</td><td colspan="2">Greedy Decoding</td><td colspan="2">Sampling</td></tr><tr><td> $\pi_1^* - \pi_4^*$ </td><td> $\pi_5^*$ </td><td> $\pi_1^* - \pi_4^*$ </td><td> $\pi_5^*$ </td></tr><tr><td>None</td><td>0.00%</td><td>0.20%</td><td>0.00%</td><td>0.89%</td></tr><tr><td>No &lt;THINK&gt;</td><td>0.00%</td><td>0.20%</td><td>0.00%</td><td>0.90%</td></tr><tr><td>Only from  $A_{i+1}$ </td><td>1.66%</td><td>0.20%</td><td>1.28%</td><td>0.94%</td></tr></table>

As shown in Table 2, recovering rules from intermediate phrasebooks $(\pi_1,\dots ,\pi_4)$ is nearly impossible without token filtering. For the final phrasebook $(\pi_{5})$ , results remain near-random, where random guess probability is $1\%$ for a 2-tuple when $n = 10$ . Even with filtering (ii), recovery success rates remain only slightly above random. Setup details and more results are available in Appendix C.3.

# 5. Mathematical Analysis

Having empirically demonstrated the sample efficiency benefits of context-enhanced learning, we now formalize them through a mathematical lens. We first define the sample complexity of an algorithm for learning the task.

Definition 5.1 (informal). Sample complexity for an algorithm to learn $MLT_{\Pi}$ is defined as the minimum total length of all (possibly repeated) input sequences required by the algorithm to return an $MLT_{\Pi}$ -capable model $f_{\theta}$ .

Analysis of gradient-based learning on a multilayer transformer (let alone a 28-layer model like Llama 3.2-3B) is an open mathematical question. We use our mechanistic findings (i.e., translation layers map onto transformer layers) to propose a surrogate model to think about how transformers learn $\mathbf{MLT}(d,n)$ . In this surrogate model we demonstrate that learning a task $\mathbf{MLT}_{\Pi}$ via vanilla SFT will require $n^{\Omega(d)}$ samples. Then, we prove that, with context-enhanced learning, the surrogate model can learn $\mathbf{MLT}_{\Pi^{*}}$ with a sample complexity of $\mathcal{O}(\text{poly}(n)d\log d)$ .

# 5.1. Surrogate Model (SURR-MLT)

Formalization of the surrogate model, in short SURR-MLT, relies on observations from Section 3.3, which reveal that an ICL-capable model (Definition 2.2) performs the translation task step-by-step, with earlier phrasebook processed by lower layers. This aligns with how the model stores internalized knowledge in layers after context-enhanced learning. Our SURR-MLT represents an idealized and simplified transformer that has already been “pre-conditioned” to solve MLT in this sequential fashion. Using SURR-MLT, we will show the benefits of context-enhanced learning.

Without loss of generality we assume the alphabet sets as $A_{1},\ldots,A_{d+1}:=A=\{1,2,\ldots,n\}$ . Surr-MLT will represent a length-L sequence $\boldsymbol{s}_{i}=(s_{i,1},\ldots,s_{i,L})$ as an embedding matrix $V_{i}\in R^{n^{2}\times L/2}$ that uses $\{\boldsymbol{v}(s_{i,1},s_{i,2}),\cdots,\boldsymbol{v}(s_{i,L-1},s_{i,L})\}$ as columns. Here, for any 2-tuple $(a,b)$ , $\boldsymbol{v}(a,b)\in\mathbb{R}^{n^{2}}$ represents a one-hot vector with 1 at dimension $an+b$ . Surr-MLT operates on embedding matrices, transforming $V_{1}\to V_{2}\to\cdots\to V_{d+1}$ . Each layer i will be primarily defined by two matrices, $C_{i},W_{i}\in R^{n^{2}\times n^{2}}$ . $C_{i}$ presents a (possibly partial or completely dropped) phrasebook that is provided in-context, and $W_{i}$ is a trainable parameter storing phrasebook information during context-enhanced learning $^{6}$ .

Definition 5.2. SURR-MLT with trainable parameters $\{\pmb{W}_i\}_{i=1}^d$ and in-context representation $\{\pmb{C}_i\}_{i=1}^d$ , is represented by its operation on an input $s_1$ as

$$
\begin{array}{l} \boldsymbol {V} _ {d + 1} = \text { S   U   R   R   -   M   L   T } _ {\{\boldsymbol {W} _ {i} \} _ {i = 1} ^ {d}} \left(\left\{\boldsymbol {C} _ {i} \right\} _ {i = 1} ^ {d}, \boldsymbol {V} _ {1}\right), \quad \text { where } \\ \boldsymbol {V} _ {i + 1} = \operatorname{HardMax} \left(\boldsymbol {C} _ {i} + \boldsymbol {W} _ {i}\right) \text {Shift} \left(\boldsymbol {V} _ {i}\right), \text {for} i \geq 1, \\ \end{array}
$$

and $V_{1}$ is the embedding matrix for the input string $s_{1}$ .

Here Shift represents Circular shift operation and is defined as a Hadamard product on the embedding matrices (details in Definition G.2). HardMax represents hard-max function converting $C_{i} + W_{i}$ to a binary column stochastic matrix. In the following discussion, we show 2 examples where the surrogate model can perfectly represent an ICL-capable model and $MLT_{\Pi^{*}}$ -capable model.

Case 1 (Representing MLT(d, n)-ICL-capable): The trainable matrices are all 0's as the model hasn't undergone context-enhanced learning. To produce the output for a task MLT $_{\Pi}$ , SURR-MLT takes phrasebooks into in-context representations by setting each $C_{i}$ as a stochastic matrix Matrix( $\pi_{i}$ ), which represents rules of $\pi_{i}$ as one-hot columns (please see Definition G.4 and Lemma G.5).

Case 2 (Representing MLT $_{\Pi^{*}}$ -capable): For a model that has performed context-enhanced learning on a translation task MLT $_{\Pi^{*}}$ , no in-context information will be provided to the surrogate model and so $\{C_{i}\}_{i=1}^{d}$ will be all 0s. A MLT $_{\Pi^{*}}$ -capable-model should contain the phrasebooks as $\{\text{Matrix}(\pi_{i}^{*})\}_{i=1}^{d}$ in its trainable parameters $\{W_{i}\}_{i=1}^{d}$ .

SURR-MLT as an Ideal Transformer for MLT: The following theorem constructs a transformer that can simulate SURR-MLT. Thus, while our discussions and proofs focus on the surrogate model for simplicity, they remain fully applicable to the transformer architecture.

Theorem 5.3 (cf Lemma H.4). There exists a transformer that can simulate SURR-MLT with 2d self-attention and 2d MLP layers with embedding dimension $2n^{2} + 2d + 4$ .

# 5.2. Sample Complexity for Vanilla SFT

For the surrogate model, vanilla SFT corresponds to always setting in-context representations $\{C_i\}_{i=1}^d$ to 0s when training for $\mathbf{MLT}_{\Pi^*}$ . While we use the surrogate model for consistency in our discussion, the argument generalizes to any model learning $\mathbf{MLT}_{\Pi^*}$ with vanilla SFT.

Our analysis is built on the Statistical Query (SQ) framework (Kearns, 1998), which measures the difficulty of learning tasks using algorithms that rely on expectation estimates of specific functions to approximate the true solution. Gradient-based methods, such as Stochastic Gradient Descent (SGD), fall under this framework as they compute gradients by estimating expectations of loss functions and their derivatives.

The complexity of learning a task is quantified by the SQ dimension, which measures the number of candidate functions that are pairwise uncorrelated under the input distribution and difficult to distinguish with limited samples. A higher SQ dimension implies a richer hypothesis class, that requires more samples to identify the correct function. We show in the following theorem that the SQ dimension of $\mathbf{MLT}(d,n)$ grows exponentially with task parameters.

Theorem 5.4. SQ dimension of MLT(d, n) under uniform input distribution is at least $n^{\Omega(d)}$ .

Informally this implies that any algorithm that tries to learn a $MLT_{\Pi^{*}}$ -capable SURR-MLT with trainable parameters $\{W_{i}\}_{i=1}^{d}$ , and in-context information $\{C_{i}\}_{i=1}^{d}$ always fixed at 0s, by minimizing loss across samples will require at least $n^{\Omega(d)}$ sample complexity.

A corollary is that for vanilla SFT with SGD, sample complexity to learn $MLT_{\Pi^{*}}$ can be at least $n^{\Omega(d)}$ . This is adapted from Edelman et al. (2023), who analyse for sparse parity that has similar SQ dimension (Corollary F.6).

Informal proof for SQ dimension: Our proof extensively analyses the case where number of characters is 2 (lem. F.10), on which we will build proof for general n (lem. F.20). The proof for n = 2 leverages uncorrelations between two randomly selected translation tasks; i.e. we show that for two random set of phrasebooks $\Pi^{\alpha}, \Pi^{\beta}$ , the translation tasks $MLT_{\Pi^{\alpha}}, MLT_{\Pi^{\beta}}$ will have 0 output correlation with probability at least $1 - 2^{-\Omega(d)}$ w.r.t. random choice of $\Pi^{\alpha}, \Pi^{\beta}$ (Lemma F.13). We then show that we can pick exponentially many such random set of phrasebooks for which the translation tasks will be pairwise uncorrelated. This translates to a high SQ dimension.

# 5.3. Sample Complexity of Context-Enhanced Learning

Here, we show that context-enhanced learning substantially improves the sample complexity of learning in SURR-MLT. An MLT(d,n)-ICL-capable model gets a set of phrasebooks $\Pi^{*}$ by setting $\{\text{Matrix}(\pi_{i}^{*})\}_{i=1}^{d}$ as in-context representations $\{C_{i}\}_{i=1}^{d}$ . When a curriculum is followed such that a translation rule is dropped from a phrasebook, say $\pi_{i}^{*}$ , SURR-MLT will set the corresponding column in its in-context representation $C_{i}$ as 0's. Denote the zero-ed out column by $\boldsymbol{C}_{i}^{(j)}$ for some generic column index j, we note that the loss will be low if and only if the corresponding column in the learnable parameters $\boldsymbol{W}_{i}^{(j)}$ exactly matches $\text{Matrix}(\pi_{i}^{*})^{(j)}$ .

A heuristic search algorithm, that searches among $n^{2}$ possibilities for the dropped rule and stores its one-hot representation in the corresponding column in $W_{i}$ , can be used to minimize the loss. By sequentially dropping rules, followed by a search and store process, the algorithm achieves polynomial sample complexity.

Theorem 5.5 (Informal; cf Corollary G.19). For any task $\mathbf{MLT}_{\Pi^*}$ , there is a heuristic search algorithm paired with a curriculum of iteratively dropping rules from phrasebooks, that can learn a $\mathbf{MLT}_{\Pi^*}$ -capable SURR-MLT with sample complexity $\mathcal{O}(n^6 d\log d)$ with high probability.

The enumerative step in the heuristic search algorithm requires $\Theta(n^{2})$ steps on average, as the algorithm needs to search over $\Theta(n^{2})$ possibilities when a rule is dropped from a phrasebook. Instead, we show that gradient descent requires only a few steps per dropped rule. Dropping a rule sets the respective column in the in-context representation to 0, causing the gradient for the corresponding column in the trainable parameters to strongly align with the missing column. We formally present results for d = 2; due to exponentially growing number of terms to analyse with higher d, we keep the result for general d as a conjecture.

Theorem 5.6 (Informal; cf Corollary G.25). When d = 2, there is a gradient descent based algorithm, paired with a curriculum of iteratively dropping a random rule from phrasebooks, that can return $MLT_{\Pi^{*}}$ -capable SURR-MLT with sample complexity $\mathcal{O}(n^{4})$ with high probability.

While theoretical analysis of GD dynamics beyond d = 2 is challenging, empirically we show that when training with SGD, the trainable parameters for much deeper SURR-MLT can quickly learn the set of phrasebooks in $\Pi^{*}$ . In Figure 5, we show context-enhanced learning results for a SURR-MLT on a randomly selected set of phrasebooks $\Pi^{*}$ in MLT(10, 10). Regardless of whether we learn each layer separately with layer-wise SGD or all layers simultaneously with SGD in the surrogate model, the trainable parameters learn the phrasebooks in $\Pi^{*}$ very quickly. More discussions are deferred to Appendix G.5.

![](images/f32521801b25e4a007d85c7b13ed6ca0ac6cc77514eb3a98371857e1a0f249ea.jpg)

<details>
<summary>line</summary>

| Steps | Layer-wise GD |
| ----- | ------------- |
| 0     | 0.0           |
| 2500  | 0.8           |
| 5000  | 0.95          |
| 7500  | 1.0           |
| 10000 | 1.0           |
</details>

![](images/b115ab9424794a201c6221dcb08ffbbd043bb9318796e6440aa8b6152d5f9922.jpg)

<details>
<summary>line</summary>

| Steps | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 | Line 6 | Line 7 | Line 8 | Line 9 |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| 0     | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    |
| 2500  | 0.8    | 0.7    | 0.6    | 0.5    | 0.4    | 0.3    | 0.2    | 0.1    | 0.05   |
| 5000  | 1.0    | 0.9    | 0.8    | 0.7    | 0.6    | 0.5    | 0.4    | 0.3    | 0.2    |
| 7500  | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    |
| 10000 | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    |
</details>

![](images/66c2e0f959405bf969f899b7f083b6c90c9f33ff5693564bafafd9e9bc64b8ce.jpg)  
Figure 5. Percentage of columns in HardMax( $W_{i}$ ) matching Matrix( $\pi_{i}^{*}$ ) when optimizing a SURR-MLT for MLT(10, 10). Learnable parameters of all layers quickly converge to the correct values, regardless of whether layer-wise updates are imposed.

# 5.4. Insight from SURR-MLT: Gradient Quality

In this section, we show the major difference between context-enhanced learning and vanilla SFT to be the amount of predictive information in gradients for the trainable parameters. Our theoretical analysis in Theorem 5.6 on SURR-MLT primarily shows that when a single translation rule in a phrasebook $\pi_{i}^{*}$ is dropped, equivalently a column in the in-context representation, say $C_{i}^{(j)}$ , is zero-ed out, the gradient of the corresponding column in trainable parameter $\boldsymbol{W}_{i}^{(j)}$ aligns strongly with the one-hot vector representation of the dropped rule, $\text{Matrix}(\pi_{i}^{*})^{(j)}$ . However, this strong alignment heavily relies on the presence of other rules in-context. When more rules are dropped, the gradient signal gets increasingly “inaccurate”. We quantitatively characterize such degradation as follows:

Consider an ICL-capable SURR-MLT, we focus on a column (with index $j$ , denoted as superscript) $\boldsymbol{W}_1^{(j)}$ in the first layer with ground truth $\mathrm{Matrix}(\pi_1^*)(^j)$ . We define gradient prediction accuracy (see formal definition in Definition B.1) for $\boldsymbol{W}_1^{(j)}$ as the probability that the argmax entry of the negative stochastic batch gradient on $\boldsymbol{W}_1^{(j)}$ matches the argmax entry of $\mathrm{Matrix}(\pi_1^*)(^j)$ . Intuitively, a higher gradient prediction accuracy will make learning $\mathrm{Matrix}(\pi_1^*)$ easier. We track the accuracy metric when only $\boldsymbol{C}_i^{(j)}$ is zero-ed out as well as when taking more aggressive context dropout schemes.

![](images/070b1eb8fa811a1131dc3b62ccdbd2b9d5d36d5b25007a8f59fff9544f67c321.jpg)

<details>
<summary>line</summary>

| Rule Dropping Rate | Batch size 10^0 | Batch size 10^1 | Batch size 10^2 | Batch size 10^3 |
| ------------------ | --------------- | --------------- | --------------- | --------------- |
| 0.00               | 1.0             | 1.0             | 1.0             | 1.0             |
| 0.25               | 0.95            | 0.98            | 0.99            | 0.995           |
| 0.50               | 0.85            | 0.92            | 0.95            | 0.97            |
| 0.75               | 0.70            | 0.85            | 0.90            | 0.93            |
| 1.00               | 0.55            | 0.75            | 0.85            | 0.90            |
</details>

![](images/519c85c77fb7bf8bde99a3befcb374baabd57e215e81907968400ebac363c99f.jpg)

<details>
<summary>line</summary>

| Max. Phrasebook Index | Rule Dropping Rate |
| --------------------- | ------------------ |
| 2                     | 0.1                |
| 4                     | 0.2                |
| 6                     | 0.3                |
| 8                     | 0.4                |
| 10                    | 0.5                |
</details>

Figure 6. Gradient prediction accuracy when (left) varying dropping rates for rules in $\pi_{1}$ and batch sizes of gradient computation, and (right) varying dropping rates and max phrasebook index (k) for rules dropped from $\pi_{1},\ldots,\pi_{k}$ . Higher dropping rates always lead to lower gradient prediction accuracy.

In Figure 6, we report the gradient prediction accuracy with dropout schemes involving (1) multiple rules dropped out from the first phrasebook and (2) rules dropped from multiple phrasebooks. We can see that higher dropping rates significantly degrade the gradient prediction accuracy, highlighting the necessity of proper in-context information in order to obtain the optimization benefit. More details on the metric and experiments are deferred to Appendix B.1. We note that finding above is based on observations on the simplified surrogate model. A quantitative characterization of the optimization benefit for context-enhanced learning in the LLM regime is left as an interesting future work.

# 6. Related Works

In this section, we discuss related works to this paper. More related works regarding (1) compositional and OOD generalization and (2) mechanistic understanding of transformers are discussed more extensively in Appendix D.

# Learning using Privileged Information (LUPI)

LUPI was formally introduced by Vapnik & Vashist (2009) in kernel SVMs, and it is heavily related to concepts of Learning with Side Information (Kuusela & Ocone, 2002; Jonschkowski et al., 2015; Zhang et al., 2018). Primarily, both concepts refer to training a model with additional information that could help training but may not be available at test time. This framework has been extended theoretically for classification tasks (e.g. Pechyony & Vapnik (2010); Momeni et al. (2018)) and has been used to explain benefits of knowledge distillation (Vapnik et al., 2015; Lopez-Paz et al., 2015). To name a few applications, this concept has been heavily studied for improving boosting for classification tasks (Chen et al., 2012), visual and video encoders (Hoffman et al., 2016; Cheng et al., 2020; Xu et al., 2017), human preference predictions (Farias & Li, 2019), multiagent games (Sessa et al., 2020), speech recognition (Synnaeve et al., 2014), and medical recognition (Ceccarelli & Maratea, 2008; Sabeti et al., 2020).

While LUPI has rich application in classification, extending LUPI to LLMs introduces unique challenges due to their auto-regressive training. Such a concept raises questions on whether applying auto-regressive loss on the additional information is necessary to get the benefits of additional supervision information in context, and how the additional information changes the training behavior of LLMs. Hence, our work is a nontrivial generalization of this framework to LLMs that connects to their in-context learning strengths.

# In-context learning and memorization

Recent works have studied emergence of ICL and competition with in-weights learning (IWL) (equivalent to knowledge memorization) during pre-training of language models (Chan et al., 2022; Reddy, 2024; Singh et al., 2023; 2024; Nguyen & Reddy, 2024). These works show that data distribution properties affect the behavior of the model during training. Our work can be thought of as contemporary to the works above, where we show that strong ICL capabilities can be utilized for improving knowledge on a task by context-enhanced learning.

Benefits of in-context learning: In-context learning has been primarily studied in the context of few-shot prompting of large language models. Better supervision with in-context supervision can help in improved performance (e.g. some representative works (Arora et al., 2022; Si et al., 2022; Wu et al., 2022; Lu et al., 2021; Su et al., 2022)), OOD generalization and factuality (reduced hallucination) (Yang et al., 2023; Dhuliawala et al., 2023; Chen et al., 2023; Didolkar et al., 2024), and more structured latent representations (Park et al., 2024) for large language models. On the other hand, we show that improved supervision with in-context supervision can also help a model learn faster in SFT, while seemingly not leaking the in-context information in its output probabilities.

# 7. Discussion, Limitations, and Future work

Some experimental works have implicitly used the notion of context-enhanced learning but the current paper formalized this notion for auto-regressive models and showed, using MLT, that this form of learning can be exponentially more sample-efficient than standard SFT. At the end of training it is hard to recover the in-context information seen during training from the model's output probabilities. We note that this finding appears to have implications about copyright law (e.g., whether or not LLM training amounts to “transformative use” of text (Carlini et al., 2021; 2022; Karamolegkou et al., 2023)) whose further study is left for future work.

Our experiments focus on a synthetic MLT task for a few reasons: (1) to ensure that the task is absent from LLM pre-training, which allows precise quantification of benefits of context-enhanced learning, including not revealing the curriculum text at inference time. (2) the task is too difficult (at least for Llama 3.2 3B model) to learn via vanilla SFT, but is learnable via context-enhanced learning. Extending these findings to real-world complicated tasks (e.g., in math and coding) is left for future work.

Our convergence analysis for context-enhanced learning relies on a surrogate model, and extending it to an actual transformer remains an open challenge for theory of deep learning. Extending formalization of context-enhanced learning to explore LLM training in multi-agent settings—where models collaborate and learn from each other to discover novel concepts—would be an exciting avenue for future research.

# Acknowledgment

We thank Yun Cheng, Simon Park, Tianyu Gao, Yihe Dong, Zixuan Wang, Haoyu Zhao, and Bingbin Liu for discussions, suggestions, and proof-reading at various stages of the paper. AP, SA acknowledge funding from NSF, PLI, DARPA, ONR, and OpenAI. XZ is additionally supported by a Gordon Y.S. Wu Fellowship in Engineering.

# Impact statement

We formulate a basic notion “context-enhanced learning”, and study how it is different from usual learning. One consequence of our study is that enhancing the context with good quality data can enhance the quality of the learner.

This enhancement could be used with privileged information and it appears that the training can be done in such a way that the model does not leak this privileged information. This can be seen as enhancing privacy, since there is a lower chance of leaking privileged training data.

The flip side is that it suggests — albeit in very toy and synthetic setting, with full study left for future work — that model training could use off-limits data (albeit with no gradient updates on it) and this use might not be detectable from querying the trained model. But this is hypothetical at this point since the paper concerns a very toy setting.

# References

Akyurek, E., Schuurmans, D., Andreas, J., Ma, T., and Zhou, D. What learning algorithm is in-context learning? investigations with linear models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=0g0X4H8yN4I.   
Akyürek, E., Wang, B., Kim, Y., and Andreas, J. In-context language learning: Architectures and algorithms. arXiv preprint arXiv:2401.12973, 2024.   
Allen-Zhu, Z. and Li, Y. Physics of language models: Part 1, Context-free grammar. arXiv preprint arXiv:2305.13673, 2023a.   
Allen-Zhu, Z. and Li, Y. Physics of language models: Part 3.2, Knowledge manipulation. arXiv preprint arXiv:2309.14402, 2023b.   
Allen-Zhu, Z. and Li, Y. Physics of language models: Part 3.3, knowledge capacity scaling laws. arXiv preprint arXiv:2404.05405, 2024.   
Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V., Slone, A., Gur-Ari, G., Dyer, E., and Neyshabur, B. Exploring length generalization in large

language models. Advances in Neural Information Processing Systems, 35:38546–38556, 2022.   
Arora, S., Narayan, A., Chen, M. F., Orr, L., Guha, N., Bhatia, K., Chami, I., Sala, F., and Ré, C. Ask me anything: A simple strategy for prompting language models. arXiv preprint arXiv:2210.02441, 2022.   
Bavarian, M., Jun, H., Tezak, N., Schulman, J., McLeavey, C., Tworek, J., and Chen, M. Efficient training of language models to fill in the middle. arXiv preprint arXiv:2207.14255, 2022.   
Bhattamishra, S., Ahuja, K., and Goyal, N. On the ability and limitations of transformers to recognize formal languages. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 7096–7116, 2020.   
Blum, A., Furst, M., Jackson, J., Kearns, M., Mansour, Y., and Rudich, S. Weakly learning dnf and characterizing statistical query learning using fourier analysis. In Proceedings of the twenty-sixth annual ACM symposium on Theory of computing, pp. 253–262, 1994.   
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.   
Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pp. 2633–2650, 2021.   
Carlini, N., Ippolito, D., Jagielski, M., Lee, K., Tramer, F., and Zhang, C. Quantifying memorization across neural language models. arXiv preprint arXiv:2202.07646, 2022.   
Ceccarelli, M. and Maratea, A. Improving fuzzy clustering of biological data by metric learning with side information. International Journal of Approximate Reasoning, 47(1):45–57, 2008.   
Chan, S., Santoro, A., Lampinen, A., Wang, J., Singh, A., Richemond, P., McClelland, J., and Hill, F. Data distributional properties drive emergent in-context learning in transformers. Advances in neural information processing systems, 35:18878–18891, 2022.   
Chen, J., Liu, X., and Lyu, S. Boosting with side information. In Asian Conference on Computer Vision, pp. 563–577. Springer, 2012.   
Chen, J., Pan, X., Yu, D., Song, K., Wang, X., Yu, D., and Chen, J. Skills-in-context prompting: Unlocking

compositionality in large language models. arXiv preprint arXiv:2308.00304, 2023.   
Cheng, L., Zhou, X., Zhao, L., Li, D., Shang, H., Zheng, Y., Pan, P., and Xu, Y. Weakly supervised learning with side information for noisy labeled images. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXX 16, pp. 306–321. Springer, 2020.   
Choi, Y., Asif, M. A., Han, Z., Willes, J., and Krishnan, R. G. Teaching llms how to learn with contextual fine-tuning. arXiv preprint arXiv:2503.09032, 2025.   
Deng, Y., Choi, Y., and Shieber, S. From explicit cot to implicit cot: Learning to internalize cot step by step. arXiv preprint arXiv:2405.14838, 2024.   
Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X., Celikyilmaz, A., and Weston, J. Chain-of-verification reduces hallucination in large language models. arXiv preprint arXiv:2309.11495, 2023.   
Didolkar, A., Goyal, A., Ke, N. R., Guo, S., Valko, M., Lillicrap, T., Rezende, D., Bengio, Y., Mozer, M., and Arora, S. Metacognitive capabilities of llms: An exploration in mathematical problem solving. arXiv preprint arXiv:2405.12205, 2024.   
Donahue, C., Lee, M., and Liang, P. Enabling language models to fill in the blanks. arXiv preprint arXiv:2005.05339, 2020.   
Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
Edelman, B. L., Goel, S., Kakade, S., Malach, E., and Zhang, C. Pareto frontiers in neural feature learning: Data, compute, width, and luck. arXiv preprint arXiv:2309.03800, 2023.   
Eldan, R. and Li, Y. TinyStories: How small can language models be and still speak coherent English? arXiv preprint arXiv:2305.07759, 2023.   
Farias, V. F. and Li, A. A. Learning preferences with side information. Management Science, 65(7):3131–3149, 2019.   
Gao, T., Wettig, A., He, L., Dong, Y., Malladi, S., and Chen, D. Metadata conditioning accelerates language model pre-training. arXiv preprint arXiv:2501.01956, 2025.   
Gong, Z., He, D., Shen, Y., Liu, T.-Y., Chen, W., Zhao, D., Wen, J.-R., and Yan, R. Finding the dominant winning ticket in pre-trained language models. In Findings of the

Association for Computational Linguistics: ACL 2022, pp. 1459–1472, 2022.   
Goyal, S., Ji, Z., Rawat, A. S., Menon, A. K., Kumar, S., and Nagarajan, V. Think before you speak: Training language models with pause tokens. arXiv preprint arXiv:2310.02226, 2023.   
Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., and Tian, Y. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.   
Hendrycks, D. and Gimpel, K. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.   
Hoffman, J., Gupta, S., and Darrell, T. Learning with side information through modality hallucination. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 826–834, 2016.   
Jonschkowski, R., Höfer, S., and Brock, O. Patterns for learning with side information. arXiv preprint arXiv:1511.06429, 2015.   
Karamolegkou, A., Li, J., Zhou, L., and Søgaard, A. Copyright violations and large language models. arXiv preprint arXiv:2310.13771, 2023.   
Kearns, M. Efficient noise-tolerant learning from statistical queries. Journal of the ACM (JACM), 45(6):983–1006, 1998.   
Kenton, J. D. M.-W. C. and Toutanova, L. K. Bert: Pretraining of deep bidirectional transformers for language understanding. In Proceedings of naacL-HLT, volume 1. Minneapolis, Minnesota, 2019.   
Knudsen, L. R. Practically secure feistel ciphers. In International Workshop on Fast Software Encryption, pp. 211–221. Springer, 1993.   
Kuusela, P. and Ocone, D. Learning with side information: Part i. 02 2002.   
Li, D., You, J., Funakoshi, K., and Okumura, M. A-tip: attribute-aware text infilling via pre-trained language model. In Proceedings of the 29th International Conference on Computational Linguistics, pp. 5857–5869, 2022.   
Li, Y., Li, Y., and Risteski, A. How do transformers learn topic structure: Towards a mechanistic understanding. In International Conference on Machine Learning, pp. 19689–19729. PMLR, 2023.   
Liao, H., He, S., Hao, Y., Li, X., Zhang, Y., Liu, K., and Zhao, J. Skintern: Internalizing symbolic knowledge for distilling better cot capabilities into small language models. arXiv preprint arXiv:2409.13183, 2024.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.   
Lopez-Paz, D., Bottou, L., Schölkopf, B., and Vapnik, V. Unifying distillation and privileged information. arXiv preprint arXiv:1511.03643, 2015.   
Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.   
Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.   
Lu, Y., Bartolo, M., Moore, A., Riedel, S., and Stenetorp, P. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. arXiv preprint arXiv:2104.08786, 2021.   
Momeni, A., Tatwawadi, K., and Stanford, U. Understanding lupi (learning using privileged information). Ionosphere, 201(7):6, 2018.   
Motwani, R. Randomized Algorithms. Cambridge University Press, 1995.   
Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. Progress measures for grokking via mechanistic interpretability. In The Eleventh International Conference on Learning Representations, 2023.   
Nguyen, A. and Reddy, G. Differential learning kinetics govern the transition from memorization to generalization during in-context learning. arXiv preprint arXiv:2412.00104, 2024.   
Panigrahi, A., Saunshi, N., Zhao, H., and Arora, S. Task-specific skill localization in fine-tuned language models. In International Conference on Machine Learning, pp. 27011–27033. PMLR, 2023.   
Park, C. F., Lee, A., Lubana, E. S., Yang, Y., Okawa, M., Nishi, K., Wattenberg, M., and Tanaka, H. Iclr: In-context learning of representations. arXiv preprint arXiv:2501.00070, 2024.   
Pechyony, D. and Vapnik, V. On the theory of learning with privileged information. Advances in neural information processing systems, 23, 2010.   
Pfau, J., Merrill, W., and Bowman, S. R. Let's think dot by dot: Hidden computation in transformer language models. arXiv preprint arXiv:2404.15758, 2024.

Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A., and Lewis, M. Measuring and narrowing the compositionality gap in language models. arXiv preprint arXiv:2210.03350, 2022.   
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.   
Ramesh, R., Lubana, E. S., Khona, M., Dick, R. P., and Tanaka, H. Compositional capabilities of autoregressive transformers: A study on synthetic, interpretable tasks. arXiv preprint arXiv:2311.12997, 2023.   
Reddy, G. The mechanistic basis of data dependence and abrupt learning in an in-context classification task. In The Twelfth International Conference on Learning Representations, 2024.   
Sabeti, E., Drews, J., Reamaroon, N., Warner, E., Sjoding, M. W., Gryak, J., and Najarian, K. Learning using partially available privileged information and label uncertainty: application in detection of acute respiratory distress syndrome. IEEE journal of biomedical and health informatics, 25(3):784–796, 2020.   
Sessa, P. G., Bogunovic, I., Krause, A., and Kamgarpour, M. Contextual games: Multi-agent learning with side information. Advances in Neural Information Processing Systems, 33:21912–21922, 2020.   
Si, C., Gan, Z., Yang, Z., Wang, S., Wang, J., Boyd-Graber, J., and Wang, L. Prompting gpt-3 to be reliable. arXiv preprint arXiv:2210.09150, 2022.   
Singh, A., Chan, S., Moskovitz, T., Grant, E., Saxe, A., and Hill, F. The transient nature of emergent in-context learning in transformers. Advances in Neural Information Processing Systems, 36:27801–27819, 2023.   
Singh, A. K., Moskovitz, T., Hill, F., Chan, S. C., and Saxe, A. M. What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation. arXiv preprint arXiv:2404.07129, 2024.   
Spencer, J. Asymptotic lower bounds for ramsey functions. Discrete Mathematics, 20:69-76, 1977.   
Su, D., Sukhbaatar, S., Rabbat, M., Tian, Y., and Zheng, Q. Dualformer: Controllable fast and slow thinking by learning with randomized reasoning traces. arXiv preprint arXiv:2410.09918, 2024.   
Su, H., Kasai, J., Wu, C. H., Shi, W., Wang, T., Xin, J., Zhang, R., Ostendorf, M., Zettlemoyer, L., Smith, N. A.,

et al. Selective annotation makes language models better few-shot learners. arXiv preprint arXiv:2209.01975, 2022.   
Synnaeve, G., Schatz, T., and Dupoux, E. Phonetics embedding learning with side information. In 2014 IEEE Spoken Language Technology Workshop (SLT), pp. 106–111, 2014. doi: 10.1109/SLT.2014.7078558.   
Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
Vapnik, V. and Vashist, A. A new learning paradigm: Learning using privileged information. Neural networks, 22(5-6):544–557, 2009.   
Vapnik, V., Izmailov, R., et al. Learning using privileged information: similarity control and knowledge transfer. J. Mach. Learn. Res., 16(1):2023–2049, 2015.   
Wang, B., Yue, X., Su, Y., and Sun, H. Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization. arXiv preprint arXiv:2405.15071, 2024.   
Wei, B., Huang, K., Huang, Y., Xie, T., Qi, X., Xia, M., Mittal, P., Wang, M., and Henderson, P. Assessing the brittleness of safety alignment via pruning and low-rank modifications. arXiv preprint arXiv:2402.05162, 2024.   
Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022.   
Wu, Z., Wang, Y., Ye, J., and Kong, L. Self-adaptive in-context learning: An information compression perspective for in-context example selection and ordering. arXiv preprint arXiv:2212.10375, 2022.   
Xu, H., Gao, Y., Yu, F., and Darrell, T. End-to-end learning of driving models from large-scale video datasets. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2174–2182, 2017.   
Yang, L., Zhang, S., Yu, Z., Bao, G., Wang, Y., Wang, J., Xu, R., Ye, W., Xie, X., Chen, W., et al. Supervised knowledge makes large language models better in-context learners. arXiv preprint arXiv:2312.15918, 2023.

Yang, S., Gribovskaya, E., Kassner, N., Geva, M., and Riedel, S. Do large language models latently perform multi-hop reasoning? arXiv preprint arXiv:2402.16837, 2024.   
Yao, S., Peng, B., Papadimitriou, C., and Narasimhan, K. Self-attention networks can process bounded hierarchical languages. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 3770–3785, 2021.   
Yu, D., Kaur, S., Gupta, A., Brown-Cohen, J., Goyal, A., and Arora, S. Skill-mix: A flexible and expandable family of evaluations for ai models. arXiv preprint arXiv:2310.17567, 2023.   
Yu, P., Xu, J., Weston, J., and Kulikov, I. Distilling system 2 into system 1. arXiv preprint arXiv:2407.06023, 2024.   
Zhang, B. and Sennrich, R. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.   
Zhang, R., Nie, F., Guo, M., Wei, X., and Li, X. Joint learning of fuzzy k-means and nonnegative spectral clustering with side information. IEEE transactions on Image Processing, 28(5):2152–2162, 2018.   
Zhao, H., Panigrahi, A., Ge, R., and Arora, S. Do transformers parse while predicting the masked word? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 16513–16542, 2023.   
Zhao, H., Kaur, S., Yu, D., Goyal, A., and Arora, S. Can models learn skill composition from examples? In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.   
Zhong, Z., Liu, Z., Tegmark, M., and Andreas, J. The clock and the pizza: Two stories in mechanistic explanation of neural networks. Advances in neural information processing systems, 36:27223–27250, 2023.   
Zhou, H., Bradley, A., Littwin, E., Razin, N., Saremi, O., Susskind, J., Bengio, S., and Nakkiran, P. What algorithms can Transformers learn? A study in length generalization. arXiv preprint arXiv:2310.16028, 2023.   
Zou, J., Zhou, M., Li, T., Han, S., and Zhang, D. Prompt-intern: Saving inference costs by internalizing recurrent prompt during large language model fine-tuning. arXiv preprint arXiv:2407.02211, 2024.

# Appendix

# A. Overview of the Appendix and Common Notations

Here, we outline the structure for the appendix for easier readability. Due to space constraints, we had to defer a lot of details from the main paper. Appendix B.1 contains details on the experiments on gradient prediction accuracy from Section 5.4. Appendix B.4 shows the details on CoT internalization pipeline that we followed to get an MLT(d, n)-ICL-capable model. Appendix B.5 gives more details on the context-enhanced learning experiments conducted in Section 3.

Appendix C shows mechanistic experiments, designed in Section 3.3, for additional experimental settings. Appendix C.3 present additional details and results for information recovery through querying experiments conducted in Section 4. Appendix D presents additional relevant related works, including discussion on OOD generalization of language models and mechanistic understanding of large transformer architectures. Appendix E discusses few properties on the MLT. Appendix F presents the formal theorems on the computational hardness of learning the translation task and their proofs, which were informally outlined in Section 5.2 in the main paper. Appendix G then presents the formal statements and proofs for context-enhanced learning in the surrogate model, which were informally outlined in Section 5.1 in the main paper. Finally, Appendix H presents the theoretical construction of an ideal transformer that can simulate the surrogate model. We present extensive details on prompts, and examples of training sequences, that we used for context-enhanced learning in Appendix I.

Table 3. Important notations 

<table><tr><td>Scope of Notation</td><td>Symbol</td><td>Description</td></tr><tr><td rowspan="4">General Notations</td><td> $A$ </td><td>A set</td></tr><tr><td> $A$ </td><td>A matrix</td></tr><tr><td> $A^{(j)}$ </td><td>The  $j$ -th column of matrix  $A$ </td></tr><tr><td> $e_k$ </td><td>One-hot embedding vector with 1 at dimension  $k$ </td></tr><tr><td rowspan="6">Language Modeling</td><td> $X$ </td><td>All possible text strings (input space for a causal LM)</td></tr><tr><td> $\mathcal{Y}$ </td><td>All possible distribution over texts (output space for a causal LM)</td></tr><tr><td> $f_\theta$ </td><td>Causal LM parameterized by  $\theta$ </td></tr><tr><td> $g$ </td><td>Language task mapping inputs  $x \in X_g \subset X$  to a distribution  $Y \in \mathcal{Y}$ </td></tr><tr><td> $\ell_{\text{auto}}$ </td><td>Auto-regressive cross entropy loss</td></tr><tr><td> $\text{CURR}_g(x,t)$ </td><td>In-context curriculum for learning task  $g$  with input  $x$  at step  $t$ </td></tr><tr><td rowspan="10">MLT Translation Task</td><td> $d$ </td><td>Depth of translation task</td></tr><tr><td> $n$ </td><td>Number of characters in each alphabet</td></tr><tr><td> $A$ </td><td>An alphabet set</td></tr><tr><td> $s$ </td><td>A sequence in alphabet  $A$ </td></tr><tr><td> $\pi$ </td><td>A phrasebook between 2-tuples in two alphabets, e.g. ( $\pi_1 : A_1^2 \to A_2^2$ ).</td></tr><tr><td> $\mathcal{B}^\pi$ </td><td>Space of all possible phrasebook on  $A_1^2 \to A_2^2$ </td></tr><tr><td> $\Pi$ </td><td>A set of phrasebooks  $\{\pi_i\}$  defining a translation task</td></tr><tr><td> $\text{MLT}_\Pi$ </td><td>Translation task with a set of phrasebooks  $\Pi$ </td></tr><tr><td> $\text{MLT}(d,n)$ </td><td>Family of translation tasks of depth  $d$  and  $n$  characters</td></tr><tr><td> $\text{STR}(\pi)$ </td><td>Descriptive text for a phrasebook  $\pi$  (see Section 2.2)</td></tr><tr><td rowspan="7">Surrogate Model</td><td> $V_i$ </td><td>Embedding matrix representing sequence  $s_i$  (Definition G.1)</td></tr><tr><td> $C_i$ </td><td>In-context information matrix for level  $i$ </td></tr><tr><td> $W_i$ </td><td>Trainable parameter matrix for level  $i$ </td></tr><tr><td> $P_i$ </td><td>Effective translation matrix for level  $i$  (Definition G.6)</td></tr><tr><td>HardMax</td><td>Column-wise hard-max function</td></tr><tr><td>Matrix( $\pi$ )</td><td>Matrix representation of a phrasebook  $\pi$  (Definition G.4)</td></tr><tr><td> $\text{SURR-MLT}_{\{W_i\}_{i=1}^d}$ </td><td>Surrogate model parameterized by  $W_i$ &#x27;s (Definition G.7)</td></tr></table>

# B. Deferred definitions, and experimental details from the main paper

# B.1. Gradient Prediction Accuracy

Our theoretical analysis in Theorem 5.6 was built on the fact that when a translation rule in a phrasebook is dropped by zeroing out a column in an in-context representation $C_{i}$ , the gradient for the corresponding column in $W_{i}$ points to the direction of the dropped rule.

On the other hand, we show that when multiple rules are simultaneously dropped from the phrasebooks, the gradients for the trainable parameters become increasingly noisy. To quantify this degradation, we compute gradient for each column of the trainable parameters for an ICL-capable model, when the corresponding rule is dropped from phrasebooks by zeroing out the relevant column in the in-context representations. We then compute whether the computed gradient points to the right rule. By progressively increasing the number of simultaneously dropped rules, we measure the resulting degradation in the accuracy of the gradient's predictions.

More formally, denote RANDOM-DROP as an operation that takes in column dropping rates per layer $p_{1}, \cdots, p_{d}$ , set of phrasebooks $\Pi^{*}$ , and returns in-context representations $\{C_{i}\}_{i=1}^{d}$ , such that $\boldsymbol{C}_{i}^{(j)} = \text{Matrix}(\pi_{i}^{*})^{(j)}$ (jth columns of $C_{i}$ and $\text{Matrix}(\pi_{i}^{*})$ are equal) with probability $1 - p_{i}$ and 0 otherwise. Then,

Definition B.1. For a set of column dropping rates per layer $p_{1}, \cdots, p_{d}$ , set of phrasebooks $\Pi^{*}$ and $\mathbf{MLT}(d, n)$ -ICL-capable model, predictive accuracy of gradients is defined as

$$
\begin{array}{l} \mathbb {E} _ {\left\{\boldsymbol {C} _ {i} \right\} _ {i = 1} ^ {d}} = \text { RANDOM - DROP } (p _ {1}, \dots , p _ {d}, \boldsymbol {\Pi} ^ {*}) \\ \mathbb {E} _ {j \in [ 1, n ^ {2} ] | \boldsymbol {C} _ {1} ^ {(j)} = 0} \mathbb {I} \left[ \mathrm{HardMax} \left(- \nabla_ {\boldsymbol {W} _ {1} ^ {(j)}} \mathcal {L}\right) = \left(\mathrm{Matrix} \left(\boldsymbol {\pi} _ {i} ^ {*}\right)\right) ^ {(j)} \right] \\ \mathcal {L} = \mathbb {E} _ {\boldsymbol {s} _ {1}} \ell \left(\mathrm{SURR-MLT} _ {\{\boldsymbol {W} _ {i} \} _ {i = 1} ^ {d}} \left(\{\boldsymbol {C} _ {i} \} _ {i = 1} ^ {d}, \boldsymbol {V} _ {1}\right), \mathrm{MLT} _ {\boldsymbol {\Pi} ^ {*}} (\boldsymbol {s} _ {1})\right), \\ \end{array}
$$

where $\ell$ , adapted from Section 2.1, computes cross-entropy loss on the predicted output embeddings of surrogate model using true output string and I denotes the indicator function.

The above definition computes gradients on the expected loss of the model. We primarily focus on predicting the trainable parameters of the first layer, i.e. $W_{1}$ , as that is the deepest layer in the surrogate model and intuitively should suffer the most with noise accumulation from dropped rules. On the other hand, we can further adapt the definition to compute the accuracy for batched gradients, where the gradients are computed using average loss on a randomly sampled batch of input sequences.

In Figure 6, we report the predictive accuracy of gradient for an MLT(d, n)-ICL-capable model, and its behavior with varying batch size and the column dropping rates. We report for two cases, one where column dropping rates is non-zero only for the first phrasebook, and one where we increase the number of phrasebooks for which rules are independently and uniformly dropped. In both cases,

- Increased column dropping rates leads to noisier gradients and reduced prediction accuracy.   
- Larger batch sizes improve gradient accuracy but cannot fully compensate for high dropout rates.   
- Dropping rules from multiple phrasebooks significantly degrades gradient prediction accuracy.

# B.2. Hidden Representations of a Model

A transformer $f_{\theta}$ with embedding dimension p and K layers takes any input sequence x, say of length L, converts to an embedding matrix $H_{1} \in R^{L \times p}$ , and modifies the embeddings using a succession of K transformer layers; which we will denote by $f_{\theta}^{(1)}$ , $f_{\theta}^{(2)}$ , $\cdots$ , $f_{\theta}^{(K)}$ . We refer to the hidden representations for the input x, with embedding matrix $H_{1}$ , as the output of the model after every layer. We will denote them as $H_{i+1} \in R^{L \times p}$ for the output of layer $f_{\theta}^{(i)}$ . That is,

$$
\boldsymbol {H} _ {i + 1} = f _ {\theta} ^ {(i)} \circ \dots \circ f _ {\theta} ^ {(2)} \circ f _ {\theta} ^ {(1)} (\boldsymbol {H} _ {1}), \quad \text { for   all } i \geq 1.
$$

$\ell_{2}$ -norm in change in hidden representation with perturbation in in-context information For an MLT(d,n)-ICL-capable model, we supply in-context information for the textual description of a set of phrasebooks $\Pi$ as $\mathrm{STR}(\Pi) = [\mathrm{STR}(\pi_{1}),\cdots,\mathrm{STR}(\pi_{d})]$ . Our inputs to the transformer for an input string $s_{1}$ will be of the form $[\mathrm{STR}(\Pi),s_{1},<\mathrm{THINK}>,\ldots,s_{d+1}]$ , where $s_{d+1} = MLT_{\Pi}(s_{1})$ . By the definition of hidden representations, $H_{2},\cdots,H_{K+1}$ will denote the output of the transformer layers for this input string. However, we will be only interested in the hidden representations for the tokens involved in the tokens for $<THINK>$ , $\ldots$ , $s_{d+1}$ ; and we will refer to the corresponding subsets of $H_{2},\cdots,H_{K+1}$ that represent these specific tokens as $V_{2},\cdots,V_{K+1}$ .

Now, suppose we randomly take a phrasebook $\pi_{i}$ in $\Pi$ and change to a random phrasebook $\tilde{\pi}_{i}$ . The corresponding textual description that will augment the context for an input string will then be $[\mathrm{STR}(\pi_{1}),\cdots,\mathrm{STR}(\pi_{i-1}),\mathrm{STR}(\tilde{\pi}_{i}),\mathrm{STR}(\pi_{i+1}),\cdots,\mathrm{STR}(\pi_{d})]$ . If $\tilde{V}_{2},\cdots,\tilde{V}_{K+1}$ now denote the hidden representations that represent the tokens for <THINK>, $\ldots,s_{d+1}$ , then the $\ell_{2}$ -norm in the change of the hidden representation after layer j (for any $1\leq j\leq K$ ) with the perturbation in $\Pi$ will be given by $\left\|\tilde{V}_{j}-V_{j}\right\|_{2}$ .

# B.3. Definition of a “stitched” model

We reuse notations from Appendix B.2. Suppose we have an $\mathbf{MLT}(d,n)$ -ICL-capable model $f_{\theta}$ and an $MLT_{\Pi^{*}}$ -capable model $f_{\theta^{*}}$ . Their corresponding transformer layers are denoted by $f_{\theta}^{(1)}, f_{\theta}^{(2)}, \cdots, f_{\theta}^{(K)}$ and $f_{\theta^{*}}^{(1)}, f_{\theta^{*}}^{(2)}, \cdots, f_{\theta^{*}}^{(K)}$ . Each model takes in an input sequence and processes them with their K transformer layers.

Formally, we will write for the ICL-capable model. It takes in input sequence x, and converts to an embedding matrix, say $H_{1}$ , and the output after the K layers are given by:

$$
\boldsymbol {H} _ {K + 1} = f _ {\theta} ^ {(d)} \circ \dots \circ f _ {\theta} ^ {(2)} \circ f _ {\theta} ^ {(1)} (\boldsymbol {H} _ {1}).
$$

Process of “stitching”: The process of stitching takes in two parameters $L_{start}$ and $L_{end}$ and replaces all layers from $L_{start}$ to $L_{end}$ in $f_{\theta}$ with the corresponding layers in $f_{\theta^{*}}$ to give a “stitched” model, say $f_{\theta,\theta^{*},L_{start},L_{end}}$ . The output of the “stitched” model $f_{\theta,\theta^{*},L_{start},L_{end}}$ on an input sequence x will be given by

$$
\boldsymbol {H} _ {K + 1} = f _ {\theta} ^ {(d)} \circ \dots \circ f _ {\theta} ^ {(L _ {\text {end + 1}})} \circ \underbrace {f _ {\theta^ {*}} ^ {(L _ {\text {end}})} \circ \cdots f _ {\theta^ {*}} ^ {(L _ {\text {start}})}} _ {\text {Layers are replaced by layers from} f _ {\theta^ {*}}} \circ f _ {\theta} ^ {(L _ {\text {start - 1}})} \circ f _ {\theta} ^ {(2)} \circ f _ {\theta} ^ {(1)} (\boldsymbol {H} _ {1}).
$$

# B.4. Pipeline on CoT Internalization

We randomly sample M sets of phrasebooks $\pi_{1},\ldots,\pi_{M}$ not equal to $\Pi^{*}$ . For each set of phrasebooks $\pi_{i}$ , we randomly sample a single input sequence $s_{1}$ and compute all the intermediate translation steps $s_{2},\ldots,s_{d+1}$ . We first train the model to do robust explicit CoT by auto-regressive training on sequences $[\mathrm{STR}(\pi_{i}),s_{1},s_{2},\ldots,s_{d},s_{d+1}]$ with loss computed over $s_{2},\ldots,s_{d+1}$ . Then we follow common CoT internalization strategies (Deng et al., 2024; Hao et al., 2024; Yu et al., 2024; Su et al., 2024) and gradually replace the intermediate sequences by <THINK> tokens in training. After all intermediate sequences have been replaced, the model has low loss on $s_{d+1}$ with input $[\mathrm{STR}(\Pi_{1}),s_{1},<THINK>,\ldots,<THINK>,s_{d+1}]$ , satisfying Definition 2.2. Since we only sample on sequence per set of phrasebooks, there is little memorization on particular phrasebooks.

Details on training hyperparameters: We use $M = 3 \times 10^{5}$ random sets of phrasebooks with length between 20 and 40, for getting MLT(5, 8)-ICL-capable model, and $M = 10^{6}$ random sets of phrasebooks with length between 20 and 40, for getting MLT(5, 10)-ICL-capable model. We use cosine learning rate schedule (Loshchilov & Hutter, 2016), with peak

learning rate $10^{-4}$ and a 6% warmup phase, where learning rate is linearly increased from 0 to the peak. We use AdamW optimizer (Loshchilov & Hutter, 2019) with weight decay fixed at $10^{-4}$ . We use a batch size of 64 for training.

CoT internalization curriculum: For the first 10% fraction of training, we train the model with explicit CoT tokens that contain the intermediate steps in translation. Then between 10% to 60% fractions of training, CoT tokens are gradually replaced by <THINK> tokens, with the rate of replacement increasing linearly from 0% to 100%. We follow a deterministic first-to-last order for replacing CoT tokens; earlier CoT tokens are replaced first with <THINK> tokens. After that, the model is trained with the <THINK> CoT tokens till the end of training.

# B.5. Experiment configuration for context-enhanced learning

For context-enhanced learning, we create supervised datasets $D_{\Pi^{*}}$ of different sizes; each containing between $10^{4}$ to $10^{6}$ samples. When performing Annealing Dropout or Fixed Dropout, at each step of training, we randomly perform dropout on the rules of all phrasebooks or apply dropout to the rules of a randomly sampled phrasebook to define curriculum-text. That is, if $\pi_{1}^{*},\ldots,\pi_{5}^{*}$ represent the phrasebooks, then at each step of training, we either randomly drop rules uniformly from all of $\pi_{1}^{*},\ldots,\pi_{5}^{*}$ , or just drop from one of the phrasebooks randomly selected from $\pi_{1}^{*},\ldots,\pi_{5}^{*}$ , while keeping the rules of all other phrasebooks intact, to create curriculum-text.

Hyperparameters: Training hyperparameters are set equal to the optimization hyperparameters used in preparation of ICL-capable training phase (Appendix B.4), except we set weight decay to 0 in all experiments. We report the performance of the trained model after single epoch of training on each $D_{\Pi^*}$ and plot against the size of the dataset in Figure 2.

# C. Additional Experiments Results

# C.1. Selective Internalization of Context

In this experiment, we test whether the model can internalize rules that don't incur an increase in loss when dropped during training. To do so, we ablate on Annealing Dropout. We select a phrasebook and create 2 splits of rules in the phrasebook; one set of rules will be utilized by training samples for performing the translation task (which we call the training split), while other set of rules will appear in curriculum-text during training but never utilized for the translation task (which we call the heldout split).

At test time, we measure the performance of the trained model on 2 sets of evaluation examples, one that only uses rules from the training split for their translation (equal to training distribution), and other that uses rules only from the heldout split for their translation (different from training distribution). The model is being measured without any phrasebooks information at evaluation. We conduct the above experiment for each phrasebook, i.e. we create 5 sets of experiments where we only create heldout split for one specific level of phrasebook. In Figure 7, we show that the model fails to perform any translation that use the rules from the held-out split in all settings. This shows that the model only internalizes those rules that are important for the translation task for the training samples, and which incurs an increase in training loss when dropped.

Only with rules used in training examples Without any rules used in training examples

![](images/01306c3f5ccfbd8e76e79e0c27e59a6e8d2482f5da18d8c58eaffca98a6e56c3.jpg)

<details>
<summary>line</summary>

| Number of Samples | Test Accuracy (without context) |
| ----------------- | -------------------------------- |
| 10^4              | 0.00                             |
| 10^5              | 1.00                             |
| 10^6              | 1.00                             |
</details>

![](images/af620bc61e93ee23841698ad63d99bc6d5dc68f2f7fc2fd9756c4dd69bfc7562.jpg)

<details>
<summary>line</summary>

| Number of Samples |Holding out 1/2 of π₁ |
| ----------------- | ---------------------- |
| 10^4              | 0                      |
| 10^5              | 1.0                    |
| 10^6              | 1.0                    |
</details>

![](images/6fd6accc2fa8ea80e75dd127c46b8491cdc31efcfbf5a87a61291f1347ce4f9e.jpg)

![](images/52879e16265585294ed1bb24469744a344823ed20df827d5e0aaebd30e1fe611.jpg)

<details>
<summary>line</summary>

| Number of Samples | Holding out 1/2 of π₃ |
| ----------------- | --------------------- |
| 10⁴               | 0                     |
| 10⁵               | 1/2                   |
| 10⁶               | 1/2                   |
</details>

![](images/e5c21164bd5ad53235fdc0d6799faf718230911e2a6ef36830fdbc3b1ca9ffc2.jpg)

<details>
<summary>line</summary>

| Number of Samples |Dπ⁻¹ |
| ----------------- | ---- |
| 10⁴               | 0.5  |
| 10⁵               | 1.0  |
| 10⁶               | 1.0  |
</details>

Figure 7. Ablations on Annealing Dropout for MLT(5, 10) assess whether models internalize rules not used in training samples. (Left to right) $1 \leq i \leq 5$ : Five independent experiments where a held-out split is created for phrasebook $\pi_i^*$ , and training excludes samples that use the translation rules in the heldout split of $\pi_i^*$ . Evaluation is conducted on two sets: one where samples do not use held-out rules from $\pi_i^*$ and one where only held-out rules from $\pi_i^*$ are used. The model's random performance on the latter indicates that it internalizes only rules used during training, particularly those whose removal increases loss.

# C.2. Mechanistic Insights

Here we provide additional figures corresponding to Figure 3 and Figure 4, but in more settings (n = 10 vs n = 8, Annealing Dropout vs Fixed Dropout).

![](images/81184eea00acec78d87f24dbf2b8462564a793459f0f2a87743d000db2d8f9c8.jpg)  
Figure 8. Comparison between perturbing used rules (top row, identical to Figure 3) and unused rules (bottom row) in context. When perturbing rules used in the translation at a later phrasebook, representation changes at later layers. However only perturbing unused rules leads to negligible representation changes. This experiment rules out the possibility that the affected depth is only dependent on position of the perturbation instead of the semantic content.

![](images/4c4d04385c1797193130b950520b36b8f71b6fc0cf4a4d9784de1f39772dcb10.jpg)

<details>
<summary>heatmap</summary>

| Layer index | 0    | 5    | 10   | 15   | 20   | 25   |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Row 1       | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 2       | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 3       | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 4       | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 5       | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 6       | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 7       | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 8       | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 9       | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 10      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 11      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 12      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 13      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 14      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 15      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 16      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 17      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 18      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 19      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 20      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 21      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 22      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 23      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 24      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 25      | 0    | 0    | 0    | 0    | 0    | 0    |
| Row 26      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 27      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 28      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 29      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 30      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 31      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 32      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 33      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 34      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 35      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 36      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 37      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 38      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 39      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 40      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 41      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 42      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 43      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 44      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 45      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 46      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 47      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 48      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 49      | -1   | -1   | -1   | -1   | -1   | -1   |
| Row 50      | -2   +2| +2+2 +2| +2+2 +2| +2+2 +2| +2+2 +2| +2+2 +2|
| Row_5     [Layer index] = Row_4; Row_5 = Row_6; Row_7 = Row_8; Row_9 = Row_9; Row_4 = Row_5; Row_6 = Row_7; Row_8 = Row_8; Row_9 = Row_9. The values in the table represent the perturbation magnitude for each layer.
</details>

![](images/a8cfc16636a947d3f0821009c10d96556d4fd9897e68c780716818a27e4fe1f7.jpg)

<details>
<summary>heatmap</summary>

| Layer index | 0    | 5    | 10   | 15   | 20   | 25   |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0           | 0    | 0    | 0    | 0    | 0    | 0    |
| 5           | 0    | 0    | 0    | 0    | 0    | 0    |
| 10          | 0    | 0    | 0    | 0    | 0    | 0    |
| 15          | 0    | 0    | 0    | 0    | 0    | 0    |
| 20          | 0    | 0    | 0    | 0    | 0    | 0    |
| 25          | 0    | 0    | 0    | 0    | 0    | 0    |
</details>

![](images/cbcf9892ea961fa33c1be8d8e6a431d009328c2964d1c6c8d1ee6df27416e506.jpg)

<details>
<summary>heatmap</summary>

| Layer index | 0    | 5    | 10   | 15   | 20   | 25   |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0           | 0    | 0    | 0    | 0    | 0    | 0    |
| 5           | 0    | 0    | 0    | 0    | 0    | 0    |
| 10          | 0    | 0    | 0    | 0    | 0    | 0    |
| 15          | 0    | 0    | 0    | 0    | 0    | 0    |
| 20          | 0    | 0    | 0    | 0    | 0    | 0    |
| 25          | 0    | 0    | 0    | 0    | 0    | 0    |
</details>

![](images/e45e0c3d8dadf3bdb193a11186b970dc761671080de290f5b91705fa094952bd.jpg)

<details>
<summary>heatmap</summary>

| Layer index | 0    | 5    | 10   | 15   | 20   | 25   |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0           | 0    | 0    | 0    | 0    | 0    | 0    |
| 5           | 0    | 0    | 0    | 0    | 0    | 0    |
| 10          | 0    | 0    | 0    | 0    | 0    | 0    |
| 15          | 0    | 0    | 0    | 0    | 0    | 0    |
| 20          | 0    | 0    | 0    | 0    | 0    | 0    |
| 25          | 0    | 0    | 0    | 0    | 0    | 0    |
</details>

![](images/c5fa24682d10e5caf3053caa7d7999d4c904b5ceb3d3d3f54ddbbec9ff5f76b6.jpg)

<details>
<summary>heatmap</summary>

| Layer index | 0    | 5    | 10   | 15   | 20   | 25   |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0           | 0    | 0    | 0    | 0    | 0    | 0    |
| 5           | 0    | 0    | 0    | 0    | 0    | 0    |
| 10          | 0    | 0    | 0    | 0    | 0    | 0    |
| 15          | 0    | 0    | 0    | 0    | 0    | 0    |
| 20          | 0    | 0    | 0    | 0    | 0    | 0    |
| 25          | 0    | 0    | 0    | 0    | 0    | 0    |
</details>

![](images/4d56807bfd92b8377af6eda3d4663adfa681c7d96558c04ccf683d6b65060fe7.jpg)

<details>
<summary>heatmap</summary>

| Layer index | 0    | 5    | 10   | 15   | 20   | 25   |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0           | 0    | 0    | 0    | 0    | 0    | 0    |
| 5           | 0    | 0    | 0    | 0    | 0    | 0    |
| 10          | 0    | 0    | 0    | 0    | 0    | 0    |
| 15          | 0    | 0    | 0    | 0    | 0    | 0    |
| 20          | 0    | 0    | 0    | 0    | 0    | 0    |
| 25          | 0    | 0    | 0    | 0    | 0    | 0    |
</details>

![](images/fc2b92312bac87f0f6f869fbedf8722425313a675aa0c1b15c69a006dbbf5d26.jpg)

<details>
<summary>heatmap</summary>

| Layer index | Value |
| ----------- | ----- |
| 0           | 0     |
| 5           | 0     |
| 10          | 0     |
| 15          | 0     |
| 20          | 0     |
| 25          | 0     |
</details>

![](images/f042a5237128da7ca5fc478887eb5eaf1e8840861b8605e8bcd48c33a6601409.jpg)

<details>
<summary>heatmap</summary>

| Layer index | Value |
| ----------- | ----- |
| 0           | 0     |
| 5           | 0     |
| 10          | 0     |
| 15          | 0     |
| 20          | 0     |
| 25          | 0     |
</details>

![](images/dffb46398df0bf18ad6d463dd26984af6a10c498eca29bad496199f3c04c6622.jpg)

<details>
<summary>heatmap</summary>

The translation result is: Sequence in language 1: E
H C A F F G
HD H D B
D R A E H F
H B G A E
H B G A E
H F F G A:
Sequence in language 2:
<T><T><T><T><T><T><T><T>
<T><T><T><T><T><T><T><T>
<T><T><T><T><T><T><T>
<T><T><T><T><T><T><T>
<T><T><T><T><T><T><T>
<T><T><T><T><T><T><T>
<T><T><T><T><T><T><T>
<T><T><T><T><T><T>
<T><T><T><T><T><T>
<T><T><T><T><T><T>
<T><T><T><T><T><T>
<T><T><T><T><T><T>
<T><T><T><T><T><T>
<T><T><T><T><T><T>
<T><T><T><T><T><T>
<T><T><R: Sequence in language 3: T
Language 4: Sequence in language 4: T
Language 5: Sequence in language 5: T
Language 6: Sequence in language 6: t
Language 7: Sequence in language 7: T
Language 8: Sequence in language 8: T
Language 9: Sequence in language 9: T
Language 10: Sequence in language 10: T
Language 11: Sequence in language 11: T
Language 12: Sequence in language 12: T
Language 13: Sequence in language 13: T
Language 14: Sequence in language 14: T
Language 15: Sequence in language 15: T
Language 16: Sequence in language 16: T
Language 17: Sequence in language 17: T
Language 18: Sequence in language 18: T
Language 19: Sequence in language 19: T
Language 20: Sequence in language 20: T
Language 21: Sequence in language 21: T
Language 22: Sequence in language 22: T
Language 23: Sequence in language 23: T
Language 24: Sequence in language 24: T
Language 25: Sequence in language 25: T
Language 26: Sequence in language 26: T
Language 27: Sequence in language 27: T
Language 28: Sequence in language 28: T
Language 29: Sequence in language 29: T
Language 30: Sequence in language 30: T
Language 31: Sequence in language 31: T
Language 32: Sequence in language 32: T
Language 33: Sequence in language 33: T
Language 34: Sequence in language 34: T
Language 35: Sequence in language 35: T
Language 36: Sequence in language 36: T
Language 37: Sequence in language 37: T
Language 38: Sequence in language 38: T
Language 39: Sequence in language 39: T
Language 40: Sequence in language 40: T
Language 41: Sequence in language 41: T
Language 42: Sequence in language 42: T
Language 43: Sequence in language 43: T
Language 44: Sequence in language 44: T
Language 45: Sequence in language 45: T
Language 46: Sequence in language 46: T
Language 47: Sequence in language 47: T
Language 48: Sequence in language 48: T
Language 49: Sequence in language 49: T
Language 50: Sequence in language 50: T
Language 51: Sequence in language 51: T
Language 52: Sequence in language 52: T
Language 53: Sequence in language 53: T
Language 54: Sequence in language 54: T
Language 55: Sequence in language 55: T
Language 56: Sequence in language 56: T
Language 57: Sequence in language 57: T
Language 58: Sequence in language 58: T
Language 59: Sequence in language 59: T
Language 60: Sequence in language 60: T
Language 61: Sequence in language 61: T
Language 62: Sequence in language 62: T
Language 63: Sequence in language 63: T
Language 64: Sequence in language 64: T
Language 65: Sequence in language 65: T
Language 66: Sequence in language 66: T
Language 67: Sequence in language 67: T
Language 68: Sequence in language 68: T
Language 69: Sequence in language 69: T
Language 70: Sequence in language 70: T
Language 71: Sequence in language 71: T
Language 72: Sequence in language 72: T
Language 73: Sequence in language 73: T
Language 74: Sequence in language 74: T
Language 75: Sequence in language 75: T
Language 76: Sequence in language 76: T
Language 77: Sequence in language 77: T
Language 78: Sequence in language 78: T
Language 79: Sequence in language 79: T
Language 80: Sequence in language 80: T
Language 81: Sequence in language 81: T
Language 82: Sequence in language 82: T
Language 83: Sequence in language 83: T
Language 84: Sequence in language 84: T
Language 85: Sequence in language 85: T
Language 86: Sequence in language 86: T
Language 87: Sequence in language 87: T
Language 88: Sequence in language 88: T
Language 89: Sequence in language 89: T
Language 90: Sequence in language 90: T
Language 91: Sequence in language 91: T
Language 92: Sequence in language 92: T
Language 93: Sequence in language 93: T
Language 94: Sequence in language 94: T
Language 95: Sequence in language 95: T
Language 96: Sequence in language 96: T
Language 97: Sequence in language 97: T
Language 98: Sequence in language 98: T
Language 99: Sequence in language 99: T
Vp p:
q p s s v r u p q p v r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r<nl>
</details>

![](images/8508ba02e03922905227cf99b16ebeecb669c8f8e05d6ec64b4c482bf7803706.jpg)

<details>
<summary>heatmap</summary>

The translation result is: Sequence in language 1: E, GJJDAD, CBABDB, DBBGHE, GCCGCE, AGDAC; Sequence in language 2: <T>: Sequence in language 3: T: Sequence in language 4: T: Sequence in language 5: T: Sequence in language 6: T: Sequence in language 7: T: Sequence in language 8: T: Sequence in language 9: T: Sequence in language 10: T: Sequence in language 11: T: Sequence in language 12: T: Sequence in language 13: T: Sequence in language 14: T: Sequence in language 15: T: Sequence in language 16: T: Sequence in language 17: T: Sequence in language 18: T: Sequence in language 19: T: Sequence in language 20: T: Sequence in language 21: T: Sequence in language 22: T: Sequence in language 23: T: Sequence in language 24: T: Sequence in language 25: T: Sequence in language 26: T: Sequence in language 27: T: Sequence in language 28: T: Sequence in language 29: T: Sequence in language 30: T: Sequence in language 31: T: Sequence in language 32: T: Sequence in language 33: T: Sequence in language 34: T: Sequence in language 35: T: Sequence in language 36: T: Sequence in language 37: T: Sequence in language 38: T: Sequence in language 39: T: Sequence in language 40: T: Sequence in language 41: T: Sequence in language 42: T: Sequence in language 43: T: Sequence in language 44: T: Sequence in language 45: T: Sequence in language 46: T: Sequence in language 47: T: Sequence in language 48: T: Sequence in language 49: T: Sequence in language 50: T: Sequence in language 51: T: Sequence in language 52: T: Sequence in language 53: T: Sequence in language 54: T: Sequence in language 55: T: Sequence in language 56: T: Sequence in language 57: T: Sequence in language 58: T: Sequence in language 59: T: Sequence in language 60: T: Sequence in language 61: T: Sequence in language 62: T: Sequence in language 63: T: Sequence in language 64: T: Sequence in language 65: T: Sequence in language 66: T: Sequence in language 67: T: Sequence in language 68: T: Sequence in language 69: T: Sequence in language 70: T: Sequence in language 71: T: Sequence in language 72: T: Sequence in language 73: T: Sequence in language 74: T: Sequence in language 75: T: Sequence in language 76: T: Sequence in language 77: T: Sequence in language 78: T: Sequence in language 79: T: Sequence in language 80: T: Sequence in language 81: T: Sequence in language 82; Y = α β γ β ΔZ; X = β γ γ β; Legend indicates values ranging from ~0 to ~50. The chart visualizes the distribution of perturbation indices across different translation sequences.
</details>

(a) MLT(5, 8)-ICL-capable Model   
(b) MLT(5, 10)-ICL-capable Model   
Figure 9. Evidence for sequential processing in MLT-ICL capable models $(n = 8, n = 10)$ . The left figure is identical to Figure 3, except we substitute the entire phrasebook $\pi_{i}$ with another random phrasebook of the same level $\hat{\pi}_{i}$ . We observe the same behavior for MLT(5, 10)-ICL-capable model: perturbing later phrasebooks in the context changes output representations in the later layers.

![](images/df2f469353d20b908205f55909d338a1dfd018d451cb0f245aff86a2e8780fd8.jpg)

<details>
<summary>heatmap</summary>

| L_end | L_start | Evaluation Accuracy |
|-------|---------|---------------------|
| 1     | 1       | 0.0                 |
| 4     | 4       | 0.0                 |
| 7     | 7       | 0.0                 |
| 10    | 10      | 0.0                 |
| 13    | 13      | 0.0                 |
| 16    | 16      | 0.0                 |
| 19    | 19      | 0.0                 |
| 22    | 22      | 0.0                 |
| 25    | 25      | 0.0                 |
| 28    | 28      | 0.0                 |
</details>

(a) $D_{\Pi^{*}}$ with 10000 samples

![](images/bb141598f5636936f120f2cd9f47a7c69c084521c6ed7d8e6a0970780f56d086.jpg)  
(b) $D_{\Pi^{*}}$ with 25000 samples

![](images/3d8616735b0a3c2ba066c4f3bd3bb49e4f2d897bd37520e393cc0dd2b4bf0cae.jpg)

<details>
<summary>heatmap</summary>

| L_end | L_start | Evaluation Accuracy |
|-------|---------|---------------------|
| 1     | 1       | 0.0                 |
| 4     | 4       | 0.2                 |
| 7     | 7       | 0.4                 |
| 10    | 10      | 0.6                 |
| 13    | 13      | 0.8                 |
| 16    | 16      | 1.0                 |
| 19    | 19      | 0.8                 |
| 22    | 22      | 0.6                 |
| 25    | 25      | 0.4                 |
| 28    | 28      | 0.2                 |
</details>

(c) $D_{\Pi^{*}}$ with 50000 samples

![](images/6f833e014cbce7257df1b6e08808ecd0846a6207e5e9cce3f80a3c3ce7b6d656.jpg)

<details>
<summary>heatmap</summary>

| L_end | Lstart | Evaluation Accuracy |
|-------|--------|---------------------|
| 1     | 1      | 0.0                 |
| 4     | 4      | 0.2                 |
| 7     | 7      | 0.4                 |
| 10    | 10     | 0.6                 |
| 13    | 13     | 0.8                 |
| 16    | 16     | 1.0                 |
| 19    | 19     | 0.8                 |
| 22    | 22     | 0.6                 |
| 25    | 25     | 0.4                 |
| 28    | 28     | 0.2                 |
</details>

(d) $D_{\Pi^{*}}$ with 100000 samples

![](images/c1b7a4dff5ca1dd2bfbc54ace7d2193ca4165a65e6cfbc18039a1002a70dbc27.jpg)

<details>
<summary>heatmap</summary>

| Droping STR(π₁*) | L_end | L_start | Evaluation Accuracy |
| ---------------- | ----- | ------- | ------------------- |
| π₁*              | 1     | 1       | 0.0                 |
| π₁*              | 4     | 7       | 0.2                 |
| π₁*              | 7     | 10      | 0.4                 |
| π₁*              | 10    | 13      | 0.6                 |
| π₁*              | 13    | 16      | 0.8                 |
| π₁*              | 16    | 19      | 1.0                 |
| π₁*              | 19    | 22      | 0.8                 |
| π₁*              | 22    | 25      | 0.6                 |
| π₁*              | 25    | 28      | 0.4                 |
| π₂*              | 1     | 1       | 0.0                 |
| π₂*              | 4     | 7       | 0.2                 |
| π₂*              | 7     | 10      | 0.4                 |
| π₂*              | 10    | 13      | 0.6                 |
| π₂*              | 13    | 16      | 0.8                 |
| π₂*              | 16    | 19      | 1.0                 |
| π₂*              | 19    | 22      | 0.8                 |
| π₂*              | 22    | 25      | 0.6                 |
| π₂*              | 25    | 28      | 0.4                 |
| π₃*              | 1     | 1       | 0.0                 |
| π₃*              | 4     | 7       | 0.2                 |
| π₃*              | 7     | 10      | 0.4                 |
| π₃*              | 10    | 13      | 0.6                 |
| π₃*              | 13    | 16      | 0.8                 |
| π₃*              | 16    | 19      | 1.0                 |
| π₃*              | 19    | 22      | 0.8                 |
| π₃*              | 22    | 25      | 0.6                 |
| π₃*              | 25    | 28      | 0.4                 |
| π₄*              | 1     | 1       | 0.0                 |
| π₄*              | 4     | 7       | 0.2                 |
| π₄*              | 7     | 10      | 0.4                 |
| π₄*              | 10    | 13      | 0.6                 |
| π₄*              | 13    | 16      | 0.8                 |
| π₄*              | 16    | 19      | 1.0                 |
| π₄*              | 19    | 22      | 0.8                 |
| π₄*              | 22    | 25      | 0.6                 |
| π₄*              | 25    | 28      | 0.4                 |
| π₅*              | 1     | 1       | 0.0                 |
| π₅*              | 4     | 7       | 0.2                 |
| π₅*              | 7     | 10      | 0.4                 |
| π₅*              | 10    | 13      | 0.6                 |
| π₅*              | 13    | 16      | 0.8                 |
| π₅*              | 16    | 19      | 1.0                 |
| π₅*              | 19    | 22      | 0.8                 |
| π₅*              | 22    | 25      | 0.6                 |
| π₅*              | 25    | 28      | 0.4                 |
</details>

(e) $D_{\Pi^{*}}$ with 250000 samples   
Figure 10. Repeated experiments from Figure 4 for models trained with context-enhanced learning at varying supervised dataset sizes (Plots for row (d) are identical to the plots in Figure 4). We observe that phrasebooks are progressively internalized with the number of training samples available during context-enhanced learning; later phrasebooks are internalized with fewer samples than the earlier ones. We observe similar localization patterns across layers from different phrasebooks across the models, however, we also observe that the localization patterns get increasingly sparser as training continues.

![](images/abbd81d6226dc74ca2169ae7358f7117d6eb758b4cc30a4e4cc0092f3dc7b03d.jpg)

<details>
<summary>heatmap</summary>

| L_end | L_start | Dropping STR(π₁*) | Dropping STR(π₂*) | Dropping STR(π₃*) | Dropping STR(π₄*) | Dropping STR(π₅*) |
|-------|---------|-------------------|-------------------|-------------------|-------------------|-------------------|
| 1     | 1       | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
| 4     | 4       | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
| 7     | 7       | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
| 10    | 10      | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
| 13    | 13      | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
| 16    | 16      | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
| 19    | 19      | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
| 22    | 22      | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
| 25    | 25      | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
| 28    | 28      | 0.0               | 0.0               | 0.0               | 0.0               | 0.0               |
</details>

(a) $D_{\Pi^{*}}$ with 25000 samples

![](images/27a15d42e6ee6ccbfbd0ace2dd6b30e54ce06462e255e5d8ebe6de5aa80ee7e7.jpg)

<details>
<summary>heatmap</summary>

| L_end | L_start | Evaluation Accuracy |
|-------|---------|---------------------|
| 1     | 1       | 0.0                 |
| 4     | 4       | 0.2                 |
| 7     | 7       | 0.4                 |
| 10    | 10      | 0.6                 |
| 13    | 13      | 0.8                 |
| 16    | 16      | 1.0                 |
| 19    | 19      | 0.8                 |
| 22    | 22      | 0.6                 |
| 25    | 25      | 0.4                 |
| 28    | 28      | 0.2                 |
</details>

(b) $D_{\Pi^{*}}$ with 50000 samples

![](images/d6291b6f8eea8a836aeaedc8179b6738d4250c5bddd9d3a23af42e904f51cb9a.jpg)

<details>
<summary>heatmap</summary>

| Droping STR(π₁*) | L_end | L_start | Evaluation Accuracy |
| ---------------- | ----- | ------- | ------------------- |
| π₂*              | 1     | 1       | 0.0                 |
| π₂*              | 4     | 4       | 0.2                 |
| π₂*              | 7     | 7       | 0.4                 |
| π₂*              | 10    | 10      | 0.6                 |
| π₂*              | 13    | 13      | 0.8                 |
| π₂*              | 16    | 16      | 1.0                 |
| π₂*              | 19    | 19      | 0.8                 |
| π₂*              | 22    | 22      | 0.6                 |
| π₂*              | 25    | 25      | 0.4                 |
| π₂*              | 28    | 28      | 0.2                 |
| π₃*              | 1     | 1       | 0.0                 |
| π₃*              | 4     | 4       | 0.2                 |
| π₃*              | 7     | 7       | 0.4                 |
| π₃*              | 10    | 10      | 0.6                 |
| π₃*              | 13    | 13      | 0.8                 |
| π₃*              | 16    | 16      | 1.0                 |
| π₃*              | 19    | 19      | 0.8                 |
| π₃*              | 22    | 22      | 0.6                 |
| π₃*              | 25    | 25      | 0.4                 |
| π₃*              | 28    | 28      | 0.2                 |
| π₄*              | 1     | 1       | 0.0                 |
| π₄*              | 4     | 4       | 0.2                 |
| π₄*              | 7     | 7       | 0.4                 |
| π₄*              | 10    | 10      | 0.6                 |
| π₄*              | 13    | 13      | 0.8                 |
| π₄*              | 16    | 16      | 1.0                 |
| π₄*              | 19    | 19      | 0.8                 |
| π₄*              | 22    | 22      | 0.6                 |
| π₄*              | 25    | 25      | 0.4                 |
| π₄*              | 28    | 28      | 0.2                 |
| π₅*              | 1     | 1       | 0.0                 |
| π₅*              | 4     | 4       | 0.2                 |
| π₅*              | 7     | 7       | 0.4                 |
| π₅*              | 10    | 10      | 0.6                 |
| π₅*              | 13    | 13      | 0.8                 |
| π₅*              | 16    | 16      | 1.0                 |
| π₅*              | 19    | 19      | 0.8                 |
| π₅*              | 22    | 22      | 0.6                 |
| π₅*              | 25    | 25      | 0.4                 |
| π₅*              | 28    | 28      | 0.2                 |
</details>

(c) $D_{\Pi^{*}}$ with 100000 samples

![](images/a316291fa637e2e155696eabeee2c78eed5834d4089b9d7a22024efbdd7d7517.jpg)

<details>
<summary>heatmap</summary>

| Droppling STR(π₁*) | L_end | L_start | Evaluation Accuracy |
| ------------------ | ----- | ------- | ------------------- |
| π₁*                | 1     | 1       | 0.0                 |
| π₁*                | 4     | 4       | 0.2                 |
| π₁*                | 7     | 7       | 0.4                 |
| π₁*                | 10    | 10      | 0.6                 |
| π₁*                | 13    | 13      | 0.8                 |
| π₁*                | 16    | 16      | 1.0                 |
| π₁*                | 19    | 19      | 0.8                 |
| π₁*                | 22    | 22      | 0.6                 |
| π₁*                | 25    | 25      | 0.4                 |
| π₁*                | 28    | 28      | 0.2                 |
| π₂*                | 1     | 1       | 0.0                 |
| π₂*                | 4     | 4       | 0.2                 |
| π₂*                | 7     | 7       | 0.4                 |
| π₂*                | 10    | 10      | 0.6                 |
| π₂*                | 13    | 13      | 0.8                 |
| π₂*                | 16    | 16      | 1.0                 |
| π₂*                | 19    | 19      | 0.8                 |
| π₂*                | 22    | 22      | 0.6                 |
| π₂*                | 25    | 25      | 0.4                 |
| π₂*                | 28    | 28      | 0.2                 |
| π₃*                | 1     | 1       | 0.0                 |
| π₃*                | 4     | 4       | 0.2                 |
| π₃*                | 7     | 7       | 0.4                 |
| π₃*                | 10    | 10      | 0.6                 |
| π₃*                | 13    | 13      | 0.8                 |
| π₃*                | 16    | 16      | 1.0                 |
| π₃*                | 19    | 19      | 0.8                 |
| π₃*                | 22    | 22      | 0.6                 |
| π₃*                | 25    | 25      | 0.4                 |
| π₃*                | 28    | 28      | 0.2                 |
| π₄*                | 1     | 1       | 0.0                 |
| π₄*                | 4     | 4       | 0.2                 |
| π₄*                | 7     | 7       | 0.4                 |
| π₄*                | 10    | 10      | 0.6                 |
| π₄*                | 13    | 13      | 0.8                 |
| π₄*                | 16    | 16      | 1.0                 |
| π₄*                | 19    | 19      | 0.8                 |
| π₄*                | 22    | 22      | 0.6                 |
| π₄*                | 25    | 25      | 0.4                 |
| π₄*                | 28    | 28      | 0.2                 |
| π₅*                | 1     | 1       | 0.0                 |
| π₅*                | 4     | 4       | 0.2                 |
| π₅*                | 7     | 7       | 0.4                 |
| π₅*                | 10    | 10      | 0.6                 |
| π₅*                | 13    | 13      | 0.8                 |
| π₅*                | 16    | 16      | 1.0                 |
| π₅*                | 19    | 19      | 0.8                 |
| π₅*                | 22    | 22      | 0.6                 |
| π₅*                | 25    | 25      | 0.4                 |
| π₅*                | 28    | 28      | 0.2                 |
</details>

(d) $D_{\Pi^{*}}$ with 250000 samples

![](images/aa986dfc4429dc61aa3ac794433507d9fea1e76eeb86403261b3cc3fa9b6fb89.jpg)

<details>
<summary>heatmap</summary>

| Droping STR(π₁*) | L_end | L_start | Evaluation Accuracy |
| ----------------- | ----- | ------- | ------------------- |
| π₁*               | 1     | 1       | 0.0                 |
| π₁*               | 4     | 4       | 0.2                 |
| π₁*               | 7     | 7       | 0.4                 |
| π₁*               | 10    | 10      | 0.6                 |
| π₁*               | 13    | 13      | 0.8                 |
| π₁*               | 16    | 16      | 1.0                 |
| π₁*               | 19    | 19      | 0.8                 |
| π₁*               | 22    | 22      | 0.6                 |
| π₁*               | 25    | 25      | 0.4                 |
| π₁*               | 28    | 28      | 0.2                 |
| π₂*               | 1     | 1       | 0.0                 |
| π₂*               | 4     | 4       | 0.2                 |
| π₂*               | 7     | 7       | 0.4                 |
| π₂*               | 10    | 10      | 0.6                 |
| π₂*               | 13    | 13      | 0.8                 |
| π₂*               | 16    | 16      | 1.0                 |
| π₂*               | 19    | 19      | 0.8                 |
| π₂*               | 22    | 22      | 0.6                 |
| π₂*               | 25    | 25      | 0.4                 |
| π₂*               | 28    | 28      | 0.2                 |
| π₃*               | 1     | 1       | 0.0                 |
| π₃*               | 4     | 4       | 0.2                 |
| π₃*               | 7     | 7       | 0.4                 |
| π₃*               | 10    | 10      | 0.6                 |
| π₃*               | 13    | 13      | 0.8                 |
| π₃*               | 16    | 16      | 1.0                 |
| π₃*               | 19    | 19      | 0.8                 |
| π₃*               | 22    | 22      | 0.6                 |
| π₃*               | 25    | 25      | 0.4                 |
| π₃*               | 28    | 28      | 0.2                 |
| π₄*               | 1     | 1       | 0.0                 |
| π₄*               | 4     | 4       | 0.2                 |
| π₄*               | 7     | 7       | 0.4                 |
| π₄*               | 10    | 10      | 0.6                 |
| π₄*               | 13    | 13      | 0.8                 |
| π₄*               | 16    | 16      | 1.0                 |
| π₄*               | 19    | 19      | 0.8                 |
| π₄*               | 22    | 22      | 0.6                 |
| π₄*               | 25    | 25      | 0.4                 |
| π₄*               | 28    | 28      | 0.2                 |
| π₅*               | 1     | 1       | 0.0                 |
| π₅*               | 4     | 4       | 0.2                 |
| π₅*               | 7     | 7       | 0.4                 |
| π₅*               | 10    | 10      | 0.6                 |
| π₅*               | 13    | 13      | 0.8                 |
| π₅*               | 16    | 16      | 1.0                 |
| π₅*               | 19    | 19      | 0.8                 |
| π₅*               | 22    | 22      | 0.6                 |
| π₅*               | 25    | 25      | 0.4                 |
| π₅*               | 28    | 28      | 0.2                 |
</details>

(e) $D_{\Pi^{*}}$ with 500000 samples   
Figure 11. Repeated experiments from Figure 10 for models trained with Fixed Dropout instead of Annealing Dropout. Observations remain similar. This suggests that the position of phrasebook internalization is primarily decided by the MLT(d, n)-ICL-capable initialization, and less dependent on the specific dropout curriculum we use for context-enhanced learning.

# C.3. (Non)Verbatim Memorization of Phrasebook Rules

In this subsection, we provide a more detailed explanation for the evaluation on the feasibility of recovering phrasebook rules from models trained with context-enhanced learning with phrasebook excerpts in context.

Given a MLT $_{\Pi^{*}}$ -capable model $f_{\theta^{*}}$ trained with context-enhanced learning where phrasebook rules from $\mathrm{STR}(\Pi^{*})$ are provided within the context during training, we test if the model retains explicit memory of the textual format of rules. Namely, for each of the phrasebook rule of the form “a b -> C D” in $\mathrm{STR}(\Pi^{*})$ , whether it can complete the ground truth output tokens “C D” providing “a b ->” and additional context of the phrasebook $\mathrm{STR}(\Pi^{*})$ before “a b -> C D”. We conduct the test by computing the forward pass through $f_{\theta^{*}}$ with $\mathrm{STR}(\Pi^{*})$ as the input (see exact input format in Figure 17).

Now we formally define the query strategies and metrics we used for creating Table 2. Suppose there are $h$ unique tokens and $\mathrm{STR}(\Pi^{*})$ contains $L$ tokens, let $S \in \mathbb{R}^{h \times L}$ denote the corresponding output logit score matrix when we pass $\mathrm{STR}(\Pi^{*})$ into $f_{\theta^{*}}$ . For a translation rule with ground truth output tokens of index $a_1, a_2 \in A_{i+1} \subset [h]$ , let $S^{(k)}$ and $S^{(k+1)} \in \mathbb{R}^h$ denote the logit vector corresponding to predicting these two entries.

If we are doing greedy decoding, then we will recover the correct phrasebook rule if and only if we greedily select both $a_{1}$ from $\boldsymbol{S}^{(k)}$ and $a_{2}$ from $\boldsymbol{S}^{(k+1)}$ , so the probability of correctly recovering $(a_{1}, a_{2})$ conditioned on S is

$$
\mathbb {P} \left[ \text { Greedy - Recover } (a _ {1}, a _ {2}) \right] = \mathbb {1} \left[ \underset {i \in [ h ]} {\arg \max} \boldsymbol {S} ^ {(k)} = a _ {1} \right] \cdot \mathbb {1} \left[ \underset {i \in [ h ]} {\arg \max} \boldsymbol {S} ^ {(k + 1)} = a _ {2} \right].
$$

Meanwhile, if we apply random sampling with softmax temperature of 1, the probability of correctly recovering $(a_{1}, a_{2})$ conditioned on $S$ is just then

$$
\mathbb {P} \left[ \text {Sampling - Recover} (a _ {1}, a _ {2}) \right] = \left(\frac {\exp \left(\boldsymbol {S} _ {a _ {1}} ^ {(k)}\right)}{\sum_ {i \in [ h ]} \exp \left(\boldsymbol {S} _ {i} ^ {(k)}\right)}\right) \cdot \left(\frac {\exp \left(\boldsymbol {S} _ {a _ {1}} ^ {(k + 1)}\right)}{\sum_ {i \in [ h ]} \exp \left(\boldsymbol {S} _ {i} ^ {(k + 1)}\right)}\right).
$$

Recall from Section 4 that we have also introduced two stronger adversaries with additional token filtering when doing decoding: (1) setting probability of <THINK> tokens to zero (2) when querying for a rule in $\pi_i$ with output alphabet $A_{i+1}$ , setting probability of all tokens outside $A_{i+1}$ to zero.

Denoting the token index of <THINK> as $a_{<THINK>}$ , then filter 1 corresponds to

$$
\mathbb {P} \left[ \text {Greedy - Filter1 - Recover} (a _ {1}, a _ {2}) \right] = \mathbb {1} \left[ \underset {i \in [ h ] \setminus \{a _ {<   \text {THINK} >} \}} {\arg \max} \boldsymbol {S} ^ {(k)} = a _ {1} \right] \cdot \mathbb {1} \left[ \underset {i \in [ h ] \setminus \{a _ {<   \text {THINK} >} \}} {\arg \max} \boldsymbol {S} ^ {(k + 1)} = a _ {2} \right],
$$

$$
\mathbb {P} \left[ \text {Sampling - Filter1 - Recover} (a _ {1}, a _ {2}) \right] = \left(\frac {\exp \left(\boldsymbol {S} _ {a _ {1}} ^ {(k)}\right)}{\sum_ {i \in [ h ] \setminus \{a _ {<   \text {THINK} >} \}} \exp \left(\boldsymbol {S} _ {i} ^ {(k + 1)}\right)}\right) \cdot \left(\frac {\exp \left(\boldsymbol {S} _ {a _ {1}} ^ {(k)}\right)}{\sum_ {i \in [ h ] \setminus \{a _ {<   \text {THINK} >} \}} \exp \left(\boldsymbol {S} _ {i} ^ {(k + 1)}\right)}\right).
$$

Similarly, for filter 2, the probabilities are

$$
\mathbb {P} \left[ \text { Greedy - Filter2 - Recover } (a _ {1}, a _ {2}) \right] = \mathbb {1} \left[ \underset {i \in A _ {i + 1}} {\arg \max} \boldsymbol {S} ^ {(k)} = a _ {1} \right] \cdot \mathbb {1} \left[ \underset {i \in A _ {i + 1}} {\arg \max} \boldsymbol {S} ^ {(k + 1)} = a _ {2} \right],
$$

$$
\mathbb {P} \left[ \text {Sampling - Filter2 - Recover} (a _ {1}, a _ {2}) \right] = \left(\frac {\exp \left(\boldsymbol {S} _ {a _ {1}} ^ {(k)}\right)}{\sum_ {i \in A _ {i + 1}} \exp \left(\boldsymbol {S} _ {i} ^ {(k + 1)}\right)}\right) \cdot \left(\frac {\exp \left(\boldsymbol {S} _ {a _ {1}} ^ {(k)}\right)}{\sum_ {i \in A _ {i + 1}} \exp \left(\boldsymbol {S} _ {i} ^ {(k + 1)}\right)}\right).
$$

Note that the second filter is a very strong adversarial assumption which assumes the user already has side information on the set of tokens contained in alphabet $A_{i}$ and $A_{i + 1}$ . To compute the final statistics, we sample 20 permutations of $\mathrm{STR}(\Pi^{*})$ for forward passes and compute the mean of the above statistics over all atomic phrasebook rules appearing in the context. That is 1280 entries for each phrasebook in the case of $n = 8$ and 2000 entries for each phrasebook in the case of $n = 10$ .

Table 4. Recovery Success Rate For n = 8, d = 5 Cases (Rounded to 2 decimals) Random Guess Baseline: 1.56% 

<table><tr><td rowspan="2">Curriculum</td><td rowspan="2"># Training Samples</td><td rowspan="2">Query Method</td><td colspan="2">Greedy Decoding</td><td colspan="2">Sampling (T=1)</td></tr><tr><td> $\pi_1 - \pi_4$ </td><td> $\pi_5$ </td><td> $\pi_1 - \pi_4$ </td><td> $\pi_5$ </td></tr><tr><td rowspan="3">Annealing Dropout</td><td rowspan="3">50000</td><td>Base</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.01%</td></tr><tr><td>Rule out</td><td>0.06%</td><td>1.95%</td><td>0.00%</td><td>0.54%</td></tr><tr><td>Only keeping  $A_i$ </td><td>3.18%</td><td>1.95%</td><td>1.82%</td><td>1.49%</td></tr><tr><td rowspan="3">Annealing Dropout</td><td rowspan="3">100000</td><td>Base</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.64%</td></tr><tr><td>Rule out</td><td>0.00%</td><td>0.08%</td><td>0.00%</td><td>1.25%</td></tr><tr><td>Only keeping  $A_i$ </td><td>1.37%</td><td>0.08%</td><td>1.69%</td><td>1.80%</td></tr><tr><td rowspan="3">Annealing Dropout</td><td rowspan="3">250000</td><td>Base</td><td>0.00%</td><td>0.23%</td><td>0.00%</td><td>1.25%</td></tr><tr><td>Rule out</td><td>0.00%</td><td>0.23%</td><td>0.00%</td><td>1.28%</td></tr><tr><td>Only keeping  $A_i$ </td><td>2.95%</td><td>0.23%</td><td>2.53%</td><td>1.38%</td></tr><tr><td rowspan="3">Fixed Dropout</td><td rowspan="3">50000</td><td>Base</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Rule out</td><td>0.08%</td><td>0.78%</td><td>0.00%</td><td>0.09%</td></tr><tr><td>Only keeping  $A_i$ </td><td>0.82%</td><td>0.86%</td><td>1.48%</td><td>1.43%</td></tr><tr><td rowspan="3">Fixed Dropout</td><td rowspan="3">100000</td><td>Base</td><td>0.00%</td><td>0.00%</td><td>0.00%</td><td>0.21%</td></tr><tr><td>Rule out</td><td>0.43%</td><td>0.00%</td><td>0.03%</td><td>0.80%</td></tr><tr><td>Only keeping  $A_i$ </td><td>2.36%</td><td>0.00%</td><td>1.95%</td><td>1.32%</td></tr><tr><td rowspan="3">Fixed Dropout</td><td rowspan="3">250000</td><td>Base</td><td>0.00%</td><td>0.47%</td><td>0.02%</td><td>0.51%</td></tr><tr><td>Rule out</td><td>0.68%</td><td>1.09%</td><td>0.11%</td><td>0.73%</td></tr><tr><td>Only keeping  $A_i$ </td><td>2.23%</td><td>1.09%</td><td>2.19%</td><td>1.10%</td></tr></table>

Table 5. Recovery Success Rate For n = 10, d = 5 Cases (Rounded to 2 decimals) Random Guess Baseline: 1% 

<table><tr><td rowspan="2">Curriculum</td><td rowspan="2"># Training Samples</td><td rowspan="2">Query Method</td><td colspan="2">Greedy Decoding</td><td colspan="2">Sampling (T=1)</td></tr><tr><td> $\pi_1 - \pi_4$ </td><td> $\pi_5$ </td><td> $\pi_1 - \pi_4$ </td><td> $\pi_5$ </td></tr><tr><td rowspan="3">Annealing Dropout</td><td rowspan="3">100000</td><td>Base</td><td>0.00%</td><td>0.20%</td><td>0.00%</td><td>0.89%</td></tr><tr><td>Rule out</td><td>0.00%</td><td>0.20%</td><td>0.00%</td><td>0.90%</td></tr><tr><td>Only keeping  $A_i$ </td><td>1.66%</td><td>0.20%</td><td>1.28%</td><td>0.94%</td></tr><tr><td rowspan="3">Annealing Dropout</td><td rowspan="3">250000</td><td>Base</td><td>0.00%</td><td>1.80%</td><td>0.00%</td><td>1.12%</td></tr><tr><td>Rule out</td><td>0.00%</td><td>1.80%</td><td>0.00%</td><td>1.13%</td></tr><tr><td>Only keeping  $A_i$ </td><td>3.05%</td><td>1.80%</td><td>1.72%</td><td>1.23%</td></tr><tr><td rowspan="3">Fixed Dropout</td><td rowspan="3">250000</td><td>Base</td><td>0.00%</td><td>1.60%</td><td>0.00%</td><td>1.07%</td></tr><tr><td>Rule out</td><td>0.05%</td><td>1.60%</td><td>0.01%</td><td>1.07%</td></tr><tr><td>Only keeping  $A_i$ </td><td>2.46%</td><td>2.15%</td><td>1.99%</td><td>1.39%</td></tr><tr><td rowspan="3">Fixed Dropout</td><td rowspan="3">500000</td><td>Base</td><td>0.26%</td><td>2.05%</td><td>0.05%</td><td>1.13%</td></tr><tr><td>Rule out</td><td>0.33%</td><td>2.05%</td><td>0.07%</td><td>1.13%</td></tr><tr><td>Only keeping  $A_i$ </td><td>2.11%</td><td>2.05%</td><td>1.91%</td><td>1.34%</td></tr></table>

Here we report the query success rate for a variety of models trained with context-enhanced learning using different curriculum on different datasets sizes. All models reaches nearly perfect test accuracy when no context is provided (see Figure 2), i.e., the in context phrasebook rules played significant role in the learning of the models.

For all runs, we can see that it is nearly impossible (with recovery probability $< 0.1\%$ in most cases) to recover the correct phrasebook rules for hidden steps $(\pi_1, \ldots, \pi_4)$ even when we provide the correct partial phrasebooks in context (see columns corresponding to Query Method "Base"). Ruling out $<\text{THINK}>$ token when sampling also did not significantly increase the recovery rate. We note that the phrasebook knowledge are not memorized in an completely undetectable manner, as the strongest token filtering gives non-random probability of outputting the correct target 2-tuple. However the probability is still very low ( $< 3\%$ ), which can be considered as negligible to recover the full correct phrasebooks $\text{STR}(\Pi^*)$ .

# D. Additional Related Works

# Differences with Masked Language Modeling (MLM) and Language Infilling

MLM models like BERT, RoBERTa, and T5 (Kenton & Toutanova, 2019; Liu et al., 2019; Raffel et al., 2020) train models by either masking or removing tokens from a sequence and compute loss on the model's prediction on the missing tokens. This concept has been adapted for training auto-regressive models via language infilling task (Bavarian et al., 2022; Li et al., 2022; Donahue et al., 2020; Li et al., 2022). The primary difference from these works is that context-enhanced learning does not take loss on the context tokens when they are removed from our curriculum-text.

# Compositional and OOD generalization

Measuring generalization for a transformer beyond training distribution has been a study of interest in many prior works. OOD generalization is measured by training a transformer on simpler examples and measuring its performance on harder ones. Prominent studies include length generalization, informally defined as the ability of the model to reason longer than what it has been trained on (Zhou et al., 2023; Anil et al., 2022), and compositional generalization on concepts, defined as the ability of the model to reason on composition of the concepts that it has seen during training (Press et al., 2022; Allen-Zhu & Li, 2023b; Ramesh et al., 2023; Yu et al., 2023; Zhao et al., 2024; Wang et al., 2024; Yang et al., 2024). Our experiments on Fixed Dropout in Section 3, where we train with $20\%$ dropout on the phrasebooks information but measure performance with $100\%$ dropout at test time, measures compositional OOD generalization behavior of the language model. The results show that the model internalizes the rules from the phrasebooks in an atomic way, and re-compose them together as necessary at test time.

# Mechanistic behavior of transformers with synthetic datasets:

Our work builds on a growing body of research exploring the behavior of transformers trained on synthetic datasets. Prior studies have examined tasks such as modular addition (Nanda et al., 2023; Zhong et al., 2023), context-free grammars (Zhao et al., 2023; Allen-Zhu & Li, 2023a), regular and n-gram languages (Bhattamishra et al., 2020; Yao et al., 2021; Akyürek et al., 2024; Li et al., 2023), and synthetic article-style datasets (Allen-Zhu & Li, 2023b; 2024; Eldan & Li, 2023). While our work is structurally similar to these studies, it investigates mechanistic study on context-enhanced learning that has not been explored in previous works.

# E. Properties of MLT

MLT(d, n) is defined by the phrasebooks $\pi_{1}, \cdots, \pi_{d}$ at each of its translation layers. We use $B^{\pi}$ as all possible set of bijective maps that can be used to define the phrasebooks. We will use variable $\pi$ to refer to an arbitrary bijective map from the set $B^{\pi}$ . For simplicity of proof, we will refer to any alphabet set A of size n as $\{0, 1, \cdots, n - 1\}$ .

Here, we formally mention some of the properties of $\mathbf{MLT}(d,n)$ .

Lemma E.1 (Invertibility of sequence translation). For any level $i \in [d]$ , fixing $\{\pi_{i}, \pi_{i+1}, \ldots, \pi_{d}\}$ gives a bijection between $s_{i}$ and $s_{d+1}$ .

Proof. All of the four operations involved in the mapping from $s_i$ to $s_{i+1}$ are invertible.

Structure of the mappings: The mappings $\pi_{i}$ are selected as random bijective maps between 2-tuples of characters in $A_{i}^{2}$ and 2-tuples of characters in $A_{i+1}^{2}$ . A combinatorial argument can then give the number of such possible phrasebooks to be $n^{2}!$ .

Lemma E.2 (Number of mappings in each level). The number of possible bijective maps between 2-tuples of characters from alphbabet sets $A_{i}, A_{i+1}$ , each being of size $n$ , is $n^2!$ , i.e. $|\mathcal{B}^{\pi}| = n^2!$ .

Proof. Each alphabet set contains $n^{2}$ possible 2-tuples of characters. If we fix an order in which the 2-tuples appear in $A_{i+1}^{2}$ , then the number of bijective maps between $A_{i}^{2}$ and $A_{i+1}^{2}$ can be reduced to the number of possible ordering of the 2-tuples in $A_{i}^{2}$ . The number of possible orderings is $n^{2}!$ .

Importance of Circular shift: The composition of the d random bijective maps can be demonstrated to result in another random bijective map. Consequently, without the Circular shift, each character in the output sequence $s_{d+1}$ depends on only two characters from the input sequence $s_{1}$ via a shared random map across all the 2-tuples.

Incorporating Circular shift on the other hand enables each character in the output sequence to depend on 2d characters from the input sequence. This is because the character positions are shifted to the right at each step, causing the input 2-tuples to the bijective map to also shift to the right at each stage. As we will elaborate later, incorporating Circular shift increases the required number of training samples to learn the set of phrasebooks from input and label pairs to $n^{\Omega(d)}$ , whereas without Circular shift, this requirement is only $\mathcal{O}(n^{2})$ .

Representing the mappings on 2-tuples in-context: Each map can be defined using $\mathcal{O}(n^{2})$ characters, as it can be simply defined by the $n^{2}$ relations each connecting 2 unique random 2-tuples from their corresponding alphabet sets. Thus, defining d maps in-context will require $\mathcal{O}(n^{2}d)$ characters in curriculum-text. In contrast, describing a completely random bijective mapping that maps d-tuples of characters in input sequence to a character in output sequence will require $\Omega(d\binom{n}{d})$ bits. Thus, MLT with d translation steps involving random mappings on 2-tuples and Circular shift helps define a mapping where each character in the output sequence can depend on d character in the input sequence, and the set of phrasebooks can be described using $\mathcal{O}(n^{2}d)$ characters in curriculum-text.

# F. Lower Bound: Hardness of Learning MLT(d, n) without Context

# F.1. Brief introduction to SQ framework

Statistical query (SQ) bounds : The statistical query (SQ) framework measures the computational hardness of learning a task in the presence of noise. It measures the hardness of learning a task by the number of statistical queries needed by a learning algorithm to learn the true function. Statistical queries are defined by some polynomially-computable property Q of labeled instances and a tolerance parameter $\tau \in [0, 1]$ over $(x, y) \sim D$ where D is the data distribution. For a query, the algorithm receives a response from the oracle within $\tau$ error of the true value. The statistical dimension, or SQ-dim, is measured in terms of the number of functions in the hypothesis class that the learning algorithm needs to distinguish and the number of queries necessary to do the same. Correlation between two functions is used to define the statistical query dimension.

Definition F.1. Correlation of two functions $f_{1}, f_{2}$ on a domain X with respect to a distribution D is given by

$$
\operatorname{Correlation} \left(f _ {1}, f _ {2}, \mathcal {D}\right) := \left| \operatorname * {P r} _ {x \in \mathcal {D}} \left[ f _ {1} (x) = f _ {2} (x) \right] - \operatorname * {P r} _ {x \in \mathcal {D}} \left[ f _ {1} (x) \neq f _ {2} (x) \right] \right|.
$$

For functions $f_{1}, f_{2}: \mathcal{X} \to \{0,1\}$ , the above definition is also equivalent to

$$
\operatorname{Correlation} \left(f _ {1}, f _ {2}, \mathcal {D}\right) := \left| 1 - 2 \mathbb {E} _ {x \in \mathcal {D}} \left[ f _ {1} (x) \oplus f _ {2} (x) \right] \right|.
$$

Remark F.2. Two functions $f_{1}, f_{2}: \mathcal{X} \to \{0, 1\}$ are said to be uncorrelated w.r.t. $\mathcal{D}$ if

$$
\operatorname * {P r} _ {x \in \mathcal {D}} [ f _ {1} (x) = f _ {2} (x) ] = \operatorname * {P r} _ {x \in \mathcal {D}} [ f _ {1} (x) \neq f _ {2} (x) ]
$$

$$
\text {(alternately)} \mathbb {E} _ {x \in \mathcal {D}} [ f _ {1} (x) \oplus f _ {2} (x) ] = \frac {1}{2}.
$$

On the other hand, if

$$
P r _ {x \in \mathcal {D}} [ f _ {1} (x) = f _ {2} (x) ] = 1 \quad (\text { or } 0)
$$

$$
(\text { alternately }) \mathbb {E} _ {x \in \mathcal {D}} [ f _ {1} (x) \oplus f _ {2} (x) ] = 0 \quad (\text { or } 1),
$$

then Correlation $(f_{1}, f_{2}, \mathcal{D}) = 1$ .

We take the following formal definitions of SQ-dim and its relation to computational hardness in the SQ framework from (Blum et al., 1994).

Definition F.3 (Definition 2 in Blum et al. (1994)). For a function class $\mathcal{F}$ of boolean functions over $\{0,1\}^n$ and $\mathcal{D}$ a distribution over $\{0,1\}^n$ , $\mathrm{SQ - dim}(\mathcal{F},\mathcal{D})$ , the statistical query dimension of $\mathcal{F}$ with respect to $\mathcal{D}$ , is defined to be the largest natural number $\mu$ such that $\mathcal{F}$ contains $\mu$ functions $f_{1},\dots ,f_{\mu}$ with the property that for all $i\neq j$ we have:

$$
\text { Correlation } (f _ {i}, f _ {j}, \mathcal {D}) := \left| \operatorname * {P r} _ {x \in \mathcal {D}} [ f _ {i} = f _ {j} ] - \operatorname * {P r} _ {x \in \mathcal {D}} [ f _ {i} \neq f _ {j} ] \right| \leq \frac {1}{\mu^ {3}}.
$$

Theorem F.4 (Theorem 12 in Blum et al. (1994)). Let $\mathcal{F}$ be a class of functions $\{0,1\}^n$ and $\mathcal{D}$ a distribution such that $SQ\text{-dim}(\mathcal{F},\mathcal{D})\geq \mu \geq 16$ . Then if all queries are made with a tolerance of atleast $\frac{1}{\mu^{1/3}}$ , at least $\mu^{1/3}/2$ queries are required to learn $\mathcal{F}$ with error less than $1/2 - 1/\mu^3$ in the statistical query model.

# F.2. Lower bound lemma

Here, we mention the 2 main theorems that study the SQ dimension of the MLT task at hand. The first theorem shows the SQ dimension bound when number of characters n = 2, which is then adapted to get the SQ dimension bound for general n. Proofs of both the theorems are given in Appendix F.4 and Appendix F.5 respectively.

Theorem F.10 (SQ dimension for n = 2). The family of translation task $MLT(d, 2)$ on input distribution $\mathcal{U}(\{0, 1\}^{2d})$ has statistical query dimension $SQ\text{-dim}(MLT(d, 2))$ atleast $2^{\Omega(d)}$ .

Theorem F.20 (SQ dimension for general n). For the translation task $MLT(d, n)$ that has depth d and n characters per level, the statistical query dimension $SQ-\dim(MLT(d, n))$ is at least $n^{\Omega(d)}$ .

Notations: We will require the following notations for proving the above theorems. First, we will denote a 2-tuple that contains arbitrary characters $a$ and $b$ as $(a,b)$ . For a bijective map $\pi, \pi((a,b))_i$ will represent $i$ th character in the output of $\pi$ on any 2-tuple input $(a,b)$ and any $i \in \{1,2\}$ . Similarly, for any set of phrasebooks $\Pi$ , we will use $\mathbf{MLT}_{\Pi}(s_1)_i$ to denote the $i$ th character in the output of $\mathbf{MLT}_{\Pi}$ on any $L$ length input sequence $s_1$ and any $i \in [1,L]$ .

Recall that for any input sequence $s_{1}$ , the output of the ith level of a MLT task will be denoted as $s_{i}$ . Furthermore, to denote the jth character (arbitrary) in the sequence $s_{i}$ , we will use $s_{i,j}$ .

# F.2.1. BOUNDS FOR SGD

The following corollary measures the sample complexity needed to learn $MLT_{\Pi^{*}}$ by SGD. It has been adapted from proposition 3 in Edelman et al. (2023), who study the sample complexity for learning d-sparse parity task on n length sequences, whose SQ dimension is $\binom{n}{d} = \Theta(n^{d})$ . We simply state the corollary without specifying the proof.

Loss function and SGD updates: Consider training of a model $f_{\theta}$ with r parameters that is trained with mean squared error, i.e. $\ell_{\mathrm{auto}}(f_{\theta}([{\mathrm{CURR}}_{g}(x,t),x,y]),y)=\|f_{\theta}([{\mathrm{CURR}}_{g}(x,t),x,y])-y\|^{2}$ . The empirical loss on a batch S of batch size B from the supervised dataset $D_{\Pi^{*}}$ will be denoted by $L_{S}(f_{\theta},\mathbf{MLT}_{\Pi^{*}})=\mathbb{E}_{(x,y)\sim S}\ell_{\mathrm{auto}}(f_{\theta}([{\mathrm{CURR}}_{g}(x,t),x,y]),y)=\|f_{\theta}([{\mathrm{CURR}}_{g}(x,t),x,y])-y\|^{2}$ ; the population loss will be denoted by $L_{\mathcal{U}(\{0,1\}^{L})}(f_{\theta},\mathbf{MLT}_{\Pi^{*}})$ . SGD updates are of the form:

$$
\theta_ {t + 1} = \theta_ {t} - \eta_ {t} (\nabla_ {\theta} L _ {S _ {t}} (f _ {\theta_ {t}}, \mathbf {M L T} _ {\boldsymbol {\Pi} ^ {*}}) + R (\theta_ {t}) + \zeta_ {t}).
$$

for some sample $S_{t}$ , step size $\eta_{t}$ , regularizer $R(\cdot)$ , and adversarial noise $\zeta_{t} \in [-\tau, \tau]^{r}$ . For simplicity, we assume the gradient $\nabla_{\theta} L_{S_t}(\theta_t)$ is bounded on all parameters $\theta$ in parameter space of the model.

Fake trajectory: Suppose 0 denote the constant function that maps all inputs to 0. Then, with the contemporary losses $L_{S}(f_{\theta}, \mathbf{0})$ and $L_{\mathcal{U}(\{0,1\}^{L})}(f_{\theta}, \mathbf{0})$ that computes difference from this constant function, consider the following trajectory $\tilde{\theta}_{1}, \cdots, \tilde{\theta}_{t}, \cdots$ starting from the same initiation $\theta_{0}$ :

$$
\tilde {\theta} _ {t + 1} = \tilde {\theta} _ {t} - \eta_ {t} (\nabla_ {\theta} L _ {S _ {t}} (f _ {\tilde {\theta} _ {t}}, \mathbf {0}) + R (\tilde {\theta} _ {t})).
$$

Assumption F.5. For all $t$ , suppose $\left\| \nabla_{\theta} L_{\mathcal{U}(\{0,1\}^L)}(f_{\tilde{\theta}_t}, \mathbf{0}) - \nabla_{\theta} L_{S_t}(f_{\tilde{\theta}_t}, \mathbf{0}) \right\|_2 \leq \tau / 2$ .

Corollary F.6 (Sample complexity bounds for SGD). Fix an initialization $\theta_{0}$ such that $f_{\theta_{0}}$ is statistically independent (correlation 0) from all possible $MLT_{\Pi^{*}}$ . Under this assumption, if $\frac{LTB}{\tau^{2}} \leq n^{\Omega(d)}/r$ , then there exists at least one task

$MLT_{\Pi^{*}} \in MLT(d, n)$ for which the functions obtained after the first T SGD updates, $f_{\theta_{1}}, \ldots, f_{\theta_{T}}$ , remain statistically independent (correlation 0) from $MLT_{\Pi^{*}}$ despite training on it.

LTB is the product of input length, total training steps, and batch size, which represents sample complexity used by the SGD algorithm.

# F.3. Useful lemmas for $n = 2$

Here, we mention some useful lemmas for characterizing the bijective maps on $\{0,1\}^{2}\rightarrow\{0,1\}^{2}$ that we will regularly use for the proof of Theorem F.10. The first lemma will show that each bijective map can be represented by three operations on input characters, copy, xor, and not operations. The second lemma measures correlation on the outputs of two randomly sampled bijective maps. The proofs are given in Appendix F.6.

We define the following necessary operations to describe the random bijective maps on $\{0,1\}^{2}\to\{0,1\}^{2}$ :

1. copy: Given a tuple $(x_{1}, x_{2})$ , and a position argument $i \in \{1, 2\}$ , the operation returns value of $x_{i}$ as output. We will use $\mathbf{copy}_1$ and $\mathbf{copy}_2$ to indicate the copy operation on positions 1 and 2 respectively.   
2. not: Given a variable $x_{i}$ , this operation returns flipped value of $x_{i}$ . That is, if $x_{i} = 1$ , then it returns 0 and vice-versa.   
3. xor: Given a tuple $(x_{1}, x_{2})$ , this operation returns $x_{1} \oplus x_{2}$ .

Lemma F.21 (Formulation of bijective maps for n = 2). Any bijective map $\pi : \{0,1\}^{2} \to \{0,1\}^{2}$ can be expressed using copy, not, and xor operations. Furthermore, from 24 possible maps for $\pi$ ,

1. There are 6 maps $\Delta_1, \Delta_2, \cdots, \Delta_6$ for which characters in the output tuple can be defined by copy and xor operations on the characters in the input tuple.   
2. Mirror maps: For each map $\pi\in\{\Delta_{1},\Delta_{2},\cdots,\Delta_{6}\}$ , there exist mirror maps $\pi_{(1)},\pi_{(2)},\pi_{(3)}$ whose output on each input tuple can be defined by selective not operations on either or both characters of the output tuple of $\pi$ . We call $\{\pi,\pi_{(1)},\pi_{(2)},\pi_{(3)}\}$ as a mirror map set of $\pi$ , in short, $\text{Mirrorset}(\pi)$ .

Remark F.7. Mirror map set definition is general and isn't restricted to maps $\pi\in\{\Delta_{1},\Delta_{2},\cdots,\Delta_{6}\}$ . Because mirror maps are defined in terms of not operation, a mirror map set $\text{Mirrorset}(\pi)$ for $\pi\in\{\Delta_{1},\Delta_{2},\cdots,\Delta_{6}\}$ is also equivalent to $\text{Mirrorset}(\pi_{(i)})$ for $i\in\{1,2,3\}$ .

Remark F.8. Lemma F.21 can be re-stated as follows: the 24 possible bijective maps $\{0,1\}^{2}\to\{0,1\}^{2}$ can be grouped into 6 family of maps, each containing 4 maps and represented by a unique map from $\{\Delta_{1},\Delta_{2},\cdots,\Delta_{6}\}$ . Each family is defined by mirror map set of their corresponding representative map.

Lemma F.22 (Correlation of bijective maps for n = 2). For two randomly selected maps $\pi^{\alpha}, \pi^{\beta} : \{0,1\}^{2} \to \{0,1\}^{2}$ , the following hold true.

1. Correlation at atleast one output character pair: Fix an $i, j \in \{0, 1\}$ . With probability $\frac{1}{3}$ w.r.t. the random selection of the maps, the ith character in the output of $\pi^{\alpha}$ has perfect correlation to jth character in the output of $\pi^{\beta}$ , i.e.

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {i}, \pi^ {\beta} (\cdot) _ {j}, \mathcal {U} (\{0, 1 \} ^ {2})) := \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \pi^ {\alpha} (x) _ {i} \neq \pi^ {\beta} (x) _ {j} ] - \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \pi^ {\alpha} (x) _ {i} = \pi^ {\beta} (x) _ {j} ] \right| = 1.
$$

2. Correlation at both output character pairs: With probability $\frac{1}{3}$ w.r.t. the random selection of the maps, both the characters in the outputs of $\pi^{\alpha}, \pi^{\beta}$ have perfect correlation, i.e. either one of the cases hold true

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {1}, \pi^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})) = 1.
$$

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {2}, \pi^ {\beta} (\cdot) _ {2}, \mathcal {U} (\{0, 1 \} ^ {2})) = 1.
$$

or

$$
\operatorname{Correlation} \left(\pi^ {\alpha} (\cdot) _ {1}, \pi^ {\beta} (\cdot) _ {2}, \mathcal {U} \left(\{0, 1 \} ^ {2}\right)\right) = 1.
$$

$$
C o r r e l a t i o n (\pi^ {\alpha} (\cdot) _ {2}, \pi^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})) = 1.
$$

Other cases are not possible, i.e. for any $i \in \{1,2\}$ , both Correlation $(\pi^{\alpha}(\cdot)_i,\pi^{\beta}(\cdot)_1,\mathcal{U}(\{0,1\}^{2})) = 1$ and Correlation $(\pi^{\alpha}(\cdot)_i,\pi^{\beta}(\cdot)_2,\mathcal{U}(\{0,1\}^{2})) = 1$ can't hold true.

Remark F.9. Implication of Lemma F.22 is as follows: With probability at most $\frac{1}{3}$ , output of two random maps can stay correlated at either one output character pair or both pairs of output characters. This would suggest that the probability of the output of MLT under two random set of phrasebooks staying correlated should decay exponentially with the depth of the task.

# F.4. Proof for Statistical dimension lower bound for n = 2

We repeat the lemma of interest for presentation.

Theorem F.10 (SQ dimension for $n = 2$ ). The family of translation task $MLT(d, 2)$ on input distribution $\mathcal{U}(\{0, 1\}^{2d})$ has statistical query dimension $SQ - \dim(MLT(d, 2))$ at least $2^{\Omega(d)}$ .

Proof. We narrow our argument to the first character output of the task. The proof goes through 2 major steps.

1. First, we show that two random instances of $\mathbf{MLT}(d,2)$ , defined by 2 set of phrasebooks $\{\pi_{1}^{\alpha},\cdots,\pi_{d}^{\alpha}\}$ and $\{\pi_{1}^{\beta},\cdots,\pi_{d}^{\beta}\}$ , will be uncorrelated with probability at least $1-(1/3)(7/9)^{d-1}$ w.r.t. the selection of the random set of phrasebooks. The lemma is formally given in Lemma F.13.   
2. Then, we can apply Lovász local lemma (Theorem F.11) to show that we can create $2^{\Omega(d)}$ instances of $\mathbf{MLT}(d,2)$ that will have zero pairwise correlation of their output (Corollary F.12).

By the definition of SQ-dim from Definition F.3, the above observations suggest that $\mathrm{SQ-dim}(\mathbf{MLT}(d,2))\geq2^{\Omega(d)}$ . ☐

Theorem F.11 (Lovász local lemma, theorem 1.5 in Spencer (1977)). Let $A_1, \cdots, A_k$ be events in some probability space with $\Pr(A_i) \leq p$ , $1 \leq i \leq k$ , such that each event is dependent on almost $k_0$ other events. If $ep(k_0 + 1) < 1$ , then $\Pr(\neg A_1 \land \neg A_2 \land \neg A_3 \cdots \land \neg A_k) > 0$ .

Corollary F.12 (Number of uncorrelated instances of $MLT(d,2)$ ). There exists a set of $2^{\Omega(d)}$ instances in $MLT(d,2)$ that are pairwise uncorrelated to each other on input distribution $\mathcal{U}(\{0,1\}^{2d})$ .

Proof. Suppose $\Pi_{1},\cdots,\Pi_{k}$ represent k randomly sampled set of phrasebooks. For each $1\leq i<j\leq k$ , we will denote event $A_{ij}$ as the event that the outputs of $MLT_{\Pi_{i}}$ and $MLT_{\Pi_{j}}$ are uncorrelated on input distribution $\mathcal{U}(\{0,1\}^{2d})$ . This happens with probability $p=1-(1/3)(7/9)^{d-1}$ from Lemma F.13. Because each event can almost depend on almost $k(k+1)/2<k^{2}$ events (total number of events), by Theorem F.11, if

$$
e p (k ^ {2} + 1) <   1,
$$

then there exists a list of such set of phrasebooks which are pairwise uncorrelated w.r.t. the outputs of their corresponding translation tasks on input distribution $\mathcal{U}(\{0,1\}^{2d})$ . Solving the above for k, we can set $k = ((ep)^{-1} - 1)^{1/2} = 2^{\Omega(d)}$ . ☐

# F.4.1. AUXILIARY LEMMAS

Here, we will prove the following primary lemma, that is used to prove Theorem F.10.

Lemma F.13. With probability at least $1 - \frac{1}{3}\left(\frac{7}{9}\right)^{d-1}$ w.r.t. random map selection, the following holds true for 2 sets of random phrasebooks $\Pi^{\alpha} = \{\pi_{1}^{\alpha}, \cdots, \pi_{d}^{\alpha}\}$ and $\Pi^{\beta} = \{\pi_{1}^{\beta}, \cdots, \pi_{d}^{\beta}\}$ :

$$
\text {Correlation} (\boldsymbol {M L T} _ {\boldsymbol {\Pi} ^ {\alpha} = \{\pi_ {\ell} ^ {\alpha} \} _ {\ell = 1} ^ {d}} (\cdot) _ {1}, \boldsymbol {M L T} _ {\boldsymbol {\Pi} ^ {\beta} = \{\pi_ {\ell} ^ {\beta} \} _ {\ell = 1} ^ {d}} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2 d})) = 0.
$$

Proof. We first dive into the dependencies between characters in input and output sequences in the translation process. In Lemma F.15, we show that at any level $1 \leq \ell \leq d$ of $\mathbf{MLT}_{\Pi}$ with phrasebooks $\Pi = \{\pi_{\ell}\}_{\ell=1}^{d}$ , the following relation holds true on an input $s_1 \in \{0, 1\}^{2d}$ :

$$
\left(\boldsymbol {s} _ {\ell + 1, 1}, \boldsymbol {s} _ {\ell + 1, 2}\right) = \pi_ {\ell} \left(\left(\boldsymbol {s} _ {\ell + 1, 2}, \boldsymbol {s} _ {\ell + 1, 3}\right)\right).
$$

We will use superscripts $\alpha$ and $\beta$ to differentiate the intermediate outputs of MLT when the phrasebooks are set as $\Pi^{\alpha} = \{\pi_{1}^{\alpha}, \cdots, \pi_{d}^{\alpha}\}$ and $\Pi^{\beta} = \{\pi_{1}^{\beta}, \cdots, \pi_{d}^{\beta}\}$ respectively. We will use an induction strategy to find the correlation of $s_{d+1,1}^{\alpha}$ and $s_{d+1,1}^{\beta}$ . To do so, we will require the following variables:

1. $p_{\ell,i;j}$ : For one pair of $i,j\in\{2,3\}$ , this represents the probability at a level $2\leq\ell\leq1+d$ w.r.t. the randomness of $\{\pi_{1}^{\alpha},\cdots,\pi_{\ell-1}^{\alpha}\}$ and $\{\pi_{1}^{\beta},\cdots,\pi_{\ell-1}^{\beta}\}$ , that $s_{\ell,i}^{\alpha}$ and $s_{\ell,j}^{\beta}$ are perfectly correlated. Additionally, we define $p_{\ell,1;1}$ that represents the probability at a level $1\leq\ell\leq1+d$ w.r.t. the randomness of $\{\pi_{1}^{\alpha},\cdots,\pi_{\ell-1}^{\alpha}\}$ and $\{\pi_{1}^{\beta},\cdots,\pi_{\ell-1}^{\beta}\}$ , that $s_{\ell,1}^{\alpha}$ and $s_{\ell,1}^{\beta}$ are perfectly correlated.   
2. $p_{\ell,\text{both}}$ : This represents the probability at a level $2 \leq \ell \leq 1 + d$ w.r.t. the randomness of $\{\pi_{1}^{\alpha}, \cdots, \pi_{\ell-1}^{\alpha}\}$ and $\{\pi_{1}^{\beta}, \cdots, \pi_{\ell-1}^{\beta}\}$ , that either $s_{\ell,2}^{\alpha}, s_{\ell,2}^{\beta}$ are correlated and $s_{\ell,3}^{\alpha}, s_{\ell,3}^{\beta}$ are correlated, or $s_{\ell,2}^{\alpha}, s_{\ell,3}^{\beta}$ are correlated and $s_{\ell,3}^{\alpha}, s_{\ell,2}^{\beta}$ are perfectly correlated.

There are three relations that we need to keep in mind, before we proceed with the induction proof.

1. First, the correlations are circular in nature, i.e., for $j \in \{2,3\}$ and any integer $k$ , $p_{\ell,j;j} = p_{\ell,j;(j + 2k)\% 2d}$ (Lemma F.17). We will use this relation to connect $p_{\ell,1;1}$ with $p_{\ell,3;3}$ , i.e.

$$
p _ {\ell , 1; 1} = p _ {\ell , 3; 3}. \tag {1}
$$

2. Second, $p_{\ell, \text{both}} \leq \max_{ij} p_{\ell, i:j}$ . Intuitively, this is because getting correlations at both positions must be at most as probable as getting correlations at one of the positions.

$$
p _ {\ell , \text { both }} \leq \max _ {i, j} p _ {\ell , i; j}. \tag {2}
$$

3. Third, cross-correlation probabilities given by $p_{\ell,2;3}$ and $p_{\ell,3;2}$ must be $\leq \frac{1}{2}(p_{\ell,2;2} + p_{\ell,3;3})$ (Lemma F.19).

$$
p _ {\ell , i; j} \leq \frac {1}{2} (p _ {\ell , i; i} + p _ {\ell , j; j}), \text {   for   any   } i, j \in \{2, 3 \}, i \neq j. \tag {3}
$$

Base condition: We will use the values of the variables at $\ell = 2$ as our base condition. The input for first layer of translation is same to both $MLT_{\Pi^{\alpha}}$ , $MLT_{\Pi^{\beta}}$ . Then, for an input $s_{1}$ ,

$$
(\boldsymbol {s} _ {2, 1} ^ {\alpha}, \boldsymbol {s} _ {2, 2} ^ {\alpha}) = \pi_ {1} ^ {\alpha} ((\boldsymbol {s} _ {1, 2}, \boldsymbol {s} _ {1, 3}))
$$

$$
(\boldsymbol {s} _ {2, 1} ^ {\beta}, \boldsymbol {s} _ {2, 2} ^ {\beta}) = \pi_ {1} ^ {\beta} ((\boldsymbol {s} _ {1, 2}, \boldsymbol {s} _ {1, 3})).
$$

We can then use Lemma F.22 to show that

$$
p _ {2, i; j} = \frac {1}{3}, \text {   for   all   } i, j \in \{1, 2 \},
$$

$$
p _ {2, \text { both }} = \frac {1}{3}.
$$

Connecting the variables at $\ell$ and $\ell + 1$ : We connect $p_{\ell + 1,2;2}$ to $p_{\ell,2;3}$ , $p_{\ell,3;2}$ , $p_{\ell,3;3}$ and $p_{\ell,\text{both}}$ . This will undergo a case by case analysis.

1. First, if under phrasebooks $\pi_{1}^{\alpha},\cdots,\pi_{\ell-1}^{\alpha}$ and $\pi_{1}^{\beta},\cdots,\pi_{\ell-1}^{\beta}$ , if

(either) $s_{\ell,2}^{\alpha}$ and $s_{\ell,2}^{\beta}$ are perfectly correlated, and $s_{\ell,3}^{\alpha}$ and $s_{\ell,3}^{\beta}$ are perfectly correlated

(or) $s_{\ell,2}^{\alpha}$ and $s_{\ell,3}^{\beta}$ are perfectly correlated, and $s_{\ell,3}^{\alpha}$ and $s_{\ell,2}^{\beta}$ are perfectly correlated, (4)

we can use Lemma F.14 to show that with probability $\frac{1}{3}$ w.r.t. the selection of $\pi_{\ell}^{\alpha}$ and $\pi_{\ell}^{\beta}$ , $s_{\ell+1,2}^{\alpha}$ and $s_{\ell+1,2}^{\beta}$ will be correlated. Conditions necessary for Lemma F.14, i.e. uniform distribution of the characters in the input sequence, are shown to hold true in Lemma F.16. The probability w.r.t. the selection of phrasebooks $\pi_{1}^{\alpha},\cdots,\pi_{\ell-1}^{\alpha}$ and $\pi_{1}^{\beta},\cdots,\pi_{\ell-1}^{\beta}$ such that Equation (4) holds true is given by the variable $p_{\ell,both}$ .

2. On the other hand, if there exists a pair $i, j \in \{2, 3\}$ , such that under phrasebooks $\pi_{1}^{\alpha}, \cdots, \pi_{\ell-1}^{\alpha}$ and $\pi_{1}^{\beta}, \cdots, \pi_{\ell-1}^{\beta}$ , $s_{\ell,i}^{\alpha}$ and $s_{\ell,j}^{\beta}$ are perfectly correlated, then with probability $\frac{1}{9}$ w.r.t. the selection of $\pi_{\ell}^{\alpha}$ and $\pi_{\ell}^{\beta}$ , $s_{\ell+1,2}^{\alpha}$ and $s_{\ell+1,2}^{\beta}$ will be correlated. We again refer to Lemma F.14 for this statement. Probability that this condition happens under phrasebooks $\pi_{1}^{\alpha}, \cdots, \pi_{\ell-1}^{\alpha}$ and $\pi_{1}^{\beta}, \cdots, \pi_{\ell-1}^{\beta}$ is given by the variable $p_{\ell,i;j}$ .

3. If none of the above situation occurs, then for all pairs $i,j \in \{2,3\}$ , $s_{\ell,i}^{\alpha}$ and $s_{\ell,j}^{\beta}$ are uncorrelated, and so, by Lemma F.22, correlation between $s_{\ell + 1,2}^{\alpha}$ and $s_{\ell + 1,2}^{\beta}$ will stay 0 for any choice of $\pi_{\ell}^{\alpha}$ and $\pi_{\ell}^{\beta}$ .

Thus, combining the 3 cases, we must have

$$
p _ {\ell + 1, 2; 2} \leq \frac {1}{3} p _ {\ell , \text { both }} + \frac {1}{9} \sum_ {i, j \in \{2, 3 \}} p _ {\ell , i; j} \tag {5}
$$

$$
\leq \frac {1}{3} p _ {\ell , \text { both }} + \frac {4}{9} \max _ {i, j \in \{2, 3 \}} p _ {\ell , i; j}
$$

$$
\leq \frac {1}{3} \max _ {i, j \in \{2, 3 \}} p _ {\ell , i; j} + \frac {4}{9} \max _ {i, j \in \{2, 3 \}} p _ {\ell , i; j} = \frac {7}{9} \max _ {i, j \in \{2, 3 \}} p _ {\ell , i; j} \tag {6}
$$

Reasoning for each step is as follows:

1. Equation (5) follows from conditional probability computations using the 3 cases that we discussed before.   
2. Equation (6) uses Equation (2) to connect $p_{\ell,\text{both}}$ to $p_{\ell,i;j}$ .

We can give the same inequality for $p_{\ell + 1,1;1}$ . As $p_{\ell + 1,1;1} = p_{\ell + 1,3;3}$ from Equation (1) and $p_{\ell,2;3}$ and $p_{\ell,3;2}$ must be almost the average of $p_{\ell,2;2}$ and $p_{\ell,3;3}$ (Equation (3)), we can write

$$
\max _ {i, j \in \{2, 3 \}} p _ {\ell + 1, i; j} \leq \frac {7}{9} \max _ {i, j \in \{2, 3 \}} p _ {\ell , i; j}.
$$

This implies that the probability of correlation at any output character under the two random set of phrasebooks decays at the rate of $\frac{7}{9}$ . Solving the recurrence will give:

$$
\max _ {i, j \in \{2, 3 \}} p _ {d + 1, i; j} \leq \left(\frac {7}{9}\right) ^ {d - 1} p _ {2, i; j} = \left(\frac {7}{9}\right) ^ {d - 1} \frac {1}{3}.
$$

As $p_{d+1,3;3}$ should be equal to $p_{d+1,1;1}$ , this implies that with probability at least $1 - p_{d+1,3;3} = 1 - \left(\frac{7}{9}\right)^{d-1} \frac{1}{3}$ , the following must hold true:

$$
\text { Correlation } (\mathbf {M L T} _ {\Pi^ {\alpha}} (\cdot) _ {1}, \mathbf {M L T} _ {\Pi^ {\beta}} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2 d})) = 0.
$$

Lemma F.14. Suppose $f, g : \{0, 1\}^{k} \to \{0, 1\}^{2}$ denote 2 functions for some arbitrary k, satisfying conditions for uniformity in output distribution, i.e.

$$
\mathbb {E} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {k})} f (x) _ {j} = 1 / 2, \quad \mathbb {E} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {k})} g (x) _ {j} = 1 / 2
$$

$$
\mathbb {E} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {k})} f (x) _ {1} \oplus f (x) _ {2} = 1 / 2, \quad \mathbb {E} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {k})} g (x) _ {1} \oplus g (x) _ {2} = 1 / 2
$$

for all $j \in \{1, 2\}$ . Then, the following holds true:

1. Both output character pairs have correlations between $f$ and $g$ : If

$$
\begin{array}{l} (e i t h e r) \text { Correlation } (f (\cdot) _ {1}, g (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1 \text {   and   } \text { Correlation } (f (\cdot) _ {2}, g (\cdot) _ {2}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1 \\ (o r) \operatorname{Correlation} (f (\cdot) _ {1}, g (\cdot) _ {2}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1 \text {   and   } \operatorname{Correlation} (f (\cdot) _ {2}, g (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1, \\ \end{array}
$$

then for two randomly picked phrasebooks $\pi^{\alpha},\pi^{\beta}$

(a) Correlation of an output character pair of $\pi^{\alpha}(f(\cdot))$ and $\pi^{\beta}(g(\cdot))$ : Fix an $i, j \in \{1, 2\}$ . With probability $1/3$ w.r.t. the random selections of $\pi^{\alpha}, \pi^{\beta}$ ,

$$
\text { Correlation } (\pi^ {\alpha} (f (\cdot)) _ {i}, \pi^ {\alpha} (g (\cdot)) _ {j}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1
$$

(b) Correlation of both character pairs in output of $\pi^{\alpha}(f(\cdot))$ and $\pi^{\beta}(g(\cdot))$ : With probability $1/3$ w.r.t. the random selections of $\pi^{\alpha}, \pi^{\beta}$ , one of the following two conditions hold true.

$$
(e i t h e r) C o r r e l a t i o n (\pi^ {\alpha} (f (\cdot)) _ {1}, \pi^ {\alpha} (g (\cdot)) _ {1}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1 a n d C o r r e l a t i o n (\pi^ {\alpha} (f (\cdot)) _ {2}, \pi^ {\alpha} (g (\cdot)) _ {2}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1
$$

$$
(o r) \operatorname{Correlation} (\pi^ {\alpha} (f (\cdot)) _ {1}, \pi^ {\alpha} (g (\cdot)) _ {2}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1 \text {and} \operatorname{Correlation} (\pi^ {\alpha} (f (\cdot)) _ {2}, \pi^ {\alpha} (g (\cdot)) _ {1}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1
$$

2. Only a pair of characters have correlations between f and g: If there exists only one pair $i, j \in \{0, 1\}$ such that

$$
\text { Correlation } (f (\cdot) _ {i}, g (\cdot) _ {j}, \mathcal {U} (\{0, 1 \} ^ {k}) = 1,
$$

and for all other $i', j'$ , Correlation $(f(\cdot)_{i'}, g(\cdot)_{j'}, \mathcal{U}(\{0,1\}^k)) = 0$ , then

(a) Fix any $i', j' \in \{1, 2\}$ . With probability $\frac{1}{9}$ w.r.t. the random selections of $\pi^{\alpha}, \pi^{\beta}$ ,

$$
\text { Correlation } (\pi^ {\alpha} (f (\cdot)) _ {i ^ {\prime}}, \pi^ {\beta} (g (\cdot)) _ {j ^ {\prime}}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1.
$$

If the above condition holds true, for all other $\bar{i}, \bar{j}$ pairs,

$$
\text { Correlation } (\pi^ {\alpha} (f (\cdot)) _ {\bar {i}}, \pi^ {\beta} (g (\cdot)) _ {\bar {j}}, \mathcal {U} (\{0, 1 \} ^ {k})) = 0.
$$

3. No correlations between $f$ and $g$ : If for all pairs $i, j \in \{1, 2\}$ , Correlation $(f(\cdot)_i, g(\cdot)_j, \mathcal{U}(\{0, 1\}^k)) = 0$ , then for all pairs $i', j' \in \{1, 2\}$ ,

$$
\text { Correlation } (\pi^ {\alpha} (f (\cdot)) _ {i ^ {\prime}}, \pi^ {\beta} (g (\cdot)) _ {j ^ {\prime}}, \mathcal {U} (\{0, 1 \} ^ {k})) = 0.
$$

Proof. We will prove each case separately.

1. Both output character pairs have correlations between $f$ and $g$ : We will consider the case when $\text{Correlation}(f(\cdot)_1, g(\cdot)_1, \mathcal{U}(\{0, 1\}^k)) = 1$ and $\text{Correlation}(f(\cdot)_2, g(\cdot)_2, \mathcal{U}(\{0, 1\}^k)) = 1$ , proof for the other case is similar. Then, there are 4 cases possible.

$$
\text { Subcase   1: } f (x) _ {1} = g (x) _ {1}, f (x) _ {2} = g (x) _ {2} \quad \text { for   all } x \in \{0, 1 \} ^ {k}
$$

$$
\text { Subcase   2: } f (x) _ {1} = \mathbf {n o t} (g (x) _ {1}), f (x) _ {2} = g (x) _ {2} \quad \text { for   all } x \in \{0, 1 \} ^ {k}
$$

$$
\text { Subcase   3: } f (x) _ {1} = g (x) _ {1}, f (x) _ {2} = \mathbf {n o t} (g (x) _ {2}) \quad \text { for   all } x \in \{0, 1 \} ^ {k}
$$

$$
\text { Subcase   4: } f (x) _ {1} = \mathbf {n o t} (g (x) _ {1}), f (x) _ {2} = \mathbf {n o t} (g (x) _ {2}) \quad \text { for   all } x \in \{0, 1 \} ^ {k}
$$

Because $f(\cdot)$ and $g(\cdot)$ are inputs to the phrasebooks $\pi^{\alpha}$ and $\pi^{\beta}$ respectively and $f(\cdot)$ and $g(\cdot)$ are related by selective not operations in their outputs in each of the possible 4 subcases, the output of $\pi^{\beta}(g(\cdot))$ can be replaced by $\tilde{\pi}^{\beta}(f(\cdot))$ for a mirror map $\tilde{\pi}^{\beta}$ of $\pi^{\beta}$ . We showcase this formally for one subcase, say subcase 2; argument for other subcases are similar.

Suppose Subcase 2 is true: Then, for any two phrasebooks $\pi^{\alpha}$ and $\pi^{\beta}$ , suppose $\tilde{\pi}^{\beta}$ denotes a mirror map of $\pi^{\beta}$ such that

$$
\tilde {\pi} ^ {\beta} (x) _ {1} = \mathbf {n o t} (\pi^ {\alpha} (x) _ {1}), \tilde {\pi} ^ {\beta} (x) _ {2} = \pi^ {\alpha} (x) _ {2}, \quad \mathrm{forall} x \in \{0, 1 \} ^ {2}.
$$

Such a map exists by Lemma F.21. Then, for a pair $i, j \in \{1, 2\}$ ,

$$
\begin{array}{l} \operatorname{Correlation} \left(\pi^ {\alpha} (f (\cdot)) _ {i}, \pi^ {\beta} (g (\cdot)) _ {j}, \mathcal {U} \left(\{0, 1 \} ^ {k}\right)\right) \\ = \text { Correlation } (\pi^ {\alpha} ((f (\cdot) _ {1}, f (\cdot) _ {2})) _ {i}, \pi^ {\beta} ((g (\cdot) _ {1}, g (\cdot) _ {2})) _ {j}, \mathcal {U} (\{0, 1 \} ^ {k})) (7) \\ = \text { Correlation } (\pi^ {\alpha} ((f (\cdot) _ {1}, f (\cdot) _ {2})) _ {i}, \pi^ {\beta} ((\mathbf {n o t} (f (\cdot) _ {1}), f (\cdot) _ {2})) _ {j}, \mathcal {U} (\{0, 1 \} ^ {k})) (8) \\ = \text { Correlation } (\pi^ {\alpha} ((f (\cdot) _ {1}, f (\cdot) _ {2})) _ {i}, \tilde {\pi} ^ {\beta} ((f (\cdot) _ {1}, f (\cdot) _ {2})) _ {j}, \mathcal {U} (\{0, 1 \} ^ {k})) (9) \\ = \text { Correlation } (\pi^ {\alpha} (\cdot) _ {i}, \tilde {\pi} ^ {\beta} (\cdot) _ {j}, \mathcal {U} (\{0, 1 \} ^ {2})). (10) \\ \end{array}
$$

Reasoning for each step is as follows.

- Because the outputs of $f$ and $g$ are a 2-tuple on any input, Equation (7) simply replaces $f(x)$ (similarly $g(x)$ ) as $(f(x)_1, f(x)_2)$ for any input $x$ .   
- Equation (8) uses the relation between the outputs of $f$ and $g$ when subcase 2 is true.   
- Equation (9) simply replaces $\pi^{\beta}$ with $\tilde{\pi}^{\beta}$ due to their relation as mirror phrasebooks.   
- Finally, because of the assumption on the output of $f$ , i.e. $\mathbb{E}_{x\sim \mathcal{U}(\{0,1\}^k)}f(x)_j = 1 / 2, \mathbb{E}_{x\sim \mathcal{U}(\{0,1\}^k)}f(x)_1 \oplus f(x)_2 = 1 / 2$ , the distribution of outputs of $f$ is identical to $\mathcal{U}(\{0,1\}^2)$ .

Hence, Correlation $(\pi^{\alpha}(f(\cdot))_{i},\pi^{\beta}(g(\cdot))_{j},\mathcal{U}(\{0,1\}^{k}))$ boils down to Correlation $(\pi^{\alpha}(\cdot)_{i},\tilde{\pi}^{\beta}(\cdot)_{j},\mathcal{U}(\{0,1\}^{2}))$ . We can then use Lemma F.22 to show the probabilities of each of the desired conditions.

2. Only a pair of characters in output have correlations between $f$ and $g$ : For typographical simplicity, consider the case when

$$
\operatorname{Correlation} (f (\cdot) _ {1}, g (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {k})) = 1,
$$

and for all pairs $i', j'$ with atleast one of them being not equal to 1, $\text{Correlation}(f(\cdot)_{i'}, g(\cdot)_{j'}, \mathcal{U}(\{0,1\}^k)) = 0$ . The argument when the condition holds for other i, j pairs can be similarly handled. Then, there are 2 cases possible.

$$
\text { Subcase   1:   } f (x) _ {1} = g (x) _ {1}, \text {   for   all   } x \in \{0, 1 \} ^ {k}
$$

$$
\text { Subcase   2: } f (x) _ {1} = \mathbf {n o t} (g (x) _ {1}), \text { for   all } x \in \{0, 1 \} ^ {k},
$$

while in each of these pairs, $(f(\cdot)_{2}, g(\cdot)_{2})$ , $(f(\cdot)_{2}, g(\cdot)_{1})$ , and $(f(\cdot)_{1}, g(\cdot)_{2})$ , the output bits of f and g are completely independent of each other. We only consider subcase 1; subcase 2 can be handled using mirror phrasebooks similar to the proof for case 1 (in particular with Equation (9)).

For simplicity, we will argue for $i'$ , $j' = 1, 1$ ; the argument about other $i'$ , $j'$ pairs are similar. The proof will follow by two steps,

- Step (a): We first argue about the probability with which $\text{Correlation}(\pi^{\alpha}(f(\cdot))_{1},\pi^{\beta}(g(\cdot))_{1},\mathcal{U}(\{0,1\}^{k}))$ is equal to 1,   
- Step (b): We then argue that if the condition in (a) holds true, then $\text{Correlation}(\pi^{\alpha}(f(\cdot))_{i'}, \pi^{\beta}(g(\cdot))_{j'}, \mathcal{U}(\{0,1\}^{k}))$ must be 0 for any other pair $i', j'$ where atleast one of them is not equal to 1.

Step (a): For simplicity, we will assume that $\pi^{\alpha},\pi^{\beta}\in\{\Delta_{1},\Delta_{2},\cdots,\Delta_{6}\}$ defined in Table 7; other cases can be handled similarly as Lemma F.23.

Correlation $(\pi^{\alpha}(f(\cdot))_{1},\pi^{\beta}(g(\cdot))_{1},\mathcal{U}(\{0,1\}^{k}))$

$$
\begin{array}{l} = \text { Correlation } (\pi^ {\alpha} ((f (\cdot) _ {1}, f (\cdot) _ {2})) _ {1}, \pi^ {\beta} ((g (\cdot) _ {1}, g (\cdot) _ {2})) _ {1}, \mathcal {U} (\{0, 1 \} ^ {k})) (11) \\ = \text { Correlation } (\pi^ {\alpha} ((f (\cdot) _ {1}, f (\cdot) _ {2})) _ {1}, \pi^ {\beta} ((f (\cdot) _ {1}, g (\cdot) _ {2})) _ {1}, \mathcal {U} (\{0, 1 \} ^ {k})) (12) \\ = \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {k})} \left[ \pi^ {\alpha} \left(\left(f (x) _ {1}, f (x) _ {2}\right)\right) _ {1} = \pi^ {\beta} \left(\left(f (x) _ {1}, g (x) _ {2}\right)\right) _ {1} \right] \right. (13) \\ - \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {k}} \left[ \pi^ {\alpha} ((f (x) _ {1}, f (x) _ {2})) _ {1} \neq \pi^ {\beta} ((f (x) _ {1}, g (x) _ {2})) _ {1} \right] \\ = \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \})} \left[ \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \pi^ {\alpha} ((x, y)) _ {1} = \pi^ {\beta} ((x, y ^ {\prime})) _ {1} \right] - \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \pi^ {\alpha} ((x, y)) _ {1} = \pi^ {\beta} ((x, y ^ {\prime})) _ {1} \right] \right] \right| (14) \\ \end{array}
$$

The reasoning for each step is as follows:

- Because the outputs of $f$ and $g$ are a 2-tuple on any input, Equation (11) simply replaces $f(x)$ (similarly $g(x)$ ) as $(f(x)_1, f(x)_2)$ for any input $x$ .   
- Equation (12) uses the relation between the outputs of $f$ and $g$ when subcase 1 is true.   
• Equation (9) simply writes the definition of correlation.   
- Finally, because of the assumption on the output of $f$ , i.e. $\mathbb{E}_{x\sim \mathcal{U}(\{0,1\}^k)}f(x)_j = 1 / 2, \mathbb{E}_{x\sim \mathcal{U}(\{0,1\}^k)}f(x)_1 \oplus f(x)_2 = 1 / 2$ , the distribution of outputs of $f$ can be shown to be identical to $\mathcal{U}(\{0,1\}^2)$ .

From the definitions of $\pi^{\alpha},\pi^{\beta}\in\{\Delta_{1},\Delta_{2},\cdots,\Delta_{6}\}$ in Table 7, the first character in the outputs of $\pi^{\alpha}$ and $\pi^{\beta}$ can be expressed using 3 operators on the input characters, which are $copy_{1},copy_{2}$ , and xor. These operators are independently selected for $\pi^{\alpha}$ and $\pi^{\beta}$ , as they are randomly picked from these 6 possibilities. Formally, for any tuple of variables $(x,y)$ and $(x,y')$ ,

$$
\mathbf {c o p y} _ {1} (x, y) = x, \quad \mathbf {c o p y} _ {2} (x, y) = y, \quad \mathbf {x o r} (x, y) = x \oplus y,
$$

$$
\mathbf {c o p y} _ {1} (x, y ^ {\prime}) = x, \quad \mathbf {c o p y} _ {2} (x, y ^ {\prime}) = y ^ {\prime}, \quad \mathbf {x o r} (x, y ^ {\prime}) = x \oplus y ^ {\prime}.
$$

However, because $x, y, y'$ are independent variables, only operations $\mathbf{copy}_1(x, y)$ and $\mathbf{copy}_1(x, y')$ for $(x, y, y') \sim \mathcal{U}(\{0, 1\}^3)$ will be correlated. This can be verified by writing the definition of correlation and using $\mathbf{copy}_1$ for the first character output of $\pi^\alpha$ and $\pi^\beta$ :

$$
\begin{array}{l} \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \})} \left[ \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \pi^ {\alpha} ((x, y)) _ {1} = \pi^ {\beta} ((x, y ^ {\prime})) _ {1} \right] - \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \pi^ {\alpha} ((x, y)) _ {1} = \pi^ {\beta} ((x, y ^ {\prime})) _ {1} \right] \right] \right| \\ = \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \})} \left[ \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \mathbf {c o p y} _ {1} (x, y) = \mathbf {c o p y} _ {1} (x, y ^ {\prime}) ] - \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \mathbf {c o p y} _ {1} (x, y) \neq \mathbf {c o p y} _ {1} (x, y ^ {\prime}) ] \right] \right| = 1, \\ \end{array}
$$

as the first term is 1 and the second term is 0. However, if you pick any other pair of operations, the correlation will be 0. We demonstrate by using $copy_{1}$ and xor for $\pi^{\alpha}$ and $\pi^{\beta}$ respectively.

$$
\begin{array}{l} \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \})} \left[ \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \pi^ {\alpha} ((x, y)) _ {1} = \pi^ {\beta} ((x, y ^ {\prime})) _ {1} \right] - \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \pi^ {\alpha} ((x, y)) _ {1} = \pi^ {\beta} ((x, y ^ {\prime})) _ {1} \right] \right] \right| \\ = \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \})} \left[ \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \mathbf {c o p y} _ {1} (x, y) = \mathbf {x o r} (x, y ^ {\prime}) ] - \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \mathbf {c o p y} _ {1} (x, y) \neq \mathbf {x o r} (x, y ^ {\prime}) ] \right] \right| \\ = \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \})} \left[ \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ x = x \oplus y ^ {\prime} ] - \operatorname * {P r} _ {y, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ x \neq x \oplus y ^ {\prime} ] \right] \right| = 0, \\ \end{array}
$$

as both terms are equal to $\frac{1}{2}$ in the final step. By a counting argument, one can reason that the probability of $corr(\pi^{\alpha}(f(\cdot))_{1},\pi^{\beta}(g(\cdot))_{1},\mathcal{U}(\{0,1\}^{k}))$ being 1 is equal to the probability of $copy_{1}$ being selected to define the first characters of both $\pi^{\alpha}$ and $\pi^{\beta}$ , which will be equal to $\frac{1}{9}$ .

Step (b): Now, say we have selected a pair $\pi^{\alpha}$ and $\pi^{\beta}$ such that $\text{Correlation}(\pi^{\alpha}(f(\cdot))_{1},\pi^{\beta}(g(\cdot))_{1},\mathcal{U}(\{0,1\}^{k})) = 1$ . From the proof of step (a), this is only possible when $\mathbf{copy}_1$ was selected to define the first characters in the outputs of

$\pi^{\alpha}$ and $\pi^{\beta}$ . Any other operation pairs for defining the first characters in the outputs of $\pi^{\alpha}$ and $\pi^{\beta}$ would have meant correlation to be 0.

We can use this same argument to show that for any other pair $i', j'$ where at least one of them is not equal to 1. Correlation $(\pi^{\alpha}(f(\cdot))_{i'}, \pi^{\beta}(g(\cdot))_{j'}, \mathcal{U}(\{0,1\}^{k}))$ must be 0. This is because once $copy_{1}$ has been used to define the operation for the first character, it can't be used to define the operation for the second character (please refer at the truth tables for $\{\Delta_{1},\cdots,\Delta_{6}\}$ in Table 7). Following a similar argument, we can then show that correlation between output characters $\pi^{\alpha}(f(\cdot))_{i'}, \pi^{\beta}(g(\cdot))_{j'}$ will be 0, as $copy_{1}$ can't be used to define at least one of these characters.

3. The third case is when none of the above conditions hold true. That is, for any pair $i, j \in \{1, 2\}$ ,

$$
\operatorname{Correlation} (f (\cdot) _ {i}, g (\cdot) _ {j}, \mathcal {U} (\{0, 1 \} ^ {k})) = 0.
$$

In such case, following similar arguments as case 1 and 2, one can show that for any $i, j \in \{1, 2\}$ , for any choice of $\pi^{\alpha}$ and $\pi^{\beta}$ :

$$
\begin{array}{l} \text { Correlation } (\pi^ {\alpha} (f (\cdot)) _ {i}, \pi^ {\beta} (g (\cdot)) _ {j}, \mathcal {U} (\{0, 1 \} ^ {k})) \\ = \left| \operatorname * {P r} _ {x, y \sim \mathcal {U} (\{0, 1 \} ^ {2})} \operatorname * {P r} _ {x ^ {\prime}, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \pi^ {\alpha} ((x, y)) _ {i} = \pi^ {\beta} ((x ^ {\prime}, y ^ {\prime})) _ {j} \right] \right. \\ - \operatorname * {P r} _ {x, y \sim \mathcal {U} (\{0, 1 \} ^ {2})} \operatorname * {P r} _ {x ^ {\prime}, y ^ {\prime} \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \pi^ {\alpha} ((x, y)) _ {i} \neq \pi^ {\beta} ((x ^ {\prime}, y ^ {\prime})) _ {j} \right] \Bigg | \\ = 0, \\ \end{array}
$$

due to independence of inputs to $\pi^{\alpha}$ and $\pi^{\beta}$ .

![](images/33a829e040d5582e1eb8d8b9977e89cf529c3b2b4c8db85ac3d0d39626a54d66.jpg)

Lemma F.15. At any step i of $MLT(d,2)$ with phrasebooks $\{\pi_{\ell}\}_{\ell=1}^{d}$ , the following relation holds true for the intermediate outputs $\{s_{i}\}_{i=2}^{d+1}$ on an input $s_{1}\in\{0,1\}^{L}$ :

$$
\left(\boldsymbol {s} _ {i + 1, 1}, \boldsymbol {s} _ {i + 1, 2}\right) = \pi_ {i} \left(\left(\boldsymbol {s} _ {i, 2}, \boldsymbol {s} _ {i, 3}\right)\right).
$$

Proof. MLT(d,2) has 2 primary steps: Circular shift and Translate. By Circular shift, first, we first get sequence $\tilde{s}_i$ , where for any $j\in [L]$ we have $\tilde{s}_{i,j} = s_{i,(j + 1)\% L}$ . After Translate step,

$$
(\boldsymbol {s} _ {i + 1, 1}, \boldsymbol {s} _ {i + 1, 2}) = \pi_ {i} ((\tilde {\boldsymbol {s}} _ {i, 1}, \tilde {\boldsymbol {s}} _ {i, 2})) = \pi_ {i} ((\boldsymbol {s} _ {i, 2}, \boldsymbol {s} _ {i, 3})).
$$

![](images/c5693251cec1e7031a05d1b1674eaada72b4bb3eb49f445ec60d8764353d996b.jpg)

Lemma F.16. At any step $i$ of $\mathbf{MLT}(d,2)$ with phrasebooks $\{\pi_{\ell}\}_{\ell = 1}^{d}$ , the following conditions hold true for the intermediate output $s_i$ .

$$
\mathbb {E} _ {\boldsymbol {s} _ {1} \sim \mathcal {U} (\{0, 1 \} ^ {L})} \boldsymbol {s} _ {i, j} = \frac {1}{2}, f o r a l l 1 \leq j \leq L
$$

$$
\mathbb {E} _ {\boldsymbol {s} _ {1} \sim \mathcal {U} (\{0, 1 \} ^ {L})} \boldsymbol {s} _ {i, j} \oplus \boldsymbol {s} _ {i, j ^ {\prime}} = \frac {1}{2}, f o r a l l j \neq j ^ {\prime}.
$$

The above conditions are equivalent to showing that $s_{i,j}$ behaves like a uniformly random boolean variable, independent of any other character $s_{i,j'}$ for all coordinates $j' \neq j$ .

Proof. The proof will follow by induction on the output of the translation task at each step. We will show the result for coordinate j = 1 in $s_{i}$ ; similar argument holds for other coordinates j.

Base condition: At layer $i = 1$ , the $s_1$ represents the input sequence from $\mathcal{U}(\{0,1\}^L)$ . By definition of uniform distribution, the conditions hold true for the input.

<table><tr><td>Operator</td><td>Expected value</td><td colspan="3">Expected  $\oplus$  value with operator</td></tr><tr><td></td><td></td><td> $copy_1$ </td><td> $copy_2$ </td><td>xor</td></tr><tr><td> $copy_1$ </td><td> $\mathbb{E}_{s_1} s_{i-1,2} = 1/2$ </td><td>-</td><td> $\mathbb{E}_{s_1} s_{i-1,2} \oplus s_{i-1,3} = 1/2$ </td><td> $\mathbb{E}_{s_1} s_{i-1,3} = 1/2$ </td></tr><tr><td> $copy_2$ </td><td> $\mathbb{E}_{s_1} s_{i-1,3} = 1/2$ </td><td> $\mathbb{E}_{s_1} s_{i-1,2} \oplus s_{i-1,3} = 1/2$ </td><td>-</td><td> $\mathbb{E}_{s_1} s_{i-1,2} = 1/2$ </td></tr><tr><td>xor</td><td> $\mathbb{E}_{s_1} s_{i-1,2} \oplus s_{i-1,3} = 1/2$ </td><td> $\mathbb{E}_{s_1} s_{i-1,3} = 1/2$ </td><td> $\mathbb{E}_{s_1} s_{i-1,2} = 1/2$ </td><td>-</td></tr></table>

Table 6. $\mathbb{E}\pmb{s}_{i,1}$ (similarly $\mathbb{E}\pmb{s}_{i,2}$ ) and $\mathbb{E}\pmb{s}_{i,1} \oplus \pmb{s}_{i,2}$ under different $\pi_i$ phrasebooks, defined by copy and xor operations on $s_{i-1,2}$ and $s_{i-1,3}$ .

Induction step: Argument for general i > 1: Suppose the conditions are true for all layers $1 \leq \ell < i$ . Then for layer i, we will provide an argument for the condition to hold true for j = 1 and j = 2, arguments for other js will extend similarly. By Lemma F.15,

$$
\left(\boldsymbol {s} _ {i, 1}, \boldsymbol {s} _ {i, 2}\right) = \pi_ {i} \left(\left(\boldsymbol {s} _ {i - 1, 2}, \boldsymbol {s} _ {i - 1, 3}\right)\right).
$$

From Lemma F.21, we have each output character can be defined in terms of copy, xor, and not operations on input characters. Then, the relations for $s_{i,1}$ , $s_{i,2}$ can be computed as follows:

- If a map $\pi^{\alpha}$ is selected from $\text{Mirrorset}(\pi)$ for some $\pi \in \{\Delta_1, \cdots, \Delta_6\}$ , then one can show that the conditions hold true for $\pi_i = \pi^\alpha$ if the conditions hold true for $\pi_i = \pi$ . This follows because the output of $\pi^\alpha$ follows from the output of $\pi$ by selective not operations to the output of $\pi$ . As not operation won't change the expected values of a variable which behaves like a random boolean variable, the argument follows.   
- Now, we show that for $\pi_i = \pi$ for some $\pi \in \{\Delta_1, \cdots, \Delta_6\}$ , the conditions hold true. For these phrasebooks, the output characters are defined by copy and xor operations. We use the definitions of these phrasebooks from Table 7 and show the expected values $s_{i,1}$ in terms of expected values of $s_{i-1,1}$ and $s_{i-1,2}$ , and the expected values $s_{i,1} \oplus s_{i,2}$ in terms of expected values of $s_{i-1,1}$ and $s_{i-1,2}$ in Table 6.

Both of the above arguments then can be combined to show that

$$
\mathbb {E} _ {\boldsymbol {s} _ {1} \sim \mathcal {U} (\{0, 1 \} ^ {L})} \boldsymbol {s} _ {i, j} = \frac {1}{2}, \text {   for   } j \in \{1, 2 \}
$$

$$
\mathbb {E} _ {\boldsymbol {s} _ {1} \sim \mathcal {U} (\{0, 1 \} ^ {L})} \boldsymbol {s} _ {i, 1} \oplus \boldsymbol {s} _ {i, 2} = \frac {1}{2}.
$$

We can similarly extend the argument for expectation of each individual character to other positions $j > 2$ , i.e. $\mathbb{E}_{\boldsymbol{s}_1 \sim \mathcal{U}(\{0,1\}^L)} \boldsymbol{s}_{i,j} = \frac{1}{2}$ for all other possible $js$ . We can also similarly extend the argument for joint expectation of two consecutive characters $\boldsymbol{s}_{i,2k+1} \oplus \boldsymbol{s}_{i,2k+2}$ for any general $k$ .

The remaining argument will be to show that characters that don't form a consecutive 2-tuple are going to be independent of each other as well. Consider any two 2-tuples $(\pmb{s}_{i,2k + 1},\pmb{s}_{i,2k + 2})$ and $(\pmb{s}_{i,2k' + 1},\pmb{s}_{i,2k' + 2})$ , with $k\neq k'$ . By adapting Lemma F.15, one can show that

$$
\left(\boldsymbol {s} _ {i, 2 k + 1}, \boldsymbol {s} _ {i, 2 k + 2}\right) = \pi_ {i} \left(\left(\boldsymbol {s} _ {i - 1, 2 k + 2}, \boldsymbol {s} _ {i, 2 k + 3}\right)\right)
$$

$$
\left(\pmb {s} _ {i, 2 k ^ {\prime} + 1}, \pmb {s} _ {i, 2 k ^ {\prime} + 2}\right) = \pi_ {i} \big (\left(\pmb {s} _ {i - 1, 2 k ^ {\prime} + 2}, \pmb {s} _ {i, 2 k ^ {\prime} + 3}\right) \big)
$$

The primary thing to note here is that both the tuples depend on two distinct tuples in layer i - 1. By induction assumption, characters across these two 2-tuples are independent of each other. As $\pi_{i}$ is a bijective map, this will also suggest that characters across the 2-tuples in the resulting output must also be independent of each other. This will give the final argument.

![](images/9b623c13264c8ba8125a01fb72782f2541b6e04c5f3e17deaa36f31d7fc2d3cc.jpg)

Lemma F.17. Under the assumption that sequence lengths L are even, for any $2 \leq i \leq d + 1$ , where $s_{i,j}^{\alpha}, s_{i,j}^{\beta}$ denote the output after i - 1st translation step for a random sequence $\mathbf{s}_{1} \sim \mathcal{U}(\{0,1\}^{L})$ at any position $1 \leq j \leq L$ under the two set of phrasebooks $\Pi_{:i}^{\alpha} := \{\pi_{\ell}^{\alpha}\}_{\ell=1}^{i-1}$ and $\Pi_{:i}^{\beta} := \{\pi_{\ell}^{\beta}\}_{\ell=1}^{i-1}$ , the following holds true for all positions j:

Correlation( $\boldsymbol{s}_{i,j}^{\alpha}, \boldsymbol{s}_{i,j}^{\beta}, \mathcal{U}(\{0,1\}^{L})$ ) = Correlation( $\boldsymbol{s}_{i,(j+2k)\%L}^{\alpha}, \boldsymbol{s}_{i,(j+2k)\%L}^{\beta}, \mathcal{U}(\{0,1\}^{L})$ ), for all integer $k$ .

Proof. Fix an integer k. By Definition F.1, the necessary condition would be to show

$$
\left| 1 - 2 \mathbb {E} _ {\pmb {s} _ {1} \sim \{0, 1 \} ^ {L}} \pmb {s} _ {i, j} ^ {\alpha} \oplus \pmb {s} _ {i, j} ^ {\beta} \right| = \left| 1 - 2 \mathbb {E} _ {\pmb {s} _ {1} \sim \{0, 1 \} ^ {L}} \pmb {s} _ {i, (j + 2 k) \% L} ^ {\alpha} \oplus \pmb {s} _ {i, (j + 2 k) \% L} ^ {\beta} \right|.
$$

We will look at the behavior of the function $h(\boldsymbol{s}_1) = \boldsymbol{s}_{i,j}^\alpha \oplus \boldsymbol{s}_{i,j}^\beta$ . Denote $\mathcal{C}_k$ as a circular operation that takes a sequence $\boldsymbol{s}_1$ and returns a shifted sequence $\mathcal{O}\boldsymbol{s}_1$ , i.e. if $\mathcal{O}\boldsymbol{s}_1 = \mathcal{C}_k(\boldsymbol{s}_1)$ then for all $j$ , $\mathcal{O}\boldsymbol{s}_{1,j} = \boldsymbol{s}_{1,(j + 2k)\% L}$ . Note that, we can also define an inverse function $\mathcal{C}_k^{-1}$ and $\boldsymbol{s}_1 = \mathcal{C}_k^{-1}(\mathcal{C}_k(\boldsymbol{s}_1))$ .

From the definition of $MLT_{\Pi}$ for any phrasebook $\Pi$ , we have for any input $s_{1}$ :

$$
\pmb {s} _ {i} = T _ {\pi_ {i - 1}} \circ \dots \circ T _ {\pi_ {1}} (\pmb {s} _ {1}),
$$

where for any level i, $T_{\pi_{i}}$ denotes the translation step at level i that includes Circular shift and Translate using $\pi_{i}$ .

In Lemma F.18, we show that for any translation task $MLT_{\Pi}$ and any input $s_{1}$ ,

$$
T _ {\pi_ {i - 1}} \circ \dots \circ T _ {\pi_ {1}} (\boldsymbol {s} _ {1}) = \mathcal {C} _ {k} ^ {- 1} \left(T _ {\pi_ {i - 1}} \circ \dots \circ T _ {\pi_ {1}} \circ \mathcal {C} _ {k} (\boldsymbol {s} _ {1})\right).
$$

On input $\mathcal{O}\boldsymbol{s}_1 = \mathcal{C}_k(\boldsymbol{s}_1)$ , if $\mathcal{O}\boldsymbol{s}_i^\alpha$ and $\mathcal{O}\boldsymbol{s}_i^\beta$ represent the output of translations using $\Pi_{:i}^{\alpha} := \{\pi_\ell^\alpha\}_{\ell=1}^{i-1}$ and $\Pi_{:i}^\beta := \{\pi_\ell^\beta\}_{\ell=1}^{i-1}$ respectively, then the above statement says that

$$
\boldsymbol {s} _ {i} ^ {\alpha} = \mathcal {C} _ {k} ^ {- 1} \left(^ {\circledcirc} \boldsymbol {s} _ {i} ^ {\alpha}\right), \quad \boldsymbol {s} _ {i} ^ {\beta} = \mathcal {C} _ {k} ^ {- 1} \left(^ {\circledcirc} \boldsymbol {s} _ {i} ^ {\beta}\right)
$$

$$
\text {(or)} \mathcal {C} _ {k} (\boldsymbol {s} _ {i} ^ {\alpha}) = \mathcal {O} \boldsymbol {s} _ {i} ^ {\alpha}, \quad \mathcal {C} _ {k} (\boldsymbol {s} _ {i} ^ {\beta}) = \mathcal {O} \boldsymbol {s} _ {i} ^ {\beta}
$$

Thus, for any position j, we will have $\mathcal{C}_{k}\left(\boldsymbol{s}_{i}^{\alpha}\right)_{j}=\mathcal{O}\boldsymbol{s}_{i,j}^{\alpha}$ (and similarly for $\mathcal{O}\boldsymbol{s}_{i,j}^{\beta}$ ). We can then compute the value of h on input $\mathcal{O}s_{1}$ as follows:

$$
\begin{array}{l} h (\mathcal {O} \boldsymbol {s} _ {1}) = \mathcal {O} \boldsymbol {s} _ {i, j} ^ {\alpha} \oplus \mathcal {O} \boldsymbol {s} _ {i, j} ^ {\beta} \\ = \mathcal {C} _ {k} \left(\boldsymbol {s} _ {i} ^ {\alpha}\right) _ {j} \oplus \mathcal {C} _ {k} \left(\boldsymbol {s} _ {i} ^ {\beta}\right) _ {j} \\ = \boldsymbol {s} _ {i, (j + 2 k) \% L} ^ {\alpha} \oplus \boldsymbol {s} _ {i, (j + 2 k) \% L} ^ {\beta} \\ \end{array}
$$

The last step follows from using the definition of $C_{k}$ . On the other hand,

$$
\mathbb {E} _ {\mathcal {O} \boldsymbol {s} _ {1} \sim \mathcal {U} (\{0, 1 \} ^ {L})} h (\mathcal {O} \boldsymbol {s} _ {1}) = \mathbb {E} _ {\boldsymbol {s} _ {1} \sim \mathcal {U} (\{0, 1 \} ^ {L})} h (\boldsymbol {s} _ {1}),
$$

as the uniform distribution can be shown to not change under circular function $C_{k}$ . This will imply:

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {s} _ {1} \sim \mathcal {U} (\{0, 1 \} ^ {L})} \boldsymbol {s} _ {i, (j + 2 k) \% L} ^ {\alpha} \oplus \boldsymbol {s} _ {i, (j + 2 k) \% L} ^ {\beta} = \mathbb {E} _ {\boldsymbol {s} _ {1} \sim \{0, 1 \} ^ {L}} \boldsymbol {s} _ {i, j} ^ {\alpha} \oplus \boldsymbol {s} _ {i, j} ^ {\beta} \\ \Longrightarrow \left| 1 - 2 \mathbb {E} _ {\boldsymbol {s} _ {1} \sim \{0, 1 \} ^ {L}} \boldsymbol {s} _ {i, j} ^ {\alpha} \oplus \boldsymbol {s} _ {i, j} ^ {\beta} \right| = \left| 1 - 2 \mathbb {E} _ {\boldsymbol {s} _ {1} \sim \{0, 1 \} ^ {L}} \boldsymbol {s} _ {i, (j + 2 k) \% L} ^ {\alpha} \oplus \boldsymbol {s} _ {i, (j + 2 k) \% L} ^ {\beta} \right| \\ \implies \text {Correlation} (\boldsymbol {s} _ {i, j} ^ {\alpha}, \boldsymbol {s} _ {i, j} ^ {\beta}, \mathcal {U} (\{0, 1 \} ^ {L})) = \text {Correlation} (\boldsymbol {s} _ {i, (j + 2 k) \% L} ^ {\alpha}, \boldsymbol {s} _ {i, (j + 2 k) \% L} ^ {\beta}, \mathcal {U} (\{0, 1 \} ^ {L})). \\ \end{array}
$$

![](images/8eac6426ace86e288b35e2a59f7cc71e121391ad62315aeff9aa54d397a4d10f.jpg)

Lemma F.18. For any translation task $MLT_{\Pi}$ , at any level $i \leq d$ and any input $s_{1}$ ,

$$
T _ {\pi_ {i}} \circ \dots \circ T _ {\pi_ {1}} (\boldsymbol {s} _ {1}) = \mathcal {C} _ {k} ^ {- 1} \left(T _ {\pi_ {i}} \circ \dots \circ T _ {\pi_ {1}} \circ \mathcal {C} _ {k} (\boldsymbol {s} _ {1})\right).
$$

Proof. Denote $C_{k}$ as a circular operation that takes a sequence $s_{1}$ and returns a shifted sequence $\mathcal{O}s_{1}$ , i.e. if $\mathcal{O}s_{1} = \mathcal{C}_{k}(s_{1})$ then for all $j, \mathcal{O}s_{1,j} = s_{1,(j+2k)\%L}$ . Note that, we can also define an inverse function $C_{k}^{-1}$ and $s_{1} = \mathcal{C}_{k}^{-1}(\mathcal{C}_{k}(s_{1}))$ .

Recall that for any level i, $T_{\pi_{i}}$ denotes the translation step at level i that includes Circular shift, and Translate with $\pi_{i}$ . The argument will again follow by an induction step.

Base condition: i = 1: We need to show that

$$
T _ {\pi_ {1}} \left(\boldsymbol {s} _ {1}\right) = \mathcal {C} _ {k} ^ {- 1} \left(T _ {\pi_ {1}} \circ \mathcal {C} _ {k} \left(\boldsymbol {s} _ {1}\right)\right).
$$

To do so, we will look at the behavior of the first 2 characters. Argument for others can be extended. If $\mathcal{O} s_1 = \mathcal{C}_k(s_1)$ , $s_2 = T_{\pi_1}(s_1)$ , $\mathcal{O} s_2 = T_{\pi_1}(\mathcal{O} s_1)$ , then by Lemma F.15,

$$
\begin{array}{l} \left(\boldsymbol {s} _ {2, 1}, \boldsymbol {s} _ {2, 2}\right) = \pi_ {1} \left(\left(\boldsymbol {s} _ {1, 2}, \boldsymbol {s} _ {1, 3}\right)\right) \\ (^ {\circ} \boldsymbol {s} _ {2, 1}, ^ {\circ} \boldsymbol {s} _ {2, 2}) = \pi_ {1} ((^ {\circ} \boldsymbol {s} _ {1, 2}, ^ {\circ} \boldsymbol {s} _ {1, 3})). \\ \end{array}
$$

But by definition of $\mathcal{C}_k$ ,

$$
(\mathcal {O} \boldsymbol {s} _ {1, 2}, \mathcal {O} \boldsymbol {s} _ {1, 3}) = (\boldsymbol {s} _ {1, (2 + 2 k) \% L}, \boldsymbol {s} _ {1, (3 + 2 k) \% L}).
$$

On the other hand, Lemma F.15 can be adapted to give

$$
\big (\boldsymbol {s} _ {2, (1 + 2 k) \% L}, \boldsymbol {s} _ {2, (2 + 2 k) \% L} \big) = \pi_ {1} (\boldsymbol {s} _ {1, (2 + 2 k) \% L}, \boldsymbol {s} _ {1, (3 + 2 k) \% L}).
$$

Thus, we can show by combining the above 3 steps that

$$
\begin{array}{l} \left(^ {\circ} \boldsymbol {s} _ {2, 1}, ^ {\circ} \boldsymbol {s} _ {2, 2}\right) = \pi_ {1} \left(\left(^ {\circ} \boldsymbol {s} _ {1, 2}, ^ {\circ} \boldsymbol {s} _ {1, 3}\right)\right) \\ = \pi_ {1} (\boldsymbol {s} _ {1, (2 + 2 k) \% L}, \boldsymbol {s} _ {1, (3 + 2 k) \% L}) \\ = \left(\boldsymbol {s} _ {2, (1 + 2 k) \% L}, \boldsymbol {s} _ {2, (2 + 2 k) \% L}\right). \\ \end{array}
$$

We can extend the above argument to show that for any position j,

$$
\mathcal {O} _ {\boldsymbol {s} _ {2, j}} = \boldsymbol {s} _ {2, (j + 2 k) \% L},
$$

which by definition of $\mathcal{C}_k$ , implies

$$
{ } ^ { \circledcirc } \boldsymbol { s } _ { 2 } = \mathcal { C } _ { k } ( \boldsymbol { s } _ { 2 } ) \quad \text {   or   } \boldsymbol { s } _ { 2 } = \mathcal { C } _ { k } ^ { - 1 } \left( { } ^ { \circledcirc } \boldsymbol { s } _ { 2 } \right) ,
$$

which can be further simplified (using the notations: $\mathcal{O}\boldsymbol{s}_1 = \mathcal{C}_k(\boldsymbol{s}_1),\boldsymbol{s}_2 = T_{\pi_1}(\boldsymbol{s}_1),\mathcal{O}\boldsymbol{s}_2 = T_{\pi_1}(\mathcal{O}\boldsymbol{s}_1))$

$$
\begin{array}{l} \boldsymbol {s} _ {2} = \mathcal {C} _ {k} ^ {- 1} \left(^ {\circledcirc} \boldsymbol {s} _ {2}\right) \\ = \mathcal {C} _ {k} ^ {- 1} \left(T _ {\pi_ {1}} \left(^ {\circledcirc} \boldsymbol {s} _ {1}\right)\right) \\ = \mathcal {C} _ {k} ^ {- 1} \left(T _ {\pi_ {1}} \left(\mathcal {C} _ {k} \left(\boldsymbol {s} _ {1}\right)\right)\right) := \mathcal {C} _ {k} ^ {- 1} \left(T _ {\pi_ {1}} \circ \mathcal {C} _ {k} \left(\boldsymbol {s} _ {1}\right)\right). \\ \end{array}
$$

General argument for i: Suppose the induction condition holds true for all layers $\ell < i$ . Then,

$$
T _ {\pi_ {i}} \circ \dots \circ T _ {\pi_ {1}} (\boldsymbol {s} _ {1}) = T _ {\pi_ {i}} \left(\mathcal {C} _ {k} ^ {- 1} \left(T _ {\pi_ {i - 1}} \dots \circ T _ {\pi_ {1}} \circ \mathcal {C} _ {k} (\boldsymbol {s} _ {1})\right)\right).
$$

We can then follow the same argument as the base condition, and show that

$$
T _ {\pi_ {i}} \left(\mathcal {C} _ {k} ^ {- 1} \left(T _ {\pi_ {i - 1}} \dots \circ T _ {\pi_ {1}} \circ \mathcal {C} _ {k} (\boldsymbol {s} _ {1})\right)\right) = \mathcal {C} _ {k} ^ {- 1} \left(T _ {\pi_ {i}} \dots \circ T _ {\pi_ {1}} \circ \mathcal {C} _ {k} (\boldsymbol {s} _ {1})\right).
$$

Lemma F.19. For any $1 \leq i \leq d$ , if $s_{i,j}^{\alpha}, s_{i,j}^{\beta}$ denote the output after i - 1st translation step for a random sequence $s_{1} \sim \mathcal{U}(\{0,1\}^{L})$ at any position $1 \leq j \leq L$ under the two sets of random phrasebooks $\Pi_{:i}^{\alpha} := \{\pi_{\ell}^{\alpha}\}_{\ell=1}^{i-1}$ and $\Pi_{:i}^{\beta} := \{\pi_{\ell}^{\beta}\}_{\ell=1}^{i-1}$ , the following holds true for all positions j:

$$
\begin{array}{l} \operatorname * {P r} _ {\boldsymbol {\Pi} _ {: i} ^ {\alpha}, \boldsymbol {\Pi} _ {: i} ^ {\beta}} \left[ C o r r e l a t i o n \left(\boldsymbol {s} _ {i, j} ^ {\alpha}, \boldsymbol {s} _ {i, j + 1} ^ {\beta}, \mathcal {U} (\{0, 1 \} ^ {L})\right) = 1 \right] \\ \leq \frac {1}{2} \left(\operatorname * {P r} _ {\boldsymbol {\Pi} _ {: i} ^ {\alpha}, \boldsymbol {\Pi} _ {: i} ^ {\beta}} \left[ \text {Correlation} \left(\boldsymbol {s} _ {i, j} ^ {\alpha}, \boldsymbol {s} _ {i, j} ^ {\beta}, \mathcal {U} (\{0, 1 \} ^ {L})\right) = 1 \right] + \operatorname * {P r} _ {\boldsymbol {\Pi} _ {: i} ^ {\alpha}, \boldsymbol {\Pi} _ {: i} ^ {\beta}} \left[ \text {Correlation} \left(\boldsymbol {s} _ {i, j + 1} ^ {\alpha}, \boldsymbol {s} _ {i, j + 1} ^ {\beta}, \mathcal {U} (\{0, 1 \} ^ {L})\right) = 1 \right]\right). \\ \end{array}
$$

Proof. We will prove the required result with a counting argument. We will create a family of phrasebooks: let $FAMILY(j)$ and $FAMILY(j+1)$ denote two sets of phrasebooks, such that for every phrasebook $\Pi_{:i}^{\alpha}$ in $FAMILY(j)$ , there exists at least one phrasebook $\Pi_{:i}^{\beta}$ in $FAMILY(j+1)$ such that

$$
\text { Correlation } \left(\boldsymbol {s} _ {i, j} ^ {\alpha}, \boldsymbol {s} _ {i, j + 1} ^ {\beta}, \mathcal {U} (\{0, 1 \} ^ {L})\right) = 1,
$$

(Equiv. to saying translations for $\Pi_{:i}^{\alpha}$ and $\Pi_{:i}^{\beta}$ have correlations at position $j$ and $j + 1$ )

where $s_{i,j}^{\alpha}$ , $s_{i,j+1}^{\beta}$ are outputs on a random sequence $s_{1} \sim \mathcal{U}(\{0,1\}^{L})$ corresponding to using $\Pi_{:i}^{\alpha}$ and $\Pi_{:i}^{\beta}$ respectively.

We then apply a grouping algorithm GROUP to group correlated phrasebooks together in each family. That is, in FAMILY(j), we create groups of phrasebooks $\{S_{1}, S_{2}, \cdots\}$ such that for any two phrasebooks $\Pi_{:i}^{\alpha}$ and $\Pi_{:i}^{\alpha'}$ that belong to a group S,

$$
\text { Correlation } \left(\boldsymbol {s} _ {i, j} ^ {\alpha}, \boldsymbol {s} _ {i, j} ^ {\alpha^ {\prime}}, \mathcal {U} (\{0, 1 \} ^ {L})\right) = 1,
$$

(Equiv. to saying translations for $\Pi_{:i}^{\alpha}$ and $\Pi_{:i}^{\beta}$ have correlations at position j)

where $s_{i,j}^{\alpha}, s_{i,j+1}^{\alpha'}$ are outputs on a random sequence $s_{1} \sim \mathcal{U}(\{0,1\}^{L})$ corresponding to using $\Pi_{:i}^{\alpha'}$ and $\Pi_{:i}^{\alpha'}$ respectively. We call the resulting output of this operation as $\text{GROUP}(\text{FAMILY}(j))$ . Similarly, we compute $\text{GROUP}(\text{FAMILY}(j+1))$ .

We can observe the following two characteristics of $\text{GROUP}(\text{FAMILY}(j))$ and $\text{GROUP}(\text{FAMILY}(j+1))$ :

1. For every set $S_1 \in \mathrm{GROUP}(\mathrm{FAMILY}(j))$ there will exist one set $S_2 \in \mathrm{GROUP}(\mathrm{FAMILY}(j + 1))$ , such that for all phrasebooks $\Pi_{:i}^{\alpha} \in S_1$ and $\Pi_{:i}^{\beta} \in S_2$ , correlation will be 1 for output at positions $j$ and $j + 1$ .   
2. For every set $S_{1} \in \mathrm{GROUP}(\mathrm{FAMILY}(j))$ there can't exist two sets $S_{2}, S_{2}^{*} \in \mathrm{GROUP}(\mathrm{FAMILY}(j + 1))$ , such that the phrasebooks in $S_{1}$ are correlated to phrasebooks from both $S_{2}, S_{2}^{*}$ for output at positions $j$ and $j + 1$ respectively. Otherwise, we could have merged $S_{2}$ and $S_{2}^{*}$ under the GROUP operation.

Let CORRELATION-MAP denote the map between $\text{GROUP}(\text{FAMILY}(j))$ and $\text{GROUP}(\text{FAMILY}(j+1))$ , which connects sets $S_{1} \in \text{GROUP}(\text{FAMILY}(j))$ to a set $S_{2} \in \text{GROUP}(\text{FAMILY}(j+1))$ such that any two phrasebooks in $S_{1}$ and $S_{2}$ have correlations for output at positions j and $j+1$ respectively.

The result will then follow from a counting argument. The number of possible pairs (can be identical phrasebooks) that can give correlations for output at position j are given by: $\sum_{S_{1}\in\text{GROUP}(\text{FAMILY}(j))}|S_{1}|^{2}$ . Similarly, the number of possible pairs that can give correlations for output at position $j+1$ are given by: $\sum_{S_{2}\in\text{GROUP}(\text{FAMILY}(j+1))}|S_{2}|^{2}$ . On the other hand, the number of possible pairs that can give correlations for output at position j and $j+1$ respectively are given by: $\sum_{S_{1}\in\text{GROUP}(j);S_{2}=\text{CORRELATION-MAP}(S_{1})}|S_{1}||S_{2}|$ . Applying the AM-GM inequality, we can show that the average of the number of pairs for which correlation is 1 for output characters at either position j or $j+1$ is higher than the number of pairs for which correlation is 1 for output at positions j and $j+1$ .

# F.5. Proof for Statistical query lower bound for general n

We present the main theorem statement again for readability.

Theorem F.20 (SQ dimension for general n). For the translation task $MLT(d,n)$ that has depth d and n characters per level, the statistical query dimension $SQ\text{-dim}(MLT(d,n))$ is at least $n^{\Omega(d)}$ .

Proof. We adapt the SQ-dimension proof for $\mathbf{MLT}(d,2)$ to show the SQ-dimension proof for $\mathbf{MLT}(d,n)$ . We will design a family of set of phrasebooks $\Pi=\{\pi_{i}:\{0,1,\cdots,n-1\}^{2}\to\{0,1,\cdots,n-1\}^{2}\}_{i=1}^{d}$ , where phrasebooks in $\Pi$ are built on top of a translation task in $\mathbf{MLT}(\log_{2}n,2)$ .

For a character $a \in \{0, 1, \cdots, n - 1\}$ , suppose $\text{BIT}(a) \in \{0, 1\}^{\log_2 n}$ indicates its binary representation, and NUMERIC represents the map from binary representation to its corresponding numeric representation. Then, we design each phrasebook $\pi$ using a random translation task $\nu \in \mathbf{MLT}(\log_2 n, 2)$ . For any tuple $(a, b) \in \{0, 1, \cdots, n - 1\}^2$ , output of $\pi$ is given as

$$
\pi (a, b) = \left(o _ {1}, o _ {2}\right), \text {   where   }
$$

$$
o _ {1} = \text { NUMERIC } \left(\nu \left(\{\tilde {\boldsymbol {a}} _ {i} \oplus \tilde {\boldsymbol {b}} _ {i} \} _ {i = 1} ^ {\log_ {2} n}\right)\right)
$$

$$
o _ {2} = \text { NUMERIC } \left(\nu (\tilde {\boldsymbol {b}})\right)
$$

$$
\tilde {\boldsymbol {a}} = \operatorname{BIT} (a)
$$

$$
\tilde {\boldsymbol {b}} = \operatorname{BIT} (b)
$$

Primarily, the phrasebooks are defined as follows:

1. On a 2-tuple of characters $(a, b)$ , we first compute their binary representations (BIT(a), BIT(b)). We then compute two intermediate outputs, one where a xor operation is applied on BIT(a), BIT(b) at each bit, and another where BIT(b) is simply copied. This operation is equivalent to applying a deterministic map on 2-tuples of binary characters (identical to $\Delta_6$ from Table 7), where 2-tuples are created by pairing binary bits in binary representation of $a$ and $b$ .   
2. We then apply a random MLT task of depth $\log_{2} n$ on each of the intermediate outputs. This applies a random bijective map on the sequence of bits in the intermediate outputs, using a translation task in MLT( $\log_{2} n, 2$ ) (Lemma E.1).   
3. The final tuple of characters is returned by applying a NUMERIC operation on the binary representations.

Thus, we have narrowed our focus on a special group of tasks from $\mathbf{MLT}(d,n)$ that applies $\mathbf{MLT}(\log_{2}n,2)$ on the binary representations of the characters at each level. By Theorem F.10, at any level $1 \leq i \leq d$ , we can create $2^{\Omega(\log_{2}n)}$ phrasebooks, the output of which are pairwise uncorrelated. Now, we can compose these uncorrelated phrasebooks to give multiple set of phrasebooks that are uncorrelated. That is, following a similar proof as Theorem F.10, we can show that we can create $\left(2^{\Omega(\log_{2}n)}\right)^{\Omega(d)} = n^{\Omega(d)}$ set of phrasebooks, using the above restriction, that are pairwise uncorrelated on any bit in the binary representation of the output of their corresponding translation tasks. This will translate to the output of the translation task in numeric form as well, as the mapping between binary representation and numeric form of a digit is bijective.

# F.6. Proofs of Useful lemmas

Here we give the proofs for the useful lemmas necessary to prove Theorem F.10. We repeat the lemma statements for easier readability.

Lemma F.21 (Formulation of bijective maps for n = 2). Any bijective map $\pi : \{0,1\}^{2} \to \{0,1\}^{2}$ can be expressed using copy, not, and xor operations. Furthermore, from 24 possible maps for $\pi$ ,

1. There are 6 maps $\Delta_{1}, \Delta_{2}, \cdots, \Delta_{6}$ for which characters in the output tuple can be defined by copy and xor operations on the characters in the input tuple.

2. Mirror maps: For each map $\pi\in\{\Delta_{1},\Delta_{2},\cdots,\Delta_{6}\}$ , there exist mirror maps $\pi_{(1)},\pi_{(2)},\pi_{(3)}$ whose output on each input tuple can be defined by selective not operations on either or both characters of the output tuple of $\pi$ . We call $\{\pi,\pi_{(1)},\pi_{(2)},\pi_{(3)}\}$ as a mirror map set of $\pi$ , in short, $\text{Mirrorset}(\pi)$ .

Proof. There are 4 possible tuples $(0,0),(0,1),(1,0),(1,1)$ . By Lemma E.2, the number of possible bijective maps (phrasebooks) that connect 2-tuples are $4! = 24$ . We will show that the output tuple of each map can be represented by 3 operations.

Operation not creates mirror maps: For each map $\pi$ , there exists 3 alternative maps $\pi_{(1)}, \pi_{(2)}, \pi_{(3)}$ such that if $\pi(a, b)_{i}$ represents the ith character in the output tuple for input tuple $(a, b)$ :

<table><tr><td>Map name (π)</td><td colspan="4">Output for corresponding input tuple π(a,b)</td><td colspan="2">General formulation on output for map π</td></tr><tr><td></td><td>(0,0)</td><td>(0,1)</td><td>(1,0)</td><td>(1,1)</td><td> $\pi(a,b)_1$ </td><td> $\pi(a,b)_2$ </td></tr><tr><td> $\Delta_1$ </td><td>(0,0)</td><td>(0,1)</td><td>(1,0)</td><td>(1,1)</td><td> $\textbf{copy}_1(a,b)$ </td><td> $\textbf{copy}_2(a,b)$ </td></tr><tr><td> $\Delta_2$ </td><td>(0,0)</td><td>(0,1)</td><td>(0,1)</td><td>(1,0)</td><td> $\textbf{copy}_1(a,b)$ </td><td> $\textbf{xor}(a,b)$ </td></tr><tr><td> $\Delta_3$ </td><td>(0,0)</td><td>(1,0)</td><td>(0,1)</td><td>(1,1)</td><td> $\textbf{copy}_2(a,b)$ </td><td> $\textbf{copy}_1(a,b)$ </td></tr><tr><td> $\Delta_4$ </td><td>(0,0)</td><td>(1,1)</td><td>(0,1)</td><td>(1,0)</td><td> $\textbf{copy}_2(a,b)$ </td><td> $\textbf{xor}(a,b)$ </td></tr><tr><td> $\Delta_5$ </td><td>(0,0)</td><td>(1,0)</td><td>(1,1)</td><td>(0,1)</td><td> $\textbf{xor}(a,b)$ </td><td> $\textbf{copy}_1(a,b)$ </td></tr><tr><td> $\Delta_6$ </td><td>(0,0)</td><td>(1,1)</td><td>(1,0)</td><td>(0,1)</td><td> $\textbf{xor}(a,b)$ </td><td> $\textbf{copy}_2(a,b)$ </td></tr></table>

Table 7. The table captures the definition of 6 bijective maps (phrasebooks) on 2-tuples $\{0,1\}^{2}\rightarrow\{0,1\}^{2}$ , whose output characters can be defined in terms of copy and xor operations on the input characters. Any other bijective map can be shown to belong to Mirrorset of one of these maps.

\- $\pi_{(1)}$ selectively applies the not operation to the first character in the output tuple of $\pi$ on any input tuple, i.e.

$$
\pi_ {(1)} (a, b) = (\mathbf {n o t} (\pi (a, b) _ {1}), \pi (a, b))
$$

for all tuples $(a,b)\in \{0,1\}^2$

\- $\pi_{(2)}$ selectively applies the not operation to the second character in the output tuple of $\pi$ on any input tuple, i.e.

$$
\pi_ {(2)} (a, b) = (\pi (a, b) _ {1}, \mathbf {n o t} (\pi (a, b)))
$$

for all tuples $(a,b)\in \{0,1\}^2$

\- $\pi_{(3)}$ selectively applies the not operation to both characters in the output tuple of $\pi$ on any input tuple, i.e.

$$
\pi_ {(3)} (a, b) = (\mathbf {n o t} (\pi (a, b) _ {1}), \mathbf {n o t} (\pi (a, b)))
$$

for all tuples $(a,b)\in \{0,1\}^2$

Thus, for each map $\pi$ , there exist 3 other alternative maps that simply modify the output of map $\pi$ with the not operation.

After removing the mirror maps: We now show that there 6 possible maps that apply either a xor or a copy on the input characters to get the output characters. We name them $\Delta_{1}, \Delta_{2}, \cdots, \Delta_{6}$ . We give the output of each map on the 4 tuples in Table 7 and show that the each character in the output tuple can be represented using copy and xor operations.

Lemma F.22 (Correlation of bijective maps for $n = 2$ ). For two randomly selected maps $\pi^{\alpha}, \pi^{\beta} : \{0,1\}^{2} \to \{0,1\}^{2}$ , the following hold true.

1. Correlation at atleast one output character pair: Fix an $i, j \in \{0, 1\}$ . With probability $\frac{1}{3}$ w.r.t. the random selection of the maps, the ith character in the output of $\pi^{\alpha}$ has perfect correlation to jth character in the output of $\pi^{\beta}$ , i.e.

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {i}, \pi^ {\beta} (\cdot) _ {j}, \mathcal {U} (\{0, 1 \} ^ {2})) := \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \pi^ {\alpha} (x) _ {i} \neq \pi^ {\beta} (x) _ {j} ] - \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \pi^ {\alpha} (x) _ {i} = \pi^ {\beta} (x) _ {j} ] \right| = 1.
$$

2. Correlation at both output character pairs: With probability $\frac{1}{3}$ w.r.t. the random selection of the maps, both the characters in the outputs of $\pi^{\alpha}, \pi^{\beta}$ have perfect correlation, i.e. either one of the cases hold true

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {1}, \pi^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})) = 1.
$$

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {2}, \pi^ {\beta} (\cdot) _ {2}, \mathcal {U} (\{0, 1 \} ^ {2})) = 1.
$$

or

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {1}, \pi^ {\beta} (\cdot) _ {2}, \mathcal {U} (\{0, 1 \} ^ {2})) = 1.
$$

$$
C o r r e l a t i o n (\pi^ {\alpha} (\cdot) _ {2}, \pi^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})) = 1.
$$

Other cases are not possible, i.e. for any $i \in \{1,2\}$ , both Correlation $(\pi^{\alpha}(\cdot)_i, \pi^{\beta}(\cdot)_1, \mathcal{U}(\{0,1\}^2)) = 1$ and Correlation $(\pi^{\alpha}(\cdot)_i, \pi^{\beta}(\cdot)_2, \mathcal{U}(\{0,1\}^2)) = 1$ can't hold true.

Proof. We prove the lemma for case 1, cases 2 and 3 can be similarly proved. From Lemma E.2, $\pi^{\alpha}$ and $\pi^{\beta}$ can be randomly selected from a set of 24 possible candidates. On the other hand, Lemma F.21 shows that there are 6 maps $\{\Delta_{1},\cdots,\Delta_{6}\}$ whose output can be defined in terms of copy and xor operations of characters in the input tuple. For each $\pi$ in this set, there are mirror maps $\pi_{(1)},\pi_{(2)},\pi_{(3)}$ whose output are defined by selective not operations on the output of $\pi$ , and the set of 4 maps is represented by Mirrorset( $\pi$ ).

The proof will follow from 2 steps: first, we argue about correlations when $\pi^{\alpha}$ and $\pi^{\beta}$ are selected from $\{\Delta_{1},\cdots,\Delta_{6}\}$ , and then we argue about the general case when $\pi^{\alpha}$ and $\pi^{\beta}$ are selected from the general set of bijective maps.

- In Lemma F.24, we show that for two maps that are randomly selected from $\{\Delta_1, \cdots, \Delta_6\}$ , the correlation is 1 with probability $1/3$ .   
- The remaining possibility is when $\pi^{\alpha}$ and $\pi^{\beta}$ belong to $\text{Mirrorset}(\pi_i)$ and $\text{Mirrorset}(\pi_j)$ for some $\pi_i, \pi_j \in \{\Delta_1, \cdots, \Delta_6\}$ . We show in Lemma F.23, correlation of $\pi^{\alpha}$ and $\pi^{\beta}$ will be equal to correlation of $\pi_i, \pi_j$ .

Thus, we can combine all the observations to show that for two random maps $\pi^{\alpha}, \pi^{\beta}$ that belong to $\text{Mirrorset}(\pi_{i})$ and $\text{Mirrorset}(\pi_{j})$ for some $\pi_{i}, \pi_{j} \in \{\Delta_{1}, \cdots, \Delta_{6}\}$ ,

$$
\begin{array}{l} \text { Correlation } (\pi^ {\alpha} (\cdot) _ {1}, \pi^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})) \\ = \operatorname{Correlation} (\pi_ {i} (\cdot) _ {1}, \pi_ {j} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})) = \left\{ \begin{array}{l l} 1, & \text { w.p. } 1 / 3 \text { w.r.t.   randomness   in } \pi_ {i}, \pi_ {j} \\ 0, & \text { otherwise. } \end{array} \right. \\ \end{array}
$$

![](images/c51080ef42fa9f0a6690deb179cab05d7b04176f2892a1c7f6ab22233d6a4193.jpg)

Lemma F.23. The following holds true for any maps $\pi^{\alpha}$ and $\pi^{\beta}$ with $\pi^{\alpha},\pi^{\beta}\in\{\Delta_{1},\cdots,\Delta_{6}\}$ and for all $\tilde{\pi}^{\alpha}\in\text{Mirrorset}(\pi^{\alpha}),\tilde{\pi}^{\beta}\in\text{Mirrorset}(\pi^{\beta})$ :

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {1}, \pi^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})) = \text { Correlation } (\tilde {\pi} ^ {\alpha} (\cdot) _ {1}, \tilde {\pi} ^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})),
$$

Proof. We prove as follows:

$$
\begin{array}{l} \text { Correlation } (\pi^ {\alpha} (\cdot) _ {1}, \pi^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})) = \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \pi^ {\alpha} (x) _ {1} \neq \pi^ {\beta} (x) _ {1} ] - \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \pi^ {\alpha} (x) _ {1} = \pi^ {\beta} (x) _ {1} ] \right| \\ = \left| 2 \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \pi^ {\alpha} (x) _ {1} \neq \pi^ {\beta} (x) _ {1} ] - 1 \right| (15) \\ = \left\{ \begin{array}{l} \left| 2 \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \tilde {\pi} ^ {\alpha} (x) _ {1} \neq \tilde {\pi} ^ {\beta} (x) _ {1} \right] - 1 \right|, \text {if condition}" c 1" \text {is true} \\ \left| 1 - 2 \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} \left[ \tilde {\pi} ^ {\alpha} (x) _ {1} = \tilde {\pi} ^ {\beta} (x) _ {1} \right] \right|, \text {if condition}" c 2" \text {is true} \end{array} \right. (16) \\ = \left| \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \tilde {\pi} ^ {\alpha} (x) _ {1} \neq \tilde {\pi} ^ {\beta} (x) _ {1} ] - \operatorname * {P r} _ {x \sim \mathcal {U} (\{0, 1 \} ^ {2})} [ \tilde {\pi} ^ {\alpha} (x) _ {1} = \tilde {\pi} ^ {\beta} (x) _ {1} ] \right| \\ = \text { Correlation } (\tilde {\pi} ^ {\alpha} (\cdot) _ {1}, \tilde {\pi} ^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})), (17) \\ \end{array}
$$

where the second and the penultimate steps follow from the law of total probability. Here, condition “c1” holds when either case is true,

$$
\tilde {\pi} ^ {\alpha} (x) _ {1} = \mathbf {n o t} (\pi^ {\alpha} (x) _ {1}), \quad \tilde {\pi} ^ {\beta} (x) _ {1} = \mathbf {n o t} (\pi^ {\beta} (x) _ {1}), \quad \text { for   all } x \in \{0, 1 \} ^ {2}
$$

$$
\tilde {\pi} ^ {\alpha} (x) _ {1} = \pi^ {\alpha} (x) _ {1}, \quad \tilde {\pi} ^ {\beta} (x) _ {1} = \pi^ {\beta} (x) _ {1}, \quad \mathrm{forall} x \in \{0, 1 \} ^ {2}.
$$

and condition “c2” holds when either case is true,

$$
\tilde {\pi} ^ {\alpha} (x) _ {1} = \pi^ {\alpha} (x) _ {1}, \quad \tilde {\pi} ^ {\beta} (x) _ {1} = \mathbf {n o t} (\pi^ {\beta} (x) _ {1}), \quad \text { for   all } x \in \{0, 1 \} ^ {2}
$$

$$
\tilde {\pi} ^ {\alpha} (x) _ {1} = \mathbf {n o t} (\pi^ {\alpha} (x) _ {1}), \quad \tilde {\pi} ^ {\beta} (x) _ {1} = \pi^ {\beta} (x) _ {1}, \quad \text { for   all } x \in \{0, 1 \} ^ {2}.
$$

One of condition “c1” or condition “c2” is true because $\tilde{\pi}^{\alpha}$ and $\pi^{\alpha}$ (similarly, $\tilde{\pi}^{\beta}$ and $\pi^{\beta}$ ) are mirror maps.

![](images/00a375b23e831e152bc59cc20e708b9d7f270be7ade1164fc5bbfb07482c02a7.jpg)

Lemma F.24. The following holds true for two randomly selected maps $\pi^{\alpha}$ and $\pi^{\beta}$ with $\pi^{\alpha},\pi^{\beta}\in\{\Delta_{1},\cdots,\Delta_{6}\}$ :

\- With probability $1/3$ w.r.t. random selection,

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {1}, \pi^ {\beta} (\cdot) _ {1}, \mathcal {U} (\{0, 1 \} ^ {2})) > 0 \quad (= 1).
$$

\- With probability $1/3$ w.r.t. random selection,

$$
\text { Correlation } (\pi^ {\alpha} (\cdot) _ {2}, \pi^ {\beta} (\cdot) _ {2}, \mathcal {U} (\{0, 1 \} ^ {2})) > 0 \quad (= 1).
$$

Proof. We prove for case 1, proof for case 2 is analogous.

Among $\{\Delta_{1},\cdots,\Delta_{6}\}$ , we can create three family of maps: $F_{copy_{1}}:\{\Delta_{1},\Delta_{2}\}$ , $F_{copy_{2}}:\{\Delta_{3},\Delta_{4}\}$ , $F_{xor}:\{\Delta_{5},\Delta_{6}\}$ that are identical in operation ( $copy_{1},copy_{2},xor$ respectively) at the first character in output tuple. This would imply, if the $\pi^{\alpha}$ and $\pi^{\beta}$ both belong to one of these families, Correlation( $\pi^{\alpha}(\cdot)_{1},\pi^{\beta}(\cdot)_{1},\mathcal{U}(\{0,1\}^{2})$ ) will be 1.

On the other hand, one can show that for any operation $f_{1}, f_{2} \in \{copy_{1}, copy_{2}, xor\}$ with $f_{1} \neq f_{2}$ will have $\text{Correlation}(f_{1}, f_{2}, \mathcal{U}(\{0, 1\}^{2})) = 0$ . That would then suggest that if $\pi^{\alpha}$ and $\pi^{\beta}$ belong to different families among $F_{copy_{1}}, F_{copy_{2}}, F_{xor}$ , then $\text{Correlation}(\pi^{\alpha}(\cdot)_{1}, \pi^{\beta}(\cdot)_{1}, \mathcal{U}(\{0, 1\}^{2}))$ will be 0.

By a simple counting argument, with probability $\frac{1}{3}$ , two maps $\pi^{\alpha}$ and $\pi^{\beta}$ randomly selected from $\{\Delta_{1},\cdots,\Delta_{6}\}$ will have non-zero correlation.

# G. Upper Bound: Context-Enhanced Learning of MLT(d, n) with Simple Surrogate Model

# G.1. Setup of Surrogate Model with In-context Capability

Given $d + 1$ alphabets $A_{1}, \ldots, A_{d+1}$ of size n and d bijective phrasebooks $\pi_{i}: A_{i}^{2} \to A_{i+1}^{2}$ . The input of the translation process is an even-length sequence in the first alphabet, which we denote as $s_{1} \in A_{1}^{L}$ where L is the sequence length. The translation process modifies the input string recursively from $s_{i}$ to $s_{i+1}$ through the following 2 sub-processes:

1. Circular shift: The characters in $\pmb{s}_i \in A_i^L$ are shifted by 1 character leftward (and wrapped around to the end if necessary) to give sequence $\tilde{\pmb{s}}_i \in A_i^L$ . Formally, for each $j \in [1,L]$ we have $\tilde{\pmb{s}}_{i,j} = \pmb{s}_{i,(j + 1)\% L}$ .   
2. Translate: Using the phrasebook $\pi_i: A_i^2 \to A_{i+1}^2$ , we translate 2-tuples (bigrams) of consecutive characters in sequence $\tilde{\boldsymbol{s}}_i$ to create $\boldsymbol{s}_{i+1}$ . That is, for every odd $j \in [1, L]$ , $(\boldsymbol{s}_{i+1,j}, \boldsymbol{s}_{i+1,j+1}) = \pi_i(\tilde{\boldsymbol{s}}_{i,j}, \tilde{\boldsymbol{s}}_{i,j+1})$ .

Now let us revisit the surrogate model introduced in Section 5.1. Without loss of generality let $A_1 = A_2 = \cdots = A_{d+1} := A = \{1, 2, \ldots, n\}$ . For any single character $a \in A$ , let its vector representation be a one-hot vector $e_a \in \mathbb{R}^n$ such that $(e_a)_a = 1$ . For any 2-tuple $(a, b) \in A^2$ , let its vector representation be a $n^2$ -dimensional vector $v(a, b) \triangleq e_a \otimes e_b$ . Note that $v(a, b)$ is also a one-hot vector where

$$
\boldsymbol {v} (a, b) _ {i} = \left\{ \begin{array}{l l} 1 & \text { if } i = a n + b \\ 0 & \text { elsewhere } \end{array} \right..
$$

We use the notation $\bar{e}_{a}$ (long one-hot) to denote a one-hot vector in $R^{n^{2}}$ with a-th position being 1 to avoid confusion.

Definition G.1 (Matrix Representation of Sequence).

For a length-L input sequence $\boldsymbol{s}_{i}=(s_{i,1},\ldots,s_{i,L})$ , let its matrix representation be $\mathbf{Mat}(\boldsymbol{s}_{i})\triangleq\boldsymbol{V}_{i}\in\mathbb{R}^{n^{2}\times L/2}$ that

$$
\boldsymbol {V} _ {i} = \left[ \begin{array}{c c c c} | & | & \dots & | \\ \boldsymbol {v} (s _ {i, 1}, s _ {i, 2}) & \boldsymbol {v} (s _ {i, 3}, s _ {i, 4}) & \dots & \boldsymbol {v} (s _ {i, L - 1}, s _ {i, L}) \\ | & | & \dots & | \end{array} \right]
$$

For each $j \in [L/2]$ , we use $\boldsymbol{V}_{i}^{(j)}$ to denote the j-th column of $V_{i}$ . We also denote the above conversion from a sequence $s_{i}$ to its matrix form as $\boldsymbol{V}_{i} = \mathbf{Mat}(\boldsymbol{s}_{i})$ and assume that $V_{1}$ serves as the input to the surrogate model.

Note that the matricization operation is invertible by construction: for each column $\boldsymbol{V}_{i}^{(j)}$ , let $x = \arg\max \boldsymbol{V}_{i}^{(j)}$ , we may read off the two characters in the original alphabet by computing $\mathbf{Mat}^{-1}(\boldsymbol{V}_{i}^{(j)}) = (\lceil x/n \rceil, x\%n)$ .

At each level of translation, we assume the surrogate model will perform the following operations to $V_{i}$ :

(a) Circular shift from $V_{i}$ to $\tilde{V}_{i}$

Definition G.2 (Circular Shifting Operator Shift).

Given a matrix representation $V \in R^{n^{2} \times L/2}$ of a sequence s, the circular shifting operator Shift acts on V as $\text{Shift}(V) := \tilde{V} \in \mathbb{R}^{n^{2} \times L/2}$ where for all $j \in [L/2]$ ,

$$
\tilde{\boldsymbol{V}}^{(j)} = \boldsymbol {Q}\boldsymbol{V}^{(j)}\odot \boldsymbol {Q}^{\top}\boldsymbol{V}^{((j + 1)\% L)}
$$

where $\boldsymbol{Q} = (I_{n} \otimes \mathbf{1}_{n}) (\mathbf{1}_{n} \otimes I_{n})^{\top}$ , $1_{n} \in R^{n \times 1}$ is the all-ones vector, and $\odot$ is the Hadamard product.

Lemma G.3 (Equivalence of Shift and circular shift).

For any sequence $s \in A^{L}$ , let $\tilde{s}$ be the circular shifted s, then

$$
\mathbf {M a t} (\tilde {\boldsymbol {s}}) = \text { Shift } (\mathbf {M a t} (\boldsymbol {s})).
$$

Proof of Lemma G.3. In this proof we will show that $\mathbf{Mat}(\tilde{\mathbf{s}})$ and $\mathbf{Shift}(\mathbf{Mat}(\mathbf{s}))$ agrees on every column.

Fix a column $j \in [n/2]$ , without loss of generality let the input sequence s be $(a, b, c, d) \in A$ starting from the $(2j - 1)$ -th position to the $(2j + 2)$ -th position (wrapped around when necessary). By construction each output column of $\tilde{\boldsymbol{V}}^{(j)}$ is dependent on at most $\boldsymbol{V}^{(j)}$ and $\boldsymbol{V}^{(j+1)}$ which corresponds to 4 characters in the sequence s.

By definition of $\mathbf{V}$ we then have $\mathbf{V}^{(j)} = \mathbf{v}(a,b) = \mathbf{e}_a \otimes \mathbf{e}_b$ and $\mathbf{V}^{(j + 1)\% L} = \mathbf{v}(c,d) = \mathbf{e}_c \otimes \mathbf{e}_d$ . It follows that

$$
\begin{array}{l} \tilde {\boldsymbol {V}} ^ {(j)} = \boldsymbol {Q} \boldsymbol {V} ^ {(j)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {V} ^ {((j + 1) \% L)} \\ = \left(\left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {a} \otimes \boldsymbol {e} _ {b}\right)\right) \odot \left(\left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {c} \otimes \boldsymbol {e} _ {d}\right)\right) \\ = \left(\left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \boldsymbol {e} _ {a} \otimes I _ {n} ^ {\top} \boldsymbol {e} _ {b}\right)\right) \odot \left(\left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \boldsymbol {e} _ {c} \otimes \mathbf {1} _ {n} ^ {\top} \boldsymbol {e} _ {d}\right)\right) \\ = \left(\left(I _ {n} \otimes \mathbf {1} _ {n}\right) (1 \otimes \boldsymbol {e} _ {b})\right) \odot \left(\left(\mathbf {1} _ {n} \otimes I _ {n}\right) (\boldsymbol {e} _ {c} \otimes 1)\right) \quad \left(\boldsymbol {e} _ {a}, \boldsymbol {e} _ {d} \text {   are   one - hot,   } \mathbf {1} _ {n} ^ {\top} \boldsymbol {e} _ {a} = \mathbf {1} _ {n} ^ {\top} \boldsymbol {e} _ {d} = 1\right) \\ = \left(\left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\boldsymbol {e} _ {b} \otimes 1\right)\right) \odot \left(\left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(1 \otimes \boldsymbol {e} _ {c}\right)\right) \\ = \left(\boldsymbol {e} _ {b} \otimes \mathbf {1} _ {n}\right) \odot \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {c}\right) \\ = \boldsymbol {e} _ {b} \otimes \boldsymbol {e} _ {c}. \quad (\text { by   definition   of   Kronecker   product }) \\ \end{array}
$$

Note that $e_{b} \otimes e_{c}$ is just $\mathbf{Mat}(\tilde{\boldsymbol{s}})^{(j)}$ since the $(2j - 1)$ -th and the 2j-th character of the shifted sequence $\tilde{s}$ is now $(b, c)$ .

This concludes the proof.

# (b) Translation from $\tilde{V}_{i}$ to $V_{i+1}$

With Shift effectively completing Merge, Circular shift, and Split, what remains is the translation leveraging $\pi_{i}$ . Since $A_{i}=A_{i+1}=A=[n]$ , there is a natural bijection between the space of binary column-stochastic matrix and the space of all possible (not necessarily bijective) mappings between 2-tuples from A. Concretely

Definition G.4 (Column-Stochastic Matrix Representation of phrasebook).

Given phrasebook $\pi: A^{2} \to A^{2}$ , its matrix representation is defined to be $\text{Matrix}(\pi) \in \mathbb{R}^{n^{2} \times n^{2}}$ that for $i, j \in [n^{2}]$ ,

$$
\text {Matrix} (\pi) _ {j, i} = \left\{ \begin{array}{ll}1 & \text {if} \pi (\lceil i / n \rceil ,i\% n) = (\lceil j / n \rceil ,j\% n)\\ 0 & \text {elsewhere} \end{array} \right..
$$

Let $V_{i+1} = Matrix\tilde{V}_{i}$ where Matrix is a binary column-stochastic matrix, we can show that, a complete translation process for one level can be formally expressed as follows:

Lemma G.5 (Equivalence of Matrix and Translate).

For any sequence $s_{i} \in A^{L}$ , let $\pi_{i}$ be a bijective phrasebook $A^{2} \to A^{2}$ defined in Section 2.2, then

$$
\boldsymbol {V} _ {i + 1} = \operatorname{Matrix} \left(\pi_ {i}\right) \text {Shift} \left(\mathbf {M a t} \left(\boldsymbol {s} _ {i}\right)\right) = \mathbf {M a t} \left(T _ {\pi_ {i}} \left(\boldsymbol {s} _ {i}\right)\right) = \mathbf {M a t} \left(\boldsymbol {s} _ {i + 1}\right).
$$

Proof of Lemma G.5. Fix any column $j \in [L/2]$ , let $(a, b)$ be the $(2j - 1)$ -th and the 2j-th character of the shifted sequence $\tilde{s}_{i}$ . Let $(c, d) = \pi_{i}(a, b)$ , by the translation construction we know the $(2j - 1)$ -th and the 2j-th character of $s_{i+1}$ is just c and d. Hence $\mathbf{Mat}(\mathbf{s}_{i+1})^{(j)} = v(c, d)$ .

From Lemma G.3 we know that $\text{Shift}(\mathbf{Mat}(\boldsymbol{s}_{i}))^{(j)} = v(a,b)$ . By construction of $\text{Matrix}(\pi_{i})$ above, we know that $\text{Matrix}(\pi_{i})\text{Shift}(\mathbf{Mat}(\boldsymbol{s}_{i}))^{(j)}$ is a one-hot vector at the $(cn + d)$ -th position, i.e. $\boldsymbol{V}_{i+1}^{(j)} = \text{Matrix}(\pi_{i})\text{Shift}(\mathbf{Mat}(\boldsymbol{s}_{i}))^{(j)} = \boldsymbol{e}_{c} \otimes \boldsymbol{e}_{c} = \boldsymbol{v}(c,d)$ . Since $\mathbf{Mat}(\boldsymbol{s}_{i+1})^{(j)} = \boldsymbol{V}_{i+1}^{(j)}$ for all j, we have $V_{i+1} = \mathbf{Mat}(\boldsymbol{s}_{i+1})$ .

# Parameterization of the Translation Matrix P.

With the two key operations in place, now we can introduce the surrogate model in its full detail. Since the multi-level translation task is a naturally sequential operation, we model the surrogate operations as a multi-layer network as well (which also matches with the solution found by Llama-based models when trained on real data as in Section 3.3).

For each level, the surrogate model needs to present both in-context learning capability at the initialization and in-weight capability toward the end of context-enhanced learning on a certain set of phrasebooks $\Pi^{*}$ . The in-weight capability requires certain parameter to store $\Pi^{*}$ on its own that is independent of the context.

To capture both capabilities at the same time, we parameterize the translation matrix $P_{i}$ as a combination of in-context information $C_{i} \in R^{n^{2} \times n^{2}}$ and in-weight memory $W_{i} \in R^{n^{2} \times n^{2}}$ .

# Definition G.6 (Effective Translation Matrix).

For in-context information $C_{i} \in R^{n^{2} \times n^{2}}$ and in-weight memory $W_{i} \in R^{n^{2} \times n^{2}}$ at level i, the corresponding effective translation matrix $P_{i}$ is defined as

$$
\boldsymbol {P} _ {i} = \operatorname{HardMax} \left(\boldsymbol {C} _ {i} + \boldsymbol {W} _ {i}\right)
$$

where HardMax is the column-wise hard-max function converting $C_{i} + W_{i}$ to a binary column stochastic matrix.

Note that column k in $P_{i}$ , which we denote as $P_{i}^{(k)}$ , is equal to the one-hot vector at $\arg\max(C_{i}^{(k)} + W_{i}^{(k)})$ .

# Surrogate Model with In-Context Capability

Now we can formally introduce the surrogate model.

# Definition G.7 (Surrogate Model for MLT).

The surrogate model for $\mathbf{MLT}(d,n)$ can be represented by the recursive expression

$$
\boldsymbol {V} _ {i + 1} = \operatorname{HardMax} \left(\boldsymbol {C} _ {i} + \boldsymbol {W} _ {i}\right) \text { Shift } (\boldsymbol {V} _ {i}) \tag {18}
$$

The learnable parameters for the surrogate model are the weight matrices $\{W_{i}\}_{i=1}^{d} := \{W_{1}, W_{2}, \ldots, W_{d}\}$ . We denote the surrogate model parameterized by $\{W_{i}\}_{i=1}^{d}$ as $\text{SURR-MLT}_{\{W_{i}\}_{i=1}^{d}}(\cdot)$ which maps the input and the in-context information to the output as

$$
\boldsymbol {V} _ {d + 1} = \operatorname{SURR-MLT} _ {\{\boldsymbol {W} _ {i} \} _ {i = 1} ^ {d}} (\boldsymbol {C} _ {1}, \boldsymbol {C} _ {2}, \dots , \boldsymbol {C} _ {d}, \boldsymbol {V} _ {1})
$$

$$
\triangleq \operatorname{HardMax} \left(\boldsymbol {C} _ {d} + \boldsymbol {W} _ {d}\right) \tag {19}
$$

$$
\operatorname{Shift} \left(\operatorname{HardMax} \left(\boldsymbol {C} _ {d - 1} + \boldsymbol {W} _ {d - 1}\right) \operatorname{Shift} \left(\dots \operatorname{HardMax} \left(\boldsymbol {C} _ {1} + \boldsymbol {W} _ {1}\right) \operatorname{Shift} \left(\boldsymbol {V} _ {1}\right) \dots\right)\right).
$$

In this surrogate model, the in-context descriptive text DESC is just $\{C_{1},\ldots,C_{d}\}$ , where each matrix is directly being passed into the corresponding layer. When providing information about a set of phrasebooks $\Pi=\{\pi_{i}\}_{i=1}^{d}$ , we have $\boldsymbol{C}_{i}=\operatorname{Matrix}(\pi_{i})$ where $\operatorname{Matrix}(\pi_{i})$ is the column-stochastic matrix representation of $\pi_{i}$ as defined in Definition G.4.

When partial phrasebook information is provided (corresponding to dropping certain ab->CD entries in the language model context), we zero-out the corresponding column in $C_{i}$ . When no in-context information is provided for level i, we just have $C_{i}$ be the all-zero matrix containing no information (assuming zero as prior).

For simplicity of presentation, in Definition G.7 the in-context information $C_i$ 's are provided directly to the corresponding layers. We note that the equivalent operations can be exactly re-parameterized such that $C_i$ 's are provided in-context (in concatenation with $V_1$ ). The reparameterization of the surrogate model is provided below:

Definition G.8 (Context-Augmented Surrogate Model for MLT).

Let the context-augmented input be

$$
\mathbf {X} _ {1} = \left[ \boldsymbol {C} _ {1}, \dots , \boldsymbol {C} _ {d}, \boldsymbol {V} _ {1} \right] \in \mathbb {R} ^ {n ^ {2} \times (d n ^ {2} + L)},
$$

then we can rewrite the same surrogate model as

$$
\begin{array}{l} \mathbf {X} _ {i + 1} = \mathbf {X} _ {i} \left[ \begin{array}{c c} I _ {d n ^ {2}} & 0 \\ 0 & 0 _ {L \times L} \end{array} \right] + \left(\left(\mathbf {X} _ {i} \left[ \begin{array}{l} e _ {i} \otimes I _ {n ^ {2}} \\ 0 _ {L \times n ^ {2}} \end{array} \right] + \boldsymbol {W} _ {i}\right) \text {Shift} \left(\mathbf {X} _ {i} \left[ \begin{array}{c} 0 _ {n ^ {2} d \times L} \\ I _ {L} \end{array} \right]\right) \left[ \begin{array}{c c} 0 _ {d n ^ {2} \times d n ^ {2}} & 0 \\ 0 & I _ {L} \end{array} \right]\right) \tag {20} \\ = [ \boldsymbol {C} _ {1}, \dots , \boldsymbol {C} _ {d}, \boldsymbol {V} _ {i + 1} ] \\ \end{array}
$$

With the reparameterization, the model is capable of generating $[C_{1},\ldots,C_{d},V_{d+1}]$ with input $[C_{1},\ldots,C_{d},V_{1}]$ , with all information provided in-context in the input.

Now let us check what does Definition 2.2 (ICL-capable) and Definition 2.1 (specific task-capable) mean in the context of the surrogate model. To make things more rigorous we introduce two stronger notions of capabilities:

Definition G.9 (Strongly MLT(d, n)-ICL-capable surrogate model).

We say a surrogate model SURR-MLT $_{ \{ W_{i} \}_{i=1}^{d} }(\cdot)$ is strongly MLT(d, n)-ICL-capable if for any set of phrasebooks $\Pi = \{\pi_{i}\}_{i=1}^{d}$ in MLT(n, d), for any input sequence $s_{1} \in A^{L}$ where L is even, we have

$$
\text { Surr - MLT } _ {\{\boldsymbol {W} _ {i} \} _ {i = 1} ^ {d}} (\text { Matrix } (\pi_ {1}), \ldots , \text { Matrix } (\pi_ {d}), \mathbf {M a t} (\boldsymbol {s} _ {1})) = \mathbf {M a t} (\mathbf {M L T} _ {\Pi} (\boldsymbol {s} _ {1})).
$$

Definition G.10 (Strongly MLT $_{\Pi^{*}}$ -capable surrogate model).

For a fixed set of phrasebooks $\Pi^{*} = \{\pi_{i}^{*}\}_{i=1}^{d}$ in MLT(n,d), we say a surrogate model SURR-MLT $\{W_{i}\}_{i=1}^{d}(\cdot)$ is strongly MLT $_{\Pi^{*}}$ -capable if for any input sequence $s_{1} \in A^{L}$ where $L$ is even, we have

$$
\operatorname{SURR-MLT} _ {\left\{\boldsymbol {W} _ {i} \right\} _ {i = 1} ^ {d}} (\mathbf {0}, \dots , \mathbf {0}, \mathbf {M a t} (\boldsymbol {s} _ {1})) = \mathbf {M a t} (\mathbf {M L T} _ {\Pi^ {*}} (\boldsymbol {s} _ {1})).
$$

Now we can show the following properties of the surrogate model SURR-MLT $\{\pmb{W}_i\}_{i=1}^d (\cdot)$ :

Lemma G.11. When $\| \pmb{W}_i\| _0 < \frac{1}{2}$ for all $i\in [d]$ , SURR-MLT $\{\pmb{W}_i\}_{i = 1}^d$ is strongly $\mathbf{MLT}(d,n)$ -ICL-capable.

Proof. Fix any set of phrasebooks $\Pi = \{\pi_i\}_{i=1}^d$ and its corresponding matrix representations $\{\text{Matrix}(\pi_i)\}_{i=1}^d$ . Since $\|\boldsymbol{W}_i\|_0 < \frac{1}{2}$ , for any column $k$ , no entries in $\boldsymbol{W}_i^{(k)}$ can flip the argmax of $\text{Matrix}(\pi_i)^{(k)} + \boldsymbol{W}_i^{(k)}$ away from being $\arg\max\text{Matrix}(\pi_i)^{(k)}$ . Therefore we have $\boldsymbol{P}_i = \text{HardMax}(\text{Matrix}(\pi_i) + \boldsymbol{W}_i) = \text{Matrix}(\pi_i)$ for all layers $i \in [d]$ .

By Lemma G.5, for all $i \in [d]$ we have $P_{i} = \text{Matrix}(\pi_{i})$ recovering $T_{\pi_{i}}$ . Hence for any input sequence $s_{1} \in A^{L}$ , we have $\text{SURR-MLT}_{\{\boldsymbol{W}_{i}\}_{i=1}^{d}}(\{(\pi_{1})\}, \ldots, \{(\pi_{d})\}, \mathbf{Mat}(\boldsymbol{s}_{1})) = \mathbf{Mat}(\mathbf{MLT}_{\Pi}(\boldsymbol{s}_{1}))$ . ☐

Lemma G.12. Fix a target set of phrasebooks $\Pi^{*} = \{\pi_{i}^{*}\}_{i=1}^{d}$ , when $\operatorname{HardMax}\left(\boldsymbol{W}_{i}\right) = \operatorname{Matrix}\left(\pi_{i}^{*}\right)$ for all $i \in [d]$ , $\operatorname{SURR-MLT}_{\{\boldsymbol{W}_{i}\}_{i=1}^{d}}(\cdot)$ is strongly $\mathbf{MLT}_{\Pi^{*}}$ -capable.

Proof. By Lemma G.5, for all $i \in [d]$ we have $\boldsymbol{P}_{i} = \text{HardMax}(\boldsymbol{W}_{i} + \boldsymbol{0}) = \text{Matrix}(\pi_{i}^{*})$ recovering $T_{\pi_{i}^{*}}$ . Hence for any input sequence $s_{1} \in A^{L}$ , we have $\text{SURR-MLT}_{\{\boldsymbol{W}_{i}\}_{i=1}^{d}}(\boldsymbol{0}, \ldots, \boldsymbol{0}, \mathbf{Mat}(s_{1})) = \mathbf{Mat}(\mathbf{MLT}_{\Pi^{*}}(s_{1}))$ . ☐

Lemma G.11 suggests that when the weight matrices have small initializations, the model has perfect ICL capability. Meanwhile Lemma G.12 suggests that when the weights $W_{i}$ recover $\text{Matrix}(\pi_{i}^{*})$ in the column-wise hard-max sense, then the surrogate model can perform $MLT_{\Pi^{*}}$ when no context is being provided ( $C_{i}=0$ ).

# G.2. Learning $\Pi^{*}$ in MLT $(d, n)$ with Heuristics Search

In this section, we provide a brute-force algorithm that can learn any target set of phrasebooks $\Pi^{*}$ in $\mathbf{MLT}(d,n)$ using a single “short” sequence whose length is not exponentially dependent on d.

Before proceeding to the details, let us first investigate more on the nature of MLT and the surrogate model. For simplicity of notations, given $\Pi^{*} = \{\pi_{1}^{*}, \ldots, \pi_{d}^{*}\}$ , we denote the general translation operator Matrix as P, and denote the translation operator Matrix $^{(\pi_{1}^{*})}$ as $W_{i}^{*}$ . Also, with slight abuse of notations we use $\mathbf{MLT}_{\Pi^{*}}(V_{1})$ to denote $\mathbf{Mat}\left(\mathbf{MLT}_{\Pi^{*}}(\mathbf{Mat}^{-1}(V_{1}))\right)$ .

First, we characterize the input sequence that is good for providing learning signals.

Definition G.13 ( $\Pi^{*}$ -coverable input).

Fix a target set of phrasebooks $\Pi^{*} = \{\pi_{1}^{*}, \ldots, \pi_{d}^{*}\}$ in $\mathbf{MLT}(d, n)$ and an input matrix $V_{1} \in R^{n^{2} \times L}$ , let $\tilde{V}_{1}^{*}, V_{2}^{*}, \tilde{V}_{2}^{*}, \ldots, \tilde{V}_{d}^{*}, V_{d+1}^{*} \in \mathbb{R}^{n^{2} \times L}$ be the intermediate outputs when applying $MLT_{\Pi^{*}}$ on $V_{1}$ . We say $V_{1}$ is $\Pi^{*}$ -coverable if for all levels $i \in [d]$ , $\tilde{V}_{i}^{*}$ is of rank- $n^{2}$ .

Note that as a matrix with only one-hot columns, $\tilde{V}_i^*$ being rank- $n^2$ suggests that for all $k\in [n^2]$ , there exists some column $j\in [L]$ such that $\tilde{V}_i^{*(j)} = e_k$ . In the context of the translation process, it means that the correct translation process of a $\Pi^{*}$ -coverable input $V_{1}$ would require all entries of all phrasebooks in $\Pi^{*}$ .

Next we will show that if we use the context to condition all translation operators $P_{l}$ of the surrogate model to be $\boldsymbol{P}^{(\pi_{l}^{*})}$ for all but one level $l \in [d] \setminus \{i\}$ (i as the unconditioned level), then for the surrogate model to correctly perform $MLT_{\Pi^{*}}$ , the operator for the unconditioned level i must also be equal to $\boldsymbol{P}^{(\pi_{i}^{*})}$ .

Lemma G.14 (Uniqueness of a single $P_{i}$ when conditioning all other levels).

Fix a target set of phrasebooks $\Pi^{*} = \{\pi_{1}^{*},\dots,\pi_{d}^{*}\}$ in MLT $(d,n)$ and a $\Pi^{*}$ -coverable input $V_{1}$ . For any level $i\in [d]$ , consider a surrogate model SURR-MLT $\{_{\boldsymbol{W}_i\}_{i = 1}^d (\boldsymbol{C}_1,\boldsymbol{C}_2,\dots,\boldsymbol{C}_d,\cdot)$ with certain context $C_1,C_2,\ldots ,C_d$ such that $P_{l} =$ $\boldsymbol{W}_l^*$ for all $l\in [d]$ except $l = i$ . Then SURR-MLT $\{_{\boldsymbol{W}_i\}_{i = 1}^d (\boldsymbol{C}_1,\boldsymbol{C}_2,\dots,\boldsymbol{C}_d,V_1) = \mathbf{MLT}_{\Pi^*}(V_1)$ if and only if

$$
\boldsymbol {P} _ {i} = \operatorname{HardMax} \left(\boldsymbol {C} _ {i} + \boldsymbol {W} _ {i}\right) = \boldsymbol {W} _ {i} ^ {*}.
$$

Proof. This lemma is a direct consequence of the bijective property of the translation process shown in Lemma E.1. Let $\tilde{V}_1^*, V_2^*, \tilde{V}_2^*, \ldots, \tilde{V}_d^*, V_{d+1}^* \in \mathbb{R}^{n^2 \times L}$ be the intermediate outputs when applying $\mathbf{MLT}_{\Pi^*}$ on $V_1$ , and let $\tilde{V}_1, V_2, \tilde{V}_2, \ldots, \tilde{V}_d, V_{d+1} \in \mathbb{R}^{n^2 \times L}$ be the intermediate outputs when applying SURR-MLT $_{\{W_i\}_{i=1}^d}$ ( $C_1, C_2, \ldots, C_d, \cdot$ ) as described. Since we assume $P_l = P^{(\pi_l^*)}$ for all $l < i$ , we have $V_i = V_i^*$ and therefore $\tilde{V}_i = \tilde{V}_i^*$ .

On the other end, since $\operatorname{SURR-MLT}_{\{\boldsymbol{W}_{i}\}_{i=1}^{d}}(\boldsymbol{C}_{1},\boldsymbol{C}_{2},\ldots,\boldsymbol{C}_{d},\boldsymbol{V}_{1})=\mathbf{MLT}_{\boldsymbol{\Pi}^{*}}(\boldsymbol{V}_{1})=\boldsymbol{V}_{d+1}^{*}$ and $\boldsymbol{P}_{l}=\boldsymbol{P}^{(\pi_{l}^{*})}$ for all l>i, by the invertible property of the translation process (Lemma E.1) we must have $\tilde{V}_{i+1}=\tilde{V}_{i+1}^{*}$ and thus $V_{i+1}=V_{i+1}^{*}$ . Combining both ends we know that

$$
\boldsymbol {P} _ {i} \tilde {\boldsymbol {V}} _ {i} = \boldsymbol {V} _ {i + 1} = \boldsymbol {V} _ {i + 1} ^ {*} = \boldsymbol {W} _ {i} ^ {*} \tilde {\boldsymbol {V}} _ {i} ^ {*} = \boldsymbol {W} _ {i} ^ {*} \tilde {\boldsymbol {V}} _ {i}. \tag {21}
$$

Since $\tilde{V}_{i} = \tilde{V}_{i}^{*}$ is rank $n^{2}$ by the $\Pi^{*}$ -coverable assumption and $P_{i} \in R^{n^{2} \times n^{2}}$ , it must be so that $P_{i} = W_{i}^{*}$ .

If we further condition on the held-out level $P_{i}$ such that we only leave one column of $P_{i}^{(k)}$ not necessarily equal to $W_{i}^{*(k)}$ , we have the following corollary

Corollary G.15 (Uniqueness of a single $P_{i}$ column when conditioning everything else).

Fix a target set of phrasebooks $\Pi^{*} = \{\pi_{1}^{*},\ldots ,\pi_{d}^{*}\}$ in MLT(d,n) and a $\Pi^{*}$ -coverable input $V_{1}$ . For any level $i\in [d]$ and translation entry $k\in [n^2 ]$ , consider a surrogate model SURR-MLT $_{\{W_i\}_{i = 1}^d}(C_1,C_2,\dots,C_d,\cdot)$

with certain context $C_1, C_2, \ldots, C_d$ such that $P_l^{(j)} = W_l^{*(j)}$ for all $(l,j) \in [d] \times [n^2] \setminus \{(i,k)\}$ . Then $\mathrm{SURR - MLT}_{\{\boldsymbol{W}_i\}_{i=1}^d}(\boldsymbol{C}_1, \boldsymbol{C}_2, \ldots, \boldsymbol{C}_d, \boldsymbol{V}_1) = \mathbf{MLT}_{\boldsymbol{\Pi}^*}(\boldsymbol{V}_1)$ if and only if

$$
\boldsymbol {P} _ {i} ^ {(k)} = \mathrm{HardMax} \left(\boldsymbol {C} _ {i} ^ {(j)} + \boldsymbol {W} _ {i} ^ {(j)}\right) = \boldsymbol {W} _ {i} ^ {* (k)}.
$$

This suggests that if we condition everything else except for one column of the translation operator, then to match the final output on a $\Pi^{*}$ -coverable sequence, the model must recover the held-out column to be the same as the ground truth in the set of phrasebooks.

Now we can introduce the search algorithm, which simply enumerate over all translation columns $\boldsymbol{W}_{i}^{(j)}$ as learning target, generate a contextual information that only leaves that column unconditioned, and search over all possible one-hot vectors for $\boldsymbol{W}_{i}^{(j)}$ until the output matches with $MLT_{\Pi^{*}}$ . Once the output matches, by Corollary G.15 we know $\text{HardMax}(\boldsymbol{W}_{i}^{(j)})$ recovers $\boldsymbol{W}_{i}^{*(j)}$ and we move on to the next learning target. The algorithm can be formalized as follows:

Algorithm 2 Context-Enhanced Searching Algorithm for MLT(d, n)   
1: Input:
2: input $V_{1} \in R^{n^{2} \times L}$ , label $V_{d+1}^{*} \in R^{n^{2} \times L}$ , descriptive text $W_{1}^{*}, \ldots, W_{d}^{*} \in R^{n^{2} \times n^{2}}$ 3:
4: Initialize $W_{1}, \ldots, W_{d} \leftarrow 0$ # Start with zero initialization
5: for i = 1 to d do
6:    for k = 1 to $n^{2}$ do
7:    Initialize $C_{i(k)} \leftarrow W_{i}^{*}(I_{n^{2}} - \text{diag}(\bar{e}_{k}))$ # Create masked context matrix
8:    # Search Loop
9:    for a = 1 to $n^{2}$ do
10: $W_{i}^{(k)} \leftarrow \bar{e}_{a}$ # Search over one-hot columns
11: $V_{d+1} \leftarrow SURR-MLT_{\{W_{i}\}_{i=1}^{d}}(W_{1}^{*} \ldots, W_{i-1}^{*}, C_{i(k)}, W_{i+1}^{*} \ldots, W_{d}^{*}, V_{1})$ 12:    if $V_{d+1} = V_{d+1}^{*}$ then
13:    break    # Break when found the right column
14:    end if
15:    end for
16:    end for
17: end for
18: Return $W_{1}, \ldots, W_{d}$ .

Theorem G.16 (Learning $\Pi^{*}$ with context-enhanced search with $\Pi^{*}$ -coverable input).

For any target set of phrasebooks $\Pi^{*}=\{\pi_{1}^{*},\ldots,\pi_{d}^{*}\}$ in MLT(d,n), given an $\Pi^{*}$ -coverable input $V_{1}$ and the corresponding ground truth label $\mathbf{V}_{d+1}^{*}=\mathbf{MLT}_{\mathbf{\Pi}^{*}}(\mathbf{V}_{1})$ , Algorithm 2 terminates with $W_{i}=W_{i}^{*}$ for all $i\in[d]$ with $O(n^{4}d)$ forward passes through the surrogate model.

Proof. The statement can be proven by a simple induction.

Let the inductive hypothesis be such that when the enumeration goes to k-th column of the i-th layer, if $\operatorname{HardMax}(\boldsymbol{W}_{l}^{(j)} + \boldsymbol{W}_{l}^{*(j)}) = \boldsymbol{W}_{l}^{*(j)}$ for all $(l,j) \in [d] \times [n^{2}]$ and $\boldsymbol{W}_{l}^{(j)} = \boldsymbol{W}_{l}^{*(j)}$ for all $(l,j)$ such that l < i or $l = i \wedge j < k$ , then the search loop (Algorithm 2: line 13) breaks with $\boldsymbol{W}_{i}^{(k)} = \boldsymbol{W}_{i}^{*(k)}$ while $\operatorname{HardMax}(\boldsymbol{W}_{l}^{(j)} + \boldsymbol{W}_{l}^{*(j)}) = \boldsymbol{W}_{l}^{*(j)}$ for all $(l,j) \in [d] \times [n^{2}]$ is preserved.

The base case is satisfied as with zero initialization, we have $\text{HardMax}(\boldsymbol{W}_{l}^{(j)} + \boldsymbol{W}_{l}^{*(j)}) = \text{HardMax}(\boldsymbol{0} + \boldsymbol{W}_{l}^{*(j)}) = \boldsymbol{W}_{l}^{*(j)}$ for all $(l, j) \in [d] \times [n^{2}]$ and there are no requirements for $\boldsymbol{W}_{i}^{(k)} = \boldsymbol{W}_{i}^{*(k)}$ yet.

For the induction step, we note that with the condition of $\mathrm{HardMax}(\boldsymbol{W}_l^{(j)} + \boldsymbol{W}_l^{*(j)}) = \boldsymbol{W}_l^{*(j)}$ for all $(l,j)\in [d]\times [n^2],$

$W_{1}^{*}\ldots,W_{i-1}^{*},C_{i(k)},W_{i+1}^{*}\ldots,W_{d}^{*}$ will correctly condition all columns of P's except for the $P_{i}^{(k)}$ since

$$
\boldsymbol {C} _ {i (k)} ^ {(k)} = \boldsymbol {W} _ {i} ^ {*} \left(\boldsymbol {I} _ {n ^ {2}} - \mathrm{diag} (\bar {\boldsymbol {e}} _ {k})\right) ^ {(k)} = \mathbf {0}. \tag {22}
$$

Thus by Corollary G.15, we know that the search loop will terminate when it finds $\boldsymbol{W}_{i}^{(k)} = \boldsymbol{W}_{i}^{*(k)}$ . The newly added column provides the correct inductive hypothesis on $\boldsymbol{W}_{l}^{(j)} = \boldsymbol{W}_{l}^{*(j)}$ for the next enumeration step.

By induction to i = d and $k = n^{2}$ , we will be able to recover $W_{i} = W_{i}^{*}$ for all $i \in [d]$ .

Given that we can learn $W_{i}$ effectively with $\Pi^{*}$ -coverable input, how should we construct such inputs? It turned out that with high probability, short random strings suffices.

Lemma G.17 (Distribution of intermediate sequences). Fix a target set of phrasebooks $\Pi^{*} = \{\pi_{1}^{*},\dots,\pi_{d}^{*}\}$ in $\mathbf{MLT}(d,n)$ . Let $\mathbf{V}_1\in \mathbb{R}^{n^2\times L}$ be a random input matrix to $\mathbf{MLT}(d,n)$ such that each column $\mathbf{V}_1^{(j)}$ is i.i.d. sampled from $\mathcal{U}(\{\bar{\boldsymbol{e}}_k\}_{k = 1}^{n^2})$ (the uniform distribution over one-hot vectors $\{\bar{\boldsymbol{e}}_k\}_{k = 1}^{n^2}$ ), the columns of intermediate random sequences $\tilde{\mathbf{V}}_1^*,\mathbf{V}_2^*,\tilde{\mathbf{V}}_2^*,\ldots ,\tilde{\mathbf{V}}_d^*,\mathbf{V}_{d + 1}^*\in \mathbb{R}^{n^2\times L}$ obtained by passing the input $\mathbf{V}_1$ through $\mathbf{MLT}_{\Pi^*}$ also follow the same i.i.d. uniform distribution.

Proof. We will prove the claim by induction on depth i. Let the inductive hypothesis be that columns in $V_{i}$ independently follow an uniform distribution over the one-hot vectors $\left\{\bar{e}_{k}\right\}_{k=1}^{n^{2}}$ . Note that the base case is just the assumption.

Now we prove for the inductive step. For any $j \in [L]$ , we can write $\mathbf{V}_i^{(j)} = e_{\mathrm{a}_{(2j-1)}} \otimes e_{\mathrm{a}_{(2j)}}$ where $\mathrm{a}_{(i)}$ 's follows i.i.d. $\mathcal{U}([n])$ . Intuitively this means each 2-tuple in the random input sequence is formed from two i.i.d. uniformly random characters, which is straightforward by construction.

Now by Lemma G.3 we have $\tilde{\mathbf{V}}_i^{*(j)} = e_{\mathrm{a}_{(2j)}}\otimes e_{\mathrm{a}_{(2j + 1)}}$ also following $\mathcal{U}(\{\bar{e}_k\}_{k = 1}^n)$ . Note that there is total independency of $\tilde{\mathbf{V}}_i^{*(j)}$ with respect to the set of any other columns of $\tilde{\mathbf{V}}_i$ since $\mathrm{a}_{(2j)}$ and $\mathrm{a}_{(2j + 1)}$ are independent from the generative process of any other columns in $\tilde{\mathbf{V}}_i$ .

With columns in $\tilde{V}_{i}$ i.i.d. following $\mathcal{U}(\{\bar{e}_{k}\}_{k=1}^{n^{2}})$ , permuting the indices via $P^{(\pi_{i}^{*})}$ does not change the distribution by symmetry of the uniform distribution. Therefore we have columns of $V_{i+1}=P^{(\pi_{i}^{*})}\tilde{V}_{i}$ also i.i.d. distributed as $\mathcal{U}(\{\bar{e}_{k}\}_{k=1}^{n^{2}})$ and this complete the inductive step. By induction on i we have the desired statement proved.

Since each column in $\tilde{V}_{i}$ i.i.d. follows $\mathcal{U}(\{\bar{e}_{k}\}_{k=1}^{n^{2}})$ , sampling $\tilde{V}_{i}$ to be rank $n^{2}$ becomes identical to the classic coupon collection problem (see Lemma G.29 from Motwani (1995)). Thus we have the following bound:

Lemma G.18 (Short $\Pi^{*}$ -coverable random sequence).

A random sequence $V_{1}$ of length $L \geq 2n^{2} \log \frac{nd}{\delta}$ is $\Pi^{*}$ -coverable with probability at least $1 - \delta$ .

Proof. Let the event $A_{i}$ denote that $\tilde{\mathbf{V}}_i$ is not rank $n^2$ . By Lemma G.17, each column of $\tilde{\mathbf{V}}_i$ is i.i.d. distributed following $\mathcal{U}(\{\bar{e}_k\}_{k=1}^{n^2})$ . Thus making $\tilde{\mathbf{V}}_i$ being rank $n^2$ is equivalent to a coupon collecting problem (Motwani, 1995) with set size $n^2$ . By Lemma G.29 we know that with $L = 2n^2 \log \frac{nd}{\delta}$ , $\mathbb{P}[A_i] \leq \frac{\delta}{d}$ . Thus by a simple union bound the probability that $\tilde{\mathbf{V}}_i$ being not $\Pi^*$ -coverable is

$$
\mathbb {P} \left[ \bigcup_ {i = 1} ^ {d} A _ {i} \right] \leq \sum_ {i = 1} ^ {d} \mathbb {P} \left[ A _ {i} \right] \leq d \frac {\delta}{d} = \delta . \tag {23}
$$

Now we can apply the above result and extend Theorem G.16.

Corollary G.19 (Learning $\Pi^{*}$ with random input using heuristics search).

For any target set of phrasebooks $\Pi^{*}=\{\pi_{1}^{*},\ldots,\pi_{d}^{*}\}$ in MLT(d,n), with probability at least $1-\delta$ over a uniformly random input $V_{1}$ of length $L=2n^{2}\log\frac{nd}{\delta}$ , Algorithm 2 provided with ground truth label $\mathbf{V}_{d+1}^{*}=\mathbf{MLT}_{\mathbf{\Pi}^{*}}(\mathbf{V}_{1})$ terminates with $W_{i}=W_{i}^{*}$ for all $i\in[d]$ with $O(n^{4}d)$ forward passes through the surrogate model.

# G.3. Learning $\Pi^{*}$ in MLT(2, $n$ ) with Surrogate Gradient Descent

In this section we take the analysis one step beyond the heuristics searching regime. We will show that any set of phrasebooks $\Pi^{*}=\left\{\pi_{1}^{*},\pi_{2}^{*}\right\}$ can be sample-efficiently learned by a gradient-descent based algorithm. In this particular case, the surrogate model is parameterized by

$$
\boldsymbol {V} _ {3} = \operatorname{Surr-MLT} _ {\left\{\boldsymbol {W} _ {i} \right\} _ {i = 1} ^ {d}} \left(\boldsymbol {C} _ {1}, \boldsymbol {C} _ {2}, \boldsymbol {V} _ {1}\right) \triangleq \operatorname{HardMax} \left(\boldsymbol {C} _ {2} + \boldsymbol {W} _ {2}\right) \text {Shift} \left(\operatorname{HardMax} \left(\boldsymbol {C} _ {1} + \boldsymbol {W} _ {1}\right) \text {Shift} \left(\boldsymbol {V} _ {1}\right)\right). \tag {24}
$$

We start with any weight initializations $\boldsymbol{W}_{1}^{(0)}$ , $\boldsymbol{W}_{2}^{(0)} \in \mathbb{R}^{n^{2} \times n^{2}}$ satisfying $\| \boldsymbol{W}_{1}^{(0)} \|_{1} < \frac{1}{2}$ , $\| \boldsymbol{W}_{2}^{(0)} \|_{1} < \frac{1}{2}$ , by Lemma G.11 the initialization is strongly $\mathbf{MLT}(d, n)$ -ICL-capable.

For simplicity, we denote the ground truth permutation matrix induced by $\pi_{1}^{*}$ as $\boldsymbol{W}_{1}^{*}\triangleq\boldsymbol{P}^{(\pi_{1}^{*})}$ and similarly the ground truth permutation matrix induced by $\pi_{2}^{*}$ as $\boldsymbol{W}_{2}^{*}\triangleq\boldsymbol{P}^{(\pi_{2}^{*})}$ . From Lemma G.12 we know that the learning is successful if we have $\operatorname{HardMax}\left(\boldsymbol{W}_{1}\right)=\boldsymbol{W}_{1}^{*}$ and $\operatorname{HardMax}\left(\boldsymbol{W}_{2}\right)=\boldsymbol{W}_{2}^{*8}$ (i.e. the maximum index of each weight column agrees with that of the ground truth).

We employ a layer-wise gradient descent algorithm for the learning process. The algorithm takes in a single fixed sequence $s_{1}$ with matrix representation $V_{1}$ and its corresponding ground truth label

$$
\boldsymbol {V} _ {3} ^ {*} \triangleq \mathbf {M a t} (\mathbf {M L T} _ {\boldsymbol {\Pi} ^ {*}} (\boldsymbol {s} _ {1})) = \text { S   U   R   R   -   M   L   T } _ {\left(\boldsymbol {W} _ {1} ^ {(0)}, \boldsymbol {W} _ {2} ^ {(0)}\right)} \left(\boldsymbol {W} _ {1} ^ {*}, \boldsymbol {W} _ {2} ^ {*}, \boldsymbol {V} _ {1}\right). \tag {25}
$$

Given the input and label, we employ the following gradient descent based algorithm to update the weights:

The training happens in a layer-wise and column-wise fashion: We first freeze $W_{2}$ and set $W_{1}$ as the trainable parameter. For each entry $k \in [n^{2}]$ , we create a context matrix $C_{1(k)} \triangleq W_{1}^{*}(I_{n^{2}} - \text{diag}(e_{k}))$ which essentially creates a copy of $W_{1}^{*}$ except of setting the k-th column to be zero. Then we take a forward pass through the surrogate model with the one-column dropped-out context and get output $V_{3(1,k)} \triangleq \text{SURR-MLT}_{(W_{1},W_{2})}(C_{1(k)}, W_{2}^{*}, V_{1})$ . Here the subscript $\cdot_{(1,k)}$ denotes the final output when the i-th column of the first context matrix is being dropped.

The weight update follows a surrogate gradient update scheme where we use the MSE loss: $\mathcal{L} = \|\boldsymbol{V}_{3(1,k)} - \boldsymbol{V}_{3}^{*}\|_{2}^{2}$ . Since it is difficult to take gradient through the hardmax function, we instead compute the gradient of the loss with respect to the translation matrix $P_{1} = \text{HardMax}\left(\boldsymbol{W}_{1} + \boldsymbol{C}_{1(k)}\right)$ and apply the update $\boldsymbol{W}_{1}^{(k)} \leftarrow \boldsymbol{W}_{1}^{(k)} - \frac{\partial\mathcal{L}}{\partial\boldsymbol{P}_{1}^{(k)}}$ . We apply such gradient update twice for each dropped column k.

For the second layer, we freeze the first layer $\pmb{W}_1$ and apply one-column dropouts to the $C_2$ . Similarly we apply the surrogate gradient update $\pmb{W}_2 \leftarrow \pmb{W}_2 - \frac{\partial\mathcal{L}}{\partial P_2}$ but we only need one gradient step per column.

We claim the surrogate gradient descent update can correctly recover $P_{1}^{*}$ and $P_{2}^{*}$ similar to the heuristics search case.

Theorem G.24 (Learning $\Pi^{*}$ with context-enhanced surrogate GD with $\Pi^{*}$ -coverable input).

For any initialization $\boldsymbol{W}_{1(0)}, \boldsymbol{W}_{2(0)} \in \mathbb{R}^{n^{2} \times n^{2}}$ such that $\|W_{1(0)}\|_{0} \leq \frac{1}{2}$ and $\|W_{2(0)}\|_{0} \leq \frac{1}{2}$ , for any target set of phrasebooks $\Pi^{*} = \{\pi_{1}^{*}, \pi_{2}^{*}\}$ in MLT(2, n), given an $\Pi^{*}$ -coverable input $V_{1}$ and the corresponding ground truth label $\boldsymbol{V}_{3}^{*} = \boldsymbol{MLT}_{\Pi^{*}}(\boldsymbol{V}_{1})$ , Algorithm 2 terminates with HardMax $(\boldsymbol{W}_{1}) = \boldsymbol{W}_{1}^{*}$ and HardMax $(\boldsymbol{W}_{2}) = \boldsymbol{W}_{2}^{*}$ .

To prove for Theorem G.24, we will carefully analyze the learning of the first layer and second layer respectively, and provide a similar induction argument as in the proof for the heuristics search case.

Algorithm 3 Context-Enhanced Layerwise Gradient Descent   
1: Input: input $V_{1} \in R^{n^{2} \times L}$ , label $V_{3}^{*} \in R^{n^{2} \times L}$ , descriptive text $W_{1}^{*}, W_{2}^{*} \in R^{n^{2} \times n^{2}}$ , init $W_{1}^{(0)}, W_{2}^{(0)} \in R^{n^{2} \times n^{2}}$ 2:
3: # Train the first layer
4: for k = 1 to $n^{2}$ do
5: $C_{1(k)} \triangleq W_{1}^{*}(I_{n^{2}} - \text{diag}(e_{k}))$ # Create context matrix with k-th column dropped.
6: for t = 1 to 2 do
7: $V_{3(1,k)} \leftarrow \text{SURR-MLT}_{(W_{1},W_{2})}(C_{1(k)}, W_{2}^{*}, V_{1})$ # Forward pass
8: $L \leftarrow \|V_{3(1,k)} - V_{3}^{*}\|_{2}^{2}$ 9: $W_{1}^{(k)} \leftarrow W_{1}^{(k)} - \frac{\partial l}{\partial P_{1}^{(k)}}$ # Surrogate gradient update
10: end for
11: end for
12:
13: # Train the second layer
14: for k = 1 to $n^{2}$ do
15: $C_{2(k)} \triangleq W_{1}^{*}(I_{n^{2}} - \text{diag}(e_{k}))$ # Create context matrix with k-th column dropped.
16: $V_{3(2,k)} \leftarrow \text{SURR-MLT}_{(W_{1},W_{2})}(W_{1}^{*}, C_{2(k)}, V_{1})$ # Forward pass
17: $L \leftarrow \|V_{3(2,k)} - V_{3}^{*}\|_{2}^{2}$ 18: $W_{2}^{(k)} \leftarrow W_{2}^{(k)} - \frac{\partial l}{\partial P_{2}^{(k)}}$ # Surrogate gradient update
19: end for
20: Return $W_{1}, W_{2}$

# G.3.1. LEARNING THE FIRST LAYER

To study the learning process we first need to compute the closed-form gradient $\frac{\partial\mathcal{L}}{\partial P_{1}^{(k)}}$ , which requires the following lemma:

Lemma G.20 (Gradient with respect to incorrect column in $P_{1}$ ).

When only the $k$ -th column of translation matrix $\pmb{P}_1^{(k)}$ is not equal to $\pmb{P}_1^{*(k)}$ , let $\pmb{P}_1^{(k)} = \pmb{e}_a \otimes \pmb{e}_b$ and $\pmb{P}_1^{*(k)} = \pmb{e}_{a^*} \otimes \pmb{e}_{b^*}$ for some $a, b, a^*, b^* \in [n]$ . If $\pmb{P}_1^{(k)}$ is used in the forward pass, there exists $\alpha \in \mathbb{Z}_+$ and $\beta \in \mathbb{N}$ such that the gradient of $\mathcal{L}$ with respect to $\pmb{P}_1^{(k)}$ is of the form

$$
\frac {\partial \mathcal {L}}{\partial P _ {1} ^ {(k)}} = \left\{ \begin{array}{l l} (2 \alpha + 2 \beta) (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b}) - (2 \alpha + 2 \beta) (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}) + 2 \beta (\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}) & i f a ^ {*} = a, b ^ {*} \neq b \\ (2 \alpha + 2 \beta) (\boldsymbol {e} _ {a} \otimes \mathbf {1} _ {n}) - (2 \alpha + 2 \beta) (\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}) + 2 \beta (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}) & i f a ^ {*} \neq a, b ^ {*} = b \\ (2 \alpha + 2 \beta) (\boldsymbol {e} _ {a} \otimes \mathbf {1} _ {n}) + (2 \alpha + 2 \beta) (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b}) - 2 \alpha (\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}) - 2 \alpha (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}) & i f a ^ {*} \neq a, b ^ {*} \neq b. \end{array} \right.
$$

Proof of Lemma G.20. In this proof we use $\bar{e}_{a}$ (long one-hot) to denote the one-hot vector in $R^{n^{2}}$ with 1 on the a-th index and use $e_{a}$ (short one-hot) to denote the one-hot vector in $R^{n}$ with 1 on the a-th index. We use $\tilde{V}_{1}, V_{2}, \tilde{V}_{2}$ and $V_{3}$ to denote the intermediate sequences attained with translation matrix $P_{1}^{(k)}$ and $\tilde{V}_{1}^{*}, V_{2}^{*}, \tilde{V}_{2}^{*}$ and $V_{3}^{*}$ to denote the counterfactual intermediate sequences should the forward pass is done with the ground truth translations $P_{1}^{*(k)} = W_{1}^{*}$ .

Now we can proceed to the gradient calculations. First note that with $\mathcal{L}(\boldsymbol{P}_{1})=\|\boldsymbol{V}_{3}-\boldsymbol{V}_{3}^{*}\|_{2}^{2}$ , by chain rule we have

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1}} = \sum_ {j = 1} ^ {L} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1}}\right) ^ {\top} \left(\frac {\partial \| \boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)} \| _ {2} ^ {2}}{\partial \boldsymbol {V} _ {3} ^ {(j)}}\right) = 2 \sum_ {j = 1} ^ {L} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right). \tag {26}
$$

Specifically for each column $l \in [n^{2}]$ of $P_{1}$ we have

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1} ^ {(l)}} = 2 \sum_ {j = 1} ^ {L} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(l)}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right). \tag {27}
$$

Let us fix a particular column $j \in [L]$ in the output $V_{3}$ and compute the gradients. With $M^{(j)}$ as the $j$ -th column of the matrix $M$ , the computation graph for the forward pass is of the form:

$$
\begin{array}{c c c c c c c c c} \boldsymbol {V} _ {1} ^ {(j)} & \to & \tilde {\boldsymbol {V}} _ {1} ^ {(j)} & \stackrel {{P _ {1}}} {{\longrightarrow}} & \boldsymbol {V} _ {2} ^ {(j)} & \to & \tilde {\boldsymbol {V}} _ {2} ^ {(j)} & \stackrel {{P _ {2}}} {{\longrightarrow}} & \boldsymbol {V} _ {3} ^ {(j)} \\ & \nearrow & & & \nearrow & & & \end{array}
$$

$$
\boldsymbol {V} _ {1} ^ {(j + 1)} \rightarrow \tilde {\boldsymbol {V}} _ {1} ^ {(j + 1)} \xrightarrow {\boldsymbol {P} _ {1}} \boldsymbol {V} _ {2} ^ {(j + 1)} \tag {28}
$$

$$
\nearrow
$$

$$
V _ {1} ^ {(j + 2)}
$$

We can see that $\boldsymbol{V}_{3}^{(j)}$ is only affected by $P_{1}$ through $\boldsymbol{V}_{2}^{(j)}$ and $\boldsymbol{V}_{2}^{(j+1)}$ .

Let us first compute $\partial V_3^{(j)} / \partial P_1$ . Assume $V_2^{(j)} = \bar{e}_p$ and $V_2^{(j + 1)} = \bar{e}_q$ for some $p, q \in [n^2]$ , $V_3^{(j)}$ is computed as

$$
\boldsymbol {V} _ {3} ^ {(j)} = \boldsymbol {W} _ {2} ^ {*} \tilde {\boldsymbol {V}} _ {2} ^ {(j)} = \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {V} _ {2} ^ {(j)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {V} _ {2} ^ {(j + 1)}\right) = \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right). \tag {29}
$$

We may express the Hadamard product in the following two ways:

$$
\begin{array}{l} \boldsymbol {V} _ {3} ^ {(j)} = \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) = \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}; \\ = \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)} \odot \boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) = \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}. \\ \end{array}
$$

Therefore when $p \neq q$ we have

$$
\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(p)}} = \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \boldsymbol {Q}; \quad \frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(q)}} = \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {Q} ^ {\top}; \quad \forall l \notin \{p, q \}: \frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(l)}} = \mathbf {0}. \tag {31}
$$

When $p = q$ , the expression would be

$$
\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(p)}} = \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {Q} + \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {Q} ^ {\top}; \quad \forall l \neq p: \frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(l)}} = \mathbf {0}. \tag {32}
$$

Now we move on to compute $(\boldsymbol{V}_{3}^{(j)} - \boldsymbol{V}_{3}^{*(j)})$ . There are in total four cases to consider: depending on whether $\boldsymbol{P}_{1}^{(k)}$ is being used when computing $\tilde{\boldsymbol{V}}_{2}^{(j)}$ and $\tilde{\boldsymbol{V}}_{2}^{(j+1)}$ . Let us go over these cases one-by-one.

1. Case 1: $\tilde{V}_1^{(j)}\neq \bar{e}_k$ and $\tilde{V}_1^{(j + 1)}\neq \bar{e}_k$

Assume $\tilde{\boldsymbol{V}}_1^{(j)} = \bar{\boldsymbol{e}}_p$ and $\tilde{\boldsymbol{V}}_1^{(j + 1)} = \bar{\boldsymbol{e}}_q$ for some $p,q\neq k$ . Since $\boldsymbol{V}_2^{(j)} = \boldsymbol{P}_1\tilde{\boldsymbol{V}}_1^{(j)}$ while $\boldsymbol{P}_1$ equals $\boldsymbol{P}_1^*$ for all columns not equal to $k$ by assumption, we have $\boldsymbol{V}_2^{(j)} = \boldsymbol{P}_1\bar{\boldsymbol{e}}_p = \boldsymbol{P}_1^{(p)} = \boldsymbol{P}_1^{*(p)} = \boldsymbol{P}_1^*\bar{\boldsymbol{e}}_p = \boldsymbol{V}_2^{*(j)}$ . With an identical argument we have $\boldsymbol{V}_2^{(j + 1)} = \boldsymbol{V}_2^{*(j + 1)}$ . Thus in this case $\boldsymbol{V}_3^{(j)} = \boldsymbol{V}_3^{*(j)}$ and hence

$$
\left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right) = \mathbf {0}. \tag {33}
$$

2. Case 2: $\tilde{V}_1^{(j)} = \bar{e}_k$ and $\tilde{V}_1^{(j + 1)}\neq \bar{e}_k$

Assume $\tilde{\boldsymbol{V}}_1^{(j + 1)} = \bar{\boldsymbol{e}}_q$ for some $q\neq k$ , from case 1 we know $\boldsymbol{V}_2^{(j + 1)} = \boldsymbol{V}_2^{*(j + 1)}$ . However since $\tilde{\boldsymbol{V}}_1^{(j)} = \bar{\boldsymbol{e}}_k$ we have $\boldsymbol{V}_2^{(j)} = \boldsymbol{P}_1\bar{\boldsymbol{e}}_k = \boldsymbol{P}_1^{(k)}$ , which is not equal to $\boldsymbol{V}_2^{*(j)} = \boldsymbol{P}_1^{*(k)}$ . Therefore

$$
\begin{array}{l} \boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)} = \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) - \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {* (k)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \\ = \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)} \odot \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right)\right) \tag {34} \\ = \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right). \\ \end{array}
$$

For the gradient with respect to $P_1^{(k)}$ , combining Equation (34) with Equation (31) (substituting $p = k$ ) we have

$$
\begin{array}{l} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(k)}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right) = \left(\boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \boldsymbol {Q}\right) ^ {\top} \left(\boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right)\right) \\ = \boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \boldsymbol {W} _ {2} ^ {* \top} \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) (35) \\ = \boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(q)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) \\ = \left(\left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}\right) \otimes I _ {n}\right) \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) (byLemmaG.27) \\ \end{array}
$$

where we dropped $\boldsymbol{W}_2^{*\top}\boldsymbol{W}_2^*$ in the third step since $\boldsymbol{W}_2^*$ is a permutation matrix and $\boldsymbol{W}_2^{*\top}\boldsymbol{W}_2^* = I_n$ .

Now plugging in $P_1^{(k)} = e_a \otimes e_b$ and $P_1^{*(k)} = e_{a^*} \otimes e_{b^*}$ we have

$$
\begin{array}{l} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(k)}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right) = \left(\left(\boldsymbol {1} _ {n} \boldsymbol {1} _ {n} ^ {\top}\right) \otimes I _ {n}\right) \left(\boldsymbol {e} _ {a} \otimes \boldsymbol {e} _ {b}\right) - \left(\left(\boldsymbol {1} _ {n} \boldsymbol {1} _ {n} ^ {\top}\right) \otimes I _ {n}\right) \left(\boldsymbol {e} _ {a ^ {*}} \otimes \boldsymbol {e} _ {b ^ {*}}\right) \\ = \left(\left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top} \boldsymbol {e} _ {a}\right) \otimes I _ {n} \boldsymbol {e} _ {b}\right) - \left(\left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top} \boldsymbol {e} _ {a ^ {*}}\right) \otimes I _ {n} \boldsymbol {e} _ {b ^ {*}}\right) \tag {36} \\ = \mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b} - \mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}} \\ = \mathbf {1} _ {n} \otimes \left(\boldsymbol {e} _ {b} - \boldsymbol {e} _ {b ^ {*}}\right). \\ \end{array}
$$

3. Case 3: $\tilde{V}_1^{(j)}\neq \bar{e}_k$ and $\tilde{V}_1^{(j + 1)} = \bar{e}_k$

This is a symmetric case with respect to case 2. Assume $\tilde{V}_1^{(j)} = \bar{e}_p$ for some $p \neq k$ , from case 1 we know $V_2^{(j)} = V_2^{*(j)}$ . However since $\tilde{V}_1^{(j)} = \bar{e}_k$ we have $V_2^{(j+1)} = P_1\bar{e}_k = P_1^{(k)}$ , which is not equal to $V_2^{*(j+1)} = P_1^{*(k)}$ . Therefore similar to case 2 we have

$$
\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)} = \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right). \tag {37}
$$

Combining Equation (37) with Equation (32) gives

$$
\begin{array}{l} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(k)}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right) = \left(\boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {Q} ^ {\top}\right) ^ {\top} \left(\boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right)\right) \\ = \boldsymbol {Q} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {W} _ {2} ^ {* \top} \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) (38) \\ = \boldsymbol {Q} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(p)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) \\ = \left(I _ {n} \otimes (\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top})\right) \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) (byLemmaG.28) \\ \end{array}
$$

Now plugging in $P_1^{(k)} = e_a \otimes e_b$ and $P_1^{*(k)} = e_{a^*} \otimes e_{b^*}$ we have

$$
\begin{array}{l} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1} ^ {(k)}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right) = \left(I _ {n} \otimes \left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}\right)\right) \left(\boldsymbol {e} _ {a} \otimes \boldsymbol {e} _ {b}\right) - \left(I _ {n} \otimes \left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}\right)\right) \left(\boldsymbol {e} _ {a ^ {*}} \otimes \boldsymbol {e} _ {b ^ {*}}\right) \\ = \left(I _ {n} \boldsymbol {e} _ {a} \otimes \left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top} \boldsymbol {e} _ {b}\right)\right) - \left(I _ {n} \boldsymbol {e} _ {a ^ {*}} \otimes \left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top} \boldsymbol {e} _ {b ^ {*}}\right)\right) \tag {39} \\ = \boldsymbol {e} _ {a} \otimes \mathbf {1} _ {n} - \boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n} \\ = \left(\boldsymbol {e} _ {a} - \boldsymbol {e} _ {a ^ {*}}\right) \otimes \mathbf {1} _ {n}. \\ \end{array}
$$

4. Case 4: $\tilde{V}_1^{(j)} = \tilde{V}_1^{(j + 1)} = \bar{e}_k$ .

This is the most complicated case since the loss is contributed by two different paths. We can first decompose the negative residual as

$$
\begin{array}{l} \boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)} = \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) - \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {* (k)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \\ = \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) - \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \tag {40} \\ + \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) - \boldsymbol {W} _ {2} ^ {*} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {* (k)} \odot \boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \\ = \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) + \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (p)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right). \\ \end{array}
$$

Combining with Equation (32), we have

$$
\begin{array}{l} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right) \\ = \left(\boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \boldsymbol {Q} + \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \boldsymbol {Q} ^ {\top}\right) ^ {\top} \\ \left(\boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) + \boldsymbol {W} _ {2} ^ {*} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right)\right) \\ = \boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) (a) \\ + \boldsymbol {Q} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) (b) \\ + \boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) (c) \\ + \boldsymbol {Q} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right). (d) \\ \end{array}
$$

Now let us analyze the four cross terms term-by-term.

(a) With $P_1^{(k)} = e_a \otimes e_b$ , by Lemma G.26 we have $Q^\top P_1^{(k)} = 1_n \otimes e_a$ and $QP_1^{(k)} = e_b \otimes 1_n$ . Therefore $Q^\top P_1^{(k)} \odot QP_1^{(k)} = e_b \otimes e_a$ and hence

$$
\operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) = \operatorname{diag} \left(\boldsymbol {e} _ {b} \otimes \boldsymbol {e} _ {a}\right) = \operatorname{diag} \left(\boldsymbol {e} _ {b}\right) \otimes \operatorname{diag} \left(\boldsymbol {e} _ {a}\right) \tag {41}
$$

It follows that

$$
\begin{array}{l} \boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \boldsymbol {Q} ^ {\top} \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \left(\operatorname{diag} \left(\boldsymbol {e} _ {b}\right) \otimes \operatorname{diag} \left(\boldsymbol {e} _ {a}\right)\right) \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {b} \otimes \operatorname{diag} \left(\boldsymbol {e} _ {a}\right)\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \tag {42} \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {b} \otimes \boldsymbol {e} _ {a} ^ {\top}\right) \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {b} \boldsymbol {e} _ {a} ^ {\top} \otimes 1\right) \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(\boldsymbol {e} _ {b} \boldsymbol {e} _ {a} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \\ \end{array}
$$

Plugging into (a) we have

$$
\begin{array}{l} (a) = \boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(\boldsymbol {e} _ {b} \boldsymbol {e} _ {a} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {a} \otimes \boldsymbol {e} _ {b}\right) - \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(\boldsymbol {e} _ {b} \boldsymbol {e} _ {a} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {a ^ {*}} \otimes \boldsymbol {e} _ {b ^ {*}}\right) \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(\boldsymbol {e} _ {b} \boldsymbol {e} _ {a} ^ {\top} \boldsymbol {e} _ {a} \otimes 1\right) - \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(\boldsymbol {e} _ {b} \boldsymbol {e} _ {a} ^ {\top} \boldsymbol {e} _ {a ^ {*}} \otimes 1\right) \tag {43} \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(1 \otimes \boldsymbol {e} _ {b} \boldsymbol {e} _ {a} ^ {\top} \boldsymbol {e} _ {a}\right) - \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(1 \otimes \boldsymbol {e} _ {b} \boldsymbol {e} _ {a} ^ {\top} \boldsymbol {e} _ {a ^ {*}}\right) \\ = \left\{ \begin{array}{l l} \mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b} & \text { when } a \neq a ^ {*} \\ \mathbf {0} & \text { otherwise } \end{array} \right. \\ \end{array}
$$

(b) We have seen the same term as in case 3, by Lemma G.28 we have

$$
\text {(b)} = \boldsymbol {Q} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \boldsymbol {Q} ^ {\top} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) = \left(\boldsymbol {e} _ {a} - \boldsymbol {e} _ {a ^ {*}}\right) \otimes \mathbf {1} _ {n}. \tag {44}
$$

(c) With $P_1^{(k)} = e_a \otimes e_b$ and $P_1^{*(k)} = e_{a^*} \otimes e_{b^*}$ , we have

$$
\operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) = \operatorname{diag} \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {a}\right) \operatorname{diag} \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {a ^ {*}}\right) = \left\{ \begin{array}{l l} \operatorname{diag} \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {a}\right) & \text { if } a = a ^ {*} \\ \mathbf {0} & \text { otherwise } \end{array} \right. \tag {45}
$$

Thus by Lemma G.27 we have

$$
\boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \boldsymbol {Q} = \left\{ \begin{array}{l l} \left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}\right) \otimes I _ {n} & \text { if } a = a ^ {*} \\ \mathbf {0} & \text { otherwise } \end{array} \right. \tag {46}
$$

When $a = a^{*}$ , we then have

$$
\begin{array}{l} \boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) \\ = \left(\left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}\right) \otimes I _ {n}\right) \left(\boldsymbol {e} _ {a} \otimes \boldsymbol {e} _ {b}\right) - \left(\left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}\right) \otimes I _ {n}\right) \left(\boldsymbol {e} _ {a ^ {*}} \otimes \boldsymbol {e} _ {b ^ {*}}\right) \tag {47} \\ = \mathbf {1} _ {n} \otimes \left(\boldsymbol {e} _ {b} - \boldsymbol {e} _ {b ^ {*}}\right). \\ \end{array}
$$

Thus in summary

$$
(c) = \boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) = \left\{ \begin{array}{l l} \mathbf {1} _ {n} \otimes \left(\boldsymbol {e} _ {b} - \boldsymbol {e} _ {b ^ {*}}\right) & \text { if } a = a ^ {*} \\ \mathbf {0} & \text { otherwise. } \end{array} \right. \tag {48}
$$

(d) Now for the last term, note that

$$
\begin{array}{l} \boldsymbol {Q} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \boldsymbol {Q} \\ = \boldsymbol {Q} \operatorname{diag} \left(\boldsymbol {e} _ {b} \otimes \mathbf {1} _ {n}\right) \operatorname{diag} \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {a ^ {*}}\right) \boldsymbol {Q} \\ = Q \operatorname{diag} \left(e _ {b} \otimes e _ {a ^ {*}}\right) Q \\ = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \left(\operatorname{diag} \left(\boldsymbol {e} _ {b}\right) \otimes \operatorname{diag} \left(\boldsymbol {e} _ {a ^ {*}}\right)\right) \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \tag {49} \\ = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \left(\operatorname{diag} \left(\boldsymbol {e} _ {b}\right) \otimes \boldsymbol {e} _ {a ^ {*}}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \\ = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\boldsymbol {e} _ {b} ^ {\top} \otimes \boldsymbol {e} _ {a ^ {*}}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \\ = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \otimes 1\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \\ = \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right). \\ \end{array}
$$

Plugging in $(P_1^{(k)} - P_1^{*(k)})$ , we have that

$$
\begin{array}{l} \boldsymbol {Q} \operatorname{diag} \left(\boldsymbol {Q} \boldsymbol {P} _ {1} ^ {(k)}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {P} _ {1} ^ {* (k)}\right) \boldsymbol {Q} \left(\boldsymbol {P} _ {1} ^ {(k)} - \boldsymbol {P} _ {1} ^ {* (k)}\right) \\ = \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {a} \otimes \boldsymbol {e} _ {b}\right) - \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {a ^ {*}} \otimes \boldsymbol {e} _ {b ^ {*}}\right) \\ = \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \otimes \mathbf {1} _ {n}\right) (1 \otimes \boldsymbol {e} _ {b}) - \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \otimes \mathbf {1} _ {n}\right) (1 \otimes \boldsymbol {e} _ {b ^ {*}}) \\ = \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \otimes \mathbf {1} _ {n}\right) \left(\boldsymbol {e} _ {b} \otimes 1\right) - \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \otimes \mathbf {1} _ {n}\right) \left(\boldsymbol {e} _ {b ^ {*}} \otimes 1\right) \tag {50} \\ = \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \boldsymbol {e} _ {b} \otimes \mathbf {1} _ {n}\right) - \left(\boldsymbol {e} _ {a ^ {*}} \boldsymbol {e} _ {b} ^ {\top} \boldsymbol {e} _ {b ^ {*}} \otimes \mathbf {1} _ {n}\right) \\ = \left\{ \begin{array}{l l} \boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n} & \text { if } b \neq b ^ {*} \\ \boldsymbol {0} & \text { otherwise. } \end{array} \right. \\ \end{array}
$$

Summing the four terms together, we then have

$$
\left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {1}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right) = \left\{ \begin{array}{l l} 0 & \text { when } a ^ {*} = a, b ^ {*} = b \\ \mathbf {1} _ {n} \otimes (\boldsymbol {e} _ {b} - \boldsymbol {e} _ {b ^ {*}}) + \boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n} & \text { when } a ^ {*} = a, b ^ {*} \neq b \\ \mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}} + (\boldsymbol {e} _ {a} - \boldsymbol {e} _ {a ^ {*}}) \otimes \mathbf {1} _ {n} & \text { when } a ^ {*} \neq a, b ^ {*} = b \\ \mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b} + \boldsymbol {e} _ {a} \otimes \mathbf {1} _ {n} & \text { when } a ^ {*} \neq a, b ^ {*} \neq b. \end{array} \right. \tag {51}
$$

Now we are ready to provide the gradient expression for the loss over the entire sequence. Observe that for every consecutive sequence of $m$ columns $\{V_1^{*(j)}, V_1^{*(j+1)}, \ldots, V_1^{*(j+m-1)}\}$ that all equals to $\bar{e}_k$ , it will result in one incorrect column $V_3^{(j-1)}$ in case 2, one incorrect column $V_3^{(j+m-1)}$ in case 3, and $m-1$ incorrect columns $(V_3^{(j)}, \ldots, V_3^{(j+m-2)})$ in case 4.

$$
\begin{array}{l} \boldsymbol {V} _ {1} ^ {(j - 1)} \quad \rightarrow \quad \tilde {\boldsymbol {V}} _ {1} ^ {(j - 1)} (= \bar {\boldsymbol {e}} _ {p} \neq \bar {\boldsymbol {e}} _ {k}) \quad \xrightarrow {\boldsymbol {P} _ {1} ^ {(p)}} \quad \boldsymbol {V} _ {2} ^ {(j - 1)} \quad \rightarrow \quad \tilde {\boldsymbol {V}} _ {2} ^ {(j - 1)} \quad \xrightarrow {\boldsymbol {W} _ {2} ^ {*}} \quad \boldsymbol {V} _ {3} ^ {(j - 1)} (\text { case   2 }) \\ \nearrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \boldsymbol {V} _ {1} ^ {(j)} \quad \rightarrow \quad \tilde {\boldsymbol {V}} _ {1} ^ {(j)} (= \bar {\boldsymbol {e}} _ {k}) \quad \xrightarrow {\boldsymbol {P} _ {1} ^ {(k)}} \quad \boldsymbol {V} _ {2} ^ {(j)} \quad \rightarrow \quad \tilde {\boldsymbol {V}} _ {2} ^ {(j)} \quad \xrightarrow {\boldsymbol {W} _ {2} ^ {*}} \quad \boldsymbol {V} _ {3} ^ {(j)} (\text { case   4 }) \\ \nearrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \begin{array}{c c} \vdots & \vdots \\ \nearrow & \nearrow \end{array} \\ \boldsymbol {V} _ {1} ^ {(j + m - 2)} \rightarrow \tilde {\boldsymbol {V}} _ {1} ^ {(j + m - 2)} (= \bar {\boldsymbol {e}} _ {k}) \xrightarrow {\boldsymbol {P} _ {1} ^ {(k)}} \boldsymbol {V} _ {2} ^ {(j + m - 2)} \rightarrow \tilde {\boldsymbol {V}} _ {2} ^ {(j + m - 2)} \xrightarrow {\boldsymbol {W} _ {2} ^ {*}} \boldsymbol {V} _ {3} ^ {(j + m - 2)} (\text {case 4}) \\ \nearrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \boldsymbol {V} _ {1} ^ {(j + m - 1)} \rightarrow \tilde {\boldsymbol {V}} _ {1} ^ {(j + m - 1)} (= \bar {\boldsymbol {e}} _ {k}) \xrightarrow {\boldsymbol {P} _ {1} ^ {(k)}} \boldsymbol {V} _ {2} ^ {(j + m - 1)} \rightarrow \tilde {\boldsymbol {V}} _ {2} ^ {(j + m - 1)} \xrightarrow {\boldsymbol {W} _ {2} ^ {*}} \boldsymbol {V} _ {3} ^ {(j + m - 1)} (\text {case 3}) \\ \nearrow \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \\ \boldsymbol {V} _ {1} ^ {(j + m)} \quad \rightarrow \quad \tilde {\boldsymbol {V}} _ {1} ^ {(j + m)} (= \bar {\boldsymbol {e}} _ {q} \neq \bar {\boldsymbol {e}} _ {k}) \quad \stackrel {{P _ {1} ^ {(q)}}} {{\longrightarrow}} \quad \boldsymbol {V} _ {2} ^ {(j + m)} \quad \rightarrow \quad \tilde {\boldsymbol {V}} _ {2} ^ {(j + m)} \quad \stackrel {{W _ {2} ^ {*}}} {{\longrightarrow}} \quad \boldsymbol {V} _ {3} ^ {(j + m)} (\text {case 1 or 2}) \\ \end{array}
$$

Figure 12. Error propagation of $P_1^{(k)}$

For illustration, one can refer to the computation graph in Figure 12. In the graph, green entries agrees with the counterfactual values with correct $P_1^{*(k)}$ , Red and pink entries are incorrect entries where red entries are consequence solely dependent on $P_1^{(k)}$ (case 4) and pink entries depend on other correct columns (case 2,3).

Assume that in total there are $\alpha$ columns in $V_{3}$ under case 2, $\alpha$ columns in $V_{3}$ under case 3, and $\beta$ columns in $V_{3}$ under case 4, then by Equation (27) the total gradient can be expressed as follows:

\- When $a = a^{*}, b \neq b^{*}$ :

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1} ^ {(k)}} = 2 \alpha \left(\mathbf {1} _ {n} \otimes \left(\boldsymbol {e} _ {b} - \boldsymbol {e} _ {b ^ {*}}\right) + 2 \alpha \left(\boldsymbol {e} _ {a} - \boldsymbol {e} _ {a ^ {*}}\right) \otimes \mathbf {1} _ {n}\right) + 2 \beta \left(\mathbf {1} _ {n} \otimes \left(\boldsymbol {e} _ {b} - \boldsymbol {e} _ {b ^ {*}}\right) + \boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}\right) \tag {52}
$$

$$
= (2 \alpha + 2 \beta) (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b}) - (2 \alpha + 2 \beta) (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}) + 2 \beta (\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}).
$$

\- When $a \neq a^{*}, b = b^{*}$ :

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1} ^ {(k)}} = 2 \alpha \left(\mathbf {1} _ {n} \otimes \left(\boldsymbol {e} _ {b} - \boldsymbol {e} _ {b ^ {*}}\right)\right) + 2 \alpha \left(\left(\boldsymbol {e} _ {a} - \boldsymbol {e} _ {a ^ {*}}\right) \otimes \mathbf {1} _ {n}\right) + 2 \beta \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}} + \left(\boldsymbol {e} _ {a} - \boldsymbol {e} _ {a ^ {*}}\right) \otimes \mathbf {1} _ {n}\right) \tag {53}
$$

$$
= (2 \alpha + 2 \beta) (\boldsymbol {e} _ {a} \otimes \mathbf {1} _ {n}) - (2 \alpha + 2 \beta) (\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}) + 2 \beta (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}).
$$

\- When $a \neq a^{*}, b \neq b^{*}$ :

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1} ^ {(k)}} = 2 \alpha \left(\mathbf {1} _ {n} \otimes \left(\boldsymbol {e} _ {b} - \boldsymbol {e} _ {b ^ {*}}\right)\right) + 2 \alpha \left(\left(\boldsymbol {e} _ {a} - \boldsymbol {e} _ {a ^ {*}}\right) \otimes \mathbf {1} _ {n}\right) + 2 \beta \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b} + \boldsymbol {e} _ {a} \otimes \mathbf {1} _ {n}\right) \tag {54}
$$

$$
= (2 \alpha + 2 \beta) (\boldsymbol {e} _ {a} \otimes \mathbf {1} _ {n}) + (2 \alpha + 2 \beta) (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b}) - 2 \alpha (\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}) - 2 \alpha (\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}).
$$

This gives the desired expression of gradient//

![](images/92098d6c5923c5b45b9078e60df6f0f25791e97e025b87cd1a0e5f1f8bb1870d.jpg)

Now with the gradient expression, we are ready to prove for the learning of a single missing column.

# Lemma G.21 (Learning Column of $W_{1}$ ).

Fix an input sequence $V_{1} \in R^{n^{2} \times L}$ and any column index $k \in [n^{2}]$ , if $\text{HardMax}(C_{2} + W_{2}) = W_{2}^{*}$ and $\text{HardMax}(C_{1(k)} + W_{1})$ equals to $W_{1}^{*}$ everywhere except for the k-th column and if there exists a non-empty subset of indices $J \subset [L]$ such that $V_{1}^{*(j)} = \bar{e}_{k}$ for all $j \in J$ , for any initialization $\boldsymbol{W}_{1(0)}^{(k)} \in \mathbb{R}^{n^{2}}$ such that $\|W_{1(0)}^{(k)}\|_{0} \leq \frac{1}{2}$ . Taking two surrogate gradient updates on $\boldsymbol{W}_{1(0)}^{(k)}$ as described in Algorithm 3 gives $\boldsymbol{W}_{1(2)}^{(k)}$ such that $\text{HardMax}(\boldsymbol{W}_{1(2)}^{(k)}) = \boldsymbol{W}_{1}^{*(k)}$ .

# Proof of Lemma G.21.

Without loss of generality, assume at the initialization $P_{1(0)}^{(k)} = e_{a_{(0)}} \otimes e_{b_{(0)}}$ and $P_{1}^{*(k)} = e_{a^{*}} \otimes e_{b^{*}}$ for some $a_{(0)}, b_{(0)}, a^{*}, b^{*} \in [n]$ . We first note that the conditions specified in the lemma meets the assumptions required by Lemma G.20, namely there is only one incorrect column in $W_{1}$ missing and that column is being used in the forward pass at least one time (since J is non-empty).

To prove $\mathrm{HardMax}(\boldsymbol{W}_{1(2)}^{(k)}) = \boldsymbol{W}_1^{*(k)}$ , it is sufficient to show that $\arg \max (\boldsymbol{W}_{1(2)}^{(k)}) = a^{*}n + b^{*}$ .

Now we will leverage the gradient expressions in Lemma G.20. We will dive into three different cases:

\- When $a_{(0)} \neq a^{*}$ and $b_{(0)} \neq b^{*}$ .

By Lemma G.20 we have for some $\alpha \in \mathbb{Z}_+$ and $\beta \in \mathbb{N}$ that

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1 (0)} ^ {(k)}} = (2 \alpha + 2 \beta) (\boldsymbol {e} _ {a _ {(0)}} \otimes \boldsymbol {1} _ {n}) + (2 \alpha + 2 \beta) (\boldsymbol {1} _ {n} \otimes \boldsymbol {e} _ {b _ {(0)}}) - 2 \alpha (\boldsymbol {e} _ {a ^ {*}} \otimes \boldsymbol {1} _ {n}) - 2 \alpha (\boldsymbol {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}). \tag {55}
$$

It is not hard to verify that the unique smallest entry is of index $a^{*}n + b^{*}$ with value $-4\alpha$ . This entry is contributed by the intersection of $-2\alpha \left( e_{a^{*}} \otimes 1_{n} \right) - 2\alpha \left( 1_{n} \otimes e_{b^{*}} \right)$ , the remaining smaller entries are of value $-2\alpha$ contributed by non-intersecting entries in the same expression above.

Thus we know $\arg \min \partial \mathcal{L} / \partial P_{1(0)}^{(k)} = a^{*}n + b^{*}$ with a margin of at least $-2$ (since $\alpha \geq 1$ ).

Now we can apply the gradient step to the weight initialization. Since $\|\boldsymbol{W}_{1(0)}^{(k)}\|_{0} \leq \frac{1}{2}$ , the margin of $a^{*}n + b^{*}$ dominates the largest margin in the initialization (which is 1), we have

$$
\arg \max \left(\boldsymbol {W} _ {1 (1)} ^ {(k)}\right) = \arg \max \left(\boldsymbol {W} _ {1 (0)} ^ {(k)} - \frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1 (0)} ^ {(k)}}\right) = a ^ {*} n + b ^ {*}. \tag {56}
$$

Therefore $P_{1(1)}^{(k)} = P_1^{*(k)}$ with the first step. We will reach zero loss after the first step and hence the second step is static, so we have shown $\arg \max(\boldsymbol{W}_{1(2)}^{(k)}) = a^{*}n + b^{*}$ as desired.

\- When $a_{(0)} = a^{*}$ and $b_{(0)} \neq b^{*}$ .

By Lemma G.20 we have for some $\alpha \in \mathbb{Z}_{+}$ and $\beta \in \mathbb{N}$ that

$$
\frac {\partial \mathcal {L}}{\partial P _ {1 (0)} ^ {(k)}} = (2 \alpha + 2 \beta) \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b _ {(0)}}\right) - (2 \alpha + 2 \beta) \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}\right) + 2 \beta \left(\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}\right). \tag {57}
$$

In this case we no longer have an unique smallest entry, the set of negative entries is of index $nx + b^*$ where $x \in [n]$ . The values are $-2\alpha$ for the case of $x = a^*$ and $-2(\alpha + \beta)$ for other $x$ 's. These negative entries are contributed by the $-(2\alpha + 2\beta)(1_n \otimes e_{b^*})$ , and all other entries are at least 0.

Thus we also have a negative margin of at least -2 since $\alpha \geq 1$ . Therefore after taking the first gradient step, we know that there exists some $x \in [n]$ such that

$$
\arg \max \left(\boldsymbol {W} _ {1 (1)} ^ {(k)}\right) = \arg \max \left(\boldsymbol {W} _ {1 (0)} ^ {(k)} - \frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1 (0)} ^ {(k)}}\right) = n x + b ^ {*}. \tag {58}
$$

Let $P_{1(1)}^{(k)} = \arg \max(\boldsymbol{W}_{1(1)}^{(k)}) = \boldsymbol{e}_{a_{(1)}} \otimes \boldsymbol{e}_{b_{(1)}}$ , then now we are in the case of $b_{(1)} = b^*$ since the negative margin of -2 dominates any entry-wise difference in the initialization as $\| \boldsymbol{W}_{1(0)}^{(k)} \|_0 \leq \frac{1}{2}$ . If $\beta = 0$ and it happens that $a_{(1)} = a^*$ , then we have zero loss after the first step and we are done as the second step will be static. If $b_{(1)} \neq b^*$ , then by Lemma G.20 the second step gradient is of the form

$$
\frac {\partial \mathcal {L}}{\partial P _ {1 (1)} ^ {(k)}} = (2 \alpha + 2 \beta) \left(\boldsymbol {e} _ {a _ {(1)}} \otimes \mathbf {1} _ {n}\right) - (2 \alpha + 2 \beta) \left(\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}\right) + 2 \beta \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}\right). \tag {59}
$$

This is a bit tricky to analyze directly since we no longer have the small entry-wise difference from the initialization in the weights, but one may note that the sum of the two update steps is of the form

$$
\begin{array}{l} \frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1 (1)} ^ {(k)}} + \frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1 (0)} ^ {(k)}} \\ = (2 \alpha + 2 \beta) \left(\boldsymbol {e} _ {a _ {(1)}} \otimes \mathbf {1} _ {n}\right) - (2 \alpha + 2 \beta) \left(\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}\right) + 2 \beta \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}\right) \tag {60} \\ + (2 \alpha + 2 \beta) \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b _ {(0)}}\right) - (2 \alpha + 2 \beta) \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}\right) + 2 \beta \left(\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}\right) \\ = (2 \alpha + 2 \beta) \left(\boldsymbol {e} _ {a _ {(1)}} \otimes \mathbf {1} _ {n}\right) + (2 \alpha + 2 \beta) \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b _ {(0)}}\right) - 2 \alpha \left(\boldsymbol {e} _ {a ^ {*}} \otimes \mathbf {1} _ {n}\right) - 2 \alpha \left(\mathbf {1} _ {n} \otimes \boldsymbol {e} _ {b ^ {*}}\right). \\ \end{array}
$$

This is identical to the single step gradient as in Equation (55) in the first case, so follow the identical argument we have

$$
\arg \max \left(\boldsymbol {W} _ {1 (2)} ^ {(k)}\right) = \arg \max \left(\boldsymbol {W} _ {1 (0)} ^ {(k)} - \left(\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1 (0)} ^ {(k)}} + \frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {1 (1)} ^ {(k)}}\right)\right) = a ^ {*} n + b ^ {*} \tag {61}
$$

as desired.

\- When $a_{(0)} \neq a^{*}$ and $b_{(0)} = b^{*}$

Note that all expressions are symmetric with respect to $a$ and $b$ with an additional swap of the Kronecker products, so we may follow the exact same argument as in the case of $a_{(0)} = a^*$ and $b_{(0)} \neq b^*$ and arrive at the same conclusion.

Thus for any initializations, after two surrogate gradient steps on $\boldsymbol{W}_{1}^{(k)}$ , we have $\text{HardMax}(\boldsymbol{W}_{1(2)}^{(k)}) = \boldsymbol{W}_{1}^{*(k)}$ .

# G.3.2. LEARNING THE SECOND LAYER

Now for the second layer, we will similarly first derive the gradient (which is much simpler) and show that with only one gradient step, one can learn the correct entry.

Lemma G.22 (Gradient with respect to incorrect column in $P_{2}$ ).

When only the $k$ -th column of translation matrix $\pmb{P}_2^{(k)}$ is not equal to $\pmb{P}_2^{*(k)}$ . If $\pmb{P}_2^{(k)}$ is used in the forward pass, there exists $\alpha \in \mathbb{Z}_+$ such that the gradient of $\mathcal{L}$ with respect to $\pmb{P}_2^{(k)}$ is of the form

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {2} ^ {(k)}} = 2 \alpha \left(\boldsymbol {P} _ {2} ^ {(k)} - \boldsymbol {P} _ {2} ^ {* (k)}\right).
$$

Proof. We follow the same set of notations as used in the proof for Lemma G.20. In particular, we use $\tilde{V}_{1}, V_{2}, \tilde{V}_{2}$ and $V_{3}$ to denote the intermediate sequences attained with translation matrix $P_{2}^{(k)}$ and $\tilde{V}_{1}^{*}, V_{2}^{*}, \tilde{V}_{2}^{*}$ and $V_{3}^{*}$ to denote the counterfactual intermediate sequences should the forward pass is done with the ground truth translations $P_{2}^{*(k)} = W_{1}^{*}$ . Since we assume that $P_{1} = P_{1}^{*}$ , we have $\tilde{V}_{2} = \tilde{V}_{2}^{*}$ . Therefore for all column j such that $V_{3}^{(j)} \neq V_{3}^{*(j)}$ , it must be so that $\tilde{V}_{2}^{*(j)} = \bar{e}_{k}$ , and the residual can be written as

$$
\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)} = \boldsymbol {P} _ {2} \tilde {\boldsymbol {V}} _ {2} ^ {* (j)} - \boldsymbol {P} _ {2} ^ {*} \tilde {\boldsymbol {V}} _ {2} ^ {* (j)} = \boldsymbol {P} _ {2} \bar {\boldsymbol {e}} _ {k} - \boldsymbol {P} _ {2} ^ {*} \bar {\boldsymbol {e}} _ {k} = \boldsymbol {P} _ {2} ^ {(k)} - \boldsymbol {P} _ {2} ^ {* (k)}. \tag {62}
$$

Assume that $P_{2}^{(k)}$ has been used $\alpha$ times in the forward pass, follow the chain rule we then have

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {2} ^ {(k)}} = 2 \sum_ {j = 1} ^ {L} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {2} ^ {(k)}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right) = 2 \alpha \left(\boldsymbol {P} _ {2} ^ {(k)} - \boldsymbol {P} _ {2} ^ {* (k)}\right). \tag {63}
$$

![](images/3c792d96c3bff0d6fa76a167df8a9b32482f7cbc3ae66c113c43e3f24902193f.jpg)

# Lemma G.23 (Learning $W_{2}$ ).

Fix an input sequence $V_{1} \in R^{n^{2} \times L}$ and any column index $k \in [n^{2}]$ , if $\text{HardMax}(C_{1} + W_{1}) = W_{1}^{*}$ and $\text{HardMax}(C_{2(k)} + W_{2})$ equals to $W_{2}^{*}$ everywhere except for the k-th column and if there exists a non-empty subset of indices $J \subset [L]$ such that $V_{2}^{*(j)} = \bar{e}_{k}$ for all $j \in J$ , for any initialization $\boldsymbol{W}_{2(0)}^{(k)} \in \mathbb{R}^{n^{2}}$ such that $\|W_{2(0)}^{(k)}\|_{0} \leq \frac{1}{2}$ . Taking one surrogate gradient updates on $\boldsymbol{W}_{2(0)}^{(k)}$ as described in Algorithm 3 gives $\boldsymbol{W}_{2(1)}^{(k)}$ such that $\text{HardMax}(\boldsymbol{W}_{2(1)}^{(k)}) = \boldsymbol{W}_{2}^{*(k)}$ .

Proof. Without loss of generality, assume at the initialization $P_{2(0)}^{(k)} = e_{a_{(0)}} \otimes e_{b_{(0)}}$ and $P_{2}^{*(k)} = e_{a^{*}} \otimes e_{b^{*}}$ for some $a_{(0)}, b_{(0)}, a^{*}, b^{*} \in [n]$ . We first note that the conditions specified in the lemma meets the assumptions required by Lemma G.22, namely there is only one incorrect column in $W_{2}$ missing and that column is being used in the forward pass at least one time (since J is non-empty). To prove $\text{HardMax}(W_{2(1)}^{(k)}) = W_{2}^{*(k)}$ , it is sufficient to show that $\arg\max(W_{2(1)}^{(k)}) = a^{*}n + b^{*}$ . By Lemma G.22, we have

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {2} ^ {(k)}} = 2 \sum_ {j = 1} ^ {L} \left(\frac {\partial \boldsymbol {V} _ {3} ^ {(j)}}{\partial \boldsymbol {P} _ {2} ^ {(k)}}\right) ^ {\top} \left(\boldsymbol {V} _ {3} ^ {(j)} - \boldsymbol {V} _ {3} ^ {* (j)}\right) = 2 \alpha \left(\boldsymbol {P} _ {2} ^ {(k)} - \boldsymbol {P} _ {2} ^ {* (k)}\right) = 2 \alpha \boldsymbol {e} _ {a _ {(0)}} \otimes \boldsymbol {e} _ {b _ {(0)}} - 2 \alpha \boldsymbol {e} _ {a ^ {*}} \otimes \boldsymbol {e} _ {b ^ {*}}. \tag {64}
$$

Since $\alpha \geq 1$ , the negative margin of the $a^{*}n + b^{*}$ -th entry dominates the initial difference in the initialization which is bounded by $\| W_{2(0)}^{(k)}\| _0\leq \frac{1}{2}$ . Thus we have

$$
\arg \max \left(\boldsymbol {W} _ {2 (1)} ^ {(k)}\right) = \arg \max \left(\boldsymbol {W} _ {2 (0)} ^ {(k)} - \frac {\partial \mathcal {L}}{\partial \boldsymbol {P} _ {2 (0)} ^ {(k)}}\right) = a ^ {*} n + b ^ {*} \tag {65}
$$

as desired.

![](images/4e0160e4cfcf8488259e5ad78e796918d1014e7827acb76e2baf8f241aedb836.jpg)

# G.3.3. PROOF FOR THEOREM G.24

Now we are ready to prove for the main theorem restated below:

Theorem G.24 (Learning $\Pi^{*}$ with context-enhanced surrogate GD with $\Pi^{*}$ -coverable input).

For any initialization $\boldsymbol{W}_{1(0)}$ , $\boldsymbol{W}_{2(0)} \in \mathbb{R}^{n^{2} \times n^{2}}$ such that $\|W_{1(0)}\|_{0} \leq \frac{1}{2}$ and $\|W_{2(0)}\|_{0} \leq \frac{1}{2}$ , for any target set of phrasebooks $\Pi^{*} = \{\pi_{1}^{*}, \pi_{2}^{*}\}$ in MLT(2, n), given an $\Pi^{*}$ -coverable input $V_{1}$ and the corresponding ground truth label $\boldsymbol{V}_{3}^{*} = \text{MLT}_{\Pi^{*}}(\boldsymbol{V}_{1})$ , Algorithm 2 terminates with HardMax ( $W_{1}$ ) = $W_{1}^{*}$ and HardMax ( $W_{2}$ ) = $W_{2}^{*}$ .

Proof. The statement can be proven by a similar induction as in the proof for Theorem G.16.

Let the induction hypothesis be such that when the enumeration goes to k-th column of the i-th layer, if $\operatorname{HardMax}(\boldsymbol{W}_{l}^{(j)} + \boldsymbol{W}_{l}^{*(j)}) = \boldsymbol{W}_{l}^{*(j)}$ for all $(l, j) \in [2] \times [n^{2}]$ and $\operatorname{HardMax}\left(\boldsymbol{W}_{l}^{(j)}\right) = \boldsymbol{W}_{l}^{*(j)}$ for all $(l, j)$ such that l < i or $l = i \wedge j < k$ , then the gradient update on the k-th column of the i-th layer ends with $\boldsymbol{W}_{i}^{(k)} = \boldsymbol{W}_{i}^{*(k)}$ while $\operatorname{HardMax}(\boldsymbol{W}_{l}^{(j)} + \boldsymbol{W}_{l}^{*(j)}) = \boldsymbol{W}_{l}^{*(j)}$ for all $(l, j) \in [2] \times [n^{2}]$ is preserved.

The base case is satisfied as with initialization of $\|W_{1(0)}\|_{0} \leq \frac{1}{2}$ and $\|W_{2(0)}\|_{0} \leq \frac{1}{2}$ , by Lemma G.11, we have $\text{HardMax}(\boldsymbol{W}_{l}^{(j)} + \boldsymbol{W}_{l}^{*(j)}) = \text{HardMax}(\boldsymbol{0} + \boldsymbol{W}_{l}^{*(j)}) = \boldsymbol{W}_{l}^{*(j)}$ for all $(l,j) \in [d] \times [n^{2}]$ and there are no requirements for $\boldsymbol{W}_{i}^{(k)} = \boldsymbol{W}_{i}^{*(k)}$ yet.

For the induction step, we note that with the inductive hypothesis of $\mathrm{HardMax}(\boldsymbol{W}_l^{(j)} + \boldsymbol{W}_l^{*(j)}) = \boldsymbol{W}_l^{*(j)}$ for all $(l,j)\in [2]\times [n^2],(\boldsymbol{C}_{1(k)},\boldsymbol{W}_2^*)$ (when $i = 1$ ) or $(\boldsymbol{W}_1^*,\boldsymbol{C}_{2(k)})$ (when $i = 2$ ) will correctly condition all columns of $\pmb{P}$ 's except for the $P_{i}^{(k)}$ since

$$
\boldsymbol {C} _ {i (k)} ^ {(k)} = \boldsymbol {W} _ {i} ^ {*} \left(\boldsymbol {I} _ {n ^ {2}} - \mathrm{diag} (\bar {\boldsymbol {e}} _ {k})\right) ^ {(k)} = \mathbf {0}. \tag {66}
$$

Thus by Lemma G.21 (when $i = 1$ ) or Lemma G.23 (when $i = 2$ ), we know that after updating $W_{i}^{(k)}$ , we have $\mathrm{HardMax}(\boldsymbol{W}_i^{(k)}) = \boldsymbol{W}_i^{*(k)}$ . The newly added column provides the correct inductive hypothesis on $\mathrm{HardMax}(\boldsymbol{W}_l^{(j)}) = \boldsymbol{W}_l^{*(j)}$ for the next enumeration step.

By induction to i = 2 and $k = n^{2}$ , we will be able to recover $\text{HardMax}(\boldsymbol{W}_{i}) = \boldsymbol{W}_{i}^{*}$ for all $i \in [d]$ .

Similarly we may use the coupon collecting argument to generalize the input to uniformly random strings as follows:

Corollary G.25 (Learning $\Pi^{*}$ in MLT(2, n) with context-enhanced surrogate GD with random input).

For any initialization $\boldsymbol{W}_{1(0)},\boldsymbol{W}_{2(0)}\in\mathbb{R}^{n^{2}\times n^{2}}$ such that $\|W_{1(0)}\|_{0}\leq\frac{1}{2}$ and $\|W_{2(0)}\|_{0}\leq\frac{1}{2}$ , for any target set of phrasebooks $\Pi^{*}=\{\pi_{1}^{*},\pi_{2}^{*}\}$ in MLT(2,n), with probability at least $1-\delta$ over a uniformly random input $V_{1}$ of length $L=2n^{2}\log\frac{2n}{\delta}$ , Algorithm 2 provided with the ground truth label $V_{3}^{*}=MLT_{\Pi^{*}}(V_{1})$ terminates with $\operatorname{HardMax}(W_{1})=W_{1}^{*}$ and $\operatorname{HardMax}(W_{2})=W_{2}^{*}$ .

# G.4. Auxiliary Lemmas for Learning Surrogate Models

Lemma G.26. For any long one-hot vector $\pmb{v} = \pmb{e}_a\otimes \pmb{e}_b$ , with $Q = (I_n\otimes 1_n)(1_n\otimes I_n)^\top$ we have

$$
\boldsymbol {Q} \boldsymbol {v} = \boldsymbol {e} _ {b} \otimes \mathbf {1} _ {n}; \quad \boldsymbol {Q} ^ {\top} \boldsymbol {v} = \mathbf {1} _ {n} \otimes \boldsymbol {e} _ {a}. \tag {67}
$$

Proof. Note that with $\pmb{v} = \pmb{e}_a \otimes \pmb{e}_b$ , we have

$$
\boldsymbol {Q} \boldsymbol {v} = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {a} \otimes \boldsymbol {e} _ {b}\right) = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(1 \otimes \boldsymbol {e} _ {b}\right) = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\boldsymbol {e} _ {b} \otimes 1\right) = \boldsymbol {e} _ {b} \otimes \mathbf {1} _ {n};
$$

$$
\boldsymbol {Q} ^ {\top} \boldsymbol {v} = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \left(\boldsymbol {e} _ {a} \otimes \boldsymbol {e} _ {b}\right) = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(\boldsymbol {e} _ {a} \otimes 1\right) = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(1 \otimes \boldsymbol {e} _ {a}\right) = \mathbf {1} _ {n} \otimes \boldsymbol {e} _ {a}.
$$

![](images/b075e9209c9eb62319079ecc5b15dbf1fea564992271abcf33d30943ee0fc7de.jpg)

Lemma G.27. For any one-hot vector $\pmb{v} = \pmb{e}_a\otimes \pmb{e}_b\in \mathbb{R}^{n^2}$ ,

$$
\boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {v}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {v}\right) \boldsymbol {Q} = \left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}\right) \otimes I _ {n}.
$$

Proof. By Lemma G.26, $\pmb{Q}^{\top}\pmb{v} = \mathbf{1}_n\otimes \pmb{e}_a$ . Therefore $\mathrm{diag}\left(\pmb{Q}^{\top}\pmb{v}\right) = \mathrm{diag}\left(\mathbf{1}_n\right)\otimes \mathrm{diag}\left(\pmb{e}_a\right) = I_n\otimes \mathrm{diag}\left(\pmb{e}_a\right)$ . Thus

$$
\operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {v}\right) \boldsymbol {Q} = \left(I _ {n} \otimes \operatorname{diag} \left(\boldsymbol {e} _ {a}\right)\right) \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) = \left(I _ {n} \otimes \boldsymbol {e} _ {a}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \tag {69}
$$

and therefore we have

$$
\begin{array}{l} \boldsymbol {Q} ^ {\top} \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {v}\right) \operatorname{diag} \left(\boldsymbol {Q} ^ {\top} \boldsymbol {v}\right) \boldsymbol {Q} = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} \otimes \boldsymbol {e} _ {a} ^ {\top}\right) \left(I _ {n} \otimes \boldsymbol {e} _ {a}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) (I _ {n} \otimes 1) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \quad \left(\text { since } e _ {a} ^ {\top} e _ {a} = 1\right) \\ = \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(1 \otimes I _ {n}\right) \left(\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}\right) \tag {70} \\ = (\mathbf {1} _ {n} \otimes I _ {n}) (\mathbf {1} _ {n} ^ {\top} \otimes I _ {n} ^ {\top}) \\ = \left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}\right) \otimes I _ {n} \\ \end{array}
$$

Lemma G.28. For any one-hot vector $\pmb{v} = \pmb{e}_a\otimes \pmb{e}_b\in \mathbb{R}^{n^2}$ ,

$$
\boldsymbol {Q} \operatorname{diag} (\boldsymbol {Q} \boldsymbol {v}) \operatorname{diag} (\boldsymbol {Q} \boldsymbol {v}) \boldsymbol {Q} ^ {\top} = I _ {n} \otimes \left(\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}\right).
$$

Proof. This proof is very similar to the proof for Lemma G.27. By Lemma G.26, $Q^{\top}v = 1_{n} \otimes e_{a}$ . Therefore $\text{diag}(Qv) = \text{diag}(e_{b}) \otimes \text{diag}(1_{n}) = \text{diag}(e_{b}) \otimes I_{n}$ . Thus

$$
\operatorname{diag} (\boldsymbol {Q} \boldsymbol {v}) \boldsymbol {Q} ^ {\top} = \left(\operatorname{diag} \left(\boldsymbol {e} _ {b}\right) \otimes I _ {n}\right) \left(\mathbf {1} _ {n} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) = \left(\boldsymbol {e} _ {b} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \tag {71}
$$

and therefore we have

$$
\begin{array}{l} \boldsymbol {Q} \operatorname{diag} (\boldsymbol {Q} \boldsymbol {v}) \operatorname{diag} (\boldsymbol {Q} \boldsymbol {v}) \boldsymbol {Q} ^ {\top} = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(\boldsymbol {e} _ {b} ^ {\top} \otimes I _ {n}\right) \left(\boldsymbol {e} _ {b} \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \\ = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(1 \otimes I _ {n}\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \quad \left(\text { since } e _ {b} ^ {\top} e _ {b} = 1\right) \\ = \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \left(I _ {n} \otimes 1\right) \left(I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}\right) \tag {72} \\ = (I _ {n} \otimes \mathbf {1} _ {n}) (I _ {n} ^ {\top} \otimes \mathbf {1} _ {n} ^ {\top}) \\ = I _ {n} \otimes (\mathbf {1} _ {n} \mathbf {1} _ {n} ^ {\top}). \\ \end{array}
$$

![](images/9deefea3175001c6eb6da11089ff595fde663abac2724a02025928ea5dde36bc.jpg)

Lemma G.29 (Tail Bound for Coupon Collector Problem (Motwani, 1995)).

For a set $S$ of size $n$ , with probability at least $1 - \delta$ one can cover all unique elements of $S$ in $n \log \frac{n}{\delta}$ independent uniformly random sampling trials from $S$ .

# G.5. Learning $\Pi^{*}$ in $\mathrm{MLT}_{\Pi^{*}}$ with Gradient Descent (Empirical Evidence)

In this section we provide more details on empirically optimizing the simple surrogate model SURR-MLT $_{ \{ W_{i} \}_{i=1}^{d} }$ , which was only briefly discussed in the main text by the end of Section 5.1. We will first introduce the approximations we made to the surrogate model to make gradient-based optimization easy and stable, then we will present empirical results on the model learning target sets of phrasebooks MLT $_{\Pi^{*}}$ in MLT(5, 10), MLT(10, 10), and even MLT(20, 10).

# G.5.1. APPROXIMATED LATENT MODEL FOR GD

Recall that with input sequence represented by $\pmb{V}_1 \in \mathbb{R}^{n^2 \times L}$ , the surrogate model for a depth- $d$ translation is being recursively defined by the translation + shifting operations

$$
\boldsymbol {V} _ {i + 1} = \operatorname{HardMax} \left(\boldsymbol {C} _ {i} + \boldsymbol {W} _ {i}\right) \text { Shift } (\boldsymbol {V} _ {i}) \tag {73}
$$

until we reach $V_{d+1}$ . While this model captures the essence of transition from ICL capability to memorization of specific set of phrasebooks, HardMax is making it not directly differentiable and hard to optimize. To address this issue, we approximate it with an column-wise softmax function with very low temperature (T = 1/25). The recursive definition in the approximated model is then

$$
\tilde {\boldsymbol {V}} _ {i + 1} = \operatorname{SoftMax} \left(2 5 \left(\boldsymbol {C} _ {i} + \boldsymbol {W} _ {i}\right)\right) \operatorname{Shift} \left(\tilde {\boldsymbol {V}} _ {i}\right). \tag {74}
$$

We denote the recursive surrogate model with the softmax substitution as $\mathrm{SURR - MLT}_{\{\boldsymbol{W}_i\}_{i = 1}^d}\left(\boldsymbol{C}_1,\boldsymbol{C}_2,\dots ,\boldsymbol{C}_d,\boldsymbol{V}_1\right)$ where

$$
\begin{array}{l} \tilde {\boldsymbol {V}} _ {d + 1} = \text { SurrR - MLT } _ {\{\boldsymbol {W} _ {i} \} _ {i = 1} ^ {d}} (\boldsymbol {C} _ {1}, \boldsymbol {C} _ {2}, \ldots , \boldsymbol {C} _ {d}, \boldsymbol {V} _ {1}) \\ \triangleq \operatorname{SoftMax} \left(2 5 \boldsymbol {C} _ {d} + 2 5 \boldsymbol {W} _ {d}\right) \tag {75} \\ \text { Shift } \left(\text { SoftMax } (2 5 \boldsymbol {C} _ {d - 1} + 2 5 \boldsymbol {W} _ {d - 1}) \text { Shift } \left(\dots \text { SoftMax } (2 5 \boldsymbol {C} _ {1} + 2 5 \boldsymbol {W} _ {1}) \text { Shift } (\tilde {\boldsymbol {V}} _ {1}) \dots\right)\right). \\ \end{array}
$$

We define the objective function as the column-wise cross-entropy loss between the final output and the input. Namely for input $V_{1}$ with prediction $\tilde{V}_{d+1}$ and ground truth label $V_{d+1}^{*}$ , the loss is computed as

$$
\mathcal {L} = \sum_ {k = 1} ^ {L} \text { CrossEntropy } (\tilde {\boldsymbol {V}} _ {d + 1} ^ {(k)}, \boldsymbol {V} _ {d + 1} ^ {* (k)}). \tag {76}
$$

We follow the same masking (dropout) curriculum as described in Appendix G.2 and Appendix G.3, that at each step we zero-out a single column from a single context matrix $C_{i}$ . We experiment on two gradient update schemes:

- Layer-wise Training: at each step, if we are masking a column on $C_i$ , we only compute the gradient with respect to $W_i$ and update it. This training is more akin to the theoretical analysis described in Appendix G.3.   
- Full Parameter Training: at any step, we compute the gradient with respect to each of the weight matrices and update all parameters. This is more akin to the real gradient-based training as we do not have the heuristics for localized update.

To allow for fast and stable training, we adopt a very large learning rate of $\eta = 100$ and apply parameter clipping between [0, 1] after each update. The complete algorithm is described as follows:

Algorithm 4 Layerwise Gradient Descent with Context-Enhanced Learning For Optimizing SURR-MLT   
1: Input:
2: input $V_{1} \in R^{n^{2} \times L}$ , label $V_{d+1}^{*} \in R^{n^{2} \times L}$ , descriptive text $W_{1}^{*}, \ldots, W_{d}^{*} \in R^{n^{2} \times n^{2}}$ , learning rate $\eta$ , total steps T
3:
4: Initialize $W_{1}, \ldots, W_{d} \leftarrow 0$ # Start with zero initialization
5: for t = 1 to T do
6: $i \leftarrow \lfloor (t - 1)/n^{2} \rfloor \% d + 1$ # Get the layer to be masked
7: $k \leftarrow ((t - 1) \% n^{2}) + 1$ # Get the column index to be masked
8: Initialize $C_{i(k)} \leftarrow W_{i}^{*}(I_{n^{2}} - \text{diag}(\bar{e}_{k}))$ # Create masked context matrix
9: $\tilde{V}_{d+1} \leftarrow \text{SURR-MLT}_{\{W_{i}\}_{i=1}^{d}}(W_{1}^{*} \ldots, W_{i-1}^{*}, C_{i(k)}, W_{i+1}^{*} \ldots, W_{d}^{*}, V_{1})$ 10: $L \leftarrow \text{CrossEntropy}(\tilde{V}_{d+1}, V_{d+1}^{*})$ 11: $W_{i} \leftarrow W_{i} - \eta \nabla_{W_{i}} L$ # Update the weight for the layer with mask
12: end for
13: Return $W_{1}, \ldots, W_{d}$ .

Algorithm 5 Full Parameter Gradient Descent with Context-Enhanced Learning For Optimizing SURR-MLT   
1: Input:
2: input $V_{1} \in R^{n^{2} \times L}$ , label $V_{d+1}^{*} \in R^{n^{2} \times L}$ , descriptive text $W_{1}^{*}, \ldots, W_{d}^{*} \in R^{n^{2} \times n^{2}}$ , learning rate $\eta$ , total steps T
3:
4: Initialize $W_{1}, \ldots, W_{d} \leftarrow 0$ # Start with zero initialization
5: for t = 1 to T do
6: $i \leftarrow \lfloor (t - 1)/n^{2} \rfloor \% d + 1$ # Get the layer to be masked
7: $k \leftarrow ((t - 1) \% n^{2}) + 1$ # Get the column index to be masked
8: Initialize $C_{i(k)} \leftarrow W_{i}^{*}(I_{n^{2}} - \text{diag}(\bar{e}_{k}))$ # Create masked context matrix
9: $\tilde{V}_{d+1} \leftarrow \text{SURR-MLT}_{\{W_{i}\}_{i=1}^{d}}(W_{1}^{*} \ldots, W_{i-1}^{*}, C_{i(k)}, W_{i+1}^{*} \ldots, W_{d}^{*}, V_{1})$ 10: $L \leftarrow CrossEntropy(\tilde{V}_{d+1}, V_{d+1}^{*})$ 11: for l = 1 to d do
12: $W_{l} \leftarrow W_{l} - \eta \nabla_{W_{l}} L$ # Update the weight for all layers
13: end for
14: end for
15: Return $W_{1}, \ldots, W_{d}$ .

![](images/882217d878b530c3c38a626b6fc44cd7450e687bec24ebeeacc3f13d73857040.jpg)

<details>
<summary>line</summary>

| Steps | Layer-wise GD |
| ----- | ------------- |
| 0     | 0.0           |
| 500   | 0.2           |
| 1000  | 0.3           |
| 1500  | 0.6           |
| 2000  | 0.8           |
| 2500  | 0.9           |
| 3000  | 1.0           |
| 3500  | 1.0           |
| 4000  | 1.0           |
| 4500  | 1.0           |
</details>

![](images/72d91db81945588577bf5e2e1508000e2a83138c70a26f1fb36e04fa382d3bfd.jpg)

<details>
<summary>line</summary>

| Steps | Series 1 | Series 2 | Series 3 | Series 4 |
| ----- | -------- | -------- | -------- | -------- |
| 0     | 0.0      | 0.0      | 0.0      | 0.0      |
| 500   | 0.1      | 0.1      | 0.1      | 0.1      |
| 1000  | 0.2      | 0.2      | 0.2      | 0.2      |
| 1500  | 0.3      | 0.3      | 0.3      | 0.3      |
| 2000  | 0.5      | 0.5      | 0.5      | 0.5      |
| 2500  | 0.7      | 0.7      | 0.7      | 0.7      |
| 3000  | 0.8      | 0.8      | 0.8      | 0.8      |
| 3500  | 0.9      | 0.9      | 0.9      | 0.9      |
| 4000  | 1.0      | 1.0      | 1.0      | 1.0      |
| 4500  | 1.0      | 1.0      | 1.0      | 1.0      |
</details>

$W_{1}$ $W_{2}$ $W_{3}$ $W_{4}$ $W_{5}$

(a) MLT(5, 10)   
![](images/5bc679cba014f1e841591f8741aecd7ff84a8e5af696b755e268c6ff374ad163.jpg)

<details>
<summary>line</summary>

| Steps | Layer-wise GD |
| ----- | ------------- |
| 0     | 0.0           |
| 2500  | 0.6           |
| 5000  | 0.9           |
| 7500  | 1.0           |
| 10000 | 1.0           |
</details>

![](images/8a39915be58900944f649eb4e38b2b77f3bdc667b0b9ed495c9a25062f97138f.jpg)

<details>
<summary>line</summary>

| Steps | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
|-------|--------|--------|--------|--------|--------|
| 0     | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    |
| 2500  | 0.8    | 0.6    | 0.4    | 0.3    | 0.2    |
| 5000  | 1.0    | 0.9    | 0.8    | 0.7    | 0.6    |
| 7500  | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    |
| 10000 | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    |
</details>

W1 W2 W3 W4 W5 W6 W7 W8 W9 W10

(b) MLT(10, 10)   
![](images/32506c3ade62a70b4e3ca025ff1361e94edd907fb863d8359fa9c3f90f6a6500.jpg)

<details>
<summary>line</summary>

| Steps | Layer-wise GD |
| ----- | ------------- |
| 0     | 0.0           |
| 2500  | 0.2           |
| 5000  | 0.4           |
| 7500  | 0.6           |
| 10000 | 0.8           |
| 12500 | 0.9           |
| 15000 | 1.0           |
| 17500 | 1.0           |
| 20000 | 1.0           |
</details>

W1
W2
W3
W4
W5
W6
W7
W8
W9
W10
W11
W12
W13
W14
W15
W16
W17
W18
W19
W20

![](images/7044c2a2f5d7c1076d026119c654da0e058e599e01da6217e2f9e45541fcef69.jpg)

<details>
<summary>line</summary>

| Steps | Value |
| ----- | ----- |
| 0     | 0.0   |
| 2500  | 0.3   |
| 5000  | 0.6   |
| 7500  | 0.8   |
| 10000 | 1.0   |
| 12500 | 1.0   |
| 15000 | 1.0   |
| 17500 | 1.0   |
| 20000 | 1.0   |
</details>

(c) MLT(20, 10)   
Figure 13. We perform Layer-wise and Full Parameter Training with Gradient Descent (defined in Appendix G.5.1) on the SURR-MLT (Definition G.7) designed for $\mathbf{MLT}_{\Pi^*}$ in $\mathbf{MLT}(5,10),\mathbf{MLT}(10,10),\mathbf{MLT}(20,10)$ respectively (alternately, depth $d = 5,10,20$ respectively, while number of characters is fixed at $n = 10$ ). Here, we report the portion of columns from the trainable parameters, which after HardMax application $\{\mathrm{HardMax}(\boldsymbol{W}_i)\}_{i=1}^d$ , align with the corresponding stochastic matrices of the phrasebooks $\{\mathrm{Matrix}(\pi_i^*)\}_{i=1}^d$ . We observe that under both algorithms, the trainable parameters quickly learn the relevant stochastic matrices.

![](images/7caf1c22ba9f649a1e42d7350b3c96375d9f63cfe56056fbaa3c8921a8f1eb57.jpg)  
Figure 14. Detailed analysis on the Full parameter training behavior of the trainable parameters in SURR-MLT from Figure 13 for $MLT_{\Pi^{*}}$ in $MLT(10,10)$ (i.e. depth d=10 and number of characters n=10). We report the behavior of all odd-index parameters $W_{1}, W_{3}, \cdots, W_{9}$ . (left to right) first, we show the number of columns of the trainable parameter, which after $HardMax(W_{i})$ align with the corresponding columns of $Matrix(\pi_{i}^{*})$ . Second, third and fourth visualize the matrices, $W_{i}$ , $HardMax(W_{i})$ , and $Matrix(\pi_{i}^{*})$ respectively. All matrices learn to match $Matrix(\pi_{i}^{*})$ at the end of training with HardMax operation.

# H. Construction of a Transformer that can Simulate the Latent Model

# H.1. Useful definitions and lemmas

Definition H.1 (Relative self-attention with 1 head). For a set of matrices $\{W_{query}, W_{key}, W_{value}\}$ with each matrix $\in R^{k \times k}$ for some k > 0 and a set of $(t + 1)$ biases $\{b_i\}_{t \leq i \leq 0}$ , the self-attention computation on an input sequence $x_1, \cdots, x_L$ with each $x_{p_2} \in R^k$ is given by the output sequence $y_1, \cdots, y_L$ , where for all $p_1 \in [1, L]$

$$
\boldsymbol {y} _ {p _ {1}} = \sum_ {p _ {2} = 1} ^ {L} a _ {p _ {1}, p _ {2}} \boldsymbol {o} _ {j}, \text {   where   } a _ {p _ {1} p _ {2}} = \frac {e ^ {\boldsymbol {q} _ {p _ {1}} ^ {\top} \boldsymbol {k} _ {p _ {2}} + b _ {p _ {2} - p _ {1}}}}{\sum_ {p _ {2} ^ {\prime} \leq p _ {1}} e ^ {\boldsymbol {q} _ {p _ {1}} ^ {\top} \boldsymbol {k} _ {p _ {2} ^ {\prime}} + b _ {p _ {2} ^ {\prime} - p _ {1}}}} \text {   if   } p _ {2} \leq p _ {1}, 0 \text {   otherwise }
$$

$$
\boldsymbol {q} _ {p _ {2}} = \boldsymbol {W} _ {\text { query }} \boldsymbol {x} _ {p _ {2}}, \quad \boldsymbol {k} _ {p _ {2}} = \boldsymbol {W} _ {\text { key }} \boldsymbol {x} _ {p _ {2}}, \quad \boldsymbol {o} _ {p _ {2}} = \boldsymbol {W} _ {\text { value }} \boldsymbol {x} _ {p _ {2}}, \quad \text { for   all } p _ {2} \in [ 1, L ].
$$

For a relative self-attention with H heads, we will simply add the output of the H heads as the final output.

Definition H.2 (MLP). For a set of matrices $W_{outer} \in R^{k \times H}$ , $W_{inner} \in R^{H \times k}$ for some k, H > 0 and an activation function $\sigma$ , the output of the MLP layer on an input sequence $x_{1}, \cdots, x_{L}$ with each $x_{i} \in R^{k}$ is given by the output sequence $y_{1}, \cdots, y_{L}$ , where for all i

$$
\boldsymbol {y} _ {i} = \boldsymbol {W} _ {\text {outer}} \sigma (\boldsymbol {W} _ {\text {inner}} \boldsymbol {x} _ {i}).
$$

Lemma H.3. For GELU (Hendrycks & Gimpel, 2016) activation function, which takes $x \in \mathbb{R}$ as input and returns $x\Phi(x)$ as output, with $\Phi(x)$ representing the standard Gaussian cumulative distribution function, for any two variables $x, y \in \mathbb{R}$ , the following holds true:

$$
\sqrt {\pi / 2} \left(G E L U (x + y) - G E L U (x) - G E L U (y)\right) = x y + \mathcal {O} (x ^ {3} y ^ {3}).
$$

The above lemma has been taken from Akyurek et al. (2023).

# H.2. Transformer construction

Recall that a translation task in $\mathbf{MLT}(d,n)$ involves two primary operations at each step: Circular shift and Translate. We will refer to the surrogate model to use notations for different operations in the translation task. Recall from Equation (18), the surrogate model for $\mathbf{MLT}(d,n)$ , denoted by $\text{SURR-MLT}_{\{\boldsymbol{W}_{i}\}_{i=1}^{d}}(\cdot)$ with trainable parameters $\{\boldsymbol{W}_{i}\}_{i=1}^{d}=\{\boldsymbol{W}_{1},\cdots,\boldsymbol{W}_{d}\}$ , can be represented by the following recursive expression

$$
\boldsymbol {V} _ {i + 1} = \operatorname{HardMax} \left(\boldsymbol {C} _ {i} + \boldsymbol {W} _ {i}\right) \text {Shift} \left(\boldsymbol {V} _ {i}\right) := \operatorname{HardMax} \left(\boldsymbol {C} _ {i} + \boldsymbol {W} _ {i}\right) \tilde {\boldsymbol {V}} _ {i}, \tag {77}
$$

for $1 \leq i \leq d$ . Here Shift represents Circular shift operation, and $\mathrm{HardMax}(\boldsymbol{C}_i + \boldsymbol{W}_i)$ represents Translate operation, where the operation can be done either using the relevant in-context information $\boldsymbol{C}_i$ or the in-weights memory parameters $\boldsymbol{W}_i$ when in-context information isn't provided in form of $\boldsymbol{C}_i$ . HardMax is the column-wise hard-max function converting $\boldsymbol{C}_i + \boldsymbol{W}_i$ to a binary column stochastic matrix.

Lemma H.4. For the family of translation tasks $MLT(d, n)$ , there exists a transformer model with embedding size $2n^{2} + 2d + 4$ , 2d relative self-attention layers (containing either 1 or 3 heads), and 2d MLP layers that can simulate the surrogate model, $\text{SURR-MLT}_{\{\boldsymbol{W}_{i}\}_{i=1}^{d}}(\cdot)$ .

For each input sequence $s_{1}$ of length L, the input sequence of embeddings to the transformer will be of length $n^{2}d + L/2 + d$ , where the last d embeddings are padding tokens (<P>) and given as 0s, and the length of output sequence of the transformer model will be $n^{2}d + d + L/2$ , where the last L/2 output embeddings will be used for loss computation. The middle d embeddings in output are represented by <THINK> tokens. $^{a}$

Outline of the construction: Our argument will be for any general MLT(d, n). To create the transformer, we will create similar transformer modules that handle $\text{HardMax}(\boldsymbol{C}_{i} + \boldsymbol{W}_{i}) \text{ Shift}(\boldsymbol{V}_{i})$ for each i. We will refer to the constructed modules as MLT-MODULE and will mention the specific step i as an argument when we attempt to use the module to perform translation step $\text{HardMax}(\boldsymbol{C}_{i} + \boldsymbol{W}_{i}) \text{ Shift}(\boldsymbol{V}_{i})$ . W.l.o.g., we will assume we are building a MLT-MODULE to perform translation step $\text{HardMax}(\boldsymbol{C}_{i} + \boldsymbol{W}_{i}) \text{ Shift}(\boldsymbol{V}_{i})$ . MLT-MODULE will contain two modules, CIRCULAR SHIFT-MODULE and TRANSLATE-MODULE, which simulate Circular shift and Translate operations, and have been outlined in Algorithms 6 and 7.

Next, we explain the structure of the embeddings in the transformer architecture. Our embeddings will be built on the context representation matrices $\{C_{1},\cdots,C_{d}\}$ and the matrix representation $V_{1}$ from input sequence $s_{1}:=(s_{1,1},\cdots,s_{1,L})$ and subsequent intermediate representation of the translation task. Furthermore, our embeddings will contain additional information like indices of the context matrices when utilizing them in-context, segment indicators that represent whether embeddings represent context matrices, or the input sequence tokens, and start and end indicators that indicate the start and the end embeddings representing the input sequence. This information can be extracted from the input sequence using a few input processing layers, though we do not delve into the specifics.

# H.3. Structure of input embeddings to MLT-MODULE

Our embeddings will be split into 4 components: token, context matrix indicator, start and end indicator, and segment indicator. To maintain simplicity in our discussion, we will present them as 4 separate embeddings.

For a module that will represent $\text{HardMax}(\boldsymbol{C}_{i} + \boldsymbol{W}_{i}) \text{Shift}(\boldsymbol{V}_{i})$ , we assume the input will be provided as 2 major segments.

1. First segment: in-context matrices We will give the in-context information $\{C_{p_1}\}_{p_1 = 1}^d$ as follows.: each context matrix $C_{p_1}$ will be fed as $n^2$ token embeddings, $\{[e_j; C_{p_1}e_j] \in \mathbb{R}^{2n^2}\}_{j=1}^n$ , where $e_j$ represents a one-hot $n^2$ dimensional vector that contains 1 in position $j$ and $[e_j; C_{p_1}e_j]$ represents a concatenation of $e_j$ and $C_i e_j$ . Thus, the in-context information will look as follows

$$
\{[ \pmb {e} _ {j}; \pmb {C} _ {1} \pmb {e} _ {j} ] \} _ {j = 1} ^ {n ^ {2}}, \dots , \{[ \pmb {e} _ {j}; \pmb {C} _ {d} \pmb {e} _ {j} ] \} _ {j = 1} ^ {n ^ {2}}
$$

Additional embedding: context matrix level indicator In order to differentiate the different context matrices, we will use an additional d dimensions in the embeddings to represent a one-hot vector that indicates the index of the corresponding step they will be used for. For simplicity, we will represent these dimensions separately as separate embeddings: $l_{p_{1}} \in R^{d}$ , which are one-hot vectors that contain 1 in position $p_{1}$ and 0 otherwise.

2. Second segment: input query For a length-L input sequence $\boldsymbol{s}_{i} = (\boldsymbol{s}_{i,1}, \boldsymbol{s}_{i,2}, \ldots, \boldsymbol{s}_{i,L})$ , we will use the sequence of columns of its matrix representation $V_{i} \in R^{n^{2} \times L/2}$ , appended by 0s to match embedding sizes, as token embeddings for the input query. A padding embedding <P>, containing 0s, follows this sequence as an end of sequence embedding.

Additional embedding: start and end indicator embedding We will have 2 additional dimensions representing whether a token represents the start or the end of the input query sequence (first or second dimension activated respectively). Start of the input query sequence is determined by the first embedding in the input query, while end of the input query sequence is determined by the first padding embedding containing 0s after the input query sequence. We will represent these dimensions separately as separate embeddings: $\{b_1, b_2, 0 \in \mathbb{R}^2\}_{i \in L}$ , which are one-hot vectors. $b_1$ contains 1 in dimension 1 if the embeddings represent start of the sequence, $b_2$ contains 1 in dimension 2 if the embeddings represent end of the sequence. Other embeddings have 0s.

Additional embedding: segment embedding We will differentiate the input embeddings in the two segments using 2 dimensions that represent one-hot vectors indicating segment indices. We will represent these dimensions separately as separate embeddings: $g_{1}, g_{2} \in R^{2}$ , where both are one-hot vectors, with $g_{1}$ containing 1 in dimension 1 and $g_{2}$ containing 1 in dimension 2.

All the notations have been summarized in Table 8.

Additional optional inputs: There might be additional input embeddings, represented as <THINK> in the first segment, which are null inputs and are ignored during self-attention computation. As discussed next, we will right shift the sequence by 1 at each step, in order to handle Circular shift operation with causal masking in transformers.

# H.4. Structure of the output embeddings from MLT-MODULE

All the embeddings in the first segment are kept intact. In the second segment, the module outputs a null output <THINK>, followed by L/2 output embeddings that represent the columns of $V_{i}$ . We will require <THINK> to represent Circular shift with causal self-attention in transformers. <THINK> will be ignored in self-attention computation and so, we will ignore their discussion for simplicity of presentation. Other embedding values are kept intact for the generated output, except start and end indicator embeddings $b_{1}, b_{2}$ , which need to be right shifted at each step. The right shift operation can be handled similar to our computations on the token embeddings and so, we ignore them in our construction below. We summarize these in Table 9.

<table><tr><td>Embedding Name</td><td>Dimension size</td><td>First segment values (In-context information)</td><td>Second segment values (Input sequence embeddings)</td></tr><tr><td>Token</td><td> $2n^{2}$ </td><td> $\{[e_{j};C_{1}e_{j}]\}_{j=1}^{n^{2}},\cdots,\{[e_{j};C_{d}e_{j}]\}_{j=1}^{n^{2}}$ </td><td> $[V_{i-1}^{(1)};0],\cdots,[V_{i-1}^{(L/2)};0],<\mathbb{P}>$ </td></tr><tr><td>Context matrix index indicator</td><td> $d$ </td><td> $\{l_{1}\}_{[1,n^{2}],\{l_{2}\}_{[1,n^{2}],\cdots\{l_{d}\}_{[1,n^{2}]}}$ </td><td> $0,\cdots,0$ </td></tr><tr><td>Start and End indicator</td><td>2</td><td> $0,\cdots,0$ </td><td> $b_{1},0,\cdots,0,b_{2}$ </td></tr><tr><td>Segment</td><td>2</td><td> $g_{1},\cdots,g_{1}$ </td><td> $g_{2},\cdots,g_{2}$ </td></tr></table>

Table 8. Input embeddings to MLT-MODULE that simulates $\mathrm{HardMax}(\mathbf{C}_i + \mathbf{W}_i)$ Shift $(\mathbf{V}_i)$ . $e_j$ indicates a one-hot $n^2$ dimensional vector that contains 1 in dimension $j$ .

<table><tr><td>Embedding Name</td><td>Dimension size</td><td>First segment values(In-context information)</td><td>Second segment values(Input sequence embeddings)</td></tr><tr><td>Token</td><td> $2n^{2}$ </td><td> $\{[e_{j};C_{1}e_{j}]\}_{j=1}^{n^{2}},\cdots,\{[e_{j};C_{d}e_{j}]\}_{j=1}^{n^{2}}$ </td><td></td></tr><tr><td>Context matrix index indicator</td><td> $d$ </td><td> $\{l_{1}\}_{[1,n^{2}],\{l_{2}\}_{[1,n^{2}],\cdots\{l_{d}\}_{[1,n^{2}]}}$ </td><td> $\mathbf{0},\mathbf{0},\cdots,\mathbf{0}$ </td></tr><tr><td>Start and End indicator</td><td>2</td><td> $\mathbf{0},\cdots,\mathbf{0}$ </td><td> $\mathbf{0},\mathbf{b}_{1},\mathbf{0},\cdots,\mathbf{0},\mathbf{b}_{2}$ </td></tr><tr><td>Segment</td><td>2</td><td> $\mathbf{g}_{1},\cdots,\mathbf{g}_{1}$ </td><td> $\mathbf{0},\mathbf{g}_{2},\cdots,\mathbf{g}_{2}$ </td></tr></table>

Table 9. Output of the transformer module MLT-MODULE that simulates $\mathrm{HardMax}(\mathbf{C}_i + \mathbf{W}_i)$ Shift $(\mathbf{V}_i)$ . <THINK> represents a null output and won't be attended to in the future modules. We ignore this symbol for simplicity, when analyzing any module. $e_j$ indicates a one-hot $n^2$ dimensional vector that contains 1 in dimension $j$ .

Constructing the MLT-MODULE: The MLT-MODULE consists of 2 self-attention layers and 2 MLP layers. We use one self-attention layer and an MLP layer to represent Circular shift operation, one self-attention layer to represent $C_{i}$ Shift( $V_{i}$ ), and one MLP layer to represent HardMax( $C_{i} + W_{i}$ ) Shift( $V_{i}$ ). We name the two modules for Circular shift and Translate as CIRCULAR SHIFT-MODULE and TRANSLATE-MODULE respectively. We have outlined their constructions in Algorithms 6 and 7.

# H.5. Step 1 (CIRCULAR SHIFT-MODULE): Represent Circular shift using a self-attention and an MLP layer

As Circular shift only focuses on the input query sequence and not the in-context matrices, we will simply focus the module's operation on embeddings in the second segment. The effect of the operation on embeddings in the first segment can be removed using a gated residual connection. From Lemma G.3, we have that the output of the Circular shift operation on any sequence, represented by its matrix representation $V_{i}$ , can be written as

$$
\tilde {\boldsymbol {V}} _ {i} ^ {(j)} = Q \boldsymbol {V} _ {i} ^ {(j)} \odot Q ^ {\top} \boldsymbol {V} _ {i} ^ {((j + 1) \% L)}, \text { for all } 1 \leq j \leq L / 2.
$$

where $Q = (I_n \otimes \mathbf{1}_n)(\mathbf{1}_n \otimes I_n)^\top$ , $\mathbf{1}_n \in \mathbb{R}^{n \times 1}$ is the all-ones vector, and $\odot$ is the Hadamard product.

In order to represent the operation, we will first use a self-attention layer to compute $(\mathbf{1}_{n}\otimes I_{n})^{\top}\mathbf{V}_{i}^{(j)}$ and $(I_{n}\otimes\mathbf{1}_{n})\mathbf{V}_{i}^{(j+1\%L)}$ at each column j. Because the computation of $\tilde{\mathbf{V}}_{i}^{(j)}$ requires the model to look forward to $\mathbf{V}_{i}^{(j+1)}$ , we need to shift the computation of $\tilde{\mathbf{V}}_{i}^{(j)}$ to position $j+1$ , as a causal attention mask is involved in self-attention computation.

After right shift operation, we will represent the output of the self-attention computation as $\langle THINK\rangle$ , $[o_{2};0]$ , $[o_{3};0]$ , $\cdots$ , $[o_{L/2+1};0]$ , and we will ignore the $\langle THINK\rangle$ embedding. Note that the second half of the output embeddings will still contain 0s and we will ignore them in the current computation. Then, the above computation can be rephrased as

$$
\mathbf {o} _ {j} = Q \boldsymbol {V} _ {i} ^ {(j - 1)} \odot Q ^ {\top} \boldsymbol {V} _ {i} ^ {(j)}, \text {   for   all   } 2 \leq j \leq L / 2.
$$

$$
\mathbf {o} _ {L / 2 + 1} = Q \boldsymbol {V} _ {i} ^ {(L / 2)} \odot Q ^ {\top} \boldsymbol {V} _ {i} ^ {(1)}
$$

Self-attention layer: The computation of $o_{j}$ , for $2 \leq j \leq L/2$ , requires the computation of $(\mathbf{1}_{n} \otimes I_{n})^{\top} \mathbf{V}_{i}^{(j-1)}$ and $(I_{n} \otimes \mathbf{1}_{n}) \mathbf{V}_{i}^{(j)}$ . This will require 2 attention heads, one head that attends to itself, and another that attends to previous embedding at each position. We will outline both below. We will require one additional head, as computing $o_{L/2+1}$ will require the model to compute $(I_{n} \otimes \mathbf{1}_{n}) \mathbf{V}_{i}^{(1)}$ .

1. Attention Head 1 computes $(\mathbf{1}_n\otimes I_n)^\top \mathbf{V}_i^{(j - 1)}$ at position $j$ for all $2\leq j\leq L / 2 + 1$ . This can be done using a self-attention head (Definition H.1) that sets query and key matrices $\pmb{W}_{query},\pmb{W}_{key}$ , and biases $\{b_i\}_{t\leq i\leq 0}$ such that the attention score between embeddings at any two positions $p_1,p_2$ is given as follows:

$$
a _ {p _ {1}, p _ {2}} = 1, \text { if } p _ {2} - p _ {1} = - 1, \quad 0 \text { otherwise }
$$

$W_{value}$ is set such that for any input x, the output of $W_{value}x$ is given by

$$
\left(\boldsymbol {W} _ {\text { value }} \boldsymbol {x}\right) _ {p _ {1}} = \sum_ {j = 0} ^ {n - 1} x _ {n \cdot j + p _ {1}}.
$$

In simple words, this operation simply adds up the values in dimensions $p_{1}, p_{1} + n, p_{1} + 2n, \cdots$ and stores them at position $p_{1}$ for all $1 \leq p_{1} \leq n$ .

2. Attention Head 2 computes $(I_n \otimes \mathbf{1}_n)$ $V_i^{(j)}$ at position $j$ for all $2 \leq j \leq L/2$ . This can be done using a self-attention head (Definition H.1) that sets $W_{query}$ , $W_{key}$ , $W_{value}$ and biases $\{b_i\}_{t \leq i \leq 0}$ such that the attention score between embeddings at any two positions $p_1, p_2$ is given as follows:

$$
a _ {p _ {1}, p _ {2}} = 1, \text { if } p _ {2} - p _ {1} = 0, \quad 0 \text { otherwise }
$$

$W_{value}$ is set such that for any input x, the output of $W_{value}x$ is given by

$$
\left(\boldsymbol {W} _ {\text { value }} \boldsymbol {x}\right) _ {p _ {1} + n} = \sum_ {j = 1} ^ {n} x _ {j + p _ {1} n - n}.
$$

In simple words, this operation simply adds up the values in dimensions $1 + (p_{1} - 1)n, 2 + (p_{1} - 1)n, \cdots$ and stores them at dimension $p_{1} + n$ for all $1 \leq p_{1} \leq n$ .

3. Attention head 3 will compute $(I_n \otimes \mathbf{1}_n)$ $V_i^{(1)}$ and store in $\mathbf{o}_{L/2+1}$ . This can be done by a self-attention layer which activates only between positions $p_1$ and $p_2$ that represent the start and the end tokens of the sequence, i.e. contain $b_1$ and $b_2$ as start and end indicator embeddings, and is 0 otherwise. $W_{value}$ is set same as attention head 2.

The output of the three heads are simply added up. Hence, at each position $2 \leq j \leq L / 2 + 1$ , the output $\pmb{o}_j$ has $(\mathbf{1}_n \otimes I_n)^\top V_i^{(j-1)}$ in $[1, n]$ dimensions and $(I_n \otimes \mathbf{1}_n) V_i^{(j\%L)}$ in $[n+1, 2n]$ dimensions.

MLP layer: The objective with the MLP layer (Definition H.2) will be to multiply $(\mathbf{1}_{n} \otimes I_{n})^{\top} \mathbf{V}_{i}^{(j-1)}$ present in $[1, n]$ dimensions and $(I_{n} \otimes \mathbf{1}_{n}) \mathbf{V}_{i}^{(j\%L)}$ present in $[n+1, 2n]$ dimensions in each position j. This can be done by using an MLP layer with GELU activation by using Lemma H.3. The weights of the MLP layer are set as follows: $W_{inner}$ is set such that

for all input x, we have

$$
\left(\boldsymbol {W} _ {\text { inner }} \boldsymbol {x}\right) _ {i} = \frac {1}{N} x _ {i} + \frac {1}{N} x _ {n + i}, \quad \text { for   all } 1 \leq i \leq n,
$$

$$
\left(\boldsymbol {W} _ {\text { inner }} \boldsymbol {x}\right) _ {i + n} = \frac {1}{N} x _ {i}, \quad \text { for   all } 1 \leq i \leq n,
$$

$$
\left(\boldsymbol {W} _ {\text {inner}} \boldsymbol {x}\right) _ {i + 2 n} = \frac {1}{N} x _ {n + i}, \quad \text {for all} 1 \leq i \leq n.
$$

$W_{outer}$ is set such that for all $\pmb{x} \in \mathbb{R}^{3n}$

$$
\left(\boldsymbol {W} _ {\text { outer }} \boldsymbol {x}\right) _ {i} = N ^ {2} (x _ {i} - x _ {i + n} - x _ {i + 2 n}), \quad \text { for   all } 1 \leq i \leq n.
$$

All other coordinates in these matrices are set as 0s. N is set as a large number (say 100). By Lemma H.3, the output of the MLP layer will be <THINK>, $o_{2}, \cdots, o_{L/2+1}$ , with $o_{j}$ containing

$$
\tilde {\boldsymbol {V}} _ {i} ^ {(j - 1)} + \mathcal {O} (N ^ {- 4}) := (\mathbf {1} _ {n} \otimes I _ {n}) ^ {\top}   \boldsymbol {V} _ {i} ^ {(j - 1)} \odot (I _ {n} \otimes \mathbf {1} _ {n})   \boldsymbol {V} _ {i} ^ {(j \% L)} + \mathcal {O} (N ^ {- 4})
$$

at each position $2 \leq j \leq L / 2$ .

<table><tr><td>Embedding Name</td><td>Dimension size</td><td>First segment values(In-context information)</td><td>Second segment values(Input query sequence embeddings)</td></tr><tr><td>Token</td><td> $2n^{2}$ </td><td> $\{[e_{j};C_{1}e_{j}]\}_{j=1}^{n^{2}},\cdots,\{[e_{j};C_{d}e_{j}]\}_{j=1}^{n^{2}}$ </td><td> $<THINK>,[\tilde{V}_{i}^{(1)};0],\cdots,[\tilde{V}_{i}^{(L/2)};0]$ </td></tr><tr><td>Context matrix index indicator</td><td> $d$ </td><td> $\{l_{1}\}_{[1,n^{2}],\{l_{2}\}_{[1,n^{2}],\cdots\{l_{d}\}_{[1,n^{2}]}}$ </td><td> $0,0,\cdots,0$ </td></tr><tr><td>Start and End indicator</td><td>2</td><td> $0,\cdots,0$ </td><td> $0,b_{1},0,\cdots,0,b_{2}$ </td></tr><tr><td>Segment</td><td>2</td><td> $g_{1},\cdots,g_{1}$ </td><td> $0,g_{2},\cdots,g_{2}$ </td></tr></table>

Table 10. Output of CIRCULAR SHIFT-MODULE in MLT-MODULE that simulates Circular shift, i.e. computes Shift( $V_i$ ) at second segment token embeddings. <THINK> represents a null output and won't be attended to in the future modules. We ignore this symbol for simplicity, when analyzing any module. $e_j$ indicates a one-hot $n^2$ dimensional vector that contains 1 in dimension $j$ .

# H.6. Step 2 (TRANSLATE-MODULE): Translate as a module containing a self-attention and an MLP layer

Our current token embeddings are given as <THINK>, $[o_{2};0]$ , $[o_{3};0]$ , $\cdots$ , $[o_{L/2+1};0]$ , where each $o_{j}$ contain $\tilde{V}_{i}^{(j-1)}$ . Other embeddings have been kept intact. The in-context information are given as $\{[e_{j};C_{1}e_{j}]\}_{j=1}^{n^{2}}$ , $\cdots$ , $\{[e_{j};C_{d}e_{j}]\}_{j=1}^{n^{2}}$ . We will first use a self-attention layer to compute $[C_{i}\tilde{V}_{i}^{(j-1)};\tilde{V}_{i}^{(j-1)}]$ at position $2\leq j\leq L/2+1$ . We then use an MLP layer to represent $\text{HardMax}(C_{i}+W_{i})\tilde{V}_{i}^{(j-1)}$ .

Self-attention layer to represent $[C_{i}\tilde{\mathbf{V}}_{i}^{(j-1)};\tilde{\mathbf{V}}_{i}^{(j-1)}]$ : We will use two attention heads.

1. The first attention head computes $C_{i}\tilde{V}_{i}^{(j-1)}$ : Matrices $W_{query}$ , $W_{key}$ are set such that the attention score between an token embedding in first segment $[e_{r};C_{\ell}e_{r}]$ and a token embedding in second segment $o_{j}$ is given by

$$
\langle \boldsymbol {e} _ {r}, \tilde {\boldsymbol {V}} _ {i} ^ {(j - 1)} \rangle , \text {   if   } \ell = i, \text {   and   } 0 \text {   otherwise. }
$$

for any $j \in [2, L/2 + 1]$ , $r \in [1, d]$ . The condition requires the model to attend to $C_{i}$ and ignore other in-context information. The condition can be set using the Context matrix index indicator vectors $l_{\ell}$ which is present in each in-context information embedding.

The attention between any two input sequence embedding $o_{j}$ and $o_{j'}$ is computed as 0s. The distinction between the attention scores of pairs of embeddings in second segment, $o_{j}$ and $o_{j'}$ , v/s attention scores between a token embedding in second segment and a token embedding in first segment, $o_{j}$ and $[e_{r}; C_{\ell}e_{r}]$ , can be done by using the segment indicator embeddings $g_{1}$ and $g_{2}$ used to differentiate token embeddings in first segment and the input sequence embedding vectors.

Matrix $W_{value}$ is set such that the columns of each $C_{\ell}s$ are picked from the token embeddings in the first segment: $\{\{[e_{r};C_{\ell}e_{r}]\}_{r=1}^{n^{2}}\}_{\ell=1}^{d}$ .

2. The second attention head simply copies the input $\tilde{\mathbf{V}}_{i}^{(j-1)}$ : This can be done with an attention head that attends to itself at each position j and copies $\tilde{\mathbf{V}}_{i}^{(j-1)}$ to output.

The output of the two attention heads are simply added up. The output embeddings will now look as follows: <THINK>, $\{[C_{i}\tilde{V}_{i}^{(j-1)};\tilde{V}_{i}^{(j-1)}]\}_{j=1}^{L/2}$ .

MLP to represent $\text{HardMax}(\boldsymbol{C}_{i} + \boldsymbol{W}_{i})\tilde{\boldsymbol{V}}_{i}^{(j-1)}$ : Our current token embeddings at any position j contain both $\boldsymbol{C}_{i}\tilde{\boldsymbol{V}}_{i}^{(j-1)}$ and $\tilde{\boldsymbol{V}}_{i}^{(j-1)}$ . The first layer of MLP can be used to compute $(\boldsymbol{C}_{i} + \boldsymbol{W}_{i})\tilde{\boldsymbol{V}}_{i}^{(j-1)}$ by setting the weights of the layer using $W_{i}$ . We simulate HardMax operation as follows:

$$
\tilde {\mathbf {o}} _ {j} / \left\| \tilde {\mathbf {o}} _ {j} \right\| _ {2}, \text {   where   } \tilde {\mathbf {o}} _ {j} = \operatorname{GELU} ((C _ {i} + W _ {i}) \tilde {V} _ {i} ^ {(j - 1)})
$$

The $\ell_2$ normalization is equivalent to RMSnorm operation (Zhang & Sennrich, 2019). This is an approximation of the HardMax function, which are equivalent only under the following conditions: for each column $j$

1. either $C_i^{(j)}$ or $\pmb{W}_i^{(j)}$ are all 0s.   
2. $C_{i}^{(j)}$ and $W_{i}^{(j)}$ are both one-hot vectors and they match at the corresponding activated dimension.

# Algorithm 6 CIRCULAR SHIFT-MODULE: Self-attention and MLP layers for Circular shift

Require: Input embeddings (Token, Context matrix index indicator, Start and End Indicator, and Segment embeddings split into 2 segments) (Table 8). Important ones (for the current module) are

1. Token embeddings: First segment contains $\{[e_{j};C_{1}e_{j}]\}_{j=1}^{n^{2}},\cdots,\{[e_{j};C_{d}e_{j}]\}_{j=1}^{n^{2}}$ and the second segment contains $[Shift(V_{i-1}^{(1)});0],\cdots,[Shift(V_{i-1}^{(L/2)});0],0$   
2. Start and End indicator: First segment contains all 0s and second segments contains $\pmb{b}_1$ and $\pmb{b}_2$ at first and last embedding, while containing all 0s everywhere else.

Step a: Using a self-attention layer with 3 attention heads, change the token embeddings in second segment as <THINK>, $\{o_{j}\}_{j=2}^{L/2+1}$ s.t. $o_{j}$ has $(\mathbf{1}_{n} \otimes I_{n})^{\top} \mathbf{V}_{i}^{(j-1)}$ in [1, n] dimensions and $(I_{n} \otimes \mathbf{1}_{n}) \mathbf{V}_{i}^{(j\%L)}$ in $[n+1, 2n]$ dimensions. Primarily,

\- Attention head 1: Computes attention score between any two positions $p_1, p_2$ as $a_{p_1, p_2} = 1$ iff $p_2 - p_1 = -1$ and 0 otherwise. Value matrix $W_{value}$ is set such that for any input $x$ , the output of $W_{value}x$ is given by (for all $p_1 \in [1, n]$ )

$$
\left(\boldsymbol {W} _ {\text { value }} \boldsymbol {x}\right) _ {p _ {1}} = \sum_ {j = 0} ^ {n - 1} x _ {n \cdot j + p _ {1}}.
$$

\- Attention head 2: Computes attention score between any two positions $p_1, p_2$ as $a_{p_1, p_2} = 1$ iff $p_2 - p_1 = 0$ and 0 otherwise. $W_{value}$ is set such that for any input $x$ , the output of $W_{value}x$ is given by (for all $p_1 \in [1, n]$ )

$$
\left(\boldsymbol {W} _ {\text { value }} \boldsymbol {x}\right) _ {p _ {1} + n} = \sum_ {j = 1} ^ {n} x _ {j + p _ {1} n - n}.
$$

\- Attention head 3: Computes attention score between any two positions $p_1, p_2$ as $a_{p_1, p_2} = 1$ iff $\boldsymbol{b}_2$ and $\boldsymbol{b}_1$ are present as at positions $p_1$ and $p_2$ respectively and 0 otherwise. $W_{value}$ is set such that for any input $\boldsymbol{x}$ , the output of $W_{value}\boldsymbol{x}$ is given by (for all $p_1 \in [1, n]$ )

$$
\left(\boldsymbol {W} _ {\text { value }} \boldsymbol {x}\right) _ {p _ {1} + n} = \sum_ {j = 1} ^ {n} x _ {j + p _ {1} n - n}.
$$

Sum the output of the three heads.

Step b: Use MLP layer to change the token embeddings in second segment as <THINK>, $\{o_{j}\}_{j=2}^{L/2+1}$ s.t. $o_{j}$ has $\tilde{V}_{i}^{(j-1)}$ with some small error. Primary computation at each position j is given as (for a large N)

$$
\begin{array}{l} \sqrt {2 / \pi} N ^ {2} \left(\text { GELU } (\frac {1}{N} \left(\mathbf {1} _ {n} \otimes I _ {n}\right) ^ {\top} \boldsymbol {V} _ {i} ^ {(j - 1)} + \frac {1}{N} \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \boldsymbol {V} _ {i} ^ {(j \% L)})\right) \\ \left. - \operatorname{GELU} \left(\frac {1}{N} \left(\mathbf {1} _ {n} \otimes I _ {n}\right) ^ {\top} \boldsymbol {V} _ {i} ^ {(j - 1)}\right) - \operatorname{GELU} \left(\frac {1}{N} \left(I _ {n} \otimes \mathbf {1} _ {n}\right) \boldsymbol {V} _ {i} ^ {(j \% L)}\right)\right), \\ \end{array}
$$

which will return

$$
\tilde {\boldsymbol {V}} _ {i} ^ {(j - 1)} + \mathcal {O} (N ^ {- 4}) := (\mathbf {1} _ {n} \otimes I _ {n}) ^ {\top}   \boldsymbol {V} _ {i} ^ {(j - 1)} \odot (I _ {n} \otimes \mathbf {1} _ {n})   \boldsymbol {V} _ {i} ^ {(j \% L)} + \mathcal {O} (N ^ {- 4})
$$

Return the output embeddings (as given in Table 10).

# Algorithm 7 TRANSLATE-MODULE: Self-attention and MLP layers for Translate

Require: We will require an index, and input embeddings as input:

- Index $i$ (indicating index of the MLT-MODULE it is a part of),   
- Embeddings (Token, Context matrix index indicator, Start and End Indicator, and Segment embeddings split into 2 segments) from the output of its preceding CIRCULAR SHIFT-MODULE (Table 10). Important ones (for the current module) are   
1. Token embeddings: First segment contains $\{[e_{j};C_{1}e_{j}]\}_{j=1}^{n^{2}},\cdots,\{[e_{j};C_{d}e_{j}]\}_{j=1}^{n^{2}}$ and the second segment contains <THINK>, $[\tilde{\mathbf{V}}_{i}^{(1)};\mathbf{0}],\cdots,[\tilde{\mathbf{V}}_{i}^{(L/2)};\mathbf{0}]$   
2. Context matrix index indicator: First segment contains $\{l_{1}\}_{[1,n^{2}]},\{l_{2}\}_{[1,n^{2}]},\cdots\{l_{d}\}_{[1,n^{2}]}$ and second segments contains all 0s vectors.

Step a: Using a self-attention layer, change token embeddings in the second segment as <THINK>, $\{[C_{i}\tilde{V}_{i}^{(j-1)};\tilde{V}_{i}^{(j-1)}]\}_{j=1}^{L/2}$ .

\- Primarily, the self-attention score between embeddings that contain a second segment token embedding $[\tilde{\mathbf{V}}_i^{(p_2)}; \mathbf{0}]$ and a first segment token embedding $[\pmb{e}_{p_1}; \pmb{C}_r\pmb{e}_{p_1}]$ (for any $r \in [1, d]$ , $p_1 \in [1, n^2], p_2 \in [1, L]$ ) is computed as

$$
\langle \boldsymbol {e} _ {p _ {1}}, \tilde {\boldsymbol {V}} _ {i} ^ {(p _ {2})} \rangle \cdot \langle \boldsymbol {l} _ {r}, \boldsymbol {l} _ {i} \rangle ,
$$

where $l_{r}$ is the corresponding context matrix index indicator for the first segment embedding under consideration and $l_{i}$ is constructed using the index i.

\- $C_r e_{p_1}$ is used as value vector from each first segment embeddings.

Step b: Using an MLP layer, change token embeddings in the second segment to contain <THINK>, $\{o_{j}\}_{j=2}^{L/2+1}$ , where

$$
\mathbf {o} _ {j} = \tilde {\mathbf {o}} _ {j} / \| \tilde {\mathbf {o}} _ {j} \| _ {2}, \text {   where   } \tilde {\mathbf {o}} _ {j} = \operatorname{GELU} ((C _ {i} + W _ {i}) \tilde {V} _ {i} ^ {(j - 1)})
$$

Return the output embeddings (as given in Table 9).

# H.7. Trainability of the constructed transformer

How does our constructed transformer perform on the MLT task? To do so, we hand-construct the designed transformer and train the transformer model on a random translation task in MLT(5, 10), that has depth 5 and number of characters 10 in each translation level. We train only parameters $\{\pmb{W}_i\}_{i=1}^d$ using Layer-wise and Full Parameter optimization algorithms from Algorithms 4 and 5 respectively.

However, we make two changes. First, we use Adam optimizer instead of SGD, as we found SGD to get stuck frequently at bad minimas. Next, instead of masking in-context representation of layers in a rotating curriculum fashion, where each layer $i$ 's in-context representation were masked in intervals $[j \cdot (i - 1)n^2, j \cdot in^2]$ for all integers $j \geq 0$ , we mix in the masking of all layers together by randomly picking a layer $i$ and mask a random column from $C_i$ . We found that mixing in masking of all layers helps train the model faster.

We vary the peak learning rate as $\{10^{-2}, 10^{-3}, 10^{-4}\}$ . We use a cosine decay learning rate schedule with warmup steps 100 and minimum learning rate at end of training as $0.1\times$ the peak learning rate. We train with cross entropy loss, and use a total of 51200 training sequences of length 20 during the course of training. We use a batch size of 32 per gradient update step. We also use a small $\ell_{1}$ regularization, with strength $10^{-4}$ , on the rows of the parameters to help optimization.

Observations: We report the performance of a model trained with both Layer-wise and Full Parameter optimization algorithm with peak learning rate $10^{-3}$ and $\ell_{1}$ regularization with strength $10^{-4}$ in Figure 15. We observe that under both algorithms, the trainable parameters quickly learn the relevant matrices that represent the true set of phrasebooks to minimize the training loss.

![](images/895f2e7b91ae60b776febf0f04dbdfff16dd291f6ccb7d2b510313a85d041b82.jpg)

<details>
<summary>line</summary>

| Steps | Line 1 | Line 2 | Line 3 | Line 4 |
| ----- | ------ | ------ | ------ | ------ |
| 0     | 0.0    | 0.0    | 0.0    | 0.0    |
| 500   | 0.9    | 0.8    | 0.7    | 0.6    |
| 1000  | 1.0    | 0.95   | 0.85   | 0.75   |
| 1500  | 1.0    | 1.0    | 0.95   | 0.85   |
</details>

![](images/7dd8092fc72308fbe364cf984b269bf81172ac35f92baae54f0c7db04bff7c01.jpg)

<details>
<summary>line</summary>

| Steps | W1    | W2    | W3    | W4    | W5    |
|-------|-------|-------|-------|-------|-------|
| 0     | 0.0   | 0.0   | 0.0   | 0.0   | 0.0   |
| 500   | 0.8   | 0.7   | 0.6   | 0.5   | 0.4   |
| 1000  | 0.95  | 0.9   | 0.85  | 0.8   | 0.75  |
| 1500  | 1.0   | 1.0   | 1.0   | 1.0   | 1.0   |
</details>

Figure 15. We perform Layer-wise and Full Parameter Training with Gradient Descent (defined in Appendix G.5.1) on the handconstructed transformer (Lemma H.4) that represents SURR-MLT (Definition G.7) designed for $\mathbf{MLT}_{\Pi^*}$ in $\mathbf{MLT}(5,10)$ . Here, we report the portion of columns from the trainable parameters, which after HardMax application $\{\text{HardMax}(\boldsymbol{W}_i)\}_{i=1}^d$ , align with the corresponding stochastic matrices of the phrasebooks $\{\text{Matrix}(\pi_i^*)\}_{i=1}^d$ . We observe that under both algorithms, the trainable parameters quickly learn the relevant stochastic matrices.

# I. Additional Experiment Setups

# I.1. Format of Training Text

Here we present examples of training data used for experiments in Section 3

INPUT:   
```txt
"<|begin_of_text|> <|begin_of_text|> <|begin_of_text|> <|start_header_id|> user <|end_header_id |

You are performing a special translation task called language_task_2. The subset of dictionaries used are as follows:
Dictionary used from language 1 to language 2:
DA -> NJ; AD -> JI; BC -> OI; CC -> NN; HE -> PK; FE -> ML; GH -> LN; EG -> PJ; AF -> KM;
EC -> KP; BE -> KI; FD -> MN; ED -> PO; BA -> PL; BD -> IP; DG -> II; DH -> IO; EF -> LM;
HB -> JK; EA -> KJ; AH -> LJ; GC -> IK; FB -> PI; FG -> JO;
Dictionary used from language 2 to language 3:
LP -> VX; MK -> RW; LI -> XR; NO -> RR; MO -> UQ; IL -> WV; OK -> QW; PM -> RU; JI -> VS;
IM -> XW; OI -> QU; NP -> RQ; IN -> QX; NL -> RS; IJ -> QT; NJ -> ST; LO -> UV; PJ -> TX;
JL -> QR; PK -> WR; NM -> SU; MI -> QS; PO -> SS; KK -> UU; PL -> XU;
Dictionary used from language 3 to language 4:
XU -> Yd; RW -> ac; TQ -> YY; UR -> ZY; TX -> ed; WW -> eY; UX -> ff; XR -> bY; QQ -> bc;
SS -> cc; VU -> Zd; RS -> fa; VW -> Ye; RU -> fc; US -> ca; UW -> cf; WX -> dY; RQ -> cz;
VQ -> Zf; WS -> bd; VR -> aa; RX -> fZ; UQ -> dc; QV -> dz; SW -> Yb;
Dictionary used from language 4 to language 5:
ca->mj; bY->nk; cf->km; cb->ln; Zd->gh; dc->ng; bf->ik; fz->ii; ZY->gl;
Ya->lll; be->lii; YY->mg;dZ->gg;dce->kl; fc->jg; ff->kn;bZ->nh; ea->im;
ce->jh; Yd->ih; Yb->hi; Yf->hk;aY->kj;a a->mk;cY->hm;
Dictionary used from language 5 to language 6:
kg->ps; hh->rq;jm->oo;ih->qs;l m->po;i j->oq;l j->qu;k m->vu; gh->pp;
gn->vp;n h->ss;m m->so;m k->vt;m i->ut; hk->to;n k->ru;j n->tu; gl->ts;
jg->vv; hg->us;n l->su;l k->qo; hl->tp;i i->pq;k i->ro;l h->up;
Now please perform language_task_2 translation from the following sequence in language 1 to language 6. Do not use code! You must only reponse in the form: "Sequence in language 1: [the sequence in language 1]; Sequence in language 2: [the sequence in language 2]; Sequence in language 3: [the sequence in language 3]; Sequence in language 4: [the sequence in language 4]; Sequence in language 5: [the sequence in language 5]; Sequence in language 6: [the sequence in language 6]". The sequence you need to translate from language 1 is: CBEFEBDECBCAHEFBCADFGBDGHED E.<|eot_id|><|start_header_id|>assistant<|end_header_id|>" 
```

LABEL:   
```txt
"<|begin_of_text|>The translation result is: Sequence in language 1: C B E F E B D E C B C A H E F B C A D F G B D G H E D E; Sequence in language 2: K I M L I P K P O I L J L M O I J I J O I P L N P O K P * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * / d Y a c f a Y b Z f f c b c Y Y Y Y f f Z Y b c e Y f Z * * * * * * * * * ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** 
```  
Figure 16. Input with complete context and explicit chain-of-thought. We use this data format at the beginning of stage 1 training.

INPUT:   
```ocaml
"<|begin_of_text|> <|begin_of_text|> <|begin_of_text|> <|start_header_id|> user <|end_header_id |

You are performing a special translation task called language_task_2. The subset of dictionaries used are as follows:
Dictionary used from language 1 to language 2:
DA -> NJ; AD -> JI; BC -> OI; CC -> NN; HE -> PK; FE -> ML; GH -> LN; EG -> PJ; AF -> KM;
EC -> KP; BE -> KI; FD -> MN; ED -> PO; BA -> PL; BD -> IP; DG -> II; DH -> IO; EF -> LM;
HB -> JK; EA -> KJ; AH -> LJ; GC -> IK; FB -> PI; FG -> JO;
Dictionary used from language 2 to language 3:
LP -> VX; MK -> RW; LI -> XR; NO -> RR; MO -> UQ; IL -> WV; OK -> QW; PM -> RU; JI -> VS;
IM -> XW; OI -> QU; NP -> RQ; IN -> QX; NL -> RS; IJ -> QT; NJ -> ST; LO -> UV; PJ -> TX;
JL -> QR; PK -> WR; NM -> SU; MI -> QS; PO -> SS; KK -> UU; PL -> XU;
Dictionary used from language 3 to language 4:
XU -> Yd; RW -> ac; TQ -> YY; UR -> ZY; TX -> ed; WW -> eY; UX -> ff; XR -> bY; QQ -> bc;
SS -> cc; VU -> Zd; RS -> fa; VW -> Ye; RU -> fc; US -> ca; UW -> cf; WX -> dY; RQ -> cz;
VQ -> Zf; WS -> bd; VR -> aa; RX -> fZ; UQ -> dc; QV -> ddZ; SW -> Yb;
Dictionary used from language 4 to language 5:
ca->mj; bY->nk; cf->km; cb->ln; Zd->gh; dc->ng; bf->ik; fz->ii; ZY->gl;
Ya->lll; be->lii; YY->mg;dZ->gg;dce->kl; fc->jg; ff->kn;bZ->nh; ea->im;
ce->jh; Yd->ih; Yb->hi; Yf->hk;aY->kj;a a->mk;cY->hm;
Dictionary used from language 5 to language 6:
kg->ps; hh->rq;jm->oo;i h->qs;l m->po;i j->oq;l j->qu;k m->vu; gh->pp;
gn->vp;n h->ss;m m->so;m k->vt;m i->ut; hk->to;n k->ru;j n->tu; gl->ts;
jg->vv; hg->us;n l->su;l k->qo; hl->tp;i i->pq;k i->ro;l h->up;
Now please perform language_task_2 translation from the following sequence in language 1 to language 6. Do not use code! You must only reponse in the form: "Sequence in language 1: [the sequence in language 1]; Sequence in language 2: [the sequence in language 2]; Sequence in language 3: [the sequence in language 3]; Sequence in language 4: [the sequence in language 4]; Sequence in language 5: [the sequence in language 5]; Sequence in language 6: [the sequence in language 6]". The sequence you need to translate from language 1 is: CBEFEBDECBCAHEFBCADFGBDGHED E.<|eot_id|><|start_header_id|>assistant<|end_header_id|>" 
```

LABEL:   
```html
"<|begin_of.text|>The translation result is: Sequence in language 1: C B E F E B D E C B C A H E F B C A D F G B D G H E D E; Sequence in language 2: <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> </td><td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
</td></tr>
<|content_end|> 
```  
Figure 17. Input with complete context and internalized chain-of-thought. We use this data format by the end of stage 1 training as well as the base format for stage 2 training (before dropout)

INPUT:   
```txt
"<|begin_of_text|> <|begin_of_text|> <|begin_of_text|> <|start_header_id|> user <|end_header_id|>
You are performing a special translation task called language_task_2. The subset of dictionaries used are as follows:
Dictionary used from language 1 to language 2: ;
Dictionary used from language 2 to language 3: ;
Dictionary used from language 3 to language 4: ;
Dictionary used from language 4 to language 5: ;
Dictionary used from language 5 to language 6:;
Now please perform language_task_2 translation from the following sequence in language 1 to language 6. Do not use code! You must only reponse in the form: "Sequence in language 1: [the sequence in language 1]; Sequence in language 2: [the sequence in language 2]; Sequence in language 3: [the sequence in language 3]; Sequence in language 4: [the sequence in language 4]; Sequence in language 5: [the sequence in language 5]; Sequence in language 6: [the sequence in language 6]". The sequence you need to translate from language 1 is: C B E F E B D E C B C A H E F B C A D F G B D G H E D E.<|eot_id|><|start_header_id|>assistant<|end_header_id|>" 
```

LABEL:   
```txt
"<|begin_of_text|>The translation result is: Sequence in language 1: C B E F E B D E C B C A H E F B C A D F G B D G H E D E; Sequence in language 2: <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> <T> </td></tr><tr><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><td><table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr></table></tr>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t>
</t> 
```  
Figure 18. Input with completely masked context and internalized chain-of-thought. We use this data format to evaluate the model's capability on conducting translation without anything (useful information) in context.