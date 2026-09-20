# MASTERING SYMBOLIC OPERATIONS: AUGMENTING LANGUAGE MODELS WITH COMPILED NEURAL NETWORKS

Yixuan Weng $^{1*}$ , Minjun Zhu $^{1,2,*}$ , Fei Xia $^{1,2}$ , Bin Li $^{3}$ , Shizhu He $^{1,2,\otimes}$ , Kang Liu $^{1,2,\otimes}$ , Jun Zhao $^{1,2}$

$^{1}$ The Laboratory of Cognition and Decision Intelligence for Complex Systems, IA, CAS   
$^{2}$ School of Artificial Intelligence, University of Chinese Academy of Sciences   
$^{3}$ College of Electrical and Information Engineering, Hunan University

wengsyx@gmail.com, {shizhu.he, kliu, jzhao}@nlpr.ia.ac.cn

https://github.com/wengsyx/Neural-Comprehension

# ABSTRACT

Language models' (LMs) proficiency in handling deterministic symbolic reasoning and rule-based tasks remains limited due to their dependency implicit learning on textual data. To endow LMs with genuine rule comprehension abilities, we propose "Neural Comprehension" - a framework that synergistically integrates compiled neural networks (CoNNs) into the standard transformer architecture. CoNNs are neural modules designed to explicitly encode rules through artificially generated attention weights. By incorporating CoNN modules, the Neural Comprehension framework enables LMs to accurately and robustly execute rule-intensive symbolic tasks. Extensive experiments demonstrate the superiority of our approach over existing techniques in terms of length generalization, efficiency, and interpretability for symbolic operations. Furthermore, it can be applied to LMs across different model scales, outperforming tool-calling methods in arithmetic reasoning tasks while maintaining superior inference efficiency. Our work highlights the potential of seamlessly unifying explicit rule learning via CoNNs and implicit pattern learning in LMs, paving the way for true symbolic comprehension capabilities.

# 1 INTRODUCTION

Language models (LMs), particularly large language models (LLMs), have exhibited impressive performance on complex reasoning tasks (Brown et al., 2020; Zhang et al., 2022; Chowdhery et al., 2022; Wei et al., 2022a; Suzgun et al., 2022). Despite this, the proficiency of LMs in tackling deterministic symbolic reasoning and rule-based tasks is still limited (Welleck et al.; Razeghi et al., 2022). For example, GPT-3's arithmetic performance declines with higher digit numbers (Brown et al., 2020), and its mathematical accuracy is influenced by word frequency in training data (Razeghi et al., 2022). Moreover, length generalization (Anil et al., 2022) remains a challenge even for 100-billion-parameter models, such as GPT-4 (Bubeck et al., 2023). We hypothesize that these limitations stem from LMs' dependency on implicitly learning rules from textual data. As illustrated in Figure 1, a simple length generalization experiment using addition tasks with varying numbers of digits highlights this limitation. Performance deteriorates as test length increases, indicating that these models strongly rely on statistical patterns in the data rather than capturing fundamental logical structures. This implicit learning of statistical patterns constrains LMs' accuracy in executing symbolic operations tasks.

![](images/66b6ffb8b655c20987c5df7a5d9de9b7f2a148e0391fda4e4ce7d940cc9f8549.jpg)

<details>
<summary>line</summary>

| Enter Text Length (Digits) | GPT-4 Solve Rate (%) | GPT-3.5 Solve Rate (%) | T5-base Solve Rate (%) |
| --------------------------- | -------------------- | ---------------------- | ---------------------- |
| 3                           | 100                  | 100                    | 100                    |
| 10                          | ~95                  | ~85                    | ~20                    |
| 20                          | ~90                  | ~60                    | ~5                     |
| 30                          | ~85                  | ~40                    | ~5                     |
</details>

Figure 1: The length generalization of T5 (with fine-tune), GPT-3.5 and GPT-4 (with few-shot) on symbolic operations (Addition) tasks. To evaluate the model's proficiency, we conducted experiments on tasks ranging from 3 to 30 digits, with longer than 10 digits being out-of-distribution of training data.

We propose a transformer-based language model framework, termed "Neural Comprehension", which synergistically integrates a pretrained LM (Li et al., 2021b) and compiled neural networks (CoNNs) (Weiss et al., 2021), combines their complementary strengths in a plug-and-play manner, to achieve high accuracy and robust performance. CoNNs are neural networks but the rules are explicitly coded through transformer-liked structures and attention. Therefore, the CoNN is human-controllable, executing rules through artificially generated attention weights, and can achieve perfect accuracy once compiled network is done. Neural Comprehension relies solely on neural networks without requiring additional tools. It employs a token-by-token generative method, analogous to GPT-3, where each token can be generated by either the pretrained LM or one of the CoNNs. The Neural Comprehension comprises a pretrained LM and multiple sets of CoNNs. The implementation of the Neural Comprehension framework facilitates the integration of rule-intensive abilities and reasoning capabilities into LMs, endowing them with genuine symbolic comprehension skills.

We conduct extensive experiments to evaluate the performance of our proposed Neural Comprehension method on a variety of rule-intensive tasks. Our experimental results demonstrate the effectiveness of our approach in comparison with existing state-of-the-art techniques, such as vanilla fine-tuning, few-shot learning, and Chain-of-Thought reasoning (Wei et al., 2022b). Specifically, Neural Comprehension outperforms these methods in terms of accuracy, efficiency, and interpretability, showcasing its superiority in handling rule-intensive tasks. On the other hand, compared to the Tool-Based method (Mialon et al., 2023), Neural Comprehension provides a unified end-to-end neural network framework, eliminating the need for external interpreters and having higher inference efficiency. Historically, LMs are far from mastering robust symbolic task, such as symbolic operations and arithmetic reasoning (Stolfo et al., 2023). Our research provides a compelling case for LMs in neural network frameworks mastering symbolic operations, highlighting its potential to transform the landscape of symbolic reasoning and numerical computation capabilities for LMs.

# Contributions Our main contributions are as follows:

- We pioneer the implementation of flawless execution rule-intensive symbolic operations for language models that rely on neural networks. By employing a plug-and-play way, we successfully integrate CoNNs, which are interpretable and human-controllable, into the language model. Our method facilitates direct rule deduction without the need for learning from conditional probabilities, leading to a more robust and effective approach. (Section 4)   
- We have built the AutoCoNN toolkit to make Neural Comprehension scalable, which leverages LLMs' contextual learning capabilities to automatically generate new CoNNs. Our method can be easily extended to various symbolic operations tasks. (Seciton 4.3)   
- Our experimental results on symbolic tasks and real-world arithmetic reasoning tasks demonstrate the superior performance of our method in comparison to existing techniques. Notably, our LM achieves flawless execution on symbolic reasoning tasks. (Section 5.1 5.2)   
- It is worth noting that tool-based methods are only applicable to language models with code generation capabilities and require the cooperation of external interpreters. Our experiments demonstrate that the symbolic processing capabilities of neural understanding are on par with tool-based methods, but are applicable to models ranging from small ones like T5-Small (60M) to large ones like GLM-130B (130B) and GPT-3.5 (175B). (Section 5.3)   
- We also studied the potential of combining multiple CoNNs and found that adding correlated CoNNs can continuously increase performance, while adding uncorrelated CoNNs rarely leads to performance degradation. This provides a new approach for model fusion, enabling the model to easily acquire new knowledge. (Section 5.4)

# 2 RELATED WORKS

Pretrained Language Models encompass those trained on general-purpose corpora (Lewis et al., 2019; Scao et al., 2022; Sun et al., 2022) and specialized symbolic tasks (Geva et al., 2020; Lewkowycz et al., 2022; Yang et al., 2023). They primarily aim to capture statistical patterns in language, which limits their capacity for symbolic reasoning. Symbolic reasoning involves manipulating abstract symbols and logical rules to derive new knowledge (Shindo et al., 2021; Yang & Deng, 2021) and necessitates the ability to extrapolate to novel situations and reason about concepts

absent in the training data (Fujisawa & Kanai, 2022). Due to the constraints of gradient learning, LMs face challenges in wholly solving symbolic problems (Stolfo et al., 2023).

In-Context Learning has emerged as a promising approach to address these challenges (Dong et al., 2022) and closely approximate the predictors computed by gradient descent (Akyürek et al., 2022). By prompting the language model to generate an explanation before generating an answer, the chain of thought (Wei et al., 2022b; Kojima et al., 2022; Zhang et al., 2023; Zhou et al., 2022a) encourages the model to think sequentially. This technique has been employed in various numerical and symbolic reasoning tasks, such as scratchpad prompting (Nye et al., 2021) for length generalization (Anil et al., 2022) and utilizing the chain of thought to perform arithmetic operations like summing pairs of single digits with carry (Zhou et al., 2022b). However, this approach often necessitates substantial computational resources, and achieving perfect accuracy remains challenging.

Augmented Language Models have been proposed as an alternative, supplementing language models with external tools (Mialon et al., 2023). Examples include generating Python code for numerical reasoning (Gao et al., 2022; Chen et al., 2022) or incorporating tool usage as a pretraining task (Schick et al., 2023). However, using external tools lacks a unified framework with language models and instead relies on the normativity of program generation.

# 3 PRELIMINARIES

Example 1 The Parity CoNN

If we want the transformer to perform the Parity task full accurately:

Input: 1 0

1. Select(Indices, Indices, True) = $\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$   
2. Aggregate $\left( \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}, [1, 0] \right) = [1, 1]$   
3. Zipmap([1 1], Lambda x : 0 if x % 2 == 0 else 1) = [1 1]   
Output: [1 1]

Figure 2: Demonstration of the principles of Parity CoNN.

Compiled Neural Network (CoNN). CoNN is a transformer-based neural network leveraging artificially compiled attention weights to execute rules. CoNN has multiple attention layers and Multi-Layer Perceptron (MLP) layers, and each attention layer facilitates interactions between tokens. For example, in Figure 2, the multiplication of query and key elements representing a "Select" operation in CoNN. Subsequent multiplication with value elements indicates an "Aggregate" operation. The MLP layer is responsible for the token itself and is referred to as the "Zipmap" operation (Weiss et al., 2021). By utilizing the three operations (Select, Aggregate, and Zipmap) to represent the sequence-to-sequence process, we can convert symbolic parity task into transformer weights (Lindner et al., 2023). CoNN can also stack multiple layers to address various human-defined rule problems, such as mathematical calculations and symbol operations $^{1}$ . Figure 2 shows an example of Parity CoNN: The first step is to obtain a matrix of all ones to that of the sequence using the "Select" operation. Secondly, the "Aggregate" operation is used to combine the matrix obtained in the previous step with the input sequence (with the aim of calculating the total number of 0's and 1's in the sequence). The third step involves determining whether the total count is odd or even by "Zipmap".

# 4 METHOD

Language models excel in language understanding tasks, but lack robust capabilities for symbolic tasks. We propose a Neural Comprehension framework that make CoNN guided by abstract rules into language models in a plug-and-play fashion, which integrates the language model's implicit learning parameters and CoNNs' explicit learning parameters. In Neural Comprehension, we designed CoNNs in neural networks with weights and structures directly encoding the symbolic rules within the standard architecture of the LM to enhance deterministic rule comprehension and allow for deterministic execution of rule-based tasks.

![](images/fdc72cda7aff04744233878e09960073043d7fa0e01617af9dcc2a0817151213.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Input
        A["LMCON"] --> B["LMCON"]
        C["216582"] --> A
        D["="] --> B
        E["147843"] --> B
        F["meters"] --> B
        G["-"] --> A
        H["216582"] --> A
        I["216582"] --> J["216582"]
        K["="] --> L["LMCON"]
        M["147843"] --> N["LMCON"]
    end

    subgraph Output
        O["To find out how much farther Stanley ran than walked, we need to subtract the distance he walked from the distance he ran 364425 - 216582 = 147843 meters. Therefore, Stanley ran 147843 meters farther than he walked."]
    end

    subgraph Input
        P["Pre-training Neural Network"]
        Q["Compiled Neural Network"]
    end

    subgraph Language Models with Compiled Neural Networks
        R["Two neural networks are similar to the decoder structure of transformer"]
        S["Language Models with Compiled Neural Networks"]
    end

    style Input fill:#f9f,stroke:#333
    style Output fill:#ccf,stroke:#333
```
</details>

Figure 3: The architecture of the proposed Neural Comprehension framework.

# 4.1 NEURAL COMPREHENSION

Neural Comprehension is a MoE-style (Shazeer et al., 2017) neural network framework we designed, which is entirely composed of neural networks and maintains the generative seq2seq style. It uses predefined CoNNs as gating to determine when to output results from CoNNs. This approach is simple and plug-and-play, allowing combination with pretrained LMs. As illustrated in Figure 3, the language model encodes the context and produces the textual and reasoning process context $D(x)$ step by step, a decoder architecture to generate the subsequent context step by step iteratively. And CoNNs handle sequence transformations involving rules, when a rule-required operation emerges, CoNN's attention is utilized to calculate specific values. For example, in Figure 3, when calculating 364425-216582, the only pretrained language model may output 148843, which is incorrect. However, the Subtraction CoNN can correct the result to 147843 in the neural comprehension framework. This dynamically encoded process improves intermediate results interpretability and final result accuracy. Neural Comprehension combines LM and CoNNs in a piecewise function to perform gradient update. LM's hidden state output is $H_{L} = \left(H_{L_1} \cdots H_{L_{d_L}}\right)^{\top} \in \mathbb{R}^{d_L}, \quad H_{L_i} \in (0,1)$ , and CoNN's output is $H_{C} = \left(H_{C_1} \cdots H_{C_{d_C}}\right)^{\top} \in \mathbb{R}^{d_C}, \quad H_{C_i} \in \{0,1\}$ ,

$$
\hat {i} = \underset {i} {\operatorname{argmax}} \left[ \binom{I _ {d _ {L}}, 0}{0, \beta I _ {d _ {C}}} \binom{H _ {L},}{H _ {C}} \right], \quad \beta \in \{0, 1 \} \tag {1}
$$

where $H_{C}$ is a one-hot vector, and $d_{L}$ and $d_{C}$ here refer to the vocabulary size of the Model's decode output. $^{2}$ The Neural Comprehension combines the LLM's hidden state output, $H_{L}$ , and CoNN's output, $H_{C}$ , using identity matrices $I_{d_{L}}$ (for $d_{L}$ ) and $I_{d_{C}}$ (for $d_{C}$ ) to concatenate them for model fusion. Specifically, the hidden state representation matrix is obtained by extending the original hidden state representation matrix of the LM with the hidden state matrix on the CoNNs' vocabulary through a block matrix operation, resulting in a larger matrix.

# 4.2 GRADIENT MODIFICATION IN NEURAL COMPREHENSION

To better appreciate the benefits of our method in handling rule-intensive tasks and improving accuracy, it is crucial to understand the gradient perspective of In-Context Learning (ICL). Recent studies on ICL algorithms have shown that the learning process of language models within the optimization process in ICL can be viewed as a search for suitable gradients to minimize the loss

function. (Garg et al., 2022; Von Oswald et al., 2023). Due to the implicit learning nature of standard ICL methods, gradients learned from data may not always be ideal for addressing rule-intensive tasks. Therefore, our proposed method introduces an explicit learning component to provide more appropriate gradient updates for such tasks, ultimately leading to enhanced overall performance. We focus on elucidating the changes in the gradient introduced by the Neural Comprehension model, the gradient of the model during the execution of ICL can be partitioned into two categories based on the origin of the gradients:

$$
\text { Gradient } = \left\{ \begin{array}{l l} G _ {T} & \text { Text:   Language   Model } \\ G _ {R} & \text { Rule:   CoNNs } \end{array} \right. \tag {2}
$$

Here, $G_{T}$ represents the gradients derived implicitly from the language model (LM) and corresponds to the text-based learning aspect of the model. Conversely, $G_{R}$ represents the gradients explicitly derived from the CoNNs, encoding rule-based knowledge. The specific computation process can be seen in Equation 1. Note that the gradients' decomposition is only approximate and may not reflect the true generating process of text. The Neural Comprehension framework integrates both gradient sources to optimize the ICL process. In linear regression problems (Akyürek et al., 2022), the loss function can be expressed as a piecewise function according to Equation 1, here $P_{1}(x)$ is the LM and $P_{2}(x)$ is CoNN, the In-context-learner can be separate into two process:

$$
L = \left\| y - \beta^ {\top} x \right\| ^ {2} \tag {3}
$$

$$
= \left\{ \begin{array}{l l} \left\| y - G _ {T} ^ {\top} x \right\| ^ {2} & x \in P _ {1} (x) \\ \left\| y - G _ {R} ^ {\top} x \right\| ^ {2} & x \in P _ {2} (x) \end{array} \right. \tag {4}
$$

Based on the partitioned gradient as defined in Equation 2, the overall gradient of the Neural Comprehension model can be obtained by computing their individual gradients concerning the respective $\beta$ :

$$
\underbrace {\frac {\partial L}{\partial \beta}} _ {\text { Gradient }} = \left\{ \begin{array}{l l} \frac {\partial L}{\partial G _ {T}} & x \in P _ {1} (x) \\ \frac {\partial L}{\partial G _ {R}} & x \in P _ {2} (x) \end{array} \right. \tag {5}
$$

This partitioning allows the Neural Comprehension model to specifically address the gradient requirements of both implicit learning via LM and explicit learning via CoNNs. It is crucial to note that CoNNs are designed to minimize the loss associated with rule-based tasks, essentially providing an optimal gradient for tasks involving rule-intensive operations. This leads to a substantial improvement in the model's accuracy for rule-based tasks, as the gradient updates provided by CoNNs are more suitable for rule learning compared to the initially available gradients from the LM. By amalgamating both of gradient sources, the Neural Comprehension model achieves a more refined optimization of in-context learning. The Neural Comprehension model effectively balances the need for implicit and explicit learning within the ICL framework, leading to enhanced overall performance in terms of accuracy and interpretability.

# 4.3 AutoCONN TOOLKIT

To improve the scalability of Neural Comprehension frameworks, we propose the AutoCoNN toolkit, which can automatically generate new CoNNs and adapt Neural Comprehension given by "Instruct" and "Example", where "Instruct" serves as explicit symbolic definitions, and "Example" provides some input-output pairs for the operation. Considering that LLMs like GPT-4 can describe symbolic reasoning processes but cannot faithfully execute them (Cai et al., 2023), AutoCoNN Toolkit enables converting the reasoning process of symbols into CoNNs while maintaining a complete neural network framework without extra interpreters. The AutoCoNN process is divided into three steps. First, we provide 24 Tracr code (Lindner et al., 2023) examples as context. Then the LLM is asked to generate 20 different Tracr codes by sampling decoding based on the "Instruct" and "Example". Finally, we convert these codes into pytorch-form (Paszke et al., 2019) CoNNs and filter them on the pre-provided Examples to obtain accurate CoNNs. We provide further experimental analysis to test the performance of AutoCoNN in Appendix D. Meanwhile, the Parity, Reverse, Copy and Last Letter CoNN mentioned in the Section 5 are constructed by AutoCoNN, which demonstrates the practical value of AutoCoNN.

# 5 EXPERIMENTS AND RESULT

5.1 SYMBOLIC TASKS   
![](images/c910442cb38d527bf1e913ea683b9ce43c1a4e39e936c0e684701f95cbae2203.jpg)

<details>
<summary>line</summary>

| Parity | T5-small I | T5-large | GPT-3 | Scratchpad (T5-base) | T5-base | GPT-3.5 | GLM | Neural Comprehension |
| ------ | ---------- | -------- | ----- | -------------------- | ------- | ------- | --- | ------------------- |
| 1      | 100        | 100      | 70    | 100                  | 100     | 50      | 50  | 100                 |
| 5      | 90         | 80       | 40    | 95                   | 85      | 45      | 45  | 100                 |
| 10     | 85         | 75       | 50    | 90                   | 80      | 50      | 50  | 100                 |
| 15     | 80         | 70       | 60    | 85                   | 75      | 55      | 55  | 100                 |
| 20     | 75         | 65       | 55    | 80                   | 70      | 60      | 60  | 100                 |
| 25     | 70         | 60       | 50    | 75                   | 65      | 65      | 65  | 100                 |
| 30     | 65         | 55       | 45    | 70                   | 60      | 70      | 70  | 100                 |
| 35     | 60         | 50       | 40    | 65                   | 55      | 75      | 75  | 100                 |
| 40     | 55         | 45       | 35    | 60                   | 50      | 80      | 80  | 100                 |
</details>

![](images/7fa7bfaad7331ceac8ba5c088cd1f2bb4f3a6f0f79aef12933144353585971b6.jpg)

<details>
<summary>line</summary>

| Reverse | T5-small | T5-large | GPT-3 | T5-base | GPT-3.5 | GLM | Neural Comprehension |
| ------- | -------- | -------- | ----- | ------- | ------- | --- | ------------------- |
| 1       | 0        | 0        | 90    | 0       | 100     | 0   | 100                 |
| 5       | 30       | 40       | 10    | 20      | 80      | 0   | 100                 |
| 10      | 40       | 60       | 10    | 40      | 60      | 0   | 100                 |
| 15      | 50       | 70       | 10    | 60      | 40      | 0   | 100                 |
| 20      | 60       | 80       | 10    | 70      | 20      | 0   | 100                 |
| 25      | 55       | 75       | 10    | 65      | 10      | 0   | 100                 |
| 30      | 55       | 75       | 10    | 65      | 10      | 0   | 100                 |
| 35      | 55       | 75       | 10    | 65      | 10      | 0   | 100                 |
| 40      | 55       | 75       | 10    | 65      | 10      | 0   | 100                 |
</details>

![](images/7f2bd80c505e77b5ee1a3736f63ffd43904982a0f0781b962486e588f430d667.jpg)

<details>
<summary>line</summary>

| Addition | T5-small | T5-large | GPT-3 | Algorithm (GPT-3.5) | T5-base | GPT-3.5 | GLM | Neural Comprehension |
| -------- | -------- | -------- | ----- | ------------------- | ------- | ------- | --- | -------------------- |
| 3        | 0        | 0        | 20    | 100                 | 0       | 0       | 0   | 100                  |
| 5        | 0        | 0        | 100   | 100                 | 0       | 0       | 0   | 100                  |
| 10       | 0        | 90       | 0     | 100                 | 90      | 90      | 0   | 100                  |
| 15       | 0        | 80       | 0     | 100                 | 80      | 80      | 0   | 100                  |
| 20       | 0        | 60       | 0     | 100                 | 60      | 60      | 0   | 100                  |
| 25       | 0        | 40       | 0     | 100                 | 40      | 40      | 0   | 100                  |
| 30       | 0        | 20       | 0     | 100                 | 20      | 20      | 0   | 100                  |
| 35       | 0        | 10       | 0     | 100                 | 10      | 10      | 0   | 100                  |
| 40       | 0        | 5        | 0     | 100                 | 5       | 5       | 0   | 100                  |
</details>

![](images/4a55aea7cfaf37e25a1646e045738fb26da4b22e5607da01d2df95f620c9c931.jpg)

<details>
<summary>line</summary>

| Subtraction | T5-small1 | T5-large | GPT-3 | T5-base | GPT-3.5 | GLM | Neural Comprehension |
| ----------- | --------- | -------- | ----- | ------- | ------- | --- | -------------------- |
| 3           | 0         | 0        | 70    | 0       | 0       | 0   | 100                  |
| 5           | 0         | 20       | 20    | 0       | 0       | 0   | 100                  |
| 10          | 0         | 80       | 60    | 0       | 0       | 0   | 100                  |
| 15          | 0         | 70       | 50    | 0       | 0       | 0   | 100                  |
| 20          | 0         | 40       | 30    | 0       | 0       | 0   | 100                  |
| 25          | 0         | 30       | 20    | 0       | 0       | 0   | 100                  |
| 30          | 0         | 10       | 10    | 0       | 0       | 0   | 100                  |
| 35          | 0         | 5        | 5     | 0       | 0       | 0   | 100                  |
| 40          | 0         | 2        | 2     | 0       | 0       | 0   | 100                  |
</details>

Figure 4: Comparison of Neural Comprehension and other implicit learning-based methods in symbolic operations tasks to test length generalization performance. In this, the T5 model uses the Vanilla Fine-tune method for learning, and LLMs use the Few-shot learning method. In Neural Comprehension, each task has a different CoNN, namely Parity, Reverse, Addition, and Subtraction.

<table><tr><td>Techniques</td><td>In-distribution</td><td>Out-of-distribution</td><td>Time and Space Complexity</td><td>Interpretability</td></tr><tr><td>Vanilla Fine-tune (For LM)</td><td>✓✓</td><td>✗</td><td>✓✓</td><td>✗</td></tr><tr><td>Vanilla Few-shot (For LLM)</td><td>✓</td><td>✓</td><td>✓✓</td><td>✗</td></tr><tr><td>Scratchpad (Anil et al., 2022)</td><td>✓✓</td><td>✓</td><td>✗</td><td>✓</td></tr><tr><td>Algorithmic (Zhou et al., 2022b)</td><td>✓✓</td><td>✓</td><td>✗</td><td>✓</td></tr><tr><td>Neural Comprehension (Ours)</td><td>✓✓</td><td>✓✓</td><td>✓✓</td><td>✓✓</td></tr></table>

Table 1: Performance on Symbolic operations tasks of five techniques that language models admit: (1) Vanilla Finetuning, (2) Vanilla Few-shot, (3) Scratchpad (Chain-of-Thought reasoning), (4) Algorithmic (Chain-of-Thought reasoning) and (5) Neural Comprehension. We find that the first four learning-based methods have different modes of failure regarding in and out-of-distribution coverage for symbolic operations. However, Neural Comprehension has strong advantages in terms of length generalization, efficiency, and interpretability. ✗ signifies poor √ signifies nontrivial, √√ signifies near-perfect performance. (\*) Refers to task-dependency.

We conduct a length generalization experiment (Anil et al., 2022) to examine the distinctions between the Neural Comprehension and learning-based methods, as depicted in Figure 4. Our experimental design encompasses $1000 \times 40$ independent test sets, comprising problems with varying digit lengths from 1 to 40 digits. 10 to 20 digits within the range are provided by us for methods based on implicit learning for training; during the testing phase, this range is called In-Dist. Furthermore, we present results for both Scratchpad (Anil et al., 2022) and Algorithmic (Zhou et al., 2022b) approaches.

The results of our experiment demonstrate that the Vanilla Fine-tune (red lines) method performs optimally on the in-domain (10-20 digit) training set, while its performance deteriorates for both more simplistic and more intricate. This finding suggests that the absence of relevant samples in the training set may cause gradient descent-based language models to underperform on both simpler and more complex tasks. As further discussed in the appendix D.1, this phenomenon can be attributed to the inherent generalization limitations of statistical models and the position bias of language models.

Considering the Vanilla Few-shot method (green lines), we determine that its performance is less impacted by the prompt sample range compared to Vanilla Fine-tune. Large language models, which are trained on extensive text corpora, excel at solving more straightforward problems such as symbolic operations within a ten-digit range. Nevertheless, performance remains below par for test sets with more than ten digits, even when prompted with 10-20 digit samples.

Observing CoT-like methods (we use GPT-3.5), including Scratchpad and Algorithmic, unveils their robust length generalization capabilities. Scratchpad works by requiring large language models to record intermediate steps, while Algorithmic employs a similar approach to record the carry operations involved in the addition process. This can be primarily attributed to their proficiency in decomposing complex problems into smaller incremental steps and maintaining intermediate states. However, these methods necessitate substantial computational resources, and extending the length beyond the input limit of the model becomes challenging.

Our study reveals that Neural Comprehension attains remarkably high accuracy in symbolic operations. This implies that Neural Comprehension, unlike conventional methods, does not rely on training data and remains unaffected by discrepancies in input lengths for in-distribution and out-of-distribution data. Consequently, it alleviates the requirement for step-by-step work tracking, and language models with CoNNs only need relatively fewer computational steps to execute sequence operations directly. Encoding rules into neural network modules endows us with greater interpretability, enabling language models to flawlessly perform purely symbolic operation tasks.

# 5.2 SYMBOLIC REASONING

In this section, we investigate the performance of Neural Comprehension in terms of symbolic reasoning capabilities. Our hypothesis is that, although pretrained Language Models (LMs) demonstrate strong language understanding abilities, they lack the capacity to deduce and comprehend rules regarding symbolic reasoning tasks. Thus, we aim to evaluate whether the incorporation of compiled neural networks in the form of CoNNs can address this limitation and improve the LM's symbolic reasoning abilities.

To assess the performance of the rule comprehension component (CoNNs) in symbolic reasoning, we devise an experiment that measures the model's accuracy using intermediate processes and represents them in a "Chain of Thought"-like manner. In doing so, the experiment decomposes language understanding and rule comprehension explicitly into simpler outputs, avoiding the complexities of reasoning and additional error propagation in the models. Example outputs from this approach can be found in Appendix F. We observed that neural comprehension improves the symbolic reasoning capabilities of pretrained language models in most cases (Neural Comprehension almost always outperforms Vanilla Fine-tune in Figure 5), and can fit faster. This observation suggests that the introduction of compiled neural networks has a positive impact on pretrained LMs, addressing rule comprehension limitations in symbolic reasoning tasks.

![](images/9f1d31737d574f163198da72deca950ffea5e9274c20b660ab70dac967a99e7e.jpg)  
Figure 5: In the iterative process of gradient descent during training. The bleu line represents a language model that incorporates neural comprehension, and the red line represents the original language model. Additionally, we provide Direct, which is a direct prediction of the final result, as a reference.

# 5.3 ARITHMETIC REASONING

Arithmetic reasoning serves as a suitable testbed for evaluating language models and their ability to address real-world problems. In this study, we examine the AddSub $^{+}$ dataset variants that involve different digit lengths, utilizing the Addition and Subtraction models from the CoNNs family. Notably, the capabilities of Neural Comprehension extend beyond these tasks, as CoNNs can also simulate calculators that support multiplication and division operations, and potentially perform

![](images/7eebd36ae1346aae4ed28ba79b6367bc8e188ab64f27117de05e592d4b78fca6.jpg)  
- Vanilla CoT ---- PAL - Neural Comprehension Improve Performance

Figure 6: We conducted simulations of the AddSub dataset with varying digits by modifying the "lEquations" parameter. We then tested the performance of three LLMs with and without Neural Comprehension in generating CoT outputs for $\mathrm{AddSub^{+}}$ . And we reported the solve rates of three LLMs and compared the solve rates of using additional tools (PAL (Gao et al., 2022)). 

<table><tr><td>Addition</td><td>llama-2-7b</td><td>llama-2-70b</td><td>GLM-130B</td><td>Avg</td></tr><tr><td>CoT</td><td>6.3</td><td>43.8</td><td>6.3</td><td>18.8</td></tr><tr><td>PAL</td><td>18.7</td><td>37.5</td><td>6.3</td><td>20.8</td></tr><tr><td>NC (Ours)</td><td>12.5</td><td>43.8</td><td>7.2</td><td>21.2</td></tr></table>

<table><tr><td>Subtraction</td><td>llama-2-7b</td><td>llama-2-70b</td><td>GLM-130B</td><td>Avg</td></tr><tr><td>CoT</td><td>6.3</td><td>27.9</td><td>1.8</td><td>12.0</td></tr><tr><td>PAL</td><td>7.2</td><td>31.5</td><td>3.6</td><td>14.1</td></tr><tr><td>NC (Ours)</td><td>9.9</td><td>32.4</td><td>4.5</td><td>15.6</td></tr></table>

<table><tr><td>Mixed</td><td>llama-2-7b</td><td>llama-2-70b</td><td>GLM-130B</td><td>Avg</td></tr><tr><td>CoT</td><td>11.1</td><td>16.7</td><td>0.0</td><td>9.3</td></tr><tr><td>PAL</td><td>11.1</td><td>27.8</td><td>5.6</td><td>14.8</td></tr><tr><td>NC (Ours)</td><td>27.8</td><td>33.3</td><td>5.6</td><td>22.2</td></tr></table>

Table 2: Experiments on the addition and subtraction subset for GSM8K-Hard dataset showing performance comparisons across different models and methods: Only Addition (left), Only Subtraction (center), and Mixed addition and subtraction (right), using Vanilla CoT, PAL, and NC (Neural Comprehension) methods.

linear algebra computations or even in-context learning algorithms that employ backpropagation (Giannou et al., 2023).

To evaluate the impact of Neural Comprehension on arithmetic reasoning, we compare the output of vanilla CoT language models and those incorporating Neural Comprehension, using the vanilla CoT baseline as a reference. As demonstrated in Figure 6 and Table 2, the vanilla CoT model struggles to extrapolate and solve arithmetic problems involving longer digit lengths. However, integrating Neural Comprehension significantly improves the performance of language models on such complex arithmetic tasks. Since we only incorporated the Addition and Subtraction CoNNs, we attribute the observed performance enhancement to the increased computational accuracy of the language model. For further evidence, we present additional experimental results on widely-used arithmetic reasoning datasets in Appendix E.2, which reinforce the benefits of using Neural Comprehension over the vanilla CoT model.

In comparison to language models employing external tools like PAL (Gao et al., 2022), our findings suggest that Neural Comprehension offers greater flexibility for LM. Firstly, by design, it minimizes the necessity for additional processing steps or external tools, leading to an efficient direct computational approach. This contrasts with Tool-based methods that often require additional programming and execution steps, increasing complexity and computational resources. Moreover, CoNN maintains end-to-end differentiability, crucial for models adapting to new data or tasks. In contrast, Tool-based methods are non-differentiable, limiting their adaptability in reinforcement learning settings or tasks with sparse delayed rewards (Chung et al., 2022; Ouyang et al., 2022). Furthermore, CoNN's modularity enhances performance across various model scales, applicable regardless of a language model's ability to call functions, unlike tools only operable in large, additionally code-trained models. Thus, the Neural Comprehension framework's efficiency, unified end-to-end neural network architecture, and extensive applicability constitute its distinct advantages over the Tool-based approach, offering a robust and scalable solution for a multitude of linguistic and computational challenges.

# 5.4 ABLATION AND ANALYSES: MODULE COMBINATION FOR NEURAL COMPREHENSION

Efficiently deploying multiple CoNNs is crucial for achieving exceptional Neural Comprehension performance. As depicted in Figure 7, the amalgamation of distinct CoNNs, tailored for both symbolic and arithmetic reasoning tasks within the language model framework, can lead to remarkable benefits. Similar to ToolFormer (Schick et al., 2023), we combine multiple different CoNNs into

![](images/c48e18fd7701b0409abfbde7334344fec78e11583aed6c3ab7627d727ba5c288.jpg)  
Figure 7: In Neural Comprehension framework, the performance of multiple different module combination is demonstrated. The left side shows the effect of combining a pretrained language model with a CoNN, while the right side shows the impact of combining a language model with multiple CoNNs. For different tasks, we categorize CoNNs as Correlated (green) and Uncorrelated (red), indicating whether the CoNN is related to the current task or not.

one framework, enabling the language model to have multiple capabilities. We conduct experiments on Last Letter Concatenation tass and AddSub $^{+}$ dataset, which shows the plug-and-play gating mechanism can still well control these CoNNs to output what should be output. It is observed that integrating pertinent CoNNs bolsters the performance of the initial language model, whereas the inclusion of unrelated language models rarely causes detrimental effects, regardless of whether single or multiple CoNNs are combined.

This can be ascribed to the refined design of the Neural Comprehension framework, which ensures the precise execution of assigned tasks by CoNNs without interference from irrelevant modules. Each CoNN module is adept at generating the appropriate output when needed, thereby preventing the emergence of erroneous results from unrelated components. Importantly, as seen in Appendix B.3, the parameter count for each CoNN module ranges from 1/1000 to 1/1000000 of that for GPT-3, and the experiments in Appendix D.3 show that the inference latency in the neural understanding framework only increases by 1%-3% compared to Vanilla.

This observation underscores the remarkable scalability of the Neural Comprehension framework, which possesses the capability to not only accommodate existing knowledge concepts but also assimilate novel ones as the number of CoNNs expands. Theoretically, the integration of tens of thousands of CoNN modules within language models holds the potential to foster a comprehensive understanding of concepts.

# 6 CONCLUSION

We have observed that language models lack an intrinsic comprehension of rule-based concepts and explored how Neural Comprehension can integrate compiled neural networks into the language model framework in a simple and plug-and-play manner. On the one hand, we demonstrated the superiority of our approach over existing learning-based methods, where our method implements comparable improvements to external tools within the neural network framework but does not require additional interpreters. This also enables language models without coding capabilities to possess symbolic manipulation abilities. On the other hand, compared to external tools, gradients can propagate without proxies, allowing better integration and full differentiability. The Neural Comprehension solves the issue of language models themselves being unable to perform robust symbolic operations and providing a foundation for future work on unifying both implicit and explicit learning in language models and facilitating seamless integration.

# REPRODUCIBILITY STATEMENT

All CoNN models mentioned in this paper have been saved in Pytorch format in the Supplementary Materials, with dropout set to 0 to ensure deterministic outputs that conform to human-specified rules. The code for the AutoCoNN toolkit and Neural Comprehension framework in this paper can be found in the code portion of the Supplementary Materials. Details of all experiments setting referenced in this paper are included in Appendix F.1. Detailed descriptions of all tasks, datasets, and baselines mentioned in this paper are provided in Appendix F.2. Details of all few-shot prompts referenced in this paper are included in Appendix G.

# ACKNOWLEDGEMENTS

This work was supported by the National Key R&D Program of China (No.2022ZD0118501) and the National Natural Science Foundation of China (No.U1936207, No.62376270, No.62171183). Youth Innovation Promotion Association CAS, and OPPO Research Fund.

# ACKNOWLEDGEMENTS FOR COLLEAGUES

We appreciate the interest shown in this work by our colleagues:

\- Haining Xie, Huanxuan Liao, Jiachun Li, Liang Gui, Pengfan Du, Pengfei Cao, Shaoru Guo, Wangtao Sun, Wenting Li, Xiusheng Huang, Yao Xu, Yifan Wei, Zhao Yang, Zhiqi Wang, Zhongtao Jiang, Zhuoran Jin, Ziyang Huang, who offered enthusiasm and backing.

# ACKNOWLEDGEMENTS FOR FRIENDS

We gratefully acknowledge the unwavering support and friendship from our community that served as the foundation for this research journey. Their companionship, through weekly board game and Werewolf gatherings, provided a reprieve from the rigors of research, allowing us to return refreshed and reinvigorated each week. Beyond the joyful times, they offered empathy during setbacks, perspective during challenges, and reassurance that progress awaits persistent effort. Their laughter, solidarity, and care kept us grounded, hopeful, and honest. The bonds formed enriched our lives immeasurably. We could not have navigated this meaningful journey without their understanding and encouragement. The impact of their compassion extends far beyond the technical contributions detailed herein. We express our sincerest appreciation, respect, and admiration for making this adventure a truly memorable experience:

\- Bingmei Sun, Boyuan Jiang, HaoChen Cao, Zixuan Cao, Donghui Li, Dongfang Suze, Ertan Zhuang, Jiajia Li, Mingwei Zhang, Lin Zhang, Mingwen Niu, Min Xiao, Qiaomu Tan, Tianyu Mu, Yuxin Liu, Xiaoyan Yu, Yuke Shi, Yixuan Li, Yang Zhou.

# REFERENCES

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. arXiv preprint arXiv:2211.15661, 2022.   
Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. north american chapter of the association for computational linguistics, 2019.   
Cem Anil, Yuhuai Wu, Anders Johan Andreassen, Aitor Lewkowycz, Vedant Misra, Vinay Venkatesh Ramasesh, Ambrose Slone, Guy Gur-Ari, Ethan Dyer, and Behnam Neyshabur. Exploring length generalization in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=zSkYVeX7bC4.   
Patel Arkil, Bhattamishra Satwik, and Goyal Navin. Are nlp models really able to solve simple math word problems? 2021.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with gpt-4, 2023.   
Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. Large language models as tool makers. arXiv preprint arXiv:2305.17126, 2023.   
Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.   
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.   
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022.   
Peter Clark, Oyvind Tafjord, and Kyle Richardson. Transformers as soft reasoners over language. In Christian Bessiere (ed.), Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pp. 3882–3890. International Joint Conferences on Artificial Intelligence Organization, 7 2020. doi: 10.24963/ijcai.2020/537. URL https://doi.org/10.24963/ijcai.2020/537. Main track.   
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.   
Xavier Daull, Patrice Bellot, Emmanuel Bruno, Vincent Martin, and Elisabeth Murisasco. Complex qa and language models hybrid architectures, survey. arXiv preprint arXiv:2302.09051, 2023.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.   
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey for in-context learning. arXiv preprint arXiv:2301.00234, 2022.   
Ippei Fujisawa and Ryota Kanai. Logical tasks for measuring extrapolation and rule comprehension. arXiv preprint arXiv:2211.07727, 2022.   
Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. arXiv preprint arXiv:2211.10435, 2022.   
Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583–30598, 2022.   
Mor Geva, Ankit Gupta, and Jonathan Berant. Injecting numerical reasoning skills into language models. arXiv preprint arXiv:2004.04487, 2020.   
Angeliki Giannou, Shashank Rajput, Jy yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers, 2023.

Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning to solve arithmetic word problems with verb categorization. empirical methods in natural language processing, 2014.   
Minghao Hu, Yuxing Peng, Zhen Huang, and Dongsheng Li. A multi-type multi-span network for reading comprehension that requires discrete reasoning. empirical methods in natural language processing, 2019.   
Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. arXiv preprint arXiv:2210.11610, 2022.   
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=e2TBb5y0yFf.   
Rik Koncel-Kedziorski, Hannaneh Hajishirzi, Ashish Sabharwal, Oren Etzioni, and Siena Dumas Ang. Parsing algebraic word problems into equations. Transactions of the Association for Computational Linguistics, 2015.   
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019.   
Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. arXiv preprint arXiv:2206.14858, 2022.   
Bin Li, Encheng Chen, Hongru Liu, Yixuan Weng, Bin Sun, Shutao Li, Yongping Bai, and Meiling Hu. More but correct: Generating diversified and entity-revised medical response. arXiv preprint arXiv:2108.01266, 2021a.   
Bin Li, Yixuan Weng, Bin Sun, and Shutao Li. Learning to locate visual answer in video corpus using question. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023.   
Bin Li, Yixuan Weng, Fei Xia, and Hanjun Deng. Towards better chinese-centric neural machine translation for low-resource languages. Computer Speech & Language, 84:101566, 2024.   
Junyi Li, Tianyi Tang, Wayne Xin Zhao, and Ji-Rong Wen. Pretrained language models for text generation: A survey, 2021b.   
Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.   
David Lindner, János Kramár, Matthew Rahtz, Thomas McGrath, and Vladimir Mikulik. Tracr: Compiled transformers as a laboratory for interpretability. arXiv preprint arXiv:2301.05062, 2023.   
Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, et al. Augmented language models: a survey. arXiv preprint arXiv:2302.07842, 2023.   
Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474, 2022.   
Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114, 2021.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. 2022.   
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf.   
Ethan Perez, Douwe Kiela, and Kyunghyun Cho. True few-shot learning with language models. Advances in neural information processing systems, 34:11054–11070, 2021.   
Xinyu Pi, Qian Liu, Bei Chen, Morteza Ziyadi, Zeqi Lin, Yan Gao, Qiang Fu, Jian-Guang Lou, and Weizhu Chen. Reasoning like program executors. 2022.   
Jing Qian, Hong Wang, Zekun Li, Shiyang Li, and Xifeng Yan. Limitations of language models in arithmetic and symbolic induction. arXiv preprint arXiv:2208.05051, 2022.   
Yasaman Razeghi, Robert L Logan IV, Matt Gardner, and Sameer Singh. Impact of pretraining term frequencies on few-shot reasoning. arXiv preprint arXiv:2202.07206, 2022.   
Subhro Roy and Dan Roth. Solving general arithmetic word problems. arXiv: Computation and Language, 2016.   
Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.   
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.   
Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost, 2018.   
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. CoRR, abs/1701.06538, 2017. URL http://arxiv.org/abs/1701.06538.   
Hikaru Shindo, Devendra Singh Dhami, and Kristian Kersting. Neuro-symbolic forward reasoning. arXiv preprint arXiv:2110.09383, 2021.   
Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ambrose Slone, Ameet Rahane, Anantharaman S. Iyer, Anders Andreassen, Andrea Madotto, Andrea Santilli, Andreas Stuhlmüller, Andrew Dai, Andrew La, Andrew Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karaka{s}, B. Ryan Roberts, Bao Sheng Loe, Barret Zoph, Bart{ł}omiej Bojanowski, Batuhan Özyurt, Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Howald, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, César Ferri Ramírez, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Chris Waites, Christian Voigt, Christopher D. Manning, Christopher Potts, Cindy Ramirez, Clara E. Rivera, Clemencia Siro, Colin Raffel, Courtney

Ashcraft, Cristina Garbacea, Damien Sileo, Dan Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Moseguí González, Danielle Perszyk, Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, David Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth Donoway, Ellie Pavlick, Emanuele Rodola, Emma Lam, Eric Chu, Eric Tang, Erkut Erdem, Ernie Chang, Ethan A. Chi, Ethan Dyer, Ethan Jerzak, Ethan Kim, Eunice Engefu Manyasi, Evgenii Zheltonozhskii, Fanyue Xia, Fatemeh Siar, Fernando Martínez-Plumed, Francesca Happé, Francois Chollet, Frieda Rong, Gaurav Mishra, Genta Indra Winata, Gerard de Melo, Germán Kruszewski, Giambattista Parascandolo, Giorgio Mariani, Gloria Wang, Gonzalo Jaimovitch-López, Gregor Betz, Guy Gur-Ari, Hana Galijasevic, Hannah Kim, Hannah Rashkin, Hannaneh Hajishirzi, Harsh Mehta, Hayden Bogar, Henry Shevlin, Hinrich Schütze, Hiromu Yakura, Hongming Zhang, Hugh Mee Wong, Ian Ng, Isaac Noble, Jaap Jumelet, Jack Geissinger, Jackson Kernion, Jacob Hilton, Jaehoon Lee, Jaime Fernández Fisac, James B. Simon, James Koppel, James Zheng, James Zou, Jan Kocoń, Jana Thompson, Jared Kaplan, Jarema Radom, Jascha Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova, Jelle Bosscher, Jennifer Marsh, Jeremy Kim, Jeroen Taal, Jesse Engel, Jesujoba Alabi, Jiacheng Xu, Jiaming Song, Jillian Tang, Joan Waweru, John Burden, John Miller, John U. Balis, Jonathan Berant, Jörg Frohberg, Jos Rozen, Jose Hernandez-Orallo, Joseph Boudeman, Joseph Jones, Joshua B. Tenenbaum, Joshua S. Rule, Joyce Chua, Kamil Kanclerz, Karen Livescu, Karl Krauth, Karthik Gopalakrishnan, Katerina Ignatyeva, Katja Markert, Kaustubh D. Dhole, Kevin Gimpel, Kevin Omondi, Kory Mathewson, Kristen Chiafullo, Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson, Laria Reynolds, Leo Gao, Li Zhang, Liam Dugan, Lianhui Qin, Lidia Contreras-Ochando, Louis-Philippe Morency, Luca Moschella, Lucas Lam, Lucy Noble, Ludwig Schmidt, Luheng He, Luis Oliveros Colón, Luke Metz, Lütfi Kerem {S}enel, Maarten Bosma, Maarten Sap, Maartje ter Hoeve, Maheen Farooqi, Manaal Faruqui, Mantas Mazeika, Marco Baturan, Marco Marelli, Marco Maru, Maria Jose Ramírez Quintana, Marie Tolkiehn, Mario Giulianelli, Martha Lewis, Martin Potthast, Matthew L. Leavitt, Matthias Hagen, Mátyás Schubert, Medina Orduna Baitemirova, Melody Arnaud, Melvin McElrath, Michael A. Yee, Michael Cohen, Michael Gu, Michael Ivanitskiy, Michael Starritt, Michael Strube, Micha{1} Sw{e}drowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike Cain, Mimee Xu, Mirac Suzgun, Mo Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva, Mozhdeh Gheini, Mukund Varma T, Nanyun Peng, Nathan Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas Cameron, Nicholas Roberts, Nick Doiron, Nikita Nangia, Niklas Deckers, Niklas Muennighoff, Nitish Shirish Keskar, Niveditha S. Iyer, Noah Constant, Noah Fiedel, Nuan Wen, Oliver Zhang, Omar Agha, Omar Elbaghdadi, Omer Levy, Owain Evans, Pablo Antonio Moreno Casares, Parth Doshi, Pascale Fung, Paul Pu Liang, Paul Vicol, Pegah Alipoormolabashi, Peiyuan Liao, Percy Liang, Peter Chang, Peter Eckersley, Phu Mon Htut, Pinyu Hwang, Piotr Mi{1}kowski, Piyush Patil, Pouya Pezeshkpour, Priti Oli, Qiaozhu Mei,Qing Lyu,Qinlang Chen,Rabin Banjade,Rachel Etta Rudolph,Raefer Gabriel,Rahel Habacker,Ramón Risco Delgado,Raphaël Millière,Rhythm Garg,Richard Barnes,Rif A. Saurous,Riku Arakawa,Robbe Raymaekers,Robert Frank,Rohan Sikand,Roman Novak,Roman Sitelew,Ronan LeBras,Rosanne Liu,Rowan Jacobs,Rui Zhang,Ruslan Salakhutdinov,Ryan Chi,Ryan Lee,Ryan Stovall,Ryan Teehan,Rylan Yang,Sahib Singh,Saif M. Mohammad,Sajant Anand,Sam Dillavou,Sam Shleifer,Sam Wiseman,Samuel Gruetter,Samuel R. Bowman,Samuel S. Schoenholz,Sanghyun Han,Sanjeev Kwatra,Sarah A. Rous,Sarik Ghazarian,Sayan Ghosh,Sean Casey,Sebastian Bischoff,Sebastian Gehrmann,Sebastian Schuster,Sepideh Sadeghi-Shadi Hamdan.Sharon Zhou.Shashank SrivastavaSherry Shi/Shikhar Singh/Shima Asaadi.Shixiang ShaneGu.Shubh Pachchigar Shubham Toshniwal.Shyam Upadhyay.Shyamolima (Shammie) Debnath,Siamak Shakeri,Simon Thormeyer,Simone Melzi,Siva Reddy,Sneha Priscilla Makini,Soo-HwanLee,Spencer Torene,Sriharsha Hatwar,Stanislas Dehaene,Stefan Divic,Stefano Ermon,stella Biderman.Stephanie Lin.Stephen PrasadSteven T. Piantadosi,Suart M. Shieber.Summer Misherghi,Svetlana Kiritchenko,Swaroop Mishra,Tal Linzen,Tal Schuster,Tao Li,Tao Yu,Tariq Ali,Tatsu Hashimoto,Telin Wu,Théo Desbordes,Thodore Rothschild,Thomas Phan,Tianle Wang,Tiberius Nkinyili,Timo Schick,Timofei Kornev,Timothy Telleen-Lawton,Titus Tunduny,Tobias Gerstenberg,Trenton Chang-Trishala Neeraj,Tushar Khot,Tyler Shultz,Uri Shaham,Vedant Misra,Vera DembergVictoria Nyamai,Vikas Raunak,Vinay Ramasesh,Vinay Uday Prabhu,Vishakh Padmakumar,Vivek SrikumarWilliam FedusWilliam SaundersWilliam Zhang,Wout Vossen,Xiang Ren,Xiaoyu Tong,Xinran Zhao,Xinyi Wu,Xudong Shen,Yadollah Yaghoobzadeh

Yair Lakretz, Yangqiu Song, Yasaman Bahri, Yejin Choi, Yichi Yang, Yiding Hao, Yifu Chen, Yonatan Belinkov, Yu Hou, Yufang Hou, Yuntao Bai, Zachary Seid, Zhuoye Zhao, Zijian Wang, Zijie J. Wang, Zirui Wang, and Ziyi Wu. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. 2022.   
Alessandro Stolfo, Zhijing Jin, Kumar Shridhar, Bernhard Schoelkopf, and Mrinmaya Sachan. A causal framework to quantify the robustness of mathematical reasoning with language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 545–561, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.32. URL https://aclanthology.org/2023.acl-long.32.   
Tian-Xiang Sun, Xiang-Yang Liu, Xi-Peng Qiu, and Xuan-Jing Huang. Paradigm shift in natural language processing. Machine Intelligence Research, 19(3):169–183, 2022. ISSN 2731-538X. doi: 10.1007/s11633-022-1331-6. URL https://www.mi-research.net/en/article/doi/10.1007/s11633-022-1331-6.   
Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.   
Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In International Conference on Machine Learning, pp. 35151–35174. PMLR, 2023.   
Xiao Wang, Guangyao Chen, Guangwu Qian, Pengcheng Gao, Xiao-Yong Wei, Yaowei Wang, Yonghong Tian, and Wen Gao. Large-scale multi-modal pre-trained models: A comprehensive survey. Machine Intelligence Research, 20(4):447–482, 2023a. ISSN 2731-538X. doi: 10.1007/s11633-022-1410-8. URL https://www.mi-research.net/en/article/doi/10.1007/s11633-022-1410-8.   
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023b. URL https://openreview.net/forum?id=1PL1NIMMrw.   
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022a.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed H Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, 2022b.   
Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like transformers. In International Conference on Machine Learning, pp. 11080–11090. PMLR, 2021.   
Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural text generation with unlikelihood training. In International Conference on Learning Representations.   
Yixuan Weng and Bin Li. Visual answer localization with cross-modal mutual knowledge transfer. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2023.   
Yixuan Weng, Zhiqi Wang, Huanxuan Liao, Shizhu He, Shengping Liu, Kang Liu, and Jun Zhao. Lmtuner: An user-friendly and highly-integrable training framework for fine-tuning large language models. arXiv preprint arXiv:2308.10252, 2023a.   
Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 2550–2575, 2023b.

Fei Xia, Bin Li, Yixuan Weng, Shizhu He, Kang Liu, Bin Sun, Shutao Li, and Jun Zhao. Medconqa: Medical conversational question answering system based on knowledge graphs. In Proceedings of the The 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 148–158, 2022.   
Kaiyu Yang and Jia Deng. Learning symbolic rules for reasoning in quasi-natural language. arXiv preprint arXiv:2111.12038, 2021.   
Zhen Yang, Ming Ding, Qingsong Lv, Zhihuan Jiang, Zehai He, Yuyi Guo, Jinfeng Bai, and Jie Tang. Gpt can solve mathematical problems without a calculator, 2023.   
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.   
Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=5NTt8GFjUHkr.   
Yang Zhao, Jiajun Zhang, and Chengqing Zong. Transformer: A general framework from machine translation to others. Machine Intelligence Research, 20(4):514–538, 2023. ISSN 2731-538X. doi: 10.1007/s11633-022-1393-5. URL https://www.mi-research.net/en/article/doi/10.1007/s11633-022-1393-5.   
Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625, 2022a.   
Hattie Zhou, Azade Nova, Hugo Larochelle, Aaron Courville, Behnam Neyshabur, and Hanie Sedghi. Teaching algorithmic reasoning via in-context learning. arXiv preprint arXiv:2211.09066, 2022b.   
Minjun Zhu, Yixuan Weng, Shizhu He, Kang Liu, and Jun Zhao. Reasonchainqa: Text-based complex question answering with explainable evidence chains. arXiv preprint arXiv:2210.08763, 2022.

# A DISCUSSION OF LIMITATIONS AND FUTURE WORK

We have presented a novel framework that integrates Compiled Neural Networks (CoNNs) with existing language models to bolster their rule understanding abilities. Although our approach has shown promising performance improvements on symbolic and arithmetic reasoning tasks, there are several limitations and potential avenues for future research that warrant further exploration.

A significant limitation of our current framework lies in the more efficient and natural incorporation of CoNNs into language models. Currently, our method employs a sparse neural network that treats the pretrained language model and CoNNs as separate modules. A more desirable solution is to leverage a dense neural network, simultaneously utilizing the benefits of both components. Examining the large-scale applicability of CoNNs is a beneficial endeavor. Although our experiments have been conducted on a relatively small scale (up to five stacked CoNNs), the advancements and abilities language models may gain from larger-scale combinations of CoNNs remain unclear. Exploring the scalability of our method and the performance advantages of deploying more complex CoNN architectures in language models could provide valuable insights into their potential.

Another promising area of research is the inclusion of explicit knowledge into CoNNs. While the current implementation centers on encoding rules into CoNNs, future work could exploit techniques from knowledge graphs to compile explicit knowledge into language models. This may significantly enhance language models' interpretability and knowledge representation capabilities, potentially resulting in improved performance on an even broader range of tasks.

In conclusion, our work on enhancing language models' rule understanding capabilities through CoNN integration has yielded promising results, albeit with some limitations and remaining challenges. By addressing these areas and extending our approach, we believe that it can ultimately lead to the development of more powerful, interpretable, and knowledge-rich language models.

# B COMPILED NEURAL NETWORKS

In this section, we will discuss the concept, implementation, and potential of Compiled Neural Networks (CoNN), a type of neural network inspired from previous works on transformers. CoNNs can perform diverse tasks such as computer arithmetic and linear algebra, demonstrating a wide range of applications in LMs and beyond.

# B.1 INTRODUCTION

Transformers have garnered significant attention due to their ability to capture high-order relations and manage long-term dependencies across tokens through attention mechanisms. This enables transformers to model contextual information effectively. Pretrained language models, such as GPT-3 (Brown et al., 2020), exploit contextual learning to invoke various modules for different tasks, like performing arithmetic upon receiving arithmetic prompts. To further enhance rule comprehension in such models, CoNN-based modules are introduced as a part of Neural Comprehension.

Distinct from common models like BERT (Devlin et al., 2018), CoNNs leverage a transformer structure and derive their weights from specialized design rather than pretraining. Each Attention layer and Multilayer Perceptron (MLP) layer in a CoNN represents a specific sequence transformation, leading to a neural network module embodying explicit and interpretable operations.

RASP (Weiss et al., 2021) is a Restricted Access Sequence Processing Language that abstracts the computational model of Transformer-encoder by mapping its essential components, such as attention and feed-forward computation, into simple primitives like select, aggregate, and zipcode. This language enables RASP programs to perform various tasks like creating histograms, sorting, and even logical inference, as demonstrated by Clark et al. (2020).

Tracr (Lindner et al., 2023) serves as a compiler that converts human-readable RASP code into weights for a GPT-like transformer architecture with only a decoder module. The Tracr framework uses JAX to transform RASP-defined code into neural network weights. Our neural reasoning framework employs weights generated by Tracr, which are then converted into PyTorch weights to be compatible with the pretrained language model.

Looped Transformers as Programmable Computers (Giannou et al., 2023) introduces a novel transformer framework that simulates basic computing blocks, such as edit operations on input sequences, non-linear functions, function calls, program counters, and conditional branches. This is achieved by reverse engineering attention and hardcoding unique weights into the model, creating a looped structure. The resulting CoNN can emulate a general-purpose computer with just 13 layers of transformers, and even implement backpropagation-based context learning algorithms, showcasing the approach's vast application prospects.

Overall, the potential applications of CoNNs are extensive, given their capacity to perform a wide array of tasks beyond natural language processing. CoNNs offer increased interpretability and transparency through explicitly defined operations, which is vital in fields such as medical diagnosis and legal decision-making. Additionally, CoNNs can lead to more efficient and effective neural network architectures by reducing pretraining requirements and facilitating improved optimization of network parameters.

# B.2 EXAMPLE

In this subsection, we briefly describe how computational processes can be represented using transformer code and demonstrate how new CoNN weights can be obtained with the aid of the Tracr compiler.

# B.2.1 PARITY CoNN

In the introduction, we tried to introduce how to perform parity checking on a sequence containing [0 | 1] using a CoNN. Whenever we need to check the sequence, this CoNN can output the completely correct answer.

THE TRACR CODE OF PARITY CONN   
```python
def parity(sop) -> rasp.SOp:
    """Multiply the length of each token."""
    sop = rasp.SequenceMap(lambda x, y: x * y, sop, length).named('map_length')

    """Add each bit."""
    out = rasp.numerical(rasp.Aggregate(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison. TRUE).named('Select'), rasp.numerical(rasp.Map(lambda x: x, sop).named('map_length')), default=0).named('Aggregate'))

    """Calculate whether the remainder of dividing it by 2 is odd or even."""
    out = rasp.Map(lambda x: 0 if x % 2 == 0 else 1, out).named('Zipmap')

    return out 
```

![](images/b6326beae3699cc787b69d9acb551e90d89f84ec270f56796627be64efbd43c5.jpg)

Figure 8: Input the $[1,0,0,0,1]$ (target output = 0) for Parity CoNN.   
![](images/e4b2641683b5fb8d1b613eebb59f214ea482d916ea84145972f62a4c6d3f3d9a.jpg)  
Figure 9: Input the $[1,0,1,0,1]$ (target output = 1) for Parity CoNN.

Figures 8 and 9 present two distinct input sequences, and illustrate the corresponding hidden state and final output obtained after passing through the internal layers of the Parity CoNN architecture.

# B.2.2 REVERSE CoNN

Figures 10 and 11 show the hidden state and output of Reverse CoNN when inputting text. The embedding of CoNN can be customized, so tokens can be either words like 'hello' or individual letters.

# B.2.3 ADDITION CoNN

Due to the high complexity of the model, we decided to omit the hidden state transformation for the Addition CoNN. However, we have provided code later in the text that will allow for easy implementation of this CoNN. The code includes add\_in\_the\_same\_position and

THE TRACR CODE OF REVERSE CONN   
```python
def reverse(sop) -> rasp.SOp:
    """Get the indices from back to front."""
    opp_idx = (length - rasp.indices).named("opp_idx")

    """opp_idx - 1, so that the first digit of indices = 0."""
    opp_idx = (opp_idx - 1).named("opp_idx-1")

    """Use opp_idx to query indices, get the Select."""
    reverse_selector = rasp.Select(rasp.indices, opp_idx, rasp.Comparison.EQ).named("reverse_selector")

    """Aggregate the reverse_selector and sop"""
    return rasp.Aggregate(reverse_selector, sop).named("reverse") 
```

![](images/24908a468b2598a72ad2131ad53c7a5478010d3c1fda04715ab8688bb4122b09.jpg)

<details>
<summary>bar_stacked</summary>

| Category | Input | Attrn 1 | MI P 1 | Attrn 2 | MI P 2 | Attrn 3 | MI P 3 | Attrn 4 | MI P 4 | Attrn 5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| bos | 0.8 | 0.1 | 0.6 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 |
| hello | 0.8 | 0.1 | 0.6 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 |
| world | 0.8 | 0.1 | 0.6 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 |
| bos | 0.8 | 0.1 | 0.6 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 |
| hello | 0.8 | 0.1 | 0.6 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 | 1.2 | 0.1 |
| world | 0.8 | 0.1 | 0.6 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 | 1.2 | 0.1 |
| bos | 0.8 | 0.1 | 0.6 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 | 1.2 | 0.1 |
| hello | 0.8 | 0.1 | 0.6 | 0.1 | 0.7 | 0.1 | 0.7 | 0.1 | 1.2 | 0.1 |
| world | 0.8 | 0.1 | 0.6 | 0.1 | 0.7 | 0.1 | 0.7 | 0   | 1    |    |
| bos + blos + hello - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - world - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - city - town<nl>
</details>

Figure 10: Input the ['hello', 'world'] for Reverse CoNN.

![](images/9c48148a7d30507e186b053eec67f4cd073f6da78f3c5c28119fe5f215f05b1f.jpg)

<details>
<summary>line</summary>

| Group | bior | bior | bior | bior | bior | bior | bior | bior | bior | bior | bior | bior |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Input | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Attn 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| MLP 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Attn 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| MLP 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Attn 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| MLP 3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Attn 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| MLP 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Attn 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| MLP 5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
Legend: The chart displays the sequences of "Input" through "MLP" (with some lines in the middle) and "Attn" (in the middle). The y-axis is labeled with the same label as "MLP", and the x-axis is labeled as "bior" (the first value). The legend indicates that "MLP" is used to denote the specific sequence or condition for each group.
</details>

Figure 11: Input the ['r', 'e', 'v', 'e', 'r', 's', 'e'] for Reverse CoNN.

add\_carry functions, which are used to calculate the addition and carry of pairs in the CoNN respectively. We divide the entire operation into two models. For the add\_carry model, we refer to the approach of ALBERT. After the output of the add\_in\_the\_same\_position model, we cyclically use the add\_carry model L times, where L is the length of the text, to ensure that all digits can carry. It is important to note that this particular Addition CoNN is only capable of performing addition operations on natural numbers.

# B.2.4 SUBTRACTION CoNN

The subtraction CoNN is similar to the addition CoNN. First, each digit is subtracted from its corresponding digit, and then it is determined whether to carry over. For ease of experimentation, this subtraction CoNN only supports subtraction of natural numbers where the minuend is greater than the subtrahend.

# B.3 CONN MODEL PARAMETERS

The parameter sizes of all CoNN models used in this work are listed in Table 3. It is noteworthy that even for GPT-3, which has parameters that are orders of magnitude larger, it remains challenging to

THE TRACR CODE OF ADDITION CONN   
```python
def split(sop, token, index):
    """Match the position of target token"""
    target_position = rasp.Aggregate(rasp.Select(sop, rasp.Map(lambda x: token, sop), rasp.Comparison.EQ), rasp.indices)

    """If need to match the front position."""
    if index == 0:
    out = rasp.Aggregate(rasp.Select(rasp.indices, rasp.indices - (length - target_position), rasp.Comparison.EQ),
    sop) # Move the sop on the left side of the token to the far right.
    return rasp.SequenceMap(lambda x, i: x if i == 2 else "_", out, rasp.categorical(rasp.SequenceMap(lambda x, i: 2 if x >= i else 0, rasp.indices, length - target_position))) # Use "_" to fill the empty position on the left.

    """If need to match the finally number."""
    else:
    return rasp.SequenceMap(lambda x, i: x if i else "_", sop,
    rasp.SequenceMap(lambda x, i: 1 if x > i else 0, rasp.indices, target_position)).named(f"shift") # Use "_" to fill the empty position on the left.

def atoi(sop):
    """Converts all text to number, and uses 0 for strings of types other than numbers, It may be mixed with 'str' or 'int'.

    """
    return rasp.SequenceMap(lambda x, i: int(x) if x.isdigit() else 0, sop, rasp.indices).named("atoi")

def shift(sop):
    """Get the target indices."""
    idx = (rasp.indices - 1).named("idx-1")

    """Use opp_idx to query indices, get the Select."""
    selector = rasp.Select(idx, rasp.indices, rasp.Comparison.EQ).named("shift_selector")

    """Aggregates the sops and selectors (converted from indexes)."""
    shift = rasp.Aggregate(selector, sop).named("shift")
    return shift

def add_in_the_same_position(sop):
    x = atoi(split(sop,'+',0)) + atoi(split(sop,'+',1))
    return x

def carry(sop):
    weight = shift(rasp.Map(lambda n:1 if n>9 else 0,sop))

    weight = rasp.Aggregate(rasp.Select(rasp.indices, rasp.indices, lambda key,query:key == query),weight,default=0)
    x = rasp.Map(lambda n:n-10 if n>9 else n,sop)
    return x + weight 
```

THE TRACR CODE OF SUBTRACTION CONN   
```python
def split(sop, token, index):...
def atoi(sop):...
def shift(sop):...
def sub_in_the_same_position(sop):
    x = atoi(split(sop,'-',0)) - atoi(split(sop,'-',1))
    return x
def carry(sop):
    weight = shift(rasp.Map(lambda n:1 if n<0 else 0,sop))
    weight = rasp.Aggregate(rasp.Select(rasp.indices,rasp.indices,lambda key,query:key == query),weight,default=0)
    x = rasp.Map(lambda n:n+10 if n<0 else n,sop)
    return x - weight 
```

<table><tr><td>Model</td><td>Layers</td><td>Heads</td><td>Vocabulary Size</td><td>Window Size</td><td>Hidden Size</td><td>MLP Hidden Size</td><td># Parameters</td><td>Compared to GPT-3</td></tr><tr><td>Parity</td><td>4</td><td>1</td><td>4</td><td>40</td><td>132</td><td>1959</td><td>2.2M</td><td> $\approx 1/100,000$ </td></tr><tr><td>Reverse</td><td>4</td><td>1</td><td>28</td><td>40</td><td>297</td><td>1640</td><td>4.3M</td><td> $\approx 1/50,000$ </td></tr><tr><td>Last Letter</td><td>3</td><td>1</td><td>28</td><td>16</td><td>103</td><td>32</td><td>62.6K</td><td> $\approx 1/3,000,000$ </td></tr><tr><td>Copy</td><td>1</td><td>1</td><td>28</td><td>16</td><td>69</td><td>26</td><td>8.8K</td><td> $\approx 1/20,000,000$ </td></tr><tr><td>Add_in_the_same_position</td><td>7</td><td>1</td><td>13</td><td>40</td><td>535</td><td>6422</td><td>51.8M</td><td> $\approx 1/3000$ </td></tr><tr><td>Add_Carry</td><td>3</td><td>1</td><td>122</td><td>40</td><td>130</td><td>52</td><td>117K</td><td> $\approx 1/1,500,000$ </td></tr><tr><td>Sub_in_the_same_position</td><td>7</td><td>1</td><td>13</td><td>40</td><td>535</td><td>6422</td><td>51.8M</td><td> $\approx 1/3000$ </td></tr><tr><td>Sub_Carry</td><td>3</td><td>1</td><td>122</td><td>40</td><td>130</td><td>52</td><td>117K</td><td> $\approx 1/1,500,000$ </td></tr></table>

Table 3: We reported on a CoNN with a single function, including its actual parameter size and comparison with the parameters of GPT-3.

solve symbolic problems. However, with the use of compiled neural networks, only a small number of parameters are needed to achieve Neural Comprehension.

# B.4 ENVIRONMENTAL AND HUMAN-CENTRIC BENEFITS OF COMPILED NEURAL NETWORKS

Compiled Neural Networks (CoNNs) address concerns related to the environmental impact of training large models and the need for human-centric computing. CoNN models can reduce energy consumption and carbon emissions by minimizing extensive pretraining and decreasing parameter size, as seen in Table 3. This reduction in computational power and energy requirements makes both the training and inference processes more environmentally friendly. Additionally, Neural Comprehension offers a more interpretable and transparent alternative to conventional deep learning models. CoNN's explicit operation definitions and specialized architecture enable users to comprehend the reasoning behind model decisions, fostering trust and facilitating human-AI collaboration. Increased interpretability also allows for scrutiny of model behavior, promoting the development of fair, accountable, and transparent systems aligned with ethical considerations and human values.'

# C EXPERIMENT FOR AUTOCONN

# C.1 METHOD

For experts, they may need to spend a lot of time writing code suitable for CoNN, while non-expert users find it hard to obtain or modify CoNN. These issues limit the efficient combination of CoNN and LM, so we utilized the few-shot ability of language models to make the AutoCoNN toolkit (Weng et al., 2023a). In this section, we will show a series of detailed experiments on AutoCoNN to demonstrate this. First, It is the Demo of AutoCoNN code:

<table><tr><td>CoNN Model</td><td>Example=1</td><td>Example=2</td><td>Example=5</td></tr><tr><td>Parity Model</td><td>5/10</td><td>10/10</td><td>10/10</td></tr><tr><td>Reverse Model</td><td>10/10</td><td>10/10</td><td>10/10</td></tr><tr><td>Last Letter Model</td><td>9/10</td><td>10/10</td><td>10/10</td></tr><tr><td>Copy Model</td><td>10/10</td><td>10/10</td><td>10/10</td></tr></table>

Table 4: For each CoNN model, we selected ten groups of models that were judged to be correct by AutoCoNN. We manually evaluated whether these models were indeed correct. The 'Example=x' means that x Examples were provided in the validation stage.

# DEMO OF AUTOCONN

```python
from NeuralCom.AutoCoNN import AutoCoNN

INSTRUCT = 'Create_an_SOp_that_is_the_last_letter_of_a_word'
VOCAB = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
EXAMPLE = [[['a', 'b', 'c'], ['c', 'c', 'c']], [['b', 'd'], ['d', 'd']]]

auto = AutoCoNN()
model, tokenizer = auto(instruct=INSTRUCT, vocab=VOCAB, example=EXAMPLE) 
```

Table 5 shows the efficiency comparison between experts and AutoCoNN. This demonstrates that the AutoCoNN toolkit can generate various CoNNs faster. But we also found that for more difficult ones like Addition and Subtraction, it fails to successfully generate, which becomes one of the limitations of AutoCoNN. On the other hand, we tried providing only "Instruct" or "Example" for AutoCoNN to generate $^{3}$ , and often "Instruct" can generate CoNN with higher accuracy, while "Example" cannot.

This shows that giving explicit operational instructions performs better than directly observing data in AutoCoNN.

<table><tr><td>CoNN Model</td><td>Expert&#x27;s Working Time</td><td>Success by AutoCoNN</td><td>Can AutoCoNN solve</td><td>AutoCoNN (w. Instruct)</td><td>AutoCoNN (w. Example)</td></tr><tr><td>Parity Model</td><td>1 hours</td><td>8/20</td><td> $\checkmark \checkmark$ </td><td>7/20</td><td>3/20</td></tr><tr><td>Reverse Model</td><td>0.5 hour</td><td>15/20</td><td> $\checkmark \checkmark$ </td><td>16/20</td><td>11/20</td></tr><tr><td>Last Letter Model</td><td>0.5 hour</td><td>13/20</td><td> $\checkmark \checkmark$ </td><td>12/20</td><td>10/20</td></tr><tr><td>Copy Model</td><td>0.2 hour</td><td>17/20</td><td> $\checkmark \checkmark$ </td><td>17/20</td><td>15/20</td></tr><tr><td>Addition Model</td><td>48 hours</td><td>0/20</td><td> $\times$ </td><td>0/20</td><td>0/20</td></tr><tr><td>Subtraction Model</td><td>48 hours</td><td>0/20</td><td> $\times$ </td><td>0/20</td><td>0/20</td></tr></table>

Table 5: Comparison between AutoCoNN and Expert Built CoNN. The column 'Expert's Working Time' refers to the time required for a trained engineer to write the CoNN code; 'Success by AutoCoNN' refers to the accuracy of 20 results generated by using GPT-3.5 for diverse decoding; 'Can AutoCoNN solve' refers to whether AutoCoNN can identify suitable CoNN code from the 20 results through validation. It is worth noting that in this experiment, we use sampling decoding with temperature=0.7 to generate 20 different CoNNs codes, which we convert to Pytorch versions of CoNNs models. We report the accuracy of the CoNNs codes through manual (expert) evaluation.

It is difficult for non-expert users to assess the accuracy of the generated code, we automatically utilize the Example information to verify the accuracy of the CoNN model - checking whether the output result of the input sequence is exactly consistent with the Example. The results shown in Table 4 demonstrate that generally 2 Examples are sufficient to select an accurate CoNN model, which means it is very easy for users to use and demonstrate. However, considering the varying difficulty of different tasks, we still suggest non-expert users provide more Examples to ensure the accuracy of the generated CoNN.

# D EXPERIMENTAL SETTINGS

In this study, we primarily explore the capacity of language models to address symbolic reasoning tasks, concentrating on three areas: symbolic operations, symbolic reasoning, and arithmetic reasoning.

Symbolic Operations Building upon the approaches developed by Anil et al. (2022) and Qian et al. (2022), we examine the following tasks: Parity, Reverse, Addition and Subtraction. These tasks do not require complex text understanding, but only require faithfully implementing symbolic operations and outputting the corresponding results.

Symbolic Reasoning We employ the experimental framework of Wei et al. (2022b) for the two tasks, Last Letter Concatenation and Coin Flip. These tasks require a combination of language understanding and rule comprehension abilities.

Arithmetic Reasoning To evaluate the method's generalization ability from symbolic operations to arithmetic reasoning in addition and subtraction tasks, we use five established arithmetic reasoning datasets: AddSub (Hosseini et al., 2014), SingleEq (Koncel-Kedziorski et al., 2015), MultiArith (Roy & Roth, 2016), GSM8K (Cobbe et al., 2021), and SVAMP (Arkil et al., 2021). Additionally, we introduce the $\mathrm{AddSub^{+}}$ dataset, containing tasks of varying complexity based on the number of digits involved in arithmetic operations, ranging from 1-digit addition to 20-digit addition/subtraction tasks.

# E SUPPLEMENTARY EXPERIMENT

# E.1 THE EFFECT OF TRAINING DATA SCALE ON LENGTH GENERALIZATION OF GRADIENT-BASED MODELS

To investigate the impact of training data scale on out-of-distribution (OOD) performance, we conducted experiments using the T5-large model with varying amounts of in-distribution training data. The experimental setup closely followed that of Main Figure 3, utilizing numbers with 10 to

![](images/86ba48ca189c49de8bc61d665d4c10815fc1a23da64bc4d788aaa7051c104d78.jpg)

<details>
<summary>surface_3d</summary>

| Text Leng | Training Data | Accuracy (%) |
| --------- | ------------- | ------------ |
| 0         | 1M            | 100          |
| 5         | 3M            | 95           |
| 10        | 5M            | 90           |
| 15        | 7M            | 85           |
| 20        | 9M            | 80           |
| 25        | 1M            | 75           |
| 30        | 3M            | 70           |
| 35        | 5M            | 65           |
| 40        | 7M            | 60           |
</details>

(a) Parity

![](images/62c65db8dad1417f8e07912f2f8489fc2e38541f66041348aa25d6b70404f813.jpg)

<details>
<summary>surface_3d</summary>

| Text Length | Training Data | Accuracy (%) |
| ----------- | ------------- | ------------ |
| 0           | 1M            | ~80          |
| 5           | 3M            | ~75          |
| 10          | 5M            | ~70          |
| 15          | 3M            | ~65          |
| 20          | 3M            | ~60          |
| 25          | 3M            | ~55          |
| 30          | 3M            | ~50          |
| 35          | 3M            | ~45          |
| 40          | 3M            | ~40          |
| 40          | 5M            | ~35          |
| 40          | 10M           | ~30          |
| 40          | 10M           | ~25          |
| 40          | 3M            | ~20          |
| 40          | 5M            | ~15          |
| 40          | 10M           | ~10          |
| 40          | 3M            | ~5           |
| 40          | 10M           | ~0           |
</details>

(b) Reverse

![](images/11ad57506632fe6d0c7ce2a815ae23378b48d0678b40f03af66ef0fa653a4e2f.jpg)

<details>
<summary>surface_3d</summary>

| Text Length | Training Data | Accuracy (%) |
| ----------- | ------------- | ------------ |
| 5           | 1M            | ~20          |
| 10          | 3M            | ~40          |
| 15          | 5M            | ~60          |
| 20          | 7M            | ~80          |
| 25          | 9M            | ~90          |
| 30          | 1M            | ~85          |
| 35          | 3M            | ~70          |
| 40          | 5M            | ~50          |
</details>

(c) Addition

![](images/f1f6839cb2d5e48a5b17bdc13e8aa9ac321e5d2d8b02c7501a5ad01535d612b5.jpg)

<details>
<summary>surface_3d</summary>

| Text Length | Training Data | Accuracy (%) |
| ----------- | ------------- | ------------ |
| 3M          | 1M            | ~70          |
| 5M          | 3M            | ~80          |
| 1M          | 5M            | ~90          |
| 3M          | 1M            | ~95          |
| 5M          | 3M            | ~98          |
| 1M          | 5M            | ~99          |
| 3M          | 1M            | ~100         |
| 5M          | 3M            | ~99          |
| 1M          | 5M            | ~98          |
| 3M          | 1M            | ~95          |
| 5M          | 3M            | ~90          |
| 1M          | 5M            | ~80          |
| 3M          | 1M            | ~70          |
| 5M          | 3M            | ~60          |
| 1M          | 5M            | ~50          |
| 3M          | 1M            | ~40          |
| 5M          | 3M            | ~30          |
| 1M          | 5M            | ~20          |
| 3M          | 1M            | ~10          |
| 5M          | 3M            | ~5           |
| 1M          | 5M            | ~2           |
| 3M          | 1M            | ~1           |
| 5M          | 3M            | ~0.5         |
| 1M          | 5M            | ~0.2         |
| 3M          | 1M            | ~0.1         |
| 5M          | 3M            | ~0.05        |
| 1M          | 5M            | ~0.02        |
| 3M          | 1M            | ~0.01        |
| 5M          | 3M            | ~0.005       |
| 1M          | 5M            | ~0.002       |
| 3M          | 1M            | ~0.001       |
| 5M          | 3M            | ~0.0005      |
| 1M          | 5M            | ~0.0002      |
| 3M          | 1M            | ~0.0001      |
| 5M          | 3M            | ~0.00005     |
| 1M          | 5M            | ~0.00002     |
| 3M          | 1M            | ~0.00001     |
| 5M          | 3M            | ~0.000005    |
| 1M          | 5M            | ~0.000002    |
| 3M          | 1M            | ~0.000001    |
| 5M          | 3M            | ~0.0000005   |
| 1M          | 5M            | ~0.0000002   |
| 3M          | 1M            | ~0.0000001   |
| 5M          | 3M            | ~0.00000005  |
| 1M          | 5M            | ~0.00000002  |
| 3M          | 1M            | ~0.00000001  |
| 5M          | 3M            | ~0.000000005 |
| 1M          | 5M            | ~0.000000002 |
| 3M          | 1M            | ~0.000000001 |
| 5M          | 3M            | ~0.0000000005|
| 1M          | 5M            | ~0.0000000002|
| 3M          | 1M            | ~0.0000000001|
| 5M          | 3M            | ~0.00000000005|
| 1M          | 5M            | ~0.00000000002|
| 3M          | 1M            | ~0.00000000001|
| 5M          | 3M            | ~0.000000000005|
| 1M          | 5M            | ~-1           |
| 3M          | 1M            | -2           |
| 5M          | 3M            | -3           |
| 1M          | 5M            | -4           |
| 3M          | 1M            | -5           |
| 5M          | 3M            | -6           |
| 1M          | 5M            | -7           |
| 3M          | 1M            | -8           |
| 5M          | 3M            | -9           |
| 1M          | 5M            | -10          |
| 3M          | 1M            | -11          |
| 5M          | 3M            | -12          |
| 1M          | 5M            | -13          |
| 3M          | 1M            | -14          |
| 5M          | 3M            | -15          |
| 1M          | 5M            | -16          |
| 3M          | 1M            | -17          |
| 5M          | 3M            | -18          |
| 1M          | 5M            | -19          |
| 3M          | 1M            | -2              |
| 5M          | 3M            | -2.9         |
| 1M          | 5M            | -3.8         |
| 3M          | 1M            | -4.7         |
| 5M          | 3M            | -5.6         |
| 1M          | 5M            | -6.5         |
| 3M          | 1M            | -7.4         |
| 5M          | 3M            | -8.3         |
| 1M          | 5M            | -9.2         |
| 3M          | 1M            | -1             |
| 5M          | 3M            | -1.9         |
| 1M          | 5M            | -2.8         |
| 3M          | 1M            | -3.7         |
| 5M          | 3M            | -4.6         |
| 1M          | 5M            | -5.5         |
| 3M          | 1M            | -6.4         |
| 5M          | 3M            | -7.3         |
| 1M          | 5M            | -8.2         |
| 3M          | 1M            | -9.1         |
| 5M          | 3M            | -1             |
| 1M          | 5M            | -1.9         |
|
| 3M          | 1M            | -2.8         |
|
| 5M          | 3M            | -3.7         |
|
| 1M          | 5M            | -4.6         |
|
| 3M          | 1M            | -5.5         |
|
| 5M          | 3M            | -6.4         |
|
| 1M          | 5M            | -7.3         |
|
| 3M          | 1M            | -8.2         |
|
| 5M          | 3M            | -9.1         |
|
| 1M          | 5M            | -1             |
|
| 3M          | 1M            | -1.9         |
|
| All Text Lengths are not provided in the image; the actual text length values are estimated based on the code execution of the data source and the number of words used in the image.
</details>

(d) Subtraction   
Figure 12: Length Generalization Performance of Language Models with Different Dataset Sizes.

20 digits as the training set but varying the number of training examples between 1 million and 15 million. The peak validation set performance for each experiment is reported in Figure 12.

The results in Figure 12 show that increasing the scale of the In-Dist training data leads to only marginal improvements in OOD performance. This finding is discouraging, suggesting that gradient-based language models face challenges in capturing the true underlying meaning of symbols and their transformation rules based on the data distribution alone.

# E.2 REAL-WORLD ARITHMETIC REASONING TASKS

As model parameters, training calculations, and dataset sizes have increased, language models have gained new capabilities (Srivastava et al., 2022; Wei et al., 2022a), such as Machine Translation (Zhao et al., 2023; Li et al., 2024), complex QA (Zhu et al., 2022; Daull et al., 2023), Multimodal QA (Wang et al., 2023a; Li et al., 2023; Weng & Li, 2023), coding (Li et al., 2022; Nijkamp et al., 2022), few-shot learning (Brown et al., 2020; Perez et al., 2021), medical diagnosis (Li et al., 2021a; Xia et al., 2022), and chain of thought (Wei et al., 2022b; Weng et al., 2023b).

In Table 6, we compared Vanilla CoT with the Neural Comprehension framework for arithmetic reasoning tasks. We integrated the Addition and Subtraction CoNNs with LLMs and observed improved performance across several tasks. This suggests that the proposed Neural Comprehension framework can compensate for the difficulties faced by large-scale language models in computational tasks. Nevertheless, the performance improvement is not as significant due to the choice of specific CoNN models to ensure clarity in our experiments. Designing CoNN models to support more general arithmetic tasks could potentially yield more substantial improvements. In addition, since the Neural

<table><tr><td colspan="2">Method</td><td>GSM8K</td><td>SingleEq</td><td>AddSub</td><td>MultiArith</td><td>SVAMP</td><td>Average</td></tr><tr><td colspan="2">Previous SOTA (Fintune)</td><td> $35^a/57^b$ </td><td> $32.5^c$ </td><td> $94.9^d$ </td><td> $60.5^e$ </td><td> $57.4^f$ </td><td>-</td></tr><tr><td colspan="2">GPT-3 Standard</td><td>19.7</td><td>86.8</td><td>90.9</td><td>44.0</td><td>69.9</td><td>62.26</td></tr><tr><td rowspan="2">GPT-3 (175B) code-davinci-001</td><td>CoT</td><td>13.84</td><td>62.02</td><td>57.22</td><td>45.85</td><td>38.42</td><td>43.47</td></tr><tr><td>CoT + Neural Comprehension</td><td> $13.95_{(+0.11)}$ </td><td> $62.83_{(+0.81)}$ </td><td> $60.25_{(+3.03)}$ </td><td> $45.85_{(+0.0)}$ </td><td> $38.62_{(+0.2)}$ </td><td> $44.30_{(+0.83)}$ </td></tr><tr><td rowspan="2">GPT-3.5 (175B) code-davinci-002</td><td>CoT</td><td>60.20</td><td>91.01</td><td>82.78</td><td>96.13</td><td>75.87</td><td>81.20</td></tr><tr><td>CoT + Neural Comprehension</td><td> $60.42_{(+0.22)}$ </td><td> $91.01_{(+0.0)}$ </td><td> $82.78_{(+0.0)}$ </td><td> $96.13_{(+0.0)}$ </td><td> $76.09_{(+0.22)}$ </td><td> $81.29_{(+0.09)}$ </td></tr></table>

Table 6: Problem solve rate (%) on arithmetic reasoning datasets. The previous SoTA baselines are obtained from: (a) GPT-3 175B finetuned (Cobbe et al., 2021); (b) GPT-3 175B finetuned plus an additional 175B verifier(Cobbe et al., 2021); (c) Hu et al. (2019); (d) Roy & Roth (2016); (e) Roy & Roth (2016); (f) Amini et al. (2019); (f) Pi et al. (2022)

Comprehension framework improves the gap between the data distribution learned by the language model during training through gradient descent and the real rules, it can also be combined with some existing logical improvements to language models, including self-consistency (Wang et al., 2023b), least-to-most (Zhou et al., 2022a), self-improve (Huang et al., 2022), and self-verification (Weng et al., 2023b). It can also be combined with some zero-shot methods (Kojima et al., 2022; Zhang et al., 2023).

To further evaluate the effectiveness of the Neural Comprehension framework, Table 7 presents the results of fine-tuning T5 models with Addition and Subtraction CoNN on the GSM8K training dataset. The comparison of three different-sized models reveals that the framework can model deterministic rules defined by humans, thus avoiding the uncertainty associated with gradient descent learning from data distribution.

<table><tr><td>Method</td><td>T5-small</td><td>T5-base</td><td>T5-large</td></tr><tr><td>Origin</td><td>1.74</td><td>1.52</td><td>3.87</td></tr><tr><td>Neural Comprehension</td><td>1.82</td><td>1.59</td><td>4.02</td></tr><tr><td>Ours Improve</td><td>+0.08</td><td>+0.07</td><td>+0.15</td></tr></table>

Table 7: The test set problem-solving rate (%) of the T5 model on the GSM8K dataset.

E.3 THE EFFICIENCY OF NEURAL COMPREHENSION 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Params</td><td rowspan="2">Task</td><td colspan="2">Vanilla</td><td colspan="2">Neural Comprehension</td><td colspan="2"> $\delta_{\text{Time}}$ </td></tr><tr><td>GPU</td><td>CPU</td><td>GPU</td><td>CPU</td><td>GPU</td><td>CPU</td></tr><tr><td>T5-small</td><td>60M</td><td>Coin Flip</td><td>5.280s</td><td>5.720s</td><td>5.431s</td><td>5.872s</td><td>0.151s (2.86%)</td><td>0.152s (2.66%)</td></tr><tr><td>T5-base</td><td>220M</td><td>Coin Flip</td><td>7.865s</td><td>13.767s</td><td>8.010s</td><td>13.939s</td><td>0.145s (1.84%)</td><td>0.172s (1.25%)</td></tr><tr><td>T5-large</td><td>770M</td><td>Coin Flip</td><td>14.055s</td><td>32.953s</td><td>14.194s</td><td>33.120s</td><td>0.139s (0.99%)</td><td>0.167s (0.51%)</td></tr><tr><td>T5-small</td><td>60M</td><td>Last Letter Concatenation</td><td>16.233s</td><td>28.309s</td><td>16.744s</td><td>28.720s</td><td>0.511s (3.15%)</td><td>0.411s (1.45%)</td></tr><tr><td>T5-base</td><td>220M</td><td>Last Letter Concatenation</td><td>28.912s</td><td>55.660s</td><td>29.426s</td><td>56.087s</td><td>0.514s (1.78%)</td><td>0.427s (0.77%)</td></tr><tr><td>T5-large</td><td>770M</td><td>Last Letter Concatenation</td><td>49.584s</td><td>103.739s</td><td>50.066s</td><td>104.134s</td><td>0.482s (0.97%)</td><td>0.395s (0.38%)</td></tr></table>

Table 8: In Neural Comprehension framework, the inference latency comparison of the T5 model.

To evaluate the efficacy of Neural Comprehension, we conducted further experiments comparing the inference latency of both the Vanilla and Neural Comprehension frameworks on an equal number of sequences and equal sequence lengths using GPU and CPU configurations. We employed a batch size of 1 and assessed the inference latency of Neural Comprehension in conjunction with various T5 model sizes across two symbolic inference tasks to ascertain efficiency. The full results are detailed in Table 8.

Our findings reveal that implementing Neural Comprehension increases computational requirements, primarily attributed to the supplementary parameter volume and computational demands introduced by CoNNs. However, as the scale of pretrained language models expands, the proportion of $\delta_{Time}$ within

the Neural Comprehension framework progressively diminishes, particularly for larger language models.

F IMPLEMENTATION AND DETAILS 

<table><tr><td>Model</td><td>Model Creator</td><td>Modality</td><td>Version</td><td># Parameters</td><td>Tokenizer</td><td>Window Size</td><td>Access</td></tr><tr><td>T5-small</td><td>Google</td><td>Text</td><td>T5.1.0</td><td>60M</td><td>T5</td><td>512</td><td>Open</td></tr><tr><td>T5-base</td><td>Google</td><td>Text</td><td>T5.1.0</td><td>220M</td><td>T5</td><td>512</td><td>Open</td></tr><tr><td>T5-large</td><td>Google</td><td>Text</td><td>T5.1.0</td><td>770M</td><td>T5</td><td>512</td><td>Open</td></tr><tr><td>GLM-130B</td><td>Tsinghua University</td><td>Text</td><td>GLM-130B</td><td>130B</td><td>ICE</td><td>2048</td><td>open</td></tr><tr><td>GPT-3</td><td>OpenAI</td><td>Text,Code</td><td>code-davinci-001</td><td>175B*</td><td>GPT-2</td><td>2048</td><td>limited</td></tr><tr><td>GP-3.5</td><td>OpenAI</td><td>Text,Code,Instruct</td><td>code-davinci-002</td><td>175B*</td><td>GPT-2</td><td>8096</td><td>limited</td></tr><tr><td>GPT-4</td><td>OpenAI</td><td>Text,Code,Instruct,...</td><td>gpt-4</td><td>175B0*</td><td>GPT-2</td><td>8000</td><td>limited</td></tr></table>

Table 9: Models. Description of the models evaluated in this effort: provenance for this information is provided in models. \* indicates that we believe the associated OpenAI models are this size, but this has not been explicitly confirmed to our knowledge.

In this section, we provide a detailed description of the experimental setup from a model and dataset perspective, ensuring repeatability of our experiments.

# F.1 MODEL

Our experiments primarily involve the T5, GPT, and GLM-130B families of models. Neural Comprehension framework supports seamless integration with language models having decoder structures, regardless of the scale of the language model. We fine-tune the T5 models, while the larger models with over 10 billion parameters are used for few-shot In-context learning. Table 9 presents a comprehensive list of all models used in our experiments $^{4}$ .

# F.1.1 FINE-TUNING

For the T5 models, we employ the standard fine-tuning approach using the pretrained models as a starting point. We follow the pre-processing steps in the T5 original paper, which involves set the input text max length to 150 and using the tokenizer to process the data. We use a batch size of 64 for all models and the Adafactor optimizer (Shazeer & Stern, 2018) with a learning rate of $1 \times 10^{-4}$ . The models are trained for a maximum of 20 epochs. We use a cosine learning rate schedule with a warm-up phase comprising $5\%$ of the total number of training steps. We employ a dropout rate of 0.1 during training to mitigate overfitting. Our experiments utilize the PyTorch framework (Paszke et al., 2019) for training and inference. Table 5 displays the parameter settings for the T5 models during training, which is conducted on four NVIDIA A6000 GPUs with 48GB of memory each.

<table><tr><td>Training Setting</td><td>Configuration</td></tr><tr><td>optimizer</td><td>Adafactor</td></tr><tr><td>base learning rate</td><td> $1 \times 10^{-4}$ </td></tr><tr><td>weight decay</td><td> $2 \times 10^{-5}$ </td></tr><tr><td>decay rate</td><td>-0.8</td></tr><tr><td>optimizer eps</td><td> $(1 \times 10^{-30}, 2 \times 10^{-3})$ </td></tr><tr><td>batch size</td><td>64</td></tr><tr><td>training epochs</td><td>20</td></tr><tr><td>gradient clip</td><td>1.0</td></tr></table>

Table 10: Training Setting

In Table 10, we list the hyperparameters used to train the T5 model. We carefully selected these parameters to ensure that the within-distribution validation accuracy roughly converged. We report all peak validation set results, and in every experiment we ran, we found that within-distribution validation accuracy monotonically increased during training iterations (until it approached $100\%$ ), and we never observed overfitting. This may be due to the regular nature of the tasks we considered in the paper. We followed the Anil et al. (2022)'s setup and did not use OOD performance in model selection, as this would constitute "peaking at the test conditions". Regarding the number of training iterations, we also tried training the T5 model with more iterations in the addition task, but this did not lead to substantial differences in OOD performance (i.e., it was still equally poor).

# F.1.2 FEW-SHOT IN-CONTEXT LEARNING

For the few-shot context learning on GPT and GLM-130B models, we employ an approach inspired by the recent work on GPT-3 (Brown et al., 2020). In this methodology, the models are provided a context consisting of a few examples of the input-output pairs in natural language format. Following this, the models are expected to generalize and perform well on the task without any explicit fine-tuning. We carefully design the context to include diverse examples that represent the range of input types and required model reasoning. Importantly, we limit the context length to be within the maximum token limits of the models. For instance, GPT-3 has a token limit of 2048. Due to limited access to the GPT family of models, we utilize the official API for these experiments $^{5}$ . For the GLM-130B, we employ the FasterTransformer framework to set up local inference with INT4 on eight NVIDIA GeForce RTX 3090 GPUs with 24GB of memory each.

To compare with CoT and PAL which experimented on GPT-3 series models, we simulated Neural Comprehension (NC) within the constraints of the API access. We treated the API's output as if it were part of the Neural Comprehension structure. This involved a simulated gating mechanism, where the output from the API was corrected using CoNNs form left to right, and then the adjusted response (Truncate the text after the altered text.) was changed into the API' input for continued generation. This simulation was to ensure that the benefits of NC could be compared fairly with the existing results from PAL.

# F.2 TASKS AND DATASET

In this paper, all data sets related to length generalization consist of independent data sets with the same number of digits but different lengths, and each digit in the test set is unique. Therefore, there may be slight fluctuations between data sets of different lengths, but the overall trend is generally clear. To further illustrate the differences between data sets of different lengths, the following examples are provided:

$$
\text { Parity: } \quad \underbrace {1 1 0} _ {\text { Length } = 3} = 0 \quad \quad \underbrace {1 0 1 1 0 0} _ {\text { Length } = 6} = 1 \quad \quad \underbrace {0 1 0 0 0 1 1 1 0 1 0 1} _ {\text { Length } = 1 2} = 0
$$

$$
\text { Reverse: } \quad \underbrace {\mathrm{abc}} _ {\text { Length } = 3} = \mathrm{cba} \quad \underbrace {\text { figure }} _ {\text { Length } = 6} = \text { erugif } \quad \underbrace {\text { accomplished }} _ {\text { Length } = 1 2} = \text { dehsilpmocca }
$$

$$
\text {Addition:} \quad \underbrace {1 + 2} _ {\text {Length} = 3} = 3 \quad \underbrace {1 8 + 2 4 5} _ {\text {Length} = 6} = 2 6 3 \quad \underbrace {4 8 8 6 4 + 9 6 4 3 1 5} _ {\text {Length} = 1 2} = 1 0 1 3 1 7 9
$$

Arithmetic Reasoning: Joan found $\underbrace{6546634574688499}_{Length = 16}$ seashells and Jessica found

3855196602063621 seashells on the beach. How many seashells did they find Length = 16

together?

# F.2.1 DATA GENERATION DETAILS

Synthetic Parity Dataset: We sample instances of lengths 1-40 from a uniform Bernoulli distribution. We first uniformly sample the number of ones, and then randomly shuffle the positions of each one within a fixed-length bitstring. For the experiments in Figure 5.1, we train T5 on lengths 10-20, with 99000 training samples per bit. For all methods, we test each bit using 1000 samples.

Synthetic Reverse Dataset: We selected a dataset of instances with lengths ranging from 1 to 40. For the experiments in Figure 5.1, the training set consists of 99000 samples each for lengths 10-20. Each input is a randomly generated word string of specified length, where the letters are selected uniformly at random from a set of 26 letters using a Bernoulli distribution (without necessarily having any actual meaning), and all letters are converted to lowercase. The test set for lengths 1-40 contains at least 1000 test samples.

PARITY DATASET   
```python
def generate_parity_data(n):
    data = []
    for _ in range(100000):
    input_str = ''.join(str(random.randint(0, 1)) for _ in range(n))
    label = sum(int(x) for x in input_str) % 2
    data.append({'input': input_str, 'label': label})
    return data

parity_data = {}
for n in range(1, 41):
    parity_data[n] = generate_parity_data(n)

REVERSE DATASET
reverse_data = {}
for n in range(1, 41):
    reverse_data[n] = []
    for _ in range(100000):
    word = ''.join(random.choice(string.ascii_lowercase) for _ in range(n))
    reverse_data[n].append({'input':word,'label':''.join(list(reversed(word)))}) 
```

Synthetic Addition and subtraction Dataset: Addition and subtraction are fundamental arithmetic operations that are commonly taught in primary school. To generate the dataset, we takes as input the number of digits n and returns a list of 100000 examples. The function first calculates the remainder k when $(n-1)$ is divided by 2, and then divides $(n-1)$ by 2 if n is greater than 2, else it sets n to itself. This ensures that the length of the first number is either equal to or one digit longer than the length of the second number. The function then generates 100000 examples using randomly generated numbers. Specifically, it generates two numbers a and b where a is a random integer between $10^{(n+k-1)}$ and $10^{(n+k)} - 1$ , and b is a random integer between $10^{(n-1)}$ and $10^{n} - 1$ . It then appends each example to the list data in the form of a dictionary with the input as the string "a+b" and the label as the sum a+b.

For the experiments in Figure Figure 1, we provided 99000 training data examples for addition with numbers ranging from 3 to 10 digits in length for the T5 model. For the GPT-3.5 and GPT-4 models, we provided 8 few-shot samples within 10 digits. We evaluated the performance of all three models on numbers ranging from 3 to 30 digits in length, with 1000 test samples per digit. On the other hand, for the experiments in Figure Figure 5.1, we provided 99000 training data examples for addition with numbers ranging from 10 to 20 digits in length for the T5 model. For the GPT-3.5 and GPT-4 models, we provided 8 few-shot samples within the range of 10 to 20 digits. We evaluated the performance of all three models on numbers ranging from 3 to 30 digits in length, with 1000 test samples per digit.

For subtraction, we use a similar approach.

# F.2.2 SYMBOLIC REASONING DATASET

Coin Flip: We followed Wei et al. (2022b)'s setup and randomly concatenated first and last names from the top 1000 names in the census data (https://namecensus.com/) to create the

ADDITION DATASET   
```python
# Generate two random n-digit numbers and their sum
def generate_additive_example(n):
    data = []
    k = (n - 1) % 2
    n = (n - 1) // 2 if n > 2 else n

    for _ in range(100000):
    a = random.randint(10**(n+k-1), 10**(n+k) - 1)
    b = random.randint(10**(n-1), 10**n - 1)
    data.append({'input': str(a) + ' + ' + str(b), 'label': a + b})
    return data

# Generate an additive data set for digits ranging from 3 to 40
additive_data = {}
for n in range(3, 41):
    additive_data[str(n)] = generate_additive_example(n) 
```

SUBTRACTION DATASET   
```python
# Generate two random n-digit numbers and their minus
def generate_minus_example(n):
    data = []
    k = (n - 1) % 2
    n = (n - 1) // 2 if n > 2 else n
    for _ in range(100000):
    a = random.randint(10**(n+k-1), 10**(n+k) - 1)
    b = random.randint(10**(n-1), 10**n - 1)
    if a > b:
    data.append({'input': str(a)+'-' + str(b), 'label': a - b})
    else:
    data.append({'input': str(b)+'-' + str(a), 'label': b - a})
    return data

# Generate an subtraction data set for digits ranging from 3 to 40 minus_data = {}
for n in range(3, 41):
    minus_data[str(n)] = generate_minus_example(n) 
```  
COIN FILP DATASET

```python
dataset = []
for i in range(500):
    # randomly choose two names from the name_list
    for o in range(2,5):
    sentence = 'A_coin_is_heads_up.'
    label = []
    for time in range(o):
    name = random.sample(names, k=1)[0]

    # randomly choose whether to flip the coin or not
    flip = random.choice([True, False])

    # generate the statement and label based on whether the coin was flipped or not
    if flip:
    sentence += f"{name.capitalize()}_flips_the_coin."
    label.append(1)
    else:
    sentence += f"{name.capitalize()}_does_not_flip_the_coin."
    label.append(0)
    sentence += 'Is_the_coin_still_heads_up?

    dataset.append({'question':sentence, 'answer':{0:'yes',1:'no'}[sum(label)%2]}) 
```

<NAME> token. In our work, flipping a coin corresponds to 1 and not flipping a coin corresponds to 0. To make the inputs as close to English as possible without using too many symbols, we used the sentence models "<NAME> flips the coin." and "<NAME> does not flip the coin." to represent whether the coin was flipped or not. This task is similar to the parity task, but requires further semantic understanding. We constructed a training set of 1500 samples, with 500 samples for each of 2-4 coin flips. For the test set, we selected 100 non-overlapping samples for each of 2-4 coin flips, and evaluated the model every 5 steps.

Last Letter Concatenation: We followed Wei et al. (2022b)'s setup and randomly concatenated first and last names from the top 1000 names in the census data to create the <NAME> token. This task requires the model to connect the last letter of each word in a concatenated name. This task requires Neural Comprehension of rules in two aspects. First, it requires the model to correctly identify the last letter of each word. Second, it requires the model to concatenate all the last letters of the words together. We concatenated 2-5 first or last names, and constructed a training set of 1500 samples, with 500 samples for each name length of 2-4. For the test set, we selected 100 non-overlapping samples for each name length of 2-4, and evaluated the model every 5 steps.

# F.2.3 ARITHMETICAL REASONING DATASET

In Table 11, we summarize the information of all arithmetic reasoning datasets used in this work. We provide the links to access these datasets:

- GSM8K: https://github.com/openai/grade-school-math   
- SingleEq: https://gitlab.cs.washington.edu/ALGES/TACL2015

<table><tr><td>Dataset</td><td>Number of samples</td><td>Average words</td><td>Answer Format</td><td>Lience</td></tr><tr><td>GSM8K</td><td>1319</td><td>46.9</td><td>Number</td><td>MIT License</td></tr><tr><td>SingleEq</td><td>508</td><td>27.4</td><td>Number</td><td>MIT License</td></tr><tr><td>AddSub</td><td>395</td><td>31.5</td><td>Number</td><td>Unspecified</td></tr><tr><td>MultiArith</td><td>600</td><td>31.8</td><td>Number</td><td>Unspecified</td></tr><tr><td>SVAMP</td><td>1000</td><td>31.8</td><td>Number</td><td>MIT License</td></tr></table>

Table 11: Arithmetical Reasoning Dataset Description.

- AddSub: https://www.cs.washington.edu/nlp/ arithmetic   
- MultiArith: http://cogcomp.cs.illinois.edu/page/resource\_view/98   
- SVAMP: https://github.com/arkilpatel/SVAMP

# G SOME EXAMPLES OF NEURAL COMPREHENSION

In this section, we will use gray font to represent the task input, yellow font to represent the neural network output during training, and blue background to represent the neural network output during generated.

# G.1 SYNTHETIC SYMBOLIC

```yaml
Q: 1011001010 A: 1
Q: 01111011000 A: 0
Q: 1010011001110 A: 1
Q: 10000001001001 A: 0
Q: 110100011110001 A: 0
Q: 1110011001010110 A: 1
Q: 1100000111011000101 A: 1
Q: 01100000110110010001 A: 0
——(LLM's few-shot prompt)—
Q: 011110001010101101011 A: 0 
```  
Table 12: The example of Parity

Q: neofascism A: msicsafoen   
Q: betaquinine A: eniniuqateb   
Q: corediastasis A: sisatsaideroc   
Q: ferroelectronic A: cinortceleorref   
Q: cryoprecipitation A: noitatipicerpoyrc   
Q: cryofibrinogenemia A: aimenegonirbifoyrc   
Q: chemocarcinogenesis A: sisenegonicracomehc   
Q: ponjpcdqjuuhiviojmby A: ybmjoivihuujqdcpjnop
——(LLM's few-shot prompt)——  
Q: helloworldhellochina A: anihcollehdlrowolleh

Table 13: The example of Reverse 

<table><tr><td>Q: 82637+3058 A: 85695</td></tr><tr><td>Q: 58020+96632 A: 154652</td></tr><tr><td>Q: 717471+58704 A: 776175</td></tr><tr><td>Q: 298309+702858 A: 1001167</td></tr><tr><td>Q: 1061462+2623780 A: 3685242</td></tr><tr><td>Q: 58720970+61609034 A: 120330004</td></tr><tr><td>Q: 364920479+78861480 A: 443781959</td></tr><tr><td>Q: 6050330002+211065324 A: 6261395326</td></tr><tr><td>——(LLM&#x27;s few-shot prompt)——</td></tr><tr><td>Q: 20021012+20021004 A: 40042016</td></tr></table>

Table 14: The example of Addition

<table><tr><td>Q: 82637-3058 A: 79579</td></tr><tr><td>Q: 96632-58020 A: 38612</td></tr><tr><td>Q: 717471-58704 A: 658767</td></tr><tr><td>Q: 702858-298309 A: 404549</td></tr><tr><td>Q: 2623780-1061462 A: 1562318</td></tr><tr><td>Q: 68720970-61609034 A: 7111936</td></tr><tr><td>Q: 364920479-78861480 A: 286058999</td></tr><tr><td>Q: 6050330002-211065324 A: 393967676——(LLM&#x27;s few-shot prompt)———</td></tr><tr><td>Q: 20021012-20021004 A: 8</td></tr></table>

Table 15: The example of Subtraction

G.2 SYMBOLIC REASONING   
```txt
A coin is heads up. Devin flips the coin. Maxwell does not flip the coin.
James flips the coin. Kenneth flips the coin. Is the coin still heads up?
1 0 1 1 -> 1

A coin is heads up. Ira flips the coin. Danny does not flip the coin.
Horace flips the coin. Is the coin still heads up?
1 0 1 -> 0 
```  
Table 16: The example of Coin Flip

```txt
Take the last letters of the words in Elias Earnest Milton and concatenate them.
The last letter of Elias -> s The last letter of Earnest -> t
The last letter of Milton -> n The answer is stn

Take the last letters of the words in Randolph Weldon Olin Robbieänd concatenate them.
The last letter of Randolph -> h The last letter of Weldon -> n
The last letter of Olin -> n The last letter of Robbie -> e The answer is hnne 
```  
Table 17: The example of Last Letter Concatenation

G.3 ARITHMETICAL RESONING   
```txt
Joan found 65466345746884996 seashells and Jessica found 38551966020636213  
seashells on the beach. How many seashells did they find together?  
Joan started with 65466345746884996 seashells. She gave some to Sam. So: 65466345746884996 - 38551966020636213 = 2691437972624878  
The answer is 2691437972624878 
```  
Table 18: The example of Arithmetirc Reasoning