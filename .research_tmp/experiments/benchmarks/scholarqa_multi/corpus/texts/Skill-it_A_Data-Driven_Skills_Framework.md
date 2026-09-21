# Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models

Mayee F. Chen $^{*1}$

Nicholas Roberts $^{2}$ Frederic Sala $^{2}$

Kush Bhatia $^{1}$ Jue Wang $^{3}$ Christopher Ré $^{1}$

Ce Zhang $^{3,4}$

$^{1}$ Department of Computer Science, Stanford University $^{2}$ Department of Computer Sciences, University of Wisconsin-Madison $^{3}$ Together AI $^{4}$ Department of Computer Science, University of Chicago

July 28, 2023

# Abstract

The quality of training data impacts the performance of pre-trained large language models (LMs). Given a fixed budget of tokens, we study how to best select data that leads to good downstream model performance across tasks. We develop a new framework based on a simple hypothesis: just as humans acquire interdependent skills in a deliberate order, language models also follow a natural order when learning a set of skills from their training data. If such an order exists, it can be utilized for improved understanding of LMs and for data-efficient training. Using this intuition, our framework formalizes the notion of a skill and of an ordered set of skills in terms of the associated data. First, using both synthetic and real data, we demonstrate that these ordered skill sets exist, and that their existence enables more advanced skills to be learned with less data when we train on their prerequisite skills. Second, using our proposed framework, we introduce an online data sampling algorithm, SKILL-IT, over mixtures of skills for both continual pre-training and fine-tuning regimes, where the objective is to efficiently learn multiple skills in the former and an individual skill in the latter. On the LEGO synthetic in the continual pre-training setting, SKILL-IT obtains 36.5 points higher accuracy than random sampling. On the Natural Instructions dataset in the fine-tuning setting, SKILL-IT reduces the validation loss on the target skill by 13.6% versus training on data associated with the target skill itself. We apply our skills framework on the recent RedPajama dataset to continually pre-train a 3B-parameter LM, achieving higher accuracy on the LM Evaluation Harness with 1B tokens than the baseline approach of sampling uniformly over data sources with 3B tokens.

# 1 Introduction

Large language models (LMs) exhibit remarkable capabilities, including producing creative content $[55]$ , writing source code $[8]$ , or chatting with users $[7]$ . A key ingredient in enabling models to perform such tasks is the data on which the models are trained $[17, 19, 59]$ . A natural way to unlock particular capabilities is to improve this training data. However, it is unclear how to select data from a large corpus for these capabilities given a fixed budget of training tokens, as data selection methods for current state-of-the-art LMs mostly rely on heuristics for filtering and mixing together different datasets $[32, 59]$ . We lack a formal framework for capturing how data influences the model's capabilities and how to utilize this data effectively for improving LM performance.

To develop such a framework, we take inspiration from how humans acquire knowledge. A classic idea in education literature is the concept of skills that form a learning hierarchy $[65]$ . For example, one study found that students learned mathematical and scientific skills most quickly when these skills were presented in a particular order $[11]$ . We seek to understand the extent that similar skill-based orderings characterize LM training. Such orderings, if they exist, may provide a better understanding of LMs as well as a mechanism for data-efficient training. For instance, to train an LM for Spanish question generation, we wish to know if training first on related but simpler tasks, such as Spanish grammar and English question generation, helps.

We study if the idea of skill orderings can help us build a framework that relates data to LM training and behavior. This requires addressing two challenges revolving around the connection between skills and data. First, in order to show that there exist sets of skills that the LM learns most efficiently in some particular order, an operational definition of LM skill and skill ordering must be developed and validated on data. In initial experiments, we investigated if semantic groupings of data, such as metadata attributes or embedding clusters, were sufficient to represent a skill and characterize how models learn. For

![](images/fe0b58ba1fd2d61662ab933dd45cc3bb5c1c4ef846c1cb67964a2dd5cfd5a19a.jpg)

<details>
<summary>text_image</summary>

p₁
p₂
p₃
p₄
Data
</details>

![](images/66297e712de83ba922ec299970a2bb93c81682f8d551a4c1948fd55bdbf176fb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1"] --> B["2"]
    A --> C["3"]
    A --> D["4"]
    B --> C
    B --> D
    C --> D
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
```
</details>

![](images/39102bf3ba36d0f547a7057c45a70742edc53d3eace9d3c4f2ac91ee3f6afdff.jpg)

<details>
<summary>line</summary>

| Training | Spanish QG Validation Loss |
| -------- | -------------------------- |
| 1        | 0.5                        |
| 2        | 0.3                        |
| 3        | 0.2                        |
| 4        | 0.1                        |
</details>

Figure 1: Inspired by how humans acquire knowledge, we hypothesize that LMs best learn skills in a particular order and that this can help improve our understanding and training of LMs. We show that these ordered skill sets exist in real data, which enables skills to be learned with less data given that we train on their prerequisite skills. We then propose SKILL-IT, an online data selection algorithm that learns skills quickly by exploiting their ordering.

instance, we partitioned the Alpaca dataset $[56]$ by instruction type—a technique used to capture dataset diversity $[62]$ —but we found that sampling based on instruction types and random sampling resulted in similar model performance, suggesting that not just any existing notion of data groups can characterize skills.

Second, these definitions of skills must be used to construct sampling distributions to actually improve model training. To develop criteria for a data selection algorithm that learns skills efficiently, we identify challenges that naive selection approaches face. The standard approach of random uniform sampling over data fails to learn skills optimally due to not accounting for skill imbalance and ordering. Skills can be distributed unevenly in the data, with more complex skills being rare—for instance, Spanish and question generation (QG) are 5% and 4% of the Natural Instructions dataset $[63]$ , respectively, but Spanish QG is only 0.2%. Random sampling also provides no mechanism for taking into account a particular training order and dependency structure on skills. More sophisticated techniques like curriculum learning account for sample-level ordering, but not skills or their dependencies. Our goal framework must account for these issues of imbalance and ordering.

Skill-based framework We define a skill as a unit of behavior that a model can learn using an associated slice of data (Definition 2.1). An ordered skill set is a collection of skills with a directed skills graph that is neither complete nor empty, where an edge from a prerequisite skill to a skill exists if the amount of training it takes to learn the skill can be reduced if the prerequisite skill is also learned (Definition 2.2, Figure 1 left, center). We show that ordered skill sets exist in synthetic and real datasets using this operational definition. Interestingly, the existence of these ordered skill sets unveils that one can learn a skill quickly not by training solely on that skill, but on a mixture of that skill and prerequisite skills. For instance, in Figure 3 we observe that Spanish QG can be learned more efficiently when the model also learns English QG and Spanish—we can achieve 4% lower validation loss than training on only Spanish QG over a fixed budget of overall training steps.

Next, given an ordered skill set to train on, we use our framework to propose methods for how to select data so that the LM learn skills faster: skill-stratified sampling and an online generalization, SKILL-IT. We address the issue of unevenly distributed skills in datasets by proposing skill-stratified sampling, a simple approach that allows us to explicitly optimize for learning skills by uniformly sampling relevant skills (such as a target skill and its prerequisite skills in fine-tuning). Skill-stratified sampling uses the construction of the ordered skill set but is static, which does not incorporate the ordering as training proceeds and results in oversampling skills that may be already learned early on in training. We address this issue by proposing an online data selection algorithm, SKILL-IT, for selecting mixtures of training skills that allocates more weight towards learning skills that are not yet learned or towards prerequisite influential skills (Figure 1 right). SKILL-IT is derived from an online optimization problem over the training skills for minimizing loss on a set of evaluation skills given a fixed budget of data and the skills graph. SKILL-IT is inspired by online mirror descent and can be adapted for continual pre-training, fine-tuning, or out-of-domain evaluation depending on the relationship between the evaluation skill set and the training skill set.

We evaluate SKILL-IT on synthetic and real datasets at two model scales, 125M and 1.3B parameters. For the continual pre-training setting, we show on the LEGO synthetic [72] that we obtain a 35.8 point improvement in accuracy over randomly selecting training data and curriculum learning [3]. For the fine-tuning setting, we show that on the widely-used Natural Instructions dataset [40, 64], our algorithm over a mixture of skills is able to achieve up to $13.6\%$ lower loss on that skill than solely training on that skill, given the same overall training budget. For the out-of-domain setting when our

![](images/f1dd30076dcff0d550cbb65f262e6a5720f8bc0cc1e901fe594d163163168edb.jpg)

<details>
<summary>heatmap</summary>

|        | 0.0  | 0.5  | 1.0  | 1.5  | 2.0  |
| ------ | ---- | ---- | ---- | ---- | ---- |
| Row 1  | 0.0  | 0.5  | 1.0  | 1.5  | 2.0  |
| Row 2  | 0.5  | 1.0  | 1.5  | 2.0  | 2.5  |
| Row 3  | 1.0  | 1.5  | 2.0  | 2.5  | 3.0  |
| Row 4  | 1.5  | 2.0  | 2.5  | 3.0  | 3.5  |
| Row 5  | 2.0  | 2.5  | 3.0  | 3.5  | 4.0  |
| Row 6  | 2.5  | 3.0  | 3.5  | 4.0  | 4.5  |
| Row 7  | 3.0  | 3.5  | 4.0  | 4.5  | 5.0  |
| Row 8  | 3.5  | 4.0  | 4.5  | 5.0  | 5.5  |
| Row 9  | 4.0  | 4.5  | 5.0  | 5.5  | 6.0  |
| Row10 | 4.5  | 5.0  | 5.5  | 6.0  | 6.5  |
| Row11 | 5.0  | 5.5  | 6.0  | 6.5  | 7.0  |
| Row12 | 5.5  | 6.0  | 6.5  | 7.0  | 7.5  |
| Row13 | 6.0  | 6.5  | 7.0  | 7.5  | 8.0  |
| Row14 | 6.5  | 7.0  | 7.5  | 8.0  | 8.5  |
| Row15 | 7.0  | 7.5  | 8.0  | 8.5  | 9.0  |
| Row16 | 7.5  | 8.0  | 8.5  | 9.0  |      |
| Row17 |      |      |      |      |      |
| Row18 |      |      |      |      |      |
| Row19 |      |      |      |      |      |
| Row20 |      |      |      |      |      |
| Row21 |      |      |      |      |      |
| Row22 |      |      |      |      |      |
| Row23 |      |      |      |      |      |
| Row24 |      |      |      |      |      |
| Row25 |      |      |      |      |      |
| Row26 |      |      |      |      |      |
| Row27 |      |      |      |      |      |
| Row28 |      |      |      |      |      |
| Row29 |      |      |      |      |      |
| Row30+|     nan|    nan|   nan|   nan|   nan|
</details>

![](images/204a0d43cbde6bad2d530f436fd49aa5a24e249d8721198f5a5b4fb3fca7980a.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|---|
| 0.0 | 0.0 | 0.0 |
| 0.0 | 0.5 | 0.5 |
| 0.0 | 1.0 | 1.0 |
| 0.0 | 1.5 | 1.5 |
| 0.5 | 0.0 | 0.0 |
| 0.5 | 0.5 | 0.5 |
| 0.5 | 1.0 | 1.0 |
| 0.5 | 1.5 | 1.5 |
| 1.0 | 0.0 | 0.0 |
| 1.0 | 0.5 | 0.5 |
| 1.0 | 1.0 | 1.0 |
| 1.0 | 1.5 | 1.5 |
| 1.5 | 0.0 | 0.0 |
| 1.5 | 0.5 | 0.5 |
| 1.5 | 1.0 | 1.0 |
| 1.5 | 1.5 | 1.5 |
| 2.0 | 0.0 | 0.0 |
| 2.0 | 0.5 | 0.5 |
| 2.0 | 1.0 | 1.0 |
| 2.0 | 1.5 | 1.5 |
| 2.5 | 0.0 | 0.0 |
| 2.5 | 0.5 | 0.5 |
| 2.5 | 1.0 | 1.0 |
| 2.5 | 1.5 | 1.5 |
| 3.0 | 0.0 | 0.0 |
| 3.0 | 0.5 | 0.5 |
| 3.0 | 1.0 | 1.0 |
| 3.0 | 1.5 | 1.5 |
| 3.5 | 0.0 | 0.0 |
| 3.5 | 0.5 | 0.5 |
| 3.5 | 1.0 | 1.0 |
| 3.5 | 1.5 | 1.5 |
| 4.0 | 0.0 | 0.0 |
| 4.0 | 0.5 | 0.5 |
| 4.0 | 1.0 | 1.0 |
| 4.0 | 1.5 | 1.5 |
| 4.5 | 0.0 | 0.0 |
| 4.5 | 0.5 | 0.5 |
| 4.5 | 1.0 | 1.0 |
| 4.5 | 1.5 | 1.5 |
| 5.0 | 0.0 | 0.0 |
| 5.0 | 0.5 | 0.5 |
| 5.0 | 1.0 | 1.0 |
| 5.0 | 1.5 | 1.5 |
| 5.5 | 0.0 | 0.0 |
| 5.5 | 0.5 | 0.5 |
| 5.5 | 1.0 | 1.0 |
| 5.5 | 1.5 | 1.5 |
| 6.0 | 0.0 | 0.0 |
| 6.0 | 0.5 | 0.5 |
| 6.0 | 1.0 | 1.0 |
| 6.0 | 1.5 | 1.5 |
| 6.5 | 0.0 | 0.0 |
| 6.5 | 0.5 | 0.5 |
| 6.5 | 1.0 | 1.0 |
| 6.5 | 1.5 | 1.5 |
| 7.0 | 0.0 | 0.0 |
| 7.0 | 0.5 | 0.5 |
| 7.0 | 1.0 | 1.0 |
| 7.0 | 1.5 | 1.5 |
| 7.5 | 0.0 | 0.0 |
| 7.5 | 0.5 | 0.5 |
| 7.5 | 1.0 | 1.0 |
| 7.5 | 1.5 | 1.5 |
| 8.0 | 0.0 | 0.0 |
| 8.0 | 0.5 | 0.5 |
| 8.0 | 1.0 | 1.0 |
| 8.0 | 1.5 | 1.5 |
| 8.5 | 0.0 | 0.0 |
| 8.5 | 0.5 | 0.5 |
| 8.5 | 1.0 | 1.0 |
| 8.5 | 1.5 | 1.5 |
| 9.0 | 0.0 | -16666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666
</details>

![](images/008113abf6d0afee9b813b3b376344c98772f80fec2cd00ba047db10f8060962.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|---|
| 1 | 1 | 0.8 |
| 2 | 2 | 0.6 |
| 3 | 3 | 0.4 |
| 4 | 4 | 0.2 |
| 5 | 5 | 0.0 |
</details>

Figure 2: Heatmaps of adjacency matrices we compute for skill graphs for Alpaca, Pile of Law, and Natural Instructions. Negative elements and diagonals are thresholded to 0 for clarity. See Appendix C.2 for descriptions of how they were constructed and larger versions.

training skills do not align perfectly with evaluation skills, our algorithm is able to achieve the lowest loss on 11 out of 12 evaluation skills corresponding to task categories in the Natural Instructions test tasks dataset over random and skill-stratified sampling on the training data. We finally apply our framework to a case study on the recent RedPajama 1.2 trillion token dataset [57]. We use the data mixture produced by SKILL-IT to continually pre-train a 3B parameter model. We find that SKILL-IT achieves higher accuracy with 1B tokens than uniform sampling over data sources with 3B tokens.

# 2 Skills framework

First, we propose definitions of skills and ordered skill sets in order to formalize our intuition around how models learn skills, and we demonstrate that not just any existing notion of data groups can characterize an ordered skill set in the dataset. Then, we demonstrate the existence of ordered skill sets on synthetic and real data, which show how viewing data through a skills-based framework can help with training and understanding model performance. Finally, we explore unsupervised skill recovery from data, finding that embedding-based approaches do not adequately recover synthetic skills.

# 2.1 Definitions

We first present a definition of an individual skill. Let the input space of all possible text data be X, where $x \in X$ is an individual text sample that a next-token-prediction LM $f \in F : X \to X$ is trained on. We quantify learning via a metric $L : F \times X \to R$ , which maps from a model and evaluation data to a scalar quantity. In our setup, we use the cross-entropy validation loss applied over next-token predictions as our metric L.

Definition 2.1 (Skill). A skill $s$ is a unit of behavior with associated data $\mathcal{X}_s \subseteq \mathcal{X}$ such that if $f$ is trained on an dataset $\mathcal{D}_s \subset \mathcal{X}_s$ , then $f$ has improved metric $L$ on samples belonging to $\mathcal{X}_s \backslash \mathcal{D}_s$ on average.

This definition of a skill is flexible—it simply means that given a training dataset associated with the skill, a model f has an improved metric (e.g., decreasing validation loss) when evaluated on validation data associated with this skill. Under this definition, a skill could be a granular task, such as Spanish question generation for a subset of Wikipedia articles, or can be defined over a data source, such as next-token prediction of legal data from tax court rulings. However, our next definition, the ordered skill set, has a more specific construction and provides a framework for how models learn across dependent skills.

Definition 2.2 (Ordered skill set, skills graph). An ordered skill set for f is a collection of skills $S = \{s_{1}, \ldots, s_{k}\}$ over which there is a directed skills graph $G = (\mathcal{S}, E)$ on the skill set that is neither complete or empty, where $(s_{i}, s_{j}) \in E$ if the amount of data needed to learn $s_{j}$ when uniformly sampling from $D_{s_{i}} \cup D_{s_{j}}$ is no more than the amount of data needed when sampling only from $D_{s_{j}}$ . We equate learning a skill $s_{j}$ to f attaining a certain value of L or lower on average over $X_{s_{j}} \setminus D_{s_{j}}$ .

This definition isolates complete and empty graphs as extrema that do not capture meaningful sets of skills. We discuss the three types of skill graphs—complete, empty, intermediate—and their implications for data selection. In particular, we discuss how several initial attempts of defining skills over datasets via semantic groupings resulted in the extrema cases (see Appendix C.2 for full results):

\- The complete graph demonstrates that all skills influence each other. A random partition is an example of a skill set that yields a complete graph. This graph suggests that the best approach for learning any skill or set of skills is random sampling on the dataset. This is not a setting where we can gain much with skill-based sampling. For example, using instruction types as skills on the Alpaca dataset results in a nearly complete estimated skills graph (97.4% dense), and we

![](images/206542c948e30ab22249ce070c6b8c38c212b72d8ad48aebbdd854520b998139.jpg)

<details>
<summary>line</summary>

| Steps | Trained on skill 3 | Trained on skills 1, 2, 3 |
| ----- | ------------------ | ------------------------- |
| 0     | 0.7                | 0.8                       |
| 1000  | 0.6                | 0.8                       |
| 2000  | 0.5                | 0.1                       |
| 3000  | 0.4                | 0.0                       |
| 4000  | 0.3                | 0.0                       |
| 5000  | 0.1                | 0.0                       |
| 6000  | 0.0                | 0.0                       |
</details>

![](images/b71b608f1f0287df838ee206eacb68e638007b63b8832946fb4b37b1863c0eb5.jpg)

<details>
<summary>line</summary>

| Steps | Trained on skill 1 | Trained on skills 1, 2 |
| ----- | ------------------ | ---------------------- |
| 0     | 2.2                | 2.2                    |
| 500   | 2.1                | 2.1                    |
| 1000  | 0.0                | 0.0                    |
| 1500  | 0.0                | 0.0                    |
| 2000  | 0.0                | 0.0                    |
| 2500  | 0.0                | 0.0                    |
| 3000  | 0.0                | 0.0                    |
| 3500  | 0.0                | 0.0                    |
| 4000  | 0.0                | 0.0                    |
| 4500  | 0.0                | 0.0                    |
| 5000  | 0.0                | 0.0                    |
| 5500  | 0.0                | 0.0                    |
| 6000  | 0.0                | 0.0                    |
</details>

![](images/4ae55ac403662ba41ee7529c6bcd38dc926d1dbb2f769e6c1f3fc47317e75292.jpg)

<details>
<summary>line</summary>

| Steps | Trained on Spanish QG Validation Loss | Trained on [Spanish, English] x [QA, QG] Validation Loss | Trained on stance detection Validation Loss | Trained on stance detection, text matching Validation Loss |
|-------|----------------------------------------|----------------------------------------------------------|---------------------------------------------|---------------------------------------------------------------|
| 0     | 2.8                                    | 2.8                                                      | 2.8                                         | 2.8                                                           |
| 100   | 2.4                                    | 2.4                                                      | 1.6                                         | 1.6                                                           |
| 200   | 2.4                                    | 2.4                                                      | 1.5                                         | 1.5                                                           |
| 300   | 2.4                                    | 2.4                                                      | 1.5                                         | 1.5                                                           |
| 400   | 2.4                                    | 2.4                                                      | 1.5                                         | 1.5                                                           |
| 500   | 2.4                                    | 2.4                                                      | 1.5                                         | 1.5                                                           |
| 600   | 2.4                                    | 2.4                                                      | 1.5                                         | 1.5                                                           |
</details>

Figure 3: On the LEGO synthetic, 3-digit addition, and Natural Instructions, we identify examples of ordered skill sets in which training on a mixture of skills helps learn an individual skill faster than just training on that skill itself, given a fixed training budget.

find that stratified sampling on these skills only improves validation loss per skill by 0.007 points over random sampling on average (Figure 2 left), suggesting that utilizing skills does not improve model performance in this case.

- The empty graph demonstrates that each skill is independent. This can occur if skills are too granular; for instance, learning Spanish math problems is unlikely to help with English poem generation. This graph suggests that the best approach for learning an individual skill is to train on the skill itself. We see that empty graphs exist in real data; in Figure 2 (center), using data sources as skills on the Pile of Law [21] results in a nearly empty skills graph (3.9% dense).   
- Graphs that are neither empty nor complete thus suggest a nontrivial order of how skill influence each other. This is the setting in which we expect that identifying skills and exploiting their ordering will help the most. In Figure 2 right, we use task categories, which capture broader reasoning patterns, as skills on Natural Instructions and find that the estimated graph has intermediate density (42.7% dense). We show concrete examples of how skills can be learned more efficiently on Natural Instructions in Section 2.2.

While these intuitive groupings result in ordered skill sets on some datasets (e.g., task categories on NI), this is not always the case (e.g., instruction types on Alpaca and sources on Pile of Law). Even though these groupings capture some notion of diversity in the dataset, our findings suggest that not just any semantic grouping induces an ordered skill set. We now empirically demonstrate that our definition of ordered skill sets aligns with how models learn and can be exploited for more data-efficient training.

# 2.2 Examples of skills and ordered skill sets

We provide examples of ordered skill sets on the LEGO synthetic dataset, an addition synthetic dataset, and subsets of the Natural Instructions dataset. On these datasets, we find that certain skills are better learned when trained along with their prerequisite skills rather than in isolation.

LEGO skills The LEGO synthetic, first introduced in [72], can evaluate a model's ability to follow a chain of reasoning. In this synthetic, the letters of the alphabet, $\mathcal{A}$ , are variables each with some binary label in $\{0,1\}$ . An individual sample consists of $k$ clauses for some fixed $k$ across the dataset, each of the form $a = gx$ where $a, x \in \mathcal{A}$ and $g$ is either a negation ("not") or assertion ("val"), e.g. we assign $a$ to the value of $x$ , or we assign $a$ to the opposite label. At the end of the sentence, we prompt the model for what the value of one of these variables is. Two samples $x \in \mathcal{X}$ are given below for $k = 5$ :

Input: b = not y, r = val 1, m = val b, q = val m, y = not r. Output: b = 1.

Input: c = val x, p = val f, x = val k, f = not c, k = val 0. Output: k = 0.

These samples each correspond to a chain of reasoning; for instance the first sample has the chain $r, y, b, m, q$ , where knowing $q$ 's label requires the most reasoning steps. We define the $i$ th skill $s_i$ as the model's ability to know the $i$ th variable of the chain. From our example above, the first sample belongs to $\mathcal{X}_{s_3}$ and the second sample belongs to $\mathcal{X}_{s_1}$ . To demonstrate the existence of ordered skill sets, we continually pre-train the 125M parameter GPT-Neo model [5, 13] over various mixtures of LEGO skills with $k = 5$ . In Figure 3 (left), we find that in $35.9\%$ fewer training steps, training on a balanced mixture of $\mathcal{X}_{s_1}, \mathcal{X}_{s_2}$ , and $\mathcal{X}_{s_3}$ resulted in the same validation loss of 0.01 as training solely on $\mathcal{X}_{s_3}$ . This suggests that $s_1, s_2$ helped unlock performance on $s_3$ and that there exist edges from $s_1$ or $s_2$ to $s_3$ in the skill graph. Additional observations are available in Appendix D.1, where we examine other edges as well as more complex reasoning chains, and the full skills graph corresponding to the ordered skill set for LEGO with $k = 5$ is in Figure 10.

Addition skills We consider a variant of a synthetic 5-digit addition dataset analyzed in [44]. We show the existence of ordered skill sets for a simplified 3-digit addition dataset where we treat each digit prediction as a skill—the outputs, in this case, are the integers $\{0,1,\dots,9\}$ . Examples are of the following form:

Input: A = 106 + 071, A 0 = ? Output: 7 Input: A = 606 + 879, A 2 = ? Output: 4

where ‘A 0’ refers to the ones digit of the output $(s_{1})$ and ‘A 2’ refers to the hundreds digit $(s_{3})$ . In Figure 3 (center), we find that in 32% fewer training steps, training on a balanced mixture of $X_{s_{1}}$ , and $X_{s_{2}}$ resulted in the same validation loss of 0.01 as training solely on $X_{s_{1}}$ . That is, the ones digit addition skill can be improved by simultaneously learning the tens digit addition skill, even though the former should not require information from the latter—this is in line with observations from prior work that models do not always learn the ones digit addition first [44]. The full skills graph corresponding to the ordered skill set over 3-digit addition is in Figure 11.

Natural Instructions (NI) skills We show that ordered skill sets exist in NI [63] when we treat task categories as skills.

- In Figure 3 (top right), we show that ordered skill sets exist over crosslingual task categories. Training on Spanish question generation (QG) along with equal parts of English QG, Spanish question answering (QA), and English QA results in $4.1\%$ lower validation loss than training only on Spanish QG. Remarkably, the former only uses $25\%$ of the latter's Spanish QG data. This suggests that there are edges from Spanish QA, English QA, and English QG to Spanish QG.   
- In Figure 3 (bottom right), we see that training on the task category Text Matching along with Stance Detection helps decrease the loss on Stance Detection by $11\%$ . This suggests that these categories, which both involve understanding the relationship between two input texts, share an edge.

The full skills graphs corresponding to the ordered skill sets over these task categories are in Figure 13. While equating task categories to skills may be noisy, these examples suggest that there is signal within real data that suggests that ordered skill sets can improve data efficiency.

# 2.3 Skill recovery

A final component of characterizing skills is unsupervised recovery of ordered skill sets. We consider embedding-based clustering approaches and a loss-based clustering approach for recovering LEGO skills. When clustering data using various trained and pre-trained embeddings, we find that they were unable to achieve above 39% accuracy on LEGO. Instead, we find that taking 10 random training runs and clustering data by their loss per timestep per run recovers the skills with 61% accuracy (Table 3). The intuition behind this method is that the validation losses on points from the same skill have similar trajectories as models learn. We discuss this approach more in Appendix D.2.

# 3 Skills-based data selection

Now that we have established the existence of ordered skill sets, we discuss how to use them for data selection. We state the data selection problem for learning across skills in Section 3.1. We discuss how to learn the skills graph that will be exploited in our data selection methods in Section 3.2. We then introduce two sampling methods that utilize the graph, a simple skill-stratified sampling method and the online sampling method SKILL-IT, in Section 3.3.

# 3.1 Problem statement

We are given an ordered training skill set $S_{train} = \{s_{train,1}, \ldots, s_{train,k}\}$ on the training data, each with associated support set $X_{s_{train,1}}, \ldots, X_{s_{train,k}}$ , and an ordered evaluation skill set $S_{eval} = \{s_{eval,1}, \ldots, s_{eval,m}\}$ of m evaluation skills on a separate evaluation dataset. We aim to select n samples from $S_{train}$ via a mixture of training skills, $p \in \Delta^{k-1}$ , to achieve three goals depending on how $S_{eval}$ is constructed:

\- Continual pre-training: when $S_{\text{eval}} = S_{\text{train}}$ , our goal is select a mixture of training skills to learn all of them.

\- Fine-tuning: when $S_{\text{eval}} \subset S_{\text{train}}$ , our goal is to select a mixture of training skills to learn an individual target skill or subset of these skills.

\- Out-of-domain: when $S_{\text{eval}} \cap S_{\text{train}} = \emptyset$ , our goal is to select a mixture of training skills to learn a disjoint set of evaluation skills we cannot train on. This can arise when we have a separate downstream validation dataset or the skills identified in the training dataset are noisy.

Furthermore, we have a skills graph $G = (\mathcal{S}_{\mathrm{train}} \cup \mathcal{S}_{\mathrm{eval}}, E)$ , where $E \subseteq S_{train} \times S_{eval}$ and $A \in R^{k \times m}$ is a weighted adjacency submatrix, where $A_{ij}$ describes the strength of the edge from $s_{train,i}$ to $s_{eval,j}$ . In Table 1, we summarize how the three different settings are constructed and how A varies across them. Next, we discuss how A can be estimated from the data.

# 3.2 Skills graph learning

The skills graph is important for determining how to sample from the ordered skill set for training efficiently. We present two approaches for learning the skills graph—brute-force and linear approximation. Algorithms are provided in Appendix B.2. By definition 2.2, the brute-force way of identifying edges involves fixing an overall training budget of H steps and 1)

<table><tr><td>Setting</td><td> $\mathcal{S}_{\text{eval}}$ </td><td>Skills graph</td></tr><tr><td>Continual pre-training</td><td> $\mathcal{S}_{\text{eval}} = \mathcal{S}_{\text{train}}$ </td><td> $A \in \mathbb{R}^{k \times k}$ , edges among all  $\mathcal{S}_{\text{train}}$ </td></tr><tr><td>Fine-tuning</td><td> $\mathcal{S}_{\text{eval}} \subset \mathcal{S}_{\text{train}}$ </td><td> $A \in \mathbb{R}^{k \times m}$ , edges from all training skills to target skill subset</td></tr><tr><td>Out-of-domain</td><td> $\mathcal{S}_{\text{eval}} \cap \mathcal{S}_{\text{train}} = \emptyset$ </td><td> $A \in \mathbb{R}^{k \times m}$ , edges from all training skills to separate evaluation skill set</td></tr></table>

Table 1: Summary of three settings—continual pre-training, fine-tuning, and out-of-domain. These settings are determined by how $S_{eval}$ is defined and result in different skills graphs used for our sampling methods.

<table><tr><td colspan="2">Algorithm 1: SKILL-IT Online Data Selection Algorithm</td></tr><tr><td>1:</td><td>Input: Ordered training skill set  $S_{train}$ , ordered evaluation skill set  $S_{eval}$ . Learning rate  $\eta$ ,  $T$  rounds,  $n$  samples,  $H$  training steps per run for graph learning, model  $f_1$ , window parameter  $w$ .</td></tr><tr><td>2:</td><td> $A \leftarrow \text{LEARNGRAPH}(S_{\text{train}}, S_{\text{eval}}, H, f_1)$  (Alg. 2, 3).</td></tr><tr><td>3:</td><td>Initialize  $p_1^i = \exp(\eta \sum_{j=1}^m A_{ij})$  for all  $i \in [k]$ , the softmax over  $A$ .</td></tr><tr><td>4:</td><td>for  $t = 1, \ldots, T - 1$  do</td></tr><tr><td>5:</td><td>Observe losses  $L_{\text{eval},j}(f_t)$  for all  $s_{\text{eval},j} \in S_{\text{eval}}$ .</td></tr><tr><td>6:</td><td>Train model  $f_t$  with  $n/T$  samples from mixture  $p_t$  over  $S_{train}$ . Update model  $f_{t+1} = \Phi(f_t, p_t)$ .</td></tr><tr><td>7:</td><td>Set  $p_{t+1}^i = \exp(\eta \sum_{\tau=t-w+1}^t \sum_{j=1}^m A_{ij} L_{\text{eval},j}(f_\tau))$ .</td></tr><tr><td>8:</td><td>end for</td></tr></table>

training and evaluating the model on each $s_{i}$ and 2) training the model on each pair of $(s_{i}, s_{j})$ and evaluating on $s_{i}$ and $s_{j}$ . If the loss on $s_{j}$ when trained on both $s_{i}$ and $s_{j}$ is lower, there exists an edge from $s_{i}$ to $s_{j}$ . This approach has runtime $\mathcal{O}(Hk^{2})$ , which is feasible for small k. When k is large, we can approximate this approach in linear time by training on each $s_{i}$ for h < H steps and setting $A_{ij} > 0$ if the loss on $s_{j}$ decreases over h steps for a runtime of $\mathcal{O}(hk)$ . This linear approach is necessary in the out-of-domain setting when $S_{eval}$ and $S_{train}$ are disjoint, as we do not train on data associated with $S_{eval}$ . In addition, both graph learning approaches can be performed on a smaller model, and the learned graph can be used for data selection for training a larger model (Appendix D.4).

# 3.3 Skills graph-aware sampling

We present two approaches for sampling over the mixture of training skills according to the skills graph: skill-stratified sampling, which samples uniformly over relevant training skills according to A, and SKILL-IT, which is an online generalization that incorporates knowledge of how skills are being learned throughout training.

# 3.3.1 Skill-stratified sampling

A straightforward sampling approach is to discard training skills that do not benefit the evaluation skills and sample uniformly over the set of relevant training skills, which we call skill-stratified sampling. For continual pre-training, the relevant skills are the entire training skill set; for each $s_{train,i} \in S_{train}$ , $\Pr(s_{\text{train},i}) = \frac{1}{k}$ . This enables each skill to have sufficient training data. For fine-tuning, the relevant skills are the target skills and prerequisite skills, which can be identified via positive entries of the ith column of A with $S_{prereq} = \{s_{train,i} : \exists s_{eval,j} \text{ s.t. } A_{ij} > 0\}$ . We then set $\Pr(s) = \frac{1}{|\mathcal{S}_{\text{prereq}} \cup \mathcal{S}_{\text{eval}}|}$ for $s \in S_{prereq} \cup S_{eval}$ . For the out-of-domain setting, skill-stratified sampling is over the set of prerequisite skills. For each $s \in S_{prereq}$ , we set $\Pr(s) = \frac{1}{|\mathcal{S}_{\text{prereq}}|}$ . Next, we propose our online algorithm that exploits the graph dynamically for more efficient training.

# 3.3.2 SKILL-IT online data selection algorithm

Despite accounting for prerequisite skills, one shortcoming of skill-stratified sampling is that even if a skill has already obtained sufficiently low validation loss early during training, we will continue to allocate the same weight to that skill throughout training. Therefore, we formulate our data selection problem as an online learning problem and propose SKILL-IT, which both prioritizes prerequisite skills and skills that are not yet learned.

We are given a budget of $T$ rounds and $n$ total samples to train on. At round $t$ , we select a mixture $p_t \in \Delta^{k-1}$ from the $k$ -dimensional unit simplex, and for each training skill $s_{\text{train},i} \in S_{\text{train}}$ , we sample from $\mathcal{X}_{s_{\text{train},i}}$ with proportion $p_t^i$ for a total of $\frac{n}{T}$ samples per round. Let $f_t$ be the model at at the start of round $t$ . We can define $f_t$ recursively as a function of the previous round's model $f_{t-1}$ and mixture $p_{t-1}$ via a dynamics function $\Phi: \mathcal{F} \times \Delta^{k-1} \to \mathcal{F}$ ; that is, $f_t = \Phi(f_{t-1}, p_{t-1})$ . Let $L_{\text{eval},j}(f_t)$ be the validation loss of $f_t$ on $s_{\text{eval},j}$ . Our goal is to select $p_1, \ldots, p_T$ to minimize loss per evaluation skill at

the end of training:

$$
\underset {p _ {1}, \dots , p _ {T} \in \Delta^ {k - 1}} {\text { minimize }} \frac {1}{m} \sum_ {j = 1} ^ {m} L _ {\text { eval }, j} (f _ {T}). \tag {1}
$$

This optimization problem is challenging to solve without additional assumptions. In order to make the problem tractable, we impose an explicit dynamics rule for the each evaluation skill's loss $L_{\mathrm{eval},j}$ in terms of the current loss and data mixture. Assuming for simplicity that $S_{\mathrm{eval}} \subseteq S_{\mathrm{train}}$ , a simple rule would be $L_{\mathrm{eval},j}(f_t) = L_{\mathrm{eval},j}(\Phi(f_{t-1}, p_{t-1})) := L_{\mathrm{eval},j}(f_{t-1})(1 - \alpha p_{t-1}^j)$ for $\alpha \in [0, 1]$ . That is, we expect that allocating more data to skill $j$ should result in the validation loss on skill $j$ decreasing. However, such an expression assumes that only training on the $j$ th skill will help learn the $j$ th skill. Instead, Section 2.2 suggests that there are other skills that may help with the $j$ th skill. We propose the following dynamics:

$$
L _ {\text { eval }, j} (f _ {t}) = L _ {\text { eval }, j} (f _ {t - 1}) (1 - A _ {:, j} ^ {\top} p _ {t - 1}), \tag {2}
$$

where $A_{:,j}$ is the column with weights of all skills that influence $s_{\mathrm{eval},j}$ , and we absorb the scalar $\alpha$ into $A$ . The optimization problem in (1) can thus be simplified as follows:

$$
\underset {p _ {1}, \dots , p _ {T} \in \Delta^ {k - 1}} {\text { minimize }} \frac {1}{m} \sum_ {j = 1} ^ {m} L _ {\text { eval }, j} (f _ {T}) \tag {3}
$$

$$
\mathrm{s.t} f _ {t} = \Phi (f _ {t - 1}, p _ {t - 1}) \forall t = 1, \dots T
$$

$$
L _ {\text { eval }, j} (f _ {t}) = L _ {\text { eval }, j} (f _ {t - 1}) (1 - A _ {:, j} ^ {\top} p _ {t - 1}) \forall j \in [ m ]
$$

In Appendix B, we derive the following update rule via online mirror descent [45] for learning rate $\eta > 0$ :

$$
p _ {t + 1} ^ {i} = p _ {t} ^ {i} \exp \left(\eta \sum_ {j = 1} ^ {m} A _ {i j} L _ {\text { eval }, j} (f _ {t})\right). \tag {4}
$$

In addition, when equation 4 is expanded, we have that $p_{t + 1}^{i} = p_{1}^{i}\exp \left(\eta \sum_{\tau = 1}^{t}\sum_{j = 1}^{m}A_{ij}L_{\mathrm{eval},j}(f_{\tau})\right)$ . Since this summation over $\tau$ results in diminishing strength of updates, we change it to a moving window of size $w$ . Our full method is in Algorithm 1.

Intuitively, at each step we adjust the weight on skill $i$ based on the losses of skills that $i$ influences, with the assumption that more training data helps decrease loss. Note that when we use our algorithm with a complete graph or empty graph, we achieve expected behavior discussed in Section 2.1. For the complete graph, our algorithm reduces to stratified sampling. When we have a skill set with an empty graph, the update rule reduces to sampling proportional to each skill's validation loss.

# 4 Experimental results

Given an ordered skill set, we aim to validate SKILL-IT's ability to select data for efficiently learning skills in the continual pre-training, fine-tuning, and out-of-domain settings. We provide full tables of results in Appendix D.3.1 and results where we learn the skills graph on the 125M model and use it for the 1.3B parameter model in Appendix D.4. Skills graphs are in Appendix C.2, weight trajectories for SKILL-IT are in Appendix D.3.2, and ablations on the graph and online components of SKILL-IT are in Appendix D.5.

# 4.1 Continual pre-training

Setup We evaluate the ability of SKILL-IT to select data for efficiently learning over all skills. We measure average validation loss per skill after a fixed number of training steps. We construct the LEGO synthetic and addition synthetic with k = 5 and 3, respectively, and an imbalanced dataset over the skills. On the Natural Instructions dataset, we use 23 of the task categories as skills.

Baselines We compare SKILL-IT against three baselines that do not account for skills: random sampling, curriculum learning, and anticurriculum learning. Random sampling is a standard procedure for selecting samples given no additional information. Curriculum learning $[3]$ and anticurriculum learning $[67]$ score the samples from easiest to hardest and vice versa, respectively, and sample over an expanding set of the lowest scored samples at every epoch; we use the pre-trained

![](images/7250ace7ef3c35bbe5e276ba3df9cc3cedcba82b097aedb06e89866f90e0a165.jpg)  
Figure 4: Performance of SKILL-IT on each skill in the continual pre-training setting (learning over all skills in the ordered training skill set) on the LEGO synthetic (left) and addition synthetic (right).

model's loss to rank points. We evaluate skill-stratified sampling, which uses knowledge of the skills but is not online, and include an additional skills curriculum baseline in Appendix D.3.1

Analysis Our results are shown in Figure 4. Across our experiments we find that SKILL-IT outperforms baselines that do not use skills as well as skill-stratified sampling. On the LEGO dataset, all three baselines that do not utilize a notion of skills exhibit plateauing loss on four of the skills. Both skill-stratified sampling and SKILL-IT are able to significantly reduce loss on all skills, but the former is slower. Halfway through training, SKILL-IT exhibits an accuracy improvement between 9.9 and 25.9 points over other approaches, reaching a final accuracy of 99.4 (Figure 19). SKILL-IT outperforms skill-stratified sampling by initially allocating more weight to prerequisite skills and eventually allocating more weight to skills that are learned more slowly (Figure 20). On the addition synthetic with k = 3, SKILL-IT converges to near-zero validation loss faster than the baselines on skills 1 and 2. While the random baseline may seem competitive at first glance, it fails to learn skill 1 (adding together the ones digits), which hurts its average loss per skill. On NI, the validation loss from SKILL-IT is 3.2% lower than from random sampling (Table 7). Our results suggest that exploiting the construction and ordering of skills is critical to learning skills quickly.

# 4.2 Fine-tuning

![](images/d8f41b5dce3c4d1aec738ceef823459d1c19ed4ac226d8472a834a213883ccad.jpg)

<details>
<summary>line</summary>

| Steps | Skill 3 only | Skill-stratified | Skill-It |
| ----- | ------------ | ---------------- | -------- |
| 0     | 0.75         | 0.80             | 0.75     |
| 1000  | 0.60         | 0.70             | 0.65     |
| 2000  | 0.40         | 0.30             | 0.20     |
| 3000  | 0.30         | 0.15             | 0.10     |
| 4000  | 0.20         | 0.10             | 0.05     |
| 5000  | 0.10         | 0.05             | 0.02     |
| 6000  | 0.05         | 0.02             | 0.01     |
</details>

![](images/a02074af30ba2c2e44d8e0eb852b67646b9f25598425e352e25ddee617c8dbf7.jpg)

<details>
<summary>line</summary>

| Steps | Skill 1 only | Skill stratified | Skill-It |
| ----- | ------------ | ---------------- | -------- |
| 0     | 2.5          | 2.5              | 2.5      |
| 1000  | 0.5          | 0.5              | 0.5      |
| 2000  | 0.1          | 0.1              | 0.1      |
| 3000  | 0.05         | 0.05             | 0.05     |
| 4000  | 0.02         | 0.02             | 0.02     |
| 5000  | 0.01         | 0.01             | 0.01     |
| 6000  | 0.0          | 0.0              | 0.0      |
</details>

![](images/29472f573ecaf681d5ee9f2d28ca78f742d2c95ee484457605283df7c0bbfeae.jpg)

<details>
<summary>line</summary>

| Steps | Spanish QG only | Skill-stratified | Skill-It |
| ----- | --------------- | ---------------- | -------- |
| 0     | 2.5             | 2.5              | 2.5      |
| 200   | 2.45            | 2.45             | 2.4      |
| 400   | 2.45            | 2.38             | 2.35     |
| 600   | 2.45            | 2.35             | 2.32     |
</details>

![](images/64d9fd798bddeb4013a275c39eac8c47e6d4f521ffeb1c30d1561794436d47bd.jpg)

<details>
<summary>line</summary>

| Steps | Stance detection only | Skill-stratified | Skill-It |
| ----- | --------------------- | ---------------- | -------- |
| 0     | 1.7                   | 1.7              | 1.7      |
| 200   | 1.5                   | 1.5              | 1.5      |
| 400   | 1.45                  | 1.35             | 1.3      |
| 600   | 1.45                  | 1.3              | 1.25     |
</details>

Figure 5: Performance of SKILL-IT in the fine-tuning setting (learning a target skill using the ordered training skill set) on LEGO, addition, and NI.

Setup We evaluate the ability of SKILL-IT to select data from an ordered training skill set for learning a target skill. Mirroring Figure 3, we evaluate on LEGO target skill 3 (third in reasoning chain), on the addition synthetic's skill 1 (ones place digit addition), and on NI's Spanish QG and Stance Detection.

Baselines We compare SKILL-IT against training on the target skill only and skill-stratified sampling over prerequisite skills and the target skill. The skill-stratified sampling approach uses the ordered skill set to identify prerequisite skills, but does not exploit them dynamically.

Analysis Our results are shown in Figure 5. On LEGO, SKILL-IT results in the same validation loss of 0.01 as training only on the target skill in 38.1% fewer steps. We observe a similar trend on addition, with SKILL-IT converging to a validation loss of 0.01 in 59% fewer steps required to do so when training only on the target skill. Finally, on NI, SKILL-IT improves validation loss on Spanish question generation by 5.3% and Stance Detection by 13.6% over just training on the respective

![](images/b1e16321831723bb1b0c0a07942911e10dbca5f3517a7e889277427c35f8c7a1.jpg)

<details>
<summary>line</summary>

| Model | Validation Loss |
|-------|-----------------|
| Green | 3.10            |
| Blue  | 3.08            |
| Orange| 3.08            |
</details>

![](images/8cea7e2087826ea85c31b3ca0a0579e1a0589e129a8ebf198970f70944a97620.jpg)

![](images/5021148e8ef927e328e291ddbfb8785b9d1ff0a51c5f9920cdc08a0160bac65a.jpg)

![](images/671a2f6bb0d17937122a829b1003d7b81f93e6e5870d17d16a553690af89d2bf.jpg)

![](images/09659a7b13f87d8d96bbfdcdf442dffa8499a7c88f8f81e2bc154ac9b229902e.jpg)

<details>
<summary>line</summary>

| Step | Validation Loss (Green) | Validation Loss (Blue) | Validation Loss (Orange) |
| ---- | ------------------------ | ----------------------- | ------------------------- |
| 0    | 2.38                     | 2.38                    | 2.38                      |
| 10   | 2.35                     | 2.36                    | 2.37                      |
| 20   | 2.34                     | 2.35                    | 2.36                      |
| 30   | 2.33                     | 2.34                    | 2.36                      |
| 40   | 2.33                     | 2.34                    | 2.36                      |
| 50   | 2.33                     | 2.34                    | 2.36                      |
| 60   | 2.33                     | 2.34                    | 2.36                      |
| 70   | 2.33                     | 2.34                    | 2.36                      |
| 80   | 2.33                     | 2.34                    | 2.36                      |
| 90   | 2.33                     | 2.34                    | 2.36                      |
| 100  | 2.33                     | 2.34                    | 2.36                      |
</details>

![](images/b66e2d384aeb296c8d130031c0da666bc949df777b87be34be0a496969711542.jpg)

![](images/d864dbdbd66c99a39eab72ebf999f4704cbefc71fbcdfe5c01b58726adcc58a2.jpg)

![](images/763f9149913785e7382d00ef671c978cc5e9521d38749b7c23da7576cbe200d8.jpg)

![](images/2708237b85d778a698d555ae26af819c96144b741ba84b78ea7003964dea6942.jpg)

<details>
<summary>line</summary>

| Steps | Validation Loss (Orange) | Validation Loss (Blue) | Validation Loss (Green) |
| ----- | ------------------------ | ---------------------- | ----------------------- |
| 0     | 2.63                     | 2.63                   | 2.63                    |
| 1000  | 2.62                     | 2.61                   | 2.59                    |
| 2000  | 2.62                     | 2.59                   | 2.57                    |
| 3000  | 2.62                     | 2.58                   | 2.56                    |
| 4000  | 2.62                     | 2.58                   | 2.56                    |
| 5000  | 2.62                     | 2.58                   | 2.56                    |
</details>

![](images/221acc0bdf447ae9816e092d3e27aed11df854d2a822191c5e126fd3f041f70e.jpg)

<details>
<summary>line</summary>

| Steps | Line 1 | Line 2 | Line 3 |
| ----- | ------ | ------ | ------ |
| 0     | 2.54   | 2.54   | 2.54   |
| 1000  | 2.50   | 2.50   | 2.50   |
| 2000  | 2.48   | 2.48   | 2.48   |
| 3000  | 2.47   | 2.47   | 2.47   |
| 4000  | 2.46   | 2.46   | 2.46   |
| 5000  | 2.45   | 2.45   | 2.45   |
</details>

![](images/b10bb68bcda99008b1a6ec7f2de57c2acec13c5f0759ac551c4d4c9c604cd2d6.jpg)

<details>
<summary>line</summary>

| Steps | Line 1 | Line 2 | Line 3 |
| ----- | ------ | ------ | ------ |
| 0     | 3.05   | 3.05   | 3.05   |
| 1000  | 3.04   | 3.04   | 3.04   |
| 2000  | 3.035  | 3.035  | 3.035  |
| 3000  | 3.03   | 3.03   | 3.03   |
| 4000  | 3.028  | 3.028  | 3.028  |
| 5000  | 3.027  | 3.027  | 3.027  |
</details>

![](images/b6a993ff497f53c6f95b5f3f887a1fb49e9ebbeca723f0b438e0e11ec17a75e0.jpg)

<details>
<summary>line</summary>

| Steps | Line 1 | Line 2 | Line 3 |
| ----- | ------ | ------ | ------ |
| 0     | 1.70   | 1.70   | 1.70   |
| 1000  | 1.68   | 1.69   | 1.68   |
| 2000  | 1.67   | 1.68   | 1.68   |
| 3000  | 1.67   | 1.68   | 1.68   |
| 4000  | 1.67   | 1.68   | 1.68   |
| 5000  | 1.67   | 1.68   | 1.68   |
</details>

Figure 6: Performance of SKILL-IT in the out-of-domain setting for the NI test task split. SKILL-IT uses the graph between the train and evaluation skills to produce an online mixture on the training dataset.   
![](images/7d6ffeafdc4b1f93d7de7385d731a003703393f3a75053f46a7ba37a947ab14d.jpg)

<details>
<summary>line</summary>

| Number of tokens (billion) | Uniform | Skill-It |
| --------------------------- | ------- | -------- |
| 0                           | 0.638   | 0.638    |
| 1                           | 0.642   | 0.644    |
| 2                           | 0.643   | 0.644    |
| 3                           | 0.643   | 0.644    |
</details>

<table><tr><td>RedPajama source</td><td>SKILL-IT mixture</td></tr><tr><td>ArXiv</td><td>0.1370</td></tr><tr><td>Books</td><td>0.0437</td></tr><tr><td>C4</td><td>0.4195</td></tr><tr><td>CommonCrawl</td><td>0.0732</td></tr><tr><td>GitHub</td><td>0.189</td></tr><tr><td>StackExchange</td><td>0.0892</td></tr><tr><td>Wikipedia</td><td>0.0484</td></tr></table>

Figure 7: Left: Accuracy on LM Evaluation Harness for continual pre-training of a 3B parameter model using SKILL-IT on the RedPajama dataset. We achieve higher accuracy at 1B additional tokens than uniform at 3B tokens. Right: SKILL-IT mixture over RedPajama sources.

target skill only. In this setting, a significant portion of the improvement over training only on the target skill comes from identification of prerequisite skills through the learned graph in the skill-stratified sampling method. SKILL-IT is further able to improve performance with finer-grained dynamic weighting on prerequisite skills.

# 4.3 Out-of-domain setting

Natural Instructions We evaluate the ability of SKILL-IT to select data from a set of training skills for learning a disjoint set of evaluation skills that we cannot train on. We use all 59 task categories in the NI train tasks split as the training skills and the 12 task categories in the test tasks split as our evaluation skills. We compare SKILL-IT against random and skill-stratified sampling, both of which do not exploit the relationships between training skills and evaluation skills. SKILL-IT achieves the lowest loss on 11 out of 12 task categories over random and skill-stratified sampling (Figure 6, tables in Appendix).

RedPajama We use SKILL-IT to produce a data mixture on the RedPajama dataset. The training skills are the data sources comprising the dataset, and the evaluation skills are several tasks from the Language Model Evaluation Harness $[14]$ . SKILL-IT with T = 1 (i.e. a static, graph-based mixture) yields the mixture in Figure 7 (right). We continually pre-train a 3B parameter model trained on one trillion tokens for three billion additional tokens using this mixture, and see that it outperforms uniform sampling over the data sources (Figure 7 left). In particular, SKILL-IT achieves higher accuracy with 1B additional tokens than uniform with 3B additional tokens.

# 5 Related work

Data selection for LMs There have been several studies of large-scale data selection for LMs. Data deduplication $[1, 22, 32]$ , in which identical or nearly identical samples are removed, is a method that enables LMs to be trained on smaller, cleaned datasets and has been increasingly used as a pre-processing step for training data $[4, 59, 71]$ . Other methods applied at scale involve ensuring high quality of data by explicitly filtering out samples or comparing the training dataset with a cleaned reference dataset $[7, 31, 59]$ . Importance reweighting approaches have also been proposed for identifying training data from a large corpus that best approximates a smaller target distribution $[69]$ , and influence functions have been used to select a subset of training data to improve performance on downstream tasks $[61]$ . These approaches can identify data

pertaining to a particular target distribution or filter out low quality data according to some heuristic, while our work aims to understand how the choice of data is related to the numerous skills that LMs learn.

Recent development of LMs has shifted focus from emphasizing the scale of the model to prioritizing the training data utilized. For example, models like Alpaca $[56]$ , Vicuna $[9]$ , and Koala $[15]$ are all based on the LLaMA model combined with instruction data generated by an existing LM. Palm 2's technical report states that the data mixture was a critical component of the final model $[17]$ , and Mosaic ML's recent MPT model was trained on a hand-engineered mixture of the RedPajama dataset $[42]$ . However, these works lack rigorous explanation for why their training datasets were constructed in this way.

Finally, perhaps most related to our approach is the contemporary work DoReMi $[68]$ , which uses group distributionally robust optimization on a smaller LM to select data source mixtures for training a larger LM. Their approach focuses on selecting data at the data source level for optimizing worst-case performance across the training data sources, rather than at the more general skills level for a variety of target skill sets. Furthermore, we focus on understanding how skills are related to each other and induce some order in how LMs learn by explicitly modeling skill graph structure, which we find to be important for data-efficient LM training (see ablations in Appendix D.5).

Data selection methods Many data selection methods have been proposed for supervised, task-specific settings. In this setting, the most typical objective is dataset condensation, which aims to identify a small subset of data that captures the larger dataset's properties with respect to the model. Some approaches include constructing coresets $[30, 47]$ , identifying samples that the model forgets during training $[58]$ ; identifying samples with the largest gradients $[46]$ or gradients that approximate the overall gradient $[39]$ ; clustering in embedding space and selecting points farthest from cluster centers $[53]$ ; and selecting samples with the highest uncertainty or entropy $[33]$ . These approaches have also been shown to transfer from smaller models to larger models $[10]$ . Unlike these methods, we study how to select data for learning one or many skills at the mixture level for LMs instead of the instance level.

Another area of interest is data selection for domain adaptation and multitask learning. For domain adaptation, there are a wide range of methods that select data to best match the target distribution. For example, the Moore-Lewis method matches data based on the difference in cross-entropy using a model trained on the target versus a model trained on the source data $[41]$ . Several other approaches suggest training a model to distinguish between source and target and selecting points with high uncertainty $[50]$ , or selecting points based on some divergence in an embedding space $[51]$ . In comparison to these approaches, our work focuses on learning one or many skills and also finds that embedding-based heuristics do not fully identify skills.

Data attribution Another perspective on understanding training data is data attribution, which seeks to identify what data is responsible for particular model behaviors. Influence functions $[28]$ and shapley values $[16]$ are two ways to quantify the role of individual samples. Datamodels $[23]$ fit a model to predict behavior given a subset of training data, providing a framework for understanding individual samples as well as dataset counterfactuals. Simfluence $[20]$ fits a Markov process to a set of training trajectories for finer-grained understanding of how data impacts training. We focus on understanding how groups of data associated with skills elicit broader model capabilities, and utilize this understanding to select data for more efficient training.

Curriculum learning Curriculum learning $[3]$ proposes to show the model data in order from easy samples to hard ones. Various criteria have been used to determine hardness, and anticurriculum as well as various pacing functions and mixing rates have been explored $[54]$ . Curriculum learning can also be performed at the group level $[60]$ . More sophisticated approaches include parametrizing each sample with a dynamic importance $[52]$ , and also accounting for irrelevant and noisy data $[38]$ . Our approach similarly utilizes a curriculum, but it is defined over a skills graph and does not necessarily align with training on easiest to hardest skills.

How LMs learn Many different explanations for how LMs learn from data have been proposed. One hypothesis is that there exist discrete, universal building blocks of LM knowledge called quanta, and power law scaling emerges from a learning over a particular distribution of quanta in the right order [37]. Another is that chain of thought reasoning emerges due to local clusters of latent variables that influence each other, which can be validated by studying the LM's ability to do conditional inference given intermediate variables [48]. Others have provided theoretical analysis of how transformers learn topics by studying co-occurrences of words in the training data [34]. Empirically, how models learn is still a mystery—for instance, models trained on code are found to perform fairly well at commensense reasoning [36]. Our work initiates a study on how LMs learn various skills and how to exploit this for better data selection.

Task selection In multitask auxiliary learning, the goal is to train a model to perform well on a target task(s) by selecting the most beneficial source tasks to train on. One can use feature similarity to select tasks $[29]$ , but we find in our synthetics that feature similarity does not always recover skills. In Taskonomy $[70]$ , a hypergraph over a set of tasks is learned and used to select tasks. The methods used to develop the taxonomy can be applied to further expand our graph learning (e.g., studying transitive and higher-order properties). However, their focus is on task selection in computer vision rather than data selection for LMs to learn skills. Lastly, the contemporary work of TaskWeb $[24]$ builds a graph among 22 common NLP tasks in order to determine what the best source tasks are for a target task. Their definition of an edge in the task graph is less strict than ours (their comparison is on if training on additional data from $s_{i}$ helps with $s_{j}$ , while we fix the overall amount of training data over both $s_{i}$ and $s_{j}$ ). Overall, our approach is similar in use of the skills graph, but we incorporate it into a dynamic sampling algorithm. Furthermore, we look more broadly at skills, rather than tasks, and characterize when we expect using the skills graph to improve model performance.

Education The notion of skill has been studied in education. Classical research on learning hierarchies $[66]$ identify sets of skills that make up subordinate capabilities for students. For instance, $[12]$ identified that in order for students to solve linear equations, there were many prerequisite skills, ranging from the simplest being symbol recognition to the most complex being the ability to add, subtract, multiple, and divide from both sides of the equation. More recently, decision-making over lesson sequences based on skills, e.g., what the student already knows versus what the lesson teaches, has become an area of interest in personalized learning $[49]$ .

# 6 Conclusion

Given a fixed budget of data, knowing what data to train on to induce various capabilities in an LM is challenging. As LMs continue to improve, it will become increasingly important to extract as much signal as possible from the data and to direct that signal towards acquiring a broad variety of capabilities. In this paper, we introduce a skills-based framework for understanding how LMs learn and for selecting training data. We hope our study invites others to build on such a notion of skill and further explore how to align skills with data.

# Acknowledgements

We thank Together Computer (https://together.xyz/) for providing portions of the compute used to train models in this paper. We thank Sabri Eyuboglu, Karan Goel, Arjun Desai, Neel Guha, Michael Zhang, Vishnu Sarrukai, Simran Arora, Ben Spector, Brandon Yang, Gautam Machiraju, and Sang Michael Xie for their helpful feedback and discussion.

We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize), NSF under Nos. CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML); US DEVCOM ARL under No. W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under No. N000141712266 (Unifying Weak Supervision); ONR N00014-20-1-2480: Understanding and Applying Non-Euclidean Geometry in Machine Learning; N000142012275 (NEPTUNE); NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce, Total, the HAI-GCP Cloud Credits for Research program, the Stanford Data Science Initiative (SDSI), and members of the Stanford DAWN project: Facebook, Google, and VMware. FS is supported by NSF CCF2106707 and the Wisconsin Alumni Research Foundation (WARF). The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views, policies, or endorsements, either expressed or implied, of NIH, ONR, or the U.S. Government.

# References

[1] Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S. Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication, 2023.   
[2] Yuntao Bai, Andy Jones, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022.   
[3] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.   
[4] Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for analyzing large language models across training and scaling, 2023.   
[5] Sid Black, Gao Leo, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, March 2021. If you use this software, please cite it using these metadata.   
[6] Rishi Bommasani, Percy Liang, et al. On the opportunities and risks of foundation models, 2021.   
[7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.   
[8] Mark Chen, Jerry Tworek, et al. Evaluating large language models trained on code, 2021.   
[9] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, March 2023.   
[10] Cody Coleman, Christopher Yeh, Stephen Mussmann, Baharan Mirzasoleiman, Peter Bailis, Percy Liang, Jure Leskovec, and Matei Zaharia. Selection via proxy: Efficient data selection for deep learning, 2019.   
[11] Robert M Gagne. The acquisition of knowledge. Psychological review, 69(4):355, 1962.   
[12] Robert M Gagne and Noel E Paradise. Abilities and learning sets in knowledge acquisition. Psychological Monographs: General and Applied, 75(14):1, 1961.   
[13] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.   
[14] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, September 2021.   
[15] Xinyang Geng, Arnav Gudibande, Hao Liu, Eric Wallace, Pieter Abbeel, Sergey Levine, and Dawn Song. Koala: A dialogue model for academic research. Blog post, April 2023.   
[16] Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning. In International Conference on Machine Learning, pages 2242–2251. PMLR, 2019.   
[17] Google. Palm2 technical report. Technical report, 2023.   
[18] Anupam Gupta. Advanced algorithms: Notes for cmu 15-850 (fall 2020), 2020.   
[19] Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. Don’t stop pretraining: Adapt language models to domains and tasks. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020.   
[20] Kelvin Guu, Albert Webson, Ellie Pavlick, Lucas Dixon, Ian Tenney, and Tolga Bolukbasi. Simfluence: Modeling the influence of individual training examples by simulating training runs, 2023.   
[21] Peter Henderson\*, Mark S. Krass\*, Lucia Zheng, Neel Guha, Christopher D. Manning, Dan Jurafsky, and Daniel E. Ho. Pile of law: Learning responsible data filtering from the law and a 256gb open-source legal dataset, 2022.

[22] Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, Scott Johnston, Ben Mann, Chris Olah, Catherine Olsson, Dario Amodei, Nicholas Joseph, Jared Kaplan, and Sam McCandlish. Scaling laws and interpretability of learning from repeated data, 2022.   
[23] Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Predicting predictions from training data, 2022.   
[24] Joongwon Kim, Akari Asai, Gabriel Ilharco, and Hannaneh Hajishirzi. Taskweb: Selecting better source tasks for multi-task nlp, 2023.   
[25] Hannah Rose Kirk, Yennie Jun, Filippo Volpin, Haider Iqbal, Elias Benussi, Frederic Dreyer, Aleksandar Shtedritski, and Yuki Asano. Bias out-of-the-box: An empirical analysis of intersectional occupational biases in popular generative language models. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 2611–2624. Curran Associates, Inc., 2021.   
[26] Nikita Kitaev, Steven Cao, and Dan Klein. Multilingual constituency parsing with self-attention and pre-training. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3499–3505, Florence, Italy, July 2019. Association for Computational Linguistics.   
[27] Nikita Kitaev and Dan Klein. Constituency parsing with a self-attentive encoder. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2676–2686, Melbourne, Australia, July 2018. Association for Computational Linguistics.   
[28] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions, 2017.   
[29] Po-Nien Kung, Sheng-Siang Yin, Yi-Cheng Chen, Tse-Hsuan Yang, and Yun-Nung Chen. Efficient multi-task auxiliary learning: Selecting auxiliary data by feature similarity. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 416–428, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics.   
[30] Michael Langberg and Leonard J. Schulman. Universal approximators for integrals, pages 598–607.   
[31] Hugo Laurençon, Lucile Saulnier, et al. The bigscience roots corpus: A 1.6tb composite multilingual dataset, 2023.   
[32] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022.   
[33] David D Lewis. A sequential algorithm for training text classifiers: Corrigendum and additional data. In Acm Sigir Forum, volume 29, pages 13–19. ACM New York, NY, USA, 1995.   
[34] Yuchen Li, Yuanzhi Li, and Andrej Risteski. How do transformers learn topic structure: Towards a mechanistic understanding, 2023.   
[35] Paul Pu Liang, Chiyu Wu, Louis-Philippe Morency, and Ruslan Salakhutdinov. Towards understanding and mitigating social biases in language models. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 6565–6576. PMLR, 18–24 Jul 2021.   
[36] Aman Madaan, Shuyan Zhou, Uri Alon, Yiming Yang, and Graham Neubig. Language models of code are few-shot commonsense learners, 2022.   
[37] Eric J. Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural scaling, 2023.   
[38] Sören Mindermann, Muhammed Razzak, Winnie Xu, Andreas Kirsch, Mrinank Sharma, Adrien Morisot, Aidan N. Gomez, Sebastian Farquhar, Jan Brauner, and Yarin Gal. Prioritized training on points that are learnable, worth learning, and not yet learned (workshop version), 2021.   
[39] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models, 2019.   
[40] Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In ACL, 2022.

[41] Robert C. Moore and William Lewis. Intelligent selection of language model training data. In Proceedings of the ACL 2010 Conference Short Papers, pages 220–224, Uppsala, Sweden, July 2010. Association for Computational Linguistics.   
[42] MosaicML. Introducing mpt-7b: A new standard for open-source, commercially usable llms, 2023.   
[43] Moin Nadeem, Anna Bethke, and Siva Reddy. Stereoset: Measuring stereotypical bias in pretrained language models. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2021.   
[44] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In The Eleventh International Conference on Learning Representations, 2023.   
[45] Arkadij Semenovič Nemirovskij and David Borisovich Yudin. Problem complexity and method efficiency in optimization. 1983.   
[46] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training, 2021.   
[47] Jeff M. Phillips. Coresets and sketches, 2016.   
[48] Ben Prystawski and Noah D. Goodman. Why think step-by-step? reasoning emerges from the locality of experience, 2023.   
[49] Siddharth Reddy, Igor Labutov, and Thorsten Joachims. Latent skill embedding for personalized lesson sequence recommendation, 2016.   
[50] Sebastian Ruder, Parsa Ghaffari, and John G. Breslin. Data selection strategies for multi-domain sentiment analysis, 2017.   
[51] Sebastian Ruder and Barbara Plank. Learning to select data for transfer learning with bayesian optimization. Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, 2017.   
[52] Shreyas Saxena, Oncel Tuzel, and Dennis DeCoste. Data parameters: A new family of parameters for learning a differentiable curriculum. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.   
[53] Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S. Morcos. Beyond neural scaling laws: beating power law scaling via data pruning, 2022.   
[54] Petru Soviany, Radu Tudor Ionescu, Paolo Rota, and Nicu Sebe. Curriculum learning: A survey. International Journal of Computer Vision, 130(6):1526–1565, Apr 2022.   
[55] Claire Stevenson, Iris Smal, Matthijs Baas, Raoul Grasman, and Han van der Maas. Putting gpt-3's creativity to the (alternative uses) test, 2022.   
[56] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford\_alpaca, 2023.   
[57] Together. Redpajama-data: An open source recipe to reproduce llama training dataset, 2023.   
[58] Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J. Gordon. An empirical study of example forgetting during deep neural network learning, 2018.   
[59] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.   
[60] Neeraj Varshney, Swaroop Mishra, and Chitta Baral. Let the model decide its curriculum for multitask learning, 2022.   
[61] Xiao Wang, Weikang Zhou, Qi Zhang, Jie Zhou, Songyang Gao, Junzhe Wang, Menghan Zhang, Xiang Gao, Yunwen Chen, and Tao Gui. Farewell to aimless large-scale pretraining: Influential subset selection for language model, 2023.   
[62] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions, 2022.

[63] Yizhong Wang, Swaroop Mishra, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks, 2022.   
[64] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners, 2021.   
[65] Richard T White. Research into learning hierarchies. Review of Educational Research, 43(3):361–375, 1973.   
[66] Richard T. White and Robert M. Gagné. Past and future research on learning hierarchies. Educational Psychologist, 11(1):19–28, 1974.   
[67] Xiaoxia Wu, Ethan Dyer, and Behnam Neyshabur. When do curricula work?, 2020.   
[68] Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, and Adams Wei Yu. Doremi: Optimizing data mixtures speeds up language model pretraining, 2023.   
[69] Sang Michael Xie, Shibani Santurkar, Tengyu Ma, and Percy Liang. Data selection for language models via importance resampling, 2023.   
[70] Amir R. Zamir, Alexander Sax, William Shen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, Jun 2018.   
[71] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022.   
[72] Yi Zhang, Arturs Backurs, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, and Tal Wagner. Unveiling transformers with lego: a synthetic reasoning task, 2022.

# A Broader Impacts and Limitations

Broader Impacts As more LMs are developed, a key criteria for their adoption and utility is if they exhibit a wide array of useful capabilities, such as generating harmless content, summarizing essays, and being conversational with the user. While improvements in other parts of the LM development pipeline such as training and architecture are important, many recent advances in building LMs with a wide array of useful capabilities have come from the data itself $[9, 15, 17, 42, 56]$ . Our work is fundamental in investigating how LMs learn and how to select data to learn skills more efficiently. However, we recognize that data selection methods can always be utilized to optimize for particular skills that may be considered malicious or negatively target or exclude specific groups $[2]$ . Furthermore, pre-trained LMs have been found to have various biases $[6, 25, 35, 43]$ .

Limitations The skills graph can either be provided (e.g., using a knowledge graph) or learned. Our work learns the skills graph using Algorithm 2 or Algorithm 3, which requires initial training runs on pairs of skills or each skill, respectively. This can be made more efficient by performing these training runs on a smaller model and for fewer number of steps, but tradeoffs here have yet to be thoroughly investigated. SKILL-IT also assumes that the ordered skill set is provided; as discussed in sections 2.1 and 2.3, it is challenging to recover ordered skill sets simply via metadata attributes or embedding clustering. Otherwise, the best way to sample over collections of skills that form a complete or empty graph is random or stratified sampling with no ordering to exploit. Our loss-based clustering approach presented in section 2.3 demonstrates that grouping by losses can provide an explanation for how skills are defined over data. An important direction for future work is to use such a clustering approach or other unsupervised algorithms in an end-to-end pipeline for skill discovery, skill graph learning, and data selection based on such skills.

# B Additional Algorithmic Details

# B.1 Derivation of SKILL-IT Update Rule

First, we provide the derivation of our update rule from online mirror descent using the proximal point view [18]. We restate our optimization problem from (3):

$$
\underset {p _ {1}, \dots , p _ {T} \in \Delta^ {k - 1}} {\text { minimize }} \frac {1}{m} \sum_ {j = 1} ^ {m} L _ {\text { eval }, j} (f _ {T}) \tag {5}
$$

$$
\text { s.t } \quad L _ {\text { eval }, j} (f _ {t}) = L _ {\text { eval }, j} (f _ {t - 1}) (1 - \alpha A _ {:, j} ^ {\top} p _ {t - 1})   \forall j \in [ m ], t = 1, \dots , T
$$

$$
f _ {t} = \Phi (f _ {t - 1}, p _ {t - 1}) \forall t = 1, \ldots T
$$

Let $\bar{L}_t(p) = \frac{1}{m}\sum_{i = j}^{m}L_{\mathrm{eval},j}(f_{t + 1}) = \frac{1}{m}\sum_{i = j}^{m}L_{\mathrm{eval},j}(\Phi (f_t,p))$ ; that is, $p$ is the mixture we must choose at time $t$ and $\bar{L}_t$ is the average loss per skill of the model after it is trained on $p$ at round $t$ . A greedy approximation of (5) is minimize $\bar{L}_t(p)$ , given the model and mixtures at previous rounds. A linear approximation of $\bar{L}_t(p)$ is

$$
\bar {L} _ {t} (p) \approx \bar {L} _ {t} (p _ {t - 1}) + \langle \nabla \bar {L} _ {t - 1} (p _ {t - 1}), p - p _ {t - 1} \rangle \tag {6}
$$

Then, the problem of minimizing $\bar{L}_{t}(p)$ can be approximated as

$$
\operatorname{argmin} _ {p \in \Delta^ {k - 1}} \left\langle \eta \nabla \bar {L} _ {t - 1} (p _ {t - 1}), p \right\rangle \tag {7}
$$

after we drop terms from (6) that do not depend on p. Note that the $\eta$ is a constant and does not impact the solution. The optimal solution to this problem is selecting the p that has the most weight on the slice with the largest gradient (e.g., a follow-the-leader sort of algorithm). To improve stability and prevent overfitting, we introduce regularization via a Bregman divergence $D_{h}(p||p_{t-1}) = h(p) - h(p_{t-1}) - \langle \nabla h(p_{t-1}), p - p_{t-1} \rangle$ . After dropping terms that do not contain p, our problem is now

$$
\operatorname{argmin} _ {p \in \Delta^ {k - 1}} \left\langle \eta \nabla \bar {L} _ {t - 1} \left(p _ {t - 1}\right), p \right\rangle + h (p) - \left\langle \nabla h \left(p _ {t - 1}\right), p \right\rangle \tag {8}
$$

Taking the gradient and setting it equal to 0 gives us

$$
\eta \nabla \bar {L} _ {t - 1} (p _ {t - 1}) + \nabla h (p) - \nabla h (p _ {t - 1}) = 0 \tag {9}
$$

Algorithm 2: LEARNGRAPH (Brute-Force)   
1: Input: Ordered skill set $S = \{s_{1}, \ldots, s_{k}\}$ . Number of training steps H, base model f.
2: for $j \in [k]$ do
3: Train f on samples from $X_{s_{j}}$ for H steps and denote $f_{H,j}$ to be the model after training.
4: Observe change in loss, $\delta_{j}^{j} = L_{\text{eval},j}(f) - L_{\text{eval},j}(f_{H,j})$ .
5: end for
6: for $i, j \in [k]$ do
7: Train f on samples from $X_{s_{i}} \cup X_{s_{j}}$ for H steps and denote $f_{H,i,j}$ to be the model after training.
8: Observe change in loss, $\delta_{j}^{i,j} = L_{\text{eval},j}(f) - L_{\text{eval},j}(f_{H,i,j})$ .
9: if $\delta_{j}^{ij} > \delta_{j}^{j}$ then
10: Draw edge $s_{i} \to s_{j}$ and set $A_{ij} > 0$ .
11: end if
12: end for
13: Return Adjacency matrix $A \in R^{k \times k}$

Algorithm 3: LEARNGRAPH (Approximate)   
1: Input: Ordered skill sets $S_{train}$ and $S_{eval}$ . Number of training steps H, base model f.
2: for $i \in [k]$ do
3: Train f on samples from $X_{s_{train,i}}$ for H steps and denote $f_{H,i}$ to be the model after training.
4: for $j \in [m]$ do
5: Observe change in loss, $\delta_{j}^{i} = L_{\text{eval},j}(f) - L_{\text{eval},j}(f_{H,i})$ .
6: If $\delta_{j}^{i} > 0$ , draw edge $s_{train,i} \to s_{eval,j}$ and set $A_{ij} > 0$ .
7: end for
8: end for
9: Return Bipartite adjacency submatrix $A \in R^{k \times m}$

Similar to in standard multiplicative weights, we set $h(p) = \sum_{i} p_{i} \ln p_{i}$ and $\nabla h(p) = [\ln p_{i} + 1]_{i}$ . Then,

$$
\begin{array}{l} \ln p ^ {i} = \ln p _ {t - 1} ^ {i} - \eta \nabla_ {i} L _ {t - 1} (p _ {t - 1}) \\ \Rightarrow p _ {t + 1} ^ {i} = p _ {t} ^ {i} \exp (- \eta \nabla_ {i} \bar {L} _ {t} (p _ {t})) \tag {10} \\ \end{array}
$$

where $\nabla_{i}$ is the $i$ th element of the gradient. Now we wish to compute $\nabla_{i}\bar{L}_{t}(p_{t}) = \frac{1}{m}\sum_{j = 1}^{m}\nabla_{i}[L_{\mathrm{eval},j}(f_{t + 1})] = \frac{1}{m}\sum_{j = 1}^{m}\nabla_{i}[L_{\mathrm{eval},j}(\Phi (f_{t},p_{t}))]$ . Recall the dynamics model for $L_{\mathrm{eval}}$ :

$$
L _ {\text { eval }, j} (f _ {t + 1}) = L _ {\text { eval }, j} (f _ {t}) (1 - A _ {:, j} ^ {\top} p _ {t}), \tag {11}
$$

The gradient of this model with respect to each training skill $s_{i}$ is

$$
\nabla_ {i} L _ {\text { eval }, j} (f _ {t + 1}) = - A _ {i j} L _ {\text { eval }, j} (f _ {t}) \tag {12}
$$

$$
\Rightarrow \nabla_ {i} \bar {L} _ {t} (p _ {t}) = \frac {1}{m} \sum_ {j = 1} ^ {m} - A _ {i j} L _ {\text { eval }, j} (f _ {t})
$$

Plugging this back into (10),

$$
p _ {t + 1} ^ {i} = p _ {t} ^ {i} \exp \left(\eta \sum_ {j = 1} ^ {m} A _ {i j} L _ {\text { eval }, j} (f _ {t})\right), \tag {13}
$$

where we can absorb the $\frac{1}{m}$ into $\eta$ .

# B.2 Graph Learning Method

We provide algorithms for learning the graph over an ordered skill set. In Algorithm 2, we discuss the brute-force approach for learning the adjacency matrix. This approach only works when $S_{eval} \subseteq S_{train}$ (e.g. pre-training and fine-tuning

cases), so we denote $S = S_{train}$ in the algorithm box. In Algorithm 3, we discuss the linear approach for learning the adjacency matrix. This approach works even in the out-of-domain case when $S_{eval}$ and $S_{train}$ are disjoint.

In both approaches, the exact value of $A_{ij}$ can vary, but we can typically set it proportional to $\delta_{j}^{i,j} - \delta_{j}^{j}$ , the difference between the changes in loss, in the brute-force case or $\delta_{j}^{i}$ , the change in loss itself, in the approximate case. The exact constructions and methods for learning each A in our experiments are in Appendix C.2.

# C Additional Experimental Details

# C.1 Datasets

We present details about each dataset used, including information on the skills and the validation dataset. A summary is presented in Table 2.

<table><tr><td>Dataset</td><td>Skill</td><td># skills</td><td>Validation data</td></tr><tr><td>Alpaca</td><td>Instruction type</td><td>38</td><td>50 samples per skill</td></tr><tr><td>Pile of Law</td><td>Legal data source</td><td>31</td><td>645 samples per skill</td></tr><tr><td>LEGO</td><td>Reasoning chain depth</td><td>5</td><td>100 samples per skill</td></tr><tr><td>Addition</td><td>Digit</td><td>3</td><td>100 samples per skill</td></tr><tr><td>NI (pre-training)</td><td>Task category</td><td>23</td><td>50 samples per task</td></tr><tr><td>NI (Spanish QG)</td><td>Task category × language</td><td>4</td><td>100 samples per task</td></tr><tr><td>NI (stance detection)</td><td>Task category</td><td>2</td><td>50 samples per task</td></tr><tr><td>NI (out-of-domain)</td><td>Task category</td><td>59, 12</td><td>400 samples per task</td></tr><tr><td>RedPajama</td><td>Data source</td><td>7</td><td>LM eval harness</td></tr></table>

Table 2: We list each dataset used as well as its corresponding skill. We include the number of skills in the training dataset, as well as details on how the validation dataset is constructed.

- Alpaca dataset [56]: the Alpaca dataset consists of 52K instruction examples that were generated from text-davinci-003. We applied the Berkeley Neural Parser [26, 27] to each instruction, keeping 40777 samples it was able to parse successfully. If the sample began with a question, we annotated it with the skill “question”, and otherwise we annotated it with the verb identified from the parser. We grouped the data into a total of 38 skills, such as "list", "edit", "calculate", "describe" and "identify".   
- Pile of Law [21]: the Pile of Law dataset consists of various sources of legal and administrative data, ranging from tax rulings to the world's constitutions. We evaluate on a subset of the Pile of Law validation dataset consisting of 13883 samples, where we selected max(645, source size) samples per source. We truncated each sample to be no more than 100K characters.   
- LEGO [72]: for the LEGO synthetic, we set $k = 5$ and sample 192000 points across the skills. Our validation dataset consisted of 100 samples per skill.   
- Addition: for the 3-digit addition synthetic, we set $k = 3$ and sample 192000 points across the skills. We use a validation dataset of 100 samples per skill.   
- Natural Instructions [40, 63]: the Natural Instructions dataset is a large collection of tasks and their definitions in natural language. For the pre-training setting, we used a set of 23 task categories that had the largest degree (in-degree + out-degree) in the learned skills graph, for a total of 1,232,437 samples and 425 tasks to select from. We evaluated on 50 samples per task.   
For the fine-tuning setting with Spanish question generation, we select data over 4 skills (Spanish question generation, Spanish question answering, English question generation, English question answering) for a total of 513210 samples and 212 tasks to select from. We evaluated on 100 samples per task.   
For the fine-tuning setting with stance detection, we select data over 2 skills (stance detection, text matching) for a total of 50990 samples and 19 tasks to select from. We evaluated on 50 samples per task.   
For the out-of-domain setting, we select data over all 59 task categories for a total of 2,417,867 samples and 753 tasks to select from. The test split consisted of 12 task categories and 119 tasks, and we evaluated on min(400, task size) samples per task.   
- RedPajama [57]: the RedPajama dataset is a 1-trillion token dataset that aims to reproduce the LLaMA [59] training dataset. We select over the 7 data sources and evaluate using the LM evaluation harness [14].

![](images/2a34094fbf2159b515bc7700c102e88978896307e42acf71af738277256e7977.jpg)  
Figure 8: Alpaca heatmap where i, jth entry is $\max(0, \delta_{j}^{i})$ (the change in loss on $s_{j}$ after training on $s_{i}$ for 150 steps). Diagonal entries are set to 0 for clearer visualization.

# C.2 Graph Learning Details

We describe how the skills graph was learned on each dataset.

- Alpaca (Figure 8): we use Algorithm 3 and train for $K = 150$ steps per skill. Each edge $i \to j$ has a weight of $\delta_j^i$ , the difference in loss on skill $j$ before and after training on $i$ . Next, we compare the average validation loss of skill-stratified sampling versus random sampling when we train for $K = 1000$ steps. We find that skill-stratified sampling only does 0.007 better than random sampling, confirming that Alpaca's dense skills graph suggests that random sampling is the best we can do.   
- Pile of Law (Figure 9): we use Algorithm 3 and train for $K = 150$ steps. Each edge $i \to j$ has a weight of $\delta_j^i$ , the difference in loss on skill $j$ before and after training on $i$ .   
- LEGO (Figure 10): we use both Algorithm 2 and Algorithm 3 and train for $K = 6000$ steps each. Each edge $i \to j$ has a weight of 0.5 if the amount of data associated with skill $j$ that is needed to reach 0.01 validation loss is less when training on $(i, j)$ than on $j$ (edges are set to 0 if 0.01 validation loss is not reached, even if loss is decreasing). Each edge $i \to j$ is also set to 0.5 if training on $i$ decreases loss directly on $j$ . We set each diagonal entry of $A$ to be 1.   
- Addition (Figure 11): we use Algorithm 2 and train for $K = 6000$ steps. Each edge $i \to j$ has a weight of 0.5 if the amount of data associated with skill $j$ that is needed to reach 0.01 validation loss is less when training on $(i, j)$ than on $j$ (edges are set to 0 if 0.01 validation loss is not reached, even if loss is decreasing). We set each diagonal entry of $A$ to be 1.   
- Natural Instructions (Figure 12, 13, 14): we use Algorithm 3. For the pre-training setting, we train for $K = 600$ steps and assign each edge $i \to j$ a weight $\delta_j^i$ equal to the change in loss on $j$ in the first 100 steps for all $i, j \in [k]$ , including diagonal entries. For the fine-tuning setting, we train for $K = 600$ steps and assign each edge $i \to j$ a weight $\delta_j^i$ equal to the change in loss before and after training. For the out-of-domain setting, we train for $K = 600$ steps and assign each edge $i \to j$ a weight $\delta_j^i$ equal to the change in loss before and after training in the first 100 steps.   
- RedPajama (Figure 15): we use Algorithm 3 and train for 1 billion tokens per data source. We assign each edge $i \to j$ a weight $\delta_j^i$ equal to the change in perplexity on the validation datalsoa before and after training.

# C.3 Training Details

We describe the parameters used for SKILL-IT.

# SKILL-IT pre-training

![](images/9c1a08a40eca8e8992753069d994711c5ced90e22e2430033cee3fcb0bfef923.jpg)  
Figure 9: Pile of Law heatmap where i, jth entry is $\max(0, \delta_{j}^{i})$ (the change in loss on $s_{j}$ after training on $s_{i}$ for 150 steps). Diagonal entries are set to 0 for clearer visualization.

![](images/17b57984da0166c85dbdb9ad64d26bbfee05e5f0f1de3b8fd3a2302127315f11.jpg)

<details>
<summary>heatmap</summary>

| Row | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | 0.2 | 0.8 | 0.6 | 0.4 | 0.2 |
| 2 | 0.6 | 0.4 | 0.8 | 0.2 | 0.0 |
| 3 | 0.0 | 0.0 | 0.6 | 0.4 | 0.2 |
| 4 | 0.0 | 0.0 | 0.0 | 0.6 | 0.2 |
| 5 | 0.0 | 0.0 | 0.0 | 0.4 | 0.2 |
</details>

Figure 10: LEGO heatmap with k = 5 where i, jth entry is set to 0.5 if the number of steps needed to reach 0.01 loss on skill j when training on a balanced mixture of skills i and j is less than when training on skill j only.

![](images/df98670d3a407e46aeed08e5331fa38787222eba31e9f51473ed1419b2c4b214.jpg)

<details>
<summary>heatmap</summary>

|   | 1 | 2 | 3 |
|---|---|---|---|
| 1 | 0.5 | 0.9 | 0.0 |
| 2 | 0.6 | 0.7 | 0.0 |
| 3 | 0.8 | 0.4 | 0.1 |
</details>

Figure 11: Addition heatmap with k = 3 where i, jth entry is set to 0.5 if the number of steps needed to reach 0.01 loss on skill j when training on a balanced mixture of skills i and j is less than when training on skill j only.

![](images/2b043a8cb397d087c015b3c16e07d833f52545abece948f691ba77a959a5bb8c.jpg)  
Figure 12: Natural Instructions heatmap where i, jth entry is $\max(0, \delta_{j}^{i})$ (the change in loss on $s_{j}$ after training on $s_{i}$ for 100 steps). Diagonal entries are set to 0 for clearer visualization.

![](images/679c218d52891cbb5f54c874ed61fb925100e88b3feb26fb3107f8471efc5a81.jpg)

<details>
<summary>heatmap</summary>

Spanish question generation
| | English QA | Spanish QA | English QG | Spanish QG |
|---|---|---|---|---|
| English QA | 0.3 | 0.25 | 0.25 | 0.35 |
| Spanish QA | 0.3 | 0.45 | 0.25 | 0.35 |
| English QG | 0.25 | 0.25 | 0.35 | 0.25 |
| Spanish QG | 0.25 | 0.35 | 0.25 | 0.35 |
</details>

![](images/5dbc7b6cf891f6125e22ba23ba6a4f4862e58291fe8bb7648d62b24a446458cd.jpg)

<details>
<summary>heatmap</summary>

| Stance Detection | Text Matching |
| ---------------- | ------------- |
| 1.0              | 1.0           |
| 0.5              | 0.5           |
| 0.0              | 0.0           |
</details>

Figure 13: Spanish question generation and stance detection heatmaps where i, jth entry is $\max(0, \delta_{j}^{i})$ (the change in loss on $s_{j}$ after training on $s_{i}$ for 100 steps).

![](images/481686c01d1fa8fc41bb8aa2e0e138e7b4f50d8b81b84f7ef41a05627fd8739d.jpg)  
Figure 14: Natural Instructions heatmap for out-of-domain setting where rows are for the training skills and columns are for the evaluation skills. The i, jth entry is $\max(0, \delta_{j}^{i})$ (the change in loss on $s_{j}$ after training on $s_{i}$ for 100 steps).

![](images/f33540fb7168f96e15625fbabbb7607d1f5fa439b95e40ed4ecba2730f835f4e.jpg)

<details>
<summary>heatmap</summary>

|        | arc_challenge | arc_easy | boolq | copa | hellaswag | lambda_openai | piqa | winogrande |
| ------ | ------------- | -------- | ----- | ---- | --------- | -------------- | ---- | ---------- |
| arxiv  | 0.0           | 0.0      | 0.1   | 0.0  | 0.0       | 0.0            | 0.0  | 0.0        |
| books  | 0.0           | 0.0      | 0.1   | 0.0  | 0.0       | 0.0            | 0.0  | 0.0        |
| c4     | 0.0           | 0.0      | 0.1   | 0.0  | 0.0       | 0.0            | 0.0  | 0.0        |
| common_crawl | 0.0    | 0.0      | 0.1   | 0.0  | 0.0       | 0.0            | 0.0  | 0.0        |
| github | 0.0           | 0.0      | 0.1   | 0.0  | 0.0       | 0.0            | 0.0  | 0.0        |
| stackexchange | 0.0    | 0.0      | 0.1   | 0.1  | 0.1       | 0.1            | 0.1  | 0.1        |
| wikipedia   | 0.0           | 0.0      | 0.1   | 0.1  | 0.1       | 0.1            | 0.1  | 0.1        |
</details>

Figure 15: RedPajama heatmap for out-of-domain setting where rows are for the training skills and columns are for the evaluation skills. The i,jth entry is $\max(0, \delta_{j}^{i})$ (the change in perplexity on $s_{j}$ after training on $s_{i}$ for 1B tokens).

- LEGO: $\eta = 0.5, T = 6, w = 3$ . We train for 6000 steps.   
- Addition: $\eta = 0.1, T = 5, w = 3$ . We train for 6000 steps.   
- Natural Instructions (pre-training): $\eta = 0.2$ , $T = 1$ . We train for 5000 steps.

For the LEGO random baseline, when we selected points at random, we used an imbalanced training dataset with proportions 1:1:1:3:5. For the addition random baseline, we used an imbalanced dataset with randomly selected proportions: 13:14:18. For the curriculum learning baselines, the pacing function, $g(i)$ , denotes the size of the subset of the highest scoring samples that we uniformly select from in the $i$ th epoch. We define our pacing function as $g(i) = \frac{iH}{M}$ , where $H$ is the number of steps and $M$ is 5 epochs for LEGO and NI, and 3 for addition.

# SKILL-IT fine-tuning

- LEGO: $\eta = 0.5, T = 10, w = 3$ . We train for 6000 steps.   
- Addition: $\eta = 0.1, T = 5, w = 3$ . We train for 6000 steps.   
- Natural Instructions (Spanish QG): $\eta = 0.8$ , $T = 6$ , $w = 3$ . We train for 600 steps.   
- Natural Instructions (stance detection): $\eta = 0.2$ , $T = 6$ , $w = 3$ . We train for 600 steps.

# SKILL-IT out-of-domain

- Natural Instructions: $\eta = 0.2$ , $T = 10$ , $w = 3$ . We train for 5000 steps.   
- RedPajama: $\eta = 100, T = 1$ . We train for 3 billion tokens.

All results are computed over 5 random seeds.

Batch sizes of 32 and 64 were used for the LEGO and addition synthetic on the 125M and 1.3B parameter model, respectively. Batch sizes of 4 and 16 were used for the Natural Instructions experiments on the 125M and 1.3B parameter model.

For the out-of-domain Natural Instructions experiment and Alpaca graph learning experiments, a learning rate of 5e-6 with linear scheduler and 50 warmup steps was used. For the Natural Instructions continual pre-training experiment on the 1.3B parameter model, a learning rate of 1e-6 was used. All other experiments used a learning rate of 5e-5. All experiments used AdamW with betas = 0.9, 0.999, eps = 1e-8, and weight decay = 0.01. A context window of 512 was used for all experiments except LEGO and addition, which used a window of 128.

Experiments with the Addition dataset were run using an Nvidia RTX A6000. Other experiments using the GPT-Neo 125M parameter model were run on an Nvidia Tesla P100. Experiments using the GPT-Neo 1.3B parameter model were run on an Nvidia Tesla A100.

![](images/7089d055c3e43629101381253ea53194cd2fa73750e4a9dc9c8e7dfd1c99e02b.jpg)

<details>
<summary>line</summary>

| Steps | Trained on skill 4 | Trained on skills 2, 4 |
| ----- | ------------------ | ---------------------- |
| 0     | 0.7                | 0.7                    |
| 1000  | 0.7                | 0.9                    |
| 2000  | 0.7                | 0.7                    |
| 3000  | 0.7                | 0.65                   |
| 4000  | 0.7                | 0.6                    |
| 5000  | 0.7                | 0.5                    |
| 6000  | 0.7                | 0.4                    |
</details>

![](images/dcd9d4336405360476802e5111c34ceaa67386f747c2e54a3a213a3709298339.jpg)

<details>
<summary>line</summary>

| Steps | Trained on skill 4 | Trained on skills 3, 4 |
| ----- | ------------------ | ---------------------- |
| 0     | 0.72               | 0.72                   |
| 1000  | 0.68               | 0.71                   |
| 2000  | 0.70               | 0.71                   |
| 3000  | 0.69               | 0.70                   |
| 4000  | 0.69               | 0.70                   |
| 5000  | 0.69               | 0.69                   |
| 6000  | 0.69               | 0.69                   |
</details>

Figure 16: Performance on LEGO skill 4 when training on skill 4, skills 2 and 4, and skills 3 and 4. Even though skill 3 and skill 4 share an edge in the LEGO synthetic's underlying reasoning chain (i.e. a model predicting correct for the fourth variable is one extra step beyond predicting correct for the third variable), we find that training on skills 2 and 4 helps improve performance on skill 4 more.   
![](images/52601de3c7822616b19da81072e6218ea2169433e4a1d58c95d512208c919894.jpg)

<details>
<summary>line</summary>

| Steps | Trained on skill 3 | Trained on skill 2 |
| ----- | ------------------ | ------------------ |
| 0     | 0.7                | 0.7                |
| 500   | 0.05               | 0.05               |
| 1000  | 0.1                | 0.05               |
| 1500  | 0.0                | 0.0                |
| 2000  | 0.0                | 0.0                |
| 2500  | 0.0                | 0.0                |
| 3000  | 0.0                | 0.0                |
</details>

![](images/15fbf73c6b1f43878f38bcb7c7dff391b7f974d80b568f97297d1fa33611fc4d.jpg)

<details>
<summary>line</summary>

| Steps | Trained on skill 3 | Trained on skill 2 |
| ----- | ------------------ | ------------------ |
| 0     | 0.7                | 0.7                |
| 500   | 0.1                | 0.0                |
| 1000  | 0.15               | 0.0                |
| 1500  | 0.05               | 0.0                |
| 2000  | 0.0                | 0.0                |
| 2500  | 0.0                | 0.0                |
| 3000  | 0.0                | 0.0                |
</details>

Figure 17: Performance on LEGO skill 2 and 3 when training on skills 2 and 3. The reasoning pattern is a tree rather than a chain over k = 4 variables. Skills 2 and 3 are at the same “depth” in the graph and both depend on skill 1, so there is positive influence between the skills despite there being no edge between 2 and 3 in the LEGO reasoning graph.

# D Additional Experimental Results

# D.1 Additional examples of LEGO ordered skill sets

For the LEGO synthetic, it may appear obvious that the skills graph is equivalent to the reasoning chain over the variables. However, in Figure 16 we see that this is not the case. Training on skills 2 and 4 together results in lower loss on skill 4 than when trained on skill 4 alone. However, training on skills 3 and 4 together results in roughly the same loss on skill 4 as when training on skill 4 alone, even though skill 3 and skill 4 share an edge in the LEGO synthetic's underlying reasoning chain. This suggests that our intuition for how skills influence each other does not always match how the model learns skills.

Next, we consider a slightly more complex reasoning pattern on the LEGO synthetic. Instead of a chain, we construct a tree, where two variables in the LEGO synthetic are both defined in terms of the same parent variable. For example,

Input: c = val 1, y = not w, v = val c, w = not c. Output: y = 1.

In this example, k = 4 and both v and w are written in terms of c, and the reasoning graph has edges $1 \rightarrow 2$ , $1 \rightarrow 3$ , $2 \rightarrow 4$ . In this case, we see that training on skill 2 or skill 3 both improve losses on skills 2 and 3 (Figure 17). However, unlike the previous figures, training on skills 2 and 4 or skills 3 and 4 do not significantly help reduce loss on skill 4 (Figure 18). Again, these measurements demonstrate that the reasoning graph does not necessarily equal the skills graph.

# D.2 Unsupervised skill recovery

We explore several clustering techniques for recovering the skills in the LEGO synthetic on the validation dataset. Our results are shown in Table 3.

We first cluster based on the pre-trained model embeddings of the last token and the average token. We also report

Model performance on LEGO skill 4 (tree)   
![](images/cbaafeb576343392077db762cdbc846fdfca01857da1dc2d36f09eaff22c2476.jpg)

<details>
<summary>line</summary>

| Steps | Trained on skill 4 | Trained on skills 2, 4 |
| ----- | ------------------ | ---------------------- |
| 0     | 0.8                | 0.8                    |
| 500   | 0.4                | 0.6                    |
| 1000  | 0.2                | 0.4                    |
| 1500  | 0.1                | 0.3                    |
| 2000  | 0.05               | 0.2                    |
| 2500  | 0.02               | 0.1                    |
| 3000  | 0.0                | 0.0                    |
</details>

Model performance on LEGO skill 4 (tree)   
![](images/343ff00d7f99470dff743e658451175ef26c9404405299d0da76113b879c4277.jpg)

<details>
<summary>line</summary>

| Steps | Trained on skill 4 | Trained on skills 3, 4 |
| ----- | ------------------ | ---------------------- |
| 0     | 0.75               | 0.75                   |
| 500   | 0.5                | 0.6                    |
| 1000  | 0.2                | 0.4                    |
| 1500  | 0.1                | 0.2                    |
| 2000  | 0.05               | 0.1                    |
| 2500  | 0.02               | 0.05                   |
| 3000  | 0.01               | 0.02                   |
</details>

Figure 18: Performance on LEGO skill 4 when training on skills 2, 4 and skills 3, 4. We find that in both cases, the benefit from training on additional skills is minor. For instance, training on 2 and 4 reaches 0.01 loss in 2700 steps, while training on 4 only reaches it in 2100 steps. 

<table><tr><td>Cluster method</td><td>Accuracy</td></tr><tr><td>Pretrained embedding of last token</td><td>24.8 ± 0.5</td></tr><tr><td>Pretrained embedding of average token</td><td>25.2 ± 1.1</td></tr><tr><td>Trained model embedding of last token</td><td>38.4 ± 0.8</td></tr><tr><td>Sentence-BERT embedding</td><td>23.9 ± 0.7</td></tr><tr><td>Losses over multiple runs</td><td>61.0 ± 1.6</td></tr></table>

Table 3: Clustering-based skill recovery methods on the LEGO dataset. The validation dataset we cluster consists of 500 points with $k = 5$ , and results are reported over 10 runs of k-means.

accuracies of clustering based on the trained model embedding's last token, where we train the model using random sampling for 6000 steps, and clustering based on Sentence-BERT embeddings. Among these four methods, using the trained model embeddings has the highest accuracy of 38.4 points.

Next, we cluster points based on losses. In particular, we do 10 runs, each for 6000 steps and with a randomly sampled mixture of skills. For each run, we evaluate the model on the validation dataset at 120 checkpoints. Then, each sample in the validation dataset has 1200 losses associated with it, comprising a feature vector for that sample. We perform k-means clustering on these features, which has an accuracy of 61.0 points, significantly higher than the second best accuracy of 38.4.

# D.3 Full results for Section 4

# D.3.1 Per-skill performance

In this section, we provide tables containing the per skill break-down of our results from Section 4.

Continual Pre-training In the continual pre-training setting, we report two additional baselines that combine curriculum learning with skills. Curriculum learning has been proposed for multitask learning $[60]$ , in which groups of data are ranked by their average score and then trained in order of this ranking (with mixing of previously seen groups to avoid forgetting). We construct two baselines, Skill-curriculum and Skill-anticurriculum, using Algorithm 1 from $[60]$ . In contrast to the random baseline which has imbalanced skills, this approach has knowledge of skills and thus uses a skill-stratified training dataset to sample from. We set the fraction of the previous group to be frac = 0.4, as we found that setting frac = 0.0 resulted in forgetting.

We report loss per skill for the LEGO synthetic in Table 4, which corresponds to the results in Figure 4. We report accuracy per skill in Table 5 and Figure 19. We report the loss per skill for the Addition synthetic in Table 6, which also correspond to to the results in Figure 4. Finally, we report validation loss per task category for the Natural Instructions continual pre-training experiment in Table 7, where we find that SKILL-IT outperforms random sampling by 3.2% on average across skills.

<table><tr><td></td><td>Skill 1</td><td>Skill 2</td><td>Skill 3</td><td>Skill 4</td><td>Skill 5</td><td>Average</td></tr><tr><td>Random</td><td> $0_{\pm 0.000}$ </td><td> $0.675_{\pm 0.041}$ </td><td> $0.688_{\pm 0.008}$ </td><td> $0.673_{\pm 0.049}$ </td><td> $0.667_{\pm 0.056}$ </td><td> $0.541_{\pm 0.031}$ </td></tr><tr><td>Curriculum</td><td> $0_{\pm 0.000}$ </td><td> $0.645_{\pm 0.052}$ </td><td> $0.686_{\pm 0.018}$ </td><td> $0.674_{\pm 0.042}$ </td><td> $0.671_{\pm 0.0459}$ </td><td> $0.535_{\pm 0.029}$ </td></tr><tr><td>Anticurriculum</td><td> $0_{\pm 0.000}$ </td><td> $0.690_{\pm 0.003}$ </td><td> $0.695_{\pm 0.004}$ </td><td> $0.693_{\pm 0.003}$ </td><td> $0.689_{\pm 0.004}$ </td><td> $0.554_{\pm 0.001}$ </td></tr><tr><td>Skill-stratified</td><td> $0_{\pm 0.000}$ </td><td> $0.045_{\pm 0.036}$ </td><td> $0.056_{\pm 0.029}$ </td><td> $0.079_{\pm 0.044}$ </td><td> $0.050_{\pm 0.025}$ </td><td> $0.046_{\pm 0.022}$ </td></tr><tr><td>Skill-curriculum</td><td> $0_{\pm 0.000}$ </td><td> $0.484_{\pm 0.200}$ </td><td> $0.698_{\pm 0.027}$ </td><td> $0.697_{\pm 0.010}$ </td><td> $0.689_{\pm 0.007}$ </td><td> $0.514_{\pm 0.040}$ </td></tr><tr><td>Skill-anticurriculum</td><td> $0.001_{\pm 0.001}$ </td><td> $0.174_{\pm 0.118}$ </td><td> $0.245_{\pm 0.091}$ </td><td> $0.443_{\pm 0.125}$ </td><td> $0.566_{\pm 0.118}$ </td><td> $0.286_{\pm 0.060}$ </td></tr><tr><td>SKILL-IT</td><td> $0_{\pm 0.000}$ </td><td> $0.002_{\pm 0.002}$ </td><td> $0.024_{\pm 0.031}$ </td><td> $0.013_{\pm 0.010}$ </td><td> $0.022_{\pm 0.021}$ </td><td> $0.012_{\pm 0.008}$ </td></tr></table>

Table 4: Results on validation loss per skill for LEGO pre-training experiment, averaged over 5 random seeds.

<table><tr><td></td><td>Skill 1</td><td>Skill 2</td><td>Skill 3</td><td>Skill 4</td><td>Skill 5</td><td>Average</td></tr><tr><td>Random</td><td> $100.0 \pm 0.0$ </td><td> $54.2 \pm 5.9$ </td><td> $58.0 \pm 3.1$ </td><td> $48.0 \pm 6.3$ </td><td> $54.4 \pm 7.3$ </td><td> $62.9 \pm 3.5$ </td></tr><tr><td>Curriculum</td><td> $100.0 \pm 0.0$ </td><td> $60.0 \pm 10.6$ </td><td> $55.2 \pm 5.8$ </td><td> $51.2 \pm 6.3$ </td><td> $51.8 \pm 6.1$ </td><td> $63.6 \pm 3.6$ </td></tr><tr><td>Anticurriculum</td><td> $100.0 \pm 0.0$ </td><td> $53.4 \pm 2.3$ </td><td> $49.0 \pm 4.8$ </td><td> $48.2 \pm 6.4$ </td><td> $56.0 \pm 5.7$ </td><td> $61.3 \pm 2.2$ </td></tr><tr><td>Skill-stratified</td><td> $100.0 \pm 0.0$ </td><td> $98.2 \pm 1.8$ </td><td> $98.2 \pm 1.3$ </td><td> $97.8 \pm 1.6$ </td><td> $98.2 \pm 1.3$ </td><td> $98.5 \pm 0.9$ </td></tr><tr><td>Skill-curriculum</td><td> $100.0 \pm 0.0$ </td><td> $75.2 \pm 30.1$ </td><td> $52.2 \pm 3.7$ </td><td> $51.0 \pm 4.6$ </td><td> $54.4 \pm 3.1$ </td><td> $66.6 \pm 7.7$ </td></tr><tr><td>Skill-anticurriculum</td><td> $100.0 \pm 0.0$ </td><td> $90.2 \pm 8.1$ </td><td> $88.2 \pm 8.3$ </td><td> $73.2 \pm 12.2$ </td><td> $62.4 \pm 9.4$ </td><td> $82.8 \pm 4.9$ </td></tr><tr><td>SKILL-IT</td><td> $100.0 \pm 0.0$ </td><td> $99.2 \pm 0.8$ </td><td> $99.0 \pm 1.0$ </td><td> $99.4 \pm 0.5$ </td><td> $99.6 \pm 0.5$ </td><td> $99.4 \pm 0.2$ </td></tr></table>

Table 5: Results on accuracy per skill (binary classification) for LEGO pre-training experiment, averaged over 5 random seeds.

![](images/eb9186c2ba67b9c96512581ab74016e09bbfe6ae58d31cc0906e972e8b78a247.jpg)

<details>
<summary>line</summary>

| Step | Accuracy |
| ---- | -------- |
| 0    | 0.4      |
| 500  | 0.7      |
| 1000 | 0.9      |
| 1500 | 1.0      |
| 2000 | 0.8      |
| 2500 | 0.6      |
| 3000 | 0.7      |
| 3500 | 0.8      |
| 4000 | 0.6      |
| 4500 | 0.4      |
| 5000 | 1.0      |
| 5500 | 1.0      |
| 6000 | 1.0      |
</details>

![](images/d5e809c8c785de85d8ffd87d04b6e0392571553e27be54cebac0c4954f09a8af.jpg)

<details>
<summary>line</summary>

| Step | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 | Series 6 | Series 7 |
|------|----------|----------|----------|----------|----------|----------|----------|
| 0    | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 1000 | 0.8      | 0.6      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 2000 | 0.9      | 0.7      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 3000 | 0.95     | 0.8      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 4000 | 0.98     | 0.9      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 5000 | 0.99     | 0.92     | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 6000 | 1.0      | 0.95     | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
</details>

![](images/c92b422c288cf5d006823e92b976c5ded38b81f478792daceeb62e04d1992fec.jpg)

<details>
<summary>line</summary>

| Step | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 | Series 6 | Series 7 |
|------|----------|----------|----------|----------|----------|----------|----------|
| 0    | 0.4      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 1000 | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 2000 | 0.6      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 3000 | 0.7      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 4000 | 0.8      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 5000 | 0.9      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      | 0.5      |
| 6000 | 1.0      | 0.9      | 0.9      | 0.9      | 0.9      | 0.9      | 0.9      |
</details>

![](images/3917cb5e845acddd9553e123fc57cb2af778271e1957cf367e9bdb0cb7a22dae.jpg)

<details>
<summary>line</summary>

| Steps | Accuracy (Line 1) | Accuracy (Line 2) | Accuracy (Line 3) | Accuracy (Line 4) | Accuracy (Line 5) | Accuracy (Line 6) | Accuracy (Line 7) |
|-------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| 0     | 0.5               | 0.5               | 0.5               | 0.5               | 0.5               | 0.5               | 0.5               |
| 2000  | 0.6               | 0.55              | 0.52              | 0.51              | 0.5               | 0.48              | 0.49              |
| 4000  | 0.8               | 0.7               | 0.6               | 0.55              | 0.52              | 0.49              | 0.51              |
| 6000  | 0.95              | 0.9               | 0.75              | 0.6               | 0.55              | 0.5               | 0.52              |
</details>

![](images/0f1519135a2d6977231594e8cf474d4f54d0a1369cc5ed6104212eb403b9b5d8.jpg)

<details>
<summary>line</summary>

| Steps | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 | Line 6 | Line 7 | Line 8 | Line 9 |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| 0     | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    |
| 1000  | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    |
| 2000  | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    | 0.5    |
| 3000  | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    |
| 4000  | 0.8    | 0.8    | 0.8    | 0.8    | 0.8    | 0.8    | 0.8    | 0.8    | 0.8    |
| 5000  | 0.9    | 0.9    | 0.9    | 0.9    | 0.9    | 0.9    | 0.9    | 0.9    | 0.9    |
| 6000  | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    |
</details>

![](images/f9a45d4cda42be7d529a76ce04d2e79125013e0b6bba4abdbad39834499c2a2a.jpg)

<details>
<summary>line</summary>

| Steps | Random | Curriculum | Anticurriculum | Skill-stratified | Skill-curriculum | Skill-anticurriculum | Skill-It |
| ----- | ------ | ---------- | -------------- | ---------------- | ---------------- | -------------------- | -------- |
| 0     | 0.5    | 0.5        | 0.5            | 0.5              | 0.5              | 0.5                  | 0.2      |
| 2000  | 0.6    | 0.6        | 0.6            | 0.7              | 0.6              | 0.6                  | 0.8      |
| 4000  | 0.6    | 0.6        | 0.6            | 0.9              | 0.6              | 0.8                  | 1.0      |
| 6000  | 0.6    | 0.6        | 0.6            | 1.0              | 0.6              | 0.8                  | 1.0      |
</details>

Figure 19: Accuracy of SKILL-IT on each skill (binary classification) on the LEGO synthetic in the continual pre-training setting. SKILL-IT attains higher accuracy more quickly than baselines that both do and do not utilize the notion of skills. 

<table><tr><td></td><td>Skill 1</td><td>Skill 2</td><td>Skill 3</td><td>Average</td></tr><tr><td>Random</td><td> $0.008 \pm 0.007$ </td><td> $0.020 \pm 0.019$ </td><td> $0.005 \pm 0.005$ </td><td> $0.011 \pm 0.014$ </td></tr><tr><td>Curriculum</td><td> $0.009 \pm 0.011$ </td><td> $0.010 \pm 0.008$ </td><td> $0.008 \pm 0.010$ </td><td> $0.009 \pm 0.010$ </td></tr><tr><td>Anticurriculum</td><td> $0.007 \pm 0.010$ </td><td> $0.012 \pm 0.013$ </td><td> $0.008 \pm 0.017$ </td><td> $0.009 \pm 0.014$ </td></tr><tr><td>Skill-stratified</td><td> $0.012 \pm 0.011$ </td><td> $0.015 \pm 0.015$ </td><td> $0.010 \pm 0.020$ </td><td> $0.012 \pm 0.016$ </td></tr><tr><td>Skill-curriculum</td><td> $0.016 \pm 0.013$ </td><td> $0.019 \pm 0.013$ </td><td> $0.010 \pm 0.003$ </td><td> $0.015 \pm 0.010$ </td></tr><tr><td>Skill-anticurriculum</td><td> $0.005 \pm 0.008$ </td><td> $0.037 \pm 0.028$ </td><td> $1.141 \pm 1.126$ </td><td> $0.395 \pm 0.371$ </td></tr><tr><td>SKILL-IT</td><td> $0.004 \pm 0.003$ </td><td> $0.009 \pm 0.007$ </td><td> $0.013 \pm 0.017$ </td><td> $0.009 \pm 0.011$ </td></tr></table>

Table 6: Results on validation loss per skill for Addition pre-training experiment, averaged over 5 random seeds.

<table><tr><td>Skill</td><td>Random</td><td>Curriculum</td><td>Anticurriculum</td><td>Skill-stratified</td><td>Skill-curriculum</td><td>Skill-anticurriculum</td><td>SKILL-IT</td></tr><tr><td>Answer Verification</td><td> $2.297_{\pm 0.058}$ </td><td> $2.368_{\pm 0.055}$ </td><td> $2.391_{\pm 0.061}$ </td><td> $2.180_{\pm 0.059}$ </td><td> $2.249_{\pm 0.116}$ </td><td> $2.325_{\pm 0.085}$ </td><td> $2.158_{\pm 0.059}$ </td></tr><tr><td>Code to Text</td><td> $0.246_{\pm 0.021}$ </td><td> $0.203_{\pm 0.019}$ </td><td> $1.099_{\pm 0.115}$ </td><td> $0.178_{\pm 0.016}$ </td><td> $0.126_{\pm 0.009}$ </td><td> $1.232_{\pm 0.070}$ </td><td> $0.223_{\pm 0.017}$ </td></tr><tr><td>Discourse Connective Identification</td><td> $2.927_{\pm 0.069}$ </td><td> $3.084_{\pm 0.067}$ </td><td> $2.932_{\pm 0.058}$ </td><td> $2.805_{\pm 0.071}$ </td><td> $2.891_{\pm 0.001}$ </td><td> $2.925_{\pm 0.011}$ </td><td> $2.784_{\pm 0.068}$ </td></tr><tr><td>Entity Generation</td><td> $2.033_{\pm 0.421}$ </td><td> $2.012_{\pm 0.437}$ </td><td> $2.363_{\pm 0.234}$ </td><td> $1.803_{\pm 0.384}$ </td><td> $1.853_{\pm 0.483}$ </td><td> $2.068_{\pm 0.719}$ </td><td> $1.863_{\pm 0.418}$ </td></tr><tr><td>Entity Relation Classification</td><td> $1.020_{\pm 0.147}$ </td><td> $1.014_{\pm 0.140}$ </td><td> $1.533_{\pm 0.138}$ </td><td> $0.859_{\pm 0.131}$ </td><td> $0.825_{\pm 0.022}$ </td><td> $0.959_{\pm 0.009}$ </td><td> $0.908_{\pm 0.146}$ </td></tr><tr><td>Information Extraction</td><td> $2.154_{\pm 0.040}$ </td><td> $2.247_{\pm 0.037}$ </td><td> $2.352_{\pm 0.042}$ </td><td> $2.140_{\pm 0.037}$ </td><td> $2.286_{\pm 0.022}$ </td><td> $2.338_{\pm 0.025}$ </td><td> $2.073_{\pm 0.042}$ </td></tr><tr><td>Irony Detection</td><td> $3.024_{\pm 0.154}$ </td><td> $3.798_{\pm 0.095}$ </td><td> $2.942_{\pm 0.158}$ </td><td> $2.680_{\pm 0.146}$ </td><td> $3.889_{\pm 0.066}$ </td><td> $2.099_{\pm 0.152}$ </td><td> $2.797_{\pm 0.155}$ </td></tr><tr><td>Preposition Prediction</td><td> $0.979_{\pm 0.124}$ </td><td> $0.887_{\pm 0.147}$ </td><td> $1.488_{\pm 0.213}$ </td><td> $0.845_{\pm 0.152}$ </td><td> $0.941_{\pm 0.019}$ </td><td> $1.044_{\pm 0.029}$ </td><td> $0.876_{\pm 0.173}$ </td></tr><tr><td>Punctuation Error Detection</td><td> $2.950_{\pm 0.065}$ </td><td> $3.120_{\pm 0.052}$ </td><td> $2.961_{\pm 0.064}$ </td><td> $3.264_{\pm 0.061}$ </td><td> $3.019_{\pm 0.010}$ </td><td> $3.360_{\pm 0.013}$ </td><td> $3.216_{\pm 0.055}$ </td></tr><tr><td>Question Answering</td><td> $2.277_{\pm 0.005}$ </td><td> $2.367_{\pm 0.006}$ </td><td> $2.398_{\pm 0.006}$ </td><td> $2.542_{\pm 0.004}$ </td><td> $2.689_{\pm 0.001}$ </td><td> $2.707_{\pm 0.016}$ </td><td> $2.448_{\pm 0.008}$ </td></tr><tr><td>Question Generation</td><td> $2.617_{\pm 0.005}$ </td><td> $2.777_{\pm 0.015}$ </td><td> $2.695_{\pm 0.008}$ </td><td> $2.783_{\pm 0.021}$ </td><td> $3.062_{\pm 0.006}$ </td><td> $2.876_{\pm 0.032}$ </td><td> $2.666_{\pm 0.012}$ </td></tr><tr><td>Question Understanding</td><td> $1.965_{\pm 0.051}$ </td><td> $2.199_{\pm 0.059}$ </td><td> $2.060_{\pm 0.033}$ </td><td> $1.958_{\pm 0.051}$ </td><td> $2.385_{\pm 0.022}$ </td><td> $2.100_{\pm 0.054}$ </td><td> $1.895_{\pm 0.043}$ </td></tr><tr><td>Sentence Expansion</td><td> $2.501_{\pm 0.095}$ </td><td> $2.598_{\pm 0.097}$ </td><td> $2.583_{\pm 0.074}$ </td><td> $2.225_{\pm 0.095}$ </td><td> $2.311_{\pm 0.076}$ </td><td> $2.408_{\pm 0.074}$ </td><td> $2.236_{\pm 0.083}$ </td></tr><tr><td>Sentiment Analysis</td><td> $3.203_{\pm 0.012}$ </td><td> $3.415_{\pm 0.016}$ </td><td> $3.209_{\pm 0.010}$ </td><td> $3.278_{\pm 0.014}$ </td><td> $3.607_{\pm 0.012}$ </td><td> $3.308_{\pm 0.015}$ </td><td> $3.213_{\pm 0.012}$ </td></tr><tr><td>Stance Detection</td><td> $1.810_{\pm 0.100}$ </td><td> $1.775_{\pm 0.120}$ </td><td> $2.231_{\pm 0.128}$ </td><td> $1.385_{\pm 0.070}$ </td><td> $1.361_{\pm 0.114}$ </td><td> $1.823_{\pm 0.189}$ </td><td> $1.556_{\pm 0.125}$ </td></tr><tr><td>Summarization</td><td> $2.961_{\pm 0.015}$ </td><td> $3.149_{\pm 0.023}$ </td><td> $3.041_{\pm 0.014}$ </td><td> $2.960_{\pm 0.019}$ </td><td> $3.323_{\pm 0.028}$ </td><td> $3.021_{\pm 0.013}$ </td><td> $2.907_{\pm 0.012}$ </td></tr><tr><td>Text Categorization</td><td> $2.488_{\pm 0.023}$ </td><td> $2.692_{\pm 0.029}$ </td><td> $2.553_{\pm 0.006}$ </td><td> $2.570_{\pm 0.015}$ </td><td> $3.001_{\pm 0.007}$ </td><td> $2.635_{\pm 0.014}$ </td><td> $2.448_{\pm 0.017}$ </td></tr><tr><td>Text Matching</td><td> $2.177_{\pm 0.059}$ </td><td> $2.232_{\pm 0.055}$ </td><td> $2.316_{\pm 0.048}$ </td><td> $2.152_{\pm 0.061}$ </td><td> $2.324_{\pm 0.004}$ </td><td> $2.304_{\pm 0.035}$ </td><td> $2.093_{\pm 0.054}$ </td></tr><tr><td>Text Simplification</td><td> $2.155_{\pm 0.023}$ </td><td> $2.193_{\pm 0.039}$ </td><td> $2.325_{\pm 0.033}$ </td><td> $1.926_{\pm 0.026}$ </td><td> $2.037_{\pm 0.005}$ </td><td> $2.156_{\pm 0.011}$ </td><td> $1.952_{\pm 0.026}$ </td></tr><tr><td>Text to Code</td><td> $0.560_{\pm 0.037}$ </td><td> $0.495_{\pm 0.036}$ </td><td> $1.215_{\pm 0.052}$ </td><td> $0.490_{\pm 0.029}$ </td><td> $0.433_{\pm 0.014}$ </td><td> $1.455_{\pm 0.086}$ </td><td> $0.553_{\pm 0.042}$ </td></tr><tr><td>Toxic Language Detection</td><td> $3.106_{\pm 0.027}$ </td><td> $3.496_{\pm 0.017}$ </td><td> $3.058_{\pm 0.029}$ </td><td> $3.199_{\pm 0.024}$ </td><td> $3.758_{\pm 0.025}$ </td><td> $3.155_{\pm 0.050}$ </td><td> $3.129_{\pm 0.020}$ </td></tr><tr><td>Word Semantics</td><td> $2.092_{\pm 0.027}$ </td><td> $2.334_{\pm 0.034}$ </td><td> $2.156_{\pm 0.064}$ </td><td> $1.916_{\pm 0.043}$ </td><td> $1.784_{\pm 0.048}$ </td><td> $2.424_{\pm 0.038}$ </td><td> $1.952_{\pm 0.019}$ </td></tr><tr><td>Wrong Candidate Generation</td><td> $2.438_{\pm 0.021}$ </td><td> $2.606_{\pm 0.039}$ </td><td> $2.519_{\pm 0.027}$ </td><td> $2.506_{\pm 0.026}$ </td><td> $2.849_{\pm 0.029}$ </td><td> $2.574_{\pm 0.018}$ </td><td> $2.432_{\pm 0.025}$ </td></tr><tr><td>Average</td><td> $2.173_{\pm 0.028}$ </td><td> $2.307_{\pm 0.025}$ </td><td> $2.366_{\pm 0.026}$ </td><td> $2.115_{\pm 0.027}$ </td><td> $2.304_{\pm 0.031}$ </td><td> $2.317_{\pm 0.052}$ </td><td> $2.103_{\pm 0.032}$ </td></tr></table>

Table 7: Validation loss per skill for data selection in continual pre-training setting on a subset of the Natural Instructions Dataset.

Out-of-domain In Table 8, we provide a breakdown of validation loss per evaluation skill under random sampling on the training data, skill-stratified sampling over prerequisite skills (e.g., the nonzero rows in Figure 14), and SKILL-IT.

<table><tr><td>Skill</td><td>Random</td><td>Skill-stratified</td><td>SKILL-IT</td></tr><tr><td>Answerability Classification</td><td> $3.048_{\pm 0.003}$ </td><td> $3.076_{\pm 0.002}$ </td><td> $3.043_{\pm 0.003}$ </td></tr><tr><td>Cause Effect Classification</td><td> $2.068_{\pm 0.004}$ </td><td> $2.101_{\pm 0.005}$ </td><td> $2.067_{\pm 0.006}$ </td></tr><tr><td>Coreference Resolution</td><td> $3.101_{\pm 0.003}$ </td><td> $3.142_{\pm 0.004}$ </td><td> $3.099_{\pm 0.004}$ </td></tr><tr><td>Data to Text</td><td> $2.363_{\pm 0.004}$ </td><td> $2.388_{\pm 0.005}$ </td><td> $2.359_{\pm 0.005}$ </td></tr><tr><td>Dialogue Act Recognition</td><td> $2.329_{\pm 0.009}$ </td><td> $2.364_{\pm 0.010}$ </td><td> $2.320_{\pm 0.009}$ </td></tr><tr><td>Grammar Error Correction</td><td> $2.399_{\pm 0.008}$ </td><td> $2.418_{\pm 0.009}$ </td><td> $2.389_{\pm 0.007}$ </td></tr><tr><td>Keyword Tagging</td><td> $2.744_{\pm 0.005}$ </td><td> $2.760_{\pm 0.007}$ </td><td> $2.733_{\pm 0.006}$ </td></tr><tr><td>Overlap Extraction</td><td> $2.749_{\pm 0.011}$ </td><td> $2.763_{\pm 0.012}$ </td><td> $2.733_{\pm 0.010}$ </td></tr><tr><td>Question Rewriting</td><td> $2.591_{\pm 0.009}$ </td><td> $2.628_{\pm 0.011}$ </td><td> $2.586_{\pm 0.010}$ </td></tr><tr><td>Textual Entailment</td><td> $2.472_{\pm 0.002}$ </td><td> $2.503_{\pm 0.003}$ </td><td> $2.468_{\pm 0.002}$ </td></tr><tr><td>Title Generation</td><td> $3.027_{\pm 0.002}$ </td><td> $3.037_{\pm 0.002}$ </td><td> $3.015_{\pm 0.002}$ </td></tr><tr><td>Word Analogy</td><td> $1.665_{\pm 0.016}$ </td><td> $1.682_{\pm 0.015}$ </td><td> $1.668_{\pm 0.016}$ </td></tr><tr><td>Average</td><td> $2.546_{\pm 0.003}$ </td><td> $2.572_{\pm 0.003}$ </td><td> $2.540_{\pm 0.003}$ </td></tr></table>

Table 8: Validation loss per skill for data selection in out-of-domain setting over Natural Instructions train task split and test task split.

In Table 9 we provide a breakdown of the RedPajama experiment's accuracy per evaluation skill, corresponding to the results in Figure 7.

<table><tr><td rowspan="2"></td><td colspan="2">1 Billion Tokens</td><td colspan="2">2 Billion Tokens</td><td colspan="2">3 Billion Tokens</td></tr><tr><td>Uniform</td><td>SKILL-IT</td><td>Uniform</td><td>SKILL-IT</td><td>Uniform</td><td>SKILL-IT</td></tr><tr><td>ARC Challenge (acc norm)</td><td>35.4</td><td>34.6</td><td>35.3</td><td>34.9</td><td>34.6</td><td>34.8</td></tr><tr><td>ARC Easy (acc norm)</td><td>62.2</td><td>61.2</td><td>62.4</td><td>61.7</td><td>62.5</td><td>62.0</td></tr><tr><td>BoolQ</td><td>68.9</td><td>68.2</td><td>67.7</td><td>68.6</td><td>67.2</td><td>68.7</td></tr><tr><td>COPA</td><td>81.0</td><td>82.0</td><td>80.0</td><td>81.0</td><td>81.0</td><td>81.0</td></tr><tr><td>HellaSwag (acc norm)</td><td>63.9</td><td>63.7</td><td>63.8</td><td>63.9</td><td>64.0</td><td>63.9</td></tr><tr><td>LAMBADA OpenAI</td><td>64.4</td><td>67.0</td><td>65.9</td><td>66.7</td><td>66.8</td><td>66.0</td></tr><tr><td>PIQA (acc norm)</td><td>74.8</td><td>75.0</td><td>75.5</td><td>75.2</td><td>75.0</td><td>75.7</td></tr><tr><td>Winogrande</td><td>62.8</td><td>63.9</td><td>63.9</td><td>63.2</td><td>63.4</td><td>63.1</td></tr><tr><td>Average accuracy</td><td>64.2</td><td>64.4</td><td>64.3</td><td>64.4</td><td>64.3</td><td>64.4</td></tr></table>

Table 9: Performance of model trained on RedPajama with uniform sampling and SKILL-IT on LM evaluation harness. Unless otherwise noted, accuracy is reported for each task.

# D.3.2 Weight trajectories

We provide SKILL-IT's weight trajectories for each result. The weight per skill across training steps for the LEGO pre-training experiment corresponding to Figure 4 (left) is shown in Figure 20. We see that SKILL-IT initially allocates more weight to skill 2 and less to 1, 3, 4, 5. Since skill 1 is learned quickly, the weight on skill 1 immediately drops to below 0.1 at 1000 steps. The weight on skills 3, 4, and 5 increase from around 0 to 3000 steps, during which their respective validation losses are higher than those of skills 1 and 2. Near the end of training, all losses are converging to 0, and so the weight per skill is roughly uniform.

The weight per skill across training steps for the addition pre-training experiment corresponding to Figure 4 (right) is shown in Figure 21. SKILL-IT allocates more weight to skill 2, which has an edge to skill 1 as shown in Figure 11. It also allocates very little weight to skill 3, which is learned faster than the other two skills. Eventually, it puts more weight on skill 1, the hardest skill, and then converges to uniform sampling as all validation losses approach 0.

The weight per skill across training steps for the LEGO fine-tuning experiment and the Spanish question generation and stance detection experiments corresponding to Figure 5 is shown in Figure 22. Since there is only one target skill in these experiments, the mixture of weights approaches uniform as the loss on the target skill approaches 0. It is interesting

![](images/dadf4270e012905d56f22fc64ace00fc83a9dffb5617ce2374fb589924c46c27.jpg)

<details>
<summary>line</summary>

| Steps | Skill 1 | Skill 2 | Skill 3 | Skill 4 | Skill 5 |
|-------|---------|---------|---------|---------|---------|
| 0     | 0.07    | 0.35    | 0.21    | 0.20    | 0.16    |
| 1000  | 0.07    | 0.35    | 0.21    | 0.20    | 0.18    |
| 2000  | 0.08    | 0.25    | 0.22    | 0.22    | 0.23    |
| 3000  | 0.12    | 0.24    | 0.19    | 0.22    | 0.23    |
| 4000  | 0.16    | 0.24    | 0.21    | 0.21    | 0.18    |
| 5000  | 0.18    | 0.23    | 0.21    | 0.21    | 0.19    |
| 6000  | 0.18    | 0.22    | 0.21    | 0.21    | 0.19    |
</details>

Figure 20: Weight per skill for LEGO pre-training experiment. SKILL-IT initially allocates more weight to skill 2, but eventually puts more weight on harder skills (3, 4, 5) before converging to uniform sampling when all losses converge roughly to 0.

to explore how to reduce edge weights and regularization so that the mixture approaches the target skill instead, although preliminary experiments where we decayed the edge weight and the strength of the Bregman divergence term did not appear better. We hypothesize that since training on a uniform mixture (as in Figure 3) did strictly better than training on the target skill and their loss curves did not intersect during the training run, it is better to allocate non-negligible weight on all skills throughout the training run.

The weight per skill across training steps for the Natural Instructions out-of-domain experiment corresponding to Figure 6 is shown in Figure 23, where the legend is provided for the top 10 task categories with the largest weights. While the initial weights based on the skills graph roughly establishes the order of weight magnitude, the differences among the losses on the evaluation skills increases the range of weights as training continues. As validation losses saturate, the weights also converge to fixed values.

# D.4 Experiments on 1.3B parameter model

We demonstrate that the skills graph learned on the 125M parameter model can be used for data selection with the GPT-Neo-1.3B model. We present results in the continual pre-training setting on the LEGO synthetic and Natural Instructions.

All results are reported over 3 random seeds. For the LEGO experiment, we train for 1500 steps with $\eta = 0.5$ , T = 30, w = 3. For the NI experiment, we train for 5000 steps with $\eta = 0.2$ , and T = 1. The skill graphs were learned using the 125M parameter model as described in section C.2.

In Figure 24, we train the 1.3B model using SKILL-IT for the LEGO synthetic and find that it still outperforms random and skill-stratified sampling on average. In particular, while performance across sampling methods is similar for early skills, the discrepancy is larger for skill 5, for which SKILL-IT allocates more weight to dynamically. In Figure 25, we provide the weight trajectories of SKILL-IT. We observe that the weight trajectories are similar to that on the 125M parameter model, where initial weight is allocated towards skill 2. Later on, more weight is allocated towards skills 4 and 5, whose losses are higher, and eventually the weight mixture converges to uniform as all losses converge to near 0.

In Table 10, we report performance of SKILL-IT with the 1.3B model on the Natural Instructions pre-training experiment and find that the trends from the smaller model hold—SKILL-IT outperforms random and skill-stratified sampling on average.

# D.5 Ablations

We report ablations on the skills graph and the online component of SKILL-IT. Instead of using $A$ in Algorithm 1, we study the performance when the identity matrix is used instead; intuitively, this corresponds to a misspecified skills graph where

Skill-It addition weights   
![](images/ba0a446575f8d4a58a89b23bec8124da367618852d36ec0e3e533b873d930e19.jpg)

<details>
<summary>line</summary>

| Steps | Skill 1 | Skill 2 | Skill 3 |
| ----- | ------- | ------- | ------- |
| 0     | 0.32    | 0.36    | 0.32    |
| 1000  | 0.28    | 0.36    | 0.32    |
| 1500  | 0.28    | 0.56    | 0.16    |
| 2000  | 0.28    | 0.56    | 0.16    |
| 2500  | 0.38    | 0.42    | 0.20    |
| 3500  | 0.38    | 0.42    | 0.34    |
| 4000  | 0.34    | 0.34    | 0.34    |
| 6000  | 0.34    | 0.34    | 0.34    |
</details>

Figure 21: Weight per skill for addition pre-training experiment. SKILL-IT initially allocates more weight to skill 2, which has an edge to skill 1, while allocating little weight to skill 3 which is learned quickly. Eventually, SKILL-IT puts more weight on the harder skill 1 before converging to uniform sampling when all losses roughly approach 0.

![](images/24c35800552ca7e6eb25a49eabba9f9e0968e4c86bb021cadc2654e52aca6f7a.jpg)

<details>
<summary>line</summary>

| Steps | Skill 1 | Skill 2 | Skill 3 |
| ----- | ------- | ------- | ------- |
| 0     | 0.24    | 0.38    | 0.38    |
| 500   | 0.25    | 0.37    | 0.37    |
| 1000  | 0.25    | 0.37    | 0.37    |
| 1500  | 0.30    | 0.37    | 0.35    |
| 2000  | 0.32    | 0.37    | 0.34    |
| 2500  | 0.33    | 0.37    | 0.34    |
| 3000  | 0.33    | 0.37    | 0.34    |
| 3500  | 0.33    | 0.37    | 0.34    |
| 4000  | 0.33    | 0.37    | 0.34    |
| 4500  | 0.33    | 0.37    | 0.34    |
| 5000  | 0.33    | 0.37    | 0.34    |
| 5500  | 0.33    | 0.37    | 0.34    |
| 6000  | 0.33    | 0.37    | 0.34    |
</details>

![](images/4bb45215b74487a2c1447cfed570373b20d62ea1d02321c1d9c3cb70c1e88e3a.jpg)

<details>
<summary>line</summary>

| Steps | English QA | Spanish QA | English QG | Spanish QG |
| ----- | ---------- | ---------- | ---------- | ---------- |
| 0     | 0.21       | 0.36       | 0.22       | 0.29       |
| 100   | 0.14       | 0.35       | 0.15       | 0.36       |
| 200   | 0.14       | 0.35       | 0.16       | 0.35       |
| 300   | 0.14       | 0.35       | 0.16       | 0.35       |
| 400   | 0.15       | 0.34       | 0.17       | 0.34       |
| 500   | 0.15       | 0.34       | 0.17       | 0.34       |
| 600   | 0.15       | 0.34       | 0.17       | 0.34       |
</details>

![](images/6c39ea205978465a6be57868cb07d6ed718a4053afeb55883b7811290f0ab4a5.jpg)

<details>
<summary>line</summary>

| Steps | Stance Detection | Text Matching |
| ----- | ---------------- | ------------- |
| 0     | 0.53             | 0.47          |
| 100   | 0.58             | 0.42          |
| 200   | 0.57             | 0.42          |
| 300   | 0.57             | 0.42          |
| 400   | 0.56             | 0.44          |
| 500   | 0.56             | 0.44          |
| 600   | 0.56             | 0.44          |
</details>

Figure 22: Weight per skill for fine-tuning experiments. Left: LEGO; Center: Spanish question generation; Right: stance detection.

![](images/305f501b18cead0104aef28d6ee3e63086bfbc39a0bf622233ced6f6aacda97d.jpg)

<details>
<summary>line</summary>

| Steps | question_generation | question_answering | text_categorization | sentiment_analysis | wrong_candidate_generation | text_matching | summarization | information_extraction | question_understanding | toxic_language_detection |
| ----- | ------------------- | ------------------ | ------------------ | ------------------ | -------------------------- | -------------- | ------------- | ----------------------- | ---------------------- | ------------------------ |
| 0     | 0.02                | 0.02               | 0.02               | 0.02               | 0.02                       | 0.02           | 0.02          | 0.02                    | 0.02                   | 0.02                     |
| 1000  | 0.14                | 0.13               | 0.08               | 0.07               | 0.06                       | 0.05           | 0.05          | 0.04                    | 0.03                   | 0.04                     |
| 2000  | 0.14                | 0.13               | 0.08               | 0.07               | 0.06                       | 0.05           | 0.05          | 0.04                    | 0.03                   | 0.04                     |
| 3000  | 0.14                | 0.13               | 0.08               | 0.07               | 0.06                       | 0.05           | 0.05          | 0.04                    | 0.03                   | 0.04                     |
| 4000  | 0.14                | 0.13               | 0.08               | 0.07               | 0.06                       | 0.05           | 0.05          | 0.04                    | 0.03                   | 0.04                     |
| 5000  | 0.14                | 0.13               | 0.08               | 0.07               | 0.06                       | 0.05           | 0.05          | 0.04                    | 0.03                   | 0.04                     |
</details>

Figure 23: Weight per skill for Natural Instructions out-of-domain experiment. The legend shows the top 10 skills with the largest weight. While the relative order of weight magnitude does not change significantly across training, the incorporation of loss dramatically increases the range of the weights, showing the importance of an online algorithm.

![](images/88d888b98bc1d3db3fa8b78f073b859cfac2425c97808ffb1641ea8b9870582c.jpg)

<details>
<summary>line</summary>

| Step | Validation Loss (Log) |
| ---- | --------------------- |
| 0    | 0.1                   |
| 100  | 0.01                  |
| 200  | 0.005                 |
| 300  | 0.003                 |
| 400  | 0.002                 |
| 500  | 0.001                 |
| 600  | 0.0008                |
| 700  | 0.0006                |
| 800  | 0.0005                |
| 900  | 0.0004                |
| 1000 | 0.0003                |
| 1100 | 0.0002                |
| 1200 | 0.0001                |
| 1300 | 0.0001                |
| 1400 | 0.0001                |
| 1500 | 0.0001                |
</details>

![](images/473393d548574f0675cc21174fd24528f7af7b696a48cb42e7b0c6ccfd006d40.jpg)

<details>
<summary>line</summary>

| Step | Green Line | Orange Line | Blue Line |
|------|------------|-------------|-----------|
| 0    | 1.0        | 1.0         | 1.0       |
| 500  | 0.01       | 0.05        | 0.03      |
| 1000 | 0.001      | 0.005       | 0.003     |
| 1500 | 0.0001     | 0.001       | 0.0005    |
</details>

![](images/88a44837827d09cb8e22bc61cdf49c86c4ff7d08dffc9d872c1f9909d91a146a.jpg)

<details>
<summary>line</summary>

| Step | Blue Line | Orange Line | Green Line |
|------|-----------|-------------|------------|
| 0    | 1.0       | 1.0         | 1.0        |
| 500  | 0.1       | 0.1         | 0.1        |
| 1000 | 0.01      | 0.01        | 0.01       |
| 1500 | 0.001     | 0.001       | 0.001      |
</details>

![](images/d89a2b5869b9e0d6cfee212ffe86b8488eec052b601a9f896eec1417df5f47ac.jpg)

<details>
<summary>line</summary>

| Steps | Validation Loss (Log) |
| ----- | --------------------- |
| 0     | 1.0                   |
| 500   | 0.1                   |
| 1000  | 0.01                  |
| 1500  | 0.001                 |
</details>

![](images/9a10ae55b73eebea315507bd210d348e0d102ca063b20b14c544f2ec3a420179.jpg)

<details>
<summary>line</summary>

| Steps | Blue Line | Orange Line | Green Line |
|-------|-----------|-------------|------------|
| 0     | 1.0       | 1.0         | 1.0        |
| 500   | 0.3       | 0.25        | 0.2        |
| 1000  | 0.05      | 0.04        | 0.03       |
| 1500  | 0.01      | 0.01        | 0.005      |
</details>

![](images/2f0a8d91cb32faaba6c7664e0ca09ff2a053ecdad10031aeb9077e407ae0ff92.jpg)

<details>
<summary>line</summary>

| Steps | Random | Skill-stratified | Skill-It |
| ----- | ------ | ---------------- | -------- |
| 0     | 1.0    | 1.0              | 1.0      |
| 500   | 0.1    | 0.1              | 0.1      |
| 1000  | 0.01   | 0.01             | 0.01     |
| 1500  | 0.001  | 0.001            | 0.001    |
</details>

Figure 24: Performance of SKILL-IT for LEGO pre-training setting when skills graph is learned on a 125M parameter model and used for data selection with a 1.3B model. SKILL-IT on average still outperforms random and skill-stratified sampling, suggesting that findings on ordered skill sets can transfer from small models to large models.

Skill-It LEGO weights for 1.3B param model   
![](images/f8902dc6b14f14ab5c21734aec04b315d1f8b6b5d886e8763ab85d62793ef031.jpg)

<details>
<summary>line</summary>

| Steps | Skill 1 | Skill 2 | Skill 3 | Skill 4 | Skill 5 |
|-------|---------|---------|---------|---------|---------|
| 0     | 0.0     | 0.35    | 0.18    | 0.18    | 0.17    |
| 100   | 0.03    | 0.65    | 0.19    | 0.22    | 0.10    |
| 200   | 0.08    | 0.38    | 0.20    | 0.23    | 0.25    |
| 300   | 0.10    | 0.25    | 0.19    | 0.22    | 0.27    |
| 400   | 0.12    | 0.24    | 0.18    | 0.21    | 0.27    |
| 500   | 0.13    | 0.23    | 0.18    | 0.21    | 0.26    |
| 600   | 0.14    | 0.22    | 0.18    | 0.21    | 0.25    |
| 700   | 0.15    | 0.21    | 0.18    | 0.21    | 0.24    |
| 800   | 0.16    | 0.21    | 0.18    | 0.21    | 0.23    |
| 900   | 0.17    | 0.21    | 0.18    | 0.21    | 0.22    |
| 1000  | 0.18    | 0.21    | 0.19    | 0.21    | 0.21    |
| 1100  | 0.19    | 0.21    | 0.19    | 0.21    | 0.21    |
| 1200  | 0.19    | 0.21    | 0.19    | 0.21    | 0.21    |
| 1300  | 0.19    | 0.21    | 0.19    | 0.21    | 0.21    |
| 1400  | 0.19    | 0.21    | 0.19    | 0.21    | 0.21    |
| 1500  | 0.19    | 0.21    | 0.19    | 0.21    | 0.21    |
</details>

Figure 25: Weight per skill for LEGO pre-training experiment on 1.3B parameter model. The trajectories are similar to those of the 125M parameter model in Figure 20. SKILL-IT initially allocates more weight to skill 2, but eventually puts more weight on skills 4 and 5 before converging to uniform sampling when all losses converge to near 0.

<table><tr><td>Skill</td><td>Random</td><td>Skill-stratified</td><td>SKILL-IT</td></tr><tr><td>Answer Verification</td><td> $2.005 \pm 0.059$ </td><td> $1.903 \pm 0.069$ </td><td> $1.890 \pm 0.072$ </td></tr><tr><td>Code to Text</td><td> $0.302 \pm 0.032$ </td><td> $0.204 \pm 0.022$ </td><td> $0.269 \pm 0.032$ </td></tr><tr><td>Discourse Connective Identification</td><td> $2.529 \pm 0.046$ </td><td> $2.372 \pm 0.054$ </td><td> $2.393 \pm 0.056$ </td></tr><tr><td>Entity Generation</td><td> $2.108 \pm 0.328$ </td><td> $1.788 \pm 0.429$ </td><td> $1.885 \pm 0.461$ </td></tr><tr><td>Entity Relation Classification</td><td> $1.130 \pm 0.048$ </td><td> $0.836 \pm 0.006$ </td><td> $0.841 \pm 0.010$ </td></tr><tr><td>Information Extraction</td><td> $2.032 \pm 0.013$ </td><td> $1.992 \pm 0.006$ </td><td> $1.933 \pm 0.013$ </td></tr><tr><td>Irony Detection</td><td> $2.802 \pm 0.125$ </td><td> $2.528 \pm 0.146$ </td><td> $2.585 \pm 0.149$ </td></tr><tr><td>Preposition Prediction</td><td> $1.095 \pm 0.040$ </td><td> $0.686 \pm 0.041$ </td><td> $0.774 \pm 0.029$ </td></tr><tr><td>Punctuation Error Detection</td><td> $2.633 \pm 0.027$ </td><td> $3.188 \pm 0.055$ </td><td> $2.726 \pm 0.025$ </td></tr><tr><td>Question Answering</td><td> $1.947 \pm 0.003$ </td><td> $2.119 \pm 0.003$ </td><td> $2.073 \pm 0.001$ </td></tr><tr><td>Question Generation</td><td> $2.214 \pm 0.007$ </td><td> $2.345 \pm 0.008$ </td><td> $2.263 \pm 0.010$ </td></tr><tr><td>Question Understanding</td><td> $1.928 \pm 0.020$ </td><td> $1.837 \pm 0.031$ </td><td> $1.700 \pm 0.042$ </td></tr><tr><td>Sentence Expansion</td><td> $2.054 \pm 0.018$ </td><td> $1.828 \pm 0.060$ </td><td> $1.853 \pm 0.058$ </td></tr><tr><td>Sentiment Analysis</td><td> $2.771 \pm 0.009$ </td><td> $2.818 \pm 0.006$ </td><td> $2.774 \pm 0.007$ </td></tr><tr><td>Stance Detection</td><td> $1.814 \pm 0.151$ </td><td> $1.500 \pm 0.117$ </td><td> $1.628 \pm 0.149$ </td></tr><tr><td>Summarization</td><td> $2.531 \pm 0.009$ </td><td> $2.472 \pm 0.012$ </td><td> $2.440 \pm 0.013$ </td></tr><tr><td>Text Categorization</td><td> $2.289 \pm 0.016$ </td><td> $2.341 \pm 0.021$ </td><td> $2.231 \pm 0.022$ </td></tr><tr><td>Text Matching</td><td> $1.967 \pm 0.008$ </td><td> $1.913 \pm 0.005$ </td><td> $1.872 \pm 0.005$ </td></tr><tr><td>Text Simplification</td><td> $1.861 \pm 0.003$ </td><td> $1.692 \pm 0.023$ </td><td> $1.698 \pm 0.022$ </td></tr><tr><td>Text to Code</td><td> $0.614 \pm 0.030$ </td><td> $0.518 \pm 0.030$ </td><td> $0.585 \pm 0.022$ </td></tr><tr><td>Toxic Language Detection</td><td> $2.853 \pm 0.020$ </td><td> $2.911 \pm 0.019$ </td><td> $2.862 \pm 0.018$ </td></tr><tr><td>Word Semantics</td><td> $1.999 \pm 0.023$ </td><td> $1.870 \pm 0.039$ </td><td> $1.902 \pm 0.024$ </td></tr><tr><td>Wrong Candidate Generation</td><td> $2.187 \pm 0.028$ </td><td> $2.192 \pm 0.023$ </td><td> $2.140 \pm 0.020$ </td></tr><tr><td>Average</td><td> $1.985 \pm 0.022$ </td><td> $1.907 \pm 0.027$ </td><td> $1.883 \pm 0.032$ </td></tr></table>

Table 10: Results when skills graph for Natural Instructions learned on 125M parameter model is used for data selection with a 1.3B model. We see that SKILL-IT on average still outperforms random and skill-stratified sampling, even though the edges used by SKILL-IT are not derived from the larger model.

no skill influences another skill. We refer this approach as “No graph”. Note that the opposite case of a complete graph recovers skill-stratified sampling, which we already have as a baseline.

Second, instead of sampling over multiple rounds and weighting according to the loss of each skill, we study the effect of setting T = 1, which only uses a softmax on A to yield static weights on the skills. We refer to this approach as “Static”. We omit results on Natural Instructions continual pre-training, since SKILL-IT uses T = 1 and using no graph with T = 1 recovers skill-stratified sampling. Intuitively, we expect the static version of SKILL-IT to perform somewhat well unless there is significant discrepancy among the losses (e.g. in synthetics where the loss on one skill can be close to 0 while the other is not, versus in Natural Instructions where all losses decrease consistently). For both ablations, we sweep over values of $\eta = [0.1, 0.2, 0.5, 0.8]$ .

Figure 26 shows the comparison between SKILL-IT and no graph on the continual pre-training LEGO experiment, and Figure 27 shows the comparison between SKILL-IT and a static approach. We see that both the graph and the online dynamics of SKILL-IT are important for its performance. In particular, using no graph results in allocating significant weight to harder skills early on, even though many of them have easier prerequisite skills (such as skill 3 having edges to skills 1 and 2). Using a static graph results in consistent allocation of significant weight to prerequisite skills even after their validation losses converge to near 0, and thus the harder skills that have higher loss are not learned quickly afterwards.

We perform the same ablation on the Addition dataset—the results for this are shown in Figures 28 and Figure 29. We find that these simple baselines, including using a static graph and no graph perform similarly to SKILL-IT on average across all skills—while SKILL-IT performs the best on skill 2 compared to vanilla multiplicative weights, and SKILL-IT performs the best on skill 1 compared to a static graph. This suggests that Addition is somewhat easier than the other datasets that we consider, as SKILL-IT still outperforms other baselines, as shown in Figure 4.

Figure 30 compares SKILL-IT, no graph, and static data selection for the LEGO fine-tuning experiment. No graph can be interpreted as allocating equal weight to all training skills not equal to the target skill, and varying this weight versus the weight on the target skill. While SKILL-IT and setting T = 1 behave similarly, we see that SKILL-IT is slightly better than using no graph. For instance, SKILL-IT obtains a validation loss of 0.05 in 2000 steps, compared to 2050-2200 steps when using no graph.

Figure 31 and 32 compare SKILL-IT, no graph, and static data selection for the Natural Instructions fine-tuning experiments. For both Spanish QG and stance detection, SKILL-IT attains lower loss than using no graph or using T = 1 round.

Figure 33 compares SKILL-IT and static data selection for the Natural Instructions out-of-domain experiment. SKILL-IT

![](images/7e745b2fc2b5083c2f5d4fb67bf5348cae50c5b91b6dafa6fb681dcc3b9f0a5e.jpg)

<details>
<summary>line</summary>

| Step | Validation Loss (Log) |
| ---- | --------------------- |
| 0    | 1.0                   |
| 2000 | 0.001                 |
| 4000 | 0.0005                |
| 6000 | 0.0001                |
</details>

![](images/47a22b9905c1dda8957d4447d53cff1198004bb4380cdabf8d377f6139863212.jpg)

<details>
<summary>line</summary>

| Step | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
|------|--------|--------|--------|--------|--------|
| 0    | 1.0    | 1.0    | 1.0    | 1.0    | 1.0    |
| 1000 | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    |
| 2000 | 0.05   | 0.05   | 0.05   | 0.05   | 0.05   |
| 3000 | 0.02   | 0.02   | 0.02   | 0.02   | 0.02   |
| 4000 | 0.01   | 0.01   | 0.01   | 0.01   | 0.01   |
| 5000 | 0.005  | 0.005  | 0.005  | 0.005  | 0.005  |
| 6000 | 0.002  | 0.002  | 0.002  | 0.002  | 0.002  |
</details>

![](images/548408ad1f96b1d6ab54c185cffdfe559ce92d5b6f8e12aa22288cf6c2a43d6d.jpg)

<details>
<summary>line</summary>

| Step | Value |
| ---- | ----- |
| 0    | ~1.0  |
| 2000 | ~0.8  |
| 4000 | ~0.5  |
| 6000 | ~0.3  |
</details>

![](images/fce89d7babc1ab305824c7fd7919f663530183ee042e467b44b3a9f29c4ed6fc.jpg)

<details>
<summary>line</summary>

| Steps | Validation Loss (Log) |
| ----- | --------------------- |
| 0     | ~1.0                  |
| 2000  | ~0.5                  |
| 4000  | ~0.1                  |
| 6000  | ~0.01                 |
</details>

![](images/83edef27bf83da8583dfec974be825ccc27a24b2460916a7546e21e9b130db17.jpg)

<details>
<summary>line</summary>

| Steps | Value     |
| ----- | --------- |
| 0     | ~1.0      |
| 2000  | ~0.8      |
| 4000  | ~0.3      |
| 6000  | ~0.1      |
</details>

![](images/3d97295d03a4ed889a19dd0159ba9ec38d375aa9551856dceec313ff9d619d39.jpg)

<details>
<summary>line</summary>

| Steps | No graph (0.1) | No graph (0.2) | No graph (0.5) | No graph (0.8) | Skill-It |
| ----- | -------------- | -------------- | -------------- | -------------- | -------- |
| 0     | ~10^1          | ~10^1          | ~10^1          | ~10^1          | ~10^1    |
| 2000  | ~10^-1         | ~10^-1         | ~10^-1         | ~10^-1         | ~10^-1   |
| 4000  | ~10^-2         | ~10^-2         | ~10^-2         | ~10^-2         | ~10^-2   |
| 6000  | ~10^-2         | ~10^-2         | ~10^-2         | ~10^-2         | ~10^-2   |
</details>

Figure 26: Comparison of SKILL-IT versus using the identity adjacency matrix (no skills graph) with $\eta = 0.1, 0.2, 0.5, 0.8$ on the LEGO continual pre-training experiment. The latter does not capture the relationship between skills, and we find that SKILL-IT attains lower loss on all skills.

![](images/b42d2a0a0f7d66ddea33e4b33774e8526d05b64c767ec061071553458b506573.jpg)

<details>
<summary>line</summary>

| Step | Validation Loss (Log) |
| ---- | --------------------- |
| 0    | 1.0                   |
| 1000 | 0.01                  |
| 2000 | 0.001                 |
| 3000 | 0.005                 |
| 4000 | 0.01                  |
| 5000 | 0.001                 |
| 6000 | 0.0001                |
</details>

![](images/6fc3e04a5fb94024eece90834e60aa4a483709dec2beb22207793eea147abeaf.jpg)

<details>
<summary>line</summary>

| Step | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 |
|------|----------|----------|----------|----------|----------|
| 0    | 1.0      | 1.0      | 1.0      | 1.0      | 1.0      |
| 1000 | 0.1      | 0.1      | 0.1      | 0.1      | 0.1      |
| 2000 | 0.05     | 0.05     | 0.05     | 0.05     | 0.05     |
| 3000 | 0.02     | 0.02     | 0.02     | 0.02     | 0.02     |
| 4000 | 0.01     | 0.01     | 0.01     | 0.01     | 0.01     |
| 5000 | 0.005    | 0.005    | 0.005    | 0.005    | 0.005    |
| 6000 | 0.002    | 0.002    | 0.002    | 0.002    | 0.002    |
</details>

![](images/c28faf00a35dbeea150039d08583061eff24e494701c69a3ea9352b3d1c0c34d.jpg)

<details>
<summary>line</summary>

| Step | Red Line | Green Line | Orange Line | Blue Line | Purple Line |
|------|----------|------------|-------------|-----------|-------------|
| 0    | 1.0      | 1.0        | 1.0         | 1.0       | 1.0         |
| 2000 | ~0.8     | ~0.7       | ~0.6        | ~0.5      | ~0.4        |
| 4000 | ~0.5     | ~0.4       | ~0.3        | ~0.2      | ~0.1        |
| 6000 | ~0.3     | ~0.2       | ~0.1        | ~0.05     | ~0.02       |
</details>

![](images/7bab4d83a2c8d67557bbeeee305ed5537621484cefad844a5095c9d5ab8a372c.jpg)

<details>
<summary>line</summary>

| Steps | Validation Loss (Log) |
| ----- | --------------------- |
| 0     | 1.0                   |
| 2000  | 0.5                   |
| 4000  | 0.1                   |
| 6000  | 0.01                  |
</details>

![](images/02358db1cda353e82864b8f67a299467c782a134f32ab80aa9ff6d0e7b3464e5.jpg)

<details>
<summary>line</summary>

| Steps | Red Line | Green Line | Orange Line | Blue Line | Purple Line |
|-------|----------|------------|-------------|-----------|-------------|
| 0     | 1.0      | 1.0        | 1.0         | 1.0       | 1.0         |
| 2000  | 0.9      | 0.9        | 0.9         | 0.9       | 0.9         |
| 4000  | 0.7      | 0.6        | 0.5         | 0.5       | 0.3         |
| 6000  | 0.5      | 0.4        | 0.3         | 0.3       | 0.1         |
</details>

![](images/7dbc914928c948a716bc2df951e64e6ba3e7c76f9c3fb589b20b3d6d2988cd9e.jpg)

<details>
<summary>line</summary>

| Steps | Static (0.1) | Static (0.2) | Static (0.5) | Static (0.8) | Skill-It |
|-------|--------------|--------------|--------------|--------------|----------|
| 0     | ~1.0         | ~1.0         | ~1.0         | ~1.0         | ~1.0     |
| 2000  | ~0.3         | ~0.4         | ~0.5         | ~0.6         | ~0.2     |
| 4000  | ~0.1         | ~0.2         | ~0.3         | ~0.4         | ~0.05    |
| 6000  | ~0.05        | ~0.1         | ~0.2         | ~0.3         | ~0.01    |
</details>

Figure 27: Comparison of SKILL-IT versus using static data selection $(T = 1)$ with $\eta = 0.1, 0.2, 0.5, 0.8$ on the LEGO continual pre-training experiment. While SKILL-IT eventually allocates more weights to skills 3, 4, 5, which have higher loss, the static approach is not able to do this. We find that SKILL-IT attains lower loss on all skills.

![](images/38c7d13f05b62dc2338823d847bfcbbc8fab3f0a8c231b9cd4c32dcd56858282.jpg)

<details>
<summary>line</summary>

| Step | Validation Loss (Log) |
| ---- | --------------------- |
| 0    | 1.0                   |
| 2000 | 0.1                   |
| 4000 | 0.01                  |
| 6000 | 0.001                 |
</details>

![](images/d9388ac6fe710c8c18b0168d725097ec2c4efdb3a40c150a04d4139ffd8e809b.jpg)

<details>
<summary>line</summary>

| Step | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 |
|------|----------|----------|----------|----------|----------|
| 0    | 10^0     | 10^0     | 10^0     | 10^0     | 10^0     |
| 2000 | ~10^-1   | ~10^-1   | ~10^-1   | ~10^-1   | ~10^-1   |
| 4000 | ~10^-2   | ~10^-2   | ~10^-2   | ~10^-2   | ~10^-2   |
| 6000 | ~10^-2   | ~10^-2   | ~10^-2   | ~10^-2   | ~10^-2   |
</details>

![](images/4553706b40b2409361e2041d99c94d9b4e48d1c3caae136c28202ccf976f1829.jpg)

<details>
<summary>line</summary>

| Steps | Validation Loss (Log) |
| ----- | --------------------- |
| 0     | 1.0                   |
| 2000  | 0.1                   |
| 4000  | 0.01                  |
| 6000  | 0.001                 |
</details>

![](images/d1a5e08eaa5b56560f0eff5519aa1924cf7a24b2ae1bbf0054ea2ca6e7667e65.jpg)

<details>
<summary>line</summary>

| Steps | No graph (0.1) | No graph (0.2) | No graph (0.5) | No graph (0.8) | Skill-lt |
|-------|----------------|----------------|----------------|----------------|----------|
| 0     | 10^0           | 10^0           | 10^0           | 10^0           | 10^0     |
| 2000  | ~10^-1         | ~10^-1         | ~10^-1         | ~10^-1         | ~10^-1   |
| 4000  | ~10^-2         | ~10^-2         | ~10^-2         | ~10^-2         | ~10^-2   |
| 6000  | ~10^-3         | ~10^-3         | ~10^-3         | ~10^-3         | ~10^-3   |
</details>

Figure 28: Comparison of SKILL-IT versus using the identity adjacency matrix (no skills graph) with $\eta = 0.1, 0.2, 0.5, 0.8$ on the Addition continual pre-training experiment. The latter does not capture the relationship between skills, and we find that SKILL-IT attains lower loss on skill 2, but attains similar performance to methods that do not use the skills graph.   
![](images/b4aca2952e56b4ec14d7b18aa431577bfdb25dba1f8db356b4ee528e33482427.jpg)

<details>
<summary>line</summary>

| Step | Validation Loss (Log) |
| ---- | --------------------- |
| 0    | 1.0                   |
| 2000 | 0.1                   |
| 4000 | 0.01                  |
| 6000 | 0.001                 |
</details>

![](images/76c67e739545f8491dd6334a265fa1304738b5174990440fd5a47519230f5443.jpg)

<details>
<summary>line</summary>

| Step | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 |
|------|----------|----------|----------|----------|----------|
| 0    | 10^0     | 10^0     | 10^0     | 10^0     | 10^0     |
| 2000 | ~10^-1   | ~10^-1   | ~10^-1   | ~10^-1   | ~10^-1   |
| 4000 | ~10^-2   | ~10^-2   | ~10^-2   | ~10^-2   | ~10^-2   |
| 6000 | ~10^-2   | ~10^-2   | ~10^-2   | ~10^-2   | ~10^-2   |
</details>

![](images/75e5291a24240ee0109c3699775b6a750b57170b40bb10f9c6a432b3519a8196.jpg)

<details>
<summary>line</summary>

| Steps | Validation Loss (Log) |
| ----- | --------------------- |
| 0     | 1.0                   |
| 1000  | 0.1                   |
| 2000  | 0.05                  |
| 3000  | 0.02                  |
| 4000  | 0.01                  |
| 5000  | 0.005                 |
| 6000  | 0.002                 |
</details>

![](images/d7731bf54261aa850ac27456fd275d581ca01ad925a1bf5d6e41430436ab9446.jpg)

<details>
<summary>line</summary>

| Steps | Static (0.1) | Static (0.2) | Static (0.5) | Static (0.8) | Skill-lt |
|-------|--------------|--------------|--------------|--------------|----------|
| 0     | 10^0         | 10^0         | 10^0         | 10^0         | 10^0     |
| 2000  | ~10^-1       | ~10^-1       | ~10^-1       | ~10^-1       | ~10^-1   |
| 4000  | ~10^-2       | ~10^-2       | ~10^-2       | ~10^-2       | ~10^-2   |
| 6000  | ~10^-3       | ~10^-3       | ~10^-3       | ~10^-3       | ~10^-3   |
</details>

Figure 29: Comparison of SKILL-IT versus using static data selection $(T = 1)$ with $\eta = 0.1, 0.2, 0.5, 0.8$ on the Addition continual pre-training experiment. We find that SKILL-IT attains lower loss on skill 1, but attains similar performance to the static methods.

![](images/1ed5b12a3694cd02347c37d4dcd0b8f926f37c8944721342f039598554f814db.jpg)

<details>
<summary>line</summary>

| Steps | No graph (0.1) | No graph (0.2) | No graph (0.5) | No graph (0.8) | Skill-It |
|-------|----------------|----------------|----------------|----------------|----------|
| 0     | 0.8            | 0.8            | 0.8            | 0.8            | 0.8      |
| 1000  | 0.7            | 0.7            | 0.7            | 0.7            | 0.7      |
| 2000  | 0.1            | 0.1            | 0.1            | 0.1            | 0.1      |
| 3000  | 0.05           | 0.05           | 0.05           | 0.05           | 0.05     |
| 4000  | 0.02           | 0.02           | 0.02           | 0.02           | 0.02     |
| 5000  | 0.01           | 0.01           | 0.01           | 0.01           | 0.01     |
| 6000  | 0.0            | 0.0            | 0.0            | 0.0            | 0.0      |
</details>

![](images/e10a42f075c8ad171d7d0593f63807d6e8bbb1d695a91f2595e6a543c55e8f8f.jpg)

<details>
<summary>line</summary>

| Steps | Static (0.1) | Static (0.2) | Static (0.5) | Static (0.8) | Skill-It |
|-------|--------------|--------------|--------------|--------------|----------|
| 0     | 0.75         | 0.75         | 0.75         | 0.75         | 0.75     |
| 1000  | 0.85         | 0.85         | 0.85         | 0.85         | 0.85     |
| 2000  | 0.10         | 0.10         | 0.10         | 0.10         | 0.10     |
| 3000  | 0.02         | 0.02         | 0.02         | 0.02         | 0.02     |
| 4000  | 0.01         | 0.01         | 0.01         | 0.01         | 0.01     |
| 5000  | 0.01         | 0.01         | 0.01         | 0.01         | 0.01     |
| 6000  | 0.01         | 0.01         | 0.01         | 0.01         | 0.01     |
</details>

Figure 30: Comparison of SKILL-IT versus using no graph (left) and static data selection (right) with $\eta = 0.1, 0.2, 0.5, 0.8$ on the LEGO fine-tuning experiment. All approaches have roughly the same loss trajectories, but SKILL-IT is slightly lower than using no graph.

![](images/78fbc77a4de13a0e2cd96d2bff2956447259b0cfe65e2c308d04673c2163d73e.jpg)

<details>
<summary>line</summary>

| Steps | No graph (0.1) | No graph (0.2) | No graph (0.5) | No graph (0.8) | Skill-It |
| ----- | -------------- | -------------- | -------------- | -------------- | -------- |
| 0     | 2.54           | 2.54           | 2.54           | 2.50           | 2.54     |
| 200   | 2.44           | 2.44           | 2.44           | 2.44           | 2.42     |
| 300   | 2.41           | 2.40           | 2.41           | 2.42           | 2.38     |
| 400   | 2.39           | 2.39           | 2.40           | 2.41           | 2.36     |
| 500   | 2.38           | 2.37           | 2.39           | 2.40           | 2.34     |
| 600   | 2.37           | 2.36           | 2.38           | 2.40           | 2.33     |
</details>

![](images/3b5dce80e9f1f9133a7b97b2101ff4b23de54a019d8bd27b37f8b3eda2bdcf6d.jpg)

<details>
<summary>line</summary>

| Steps | Static (0.1) | Static (0.2) | Static (0.5) | Static (0.8) | Skill-It |
| ----- | ------------ | ------------ | ------------ | ------------ | -------- |
| 0     | 2.53         | 2.52         | 2.51         | 2.50         | 2.50     |
| 200   | 2.44         | 2.43         | 2.42         | 2.41         | 2.41     |
| 300   | 2.40         | 2.39         | 2.38         | 2.37         | 2.37     |
| 400   | 2.37         | 2.36         | 2.35         | 2.34         | 2.34     |
| 500   | 2.35         | 2.34         | 2.33         | 2.32         | 2.32     |
| 600   | 2.34         | 2.33         | 2.32         | 2.31         | 2.31     |
</details>

Figure 31: Comparison of SKILL-IT versus using no graph (left) and static data selection (right) with $\eta = 0.1, 0.2, 0.5, 0.8$ on the Natural Instructions Spanish QG fine-tuning experiment. SKILL-IT attains lower validation loss than both no graph and static data selection.

![](images/c3738570729d381d605009b56465bd7a0ee59b74a03a4bda0fb345c0fbded24b.jpg)

<details>
<summary>line</summary>

| Steps | No graph (0.1) | No graph (0.2) | No graph (0.5) | No graph (0.8) | Skill-It |
| ----- | -------------- | -------------- | -------------- | -------------- | -------- |
| 0     | 1.7            | 1.7            | 1.7            | 1.7            | 1.7      |
| 200   | 1.5            | 1.5            | 1.5            | 1.5            | 1.5      |
| 300   | 1.4            | 1.4            | 1.4            | 1.4            | 1.4      |
| 400   | 1.35           | 1.35           | 1.35           | 1.35           | 1.3      |
| 500   | 1.3            | 1.3            | 1.3            | 1.3            | 1.25     |
| 600   | 1.3            | 1.3            | 1.3            | 1.3            | 1.25     |
</details>

![](images/029f8369c3df53a976a84cee0d21536cad2ca20e6653dcb4a471e5f4da5e5b55.jpg)

<details>
<summary>line</summary>

| Steps | Static (0.1) | Static (0.2) | Static (0.5) | Static (0.8) | Skill-It |
| ----- | ------------ | ------------ | ------------ | ------------ | -------- |
| 0     | 1.7          | 1.7          | 1.7          | 1.7          | 1.65     |
| 200   | 1.5          | 1.5          | 1.5          | 1.5          | 1.45     |
| 300   | 1.4          | 1.4          | 1.4          | 1.4          | 1.35     |
| 400   | 1.35         | 1.35         | 1.35         | 1.35         | 1.3      |
| 500   | 1.3          | 1.3          | 1.3          | 1.3          | 1.25     |
| 600   | 1.3          | 1.3          | 1.3          | 1.3          | 1.25     |
</details>

Figure 32: Comparison of SKILL-IT versus using no graph (left) and static data selection (right) with $\eta = 0.1, 0.2, 0.5, 0.8$ on the Natural Instructions stance detection fine-tuning experiment. SKILL-IT attains lower validation loss than both no graph and static data selection.

![](images/94d95b4aaa0d2650cfd8f767f8970d1cce3b1ef15977f5bdd8a487d096856bcd.jpg)  
Figure 33: Comparison of SKILL-IT versus using static data selection with $\eta = 0.1, 0.2, 0.5, 0.8$ on the Natural Instructions out-of-domain experiment. SKILL-IT attains the lowest validation loss on 7 out of 12 evaluation skills, and an average loss of 2.540 compared to a range of 2.541-2.551 for static data selection.

attains the lowest validation loss on 7 out of 12 evaluation skills. It has an average loss of 2.540 compared to a range of 2.541-2.551 for static data selection.