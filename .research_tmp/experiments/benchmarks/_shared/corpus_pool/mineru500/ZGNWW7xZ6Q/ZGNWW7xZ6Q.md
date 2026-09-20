# REASONING ON GRAPHS: FAITHFUL AND INTERPRETABLE LARGE LANGUAGE MODEL REASONING

Linhao Luo, Yuan-Fang Li, Gholamreza Haffari

Monash University

Australia

{linhao.luo, yuanfang.li, Gholamreza.Haffari}@monash.edu

Shirui Pan\*

Griffith University

Australia

s.pan@griffith.edu.au

# ABSTRACT

Large language models (LLMs) have demonstrated impressive reasoning abilities in complex tasks. However, they lack up-to-date knowledge and experience hallucinations during reasoning, which can lead to incorrect reasoning processes and diminish their performance and trustworthiness. Knowledge graphs (KGs), which capture vast amounts of facts in a structured format, offer a reliable source of knowledge for reasoning. Nevertheless, existing KG-based LLM reasoning methods only treat KGs as factual knowledge bases and overlook the importance of their structural information for reasoning. In this paper, we propose a novel method called reasoning on graphs (RoG) that synergizes LLMs with KGs to enable faithful and interpretable reasoning. Specifically, we present a planning-retrieval-reasoning framework, where RoG first generates relation paths grounded by KGs as faithful plans. These plans are then used to retrieve valid reasoning paths from the KGs for LLMs to conduct faithful reasoning. Furthermore, RoG not only distills knowledge from KGs to improve the reasoning ability of LLMs through training but also allows seamless integration with any arbitrary LLMs during inference. Extensive experiments on two benchmark KGQA datasets demonstrate that RoG achieves state-of-the-art performance on KG reasoning tasks and generates faithful and interpretable reasoning results $^{1}$ .

# 1 INTRODUCTION

Large language models (LLMs) have shown great performance in many NLP tasks (Brown et al., 2020; Bang et al., 2023). What's especially striking is their ability to handle complex tasks through reasoning (Wei et al., 2022; Huang & Chang, 2023). To further unleash LLMs' reasoning ability, the plan-and-solve paradigm (Wang et al., 2023c) has been proposed, in which LLMs are prompted to generate a plan and execute each reasoning step. In this way, LLMs decompose complex reasoning tasks into a series of sub-tasks and solve them step by step (Khot et al., 2022).

Despite their success, LLMs are still limited by the lack of knowledge and prone to hallucinations during reasoning, which can lead to errors in reasoning processes (Hong et al., 2023; Wang et al., 2023b). For example, as shown in Figure 1, LLMs do not have the latest knowledge and hallucinate an incorrect reasoning step: “has a daughter”. These issues largely diminish the performance and trustworthiness of LLMs in high-stakes scenarios, such as legal judgment and medical diagnosis.

To tackle the issues, knowledge graphs (KGs) have been incorporated to improve the reasoning ability of LLMs (Pan et al., 2024; Luo et al., 2023a). KGs capture abundant factual knowledge in a structured format, which provides a faithful knowledge source for reasoning. As a typical reasoning

![](images/4d937d9738304242a8c8693836a7b6fd6c9fcf42ceee88048a5740d808a1d6f7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Question\nWhat product did Apple release in 2023?"] --> B["Output\nSorry, I do not have knowledge after Sept. 2021.\nCould you provide some additional information?"]
    B --> C["Lack of Knowledge\nFactual Knowledge ↑\nTriple: (Iphone 15, released_at, 2023)"]
    C --> D["Question\nWho is the brother of Justin Bieber"]
    D --> E["Output\nJustin Bieber is the child of Jeremy Bieber, who has a daughter named Allie Bieber. Thus, the brother of Justin Bieber is Allie Bieber."]
    E --> F["Hallucination\nReasoning Guidance ↑\nRelation path: child_of → has_son"]
```
</details>

Figure 1: The issues of lack of knowledge and hallucination in LLMs reasoning and how they can be addressed by triples and relation paths from KGs.

task, knowledge graph question answering (KGQA) aims to obtain answers based on knowledge from KGs (Sun et al., 2019). Previous works that jointly use KGs and LLMs for KGQA reasoning can be broadly divided into two categories: 1) semantic parsing methods (Lan & Jiang, 2020; Ye et al., 2022), which use LLMs to convert questions into logical queries that are executed on KGs to obtain answers; and 2) retrieval-augmented methods (Li et al., 2023; Jiang et al., 2023), which retrieve triples from KGs as knowledge context and uses LLMs to obtain the final answers.

Although semantic parsing methods can generate more accurate and interpretable results by leveraging reasoning on KGs, the generated logical queries can often be non-executable and yield no answers, due to syntax and semantic limitations (Yu et al., 2022a). Retrieval-augmented methods are more flexible and exploit the ability of LLMs for reasoning. However, they only treat KGs as factual knowledge bases and overlook the importance of their structural information for reasoning (Jiang et al., 2022). For instance, as shown in Figure 1, a relation path, which is a sequence of relations, “child\_of→has\_son” can be used to obtain answers to the question “Who is the brother of Justin Bieber?”. Therefore, it is essential to enable LLMs to directly reason on KGs to achieve faithful and interpretable reasoning.

In this paper, we propose a novel method called reasoning on graphs (RoG) that synergizes LLMs with KGs to conduct faithful and interpretable reasoning. To alleviate the issues of hallucinations and lack of knowledge, we present a planning-retrieval-reasoning framework, where RoG first generates relation paths grounded by KGs as faithful plans via the planning module. These plans are then used to retrieve valid reasoning paths from KGs to conduct faithful reasoning by the retrieval-reasoning module. In this way, we not only retrieve the latest knowledge from KGs but also consider the guidance of KG structure for reasoning and explanations. Moreover, the planning module of RoG can be plug-and-play with different LLMs during inference to improve their performance. Based on this framework, RoG is optimized by two tasks: 1) planning optimization, where we distill knowledge from KGs into LLMs to generate faithful relation paths as plans; and 2) retrieval-reasoning optimization, where we enable LLMs to conduct faithful reasoning based on retrieved paths and generate interpretable results. We conduct extensive experiments on two benchmark KGQA datasets, and the results demonstrate that RoG achieves state-of-the-art performance on KG reasoning tasks and generates faithful and interpretable reasoning results.

# 2 RELATED WORK

LLM Reasoning Prompt. Many studies have been proposed to harness the reasoning ability of LLMs to handle complex tasks through prompting (Wei et al., 2022; Wang et al., 2022; Yao et al., 2023; Besta et al., 2023). Plan-and-solve (Wang et al., 2023c) prompts LLMs to generate a plan and conduct reasoning based on it. DecomP (He et al., 2021) prompts LLMs to decompose the reasoning task into a series of sub-tasks and solve them step by step. However, the problem of hallucinations and lack of knowledge affect the faithfulness of LLMs' reasoning. ReACT (Yao et al., 2022) treats LLMs as agents, which interact with the environment to get the latest knowledge for reasoning. To explore faithful reasoning, FAME (Hong et al., 2023) introduces the Monte-Carlo planning to generate faithful reasoning steps. RR (He et al., 2022) and KD-CoT Wang et al. (2023b) further retrieve relevant knowledge from KGs to produce faithful reasoning plans for LLMs.

Knowledge Graph Question Answering (KGQA). Conventional embedding-based methods represent the entities and relations in embedding space and design special model architectures (e.g., Key-Value memory networks, sequential models, and graph neural networks) to reason answers

![](images/cf4869774c175cede94ba7a82bb46510565531df8b9a493f21befeefb585d1f1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Joe Biden"] -->|born_in| B["Scranton"]
    A -->|graduate_from| C["University of Delaware"]
    A -->|profession| D["Till Biden"]
    A -->|property| E["President"]
    A -->|work_in| F["Washington D.C."]
    B -->|city_of| G["USA"]
    C -->|located_in| G
    D -->|member_of| H["Democratic Party"]
    E --> I["New York"]
    F --> J["Miami"]
    G --> K["Relation Paths Z"]
    G --> L["Reasoning Paths W_Z"]
    G --> M["② Retrieval"]
    G --> N["Prompt: Please generate helpful relation paths for answering the question."]
    G --> O["① Planning"]
    G --> P["③ Reasoning"]
    G --> Q["Prompt: Please answer the question based on the reasoning paths and explain why."]
    Q --> R["Answer: The answer is USA, because ...."]
    R --> S["Pθ(a|q, Z, G)"]
    S --> T["Reasoning Optimization"]
    T --> U["Large Language Models (LLMs)"]
    U --> V["Instruction Finetuning"]
    V --> W["Question: What is the nationality of Joe Biden?"]
    W --> X["Inference"]
    X --> Y["Inference to Instruction Finetuning"]
    Y --> Z["Planning Optimization"]
    Z --> AA["Pθ(z|q)"]
    AA --> AB["① Planning"]
    AB --> AC["Prompt: Please answer the reasoning paths and explain why."]
```
</details>

Figure 2: The overall framework of reasoning on graphs (RoG). 1) given a question, we first prompt LLMs to generate several relation paths that are grounded by KGs as plans. 2) Then, we retrieve reasoning paths from KGs using the plans. 3) Finally, we conduct faithful reasoning based on the retrieved reasoning paths and generate answers with interpretable explanations. The orange and red rectangles denote the entities mentioned in the question and answer, respectively.

(Miller et al., 2016; He et al., 2021; Yasunaga et al., 2021). To integrate LLMs for KGQA, retrieval-augmented methods aim to retrieve the relative facts from the KGs to improve the reasoning performance (Li et al., 2023; Karpukhin et al., 2020). Recently, UniKGQA (Jiang et al., 2022) which unifies the graph retrieval and reasoning process into a single model with LLMs, achieves STOA performance. Semantic parsing methods convert the question into a structural query (e.g., SPARQL) by LLMs, which can be executed by a query engine to reason the answers on KGs (Sun et al., 2020; Lan & Jiang, 2020). However, these methods heavily rely on the quality of generated queries. If the query is not executable, no answers will be generated. DECAF (Yu et al., 2022a) combines semantic parsing and LLMs reasoning to jointly generate answers, which also reach salient performance on KGQA tasks.

# 3 PRELIMINARY

Knowledge Graphs (KGs) contain abundant factual knowledge in the form of a set of triples: $\mathcal{G} = \{(e, r, e') | e, e' \in \mathcal{E}, r \in \mathcal{R}\}$ , where E and R denote the set of entities and relations, respectively.

Relation Paths are a sequence of relations: $z = \{r_{1}, r_{2}, \ldots, r_{l}\}$ , where $r_{i} \in R$ denotes the i-th relation in the path and l denotes the length of the path.

Reasoning Paths are the instances of a relation path $z$ in KGs: $w_{z} = e_{0} \xrightarrow{r_{1}} e_{1} \xrightarrow{r_{2}} \ldots \xrightarrow{r_{l}} e_{l}$ , where $e_{i} \in \mathcal{E}$ denotes the $i$ -th entity and $r_{i}$ denotes the $i$ -th relation in the relation path $z$ .

Example 1. Given a relation path: $z = \text{marry\_to} \rightarrow \text{father\_of}$ , a reasoning path instance could be: $w_{z} = \text{Alice} \xrightarrow{\text{marry\_to}} \text{Bob} \xrightarrow{\text{father\_of}} \text{Charlie}$ , which denotes “Alice” is married to “Bob” and “Bob” is the father of “Charlie”.

Knowledge Graph Question Answering (KGQA) is a typical reasoning task based on KGs. Given a natural language question q and a KG G, the task aims to design a function f to predict answers $a \in A_{q}$ based on knowledge from G, i.e., $a = f(q, \mathcal{G})$ . Following previous works (Sun et al., 2019; Jiang et al., 2022), we assume the entities $e_{q} \in T_{q}$ mentioned in q and answers $a \in A_{q}$ are labeled and linked to the corresponding entities in G, i.e., $T_{q}, A_{q} \subseteq E$ .

# 4 APPROACH

In this section, we introduce our method: reasoning on graphs (Rog), which contains two components: 1) a planning module that generates relation paths grounded by KGs as faithful plans; 2)

a retrieval-reasoning module that first retrieves valid reasoning paths from KGs according to the plans, then conducts faithful reasoning based on retrieved reasoning paths and generates answers with interpretable explanations. The overall framework of RoG is illustrated in Figure 2.

# 4.1 REASONING ON GRAPHS: PLANNING-RETRIEVAL-REASONING

Recently, many techniques have been explored to improve the reasoning ability of LLMs by planning, which first prompts LLMs to generate a reasoning plan and then conduct reasoning based on it (Wang et al., 2023c). However, LLMs are known for having hallucination issues, which are prone to generating incorrect plans and leading to wrong answers (Ji et al., 2023). To address this issue, we present a novel planning-retrieval-reasoning framework, which makes the reasoning plans grounded by KGs and then retrieves faithful reasoning paths for LLM reasoning.

Relation paths, which capture semantic relations between entities, have been utilized in many reasoning tasks on KGs (Wang et al., 2021; Xu et al., 2022). Moreover, compared to the dynamically updated entities, the relations in KGs are more stable (Wang et al., 2023a). By using relation paths, we can always retrieve the latest knowledge from KGs for reasoning. Therefore, relation paths can serve as faithful plans for reasoning the answer to KGQA task.

Example 2. Given a question “Who is the child of Alice”, we can generate a relation path as the plan: $z = \text{marry\_to} \rightarrow \text{father\_of}$ . This relation path expresses the plan: 1) find the person that “Alice” is married to; 2) find the child of that person. We can execute the plan (relation path) by retrieving a reasoning path from KGs as: $w_{z} = \text{Alice} \xrightarrow{\text{marry\_to}} \text{Bob} \xrightarrow{\text{father\_of}} \text{Charlie}$ . Finally, we can answer the question based on the reasoning path, which is “Charlie”.

By treating relation paths as plans, we can make sure the plans are grounded by KGs, which enables LLMs to conduct faithful and interpretable reasoning on graphs. In a nutshell, we formulate our RoG as an optimization problem that aims to maximize the probability of reasoning the answer from a knowledge graph G w.r.t the question q by generating relation paths z as the plan:

$$
P _ {\theta} (a | q, \mathcal {G}) = \sum_ {z \in \mathcal {Z}} P _ {\theta} (a | q, z, \mathcal {G}) P _ {\theta} (z | q), \tag {1}
$$

where $\theta$ denotes the parameters of LLMs, z denotes the relation paths (plans) generated by LLMs, and Z denotes the set of possible relation paths. The latter term $P_{\theta}(z|q)$ is the probability of generating a faithful relation path z grounded by KG, given the question q, which is realized by the planning module. The former term $P_{\theta}(a|q,z,\mathcal{G})$ is the probability of reasoning an answer a given the question q, relation path z, and KG G, which is computed by the retrieval-reasoning module.

# 4.2 OPTIMIZATION FRAMEWORK

Despite the advantage of generating relation paths as plans, the LLMs have zero knowledge of the relations contained in KGs. Therefore, LLMs cannot directly generate relation paths grounded by KGs as faithful plans. Moreover, LLMs might not understand the reasoning paths correctly and conduct reasoning based on them. To address these issues, we design two instruction tuning tasks: 1) planning optimization, which distills the knowledge from KGs into LLMs to generate faithful relation paths as plans; 2) retrieval-reasoning optimization, which enables LLMs to reason based on the retrieved reasoning paths.

The objective function in equation 1 can be optimized by maximizing the evidence lower bound (ELBO) (Jordan et al., 1999), which is formulated as

$$
\log P (a | q, \mathcal {G}) \geq \mathbb {E} _ {z \sim Q (z)} [ \log P _ {\theta} (a | q, z, \mathcal {G}) ] - D _ {\mathrm{KL}} (Q (z) \| P _ {\theta} (z | q)), \tag {2}
$$

where $Q(z)$ denotes the posterior distribution of faithful relation paths grounded by KGs. The latter term minimizes the KL divergence between the posterior and the prior, which encourages LLMs to generate faithful relation paths (planning optimization). The former term maximizes the expectation that retrieval-reasoning module generates correct answers based on the relation paths and KGs (retrieval-reasoning optimization).

Planning optimization. In planning optimization, we aim to distill the knowledge from KGs into LLMs to generate faithful relation paths as plans. This can be achieved by minimizing the KL divergence with the posterior distribution of faithful relation paths $Q(z)$ , which can be approximated by the valid relation paths in KGs.

Given a question $q$ and answer $a$ , we could find the path instances $w_{z}(e_{q}, e_{a}) = e_{q} \xrightarrow{r_{1}} e_{1} \xrightarrow{r_{2}} \ldots \xrightarrow{r_{l}} e_{a}$ connecting $e_{q}$ and $e_{a}$ in KGs. The corresponding relation path $z = \{r_{1}, r_{2}, \ldots, r_{l}\}$ can be considered valid and serve as a faithful plan for answering the question $q$ . The posterior distribution $Q(z)$ can be formally approximated as

$$
Q (z) \simeq Q (z | a, q, \mathcal {G}) = \left\{ \begin{array}{l} \frac {1}{| \mathcal {Z} |}, \exists w _ {z} (e _ {q}, e _ {a}) \in \mathcal {G}, \\ 0, e l s e, \end{array} \right. \tag {3}
$$

where we assume a uniform distribution over all valid relation paths $\mathcal{Z}$ , and $\exists w_{z}(e_{q},e_{a})\in \mathcal{G}$ denote the existence of a path instance connecting the question $e_q$ and answer $e_a$ entities in $\mathcal{G}$ . Therefore, the KL divergence can be calculated as

$$
\mathcal {L} _ {\text { plan }} = D _ {\mathrm{KL}} (Q (z) \| P _ {\theta} (z | q)) = D _ {\mathrm{KL}} (Q (z | a, q, \mathcal {G}) \| P _ {\theta} (z | q)) \simeq - \frac {1}{| \mathcal {Z} ^ {*} |} \sum_ {z \in \mathcal {Z} ^ {*}} \log P _ {\theta} (z | q), \tag {4}
$$

where we use the shortest paths $Z^{*} \subseteq Z$ between $e_{q}$ and $e_{a}$ in KGs as supervision signals (Zhang et al., 2022). The detailed derivation of can be found in Appendix A.1. By optimizing the equation 4, we maximize the probability of LLMs generating faithful relation paths through distilling the knowledge from KGs.

Retrieval-reasoning optimization. In retrieval-reasoning optimization, we aim to enable LLMs to conduct reasoning based on the retrieved reasoning paths. For the retrieval-reasoning module, we follow the FiD framework (Izacard & Grave, 2021; Singh et al., 2021), which allows reasoning on multiple retrieved reasoning paths $^{2}$ , formulated as

$$
P _ {\theta} (a | q, \mathcal {Z}, \mathcal {G}) = \prod_ {z \in \mathcal {Z}} P _ {\theta} (a | q, z, \mathcal {G}). \tag {5}
$$

By approximating the expectation with K sampled plans $\mathcal{Z}_K^* \subseteq Z^*$ , the objective function of reasoning optimization can be written as

$$
\mathcal {L} _ {\text { reason }} = \mathbb {E} _ {z \sim Q (z | a, q, \mathcal {G})} [ \log P _ {\theta} (a | q, z, \mathcal {G}) ] = \sum_ {z \in \mathcal {Z} _ {K} ^ {*}} \log P _ {\theta} (a | q, z, \mathcal {G}) = \log P _ {\theta} (a | q, \mathcal {Z} _ {K} ^ {*}, \mathcal {G}). \tag {6}
$$

This maximizes the probability of LLMs generating correct answers based on the retrieved reasoning paths.

The final objective function of RoG is the combination of the planning optimization and retrieval-reasoning optimization, which can be formulated as

$$
\mathcal {L} = \log \underbrace {P _ {\theta} (a | q , \mathcal {Z} _ {K} ^ {*} , \mathcal {G})} _ {\text { Retrieval - reasoning }} + \underbrace {\frac {1}{| \mathcal {Z} ^ {*} |} \sum_ {z \in Z ^ {*}} \log P _ {\theta} (z | q)} _ {\text { Planning }}. \tag {7}
$$

From equation 7, we can see that we adopt the same LLM for both planning and reasoning, which are jointly trained on two instruction-tuning tasks, i.e., (planning and retrieval-reasoning). We will discuss the implementation details of these two tasks in the following subsections.

# 4.3 PLANNING MODULE

The planning module aims to generate faithful relation paths as plans for answering the question. To utilize the instruction-following ability of LLMs (Wei et al., 2021), we design a simple instruction template that prompts LLMs to generate relation paths:

Please generate a valid relation path that can be helpful for answering the following question: <Question>

where <Question> indicates the question q. The question together with the instruction template is fed into LLMs to generate the relation paths, which are structurally formatted as a sentence:

$$
z = <   \text { PATH } > r _ {1} <   \text { SEP } > r _ {2} <   \text { SEP } > \dots <   \text { SEP } > r _ {l} <   / \text { PATH } >
$$

where <PATH>, <SEP>, </PATH> are special tokens indicating the start, separator, and end of the relation path, respectively $^{3}$ .

Therefore, the optimization of $\mathcal{L}_{\mathrm{plan}}$ can be achieved as

$$
\underset {\theta} {\arg \max} \frac {1}{| \mathcal {Z} ^ {*} |} \sum_ {z \in \mathcal {Z} ^ {*}} \log P _ {\theta} (z | q) = \frac {1}{| \mathcal {Z} ^ {*} |} \sum_ {z \in \mathcal {Z} ^ {*}} \log \prod_ {i = 1} ^ {| z |} P _ {\theta} (r _ {i} | r _ {<   i}, q), \tag {8}
$$

where $P_{\theta}(z|q)$ denotes the prior distribution of generating faithful relation path z, and $P_{\theta}(r_{i}|r_{<i},q)$ denotes the probability of each token in z generated by LLMs.

# 4.4 RETRIEVAL-REASONING MODULE

Retrieval. Given a question q and a relation path as plan z, the retrieval module aims to retrieve the reasoning paths $w_{z}$ from KG G. The retrieval process can be conducted by finding paths in G that start from the question entities $e_{q}$ and follow the relation paths z, formulated as

$$
\mathcal {W} _ {z} = \left\{w _ {z} (e _ {q}, e _ {*}) | w _ {z} (e _ {q}, e _ {*}) = e _ {q} \xrightarrow {r _ {1}} e _ {1} \xrightarrow {r _ {2}} \dots \xrightarrow {r _ {l}} e _ {a *}, w _ {z} (e _ {q}, e _ {*}) \in \mathcal {G} \right\}. \tag {9}
$$

We adopt a constrained breadth-first search to retrieve the reasoning paths $w_{z}$ from KGs. In experiments, all retrieved paths are used for reasoning. The detailed retrieval algorithm can be found in Appendix A.3.

Despite we can utilize the retrieved reasoning paths and directly get the answers with majority voting. The retrieved reasoning paths could be noisy and irrelevant to the questions, which leads to incorrect answers (He et al., 2021; Zhang et al., 2022). Therefore, we propose a reasoning module to explore the ability of LLMs to identify the important reasoning paths and answer the questions based on them.

Reasoning. The reasoning module takes the question q and a set of reasoning paths $W_{z}$ to generate answers a. Similarly, we design a reasoning instruction prompt to guide LLMs to conduct reasoning based on the retrieved reasoning paths $W_{z}$ . The $W_{z}$ are also formatted as a series of structural sentences. The detailed prompt can be found in Appendix A.10.

The optimization of $L_{reason}$ can be written as

$$
\underset {\theta} {\arg \max} \log P _ {\theta} (a | q, \mathcal {Z} _ {K} ^ {*}, \mathcal {G}) = \log \sum_ {z \in \mathcal {Z} _ {K} ^ {*}} \sum_ {w _ {z} \in \mathcal {W} _ {z}} \prod_ {i = 1} ^ {| a |} P _ {\theta} (t _ {i} | t _ {<   i}, q, w _ {z}), \tag {10}
$$

where $P_{\theta}(a|q,\mathcal{Z}_{K},\mathcal{G})$ denotes probability of reasoning the correct answer a based on K relation paths $Z_{K}$ , and $t_{*}$ denote the tokens of answers a.

# 5 EXPERIMENT

In our experiments, we aim to answer the following research questions: RQ1: Can RoG achieve state-of-the-art performance on the KGQA tasks? RQ2: Can the planning module of RoG be integrated with other LLMs to improve their performance? RQ3: Can RoG conduct faithful reasoning and generate interpretable reasoning results?

# 5.1 EXPERIMENT SETTINGS

Datasets. We evaluate the reasoning ability of RoG on two benchmark KGQA datasets: WebQuestionSP (WebQSP) (Yih et al., 2016) and Complex WebQuestions (CWQ) (Talmor & Berant, 2018), which contain up to 4-hop questions. Freebase (Bollacker et al., 2008) is the background knowledge graph for both datasets, which contains around 88 million entities, 20 thousand relations, and 126 million triples. The details of the datasets are described in Appendix A.4.

Table 1: Performance comparison with different baselines on the two KGQA datasets. 

<table><tr><td rowspan="2">Type</td><td rowspan="2">Methods</td><td colspan="2">WebQSP</td><td colspan="2">CWQ</td></tr><tr><td>Hits@1</td><td>F1</td><td>Hits@1</td><td>F1</td></tr><tr><td rowspan="5">Embedding</td><td>KV-Mem (Miller et al., 2016)</td><td>46.7</td><td>34.5</td><td>18.4</td><td>15.7</td></tr><tr><td>EmbedKGQA (Saxena et al., 2020)</td><td>66.6</td><td>-</td><td>45.9</td><td>-</td></tr><tr><td>NSM (He et al., 2021)</td><td>68.7</td><td>62.8</td><td>47.6</td><td>42.4</td></tr><tr><td>TransferNet (Shi et al., 2021)</td><td>71.4</td><td>-</td><td>48.6</td><td>-</td></tr><tr><td>KGT5 Saxena et al. (2022)</td><td>56.1</td><td>-</td><td>36.5</td><td>-</td></tr><tr><td rowspan="4">Retrieval</td><td>GraftNet (Sun et al., 2018)</td><td>66.4</td><td>60.4</td><td>36.8</td><td>32.7</td></tr><tr><td>PullNet (Sun et al., 2019)</td><td>68.1</td><td>-</td><td>45.9</td><td>-</td></tr><tr><td>SR+NSM (Zhang et al., 2022)</td><td>68.9</td><td>64.1</td><td>50.2</td><td>47.1</td></tr><tr><td>SR+NSM+E2E (Zhang et al., 2022)</td><td>69.5</td><td>64.1</td><td>49.3</td><td>46.3</td></tr><tr><td rowspan="4">Semantic Parsing</td><td>SPARQL (Sun et al., 2020)</td><td>-</td><td>-</td><td>31.6</td><td>-</td></tr><tr><td>QGG (Lan &amp; Jiang, 2020)</td><td>73.0</td><td>73.8</td><td>36.9</td><td>37.4</td></tr><tr><td>ArcaneQA (Gu &amp; Su, 2022)</td><td>-</td><td>75.3</td><td>-</td><td>-</td></tr><tr><td>RnG-KBQA (Ye et al., 2022)</td><td>-</td><td>76.2</td><td>-</td><td>-</td></tr><tr><td rowspan="5">LLMs</td><td>Flan-T5-xl (Chung et al., 2022)</td><td>31.0</td><td>-</td><td>14.7</td><td>-</td></tr><tr><td>Alpaca-7B (Taori et al., 2023)</td><td>51.8</td><td>-</td><td>27.4</td><td>-</td></tr><tr><td>LLaMA2-Chat-7B (Touvron et al., 2023)</td><td>64.4</td><td>-</td><td>34.6</td><td>-</td></tr><tr><td>ChatGPT</td><td>66.8</td><td>-</td><td>39.9</td><td>-</td></tr><tr><td>ChatGPT+CoT</td><td>75.6</td><td>-</td><td>48.9</td><td>-</td></tr><tr><td rowspan="4">LLMs+KGs</td><td>KD-CoT (Wang et al., 2023b)</td><td>68.6</td><td>52.5</td><td>55.7</td><td>-</td></tr><tr><td>UniKGQA (Jiang et al., 2022)</td><td>77.2</td><td>72.2</td><td>51.2</td><td>49.1</td></tr><tr><td>DECAF (DPR+FiD-3B) (Yu et al., 2022a)</td><td>82.1</td><td>78.8</td><td>-</td><td>-</td></tr><tr><td>RoG</td><td>85.7</td><td>70.8</td><td>62.6</td><td>56.2</td></tr></table>

Baselines. We compare RoG with 21 baselines grouping into 5 categories: 1) Embedding-based methods, 2) Retrieval-augmented methods, 3) Semantic parsing methods, 4) LLMs, and 5) LLMs+KGs methods. The details of each baseline are described in Appendix A.5.

Evaluation Metrics. Following previous works, we use Hits@1 and F1 as the evaluation metrics. Hits@1 measures the proportion of questions whose top-1 predicted answer is correct. Since a question may correspond to multiple answers, F1 considers the coverage of all answers, which balances the precision and recall of the predicted answers.

Implementations. For RoG, we use LLaMA2-Chat-7B (Touvron et al., 2023) as the LLM backbone, which is instruction finetuned on the training split of WebQSP and CWQ as well as Freebase for 3 epochs. We generate the top-3 relation paths using beam-search for each question. Since UniKGQA (Jiang et al., 2022) and DECAF (Yu et al., 2022a) are state-of-the-art methods, we directly refer their results and those of the other baselines reported in their papers for comparisons. For LLMs, we use zero-shot prompting to conduct KGQA. The detailed settings are described in Appendix A.6.

# 5.2 RQ1: KGQA PERFORMANCE COMPARISON

Main Results. In this section, we compare RoG with other baselines on KGQA tasks. The results are shown in Table 1. Our method achieves the best performance on both datasets across most metrics. Specifically, compared to the SOTA method DECAF (Yu et al., 2022a) on WebQSP, our method improves Hits@1 by 4.4%. On the CWQ dataset, which is more challenging due to multi-hop questions, our method improves both Hits@1 and F1 by 22.3% and 14.4% against the SOTA model UniKGQA (Jiang et al., 2022). These results demonstrate the superior reasoning ability of our method in KGQA.

Among other methods, retrieval-augmented approaches outperform conventional embedding-based methods by retrieving relevant subgraphs from KGs, which reduces reasoning complexity. Furthermore, SR+NSM and SR+NSM+E2E adopt relation paths-based retrieval which achieves better performance, highlighting the importance of relation paths. Semantic parsing methods perform better than retrieval methods on WebQSP but worse on CWQ due to the complexity of generating logical queries for complex questions in CWQ. Although LLMs-based methods achieve comparable performance, they are limited by hallucinations and lack of knowledge as shown in Section 5.4. The LLMs+KGs methods achieve the second-best performance, which demonstrates the effectiveness of unifying KGs and LLMs for reasoning.

Ablation Study. We conduct an ablation study to analyze the effectiveness of the planning module and reasoning module in our method (Rog). We compare four variants: 1) w/o planning, where we remove the planning module and perform reasoning without retrieved reasoning paths; 2) w/o reasoning, where we remove the reasoning module and use all answers from retrieved reasoning

![](images/85fee25064ac0c7c4c13d5f68b2d28cbf7b893144764afa1317739946e70e1a3.jpg)  
Figure 3: Faithfulness of top-K generated relation paths. Green bars denote the average number of retrieved reasoning paths, solid lines denote the answer coverage of the retrieved paths, and dashed line denote the answer coverage of the reasoning module based on the retrieved reasoning paths.

paths as results; 3) w/ random plans, where we randomly retrieve reasoning paths from KGs and feed them into the reasoning module; 4) w/ vote reasoning, where we adopt the majority voting to select top-5 answers from retrieved reasoning paths. The results are shown in Table 2.

From the results, it is evident that without a planning module, our method degenerates to conventional LLMs that solely rely on questions as input, suffering from the lack of knowledge issue. Although removing the reasoning module leads to high recall due to an increased number of answers, precision drops significantly because of noise in retrieved paths. This demonstrates the effectiveness of the reasoning module in identifying important reasoning paths and filtering out noise. Furthermore, using random plans achieves worse performance than removing the planning module, highlighting the importance of a planning module who generates faithful reasoning plans. Using a simple majority vote reasoning can improve the results which also demonstrate the necessity of reasoning module.

Table 2: Ablation studies of RoG. 

<table><tr><td rowspan="2">Method</td><td colspan="3">WebQSP</td><td colspan="3">CWQ</td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>RoG</td><td>74.77</td><td>75.84</td><td>70.81</td><td>57.69</td><td>58.19</td><td>56.17</td></tr><tr><td>RoG w/o planning</td><td>57.26</td><td>50.16</td><td>49.69</td><td>35.35</td><td>34.77</td><td>33.76</td></tr><tr><td>RoG w/o reasoning</td><td>46.90</td><td>79.85</td><td>49.56</td><td>18.88</td><td>67.89</td><td>22.26</td></tr><tr><td>RoG w/ random plans</td><td>38.66</td><td>38.31</td><td>35.24</td><td>38.99</td><td>39.29</td><td>37.64</td></tr><tr><td>RoG w/ vote reasoning</td><td>54.80</td><td>60.44</td><td>47.96</td><td>22.92</td><td>47.98</td><td>26.52</td></tr></table>

# 5.3 RQ2: PLUG-AND-PLAY ROG PLANNING MODULE

In this section, we evaluate the effectiveness of integrating the planning module of RoG with different LLMs during inference to improve their performance. Specifically, we first adopt the planning module of RoG to generate relation paths and feed the retrieved reasoning paths as context into different LLMs for reasoning. The results are presented in Table 3. To account for the fact that it is difficult to extract the number of answers from LLM's output. We only report the Hits@1 and Recall metrics.

Table 3: Effects of integrating the planning module of RoG with different LLMs for reasoning. 

<table><tr><td rowspan="2">Methods</td><td colspan="2">WebQSP</td><td colspan="2">CWQ</td></tr><tr><td>Hits@1</td><td>Recall</td><td>Hits@1</td><td>Recall</td></tr><tr><td>ChatGPT</td><td>66.77</td><td>49.27</td><td>39.90</td><td>35.07</td></tr><tr><td>ChatGPT + RoG Planning</td><td>81.51</td><td>71.60</td><td>52.68</td><td>48.51</td></tr><tr><td>Alpaca-7B</td><td>51.78</td><td>33.65</td><td>27.44</td><td>23.62</td></tr><tr><td>Alpaca-7B + RoG Planning</td><td>56.16</td><td>74.20</td><td>44.04</td><td>38.46</td></tr><tr><td>LLaMA2-Chat-7B</td><td>64.37</td><td>44.61</td><td>34.60</td><td>29.91</td></tr><tr><td>LLaMA2-Chat-7B + RoG Planning</td><td>74.20</td><td>56.16</td><td>56.41</td><td>51.99</td></tr><tr><td>Flan-T5-xl</td><td>30.95</td><td>17.08</td><td>14.69</td><td>12.25</td></tr><tr><td>Flan-T5-xl + RoG Planning</td><td>67.87</td><td>44.93</td><td>37.81</td><td>32.57</td></tr></table>

From the results, we can notice that the performance of all LLMs is substantially improved by integrating the planning module of RoG. Specifically, the Hits@1 of ChatGPT, Alpaca, LLaMA2, and Flan-T5 are improved by 8.5%, 15.3%, and 119.3%, respectively. This demonstrates that the planning module of RoG can be seamlessly integrated with other LLMs to improve their performance without retraining.

# 5.4 RQ3: FAITHFUL REASONING AND INTERPRETABLE RESULTS

Quantitative Results. To evaluate the faithfulness of the relation paths, we illustrate the quantitative results in Figure 3. In experiments, we sweep the number of top-K relation paths generated by RoG. From results, we can see that the number of retrieved reasoning paths increases with K, which also leads to more answers to be covered (recall). This demonstrates the faithfulness of the

Table 4: Example of lack of knowledge for ChatGPT-CoT and interpretable predictions of RoG. 

<table><tr><td>Question</td><td>What type of government is used in the country with Northern District?</td></tr><tr><td>Answer</td><td>Parliamentary system</td></tr><tr><td>ChatGPT + CoT</td><td>To determine what type of government is used in a country with a Northern District, we need more specific information. The term &quot;Northern District&quot; does not specify the name of any particular country or provide enough context to identify its government type. Could you please provide additional details or clarify the specific country you are referring to?</td></tr><tr><td>Reasoning Paths</td><td>Northern District → location.administrative_division.first_level_division_of → Israel → government.form_of_government.countries → Parliamentary system</td></tr><tr><td>RoG</td><td>The type of government used in the country with the Northern District is a Parliamentary system.Explanations:1. Northern District → location.administrative_division.first_level_division_of → Israel: The Northern District is a part of Israel.2. Israel → government.form_of_government.countries → Parliamentary system: Israel has a Parliamentary system of government.So, by following the reasoning paths provided, we can conclude that the country with the Northern District, which is Israel, uses a Parliamentary system of government.</td></tr></table>

Table 5: Example of the hallucination of ChatGPT-CoT and faithful reasoning of RoG. 

<table><tr><td>Question</td><td>Where is the home stadium of the team who won the 1946 World Series championship?</td></tr><tr><td>Answer</td><td>Busch Stadium</td></tr><tr><td>ChatGPT + CoT</td><td>Step 1: Identify the team that won the 1946 World Series championship. The team that won the 1946 World Series championship was the St. Louis Cardinals.Step 2: Determine the home stadium of the St. Louis Cardinals during that time. The St. Louis Cardinals’ home stadium during the 1946 season wasSportsman’s Park.Therefore, the possible answer to the question ”Where is the home stadium of the team who won the 1946 World Series championship?” is: -Sportsman’s Park.</td></tr><tr><td>Reasoning Paths</td><td>1946 World Series → sports.sports_team.championships → St. Louis Cardinals → sports.sports_team.arena_stadium →Busch Stadium1946 World Series → sports.sports_team.championships → St. Louis Cardinals → sports.sports_team.arena_stadium →Roger Dean Stadium</td></tr><tr><td>RoG</td><td>Based on the reasoning paths provided, the home stadium of the team who won the 1946 World Series championship isBusch Stadium.This conclusion is based on the following path:1. 1946 World Series → sports.sports_team.championships → St. Louis Cardinals → sports.sports_team.arena_stadium → Busch StadiumThe reasoning path leads us to the St. Louis Cardinals as the team that won the 1946 World Series, and Busch Stadium is the stadium associated with the St. Louis Cardinals. Therefore,Busch Stadiumis the home stadium of the team that won the 1946 World Series championship.</td></tr></table>

relation paths by retrieving the answers. However, more retrieved reasoning paths also lead to more noise and retrieval time (shown in Appendix A.7.4), which decreases the precision and makes little contribution to the final results (reasoning-f1). Therefore, we set K = 3 in experiments.

Case studies. We also illustrate two case studies in Table 4 and Table 5. In Table 4, we can find that ChatGPT+CoT suffers from the lack of knowledge issue and cannot answer the question. On the contrary, RoG can generate faithful relation paths and retrieve valid reasoning paths from KGs for reasoning. Besides, RoG can provide interpretable explanations based on the reasoning paths. In Table 5, we can see that ChatGPT+CoT suffers from hallucinations and generates incorrect answers. In contrast, although the retrieved reasoning paths contain noises, the reasoning module can identify the correct reasoning paths and conduct faithful reasoning. These results demonstrate the effectiveness of RoG in conducting faithful reasoning and generating interpretable results. More cases can be found in Appendices A.8 and A.9.

# 6 CONCLUSION

In this paper, we propose a novel method called reasoning on graphs (RoG) that synergizes LLMs with KGs to conduct faithful and interpretable reasoning. To alleviate the issues of hallucinations and lack of knowledge, we present a planning-retrieval-reasoning framework, which allows LLMs to access the latest knowledge while reasoning based on faithful plans on graphs. RoG not only enhances the reasoning capability of LLMs by distilling knowledge from KGs through training but also enables seamless integration with any LLMs during inference. Extensive experiments on two benchmark KGQA datasets demonstrate the superiority of RoG in reasoning ability and interpretability.

# ACKNOWLEDGMENTS

This material is based on research partially sponsored by the DARPA Assured Neuro Symbolic Learning and Reasoning (ANSR) program under award number FA8750-23-2-1016 and the DARPA Knowledge Management at Scale and Speed (KMASS) program under award number HR00112220047. This research is partly supported by the ARC Future Fellowship FT190100039 to G.H. S. Pan was supported in part by the Australian Research Council (ARC) under grants FT210100097 and DP240101547, and the CSIRO – National Science Foundation (US) AI Research Collaboration Program.

# ETHICS STATEMENT

Our work is solely dedicated to tackling scientific problems and does not involve any human subjects, animals, or environmentally sensitive materials. Consequently, we do not anticipate any potential ethical risks or conflicts of interest in our study. Our research strives to adhere to the highest standards of scientific integrity and ethical conduct throughout the entire process, guaranteeing the validity and reliability of our findings.

# REPRODUCIBILITY STATEMENT

Our model has been meticulously formalized within the main body of the text to ensure clarity and facilitate comprehensive understanding. Additionally, we provide detailed discussions of implementation in Appendices A.4 to A.6, encompassing dataset information, baselines, experimental settings, and model configurations. The code and pre-trained model weights have been publicized.

# REFERENCES

Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. arXiv preprint arXiv:2302.04023, 2023.   
Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687, 2023.   
Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a collaboratively created graph database for structuring human knowledge. In Proceedings of the 2008 ACM SIGMOD international conference on Management of data, pp. 1247–1250, 2008.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.   
Antonia Creswell and Murray Shanahan. Faithful reasoning using large language models. arXiv preprint arXiv:2208.14271, 2022.   
Yu Gu and Yu Su. Arcaneqa: Dynamic program induction and contextualized encoding for knowledge base question answering. In Proceedings of the 29th International Conference on Computational Linguistics, pp. 1718–1731, 2022.   
Gaole He, Yunshi Lan, Jing Jiang, Wayne Xin Zhao, and Ji-Rong Wen. Improving multi-hop knowledge base question answering by learning intermediate supervision signals. In Proceedings of the 14th ACM international conference on web search and data mining, pp. 553–561, 2021.

Hangfeng He, Hongming Zhang, and Dan Roth. Rethinking with retrieval: Faithful large language model inference. arXiv preprint arXiv:2301.00303, 2022.   
Ruixin Hong, Hongming Zhang, Hong Zhao, Dong Yu, and Changshui Zhang. Faithful question answering with monte-carlo planning. ACL2023, 2023.   
Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey. ACL 2023, 2023.   
Gautier Izacard and Édouard Grave. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 874–880, 2021.   
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023.   
Jinhao Jiang, Kun Zhou, Xin Zhao, and Ji-Rong Wen. Unikgqa: Unified retrieval and reasoning for solving multi-hop question answering over knowledge graph. In The Eleventh International Conference on Learning Representations, 2022.   
Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yaliang Li, and Ji-Rong Wen. Reasoninglm: Enabling structural subgraph reasoning in pre-trained language models for question answering over knowledge graph. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3721–3735, 2023.   
Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yang Song, Chen Zhu, Hengshu Zhu, and Ji-Rong Wen. Kg-agent: An efficient autonomous agent framework for complex reasoning over knowledge graph. arXiv preprint arXiv:2402.11163, 2024.   
Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, and Qingsong Wen. Time-llm: Time series forecasting by reprogramming large language models. In International Conference on Learning Representations, 2024a.   
Ming Jin, Yifan Zhang, Wei Chen, Kexin Zhang, Yuxuan Liang, Bin Yang, Jindong Wang, Shirui Pan, and Qingsong Wen. Position paper: What can large language models tell us about time series analysis. arXiv preprint arXiv:2402.02713, 2024b.   
Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. An introduction to variational methods for graphical models. Machine learning, 37:183–233, 1999.   
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769–6781, 2020.   
Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. In The Eleventh International Conference on Learning Representations, 2022.   
Yunshi Lan and Jing Jiang. Query graph generation for answering multi-hop complex questions from knowledge bases. Association for Computational Linguistics, 2020.   
Shiyang Li, Yifan Gao, Haoming Jiang, Qingyu Yin, Zheng Li, Xifeng Yan, Chao Zhang, and Bing Yin. Graph reasoning for question answering with triplet retrieval. In Findings of the Association for Computational Linguistics: ACL 2023, 2023.   
Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu, and Fuchun Sun. Reasoning over different types of knowledge graphs: Static, temporal and multi-modal. arXiv preprint arXiv:2212.05767, 2022.

Ke Liang, Yue Liu, Sihang Zhou, Wenxuan Tu, Yi Wen, Xihong Yang, Xiangjun Dong, and Xinwang Liu. Knowledge graph contrastive learning based on relation-symmetrical structure. IEEE Transactions on Knowledge and Data Engineering, pp. 1–12, 2023. doi: 10.1109/TKDE.2023.3282989.   
Yixin Liu, Kaize Ding, Qinghua Lu, Fuyi Li, Leo Yu Zhang, and Shirui Pan. Towards self-interpretable graph-level anomaly detection. Advances in Neural Information Processing Systems, 36, 2023a.   
Yixin Liu, Kaize Ding, Jianling Wang, Vincent Lee, Huan Liu, and Shirui Pan. Learning strong graph neural networks with weak information. In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2023b.   
Linhao Luo, Jiaxin Ju, Bo Xiong, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Chatrule: Mining logical rules with large language models for knowledge graph reasoning. arXiv preprint arXiv:2309.01538, 2023a.   
Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Normalizing flow-based neural process for few-shot knowledge graph completion. In The 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2023b.   
Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston. Key-value memory networks for directly reading documents. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1400–1409, 2016.   
Shirui Pan, Yizhen Zheng, and Yixin Liu. Integrating graphs with large language models: Methods and prospects. arXiv preprint arXiv:2310.05499, 2023.   
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering (TKDE), 2024.   
Apoorv Saxena, Aditay Tripathi, and Partha Talukdar. Improving multi-hop question answering over knowledge graphs using knowledge base embeddings. In Proceedings of the 58th annual meeting of the association for computational linguistics, pp. 4498–4507, 2020.   
Apoorv Saxena, Adrian Kochsiek, and Rainer Gemulla. Sequence-to-sequence knowledge graph completion and question answering. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2814–2828, 2022.   
Jiaxin Shi, Shulin Cao, Lei Hou, Juanzi Li, and Hanwang Zhang. Transfernet: An effective and transparent framework for multi-hop question answering over relation graph. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 4149–4158, 2021.   
Devendra Singh, Siva Reddy, Will Hamilton, Chris Dyer, and Dani Yogatama. End-to-end training of multi-document reader and retriever for open-domain question answering. Advances in Neural Information Processing Systems, 34:25968–25981, 2021.   
Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhutdinov, and William Cohen. Open domain question answering using early fusion of knowledge bases and text. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 4231–4242, 2018.   
Haitian Sun, Tania Bedrax-Weiss, and William Cohen. Pullnet: Open domain question answering with iterative retrieval on knowledge bases and text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2380–2390, 2019.   
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=nnV01PvbTv.

Yawei Sun, Lingling Zhang, Gong Cheng, and Yuzhong Qu. Sparqa: skeleton-based semantic parsing for complex questions over knowledge bases. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 8952–8959, 2020.   
Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. Entailer: Answering questions with faithful and truthful chains of reasoning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 2078–2093, 2022.   
Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex questions. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 641–651, 2018.   
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Stanford alpaca: an instruction-following llama model. https://github.com/tatsu-lab/stanford\_alpaca, 2023.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
Hongwei Wang, Hongyu Ren, and Jure Leskovec. Relational message passing for knowledge graph completion. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 1697–1707, 2021.   
Jiapu Wang, Boyue Wang, Meikang Qiu, Shirui Pan, Bo Xiong, Heng Liu, Linhao Luo, Tengfei Liu, Yongli Hu, Baocai Yin, et al. A survey on temporal knowledge graph completion: Taxonomy, progress, and prospects. arXiv preprint arXiv:2308.02457, 2023a.   
Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong, and Zhang Xiong. Knowledge-driven cot: Exploring faithful reasoning in llms for knowledge-intensive question answering. arXiv preprint arXiv:2308.13259, 2023b.   
Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. arXiv preprint arXiv:2305.04091, 2023c.   
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2022.   
Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2021.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.   
Xiaohan Xu, Peng Zhang, Yongquan He, Chengpeng Chao, and Chaoyang Yan. Subgraph neighboring relations infomax for inductive link prediction on knowledge graphs. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, 2022.   
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2022.   
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.

Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn: Reasoning with language models and knowledge graphs for question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 535–546, 2021.   
Xi Ye, Semih Yavuz, Kazuma Hashimoto, Yingbo Zhou, and Caiming Xiong. Rng-kbqa: Generation augmented iterative ranking for knowledge base question answering. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6032–6043, 2022.   
Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. The value of semantic parse labeling for knowledge base question answering. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 201–206, 2016.   
Donghan Yu, Sheng Zhang, Patrick Ng, Henghui Zhu, Alexander Hanbo Li, Jun Wang, Yiqun Hu, William Yang Wang, Zhiguo Wang, and Bing Xiang. Decaf: Joint decoding of answers and logical forms for question answering over knowledge bases. In The Eleventh International Conference on Learning Representations, 2022a.   
Donghan Yu, Chenguang Zhu, Yuwei Fang, Wenhao Yu, Shuohang Wang, Yichong Xu, Xiang Ren, Yiming Yang, and Michael Zeng. Kg-fid: Infusing knowledge graph in fusion-in-decoder for open-domain question answering. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4961–4974, 2022b.   
Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, and Hong Chen. Subgraph retrieval enhanced model for multi-hop knowledge base question answering. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5773–5784, 2022.   
Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D Manning, and Jure Leskovec. Greaselm: Graph reasoning enhanced language models. In International conference on learning representations, 2021.   
Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander Smola, and Le Song. Variational reasoning for question answering with knowledge graph. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.   
Zicheng Zhao, Linhao Luo, Shirui Pan, Quoc Viet Hung Nguyen, and Chen Gong. Towards few-shot inductive link prediction on knowledge graphs: A relational anonymous walk-guided neural process approach. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 515–532, 2023.   
Xin Zheng, Miao Zhang, Chunyang Chen, Chaojie Li, Chuan Zhou, and Shirui Pan. Multi-relational graph neural architecture search with fine-grained message passing. In 2022 IEEE International Conference on Data Mining (ICDM), pp. 783–792. IEEE, 2022a.   
Xin Zheng, Yixin Liu, Zhifeng Bao, Meng Fang, Xia Hu, Alan Wee-Chung Liew, and Shirui Pan. Towards data-centric graph machine learning: Review and outlook. arXiv preprint arXiv:2309.10979, 2023a.   
Xin Zheng, Miao Zhang, Chunyang Chen, Soheila Molaei, Chuan Zhou, and Shirui Pan. Gnnevaluator: Evaluating gnn performance on unseen graphs without labels. In Advances in Neural Information Processing Systems (NeurIPS), 2023b.   
Xin Zheng, Miao Zhang, Chunyang Chen, Quoc Viet Hung Nguyen, Xingquan Zhu, and Shirui Pan. Structure-free graph condensation: From large-scale graphs to condensed graph-free data. In Advances in Neural Information Processing Systems (NeurIPS), 2023c.   
Yizhen Zheng, Shirui Pan, Vincent Lee, Yu Zheng, and Philip S Yu. Rethinking and scaling up graph contrastive learning: An extremely efficient approach with group discrimination. Advances in Neural Information Processing Systems, 35:10809–10820, 2022b.

Yizhen Zheng, Huan Yee Koh, Jiaxin Ju, Anh TN Nguyen, Lauren T May, Geoffrey I Webb, and Shirui Pan. Large language models for scientific synthesis, inference and explanation. arXiv preprint arXiv:2310.07984, 2023d.

Yizhen Zheng, He Zhang, Vincent Lee, Yu Zheng, Xiao Wang, and Shirui Pan. Finding the missing-half: Graph complementary learning for homophily-prone and heterophily-prone graphs. International Conference on Machine Learning, 2023e.

# A APPENDIX

# A.1 DETAILED DERIVATION OF THE PLANNING MODULE

In this section, we provide a detailed derivation of the planning module. By approximating the $Q(z)$ with $Q(z|a,q,\mathcal{G})$ , the KL divergence can be calculated as

$$
\begin{array}{l} \mathcal {L} _ {\text { plan }} = D _ {\mathrm{KL}} (Q (z) \| P _ {\theta} (z | q)) = D _ {\mathrm{KL}} (Q (z | a, q, \mathcal {G}) \| P _ {\theta} (z | q)), \\ = \mathbb {E} _ {z \sim Q (z | a, q, \mathcal {G})} [ \log Q (z | a, q, \mathcal {G}) - \log P _ {\theta} (z | q) ], \tag {11} \\ = - \mathbb {E} _ {z \sim Q (z | a, q, \mathcal {G})} \log P _ {\theta} (z | q) + \text { CONST }, \\ \end{array}
$$

where the expectations cannot be computed exactly because of the large number of valid relation paths $\mathcal{Z}$ , so we approximate it by using the shortest paths $z \in \mathcal{Z}^* \subset \mathcal{Z}$ between $e_q$ and $e_a$ in KGs (Zhang et al., 2022). This can be formulated as

$$
\mathcal {L} _ {\text { plan }} = - \sum_ {z \in \mathcal {Z} ^ {*}} Q (z | a, q, \mathcal {G}) \log P _ {\theta} (z | q) + \text { CONST. } \tag {12}
$$

Based on equation 3, by assuming a uniform distribution over the set of shortest paths $\mathcal{Z}^*$ , we can rewrite the equation 12 as

$$
\mathcal {L} _ {\text { plan }} = - \frac {1}{| \mathcal {Z} ^ {*} |} \sum_ {z \in \mathcal {Z} ^ {*}} \log P _ {\theta} (z | q) + \text { CONST }, \tag {13}
$$

where the CONST is omitted in the final optimization since it makes no contributions to the loss.

# A.2 DETAILED RELATED WORK

# A.2.1 LLM REASONING PROMPT

Many studies have been proposed to harness the reasoning ability of LLMs to handle complex tasks through prompting (Wei et al., 2022; Wang et al., 2022; Yao et al., 2023; Besta et al., 2023; Pan et al., 2023; Zheng et al., 2023d; Jin et al., 2024a;b). Chain-of-Thought (Wei et al., 2022) enables LLMs to generate a reasoning chain that could be helpful to reasoning. Tree of thoughts (Yao et al., 2023) expands the reasoning chain to a tree structure to explore more reasoning paths. Graph of thoughts (Besta et al., 2023) further models the reasoning chain as a graph with an aggregation operation to synergize the reasoning paths. Plan-and-solve (Wang et al., 2023c) prompts LLMs to generate a plan and execute based on it. DecomP (He et al., 2021) prompts LLMs to decompose the reasoning task into a series of sub-tasks and solve them step by step. However, the problem of hallucinations and lack of knowledge affect the faithfulness of the reasoning. ReACT (Yao et al., 2022) treats LLMs as agents, which interact with the environment to get the latest knowledge for reasoning. To explore faithful reasoning, Entailer (Tafjord et al., 2022) introduces a verifier to validate the reasoning steps generated by LLMs. Creswell & Shanahan (2022) present a framework including two LLMs that are used for selecting and generating reasoning steps, respectively. FAME (Hong et al., 2023) introduces the Monte-Carlo planning to generate faithful reasoning steps. RR (He et al., 2022) and KD-CoT Wang et al. (2023b) aim to retrieve relevant knowledge from KGs to produce faithful reasoning plans for LLMs. Think-on-Graph (Sun et al., 2024) and KG-Agent (Jiang et al., 2024) treat LLMs as agents to interact with KGs via prompting to get the latest knowledge for reasoning.

# A.2.2 KNOWLEDGE GRAPH QUESTION ANSWERING

Knowledge graphs contain abundant factual knowledge in a structured format, which has attracted great attention from researchers (Zheng et al., 2022a; 2023a; Pan et al., 2023; Liang et al., 2023). Knowledge graph reasoning aims to derive new insights based on the existing graph structure and facts (Liang et al., 2022; Wang et al., 2023a; Luo et al., 2023b; Zhao et al., 2023). Graph neural networks that effectively capture the structure information have also been widely used in KG reasoning Zheng et al. (2022b; 2023e;b;c); Liu et al. (2023b;a). As a typical reasoning task, knowledge graph question answering (KGQA) aims to obtain answers based on knowledge from KGs, which can be generally divided into three categories: 1) embedding-based methods, 2) retrieval-augmented methods, and 3) semantic parsing methods.

Embedding-based methods model the entities and relations in embedding space and design special model architectures to reason answers. KV-Mem (Miller et al., 2016) adopts a Key-Value memory network to store triples for reasoning. EmbedKGQA (Saxena et al., 2020) and NSM (He et al., 2021) utilize the sequential model to mimic the multi-hop reasoning process. QA-GNN (Yasunaga et al., 2021) and Greaselm (Zhang et al., 2021) further adopt the graph neural network to capture the graph structure for reasoning. However, these methods need to design different model architectures, which are not flexible and generalizable.

Retrieval-augmented methods aims to retrieve the relative facts from the KGs to improve the reasoning performance. Early works adopt the page rank or random walk algorithm to retrieve subgraphs from KGs for reasoning (Sun et al., 2018; 2019). However, they ignore the semantic information in questions and lead to noisy retrieval results. Zhang et al. (2022) proposes a relation paths-based subgraph retrieval, resulting in a better retrieval and QA performance. Other lines of studies retrieving triples from KGs via BM25 (Li et al., 2023) or DPR (Karpukhin et al., 2020; Yu et al., 2022b) to improve the performance of LLMs. They discard the structure information in KGs which leads to suboptimal results. Recently, UniKGQA (Jiang et al., 2022) unifies the graph retrieval and reasoning process into a single model with LLMs, which achieves state-of-the-art performance on KGQA tasks.

Semantic parsing methods parse the question into a structural query (e.g., SPARQL) which can be executed by a query engine to get answers (Sun et al., 2020; Lan & Jiang, 2020). ArcaneQA (Gu & Su, 2022) dynamically generates the query based on results from previous steps. RnG-KBQA (Ye et al., 2022) first enumerate all possible queries and then rank them to get the final output. These methods heavily rely on the quality of generated queries. If the query is not executable, no answers will be generated. DECAF (Yu et al., 2022a) combines semantic parsing and LLMs reasoning to jointly generate answers, which also reach salient performance on KGQA tasks.

# A.3 RETRIEVAL ALGORITHM

Given a question q and a relation path as plan z, we adopt a constrained breadth-first search to retrieve the reasoning paths. The pseudocode code is presented in Algorithm 1.

We first initialize a queue of current reasoning paths Q with the entities in the question $T_{q}$ (line 3-5). Then, we iteratively expand each reasoning path in Q by adding the triples that are connected to the entities in the queue following the relation in relation path (line 11-19). The reasoning path is expanded until the length is equal to the length of the relation path. The expanded reasoning path is added to the set $W_{z}$ as the final results (line 8-10).

# A.4 DATASETS

We adopt two benchmark KGQA datasets: WebQuestionSP (WebQSP) $^{4}$ (Yih et al., 2016) and Complex WebQuestions (CWQ) $^{5}$ (Talmor & Berant, 2018) in this work. We follow previous works (Sun et al., 2018; Jiang et al., 2022) to use the same train and test splits for fair comparison. The statistic of the datasets are given in Table 6. The statistics of the answer numbers and reasoning hops are presented in Table 7 and Table 8, respectively.

Algorithm 1: Retrieve reasoning paths based on relation paths

Input: Question q, relation path $z = \{r_1, r_2, \ldots, r_l\}$ , KG G.

Output: Reasoning paths $W_z$ .

1 $W_z \leftarrow \emptyset$ ;

2 $Q \leftarrow \text{Queue}()$ ;

3 $\text{foreach } e_q \in T_q \text{ do}$ 4 | $Q.\text{append}((e_q, [])); // Initialize queue with question entities.$ 5 end

6 while $Q \neq \emptyset$ do

7 (s, $w_z$ ) $\leftarrow Q.\text{pop}()$ ;

8 if $len(w_z) = len(z)$ then

9 | $W_z.\text{append}(w_z)$ ;

10 end

11 if $len(w_z) < len(z)$ then

12 r $\leftarrow z[\text{len}(w_z) + 1]; // Get relation for next step.$ 13 foreach (s, $r', t) \in G$ do

14 if $r' = r$ then

15 w'_z.append((s, r, t)); // Expand the reasoning path.

16 Q.append((t, w'_z));

17 end

18 end

19 end

20 end

21 return $W_z$ ;

Table 6: Statistics of datasets. 

<table><tr><td>Datasets</td><td>#Train</td><td>#Test</td><td>Max #hop</td></tr><tr><td>WebQSP</td><td>2,826</td><td>1,628</td><td>2</td></tr><tr><td>CWQ</td><td>27,639</td><td>3,531</td><td>4</td></tr></table>

Table 7: Statistics of the number of answers for questions in WebQSP and CWQ. 

<table><tr><td>Dataset</td><td>#Ans = 1</td><td>2 ≥ #Ans ≤ 4</td><td>5 ≥ #Ans ≤ 9</td><td>#Ans ≥ 10</td></tr><tr><td>WebQSP</td><td>51.2%</td><td>27.4%</td><td>8.3%</td><td>12.1%</td></tr><tr><td>CWQ</td><td>70.6%</td><td>19.4%</td><td>6%</td><td>4%</td></tr></table>

Table 8: Statistics of the question hops in WebQSP and CWQ. 

<table><tr><td>Dataset</td><td>1 hop</td><td>2 hop</td><td> $\geq 3$ hop</td></tr><tr><td>WebQSP</td><td>65.49 %</td><td>34.51%</td><td>0.00%</td></tr><tr><td>CWQ</td><td>40.91 %</td><td>38.34%</td><td>20.75%</td></tr></table>

Table 9: Statistics of MetaQA-3hop datasets. 

<table><tr><td>Datasets</td><td>#Train</td><td>#Test</td><td>#hop</td></tr><tr><td>MetaQA-3hop</td><td>1,000</td><td>1,4274</td><td>3</td></tr></table>

Table 10: Statistics of constructed knowledge graphs. 

<table><tr><td>KG</td><td>#Entities</td><td>#Relations</td><td>#Triples</td></tr><tr><td>Freebase</td><td>2,566,291</td><td>7,058</td><td>8,309,195</td></tr><tr><td>Wiki-Movie</td><td>43,234</td><td>9</td><td>133,582</td></tr></table>

Table 11: The statistics of the instances in the instruction-tuning datasets. 

<table><tr><td>Planing Data</td><td>Retrieval-reasoning Data</td><td>Interpretability Data</td></tr><tr><td>216,006</td><td>30,465</td><td>2,000</td></tr></table>

To evaluate the transferability of RoG to other KGs. We further select the MetaQA-3hop dataset (Zhang et al., 2018) which is based on Wiki-Movies KGs $^{6}$ . We select 1000 samples from the training split. The statistic of the dataset is presented in Table 9.

Both WebQSP and CWQ can be reasoned based on Freebase KGs $^{7}$ (Bollacker et al., 2008). To reduce the size of KGs, following previous works (He et al., 2021; Jiang et al., 2022), we construct a subgraph of Freebase by extracting all triples that contain within the max reasoning hops of question entities in WebQSP and CWQ. Similarly, we construct a subgraph of Wiki-Movies KGs for MetaQA-3hop. The statistics of constructed KGs are presented in Table 10.

We construct the instruction-tuning dataset using the training split of WebQSP and CWQ datasets. For planning optimization, we generate data by extracting the shortest paths that connect the questions and answers, which serve as supervisory signals. In the retrieval-reasoning optimization phase, we feed both extracted shortest paths together with the questions to predict the answers. To enhance interpretability, we randomly select 1000 samples from each dataset along with their reasoning paths and answers. These are then input into ChatGPT to produce interpretive responses, which help empower our method in generating results with good explainability. The statistics of final training datasets are shown in Table 11.

# A.5 BASELINES

We compare RoG with 21 baselines grouping into 5 categories: 1) Embedding-based methods, 2) Retrieval-augmented methods, 3) Semantic parsing methods, 4) LLMs, and 5) LLMs+KGs methods. The details of each baseline are described as follows.

# Embedding-based methods.

- KV-Mem (Miller et al., 2016) adopts a Key-Value memory network to store triples and perform multi-hop reasoning by iterative operating on the memory.   
- EmbedKGQA (Saxena et al., 2020) models the reasoning on KGs as a sequential link prediction problem by using the embedding of entities and questions.   
- NSM (He et al., 2021) utilizes the sequential model to mimic the multi-hop reasoning process.   
- TransferNet (Shi et al., 2021) adopts a graph neural network to capture the relevance between entities and questions for reasoning.   
- KGT5 (Saxena et al., 2022) finetunes a sequence-to-sequence framework on KGs and generates answers based on the input question.

# Retrieval-augmented methods.

- GraftNet (Sun et al., 2018) retrieves relevant subgraphs from KGs with entity linking.   
- PullNet (Sun et al., 2019) trains a retrieval model composed of a LSTM and a graph neural network to retrieve a question-specific subgraph.

- SR+NSM (Zhang et al., 2022) proposes a relation-path retrieval to retrieve subgraphs for multi-hop reasoning.   
- SR+NSM+E2E (Zhang et al., 2022) further adopts an end-to-end training strategy to jointly train the retrieval and reasoning modules of SR+NSM.

# Semantic parsing methods.

- SPARQL (Sun et al., 2020) presents a novel skeleton grammar to represent the high-level structure of a complex question with language modes.   
- QGG (Lan & Jiang, 2020) generates a query graph for a question by simultaneously adding constraints and extending relation paths.   
- ArcaneQA (Gu & Su, 2022) dynamically generates the query based on results from previous steps.   
- RnG-KBQA (Ye et al., 2022) first enumerates all possible queries and then ranks them to get the final output.

# Large language models (LLMs).

- Flan-T5 (Chung et al., 2022) is an enhanced version of T5 models that is instruction fine-tuned on mixture of tasks.   
- Alpaca (Taori et al., 2023) is based on LLaMA and finetuned on an instruction-following dataset.   
- LLaMA2-Chat (Touvron et al., 2023) is a large language model that is optimized for dialogue purposes.   
- ChatGPT $^{8}$ is a powerful closed-source LLM that could follow instructions to conduct complex tasks $^{9}$ .   
- ChatGPT+CoT (Wei et al., 2022) uses the Chain-of-thought prompt to improve the reason ability of ChatGPT.

# LLMs+KGs methods.

- KD-CoT Wang et al. (2023b) retrieves relevant knowledge from KGs to generate faithful reasoning plans for LLMs.   
- UniKGQA (Jiang et al., 2022) unifies the graph retrieval and reasoning process into a single model with LLMs, which achieves state-of-the-art performance on KGQA tasks.   
- DECAF (Yu et al., 2022a) combines semantic parsing and LLMs reasoning to jointly generate answers, which also reach salient performance on KGQA tasks.

# A.6 IMPLEMENTATION SETTINGS

For RoG, we use LLaMA2-Chat-7B (Touvron et al., 2023) as the LLM backbone, which is instruction finetuned on the training split of WebQSP and CWQ as well as Freebase for 3 epochs. The batch size is set to 4 and the learning rate is set to 2e-5. We use the cosine learning rate scheduler policy with the warmup ratio set to 0.03. The training is conducted on 2 A100-80G GPUs for 38 hours. During inference, we first adopt the LLM to generate top-K relation paths with the highest probability as the plans. Then, we adopt the Algorithm 1 to retrieve reasoning paths, which are fed into the LLM to reason the final answers.

For LLM beelines, we use zero-shot prompting to conduct KGQA, which directly asks LLMs to answer the question. For other baselines, we directly copy their results reported in UniKGQA (Jiang et al., 2022) and DECAF (Yu et al., 2022a) for comparisons.

For combining the planning module of RoG with different LLMs, we use the planning module to generate plans (relation paths), which are executed on KGs to retrieve the reasoning paths. The retrieved paths are fed into different LLMs during inference by utilizing the reasoning prompts template shown in Appendix A.10.

Table 12: Performance of RoG on MetaQA-3hop. 

<table><tr><td rowspan="2">Strategies</td><td colspan="2">MetaQA-3hop</td></tr><tr><td>Hits@1</td><td>F1</td></tr><tr><td>RoG (train from scratch)</td><td>84.81</td><td>41.32</td></tr><tr><td>RoG (transfer from Freebase)</td><td>88.98</td><td>50.68</td></tr></table>

Table 13: Training time comparison. 

<table><tr><td>Method</td><td>Training on Freebase</td><td>Transferring to Wiki-Movies</td></tr><tr><td>RoG</td><td>38 hours</td><td>2 hours</td></tr></table>

# A.7 ADDITIONAL EXPERIMENT RESULTS

# A.7.1 TRANSFERABILITY TO OTHER KGs

We evaluate the transferability of RoG to other KGs. We select the MetaQA-3hop dataset (Zhang et al., 2018) which is based on Wiki-Movies KGs. We select 1000 samples from the training split and utilize two training strategies to finetune RoG: 1) training from scratch, where we directly train RoG from LLaMA2-Chat with 1000 samples; 2) transfer from Freebase, where we conduct a further finetuning based on RoG trained for Freebase. The results are shown in Table 12. From results, we can see that transfer from Freebase achieves better performance than training from scratch, which demonstrates the transferability of RoG to other KGs.

We also compare the training time on Freebase and transferring to Wiki-Movies KGs. From results shown in Table 13, we can see that the training time on Freebase is 38 hours, while the transferring time is only 2 hours. This demonstrates the efficiency of transferring RoG to other KGs.

# A.7.2 PERFORMANCE WITH DIFFERENT TRAINING DATA

In our experiment, we finetune RoG jointly on the training set of both WebQSP and CWQ datasets to maximize the ability of RoG for reasoning on Freebase. To fairly compare with other methods (e.g., UniKGQA (Jiang et al., 2022)) that are only trained on single dataset, we provide additional results of the performance of RoG trained on single dataset. From the results shown in Tables 14 and 15, we can see that RoG trained on single dataset still outperforms the STOA baselines (UniKGQA). Besides, we can also find that jointly training RoG on multiple datasets can further improve the performance. In the future, we will try to generate more QA datasets from Freebase to further improve the reasoning ability of RoG.

# A.7.3 PERFORMANCE COMPARISON WITH DIFFERENT FINETUNED LLMs

In Table 1, we report the zero-shot performance of different LLMs. However, RoG is finetuned on the training split of the QA dataset. To make a fair comparison, we further report the performance of the LLMs finetuned on the training split of the QA dataset in Table 16. From the results, we can see that RoG still outperforms the finetuned LLMs.

Table 14: Performance on WebQSP with different training data. 

<table><tr><td>Method</td><td>Training Data</td><td>Hits@1</td><td>F1</td></tr><tr><td>UniKGQA</td><td>WebQSP</td><td>77.2</td><td>72,2</td></tr><tr><td>RoG</td><td>WebQSP</td><td>81.5</td><td>61.8</td></tr><tr><td>RoG</td><td>WebQSP+CWQ</td><td>85.7</td><td>70.8</td></tr></table>

Table 15: Performance on CWQ with different training data. 

<table><tr><td>Method</td><td>Training Data</td><td>Hits@1</td><td>F1</td></tr><tr><td>UniKGQA</td><td>CWQ</td><td>51.2</td><td>49.1</td></tr><tr><td>RoG</td><td>CWQ</td><td>59.1</td><td>52.9</td></tr><tr><td>RoG</td><td>WebQSP+CWQ</td><td>62.6</td><td>56.2</td></tr></table>

Table 16: Performance comparison with different finetuned LLMs (Hits@1). 

<table><tr><td>Method</td><td>WebQSP</td><td>CWQ</td></tr><tr><td>Alpaca-7B (Zero Shot)</td><td>51.78</td><td>27.44</td></tr><tr><td>LLaMA2-Chat-7B (Zero Shot)</td><td>64.37</td><td>34.60</td></tr><tr><td>Alpaca-7B (Finetuned)</td><td>74.57</td><td>55.98</td></tr><tr><td>LLaMA2-Chat-7B (Finetuned)</td><td>73.89</td><td>53.49</td></tr><tr><td>RoG</td><td>85.75</td><td>62.65</td></tr></table>

# A.7.4 RETRIEVAL COSTS

We present the retrieval time and number of retrieved reasoning paths in Figure 4. From results, we can see that the retrieval time increases with the number of top-K relation paths. Therefore, we should select a proper K to balance the retrieval time and the number of retrieved reasoning paths. In experiments, we set K = 3.

# A.7.5 PERFORMANCE ON DIFFERENT HOPS

We present the performance of RoG and its variants on different hops of questions in Table 17. From results, we can see that RoG achieves better performance than its variants on different hops of questions, especially questions with more than 3 hops. This demonstrates the importance of relation paths for improving the reasoning performance of LLMs on complex questions.

# A.7.6 PERFORMANCE ON DIFFERENT ANSWER NUMBERS

We also present the performance of RoG and its variants on questions with different numbers of answers in Table 18. From results, we can see that RoG achieves better performance than its variants on questions with different numbers of answers. Specifically, with the number of answer increasing, the performance of RoG w/o planning decreases significantly due to the lack of knowledge from

![](images/c192bd8a06d0fae39120295f7e3bea60f8e3aeb093d527b9562a8b5f5395bccb.jpg)

![](images/bd7a7e4027a28c938f9b2935454750b0d2cf3ab17447121d692d086f00fa88c3.jpg)

<details>
<summary>bar</summary>

| Top-K relation paths | # Retrieved Reasoning Paths | Retrieval Time (s) |
| --------------------- | --------------------------- | ------------------ |
| 1                     | 8                           | 0.5                |
| 2                     | 19                          | 1.0                |
| 3                     | 27                          | 1.5                |
| 4                     | 34                          | 1.7                |
| 5                     | 40                          | 1.8                |
</details>

![](images/d3c0d5d533b0fdf35ef73b7d7d7beabfe7275396bd25b43b685b0a156ed6eeca.jpg)

<details>
<summary>bar</summary>

| Top-K relation paths | # Retrieved Reasoning Paths | Retrieval Time (s) |
| --------------------- | --------------------------- | ------------------ |
| 1                     | 35                          | 0.8                |
| 2                     | 65                          | 1.2                |
| 3                     | 90                          | 1.5                |
| 4                     | 105                         | 2.0                |
| 5                     | 120                         | 2.5                |
</details>

Figure 4: Average retrieval time and average number of retrieved reasoning paths w.r.t. the number of top-K relation paths.

Table 17: F1 scores of RoG and its variants for different hops of questions. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">WebQSP</td><td colspan="3">CWQ</td></tr><tr><td>1 hop</td><td>2 hop</td><td> $\geq 3$ hop</td><td>1 hop</td><td>2 hop</td><td> $\geq 3$ hop</td></tr><tr><td>RoG</td><td>77.03</td><td>64.86</td><td>-</td><td>62.88</td><td>58.46</td><td>37.82</td></tr><tr><td>RoG w/o reasoning</td><td>57.06</td><td>25.49</td><td>-</td><td>17.06</td><td>34.25</td><td>17.07</td></tr><tr><td>RoG w/o planning</td><td>50.33</td><td>51.66</td><td>-</td><td>31.04</td><td>33.93</td><td>23.29</td></tr></table>

Table 18: F1 scores of RoG and its variants for questions with varying numbers of answers. 

<table><tr><td rowspan="2">Methods</td><td colspan="4">WebQSP</td><td colspan="4">CWQ</td></tr><tr><td>#Ans = 1</td><td>2 ≥ #Ans ≤ 4</td><td>5 ≥ #Ans ≤ 9</td><td>#Ans ≥ 10</td><td>#Ans = 1</td><td>2 ≥ #Ans ≤ 4</td><td>5 ≥ #Ans ≤ 9</td><td>#Ans ≥ 10</td></tr><tr><td>RoG</td><td>67.89</td><td>79.39</td><td>75.04</td><td>58.33</td><td>56.90</td><td>53.73</td><td>58.36</td><td>43.62</td></tr><tr><td>RoG w/o reasoning</td><td>33.49</td><td>52.80</td><td>58.05</td><td>66.01</td><td>16.61</td><td>27.06</td><td>40.10</td><td>34.45</td></tr><tr><td>RoG w/o planning</td><td>55.03</td><td>51.08</td><td>44.81</td><td>27.00</td><td>34.08</td><td>34.16</td><td>31.67</td><td>25.21</td></tr></table>

KGs. Although, RoG w/o reasoning can retrieve more answers to improve the performance. It still is inferior to RoG due to the lack of reasoning ability to remove the noise.

# A.8 CASE STUDIES: RELATION PATHS

We illustrate several examples of relation paths generated by RoG in Table 19.

# A.9 CASE STUDIES: INTERPRETABLE RESULTS

We illustrate several examples of interpretable reasoning results generated by RoG in Table 20.

# A.10 PROMPTS

Planning module aims to generate faithful relation paths as plans for answering the question. The instruction template is presented as follows:

# Planning Prompt Template

Please generate a valid relation path that can be helpful for answering the following question: <Question>

where <Question> indicates the question.

The reasoning module takes the question $q$ and a set of reasoning paths $\mathcal{W}_z$ to generate answers $a$ . The instruction template is presented as follows:

# Reasoning Prompt Template

Based on the reasoning paths, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.

Reasoning Paths:
<Reasoning Paths>

Question:
<Question>

where <Reasoning Paths> denotes the retrieved reasoning paths $W_{z}$ which are formatted as a series of structural sentences:

$$
e _ {0} \rightarrow r _ {1} \rightarrow e _ {1} \rightarrow \dots \rightarrow r _ {l} \rightarrow e _ {l}
$$

...

$$
e _ {0} \rightarrow r _ {1} \rightarrow e _ {1} \rightarrow \dots \rightarrow r _ {l} \rightarrow e _ {l}.
$$

To exploit the explanation ability of RoG, we design a new instruction template for the reasoning module to generate interpretable results. The instruction template is presented as follows:

# Explanation Prompt Template

Based on the reasoning paths, please answer the given question and explain why.

Here are some examples:
<Examples>

Reasoning Paths:
<Reasoning Paths>

Question:
<Question>

where the Examples denotes a few-shot human-annotated examples to demonstrate the explanation process.

Table 19: Examples of the generated relation paths. 

<table><tr><td>Question</td><td>Top-3 Relation Paths</td></tr><tr><td>what does jamaican people speak?</td><td> $z_1$ : location.country.languages_spoken $z_2$ : language.human_languageCountries_spoken_in $z_3$ : location.country.official_language</td></tr><tr><td>where is jamarcus russell from?</td><td> $z_1$ : location.location.people_born_here $z_2$ : people.person.place_of_birth $z_3$ : sports.sports_league_draft_pick.player → sports.sports_league_draft_pick.location</td></tr><tr><td>where did edgar allan poe died?</td><td> $z_1$ : people.deceased_person.place_of_death $z_2$ : people.cause_of_death.people $z_3$ : people.person.place_of_birth</td></tr><tr><td>what highschool did harper lee go to?</td><td> $z_1$ : people.person.education → education.educational_institution.students_graduates $z_2$ : education.education.student → education.educational_institution.students_graduates $z_3$ : people.person.education → education.education.institutio</td></tr><tr><td>what are the songs that justin bieber wrote?</td><td> $z_1$ : music.recording.artist $z_2$ : music.composition.composer $z_3$ : music.composer.compositions</td></tr><tr><td>what are the religions practiced in indonesia?</td><td> $z_1$ : people.person.nationality → people.person.religion $z_2$ : location.statistical_region.religions → location.religion_percentage.religion $z_3$ : location.country.languages_spoken → religion.religion.languages</td></tr><tr><td>Lou Seal is the mascot for the team that last won the World Series when?</td><td> $z_1$ : sports.mascot.team → sports.sports_championship_event.champion $z_2$ : sports.mascot.team → sports.sports_team.championships $z_3$ : sports.sports_championship_event.championship</td></tr><tr><td>What type of government is used in the country with Northern District?</td><td> $z_1$ : location.administrative_division.first_level_division_of → government.form_of_government.countries $z_2$ : location.administrative_division.first_level_division_of → location.country.form_of_government $z_3$ : administrative_division.first_level_division_of → government.form_of_government.countries</td></tr><tr><td>The people from the country that contains Nord-Ouest Department speak what languages today?</td><td> $z_1$ : location.administrative_division.first_level_division_of → language.human_language.countries_spoken_in $z_2$ : location.administrative_division.first_level_division_of → location.country.languages_spoken $z_3$ : base.aareas.schema.administrative_area.administrative_parent → location.country.languages_spoken</td></tr><tr><td>What stadium does the team with mascot named Hank play at?</td><td> $z_1$ : sports.mascot.team → sports.sports_facility.teams $z_2$ : sports.sports_team.team_mascot → sports.sports_facility.teams $z_3$ : sports.mascot.team → sports.sports_team.arena_stadium</td></tr><tr><td>Which popular sports team in Spain, that won the 2014 Eurocup Finals championship?</td><td> $z_1$ : sports.sports_team.championships → sports.sports_team_location.teams $z_2$ : sports.sports_team.location → sports.sports_team_location.teams $z_2$ : sports.sports_team.location → sports.sports_team_location.teams $z_3$ : sports.sports_team_location.teams</td></tr><tr><td>What educational institution with the mascot named Washington University Bear did Tennessee Williams go to?</td><td> $z_1$ : education.educational_institution.mascot $z_2$ : people.person.nationality → location.location.containedby $z_3$ : education.education.student → education.education.institution</td></tr><tr><td>Who is the current head coach of the NFL squad owned by the Rooney family?</td><td> $z_1$ : sports.professional_sports_team.owner_s → american_football.football_coach.current_team_head_coached $z_2$ : sports.professional_sports_team.owner_s → american_football.football_team.current_head_coach $z_3$ : american_football.football_coach.current_team_head_coached</td></tr><tr><td>What is the home field of the sports team whose mascot is named Fredbird?</td><td> $z_1$ : sports.mascot.team → sports.sports_facility.teams $z_2$ : sports.sports_team.team_mascot → sports.sports_facility.teams $z_3$ : sports.mascot.team → sports.sports_team.arena_stadium</td></tr></table>

Table 20: Examples of faithful and interpretable reasoning of RoG. 

<table><tr><td>Question</td><td>Lou Seal is the mascot for the team that last won the World Series when?</td></tr><tr><td>Answer</td><td>2014</td></tr><tr><td>Reasoning Paths</td><td>Lou Seal → sports.mascot.team → San Francisco Giants → sports.sports_championship_event.champion → 2014 World Series</td></tr><tr><td>RoG</td><td>Based on the provided reasoning paths, we can determine when the team for which Lou Seal is the mascot last won the World Series:1. Lou Seal is associated with the San Francisco Giants.2. The San Francisco Giants are associated with winning the 2014 World Series.Therefore, the San Francisco Giants, the team for which Lou Seal is the mascot, last won the World Series in 2014.</td></tr><tr><td>Question</td><td>what is my timezone in louisiana?</td></tr><tr><td>Answer</td><td>Central Time Zone</td></tr><tr><td>Reasoning Paths</td><td>Louisiana → location.location.partially_containedby → Tangipahoa River → geography.river.origin → Osyka → location.location.time_zones → Central Time ZoneLouisiana → location.location.containedby → Alexandria → location.location.containedby → United States of America → location.location.time_zones → Central Time Zone</td></tr><tr><td>RoG</td><td>Based on the reasoning paths provided, it can be determined that Louisiana is in the Central Time Zone.Here&#x27;s the explanation:1. The first reasoning path tells us that Louisiana is partially contained by the Tangipahoa River, which is a geographical feature. However, it doesn&#x27;t directly mention the time zone.2. The second reasoning path establishes that Louisiana is contained by Alexandria, which is in turn contained by the United States of America. The United States of America is in the Central Time Zone.Therefore, based on these reasoning paths, the time zone in Louisiana is the Central Time Zone.</td></tr><tr><td>Question</td><td>Which child of Walt Disney died from lung cancer?</td></tr><tr><td>Answer</td><td>Sharon Mae Disney</td></tr><tr><td>Reasoning Paths</td><td>Lung cancer → people.deceased_person.cause_of_death → Sharon Mae DisneyLung cancer → people.deceased_person.cause_of_death → Walt DisneyWalt Disney → people.deceased_person.cause_of_death → Lung cancerWalt Disney → people.person.children → Diane Disney MillerWalt Disney → people.cause_of_death.people → Circulatory collapseWalt Disney → people.person.parents → Flora Call DisneyWalt Disney → people.person.parents → Elias DisneyWalt Disney → people.person.parents → Sharon Mae Disney</td></tr><tr><td>RoG</td><td>Based on the reasoning paths provided, it appears that Sharon Mae Disney is the child of Walt Disney who died from lung cancer.The reasoning path &quot;Walt Disney → people.deceased_person.cause_of_death → Lung cancer&quot; directly connects Walt Disney to lung cancer as a cause of death, and since Sharon Mae Disney is also linked to lung cancer as a cause of death through the path &quot;Lung cancer → people.deceased_person.cause_of_death → Sharon Mae Disney,&quot; it can be concluded that Sharon Mae Disney is the child of Walt Disney who died from lung cancer.</td></tr></table>