# Generative Representational Instruction Tuning

Niklas Muennighoff $^{c}$ Hongjin Su $^{h}$ Liang Wang $^{m}$ Nan Yang $^{m}$

Furu Wei $^{m}$ Tao Yu $^{h}$ Amanpreet Singh $^{c}$ Douwe Kiela $^{c}$

$^{c}$ Contextual AI $^{h}$ The University of Hong Kong $^{m}$ Microsoft Corporation

niklas@contextual.ai

# Abstract

All text-based language problems can be reduced to either generation or embedding. Current models only perform well at one or the other. We introduce generative representational instruction tuning (GRIT) whereby a large language model is trained to handle both generative and embedding tasks by distinguishing between them through instructions. Compared to other open models, our resulting GRITLM 7B sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks. By scaling up further, GRITLM 8x7B outperforms all open generative language models that we tried while still being among the best embedding models. Notably, we find that GRIT matches training on only generative or embedding data, thus we can unify both at no performance loss. Among other benefits, the unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by $>60\%$ for long documents, by no longer requiring separate retrieval and generation models. Models, code, etc. are freely available at https://github.com/ContextualAI/gritlm.

![](images/6a240084835a6642fb282257fe9bc838efd3e938769afc85264dc547ce5cb726.jpg)  
Figure 1: Performance of various models on text representation (embedding) and generation tasks. GRITLM is the first model to perform best-in-class at both types of tasks simultaneously.

![](images/661354d967806522579727188304faed8d3560f6b93dc788eb453c2330c7ded3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["GRIT"] --> B["Representational Instruction Tuning"]
    A --> C["Generative Representational Instruction Tuning"]
    A --> D["Generative Instruction Tuning"]
    B --> E["Given a scientific paper title, retrieve the paper's abstract\nBitcoin: A Peer-to-Peer Electronic Cash System"]
    B --> F["If an obscure legal term is given as the query, fetch text from law books or legal databases that can help explain the term. celestial res nullius"]
    B --> G["Please write me a blog post about my recent hike of Mt. Fuji at midnight."]
    B --> H["You have two ropes, each takes exactly 1 hour to burn. How would you use them to time exactly 15 minutes? The ropes are of uneven densities, so half the rope does not necessarily take half the time."]
    C --> I["0.42, 1.52, -0.01..."]
    C --> J["0.01, -1.01, 0.45..."]
    C --> K["Sure, here is the blog post.\nIt was August the 10th when I arrived at Lake Kawaguchi from..."]
    C --> L["You start by..."]
```
</details>

Figure 2: GRIT. The same model handles both text representation and generation tasks based on the given instruction. For representation tasks, instructions ideally contain the target domain, intent, and unit [5]. The representation is a tensor of numbers, while the generative output is text.

# 1 Introduction

Creating a single general model that performs well at a wide range of tasks has been a long-standing goal of the field of artificial intelligence $[75, 68, 21, 131, 141]$ . Recently, large language models (LLMs) have emerged as a promising direction for a single multi-task model $[126, 13]$ . Prior work has argued that all text-based language problems can be reduced to generation and thus handled by a single LLM $[129, 38]$ .

However, tasks that use embeddings, such as clustering or retrieval $[109]$ , have largely been ignored from this perspective. Today, text embeddings power many critical real-world applications ranging from search engines to user-facing chatbots $[64, 146]$ . While integrating text embeddings into the generative paradigm is possible by generating a sequence of numbers to form the embedding tensor, it becomes impractical due to the high dimensionality and precision requirements of embeddings. Thus, it is more common and much easier to use the hidden state of the model as the embedding representation, which is already a numeric tensor $[106, 160, 104]$ . However, doing so for current generative models leads to poor performance. For example, while the T5 model $[129, 135]$ can handle any generative task in a sequence-to-sequence fashion, it requires finetuning to make its hidden state useful for text embedding $[113, 114]$ during which it loses its generative capabilities.

We introduce GRIT (generative representational instruction tuning) which unifies embedding and generative tasks, leading to a model that excels at both tasks as shown in Figure 1. Figure 2 depicts how GRIT combines two previously disjoint training paradigms: (1) Generative instruction tuning, whereby the model is trained to respond to instructions by generating an answer [166, 135]; and (2) Representational instruction tuning, whereby the model is trained to represent a provided input according to an instruction [145, 5]. Via the instructions and separate loss functions the model learns to differentiate the two streams. We test our approach on models with up to 47B parameters and, due to its simplicity, we expect the method to generalize to any LLM, even non-transformers. This unification via GRIT leads to three advantages:

a) Performance: Our unified model matches the performance of embedding-only and generative-only variants, even outperforming them on some tasks. At 7B parameters, GRITLM sets a new state of the art on the Massive Text Embedding Benchmark [109] among open models and at the same time outperforms much larger models on generative tasks, such as Llama 2 70B. By scaling further, GRITLM 8x7B is the best open generative language model on our task average, while only using 13B parameters at inference. Further, as our models use sliding window attention [20, 9] they can handle generative and embedding inputs of arbitrary length.   
b) Efficiency: Generative and embedding models are commonly used together to make up for each other's deficiencies [57, 86]. One such scenario is Retrieval-Augmented Generation (RAG) [86], where an embedding model is used to retrieve context that is provided to the generative model to

answer a user query. This requires passing the user query and the context into both the generative and the embedding model for a total of four forward passes. With GRITLM, the embedding and generative model are equivalent, allowing us to cache computations and halve the necessary number of forward passes. We find that this can lead to $>60\%$ faster RAG at inference with long documents.
c) Simplicity: Currently, API providers such as OpenAI provide separate generative and embedding endpoints. This requires separate load balancing, additional storage, and more complex serving software. A single model that handles both use cases significantly simplifies infrastructure needs.

Compared to generative instruction tuning, the main downside of GRIT is that it requires more finetuning compute due to training with two objectives. However, finetuning is generally cheap compared to pretraining, thus we think the benefits vastly outstrip this problem. Further, when considering training a separate generative and embedding model from scratch (e.g. for RAG), GritLM is generally cheaper when incorporating the pretraining compute, as there is only one pretraining and finetuning for GritLM, not separate ones for both a generative and an embedding model. Thus we recommend practitioners building instruction-following language models to adopt GRIT.

Alongside GRIT, we introduce novel performance improvements for embedding models including the use of bidirectional attention with mean pooling for LLM embeddings and making in-batch negatives stem from the same dataset rather than any dataset, as well as novelties for generative models such as mixing sample- and token-level loss aggregation. We ablate these in detail in §3.3. We also propose new ways to reduce memory requirements during embedding model training in Appendix H.

# 2 GRIT

![](images/1cb108d34a2eff2d91296623c2d3462ce109e1164c81adf0ac159ef2b0590233.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph 'Representation'
        A1[" "] --> B1[" "]
        A2[" "] --> B2[" "]
        A3[" "] --> B3[" "]
        A4[" "] --> B4[" "]
        A5[" "] --> B5[" "]
        A6[" "] --> B6[" "]
        A7[" "] --> B7[" "]
        A8[" "] --> B8[" "]
        A9[" "] --> B9[" "]
    end

    subgraph 'Generation'
        C1[" "] --> D1[" "]
        C2[" "] --> D2[" "]
        C3[" "] --> D3[" "]
        C4[" "] --> D4[" "]
        C5[" "] --> D5[" "]
        C6[" "] --> D6[" "]
        C7[" "] --> D7[" "]
        C8[" "] --> D8[" "]
        C9[" "] --> D9[" "]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4
    A5 --> B5
    A6 --> B6
    A7 --> B7
    A8 --> B8
    A9 --> B9
    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4
    A5 --> B5
    A6 --> B6
    A7 --> B7
    A8 --> B8
    A9 --> B9
    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4
    A5 --> B5
    style A1 fill:#f0f0f0,stroke:#333
    style A2 fill:#f0f0f0,stroke:#333
    style A3 fill:#f0f0f0,stroke:#333
    style A4 fill:#f0f0f0,stroke:#333
    style A5 fill:#f0f0f0,stroke:#333
    style A6 fill:#f0f0f0,stroke:#333
    style A7 fill:#f0f0f0,stroke:#333
    style A8 fill:#f0f0f0,stroke:#333
    style A9 fill:#f0f0f0,stroke:#333
```
</details>

Figure 3: GRITLM architecture and format. Left: GRITLM uses bidirectional attention over the input for embedding tasks. Mean pooling is applied over the final hidden state to yield the final representation. Right: GRITLM uses causal attention over the input for generative tasks. A language modeling head on top of the hidden states predicts the next tokens. The format supports conversations with multiple turns (indicated with “...”).

GRIT unifies representational instruction tuning $[145, 5, 162]$ and generative instruction tuning $[166, 135, 110]$ into a single model. We finetune a pretrained large language model $[13]$ with embedding and generative instruction data in a consistent format as depicted in Figure 3. For embedding data, we follow prior work and compute the loss using a contrastive objective with in-batch negatives $[18, 52]$ :

$$
\mathcal {L} _ {\text { Rep }} = - \frac {1}{M} \sum_ {i = 1} ^ {M} \log \frac {\exp (\tau \cdot \sigma (f _ {\theta} (q ^ {(i)}) , f _ {\theta} (d ^ {(i)})))}{\sum_ {j = 1} ^ {M} \exp (\tau \cdot \sigma (f _ {\theta} (q ^ {(i)}) , f _ {\theta} (d ^ {(j)})))} \tag {1}
$$

where f is GRITLM parametrized by the model $\theta$ , $\tau$ is a temperature hyperparameter and $\sigma$ corresponds to pooling applied to each output followed by cosine similarity. q and d are query and document samples. As depicted in Figure 3, we use bidirectional attention followed by mean pooling,

which corresponds to averaging the hidden states across the sequence length. During pooling, we only average the final hidden states of the input sample, ignoring the instruction and format tokens. However, the instruction and format tokens still influence the final representation through the self-attention mechanism $[158]$ .

To compute the loss on generative data, we use the language modeling objective whereby the model needs to predict the next token $[125, 126]$ :

$$
\mathcal {L} _ {\text { Gen }} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \log P (f _ {\theta , \eta} (x ^ {(i)}) | f _ {\theta , \eta} (x ^ {(<   i)})) \tag {2}
$$

where f is GRITLM parametrized by the model $\theta$ and the language modeling head $\eta$ , which is only used for generation. x are generative training samples. We only compute loss over predicted tokens i.e. “{response}<s>” in Figure 3. A key consideration is whether the generative loss is aggregated at the sample or token level. Aggregating at the sample level corresponds to giving each sample the same weight within a batch regardless of its token count. Such aggregation is commonly used for instruction tuning, as it can boost performance on discriminative tasks [110]. However, Muennighoff et al. [110] also show how this in turn leads to a model biased toward short generations. Meanwhile, aggregation at the token level corresponds to giving each token the same weight, thus samples with many tokens become more important. This usually leads to a model producing longer generations, which can be important for performance on generative tasks. Especially, human or machine-evaluated generative tasks, such as AlpacaEval [91], are known to be biased toward preferring longer generations [164]. Note that when every sample has the same sequence length such as during pretraining or when the batch size is 1, token and sample level generative loss are equal to each other. One can also mix the two to balance their trade-offs, for example doing token level loss across a subset of the batch and then giving each subset the same weight. We explore the trade-offs in our ablations in §3.3. We sum the objectives with optional loss weights $\lambda_{Rep}$ and $\lambda_{Gen}$ :

$$
\mathcal {L} _ {\text { GRIT }} = \lambda_ {\text { Rep }} \mathcal {L} _ {\text { Rep }} + \lambda_ {\text { Gen }} \mathcal {L} _ {\text { Gen }} \tag {3}
$$

Notably, our formulation supports differing numbers of embedding samples $(M)$ and generative samples/tokens $(N)$ . This allows for significantly increasing the embedding batch size while keeping the generative batch size fixed. A large embedding batch size is often key to well-performing text embedding models [171]. However, it comes at the cost of requiring more compute at each step.

# 3 Experiments

In this section, we first outline our experimental setup in §3.1. In §3.2, we discuss and benchmark the embedding and generative performance of our models. Finally, in §3.3, we ablate the settings that led to our final models, including training data, precision, pooling, sequence length, and loss weights.

# 3.1 Setup

We finetune our final models from Mistral 7B [69] and Mixtral 8x7B [70] using adaptations of E5 [162] and the Tülu 2 data [65]. For E5, we adapt it by adding S2ORC [93] to increase its scientific data (“E5S”), while for Tülu 2 we filter out their custom prompts that contain answers related to the origin of their model. For GRITLM 7B, we use a batch size of 2048 for embedding data and 256 for generative data and we train the model for a total of 1253 steps corresponding to one epoch on the generative data and 1.36 epochs on the embedding data. For GRITLM 8x7B, the embedding batch size is 256 due to compute limitations. We use several strategies to reduce the memory required during training including a novel technique to split the embedding triplet into separate forward and backward passes detailed in Appendix H. Other hyperparameters are detailed in the ablation experiments in §3.3 and Appendix I.

For embedding performance we evaluate using the 56 main datasets from MTEB [109]. For generative performance, we largely follow the evaluation setup of Ivison et al. [65] except that we use the HumanEvalSynthesize [107] variant of HumanEval, as it is more adequate for instruction-following models. We explain each task in more detail in Appendix E.

Table 1: Embedding performance of GRITLM and others. We indicate parameter counts where available (B=billions). See Appendix E for task, metric, and dataset details. Appendix G contains per-dataset results of GRITLM models. LLMs not finetuned for embedding (Llama 2 70B, Mistral 7B (Instruct), GPT-J 6B, Gen.-only) are evaluated with weighted-mean pooling [106]. ♥ Results from the MTEB leaderboard (https://hf.co/spaces/mteb/leaderboard) 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td colspan="9">Proprietary models♥</td></tr><tr><td>OpenAI v3</td><td>75.5</td><td>49.0</td><td>85.7</td><td>59.2</td><td>55.4</td><td>81.7</td><td>29.9</td><td>64.6</td></tr><tr><td colspan="9">Other Open Models♥</td></tr><tr><td>Llama 2 70B</td><td>60.4</td><td>29.0</td><td>47.1</td><td>38.5</td><td>9.0</td><td>49.1</td><td>26.1</td><td>35.6</td></tr><tr><td>Mistral 7B</td><td>63.5</td><td>34.6</td><td>53.5</td><td>43.2</td><td>13.2</td><td>57.4</td><td>19.7</td><td>40.5</td></tr><tr><td>Mistral 7B Instruct</td><td>67.1</td><td>34.6</td><td>59.6</td><td>44.8</td><td>16.3</td><td>63.4</td><td>25.9</td><td>43.7</td></tr><tr><td>GPT-J 6B</td><td>66.2</td><td>39.0</td><td>60.6</td><td>48.9</td><td>19.8</td><td>60.9</td><td>26.3</td><td>45.2</td></tr><tr><td>SGPT BE 5.8B</td><td>68.1</td><td>40.3</td><td>82.0</td><td>56.6</td><td>50.3</td><td>78.1</td><td>31.5</td><td>58.9</td></tr><tr><td>Instructor XL 1.5B</td><td>73.1</td><td>44.7</td><td>86.6</td><td>57.3</td><td>49.3</td><td>83.1</td><td>32.3</td><td>61.8</td></tr><tr><td>BGE Large 0.34B</td><td>76.0</td><td>46.1</td><td>87.1</td><td>60.0</td><td>54.3</td><td>83.1</td><td>31.6</td><td>64.2</td></tr><tr><td>E5 Mistral 7B</td><td>78.5</td><td>50.3</td><td>88.3</td><td>60.2</td><td>56.9</td><td>84.6</td><td>31.4</td><td>66.6</td></tr><tr><td colspan="9">GRITLM</td></tr><tr><td>Gen.-only 7B</td><td>65.4</td><td>32.7</td><td>54.2</td><td>43.0</td><td>13.7</td><td>60.2</td><td>21.1</td><td>41.2</td></tr><tr><td>Emb.-only 7B</td><td>78.8</td><td>51.1</td><td>87.1</td><td>60.7</td><td>57.5</td><td>83.8</td><td>30.2</td><td>66.8</td></tr><tr><td>GRITLM 7B</td><td>79.5</td><td>50.6</td><td>87.2</td><td>60.5</td><td>57.4</td><td>83.4</td><td>30.4</td><td>66.8</td></tr><tr><td>GRITLM 8x7B</td><td>78.5</td><td>50.1</td><td>85.0</td><td>59.8</td><td>55.1</td><td>83.3</td><td>29.8</td><td>65.7</td></tr></table>

Table 2: Generative performance of GRITLM and others. We indicate parameter counts where available (B=billions). See Appendix E for dataset, setup, metric details, and abbreviations. Results from Ivison et al. [65] except for numbers marked with $\spadesuit$ which are from Touvron et al. [156] and $\dagger$ which are from us. For models that cannot be easily used as chat models, we set Alpaca to 0. 

<table><tr><td>Dataset (→)</td><td>MMLU</td><td>GSM8K</td><td>BBH</td><td>TyDi QA</td><td>HumanEval</td><td>Alpaca</td><td>Avg.</td></tr><tr><td>Setup (→)</td><td>0 FS</td><td>8 FS, CoT</td><td>3 FS, CoT</td><td>1 FS, GP</td><td>0 FS</td><td>0 FS, 1.0</td><td></td></tr><tr><td>Metric (→)</td><td>EM</td><td>EM</td><td>EM</td><td>F1</td><td>pass@1</td><td>% Win</td><td></td></tr><tr><td colspan="8">Proprietary models♥</td></tr><tr><td>GPT-4-0613</td><td>81.4</td><td>95.0</td><td>89.1</td><td>65.2</td><td> $86.6^{\dagger}$ </td><td>91.2</td><td>84.8</td></tr><tr><td colspan="8">Other Open Models♥</td></tr><tr><td>GPT-J 6B</td><td>27.7</td><td>2.5</td><td>30.2</td><td>9.4</td><td>9.8</td><td>0.0</td><td>13.3</td></tr><tr><td>SGPT BE 5.8B</td><td>24.4</td><td>1.0</td><td>0.0</td><td>22.8</td><td>0.0</td><td>0.0</td><td>8.0</td></tr><tr><td>Zephyr 7B β</td><td>58.6</td><td>28.0</td><td>44.9</td><td>23.7</td><td>28.5</td><td>85.8</td><td>44.9</td></tr><tr><td>Llama 2 7B</td><td>41.8</td><td>12.0</td><td>39.3</td><td>51.2</td><td> $12.8^{\blacklozenge}$ </td><td>0.0</td><td>26.2</td></tr><tr><td>Llama 2 13B</td><td>52.0</td><td>25.0</td><td>48.9</td><td>56.5</td><td> $18.3^{\blacklozenge}$ </td><td>0.0</td><td>33.5</td></tr><tr><td>Llama 2 70B</td><td>64.5</td><td>55.5</td><td>66.0</td><td>62.6</td><td> $29.9^{\blacklozenge}$ </td><td>0.0</td><td>46.4</td></tr><tr><td>Llama 2 Chat 13B</td><td>53.2</td><td>9.0</td><td>40.3</td><td>32.1</td><td> $19.6^{\dagger}$ </td><td>91.4</td><td>40.9</td></tr><tr><td>Llama 2 Chat 70B</td><td>60.9</td><td>59.0</td><td>49.0</td><td>44.4</td><td> $34.3^{\dagger}$ </td><td>94.5</td><td>57.0</td></tr><tr><td>Tülu 2 7B</td><td>50.4</td><td>34.0</td><td>48.5</td><td>46.4</td><td> $24.5^{\dagger}$ </td><td>73.9</td><td>46.3</td></tr><tr><td>Tülu 2 13B</td><td>55.4</td><td>46.0</td><td>49.5</td><td>53.2</td><td>31.4</td><td>78.9</td><td>52.4</td></tr><tr><td>Tülu 2 70B</td><td>67.3</td><td>73.0</td><td>68.4</td><td>53.6</td><td>41.6</td><td>86.6</td><td>65.1</td></tr><tr><td>Mistral 7B</td><td>60.1</td><td>44.5</td><td>55.6</td><td>55.8</td><td>30.5</td><td>0.0</td><td>41.1</td></tr><tr><td>Mistral 7B Instruct</td><td>53.0</td><td>36.0</td><td>38.5</td><td>27.8</td><td>34.0</td><td>75.3</td><td>44.1</td></tr><tr><td>Mixtral 8x7B Instruct</td><td>68.4</td><td>65.0</td><td>55.9</td><td>24.3</td><td>53.5</td><td>94.8</td><td>60.3</td></tr><tr><td colspan="8">GRITLM</td></tr><tr><td>Emb.-only 7B</td><td>23.5</td><td>1.0</td><td>0.0</td><td>21.0</td><td>0.0</td><td>0.0</td><td>7.6</td></tr><tr><td>Gen.-only 7B</td><td>57.5</td><td>52.0</td><td>55.4</td><td>56.6</td><td>34.5</td><td>75.4</td><td>55.2</td></tr><tr><td>GRITLM 7B</td><td>57.6</td><td>57.5</td><td>54.8</td><td>55.4</td><td>32.8</td><td>74.8</td><td>55.5</td></tr><tr><td>GRITLM 8x7B</td><td>66.7</td><td>61.5</td><td>70.2</td><td>58.2</td><td>53.4</td><td>84.0</td><td>65.7</td></tr></table>

# 3.2 Main Results

GRIT leads to a state-of-the-art embedding and generative model We benchmark GRITLM 7B, GRITLM 8x7B and generative- and embedding-only variants with other models in Table 1 and Table 2. We find that GRITLM 7B outperforms all prior open models on the Massive Text Embedding Benchmark [109] while still outperforming all generative models up to its size of 7 billion parameters. GRIT models are the only ones that can handle both embedding and generation at best-in-class performance (Figure 1). For example, using Llama 70B [156] for embedding leads to a score of only 35.6 on MTEB as depicted in Table 1. GRITLM almost doubles that performance on MTEB leading to state-of-the-art performance, while still outperforming Llama 70B on generative tasks by more than $20\%$ (Table 2). Scaling even further, GRITLM 8x7B outperforms all openly available models on our generative average. Its embedding performance slightly decreases from GRITLM 7B. This is likely because we had to decrease its embedding batch size from 2048 for GRITLM 7B to only 256 for GRITLM 8x7B due to compute limitations (§3.1). We also train embedding-only and generative-only variants of GRITLM that only use representational or generative instruction tuning but are otherwise equivalent. Benchmarking the embedding-only variant or SGPT BE 5.8B [106] on generative tasks in Table 2 by simply re-adding the language modeling head that was dropped during embedding finetuning leads to around random performance (25.0 is the random baseline on MMLU). Similarly, benchmarking the embedding performance of the generative-only model only leads to a score of 41.2 in Table 1. Thus, joint optimization via the GRIT approach is critical to achieve strong performance for both embedding and generation. We note, however, that with 7 billion parameters GRITLM 7B is significantly more costly to run than many other embedding models in Table 1, such as BGE Large with only 335 million parameters [171]. In addition, GRITLM 7B produces representations of 4096 dimensions, which require $4\times$ more storage than the 1024-dimensional embeddings of BGE Large.

GRITLM matches embedding-only and generative-only variants We find that unifying the two objectives via GRITLM matches both the generative-only and the embedding-only variants. This is similar to observations made for visual models $[178]$ . However, while GRITLM is trained for the same number of steps as the embedding-only and generative-only variants, it requires more compute per training step as it does a forward and backward pass on both embedding and generative data.

Table 3: Reranking (Rerank) using GRITLM as both Bi- and Cross-Encoder. 

<table><tr><td>MTEB DS (↓)</td><td>No Rerank</td><td>Rerank top 10</td></tr><tr><td>ArguAna</td><td>63.24</td><td>64.39</td></tr><tr><td>ClimateFEVER</td><td>30.91</td><td>31.85</td></tr><tr><td>CQADupstack</td><td>49.42</td><td>50.05</td></tr><tr><td>DBPedia</td><td>46.60</td><td>47.82</td></tr><tr><td>FiQA2018</td><td>59.95</td><td>60.39</td></tr><tr><td>FEVER</td><td>82.74</td><td>82.85</td></tr><tr><td>HotpotQA</td><td>79.40</td><td>80.46</td></tr><tr><td>NFCorpus</td><td>40.89</td><td>41.23</td></tr><tr><td>NQ</td><td>70.30</td><td>71.49</td></tr><tr><td>MSMARCO</td><td>41.96</td><td>42.47</td></tr><tr><td>QuoraRetrieval</td><td>89.47</td><td>88.67</td></tr><tr><td>SCIDOCS</td><td>24.41</td><td>24.54</td></tr><tr><td>SciFact</td><td>79.17</td><td>79.28</td></tr><tr><td>TRECCOVID</td><td>74.80</td><td>75.24</td></tr><tr><td>Touche2020</td><td>27.93</td><td>28.41</td></tr><tr><td>Average</td><td>57.4</td><td>57.9</td></tr></table>

Reranking with GRITLM For retrieval tasks, it is common to follow the embedding-based retrieval stage by a reranking stage [115]. In the reranking stage, for each query, the top-k chosen documents are reranked based on a usually more expensive but more performant method. For LLMs, prior work has shown that this can be done by passing each of the k documents together with the query to the model and scoring the pair with log probabilities [106]. Note that this scales quadratically with the number of documents and queries and is thus usually too expensive for the first stage (“Cross-Encoder”). Meanwhile, using embeddings for the first stage is much cheaper as it only requires passing each query and each document once and thus scales linearly (“Bi-Encoder”). More recent work relies on instructions to use LLMs for reranking [147, 98, 121, 122]. While prior work uses separate models for the embedding and reranking stages, GRITLM can be used for both stages due to its unified capabilities. In Table 3, we display the retrieval performance of GRITLM 7B when additionally allowing it to rerank its top 10 documents for each query. For reranking, we use the model's generative capabilities following the permutation generation approach from Sun et al. [147] and reusing their prompt. We find that reranking via the generative capabilities of GRITLM 7B allows it to improve on its own embedding

performance on almost every retrieval dataset. Increasing the top-k documents beyond ten is likely to further improve results, however, at the cost of more compute [106].

Few-shot embedding does not work For generative models it has been well-established that providing in-context examples (“few-shots”, FS) improves performance $[13]$ . However, to the best of our knowledge, there has been no work on in-context learning with embedding models. In Table 4, we benchmark the default 0-shot format versus providing a single few-shot example following the task instruction. We take the few-shot example from the respective evaluation dataset (see §P.2 for the prompts). We find that providing few-shot examples overall worsens performance. While there are small gains among PairClassification tasks (SprintDup. and TwitterURL), these are marginal and inconsistent. For the model trained on MEDI2, we even include few-shot embedding samples in the training data for around 5% of training samples. However, the model seems not to have learned to make good use of the few-shot examples.

Table 4: Few-shot embedding. The 12 MTEB datasets (“DS”) are grouped by the 7 main MTEB tasks in the same order as in Table 1. 

<table><tr><td rowspan="2">Train DS (→)MTEB DS (↓)</td><td colspan="2">E5S</td><td colspan="2">MEDI2</td></tr><tr><td>0 FS</td><td>1 FS</td><td>0 FS</td><td>1 FS</td></tr><tr><td>Banking77</td><td>88.5</td><td>88.3</td><td>88.1</td><td>87.9</td></tr><tr><td>Emotion</td><td>52.8</td><td>51.0</td><td>52.5</td><td>51.9</td></tr><tr><td>IMDB</td><td>95.0</td><td>93.9</td><td>94.3</td><td>92.2</td></tr><tr><td>BiorxivS2S</td><td>39.8</td><td>39.4</td><td>37.6</td><td>37.4</td></tr><tr><td>SprintDup.</td><td>93.0</td><td>94.9</td><td>95.2</td><td>95.7</td></tr><tr><td>TwitterSem</td><td>81.1</td><td>77.9</td><td>76.8</td><td>73.9</td></tr><tr><td>TwitterURL</td><td>87.4</td><td>87.1</td><td>85.9</td><td>86.1</td></tr><tr><td>ArguAna</td><td>63.2</td><td>51.7</td><td>53.5</td><td>53.2</td></tr><tr><td>SCIDOCS</td><td>24.4</td><td>19.7</td><td>25.5</td><td>25.5</td></tr><tr><td>AskUbuntu</td><td>67.3</td><td>64.7</td><td>66.6</td><td>66.0</td></tr><tr><td>STS12</td><td>77.3</td><td>78.0</td><td>76.6</td><td>73.5</td></tr><tr><td>SummEval</td><td>30.4</td><td>29.5</td><td>29.1</td><td>31.5</td></tr></table>

Aligning GRITLM It is common to follow the instruction finetuning stage of generative language models by an alignment tuning stage using methods like PPO $[138]$ , DPO $[128]$ , or KTO $[44]$ (“HALOs” $[44]$ ). We experiment with further finetuning GRITLM using KTO and benchmark the resulting models in Table 5. We train using the binarized UltraFeedback dataset $[30]$ . During this KTO stage, no further embedding training is performed, thus it leads to a slight performance drop on the MTEB average (66.8 to 66.7 and 65.7 to 65.2). However, the average generative performance of the KTO-tuned models is stronger. Notably, AlpacaEval jumps by >10 points for both models. On the more recent Alpaca 2.0 $[39]$ , GritLM-8x7B-KTO has a length-controlled win rate of 18.5 with an average length of 1662 (not depicted). Thus, the KTO-finetuned models may be more useful for use cases where the generative performance is more important. Future work may consider continuing the embedding training during the alignment tuning stage. It may also be possible to develop an alignment tuning method specifically for embedding performance and combine it with generative alignment via KTO.

Table 5: Aligning GRITLM with KTO after GRIT. The upper table depicts embedding performance while the lower depicts generative performance. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc. 12</td><td>Clust. V-Meas. 11</td><td>PairCLF AP 3</td><td>Rerank MAP 4</td><td>Retrieval nDCG 15</td><td>STS Spear. 10</td><td>Summ. Spear. 1</td><td>Avg. 56</td></tr><tr><td>GRITLM 7B</td><td>79.5</td><td>50.6</td><td>87.2</td><td>60.5</td><td>57.4</td><td>83.4</td><td>30.4</td><td>66.8</td></tr><tr><td>GRITLM 7B KTO</td><td>79.6</td><td>50.1</td><td>87.1</td><td>60.5</td><td>57.1</td><td>83.5</td><td>30.5</td><td>66.7</td></tr><tr><td>GRITLM 8x7B</td><td>78.5</td><td>50.1</td><td>85.0</td><td>59.8</td><td>55.1</td><td>83.3</td><td>29.8</td><td>65.7</td></tr><tr><td>GRITLM 8x7B KTO</td><td>78.7</td><td>50.0</td><td>84.4</td><td>59.4</td><td>54.1</td><td>82.5</td><td>30.8</td><td>65.2</td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU 0 FS EM</td><td>GSM8K 8 FS, CoT EM</td><td>BBH 3 FS, CoT EM</td><td>TyDi QA 1 FS, GP F1</td><td colspan="2">HumanEval 0 FS pass@1</td><td>Alpaca 0 FS, 1.0 % Win</td><td>Avg.</td></tr><tr><td>GRITLM 7B</td><td>57.6</td><td>57.5</td><td>54.8</td><td>55.4</td><td colspan="2">32.8</td><td>74.8</td><td>55.5</td></tr><tr><td>GRITLM 7B KTO</td><td>57.6</td><td>57.5</td><td>55.4</td><td>55.8</td><td colspan="2">31.5</td><td>86.7</td><td>57.4</td></tr><tr><td>GRITLM 8x7B</td><td>66.7</td><td>61.5</td><td>70.2</td><td>58.2</td><td colspan="2">53.4</td><td>84.0</td><td>65.7</td></tr><tr><td>GRITLM 8x7B KTO</td><td>66.8</td><td>79.5</td><td>67.1</td><td>31.4</td><td colspan="2">56.8</td><td>95.3</td><td>66.2</td></tr></table>

# 3.3 Ablations

Attention and pooling We train GRITLM starting from a pretrained decoder language model which has been trained with causal attention. Prior work has shown that while embeddings of causal LLMs are competitive, they are outperformed by BERT-like encoders with bidirectional attention at the same number of parameters $[106, 34]$ . This lines up with intuition, as bidirectional attention allows the model to adjust the representation of the first tokens based on information obtained from future tokens. Meanwhile, causal attention only allows information to propagate one way. Thus, for causal attention early tokens may yield poor representations due to a lack of understanding of the entire sample. To counter this issue, we experiment with adapting the model during finetuning to learn to use bidirectional attention. In Table 6 we find that adapting the causally pretrained LLM with bidirectional attention provides the best embedding performance. For fully causal embeddings, we confirm findings from Muennighoff $[106]$ that position-weighted mean pooling (“Wmean”) leads to better embedding performance than taking the embedding of the last token despite recent work finding the opposite $[181, 97]$ . For last token pooling, we follow Zhang et al. $[181]$ and use a special token. We find that adapting the model to be a PrefixLM $[129]$ , whereby the attention over the generative instruction is bidirectional but still causal for the response (“Sample”) worsens performance in contrast to prior work $[163]$ . Thus, we stick with fully causal generation. The unified variant significantly outperforms the embedding-only variants, while underperforming the best generative-only variant. However, once we switched from MEDI to the E5 dataset in later ablations the embedding-only variant matched the unified variant. Meanwhile, the worse generative performance of the unified model was due to a suboptimal loss setting that we fixed in the loss ablations.

Base model The GRITLM approach generalizes to any generative language model, thus we ablate initializing from GPT-J 6B [159], Llama 2 7B or Mistral 7B [69]. Using Mistral 7B leads to the best performance for both embedding and generative tasks. For generative tasks, this is expected as the pretrained Mistral 7B performs the best among the three (Table 2). However, for embedding tasks, GPT-J outperforms Mistral 7B (Table 1). Thus, the embedding performance of a pretrained model is not predictive of its embedding performance after finetuning. Rather, its generative performance appears to be a more reliable indicator of its embedding performance after finetuning.

Generative dataset We benchmark our filtered Tülu 2 introduced in §3.1 [65] with UltraChat [36, 157] and the OpenAssistant version from OctoPack [107, 84, 94]. Using Tülu 2 leads to better performance on every generative task considered (see Appendix F for per-task results). This is likely due to Tülu 2 containing a larger diversity of tasks [65]. Another possible reason is that Tülu 2 may have been carefully tuned on the generative evaluation datasets, as we use largely the same evaluation setup as the creators of Tülu 2 [65].

Embedding dataset We benchmark MEDI [145], a new version of MEDI with better negatives which we build and call MEDI2, and the E5 dataset [162]. While MEDI and MEDI2 always preface instructions with “Represent” (see e.g. Figure 10), the E5 dataset places no constraint on the instruction prefix (see e.g. Figure 11). Thus, when using the E5 dataset the “<embedl>” formatting is critical to tell the model that it will be subject to the representation loss, not the generative loss (Figure 3). Further, MEDI and MEDI2 always contain instructions for both queries and documents, which we refer to as two-sided instructions. Meanwhile, the E5 dataset uses one-sided instructions for asymmetric datasets [106], whereby the documents receive no instructions, only the queries. The advantage of not using document instructions is that the document corpus can be encoded once and then cached and reused across a variety of tasks. During training on E5, symmetric tasks are also in a one-sided setting, but we still evaluate them in the two-sided format. This should not be a problem as the cosine similarity function we use during training is transitive: if sentence A with instruction is similar to sentence B without instruction, and sentence B without instruction is similar to sentence C with instruction, then we can confidently say that sentence A with instruction is also similar to sentence C with instruction. As depicted in Table 6, using the E5 dataset performs best by a wide margin. An inspection of samples, suggests that this is likely due to its superior hard negatives and diversity of tasks generated by GPT-4 (Appendix O). For our final runs with the E5 dataset, we additionally add scientific data ( $§3.1$ ).

<table><tr><td>Attention Emb Instruction Sample</td><td>Attention Gen Instruction Sample</td><td>Pooling</td><td>Emb</td><td>Gen</td></tr><tr><td colspan="5">Embedding Only</td></tr><tr><td>Causal</td><td></td><td>Wmean</td><td>60.0</td><td>-</td></tr><tr><td>Causal Bidirectional</td><td></td><td>Mean</td><td>61.0</td><td>-</td></tr><tr><td>Bidirectional</td><td></td><td>Mean</td><td>61.8</td><td>-</td></tr><tr><td colspan="5">Generative Only</td></tr><tr><td></td><td>Causal</td><td></td><td>-</td><td>55.2</td></tr><tr><td></td><td>Bidirectional Causal</td><td></td><td>-</td><td>50.7</td></tr><tr><td colspan="5">Unified</td></tr><tr><td>Causal</td><td>Causal</td><td>Last token</td><td>61.2</td><td>53.0</td></tr><tr><td>Causal</td><td>Causal</td><td>Wmean</td><td>62.8</td><td>52.8</td></tr><tr><td>Bidirectional</td><td>Causal</td><td>Mean</td><td>64.0</td><td>52.9</td></tr></table>

(a) Attention and pooling ablations. Wmean is position-weighted mean pooling [106].

<table><tr><td>Variant</td><td>Emb</td><td>Gen</td></tr><tr><td>Mistral 7B</td><td>54.6</td><td>22.4</td></tr><tr><td>Llama 2 7B</td><td>48.2</td><td>20.8</td></tr><tr><td>GPT-J 6B</td><td>51.9</td><td>14.0</td></tr></table>

(b) Base model

<table><tr><td>Dataset</td><td>Emb</td></tr><tr><td>MEDI</td><td>64.0</td></tr><tr><td>MEDI2</td><td>64.7</td></tr><tr><td>E5</td><td>66.0</td></tr></table>

(c) Embedding dataset

<table><tr><td>Dataset</td><td>Gen</td></tr><tr><td>Tülu 2</td><td>55.2</td></tr><tr><td>OASST</td><td>37.7</td></tr><tr><td>UltraChat</td><td>47.4</td></tr></table>

(d) Generative dataset

<table><tr><td>Variant</td><td>Emb</td><td>Gen</td></tr><tr><td>No head</td><td>62.7</td><td>49.2</td></tr><tr><td>-&gt; 1024</td><td>62.1</td><td>48.0</td></tr></table>

(e) Embedding head

<table><tr><td>BS Emb:Gen</td><td>Emb</td><td>Gen</td></tr><tr><td>256:256</td><td>63.2</td><td>53.4</td></tr><tr><td>4096:256</td><td>64.2</td><td>53.3</td></tr></table>

(f) Batch size (BS)

<table><tr><td>Precision</td><td>Emb</td><td>Gen</td></tr><tr><td>FP32</td><td>66.3</td><td>52.4</td></tr><tr><td>BF16*</td><td>66.5</td><td>55.0</td></tr></table>

(g) Precision

<table><tr><td>IBN origin</td><td>Emb</td><td>Gen</td></tr><tr><td>Any dataset</td><td>66.0</td><td>50.9</td></tr><tr><td>Same dataset</td><td>66.0</td><td>51.1</td></tr></table>

(h) In-batch negatives (IBN)

<table><tr><td>Format</td><td>Gen</td></tr><tr><td>Tülu 2</td><td>55.2</td></tr><tr><td>Zephyr β</td><td>49.0</td></tr></table>

(i) Format

<table><tr><td>Tokens</td><td>Emb</td><td>Gen</td></tr><tr><td>512</td><td>64.1</td><td>52.2</td></tr><tr><td>2048</td><td>64.7</td><td>53.8</td></tr></table>

(j) Emb training max tokens

<table><tr><td>Gen loss type</td><td> $\mathcal{L}_{\text{Rep}}/\mathcal{L}_{\text{Gen}}$ </td><td>Emb</td><td>Gen</td></tr><tr><td>Token</td><td>2.4</td><td>66.1</td><td>54.4</td></tr><tr><td>Token</td><td>6.0</td><td>66.5</td><td>55.0</td></tr><tr><td>Mix (32 -&gt; 8)</td><td>4.1</td><td>66.7</td><td>55.4</td></tr></table>

<table><tr><td>Gen loss type</td><td>AlpacaEval</td></tr><tr><td>Mix (4 -&gt; 64)</td><td>67.6</td></tr><tr><td>Mix (32 -&gt; 8)</td><td>74.7</td></tr></table>

(k) Loss ablations. $L_{Rep}/L_{Gen}$ is the loss ratio of the 1st step adjusted via $\lambda_{Rep}$ and $\lambda_{Gen}$ . Mix refers to mixing sample and token level loss, e.g. (32->8) is token level loss across 32 samples and then sample level loss across 8 sub-batches for a total batch size of 256.   
Table 6: GRIT ablations. Emb corresponds to the MTEB average, while Gen corresponds to the average across generative tasks (Appendix E). The embedding head variant “->1024” corresponds to down-projecting the final hidden state with a linear layer from 4096 to 1024 dimensions, only for embedding tasks. BF16\* means that some computations are still in FP32 as explained in §3.3. The setting chosen for GRITLM is bold. Once an ablation was successful, we adopted its setting, thus the bold performance slightly varies from one table to the next. For example, the base model ablation (b) is done for just 100 hundred steps with sub-optimal formatting. Full results are in Appendix F.

Embedding head The cost of caching the embeddings of a large document corpus is directly proportional to the embedding dimensionality. To minimize such costs, we experiment with adding an embedding head consisting of a linear layer with activation that down-projects the embedding $[113, 106]$ . This layer is only used for embedding tasks. Down-projecting the embeddings four-fold (from 4096 to 1024) leads to an embedding performance decrease of around 1%. This may be acceptable for certain use cases where the saved storage is more important. However, for our final model, we do not use such a head to keep it simple and achieve maximum performance. Search techniques $[3, 73, 37]$ or dimensionality reduction techniques such as Principal Component Analysis still allow for reducing the embedding dimension of our final model post-training while maintaining most of the performance. Similar to the storage cost-performance trade-off we explore here, we hypothesize that there is a speed/cost-performance trade-off with taking the embedding from different layers of our model. For example, we could train using the embedding after half the layers of the model, thus speeding up the embedding model by 50% while likely only incurring a small drop in embedding performance.

Batch size Due to the utilization of in-batch negatives for contrastive training ( $\S2$ ), a larger batch size provides a more accurate gradient. Thus, scaling up the batch size is a key ingredient in most well-performing embedding models $[171, 161]$ . We experiment with scaling up the embedding batch size to 4096 while keeping it at 256 for generative data. This leads to a 1.0 gain on the embedding average while generative performance remains stable. Especially the 15 retrieval datasets that are part of the embedding average benefit from the increase in batch size (see Table 20). For our final model, we use a batch size of 2048 for embedding and 256 for generative data.

Precision The parameters of the Mistral 7B model are in bfloat16 (BF16) precision as it was pretrained in this format. We experiment with finetuning it with float32 (FP32) precision versus keeping the BF16 format and training with mixed precision. FP32 training is more costly, however, the additional precision may result in a better model. Our intuition is that more precision is important for embedding but not as much for generation. This is because while for generative tasks evaluated greedily, the model output is a discretionary argmax over the predictions of the language modeling head, for embedding tasks it is a continuous representation. Thus, small differences due to a lack of precision may not change the model's generation but will affect its representation. Hence, for embedding tasks, we always cast the hidden states to FP32 during the pooling operation and keep them this way for the similarity computation. Not keeping them in FP32 after pooling worsens performance slightly, but may be necessary for cheap storage (see Appendix L). In addition, some operations such as layer normalization [7] are also performed in FP32 even for BF16 training due to PyTorch autocast [184]. In Table 6, we find that there is no benefit from doing even more computations in FP32 besides the ones listed above. Thus, we train and evaluate all our other models in BF16 mixed precision to speed up training and inference.

In-batch negatives We always use in-batch negatives for embedding training ( $\S2$ ), however, we ablate whether or not they come from the same dataset. We hypothesize that making them all come from the same dataset leads to better negatives as the model needs to distinguish them based on more nuanced differences. In practice, we find that the average embedding performance remains around the same. However, we notice a 1.3 jump on the 15-dataset Retrieval average (Table 22). Thus, we stick with the variant where in-batch negatives stem from the same dataset.

Format Our chosen format is depicted in Figure 3, which is equivalent to Tülu 2 [65] for generative tasks. We also benchmark the Zephyr $\beta$ format [157], which has an additional end-of-sequence token (“</s>”) after each user utterance. We find that it performs worse on generative tasks. The additional end-of-sequence after the user utterance increases the likelihood of the model generating another end-of-sequence token earlier than necessary. This significantly harms HumanEvalSynthesize performance and slightly reduces AlpacaEval, where long generations can be critical (see Appendix F for task-specific performance).

Max tokens Our base model, Mistral 7B, can handle sequences of arbitrary length due to its sliding window attention [69]. As finetuning with longer sequences is more expensive we ablate its benefits. We compare training with a maximum token limit of 512 versus 2048 for embedding documents. For embedding queries, we always use 256, and for generative data, we always use 2048. We find that increasing the embedding document sequence length during training slightly boosts performance on both embedding and generation even though we still evaluate embedding tasks with 512. This

boost likely comes from our training data containing many documents beyond 512 tokens, which need to be truncated if the maximum sequence length is 512. Such truncation may remove the critical parts that make two texts a positive or a negative contrastive pair and thus hinder learning. As our embedding evaluation (MTEB) contains few documents longer than 512 tokens there is little truncation happening at evaluation $[109, 59, 58]$ . Note that just like their base models, our final models GRITLM 7B and GRITLM 8x7B can produce embeddings for sequences of arbitrary length. However, due to a lack of benchmarks, we do not know how well the embeddings of our models perform for input sequences longer than 512 tokens.

Loss ablations As detailed in §2, we experiment with both token and sample level generative loss. Further, we ablate the representation and generative loss weights, $\lambda_{Rep}$ and $\lambda_{Gen}$ . For the unified visual model CoCa, the authors find that giving a weight of 2 to generation and 1 to embedding boosts performance on both streams [178]. However, rather than the weights, we argue that the loss ratio, $L_{Rep}/L_{Gen}$ , is of more interest as it reveals which objective has a larger impact on the optimization of the model. We maintain a ratio of $L_{Rep}/L_{Gen} > 1$ i.e. giving more weight to the representation loss. This is because the model has already been pretrained with the generative loss, thus we expect less additional generative training to be necessary. Meanwhile, the contrastive loss for embedding data is new to the model, thus we expect more learning to be needed on the embedding side. Further, the embedding loss drops off extremely quickly as can be seen in the loss graphs in Appendix C. Thus, even though the representation loss has a higher weight at the start, throughout training they have very similar weights with both hovering around a loss of 1.0. We find that mixing sample and token level generative loss leads to the best performance by a small margin. As expected in §2, token level loss to some degree is critical for good performance on AlpacaEval. For “Mix (4 -> 64)” token level loss is applied across only 4 samples and then sample level loss across 64 sub-batches, which leads to a 7-point drop in AlpacaEval performance. This drop is accompanied by a decrease in median AlpacaEval generation length from 941 to 865. Thus, token level loss across many samples is critical to maintaining long generations, which directly impacts the AlpacaEval score.

# 4 RAG with GRIT

Method By unifying embedding and generation, GRITLM simplifies Retrieval-Augmented Generation (RAG). Figure 4 displays how forward passes can be reduced by caching. Specifically, we break down the caching alternatives into:

(a) Query Caching: In traditional RAG, the query needs to be passed both through the embedding model and later through the generative model. In Query Caching, we cache the key-value states from the embedding forward pass and reuse them for the generative pass, exploiting the property that both are the same model: GRITLM. Thus, we save compute equivalent to one forward pass of the query. Equivalently, we can also perform the generative forward pass over the query first and use its representation to retrieve the document on the fly (depicted in Figure 4). Note that Query Caching can be completely equivalent to RAG if the query is placed at the beginning of the prompt such that it only attends to itself through causal attention.

(b) Doc Caching: Here we cache the documents, D. When the index is created, we also save the key-value states of every document and add them to the index. Thus, the index consists of the document embeddings and key-value states. Note that the computational cost of creating the index remains the same as the key-value states have to be computed even if only embeddings are desired. At inference, we still retrieve based on embedding similarity but the index returns the key-value states instead of the text passage. These key-value states are then provided to the model to avoid having to recompute them. This effectively saves a forward pass for every in-context document at inference. However, this method increases the necessary storage. While the text passages no longer need to be stored, the key-value states now need to be stored and they usually require more storage depending on the model. We note that Document Caching also works for models other than GRITLM. However, for such models, one needs to pass all documents through the generation model ahead of time, thus increasing the cost of creating the index. To maintain equivalence with RAG, the document should be at the beginning of the prompt for Document Caching (opposite of Query Caching).

(b) Query-Doc Caching / Doc-Query Caching: We can also combine Query Caching and Doc Caching to save even more inference costs. However, combining them inevitably leads to discrepancies compared to RAG, as in traditional RAG either the query or the document is conditioned on the other one. Meanwhile, if both are cached then they are not conditioned on one another via the

![](images/2fe8eb32509566c0d7f0049a8389076b83b72fcda1e24f0ad926adfe0994b298.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Traditional RAG
        A["To slow down your speed of aging, you can..."]
        B["Generative Model"]
        C["How to prevent aging?"]
        D["Technological and lifestyle factors may influence an individual's longevity. Cellular reprogramming..."]
    end

    subgraph Query Caching
        E["Index"] --> F["GritLM"]
        G["1st Cache: Reuse query representation for retrieval"] --> H["Index"]
        I["How to prevent aging?"] --> J["Technological and lifestyle factors may influence an individual's longevity. Cellular reprogramming..."]
        K["To slow down your speed of aging, you can..."]
    end

    subgraph Query-Doc Caching
        L["Index"] --> M["GritLM"]
        N["1st Cache: Reuse query representation for retrieval"] --> O["Index"]
        P["To slow down your speed of aging, you can..."]
        Q["2nd Cache: Reuse document key-value states for generation"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    N --> O
    O --> P
    P --> Q
```
</details>

Figure 4: RAG with GRIT. Left: Traditional Retrieval-Augmented Generation (RAG) relies on a separate embedding model and generative model. Right: GRITLM simplifies RAG as it handles both embedding and generation. Query Caching removes the duplicate forward pass of the query by reusing its representation. Query-Doc Caching also removes the forward pass on the document during inference, as the cached index also stores the document key-value states.

self-attention mechanism. We refer to Query-Doc Caching if the query is followed by the document in the prompt and to Doc-Query Caching if the document comes first.

Table 7: RAG benchmarking on Natural Questions with GRITLM 7B. For RAG, the retrieved context is simply placed in the context of the language model in contrast to our caching alternatives (Figure 4). We measure CPU and GPU latencies on an “Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz” and one “NVIDIA H100 80GB HBM3”, respectively. Sample A has a query of 1 token and a document of 4000 tokens, and sample B is the inverse. For each approach, we generate 16 tokens. Storage consists of the index and passages, except for Doc Caching variants where it is the index and key-value states. The index is stored in float32, while key-value states are stored in bfloat16. 

<table><tr><td rowspan="2"></td><td rowspan="2">Match(0-shot,↑)</td><td colspan="2">CPU Latency (s,↓)</td><td colspan="2">GPU Latency (s,↓)</td><td rowspan="2">Storage (↓)</td></tr><tr><td>Sample A</td><td>Sample B</td><td>Sample A</td><td>Sample B</td></tr><tr><td>No RAG</td><td>21.00</td><td>4.3 ± 0.36</td><td>13.69 ± 1.0</td><td>0.24 ± 0.04</td><td>0.38 ± 0.04</td><td>0GB</td></tr><tr><td colspan="7">Query then document prompt</td></tr><tr><td>RAG</td><td>30.50</td><td>11.64 ± 0.74</td><td>14.88 ± 0.87</td><td>0.39 ± 0.02</td><td>0.40 ± 0.02</td><td>43GB</td></tr><tr><td>Query Caching</td><td>25.46</td><td>18.30 ± 0.76</td><td>6.87 ± 0.89</td><td>0.44 ± 0.03</td><td> $\underline{0.27 \pm 0.02}$ </td><td>43GB</td></tr><tr><td>Query-Doc Caching</td><td>21.63</td><td> $\underline{5.12 \pm 0.23}$ </td><td> $\underline{6.62 \pm 0.97}$ </td><td> $\underline{0.27 \pm 0.03}$ </td><td>0.29 ± 0.01</td><td>30TB</td></tr><tr><td colspan="7">Document then query prompt</td></tr><tr><td>RAG</td><td>30.47</td><td>14.18 ± 1.01</td><td>15.33 ± 0.87</td><td>0.39 ± 0.01</td><td>0.4 ± 0.01</td><td>43GB</td></tr><tr><td>Doc Caching</td><td>33.38</td><td>5.25 ± 0.34</td><td>23.23 ± 1.05</td><td> $\underline{0.27 \pm 0.03}$ </td><td>0.45 ± 0.02</td><td>30TB</td></tr><tr><td>Doc-Query Caching</td><td>18.39</td><td> $\underline{5.23 \pm 0.37}$ </td><td> $\underline{6.41 \pm 0.96}$ </td><td> $\underline{0.26 \pm 0.03}$ </td><td> $\underline{0.27 \pm 0.02}$ </td><td>30TB</td></tr></table>

Setup We benchmark the different caching variants using data from Natural Questions $[83]$ . Our implementation is adapted from Izacard et al. $[67]$ , however, we use a significantly smaller index of only 2,681,468 documents stemming from the BEIR NQ corpus $[154]$ . We score models using the match score, whereby we check if any of the correct answers are anywhere in the generation. Prior

![](images/0dd117118430be4bd41cc0530cf37cd3820effb60d5a749a812e6d30cd865ef4.jpg)  
No Retrieval Query Caching Query-Doc Caching Traditional RAG Doc Caching Doc-Query Caching

Figure 5: Inference latency of RAG with GRITLM 7B. When benchmarking scaling query length (left), document length is fixed at 1, whereas query length is fixed at 1 when scaling document length (right). In addition to the query/doc lengths, the formatting and prompt take up around 40 tokens. We visualize the standard deviation across 100 runs as the shaded area. For each approach, we generate 16 tokens. We measure CPU and GPU latencies on an “Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz” and one “NVIDIA H100 80GB HBM3”, respectively.

work usually uses exact match, whereby they check if the generation exactly matches the answer. However, as our model is more chatty, it tends to answer in a few sentences and thus exact match often fails to give credit to correct answers. Inspecting the first 20 samples of the “No RAG” baseline, we find that exact match leads to 4 false negatives that are correctly credited by the match metric. We did not find any false positives from choosing the match metric in those samples. We do not use any instructions for embedding, solely the format as presented in Figure 3.

Performance As depicted in Table 7, RAG performs better than the “No RAG” baseline where the model is not provided any context. This validates that despite its small size compared to prior work [92], our index is still valuable. While Query and Doc Caching can theoretically lead to the exact same performance as RAG, we experience differences stemming from two reasons: 1) Attention: Our model is trained to embed with bidirectional attention ( $\S2$ ) and thus we use bidirectional attention when embedding query or document. Meanwhile, the generative model expects causal key-value states. In the Query-Doc/Doc-Query setup, there is an additional mismatch in either the documents or the queries not having attended to the other one, as both need to be embedded and cached separately.

2) Formatting: The query is formatted in the embedding format as depicted in Figure 3, which the model has not seen in conjunction with a generative task during training. This could further lead to a performance drop. Due to 1) and 2), Query Caching leads to a performance drop compared to traditional RAG. However, the Query Caching performance of 25.46 is still better than not using RAG, thus it comes down to a speed-performance trade-off. Formatting the RAG baseline using the embedding format (Figure 3) reduces its score from 30.50 to 29.36 (not depicted), thus the additional four-point discrepancy of Query Caching and the majority of the damage is because of the attention issue. Meanwhile, Doc Caching slightly improves performance resulting in the best match score among all methods considered. This is possibly because, unlike the query, the document does not need to be as thoroughly understood, and skimming it may suffice. Thus, the slightly corrupted key-value states do not result in a performance drop. Query-Doc and Doc-Query Caching only

perform near the “No RAG” baseline in our experiments, which may limit their usefulness in practice. This is likely caused by the additional attention mismatch that they introduce. This issue as well as the formatting issue could likely be solved by an additional RAG finetuning stage on top of GRITLM, which we leave to future work.

Latency In Figure 5, we show how caching leads to significant speed-ups over RAG on both CPUs and GPUs for long sequences. If only 250 tokens are cached, however, we find the speed-up to be negligible. In Table 7, we display that for 4000 tokens, Query Caching is 54% and 33% faster on CPUs and GPUs, respectively (Sample B). For Doc Caching it is 63% and 31% (Sample A). If going beyond 4000 tokens the speed-ups will be even larger. However, for the opposite samples in Table 7 speed remains around the same. This is because while for Sample A, Doc Caching caches 4000 tokens, for Sample B it caches only 1 token, which does not provide any speed-up. Thus, Doc Caching should be used when documents are expected to be very long, while Query Caching should be used when queries are expected to be very long. In a production setting, a simple input length check could switch from one caching mode to the other. As is the case in Table 7, caching can match or even be faster than not using retrieval at all (“No RAG”). This could be due to the embedding forward pass not using the language modeling head. For Query Caching, the language modeling head is only used for the tokens that are generated, while for “RAG” and “No RAG” it is used for the entire input. The matrix multiplication with the language modeling head is computationally expensive due to its high dimensionality, which could cause the slower speed of the no retrieval baseline. Query-Doc Caching and Doc-Query Caching cache both documents and queries and thus lead to major speed-ups for both Sample A and Sample B in Table 7. Overall, speed-ups are larger on CPUs, as GPUs can process the entire sequence in parallel, thus the advantage of caching parts of it is smaller. We also note that our RAG baseline uses our 7B parameter model for both the embedding and generative model but without caching. In practice, it is often common to have an embedding model that is much smaller and cheaper than the generative model. Nonetheless, as caching with GRITLM-7B approaches the No RAG latency in Table 7, we still expect it to be faster than setups with smaller embedding models for long sequences. In addition, it would lead to significantly better performance in that case due to the state-of-the-art retrieval performance of GRITLM.

Storage In most RAG setups the embeddings of all documents are computed ahead of time and stored to be later used at inference. This is referred to as the index. In traditional RAG, the documents themselves still need to be stored, as the index is only used for finding the document ID, which is then used to fetch the document text and pass it to the generative model. For Doc Caching variants documents no longer need to be stored, however, the key-value states need to be stored together with the index. The key-value states take up a lot of storage, as for each batch they consist of two tensors of shape (batch size, number of heads, sequence length, dimension per head). For our 2,681,468 documents and the 7-billion parameter GRITLM model, this leads to around 30TB of key-value states. However, unlike the index, the key-value states can be fully offloaded to disk and do not need to be kept in memory. Once the document ID has been determined via the index, the corresponding key-value state can be simply loaded from disk. For a single sample, this corresponds to loading around 12.5MB of key-value states into memory.

# 5 Discussion

Further unification To the best of our knowledge, GRITLM is the first model to unify text embedding and generation, and thus all text-based language problems, into a single model at strong performance. However, many adjacent directions remain to be improved or unified. (a) Multilinguality: Our model is also capable of embedding and generation in non-English languages as seen in its TyDi QA performance (Table 2). However, major performance gains on non-English tasks are likely possible through both data [110, 176] and architecture changes [15, 48, 42] targeting multilinguality. (b) Multimodality: Many embedding and generative problems are not purely text-based, such as joint embedding of images and text [124], generative image captioning [63], image-text pair classification [105, 80] or speech versions of every text problem [76]. It remains to be explored whether they can be as easily unified as text embedding and generation in this work.

Why does GRIT work? GRIT unifies embedding and generative tasks into a single model at no performance loss on either one, which may seem surprising. When the embedding dataset is

MEDI2, we show that embedding performance even improves once the generative objective is added compared to an otherwise equivalent embedding-only model ( $\S3.3$ ). We think that our results confirm that generative language modeling and text embeddings are two sides of the same coin. Both tasks require a model to have a deep understanding of natural language and only differ in the way that understanding is expressed. Possibly, our unified model contains a small number of parameters that act as a switch to make the final representations either useful for mean pooling and subsequent embedding tasks or primed for the language modeling head and subsequent generative tasks. We are excited about future work exploring what is happening inside of GRITLM. To support such research, we release all our work freely.

Optimizing RAG with GRITLM RAG and the caching variants we have presented in this work operate on a frozen language model. Meanwhile, there has been extensive work on optimizing a generative model specifically for interaction with a retrieval system $[53, 187, 4]$ . These works commonly optimize only the retriever $[140]$ or only the reader $[12, 174, 6, 95]$ . However, recent work has shown that jointly optimizing both models leads to the best performance $[92]$ . With its state-of-the-art retrieval and generative performance, GRITLM can act as both the retriever and reader in a single model. Thus, optimizing either one also changes the parameters of the other. This has the potential to significantly simplify the joint optimization of the retriever and reader. For example, it may suffice to only use the next-token objective (Equation 2) to penalize the retriever for providing irrelevant context and at the same time the reader for poor use of the given context. This is in contrast to separate models and objective functions used in Lin et al. $[92]$ .

# 6 Related Work

The story of text embedding and text generation has been a story of unification.

Embedding Models used to focus on word representations $[119, 100]$ that struggled generalizing to entire sentences or passages $[27]$ . InferSent $[28]$ , SBERT $[132]$ and similar models $[114, 113]$ emerged that handle both the embedding of words and sentences at good quality by considering context when present. However, for strong performance, they require separate models for symmetric and asymmetric tasks $[109, 106]$ . Symmetric embedding tasks are ones where the query and document are expected to come from the same distribution, such as STS. Meanwhile, for asymmetric tasks, they come from different distributions and as such could have very different sequence lengths like in retrieval. For example, the MTEB benchmark $[109]$ revealed that SentT5 $[114]$ only performs well at symmetric tasks, while GTR $[113]$ only at asymmetric tasks despite both using T5 $[129]$ as their base model. Recent embedding models have been able to unify symmetric and asymmetric tasks into a single model by differentiating them in the prompt $[171, 161]$ . Further, including detailed instructions in the prompt has allowed unifying practically any embedding task into a single model $[145]$ .

Generative Models used to be tailored to a single task, such as translation $[148]$ or question answering $[175]$ . McCann et al. $[99]$ cast multiple generative tasks as question answering to unify them within a single model, however, performance was still limited and it did not generalize to arbitrary tasks. Large-scale self-supervised pretraining has enabled the use of a single large language model (LLM) for practically any generative task $[13, 22, 127, 11, 136, 55, 143, 1, 88]$ . However, using an LLM without careful prompting often leads to poor performance $[133, 102]$ . Finetuning LLMs on instructions has emerged as a method to significantly ease the usage of the models to apply them to any generative task with strong results $[166, 135, 101, 165, 103, 110, 66, 189, 142, 186]$ .

The two streams of embedding and generative models have respectively been unified into a single model that handles any task within its stream. Unifying the two streams into a single model that handles any task both for embedding and generation is the natural next step toward a general multi-task model. Besides generation, LLMs have also shown promise for text embeddings $[106, 112, 71, 90, 89]$ . SGPT $[106]$ was an early work in that direction. SGPT only changes 0.01% of the parameters of a large language model via BitFit $[179]$ to adapt it to produce well-performing embeddings. Thus, one only needs to change this small amount of parameters to switch from one stream to the other. However, SGPT still required separate asymmetric and symmetric models and did not consider the full breadth of embedding tasks. GRITLM addresses these deficiencies. GRITLM does not require switching out biases, leverages instructions to handle asymmetric or symmetric use cases, and considers the full breadth of embedding and generative tasks.

# 7 Conclusion

We present GRIT to unify text embedding and generation, and thus all text-based language problems, into a single model: GRITLM. GRITLM 7B achieves state-of-the-art performance on the Massive Text Embedding Benchmark among open models, while at the same time beating all generative models up to its size. Notably, it matches the performance of otherwise equivalent embedding-only and generative-only variants allowing us to unify the two streams at no performance loss. By adding only 5B parameters at inference, GRITLM 8x7B is the best open generative language model among the many we have tried including much larger models based on Llama 2 with 70B parameters. Unlike the other generative models, GRITLM 8x7B also boasts very strong embedding performance thanks to the GRIT approach. Due to its unified capabilities, GRITLM can be used as both the Bi-Encoder and Cross-Encoder in a reranking pipeline leading to performance improvements on 15 out of 16 retrieval datasets. Further, we conduct extensive ablations uncovering key insights for researchers of both embedding and generative models: causal language models for embedding should be finetuned with bidirectional attention and mean pooling, embedding performance of language models before finetuning is not predictive of embedding performance after finetuning, finetuning embedding models in BF16 mixed precision can match FP32 finetuning, pooling and similarity computations should be performed in FP32 precision during BF16 finetuning, generative models should be instruction-tuned with some form of token level loss, etc. Finally, we show that GRIT simplifies the field using the examples of reranking and RAG. For reranking, we are able to improve retrieval performance by around $10\%$ by reusing GRITLM as reranker instead of having to rely on a separate model. For RAG, by unifying the retriever and reader into a single model, GRITLM allows caching operations leading to inference speed-ups of $>60\%$ for long sequences at no performance loss with GRIT Doc Caching. We believe GRIT paves the way for a paradigm shift in language modeling, where embedding and generation seamlessly coexist in a single model. As such, we highlight the various limitations of this work and point the community to potential future research in Appendix R.

# Acknowledgments and Disclosure of Funding

We thank everyone at Contextual AI, the University of Hong Kong, and Microsoft for their support. We are grateful to Hamish Ivison and Yizhong Wang for help with using the Tülu 2 repository. We thank Akari Asai, Alexander M. Rush, Brian Lester, Colin Raffel, Danqi Chen, Derek Tam, John X. Morris, Haokun Liu, Hong Liu, Mengzhou Xia, Michael Matena, Muqeeth Mohammed, Omar Khattab, Shayne Longpre, Tengyu Ma, Teven Le Scao, and Tianyu Gao for discussions and feedback.

# References

[1] Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. 2023. SantaCoder: don't reach for the stars!   
[2] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. PaLM 2 Technical Report.   
[3] Sunil Arya, David M Mount, Nathan S Netanyahu, Ruth Silverman, and Angela Y Wu. 1998. An optimal algorithm for approximate nearest neighbor searching fixed dimensions.   
[4] Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. 2023. Retrieval-based Language Models and Applications.   
[5] Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, and Wen tau Yih. 2022. Task-aware Retrieval with Instructions.   
[6] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.   
[7] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer Normalization.   
[8] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.   
[9] Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020. Longformer: The Long-Document Transformer.   
[10] Loubna Ben Allal, Niklas Muennighoff, Logesh Kumar Umapathi, Ben Lipkin, and Leandro von Werra. 2022. A framework for the evaluation of code generation models.   
[11] Teven Le Scao BigScience Workshop, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, et al. 2023. BLOOM: A 176B-Parameter Open-Access Multilingual Language Model.   
[12] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022. Improving language models by retrieving from trillions of tokens.   
[13] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language Models are Few-Shot Learners.   
[14] Isabel Cachola, Kyle Lo, Arman Cohan, and Daniel S. Weld. 2020. TLDR: Extreme Summarization of Scientific Documents.   
[15] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.   
[16] Lingjiao Chen, Matei Zaharia, and James Zou. 2023. How is ChatGPT's behavior changing over time?   
[17] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, et al. 2021. Evaluating Large Language Models Trained on Code.   
[18] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A Simple Framework for Contrastive Learning of Visual Representations.

[19] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollar, and C. Lawrence Zitnick. 2015. Microsoft COCO Captions: Data Collection and Evaluation Server.   
[20] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. 2019. Generating Long Sequences with Sparse Transformers.   
[21] Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. 2021. Unifying Vision-and-Language Tasks via Text Generation.   
[22] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. PaLM: Scaling Language Modeling with Pathways.   
[23] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, et al. 2022. Scaling Instruction-Finetuned Language Models.   
[24] Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020. TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages.   
[25] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems.   
[26] Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S. Weld. 2020. SPECTER: Document-level Representation Learning using Citation-informed Transformers.   
[27] Alexis Conneau and Douwe Kiela. 2018. SentEval: An Evaluation Toolkit for Universal Sentence Representations.   
[28] Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes. 2018. Supervised Learning of Universal Sentence Representations from Natural Language Inference Data.   
[29] William Coster and David Kauchak. 2011. Simple English Wikipedia: A New Text Simplification Task.   
[30] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. 2023. UltraFeedback: Boosting Language Models with High-quality Feedback.   
[31] Tri Dao. 2023. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.   
[32] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.   
[33] DataCanary, hilfialkaff, Meg Risdal Lili Jiang, Nikhil Dandekar, and tomtung. 2017. Quora Question Pairs.   
[34] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pretraining of Deep Bidirectional Transformers for Language Understanding.   
[35] Kaustubh D. Dhole, Varun Gangal, Sebastian Gehrmann, Aadesh Gupta, Zhenhao Li, Saad Mahamood, Abinaya Mahendiran, Simon Mille, Ashish Shrivastava, Samson Tan, et al. 2022. NL-Augmenter: A Framework for Task-Sensitive Natural Language Augmentation.   
[36] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. 2023. Enhancing Chat Language Models by Scaling High-quality Instructional Conversations.   
[37] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024. The Faiss library.

[38] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2021. All nlp tasks are generation tasks: A general pretraining framework.   
[39] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. 2024. Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. arXiv preprint arXiv:2404.04475.   
[40] Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback.   
[41] Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur G"uney, Volkan Cirik, and Kyunghyun Cho. 2017. SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine.   
[42] Paul-Ambroise Duquenne, Holger Schwenk, and Benoît Sagot. 2023. SONAR: Sentence-Level Multimodal and Language-Agnostic Representations.   
[43] Hady ElSahar, Pavlos Vougiouklis, Arslen Remaci, Christophe Gravier, Jonathon S. Hare, Frédérique Laforest, and Elena Paslaru Bontas Simperl. 2018. T-REx: A Large Scale Alignment of Natural Language with Knowledge Base Triples.   
[44] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. KTO: Model Alignment as Prospect Theoretic Optimization.   
[45] Alexander R. Fabbri, Wojciech Kryściński, Bryan McCann, Caiming Xiong, Richard Socher, and Dragomir Radev. 2021. SummEval: Re-evaluating Summarization Evaluation.   
[46] Anthony Fader, Luke Zettlemoyer, and Oren Etzioni. 2014. Open question answering over curated and extracted knowledge bases.   
[47] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: Long Form Question Answering.   
[48] Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022. Language-agnostic BERT Sentence Embedding.   
[49] Katja Filippova and Yasemin Altun. 2013. Overcoming the Lack of Parallel Data in Sentence Compression.   
[50] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2021. A framework for few-shot language model evaluation.   
[51] Luyu Gao, Yunyi Zhang, Jiawei Han, and Jamie Callan. 2021. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup.   
[52] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2022. SimCSE: Simple Contrastive Learning of Sentence Embeddings.   
[53] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation for Large Language Models: A Survey.   
[54] David Graff, Junbo Kong, Ke Chen, and Kazuaki Maeda. 2003. English gigaword.   
[55] Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi. 2024. OLMo: Accelerating the Science of Language Models.

[56] Mansi Gupta, Nitish Kulkarni, Raghuveer Chanda, Anirudha Rayasam, and Zachary C Lipton. 2019. AmazonQA: A Review-Based Question Answering Task.   
[57] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. REALM: Retrieval-Augmented Language Model Pre-Training.   
[58] Michael Günther, Louis Milliken, Jonathan Geuter, Georgios Mastrapas, Bo Wang, and Han Xiao. 2023. Jina Embeddings: A Novel Set of High-Performance Sentence Embedding Models.   
[59] Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram, Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, Maximilian Werk, Nan Wang, and Han Xiao. 2024. Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents.   
[60] Felix Hamborg, Norman Meuschke, Corinna Breitinger, and Bela Gipp. 2017. news-please - A Generic News Crawler and Extractor.   
[61] Dan Hendrycks, Nicholas Carlini, John Schulman, and Jacob Steinhardt. 2022. Unsolved Problems in ML Safety.   
[62] Christopher Hidey and Kathy McKeown. 2016. Identifying Causal Relations Using Parallel Wikipedia Articles.   
[63] Md. Zakir Hossain, Ferdous Sohel, Mohd Fairuz Shiratuddin, and Hamid Laga. 2018. A Comprehensive Survey of Deep Learning for Image Captioning.   
[64] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. 2020. Embedding-based retrieval in facebook search.   
[65] Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. 2023. Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2.   
[66] Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Daniel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Koura, Xian Li, Brian O'Horo, Gabriel Pereyra, Jeff Wang, Christopher Dewan, Asli Celikyilmaz, Luke Zettlemoyer, and Ves Stoyanov. 2023. OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization.   
[67] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. 2022. Atlas: Few-shot Learning with Retrieval Augmented Language Models.   
[68] Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. 2021. Perceiver: General Perception with Iterative Attention.   
[69] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B.   
[70] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of Experts.   
[71] Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang. 2023. Scaling Sentence Embeddings with Large Language Models.

[72] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, and Xinghua Lu. 2019. PubMedQA: A Dataset for Biomedical Research Question Answering.   
[73] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with GPUs.   
[74] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.   
[75] Lukasz Kaiser, Aidan N. Gomez, Noam Shazeer, Ashish Vaswani, Niki Parmar, Llion Jones, and Jakob Uszkoreit. 2017. One Model To Learn Them All.   
[76] Uday Kamath, John Liu, and James Whitaker. 2019. Deep learning for NLP and speech recognition.   
[77] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering.   
[78] Phillip Keung, Yichao Lu, György Szarvas, and Noah A. Smith. 2020. The Multilingual Amazon Reviews Corpus.   
[79] Daniel Khashabi, Amos Ng, Tushar Khot, Ashish Sabharwal, Hannaneh Hajishirzi, and Chris Callison-Burch. 2021. GooAQ: Open Question Answering with Diverse Answer Types.   
[80] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Casey A Fitzpatrick, Peter Bull, Greg Lipstein, Tony Nelli, Ron Zhu, et al. 2021. The hateful memes challenge: Competition report.   
[81] Diederik P. Kingma and Jimmy Ba. 2017. Adam: A Method for Stochastic Optimization.   
[82] Mahnaz Koupaee and William Yang Wang. 2018. WikiHow: A Large Scale Text Summarization Dataset.   
[83] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019. Natural questions: a benchmark for question answering research.   
[84] Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. 2023. OpenAssistant Conversations – Democratizing Large Language Model Alignment.   
[85] Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. 2017. Zero-Shot Relation Extraction via Reading Comprehension.   
[86] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.   
[87] Patrick Lewis, Yuxiang Wu, Linqing Liu, Pasquale Minervini, Heinrich Küttler, Aleksandra Piktus, Pontus Stenetorp, and Sebastian Riedel. 2021. PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them.   
[88] Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. 2023. StarCoder: may the source be with you!   
[89] Xianming Li and Jing Li. 2023. AnglE-optimized Text Embeddings.   
[90] Xianming Li and Jing Li. 2023. DeeLM: Dependency-enhanced Large Language Model for Sentence Embeddings.

[91] Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. AlpacaEval: An Automatic Evaluator of Instruction-following Models.   
[92] Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Scott Yih. 2023. RA-DIT: Retrieval-Augmented Dual Instruction Tuning.   
[93] Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Dan S. Weld. 2020. S2ORC: The Semantic Scholar Open Research Corpus.   
[94] Shayne Longpre, Robert Mahari, Anthony Chen, Naana Obeng-Marnu, Damien Sileo, William Brannon, Niklas Muennighoff, Nathan Khazam, Jad Kabbara, Kartik Perisetla, Xinyi Wu, Enrico Shippole, Kurt Bollacker, Tongshuang Wu, Luis Villa, Sandy Pentland, Deb Roy, and Sara Hooker. 2023. The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI.   
[95] Hongyin Luo, Yung-Sung Chuang, Yuan Gong, Tianhua Zhang, Yoon Kim, Xixin Wu, Danny Fox, Helen Meng, and James Glass. 2023. SAIL: Search-Augmented Instruction Learning.   
[96] Risto Luukkonen, Ville Komulainen, Jouni Luoma, Anni Eskelinen, Jenna Kanerva, Hanna-Mari Kupari, Filip Ginter, Veronika Laippala, Niklas Muennighoff, Aleksandra Piktus, Thomas Wang, Nouamane Tazi, Teven Le Scao, Thomas Wolf, Osma Suominen, Samuli Sairanen, Mikko Merioksa, Jyrki Heinonen, Aija Vahtola, Samuel Antao, and Sampo Pyysalo. 2023. FinGPT: Large Generative Models for a Small Language.   
[97] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2023. Fine-Tuning LLaMA for Multi-Stage Text Retrieval.   
[98] Xueguang Ma, Xinyu Zhang, Ronak Pradeep, and Jimmy Lin. 2023. Zero-Shot Listwise Document Reranking with a Large Language Model.   
[99] Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. 2018. The Natural Language Decathlon: Multitask Learning as Question Answering.   
[100] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Distributed Representations of Words and Phrases and their Compositionality.   
[101] Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2022. MetaICL: Learning to Learn In Context.   
[102] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2022. Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?   
[103] Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. 2022. Cross-Task Generalization via Natural Language Crowdsourcing Instructions.   
[104] John X. Morris, Volodymyr Kuleshov, Vitaly Shmatikov, and Alexander M. Rush. 2023. Text Embeddings Reveal (Almost) As Much As Text.   
[105] Niklas Muennighoff. 2020. Vilio: State-of-the-art Visio-Linguistic Models applied to Hateful Memes.   
[106] Niklas Muennighoff. 2022. SGPT: GPT Sentence Embeddings for Semantic Search.   
[107] Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, and Shayne Longpre. 2023. OctoPack: Instruction Tuning Code Large Language Models.   
[108] Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. 2023. Scaling Data-Constrained Language Models.

[109] Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. MTEB: Massive Text Embedding Benchmark.   
[110] Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. 2023. Crosslingual Generalization through Multitask Finetuning.   
[111] Shashi Narayan, Shay B. Cohen, and Mirella Lapata. 2018. Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization.   
[112] Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David Schnurr, Felipe Petroski Such, Kenny Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, and Lilian Weng. 2022. Text and Code Embeddings by Contrastive Pre-Training.   
[113] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego, Ji Ma, Vincent Y. Zhao, Yi Luan, Keith B. Hall, Ming-Wei Chang, and Yinfei Yang. 2021. Large Dual Encoders Are Generalizable Retrievers.   
[114] Jianmo Ni, Gustavo Hernández Ábrego, Noah Constant, Ji Ma, Keith B. Hall, Daniel Cer, and Yinfei Yang. 2021. Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models.   
[115] Rodrigo Nogueira and Kyunghyun Cho. 2020. Passage Re-ranking with BERT.   
[116] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, et al. 2023. GPT-4 Technical Report.   
[117] Ankit Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. 2022. MedMCQA : A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering.   
[118] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library.   
[119] Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation.   
[120] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021. KILT: a Benchmark for Knowledge Intensive Language Tasks.   
[121] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models.   
[122] Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!   
[123] Yifu Qiu, Hongyu Li, Yingqi Qu, Ying Chen, Qiaoqiao She, Jing Liu, Hua Wu, and Haifeng Wang. 2022. DuReader\_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine.   
[124] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision.

[125] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training.   
[126] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners.   
[127] Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. 2022. Scaling Language Models: Methods, Analysis & Insights from Training Gopher.   
[128] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model.   
[129] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2023. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.   
[130] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ Questions for Machine Comprehension of Text.   
[131] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, and Nando de Freitas. 2022. A Generalist Agent.   
[132] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.   
[133] Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. Learning To Retrieve Prompts for In-Context Learning.   
[134] Alexander M. Rush, Sumit Chopra, and Jason Weston. 2015. A Neural Attention Model for Abstractive Sentence Summarization.   
[135] Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. 2022. Multitask Prompted Training Enables Zero-Shot Task Generalization.   
[136] Teven Le Scao, Thomas Wang, Daniel Hesslow, Lucile Saulnier, Stas Bekman, M Saiful Bari, Stella Biderman, Hady Elsahar, Niklas Muennighoff, Jason Phang, Ofir Press, Colin Raffel, Victor Sanh, Sheng Shen, Lintang Sutawika, Jaesung Tae, Zheng Xin Yong, Julien Launay, and Iz Beltagy. 2022. What Language Model to Train if You Have One Million GPU Hours?   
[137] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools.   
[138] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.   
[139] Zejiang Shen, Kyle Lo, Lauren Yu, Nathan Dahlberg, Margo Schlanger, and Doug Downey. 2022. Multi-LexSum: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities.   
[140] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. 2023. REPLUG: Retrieval-Augmented Black-Box Language Models.   
[141] Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela. 2022. FLAVA: A Foundational Language And Vision Alignment Model.

[142] Shivalika Singh, Freddie Vargus, Daniel Dsouza, Börje F. Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura OMahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Souza Moura, Dominik Krzemiński, Hakimeh Fadaei, Irem Ergün, Ifeoma Okoh, Aisha Alaagib, Oshan Mudannayake, Zaid Alyafeai, Vu Minh Chien, Sebastian Ruder, Surya Guthikonda, Emad A. Alghamdi, Sebastian Gehrmann, Niklas Muennighoff, Max Bartolo, Julia Kreutzer, Ahmet Üstün, Marzieh Fadaee, and Sara Hooker. 2024. Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning.   
[143] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. 2024. Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research.   
[144] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2023. Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models.   
[145] Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. 2023. One Embedder, Any Task: Instruction-Finetuned Text Embeddings.   
[146] Ming-Hsiang Su, Chung-Hsien Wu, Kun-Yi Huang, Qian-Bei Hong, and Hsin-Min Wang. 2017. A chatbot using LSTM-based multi-layer embedding for elderly care.   
[147] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents.   
[148] Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to Sequence Learning with Neural Networks.   
[149] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. 2022. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.   
[150] Flax Sentence Embeddings Team. 2021. Stack Exchange question pairs.   
[151] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, et al. 2023. Gemini: A Family of Highly Capable Multimodal Models.   
[152] Sentence Transformers Team. 2021. Reddit Title Body.   
[153] Sentence Transformers Team. 2021. (Title, Body) pairs from the npr.org website.   
[154] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models.   
[155] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for Fact Extraction and VERification.   
[156] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and Efficient Foundation Language Models.

[157] Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. 2023. Zephyr: Direct Distillation of LM Alignment.   
[158] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2023. Attention Is All You Need.   
[159] Ben Wang and Aran Komatsuzaki. 2021. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model.   
[160] Bin Wang and C. C. Jay Kuo. 2020. SBERT-WK: A Sentence Embedding Method by Dissecting BERT-based Word Models.   
[161] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text Embeddings by Weakly-Supervised Contrastive Pretraining.   
[162] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Improving Text Embeddings with Large Language Models.   
[163] Thomas Wang, Adam Roberts, Daniel Hesslow, Teven Le Scao, Hyung Won Chung, Iz Beltagy, Julien Launay, and Colin Raffel. 2022. What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?   
[164] Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. 2023. How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources.   
[165] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. 2022. Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks.   
[166] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022. Finetuned Language Models Are Zero-Shot Learners.   
[167] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.   
[168] Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. 2023. Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning.   
[169] Shitao Xiao and Zheng Liu. 2022. RetroMAE v2: Duplex Masked Auto-Encoder For Pre-Training Retrieval-Oriented Language Models.   
[170] Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao. 2022. RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder.   
[171] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-Pack: Packaged Resources To Advance General Chinese Embedding.   
[172] Xiaohui Xie, Qian Dong, Bingning Wang, Feiyang Lv, Ting Yao, Weinan Gan, Zhijing Wu, Xiangsheng Li, Haitao Li, Yiqun Liu, and Jin Ma. 2023. T2Ranking: A large-scale Chinese Benchmark for Passage Ranking.   
[173] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.

[174] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. 2023. Retrieval-Augmented Multimodal Language Modeling.   
[175] Jun Yin, Xin Jiang, Zhengdong Lu, Lifeng Shang, Hang Li, and Xiaoming Li. 2016. Neural Generative Question Answering.   
[176] Zheng-Xin Yong, Hailey Schoelkopf, Niklas Muennighoff, Alham Fikri Aji, David Ifeoluwa Adelani, Khalid Almubarak, M Saiful Bari, Lintang Sutawika, Jungo Kasai, Ahmed Baruwa, Genta Indra Winata, Stella Biderman, Edward Raff, Dragomir Radev, and Vassilina Nikoulina. 2023. BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting.   
[177] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions.   
[178] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. 2022. CoCa: Contrastive Captioners are Image-Text Foundation Models.   
[179] Elad Ben Zaken, Shauli Ravfogel, and Yoav Goldberg. 2022. BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.   
[180] Xiang Zhang, Junbo Zhao, and Yann LeCun. 2016. Character-level Convolutional Networks for Text Classification.   
[181] Xin Zhang, Zehan Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, and Min Zhang. 2023. Language Models are Universal Embedders.   
[182] Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin. 2021. Mr. TyDi: A Multi-lingual Benchmark for Dense Retrieval.   
[183] Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. 2022. Making a MIRACL: Multilingual Information Retrieval Across a Continuum of Languages.   
[184] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. 2023. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.   
[185] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.   
[186] Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. LIMA: Less Is More for Alignment.   
[187] Yutao Zhu, Huaying Yuan, Shutting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan Chen, Zhicheng Dou, and Ji-Rong Wen. 2024. Large Language Models for Information Retrieval: A Survey.   
[188] Terry Yue Zhuo, Armel Zebaze, Nitchakarn Suppattarachai, Leandro von Werra, Harm de Vries, Qian Liu, and Niklas Muennighoff. 2024. Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models.   
[189] Ahmet Üstün, Viraat Aryabumi, Zheng-Xin Yong, Wei-Yin Ko, Daniel D'souza, Gbemileke Onilude, Neel Bhandari, Shivalika Singh, Hui-Lee Ooi, Amr Kayid, Freddie Vargus, Phil Blunsom, Shayne Longpre, Niklas Muennighoff, Marzieh Fadaee, Julia Kreutzer, and Sara Hooker. 2024. Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model.

# Appendix

# Contents

A Contributions 29

B Artifacts 29

C Loss Curves 31

D Additional RAG results 32

E Evaluation 32

F Ablations Detailed Results 33

G GRITLM MTEB Full Results 38

H Reducing Embedding Training Memory 39

I Hyperparameters 40

J Embedding Instruction for Generative Models 41

K HumanEval Format 41

L Embedding in FP32 vs BF16 41

M Unreliability of MT-Bench 42

N Dataset Composition 43

O Dataset Samples 44

P Evaluation Prompts 47

P.1 Embedding Prompts 47

P.2 Embedding Few-Shot Prompts 53

P.3 Generative Prompts 57

P.4 RAG Prompts 63

Q Hardware 66

R Limitations and Future Work 66

S Version Control 67

# A Contributions

Niklas Muennighoff conceived of the project and led its execution. He designed the experiments and implemented, trained, and evaluated all models, and wrote most of the paper. Hongjin Su created MEDI2, implemented and ran reranking, and helped with experiments. Liang Wang ran several dataset ablations and contributed to framing. Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela advised the project. All authors helped edit the paper.

# B Artifacts

Table 8: Produced artifacts. 

<table><tr><td>Artifact</td><td>Public Link</td></tr><tr><td colspan="2">Table 5</td></tr><tr><td>7B KTO8x7B KTO</td><td>https://hf.co/GritLM/GritLM-7B-KTOhttps://hf.co/GritLM/GritLM-8x7B-KTO</td></tr><tr><td colspan="2">Table 12</td></tr><tr><td>CCCC WMCCCC LTBBCC M</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medithttps://hf.co/GritLM/gritlm_m7_sq2048_medilasttokenhttps://hf.co/GritLM/gritlm_m7_sq2048_medi_bbcc</td></tr><tr><td colspan="2">Table 13</td></tr><tr><td>CC WMCB MBB M</td><td>https://hf.co/GritLM/emb_m7_sq2048_medithttps://hf.co/GritLM/emb_m7_sq2048_medi_cbhttps://hf.co/GritLM/emb_m7_sq2048_medi_bb</td></tr><tr><td colspan="2">Table 14</td></tr><tr><td>CCBCBCIL</td><td>https://hf.co/GritLM/gen_m7_sq2048_tulu2_ep1https://hf.co/GritLM/gen_m7_sq2048_tulu2_bchttps://hf.co/GritLM/gen_m7_sq2048_tulu2_bcil</td></tr><tr><td colspan="2">Table 15</td></tr><tr><td>Mistral 7BLlama 2 7BGPT-J 6B</td><td>https://hf.co/GritLM/gritlm_m7_sq1024_st100_zephfmthttps://hf.co/GritLM/gritlm_l7_sq1024_st100_zephfmthttps://hf.co/GritLM/gritlm_g6_sq1024_st100_zephfmt</td></tr><tr><td colspan="2">Table 16</td></tr><tr><td>MEDIMEDI2 NNIMEDI2MEDI2 + W</td><td>https://hf.co/GritLM/emb_m7_sq2048_medithttps://hf.co/GritLM/emb_m7_sq2048_medi2nnithttps://hf.co/GritLM/emb_m7_sq2048_medi2https://hf.co/GritLM/emb_m7_sq2048_medi2weights</td></tr><tr><td colspan="2">Table 17</td></tr><tr><td>MEDIMEDI2BBCC MEDIIBBCC MEDI2BBCC MEDI2BGEBBCC E5</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medithttps://hf.co/GritLM/gritlm_m7_sq2048_medi2https://hf.co/GritLM/gritlm_m7_sq2048_medi2_bbcchttps://hf.co/GritLM/gritlm_m7_sq2048_medi2bbcchttps://hf.co/GritLM/gritlm_m7_sq2048_medi2bge_bbcchttps://hf.co/GritLM/gritlm_m7_sq2048_e5_bbcc</td></tr><tr><td colspan="2">Table 18</td></tr><tr><td>Tülu 2 1 EP</td><td>https://hf.co/GritLM/gen_m7_sq2048_tulu2_ep1</td></tr><tr><td>Tülu 2 2 EP</td><td>https://hf.co/GritLM/gen_m7_sq2048_tulu2_ep2</td></tr><tr><td>OASST 1 EP</td><td>https://hf.co/GritLM/gen_m7_sq2048_oasst_ep1</td></tr><tr><td>OASST 2 EP</td><td>https://hf.co/GritLM/gen_m7_sq2048_oasst_ep2</td></tr><tr><td>UltraChat</td><td>https://hf.co/GritLM/gen_m7_sq2048_ultrachat_ep1</td></tr><tr><td colspan="2">Table 19</td></tr><tr><td>No head</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medi_gendups</td></tr><tr><td>-&gt; 1024</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medi_proj1024_gendups</td></tr><tr><td colspan="2">Table 20</td></tr><tr><td>256</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medi2</td></tr><tr><td>4096</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medi2_bs4096</td></tr><tr><td colspan="2">Table 21</td></tr><tr><td>BF16</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_e5s_bbcc_bs2048_token6</td></tr><tr><td>FP32</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_e5s_bbcc_bs2048_token6_fp32</td></tr><tr><td colspan="2">Table 22</td></tr><tr><td>Any dataset</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_e5_bbcc_token_anyneg</td></tr><tr><td>Same dataset</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_e5_bbcc_token</td></tr><tr><td colspan="2">Table 23</td></tr><tr><td>Tülu 2</td><td>https://hf.co/GritLM/gen_m7_sq2048_tulu2_ep1</td></tr><tr><td>Zephyr β</td><td>https://hf.co/GritLM/gen_m7_sq2048_tulu2_ep1_zephfmt</td></tr><tr><td colspan="2">Table 24</td></tr><tr><td>MEDI 2048</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medi</td></tr><tr><td>MEDI 4096</td><td>https://hf.co/GritLM/gritlm_m7_sq4096_medi</td></tr><tr><td>BBCC MEDI2 512</td><td>https://hf.co/GritLM/gritlm_m7_sq512_medi2_bbcc</td></tr><tr><td>BBCC MEDI2 2048</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medi2_bbcc</td></tr><tr><td colspan="2">Table 25</td></tr><tr><td>E5 Token 4.2</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_e5s_bbcc_bs2048_token4</td></tr><tr><td>E5 Token 6.0</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_e5s_bbcc_bs2048_token6</td></tr><tr><td>E5 Mix 32 -&gt; 8</td><td>https://hf.co/GritLM/GritLM-7b</td></tr><tr><td>MEDI2 Mix 4 -&gt; 64</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medi2_bbcc</td></tr><tr><td>MEDI2 Mix 32 -&gt; 8</td><td>https://hf.co/GritLM/gritlm_m7_sq2048_medi2_bbcc_bs4096</td></tr><tr><td colspan="2">Other</td></tr><tr><td>Code</td><td>https://github.com/ContextualAI/gritlm</td></tr><tr><td>Logs</td><td>https://wandb.ai/muennighoff/gritlm</td></tr><tr><td>Tülu 2</td><td>https://hf.co/datasets/GritLM/tulu2</td></tr><tr><td>MEDI</td><td>https://hf.co/datasets/GritLM/medi</td></tr><tr><td>MEDI2</td><td>https://hf.co/datasets/GritLM/medi2</td></tr><tr><td>MEDI2BGE</td><td>https://hf.co/datasets/GritLM/medi2bge</td></tr><tr><td>GRITLM 7B NQ Index (§4)</td><td>https://hf.co/datasets/GritLM/index</td></tr></table>

Main artifacts 

<table><tr><td>GRITLM 7B</td><td>https://hf.co/GritLM/GritLM-7B</td></tr><tr><td>GRITLM 8x7B</td><td>https://hf.co/GritLM/GritLM-8x7B</td></tr></table>

Table 9: Used artifacts released by others. 

<table><tr><td>Model / Dataset</td><td>Public Link</td></tr><tr><td>GPT-4 [116]OpenAI v3 [116]Gemini [151]</td><td>https://openai.com/gpt-4https://openai.com/blog/new-embedding-models-and-api-updateshttps://deepmind.google/technologies/gemini/</td></tr><tr><td>Llama 2 [156]Mistral 7B [69]Mistral 7B Instruct [69]Mixtral 8x7B [70]Mixtral 8x7B Instruct [70]Tülu 2 [65]</td><td>https://hf.co/meta-llamahttps://hf.co/mistralai/Mistral-7B-v0.1https://hf.co/mistralai/Mistral-7B-Instruct-v0.1https://hf.co/mistralai/Mixtral-8x7B-v0.1https://hf.co/mistralai/Mixtral-8x7B-Instruct-v0.1https://hf.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101</td></tr><tr><td>GPT-J 6B [159]SGPT BE 5.8B [106]</td><td>https://hf.co/EleutherAI/gpt-j-6bhttps://hf.co/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit</td></tr><tr><td>Instructor-XL 1.5B [145]BGE Large 0.34B [171]Zephyr 7B β [157]E5 Mistral 7B [162]UltraChat [36, 157]OASST [84, 107]</td><td>https://hf.co/hkunlp/instructor-xlhttps://hf.co/BAAI/bge-large-en-v1.5https://hf.co/HuggingFaceH4/zephyr-7b-betahttps://hf.co/intfloat/e5-mistral-7b-instructhttps://hf.co/datasets/HuggingFaceH4/ultrachat_200khttps://hf.co/datasets/bigcode/oasst-octopack</td></tr></table>

# C Loss Curves

![](images/6fca0f9aa9009faff0097d8bd7bee6c2f31d9b5771583c706a0dc3a55342d241.jpg)

<details>
<summary>line</summary>

| Steps | Training Loss |
| ----- | ------------- |
| 0     | 5.0           |
| 250   | 1.0           |
| 500   | 0.8           |
| 750   | 0.6           |
| 1000  | 0.4           |
| 1250  | 0.3           |
</details>

![](images/ef7f45741f5e575839c145ed130f4f66a8e6390e6c9911cb829a30fb11812c1f.jpg)

<details>
<summary>line</summary>

| Steps | Generative Loss |
| ----- | --------------- |
| 0     | 1.20            |
| 100   | 2.50            |
| 250   | 1.40            |
| 500   | 1.35            |
| 750   | 1.15            |
| 1000  | 1.05            |
| 1250  | 1.10            |
</details>

Figure 6: GRITLM 7B training loss smoothed with exponential moving average smoothing and a weight of 0.9.

![](images/d8789efb71c46260743df42668cbba268d0398b2fddeab1e394f6119349f8982.jpg)

<details>
<summary>line</summary>

| Steps | Training Loss |
| ----- | ------------- |
| 0     | 7.0           |
| 1250  | 0.5           |
</details>

![](images/927cda2823a5fda2e61660bff5364746e53733c2d66bb2c36d68203651382cdc.jpg)

<details>
<summary>line</summary>

| Steps | Generative Loss |
| ----- | --------------- |
| 0     | 0.25            |
| 250   | 0.85            |
| 500   | 0.80            |
| 750   | 0.82            |
| 1000  | 0.78            |
| 1250  | 0.76            |
</details>

Figure 7: GRITLM 8X7B training loss smoothed with exponential moving average smoothing and a weight of 0.9.

# D Additional RAG results

In §4, we find that doc caching is the most promising caching variant out of the ones we propose. This is because (a) documents are usually significantly longer than queries, thus caching documents has the highest potential to reduce latency, (b) it maintains performance of regular RAG (Table 7, and (c) it even works for non-GRIT models though it requires more time to construct the cache for non-GRIT models (§4). Thus we further experiment with doc caching in Table 10 to verify its performance on other datasets. Similar to Natural Questions in Table 7, we observe that doc caching maintains performance of regular RAG (even slightly improves)

Table 10: Additional doc caching results. We use the same setup as in Table 7 to benchmark doc caching on two additional datasets: TriviaQA [74] and MMLU [61]. 

<table><tr><td>Dataset (→)Metric (→)</td><td>TriviaQA Match (0-shot, ↑)</td><td>MMLU</td></tr><tr><td>RAG</td><td>52.12</td><td>51.10</td></tr><tr><td>Doc Caching</td><td>57.93</td><td>53.46</td></tr></table>

for TriviaQA and MMLU despite the attention mismatch. Note that the attention mismatch problem can always be resolved by simply not using bidirectional attention for the embedding part and thereby guarantee the same performance as not using RAG, however, not using bidirectional attention comes at a slight reduction in embedding performance according to our ablation experiments ( $\S3.3$ ).

We also benchmark the BGE series of embedding models $[171]$ in Table 11 for RAG. We find performance to be significantly worse than with GRITLM in Table 7. Based on a manual inspection of samples, it appears that the embedding models commonly retrieve irrelevant passages that confuse the generative model. There may be other smaller embedding models or other generative models that may perform better, but overall we expect the RAG performance to be a function of the embedding and generative performance of the individual components (e.g. if an embedding model performs better than GRITLM, we would expect it to lead to better RAG performance; BGE generally does not perform better on embedding as shown in Table 1).

Table 11: Additional RAG results with BGE. We use the same setup as in Table 7 to benchmark BGE embedding models with the "Query then document" prompt. The generative model is still GRITLM 7B. 

<table><tr><td>Dataset (→)Metric (→)</td><td>NQMatch (0-shot, ↑)</td></tr><tr><td>BGE Large 0.34B</td><td>10.39</td></tr><tr><td>BGE Base 0.11B</td><td>10.31</td></tr><tr><td>BGE Small 0.03B</td><td>10.17</td></tr></table>

# E Evaluation

For evaluating GRITLM, we select the most commonly used embedding and generative benchmarks:

Embedding To evaluate embedding performance we use the 7 main tasks from MTEB [109].

(1) Classification (CLF): A logistic regression classifier is trained on embeddings from texts with different labels. The classifier is scored with F1.   
(2) Clustering (Clust.): K-means clustering is performed on embeddings from different sources. The agreement of the clusters with respect to the source labels is scored with V-measure.   
(3) Pair Classification (PairCLF): The cosine similarity of two embeddings with a binary label is computed. The optimal similarity threshold across all samples is found and scored with AP (average precision).   
(4) Reranking (Rerank) A query embedding and reference embeddings are compared with cosine similarity. The similarities are scored versus the ground truth ranking of the references via MAP (mean AP).   
(5) Retrieval: A query embedding and embeddings of references are compared with cosine similarity. The position of the correct reference(s) in the top ten with the highest cosine similarity is scored with nDCG@10 (normalized discounted cumulative gain).   
(6) STS: The cosine similarity of two embeddings is compared with a ground truth continuous score of their similarity and scored with Spearman correlation.   
(7) Summarization (Summ.) Human-written and machine-written summaries of the same text are embedded. The cosine similarity of the embeddings is compared to human ratings of the machine summaries and scored with Spearman correlation.   
Among the tasks, Reranking, Retrieval, and Summarization are asymmetric i.e. there are two different kinds of embeddings: queries and documents. Others are symmetric i.e. there is only one kind. We use instructions for every dataset specified in §P.1. Notably, for some models, we use different instructions for query and document embeddings when dealing with asymmetric tasks. The datasets within each task cover diverse domains ranging from scientific papers to casual conversations.

Generation For evaluating the generative performance of GRITLM, we largely follow the evaluation setup of Tülu [164, 65] using open-source frameworks [50, 10].

(1) Multiple-Choice Question Answering via MMLU [61]: Models are tasked to answer knowledge-intensive questions from different fields, such as humanities, social sciences, and hard sciences. No few-shots (FS) are provided and answers are evaluated with exact match.   
(2) Problem solving via GSM [25]: Models are tasked to solve a math problem requiring multi-step reasoning. 8 few-shot (FS) examples with chain-of-thought reasoning (CoT) [167] are provided and exact match is measured.   
(3) Multilingual Closed-book Question Answering via TyDi QA [24]: Models are tasked to answer a question in one of six languages. We evaluate in the Gold Passage and no-context setting following Anil et al. [2].   
(4) Code Generation via HumanEvalSynthesize $[107, 17]$ : We use the HumanEvalSynthesize Python dataset $[107]$ , which is adapted from HumanEval $[17]$ for easy evaluation of instruction-following models. Using the instruction format is different from Ivison et al. $[65]$ who use HumanEval without an instruction format which is not how the model is used in practice. Following Muennighoff et al. $[107]$ , we score pass@1 using 20 samples and a temperature of 0.2.   
(5) Boolean Expressions, Causal Judgement, etc. via BBH [144, 149] We evaluate a variety of reasoning tasks using BIG-Bench Hard (BBH) [144, 149]. Similar to GSM8K, 3 FS CoT examples are provided and exact match is measured.   
(6) Open-ended writing, Summarization, Role-playing, etc. via AlpacaEval (Alpaca) [91, 40] We evaluate a variety of open-ended generation tasks via the original 1.0 version of AlpacaEval [91, 40]. GPT-4 [116] is used to determine the win rate of generations compared to provided GPT-3 [13] answers. We differ from Ivison et al. [65] in that we reduce the maximum token length to 6144 from 8192. We do not use MT-Bench due to its limitations pointed out in Appendix M. To ensure reproducibility, we use greedy evaluation throughout.

# F Ablations Detailed Results

We display a breakdown of the results from Table 6 in Table 12 to Table 23. For MTEB per-dataset results, we refer to Appendix G, the MTEB leaderboard (https://hf.co/spaces/mteb/leaderboard) and our released result files (https://hf.co/datasets/GritLM/results).

Table 12: Unified models attention and pooling ablations. The sequence of Cs and Bs refers to the attention mechanism for (from left to right): Emb instruction, Emb sample, Gen instruction, Gen sample, where C=Causal, B=Bidirectional, Emb=Embedding and Gen=Generative. WM, LT and M refer to position-weighted mean, last token and mean pooling, respectively. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td>CCCC WM</td><td>77.9</td><td>47.9</td><td>81.5</td><td>59.0</td><td>49.4</td><td>80.3</td><td>29.4</td><td>62.8</td></tr><tr><td>CCCC LT</td><td>78.8</td><td>46.9</td><td>84.5</td><td>59.6</td><td>43.9</td><td>78.7</td><td>29.3</td><td>61.2</td></tr><tr><td>BBCC M</td><td>79.0</td><td>48.6</td><td>86.3</td><td>59.5</td><td>49.9</td><td>81.7</td><td>30.1</td><td>63.8</td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU0 FSEM</td><td>GSM8K8 FS, CoTEM</td><td>BBH3 FS, CoTEM</td><td>TyDi QA1 FS, GPF1</td><td colspan="2">HumanEval0 FSpass@1</td><td>Alpaca0 FS, 1.0% Win</td><td>Avg.</td></tr><tr><td>CCCC WM</td><td>57.5</td><td>45.0</td><td>53.1</td><td>56.0</td><td colspan="2">32.3</td><td>72.9</td><td>52.8</td></tr><tr><td>CCCC LT</td><td>57.2</td><td>45.5</td><td>54.7</td><td>54.0</td><td colspan="2">31.1</td><td>75.7</td><td>53.0</td></tr><tr><td>BBCC M</td><td>57.0</td><td>46.5</td><td>54.5</td><td>55.0</td><td colspan="2">30.4</td><td>73.8</td><td>52.9</td></tr></table>

Table 13: Embedding-only models attention and pooling ablations. The sequence of Cs and Bs refers to the attention mechanism for (from left to right): Emb instruction, Emb sample, where C=Causal, B=Bidirectional and Emb=Embedding. WM and M refer to position-weighted mean and mean pooling, respectively. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td>CC WM</td><td>77.1</td><td>44.0</td><td>83.3</td><td>57.0</td><td>43.2</td><td>79.6</td><td>29.4</td><td>60.0</td></tr><tr><td>CB M</td><td>76.4</td><td>45.5</td><td>83.1</td><td>56.8</td><td>45.7</td><td>80.6</td><td>30.4</td><td>61.0</td></tr><tr><td>BB M</td><td>77.3</td><td>46.0</td><td>83.8</td><td>58.2</td><td>46.8</td><td>81.0</td><td>32.3</td><td>61.8</td></tr></table>

Table 14: Generative-only models attention ablations. The sequence of Cs and Bs refers to the attention mechanism for (from left to right): Gen instruction, Gen sample, where C=Causal and B=Bidirectional. IL=interleaved, whereby the bidirectional attention is interleaved with causal attention in multi-turn samples (bidirectional for instructions, causal for answers). This allows for faster generation in multi-turn settings as the kv-cache of the answer can be reused. 

<table><tr><td>Dataset (→)</td><td>MMLU</td><td>GSM8K</td><td>BBH</td><td>TyDi QA</td><td>HumanEval</td><td>Alpaca</td><td>Avg.</td></tr><tr><td>Setup (→)</td><td>0 FS</td><td>8 FS, CoT</td><td>3 FS, CoT</td><td>1 FS, GP</td><td>0 FS</td><td>0 FS, 1.0</td><td></td></tr><tr><td>Metric (→)</td><td>EM</td><td>EM</td><td>EM</td><td>F1</td><td>pass@1</td><td>% Win</td><td></td></tr><tr><td>CC</td><td>57.5</td><td>52.0</td><td>55.4</td><td>56.6</td><td>34.5</td><td>75.4</td><td>55.2</td></tr><tr><td>BC</td><td>57.2</td><td>50.0</td><td>49.3</td><td>52.0</td><td>30.6</td><td>64.8</td><td>50.7</td></tr><tr><td>BC IL</td><td>52.6</td><td>41.0</td><td>46.9</td><td>45.4</td><td>-</td><td>-</td><td>-</td></tr></table>

Table 15: Base model ablations. Models are only trained for 100 steps and with other sub-optimal settings, such as the Zephyr format, that were rectified through later ablations. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ. Spear.1</td><td>Avg.56</td></tr><tr><td>Mistral 7B</td><td>70.6</td><td>43.7</td><td>74.0</td><td>54.8</td><td>35.3</td><td>72.9</td><td>31.2</td><td>54.6</td></tr><tr><td>Llama 2 7B</td><td>68.1</td><td>38.0</td><td>64.1</td><td>50.2</td><td>24.2</td><td>67.7</td><td>30.5</td><td>48.2</td></tr><tr><td>GPT-J 6B</td><td>70.7</td><td>41.4</td><td>69.6</td><td>53.9</td><td>29.7</td><td>70.4</td><td>29.8</td><td>51.9</td></tr><tr><td colspan="9"></td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU0 FSEM</td><td>GSM8K8 FS, CoTEM</td><td>BBH3 FS, CoTEM</td><td>TyDi QA1 FS, GPF1</td><td colspan="2">HumanEval0 FSpass@1</td><td colspan="2">Avg.</td></tr><tr><td>Mistral 7B</td><td>35.0</td><td>11.0</td><td>31.6</td><td>20.5</td><td colspan="2">13.8</td><td colspan="2">22.4</td></tr><tr><td>Llama 2 7B</td><td>35.8</td><td>7.0</td><td>27.2</td><td>21.0</td><td colspan="2">12.9</td><td colspan="2">20.8</td></tr><tr><td>GPT-J 6B</td><td>27.5</td><td>3.5</td><td>22.2</td><td>8.7</td><td colspan="2">8.0</td><td colspan="2">14.0</td></tr></table>

Table 16: Embedding-only models embedding dataset ablations. NNI = No Natural Instructions, corresponding to not including natural instructions in the data. II = evaluating with the Instructor-XL instructions [145]. Other models use our new structure with domain, intent, and unit depicted in Figure 3. Thus, MEDI2 NNI II and MEDI2 NNI are the same model and only differ in the evaluation instruction set. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td>MEDI II</td><td>77.1</td><td>44.0</td><td>83.3</td><td>57.0</td><td>43.2</td><td>79.6</td><td>29.4</td><td>60.0</td></tr><tr><td>MEDI2 NNI II</td><td>74.0</td><td>43.5</td><td>80.5</td><td>56.6</td><td>46.1</td><td>78.4</td><td>29.5</td><td>59.6</td></tr><tr><td>MEDI2 NNI</td><td>74.2</td><td>44.5</td><td>80.7</td><td>57.3</td><td>49.5</td><td>79.6</td><td>30.8</td><td>61.1</td></tr><tr><td>MEDI2</td><td>75.1</td><td>43.8</td><td>80.6</td><td>57.5</td><td>50.2</td><td>81.7</td><td>31.9</td><td>61.7</td></tr><tr><td>MEDI2 + Weights</td><td>74.4</td><td>42.7</td><td>78.4</td><td>57.7</td><td>50.2</td><td>81.4</td><td>30.5</td><td>61.2</td></tr></table>

Table 17: Unified models embedding dataset ablations. The sequence of Cs and Bs refers to the attention mechanism for (from left to right): Emb instruction, Emb sample, where C=Causal, B=Bidirectional, and Emb=Embedding. WM and M refer to position-weighted mean and mean pooling, respectively. MEDI2BGE corresponds to our MEDI2 dataset with negatives coming from the BGE training dataset MTP [171]. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc. 12</td><td>Clust. V-Meas. 11</td><td>PairCLF AP 3</td><td>Rerank MAP 4</td><td>Retrieval nDCG 15</td><td>STS Spear. 10</td><td>Summ. Spear. 1</td><td>Avg. 56</td></tr><tr><td>CCCC WM MEDI</td><td>77.9</td><td>47.9</td><td>81.5</td><td>59.0</td><td>49.4</td><td>80.3</td><td>29.4</td><td>62.8</td></tr><tr><td>CCCC WM MEDI2</td><td>76.5</td><td>47.0</td><td>82.5</td><td>59.4</td><td>51.4</td><td>81.9</td><td>30.2</td><td>63.2</td></tr><tr><td>BBCC M MEDI</td><td>79.1</td><td>48.8</td><td>86.4</td><td>59.6</td><td>50.3</td><td>81.3</td><td>31.0</td><td>64.0</td></tr><tr><td>BBCC M MEDI2</td><td>77.0</td><td>48.7</td><td>86.0</td><td>61.0</td><td>53.6</td><td>83.0</td><td>29.1</td><td>64.7</td></tr><tr><td>BBCC M MEDI2BGE</td><td>77.0</td><td>48.9</td><td>86.9</td><td>61.3</td><td>53.1</td><td>82.8</td><td>29.4</td><td>64.7</td></tr><tr><td>BBCC M E5</td><td>79.7</td><td>49.5</td><td>86.2</td><td>59.6</td><td>55.3</td><td>83.6</td><td>29.9</td><td>66.0</td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU 0 FS EM</td><td>GSM8K 8 FS, CoT EM</td><td>BBH 3 FS, CoT EM</td><td>TyDi QA 1 FS, GP F1</td><td>HumanEval 0 FS pass@1</td><td>Alpaca 0 FS, 1.0 % Win</td><td>Avg.</td><td></td></tr><tr><td>CCCC WM MEDI</td><td>57.5</td><td>45.0</td><td>53.1</td><td>56.0</td><td>32.3</td><td>72.9</td><td>52.8</td><td></td></tr><tr><td>CCCC WM MEDI2</td><td>57.1</td><td>49.0</td><td>53.3</td><td>55.3</td><td>32.3</td><td>73.6</td><td>53.4</td><td></td></tr><tr><td>BBCC M MEDI</td><td>57.0</td><td>46.5</td><td>54.5</td><td>55.0</td><td>30.4</td><td>73.8</td><td>52.9</td><td></td></tr><tr><td>BBCC M MEDI2</td><td>57.0</td><td>50.5</td><td>53.8</td><td>54.7</td><td>32.3</td><td>74.7</td><td>53.8</td><td></td></tr><tr><td>BBCC M MEDI2BGE</td><td>57.4</td><td>48.0</td><td>54.7</td><td>55.1</td><td>32.0</td><td>74.7</td><td>53.7</td><td></td></tr><tr><td>BBCC M E5</td><td>57.3</td><td>47.5</td><td>54.2</td><td>54.6</td><td>33.6</td><td>75.4</td><td>53.8</td><td></td></tr></table>

Table 18: Generative dataset ablations. EP = number of epochs. 

<table><tr><td>Dataset (→)</td><td>MMLU</td><td>GSM8K</td><td>BBH</td><td>TyDi QA</td><td>HumanEval</td><td>Alpaca</td><td>Avg.</td></tr><tr><td>Setup (→)</td><td>0 FS</td><td>8 FS, CoT</td><td>3 FS, CoT</td><td>1 FS, GP</td><td>0 FS</td><td>0 FS, 1.0</td><td></td></tr><tr><td>Metric (→)</td><td>EM</td><td>EM</td><td>EM</td><td>F1</td><td>pass@1</td><td>% Win</td><td></td></tr><tr><td>Tülu 2 1 EP</td><td>57.5</td><td>52.0</td><td>55.4</td><td>56.6</td><td>34.5</td><td>75.4</td><td>55.2</td></tr><tr><td>Tülu 2 2 EP</td><td>58.2</td><td>53.0</td><td>51.9</td><td>54.1</td><td>37.4</td><td>80.5</td><td>55.9</td></tr><tr><td>OASST 1 EP</td><td>53.8</td><td>24.0</td><td>41.1</td><td>28.2</td><td>27.4</td><td>51.7</td><td>37.7</td></tr><tr><td>OASST 2 EP</td><td>52.4</td><td>17.5</td><td>45.7</td><td>29.2</td><td>19.8</td><td>61.3</td><td>37.7</td></tr><tr><td>UltraChat</td><td>56.1</td><td>43.0</td><td>53.8</td><td>35.0</td><td>25.9</td><td>70.3</td><td>47.4</td></tr></table>

Table 19: Embedding Head. “-> 1024” refers to down-projecting the final hidden state with a linear layer from 4096 to 1024 dimensions only for embedding tasks. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td>No head-&gt; 1024</td><td>77.776.9</td><td>47.947.6</td><td>81.382.1</td><td>58.658.6</td><td>49.248.0</td><td>80.480.1</td><td>29.529.8</td><td>62.762.1</td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU0 FSEM</td><td>GSM8K8 FS, CoTEM</td><td>BBH3 FS, CoTEM</td><td>TyDi QA1 FS, GPF1</td><td>HumanEval0 FSpass@1</td><td>Alpaca0 FS, 1.0% Win</td><td></td><td>Avg.</td></tr><tr><td>No head-&gt; 1024</td><td>54.253.6</td><td>42.537.0</td><td>50.648.8</td><td>53.954.4</td><td>28.426.6</td><td></td><td>65.567.3</td><td>49.248.0</td></tr></table>

Table 20: Embedding batch size ablations. 256 and 4096 indicate the respective embedding batch size. The generative batch size is always 256. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ. Spear.1</td><td>Avg.56</td></tr><tr><td>MEDI2 256</td><td>76.5</td><td>47.0</td><td>82.5</td><td>59.4</td><td>51.4</td><td>81.9</td><td>30.2</td><td>63.2</td></tr><tr><td>MEDI2 4096</td><td>77.1</td><td>48.0</td><td>84.1</td><td>60.2</td><td>52.8</td><td>82.8</td><td>30.5</td><td>64.2</td></tr><tr><td colspan="9"></td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU0 FSEM</td><td>GSM8K8 FS, CoTEM</td><td>BBH3 FS, CoTEM</td><td>TyDi QA1 FS, GPF1</td><td>HumanEval0 FSpass@1</td><td>Alpaca0 FS, 1.0% Win</td><td>Avg.</td><td></td></tr><tr><td>MEDI2 256</td><td>57.1</td><td>49.0</td><td>53.3</td><td>55.3</td><td>32.3</td><td>73.6</td><td>53.4</td><td></td></tr><tr><td>MEDI2 4096</td><td>57.7</td><td>48.0</td><td>53.2</td><td>54.5</td><td>32.0</td><td>74.3</td><td>53.3</td><td></td></tr></table>

Table 21: Precision ablations. BF16 refers to bfloat16 mixed precision and FP32 to float32 precision. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td>BF16</td><td>79.7</td><td>50.2</td><td>87.6</td><td>60.2</td><td>56.5</td><td>83.4</td><td>30.8</td><td>66.5</td></tr><tr><td>FP32</td><td>79.6</td><td>50.3</td><td>87.2</td><td>59.9</td><td>56.1</td><td>83.3</td><td>30.9</td><td>66.3</td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU0 FSEM</td><td>GSM8K8 FS, CoTEM</td><td>BBH3 FS, CoTEM</td><td>TyDi QA1 FS, GPF1</td><td colspan="2">HumanEval0 FSpass@1</td><td>Alpaca0 FS, 1.0% Win</td><td>Avg.</td></tr><tr><td>BF16</td><td>58.2</td><td>51.5</td><td>52.8</td><td>55.9</td><td colspan="2">37.3</td><td>74.4</td><td>55.0</td></tr><tr><td>FP32</td><td>55.9</td><td>52.0</td><td>49.9</td><td>53.9</td><td colspan="2">31.2</td><td>71.3</td><td>52.4</td></tr></table>

Table 22: In-batch negatives ablations. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td>Any dataset</td><td>79.7</td><td>49.8</td><td>85.5</td><td>59.8</td><td>54.9</td><td>83.9</td><td>30.5</td><td>66.0</td></tr><tr><td>Same dataset</td><td>79.5</td><td>48.9</td><td>87.4</td><td>59.0</td><td>56.2</td><td>83.0</td><td>30.5</td><td>66.0</td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU0 FSEM</td><td>GSM8K8 FS, CoTEM</td><td>BBH3 FS, CoTEM</td><td>TyDi QA1 FS, GPF1</td><td>HumanEval0 FSpass@1</td><td>Alpaca0 FS, 1.0% Win</td><td></td><td>Avg.</td></tr><tr><td>Any dataset</td><td>56.1</td><td>43.5</td><td>53.1</td><td>46.6</td><td>33.5</td><td>72.3</td><td></td><td>50.9</td></tr><tr><td>Same dataset</td><td>55.0</td><td>45.0</td><td>54.4</td><td>49.3</td><td>29.6</td><td>73.4</td><td></td><td>51.1</td></tr></table>

Table 23: Generative format ablations. 

<table><tr><td>Dataset (→)</td><td>MMLU</td><td>GSM8K</td><td>BBH</td><td>TyDi QA</td><td>HumanEval</td><td>Alpaca</td><td>Avg.</td></tr><tr><td>Setup (→)</td><td>0 FS</td><td>8 FS, CoT</td><td>3 FS, CoT</td><td>1 FS, GP</td><td>0 FS</td><td>0 FS, 1.0</td><td></td></tr><tr><td>Metric (→)</td><td>EM</td><td>EM</td><td>EM</td><td>F1</td><td>pass@1</td><td>% Win</td><td></td></tr><tr><td>Tülu 2 format</td><td>57.5</td><td>52.0</td><td>55.4</td><td>56.6</td><td>34.5</td><td>75.4</td><td>55.2</td></tr><tr><td>Zephyr β format</td><td>57.3</td><td>53.5</td><td>52.7</td><td>59.1</td><td>0.0</td><td>71.2</td><td>49.0</td></tr></table>

Table 24: Unified models max tokens ablations. X:Y refers to “maximum tokens allowed for embedding documents during training”:“maximum tokens allowed for queries and documents during embedding evaluation”. The sequence of Cs and Bs refers to the attention mechanism for (from left to right): Emb instruction, Emb sample, where C=Causal, B=Bidirectional, and Emb=Embedding. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc. 12</td><td>Clust. V-Meas. 11</td><td>PairCLF AP 3</td><td>Rerank MAP 4</td><td>Retrieval nDCG 15</td><td>STS Spear. 10</td><td>Summ. Spear. 1</td><td>Avg. 56</td></tr><tr><td>MEDI 2048:512</td><td>77.9</td><td>47.9</td><td>81.5</td><td>59.0</td><td>49.4</td><td>80.3</td><td>29.4</td><td>62.8</td></tr><tr><td>MEDI 2048:4096</td><td>77.9</td><td>47.9</td><td>81.5</td><td>59.0</td><td>49.4</td><td>80.2</td><td>31.3</td><td>62.8</td></tr><tr><td>MEDI 4096:512</td><td>76.7</td><td>47.3</td><td>79.8</td><td>58.8</td><td>47.0</td><td>78.5</td><td>30.0</td><td>61.3</td></tr><tr><td>MEDI 4096:4096</td><td>76.8</td><td>47.2</td><td>79.8</td><td>58.8</td><td>46.9</td><td>78.2</td><td>29.9</td><td>61.3</td></tr><tr><td>MEDI2 BBCC 2048:512</td><td>77.0</td><td>48.7</td><td>86.0</td><td>61.0</td><td>53.6</td><td>83.0</td><td>29.1</td><td>64.7</td></tr><tr><td>MEDI2 BBCC 512:512</td><td>76.9</td><td>47.6</td><td>85.5</td><td>61.0</td><td>52.8</td><td>82.3</td><td>28.8</td><td>64.1</td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU 0 FS EM</td><td>GSM8K 8 FS, CoT EM</td><td>BBH 3 FS, CoT EM</td><td>TyDi QA 1 FS, GP F1</td><td>HumanEval 0 FS pass@1</td><td>Alpaca 0 FS, 1.0 % Win</td><td>Avg.</td><td></td></tr><tr><td>MEDI 2048:512/4096</td><td>57.4</td><td>45.0</td><td>53.1</td><td>56.0</td><td>32.3</td><td>72.9</td><td>52.8</td><td></td></tr><tr><td>MEDI 4096:512/4096</td><td>53.8</td><td>43.0</td><td>52.7</td><td>54.8</td><td>30.1</td><td>-</td><td>-</td><td></td></tr><tr><td>MEDI2 BBCC 2048:512</td><td>57.0</td><td>50.5</td><td>53.8</td><td>54.7</td><td>32.3</td><td>74.7</td><td>53.8</td><td></td></tr><tr><td>MEDI2 BBCC 512:512</td><td>56.9</td><td>46.5</td><td>53.1</td><td>52.6</td><td>31.2</td><td>72.8</td><td>52.2</td><td></td></tr></table>

Table 25: Loss ablations. E.g. Mix (32 -> 8) corresponds to token level loss across 32 samples and then sample level loss across 8 sub-batches for a total batch size of 256. E.g. 2.4 refers to the loss ratio of the 1st step: $L_{Emb}/L_{Gen}$ . 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td>E5S Token 2.4</td><td>79.5</td><td>50.1</td><td>86.5</td><td>60.0</td><td>55.6</td><td>83.2</td><td>30.3</td><td>66.1</td></tr><tr><td>E5S Token 6.0</td><td>79.7</td><td>50.2</td><td>87.6</td><td>60.2</td><td>56.5</td><td>83.4</td><td>30.8</td><td>66.5</td></tr><tr><td>E5S Mix (32 -&gt; 8) 4.1</td><td>79.4</td><td>50.5</td><td>87.2</td><td>60.5</td><td>57.4</td><td>83.4</td><td>30.4</td><td>66.7</td></tr><tr><td>Dataset (→)Setup (→)Metric (→)</td><td>MMLU0 FSEM</td><td>GSM8K8 FS, CoTEM</td><td>BBH3 FS, CoTEM</td><td>TyDi QA1 FS, GPF1</td><td>HumanEval0 FSpass@1</td><td>Alpaca0 FS, 1.0% Win</td><td>Avg.</td><td></td></tr><tr><td>E5S Token 2.4</td><td>57.9</td><td>48.5</td><td>53.5</td><td>56.5</td><td>35.2</td><td>75.0</td><td>54.4</td><td></td></tr><tr><td>E5S Token 6.0</td><td>58.2</td><td>51.5</td><td>52.8</td><td>55.9</td><td>37.3</td><td>74.4</td><td>55.0</td><td></td></tr><tr><td>E5S Mix (32 -&gt; 8) 4.1</td><td>57.6</td><td>57.0</td><td>54.8</td><td>55.4</td><td>32.8</td><td>74.8</td><td>55.4</td><td></td></tr><tr><td>MEDI2 Mix (4 -&gt; 64) 11.7</td><td>57.0</td><td>48.0</td><td>53.7</td><td>55.0</td><td>35.8</td><td>67.6</td><td>52.9</td><td></td></tr><tr><td>MEDI2 Mix (32 -&gt; 8) 10.2</td><td>57.0</td><td>50.5</td><td>53.8</td><td>54.7</td><td>32.3</td><td>74.7</td><td>53.8</td><td></td></tr></table>

# G GRITLM MTEB Full Results

Table 26: MTEB full results from Table 1. 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Gen-only</td><td rowspan="2">Emb-only</td><td colspan="2">GRITLM</td></tr><tr><td>7B</td><td>8x7B</td></tr><tr><td>AmazonCounterfactualClassification</td><td>70.06</td><td>82.55</td><td>81.18</td><td>80.48</td></tr><tr><td>AmazonPolarityClassification</td><td>74.74</td><td>96.19</td><td>96.52</td><td>96.32</td></tr><tr><td>AmazonReviewsClassification</td><td>38.63</td><td>57.28</td><td>57.81</td><td>57.18</td></tr><tr><td>Banking77Classification</td><td>71.25</td><td>88.73</td><td>88.47</td><td>87.46</td></tr><tr><td>EmotionClassification</td><td>36.61</td><td>51.83</td><td>52.81</td><td>50.06</td></tr><tr><td>ImdbClassification</td><td>73.94</td><td>94.58</td><td>95.00</td><td>94.32</td></tr><tr><td>MassiveIntentClassification</td><td>66.82</td><td>79.37</td><td>80.78</td><td>79.72</td></tr><tr><td>MassiveScenarioClassification</td><td>71.27</td><td>81.20</td><td>82.09</td><td>81.09</td></tr><tr><td>MTOPDomainClassification</td><td>85.40</td><td>96.72</td><td>96.16</td><td>95.29</td></tr><tr><td>MTOPIntentClassification</td><td>75.60</td><td>87.19</td><td>87.13</td><td>87.08</td></tr><tr><td>ToxicConversationsClassification</td><td>66.36</td><td>68.37</td><td>70.80</td><td>70.89</td></tr><tr><td>TweetSentimentExtractionClassification</td><td>54.61</td><td>61.91</td><td>64.78</td><td>62.48</td></tr><tr><td>ArxivClusteringP2P</td><td>45.40</td><td>50.87</td><td>51.67</td><td>50.72</td></tr><tr><td>ArxivClusteringS2S</td><td>29.86</td><td>47.35</td><td>48.11</td><td>48.01</td></tr><tr><td>BiorxivClusteringP2P</td><td>33.45</td><td>40.18</td><td>40.87</td><td>41.41</td></tr><tr><td>BiorxivClusteringS2S</td><td>23.02</td><td>39.60</td><td>39.80</td><td>38.67</td></tr><tr><td>MedrxivClusteringP2P</td><td>27.49</td><td>36.61</td><td>36.52</td><td>36.54</td></tr><tr><td>MedrxivClusteringS2S</td><td>23.17</td><td>37.28</td><td>36.80</td><td>37.24</td></tr><tr><td>RedditClustering</td><td>23.28</td><td>63.52</td><td>61.30</td><td>63.01</td></tr><tr><td>RedditClusteringP2P</td><td>55.00</td><td>67.81</td><td>67.26</td><td>65.86</td></tr><tr><td>StackExchangeClustering</td><td>47.14</td><td>75.53</td><td>77.33</td><td>74.41</td></tr><tr><td>StackExchangeClusteringP2P</td><td>33.95</td><td>46.22</td><td>41.33</td><td>38.52</td></tr><tr><td>TwentyNewsgroupsClustering</td><td>18.15</td><td>56.8</td><td>55.70</td><td>57.16</td></tr><tr><td>SprintDuplicateQuestions</td><td>51.57</td><td>93.37</td><td>93.00</td><td>91.24</td></tr><tr><td>TwitterSemEval2015</td><td>50.60</td><td>80.61</td><td>81.08</td><td>77.21</td></tr><tr><td>TwitterURLCorpus</td><td>60.36</td><td>87.20</td><td>87.40</td><td>86.45</td></tr><tr><td>AskUbuntuDupQuestions</td><td>49.02</td><td>68.13</td><td>67.34</td><td>65.60</td></tr><tr><td>MindSmallReranking</td><td>27.83</td><td>32.19</td><td>31.81</td><td>32.84</td></tr><tr><td>SciDocsRR</td><td>56.65</td><td>87.00</td><td>86.84</td><td>86.43</td></tr><tr><td>StackOverflowDupQuestions</td><td>38.42</td><td>55.48</td><td>55.96</td><td>54.33</td></tr><tr><td>ArguAna</td><td>35.96</td><td>62.95</td><td>63.24</td><td>59.49</td></tr><tr><td>ClimateFEVER</td><td>8.96</td><td>31.09</td><td>30.91</td><td>28.69</td></tr><tr><td>CQADupstackRetrieval</td><td>7.20</td><td>50.83</td><td>49.42</td><td>47.63</td></tr><tr><td>DBPedia</td><td>2.15</td><td>47.06</td><td>46.60</td><td>46.54</td></tr><tr><td>FEVER</td><td>5.02</td><td>85.41</td><td>82.74</td><td>85.02</td></tr><tr><td>FiQA2018</td><td>6.27</td><td>60.22</td><td>59.95</td><td>49.89</td></tr><tr><td>HotpotQA</td><td>6.67</td><td>79.15</td><td>79.40</td><td>73.83</td></tr><tr><td>MSMARCO</td><td>0.66</td><td>41.55</td><td>41.96</td><td>35.55</td></tr><tr><td>NFCorpus</td><td>3.74</td><td>41.69</td><td>40.89</td><td>39.05</td></tr><tr><td>NQ</td><td>2.14</td><td>69.46</td><td>70.30</td><td>63.87</td></tr><tr><td>QuoraRetrieval</td><td>64.42</td><td>89.08</td><td>89.47</td><td>87.70</td></tr><tr><td>SCIDOCS</td><td>2.32</td><td>24.86</td><td>24.41</td><td>23.06</td></tr><tr><td>SciFact</td><td>35.58</td><td>78.92</td><td>79.17</td><td>77.02</td></tr><tr><td>Touche2020</td><td>3.06</td><td>24.30</td><td>27.93</td><td>27.97</td></tr><tr><td>TRECCOVID</td><td>20.92</td><td>75.29</td><td>74.8</td><td>81.07</td></tr><tr><td>BIOSSES</td><td>70.87</td><td>86.20</td><td>86.35</td><td>87.34</td></tr><tr><td>SICK-R</td><td>58.95</td><td>83.03</td><td>83.13</td><td>80.56</td></tr><tr><td>STS12</td><td>44.25</td><td>78.07</td><td>77.34</td><td>73.69</td></tr><tr><td>STS13</td><td>64.22</td><td>85.98</td><td>85.04</td><td>85.82</td></tr><tr><td>STS14</td><td>52.24</td><td>83.92</td><td>82.91</td><td>82.05</td></tr><tr><td>STS15</td><td>64.53</td><td>89.18</td><td>88.13</td><td>88.8</td></tr><tr><td>STS16</td><td>65.89</td><td>86.83</td><td>86.24</td><td>86.2</td></tr><tr><td>STS17</td><td>69.64</td><td>89.7</td><td>90.13</td><td>91.46</td></tr><tr><td>STS22</td><td>57.29</td><td>68.41</td><td>68.63</td><td>69.21</td></tr><tr><td>STSBenchmark</td><td>53.89</td><td>86.74</td><td>85.64</td><td>87.43</td></tr><tr><td>SummEval</td><td>21.14</td><td>30.18</td><td>30.37</td><td>29.82</td></tr><tr><td>Average</td><td>41.21</td><td>66.82</td><td>66.76</td><td>65.66</td></tr></table>

H Reducing Embedding Training Memory   
![](images/4ed6df354f3814c8b7b94c5c6572f05f9cab1c13046b465da9da50529e2b2533.jpg)

<details>
<summary>line</summary>

| Steps | Backpropagate only passage embeddings | Backpropagate only query embeddings | Backpropagate both but separately | Backpropagate both |
| ----- | -------------------------------------- | ------------------------------------ | --------------------------------- | ------------------ |
| 0     | 3.8                                    | 3.8                                  | 3.8                               | 3.8                |
| 100   | 3.5                                    | 3.4                                  | 3.3                               | 3.2                |
| 200   | 2.5                                    | 2.2                                  | 2.0                               | 1.8                |
| 300   | 2.0                                    | 1.5                                  | 1.2                               | 1.0                |
| 400   | 2.5                                    | 1.0                                  | 0.8                               | 0.6                |
| 500   | 3.0                                    | 0.6                                  | 0.5                               | 0.4                |
| 600   | 3.5                                    | 0.4                                  | 0.3                               | 0.2                |
| 700   | 3.5                                    | 0.3                                  | 0.2                               | 0.1                |
| 800   | 3.5                                    | 0.2                                  | 0.1                               | 0.1                |
| 900   | 3.5                                    | 0.1                                  | 0.1                               | 0.1                |
| 1000  | 3.5                                    | 0.1                                  | 0.1                               | 0.1                |
| 1100  | 3.5                                    | 0.1                                  | 0.1                               | 0.1                |
| 1200  | 3.5                                    | 0.1                                  | 0.1                               | 0.1                |
</details>

Figure 8: Embedding memory ablations. Passage corresponds to both positive and document embeddings. Loss is smoothed with exponential moving average smoothing and a weight of 0.99.

Generative training only requires sufficient memory to perform a forward and backward pass on a single training sample of a given sequence length. Meanwhile, naive embedding training with in-batch negatives requires sufficient memory to accommodate a forward and a backward pass on

Listing 1: Splitting of the embedding pass to save memory, simplified.   
```python
def distributed_contrastive_loss(q, p, n):
    # Gather in-batch negatives across devices...
    # Compute contrastive loss...

    # Split triplet into three forward passes
    pos_rep = model(pos)
    with torch.no_grad():
    q_rep = model(query)
    neg_rep = model(neg)

    # Only perform backward pass on positive documents
    loss = distributed_contrastive_loss(q_rep, pos_rep, neg_rep)
    loss.backward()

    pos_rep = pos_rep.detach()
    # Perform forward + backward on negatives & reuse rest
    neg_rep = model(neg)
    loss = distributed_contrastive_loss(q_rep, pos_rep, neg_rep)
    loss.backward()

    # Perform forward + backward on queries & reuse rest
    neg_rep = neg_rep.detach()
    q_rep = model(query)
    loss = distributed_contrastive_loss(q_rep, pos_rep, neg_rep)
    loss.backward() 
```

3 \* bs samples. The 3 corresponds to the need for passing a triplet of a query, a positive, and a negative document (Equation 1). The batch size (bs) factor corresponds to the need for forwarding all samples together as regular gradient accumulation does not work with in-batch negatives. Below we outline the strategies we employ to reduce these memory needs.

Triplet As the full triplet is only required for loss calculation (Equation 1), it can be split across separate forward and backward passes. To avoid the memory requirements of gradients in PyTorch Autograd [118], this requires two additional forward passes without gradients. Simplified code representing this procedure is depicted in Listing 1. In our training, it was sufficient to only split the triplet into two parts: query and passages, where passages consist of both a positive and a negative document. Thus, we only incur the cost of one additional forward pass without gradients on the query. Alternatively, one could only backpropagate on a subset of the embeddings, however, we show in Figure 8 that this leads to worse performance.

In-batch negatives There are two strategies to reduce the batch size memory requirement to that of a single batch while using nearly unlimited in-batch negatives. (1) Distributed Training: The best strategy is to distribute the training across up to bs GPUs. The representations can then be gathered across GPUs to compute the contrastive loss with in-batch negatives. (2) GradCache: If enough GPUs are not available, GradCache [51] can be used. GradCache maintains in-batch negatives while allowing computation of gradients for each triplet at a time, thus effectively corresponding to gradient accumulation for contrastive loss. However, it comes at the cost of additional forward passes.

Across training runs, we make use of all three strategies (splitting, distributed training, GradCache).

# I Hyperparameters

We finetune all parameters of our models for up to 1253 steps. Our learning rate is 2e-5, we use $3\%$ of steps for linear warm-up of the learning rate and decay it linearly to 0 over training. To save memory, we use PyTorch FSDP [184], gradient checkpointing, BF16 mixed precision training, and strategies outlined in Appendix H. During training, we use a sequence length of 2048 for generative samples, 256 for embedding queries, and 2048 for embedding documents unless otherwise specified.

We finetune using the Adam optimizer [81] with beta1=0.9 and beta2=0.999 and no weight decay. We also use Flash-Attention 2 [32, 31] via PyTorch SDPA.

We evaluate models using the settings put forth by the creators of MTEB $[109]$ , Tülu $[65, 162]$ and HumanEvalSynthesize $[107, 188]$ . For MTEB, we evaluate using a maximum sequence length of 512 unless otherwise specified.

# J Embedding Instruction for Generative Models

As prior instruction-tuned models have been trained without an embedding objective, it is unclear whether one should add an instruction when evaluating them on embedding tasks. We benchmark the Mistral 7B instruct model on MTEB with and without instruction in Table 27. We find that performance is around the same, however, adding instructions performs slightly better. Thus, we add an instruction for all instruction-tuned models when benchmarking their embedding performance.

Table 27: Benchmarking the benefit of an embedding instruction for generative instruction-tuned models. When an instruction is used ("Mistral Instruct w/"), we use the default instructions from Instructor XL with the prompt template of the Mistral Instruct model. For no instruction ("Mistral Instruct w/o"), the procedure is the same as for the base model ("Mistral") 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td>Mistral</td><td>63.5</td><td>34.6</td><td>53.5</td><td>43.2</td><td>13.2</td><td>57.4</td><td>19.7</td><td>40.5</td></tr><tr><td>Mistral Instruct w/o</td><td>65.4</td><td>35.6</td><td>60.2</td><td>44.6</td><td>16.8</td><td>61.1</td><td>25.9</td><td>43.3</td></tr><tr><td>Mistral Instruct w/</td><td>67.1</td><td>34.6</td><td>59.6</td><td>44.8</td><td>16.3</td><td>63.4</td><td>25.9</td><td>43.7</td></tr></table>

# K HumanEval Format

In Tülu 2 [65], models are evaluated on HumanEval [17] without the model's chat format. As this does not reflect the intended usage of the models, we instead use the appropriate chat format for evaluating HumanEval. To do so, we use the instructions and evaluation procedure from HumanEvalSynthesize [107]. In

Table 28: HumanEvalSynthesize with different formats using Tülu 2 7B. 

<table><tr><td rowspan="2">Format</td><td colspan="2">Tülu 2 7B</td></tr><tr><td>No Chat</td><td>Chat</td></tr><tr><td>Pass@1</td><td>23.4</td><td>24.5</td></tr><tr><td>Pass@10</td><td>32.4</td><td>31.3</td></tr></table>

Table 28 we benchmark the impact this has on performance for the Tülu 2 7B model [65]. We find that the performance is around equivalent and thus use the chat format for all evaluations of chat models. For non-chat models, we use the original HumanEval continuation format as proposed by Chen et al. [17]

# L Embedding in FP32 vs BF16

We perform all training and evaluations in BF16 (bfloat16) mixed precision to speed up computations. We verified that it performs comparably to FP32 (float32) on MTEB in Table 29. Note that pooling and subsequent similarity computations are still in FP32.

Table 29: Embeddings in FP32 vs BF16. Benchmarking of the raw Mistral 7B model. “FP32” corresponds to doing all computations in float32 precision. “BF16” and “BF16 Cache” corresponds to doing most operations in bfloat16 except for operations that PyTorch auto casts to float32 (e.g. normalization), pooling and similarity computations. For “BF16 Cache”, we cast the embeddings after pooling to BF16 and then back to FP32 before similarity computations. This corresponds to locally caching the embeddings in BF16 to save storage and then casting them to FP32 at inference. 

<table><tr><td>Task (→)Metric (→)Dataset # (→)</td><td>CLF Acc.12</td><td>Clust.V-Meas.11</td><td>PairCLF AP3</td><td>Rerank MAP4</td><td>Retrieval nDCG15</td><td>STS Spear.10</td><td>Summ.Spear.1</td><td>Avg.56</td></tr><tr><td>FP32</td><td>63.46</td><td>34.62</td><td>53.56</td><td>43.24</td><td>13.26</td><td>57.38</td><td>19.87</td><td>40.51</td></tr><tr><td>BF16</td><td>63.47</td><td>34.60</td><td>53.52</td><td>43.24</td><td>13.24</td><td>57.38</td><td>19.68</td><td>40.50</td></tr><tr><td>BF16 Cache</td><td>63.47</td><td>34.56</td><td>53.52</td><td>43.25</td><td>13.11</td><td>57.38</td><td>19.71</td><td>40.46</td></tr></table>

# M Unreliability of MT-Bench

We experiment with using MT-Bench with its recommended absolute scores for our generative evaluation $[185]$ . However, we find that as soon as we switch the LLM Evaluator from GPT-4 to GPT-4 Turbo, the scores change significantly (Table 30). GPT-4 is a closed-source model with changes happening behind the scenes that users may not know about $[16]$ . Thus, if OpenAI decides to change GPT-4, all existing MT-Bench absolute scores would essentially become obsolete. The same applies if the API is retired. To alleviate this, we also experiment with using Zephyr 7B $\beta$ $[157]$ and Llama 2 70B Chat $[156]$ as evaluators, however, we find them to often not provide any rating as they

Table 30: Using GPT-4 vs GPT-4 Turbo as a judge for MT-Bench. Each evaluator is provided with the same generations of the same instruction-tuned model. 

<table><tr><td></td><td>GPT-4</td><td>GPT-4 Turbo</td><td>Drop</td></tr><tr><td>Turn 1</td><td>4.08</td><td>3.05</td><td>25%</td></tr><tr><td>Turn 2</td><td>2.64</td><td>1.88</td><td>29%</td></tr><tr><td>Avg.</td><td>3.36</td><td>2.48</td><td>26%</td></tr></table>

struggle to understand the prompt. While AlpacaEval $[40, 91]$ , which we use, shares some of these problems, its comparison-based evaluation is more stable. This is because comparing if one generation is better than another generally has an objective ground truth solution. Meanwhile, there is no objective solution as to whether an absolute score of a given generation should be 3 or 4 (MT-Bench has eleven levels from 0-10). This is up to the subjective value system of the evaluator.

# N Dataset Composition

Table 31: E5S dataset composition. 

<table><tr><td>Dataset (↓)</td><td>Num samples</td></tr><tr><td>DuReader [123]</td><td>86,395</td></tr><tr><td>ELI5 [47]</td><td>50293</td></tr><tr><td>FEVER [155]</td><td>71,257</td></tr><tr><td>GPT4 Bitext [162]</td><td>89,324</td></tr><tr><td>GPT4 P2P [162]</td><td>16,842</td></tr><tr><td>GPT4 P2S [162]</td><td>121,878</td></tr><tr><td>GPT4 Retrieval [162]</td><td>166,602</td></tr><tr><td>GPT4 S2S [162]</td><td>13,481</td></tr><tr><td>GPT4 STS [162]</td><td>98,626</td></tr><tr><td>HotpotQA [173]</td><td>68,659</td></tr><tr><td>NLI [52]</td><td>275,601</td></tr><tr><td>MIRACL [183]</td><td>40,203</td></tr><tr><td>MSMARCO [8]</td><td>244,582</td></tr><tr><td>MSMARCO Doc [8]</td><td>71,594</td></tr><tr><td>Mr. TyDi [182]</td><td>48,729</td></tr><tr><td>NQ [83]</td><td>71,408</td></tr><tr><td>S2ORC [93]</td><td>80,000</td></tr><tr><td>SQuAD [130]</td><td>87,599</td></tr><tr><td>T2Ranking [172]</td><td>112,335</td></tr><tr><td>TriviaQA [77]</td><td>60,296</td></tr><tr><td>Quora [33]</td><td>14,926</td></tr><tr><td>Total</td><td>1,890,630</td></tr></table>

Table 32: MEDI2 dataset composition. 

<table><tr><td>MEDI Dataset (↓)</td><td>Num samples</td></tr><tr><td>AGNews [180]</td><td>199,792</td></tr><tr><td>Altlex [62]</td><td>112,602</td></tr><tr><td>Amazon QA [56]</td><td>199,180</td></tr><tr><td>Amazon Review [78]</td><td>198,298</td></tr><tr><td>CC News [60]</td><td>190,503</td></tr><tr><td>CNN/Dailymail [45]</td><td>189,407</td></tr><tr><td>COCO Captions [19]</td><td>82,783</td></tr><tr><td>ELI5 [47]</td><td>196,572</td></tr><tr><td>FEVER KILT [155, 120]</td><td>71,257</td></tr><tr><td>Flickr 30k [177]</td><td>31,783</td></tr><tr><td>Gigaword [134, 54]</td><td>200,000</td></tr><tr><td>GooAQ [79]</td><td>199,981</td></tr><tr><td>HotpotQA KILT [173, 120]</td><td>65,351</td></tr><tr><td>NLI [52]</td><td>277,195</td></tr><tr><td>MSMARCO [8]</td><td>491,980</td></tr><tr><td>MedMCQA [117]</td><td>156,905</td></tr><tr><td>Multi-LexSum [139]</td><td>2,771</td></tr><tr><td>NPR [153]</td><td>193,399</td></tr><tr><td>NQ [83]</td><td>73,226</td></tr><tr><td>PAQ [87]</td><td>190,162</td></tr><tr><td>PubMedQA [72]</td><td>190,481</td></tr><tr><td>Reddit [152]</td><td>196,247</td></tr><tr><td>S2ORC [93]</td><td>193,458</td></tr><tr><td>SQuAD [130]</td><td>84,105</td></tr><tr><td>SciTLDR [14]</td><td>1,742</td></tr><tr><td>SearchQA [41]</td><td>114,520</td></tr><tr><td>Sentence Compression [49]</td><td>179,996</td></tr><tr><td>SimpleWiki [29]</td><td>102,035</td></tr><tr><td>StackExchange [150]</td><td>201,050</td></tr><tr><td>SuperNI (300 datasets) [165]</td><td>2,682,465</td></tr><tr><td>SPECTER [26]</td><td>684,000</td></tr><tr><td>T-REx KILT [43, 120]</td><td>191,383</td></tr><tr><td>Quora [33]</td><td>101,762</td></tr><tr><td>WikiAnswers [46]</td><td>200,000</td></tr><tr><td>WikiHow [82]</td><td>128,542</td></tr><tr><td>XSum [111]</td><td>190,427</td></tr><tr><td>Yahoo [180]</td><td>198,346</td></tr><tr><td>Zeroshot KILT [85, 120]</td><td>124,547</td></tr><tr><td>Total</td><td>9,084,806</td></tr></table>

# O Dataset Samples

# Query instruction:

Represent the sentence for retrieving supporting documents;

# Query sample:

what two plates form the san andreas fault

# Positive instruction:

Represent the document for retrieval;

# Positive sample:

The San Andreas Fault marks the junction between the North American and Pacific Plates. The fault is 1300 km long, extends to at least 25 km in depth, and has a north west-south east trend. It is classified as a right lateral (dextral) strike-slip fault. Loading the player ...

# Negative instruction:

Represent the document for retrieval;

# Negative sample:

The San Andreas Fault is the sliding boundary between the Pacific Plate and the North American Plate. It slices California in two from Cape Mendocino to the Mexican border. San Diego, Los Angeles and Big Sur are on the Pacific Plate.

Figure 9: MEDI sample.

# Query instruction:

Represent this question to retrieve a fitting Wikipedia passage (formal)

# Query sample:

which two plates meet along the west coast of the USA

# Positive instruction:

Represent this Wikipedia text in order to get a user query which it answers!

# Positive sample:

on to a transitional deformation zone in the Chersky Range, then the Ulakhan Fault between it and the Okhotsk Plate, and finally the Aleutian Trench to the end of the Queen Charlotte Fault system. The westerly boundary is the Queen Charlotte Fault running offshore along the coast of Alaska and the Cascadia subduction zone to the north, the San Andreas Fault through California, the East Pacific Rise in the Gulf of California, and the Middle America Trench to the south.
On its western edge, the Farallon Plate has been subducting

# Negative instruction:

Represent this passage to easily find a natural-written user question that can be answered by it.

# Negative sample:

the continental margin.

Types.

There are two types of continental margins: äctiveänd passive margins.

Active margins are typically associated with lithospheric plate boundaries. These active margins can be convergent or transform margins, and are also places of high tectonic activity, including volcanoes and earthquakes. The West Coast of North America and South America are active margins. Active continental margins are typically narrow from coast to shelf break, with steep descents into trenches. Convergent active margins occur where oceanic plates meet continental

Figure 10: MEDI2 sample.

# Query instruction:

Given a question, retrieve Wikipedia passages that answer the question

# Query sample:

which two plates meet along the west coast of the USA

# Positive sample:

North American Plate boundary is the Queen Charlotte Fault running offshore along the coast of Alaska and the Cascadia subduction zone to the north, the San Andreas Fault through California, the East Pacific Rise in the Gulf of California, and the Middle America Trench to the south. On its western edge, the Farallon Plate has been subducting under the North American Plate since the Jurassic Period. The Farallon Plate has almost completely subducted beneath the western portion of the North American Plate leaving that part of the North American Plate in contact with the Pacific Plate as the San Andreas Fault. The Juan

# Negative sample:

Caribbean Plate Caribbean Plate The Caribbean Plate is a mostly oceanic tectonic plate underlying Central America and the Caribbean Sea off the north coast of South America. Roughly 3.2 million square kilometers (1.2 million square miles) in area, the Caribbean Plate borders the North American Plate, the South American Plate, the Nazca Plate and the Cocos Plate. These borders are regions of intense seismic activity, including frequent earthquakes, occasional tsunamis, and volcanic eruptions. The northern boundary with the North American plate is a transform or strike-slip boundary which runs from the border area of Belize, Guatemala (Motagua Fault), and Honduras in Central

Figure 11: E5 sample. The E5 dataset does not use instructions for documents, thus the positive and negative samples do not have instructions.

# Instruction:

Q: Lloyd, Mark, and Michael have their Pokemon cards collection. Currently, Mark has thrice as many cards as Lloyd but has 10 fewer cards than Michael. If Michael has 100 cards now, how many more cards should they collect so that all three of them will have a total of 300 cards?

A: 80

Explain how we arrive at this answer:

# Sample:

Explanation: Mark has 10 fewer cards than Michael so Mark has 100 cards - 10 cards = 90 cards. So, Lloyd has 90 cards / 3 = 30 cards. All three of them have 90 cards + 30 cards + 100 cards = 220 cards. Thus, they need to collect 300 cards - 220 cards = 80 more cards.

Figure 12: Tülu 2 sample.

# P Evaluation Prompts

# P.1 Embedding Prompts

Table 33 contains the prompt for each MTEB dataset when training on the E5 dataset, which are the same instructions as used in Wang et al. [162]. Table 34 contains the MTEB prompts we use when training on MEDI2, which we wrote ourselves. For models trained on MEDI, we use the instructions for Instructor-XL from Su et al. [145].

Table 33: Instructions used for evaluation on the MTEB benchmark when training with the E5 dataset. “STS\*” indicates we use the same instructions for all the STS tasks. For retrieval datasets, we do not use an instruction for the document and only display the query instruction. 

<table><tr><td>Task Name</td><td>Instruction</td></tr><tr><td>AmazonCounterfactualClassif.</td><td>Classify a given Amazon customer review text as either counterfactual or not-counterfactual</td></tr><tr><td>AmazonPolarityClassification</td><td>Classify Amazon reviews into positive or negative sentiment</td></tr><tr><td>AmazonReviewsClassification</td><td>Classify the given Amazon review into its appropriate rating category</td></tr><tr><td>Banking77Classification</td><td>Given a online banking query, find the corresponding intents</td></tr><tr><td>EmotionClassification</td><td>Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise</td></tr><tr><td>ImdbClassification</td><td>Classify the sentiment expressed in the given movie review text from the IMDB dataset</td></tr><tr><td>MassiveIntentClassification</td><td>Given a user utterance as query, find the user intents</td></tr><tr><td>MassiveScenarioClassification</td><td>Given a user utterance as query, find the user scenarios</td></tr><tr><td>MTOPDomainClassification</td><td>Classify the intent domain of the given utterance in task-oriented conversation</td></tr><tr><td>MTOPIntentClassification</td><td>Classify the intent of the given utterance in task-oriented conversation</td></tr><tr><td>ToxicConversationsClassif.</td><td>Classify the given comments as either toxic or not toxic</td></tr><tr><td>TweetSentimentClassification</td><td>Classify the sentiment of a given tweet as either positive, negative, or neutral</td></tr><tr><td>ArxivClusteringP2P</td><td>Identify the main and secondary category of Arxiv papers based on the titles and abstracts</td></tr><tr><td>ArxivClusteringS2S</td><td>Identify the main and secondary category of Arxiv papers based on the titles</td></tr><tr><td>BiorxivClusteringP2P</td><td>Identify the main category of Biorxiv papers based on the titles and abstracts</td></tr><tr><td>BiorxivClusteringS2S</td><td>Identify the main category of Biorxiv papers based on the titles</td></tr><tr><td>MedrxivClusteringP2P</td><td>Identify the main category of Medrxiv papers based on the titles and abstracts</td></tr><tr><td>MedrxivClusteringS2S</td><td>Identify the main category of Medrxiv papers based on the titles</td></tr><tr><td>RedditClustering</td><td>Identify the topic or theme of Reddit posts based on the titles</td></tr><tr><td>RedditClusteringP2P</td><td>Identify the topic or theme of Reddit posts based on the titles and posts</td></tr><tr><td>StackExchangeClustering</td><td>Identify the topic or theme of StackExchange posts based on the titles</td></tr><tr><td>StackExchangeClusteringP2P</td><td>Identify the topic or theme of StackExchange posts based on the given paragraphs</td></tr><tr><td>TwentyNewsgroupsClustering</td><td>Identify the topic or theme of the given news articles</td></tr><tr><td>SprintDuplicateQuestions</td><td>Retrieve duplicate questions from Sprint forum</td></tr><tr><td>TwitterSemEval2015</td><td>Retrieve tweets that are semantically similar to the given tweet</td></tr><tr><td>TwitterURLCorpus</td><td>Retrieve tweets that are semantically similar to the given tweet</td></tr><tr><td>AskUbuntuDupQuestions</td><td>Retrieve duplicate questions from AskUbuntu forum</td></tr><tr><td>MindSmallReranking</td><td>Retrieve relevant news articles based on user browsing history</td></tr><tr><td>SciDocsRR</td><td>Given a title of a scientific paper, retrieve the titles of other relevant papers</td></tr><tr><td>StackOverflowDupQuestions</td><td>Retrieve duplicate questions from StackOverflow forum</td></tr><tr><td>ArguAna</td><td>Given a claim, find documents that refute the claim</td></tr><tr><td>ClimateFEVER</td><td>Given a claim about climate change, retrieve documents that support or refute the claim</td></tr><tr><td>CQADupstackRetrieval</td><td>Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question</td></tr><tr><td>DBPedia</td><td>Given a query, retrieve relevant entity descriptions from DBPedia</td></tr><tr><td>FEVER</td><td>Given a claim, retrieve documents that support or refute the claim</td></tr><tr><td>FiQA2018</td><td>Given a financial question, retrieve user replies that best answer the question</td></tr><tr><td>HotpotQA</td><td>Given a multi-hop question, retrieve documents that can help answer the question</td></tr><tr><td>MSMARCO</td><td>Given a web search query, retrieve relevant passages that answer the query</td></tr><tr><td>NFCorpus</td><td>Given a question, retrieve relevant documents that best answer the question</td></tr><tr><td>NQ</td><td>Given a question, retrieve Wikipedia passages that answer the question</td></tr><tr><td>QuoraRetrieval</td><td>Given a question, retrieve questions that are semantically equivalent to the given question</td></tr><tr><td>SCIDOCS</td><td>Given a scientific paper title, retrieve paper abstracts that are cited by the given paper</td></tr><tr><td>SciFact</td><td>Given a scientific claim, retrieve documents that support or refute the claim</td></tr><tr><td>Touche2020</td><td>Given a question, retrieve detailed and persuasive arguments that answer the question</td></tr><tr><td>TRECCOVID</td><td>Given a query on COVID-19, retrieve documents that answer the query</td></tr><tr><td>STS*</td><td>Retrieve semantically similar text.</td></tr><tr><td>SummEval</td><td>Given a news summary, retrieve other semantically similar summaries</td></tr></table>

Table 34: Instructions used for evaluation on the MTEB benchmark when training with the MEDI2 dataset. For asymmetric datasets, Q refers to instructions for queries, while D refers to document instructions. 

<table><tr><td>Task Name</td><td>Instruction</td></tr><tr><td>AmazonCounterfactualClassification</td><td>Represent the text to find another sentence with the same counter-factuality, e.g. sentences with "would", "wish", etc. should match with other sentences of that kind.</td></tr><tr><td>AmazonPolarityClassification</td><td>Represent the review for finding another Amazon review with the same sentiment (positive / negative)</td></tr><tr><td>AmazonReviewsClassification</td><td>Represent the review for finding another Amazon review with the same rating</td></tr><tr><td>Banking77Classification</td><td>Represent the text for finding another one-sentence banking query with the same intent</td></tr><tr><td>EmotionClassification</td><td>Represent the text for finding another one-sentence text with the same emotion</td></tr><tr><td>ImdbClassification</td><td>Represent the text for finding another one-sentence movie review with the same sentiment</td></tr><tr><td>MassiveIntentClassification</td><td>Represent the text for finding another text of a few words with the same intent</td></tr><tr><td>MassiveScenarioClassification</td><td>Represent the text for finding another text of a few words about the same scenario</td></tr><tr><td>MTOPDomainClassification</td><td>Represent the text for finding another text of a few words about the same domain</td></tr><tr><td>MTOPIntentClassification</td><td>Represent the text for finding another text of a few words with the same intent</td></tr><tr><td>ToxicConversationsClassification</td><td>Represent the text for finding another comment of up to a passage in length with the same level of toxicity (either toxic or not toxic)</td></tr><tr><td>TweetSentimentExtractionClassification</td><td>Represent the tweet for finding another tweet with the same sentiment (positive / neutral / negative)</td></tr><tr><td>ArxivClusteringP2P</td><td>Represent the text to find another arXiv title with abstract (concatenated) about the same topic</td></tr><tr><td>ArxivClusteringS2S</td><td>Represent the text to find another arXiv title about the same topic</td></tr><tr><td>BiorxivClusteringP2P</td><td>Represent the text to find another bioRxiv title with abstract (concatenated) about the same topic</td></tr><tr><td>BiorxivClusteringS2S</td><td>Represent the text to find another bioRxiv title about the same topic</td></tr><tr><td>MedrxivClusteringS2S</td><td>Represent the text to find another medRxiv title about the same topic</td></tr><tr><td>MedrxivClusteringP2P</td><td>Represent the text to find another medRxiv title with abstract (concatenated) about the same topic</td></tr><tr><td>RedditClustering</td><td>Represent the text to find another Reddit community title that stems from the same subreddit</td></tr><tr><td>RedditClusteringP2P</td><td>Represent the text to find another Reddit community title with post (concatenated) from the same subreddit</td></tr><tr><td>StackExchangeClustering</td><td>Represent the text to find another StackExchange title that stems from the same StackExchange</td></tr><tr><td>StackExchangeClusteringP2P</td><td>Represent the text to find another StackExchange title with post (concatenated) that stems from the same StackExchange</td></tr><tr><td>TwentyNewsgroupsClustering</td><td>Represent the title to find a similar news title from the same newsgroup</td></tr><tr><td>SprintDuplicateQuestions</td><td>Represent the question to be matched with another duplicate user question from the Sprint community forum</td></tr><tr><td>TwitterSemEval2015</td><td>Represent the tweet to find another tweet that is a paraphrase of it</td></tr><tr><td>TwitterURLCorpus</td><td>Represent the tweet to find another tweet that is a paraphrase of it</td></tr><tr><td>ArguAna Q</td><td>Represent the passage to find a passage with a counter-argument about the same topic to it</td></tr><tr><td>ArguAna D</td><td>Represent the passage to find a passage with a counter-argument about the same topic to it</td></tr><tr><td>ClimateFEVER Q</td><td>Represent the climate-based claim to find a Wikipedia abstract to support it</td></tr><tr><td>ClimateFEVER D</td><td>Represent the Wikipedia abstract to find a climate-related claim that it supports</td></tr><tr><td>CQADupstackAndroidRetrieval Q</td><td>Represent the title of a user question to find a duplicate user question title with body from the Android StackExchange forum</td></tr><tr><td>CQADupstackAndroidRetrieval D</td><td>Represent the question title with body posted by a user to find a duplicate user question title from the Android StackExchange forum</td></tr><tr><td>CQADupstackEnglishRetrieval Q</td><td>Represent the title of a user question to find a duplicate user question title with body from the English StackExchange forum</td></tr><tr><td>CQADupstackEnglishRetrieval D</td><td>Represent the question title with body posted by a user to find a duplicate user question title from the English StackExchange forum</td></tr><tr><td>CQADupstackGamingRetrieval Q</td><td>Represent the title of a user question to find a duplicate user question title with body from the Gaming StackExchange forum</td></tr><tr><td>CQADupstackGamingRetrieval D</td><td>Represent the question title with body posted by a user to find a duplicate user question title from the Gaming StackExchange forum</td></tr><tr><td>CQADupstackGisRetrieval Q</td><td>Represent the title of a user question to find a duplicate user question title with body from the Gis StackExchange forum</td></tr><tr><td>CQADupstackGisRetrieval D</td><td>Represent the question title with body posted by a user to find a duplicate user question title from the Gis StackExchange forum</td></tr><tr><td>CQADupstackMathematicaRetrieval Q</td><td>Represent the title of a user question to find a duplicate user question title with body from the Mathematica StackExchange forum</td></tr><tr><td>CQADupstackMathematicaRetrieval D</td><td>Represent the question title with body posted by a user to find a duplicate user question title from the Mathematica StackExchange forum</td></tr><tr><td>CQADupstackPhysicsRetrieval Q</td><td>Represent the title of a user question to find a duplicate user question title with body from the Physics StackExchange forum</td></tr><tr><td>CQADupstackPhysicsRetrieval D</td><td>Represent the question title with body posted by a user to find a duplicate user question title from the Physics StackExchange forum</td></tr></table>

CQADupstackProgrammersRetrieval Q

CQADupstackProgrammersRetrieval D

CQADupstackStatsRetrieval Q

CQADupstackStatsRetrieval D

CQADupstackTexRetrieval Q

CQADupstackTexRetrieval D

CQADupstackUnixRetrieval Q

CQADupstackUnixRetrieval D

CQADupstackWebmastersRetrieval Q

CQADupstackWebmastersRetrieval D

CQADupstackWordpressRetrieval Q

CQADupstackWordpressRetrieval D

DBPedia Q

DBPedia D

FEVER Q

FEVER D

FiQA2018 Q

FiQA2018 D

HotpotQA Q

HotpotQA D

MSMARCO Q

MSMARCO D

NFCorpus Q

Represent the title of a user question to find a duplicate user question title with body from the Programmers StackExchange forum

Represent the question title with body posted by a user to find a duplicate user question title from the Programmers StackExchange forum

Represent the title of a user question to find a duplicate user question title with body from the Stats StackExchange forum

Represent the question title with body posted by a user to find a duplicate user question title from the Stats StackExchange forum

Represent the title of a user question to find a duplicate user question title with body from the Tex StackExchange forum

Represent the question title with body posted by a user to find a duplicate user question title from the Tex StackExchange forum

Represent the title of a user question to find a duplicate user question title with body from the Unix StackExchange forum

Represent the question title with body posted by a user to find a duplicate user question title from the Unix StackExchange forum

Represent the title of a user question to find a duplicate user question title with body from the Webmasters StackExchange forum

Represent the question title with body posted by a user to find a duplicate user question title from the Webmasters StackExchange forum

Represent the title of a user question to find a duplicate user question title with body from the WordPress StackExchange forum

Represent the question title with body posted by a user to find a duplicate user question title from the WordPress StackExchange forum

Represent the entity to find a title with abstract about this entity from the DBPedia corpus

Represent the title with abstract of a DBPedia corpus entry to find the entity of a few words it is about

Represent the claim to find a Wikipedia abstract to support it

Represent the Wikipedia abstract to find a claim that it supports

Represent the StackExchange user query to find a StackExchange post from the Investment topic that answers it

Represent the StackExchange post from the Investment topic to find a StackExchange user query that it answers

Represent the multi-hop question to find a Wikipedia passage that answers it

Represent the Wikipedia passage to find a multi-hop question that it answers

Represent the Bing user search query to find a passage that adequately addresses it

Represent the passage for finding a Bing user search query about it

Represent the query from NutritionFacts to find a title with text of a medical document from PubMed about it

<table><tr><td>NFCorpus D</td><td>Represent this text of a medical document from PubMed to find a query someone may enter at NutritionFacts that it answers</td></tr><tr><td>NQ Q</td><td>Represent the Google search query to find an answer span from a Wikipedia article that addresses it</td></tr><tr><td>NQ D</td><td>Represent the Wikipedia article span to find a Google search query that would be addressed by it</td></tr><tr><td>SCIDOCS Q</td><td>Represent the scientific paper title to find the title with abstract of a scientific paper on PubMed that it has likely cited</td></tr><tr><td>SCIDOCS D</td><td>Represent the title with abstract of this scientific paper to find the title of another scientific paper on PubMed that likely cites this article</td></tr><tr><td>SciFact Q</td><td>Represent the scientific claim to find a scientific paper abstract from PubMed to support it</td></tr><tr><td>SciFact D</td><td>Represent the scientific paper abstract from PubMed to find a scientific claim that it supports</td></tr><tr><td>TRECCOVID Q</td><td>Represent the search query to find a scientific article about COVID-19 that adequately addresses the query</td></tr><tr><td>TRECCOVID D</td><td>Represent the scientific article about COVID-19 to find a user query that it adequately addresses</td></tr><tr><td>Touche2020 Q</td><td>Represent the question to find a title with passage of an argument from args.me that takes a stance about it</td></tr><tr><td>Touche2020 D</td><td>Represent the title with passage of an argument from args.me to find a question that it takes a stance about</td></tr><tr><td>QuoraRetrieval Q</td><td>Represent the Quora question to find another short duplicate question on Quora</td></tr><tr><td>QuoraRetrieval D</td><td>Represent the Quora question to find another short duplicate question on Quora</td></tr><tr><td>AskUbuntuDupQuestions Q</td><td>Represent the query to find a duplicate query on the AskUbuntu community forum</td></tr><tr><td>AskUbuntuDupQuestions D</td><td>Represent the query to find a duplicate query on the AskUbuntu community forum</td></tr><tr><td>MindSmallReranking Q</td><td>Represent the news headline to find another news headline that the same reader would enjoy</td></tr><tr><td>MindSmallReranking D</td><td>Represent the news headline to find another news headline that the same reader would enjoy</td></tr><tr><td>SciDocsRR Q</td><td>Represent the title to find a similar scientific paper title</td></tr><tr><td>SciDocsRR D</td><td>Represent the title to find a similar scientific paper title</td></tr><tr><td>StackOverflowDupQuestions Q</td><td>Represent the query to find a duplicate query on the StackOverflow Java/JavaScript/Python community forums</td></tr><tr><td>StackOverflowDupQuestions D</td><td>Represent the query to find a duplicate query on the StackOverflow Java/JavaScript/Python community forums</td></tr><tr><td>BIOSSES</td><td>Represent the text to find another biological statement with the same meaning</td></tr><tr><td>SICK-R</td><td>Represent the sentence to find another sentence with the same meaning</td></tr><tr><td>STS12</td><td>Represent the sentence to find another sentence with the same meaning</td></tr><tr><td>STS13</td><td>Represent the sentence to find another sentence with the same meaning</td></tr><tr><td>STS14</td><td>Represent the sentence to find another sentence with the same meaning</td></tr><tr><td>STS15</td><td>Represent the sentence to find another sentence with the same meaning</td></tr><tr><td>STS16</td><td>Represent the sentence to find another sentence with the same meaning</td></tr><tr><td>STS17</td><td>Represent the sentence to find another sentence with the same meaning</td></tr><tr><td>STS22</td><td>Represent the sentence to find another sentence with the same meaning</td></tr><tr><td>STSBenchmark</td><td>Represent the sentence to find another sentence with the same meaning</td></tr><tr><td>SummEval Q</td><td>Represent the human-written summary to find a high-quality machine-written summary of the same news article</td></tr><tr><td>SummEval D</td><td>Represent the machine-written summary to find a human-written summary with similar quality of the same news article</td></tr></table>

# P.2 Embedding Few-Shot Prompts

Table 35: 1-shot example for the model trained on E5S. The example is appended to the respective instruction in Table 33 separated by two newlines. 

<table><tr><td>Task Name</td><td>Instruction</td></tr><tr><td>Banking77Classification</td><td>For example given "I am still waiting on my card?", it would match with "card_arrival"</td></tr><tr><td>EmotionClassification</td><td>For example given "ive been feeling a little burdened lately wasnt sure why that was", it would match with "sadness"</td></tr><tr><td>ImdbClassification</td><td>For example given "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.&lt;br /&gt;&lt;br /&gt;One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).&lt;br /&gt;&lt;br /&gt;One might better spend one's time staring out a window at a tree growing.&lt;br /&gt;&lt;br /&gt;", it would match with "negative"</td></tr><tr><td>BiorxivClusteringP2P</td><td>For example given "Association of CDH11 with ASD revealed by matched-gene co-expression analysis and mouse behavioral studies", it would match with "neuroscience"</td></tr><tr><td>TwitterSemEval2015</td><td>For example given "The Ending to 8 Mile is my fav part of the whole movie", it would match with "Those last 3 battles in 8 Mile are THE shit"</td></tr><tr><td>TwitterURLCorpus</td><td>For example given "Liberals, dont let Donald Trump tarnish L.L. Beans sterling brand reputation ", it would match with "Liber- als, Don&amp;rsquo;t Let Donald Trump Tarnish L.L. Bean&amp;rsquo;s Sterling Brand Reputation"</td></tr><tr><td>SprintDuplicateQuestions</td><td>For example given "Why is it impossible for me to find a easy way to send a picture with text on my Kyocera DuraCore?", it would match with "Send or receive a picture with text - Kyocera DuraCore"</td></tr><tr><td>AskUbuntuDupQuestions</td><td>For example given "what is a short cut i can use to switch appli- cations?", you should retrieve "keyboard short cut for switching between two or more instances of the same application?"</td></tr><tr><td>ArguAna</td><td>For example given "People will die if we don't do animal testing Every year, 23 new drugs are introduced in the UK alone.[13] Almost all will be tested on animals. A new drug will be used for a long time. Think of all the people saved by the use of penicillin. If drugs cost more to test, that means drug companies will develop less. This means more people suffering and dying", you should retrieve "animals science science general ban animal testing junior Many of these drugs are "me too" drugs – ones with a slight change that doesn't make much difference to an existing drug. [14] So often the benefits from animal testing are marginal, and even if there was a slight increase in human suffering, it would be worth it based on the animal suffering saved."</td></tr><tr><td>SCIDOCS</td><td>For example given "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect", you should retrieve "A Hybrid EP and SQP for Dynamic Economic Dispatch with Nonsmooth Fuel Cost Function Dynamic economic dispatch (DED) is one of the main functions of power generation opera- tion and control. It determines the optimal settings of generator units with predicted load demand over a certain period of time. The objective is to operate an electric power system most eco- nomically while the system is operating within its security limits. This paper proposes a new hybrid methodology for solving DED. The proposed method is developed in such a way that a simple evolutionary programming (EP) is applied as a based level search, which can give a good direction to the optimal global region, and a local search sequential quadratic programming (SQP) is used as a fine tuning to determine the optimal solution at the final. Ten units test system with nonsmooth fuel cost function is used to illustrate the effectiveness of the proposed method compared with those obtained from EP and SQP alone."</td></tr><tr><td>STS12</td><td>For example given "Counties with population declines will be Ver- million, Posey and Madison.", it would match with "Vermillion, Posey and Madison County populations will decline."</td></tr></table>

SummEval

The provided query could be "Mexican restaurant has decided to tap into \$70 billion food delivery market. Fast-casual chain will work with the Postmates app to allow mobile orders. App works in similar way to Uber, using hired drivers to deliver the food. But the chain will add a 9% service charge - on top of Postmates\$5 rate." and the positive "chipotle has decided to tap into the \$70 billion food delivery market by teaming up with an app to bring burritos straight to customers doors . the fast-casual chain will work with the postmates app to begin offering delivery for online and mobile orders in 67 cities . the restaurant plans to add a nine per cent service charge - with the delivery fees for postmates beginning at \$5 and up depending on distance and demand ."

Table 36: 1-shot example for the model trained on MEDI2. The example is appended to the respective instruction in Table 34 separated by two newlines. 

<table><tr><td>Task Name</td><td>Instruction</td></tr><tr><td>Banking77Classification</td><td>The provided query could be "I am still waiting on my card?" and the positive "What can I do if my card still hasn't arrived after 2 weeks?"</td></tr><tr><td>EmotionClassification</td><td>The provided query could be "ive been feeling a little burdened lately wasnt sure why that was" and the positive "i feel like i have to make the suffering i m seeing mean something"</td></tr><tr><td>ImdbClassification</td><td>The provided query could be "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.&lt;br /&gt;&lt;br /&gt;One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).&lt;br /&gt;&lt;br /&gt;One might better spend one's time staring out a window at a tree growing.&lt;br /&gt;&lt;br /&gt;" and the positive "The silent one-panel cartoon Henry comes to Fleischer Studios, billed as "The world's funniest human" in this dull little cartoon. Betty, long past her prime, thanks to the Production Code, is running a pet shop and leaves Henry in charge for far too long – five minutes. A bore."</td></tr><tr><td>SprintDuplicateQuestions</td><td>The provided query could be "Why is it impossible for me to find a easy way to send a picture with text on my Kyocera DuraCore?" and the positive "Send or receive a picture with text - Kyocera DuraCore"</td></tr><tr><td>TwitterSemEval2015</td><td>For example given "The Ending to 8 Mile is my fav part of the whole movie", it would match with "Those last 3 battles in 8 Mile are THE shit"</td></tr><tr><td>TwitterURLCorpus</td><td>For example given "Liberals , dont let Donald Trump tarnish L.L. Beans sterling brand reputation ", it would match with "Liberals, Don&amp;rsquo;t Let Donald Trump Tarnish L.L. Bean&amp;rsquo;s Sterling Brand Reputation"</td></tr><tr><td>AskUbuntuDupQuestions</td><td>The provided query could be "what is a short cut i can use to switch applications ?" and the positive "keyboard short cut for switching between two or more instances of the same application ?"</td></tr><tr><td>ArguAna</td><td>The provided query could be "People will die if we don't do animal testing Every year, 23 new drugs are introduced in the UK alone.[13] Almost all will be tested on animals. A new drug will be used for a long time. Think of all the people saved by the use of penicillin. If drugs cost more to test, that means drug companies will develop less. This means more people suffering and dying" and the positive "animals science science general ban animal testing junior Many of these drugs are "me too" drugs – ones with a slight change that doesn't make much difference to an existing drug. [14] So often the benefits from animal testing are marginal, and even if there was a slight increase in human suffering, it would be worth it based on the animal suffering saved."</td></tr><tr><td>SCIDOCS</td><td>The provided query could be "A Direct Search Method to solve Economic Dispatch Problem with Valve-Point Effect" and the positive "A Hybrid EP and SQP for Dynamic Economic Dispatch with Nonsmooth Fuel Cost Function Dynamic economic dispatch (DED) is one of the main functions of power generation operation and control. It determines the optimal settings of generator units with predicted load demand over a certain period of time. The objective is to operate an electric power system most economically while the system is operating within its security limits. This paper proposes a new hybrid methodology for solving DED. The proposed method is developed in such a way that a simple evolutionary programming (EP) is applied as a based level search, which can give a good direction to the optimal global region, and a local search sequential quadratic programming (SQP) is used as a fine tuning to determine the optimal solution at the final. Ten units test system with nonsmooth fuel cost function is used to illustrate the effectiveness of the proposed method compared with those obtained from EP and SQP alone."</td></tr><tr><td>STS12</td><td>The provided query could be "Counties with population declines will be Vermillion, Posey and Madison." and the positive "Vermillion, Posey and Madison County populations will decline."</td></tr><tr><td>SummEval</td><td>The provided query could be "Mexican restaurant has decided to tap into $70 billion food delivery market. Fast-casual chain will work with the Postmates app to allow mobile orders. App works in similar way to Uber, using hired drivers to deliver the food. But the chain will add a 9% service charge - on top of Postmates$5 rate." and the positive "chipotle has decided to tap into the $ 70 billion food delivery market by teaming up with an app to bring burritos straight to customers doors . the fast-casual chain will work with the postmates app to begin offering delivery for online and mobile orders in 67 cities . the restaurant plans to add a nine per cent service charge - with the delivery fees for postmates beginning at $ 5 and up depending on distance and demand ."</td></tr></table>

# P.3 Generative Prompts

Figure 13 until Figure 18 contain the prompts with examples used for our generative tasks.

# Input:

<s><|user|>

The following are multiple choice questions (with answers) about abstract algebra.

Find the degree for the given field extension $\mathrm{Q}(\mathrm{sqrt}(2), \mathrm{sqrt}(3), \mathrm{sqrt}(18))$ over $\mathrm{Q}$ .

A. 0

B. 4

C. 2

D. 6

Answer:

<lassistant|>

The answer is:

# Correct completion:

B

Figure 13: MMLU prompt example.

# Input:

<s><|user|>

Answer the following questions.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been $21 - 15 = 6$ . So the answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

Answer: There are originally 3 cars. 2 more cars arrive. $3 + 2 = 5$ . So the answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had $32 + 42 = 74$ . After eating 35, they had $74 - 35 = 39$ . So the answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny $20 - 12 = 8$ . So the answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. $5 + 4 = 9$ . So the answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 \* 4 = 20 computers were added. $9 + 20$ is 29. So the answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had $58 - 23 = 35$ . After losing 2 more, he had $35 - 2 = 33$ golf balls. So the answer is 33.

Question: Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left? Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Question: The girls are trying to raise money for a carnival. Kim raises \$320 more than Alexandra, who raises \$430, and Maryam raises \$400 more than Sarah, who raises \$300. How much money, in dollars, did they all raise in total?

<|assistant|>

Answer:

# Correct completion:

Kim raises 320+430=750 dollars. Maryam raises 400+300=700 dollars. They raise 750+430+400+700=2280 dollars. So the answer is 2280.

Figure 14: GSM8K prompt example.

# Input:

<s><|user|>

Questions that involve enumerating objects and asking the model to count them.

Q: I have a blackberry, a clarinet, a nectarine, a plum, a strawberry, a banana, a flute, an orange, and a violin. How many fruits do I have?

A: Let's think step by step.

We first identify the fruits on the list and include their quantity in parentheses:

- blackberry (1)   
- nectarine (1)   
- plum (1)   
- strawberry (1)   
- banana (1)   
- orange (1)

Now, let's add the numbers in parentheses: $1 + 1 + 1 + 1 + 1 + 1 = 6$ . So the answer is 6.

Q: I have an orange, a raspberry, two peaches, a blackberry, an apple, a grape, a nectarine, and three plums. How many fruits do I have?

A: Let's think step by step.

We first identify the fruits on the list and include their quantity in parentheses:

- orange (1)   
- raspberry (1)   
- peaches (2)   
- blackberry (1)   
- apple (1)   
- grape (1)   
- nectarine (1)   
- plums (3)

Now, let's add the numbers in parentheses: $1 + 1 + 2 + 1 + 1 + 1 + 1 + 3 = 11$ . So the answer is 11.

Q: I have a lettuce head, a head of broccoli, an onion, a stalk of celery, two carrots, a garlic, and a yam. How many vegetables do I have?

A: Let's think step by step.

We first identify the vegetables on the list and include their quantity in parentheses:

- lettuce (1)   
- broccoli (1)   
- onion (1)   
- celery (1)   
- carrots (2)   
- garlic (1)   
- yam (1)

Now, let's add the numbers in parentheses: $1 + 1 + 1 + 1 + 2 + 1 + 1 = 8$ . So the answer is 8.

Q: I have a banana, four strawberries, an apple, two peaches, a plum, a blackberry, and two raspberries. How many fruits do I have?

<lassistant|>

# Correct completion:

12

Figure 15: BBH prompt example.

# Input:

<s><|user|>

Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.

Bagian: Mula-mula pada pelukis seorang pelukis pemandangan Wahdi Sumanta, Abdullah Suriosubroto (ayah Basuki Abdullah). Kemudian bertemu dan berkenalan dengan Affandi, Sudarso, dan Barli. Mereka lalu membentuk kelompok Lima serangkai. Di rumah tempat tinggal Affandi mereka mengadakan latihan melukis bersama dengan tekun dan mendalam. Dari Wahdi, ia banyak menggali pengetahuan tentang melukis. Kegiatannya bukan hanya melukis semata, tetapi pada waktu senggang ia menceburkan diri pada kelompok sandiwara Sunda sebagai pelukis dekor. Dari pengalaman itulah, ia mengasah kemampuannya.

Pertanyaan: dari manakah Hendra Gunawan belajar melukis?

Jawaban: kelompok Lima serangkai

Bagian: Empat Sehat Lima Sempurna adalah kampanye yang dilakukan pemerintah sejak tahun 1955 untuk membuat masyarakat memahami pola makan yang benar.[1]. Dalam konsep 4 sehat 5 sempurna, makanan dibagi atas empat sumber nutrisi penting, yaitu makanan pokok, lauk pauk, sayur-mayur, buah-buahan, dan disempurnakan dengan susu bila mampu, menjadi lima sempurna[2] Konsep ini menekankan pentingnya empat golongan makanan berupa sumber kalori untuk tenaga, protein untuk pembangun, sayur dan buah sumber vitamin dan mineral untuk pemeliharaan.[1]

Pertanyaan: siapakah yang mencptakan Ide 4 sehat 5 sempurna pertama kali?

<|assistant|>

Jawaban:

# Correct completion:

pemerintah

Figure 16: TyDi QA prompt example from Indonesian.

# Input:

<|user|>

Write a Python function ‘has\_close\_elements(numbers: List[float], threshold: float) -> bool’ to solve the following problem: Check if in given list of numbers, are any two numbers closer to each other than given threshold.

```txt
>>> has_close_elements([1.0, 2.0, 3.0], 0.5)
False
>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
True 
```  
<lassistant|>

```python
from typing import List
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
given threshold.
>>> has_close_elements([1.0, 2.0, 3.0], 0.5)
False
>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
True
""" 
```

# Correct completion:

```python
for idx, elem in enumerate(numbers):
    for idx2, elem2 in enumerate(numbers):
    if idx != idx2:
    distance = abs(elem - elem2)
    if distance < threshold:
    return True
return False 
```  
Figure 17: HumanEvalSynthesize prompt example.

# Input:

<s><|user|>

If you were a Shakespearean character, how would you declare your love for someone in a soliloquy?

# Completion by GPT-4 Turbo:

As an AI, I lack personal emotions and experiences, but I can craft a soliloquy in the style of a Shakespearean character declaring love. Here's an example:

(Imagine a dimly lit stage, a single spotlight shining down upon the forlorn lover, who steps forward to share the depths of their heart.)

O, speak, thou lustrous beacon of the night,
With stars enkindled in thine eyes so bright.
Thou art the daybreak to my endless gloom,
The rose that in my barren heart doth bloom

Forsooth, this love, a tempest full of grace,
Hath swept upon my soul without a trace.
My heart, once caged, now beats beneath thy gaze,
And yearns to laud thy beauty all its days.

What art thou, Love? A tyrant or a muse?
That rends my sense and all my thoughts confuse?
Thou art the riddle none can e'er resolve,
Yet in thy puzzle, my life doth revolve.

With every breath, I thee do magnify,
And to the world, my passion testify.
Thine essence hath become my sole refrain,
A melody that in my mind doth reign.

Shall I compare thee to a summer's day? Thou art more lovely and more temperate. Rough winds do shake the darling buds of May, And summer's lease hath all too short a date.

But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st.
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st.

So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee.

(With a heavy sigh, the lover steps back, their heart laid bare, the words hanging in the air like the fragrance of an unseen bloom, their love declared, though the object of their affection remains unaware, the audience enthralled by the vulnerability of true, unspoken love.)

Figure 18: AlpacaEval prompt example.

# P.4 RAG Prompts

# Generative input:

```txt
<s><|user|>
what was the name of darth vader star destroyer
<|assistant|
The answer is 
```

# Generation:

```txt
not clear. Darth Vader had a star destroyer, but it was 
```  
Figure 19: No RAG example and generation by GRITLM 7B.

# Generative input:

```txt
<s><|user|>
what was the name of darth vader star destroyer 
```

Star Destroyer The iconic Star Destroyer first appears in the opening scene of Star Wars, as Darth Vader's flagship, the Devastator, chases the Tantive IV above Tatooine. This appearance shows the Imperial ship's massive size in comparison to the Tantive IV.

```txt
Optionally using the prior context answer the query prior to it <lassistant>
The answer is 
```

# Generation:

```txt
the Devastator.</s> 
```  
Figure 20: RAG query then document example and generation by GRITLM 7B.

# Generative input:

```txt
<s><|user|> 
```

Star Destroyer The iconic Star Destroyer first appears in the opening scene of Star Wars, as Darth Vader's flagship, the Devastator, chases the Tantive IV above Tatooine. This appearance shows the Imperial ship's massive size in comparison to the Tantive IV.

what was the name of darth vader star destroyer

Answer the prior query while optionally using the context prior to it <lassistant>
The answer is

# Generation:

```txt
the Devastator.</s> 
```  
Figure 21: RAG document then query example and generation by GRITLM 7B.

# Embedding input (query) passed via key-value states:

```twig
<s><|embed|> 
```

what was the name of darth vader star destroyer

# Generative input:

```txt
\n<|user|> 
```

Star Destroyer The iconic Star Destroyer first appears in the opening scene of Star Wars, as Darth Vader's flagship, the Devastator, chases the Tantive IV above Tatooine. This appearance shows the Imperial ship's massive size in comparison to the Tantive IV.

Optionally using the prior context answer the query prior to it <lassistant>
The answer is

# Generation:

```erb
Star Destroyer.</s> 
```  
Figure 22: GRIT Query Caching example and generation by GRITLM 7B.

# Embedding input (doc) passed via key-value states and cached in the index:

```twig
<s><|embed|> 
```

Star Destroyer The iconic Star Destroyer first appears in the opening scene of Star Wars, as Darth Vader's flagship, the Devastator, chases the Tantive IV above Tatooine. This appearance shows the Imperial ship's massive size in comparison to the Tantive IV.

# Generative input:

```txt
\n<|user|> 
```

what was the name of darth vader star destroyer

Answer the prior query while optionally using the context prior to it

```twig
<lassistant|> 
```

The answer is

# Generation:

Devastator. The iconic Star Destroyer first appears in the opening

Figure 23: GRIT Doc Caching example and generation by GRITLM 7B.

# Embedding input (doc) passed via key-value states and cached in the index:

```twig
<s><|embed|> 
```

Star Destroyer The iconic Star Destroyer first appears in the opening scene of Star Wars, as Darth Vader's flagship, the Devastator, chases the Tantive IV above Tatooine. This appearance shows the Imperial ship's massive size in comparison to the Tantive IV.

# Embedding input (query) passed via key-value states:

```twig
<s><|embed|> 
```

what was the name of darth vader star destroyer

# Generative input:

```txt
\n<|user|> 
```

Answer the prior query while optionally using the context prior to it

```twig
<lassistant|> 
```

The answer is

# Generation:

the Star Destroyer. The Star Destroyer is a massive spacecraft

Figure 24: GRIT Doc-Query Caching example and generation by GRITLM 7B. Unlike for Doc Caching, we prepend the bos token (“<s>”) to both query and document, which improved the match score from 14.13 to 18.39.

Embedding input (query) passed via key-value states:   
```txt
<s><|embed|>
what was the name of darth vader star destroyer 
```

Embedding input (doc) passed via key-value states and cached in the index:   
```txt
<lembed>
Star Destroyer The iconic Star Destroyer first appears in the opening scene of Star Wars, as Darth Vader's flagship, the Devastator, chases the Tantive IV above Tatooine. This appearance shows the Imperial ship's massive size in comparison to the Tantive IV. 
```

Generative Input:   
```txt
\n<luserl>
Optionally using the prior context answer the query prior to it
<classistantl>
The answer is 
```

Generation:   
```txt
the Star Destroyer. 
```  
Figure 25: GRIT Query-Doc Caching example and generation by GRITLM 7B.

# Q Hardware

For the training of GRITLM 7B, we used 8 nodes with 8 NVIDIA A100 80GB GPUs each for 48 hours corresponding to 3,072 GPU hours. Meanwhile for GRITLM 8x7B, we used 32 nodes with 8 NVIDIA H100 80GB GPUs each for 80 hours corresponding to 20,480 GPU hours. As we train both models for 1253 steps, this corresponds to several minutes per step. This slow training time is mainly due to (a) a large batch size per step, (b) large models and our associated strategies to make them fit into memory at the cost of speed (Appendix H, Appendix I), and (c) a cluster with slow inter-node communication. The Gen.-only and Emb.-only models in Table 1 used 72 and 1760 H100 80GB GPU hours, respectively. Adding up all ablations and evaluations, we likely used somewhere around 100,000 GPU hours.

# R Limitations and Future Work

GRITLM Agents Future work may consider using the embedding capability to let the generative model initiate a search over an index when it deems necessary. Currently, this is often accomplished via external retrieval plugins. Such plugins are no longer necessary if the model can retrieve on its own. Teaching the model to invoke its own embedding capability likely requires additional finetuning (just like teaching it to invoke an external plugin $[137]$ ). A sample could look something like:

“<|user|>\nWhat is the capital of Japan?\n<|internal|>\nI am not sure I know this. Let me produce an embedding for it and search for the answer. Retrieve answers for this query.\n<|embed|>\nWhat is the capital of Japan?\n<|output|>\nTokyo, Japan's busy capital, mixes the ultramodern and the traditional..\n<|assistant|>\nThe capital of Japan is Tokyo.\n</s>”

Pretraining For our experiments we take an off-the-shelf pretrained language model. However, it should also be possible to use the GRIT approach to pretrain from scratch. As labeled embedding data is likely too scarce for pretraining, one could either rely on unsupervised approaches for the embedding objective, such as RetroMAE $[170, 169]$ , or use methods like data augmentation $[35]$ , pruning $[168]$ or multi-epoch training to deal with the data constraint $[108, 96]$ .

Format Efficiency Our format in Figure 3 is inefficient, as encoding the embedding format, <s><|user|>\n<|embed|>\n, requires 13 tokens and encoding the generative format, <s><|user|>\n<|assistant|>\n</s>, requires 15 tokens. Using special tokens could simplify this and thus make training and inference slightly cheaper.

Training efficiency: Packing and Reusing It is common to pack samples during generative instruction tuning to maximize efficiency $[23, 110]$ . Packing embedding samples during training should also be possible by ensuring attention is only paid to each respective sample. Going even further is it possible to pack generative and embedding training data into the same sample and reuse the same sample for both tasks? This could look similar to the example provided in “GRITLM Agents” with the generative loss applied over the assistant response and the contrastive loss applied to the representation of the text following “<|embed|>”. By reusing samples it may be possible to significantly decrease the resources needed for GRIT.

# S Version Control

V2 → V3 (2025-03):

• Fixed a figure reference in §4   
- Elaborated more on training discussion and potential avenues for improvement in Appendix R   
- Added Appendix D   
- Rephrased the end of §1 to better motivate that GritLM requires less compute than separate generative and embedding models when considering pretraining   
- Added more discussion on the potential speed-performance trade-off of using a smaller and faster embedding model by using the embedding from intermediate layers of GritLM in §3.3 after our embedding head ablation   
- Added resources used by Gen.-only and Emb.-only baselines in Appendix Q

V1 → V2 (2024-04):

- Added KTO experiments in §3.2   
• Fixed sample in Figure 12