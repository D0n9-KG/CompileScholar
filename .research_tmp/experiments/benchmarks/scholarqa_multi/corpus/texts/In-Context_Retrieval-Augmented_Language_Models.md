# In-Context Retrieval-Augmented Language Models

Ori Ram* Yoav Levine* Itay Dalmedigos Dor Muhlgay Amnon Shashua Kevin Leyton-Brown Yoav Shoham AI21 Labs

{orir, yoavl, itayd, dorm, amnons, kevinlb, yoavs}@ai21.com

# Abstract

Retrieval-Augmented Language Modeling (RALM) methods, which condition a language model (LM) on relevant documents from a grounding corpus during generation, were shown to significantly improve language modeling performance. In addition, they can mitigate the problem of factually inaccurate text generation and provide natural source attribution mechanism. Existing RALM approaches focus on modifying the LM architecture in order to facilitate the incorporation of external information, significantly complicating deployment.

This paper considers a simple alternative, which we dub In-Context RALM: leaving the LM architecture unchanged and prepending grounding documents to the input, without any further training of the LM. We show that In-Context RALM that builds on off-the-shelf general purpose retrievers provides surprisingly large LM gains across model sizes and diverse corpora. We also demonstrate that the document retrieval and ranking mechanism can be specialized to the RALM setting to further boost performance.

We conclude that In-Context RALM has considerable potential to increase the prevalence of LM grounding, particularly in settings where a pretrained LM must be used without modification or even via API access.<sup>1</sup>

# 1 Introduction

Recent advances in language modeling (LM) have dramatically increased the usefulness of machine-generated text across a wide range of use-cases and domains (Brown et al., 2020). However, the mainstream paradigm of generating text with LMs bears inherent limitations in access to external knowledge. First, LMs are not coupled with any

![](dt=2026-06-04/ht=15/ad85421ec2f6c90a27c591f5fea3c375312e7716071267406e60cb09bcd4d582.jpg)

source attribution, and must be trained in order to incorporate up-to-date information that was not seen during training. More importantly, they tend to produce factual inaccuracies and errors (Lin et al., 2022; Maynez et al., 2020; Huang et al., 2020). This problem is present in any LM generation scenario, and is exacerbated when generation is made in uncommon domains or private data.

A promising approach for addressing the above is Retrieval-Augmented Language Modeling (RALM), grounding the LM during generation by conditioning on relevant documents retrieved from an external knowledge source. RALM systems include two high level components: (i) document selection, selecting the set of documents upon which to condition; and (ii) document reading, determining how to incorporate the selected documents into the LM generation process.

Leading RALM systems introduced recently

arXiv:2302.00083v3 [cs.CL] 1 Aug 2023

* Equal contribution.

Our code is available at https://github.com/

AI21Labs/in-context-ralm

![](dt=2026-06-04/ht=15/cd33bb3e78f38da88feec8e0c91239a1e151bd53734b9818f836361a07356324.jpg)

tend to be focused on altering the language model architecture (Khandelwal et al., 2020; Borgeaud et al., 2022; Zhong et al., 2022; Levine et al., 2022c; Li et al., 2022). Notably, Borgeaud et al. (2022) introduced RETRO, featuring document reading via nontrivial modifications that require further training to the LM architecture, while using an off-the-shelf frozen BERT retriever for document selection. Although the paper's experimental findings showed impressive performance gains, the need for changes in architecture and dedicated retraining has hindered the wide adoption of such models.

In this paper, we show that a very simple document reading mechanism can have a large impact, and that substantial gains can also be made by adapting the document selection mechanism to the task of language modeling. Thus, we show that many of the benefits of RALM can be achieved while working with off-the-shelf LMs, even via API access. Specifically, we consider a simple but powerful RALM framework, dubbed In-Context RALM (presented in Section 3), which employs a zero-effort document reading mechanism: we simply prepend the selected documents to the LM's input text (Figure 2).

Section 4 describes our experimental setup. To show the wide applicability of our framework, we performed LM experiments on a suite of five diverse corpora: WikiText-103 (Merit et al., 2016), RealNews (Zellers et al., 2019), and three datasets from The Pile (Gao et al., 2021): ArXiv, Stack Exchange and FreeLaw. We use open-source LMs ranging from 110M to 66B parameters (from the GPT-2, GPT-Neo, OPT and LLaMA model families).

In Section 5 we evaluate the application of off-the-shelf retrievers to our framework. In this minimal-effort setting, we found that In-Context RALM led to LM performance gains equivalent to increasing the LM's number of parameters by $2 - 3 \times$ across all of the text corpora we examined. In Section 6 we investigate methods for adapting doc

ument ranking to the LM task, a relatively underexplored RALM degree of freedom. Our adaptation methods range from using a small LM to perform zero-shot ranking of the retrieved documents, up to training a dedicated bidirectional reranker by employing self-supervision from the LM signal. These methods lead to further gains in the LM task corresponding to an additional size increase of $2 \times$ in the LM architecture.

As a concrete example of the gains, a 345M parameter GPT-2 enhanced by In-Context RALM outperforms a 762M parameter GPT-2 when employing an off-the-shelf BM25 retriever (Robertson and Zaragoza, 2009), and outperforms a 1.5B parameter GPT-2 when employing our trained LM-oriented reranker (see Figure 1). For large model sizes, our method is even more effective: In-Context RALM with an off-the-shelf retriever improved the performance of a 6.7B parameter OPT model to match that of a 66B parameter parameter OPT model (see Figure 4).

In Section 7 we demonstrate the applicability of In-Context RALM to downstream open-domain questions answering (ODQA) tasks.

In a concurrent work, Shi et al. (2023) also suggest to augment off-the-shelf LMs with retrieved texts by preponding them to the input. Their results are based on training a dedicated retriever for language modeling. In contrast, we focus on the gains achievable in using off-the-shelf retrievers for this task. We show strong gains of this simpler setting by investigating: (1) which off-the-shelf retriever is best suited for language modeling, (2) the frequency of retrieval operations, and (3) the optimal query length. In addition, we boost the off-the-shelf retrieval performance by introducing two reranking methods that demonstrate further gains in perplexity.

We believe that In-Context RALM can play two important roles in making RALM systems more powerful and more prevalent. First, given its simple reading mechanism, In-Context RALM can serve as a clean probe for developing document retrieval

methods that are specialized for the LM task. These in turn can be used to improve both In-Context RALM and other more elaborate RALM methods that currently leverage general purpose retrievers. Second, due to its compatibility with off-the-shelf LMs, In-Context RALM can help drive wider deployment of RALM systems.

# 2 Related Work

RALM approaches can be roughly divided into two families of models: (i) nearest-neighbor language models (also called kNN-LM), and (ii) retrieve and read models. Our work belongs to the second family, but is distinct in that it involves no further training of the LM.

Nearest Neighbor Language Models The $k$ -NN-LM approach was first introduced in Khandelwal et al. (2020). The authors suggest a simple inference-time model that interpolates between two next-token distributions: one induced by the
LM itself, and one induced by the $k$ neighbors from the retrieval corpus that are closest to the query token in the LM embedding space. Zhong et al. (2022) suggest a framework for training these models.

While they showed significant gains from $k$ -NN-LM, the approach requires storing the representations for each token in the corpus, an expensive requirement even for a small corpus like Wikipedia. Although numerous approaches have been suggested for alleviating this issue (He et al., 2021; Alon et al., 2022), scaling any of them to large corpora remains an open challenge.

Retrieve and Read Models This family of RALMs creates a clear division between document selection and document reading components. All prior work involves training the LM. We begin by describing works that use this approach for tackling downstream tasks, and then mention works oriented towards RALM. Lewis et al. (2020) and Izacard and Grave (2021) fine tuned encoder-decoder architectures for downstream knowledge-intensive tasks. Izacard et al. (2022b) explored different ways of pretraining such models, while Levine et al.

(2022c) pretrained an autoregressive LM on clusters of nearest neighbors in sentence embedding space. Levine et al. (2022a) showed competitive open domain question-answering performance by prompt-tuning a frozen LM as a reader. Guu et al. (2020) pretrained REALM, a retrieval augmented bidirectional, masked LM, later fine-tuned

for open-domain question answering. The work closest to this paper—with a focus on the language modeling task—is RETRO (Bergeaud et al., 2022), which modifies an autoregressive LM to attend to relevant documents via chunked cross-attention, thus introducing new parameters to the model. Our In-Context RALM differs from prior work in this family of models in two key aspects:

# 3 Our Framework

# 3.1 In-Context RALM

Language models define probability distributions over sequences of tokens. Given such a sequence $x_{1}, \ldots, x_{n}$ , the standard way to model its probability is via next-token prediction: $p(x_{1}, \ldots, x_{n}) = \prod_{i=1}^{n} p(x_{i} | x_{<i})$ , where $x_{<i} := x_{1}, \ldots, x_{i-1}$ is the sequence of tokens preceding $x_{i}$ , also referred to as its prefix. This autoregressive model is usually implemented via a learned transformer network (Vaswani et al., 2017) parameterized by the set of parameters $\theta$ :

$$
p \left(x _ {1}, \dots , x _ {n}\right) = \prod_ {i = 1} ^ {n} p _ {\theta} \left(x _ {i} \mid x _ {<   i}\right), \tag {1}
$$

where the conditional probabilities are modeled by employing a causal self-attention mask (Radford et al., 2018). Notably, leading LMs such as GPT-2 (Radford et al., 2019), GPT-3 (Brown et al., 2020), OPT (Zhang et al., 2022) or Jurassic-1 (Lieber et al., 2021) follow this simple parameterization.

Retrieval augmented language models (RALMs) add an operation that retrieves one or more documents from an external corpus $\mathcal{C}$ , and condition the above LM predictions on these documents. Specifically, for predicting $x_{i}$ , the retrieval operation from $\mathcal{C}$ depends on its prefix: $\mathcal{R}_{\mathcal{C}}(x_{< i})$ , so the most general RALM decomposition is: $p(x_{1},\dots,x_{n}) = \prod_{i=1}^{n} p(x_{i}|x_{< i},\mathcal{R}_{\mathcal{C}}(x_{< i}))$ . In order to condition the LM generation on the retrieved document, previous RALM approaches used specialized architectures or algorithms (see §2). Inspired by the success of In-Context Learning (Brown et al., 2020; Dong et al., 2023), In-Context RALM refers to the following specific, simple method of concatenating

the retrieved documents $^2$ within the Transformer's input prior to the prefix (see Figure 2), which does not involve altering the LM weights $\theta$ :

$$
\begin{array}{l} p (x _ {1}, \ldots , x _ {n}) = \\ \prod_ {i = 1} ^ {n} p _ {\theta} \left(x _ {i} \mid \left[ \mathcal {R} _ {\mathcal {C}} \left(x _ {<   i}\right); x _ {<   i} \right]\right), \tag {2} \\ \end{array}
$$

where $[a; b]$ denotes the concatenation of strings $a$ and $b$ .

Since common Transformer-based LM implementations support limited length input sequences, when the concatenation of the document and the input sequence exceeds this limit we remove tokens from the beginning of $x$ until the overall input length equals that allowed by the model. Because our retrieved documents are passages of limited length, we always have enough context left from $x$ (see §4.3).

# 3.2 RALM Design Choices

We detail below two practical design choices often made in RALM systems. In §5, we investigate the effect of these in the setting of In-Context RALM.

Retrieval Stride While in the above formulation a retrieval operation can occur at each generation step, we might want to perform retrieval only once every $s > 1$ tokens due to the cost of calling the retriever, and the need to replace the documents in the LM prefix during generation. We refer to $s$ as the retrieval stride. This gives rise to the following In-Context RALM formulation (which reduces back to Eq. (2) for $s = 1$ ):

$$
\begin{array}{l} p (x _ {1}, \ldots , x _ {n}) = \\ \prod_ {j = 0} ^ {n _ {s} - 1} \prod_ {i = 1} ^ {s} p _ {\theta} \left(x _ {s \cdot j + i} \mid \left[ \mathcal {R} _ {\mathcal {C}} \left(x _ {\leq s \cdot j}\right); x _ {<   (s \cdot j + i)} \right]\right), \tag {3} \\ \end{array}
$$

where $n_s = n / s$ is the number of retrieval strides.

Notably, in this framework the runtime costs of each retrieval operation is composed of (a) applying the retriever itself, and (b) recomputing the embeddings of the prefix. In §5.2 we show that using smaller retrieval strides, i.e., retrieving as often as possible, is superior to using larger ones (though In-Context RALM with larger strides already provides large gains over vanilla LM). Thus, choosing the retrieval stride is ultimately a tradeoff between runtime and performance.

Retrieval Query Length While the retrieval query above in principle depends on all prefix tokens $x_{\leq s \cdot j}$ , the information at the very end of the prefix is typically the most relevant to the generated tokens. If the retrieval query is too long then this information can be diluted. To avoid this, we restrict the retrieval query at stride $j$ to the last $\ell$ tokens of the prefix, i.e., we use $q_j^{s,\ell} := x_{s \cdot j - \ell + 1}, \ldots, x_{s \cdot j}$ . We refer to $\ell$ as the retrieval query length. Note that prior RALM work couples the retrieval stride $s$ and the retrieval query length $\ell$ (Borgeaud et al., 2022). In §5, we show that enforcing $s = \ell$ degrades LM performance. Integrating these hyper-parameters into the In-Context RALM formulation gives

$$
\begin{array}{l} p (x _ {1}, \dots , x _ {n}) = \\ \prod_ {j = 0} ^ {n _ {s} - 1} \prod_ {i = 1} ^ {s} p _ {\theta} \left(x _ {s \cdot j + i} \mid \left[ \mathcal {R} _ {\mathcal {C}} \left(q _ {j} ^ {s, \ell}\right); x _ {<   (s \cdot j + i)} \right]\right). \tag {4} \\ \end{array}
$$

# 4 Experimental Details

We now describe our experimental setup, including all models we use and their implementation details.

# 4.1 Datasets

We evaluated the effectiveness of In-Context RALM across five diverse language modeling datasets and two common open-domain question answering datasets.

Language Modeling The first LM dataset is WikiText-103 (Merit et al., 2016), which has been extensively used to evaluate RALMs (Khandelwal et al., 2020; He et al., 2021; Borgeaud et al., 2022; Alon et al., 2022; Zhong et al., 2022). Second, we chose three datasets spanning diverse subjects from The Pile (Gao et al., 2021): ArXiv, Stack Exchange and FreeLaw. Finally, we also investigated RealNews (Zellers et al., 2019), since The Pile lacks a corpus focused only on news (which is by nature a knowledge-intensive domain).

Open-Domain Question Answering In order to evaluate In-Context RALM on downstream tasks as well, we use the Natural Questions (NQ; Kwiatkowski et al. 2019) and TriviaQA (
Joshi et al., 2017) open-domain question answering datasets.

# 4.2 Models

Language Models We performed our experiments using the four models of GPT-2 (110M-1.5B; Radford et al. 2019), three models of GPTNeo and GPT-J (1.3B-6B; Black et al. 2021; Wang

2We always use a single document, but it is conceptually simple to support multiple documents as well.

and Komatsuzaki 2021), eight models of OPT (125M-66B; Zhang et al. 2022) and three models of LLaMA (7B-33B; Touvron et al. 2023). All models are open source and publicly available. $^{3}$

We elected to study these particular models for the following reasons. The first four (GPT-2) models were trained on WebText (Radford et al., 2019), with Wikipedia documents excluded from their training datasets. We were thus able to evaluate our method's "zero-shot" performance when retrieving from a novel corpus (for WikiText-103). The rest of the models brought two further benefits. First, they allowed us to investigate how our methods scale to models larger than GPT-2.

Second, the fact that Wikipedia was part of their training data allowed us to investigate the usefulness of In-Context RALM for corpora seen during training. The helpfulness of such retrieval has been demonstrated for previous RALM methods (Khandelwal et al., 2020) and has also been justified theoretically by Levine et al. (2022c).

We ran all models with a maximum sequence length of 1,024, even though GPT-Neo, OPT and LLaMA models support a sequence length of 2,048.

Retrievers We experimented with both sparse (word-based) and dense (neural) retrievers. We used BM25 (Robertson and Zaragoza, 2009) as our sparse model. For dense models, we experimented with (i) a frozen BERT-base (Devlin et al., 2019) followed by mean pooling, similar to Borgeaud et al. (2022); and (ii) the Contriever (Izacard et al., 2022a) and Spider (Ram et al., 2022) models, which are dense retrievers that were trained in unsupervised manners.

Reranking When training rerankers (Section 6.2), we initialized from RoBERTa-base (Liu et al., 2019).

# 4.3 Implementation Details

We implemented our code base using the Transformers library (Wolf et al., 2020). We based our dense retrieval code on the DPR repository (Karpukhin et al., 2020).

![](dt=2026-06-04/ht=15/b0ef8b013c5b0b7f3ec0544357283fd83fbb3c87c06fce2ee73ff2a78e26a852.jpg)

Retrieval Corpora For WikiText-103 and ODQA datasets, we used the Wikipedia corpus from Dec. 20, 2018, standardized by Karpukhin et al. (2020) using the preprocessing from Chen et al. (2017). To avoid contamination, we found and removed all 120 articles of the development and test set of WikiText-103 from the corpus. For the remaining datasets, we used their training data as the retrieval corpus. Similar to Karpukhin et al. (2020), our retrieval corpora consist of non-overlapping passages of 100 words (which translate to less than 150 tokens for the vast majority of passages). Thus, we truncate our retrieved passages at 256 tokens when input to the models, but they are usually much smaller.

Retrieval For sparse retrieval, we used the Py-serini library (Lin et al., 2021). For dense retrieval, we applied exact search using FAISS (Johnson et al., 2021).

# 5 The Effectiveness of In-Context RALM with Off-the-Shelf Retrievers

We now empirically show that despite its simple document reading mechanism, In-Context RALM leads to substantial LM gains across our diverse evaluation suite. We begin in this section by investigating the effectiveness of off-the-shelf retrievers for In-Context RALM; we go on in §6 to show that further LM gains can be made by tailoring document ranking functions to the LM task.

The experiments in this section provided us with a recommended configuration for applying In

3All models are available for use use via https://huggingface.co/

4 In preliminary experiments, we observed similar improvements from In-Context RALM when using a sequence length of 2,048. We used a sequence length of 1,024 in order to facilitate a direct comparison between all models.

Table 1: Perplexity on the test set of WikiText-103, RealNews and three datasets from the Pile. For each LM, we report: (a) its performance without retrieval, (b) its performance when fed the top-scored passage by BM25 (\$5), and (c) its performance when applied on the top-scored passage of each of our two suggested rerankers (\$6). All models share the same vocabulary, thus token-level perplexity (token ppl) numbers are comparable. For WikiText we follow prior work and report word-level perplexity (word ppl).

![](dt=2026-06-04/ht=15/e4c338cee1a8e189d03980a0d347410eecfba1ea45e8342b506a93c2578f0f2d.jpg)

<table><tr><td rowspan="2">Model</td><td rowspan="2">Retrieval</td><td rowspan="2">Reranking</td><td>WikiText-103</td><td>RealNews</td><td>ArXiv</td><td>Stack Exch.</td><td>FreeLaw</td></tr><tr><td>word ppl</td><td>token ppl</td><td>token ppl</td><td>token ppl</td><td>token ppl</td></tr><tr><td rowspan="4">GPT-2 S</td><td>-</td><td>-</td><td>37.5</td><td>21.3</td><td>12.0</td><td>12.8</td><td>13.0</td></tr><tr><td>BM25 §5</td><td>-</td><td>29.6</td><td>16.1</td><td>10.9</td><td>11.3</td><td>9.6</td></tr><tr><td>BM25</td><td>Zero-shot §6.1</td><td>28.6</td><td>15.5</td><td>10.1</td><td>10.6</td><td>8.8</td></tr><tr><td>BM25</td><td>Predictive §6.2</td><td>26.8</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="4">GPT-2 M</td><td>-</td><td>-</td><td>26.3</td><td>15.7</td><td>9.3</td><td>8.8</td><td>9.6</td></tr><tr><td>BM25 §5</td><td>-</td><td>21.5</td><td>12.4</td><td>8.6</td><td>8.1</td><td>7.4</td></tr><tr><td>BM25</td><td>Zero-shot §6.1</td><td>20.8</td><td>12.0</td><td>8.0</td><td>7.7</td><td>6.9</td></tr><tr><td>BM25</td><td>Predictive §6.2</td><td>19.7</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="4">GPT-2 L</td><td>-</td><td>-</td><td>22.0</td><td>13.6</td><td>8.4</td><td>8.5</td><td>8.7</td></tr><tr><td>BM25 §5</td><td>-</td><td>18.1</td><td>10.9</td><td>7.8</td><td>7.8</td><td>6.8</td></tr><tr><td>BM25</td><td>Zero-shot §6.1</td><td>17.6</td><td>10.6</td><td>7.3</td><td>7.4</td><td>6.4</td></tr><tr><td>BM25</td><td>Predictive §6.2</td><td>16.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="4">GPT-2 XL</td><td>-</td><td>-</td><td>20.0</td><td>12.4</td><td>7.8</td><td>8.0</td><td>8.0</td></tr><tr><td>BM25 §5</td><td>-</td><td>16.6</td><td>10.1</td><td>7.2</td><td>7.4</td><td>6.4</td></tr><tr><td>BM25</td><td>Zero-shot §6.1</td><td>16.1</td><td>9.8</td><td>6.8</td><td>7.1</td><td>6.0</td></tr><tr><td>BM25</td><td>Predictive §6.2</td><td>15.4</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

Table 2: The performance of models from the LLaMA family, measured by word-level perplexity on the test set of WikiText-103.

![](dt=2026-06-04/ht=15/3c27eac8139f7ab5fe2b135858ae789dbab2c9a272cdc6d667f0d31cb33db605.jpg)

<table><tr><td>Model</td><td>Retrieval</td><td>WikiText-103
word ppl</td></tr><tr><td rowspan="2">LLaMA-7B</td><td>-</td><td>9.9</td></tr><tr><td>BM25, §5</td><td>8.8</td></tr><tr><td rowspan="2">LLaMA-13B</td><td>-</td><td>8.5</td></tr><tr><td>BM25, §5</td><td>7.6</td></tr><tr><td rowspan="2">LLaMA-33B</td><td>-</td><td>6.3</td></tr><tr><td>BM25, §5</td><td>6.1</td></tr></table>

Context RALM: applying a sparse BM25 retriever that receives $\ell = 32$ query tokens and is applied as frequently as possible. Practically, we retrieve every $s = 4$ tokens ( $\ell$ and $s$ are defined in §3). Table 1 shows for the GPT-2 models that across all the examined corpora, employing In-Context RALM with an off-the-shelf retriever improved LM perplexity to a sufficient extent that it matched that of a $2 - 3 \times$ larger model. Figure 4 and Tables 2 and 5 show that this trend hold
s across model sizes up to 66B parameters, for both WikiText-103 and

RealNews.

# 5.1 BM25 Outperforms Off-the-Shelf Neural Retrievers in Language Modeling

We experimented with different off-the-shelf general purpose retrievers, and found that the sparse (lexical) BM25 retriever (Robertson and Zaragoza, 2009) outperformed three popular dense (neural) retrievers: the self-supervised retrievers Contriever (Izacard et al., 2022a) and Spider (Ram et al., 2022), as well as a retriever based on the average pooling of BERT embeddings that was used in the RETRO system (Borgeaud et al., 2022). We conducted a minimal hyper-parameter search on the query length $\ell$ for each of the retrievers, and found that $\ell = 32$ was optimal for BM25 (Figure 6), and $\ell = 64$ worked best for dense retrievers (Figures 9, 10).

Figure 3 compares the performance gains of In-Context RALM with these four general-purpose retrievers. The BM25 retriever clearly outperformed all dense retrievers. This outcome is consistent with prior work showing that BM25 outperforms neural retrievers across a wide array of tasks, when applied in zero-shot settings (Thakur et al., 2021). This result renders In-Context RALM even more

![](dt=2026-06-04/ht=15/50f5e20e2e014f0951eb912c46dbe8918b0aa2d67afa212c5f2c39e9687e5e5a.jpg)

![](dt=2026-06-04/ht=15/18a9cad953cccb34156523fcaaa8461d31c49f2788a07f3bf3be88f86049ff4a.jpg)

appealing since applying a BM25 retriever is significantly cheaper than the neural alternatives.

# 5.2 Frequent Retrieval Improves Language Modeling

We investigated the effect of varying the retrieval stride $s$ (i.e., the number of tokens between consecutive retrieval operations). Figure 5 shows that LM performance improved as the retrieval operation became more frequent. This supports the intuition that retrieved documents become more relevant the closer the retrieval query becomes to the generated tokens. Of course, each retrieval operation imposes a runtime cost. To balance performance and runtime, we used $s = 4$ in our experiments. For comparison, RETRO employed a retrieval frequency of $s = 64$ (Borgeaud et al., 2022), which leads to large degradation in perplexity. Intuitively, retrieving with high frequency (low retrieval stride) allows to ground the LM in higher resolution.

# 5.3 A Contextualization vs. Recency Tradeoff in Query Length

We also investigated the effect of varying $\ell$ , the length of the retrieval query for BM25. Figure 6 reveals an interesting tradeoff and a sweet spot around a query length of 32 tokens. Similar experiments for dense retrievers are given in App. A. We conjecture that when the retriever query is too short, it does not include enough of the input context, decreasing the retrieved document's relevance. Conversely, excessively growing the retriever query deemphasizes the tokens at the very end of the prefix, diluting the query's relevance to the LM task.

# 6 Improving In-Context RALM with LM-Oriented Reranking

Since In-Context RALM uses a fixed document reading component by definition, it is natural to ask whether performance can be improved by specializing its document retrieval mechanism to the LM task. Indeed, there is considerable scope for improvement: the previous section considered conditioning the model only on the first document re

![](dt=2026-06-04/ht=15/f9ac32fabf0f339cef329ba9111ff2a21e3cb4f0fb9e3c86eb7a33ab5909448d.jpg)

![](dt=2026-06-04/ht=15/97f17a3bca5223a30c62b5fee795b29ca16ec35795c425724c84d6e423141ad5.jpg)

trieved by the BM25 retriever. This permits very limited semantic understanding of the query, since BM25 is based only on the bag of words signal. Moreover, it offers no way to accord different degrees of importance to different retrieval query tokens, such as recognizing that later query tokens are more relevant to the generated text.

In this section, we focus on choosing which document to present to the model, by reranking the top- $k$ documents returned by the BM25 retriever. We use Figure 7 as motivation: it shows the large potential for improvement among the top-16 documents returned by the BM25 retriever. We act upon

![](dt=2026-06-04/ht=15/5cba1e6daa7478e2849e8d00239b5297142ab135f5b9ebb077fe757e85e4654c.jpg)

this motivation by using two rerankers. Specifically, in §6.1 we show performance gains across our evaluation suite obtained by using an LM to perform zero-shot reranking of the top- $k$ BM25 retrieved documents (results in third row for each of the models in Table 1). Then, in §6.2 we show that training a specialized bidirectional reranker of the top- $k$ BM25 retrieved documents in a self-supervised manner via the LM signal can provide further LM gains (results in forth row for each of the models in Table 1).

# 6.1 LMs as Zero-Shot Rerankers

First, we used off-the-shelf language models as document rerankers for the In-Context RALM setting. Formally, for a query $q$ consisting of the last $\ell$ tokens in the prefix of the LM input $x$ , let $\{d_1, \dots, d_k\}$ be the top- $k$ documents returned by BM25. For retrieval iteration $j$ , let the text for generation be $y := x_{s \cdot j + 1}, \dots, x_{s \cdot j + s}$ . Ideally, we would like to find the document $d_i^*$ that maximizes the probability of the text for generation, i.e.,

$$
i ^ {*} = \arg \max  _ {i \in [ k ]} p _ {\theta} (y | [ d _ {i}; x _ {\leq s \cdot j} ]). \tag {5}
$$

However, at test time we do not have access to the tokens of $y$ . Instead, we used the last prefix tokens (which are available at test time), denoted by $y'$ , for reranking. Formally, let $s'$ be a hyper-parameter that determines the number of the prefix tokens by which to rerank. We define $y' := x_{s \cdot j - s' + 1}, \ldots, x_{s \cdot j}$ (i.e., the stride of length $s'$ that precedes $y$ ) and choose the document $d_i$ such

In both §6.1 and §6.2 we use $k = 16$ .

Table 3: Perplexity for zero-shot reranking (§6.1) where the reranking models is smaller than the LM, or the LM itself. Reranking is performed on the top 16 documents retrieved by BM25. Using a GPT-2 110M (S) instead of a larger language model as a reranker leads to only a minor degradation.

![](dt=2026-06-04/ht=15/3d04b0bfc4579f74ac91ed3ba4ef940a785cb09e5dbcc0a220fb6ae98f0350de.jpg)

<table><tr><td rowspan="2">Model</td><td rowspan="2">Reranking Model</td><td>WikiText-103</td><td>RealNews</td></tr><tr><td>word ppl</td><td>token ppl</td></tr><tr><td rowspan="2">GPT-2 345M (M)</td><td>GPT-2 110M (S)</td><td>20.8</td><td>12.1</td></tr><tr><td>GPT-2 345M (M)</td><td>20.8</td><td>12.0</td></tr><tr><td rowspan="2">GPT-2 762M (L)</td><td>GPT-2 110M (S)</td><td>17.7</td><td>10.7</td></tr><tr><td>GPT-2 762M (L)</td><td>17.6</td><td>10.6</td></tr><tr><td rowspan="2">GPT-2 1.5B (XL)</td><td>GPT-2 110M (S)</td><td>16.2</td><td>9.9</td></tr><tr><td>GPT-2 1.5B (XL)</td><td>16.1</td><td>9.8</td></tr></table>

that

$$
\hat {i} = \arg \max  _ {i \in [ k ]} p _ {\phi} \left(y ^ {\prime} \mid \left[ d _ {i}; x _ {\leq (s \cdot j - s ^ {\prime})} \right]\right). \tag {6}
$$

The main motivation is that since BM25 is a lexical retriever, we want to incorporate a semantic signal induced by the LM. Also, this reranking shares conceptual similarities with the reranking framework of Sachan et al. (2022) for open-domain question answering, where $y'$ (i.e., the last prefix tokens) can be thought of as their "question".

Note that our zero-shot reranking does not require that the LM used for reranking is the same model as th
e LM used for generation (i.e., the LM in Eq. (6), parameterized by $\phi$ , does not need to be the LM in Eq. (2), parameterized by $\theta$ ). This observation unlocks the possibility of reranking with smaller (and thus faster) models, which is important for two main reasons: (i) Reranking $k$ documents requires $k$ forward passes; and (ii) it allows our methods to be used in cases where the actual LM's log probabilities are not available (for example, when the LM is accessed through an API).

Results A minimal hyper-parameter search on the development set of WikiText-103 revealed that the optimal query length is $s' = 16$ , so we proceed with this value going forward. Table 1 shows the results of letting the LM perform zero-shot reranking on the top-16 documents retrieved by BM25 (third row for each of the models). It is evident that reranking yielded consistently better results than simply taking the first result returned by the retriever.

Table 3 shows that a small LM (GPT-2 117M) can be used to rerank the documents for all larger GPT-2 models, with roughly the same performance as having each LM perform reranking for itself, supporting the applicability of this method for LMs that are only accessible via an API.

# 6.2 Training LM-dedicated Rerankers

Next, we trained a reranker to choose one of the top- $k$ documents retrieved by BM25. We refer to this approach as Predictive Reranking, since the reranker learns to choose which document will help in "predicting" the upcoming text. For this process, we assume availability of training data from the target corpus. Our reranker is a classifier that gets a prefix $x_{\leq s \cdot j}$ and a document $d_i$ (for $i \in [k]$ ), and produces a scalar $f(x_{\leq s \cdot j}, d_i)$ that should resemble the relevance of $d_i$ for the continuation of $x_{\leq s \cdot j}$ .

We then normalize these relevance scores:

$$
p _ {\text {r a n k}} \left(d _ {i} \mid x _ {\leq s \cdot j}\right) = \frac {\exp \left(f \left(x _ {\leq s \cdot j} , d _ {i}\right)\right)}{\sum_ {i ^ {\prime} = 1} ^ {k} \exp \left(f \left(x _ {\leq s \cdot j} , d _ {i ^ {\prime}}\right)\right)}, \tag {7}
$$

and choose the document $d_{\dot{i}}$ such that

$$
\hat {i} = \arg \max  _ {i \in [ k ]} p _ {\text {r a n k}} \left(d _ {i} \mid x _ {\leq s \cdot j}\right). \tag {8}
$$

Collecting Training Examples To train our predictive reranker, we collected training examples as follows. Let $x_{\leq s \cdot j}$ be a prefix we sample from the training data, and $y := x_{s \cdot j + 1}, \ldots, x_{s \cdot j + s}$ be the text for generation upcoming in its next stride. We run BM25 on the query $q_j^{s,\ell}$ derived from $x_{\leq s \cdot j}$ (see §3.2) and get $k$ documents $\{d_1, \ldots, d_k\}$ . For each document $d_i$ , we then run the LM to compute $p_\theta(y|[d_i; x_{\leq s \cdot j}])$ similar to Eq. (4).

Note we do not require that the two models share the same vocabulary.

<sup>7</sup>We experimented with $s' \in \{4, 8, 16, 32\}$ .

![](dt=2026-06-04/ht=15/c2e73a288e4cf8cef2b493b5884fa04b9f3a0376ba651501c7dd272115de1424.jpg)

Training Our reranker was a fine-tuned RoBERTa-base (Liu et al., 2019) that trained for 10,000 steps with a peak learning rate of $10^{-5}$ and a batch size of 32. Overall, we created 300,000 examples from the training set of WikiText-103 as explained above. The loss function we use to train the reranker follows previous work (Guu et al., 2020; Lewis et al., 2020):

$$
- \log \sum_ {i = 1} ^ {k} p _ {\text {r a n k}} \left(d _ {i} \mid x _ {\leq s \cdot j}\right) \cdot p _ {\theta} (y | [ d _ {i}; x _ {\leq s \cdot j} ]). \tag {9}
$$

Note that unlike those works, we train only the reranker $(p_{\mathrm{rank}})$ , keeping the LM weights $\theta$ frozen.

Results Table 1 shows the result of our predictive reranker, trained on WikiText-103. Specifically, we trained it with data produced by GPT-2 110M (S), and tested its effectiveness for all GPT-2 models. We observed significant gains obtained from Predictive Reranking. For example, the perplexity of GPT-2 110M (S) improved from 29.6 to 26.8, and that of GPT-2 1.5B (XL) improved from 16.6 to 15.4. This trend held for the other two models as well. Overall, these results demonstrate that training a reranker with domain-specific data was more effective than zero-shot reranking (Section 6.1).

Note that these results—while impressive—still leave room for further improvements, compared to the top-16 BM25 oracle results (see Figure 7). Moreover, the oracle results themselves can be improved by retrieving $k > 16$ documents via a BM25 retriever, or by training stronger retrievers dedicated to the RALM task. We leave this direction for future work.

Table 4: Zero-shot results of In-Context RALM on the test set of Natural Questions and TriviaQA measured by exact match. In the open-book setting, we include the top two documents returned by DPR.

![](dt=2026-06-04/ht=15/636e500093fba2047570af1e7823bdca1c15df92f7a961275c1f226395c82248.jpg)

<table><tr><td>Model</td><td>Retrieval</td><td>NQ</td><td>TriviaQA</td></tr><tr><td rowspan="2">LLaMA-7B</td><td>-</td><td>10.3</td><td>47.5</td></tr><tr><td>DPR</td><td>28.0</td><td>56.0</td></tr><tr><td rowspan="2">LLaMA-13B</td><td>-</td><td>12.0</td><td>54.8</td></tr><tr><td>DPR</td><td>31.0</td><td>60.1</td></tr><tr><td rowspan="2">LLaMA-33B</td><td>-</td><td>13.7</td><td>58.3</td></tr><tr><td>DPR</td><td>32.3</td><td>62.7</td></tr></table>

# 7 In-Context RALM for Open-Domain Question Answering

So far, we evaluated our framework on language modeling benchmarks. To test its efficacy in additional scenarios, and specifically downstream tasks, we now turn to evaluate In-Context RALM on open-domain question answering (ODQA; Chen et al. 2017). This experiment is intended to verify, in a controlled environment, that LMs can leverage retrieved documents without further training and without any training examples. Specifically, we use the LLaMA family (Touvron et al.

, 2023) with and without In-Context RALM (often referred to in ODQA literature as open-book and closed-book settings, respectively). In contrast to most prior work on ODQA (e.g., Izacard and Grave 2021; Fajcik et al. 2021; Izacard et al. 2022b; Levine et al. 2022b), our "reader" (i.e., the model that gets the question along with its corresponding retrieved documents, and returns the answer) is simply a frozen large LM: not pretrained, fine-tuned or prompted to be retrieval-augmented. For the closed-book setting, we utilize the prompt of Touvron et al. (2023).

For the open-book setting, we extend this prompt to include retrieved documents (see App. C). We use DPR (Karpukhin et al., 2020) as our retriever.

Varying the Number of Documents To investigate the effect of the number of documents shown to the model, we performed a minimal analysis on the development set of NQ and TriviaQA. Figure 8 demonstrates that showing documents in context significantly improves the model's performance. In addition, most of the gain can be obtained by using only two documents (or even a single one in some cases).

Results Table 4 gives the results of In-Context RALM on the test set of Natural Questions and TriviaQA. Motivated by our previous findings, we used two retrieved documents. It is evident that showing the model relevant documents significantly boosted its performance. For example, adding retrieved documents improved LLaMA-13B in the zero-shot setting by more than 18 points on NQ (from $12.0\%$ to $31.0\%$ ) and more than 5 points on TriviaQA (from $54.8\%$ to $60.1\%$ ).

# 8 Discussion

Retrieval from external sources has become a common practice in knowledge-intensive tasks (such as factual question answering, fact checking, and more; Petroni et al. 2021). In parallel, recent breakthroughs in LM generation capabilities has le
d to LMs that can generate useful long texts. However, factual inaccuracies remain a common way in which machine-generated text can fall short, and lack of direct provenance makes it hard to trust machine generated text.

This makes language modeling both a promising and an urgent new application area for knowledge grounding, and motivates promoting RALM approaches. Prior research has already investigated RALM, of course, but it is not yet widely deployed. One likely reason is that existing approaches rely upon fine-tuning the LM, which is typically difficult and costly, and is even impossible for LMs accessible only via an API.

This paper presented the framework of In-Context RALM, enabling frozen, off-the-shelf LMs to benefit from retrieval. We demonstrated that substantial performance gains can be achieved by using general purpose retrievers, and showed that additional gains can be achieved by tailoring the document selection to the LM setting. A recent work by Muhlgay et al. (2023) demonstrates that In-Context RALM is indeed able to improve the factuality of large LMs.

Several directions for further improvement remain for future work. First, this paper considers only the case of prepending a single external document to the context; adding more documents could drive further gains (for example, using the framework of Ratner et al. 2022). Second, we retrieved documents every fixed interval of $s$ tokens, but see potential for large latency and cost gains by retrieving more sparsely, such as only when a specialized model predicts that retrieval is needed.

We release the code used in this work, for the

community to use and improve over. We hope it will drive further research of RALM, which will enable its wider adoption.

# Acknowledgements

We would like to thank the reviewers and the Action Editor for their valuable feedback.

# References

Uri Alon, Frank Xu, Junxian He, Sudipta Sengupta, Dan Roth, and Graham Neubig. 2022. Neurosymbolic language modeling with automaton-augmented retrieval. In ICML.

Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow.

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, JeanBaptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego De Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack Rae, Erich Elsen, and Laurent Sifre. 2022. Improving language models by retrieving from trillions of tokens. In ICML.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems.

Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer open-domain questions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long

# A Query Length Ablations

Figure 9 and Figure 10 show ablations on the optimal query length $\ell$ for off-the-shelf dense retrievers (BERT and Contriever respectively). We omit the results of Spider as they are almost identical to those of Contriever. Consistently, using $\ell = 64$

(tokens) is optimal. This is in contrast to similar experiments we conducted for BM25 (cf. Figure 6), where $\ell = 32$ is optimal.

# B GPT-Neo Results

Table 5 gives the results of applying In-Context RALM to the models from the GPT-Neo model family on WikiText-103 and RealNews.

# C Open-Domain Question Answering Experiments: Further Details

Closed-Book Setting For the closed-book setting, we adopt the prompt of Touvron et al. (2023):

Answer these questions:

Q: Who got the first noble prize in physics?

A:

Open-Book Setting For the open-book setting, we extend the above prompt as follows:

Nobel Prize

A group including 42 Swedish writers, artists, and literary critics protested against this decision, having expected Leo Tolstoy to be awarded. Some, including Burton Feldman, have criticised this prize because they...

Nobel Prize in Physiology or Medicine

In the last half century there has been an increasing tendency for scientists to work as teams, resulting in controversial exclusions. Alfred Nobel was born on 21 October 1833 in Stockholm, Sweden, into a family of engineers...

Based on these texts, answer these questions:

Q: Who got the first noble prize in physics?

A:

Table 5: The performance of models from the GPT-Neo family, measured by word-level perplexity on the test set of WikiText-103 and token-level perplexity on the development set of RealNews.

![](dt=2026-06-04/ht=15/fb31f38f49206d3b424e3e168ffc317a3ba61aaa512da6d60a4f3a786f9e7a20.jpg)

<table><tr><td>Model</td><td>Retrieval</td><td>Wiki-103 word ppl</td><td>RealNews token ppl</td></tr><tr><td rowspan="2">GPT-Neo 1.3B</td><td>-</td><td>17.5</td><td>12.3</td></tr><tr><td>BM25, §5</td><td>14.6</td><td>9.9</td></tr><tr><td rowspan="2">GPT-Neo 2.7B</td><td>-</td><td>15.1</td><td>11.0</td></tr><tr><td>BM25, §5</td><td>12.8</td><td>9.0</td></tr><tr><td rowspan="2">GPT-J 6B</td><td>-</td><td>11.6</td><td>9.2</td></tr><tr><td>BM25, §5</td><td>10.0</td><td>7.7</td></tr></table>

![](dt=2026-06-04/ht=15/4d464a23066125a4cc6fa428f09d463718404dc7c02a846aaf5975a822d4318c.jpg)

![](dt=2026-06-04/ht=15/70a397136bd8330b12d58d5f7ef9377a9e7ad4bfc5597c70a42c88802644ae62.jpg)