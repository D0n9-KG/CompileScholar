# MTEB: Massive Text Embedding Benchmark

Niklas Muennighoff $^{1}$ , Nouamane Tazi $^{1}$ , Loïc Magne $^{1}$ , Nils Reimers $^{2*}$

$^{1}$ Hugging Face $^{2}$ cohere.ai

$^{1}$ firstname@huggingface.co $^{2}$ info@nils-reimers.de

# Abstract

Text embeddings are commonly evaluated on a small set of datasets from a single task not covering their possible applications to other tasks. It is unclear whether state-of-the-art embeddings on semantic textual similarity (STS) can be equally well applied to other tasks like clustering or reranking. This makes progress in the field difficult to track, as various models are constantly being proposed without proper evaluation. To solve this problem, we introduce the Massive Text Embedding Benchmark (MTEB). MTEB spans 8 embedding tasks covering a total of 58 datasets and 112 languages. Through the benchmarking of 33 models on MTEB, we establish the most comprehensive benchmark of text embeddings to date. We find that no particular text embedding method dominates across all tasks. This suggests that the field has yet to converge on a universal text embedding method and scale it up sufficiently to provide state-of-the-art results on all embedding tasks. MTEB comes with open-source code and a public leaderboard at https://github.com/embeddings-benchmark/mteb.

# 1 Introduction

Natural language embeddings power a variety of use cases from clustering and topic representation (Aggarwal and Zhai, 2012; Angelov, 2020) to search systems and text mining (Huang et al., 2020; Zhu et al., 2021; Nayak, 2019) to feature representations for downstream models (Saharia et al., 2022; Borgeaud et al., 2022). Using generative language models or cross-encoders for these applications is often intractable, as they may require exponentially more computations (Reimers and Gurevych, 2019).

However, the evaluation regime of current text embedding models rarely covers the breadth of their possible use cases. For example, Sim-CSE (Gao et al., 2021b) or SBERT (Reimers and Gurevych, 2019) solely evaluate on STS and classification tasks, leaving open questions about the transferability of the embedding models to search or clustering tasks. STS is known to poorly correlate with other real-world use cases (Neelakantan et al., 2022; Wang et al., 2021). Further, evaluating embedding methods on many tasks requires implementing multiple evaluation pipelines. Implementation details like pre-processing or hyperparameters may influence the results making it unclear whether performance improvements simply come from a favorable evaluation pipeline. This leads to the “blind” application of these models to new use cases in industry or requires incremental work to reevaluate them on different tasks.

The Massive Text Embedding Benchmark (MTEB) aims to provide clarity on how models perform on a variety of embedding tasks and thus serves as the gateway to finding universal text embeddings applicable to a variety of tasks. MTEB consists of 58 datasets covering 112 languages from 8 embedding tasks: Bitext mining, classification, clustering, pair classification, reranking, retrieval, STS and summarization. MTEB software is available open-source $^{1}$ enabling evaluation of any embedding model by adding less than 10 lines of code. Datasets and the MTEB leaderboard are available on the Hugging Face Hub $^{2}$ .

We evaluate over 30 models on MTEB with additional speed and memory benchmarking to provide a holistic view of the state of text embedding models. We cover both models available open-source as well as models accessible via APIs, such as the OpenAI Embeddings endpoint. We find there to be no single best solution, with different models dominating different tasks. Our benchmarking sheds light on the weaknesses and strengths of individual

models, such as SimCSE's (Gao et al., 2021b) low performance on clustering and retrieval despite its strong performance on STS. We hope our work makes selecting the right embedding model easier and simplifies future embedding research.

# 2 Related Work

# 2.1 Benchmarks

Benchmarks, such as (Super)GLUE (Wang et al., 2018, 2019) or Big-BENCH (Srivastava et al., 2022), and evaluation frameworks (Gao et al., 2021a) play a key role in driving NLP progress. Yearly released SemEval datasets (Agirre et al., 2012, 2013, 2014, 2015, 2016) are commonly used as the go-to benchmark for text embeddings. SemEval datasets correspond to the task of semantic textual similarity (STS) requiring models to embed similar sentences with geometrically close embeddings. Due to the limited expressivity of a single SemEval dataset, SentEval (Conneau and Kiela, 2018) aggregates multiple STS datasets. SentEval focuses on fine-tuning classifiers on top of embeddings. It lacks tasks like retrieval or clustering, where embeddings are directly compared without additional classifiers. Further, the toolkit was proposed in 2018 and thus does not provide easy support for recent trends like text embeddings from transformers (Reimers and Gurevych, 2019). Due to the insufficiency of STS benchmarking, USEB (Wang et al., 2021) was introduced consisting mostly of reranking tasks. Consequently, it does not cover tasks like retrieval or classification. Meanwhile, the recently released BEIR Benchmark (Thakur et al., 2021) has become the standard for the evaluation of embeddings for zero-shot information retrieval.

MTEB unifies datasets from different embedding tasks into a common, accessible evaluation framework. MTEB incorporates SemEval datasets (STS11 - STS22) and BEIR alongside a variety of other datasets from various tasks to provide a holistic performance review of text embedding models.

# 2.2 Embedding Models

Text embedding models like Glove (Pennington et al., 2014) lack context awareness and are thus commonly labeled as Word Embedding Models. They consist of a layer mapping each input word to a vector often followed by an averaging layer to provide a final embedding invariant of input length. Transformers (Vaswani et al., 2017) inject context awareness into language models via self-attention and form the foundation of most recent embedding models. BERT (Devlin et al., 2018) uses the transformer architecture and performs large-scale self-supervised pre-training. The resulting model can directly be used to produce text embeddings via an averaging operation alike Glove. Building on InferSent (Conneau et al., 2017), SBERT (Reimers and Gurevych, 2019) demonstrated it to be beneficial to perform additional fine-tuning of the transformer for competitive embedding performance. Most recent fine-tuned embedding models use a contrastive loss objective to perform supervised fine-tuning on positive and negative text pairs (Gao et al., 2021b; Wang et al., 2021; Ni et al., 2021b; Muennighoff, 2022). Due to the large variety of available pre-trained transformers (Wolf et al., 2020), there is an at least equally large variety of potential text embedding models to be explored. This leads to confusion about which model provides practitioners with the best performance for their embedding use case.

We benchmark both word embedding and transformer models on MTEB quantifying gains provided by often much slower context aware models.

# 3 The MTEB Benchmark

# 3.1 Desiderata

MTEB is built on a set of desiderata: (a) Diversity: MTEB aims to provide an understanding of the usability of embedding models in various use cases. The benchmark comprises 8 different tasks, with up to 15 datasets each. Of the 58 total datasets in MTEB, 10 are multilingual, covering 112 different languages. Sentence-level and paragraph-level datasets are included to contrast performance on short and long texts. (b) Simplicity: MTEB provides a simple API for plugging in any model that given a list of texts can produce a vector for each list item with a consistent shape. This makes it possible to benchmark a diverse set of models. (c) Extensibility: New datasets for existing tasks can be benchmarked in MTEB via a single file that specifies the task and a Hugging Face dataset name where the data has been uploaded (Lhoest et al., 2021). New tasks require implementing a task interface for loading the data and an evaluator for benchmarking. We welcome dataset, task or metric contributions from the community via pull requests to continue the development of MTEB. (d) Reproducibility: Through versioning at a dataset and software level, we aim to make it easy to repro-

![](images/d1e16fe966ee881854826b59e69845fe7359eb9b01b51904c14cc10d05f39098.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Clustering"] --> B["ArxivP2P"]
    A --> C["MedrxivP2P"]
    A --> D["StackExchange"]
    A --> E["BiorxivP2P"]
    A --> F["BiorxivS2S"]
    A --> G["Reddit"]
    A --> H["StackExchangeP2P"]
    A --> I["TwentyNewsgroup"]
```
</details>

![](images/7008293d7262f074bfe69bd5d5aaf7322c3d699a2d5348d98937dd840bafcb33.jpg)

<details>
<summary>text_image</summary>

MTEB
Massive Text
Embedding Benchmark
8 Tasks
58 Datasets
</details>

![](images/6d7ce79914a88edcfba7e5ebe2a43b669b56d432b7e621305dbd812e0471624d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Classification"] --> B["AmazonCounterfactual"]
    A --> C["AmazonPolarity"]
    A --> D["AmazonReviews"]
    A --> E["Banking77"]
    A --> F["Emotion"]
    A --> G["Imdb"]
    A --> H["MassiveIntent"]
    A --> I["MassiveScenario"]
    A --> J["MTOPDomain"]
    A --> K["MTOPIntent"]
    A --> L["ToxicConversations"]
    A --> M["TweetSentimentExtraction"]
```
</details>

![](images/c345ca758b6f4df563acd0aeaea1ad9cf66c6d183f13e83599da6871ee40e609.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Bitext Mining"] --> B["BUCC"]
    A --> C["Tatoeba"]
```
</details>

![](images/4235efab76f09b36da378fce0f97aa563b40484d40b856e36a175598fcaa220b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["STS"] --> B["BIOSESS"]
    A --> C["SICK-R"]
    A --> D["STS11"]
    A --> E["STS12"]
    A --> F["STS13"]
    A --> G["STS14"]
    A --> H["STS15"]
    A --> I["STS16"]
    A --> J["STS17"]
    A --> K["STS22"]
    A --> L["STSB"]
```
</details>

![](images/1173cb2b4aa2972750e94ab2b4e79fef98ec83521b762ead31cc5df6065bef47.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Pair Classification"] --> B["SprintDuplicateQuestions"]
    A --> C["TwitterSemEval2015"]
    A --> D["TwitterURLCorpus"]
```
</details>

![](images/d2dbf5cc8547d6bbe5443eb0b213c9a0b10a0d597d451145a27474aa5933e6f4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Retrieval"] --> B["ArguAna"]
    A --> C["ClimateFEVER"]
    A --> D["DBPedia"]
    A --> E["CQADupstackRetrieval"]
    A --> F["FEVER"]
    A --> G["FiQA2018"]
    A --> H["HotpotQA"]
    A --> I["MSMARCO"]
    A --> J["NFCorpus"]
    A --> K["NQ"]
    A --> L["Quora"]
    A --> M["SCIDOCS"]
    A --> N["SciFact"]
    A --> O["Touche2020"]
    A --> P["TRECCOVID"]
```
</details>

![](images/95dad0b1a041115c7fb328d3bcf5cc701b8675dd2eea25e67df187208f63d844.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["STS17"] --> B["SummEval"]
    C["STS22"] --> B
    D["STSB"] --> B
    B --> E["SummEval"]
```
</details>

![](images/7ebb9bbd817f26ff48ea7047c3d344dd1b625bb520f068bc5153b0e7cebbf4fc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Reranking"] --> B["AskUbuntuDupQuestions"]
    A --> C["MindSmallReranking"]
    A --> D["SciDocsRR"]
    A --> E["StackOverFlowDupQuestions"]
```
</details>

Figure 1: An overview of tasks and datasets in MTEB. Multilingual datasets are marked with a purple shade.

duce results in MTEB. JSON files corresponding to all results available in this paper have been made available together with the MTEB benchmark $^{4}$ .

# 3.2 Tasks and Evaluation

Figure 1 provides an overview of tasks and datasets available in MTEB. Dataset statistics are available in Table 2. The benchmark consists of the following 8 task types:

Bitext Mining Inputs are two sets of sentences from two different languages. For each sentence in the first set, the best match in the second set needs to be found. The matches are commonly translations. The provided model is used to embed each sentence and the closest pairs are found via cosine similarity. F1 serves as the main metric for bitext mining. Accuracy, precision and recall are also computed.

Classification A train and test set are embedded with the provided model. The train set embeddings are used to train a logistic regression classifier with 100 maximum iterations, which is scored on the test set. The main metric is accuracy with average precision and f1 additionally provided.

Clustering Given a set of sentences or paragraphs, the goal is to group them into meaningful clusters. A mini-batch k-means model with batch size 32 and k equal to the number of different labels (Pedregosa et al., 2011) is trained on the embedded texts. The model is scored using v-measure (Rosenberg and Hirschberg, 2007). V-measure does not depend on the cluster label, thus the permutation of labels does not affect the score.

Pair Classification A pair of text inputs is provided and a label needs to be assigned. Labels are typically binary variables denoting duplicate or paraphrase pairs. The two texts are embedded and their distance is computed with various metrics (cosine similarity, dot product, euclidean distance, manhattan distance). Using the best binary threshold accuracy, average precision, f1, precision and recall are computed. The average precision score based on cosine similarity is the main metric.

Reranking Inputs are a query and a list of relevant and irrelevant reference texts. The aim is to rank the results according to their relevance to the query. The model is used to embed the references which are then compared to the query using cosine similarity. The resulting ranking is scored for each query and averaged across all queries. Metrics are mean MRR@k and MAP with the latter being the main metric.

Retrieval Each dataset consists of a corpus, queries and a mapping for each query to relevant documents from the corpus. The aim is to find these relevant documents. The provided model is used

![](images/07b2661a2f92af66ba523297c1090b266286938056f6af333afe0710ecfad477.jpg)

<details>
<summary>heatmap</summary>

| Topic | Value |
|---|---|
| AmazonCounterfactualClassification | 83 |
| AmazonPolarityClassification | 85 |
| AmazonReviewsClassification | 90 |
| BankingT7Classification | 92 |
| EmotionClassification | 94 |
| ImoDClassification | 96 |
| MassiveIntentClassification | 98 |
| MassiveScenarioClassification | 100 |
| MTOPDomainClassification | 102 |
| MTOPointenClassification | 104 |
| ToxicConversationsClassification | 106 |
| TweetSentimentExtractionClassification | 108 |
| ArxivClusteringP2P | 110 |
| ArxivClusteringS2S | 112 |
| BiorxivClusteringP2P | 114 |
| BiorxivClusteringS2S | 116 |
| MedrixvClusteringP2P | 118 |
| MedrixvClusteringS2S | 120 |
| MTOPointenClassification | 122 |
| RedditClustering | 124 |
| RedditClusteringP2P | 126 |
| StackExchangeClusteringP2P | 128 |
| StackExchangeClusteringP2P | 130 |
| TwentyNewsgroupsClustering | 132 |
| SprintDuplicateQuestions | 134 |
| TwitterSemEval2015 | 136 |
| TwitterURLCorpus | 138 |
| AskUbuntuDupQuestions | 140 |
| MindSmallReranking | 142 |
| SciDocsRR | 144 |
| StackOverflowDupQuestions | 146 |
| ArguAna | 148 |
| ClimateFEVER | 150 |
| COADupstackAndroidRetrieval | 152 |
| COADupstackEnglishRetrieval | 154 |
| COADupstackGamingRetrieval | 156 |
| COADupstackGisRetrieval | 158 |
| COADupstackMathematicaRetrieval | 160 |
| COADupstackPhysicsRetrieval | 162 |
| COADupstackProgrammersRetrieval | 164 |
| COADupstackStatsRetrieval | 166 |
| COADupstackTexRetrieval | 168 |
| COADupstackUnixRetrieval | 170 |
| COADupstackWebmastersRetrieval | 172 |
| COADupstackWordPressRetrieval | 174 |
| COADupstackWordpressRetrieval | 176 |
| DBPedia | 178 |
| FEVER | 180 |
| FiO2A0LB | 182 |
| HotpotOA | 184 |
| MSMARCO | 186 |
| NFCorpus | 188 |
| QuoraRetrieval | 190 |
| SCIDOCs | 192 |
| ScifAct | 194 |
| Touche2020-00 | 196 |
| TRECCOVID | 198 |
| BIOSSSES | 200 |
| SICK-R | 202 |
| STS12 | 204 |
| STS13 | 206 |
| STS14 | 208 |
| STS15 | 210 |
| STS16 | 212 |
| STS17 | 214 |
| STS22 | 216 |
| STSBenchmark | 218 |
| SummEval | 220 |

| Category Description (Value) for each category in the table: The values in the table represent the sum of the values for each category. The values are explicitly labeled on the table above each bar.
</details>

Figure 2: Similarity of MTEB datasets. We use the best model on MTEB STS (ST5-XXL, see Table 1) to embed 100 samples for each dataset. Cosine similarities between the averaged embeddings are computed and visualized.

to embed all queries and all corpus documents and similarity scores are computed using cosine similarity. After ranking the corpus documents for each query based on the scores, nDCG@k, MRR@k, MAP@k, precision@k and recall@k are computed for several values of k. nDCG@10 serves as the main metric. MTEB reuses datasets and evaluation from BEIR (Thakur et al., 2021).

Semantic Textual Similarity (STS) Given a sentence pair the aim is to determine their similarity. Labels are continuous scores with higher numbers indicating more similar sentences. The provided model is used to embed the sentences and their similarity is computed using various distance metrics. Distances are benchmarked with ground truth similarities using Pearson and Spearman correlations. Spearman correlation based on cosine similarity serves as the main metric (Reimers et al., 2016).

Summarization A set of human-written and machine-generated summaries are provided. The aim is to score the machine summaries. The provided model is first used to embed all summaries. For each machine summary embedding, distances to all human summary embeddings are computed. The closest score (e.g. highest cosine similarity) is kept and used as the model's score of a single machine-generated summary. Pearson and Spearman correlations with ground truth human assessments of the machine-generated summaries are computed. Like for STS, Spearman correlation based on cosine similarity serves as the main metric (Reimers et al., 2016).

# 3.3 Datasets

To further the diversity of MTEB, datasets of varying text lengths are included. All datasets are grouped into three categories:

Sentence to sentence (S2S) A sentence is compared with another sentence. An example of S2S are all current STS tasks in MTEB, where the similarity between two sentences is assessed.

Paragraph to paragraph (P2P) A paragraph is compared with another paragraph. MTEB imposes no limit on the input length, leaving it up to the

<table><tr><td>Num. Datasets (→)</td><td>Class. 12</td><td>Clust. 11</td><td>PairClass. 3</td><td>Rerank. 4</td><td>Retr. 15</td><td>STS 10</td><td>Summ. 1</td><td>Avg. 56</td></tr><tr><td colspan="9">Self-supervised methods</td></tr><tr><td>Glove</td><td>57.29</td><td>27.73</td><td>70.92</td><td>43.29</td><td>21.62</td><td>61.85</td><td>28.87</td><td>41.97</td></tr><tr><td>Komninos</td><td>57.65</td><td>26.57</td><td>72.94</td><td>44.75</td><td>21.22</td><td>62.47</td><td>30.49</td><td>42.06</td></tr><tr><td>BERT</td><td>61.66</td><td>30.12</td><td>56.33</td><td>43.44</td><td>10.59</td><td>54.36</td><td>29.82</td><td>38.33</td></tr><tr><td>SimCSE-BERT-unsup</td><td>62.50</td><td>29.04</td><td>70.33</td><td>46.47</td><td>20.29</td><td>74.33</td><td>31.15</td><td>45.45</td></tr><tr><td colspan="9">Supervised methods</td></tr><tr><td>SimCSE-BERT-sup</td><td>67.32</td><td>33.43</td><td>73.68</td><td>47.54</td><td>21.82</td><td>79.12</td><td>23.31</td><td>48.72</td></tr><tr><td>coCondenser-msmarco</td><td>64.71</td><td>37.64</td><td>81.74</td><td>51.84</td><td>32.96</td><td>76.47</td><td>29.50</td><td>52.35</td></tr><tr><td>Contriever</td><td>66.68</td><td>41.10</td><td>82.53</td><td>53.14</td><td>41.88</td><td>76.51</td><td>30.36</td><td>56.00</td></tr><tr><td>SPECTER</td><td>52.37</td><td>34.06</td><td>61.37</td><td>48.10</td><td>15.88</td><td>61.02</td><td>27.66</td><td>40.28</td></tr><tr><td>LaBSE</td><td>62.71</td><td>29.55</td><td>78.87</td><td>48.42</td><td>18.99</td><td>70.80</td><td>31.05</td><td>45.21</td></tr><tr><td>LASER2</td><td>53.65</td><td>15.28</td><td>68.86</td><td>41.44</td><td>7.93</td><td>55.32</td><td>26.80</td><td>33.63</td></tr><tr><td>MiniLM-L6</td><td>63.06</td><td>42.35</td><td>82.37</td><td>58.04</td><td>41.95</td><td>78.90</td><td>30.81</td><td>56.26</td></tr><tr><td>MiniLM-L12</td><td>63.21</td><td>41.81</td><td>82.41</td><td>58.44</td><td>42.69</td><td>79.80</td><td>27.90</td><td>56.53</td></tr><tr><td>MiniLM-L12-multilingual</td><td>64.30</td><td>37.14</td><td>78.45</td><td>53.62</td><td>32.45</td><td>78.92</td><td>30.67</td><td>52.44</td></tr><tr><td>MPNet</td><td>65.07</td><td>43.69</td><td>83.04</td><td>59.36</td><td>43.81</td><td>80.28</td><td>27.49</td><td>57.78</td></tr><tr><td>MPNet-multilingual</td><td>67.91</td><td>38.40</td><td>80.81</td><td>53.80</td><td>35.34</td><td>80.73</td><td>31.57</td><td>54.71</td></tr><tr><td>Ada Similarity</td><td>70.44</td><td>37.52</td><td>76.86</td><td>49.02</td><td></td><td>78.60</td><td>26.94</td><td></td></tr><tr><td>SGPT-125M-nli</td><td>61.46</td><td>30.95</td><td>71.78</td><td>47.56</td><td>20.90</td><td>74.71</td><td>30.26</td><td>45.97</td></tr><tr><td>SGPT-5.8B-nli</td><td>70.14</td><td>36.98</td><td>77.03</td><td>52.33</td><td>32.34</td><td>80.53</td><td>30.38</td><td>53.74</td></tr><tr><td>SGPT-125M-msmarco</td><td>60.72</td><td>35.79</td><td>75.23</td><td>50.58</td><td>37.04</td><td>73.41</td><td>28.90</td><td>51.23</td></tr><tr><td>SGPT-1.3B-msmarco</td><td>66.52</td><td>39.92</td><td>79.58</td><td>54.00</td><td>44.49</td><td>75.74</td><td>25.44</td><td>56.11</td></tr><tr><td>SGPT-2.7B-msmarco</td><td>67.13</td><td>39.83</td><td>80.65</td><td>54.67</td><td>46.54</td><td>76.83</td><td>27.87</td><td>57.12</td></tr><tr><td>SGPT-5.8B-msmarco</td><td>68.13</td><td>40.35</td><td>82.00</td><td>56.56</td><td>50.25</td><td>78.10</td><td>24.75</td><td>58.81</td></tr><tr><td>SGPT-BLOOM-7.1B-msmarco</td><td>66.19</td><td>38.93</td><td>81.90</td><td>55.65</td><td>48.21</td><td>77.74</td><td>24.99</td><td>57.44</td></tr><tr><td>GTR-Base</td><td>65.25</td><td>38.63</td><td>83.85</td><td>54.23</td><td>44.67</td><td>77.07</td><td>29.67</td><td>56.19</td></tr><tr><td>GTR-Large</td><td>67.14</td><td>41.60</td><td>85.33</td><td>55.36</td><td>47.42</td><td>78.19</td><td>29.50</td><td>58.28</td></tr><tr><td>GTR-XL</td><td>67.11</td><td>41.51</td><td>86.13</td><td>55.96</td><td>47.96</td><td>77.80</td><td>30.21</td><td>58.42</td></tr><tr><td>GTR-XXL</td><td>67.41</td><td>42.42</td><td>86.12</td><td>56.65</td><td>48.48</td><td>78.38</td><td>30.64</td><td>58.97</td></tr><tr><td>ST5-Base</td><td>69.81</td><td>40.21</td><td>85.17</td><td>53.09</td><td>33.63</td><td>81.14</td><td>31.39</td><td>55.27</td></tr><tr><td>ST5-Large</td><td>72.31</td><td>41.65</td><td>84.97</td><td>54.00</td><td>36.71</td><td>81.83</td><td>29.64</td><td>57.06</td></tr><tr><td>ST5-XL</td><td>72.84</td><td>42.34</td><td>86.06</td><td>54.71</td><td>38.47</td><td>81.66</td><td>29.91</td><td>57.87</td></tr><tr><td>ST5-XXL</td><td>73.42</td><td>43.71</td><td>85.06</td><td>56.43</td><td>42.24</td><td>82.63</td><td>30.08</td><td>59.51</td></tr></table>

Table 1: Average of the main metric (see Section 3.2) per task per model on MTEB English subsets.

models to truncate if necessary. Several clustering tasks are framed as both S2S and P2P tasks. The former only compare titles, while the latter include both title and content. For ArxivClustering, for example, abstracts are concatenated to the title in the P2P setting.

Sentence to paragraph (S2P) A few retrieval datasets are mixed in a S2P setting. Here a query is a single sentence, while documents are long paragraphs consisting of multiple sentences.

Similarities across 56 MTEB datasets are visualized in Figure 2. Several datasets rely on the same corpora, such as ClimateFEVER and FEVER, resulting in a score of 1. Clusters of similar datasets can be seen among CQADupstack variations and STS datasets. S2S and P2P variations of the same dataset tend to also be similar. Scientific datasets, such as SciDocsRR, SciFact, ArxivClustering, show high similarities among each other even when coming from different tasks (Reranking, Retrieval and Clustering in this case).

# 4 Results

# 4.1 Models

We evaluate on the test splits of all datasets except for MSMARCO, where the dev split is used following Thakur et al. (2021). We benchmark models claiming state-of-the-art results on various embedding tasks leading to a high representation of transformers (Vaswani et al., 2017). We group models into self-supervised and supervised methods.

![](images/da3f3b1d1884f0fdd08c0c34b75af52d6ad85f24134bf1e32798e945d4646ab7.jpg)  
Figure 3: MTEB performance scales with model size. The smallest SGPT variant underperforms similar-sized GTR and ST5 variants. This may be due to the bias-only fine-tuning SGPT employs, which catches up with full fine-tuning only as model size and thus the number of bias parameters increases (Muennighoff, 2022).

Self-supervised methods (a) Transformer-based BERT (Devlin et al., 2018) is trained using self-supervised mask and sentence prediction tasks. By taking the mean across the sequence length (mean-pooling) the model can directly be used to produce text embeddings. SimCSE-Unsup (Gao et al., 2021b) uses BERT as a foundation and performs additional self-supervised training. (b) Non-transformer: Komninos (Komninos and Manandhar, 2016) and Glove (Pennington et al., 2014) are two word embedding models that directly map words to vectors. Hence, their embeddings lack context awareness, but provide significant speed-ups.

Supervised methods The original transformer model (Vaswani et al., 2017) consists of an encoder and decoder network. Subsequent transformers often train only encoders like BERT (Devlin et al., 2018) or decoders like GPT (Radford et al., 2019).

(a) Transformer encoder methods coCondenser (Gao and Callan, 2021), Contriever (Izacard et al., 2021), LaBSE (Feng et al., 2020) and SimCSE-BERT-sup (Gao et al., 2021b) are based on the pre-trained BERT model (Devlin et al., 2018). coCondenser and Contriever add a self-supervised stage prior to supervised fine-tuning for a total of three training stages. LaBSE uses BERT to perform additional pre-training on parallel data to produce a competitive bitext mining model. SPECTER (Cohan et al., 2020a) relies on the pre-trained SciBERT (Beltagy et al., 2019) variant instead and fine-tunes on citation graphs. GTR (Ni et al., 2021b) and ST5 (Ni et al., 2021a) are based on the encoder part of the T5 model (Raffel et al., 2020) and only differ in their fine-tuning datasets. After additional self-supervised training, ST5 does contrastive fine-tuning on NLI (Ni et al., 2021a; Gao et al., 2021b) being geared towards STS tasks. Meanwhile, GTR fine-tunes on MS-MARCO and focuses on retrieval tasks. MPNet and MiniLM correspond to fine-tuned embedding models (Reimers and Gurevych, 2019) of the pretrained MPNet (Song et al., 2020) and MiniLM (Wang et al., 2020) models using diverse datasets to target any embedding use case.

(b) Transformer decoder methods SGPT Bi-Encoders (Muennighoff, 2022) perform contrastive fine-tuning of $< 0.1\%$ of pre-trained parameters using weighted-mean pooling. Similar to ST5 and GTR, SGPT-nli models are geared towards STS, while SGPT-msmarco models towards retrieval. SGPT-msmarco models embed queries and documents for retrieval with different special tokens to help the model distinguish their role. For non-retrieval tasks, we use its query representations. We benchmark publicly available SGPT models based on GPT-NeoX (Andonian et al., 2021), GPT-J (Wang and Komatsuzaki, 2021) and BLOOM (Scao et al., 2022). Alternatively, cpt-text (Neelakantan et al., 2022) passes pre-trained GPT decoders through a two-stage process using last token pooling to provide embeddings from decoders. We benchmark their models via the OpenAI Embeddings API4.   
(c) Non-transformer LASER (Heffernan et al., 2022) is the only context aware non-transformer model we benchmark, relying on an LSTM (Hochreiter and Schmidhuber, 1997) instead. Similar to LaBSE, the model trains on parallel data and focuses on bitext mining applications.

![](images/0fa08dc37ce3727c2d944d4e37ce3cbb9aca183be374ee84063a4b058b2a5408.jpg)

<details>
<summary>bubble</summary>

| Model               | Speed (examples per sec) | MTEB Score |
| ------------------- | ------------------------ | ---------- |
| ST5-XXL             | ~10^2                    | ~59        |
| GTR-XXL             | ~10^2                    | ~59        |
| SGPT-5.8B-msmarco  | ~10^2                    | ~53        |
| SGPT-5.8B-nli      | ~10^2                    | ~53        |
| GTR-Base            | ~10^3                    | ~56        |
| ST5-Base            | ~10^3                    | ~56        |
| MPNet               | ~10^3                    | ~57        |
| MiniLM-L12          | ~10^3                    | ~56        |
| Contriever          | ~10^3                    | ~56        |
| coCondenser-msmarco | ~10^3                    | ~52        |
| SGPT-125M-msmarco  | ~10^3                    | ~51        |
| SimCSE-BERT-sup     | ~10^3                    | ~49        |
| SGPT-125M-nli       | ~10^3                    | ~46        |
| SimCSE-BERT-unsup   | ~10^3                    | ~46        |
| LaBSE               | ~10^3                    | ~46        |
| SimCSE-BERT-sup     | ~10^3                    | ~46        |
| SPECTER             | ~10^3                    | ~40        |
| BERT                | ~10^3                    | ~39        |
| LASER2              | ~10^3                    | ~35        |
| Glove               | ~10^4                    | ~42        |
| Komninos            | ~10^4                    | ~42        |
</details>

Figure 4: Performance, speed, and size of produced embeddings (size of the circles) of different embedding models. Embedding sizes range from 1.2 kB (Glove / Komninos) to 16.4 kB (SGPT-5.8B) per example. Speed was benchmarked on STS15 using 1x Nvidia A100 80GB with CUDA 11.6.

# 4.2 Analysis

Based on the results in Table 1, we observe that there is considerable variability between tasks. No model claims the state-of-the-art in all seven English tasks. There is even more variability in the results per dataset present in the appendix. Further, there remains a large gap between self-supervised and supervised methods. Self-supervised large language models have been able to close this gap in many natural language generation tasks (Chowdhery et al., 2022). However, they appear to still require supervised fine-tuning for competitive embedding performance.

We find that performance strongly correlates with model size, see Figure 3. A majority of MTEB tasks are dominated by multi-billion parameter models. However, these come at a significant cost as we investigate in Section 4.3.

Classification ST5 models dominate the classification task across most datasets, as can be seen in detail in the full results in the appendix. ST5-XXL has the highest average performance, 3% ahead of the best non-ST5 model, Ada Similarity.

Clustering Despite being almost 50x smaller, the MPNet embedding model is on par with the ST5- XXL state-of-the-art on Clustering. This may be due to the large variety of datasets MPNet (and MiniLM) has been fine-tuned on. Clustering requires coherent distances between a large number of embeddings. Models like SimCSE-sup or SGPT-nli, which are only fine-tuned on a single dataset, NLI, may produce incoherent embeddings when encountering topics unseen during fine-tuning. Relatedly, we find that the query embeddings of SGPT-msmarco and the Ada Search endpoint are competitive with SGPT-nli and the Ada Similarity endpoint, respectively. We refer to the public leaderboard $^{5}$ for Ada Search results. This could be due to the MSMARCO dataset being significantly larger than NLI. Thus, while the OpenAI docs recommend using the similarity embeddings for clustering use cases $^{6}$ , the retrieval query embeddings may be the better choice in some cases.

Pair Classification GTR-XL and GTR-XXL have the strongest performance. Pair classification is closest to STS in its framing, yet models rank significantly differently on the two tasks. This

![](images/44b636f98366889b838c829e88dc78fd94b27446da249497be7ed59edba4cb59.jpg)  
(a) Bitext Mining on Tatoeba

![](images/e4707511909e800e6f8c5a0ec8e8a72b7393f80b9241109d8887e2f125c1530d.jpg)

<details>
<summary>line</summary>

| X | Accuracy |
| --- | --- |
| a | 0.85 |
| b | 0.82 |
| c | 0.78 |
| d | 0.75 |
| e | 0.72 |
| f | 0.69 |
| g | 0.66 |
| h | 0.63 |
| i | 0.60 |
| j | 0.57 |
| k | 0.54 |
| l | 0.51 |
| m | 0.48 |
| n | 0.45 |
| o | 0.42 |
| p | 0.39 |
| q | 0.36 |
| r | 0.33 |
| s | 0.30 |
| t | 0.27 |
| u | 0.24 |
| v | 0.21 |
| w | 0.18 |
| x | 0.15 |
| y | 0.12 |
| z | 0.09 |
| w | 0.06 |
| x | 0.03 |
| y | 0.00 |
| z | -0.03 |
</details>

(b) Multilingual Classification

![](images/95355913cc19e1fa07710e29a688a57c5cf38ecd0246f727ad6c79178d104e7e.jpg)

<details>
<summary>line</summary>

|        | Cox. Sim. Speckman Curr. |
| ------ | ------------------------ |
| ko     | 0.7                      |
| fr     | 0.65                     |
| es     | 0.7                      |
| en     | 0.6                      |
| ar     | 0.55                     |
| k      | 0.5                      |
| zh     | 0.45                     |
| nu     | 0.4                      |
| tr     | 0.35                     |
| de     | 0.3                      |
| pl     | 0.25                     |
</details>

(c) Multi- and Crosslingual STS   
![](images/11eaa8bfe4d123ae2f099a1515939ae3c2ef4f05ce7c7c2c40ac0431548573ba.jpg)  
Figure 5: MTEB multilingual performance. Bitext mining is dominated by LaBSE, while classification and STS results are mixed. SGPT-BLOOM-7B1-msmarco tends to perform well on the languages BLOOM has been pretrained on, such as Chinese, French and Portuguese.

highlights the importance of benchmarking on a diverse set of tasks to avoid blindly reusing a model for a different task.

Reranking MPNet and MiniLM models perform strongly on reranking tasks. On SciDocsRR (Cohan et al., 2020a) they perform far better than bigger models, which is likely due to parts of SciDocsRR being included in their training data. Our scale of experiments and that of model pre-training make controlling for data contamination challenging. Thus, we ignore overlap of MTEB datasets with model training datasets in MTEB scores. As long as enough datasets are averaged, we believe these effects to be insignificant.

Retrieval SGPT-5.8B-msmarco is the best embedding model on the BEIR subset in MTEB as well as on the full BEIR benchmark (Thakur et al., 2021; Muennighoff, 2022). The even larger 7.1B SGPT model making use of BLOOM (Scao et al., 2022) performs significantly weaker, which is likely due to the multilinguality of BLOOM. Models geared towards STS (SimCSE, ST5, SGPT-nli) perform badly on retrieval tasks. Retrieval tasks are unique in that there are two distinct types of texts: Queries and documents (“asymmetric”), while other tasks only have a single type of text (“symmetric”). On the QuoraRetrieval dataset, which has been shown to be largely symmetric (Muennighoff, 2022), the playing field is more even with SGPT-5.8B-nli outperforming SGPT-5.8B-msmarco, see Table 11.

STS & Summarization Retrieval models (GTR, SGPT-msmarco) perform badly on STS, while ST5-XXL has the highest performance. This highlights the bifurcation of the field into separate embedding models for retrieval (asymmetric) and similarity (symmetric) use cases (Muennighoff, 2022).

# 4.3 Efficiency

We investigate the latency-performance trade-off of models in Figure 4. The graph allows for significant elimination of model candidates in the model selection process. It brings model selection down to three clusters:

Maximum speed Word Embedding models offer maximum speed with Glove taking the lead on both performance and speed, thus making the choice simple in this case.

Maximum performance If latency is less important than performance, the left-hand side of the graph offers a cluster of highly performant, but slow models. Depending on the task at hand, GTR-XXL, ST5-XXL or SGPT-5.8B may be the right choice, see Section 4.2. SGPT-5.8B comes with the additional caveat of its high-dimensional embeddings requiring more storage.

Speed and performance The fine-tuned MPNet and MiniLM models lead the middle cluster making the choice easy.

# 4.4 Multilinguality

MTEB comes with 10 multilingual datasets across bitext mining, classification and STS tasks. We in-

vestigate performance on these in Figure 5. Tabular results can be found in Tables 12, 13 and 14.

Bitext Mining LaBSE (Feng et al., 2020) performs strongly across a wide array of languages in bitext mining. Meanwhile, LASER2 shows high variance across different languages. While there are additional language-specific LASER2 models available for some of the languages we benchmark, we use the default multilingual LASER2 model for all languages. This is to provide a fair one-to-one comparison of models. In practice, however, the high variance of LASER2's performance may be resolved by mixing its model variants. MP-Net, MiniLM and SGPT-BLOOM-7B1-msmarco perform poorly on languages they have not been pre-trained on, such as German for the latter.

Classification & STS On multilingual classification and STS, the multilingual MPNet provides the overall strongest performance. It outperforms the slightly faster multilingual MiniLM on almost all languages. Both models have been trained on the same languages, thus bringing decision-making down to performance vs speed. SGPT-BLOOM-7B1-msmarco provides state-of-the-art performance on languages like Hindi, Portuguese, Chinese or French, which the model has seen extensively during pre-training. It also performs competitively on languages like Russian or Japanese that unintentionally leaked into its pre-training data (Muennighoff et al., 2022). However, it is not much ahead of the much cheaper MPNet. LASER2 performs consistently worse than other models.

# 5 Conclusion

In this work, we presented the Massive Text Embedding Benchmark (MTEB). Consisting of 8 text embedding tasks with up to 15 datasets each and covering 112 languages, MTEB aims to provide reliable embedding performance estimates. By open-sourcing MTEB alongside a leaderboard, we provide a foundation for further pushing the state-of-the-art of available text embeddings.

To introduce MTEB, we have conducted the most comprehensive benchmarking of text embeddings to date. Through the course of close to 5,000 experiments on over 30 different models, we have set up solid baselines for future research to build on. We found model performance on different tasks to vary strongly with no model claiming state-of-the-art on all tasks. Our studies on scaling behavior, model efficiency and multilinguality revealed various intricacies of models that should ease the decision-making process for future research or industry applications of text embeddings.

We welcome task, dataset or metric contributions to the MTEB codebase $^{7}$ as well as additions to the leaderboard via our automatic submission format $^{8}$ .

# 6 Limitations of MTEB

While MTEB aims to be a diverse benchmark to provide holistic performance reviews, the benchmark has its limitations. We list them here:

1. Long document datasets MTEB covers multiple text lengths (S2S, P2P, S2P), but very long documents are still missing. The longest datasets in MTEB have a few hundred words, and longer text sizes could be relevant for use cases like retrieval.   
2. Task imbalance Tasks in MTEB have a different amount of datasets with summarization consisting of only a single dataset. This means MTEB average scores, which are computed over all datasets, are biased towards tasks with many datasets, notably retrieval, classification and clustering. As MTEB grows, we hope to add more datasets to currently underrepresented tasks like summarization or pair classification.   
3. Multinguality MTEB contains multilingual classification, STS and bitext mining datasets. However, retrieval and clustering are English-only. SGPT-BLOOM-7B1-msmarco is geared towards multilingual retrieval datasets and due to the lack thereof cannot be comprehensively benchmarked in MTEB. Further, MTEB does not contain any code datasets that could be used to benchmark code models (Neelakantan et al., 2022; Allal et al., 2023). It should be easy to extend MTEB with datasets, such as CodeSearchNet (Husain et al., 2019), TyDI QA (Clark et al., 2020), XOR QA (Asai et al., 2020) or MIRACL (Zhang et al., 2022).   
4. Additional modalities Text embeddings are commonly used as input features for downstream models, such as in our classification task. This can involve other modalities, notably image content (Carvalho et al., 2018; Tan and Bansal, 2019; Muennighoff, 2020; Nichol et al., 2021; Saharia

et al., 2022; Weinbach et al., 2022). We have focused solely on natural language applications and leave extensive benchmarking of text embeddings as inputs for other modalities to future work.

# Acknowledgments

This work was granted access to the HPC resources of Institut du développement et des ressources en informatique scientifique (IDRIS) du Centre national de la recherche scientifique (CNRS) under the allocation 2021-A0101012475 made by Grand équipement national de calcul intensif (GENCI). In particular, all the evaluations and data processing ran on the Jean Zay cluster of IDRIS, and we want to thank the IDRIS team for responsive support throughout the project, in particular Rémi Lacroix.

We thank Douwe Kiela, Teven Le Scao and Nandan Thakur for feedback and suggestions.

# References

Charu C Aggarwal and ChengXiang Zhai. 2012. A survey of text clustering algorithms. In Mining text data, pages 77–128. Springer.   
Eneko Agirre, Carmen Banea, Claire Cardie, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Inigo Lopez-Gazpio, Montse Maritxalar, Rada Mihalcea, et al. 2015. Semeval-2015 task 2: Semantic textual similarity, english, spanish and pilot on interpretability. In Proceedings of the 9th international workshop on semantic evaluation (SemEval 2015), pages 252–263.   
Eneko Agirre, Carmen Banea, Claire Cardie, Daniel M Cer, Mona T Diab, Aitor Gonzalez-Agirre, Weiwei Guo, Rada Mihalcea, German Rigau, and Janyce Wiebe. 2014. Semeval-2014 task 10: Multilingual semantic textual similarity. In SemEval@COLING, pages 81–91.   
Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, Aitor Gonzalez Agirre, Rada Mihalcea, German Rigau Claramunt, and Janyce Wiebe. 2016. Semeval-2016 task 1: Semantic textual similarity, monolingual and cross-lingual evaluation. In SemEval-2016. 10th International Workshop on Semantic Evaluation; 2016 Jun 16-17; San Diego, CA. Stroudsburg (PA): ACL; 2016. p. 497-511. ACL (Association for Computational Linguistics).   
Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. 2012. Semeval-2012 task 6: A pilot on semantic textual similarity. In \*SEM 2012: The First Joint Conference on Lexical and Computational Semantics–Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pages 385–393.

Eneko Agirre, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, and Weiwei Guo. 2013. \* sem 2013 shared task: Semantic textual similarity. In Second joint conference on lexical and computational semantics (\* SEM), volume 1: proceedings of the Main conference and the shared task: semantic textual similarity, pages 32–43.   
Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. 2023. Santacoder: don't reach for the stars! arXiv preprint arXiv:2301.03988.   
Alex Andonian, Quentin Anthony, Stella Biderman, Sid Black, Preetham Gali, Leo Gao, Eric Hallahan, Josh Levy-Kramer, Connor Leahy, Lucas Nestler, Kip Parker, Michael Pieler, Shivanshu Purohit, Tri Songz, Phil Wang, and Samuel Weinbach. 2021. GPT-NeoX: Large scale autoregressive language modeling in pytorch.   
Dimo Angelov. 2020. Top2vec: Distributed representations of topics. arXiv preprint arXiv:2008.09470.   
Akari Asai, Jungo Kasai, Jonathan H Clark, Kenton Lee, Eunsol Choi, and Hannaneh Hajishirzi. 2020. Xor qa: Cross-lingual open-retrieval question answering. arXiv preprint arXiv:2010.11856.   
Iz Beltagy, Kyle Lo, and Arman Cohan. 2019. Scibert: A pretrained language model for scientific text. arXiv preprint arXiv:1903.10676.   
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022. Improving language models by retrieving from trillions of tokens. In International Conference on Machine Learning, pages 2206–2240. PMLR.   
Micael Carvalho, Rémi Cadène, David Picard, Laure Soulier, Nicolas Thome, and Matthieu Cord. 2018. Cross-modal retrieval in the cooking context: Learning semantic text-image embeddings. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pages 35–44.   
Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić. 2020. Efficient intent detection with dual sentence encoders.   
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.   
Jonathan H Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020. Tydi qa: A benchmark for information-seeking question answering in typologically diverse languages. Transactions of the Association for Computational Linguistics, 8:454–470.

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S Weld. 2020a. Specter: Document-level representation learning using citation-informed transformers. arXiv preprint arXiv:2004.07180.   
Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S. Weld. 2020b. Specter: Document-level representation learning using citation-informed transformers.   
Alexis Conneau and Douwe Kiela. 2018. Senteval: An evaluation toolkit for universal sentence representations. arXiv preprint arXiv:1803.05449.   
Alexis Conneau, Douwe Kiela, Holger Schwenk, Loic Barrault, and Antoine Bordes. 2017. Supervised learning of universal sentence representations from natural language inference data. arXiv preprint arXiv:1705.02364.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.   
Alexander R. Fabbri, Wojciech Kryściński, Bryan McCann, Caiming Xiong, Richard Socher, and Dragomir Radev. 2020. Summeval: Re-evaluating summarization evaluation.   
Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2020. Language-agnostic bert sentence embedding. arXiv preprint arXiv:2007.01852.   
Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. 2022. Massive: A 1m-example multilingual natural language understanding dataset with 51 typologically-diverse languages.   
Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, et al. 2021a. A framework for few-shot language model evaluation. Version v0. 0.1. Sept.   
Luyu Gao and Jamie Callan. 2021. Unsupervised corpus aware language model pre-training for dense passage retrieval. arXiv preprint arXiv:2108.05540.   
Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021b. Simcse: Simple contrastive learning of sentence embeddings. arXiv preprint arXiv:2104.08821.   
Gregor Geigle, Nils Reimers, Andreas Rücklé, and Iryna Gurevych. 2021. Tweac: Transformer with extendable qa agent classifiers.   
Kevin Heffernan, Onur Çelebi, and Holger Schwenk. 2022. Bitext mining using distilled sentence representations for low-resource languages. arXiv preprint arXiv:2205.12654.

Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation, 9(8):1735–1780.   
Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. 2020. Embedding-based retrieval in facebook search. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2553–2561.   
Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. Code-searchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436.   
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Towards unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118.   
Alexandros Komninos and Suresh Manandhar. 2016. Dependency based embeddings for sentence classification tasks. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies, pages 1490–1500.   
Wuwei Lan, Siyu Qiu, Hua He, and Wei Xu. 2017. A continuously growing dataset of sentential paraphrases. In Proceedings of The 2017 Conference on Empirical Methods on Natural Language Processing (EMNLP), pages 1235–1245. Association for Computational Linguistics.   
Quentin Lhoest, Albert Villanova del Moral, Yacine Jernite, Abhishek Thakur, Patrick von Platen, Suraj Patil, Julien Chaumond, Mariama Drame, Julien Plu, Lewis Tunstall, et al. 2021. Datasets: A community library for natural language processing. arXiv preprint arXiv:2109.02846.   
Haoran Li, Abhinav Arora, Shuohui Chen, Anchit Gupta, Sonal Gupta, and Yashar Mehdad. 2020. Mtop: A comprehensive multilingual task-oriented semantic parsing benchmark.   
Xueqing Liu, Chi Wang, Yue Leng, and Cheng Xiang Zhai. 2018. Linkso: a dataset for learning to retrieve similar question answer pairs on software development forums. In Proceedings of the 4th ACM SIGSOFT International Workshop on NLP for Software Engineering, pages 2–5.   
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA. Association for Computational Linguistics.

Julian McAuley and Jure Leskovec. 2013. Hidden factors and hidden topics: Understanding rating dimensions with review text. RecSys '13, New York, NY, USA. Association for Computing Machinery.   
Niklas Muennighoff. 2020. Vilio: State-of-the-art visio-linguistic models applied to hateful memes. arXiv preprint arXiv:2012.07788.   
Niklas Muennighoff. 2022. Sgpt: Gpt sentence embeddings for semantic search. arXiv preprint arXiv:2202.08904.   
Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. 2022. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786.   
Pandu Nayak. 2019. Understanding searches better than ever before.   
Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. 2022. Text and code embeddings by contrastive pretraining. arXiv preprint arXiv:2201.10005.   
Jianmo Ni, Gustavo Hernández Ábrego, Noah Constant, Ji Ma, Keith B Hall, Daniel Cer, and Yinfei Yang. 2021a. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. arXiv preprint arXiv:2108.08877.   
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego, Ji Ma, Vincent Y Zhao, Yi Luan, Keith B Hall, Ming-Wei Chang, et al. 2021b. Large dual encoders are generalizable retrievers. arXiv preprint arXiv:2112.07899.   
Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2021. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741.   
James O'Neill, Polina Rozenshtein, Ryuichi Kiryo, Motoko Kubota, and Danushka Bollegala. 2021. I wish i would have loved this one, but i didn't – a multilingual dataset for counterfactual detection in product reviews.   
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.   
Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 1532–1543.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.   
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140):1–67.   
Nils Reimers, Philip Beyer, and Iryna Gurevych. 2016. Task-oriented intrinsic evaluation of semantic textual similarity. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 87–96.   
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.   
Facebook Research. Tatoeba multilingual test set.   
Andrew Rosenberg and Julia Hirschberg. 2007. V-measure: A conditional entropy-based external cluster evaluation measure. pages 410–420.   
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487.   
Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. 2018. CARER: Contextualized affect representations for emotion recognition. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3687–3697, Brussels, Belgium. Association for Computational Linguistics.   
Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. 2022. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100.   
Darsh Shah, Tao Lei, Alessandro Moschitti, Salvatore Romeo, and Preslav Nakov. 2018. Adversarial domain adaptation for duplicate question detection. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1056–1063, Brussels, Belgium. Association for Computational Linguistics.   
Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020. Mpnet: Masked and permuted pretraining for language understanding. Advances in Neural Information Processing Systems, 33:16857–16867.   
Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta,

Adrià Garriga-Alonso, et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615.   
Hao Tan and Mohit Bansal. 2019. Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490.   
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.   
Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019. Superglue: A stickier benchmark for general-purpose language understanding systems. Advances in neural information processing systems, 32.   
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461.   
Ben Wang and Aran Komatsuzaki. 2021. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax.   
Kexin Wang, Nils Reimers, and Iryna Gurevych. 2021. Tsdae: Using transformer-based sequential denoising auto-encoder for unsupervised sentence embedding learning. arXiv preprint arXiv:2104.06979.   
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in Neural Information Processing Systems, 33:5776–5788.   
Samuel Weinbach, Marco Bellagente, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Björn Deiseroth, Koen Oostermeijer, Hannah Teufel, and Andres Felipe Cruz-Salinas. 2022. M-vader: A model for diffusion with multimodal context. arXiv preprint arXiv:2212.02936.   
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations, pages 38–45.

Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, et al. 2020. Mind: A large-scale dataset for news recommendation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3597–3606.   
Wei Xu, Chris Callison-Burch, and William B Dolan. 2015. Semeval-2015 task 1: Paraphrase and semantic similarity in twitter (pit). In Proceedings of the 9th international workshop on semantic evaluation (SemEval 2015), pages 1–11.   
Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. 2022. Making a miracl: Multilingual information retrieval across a continuum of languages. arXiv preprint arXiv:2210.09984.   
Jeffrey Zhu, Mingqin Li, Jason Li, and Cassandra Oduola. 2021. Bing delivers more contextualized search using quantized transformer inference on nvidia gpus in azure.   
Pierre Zweigenbaum, Serge Sharoff, and Reinhard Rapp. 2016. Towards preparation of the second bucc shared task: Detecting parallel sentences in comparable corpora. In Proceedings of the Ninth Workshop on Building and Using Comparable Corpora. European Language Resources Association (ELRA), Portoroz, Slovenia, pages 38–43.   
Pierre Zweigenbaum, Serge Sharoff, and Reinhard Rapp. 2017. Overview of the second bucc shared task: Spotting parallel sentences in comparable corpora. In Proceedings of the 10th Workshop on Building and Using Comparable Corpora, pages 60–67.   
Pierre Zweigenbaum, Serge Sharoff, and Reinhard Rapp. 2018. Overview of the third bucc shared task: Spotting parallel sentences in comparable corpora. In Proceedings of 11th workshop on building and using comparable corpora, pages 39–42.

# A Datasets

Table 2 provides a summary along with statistics of all MTEB tasks. In the following, we give a brief description of each dataset included in MTEB.

# A.1 Clustering

ArxivClusteringS2S, ArxivClusteringP2P, BiorxivClusteringS2S, BiorxivClusteringP2P, MedrxivClusteringP2P, MedrxivClusteringS2S These datasets are custom-made for MTEB using the public APIs from arXiv $^{9}$ and bioRxiv/medRxiv $^{10}$ . For S2S datasets, the input text is simply the title of the paper, while for P2P the input text is the concatenation of the title and the abstract. The cluster labels are generated using categories given to the papers by humans. For bioRxiv and medRxiv this category is unique, but for arXiv multiple categories can be given to a single paper so we only use the first one. For bioRxiv and medRxiv there is only one level of category (e.g. biochemistry, genetics, microbiology, etc.) hence we only perform clustering based on that label. For arXiv there is a main category and secondary category: for example "cs.AI" means the main category is Computer Science and the sub-category is AI, math.AG means the main category is Mathematics and the sub-category is Algebraic Geometry etc. Hence, we create three types of splits:

(a) Main category clustering Articles are only clustered based on the main category (Math, Physics, Computer Science etc.). This split evaluates coarse clustering capacity of a model.   
(b) Secondary category clustering within the same main category Articles are clustered based on their secondary category, but within a given main category, for example only Math papers that need to be clustered into Algebraic Geometry, Functional Analysis, Numerical Analysis etc. This split evaluates fine-grained clustering capacity of a model, as differentiating some sub-categories can be very difficult.   
(c) Secondary category clustering Articles are clustered based on their secondary category for all main categories, so the labels can be Number Theory, Computational Complexity, Astrophysics of Galaxies etc. These splits evaluate fine-grained

clustering capacity, as well as multi-scale capacities i.e. is a model able to both separate Maths from Physics as well as Probability from Algebraic Topology at the same time.

For every dataset, split and strategy, we select subsets of all labels and then sample articles from those labels. This yields splits with a varying amount and size of clusters.

RedditClustering (Geigle et al., 2021): Clustering of titles from 199 subreddits. Clustering of 25 splits, each with 10-50 classes, and each class with 100 - 1000 sentences

RedditClusteringP2P Dataset created for MTEB using available data from Reddit posts $^{11}$ . The task consists of clustering the concatenation of title+post according to their subreddit. It contains 10 splits, with 10 and 100 clusters per split and 1,000 to 100,000 posts.

StackExchangeClustering (Geigle et al., 2021) Clustering of titles from 121 stackexchanges. Clustering of 25 splits, each with 10-50 classes, and each class with 100-1000 sentences.

StackExchangeClusteringP2P Dataset created for MTEB using available data from StackExchange posts $^{12}$ . The task consists of clustering the concatenation of title and post according to their subreddit. It contains 10 splits, with 10 to 100 clusters and 5,000 to 10,000 posts per split.

TwentyNewsgroupsClustering $^{13}$ Clustering of the 20 Newsgroups dataset, given titles of article the goal is to find the newsgroup (20 in total). Contains 10 splits, each with 20 classes, with each split containing between 1,000 and 10,000 titles.

# A.2 Classification

AmazonCounterfactual (O'Neill et al., 2021) A collection of Amazon customer reviews annotated for counterfactual detection pair classification. For each review the label is either "counterfactual" or "not-counterfactual". This is a multilingual dataset with 4 available languages.

<table><tr><td>Name</td><td>Type</td><td>Categ.</td><td>#Lang.</td><td>Train Samples</td><td>Dev Samples</td><td>Test Samples</td><td>Train avg. chars</td><td>Dev avg. chars</td><td>Test avg. chars</td></tr><tr><td>BUCC</td><td>BitextMining</td><td>s2s</td><td>4</td><td>0</td><td>0</td><td>641684</td><td>0</td><td>0</td><td>101.3</td></tr><tr><td>Tatoeba</td><td>BitextMining</td><td>s2s</td><td>112</td><td>0</td><td>0</td><td>2000</td><td>0</td><td>0</td><td>39.4</td></tr><tr><td>AmazonCounterfactualClassification</td><td>Classification</td><td>s2s</td><td>4</td><td>4018</td><td>335</td><td>670</td><td>107.3</td><td>109.2</td><td>106.1</td></tr><tr><td>AmazonPolarityClassification</td><td>Classification</td><td>p2p</td><td>1</td><td>3600000</td><td>0</td><td>400000</td><td>431.6</td><td>0</td><td>431.4</td></tr><tr><td>AmazonReviewsClassification</td><td>Classification</td><td>s2s</td><td>6</td><td>1200000</td><td>30000</td><td>30000</td><td>160.5</td><td>159.2</td><td>160.4</td></tr><tr><td>Banking77Classification</td><td>Classification</td><td>s2s</td><td>1</td><td>10003</td><td>0</td><td>3080</td><td>59.5</td><td>0</td><td>54.2</td></tr><tr><td>EmotionClassification</td><td>Classification</td><td>s2s</td><td>1</td><td>16000</td><td>2000</td><td>2000</td><td>96.8</td><td>95.3</td><td>96.6</td></tr><tr><td>ImdbClassification</td><td>Classification</td><td>p2p</td><td>1</td><td>25000</td><td>0</td><td>25000</td><td>1325.1</td><td>0</td><td>1293.8</td></tr><tr><td>MassiveIntentClassification</td><td>Classification</td><td>s2s</td><td>51</td><td>11514</td><td>2033</td><td>2974</td><td>35.0</td><td>34.8</td><td>34.6</td></tr><tr><td>MassiveScenarioClassification</td><td>Classification</td><td>s2s</td><td>51</td><td>11514</td><td>2033</td><td>2974</td><td>35.0</td><td>34.8</td><td>34.6</td></tr><tr><td>MTOPDomainClassification</td><td>Classification</td><td>s2s</td><td>6</td><td>15667</td><td>2235</td><td>4386</td><td>36.6</td><td>36.5</td><td>36.8</td></tr><tr><td>MTOPIntentClassification</td><td>Classification</td><td>s2s</td><td>6</td><td>15667</td><td>2235</td><td>4386</td><td>36.6</td><td>36.5</td><td>36.8</td></tr><tr><td>ToxicConversationsClassification</td><td>Classification</td><td>s2s</td><td>1</td><td>50000</td><td>0</td><td>50000</td><td>298.8</td><td>0</td><td>296.6</td></tr><tr><td>TweetSentimentExtractionClassification</td><td>Classification</td><td>s2s</td><td>1</td><td>27481</td><td>0</td><td>3534</td><td>68.3</td><td>0</td><td>67.8</td></tr><tr><td>ArxivClusteringP2P</td><td>Clustering</td><td>p2p</td><td>1</td><td>0</td><td>0</td><td>732723</td><td>0</td><td>0</td><td>1009.9</td></tr><tr><td>ArxivClusteringS2S</td><td>Clustering</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>732723</td><td>0</td><td>0</td><td>74.0</td></tr><tr><td>BiorxivClusteringP2P</td><td>Clustering</td><td>p2p</td><td>1</td><td>0</td><td>0</td><td>75000</td><td>0</td><td>0</td><td>1666.2</td></tr><tr><td>BiorxivClusteringS2S</td><td>Clustering</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>75000</td><td>0</td><td>0</td><td>101.6</td></tr><tr><td>MedrxivClusteringP2P</td><td>Clustering</td><td>p2p</td><td>1</td><td>0</td><td>0</td><td>37500</td><td>0</td><td>0</td><td>1981.2</td></tr><tr><td>MedrxivClusteringS2S</td><td>Clustering</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>37500</td><td>0</td><td>0</td><td>114.7</td></tr><tr><td>RedditClustering</td><td>Clustering</td><td>s2s</td><td>1</td><td>0</td><td>420464</td><td>420464</td><td>0</td><td>64.7</td><td>64.7</td></tr><tr><td>RedditClusteringP2P</td><td>Clustering</td><td>p2p</td><td>1</td><td>0</td><td>0</td><td>459399</td><td>0</td><td>0</td><td>727.7</td></tr><tr><td>StackExchangeClustering</td><td>Clustering</td><td>s2s</td><td>1</td><td>0</td><td>417060</td><td>373850</td><td>0</td><td>56.8</td><td>57.0</td></tr><tr><td>StackExchangeClusteringP2P</td><td>Clustering</td><td>p2p</td><td>1</td><td>0</td><td>0</td><td>75000</td><td>0</td><td>0</td><td>1090.7</td></tr><tr><td>TwentyNewsgroupsClustering</td><td>Clustering</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>59545</td><td>0</td><td>0</td><td>32.0</td></tr><tr><td>SprintDuplicateQuestions</td><td>PairClassification</td><td>s2s</td><td>1</td><td>0</td><td>101000</td><td>101000</td><td>0</td><td>65.2</td><td>67.9</td></tr><tr><td>TwitterSemEval2015</td><td>PairClassification</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>16777</td><td>0</td><td>0</td><td>38.3</td></tr><tr><td>TwitterURLCorpus</td><td>PairClassification</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>51534</td><td>0</td><td>0</td><td>79.5</td></tr><tr><td>AskUbuntuDupQuestions</td><td>Reranking</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>2255</td><td>0</td><td>0</td><td>52.5</td></tr><tr><td>MindSmallReranking</td><td>Reranking</td><td>s2s</td><td>1</td><td>231530</td><td>0</td><td>107968</td><td>69.0</td><td>0</td><td>70.9</td></tr><tr><td>SciDocsRR</td><td>Reranking</td><td>s2s</td><td>1</td><td>0</td><td>19594</td><td>19599</td><td>0</td><td>69.4</td><td>69.0</td></tr><tr><td>StackOverflowDupQuestions</td><td>Reranking</td><td>s2s</td><td>1</td><td>23018</td><td>3467</td><td>3467</td><td>49.6</td><td>49.8</td><td>49.8</td></tr><tr><td>ArguAna</td><td>Retrieval</td><td>p2p</td><td>1</td><td>0</td><td>0</td><td>10080</td><td>0</td><td>0</td><td>1052.9</td></tr><tr><td>ClimateFEVER</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>5418128</td><td>0</td><td>0</td><td>539.1</td></tr><tr><td>CQADupstackAndroidRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>23697</td><td>0</td><td>0</td><td>578.7</td></tr><tr><td>CQADupstackEnglishRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>41791</td><td>0</td><td>0</td><td>467.1</td></tr><tr><td>CQADupstackGamingRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>46896</td><td>0</td><td>0</td><td>474.7</td></tr><tr><td>CQADupstackGisRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>38522</td><td>0</td><td>0</td><td>991.1</td></tr><tr><td>CQADupstackMathematicaRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>17509</td><td>0</td><td>0</td><td>1103.7</td></tr><tr><td>CQADupstackPhysicsRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>39355</td><td>0</td><td>0</td><td>799.4</td></tr><tr><td>CQADupstackProgrammersRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>33052</td><td>0</td><td>0</td><td>1030.2</td></tr><tr><td>CQADupstackStatsRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>42921</td><td>0</td><td>0</td><td>1041.0</td></tr><tr><td>CQADupstackTexRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>71090</td><td>0</td><td>0</td><td>1246.9</td></tr><tr><td>CQADupstackUnixRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>48454</td><td>0</td><td>0</td><td>984.7</td></tr><tr><td>CQADupstackWebmastersRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>17911</td><td>0</td><td>0</td><td>689.8</td></tr><tr><td>CQADupstackWordpressRetrieval</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>49146</td><td>0</td><td>0</td><td>1111.9</td></tr><tr><td>DBPedia</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>4635989</td><td>4636322</td><td>0</td><td>310.2</td><td>310.1</td></tr><tr><td>FEVER</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>5423234</td><td>0</td><td>0</td><td>538.6</td></tr><tr><td>FiQA2018</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>58286</td><td>0</td><td>0</td><td>760.4</td></tr><tr><td>HotpotQA</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>5240734</td><td>0</td><td>0</td><td>288.6</td></tr><tr><td>MSMARCO</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>8848803</td><td>8841866</td><td>0</td><td>336.6</td><td>336.8</td></tr><tr><td>MSMARCOv2</td><td>Retrieval</td><td>s2p</td><td>1</td><td>138641342</td><td>138368101</td><td>0</td><td>341.4</td><td>342.0</td><td>0</td></tr><tr><td>NFCorpus</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>3956</td><td>0</td><td>0</td><td>1462.7</td></tr><tr><td>NQ</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>2684920</td><td>0</td><td>0</td><td>492.7</td></tr><tr><td>QuoraRetrieval</td><td>Retrieval</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>532931</td><td>0</td><td>0</td><td>62.9</td></tr><tr><td>SCIDOCS</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>26657</td><td>0</td><td>0</td><td>1161.9</td></tr><tr><td>SciFact</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>5483</td><td>0</td><td>0</td><td>1422.3</td></tr><tr><td>Touche2020</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>382594</td><td>0</td><td>0</td><td>1720.1</td></tr><tr><td>TRECCOVID</td><td>Retrieval</td><td>s2p</td><td>1</td><td>0</td><td>0</td><td>171382</td><td>0</td><td>0</td><td>1117.4</td></tr><tr><td>BIOSSES</td><td>STS</td><td>s2s</td><td>1</td><td>200</td><td>200</td><td>200</td><td>156.6</td><td>156.6</td><td>156.6</td></tr><tr><td>SICK-R</td><td>STS</td><td>s2s</td><td>1</td><td>19854</td><td>19854</td><td>19854</td><td>46.1</td><td>46.1</td><td>46.1</td></tr><tr><td>STS12</td><td>STS</td><td>s2s</td><td>1</td><td>4468</td><td>0</td><td>6216</td><td>100.7</td><td>0</td><td>64.7</td></tr><tr><td>STS13</td><td>STS</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>3000</td><td>0</td><td>0</td><td>54.0</td></tr><tr><td>STS14</td><td>STS</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>7500</td><td>0</td><td>0</td><td>54.3</td></tr><tr><td>STS15</td><td>STS</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>6000</td><td>0</td><td>0</td><td>57.7</td></tr><tr><td>STS16</td><td>STS</td><td>s2s</td><td>1</td><td>0</td><td>0</td><td>2372</td><td>0</td><td>0</td><td>65.3</td></tr><tr><td>STS17</td><td>STS</td><td>s2s</td><td>11</td><td>0</td><td>0</td><td>500</td><td>0</td><td>0</td><td>43.3</td></tr><tr><td>STS22</td><td>STS</td><td>p2p</td><td>18</td><td>0</td><td>0</td><td>8060</td><td>0</td><td>0</td><td>1992.8</td></tr><tr><td>STSBenchmark</td><td>STS</td><td>s2s</td><td>1</td><td>11498</td><td>3000</td><td>2758</td><td>57.6</td><td>64.0</td><td>53.6</td></tr><tr><td>SummEval</td><td>Summarization</td><td>p2p</td><td>1</td><td>0</td><td>0</td><td>2800</td><td>0</td><td>0</td><td>359.8</td></tr></table>

Table 2: Tasks in MTEB

AmazonPolarity (McAuley and Leskovec, 2013) A collection of Amazon customer reviews annotated for polarity classification. For each review the label is either "positive" or "negative".

AmazonReviews (McAuley and Leskovec, 2013) A collection of Amazon reviews designed to aid research in multilingual text classification. For each review the label is the score given by the review between 0 and 4 (1-5 stars). This is a

multilingual dataset with 6 available languages.

Banking77 (Casanueva et al., 2020) Dataset composed of online banking queries annotated with their corresponding intents. For each user query the label is an intent among 77 intents like 'activate\_my\_card', 'apple\_pay', 'bank\_transfer', etc.

Emotion (Saravia et al., 2018) Dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.

Imdb (Maas et al., 2011) Large movie review dataset with labels being positive or negative.

MassiveIntent (FitzGerald et al., 2022) A collection of Amazon Alexa virtual assistant utterances annotated with the associated intent. For each user utterance the label is one of 60 intents like 'play\_music', 'alarm\_set', etc. This is a multilingual dataset with 51 available languages.

MassiveScenario (FitzGerald et al., 2022) A collection of Amazon Alexa virtual assistant utterances annotated with the associated intent. For each user utterance the label is a theme among 60 scenarios like 'music', 'weather', etc. This is a multilingual dataset with 51 available languages.

MTOPDomain / MTOPIntent Multilingual sentence datasets from the MTOP (Li et al., 2020) benchmark. We refer to their paper for details.

ToxicConversations Dataset from Kaggle competition $^{14}$ . Collection of comments from the Civil Comments platform together with annotations if the comment is toxic or not.

TweetSentimentExtraction Dataset from Kaggle competition $^{15}$ . Sentiment classification of tweets as neutral, positive or negative.

# A.3 Pair Classification

SprintDuplicateQuestions (Shah et al., 2018): Collection of questions from the Sprint community. The goal is to classify a pair of sentences as duplicates or not.

TwitterSemEval2015 (Xu et al., 2015) Paraphrase-Pairs of Tweets from the SemEval 2015 workshop. The goal is to classify a pair of tweets as paraphrases or not.

TwitterURLCorpus (Lan et al., 2017) Paraphrase-Pairs of Tweets. The goal is to classify a pair of tweets as paraphrases or not.

# A.4 Bitext Mining

BUCC (Zweigenbaum et al., 2016, 2017, 2018) BUCC provides big set of sentences ( $\sim$ 10-70k each) for English, French, Russian, German and Chinese, along with associated pairs annotation. The annotated pairs here corresponds to a pairs of translated sentences, i.e. a sentence and its translation in the other language.

Tatoeba (Research) Tatoeba provides sets of sentences (1000 sentences each) for 112 languages with annotated associated pairs. Each pair is one sentence and its translation in another language.

# A.5 Reranking

AskUbuntuDupQuestions $^{16}$ Questions from AskUbuntu with manual annotations marking pairs of questions as similar or dissimilar.

MindSmall (Wu et al., 2020) Large-scale English Dataset for News Recommendation Research. Ranking news article titles given the title of a news article. The idea is to recommend other news from the one you are reading.

SciDocsRR (Cohan et al., 2020b) Ranking of related scientific papers based on their title.

StackOverflowDupQuestions (Liu et al., 2018) Stack Overflow Duplicate Questions Task for questions with the tags Java, JavaScript and Python, ranking questions as duplicates or not.

# A.6 Semantic Textual Similarity (STS)

STS12, STS13, STS14, STS15, STS16, STS17, STS22, STSBenchmark (Agirre et al., 2012, 2013) $^{17181920}$ Original STS benchmark, with scores from 0 to 5. The selection of sentences includes text from image captions, news headlines and user forums. In total they contain between 1,000 and 20,000 sentences. STS12 - STS16 and

STSBenchmark are monolingual english benchmarks. STS17 and STS22 contain crosslingual pairs of sentences, where the goal is to assess the similarity of two sentences in different languages. STS17 has 11 language pairs (among Korean, Arabic, English, French, German, Turkish, Spanish, Italian and Dutch) and STS22 has 18 language pairs (among Arabic, English, French, German, Turkish, Spanish, Polish, Italian, Russian and Chinese).

BIOSSES $^{21}$ Contains 100 sentence pairs from the biomedical field.

SICK-R (Agirre et al., 2014) Sentences Involving Compositional Knowledge (SICK) contains a large number of sentence pairs (10 0000) that are lexically, syntactically and semantically rich.

# A.7 Summarization

SummEval (Fabbri et al., 2020) Summaries generated by recent summarization models trained on CNN or DailyMail alongside human annotations.

# A.8 Retrieval

We refer to the BEIR paper (Thakur et al., 2021), which contains description of each dataset. For MTEB, we include all publicly available datasets: ArguAna, ClimateFEVER, CQADupstack, DB-Pedia, FEVER, FiQA2018, HotpotQA, MSMARCO, NFCorpus, NQ, Quora, SCIDOCS, SciFact, Touche2020, TRECCOVID.

# B Examples

Tables 3-9 provide examples for each dataset for each task. For retrieval datasets, we refer to the BEIR paper (Thakur et al., 2021).

# C Correlations

Figure 6 provides correlation heatmaps for model performance and MTEB tasks.

# D Models

Table 10 provides publicly available model checkpoints used for MTEB evaluation.

# E Additional results

Tables 11 until the end provide results on individual datasets of MTEB. The results are additionally

available in json format on the Hugging Face Hub $^{22}$ and can be inspected on the leaderboard $^{23}$ .

<table><tr><td>Dataset</td><td>Text</td><td>Label</td></tr><tr><td>AmazonCounterfactualClassification</td><td>In person it looks as though it would have cost a lot more.</td><td>counterfactual</td></tr><tr><td>AmazonPolarityClassification</td><td>an absolute masterpiece I am quite sure any of you actually taking the time to read this have played the game at least once, and heard at least a few of the tracks here. And whether you were aware of it or not, Mitsuda&#x27;s music contributed greatly to the...</td><td>positive</td></tr><tr><td>AmazonReviewsClassification</td><td>solo llega una unidad cuando te obligan a comprar dos Te obligan a comprar dos unidades y te llega solo una y no hay forma de reclamar, una autentica estafa, no compreis!!</td><td>0</td></tr><tr><td>Banking77Classification</td><td>What currencies is an exchange rate calculated in?</td><td>exchange_rate</td></tr><tr><td>EmotionClassification</td><td>i feel so inhibited in someone elses kitchen like im painting on someone elses picture</td><td>sadness</td></tr><tr><td>ImdbClassification</td><td>When I first saw a glimpse of this movie, I quickly noticed the actress who was playing the role of Lucille Ball. Rachel York&#x27;s portrayal of Lucy is absolutely awful. Lucille Ball was an astounding comedian with incredible talent. To think about a legend like Lucille Ball being portrayed the way she was in the movie is horrendous. I cannot believe...</td><td>negative</td></tr><tr><td>MassiveIntentClassification</td><td>réveille-moi à neuf heures du matin le vendredi</td><td>alarm_set</td></tr><tr><td>MassiveScenarioClassification</td><td>tell me the artist of this song</td><td>music</td></tr><tr><td>MTOPDomainClassification</td><td>Maricopa County weather forecast for this week</td><td>weather</td></tr><tr><td>MTOPIntentClassification</td><td>what ingredients do is have left</td><td>GET_INFO_RECIPES</td></tr><tr><td>ToxicConversationsClassification</td><td>The guy&#x27;s a damn cop, so what do you expect?</td><td>toxic</td></tr><tr><td>TweetSentimentExtractionClassification</td><td>I really really like the song Love Story by Taylor Swift</td><td>positive</td></tr></table>

Table 3: Classification examples

<table><tr><td>Dataset</td><td>Text</td><td>Cluster</td></tr><tr><td>ArxivClusteringP2P</td><td>Finite groups of rank two which do not involve  $Qd(p)$ . Let  $p > 3$  be a prime. We show that if  $G$  is a finite group with  $p$ -rank equal to 2, then  $G$  involves  $Qd(p)$  if and only if  $G$   $p'$ -involves  $Qd(p)$ . This allows us to use a version of Glauberman&#x27;s ZJ-theorem to give a more direct construction of finite group actions on mod- $p$  homotopy spheres. We give an example to illustrate that the above conclusion does not hold for  $p \leq 3$ .</td><td>math</td></tr><tr><td>ArxivClusteringS2S</td><td>Vertical shift and simultaneous Diophantine approximation on polynomial curves</td><td>math</td></tr><tr><td>BiorxivClusteringP2P</td><td>Innate Immune sensing of Influenza A viral RNA through IFI16 promotes pyroptotic cell death Programmed cell death pathways are triggered by various stresses or stimuli, including viral infections. The mechanism underlying the regulation of these pathways upon Influenza A virus IAV infection is not well characterized. We report that a cytosolic DNA sensor IFI16 is...</td><td>immunology</td></tr><tr><td>BiorxivClusteringS2S</td><td>Association of CDH11 with ASD revealed by matched-gene co-expression analysis and mouse behavioral</td><td>neuroscience</td></tr><tr><td>MedrxivClusteringP2P</td><td>Temporal trends in the incidence of haemophagocytic lymphohistiocytosis: a nationwide cohort study from England 2003-2018. Haemophagocytic lymphohistiocytosis (HLH) is rare, results in high mortality and is increasingly being diagnosed. Little is known about what is driving the apparent rise in the incidence of this disease. Using national linked electronic health data from hospital admissions and death certification cases of HLH that were diagnosed in England between 1/1/2003 and 31/12/2018 were identified using a previously validated approach. We calculated incidence...</td><td>infectious diseases</td></tr><tr><td>MedrxivClusteringS2S</td><td>Current and Lifetime Somatic Symptom Burden Among Transition-aged Young Adults on the Autism Spectrum</td><td>psychiatry and clinical psychology</td></tr><tr><td>RedditClustering</td><td>Could anyone tell me what breed my bicolor kitten is?</td><td>r/cats</td></tr><tr><td>RedditClusteringP2P</td><td>Headaches after working out? Hey guys! I&#x27;ve been diagnosed with adhd since I was seven. I just recently got rediagnosed (22f) and I&#x27;ve been out on a different medication, adderall I was normally taking vyvanse but because of cost and no insurance adderall was more affordable. I&#x27;ve noticed that if I take adderall and workout...</td><td>r/ADHD</td></tr><tr><td>StackExchangeClustering</td><td>Does this property characterize a space as Hausdorff?</td><td>math.stackexchange.com</td></tr><tr><td>StackExchangeClusteringP2P</td><td>Google play services error DEBUG: Application is pausing, which disconnects the RTMP client. I am having this issue from past day with Google Play Services Unity. What happens is, when I install app directly ot device via Unity, the Google Play Services work fine but when I upload it as beta to play store console and install it via that then it starts to give &quot;DEBUG: Application is pausing, which disconnects the RTMP client&quot; error. I have a proper SHA1 key.</td><td>unity</td></tr><tr><td>TwentyNewsgroupsClustering</td><td>Commercial mining activities on the moon</td><td>14</td></tr></table>

Table 4: Clustering examples

<table><tr><td>Dataset</td><td>Sentence 1</td><td>Sentence 2</td><td>Label</td></tr><tr><td>SprintDuplicateQuestions</td><td>Franklin U722 USB modem signal strength</td><td>How do I know if my Franklin U772 USB Modem has a weak signal ?</td><td>1</td></tr><tr><td>TwitterSemEval2015</td><td>All the home alones watching 8 mile","All the home alones watching 8 mile</td><td>The last rap battle in 8 Mile nevr gets old ahah</td><td>0</td></tr><tr><td>TwitterURLCorpus</td><td>How the metaphors we use to describe discovery affect men and women in the sciences</td><td>Light Bulbs or Seeds ? How Metaphors for Ideas Influence Judgments About Genius</td><td>0</td></tr><tr><td>Dataset</td><td>Query</td><td>Positive</td><td>Negative</td></tr><tr><td>AskUbuntuDupQuestions</td><td>change the application icon theme but not changing the panel icons</td><td>change folder icons in ubuntu-mono-dark theme</td><td>change steam tray icon back to default</td></tr><tr><td>MindSmallReranking</td><td>Man accused in probe of Giuliani associates is freed on bail</td><td>Studies show these are the best and worst states for your retirement</td><td>There are 14 cheap days to fly left in 2019: When are they and what deals can you score?</td></tr><tr><td>SciDocsRR</td><td>Discovering social circles in ego networks</td><td>Benchmarks for testing community detection algorithms on directed and weighted graphs with overlapping communities.</td><td>Improving www proxies performance with greedy-dual-size-frequency caching policy</td></tr><tr><td>StackOverflowDupQuestions</td><td>Java launch error selection does not contain a main type</td><td>Error: Selection does not contain a main type</td><td>Selection Sort in Java</td></tr></table>

Table 5: Pair classification examples. Labels are binary.

Table 6: Reranking examples

<table><tr><td>Dataset</td><td>Sentence 1</td><td>Sentence 2</td><td>Score</td></tr><tr><td>BIOSSES</td><td>It has recently been shown that Craf is essential for Kras G12D-induced NSCLC.</td><td>It has recently become evident that Craf is essential for the onset of Kras-driven non-small cell lung cancer.</td><td>4.0</td></tr><tr><td>SICK-R</td><td>A group of children is playing in the house and there is no man standing in the background</td><td>A group of kids is playing in a yard and an old man is standing in the background</td><td>3.2</td></tr><tr><td>STS12</td><td>Nationally, the federal Centers for Disease Control and Prevention recorded 4,156 cases of West Nile, including 284 deaths.</td><td>There were 293 human cases of West Nile in Indiana in 2002, including 11 deaths statewide.</td><td>1.7</td></tr><tr><td>STS13</td><td>this frame has to do with people (the residents) residing in locations, sometimes with a co-resident.</td><td>inhabit or live in; be an inhabitant of;</td><td>2.8</td></tr><tr><td>STS14</td><td>then the captain was gone.</td><td>then the captain came back.</td><td>0.8</td></tr><tr><td>STS15</td><td>you’ll need to check the particular policies of each publisher to see what is allowed and what is not allowed.</td><td>if you need to publish the book and you have found one publisher that allows it.</td><td>3.0</td></tr><tr><td>STS16</td><td>you do not need to worry.</td><td>you don’t have to worry.</td><td>5.0</td></tr><tr><td>STS17</td><td>La gente muestra su afecto el uno por el otro.</td><td>A women giving something to other lady.</td><td>1.4</td></tr><tr><td>STS22</td><td>El secretario general de la Asociación Gremial de los Trabajadores del Subte y Premetro de Metrodelegados, Beto Pianelli, dijo que el Gobierno porteño debe convocar “inmediatamente” a licitación para la compra de nuevos trenes y retirar los que quedan en circulación...</td><td>En diálogo con el servicio informativo de la Radio Pública, el ministro de Salud de la Nación, Ginés González García, habló sobre el avance del coronavirus en la Argentina y se manifestó a favor de prorrogar la cuarentena obligatoria dispuesta por...</td><td>1</td></tr><tr><td>STSBenchmark</td><td>A man is playing the cello.</td><td>A man seated is playing the cello.</td><td>4.25</td></tr></table>

Table 7: STS examples. Scores are continuous between 0 and 5 (included).

<table><tr><td>Dataset</td><td>First set sentence</td><td>Second set sentence</td></tr><tr><td>BUCC</td><td>Morales remporte l’élection présidentielle de 2005 à la majorité absolue.</td><td>Morales went on to win the 2005 presidential election with an absolute majority.</td></tr><tr><td>Tatoeba</td><td>Chi le ha detto che Tom l’ha fatto?</td><td>Who told you that Tom did that?</td></tr></table>

Table 8: Bitext mining examples

<table><tr><td>Dataset</td><td>Human Summary</td><td>Machine Summary</td><td>Relevance</td></tr><tr><td>SummEval</td><td>V. Stiviano must pay back $2.6 million in gifts from Donald Sterling. Sterling&#x27;s wife claimed the ex-Clippers used the couple&#x27;s money for the gifts. The items included a Ferrari, two Bentleys and a Range Rover.</td><td>donald sterling , nba team last year . sterling &#x27;s wife sued for $ 2.6 million in gifts . sterling says he is the former female companion who has lost the . sterling has ordered v. stiviano to pay back $ 2.6 m in gifts after his wife sued . sterling also includes a $ 391 easter bunny costume , $ 299 and a $ 299 .</td><td>1.7</td></tr></table>

Table 9: Summarization example

![](images/b132374cbbbd75d76c3ba4d08ca0ee6994fd4133c2e8330317cfa26103e96532.jpg)  
(a) Model correlation based on all results

![](images/ef83f0f13e72d19974c96086fa2e7665a6c0771574b52a3e3f3c7e08cd9ea3a9.jpg)

<details>
<summary>heatmap</summary>

| | Class. | Class. | PairClass. | Renank. | Retr. | STS | Summ. |
|---|---|---|---|---|---|---|---|
| Class. | 68 |  |  |  |  |  | 3 |
| Class. | 72 | 81 |  |  |  |  | -4 |
| PairClass. | 58 | 95 | 83 |  |  |  | 8 |
| Renank. | 57 | 85 | 87 | 90 |  |  | -8 |
| Retr. | 79 | 75 | 85 | 78 | 69 |  |  |
| STS |  |  |  |  |  |  | 0 |
| Summ. |  |  |  |  |  |  |  |
The color intensity corresponds to the value in each cell, with darker shades indicating higher values. The chart is a standard matrix visualization for statistical analysis.
</details>

(b) Task correlation based on average task results   
Figure 6: Pearson correlations across model and task results. Left: Size variants of the same architecture show high correlations. Right: Performance on clustering and reranking correlates strongest, while summarization and classification show weaker correlation with other tasks.

<table><tr><td>Model</td><td>Public Checkpoint</td></tr><tr><td>Glove</td><td>https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d</td></tr><tr><td>Komninos</td><td>https://huggingface.co/sentence-transformers/average_word_embeddings_komninos</td></tr><tr><td>BERT</td><td>https://huggingface.co/bert-base-uncased</td></tr><tr><td>SimCSE-BERT-unsup</td><td>https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased</td></tr><tr><td>SimCSE-BERT-sup</td><td>https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased</td></tr><tr><td>coCondenser-msmarco</td><td>https://huggingface.co/sentence-transformers/msmarco-bert-co-condensor</td></tr><tr><td>Contriever</td><td>https://huggingface.co/nthakur/contriever-base-msmarco</td></tr><tr><td>SPECTER</td><td>https://huggingface.co/sentence-transformers/allenai-specter</td></tr><tr><td>LaBSE</td><td>https://huggingface.co/sentence-transformers/LaBSE</td></tr><tr><td>LASER2</td><td>https://github.com/facebookresearch/LASER</td></tr><tr><td>MiniLM-L6</td><td>https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2</td></tr><tr><td>MiniLM-L12</td><td>https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2</td></tr><tr><td>MiniLM-L12-multilingual</td><td>https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2</td></tr><tr><td>MPNet</td><td>https://huggingface.co/sentence-transformers/all-mpnet-base-v2</td></tr><tr><td>MPNet-multilingual</td><td>https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2</td></tr><tr><td>MiniLM-L12-multilingual</td><td>https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2</td></tr><tr><td>SGPT-125M-nli</td><td>https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-nli-bitfit</td></tr><tr><td>SGPT-5.8B-nli</td><td>https://huggingface.co/Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit</td></tr><tr><td>SGPT-125M-msmarco</td><td>https://huggingface.co/Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit</td></tr><tr><td>SGPT-1.3B-msmarco</td><td>https://huggingface.co/Muennighoff/SGPT-1.3B-weightedmean-msmarco-specb-bitfit</td></tr><tr><td>SGPT-2.7B-msmarco</td><td>https://huggingface.co/Muennighoff/SGPT-2.7B-weightedmean-msmarco-specb-bitfit</td></tr><tr><td>SGPT-5.8B-msmarco</td><td>https://huggingface.co/Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit</td></tr><tr><td>SGPT-BLOOM-7.1B-msmarco</td><td>https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco</td></tr><tr><td>SGPT-BLOOM-1.7B-nli</td><td>https://huggingface.co/bigscience-data/sgpt-bloom-1b7-nli</td></tr><tr><td>GTR-Base</td><td>https://huggingface.co/sentence-transformers/gtr-t5-base</td></tr><tr><td>GTR-Large</td><td>https://huggingface.co/sentence-transformers/gtr-t5-large</td></tr><tr><td>GTR-XL</td><td>https://huggingface.co/sentence-transformers/gtr-t5-xl</td></tr><tr><td>GTR-XXL</td><td>https://huggingface.co/sentence-transformers/gtr-t5-xxl</td></tr><tr><td>ST5-Base</td><td>https://huggingface.co/sentence-transformers/sentence-t5-base</td></tr><tr><td>ST5-Large</td><td>https://huggingface.co/sentence-transformers/sentence-t5-large</td></tr><tr><td>ST5-XL</td><td>https://huggingface.co/sentence-transformers/sentence-t5-xl</td></tr><tr><td>ST5-XXL</td><td>https://huggingface.co/sentence-transformers/sentence-t5-xxl</td></tr></table>

Table 10: Publicly available model links used for evaluation

<table><tr><td>Dataset</td><td>Glove</td><td>Kreemont</td><td>BERT</td><td>SimSE/BEERT-score</td><td>SimSE/BEERT-score</td><td>ConCond-mance</td><td>ConCond-mance</td><td>SPFCTER</td><td>LaRSE</td><td>LASER2</td><td>MinLM-L6</td><td>MinalLM-L12</td><td>MinalLM-L12</td><td>MPNet</td><td>MPNet-monant</td><td>SGPT-12SM-nk</td><td>SGPT-3SB-nk</td><td>SGPT-3SB-monant</td><td>SGPT-12BM-monant</td><td>SGPT-2TB-monant</td><td>SGPT-5TB-monant</td><td>SGPT-BLOOM-17B-monant</td><td>GTR-Base</td><td>GTR-Large</td><td>GTR-Base</td><td>GTR-Large</td><td>GTR-XXL</td><td>ST5-Base</td><td>ST5-Large</td><td>ST5-XXL</td><td></td><td></td></tr><tr><td>AmmestCountcentralClassification</td><td>56.91</td><td>40.54</td><td>74.29</td><td>67.09</td><td>75.79</td><td>64.06</td><td>72.19</td><td>58.70</td><td>75.93</td><td>76.84</td><td>64.15</td><td>65.28</td><td>71.57</td><td>65.27</td><td>75.81</td><td>76.40</td><td>65.88</td><td>74.07</td><td>61.24</td><td>61.21</td><td>67.57</td><td>60.22</td><td>68.06</td><td>69.33</td><td>70.03</td><td>68.60</td><td>67.30</td><td>75.82</td><td>75.81</td><td>76.01</td><td>77.07</td><td></td></tr><tr><td>AmmostPolarityClassification</td><td>60.32</td><td>59.59</td><td>71.33</td><td>74.48</td><td>82.47</td><td>66.88</td><td>68.63</td><td>57.77</td><td>68.95</td><td>61.01</td><td>62.58</td><td>62.98</td><td>69.21</td><td>67.13</td><td>76.41</td><td>92.83</td><td>74.94</td><td>82.31</td><td>63.40</td><td>73.21</td><td>71.44</td><td>71.26</td><td>68.97</td><td>67.82</td><td>73.92</td><td>74.58</td><td>75.05</td><td>85.12</td><td>92.87</td><td>93.17</td><td>92.79</td><td></td></tr><tr><td>AmmostTumorClassification</td><td>29.67</td><td>26.01</td><td>33.56</td><td>33.45</td><td>39.69</td><td>34.83</td><td>37.42</td><td>26.20</td><td>35.80</td><td>28.71</td><td>31.78</td><td>30.79</td><td>35.13</td><td>31.16</td><td>38.51</td><td>47.43</td><td>35.10</td><td>41.38</td><td>31.17</td><td>34.96</td><td>35.75</td><td>39.19</td><td>35.86</td><td>38.74</td><td>39.19</td><td>38.62</td><td>38.03</td><td>38.74</td><td>40.04</td><td>40.04</td><td>40.04</td><td></td></tr><tr><td>BosingTCh Classification</td><td>67.69</td><td>67.05</td><td>63.43</td><td>73.55</td><td>75.76</td><td>82.35</td><td>80.02</td><td>66.06</td><td>69.85</td><td>57.76</td><td>79.75</td><td>80.40</td><td>79.77</td><td>81.88</td><td>81.07</td><td>67.04</td><td>74.84</td><td>71.78</td><td>87.70</td><td>82.06</td><td>83.22</td><td>84.49</td><td>84.33</td><td>79.26</td><td>81.21</td><td>82.22</td><td>82.32</td><td>76.48</td><td>78.46</td><td>80.88</td><td>82.31</td><td></td></tr><tr><td>EmstaticClassification</td><td>36.93</td><td>33.18</td><td>35.28</td><td>42.22</td><td>44.83</td><td>43.91</td><td>44.77</td><td>24.82</td><td>37.22</td><td>24.83</td><td>38.43</td><td>41.17</td><td>42.37</td><td>39.73</td><td>45.84</td><td>80.32</td><td>42.23</td><td>49.82</td><td>30.08</td><td>46.39</td><td>49.22</td><td>49.68</td><td>44.83</td><td>42.20</td><td>46.32</td><td>45.55</td><td>43.19</td><td>51.36</td><td>51.73</td><td>51.95</td><td>48.57</td><td></td></tr><tr><td>EmoDistributionClassification</td><td>62.57</td><td>59.00</td><td>60.13</td><td>69.02</td><td>63.13</td><td>60.17</td><td>57.06</td><td>26.20</td><td>57.90</td><td>37.90</td><td>51.61</td><td>50.79</td><td>50.46</td><td>46.46</td><td>64.73</td><td>89.38</td><td>62.90</td><td>74.33</td><td>80.67</td><td>64.05</td><td>63.55</td><td>66.64</td><td>63.05</td><td>65.00</td><td>66.00</td><td>65.00</td><td>65.00</td><td>65.00</td><td>65.00</td><td>65.00</td><td>65.00</td><td></td></tr><tr><td>MonocclusionClassification</td><td>56.39</td><td>57.21</td><td>59.88</td><td>59.64</td><td>63.95</td><td>70.40</td><td>53.73</td><td>51.78</td><td>51.43</td><td>47.90</td><td>67.40</td><td>67.15</td><td>67.84</td><td>66.92</td><td>65.57</td><td>83.18</td><td>58.57</td><td>70.00</td><td>61.41</td><td>60.65</td><td>60.01</td><td>70.39</td><td>69.67</td><td>67.05</td><td>70.06</td><td>70.23</td><td>70.61</td><td>69.74</td><td>71.78</td><td>72.09</td><td>73.44</td><td></td></tr><tr><td>MonocclusionProportionClassification</td><td>66.05</td><td>65.01</td><td>64.28</td><td>66.25</td><td>70.78</td><td>73.73</td><td>58.50</td><td>58.38</td><td>66.61</td><td>55.92</td><td>75.76</td><td>76.00</td><td>76.51</td><td>76.51</td><td>75.35</td><td>72.57</td><td>69.63</td><td>76.54</td><td>69.74</td><td>76.04</td><td>75.90</td><td>76.29</td><td>75.43</td><td>75.40</td><td>76.00</td><td>76.00</td><td>75.90</td><td>75.90</td><td>75.90</td><td>76.00</td><td>76.00</td><td></td></tr><tr><td>MTOPDominantClassification</td><td>79.11</td><td>78.57</td><td>82.63</td><td>81.71</td><td>84.25</td><td>91.34</td><td>93.18</td><td>74.53</td><td>86.06</td><td>75.36</td><td>91.96</td><td>91.90</td><td>87.06</td><td>92.08</td><td>89.24</td><td>89.89</td><td>81.52</td><td>80.64</td><td>86.96</td><td>92.08</td><td>92.56</td><td>93.47</td><td>91.68</td><td>92.42</td><td>94.01</td><td>93.60</td><td>93.84</td><td>90.34</td><td>90.99</td><td>90.73</td><td>92.49</td><td></td></tr><tr><td>TumorClassification</td><td>83.82</td><td>81.14</td><td>86.14</td><td>83.80</td><td>86.11</td><td>71.07</td><td>71.07</td><td>50.05</td><td>83.01</td><td>69.13</td><td>82.18</td><td>62.45</td><td>63.75</td><td>70.21</td><td>68.69</td><td>84.80</td><td>78.24</td><td>70.08</td><td>62.25</td><td>71.19</td><td>72.49</td><td>72.49</td><td>72.49</td><td>72.49</td><td>72.49</td><td>72.49</td><td>72.49</td><td>72.49</td><td>72.49</td><td>72.49</td><td>72.49</td><td></td></tr><tr><td>TotalConcentrationClassification</td><td>65.40</td><td>67.76</td><td>70.01</td><td>68.82</td><td>72.06</td><td>64.01</td><td>67.77</td><td>57.44</td><td>66.90</td><td>54.05</td><td>66.99</td><td>67.47</td><td>66.07</td><td>60.86</td><td>71.02</td><td>50.00</td><td>62.75</td><td>69.94</td><td>62.66</td><td>68.73</td><td>68.84</td><td>67.71</td><td>66.55</td><td>66.60</td><td>68.65</td><td>67.52</td><td>68.48</td><td>68.20</td><td>71.73</td><td>70.95</td><td>70.04</td><td></td></tr><tr><td>TotalConcentrationProportionClassification</td><td>80.80</td><td>80.68</td><td>81.80</td><td>83.36</td><td>79.72</td><td>85.74</td><td>86.10</td><td>56.25</td><td>85.12</td><td>59.22</td><td>85.83</td><td>84.25</td><td>56.12</td><td>55.46</td><td>89.03</td><td>93.33</td><td>54.82</td><td>60.55</td><td>62.44</td><td>83.67</td><td>83.87</td><td>56.05</td><td>56.05</td><td>56.05</td><td>56.05</td><td>56.05</td><td>56.05</td><td>56.05</td><td>56.05</td><td>56.05</td><td>56.05</td><td></td></tr><tr><td>AcraClusteringPTP</td><td>32.56</td><td>34.73</td><td>35.19</td><td>32.61</td><td>35.18</td><td>36.94</td><td>42.61</td><td>44.75</td><td>52.13</td><td>17.77</td><td>46.55</td><td>46.07</td><td>38.33</td><td>48.38</td><td>37.78</td><td>41.49</td><td>34.75</td><td>40.55</td><td>39.31</td><td>43.38</td><td>44.72</td><td>45.59</td><td>44.59</td><td>37.49</td><td>35.50</td><td>37.90</td><td>37.90</td><td>39.28</td><td>41.62</td><td>41.62</td><td>42.89</td><td></td></tr><tr><td>AraicClusteringQS</td><td>23.14</td><td>26.01</td><td>27.53</td><td>24.68</td><td>27.54</td><td>29.03</td><td>32.32</td><td>35.27</td><td>22.05</td><td>12.59</td><td>37.86</td><td>37.50</td><td>31.55</td><td>39.72</td><td>31.68</td><td>28.47</td><td>24.68</td><td>32.49</td><td>28.24</td><td>33.71</td><td>35.08</td><td>38.86</td><td>38.03</td><td>27.18</td><td>30.55</td><td>30.45</td><td>32.39</td><td>27.26</td><td>29.44</td><td>31.17</td><td>33.47</td><td></td></tr><tr><td>BiotinClusteringQS</td><td>29.27</td><td>29.92</td><td>30.12</td><td>24.90</td><td>30.15</td><td>32.35</td><td>34.97</td><td>29.88</td><td>29.84</td><td>12.68</td><td>38.48</td><td>38.06</td><td>33.92</td><td>39.60</td><td>33.09</td><td>36.86</td><td>28.93</td><td>33.96</td><td>33.63</td><td>35.06</td><td>34.41</td><td>36.51</td><td>36.36</td><td>27.66</td><td>29.00</td><td>30.00</td><td>30.00</td><td>30.00</td><td>30.00</td><td>30.00</td><td>30.00</td><td></td></tr><tr><td>BiomerClusteringQS</td><td>19.18</td><td>20.71</td><td>24.77</td><td>19.55</td><td>24.67</td><td>28.16</td><td>29.08</td><td>34.53</td><td>20.57</td><td>8.83</td><td>33.17</td><td>33.21</td><td>29.44</td><td>35.02</td><td>29.60</td><td>27.53</td><td>23.08</td><td>29.13</td><td>27.04</td><td>30.71</td><td>30.53</td><td>33.73</td><td>32.48</td><td>23.25</td><td>25.72</td><td>26.06</td><td>27.50</td><td>22.92</td><td>24.02</td><td>24.67</td><td>28.86</td><td></td></tr><tr><td>CubricClusteringQS</td><td>26.15</td><td>26.11</td><td>26.09</td><td>23.40</td><td>26.20</td><td>30.23</td><td>31.59</td><td>18.06</td><td>30.13</td><td>34.17</td><td>34.11</td><td>34.11</td><td>31.59</td><td>35.96</td><td>31.90</td><td>28.39</td><td>28.30</td><td>30.12</td><td>31.37</td><td>32.08</td><td>31.59</td><td>31.59</td><td>31.59</td><td>27.12</td><td>31.59</td><td>31.59</td><td>32.08</td><td>27.12</td><td>27.12</td><td>27.12</td><td>27.12</td><td></td></tr><tr><td>MidroClusteringQS</td><td>20.38</td><td>21.50</td><td>23.60</td><td>21.97</td><td>24.12</td><td>27.01</td><td>27.27</td><td>31.66</td><td>24.82</td><td>16.63</td><td>32.29</td><td>32.24</td><td>30.87</td><td>32.87</td><td>31.70</td><td>26.50</td><td>24.93</td><td>28.02</td><td>26.87</td><td>29.45</td><td>28.77</td><td>28.76</td><td>29.26</td><td>25.13</td><td>27.39</td><td>26.69</td><td>27.56</td><td>26.13</td><td>26.33</td><td>26.93</td><td>26.82</td><td></td></tr><tr><td>MidroClusteringQS</td><td>28.40</td><td>28.40</td><td>27.24</td><td>32.14</td><td>40.27</td><td>40.64</td><td>54.89</td><td>28.78</td><td>42.97</td><td>28.78</td><td>50.67</td><td>50.47</td><td>42.20</td><td>54.87</td><td>45.24</td><td>42.47</td><td>33.70</td><td>42.13</td><td>40.23</td><td>48.22</td><td>46.47</td><td>40.47</td><td>38.13</td><td>36.13</td><td>38.13</td><td>38.13</td><td>38.13</td><td>38.13</td><td>38.13</td><td>38.13</td><td>38.13</td><td></td></tr><tr><td>ReddClusteringPTP</td><td>35.82</td><td>7.37</td><td>43.32</td><td>45.14</td><td>47.74</td><td>53.53</td><td>57.58</td><td>35.06</td><td>49.14</td><td>26.42</td><td>54.15</td><td>54.80</td><td>50.73</td><td>56.77</td><td>51.31</td><td>58.10</td><td>41.01</td><td>48.02</td><td>49.09</td><td>53.18</td><td>54.17</td><td>55.75</td><td>54.52</td><td>58.53</td><td>61.67</td><td>63.11</td><td>62.84</td><td>59.67</td><td>62.50</td><td>62.34</td><td>64.46</td><td></td></tr><tr><td>SickelChorasanClusteringPTP</td><td>24.55</td><td>43.80</td><td>43.58</td><td>59.44</td><td>59.44</td><td>52.25</td><td>59.44</td><td>32.25</td><td>59.44</td><td>15.63</td><td>58.00</td><td>55.90</td><td>49.80</td><td>53.80</td><td>52.98</td><td>53.52</td><td>44.50</td><td>54.13</td><td>52.74</td><td>50.86</td><td>59.19</td><td>59.21</td><td>54.13</td><td>64.71</td><td>64.71</td><td>64.71</td><td>64.71</td><td>64.71</td><td>64.71</td><td>64.71</td><td>64.71</td><td></td></tr><tr><td>SickelChorasanProportionPTP</td><td>24.51</td><td>26.23</td><td>26.95</td><td>28.50</td><td>29.45</td><td>30.48</td><td>32.25</td><td>31.46</td><td>28.83</td><td>15.63</td><td>58.00</td><td>33.13</td><td>31.69</td><td>34.28</td><td>32.94</td><td>30.43</td><td>28.21</td><td>31.12</td><td>32.66</td><td>32.36</td><td>32.97</td><td>33.95</td><td>34.31</td><td>33.01</td><td>33.21</td><td>32.73</td><td>32.85</td><td>35.68</td><td>36.86</td><td>34.79</td><td>35.25</td><td></td></tr><tr><td>StressClusteringQS</td><td>23.81</td><td>41.42</td><td>23.21</td><td>23.21</td><td>34.88</td><td>38.68</td><td>44.82</td><td>22.25</td><td>21.28</td><td>17.28</td><td>41.88</td><td>40.47</td><td>38.73</td><td>41.86</td><td>38.74</td><td>46.28</td><td>38.24</td><td>37.28</td><td>37.23</td><td>38.13</td><td>40.89</td><td>40.89</td><td>37.28</td><td>40.17</td><td>39.93</td><td>39.93</td><td>39.93</td><td>40.17</td><td>39.93</td><td>39.93</td><td>39.93</td><td></td></tr><tr><td>SpiretPrincipalQuestions</td><td>86.96</td><td>85.55</td><td>56.82</td><td>69.41</td><td>69.30</td><td>96.09</td><td>95.55</td><td>71.63</td><td>89.76</td><td>65.54</td><td>94.55</td><td>92.45</td><td>89.46</td><td>90.15</td><td>90.55</td><td>77.85</td><td>77.73</td><td>80.54</td><td>89.89</td><td>92.58</td><td>93.47</td><td>93.84</td><td>94.93</td><td>94.55</td><td>95.05</td><td>95.45</td><td>95.68</td><td>91.23</td><td>89.01</td><td>91.44</td><td>88.89</td><td></td></tr><tr><td>TwitterSentExUS 2015</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="4"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="4"></td><td rowspan="20"></td><td rowspan="20"></td><td rowspan="20"></td><td rowspan="20"></td><td rowspan="20"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="16"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="14"></td><td rowspan="14"></td><td rowspan="14"></td><td rowspan="14"></td><td rowspan="14"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="11"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="8"></td><td rowspan="8"></td><td rowspan="8"></td><td rowspan="8"></td><td rowspan="8"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="6"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="5"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 11: All English results. The main score for each task is reported as described in Section 3.2.

<table><tr><td>Dataset</td><td>Language</td><td>LASER2</td><td>LaBSE</td><td>MiniLM-L12-multilingual</td><td>MPNet-multilingual</td><td>SGPT-BLOOM-7.1B-msmarco</td></tr><tr><td>BUCC</td><td>de-en</td><td>99.21</td><td>99.35</td><td>97.11</td><td>98.59</td><td>54.00</td></tr><tr><td>BUCC</td><td>fr-en</td><td>98.39</td><td>98.72</td><td>94.99</td><td>96.89</td><td>97.06</td></tr><tr><td>BUCC</td><td>ru-en</td><td>97.62</td><td>97.78</td><td>95.06</td><td>96.44</td><td>45.30</td></tr><tr><td>BUCC</td><td>zh-en</td><td>97.70</td><td>99.16</td><td>95.63</td><td>97.56</td><td>97.96</td></tr><tr><td>Tatoeba</td><td>sqi-eng</td><td>97.22</td><td>96.76</td><td>98.17</td><td>98.57</td><td>10.38</td></tr><tr><td>Tatoeba</td><td>fry-eng</td><td>42.07</td><td>89.31</td><td>31.13</td><td>43.54</td><td>24.62</td></tr><tr><td>Tatoeba</td><td>kur-eng</td><td>19.09</td><td>83.59</td><td>46.94</td><td>61.44</td><td>8.26</td></tr><tr><td>Tatoeba</td><td>tur-eng</td><td>98.03</td><td>98.00</td><td>95.08</td><td>96.17</td><td>6.15</td></tr><tr><td>Tatoeba</td><td>deu-eng</td><td>99.07</td><td>99.20</td><td>97.02</td><td>97.73</td><td>70.10</td></tr><tr><td>Tatoeba</td><td>nld-eng</td><td>95.35</td><td>96.07</td><td>94.58</td><td>95.50</td><td>29.74</td></tr><tr><td>Tatoeba</td><td>ron-eng</td><td>96.52</td><td>96.92</td><td>95.30</td><td>96.43</td><td>27.23</td></tr><tr><td>Tatoeba</td><td>ang-eng</td><td>25.22</td><td>59.28</td><td>10.24</td><td>16.72</td><td>28.76</td></tr><tr><td>Tatoeba</td><td>ido-eng</td><td>80.86</td><td>89.42</td><td>40.25</td><td>43.91</td><td>43.91</td></tr><tr><td>Tatoeba</td><td>jav-eng</td><td>9.95</td><td>79.77</td><td>17.04</td><td>23.39</td><td>15.02</td></tr><tr><td>Tatoeba</td><td>isl-eng</td><td>94.32</td><td>94.75</td><td>24.07</td><td>59.25</td><td>6.29</td></tr><tr><td>Tatoeba</td><td>slv-eng</td><td>95.40</td><td>96.03</td><td>96.92</td><td>97.08</td><td>10.14</td></tr><tr><td>Tatoeba</td><td>cym-eng</td><td>5.85</td><td>92.00</td><td>13.25</td><td>22.31</td><td>6.97</td></tr><tr><td>Tatoeba</td><td>kaz-eng</td><td>53.30</td><td>87.49</td><td>34.89</td><td>61.49</td><td>3.32</td></tr><tr><td>Tatoeba</td><td>est-eng</td><td>96.43</td><td>96.55</td><td>97.33</td><td>98.40</td><td>4.76</td></tr><tr><td>Tatoeba</td><td>heb-eng</td><td>0.00</td><td>91.53</td><td>86.88</td><td>88.26</td><td>1.69</td></tr><tr><td>Tatoeba</td><td>gla-eng</td><td>1.52</td><td>85.66</td><td>3.61</td><td>4.72</td><td>2.09</td></tr><tr><td>Tatoeba</td><td>mar-eng</td><td>92.93</td><td>92.65</td><td>92.38</td><td>93.83</td><td>45.53</td></tr><tr><td>Tatoeba</td><td>lat-eng</td><td>64.81</td><td>80.07</td><td>19.47</td><td>24.25</td><td>28.76</td></tr><tr><td>Tatoeba</td><td>bel-eng</td><td>79.54</td><td>95.00</td><td>67.73</td><td>79.94</td><td>8.03</td></tr><tr><td>Tatoeba</td><td>pms-eng</td><td>36.23</td><td>64.57</td><td>30.70</td><td>34.19</td><td>31.94</td></tr><tr><td>Tatoeba</td><td>gle-eng</td><td>4.20</td><td>93.80</td><td>11.62</td><td>16.85</td><td>3.26</td></tr><tr><td>Tatoeba</td><td>pes-eng</td><td>93.13</td><td>94.70</td><td>92.59</td><td>93.47</td><td>12.13</td></tr><tr><td>Tatoeba</td><td>nob-eng</td><td>95.77</td><td>98.40</td><td>97.73</td><td>98.53</td><td>21.07</td></tr><tr><td>Tatoeba</td><td>bul-eng</td><td>93.57</td><td>94.58</td><td>92.65</td><td>93.52</td><td>20.09</td></tr><tr><td>Tatoeba</td><td>cbk-eng</td><td>77.17</td><td>79.44</td><td>55.37</td><td>58.68</td><td>64.63</td></tr><tr><td>Tatoeba</td><td>hun-eng</td><td>95.20</td><td>96.55</td><td>91.58</td><td>94.18</td><td>5.07</td></tr><tr><td>Tatoeba</td><td>uig-eng</td><td>56.49</td><td>92.40</td><td>24.39</td><td>48.35</td><td>1.27</td></tr><tr><td>Tatoeba</td><td>rus-eng</td><td>92.58</td><td>93.75</td><td>91.87</td><td>92.92</td><td>59.84</td></tr><tr><td>Tatoeba</td><td>spa-eng</td><td>97.33</td><td>98.40</td><td>95.42</td><td>97.00</td><td>94.48</td></tr><tr><td>Tatoeba</td><td>hye-eng</td><td>88.72</td><td>94.09</td><td>93.28</td><td>94.38</td><td>0.50</td></tr><tr><td>Tatoeba</td><td>tel-eng</td><td>96.72</td><td>97.86</td><td>36.40</td><td>79.73</td><td>64.62</td></tr><tr><td>Tatoeba</td><td>afr-eng</td><td>92.59</td><td>96.18</td><td>58.22</td><td>72.96</td><td>16.62</td></tr><tr><td>Tatoeba</td><td>mon-eng</td><td>3.42</td><td>95.91</td><td>95.04</td><td>96.14</td><td>2.85</td></tr><tr><td>Tatoeba</td><td>arz-eng</td><td>66.16</td><td>76.00</td><td>51.26</td><td>55.69</td><td>70.66</td></tr><tr><td>Tatoeba</td><td>hrv-eng</td><td>96.72</td><td>96.95</td><td>95.98</td><td>97.00</td><td>12.79</td></tr><tr><td>Tatoeba</td><td>nov-eng</td><td>60.02</td><td>74.38</td><td>47.99</td><td>50.23</td><td>52.23</td></tr><tr><td>Tatoeba</td><td>gsw-eng</td><td>27.52</td><td>46.50</td><td>25.74</td><td>25.12</td><td>21.03</td></tr><tr><td>Tatoeba</td><td>nds-eng</td><td>77.13</td><td>79.42</td><td>32.16</td><td>38.88</td><td>23.92</td></tr><tr><td>Tatoeba</td><td>ukr-eng</td><td>93.52</td><td>93.97</td><td>92.82</td><td>92.67</td><td>22.06</td></tr><tr><td>Tatoeba</td><td>uzb-eng</td><td>23.20</td><td>84.23</td><td>17.14</td><td>23.19</td><td>4.71</td></tr><tr><td>Tatoeba</td><td>lit-eng</td><td>96.20</td><td>96.47</td><td>93.16</td><td>95.37</td><td>4.49</td></tr><tr><td>Tatoeba</td><td>ina-eng</td><td>93.93</td><td>95.37</td><td>79.13</td><td>84.32</td><td>73.67</td></tr><tr><td>Tatoeba</td><td>lfn-eng</td><td>63.39</td><td>67.54</td><td>47.02</td><td>49.56</td><td>44.85</td></tr><tr><td>Tatoeba</td><td>zsm-eng</td><td>95.41</td><td>95.62</td><td>95.31</td><td>95.80</td><td>79.95</td></tr><tr><td>Tatoeba</td><td>ita-eng</td><td>94.32</td><td>92.72</td><td>93.05</td><td>93.76</td><td>65.04</td></tr><tr><td>Tatoeba</td><td>cmn-eng</td><td>85.62</td><td>95.10</td><td>94.93</td><td>95.83</td><td>91.45</td></tr><tr><td>Tatoeba</td><td>lvs-eng</td><td>95.33</td><td>95.88</td><td>97.87</td><td>97.53</td><td>6.55</td></tr><tr><td>Tatoeba</td><td>glg-eng</td><td>96.14</td><td>96.82</td><td>94.00</td><td>95.32</td><td>79.86</td></tr><tr><td>Tatoeba</td><td>ceb-eng</td><td>9.93</td><td>64.42</td><td>8.05</td><td>7.39</td><td>6.64</td></tr><tr><td>Tatoeba</td><td>bre-eng</td><td>31.2</td><td>15.07</td><td>5.56</td><td>6.42</td><td>4.67</td></tr><tr><td>Tatoeba</td><td>ben-eng</td><td>89.43</td><td>88.55</td><td>36.48</td><td>64.90</td><td>75.98</td></tr><tr><td>Tatoeba</td><td>swg-eng</td><td>33.10</td><td>59.36</td><td>26.31</td><td>22.80</td><td>16.89</td></tr><tr><td>Tatoeba</td><td>arq-eng</td><td>26.63</td><td>42.69</td><td>18.60</td><td>19.84</td><td>27.75</td></tr><tr><td>Tatoeba</td><td>kab-eng</td><td>65.88</td><td>4.31</td><td>1.16</td><td>1.41</td><td>1.69</td></tr><tr><td>Tatoeba</td><td>fra-eng</td><td>94.28</td><td>94.86</td><td>91.72</td><td>93.12</td><td>91.44</td></tr><tr><td>Tatoeba</td><td>por-eng</td><td>94.54</td><td>94.14</td><td>92.13</td><td>93.02</td><td>92.62</td></tr><tr><td>Tatoeba</td><td>tat-eng</td><td>34.74</td><td>85.92</td><td>10.25</td><td>10.89</td><td>3.59</td></tr><tr><td>Tatoeba</td><td>oci-eng</td><td>58.13</td><td>65.81</td><td>38.57</td><td>43.49</td><td>40.17</td></tr><tr><td>Tatoeba</td><td>pol-eng</td><td>97.32</td><td>97.22</td><td>94.28</td><td>96.95</td><td>14.09</td></tr><tr><td>Tatoeba</td><td>war-eng</td><td>8.25</td><td>60.29</td><td>7.25</td><td>7.42</td><td>10.38</td></tr><tr><td>Tatoeba</td><td>aze-eng</td><td>82.41</td><td>94.93</td><td>62.10</td><td>76.36</td><td>6.32</td></tr><tr><td>Tatoeba</td><td>vie-eng</td><td>96.73</td><td>97.20</td><td>95.12</td><td>97.23</td><td>94.20</td></tr><tr><td>Tatoeba</td><td>nno-eng</td><td>72.75</td><td>94.48</td><td>76.34</td><td>81.41</td><td>16.28</td></tr><tr><td>Tatoeba</td><td>cha-eng</td><td>14.86</td><td>31.77</td><td>15.98</td><td>12.59</td><td>23.26</td></tr><tr><td>Tatoeba</td><td>mhr-eng</td><td>6.86</td><td>15.74</td><td>6.89</td><td>7.57</td><td>1.56</td></tr><tr><td>Tatoeba</td><td>dan-eng</td><td>95.22</td><td>95.71</td><td>94.80</td><td>96.17</td><td>23.52</td></tr><tr><td>Tatoeba</td><td>ell-eng</td><td>96.20</td><td>95.35</td><td>95.43</td><td>94.93</td><td>5.34</td></tr><tr><td>Tatoeba</td><td>amh-eng</td><td>80.82</td><td>91.47</td><td>36.21</td><td>53.49</td><td>0.03</td></tr><tr><td>Tatoeba</td><td>pam-eng</td><td>3.24</td><td>10.73</td><td>5.41</td><td>5.39</td><td>5.85</td></tr><tr><td>Tatoeba</td><td>hsb-eng</td><td>45.75</td><td>67.11</td><td>36.10</td><td>44.32</td><td>9.68</td></tr><tr><td>Tatoeba</td><td>srp-eng</td><td>93.64</td><td>94.43</td><td>92.24</td><td>94.12</td><td>11.69</td></tr><tr><td>Tatoeba</td><td>epo-eng</td><td>96.61</td><td>98.20</td><td>41.73</td><td>55.12</td><td>26.20</td></tr><tr><td>Tatoeba</td><td>kzj-eng</td><td>4.46</td><td>11.33</td><td>6.24</td><td>5.88</td><td>5.17</td></tr><tr><td>Tatoeba</td><td>awa-eng</td><td>33.74</td><td>71.70</td><td>33.43</td><td>42.83</td><td>35.01</td></tr><tr><td>Tatoeba</td><td>fao-eng</td><td>57.04</td><td>87.40</td><td>27.51</td><td>38.24</td><td>12.61</td></tr><tr><td>Tatoeba</td><td>mal-eng</td><td>98.16</td><td>98.45</td><td>32.20</td><td>88.46</td><td>83.30</td></tr><tr><td>Tatoeba</td><td>ile-eng</td><td>87.88</td><td>85.58</td><td>57.71</td><td>60.36</td><td>59.59</td></tr><tr><td>Tatoeba</td><td>bos-eng</td><td>95.86</td><td>94.92</td><td>93.27</td><td>94.02</td><td>13.65</td></tr><tr><td>Tatoeba</td><td>cor-eng</td><td>4.45</td><td>10.11</td><td>3.42</td><td>3.53</td><td>2.83</td></tr><tr><td>Tatoeba</td><td>cat-eng</td><td>95.80</td><td>95.38</td><td>94.42</td><td>96.05</td><td>88.31</td></tr><tr><td>Tatoeba</td><td>eus-eng</td><td>93.32</td><td>95.01</td><td>23.18</td><td>31.33</td><td>53.38</td></tr><tr><td>Tatoeba</td><td>yue-eng</td><td>87.75</td><td>89.58</td><td>71.45</td><td>77.58</td><td>77.03</td></tr><tr><td>Tatoeba</td><td>swe-eng</td><td>95.31</td><td>95.63</td><td>94.42</td><td>95.45</td><td>19.53</td></tr><tr><td>Tatoeba</td><td>dtp-eng</td><td>7.39</td><td>10.85</td><td>5.69</td><td>5.03</td><td>3.41</td></tr><tr><td>Tatoeba</td><td>kat-eng</td><td>81.16</td><td>95.02</td><td>95.44</td><td>95.46</td><td>0.42</td></tr><tr><td>Tatoeba</td><td>jpn-eng</td><td>93.78</td><td>95.38</td><td>90.41</td><td>92.51</td><td>71.36</td></tr><tr><td>Tatoeba</td><td>csb-eng</td><td>27.03</td><td>52.57</td><td>21.56</td><td>23.73</td><td>10.03</td></tr><tr><td>Tatoeba</td><td>xho-eng</td><td>4.68</td><td>91.55</td><td>4.52</td><td>6.53</td><td>5.51</td></tr><tr><td>Tatoeba</td><td>orv-eng</td><td>23.24</td><td>38.93</td><td>15.10</td><td>23.77</td><td>5.79</td></tr><tr><td>Tatoeba</td><td>ind-eng</td><td>92.98</td><td>93.66</td><td>92.74</td><td>93.50</td><td>88.04</td></tr><tr><td>Tatoeba</td><td>tuk-eng</td><td>16.35</td><td>75.27</td><td>15.16</td><td>14.91</td><td>5.48</td></tr><tr><td>Tatoeba</td><td>max-eng</td><td>36.96</td><td>63.26</td><td>45.25</td><td>48.77</td><td>36.14</td></tr><tr><td>Tatoeba</td><td>swh-eng</td><td>55.66</td><td>84.50</td><td>14.48</td><td>16.02</td><td>16.74</td></tr><tr><td>Tatoeba</td><td>hin-eng</td><td>95.32</td><td>96.87</td><td>97.62</td><td>97.75</td><td>85.23</td></tr><tr><td>Tatoeba</td><td>dsb-eng</td><td>42.34</td><td>64.81</td><td>33.43</td><td>36.85</td><td>8.78</td></tr><tr><td>Tatoeba</td><td>ber-eng</td><td>77.63</td><td>8.40</td><td>4.43</td><td>4.88</td><td>4.92</td></tr><tr><td>Tatoeba</td><td>tam-eng</td><td>87.32</td><td>89.0</td><td>24.64</td><td>73.60</td><td>72.76</td></tr><tr><td>Tatoeba</td><td>slk-eng</td><td>95.82</td><td>96.5</td><td>95.15</td><td>96.62</td><td>9.98</td></tr><tr><td>Tatoeba</td><td>tgl-eng</td><td>63.19</td><td>96.02</td><td>13.09</td><td>17.67</td><td>10.70</td></tr><tr><td>Tatoeba</td><td>ast-eng</td><td>76.35</td><td>90.68</td><td>62.17</td><td>70.08</td><td>71.13</td></tr><tr><td>Tatoeba</td><td>mkd-eng</td><td>93.63</td><td>93.6</td><td>91.00</td><td>93.02</td><td>10.47</td></tr><tr><td>Tatoeba</td><td>khm-eng</td><td>74.19</td><td>78.37</td><td>32.11</td><td>58.80</td><td>0.37</td></tr><tr><td>Tatoeba</td><td>ces-eng</td><td>95.52</td><td>96.68</td><td>95.12</td><td>95.73</td><td>9.55</td></tr><tr><td>Tatoeba</td><td>tzl-eng</td><td>36.56</td><td>58.88</td><td>25.46</td><td>34.21</td><td>27.82</td></tr><tr><td>Tatoeba</td><td>urd-eng</td><td>84.23</td><td>93.22</td><td>94.57</td><td>95.12</td><td>70.10</td></tr><tr><td>Tatoeba</td><td>ara-eng</td><td>90.14</td><td>88.80</td><td>87.93</td><td>90.19</td><td>85.37</td></tr><tr><td>Tatoeba</td><td>kor-eng</td><td>87.97</td><td>90.95</td><td>92.52</td><td>93.07</td><td>22.39</td></tr><tr><td>Tatoeba</td><td>yid-eng</td><td>2.49</td><td>88.79</td><td>14.38</td><td>30.73</td><td>0.16</td></tr><tr><td>Tatoeba</td><td>fin-eng</td><td>96.98</td><td>96.37</td><td>93.10</td><td>95.92</td><td>3.41</td></tr><tr><td>Tatoeba</td><td>tha-eng</td><td>96.38</td><td>96.14</td><td>96.72</td><td>95.99</td><td>2.22</td></tr><tr><td>Tatoeba</td><td>wuu-eng</td><td>75.09</td><td>90.18</td><td>76.00</td><td>78.25</td><td>79.58</td></tr></table>

Table 12: Multilingual bitext mining results. Scores are f1.

<table><tr><td>Dataset</td><td>Language</td><td>LASER2</td><td>LaBSE</td><td>MiniLM-L12-multilingual</td><td>MPNet-multilingual</td><td>SGPT-BLOOM-7.1B-msmarco</td></tr><tr><td>AmazonCounterfactualClassification</td><td>de</td><td>67.82</td><td>73.17</td><td>68.35</td><td>69.95</td><td>61.35</td></tr><tr><td>AmazonCounterfactualClassification</td><td>ja</td><td>68.76</td><td>76.42</td><td>63.45</td><td>69.79</td><td>58.23</td></tr><tr><td>AmazonReviewsClassification</td><td>de</td><td>31.07</td><td>39.92</td><td>35.91</td><td>39.52</td><td>29.70</td></tr><tr><td>AmazonReviewsClassification</td><td>es</td><td>32.72</td><td>39.39</td><td>37.49</td><td>39.99</td><td>35.97</td></tr><tr><td>AmazonReviewsClassification</td><td>fr</td><td>31.12</td><td>38.52</td><td>35.30</td><td>39.00</td><td>35.92</td></tr><tr><td>AmazonReviewsClassification</td><td>ja</td><td>28.94</td><td>36.44</td><td>33.24</td><td>36.64</td><td>27.64</td></tr><tr><td>AmazonReviewsClassification</td><td>zh</td><td>30.89</td><td>36.45</td><td>35.26</td><td>37.74</td><td>32.63</td></tr><tr><td>MassiveIntentClassification</td><td>af</td><td>38.01</td><td>56.12</td><td>45.88</td><td>52.32</td><td>47.85</td></tr><tr><td>MassiveIntentClassification</td><td>am</td><td>12.70</td><td>55.71</td><td>36.75</td><td>41.55</td><td>33.30</td></tr><tr><td>MassiveIntentClassification</td><td>ar</td><td>37.16</td><td>50.86</td><td>45.14</td><td>51.43</td><td>59.25</td></tr><tr><td>MassiveIntentClassification</td><td>az</td><td>19.98</td><td>58.97</td><td>47.42</td><td>56.98</td><td>45.24</td></tr><tr><td>MassiveIntentClassification</td><td>bn</td><td>42.51</td><td>58.22</td><td>35.34</td><td>48.79</td><td>61.59</td></tr><tr><td>MassiveIntentClassification</td><td>cy</td><td>17.33</td><td>50.16</td><td>26.12</td><td>27.87</td><td>44.92</td></tr><tr><td>MassiveIntentClassification</td><td>da</td><td>45.61</td><td>58.25</td><td>57.73</td><td>62.77</td><td>51.23</td></tr><tr><td>MassiveIntentClassification</td><td>de</td><td>44.79</td><td>56.21</td><td>50.71</td><td>59.57</td><td>56.10</td></tr><tr><td>MassiveIntentClassification</td><td>el</td><td>46.71</td><td>57.03</td><td>58.70</td><td>62.62</td><td>46.13</td></tr><tr><td>MassiveIntentClassification</td><td>es</td><td>45.44</td><td>58.32</td><td>59.66</td><td>64.43</td><td>66.35</td></tr><tr><td>MassiveIntentClassification</td><td>fa</td><td>45.01</td><td>62.33</td><td>61.02</td><td>65.34</td><td>51.20</td></tr><tr><td>MassiveIntentClassification</td><td>fi</td><td>45.94</td><td>60.12</td><td>57.54</td><td>62.28</td><td>45.33</td></tr><tr><td>MassiveIntentClassification</td><td>fr</td><td>46.13</td><td>60.47</td><td>60.25</td><td>64.82</td><td>66.95</td></tr><tr><td>MassiveIntentClassification</td><td>he</td><td>42.55</td><td>56.55</td><td>52.51</td><td>58.21</td><td>43.18</td></tr><tr><td>MassiveIntentClassification</td><td>hi</td><td>40.20</td><td>59.40</td><td>58.37</td><td>62.77</td><td>63.54</td></tr><tr><td>MassiveIntentClassification</td><td>hu</td><td>42.77</td><td>59.52</td><td>60.41</td><td>63.87</td><td>44.73</td></tr><tr><td>MassiveIntentClassification</td><td>hy</td><td>28.07</td><td>56.20</td><td>51.60</td><td>57.74</td><td>38.13</td></tr><tr><td>MassiveIntentClassification</td><td>id</td><td>45.81</td><td>61.12</td><td>59.85</td><td>65.43</td><td>64.06</td></tr><tr><td>MassiveIntentClassification</td><td>is</td><td>39.86</td><td>54.90</td><td>30.83</td><td>37.05</td><td>44.35</td></tr><tr><td>MassiveIntentClassification</td><td>it</td><td>48.25</td><td>59.83</td><td>59.61</td><td>64.68</td><td>60.77</td></tr><tr><td>MassiveIntentClassification</td><td>ja</td><td>45.30</td><td>63.11</td><td>60.89</td><td>63.74</td><td>61.22</td></tr><tr><td>MassiveIntentClassification</td><td>jv</td><td>24.30</td><td>50.98</td><td>32.37</td><td>36.49</td><td>50.94</td></tr><tr><td>MassiveIntentClassification</td><td>ka</td><td>22.70</td><td>48.35</td><td>43.03</td><td>49.85</td><td>33.84</td></tr><tr><td>MassiveIntentClassification</td><td>km</td><td>22.48</td><td>48.55</td><td>40.04</td><td>45.47</td><td>37.34</td></tr><tr><td>MassiveIntentClassification</td><td>kn</td><td>4.32</td><td>56.24</td><td>40.98</td><td>50.63</td><td>53.54</td></tr><tr><td>MassiveIntentClassification</td><td>ko</td><td>44.26</td><td>60.99</td><td>50.30</td><td>61.82</td><td>53.36</td></tr><tr><td>MassiveIntentClassification</td><td>lv</td><td>39.75</td><td>57.10</td><td>54.68</td><td>61.29</td><td>46.50</td></tr><tr><td>MassiveIntentClassification</td><td>ml</td><td>41.33</td><td>57.91</td><td>42.41</td><td>54.34</td><td>58.27</td></tr><tr><td>MassiveIntentClassification</td><td>mn</td><td>16.20</td><td>58.50</td><td>51.77</td><td>56.59</td><td>40.28</td></tr><tr><td>MassiveIntentClassification</td><td>ms</td><td>43.23</td><td>58.60</td><td>54.76</td><td>60.70</td><td>59.65</td></tr><tr><td>MassiveIntentClassification</td><td>my</td><td>25.37</td><td>57.35</td><td>52.01</td><td>57.09</td><td>37.42</td></tr><tr><td>MassiveIntentClassification</td><td>nb</td><td>37.74</td><td>57.91</td><td>55.50</td><td>62.60</td><td>49.41</td></tr><tr><td>MassiveIntentClassification</td><td>nl</td><td>45.00</td><td>59.37</td><td>59.51</td><td>63.57</td><td>52.09</td></tr><tr><td>MassiveIntentClassification</td><td>pl</td><td>44.99</td><td>59.71</td><td>59.43</td><td>64.30</td><td>50.48</td></tr><tr><td>MassiveIntentClassification</td><td>pt</td><td>48.55</td><td>60.16</td><td>61.27</td><td>64.89</td><td>66.69</td></tr><tr><td>MassiveIntentClassification</td><td>ro</td><td>44.30</td><td>57.92</td><td>58.39</td><td>62.80</td><td>50.53</td></tr><tr><td>MassiveIntentClassification</td><td>ru</td><td>44.29</td><td>60.67</td><td>59.04</td><td>63.26</td><td>58.32</td></tr><tr><td>MassiveIntentClassification</td><td>sl</td><td>44.72</td><td>59.37</td><td>57.36</td><td>63.51</td><td>47.74</td></tr><tr><td>MassiveIntentClassification</td><td>sq</td><td>46.12</td><td>58.03</td><td>56.59</td><td>62.49</td><td>48.94</td></tr><tr><td>MassiveIntentClassification</td><td>sv</td><td>45.95</td><td>59.66</td><td>59.43</td><td>64.73</td><td>50.79</td></tr><tr><td>MassiveIntentClassification</td><td>sw</td><td>31.89</td><td>51.62</td><td>29.57</td><td>31.95</td><td>49.81</td></tr><tr><td>MassiveIntentClassification</td><td>ta</td><td>29.63</td><td>55.04</td><td>36.77</td><td>50.17</td><td>56.40</td></tr><tr><td>MassiveIntentClassification</td><td>te</td><td>36.03</td><td>58.32</td><td>40.72</td><td>52.82</td><td>54.71</td></tr><tr><td>MassiveIntentClassification</td><td>th</td><td>43.39</td><td>56.58</td><td>58.97</td><td>61.11</td><td>44.43</td></tr><tr><td>MassiveIntentClassification</td><td>tl</td><td>29.73</td><td>55.28</td><td>33.67</td><td>38.83</td><td>50.21</td></tr><tr><td>MassiveIntentClassification</td><td>tr</td><td>43.93</td><td>60.91</td><td>59.90</td><td>64.54</td><td>46.56</td></tr><tr><td>MassiveIntentClassification</td><td>ur</td><td>26.11</td><td>56.70</td><td>52.80</td><td>56.37</td><td>56.75</td></tr><tr><td>MassiveIntentClassification</td><td>vi</td><td>44.33</td><td>56.67</td><td>56.61</td><td>59.68</td><td>64.53</td></tr><tr><td>MassiveIntentClassification</td><td>zh-CN</td><td>40.62</td><td>63.86</td><td>61.99</td><td>65.33</td><td>67.07</td></tr><tr><td>MassiveIntentClassification</td><td>zh-TW</td><td>32.93</td><td>59.51</td><td>58.77</td><td>62.35</td><td>62.89</td></tr><tr><td>MassiveScenarioClassification</td><td>af</td><td>47.10</td><td>63.39</td><td>53.64</td><td>59.67</td><td>51.47</td></tr><tr><td>MassiveScenarioClassification</td><td>am</td><td>17.70</td><td>62.02</td><td>41.89</td><td>48.97</td><td>34.87</td></tr><tr><td>MassiveScenarioClassification</td><td>ar</td><td>45.21</td><td>57.72</td><td>51.74</td><td>57.78</td><td>65.21</td></tr><tr><td>MassiveScenarioClassification</td><td>az</td><td>28.21</td><td>63.48</td><td>52.06</td><td>61.53</td><td>45.58</td></tr><tr><td>MassiveScenarioClassification</td><td>bn</td><td>50.52</td><td>61.84</td><td>41.17</td><td>54.53</td><td>67.30</td></tr><tr><td>MassiveScenarioClassification</td><td>cy</td><td>22.58</td><td>56.13</td><td>31.72</td><td>35.26</td><td>46.29</td></tr><tr><td>MassiveScenarioClassification</td><td>da</td><td>54.87</td><td>65.24</td><td>66.87</td><td>71.00</td><td>53.52</td></tr><tr><td>MassiveScenarioClassification</td><td>de</td><td>54.34</td><td>62.39</td><td>57.40</td><td>67.34</td><td>61.74</td></tr><tr><td>MassiveScenarioClassification</td><td>el</td><td>55.47</td><td>64.58</td><td>66.14</td><td>68.81</td><td>48.96</td></tr><tr><td>MassiveScenarioClassification</td><td>es</td><td>52.77</td><td>63.61</td><td>65.04</td><td>70.42</td><td>73.34</td></tr><tr><td>MassiveScenarioClassification</td><td>fa</td><td>52.50</td><td>67.46</td><td>65.86</td><td>69.88</td><td>53.17</td></tr><tr><td>MassiveScenarioClassification</td><td>fi</td><td>52.63</td><td>64.58</td><td>63.75</td><td>67.60</td><td>44.69</td></tr><tr><td>MassiveScenarioClassification</td><td>fr</td><td>54.32</td><td>65.10</td><td>66.06</td><td>70.69</td><td>72.91</td></tr><tr><td>MassiveScenarioClassification</td><td>he</td><td>52.41</td><td>63.53</td><td>59.20</td><td>65.16</td><td>43.10</td></tr><tr><td>MassiveScenarioClassification</td><td>hi</td><td>47.37</td><td>64.40</td><td>65.21</td><td>67.92</td><td>69.27</td></tr><tr><td>MassiveScenarioClassification</td><td>hu</td><td>53.43</td><td>65.82</td><td>66.56</td><td>70.30</td><td>45.16</td></tr><tr><td>MassiveScenarioClassification</td><td>hy</td><td>33.57</td><td>61.25</td><td>56.11</td><td>63.02</td><td>38.73</td></tr><tr><td>MassiveScenarioClassification</td><td>id</td><td>54.38</td><td>65.84</td><td>66.16</td><td>70.73</td><td>70.13</td></tr><tr><td>MassiveScenarioClassification</td><td>is</td><td>49.78</td><td>61.94</td><td>37.52</td><td>44.16</td><td>44.21</td></tr><tr><td>MassiveScenarioClassification</td><td>it</td><td>54.84</td><td>64.09</td><td>65.00</td><td>69.73</td><td>65.57</td></tr><tr><td>MassiveScenarioClassification</td><td>ja</td><td>54.12</td><td>67.72</td><td>66.50</td><td>69.69</td><td>65.76</td></tr><tr><td>MassiveScenarioClassification</td><td>jv</td><td>32.71</td><td>58.29</td><td>38.60</td><td>44.20</td><td>54.79</td></tr><tr><td>MassiveScenarioClassification</td><td>ka</td><td>26.92</td><td>53.38</td><td>50.66</td><td>57.30</td><td>32.99</td></tr><tr><td>MassiveScenarioClassification</td><td>km</td><td>27.23</td><td>56.18</td><td>46.96</td><td>53.14</td><td>39.34</td></tr><tr><td>MassiveScenarioClassification</td><td>kn</td><td>10.06</td><td>61.74</td><td>45.73</td><td>56.08</td><td>60.50</td></tr><tr><td>MassiveScenarioClassification</td><td>ko</td><td>52.01</td><td>67.26</td><td>55.66</td><td>68.52</td><td>55.69</td></tr><tr><td>MassiveScenarioClassification</td><td>lv</td><td>44.82</td><td>61.87</td><td>59.80</td><td>66.28</td><td>44.35</td></tr><tr><td>MassiveScenarioClassification</td><td>ml</td><td>49.10</td><td>62.26</td><td>47.69</td><td>60.13</td><td>65.53</td></tr><tr><td>MassiveScenarioClassification</td><td>mn</td><td>21.51</td><td>62.60</td><td>57.07</td><td>60.85</td><td>38.72</td></tr><tr><td>MassiveScenarioClassification</td><td>ms</td><td>53.60</td><td>65.63</td><td>61.71</td><td>65.81</td><td>64.99</td></tr><tr><td>MassiveScenarioClassification</td><td>my</td><td>29.72</td><td>62.94</td><td>59.10</td><td>63.03</td><td>36.84</td></tr><tr><td>MassiveScenarioClassification</td><td>nb</td><td>43.90</td><td>64.29</td><td>64.25</td><td>70.24</td><td>51.80</td></tr><tr><td>MassiveScenarioClassification</td><td>nl</td><td>53.33</td><td>65.16</td><td>65.52</td><td>70.37</td><td>56.32</td></tr><tr><td>MassiveScenarioClassification</td><td>pl</td><td>52.92</td><td>64.56</td><td>65.04</td><td>68.99</td><td>49.98</td></tr><tr><td>MassiveScenarioClassification</td><td>pt</td><td>53.41</td><td>63.28</td><td>65.79</td><td>70.09</td><td>71.46</td></tr><tr><td>MassiveScenarioClassification</td><td>ro</td><td>50.48</td><td>62.41</td><td>64.17</td><td>67.95</td><td>53.69</td></tr><tr><td>MassiveScenarioClassification</td><td>ru</td><td>51.84</td><td>65.25</td><td>65.24</td><td>69.92</td><td>61.60</td></tr><tr><td>MassiveScenarioClassification</td><td>sl</td><td>51.29</td><td>64.25</td><td>64.01</td><td>70.81</td><td>48.04</td></tr><tr><td>MassiveScenarioClassification</td><td>sq</td><td>55.65</td><td>64.54</td><td>64.31</td><td>69.63</td><td>50.06</td></tr><tr><td>MassiveScenarioClassification</td><td>sv</td><td>54.64</td><td>66.01</td><td>67.14</td><td>71.60</td><td>51.73</td></tr><tr><td>MassiveScenarioClassification</td><td>sw</td><td>42.04</td><td>58.36</td><td>34.86</td><td>37.29</td><td>54.22</td></tr><tr><td>MassiveScenarioClassification</td><td>ta</td><td>36.72</td><td>59.08</td><td>42.62</td><td>55.96</td><td>62.77</td></tr><tr><td>MassiveScenarioClassification</td><td>te</td><td>42.08</td><td>64.13</td><td>46.46</td><td>58.81</td><td>62.59</td></tr><tr><td>MassiveScenarioClassification</td><td>th</td><td>52.15</td><td>64.34</td><td>67.01</td><td>69.44</td><td>45.18</td></tr><tr><td>MassiveScenarioClassification</td><td>tl</td><td>37.34</td><td>60.23</td><td>37.37</td><td>43.99</td><td>52.06</td></tr><tr><td>MassiveScenarioClassification</td><td>tr</td><td>52.56</td><td>65.43</td><td>66.55</td><td>70.4</td><td>47.21</td></tr><tr><td>MassiveScenarioClassification</td><td>ur</td><td>32.60</td><td>61.52</td><td>60.43</td><td>62.9</td><td>64.26</td></tr><tr><td>MassiveScenarioClassification</td><td>vi</td><td>50.97</td><td>61.05</td><td>60.72</td><td>65.71</td><td>70.61</td></tr><tr><td>MassiveScenarioClassification</td><td>zh-CN</td><td>50.22</td><td>70.85</td><td>67.44</td><td>71.23</td><td>73.95</td></tr><tr><td>MassiveScenarioClassification</td><td>zh-TW</td><td>42.32</td><td>67.08</td><td>65.70</td><td>68.73</td><td>70.30</td></tr><tr><td>MTOPDomainClassification</td><td>de</td><td>74.08</td><td>86.95</td><td>79.20</td><td>85.73</td><td>82.05</td></tr><tr><td>MTOPDomainClassification</td><td>es</td><td>73.47</td><td>84.07</td><td>83.04</td><td>86.96</td><td>93.55</td></tr><tr><td>MTOPDomainClassification</td><td>fr</td><td>72.26</td><td>84.14</td><td>78.63</td><td>81.21</td><td>90.98</td></tr><tr><td>MTOPDomainClassification</td><td>hi</td><td>72.95</td><td>85.11</td><td>81.36</td><td>84.76</td><td>89.33</td></tr><tr><td>MTOPDomainClassification</td><td>th</td><td>72.68</td><td>81.24</td><td>79.99</td><td>82.51</td><td>60.49</td></tr><tr><td>MTOPIntentClassification</td><td>de</td><td>51.62</td><td>63.42</td><td>54.23</td><td>61.27</td><td>61.92</td></tr><tr><td>MTOPIntentClassification</td><td>es</td><td>52.75</td><td>64.44</td><td>60.28</td><td>66.59</td><td>74.49</td></tr><tr><td>MTOPIntentClassification</td><td>fr</td><td>50.12</td><td>62.01</td><td>54.05</td><td>59.76</td><td>69.12</td></tr><tr><td>MTOPIntentClassification</td><td>hi</td><td>45.55</td><td>62.58</td><td>59.90</td><td>62.37</td><td>64.85</td></tr><tr><td>MTOPIntentClassification</td><td>th</td><td>50.07</td><td>64.61</td><td>61.96</td><td>64.80</td><td>49.36</td></tr></table>

Table 13: Multilingual classification results. Scores are accuracy.

<table><tr><td>Dataset</td><td>Language</td><td>Komninos</td><td>LASER2</td><td>LaBSE</td><td>MiniLM-L12-multilingual</td><td>MPNet-multilingual</td><td>SGPT-BLOOM-7.1B-msmarco</td></tr><tr><td>STS17</td><td>ko-ko</td><td>2.54</td><td>70.52</td><td>71.32</td><td>77.03</td><td>83.41</td><td>66.89</td></tr><tr><td>STS17</td><td>ar-ar</td><td>13.78</td><td>67.47</td><td>69.07</td><td>79.16</td><td>79.10</td><td>76.42</td></tr><tr><td>STS17</td><td>en-ar</td><td>9.08</td><td>65.05</td><td>74.51</td><td>81.22</td><td>80.85</td><td>78.07</td></tr><tr><td>STS17</td><td>en-de</td><td>-3.11</td><td>66.66</td><td>73.85</td><td>84.22</td><td>83.28</td><td>59.10</td></tr><tr><td>STS17</td><td>en-tr</td><td>-0.45</td><td>70.05</td><td>72.07</td><td>76.74</td><td>74.90</td><td>11.80</td></tr><tr><td>STS17</td><td>es-en</td><td>-8.18</td><td>55.30</td><td>65.71</td><td>84.44</td><td>86.11</td><td>78.22</td></tr><tr><td>STS17</td><td>es-es</td><td>48.23</td><td>79.67</td><td>80.83</td><td>85.56</td><td>85.14</td><td>86.00</td></tr><tr><td>STS17</td><td>fr-en</td><td>5.81</td><td>70.82</td><td>76.98</td><td>76.59</td><td>81.17</td><td>80.46</td></tr><tr><td>STS17</td><td>it-en</td><td>3.64</td><td>70.98</td><td>76.99</td><td>82.35</td><td>84.24</td><td>51.58</td></tr><tr><td>STS17</td><td>nl-en</td><td>-0.44</td><td>68.12</td><td>75.22</td><td>81.71</td><td>82.51</td><td>45.85</td></tr><tr><td>STS22</td><td>de</td><td>33.04</td><td>25.69</td><td>48.58</td><td>44.64</td><td>46.70</td><td>30.05</td></tr><tr><td>STS22</td><td>es</td><td>48.53</td><td>54.92</td><td>63.18</td><td>56.56</td><td>59.91</td><td>65.41</td></tr><tr><td>STS22</td><td>pl</td><td>12.47</td><td>18.34</td><td>39.30</td><td>33.74</td><td>33.65</td><td>31.13</td></tr><tr><td>STS22</td><td>tr</td><td>47.38</td><td>36.97</td><td>58.15</td><td>53.39</td><td>56.30</td><td>47.14</td></tr><tr><td>STS22</td><td>ar</td><td>32.42</td><td>42.57</td><td>57.67</td><td>46.2</td><td>52.19</td><td>58.67</td></tr><tr><td>STS22</td><td>ru</td><td>19.44</td><td>39.24</td><td>57.49</td><td>57.08</td><td>58.74</td><td>43.36</td></tr><tr><td>STS22</td><td>zh</td><td>4.78</td><td>49.41</td><td>63.02</td><td>58.75</td><td>61.75</td><td>66.78</td></tr><tr><td>STS22</td><td>fr</td><td>49.43</td><td>58.61</td><td>77.95</td><td>70.55</td><td>74.30</td><td>80.38</td></tr><tr><td>STS22</td><td>de-en</td><td>28.65</td><td>32.35</td><td>50.14</td><td>52.65</td><td>50.81</td><td>51.16</td></tr><tr><td>STS22</td><td>es-en</td><td>26.97</td><td>54.34</td><td>71.86</td><td>67.33</td><td>70.26</td><td>75.06</td></tr><tr><td>STS22</td><td>it</td><td>57.77</td><td>60.31</td><td>72.22</td><td>55.22</td><td>60.65</td><td>65.65</td></tr><tr><td>STS22</td><td>pl-en</td><td>45.55</td><td>53.63</td><td>69.41</td><td>69.02</td><td>73.07</td><td>53.31</td></tr><tr><td>STS22</td><td>zh-en</td><td>14.05</td><td>46.19</td><td>64.02</td><td>65.71</td><td>67.96</td><td>68.45</td></tr><tr><td>STS22</td><td>es-it</td><td>41.10</td><td>42.21</td><td>69.69</td><td>47.67</td><td>53.70</td><td>65.50</td></tr><tr><td>STS22</td><td>de-fr</td><td>14.77</td><td>37.41</td><td>53.28</td><td>51.73</td><td>62.34</td><td>53.28</td></tr><tr><td>STS22</td><td>de-pl</td><td>11.21</td><td>15.67</td><td>58.69</td><td>44.22</td><td>40.53</td><td>43.05</td></tr><tr><td>STS22</td><td>fr-pl</td><td>39.44</td><td>39.44</td><td>61.98</td><td>50.71</td><td>84.52</td><td>28.17</td></tr><tr><td>Average</td><td>mix</td><td>22.14</td><td>51.55</td><td>65.67</td><td>64.23</td><td>67.71</td><td>57.81</td></tr></table>

Table 14: Multilingual STS Results. Scores are Spearman correlations of cosine similarities.