# XOR QA: Cross-lingual Open-Retrieval Question Answering

Akari Asai $^{\clubsuit}$ , Jungo Kasai $^{\clubsuit}$ , Jonathan H. Clark $^{\clubsuit}$ , Kenton Lee $^{\clubsuit}$ , Eunsol Choi $^{\heartsuit}$ , Hannaneh Hajishirzi $^{\clubsuit\spadesuit}$

University of Washington ◆ Google Research

♥The University of Texas at Austin ♠Allen Institute for AI

{akari, jkasai, hannaneh}@cs.washington.edu

{jhclark, kentonl}@google.com, eunsol@cs.utexas.edu

# Abstract

Multilingual question answering tasks typically assume that answers exist in the same language as the question. Yet in practice, many languages face both information scarcity—where languages have few reference articles—and information asymmetry—where questions reference concepts from other cultures. This work extends open-retrieval question answering to a cross-lingual setting enabling questions from one language to be answered via answer content from another language. We construct a large-scale dataset built on 40K information-seeking questions across 7 diverse non-English languages that TYDI QA could not find same-language answers for. Based on this dataset, we introduce a task framework, called Cross-lingual Open-Retrieval Question Answering (XOR QA), that consists of three new tasks involving cross-lingual document retrieval from multilingual and English resources. We establish baselines with state-of-the-art machine translation systems and cross-lingual pretrained models. Experimental results suggest that XOR QA is a challenging task that will facilitate the development of novel techniques for multilingual question answering. Our data and code are available at https://nlp.cs.washington.edu/xorqa/.

# 1 Introduction

Information-seeking questions—questions from people who are actually looking for an answer—have been increasingly studied in question answering (QA) research. Fulfilling these information needs has led the research community to look further for answers: beyond paragraphs and articles toward performing open retrieval $^{1}$ on large-scale document collections (Chen and Yih, 2020). Yet

![](images/4667cf9755f3d2b90af16ce3575d66158a24e25f3a6d9939442a81dfd8435c65.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["ロング・ポールの学部時代の専攻は？[Japanese"] (What did Ron Paul major in during undergraduate?)] --> B["Multilingual document collections (Wikipedias)"]
    B --> C["ロング・ポール (ja.wikipedia)"]
    B --> D["Ron Paul (en.wikipedia)"]
    C --> E["高校卒業後はゲティスバーグ大学へ進学。(After high school, he went to Gettysburg College.)"]
    D --> F["Paul went to Gettysburg College, where he was a member of the Lambda Chi Alpha fraternity. He graduated with a B.S. degree in Biology in 1957."]
    G["生物学 (Biology)"] --> D
```
</details>

Figure 1: Overview of XOR QA. Given a question in $L_{i}$ , the model finds an answer in either English or $L_{i}$ Wikipedia and returns an answer in English or $L_{i}$ . $L_{i}$ is one of the 7 typologically diverse languages.

the bulk of this work has been exclusively on English. In this paper, we bring together for the first time information-seeking questions, open-retrieval QA, and multilingual QA to create a multilingual open-retrieval QA dataset that enables cross-lingual answer retrieval.

While multilingual open QA systems would benefit the many speakers of non-English languages, there are several pitfalls in designing such a dataset. First, a multilingual QA dataset should include questions from non-English native speakers to represent real-world applications. Questions in most recent multilingual QA datasets (Lewis et al., 2020; Artetxe et al., 2020; Longpre et al., 2020) are translated from English, which leads to English-centric questions such as questions about American sports, cultures and politics. Second, it is important to support retrieving answers in languages other than the original language due to information scarcity of low-resource languages (Miniwatts Marketing Group, 2011). Moreover, questions strongly related to entities from other cultures are less likely to have answer content in the questioner's language

due to cultural bias (information asymmetry, Callahan and Herring, 2011). For example, Fig. 1 shows that the Japanese Wikipedia article of an American politician, Ron Paul, does not have information about his college degree perhaps because Japanese Wikipedia editors are less interested in specific educational backgrounds of American politicians.

In this paper, we introduce the task of cross-lingual open-retrieval question answering (XOR QA) which aims at answering multilingual questions from non-English native speakers given multilingual resources. To support research in this area, we construct a dataset (called XOR-TYDI QA) of 40k annotated questions and answers across 7 typologically diverse languages. Questions in our dataset are inherited from TYDI QA (Clark et al., 2020), which are written by native speakers and are originally unanswerable due to the information scarcity or asymmetry issues. XOR-TYDI QA is the first large-scale cross-lingual open-retrieval QA dataset that consists of information-seeking questions from native speakers and multilingual reference documents.

XOR-TYDI QA is constructed with an annotation pipeline that allows for cross-lingual retrieval from large-scale Wikipedia corpora ( $§2$ ). Unanswerable questions in TYDI QA are first translated into English by professional translators. Then, annotators find answers to translated queries given English Wikipedia using our new model-in-the-loop annotation framework that reduces annotation errors. Finally, answers are verified and translated back to the target languages.

Building on the dataset, we introduce three new tasks in the order of increasing complexity ( $§3$ ). In XOR-RETRIEVE, a system retrieves English Wikipedia paragraphs with sufficient information to answer the question posed in the target language. XOR-ENGLISHSPAN takes one step further and finds a minimal answer span from the retrieved English paragraphs. Finally, XOR-FULL expects a system to generate an answer end to end in the target language by consulting both English and the target language's Wikipedia. XOR-FULL is our ultimate goal, and the first two tasks enable researchers to diagnose where their models fail and develop under less coding efforts and resources.

We provide baselines that extend state-of-the-art open-retrieval QA systems (Asai et al., 2020; Karpukhin et al., 2020) to our multilingual retrieval setting. Our best baseline achieves an average of 18.7 F1 points on XOR-FULL. This result indicates that XOR-TYDI QA poses unique challenges to tackle toward building a real-world open-retrieval QA system for diverse languages. We expect that our dataset opens up new challenges to make progress in multilingual representation learning.

# 2 The XOR-TYDI QA Dataset

Our XOR-TYDI QA dataset comprises questions inherited from TYDI QA (Clark et al., 2020) and answers augmented with our annotation process across 7 typologically diverse languages. We focus on cross-lingual retrieval from English Wikipedia because in our preliminary investigation we were able to find answers to a majority of the questions from resource-rich English Wikipedia, and native speakers with much annotation experience were readily available via crowdsourcing in English.

# 2.1 XOR-TYDI QA Collection

Our annotation pipeline proceeds with four steps: 1) collection of questions from TYDI QA without a same-language answer which require cross-lingual reference to answer ( $§2.1.1$ ); 2) question translation from a target language to the pivot language of English where the missing information may exist ( $§2.1.2$ ); 3) answer retrieval in the pivot language given a set of candidate documents ( $§2.1.3$ ); 4) answer verification and translation from the pivot language back to the original language ( $§2.1.4$ ). Fig. 2 shows an overview of the pipeline.

# 2.1.1 Question Selection

Our questions are collected from unanswerable questions in TYDI QA. A question is unanswerable in TYDI QA if an annotator cannot select a passage answer (a paragraph in the article that contains an answer). We randomly sample 5,000 questions without any passage answer annotations (unanswerable questions) from the TYDI QA training data, and split them into training (4,500) and development (500) sets. We use the development data from TYDI QA as our test data, since the TYDI QA's original test data is not publicly available. $^{2}$ We choose 7 languages with varying amounts of Wikipedia data out of the 10 non-English languages based on the cost and availability

![](images/9f25b9d7d5ec124a8cb5c1be82d082126e92d4a93c9eab73d694ac8b5584e709.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1. Question Selection"] --> B["TyDiQA Cross-lingual (Q_L, No answer)"]
    C["2. Question Translation Q_L → Q_en"] --> D["In-language (Q_L, A_L)"]
    E["3. Answer Retrieval in English (Q_en, P_en)"] --> F["Paragraph ranking"]
    G["4. Answer Translation (Q_en, P_en, A_en → A_L)"] --> H["XOR-TyDiQA"]
    
    B --> I["Word: 罗ルの学部時代の専攻は何ですか？"]
    D --> J["What did Ron Paul major in during undergraduate?"]
    F --> K["Paragraph retrieval"]
    K --> L["Search Engine"]
    K --> M["Top English Wikipedia articles"]
    K --> N["Ron Paul is an American politician ..."]
    G --> O["Answer Annotation @Mechanical turk"]
    H --> P["What did Ron Paul major in during undergraduate?"]
    P --> Q["Paul went to Gettysburg College ... He graduated with a B.S. degree in Biology in 1957."]
    H --> R["Answer verification"]
    H --> S["Human translation"]
    H --> T["生物学"]
```
</details>

Figure 2: Overview of the annotation process for XOR-TYDI QA.

of translators: $^{3}$ Arabic, Bengali, Finnish, Japanese, Korean, Russian and Telugu.

# 2.1.2 Question Translation

We use a professional translation service, Gengo, $^{4}$ to translate all collected questions into English. Since named entities are crucial for QA, we instruct translators to carefully translate them by searching for common English translations from English Wikipedia or other external sources. We perform manual quality assessment by native speakers on 50 translation samples, finding that more than 95% are correct. Note that while these translations are a part of the annotation procedure (due to the inherently cross-lingual nature of this task), they are not provided to models during evaluation.

# 2.1.3 Answer Retrieval in English

We use Amazon Mechanical Turk to retrieve answers to translated English questions given English Wikipedia articles. Annotators are instructed to select passage answers (gold paragraphs) and minimal answer spans as in Clark et al. (2020).

To annotate answers to information-seeking queries, previous work first identifies relevant Wikipedia articles using Google Search, and then annotators attempt to find answers there. Asai and Choi (2020) show that in information-seeking QA datasets many questions were annotated as “unanswerable” due to two systematic errors: retrieval error where the search engine failed to retrieve a relevant article and answer annotation error where the annotator overlooks answer content. Importantly, these two types of annotation errors present a tradeoff: if we retrieve many articles, retrieval errors will be reduced at the expense of answer annotation errors because annotators have to find answer context among many candidate articles.

Collaborative model-in-the-loop. To find a middle ground in the tradeoff, we introduce a collaborative model-in-the-loop framework that uses Google Search and a state-of-the-art paragraph ranker. We first run Google Search to retrieve as many as top 10 Wikipedia articles, resulting in 387 paragraphs per question on average. We score them with Path Retriever (Asai et al., 2020) and present the five highest scoring paragraphs. Annotators are asked to skim these five paragraphs first; if they cannot find any answer content, they are asked to read the rest of the paragraphs, where the Wikipedia section headings guide their reading. To incentivize workers to find answers beyond the pre-selected ones, we carefully communicate with workers and send additional rewards to annotators who actively read the rest of the paragraphs and find answers for questions that other annotators may overlook. We found about 70% of the answers from the 5 paragraphs and 30% from the rest of the paragraphs in the top 10 articles. This means that while our paragraph ranking was effective, the annotators did not fully rely on it, thereby mitigating the influence of the passage ranking model on the dataset. See Appendix §B.1 for annotation interface details.

Quality control for QA annotation. We first recruit MTurkers with a high approval rate ( $\geq 96\%$ ) located in English-speaking countries, and all workers first annotate the same qualification batch. We assess the quality of those submissions and select high-quality annotators. Consequently, 40 out of more than 200 workers were qualified and 24 workers annotated most of our data. More details are in Appendix B.3.

<table><tr><td>%</td><td>Ar</td><td>Bn</td><td>Fi</td><td>Ja</td><td>Ko</td><td>Ru</td><td>Te</td><td>All</td></tr><tr><td>TYDI QA</td><td>82</td><td>42</td><td>57</td><td>50</td><td>29</td><td>69</td><td>28</td><td>50</td></tr><tr><td>XOR-TYDI QA</td><td>92</td><td>82</td><td>83</td><td>77</td><td>68</td><td>83</td><td>44</td><td>72</td></tr><tr><td>Improvement</td><td>10</td><td>40</td><td>26</td><td>27</td><td>39</td><td>14</td><td>16</td><td>22</td></tr></table>

Table 1: Percentage of the questions with short answers (answerable questions) in the original TYDI QA dataset (dev) and XOR-TYDI QA. The third row (Improvement) represents the percentage of the questions that become answerable by searching the English Wikipedia articles.

# 2.1.4 Answer Verification and Translation

We verify the annotated answers and translate those answers back to the target languages (cross-lingual data). Finally, we mix the annotated cross-lingual data with the same-language data from TYDI QA to reflect the actual question distributions from native speakers (in-language data).

Answer verification. We trained undergraduate students who are native English speakers to verify the annotated paragraphs and short answers. Only 8% of the answers were marked as incorrect through the verification phase and were later corrected by our pool of high-quality crowdworkers who yielded less than 1% annotation error.

Answer translation. We again use Gengo to translate answers from English back to the original languages. We give translators further instructions to normalize answers such that they are consistent with answers in TYDI QA. For example, some languages use their own unique set of numerals rather than Arabic numerals to represent numeric answers (e.g., Bengali numerals, Chinese numerals in Japanese text). The details of the answer translation process are described in Appendix §B.4. Note that because of the cost of answer translations, we conduct this answer translation process for evaluation sets only.

# 2.2 The XOR-TYDI QA Corpus

Dataset statistics. $^{5}$ Table 1 shows the percentages of the questions annotated with short answers in the original TYDI QA and our XOR-TYDI QA, and Table 2 shows statistics of XOR-TYDI QA. As seen in Table 1, cross-lingual retrieval significantly increases the answer coverage in all languages by up to 40% (Bengali), and consequently we found answers for more than 50% of the original information-seeking questions in 6 out of the 7 languages. $^{6}$ This result confirms the effectiveness of searching multilingual document collections to improve the answer coverage. Detailed statistics of the numbers of long answers, short answers, and unanswered questions are in Appendix §B.5. We also release the 30k manually translated questions for our training set, which could be used to train multilingual models or machine translation models.

<table><tr><td rowspan="2"></td><td colspan="3">Cross-lingual</td><td colspan="3">In-language</td></tr><tr><td>Train</td><td>Dev</td><td>Test</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>Ar</td><td>2,574</td><td>350</td><td>137</td><td>15,828</td><td>358</td><td>1132</td></tr><tr><td>Bn</td><td>2,582</td><td>312</td><td>128</td><td>2,428</td><td>115</td><td>139</td></tr><tr><td>Fi</td><td>2,088</td><td>360</td><td>530</td><td>7,680</td><td>255</td><td>1,197</td></tr><tr><td>Ja</td><td>2,288</td><td>296</td><td>449</td><td>5,527</td><td>137</td><td>867</td></tr><tr><td>Ko</td><td>2,469</td><td>299</td><td>646</td><td>1,856</td><td>72</td><td>505</td></tr><tr><td>Ru</td><td>1,941</td><td>255</td><td>235</td><td>7,349</td><td>313</td><td>1,125</td></tr><tr><td>Te</td><td>1,308</td><td>238</td><td>374</td><td>5,451</td><td>113</td><td>712</td></tr></table>

Table 2: Dataset size of the XOR-TYDI QA corpus (answered data). Cross-lingual data comes from our reannotated questions that did not originally have same-language answers in TYDI QA. In-language data are taken directly from answerable questions in TYDI QA.

Qualitative examples. Table 3 illustrates that finding relevant articles from multilingual document collections is important to answer questions asked by users with diverse linguistic and cultural backgrounds. The first question is unanswerable in Korean Wikipedia, but there is a clear description about who was the prime minister of France at the time in English Wikipedia. The second example shows English Wikipedia sometimes contains rich information about a target language-specific topic (e.g., economy in Krasnodar, a city in Russia). Those examples demonstrate the effectiveness of searching for answers in another language with more abundant knowledge sources. In the last question of Table 3, on the other hand, only the Wikipedia of the target language can provide the answer. XOR QA allows for both retrieval paths.

Comparison with other datasets. Table 4 compares XOR-TYDI QA and existing multilingual QA datasets. XOR-TYDI QA has three key properties that are distinct from these QA benchmarks. First, since all questions are inherited from TYDI QA, they are information-seeking questions written by

<table><tr><td>L</td><td>Original Question:  $Q_{L}$  ( $Q_{en}$ )</td><td>Passage Answer:  $P_{en}$  or  $P_{L}$ </td><td>Minimal Answer in English:  $A_{en}$ </td><td>Final Answer:  $A_{L}$ </td></tr><tr><td>Ko</td><td>1993년 프랑스 총리는 누구인가요? (Who was the French Prime Minister in 1993?)</td><td>Mayor of Neuilly-sur-Seine from 1983 to 2002, he was Minister of the Budget under Prime Minister Édouard Balladur (1993–1995).</td><td>Édouard Balladur</td><td>에두아르발라뒤르</td></tr><tr><td>Ru</td><td>Какая средняя зарплата в Краснодаре на сегодняшний день? (What is the average wage in Krasnodar?)</td><td>Krasnodar has the lowest unemployment rate among the cities of the Southern Federal District at 0.3% of the total working-age population. In addition, Krasnodar holds the first place in terms of highest average salary—21,742 rubles per capita.</td><td>21,742 rubles</td><td>21,742 рубля</td></tr><tr><td>Ja</td><td>速水堅曹はとこで製糸技術を学んだ? (Where did Kenso Hayami learn the silk-reeling technique?)</td><td>藩営前橋製糸所を前橋に開設。カスパル・ミュラーから直接、器械製糸技術を学び (he founded Hanei Maebashi Silk Mill and learned instrumental silk reeling techniques directly from Caspal Müller)</td><td>-</td><td>藩営前橋製糸所 (Hanei Maebashi Silk Mill)</td></tr></table>

Table 3: Examples newly annotated for Korean (Ko) and Russian (Ru) questions. The bottom example is an answerable question from TYDI QA for which only Japanese Wikipedia includes the correct answer.

<table><tr><td>Dataset</td><td>Asked by native speakers</td><td>Open-retrieval</td><td>Cross-lingual</td></tr><tr><td>TYDI QA</td><td>√</td><td>✗</td><td>✗</td></tr><tr><td>MLQA</td><td>✗</td><td>✗</td><td>√</td></tr><tr><td>XQuAD</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>MKQA</td><td>✗</td><td>WikiData</td><td>✗</td></tr><tr><td>MLQA-R</td><td>✗</td><td>21k sents</td><td>√</td></tr><tr><td>XQuAD-R</td><td>✗</td><td>13k sents</td><td>√</td></tr><tr><td>XOR-TYDI QA</td><td>√</td><td>Wikipedia</td><td>√</td></tr></table>

Table 4: Comparison with recent multilingual QA datasets. MKQA's answers are aligned to WikiData.

native speakers, and better reflect native speakers' interests and their own linguistic phenomena. This distinguishes XOR-TYDI QA from translation-based datasets such as MLQA (Lewis et al., 2020) and MKQA (Longpre et al., 2020). Second, our dataset requires cross-lingual retrieval unlike other multilingual datasets such as TYDI QA or XQuAD (Artetxe et al., 2020), which focus on same-language QA. Lastly, questions in XOR-TYDI QA require open retrieval from Wikipedia, whereas MLQA-R and XQuAD-R (Roy et al., 2020) limit the search space to matching each question with the predetermined 21k/31k sentences.

# 3 XOR QA Tasks and Baselines

We introduce three new tasks (Fig. 3): XOR-RETRIEVE, XOR-ENGLISHSPAN, and XOR-FULL with our newly collected XOR-TYDI QA dataset and construct strong baselines for each task. XOR-FULL defines our goal of building a multilingual open-retrieval QA system that uses both crosslingual and in-language questions from XOR-TYDI QA. To diagnose where models fail and to allow researchers to use the data with less coding effort or computational resource, we also introduce the first two intermediate tasks that only use the cross-lingual data (Table 2). We denote the target language by $L_{i}$ . We also denote the English Wikipedia collection by $W_{eng}$ and the Wikipedia collection in each target language $L_{i}$ by $W_{i}$ . We experiment with baselines using black-box APIs as a reference, but we encourage the community to use white-box systems so that all experimental details can be understood. Nonetheless, we release the intermediate results from those external APIs to make our results reproducible. All of the white-box system results can be reproduced using our codebase.

# 3.1 XOR-RETRIEVE: Cross-lingual Paragraph Retrieval

Task. Given a question in $L_{i}$ and English Wikipedia $W_{eng}$ , the task is to retrieve English paragraphs for the question. Finding evidence paragraphs from large-scale document collections like Wikipedia is a challenging task, especially when a query and documents are in different languages and systems cannot perform lexical matching.

Evaluation. Different open-retrieval QA models use different units for retrieval. To make fair comparisons across various models, we measure the recall by computing the fraction of the questions for which the minimal answer is contained in the top n tokens selected. We evaluate with $n = 2k$ , 5k: R@2kt and R@5kt (kilo-tokens).

![](images/dc45dec99c9e6a21f574486ea52295689ca5df880e1e6789ba45f2b6b7b562a4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1. XOR-Retrieve"] --> B["Retriever"]
    B --> C["Translation"]
    C --> D["What did Ron Paul major in during undergraduate?"]
    D --> E["Eng. Retriever"]
    E --> F["Paul went to Gettysburg College ... He graduated with a B.S. degree in Biology in 1957"]
    F --> G["Brook of the English Language"]
    G --> H["Brook of the English Language"]
    H --> I["No answer"]
    
    J["2. XOR-EnglishSpan"] --> K["Reader"]
    K --> L["Eng. Reader (Qen, Pen)"]
    K --> M["Multilingual Reader (QL, Pen)"]
    M --> N["(QL, PL)"]
    N --> O["Biology"]
    O --> P["Translation"]
    
    Q["3. XOR-Full"] --> R["生物学"]
    R --> S["No answer"]
    
    T["ロル・ポールの学部時代の専攻は？"] --> U["Monolingual Retriever in L"]
    U --> V["高校卒業後はゲティスバーグにあるゲティスバーグ大学へ進学。"]
    V --> W["En.wikipedia et al."]
    W --> X["unlabeled source image"]
```
</details>

Figure 3: Overview of the tasks and baselines. Each dotted rectangle represents one of the three tasks and surrounds used pipeline modules.

Translate baselines. We first translate queries into English, and then paragraphs are retrieved in a monolingual way. For query translation, we train transformer machine translation (MT) models on publicly available corpora for easy replication. We also run Google's online machine translation service (GMT). This is not completely reproducible as these systems get constantly updated; nor do we know what model and training data they use. We encourage the community to use open MT systems where system details are available. For retrieval, we explore term-based retrieval (BM25, Robertson and Zaragoza 2009), term-based retrieval followed by neural paragraph ranking (Path Retriever, Asai et al. 2020), and end-to-end neural retrieval (DPR, Karpukhin et al. 2020).

Multilingual baselines. Alternatively, we can directly apply a multilingual pretrained model to retrieve paragraphs. We initialize and train a DPR encoder with multilingual BERT to enable multilingual document retrieval (Devlin et al., 2019).

# 3.2 XOR-ENGLISHSPAN: L-to-English Open-Retrieval QA

Task. Given a question in $L_{i}$ and English Wikipedia $W_{eng}$ , a system retrieves paragraphs from $W_{eng}$ and extracts an answer. This task is equivalent to existing open-retrieval QA tasks (Chen et al., 2017), except that the query is not in English. This task involves challenging cross-lingual retrieval and question answering on the $L_{i}$ query and English evidence paragraphs.

Evaluation. We use Exact Match (EM) and F1 over the annotated answer's token set following prior work (Rajpurkar et al., 2016).

Baselines. Our pipeline uses a machine reading model to find a minimal span that answers the question given paragraphs selected from the previous XOR-RETRIEVE step. In particular, for the translate baselines, we use the same approach as state-of-the-art models (Asai et al., 2020; Karpukhin et al., 2020) that jointly predicts a span and a relevance score of each paragraph to the question. For the multilingual baseline where queries are not automatically translated during evaluation, we build a reader model with multilingual BERT.

# 3.3 XOR-FULL: Round Trip

Task. Given a question in target language $L_{i}$ and Wikipedia in both English and $L_{i}$ ( $W_{eng}$ and $W_{i}$ ), a system is required to generate an answer in $L_{i}$ . In this task, a system does not know a priori in which language we can find information that the user is seeking. Note that the XOR-FULL evaluation data includes both cross-lingual and in-language data, while XOR-RETRIEVE and XOR-ENGLISHSPAN only use cross-lingual data during evaluation.

Evaluation. Some answers in XOR-FULL are translated from English so the same spans may not exist in the target language's Wikipedia. For this reason, we use token-level BLEU scores (Papineni et al., 2002) over a ground-truth token set in addition to F1 and EM. The same tokenizer is applied to ground-truth and predicted answers to compute token-level F1 and BLEU. $^{7}$

Baselines. Unlike the previous two tasks, evidence paragraphs can be found both in the target language and English, and a system has to output final answers based on the most plausible paragraphs. In this work, we introduce a simple multi-

lingual baseline that first looks for answers in the target language and then English if no answers are found in the target language. Specifically, we apply monolingual retrieval (i.e., BM25, Google Custom Search) for $W_{i}$ and a multilingual machine reading model based on XLM-RoBERTa (Conneau et al., 2020) to find in-language answers in the target language (monolingual model; the bottom half of Fig. 3). If no answers are found by the monolingual model, we apply an XOR-ENGLISHSPAN baseline and translate English answers into the target language (the top half of Fig. 3).

# 4 Experiments and Analysis

We present results from the baselines discussed above. We find that the three XOR QA tasks present challenges even for the strong models.

# 4.1 Experimental Setup

For training, we first finetune the retrieval and machine reading models with the Natural Questions data (Kwiatkowski et al., 2019) and then further finetune on our XOR-TYDI QA data. For the BM25 retrieval baseline, we use ElasticSearch $^{8}$ to store and search documents using BM25 similarities. For both Path Retriever and DPR, we run the official open-source code. For our MT systems, we train base-sized (large for Russian) autoregressive transformers (Vaswani et al., 2017) on parallel corpora from OPUS (Tiedemann and Nygaard, 2004), MultiUN (Ziemski et al., 2016), or WMT19 (Barrault et al., 2019). All data are encoded into subwords by BPE (Sennrich et al., 2016) or SentencePiece (Kudo and Richardson, 2018). We use the fairseq library (Ott et al., 2019). Additional experimental details and full lists of hyperparameters are available in Appendix §C.

We only evaluate questions having answers and do not give credit to predicting “no answers” as in prior open-retrieval work (Lee et al., 2019). For XOR-RETRIEVE and XOR-ENGLISHSPAN, we use cross-lingual data only and both cross-lingual and in-language data for XOR-FULL.

# 4.2 XOR-RETRIEVE Experiments

Table 5 shows the R@5kt (as defined in §3.1) for different retrieval and query translation systems. $^{9}$ We also report the performance with the human

<table><tr><td rowspan="2"></td><td colspan="3">Human Translation</td><td colspan="2">GMT</td><td colspan="2">Our MT</td><td rowspan="2">Multi. DPR</td></tr><tr><td>DPR</td><td>PATH</td><td>BM</td><td>DPR</td><td>PATH</td><td>DPR</td><td>PATH</td></tr><tr><td>Ar</td><td>68.3</td><td>70.0</td><td>41.6</td><td>67.5</td><td>63.3</td><td>52.5</td><td>51.6</td><td>50.4</td></tr><tr><td>Bn</td><td>85.6</td><td>82.0</td><td>57.0</td><td>83.2</td><td>78.9</td><td>63.2</td><td>64.8</td><td>57.7</td></tr><tr><td>Fi</td><td>73.1</td><td>70.2</td><td>43.7</td><td>68.1</td><td>64.1</td><td>65.9</td><td>59.5</td><td>58.9</td></tr><tr><td>Ja</td><td>68.9</td><td>63.0</td><td>38.8</td><td>60.1</td><td>52.3</td><td>52.1</td><td>41.7</td><td>37.3</td></tr><tr><td>Ko</td><td>70.9</td><td>63.6</td><td>43.8</td><td>66.3</td><td>54.0</td><td>46.5</td><td>37.6</td><td>42.8</td></tr><tr><td>Ru</td><td>65.2</td><td>63.7</td><td>35.2</td><td>60.4</td><td>56.5</td><td>47.3</td><td>38.1</td><td>44.0</td></tr><tr><td>Te</td><td>72.2</td><td>64.1</td><td>44.6</td><td>65.0</td><td>62.5</td><td>22.7</td><td>18.1</td><td>44.9</td></tr><tr><td>Av.</td><td>72.1</td><td>68.1</td><td>43.5</td><td>67.2</td><td>61.7</td><td>50.0</td><td>44.5</td><td>48.0</td></tr></table>

Table 5: R@5kt ( $§3.1$ ) on the test data in the XOR-RETRIEVE setting. PATH and BM denote Path Retriever and BM25 respectively. Multi. is a multilingual approach that bypasses the query translation step.

English translations of the questions used during the dataset collection as an upper bound of translate baselines. The best R@5kt macro-averaged over the 7 languages comes from running DPR on human translations: 72.1. Machine translation systems achieve averages of 67.2 (GMT) and 50.0 (our MT) again with DPR. The discrepancy between human and machine translation suggests that even state-of-the-art translation systems struggle to translate questions precisely enough to retrieve an evidence paragraph. Although the difference between GMT and our MT systems shows the effectiveness of industrial MT systems (large parallel data, model architecture, etc.), there remains a substantial performance gap from human translation. The translate baselines outperform the multilingual approach apart from Telugu, where our MT suffers from small parallel data (114k sentences), and as a result the multilingual approach performs better.

BM25 substantially underperforms the other two models across the board. DPR generally achieves similar performance, if not better, compared to Path Retriever despite the fact that Path Retriever was used in our annotation ( $§2.1.3$ ). As we found that these patterns persisted in all the following experiments, we will only report results with DPR.

# 4.3 XOR-ENGLISHSPAN Experiments

Table 6 shows the performance of the baseline models in XOR-ENGLISHSPAN. The average macro F1 score with queries translated by human translators is 38.2, substantially higher than that of MT-based models: 32.9 and 20.5 F1 points for GMT and our MT respectively. This suggests that errors in automatic query translation affect later layers in the pipeline. The multilingual approach consistently underperforms translation-based methods, similarly to XOR-RETRIEVE. As in XOR-RETRIEVE,

<table><tr><td rowspan="2"></td><td colspan="2">Human Translation</td><td colspan="2">GMT</td><td colspan="2">Our MT</td><td colspan="2">Multi.</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>Ar</td><td>43.2</td><td>32.8</td><td>39.5</td><td>28.5</td><td>28.0</td><td>23.4</td><td>17.9</td><td>11.7</td></tr><tr><td>Bn</td><td>43.4</td><td>35.9</td><td>42.1</td><td>34.4</td><td>25.6</td><td>20.3</td><td>19.4</td><td>14.1</td></tr><tr><td>Fi</td><td>34.8</td><td>26.0</td><td>28.2</td><td>21.3</td><td>29.3</td><td>22.1</td><td>24.5</td><td>18.3</td></tr><tr><td>Ja</td><td>29.9</td><td>22.3</td><td>23.5</td><td>17.4</td><td>19.2</td><td>13.8</td><td>13.1</td><td>10.7</td></tr><tr><td>Ko</td><td>36.9</td><td>28.8</td><td>30.5</td><td>23.8</td><td>19.4</td><td>14.2</td><td>14.3</td><td>9.9</td></tr><tr><td>Ru</td><td>37.0</td><td>29.4</td><td>34.8</td><td>26.4</td><td>18.4</td><td>13.6</td><td>17.2</td><td>11.1</td></tr><tr><td>Te</td><td>42.4</td><td>35.0</td><td>31.6</td><td>25.1</td><td>3.8</td><td>2.7</td><td>14.4</td><td>10.2</td></tr><tr><td>Av.</td><td>38.2</td><td>30.0</td><td>32.9</td><td>25.3</td><td>20.5</td><td>15.7</td><td>17.2</td><td>12.3</td></tr></table>

Table 6: Performance on XOR-ENGLISHSPAN. The rightmost Multi. section is a multilingual approach without query translation ( $§3.1$ ).

Telugu was an exception. The multilingual baseline significantly outperforms the translation-based approach with our MT system (14.4 vs. 3.6 F1 points). Query translation errors propagate to and directly impact downstream QA tasks in the languages with limited parallel data for MT training, and machine translation-based approaches may perform poorly. This encourages the research community to explore multilingual pretrained models to build a robust multilingual open-retrieval QA system for low-resource languages.

Similar to the original TYDI QA dataset, the performance on XOR-ENGLISHSPAN varies across languages, which can be partially explained by the differing sets of questions (Clark et al., 2020). The best baseline achieves 39.5 in Arabic compared to 23.5 F1 points in Japanese, which may come from differences in question difficulty as well as how the models are trained for each language.

# 4.4 XOR-FULL Experiments

Table 7 presents results on the XOR-FULL task. The first pipeline, which uses GMT, Google Search (GS), and DPR, yields the best average performance: 18.7 F1, 12.1 EM, and 16.8 BLEU points. This indicates that systems like GMT and GS, which are typically trained on large data, are effective. Yet, we encourage the community to experiment on top of open systems such that all experimental details can be fully reported and understood. Replacing GMT with our MT (second row) results in a large performance drop in Bengali (6.6 vs. 19.0 F1 points) and Telugu (1.7 vs. 13.6). Further replacing GS with BM25 retrieval in the target languages (third row) causes a large performance drop in all languages (e.g., 9.7 vs. 16.4 in Korean). Consistent with the previous tasks, the multilingual approach shown in the forth row underperforms the translation-based counterpart (15.7 vs. 18.7 F1 points on average). Similar baselines perform considerably better in prior open-retrieval QA datasets, such as MKQA (30 EM points, Longpre et al., 2020) and NQ questions (40 F1, Karpukhin et al., 2020). This gap illustrates the multidimensional challenge of XOR-TYDI QA.

# 4.5 Further Analysis

Effects of translation performance on overall QA results. Table 8 compares the query translation BLEU scores and the final QA F1 performance of the translation-based baseline with three different MT systems in XOR-ENGLISHSPAN: GMT, Our MT, and Helsinki (Tiedemann and Thottingal, 2020). GMT significantly outperforms the other two baselines, demonstrating that its training setup may yield large improvements in these languages; similarly, in cases where additional parallel training data is not available, multilingual models may remain strong modeling tools. On the other hand, it is noteworthy that high BLEU scores do not always lead to better QA performance. In Bengali and Finnish, while Helsinki achieves a considerably better BLEU score than our MT (33.0 vs. 30.8 in Bengali and 29.8 vs. 27.4 in Finnish), our MT is 3.9 and 1.3 F1 points better in downstream XOR-ENGLISHSPAN, respectively. See Appendix §D.3 for an example of translation errors resulting in QA errors. Those results suggest that the BLEU score is not always indicative of the downstream performance and that evaluating MT performance in the context of XOR QA would be important for improvements of multilingual QA systems.

Single language Wikipedia ablations in XOR-FULL. To assess our models' ability to benefit from multilingual collections, we try restricting the retrieval target to single language Wikipedia: English $W_{eng}$ only or target language $W_i$ only. In $W_{eng}$ only, the best system, which applies GMT and DPR, underperforms the best pipeline that uses both $\mathbf{W}_{i,eng}$ in all languages except for Finnish and Japanese. Similarly, the $W_i$ only setting generally underperforms the best $\mathbf{W}_{i,eng}$ pipeline. These results illustrate the importance of searching multilingual collections. See Table 15 for the full results.

# 5 Related Work

Multilingual QA Much recent effort has been made to create non-English QA datasets to over-

<table><tr><td colspan="2">Translation</td><td colspan="2">Retrieval</td><td colspan="7">Target Language  $L_i$  F1</td><td colspan="3">Macro Average</td></tr><tr><td>Query</td><td>Answer</td><td> $L_i$ </td><td>Eng.</td><td>Ar</td><td>Bn</td><td>Fi</td><td>Ja</td><td>Ko</td><td>Ru</td><td>Te</td><td>F1</td><td>EM</td><td>BLEU</td></tr><tr><td>GMT</td><td>GMT</td><td>GS</td><td>DPR</td><td>31.5</td><td>19.0</td><td>18.3</td><td>8.8</td><td>20.1</td><td>19.8</td><td>13.6</td><td>18.7</td><td>12.1</td><td>16.8</td></tr><tr><td>Our MT</td><td>Our MT</td><td>GS</td><td>DPR</td><td>29.6</td><td>6.6</td><td>15.5</td><td>7.6</td><td>16.4</td><td>18.7</td><td>1.7</td><td>13.7</td><td>8.7</td><td>12.0</td></tr><tr><td>Our MT</td><td>Our MT</td><td>BM25</td><td>DPR</td><td>12.1</td><td>22.0</td><td>9.3</td><td>5.4</td><td>9.7</td><td>7.4</td><td>0.8</td><td>9.5</td><td>6.0</td><td>8.9</td></tr><tr><td>-</td><td>GMT</td><td>GS</td><td>DPR</td><td>30.5</td><td>10.6</td><td>16.9</td><td>8.2</td><td>17.6</td><td>19.8</td><td>6.0</td><td>15.7</td><td>10.0</td><td>13.9</td></tr></table>

Table 7: Performance on XOR-FULL (test data F1 scores). “GS” denotes Google Search retrieval.

<table><tr><td rowspan="2">Query Translator</td><td colspan="7">MT BLEU</td><td colspan="7">XOR-ENGLISHSPAN F1</td></tr><tr><td>Ar</td><td>Bn</td><td>Fi</td><td>Ja</td><td>Ko</td><td>Ru</td><td>Avg</td><td>Ar</td><td>Bn</td><td>Fi</td><td>Ja</td><td>Ko</td><td>Ru</td><td>Avg</td></tr><tr><td>GMT</td><td>53.9</td><td>86.9</td><td>30.2</td><td>38.2</td><td>44.7</td><td>52.9</td><td>51.8</td><td>35.4</td><td>42.1</td><td>31.8</td><td>27.2</td><td>32.5</td><td>34.7</td><td>34.0</td></tr><tr><td>Our MT</td><td>33.7</td><td>30.8</td><td>27.4</td><td>19.7</td><td>30.8</td><td>21.7</td><td>27.4</td><td>20.9</td><td>25.2</td><td>31.9</td><td>19.6</td><td>25.3</td><td>16.1</td><td>23.2</td></tr><tr><td>Helsinki</td><td>35.9</td><td>33.0</td><td>29.8</td><td>19.8</td><td>31.8</td><td>37.3</td><td>31.1</td><td>28.4</td><td>21.3</td><td>30.6</td><td>19.0</td><td>25.3</td><td>29.6</td><td>25.7</td></tr></table>

Table 8: F1 scores on XOR-ENGLISHSPAN and the BLEU scores in query translation on the dev set. All configurations use DPR. Telugu is excluded since Helsinki does not support it as of October, 2020.

come the data scarcity in non-English languages. In addition to the datasets we already discussed in §2.2, several other non-English reading comprehension datasets have been created (Asai et al., 2018; Lim et al., 2019; Mozannar et al., 2019; d'Hoffschmidt et al., 2020). Liu et al. (2019) developed a template-based cloze task, leading to different data distributions from realistic questions with a great degree of lexical overlap between questions and reference paragraphs (Lee et al., 2019). More recently, Hardalov et al. (2020) introduced EXAMS, a multilingual multiple-choice reading comprehension dataset from school exams.

Our XOR-TYDI QA is also closely related to QA@CLEF 2003-2008 (Magnini et al., 2003, 2004; Vallin et al., 2005; Magnini et al., 2006; Giampiccolo et al., 2007; Forner et al., 2008); both QA@CLEF and XOR-TYDI QA attempt to develop and evaluate multilingual QA systems. Nevertheless, there are three crucial differences. First, our XOR-TYDI QA has a large number of questions that are required for training current state-of-the-art QA models like DPR, while QA@CLEF only has 200 evaluation questions for each language without training data (Forner et al., 2010). Secondly, the languages tested in QA@CLEF are all European languages, with the one exception of Indonesian; XOR-TYDI QA includes typologically diverse languages. Lastly, the task setup of QA@CLEF 2003-2008 is either monolingual—questions and documents are written in the same non-English language—or cross-lingual—the source and target languages are prespecified (Forner et al., 2010). In XOR QA, questions are asked in a target language but a system does not know in which language it can find an answer in a non-parallel Wikipedia collection. Those differences from QA@CLEF tasks better simulate real-world scenarios and introduce new challenges that have yet to be extensively studied.

Cross-lingual Information Retrieval Cross-lingual Information Retrieval (CLIR) is the task of retrieving relevant documents when the document collection is in a different language from the query language (Hull and Grefenstette, 1996). The retrieval component in XOR QA is closely related to CLIR, but differs in several critical ways. First, since the end goal of XOR QA is QA, XOR QA queries always take question forms rather than search key words. Further, while CLIR typically retrieves documents from a single (low-resource) language (Zhang et al., 2019), XOR QA considers documents from both English and the query language. In many applications, we do not know a priori in which language we can find target information. Lastly, our document collection is orders of magnitude bigger than typical CLIR benchmarks (Sasaki et al., 2018; Zhang et al., 2019).

# 6 Conclusion

We presented the task of XOR QA, in which a system retrieves and reads documents across languages to answer non-English information-seeking questions. We introduced a new large-scale XOR QA dataset, XOR-TYDI QA, with 40k newly annotated open-retrieval questions that cover seven typologically diverse languages. Our experiments showed that XOR-TYDI QA is a challenging benchmark that can benefit from further effort in both QA and multilinguality communities.

# Acknowledgments

This research was supported by gifts from Google, the Allen Distinguished Investigator Award, the Sloan Fellowship, and the Nakajima Foundation Fellowship. We thank Sewon Min, Kristina Toutanova, David Wadden, the members of the UW NLP group, and the anonymous reviewers for their insightful feedback on this paper, Nancy Li, Xun Cao, Hitesh Boinpally, Samek Mulepati, Casey Zhao, Vitaly Nikolaev, Soumyadip Sengupta, Bindita Chaudhuri, and Aditya Kusupati for their help on our annotations and dataset proofing, and Nelson Liu and Pradeep Dasigi for their suggestions on the annotation interface and Amazon Mechanical Turk crowdsourcing.

# Legal and Ethical Considerations

Were workers told what the dataset would be used for and did they consent? Crowdworkers consented to have their responses used in this way through the Amazon Mechanical Turk Participation Agreement.

If it relates to people, could this dataset expose people to harm or legal action? Our dataset can include incorrect information to the extent that Wikipedia can have wrong information about people. Nonetheless, we performed extensive quality control and answer verification to minimize the risk of harming people.

If it relates to people, does it unfairly advantage or disadvantage a particular social group? One fundamental problem with the existing question answering benchmarks is that most of their questions are written by native English speakers and overly represent English-centric topics, such as American politics, sports, and culture. As such, models trained and developed on those datasets are likely to fail to serve people with diverse language and cultural backgrounds. XOR-TYDI QA remedies this long-standing problem by annotating questions from native speakers of diverse languages. Thus, we encourage researchers and developers to benchmark on XOR-TYDI QA to mitigate the potential bias and unfairness of QA systems. We acknowledge, however, that this dataset still covers a very limited subset of languages in the world. We release a datasheet (Gebru et al., 2018) for our dataset to further document ethical implications. $^{10}$

# References

Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. 2020. On the cross-lingual transferability of monolingual representations. In ACL.   
Akari Asai and Eunsol Choi. 2020. Challenges in information seeking QA: Unanswerable questions and paragraph retrieval.   
Akari Asai, Akiko Eriguchi, Kazuma Hashimoto, and Yoshimasa Tsuruoka. 2018. Multilingual extractive reading comprehension by runtime machine translation.   
Akari Asai, Kazuma Hashimoto, Hannaneh Hajishirzi, Richard Socher, and Caiming Xiong. 2020. Learning to retrieve reasoning paths over Wikipedia graph for question answering. In ICLR.   
Loïc Barrault, Ondřej Bojar, Marta R. Costa-jussà, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Philipp Koehn, Shervin Malmasi, Christof Monz, Mathias Müller, Santanu Pal, Matt Post, and Marcos Zampieri. 2019. Findings of the 2019 conference on machine translation (WMT19). In WMT.   
Ewa S Callahan and Susan C Herring. 2011. Cultural bias in Wikipedia content on famous persons. JA-SIST.   
Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer open-domain questions. In ACL.   
Danqi Chen and Wen-tau Yih. 2020. Open-domain question answering. In ACL: Tutorial Abstracts.   
Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020. TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages. TACL.   
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In ACL.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL.   
Martin d'Hoffschmidt, Wacim Belblidia, Quentin Heinrich, Tom Brendlé, and Maxime Vidal. 2020. FQuAD: French question answering dataset. In Findings of EMNLP.   
Pamela Forner, Danilo Giampiccolo, Bernardo Magnini, Anselmo Peñas, Álvaro Rodrigo, and Richard Sutcliffe. 2010. Evaluating multilingual question answering systems at CLEF. In LREC.

Pamela Forner, Anselmo Peñas, Eneko Agirre, Iñaki Alegria, Corina Forăscu, Nicolas Moreau, Petya Osenova, Prokopis Prokopidis, Paulo Rocha, Bogdan Sacaleanu, et al. 2008. Overview of the CLEF 2008 multilingual question answering track. In CLEF.   
Timnit Gebru, J. Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, H. Wallach, Hal Daumé, and K. Crawford. 2018. Datasheets for datasets. In FAT/ML.   
Danilo Giampiccolo, Pamela Forner, Jesús Herrera, Anselmo Peñas, Christelle Ayache, Corina Forascu, Valentin Jijkoun, Petya Osenova, Paulo Rocha, Bogdan Sacaleanu, et al. 2007. Overview of the CLEF 2007 multilingual question answering track. In CLEF.   
Momchil Hardalov, Todor Mihaylov, Dimitrina Zlatkova, Yoan Dinkov, Ivan Koychev, and Preslav Nakov. 2020. EXAMS: A multi-subject high school examinations dataset for cross-lingual and multilingual question answering. In EMNLP.   
David A. Hull and Gregory Grefenstette. 1996. Querying across languages: a dictionary-based approach to multilingual information retrieval. In SIGIR.   
Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In EMNLP.   
Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello Federico, Nicola Bertoldi, Brooke Cowan, Wade Shen, Christine Moran, Richard Zens, Chris Dyer, Ondřej Bojar, Alexandra Constantin, and Evan Herbst. 2007. Moses: Open source toolkit for statistical machine translation. In ACL System Demonstrations.   
Taku Kudo. 2006. MeCab: Yet another part-of-speech and morphological analyzer.   
Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In EMNLP System Demonstrations.   
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A benchmark for question answering research. TACL.   
Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In ACL.   
Patrick Lewis, Barlas Oğuz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. 2020. MLQA: Evaluating cross-lingual extractive question answering. In ACL.

Seungyoung Lim, Myungji Kim, and Jooyoul Lee. 2019. KorQuaAD1.0: Korean QA dataset for machine reading comprehension.   
Jiahua Liu, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2019. XQA: A cross-lingual open-domain question answering dataset. In ACL.   
Shayne Longpre, Yi Lu, and Joachim Daiber. 2020. MKQA: A linguistically diverse benchmark for multilingual open domain question answering.   
Bernardo Magnini, Danilo Giampiccolo, Pamela Forner, Christelle Ayache, Valentin Jijkoun, Petya Osenova, Anselmo Penas, Paulo Rocha, Bogdan Sacaleanu, and Richard Sutcliffe. 2006. Overview of the CLEF 2006 multilingual question answering track. In CLEF.   
Bernardo Magnini, Simone Romagnoli, Alessandro Vallin, Jesús Herrera, Anselmo Penas, Víctor Peinado, Felisa Verdejo, and Maarten de Rijke. 2003. The multiple language question answering track at CLEF 2003. In CLEF.   
Bernardo Magnini, Alessandro Vallin, Christelle Ayache, Gregor Erbach, Anselmo Peñas, Maarten De Rijke, Paulo Rocha, Kiril Simov, and Richard Sutcliffe. 2004. Overview of the CLEF 2004 multilingual question answering track. In CLEF.   
Miniwatts Marketing Group. 2011. Internet world stats: Usage and population statistics.   
Hussein Mozannar, Elie Maamary, Karl El Hajal, and Hazem Hajj. 2019. Neural Arabic question answering. In WANLP.   
Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In NAACL System Demonstrations.   
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a method for automatic evaluation of machine translation. In ACL.   
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In EMNLP.   
Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval.   
Uma Roy, Noah Constant, Rami Al-Rfou, Aditya Barua, Aaron Phillips, and Yinfei Yang. 2020. LAREQA: Language-agnostic answer retrieval from a multilingual pool. In EMNLP.   
Shota Sasaki, Shuo Sun, Shigehiko Schamoni, Kevin Duh, and Kentaro Inui. 2018. Cross-lingual learning-to-rank with shared representations. In NAACL.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In ACL.   
Jörg Tiedemann and Lars Nygaard. 2004. The OPUS corpus - parallel and free. In LREC.   
Jörg Tiedemann and Santhosh Thottingal. 2020. OPUS-MT — Building open translation services for the World. In EAMT.   
Alessandro Vallin, Bernardo Magnini, Danilo Giampiccolo, Lili Aunimo, Christelle Ayache, Petya Osenova, Anselmo Peñas, Maarten De Rijke, Bogdan Sacaleanu, Diana Santos, et al. 2005. Overview of the CLEF 2005 multilingual question answering track. In CLEF.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In NeurIPS.   
Rui Zhang, Caitlin Westerfield, Sungrok Shim, Garrett Bingham, Alexander R. Fabbri, Neha Verma, William T Hu, and Dragomir R. Radev. 2019. Improving low-resource cross-lingual document retrieval by reranking with deep bilingual representations. In ACL.   
Michał Ziemski, Marcin Junczys-Dowmunt, and Bruno Pouliquen. 2016. The United Nations parallel corpus v1.0. In LREC.

# Appendix

# A Spirit behind Annotation Interface Design

Open-retrieval annotation desiderata. Open-retrieval QA annotation comes with unique challenges. In article-oriented QA such as SQuAD (Rajpurkar et al., 2016), all labels are with regard to a single document and a single human can indeed read the whole document. In open-retrieval QA, answers can be retrieved from millions of documents. Because exhaustively reading so much content is impossible for humans, the notion of “human performance” must be reconsidered in this context. This is why we only evaluate questions having answers in the open-retrieval setting and discard those where no answer was found—it is difficult to prove an answer does not exist in the millions of documents.

Limits of traditional annotation. In addition to fundamental problems of information scarcity and asymmetry in multilingual QA, questions can be labeled as unanswerable simply because of annotation errors. Annotation procedures for information-seeking QA data usually have each annotator read a single Wikipedia article retrieved by a search engine and label a correct answer span or label the question as not answered by the article (Kwiatkowski et al., 2019; Clark et al., 2020). In this procedure, the answer coverage is underestimated when the search engine fails to retrieve relevant articles (retrieval errors) or the annotator overlooks answer content from the selected articles (answer annotation errors, Asai and Choi, 2020). Importantly, these two types of annotation errors present a tradeoff: if we retrieve many articles, retrieval errors will be reduced at the expense of answer annotation errors because annotators have to find answer context among many candidate articles. An annotation procedure that misses too many answers will lead to an artificially small dataset.

# B Additional Details of Dataset Creation

# B.1 Annotation Interface

In this section, we describe the details of the annotation interface we used for answer annotation in English ( $§2.1.3$ ). The annotation interface can be seen in Figs. 4 and 5. To maximize the answer coverage for open-retrieval questions, we first rank paragraphs from top articles retrieved by Google

Search. During this paragraph ranking process, we only consider top 5 paragraphs and exclude the articles ranked from top 6 to 10. Increasing the number of the initial articles introduces more noise and confuses our paragraph ranking model, while human annotators sometimes found that those low-ranked articles relevant and retrieved answers from them as discussed in §2.1.3. In the annotation interface, we first present those top 5 paragraphs first (the ones highlighted in light blue in Fig. 4). When annotators do not find answers in the pre-selected top 5 paragraphs, they will explore more paragraphs and articles by expanding originally collapsed articles as in Fig. 5.

![](images/5324a69a6a3750953e6c92c97501ce80312e81ee2bd82d58ae2bdea66b3dc0a9.jpg)

<details>
<summary>text_image</summary>

Who was the first Prime Minister of Japan?
Prime Minister of Japan (collapse article) (hide full article)
Introduction
The is the head of government and chief executive of Japan. The Prime Minister is appointed by the Emperor of Japan after being designated by the National Diet and must enjoy the confidence of the House of Representatives to remain in office. He is the chairman of the Cabinet and appoints and dismisses the other Ministers of State. The literal translation of the Japanese name for the office is "Minister for the Comprehensive Administration of ("or" the Presidency over) the Cabinet".
History.
Before the adoption of the Meiji Constitution, Japan had in practice no written constitution. Originally, a Chinese-inspired legal system known as "nitsuryo" was enacted in the late Asuka period and early Nara period. It described a government based on an elaborate and rational metricofic bureaucracy, serving, in theory, under the ultimate authority of the Emperor; although in practice, real power was often held elsewhere, such as in the hands of the Fujiwara clan, who intermarried with the Imperial Family in the Helian period, or by the ruling "shōgun". Theoretically, the last "nitsuryo" code, the YoY0 Code enacted in 752, was still in force at the time of the Meiji Restoration.
Under this system, the was the head of the "Daijō-kan" (Department of State), the highest organ of Japan's pre-modern Imperial government during the Helian period and until briefly under the Meiji Constitution with the appointment of Sanjō Sanetomi in 1871. The office was replaced in 1885 with the appointment of IoT Hirobumi to the new position of Prime Minister, four years before the enactment of the Meiji Constitution, which mentions neither the Cabinet nor the position of Prime Minister's explicitly. It took its current form with the adoption of the Constitution of Japan in 1947.
To date, 62 people have served this position. The current Prime Minister is Shinzō Abe, who re-took office on December 26, 2012. He is the first former Prime Minister to return to office since 1948, and the 5th longest serving Prime Minister to date.
</details>

Figure 4: Annotation interface (expanded). The blue highlighted paragraphs are ranked high by the BERT paragraph ranker, and the orange highlighted paragraph is the one clicked by the annotator.

![](images/5aeba3564df80da1b362575d0483f0ac7e3205166a879e5ce2f5946a51f98a2b.jpg)

<details>
<summary>text_image</summary>

Who was the first Prime Minister of Japan?

Prime Minister of Japan [show paragraphs] [show full article]

List of Japanese prime ministers by longevity [collapse article]

Introduction
This is a list of Japanese prime ministers by longevity. It consists of Prime Ministers and interim Prime Ministers of Japan who have held the office. If a Prime Minister served more than one non-consecutive term, the dates given are for the beginning of their first term, and the end of their last term.

To avoid confusion and maintain consistency, the name of the Prime Ministers are listed in the Western style (given name, family name). Where the person in question is still living, the longevity is calculated up to .

Overview.
The median age of a Prime Minister first taking office is 61 years and 2 months. This falls between Tomosaburō Katō and Keizō Obuchi. The youngest Prime Minister was the first, Hirobumi itō, who took office at the age of 44 years, 67 days. The oldest Prime Minister first take office was Kantarō Suzuki, who became Prime Minister at the age of 77 years, 79 days.

The oldest living Prime Minister is Yasuhiro Nakasone, born 27 May 1918 (age.). The youngest living former Prime Minister is Yoshihiko Noda, born 20 May 1957 (age).

The longest-lived Prime Minister was Naruhiko Higashikuni, who died at the age of 102 years, 48 days. If the oldest living Prime Minister, Yasuhiro Nakasone (born 27 May 1918) lives to 13 July 2020, he will tie this record. The shortest-lived Prime Minister was Sanetomi Sanjō, who died at the age of 53 years and 352 days.

Junichiro Koizumi [show article]

Naoto Kan [show article]
</details>

Figure 5: Annotation interface (collapsed). Annotators can choose to read full articles or collapse articles.

# B.2 Quality Control for Question Translation

We first ask Gengo translators to translate 20 sample questions following our detailed instruction before starting the task, and ask native speakers to assess the quality of translations. We filter out translators who do not provide translation results that meet our standard (e.g., wrong translations of entities, heavy reliance on public machine translation systems). We have found that some of the translators almost copy and paste outputs of existing APIs without fixing errors even when there are crucial errors. After this initial qualification process, we observe that the translation quality is sufficiently high.

# B.3 Quality Control for QA annotation

To control the QA annotation quality, we recruit workers with a high approval rate ( $\geq 96\%$ ) located in English-speaking countries and conducted a rigorous qualification procedure. In our qualification stage, we post small calibration batches and evaluate the workers' performance by expert judgements from authors and agreement with other annotators. To keep the high quality of annotations, we randomly sample qualified workers weekly and manually monitor their annotations by comparing them with gold annotations by authors. We remove qualifications when we detect too many incorrect annotations (e.g., label a paragraph about a different person as a gold paragraph) and remove the annotations done by those disqualified annotators, which are later reannotated by a qualified worker. Over 200 annotators participated in our calibration tasks. About 40 workers are qualified with 24 actively working on the final dataset. Each HIT contains 5 questions with a reward ranging from 1.5 to 2.5 USD. Qualified annotators generally spend 1-2 minutes to answer each question. We give special rewards to annotators who actively search additional paragraphs or articles; the amounts of the rewards are calculated based on the numbers of the HITs they have submitted, resulting in 5-10 USD for each payment.

# B.4 Answer Translation Instructions

During answer translation, we asked annotators to follow the instructions listed below:

- Translators need to use metric units by default, instead of imperial units.   
- If the original answers are expressed in an imperial unit, translators are encouraged to

convert them into a metric unit (e.g., Height 5'3" -> 身長 160 cm).

\- When translating proper nouns, translators are asked to use an official translation if it is available in Wikipedia; otherwise they are encouraged to transliterate them.

We also specify some language-specific instructions to make the translated answers consistent with the ones in the original TYDI QA dataset.

- For Japanese and Korean, translators do not need to spell out the numbers (e.g., 1954 -> 千九百五十四) as people usually use Arabic numerals.   
- For Bengali, we expect the numbers will be spelled out in Bengali numerals as Bengali speakers rarely use Arabic numerals.   
- For Japanese and Korean, translators use appropriate measure words (e.g., 1867년, 57歳) if those measure words are commonly added in those languages.   
- For the languages where the date needs to be expressed in some rigid format, translators need to follow the format.

# B.5 Full Data Statistics of Cross-lingual data

Seen in Table 9 are full data statistics of cross-lingual data of XOR-TYDI QA. Among the questions with “Long” answer annotations are some questions without any short answers as in Natural Questions or TYDI QA. We do not include those “Long answer only” examples in our XOR-TYDI QA evaluations.

# C Training details

We describe the details in training our baselines to facilitate easy replication of our results.

# C.1 Machine Translation Models

Table 10 lists hyperpameters for training our transformer machine translation models. We generally follow the hyperpameters for the base-sized transformer (Vaswani et al., 2017). The one exception is English↔Russian where we used pretrained transformer large models. $^{11}$ For each language direction, all data are encoded into subwords by Moses tokenization (Koehn et al., 2007, for Arabic, Finnish, and Russian) and BPE (Sennrich et al., 2016) or SentencePiece (Kudo and Richardson, 2018, for Bengali, Japanese, Korean, and Telugu). We train an autoregressive transformer (Vaswani et al., 2017) with the fairseq library (Ott et al., 2019).

# C.2 Retrieval Models

Training DPR and Path Retriever. To train an English DPR and Path Retriever, we first initialize the parameters of the models with the ones trained on Natural Questions Open data, which is available on their repository. During finetuning on XOR-TYDI QA, we use the human translated questions with the annotated gold paragraph data.

Choice of negative and positive context. Selection of positive and negative examples is crucial to train competitive neural retriever models (Karpukhin et al., 2020). We follow the hyperparameters used in the original papers (Karpukhin et al., 2020; Asai et al., 2020). To construct effective negative and positive context, we follow the approaches introduced by the authors of those works.

To train DPR, we use the original gold paragraphs (long answers) annotated by MTurkers as positive passages. Following the experimental settings of DPR on Natural Questions, we first split gold paragraphs into 100-token units, and consider the units with the original short answer annotations as positive context. For negative context, we first randomly sample one negative paragraph per question from the top 5 paragraphs pre-selected by our paragraph reranking model in §2.1.3, split the negative paragraph into 100-token units, and then randomly pick one to use it as a negative context. We also reuse the in-batch negative paragraphs as discussed in Karpukhin et al. (2020).

Regarding the training of Path Retriever, we randomly sample top 50 paragraphs from the top 10 articles retrieved for annotations and use them as negative paragraphs. We also use the annotated long answers as positive paragraphs.

Implementation details of BM25 Retrievers. To implement BM25-based retrievers for the 7 languages, we use Elasticsearch's Python client (Python Elasticsearch Client). $^{12}$ We apply the default tokenizers and analyzers for Arabic, Bengali, Finnish and Russian. Japanese and Korean are not supported by the default Elasticsearch language

<table><tr><td rowspan="2"> $L_i$ </td><td colspan="3">Train (1 way)</td><td colspan="3">Dev (2 way)</td><td colspan="3">Test (2 way)</td></tr><tr><td>Total</td><td>Long (%)</td><td>Short (%)</td><td>Total</td><td>Long (%)</td><td>Short (%)</td><td>total</td><td>Long (%)</td><td>Short (%)</td></tr><tr><td>Arabic</td><td>4,500</td><td>2,862 (63)</td><td>2,574 (57)</td><td>500</td><td>357 (71)</td><td>350 (70)</td><td>235</td><td>144 (61)</td><td>137 (58)</td></tr><tr><td>Bengali</td><td>4,500</td><td>2,822 (63)</td><td>2,582 (57)</td><td>500</td><td>330 (66)</td><td>312 (62)</td><td>185</td><td>131 (70)</td><td>128 (69)</td></tr><tr><td>Finnish</td><td>4,500</td><td>2,454 (55)</td><td>2,088 (46)</td><td>500</td><td>372 (74)</td><td>360 (72)</td><td>800</td><td>556 (69)</td><td>530 (66)</td></tr><tr><td>Japanese</td><td>4,500</td><td>2,557 (57)</td><td>2,288 (51)</td><td>500</td><td>320 (64)</td><td>296 (60)</td><td>779</td><td>477 (61)</td><td>449 (58)</td></tr><tr><td>Korean</td><td>4,500</td><td>2,674 (59)</td><td>2,469 (55)</td><td>500</td><td>314 (63)</td><td>299 (60)</td><td>1,177</td><td>684 (58)</td><td>646 (55)</td></tr><tr><td>Russian</td><td>4,500</td><td>2,178 (48)</td><td>1,941 (43)</td><td>500</td><td>270 (54)</td><td>255 (51)</td><td>470</td><td>252 (53)</td><td>235 (50)</td></tr><tr><td>Telugu</td><td>4,500</td><td>1,515 (33)</td><td>1,308 (29)</td><td>500</td><td>258 (52)</td><td>238 (47)</td><td>1,752</td><td>394 (22)</td><td>374 (21)</td></tr></table>

Table 9: Dataset statistics of the resulting XOR QA corpus (cross-lingual data only). “Long” denotes the questions with paragraph answer annotations, and “Short” denotes the questions with short answer annotations. During evaluation, we disregard the questions without short answer annotations.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>label smoothing</td><td>0.1</td></tr><tr><td># max tokens</td><td>4096</td></tr><tr><td>dropout rate</td><td>0.3</td></tr><tr><td>encoder embedding dim</td><td>512</td></tr><tr><td>encoder ffn dim</td><td>2048</td></tr><tr><td># encoder attn heads</td><td>8</td></tr><tr><td>decoder embedding dim</td><td>512</td></tr><tr><td>decoder ffn dim</td><td>2048</td></tr><tr><td># decoder attn heads</td><td>8</td></tr><tr><td>max source positions</td><td>10000</td></tr><tr><td>max target positions</td><td>10000</td></tr><tr><td>Adam lrate</td><td> $5 \times 10^{-4}$ </td></tr><tr><td>Adam  $\beta_{1}$ </td><td>0.9</td></tr><tr><td>Adam  $\beta_{2}$ </td><td>0.98</td></tr><tr><td>lr-scheduler</td><td>inverse square</td></tr><tr><td>warm-up lr</td><td> $1 \times 10^{-7}$ </td></tr><tr><td># warmup updates</td><td>4000</td></tr><tr><td># max updates</td><td>300K</td></tr><tr><td>length penalty</td><td>1.0</td></tr></table>

Table 10: Hyperparameters for our transformer machine translation models.

analyzers, so we use Kuromoji $^{13}$ and Nori plugins $^{14}$ for Japanese and Korean respectively. Note that we do not implement a BM25-based retriever for Telugu, since it is not supported by the default language analyzer and we could not find an official plugin for Telugu.

# C.3 Machine Reading Models

We use the official hyperparameters for machine reading components of DPR and Path Retriever. Table 11 shows the list of the hyperparameters used to train a multilingual machine reading model for the monolingual pipeline in XOR-FULL. We lowercased input paragraphs and questions.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>max sequence length</td><td>384</td></tr><tr><td>document stride</td><td>128</td></tr><tr><td>max query length</td><td>64</td></tr><tr><td>Adam lrate</td><td> $5 \times 10^{-4}$ </td></tr><tr><td>Adam  $\epsilon$ </td><td> $1 \times 10^{-8}$ </td></tr><tr><td>max gradient norm</td><td>1.0</td></tr><tr><td># train epochs</td><td>3.0</td></tr><tr><td>seed</td><td>42</td></tr></table>

Table 11: Hyperparameters for our machine reading model in the monolingual pipeline.

Choice of negative and positive examples. For the Path Retriever and BM25 baselines' reader, we sample three negative paragraphs per annotated question-gold paragraph pair and train a model that jointly predicts an answer span and relevance score of each paragraph to the question, following Asai et al. (2020). In DPR, the training examples are retrieved by the trained retriever, and we train the reader with 24 negative paragraphs by distant supervision (Karpukhin et al., 2020). We use human translated English questions to train English reader models, and use the original questions in $L_i$ to train a multilingual reader model.

# D Additional Results and Analysis

# D.1 Additional Experimental Results

XOR-RETRIEVE. We present the R@2kt scores of the retrieval baselines in Table 12. As shown in Table 5, given human translations, DPR generally outperforms other two retrieval baselines. We also present R@2kt and R@5kt of our DPR models on our development set in Table 13, and we observe a similar performance trend to the test set: models with queries translated by GMT outperform other models in all of the XOR-TYDI QA languages. Comparing the two baselines that do not use external black-box APIs, we see that

<table><tr><td rowspan="2"></td><td colspan="3">Human</td><td colspan="2">GMT</td><td colspan="2">Our MT</td><td rowspan="2">Multi. DPR</td></tr><tr><td>DPR</td><td>PATH</td><td>BM</td><td>DPR</td><td>PATH</td><td>DPR</td><td>PATH</td></tr><tr><td>Ar</td><td>65.8</td><td>65.0</td><td>41.6</td><td>61.7</td><td>59.1</td><td>48.3</td><td>45.0</td><td>41.2</td></tr><tr><td>Bn</td><td>72.8</td><td>78.1</td><td>57.7</td><td>72.0</td><td>58.2</td><td>54.4</td><td>60.9</td><td>43.9</td></tr><tr><td>Fi</td><td>66.5</td><td>68.0</td><td>43.7</td><td>60.6</td><td>60.3</td><td>56.7</td><td>56.6</td><td>50.3</td></tr><tr><td>Ja</td><td>62.0</td><td>59.0</td><td>38.8</td><td>52.1</td><td>50.0</td><td>41.8</td><td>36.7</td><td>29.1</td></tr><tr><td>Ko</td><td>65.0</td><td>60.0</td><td>43.8</td><td>57.9</td><td>50.3</td><td>39.4</td><td>33.8</td><td>34.5</td></tr><tr><td>Ru</td><td>57.5</td><td>59.9</td><td>35.2</td><td>51.2</td><td>54.1</td><td>39.6</td><td>34.7</td><td>35.3</td></tr><tr><td>Te</td><td>66.3</td><td>59.6</td><td>44.6</td><td>59.4</td><td>58.0</td><td>18.7</td><td>15.7</td><td>37.2</td></tr><tr><td>Av.</td><td>65.1</td><td>64.3</td><td>43.5</td><td>59.3</td><td>58.2</td><td>42.7</td><td>40.5</td><td>38.8</td></tr></table>

Table 12: R@2kt (§3.1) on the test data in the XOR-RETRIEVE setting. PATH and BM denote Path Retriever and BM25 respectively. The rightmost column is a multilingual approach that bypasses the query translation step (§3.1).

<table><tr><td rowspan="2"></td><td colspan="2">GMT</td><td colspan="2">Our MT</td><td colspan="2">Multi.</td></tr><tr><td>R@2kt</td><td>R@5kt</td><td>R@2kt</td><td>R@5kt</td><td>R@2kt</td><td>R@5kt</td></tr><tr><td>Ar</td><td>62.5</td><td>69.6</td><td>43.4</td><td>52.4</td><td>38.8</td><td>48.9</td></tr><tr><td>Bn</td><td>74.7</td><td>82.2</td><td>53.9</td><td>62.8</td><td>48.4</td><td>60.2</td></tr><tr><td>Fi</td><td>57.3</td><td>62.4</td><td>55.1</td><td>61.8</td><td>52.5</td><td>59.2</td></tr><tr><td>Ja</td><td>55.6</td><td>64.7</td><td>40.2</td><td>48.1</td><td>26.6</td><td>34.9</td></tr><tr><td>Ko</td><td>60.0</td><td>68.8</td><td>50.5</td><td>58.6</td><td>44.2</td><td>49.8</td></tr><tr><td>Ru</td><td>52.7</td><td>60.8</td><td>30.8</td><td>37.8</td><td>33.3</td><td>43.0</td></tr><tr><td>Te</td><td>72.3</td><td>79.0</td><td>20.2</td><td>32.4</td><td>39.9</td><td>55.5</td></tr><tr><td>Av.</td><td>62.2</td><td>69.6</td><td>42.0</td><td>50.6</td><td>40.5</td><td>50.2</td></tr></table>

Table 13: R@5kt ( $§3.1$ ) of DPR models (translate DPR and multilingual DPR) on the development data in the XOR-RETRIEVE setting.

the translation approach (Our MT) outperforms the multilingual one (Multi.) in Arabic, Bengali, Finnish Japanese, and Korean, while it performs poorly in Telugu. These results are consistent with the ones on the test data in Table. 5.

XOR-ENGLISHSPAN. Table 14 shows the F1 and EM scores of our DPR models on the development data in the XOR-ENGLISHSPAN setting. Similar to the results on XOR-RETRIEVE, GMT significantly outperforms our MT and our multilingual model. Probably due to the error propagation, the Telugu performance of our MT baseline is low, indicating the importance of developing a multilingual baseline that could perform well on languages with little parallel data for translation training.

XOR-FULL. We present F1, BLEU and EM scores for XOR-FULL in Tables 15, 16 and 17. We also present F1 scores and average F1, BLEU and EM scores on the development set in Table 18.

<table><tr><td rowspan="2"></td><td colspan="2">GMT</td><td colspan="2">Our MT</td><td colspan="2">Multi.</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>Ar</td><td>35.4</td><td>27.7</td><td>20.9</td><td>14.9</td><td>17.2</td><td>12.3</td></tr><tr><td>Bn</td><td>42.1</td><td>35.3</td><td>25.2</td><td>20.5</td><td>21.8</td><td>17.3</td></tr><tr><td>Fi</td><td>31.8</td><td>23.1</td><td>31.9</td><td>23.3</td><td>27.6</td><td>20.8</td></tr><tr><td>Ja</td><td>27.2</td><td>20.9</td><td>19.6</td><td>15.5</td><td>15.5</td><td>12.8</td></tr><tr><td>Ko</td><td>32.5</td><td>22.7</td><td>25.3</td><td>18.1</td><td>18.5</td><td>14.4</td></tr><tr><td>Ru</td><td>34.7</td><td>28.2</td><td>16.1</td><td>11.4</td><td>21.3</td><td>17.3</td></tr><tr><td>Te</td><td>35.0</td><td>27.4</td><td>3.6</td><td>1.7</td><td>17.7</td><td>13.1</td></tr><tr><td>Av.</td><td>35.0</td><td>27.4</td><td>20.4</td><td>15.1</td><td>19.9</td><td>15.4</td></tr></table>

Table 14: F1 and EM scores of our DPR models (translate DPR and multilingual DPR) on the development data in the XOR-ENGLISHSPAN setting.

# D.2 Additional Analysis

Single language Wikipedia ablations in XOR-FULL. In XOR-FULL, a system is expected to answer a question in the target language by consulting multilingual Wikipedia corpora, but which language answer content exists in is not known a priori (§3.3). To understand the benefit of retrieving evidence from a multilingual document pool, we run single language Wikipedia ablations. In this study, we conduct ablations in which systems only use either English Wikipedia ( $W_{eng}$ ) or the target language's Wikipedia ( $W_i$ ). We run the monolingual baselines for $W_i$ only and the cross-lingual baselines for $W_{eng}$ only. For the cross-lingual baseline, all predicted answers will be translated back to the target languages.

The bottom section of Table 15 shows the full results of single language Wikipedia ablations on XOR-FULL. In a majority of the languages, we observed performance drops from the full models that use both $W_{eng}$ and $W_{i}$ (e.g., 20.1 vs. 14.3 F1 in Korean). In Japanese and Finnish, our English Wikipedia only baselines outperform the full models. Currently, the answer aggregation process prioritizes answers predicted by monolingual models, but the monolingual models perform poorly in those two languages. Future work can address the challenges of improving evidence and answer aggregation from multilingual document collections.

Per-difficulty retrieval performance. We split our data by annotation difficulty i.e., whether or not a gold paragraph is selected by the BERT retriever used during annotation in our our collaborative annotation framework ( $§2.1.3$ ). Table 19 presents retrieval performance broken down by difficulty. We observed a large performance gap between the easy and hard subsets (65.3 for easy vs. 59.9 for

<table><tr><td rowspan="2">Wiki Corpus</td><td colspan="2">Translation</td><td colspan="2">Retrieval</td><td colspan="8">Target Language  $L_i$ </td></tr><tr><td>Query</td><td>Answer</td><td> $L_i$ </td><td>Eng.</td><td>Ar</td><td>Bn</td><td>Fi</td><td>Ja</td><td>Ko</td><td>Ru</td><td>Te</td><td>Avg.</td></tr><tr><td rowspan="4"> $W_{i,eng}$ </td><td>GMT</td><td>GMT</td><td>GS</td><td>DPR</td><td>31.5</td><td>19.0</td><td>18.3</td><td>8.8</td><td>20.1</td><td>19.8</td><td>13.6</td><td>18.7</td></tr><tr><td>Our MT</td><td>Our MT</td><td>GS</td><td>DPR</td><td>29.6</td><td>6.6</td><td>15.5</td><td>7.6</td><td>16.4</td><td>18.7</td><td>1.7</td><td>13.7</td></tr><tr><td>Our MT</td><td>Our MT</td><td>BM25</td><td>DPR</td><td>12.1</td><td>22.0</td><td>9.3</td><td>5.4</td><td>9.7</td><td>7.4</td><td>0.8</td><td>9.5</td></tr><tr><td>-</td><td>GMT</td><td>GS</td><td>mDPR</td><td>30.5</td><td>5.2</td><td>16.9</td><td>8.2</td><td>17.6</td><td>19.8</td><td>6.0</td><td>15.7</td></tr><tr><td rowspan="3"> $W_{eng}$ </td><td>GMT</td><td>GMT</td><td>-</td><td>DPR</td><td>23.9</td><td>18.5</td><td>22.9</td><td>24.1</td><td>17.5</td><td>16.8</td><td>13.2</td><td>19.5</td></tr><tr><td>Our MT</td><td>Our MT</td><td>-</td><td>DPR</td><td>7.6</td><td>5.9</td><td>16.2</td><td>9.0</td><td>5.3</td><td>5.5</td><td>0.8</td><td>7.2</td></tr><tr><td>-</td><td>GMT</td><td>-</td><td>mDPR</td><td>12.4</td><td>9.7</td><td>19.1</td><td>14.0</td><td>8.2</td><td>10.9</td><td>5.4</td><td>11.3</td></tr><tr><td rowspan="2"> $W_i$ </td><td>-</td><td>-</td><td>GS</td><td>-</td><td>29.0</td><td>0.9</td><td>9.5</td><td>6.2</td><td>14.3</td><td>18.5</td><td>0.9</td><td>11.3</td></tr><tr><td>-</td><td>-</td><td>BM25</td><td>-</td><td>12.0</td><td>22.0</td><td>9.3</td><td>5.3</td><td>9.7</td><td>7.4</td><td>-</td><td>-</td></tr></table>

Table 15: Performance on XOR-FULL task (F1 scores on the test data). “GS” denotes Google Search retrieval. The bottom section shows results from single Wikipedia baselines. ElasticSearch for BM25 does not support Telugu. “mDPR” denotes a DPR model where query and context encoders are initialized with multilingual BERT.

<table><tr><td rowspan="2">Wiki Corpus</td><td colspan="2">Translation</td><td colspan="2">Retrieval</td><td colspan="8">Target Language  $L_i$ </td></tr><tr><td>Query</td><td>Answer</td><td> $L_i$ </td><td>Eng.</td><td>Ar</td><td>Bn</td><td>Fi</td><td>Ja</td><td>Ko</td><td>Ru</td><td>Te</td><td>Avg.</td></tr><tr><td rowspan="4"> $W_{i,eng}$ </td><td>GMT</td><td>GMT</td><td>GS</td><td>DPR</td><td>22.1</td><td>10.9</td><td>13.3</td><td>3.0</td><td>20.1</td><td>11.4</td><td>9.1</td><td>12.1</td></tr><tr><td>Our MT</td><td>Our MT</td><td>GS</td><td>DPR</td><td>20.9</td><td>2.2</td><td>10.9</td><td>2.3</td><td>12.6</td><td>10.5</td><td>1.4</td><td>8.7</td></tr><tr><td>Our MT</td><td>Our MT</td><td>BM25</td><td>DPR</td><td>7.7</td><td>15.4</td><td>6.4</td><td>1.3</td><td>6.7</td><td>3.9</td><td>0.6</td><td>6.0</td></tr><tr><td>-</td><td>GMT</td><td>GS</td><td>mDPR</td><td>21.4</td><td>5.2</td><td>12.1</td><td>2.7</td><td>13.3</td><td>11.3</td><td>3.9</td><td>10.0</td></tr><tr><td rowspan="3"> $W_{eng}$ </td><td>GMT</td><td>GMT</td><td>-</td><td>DPR</td><td>12.3</td><td>10.1</td><td>16.6</td><td>14.1</td><td>11.5</td><td>10.4</td><td>8.5</td><td>12.0</td></tr><tr><td>Our MT</td><td>Our MT</td><td>-</td><td>DPR</td><td>2.5</td><td>1.5</td><td>10.3</td><td>3.3</td><td>2.9</td><td>2.5</td><td>0.5</td><td>3.4</td></tr><tr><td>-</td><td>GMT</td><td>-</td><td>mDPR</td><td>6.7</td><td>4.5</td><td>13.5</td><td>8.1</td><td>8.2</td><td>6.5</td><td>3.1</td><td>6.8</td></tr><tr><td rowspan="2"> $W_i$ </td><td>-</td><td>-</td><td>GS</td><td>-</td><td>20.6</td><td>0.7</td><td>7.1</td><td>1.5</td><td>11.5</td><td>10.4</td><td>0.8</td><td>7.5</td></tr><tr><td>-</td><td>-</td><td>BM25</td><td>-</td><td>7.7</td><td>15.3</td><td>6.4</td><td>1.3</td><td>6.7</td><td>3.9</td><td>-</td><td>-</td></tr></table>

Table 16: Performance on XOR-FULL (EM scores on the test data). “GS” denotes Google Search retrieval. The bottom section shows results from single Wikipedia baselines. ElasticSearch for BM25 does not support Telugu. “mDPR” denotes a DPR model where query and context encoders are initialized with multilingual BERT.

<table><tr><td rowspan="2">Wiki Corpus</td><td colspan="2">Translation</td><td colspan="2">Retrieval</td><td colspan="8">Target Language  $L_i$ </td></tr><tr><td>Query</td><td>Answer</td><td> $L_i$ </td><td>Eng.</td><td>Ar</td><td>Bn</td><td>Fi</td><td>Ja</td><td>Ko</td><td>Ru</td><td>Te</td><td>Avg.</td></tr><tr><td rowspan="4"> $W_{i,eng}$ </td><td>GMT</td><td>GMT</td><td>GS</td><td>DPR</td><td>29.7</td><td>22.1</td><td>18.8</td><td>2.2</td><td>13.3</td><td>18.0</td><td>13.5</td><td>16.8</td></tr><tr><td>Our MT</td><td>Our MT</td><td>GS</td><td>DPR</td><td>27.8</td><td>7.4</td><td>10.9</td><td>2.0</td><td>12.6</td><td>17.0</td><td>1.1</td><td>8.9</td></tr><tr><td>Our MT</td><td>Our MT</td><td>BM25</td><td>DPR</td><td>12.8</td><td>22.9</td><td>6.4</td><td>1.2</td><td>7.0</td><td>7.3</td><td>0.3</td><td>12.0</td></tr><tr><td>-</td><td>GMT</td><td>GS</td><td>mDPR</td><td>27.8</td><td>7.0</td><td>13.9</td><td>1.8</td><td>11.3</td><td>17.0</td><td>5.3</td><td>13.9</td></tr><tr><td rowspan="3"> $W_{eng}$ </td><td>GMT</td><td>GMT</td><td>-</td><td>DPR</td><td>24.5</td><td>21.4</td><td>20.6</td><td>6.1</td><td>10.0</td><td>14.2</td><td>13.3</td><td>15.7</td></tr><tr><td>Our MT</td><td>Our MT</td><td>-</td><td>DPR</td><td>8.6</td><td>6.7</td><td>16.7</td><td>2.8</td><td>3.9</td><td>5.0</td><td>0.3</td><td>6.3</td></tr><tr><td>-</td><td>GMT</td><td>-</td><td>mDPR</td><td>12.2</td><td>10.2</td><td>16.7</td><td>2.4</td><td>4.7</td><td>8.2</td><td>8.3</td><td>9.0</td></tr><tr><td rowspan="2"> $W_i$ </td><td>-</td><td>-</td><td>GS</td><td>-</td><td>27.3</td><td>0.7</td><td>10.4</td><td>1.6</td><td>10.4</td><td>16.6</td><td>0.8</td><td>9.7</td></tr><tr><td>-</td><td>-</td><td>BM25</td><td>-</td><td>12.8</td><td>22.9</td><td>10.6</td><td>1.2</td><td>7.0</td><td>7.3</td><td>-</td><td>-</td></tr></table>

Table 17: Performance on XOR-FULL (BLEU scores on the test data). “GS” denotes Google Search retrieval. The bottom section shows results from single Wikipedia baselines. ElasticSearch for BM25 does not support Telugu. “mDPR” denotes a DPR model where query and context encoders are initialized with multilingual BERT.

<table><tr><td colspan="2">Translation</td><td colspan="2">Retrieval</td><td colspan="7">Target Language  $L_i$ </td><td colspan="3">Macro Average</td></tr><tr><td>Query</td><td>Answer</td><td> $L_i$ </td><td>Eng.</td><td>Ar</td><td>Bn</td><td>Fi</td><td>Ja</td><td>Ko</td><td>Ru</td><td>Te</td><td>F1</td><td>EM</td><td>BLEU</td></tr><tr><td>GMT</td><td>GMT</td><td>GS</td><td>DPR</td><td>18.0</td><td>29.1</td><td>13.8</td><td>5.7</td><td>15.2</td><td>14.9</td><td>15.6</td><td>16.0</td><td>9.9</td><td>14.9</td></tr><tr><td>Our MT</td><td>Our MT</td><td>GS</td><td>DPR</td><td>17.7</td><td>4.5</td><td>13.0</td><td>5.7</td><td>15.0</td><td>14.9</td><td>8.8</td><td>11.4</td><td>6.3</td><td>10.3</td></tr><tr><td>Our MT</td><td>Our MT</td><td>BM25</td><td>DPR</td><td>9.2</td><td>15.8</td><td>14.4</td><td>4.8</td><td>7.9</td><td>5.2</td><td>0.5</td><td>8.3</td><td>4.6</td><td>7.5</td></tr><tr><td>-</td><td>GMT</td><td>GS</td><td>mDPR</td><td>17.8</td><td>15.3</td><td>12.6</td><td>5.6</td><td>15.2</td><td>15.0</td><td>10.1</td><td>13.1</td><td>7.7</td><td>12.2</td></tr></table>

Table 18: Performance on XOR-FULL (dev data F1 scores and average F1, EM and BLEU scores). “GS” denotes Google Search retrieval, and “mDPR” denotes a DPR model where query and context encoders are initialized with multilingual BERT.

<table><tr><td rowspan="2">Query Translator</td><td colspan="2">Easy</td><td colspan="2">Hard</td></tr><tr><td>R@2kt</td><td>R@5kt</td><td>R@2kt</td><td>R@5kt</td></tr><tr><td>Human</td><td>65.3</td><td>72.5</td><td>59.9</td><td>68.9</td></tr><tr><td>GMT</td><td>61.1</td><td>67.7</td><td>54.3</td><td>63.4</td></tr><tr><td>Our MT</td><td>41.9</td><td>49.9</td><td>37.7</td><td>44.5</td></tr><tr><td>Multilingual</td><td>34.3</td><td>44.3</td><td>36.1</td><td>40.9</td></tr></table>

Table 19: Macro-averaged retrieval recall on the easy and hard subsets of the development set. All configurations use DPR for retrieval. The Multilingual model avoids query translation.

hard subsets in R@2kt with human translation and DPR), suggesting that the questions from the hard subset are clearly more challenging than the ones from the easy subset.

# D.3 Qualitative Analysis on Translation Errors

One primary challenge in question translation is precisely translating key words (e.g., entities, year); our MT correctly translates a Japanese question, アーモンドアイはいつ生まれた? (When was Almond Eye born; Almond Eye is a Japanese popular race horse) $^{15}$ while Helsinki (Tiedemann and Thottingal, 2020) translates it to “When was almond born?” This resulted in retrieval errors, and Wikipedia articles related to almonds were selected. Intrinsic metrics such as BLEU would not consider the importance of these translation mistakes.