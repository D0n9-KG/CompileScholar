A Survey on Table-and-Text HybridQA: Definitions, Methods, Challenges and Future Directions 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY-NC-ND 4.0
 
 
arXiv:2212.13465v2 [cs.CL] 02 Feb 2023 
 
 

# 
A Survey on Table-and-Text HybridQA: 
 Definitions, Methods, Challenges and Future Directions

 
 
 Dingzirui Wang
 
 Affiliation:  Research Center for Social Computing and Information Retrieval
 
 Email:  {dzrwang 
 
    
 Longxu Dou
 
 Affiliation:  Harbin Institute of Technology, China
 
 Email:  lxdou 
 
    
 Wanxiang Che
 
 Email:  car}@ir.hit.edu.cn 
 

 Abstract 
 
 Table-and-text hybrid question answering (HybridQA) is a widely used and challenging NLP task commonly applied in the financial and scientific domain.
The early research focuses on migrating other QA task methods to HybridQA, while with further research, more and more HybridQA-specific methods have been present.
With the rapid development of HybridQA, the systematic survey is still under-explored to summarize the main techniques and advance further research.
So we present this work to summarize the current HybridQA benchmarks and methods, then analyze the challenges and future directions of this task.
The contributions of this paper can be summarized in three folds:
(1) first survey , to our best knowledge, including benchmarks, methods, and challenges for HybridQA;
(2) systematic investigation with the reasonable comparison of the existing systems to articulate their advantages and shortcomings;
(3) detailed analysis of challenges in four important dimensions to shed light on future directions.

 
 
 

## 1 Introduction

 
 The question-answering task with just text or tables to generate answers has been systematically studied, which we call the classic QA task.
These two sorts of evidence each have their advantages:
textual evidence is prevalent in daily communication, while tabular evidence is a well-organized display of numerical information.
However, using the heterogeneous data that combines these two types of evidence is increasingly prevalent in real applications, particularly in fields demanding numerical reasoning, like the financial and scientific domains.
This technique is known as Table-and-Text Hybrid Question Answering (HybridQA).
To fill the gap of HybridQA research, an increasing number of benchmarks Chen et al. (2020b) ; Zhu et al. (2021a) and associated methods have been present in recent years, drawing more and more attention to this task.
Considering that HybridQA is still under-researched and lacks a comprehensive survey, we present this paper to summarize the current development and help researchers get into this topic.

 
 
 The HybridQA task requires the system to generate the answer to the question based on heterogeneous knowledge, including tables and text.
Compared with the classic QA task, the HybridQA task requires the system to model these two types of evidence, which makes it harder to obtain the correct answers.

 
 {forest} 
 Figure 1: Summary of HybridQA task. 
 
 
 To advance these emerging and important research topics, several high-quality HybridQA benchmarks have been proposed.
For example, HybridQA Chen et al. (2020b) mainly focuses on evidence extraction (i.e., finding the grounded truth from hundreds of evidence candidates).
In contrast, TAT-QA Zhu et al. (2021a) concentrates on numerical reasoning (e.g., aggregation and sorting) in the hybrid context.

 
 
 According to these benchmarks, we systematically summarize the challenges of HybridQA to deepen our understanding of the task, thereby inspiring more research ideas.
Although different HybridQA benchmarks have significantly different settings, the core challenges of all benchmarks are the same, which make up the main challenges of the HybridQA task.
Concretely, we summarize four main challenges: retrieval effectiveness and efficiency, cell location of tabular evidence, relation modeling of heterogeneous evidence, and multi-hop reasoning.

 
 
 To handle these challenges, current research introduces several effective methods.
Like classic QA systems with homogeneous evidence, the HybridQA system can also be divided into retriever and reader modules Zhu et al. (2021b) .
Most retrievers employ the pre-train language model (PLM) as the encoder, while the difference is the table retrieval granularity and whether to link heterogeneous evidence.
About the reader, some approaches use machine reading comprehension (MRC) Chen et al. (2020b) ; Kumar et al. (2021) , while other systems are designed for HybridQA-specific challenges Li et al. (2022b) ; Lei et al. (2022) , which can be divided into encoder, decoder, and data manipulation.

 
 
 Although the current methods have achieved remarkable improvement, they cannot completely solve all HybridQA challenges.
To shed light on HybridQA, we discuss several promising research directions in the future, which include relation-modeling improvement, specific-knowledge injection, data augmentation, and context enrichment.
We hope these directions can make significant progress on the HybridQA task because they have been proven effective on many other tasks Nakano et al. (2021) ; Li et al. (2022a) .

 
 
 To summarize, our contributions include

 
 • 
 
 First Survey :
To the best of our knowledge, this is the first systematic survey about the HybridQA task with five parts of summarization (Figure  1 ).

 

 • 
 
 Systematic Investigation :
We propose a reasonable comparison of the existing systems to articulate their advantages and shortcomings.

 

 • 
 
 Detailed Analysis :
We summarize the main challenges of HybridQA to deepen our understanding of this new task. Then we propose four promising future directions.

 

 
 
 
 
 
 
 Context | 
 Name | 
 NR | 
 Hypothesis | 
 Hierarchical | 
 Open-QA | 
 Multi-Turn | 
 Multi-Modal | 
 Domain | 
 #Example | 
 Answer Type | 

 
 Table-and-Text | 
 HybridQA Chen et al. (2020b) | 
 | 
 | 
 | 
 | 
 | 
 | 
 Wikipedia | 
 69,611 | 
 Spans | 

 
 OTT-QA Chen et al. (2021a) | 
 | 
 | 
 | 
 ✓ | 
 | 
 | 
 Wikipedia | 
 45,481 | 
 Spans | 

 
 GeoTSQA Li et al. (2021b) | 
 ✓ | 
 | 
 | 
 | 
 | 
 | 
 Geography | 
 1,012 | 
 Choice | 

 
 FinQA Chen et al. (2021b) | 
 ✓ | 
 | 
 | 
 | 
 | 
 | 
 Finance | 
 8,283 | 
 DSL | 

 
 TAT-QA Zhu et al. (2021a) | 
 ✓ | 
 | 
 few | 
 | 
 | 
 | 
 Finance | 
 16,552 | 
 Multi-Type | 

 
 TAT-HQA Li et al. (2022c) | 
 ✓ | 
 ✓ | 
 few | 
 | 
 | 
 | 
 Finance | 
 8,283 | 
 Multi-Type | 

 
 MultiHiertt Zhao et al. (2022) | 
 ✓ | 
 | 
 ✓ | 
 | 
 | 
 | 
 Finance | 
 10,440 | 
 Multi-Type | 

 
 MISC | 
 HybriDialogue Nakamura et al. (2022) | 
 | 
 | 
 | 
 | 
 ✓ | 
 | 
 Wikipedia | 
 21,070 | 
 Spans | 

 
 ConvFinQA Chen et al. (2022b) | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 
 Finance | 
 14,115 | 
 DSL | 

 
 PACIFIC Deng et al. (2022) | 
 ✓ | 
 | 
 | 
 | 
 ✓ | 
 | 
 Finance | 
 19,008 | 
 Multi-Type | 

 
 TAT-DQA Zhu et al. (2022) | 
 ✓ | 
 | 
 few | 
 | 
 | 
 ✓ | 
 Finance | 
 16,558 | 
 Multi-Type | 

 
 MMQA Talmor et al. (2021) | 
 | 
 | 
 | 
 | 
 | 
 ✓ | 
 Wikipedia | 
 29,918 | 
 Spans | 

 

 Table 1: 
 NR: Whether the questions require numerical reasoning.
 Hypothesis: Whether the questions include the hypothesis.
 Hierarchical: Whether the tables are the hierarchical structure.
 Open-QA: Whether the benchmark is the open-QA benchmark.
 Domain: The domain of background information in the benchmark.
 #Example: Number of examples.
 Answer Type: Covered answer types.
 
 
 
 

## 2 Background

 
 In this section, we will propose the task definition and introduce two widely used evaluation metrics.

 
 

### 2.1 Task Definition

 
 Figure 2: 
An example of HybridQA.
The question and answer (with an arithmetic formula) are shown on the right.
The required evidence for answering this question is bold on the left.
 
 
 
 The HybridQA model first takes a question as the input, retrieves the relevant tables and passages as the knowledge evidence, and generates a free-formed answer as the task output with reasoning over this retrieved evidence.
The type of answer includes
 (1) the single/multi-span, from either the table cells or the passages;
 (2) the calculation result, for arithmetic question;
 (3) the choice result, following the same set of the multi-choice QA task.

 
 
 Each HybridQA data example ( Q , A , P , T , P ^ , T ^ ) (Q,A,P,T,\hat{P},\hat{T}) consists of the question Q Q , the answer A A , the passages and tables ( P , T ) (P,T) , the corresponding textual and tabular evidences ( P ^ , T ^ ) (\hat{P},\hat{T}) .
In some benchmarks, the answer A A is also affiliated with the arithmetic formula.

 
 
 In the case of Figure  2 , given the question ‘What is the ratio of the revenue in 2019 to the total revenue?’ , the model first retrieves the relevant evidence, then generate formula ‘38,100 / 125,843’ by reasoning over the evidence, and finally calculates the answer ‘30.28%’ based on the formula.

 
 
 

### 2.2 Evaluation Metrics

 
 The most common HybridQA evaluation metrics include

 
 • 
 
 Exact Match measures the percentage of predictions that match the ground truth answers.
Usually, two arithmetic answers are considered equal if their four decimal places are equal, following the rule of rounding function.

 

 • 
 
 Numeracy-Focused F 1 Score Zhu et al. (2021a) measures the average token-level overlap between the predictions and the ground truth answers, which can reduce the false negative labeling.
When an answer has multiple spans, the numeracy-focused F 1 performs a one-to-one alignment greedily based on the bag-of-word overlap on the set spans to ensure every current span can get the highest F 1 value, then compute micro-average F 1 over each span.

 

 
 
 
 
 

## 3 Benchmarks

 
 The HybridQA systems are driven by high-quality and large-quantity datasets.
In this section, we introduce the widely-used HybridQA benchmarks. which are summarized in Table  1 .

 
 
 HybridQA: 

 
 HybridQA Chen et al. (2020b) is the first HybridQA benchmark, which is also the largest cross-domain benchmark to date.
Each question and answer is relayed on a single table and multiple texts.
Each text usually is a description of information of a table cell, for example, a hyperlink page of the cell, which is crawled from Wikipedia.
For each case, the benchmark offers the golden text and table rows.
All answers to questions are the spans in evidence, which called span-based answers, and need one or more hops between heterogeneous data.

 
 
 
 OTT-QA: 

 
 To lower the difficulties of answering, HybridQA Chen et al. (2020b) annotates the related evidence to each example and the links of text and tables, which widens the gap with real-world applications.
To be more relevant to the practical applications, OTT-QA Chen et al. (2021a) blends textual and tabular evidence of each example into one single corpus that contains more than five million items and removes the relation information between them, which is called the open-QA benchmark.
So the most challenging part of this benchmark is to retrieve evidence of questions from millions of heterogeneous data, like open domain question answering.
The questions and evidence of OTT-QA are all built based on the HybridQA.
Also, all its answers are the spans in the evidence.

 
 
 
 FinQA: 

 
 Some HybridQA answers generation require numeric reasoning compatibility, while the benchmarks with only span-based questions cannot fulfill this requirement.
FinQA Chen et al. (2021b) is a finance HybridQA benchmark containing the questions of many standard financial analysis calculations.
FinQA annotates the arithmetic answer in a domain-specific language (DSL), which consists of mathematical and table operations, to reduce the difficulty of formula generation and make it more interpretable.

 
 
 
 TAT-QA: 

 
 Although FinQA has presented well-annotated numerical reasoning questions, it ignores the questions with span-based answers.
Similar to the classic QA benchmark DROP Dua et al. (2019) , TAT-QA Zhu et al. (2021a) is a collection of financial HybridQA samples that includes questions with both span-based and arithmetic answers.
Additionally, unlike the benchmarks mentioned above, each TAT-QA question is typically related to only five texts, which lowers the difficulty of retrieval.
Just like FinQA, TAT-QA also provides the formulations of arithmetic questions.

 
 
 
 TAT-HQA: 

 
 In real applications, there exist many problems requiring hypothesis, for example, “ what is the balance of 2019 if the growth is the same as 2018? ”.
TAT-HQA Li et al. (2022c) is a variant of TAT-QA Chen et al. (2020b) to simulate the hypothetical scenario questions by introducing assumptions based on the evidence.
The benchmark also annotates the hypothesis portion of each question in order to lessen the difficulty of model learning.

 
 
 
 MultiHiertt: 

 
 Hierarchical tables, which contain multi-level headers, are common in the real world but are hard to be expressed and be understood by models because of the complex table structure.
However, almost all tables of the previous benchmarks are flattened structures without multi-level headers.
To overcome this challenge, MultiHiertt Zhao et al. (2022) collects and annotates many hierarchical tables compared with questions.

 
 
 
 GeoTSQA: 

 
 GeoTSQA Li et al. (2021b) is the first scenario-based question-answering benchmark with hybrid evidence, which requires retrieving and integrating knowledge from multiple sources and applying general knowledge to a specific case described by the scenario.
This benchmark is constructed on the multiple-choice questions in the geography domain from Chinese high-school exams.
Besides tables and text, each question is also provided with four options, from which model should select one as the answer.

 
 
 
 MISC: 

 
 To adapt the HybridQA task in more realistic applications compared with the benchmarks above, another group of benchmarks extends the vanilla HybridQA to more challenging scenarios,
including the multi-turn Nakamura et al. (2022) ; Chen et al. (2022b) ; Deng et al. (2022) and the multi-modal evidence Zhu et al. (2022) ; Talmor et al. (2021) .
However, since this survey mainly focuses on the HybridQA setting, we will not discuss these specific settings in the following. Please refer to the corresponding papers for more details as listed in Table  1 .

 
 
 Figure 3: 
An illustration of the mainstream HybridQA system.
 
 
 
 
 

## 4 Challenges

 
 We can see from the datasets presented above that while having various settings, they all face similar difficulties, which are the main topics of the HybridQA approaches.
In the following, we identify four particular challenges of the HybridQA task and summarize them by illustrating (1) why is the challenge essential? and (2) why is it so challenging? 

 
 
 Trade-off between Retrieval Effectiveness and Efficiency. 

 
 The large-scale evidence could not be directly fed into the model due to limited input length.
Thus we should retrieve the most relevant evidence first.
Effectiveness and efficiency are vital attributes in the retrieval problem Karpukhin et al. (2020) .
Specifically, in the HybridQA problem,
(1) effectiveness (i.e., retrieval accuracy) would eventually determine the upper bound of QA accuracy and
(2) efficiency (i.e., retrieval latency) is essential for user experience in practical applications.
However, these two attributes are difficult to optimize simultaneously.
It is because effective modeling usually needs deep semantic interaction and complex indexing, which would inevitably increase the cost of encoding.

 
 
 
 Locating Tabular Evidence with Structure Modeling. 

 
 Tabular cell location requires the system to detect the cells related to the question.
While it’s tricky to understand the complex table structure.
The models that could only capture the linear structure are difficulty encoding tables with complicated hierarchical structures.
In addition, the question may not directly contain the related table header names, and the model needs to map the entities in the question to the correct cells based on semantics.

 
 
 
 Relation Modeling of Heterogeneous Evidence. 

 
 The relation modeling identifies the relations of different evidence, which is the basis of many reasoning steps, such as multi-hop and numerical reasoning.
The challenge is that the relations of the HybridQA task are more complex than the classic QA task, such as the positional relations in tables and the relations between tables and text.
Thus the system should model the relations between heterogeneous evidence rightly and reduce the number of input relations that are not relevant to the question.
In particular, two related pieces of evidence may only be semantically similar but not contain the same entity, which leads to the loss of effectiveness of traditional semantic matching methods.

 
 
 
 Multi-Hop Reasoning. 

 
 Many real-world questions cannot be answered in one step but require multi-step reasoning on considerable evidence.
The multi-hop reasoning is the method to solve this problem by obtaining the answer from different pieces of evidence based on the relations above.
While the difficulty is that most multi-hop questions do not provide the solving process Chen et al. (2021b) ; Zhu et al. (2021a) , which makes it hard to detect evidence reasoning order.
Besides, the multi-hop reasoning question can be seen as the multi-turn questions mixed in one question, which requires the system to decompose it during encoding or to generate sub-questions during decoding.

 
 
 
 

## 5 Methods

 
 In this section, we introduce the recent progress of HybridQA in solving the challenges which are discussed in § \lx@sectionsign 4 .
Following the definition of open-domain question answering Zhu et al. (2021b) , we divide the HybridQA system into two modules called Retriever ( § \lx@sectionsign 5.1 ) and Reader ( § \lx@sectionsign 5.2 ).
The process of the HybridQA system can be summarized in Figure  3 .
Notably, although some work Kumar et al. (2021) build a unified framework to couple these two modules, to make it more clear, we still follow the mainstream work treating them as two separate modules.

 
 
 
 
 
 | 
 Cell-Based | 
 Row-Based | 
 Table-Based | 

 
 Individual | 
 POINTR Eisenschlos et al. (2021) | 
 DuRePa Li et al. (2021a) | 
 | 

 
 MT2Net Zhao et al. (2022) | 
 | 

 
 Joint | 
 Hybrider Chen et al. (2020b) | 
 MITQA Kumar et al. (2021) | 
 FR+CBR Chen et al. (2021a) | 

 
 CARP Zhong et al. (2022) | 
 DocHopper Sun et al. (2022) | 

 
 CORE Ma et al. (2022) | 
 TTGen Li et al. (2021b) | 

 
 MuGER Wang et al. (2022) | 
 OTTeR Huang et al. (2022) | 

 

 Table 2: 
Retrievers of HybridQA system.
 
 
 

### 5.1 Retriever

 
 The goal of the retriever is to find a set of evidence to induce the QA answer.
Some HybridQA systems  Eisenschlos et al. (2021) ; Chen et al. (2020b) ; Kumar et al. (2021) ; Sun et al. (2022) directly employ the two-stage retrieval methods from open-domain QA task (i.e., classic QA).
They first adopt TF-IDF or BM25 Robertson et al. (2009) to filter out a fixed number of data  (coarse-grained ranking) , then employ PLM to calculate the relevancy between the question and the filtered evidence  Zhu et al. (2021b)   (fine-grained ranking) .
The number of retrieval data in the first stage influences the trade-off between effectiveness and efficiency of retrieval Chen et al. (2021a) ; Zhong et al. (2022) ; Chen et al. (2020b) .

 
 
 Besides these general retrievers that are directly inherited from the classic QA, many works advance the retriever to be HybridQA-specific for handling heterogeneous evidence.
For instance, we could leverage
 (1) the relations between textual and tabular evidence and
 (2) the tabular operation results just like SQL aggregate functions.
In the following, we classify the current HybridQA retrievers from two aspects:
 (1) the relations of the textual and tabular evidence, including Individually, Jointly and
 (2) the table retrieval granularity, including Cell-Based, Row-Based, and Table-Based.
The classification of retrievers about these two category methods is summarized in Table  2 .
Finally, we make a comparison of different types of retrievers.

 
 

#### 5.1.1 5.1.1 Individual

 
 The individual retrievers consider textual and tabular evidence as two separate parts.
This type of retriever can reduce the overhead of data processing and try to avoid cascading errors.

 
 
 Cell-Based: 

 
 Directly retrieving table cells is the simplest retrieval technique.
POINTR Eisenschlos et al. (2021) uses a creative table Transformer architecture to feed every table cell into the model.
However, this approach forces the model to figure out the relationships between various cells on its own, making learning more challenging.
To mitigate this problem, MT2Net Zhao et al. (2022) models the cell relations by transferring each table into multiple sentences as the retrieval items, which describe the numerical relation of the table.

 
 
 
 Row-Based: 

 
 To maintain the table structure information (e.g., cells in the same row), utilizing rows as retrieval units is another method for maintaining cell relations.
For instance, DuRePa Li et al. (2021a) linearizes and ranks every table row, where a specific token distinguishes cells of each row.

 
 
 
 

#### 5.1.2 5.1.2 Joint

 
 The joint retrievers merge heterogeneous evidence as input.
This kind of retriever takes into account the relations between the various kinds of evidence, and some of them even make these relations retrievable.

 
 
 Cell-Based: 

 
 Some benchmarks Chen et al. (2020b) have provided text and table relationships that the model can directly adapt.
Hybrider Chen et al. (2020b) connects cells with related text and question entities, and then every cell is scored and combined with that in the same row.
Yet not all provided relations are useful, so MuGER Wang et al. (2022) sees the linking also as the retrievable objects, just like cells or rows.
To help models learn the process of reasoning, CARP Zhong et al. (2022) , and CORE Ma et al. (2022) search the relations between question entities, text entities, and table cells and try to select the golden linking path.

 
 
 
 Row-Based: 

 
 Just like DuRePa Li et al. (2021a) , MITQA Kumar et al. (2021) and DocHopper Sun et al. (2022) directly concatenate rows with related textual evidence as the retriever input.
Taking into account that cell-based and row-based have different advantages, TTGen Li et al. (2021b) tries to mix two types of granularity to retrieve, which transfers table operations into sentences, such as extremum values and monotonic change, then ranks them with the question and related paragraphs.
To recover the linking between tables and rows, OTTeR Huang et al. (2022) firstly links rows and text as blocks like OTT-QA baseline Chen et al. (2021a) , then uses a modified dense retriever to obtain useful blocks.

 
 
 
 Table-Based: 

 
 Considering the prior knowledge that every text is only associated with one table, the Fusion Retriever and Cross-Block Reader system (FR+CBR), which is the baseline system of OTT-QA Chen et al. (2021a) , presents the table-text group called block, which contains one table and several related texts.
All possible blocks will be flattened and retrieved to be the input of the following modules.

 
 
 
 

#### 5.1.3 5.1.3 Comparison

 
 Individual vs. Joint: 

 
 Because the relationships of different types of evidence can help the retriever comprehend heterogeneous knowledge, the joint retriever is adapted by the majority of current systems.
However, not all application scenarios and benchmarks provide the entities linked to the questions Zhu et al. (2021a) ; Chen et al. (2021b) , and joint retrievers must establish the linking themselves.
This could introduce incorrect information and result in a cascade of errors.
Individual retrievers work well on tiny-scale corpus benchmarks, which ask the reader to model relations instead of the retriever.

 
 
 
 Cell-Based vs. Row-Based vs. Table-Based: 

 
 The best is still debatable due to the benefits and drawbacks of various retrieval granularities.
Since HybridQA answers frequently incorporate information from single or multiple cells, cell-based retrievers can overlook a large amount of useless data.
However, they are unable to simulate the cross-row or cross-column information of the table structure.
The row-based retriever can better represent the row relation than the cell-based one, but it is still challenging to model the information about the column structure.
When there are too many words in a sequence, there will be a lot of useless information, and it will be difficult to add additional important information.
The only option to manage massive amounts of table data is to use a table-based retriever because it is difficult to divide each table into cells or rows.

 
 
 
 
 
 
 Reader | 
 Method | 
 System | 

 
 Encoder-Improvement | 
 Relation Modeling | 
 DEHG Feng et al. (2022) | 

 
 RegHNT Lei et al. (2022) | 

 
 Knowledge Injection | 
 TTGen Li et al. (2021b) | 

 
 KIQA Nararatwong et al. (2022) | 

 
 Decoder-Improvement | 
 Multi-Tower Decoding | 
 TagOp Zhu et al. (2021a) | 

 
 FinMath Li et al. (2022b) | 

 
 MT2Net Zhao et al. (2022) | 

 
 L2I Li et al. (2022c) | 

 
 Multi-Hop Reasoning | 
 DocHopper Sun et al. (2022) | 

 
 CARP Zhong et al. (2022) | 

 
 CORE Ma et al. (2022) | 

 
 | 
 ReasonFuse Xia et al. (2022) | 

 
 | 
 ELASTIC Zhang and Moshfeghi (2022) | 

 
 Data-Manipulation | 
 Pre-Training | 
 PoEt Pi et al. (2022) | 

 
 Format Modification | 
 TaCube Zhou et al. (2022a) | 

 
 DuRePa Li et al. (2021a) | 

 
 FinQANet Chen et al. (2021b) | 

 
 UniRPG Zhou et al. (2022b) | 

 

 
 Table 3: 
Readers of HybridQA system.
 
 
 
 
 
 

### 5.2 Reader

 
 Given the question and retrieved evidence, the role of the reader is to extract or generate the answer.
In the classic QA task, the systems use the MRC modules as their readers Zhu et al. (2021b) .
Some HybridQA systems also employ the MRC modules as the readers for the span-based answers benchmarks Chen et al. (2020b) ; Chen et al. (2021a) ; Eisenschlos et al. (2021) ; Zhong et al. (2022) .
Many other systems have designed HybridQA-specific functions for the readers, such as the domain knowledge injection Nararatwong et al. (2022) and the relation modeling Lei et al. (2022) , which is the main topic of this section.

 
 
 In the following, we will introduce the current HybridQA system readers.
We adopt the popular encoder-decoder framework to describe the current reader structure, where a series of works focused on improving the performance of the two modules.
Besides, another series of work concentrates on the data instead of the model, which we called Data-Manipulation and summarized in Figure  4 .
In summary, we categorize the improvement methods of the HybridQA reader into Encoder-Improvement, Decoder-Improvement, and Data-Manipulation.

 
 

#### 5.2.1 5.2.1 Encoder-Improvement

 
 The encoder is the component that transforms the input into a format the model can understand.
Relation modeling and knowledge injection are just two of the techniques used in this section to increase the efficiency of encoders.
Knowledge injection focuses on how to teach the model with new information, while relation modeling focuses on how to construct the relationship of the evidence.

 
 
 Relation Modeling: 

 
 In contrast to traditional QA tasks, HybridQA data encoding requires the capacity to comprehend relationships.
Thus DEHG Feng et al. (2022) introduces the relation graph to the HybridQA task, which first builds the graph of the evidence and the question according to the established rules, then linearizes the relation graph as an input.
However, DEHG is required to recover the graph by the linearized tuples on its own, which increases the difficulty of relation understanding.
To better use graphs, RegHNT Lei et al. (2022) directly builds the graph network with model structure and adapts the relation-aware attention mechanism to encode different relation types.

 
 
 
 Knowledge Injection: 

 
 A specific domain leads to specific knowledge that may not be mentioned in training data.
TTGen Li et al. (2021b) uses K-BERT Liu et al. (2020) to obtain the representation for each input token fused with the domain-specific knowledge, which is injected by the pre-training.
Because the knowledge of Wikipedia is organized with entities (Wikipedia pages), KIQA Nararatwong et al. (2022) adapts more fine-grained knowledge injection, which firstly extracts entities from the question and evidence, then injects the knowledge about these entities into the model with the usage of LUKE Yamada et al. (2020) .

 
 
 
 

#### 5.2.2 5.2.2 Decoder-Improvement:

 
 The decoder is the module to generate various types of answers to the questions based on the representation by the encoder.
The methods of this part aim to improve the performance of the decoder, which includes multi-tower decoding and multi-hop reasoning.
The multi-tower decoding focuses on designing different decoder structures for different answer types, while the latter improves the performance of multi-hop reasoning.

 
 
 Multi-Tower Decoding: 

 
 One of the challenges of HybridQA data decoding is generating widely different types of answers, such as the span-based and the arithmetic answers.
TagOp Zhu et al. (2021a) adopts a linear layer to decide the answer type of every example and determine which input tokens will be used, then generate the answer with the decoder corresponding to its type.
Considering that the formula of arithmetic type answers can be represented in the tree format, FinMath Li et al. (2022b) uses Seq2Tree Xie and Sun (2019) as the decoder to transfer a formula into a calculation tree, which is widely used in math word problem tasks.
To address error cascades from the wrong type prediction, MT2Net Zhao et al. (2022) predicts the probability of each type and span being the answer and selects the result with the highest joint probability as the final answer.

 
 
 
 Multi-Hop Reasoning: 

 
 Multi-hop is another decoder challenge, which requires the decoder to capture the chain of reasoning.
CARP Pi et al. (2022) , and CORE Ma et al. (2022) build relation linking during retrieval, and the decoder needs to detect the linking path from question to answer spans.
The difference is that the latter uses a better method of generating and filtering relations.
To simulate the process of humans solving multi-hop questions, DocHopper Sun et al. (2022) employs an iterative decoder that generates the question after one hop until obtaining the result.
With the same motivation, ReansonFuse Xia et al. (2022) adapts multiple LSTM decoding processes to ensure that the decoder holds encoded information and can use previous decoded results.
Different from the above methods handle multiple types of multi-hop reasoning, ELASTIC Zhang and Moshfeghi (2022) mainly focuses on arithmetic questions, which generate the operators and operands step by step.

 
 
 
 

#### 5.2.3 5.2.3 Data-Manipulation

 
 Figure 4: 
An illustration of the Data-Manipulation.
 
 
 
 Data manipulation, including pre-training and data format modification, is a series of simple but effective methods to enhance the table understanding of the model by adjusting the format and scale of the data without modifying the model structures.

 
 
 Pre-Training: 

 
 PoEt Pi et al. (2022) adopts SQL execution task as a proxy of numerical reasoning task, considering the code execution and language reasoning share a similar protocol.
For example, the model can learn the numerical reasoning capability by executing arithmetic formulations.
Especially, PoEt first fine-tunes the pre-trained language model (e.g., BART and T5) on the synthesized code-related data, then fine-tunes it on the original HybridQA dataset.

 
 
 
 Format Modification: 

 
 Considering that the ability of PLM to understand data in various formats is different, modifying the input and output formats of the reader can significantly improve the ability to understand tables and generate answers.
TaCube Zhou et al. (2022a) pre-generates information about tables, like the extremum value of every column called the cube.
To avoid the embarrassment of table processing, DuRePa Li et al. (2021a) generates SQL as well as answers and decides which one to use.
Considering the poor numerical computing power of PLM, FinQANet Chen et al. (2021b) generates domain-specific language instead of the arithmetic results provided by FinQA.
While UniRPG Zhou et al. (2022b) extends numerical reasoning programs from tables to text, which can also use text spans as computation values.

 
 
 
 
 
 

## 6 Future Directions

 
 As introduced above, the existing systems have almost addressed the main challenges ( § \lx@sectionsign 4 ).
Advanced techniques are being researched that could improve the current system by making it more powerful and reliable.
In the following, we will discuss these future directions by demonstrating (1) why are these techniques promising? and (2) what is the tricky part in implementation? 

 
 
 Improve Relation Modeling: 

 
 Relation modeling is the basis of functions such as evidence retrieval and multi-hop question answering in HybridQA tasks, but the current system still has many deficiencies.
Although current methods have noticed different relations based on table structure Lei et al. (2022) , numerical comparisons and table header inclusion relations are still ignored.
Most current systems use token-based methods to model evidence relations.
However, the neural network methods can extract relations with higher accuracy Kolitsas et al. (2018) .
With the relation modeling, current systems fed them into the retrievers and readers as the input.
At the same time, the relations can also be used to spread information through multiple iterations for multi-hop questions Chen et al. (2020a) .

 
 
 
 Inject Domain Knowledge for Professional Experts: 

 
 In the realistic scenario, the HybridQA system should handle many domain-specific questions requiring specific knowledge.
Injecting knowledge effectively mitigates the lack of domain-specific knowledge Yamada et al. (2020) of the HybridQA task.
However, domain knowledge requires a specific format explaining entities, like a knowledge base or Wikipedia page, which need much human involvement.
A possible solution is to use a text-based web browsing environment Nakano et al. (2021) , and another solution is to transfer a document to the knowledge with the QA system Chen et al. (2022a) .
The current methods first train a module with injection knowledge Nararatwong et al. (2022) ; Li et al. (2021b) , while this approach may ignore some information because the model does not know which knowledge will be used in the actual Q A.
So to make better use of the domain knowledge, after receiving a question, the model retrieves the knowledge related to the question, like a paragraph in the corpus or a node in the knowledge base, and then concatenates it after the question as the model inputs.

 
 
 
 Data-Augmentation for Handling Data-Scarcity: 

 
 The current scales of HybridQA benchmarks are not large because of the labeling difficulty, which limits the performance of the current systems.
Data augmentation is an effective strategy to improve the performance of the system by automatically expanding the data scale Li et al. (2022a) , while it has not been systematically applied in the HybridQA task.
A simple but effective augmentation method is to populate the templates with entities from the evidence Yoran et al. (2022) .
Another method is to generate the question based on the given answer and evidence with the seq-to-seq model Chen et al. (2021a) .
The augmentation data can be directly used as training data.
Considering the noise in this data, a more convenient method is to design the pre-training tasks for specific compatibility of HybridQA systems Zhong et al. (2022) .

 
 
 
 Enrich Context Modeling for Realistic Scenario: 

 
 Most current research concentrates on simple context, i.e., single-turn questions and plain (text, table) evidence.
While in a realistic scenario, we could consider more complex context settings, such as multi-turn questioning and visually-rich evidence.
Concretely, for multi-turn setting Nakamura et al. (2022) ; Chen et al. (2022b) ; Deng et al. (2022) , it’s more natural and informative in expressing the complex intent with follow-up questions rather than one single-turn question.
But it’s challenging to address the ellipsis and resolution problem in a multi-turn setting.
For visually-rich evidence setting  Talmor et al. (2021) ; Zhu et al. (2022) , the visual document exhibits the positional relation of (text, table) .
Intuitively, leveraging this type of visual relation would definitely improve the context modeling ability.
Thus, it is worth exploring these promising context modeling for building an effective HybridQA model in realistic applications.

 
 
 
 

## 7 Conclusion

 
 In this paper, we comprehensively summarize and analyze the existing HybridQA task.
Firstly, we study twelve mainstream HybridQA benchmarks, including their experiment settings and application scenarios.
Considering that the solutions of these benchmarks have strong commonalities, we analyze their most common challenges, which are also the main challenges of the HybridQA task.
To demonstrate the current progress in addressing these challenges, we outline the present methods for them, compared with their advantages and disadvantages.
Although current approaches have improved systems in many aspects, improvements in some parts are still very insufficient.
So at the end of this paper, we propose four important but under-explored directions of the HybridQA task to inspire a more robust and reliable HybridQA system.

 
 
 
 

## References

 
 
 Chen et al. [2020a] 
 
Kunlong Chen, Weidi Xu, Xingyi Cheng, Zou Xiaochuan, Yuyu Zhang, Le Song,
Taifeng Wang, Yuan Qi, and Wei Chu.

 
 Question directed graph attention network for numerical reasoning
over text.

 
 In Proc. of EMNLP , 2020.

 

 
 Chen et al. [2020b] 
 
Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang
Wang.

 
 HybridQA: A dataset of multi-hop question answering over tabular
and textual data.

 
 In Proc. of EMNLP Findings , 2020.

 

 
 Chen et al. [2021a] 
 
Wenhu Chen, Ming-Wei Chang, Eva Schlinger, William Yang Wang, and William W.
Cohen.

 
 Open question answering over tables and text.

 
 In Proc. of ICLR , 2021.

 

 
 Chen et al. [2021b] 
 
Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan
Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and
William Yang Wang.

 
 FinQA: A dataset of numerical reasoning over financial data.

 
 In Proc. of EMNLP , 2021.

 

 
 Chen et al. [2022a] 
 
Wenhu Chen, William W. Cohen, Michiel de Jong, Nitish Gupta, Alessandro Presta,
Pat Verga, and John Wieting.

 
 Qa is the new kr: Question-answer pairs as knowledge bases.

 
 ArXiv , 2022.

 

 
 Chen et al. [2022b] 
 
Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and
William Yang Wang.

 
 Convfinqa: Exploring the chain of numerical reasoning in
conversational finance question answering, 2022.

 

 
 Deng et al. [2022] 
 
Yang Deng, Wenqiang Lei, Wenxuan Zhang, Wai Lam, and Tat-Seng Chua.

 
 Pacific: Towards proactive conversational question answering over
tabular and textual data in finance.

 
 ArXiv , 2022.

 

 
 Dua et al. [2019] 
 
Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and
Matt Gardner.

 
 DROP: A reading comprehension benchmark requiring discrete
reasoning over paragraphs.

 
 In Proc. of NAACL , 2019.

 

 
 Eisenschlos et al. [2021] 
 
Julian Eisenschlos, Maharshi Gor, Thomas Müller, and William Cohen.

 
 MATE: Multi-view attention for table transformer efficiency.

 
 In Proc. of EMNLP , 2021.

 

 
 Feng et al. [2022] 
 
Yue Feng, Zhen Han, Mingming Sun, and Ping Li.

 
 Multi-hop open-domain question answering over structured and
unstructured knowledge.

 
 In Proc. of ACL Findings , 2022.

 

 
 Huang et al. [2022] 
 
Junjie Huang, Wanjun Zhong, Qian Liu, Ming Gong, Daxin Jiang, and Nan Duan.

 
 Mixed-modality representation learning and pre-training for joint
table-and-text retrieval in openqa.

 
 arXiv preprint arXiv:2210.05197 , 2022.

 

 
 Karpukhin et al. [2020] 
 
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih.

 
 Dense passage retrieval for open-domain question answering.

 
 In Proc. of EMNLP , 2020.

 

 
 Kolitsas et al. [2018] 
 
Nikolaos Kolitsas, Octavian-Eugen Ganea, and Thomas Hofmann.

 
 End-to-end neural entity linking.

 
 In Proc. of CoNLL , 2018.

 

 
 Kumar et al. [2021] 
 
Vishwajeet Kumar, Saneem Chemmengath, Yash Gupta, Jaydeep Sen, Samarth
Bharadwaj, and Soumen Chakrabarti.

 
 Multi-instance training for question answering across table and
linked text.

 
 arXiv preprint arXiv:2112.07337 , 2021.

 

 
 Lei et al. [2022] 
 
Fangyu Lei, Shizhu He, Xiang Li, Jun Zhao, and Kang Liu.

 
 Answering numerical reasoning questions in table-text hybrid contents
with graph-based encoder and tree-based decoder.

 
 In Proc. of COLING , 2022.

 

 
 Li et al. [2021a] 
 
Alexander Hanbo Li, Patrick Ng, Peng Xu, Henghui Zhu, Zhiguo Wang, and Bing
Xiang.

 
 Dual reader-parser on hybrid textual and tabular evidence for open
domain question answering.

 
 In Proc. of ACL , 2021.

 

 
 Li et al. [2021b] 
 
Xiao Li, Yawei Sun, and Gong Cheng.

 
 Tsqa: Tabular scenario based question answering.

 
 Proc. of AAAI , 2021.

 

 
 Li et al. [2022a] 
 
Bohan Li, Yutai Hou, and Wanxiang Che.

 
 Data augmentation approaches in natural language processing: A
survey.

 
 AI Open , 2022.

 

 
 Li et al. [2022b] 
 
Chenying Li, Wenbo Ye, and Yilun Zhao.

 
 FinMath: Injecting a tree-structured solver for question
answering over financial reports.

 
 In Proc. of LREC , 2022.

 

 
 Li et al. [2022c] 
 
Moxin Li, Fuli Feng, Hanwang Zhang, Xiangnan He, Fengbin Zhu, and Tat-Seng
Chua.

 
 Learning to imagine: Integrating counterfactual thinking in neural
discrete reasoning.

 
 In Proc. of ACL , 2022.

 

 
 Liu et al. [2020] 
 
Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, and Ping
Wang.

 
 K-BERT: enabling language representation with knowledge graph.

 
 In Proc. of AAAI , 2020.

 

 
 Ma et al. [2022] 
 
Kaixin Ma, Hao Cheng, Xiaodong Liu, Eric Nyberg, and Jianfeng Gao.

 
 Open-domain question answering via chain of reasoning over
heterogeneous knowledge.

 
 arXiv preprint arXiv:2210.12338 , 2022.

 

 
 Nakamura et al. [2022] 
 
Kai Nakamura, Sharon Levy, Yi-Lin Tuan, Wenhu Chen, and William Yang Wang.

 
 HybriDialogue: An information-seeking dialogue dataset grounded
on tabular and textual data.

 
 In Proc. of ACL Findings , 2022.

 

 
 Nakano et al. [2021] 
 
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina
Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders,
et al.

 
 Webgpt: Browser-assisted question-answering with human feedback.

 
 arXiv preprint arXiv:2112.09332 , 2021.

 

 
 Nararatwong et al. [2022] 
 
Rungsiman Nararatwong, Natthawut Kertkeidkachorn, and Ryutaro Ichise.

 
 KIQA: Knowledge-infused question answering model for financial
table-text data.

 
 In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The
3rd Workshop on Knowledge Extraction and Integration for Deep Learning
Architectures , 2022.

 

 
 Pi et al. [2022] 
 
Xinyu Pi, Qian Liu, Bei Chen, Morteza Ziyadi, Zeqi Lin, Yan Gao, Qiang Fu,
Jian-Guang Lou, and Weizhu Chen.

 
 Reasoning like program executors.

 
 arXiv preprint arXiv:2201.11473 , 2022.

 

 
 Robertson et al. [2009] 
 
Stephen Robertson, Hugo Zaragoza, et al.

 
 The probabilistic relevance framework: Bm25 and beyond.

 
 Foundations and Trends® in Information
Retrieval , 3(4):333–389, 2009.

 

 
 Sun et al. [2022] 
 
Haitian Sun, William W. Cohen, and Ruslan Salakhutdinov.

 
 Iterative hierarchical attention for answering complex questions over
long documents, 2022.

 

 
 Talmor et al. [2021] 
 
Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai,
Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant.

 
 Multimodal{qa}: complex question answering over text, tables
and images.

 
 In Proc. of ICLR , 2021.

 

 
 Wang et al. [2022] 
 
Yingyao Wang, Junwei Bao, Chaoqun Duan, Youzheng Wu, Xiaodong He, and Tiejun
Zhao.

 
 Muger: Multi-granularity evidence retrieval and reasoning for hybrid
question answering.

 
 arXiv preprint arXiv:2210.10350 , 2022.

 

 
 Xia et al. [2022] 
 
Yuancheng Xia, Feng Li, Qing Liu, Li Jin, Zequn Zhang, Xian Sun, and Lixu Shao.

 
 Reasonfuse: Reason path driven and global-local fusion network for
numerical table-text question answering.

 
 Neurocomputing , 2022.

 

 
 Xie and Sun [2019] 
 
Zhipeng Xie and Shichao Sun.

 
 A goal-driven tree-structured neural model for math word problems.

 
 In Proc. of IJCAI , 2019.

 

 
 Yamada et al. [2020] 
 
Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, and Yuji Matsumoto.

 
 LUKE: Deep contextualized entity representations with entity-aware
self-attention.

 
 In Proc. of EMNLP , 2020.

 

 
 Yoran et al. [2022] 
 
Ori Yoran, Alon Talmor, and Jonathan Berant.

 
 Turning tables: Generating examples from semi-structured tables for
endowing language models with reasoning skills.

 
 In Proc. of ACL , 2022.

 

 
 Zhang and Moshfeghi [2022] 
 
Jiaxin Zhang and Yashar Moshfeghi.

 
 ELASTIC: Numerical reasoning with adaptive symbolic compiler.

 
 In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho,
editors, Advances in Neural Information Processing Systems , 2022.

 

 
 Zhao et al. [2022] 
 
Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang.

 
 MultiHiertt: Numerical reasoning over multi hierarchical tabular
and textual data.

 
 In Proc. of ACL , 2022.

 

 
 Zhong et al. [2022] 
 
Wanjun Zhong, Junjie Huang, Qian Liu, Ming Zhou, Jiahai Wang, Jian Yin, and Nan
Duan.

 
 Reasoning over hybrid chain for table-and-text open domain question
answering.

 
 In Proc. of IJCAI , 2022.

 

 
 Zhou et al. [2022a] 
 
Fan Zhou, Mengkang Hu, Haoyu Dong, Zhoujun Cheng, Shi Han, and Dongmei Zhang.

 
 Tacube: Pre-computing data cubes for answering numerical-reasoning
questions over tabular data.

 
 arXiv preprint arXiv:2205.12682 , 2022.

 

 
 Zhou et al. [2022b] 
 
Yongwei Zhou, Junwei Bao, Chaoqun Duan, Youzheng Wu, Xiaodong He, and Tiejun
Zhao.

 
 Unirpg: Unified discrete reasoning over table and text as program
generation.

 
 arXiv preprint arXiv:2210.08249 , 2022.

 

 
 Zhu et al. [2021a] 
 
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv,
Fuli Feng, and Tat-Seng Chua.

 
 TAT-QA: A question answering benchmark on a hybrid of tabular and
textual content in finance.

 
 In Proc. of ACL , 2021.

 

 
 Zhu et al. [2021b] 
 
Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and
Tat-Seng Chua.

 
 Retrieving and reading: A comprehensive survey on open-domain
question answering.

 
 arXiv preprint arXiv:2101.00774 , 2021.

 

 
 Zhu et al. [2022] 
 
Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng
Chua.

 
 Towards complex document understanding by discrete reasoning.

 
 In Proc. of ACM MM , 2022.