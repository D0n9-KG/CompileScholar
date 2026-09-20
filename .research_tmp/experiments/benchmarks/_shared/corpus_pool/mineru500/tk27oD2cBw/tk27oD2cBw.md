# The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications

Mirac Suzgun $^{a,*}$ Luke Melas-Kyriazi $^{b}$ Suproteem K. Sarkar $^{c}$ Scott Duke Kominers $^{c,d}$ Stuart M. Shieber $^{c}$

$^{a}$ Stanford University $^{b}$ Oxford University $^{c}$ Harvard University $^{d}$ a16z crypto

\*Correspondence to: msuzgun@stanford.edu

# Abstract

Innovation is a major driver of economic and social development, and information about many kinds of innovation is embedded in semi-structured data from patents and patent applications. Although the impact and novelty of innovations expressed in patent data are difficult to measure through traditional means, ML offers a promising set of techniques for evaluating novelty, summarizing contributions, and embedding semantics. In this paper, we introduce the Harvard USPTO Patent Dataset (HUPD), a large-scale, well-structured, and multi-purpose corpus of English-language patent applications filed to the United States Patent and Trademark Office (USPTO) between 2004 and 2018. With more than 4.5 million patent documents, HUPD is two to three times larger than comparable corpora. Unlike previously proposed patent datasets in NLP, HUPD contains the inventor-submitted versions of patent applications—not the final versions of granted patents—thereby allowing us to study patentability at the time of filing using NLP methods for the first time. It is also novel in its inclusion of rich structured metadata alongside the text of patent filings: By providing each application's metadata along with all of its text fields, the dataset enables researchers to perform new sets of NLP tasks that leverage variation in structured covariates. As a case study on the types of research HUPD makes possible, we introduce a new task to the NLP community—namely, binary classification of patent decisions. We additionally show the structured metadata provided in the dataset enables us to conduct explicit studies of concept shifts for this task. Finally, we demonstrate how our dataset can be used for three additional tasks: multi-class classification of patent subject areas, language modeling, and summarization. Overall, HUPD is one of the largest multi-purpose NLP datasets containing domain-specific textual data, along with well-structured bibliographic metadata, and aims to advance research extending language and classification models to diverse and dynamic real-world data distributions. $^{1}$

# 1 Introduction

Patents are key public indicators of innovation and technological advancement. They offer a simple yet powerful source for studying, measuring, and appraising innovation activity, economic growth, and emerging technology. Over the past two decades, the total number of patent applications filed to the United States Patent and Trademark Office (USPTO) per year has almost doubled. In the fiscal year 2020 alone, the USPTO received more than 650,000 patent filings, including requests for continued examinations $[53]$ . The competitive and regulatory landscape surrounding patent-driven innovation is rapidly evolving, but despite the clear focus in textual data in the field of patent analysis, it has yet to be systematically studied by the machine learning (ML) and natural-language processing (NLP) communities.

![](images/7721d5eaa254c32db548f86b82254e6514a82e7040148e0f25a8e128dc637ed8.jpg)

<details>
<summary>text_image</summary>

Title
Inventors
Appl. No.
File Date
IPC
Classification
CPC + USPC
Classification
(1) United States Patent
Rosenwald et al.
(2) United States Patent
(3) United States Patent
(4) United States Patent
(5) United States Patent
(6) United States Patent
(7) United States Patent
(8) United States Patent
(9) United States Patent
(10) United States Patent
(11) United States Patent
(12) United States Patent
(13) United States Patent
(14) United States Patent
(15) United States Patent
(16) United States Patent
(17) United States Patent
(18) United States Patent
(19) United States Patent
(20) United States Patent
(21) United States Patent
(22) United States Patent
(23) United States Patent
(24) United States Patent
(25) United States Patent
(26) United States Patent
(27) United States Patent
(28) United States Patent
(29) United States Patent
(30) United States Patent
(31) United States Patent
(32) United States Patent
(33) United States Patent
(34) United States Patent
(35) United States Patent
(36) United States Patent
(37) United States Patent
(38) United States Patent
(39) United States Patent
(40) United States Patent
(41) United States Patent
(42) United States Patent
(43) United States Patent
(44) United States Patent
(45) United States Patent
(46) United States Patent
(47) United States Patent
(48) United States Patent
(49) United States Patent
(50) United States Patent
(51) United States Patent
(52) United States Patent
(53) United States Patent
(54) United States Patent
(55) United States Patent
(56) United States Patent
(57) United States Patent
(58) United States Patent
(59) United States Patent
(60) United States Patent
(61) United States Patent
(62) United States Patent
(63) United States Patent
(64) United States Patent
(65) United States Patent
(66) United States Patent
(67) United States Patent
(68) United States Patent
(69) United States Patent
(70) United States Patent
(71) United States Patent
(72) United States Patent
(73) United States Patent
(74) United States Patent
(75) United States Patent
(76) United States Patent
(77) United States Patent
(78) United States Patent
(79) United States Patent
(80) United States Patent
(81) United States Patent
(82) United States Patent
(83) United States Patent
(84) United States Patent
(85) United States Patent
(86) United States Patent
(87) United States Patent
(88) United States Patent
(89) United States Patent
(90) United States Patent
(91) United States Patent
(92) United States Patent
(93) United States Patent
(94) United States Patent
(95) United States Patent
(96) United States Patent
(97) United States Patent
(98) United States Patent
(99) United States Patent
(100) United States Patent
</details>

Figure 1: Three pages of the pre-grant version of an example patent document (Method and Apparatus for Initiating a Transaction on a Mobile Device [Publication No: 2014-0207675 A1]). The highlighted sections show a subset of the 34 data fields that we include in the Harvard USPTO Patent Dataset.

The absence of large-scale, well-structured, and distilled patent data is a major hurdle preventing researchers and practitioners from applying ML tools to understand and explore innovation and technological change through patent text. In recent years, there have been some efforts to address this limitation, but all the patent corpora in NLP so far, including CLEF-IP 2011 [38], USPTO-2M [28], and BIGPATENT [46], are still limited in their scopes and features, as shown in Table 1. These datasets rely on text and information only for granted patents and only at the time of patent grant, contain restricted subsets of texts and fields, and typically focus on only one particular NLP task. We include a more detailed discussion of these NLP datasets, as well as other popular raw patent document repositories and search tools not produced primarily for the NLP community, in Section D.

It is thus highly desirable to have a free, publicly-available dataset that provides a wider and more encompassing repository of patent data—covering multiple sections and year periods—of not just granted patents but all patent applications, that allows more flexibility and control in data selection, and that can be appropriated for multiple experiments and investigations. With these desiderata in mind, we present the first public, large-scale, consistently-structured, and multi-purpose corpus of patent data to the NLP community, called the Harvard USPTO Patent Dataset (HUPD). The dataset contains more than 4.5 million English-language utility patent applications filed to the USPTO between 2004 and 2018, and aims to advance research efforts in both patent analysis and NLP.

HUPD distinguishes itself from prior NLP patent datasets in three key aspects. First, unlike previous datasets, it focuses on patent applications, not merely granted patents, since patent applications contain the original set of claims and descriptions of the proposed invention written by the applicant. This consideration allows us to have a consistent set of patent documents at the time of filing, avoiding dataset shift concerns that would be present in the studies of accepted and rejected patent applications at different revision stages. In fact, having access to the original versions of both accepted and rejected applications allows us to introduce a completely new task to the field—namely, the binary classification of patent decisions, wherein the goal is to predict the acceptance likelihood of a patent application at the time of submission. Second, ours is the first NLP dataset to feature multiple classes of rich textual and structural information present in patent applications. Whereas previous NLP datasets include only one or two of a patent's data fields (for example, description and abstract), HUPD contains 34 fields, including filing date, fine-grained classification codes, examiner information, and many others. The variety of information available for each patent application can enable NLP researchers to perform a wide range of tasks—such as analyzing the evolution of patent language and categories over time—that were not possible under previous NLP patent datasets. Third, HUPD uses information obtained directly from the USPTO, rather than from Google's Patent search as BIGPATENT does); it is larger than previous NLP patent datasets while still being clean, comprehensive, and well-structured.

We introduce our patent dataset with several audiences in mind. For the NLP community, the universe of patent applications—with its sheer breadth and wealth of textual substance and structured format—provides an ideal domain-specific laboratory for developing and evaluating new NLP tools. $^{2}$ From

<table><tr><td>Dataset</td><td># Docs</td><td>Title</td><td>Abst</td><td>Appl</td><td>Exam</td><td>Invt</td><td>PD</td><td>Claims</td><td>Bkgd</td><td>Dsc</td><td>PCs</td><td>Years</td><td>Primary Purpose</td></tr><tr><td>WIPO-alpha</td><td>75,250</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>1998-2002</td><td>Classification</td></tr><tr><td>CLEF-IP (2011)</td><td>1,500,000</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>&lt; 2009</td><td>Retrieval+Classification</td></tr><tr><td>USPTO-2M</td><td>2,000,147</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>√</td><td>2006-2015</td><td>Classification</td></tr><tr><td>BIGPATENT</td><td>1,341,362</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td></td><td>1971-2018</td><td>Summarization</td></tr><tr><td>Ours (HUPD)</td><td>4,518,263</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>2004-2018</td><td>Multi-Purpose</td></tr></table>

Table 1: Comparison of HUPD with other datasets whose primary goal is NLP patent analysis. The abbreviated columns mean the following. Abst: Abstract, Appl: Applicant Information, Exam: Examiner Information, Invt: Inventor Information, PD: Publication Date, Bkgd: Background, Dsc: Description, and PCs: IPC/CPC codes.

abstractive summarization of patent sections to information retrieval and named-entity recognition and extraction, one can perform a variety of both standard and novel NLP tasks on patent data. Furthermore, the well-structured nature of our dataset allows for researchers to study how concepts like acceptance criteria vary across contexts and over time. For the IP community, a key motivator of our work is that there are many rote tasks associated with patent filing and examination—including the categorization of patents into relevant technology areas and prior art search—in which machine learning methods could potentially be used to provide efficiency, value, and cost savings. Finally, for the general audience, our present work illustrates how NLP/ML tools may be efficiently deployed for advancing socially relevant objectives and studying diverse and dynamic application areas.

# 2 Preliminaries and Background

A patent application $^{3}$ typically consists of a title, abstract, set of claims, detailed description, drawings (if needed to describe the invention), and cross-references to related applications (if any), among other written specifications. Applications are filed to the USPTO and reviewed by examiners who are expected to have knowledge and expertise associated with the subject matter of the invention.

During the examination process, an examiner determines the patentability of the invention. The examiner decides whether the proposed invention is useful, non-obvious, and statutory, and searches for already-existing patents within the technology sphere of the invention to confirm the set of claims provided is novel. Afterwards, the examiner sends an Office action to the applicant, notifying them of the USPTO's decision. If the decision is favorable, then the applicant can choose to proceed with their application and have the USPTO issue their patent. However, if the decision is unfavorable, then the applicant receives a notification of rejection; it is then up to the applicant to decide whether they wish to respond to the rejection, continue to pursue their application, and request a reexamination.

For the task of predicting an application's outcome, it is useful to provide a formalization of what it means for a patent application to be accepted and rejected. We say that a patent application is “accepted” if it has been officially approved, granted, and published by the USPTO. There is, however, no clear-cut notion of absolute rejection in patent applications. An application might, and in fact often does (at the beginning), receive an Office action indicating a non-final or final rejection—typically on the grounds of prior art, scope of the claims, or lack of utility or novelty or obviousness, but the applicant can submit a response to the USPTO $^{4}$ and re-open the prosecution of their application, even in the case of a final rejection. As a result, a “final rejection” does not imply a patent application has no chance of eventually being accepted after revisions. In our dataset, we label an application “rejected” if it has received an Office action of rejection—final or non-final—and was ultimately abandoned by the applicant. $^{5}$ We categorize all the remaining applications, which are still waiting a

response from the USPTO, as “pending.” These distinctions are particularly useful when describing and discussing the binary classification task of patent decisions in the following sections. $^{6}$

# 3 Related Work on Patent Analysis

Existing patent datasets for NLP focus nearly exclusively on two tasks: patent subject classification and summarization. Below, we provide overview of prior datasets and studies in these areas. $^{7,8}$

Automated Subject Classification. Patents are classified by subject matter according to standard taxonomies, most notably the International Patent Classification (IPC) and Cooperative Patent Classification (CPC) systems. $^{9}$ These IPC/CPC codes are hierarchical—classified at a class level (e.g., G-Physics), subclass level (e.g., G06F-Electric Digital Data Processing), and so on. Previous studies attempted to predict the IPC or CPC codes of patents at the class and subclass levels using various statistical methods, including classical statistical learning tools $[24, 5, 8, 50, 16]$ and neural architectures $[18, 28, 59]$ . More recently, Transformers $[55]$ have been considered for this task: Lee and Hsiang $[26]$ , for instance, fine-tuned a pre-trained BERT $[9]$ to predict IPC/CPC codes of USPTO patents. Zaheer et al. $[58]$ conducted similar experiments using their BIGBIRD model $[29]$ and showed improvements over BERT models. $^{10}$

As shown in Table 1, WIPO-alpha, CLEF-IP, and USPTO-2M have been the main gymnasia for model training for the automated IPC/CPC classification tasks, but these corpora are still limited in their scopes. They contain a relatively narrow set of patent text and metadata, do not allow users to choose which year ranges to focus on during training and testing, and may sometimes be difficult to pre-process and tokenize properly. HUPD addresses these limitations, and also allows more flexibility in data selection and provides wider and more comprehensive text, field, and year coverage.

Patent Text Generation and Summarization. With the growing availability and success of large language models in recent years, there has been an interest in applying language models to patents. Sharma et al. [46] initiated such explorations, introducing the first summarization dataset on patents, called BIGPATENT, and trained summarization tools on their dataset to generate the abstract section of a patent given its description section. The BIGPATENT dataset contains 1.3 million utility patents filed to the USPTO between 1971 and 2018, and was collected from the Google Patents Public Dataset via BigQuery. Our dataset differentiates itself from BIGPATENT in three aspects: (1) HUPD includes metadata and fields, including claims, background, filing date, and examiner information, that are not present in BIGPATENT; (2) in addition to accepted patents, it contains rejected and pending applications; it thus enables the study of patent acceptance/rejection $^{11}$ ; and (3) it has approximately three times as many documents as BIGPATENT (Table 1).

Patent Acceptance Prediction. To the best of our knowledge, our study is the first work to introduce a practical definition of rejection in patent examination to identify, analyze, and discuss the patterns in and characteristics of accepted versus rejected patent applications from a purely textual perspective. In that sense, we introduce the patent decision classification task to the NLP literature.

submits a written declaration of abandonment. In the case of non-final rejections, for instance, applicants are usually given six months to revise their applications [43].

$^{6}$ We refer our readers to the study by Lemley and Sampat [27] for a discussion on data controversies around calculating patent grant rates.

$^{7}$ In Section D, we provide a more thorough and comprehensive comparison of our dataset against BIGPATENT and other existing large-scale NLP datasets (such as S2ORC [30] and WikiBio [25]).

$^{8}$ For an alternative compendium of studies that make use of deep learning techniques for patent analysis, please see the recent study by Krestel et al. [23].

$^{9}$ Established by the Strasbourg Agreement of 1971 and Administered by the WIPO, the IPC scheme is currently being used, in various capacities, in over one hundred major patent offices worldwide. The CPC scheme is an extension of the IPC scheme and has been adopted by the USPTO since 2013. It offers a more comprehensive coverage of some new technical developments and includes an additional section “Y.” In the US, every issued patent needs to be classified, based on its subject matter, according to the CPC taxonomy.

$^{10}$ In addition, Acikalin et al. [1] classify the scope of a patent in terms of exposure to the U.S. Supreme Court decision Alice Corp. v. CLS Bank International, 573 U.S. 208 (2014).

$^{11}$ Furthermore, HUPD contains the original, inventor-submitted version of patent applications, as opposed to the granted versions. The distribution of language in the granted versions of accepted patents may change after examiner comments, making granted versions less directly comparable to pending patent applications. For those patent applications that are eventually granted and published, the published version can be retrieved using the patent number metadata in our dataset.

<table><tr><td>Section</td><td>Brief Description</td><td>Avg # Tokens</td></tr><tr><td>Title</td><td>Title of the Invention</td><td>16.4</td></tr><tr><td>Abstract</td><td>Summary of the Background and Claims</td><td>132.0</td></tr><tr><td>Claims</td><td>List of Items Defining the Invention</td><td>1271.5</td></tr><tr><td>Background</td><td>Brief Statement of the Field of Art and Related Art of the Invention</td><td>627.11</td></tr><tr><td>Summary</td><td>Condensed Version of the Description</td><td>917.8</td></tr><tr><td>Description</td><td>Detailed Statement and Disclosure of the Invention</td><td>11855.6</td></tr></table>

Table 2: Brief description of and average number of tokens in each text-based section in our dataset. Typically, the description section in a patent application is almost 100 times longer than the abstract section.

# 4 The Dataset

The Harvard USPTO Patent Dataset (HUPD) contains 4,518,263 utility patent applications filed to the USPTO between January 2004 and December 2018. $^{12}$ In this section, we elaborate on the data collection process and provide details about the data format and dataset structure. We furthermore enumerate and highlight a portion of the data's statistical properties, and acknowledge the limitations and ethical considerations of the present work. Additional information is included in Section B.

Construction. As specified by US law, all patent data is publicly accessible. However, the process of obtaining the patent data and its corresponding metadata, normalizing all data to the same format, filtering missing and erroneous data, de-duplicating data, and merging all data into a single easy-to-use dataset is non-trivial. Here we provide an overview of this process.

Patent application texts were obtained from the USPTO Bulk Data Storage System (BDSS; Patent Application Data/XML Version) as XML files. $^{13}$ As not all the original patent files follow the same XML structure, we wrote regular expressions to parse all the different formats into a normalized set of data fields, and stored these data fields as structured JSON files. Filing metadata—including acceptance decisions, filing dates, titles, and classification information—were downloaded separately from the USPTO Patent Examination Research Datasets $[17, 34]$ in February 2021. This metadata was then merged with the full-text patents from the USPTO BDSS to link patents filed before 2020 with information about examiners, office actions, and additional filing information. This merging process ensures that our dataset contains both this updated metadata structure and the patent application texts.

Finally, we assembled the continuation information for each application as follows: Applications with parent filings in the USPTO continuity data files were marked with the prefix “CONT-” in the decision status field. $^{14}$ Hence, there are six labels in total for the decision status of applications: “Accepted,” “Rejected,” “Pending,” “CONT-Accepted,” “CONT-Rejected,” and “CONT-Pending.” $^{15}$

Statistics. Table 1 provides a quantitative comparison of our dataset with other patent data sources, while Table 2 gives information and statistics about the text-based sections in our dataset. HUPD builds on its counterparts, including BIGPATENT, because of not only its sheer size and wider coverage but also its ability to allow users to see the evolution of patent families over time. Figure 10 (in Section G) gives the breakdown of the applications into six decision status labels for 2011-2016.

Limitations. From a methodological point of view, the dataset is limited to patents from the U.S. and in English, and drops all image content such as drawings. From a functional point of view, some textual sections are longer than some current NLP models can process, and specialized vocabulary in certain fields can create issues for existing tokenizers.

Potential Biases. We provide an examination of HUPD for potential biases in Section C. We recommend that researchers who use HUPD consider these biases when interpreting their results—especially if their analyses involve inventor and/or examiner information.

Ethical Considerations. While building HUPD, we followed the data sheets and statements introduced by Gebru et al. [13] and Bender and Friedman [3], discussing the motivations, objectives,

<table><tr><td>Task Name</td><td>Setup</td><td>Metrics</td></tr><tr><td>Acceptance Prediction</td><td>Abstract or Claims → Patent Outcome (Decision)</td><td>Accuracy</td></tr><tr><td>IPC/CPC Classification</td><td>Abstract or Claims → IPC/CPC</td><td>TOP-k</td></tr><tr><td>Language Modeling</td><td>Abstract or Claims or Description</td><td>Perplexity</td></tr><tr><td>Abstractive Summarization</td><td>Claims or Description → Abstract</td><td>ROUGE/BLEU</td></tr></table>

Table 3: Summary of the four NLP tasks presented in this work, along with some evaluation metrics for them. Our dataset can be used to conduct many other NLP/IP experiments. See Section 5 for detailed information.

![](images/bf10d292c188bb3e5d818bd60b45bca6192c9d98cc84b4040ef7ceb829ce77ef.jpg)

<details>
<summary>line</summary>

| IPC Class Code | Absolute Frequency |
| -------------- | ------------------ |
| G06F           | 10^5               |
| H01L           | ~10^4.5            |
| H04L           | ~10^4.3            |
| H04W           | ~10^4.2            |
| A61B           | ~10^4.1            |
| G01N           | ~10^4.0            |
| H01M           | ~10^3.9            |
| G03G           | ~10^3.8            |
| G02F           | ~10^3.7            |
| C12N           | ~10^3.6            |
| G01B           | ~10^3.5            |
| B60L           | ~10^3.4            |
| F03D           | ~10^3.3            |
| G01P           | ~10^3.2            |
| A45F           | ~10^3.1            |
| A47H           | ~10^3.0            |
</details>

![](images/a65ba898dd02622c74dee17964f727a874a5cf504e8c57cc7c5e748bd6decf30.jpg)

<details>
<summary>pie</summary>

| Label | Percentage (%) |
| :--- | :--- |
| G06F | 10.4 |
| H01L | 5.7 |
| H04L | 3.9 |
| H04W | 3.5 |
| H04N | 3.0 |
| A61B | 2.4 |
| A61K | 2.2 |
| G01N | 1.7 |
| G06K | 1.7 |
| G06Q | 1.7 |
| G02B | 1.7 |
| H01M | 1.7 |
| H04B | 1.7 |
| G06T | 1.7 |
| H05K | 1.7 |
| G01R | 1.7 |
| G09G | 1.7 |
| G03G | 1.7 |
| E21B | 1.7 |
| Other Labels | 54.8 |
</details>

Figure 2: IPC distribution of accepted patent applications from 2011 to 2016 at the IPC subclass level. There are 637 IPC subclass labels in HUPD, of which the most common 20 codes make up half of the distribution. G06F-Electric Digital Data Processing is the largest IPC subclass, accounting for 10.4% of applications.

collection process, workflow, use cases, distribution, maintenance, potential contributions, and potential misuses of our dataset and research. In Section B, we share a data card for our dataset, in the style of Gehrmann et al. [14].

Distribution & Maintenance. We have publicly released all the distilled JSON-formatted textual content of patent applications, along with the data loading and tokenization code, on our website. $^{16}$

# 5 Using the Dataset

Due to the versatile nature of HUPD, a wide range of NLP tasks and experiments can be constructed from the dataset by selecting the appropriate fields and metadata contained in each patent application. In this section, we highlight four tasks that we believe to be among the most valuable and relevant to the NLP and IP communities: (i) binary classification of patent decisions, (ii) multi-label classification of patent IPC/CPC categories, (iii) language modeling, and (iv) summarization. Of course, the dataset may easily be used to conduct other investigations, such as patent clustering, prior art search, and early detection of superstar inventions. $^{17}$ In what follows, we describe our four tasks of interest in detail and contextualize their importance within the world of IP. Table 3 provides a brief overview of each task and the corresponding metrics used to measure task performance.

Patent Acceptance Prediction. Given a section of an application (in particular, the abstract, claims, or description), we predict whether the application will be accepted by the USPTO. From the perspective of the NLP community, this is a standard classification task. Yet, the potential applications and benefits of this decision task, as well as its difficulty, distinguish it from prevalent binary classification benchmarks (e.g., SST, Yelp). In our experiments, we focus on applications without parent filings to make our setup simple and clear, thereby excluding all the CONT-applications. Also, we do not include any pending applications.

Automated Subject Classification. The next task is to predict the primary IPC or CPC code of a patent application given (some subset of) the text of the application. A coherent automated classification of patent documents into different technology fields can facilitate effective assignment

of patent applications to examiners. In addition, it may help create a rigorous and standardized catalog of prior art for research and exploration. This task might also help the early identification of valuable inventions that bridge multiple technological domains or the emergence of new subject areas. In our experimental setups, we predict the main IPC codes of patent applications at the subclass level, as IPC codes are available for a larger set of patents than CPC codes. $^{18}$ There are 637 IPC codes at the subclass level in our dataset, but they are not uniformly distributed (see Figure 2); for instance, G06F-Electric Digital Data Processing constitutes 10.4% of accepted patents that were filed between 2011 and 2016 and the most popular 15 IPC codes make up almost 40%. Hence, it is difficult to achieve good classification performance by only predicting the major classes.

Language Modeling. We move next from classification to language modeling (LM). We first consider masked LM of patent claims. We focus on the claims section because it forms the basis of the invention described in a patent application; it has a distinctive language style; and it is considered to be the most legally potent part of the patents. We also conduct LM experiments on the abstract sections of patents, which are more similar to standard natural language. These language models can be used for downstream tasks, as well as for domain-specific investigations. In Section I, we demonstrate one simple but interesting application of our LMs by visualizing the average embeddings of different patent categories under this fine-tuned language model, capturing the textual evolution of innovation concepts and trends in patent applications across different technology areas. $^{19}$

Abstractive Summarization. The formulation of the final task follows naturally from the structure of our patent data: Each patent contains an abstract in which the applicant summarizes the content of the patent. We use this section as the ground truth for our abstractive summarization task, and we use either the claims or the description as the source text. We perform this conditional generation task with the same motivation that inspired Sharma et al. [46]; our setup is similar to theirs apart from the size and scope of our dataset. One difference in our current setup is that our dataset allows us to explore using either the claims or description section for the source text, whereas in Sharma et al. [46] only the description section is available. $^{20}$

# 6 Results and Discussion

In the next two sections, we establish benchmarks for the four aforementioned tasks, describe the models used, and analyze our findings. In our experiments, we typically use the abstract or claims sections as the input to the models; however, future studies using our dataset can easily include other sections, such as the background and description.

Our baselines consisted of various subsets of Bernoulli and Multinomial naive Bayes classifiers, logistic regression (Logistic), CNN, DistilBERT [44], BERT [9], DistilROBERTa [44], RoBERTa [29], and T5-Small [40] for different tasks. $^{21}$

Patent Acceptance Prediction. $^{22}$ Table 4 reports the performances of our models on the most popular IPC codes. In each IPC subclass, with the exception of G01N-Investigating or Analyzing Materials by Determining Their Chemical or Physical Properties, the best performance was achieved by the models that relied on the claims. This result is consistent with the idea that the claims define the overall scope, novelty, and usefulness of the invention. The claims, together with the prior art, provide the most useful and critical information about the patentability of an invention. It was, however, surprising to discover that there was not a significant difference between the NB classifiers and the BERT models in terms of accuracy scores across categories. We speculate that the BERT models

<table><tr><td>IPC - Section</td><td>BernNB</td><td>MultiNB</td><td>Logistic</td><td>CNN</td><td> $\text{DistilBERT}^{\text{FT}}$ </td><td> $\text{BERT}^{\text{FT}}$ </td><td> $\text{RoBERTa}^{\text{FT}}$ </td></tr><tr><td>GO6F - Abstract</td><td>61.86</td><td>61.47</td><td>58.24</td><td>60.97</td><td>61.53</td><td>61.28</td><td>61.31</td></tr><tr><td>GO6F - Claims</td><td>63.96</td><td>62.06</td><td>58.02</td><td>63.38</td><td>63.37</td><td>62.97</td><td>63.25</td></tr><tr><td>H01L - Abstract</td><td>58.98</td><td>59.05</td><td>58.54</td><td>60.71</td><td>61.46</td><td>61.85</td><td>61.85</td></tr><tr><td>H01L - Claims</td><td>60.97</td><td>60.29</td><td>59.53</td><td>62.63</td><td>62.50</td><td>61.61</td><td>61.94</td></tr><tr><td>A61B - Abstract</td><td>59.15</td><td>58.81</td><td>57.31</td><td>58.75</td><td>58.36</td><td>59.58</td><td>59.66</td></tr><tr><td>A61B - Claims</td><td>59.30</td><td>59.12</td><td>57.25</td><td>59.49</td><td>60.15</td><td>61.20</td><td>61.00</td></tr><tr><td>GO1N - Abstract</td><td>59.85</td><td>59.89</td><td>57.25</td><td>59.98</td><td>59.00</td><td>60.30</td><td>61.10</td></tr><tr><td>GO1N - Claims</td><td>58.06</td><td>57.97</td><td>58.37</td><td>59.80</td><td>60.16</td><td>60.34</td><td>60.97</td></tr></table>

Table 4: Baseline performances of our models on the binary classification of patent acceptance task. All the models were trained and evaluated on the patent applications filed to the USPTO between January 2011 and December 2016. All the test sets contained equal numbers of accepted and rejected applications, so the baseline accuracy to compare these models against is 50%. In all but one IPC category, the models trained on the claims sections yielded the best performance. None of the individual accuracy scores, however, went beyond 64%. In Section 7, we further report results on our conditional universal acceptance prediction classifier. The superscript FT on the Transformers denotes that these models were fine-tuned, not trained from scratch. (See Table 7 for our full results and Table 8 for English descriptions of the four-digit IPC codes.)

<table><tr><td rowspan="2">Section</td><td colspan="2">BernNB</td><td colspan="2">MultiNB</td><td colspan="2">CNN</td><td colspan="2">DistilBERT $^{\text{FT}}$ </td><td colspan="2">RoBERTa $^{\text{FT}}$ </td></tr><tr><td>TOP1</td><td>TOP5</td><td>TOP1</td><td>TOP5</td><td>TOP1</td><td>TOP5</td><td>TOP1</td><td>TOP5</td><td>TOP1</td><td>TOP5</td></tr><tr><td>Abstract</td><td>40.65</td><td>63.92</td><td>47.38</td><td>75.17</td><td>53.87</td><td>81.69</td><td>61.75</td><td>89.11</td><td>61.07</td><td>88.24</td></tr><tr><td>Claims</td><td>39.37</td><td>65.42</td><td>48.09</td><td>77.78</td><td>56.10</td><td>83.40</td><td>63.40</td><td>90.22</td><td>62.82</td><td>89.46</td></tr></table>

Table 5: Performances of our models on the multi-class IPC classification of patent codes at the subclass level. Our DistilBERT models yielded the best performance overall under both abstract and claims input setups. TOP1 (i.e., accuracy) checks whether our prediction is the same as the actual label, whereas TOP5 measures whether the actual label of the input is amongst our five top predictions (i.e., five classes with the highest probability weights). Our high TOP5 scores indicate that our models are good at predicting the IPC codes of both popular and underrepresented classes— Figure 8 also provides evidence towards this conclusion.

might have mirrored the behaviors of the NB classifiers at the end, failing to go beyond word-level feature extraction. Nonetheless, we note the task is difficult and sophisticated, and currently uses only a limited portion of a patent text to determine the acceptability of the invention. We posit that new advances in Transformer models for longer text may improve predictive accuracy for this task. $^{23}$

Automated Subject Classification. For the multi-class classification task, we considered only accepted patents, since they contain reliable and accredited IPC/CPC codes. Table 5 details the TOP1 and TOP5 accuracy scores for the multi-class IPC code classification results at the subclass level. First, we note that performance increases with more sophisticated models. The DistilBERT model, trained on the claims section, even achieves 63% accuracy at TOP1 and 90% accuracy at TOP5; this is notable since, as can also be inferred from Figure 2, majority-class baseline could only yield around 10% for TOP1 and 26.5% for TOP5. In fact, the clear diagonal line in Figure 8 (center) also illustrates that the DistilBERT model, for instance, has learned important features that enable it to successfully classify minority categories, such as F25C-Producing, Working or Handling Ice and D05B-Sewing. In general, the models trained on the abstract section performed as well as those trained on the claims section, perhaps an indication (in accord with the findings of Li et al. [28]) that the abstract alone contains useful information about the appropriate principal technology area the patent application might belong to. Model performance at the class level, as opposed to the subclass level, was even stronger. The DistilBERT models achieve over 80% TOP1 accuracy at the class level. Furthermore, incorrect predictions were often from similar or related technology areas. (Our high TOP5 scores also corroborate this finding.)

Language Modeling. We trained a masked language model on patent claims in the style of BERT [9]. We trained on a subset of the full patent dataset consisting of patent applications from 2011 to 2016 (1.57M applications) and evaluated on all patent applications from 2017 (187K applications). We performed masked language modeling with DistilRoBERTa [44] (82M parameters), initializing with

<table><tr><td>Model</td><td>Setting</td><td>R-1</td><td>R-2</td><td>R-L</td></tr><tr><td>T5 (Small)</td><td>Description→Abstract</td><td>62.87</td><td>47.20</td><td>54.36</td></tr><tr><td>T5 (Small)</td><td>Claims→Abstract</td><td>69.00</td><td>53.82</td><td>59.88</td></tr></table>

Table 6: Performances of our T5 summarization models on HUPD, as measured by ROUGE. Higher numbers reflect better performance. Summarization from the claims performs better than summarization from the description. Although our ROUGE scores are higher than those reported by Sharma et al. [46] on BIGPATENT, the scores are not directly comparable, since the evaluation data and tokenization schemes are different.

a model pretrained on OpenWebText [15]. We release this model publicly so that researchers may utilize it for downstream tasks. $^{24}$

Abstractive Summarization. $^{25}$ Table 6 details our results. We find that patent descriptions and claims can both be effectively summarized into patent abstracts, with claims leading to improved performance across all metrics. Qualitatively, the models produce fluent abstracts (often single long and complex sentences) complete with accurate details drawn from the claims section, as shown in Table 10. Consistent with BIGPATENT, our results suggest that the summarization of patent data could serve as a challenging, domain-specific conditional generation task for the NLP community.

# 7 Evolution of Innovation Criteria and Trends over Time

A key advantage of HUPD is its combination of unstructured textual content and rich, structured bibliographic metadata. Patents, like many other applied natural-language domains, exhibit concept and domain shifts—the criteria for innovation varies across categories and evolves over time. In this section, we discuss some ways that our dataset exhibits time- and state-varying concepts. We hope this feature of patent data allows for fresh studies of the nature of shifts across categories and supports the development of NLP models that accommodate concept or distributional shifts.

Universal Acceptance Classification. We explored how a universal decision classifier, trained on all the IPC categories, might perform on individual categories. To this end, we trained a conditional DistilBERT classifier, where the conditional information had the title, issue year, and IPC code, in addition to the abstract section, to predict the acceptance likelihood of a patent application. $^{26}$ We used the patent applications filed between 2011 and 2016. The model achieved an overall accuracy score of $62\%$ on the entire test set. When we assessed the model's performance on individual IPC subclasses, we discovered, for instance, that the model yielded almost $64.5\%$ accuracy on G06F-Electric Digital Data Processing, $61.9\%$ accuracy on H04N-Pictorial Communication, e.g., Television, and $57\%$ accuracy on G06Q-Data Processing Systems or Methods.

Cross-Category Evaluation. In order to understand and identify relationships between different patent evaluation criteria in different IPC classes, we also took each DistilBERT model trained on one IPC subclass of patents and evaluated it across all the other popular IPC subclasses. Figure 3 provides an illustration of our empirical findings. Patent technology areas that are conceptually closer to each other, such as G06F-Electric Digital Data Processing and H04L-Transmission of Digital Information, appear to have similar standards for patent acceptance. Notably, models trained in one context do not generalize to most other contexts. This suggests the criteria for patent acceptance are sensitive to the technical demands of the specific category of each application.

Performance Over Time. We can also use our models and dataset to understand the evolving criteria for patent acceptance, as well as innovation trends over time. Figure 4 shows the performance of a decision classification model trained on patent applications from 2011 to 2013 evaluated on applications produced in earlier and later years. While model performance usually deteriorates over time, suggesting changes in the features that predict patent acceptance, the rate of decay appears to be sharper for fields anecdotally thought to be faster-moving. This property of patent data may make it useful for studying concept shift that varies by class.

Additional Tasks. Given the richness of patent text and metadata, HUPD enables research on a wide range of tasks and use cases that may be explored in future work. In Section H, we describe some of these tasks—such as long sequence modeling, patent clustering, and patent examiner assignment—as well as the potential social impact and applications of our work.

![](images/9fc1f025bcda1059a65c768d0fcc6731ad24647709a848bc58d723e52dd617df.jpg)

<details>
<summary>heatmap</summary>

— EVALUATED ON —
| | G06F | H01L | A61K | H04L | H04N | A61B | G06Q | H04W | G01N |
|---|---|---|---|---|---|---|---|---|---|
| G06F | 62.66 | 56.06 | 49.85 | 60.44 | 55.75 | 55.98 | 54.45 | 54.37 | 56.13 |
| H01L | 52.00 | 62.18 | 48.63 | 52.81 | 51.64 | 51.79 | 51.09 | 49.29 | 54.14 |
| A61K | 49.90 | 51.68 | 57.06 | 51.76 | 50.00 | 51.53 | 49.64 | 50.65 | 50.16 |
| H04L | 58.34 | 52.85 | 58.34 | 62.27 | 56.88 | 52.86 | 51.24 | 55.32 | 53.66 |
| H04N | 57.13 | 53.51 | 50.30 | 56.72 | 60.94 | 54.40 | 53.50 | 51.83 | 50.48 |
| A61B | 53.59 | 51.68 | 53.59 | 55.74 | 52.85 | 58.64 | 51.38 | 51.65 | 54.46 |
| G06Q | 54.01 | 51.29 | 49.64 | 54.77 | 51.34 | 54.01 | 59.84 | 51.65 | 54.14 |
| H04W | 54.99 | 52.82 | 49.48 | 57.44 | 52.60 | 52.25 | 50.51 | 56.62 | 49.92 |
| G01N | 52.15 | 52.25 | 51.76 | 55.09 | 50.78 | 51.28 | 51.46 | 51.48 | 60.67 |
</details>

Figure 3: Cross-category evaluation of BERT acceptance prediction classifiers trained on one IPC code evaluated to predict acceptance on other IPC codes. Patent categories that are conceptually similar appear to have closer shared criteria for acceptance (for example, A61B and G01N). Most models also tend to perform well when evaluated on patent applications in H04L-Transmission of Digital Information; moreover, the model trained on H04L-Transmission of Digital Information has high predictive power across other application types. This might reflect acceptance criteria in this category involving more high-level evaluations of quality, as opposed to more domain-specific innovations in engineering or manufacturing.

![](images/298a78e15131a8ccd3f0d96bba726d6bf082e8f75162455c9f54a4fa35da242a.jpg)

<details>
<summary>line</summary>

| Year(s) | G06F - Electric Digital Data Processing | H01L - Semiconductor Devices | H04L - Transmission of Digital Information | H04W - Wireless Communication Networks | H04N - Pictorial Communication, e.g. Television | A61B - Diagnosis, Surgery, Identification | A61K - Preparations for Medical Purposes | G01N - Investigating or Analyzing Materials | G06Q - Data Processing Systems or Methods |
| ------- | ---------------------------------------- | ---------------------------- | ------------------------------------------ | -------------------------------------- | --------------------------------------------- | ----------------------------------------- | --------------------------------------- | ---------------------------------------- | ----------------------------------------- |
| 2008    | 57.5                                     | 57.8                         | 58.2                                       | 56.5                                   | 59.8                                          | 57.2                                      | 56.8                                    | 62.0                                     | 57.0                                      |
| 2009    | 57.8                                     | 60.2                         | 58.0                                       | 57.2                                   | 59.0                                          | 57.5                                      | 57.3                                    | 64.0                                     | 57.2                                      |
| 2010    | 58.0                                     | 64.0                         | 58.5                                       | 56.3                                   | 60.5                                          | 59.8                                      | 59.0                                    | 59.0                                     | 59.0                                      |
| 2011-13 | 58.2                                     | —                            | 61.8                                       | 59.0                                   | 60.5                                          | 60.2                                      | 59.2                                    | 62.5                                     | —                                         |
| 2014    | 58.0                                     | —                            | 61.5                                       | 56.0                                   | 59.5                                          | 59.0                                      | 55.2                                    | 60.5                                     | 59.5                                      |
| 2015    | 58.2                                     | —                            | 61.0                                       | 54.5                                   | 58.8                                          | 59.5                                      | 54.8                                    | 60.2                                     | —                                         |
| 2016    | —                                        | —                            | 60.8                                       | 55.0                                   | 59.2                                          | 59.2                                      | 54.2                                    | 57.8                                     | 61.8                                      |
</details>

Figure 4: Performance of a BERT decision classifier trained on applications from 2011 to 2013 and evaluated on applications produced in earlier and later years. While model performance decays over time across most categories (suggesting changing acceptance standards), acceptance criteria appear to change more quickly in faster-moving fields (e.g., H01L-Semiconductor Devices and H04W-Wireless Communication) and slower in more developed fields (e.g., A61B-Diagnosis, Surgery, Identification). In future work, it may also be illustrative to use this dataset to investigate the impact of the U.S. Supreme Court decision in Alice Corp. v. CLS Bank International, 573 U.S. 208 (2014), on the software-related patent applications issued after 2014.

# 8 Conclusion

We presented the Harvard USPTO Patent Dataset (HUPD), which is, to date, the largest and most versatile and comprehensive structured corpus of patent data constructed for the NLP community. HUPD contains 4.5 million English-language utility patent applications filed to the USPTO between 2004 and 2018. We also established benchmarks for two classification-based and two generation-based tasks on our dataset. We provided detailed qualitative analyses of the models trained for these tasks and demonstrated how our dataset presents a setting with measurable concept shift. We hope that the combination of our dataset and the models used in this paper will not only advance research in patent analysis, but eventually also help patent applicants prepare more successful patent filings and provide a domain-specific laboratory for a multitude of NLP tasks.

# Acknowledgements

We thank Christopher Bavitz, Yonatan Belinkov, Tommy Bruzzese, Dallas Card, Jiafeng Chen, Julia L. Englebert, Sebastian Gehrmann, Tayfun Gur, Umit Gurun, Dan Jurafsky, Peter Henderson, Şule Kahraman, Ryan Kearns, Megan Ma, Daniel McFarland, Drew Pendergrass, Karen Sinclair, Kyle Swanson, Andrew A. Toole, Brandon Walton, Ian Wetherbee, Benjamin H. Wittenbrink, Michihiro Yasunaga, and the members of the Lab for Economic Design at Harvard University for helpful comments and suggestions. We are especially thankful to the USPTO for providing us with the bulk data and patent research datasets, and Andrew A. Toole, the Chief Economist at the USPTO, for his help navigating the USPTO's data products. We gratefully acknowledge the support of the three Microsoft Azure credit grants from the Harvard Data Science Initiative for data storage and computation. Some of the experiments presented in this paper were run on the FASRC Cannon cluster supported by the FAS Division of Science Research Computing Group at Harvard University and the Scalable Magic switch cluster. Suzgun gratefully acknowledges the support of a Harvard University Center of Mathematical Sciences and Applications (CMSA) Economic Design Fellowship and the Harvard College Research Program. Melas-Kyriazi gratefully acknowledges the support of a Rhodes Scholarship. Sarkar gratefully acknowledges the support of a National Science Foundation (NSF) Graduate Research Fellowship. Kominers gratefully acknowledges the support of the Ng Fund and the Mathematics in Economics Research Fund of the CMSA, as well as NSF grant SciSIP-1535813.

# References

[1] Utku Acikalin, Tolga Caskurlu, Gerard Hoberg, and Gordon M. Phillips. 2022. Intellectual Property Protection Lost: The Impact on Competition and Acquisitions. SSRN Electronic Journal.   
[2] Alex Bell, Raj Chetty, Xavier Jaravel, Neviana Petkova, and John Van Reenen. 2019. Who Becomes an Inventor in America? The Importance of Exposure to Innovation. The Quarterly Journal of Economics, 134(2):647–713.   
[3] Emily M Bender and Batya Friedman. 2018. Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science. Transactions of the Association for Computational Linguistics, 6:587–604.   
[4] Steven Bird. 2006. NLTK: The Natural Language Toolkit. In Proceedings of the COLING/ACL 2006 Interactive Presentation Sessions, pages 69–72.   
[5] Xiao-Lei Chu, Chao Ma, Jing Li, Bao-Liang Lu, Masao Utiyama, and Hitoshi Isahara. 2008. Large-Scale Patent Classification with Min-Max Modular Support Vector Machines. In 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), pages 3973–3980. IEEE.   
[6] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4599–4610, Online. Association for Computational Linguistics.

[7] Mercedes Delgado, Myriam Mariani, and Fiona E Murray. 2019. The Role of Location on the Inventor Gender Gap: Women are Geographically Constrained. DRUID19 Conference.   
[8] F. Derieux, M. Bobeica, Delphine Pois, and Jean-Pierre Raysz. 2010. Combining Semantics and Statistics for Patent Classification. In CLEF.   
[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.   
[10] Jesse Dodge, Maarten Sap, Ana Marasović, William Agnew, Gabriel Ilharco, Dirk Groeneveld, and Matt Gardner. 2021. Documenting the English Colossal Clean Crawled Corpus. ArXiv, abs/2104.08758.   
[11] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. The Pile: An 800GB Dataset of Diverse Text for Language Modeling.   
[12] Evelina Gavrilova and Steffen Juranek. 2021. Female Inventors: The Drivers of the Gender Patenting Gap. Available at SSRN 3828216.   
[13] Timnit Gebru, Jamie H. Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, H. Wallach, Hal Daumé, and Kate Crawford. 2018. Datasheets for Datasets. ArXiv, abs/1803.09010.   
[14] Sebastian Gehrmann, Tosin Adewumi, Karmanya Aggarwal, Pawan Sasanka Ammanamanchi, Aremu Anuoluwapo, Antoine Bosselut, Khyathi Raghavi Chandu, Miruna Clinciu, Dipanjan Das, Kaustubh D Dhole, et al. 2021. The GEM Benchmark: Natural Language Generation, Its Evaluation and Metrics. arXiv preprint arXiv:2102.01672.   
[15] Aaron Gokaslan and Vanya Cohen. 2019. OpenWebText Corpus.   
[16] J. C. Gómez. 2019. Analysis of the Effect of Data Properties in Automated Patent Classification. Scientometrics, 121:1239 - 1268.   
[17] Stuart JH Graham, Alan C Marco, and Richard Miller. 2015. The USPTO Patent Examination Research Dataset: A Window on the Process of Patent Examination. Georgia Tech Scheller College of Business Research Paper No. WP, 43.   
[18] Mattyws F Grawe, Claudia A Martins, and Andreia G Bonfante. 2017. Automated Patent Classification Using Word Embedding. In 2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA), pages 408–411. IEEE.   
[19] Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. 2021. CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. arXiv preprint arXiv:2103.06268.   
[20] Kyle Jensen, Balázs Kovács, and Olav Sorenson. 2018. Gender Differences in Obtaining and Maintaining Patent Rights. Nature biotechnology, 36(4):307–309.   
[21] Diederik P Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In ICLR (Poster).   
[22] Narine Kokhlikyan, Vivek Miglani, Miguel Martin, Edward Wang, Bilal Alsallakh, Jonathan Reynolds, Alexander Melnikov, Natalia Kliushkina, Carlos Araya, Siqi Yan, et al. 2020. Captum: A Unified and Generic Model Interpretability Library for PyTorch. arXiv preprint arXiv:2009.07896.   
[23] Ralf Krestel, Renukswamy Chikkamath, Christoph Hewel, and Julian Risch. 2021. A Survey on Deep Learning for Patent Analysis. World Patent Information, 65:102035.   
[24] Leah S Larkey. 1999. A Patent Search and Classification System. In Proceedings of the Fourth ACM Conference on Digital libraries, pages 179–187.

[25] Rémi Lebret, David Grangier, and Michael Auli. 2016. Neural Text Generation from Structured Data with Application to the Biography Domain. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1203–1213, Austin, Texas. Association for Computational Linguistics.   
[26] Jieh-Sheng Lee and J. Hsiang. 2020. Patent Classification by Fine-Tuning BERT Language Model. World Patent Information, 61:101965.   
[27] Mark A Lemley and Bhaven N Sampat. 2008. Is the Patent Office a Rubber Stamp? Emory Law Journal, 58:181.   
[28] Shaobo Li, Jie Hu, Y. Cui, and J. Hu. 2018. DeepPatent: Patent Classification with Convolutional Neural Networks and Word Embedding. Scientometrics, 117:721–744.   
[29] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.   
[30] Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel Weld. 2020. S2ORC: The Semantic Scholar Open Research Corpus. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4969–4983, Online. Association for Computational Linguistics.   
[31] Jonathan S Masur and Lisa Larrimore Ouellette. 2020. Patent Law: Cases, Problems, and Materials. SSRN.   
[32] Leland McInnes, John Healy, and James Melville. 2018. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv preprint arXiv:1802.03426.   
[33] P.S. Menell, M.A. Lemley, R.P. Merges, and S. Balganesh. 2021. Intellectual Property in the New Technological Age 2021: Vol. I Perspectives, Trade Secrets and Patents. Clause 8 Publishing.   
[34] Richard Miller. 2020. Technical Documentation for the 2019 Patent Examination Research Dataset (Patex) Release.   
[35] Amy Motomura. 2018. Who Becomes an Inventor in America?   
[36] Ankur Parikh, Xuezhi Wang, Sebastian Gehrmann, Manaal Faruqui, Bhuwan Dhingra, Diyi Yang, and Dipanjan Das. 2020. ToTTo: A Controlled Table-To-Text Generation Dataset. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1173–1186, Online. Association for Computational Linguistics.   
[37] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. 2011. Scikit-learn: Machine Learning in Python. the Journal of machine Learning research, 12:2825–2830.   
[38] F. Piroi, M. Lupu, A. Hanbury, and V. Zenz. 2011. CLEF-IP 2011: Retrieval in the Intellectual Property Domain. In CLEF.   
[39] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2019. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. arXiv preprint arXiv:1910.10683.   
[40] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research, 21:1–67.   
[41] Michael Razavi. 2016. Introducing the Pro Se Assistance Program. USPTO.   
[42] Tarek Saier and Michael Färber. 2020. unarXive: A Large Scholarly Data Set with Publications' Full-Text, Annotated In-Text Citations, and Links to Metadata. Scientometrics, 125(3):3085–3108.

[43] Bhaven Sampat and Heidi L Williams. 2019. How Do Patents Affect Follow-on Innovation? Evidence from the Human Genome. American Economic Review, 109(1):203–36.   
[44] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. ArXiv, abs/1910.01108.   
[45] Lei Sha, Patrick Hohenecker, and Thomas Lukasiewicz. 2021. Controlling Text Edition by Changing Answers of Specific Questions. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 1288–1299, Online. Association for Computational Linguistics.   
[46] Eva Sharma, Chen Li, and Lu Wang. 2019. BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 2204–2213, Florence, Italy. Association for Computational Linguistics.   
[47] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. 2017. SmoothGrad: Removing Noise by Adding Noise. arXiv preprint arXiv:1706.03825.   
[48] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic Attribution for Deep Networks. In International Conference on Machine Learning, pages 3319–3328. PMLR.   
[49] A Toole, N Pairolero, A Giczy, J Forman, C Pulliam, M Such, K Chaki, D Orange, A Thomas Homescu, J Frumkin, et al. 2020. Inventing AI: Tracing the Diffusion of Artificial Intelligence with US Patents.   
[50] Tung Tran and Ramakanth Kavuluru. 2017. Supervised Approaches to Assign Cooperative Patent Classification (CPC) Codes to Patents. In International Conference on Mining Intelligence and Knowledge Exploration, pages 22–34. Springer.   
[51] US Social Security. 2021. Beyond the Top 1000 Names.   
[52] USPTO. 2019. Progress and Potential: A Profile of Women Inventors on U.S. Patents. Technical report.   
[53] USPTO. 2020. FY 2020 Performance and Accountability Report. Technical report.   
[54] USPTO. 2021. U.S. Patent Statistics Chart Calendar Years 1963 - 2020.   
[55] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is All You Need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pages 6000–6010.   
[56] Ian Wetherbee. 2017. Google Patents Public Datasets: Connecting Public, Paid, and Private Patent Data. Technical report.   
[57] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2019. HuggingFace's Transformers: State-of-the-Art Natural Language Processing. arXiv preprint arXiv:1910.03771.   
[58] Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. 2020. Big Bird: Transformers for Longer Sequences. arXiv preprint arXiv:2007.14062.   
[59] Huiming Zhu, Chunhui He, Yang Fang, Bin Ge, Meng Xing, and Weidong Xiao. 2020. Patent Automatic Classification Based on Symmetric Hierarchical Convolution Neural Network. Symmetry, 12(2):186.

# A Dataset Checklist from the NeurIPS Datasets & Benchmarks Track

# 1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]   
(b) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]   
(c) Did you describe the limitations of your work? [Yes] See Section 4.   
(d) Did you discuss any potential negative societal impacts of your work? [Yes] See Section B (Data Card) of the Appendix.

# 2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [N/A]   
(b) Did you include complete proofs of all theoretical results? [N/A]

# 3. If you ran experiments (e.g., for benchmarks)...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] We publicly release our models, along with the data loading and tokenization code, on our website.   
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] See Section 6 and Section G (in the Appendix).   
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [No] Most of our empirical results came from a single run due to computational constraints. They are intended to be illustrative of the tasks that can be performed with our dataset. We therefore do not report error bars.   
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] We used four RTX 8000 GPUs from an academic cluster to perform our masked language modeling and summarization experiments, which were our the most compute-intensive experiments. Each of these experiments took approximately four days, for a total of approximately 16 GPU-days each.

# 4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes] We cite the USPTO.   
(b) Did you mention the license of the assets? [Yes] The dataset is released under the Creative Commons Attribution 4.0 International License.   
(c) Did you include any new assets either in the supplemental material or as a URL? [No]   
(d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [N/A] Our data is obtained from publicly-available sources; the patent applicants are aware that their applications will be posted publicly upon submission. We have been in contact with the Office of the Chief Economist at the USPTO on collecting and compiling the data the USPTO makes available on patent text and office actions.   
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [Yes] The patent data includes inventor and examiner information, but as discussed above, the authors are aware that this information will be publicly distributed upon submission. We do not believe offensive content is common in patent data.

# 5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A].   
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A].   
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A].

# B Data Card

# B.1 Dataset Description.

Dataset Summary. See Section 4.

Languages. The dataset contains English text only.

Domain. Patents (intellectual property).

Additional Details. The dataset contains utility patent applications filed to the USPTO between January 2004 and December 2018.

# B.2 Meta Information

Dataset Curators. The dataset was created by Mirac Suzgun, Luke Melas-Kyriazi, Suproteem K. Sarkar, Scott Duke Kominers, and Stuart M. Shieber.

Licensing Information. The dataset is released under the Creative Commons Attribution 4.0 International License.

Leaderboard/Benchmarks. The dataset has no associated public leaderboard; however, it introduces four useful benchmarks for the ML/NLP and Econ/IP communities, namely binary classification of patent decisions, multi-class IPC/CPC classification of patent codes at the subclass level, language modeling, and abstractive summarization. We release the models and code used in the main experiments on our GitHub codebase so that anyone can reproduce our empirical results, build their own models, make meaningful evaluations, and compare the performances of their models to ours.

For benchmarking purposes, we decided to split the data chronologically, establishing 2011-2016 as the training range and 2017 as the test range; however, this is not an absolute split for all the experiments. The users of our dataset can easily filter the metadata and fields, specify the years that they want to focus on, choose training and test sets as they wish, and thus create their own task or experiment setups. We believe the degree of freedom our dataset offers, as well as the allowance the dataset provides for different types of experiments and investigations that NLP and IP researchers and practitioners can take using the different data fields, is of paramount importance and novelty. In particular, we do not want, in any way, to restrict the usage and flexibility of the dataset by randomly or intentionally splitting according to some criteria (e.g., split the data chronologically, based on some certain grouping of IPC/CPC codes, etc.) for all the experiments.

# B.3 Dataset Structure

Data Format and Structure. Each patent application is defined by a distinct JSON file, named after its application number, and includes information about the application and publication numbers, title, decision status, filing and publication dates, primary and secondary classification codes, inventor(s), examiner, attorney, abstract, claims, background, summary, and full description of the proposed invention, among other fields. There are also supplementary variables, such as the small-entity indicator (which denotes whether the applicant is considered to be a small entity by the USPTO) and the foreign-filing indicator (which denotes whether the application was originally filed in a foreign country). In total, there are 34 data fields for each application. A full list of data fields used in the dataset is catalogued in the next section.

Data Instances. Each patent application in our patent dataset is defined by a distinct JSON file (e.g., 8914308.json), named after its unique application number. The format of the JSON files is as follows:

```json
{
    "application_number": "...",
    "publication_number": "...",
    "title": "...",
    "decision": "...",
    "date_produced": "...",
    "date_published": "...",
    "main_cpc_label": "...",
    "cpc_labels": ["...", "...", "..."],
    "main_ipcr_label": "...",
    "ipcr_labels": ["...", "...", "..."],
    "patent_number": "...",
    "filing_date": "...",
    "patent_issue_date": "...",
    "abandon_date": "...",
    "uspc_class": "...",
    "uspc_subclass": "...",
    "examiner_id": "...",
    "examiner_name_last": "...",
    "examiner_name_first": "...",
    "examiner_name_middle": "...",
    "inventor_list": [
    {
    "inventor_name_last": "...",
    "inventor_name_first": "...",
    "inventor_city": "...",
    "inventor_state": "...",
    "inventor_country": "..."
    }
    ],
    "abstract": "...",
    "claims": "...",
    "background": "...",
    "summary": "...",
    "full_description": "..." 
```

Data Fields. In addition to distinct JSON files for each application, we consolidate patent metadata into a CSV file that links patent application numbers with both the metadata in the JSON files (e.g., date\_produced, main\_cpc\_label) and additional covariates made available by the USPTO's PatEx data associated with the patent applications. These additional features are atty\_docket\_number, file\_location, wipo\_pub\_number, wipo\_pub\_date, patent\_issue\_date, small\_entity\_indicator, foreign, appl\_status\_desc. We include code for querying subsets of patent applications by all the features we include for NLP experiments.

Data Statistics. The dataset contains 4,518,263 utility patent applications filed to the USPTO between January 2004 and December 2018.

# B.4 Dataset Creation.

Source Data. The Harvard USPTO Patent Dataset synthesizes multiple data sources from the USPTO: While the full patent application texts were obtained from the USPTO Bulk Data Storage System (Patent Application Data/XML Versions 4.0, 4.1, 4.2, 4.3, 4.4 ICE, as well as Version 1.5) as XML files, the bibliographic filing metadata were obtained from the USPTO Patent Examination Research Dataset (in February, 2021).

Annotations. Beyond our patent decision label, for which construction details are provided in Section 2, the dataset does not contain any human-written or computer-generated annotations beyond those produced by patent applicants or the USPTO.

Personal and Sensitive Information. The dataset contains information about the inventor(s) and examiner of each patent application. These details are, however, already in the public domain and available on the USPTO's Patent Application Information Retrieval (PAIR) system, as well as on Google Patents and PatentsView.

Special Test Sets. We presented four mainstream NLP experiments on HUPD, namely binary classification of patent decisions, multi-class IPC/CPC classification of patent codes at the subclass level, language modeling, and abstractive summarization. We reported our results in Section 6.

Data Shift. A major feature of HUPD is its structure, which allows it to demonstrate the evolution of concepts over time. As we illustrate in Section 7, the criteria for patent acceptance evolve over

time at different rates, depending on category. We believe this is an important feature of the dataset, not only because of the social scientific questions it raises, but also because it facilitates research on models that can accommodate concept shift in a real-world setting.

# B.5 Considerations for Using the Data

The dataset was created to build new and useful benchmarks for NLP and IP experiments, facilitate research on patent analysis, and eventually help small entities and businesses improve the quality of their patent applications with no cost.

Social Impact of the Dataset. We hope that our dataset will have a positive social impact on the ML/NLP and Econ/IP communities. We discuss these considerations in more detail in Section H.4.

Impact on Underserved Communities and Discussion of Biases. The dataset contains patent applications in English, a language with heavy attention from the NLP community. However, innovation is spread across many languages, cultures, and communities that are not reflected in this dataset. Our dataset is thus not representative of all kinds of innovation. Furthermore, patent applications require a fixed cost to draft and file and are not accessible to everyone. One goal of our dataset is to spur research that reduces the cost of drafting applications, potentially allowing for more people to seek intellectual property protection for their innovations.

Limitations. Please see the “Limitations” subsection of Section 4.

# C Discussion of Potential Biases Embedded in the Dataset

In this section, we provide a further examination of our patent dataset for potential biases. For most of our analyses, we focus on the patent applications that were filed to the USPTO between 2011 and 2016 and investigate the correlation between inventor sex, entity size, and geographic location of patent inventors and patent outcomes. Our findings are consistent with the previous work [20, 7, 2, 52, 12], showing, among other things, that female inventors are notably underrepresented in the U.S. patenting system, that small and micro entities (e.g., independent inventors, small companies, non-profit organizations) are less likely to have positive outcomes in patent obtaining than large entities (e.g., companies with more than 500 employees), and that patent filing and acceptance rates are not uniformly distributed across the US. Our empirical findings suggest that any study focusing on the acceptance prediction task, especially if it is using the inventor information or the small-entity indicator as part of the input, should be aware of the the potential biases present in the dataset and interpret their results carefully in light of those biases.

# C.1 Differences in Patent Filing: Inventor Sex and Gender Identity

The USPTO collects and provides a limited set of data fields linked to inventors and examiners—in particular, it specifies each inventor's name, along with the city, state, country of their residence, and primary examiner's name for each patent application. As noted by Motomura [35], research efforts that examine the link between demographic information (such as race, gender identity, sex, ethnicity, and socioeconomic class of inventors) and patent acceptance prospects need to match the USPTO's publicly available records with other sources. We cannot measure inventor sex or gender identity directly. However, we can estimate inventor sex from inventor names by using imputation procedures. Following the methodology of Jensen et al. [20], we used the data from the United States Social Security Administration (SSA), together with HUPD, to estimate the distribution of female and male inventors who filed patent applications to the USPTO.

The Social Security Administration data contains names obtained from Social Security card applications for births that occurred in the United States between 1880 and 2020 (inclusive) [51]. Each year contains an ordered list of tuples—where each tuple includes a name, assigned sex, and total number of Social Security card applications associated with that name and sex in that given year—all ranked by their popularity. Using this data, we could empirically estimate the assigned sex distributions for 100,364 unique names. For instance, under this model, the name “Julia” is a female name with 99.6% probability, whereas the name “Taylor” is a female name with 74.4% probability.

We took this basic statistical model to predict the assigned sex (male or female) of the inventors for each patent application filed between 2011 and 2016. To be consistent with the framework of Jensen et al. [20], we set a strict threshold and assumed that an inventor is female if this model predicts that the first name of the inventor is a female name with at least 95% probability. Similarly, we assumed that an inventor is male if the model predicted the male class with at least 95% probability. $^{27}$ Additionally, we labeled a name as unisex if it appeared in the SSA's collection of names but had a probability below the threshold value under the female and male categories, and as foreign-sounding $^{28}$ if it did not appear in the SSA's collection of names at all. $^{29}$ Though we had 2,149,443 patent applications, in total, for 2011-2016, we excluded 781,601 of these patent applications from our analyses, since they contained inventors all of whose names were labeled as foreign-sounding under the statistical model. At the end, the final sample pool included 1,367,842 patent applications that contained at least one identifiable name under the statistical model.

We found 87.5% of these patent applicants had at least one male inventor, whereas 17.2% of the same set of patent applications had at least one female inventor. $^{30}$ The numbers were similar for patent obtaining as well: Of the 554,380 granted patent applications containing at least one identifiable name

under the statistical model, 88.0% of them had at least one male inventor, whereas 15.8% of them had at least one female inventor. These findings are consistent with the empirical findings of Jensen et al. [20] and reflect a sex disparity in both patent filing and patent obtaining in the United States. These sex differences in patent filings and obtaining indicate that female inventors are underrepresented in patenting, and may also have less success getting applications granted.

One family of analyses made possible by our dataset is a more systematic analysis of the disparities in patent acceptance probabilities by sex. For example, researchers could show that among patent applications with very similar distributions of text claims, sex predicts acceptance. Causal inference studies of this nature might provide more concrete tests for examiner bias in granting patent applications. But at the same time, the presence of sex disparities in the dataset may propagate biases, for example if inventor characteristics are included in the patent acceptance prediction task. We ask users of our dataset to consider these disparities carefully as they conduct their research.

# C.2 Differences in Patent Filing: Entity Size

The USPTO offers reduced application and maintenance fees for patent applicants who are qualified for the small or micro entity status, removing financial barriers to innovation for independent inventors and small businesses with limited resources. In our dataset, we include a small-entity indicator for each patent application—denoting whether the applicant is considered to be a small entity by the USPTO. Using HUPD, we explored how much the patent acceptance rates vary across different business sizes. As before, we restricted our focus to the 2011-2016 year range.

For clarity, we define three different entity statuses under the USPTO's application system. A small entity status is entitled to a patent applicant if the applicant is representing an individual (viz., independent inventor), representing a non-profit organization (e.g., university, 501(c)(3) organization, etc.), or a small business (e.g., a company that meets the standards set forth in 13 CFR 121.801 through 121.805). A micro entity is a small entity that meets additional criteria, such as not having been the inventor of a total of more than four patent applications. While small entities are eligible for a $50\%$ discount, micro entities are eligible for a $75\%$ discount. Applicants that pay regular application and maintenance fees (e.g., companies with more than 500 employees) are considered undiscounted (large) entities.

![](images/457e8a523aed38e5c22ea0949f0da1079ae8bf35a731b9119040dafbe9e85f1c.jpg)

<details>
<summary>bar</summary>

| Entity Category       | ACCEPTED | REJECTED | CONT-ACCEPTED | CONT-REJECTED | PENDING | CONT-PENDING |
| --------------------- | -------- | -------- | ------------- | ------------- | ------- | ------------ |
| Small                 | 150000   | 145000   | 75000         | 50000         | 45000   | 25000        |
| Micro                 | 15000    | 25000    | 5000          | 3000          | 4000    | 1000         |
| Undiscounted (Large)   | 700000   | 280000   | 265000        | 80000         | 215000  | 85000        |
</details>

Figure 5: Distribution of patent decision outcomes across small, micro, and large entities from 2011 to 2016. Large entities have higher acceptance rates than small and micro entities.

Figure 5 shows the distribution of patent outcomes across three entity categories—namely small, micro, and undiscounted (large)—between 2011 and 2016. Applications filed by small and micro entities constitute only a small fraction of the patent applications. Moreover, small entities have lower acceptance rates than large entities. Finally, large entities seem to not only file more CONT-applications but also have higher CONT-acceptance rates than small entities.

Additionally, we looked at the representation of female inventors in different entity categories using the assigned sex estimation model from the previous section. Of the patent applications that had at least one male, female, or unisex-sounding American name under the statistical model, 17.8% had at least one female inventor in their inventor list in the case of the small entities. This number was slightly higher (21.8%) for the micro entities and slightly lower (16.9%) for the large entities.

# C.3 Differences in Patent Filing: Geographic Location

The USPTO limits the inventor residence information to the city and state (or foreign country) in patent applications. In our examination, we measured the geographical distribution of US-based inventors at the state-level. The top-left plot in Figure 6 shows the geographical distribution of the state-level residences of the inventors who filed patent applications to the USPTO between 2011 and 2016. According to this plot, almost a quarter of the US-based inventors had a residence in California, the most populous U.S. state, at the time of their submissions. On the other hand, 574 inventors had a residence in Alaska. The top-right plot in Figure 6 shows the geographical distribution of the residences of the US-based inventors with granted patents (for the same year-range). Finally, the plot in the second row illustrates the average patent acceptance rate of inventors from each state in the U.S. for the 2011-2016 year range. While the success rates vary across different parts of the country, some states with large corporations and concentrated entrepreneurship areas (including California, New York, Texas, and Michigan) appear to have noticeably high patent success rates. In fact, Michigan, with its $46.1\%$ success rate, has the highest patent acceptance rate amongst all the U.S. states, while Nevada $(23.1\%)$ has the lowest. These results are consistent with the findings of Bell et al. [2].

![](images/bbea3c20d6224c2086c0486e9241c028c1f2538cf466adcfc0bbce6b324b68e6.jpg)  
Figure 6: Top-left: The geographical distribution of the state-level residences of the US-based inventors who filed patent applications to the USPTO between 2011 and 2016. Top-right: The subset of the inventors from before who had granted patents. Bottom (second row): The patent acceptance rates across different states in the US.

# C.4 Differences in Patent Examination: Examiner Sex and Gender Identity

We additionally estimated the distribution of estimated assigned sex of the examiners at the USPTO. We used the same subset of the dataset as before and focused on the examiners of the patent applications that were filed between 2011 and 2016. As shown on the left plot in Figure 7, of the 10,484 patent examiners in consideration, $58.8\%$ of them had predicted male names, $20.6\%$ had predicted female names, $11.4\%$ had predicted foreign-sounding names, and the rest $(9.2\%)$ had unisex names, according to the basic statistical model used in Section C.1.

The right plot in Figure 7 illustrates the relationship between patent decision outcomes and estimated sexes of patent examiners. According to our estimation of assigned sex, and among examiners whose names we observe in the SSA data, male patent examiners have higher accepted rates than female

patent examiners: The “Accepted”/“Rejected” ratio is close to 2.0 for male patent examiners and 1.4 for female patent examiners.

![](images/18314288126bed44d3e0847e576ef285260d1c3b7ab7193f0695dd034b200dbe.jpg)

<details>
<summary>bar</summary>

| Examiner Name | Number of Patent Examiners |
| ------------- | --------------------------- |
| Male          | 6000                        |
| Female        | 2000                        |
| Unisex        | 1000                        |
| Foreign       | 1200                        |
</details>

![](images/d6ca4c137700c49ddbe7c9338bc893b132316cb2cc646bde4c07de45deb873fe.jpg)

<details>
<summary>bar</summary>

| Examiner Name | ACCEPTED | REJECTED | CONT-ACCEPTED | CONT-REJECTED | PENDING | CONT-PENDING |
| ------------- | -------- | -------- | ------------- | ------------- | ------- | ------------ |
| Male          | 510000   | 260000   | 195000        | 75000         | 160000  | 60000        |
| Female        | 345000   | 245000   | 125000        | 70000         | 90000   | 35000        |
| Unisex        | 195000   | 95000    | 70000         | 25000         | 45000   | 20000        |
| Foreign       | 200000   | 105000   | 70000         | 25000         | 50000   | 20000        |
</details>

Figure 7: Relationship between patent decision outcomes and patent examiners' estimated sexes.

# D Further Comparisons with Existing Datasets

# D.1 Comparison with BIG PATENT

It is natural to compare HUPD to one of the most widely-used existing patent datasets in NLP, BIGPATENT [46]. In comparison to BIGPATENT, HUPD contains not only significantly more patent documents (4.5 million vs. 1.3 million) but also much richer (bibliographic) meta-information about each patent document. Since BIGPATENT was proposed primarily for the task of abstractive summarization, it includes only four of the 34 data fields available in HUPD (publication number, application number, abstract, and description). Notably, BIGPATENT does not include the claims section, which is often considered to be the most important section for describing the contributions of an invention. Many tasks that can be performed with HUPD (analysis of text over time, fine-grained classification, etc.) are not possible to perform with the BIGPATENT data. Additionally, whereas BIGPATENT contains only granted patents, HUPD contains patent applications. As a result, prediction of patent acceptance, a new task we introduce to NLP, is not possible with BIGPATENT. Finally, the BIGPATENT is pre-tokenized using NLTK [4], which might cause issues for all applications that contain chemical formulae and mathematical equations. HUPD, by contrast, provides the raw patent text and may be tokenized using a custom vocabulary for tasks in which it is important to correctly represent formulae or equations.

# D.2 Comparison with Large Scientific Text Corpora

One noteworthy domain-specific NLP corpus is Allen AI's Semantic Scholar Open Research Corpus (S2ORC; Lo et al. [30]), which contains 81 million English-language academic papers spanning multiple academic disciplines. Of these papers, the authors provide full structured text from 8.1M open-access PDF files and 1.5M $\mathrm{LATEX}$ source files, along with their appropriate metadata. They include the title, author list, publication year, publication venue (or journal), abstract, paragraphs of the body of the text, section heading, figure and table information (along with their corresponding captions), equations, headers, footers, inline citations, and prior art as data fields in their JSONlines files. Due to the nature of academic papers, however, not all the papers have content for these data fields. With its size and comprehensive structured metadata, S2ORC is comparable to HUPD in its nature. Another concurrent work by Saier and Färber [42] introduced the unarXive dataset, a collection of over one million academic papers with links to almost 2.7 million unique publications; however, this dataset is not as comprehensive and well-structured as S2ORC or our dataset. Furthermore, Dasigi et al. [6] introduced Qasper, a dataset of information-seeking question-answering (QA) dataset over 1585 NLP papers, but their focus was limited to the evaluation of document-grounded QA models. Contract Understanding Atticus Dataset (CUAD; Hendrycks et al. [19]) is another annotated NLP dataset that contains more than 500 contracts with over 13,000 expert annotations and 41 label categories. CUAD is unique in the sense that the labels were annotated by legal experts and poses the identification of the relationship between different subtexts of a contact with different label categories as its primary task. WikiBio [25] is also a large scientific corpus that is worth mentioning: It contains almost 0.73 million unique biographies of famous people extracted from English Wikipedia, where each biography contains the first paragraph of the article and the infobox (fact table). The WikiBio dataset has been historically used for table-to-text generation [36], but it has been also adopted for question-answering [45]. We remark that our dataset provides a diverse range of language-based tasks to the community, including, but not limited to, binary classification of patent decision outcomes, multi-class IPC/CPC classification, patent clustering, long-sequence language modeling, abstractive summarization, document-level information extraction, named-entity recognition and extraction. HUPD is also notable in being one of the largest publicly-available collection of well-structured and domain-specific textual data. Finally, we remark that it has the potential to facilitate research and development in not only NLP but also IP.

# D.3 Comparison with Large-Scale Web-Scraped NLP Datasets

Given the size of HUPD (350GB of raw text), it is also possible to compare it to the extremely large-scale NLP corpora currently used for language model pretraining, such as Colossal Clean Crawled Corpus (C4) [39], the Pile [11], and OpenWebText [15]. These datasets contain vast quantities of text scraped from the Internet, some of which is derived from patents. In fact, a recent analysis of the C4 dataset [10] found that “patents.google.com” is the single most-frequent source of text in the corpus, as measured by number of tokens. Furthermore, Dodge et al. [10] found that this patent text was not

clean: A significant percentage was machine-translated from non-English languages and/or extracted from images with OCR. HUPD differs markedly from these web-scraped datasets because we obtain our text directly from the USPTO BDSS; textual content in our dataset is clean, carefully-extracted, and consistently-formatted. Additionally, whereas other large-scale text datasets typically contain only a stream of text, HUPD contains numerous structured data fields for each document. $^{31}$ The main focus of HUPD is not on pre-training large language models from scratch, but due to its large size and high quality, it can be a good complement to the existing extremely-large-scale web-scraped NLP corpora.

# D.4 Comparison with Popular Patent Search Tools and Large Repositories of Patent Data

In this section, we compare HUPD to popular English-language-based patent search tools and large repositories of patent data which are not constructed primarily for the ML/NLP community.

Publicly Available Patent Search and Analysis Tools. To the best of our knowledge, Google Patents is the most comprehensive and most diverse query-based patent search tool, containing over 120 million distilled patent documents (including patent applications, pre-grant publications, and granted patents), from more than one hundred patent offices all around the world. Similarly, PatentsView is a web-based patent data visualization and search tool that aims to improve the quality, impact, evaluation, and visibility of patents and patent applications filed to the USPTO. PatentsView is supported by the Office of the Chief Economist at the USPTO. The current PatentsView API allows each user to make up to 45 queries per minute. Using the API tool, users can get access to the full text of the title and abstract section of patent documents, as well the inventor, assignee, location, CPC/USPC classification details, among some other numerical and bibliographic statistics. The Lens Patent API allows search over 140 million patent meta-records worldwide. It has a free 14-day trial period (with limited API requests); afterwards, users need to pay fee for customized access, though pricing is determined on the nature and volume of the use case.

Aimed to replace the USPTO's existing search platform (Pub EAST, Pub WEST, Pat/FT, and App/FT) with a single, more comprehensive, and more modern online service, Patent Public Search is the USPTO's own recent search tool can perform and display searches over the USPTO's three big databases (namely, US-PGPUB, USPAT, and and USOCR). Patent Public Search is based on the Patents End-to-End (PE2E) search tool that the USPTO examiners use to identify prior art. There are additional online search tools developed and supported by different patent offices, such as WIPO Patentscope and Espacenet (supported by the WIPO and European Patent Office, respectively), that seek to improve public access to patent information. $^{32}$

Managed by the EPO, the PATSTAT Global is a patent statistics database that contains bibliographic data and legal information about more than 110 million patent documents (patent applications and patents) from the patent offices of many industrial and developing countries from 1980 onwards. $^{33}$ While it has wide bibliographic coverage and is regularly updated, the PATSTAT database does require users to pay a subscription free for usage. It allows users to retrieve the title, abstract, inventor/application, classification code, technical field, other bibliographic metadata, and legal history of patent documents. $^{34}$

Large Repositories of Raw US Patent Data. The USPTO makes several patent raw text datasets publicly available in XML formats. $^{35}$ As noted before, Patent Application Data/XML (Versions 4.0-4.4 ICE and Version 1.5) from the BDSS, together with the Patent Examination Research Dataset (PatEx) from USPTO's Public Patent Application Information Retrieval (PAIR) system, is the core data we have used to construct HUPD. The patent applications in the USPTO Bulk Data system

are organized by their filing application years and concatenated in big bulk XML files. Each patent application needs to be carefully extracted and unconcatenated back to individual XML documents for clean data processing purposes. $^{36}$ Additionally, the USPTO makes text data and rich bibliographic information for accepted patents available via its PatentsView service.

The Google Patents Public Datasets collection makes consistently-formatted patent text and filing information available to users, though its primary purpose is not to be used as an NLP dataset. The collection contains bibliographic information on more than 90 million patent documents from 17 countries, in addition to the full textual content of millions of US patent documents, provided by IFI CLAIMS Patent Services [56]. The data from this repository can downloaded using BigQuery. $^{37}$

The MAtrixware REsearch Collection (MAREC) Dataset $^{38}$ is a static, standardized, and multilingual dataset of patent applications and granted patents. There are 19 million raw patent documents in the dataset—written in 19 languages (though the majority of them are written in English, German, and French), spanning an almost three-decade period (1976-2008), coming from the European Patent Office (EPO), Japan Patent Office (JPO), United States Patent and Trademark Office (USPTO), as well as the World Intellectual Property Organization (WIPO). The documents in the MAREC dataset are normalized to a uniform XML format. The standardized data fields are primarily the publication number, filing date, country of invention, language, citation information, inventor information, main subject classifications (e.g., IPC codes). Almost half of the documents contain full textual content of the inventions. The dataset, whose raw data size (almost 600GB) is comparable to ours, can be accessed via BigQuery and via bulk download.

We include a more directed comparison of HUPD with other patent datasets constructed for NLP patent analysis in Table 1 and Section D.1.

# E Glossary

In this section, we provide a list of commonly used patent-related terms, concepts, and acronyms in this paper. While this list is not meant to be extensive or comprehensive, it is still inclusive and informative; it can be used by our readers as a reference guide as they read the paper. Unless otherwise marked with an asterisk $*$ at the end, the definition for each term was taken from the USPTO's Glossary. $^{39}$

- Abandonment: A patent application becomes abandoned for failure to file a complete and proper reply as the condition of the application may require within the time period provided under 37 CFR § 1.134 and § 1.136 unless an Office action indicates otherwise. Abandonment may be either of the invention or of an application. An abandoned application, in accordance with 37 CFR §§ 1.135 and 1.138, is one which is removed from the Office docket of pending applications.   
- Applicant: Inventor or joint inventors who are applying for a patent on their own invention, or the person mentioned in 37 CFR 1.42, 1.43 or 1.47 who is applying for a patent in place of the inventor.   
- Application filing date: The date the USPTO receives an application in English that includes all the following: (1) The applicant's name, (2) A name and address for correspondence (3) A clear drawing of the mark to be registered, (4) A list of the goods or services, and (5) An application filing fee for at least one class of goods or services.   
- CIP (Continuation-in-Part): An application filed during the lifetime of an earlier nonprovisional application, repeating some substantial portion or all of the earlier nonprovisional application and adding matter not disclosed in the earlier nonprovisional application.   
- Claims: define the invention and are what aspects are legally enforceable. The specification must conclude with a claim particularly pointing out and distinctly claiming the subject matter which the applicant regards as his invention or discovery. The claim or claims must conform to the invention as set forth in the remainder of the specification and the terms and phrases used in the claims must find clear support or antecedent basis in the description so that the meaning of the terms in the claims may be ascertainable (clearly understood) by reference to the description. (See 37 CFR § 1.58(a)).   
- Classification: Patents are classified (organized) in the U.S. by a system using a 3 digit class and a 3 digit subclass to describe every similar grouping of patent art. A single invention may be described by multiple classification codes.   
- Continuation: A second application for the same invention claimed in a prior nonprovisional application and filed before the first application becomes abandoned or patented.   
- Continuing application: A continuation, divisional, or continuation-in-part patent application.   
• CPC: Cooperative Patent Classification.\*   
- Design patent: May be granted to anyone who invents a new, original, and ornamental design for an article of manufacture.   
- Design patent application: An application for a patent to protect against the unauthorized use of new, original, and ornamental designs for articles of manufacture.   
- Divisional application: A later application for an independent or distinct invention disclosing and claiming (only a portion of and) only subject matter disclosed in the earlier or parent application.   
- Drawing (patent): Patent drawings must show every feature of the invention as specified in the claims. Omission of drawings may cause an application to be considered incomplete but are only required if drawings are necessary for the understanding of the subject matter sought to be patented.   
- Express abandonment: A patent application may be expressly abandoned by filing a written declaration of abandonment identifying the application in the United States Patent and Trademark Office. Express abandonment becomes effective when an appropriate official

of the Office takes action thereon. Express abandonment of the application may not be recognized by the USPTO before the date of issue or publication unless it is actually received by appropriate officials in time to act. Abandonment may be either of the invention or of an application. An abandoned application, in accordance with 37 CFR 1.135 and 1.138, is one which is removed from the USPTO docket of pending applications.

\- Filing date: the date of receipt in the Office of an application which includes (1) a specification containing a description and, if the application is a nonprovisional application, at least one claim, and (2) any required drawings.

\- Final office action (rejection): An Office action on the second or any subsequent examination or consideration by an examiner that is intended to close the prosecution of a nonprovisional patent application. Applicant's reply under 37 CFR 1.113 to a final rejection is limited either to an appeal in the case of rejection of any claim to the Board of Patent Appeals and Interferences (37 CFR 1.191) or to an amendment complying with the requirements set forth in the Office action (37 CFR 1.114 or 1.116). Reply to a final rejection must comply with 37 CFR 1.114 or include cancellation of, or appeal from the rejection of, each rejected claim. If any claim stands allowed, the reply to a final rejection must comply with any requirements or objections as to form (37 CFR 1.113(c)).

• FY (fiscal year): The federal fiscal year extends from October 1 through September 30.

\- Independent claim: A claim that does not refer back to or depend on another claim.

\- Infringement (patent): Unauthorized making, using, offering to sell, selling or importing into the United States any patented invention.

\- Inventor: One who contributes to the conception of an invention. The patent law of the United States of America requires that the applicant in a patent application must be the inventor.

• IP: Intellectual property.

\- IPC: International Patent Classification.\*

\- Issue date: The date that a patent application becomes a U.S. patent. The issue date is the date that patent rights can be exercised. U.S. patents are always issued on Tuesdays.

• MPEP: Manual of Patent Examining Procedure.

\- National stage application: An application which has entered the national phase of the Patent Cooperation Treaty by the fulfillment of certain requirements in a national Office, which is an authority entrusted with the granting of national or regional patents. Such an application is filed under 35 U.S.C. §371 in the United States and is referred to as a “371 application.”

\- Non-final Office action: An Office action letter that raises new issues and usually is the first phase of the examination process. An examining attorney will issue a non-final Office action after reviewing the application for the first time. If a new issue arises after the applicant responds to the first non-final Office action, the examining attorney will issue another non-final Office action that sets forth the new issue(s) and continues any that remain outstanding. Applicants must respond to non-final Office action letters within 6 months from the date they are issued to avoid abandonment of the application.

\- Nonprovisional patent application: An application for patent filed under 35 U.S.C. 111(a) that includes all patent applications (i.e., utility, design, plant, and reissue) except provisional applications. The nonprovisional application establishes the filing date and initiates the examination process. A nonprovisional utility patent application must include a specification, including a claim or claims; drawings, when necessary; an oath or declaration; and the prescribed filing fee.

\- Notice of abandonment: A written notification from the USPTO that an application has been declared abandoned or, in other words, is no longer pending. If the application was abandoned unintentionally or due to Office error, the applicant has a deadline of two months from the issue date of the notice of abandonment to file either (1) a petition to revive the application or (2) a request to reinstate the application.

\- Notice of allowability: A notification to the patent applicant that the application has been placed in condition for allowance.

\- Notice of allowance (NOA): A written notification from the USPTO that a specific mark has survived the opposition period following publication in the Official Gazette, and has consequently been allowed for registration. It does not mean that the mark has registered yet. Receiving a notice of allowance is another step on the way to registration. Notices of allowance are only issued for applications that have been filed based on “intent to use”. The notice of allowance is important because the issue date of the Notice of Allowance establishes the due date for filing a statement of use. After receiving the Notice of Allowance, the applicant must file a statement of use or a request for an extension of time to file a statement of use within 6 months from the issue date of the notice. If thea applicant fails to timely file a statement of use or a request for an extension of time to file a statement of use, the application will be abandoned.

\- Office action: A letter from a trademark examining attorney setting forth the legal status of a trademark application. There are several types of Office actions: examiner's amendments, priority actions, non-final Office actions, final Office actions, and suspension inquiry letters.

\- Parent application: The term “parent” is applied to an earlier application of the inventor disclosing a given invention.

\- Patent: A property right granted by the Government of the United States of America to an inventor “to exclude others from making, using, offering for sale, or selling the invention throughout the United States or importing the invention into the United States” for a limited time in exchange for public disclosure of the invention when the patent is granted.

\- Patent family: A patent family is the same invention disclosed by a common inventor(s) and patented in more than one country.

\- Patent number: A unique number assigned to a patent application when it issues as a patent.

\- PG Pub: Pre-Grant Publication of patent application at 18 months from priority date:

\- Plant application (patent): Applications to protect invented or discovered, asexually reproduced plant varieties.

\- Provisional patent application: A provisional application for patent is a U. S. national application for patent filed in the USPTO under 35 U.S.C. § 111(b). It allows filing without a formal patent claim, oath or declaration, or any information disclosure (prior art) statement. It provides the means to establish an early effective filing date in a nonprovisional patent application filed under 35 U.S.C § 111(a) and automatically becomes abandoned after one year. It also allows the term “Patent Pending” to be applied.

\- PTO: Patent and Trademark Office, former designation for USPTO.

\- Publication number: A number assigned to the publication of patent applications filed on or after November 29, 2000. It includes the year, followed by a seven digit number, followed by a kind code. Example 200011234567A1.

• USPTO: United States Patent and Trademark Office.

• USPC: United States Patent Classification.\*

\- Utility patent: May be granted to anyone who invents or discovers any new, useful, and nonobvious process, machine, article of manufacture, or composition of matter, or any new and useful improvement thereof.

\- Utility patent application: Protect useful processes, machines, articles of manufacture, and compositions of matter.

• WIPO: World Intellectual Property Organization.

\- Withdrawn patent: An allowed application for patent in which the applicant files correspondence to withdraw the patent from issue; thus preventing it from issuing on the patent issue date. The printed document is sometimes available on the day of publication, but is later retracted and will not be available in the patent database. No copy of the patent document will appear on the official USPTO web site.

# F Additional Tables and Figures

# F.1 Binary Decision Classification

# F.1.1 Complete Version of Table 4

<table><tr><td>IPC - Section</td><td>BernNB</td><td>MultiNB</td><td>Logistic</td><td>CNN</td><td> $DistilBERT^{FT}$ </td><td> $BERT^{FT}$ </td><td> $RoBERTa^{FT}$ </td></tr><tr><td>GO6F - Abstract</td><td>61.86</td><td>61.47</td><td>58.24</td><td>60.97</td><td>61.53</td><td>61.28</td><td>61.31</td></tr><tr><td>GO6F - Claims</td><td>63.96</td><td>62.06</td><td>58.02</td><td>63.38</td><td>63.37</td><td>62.97</td><td>63.25</td></tr><tr><td>H01L - Abstract</td><td>58.98</td><td>59.05</td><td>58.54</td><td>60.71</td><td>61.46</td><td>61.85</td><td>61.85</td></tr><tr><td>H01L - Claims</td><td>60.97</td><td>60.29</td><td>59.53</td><td>62.63</td><td>62.50</td><td>61.61</td><td>61.94</td></tr><tr><td>H04L - Abstract</td><td>59.35</td><td>58.75</td><td>58.75</td><td>59.89</td><td>60.54</td><td>60.52</td><td>60.05</td></tr><tr><td>H04L - Claims</td><td>62.13</td><td>61.04</td><td>58.04</td><td>62.34</td><td>61.42</td><td>61.47</td><td>61.74</td></tr><tr><td>H04W - Abstract</td><td>56.01</td><td>55.20</td><td>55.79</td><td>57.82</td><td>56.42</td><td>56.39</td><td>57.01</td></tr><tr><td>H04W - Claims</td><td>57.76</td><td>56.85</td><td>55.58</td><td>59.72</td><td>58.97</td><td>58.94</td><td>59.22</td></tr><tr><td>H04N - Abstract</td><td>60.74</td><td>60.64</td><td>58.79</td><td>60.37</td><td>62.01</td><td>61.93</td><td>61.51</td></tr><tr><td>H04N - Claims</td><td>62.51</td><td>61.01</td><td>57.53</td><td>63.98</td><td>62.82</td><td>61.98</td><td>62.14</td></tr><tr><td>A61B - Abstract</td><td>59.15</td><td>58.81</td><td>57.31</td><td>58.75</td><td>58.36</td><td>59.58</td><td>59.66</td></tr><tr><td>A61B - Claims</td><td>59.30</td><td>59.12</td><td>57.25</td><td>59.49</td><td>60.15</td><td>61.20</td><td>61.00</td></tr><tr><td>A61K - Abstract</td><td>58.14</td><td>57.82</td><td>55.46</td><td>56.85</td><td>58.47</td><td>56.45</td><td>57.08</td></tr><tr><td>A61K - Claims</td><td>57.31</td><td>57.93</td><td>56.20</td><td>59.06</td><td>58.72</td><td>57.91</td><td>57.84</td></tr><tr><td>GO1N - Abstract</td><td>59.85</td><td>59.89</td><td>57.25</td><td>59.98</td><td>59.00</td><td>60.30</td><td>61.10</td></tr><tr><td>GO1N - Claims</td><td>58.06</td><td>57.97</td><td>58.37</td><td>59.80</td><td>60.16</td><td>60.34</td><td>60.97</td></tr><tr><td>GO6Q - Abstract</td><td>61.53</td><td>61.64</td><td>58.52</td><td>60.46</td><td>61.23</td><td>61.09</td><td>61.56</td></tr><tr><td>GO6Q - Claims</td><td>63.96</td><td>63.31</td><td>57.17</td><td>62.90</td><td>61.88</td><td>62.19</td><td>63.25</td></tr></table>

Table 7: (Complete version of Table 4) Baseline performances of our models on the binary classification of patent decision task. All the models were trained and evaluated on the patent applications filed to the USPTO between 2011 and 2016. We note that BernNB and MultiNM denote Bernoulli and Multionomial NB classifiers trained on world-level unigrams (with minimum frequency of 3), respectively; Logistic a logistic regression model consisting of an embedding layer followed by a single linear layer trained on world-level unigrams; and CNN a Convolutional Neural Network with a 2-D convolutional layer and a max-pooling layer trained on world-level unigrams (with minimum frequency of 3). The superscript $^{FT}$ on the Transformer models denotes that these models were fine-tuned, not trained from scratch.

# F.1.2 Top IPC/CPC Codes and Their Brief Descriptions

<table><tr><td>IPC/CPC Code</td><td>Description</td></tr><tr><td>G06F</td><td>Electric Digital Data Processing</td></tr><tr><td>H01L</td><td>Semiconductor Devices</td></tr><tr><td>A61K</td><td>Preparations for Medical Purposes</td></tr><tr><td>H04L</td><td>Transmission of Digital Information</td></tr><tr><td>H04N</td><td>Pictorial Communication, e.g. Television</td></tr><tr><td>A61B</td><td>Diagnosis, Surgery, Identification</td></tr><tr><td>G06Q</td><td>Data Processing Systems or Methods</td></tr><tr><td>H04W</td><td>Wireless Communication Networks</td></tr><tr><td>G01N</td><td>Investigating or Analyzing Materials</td></tr></table>

Table 8: Brief descriptions of the most common nine IPC/CPC subclass codes present in our analysis year ranges.

# F.2 Multi-Class IPC/CPC Classification

# F.2.1 Words with the Highest Weights in the Bernoulli NB Classifier

<table><tr><td>IPC</td><td>Words with Highest Weights</td></tr><tr><td>G06F</td><td>data, includes, device, semiconductor, system, substrate, one, method, second, least, structure, display, information, plurality, layer, present</td></tr><tr><td>H01L</td><td>user, film, second, access, data, one, semiconductor, wherein, includes, memory, unit, region, device, portion, operation, area, used, layer</td></tr><tr><td>A61K</td><td>device, including, computer, least, image, forming, layer, files, data, part, package, driving, conductive, includes, processing, direction</td></tr><tr><td>H04L</td><td>data, includes, semiconductor, device, disposed, system, one, substrate, method, layer, display, second, plurality, least, structure, content</td></tr><tr><td>H04N</td><td>provided, data, includes, one, device, semiconductor, method, formed, second, system, electrical, least, substrate, display, information, electrode</td></tr><tr><td>A61B</td><td>data, one, includes, device, method, substrate, electrode, second, semiconductor, memory, least, light, unit, layer, application, set, plurality</td></tr><tr><td>G06Q</td><td>data, device, substrate, includes, semiconductor, one, method, structure, system, display, present, plurality, image, non, least, n, content</td></tr><tr><td>H04W</td><td>semiconductor, data, device, layer, includes, disposed, high, one, display, substrate, method, second, structure, system, machine, plurality</td></tr><tr><td>G01N</td><td>data, device, includes, one, semiconductor, method, bus, least, layer, including, memory, substrate, image, computer, content, second, application</td></tr></table>

Table 9: Examples of words (excluding stopwords) with highest probability weights in their respective IPC subclasses in our Bernoulli naive Bayes classifier trained to predict the IPC code of a patent application based on the words in its abstract section. For a given IPC subclass label y, we looked at the probability value $p(x_{i}|y)$ for each word $x_{i}$ in the vocabulary, and listed the words with the highest probability values.

# F.2.2 Confusion Matrix

![](images/5205de76b82f5da0d676dd22d2fea0dbc3e74e956f7ae8baa6a75e90d6a02e31.jpg)

<details>
<summary>heatmap</summary>

| Class | Predicted Labels | Confusion Matrix Value |
|-------|------------------|------------------------|
| G06F  | G07D             | ~0.8                   |
| H01L  | A47D             | ~0.7                   |
| H04L  | A22C             | ~0.6                   |
| H04W  | F25C             | ~0.5                   |
| H04N  | B42D             | ~0.4                   |
| A61B  | B60D             | ~0.3                   |
| A61K  | B29B             | ~0.2                   |
| G01N  | A44C             | ~0.1                   |
| G06K  | D05B             | ~0.0                   |
</details>

Figure 8: Confusion matrix of IPC code classification at the subclass level. This matrix was obtained from the DistilBERT model that was fine-tuned on the abstract sections. The IPC codes were ordered by their sizes from left to right and from top to bottom, respectively. The light diagonal line present on the center figure represents high recall values. The diagonal line disappears towards the lower right corner, since patents belonging to those IPC subclasses do not appear in our test set.

# F.2.3 Saliency Maps

To have a better understanding and appreciation of the behavior of our neural classifiers, we also made queries to identify which parts of the input texts our models might be attending to when making their predictions. To that end, we availed ourselves of various simple gradient-based saliency methods, such as SmoothGrad [47] and Integrated Gradient (IG; [48]). $^{40}$ We once again turned to the DistilBERT models for our analyses due to their relatively better performance and their potential to learn more complex representations. Figure 9 illustrates how much each input feature contributes to the prediction made by the DistilBERT model (trained on the abstract section as shown in Table 5). This analysis suggests the tokens that the model is paying attention to tend to be those that are most indicative of the technology area that each patent belongs to. $^{41}$

<table><tr><td>Patent No.</td><td>Predicted Label</td><td>True Label</td><td>Saliency Map</td></tr><tr><td>US9272708B2</td><td>B60W</td><td>B60W</td><td rowspan="2">[CLS] a vehicle system includes an autonomous mode controller and an entertainment system controller . the autonomous mode controller controls a vehicle in an autonomous mode . the entertainment system controller presents media content on a first display while the vehicle is operating in the autonomous mode and on a second display when the vehicle is operating in a non - autonomous mode . a method includes determining whether a vehicle is operating in an autonomous mode , presenting media content on a first display while the vehicle is operating in the autonomous mode , and transferring presentation of the</td></tr><tr><td colspan="3">&quot;Autonomous Vehicle Entertainment System&quot;</td></tr><tr><td>US9406017B2</td><td>G06N</td><td>G06N</td><td rowspan="2">[CLS] a system for training a neural network . a switch is linked to feature detectors in at least some of the layers of the neural network . for each training case , the switch randomly selective ##ly di ##sable ##s each of the feature detectors in accordance with a pre ##con ##fi ##gur #ned probability . the weights from each training case are then normal #ized for applying the neural network to test data . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]</td></tr><tr><td colspan="3">&quot;System and Method for Addressing Overfitting in a Neural Network&quot;</td></tr><tr><td>US9733811B2</td><td>G06F</td><td>G06F</td><td rowspan="2">[CLS] a method for profile matching includes receiving a plurality of user profiles , each user profile comprising traits of a respective user . the method includes receiving a preference indication for a first user profile of the plurality of user profiles . the method also includes determining a potential match user profile of the plurality of user profiles based on the preference indication for the first user profile . the method also includes presenting the potential match user profile to a second user . [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]</td></tr><tr><td colspan="3">&quot;Matching Process System and Method&quot;</td></tr></table>

Figure 9: Saliency maps generated using Integrated Gradient [48] for the DistilBERT model trained to predict the IPC subclass of a patent application based on its abstract section. All our visualizations were created using the Captum library [22]. The green color highlights tokens that contribute more heavily toward the class label, and the red color signifies tokens that contribute less strongly toward the class label. Our models predicted the class label correctly in each example. These three examples are the abstract sections of three patent applications by Ford, Google, and Tinder, respectively. The input tokens that contribute the most to our predictions appear to be associated with the primary technology areas of these patent applications. These qualitative findings confirm our belief that our models have learned useful domain-specific features about IPC subclasses.

# F.3 Abstractive Summarization

# F.3.1 Generated Summaries

<table><tr><td>Claims</td><td>1. An optical coherent receiver for an optical communication network, said optical coherent receiver being configured to receive a modulated optical signal and to process said modulated optical signal for generating an in-phase component and a quadrature component, said in-phase component and said quadrature component being electrical signals, said optical coherent receiver comprising a power adjuster in turn comprising: a multiplying unit configured to multiply said in-phase component by an in-phase gain thereby providing a power-adjusted in-phase component, and to multiply said quadrature component by a quadrature gain thereby providing a power-adjusted quadrature component; and a digital circuit connected between output and input of said multiplying unit and configured to compute: a common gain indicative of a sum of a power of said power-adjusted in-phase component and a power of said power-adjusted quadrature component, and a differential gain indicative of a difference between said power of said power-adjusted in-phase component and said power of said power-adjusted quadrature component; and said in-phase gain as a product between said common gain and said differential gain, and said quadrature gain as a ratio between said common gain and said differential gain. 2. An optical coherent receiver according to claim 1, wherein it further comprises an analog-to-digital unit connected at the input of said power adjuster, said analog-to-digital unit being configured to ...</td></tr><tr><td>Generated Abstract</td><td>An optical coherent receiver for an optical communication network is provided. The optical coherent receiver is configured to receive a modulated optical signal and to process the modulated optical signal for generating an in-phase component and a quadrature component. The in-phase component and the quadrature component are electrical signals. The optical coherent receiver includes a power adjuster in turn including a multiplying unit and a digital circuit. The multiplying unit is configured to multiply the in-phase component by an in-phase gain thereby providing a power-adjusted in-phase component, and to</td></tr><tr><td>Ground-Truth Abstract</td><td>It is disclosed an optical coherent receiver for an optical communication network. The optical coherent receiver is configured to receive a modulated optical signal and to process it for generating an in-phase component and a quadrature component. The optical coherent receiver comprises a power adjuster in turn comprising a multiplying unit and a retroactively connected digital circuit. The multiplying unit is configured to multiply the in-phase and quadrature components by in-phase and quadrature gains, respectively, thereby providing power-adjusted in-phase and quadrature components. The digital circuit is configured to compute: a common gain indicative of a sum of the powers of the power-adjusted in-phase and quadrature components; a differential gain indicative of a difference between the powers of the power-adjusted in-phase and quadrature components; and the in-phase and quadrature gains as a product and a ratio, respectively, between the common gain and the differential gain.</td></tr><tr><td>Claims</td><td>1. A method of assessing the arterial health of an individual comprising: a) positioning one arm of the patient such that the arm is at approximately the same height as the heart of the individual; b) attaching a fingertip photoplethysmography device to a fingertip of the hand of said arm of the individual; c) measuring the analog pulse contour of the individual using the fingertip photoplethysmography device; d) digitizing the analog pulse contour; e) analyzing the digitized analog pulse contour for stable waveforms; f) processing the stable waveforms of the digitized analog pulse contour using dynamic time warping; g) comparing the stable waveforms to a library of known disease state waveforms; and h) assigning a most probable disease state for the individual based on said comparison. 2. The method according to claim 1 wherein the stable waveforms in step (e) are identified by analyzing the digitized pulse contour for two consecutive pulse heights that are within 10% of the same value and the next five waveforms are measured and stored as the stable waveforms. 3. The method according to claim 1 wherein following step (e), the stiffness index is calculated. 4. The method according to claim 1 wherein following step (e), the reflection index is calculated. 5. A method of generating a library of waveforms from individuals having a known arterial disease comprising: a) positioning one arm of an individual having a known arterial disease such that the arm is at approximately the same height as the heart of the individual; b) attaching a fingertip photoplethysmography device to a ...</td></tr><tr><td>Generated Abstract</td><td>A method of assessing the arterial health of an individual includes positioning one arm of the patient such that the arm is at approximately the same height as the heart of the individual; attaching a fingertip photoplethysmography device to a fingertip of the hand of said arm of the individual; measuring the analog pulse contour of the individual using the fingertip photoplethysmography device; digitizing the analog pulse contour; analyzing the digitized analog pulse contour for stable waveforms; processing the stable waveforms of the digitized analog pulse contour using dynamic time warping;</td></tr><tr><td>Ground-Truth Abstract</td><td>The arterial health of an individual can be determined by: attaching a fingertip photopiehysmography device to a fingertip of the hand of the elevated arm of the individual; measuring the analog pulse contour of the individual using the fingertip photopiehysmography device; digitizing the analog pulse contour; analyzing the digitized pulse contour for stable waveforms; processing the stable waveforms of the digitized pulse contour using dynamic time warping; comparing the stable waveforms to a library of known disease state waveforms; and assigning a most probable disease state for the individual based on said comparison.</td></tr></table>

Table 10: Examples of claims summaries produced by our T5-Small model. Qualitatively, the models produce fluent patent abstracts complete with accurate details drawn from the claims section. Note also how different the structure of the above legal language is from most text used to train large language model; often, the entire abstract is a single long and complex sentence.

# G Experimental Details

# G.1 Binary Decision Classification

For this first task, we looked at the patent applications that were filed to the USPTO between January 2011 and December 2016 and excluded the pending applications from our experiments. Initially, we trained individual domain-specific classifiers, ranging from NB classifiers to RoBERTa, to predict the acceptability of patent applications in the most common IPC subclasses (see Figure 2). We used both the abstract and the claims, though separately.

![](images/8ac851be46101e449a1dc4bae0320a10d03befbb335ba686ef7de8b371b77b9a.jpg)

<details>
<summary>bar_stacked</summary>

| Years | ACCEPTED | REJECTED | PENDING | CONT-ACCEPTED | CONT-REJECTED | CONT-PENDING |
|-------|----------|----------|---------|---------------|---------------|--------------|
| 2011  | 150000   | 75000    | 10000   | 50000         | 20000         | 10000        |
| 2012  | 155000   | 80000    | 15000   | 60000         | 25000         | 15000        |
| 2013  | 145000   | 85000    | 45000   | 65000         | 30000         | 20000        |
| 2014  | 140000   | 85000    | 65000   | 65000         | 35000         | 25000        |
| 2015  | 135000   | 75000    | 75000   | 65000         | 35000         | 35000        |
| 2016  | 145000   | 65000    | 85000   | 65000         | 35000         | 45000        |
</details>

Figure 10: Distribution of decision status labels for patents filed between 2011 and 2016. Note that the relative share of pending to rejected applications increases over time as certain patents remain under review beyond the end of our metadata collection period. Approximately three quarters of patent applications are labeled as new filings.

To address the issue of imbalance in our decision status labels (see Figure 10), we used a weighted random sampler to select samples. We then randomly apportioned the data into training and test sets with an 85-15 split, but fixed the random seed across each model in each category to ensure fair comparability.

Our baselines for this task consisted of various subsets of Bernoulli and Multinomial naive Bayes classifiers, logistic regression (Logistic), CNN, DistilBERT [44], BERT [9], DistilROBERTa [44], RoBERTa [29], and T5-Small [40] for different tasks. $^{42}$

# G.2 Abstractive Summarization

We converted the summarization task into a language modeling task using the setup described by Raffel et al. [40]. The source text and summary were simply concatenated, separated by a separation phrase (“summarize:”). We used the T5-Small “Text-to-Text Transformer” architecture of Raffel et al. [40] with 60 million parameters. We initialized with a model pretrained on the C4 dataset (t5-small on HuggingFace Transformers), although we found that training from scratch produced nearly the same performance due to the size of our dataset.

We used the same training-validation subsets used in the language modeling task: Applications from 2011-2016 for training and those from 2017 for validation. We truncated the training source texts (i.e., the claims/description) to a maximum of 1024 tokens. We trained for three epochs using the Adam [21] optimizer with batch size of 32 and a learning rate of $5 \cdot 10^{-5}$ .

# H Discussion of Additional Tasks

# H.1 Long Sequence Modeling

One interesting application of HUPD that we have not yet explored is to use its patent description field as part of a benchmark for long-sequence modeling. Long-sequence modeling has recently gained significant attention due to the proliferation of efficient Transformer architectures (e.g., Longformer, Performer). Popular benchmarks for these models include synthetic tasks, byte-level text tasks, and computer vision tasks; there does not exist a large-scale long-sequence word-level NLP benchmark since it is difficult to collect a large corpus of narrowly-focused long text documents. The task of language modeling of patent descriptions might be well-suited for this purpose; HUPD contains over 4.5M descriptions with an average length of 11,855 tokens (see Table 2), which far exceeds the context length of traditional Transformer models.

# H.2 Patent Clustering

A second application of the patent data is patent clustering: Given a patent application, one is tasked with finding similar patents (or patent applications). This tasks is particularly interesting to the IP community, since patent lawyers and examiners are interested in finding prior art for a given invention. To define clusters and find related patents, it would be possible to use a combination of metadata fields including fine-grained IPC/CPC classification codes, publication date, and inventor information.

# H.3 Patent Examiner Assignment

A third application of the patent data makes use of the examiner field, which has not been used in the tasks discussed above. This task involves predicting the examiner to which a new patent application should be assigned. This task may be viewed as an even more-fine-grained version of patent classification, because patent examiners often focus on a very narrow sub-field in which they have expertise or knowledge. We note that this task is possible under our framework because HUPD contains both raw texts of patent applications and also rich bibliographic metadata about each patent application.

# H.4 Potential Social Impact and Applications & Future Work

In recent years, the USPTO has introduced a pilot program, called “Pro Se Assistance Program”, to help small businesses and independent inventors file patent applications without enlisting the aid of a registered patent attorney or a legal agent. This initiative aims to enhance the quality of applications without putting any further financial burden on the shoulders of patent applicants who might have limited resources and means, as well as educating inventors about intellectual property protection and the patent filing process $[41]$ . We hope our patent dataset and models might also provide some assistance to independent inventors and small businesses in improving the overall textual quality of their patent applications, to classify the technology areas of their inventions accurately, and to generate the abstract sections of their applications from their patent claims and/or description.

Given the scope of this paper, we could not conduct experiments on patent clustering, patent value prediction, or identification of “superstar” inventions. However, we foresee these research directions as direct valuable and impactful applications of our dataset, and suggest the research community push the boundaries on these fronts. In addition, our dataset’s combination of natural language with structured metadata makes it an ideal gymnasium for evaluating and developing models that address concept shift across contexts and over time.

# I Patent Category Visualizations

![](images/419fe3da1a2b3bbaac4036eba9052582e5804708071f92dd75a9c53c7560aa47.jpg)

<details>
<summary>scatter</summary>

| Model | Abstract | Summary | Title |
| --- | --- | --- | --- |
| Finetuned Distil-Roberta | -0.2 to 0.3 | -0.1 to 0.3 | -0.1 to 0.3 |
| Finetuned T5-Small | -0.4 to 0.4 | -0.4 to 0.4 | -0.4 to 0.4 |
| Roberta-Base | -0.2 to 0.3 | -0.1 to 0.3 | -0.1 to 0.3 |
| Mini-LM v2 | -0.4 to 0.6 | -0.4 to 0.6 | -0.4 to 0.6 |
</details>

Figure 11: Visualizations of the vector representations of the data fields (abstract, summary, and title) of patent applications, embedded using four different Transformer models. Of these four models, two were fine-tuned on HUPD (Fine-tuned DistilRoBERTa and Fine-tuned T5-Small) and two were off-the-shelf, pre-trained models (RoBERTa-Base and Mini-LM v2). The Mini-LM v2 model was trained on a paraphrase dataset to produce good sentence-embeddings for text clustering. To create the figures above, we randomly sampled 500 patent applications from each of the top nine IPC categories for each year from 2008 to 2018. For each data field and each model, we computed embedding vectors using the model and reduced the dimensionality of these vectors using PCA. The above plots show the results of this dimensionality reduction, colored by IPC codes. We see that categories cluster strongly and similar categories (e.g., H04W-Wireless Communication and H04L-Transmission of Digital Information) are often close in embedding space.

![](images/d81cefc7840382b1193bbec247e34de3521588f26da6c62f774ef2a87ac5ac03.jpg)  
Figure 12: Visualizations of the vector representations of the data fields (abstract, summary, and title) of patent applications, embedded using four different Transformer models. This figure is similar to Figure 11, but uses UMAP [32] for dimensionality reduction rather than PCA.

![](images/3481acecb845c0e2568c6b9e282c6b8176e14c270a49217aee73894a90415cc0.jpg)

<details>
<summary>scatter</summary>

| Year | Category                                      | Value |
|------|-----------------------------------------------|-------|
| 2008 | G06Q - Data Processing Systems or Methods    | 0.5   |
| 2010 | G06F - Electric Digital Data Processing      | 1.8   |
| 2012 | H04N - Pictorial Communication, e.g. Television | 3.5   |
| 2014 | H04L - Transmission of Digital Information     | 0.2   |
| 2016 | G06Q - Data Processing Systems or Methods    | 0.2   |
| 2018 | G06Q - Data Processing Systems or Methods    | 0.6   |
| 2012 | H04W - Wireless Communication Networks       | 1.9   |
| 2014 | H04W - Wireless Communication Networks       | 3.5   |
| 2016 | A61B - Diagnosis, Surgery, Identification     | 4.8   |
| 2018 | A61K - Preparations for Medical Purposes     | 3.9   |
| 2012 | H01L - Semiconductor Devices                  | 7.5   |
| 2014 | H04L - Transmission of Digital Information     | 1.5   |
| 2016 | A61B - Diagnosis, Surgery, Identification     | 4.8   |
| 2018 | A61K - Preparations for Medical Purposes     | 4.8   |
</details>

Figure 13: Depiction of the evolution over time of the averaged embeddings of the abstracts of patent applications from G06Q-Data Processing Systems or Methods relative to the other popular IPC codes in our dataset. To create this figure, as in the figures above, we randomly sampled 500 patent applications from each of the top nine IPC categories for each year from 2008 to 2018. Using our custom DistilRoBERTa model, we computed embedding vectors for the abstracts of each patent application and then reduced the dimensionality of these vectors using UMAP. The figure above shows the average UMAP embedding of the G06Q category for every other year, along with the average embeddings of the other categories (averaged across all samples from all years). The movement of the centroids of the G06Q category might be consistent with covariate shift over time in the distribution of language of patents in the category. G06F-Electric Digital Data Processing seems to move closer to H04L-Transmission of Digital Information/H04W-Wireless Communication. This evolution seems to be consistent with the digitization of data processing methods over the past two decades.