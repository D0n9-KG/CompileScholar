A Survey on Natural Language Processing for Programming 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.05773v2 [cs.CL] 06 Aug 2023 
 
 

# A Survey on Natural Language Processing for Programming

 
 
 Qingfu Zhu ♯ 
 
 Email:  qfzhu@ir.hit.edu.cn 
 
    
 Xianzhen Luo ♯ 
 
 Email:  xzluo@ir.hit.edu.cn 
 
    
 Fang Liu
 
 Affiliation:  Beihang University, Beijing, China
 
 Email:  car@ir.hit.edu.cn 
 
    
 Cuiyun Gao
 
 Affiliation:  Harbin Institute of Technology, ShenZhen, China
 
 Email:  fangliu@buaa.edu.cn 
 
    
 Wanxiang Che ♯ ♯ Harbin Institute of Technology, Harbin, China
 † † thanks: Corresponding author. 
 Email:  gaocuiyun@hit.edu.cn 
 

 Abstract 
 
 Natural language processing for programming aims to use NLP techniques to assist programming.
It is increasingly prevalent for its effectiveness in improving productivity.
Distinct from natural language, a programming language is highly structured and functional.
Constructing a structure-based representation and a functionality-oriented algorithm is at the heart of program understanding and generation.
In this paper, we conduct a systematic review covering tasks, datasets, evaluation methods, techniques, and models from the perspective of the structure-based and functionality-oriented property, aiming to understand the role of the two properties in each component.
Based on the analysis, we illustrate
unexplored areas and suggest potential directions for future work.

 
 
 
 
 

 
 

## 1 Introduction

 
 Natural language processing for programming (NLP4P) is an interdisciplinary field of NLP and software engineering (SE), aiming to use NLP techniques for assisting programming Lachmy et al. (2021) .
It could relieve developers from laborious work, e.g., by automatically writing a document for a program.
Meanwhile, it provides easy access for non-professional users to improve efficiency, e.g., by performing a cross-application operation with natural language (NL) interface  Liu et al. (2016a) .
Therefore, it is beneficial for improving the productivity of the whole society.

 
 
 Figure 1: 
An example of the structure-based and functionality-oriented property of programming language.
Colored fonts and indents denote different aspects of the structure-based property.
 
 
 
 Distinct from NL, a programming language (PL) is characterized by two properties: structure-based and functionality-oriented , as shown in Figure  1 .
First, PL is highly structure-based since it intrinsically contains multiple sophisticated structures, such as hierarchy, loops, and recursions.
Appropriately modeling the components and obtaining a structure-based representation is the key to program understanding  Mou et al. (2016) ; Allamanis et al. (2018) ; Hu et al. (2018) ; Guo et al. (2020) ; Wang et al. (2021) ; Guo et al. (2022) .
Second, PL is functionality-oriented since it is executable and ought to convert given input into expected output.
Developing an algorithm oriented to the functionality is at the heart of generating a logically correct program  Chen et al. (2021) ; Hendrycks et al. (2021) ; Li et al. (2022) ; Nijkamp et al. (2022) ; Le et al. (2022) .
Despite the benefits, the two properties cannot be directly modeled by conventional NLP approaches due to the heterogeneity between NL and PL, making the integration of the properties a fundamental topic in NLP4P.

 
 
 From the perspective of pre-training,
 Niu et al. (2022) has summarized the recent advance.
Nevertheless, the role of the structure-based and functionality-oriented property has not been sufficiently discussed.
In this paper, we focus on the properties and systematically review their effect in defining tasks (§ 2 ), constructing datasets (§ 3 ), forming evaluation methods (§ 4 ), supporting techniques (§ 5 ), and achieving SOTA performance (§ 6 ).
Based on the analysis, we further illustrate unexplored areas of current NLP4P and potential directions for future work (§ 7 ).
The contributions of this paper are summarized as follows:

 
 • 
 
 We identify two properties of PL: structure-based and functionality-oriented, which are essential for program understanding and generation, respectively.

 

 • 
 
 From the perspective of the properties, we systematically review current work, covering tasks, datasets, evaluation methods, techniques, and representative models that achieve SOTA performance.

 

 • 
 
 By analysis of current NLP4P, we illustrate unexplored areas and suggest potential directions for future work.

 

 
 
 
 Figure 2: 
Categories of NLP4P tasks. Orange and red fonts denote the structure-based and functionality-oriented tasks, respectively.
 
 
 
 

## 2 Tasks

 
 As shown in Figure 2 , we classify a task as functionality-oriented if it aims at program generation; otherwise as structure-based.
Within each category, we further divide the tasks according to application scenarios to cluster related tasks and highlight subtle differences between them.
The partition of the structure-based and functionality-oriented roughly aligns with the partition of understanding and generation in NLP.
An exception is the summarization task, which is abstractive and regarded as a generation task in NLP.
We classify it as structure-based since PL lies in its input side, and the key to the task is understanding the content of PL by the structure.

 
 

### 2.1 Summarization Tasks

 
 The summarization task summarizes a program into an NL description.
It is crucial for the maintenance of software, especially those involving multiple developers.
According to the format of the output, it can be further divided into comment generation Nie et al. (2022) and docstring generation Clement et al. (2020) .
The output of the latter contains some structural information, such as parameters and input/output examples.

 
 
 

### 2.2 Retrieval Tasks

 
 The retrieval task mainly refers to the code search .
It aims to retrieve relevant programs given NL query  Husain et al. (2019) .
It has a similar application scenario and input/output format to program synthesis. The difference is that its output is extracted from existing programs, rather than being synthesized from scratch.

 
 
 

### 2.3 Classification Tasks

 
 The classification task detects whether given programs have specific characteristics,
e.g., being cloned ( clone detection ), or being vulnerable ( vulnerability identification ).
They are essential in protecting software from the effects of ad-hoc reuse  Svajlenko et al. (2014) and cyber attacks  Zhou et al. (2019) .
The granularity of the input ranges from a coarse-grained software
repository  Hovsepyan et al. (2012) to a fine-grained function  Russell et al. (2018) ; Zhou et al. (2019) .

 
 
 Despite the fact that NL does not explicitly occur in either input or output, we include tasks of such form for two reasons.
First, PL has been demonstrated to contain abundant statistical properties similar to NL  Mou et al. (2016) .
Second, most of the ways that PL is processed are derived from NLP, like machine translation techniques  Tufano et al. (2019) in the transcription task (§  2.5 ).

 
 
 

### 2.4 Synthesis Tasks

 
 The synthesis task generates a program given a context (which can be NL, PL, or their mixture), thus can accelerate the development process.
It can be further divided into program synthesis and code completion by the formal completeness of the output.
The output of program synthesis is a relatively independent unit, such as a function and a class, while the output of code completion is less restricted, ranging from tokens to code snippets.

 
 

#### Program synthesis

 
 is also called code generation.
It is the systematic derivation of a program from a given specification Manna and Waldinger (1980) .
Conventional deductive approaches  Manna and Waldinger (1980) ; Polozov and Gulwani (2015) take logical specifications, which are logically complete but hard to write.
Inductive approaches  Lieberman (2001) list input-output examples as specifications, which are more accessible but incomplete.
In contrast, an NL specification is sufficient to describe the logic of a program.
Meanwhile, it is compatible with input-output examples by including them in a docstring.
Therefore, it can take advantage of both the deductive and inductive approaches.

 
 
 

#### Code completion

 
 is also called code suggestion in early research  Tu et al. (2014) ; Hindle et al. (2016) . It suggests the next program token given a context and has been widely applied to IDEs  Li et al. (2018) .
The application scenario includes the completion of method calls, keywords, variables, and arguments.
With the bloom of the pre-trained models, the scenario has been extended to punctuations, statements, and even code snippets  Svyatkovskiy et al. (2020) , further blurring the line between program synthesis and code completion.

 
 
 
 

### 2.5 Transcription Tasks

 
 The transcription task converts a given program to meet a specific requirement.
Concretely, program translation aims to convert between high-level PL  Roziere et al. (2020) ; Zhu et al. (2022) , e.g., C# and Java.
It can accelerate the update of projects written by deprecated PL, and the migration of algorithms implemented by various PLs.
 Code refinement aims to convert a buggy program into correct one  Wang et al. (2021) .
It is closely related to vulnerability identification but is required to fix the detected bugs simultaneously.
The transcription task differs from the synthesis task in two aspects.
First, its input program is formally complete (input program is None or a function header in program synthesis, a partial code snippet in code completion).
Second, its output can be strictly aligned with the input in both the format and the content.

 
 
 
 
 | 
 Abbr. | 
 Data | 
 Source | 
 PL | 
 Size | 
 Type | 

 
 
 
 General Dataset 
 | 
 pc | 
 Christopoulou et al.,2022 | 
 GitHub | 
 Python | 
 147 | 
 NL-PL | 

 
 PL | 

 
 tp | 
 THEPILE  ( 2020 ) | 
 GitHub, ArXiv,… | 
 - | 
 825 | 
 NL | 

 
 PL | 

 
 pb | 
 Ahmad et al.,2021 | 
 GitHub, | 
 Java, | 
 655 | 
 NL | 

 
 Stack Overflow | 
 Python | 
 PL | 

 
 ic | 
 Fried et al.,2022 | 
 GitHub, GitLab, | 
 28 | 
 216 | 
 NL | 

 
 Stack Overflow | 
 PL | 

 
 ts | 
 thestack | 
 GitHub | 
 30 | 
 3,100 | 
 PL | 

 
 ac | 
 Li et al.,2022 | 
 GitHub | 
 12 | 
 715 | 
 PL | 

 
 bq | 
 BigQuery | 
 GitHub | 
 C/C++, Go, Java, | 
 340 | 
 PL | 

 
 JS, Python | 

 
 bp | 
 BIGPYTHON  ( 2022 ) | 
 GitHub | 
 Python | 
 217 | 
 PL | 

 
 cp | 
 CodeParrot | 
 GitHub | 
 Python | 
 180 | 
 PL | 

 
 cx | 
 Chen et al.,2021 | 
 - | 
 Python | 
 159 | 
 PL | 

 
 gp | 
 GCPY  ( 2022 ) | 
 GitHub | 
 Python | 
 - | 
 PL | 

 
 
 
 Struc-based 
 | 
 
 
 CG 
 | 
 cn | 
 CodeNN  ( 2016 ) | 
 Stack Overflow | 
 C#, SQL | 
 1 | 
 NL-PL | 

 
 
 
 CS 
 | 
 csn | 
 CodeSearchNet  ( 2019 ) | 
 GitHub | 
 Go, Java, JS, PHP, | 
 17 | 
 NL-PL | 

 
 Python, Ruby | 
 PL | 

 
 
 
 CD 
 | 
 bc | 
 BigCloneBench  ( 2014 ) | 
 SeCold | 
 Java | 
 2 | 
 PL | 

 
 poj | 
 POJ-104  ( 2016 ) | 
 - | 
 C/C++ | 
 1 | 
 PL | 

 
 
 
 VI 
 | 
 dv | 
 Devign  ( 2019 ) | 
 QEMU, FFmpeg | 
 C | 
 1 | 
 PL | 

 
 
 
 Functionality-oriented 
 | 
 
 
 PS 
 | 
 cd | 
 CONCODE  ( 2018 ) | 
 GitHub | 
 Java | 
 13 | 
 NL-PL | 

 
 cn | 
 CodeNet  ( 2021 ) | 
 AIZU, AtCoder | 
 55 | 
 8 | 
 NL-PL | 

 
 cc | 
 CodeContests  ( 2022 ) | 
 CodeNet, Codeforces, | 
 C/C++, Java, | 
 3 | 
 NL-PL | 

 
   Caballero et al.,2016 | 
 Python | 

 
 ap | 
 APPS  ( 2021 ) | 
 
 
 
 Codewars, AtCoder, 
 
 Kattis, Codeforces 
 | 
 Python | 
 1 | 
 NL-PL | 

 
 he | 
 HumanEval  ( 2021 ) | 
 Hand-Craft | 
 Python | 
 1 | 
 NL-PL | 

 
 mb | 
 MBPP  ( 2021 ) | 
 Hand-Craft | 
 Python | 
 1 | 
 NL-PL | 

 
 
 
 CC 
 | 
 py | 
 PY150  ( 2016 ) | 
 GitHub | 
 Python | 
 1 | 
 PL | 

 
 gjc | 
 Github Java Corpus  ( 2013 ) | 
 GitHub | 
 Java | 
 1 | 
 PL | 

 
 
 
 PT 
 | 
 ct | 
 CodeTrans | 
 Lucene, POI, JGit, Antlr | 
 Java, C# | 
 1 | 
 PL | 

 
 
 
 CR 
 | 
 bf | 
 Bugs2Fix  ( 2019 ) | 
 GitHub | 
 Java | 
 15 | 
 PL | 

 Table 1: 
An overview of datasets.
Struc-based denotes the structure-based.
Abbreviations (Abbr.) in upper case and lower case denote tasks (Figure  2 ) and datasets, respectively.
For datasets that contain numerous kinds of PL, the total number of PLs is reported instead of concrete PL types.
The unit of data size is GB.
NL-PL in the last column denotes a parallel dataset whose all samples contain paired NL and PL.
Detached NL or PL denotes a monolingual dataset whose dominant language is NL or PL.
 
 
 
 
 

## 3 Datasets

 
 Datasets are the basis for supporting the learning process of tasks.
We thus classify current datasets into general, structure-based, and functionality-oriented, following the categories of tasks (as shown in Table  1 ).
The general dataset is slightly processed and can be used in the early learning stage of a model regardless of tasks.
The structure-based and functionality-oriented datasets are dedicated data specifically formatted for each task.

 
 

### 3.1 General vs. Dedicated

 
 There are two primary sources of general datasets:
1) open-source platforms such as GitHub, GitLab, and SeCold,
2) community-based spaces like Stack Overflow.
The datasets are automatically collected and large in scale, thus can be applied to pre-training to ensure generated PL is grammatically correct and logically valid.
However, sometimes they are noisy and non-informative.
For instance, a commit message like “update” is of little substantial content; a code snippet answer might be irrelevant to its question  Iyer et al. (2018) .

 
 
 Structure-based datasets are specially formatted to support particular tasks.
Concrete structure information is available via open-source parsers, e.g., Tree-sitter. 1 1 
 1 
 
 
 
 https://tree-sitter.github.io/tree-sitter/ 
Most functionality-oriented datasets contain a number of test cases for each sample to verify the functional correctness of synthesized programs.
Therefore, the datasets are typically hand-crafted  Chen et al. (2021) ; Austin et al. (2021) or collected from online judge websites  Iyer et al. (2018) ; Puri et al. (2021) ; Hendrycks et al. (2021) ; Li et al. (2022) , including AIZU, AtCoder, Codeforces, Codewars, and Kattis.

 
 
 

### 3.2 Parallel vs. Monolingual

 
 To further explore the potential of datasets outside their original tasks, we divide them into parallel (denoted as NL-PL) and monolingual (denoted as NL or PL).
We define a dataset as parallel if all samples include paired NL and PL, otherwise as monolingual.
The type of monolingual datasets is denoted by their dominant language.
For instance, Stack Overflow QA pairs with optional code snippets are denoted as NL, and GitHub programs with optional comments are denoted as PL.
Note that a dataset may consist of multiple subsets of different types, we explicitly list them in the last column of Table  1 .
Generally, parallel datasets are relatively homogeneous, and thus can support other tasks whose dataset is of the same type.
For example, CodeSearchNet can also be used for comment generation  Lu et al. (2021) .
In contrast, a monolingual dataset is usually task-related with specific labels, making it less transferable to other tasks.

 
 
 
 

## 4 Evaluation Methods

 
 The evaluation metric is also closely related to the task.
Considering that the retrieval and classification tasks are well-defined and their metrics (such as F1, MRR, and accuracy) are universally accepted, we focus on the summarization, synthesis, and transcription tasks, whose evaluation remains an open question.
Concretely, the output of summarization is NL. Thus the evaluation can directly refer to NLP.
For the synthesis and transcription task, whose output is PL, the functionality-oriented property is the main concern, assisted by the structure-based property as an auxiliary.

 
 

### 4.1 NL Evaluation

 
 NL evaluation can refer to NLP and be conducted by the following two complementary approaches.

 
 

#### Automatic Evaluation

 
 is usually implemented by comparing the n-grams between the predicted output and given references.
Concrete metric includes BLEU  Papineni et al. (2002) , MENTOR  Banerjee and Lavie (2005) , and ROUGE  Lin (2004) .
However, limited by the number of references, they might correlate weakly with the real quality   Liu et al. (2016b) .
Hence, it is crucial to conduct a human evaluation simultaneously.

 
 
 

#### Human Evaluation

 
 consists of several independent dimensions, such as naturalness, diversity, and informativeness.
Common annotation methods include point-wise mode  Iyer et al. (2016) ; Shi et al. (2021a) and
pair-wise mode  Panthaplackel et al. (2020) .
Human evaluation is more accurate, fine-grained, and comprehensive than automatic evaluation.
However, it is also time-consuming and labor-intensive, and thus can only be conducted on a small subset of the test set.

 
 
 
 

### 4.2 PL Evaluation

 
 PL evaluation can be conducted by the following two methods using the references and the test cases as evidence, respectively.

 
 

#### Reference based Evaluation

 
 Regarding a program as a sequence of tokens,
PL can also be evaluated by n-gram based NL metrics, such as BLEU  Wang et al. (2021) and exact match (EM,  Guo et al., 2022 ).
To further capture the structure-based property, Ren et al. (2020) propose the CodeBLEU metric, which takes AST and data flow graph into consideration.
Similar to NL, PL is expressive in that a program can be implemented differently, leading to the same weak correlation issue with a limited number of references.

 
 
 

#### Test Case based Evaluation

 
 Hendrycks et al. (2021) propose two metrics based on test cases: Test Case Average and Strict Accuracy.
Suppose there is a single generated program and a varying number of test cases for each sample.
Test Case Average computes the average test case pass rate over all samples.
Strict Accuracy is a relatively rigorous metric.
A program is regarded as accepted if and only if it passes all test cases, and the final Strict Accuracy is the ratio of accepted programs.

 
 
 Actually, we can generate more than one (e.g., K K ) program for each sample to improve the performance.
In this way, Strict Accuracy regards a sample as accepted if any of the K K programs pass all test cases.
Therefore, it is also called p ​ @ ​ k p@k in some literature.
The sampling size could be huge, but the number of submissions sometimes is limited, like the competition scenario.
To highlight the difference between the sampling and submission, Li et al. (2022) further propose the n ​ @ ​ k n@k metric, which computes the acceptance ratio when sampling k k and submitting n n programs.

 
 
 The test case based evaluation is a remarkable progress, which has already in turn improve the training process via reinforcement learning  Le et al. (2022) .
Currently, the associated datasets are only available in program synthesis.
Extending it to other functionality-oriented tasks is expected to gain similar improvement.

 
 
 
 
 

## 5 Techniques

 
 The heterogeneity between NL and PL requires extra effort in techniques to process programs.
First, the key to understanding the content of a program is appropriately representing its structure information.
Second, at the heart of program generation is elaborately designing an algorithm to achieve functional correctness.
Therefore, we introduce the techniques by structure-based understanding and functionality-oriented generation, respectively.

 
 

### 5.1 Structure-based Understanding

 
 Compared with NL, PL has more sophisticated structures, such as hierarchy, loops, and recursions.
Generally, it would benefit the performance by explicitly representing the structures with appropriate data structure, including relative distance, abstract syntax tree, control flow graph, program dependence graph, and code property graph.

 
 

#### Relative Distance

 
 typically refers to the distance between two tokens in the source code sequence.
In this way, it can be easily combined into token representations as a feature.
 Ahmad et al. (2020) represent the relative distance as a learnable embedding and introduce it into transformer models by biasing the attention mechanism.
Results show that the relative distance is an effective alternative to AST to capture the structure information.
Based on that, Zugner et al. (2021) further extend the concept of relative distance from textual context to AST.
Jointly training a model with the two types of relative distance achieves further improvement.

 
 
 

#### Abstract Syntax Tree (AST)

 
 is a tree representation that carries the syntax and structure information of a program  Shi et al. (2021b) .
It simplifies inessential parts (e.g., parentheses) of the parse tree by implying the information in its hierarchy.
Each node of AST has arbitrary number of children organized in a specific order .
Therefore, a lossless representation of AST should capture the two characteristics simultaneously.
Despite that, some AST can be complex with a deep hierarchy  Guo et al. (2020) , which delays the parsing time and increases the input length (up to 70%)  Guo et al. (2022) .

 
 
 

#### Control Flow Graph (CFG)

 
 represents a program as a graph.
Its node (also called a basic block) contains a sequence of successive statements executed together.
Edges between nodes are directed, denoting the order of execution  Allen (1970) .
CFG makes it convenient to locate specific syntactic structures (such as loops and conditional statements) and redundant statements.

 
 
 

#### Program Dependence Graph (PDG)

 
 is another graphical representation of a program.
Nodes in PDG are statements and predicate expressions, and edges denote both data dependencies and control dependencies  Ferrante et al. (1987) .
The data dependencies describe the partial order between definitions and usages of variables, and have been demonstrated to be beneficial for program understanding  Krinke (2001) ; Allamanis and Brockschmidt (2017) ; Allamanis et al. (2018) ; Guo et al. (2020) .
Similar to CFG, control dependencies also model the execution order, but it highlights a statement or a predicate itself by determining edges according to its value  Liu et al. (2020a) .

 
 
 

#### Code Property Graph (CPG)

 
 is a joint graph that merges AST, CFG, and PDG  Yamaguchi et al. (2014) .
In this way, it takes advantage of all the representations, and thus can comprehensively represent a program for structure-based tasks, such as vulnerability identification  Zhou et al. (2019) and comment summarization  Liu et al. (2020a) .

 
 
 In summary, a structure-based representation benefits program understanding.
Among the representations, relative distance takes the most concise form but has the minimum structure information, while CPG is the other extreme.
AST, CFG, and PDG are a balance between conciseness and information capacity.
As a tree representation, AST can be more easily integrated by a backbone model than the graphical CFG and PDG, and thus is the most widely used structure-based representation.

 
 
 
 

### 5.2 Functionality-oriented Generation

 
 Distinct from NL, PL is executable and ought to convert an input into expected output to implement specific functionality.
To achieve this,
it has developed a sampling-based paradigm, which first samples a large volume of candidates and subsequently selects the desired program by test case based filtering and clustering techniques  Chen et al. (2021) ; Li et al. (2022) ; Nijkamp et al. (2022) .

 
 

#### Sampling

 
 To ensure good coverage of the desired program,
first, the number of sampling should be as large as possible (up to 1M per problem in Li et al., 2022 ).
Second, it would be better to employ the standard sampling with temperature or the top-k sampling algorithm, rather than the beam search, whose generated candidates can be pretty similar to each other  Li et al. (2016) .

 
 
 

#### Filtering and Clustering

 
 The resulting programs are subsequently filtered by checking the functional correctness on given test cases  Li et al. (2022) .
However, the number of programs after filtering can still be huge if there are too many programs sampled.
To fit the scenario where the number of submissions is limited, Li et al. (2022) propose a clustering strategy.
It first clusters the programs according to their behaviors on generated test cases.
Then it selects and submits a program from the clusters one by one.
This strategy avoids repetitively submitting programs with identical bugs.
Nevertheless, it does not consider the error message feedback after each submission, which might be an interesting direction for future work.

 
 
 
 

### 5.3 Backbone Models

 
 Most of the functionality-oriented algorithms are model-agnostic and have little impact on the choice of a backbone model.
In this section, we focus on the match between the structure-based representation (e.g., AST) and backbone models.

 
 

#### Recurrent Neural Network

 
 (RNN, Mikolov et al., 2010 ) and its variant LSTM  Hochreiter and Schmidhuber (1997) 
are capable of processing variable-length inputs.
Therefore, it is well-suited for representing NL description  Liu et al. (2016a) ; Weigelt et al. (2020) and PL token sequence  Wei et al. (2019) .
Meanwhile, it accepts the structure-based representation formatted as a sequence, e.g., the pre-order traversal of AST.
However, such transforms are lossy in that the AST cannot be recovered.
To this end, Hu et al. (2018) propose a structure-based traversal (SBT) approach, adding parentheses into the sequence to mark hierarchical relationships.
Distinct from SBT that adapts data to a model,
 Shido et al. (2019) propose a Multi-way Tree-LSTM, which directly takes input as AST.
It first encodes the children of a node with a standard LSTM, and subsequently integrates the results into the node with a Tree-LSTM.

 
 
 
 
 Model | 
 Size | 
 CG/csn | 
 CS/csn | 
 CD/bc | 
 VI/dv | 
 PS/cd | 
 CC/py | 
 PT/ct | 
 CR/bf | 

 
 BLEU | 
 MRR | 
 F1 | 
 ACC | 
 BLEU | 
 ES | 
 BLEU | 
 BLEU | 

 
 CodeBERT  ( 2020 ) | 
 125M | 
 17.83 | 
 69.3 | 
 94.1 | 
 62.08 | 
 - | 
 - | 
 79.92 | 
 91.07 | 

 
 PLBART  ( 2021 ) | 
 140M | 
 18.32 | 
 68.5 | 
 93.6 | 
 63.18 | 
 36.69 | 
 68.46 | 
 83.02 | 
 88.50 | 

 
 GraphCodeBERT  ( 2020 ) | 
 125M | 
 - | 
 71.3 | 
 95.0 | 
 - | 
 - | 
 - | 
 80.58 | 
 91.31 | 

 
 UniXcoder  ( 2022 ) | 
 126M | 
 19.30 | 
 74.4 | 
 95.2 | 
 - | 
 38.23 | 
 72.00 | 
 - | 
 - | 

 
 CodeT5  ( 2021 ) | 
 220M | 
 19.55 | 
 71.5 | 
 95.0 | 
 65.78 | 
 40.73 | 
 67.12 | 
 84.03 | 
 87.64 | 

 Table 2: 
The results of structure-based models on CodeXGLUE. Abbreviations in upper case and lower case separated by “/”denote tasks (Figure  2 ) and datasets (Table  1 ), respectively. ES denotes Levenshtein edit similarity.
 
 
 
 
 
 Model | 
 Size | 
 p@1 | 
 p@100 | 

 
 CodeX  ( 2021 ) | 
 300M | 
 13.17 | 
 36.27 | 

 
 2.5B | 
 21.36 | 
 59.50 | 

 
 12B | 
 28.81 | 
 72.31 | 

 
 CodeGen  ( 2022 ) | 
 350M | 
 12.76 | 
 35.19 | 

 
 2.7B | 
 23.70 | 
 57.01 | 

 
 6.1B | 
 26.13 | 
 65.82 | 

 
 16.1B | 
 29.28 | 
 75.00 | 

 
 
 
 AlphaCode  ( 2022 ) 
 | 
 302M | 
 11.60 | 
 31.80 | 

 
 1.1B | 
 17.10 | 
 45.30 | 

 
 
 
 PANGU- 
 CODER  ( 2022 ) 
 | 
 317 M | 
 17.07 | 
 34.55 | 

 
 2.6B | 
 23.78 | 
 51.24 | 

 
 INCODER  ( 2022 ) | 
 6.7B | 
 15.20 | 
 47.00 | 

 
 CodeRL  ( 2022 ) | 
 770M | 
 14.05 | 
 46.06 | 

 Table 3: 
The results of functionality-oriented models on HumanEval benchmark.
p@k is the pass rate when sampling k k candidate programs. The results of CodeRL are computed using the officially released checkpoint.
 
 
 
 

#### Convolutional Neural Network

 
 (CNN, LeCun et al., 1989 ) extracts the features by scanning an input with a sliding window and applying stacked convolution and pooling operations on the window.
Both two operations can be parallelized, making CNN more time-efficient than RNN.
CNN in NLP4P usually takes input as execution traces  Gupta et al. (2020) , input-output pairs  Bunel et al. (2018) , and encodes them into an embedding as the output.
Similar to Tree-LSTM, CNN can also be adapted to the structure-based representation.
For instance,
 Mou et al. (2016) propose a tree-based convolutional neural network (TBCNN), which encodes AST by a weight-base and positional features.

 
 
 

#### Transformer

 
 Vaswani et al. (2017) has a similar interface to RNN. The difference lies in the following two aspects. First, it is more time-efficient by solely depending on the attention mechanism, rather than the recurrent unit.
Second, it can better capture long-term dependencies, which is essential for processing PL since programs can be pretty long  Ahmad et al. (2020) .

 
 
 Despite these approaches, some studies explore the usage of feed-forward neural network  Iyer et al. (2016) ; Loyola et al. (2017) , recursive neural network  Liang and Zhu (2018) , and graph neural network  Liu et al. (2020a) .
The architecture of the models, as well as RNN and CNN, can be flexibly adapted to customized data, e.g., the primitive AST.
While for large-scale general data, the transformer is suggested due to its high capacity and easy access to pre-training.

 
 
 
 
 Model | 
 Arch. | 
 Data | 
 Developer | 

 
 CodeBERT | 
 Enc | 
 csn | 
 Microsoft | 

 
 PLBART | 
 Enc-Dec | 
 pb | 
 UCLA | 

 
 GraphCodeBERT | 
 Enc | 
 csn | 
 Microsoft | 

 
 UniXcoder | 
 Enc-Dec | 
 csn | 
 Microsoft | 

 
 CodeT5 | 
 Enc-Dec | 
 csn, bq | 
 Salesforce | 

 
 CodeX | 
 Dec | 
 cx | 
 OpenAI | 

 
 AlphaCode | 
 Dec | 
 ac, cc, ap | 
 DeepMind | 

 
 PANGU-CODER | 
 Dec | 
 pc | 
 Huawei | 

 
 INCODER | 
 Dec | 
 ic | 
 Facebook | 

 
 CodeGen | 
 Enc-Dec | 
 tp, bp, bq | 
 Salesforce | 

 
 CodeRL | 
 Enc-Dec | 
 gp, ap | 
 Salesforce | 

 Table 4: 
The architecture (Arch.), training dataset, and developer of representative pre-training models.
Enc, Dec, and Enc-Dec denote the encoder-only, decoder-only, and encoder-decoder architecture, respectively.
 
 
 
 
 
 

## 6 Representative Pre-training Models

 
 SOTA pre-training models can be roughly divided into two categories according to the benchmarks they are evaluated.
The first category focuses on the CodeXGLUE benchmark  Lu et al. (2021) , which is composed primarily of structure-based datasets, as shown in Table  2 .
The second aims at passing the test cases of functionality-oriented datasets, as typified by HumanEval  Chen et al. (2021) in Table  3 .
We denote the two categories as structure-based models and functionality-oriented models.
Table  4 shows the architecture, training dataset, and developer of the models.

 
 

### 6.1 Structure-based Models

 
 The performance of this category is shown in Table  2 .
GraphCodeBERT, UniXcoder, and CodeT5 incorporate data dependencies of PDG, AST, and node types of AST, respectively.
As a reference, we also report the result of an encoder-only based BERT and an encoder-decoder based PLBART, neither of which utilize the structure representation.

 
 
 Generally, incorporating structure-based representation can boost the performance of NLP4P tasks.
Concretely, UniXcoder performs better on understanding tasks, while CodeT5 outperforms others on generation tasks.
For program synthesis, although CodeT5 has the highest BLEU score, it would be better to use its variant CodeRL or other functionality-oriented models.
In our experiments, the functional correctness measured by p@k of the former is not as good as that of the latter.

 
 
 

### 6.2 Functionality-oriented Models

 
 Table  3 shows the p@k results of functionality-oriented models on HumanEval.
The performance is primarily supported by the large scale sampling  Chen et al. (2021) and test case based filtering  Li et al. (2022) in the inference process.
Based on that, CodeRL feedback the execution result of example test cases 2 2 
 2 
 
 
 
 The example test cases are part of the NL description, not the test cases of the test set. into the fine-tuning process, and it is a model-agnostic approach that can combine with all the models in Table  3 to improve the performance.
Along this research line, introducing the feedback of given test cases into the upstream pre-training process is expected to gain further improvement.

 
 
 
 

## 7 Future Directions

 
 Taking advantage of both SE and NLP, NLP4P has achieved remarkable performance.
However, some features of the two fields (such as the iterations of SE and multilingual learning in NLP) have not been sufficiently explored.
Incorporating them is expected to further improve the performance.

 
 

### 7.1 Iterative NLP4P

 
 Generally, programming is an evolutionary process involving multiple iterations, rather than writing from scratch at one time.
For instance, it is difficult to solve a problem with a single submission despite the developers being experienced.
As a reference, the average accept rate of Codeforce, 3 3 
 3 
 
 
 
 http://codeforces.com/ a competitive programming website, is only 50.03%.

 
 
 Nevertheless, most existing models are trained to accomplish their tasks regardless of historical context information.
Taking the program synthesis as an instance,
once a program fails to satisfy its requirements, it will be re-generated from scratch.
The error message and previous version are not taken into account and efficiently utilized.
Iterative NLP4P, a progressive programming paradigm with a natural language interface, may shed light on this problem.
It has access to the complete historical context and thus can pay more attention to fixing existing problems and avoid introducing new bugs.

 
 
 

### 7.2 Multilingual NLP4P

 
 As the bloom of the open source software platform, e.g., GitHub,
source code, along with their NL descriptions, has accumulated to a considerable amount, making it possible to learn a data-driven NLP4P model.
However, the distribution of these data is highly unbalanced.
Most of the NL part is English, and the PL part is Java and Python.
As a result, the performance of low-resource NL and PL is much worse than the average performance.
For example, Ruby only takes a minor proportion in CodeSearchNet dataset and is inferior to other PL in both code search and comment generation tasks  Feng et al. (2020) .

 
 
 To bridge the gap between different languages, the simplest way is to translate a low-resource language into its high-resource counterpart.
For tasks whose input is low-resource NL, we can translate it into English before sending it to the model.
For tasks whose output is low-resource PL, we can first generate a Java program and subsequently translate it into the desired PL.
However, it introduces extra effort and cascading errors during the translation.
Multilingual learning approaches Conneau et al. (2020) ; Liu et al. (2020b) ; Xue et al. (2021) provide access to address the issue.
It can efficiently utilize the data presented in various languages, representing them in a unified semantic space and avoiding cascading errors.

 
 
 

### 7.3 Multi-modal NLP4P

 
 NL specification may refer to other modalities (e.g., figures) for better understanding.
For instance, the “Seven Bridge Problem”, a classical graph problem, is hard to understand by plain NL descriptions.
At the heart of the multi-modal approaches is the alignment of various modalities.
However, there is no such dataset in the area of NLP4P, and annotating a new one is costly.
Therefore, it would be crucial to utilize the knowledge entailed in the existing multi-modal datasets (e.g., COCO  Lin et al., 2014 ) and the language-vision pre-trained models (such as CLIP  Radford et al., 2021 , Flamingo  Alayrac et al., 2022 , and METALM  Hao et al., 2022 ).

 
 
 
 

## 8 Conclusion

 
 In this paper, we review a broad spectrum of NLP4P work.
We identify two intrinsic properties of PL: structure-based and functionality-oriented, which are at the heart of program understanding and generation, respectively.
They naturally partition the tasks, datasets, techniques, and models, highlighting the characteristics of each category.
Additionally, the structure-based property is the key to the choice of backbone models,
and the functionality-oriented property is the primary concern of evaluation methods.
Through the analysis, we list topics that have yet to be fully considered and might be worth researching in the future.

 
 
 

## Limitations

 
 This paper focuses on the intersection of NLP and SE.
Programming approaches whose algorithms are irrelevant to NLP and the input excludes NL are not fully discussed, e.g., the deductive and inductive program synthesis.
Similarly, universal NLP approaches not devoted to programming are less sufficiently introduced, e.g., the mechanism of the encoding and decoding processes.
The recent work whose techniques details are not publicly available yet (such as ChatGPT 4 4 
 4 
 
 
 
 https://openai.com/blog/chatgpt/ and CodeGeeX 5 5 
 5 
 
 
 
 https://github.com/THUDM/CodeGeeX ) is to be extended if more details are released.

 
 
 

## References

 
 
 Ahmad et al. (2020) 
 
Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2020.

 
 A
transformer-based approach for source code summarization .

 
 In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics , pages 4998–5007, Online. Association for
Computational Linguistics.

 

 
 Ahmad et al. (2021) 
 
Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021.

 
 Unified
pre-training for program understanding and generation .

 
 In Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies , pages 2655–2668, Online. Association for Computational
Linguistics.

 

 
 Alayrac et al. (2022) 
 
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr,
Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds,
et al. 2022.

 
 Flamingo: a visual language model for few-shot learning.

 
 arXiv preprint arXiv:2204.14198 .

 

 
 Allamanis and Brockschmidt (2017) 
 
Miltiadis Allamanis and Marc Brockschmidt. 2017.

 
 Smartpaste: Learning to adapt source code.

 
 arXiv preprint arXiv:1705.07867 .

 

 
 Allamanis et al. (2018) 
 
Miltiadis Allamanis, Marc Brockschmidt, and Mahmoud Khademi. 2018.

 
 Learning to represent programs with graphs.

 
 In International Conference on Learning Representations .

 

 
 Allamanis and Sutton (2013) 
 
Miltiadis Allamanis and Charles Sutton. 2013.

 
 Mining source code repositories at massive scale using language
modeling.

 
 In 2013 10th working conference on mining software repositories
(MSR) , pages 207–216. IEEE.

 

 
 Allen (1970) 
 
Frances E Allen. 1970.

 
 Control flow analysis.

 
 ACM Sigplan Notices , 5(7):1–19.

 

 
 Austin et al. (2021) 
 
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski,
David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and
Charles Sutton. 2021.

 
 Program synthesis with
large language models .

 
 CoRR , abs/2108.07732.

 

 
 Banerjee and Lavie (2005) 
 
Satanjeev Banerjee and Alon Lavie. 2005.

 
 METEOR: An automatic
metric for MT evaluation with improved correlation with human judgments .

 
 In Proceedings of the ACL Workshop on Intrinsic and Extrinsic
Evaluation Measures for Machine Translation and/or Summarization , pages
65–72, Ann Arbor, Michigan. Association for Computational Linguistics.

 

 
 Bunel et al. (2018) 
 
Rudy Bunel, Matthew Hausknecht, Jacob Devlin, Rishabh Singh, and Pushmeet
Kohli. 2018.

 
 Leveraging grammar and reinforcement learning for neural program
synthesis.

 
 In International Conference on Learning Representations .

 

 
 Caballero et al. (2016) 
 
Ethan Caballero, . OpenAI, and Ilya Sutskever. 2016.

 
 Description2Code
Dataset .

 

 
 Chen et al. (2021) 
 
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde
de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas
Joseph, Greg Brockman, et al. 2021.

 
 Evaluating large language models trained on code.

 

 
 Christopoulou et al. (2022) 
 
Fenia Christopoulou, Gerasimos Lampouras, Milan Gritta, Guchun Zhang, Yinpeng
Guo, Zhongqi Li, Qi Zhang, Meng Xiao, Bo Shen, Lin Li, et al. 2022.

 
 Pangu-coder: Program synthesis with function-level language modeling.

 
 arXiv preprint arXiv:2207.11280 .

 

 
 Clement et al. (2020) 
 
Colin B Clement, Dawn Drain, Jonathan Timcheck, Alexey Svyatkovskiy, and Neel
Sundaresan. 2020.

 
 Pymt5: multi-mode translation of natural language and python code
with transformers.

 
 arXiv preprint arXiv:2010.03150 .

 

 
 Conneau et al. (2020) 
 
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume
Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and
Veselin Stoyanov. 2020.

 
 Unsupervised
cross-lingual representation learning at scale .

 
 In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics , pages 8440–8451, Online. Association for
Computational Linguistics.

 

 
 Feng et al. (2020) 
 
Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun
Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020.

 
 CodeBERT: A pre-trained model for programming and natural languages .

 
 In Findings of the Association for Computational Linguistics:
EMNLP 2020 , pages 1536–1547, Online. Association for Computational
Linguistics.

 

 
 Ferrante et al. (1987) 
 
Jeanne Ferrante, Karl J Ottenstein, and Joe D Warren. 1987.

 
 The program dependence graph and its use in optimization.

 
 ACM Transactions on Programming Languages and Systems
(TOPLAS) , 9(3):319–349.

 

 
 Fried et al. (2022) 
 
Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi,
Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. 2022.

 
 Incoder: A generative model for code infilling and synthesis.

 
 arXiv preprint arXiv:2204.05999 .

 

 
 Gao et al. (2020) 
 
Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles
Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. 2020.

 
 The pile: An 800gb dataset of diverse text for language modeling.

 
 arXiv preprint arXiv:2101.00027 .

 

 
 Guo et al. (2022) 
 
Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022.

 
 UniXcoder:
Unified cross-modal pre-training for code representation .

 
 In Proceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers) , pages 7212–7225,
Dublin, Ireland. Association for Computational Linguistics.

 

 
 Guo et al. (2020) 
 
Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, LIU Shujie, Long Zhou,
Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, et al. 2020.

 
 Graphcodebert: Pre-training code representations with data flow.

 
 In International Conference on Learning Representations .

 

 
 Gupta et al. (2020) 
 
Kavi Gupta, Peter Ebert Christensen, Xinyun Chen, and Dawn Song. 2020.

 
 Synthesize, execute and debug: Learning to repair for neural program
synthesis.

 
 Advances in Neural Information Processing Systems ,
33:17685–17695.

 

 
 Hao et al. (2022) 
 
Yaru Hao, Haoyu Song, Li Dong, Shaohan Huang, Zewen Chi, Wenhui Wang, Shuming
Ma, and Furu Wei. 2022.

 
 Language models are general-purpose interfaces.

 
 arXiv preprint arXiv:2206.06336 .

 

 
 Hendrycks et al. (2021) 
 
Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora,
Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. 2021.

 
 Measuring coding challenge competence with apps.

 
 In Thirty-fifth Conference on Neural Information Processing
Systems Datasets and Benchmarks Track (Round 2) .

 

 
 Hindle et al. (2016) 
 
Abram Hindle, Earl T Barr, Mark Gabel, Zhendong Su, and Premkumar Devanbu.
2016.

 
 On the naturalness of software.

 
 Communications of the ACM , 59(5):122–131.

 

 
 Hochreiter and Schmidhuber (1997) 
 
Sepp Hochreiter and Jürgen Schmidhuber. 1997.

 
 Long short-term memory.

 
 Neural computation , 9(8):1735–1780.

 

 
 Hovsepyan et al. (2012) 
 
Aram Hovsepyan, Riccardo Scandariato, Wouter Joosen, and James Walden. 2012.

 
 Software vulnerability prediction using text analysis techniques.

 
 In Proceedings of the 4th international workshop on Security
measurements and metrics , pages 7–10.

 

 
 Hu et al. (2018) 
 
Xing Hu, Ge Li, Xin Xia, David Lo, and Zhi Jin. 2018.

 
 Deep code comment generation.

 
 In 2018 IEEE/ACM 26th International Conference on Program
Comprehension (ICPC) , pages 200–20010. IEEE.

 

 
 Husain et al. (2019) 
 
Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc
Brockschmidt. 2019.

 
 Codesearchnet challenge: Evaluating the state of semantic code
search.

 
 arXiv preprint arXiv:1909.09436 .

 

 
 Iyer et al. (2016) 
 
Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, and Luke Zettlemoyer. 2016.

 
 Summarizing source code
using a neural attention model .

 
 In Proceedings of the 54th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers) , pages 2073–2083,
Berlin, Germany. Association for Computational Linguistics.

 

 
 Iyer et al. (2018) 
 
Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, and Luke Zettlemoyer. 2018.

 
 Mapping language to
code in programmatic context .

 
 In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing , pages 1643–1652, Brussels, Belgium.
Association for Computational Linguistics.

 

 
 Krinke (2001) 
 
Jens Krinke. 2001.

 
 Identifying similar code with program dependence graphs.

 
 In Proceedings Eighth Working Conference on Reverse
Engineering , pages 301–309. IEEE.

 

 
 Lachmy et al. (2021) 
 
Royi Lachmy, Ziyu Yao, Greg Durrett, Milos Gligoric, Junyi Jessy Li, Ray
Mooney, Graham Neubig, Yu Su, Huan Sun, and Reut Tsarfaty. 2021.

 
 Proceedings of the 1st workshop on natural language processing for
programming (nlp4prog 2021).

 
 In Proceedings of the 1st Workshop on Natural Language
Processing for Programming (NLP4Prog 2021) .

 

 
 Le et al. (2022) 
 
Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Hoi.
2022.

 
 CodeRL:
Mastering code generation through pretrained models and deep reinforcement
learning .

 
 In Advances in Neural Information Processing Systems .

 

 
 LeCun et al. (1989) 
 
Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard,
Wayne Hubbard, and Lawrence D Jackel. 1989.

 
 Backpropagation applied to handwritten zip code recognition.

 
 Neural computation , 1(4):541–551.

 

 
 Li et al. (2018) 
 
Jian Li, Yue Wang, Michael R Lyu, and Irwin King. 2018.

 
 Code completion with neural attention and pointer networks.

 
 In Proceedings of the 27th International Joint Conference on
Artificial Intelligence , pages 4159–25.

 

 
 Li et al. (2016) 
 
Jiwei Li, Will Monroe, and Dan Jurafsky. 2016.

 
 A simple, fast diverse decoding algorithm for neural generation.

 
 arXiv preprint arXiv:1611.08562 .

 

 
 Li et al. (2022) 
 
Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser,
Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago,
et al. 2022.

 
 Competition-level code generation with alphacode.

 
 arXiv preprint arXiv:2203.07814 .

 

 
 Liang and Zhu (2018) 
 
Yuding Liang and Kenny Zhu. 2018.

 
 Automatic generation of text descriptive comments for code blocks.

 
 In Proceedings of the AAAI Conference on Artificial
Intelligence , volume 32.

 

 
 Lieberman (2001) 
 
Henry Lieberman. 2001.

 
 Your wish is my command: Programming by example .

 
 Morgan Kaufmann.

 

 
 Lin (2004) 
 
Chin-Yew Lin. 2004.

 
 ROUGE: A package for
automatic evaluation of summaries .

 
 In Text Summarization Branches Out , pages 74–81, Barcelona,
Spain. Association for Computational Linguistics.

 

 
 Lin et al. (2014) 
 
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva
Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014.

 
 Microsoft coco: Common objects in context.

 
 In European conference on computer vision , pages 740–755.
Springer.

 

 
 Liu et al. (2016a) 
 
Chang Liu, Xinyun Chen, Eui Chul Shin, Mingcheng Chen, and Dawn Song.
2016a.

 
 Latent attention for if-then program synthesis.

 
 Advances in Neural Information Processing Systems , 29.

 

 
 Liu et al. (2016b) 
 
Chia-Wei Liu, Ryan Lowe, Iulian Serban, Mike Noseworthy, Laurent Charlin, and
Joelle Pineau. 2016b.

 
 How NOT to evaluate
your dialogue system: An empirical study of unsupervised evaluation metrics
for dialogue response generation .

 
 In Proceedings of the 2016 Conference on Empirical Methods in
Natural Language Processing , pages 2122–2132, Austin, Texas. Association
for Computational Linguistics.

 

 
 Liu et al. (2020a) 
 
Shangqing Liu, Yu Chen, Xiaofei Xie, Jing Kai Siow, and Yang Liu.
2020a.

 
 Retrieval-augmented generation for code summarization via hybrid gnn.

 
 In International Conference on Learning Representations .

 

 
 Liu et al. (2020b) 
 
Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan
Ghazvininejad, Mike Lewis, and Luke Zettlemoyer. 2020b.

 
 Multilingual denoising pre-training for neural machine translation.

 
 Transactions of the Association for Computational Linguistics ,
8:726–742.

 

 
 Loyola et al. (2017) 
 
Pablo Loyola, Edison Marrese-Taylor, and Yutaka Matsuo. 2017.

 
 A neural architecture
for generating natural language descriptions from source code changes .

 
 In Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers) , pages 287–292,
Vancouver, Canada. Association for Computational Linguistics.

 

 
 Lu et al. (2021) 
 
Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio
Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al. 2021.

 
 Codexglue: A machine learning benchmark dataset for code
understanding and generation.

 
 In Thirty-fifth Conference on Neural Information Processing
Systems Datasets and Benchmarks Track (Round 1) .

 

 
 Manna and Waldinger (1980) 
 
Zohar Manna and Richard Waldinger. 1980.

 
 A deductive approach to program synthesis.

 
 ACM Transactions on Programming Languages and Systems
(TOPLAS) , 2(1):90–121.

 

 
 Mikolov et al. (2010) 
 
Tomas Mikolov, Martin Karafiát, Lukas Burget, Jan Cernockỳ, and Sanjeev
Khudanpur. 2010.

 
 Recurrent neural network based language model.

 
 In Interspeech , volume 2, pages 1045–1048. Makuhari.

 

 
 Mou et al. (2016) 
 
Lili Mou, Ge Li, Lu Zhang, Tao Wang, and Zhi Jin. 2016.

 
 Convolutional neural networks over tree structures for programming
language processing.

 
 In Thirtieth AAAI conference on artificial intelligence .

 

 
 Nie et al. (2022) 
 
Pengyu Nie, Jiyang Zhang, Junyi Jessy Li, Ray Mooney, and Milos Gligoric. 2022.

 
 Impact of evaluation methodologies on code summarization.

 
 In Proceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers) , pages 4936–4960.

 

 
 Nijkamp et al. (2022) 
 
Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio
Savarese, and Caiming Xiong. 2022.

 
 A conversational paradigm for program synthesis.

 
 arXiv preprint arXiv:2203.13474 .

 

 
 Niu et al. (2022) 
 
Changan Niu, Chuanyi Li, Bin Luo, and Vincent Ng. 2022.

 
 Deep learning meets software engineering: A survey on pre-trained
models of source code.

 
 arXiv preprint arXiv:2205.11739 .

 

 
 Panthaplackel et al. (2020) 
 
Sheena Panthaplackel, Pengyu Nie, Milos Gligoric, Junyi Jessy Li, and Raymond
Mooney. 2020.

 
 Learning to
update natural language comments based on code changes .

 
 In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics , pages 1853–1868, Online. Association for
Computational Linguistics.

 

 
 Papineni et al. (2002) 
 
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.

 
 Bleu: a method for
automatic evaluation of machine translation .

 
 In Proceedings of the 40th Annual Meeting of the Association
for Computational Linguistics , pages 311–318, Philadelphia, Pennsylvania,
USA. Association for Computational Linguistics.

 

 
 Polozov and Gulwani (2015) 
 
Oleksandr Polozov and Sumit Gulwani. 2015.

 
 Flashmeta: A framework for inductive program synthesis.

 
 In Proceedings of the 2015 ACM SIGPLAN International Conference
on Object-Oriented Programming, Systems, Languages, and Applications , pages
107–126.

 

 
 Puri et al. (2021) 
 
Ruchir Puri, David S Kung, Geert Janssen, Wei Zhang, Giacomo Domeniconi,
Vladimir Zolotov, Julian Dolby, Jie Chen, Mihir Choudhury, Lindsey Decker,
et al. 2021.

 
 Codenet: A large-scale ai for code dataset for learning a diversity
of coding tasks.

 
 arXiv preprint arXiv:2105.12655 .

 

 
 Radford et al. (2021) 
 
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al. 2021.

 
 Learning transferable visual models from natural language
supervision.

 
 In International Conference on Machine Learning , pages
8748–8763. PMLR.

 

 
 Raychev et al. (2016) 
 
Veselin Raychev, Pavol Bielik, and Martin Vechev. 2016.

 
 Probabilistic model for code with decision trees.

 
 ACM SIGPLAN Notices , 51(10):731–747.

 

 
 Ren et al. (2020) 
 
Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel
Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma. 2020.

 
 Codebleu: a method for automatic evaluation of code synthesis.

 
 arXiv preprint arXiv:2009.10297 .

 

 
 Roziere et al. (2020) 
 
Baptiste Roziere, Marie-Anne Lachaux, Lowik Chanussot, and Guillaume Lample.
2020.

 
 Unsupervised translation of programming languages.

 
 Advances in Neural Information Processing Systems ,
33:20601–20611.

 

 
 Russell et al. (2018) 
 
Rebecca Russell, Louis Kim, Lei Hamilton, Tomo Lazovich, Jacob Harer, Onur
Ozdemir, Paul Ellingwood, and Marc McConley. 2018.

 
 Automated vulnerability detection in source code using deep
representation learning.

 
 In 2018 17th IEEE international conference on machine learning
and applications (ICMLA) , pages 757–762. IEEE.

 

 
 Shi et al. (2021a) 
 
Ensheng Shi, Yanlin Wang, Lun Du, Hongyu Zhang, Shi Han, Dongmei Zhang, and
Hongbin Sun. 2021a.

 
 CAST:
Enhancing code summarization with hierarchical splitting and reconstruction
of abstract syntax trees .

 
 In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing , pages 4053–4062, Online and Punta Cana,
Dominican Republic. Association for Computational Linguistics.

 

 
 Shi et al. (2021b) 
 
Ensheng Shi, Yanlin Wang, Lun Du, Hongyu Zhang, Shi Han, Dongmei Zhang, and
Hongbin Sun. 2021b.

 
 Cast: Enhancing code summarization with hierarchical splitting and
reconstruction of abstract syntax trees.

 
 arXiv preprint arXiv:2108.12987 .

 

 
 Shido et al. (2019) 
 
Yusuke Shido, Yasuaki Kobayashi, Akihiro Yamamoto, Atsushi Miyamoto, and
Tadayuki Matsumura. 2019.

 
 Automatic source code summarization with extended tree-lstm.

 
 In 2019 International Joint Conference on Neural Networks
(IJCNN) , pages 1–8. IEEE.

 

 
 Svajlenko et al. (2014) 
 
Jeffrey Svajlenko, Judith F Islam, Iman Keivanloo, Chanchal K Roy, and
Mohammad Mamun Mia. 2014.

 
 Towards a big data curated benchmark of inter-project code clones.

 
 In 2014 IEEE International Conference on Software Maintenance
and Evolution , pages 476–480. IEEE.

 

 
 Svyatkovskiy et al. (2020) 
 
Alexey Svyatkovskiy, Shao Kun Deng, Shengyu Fu, and Neel Sundaresan. 2020.

 
 Intellicode compose: Code generation using transformer.

 
 In Proceedings of the 28th ACM Joint Meeting on European
Software Engineering Conference and Symposium on the Foundations of Software
Engineering , pages 1433–1443.

 

 
 Tu et al. (2014) 
 
Zhaopeng Tu, Zhendong Su, and Premkumar Devanbu. 2014.

 
 On the localness of software.

 
 In Proceedings of the 22nd ACM SIGSOFT International Symposium
on Foundations of Software Engineering , pages 269–280.

 

 
 Tufano et al. (2019) 
 
Michele Tufano, Cody Watson, Gabriele Bavota, Massimiliano Di Penta, Martin
White, and Denys Poshyvanyk. 2019.

 
 An empirical study on learning bug-fixing patches in the wild via
neural machine translation.

 
 ACM Transactions on Software Engineering and Methodology
(TOSEM) , 28(4):1–29.

 

 
 Vaswani et al. (2017) 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017.

 
 Attention is all you need.

 
 In Advances in Neural Information Processing Systems ,
volume 30. Curran Associates, Inc.

 

 
 Wang et al. (2021) 
 
Yue Wang, Weishi Wang, Shafiq Joty, and Steven CH Hoi. 2021.

 
 Codet5: Identifier-aware unified pre-trained encoder-decoder models
for code understanding and generation.

 
 arXiv preprint arXiv:2109.00859 .

 

 
 Wei et al. (2019) 
 
Bolin Wei, Ge Li, Xin Xia, Zhiyi Fu, and Zhi Jin. 2019.

 
 Code generation as a dual task of code summarization.

 
 Advances in neural information processing systems , 32.

 

 
 Weigelt et al. (2020) 
 
Sebastian Weigelt, Vanessa Steurer, Tobias Hey, and Walter F. Tichy. 2020.

 
 Programming
in Natural Language with fuSE: Synthesizing Methods from Spoken
Utterances Using Deep Natural Language Understanding .

 
 In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics , pages 4280–4295, Online. Association for
Computational Linguistics.

 

 
 Xue et al. (2021) 
 
Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya
Siddhant, Aditya Barua, and Colin Raffel. 2021.

 
 mT5: A
massively multilingual pre-trained text-to-text transformer .

 
 In Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies , pages 483–498, Online. Association for Computational
Linguistics.

 

 
 Yamaguchi et al. (2014) 
 
Fabian Yamaguchi, Nico Golde, Daniel Arp, and Konrad Rieck. 2014.

 
 Modeling and discovering vulnerabilities with code property graphs.

 
 In 2014 IEEE Symposium on Security and Privacy , pages
590–604. IEEE.

 

 
 Zhou et al. (2019) 
 
Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019.

 
 Devign: Effective vulnerability identification by learning
comprehensive program semantics via graph neural networks.

 
 Advances in neural information processing systems , 32.

 

 
 Zhu et al. (2022) 
 
Ming Zhu, Karthik Suresh, and Chandan K Reddy. 2022.

 
 Multilingual code snippets training for program translation.

 

 
 Zugner et al. (2021) 
 
Daniel Zugner, Tobias Kirschstein, Michele Catasta, Jure Leskovec, and Stephan
Gunnemann. 2021.

 
 Language-agnostic representation learning of source code from
structure and context.

 
 In International Conference on Learning Representations
(ICLR) .