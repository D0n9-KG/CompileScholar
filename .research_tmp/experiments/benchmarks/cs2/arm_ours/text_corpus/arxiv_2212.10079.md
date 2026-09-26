A Survey on Pretrained Language Models for Neural Code Intelligence 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.10079v1 [cs.SE] 20 Dec 2022 
 
 

# A Survey on Pretrained Language Models for Neural Code Intelligence

 
 
 Yichen Xu 
 
 Affiliation:  EPFL
 
 Email:  yichen.xu@epfl.ch 
 
    
 Yanqiao Zhu   
 
 Affiliation:  UCLA
 
 Email:  yzhu@cs.ucla.edu 
 

 Abstract 
 
 As the complexity of modern software continues to escalate, software engineering has become an increasingly daunting and error-prone endeavor. In recent years, the field of Neural Code Intelligence (NCI) has emerged as a promising solution, leveraging the power of deep learning techniques to tackle analytical tasks on source code with the goal of improving programming efficiency and minimizing human errors within the software industry.
Pretrained language models have become a dominant force in NCI research, consistently delivering state-of-the-art results across a wide range of tasks, including code summarization, generation, and translation.
In this paper, we present a comprehensive survey of the NCI domain, including a thorough review of pretraining techniques, tasks, datasets, and model architectures.
We hope this paper will serve as a bridge between the natural language and programming language communities, offering insights for future research in this rapidly evolving field.

 
 
 
 
 

 
 

## 1 Introduction

 
 Programming languages Pierce (2002) serve as the foundation of software, enabling humans to communicate with computers and instruct them to perform computation.
The process of developing software using programming languages, known as software development, has become a thriving industry that plays a crucial role in the modern digital world. However, software development involves a range of tasks beyond programming, including testing, documentation writing, and bug fixing, which are known to be challenging and require a high level of human expertise Brooks (1978) .

 
 
 To ease software development, code intelligence tools have emerged as a computer-aided approach to automatically analyze source code and solve software engineering tasks.
Previously, these tools were mostly based on static analysis technologies.
For example, Microsoft Intellisense 1 1 
 1 
 
 
 
 
 
 
 
 https://code.visualstudio.com/docs/editor/intellisense is a code intelligence tool that provides code completion suggestions and hints function signatures by statically analyzing user code and building a database of definitions, references, type signatures, and so on.
There are also tools for automatically detecting vulnerabilities in source code Ayewah et al. (2008) ; Engler and Musuvathi (2004) .
While these tools have been widely adopted in industry, they have limitations. One of the main limitations is that these tools are typically built for a specific programming language, requiring significant effort to migrate them to new languages. Additionally, dynamic languages like Python are difficult to analyze statically, making traditional code intelligence tools less effective for developers.

 
 
 Recently, researchers have started applying language models and leveraging pretraining strategies for code intelligence tasks, such as program synthesis Chen et al. (2021) ; Wang et al. (2021b) , documentation generation Wang et al. (2021b) ; Alon et al. (2019a) ; Feng et al. (2020) , defect detection, and program repair, inspired by the success of pretrained Transformer models on modeling sequential data Krizhevsky et al. (2017) ; Vaswani et al. (2017) .
Motivated by the software naturalness hypothesis Hindle et al. (2016) ; Buratti et al. (2020) , which suggests that programming languages can be understood and generated like natural languages, researchers have treated source code as sequential data and applied sequential neural architectures, like the Transformer model Vaswani et al. (2017) , to understand and generate programs Feng et al. (2020) ; Guo et al. (2021) .
In the Natural Language Processing (NLP) community, it has been observed that the pretraining paradigm enables models to learn high-quality context token embeddings and significantly improves downstream performance when a large amount of unannotated data is available Krizhevsky et al. (2017) ; Brown et al. (2020) .
Similarly, a vast amount of code snippets for programming languages can be found on open source platforms like GitHub Chen et al. (2021) . Thus, researchers have adopted the pretraining paradigm Devlin et al. (2019) to solve various code analytical tasks Feng et al. (2020) ; Guo et al. (2021) ; Chen et al. (2021) .
Such a data-driven approach greatly saves efforts in developing code intelligent tools for supporting different tasks and adapting to new programming languages compared to static analyzers.
To date, pretrained code language models have achieved state-of-the-art performance on a wide range of tasks and demonstrated satisfactory generalization across different languages Wang et al. (2021b) ; Chen et al. (2021) ; Li et al. (2022) .
Notably, Codex Chen et al. (2021) has made a significant milestone in program synthesis and has empowered programming intelligence tools like Copilot 2 2 
 2 
 
 
 
 
 
 
 
 https://github.com/features/copilot ,
which have even been applied to solving linear algebra and math word problems Tang et al. (2021) ; Drori and Verma (2021) .

 
 
 Despite the success of applying pretrained language models for code intelligence tasks and the flourishing research in this field, there has been a lack of systematic reviews that categorize the growing literature.
Previous works either do not adequately consider language-based code models nor provide a comprehensive review of existing model designs, downstream tasks, and datasets
 Xu et al. (2022) ; Wu et al. (2022a) ; Allamanis et al. (2018) .
In order to model and understand the semantics of programming languages, it is necessary to discuss available datasets for both pretraining and downstream tasks and how to design appropriate neural architectures and effective training schemes.
Moreover, since source code inherently has rich structural information Xu et al. (2022) ; Guo et al. (2021) , how to extract the utilize the structures as prior knowledge in designing code intelligence models is a vital problem.
This requires background knowledge about the theory and technologies in the programming language community.
 Wu et al. (2022b) survey deep learning methods for structural code understanding, with discussion on both sequence- and graph-based modeling techniques.
However, this work focuses more on the structural aspects of programs and lacks an in-depth discussion of language models and pretraining strategies.

 
 
 To bridge the knowledge of both NLP and PL communities, this paper groups existing pretraining language models for code intelligence under the umbrella term Neural Code Intelligence (NCI) and presents a systematic review in this field.
Specifically, we review neural modeling techniques for programming languages in terms of preprocessing techniques, model architectures, and learning paradigms. Next, we discuss a variety of downstream tasks for NCI and the available datasets for training and evaluating code language models. We also explore the challenges and opportunities of applying language modeling methods for code intelligence. In addition, we maintain a curated list of NCI research, news, and tools in a GitHub repository 3 3 
 3 
 
 
 
 
 
 
 
 https://github.com/Linyxus/awesome-neural-code-intelligence .
We hope that our work could shed light on the landscape of current research in this field, assist newcomers in understanding the recent research progress, and provide insights for future research.

 
 
 

## 2 Pretraining Language Models for Code

 
 Figure 1: The pipeline of code language models: the preprocessing phase tokenizes code texts into sequences of code tokens and extracts code structures; neural modeling encodes the code tokens and structures as dense embeddings or generates output code sequences; and training optimizes the model with language modeling or PL-specific objectives. 
 
 
 In this section, we review the design and training of code language models.
To review existing literature in a systematic manner, we split the code language modeling approaches into a pipeline consisting of three stages: preprocessing, sequential modeling, and training.
The pipeline is illustrated in Figure   1 and we will introduce each of them consecutively.

 
 
 Preprocessing. 

 
 For a code language model, the input will be source code snippets in the pretraining corpus or later on the downstream dataset.
It has to first preprocess the input code, including running tokenization on the code and optionally extracting prior knowledge from programming languages.

 
 
 
 Sequential modeling. 

 
 Then, it feeds the preprocessed results, mostly a sequence of tokens, into the languague model, to encode the code as dense representation, predict its property, or generate code sequences from it.

 
 
 
 Pretraining and finetuning. 

 
 Finally, we train the model with an unsupervised objective and further finetune it on downstream tasks.
In the pretraining stage, the models learns from a massive code corpus with no human annotations
o acquire general and transferable knowledge of the structures and semantics of source code.
Then, we finetune the pretrained model for specific downstream tasks.
Apart from the aforementioned pretraining-and-finetuning strategy, other learning paradigms including zero-shot, few-shot, and multi-task learning can also be applied to train the model.

 
 
 In the following sections, we review the components and different design choices of each stage.

 
 
 

### 2.1 Preprocessing

 
 The preprocessing stages accepts the raw code files as input and prepares it for the next stage by running tokenization and extracting structures of the source code.
For each sample, the output of this stage includes a sequence of tokens and optionally the structure of the source code.

 
 
 We run a tokenizer on the input code files. The obtained tokens for source code are sequences of characters which group together as an elementary unit for language modeling.

 
 
 Apart from tokenization, we can further optionally extract the structures from the code with the help of static analysis tools.
Programming languages are richly structured in their nature with syntax structures defined by its grammar rules and the semantic structures (e.g., flow graphs) revealed by semantic analysis tools.
The result of the structure extraction is typically represented in graphs.
For example, in a data dependency graph, nodes are variables and the edge from a node x x to another node y y means that the computation of y y involves x x .

 
 
 Next, we review tokenization strategies and structure extraction methods respectively.

 
 

#### 2.1.1 Tokenization of Source Code

 
 Tokenization is an indispensable preprocessing step of language models Vaswani et al. (2017) .
It splits the input text into a sequence of tokens that will be fed into the sequence model.
Previous work directly takes the tokenizers from the NLP community directly, such as BPE Sennrich et al. (2016) and SentencePiece Kudo and Richardson (2018) .
In addition, researchers also explore different strategies to better tokenize source code.
These strategies either improve the existing tokenizers that tailor for source code and make use of the naming conventions of programming languages and the tokenizer in their compilers to preserve the grammar and semantics of source code.

 
 
 Fitting subword tokenizers on source code. 

 
 The easiest way to obtain an efficient tokenizer is to fit a subword tokenizer on source code corpus.
For example, Buratti et al. (2020) propose a SentencePiece tokenizer on source code;
both CodeGPT Lu et al. (2021) and Austin et al. (2021) run the BPE algorithm Sennrich et al. (2016) on their programming-related corpus.

 
 
 
 Extending tokenizers with PL-specific tokens. 

 
 Additionally, the existing tokenizer can be improved with PL-specific tokens to encode source code efficiently.
 Phan et al. (2021) add special symbols commonly seen in source code (such as [ , { and $ ) into the SentencePiece tokenizer to better encode programming languages.
 Chen et al. (2021) propose to extend a pretrained BPE tokenizer by encoding continuous whitespace of different lengths into special tokens.
According to Chen et al. (2021) , the proposed extension
reduces the number of tokens for the same corpus and thus improves the efficiency of code language modeling.

 
 
 
 Utilizing tokenizers of compilers. 

 
 In natural languages, words are separated by spaces or punctuations, so their boundaries can be easily determined.
Unlike in natural languages that word boundaries are naturally determined by spaces and punctuations, the whitespace in programming languages is sometimes optional and the tokenizer shipped with the programming language scans tokens from source code based on a strictly-defined regular grammar Louden (1997) ; Aho et al. (2006) .
It is thus a natural idea to tokenize the source code with PL tokenizers before running the subword tokenizer of language models Kanade et al. (2020) ; Rozière et al. (2020) ; Wang et al. (2021b) .

 
 
 
 Exploiting naming conventions. 

 
 Another common preprocessing technique is to exploit naming conventions to better preserve the semantics of identifier names in the source code Svyatkovskiy et al. (2020) ; Peng et al. (2021b) .
Naming conventions are rules to concatenate words to form a valid program identifier.
Commonly used naming conventions are snake case (e.g. hello_world ) and camel case (e.g. helloWorld ).
During the tokenization phase, we can first split identifier names based on the naming conventions before running the tokenizers so that the natural language semantics of identifiers are preserved Peng et al. (2021b) .

 
 
 
 

#### 2.1.2 Extracting Structures from Source Code

 
 In addition to improving the tokenization strategy, researchers also propose to utilize the rich structural information of source code.
Programming languages have rich syntax and semantic structures, which can be extracted using parsing tools and semantic analysis techniques.
To be specific, it is possible to extract the Abstract Syntax Tree (AST), Control-Flow Graphs (CFGs), and Data-Flow Graphs (DFGs) in the preprocessing phase and utilize these structures in code modeling Peng et al. (2021b) ; Shiv and Quirk (2019) ; Guo et al. (2021) ; Wang et al. (2021a) ; Guo et al. (2022) .

 
 
 To use code structures, which are often in the form of graphs or trees, in neural networks, we can flatten them into sequences. This allows us to directly input the code structures into Transformers, which are designed to process sequential data.
One common method of flattening code structures is to obtain a sequence of nodes by traversing the graph or tree and using attention masks in the self-attention to recover the structural information,
which will be discussed in Section   2.2.2 .
An alternative method, proposed by Guo et al. (2022) , is to use an encoding function to map ASTs into sequences that can be directly input into the transformer without losing structural information.

 
 
 
 

### 2.2 Neural Modeling for Code Tokens

 
 After obtaining token sequences and the optional auxiliary structures from raw input, this stage models these code token sequences with additional structural information and produces dense code representations or generates the desired programming language snippets or natural language sequences.

 
 
 Existing code language models commonly include Recurrent Neural Networks (RNNs) Jordan (1997) ; Rumelhart et al. (1986) ; Ben-Nun et al. (2018) , Long Short-Term Memory (LSTM) networks Hochreiter and Schmidhuber (1997) ; Iyer et al. (2016) ; Yin and Neubig (2017) ; Alon et al. (2019b) and Transformers Vaswani et al. (2017) .
Among them, the majority of model extensively rely on the Transformers, the de facto architecture for sequential modeling.
Transformers stack multiple layers of self-attention layers to model sequence data,
To exploit the rich structure of programming languages, additional components and architectures specific to language specifications are also designed.

 
 

#### 2.2.1 Transformers in NCI Models

 
 Transformers are originally designed as an encoder-decoder model. However, other variants, such as encoder-only and decoder-only models, have also been proposed in the literature. In this section, we will explore the three architectures of Transformers and their applications in the field of NCI.

 
 
 Encoder-only transformers encode the input token sequence into dense vectors with a stack of self-attention blocks.
The resulting continuous embeddings have been widely used in code understanding models such as defect detection and code retrieval Guo et al. (2021) ; Feng et al. (2020) ; Buratti et al. (2020) .

 
 
 Encoder-decoder transformers , such as CodeT5 Wang et al. (2021b) and PLBART Ahmad et al. (2021) , have been leveraged in recent research to perform a variety of downstream tasks with prompting using multi-task training strategies similar to those used in T5 Raffel et al. (2020) .

 
 
 Decoder-only transformers , which are extensively used in program synthesis tasks, are designed to generate high-quality sequences. Examples include Codex Chen et al. (2021) and AlphaCode Li et al. (2022) .

 
 
 

#### 2.2.2 Exploiting Program Structures

 
 To incorporate the knowledge of structures in source code,
researchers propose to utilize the structural information in the self-attention module in Transformers.

 
 
 Syntax-based self-attention. 

 
 The self-attention module in Transformers can only model the linear position of tokens in the sequence.
While the linear modeling of sequences works perfectly for natural languages,
programming languages are structured in their nature,
and are usually parsed into Abstract Syntax Trees (ASTs) with grammar rules by compilers.
Therefore, modeling source code linearly neglects the rich syntactic structures of source code.
To model the tree structure of code syntax explicitly, it possible to modify the positional encodings in the attention module considering the syntactic structure.
For example, TPTrans Peng et al. (2021b) employs learnable positional encodings based on tree paths in the Transformer encoder.
Specifically, the tree encoding between two nodes is computed by running a Gated Recurrent Unit (GRU) network
on the path between them in ASTs.
 Shiv and Quirk (2019) propose a stack-based absolute positional encoding scheme that can represent the node position in trees.
On the decoder side, they compute the positional encodings from the partial tree on-the-fly to ensure the alignment between outputs and the positional encodings.

 
 
 
 Semantic-based self-attention. 

 
 Source code is structured not only on the syntax level, but also on the semantic level.
Static analysis technologies can extract the semantic structures of source code,
like the control-flow graphs and data-flow graphs.
 Guo et al. (2021) propose to incorporate data-flow structures into the attention module.
They first input the data flow graph as sequence together with the source code and
then mask the attention based on the edges in the data flow graph and the correspondence between identifiers in the source code and the nodes in the graph.

 
 
 
 
 

### 2.3 Training

 

#### 2.3.1 Language Model Pretraining

 
 Most code language models follow the pretraining paradigm.
The pretraining objective is a key ingredient here.
We first review the objectives that are based on language modeling in the NLP community,
then discuss the PL-specific objectives utilizing the prior knowledge of PL.

 
 
 Language modeling objectives. 

 
 Here, we briefly discuss the pretraining objectives that are widely used in natural language models and explain how to adapt these objectives for source code.

 
 
 
 • 
 
 Masked Language Modeling (MLM). 
Inspired by the cloze task Taylor (1953) ,
the MLM objective is used to pretrain BERT Devlin et al. (2019) in NLP research,
where we first mask out a portion of tokens in the input sequence and then ask the model to predict them.
A variety of encoder-only code language models utilize MLM objective to pretrain the model Buratti et al. (2020) ; Kanade et al. (2020) ; Guo et al. (2021) ; Feng et al. (2020) .

 

 • 
 
 Next Sentence Prediction (NSP). 
The NSP objective also originates in BERT Devlin et al. (2019) .
In each sample, we randomly select two sentences 𝒔 \bm{s} and 𝒕 \bm{t} ,
where 𝒕 \bm{t} can either be the real next-sentence of 𝒔 \bm{s} or sampled from other places.
The sample is organized as [ Cls ] [\textsc{Cls}] , s 1 , ⋯ , s | 𝒔 | {s}_{1},\cdots,{s}_{|\bm{s}|} , [ Sep ] [\textsc{Sep}] , t 1 , ⋯ , t | 𝒕 | {t}_{1},\cdots,{t}_{|\bm{t}|} , [ Sep ] [\textsc{Sep}] to be fed into the model.
The final embedding of [ Cls ] token is regarded as the sequence embedding and is used to predict whether 𝒕 \bm{t} is a real next-sentence.
When applying NSP on programming languages Kanade et al. (2020) ; Liu et al. (2020) , Kanade et al. (2020) proposes to use the logical lines in source code instead of the physical lines to better model the syntax of programming languages.

 

 • 
 
 Masked Span Prediction (MSP). 
The MSP obejctive is widely applied to train language models with encoder-decoder architectures Raffel et al. (2020) , which randomly masks spans of tokens in the input sequence and then predicts these masked spans combined with some sentinel tokens on the decoder side.
Code language models with a encoder-decoder architecture mostly train the model with NSP Wang et al. (2021b) .

 

 • 
 
 Unidirectional Language Modeling (LM). 
Many decoder-only language transformers Brown et al. (2020) leverage the LM objective,
where the model is to predict future token based on the previous sequence.
Specifically, when using the objective to train Transformers, we use casual attention masking, where each token in the sequence can only attend to the tokens before it.
Then, on the output layer, we employ a classification head on each output token embeddings to predict the next token the input sequence and use cross-entropy loss to optimize the model.
In NCI, code synthesis models based on GPT employ the LM objective Chen et al. (2021) ; Austin et al. (2021) .

 

 • 
 
 Denoising AutoEncoding (DAE). 
DAE is applied to train seq2seq language models Lample et al. (2018) ; Rozière et al. (2020) .
It corrupts the input sequence by randomly masking, removing, and shuffling the tokens and enforces the model to recover the original sequence from the corrupted one.
Note that the DAE loss can be regarded as an extension to the MLM loss, since it not only masks the input tokens but also permutes and drops them.
DAE is employed to train code translation models like TransCoder Rozière et al. (2020) and is shown to enhance the capability of the model to decode the internal representation of code Rozière et al. (2020) .

 

 • 
 
 Back translation. 
Back translation is a pretraining objective designed for cross-lingual language models Lample et al. (2018) .
 Rozière et al. (2020) ; Rozière et al. (2021) employ this objective to train programming language models that is capable of translating between different programming languages.
Specifically, for an input code sequence of language A A , we first ask the model to translate it to the equivalent code sequence of language B B .
Then, we enforce the model to translate the sequence in B B back to the original sequence in A A .

 

 
 
 
 
 Code-specific objectives. 

 
 Apart from objectives originated from language modeling, researchers also explore pretraining objectives specific for programming languages.
These objectives make use of the prior knowledge of programming language and software engineering domains and are proven to achieve performance improvements in many NCI tasks Rozière et al. (2021) .

 
 
 
 • 
 
 Replaced Token Detection (RTD). 
Inspired by Clark et al. (2020) , the RTD objective randomly replaces tokens in the input sequence with several tokens by a generator and trains a binary classifier to predict whether a token has been replaced.
It has been employed to train BERT-like code language models Feng et al. (2020) .

 

 • 
 
 Identifier DeOBFuscation (DOBF). 
Inspired by the notion of identifier deobfuscation in software engineering, the DOBF objective Rozière et al. (2021) first masks the function and variable names with placeholder tokens and then train the model to recover the original names from the placeholders.
DOBF has been shown to marginally improve model performance in code translation tasks Rozière et al. (2021) .

 

 • 
 
 Data-flow-based objectives. 
GraphCodeBERT Guo et al. (2021) incorporates data flow information of the input source code.
Data flow is a graph of variable dependencies in the source code,
where each node represents a variable and each edge means that the computation of one variable depends on another one.
 Guo et al. (2021) feed the data flow of source code into model input and design two data-flow-based pretraining objectives in their work,
namely Edge Prediction and Node Alignment .
Edge Prediction requires the model to predict masked edges in the data flow graph,
while Node Alignment asks the model to predict the alignment between identifier tokens in the source code and the nodes in the data flow graph.

 

 • 
 
 Contrastive Learning (CL). 
Contrastive learning trains the model by maximizing the similarity between positive samples. Specifically, it first generates various samples from each input data sample with data augmentations.
For each input sample, its augmented samples are regarded as positive samples, while other samples are treated as negative ones.
Then, CL optimizes the model by enforcing it to discriminate between positive and negative pairs.
Recently, ContraCode Jain et al. (2021) adapts the idea of CL to train source code language models.
ContraCode employs semantic-preserving code transformations as data augmentations and maximizes the agreement between embeddings of augmented programs with identical functionalities.
UniXCoder Guo et al. (2022) proposes a multi-modal contrastive learning objective that learns code semantic embeddings.

 

 
 
 
 
 

#### 2.3.2 Additional Learning Schemes

 
 Zero- and few-shot learning. 

 
 It is found that without finetuning on downstream tasks, large pretrained models are already capable of completing various tasks such as sentiment analysis, passage summarization, and question answering Brown et al. (2020) .
This is similar for code language models.
 Chen et al. (2021) propose Codex, a 12B code GPT pretrained on hundreds of gigabytes of Python source code.
It can achieve high accuracy on the program synthesis task without finetuning.
They also find that Codex largely outperforms finetuned counterparts in few-shot settings.
Surprisingly, Codex also has strong performance in solving linear algebra problems and math-word problems Drori and Verma (2021) ; Tang et al. (2021) in zero-shot settings, even if it was not optimized for any of these problems.
Moreover, researchers find that pretrained code models show promising performance for other downstream tasks such as neural execution, type inference, variable naming, and docstring generation.

 
 
 
 Multi-task learning. 

 
 Instead of training different models for different tasks, the multi-task learning paradigm trains a unified model on multiple downstream tasks Raffel et al. (2020) .
An example is CodeT5, which is a T5-based code language model Wang et al. (2021b) .
In the finetuning phase, it formalizes downstream tasks as several seq2seq problems and constructs prompts to hint the task type in the input.
Another example MulCode Wang et al. (2021a) has dedicated input and output layers for each downstream tasks and has a unified representation layer for modeling the source code.
The whole model is optimized jointly on multiple downstream tasks.

 
 
 
 
 
 

## 3 Tasks and Datasets

 
 In this section, we first review downstream NCI tasks and datasets in two categories: understanding and generation tasks.
Then, we review code corpus for pretraining code language models.

 
 

### 3.1 Downstream Tasks

 
 Downstream tasks can be summarized from four dimensions: task types , inputs and outputs , evaluation , and available datasets .
We first present a taxonomy of downstream tasks, explaining the meaning of each dimension, followed by detailed introduction to each task.
Finally, we review the datasets available for each downstream task and disucss widely-used pretraining corpus for code language models.
The downstream tasks are summarized in Table   1 .

 
 
 
 
 
 Name | 
 Type | 
 Input | 
 Output | 
 Evaluation | 
 Datasets | 

 
 Clone detection | 
 U | 
 PL | 
 Binary label | 
 Acc | 
 Svajlenko et al. (2014) | 

 
 Defect detection | 
 U | 
 PL | 
 Binary label | 
 Acc | 
 Zhou et al. (2019) , Zheng et al. (2021) | 

 
 Code retrieval | 
 U | 
 NL, PL | 
 Score | 
 MRR, S@ k k | 
 Husain et al. (2019) | 

 
 Code summarization | 
 G | 
 PL | 
 NL | 
 BLEU, ROUGE | 
 Lu et al. (2021) , LeClair and McMillan (2019) | 

 
 Program synthesis | 
 G | 
 NL, PL | 
 PL | 
 BLEU, CodeBLEU, functional correctness | 
 Iyer et al. (2018) , Chen et al. (2021) , Li et al. (2022) | 

 
 Code refinement | 
 G | 
 PL | 
 PL | 
 Acc | 
 Tufano et al. (2019) | 

 
 Code translation | 
 G | 
 PL | 
 PL | 
 Acc, BLEU, CodeBLEU | 
 Lu et al. (2021) | 

 
 Code completion | 
 G | 
 PL | 
 PL | 
 Acc, Edit Sim | 
 Lu et al. (2021) | 

 
 Table 1: Representative Neural Code Intelligence (NCI) tasks. Each task is characterized by its type (Understanding (U) or Generation (G)), input and output (Programming Language (PL) and Natural Language (NL)), evaluation protocols, and available datasets.
 
 
 
 
 • 
 
 Task types. 
Each downstream tasks can be categorized into two classes: Understanding tasks (U) and Generation tasks (G) .
An understanding task usually takes natural language and programming language sequences as input and requires the model to predict the corresponding label.
To complete such tasks, a model only needs to understand the meanings of natural language and source code.By contrast, the generation tasks will require the model to generate programming language or natural language texts from the input.
Apart from the class, we also specify the input and output of each task.
The inputs and outputs are usually Natural Languages (NL) , Programming Languages (PL) , and their labels.

 

 • 
 
 Evaluation protocols. 
Understanding tasks are typically formulated as a classification or retrieval problem, so the traditional metrics can be applied as evaluation metrics.
For generative tasks, we measure the performance in terms of the textual and structural similarity with ROUGE, BLEU, and CodeBLEU Papineni et al. (2002) ; Lin (2004) ; Ren et al. (2020) , or test the functional correctness of the generated programs Chen et al. (2021) .

 

 
 
 

#### 3.1.1 Understanding Tasks

 
 Clone detection Svajlenko et al. (2014) 

 
 involves identifying plagiarism between two code snippets by measuring their semantic similarity. The performance of clone detection algorithms can be evaluated using metrics such as accuracy.
BigCloneBench Svajlenko et al. (2014) and CodeNet Puri et al. (2021) are two representative benchmarks for evaluating the performance of code clone detection algorithms.

 
 
 
 Defect detection Zhou et al. (2019) 

 
 aims to predict the vulnerability of given source code, or whether it contains bugs and defects that may result in runtime errors or attacks.
The input for this task is typically code snippets and the output is a binary label indicating vulnerability.
As a binary classification problem, its performance can be evaluated in accuracy.
Devign Zhou et al. (2019) a defect detection dataset created from four large-scale open-source C projects, comprising 48,687 samples with diverse vulnerabilities.
D2A Zheng et al. (2021) is another defect detection dataset created by analyzing code before and after commits, providing 1,295,623 samples.

 
 
 
 Code retrieval Husain et al. (2019) ; Ling et al. (2021) 

 
 which is also known as semantic code search, aims to retrieve code snippets relevant to a given natural language query.
The task can be naturally formulated as a matching problem: the model will be given a pair of NL and PL sequences representing the query and the candidate code snippets. Then, the model should output a relevance score between the query and each of the candidate snippets.
Ranking metrics like MRR and S@ k k can be used to evaluate model outputs Ling et al. (2021) .
CodeSearchNet Challenge Husain et al. (2019) is a code retrieval dataset built from open source repositories of six popular programming languages.

 
 
 
 

#### 3.1.2 Generation Tasks

 
 Code summarization Lu et al. (2021) ; LeClair and McMillan (2019) 

 
 generates explanatory natural language documentation from the given source code snippets.
The output natural language description can be evaluated with BLEU and ROUGE scores.
The CodeSearchNet corpus Husain et al. (2019) (which comes with the CodeSearchNet Challenge dataset) can be used as the code summarization dataset Lu et al. (2021) .
FunCom LeClair and McMillan (2019) is a code summarization benchmark of over two million Java method paired with one-line natural language description.

 
 
 
 Program synthesis Chen et al. (2021) ; Li et al. (2022) 

 
 or known as code generation, requires the model to generate programs from text specifications.
The input of this task is usually a program specification, which can be natural language descriptions, pseudo-code, or example input-output test cases.
These specifications can be represented in the form of natural language, programming language or the combination of the two.
In addition to textual similarity scores like BLEU and ROUGE, generated programs can also be evaluated with CodeBLEU and functional correctness, which will take the syntactic and semantic aspects of PL into consideration.

 
 
 There are a variety of available datasets for this task.
CONCODE Iyer et al. (2018) is a Java code generation dataset scraped from GitHub where each sample asks the model to generate source code snippet from natural language query in programming contexts.
HumanEval Chen et al. (2021) is a program synthesis benchmark that consists of hundreds of hand-written coding challenges.
CodeContests Li et al. (2022) and APPS Hendrycks et al. (2021) are program synthesis datasets extracted from online coding platforms like Codeforces, CodeChef, and Codewars.

 
 
 
 Code refinement Tufano et al. (2019) ; Drain et al. (2021) 

 
 aims to fix bugs in the given code snippet and output a bug-free one.
This task can be viewed as an extension of defect detection, in the sense that code refinement has to not only find the defect but also fix it.
The output is usually evaluated with the exact matching accuracy (i.e. the predicted fix matches with the ground truth exactly), since code refinement is a correctness-critical task.
 Tufano et al. (2019) propose a code refinement dataset mined from bug-fixing commits in thousands of Java projects hosted on GitHub.

 
 
 
 Code translation Lu et al. (2021) ; Rozière et al. (2020) 

 
 translates code snippets in one language to another, which is useful for migrating code bases to a new language.
Similarly, the translated code can be evaluated with accuracy, BLEU, and CodeBLEU.
CodeTrans Lu et al. (2021) is a Java-C# code translation extracted from parallel functions in several open-source projects.

 
 
 
 Code completion 

 
 aims to complete partial code snippets.
 Lu et al. (2021) propose a code completion benchmark dataset built from PY150 Raychev et al. (2016) and GitHub Java corpus Allamanis and Sutton (2013) .
The benchmark contains two subtasks: token- and line-level completion, which require the model to predict the next single token or next line of tokens respectively.
The token level completion is evaluated with accuracy;
the line-level prediction can be evaluated with edit similarity as in CodeXGLUE Lu et al. (2021) .

 
 
 
 
 

### 3.2 Pretraining Corpus

 
 Now, we introduce several representative code pretraining corpus.
We summarize commonly-used pretraining corpus in Table   2 .

 
 
 
 
 
 Name | 
 Language | 
 Size | 
 Content | 

 
 GitHub Java | 
 Java | 
 3 GB | 
 Java code files | 

 
 PY150 | 
 Python | 
 ≈ \approx 350 MB | 
 Abstract syntax trees | 

 
 CodeSearchNet | 
 Multiple | 
 — | 
 Code-description pairs | 

 
 The Pile | 
 Multiple | 
 630.64 GB | 
 Code files | 

 
 CodeParrot | 
 Python | 
 180 GB | 
 Python code files | 

 
 The Stack | 
 Multiple | 
 3.1 TB | 
 Code files | 

 
 Table 2: Statistics of representative pretraining corpus. 
 
 
 GitHub Java corpus Allamanis and Sutton (2013) 

 
 is constructed from 14,807 Java projects on GitHub.
It contains 350 million lines of code and roughly 1.5B of code tokens, summing up to more than 3 gigabytes.

 
 
 
 PY150 Kanade et al. (2020) 

 
 contains parsed abstract syntax trees of Python programs scraped from GitHub.
The train split consists of 100,000 files and the evaluation split has 50,000 files.

 
 
 
 CodeSearchNet corpus Husain et al. (2019) 

 
 is built from GitHub projects in multiple mainstream programming languages.
It contains 2 millions of functions parsed from the source code, each paired with its natural language documentation string.

 
 
 
 The Pile Gao et al. (2021) 

 
 is an open-source English corpus for pretraining general purpose language models.
It contains 630.64 GB of source code text in multiple programming languages.

 
 
 
 CodeParrot Tunstall et al. (2022) 

 
 is a Python code corpus built from public GitHub repositories available on Google’s BigQuery.
After preprocessing, it results in a dataset containing 180 gigabytes of Python code from over 20 million source files.

 
 
 
 The Stack Kocetkov et al. (2022) 

 
 is a 3.1TB source code dataset consisting of permissively licensed code in 30 programming languages.
The dataset is collected from GitHub, after license filtration and near-deduplication.

 
 
 
 
 

## 4 Challenges and Opportunities

 
 Despite the blossom development of this field, there are still many outstanding challenges and open problems.
Next, we discuss possible opportunities for follow-up research.

 
 
 Exploiting prior knowledge of source code. 

 
 Programming languages are well-structured by its nature:
we can parse source code into abstract syntax trees with the grammar rules;
also, there are a variety of static analyzers to extract the semantic structures of programs, ranging from type checkers to flow analyzers.
Although prior work makes a few attempts in exploiting these knowledge Peng et al. (2021a) ; Shiv and Quirk (2019) ; Guo et al. (2021) ; Wang et al. (2021a) , there is still a lack of principled way to inject structure bias in language models designed for sequential data.
Additionally, programs can be executed, revealing the runtime semantics of programs, which is the essential property of source code for various tasks, in particular program synthesis Chen et al. (2021) ; Li et al. (2022) .
Most previous work Chen et al. (2021) ; Simmons-Edler et al. (2018) focuses on machine-level instructions and cannot generate general-purpose programs with high-level programming languages.
Utilizing such rich prior knowledge for NCI models, especially the runtime semantics is a promising research direction in the future.

 
 
 
 Linking project- and library-level knowledge. 

 
 The input and output of existing code language models are mostly functions, classes, and standalone code snippets.
By contrast, real-world source code projects have a hierarchy of source code modules that have to be considered simultaneously.
Existing methods are limited to understand and generate source code in a single file and lacks the capability of modelling interconnected code modules, which is essential for NCI methods to scale up to modeling real-world code projects.

 
 
 
 Broadening the horizons of NCI models. 

 
 Researchers have discovered that code language models have applications beyond understanding and generating source code. For example, program synthesis models have been used to solve linear algebra problems, math word problems, and perform automated proof search Tang et al. (2021) ; Drori and Verma (2021) ; Polu et al. (2022) .
Specifically, Polu et al. (2022) find that code language models can generate mathematical proofs, with their GPT-f model capable of proving theorems at the International Mathematical Olympiad level Polu et al. (2022) .
Codex has also been utilized in computer science education, such as solving code problems, explaining code snippets, and generating course materials Finnie-Ansley et al. (2022) ; MacNeil et al. (2022) .
ProgPrompt Singh et al. (2022) demonstrates that code language models can generate task plans for robots.
In addition, it is likely that code language models could be applied to other tasks such as accelerating SAT solvers, parsing NL into SQL queries, and solving a broader range of math problems. Applying code models in these domains is an emerging but exciting area of research.

 
 
 
 

## 5 Conclusion

 
 In this survey, we develop a systematic review of the tasks, datasets, and methodology of language-modeling-based Neural Code Intelligence (NCI) models.
Our review begins with an overview of NCI tasks and datasets, followed by a detailed analysis of existing language-modeling methods.
We also delve into the challenges and opportunities presented by this rapidly evolving field. We hope that our work will provide both novice and experienced researchers with a clear understanding of the current state of NCI research and offer insights into future trends and directions.

 
 
 

## 6 Limitations and Social Impacts

 

### 6.1 Limitations

 
 As a survey of the neural code intelligence field, one main limitation of this paper is that it focuses primarily on sequential-modeling-based models, while structural models are only briefly mentioned. While this is a deliberate decision in light of the scope of this survey, it is worth noting that structural models, which heavily rely on prior knowledge of code semantic structures, form a significant portion of NCI research and deserve further discussion.
Another limitation is that this survey does not address ethical and liability issues related to code language models. For example, large code language models such as Codex Chen et al. (2021) and AlphaCode Li et al. (2022) are often trained on large code corpora collected from open source communities. There are concerns that using open source code to train language models may violate open source licenses Robertson (2021) .

 
 
 

### 6.2 Social Impacts

 
 Our work has both positive and potentially negative impacts on society. On the positive side, our survey illustrates the current state and envisions future directions of applying language modeling to Neural Code Intelligence (NCI), which can help guide subsequent research in this field. By advancing the research in NCI, we can improve the reliability and efficiency of the software engineering industry and enable humans to build software more efficiently.
However, there is a potential negative impact as well. This survey and much of the NCI research community have not adequately addressed ethical and open source license issues. Code language models are trained on source code produced by humans, and it is thus important to respect the rights of the programmers who wrote the code.

 
 
 
 

## References

 
 
 Ahmad et al. (2021) 
 
Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021.

 
 Unified pre-training for program understanding and generation.

 
 In NAACL-HLT , pages 2655–2668.

 

 
 Aho et al. (2006) 
 
Alfred V. Aho, Monica S. Lam, Ravi Sethi, and Jeffrey D. Ullman. 2006.

 
 Compilers: Principles, Techniques, and Tools (2nd Edition) .

 
 Addison-Wesley Longman Publishing Co., Inc., USA.

 

 
 Allamanis et al. (2018) 
 
Miltiadis Allamanis, Earl T. Barr, Premkumar T. Devanbu, and Charles Sutton.
2018.

 
 A survey of machine learning for big code and naturalness.

 
 In ACM Comput. Surv. , pages 81:1–81:37.

 

 
 Allamanis and Sutton (2013) 
 
Miltiadis Allamanis and Charles Sutton. 2013.

 
 Mining source code repositories at massive scale using language
modeling.

 
 In Proceedings of the 10th Working Conference on Mining
Software Repositories , MSR ’13, page 207–216. IEEE Press.

 

 
 Alon et al. (2019a) 
 
Uri Alon, Shaked Brody, Omer Levy, and Eran Yahav. 2019a.

 
 code2seq: Generating sequences from structured representations of
code.

 
 In ICLR .

 

 
 Alon et al. (2019b) 
 
Uri Alon, Meital Zilberstein, Omer Levy, and Eran Yahav. 2019b.

 
 code2vec: Learning distributed representations of code.

 
 In Proc. ACM Program. Lang. , pages 40:1–40:29.

 

 
 Austin et al. (2021) 
 
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski,
David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and
Charles Sutton. 2021.

 
 Program synthesis with large language models.

 
 In arXiv .

 

 
 Ayewah et al. (2008) 
 
Nathaniel Ayewah, David Hovemeyer, J. David Morgenthaler, John Penix, and
William W. Pugh. 2008.

 
 Experiences using static analysis to find bugs.

 

 
 Ben-Nun et al. (2018) 
 
Tal Ben-Nun, Alice Shoshana Jakobovits, and Torsten Hoefler. 2018.

 
 Neural code comprehension: A learnable representation of code
semantics.

 
 In NeurIPS , pages 3589–3601.

 

 
 Brooks (1978) 
 
Frederick P. Brooks. 1978.

 
 The Mythical Man-Month: Essays on Softw , 1st edition.

 
 Addison-Wesley Longman Publishing Co., Inc., USA.

 

 
 Brown et al. (2020) 
 
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan,
Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter,
Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray,
Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford,
Ilya Sutskever, and Dario Amodei. 2020.

 
 Language models are few-shot learners.

 
 In NeurIPS .

 

 
 Buratti et al. (2020) 
 
Luca Buratti, Saurabh Pujar, Mihaela A. Bornea, J. Scott McCarley, Yunhui
Zheng, Gaetano Rossiello, Alessandro Morari, Jim Laredo, Veronika Thost,
Yufan Zhuang, and Giacomo Domeniconi. 2020.

 
 Exploring software naturalness through neural language models.

 
 arXiv .

 

 
 Chen et al. (2021) 
 
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira
Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg
Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy
Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder,
Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens
Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias
Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss,
William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor
Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher
Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa,
Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter
Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and
Wojciech Zaremba. 2021.

 
 Evaluating large language models trained on code.

 
 arXiv .

 

 
 Clark et al. (2020) 
 
Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. 2020.

 
 Electra: Pre-training text encoders as discriminators rather than
generators.

 
 In ICLR .

 

 
 Devlin et al. (2019) 
 
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019.

 
 Bert: Pre-training of deep bidirectional transformers for language
understanding.

 
 In NAACL-HLT , pages 4171–4186.

 

 
 Drain et al. (2021) 
 
Dawn Drain, Chen Wu, Alexey Svyatkovskiy, and Neel Sundaresan. 2021.

 
 Generating bug-fixes using pretrained transformers.

 
 Proceedings of the 5th ACM SIGPLAN International Symposium on
Machine Programming .

 

 
 Drori and Verma (2021) 
 
Iddo Drori and Nakul Verma. 2021.

 
 Solving linear algebra by program synthesis.

 
 arXiv .

 

 
 Engler and Musuvathi (2004) 
 
Dawson R. Engler and Madan Musuvathi. 2004.

 
 Static analysis versus software model checking for bug finding.

 
 In VMCAI .

 

 
 Feng et al. (2020) 
 
Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun
Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020.

 
 Codebert: A pre-trained model for programming and natural languages.

 
 In EMNLP , pages 1536–1547.

 

 
 Finnie-Ansley et al. (2022) 
 
James Finnie-Ansley, Paul Denny, Brett A. Becker, Andrew Luxton-Reilly, and
James Prather. 2022.

 
 The robots are
coming: Exploring the implications of openai codex on introductory
programming .

 
 In Australasian Computing Education Conference , ACE ’22, page
10–19, New York, NY, USA. Association for Computing Machinery.

 

 
 Gao et al. (2021) 
 
Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles
Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser,
and Connor Leahy. 2021.

 
 The pile: An 800gb dataset of diverse text for language modeling.

 

 
 Guo et al. (2022) 
 
Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022.

 
 Unixcoder: Unified cross-modal pre-training for code representation.

 
 In Annual Meeting of the Association for Computational
Linguistics .

 

 
 Guo et al. (2021) 
 
Daya Guo, Shuo Ren, Shuai Lu, Zhangyin Feng, Duyu Tang, Shujie Liu, Long Zhou,
Nan Duan, Alexey Svyatkovskiy, Shengyu Fu, Michele Tufano, Shao Kun Deng,
Colin B. Clement, Dawn Drain, Neel Sundaresan, Jian Yin, Daxin Jiang, and
Ming Zhou. 2021.

 
 Graphcodebert: Pre-training code representations with data flow.

 
 In ICLR .

 

 
 Hendrycks et al. (2021) 
 
Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora,
Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob
Steinhardt. 2021.

 
 Measuring coding challenge competence with apps.

 
 arXiv .

 

 
 Hindle et al. (2016) 
 
Abram Hindle, Earl T. Barr, Mark Gabel, Zhendong Su, and Premkumar T. Devanbu.
2016.

 
 On the naturalness of software.

 
 In Commun. ACM , pages 122–131.

 

 
 Hochreiter and Schmidhuber (1997) 
 
Sepp Hochreiter and Jürgen Schmidhuber. 1997.

 
 Long short-term
memory .

 
 Neural Comput. , 9(8):1735–1780.

 

 
 Husain et al. (2019) 
 
Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc
Brockschmidt. 2019.

 
 Codesearchnet challenge: Evaluating the state of semantic code
search.

 
 arXiv .

 

 
 Iyer et al. (2016) 
 
Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, and Luke Zettlemoyer. 2016.

 
 Summarizing source code using a neural attention model.

 
 In ACL .

 

 
 Iyer et al. (2018) 
 
Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, and Luke Zettlemoyer. 2018.

 
 Mapping language to code in programmatic context.

 
 In EMNLP , pages 1643–1652.

 

 
 Jain et al. (2021) 
 
Paras Jain, Ajay Jain, Tianjun Zhang, Pieter Abbeel, Joseph Gonzalez, and Ion
Stoica. 2021.

 
 Contrastive code representation learning.

 
 In EMNLP , pages 5954–5971.

 

 
 Jordan (1997) 
 
Michael I. Jordan. 1997.

 
 Serial order: A parallel distributed processing approach.

 
 Advances in psychology , 121:471–495.

 

 
 Kanade et al. (2020) 
 
Aditya Kanade, Petros Maniatis, Gogul Balakrishnan, and Kensen Shi. 2020.

 
 Learning and evaluating contextual embedding of source code.

 
 In ICML , pages 5110–5121.

 

 
 Kocetkov et al. (2022) 
 
Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou,
Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes,
Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. 2022.

 
 The stack: 3 tb of permissively licensed source code.

 
 ArXiv , abs/2211.15533.

 

 
 Krizhevsky et al. (2017) 
 
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2017.

 
 Imagenet classification with deep convolutional neural networks.

 
 In Commun. ACM , pages 84–90.

 

 
 Kudo and Richardson (2018) 
 
Taku Kudo and John Richardson. 2018.

 
 Sentencepiece: A simple and language independent subword tokenizer
and detokenizer for neural text processing.

 
 In EMNLP , pages 66–71.

 

 
 Lample et al. (2018) 
 
Guillaume Lample, Alexis Conneau, Ludovic Denoyer, and Marc’Aurelio Ranzato.
2018.

 
 Unsupervised machine translation using monolingual corpora only.

 
 In ICLR .

 

 
 LeClair and McMillan (2019) 
 
Alexander LeClair and Collin McMillan. 2019.

 
 Recommendations for datasets for source code summarization.

 
 In NAACL-HLT , pages 3931–3937.

 

 
 Li et al. (2022) 
 
Yujia Li, David H. Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser,
Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago,
Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin,
Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov,
James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli,
Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. 2022.

 
 Competition-level code generation with alphacode.

 
 arXiv .

 

 
 Lin (2004) 
 
Chin-Yew Lin. 2004.

 
 ROUGE: A package for
automatic evaluation of summaries .

 
 In Text Summarization Branches Out , pages 74–81, Barcelona,
Spain. Association for Computational Linguistics.

 

 
 Ling et al. (2021) 
 
Xiang Ling, Lingfei Wu, Sai gang Wang, Gaoning Pan, Tengfei Ma, Fangli Xu,
Alex X. Liu, Chunming Wu, and Shouling Ji. 2021.

 
 Deep graph matching and searching for semantic code retrieval.

 
 ACM Transactions on Knowledge Discovery from Data (TKDD) , 15:1
– 21.

 

 
 Liu et al. (2020) 
 
Fang Liu, Ge Li, Yunfei Zhao, and Zhi Jin. 2020.

 
 Multi-task learning based pre-trained language model for code
completion.

 
 In ASE , pages 473–485.

 

 
 Louden (1997) 
 
Kenneth C. Louden. 1997.

 
 Compiler Construction: Principles and Practice .

 
 PWS Publishing Co., USA.

 

 
 Lu et al. (2021) 
 
Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio
Blanco, Colin B. Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong
Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan,
Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. 2021.

 
 Codexglue - a machine learning benchmark dataset for code
understanding and generation.

 
 arXiv .

 

 
 MacNeil et al. (2022) 
 
Stephen MacNeil, Andrew Tran, Juho Leinonen, Paul Denny, Joanne Kim, Arto
Hellas, Seth Bernstein, and Sami Sarsa. 2022.

 
 Automatically generating cs learning materials with large language
models.

 

 
 Papineni et al. (2002) 
 
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.

 
 Bleu: a method for
automatic evaluation of machine translation .

 
 In Proceedings of the 40th Annual Meeting of the Association
for Computational Linguistics , pages 311–318, Philadelphia, Pennsylvania,
USA. Association for Computational Linguistics.

 

 
 Peng et al. (2021a) 
 
Dinglan Peng, Shuxin Zheng, Yatao Li, Guolin Ke, Di He, and Tie-Yan Liu.
2021a.

 
 How could neural networks understand programs?

 
 In ICML , pages 8476–8486.

 

 
 Peng et al. (2021b) 
 
Han Peng, Wenhan Wang, Yunfei Zhao, and Zhi Jin. 2021b.

 
 Integrating tree path in transformer for code representation.

 
 NeurIPS .

 

 
 Phan et al. (2021) 
 
Long N. Phan, Hieu Tran, Daniel Le, Hieu Nguyen, James T. Anibal, Alec
Peltekian, and Yanfang Ye. 2021.

 
 Cotext: Multi-task learning with code-text transformer.

 
 arXiv .

 

 
 Pierce (2002) 
 
Benjamin C. Pierce. 2002.

 
 Types and Programming Languages , 1st edition.

 
 The MIT Press.

 

 
 Polu et al. (2022) 
 
Stanislas Polu, Jesse Michael Han, Kunhao Zheng, Mantas Baksys, Igor
Babuschkin, and Ilya Sutskever. 2022.

 
 Formal mathematics statement curriculum learning.

 

 
 Puri et al. (2021) 
 
Ruchi Puri, David S. Kung, Geert Janssen, Wei Zhang, Giacomo Domeniconi,
Vladmir Zolotov, Julian Dolby, Jie Chen, Mihir R. Choudhury, Lindsey Decker,
Veronika Thost, Luca Buratti, Saurabh Pujar, and Ulrich Finkler. 2021.

 
 Project codenet: A large-scale ai for code dataset for learning a
diversity of coding tasks.

 
 ArXiv , abs/2105.12655.

 

 
 Raffel et al. (2020) 
 
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael
Matena, Yanqi Zhou, Wei Li 0133, and Peter J. Liu. 2020.

 
 Exploring the limits of transfer learning with a unified text-to-text
transformer.

 
 In J. Mach. Learn. Res. , pages 140:1–140:67.

 

 
 Raychev et al. (2016) 
 
Veselin Raychev, Pavol Bielik, and Martin Vechev. 2016.

 
 Probabilistic model
for code with decision trees .

 
 SIGPLAN Not. , 51(10):731–747.

 

 
 Ren et al. (2020) 
 
Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel
Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma. 2020.

 
 Codebleu - a method for automatic evaluation of code synthesis.

 

 
 Robertson (2021) 
 
Donald Robertson. 2021.

 
 Fsf-funded call for white papers on philosophical and legal questions around
copilot .

 

 
 Rozière et al. (2020) 
 
Baptiste Rozière, Marie-Anne Lachaux, Lowik Chanussot, and Guillaume Lample.
2020.

 
 Unsupervised translation of programming languages.

 
 In NeurIPS .

 

 
 Rozière et al. (2021) 
 
Baptiste Rozière, Marie-Anne Lachaux, Marc Szafraniec, and Guillaume Lample.
2021.

 
 Dobf: A deobfuscation pre-training objective for programming
languages.

 
 arXiv .

 

 
 Rumelhart et al. (1986) 
 
David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. 1986.

 
 Learning internal representations by error propagation.

 

 
 Sennrich et al. (2016) 
 
Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016.

 
 Neural machine translation of rare words with subword units.

 
 In ACL .

 

 
 Shiv and Quirk (2019) 
 
Vighnesh Leonardo Shiv and Chris Quirk. 2019.

 
 Novel positional encodings to enable tree-based transformers.

 
 In NeurIPS , pages 12058–12068.

 

 
 Simmons-Edler et al. (2018) 
 
Riley Simmons-Edler, Anders Miltner, and H. Sebastian Seung. 2018.

 
 Program synthesis through reinforcement learning guided tree search.

 

 
 Singh et al. (2022) 
 
Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan
Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. 2022.

 
 Progprompt: Generating situated robot task plans using large language
models.

 
 ArXiv , abs/2209.11302.

 

 
 Svajlenko et al. (2014) 
 
Jeffrey Svajlenko, Judith F. Islam, Iman Keivanloo, Chanchal Kumar Roy, and
Mohammad Mamun Mia. 2014.

 
 Towards a big data curated benchmark of inter-project code clones.

 
 2014 IEEE International Conference on Software Maintenance and
Evolution , pages 476–480.

 

 
 Svyatkovskiy et al. (2020) 
 
Alexey Svyatkovskiy, Shao Kun Deng, Shengyu Fu, and Neel Sundaresan. 2020.

 
 Intellicode compose - code generation using transformer.

 
 In ESEC/SIGSOFT FSE , pages 1433–1443.

 

 
 Tang et al. (2021) 
 
Leonard Tang, Elizabeth Ke, Nikhil Singh, Nakul Verma, and Iddo Drori. 2021.

 
 Solving probability and statistics problems by program synthesis.

 
 arXiv .

 

 
 Taylor (1953) 
 
Wilson L. Taylor. 1953.

 
 “cloze procedure”: A new tool for measuring readability.

 
 Journalism Mass Communication Quarterly , 30:415 – 433.

 

 
 Tufano et al. (2019) 
 
Michele Tufano, Cody Watson, Gabriele Bavota, Massimiliano Di Penta, Martin
White, and Denys Poshyvanyk. 2019.

 
 An empirical study on learning bug-fixing patches in the wild via
neural machine translation.

 
 ACM Transactions on Software Engineering and Methodology
(TOSEM) , 28:1 – 29.

 

 
 Tunstall et al. (2022) 
 
L. Tunstall, L. von Werra, and T. Wolf. 2022.

 
 Natural Language Processing with Transformers: Building Language
Applications with Hugging Face .

 
 O’Reilly Media.

 

 
 Vaswani et al. (2017) 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.

 
 Attention is all you need.

 
 In NIPS , pages 5998–6008.

 

 
 Wang et al. (2021a) 
 
Deze Wang, Yue Yu, Shanshan Li, Wei Dong, Ji Wang, and Qing Liao.
2021a.

 
 Mulcode: A multi-task learning approach for source code
understanding.

 
 2021 IEEE International Conference on Software Analysis,
Evolution and Reengineering (SANER) , pages 48–59.

 

 
 Wang et al. (2021b) 
 
Yue Wang, Weishi Wang, Shafiq R. Joty, and Steven C. H. Hoi.
2021b.

 
 Codet5: Identifier-aware unified pre-trained encoder-decoder models
for code understanding and generation.

 
 arXiv .

 

 
 Wu et al. (2022a) 
 
Ruoting Wu, Yu xin Zhang, Qibiao Peng, Liang Chen, and Zibin Zheng.
2022a.

 
 A survey of deep learning models for structural code understanding.

 
 ArXiv , abs/2205.01293.

 

 
 Wu et al. (2022b) 
 
Ruoting Wu, Yu xin Zhang, Qibiao Peng, Liang Chen, and Zibin Zheng.
2022b.

 
 A survey of deep learning models for structural code understanding.

 
 ArXiv , abs/2205.01293.

 

 
 Xu et al. (2022) 
 
Frank F. Xu, Uri Alon, Graham Neubig, and Vincent J. Hellendoorn. 2022.

 
 A systematic evaluation of large language models of code.

 
 ArXiv , abs/2202.13169.

 

 
 Yin and Neubig (2017) 
 
Pengcheng Yin and Graham Neubig. 2017.

 
 A syntactic neural model for general-purpose code generation.

 
 In ACL , pages 440–450.

 

 
 Zheng et al. (2021) 
 
Yunhui Zheng, Saurabh Pujar, Burn Lewis, Luca Buratti, Edward Epstein, Bo Yang,
Jim Laredo, Alessandro Morari, and Zhonglai Su. 2021.

 
 D2a: A dataset built for ai-based vulnerability detection methods
using differential analysis.

 
 2021 IEEE/ACM 43rd International Conference on Software
Engineering: Software Engineering in Practice (ICSE-SEIP) , pages 111–120.

 

 
 Zhou et al. (2019) 
 
Yaqin Zhou, Shangqing Liu, Jing Kai Siow, Xiaoning Du, and Yang Liu. 2019.

 
 Devign: Effective vulnerability identification by learning
comprehensive program semantics via graph neural networks.

 
 In NeurIPS , pages 10197–10207.