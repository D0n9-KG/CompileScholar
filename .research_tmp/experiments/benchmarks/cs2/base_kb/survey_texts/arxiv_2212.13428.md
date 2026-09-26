A Survey on Knowledge-Enhanced Pre-trained Language Models 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.13428v1 [cs.CL] 27 Dec 2022 
 
 

# A Survey on Knowledge-Enhanced Pre-trained Language Models

 
 
 Chaoqi Zhen
 
    
 Yanlei Shang
 † † thanks: 
Chaoqi Zhen, Yanlei Shang, Xiangyu Liu, Yifei Li, and Yong Chen are with the State Key Laboratory of Networking and Switching Technology, School of Computer Science (National Pilot Software Engineering School), Beijing University of Posts and Telecommunications, Beijing, 100876, China.
Dell Zhang is with Thomson Reuters Labs, London, UK.
Yanlei Shang and Yong Chen are the corresponding authors.
 E-mail: {shangyl, yong.chen}@bupt.edu.cn
 
    
 Xiangyu Liu
 
    
 Yifei Li
 
    
 Yong Chen
 † † thanks: 
Chaoqi Zhen, Yanlei Shang, Xiangyu Liu, Yifei Li, and Yong Chen are with the State Key Laboratory of Networking and Switching Technology, School of Computer Science (National Pilot Software Engineering School), Beijing University of Posts and Telecommunications, Beijing, 100876, China.
Dell Zhang is with Thomson Reuters Labs, London, UK.
Yanlei Shang and Yong Chen are the corresponding authors.
 E-mail: {shangyl, yong.chen}@bupt.edu.cn
 
    
 Dell Zhang
 † † thanks: Manuscript received Dec. 27, 2022; revised May **, ****. 

 Abstract 
 
 Natural Language Processing (NLP) has been revolutionized by the use of Pre-trained Language Models (PLMs) such as BERT.
Despite setting new records in nearly every NLP task, PLMs still face a number of challenges including poor interpretability, weak reasoning capability, and the need for a lot of expensive annotated data when applied to downstream tasks.
By integrating external knowledge into PLMs, K nowledge- E nhanced P re-trained L anguage M odels (KEPLMs) have the potential to overcome the above-mentioned limitations.
In this paper, we examine KEPLMs systematically through a series of studies.
Specifically, we outline the common types and different formats of knowledge to be integrated into KEPLMs, detail the existing methods for building and evaluating KEPLMS, present the applications of KEPLMs in downstream tasks, and discuss the future research directions.
Researchers will benefit from this survey by gaining a quick and comprehensive overview of the latest developments in this field.

 
 
 
 Index Terms:  Natural Language Processing, Pre-trained Language Models, Knowledge Bases, Memory Mechanism, Interpretability.

 
 

## I Introduction 

 
 Pre-trained language models (PLMs) are first trained on a large dataset and then directly transferred to downstream tasks, or further fine-tuned on another small dataset for specific NLP tasks.
Early PLMs, such as Skip-Gram  [ 1 ] and GloVe  [ 2 ] , are shallow neural networks, and their word embeddings (learned from window-sized contexts) are static semantic vectors, which makes them unable to deal with the problems of polysemy in dynamic environments.
With the development of deep learning, researchers have tried to leverage deep neural networks to boost tasks’ performances with dynamic semantic embeddings.
At first, people were still limited to the paradigm of supervised learning and thought without enough labeled data it would be difficult to unleash the potential of deep learning.
However, with the emergence of self-supervised learning, big language models such as BERT  [ 3 ] can learn a lot of knowledge from large-scale unlabeled text data by predicting tokens that have been covered up in advance.
Thus they have made breakthrough progress in a number of downstream NLP tasks.
Since then, many large models have started to adopt Transformer  [ 4 ] structures and self-supervised learning to solve NLP problems, and gradually PLMs have entered a phase of rapid development.
The latest phenomenal success for PLMs is OpenAI’s ChatGPT 1 1 
 1 
 
 
 
 
 
 
 https://chat.openai.com/chat .

 
 
 As research has progressed, it has been found that PLMs still struggle with poor interpretability, weakness in robustness, and a lack of reasoning ability.
Specifically, PLMs are widely recognized as black boxes whose decision process is opaque, thus making them difficult to interpret.
Additionally, PLMs may not be sufficiently robust as deep neural models are susceptible to adversarial examples.
Furthermore, PLMs are also limited in their reasoning abilities because they are purely data-driven.
All these shortcomings of PLMs can be improved by incorporating external knowledge, which leads to what we call Knowledge-Enhanced Pre-trained Language Models (KEPLMs).
Fig.  1 shows the advantages of KEPLMs in the words of ChatGPT.

 
 
 Fig. 1: The benefits of integrating external knowledge into PLMs, according to ChatGPT — one of the largest PLMs today. 
 
 
 Although there exist a few overviews or surveys of KEPLMs  [ 5 , 6 , 7 , 8 ] , this research field is growing and expanding rapidly with many new techniques emerged.
This survey aims to provide AI researchers the most comprehensive and up-to-date picture about the latest advancements in KEPLMs from different perspectives.

 
 
 The rest of this survey are organized as follows.
 Section   II explains the background of KEPLMs.
 Section   III categorizes the commonly used types and formats of knowledge for KEPLMs.
 Section   IV presents the different approaches to building KEPLMs.
 Section   V describes the possible performance metrics for evaluating KEPLMs.
 Section   VI discusses the typical applications of KEPLMs in downstream knowledge-intensive NLP tasks.
 Section   VII outlines the future research directions of KEPLMs.
 Section   VIII summarizes the contributions.

 
 
 TABLE I: Commonly Used Commonsense Knowledge 
 
 
 Knowledge Base | 
 
 
 Domain 
 | 
 
 
 Model 
 | 

 
 ConceptNet  [ 9 ] | 
 
 
 open-domain 
 | 
 
 
 KERM  [ 10 ] , Zhang et al.  [ 11 ] , QA-GNN  [ 12 ] , GreaseLM  [ 13 ] , JointLK  [ 14 ] , GLM  [ 15 ] , KagNet  [ 16 ] , KG-BART  [ 17 ] , COMET  [ 18 ] , AMS  [ 19 ] , GRF  [ 20 ] , ExBERT  [ 21 ] , Lauscher et al.  [ 22 ] , Guan et al.  [ 23 ] , Chang et al.  [ 24 ] , Yang et al.  [ 25 ] 
 | 

 
 ATOMIC  [ 26 ] | 
 
 
 social-interaction 
 | 
 
 
 Guan et al.  [ 23 ] , Mitra et al.  [ 27 ] 
 | 

 
 ATOMIC 20 20 \text{ATOMIC}_{20}^{20}   [ 28 ] | 
 
 
 social-interaction, event-centered, physical 
 | 
 
 
 Hosseini et al.  [ 29 ] 
 | 

 
 ASER  [ 30 ] | 
 
 
 eventuality 
 | 
 
 
 CoCoLM  [ 31 ] 
 | 

 
 
 TABLE II: Commonly Used Domain Knowledge 
 
 
 Domain | 
 
 
 Model 
 | 

 
 Science (Biomedical) | 
 
 
 BERT-MK  [ 32 ] , UmlsBERT  [ 33 ] , SMedBERT  [ 34 ] , KeBioLM  [ 35 ] , BioBERT  [ 36 ] , Med-BERT  [ 37 ] 
 | 

 
 Science (Other) | 
 
 
 SciBERT  [ 38 ] , MatBERT  [ 39 ] 
 | 

 
 E-commerce | 
 
 
 E-BERT  [ 40 ] , K-AID  [ 41 ] , K-PLUG  [ 42 ] 
 | 

 
 Law | 
 
 
 Legal-BERT  [ 43 , 44 ] , Lawformer  [ 45 ] 
 | 

 
 Sentiment | 
 
 
 KET  [ 46 ] , SKEP  [ 47 ] , REMOTE  [ 48 ] 
 | 

 
 Programming | 
 
 
 GraphCodeBERT  [ 49 ] 
 | 

 
 
 

## II Background 

 
 In this section, we first introduce the concept of PLMs, and then talk about the recent trend of combining PLMs and knowledge.

 
 

### II-A Pre-trained Language Models 

 
 In 2013, word2vec  [ 50 ] opened the era of pre-trained language models.
First-generation PLMs such as Skip-Gram  [ 1 ] and GloVe  [ 2 ] aim to get good word embeddings for downstream tasks directly, and their model architectures are typically shallow neural networks to allow for computational efficiency.  [ 51 ] .
Second-generation PLMs, e.g., LSTM  [ 52 ] based CoVe  [ 53 ] and ELMo  [ 54 ] as well as Transformer  [ 4 ] based BERT  [ 3 ] and GPT  [ 55 ] focus on learning word embeddings in dynamic contexts.
During that period, Transformers became most successful in almost all downstream NLP tasks and brought significant changes to the NLP field.
Today, PLMs generally refer to models based on the Transformer architecture, under the pre-train-then-fine-tune paradigm.
The representative PLMs include GPT  [ 55 ] (an auto-regressive language model based on Transformer Decoder), BERT  [ 3 ] (an auto-encoding language model based on Transformer Encoder), and BART  [ 56 ] (a sequence-to-sequence model based on both Transformer Encoder and Decoder).
Very recently, prompt learning, a new paradigm in NLP, is getting more and more popular  [ 57 ] .
It can help to make better use of knowledge in PLMs and hence empower PLMs with the ability to perform few-shot or even zero-shot learning for challenging scenarios with little or none labeled data.

 
 
 

### II-B Knowledge and PLMs 

 
 There are two lines of research on the interaction between knowledge and PLMs: one is to use PLMs as Knowledge Bases (KBs), and the other is to enhance PLMs with knowledge.
In this paper, we focus on the latter.

 
 

#### II-B 1 Using PLMs as Knowledge Bases

 
 KBs (e.g., Wikidata  [ 58 ] and ATOMIC  [ 26 ] ) store entities and their relationships, usually in the form of relation triplets.
PLMs are considered as a possible alternative to structured KBs, which has attracted many researchers.
Beginning with LAMA  [ 59 ] , many researchers explored whether PLMs could serve as structured KBs.
Wan et al.  [ 60 ] explored how to construct KBs using pre-trained language models automatically.
Heinzerling et al.  [ 61 ] investigated the relationship between accuracy and memory capacity of neural networks, arguing that PLMs can be used as KBs.
Safavi et al.  [ 62 ] argued that relational KBs represent knowledge with high accuracy but lack flexibility.
In contrast, Razniewski et al.  [ 63 ] dived into the strengths and limitations of both PLMs and KBs.
They believed that KBs with explicit knowledge could not be completely replaced by PLMs with latent knowledge.
Wang et al.  [ 64 ] found that closed-book question answering is still a challenge for generative models, and therefore generative models are not suitable to serve as KBs.
AlKhamissi et al.  [ 65 ] argued that there are five aspects at which a PLM needs to excel to qualify as a KB and found that three of them (i.e. consistency, reasoning, and interpretability) are better obtained in KBs than in PLMs.

 
 
 

#### II-B 2 Enhancing PLMs with Knowledge

 
 On the other way around, we can use knowledge to improve or extend PLMs.
In many knowledge-intensive downstream tasks, taking question-answering tasks as an example, the amount of knowledge learned by the pre-trained language model can be increased by adding parameters; however, it is far less effective than directly integrating knowledge  [ 66 ] .
Therefore, it is necessary to inject knowledge into PLMs to obtain better performance.

 
 
 Methods such as ERNIE  [ 67 ] , KnowBert  [ 68 ] , K-BERT  [ 69 ] are early attempts at incorporating knowledge into PLMs, and they have achieved great success especially on knowledge-intensive NLP tasks.
Many subsequent models were inspired by them and improved upon them.
Nowadays, more and more KEPLMs are emerging, integrating different kinds of knowledge in different ways and dealing with a variety of NLP tasks.
In what follows, we will present a comprehensive overview of KEPLMs.

 
 
 
 
 

## III Knowledge Sources for KEPLMs 

 
 In this section, we elaborate on the common types and formats of knowledge that are incorporated into PLMs.

 
 

### III-A Types of Knowledge 

 
 There are five types of knowledge that are often integrated into PLMs: linguistic knowledge, semantic knowledge, commonsense knowledge, encyclopedic knowledge, and domain knowledge.

 
 

#### III-A 1 Linguistic Knowledge

 
 Part-of-Speech Tags. 
Commonly used part-of-speech tags include pronouns, verbs, nouns, pre-positions, conjunctions, adverbs, and adjectives.
The could help the understanding of natural language text data, e.g., for sentiment analysis.
SentiLARE  [ 70 ] exploits part-of-speech tags to promote sentiment analysis.

 
 
 Syntactic Structures. 
Models acquire the structure of sentences via syntactic parsing, mainly including constituency and dependency.
Syntax-BERT  [ 71 ] uses syntax-related masks to incorporate information from constituency and dependency trees.
K-Adapter  [ 72 ] integrates dependency parsing information, which has improved the performance of the dependency relation prediction task.

 
 
 Cross-lingual Transferability. 
Sometimes PLMs could obtain cross-lingual transferability through learning from multilingual corpora.
For example, as demonstrated by XLM-K  [ 73 ] , the linguistic knowledge about one language might help the processing of another language.

 
 
 

#### III-A 2 Semantic Knowledge

 
 Semantic knowledge aims to help models catch the meaning of texts. For example,
KT-NET  [ 74 ] , SenseBERT  [ 75 ] , and LIBERT  [ 76 ] introduce semantic knowledge from WordNet  [ 77 ] and perform well on machine reading comprehension, word sense disambiguation, and lexical simplification, respectively.

 
 
 Basu et al.  [ 78 ] converted the syntax tree of the text to its corresponding semantic meanings with the help of VerbNet  [ 79 ] so that the model could understand the text.
SemBERT  [ 80 ] incorporates semantic knowledge from semantic role labeling for better reading comprehension and language inference.

 
 
 

#### III-A 3 Commonsense Knowledge

 
 Commonsense knowledge is the routine knowledge people have of their everyday world and activities  [ 81 ] . Commonly used knowledge bases are listed in Table  I .

 
 
 Commonsense knowledge is represented as triples in KBs, where head and tail entities are more often phrases than just words, which are different from encyclopedic knowledge.
For example, a commonsense triple is like (having no food, CauseDesire, go to a store) , while an encyclopedic triple is like (China, capital, Beijing) .

 
 
 As shown in Table 1, ConceptNet  [ 9 ] is the most widely used open-domain commonsense knowledge graph containing 34 relations, such as RelatedTo, IsA, Causes, etc.
It’s helpful in commonsense question answering, commonsense validation, and commonsense story generation.

 
 
 ATOMIC  [ 26 ] focuses on inferential knowledge organized by “if-then” structure, e.g., “ if X pays Y a compliment, then Y will likely return the compliment ”, covering causes vs. effects , agents vs. themes , voluntary vs. involuntary events , and actions vs. mental states , which is helpful for commonsense reasoning.
It can be employed in commonsense question generation and commonsense question-answering tasks, such as Guan et al.  [ 23 ] and Mitra et al.  [ 27 ] .

 
 
 ATOMIC 20 20 \text{ATOMIC}_{20}^{20}   [ 28 ] covers more accurate and diverse commonsense knowledge than the aforementioned commonsense knowledge sources, including three categories of social interaction, physical and event-centered.
Hosseini et al.  [ 29 ] convert triples (in ATOMIC 20 20 \text{ATOMIC}_{20}^{20}   [ 28 ] ) into natural language sentences for pre-training, which improves the performance on causal pair classification and commonsense of answering questions tasks.

 
 
 ASER  [ 30 ] is a large-scale eventuality knowledge graph, with events as nodes and discourse relations as edges.
It provides more complicated commonsense knowledge, such as the cause-effect relation between “Jim yells at Bob” and “Bob is upset” .

 
 
 

#### III-A 4 Encyclopedic Knowledge

 
 Encyclopedic knowledge covers widespread information in the open domain, in the form of texts or triples.
Wikipedia 2 2 
 2 
 
 
 
 
 
 
 https://www.wikipedia.org/ is a multilingual encyclopedia that is unstructured.
BERT utilizes Wikipedia as pre-training data to learn contextual representations; while other methods usually leverage encyclopedic knowledge via knowledge triples.

 
 
 Wikidata  [ 58 ] is the most widely used knowledge graph when incorporating encyclopedic knowledge.
KEPLMs such as K-Adapter  [ 71 ] , ERNIE  [ 67 ] , KgPLM  [ 82 ] , and ERICA  [ 83 ] , use Wikidata  [ 58 ] as knowledge sources.
Other commonly used English encyclopedic knowledge graphs include Freebase  [ 84 ] , DBpedia  [ 85 ] , and NELL  [ 86 ] .
CN-DBpedia  [ 87 ] is a widely used Chinese encyclopedic knowledge graph.
KEPLMs designed for Chinese downstream tasks, such as K-BERT  [ 69 ] , use CN-DBpedia as knowledge sources.

 
 
 Wikidata5M is a large-scale knowledge graph proposed by KEPLER  [ 88 ] that contains high-quality descriptions of entities and relations in addition to triples. KEPLER  [ 88 ] used these descriptions to initialize knowledge embeddings, and CoLAKE  [ 89 ] also adopted this approach.

 
 
 

#### III-A 5 Domain Knowledge

 
 In contrast to encyclopedic knowledge, domain knowledge is knowledge of a specific, specialized field discipline, such as biomedical, e-commerce, and sentiment, which are explored a lot, as shown in Table  II .

 
 
 Biomedical knowledge is usually represented as triples containing symptoms or diseases as head or tail entities, e.g., (bacterial pneumonia, with associated morphology, inflammation) .
E-commerce knowledge is formed with product names, while their descriptions are represented by a set of phrases.
For example, the product “iPhone XS” is described as “iOS; 4G signal; T-Mobile service; OLED screen; …”.
Sentiment knowledge could be represented in many ways, including sentiment words, word polarity, etc.

 
 
 
 

### III-B Formats of Knowledge 

 
 There are four formats of knowledge that are often incorporated into PLMs, i.e., entity lexicon, knowledge graph, plain text, and labeled images.

 
 

#### III-B 1 Entity Lexicon

 
 To incorporate knowledge through entities, we need to integrate knowledge embeddings of entities into aligned token embeddings.
Existing models propose two methods to obtain initial entity embeddings, obtained through traditional knowledge embedding algorithms, such as TransE used by ERNIE  [ 67 ] and CokeBERT  [ 90 ] , or through encoding entity descriptions, such as KEPLER  [ 88 ] .

 
 
 The first approach can fuse the information of neighboring nodes of entities in the knowledge base. But it must face the challenge of heterogeneous embedding space because the embedding vector space of words in the text and entities in KG is inconsistent. The second approach fuses the information in the same embedding space, but the entity embedding may not fully express the meaning of the entities.

 
 
 It’s simple and intuitive to inject knowledge in the form of entity embeddings.
However, entity embeddings need to be retrained when the knowledge graph is updated, and the model parameters have to be retrained if injected in the pre-training phase.

 
 
 

#### III-B 2 Knowledge Graph

 
 Triples .
Knowledge stored in the knowledge graphs is commonly in the form of semantic (RDF) triples.
To incorporate triples, we can append triples to the proper position in the text, such as K-BERT  [ 69 ] , ERNIE 3.0  [ 91 ] , Zhang et al.  [ 11 ] , and Bian et al.  [ 92 ] , or integrate their embeddings into text embeddings, such as Liu et al.  [ 93 ] .
More details are in Section   IV-B .

 
 
 Subgraphs .
Knowledge subgraphs are part of knowledge graphs that take entities as nodes and relationships as edges.
KEPLMs such as QA-GNN  [ 12 ] , GreaseLM  [ 13 ] , KG-BART  [ 17 ] , and KALA  [ 94 ] incorporated knowledge in the form of knowledge subgraphs, which are described in detail in Section   IV-B2 .

 
 
 

#### III-B 3 Plain Text

 
 To integrate knowledge in texts, we can convert knowledge triples to sentences as the pre-training corpus or add related entity definitions to texts.
Commonsense knowledge triples are suitable to be converted into sentences.
or instance, Hosseini et al.  [ 29 ] convert the triples (PersonX accidentally fell, xEffect, PersonX breaks an arm) in ATOMIC 20 20 \text{ATOMIC}_{20}^{20} into the sentence “Tracy accidentally fell. As a result, Tracy breaks an arm.”, and fed them to the model for continually pre-training.
This method is suitable for commonsense knowledge bases because the triples in them are usually phrases and only need to add conjunctions to obtain sentences.

 
 
 Dict-BERT  [ 95 ] append the definitions of the rare words to the end of the text as model input, which facilitates the model’s understanding and learning of rare words, but does not apply to polysemous words.

 
 
 

#### III-B 4 Captioned Images

 
 Unlike the knowledge presented above in the form of texts, visual knowledge refers to knowledge observed through the eyes, including the shape, size, and color of objects, which may not be mentioned in the text and need to be learned through images.
To incorporate visual knowledge, models can first retrieve context-related images, encode them, and then integrate the embeddings of images to text embeddings, such as VALM  [ 96 ] , as shown in Fig.  2 .
Visual knowledge can also be incorporated through text-image alignment pre-training objectives, such as Vokenization  [ 97 ] .

 
 
 Fig. 2: Incorporating visual knowledge from captioned images into PLMs. 
 
 
 
 
 

## IV Building KEPLMs 

 
 When we construct KEPLMs, external knowledge could be incorporated into PLMs implicitly and/or explicitly .

 
 

### IV-A Implicit Incorporation of Knowledge 

 

#### IV-A 1 Knowledge-Guided Masking Strategies

 
 PLMs represented by BERT generally use unstructured text documents from Wikipedia etc. as the corpus for pre-training.
The unstructured text data contain rich contextual semantic information from which BERT could learn the contextual knowledge of words through Masked Language Modelling (MLM).
However, entities and phrases in the text that also contain valuable information have been ignored.
By employing a knowledge-guided masking strategy beyond the level of individual words, PLMs are able to incorporate the knowledge about entities and phrases etc., as shown in Fig.   3 .

 
 
 Fig. 3: Using knowledge-guided masking strategies to build KEPLMs. 
 
 
 ERNIE  [ 98 ] adds entity-level and phrase-level masking strategies to BERT, and thus guides the pre-training of BERT to incorporate the entity and phrase information from text.
SKEP  [ 47 ] proposes to mask not entities or phrases but sentiment words so as to inject sentiment knowledge into text representations.

 
 
 Different from the simple random selection of entities or phrases for masking (as in ERNIE), GLM  [ 15 ] uses knowledge-graph informed sampling that assigns higher weights to more important entities.
Specifically, GLM’s masking strategy would mask a general word 20 % 20\% of the time and an entity 80 % 80\% of the time.
When GLM needs to mask an entity, those entities which can reach other entities in the sentence within a specific number of hops in ConceptNet  [ 9 ] are considered more critical and given higher probabilities to be chosen.
In this way, GLM can nudge the construction of KEPLMs towards more critical entities in the knowledge graph.
As illustrated in Fig.   4 , assuming that among the four entities in the given sentence, three of them “sick”, “baby” and “cry” could be reached within a specific number of hops in ConceptNet while the other one “sometimes” could not, GLM would give the former more chances than the latter to be masked for pre-training when entity-level masking is activated ; the other non-entity words in the sentence would be sampled only when word-level masking is activated.

 
 
 Fig. 4: GLM’s knowledge-graph informed sampling of entities for masking. 
 
 
 Instead of using a predefined probability to choose between the two modes of masking (as in GLM), E-BERT  [ 40 ] proposes an adaptive hybrid masking strategy that allows the model to switch between word-level and phrase-level masking in an adaptive fashion during its pre-training.
As illustrated in Fig.   5 , E-BERT  [ 40 ] enters the mode of word-level masking when r α t r \alpha^{t} and the mode of phrase-masking otherwise, where r r is a randomly generated number in each iteration.
The loss functions ℒ w \mathcal{L}_{w} and ℒ p \mathcal{L}_{p} of the two modes in each iteration are used to track the fitting progress of the learned word-level information and the learned phrase-level information, represented by η w t \eta_{w\ }^{t} and η p t {\ \eta}_{p\ }^{t} , respectively.
The relative importance of word-level masking with respect to phrase-level masking, r t r^{t} , is used further to calculate α t + 1 \alpha^{t+1} as in Eq.   1 , so that the mode with higher loss in the current iteration is more likely to be selected in the next iteration.

 
 
 
 | 
 | 
 η w t = Δ w t , t − 1 / Δ w t , 1 = [ ℒ w t − 1 − ℒ w t ] + / ( ℒ w 1 − ℒ w t ) , \displaystyle\eta_{w}^{t}=\mathrm{\Delta}_{w}^{t,t-1}/\mathrm{\Delta}_{w}^{t,1}=\left[\mathcal{L}_{w}^{t-1}-\mathcal{L}_{w}^{t}\right]_{+}\ /\ (\mathcal{L}_{w}^{1}-\mathcal{L}_{w}^{t}), | 
 | 
 (1) | 

 
 | 
 | 
 η p t = Δ p t , t − 1 / Δ p t , 1 = [ ℒ p t − 1 − ℒ p t ] + / ( ℒ p 1 − ℒ p t ) , \displaystyle\eta_{p}^{t}=\mathrm{\Delta}_{p}^{t,t-1}/\mathrm{\Delta}_{p}^{t,1}=\left[\mathcal{L}_{p}^{t-1}-\mathcal{L}_{p}^{t}\right]_{+}\ /\ (\mathcal{L}_{p}^{1}-\mathcal{L}_{p}^{t}), | 
 | 

 
 | 
 | 
 r t = η w t + 1 / η p t + 1 , \displaystyle r^{t}=\eta_{w}^{t+1}/\eta_{p}^{t+1}, | 
 | 

 
 | 
 | 
 α t + 1 = t ​ a ​ n ​ h ​ ( r t ) . \displaystyle\alpha^{t+1}=tanh\left(r^{t}\right). | 
 | 
 

 Thus, E-BERT can switch between the two modes of masking adaptively and strike a balance between them.

 
 
 Fig. 5: E-BERT’s adaptive hybrid masking strategy. 
 
 
 

#### IV-A 2 Knowledge-Related Pre-training Tasks

 
 Some methods for building KEPLMs incorporate knowledge implicitly by adding knowledge-related pre-training tasks, as shown in Fig.   6 .

 
 
 Fig. 6: Using knowledge-related pre-training tasks to build KEPLMs. 
 
 
 For example, KALM  [ 99 ] enriches the input sequence with entity signals and then adds an entity prediction task to the pre-training objective in order to help the model learn entity information better.
KEPLER  [ 88 ] adds the knowledge embedding pre-training task which shares a Transformer Encoder with the MLM, obtaining text-enhanced knowledge embeddings and knowledge-enhanced PLMs simultaneously.
Vokenization  [ 97 ] proposes the concept of voken (visualized token), i.e., token-related images; it adds a voken classification task that predicts the image corresponding to each token so as to enhance the PLMs with visual knowledge which has been shown to help some downstream NLP tasks.

 
 
 
 

### IV-B Explicit Incorporation of Knowledge 

 
 Fig. 7: Explicit incorporation of knowledge into PLMs via modifying the model input or adding knowledge fusion modules. 
 
 
 There are mainly three ways for PLMs to incorporate external knowledge explicitly: modifying the model input, adding knowledge fusion modules, and utilizing external memory.
The first two approaches insert relevant knowledge into PLMs, in the form of either additional input for the model or additional components in the model, as shown in Fig.   7 ① and ②.
The third approach keeps the text and knowledge spaces independent which can facilitate knowledge updates.

 
 

#### IV-B 1 Modifying the Model Input

 
 Some KEPLMs insert relevant knowledge triples or entity descriptions into the input for the model during its pre-training.

 
 
 There exist a few different ways to incorporate knowledge in the form of triples.
ERNIE 3.0  [ 91 ] prepends related triples to the sentences as the expanded model input.
K-BERT  [ 69 ] injects relevant triples into each sentence to generate a sentence tree for model input.
To be specific, if the input sentence has an entity “apple”, K-BERT  [ 69 ] will find the triples whose head entity is “apple” in the knowledge graph and then append the relation and tail entity of these triples to “apple” to generate a new sentence tree.
A visible matrix is created to control the level of knowledge noise.
Zhang et al.  [ 11 ] improve the visible matrix of K-BERT to further minimize the introduction of knowledge noise.
CoLAKE  [ 89 ] also introduces triples to the input text, treats the text as a fully connected word graph, and integrates knowledge to form a word-knowledge graph.
It takes inspiration from K-BERT and makes some improvements in the reduction of knowledge noise.
For the question answering task, Bian et al.  [ 92 ] convert multiple question-related knowledge triples into text according to predefined templates and feed them into the model together with the question and alternative answers for training, which obtains excellent performance on commonsense question answering.
For all the above methods that insert knowledge triples to the model input, the introduction of external knowledge may damage the original sentence structure, and therefore we must try to reduce knowledge noise in this process.

 
 
 There are also a few different ways to incorporate knowledge in the form of entities.
Dict-BERT  [ 95 ] obtains the definitions of rare words in a sentence from Wiktionary  [ 100 ] and appends them to the end of the sentence.
Similarly, DKPLM  [ 101 ] focuses on long-tail entities and uses pseudo token representations from relevant triples to replace their embeddings.
Unlike the above methods, WKLM  [ 102 ] replaces entities in the text with other entities of the same type, which are then fed into the model.
Then the model is asked to determine which entities in the sentence are correct and which are replaced.
This method does not modify the model, only the input data during its pre-training.
A few high-performance KEPLMs constructed using this method are described in detail below.

 
 
 Fig. 8: Adding knowledge triples into the model input. 
 
 
 CoLAKE  [ 89 ] modifies the model input to incorporate knowledge of entities, as shown in Fig.   8 .
Specifically, CoLAKE regards each input sentence as a fully connected graph.
It takes the entity in the input sentence as the anchor node and introduces a subgraph (composed of triples with that anchor node as the head entity in the knowledge graph) to obtain the word knowledge graph.
Then the newly added nodes from the word knowledge graph are appended behind the original input text and fed into the PLMs together for pre-training.
CoLAKE distinguishes the node types in the newly obtained input statement and initializes different nodes differently.
These nodes include word nodes, entity nodes, and relation nodes.
CoLAKE achieves a 5.2% improvement on the relation classification task in comparison to BERT without knowledge integration.

 
 
 Fig. 9: Replacing the embeddings of long-tail entities with pseudo token embeddings. 
 
 
 DKPLM  [ 101 ] proposes the concept of long-tail entities which represent the entities not been fully learned by the model from the corpus.
Strengthening the learning of such long-tail entities in the pre-training stage can enhance the model’s understanding of semantic context and eventually the language representation.
For this purpose, a measurement method KLT has been proposed to identify long-tail entities: the entities with a KLT score below the average in each sentence are regarded as the long-tail entities of that sentence.
The KLT score of an entity e e is calculated as

 
 
 
 | 
 K ​ L ​ T ​ ( e ) = 𝕀 F ​ r ​ e ​ q ​ ( e ) R f ​ r ​ e ​ q ⋅ S ​ I ​ ( e ) ⋅ K ​ C ​ ( e ) , \displaystyle KLT\left(e\right)=\mathbb{I}_{Freq\left(e\right) R_{freq}}\cdot SI\left(e\right)\cdot KC\left(e\right), | 
 | 
 (2) | 
 

 where the three terms in the equation represent the occurrence frequency of the entity in the corpus, the semantic importance, and the number of neighboring nodes within a certain number of hops in KG, respectively.
As illustrated in Fig.   9 , DKPLM replaces the embeddings of long-tail entities detected in the text with pseudo token embedding as new input to the model.
For example, suppose that the input sentence is “Yao, was selected to start for the Western Conference in the NBA All-Star Game eight times” where “Western Conference” and “All-Star Game” have been identified as long-tail entities, so the embeddings of them will be replaced by pseudo token embeddings shown as “[LTE]”.
A pseudo token embedding is encoded by related triples in the knowledge graph and the entity’s description in a certain way.
The F 1 F_{1} scores of DKPLM  [ 101 ] on entity classification and relation classification are 2.1% and 2.87% higher than RoBERTa  [ 103 ] respectively, confirming that knowledge about long-tail entities could be incorporated into PLMs to obtain better language representation and higher model performance.

 
 
 

#### IV-B 2 Adding Knowledge Fusion Modules

 
 Different from the methods introduced in Section   IV-B1 , the methods presented in this section all involve the fusion of different modal spaces.
Specifically, the text and knowledge modalities are encoded differently, and additional modules are constructed for inter-modal fusion.
As illustrated in Fig.   10 , such knowledge fusion modules mainly appear in three positions:

 
 (a) 
 
 on top of the entire PLM,

 

 (b) 
 
 between the Transformer layers of PLM,

 

 (c) 
 
 inside the Transformer layers of PLM.

 

 
 
 
 Fig. 10: Three different ways to add knowledge fusion modules to a PLM: (a) on top of the entire PLM, (b) between the Transformer layers of PLM, and (c) inside the Transformer layers of PLM. 
 
 
 The method shown in Fig.   10 (a) can be further divided into two categories.
One is the T-K structure represented by ERNIE  [ 67 ] which mainly incorporates knowledge in the form of entity embeddings:
a T-Encoder is followed by a K- Encoder, where T- Encoder encodes the text corpus and K-Encoder integrates the entity embeddings in the knowledge space into the entity embeddings in the text space.
Many KEPLMs follow this structure but differ in how they get entity embeddings.
The entity embedding in ERNIE is obtained by TransE which takes a single triple as a training sample and does not contain the information of that entity’s neighbor nodes.
Developed on top of this architecture, BERT-MK  [ 32 ] fully considers the information of neighbor nodes when learning the entity embedding in the knowledge space, incorporating more semantic information.
CokeBERT  [ 90 ] found that the entity embeddings in the former method cannot change dynamically according to the textual context.
To overcome this limitation, the closer the meaning of the neighbor node is to the text, the more its information will be incorporated into the entity embedding by CokeBERT.

 
 
 The second class of methods attaches other knowledge fusion structures after the PLM. Some KEPLMs use the attention mechanism to fuse the information in the text and knowledge modalities. Kwon et al. exploited an attention mechanism to incorporate sentence-related triples into textual embedding representations  [ 104 ] . JointLK  [ 14 ] lets each question token attend on KG nodes and each KG node attend on question tokens, and the two modal representations fuse and update mutually by multi-step interactions. KET  [ 46 ] adopts a hierarchical self-attention mechanism to incorporate sentiment knowledge into text representations. Besides, Liu et al.  [ 93 ] encode the relevant triples in the context and then fused them with the embeddings of the text using a gate mechanism.
There are other works based on interaction nodes. Both modalities exchange information through interaction nodes. QA-GNN  [ 12 ] incorporated information from the text space into the knowledge space through interaction nodes and achieved good results in commonsense question answering. Inspired by this, GreaseLM  [ 13 ] set up interaction nodes in both modalities to learn the knowledge of that modality separately and then exchange information at the fusion layer to learn the knowledge of the other modality, as shown in Fig.   11 .

 
 
 Fig. 11: The interaction nodes for text-knowledge information fusion  [ 13 ] . 
 
 
 The approach represented in Fig.   10 (b) is to add a knowledge fusion module between the Transformer layers of PLM.
KnowBERT  [ 68 ] adds new modules between Transformer Encoder blocks to incorporate knowledge about entities in sentences.
It considers the problem of polysemy that is ignored by ERNIE  [ 67 ] : for an entity that exhibits different semantic meanings in different contexts, the knowledge about it is incorporated according to its specific meaning.
KG-BART  [ 17 ] added knowledge fusion modules between the Encoder and Decoder layers to integrate information from knowledge subgraphs into the textual representation through a multi-headed graph attention mechanism.
JAKET  [ 105 ] divides the pre-trained language model into the first six layers and the last six layers.
After the text passes through the first six layers of the encoder, the hidden layer representation is obtained, and so is the entity embedding representation.
At each entity position in the text, the corresponding entity embedding representation is added and then input to the last six layers of the model for subsequent training.
The knowledge space and the text space can cyclically reinforce each other for the learning of better representations.

 
 
 The approach represented in Fig.   10 (c) is to add a fusion module inside the Transformer Layer.
For example, KALA  [ 94 ] inserts the knowledge fusion module inside the Transformer block layer, which is inspired by the idea of modulation, i.e., to modulate the embeddings in the text space with the knowledge in the knowledge space.
Adding knowledge fusion modules in this way is intuitive, and the incorporated knowledge is mainly entity representation.
Some methods consider the context of entities in the knowledge graph, e.g., BERT-MK  [ 32 ] ; some others filter entity neighbor nodes for embedding based on text context, e.g., CokeBERT  [ 90 ] .

 
 
 

#### IV-B 3 Utilizing External Memory

 
 Fig. 12: Explicit incorporation of knowledge into PLMs via the utilization of external memory. 
 
 
 The third method for building KEPLMs explicitly uses external memory, and thus keeps the knowledge space and text space separate.

 
 
 In Fig.   12 , ① illustrates the method to apply non-parametric knowledge from external memory to downstream NLP tasks.
KGLM  [ 106 ] selects and copies the facts from a related knowledge graph to generate factual sentences.
In other words, it uses a knowledge base to expand the vocabulary to supply information it has never seen before.
REALM  [ 107 ] introduces a knowledge retriever to help the model retrieve and process documents from the knowledge corpus, and thus improves the performance of open-domain question answering.
It only needs to update the knowledge corpus if the world knowledge changes.

 
 
 In Fig.   12 , ② illustrates the method of learning parametric knowledge using an additional module independent of the PLM.
K-Adapter  [ 72 ] adds adapters to learn parametric knowledge, and the parameters of the PLM itself remain unchanged during pre-training.
Such adapters are independent of each other and can be trained in parallel.
In addition, more adapters can be added when needed.

 
 
 RAG  [ 108 ] that combines nonparametric and parametric memory outperforms other parametric-only and nonparametric-only models in three open domain question-answering tasks.
Furthermore, for text generation, it can create more specific, diverse, and factual text than other parameter-only baselines.
Wilmot et al.  [ 109 ] extend RAG  [ 108 ] by adding a memory module to improve the models’ predictive performance.

 
 
 When the knowledge base has undergone some changes, keeping the knowledge in external memory has the big advantage that the KEPLM does not require re-training, which is particularly helpful for the application domains where knowledge is updated frequently.

 
 
 
 
 

## V Evaluating KEPLMs 

 
 This section presents methods for evaluating KEPLMs in terms of knowledge capacity, effectiveness, and efficiency.

 
 

### V-A Knowledge Capacity 

 
 The amount of knowledge incorporated into KEPLMs could be assessed using knowledge probes such as LAMA  [ 59 ] and LAMA-UHN  [ 110 ] .
Intuitively, KEPLMs containing more knowledge would be more powerful for downstream NLP tasks.

 
 

#### V-A 1 LAMA

 
 LAnguage Model Analysis (LAMA) probe  [ 59 ] provides a series of completion statements that assess how much knowledge is stored in the model by the average accuracy of the model predictions.
Knowledge sources for LAMA include Google-RE  [ 111 ] , T-Rex  [ 112 ] , ConceptNet  [ 9 ] , and SQuAD  [ 113 ] .
The Google-RE corpus  [ 111 ] contains five kinds of relational triples, among which “place of birth”, “date of birth”, and “place of death” were selected by LAMA and transformed into fill-in-the-blank sentences according to the artificially constructed templates.
For example, the triple “place of birth” is built as “[S] was born in [O]”, where S represents the head entity and O represents the tail entity.
T-Rex  [ 112 ] is a subset of Wikidata containing 41 relations.
The triples in it were also manually transformed into fill-in-the-blank sentences.
LAMA also selects triples from ConceptNet, covering 16 relations.
For these triples, it finds the OMCS sentence containing both the head entity and the tail entity, then masks the tail entity within the sentence to construct a fill-in-the-blank sentence.
LAMA  [ 59 ] selected 305 question-answer pairs from SQuAD and manually constructed fill-in-the-blank sentences.
For example, the question “Who developed the theory of relativity?” was rewritten as ”The theory of relativity was developed by _ ​ _ ​ _ ​ _ \_\_\_\_ ”.
LAMA is generally recognized, and many existing work utilize LAMA  [ 59 ] to measure how much knowledge the model has learned, as shown in Table  III .

 
 
 TABLE III: Evaluation of Some KEPLMs on LAMA and LAMA-UHN Datasets 
 
 
 Model | 
 LAMA | 
 LAMA-UHN | 

 
 LAMA-Google-RE | 
 LAMA-T-REx | 
 ConceptNet | 
 SQuAD | 
 LAMA-UHN-Google-RE | 
 LAMA-UHN-T-REx | 

 
 CoLAKE  [ 89 ] | 
 9.5 | 
 28.8 | 
 — | 
 — | 
 4.9 | 
 20.4 | 

 
 KEPLER-Wiki  [ 88 ] | 
 7.3 | 
 24.6 | 
 18.7 | 
 14.3 | 
 3.3 | 
 16.5 | 

 
 KEPLER-W+W  [ 88 ] | 
 7.3 | 
 24.4 | 
 17.6 | 
 10.8 | 
 4.1 | 
 17.1 | 

 
 DKPLM  [ 101 ] | 
 10.8 | 
 32.0 | 
 — | 
 — | 
 5.4 | 
 22.9 | 

 
 KgPLM  [ 82 ] | 
 9.2 | 
 27.9 | 
 — | 
 — | 
 4.9 | 
 20.4 | 

 
 K-Adapter  [ 72 ] | 
 7.0 | 
 29.1 | 
 — | 
 — | 
 3.7 | 
 23.0 | 

 
 KALM  [ 99 ] | 
 5.41 | 
 28.12 | 
 10.7 | 
 11.89 | 
 — | 
 — | 

 
 EAE  [ 114 ] | 
 9.4 | 
 37.4 | 
 10.7 | 
 22.4 | 
 — | 
 — | 

 
 XLM-K  [ 73 ] | 
 11.2 | 
 29.7 | 
 15.7 | 
 11.5 | 
 — | 
 — | 

 
 
 In Table  III , DKPLM  [ 101 ] performs better overall, which shows that long-tail entity-based learning helps the model remember factual knowledge. EAE  [ 114 ] learns entity representations directly from text rather than integrating entity knowledge into the model and performs well on all three datasets related to factual knowledge, which illustrates the effectiveness of the method.

 
 
 

#### V-A 2 LAMA-UHN

 
 E-BERT  [ 110 ] found that for the fill-in-the-blank sentences of LAMA  [ 59 ] , the model may answer depending on the surface form of the entity name; for example, in predicting the language spoken by a person with an Italian-sounding name, the model would predict that the person speaks Italian.

 
 
 To prevent the model obtaining answers from helpful entity names, E-BERT  [ 110 ] proposes LAMA-UHN (UnHelpfulNames), a subset of LAMA  [ 59 ] that focuses on factual knowledge, which deletes sentences with overly helpful entity names.
Models such as CoLAKE  [ 89 ] , KEPLER  [ 88 ] , DKPLM  [ 101 ] , KgPLM  [ 82 ] , and K-Adapter  [ 72 ] are also evaluated on LAMA-UHN, as shown in Table  III .
The performances of the models on LAMA-UHN are much lower than those on LAMA, indicating that LAMA-UHN is more challenging to the model and it can better detect how much knowledge the model can actually learn.

 
 
 

#### V-A 3 Other Knowledge Probes

 
 In addition to LAMA and LAMA-UHN, there are other new knowledge probes. LPAQA  [ 115 ] considered that some sentences in LAMA and LAMA-UHN might be constructed inappropriately so that they only provide a lower bound estimate of the knowledge contained in an LM. That is, the model might know the answer but could not give the correct answer because of the inappropriate way of questioning. For example, for the sentence ”Obama is a _ ​ _ ​ _ ​ _ \_\_\_\_ by profession”, the model is asked about Obama’s profession, but the expression is unclear. If it is replaced by ”Obama worked as a _ ​ _ ​ _ ​ _ \_\_\_\_ ”, it may predict more accurately. LPAQA aims to estimate the knowledge contained in LMs more accurately. Dolphs et al.  [ 116 ] applied example queries to LAMA probes, and the model’s performance improved significantly, which also shows that we need to detect the knowledge contained in the language model properly to avoid underestimating the model.
Unlike the above methods, AutoPrompt  [ 117 ] proposes an automated way to create prompts for measuring the amount of knowledge contained in LMs. This method saves time and effort. Moreover, prompts created by AutoPrompt can estimate knowledge in LM more accurately than manually created ones.
In addition to general domain knowledge exploration, Meng et al. propose a biomedical knowledge exploration benchmark named MedLAMA  [ 118 ] .

 
 
 
 

### V-B Effectiveness 

 
 We assess the effectiveness of a method by analyzing whether it maintains the original language representation capabilities and how many tasks’ performances it can improve.
We choose GLUE  [ 119 ] and KILT  [ 120 ] as corresponding benchmarks.

 
 

#### V-B 1 General Language Understanding Tasks

 
 We choose the General Language Understanding Evaluation (GLUE) dataset  [ 119 ] as the benchmark to assess the general language representation capabilities maintained by KEPLMs.
It is a benchmark used to measure the performance of language models, containing nine natural language understanding tasks.
It is the primary evaluation benchmark used by BERT.
Many KEPLMs based on BERT or ROBERTa  [ 103 ] were tested on GLUE to explore whether incorporating knowledge affects the model’s performance in natural language processing tasks.
ERNIE  [ 67 ] has been tested on eight datasets of GLUE with essentially the same performance as BERT-base.
It found that no additional knowledge is needed to process the tasks in GLUE, and the model does not cause a loss of textual information after incorporating knowledge.
CoLAKE  [ 89 ] , SenseBERT  [ 75 ] , AMS  [ 19 ] , and other models have also been tested on GLUE and found that the way they incorporated knowledge did not affect the original language representation capabilities of the models.
CoLAKE  [ 89 ] and AMS  [ 19 ] found that solving tasks in GLUE does not require encyclopedic and commonsense knowledge, respectively.
Although we do not know whether all models incorporating knowledge affect the language representation capabilities of the models, the experiments done by the above models suggest that maintaining the original language representation capabilities of the model while incorporating knowledge is the goal pursued by the researchers.

 
 
 

#### V-B 2 Knowledge-Intensive Language Tasks

 
 Most KEPLMs are designed for specific tasks, such as KG-BART  [ 17 ] for generative commonsense reasoning, ExBERT  [ 21 ] for natrual language inferance, GreaseLM  [ 13 ] for question answering, and so forth.
These models may perform well in one task but fail in others.
KEPLMs that can elevate performance on more tasks at the same time are of higher value, such as KGI  [ 121 ] , which can simultaneously improve the performance of fact checking, slot filling, open-domain QA, and dialog generation.

 
 
 We choose the Knowledge Intensive Language Task (KILT) dataset  [ 120 ] as the benchmark to analyze the effectiveness of KEPLMs on different knowledge-intensive tasks.
It contains 11 datasets in 5 categories of tasks, including Fact-checking, Entity linking, Slot filling, Open-domain QA, and Dialog generation.
All tasks are based on the same Wikipedia snapshot, which aims to facilitate the development of general-purpose models and enable their comparisons.

 
 
 
 

### V-C Efficiency 

 
 We assess the efficiency of a KEPLM by considering its model size and required computational resources.

 
 

#### V-C 1 Model Size

 
 Incorporating more knowledge would necessarily mean the expansion of the PLM.
Usually the performance of a language model increases with its size (i.e., the number of the model parameters)  [ 66 ] , as shown in Table  IV .
Briefly speaking, for the same level of performance, the smaller the model size, the more efficient the KEPLM.

 
 
 TABLE IV: Comparison between KEPLMs and T5 in Terms of Model Size and Task Performance 
 
 
 Model | 
 Params | 
 Task Performance | 

 
 Natural Q. | 
 Web Q. | 

 
 T5-Base | 
 220M | 
 25.9 | 
 29.1 | 

 
 T5-Large | 
 770M | 
 28.5 | 
 32.2 | 

 
 T5-3B | 
 3B | 
 30.4 | 
 34.4 | 

 
 T5-11B | 
 11B | 
 34.5 | 
 37.4 | 

 
 EAE  [ 114 ] | 
 367M | 
 — | 
 39.0 | 

 
 REALM  [ 107 ] | 
 330M | 
 40.4 | 
 40.7 | 

 
 RAG-Token  [ 108 ] | 
 626M | 
 44.1 | 
 45.5 | 

 
 RAG-Seq  [ 108 ] | 
 626M | 
 44.5 | 
 45.2 | 

 
 
 In Table  IV , the performance of T5 on these two question-answering tasks increases with model parameters.
That is to say, adding parameters can increase knowledge to some extent.
However, the improvement of tasks is far less than the increment of parameters, which is expensive.
KEPLMs such as EAE  [ 114 ] , REALM  [ 107 ] , and RAG  [ 108 ] adopt different methods to incorporate knowledge, which has far fewer parameters than T5-11B but better performance, indicating that incorporating knowledge properly can significantly improve performance efficiently.

 
 
 

#### V-C 2 Computational Resources

 
 Computational resources usually increase with the model size and can also be a metric for evaluating the efficiency of KEPLMs.
Precisely, we assess KEPLMs mainly through the training time and GPU or TPU used, as shown in Table  V .

 
 
 TABLE V: Comparison of Computational Resourced Required by KEPLMs 
 
 
 Model | 
 GPU Type | 
 GPU Num | 
 Training | 

 
 SentiLARE  [ 70 ] | 
 NVIDIA RTX 2080 Ti | 
 4 | 
 20 hours | 

 
 DKPLM  [ 101 ] | 
 NVIDIA V100 16GB | 
 8 | 
 12 hours | 

 
 CoLAKE  [ 89 ] | 
 NVIDIA V100 32GB | 
 8 | 
 38 hours | 

 
 
 We take Table  V as an example to show how to compare the efficiency of KEPLMs by computing resources.
First, we compare the GPUs used by the models.
NVIDIA V100 GPU outperforms NVIDIA RTX 2080 Ti GPU.
Then, we multiply the number of GPUs by the training time to roughly calculate the training time required for the model to run on just one GPU.
SentiLARE takes 80 hours; DKPLM takes 96 hours; and CoLAKE takes 304 hours.
The GPU used by SentiLARE is not as good as DKPLM and CoLAKE, and SentiLARE requires less training time, so SentiLARE requires the least computing resources.
The GPU memory capacity used by DKPLM is smaller than CoLAKE, and the training time of DKPLM is smaller than CoLAKE, so DKPLM requires less computing resources than CoLAKE.
Therefore, the computing resources required by these three models from low to high correspond to SentiLARE, DKPLM, and CoLAKE.
The efficiency from high to low corresponds to SentiLARE, DKPLM, CoLAKE.

 
 
 To sum up, we first look at the GPUs used by the models; then, we multiply the number of GPUs by the training time to roughly calculate the total training time using only one GPU. When GPU performance is close, the model with less total training time is more efficient.

 
 
 
 
 

## VI Applying KEPLMs 

 
 KEPLMs are able to boost the performance of knowledge-intensive downstream tasks which can be grouped into two categories according to whether there is new natural language content created by the model.

 
 

### VI-A Knowledge-Enhanced NLU 

 
 KEPLMs based on Transformer encoder only or encoder-decoder could be used for natural language understanding (NLU) tasks, such as entity typing, entity recognition, relationship extraction, sentiment analysis, question answering, language-based reasoning, and knowledge graph completion.

 
 

#### VI-A 1 Entity Typing

 
 Given an entity mention and its context, entity typing requires the model to classify the semantic type of the entity mention.
FIGER  [ 122 ] and Open Entity  [ 123 ] are the most commonly used datasets. We found that models such as CoLAKE  [ 89 ] , KnowBERT  [ 68 ] , KEPLER  [ 88 ] , DKPLM  [ 101 ] , LUKE  [ 124 ] are only tested on Open Entity;
ERICA  [ 83 ] is only tested on FIGER;
ERNIE  [ 67 ] , CokeBERT  [ 90 ] , and K-Adapter  [ 72 ] experiments on both datasets.
Open Entity is more widely used, and we think the reasons are as follows. The training set of FIGER is annotated by remote supervision, and the testing set is manually annotated. Open Entity uses manual annotation for both datasets and has more types and finer-grained classification than FIGER.
In addition, BERT-MK  [ 32 ] performs entity typing in the medical domain, using the datasets 2010 i2b2/VA  [ 125 ] , JNLPBA  [ 126 ] , and BC5CDR  [ 127 ] .

 
 
 Most of the above methods insert special tokens before and after entity mentions in a given sentence to mark entity mentions (e.g., “he had a differential diagnosis of [E] asystole [/E]”) and then use the embeddings of the special symbol preceding the entity mention (i.e., [E]) to predict the entity type.

 
 
 Li et al.  [ 128 ] proposed a new dataset WikiWiki, containing 10 million Wikipedia articles with each entity connected to the knowledge graph of Wikidata  [ 58 ] .
Compared with the existing fine-grained type recognition datasets, Wikiwiki is larger and more accurate, which can also be used for entity typing tasks.

 
 
 

#### VI-A 2 Entity Recognition

 
 The Named Entity Recognition (NER) task requires the model to identify the entity mentioned in a given text.
In has been the basis for many NLP applications in both general and specific domains (like biomedical), as shown in Table   VI .

 
 
 TABLE VI: KEPLMs for Entity Recognition 
 
 
 
 
 Domain 
 | 
 
 
 Dataset 
 | 
 
 
 Model 
 | 

 
 
 
 General 
 | 
 
 
 MSRA-NER  [ 129 ] 
 | 
 
 
 K-BERT  [ 69 ] , ERNIE  [ 98 ] , ERNIE 2.0  [ 130 ] 
 | 

 
 | 
 
 
 CoNLL-2003  [ 131 ] 
 | 
 
 
 KALA  [ 92 ] , LUKE  [ 124 ] , KMLMs  [ 132 ] 
 | 

 
 
 
 Biomedical 
 | 
 
 
 English i2b2  [ 123 , 133 , 134 ] 
 | 
 
 
 UmlsBERT  [ 33 ] 
 | 

 
 | 
 
 
 DXY-NER  [ 135 ] 
 | 
 
 
 SMedBERT  [ 34 ] 
 | 

 
 | 
 
 
 Medicine _ \_ NER  [ 136 ] 
 | 
 
 
 K-BERT  [ 69 ] 
 | 

 
 | 
 
 
 JNLPBA  [ 126 ] , BC5-chem \ BC5-disease  [ 127 ] , NCBI-disease  [ 137 ] , BC2GM  [ 138 ] 
 | 
 
 
 KeBioLM  [ 35 ] , BioBERT  [ 139 ] 
 | 

 
 
 
 Finance 
 | 
 
 
 Finance _ \_ NER  [ 140 ] 
 | 
 
 
 K-BERT  [ 69 ] 
 | 

 
 
 
 Social media 
 | 
 
 
 WNUT-17  [ 141 ] 
 | 
 
 
 KALA  [ 92 ] 
 | 

 
 
 
 Cross-lingual 
 | 
 
 
 WikiAnn NER  [ 142 ] 
 | 
 
 
 KMLMs  [ 132 ] 
 | 

 
 
 

#### VI-A 3 Relation Extraction

 
 TABLE VII: KEPLMs for Relation Extraction 
 
 
 
 
 Domain 
 | 
 
 
 Dataset 
 | 
 
 
 Model 
 | 

 
 
 
 General 
 | 
 
 
 TACRED  [ 143 ] 
 | 
 
 
 K-Adapter  [ 72 ] , ERNIE  [ 67 ] , ERICA  [ 83 ] , KEPLER  [ 88 ] , CokeBERT  [ 90 ] , LUKE  [ 124 ] , Glass et al.  [ 144 ] , DKPLM  [ 101 ] , EAE  [ 114 ] 
 | 

 
 | 
 
 
 FewRel  [ 145 ] 
 | 
 
 
 ERNIE  [ 67 ] , KEPLER  [ 88 ] , CoLAKE  [ 89 ] , JAKET  [ 105 ] , CokeBERT  [ 90 ] 
 | 

 
 
 
 Biomedical 
 | 
 
 
 2010 i2b2/VA  [ 125 ] , GAD  [ 146 ] , EU-ADR  [ 147 ] 
 | 
 
 
 BERT-MK  [ 32 ] 
 | 

 
 | 
 
 
 DXY-NER  [ 135 ] , CHIP-RE  [ 148 ] 
 | 
 
 
 SMedBERT  [ 34 ] 
 | 

 
 | 
 
 
 GAD  [ 146 ] , DDI  [ 149 ] , ChemProt  [ 150 ] 
 | 
 
 
 KeBioLM  [ 35 ] 
 | 

 
 
 KEPLMs could help to improve the extraction (and classification) of the relations between entities in a given text document.
In addition to the public field, this task is more commonly used in the biomedical domain.
As can be seen from Table   VII , the commonly used datasets in the general field are TACRED  [ 143 ] and FewRel  [ 145 ] , and TACRED is used more than FewRel.
We find that all methods using these two datasets perform better on FewRel than on TACRED.
That is to say, TACRED is more challenging than FewRel, so more and more methods tend to test model performance on TACRED.
There are many work explicitly designed for the biomedical domain, and just like NER, relation classification tasks are helpful for models to learn domain-specific knowledge.

 
 
 

#### VI-A 4 Sentiment Analysis

 
 There are two kinds of sentiment analysis tasks: sentence-level sentiment analysis and aspect-level sentiment analysis. Sentence-level sentiment analysis requires models to determine the sentiment polarity of sentences, and commonly used datasets are Stanford Sentiment Treebank SST-2  [ 151 ] and Amazon-2  [ 152 ] . Aspect-level sentiment analysis requires the model to analyze the sentiment polarity in different aspects of the context, and commonly used datasets are SemEval-2014 Task 4  [ 153 ] . SentiLARE  [ 70 ] and SKEP  [ 47 ] obtained better results than pure PLM on both sentence-level and aspect-level tasks by incorporating sentiment knowledge. Through sentiment analysis, REMOTE  [ 48 ] could detect hate speech, and KET  [ 46 ] could detect sentiment in dialogues, which helps question-answering robots make better responses.

 
 
 

#### VI-A 5 Question Answering

 
 Question answering tasks include machine reading for question answering (MRQA), open-domain question answering (Open-domain QA), and multiple-choice question answering (Multiple-choice QA) according to the question form. We present commonly used datasets for each task in Table   VIII .

 
 
 TABLE VIII: KEPLMs for Question Answering 
 
 
 
 
 Domain 
 | 
 
 
 Dataset 
 | 
 
 
 Model 
 | 

 
 
 
 MRQA 
 | 
 
 
 SQuAD 1.1  [ 113 ] , NewsQA  [ 154 ] , TriviaQA  [ 155 ] , SearchQA  [ 156 ] 
 | 
 
 
 KT-NET  [ 74 ] , KgPLM  [ 82 ] 
 | 

 
 
 
 Open-Domain 
 | 
 
 
 Natural Questions  [ 157 ] , Web Questions  [ 158 ] , TriviaQA  [ 155 ] , SearchQA  [ 156 ] 
 | 
 
 
 REALM  [ 107 ] , K-ADAPTER  [ 71 ] , WKLM  [ 102 ] , and EAE  [ 114 ] 
 | 

 
 
 
 Multi-Choice 
 | 
 
 
 CommonsenseQA  [ 159 ] , OpenBookQA  [ 160 ] , CosmosQA  [ 161 ] 
 | 
 
 
 QA-GNN  [ 12 ] , GreaseLM  [ 13 ] , JointLK  [ 14 ] 
 | 

 
 
 In Table   VIII , MRQA, also known as Extractive Question Answering, provides questions and related articles requires the model to find answers from the provided articles. The most commonly used dataset is SQuAD 1.1  [ 113 ] .

 
 
 The Open-domain QA task gives the questions without the articles containing answers, requiring models to retrieve relevant articles.
Methods such as REALM  [ 107 ] , K-ADAPTER  [ 71 ] , WKLM  [ 102 ] , and EAE  [ 114 ] are tested using some of the corresponding datasets in Table   VIII , and they achieve better results than the competitive baselines after incorporating encyclopedic knowledge.

 
 
 The multiple-choice QA task requires the model to select the correct answer based on the question and the options given. The three datasets listed in Table   VIII are all commonsense question-answering tasks. Among them, CommonsenseQA  [ 159 ] has five options, OpenBookQA  [ 160 ] and CosmosQA  [ 161 ] have four options, and CommonsenseQA is the most widely used.

 
 
 From Table   VIII , we can see that the datasets of Open-domain QA and MRQA overlap. Open-domain QA only removes the articles provided to the model based on MRQA and asks the model to retrieve them by itself so that they can share datasets.

 
 
 

#### VI-A 6 Language-based Reasoning

 
 Representative KEPLMs used for reasoning tasks include
SMedBERT  [ 34 ] and Li et al.  [ 162 ] for natural language inference ,
KMLMs  [ 132 ] for logical reasoning,
Andor et al.  [ 163 ] for mathematical reasoning,
Chang et al.  [ 24 ] and Vokenization  [ 97 ] for commonsense reasoning,
CoCoLM  [ 31 ] for reasoning about the temporal order of events, and
VALM  [ 96 ] for reasoning about object color and size.

 
 
 

#### VI-A 7 Knowledge Graph Completion

 
 Knowledge graphs often suffer from incompleteness, and many relationships between entities are missing.
KEPLMs can help infer missing links and complement knowledge graphs to a certain degree.

 
 
 Models such as GLM  [ 15 ] were tested on the WN18RR  [ 164 ] and CKBC  [ 165 ] sets. It outperforms some translation-based graph embedding models and graphs convolutional networks on WN18RR. CKBC is a generic knowledge graph derived from OMCS  [ 166 ] , and GLM  [ 15 ] outperforms KG-BERT  [ 167 ] , a model specifically designed for the knowledge graph completion task, on this dataset.

 
 
 K-PLUG  [ 42 ] performs the e-commerce knowledge graph completion task on MEPAVE  [ 168 ] , which gives a textual description of a product and asks the model to output the attribute values of the product.

 
 
 
 

### VI-B Knowledge-Enhanced NLG 

 
 KEPLMs based on Transformer decoder only or Transformer encoder-decoder could be used for natural language generation (NLG) tasks, such as sentence generation, dialogue generation, question generation, and answer generation.

 
 

#### VI-B 1 Sentence Generation

 
 The sentence generation task requires models to generate reasonable sentences, and commonly used datasets are CommonGen  [ 169 ] and ROCStories  [ 170 ] . CommonGen requires models to generate a coherent, proper sentence based on 3-5 given concepts, while KG-BART  [ 17 ] does so by incorporating commonsense knowledge subgraphs. Models such as GRF  [ 20 ] , Guan et al.  [ 171 ] , and Guan et al.  [ 23 ] can generate plausible story endings with the help of commonsense knowledge.

 
 
 

#### VI-B 2 Dialogue Generation

 
 Dialogue generation tasks require the model to generate responses based on the context of the dialogue. KnowledGPT  [ 172 ] chose to conduct experiments on Wizard  [ 173 ] and CMU _ \_ DoG  [ 174 ] datasets. Wizard has a wide range of topics, while CMU _ \_ DoG only focuses on the movie domain.

 
 
 

#### VI-B 3 Question Generation

 
 The question generation task requires the model to generate questions based on the answers. RAG  [ 108 ] proposes the Jeopardy Question Generation task, where Jeopardy consists of trying to guess an entity from the facts about it. For example, given the answer “The World Cup”, models need to generate relevant fact that points to the answer, like “In 1986, Mexico scored as the first country to host this international sports competition twice.”

 
 
 

#### VI-B 4 Answer Generation

 
 Unlike standard question-answering (QA) tasks mentioned above in Section   VI-A5 , open-domain abstractive QA, aka zero-shot QA or closed-book QA, requires the model to generate answers by itself rather than finding answers from passages or selecting from options.
RAG  [ 108 ] only uses questions and answers from the dataset of MSMARCO NLGT task v2.1  [ 175 ] , treating it as the open-domain abstractive QA task and outperforming the baseline model BART  [ 56 ] .

 
 
 
 
 

## VII Future Directions 

 
 In the above sections, we have presented KEPLMs from multiple perspectives, but there are still some other opportunities.
Here we outline and discuss a few promising research directions for KEPLMs.

 
 
 Utilizing More Types of Knowledge. 
As mentioned in Section   III , existing KEPLMs have considered many types of knowledge, but there are other types of knowledge worth investigating.
For example, temporal knowledge graphs such as HyTE  [ 176 ] contain events that reflect the relationships between different entities over time, so incorporating them into PLMs could help to perform time-related reasoning tasks.
Moreover, the phenomenal success of ChatGPT has demonstrated the power of incorporating the knowledge of human intentions and preferences into PLMs directly through Reinforcement Learning from Human Feedback (RLHF)  [ 177 ] .

 
 
 Improving the Effectiveness of Knowledge Incorporation. 
As described in Section   IV , a variety of technical approaches to incorporating knowledge into PLMs have been proposed.
Some of those methods such as KEPLER  [ 88 ] and CokeBERT  [ 90 ] rely on sophisticated joint pre-training of PLMs and KG embeddings.
However, KEPLER  [ 88 ] performs worse on entity typing and relation classification tasks than LUKE  [ 124 ] which contains only entity-level knowledge.
CokeBERT  [ 90 ] is slightly better than LUKE  [ 124 ] on some datasets, but it is not as efficient as LUKE  [ 124 ] .
Hou et al.  [ 178 ] proposed the Graph Convolution Simulator to detect knowledge integrated into PLMs.
Their examination of ERNIE  [ 67 ] and K-Adapter  [ 72 ] revealed that those KEPLMS have only incorporated a small amount of factual knowledge.
There still seems to be much room for more effective incorporation of knowledge into PLMs.

 
 
 Improving the Efficiency of Knowledge Incorporation. 
Most existing work about KEPLMs only report improvements with respect to model performance, and only a few assess the costs of knowledge incorporation as well.
More time-efficient and space-efficient solutions to KEPLMs are desired.
Many methods, such as CoLAKE  [ 89 ] and ERNIE  [ 67 ] , carry out knowledge incorporation in the pre-training stage, while some others like K-BERT  [ 69 ] , K-Adapter  [ 71 ] , and Syntax-BERT  [ 71 ] carry out knowledge incorporation in the fine-tuning stage.
The time cost of knowledge incorporation in the pre-training stage is greater than doing that in the fine-tuning stage.
It deserves more investigation to minimize the overhead in the pre-training stage while maintaining good performance.
Besides, knowledge incorporation may also increase the inference overhead of the model.
For example, GRF  [ 20 ] and KG-BART  [ 17 ] involve the construction of knowledge sub-graphs, which makes their inference time much longer.
More efficient inference strategies need to be developed for KEPLMs to facilitate their practical applications.
The additional space consumption of KEPLMs must also be carefully considered before their deployment.
For example, FaE  [ 179 ] needs an external entity memory and a factual memory containing millions of knowledge triples.
RAG  [ 108 ] relies on a non-parametric knowledge corpus containing tens of millions of documents.
Not all of these integrated entities or facts are equally useful: some of them probably play more important roles than others in enhancing the PLM.
Therefore, selecting and storing only the most critical subset of knowledge may significantly reduce the space overhead with a small sacrifice in performance.
In addition to avoiding the incorporation of less important knowledge, model compression techniques  [ 180 ] can be used to reduce the computational overhead of KEPLMs.
For example, quantization  [ 181 ] , knowledge distillation  [ 182 ] , and parameter sharing  [ 183 ] can all be applied to KEPLMs to improve their time and space efficiency.

 
 
 Exploring Other Knowledge-Intensive Tasks. 
In addition to the downstream NLP tasks listed in Section   VI , some other less-explored applications may also benefit from KEPLMs.
For example, KEPLMs are likely to improve the correctness of machine translation and the factualness of text summarization  [ 184 ] .

 
 
 Building A Unified KEPLM for Multiple Tasks. 
Most of the existing KEPLMs are designed for specific knowledge-intensive NLP tasks.
Currently to achieve SOTA performance for different tasks, one often needs to train a different KEPLM for each of them.
It is desirable to develop a unified KEPLM for multiple tasks so as to avoid the costly proliferation of KEPLMs.
There have been some early attempts towards this direction, such as KGI  [ 121 ] which is trained to improve the performance on four different tasks in the KILT  [ 120 ] benchmark.

 
 
 Performing Zero/Few-shot Learning. 
In some application domains, there are little quality labelled data, therefore zero-shot learning or few-shot learning will be particularly useful.
Thanks to the knowledge built into KEPLMs, they are more able than standard PLMs to overcome the data scarcity problem and tackle many zero/few-shot learning tasks.
KALM  [ 99 ] signals the existence of entities to the input in pre-training to integrate knowledge, significantly improving zero-shot question-answering tasks.
Li et al.  [ 128 ] introduced fine-grained type knowledge of entities, achieving superior performance in zero-shot dialog state tracking.
Other than incorporating knowledge into PLMs, one can also exploit external knowledge in prompt engineering   [ 185 , 186 , 187 , 188 ] .
It would be interesting to investigate how to maximize the combined effect of knowledge-enhanced PLMs and knowledge-enhanced prompt engineering together.

 
 
 Achieving Better Interpretability and Robustness. 
The interpretability of a model measures how easily a human can understand its results and predictions.
Schuff et al.  [ 189 ] investigated whether incorporating external knowledge can help to explain natural language inference tasks.
They have argued that there is a discrepancy between the automatic evaluation method of models and manual scoring, and the effectiveness of automatic evaluation needs to be reconsidered.
Akyürek et al.  [ 190 ] attempted to trace the predictions made by the model back to training data.
Cao et al.  [ 191 ] and LEFA  [ 192 ] , on the other hand, attempted to locate the knowledge stored in the model.
The robustness of a model refers to its resistance to input disturbances or adversarial attacks etc.
Li et al.  [ 162 ] improved the robustness of the model by introducing external lexical knowledge into the attention mechanisms.
Glass et al.  [ 144 ] demonstrated adaptive capabilities on new datasets to illustrate the model’s robustness.
There are not many existing studies which try to improve the interpretability or robustness of KEPLMs.
More in-depth investigations on these aspects would be helpful.

 
 
 

## VIII Conclusion 

 
 In summary, this survey provides a comprehensive view of current advances in the rapidly evolving field of KEPLMs.
We begin by briefly introducing KEPLMs and describing the knowledge types/formats along with the methods for knowledge incorporation.
PLMs can be enhanced by a wide range of knowledge, with encyclopedic and commonsense knowledge being most widely used, and domain-specific knowledge being increasingly explored.
Different types of knowledge come in various forms, e.g., knowledge graphs could be integrated into PLMs directly as triples or indirectly through embeddings.
The methods for knowledge incorporation can be classified into two main categories, implicit and explicit.
Implicit incorporation does not put external knowledge into the model but employs knowledge-guided masking strategies or knowledge-related pre-training tasks to mine and learn knowledge from the pre-traineing corpus.
Explicit incorporation can be adding knowledge to the input, integrating knowledge through fusion structures, or storing knowledge in external memory and retrieving it when needed.
We then introduce some off-the-shelf methods for assessing the effectiveness of KEPLMs by detecting the amount of knowledge learned by the model, propose metrics for assessing model efficiency, and suggest assessing the generality of the model based on whether it can simultaneously boost performance on various tasks.
After that, we present a list of knowledge-intensive tasks and some application areas worth considering.
A final discussion of KEPLM research directions concludes this paper. We hope this will inspire researchers to explore KEPLMs further in the future.
Finally, we discuss potential research directions for KEPLMs, which we hope will inspire future research in this area.

 
 
 

## Acknowledgments

 
 Yong Chen is supported by the Young Scientists Fund of the National Natural Science Foundation of China (Grant No. 62006005), and the National Key Research and Development Program of China (No. SQ2022YFC3300043).

 
 
 

## References

 
 
 [1] 
 
T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, “Distributed
representations of words and phrases and their compositionality,” in
 Proc. Int. Conf. Neural Inf. Process. Syst , vol. 26, 2013.

 

 
 [2] 
 
J. Pennington, R. Socher, and C. D. Manning, “Glove: Global vectors for word
representation,” in Proc. Conf. Empir. Methods Natural Lang.
Process. , 2014, pp. 1532–1543.

 

 
 [3] 
 
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep
bidirectional transformers for language understanding,” in Proc. Conf.
North Amer. Chapter Assoc. Comput. Linguistics: Hum. Lang. Technol. , 2019,
pp. 4171–4186.

 

 
 [4] 
 
A. Vaswani et al. , “Attention is all you need,” in Proc. Int.
Conf. Neural Inf. Process. Syst. , 2017, pp. 6000–6010.

 

 
 [5] 
 
P. Colon-Hernandez, C. Havasi, J. Alonso, M. Huggins, and C. Breazeal,
“Combining pre-trained language models and structured knowledge,”
 arXiv preprint arXiv:2101.12294 , 2021.

 

 
 [6] 
 
D. Yin, L. Dong, H. Cheng, X. Liu, K.-W. Chang, F. Wei, and J. Gao, “A survey
of knowledge-intensive nlp with pre-trained language models,” arXiv
preprint arXiv:2202.08772 , 2022.

 

 
 [7] 
 
X. Wei et al. , “Knowledge enhanced pretrained language models: A
compreshensive survey,” arXiv preprint arXiv:2202.08772 , 2022.

 

 
 [8] 
 
J. Yang et al. , “A survey of knowledge enhanced pre-trained models,”
 arXiv preprint arXiv:2110.00269 , 2021.

 

 
 [9] 
 
R. Speer, J. Chin, and C. Havasi, “Conceptnet 5.5: An open multilingual graph
of general knowledge,” in Proc. 31th AAAI Conf. Artif. Intell. , 2017.

 

 
 [10] 
 
Q. Dong et al. , “Incorporating explicit knowledge in pre-trained
language models for passage re-ranking,” arXiv preprint
arXiv:2204.11673 , 2022.

 

 
 [11] 
 
Y. Zhang, J. Lin, Y. Fan, P. Jin, Y. Liu, and B. Liu, “Cn-hit-it. nlp at
semeval-2020 task 4: Enhanced language representation with multiple knowledge
triples,” in Proc. 14th Workshop on Semantic Eval. , 2020, pp.
494–500.

 

 
 [12] 
 
M. Yasunaga, H. Ren, A. Bosselut, P. Liang, and J. Leskovec, “Qa-gnn:
Reasoning with language models and knowledge graphs for question answering,”
in Proc. 2021 Conf. North Amer. Chapter Assoc. Comput. Linguistics:
Hum. Lang. Technol. , 2021, pp. 535–546.

 

 
 [13] 
 
X. Zhang, A. Bosselut, M. Yasunaga, H. Ren, P. Liang, C. D. Manning, and
J. Leskovec, “Greaselm: Graph reasoning enhanced language models,” in
 ICLR , 2022, pp. 1–16.

 

 
 [14] 
 
Y. Sun, Q. Shi, L. Qi, and Y. Zhang, “Jointlk: Joint reasoning with language
models and knowledge graphs for commonsense question answering,” arXiv
preprint arXiv:2112.02732 , 2021.

 

 
 [15] 
 
T. Shen, Y. Mao, P. He, G. Long, A. Trischler, and W. Chen, “Exploiting
structured knowledge in text via graph-guided representation learning,” in
 Proc. 2020 Conf. Empir. Methods Natural Lang. Process. , 2020, pp.
8980–8994.

 

 
 [16] 
 
B. Y. Lin, X. Chen, J. Chen, and X. Ren, “Kagnet: Knowledge-aware graph
networks for commonsense reasoning,” in Proc. 2019 Conf. Empir.
Methods Natural Lang. Process. and 9th Int. Joint Conf. Natural Lang.
Process. , 2019, pp. 2829–2839.

 

 
 [17] 
 
Y. Liu, Y. Wan, L. He, H. Peng, and S. Y. Philip, “Kg-bart: Knowledge
graph-augmented bart for generative commonsense reasoning,” in Proc.
AAAI Conf. Artif. Intell. , vol. 35, no. 7, 2021, pp. 6418–6425.

 

 
 [18] 
 
A. Bosselut, H. Rashkin, M. Sap, C. Malaviya, A. Celikyilmaz, and Y. Choi,
“Comet: Commonsense transformers for automatic knowledge graph
construction,” in PProc. 57th Annu. Meeting Assoc. Comput.
Linguistics , 2019, pp. 4762–4779.

 

 
 [19] 
 
Z.-X. Ye, Q. Chen, W. Wang, and Z.-H. Ling, “Align, mask and select: A simple
method for incorporating commonsense knowledge into language representation
models,” arXiv preprint arXiv:1908.06725 , 2019.

 

 
 [20] 
 
H. Ji, P. Ke, S. Huang, F. Wei, X. Zhu, and M. Huang, “Language generation
with multi-hop reasoning on commonsense knowledge graph,” in Proc.
2020 Conf. Empir. Methods Natural Lang. Process. , 2020, pp. 725–736.

 

 
 [21] 
 
A. Gajbhiye, N. A. Moubayed, and S. Bradley, “Exbert: An external knowledge
enhanced bert for natural language inference,” in Int. Conf. Artif.
Neural Netw.  Springer, 2021, pp.
460–472.

 

 
 [22] 
 
A. Lauscher, O. Majewska, L. F. Ribeiro, I. Gurevych, N. Rozanov, and
G. Glavaš, “Common sense or world knowledge? investigating
adapter-based knowledge injection into pretrained transformers,” in
 Proc. Deep Learn. Inside Out , 2020, pp. 43–49.

 

 
 [23] 
 
J. Guan, F. Huang, Z. Zhao, X. Zhu, and M. Huang, “A knowledge-enhanced
pretraining model for commonsense story generation,” Trans. Assoc.
Comput. Linguistics , vol. 8, pp. 93–108, 2020.

 

 
 [24] 
 
T.-Y. Chang et al. , “Incorporating commonsense knowledge graph in
pretrained models for social commonsense tasks,” in Proc. Deep Learn.
Inside Out , 2021, pp. 74–79.

 

 
 [25] 
 
X. Yang, X. Zhu, Z. Shi, and T. Li, “Unsupervised pre-training with structured
knowledge for improving natural language inference,” arXiv preprint
arXiv:2109.03941 , 2021.

 

 
 [26] 
 
M. Sap et al. , “Atomic: An atlas of machine commonsense for if-then
reasoning,” in Proc. 33th AAAI Conf. Artif. Intell. , vol. 33, no. 01,
2019, pp. 3027–3035.

 

 
 [27] 
 
A. Mitra, P. Banerjee, K. K. Pal, S. Mishra, and C. Baral, “How additional
knowledge can improve natural language commonsense question answering?”
 arXiv preprint arXiv:1909.08855 , 2019.

 

 
 [28] 
 
J. D. Hwang et al. , “(comet-) atomic 2020: On symbolic and neural
commonsense knowledge graphs,” in Proc. 35th AAAI Conf. Artif.
Intell. , vol. 35, no. 7, 2021, pp. 6384–6392.

 

 
 [29] 
 
P. Hosseini, D. A. Broniatowski, and M. Diab, “Knowledge-augmented language
models for cause-effect relation classification,” in Proc. First
Workshop on Commonsense Representation Reasoning , 2022, pp. 43–48.

 

 
 [30] 
 
H. Zhang, X. Liu, H. Pan, Y. Song, and C. W.-K. Leung, “Aser: A large-scale
eventuality knowledge graph,” in Proc. Web Conf. , 2020, pp. 201–211.

 

 
 [31] 
 
C. Yu, H. Zhang, Y. Song, and W. Ng, “Cocolm: Complex commonsense enhanced
language model with discourse relations,” in Findings Assoc. Comput.
Linguistics , 2022, pp. 1175–1187.

 

 
 [32] 
 
B. He et al. , “Bert-mk: Integrating graph contextualized knowledge into
pre-trained language models,” in Findings Assoc. Comput. Linguistics ,
2020, pp. 2281–2290.

 

 
 [33] 
 
G. Michalopoulos, Y. Wang, H. Kaka, H. Chen, and A. Wong, “Umlsbert: Clinical
domain knowledge augmentation of contextual embeddings using the unified
medical language system metathesaurus,” in Proc. Conference North
Amer. Chapter Assoc. Comput. Linguistics: Hum. Lang. Technol. , 2021, pp.
1744–1753.

 

 
 [34] 
 
T. Zhang, Z. Cai, C. Wang, M. Qiu, B. Yang, and X. He, “Smedbert: A
knowledge-enhanced pre-trained language model with structured semantics for
medical text mining,” in Proc. 59th Annu. Meeting Assoc. Comput.
Linguistics and 11th Int. Joint Conf. Artif. Intell. , 2021, pp. 5882–5893.

 

 
 [35] 
 
Z. Yuan, Y. Liu, C. Tan, S. Huang, and F. Huang, “Improving biomedical
pretrained language models with knowledge,” in Proc. 20th Workshop
Biomed. Lang. Process , 2021, pp. 180–190.

 

 
 [36] 
 
J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C. H. So, and J. Kang, “Biobert: a
pre-trained biomedical language representation model for biomedical text
mining,” Bioinform. , vol. 36, no. 4, pp. 1234–1240, 2020.

 

 
 [37] 
 
N. Liu, Q. Hu, H. Xu, X. Xu, and M. Chen, “Med-bert: A pretraining framework
for medical records named entity recognition,” IEEE Trans. Ind.
Informatics , vol. 18, no. 8, pp. 5600–5608, 2022.

 

 
 [38] 
 
I. Beltagy, K. Lo, and A. Cohan, “Scibert: A pretrained language model for
scientific text,” in EMNLP-IJCNLP , 2019, pp. 3613–3618.

 

 
 [39] 
 
W. Nicholas, T. Amalie, H. Haoyan, L. Sanghoon, C. Kevin, D. John,
D. Alexander, P. Kristin, C. Gerbrand, and J. Anubhav, “The impact of
domain-specific pre-training on named entity recognition tasks in materials
science,” in http://dx.doi.org/10.2139/ssrn.3950755 , 2021, pp. 1–43.

 

 
 [40] 
 
D. Zhang, Z. Yuan, Y. Liu, F. Zhuang, H. Chen, and H. Xiong, “E-bert: a phrase
and product knowledge enhanced language model for e-commerce,” arXiv
preprint arXiv:2009.02835 , 2020.

 

 
 [41] 
 
F. Sun, F.-L. Li, R. Wang, Q. Chen, X. Cheng, and J. Zhang, “K-aid: Enhancing
pre-trained language models with domain knowledge for question answering,”
in Proc. 30th ACM Int. Conf. Inf. Knowl. Manage , 2021, pp. 4125–4134.

 

 
 [42] 
 
S. Xu et al. , “K-plug: Knowledge-injected pre-trained language model
for natural language understanding and generation in e-commerce,” in
 Findings Assoc. Comput. Linguistics , 2021, pp. 1–17.

 

 
 [43] 
 
I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras, and
I. Androutsopoulos, “LEGAL-BERT: preparing the muppets for court,” in
 EMNLP (Findings) , 2020, pp. 2898–2904.

 

 
 [44] 
 
L. Zheng, N. Guha, B. R. Anderson, P. Henderson, and D. E. Ho, “When does
pretraining help?: assessing self-supervised learning for law and the
casehold dataset of 53, 000+ legal holdings,” in ICAIL , 2021, pp.
159–168.

 

 
 [45] 
 
C. Xiao, X. Hu, Z. Liu, C. Tu, and M. Sun, “Lawformer: A pre-trained language
model for chinese legal long documents,” in AI Open , 2021, pp.
79–84.

 

 
 [46] 
 
P. Zhong, D. Wang, and C. Miao, “Knowledge-enriched transformer for emotion
detection in textual conversations,” in Proc. 2019 Conf. Empir.
Methods Natural Lang. Process. and 9th Int. Joint Conf. Natural Lang.
Process. , 2019, pp. 165–176.

 

 
 [47] 
 
H. Tian et al. , “Skep: Sentiment knowledge enhanced pre-training for
sentiment analysis,” in Proc. 58th Annu. Meeting Assoc. Comput.
Linguistics , 2020, pp. 4067–4076.

 

 
 [48] 
 
H. Yao, Y. Chen, Q. Ye, X. Jin, and X. Ren, “Refining language models with
compositional explanations,” in Proc. 34th Int. Conf. Neural Inf.
Process. Syst. , vol. 34, 2021, pp. 8954–8967.

 

 
 [49] 
 
D. Guo, S. Ren, S. Lu, Z. Feng, D. Tang, S. Liu, L. Zhou, N. Duan,
A. Svyatkovskiy, S. Fu, M. Tufano, S. K. Deng, C. B. Clement, D. Drain,
N. Sundaresan, J. Yin, D. Jiang, and M. Zhou, “Graphcodebert: Pre-training
code representations with data flow,” in ICLR , 2021, pp. 1–18.

 

 
 [50] 
 
T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word
representations in vector space,” arXiv preprint arXiv:1301.3781 ,
2013.

 

 
 [51] 
 
X. Qiu, T. Sun, Y. Xu, Y. Shao, N. Dai, and X. Huang, “Pre-trained models for
natural language processing: A survey,” Sci. China Technol. Sc. ,
vol. 63, no. 10, pp. 1872–1897, 2020.

 

 
 [52] 
 
S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural
Comput. , vol. 9, no. 8, pp. 1735–1780, 1997.

 

 
 [53] 
 
B. McCann, J. Bradbury, C. Xiong, and R. Socher, “Learned in translation:
Contextualized word vectors,” in Proc. 30th Int. Conf. Neural Inf.
Process. Syst. , 2017, pp. 6294–6305.

 

 
 [54] 
 
M. E. Peters et al. , “Deep contextualized word representations,” in
 Proc. Conference North Amer. Chapter Assoc. Comput. Linguistics: Hum.
Lang. Technol. , New Orleans, Louisiana, 2018, pp. 2227–2237.

 

 
 [55] 
 
A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, “Improving language
understanding by generative pre-training,” OpenAI Blog , 2018.
[Online]. Available:
 https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf 

 

 
 [56] 
 
M. Lewis et al. , “Bart: Denoising sequence-to-sequence pre-training for
natural language generation, translation, and comprehension,” in Proc.
58th Annu. Meeting Assoc. Comput. Linguistics , 2020, pp. 7871–7880.

 

 
 [57] 
 
P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig, “Pre-train,
prompt, and predict: A systematic survey of prompting methods in natural
language processing,” arXiv preprint arXiv:2107.13586 , 2021.

 

 
 [58] 
 
D. Vrandečić and M. Krötzsch, “Wikidata: a free collaborative
knowledgebase,” Commun. ACM , vol. 57, no. 10, pp. 78–85, 2014.

 

 
 [59] 
 
F. Petroni et al. , “Language models as knowledge bases?” arXiv
preprint arXiv:2101.12294 , 2019.

 

 
 [60] 
 
C. Wang, X. Liu, and D. Song, “Language models are open knowledge graphs,”
 arXiv preprint arXiv:2010.11967 , 2020.

 

 
 [61] 
 
B. Heinzerling and K. Inui, “Language models as knowledge bases: On entity
representations, storage capacity, and paraphrased queries,” arXiv
preprint arXiv:2008.09036 , 2020.

 

 
 [62] 
 
T. Safavi and D. Koutra, “Relational world knowledge representation in
contextual language models: A review,” arXiv preprint
arXiv:2104.05837 , 2021.

 

 
 [63] 
 
S. Razniewski, A. Yates, N. Kassner, and G. Weikum, “Language models as or for
knowledge bases,” arXiv preprint arXiv:2110.04888 , 2021.

 

 
 [64] 
 
C. Wang, P. Liu, and Y. Zhang, “Can generative pre-trained language models
serve as knowledge bases for closed-book qa?” arXiv preprint
arXiv:2106.01561 , 2021.

 

 
 [65] 
 
B. AlKhamissi, M. Li, A. Celikyilmaz, M. Diab, and M. Ghazvininejad, “A review
on language models as knowledge bases,” arXiv preprint
arXiv:2204.06031 , 2022.

 

 
 [66] 
 
A. Roberts, C. Raffel, and N. Shazeer, “How much knowledge can you pack into
the parameters of a language model?” arXiv preprint arXiv:2002.08910 ,
2020.

 

 
 [67] 
 
Z. Zhang, X. Han, Z. Liu, X. Jiang, M. Sun, and Q. Liu, “Ernie: Enhanced
language representation with informative entities,” in Proc. 57th
Annu. Meeting Assoc. Comput. Linguistics , 2019, pp. 1441–1451.

 

 
 [68] 
 
M. E. Peters et al. , “Knowledge enhanced contextual word
representations,” in Proc. 2019 Conf. Empir. Methods Natural Lang.
Process. and 9th Int. Joint Conf. Natural Lang. Process. , 2019, pp. 43–54.

 

 
 [69] 
 
W. Liu, P. Zhou, Z. Zhao, Z. Wang, Q. Ju, H. Deng, and P. Wang, “K-bert:
Enabling language representation with knowledge graph,” in Proc. AAAI
Conf. Artif. Intell. , vol. 34, no. 03, 2020, pp. 2901–2908.

 

 
 [70] 
 
P. Ke, H. Ji, S. Liu, X. Zhu, and M. Huang, “Sentilare: Sentiment-aware
language representation learning with linguistic knowledge,” arXiv
preprint arXiv:1911.02493 , 2019.

 

 
 [71] 
 
J. Bai et al. , “Syntax-bert: Improving pre-trained transformers with
syntax trees,” arXiv preprint arXiv:2103.04350 , 2021.

 

 
 [72] 
 
R. Wang et al. , “K-adapter: Infusing knowledge into pre-trained models
with adapters,” arXiv preprint arXiv:2002.01808 , 2020.

 

 
 [73] 
 
X. Jiang, Y. Liang, W. Chen, and N. Duan, “Xlm-k: Improving cross-lingual
language model pre-training with multilingual knowledge,” in Proc.
36th AAAI Conf. Artif. Intell. , vol. 36, no. 10, 2022, pp. 10 840–10 848.

 

 
 [74] 
 
A. Yang, Q. Wang, J. Liu, K. Liu, Y. Lyu, H. Wu, Q. She, and S. Li, “Enhancing
pre-trained language representations with rich knowledge for machine reading
comprehension,” in Proc. 57th Annu. Meeting Assoc. Comput.
Linguistics , 2019, pp. 2346–2357.

 

 
 [75] 
 
Y. Levine, B. Lenz, O. Dagan, O. Ram, D. Padnos, O. Sharir, S. Shalev-Shwartz,
A. Shashua, and Y. Shoham, “Sensebert: Driving some sense into bert,”
 arXiv preprint arXiv:1908.05646 , 2019.

 

 
 [76] 
 
A. Lauscher, I. Vulić, E. M. Ponti, A. Korhonen, and G. Glavaš,
“Specializing unsupervised pretraining models for word-level semantic
similarity,” arXiv preprint arXiv:1909.02339 , 2019.

 

 
 [77] 
 
G. A. Miller, “Wordnet: a lexical database for english,” Commun. ACM ,
vol. 38, no. 11, pp. 39–41, 1995.

 

 
 [78] 
 
K. Basu, S. C. Varanasi, F. Shakerin, J. Arias, and G. Gupta,
“Knowledge-driven natural language understanding of english text and its
applications,” in Proc. 35th AAAI Conf. Artif. Intell. , vol. 35,
no. 14, 2021, pp. 12 554–12 563.

 

 
 [79] 
 
K. Kipper, A. Korhonen, N. Ryant, and M. Palmer, “A large-scale classification
of english verbs,” Lang. Resour. Eval. , vol. 42, no. 1, pp. 21–40,
2008.

 

 
 [80] 
 
Z. Zhang, Y. Wu, H. Zhao, Z. Li, S. Zhang, X. Zhou, and X. Zhou,
“Semantics-aware bert for language understanding,” in Proc. AAAI
Conf. Artif. Intell. , vol. 34, no. 05, 2020, pp. 9628–9635.

 

 
 [81] 
 
J. Scott and G. Marshall, A dictionary of sociology . Oxford University Press, USA, 2009.

 

 
 [82] 
 
B. He, X. Jiang, J. Xiao, and Q. Liu, “Kgplm: Knowledge-guided language model
pre-training via generative and discriminative learning,” arXiv
preprint arXiv:2012.03551 , 2020.

 

 
 [83] 
 
Y. Qin et al. , “Erica: Improving entity and relation understanding for
pre-trained language models via contrastive learning,” in Proc. 59th
Annu. Meeting Assoc. Comput. Linguistics and 11th Int. Joint Conf. Natural
Lang. Process. , 2021, pp. 3350–3363.

 

 
 [84] 
 
K. Bollacker, C. Evans, P. Paritosh, T. Sturge, and J. Taylor, “Freebase: a
collaboratively created graph database for structuring human knowledge,” in
 Proc. ACM SIGMOD Int. Conf. Manage. Data , 2008, pp. 1247–1250.

 

 
 [85] 
 
S. Auer, C. Bizer, G. Kobilarov, J. Lehmann, R. Cyganiak, and Z. Ives,
“Dbpedia: A nucleus for a web of open data,” in The semantic
web . Springer, 2007, pp. 722–735.

 

 
 [86] 
 
A. Carlson, J. Betteridge, B. Kisiel, B. Settles, E. R. Hruschka, and T. M.
Mitchell, “Toward an architecture for never-ending language learning,” in
 Proc. 24th AAAI Conf. Artif. Intell. , 2010.

 

 
 [87] 
 
B. Xu, Y. Xu, J. Liang, C. Xie, B. Liang, W. Cui, and Y. Xiao, “Cn-dbpedia: A
never-ending chinese knowledge extraction system,” in Int. Conf. Ind.,
Eng. and Other Appl. of Appl. Intell. Syst.  Springer, 2017, pp. 428–438.

 

 
 [88] 
 
X. Wang et al. , “Kepler: A unified model for knowledge embedding and
pre-trained language representation,” Trans. Assoc. Comput.
Linguistics , vol. 9, pp. 176–194, 2021.

 

 
 [89] 
 
T. Sun et al. , “Colake: Contextualized language and knowledge
embedding,” in Proc. 28th Int. Conf. Comput. Linguistics , 2020, pp.
3660–3670.

 

 
 [90] 
 
Y. Su et al. , “Cokebert: Contextual knowledge selection and embedding
towards enhanced pre-trained language models,” AI Open , vol. 2, pp.
127–134, 2021.

 

 
 [91] 
 
Y. Sun et al. , “Ernie 3.0: Large-scale knowledge enhanced pre-training
for language understanding and generation,” arXiv preprint
arXiv:2107.02137 , 2021.

 

 
 [92] 
 
N. Bian, X. Han, B. Chen, and L. Sun, “Benchmarking knowledge-enhanced
commonsense question answering via knowledge-to-text transformation,” in
 Proc. 35th AAAI Conf. Artif. Intell. , vol. 35, no. 14, 2021, pp.
12 574–12 582.

 

 
 [93] 
 
Q. Liu, D. Yogatama, and P. Blunsom, “Relational memory-augmented language
models,” Trans. Assoc. Comput. Linguistics , vol. 10, pp. 555–572,
2022.

 

 
 [94] 
 
M. Kang, J. Baek, and S. J. Hwang, “Kala: Knowledge-augmented language model
adaptation,” arXiv preprint arXiv:2204.10555 , 2022.

 

 
 [95] 
 
W. Yu et al. , “Dict-bert: Enhancing language model pre-training with
dictionary,” in Findings Assoc. Comput. Linguistic , 2022, pp.
1907–1918.

 

 
 [96] 
 
W. Wang et al. , “Visually-augmented language modeling,” arXiv
preprint arXiv:2205.10178 , 2022.

 

 
 [97] 
 
H. Tan and M. Bansal, “Vokenization: Improving language understanding with
contextualized, visual-grounded supervision,” in Proc. Conf. Empir.
Methods Natural Lang. Process. , 2020, pp. 2066–2080.

 

 
 [98] 
 
Y. Sun et al. , “Ernie: Enhanced representation through knowledge
integration,” arXiv preprint arXiv:1904.09223 , 2019.

 

 
 [99] 
 
C. Rosset, C. Xiong, M. Phan, X. Song, P. Bennett, and S. Tiwary,
“Knowledge-aware language model pretraining,” arXiv preprint
arXiv:2007.00655 , 2020.

 

 
 [100] 
 
C. M. Meyer and I. Gurevych, “Wiktionary: A new rival for expert-built
lexicons? exploring the possibilities of collaborative lexicography,” in
 Electronic Lexicography . Oxford
University Press, 11 2012. [Online]. Available:
 https://doi.org/10.1093/acprof:oso/9780199654864.003.0013 

 

 
 [101] 
 
T. Zhang et al. , “Dkplm: Decomposable knowledge-enhanced pre-trained
language model for natural language understanding,” in Proc. 36th AAAI
Conf. Artif. Intell. , vol. 36, no. 10, 2022, pp. 11 703–11 711.

 

 
 [102] 
 
W. Xiong, J. Du, W. Y. Wang, and V. Stoyanov, “Pretrained encyclopedia: Weakly
supervised knowledge-pretrained language model,” arXiv preprint
arXiv:1912.09637 , 2019.

 

 
 [103] 
 
Y. Liu et al. , “Roberta: A robustly optimized bert pretraining
approach,” arXiv preprint arXiv:1907.11692 , 2019.

 

 
 [104] 
 
S. Kwon, C. Kang, J. Han, and J. Choi, “Why do masked neural language models
still need common sense knowledge?” arXiv preprint arXiv:1911.03024 ,
2019.

 

 
 [105] 
 
D. Yu, C. Zhu, Y. Yang, and M. Zeng, “Jaket: Joint pre-training of knowledge
graph and language understanding,” in Proc. 36th AAAI Conf. Artif.
Intell. , vol. 36, no. 10, 2022, pp. 11 630–11 638.

 

 
 [106] 
 
R. L. Logan IV, N. F. Liu, M. E. Peters, M. Gardner, and S. Singh, “Barack’s
wife hillary: Using knowledge-graphs for fact-aware language modeling,”
 arXiv preprint arXiv:1906.07241 , 2019.

 

 
 [107] 
 
K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval augmented
language model pre-training,” in Proc. 37th Int. Conf. Mach.
Learn.  PMLR, 2020, pp. 3929–3938.

 

 
 [108] 
 
P. Lewis et al. , “Retrieval-augmented generation for
knowledge-intensive nlp tasks,” in Proc. Int. Conf. Neural Inf.
Process. Syst. , vol. 33, 2020, pp. 9459–9474.

 

 
 [109] 
 
D. Wilmot and F. Keller, “Memory and knowledge augmented language models for
inferring salience in long-form stories,” in Proc. Conf. Empir.
Methods Natural Lang. Process. , 2021, pp. 851–865.

 

 
 [110] 
 
N. Poerner, U. Waltinger, and H. Schütze, “E-bert: Efficient-yet-effective
entity embeddings for bert,” in Findings Assoc. Comput. Linguistics ,
2020, pp. 803–818.

 

 
 [111] 
 
[Online]. Available:
 https://code.google.com/archive/p/relation-extraction-corpus/ 

 

 
 [112] 
 
H. Elsahar et al. , “T-rex: A large scale alignment of natural language
with knowledge base triples,” in Proc. 11th Int. Conf. Lang. Resour.
Eval. , 2018. [Online]. Available:
 https://aclanthology.org/L18-1544.pdf 

 

 
 [113] 
 
P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “Squad: 100,000+ questions
for machine comprehension of text,” in Proc. Conf. Empir. Methods
Natural Lang. Process. , 2016, pp. 2383–2392.

 

 
 [114] 
 
T. Févry, L. B. Soares, N. FitzGerald, E. Choi, and T. Kwiatkowski,
“Entities as experts: Sparse memory access with entity supervision,” in
 Proc. 2020 Conf. Empir. Methods Natural Lang. Process. , 2020, pp.
4937–4951.

 

 
 [115] 
 
Z. Jiang, F. F. Xu, J. Araki, and G. Neubig, “How can we know what language
models know?” Trans. Assoc. Comput. Linguistics , vol. 8, pp.
423–438, 2020.

 

 
 [116] 
 
L. Adolphs, S. Dhuliawala, and T. Hofmann, “How to query language models?”
 arXiv preprint arXiv:2108.01928 , 2021.

 

 
 [117] 
 
T. Shin, Y. Razeghi, R. L. Logan IV, E. Wallace, and S. Singh, “Autoprompt:
Eliciting knowledge from language models with automatically generated
prompts,” in Proc. Conf. Empir. Methods Natural Lang. Process. , 2020,
pp. 4222–4235.

 

 
 [118] 
 
Z. Meng, F. Liu, E. Shareghi, Y. Su, C. Collins, and N. Collier,
“Rewire-then-probe: A contrastive recipe for probing biomedical knowledge of
pre-trained language models,” in Proc. 60th Annu. Meeting Assoc.
Comput. Linguistics , 2022, pp. 4798–4810.

 

 
 [119] 
 
A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman, “Glue: A
multi-task benchmark and analysis platform for natural language
understanding,” in Proc. EMNLP Workshop BlackboxNLP , 2018, pp.
353–355.

 

 
 [120] 
 
F. Petroni et al. , “Kilt: A benchmark for knowledge intensive language
tasks,” in Proc. Conference North Amer. Chapter Assoc. Comput.
Linguistics: Hum. Lang. Technol. , 2021, pp. 2523–2544.

 

 
 [121] 
 
M. F. M. Chowdhury, M. Glass, G. Rossiello, A. Gliozzo, and
N. Mihindukulasooriya, “Kgi: An integrated framework for knowledge intensive
language tasks,” arXiv preprint arXiv:2204.03985 , 2022.

 

 
 [122] 
 
X. Ling, S. Singh, and D. S. Weld, “Design challenges for entity linking,”
 Trans. Assoc. Comput. Linguistics , vol. 3, pp. 315–328, 2015.

 

 
 [123] 
 
E. Choi, O. Levy, Y. Choi, and L. Zettlemoyer, “Ultra-fine entity typing,” in
 Proc. 56th Annu. Meeting Assoc. Comput. Linguistics , 2018, pp. 87–96.

 

 
 [124] 
 
I. Yamada, A. Asai, H. Shindo, H. Takeda, and Y. Matsumoto, “Luke: deep
contextualized entity representations with entity-aware self-attention,” in
 Proc. 2020 Conf. Empir. Methods Natural Lang. Process. , 2020, pp.
6442–6454.

 

 
 [125] 
 
Ö. Uzuner, B. R. South, S. Shen, and S. L. DuVall, “2010 i2b2/va challenge
on concepts, assertions, and relations in clinical text,” J. Amer.
Med. Inform. Assoc. , vol. 18, no. 5, pp. 552–556, 2011.

 

 
 [126] 
 
J.-D. Kim, T. Ohta, Y. Tsuruoka, Y. Tateisi, and N. Collier, “Introduction to
the bio-entity recognition task at jnlpba,” in Proc. Int. Joint
Workshop Natural Lang. Process. Biomedicine and its Appl.  Citeseer, 2004, pp. 70–75.

 

 
 [127] 
 
J. Li et al. , “Biocreative v cdr task corpus: A resource for chemical
disease relation extraction,” Database(Oxford) , vol. 2016, 2016.

 

 
 [128] 
 
S. Li, M. Sridhar, C. S. Prakash, J. Cao, W. Hamza, and J. McAuley,
“Instilling type knowledge in language models via multi-task qa,” in
 Findings Assoc. Comput. Linguistics: NAACL , 2022, pp. 594–603.

 

 
 [129] 
 
G.-A. Levow, “The third international chinese language processing bakeoff:
Word segmentation and named entity recognition,” in Proc. Fifth SIGHAN
Workshop Chinese Lang. Process. , 2016, pp. 108–117.

 

 
 [130] 
 
Y. Sun et al. , “Ernie 2.0: A continual pre-training framework for
language understanding,” in Proc. 34th AAAI Conf. Artif. Intell. ,
vol. 34, no. 05, 2020, pp. 8968–8975.

 

 
 [131] 
 
E. F. Sang and F. De Meulder, “Introduction to the conll-2003 shared task:
Language-independent named entity recognition,” in Proc. Conference
North Amer. Assoc. Comput. Linguistics , 2003, pp. 142–147.

 

 
 [132] 
 
L. Linlin, L. Xin, H. Ruidan, B. Lidong, J. Shafiq, and S. Luo, “Knowledge
based multilingual language model,” arXiv preprint arXiv:2111.10962 ,
2021.

 

 
 [133] 
 
Ö. Uzuner, Y. Luo, and P. Szolovits, “Evaluating the state-of-the-art in
automatic de-identification,” J. Amer. Med. Inform. Assoc. , vol. 14,
no. 5, pp. 550–563, 2007.

 

 
 [134] 
 
A. Stubbs, C. Kotfila, and Ö. Uzuner, “Automated systems for the
de-identification of longitudinal clinical narratives: Overview of 2014
i2b2/uthealth shared task track 1,” J. Biomed. Inform. , vol. 58, pp.
S11–S19, 2015.

 

 
 [135] 
 
[Online]. Available: https://portal.dxy.cn/ 

 

 
 [136] 
 
[Online]. Available: https://www.biendata.xyz/competition/CCKS2017_2/ 

 

 
 [137] 
 
R. I. Doğan, R. Leaman, and Z. Lu, “Ncbi disease corpus: a resource for
disease name recognition and concept normalization,” J. Biomed.
Inform. , vol. 47, pp. 1–10, 2014.

 

 
 [138] 
 
L. Smith et al. , “Overview of biocreative ii gene mention
recognition,” Genome Biol. , vol. 9, no. 2, pp. 1–19, 2008.

 

 
 [139] 
 
J. Lee et al. , “Biobert: a pre-trained biomedical language
representation model for biomedical text mining,” Bioinformatics ,
vol. 36, no. 4, pp. 1234–1240, 2020.

 

 
 [140] 
 
[Online]. Available: https://embedding.github.io/evaluation/ 

 

 
 [141] 
 
L. Derczynski, E. Nichols, M. van Erp, and N. Limsopatham, “Results of the
wnut2017 shared task on novel and emerging entity recognition,” in
 Proc. 3rd Workshop Noisy User-generated Text , 2017, pp. 140–147.

 

 
 [142] 
 
X. Pan, B. Zhang, J. May, J. Nothman, K. Knight, and H. Ji, “Cross-lingual
name tagging and linking for 282 languages,” in Proc. 55th Annu.
Meeting Assoc. Comput. Linguistics , vol. 1, 2017, pp. 1946–1958.

 

 
 [143] 
 
Y. Zhang, V. Zhong, D. Chen, G. Angeli, and C. D. Manning, “Position-aware
attention and supervised data improve slot filling,” in Proc. 2017
Conf. Empir. Methods Lang. Process. , 2017, pp. 35–45.

 

 
 [144] 
 
M. Glass, G. Rossiello, M. F. M. Chowdhury, and A. Gliozzo, “Robust retrieval
augmented generation for zero-shot slot filling,” in Proc. 2021 Conf.
Emplir. Methods Natural Lang. Process. , 2021, pp. 1939–1949.

 

 
 [145] 
 
X. Han et al. , “Fewrel: A large-scale supervised few-shot relation
classification dataset with state-of-the-art evaluation,” in Proc.
2018 Conf. Empir. Methods Natural Lang. Process. , 2018, pp. 4803–4809.

 

 
 [146] 
 
À. Bravo, J. Piñero, N. Queralt-Rosinach, M. Rautschka, and L. I.
Furlong, “Extraction of relations between genes and diseases from text and
large-scale data analysis: implications for translational research,”
 BMC Bioinformatics , vol. 16, no. 1, pp. 1–17, 2015.

 

 
 [147] 
 
E. M. Van Mulligen et al. , “The eu-adr corpus: annotated drugs,
diseases, targets, and their relationships,” J. Biomed. Inform. ,
vol. 45, no. 5, pp. 879–884, 2012.

 

 
 [148] 
 
[Online]. Available: http://www.cips-chip.org.cn 

 

 
 [149] 
 
M. Herrero-Zazo, I. Segura-Bedmar, P. Martínez, and T. Declerck, “The ddi
corpus: An annotated corpus with pharmacological substances and drug–drug
interactions,” J. Biomed. Inform. , vol. 46, no. 5, pp. 914–920,
2013.

 

 
 [150] 
 
M. Krallinger et al. , “Overview of the biocreative vi chemical-protein
interaction track,” in Proc. sixth BioCreative Challenge Eval.
Workshop , vol. 1, 2017, pp. 141–146.

 

 
 [151] 
 
R. Socher et al. , “Recursive deep models for semantic compositionality
over a sentiment treebank,” in Proc. Conf. Empir. Methods Natural
Lang. Process. , 2013, pp. 1631–1642.

 

 
 [152] 
 
X. Zhang, J. Zhao, and Y. LeCun, “Character-level convolutional networks for
text classification,” in Proc. Int. Conf. Neural Inf. Process. Syst. ,
2015.

 

 
 [153] 
 
M. Pontiki, D. Galanis, J. Pavlopoulos, H. Papageorgiou, I. Androutsopoulos,
and S. Manandhar, “Semeval-2014 task 4: Aspect based sentiment analysis,”
in Proc. 8th Int Workshop Semantic Eval. , 2014.

 

 
 [154] 
 
A. Fisch, A. Talmor, R. Jia, M. Seo, E. Choi, and D. Chen, “Mrqa 2019 shared
task: Evaluating generalization in reading comprehension,” in Proc.
2nd Workshop Mach. Reading Question Answering , 2019, pp. 1–13.

 

 
 [155] 
 
M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, “Triviaqa: A large scale
distantly supervised challenge dataset for reading comprehension,” in
 Proc. 55th Annu. Meeting Assoc. Comput. Linguistics , 2017, pp.
1601–1611.

 

 
 [156] 
 
M. Dunn, L. Sagun, M. Higgins, V. U. Guney, V. Cirik, and K. Cho, “Searchqa: A
new q a dataset augmented with context from a search engine,” arXiv
preprint arXiv:1704.05179 , 2017.

 

 
 [157] 
 
T. Kwiatkowski et al. , “Natural questions: A benchmark for question
answering research,” Trans. Assoc. Comput. Linguistics , vol. 7, pp.
452–466, 2019.

 

 
 [158] 
 
J. Berant, A. Chou, R. Frostig, and P. Liang, “Semantic parsing on freebase
from question-answer pairs,” in Proc. Conf. Empir. Methods Natural
Lang. Process. , 2013, pp. 1533–1544.

 

 
 [159] 
 
A. Talmor, J. Herzig, N. Lourie, and J. Berant, “Commonsenseqa: A question
answering challenge targeting commonsense knowledge,” in Proc.
Conference North Amer. Chapter Assoc. Comput. Linguistics , 2019, pp.
4149–4158.

 

 
 [160] 
 
T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, “Can a suit of armor conduct
electricity? a new dataset for open book question answering,” in Proc.
Conf. Empir. Methods Natural Lang. Process. , 2018, pp. 2381–2391.

 

 
 [161] 
 
L. Huang, R. Le Bras, C. Bhagavatula, and Y. Choi, “Cosmos qa: Machine reading
comprehension with contextual commonsense reasoning,” in Proc. 57th
Annu. Meeting Assoc. Comput. Linguistics and 9th Int. Joint Conf. Artif.
Intell. , 2019, pp. 2391–2401.

 

 
 [162] 
 
A. H. Li and A. Sethy, “Knowledge enhanced attention for robust natural
language inference,” arXiv preprint arXiv:1909.00102 , 2019.

 

 
 [163] 
 
D. Andor, L. He, K. Lee, and E. Pitler, “Giving bert a calculator: Finding
operations and arguments with reading comprehension,” in Proc. 2019
Conf. Empir. Methods Natural Lang. Process. and 9th Int. Joint Conf. Natural
Lang. Process. , 2019, pp. 5947–5952.

 

 
 [164] 
 
T. Dettmers, P. Minervini, P. Stenetorp, and S. Riedel, “Convolutional 2d
knowledge graph embeddings,” in Proc. 32th AAAI Conf. Artif. Intell. ,
2018.

 

 
 [165] 
 
X. Li, A. Taheri, L. Tu, and K. Gimpel, “Commonsense knowledge base
completion,” in Proc. 57th Annu. Meeting Assoc. Comput. Linguistics ,
2016, pp. 1445–1455.

 

 
 [166] 
 
P. Singh, T. Lin, E. T. Mueller, G. Lim, T. Perkins, and W. L. Zhu, “Open mind
common sense: Knowledge acquisition from the general public,” in OTM
Confederated Int. Conf., “On the Move to Meaningful Internet
Systems” . Berlin, Germany:
Springer, 2002, pp. 1223–1237.

 

 
 [167] 
 
L. Yao, C. Mao, and Y. Luo, “Kg-bert: Bert for knowledge graph completion,”
 arXiv preprint arXiv:1909.03193 , 2019.

 

 
 [168] 
 
T. Zhu, Y. Wang, H. Li, Y. Wu, X. He, and B. Zhou, “Multimodal joint attribute
prediction and value extraction for ecommerce product,” in Proc. Conf.
Empir. Methods Natural Lang. Process. , 2020, pp. 2129–2139.

 

 
 [169] 
 
B. Y. Lin et al. , “Commongen: A constrained text generation challenge
for generative commonsense reasoning,” in Findings Assoc. Comput.
Linguistics , 2020, pp. 1823–1840.

 

 
 [170] 
 
N. Mostafazadeh et al. , “A corpus and cloze evaluation for deeper
understanding of commonsense stories,” in Proc. Conference North Amer.
Chapter Assoc. Comput. Linguistics , 2016, pp. 839–849.

 

 
 [171] 
 
J. Guan, Y. Wang, and M. Huang, “Story ending generation with incremental
encoding and commonsense knowledge,” in Proc. 33th AAAI Conf. Artif.
Intell. , vol. 33, no. 01, 2019, pp. 6473–6480.

 

 
 [172] 
 
X. Zhao, W. Wu, C. Xu, C. Tao, D. Zhao, and R. Yan, “Knowledge-grounded
dialogue generation with pre-trained language models,” arXiv preprint
arXiv:2010.08824 , 2020.

 

 
 [173] 
 
E. Dinan, S. Roller, K. Shuster, A. Fan, M. Auli, and J. Weston, “Wizard of
wikipedia: Knowledge-powered conversational agents,” arXiv preprint
arXiv:1811.01241 , 2018.

 

 
 [174] 
 
K. Zhou, S. Prabhumoye, and A. W. Black, “A dataset for document grounded
conversations,” in Proc. 2018 Conf. Empir. Methods Natural Lang.
Process. , 2018, pp. 708–713.

 

 
 [175] 
 
T. Nguyen, M. Rosenberg, X. Song, J. Gao, S. Tiwary, R. Majumder, and L. Deng,
“Ms marco: A human generated machine reading comprehension dataset,” in
 Proc. Workshop on Cognitive Comput.: Integrating neural and symbolic
approaches 2016 co-located with the 30th Annu. Conf. Neural Inf. Process.
Syst. , 2016. [Online]. Available: http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf 

 

 
 [176] 
 
S. S. Dasgupta, S. N. Ray, and P. Talukdar, “Hyte: Hyperplane-based temporally
aware knowledge graph embedding,” in Proc. Conf. Empir. Methods
Natural Lang. Process. , 2018, pp. 2001–2011.

 

 
 [177] 
 
L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang,
S. Agarwal, K. Slama, A. Ray et al. , “Training language models to
follow instructions with human feedback,” arXiv preprint
arXiv:2203.02155 , 2022.

 

 
 [178] 
 
Y. Hou, G. Fu, and M. Sachan, “Understanding the integration of knowledge in
language models with graph convolutions,” arXiv preprint
arXiv:2202.00964 , 2022.

 

 
 [179] 
 
P. Verga, H. Sun, L. B. Soares, and W. W. Cohen, “Facts as experts: Adaptable
and interpretable neural memory over symbolic knowledge,” arXiv
preprint arXiv:2007.00849 , 2020.

 

 
 [180] 
 
Y. Cheng, D. Wang, P. Zhou, and T. Zhang, “Model compression and acceleration
for deep neural networks: The principles, progress, and challenges,”
 IEEE Signal Process. Mag. , vol. 35, no. 1, pp. 126–136, 2018.

 

 
 [181] 
 
S. Shen et al. , “Q-bert: Hessian based ultra low precision quantization
of bert,” in Proc. 34th AAAI Conf. Artif. Intell. , vol. 34, no. 05,
2020, pp. 8815–8821.

 

 
 [182] 
 
G. Hinton, O. Vinyals, J. Dean et al. , “Distilling the knowledge in a
neural network,” arXiv preprint arXiv:1503.02531 , vol. 2, no. 7,
2015.

 

 
 [183] 
 
Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, “Albert: A
lite bert for self-supervised learning of language representations,”
 arXiv preprint arXiv:1909.11942 , 2019.

 

 
 [184] 
 
W. Kryściński, B. McCann, C. Xiong, and R. Socher, “Evaluating the
factual consistency of abstractive text summarization,” in Proc. Conf.
Empir. Methods Natural Lang. Process. , 2020, pp. 9332–9346.

 

 
 [185] 
 
S. Hu, N. Ding, H. Wang, Z. Liu, J. Wang, J. Li, W. Wu, and M. Sun,
“Knowledgeable prompt-tuning: Incorporating knowledge into prompt verbalizer
for text classification,” in ACL , 2022, pp. 2225–2240.

 

 
 [186] 
 
X. Chen, N. Zhang, X. Xie, S. Deng, Y. Yao, C. Tan, F. Huang, L. Si, and
H. Chen, “Knowprompt: Knowledge-aware prompt-tuning with synergistic
optimization for relation extraction,” in WWW , 2022, pp. 2778–2788.

 

 
 [187] 
 
H. Ye, N. Zhang, S. Deng, X. Chen, H. Chen, F. Xiong, X. Chen, and H. Chen,
“Ontology-enhanced prompt-tuning for few-shot learning,” in WWW ,
2022, pp. 778–787.

 

 
 [188] 
 
B. Ryan, D. Minh-Hoang, H. Fabian, H. Yuan, M.-P. Albert, and S. Vijay,
“Improving language model predictions via prompts enriched with knowledge
graphs,” in ISWC , 2022, pp. 1–10.

 

 
 [189] 
 
H. Schuff, H.-Y. Yang, H. Adel, and N. T. Vu, “Does external knowledge help
explainable natural language inference? automatic evaluation vs. human
ratings,” in Proc. 4th BlackboxNLP Workshop Analyzing and interpreting
Neural Netw for NLP , 2021, pp. 26–41.

 

 
 [190] 
 
E. Akyürek et al. , “Tracing knowledge in language models back to
the training data,” arXiv preprint arXiv:2205.11482 , 2022.

 

 
 [191] 
 
N. De Cao, W. Aziz, and I. Titov, “Editing factual knowledge in language
models,” in Proc. Conf. Empir. Methods Natural Lang. Process. , 2021,
pp. 6491–6506.

 

 
 [192] 
 
K. Meng, D. Bau, A. Andonian, and Y. Belinkov, “Locating and editing factual
associations in gpt,” arXiv preprint arXiv:2202.05262 , 2022.

 

 
 
 
 
 
 
 | 
 
 
 Chaoqi Zhen 
received the bachelor’s degreein computer science and technology from theBeijing Unibersityof Posts and Telecommuni-cations, Beijing, China, in 2020. She is current-ly working towrd the master’s degree in com-puter science and technology from the BeijingUnibersity of Posts and Telecommunications,Beijing, China. Her research interests lie innatural language processingandknowledgerepresentation learning. 
 | 

 
 
 
 
 | 
 
 
 Yanlei Shang 
received the PhD degree in computer science and Technology from Beijing Unibersity of Posts and Telecommunications, Beijing, China, in 2006. He is currently working as a professor with the school of computer science, Beijing Unibersity of Posts and Telecommunica-tions, Beijing, China. His research interests mainly include cloud computing, big data storage and analytics, artificial intelligence and deep learning. 
 | 

 
 
 
 
 | 
 
 
 Xiangyu Liu 
received the bachelor’s degree in computer science and technology from the Beijing Unibersity of Posts and Telecom-munications, Beijing, China, in 2020. She is currently working towrd the master’s degree in computer science and technology from the Beijing Unibersity of Posts and Telecommunications, Beijing, China. His research interests lie in natural language processing, and knowledge representation learning. 
 | 

 
 
 
 
 | 
 
 
 Yifei Li 
received the bachelor’s degree in computer science and technology from the Beijing Unibersity of Posts and Telecom-munications, Bei-jing, China, in 2020. She is currently working towrd the master’s degree in computer science and technology from the Beijing Uni-bersity of Posts and Telecommunications, Beijing, China. Her re-search interests lie in natural language processing, and knowledge acquisition. 
 | 

 
 
 
 
 | 
 
 
 Yong Chen 
is now an associate professor in school of computer science, Beijing University of Posts and Telecommunications, Beijing, China. He received the Ph.D. degree in computer science and engineering from Beihang University (BUAA), Beijing, China, in 2019; and worked as a “Boya” postdoctoral with the Key Lab of Machine Perception, School of Electronics Engineering and Computer Science, Peking University, Beijing, China, from 2019 to 2021. He has been funded as a visiting PhD student at Birkbeck and UCL from January 2018 to January 2019. His research interests include machine learning, data mining, big data, numerical optimization and interpretable differential calculation. 
 | 

 
 
 
 
 | 
 
 
 Dell Zhang 
currently leads the Applied Research team at Thomson Reuters Labs in London, UK.
Prior to this role, he was a Tech Lead Manager at ByteDance AI Lab and TikTok UK, a Staff Research Scientist at Blue Prism AI Labs, and a Reader in Computer Science at Birkbeck College, University of London.
He is a Senior Member of ACM, a Senior Member of IEEE, and a Fellow of RSS.
He got his PhD from the Southeast University (SEU) in Nanjing, China, and then worked as a Research Fellow at the Singapore-MIT Alliance (SMA) until he moved to the UK in 2005. His main research interests include Machine Learning, Information Retrieval, and Natural Language Processing.
He has published 110+ papers, graduated 11 PhD students, received multiple best paper awards, and won several prizes from international data science competitions. 
 |