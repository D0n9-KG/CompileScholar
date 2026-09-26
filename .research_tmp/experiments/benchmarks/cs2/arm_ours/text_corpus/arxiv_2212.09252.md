Mind the Knowledge Gap:A Survey of Knowledge-enhanced Dialogue Systems 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2212.09252v2 [cs.CL] 20 Dec 2022 
 
 

# Mind the Knowledge Gap:
 A Survey of Knowledge-enhanced Dialogue Systems

 
 
 ⋆ Sagi Shaier
 
 Email:  sagi.shaier@colorado.edu 
 
    
 Lawrence Hunter
 
 Affiliation:  University of Colorado Denver
 
 Email:  katharina.kann@colorado.edu 
 
 Affiliation:  larry.hunter@cuanschutz.edu
 
    
 ⋆ Katharina Kann ⋆ University of Colorado Boulder
 

 Abstract 
 
 Many dialogue systems (DSs) lack characteristics humans have, such as emotion perception, factuality, and informativeness. Enhancing DSs with knowledge alleviates this problem, but, as many ways of doing so exist, keeping track of all proposed methods is difficult. Here, we present the first survey of knowledge-enhanced DSs. We define three categories of systems – internal, external, and hybrid – based on the knowledge they use. We survey the motivation for enhancing DSs with knowledge, used datasets, and methods for knowledge search, knowledge encoding, and knowledge incorporation. Finally, we propose how to improve existing systems based on theories from linguistics and cognitive science.

 
 
 
 
 

 
 

## 1 Introduction

 
 Dialogue systems (DSs) are designed to communicate with humans, either for entertainment or for task completion.
While the movement from rule-based systems to machine learning models has improved the diversity of the generated responses, they still lack many attributes human language exhibits, such as emotion perception, factuality, and informativeness. This may in turn result in user frustration, or even harm, if the requested information returned by the DS is fictitious.

 
 
 Previous work has found that incorporating knowledge (e.g., as knowledge graphs) into DSs alleviates many such problems, e.g., by boosting informativeness
 Santhanam et al. (2020) , emotional intelligence Zhou et al. (2018a) ; Liang et al. (2021) , or factuality Dziri et al. (2021) . However, an abundance of knowledge forms exist, many of considerable size. Hence, numerous methods for finding, encoding, and incorporating the relevant information from the knowledge source into the dialogue have been proposed. To the best of our knowledge, we provide the first survey to overview such works.

 
 
 Figure 1: The steps conducted by dialogue systems. If knowledge is used, its type (knowledge forms, Section 2.2 ) needs to be considered. This will determine if the system is internal, external, or hybrid. Next, we search for the relevant knowledge in the often massive knowledge database (knowledge search, Section 3 ). Then, encoding of the relevant knowledge is needed (knowledge encoding, Section 4 ) followed by incorporating the output encoding into the encoded dialogue history (knowledge incorporation, Section 5 ). Lastly and similarly to the vanilla systems that do not use knowledge, the system generate or retrieve a response based on the encoded representation. 
 
 
 More specifically, we group knowledge-enhanced DSs based on the type of knowledge they use: internal , external , and hybrid . We discuss motivations for enhancing DSs with knowledge, typically used datasets, and, based on how humans enhance dialogues with knowledge, present existing methods for knowledge search , knowledge encoding , and knowledge incorporation . Lastly, using theories derived from the human memory system, we propose possible avenues to improve knowledge-enhanced DSs.

 
 
 

## 2 Background

 

### 2.1 Knowledge

 
 In the context of this paper, we define knowledge as any information that goes beyond the pure surface form of the utterances in the dialogue history. For example, if the dialogue history is

 
 
 
 
 1) Good morning, how can I help you?

 2) OMG I already told you!!! Where is the nearest bank??

 
 
 
 
 The pure surface form is simply the characters or words (i.e., "Good", "morning", …, "??"). In comparison, knowledge is information in various forms such as knowledge graph (KG) triples about the nearest bank (e.g., [Bank_of_America, distance, 0.1]), the user’s emotional category (e.g., angry), an abstract meaning representation (AMR) describing the semantic association between words (e.g., "where" → \rightarrow question), or free text describing relevant information (e.g., "there are several banks in Colorado…").

 
 
 

### 2.2 Knowledge Forms

 
 Surveying existing work, we identify three broad categories of knowledge (according to our definition) that DS have been enhanced with, and
structure the survey according to these categories. That is, the existing work can be grouped into the category of knowledge that they use (i.e., internal, external, or hybrid), where each knowledge category is based on the location from which the knowledge was taken (e.g., KGs, Wikipedia articles, the utterance itself).

 
 
 Internal knowledge is implicit information that is extracted from the dialogue and made explicit. In simple words, this is information that can be seen as "between the lines" of the dialogue. It can be structured, such as AMR graphs or dependency trees (e.g., "where" → \rightarrow question), or unstructured, such as conversation topics (e.g., direction assistance), or the speaker’s emotional state (e.g., angry). While these may require predefined concepts (e.g., emotional categories, linguistic structures, topics), the actual information is found within the dialogue.

 
 
 In contrast, external knowledge is information that is not extracted from the dialogue and exists in an outside source. It can be structured, such as KGs and knowledge bases (KBs), or unstructured, such as free text, and take the form of triples, table entries, or strings.

 
 
 The third, hybrid knowledge , is a combination of internal and external knowledge, such as emotion categories and KGs, topic representations and KBs, or intent vectors, KGs, and free text.

 
 
 We call enhanced DS that use internal or external knowledge internal and external approaches, respectively. Those that use a combination of both are called hybrid approaches.

 
 
 

### 2.3 Dialogue Systems

 
 There are two classes of DS: task-oriented , which are designed to assist with task completion, and chit-chat , designed mainly for entertainment. Each class may be based on response retrieval or response generation.
In the retrieval method, the DS is given a corpus of predefined responses and is tasked with selecting the most appropriate Tao et al. (2021) , while in the generative method the DS generates a response token by token Chen et al. (2017) .

 
 
 The simplest approach for retrieval systems is using 2 encoders: one to encode each response and a second to encode the dialogue, followed by a similarity function between each encoded response and the dialogue. For generative systems, one may use an encoder-decoder setup, where the decoder is tasked to generate a token at a time conditioning on the encoded dialogue and the previously generated tokens. For more information about DS we refer the interested reader to Jurafsky and Martin (2008) or McTear (2021) .

 
 
 While some DS use knowledge more often, here, we will not focus on the distinction between task and non-task oriented systems or open and closed domain. Instead we will focus on knowledge and how different knowledge sources have been used.

 
 
 

### 2.4 Knowledge Enhancement Process

 
 The breakdown we present can be seen in Figure 1 . The top row shows the three knowledge-enhancement approaches discussed before (internal, external, and hybrid), in addition to the vanilla approach, which does not use knowledge.

 
 
 Given our example dialogue history in Section 2.1 , the vanilla approach encodes the utterances followed by the steps described in Section 2.3 regarding generation or retrieval systems. However, if knowledge is used, the type of knowledge must first be chosen (Section 2.2 ). Then, as the knowledge source may contain million data-points (e.g., KG triples), hindering the use of its entirety simultaneously, we use a method to find relevant parts. For example, by checking whether each triple has the word "bank" and retrieving some number of them. We discuss additional knowledge search methods in Section 3 .

 
 
 Once we find the relevant information in our knowledge source (e.g., 5 triples representing the nearest banks) we need to encode it, as it is often presented in a different form than the utterance (e.g., KG triples versus free text). For example, by encoding both the dialogue text and the triples into similar size vectors. We discuss more knowledge encoding methods in Section 4 .

 
 
 Once both the knowledge and dialogue history are encoded, we need to incorporate the knowledge representations into the encoded utterance. For example, by summing both vector representations together. We discuss knowledge incorporation methods further in Section 5 .

 
 
 Lastly and similarly to the final stage in the vanilla approach, we generate or retrieve a response given our final representation.

 
 
 

### 2.5 Motivation for Enhancing Dialogue Systems With Knowledge

 
 While the vanilla approach assumes that all the information to generate or retrieve a response is provided in the dialogue, this is often not the case. For example, some suggest that DS have limited ability to represent core semantics, such as ignoring relevant entities Bai et al. (2021) ; Chen et al. (2016) ; Su et al. (2018) . DS may also have difficulties with perception and expressions of emotions Liang et al. (2021) ; Zhou et al. (2018a) ; Compton et al. (2021) or with being specialized to certain users or domains Tigunova et al. (2019) ; Gu et al. (2020) ; Peng et al. (2020) ; Moghe et al. (2020) . For example, in the medical domain DS must generate meaningful responses with the correct entities, which they often not Li et al. (2021a) .

 
 
 Another crucial piece that is often missing from DS responses is factuality Komeili et al. (2021) ; Chaudhuri et al. (2019a) , which hinders them from being used in domains where human lives are at stake. DS often also generate responses that are too short, dull, or uninformative Wu et al. (2020) ; Yavuz et al. (2019) ; Santhanam et al. (2020) ; Parthasarathi and Pineau (2018) ; Ye et al. (2019) ; Moon et al. (2019a) ; Lian et al. (2019) ; Yang et al. (2020b) ; Luo et al. (2021) , and can be improved on answering simple factoid QA Yin et al. (2016a) which can be seen as a single-turn dialogue, or their reasoning skills Kassawat et al. (2020) .

 
 
 DS also often have difficulties with implicit knowledge perception or linguistic features Li et al. (2020b) ; Zhou et al. (2021b) ; Kumar et al. (2020) ; Wang et al. (2021b) , responding to user requests that are beyond the scope of existing APIs Jin et al. (2021a) , dealing with out-of-vocabulary entities Zhu et al. (2017) , and performing well given limited resources Zhao et al. (2020a) . Such challenges have been tackled by enhancing DS with knowledge, as will be seen shortly.

 
 
 
 

## 3 Knowledge Search

 
 We define knowledge search as the methods for finding and extracting the relevant knowledge from the data source (e.g., KG). Although the internal approach is mostly based on extraction (e.g., extracting AMR graphs), the external approach could be using both. However, the search aspect is often more important, but difficult. The following two subsections describe the internal and external approaches taken for knowledge search, with the hybrid approaches split into their external and internal parts and written within.

 
 

### 3.1 Internal Knowledge Search

 
 The methods for internal knowledge search can be seen in Table 1 .
Many focus on extracting linguistic features from the dialogue, such as AMR graphs, dependency trees, or POS tags. These are usually done using an existing library (e.g., NLTK, spaCy). In comparison, for emotional information, speaking style, or user intent, existing work often train a system to classify or extract the knowledge.
Additionally, for a given knowledge form, e.g., domains or entities, several possible search methods exist, such as string, n-gram matching,
or a classifier that tags relevant words.

 
 
 Table 1: Internal Knowledge Search Table 
 
 
 
 
 
 Knowledge Form 
 | 
 
 
 Name 
 | 
 
 
 Method 
 | 

 
 
 
 AMR graph 
 | 
 
 
 Bai et al. (2021) ; Chen et al. (2016) 
 | 
 
 
 AMR parser 
 | 

 
 
 
 Dependency tree 
 | 
 
 
 Wang et al. (2020) ; Chen et al. (2016) ; Yang et al. (2020a) 
 | 
 
 
 Dependency parser 
 | 

 
 
 
 Part-of-speech tagging 
 | 
 
 
 Su et al. (2018) ; Adolphs et al. (2021) 
 | 
 
 
 POS tagger 
 | 

 
 
 
 Emotions 
 | 
 
 
 Liang et al. (2021) ; Compton et al. (2021) ; Zhou et al. (2018a) 
 | 
 
 
 Emotion classifier 
 | 

 
 
 
 Specific domains 
 | 
 
 
 Luo et al. (2021) 
 | 
 
 
 Topic classifier 
 | 

 
 | 
 
 
 Chaudhuri et al. (2018) 
 | 
 
 
 String matching 
 | 

 
 | 
 
 
 Jin et al. (2021b) 
 | 
 
 
 Domain classifier 
 | 

 
 
 
 Entities 
 | 
 | 
 
 
 Fuzzy n-gram matching 
 | 

 
 | 
 
 
 Li et al. (2021a) ; Chaudhuri et al. (2021) 
 | 
 
 
 Entity classifier 
 | 

 
 
 
 User intent 
 | 
 
 
 Cho et al. (2021) 
 | 
 
 
 Expression matching, classifier 
 | 

 
 | 
 
 
 Kassawat et al. (2020) 
 | 
 
 
 Intent classifier 
 | 

 
 | 
 
 
 Zhu et al. (2017) 
 | 
 
 
 GRU encoder 
 | 

 
 
 
 Speaking style 
 | 
 
 
 Sun et al. (2021) 
 | 

 

 
 
 

### 3.2 External Knowledge Search

 
 The methods for external knowledge search can be seen in Table 2 .
In contrast to the string methods which examine whether the dialogue and knowledge source share strings in a binary fashion, such as having an exact string match, or non-stopwords match, the similarity methods use a score between 0 and 1 to evaluate how similar the knowledge and dialogue are.
Attention mechanisms also use a scoring approach, but solely use attention without a specialized model, normally on embedding vectors.
There are also methods that are highly specific to the type of knowledge, such as graph linking methods which links the entities found in dialogue with the corresponding graph entities, and query methods, which use a specialized syntax, such as SQL. Some methods can avoid the search entirely, for example, by using the knowledge as labels in response generation and by that "injecting" it into the parameters of the model, or by using the entire KB in response generation.

 
 
 Table 2: External Knowledge Search Table 
 
 
 
 
 
 Method Type 
 | 
 
 
 Knowledge Form 
 | 
 
 
 Name 
 | 
 
 
 Method 
 | 

 
 
 
 
 
 Similarity 
 | 
 
 
 Knowledge graph 
 | 
 
 
 Chaudhuri et al. (2019a) ; Zhou et al. (2021b) 
 | 
 
 
 Cosine similarity 
 | 

 
 | 
 | 
 
 
 Luo et al. (2021) 
 | 
 
 
 TF-IDF 
 | 

 
 | 
 
 
 Knowledge base 
 | 
 
 
 Yin et al. (2016a) ; Wang et al. (2020) ; Ramadan et al. (2018) 
 | 
 
 
 Dot product / FC layer 
 | 

 
 | 
 
 
 Free text 
 | 
 
 
 Cai et al. (2019) 
 | 
 
 
 Jaccard distance 
 | 

 
 | 
 | 
 
 
 Lian et al. (2019) ; Wu et al. (2021) ; Liu et al. (2021a) 
 | 
 
 
 Dot product 
 | 

 
 | 
 | 
 
 
 Jin et al. (2021a) ; He et al. (2021) ; Roller et al. (2021) 
 | 
 
 
 Transformer 
 | 

 
 | 
 | 
 
 
 Zhang et al. (2021) ; Yavuz et al. (2019) ; Roller et al. (2021) 
 | 
 
 
 TF-IDF 
 | 

 
 | 
 | 
 
 
 Santhanam et al. (2020) ; Robertson and Zaragoza (2009) 
 | 
 
 
 Cosine similarity 
 | 

 
 
 
 Graph linking 
 | 
 
 
 Knowledge graph 
 | 
 
 
 Wang et al. (2021b) ; Kumar et al. (2020) 
 | 
 
 
 Entity linking 
 | 

 
 
 
 String 
 | 
 | 
 
 
 Li et al. (2020b) 
 | 
 
 
 Non-stopword matching 
 | 

 
 | 
 | 
 
 
 Zhou et al. (2021b) 
 | 
 
 
 Lemmatized words matching 
 | 

 
 | 
 
 
 Knowledge base 
 | 
 
 
 He et al. (2017) ; Wang et al. (2021a) ; Zhu et al. (2017) ; Agarwal et al. (2018) 
 | 
 
 
 Exact matching 
 | 

 
 | 
 
 
 Free text 
 | 
 
 
 Tan et al. (2020) 
 | 
 
 
 Exact matching 
 | 

 
 | 
 | 
 
 
 Wu et al. (2020) 
 | 
 
 
 N-gram matching 
 | 

 
 
 
 Query 
 | 
 
 
 Knowledge graph 
 | 
 
 
 Chaudhuri et al. (2021) ; Li et al. (2021b) ; Cho et al. (2021) 
 | 
 
 
 SPARQL 
 | 

 
 | 
 
 
 Knowledge base 
 | 
 
 
 Pei et al. (2019) 
 | 
 
 
 Symbolic query 
 | 

 
 | 
 | 
 
 
 Madotto et al. (2020) 
 | 
 
 
 SQL, CYPHER 
 | 

 
 
 
 Memory networks 
 | 
 
 
 Knowledge base 
 | 
 
 
 Florez and Mueller (2019) ; Qin et al. (2019b) 
 | 
 
 
 Memory networks 
 | 

 
 
 
 Attention 
 | 
 | 
 
 
 Yang et al. (2020b) ; Le et al. (2016) ; Wen et al. (2018) ; Eric et al. (2017a) 
 | 
 
 
 Attention mechanism 
 | 

 
 | 
 
 
 Free text 
 | 
 
 
 Moghe et al. (2020) ; Ma et al. (2020) ; Zhang et al. (2019) ; Kim et al. (2020a) ; Liu et al. (2021b) ; Yavuz et al. (2019) ; Zhao et al. (2020a) ; Zhao et al. (2020b) 
 | 
 
 
 Attention mechanism 
 | 

 
 | 
 
 
 Knowledge graph 
 | 
 
 
 Moon et al. (2019a) 
 | 
 
 
 Attention mechanism 
 | 

 
 
 
 NA 
 | 
 
 
 Knowledge base 
 | 
 
 
 Gou et al. (2021) 
 | 
 
 
 Entire table 
 | 

 
 | 
 
 
 Free text 
 | 
 
 
 Xu et al. (2021b) ; Cui et al. (2021) ; Xu et al. (2021a) ; Peng et al. (2020) 
 | 
 
 
 Knowlede injection 
 | 

 
 
 
 Nearest neighbors 
 | 
 
 
 Free text 
 | 
 
 
 Komeili et al. (2021) ; Fan et al. (2021) 
 | 
 
 
 Nearest-neighbors 
 | 

 

 
 

#### Reflection

 
 The similarity methods, which in contrast to many other methods, such as using existing parsers or queries, are easily parallelizable and differentiable. However, they often require the system to be trained which is more time consuming but does not require experts to create rules.
And while the similarity methods can be parallalized, the string-based approach is simple and direct.

 
 
 Query methods are designed to be efficient, but require user knowledge and a database that is structured according to the syntax. Memory networks Sukhbaatar et al. (2015) have also been used, as they appear to be effective in incorporating KBs into neural models. Although, the more efficient methods are those that avoid the search entirely by injecting the knowledge. However, these are complex to develop and have several issues that will be discussed in the following sections.

 
 
 In comparison to the amount of papers focusing on external knowledge there is much less work done on internal knowledge, even though that we identify 3 times as many internal knowledge forms used than external ones. This may be due to lack of internal datasets visibility, which we hope to alleviate in this survey (Section 6 ).

 
 
 The existing work highlight two main problems: 1) only locating the exact relevant knowledge, 2) doing so using methods that are suitable for real-time applications. For the first, considering the enormous amount of data, only selecting the exact relevant knowledge is impressive if not ambitious; especially given that such relevant knowledge may not appear in the same sentence, triple, knowledge source, or even exist in the data at all. However, many works assume that such relevant knowledge exist in a single location. Additionally, very few use entity linking on KGs, which are often encoded, rather than string based methods. This might imply that there is a gap between those who build such KGs and those who use them.

 
 
 Many also use search methods that retrieve the top-K data which is problematic for four reasons. 1) this approach ignores the fact that some dialogues require more or less information. 2) a main reason for developing KGs/KBs triples is to reduce noise, but using all retrieved triples might introduce even more noise. 3) the approach assumes that the knowledge database is complete (i.e., the knowledge always exists). 4) the length of the concatenated retrieved knowledge might exceed the length limitation of language models (LM) and hence truncated and potentially lose the actual relevant knowledge.

 
 
 For the second problem, many use methods that are impractical for real-time applications, such as matching over all of the knowledge in the database. Notably, rarely any work mentions the time it takes for systems to work, which may significantly hinder their usage. For example, a DS that is nearly perfect but takes a month to respond will rarely be used.

 
 
 

#### Cognitively-Inspired Future Work

 
 Semantic memory networks and clusters theories, like those in Pereira et al. (2018) ; Hills et al. (2012) ; Austerweil et al. (2012) , which follow the theory that memories are organized in a network structure, suggest that current knowledge search methods can be improved. For example, instead of searching through the entire knowledge source (e.g., by string matching over each triple), by structuring knowledge in such semantic network and clusters we can use random walks Austerweil et al. (2012) 
or perform optimal foraging policy methods Hills et al. (2012) . Pulvermüller (2001) theory could also be used, by
structuring the knowledge source in such way that it will show differences when different knowledge types are triggered

 
 
 Alternatively and the quickest option, is to follow Buckner and Wheeler (2001) ’s theory about spontaneously activated memories, using models that avoid the search entirely.
However, such methods have limitations, such as the inability to use newly-added knowledge without retraining.
Future work should examine methods to avoid searching over all knowledge data, as it significantly limits the real-time applications of the DS and hence its use case. Furthermore, similarly to the human brain where knowledge search must be done quickly for survival and effective communication, DS should also aim for such speed. Future work should emphasize time in future challenges and datasets as a metric for system performance.

 
 
 It is important to remember that knowledge search is still very much an open problem. After all, humans have not perfected the task either, which can be seen when attempting to retrieve known answers to questions unsuccessfully. However as stated in Pereira et al. (2018) , humans have no direct access to their semantic knowledge network. That being said, DS do, which implies a potential to bypass human performance in such tasks.

 
 
 
 
 

## 4 Knowledge Encoding

 
 As the dialogue and knowledge often appear in different forms (e.g., free text and KG), we need to encode them to similar form so they can be incorporated later. Here we present such methods, which take knowledge in various forms and encode it. We assume that the dialogue is encoded in one of many ways which can be found in one of the resources we referenced in Section 2.3 .

 
 

### 4.1 Internal Knowledge Encoding

 
 The methods for internal knowledge encoding can be seen in Table 3 .
As dialogue has a time component to it, most work use models that can manage such information, such as RNNs, GRUs, and Transformers. While the latter has become the most popular in recent year for its long time dependencies. Trainable parameters are also used instead of a full model, such as representing the speakers using token embeddings and the emotions as trainable matrices.

 
 
 Table 3: Internal Knowledge Encoding Table 
 
 
 
 
 
 Knowledge Form 
 | 
 
 
 Name 
 | 
 
 
 Method 
 | 

 
 
 
 AMR graph 
 | 
 
 
 Bai et al. (2021) 
 | 
 
 
 Graph transformer 
 | 

 
 | 
 
 
 Chen et al. (2016) 
 | 
 
 
 FC network / RNN / CNN 
 | 

 
 
 
 Dependency tree 
 | 
 
 
 Wang et al. (2020) 
 | 
 
 
 GRU 
 | 

 
 | 
 
 
 Yang et al. (2020a) 
 | 
 
 
 Graph encoder 
 | 

 
 
 
 Part-of-speech tagging 
 | 
 
 
 Su et al. (2018) 
 | 
 
 
 NA: Knowledge injection 
 | 

 
 
 
 Noun phrases 
 | 
 
 
 Adolphs et al. (2021) 
 | 
 
 
 Transformer 
 | 

 
 
 
 Emotions 
 | 
 
 
 Liang et al. (2021) 
 | 
 
 
 Trainable parameters 
 | 

 
 | 
 
 
 Zhou et al. (2018a) 
 | 
 
 
 GRU 
 | 

 
 | 
 
 
 Compton et al. (2021) 
 | 
 
 
 Transformer 
 | 

 
 
 
 Specific domains 
 | 
 
 
 Luo et al. (2021) 
 | 
 
 
 Transformer 
 | 

 
 | 
 
 
 Chaudhuri et al. (2018) 
 | 
 
 
 GRU 
 | 

 
 | 
 
 
 Jin et al. (2021b) 
 | 
 
 
 Transformer 
 | 

 
 
 
 Entities 
 | 
 | 
 
 
 Transformer 
 | 

 
 | 
 
 
 Li et al. (2021a) 
 | 
 
 
 Transformer 
 | 

 
 
 
 User intent 
 | 
 
 
 Kassawat et al. (2020) 
 | 
 
 
 RNN 
 | 

 
 | 
 
 
 Zhu et al. (2017) 
 | 
 
 
 RNN 
 | 

 
 
 
 Speaking style 
 | 
 
 
 Sun et al. (2021) 
 | 
 
 
 GRU 
 | 

 
 
 
 Speaker representation 
 | 
 
 
 Gu et al. (2020) 
 | 
 
 
 Trainable parameters 
 | 

 

 
 
 

### 4.2 External Knowledge Encoding

 
 The methods for internal knowledge encoding can be
seen in Table 4 .
Similarly to the internal encoding methods, many works use algorithms that take time into account, such as RNN, GRU, and Transformers. There also exist work on using graph algorithms to encode the KG triples, such as RGCNN, graph Laplacian, and graph transformers, and work that simply use trainable parameters, such as averaging the KG/KB entity embeddings. For free text, the bulk of the work use transformer-based models, while for KBs there is about equal number of works that use any method.

 
 
 Table 4: External Knowledge Encoding Table. KG = knowledge graph, KB = knowledge base, FT = free text 
 
 
 
 
 
 Knowledge Form 
 | 
 
 
 Name 
 | 
 
 
 Method 
 | 

 
 
 
 
 
 Knowledge graph 
 | 
 
 
 Chaudhuri et al. (2019a) ; Kumar et al. (2020) ; Kassawat et al. (2020) ; Yang et al. (2020b) 
 | 
 
 
 Trainable parameters 
 | 

 
 | 
 
 
 Dziri et al. (2021) ; Luo et al. (2021) ; Zhou et al. (2021b) 
 | 
 
 
 Transformer 
 | 

 
 | 
 
 
 Wang et al. (2021b) 
 | 
 
 
 RGCNN 
 | 

 
 | 
 
 
 Li et al. (2020b) 
 | 
 
 
 Graph transformer 
 | 

 
 | 
 
 
 Moon et al. (2019a) 
 | 
 
 
 NN 
 | 

 
 | 
 
 
 Chaudhuri et al. (2021) 
 | 
 
 
 Graph Laplacian 
 | 

 
 
 
 Free text 
 | 
 
 
 Lian et al. (2019) ; Zhang et al. (2019) ; Yavuz et al. (2019) ; Cai et al. (2019) ; Moghe et al. (2020) 
 | 
 
 
 RNN 
 | 

 
 | 
 
 
 Zhao et al. (2020a) 
 | 
 
 
 GRU 
 | 

 
 | 
 
 
 Ye et al. (2019) 
 | 
 
 
 CNN 
 | 

 
 | 
 
 
 Zhang et al. (2021) ; Kim et al. (2020a) ; Zhao et al. (2020b) ; Wu et al. (2021) ; Roller et al. (2021) ; Liu et al. (2021a) ; Wu et al. (2020) ; He et al. (2021) ; Santhanam et al. (2020) ; Hedayatnia et al. (2020) ; Tan et al. (2020) ; Komeili et al. (2021) ; Ma et al. (2020) ; Liu et al. (2021b) ; Fan et al. (2021) ; Adolphs et al. (2021) ; Jin et al. (2021a) 
 | 
 
 
 Transformer 
 | 

 
 | 
 
 
 Adolphs et al. (2021) ; Xu et al. (2021b) ; Peng et al. (2020) ; Cui et al. (2021) ; Xu et al. (2021a) 
 | 
 
 
 Knowledge injection 
 | 

 
 | 
 
 
 Parthasarathi and Pineau (2018) 
 | 
 
 
 Trainable parameters 
 | 

 
 
 
 Knowledge base 
 | 
 
 
 Wang et al. (2021a) ; Pei et al. (2019) ; Florez and Mueller (2019) 
 | 
 
 
 Memory network 
 | 

 
 | 
 
 
 Yin et al. (2016a) ; Ramadan et al. (2018) ; Zhu et al. (2017) ; He et al. (2017) ; Le et al. (2016) 
 | 
 
 
 RNN 
 | 

 
 | 
 
 
 Wang et al. (2020) ; Agarwal et al. (2018) 
 | 
 
 
 GRU 
 | 

 
 | 
 
 
 Wen et al. (2018) ; Eric et al. (2017a) ; Qin et al. (2019b) 
 | 
 
 
 Trainable parameters 
 | 

 
 | 
 
 
 Gou et al. (2021) ; Li et al. (2021b) ; Madotto et al. (2020) 
 | 
 
 
 Transformer 
 | 

 

 
 
 

### 4.3 Discussion Thoughts

 

#### Reflection

 
 Methods for encoding knowledge are typically based on either getting a vector representation, or by representing structured knowledge as text. To obtain a vector representation, some use trainable embeddings to represent the knowledge, for example by representing each KG triple or KB entries as an average of the subject and relation or cells embeddings. This method is simplistic, as it does not require a model. However, such approach might not have the same encoding capabilities of transformer-based models. Others choose to use a model, for example by representing the KG triples, free text words, or KB cells as tokens and sending these to a transformer-based model.

 
 
 Knowledge encoding can also be categorized into three main general approaches: 1) combining the knowledge and dialogue (e.g., into one sequence) followed by a model for encoding, 2) encoding the dialogue and each relevant knowledge independently, 3) injecting the knowledge into the parameters of the model, for example by using the knowledge as a target for the model.

 
 
 Each of the three has benefits and drawbacks. The first approach is rather simplistic, as the encoder has only one representation to encode. However, if such representation does not separate the knowledge and dialogue sufficiently, the knowledge may be lost in the output representation. The second is slightly more complex, as it potentially requires multiple encoders for the dialogue and each knowledge form. However, it allows for more freedom in the next step (i.e., incorporation). That is, as the knowledge and dialogue representations are separated, the model does not need to decipher which part of the representation contributes to the knowledge. The third is the most alluring as it avoids the search and encoding portions, but has several problems. As knowledge is dynamic and constantly changing, these methods require retraining models to encode the new knowledge. Additionally, there is a question of whether such methods can truly encode knowledge that rarely appear in the training data.

 
 
 While it is difficult converting free text into KGs, doing so for KBs is rather straight forward. Some works have shown that converting KBs to KGs have benefits He et al. (2017) ; Yang et al. (2020b) , specifically, better representation of entities is possible as relational information between entities exist. That being said, not many use graph-based methods directly on KGs.

 
 
 Lastly, there is a question of whether a single vector representation, which most works use to encode the knowledge, is sufficient to encode all of the relevant information. This is apparent most often in free text, where the long documents may contain numerous relevant facts.

 
 
 

#### Cognitively-Inspired Future Work

 
 While it is yet unclear how humans convert neuron-activity directly into words, there is evidence that specific neuronal activity corresponds to specific words. Cognitively, this section can be seen as “learning”, where memories, or knowledge, are encoded in the brain. Word embeddings can be seen as such encoding, specifically, a representation of the semantic networks discussed in Section 2.5 . While most work use one embedding to represent the vocabulary, Ross (2010) , Reddy (1993) , and Schober (1998) argue that words do not contain their meaning. By this argument, different speakers may have different meanings for a word. Hence, different speakers require different word embeddings to truly represent that humans have different memories and knowledge, and view the world differently. Ross (2010) also argue that while speakers have their own representations of concept meanings, these meanings are fluid. That is, humans tailor their language to whom they talk to, and the listener interprets that language based on their knowledge of the speaker. This suggests that current fixed embeddings methods might not be sufficient, and a dynamic representation can be beneficial.

 
 
 Another important aspect of human knowledge is that it is dynamic. Our view of the world, memories, and knowledge are constantly changing
However, current systems largely assume a fixed knowledge source. This can be problematic when trained DS memorize and generate previously seen data rather than using the new data, or when the systems themselves do not have the capability of incorporating novel information without changing the architecture or retraining. Future work should examine methods of handling such dynamic knowledge source.

 
 
 Lastly, according to the interactive alignment model Pickering and Garrod (2004) , language processing in monologues is different from that in dialogues. This suggests that current methods of pretraining large LM on data, such as Wikipedia text, may not represent how humans learn language, and hence develop insufficient world models and achieve lower performance. A potential avenue to explore is comparing the performance of such systems on monologues and dialogues, in addition to developing more such rich dialogue datasets.

 
 
 
 
 

## 5 Knowledge Incorporation

 
 So far we have found the relevant knowledge to be used and encoded it. Now we need to incorporate it into the encoded dialogue history. Meaning, we need to combine the two encoded representations so our system can decide on an appropriate response. As the knowledge is encoded, we do not need to differentiate between the internal or external types. Hence, all of the incorporation methods can be seen in Table 5 , which has 6 different such methods.

 
 
 Table 5: Knowledge Incorporation Table 
 
 
 
 
 
 Method 
 | 
 
 
 Name 
 | 

 
 
 
 
 
 Aggregation followed by blending 
 | 
 
 
 Bai et al. (2021) ; Compton et al. (2021) ; Jin et al. (2021a) ; Adolphs et al. (2021) ; Chen et al. (2016) ; Li et al. (2020a) ; Tan et al. (2020) ; Hedayatnia et al. (2020) ; Santhanam et al. (2020) ; He et al. (2021) ; Wu et al. (2020) ; Liu et al. (2021a) ; Roller et al. (2021) ; Wu et al. (2021) ; Kim et al. (2020a) ; Zhao et al. (2020b) ; Zhang et al. (2021) ; Yin et al. (2016a) ; Gu et al. (2020) ; Liang et al. (2021) ; Sun et al. (2021) ; Chaudhuri et al. (2018) ; Dziri et al. (2021) ; Kumar et al. (2020) ; Zhou et al. (2021b) ; Agarwal et al. (2018) ; Wen et al. (2018) ; Li et al. (2021b) 
 | 

 
 
 
 Attention mechanism 
 | 
 
 
 Bai et al. (2021) ; Yang et al. (2020b) ; Li et al. (2021a) ; Moon et al. (2019a) ; Eric et al. (2017a) ; Moghe et al. (2020) ; Liu et al. (2021b) ; Lian et al. (2019) ; Ye et al. (2019) ; Ma et al. (2020) 
 | 

 
 
 
 Selection mechanism 
 | 
 
 
 Chaudhuri et al. (2019a) ; Wang et al. (2021b) ; Gou et al. (2021) ; Qin et al. (2019b) ; Zhao et al. (2020a) ; Zhu et al. (2017) 
 | 

 
 
 
 Knowledge injection 
 | 
 
 
 Peng et al. (2020) ; Su et al. (2018) ; Madotto et al. (2020) ; Xu et al. (2021b) ; Cui et al. (2021) ; Xu et al. (2021a) 
 | 

 
 
 
 Graph creation 
 | 
 
 
 Li et al. (2020b) ; He et al. (2017) 
 | 

 
 
 
 Memory networks 
 | 
 
 
 Ramadan et al. (2018) ; Pei et al. (2019) ; Florez and Mueller (2019) ; Wang et al. (2021a) 
 | 

 

 
 

### 5.1 Discussion Thoughts

 

#### Reflection

 
 The main incorporation methods can be seen as: 1) aggregation of the representations of the dialogue and knowledge, followed by a blending mechanism. 2) attention over the representations. 3) a mechanism which selects which representation to use. 4) knowledge injection.

 
 
 The bulk of the work has aggregated the encoded knowledge and dialogue (e.g., by summing vectors or concatenating tokens) followed by a a blending mechanism, such as a whole model, trainable parameters, or some nonlinear function. The blending mechanism is crucial, as if it does not combine the knowledge and dialogue well, the system may not understand what representation it should focus on. Unsurprisingly, transformer-based models appear in the highest frequency, likely for its great success in recent years. However, they often require long training time and a lot of data, which is often unavailable, specifically for unique domains, facts, or novel knowledge forms.

 
 
 While many models have attention mechanism included in their architecture, many have also use simpler attention mechanisms to attend over the knowledge and dialogue representations. These require much less computational power and training time, but may have lower capabilities. Some also use selection mechanisms which select which representation to use in order the generate or retrieve an appropriate response. For example, by using the copy mechanism, which allows the incorporation of tokens directly from the knowledge source. This is especially useful when the knowledge has vocabulary that is not present in the training data, in addition to solving the dynamic-knowledge problem of knowledge-injection discussed in Section 4.3 .

 
 
 Knowledge injection methods also have several problems as mentioned in Section 4.3 . Additionally, these approaches are form-specific, such as having unique losses for domains Peng et al. (2020) or POS tags Su et al. (2018) . This is unsustainable as DS should be able to incorporate multiple forms simultaneously. Simpler and less popular methods also exist, such as combining the dialogue and knowledge representations into one graph, or using memory networks most often on a KB.

 
 
 

#### Cognitively-Inspired Future Work

 
 The bulk of the work focus on a rather straightforward method of aggregating the knowledge and dialogue representations followed by a blending mechanism. However,
the human language has a an extremely complex structure that may require many different rules. The study of pragmatics Mey (2006) for example, focuses on how context contributes to language utilization, and encompasses several phenomena; implicature is when utterances contain implied information, while the relevance theory Sperber (1995) , states that every utterance contains relevant, worth listening information. Knowledge-enhanced DS are taking the right steps towards such studies by explicitly incorporating information that is hidden but implied, in addition to improving the response’s relevancy and quality. However, current DS incorporation methods are solely based on the encoded dialogue and very limited knowledge forms at a time. In comparison, humans use many context features which allow them to incorporate knowledge differently into the dialogue, such as explaining the same concept to a child versus an adult, and responding differently to different people based on their knowledge of the listener Ross (2010) .

 
 
 In order for DS to successfully mimic human communication, future work should emphasize fluidity in their response and better models of the listeners, such as various characteristics and likes. For example, by taking attributes that are rarely used by DS these days, such as the users age, emotional status, or even time of the day, as we alternate our wordings based on listeners’ maturity and generate shorter responses when we are busy.

 
 
 To communicate successfully, humans use a great amount of contexts that are based on the environment, listeners, and themselves, while DS use hardly any in comparison. Hence, a considerable potential for improvement exist.

 
 
 
 
 

## 6 Datasets

 
 Table 6 presents a comprehensive list of the datasets used in the reviewed work. It is split into the dataset name and its description.

 
 
 Table 6: Datasets Table 
 
 
 
 
 
 Dataset Name 
 | 
 
 
 Description 
 | 

 
 
 
 
 
 Topical-Chat (TC) Gopalakrishnan et al. (2019) 
 | 
 
 
 A knowledge-grounded conversations where the knowledge is on 8 topics 
 | 

 
 
 
 CMU Document Grounded Conversations Zhou et al. (2018b) 
 | 
 
 
 A knowledge-grounded conversations where the knowledge is about specific Wikipedia articles about popular movies 
 | 

 
 
 
 CMU Movie Summary Bamman et al. (2013) 
 | 
 
 
 Summaries of movie plots extracted from Wikipedia with aligned metadata extracted from Freebase 
 | 

 
 
 
 Reddit Conversation Corpus Dziri et al. (2018) 
 | 
 
 
 Dialogues extracted from Reddit, where each is composed of 3 turn exchanges 
 | 

 
 
 
 Reddit dataset Mazaré et al. (2018) 
 | 
 
 
 Dialogues that are based on personas extracted from Reddit 
 | 

 
 
 
 Reddit discussions (The Pushshift Reddit Dataset) Baumgartner et al. (2020) 
 | 
 
 
 Submissions and comments posted on subreddits 
 | 

 
 
 
 Reddit comments Boyd et al. (2020) 
 | 
 
 
 Conversations extracted from Reddit comments 
 | 

 
 
 
 Grounded Reddit conversation Qin et al. (2019a) 
 | 
 
 
 Conversations extracted from Reddit, which are linked to documents discussed in the conversations 
 | 

 
 
 
 Soccer dialogues Chaudhuri et al. (2019b) 
 | 
 
 
 Soccer dialogues along with a knowledge graph for each team 
 | 

 
 
 
 In-car dialogue dataset Eric et al. (2017b) 
 | 
 
 
 KB-grounded dialogues which span 3 distinct tasks in the in-car personal assistant space 
 | 

 
 
 
 ConvAI2 dataset Dinan et al. (2019a) 
 | 
 
 
 Conversational dataset based on the PERSONA-CHAT dataset 
 | 

 
 
 
 LIGHT dataset Shuster et al. (2021) 
 | 
 
 
 Episodes of character interactions in a text adventure game 
 | 

 
 
 
 LightWild Shuster et al. (2021) 
 | 
 
 
 Episodes in a role playing game where players converse with learning agents 
 | 

 
 
 
 LightQA Adolphs et al. (2021) 
 | 
 
 
 A derived version of LightWild, ending on a question about the episode 
 | 

 
 
 
 DailyDialog Li et al. (2017) 
 | 
 
 
 Multi-turn dialogues with human-written conversations 
 | 

 
 
 
 Empathetic Dialogues Rashkin et al. (2019a) 
 | 
 
 
 Conversations that are grounded in emotional situations 
 | 

 
 
 
 Blended Skill Talk Smith et al. (2020) 
 | 
 
 
 English conversations aimed at testing several skills, such as being engage, empathetic, and knowledgeable 
 | 

 
 
 
 OpenQA-NQ Lee et al. (2019) 
 | 
 
 
 Google queries paired with short answers extracted from Wikipedia 
 | 

 
 
 
 Multimodal EmotionLines Dataset Poria et al. (2019) 
 | 
 
 
 Extension of EmotionLines dataset 
 | 

 
 
 
 Interactive Emotional Dyadic Motion Capture Database Busso et al. (2008) 
 | 
 
 
 Emotion-segmented videos of dyadic conversations 
 | 

 
 
 
 EmoryNLP Zahiri and Choi (2018) 
 | 
 
 
 Episodes, scenes, and emotional-grounded utterances 
 | 

 
 
 
 EmpatheticDialogues Rashkin et al. (2019b) 
 | 
 
 
 Emotional-grounded conversations 
 | 

 
 
 
 MuTual Cui et al. (2020) 
 | 
 
 
 Multi-Turn dialogue reasoning dataset based on English listening comprehension exams taken by Chinese students 
 | 

 
 
 
 Crowd-sourced SocialIQA-prompted Zhou et al. (2021a) 
 | 
 
 
 Dialogues that exhibit social commonsense in an interactive setting 
 | 

 
 
 
 Wizard of Wikipedia Dinan et al. (2019b) 
 | 
 
 
 Wikipedia-grounded Conversations with many discussions topics 
 | 

 
 
 
 Wizard-of-Oz (WOZ) 2.0 Mrksic et al. (2016) 
 | 
 
 
 Expansion of the WOZ dataset 
 | 

 
 
 
 MultiDomain Wizard-of-Oz (MultiWOZ) Budzianowski et al. (2018) 
 | 
 
 
 Annotated human-human written conversations spanning 8 domains 
 | 

 
 
 
 MultiWOZ 2.1, 2.2 Eric et al. (2020) , Zang et al. (2020) 
 | 
 
 
 Extension of MultiWOZ 
 | 

 
 
 
 OR-ShARC Gao et al. (2021) 
 | 
 
 
 Conversational machine reading dataset where the gold rule text for each sample is removed and used as a KB 
 | 

 
 
 
 Alexa Prize Khatri et al. (2018) 
 | 
 
 
 Spoken conversations which are also not task-restricted, open-ended, topical, involve opinions, and are conducted with real users 
 | 

 
 
 
 Holl-E Moghe et al. (2018) 
 | 
 
 
 Movie conversations where each response is generated by copying and/or modifying sentences from unstructured background knowledge 
 | 

 
 
 
 Dialog bAbI Bordes and Weston (2016) 
 | 
 
 
 Noise-free simulated dialogues 
 | 

 
 
 
 mDSTC2 dataset Henderson et al. (2014) 
 | 
 
 
 Modified version of the DSTC2 dataset 
 | 

 
 
 
 DialogRE Yu et al. (2020) 
 | 
 
 
 Relations-annotated dialogues originating from the complete transcripts of Friends 
 | 

 
 
 
 PersonaChat Zhang et al. (2018a) 
 | 
 
 
 Multi-turn dialogues conditioned on personas 
 | 

 
 
 
 OpenSubtitles Lison and Tiedemann (2016) 
 | 
 
 
 Multilingual parallel corpora from a database of movies and TV subtitles 
 | 

 
 
 
 DSTC7-Track1 Yoshino et al. (2019) 
 | 
 
 
 Partial conversations which requires users to select the correct next utterances from a set of candidates 
 | 

 
 
 
 DSTC7-Track2 
 | 
 
 
 Facts-grounded conversations extracted from Reddit 
 | 

 
 
 
 DSTC9 Track 1 Kim et al. (2020b) 
 | 
 
 
 Conversations where the dialogue flow does not break when users have out of scope requests 
 | 

 
 
 
 DSTC 8-Track 2 Kim et al. (2019) 
 | 
 
 
 Extended the DSTC 7 Track 1 by adding 3 new dimensions 
 | 

 
 
 
 KdConv8 Zhou et al. (2020) 
 | 
 
 
 KG-grounded multi-domain knowledge-driven conversation in Chinese 
 | 

 
 
 
 E2E NLG Novikova et al. (2017) 
 | 
 
 
 A dataset in the restaurant domain 
 | 

 
 
 
 MedDialog Zeng et al. (2020) 
 | 
 
 
 Conversations in Chinese between patients and doctors covering many specialties of diseases 
 | 

 
 
 
 Unnamed medical dataset Zeng et al. (2020) 
 | 
 
 
 Conversations covering many specialties of diseases 
 | 

 
 
 
 EMPATHETICDIALOGUES Rashkin et al. (2018) 
 | 
 
 
 Conversations grounded in emotional situations 
 | 

 
 
 
 Bench’It 
 | 
 
 
 General knowledge evaluation for French question answering on Wikidata 
 | 

 
 
 
 CALOR Marzinotto et al. (2018) 
 | 
 
 
 Annotated French encyclopedic history texts 
 | 

 
 
 
 MovieChAtt Danescu-Niculescu-Mizil and
Lee (2011) 
 | 
 
 
 Conversations between pairs of characters, which are matched on IMDB 
 | 

 
 
 
 Nell Carlson et al. (2010) 
 | 
 
 
 A system that extracts information from web text to populate a KB 
 | 

 
 
 
 TVTropes 
 | 
 
 
 Tropes associated with examples of their occurrences in television, film, and literature 
 | 

 
 
 
 REDIAL Li et al. (2018) 
 | 
 
 
 Conversations on providing movie recommendations 
 | 

 
 
 
 Emotional lexicon NRC_VAD Mohammad (2018) 
 | 
 
 
 Human ratings of arousal, valence, and dominance for English words 
 | 

 
 
 
 Airline travel information system (ATIS) corpus Mesnil et al. (2015) 
 | 
 
 
 Audio and transcripts of people asking for flight information with intent categories 
 | 

 
 
 
 Multimodal E-commerce Product Attribute Value Extraction Zhu et al. (2020) 
 | 
 
 
 Textual product descriptions and product images 
 | 

 
 
 
 JDDC Chen et al. (2020) 
 | 
 
 
 Chinese E-commerce conversation corpus with intent information 
 | 

 
 
 
 E-commerce Dialogue Corpus (ECD) Zhang et al. (2018b) 
 | 
 
 
 E-commerce dataset with diverse types of conversations 
 | 

 
 
 
 SimpleQuestions Bordes et al. (2015) 
 | 
 
 
 Factoid question answering with corresponding triple fact 
 | 

 
 
 
 Music Question Answering GenQA Yin et al. (2016b) 
 | 
 
 
 Open domain factoid QA 
 | 

 
 
 
 NLPCC emotion classification dataset 
 | 
 
 
 Emotional-annotated sentences collected from Weibo 
 | 

 
 
 
 Short-Text Conversation (STC) conversation dataset Shang et al. (2015) 
 | 
 
 
 Conversations from Weibo 
 | 

 
 
 
 Doc2Dial Feng et al. (2020) 
 | 
 
 
 Document-grounded goal-oriented dialogues 
 | 

 
 
 
 Ubuntu Dialogue Corpus V1 Lowe et al. (2015) 
 | 
 
 
 Multi-turn dialogues 
 | 

 
 
 
 Ubuntu Dialogue Corpus V2 Lowe et al. (2017) 
 | 
 
 
 An updated version of the Ubuntu Dialogue Corpus 
 | 

 
 
 
 Douban Conversation Corpus Wu et al. (2017) 
 | 
 
 
 Retrieval based dataset where for each dialogue context there are multiple candidate responses 
 | 

 
 
 
 Engaging ImageChat Shuster et al. (2020) 
 | 
 
 
 Dialogues over images using many possible style trait 
 | 

 
 
 
 Multi-Genre Natural Language Inference (MultiNLI) Williams et al. (2017) 
 | 
 
 
 Sentence pairs annotated with textual entailment information 
 | 

 
 
 
 Microsoft Research Paraphrase Corpus (MRPC) Dolan and Brockett (2005) 
 | 
 
 
 Sentence pairs where each pair is labelled if it was paraphrased by a human 
 | 

 
 
 
 Multimodal Dialogue (MMD) Saha et al. (2017) 
 | 
 
 
 Multimodal domain-aware conversations between shoppers and sales agents 
 | 

 
 
 
 OpenDialKG Moon et al. (2019b) 
 | 
 
 
 Human-to-human utterances role-playing dialogues which are grounded with corresponding entities and paths from a KG 
 | 

 
 
 
 Freebase Bast et al. (2014) 
 | 
 
 
 KB with data inserted mainly by its members 
 | 

 
 
 
 ATOMIC KG Sap et al. (2019) 
 | 
 
 
 Commonsense reasoning KG 
 | 

 
 
 
 DBpedia Lehmann et al. (2015) 
 | 
 
 
 Multilingual KB extracted from Wikipedia 
 | 

 
 
 
 ConceptNet Speer et al. (2017) 
 | 
 
 
 KG has labeled edges connections between words and phrases 
 | 

 
 
 
 Wikidata 
 | 
 
 
 KB extracted from Wikimedia sister projects (e.g., Wikipedia, Wikisource) 
 | 

 
 
 
 WordNet Bordes et al. (2013) 
 | 
 
 
 Lexical English KB which groups verbs, nouns, adjectives, and adverbs into groups of concepts 
 | 

 

 
 
 

## 7 Conclusion

 
 Knowledge helps DSs narrow the gap between human and machine communication. We survey which types of knowledge past research has added to DSs. Then, we identify and discuss existing solutions to three problems which have to be solved by knowledge-enhanced DSs: knowledge search, knowledge encoding, and knowledge incorporation. Based on theories of how our brains work, we finally propose ways to improve knowledge-enhanced DSs.

 
 
 

## References

 
 
 Adolphs et al. (2021) 
 
Leonard Adolphs, Kurt Shuster, Jack Urbanek, Arthur Szlam, and Jason Weston.
2021.

 
 Reason first, then respond:
Modular generation for knowledge-infused dialogue .

 
 CoRR , abs/2111.05204.

 

 
 Agarwal et al. (2018) 
 
Shubham Agarwal, Ondřej Dušek, Ioannis Konstas, and Verena Rieser.
2018.

 
 A knowledge-grounded
multimodal search-based conversational agent .

 
 In Proceedings of the 2018 EMNLP Workshop SCAI: The 2nd
International Workshop on Search-Oriented Conversational AI , pages 59–66,
Brussels, Belgium. Association for Computational Linguistics.

 

 
 Austerweil et al. (2012) 
 
Joseph Austerweil, Joshua T Abbott, and Thomas Griffiths. 2012.

 
 Human memory search as a random walk in a semantic network .

 
 In Advances in Neural Information Processing Systems ,
volume 25. Curran Associates, Inc.

 

 
 Bai et al. (2021) 
 
Xuefeng Bai, Yulong Chen, Linfeng Song, and Yue Zhang. 2021.

 
 Semantic
representation for dialogue modeling .

 
 In Proceedings of the 59th Annual Meeting of the Association
for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers) , pages 4430–4445,
Online. Association for Computational Linguistics.

 

 
 Bamman et al. (2013) 
 
David Bamman, Brendan O’Connor, and Noah A. Smith. 2013.

 
 Learning latent personas
of film characters .

 
 In Proceedings of the 51st Annual Meeting of the Association
for Computational Linguistics, ACL 2013, 4-9 August 2013, Sofia, Bulgaria,
Volume 1: Long Papers , pages 352–361. The Association for Computer
Linguistics.

 

 
 Bast et al. (2014) 
 
Hannah Bast, Florian Bäurle, Björn Buchhold, and Elmar
Haußmann. 2014.

 
 Easy access to the
freebase dataset .

 
 In 23rd International World Wide Web Conference, WWW ’14,
Seoul, Republic of Korea, April 7-11, 2014, Companion Volume , pages 95–98.
ACM.

 

 
 Baumgartner et al. (2020) 
 
Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy
Blackburn. 2020.

 
 The
pushshift reddit dataset .

 
 In Proceedings of the Fourteenth International AAAI
Conference on Web and Social Media, ICWSM 2020, Held Virtually, Original
Venue: Atlanta, Georgia, USA, June 8-11, 2020 , pages 830–839. AAAI Press.

 

 
 Bordes et al. (2015) 
 
Antoine Bordes, Nicolas Usunier, Sumit Chopra, and Jason Weston. 2015.

 
 Large-scale simple question
answering with memory networks .

 
 CoRR , abs/1506.02075.

 

 
 Bordes et al. (2013) 
 
Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason
Weston, and Oksana Yakhnenko. 2013.

 
 Translating embeddings for modeling multi-relational data .

 
 In Advances in Neural Information Processing Systems 26: 27th
Annual Conference on Neural Information Processing Systems 2013. Proceedings
of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States ,
pages 2787–2795.

 

 
 Bordes and Weston (2016) 
 
Antoine Bordes and Jason Weston. 2016.

 
 Learning end-to-end
goal-oriented dialog .

 
 CoRR , abs/1605.07683.

 

 
 Boyd et al. (2020) 
 
Alex Boyd, Raul Puri, Mohammad Shoeybi, Mostofa Patwary, and Bryan Catanzaro.
2020.

 
 Large scale
multi-actor generative dialog modeling .

 
 In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics , pages 66–84, Online. Association for
Computational Linguistics.

 

 
 Buckner and Wheeler (2001) 
 
R L Buckner and M E Wheeler. 2001.

 
 The cognitive neuroscience of remembering.

 
 Nat. Rev. Neurosci. , 2(9):624–634.

 

 
 Budzianowski et al. (2018) 
 
Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva,
Stefan Ultes, Osman Ramadan, and Milica Gašić. 2018.

 
 MultiWOZ - a
large-scale multi-domain Wizard-of-Oz dataset for task-oriented dialogue
modelling .

 
 In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing , pages 5016–5026, Brussels, Belgium.
Association for Computational Linguistics.

 

 
 Busso et al. (2008) 
 
Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower,
Samuel Kim, Jeannette N. Chang, Sungbok Lee, and Shrikanth S. Narayanan.
2008.

 
 IEMOCAP:
interactive emotional dyadic motion capture database .

 
 Lang. Resour. Evaluation , 42(4):335–359.

 

 
 Cai et al. (2019) 
 
Deng Cai, Yan Wang, Wei Bi, Zhaopeng Tu, Xiaojiang Liu, Wai Lam, and Shuming
Shi. 2019.

 
 Skeleton-to-response:
Dialogue generation guided by retrieval memory .

 
 In Proceedings of the 2019 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers) , pages 1219–1228,
Minneapolis, Minnesota. Association for Computational Linguistics.

 

 
 Carlson et al. (2010) 
 
Andrew Carlson, Justin Betteridge, Bryan Kisiel, Burr Settles, Estevam
Hruschka, and Tom Mitchell. 2010.

 
 Toward an architecture for never-ending language learning.

 
 volume 3.

 

 
 Chaudhuri et al. (2018) 
 
Debanjan Chaudhuri, Agustinus Kristiadi, Jens Lehmann, and Asja Fischer. 2018.

 
 Improving response
selection in multi-turn dialogue systems by incorporating domain knowledge .

 
 In Proceedings of the 22nd Conference on Computational Natural
Language Learning , pages 497–507, Brussels, Belgium. Association for
Computational Linguistics.

 

 
 Chaudhuri et al. (2019a) 
 
Debanjan Chaudhuri, Md. Rashad Al Hasan Rony, Simon Jordan, and Jens Lehmann.
2019a.

 
 Using a kg-copy network for
non-goal oriented dialogues .

 
 CoRR , abs/1910.07834.

 

 
 Chaudhuri et al. (2019b) 
 
Debanjan Chaudhuri, Md. Rashad Al Hasan Rony, Simon Jordan, and Jens Lehmann.
2019b.

 
 Using a kg-copy
network for non-goal oriented dialogues .

 
 In The Semantic Web - ISWC 2019 - 18th International Semantic
Web Conference, Auckland, New Zealand, October 26-30, 2019, Proceedings, Part
I , volume 11778 of Lecture Notes in Computer Science , pages
93–109. Springer.

 

 
 Chaudhuri et al. (2021) 
 
Debanjan Chaudhuri, Md. Rashad Al Hasan Rony, and Jens Lehmann. 2021.

 
 Grounding
dialogue systems via knowledge graph aware decoding with pre-trained
transformers .

 
 In The Semantic Web - 18th International Conference, ESWC
2021, Virtual Event, June 6-10, 2021, Proceedings , volume 12731 of
 Lecture Notes in Computer Science , pages 323–339. Springer.

 

 
 Chen et al. (2017) 
 
Hongshen Chen, Xiaorui Liu, Dawei Yin, and Jiliang Tang. 2017.

 
 A survey on dialogue
systems: Recent advances and new frontiers .

 
 CoRR , abs/1711.01731.

 

 
 Chen et al. (2020) 
 
Meng Chen, Ruixue Liu, Lei Shen, Shaozu Yuan, Jingyan Zhou, Youzheng Wu,
Xiaodong He, and Bowen Zhou. 2020.

 
 The JDDC corpus: A
large-scale multi-turn Chinese dialogue dataset for E-commerce customer
service .

 
 In Proceedings of the 12th Language Resources and Evaluation
Conference , pages 459–466, Marseille, France. European Language Resources
Association.

 

 
 Chen et al. (2016) 
 
Yun-Nung (Vivian) Chen, Dilek Z. Hakkani-Tür, Gökhan Tür, Asli
Celikyilmaz, Jianfeng Gao, and Li Deng. 2016.

 
 Knowledge as a teacher: Knowledge-guided structural attention
networks.

 
 ArXiv , abs/1609.03286.

 

 
 Cho et al. (2021) 
 
Hyundong Cho, Basel Shbita, Kartik Shenoy, Shuai Liu, Nikhil Patel, Hitesh
Pindikanti, Jennifer Lee, and Jonathan May. 2021.

 
 Viola: A topic agnostic
generate-and-rank dialogue system .

 
 CoRR , abs/2108.11063.

 

 
 Compton et al. (2021) 
 
Rhys Compton, Ilya Valmianski, Li Deng, Costa Huang, Namit Katariya, Xavier
Amatriain, and Anitha Kannan. 2021.

 
 Medcod: A
medically-accurate, emotive, diverse, and controllable dialog system .

 
 In Proceedings of Machine Learning for Health , volume 158 of
 Proceedings of Machine Learning Research , pages 110–129. PMLR.

 

 
 Cui et al. (2021) 
 
Leyang Cui, Yu Wu, Shujie Liu, and Yue Zhang. 2021.

 
 Knowledge
enhanced fine-tuning for better handling unseen entities in dialogue
generation .

 
 In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing , pages 2328–2337, Online and Punta Cana,
Dominican Republic. Association for Computational Linguistics.

 

 
 Cui et al. (2020) 
 
Leyang Cui, Yu Wu, Shujie Liu, Yue Zhang, and Ming Zhou. 2020.

 
 MuTual: A
dataset for multi-turn dialogue reasoning .

 
 In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics , pages 1406–1416, Online. Association for
Computational Linguistics.

 

 
 Danescu-Niculescu-Mizil and
Lee (2011) 
 
Cristian Danescu-Niculescu-Mizil and Lillian Lee. 2011.

 
 Chameleons in imagined
conversations: A new approach to understanding coordination of linguistic
style in dialogs .

 
 In Proceedings of the 2nd Workshop on Cognitive Modeling and
Computational Linguistics , pages 76–87, Portland, Oregon, USA. Association
for Computational Linguistics.

 

 
 Dinan et al. (2019a) 
 
Emily Dinan, Varvara Logacheva, Valentin Malykh, Alexander H. Miller, Kurt
Shuster, Jack Urbanek, Douwe Kiela, Arthur Szlam, Iulian Serban, Ryan Lowe,
Shrimai Prabhumoye, Alan W. Black, Alexander I. Rudnicky, Jason Williams,
Joelle Pineau, Mikhail S. Burtsev, and Jason Weston. 2019a.

 
 The second conversational
intelligence challenge (convai2) .

 
 CoRR , abs/1902.00098.

 

 
 Dinan et al. (2019b) 
 
Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason
Weston. 2019b.

 
 Wizard of
wikipedia: Knowledge-powered conversational agents .

 
 In 7th International Conference on Learning Representations,
ICLR 2019, New Orleans, LA, USA, May 6-9, 2019 . OpenReview.net.

 

 
 Dolan and Brockett (2005) 
 
William B. Dolan and Chris Brockett. 2005.

 
 Automatically constructing
a corpus of sentential paraphrases .

 
 In Proceedings of the Third International Workshop on
Paraphrasing (IWP2005) .

 

 
 Dziri et al. (2018) 
 
Nouha Dziri, Ehsan Kamalloo, Kory W. Mathewson, and Osmar R. Zaïane.
2018.

 
 Augmenting neural response
generation with context-aware topical attention .

 
 CoRR , abs/1811.01063.

 

 
 Dziri et al. (2021) 
 
Nouha Dziri, Andrea Madotto, Osmar Zaïane, and Avishek Joey Bose. 2021.

 
 Neural path
hunter: Reducing hallucination in dialogue systems via path grounding .

 
 In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing , pages 2197–2214, Online and Punta Cana,
Dominican Republic. Association for Computational Linguistics.

 

 
 Eric et al. (2020) 
 
Mihail Eric, Rahul Goel, Shachi Paul, Abhishek Sethi, Sanchit Agarwal, Shuyang
Gao, Adarsh Kumar, Anuj Kumar Goyal, Peter Ku, and Dilek Hakkani-Tür.
2020.

 
 Multiwoz 2.1: A
consolidated multi-domain dialogue dataset with state corrections and state
tracking baselines .

 
 In Proceedings of The 12th Language Resources and Evaluation
Conference, LREC 2020, Marseille, France, May 11-16, 2020 , pages 422–428.
European Language Resources Association.

 

 
 Eric et al. (2017a) 
 
Mihail Eric, Lakshmi Krishnan, Francois Charette, and Christopher D. Manning.
2017a.

 
 Key-value retrieval
networks for task-oriented dialogue .

 
 In Proceedings of the 18th Annual SIGdial Meeting on
Discourse and Dialogue , pages 37–49, Saarbrücken, Germany. Association
for Computational Linguistics.

 

 
 Eric et al. (2017b) 
 
Mihail Eric, Lakshmi Krishnan, François Charette, and Christopher D.
Manning. 2017b.

 
 Key-value retrieval
networks for task-oriented dialogue .

 
 In Proceedings of the 18th Annual SIGdial Meeting on Discourse
and Dialogue, Saarbrücken, Germany, August 15-17, 2017 , pages 37–49.
Association for Computational Linguistics.

 

 
 Fan et al. (2021) 
 
Angela Fan, Claire Gardent, Chloé Braud, and Antoine Bordes. 2021.

 
 Augmenting transformers
with KNN-based composite memory for dialog .

 
 Transactions of the Association for Computational Linguistics ,
9:82–99.

 

 
 Feng et al. (2020) 
 
Song Feng, Hui Wan, Chulaka Gunasekara, Siva Patel, Sachindra Joshi, and Luis
Lastras. 2020.

 
 doc2dial: A
goal-oriented document-grounded dialogue dataset .

 
 In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP) , pages 8118–8128, Online. Association
for Computational Linguistics.

 

 
 Florez and Mueller (2019) 
 
Omar U. Florez and Erik Mueller. 2019.

 
 Aging memories generate more
fluent dialogue responses with memory networks .

 
 CoRR , abs/1911.08522.

 

 
 Gao et al. (2021) 
 
Yifan Gao, Jingjing Li, Michael R. Lyu, and Irwin King. 2021.

 
 Open-retrieval
conversational machine reading .

 
 CoRR , abs/2102.08633.

 

 
 Gopalakrishnan et al. (2019) 
 
Karthik Gopalakrishnan, Behnam Hedayatnia, Qinglang Chen, Anna Gottardi,
Sanjeev Kwatra, Anu Venkatesh, Raefer Gabriel, and Dilek Hakkani-Tür.
2019.

 
 Topical-chat:
Towards knowledge-grounded open-domain conversations .

 
 In Interspeech 2019, 20th Annual Conference of the
International Speech Communication Association, Graz, Austria, 15-19
September 2019 , pages 1891–1895. ISCA.

 

 
 Gou et al. (2021) 
 
Yanjie Gou, Yinjie Lei, Lingqiao Liu, Yong Dai, and Chunxu Shen. 2021.

 
 Contextualize knowledge bases with transformer for end-to-end task-oriented
dialogue systems .

 
 In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana,
Dominican Republic, 7-11 November, 2021 , pages 4300–4310. Association for
Computational Linguistics.

 

 
 Gu et al. (2020) 
 
Jia-Chen Gu, Tianda Li, Quan Liu, Xiaodan Zhu, Zhen-Hua Ling, Zhiming Su,
and Si Wei. 2020.

 
 Speaker-aware BERT for
multi-turn response selection in retrieval-based chatbots .

 
 CoRR , abs/2004.03588.

 

 
 He et al. (2017) 
 
He He, Anusha Balakrishnan, Mihail Eric, and Percy Liang. 2017.

 
 Learning symmetric
collaborative dialogue agents with dynamic knowledge graph embeddings .

 
 In Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers) , pages 1766–1776,
Vancouver, Canada. Association for Computational Linguistics.

 

 
 He et al. (2021) 
 
Huang He, Hua Lu, Siqi Bao, Fan Wang, Hua Wu, Zhengyu Niu, and Haifeng Wang.
2021.

 
 Learning to select external
knowledge with multi-scale negative sampling .

 
 CoRR , abs/2102.02096.

 

 
 Hedayatnia et al. (2020) 
 
Behnam Hedayatnia, Karthik Gopalakrishnan, Seokhwan Kim, Yang Liu, Mihail Eric,
and Dilek Hakkani-Tur. 2020.

 
 Policy-driven neural
response generation for knowledge-grounded dialog systems .

 
 In Proceedings of the 13th International Conference on Natural
Language Generation , pages 412–421, Dublin, Ireland. Association for
Computational Linguistics.

 

 
 Henderson et al. (2014) 
 
Matthew Henderson, Blaise Thomson, and Jason D. Williams. 2014.

 
 The second dialog state
tracking challenge .

 
 In Proceedings of the 15th Annual Meeting of the Special
Interest Group on Discourse and Dialogue (SIGDIAL) , pages 263–272,
Philadelphia, PA, U.S.A. Association for Computational Linguistics.

 

 
 Hills et al. (2012) 
 
Thomas T Hills, Michael N Jones, and Peter M Todd. 2012.

 
 Optimal foraging in semantic memory.

 
 Psychol. Rev. , 119(2):431–440.

 

 
 Jin et al. (2021a) 
 
Di Jin, Seokhwan Kim, and Dilek Hakkani-Tur. 2021a.

 
 Can I be of
further assistance? using unstructured knowledge access to improve
task-oriented conversational modeling .

 
 In Proceedings of the 1st Workshop on Document-grounded
Dialogue and Conversational Question Answering (DialDoc 2021) , pages
119–127, Online. Association for Computational Linguistics.

 

 
 Jin et al. (2021b) 
 
Di Jin, Seokhwan Kim, and Dilek Hakkani-Tür. 2021b.

 
 Can I be of further
assistance? using unstructured knowledge access to improve task-oriented
conversational modeling .

 
 CoRR , abs/2106.09174.

 

 
 Jurafsky and Martin (2008) 
 
Dan Jurafsky and James Martin. 2008.

 
 Speech and Language Processing , 2 edition.

 
 Pearson, Upper Saddle River, NJ.

 

 
 Kassawat et al. (2020) 
 
Firas Kassawat, Debanjan Chaudhuri, and Jens Lehmann. 2020.

 
 Incorporating joint
embeddings into goal-oriented dialogues with multi-task learning .

 
 CoRR , abs/2001.10468.

 

 
 Khatri et al. (2018) 
 
Chandra Khatri, Behnam Hedayatnia, Anu Venkatesh, Jeff Nunn, Yi Pan, Qing Liu,
Han Song, Anna Gottardi, Sanjeev Kwatra, Sanju Pancholi, Ming Cheng, Qinglang
Chen, Lauren Stubel, Karthik Gopalakrishnan, Kate Bland, Raefer Gabriel,
Arindam Mandal, Dilek Hakkani-Tür, Gene Hwang, Nate Michel, Eric
King, and Rohit Prasad. 2018.

 
 Advancing the state of the
art in open domain dialog systems through the alexa prize .

 
 CoRR , abs/1812.10757.

 

 
 Kim et al. (2020a) 
 
Byeongchang Kim, Jaewoo Ahn, and Gunhee Kim. 2020a.

 
 Sequential latent
knowledge selection for knowledge-grounded dialogue .

 
 In 8th International Conference on Learning Representations,
ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net.

 

 
 Kim et al. (2020b) 
 
Seokhwan Kim, Mihail Eric, Karthik Gopalakrishnan, Behnam Hedayatnia, Yang Liu,
and Dilek Hakkani-Tür. 2020b.

 
 Beyond domain
apis: Task-oriented conversational modeling with unstructured knowledge
access .

 
 In Proceedings of the 21th Annual Meeting of the Special
Interest Group on Discourse and Dialogue, SIGdial 2020, 1st virtual meeting,
July 1-3, 2020 , pages 278–289. Association for Computational Linguistics.

 

 
 Kim et al. (2019) 
 
Seokhwan Kim, Michel Galley, R. Chulaka Gunasekara, Sungjin Lee, Adam Atkinson,
Baolin Peng, Hannes Schulz, Jianfeng Gao, Jinchao Li, Mahmoud Adada, Minlie
Huang, Luis A. Lastras, Jonathan K. Kummerfeld, Walter S. Lasecki, Chiori
Hori, Anoop Cherian, Tim K. Marks, Abhinav Rastogi, Xiaoxue Zang, Srinivas
Sunkara, and Raghav Gupta. 2019.

 
 The eighth dialog system
technology challenge .

 
 CoRR , abs/1911.06394.

 

 
 Komeili et al. (2021) 
 
Mojtaba Komeili, Kurt Shuster, and Jason Weston. 2021.

 
 Internet-augmented dialogue
generation .

 
 CoRR , abs/2107.07566.

 

 
 Kumar et al. (2020) 
 
Gaurav Kumar, Rishabh Joshi, Jaspreet Singh, and Promod Yenigalla. 2020.

 
 AMUSED: A
multi-stream vector representation method for use in natural dialogue .

 
 In Proceedings of the 12th Language Resources and Evaluation
Conference , pages 750–758, Marseille, France. European Language Resources
Association.

 

 
 Le et al. (2016) 
 
Phong Le, Marc Dymetman, and Jean-Michel Renders. 2016.

 
 LSTM-based
mixture-of-experts for knowledge-aware dialogues .

 
 In Proceedings of the 1st Workshop on Representation Learning
for NLP , pages 94–99, Berlin, Germany. Association for Computational
Linguistics.

 

 
 Lee et al. (2019) 
 
Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019.

 
 Latent retrieval for
weakly supervised open domain question answering .

 
 In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics , pages 6086–6096, Florence, Italy.
Association for Computational Linguistics.

 

 
 Lehmann et al. (2015) 
 
Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas,
Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef,
Sören Auer, and Christian Bizer. 2015.

 
 Dbpedia - A large-scale,
multilingual knowledge base extracted from wikipedia .

 
 Semantic Web , 6(2):167–195.

 

 
 Li et al. (2021a) 
 
Bin Li, Encheng Chen, Hongru Liu, Yixuan Weng, Bin Sun, Shutao Li, Yongping
Bai, and Meiling Hu. 2021a.

 
 More but correct: Generating
diversified and entity-revised medical response .

 
 CoRR , abs/2108.01266.

 

 
 Li et al. (2020a) 
 
Linxiao Li, Can Xu, Wei Wu, Yufan Zhao, Xueliang Zhao, and Chongyang Tao.
2020a.

 
 Zero-resource knowledge-grounded dialogue generation .

 
 In Advances in Neural Information Processing Systems 33: Annual
Conference on Neural Information Processing Systems 2020, NeurIPS 2020,
December 6-12, 2020, virtual .

 

 
 Li et al. (2020b) 
 
Qintong Li, Piji Li, Zhumin Chen, and Zhaochun Ren. 2020b.

 
 Empathetic dialogue
generation via knowledge enhancing and emotion dependency modeling .

 
 CoRR , abs/2009.09708.

 

 
 Li et al. (2018) 
 
Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent
Charlin, and Chris Pal. 2018.

 
 Towards deep conversational recommendations .

 
 In Advances in Neural Information Processing Systems 31: Annual
Conference on Neural Information Processing Systems 2018, NeurIPS 2018,
December 3-8, 2018, Montréal, Canada , pages 9748–9758.

 

 
 Li et al. (2017) 
 
Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017.

 
 Dailydialog: A manually
labelled multi-turn dialogue dataset .

 
 In Proceedings of the Eighth International Joint Conference on
Natural Language Processing, IJCNLP 2017, Taipei, Taiwan, November 27 -
December 1, 2017 - Volume 1: Long Papers , pages 986–995. Asian Federation
of Natural Language Processing.

 

 
 Li et al. (2021b) 
 
Yu Li, Baolin Peng, Yelong Shen, Yi Mao, Lars Liden, Zhou Yu, and Jianfeng Gao.
2021b.

 
 Knowledge-grounded dialogue
generation with a unified knowledge representation .

 
 CoRR , abs/2112.07924.

 

 
 Lian et al. (2019) 
 
Rongzhong Lian, Min Xie, Fan Wang, Jinhua Peng, and Hua Wu. 2019.

 
 Learning to select knowledge
for response generation in dialog systems .

 
 CoRR , abs/1902.04911.

 

 
 Liang et al. (2021) 
 
Yunlong Liang, Fandong Meng, Ying Zhang, Yufeng Chen, Jinan Xu, and Jie Zhou.
2021.

 
 Infusing multi-source knowledge with heterogeneous graph neural network for
emotional conversation generation .

 
 Proceedings of the AAAI Conference on Artificial Intelligence ,
35(15):13343–13352.

 

 
 Lison and Tiedemann (2016) 
 
Pierre Lison and Jörg Tiedemann. 2016.

 
 Opensubtitles2016: Extracting large parallel corpora from movie and TV
subtitles .

 
 In Proceedings of the Tenth International Conference on
Language Resources and Evaluation LREC 2016, Portorož, Slovenia, May
23-28, 2016 . European Language Resources Association (ELRA).

 

 
 Liu et al. (2021a) 
 
Shilei Liu, Xiaofeng Zhao, Bochao Li, and Feiliang Ren. 2021a.

 
 Knowledge-grounded dialogue with reward-driven knowledge selection .

 
 In Natural Language Processing and Chinese Computing - 10th
CCF International Conference, NLPCC 2021, Qingdao, China, October 13-17,
2021, Proceedings, Part I , volume 13028 of Lecture Notes in Computer
Science , pages 455–466. Springer.

 

 
 Liu et al. (2021b) 
 
Shilei Liu, Xiaofeng Zhao, Bochao Li, Feiliang Ren, Longhui Zhang, and Shujuan
Yin. 2021b.

 
 A
Three-Stage Learning Framework for Low-Resource
Knowledge-Grounded Dialogue Generation .

 
 In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing , pages 2262–2272, Online and Punta Cana,
Dominican Republic. Association for Computational Linguistics.

 

 
 Lowe et al. (2015) 
 
Ryan Lowe, Nissan Pow, Iulian Serban, and Joelle Pineau. 2015.

 
 The ubuntu dialogue corpus:
A large dataset for research in unstructured multi-turn dialogue systems .

 
 CoRR , abs/1506.08909.

 

 
 Lowe et al. (2017) 
 
Ryan Thomas Lowe, Nissan Pow, Iulian Vlad Serban, Laurent Charlin, Chia-Wei
Liu, and Joelle Pineau. 2017.

 
 Training end-to-end dialogue systems with the ubuntu dialogue corpus .

 
 Dialogue Discourse , 8(1):31–65.

 

 
 Luo et al. (2021) 
 
Cheng Luo, Dayiheng Liu, Chanjuan Li, Li Lu, and Jiancheng Lv. 2021.

 
 Prediction, selection, and
generation: Exploration of knowledge-driven conversation system .

 
 CoRR , abs/2104.11454.

 

 
 Ma et al. (2020) 
 
Longxuan Ma, Wei-Nan Zhang, Runxin Sun, and Ting Liu. 2020.

 
 A
compare aggregate transformer for understanding document-grounded dialogue .

 
 In Findings of the Association for Computational Linguistics:
EMNLP 2020 , pages 1358–1367, Online. Association for Computational
Linguistics.

 

 
 Madotto et al. (2020) 
 
Andrea Madotto, Samuel Cahyawijaya, Genta Indra Winata, Yan Xu, Zihan Liu,
Zhaojiang Lin, and Pascale Fung. 2020.

 
 Learning
knowledge bases with parameters for task-oriented dialogue systems .

 
 In Findings of the Association for Computational Linguistics:
EMNLP 2020 , pages 2372–2394, Online. Association for Computational
Linguistics.

 

 
 Marzinotto et al. (2018) 
 
Gabriel Marzinotto, Jeremy Auguste, Frederic Bechet, Geraldine Damnati, and
Alexis Nasr. 2018.

 
 Semantic frame parsing for
information extraction : the CALOR corpus .

 
 In Proceedings of the Eleventh International Conference on
Language Resources and Evaluation (LREC 2018) , Miyazaki, Japan. European
Language Resources Association (ELRA).

 

 
 Mazaré et al. (2018) 
 
Pierre-Emmanuel Mazaré, Samuel Humeau, Martin Raison, and Antoine
Bordes. 2018.

 
 Training millions of
personalized dialogue agents .

 
 In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, Brussels, Belgium, October 31 - November 4,
2018 , pages 2775–2779. Association for Computational Linguistics.

 

 
 McTear (2021) 
 
Michael McTear. 2021.

 
 Conversational AI .

 
 Springer International Publishing.

 

 
 Mesnil et al. (2015) 
 
Grégoire Mesnil, Yann N. Dauphin, Kaisheng Yao, Yoshua Bengio, Li Deng,
Dilek Hakkani-Tür, Xiaodong He, Larry P. Heck, Gökhan
Tür, Dong Yu, and Geoffrey Zweig. 2015.

 
 Using recurrent
neural networks for slot filling in spoken language understanding .

 
 IEEE ACM Trans. Audio Speech Lang. Process. ,
23(3):530–539.

 

 
 Mey (2006) 
 
J.L. Mey. 2006.

 
 Pragmatics:
Overview .

 
 In Keith Brown, editor, Encyclopedia of Language Linguistics
(Second Edition) , second edition edition, pages 51–62. Elsevier, Oxford.

 

 
 Moghe et al. (2018) 
 
Nikita Moghe, Siddhartha Arora, Suman Banerjee, and Mitesh M. Khapra. 2018.

 
 Towards exploiting
background knowledge for building conversation systems .

 
 In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, Brussels, Belgium, October 31 - November 4,
2018 , pages 2322–2332. Association for Computational Linguistics.

 

 
 Moghe et al. (2020) 
 
Nikita Moghe, Priyesh Vijayan, Balaraman Ravindran, and Mitesh M. Khapra. 2020.

 
 On
incorporating structural information to improve dialogue response
generation .

 
 In Proceedings of the 2nd Workshop on Natural Language
Processing for Conversational AI , pages 11–24, Online. Association for
Computational Linguistics.

 

 
 Mohammad (2018) 
 
Saif Mohammad. 2018.

 
 Obtaining reliable
human ratings of valence, arousal, and dominance for 20,000 English words .

 
 In Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers) , pages 174–184,
Melbourne, Australia. Association for Computational Linguistics.

 

 
 Moon et al. (2019a) 
 
Seungwhan Moon, Pararth Shah, Anuj Kumar, and Rajen Subba. 2019a.

 
 OpenDialKG:
Explainable conversational reasoning with attention-based walks over
knowledge graphs .

 
 In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics , pages 845–854, Florence, Italy. Association
for Computational Linguistics.

 

 
 Moon et al. (2019b) 
 
Seungwhan Moon, Pararth Shah, Anuj Kumar, and Rajen Subba. 2019b.

 
 Opendialkg: Explainable
conversational reasoning with attention-based walks over knowledge graphs .

 
 In Proceedings of the 57th Conference of the Association for
Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,
2019, Volume 1: Long Papers , pages 845–854. Association for Computational
Linguistics.

 

 
 Mrksic et al. (2016) 
 
Nikola Mrksic, Diarmuid Ó Séaghdha, Tsung-Hsien Wen, Blaise
Thomson, and Steve J. Young. 2016.

 
 Neural belief tracker:
Data-driven dialogue state tracking .

 
 CoRR , abs/1606.03777.

 

 
 Novikova et al. (2017) 
 
Jekaterina Novikova, Ondrej Dusek, and Verena Rieser. 2017.

 
 The E2E dataset: New
challenges for end-to-end generation .

 
 In Proceedings of the 18th Annual SIGdial Meeting on Discourse
and Dialogue, Saarbrücken, Germany, August 15-17, 2017 , pages
201–206. Association for Computational Linguistics.

 

 
 Parthasarathi and Pineau (2018) 
 
Prasanna Parthasarathi and Joelle Pineau. 2018.

 
 Extending neural
generative conversational model using external knowledge sources .

 
 In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing , pages 690–695, Brussels, Belgium. Association
for Computational Linguistics.

 

 
 Pei et al. (2019) 
 
Jiahuan Pei, Arent Stienstra, Julia Kiseleva, and Maarten de Rijke. 2019.

 
 Sentnet: Source-aware
recurrent entity network for dialogue response selection .

 
 CoRR , abs/1906.06788.

 

 
 Peng et al. (2020) 
 
Shuke Peng, Feng Ji, Zehao Lin, Shaobo Cui, Haiqing Chen, and Yin Zhang. 2020.

 
 MTSS: learn from multiple
domain teachers and become a multi-domain dialogue expert .

 
 CoRR , abs/2005.10450.

 

 
 Pereira et al. (2018) 
 
Francisco Pereira, Bin Lou, Brianna Pritchett, Samuel Ritter, Samuel J
Gershman, Nancy Kanwisher, Matthew Botvinick, and Evelina Fedorenko. 2018.

 
 Toward a universal decoder of linguistic meaning from brain
activation.

 
 Nat. Commun. , 9(1).

 

 
 Pickering and Garrod (2004) 
 
Martin J. Pickering and Simon Garrod. 2004.

 
 Toward a
mechanistic psychology of dialogue .

 
 Behavioral and Brain Sciences , 27(2):169–190.

 

 
 Poria et al. (2019) 
 
Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik
Cambria, and Rada Mihalcea. 2019.

 
 MELD: A multimodal
multi-party dataset for emotion recognition in conversations .

 
 In Proceedings of the 57th Conference of the Association for
Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,
2019, Volume 1: Long Papers , pages 527–536. Association for Computational
Linguistics.

 

 
 Pulvermüller (2001) 
 
Friedemann Pulvermüller. 2001.

 
 Brain
reflections of words and their meaning .

 
 Trends in Cognitive Sciences , 5(12):517–524.

 

 
 Qin et al. (2019a) 
 
Lianhui Qin, Michel Galley, Chris Brockett, Xiaodong Liu, Xiang Gao, Bill
Dolan, Yejin Choi, and Jianfeng Gao. 2019a.

 
 Conversing by reading:
Contentful neural conversation with on-demand machine reading .

 
 In Proceedings of the 57th Conference of the Association for
Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,
2019, Volume 1: Long Papers , pages 5427–5436. Association for Computational
Linguistics.

 

 
 Qin et al. (2019b) 
 
Libo Qin, Yijia Liu, Wanxiang Che, Haoyang Wen, Yangming Li, and Ting Liu.
2019b.

 
 Entity-consistent
end-to-end task-oriented dialogue system with KB retriever .

 
 In Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP) , pages 133–142, Hong Kong,
China. Association for Computational Linguistics.

 

 
 Ramadan et al. (2018) 
 
Osman Ramadan, Paweł Budzianowski, and Milica Gašić. 2018.

 
 Large-scale
multi-domain belief tracking with knowledge sharing .

 
 In Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 2: Short Papers) , pages 432–437,
Melbourne, Australia. Association for Computational Linguistics.

 

 
 Rashkin et al. (2018) 
 
Hannah Rashkin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau. 2018.

 
 I know the feeling: Learning
to converse with empathy .

 
 CoRR , abs/1811.00207.

 

 
 Rashkin et al. (2019a) 
 
Hannah Rashkin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau.
2019a.

 
 Towards empathetic
open-domain conversation models: A new benchmark and dataset .

 
 In Proceedings of the 57th Conference of the Association for
Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2,
2019, Volume 1: Long Papers , pages 5370–5381. Association for Computational
Linguistics.

 

 
 Rashkin et al. (2019b) 
 
Hannah Rashkin, Eric Michael Smith, Margaret Li, and Y-Lan Boureau.
2019b.

 
 Towards empathetic
open-domain conversation models: A new benchmark and dataset .

 
 In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics , pages 5370–5381, Florence, Italy.
Association for Computational Linguistics.

 

 
 Reddy (1993) 
 
Michael J. Reddy. 1993.

 
 Metaphor and thought: The conduit metaphor: A case of frame conflict
in our language about language.

 

 
 Robertson and Zaragoza (2009) 
 
Stephen Robertson and Hugo Zaragoza. 2009.

 
 The probabilistic
relevance framework: Bm25 and beyond .

 
 Found. Trends Inf. Retr. , 3(4):333–389.

 

 
 Roller et al. (2021) 
 
Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Eric Michael Smith, Y-Lan Boureau, and Jason Weston. 2021.

 
 Recipes for
building an open-domain chatbot .

 
 In Proceedings of the 16th Conference of the European Chapter
of the Association for Computational Linguistics: Main Volume , pages
300–325, Online. Association for Computational Linguistics.

 

 
 Ross (2010) 
 
B.H. Ross. 2010.

 
 The
Psychology of Learning and Motivation: Advances in Research and Theory .

 
 Psychology of Learning and Motivation. Elsevier Science.

 

 
 Saha et al. (2017) 
 
Amrita Saha, Mitesh M. Khapra, and Karthik Sankaranarayanan. 2017.

 
 Multimodal dialogs (MMD):
A large-scale dataset for studying multimodal domain-aware conversations .

 
 CoRR , abs/1704.00200.

 

 
 Santhanam et al. (2020) 
 
Sashank Santhanam, Wei Ping, Raul Puri, Mohammad Shoeybi, Mostofa Patwary, and
Bryan Catanzaro. 2020.

 
 Local knowledge powered
conversational agents .

 
 CoRR , abs/2010.10150.

 

 
 Sap et al. (2019) 
 
Maarten Sap, Ronan Le Bras, Emily Allaway, Chandra Bhagavatula, Nicholas
Lourie, Hannah Rashkin, Brendan Roof, Noah A. Smith, and Yejin Choi. 2019.

 
 ATOMIC: an
atlas of machine commonsense for if-then reasoning .

 
 In The Thirty-Third AAAI Conference on Artificial
Intelligence, AAAI 2019, The Thirty-First Innovative Applications of
Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium
on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu,
Hawaii, USA, January 27 - February 1, 2019 , pages 3027–3035. AAAI Press.

 

 
 Schober (1998) 
 
MICHAEL F. Schober. 1998.

 
 Conversational evidence
for rethinking meaning .

 
 Social Research , 65(3):511–534.

 

 
 Shang et al. (2015) 
 
Lifeng Shang, Zhengdong Lu, and Hang Li. 2015.

 
 Neural responding
machine for short-text conversation .

 
 In Proceedings of the 53rd Annual Meeting of the Association
for Computational Linguistics and the 7th International Joint Conference on
Natural Language Processing of the Asian Federation of Natural Language
Processing, ACL 2015, July 26-31, 2015, Beijing, China, Volume 1: Long
Papers , pages 1577–1586. The Association for Computer Linguistics.

 

 
 Shuster et al. (2020) 
 
Kurt Shuster, Samuel Humeau, Antoine Bordes, and Jason Weston. 2020.

 
 Image-chat:
Engaging grounded conversations .

 
 In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics, ACL 2020, Online, July 5-10, 2020 , pages
2414–2429. Association for Computational Linguistics.

 

 
 Shuster et al. (2021) 
 
Kurt Shuster, Jack Urbanek, Emily Dinan, Arthur Szlam, and Jason Weston. 2021.

 
 Dialogue in
the wild: Learning from a deployed role-playing game with humans and bots .

 
 In Findings of the Association for Computational Linguistics:
ACL-IJCNLP 2021 , pages 611–624, Online. Association for Computational
Linguistics.

 

 
 Smith et al. (2020) 
 
Eric Michael Smith, Mary Williamson, Kurt Shuster, Jason Weston, and Y-Lan
Boureau. 2020.

 
 Can you put it
all together: Evaluating conversational agents’ ability to blend skills .

 
 In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics, ACL 2020, Online, July 5-10, 2020 , pages
2021–2030. Association for Computational Linguistics.

 

 
 Speer et al. (2017) 
 
Robyn Speer, Joshua Chin, and Catherine Havasi. 2017.

 
 Conceptnet 5.5: An open multilingual graph of general knowledge .

 
 In Proceedings of the Thirty-First AAAI Conference on
Artificial Intelligence, February 4-9, 2017, San Francisco, California,
USA , pages 4444–4451. AAAI Press.

 

 
 Sperber (1995) 
 
Daniel Sperber. 1995.

 
 Sperber: Relevance: Communication cognition (cloth) .

 

 
 Su et al. (2018) 
 
Shang-Yu Su, Kai-Ling Lo, Yi-Ting Yeh, and Yun-Nung Chen. 2018.

 
 Natural language
generation by hierarchical decoding with linguistic patterns .

 
 In Proceedings of the 2018 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 2 (Short Papers) , pages 61–66, New Orleans, Louisiana.
Association for Computational Linguistics.

 

 
 Sukhbaatar et al. (2015) 
 
Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. 2015.

 
 Weakly supervised memory
networks .

 
 CoRR , abs/1503.08895.

 

 
 Sun et al. (2021) 
 
Yajing Sun, Yue Hu, Luxi Xing, Yuqiang Xie, and Xiangpeng Wei. 2021.

 
 Know deeper:
Knowledge-conversation cyclic utilization mechanism for open-domain dialogue
generation .

 
 CoRR , abs/2107.07771.

 

 
 Tan et al. (2020) 
 
Chao-Hong Tan, Xiaoyu Yang, Zi’ou Zheng, Tianda Li, Yufei Feng, Jia-Chen
Gu, Quan Liu, Dan Liu, Zhen-Hua Ling, and Xiaodan Zhu. 2020.

 
 Learning to retrieve
entity-aware knowledge and generate responses with copy mechanism for
task-oriented dialogue systems .

 
 CoRR , abs/2012.11937.

 

 
 Tao et al. (2021) 
 
Chongyang Tao, Jiazhan Feng, Wei Wu, Rui Yan, and Daxin Jiang. 2021.

 
 A survey on response selection for retrieval-based dialogues .

 
 In IJCAI 2021 .

 

 
 Tigunova et al. (2019) 
 
Anna Tigunova, Andrew Yates, Paramita Mirza, and Gerhard Weikum. 2019.

 
 Listening between
the lines: Learning personal attributes from conversations .

 
 In The World Wide Web Conference , WWW ’19, page 1818–1828,
New York, NY, USA. Association for Computing Machinery.

 

 
 Wang et al. (2021a) 
 
Dingmin Wang, Ziyao Chen, Wanwei He, Li Zhong, Yunzhe Tao, and Min Yang.
2021a.

 
 A
template-guided hybrid pointer network for knowledge-based task-oriented
dialogue systems .

 
 In Proceedings of the 1st Workshop on Document-grounded
Dialogue and Conversational Question Answering (DialDoc 2021) , pages 18–28,
Online. Association for Computational Linguistics.

 

 
 Wang et al. (2020) 
 
Jian Wang, Junhao Liu, Wei Bi, Xiaojiang Liu, Kejing He, Ruifeng Xu, and Min
Yang. 2020.

 
 Improving knowledge-aware dialogue generation via knowledge base question
answering .

 
 In The Thirty-Fourth AAAI Conference on Artificial
Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of
Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium
on Educational Advances in Artificial Intelligence, EAAI 2020, New York,
NY, USA, February 7-12, 2020 , pages 9169–9176. AAAI Press.

 

 
 Wang et al. (2021b) 
 
Lingzhi Wang, Huang Hu, Lei Sha, Can Xu, Kam-Fai Wong, and Daxin Jiang.
2021b.

 
 Finetuning large-scale
pre-trained language models for conversational recommendation with knowledge
graph .

 
 CoRR , abs/2110.07477.

 

 
 Wen et al. (2018) 
 
Haoyang Wen, Yijia Liu, Wanxiang Che, Libo Qin, and Ting Liu. 2018.

 
 Sequence-to-sequence
learning for task-oriented dialogue with dialogue state representation .

 
 In Proceedings of the 27th International Conference on
Computational Linguistics , pages 3781–3792, Santa Fe, New Mexico, USA.
Association for Computational Linguistics.

 

 
 Williams et al. (2017) 
 
Adina Williams, Nikita Nangia, and Samuel R. Bowman. 2017.

 
 A broad-coverage challenge
corpus for sentence understanding through inference .

 
 CoRR , abs/1704.05426.

 

 
 Wu et al. (2017) 
 
Yu Wu, Wei Wu, Chen Xing, Ming Zhou, and Zhoujun Li. 2017.

 
 Sequential matching
network: A new architecture for multi-turn response selection in
retrieval-based chatbots .

 
 In Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 -
August 4, Volume 1: Long Papers , pages 496–505. Association for
Computational Linguistics.

 

 
 Wu et al. (2020) 
 
Zeqiu Wu, Michel Galley, Chris Brockett, Yizhe Zhang, Xiang Gao, Chris Quirk,
Rik Koncel-Kedziorski, Jianfeng Gao, Hannaneh Hajishirzi, Mari Ostendorf,
and Bill Dolan. 2020.

 
 A controllable model of
grounded response generation .

 
 CoRR , abs/2005.00613.

 

 
 Wu et al. (2021) 
 
Zeqiu Wu, Bo-Ru Lu, Hannaneh Hajishirzi, and Mari Ostendorf. 2021.

 
 DIALKI:
Knowledge identification in conversational systems through dialogue-document
contextualization .

 
 In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing , pages 1852–1863, Online and Punta Cana,
Dominican Republic. Association for Computational Linguistics.

 

 
 Xu et al. (2021a) 
 
Song Xu, Haoran Li, Peng Yuan, Yujia Wang, Youzheng Wu, Xiaodong He, Ying Liu,
and Bowen Zhou. 2021a.

 
 K-PLUG:
Knowledge-injected pre-trained language model for natural language
understanding and generation in E-commerce .

 
 In Findings of the Association for Computational Linguistics:
EMNLP 2021 , pages 1–17, Punta Cana, Dominican Republic. Association for
Computational Linguistics.

 

 
 Xu et al. (2021b) 
 
Yan Xu, Etsuko Ishii, Zihan Liu, Genta Indra Winata, Dan Su, Andrea Madotto,
and Pascale Fung. 2021b.

 
 Retrieval-free
knowledge-grounded dialogue response generation with adapters .

 
 CoRR , abs/2105.06232.

 

 
 Yang et al. (2020a) 
 
Shiquan Yang, Rui Zhang, and Sarah Erfani. 2020a.

 
 GraphDialog: Integrating graph knowledge into end-to-end task-oriented
dialogue systems .

 
 In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP) , pages 1878–1888, Online. Association
for Computational Linguistics.

 

 
 Yang et al. (2020b) 
 
Shiquan Yang, Rui Zhang, and Sarah M. Erfani. 2020b.

 
 Graphdialog:
Integrating graph knowledge into end-to-end task-oriented dialogue systems .

 
 In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020 ,
pages 1878–1888. Association for Computational Linguistics.

 

 
 Yavuz et al. (2019) 
 
Semih Yavuz, Abhinav Rastogi, Guan-Lin Chao, and Dilek Hakkani-Tur. 2019.

 
 DeepCopy: Grounded
response generation with hierarchical pointer networks .

 
 In Proceedings of the 20th Annual SIGdial Meeting on Discourse
and Dialogue , pages 122–132, Stockholm, Sweden. Association for
Computational Linguistics.

 

 
 Ye et al. (2019) 
 
Hao-Tong Ye, Kai-Ling Lo, Shang-Yu Su, and Yun-Nung Chen. 2019.

 
 Knowledge-grounded response
generation with deep attentional latent-variable model .

 
 CoRR , abs/1903.09813.

 

 
 Yin et al. (2016a) 
 
Jun Yin, Xin Jiang, Zhengdong Lu, Lifeng Shang, Hang Li, and Xiaoming Li.
2016a.

 
 Neural generative
question answering .

 
 In Proceedings of the Workshop on Human-Computer Question
Answering , pages 36–42, San Diego, California. Association for
Computational Linguistics.

 

 
 Yin et al. (2016b) 
 
Jun Yin, Xin Jiang, Zhengdong Lu, Lifeng Shang, Hang Li, and Xiaoming Li.
2016b.

 
 Neural generative
question answering .

 
 In Proceedings of the Twenty-Fifth International Joint
Conference on Artificial Intelligence, IJCAI 2016, New York, NY, USA, 9-15
July 2016 , pages 2972–2978. IJCAI/AAAI Press.

 

 
 Yoshino et al. (2019) 
 
Koichiro Yoshino, Chiori Hori, Julien Perez, Luis Fernando D’Haro, Lazaros
Polymenakos, R. Chulaka Gunasekara, Walter S. Lasecki, Jonathan K.
Kummerfeld, Michel Galley, Chris Brockett, Jianfeng Gao, Bill Dolan, Xiang
Gao, Huda AlAmri, Tim K. Marks, Devi Parikh, and Dhruv Batra. 2019.

 
 Dialog system technology
challenge 7 .

 
 CoRR , abs/1901.03461.

 

 
 Yu et al. (2020) 
 
Dian Yu, Kai Sun, Claire Cardie, and Dong Yu. 2020.

 
 Dialogue-based
relation extraction .

 
 In Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics , pages 4927–4940, Online. Association for
Computational Linguistics.

 

 
 Zahiri and Choi (2018) 
 
Sayyed M. Zahiri and Jinho D. Choi. 2018.

 
 Emotion detection on TV show transcripts with sequence-based convolutional
neural networks .

 
 In The Workshops of the The Thirty-Second AAAI Conference on
Artificial Intelligence, New Orleans, Louisiana, USA, February 2-7, 2018 ,
volume WS-18 of AAAI Technical Report , pages 44–52. AAAI Press.

 

 
 Zang et al. (2020) 
 
Xiaoxue Zang, Abhinav Rastogi, Srinivas Sunkara, Raghav Gupta, Jianguo Zhang,
and Jindong Chen. 2020.

 
 MultiWOZ 2.2 : A dialogue dataset with additional annotation corrections
and state tracking baselines .

 
 In Proceedings of the 2nd Workshop on Natural Language
Processing for Conversational AI , pages 109–117, Online. Association for
Computational Linguistics.

 

 
 Zeng et al. (2020) 
 
Guangtao Zeng, Wenmian Yang, Zeqian Ju, Yue Yang, Sicheng Wang, Ruisi Zhang,
Meng Zhou, Jiaqi Zeng, Xiangyu Dong, Ruoyu Zhang, Hongchao Fang, Penghui Zhu,
Shu Chen, and Pengtao Xie. 2020.

 
 MedDialog: Large-scale medical dialogue datasets .

 
 In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP) , pages 9241–9250, Online. Association
for Computational Linguistics.

 

 
 Zhang et al. (2018a) 
 
Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason
Weston. 2018a.

 
 Personalizing dialogue
agents: I have a dog, do you have pets too? 

 
 In Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20,
2018, Volume 1: Long Papers , pages 2204–2213. Association for Computational
Linguistics.

 

 
 Zhang et al. (2019) 
 
Yangjun Zhang, Pengjie Ren, and Maarten de Rijke. 2019.

 
 Improving background based
conversation with context-aware knowledge pre-selection .

 
 CoRR , abs/1906.06685.

 

 
 Zhang et al. (2018b) 
 
Zhuosheng Zhang, Jiangtong Li, Pengfei Zhu, Hai Zhao, and Gongshen Liu.
2018b.

 
 Modeling multi-turn
conversation with deep utterance aggregation .

 
 In Proceedings of the 27th International Conference on
Computational Linguistics , pages 3740–3752, Santa Fe, New Mexico, USA.
Association for Computational Linguistics.

 

 
 Zhang et al. (2021) 
 
Zhuosheng Zhang, Siru Ouyang, Hai Zhao, Masao Utiyama, and Eiichiro Sumita.
2021.

 
 Smoothing
dialogue states for open conversational machine reading .

 
 In Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing , pages 3685–3696, Online and Punta Cana,
Dominican Republic. Association for Computational Linguistics.

 

 
 Zhao et al. (2020a) 
 
Xueliang Zhao, Wei Wu, Chongyang Tao, Can Xu, Dongyan Zhao, and Rui Yan.
2020a.

 
 Low-resource
knowledge-grounded dialogue generation .

 
 In 8th International Conference on Learning Representations,
ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net.

 

 
 Zhao et al. (2020b) 
 
Xueliang Zhao, Wei Wu, Can Xu, Chongyang Tao, Dongyan Zhao, and Rui Yan.
2020b.

 
 Knowledge-grounded dialogue generation with pre-trained language models .

 
 In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP) , pages 3377–3390, Online. Association
for Computational Linguistics.

 

 
 Zhou et al. (2018a) 
 
Hao Zhou, Minlie Huang, Tianyang Zhang, Xiaoyan Zhu, and Bing Liu.
2018a.

 
 Emotional chatting machine: Emotional conversation generation with internal
and external memory .

 
 Proceedings of the AAAI Conference on Artificial Intelligence ,
32(1).

 

 
 Zhou et al. (2020) 
 
Hao Zhou, Chujie Zheng, Kaili Huang, Minlie Huang, and Xiaoyan Zhu. 2020.

 
 Kdconv: A chinese
multi-domain dialogue dataset towards multi-turn knowledge-driven
conversation .

 
 CoRR , abs/2004.04100.

 

 
 Zhou et al. (2018b) 
 
Kangyan Zhou, Shrimai Prabhumoye, and Alan W. Black. 2018b.

 
 A dataset for document
grounded conversations .

 
 In Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, Brussels, Belgium, October 31 - November 4,
2018 , pages 708–713. Association for Computational Linguistics.

 

 
 Zhou et al. (2021a) 
 
Pei Zhou, Karthik Gopalakrishnan, Behnam Hedayatnia, Seokhwan Kim, Jay Pujara,
Xiang Ren, Yang Liu, and Dilek Hakkani-Tur. 2021a.

 
 Commonsense-focused
dialogues for response generation: An empirical study .

 
 CoRR , abs/2109.06427.

 

 
 Zhou et al. (2021b) 
 
Pei Zhou, Behnam Hedayatnia, Karthik Gopalakrishnan, Seokhwan Kim, Jay Pujara,
Xiang Ren, Yang Liu, and Dilek Hakkani-Tur. 2021b.

 
 Think
before you speak: Learning to generate implicit knowledge for response
generation by self-talk .

 
 In Proceedings of the 3rd Workshop on Natural Language
Processing for Conversational AI , pages 251–253, Online. Association for
Computational Linguistics.

 

 
 Zhu et al. (2020) 
 
Tiangang Zhu, Yue Wang, Haoran Li, Youzheng Wu, Xiaodong He, and Bowen Zhou.
2020.

 
 Multimodal
joint attribute prediction and value extraction for E-commerce product .

 
 In Proceedings of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP) , pages 2129–2139, Online. Association
for Computational Linguistics.

 

 
 Zhu et al. (2017) 
 
Wenya Zhu, Kaixiang Mo, Yu Zhang, Zhangbin Zhu, Xuezheng Peng, and Qiang Yang.
2017.

 
 Flexible end-to-end dialogue
system for knowledge grounded conversation .

 
 CoRR , abs/1709.04264.