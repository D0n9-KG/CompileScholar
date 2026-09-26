Computational Models to Study Language Processing in the Human Brain: A Survey 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2403.13368v1 [cs.CL] 20 Mar 2024 
 
 

# Computational Models to Study Language Processing in the Human Brain: A Survey

 
 
 Shaonan Wang
 
 Affiliation: Institute of Automation Chinese Academy of Science
 
 Email: shaonan.wang@nlpr.ia.ac.cn 
 
    
 Jingyuan Sun
 
 Affiliation: KU Leuven
 
 Email: jingyuan.sun@kuleuven.be 
 
    
 Yunhao Zhang
 
 Affiliation: Institute of Automation Chinese Academy of Science
 
 Email: zhangyunhao2021@ia.ac.cn 
 
    
 Nan Lin
 
 Affiliation: Institute of Psychology Chinese Academy of Science
 
 Email: linn@psych.ac.cn 
 
    
 Marie-Francine Moens
 
 Affiliation: KU Leuven
 
 Email: sien.moens@kuleuven.be 
 
    
 Chengqing Zong
 
 Affiliation: Institute of Automation Chinese Academy of Science
 
 Email: cqzong@nlpr.ia.ac.cn 
 

 Abstract 
 
 Despite differing from the human language processing mechanism in implementation and algorithms, current language models demonstrate remarkable human-like or surpassing language capabilities. Should computational language models be employed in studying the brain, and if so, when and how? To delve into this topic, this paper reviews efforts in using computational models for brain research, highlighting emerging trends. To ensure a fair comparison, the paper evaluates various computational models using consistent metrics on the same dataset. Our analysis reveals that no single model outperforms others on all datasets, underscoring the need for rich testing datasets and rigid experimental control to draw robust conclusions in studies involving computational models.

 
 
 

## 1 Introduction

 
 Can computational models unravel the mysteries of language processing in the human brain? As computational language models advance, interdisciplinary research increasingly leverages them to study the brain, raising questions about their benefits and conditions of effectiveness.

 
 
 Critics argue that substantial disparities between these models and the human brain make them inappropriate for direct brain mechanism studies. One major critique focuses on the limited scope of studies establishing correlations between brain activations and model representations [ 1 , 2 ] . For instance, Guest and Martin (2023) caution against using artificial neural networks (ANNs) to conclude the mind and brain, citing potential logical fallacies: it is inappropriate to assert that if the model predicts neural activity, then the model represents the neural system. Conversely, stating that if the model represents the neural system, it predicts neural activity is appropriate [ 2 ] . A second critique underscores differences in objective functions, learning rules, and architectures when comparing models with human language processing. In the vision domain, it questions the general approach modeling human object recognition by optimizing classification performance may be misguided for a theoretical reason, namely, the human visual system may not be optimized to classify images [ 3 , 4 ] . Similarly, concerns extend to the word prediction objective function in language processing [ 5 ] . The third critique argues that computational model findings lack novelty, often restating existing knowledge. According to Barsalou (2017),” Neural encoding research tells us little about the nature of this processing. While mapping concepts between Marr’s computational and implementation levels to support neural encoding and decoding, this approach ignores Marr’s algorithmic level, central for understanding the mechanisms that implement cognition. [ 6 ] ”.

 
 
 Despite valid concerns, as George E. P. Box noted, ”All models are wrong, but some are useful.” Advanced computational language models, despite fundamental implementation differences, emulate human language abilities. Viewing them as potential frameworks for understanding brain mechanisms offers three key advantages. Firstly, computational models efficiently quantify cognitive metrics and identify neural correlates in language processing. Compared to human annotations, they are cost-effective for large dataset annotation and excel in handling complex metrics like syntactic complexity. Utilizing these models for brain correlation provides greater flexibility in analyzing naturalistic data, while traditional contrasting methods are mainly used in controlled experiments [ 7 , 8 , 9 , 10 , 11 , 12 , 13 ] . Secondly, computational models, especially large language models, demonstrate human-like behavior in diverse language tasks, offering a way to piece together information from different modules and taking a holistic perspective to delve into brain language processing mechanisms. Integrating fragmented knowledge and combining disciplines, as emphasized by Kriegeskorte and Douglas (2018), is crucial for gaining theoretical insights in brain-computational models [ 14 ] . Thirdly, these models generate prospective hypotheses to validate the linguistic phenomena underlying the brain [ 15 , 16 , 17 ] . If a model mimics human performance only with a specific structure, it implies that this architecture may capture information explaining observed behavior in the brain. In support of this notion, Kanwisher et al. (2023) propose deep networks can answer ”why” questions about the brain, indicating optimization for a task drives observed phenomena.

 
 
 To conduct a thorough examination of the efficacy of computational models in studying language processing within the brain, this research delves into the distinctive contributions made by statistical language models (SLMs), shallow embedding models (SEMs), and large language models (LLMs) over time. The study aims to elucidate how these models uniquely advance brain investigation, exploring specific contexts and methodologies. In the forthcoming sections, Section 2 provides the terminology for different computational models and cognitive measures. In section 3, we delve into the three advantages offered by these models, reviewing existing work on these aspects, and presenting a fair comparison of these models using the same training dataset and evaluation metric. Section 4 concludes the study, summarizing key findings and implications.

 
 
 

## 2 Computational models and cognitive metrics

 
 Over the last two decades, three primary computational models have been employed to investigate brain language processing [ 18 , 19 ] . Each of these models possesses unique structures and language capabilities, serving as valuable tools for quantification and simulation in the study of the human brain. By quantifying cognitive metrics and linking them to the brain, these models offer a means of quantifying the complexities of language processing in the brain, thereby bridging the gap between language symbols and neural processes.

 
 

### 2.1 Cognitive metrics

 
 In the context of language and the brain, cognitive metrics encompass measures of comprehension, information processing speed, working memory load, and various factors associated with how the brain interacts with and processes linguistic information. These metrics include reaction time, error rate, eye-tracking measures, surprisal, entropy, syntactic complexity, and semantic processing. Here, we highlight three of the most commonly utilized metrics:

 
 
 
 • 
 
 Probability-related Mertics One of the most common and direct metrics for measuring cognitive load is Surprisal, which quantifies the uncertainty associated with the occurrence of a word. Mathematically, it is defined as the negative logarithm of the probability of the word: S ⁡ ( x ) = − log ⁡ P ⁡ ( x ) S(x)=-\log P(x) . Another commonly used metric that focuses on information gain is Entropy Reduction, representing the decrease in uncertainty when a specific word is introduced. Mathematically, it is defined as the difference in entropy H ⁡ ( x ) H(x) before and after combining with the word x i + 1 x_{i+1} : H ⁡ ( x ) − H ⁡ ( x | x i + 1 ) H(x)-H(x|x_{i+1}) with H ( X ) = − ∑ i P ( x i ) ⋅ log 2 ( P ( x i ) ) H(X)=-\sum_{i}P(x_{i})\cdot\log_{2}(P(x_{i})) .

 

 • 
 
 Syntactic-Related Metric Parsing strategies analyze sentence structure in natural language. Notable approaches include top-down, bottom-up, and left-corner parsing. Top-down begins with overall sentence structure, bottom-up starts with individual words, and left-corner combines both, starting with leftmost constituents and expanding the parse tree based on the input sentence.

 

 • 
 
 Semantic and syntactic-Related Metric Representations in language involve encoding the meaning and syntactic structure of words, phrases, sentences, and discourses. Quality at each level influences overall language understanding. Effective representations capture hierarchical structure and semantic relationships within and between linguistic elements.

 

 
 
 
 

### 2.2 Statistical language models

 
 Statistical language models estimate word or phrase sequence probabilities by analyzing relationships in a corpus. They predict the next word based on preceding words and incorporate explicit structures like grammar and parsing strategies. Here, we present illustrations of two extensively employed models utilized in the investigation of brain language processing.

 
 
 Figure 1: Statistical language models. (a) 3-gram language model, which estimates the word probability based on context. (b) structural language model, which incorporates syntax and parsing for understanding word interplay and sentence organization. 
 
 
 The n-gram language model estimates word probabilities based on word and n-gram frequencies. For example, with n=3, the probability of a word like “mat” in a sentence is calculated by analyzing the frequency of the preceding three words (“on the mat”) divided by the frequency of the target word itself (“mat”), as shown in Figure 1(a).

 
 
 The structural language model utilizes formal grammars like Context-Free Grammar (CFG), Minimalist Grammar (MG), and Combinatory Categorial Grammar (CCG) to calculate word probabilities based on preceding words and syntactic structure of the sentence [ 20 ] . For example, in Probabilistic Context-Free Grammar (PCFG), operation rules with associated probabilities are selected for non-terminal symbols like noun phrases and verb phrases (S → \to NP VP). As shown in Figure 1(b), to calculate a sentence’s probability, first, operation rules for non-terminal symbols (e.g., S → \to NP VP) are chosen. These rules have individual probabilities, multiplied to compute the sentence’s probability. This product signifies the overall likelihood of generating the sentence from the given grammar. Parsing strategies like top-down explore grammar rules from the root down, with probabilities guiding rule selection. In Figure 1(b), top-down parsing begins from the root, advancing downward through the tree. Operation rule selection relies on the probability of expanding non-terminals. The process yields node counts, indicating rules needed per word. For instance, “sat” requires two steps: VP → \to P NP and P → \to on.

 
 
 

### 2.3 Shallow Embedding Models

 
 After statistical language models, pioneering shallow embedding models encode meaning through the distributional hypothesis, which posits that similar words occur in similar contexts. These models quantify semantics and transform linguistic entities into vector spaces [ 21 , 22 , 23 ] .

 
 
 Shallow embedding models, with diverse training data, design, and goals, share a common aim: capturing semantic nuances for improved language processing tasks. Notable examples include Word2Vec [ 24 ] and the Recurrent Neural Network Language Model (RNNLM) [ 25 ] , both exemplifying algorithms like Continuous Bag of Words (CBOW) and Skip-Gram within the Word2Vec model. Consider CBOW as illustrated in Figure 2(a). This model predicts a target word from its context. Mathematically, the algorithm is described as:

 

 
 | 
 L CBOW = 1 T ​ ∑ t = 1 T log ⁡ p ⁡ ( w t | w t − c , … , w t + c ) L_{\text{CBOW}}=\frac{1}{T}\sum_{t=1}^{T}\log p(w_{t}|w_{t-c},\dots,w_{t+c}) | 
 | 
 

 RNNLM, as illustrated in Figure 2(b), on the other hand, leverages RNNs to predict the next word in a sequence, considering all the previous words. Its formulation can be given by:

 

 
 | 
 p ⁡ ( w t | w 1 , w 2 , … , w t − 1 ) = softmax ​ ( h t ) p(w_{t}|w_{1},w_{2},\dots,w_{t-1})=\text{softmax}(h_{t}) | 
 | 
 

 Where h t h_{t} is the hidden state at time t t , which is a function of the previous hidden state and the current input.

 
 
 Figure 2: Shallow Embedding Models: (a) CBOW (Word2Vec) model that learn word embeddings by predicting target word based on context within a window, (b) RNN model that learning word embeddings by predicting next word based on all previous context. 
 
 
 Both models learn word representations through the utilization of hidden layers in neural networks. CBOW employs static word representations, while RNN acquires contextualized word representations. These representations capture the semantic and syntactic relations between words.

 
 
 

### 2.4 Large Language Models

 
 Following the shallow embedding model era, large language models have revolutionized natural language processing, excelling in diverse tasks within a single model and quickly adapting to new tasks with minimal examples [ 26 , 27 , 28 , 29 , 30 , 31 ] . Their remarkable improvement in generating human-like text and surpassing human-like performance on various tasks may offer new insights for brain language studies.

 
 
 GPT-3 and similar large language models [ 32 ] distinguish themselves from shallow embedding models through extensive use of massive internet data, deep architectures with numerous parameters, and sophisticated algorithms for optimizing objective functions. This enables rapid in-context learning, facilitating swift adaptation to new tasks and providing human-like responses.

 
 
 Figure 3: Large Language Models: (a) BERT: Learns word embeddings through masked language modeling, predicting randomly masked words based on surrounding context. (b) GPT: Learns word embeddings via next word prediction, predicting the next word based on all preceding context. 
 
 
 Large language models rely on transformers, employing multiple attention layers to capture intricate relationships in input data. This hierarchical architecture enables selective focus on different parts of the sequence, enhancing contextual understanding. A primary technique used in pretraining large language models is the masked language modeling (MLM), as shown in Figure 3(a). Here, certain words in a sentence are masked out, and the model is trained to predict the masked word based on its context. The objective for MLM can be represented as:

 

 
 | 
 L MLM = − log ⁡ p ⁡ ( w i | w context ) L_{\text{MLM}}=-\log p(w_{i}|w_{\text{context}}) | 
 | 
 

 where w i w_{i} is the masked word and w context w_{\text{context}} are the surrounding words.
Another common training approach is next word prediction (NWP) as shown in Figure 3(b), where the model predicts the next word in a sequence given the preceding words. The objective for this is:

 

 
 | 
 L NWP = − log ⁡ p ⁡ ( w t + 1 | w 1 , w 2 , … , w t ) L_{\text{NWP}}=-\log p(w_{t+1}|w_{1},w_{2},\dots,w_{t}) | 
 | 
 

 
 
 Given the human-like performance and responses achieved by large language models, exploring the alignment of their objective functions, structural components, attention mechanisms, and encoded meanings with the human brain could yield intriguing insights.

 
 
 
 

## 3 Utilizing computational models in brain studies

 
 This section reviews studies that employed computational models to quantify cognitive measures like surprisal, entropy, and semantic representations to study the neural mechanism underlying brain language processing.

 
 

### 3.1 Quantifying cognitive load

 
 Cognitive measures connect discrete language symbols with continuous neural processes in comprehension. For instance, higher surprisal or entropy reduction values signify increased cognitive load, demanding more effort for effective understanding. All computational models, including statistical, shallow embedding, and large language models, compute quantitative cognitive measures.

 
 
 Prior research on statistical language models for surprisal consistently demonstrates effectiveness, particularly in eye-tracking studies, linking higher surprisal values to slower reading speeds, indicative of increased language complexity [ 33 , 34 , 35 ] . In event-related potentials, surprisal shows a positive association with the amplitude of the N400 component, supporting that surprisal as a generally applicable measure of processing difficulty during language comprehension [ 36 , 37 ] . Functional magnetic resonance imaging (fMRI) studies have demonstrated the predictive capacity of surprisal in determining activation patterns [ 38 , 7 , 39 ] . Notably, Shain et al. (2020) observed that surprisal, derived from both structure-based and n-gram models, influences the language network but does not impact the domain-general multiple-demand network, indicating that predictive coding in the brain’s response to language is domain-specific and that these predictions are sensitive both to local word co-occurrence patterns and to hierarchical structure [ 40 ] . Furthermore, Heilbron et al. (2022) employed GPT-2 to compute various types of surprisal, revealing that brain responses to words are modulated by pervasive predictions spanning from phonemes and syntactic categories (parts of speech) to semantics [ 41 ] .

 
 
 Prior research predominantly employs statistical language models to compute entropy reduction, which correlates with cognitive processing difficulty [ 42 , 43 ] . This correlation holds across naturalistic text [ 44 , 45 ] and controlled experiments [ 46 ] . Intracranial signals from the anterior Inferior Temporal Sulcus (aITS) and posterior Inferior Temporal Gyrus (pITG) also correlate with word-by-word Entropy Reduction values derived from phrase structure grammars for languages. In the anterior region, this correlation persists even when combined with surprisal co-predictors from PCFG and N-gram models, confirming that the brain’s temporal lobe houses a parsing function. This function’s incremental processing difficulty profile reflects changes in grammatical uncertainty [ 47 ] . A comprehensive analysis for a detailed comparison between surprisal and entropy reduction is provided by John Hale (2016) [ 48 ] .

 
 

#### 3.1.1 Formalizing syntactic processing

 
 Parsing strategies like top-down, bottom-up, and left-corner in language processing employ unique syntactic analysis methods. It’s still debatable whether syntax is encoded in the brain. Assuming it is, researchers compared word-by-word metrics from different parsing strategies with neural activity measurements to gain profound insights into the gradual formation of syntactic structures in the brain. Two frequently used metrics are rule counts and node counts. Prior research shows a correlation between parser-applied rules, node count (refer to Figure 1(b)), and related neural activity [ 49 , 38 ] . For example, comparing bottom-up and left-corner parsing reveals left-corner parse steps correlating with activity in the left anterior temporal lobe 350-500 ms after word onset [ 50 ] . Moreover, continuous research emphasizes bottom-up and left-corner parsing’s superiority in fitting activation patterns across the left-hemisphere language network over top-down parsing. These findings suggest that individuals might process simple sentence structures through bottom-up and/or left-corner parsing, with evidence favoring bottom-up parsing [ 51 ] . In a comparative study by Zhang et al. [ 52 ] , different languages’ parsing strategies were explored, assessing working memory needs for varied language structures. The research showed that top-down parsing demands lower memory for right-branching English, while bottom-up parsing is less demanding for Chinese. Additionally, fMRI results indicated language-specific parsing preferences: Chinese favors bottom-up parsing, while English leans towards top-down parsing. For a comprehensive exploration of employing grammars and parsing strategies to investigate the construction of brain structures, consult the study by Brennan et al. (2016) [ 53 ] .

 
 
 Comparing CCG, CFG, and LLM-based predictability, evidence suggests CCG captures neural activity beyond LLM and CFG parsing steps. Augmenting CCG with the reveal operation improves fits, particularly in right adjunction parsing. Strongest effects are observed in posterior temporal lobe, around the middle temporal gyrus [ 54 ] .

 
 
 In terms of the syntactic structure of language processing, the structural language model derives surprisal estimates by analyzing the syntactic arrangement of a given text fragment. These estimates are then aligned with brain activation patterns to delve into the mechanisms underlying syntactic processing. For example, in comparing surprisals from sequential structure grammars and hierarchical phrase structure grammars, Frank and Bob (2011) found that the hierarchical sentence structure has minimal impact on anticipations for upcoming words [ 55 ] . Conversely, Brennan et al. (2016) and Brennan and Hale (2019) found that predictions based on hierarchical structure, compared to sequential information, correlated more strongly with human brain responses in passive listeners engaged in an audiobook story, as evidenced by electroencephalography signals [ 38 , 56 ] . The divergent conclusion suggests that studies based on models are sensitive to various factors, highlighting the need for standardization and uniformity. Further studies show that grammatical models such as CCG and RNNG, known for their expressiveness, provide a better match to neural signals than those derived from CFG. They notably capture activity in the left posterior temporal regions [ 57 , 58 , 54 ] . These findings advocate for CCG and RNNG as mechanistic models underpinning syntactic processing during standard human language comprehension.

 
 
 Some studies explore full neural network architectures to understand human language processing. RNNs, CNNs, and Transformers are prominent in these discussions, with RNNs particularly used for their belief in the importance of recurrent processing in human language understanding.
Merkx and Frank (2021) compared Transformer-based and RNN-based language models in measuring human reading effort. Results indicate Transformers excel over RNNs in explaining self-paced reading times and neural activity during English sentence reading. This challenges the prevalent notion of immediate and recurrent processing, indicating a cue-based retrieval process in human sentence comprehension [ 59 ] .
Seth et al. (2023) explore the relationship between CNN models and the brain, investigating if these models can explain object recognition solely based on visual properties without semantics. The study challenges the idea that semantic effects in the ventral visual pathway (VVP) during object recognition might be explained by higher-level visual object properties captured by CNN models [ 60 ] .

 
 
 
 

### 3.2 Modelling linguistic Representations

 
 Recent research has leveraged text embeddings to explore language processing in the human brain, correlating brain activation from linguistic stimuli with corresponding stimulus embeddings [ 61 , 62 , 63 , 64 , 65 , 66 ] . Techniques such as Representational Similarity Analysis and regression models have been pivotal in this analysis [ 67 , 62 , 68 , 69 ] . For example, Broderick et al. (2018) employed Word2Vec to measure semantic differences in narrative contexts, correlating these with EEG data to show EEG’s reflection of semantic processing and its alignment with N400 characteristics [ 70 ] . Similarly, Xu et al. (2016) connected shallow embeddings (GloVe, Word2Vec, RNN) with fMRI data from word observation tasks and found Skip-gram of Word2Vec and Glove performed quite well [ 71 ] . Fu et al. (2023) compared word embeddings, co-occurrence, and graph-topological methods, highlighting their differing efficacies in semantic brain pattern mapping [ 72 ] . Further, researchers have refined embeddings to better represent specific features, utilizing algebraic and neural network-based modifications [ 73 , 74 , 75 ] . Studies like Zhang et al. (2020) and Toneva et al. (2022) used modified embeddings to explore brain-based word meaning relations, revealing complex patterns of semantic categories and relations [ 76 , 9 , 77 ] . Researchers have also extended embeddings to sentence-level representations, with average pooling being a popular method [ 65 , 78 ] .

 
 
 Given the advanced prediction capabilities of large language models over shallow embeddings [ 27 , 79 ] , research now focuses on determining the most effective models for brain language representation [ 80 , 81 , 82 , 83 ] . Studies by Sun et al. (2021) compared multiple models, including large language models, confirming their superiority in predicting brain language network activities [ 13 ] . Antonello et al. (2021) introduced a novel ”language representation embedding space” for predicting brain responses during language tasks [ 84 ] . Besides semantics, large language models capture syntactic and morphological aspects, with ongoing studies aiming to distinguish these features in brain processing [ 84 , 10 , 85 ] . Caucheteux et al. (2021) [ 86 ] developed an embedding taxonomy for large language models to analyze brain activities in narrative listening. They found compositional representations engaged a broader cortical network, including the bilateral temporal, parietal, and prefrontal cortices, more extensively than lexical representations.
Zhang et al. (2022) investigated the brain’s syntactic processing using modified embeddings, revealing its distributed nature across brain networks [ 12 ] .

 
 
 Although large language model embeddings excel in aligning with brain activities, the underlying reasons and mechanisms of this synchrony are not fully understood [ 87 , 88 ] . Research aims to clarify how these models mirror human linguistic comprehension and brain structures [ 89 ] .
Sun et al. (2020) found that impairing semantic processing in large language models diminishes their brain activity alignment [ 13 ] . Conversely, Merlin and Toneva (2022) enhanced this alignment by focusing on next-word prediction and word-level semantics [ 82 ] . Aw and Toneva (2023) showed that training models on narrative summarization enhances brain activity synchronization [ 90 ] . Caucheteux et al. (2023) found that integrating multi-timescale predictions into large language models improves brain mapping. Hierarchically, frontoparietal cortices predict higher-level, longer-range representations compared to temporal cortices, emphasizing the role of hierarchical predictive coding in language processing [ 91 ] . Further, task-specific tuning of large language models in NLP tasks has been shown to influence their brain pattern alignment [ 92 , 93 , 94 ] , indicating that their semantic feature capture is a key factor in this alignment.

 
 
 

### 3.3 Verifying hypotheses for empirical validations

 
 Modern computational models, especially large language models, mimic human behavior and surpass human performance. They serve as valuable tools to verify and even generate hypotheses about the brain’s language processing. Analyzing instances where task-optimized networks mirror human behavioral and neural patterns allows for the formulation of novel hypotheses about language processing in the brain [ 17 ] .

 
 
 Limited research has delved into this realm, with studies such as those by Schrimpf et al. (2021), Goldstein et al. (2022), and Caucheteux et al. (2022) found that Transformer-based large language models, optimized for next-word prediction, align well with both behavioral and neural data in humans. The stronger the model’s performance in next-word prediction, the closer its match to human data, implying that prediction may be a key optimization aspect of the human language system [ 95 , 96 , 97 ] . Additionally, Zou et al. (2023) connected the human brain’s attention in reading to the attention weights in Transformer models, suggesting a parallel mechanism between these models and the brain. This alignment implies that the brain’s reading attention, like the Transformer models, is optimized for specific tasks [ 98 ] . These studies offer valuable insights into predictive coding and attention allocation hypotheses about language, yet only scratch the surface, confirming existing theories.

 
 
 Within the NLP domain, large language models are used to reveal the operational mechanisms of specific modules or neurons. Notably, Singh et al. (2023) demonstrated that these models can generate explanations for the response of individual fMRI voxels to language stimuli [ 99 ] . Future research can explore using large language models to generate new theories or hypotheses, uncovering insights into their output behavior. This could motivate further studies on human brain language processing.

 
 
 
 

## 4 Comparing models on a multimodal cognitive dataset

 
 Various models possess unique strengths. Among the three types of models, SLMs (N-gram models) offer simplicity and interpretability, while structural models leverage human-annotated trees for clear pattern-based predictions, requiring fewer computational resources than neural models. Despite advantages, statistical models have limited long-range dependency capture and sparse data issues for rare n-grams due to fixed context windows and word symbols. SEMs excel in semantic representation, addressing challenges like unreliable estimates and misspellings [ 100 , 101 , 102 ] . However, their static word representations lack the depth for intricate, context-dependent language processing. LLMs provide rich semantic insights with dynamic word representations [ 103 ] . Their depth captures intricate linguistic features and higher-order dependencies, facilitating swift in-context learning [ 104 , 105 , 106 , 107 ] . Yet, LLMs’ complexity and black-box nature pose challenges in interpretation [ 108 , 96 ] .

 
 
 Recent findings reveal that large language models demonstrate superior alignment with the brain compared to shallow embedding models, as evidenced by Schrimpf et al. [ 95 ] . Nevertheless, challenges in conducting fair model comparisons persist due to variations in brain datasets and metrics.

 
 
 To address this issue, we investigate how these models correlate with human cognitive data under fair comparison conditions using the same training data. We use English 1 1 
 1 
 
 
 
 https://dumps.wikimedia.org/enwiki/latest and Chinese 2 2 
 2 
 
 
 
 http://www.xinhuanet.com/whxw.htm datasets for training statistical (N-gram and structural models 3 3 
 3 
 
 
 
 https://nlp.stanford.edu/software/stanford-dependencies ), shallow (GloVe 4 4 
 4 
 
 
 
 https://nlp.stanford.edu/projects/glove/ , Word2Vec 5 5 
 5 
 
 
 
 https://radimrehurek.com/gensim/models/word2vec.html with detailed parameters in [ 109 ] ), and large (BERT-large 6 6 
 6 
 
 
 
 https://huggingface.co/bert-large-uncased , GPT2 7 7 
 7 
 
 
 
 https://huggingface.co/gpt2-medium , 24 layers, 1e-4 learning rate) language models to predict neuroimaging and eyetracking data. Word embeddings are extracted from layers yielding optimal performance 8 8 
 8 
 
 
 
 For word-level fMRI: BERT (en): 10th layer, GPT-2 (en): 1st layer, BERT (zh): 8th layer, GPT-2 (zh): 1st layer. Discourse-level fMRI: BERT (en): 13th layer, GPT-2 (en): 21st layer, BERT (zh): 23rd layer, GPT-2 (zh): 15th layer. Eye-tracking: BERT (en): 8th layer, GPT-2 (en): 4th layer, GPT-2 (zh): 3rd layer, BERT (zh): 19th layer. ). Models are trained from scratch 9 9 
 9 
 
 
 
 BERT and GPT2 were trained for one epoch, and consistent loss decrease was observed . Testing on multi-modal cognitive datasets [ 110 ] , encoding models predict fMRI or eye-tracking responses using 10-fold cross-validation. Paired t-tests assess group-level significance with a threshold of p=0.001.

 
 
 
 
 
 | 
 | 
 English | 
 Chinese | 

 
 | 
 | 
 
 
 
 fMRI | 

 
 -word | 

 
 [ 78 ] | 

 | 
 
 
 
 fMRI | 

 
 -discourse | 

 
 [ 77 ] | 

 | 
 
 
 
 EyeTracking | 

 
 -sentence | 

 
 [ 111 ] | 

 | 
 
 
 
 fMRI | 

 
 -word | 

 
 [ 112 ] | 

 | 
 
 
 
 fMRI | 

 
 -discourse | 

 
 [ 113 ] | 

 | 
 
 
 
 EyeTracking | 

 
 -sentence | 

 
 [ 114 ] | 

 | 

 
 
 
 Surprisal | 
 SLMs(Ngram) | 
 \ | 
 0.014 | 
 0.178 | 
 \ | 
 0.042 | 
 0.047 | 

 
 LLMs(GPT2) | 
 \ | 
 0.010 | 
 0.112 | 
 \ | 
 0.023 | 
 0.050 | 

 
 
 
 
 Entropy 
 
 Reduction 
 | 
 SLMs(Ngram) | 
 \ | 
 0.005 | 
 0.092 | 
 \ | 
 0.010 | 
 0.019 | 

 
 LLMs(GPT2) | 
 \ | 
 0.005 | 
 0.042 | 
 \ | 
 0.006 | 
 0.017 | 

 
 Embedding | 
 SEMs(GloVe) | 
 0.453 | 
 0.015 | 
 0.456 | 
 0.171 | 
 0.030 | 
 0.422 | 

 
 SEMs(W2V) | 
 0.491 | 
 0.012 | 
 0.456 | 
 0.181 | 
 0.037 | 
 0.424 | 

 
 LLMs(BERT) | 
 0.643 | 
 0.015 | 
 0.546 | 
 0.142 | 
 0.025 | 
 0.470 | 

 
 LLMs(GPT2) | 
 0.639 | 
 0.016 | 
 0.561 | 
 0.154 | 
 0.028 | 
 0.464 | 

 

 Table 1: Encoding results for three computational models on a multi-modal cognitive dataset. ’/’ indicates calculations are not possible due to context-based metrics, ’-’ signifies non-significant values, while all other numbers in the table demonstrate significant correlations with brain data. 
 
 
 In Table 1, it is evident that, overall, large language models consistently outperform shallow embedding models in cognitive measures of embeddings, particularly in EyeTracking and fMRI-word(English). In the case of fMRI-discourse, only minor differences are observed between the performance of large and shallow language models. Concerning surprisal and entropy reduction, all models demonstrate similar levels of effectiveness. Consequently, no single model excels uniformly across all datasets; instead, each model exhibits superiority in certain aspects.

 
 
 It is worth noting that existing methods often employ fMRI-discourse datasets but utilize models trained on diverse training datasets. When utilizing the same training datasets, conclusions may vary, implying that correlation results are more contingent on the training process than on the model’s inherent structure. Looking ahead, future endeavors should prioritize standardizing the process of employing computational models for studying the brain. This involves releasing larger benchmark testing datasets, providing models trained on the same training dataset, employing consistent metrics for model evaluation, and sharing the code for brain encoding.

 
 
 

## 5 Conclusion

 
 Statistical language models are both simple and interpretable, explicitly encoding word co-occurrence and structure, achieving comparable performance on fMRI-discourse datasets. Shallow embedding models, which excel in learning static semantic representations, efficiently quantify meaning, presenting a novel opportunity to investigate meaning in the brain. They outperform statistical models significantly on eye-tracking datasets. Large language models, exhibiting human-like behavior, open up new possibilities for exploring hypotheses about how the brain processes language. Despite their superior performance on downstream tasks, they only outperform the other two models in the embeddings metric for fMRI-word(English) and across all cognitive metrics on the EyeTracking dataset.

 
 
 This review suggests that computational models can significantly contribute to understanding the brain when used appropriately. They are particularly useful for studying cognitive load, semantic-syntactic representation, parsing strategy, and the syntactic structure of language processing in the brain. Furthermore, these models have the potential to verify existing hypotheses and generate new ones, guiding future brain studies. Despite the recent surge in attention and strong performance of large language models on downstream tasks, understanding why they correlate with the brain requires meticulous examination. It involves disentangling various factors, including training datasets, model structures, and training objectives. Conclusions drawn from such correlations are sensitive to the models used, underscoring the importance of careful consideration when extracting insights from different models. Future research should incorporate diverse datasets and various models to rigorously test the same question.

 
 
 

## References

 
 
 [1] 
 
R. Geirhos, K. Meding, and F. A. Wichmann, “Beyond accuracy: quantifying trial-by-trial behaviour of cnns and humans by measuring error consistency,” Advances in Neural Information Processing Systems , vol. 33, pp. 13890–13902, 2020.

 

 
 [2] 
 
O. Guest and A. E. Martin, “On logical inference over brains, behaviour, and artificial neural networks,” Computational Brain Behavior , pp. 1–15, 2023.

 

 
 [3] 
 
J. S. Bowers, G. Malhotra, M. Dujmović, M. L. Montero, C. Tsvetkov, V. Biscione, G. Puebla, F. Adolfi, J. E. Hummel, R. F. Heaton, et al. , “Deep problems with neural network models of human vision,” Behavioral and Brain Sciences , pp. 1–74, 2022.

 

 
 [4] 
 
A. M. Zador, “A critique of pure learning and what artificial neural networks can learn from animal brains,” Nature communications , vol. 10, no. 1, p. 3770, 2019.

 

 
 [5] 
 
F. Huettig and N. Mani, “Is prediction necessary to understand language? probably not,” Language, Cognition and Neuroscience , vol. 31, no. 1, pp. 19–31, 2016.

 

 
 [6] 
 
L. W. Barsalou, “What does semantic tiling of the cortex tell us about semantics?,” Neuropsychologia , vol. 105, pp. 18–38, 2017.

 

 
 [7] 
 
J. Hale, D. Lutz, W.-M. Luh, and J. Brennan, “Modeling fmri time courses with linguistic structure at various grain sizes,” in Proceedings of the 6th workshop on cognitive modeling and computational linguistics , pp. 89–97, 2015.

 

 
 [8] 
 
L. Wehbe, A. Vaswani, K. Knight, and T. Mitchell, “Aligning context-based statistical models of language with brain activity during reading,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pp. 233–243, 2014.

 

 
 [9] 
 
M. Toneva, T. M. Mitchell, and L. Wehbe, “Combining computational controls with natural text reveals aspects of meaning composition,” Nature computational science , vol. 2, no. 11, pp. 745–757, 2022.

 

 
 [10] 
 
A. J. Reddy and L. Wehbe, “Can fmri reveal the representation of syntactic structure in the brain?,” Advances in Neural Information Processing Systems , vol. 34, pp. 9843–9856, 2021.

 

 
 [11] 
 
M. Schrimpf, J. Kubilius, M. J. Lee, N. A. R. Murty, R. Ajemian, and J. J. DiCarlo, “Integrative benchmarking to advance neurally mechanistic models of human intelligence,” Neuron , vol. 108, no. 3, pp. 413–423, 2020.

 

 
 [12] 
 
X. Zhang, S. Wang, N. Lin, J. Zhang, and C. Zong, “Probing word syntactic representations in the brain by a feature elimination method,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 36, pp. 11721–11729, 2022.

 

 
 [13] 
 
J. Sun, S. Wang, J. Zhang, and C. Zong, “Neural encoding and decoding with distributed sentence representations,” IEEE Transactions on Neural Networks and Learning Systems , vol. 32, no. 2, pp. 589–603, 2020.

 

 
 [14] 
 
N. Kriegeskorte and P. K. Douglas, “Cognitive computational neuroscience,” Nature neuroscience , vol. 21, no. 9, pp. 1148–1160, 2018.

 

 
 [15] 
 
R. M. Cichy and D. Kaiser, “Deep neural networks as scientific models,” Trends in cognitive sciences , vol. 23, no. 4, pp. 305–317, 2019.

 

 
 [16] 
 
M. Baroni, “On the proper role of linguistically-oriented deep net analysis in linguistic theorizing,” Algebraic structures in natural language , pp. 1–16, 2022.

 

 
 [17] 
 
N. Kanwisher, M. Khosla, and K. Dobs, “Using artificial neural networks to ask ‘why’questions of minds and brains,” Trends in Neurosciences , vol. 46, no. 3, pp. 240–254, 2023.

 

 
 [18] 
 
J. T. Hale, L. Campanelli, J. Li, S. Bhattasali, C. Pallier, and J. R. Brennan, “Neurocomputational models of language processing,” Annual Review of Linguistics , vol. 8, pp. 427–446, 2022.

 

 
 [19] 
 
S. Arana, J. Pesnot Lerousseau, and P. Hagoort, “Deep learning models to study sentence comprehension in the human brain,” Language, Cognition and Neuroscience , pp. 1–19, 2023.

 

 
 [20] 
 
J. Hale, “A probabilistic earley parser as a psycholinguistic model,” in Second meeting of the north american chapter of the association for computational linguistics , 2001.

 

 
 [21] 
 
J. Pennington, R. Socher, and C. D. Manning, “Glove: Global vectors for word representation,” in Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) , pp. 1532–1543, 2014.

 

 
 [22] 
 
Q. Le and T. Mikolov, “Distributed representations of sentences and documents,” in International conference on machine learning , pp. 1188–1196, PMLR, 2014.

 

 
 [23] 
 
J. Sun, S. Wang, and C. Zong, “Memory, show the way: Memory based few shot word representation learning,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, eds.), (Brussels, Belgium), pp. 1435–1444, Association for Computational Linguistics, Oct.-Nov. 2018.

 

 
 [24] 
 
T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word representations in vector space,” arXiv preprint arXiv:1301.3781 , 2013.

 

 
 [25] 
 
T. Mikolov, S. Kombrink, L. Burget, J. Černockỳ, and S. Khudanpur, “Extensions of recurrent neural network language model,” in 2011 IEEE international conference on acoustics, speech and signal processing (ICASSP) , pp. 5528–5531, IEEE, 2011.

 

 
 [26] 
 
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems , vol. 30, 2017.

 

 
 [27] 
 
J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” arXiv preprint arXiv:1810.04805 , 2018.

 

 
 [28] 
 
A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. , “Language models are unsupervised multitask learners,” OpenAI blog , vol. 1, no. 8, p. 9, 2019.

 

 
 [29] 
 
Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert pretraining approach,” arXiv preprint arXiv:1907.11692 , 2019.

 

 
 [30] 
 
H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. , “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971 , 2023.

 

 
 [31] 
 
E. Kasneci, K. Seßler, S. Küchemann, M. Bannert, D. Dementieva, F. Fischer, U. Gasser, G. Groh, S. Günnemann, E. Hüllermeier, et al. , “Chatgpt for good? on opportunities and challenges of large language models for education,” Learning and individual differences , vol. 103, p. 102274, 2023.

 

 
 [32] 
 
T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. , “Language models are few-shot learners,” Advances in neural information processing systems , vol. 33, pp. 1877–1901, 2020.

 

 
 [33] 
 
M. F. Boston, J. Hale, R. Kliegl, U. Patil, and S. Vasishth, “Parsing costs as predictors of reading difficulty: An evaluation using the potsdam sentence corpus,” Journal of Eye Movement Research , vol. 2, no. 1, 2008.

 

 
 [34] 
 
V. Demberg and F. Keller, “Data from eye-tracking corpora as evidence for theories of syntactic processing complexity,” Cognition , vol. 109, no. 2, pp. 193–210, 2008.

 

 
 [35] 
 
S. Rauzy and P. Blache, “Robustness and processing difficulty models. a pilot study for eye-tracking data on the french treebank,” in Proceedings of the First Workshop on Eye-tracking and Natural Language Processing , pp. 21–36, 2012.

 

 
 [36] 
 
S. L. Frank, L. J. Otten, G. Galli, and G. Vigliocco, “Word surprisal predicts n400 amplitude during reading,” 2013.

 

 
 [37] 
 
S. L. Frank, L. J. Otten, G. Galli, and G. Vigliocco, “The erp response to the amount of information conveyed by words in sentences,” Brain and language , vol. 140, pp. 1–11, 2015.

 

 
 [38] 
 
J. R. Brennan, E. P. Stabler, S. E. Van Wagenen, W.-M. Luh, and J. T. Hale, “Abstract linguistic structure correlates with temporal activity during naturalistic comprehension,” Brain and language , vol. 157, pp. 81–94, 2016.

 

 
 [39] 
 
J. M. Henderson, W. Choi, M. W. Lowder, and F. Ferreira, “Language structure in the brain: A fixation-related fmri study of syntactic surprisal in reading,” Neuroimage , vol. 132, pp. 293–300, 2016.

 

 
 [40] 
 
C. Shain, I. A. Blank, M. van Schijndel, W. Schuler, and E. Fedorenko, “fmri reveals language-specific predictive coding during naturalistic sentence comprehension,” Neuropsychologia , vol. 138, p. 107307, 2020.

 

 
 [41] 
 
M. Heilbron, K. Armeni, J.-M. Schoffelen, P. Hagoort, and F. P. De Lange, “A hierarchy of linguistic predictions during natural language comprehension,” Proceedings of the National Academy of Sciences , vol. 119, no. 32, p. e2201968119, 2022.

 

 
 [42] 
 
J. Hale, “The information conveyed by words in sentences,” Journal of psycholinguistic research , vol. 32, pp. 101–123, 2003.

 

 
 [43] 
 
J. Hale, “Uncertainty about the rest of the sentence,” Cognitive science , vol. 30, no. 4, pp. 643–672, 2006.

 

 
 [44] 
 
S. L. Frank, “Uncertainty reduction as a measure of cognitive load in sentence comprehension,” Topics in cognitive science , vol. 5, no. 3, pp. 475–494, 2013.

 

 
 [45] 
 
S. Wu, A. Bachrach, C. Cardenas, and W. Schuler, “Complexity metrics in an incremental right-corner parser,” in Proceedings of the 48th annual meeting of the association for computational linguistics , pp. 1189–1198, 2010.

 

 
 [46] 
 
T. Linzen and T. F. Jaeger, “Uncertainty and expectation in sentence processing: Evidence from subcategorization distributions,” Cognitive science , vol. 40, no. 6, pp. 1382–1411, 2016.

 

 
 [47] 
 
M. Nelson, S. Dehaene, C. Pallier, and J. Hale, “Entropy reduction correlates with temporal lobe activity,” in Proceedings of the 7th workshop on cognitive modeling and computational linguistics (CMCL 2017) , pp. 1–10, 2017.

 

 
 [48] 
 
J. Hale, “Information-theoretical complexity metrics,” Language and Linguistics Compass , vol. 10, no. 9, pp. 397–412, 2016.

 

 
 [49] 
 
J. Brennan, Y. Nir, U. Hasson, R. Malach, D. J. Heeger, and L. Pylkkänen, “Syntactic structure building in the anterior temporal lobe during natural story listening,” Brain and language , vol. 120, no. 2, pp. 163–173, 2012.

 

 
 [50] 
 
J. R. Brennan and L. Pylkkänen, “Meg evidence for incremental sentence composition in the anterior temporal lobe,” Cognitive science , vol. 41, pp. 1515–1531, 2017.

 

 
 [51] 
 
M. J. Nelson, I. El Karoui, K. Giber, X. Yang, L. Cohen, H. Koopman, S. S. Cash, L. Naccache, J. T. Hale, C. Pallier, et al. , “Neurophysiological dynamics of phrase-structure building during sentence processing,” Proceedings of the National Academy of Sciences , vol. 114, no. 18, pp. E3669–E3678, 2017.

 

 
 [52] 
 
X. Zhang, S. Wang, N. Lin, and C. Zong, “Is the brain mechanism for hierarchical structure building universal across languages? an fmri study of chinese and english,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , pp. 7852–7861, 2022.

 

 
 [53] 
 
J. Brennan, “Naturalistic sentence comprehension in the brain,” Language and Linguistics Compass , vol. 10, no. 7, pp. 299–313, 2016.

 

 
 [54] 
 
M. Stanojević, J. R. Brennan, D. Dunagan, M. Steedman, and J. T. Hale, “Modeling structure-building in the brain with ccg parsing and large language models,” Cognitive Science , vol. 47, no. 7, p. e13312, 2023.

 

 
 [55] 
 
S. L. Frank and R. Bod, “Insensitivity of the human sentence-processing system to hierarchical structure,” Psychological science , vol. 22, no. 6, pp. 829–834, 2011.

 

 
 [56] 
 
J. R. Brennan and J. T. Hale, “Hierarchical structure guides rapid linguistic predictions during naturalistic listening,” PloS one , vol. 14, no. 1, p. e0207741, 2019.

 

 
 [57] 
 
J. Hale, C. Dyer, A. Kuncoro, and J. R. Brennan, “Finding syntax in human encephalography with beam search,” arXiv preprint arXiv:1806.04127 , 2018.

 

 
 [58] 
 
J. R. Brennan, C. Dyer, A. Kuncoro, and J. T. Hale, “Localizing syntactic predictions using recurrent neural network grammars,” Neuropsychologia , vol. 146, p. 107479, 2020.

 

 
 [59] 
 
D. Merkx and S. L. Frank, “Human sentence processing: Recurrence or attention?,” in Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics , (Online), pp. 12–22, Association for Computational Linguistics, June 2021.

 

 
 [60] 
 
J. von Seth, V. I. Nicholls, L. K. Tyler, and A. Clarke, “Recurrent connectivity supports higher-level visual and semantic object representations in the brain,” Mar 2023.

 

 
 [61] 
 
L. Wehbe, A. Vaswani, K. Knight, and T. M. Mitchell, “Aligning context-based statistical models of language with brain activity during reading,” in Conference on Empirical Methods in Natural Language Processing , 2014.

 

 
 [62] 
 
A. J. Anderson, B. D. Zinszer, and R. D. Raizada, “Representational similarity encoding for fmri: Pattern-based synthesis to predict brain activity using stimulus-model-similarities,” NeuroImage , vol. 128, pp. 44–53, 2016.

 

 
 [63] 
 
W. A. de Heer, A. G. Huth, T. L. Griffiths, J. L. Gallant, and F. E. Theunissen, “The hierarchical cortical organization of human speech processing,” The Journal of Neuroscience , vol. 37, pp. 6539 – 6557, 2017.

 

 
 [64] 
 
S. Jain and A. G. Huth, “Incorporating context into language encoding models for fmri,” bioRxiv , 2018.

 

 
 [65] 
 
J. Sun, S. Wang, J. Zhang, and C. Zong, “Towards sentence-level brain decoding with distributed representations,” in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 33, pp. 7047–7054, 2019.

 

 
 [66] 
 
S. Wang, J. Zhang, H. Wang, N. Lin, and C. Zong, “Fine-grained neural decoding with distributed word representations,” Information Sciences , vol. 507, pp. 256–272, 2020.

 

 
 [67] 
 
L. Wehbe, B. Murphy, P. P. Talukdar, A. Fyshe, A. Ramdas, and T. M. Mitchell, “Simultaneously uncovering the patterns of brain regions involved in different story reading subprocesses,” PLoS ONE , vol. 9, 2014.

 

 
 [68] 
 
A. G. Huth, W. A. de Heer, T. L. Griffiths, F. E. Theunissen, and J. L. Gallant, “Natural speech reveals the semantic maps that tile human cerebral cortex,” Nature , vol. 532, pp. 453 – 458, 2016.

 

 
 [69] 
 
A. J. Anderson, E. C. Lalor, F. Lin, J. R. Binder, L. Fernandino, C. J. Humphries, L. L. Conant, R. D. S. Raizada, S. Grimm, and X. Wang, “Multiple regions of a cortical network commonly encode the meaning of words in multiple grammatical positions of read sentences.,” Cerebral cortex , vol. 29 6, pp. 2396–2411, 2019.

 

 
 [70] 
 
M. P. Broderick, A. J. Anderson, G. M. Di Liberto, M. J. Crosse, and E. C. Lalor, “Electrophysiological correlates of semantic dissimilarity reflect the comprehension of natural, narrative speech,” Current Biology , vol. 28, no. 5, pp. 803–809, 2018.

 

 
 [71] 
 
H. Xu, B. Murphy, and A. Fyshe, “Brainbench: A brain-image test suite for distributional semantic models,” in Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing , pp. 2017–2021, 2016.

 

 
 [72] 
 
Z. Fu, X. Wang, X. Wang, H. Yang, J. Wang, T. Wei, X. Liao, Z. Liu, H. Chen, and Y. Bi, “Different computational relations in language are captured by distinct brain systems,” Cerebral Cortex , vol. 33, no. 4, pp. 997–1013, 2023.

 

 
 [73] 
 
A. J. Anderson, E. Bruni, U. Bordignon, M. Poesio, and M. Baroni, “Of words, eyes and brains: Correlating image-based distributional semantic models with neural representations of concepts,” in Conference on Empirical Methods in Natural Language Processing , 2013.

 

 
 [74] 
 
A. J. Anderson, D. Kiela, S. Clark, and M. Poesio, “Visually grounded and textual semantic models differentially decode brain activity associated with concrete and abstract nouns,” Transactions of the Association for Computational Linguistics , vol. 5, pp. 17–30, 2017.

 

 
 [75] 
 
B. Lyu, H. S. Choi, W. D. Marslen-Wilson, A. Clarke, B. Randall, and L. K. Tyler, “Neural dynamics of semantic composition,” Proceedings of the National Academy of Sciences of the United States of America , vol. 116, pp. 21318 – 21327, 2019.

 

 
 [76] 
 
F. Mollica, M. Siegelman, E. Diachek, S. T. Piantadosi, Z. Mineroff, R. Futrell, H. Kean, P. Qian, and E. Fedorenko, “Composition is the core driver of the language-selective network,” Neurobiology of Language , vol. 1, no. 1, pp. 104–134, 2020.

 

 
 [77] 
 
Y. Zhang, K. Han, R. Worth, and Z. Liu, “Connecting concepts in the brain by mapping cortical representations of semantic relations,” Nature communications , vol. 11, no. 1, p. 1877, 2020.

 

 
 [78] 
 
F. Pereira, B. Lou, B. Pritchett, S. Ritter, S. J. Gershman, N. Kanwisher, M. Botvinick, and E. Fedorenko, “Toward a universal decoder of linguistic meaning from brain activation,” Nature communications , vol. 9, no. 1, p. 963, 2018.

 

 
 [79] 
 
J. Sun, S. Wang, J. Zhang, and C. Zong, “Distill and replay for continual language learning,” in Proceedings of the 28th international conference on computational linguistics , pp. 3569–3579, 2020.

 

 
 [80] 
 
S. R. Oota, J. Arora, M. Gupta, R. S. Bapi, and M. Toneva, “Deep learning for brain encoding and decoding,” in Proceedings of the Annual Meeting of the Cognitive Science Society , vol. 44, 2022.

 

 
 [81] 
 
C. Caucheteux, A. Gramfort, and J.-R. King, “Deep language algorithms predict semantic comprehension from brain activity,” Scientific reports , vol. 12, no. 1, p. 16327, 2022.

 

 
 [82] 
 
G. Merlin and M. Toneva, “Language models and brain alignment: beyond word-level semantics and prediction,” arXiv preprint arXiv:2212.00596 , 2022.

 

 
 [83] 
 
H. Balabin, A. G. Liuzzi, J. Sun, P. Dupont, R. Vanderberghe, and M.-F. Moens, “Investigating neural fit approaches for sentence embedding model paradigms,” in ECAI 2023 , pp. 165–173, IOS Press, 2023.

 

 
 [84] 
 
R. Antonello, J. S. Turek, V. Vo, and A. Huth, “Low-dimensional structure in the space of language representations is reflected in brain responses,” Advances in neural information processing systems , vol. 34, pp. 8332–8344, 2021.

 

 
 [85] 
 
S. R. Oota, F. Alexandre, and X. Hinaut, “Long-term plausibility of language models and neural dynamics during narrative listening,” in Proceedings of the Annual Meeting of the Cognitive Science Society , vol. 44, 2022.

 

 
 [86] 
 
C. Caucheteux, A. Gramfort, and J.-R. King, “Disentangling syntax and semantics in the brain with deep networks,” in International conference on machine learning , pp. 1336–1348, PMLR, 2021.

 

 
 [87] 
 
A. Lindborg and M. Rabovsky, “Meaning in brains and machines: Internal activation update in large-scale language model partially reflects the n400 brain potential,” in Proceedings of the annual meeting of the cognitive science society , vol. 43, 2021.

 

 
 [88] 
 
A. Pasquiou, Y. Lakretz, J. Hale, B. Thirion, and C. Pallier, “Neural language models are not born equal to fit brain data, but training helps,” arXiv preprint arXiv:2207.03380 , 2022.

 

 
 [89] 
 
S. R. Oota, M. Gupta, and M. Toneva, “Joint processing of linguistic properties in brains and language models,” 2022.

 

 
 [90] 
 
K. L. Aw and M. Toneva, “Training language models to summarize narratives improves brain alignment,” in The Eleventh International Conference on Learning Representations , 2023.

 

 
 [91] 
 
C. Caucheteux, A. Gramfort, and J.-R. King, “Evidence of a predictive coding hierarchy in the human brain listening to speech,” Nature human behaviour , vol. 7, no. 3, pp. 430–441, 2023.

 

 
 [92] 
 
S. Jingyuan and M. Marie-Francine, “Fine-tuned vs. prompt-tuned supervised representations: Which better account for brain language representations?,” in Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23 , International Joint Conferences on Artificial Intelligence Organization, 8 2023.

 
 Main Track.

 

 
 [93] 
 
S. Jingyuan, Z. Xiaohan, and M. Marie-Francine, “Tuning in to neural encoding: Linking human brain and artificial supervised representations of language,” in Proceedings of the 26th European Conference on Artificial Intelligence ECAI-23 , 9 2023.

 
 Main Track.

 

 
 [94] 
 
S. R. Oota, J. Arora, V. Agarwal, M. Marreddy, M. Gupta, and B. Surampudi, “Neural language taskonomy: Which NLP tasks are the most predictive of fMRI brain activity?,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , (Seattle, United States), pp. 3220–3237, Association for Computational Linguistics, July 2022.

 

 
 [95] 
 
M. Schrimpf, I. A. Blank, G. Tuckute, C. Kauf, E. A. Hosseini, N. Kanwisher, J. B. Tenenbaum, and E. Fedorenko, “The neural architecture of language: Integrative modeling converges on predictive processing,” Proceedings of the National Academy of Sciences , vol. 118, no. 45, p. e2105646118, 2021.

 

 
 [96] 
 
A. Goldstein, Z. Zada, E. Buchnik, M. Schain, A. Price, B. Aubrey, S. A. Nastase, A. Feder, D. Emanuel, A. Cohen, et al. , “Shared computational principles for language processing in humans and deep language models,” Nature neuroscience , vol. 25, no. 3, pp. 369–380, 2022.

 

 
 [97] 
 
C. Caucheteux and J.-R. King, “Brains and algorithms partially converge in natural language processing,” Communications biology , vol. 5, no. 1, p. 134, 2022.

 

 
 [98] 
 
J. Zou, Y. Zhang, J. Li, X. Tian, and N. Ding, “Human attention during goal-directed reading comprehension relies on task optimization,” jun 2023.

 

 
 [99] 
 
C. Singh, A. R. Hsu, R. Antonello, S. Jain, A. G. Huth, B. Yu, and J. Gao, “Explaining black box text modules in natural language with language models,” arXiv preprint arXiv:2305.09863 , 2023.

 

 
 [100] 
 
O. Kwon, D. Kim, S.-R. Lee, J. Choi, and S. Lee, “Handling out-of-vocabulary problem in hangeul word embeddings,” in Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume , pp. 3213–3221, 2021.

 

 
 [101] 
 
R. Fu, J. Guo, B. Qin, W. Che, H. Wang, and T. Liu, “Learning semantic hierarchies via word embeddings,” in Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pp. 1199–1209, 2014.

 

 
 [102] 
 
B. Edizel, A. Piktus, P. Bojanowski, R. Ferreira, E. Grave, and F. Silvestri, “Misspelling oblivious word embeddings,” arXiv preprint arXiv:1905.09755 , 2019.

 

 
 [103] 
 
U. Naseem, I. Razzak, K. Musial, and M. Imran, “Transformer based deep intelligent contextual embedding for twitter sentiment analysis,” Future Generation Computer Systems , vol. 113, pp. 58–69, 2020.

 

 
 [104] 
 
S. Sengupta, S. Basak, P. Saikia, S. Paul, V. Tsalavoutis, F. Atiah, V. Ravi, and A. Peters, “A review of deep learning with special emphasis on architectures, applications and recent trends,” Knowledge-Based Systems , vol. 194, p. 105596, 2020.

 

 
 [105] 
 
B. Min, H. Ross, E. Sulem, A. P. B. Veyseh, T. H. Nguyen, O. Sainz, E. Agirre, I. Heintz, and D. Roth, “Recent advances in natural language processing via large pre-trained language models: A survey,” ACM Computing Surveys , vol. 56, no. 2, pp. 1–40, 2023.

 

 
 [106] 
 
S. Chan, A. Santoro, A. Lampinen, J. Wang, A. Singh, P. Richemond, J. McClelland, and F. Hill, “Data distributional properties drive emergent in-context learning in transformers,” Advances in Neural Information Processing Systems , vol. 35, pp. 18878–18891, 2022.

 

 
 [107] 
 
J. Von Oswald, E. Niklasson, E. Randazzo, J. Sacramento, A. Mordvintsev, A. Zhmoginov, and M. Vladymyrov, “Transformers learn in-context by gradient descent,” in International Conference on Machine Learning , pp. 35151–35174, PMLR, 2023.

 

 
 [108] 
 
M. Toneva and L. Wehbe, “Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain),” Advances in neural information processing systems , vol. 32, 2019.

 

 
 [109] 
 
C. Emmanuele, S. Enrico, H. Chu-Ren, A. Lenci, et al. , “Decoding word embeddings with brain-based semantic features,” Computational Linguistics , vol. 47, no. 3, pp. 663–698, 2021.

 

 
 [110] 
 
X. Zhang, Y. Zhang, C. Li, S. Wang, and C. Zong, “Mulcogbench: A multi-modal cognitive benchmark dataset for evaluating chinese and english computational language models,” Submitted .

 

 
 [111] 
 
N. Hollenstein, J. Rotsztejn, M. Troendle, A. Pedroni, C. Zhang, and N. Langer, “Zuco, a simultaneous eeg and eye-tracking resource for natural sentence reading,” Scientific data , vol. 5, no. 1, pp. 1–13, 2018.

 

 
 [112] 
 
S. Wang, Y. Zhang, X. Zhang, J. Sun, N. Lin, J. Zhang, and C. Zong, “An fmri dataset for concept representation with semantic feature annotations,” Scientific Data , vol. 9, no. 1, p. 721, 2022.

 

 
 [113] 
 
S. Wang, X. Zhang, J. Zhang, and C. Zong, “A synchronized multimodal neuroimaging dataset for studying brain language processing,” Scientific Data , vol. 9, no. 1, p. 590, 2022.

 

 
 [114] 
 
G. Zhang, P. Yao, G. Ma, J. Wang, J. Zhou, L. Huang, P. Xu, L. Chen, S. Chen, J. Gu, et al. , “The database of eye-movement measures on words in chinese reading,” Scientific Data , vol. 9, no. 1, p. 411, 2022.