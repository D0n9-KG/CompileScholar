# Comparing BERT against traditional machine learning text classification

Santiago González-Carvajal, Eduardo C. Garrido-Merchán

$^{1}$ Universidad Politécnica de Madrid, Madrid, Spain

$^{2}$ Universidad Autónoma de Madrid, Madrid, Spain
santiago.gonzalez-carvajal@alumnos.upm.es
eduardo.garrido@uam.es

Abstract. The BERT model has arisen as a popular state-of-the-art model in recent years. It is able to cope with NLP tasks such as supervised text classification without human supervision. Its flexibility to cope with any corpus delivering great results has make this approach very popular in academia and industry. Although, other approaches have been used before successfully. We first present BERT and a review on classical NLP approaches. Then, we empirically test with a suite of different scenarios the behaviour of BERT against traditional TF-IDF vocabulary fed to machine learning algorithms. The purpose of this work is adding empirical evidence to support the use of BERT as a default on NLP tasks. Experiments show the superiority of BERT and its independence of features of the NLP problem such as the language of the text adding empirical evidence to use BERT as a default technique in NLP problems.

# 1 Introduction

Natural Language Processing (NLP) methodologies have flourished and lots of papers solving different tasks of the field, such as text classification $[1]$ , named entity recognition $[17]$ or summarization $[19]$ , have been published. We can differentiate, mainly, between two types of approaches to NLP problems: Firstly, linguistic approaches $[6]$ that generally use different features of the text that the experts on the domain consider that are relevant have been extensively used. Those features could be combinations of words, or n-grams $[22]$ , grammatical categories, unambiguous meanings of words, words appearing in a particular position, categories of words and much more. These features could be built manually for an specific problem or can be retrieved by using different linguistic resources $[3]$ such as ontologies $[5]$ .

On the other hand, Machine Learning (ML) $[15]$ and deep learning based approaches $[18]$ that classically have analyzed annotated corpora of texts inferring which features of the text, typically in a bag of words fashion $[30]$ or by n-grams, are relevant for the classification automatically. Both approaches have their pros and cons, concretely, linguistic approaches have great precision but their recall is low as the context where the features are useful is not as big as the

one processed by machine learning algorithms. Although, the precision of classical NLP systems was, until recently, generally better as the one delivered by machine learning $[9]$ . Nevertheless, recently, thanks to the rise of computation, machine learning text classification dominates in scenarios where huge sizes of texts are processed.

Generally, linguistic approaches consist in applying a series of rules, which are designed by linguistic experts $[14]$ . An example of linguistic approach can be found at $[12]$ . The advantage of these type of approaches over ML based approaches is that they do not need large amounts of data. Regarding ML based approaches, they usually have a statistical base $[14]$ . We can find many examples of these type of approaches: BERT $[7]$ , Transformers $[27]$ , etc.

Another issue with traditional NLP approaches is multilingualism $[4]$ . We can design rules for a given language, but sentence structure, and even the alphabet, may change from one language to another, resulting in the need to design new rules. Some approaches such as the Universal Networking Language (UNL) standard $[25]$ try to circumvent this issue, but the multilingual resource is hard to build and requires experts on the platform. Another problem with UNL approaches and related ones, would be that, given a specific language, the different forms of expression, i.e. the way we write in, for example, Twitter, is very different from the way we write a more formal document, such as a research paper $[8]$ .

Bidirectional Encoder Representations from Transformers (BERT) is a NLP model that was designed to pretrain deep bidirectional representations from unlabeled text and, after that, be fine-tuned using labeled text for different NLP tasks $[7]$ . That way, with BERT model, we can create state-of-the-art models for many different NLP tasks $[7]$ . We can see the results obtained by BERT in different NLP tasks at $[7]$ .

In this work we compare BERT model $[7]$ with a traditional machine learning NLP approach that trains machine learning algorithms in features retrieved by the Term Frequency - Inverse Document Frequency (TF-IDF) $[29]$ algorithm as a representative of these traditional approaches $[24]$ . With this technique, we avoid the construction of a linguistic resource that need expert supervision, simulating it with the punctuation retrieved for any term by the TF-IDF technique. We lose precision by doing this operation but gain recall.

We have carried out four different experiments about text classification. In all of them, we have used two different classifiers: BERT and a traditional classifier created in the way that we have just explained. In this work we start by presenting some related work, then, we describe the models we have used in our experiments, after that, we describe the experiments we have carried out and show the obtained results and, finally, we present the conclusions drawn from the work and some future lines of work.

# 2 Related Work

In this section, we summarize the main comparisons against advanced models such as the BERT transformer and classical natural language processing. Recently, BERT has achieved state-of-the-art results in a broad range of NLP tasks $[7]$ , so the question that is discussed is whether classical NLP techniques are still useful in comparison to the outstanding behaviour of BERT and related models.

It is interesting to study how does the BERT model represent the steps of the traditional NLP pipeline $[23]$ in order to make a fair comparison. The main conclusion of this paper is that their work shows that the model adapts to the classical NLP pipeline dynamically, revising lower-level decisions on the basis of disambiguating information from higher-level representations. In other words, we can think of BERT as a generalization of the traditional NLP pipeline, hence being more dynamic.

An argument that defends classical machine learning NLP approaches is that the BERT approach need huge amounts of texts to deliver proper results. An interesting work $[26]$ that focus on a pure empirical comparison of BERT and ULMFiT $[21]$ w.r.t traditional NLP approaches in low-shot classification tasks where we only have 100-1000 labelled examples per class shows how BERT, representing the best of deep transfer learning, is the best performing approach, outperforming top classical machine learning algorithms thanks to the use of transfer learning $[7]$ . In our work, we are going to test this hypothesis under different problems that also involve texts in different languages.

A common critique of classical NLP practitioners is that the BERT model and machine learning methodologies can be fooled easily, committing errors that may be severe in certain applications and that can be easily solved by symbolic approaches. Following this reasoning, in this work $[13]$ the authors present the TextFooler baseline, that generates adversarial text in order to fool BERT's classification $[13]$ . We wonder if these experiments are representative of common scenarios and hypothesize that, although it is true that some texts may fool BERT, they are not representatives of common problems. In order to test this hypothesis, we are going to measure the results given by BERT in different languages. If BERT fail in these problems, then these adversaries may be common. Although, if BERT outperforms classical approaches under standard circumstances, then we can state that these adversarial attacks may not be common.

# 3 The BERT model and the traditional machine learning NLP methodology

Having reviewed related work, we will now introduce the traditional NLP approaches that we are comparing with BERT and then, the details of the BERT model.

# 3.1 Term Frequency - Inverse Document Frequency (TF-IDF)

A classical way to deal with a supervised learning NLP task is to build a bag-of-words model with the most weighted words given by the TF-IDF algorithm. Assuming there are N documents in the collection, and that term $t_{i}$ occurs in $n_{i}$ of these documents. Then, inverse document frequency can be computed as:

$$
i d f (t _ {i}) = \log \frac {N}{n _ {i}}. \tag {1}
$$

Actually, the original measure was an integer approximation to this formula, and the logarithm was base 2. However, (1) is the most commonly cited form of IDF. For more information we refer the reader to the original source [20].

On the other hand, given a term $t_i$ , we denote by $tf_i$ the frequency of the term $t_i$ in the document under consideration [20].

Finally, TF-IDF is defined for a given term $t_i$ in a given document as follows:

$$
t f i d f (t _ {i}) = t f _ {i} \cdot i d f (t _ {i}).
$$

In our experiments, regarding the standard NLP algorithms, we will be using TF-IDF to build a vocabulary for a machine learning model. Further details are introduced in the experiments section.

# 3.2 Bidirectional Encoder Representations from Transformers (BERT)

We now explain what we consider to be the state-of-the-art technique on natural language processing. Regarding the BERT model, there are two steps in its framework: pre-training and fine-tuning [7]. During pre-training, the model is trained on unlabeled large corpus. For fine-tuning, the model is initialized with the pre-trained parameters, and all the parameters are fine-tuned using labeled data for specific tasks.

BERT's model architecture is a multi-layer bidirectional Transformer encoder [7] based on the original implementation described in [27].

This kind of encoder is composed of a stack of $N = 6$ identical layers. Each of these layers has two sub-layers. The first one is a multi-head self-attention mechanism, and the second one, is a simple position-wise fully connected feedforward network. It employs a residual connection [11] around both sub-layers, followed by a layer normalization [2]. That is, the output of each sub-layer is $LayerNorm(x + Sublayer(x))$ , where $Sublayer(x)$ is the function implemented by the sub-layer [27].

In relation to, multi-head self-attention, first, we need to define scaled dot-product attention. It is defined as follows:

$$
A t t e n t i o n (Q, K, V) = s o f t m a x (\frac {Q K ^ {T}}{\sqrt {d _ {k}}}) V,
$$

where Q is the matrix of queries, K is the matrix of keys, V is the matrix of values and $d_{k}$ is the dimension of the Q and K matrices. Now, we can define multi-head attention as

$$
M u l t i H e a d (Q, K, V) = C o n c a t (h e a d _ {1}, \dots , h e a d _ {h}) W ^ {O},
$$

where $head_{i} = Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})$ . Multi-head attention consists on projecting the queries, keys and values h times with different, learned linear projections to $d_{k}$ , $d_{k}$ and $d_{v}$ (dimension of the values matrix), respectively. Then, on each of these projected versions of the queries, keys and values, we perform the attention function in parallel, yielding in $d_{v}$ -dimensional output values. Finally, these are concatenated and projected, resulting in the final values [27]. Self-attention means that all of the keys, values and queries come from the same place.

BERT represents a single sentence or a pair of sentences (for example, the pair $\langle question, answer\rangle$ ) as a sequence of tokens according to the following features: BERT uses WordPiece embeddings [28]. The first token of the sequence is “[CLS]”. When there is a pair of sentence, in the sequence, they are separated by the “[SEP]” token. And, an embedding is added to every token indicating whether it belongs to the first or the second sentence. For a given token, its input representation is constructed by summing the corresponding token, position, and segment embeddings [7].

Pre-training is divided into: Masked LM and Next Sentence Prediction (NSP). The first one, consists in masking some percentage of the input tokens at random (using the “[MASK]” token), and then, predict those masked tokens. The second one consists in, given two sentences A and B, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time B is a random sentence from the corpus (labeled as NotNext) [7].

Fine-tuning is straightforward since the self-attention mechanism in the Transformer allows BERT to model many downstream tasks. For each task, we simply plug in the specific inputs and outputs into BERT and fine-tune all the parameters $[7]$ .

# 4 Experiments

In order to compare BERT model with respect to the traditional machine learning NLP methodology, we have designed four experiments that are described throughout the section.

In these experiments, we will be using TfidfVectorizer from sklearn Python 3 module. After using TF-IDF to preprocess the text, we will be using Predictor from auto\_ml module (in the third and fourth experiments), and H20AutoML from h2o module (in the second experiment), to find the best model to fit the data. In the first experiment, we will, instead, show how much work needs to be done in order to get close to the results obtained, with no effort, using BERT model. For this purpose, we will be using many sklearn models and study their results in depth.

Regarding BERT's implementation, we have used the pre-trained BERT model from ktrain Python 3 module. This model expects the following directory structure: a directory which must contain two subdirectories: train and test.

Each one of them, in turn, must contain one subdirectory per class (named after the name of the class they represent). And, finally, each class directory, must contain the '.txt' files (their name is irrelevant) with the texts that belong to the class they represent.

# 4.1 IMDB experiment

In the first experiment, we have downloaded the IMDB dataset from the following website. It contains 50000 movie reviews (25000 to train the model and 25000 to test it) to perform sentiment analysis, a popular supervised learning text classification task. The dataset is classified into two different classes: Positive and negative movie reviews.

We have compared the behaviour of a pre-trained default BERT model w.r.t different popular machine learning models such as SVC or Logistic Regression that use a vocabulary extracted from a TF-IDF model obtaining the following results:

<table><tr><td>Model</td><td>Accuracy</td></tr><tr><td>BERT</td><td>0.9387</td></tr><tr><td>Voting Classifier</td><td>0.9007</td></tr><tr><td>Logistic Regression</td><td>0.8949</td></tr><tr><td>Linear SVC</td><td>0.8989</td></tr><tr><td>Multinomial NB</td><td>0.8771</td></tr><tr><td>Ridge Classifier</td><td>0.8990</td></tr><tr><td>Passive Aggressive Classifier</td><td>0.8931</td></tr></table>

Table 1. Accuracy retrieved by the different methodologies in the IMDB experiment over the validation set.

As we can see, BERT outperforms the rest of the models. It is noteworthy that obtaining these results with the traditional approaches has been far more complicated than obtaining this result with BERT.

# 4.2 RealOrNot tweets experiment

Our second experiment deals with the RealOrNot tweets written in English. We have downloaded the dataset from the following website. The task to solve here is pure binary text classification. It contains tweets classified into two different classes: Tweets about a real disaster and tweets which are not about a real disaster.

We have just used the tweet and class columns. We have also used the re Python 3 module to preprocess the tweets (\#anything -> hashtag, @anyone -> entity, etc.). After that, we have generated the directory structure that we

need to use BERT model (using 75% data to train and 25% data to validate). The obtained results have been summarized in the following table:

<table><tr><td>Model</td><td>Accuracy</td><td>Kaggle Score</td></tr><tr><td>BERT</td><td>0.8361</td><td>0.83640</td></tr><tr><td>H2OAutoML</td><td>0.7875</td><td>0.77607</td></tr></table>

Table 2. RealOrNot experiment results.

Finally, we have classified the data from the Kaggle competition with BERT. We have scored 0.83640. We can see this result here (Santiago González). Regarding the traditional approaches, the best classifier from the h2o module has turned out to be the H2OStackedEnsembleEstimator: Stacked Ensemble with model key StackedEnsemble\_BestOfFamily\_AutoML\_20200221\_120302. And, its score in the competition has been 0.77607.

# 4.3 Portuguese news experiment

Description Having seen that BERT has outperformed an AutoML technique and other classical machine learning algorithms using a vocabulary built from a traditional NLP technique such as TF-IDF in the English language, we choose to change the language to see if the BERT model also behaves well. We have downloaded the Portuguese news dataset from the following website. It contains articles from the news classified into nine different classes: ambiente, equilibriaosade, sobretudo, educacao, ciencia, tec, turismo, empreendedorsocial and comida.

We have just used the article text and class columns. We have generated the directory structure that we need to use BERT model (using 75% data to train and 25% data to validate obtaining the following results:

<table><tr><td>Model</td><td>Accuracy</td><td>Kaggle Score</td></tr><tr><td>BERT</td><td>0.9093</td><td>0.91196</td></tr><tr><td>Predictor (auto_ml)</td><td>0.8480</td><td>0.85047</td></tr></table>

Table 3. Portuguese news experiment results.

Finally, we have classified the data for the Kaggle competition scoring a 0.91196 accuracy. We can see this result here (Santiago González). Regarding the traditional methods, the best classifier has turned out to be a GradientBoostingClassifier. And, the score in the competition of this model has benn 0.85047.

# 4.4 Chinese hotel reviews experiment

Description Our last experiment involves a completely different language, Peninsular Chinese simplified characters zh-CN, where we hypothesize that, given that the way of expressing this Language is through different symbols that are not separated by spaces BERT may not output a good result. The experiment is a sentiment analysis problem involving Chinese hotel reviews. We have downloaded the dataset from the following website. It contains hotel reviews classified into two different classes: Positive hotel reviews and negative hotel reviews.

In this experiment, we have used 85% of the data to train the model and 15% of the data to validate it. Results are given in the following table:

<table><tr><td>Model</td><td>Accuracy</td></tr><tr><td>BERT</td><td>0.9381</td></tr><tr><td>Predictor (auto_ml)</td><td>0.7399</td></tr></table>

Table 4. Chinese hotel reviews results.

We can observe how, independently of the language and its characteristics, BERT behaviour outperforms classical NLP approach.

Finally, we have tried to do some predictions with BERT using Google Translator. For example, we have tried to predict a class for: 这家酒店的风景和服务都非常糟糕，which means: "the view and service of this hotel are very bad". The predicted class for this hotel review has been neg, which is correct.

Regarding the traditional approaches, the best model has turned out to be a GradientBoostingClassifier. But in this case, the model has been pretty bad, since the probability for both classes is very close. In this experiment, the importance of transfer learning has become apparent, since the dataset was pretty small compared to the ones used in the previous experiments.

# 5 Conclusions and further work

In this work we have introduced the BERT model and the classical NLP strategy where a machine learning model is trained using the features retrieved with TF-IDF and hypothesize about the behaviour of BERT w.r.t these techniques in the search of a default technique to tackle NLP tasks. We have introduced four different NLP scenarios where we have shown how BERT has outperformed the traditional NLP approach, adding empirical evidence of its superiority in average NLP problems w.r.t. classical methodologies. Furthermore, and of critical interest, implementing BERT has turned out to be far less complicated than implementing the traditional methods. It is also noteworthy the importance of transfer learning. We have been able to obtain this results thanks to pre-training.

Transfer learning has become more apparent in experiment 4.4 (which has the smallest dataset among all the experiments). We are nevertheless aware of the limitations of the BERT model. Although it seems that it is a good default for NLP tasks, its results can be improved. In order to do so, we would like to research in a hyperparameter auto-tuned BERT model for any new NLP task with Bayesian Optimization. We would like to use that auto-tuned BERT to enable classification of language messages for robots $[16]$ $[10]$ showing consciousness correlated behaviours.

# Acknowledgments

The authors gratefully acknowledge the use of the facilities of Centro de Computación Científica (CCC) at Universidad Autónoma de Madrid. The authors also acknowledge financial support from Spanish Plan Nacional I+D+i, grants TIN2016-76406-P and TEC2016-81900-REDT.

# References

1. Charu C Aggarwal and ChengXiang Zhai. A survey of text classification algorithms. In Mining text data, pages 163–222. Springer, 2012.   
2. Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.   
3. Romaric Besançon, Gaël De Chalendar, Olivier Ferret, Faiïza Gara, Olivier Mesnard, Meriama Laïb, and Nasredine Semmar. Lima: A multilingual framework for linguistic analysis and linguistic resources development and evaluation. In LREC, 2010.   
4. Daniel Bikel and Imed Zitouni. Multilingual natural language processing applications: from theory to practice. IBM Press, 2012.   
5. Justin Eliot Busch, Albert Deirchow Lin, Patrick John Graydon, and Maureen Caudill. Ontology-based parser for natural language processing, April 11 2006. US Patent 7,027,974.   
6. Erik Cambria and Bebo White. Jumping nlp curves: A review of natural language processing research. IEEE Computational intelligence magazine, 9(2):48–57, 2014.   
7. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pretraining of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.   
8. Atefeh Farzindar and Diana Inkpen. Natural language processing for social media. Synthesis Lectures on Human Language Technologies, 8(2):1–166, 2015.   
9. Eduardo C Garrido and Jesús Cardenosa Lera. Expansión supervisada de léxicos polarizados adaptable al contexto.   
10. Eduardo C Garrido-Merchán, Martin Molina, and Francisco M Mendoza. An artificial consciousness model and its relations with philosophy of mind. arXiv preprint arXiv:2011.14475, 2020.   
11. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

12. Clayton J Hutto and Eric Gilbert. Vader: A parsimonious rule-based model for sentiment analysis of social media text. In Eighth international AAAI conference on weblogs and social media, 2014.   
13. Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. Is bert really robust? natural language attack on text classification and entailment. arXiv preprint arXiv:1907.11932, 2019.   
14. Diksha Khurana, Aditya Koli, Kiran Khatter, and Sukhdev Singh. Natural language processing: State of the art, current trends and challenges. arXiv preprint arXiv:1708.05148, 2017.   
15. Christopher D Manning, Christopher D Manning, and Hinrich Schütze. Foundations of statistical natural language processing. MIT press, 1999.   
16. Eduardo C Garrido Merchán and Martín Molina. A machine consciousness architecture based on deep learning and gaussian processes. arXiv preprint arXiv:2002.00509, 2020.   
17. David Nadeau and Satoshi Sekine. A survey of named entity recognition and classification. Lingvisticae Investigaciones, 30(1):3–26, 2007.   
18. Daniel W Otter, Julian R Medina, and Jugal K Kalita. A survey of the usages of deep learning for natural language processing. IEEE Transactions on Neural Networks and Learning Systems, 2020.   
19. Cristina Puente, José Angel Olivas, E Garrido, and R Seisdedos. Creating a natural language summary from a compressed causal graph. In 2013 joint ifsa world congress and nafips annual meeting (ifsa/nafips), pages 513–518. IEEE, 2013.   
20. Stephen Robertson. Understanding inverse document frequency: on theoretical arguments for idf. Journal of documentation, 2004.   
21. Kristian Rother and Achim Rettberg. Ulmfit at germeval-2018: A deep neural language model for the classification of hate speech in german tweets. 2018.   
22. Efstathios Stamatatos. Plagiarism detection using stopword n-grams. Journal of the American Society for Information Science and Technology, 62(12):2512–2527, 2011.   
23. Ian Tenney, Dipanjan Das, and Ellie Pavlick. Bert rediscovers the classical nlp pipeline. arXiv preprint arXiv:1905.05950, 2019.   
24. Bruno Trstenjak, Sasa Mikac, and Dzenana Donko. Knn with tf-idf based framework for text categorization. Procedia Engineering, 69:1356–1364, 2014.   
25. Hiroshi Uchida and Meiying Zhu. The universal networking language beyond machine translation. In International Symposium on Language in Cyberspace, Seoul, pages 26–27, 2001.   
26. Peter Usherwood and Steven Smit. Low-shot classification: A comparison of classical and deep transfer machine learning approaches. arXiv preprint arXiv:1907.07543, 2019.   
27. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.   
28. Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.   
29. Zhang Yun-tao, Gong Ling, and Wang Yong-cheng. An improved tf-idf approach for text classification. Journal of Zhejiang University-Science A, 6(1):49–55, 2005.   
30. Yin Zhang, Rong Jin, and Zhi-Hua Zhou. Understanding bag-of-words model: a statistical framework. International Journal of Machine Learning and Cybernetics, 1(1-4):43–52, 2010.

This figure "BERT\_attention.jpg" is available in "jpg" format from:

http://arxiv.org/ps/2005.13012v2

This figure "BERT\_encoder.jpg" is available in "jpg" format from:

http://arxiv.org/ps/2005.13012v2

This figure "BERT\_input.jpg" is available in "jpg" format from:

http://arxiv.org/ps/2005.13012v2

This figure "BERT\_prefine.PNG" is available in "PNG" format from:

http://arxiv.org/ps/2005.13012v2

This figure "transformer\_encoder.PNG" is available in "PNG" format from:

http://arxiv.org/ps/2005.13012v2