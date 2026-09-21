# MEMBERSHIP INFERENCE ON WORD EMBEDDING AND BEYOND

Saeed Mahloujifar

Princeton University

sfar@princeton.edu

Huseyin A. Inan

Microsoft Research

huseyin.inan@microsoft.com

Melissa Chase

Microsoft Research

melissac@microsoft.com

Esha Ghosh

Microsoft Research

esha.ghosh@microsoft.com

Marcello Hasegawa

Microsoft Corporation

marcellh@microsoft.com

# ABSTRACT

In the text processing context, most ML models are built on word embeddings. These embeddings are themselves trained on some datasets, potentially containing sensitive data. In some cases this training is done independently, in other cases, it occurs as part of training a larger, task-specific model. In either case, it is of interest to consider membership inference attacks based on the embedding layer as a way of understanding sensitive information leakage. But, somewhat surprisingly, membership inference attacks on word embeddings and their effect in other natural language processing (NLP) tasks that use these embeddings, have remained relatively unexplored.

In this work, we show that word embeddings are vulnerable to black-box membership inference attacks under realistic assumptions. Furthermore, we show that this leakage persists through two other major NLP applications: classification and text-generation, even when the embedding layer is not exposed to the attacker. We show that our MI attack achieves high attack accuracy against a classifier model and an LSTM-based language model. Indeed, our attack is a cheaper membership inference attack on text-generative models, which does not require the knowledge of the target model or any expensive training of text-generative models as shadow models.

# 1 Introduction

There has been a rich body of work (Mireshghallah et al., 2020; Tanuwidjaja et al., 2019; Hu et al., 2021) that investigates Machine Learning (ML) pipelines through the lens of privacy and information leakage. This body of work largely investigates the question of what information ML models capture and expose beyond the task at hand. A natural attack metric that is commonly used to understand the extent of information leakage from a ML model is Membership Inference (MI) attack. In MI attacks, an attacker is given black/white/grey box access to a ML model and aims to find out if a particular set of data was used in training the ML model. MI has been investigated in many domains such as vision (Shokri et al., 2017; He et al., 2020), generative adversarial networks (Hayes et al., 2019; Chen et al., 2020), graph neural networks (He et al., 2021; Olatunji et al., 2021) among many others.

Word Embedding In the text processing context, most ML models are built on word embeddings, which provide a mapping from words in a dictionary to vectors in an embedding space; these vectors can then be used as the input for the rest of the model. These embeddings are themselves trained on some dataset, potentially containing sensitive data. In some cases this training is done independently, in other cases it occurs as part of training a larger, task specific model. In either case it makes sense to consider membership inference attacks on this embedding layer, where the adversary tries to determine if a dataset was used in training the embeddings.

Membership inference attacks in this setting, i.e., in the context of embeddings and its effect in other natural language processing (NLP) tasks that use embeddings, are relatively unexplored $^{1}$ . Given the rich set of applications where word embeddings appear as the first block of deep neural models, this is somewhat surprising.

In this work, we investigate MI attacks on word embeddings and its two major applications in the NLP domain (namely, classification and text-generation). We assume label-only (also known as black-box) access to the target model. We note that, sometimes grey-box attacks (where the adversary gets confidence scores) have been referred to as black-box attack in the literature. But here we mean black-box access in the sense that the adversary only sees the predicted label. Grey-box attack model is strictly less general than black-box attack model since score-based attacks cannot be applied when the model only exposes the predicted label, which is often the case through the API boundary in production model deployment. In this work, we ask the following questions:

- Given black-box access to a word embedding function can a semi-honest attacker $^{2}$ . infer if her data was used in training the embedding?   
- If the semi-honest attacker has label-only access a model built on an embedding (instead of the embedding layer itself) that uses some embedding function, can the attacker still infer if her data was used in training the model?

We answer both these questions in positive. We choose two different NLP tasks: a spam classifier as a classification task and a language model as a text-generation task to show that the information leakage persists even when the embedding layer is not directly exposed. Perhaps the more surprising of these two applications is the spam classifier in which the attacker only gets access to the prediction label of spam or not spam.

Our attack tries to exploit a fundamental property of any “good” embedding function. A good embedding is expected to preserve some semantic meaning and relations of the underlying objects. In the case of word embeddings, this means, a good model is expected to capture semantic relationships between words. For example, the words king and male are expected to be closer together in the embedding space as compared to the pair king and female. Our attacks exploit this property in a clever way. We try to find some special word pairs in the attacker’s dataset that are not semantically close generally. Note that we still want all these special words to belong to the vocabulary of the training dataset to rule out trivial attacks. Then we try to infer if those words appear significantly closer in the embedding space, which would be a strong signal that the attacker’s data was used in the training.

To give an intuition about these special word pairs, imagine that the attacker is trying to infer if emails from the year late 2020 were used in training an embedding function. The words remote and work would appear much closer together in the emails from 2020 than from before. So (remote, work) is a special word pair, indicative of the email set of late 2020.

We build on this intuition to first attack the word embedding function directly. Our attack is can achieve success rates of above 90% in doing membership inference attack against word embeddings. Surprisingly, we show that this leakage from word embedding is transferable to the classification and text-generation NLP tasks.

Our attack provides a new path for attacking NLP models. For instance, our MI attack on text-generative language models does not require the knowledge of the target model or any expensive shadow model training (as opposed to previous work). While we show the success of our attack on an LSTM-based (Sundermeyer et al., 2012) language model, a by-product of our attack is a cheap MI attack on language models, which we expect, would generalize to MI attacks on other large transformer-based models (Vaswani et al., 2017) for which shadow model training would be highly costly.

# Contributions

1. We provide a novel membership inference (MI) attack against word embeddings that exploits an important property of any good word embedding: that it captures semantic relationship of words and that words adjacent in training data are considered semantically close. We show the success of our attack against Word2Vec which is one of the most commonly used word embedding algorithms. Our attack can achieve MI accuracy around $90\%$ . (See Section 3).

Note that, in contrast to previous work Song and Raghunathan (2020), our attack works in a very realistic setting where the attacker does not have knowledge of or sample access to the exact training data distribution.

$D^{*},\mathcal{D}$ , and $n$ are parameters of the game.   
1. Select a bit $b \in \{0, 1\}$ uniformly at random.
2. Sample $n$ distributions $\mathcal{D}_1, \ldots, \mathcal{D}_n$ from $\mathcal{D}$ .
3. Sample $D_i \leftarrow \mathcal{D}_i$ for $i = 1, \ldots, n - 1$ 4. If $b = 0$ , set $D_n = D^*$ . If $b = 1$ , sample $D_n \leftarrow \mathcal{D}_n$ 5. $D^*$ is given to the attacker, $A$ .
6. $A$ is assumed to have some other parameters which we will denote as aux
7. Train a model $M \leftarrow L(D_1, D_2, \ldots, D_n)$ .
8. $A$ adaptively queries the model on a sequence of points $x_1, \ldots, x_t$ and receives $y_1 = M(x_1), \ldots, y_t = M(x_t)$ .
9. $A$ then outputs a bit $b'$ and wins the game if $b = b'$ .   
Figure 1: Security Experiment for attack against distribution of users

2. Then we show how to extend our attack to membership inference against a spam classifier and a text-generative model when the attacker only has black-box access to these models. We show that our MI attack achieves 65 - 90% attack accuracy against the classifier model (Section 4) and 70% attack accuracy against LSTM-based language model (Section 5).   
3. A secondary contribution of our attack is providing a new approach for MI on NLP models. Our MI attack on text-generative language models for instance does not require the knowledge of the target model or any expensive shadow model training as opposed to previous work, while still providing successful attack performance. We further note that our attack transfers through LSTM-based language model where the trained embedding layer is not based on Word2Vec. Therefore, we expect our MI attack to generalize to other large transformer-based models for which shadow model training would be highly costly.

Organization The paper is organized as follows. In Section 2 we define the threat model and formal security experiments. We discuss our attack on word embedding in Section 3, on the classifier in Section 4 and on the text-generative model in Section 5. Finally, we discuss related work in Section 6 and conclude in Section 7.

# 2 Security Model

In this paper, we focus on black-box membership inference attacks, introduced in Shokri et al. (2017), in which an attacker can query a model with the goal of determining whether a particular dataset was included in the training data.

More formally, we consider a meta-distribution D, from which we will sample n distributions $D_{i}$ . Intuitively, D captures the distribution over users and $D_{i}$ captures the distribution over datasets of one particular user.

We consider an adversary that targets a particular user with dataset $D^{*}$ .³ First, n distributions are chosen from D, one corresponding to each user who provides training data. Specific datasets $D_{1},\ldots,D_{n}$ are chosen from the respective distributions. Dataset $D^{*}$ is given to the adversary, and a model is trained on either datasets $D_{1},\ldots,D_{n-1},D^{*}$ , or $D_{1},\ldots,D_{n}$ . The adversary makes a series of black-box queries and must determine which occurred. We formally model this as a security experiment in Figure 1.

Attack success against user dataset $D^{*}$ . To evaluate how vulnerable a particular user dataset is to membership inference attack, we run the experiment in Figure 1 many times and compute the success probability.

Attack success against distribution D of users. We can also compute an average success metric for the attack against a meta distribution D by sampling many user distributions, and then sampling a specific dataset for each, computing the estimated vulnerability of each dataset, and then averaging the results $^{4}$ .

$\mathcal{D}$ and $n$ are parameters of the game.

1. Pick $n$ distributions $\mathcal{D}_i$ for $i = 1, \ldots, n$ where $n$ is even.   
2. $D_{i}\gets \mathcal{D}_{i}$   
3. Run the following several times:

(a) Initialize a bit vector $\mathbf{b} = b_{1},\ldots ,b_{n} = 0^{n}$   
(b) Pick a random $n / 2$ subset of these datasets. Let us denote these datasets $D_1', D_2', \ldots, D_{n/2}'$ .   
(c) Set the corresponding bits in $\mathbf{b}$ to 1.   
(d) Train a model $M \leftarrow L(D_1', D_2', \ldots, D_{n/2}')$ .   
(e) For $i \in [n]$ : Run the attack as follows:

- The attacker is given $D_{i}$ for $i \in [n]$ .   
- The attacker is also given some other parameters which we will denote as aux   
- The attacker adaptively queries the model on a sequence of points $x_{1}, \ldots, x_{m}$ and receives $y_{1} = M(x_{1}), \ldots, y_{m} = M(x_{m})$ .   
- The attacker outputs a bit $b_i'$ indicating whether it thinks $D_i$ was included in the training set.

(f) Compute $S = \sum_{i=1}^{n}(b_i' = b_i) / n$

4. We compute a metric for the success of the attack by averaging the values of S obtained in each of the runs. This roughly captures the average success probability of the attacker attacking a randomly selected user.

Figure 2: Security Experiment for attack against a user dataset

Computing this attack metric can be quite expensive for more complex target models, in that it requires training the target model several times for each user. An alternate way to compute this latter success metric is as follows $^{5}$ :

We begin with a dataset which contains many users (say $n$ ). We train a model on a dataset containing the data from a random $n/2$ subset of the users. Then we run the attack for each user in the dataset, giving the attacker the goal of determining whether that user was in the chosen subset. We repeat this process several times to minimize the effect of randomness, and then report the attack's average success rate. We formally define the security experiment in Figure 2.

Variations The security experiments are generic and allows for implementation by varying the parameters of the experiment. We discuss this below.

- Access to the distributions: aux could include sample access to the distributions $\mathcal{D}_1, \ldots, \mathcal{D}_n$ or include sample access to distributions similar to $\mathcal{D}_1, \ldots, \mathcal{D}_n$ , but not those. For example, $\mathcal{D}_i$ could be users sampled from the Enron email distribution while a similar distribution could be the Avocado email distribution.   
- Target model types: Model $M$ could be three different types in our attacks (embedding, discriminators, and tex-generative models), but the security experiment can allow any model. The queries and outputs in the penultimate state of the security game is decided by the type of model. For example, if $M$ is a word embedding, the input is a word and the output is an embedding vector. If $M$ is a text generative model, the input is a text sequence and the output is the predicted next word.   
- Training on a subset of the data: As a variation, we can consider the case where the training algorithm only uses a random subset of each user/tenant's data. This makes the attacker's task somewhat more difficult because, while the attacker knows the dataset of the user/tenant in question, it does not know which subset of that user/tenant's data will be used. This more accurately reflects what happens in many real world training contexts.

# 3 Word Embedding Attack

Word embedding are maps where each word $w$ from a dictionary $Q$ is represented as a vector of fixed dimensions, $d$ . We denote the vector as $v_w \in \mathbb{R}^d$ . Word embedding vectors capture the semantic meaning of words using a distance

metric. Words with similar semantic meanings will have small cosine distance in the embedding space. As formally defined in previous section, the goal of the attacker is to find out whether a set of examples corresponding to a user has been used to train a target word embedding. To reach this goal, the adversary must find a signal in the input-output behavior of the embedding that reveals this information. A general framework to find such signals is shadow model training procedure where the adversary trains a batch of embeddings $m_{1}, \ldots, m_{k}$ as shadow models, with the data from the target user and $m_{1}^{\prime}, \ldots, m_{k}^{\prime}$ without the data from the target user and then tries to find some statistical disparity between these models.

When we are in the black-box setting, the statistical disparity should be manifested in the black-box behavior of the model which makes the job of adversary even harder. One naive way to find these statistical differences is to query the trained model on a large set and use the responses as representation for the model. Then, machine learning can help to identify if there is any difference between the responses that came from the set of models that had the target user data and models that did not have the target user's data (Shokri et al., 2017). Indeed, this is the first idea that comes to mind when trying to attack word embedding, just query the model an all words in the dictionary to get a representation of model in $R^{Q \cdot d}$ (where $Q$ is the number of words in the dictionary and $d$ is the dimension of the embedding space) and then figure out the membership of the target user by investigating this high dimensional representation. There are two issues with this approach that we explain below. This approach is expected to be effective as we expect that the membership of the target data to have its largest effect on the representation of the words that are present in the target dataset.

High Dimensions One major problem with the approach proposed above is that the dimension of the model representation is very large for machine learning models to handle. The size of dictionary is huge and that will be multiplied by the dimension of the embedding model. One way to fix this is instead of working with all the words in the dictionary, we can sub-sample a small fraction of words and use the responses from the model on them as the representation of the model. This is a naive way of reducing the dimension of the representation that is independent of the target user's data. A smarter way to the sub-sampling is to sample the words based on the target user's data. For example, one can query only the words that exist in the target dataset and this reduces the dimension significantly.

Randomness in the embedding space The goal of word embedding is to map the words into a geometric space in a way that preserves the semantic closeness. For example, one would expect the representation of the word “dog” to be closer to representation of “cat” than that of “cake”. This is indeed one only requirement that one will expect from an ideal word embedding. Now, for an embedding model m if one considers a transformation of $\pi$ that maps m to $m' = \pi(m)$ such that the relative distances between pairs of words are preserved then $m'$ will be as good as m in representing words. There are many possible transformation that preserve distance such as rotation or shifting. In that sense there is a high entropy in the representation of words which makes the job of adversary harder in identifying a statistical pattern.

Our Attack In order to handle the two issue mentioned above we propose an attack that uses $l_{2}$ distance between the representation of words in the users email data. This way we resolve the second issue as we look at the relative distance between words and we expect the randomness of the word embedding not to change the relative distances between pairs of words. The reason behind this expectation is the fact that word embeddings are designed to create correlation between the semantic distances of words and their Euclidean distance in the embedding space. This technique also resolves the first issue mentioned above. We only look at the pairs of words in the target dataset which reduces the dimension from $Q \cdot d$ in the naive representation to $\binom{T}{2}$ where T is the number of words in the target dataset. In order to further reduce the dimension, in our attack, we actually just consider the consecutive pairs of words which reduces the dimension to T - 1. The reason behind this is that we expect the presence of the target data in the training set should have a more significant effect on pairs of words that are closer to each other. Below, we describe our attack algorithms.

# 3.1 Attack as per the Security Experiments

$D^{*}$ is one user's data from Enron email data.

1. Select a bit $b \in \{0, 1\}$ uniformly at random.   
2. Sample $n$ distributions $\mathcal{D}_1, \ldots, \mathcal{D}_n$ from $\mathcal{D}$ . In our experiments all these distributions are Enron email distribution.

3. Sample $D_{i} \leftarrow \mathcal{D}_{i}$ for $i = 1, \ldots, n - 1$ . Here each $D_{i}$ represents a user's data. Since Enron data does not have a notion of user, we emulate a user by selecting random emails from the dataset.   
4. If $b = 0$ , set $D_{n} = D^{*}$ . If $b = 1$ , sample $D_{n} \leftarrow \mathcal{D}_{n}$   
5. $D^{*}$ is given to the attacker, $A$ .   
6. $A$ is assumed to have a sample access to a shadow distribution $\mathrm{aux} = \mathcal{D}_{\mathrm{shadow}}$ . We describe what shadow distributions we use in Section 3.2.   
7. Train a Word2vec embedding $M \leftarrow L(D_1, D_2, \ldots, D_n)$ .   
8. $A$ adaptively queries the model on a sequence of points $x_{1},\ldots ,x_{t}$ and receives $y_{1} = M(x_{1}),\ldots ,y_{t} = M(x_{t})$ . We discuss how $A$ prepares these query points in Algorithm 1.   
9. $A$ then outputs a bit $b'$ and wins the game if $b = b'$ . We discuss how $A$ produces this output in Algorithm 2.

Algorithm 1 Training a sparse linear attack using shadow model training.

Input: A target dataset $D^{*}$ Output: A linear attack model l and a query set W

1. Following the notation of the security experiment in Figure 1, let the target tenant data $D^{*}$ be a sequence of words $D^{*} = \{w_{0},\dots ,w_{s}\}$ .   
2. Sample multiple datasets $T_{1}, \ldots, T_{k}$ such that $D^{*} \in T_{i}$ from $\mathcal{D}_{\text{shadow}}$   
3. Sample multiple datasets $T_{1}^{\prime}, \ldots, T_{k}^{\prime}$ such that $D^{*} \notin T_{i}^{\prime} D_{shadow}$   
4. Train multiple embeddings $m_{1},\ldots,m_{k},m_{1}^{\prime},\ldots,m_{k}^{\prime}$ using the respective datasets   
5. Query the embeddings on all the words in $D^{*}$   
6. Compute the distances between embeddings of adjacent words $(w_0,w_1),(w_1,w_2),\ldots ,(w_{s - 1},w_s)$ for all embeddings $m_{i}$ and $m_i^\prime$ to get $\sigma (m_i) = (||m_i(w_1) - m_i(w_0)||_2,\dots ,||m_i(w_s) - m_i(w_{s - 1})||_2)$ .   
7. Find a linear function of these distances $l$ such that $l(\sigma(m_i)) > 0 \land l(\sigma(m_i')) < 0$ from the following family of functions: $L = \left\{l(x) = \beta + \sum_{j=1}^{s} \alpha_j x_j, \forall j \in \{1, \ldots, s\} \alpha_j \in \mathbb{R}; \beta \in \mathbb{R}\right\}$ by solving the following LASSO optimization problem:

$$
\min _ {l \in L _ {D ^ {*}}} \sum_ {i = 1} ^ {k} \left(1 - l (\sigma (m _ {i}))\right) ^ {2} + \left(1 + l (\sigma (m _ {i} ^ {\prime}))\right) ^ {2} + \lambda | l | _ {1}.
$$

with appropriate parameter $\lambda$ .

8. Let $l = (\beta, \alpha_1, \ldots, \alpha_s)$ . Then construct the query set $W = \{\text{all } w_i \text{ such that } \alpha_i \neq 0 \text{ or } \alpha_{i+1} \neq 0\}$ .

9. return l as the attack model and W as query set.

Algorithm 2 Membership inference on target word embedding model.

Input: A linear model l and a query set W
Output: A prediction $b'$

1. Query $M$ on all points in $W$ and use that to calculate $l(\sigma(M))$ . Note that the adversary only needs to query words in $W$ to calculate this quantity.   
2. if $l(\sigma(M)) < 0$ then
return $b' = 0$   
3. else
return $b' = 1$   
4. end if

What is the intuition behind this attack? Our attacks have two phases: the first phase is the preparation phase where the adversary prepares some queries for the target model (word embedding in this case) without any interaction with the target model. The second phase is the query phase which is interactive. In this phase, the adversary sends queries to the target models, collects the responses and processes them. In the first phase, the attack tries to find correlation between words that are special to the target data. For instance, imagine there is an embedding model that

uses the text in this paper for training an embedding model. It is natural to expect that when that happens, the distance between words "membership" and "inference" decrease compared to when this paper is not present (unless the most of the text in the dataset is about membership inference attacks.). Our attack tries to identify such pairs of words whose distance will significantly change upon including the target dataset. Note that in training the attack model there is a parameter $\lambda$ that we can control to change the number of important pairs of words found in the users email data. Because of unique properties of LASSO regression, we know that the final linear model is going to be sparse. This means that in the query phase the attacker will only need to query a few words and not all the words in the target dataset. By controlling the $\lambda$ parameter we can change the number of pairs of words that have non-zero weight in the linear combination. Changing $\lambda$ can also change the generalization of attack to unseen models. We describe the concrete details of the our attack in Section 3.2.

# 3.2 Attack results

We use our attack against the Word2Vec algorithm (Rong, 2014) according to security experiment 3.1. Figure 3 shows the success of the attack. For this experiment, we use Enron email distribution as the shadow distribution $D_{shadow}$ , target dataset distribution and, the distribution of rest of the training data. In Figure 3, we vary the size of target dataset. As expected, the attack works better with larger target datasets. Below, some details about this experiment is presented.

Experimental details We use a subset of 10000 emails from Enron to train word embeddings using the Word2Vec algorithm. We also have a target email set with size (denoted by s) varying from 1 email to 100 emails. This means the distribution of the target email set is equal to the distribution of all other emails (In particular, in the security experiment 3.1 $D^{*}$ is sampled from the same distribution D). The job of adversary is to guess whether the target email set has been used in the training set of the embedding model. Another variable is the number of (shadow) embedding models (denoted by k) that the attacker trains to optimize the attack model.

Data Preprocessing: We first preprocess the Enron and Avocado datasets by replacing all the words that are not in a public dictionary of words with a specific dummy word. For this purpose, we use the dictionary of a pretrained word embedding model on the Google News Corpus $^{6}$ . Then both datasets are divided into two equal parts. One part for shadow model training that adversary has access to and another part for training the target embedding model.

Embedding parameters We use Word2Vec (Rong, 2014) with window size 5. We use dimension 80 for the vector representation and iterate 20 times over all the data to train the Word2Vec models. The vocabulary of the model is set to the vocabulary of the entire prepossessed Enron dataset. Then, in the process of training, the algorithm is set to ignore all words that happen with frequency less than 20 in the training data.

Lasso Parameters: Our attack uses a Lasso parameter of $\lambda = 1/\sqrt{k}$ to ensure that the number of selected word pairs does not exceed 50.

Randomness and Repetitions In the upcoming figures, we repeat each experiment 50 times. We first sample 10 different target email sets from Enron distribution and for each target email set, we repeat the experiments security experiment 5 times. At the end, we report the average of the success of adversary in all experiments.

# 3.3 Other variants of the attack

The attack described above has three major assumptions that might be unrealistic in some scenarios. First, our attack is assumed to have sample access to the distribution of text data used for training the embedding. Second, we assume that the target data that adversary aims at doing membership inference on is either completely used in the training set or not used at all. And third, we assume black box access to the embedding model. In the rest of this section, we discuss how our attack can still succeed in scenarios where the first two assumptions are violated. Then, in Sections 4 and 5.1 we discuss how the attack can still work even if it does not have access black-box access to an embedding.

Knowledge of the training distribution Assumption In order to train shadow embeddings, it seems crucial for the adversary to have sampling access to the same distribution that the model trainer uses. Although this assumption is relevant in many scenarios, it will still be interesting to see how removing this assumption can affect the adversary. In order to understand the effect of this assumption on the attack, we perform experiments where the adversary does

![](images/fc51adaa2d3974a043b39be7208643787add0ebe5f17809f65ace90d5930d752.jpg)

<details>
<summary>line</summary>

| Number of shadow models | 1 Email | 10 Emails | 50 Emails | 100 Emails |
| ----------------------- | ------- | --------- | --------- | ---------- |
| 50                      | 0.68    | 0.82      | 0.96      | 0.99       |
| 100                     | 0.80    | 0.87      | 0.98      | 1.00       |
| 150                     | 0.82    | 0.96      | 1.00      | 1.00       |
| 200                     | 0.83    | 0.97      | 1.00      | 1.00       |
| 250                     | 0.84    | 0.97      | 1.00      | 1.00       |
| 300                     | 0.84    | 0.97      | 1.00      | 1.00       |
| 350                     | 0.84    | 0.97      | 1.00      | 1.00       |
| 400                     | 0.84    | 0.97      | 1.00      | 1.00       |
</details>

Figure 3: As numbers of shadow models increase, the attack becomes more successful in performing membership inference. In all setting, the attack seem to reach a plateau after 200 shadow models. Interestingly, the attack can success even when there is a single email in the target email set.

![](images/c9b124575b8c8ad53408941c81d569cd9591467e6833b33109f78ba634252496.jpg)

<details>
<summary>line</summary>

| Number of shadow models | 1 Email | 10 Emails | 50 Emails | 100 Emails |
| ----------------------- | ------- | --------- | --------- | ---------- |
| 50                      | 0.63    | 0.74      | 0.86      | 0.81       |
| 100                     | 0.67    | 0.77      | 0.88      | 0.84       |
| 150                     | 0.66    | 0.79      | 0.95      | 0.87       |
| 200                     | 0.69    | 0.83      | 0.96      | 0.87       |
| 250                     | 0.71    | 0.84      | 0.96      | 0.90       |
| 300                     | 0.74    | 0.85      | 0.96      | 0.94       |
| 350                     | 0.79    | 0.85      | 0.97      | 0.95       |
| 400                     | 0.79    | 0.85      | 0.96      | 0.95       |
</details>

Figure 4: Success of the attack when the shadow models are trained on a proxy distribution (Avocado) instead of the original (Enron). Interestingly, the knowledge of the exact distribution is not crucial to the attack if enough number of shadow models are trained.

not have access to the same text distribution. In this set of experiments, the adversary uses the Avocado dataset to train shadow embeddings while the actual target model is trained on Enron dataset. Figure 4 shows the success of the attack in these experiments when enough number of shadow models are trained. All the parameters in this experiment is similar to the experiments done in Figure 3.

All-or-nothing Assumption Another assumption behind the attack that might be violated is that the target data $D^{*}$ will be either completely used or will not be used at all. In real scenarios, the model trainer can sub-sample some data from each user and then train on that. This can potentially hurt the attack as the identified pair of words might not appear in the training set of the model at all. In order to account for this, we run an adaptive attack against this sub-sampling defense. In this attack, the adversary incorporates the sub-sampling step in the training algorithm used

![](images/74cf9ace8f76ae19b70a62aa5106cf2bb3dce783120c06365c8f13ee9bf1f718.jpg)

<details>
<summary>line</summary>

| Number of shadow models | 1 out of 100 Emails | 10 out of 100 Emails | 50 out of 100 Emails | 100 out of 100 Emails |
| ----------------------- | ------------------- | -------------------- | -------------------- | --------------------- |
| 50                      | 0.52                | 0.54                 | 0.64                 | 1.00                  |
| 100                     | 0.58                | 0.57                 | 0.70                 | 1.00                  |
| 150                     | 0.59                | 0.61                 | 0.75                 | 1.00                  |
| 200                     | 0.58                | 0.64                 | 0.84                 | 1.00                  |
| 250                     | 0.59                | 0.65                 | 0.88                 | 1.00                  |
| 300                     | 0.59                | 0.65                 | 0.91                 | 1.00                  |
| 350                     | 0.60                | 0.66                 | 0.91                 | 1.00                  |
| 400                     | 0.60                | 0.66                 | 0.91                 | 1.00                  |
</details>

Figure 5: Success of attack when a fraction of target dataset is used in training.

for training the shadow models. When this adaptive attack is used, the attack is has to select a more diverse set of word pairs so that if some of them did not appear in the actual training set, then other pairs can still help to perform the inference. Figure 5 shows the success of our attack in this setting.

# 4 Embedding Attack on a Text-Classification Model

Our embedding attack in Section 3 works when the adversary has black-box access to the embedding. However, the embeddings might not be directly accessible by adversary. For example, adversary might only have access to a classifier that is trained on a public dataset which uses the embedding in its training process. In this section, we try to address this kind of setting.

# 4.1 Attack as per the Security Experiment

The security experiment here is identical to the experiment in Section 3.1 except that additional to $D^{*}$ it has another parameter $\mathcal{D}_c$ which a labeled data distribution that is used for training the classifier. Additionally the following steps are different:

- $D^{*}$ is the email data set of a user of Avocado dataset.   
- In Step 3, the $\mathcal{D}_i$ distributions are distribution of emails from Avocado dataset (instead of Enron).   
- In Step 7, after training Word2vec model $M$ , a dataset $D_c \leftarrow \mathcal{D}_c$ is sampled and then mapped to embedding space using $M$ to get $M(D_c)$ . Then a spam classifier is trained on $M(D_c)$ to get a linear classifier $C$ .   
- In Step 8, the attacker adaptively queries the classifier $C$ (instead of the Embedding model) and prepares a set of query documents (instead of query words) using Algorithm 3.   
- In Step 9, the attacker runs Algorithm 4 to decide on the bit $b'$ .

We now propose our attack for this setting.

Intuition behind the attack: The main goal of the design of our classification attack is to extract the information from the embedding attack described in Section 3. In particular, as we keep appending the special word $w_{i}$ to a document d, we might see some change in the prediction of classifier when the number of repetitions of the word exceeds some threshold $\sigma_{i}$ . This threshold gives us a mean to measure the closeness of two words $w_{i}$ and $w_{j}$ by comparing the value of $\sigma_{i}$ and $\sigma_{j}$ . Specifically, if word $w_{i}$ and $w_{j}$ are close together in the embedding space, we expect them to have similar behavior in changing the prediction of the classifier. Although this measure is different from the distance of the words in the embedding space, it still has enough information to make the attack succeed with proper shadow model training.

# Algorithm 3 Training the attack using shadow model training.

# Input: A target data $D^{*}$

Output: An attack model $l$ and a query set $S$ .

1. perform the preparation step of the word embedding attack as described in Algorithm 1 on target dataset $D^{*}$ to get the set of words $W$ . Also keep the embeddings $m_{1},\ldots ,m_{k},m_{1}^{\prime},\ldots ,m_{k}^{\prime}$ .   
2. Sample $2k$ different datasets $T_{1}^{c},\ldots ,T_{2k}^{c}$ from the classification distribution $\mathcal{D}_c$ and use embeddings to map them into the embedding space to get $m_1(T_1^c),\ldots ,m_k(T_k^c),m_1'(T_{k + 1}^c),\ldots ,m_k'(T_{2k}^c)$ .   
3. Train $2 \cdot k$ different linear classifiers $\{c_1, \ldots, c_k, c_1', \ldots, c_k'\}$ on the embedded datasets of previous step.   
4. create a set of queries by first sampling a set of random query documents $q_{1}, \ldots, q_{1}00$ from $D_{c}$ .   
5. For $q_{i}$ , each word $w$ in the set of word pairs $W$ and each $0 \leq j \leq 5$ , create a query point $q_{(w,i,j)}$ which is same as $q_{i}$ with the difference that it is appended with $i$ repetitions of word $w$ . Let $S$ be the union of all $q_{w,i,j}$ .   
6. Query each $c_{i}$ and $c_{i}'$ on $S$ to get a set of labels $L_{i}$ and $L_{i}'$ respectively.   
7. Create a dataset $\{(T_1, 1), \ldots, (T_k, 1), (T_1', 0), (T_k', 0)\}$ and train an attack classifier $l$ on this dataset using a Random forest classifier.   
8. return $l$ as the attack model and $S$ as the set of queries.

# Algorithm 4 Membership inference on the target classification model.

Input: Attack model l and a query set S

Output: A prediction $b'$

1. Query $C$ on $S$ to get a set of labels $L$ .   
2. return $l(L)$ .

# 4.2 Attack Success

Experiment details: We evaluate our attack in a setting where the dataset used for embedding is different from dataset used for classification. In particular, the embedding data is sampled from Avocado dataset whereas the classification data is sampled from Enron dataset. We use 30 random users from Avocado dataset to train a word embedding. Then use that embedding to transfer emails in the Enron dataset into the embedding space. Then we train a linear classifier on embedded Enron dataset. On average, we get around 93% accuracy for this classifier (See Table 3). Now the goal of adversary is to guess if the training set for the underlying embedding include the target dataset.

Experiments Our attack uses 500 shadow embedding and 500 shadow classifiers and uses lasso parameter 0.1 to come up with the set of words. Then after getting the set of words our adversary constructs a set of query points as described in the attack algorithm. Our final attack model is a random forest classifier applied on the shadow classifiers. Table 1 shows the success of our attack.

We also try another variant of the attack where the data used for classification and embedding is the same. This setting will be relevant in scenarios where the word embedding and classifier are trained together. The security experiment for this setting will be similar to the classification experiment except that there is no dataset for the classification algorithm. Instead, the classifier is trained on the same data that is used for training the embedding. Similarly, in the algorithm for training the shadow models, the adversary will use the same sampled dataset to train the shadow models. This variant of the attack has less entropy and the attack is expected to be more successful. Table 2 shows the success of attack in this setting.

<table><tr><td>TARGET DATASET SIZE</td><td>ATTACK ACCURACY</td></tr><tr><td>10</td><td>0.65±8.15</td></tr><tr><td>100</td><td>0.69±5.08</td></tr><tr><td>1000</td><td>0.79±3.75</td></tr></table>

Table 1: Attack success v.s. number of sentences in the target dataset. In these experiments, Avocado data is used for training the embedding and the Enron data is used for training the classifier.

<table><tr><td>TARGET DATASET SIZE</td><td>ATTACK ACCURACY</td></tr><tr><td>10</td><td>0.77±5.12</td></tr><tr><td>100</td><td>0.84±2.35</td></tr><tr><td>1000</td><td>0.91±1.78</td></tr></table>

Table 2: Attack success v.s. number of sentences in the target dataset. These experiments capture the setting that the same data is used to train the classification and embedding models.

<table><tr><td>MODEL</td><td>EMBEDDING DATASET</td><td>CLASSIFICATIONACCURACY (TRAIN)</td><td>CLASSIFICATIONACCURACY (TEST)</td></tr><tr><td>SVM</td><td>ENRON</td><td> $95.18 \pm 2.12$ </td><td> $94.27 \pm 1.05$ </td></tr><tr><td>SVM</td><td>AVOCADO</td><td> $93.98 \pm 1.95$ </td><td> $93.34 \pm 1.08$ </td></tr></table>

Table 3: Accuracy of the target classifier.

# 5 Embedding Attack on a Text-Generation Model

In the previous section, we show an application of our embedding attack to a text-classification model. In this section, we extend our approach to attack a text-generation model. We focus on the next-word prediction task as the text-generation setting. This is a very popular setting in which language models have been deployed in practice to perform text auto-completion in emails and predictive keyboards (Microsoft SwiftKey, [n.d.]; Chen et al., 2019). On the other hand, such models are extensively trained on personal data, e.g. users' emails, documents, chats etc., which may lead to privacy leakages as studies show in Song and Shmatikov (2019); Carlini et al. (2019, 2020); Inan et al. (2021).

# 5.1 Attack as per the Security Experiment

Dataset We use the Avocado dataset (Oard et al., 2015) in our experiments. The dataset consists of 279 users. We split up the dataset in two parts: $D_{target}$ and $D_{shadow}$ where each part contains 100 users' data picked randomly from the dataset. $D_{shadow}$ is only used in the preparation phase of the attack. $D_{target}$ is further divided in two parts as $D_{in}$ and $D_{out}$ randomly, each containing data of 50 users. The target model trains on $D_{in}$ and the attack experiment is performed on both $D_{in}$ and $D_{out}$ .

Target model Similar to the setting in Song and Shmatikov (2019), we use long short-term memory (LSTM) (Hochreiter and Schmidhuber, 1997) in our language model. LSTM is a special type of Recurrent Neural Networks (RNNs) that can capture the long-term dependency in the text sequence. In this network, the text sequence of tokens is first mapped to a sequence of embeddings. The embedding is then fed to the LSTM that learns a hidden representation for the context for predicting the next word. We use a two-layer LSTM model as the language model for the next-word prediction task. We set both the embedding dimension and LSTM hidden-representation size to 500 (gives around 50 million parameters). All model weights including the embeddings are initialized randomly before training. We use the Adam optimizer with the learning rate set to 1e-3 and batch size to 64. We repeat our experiments over five random runs. The performances of the models are presented in Table 4.

Recent work has shown both successful extraction of training data (Zanella-Béguelin et al., 2020; Carlini et al., 2020; Inan et al., 2021) and membership inference (Song and Shmatikov, 2019) in language models. The membership inference attack we describe in this section is different in the sense that our method only uses the top-1 predictions of the model, i.e. our attack is label-only, which is a strong and realistic attack setting that has been explored recently in

<table><tr><td>MODEL</td><td>TRAIN PPL</td><td>TEST PPL</td></tr><tr><td>2-LAYER LSTM</td><td>94.70±3.16</td><td>107.74±2.55</td></tr></table>

Table 4: Performances of the target models. The mean and standard deviation are given over five runs. Ppl stands for perplexity.

visual domain as well (Choquette-Choo et al., 2020; Li and Zhang, 2021). Another advantage of our attack is that it does not require any shadow training as opposed to Song and Shmatikov (2019), hence, the target model need not be assumed to be known and the attack is computationally efficient. We next describe the attack setting as per the security experiment.

1. In this experiment, each of the $n = 100$ distributions $\mathcal{D}_i$ for $i = 1, \ldots, n$ is a user's email distribution from the Avocado dataset.   
2. $D_{i}\gets \mathcal{D}_{i}$ .Here $D_{i}$ is user i's data.   
3. Run the following several times:

(a) Initialize a bit vector $\mathbf{b} = b_{1},\ldots ,b_{n} = 0^{n}$   
(b) The dataset $D_{\text{target}}$ is split in two parts: $D_{\text{in}}$ and $D_{\text{out}}$ where each part contains data of 50 users. $D_{\text{in}}$ and $D_{\text{out}}$ are constructed as follows: The user id's in $D_{\text{target}}$ are randomly permuted and then the first 50 users constitute $D_{\text{in}}$ and rest constitute $D_{\text{out}}$ . Let us denote the datasets in $D_{\text{in}}$ as $D_1', D_2', \ldots, D_{50}'$ .   
(c) Set the corresponding bits in $\mathbf{b}$ to 1.   
(d) Train a model $M \leftarrow L(D_1', D_2', \ldots, D_{50}')$ .   
(e) For $i \in [n]$ : Run the attack as follows:

- The attacker is given $D_{i}$ for $i \in [n]$ .   
- The attacker has also aux = $D_{shadow}$   
- $A$ adaptively queries the model $M$ on a sequence of points. The points are chosen as follows:

i. Preparation: This phase is identical to the preparation phase described in Section 3. The shadow data used is $D_{\text{shadow}}$ . Let $\mathcal{W}_u$ be the words output by the preparation phase for user $u \in D_{\text{target}}$ .

ii. For each user $u \in D_{\text{target}}$ : Apply Algorithm 5 with $W_u$ to get $b_u'$ .

\- The attacker outputs $b_i'$ indicating whether it thinks $D_i$ was included in the training set.

(f) Compute $S = \sum_{i=1}^{n}(b_i' = b_i) / 100$

4. We compute a metric for the success of the attack by averaging the values of $S$ obtained in each of the runs.

Let us expand the Step 3(e)ii of the security experiment introduced above. For each user in the target dataset, we apply the attack described in Section 3 using the shadow dataset, which generates a list of word pairs $(w_{i}, w_{i+1})$ . Let us denote this list as W. The attack operates on each pair $(w_{i}, w_{i+1})$ in a simple way by using the sequence that has this pair $(w_{i}, w_{i+1})$ , querying the model with the context of up to and including $w_{i}$ and checking if the top prediction returned by the model is $w_{i+1}$ . If there are multiple sequences that has the pair $(w_{i}, w_{i+1})$ , the operation is performed on all of them. If there exists a pair of words in W satisfying this condition, the corresponding user is predicted as member. Otherwise, the user is predicted as non-member. The attack is described in Algorithm 5.

Algorithm 5 Label-only membership inference attack performed in Step 3(e)ii of the security experiment.   
Input: A language model $LM(\cdot)$ and the attack data $\mathcal{W}_{\mathrm{u}}$ of user $\mathsf{u} \in \mathsf{D}_{\mathrm{target}}$ Output: The membership prediction $b_{\mathrm{u}}^{\prime}$ for $(w_{i}, w_{i+1})$ in $\mathcal{W}_{\mathrm{u}}$ do

    Construct $L = \{s \in D_{u} : (w_{i}, w_{i+1}) \in s\}$ where $s$ are sentences that contain $(w_{i}, w_{i+1})$ for $s$ in $L$ do

    Find the index $j$ such that $s_{j} = w_{i}$ Obtain the next-word prediction $p = LM(s_{1}, \ldots, s_{j})$ if $p = w_{i+1}$ then

    return $b_{\mathrm{u}}^{\prime} \leftarrow 1$ end if

end for

end for

return $b_{\mathrm{u}}^{\prime} \leftarrow 0$

<table><tr><td>ATTACK</td><td>ACCURACY</td><td>PRECISION</td><td>RECALL</td></tr><tr><td>OUR ATTACK</td><td>0.708±0.023</td><td>0.67±0.025</td><td>0.82±0.047</td></tr><tr><td>BASELINE 1</td><td>0.5±0.0</td><td>0.5±0.0</td><td>1.0±0.0</td></tr><tr><td>BASELINE 2</td><td>0.637±0.032</td><td>0.69±0.042</td><td>0.478±0.066</td></tr></table>

Table 5: Attack results of Section 5.1 over five runs (mean & standard deviation). Baseline 1 is applying Algorithm 5 for all word pairs in a user's data. Baseline 2 is applying Algorithm 5 for all word pairs except the ones that can be found in a public dictionary (GloVe).

# 5.2 Attack results

We note that for a member prediction, it is sufficient to have at least one pair of words satisfying the described condition. Therefore, a well-curated list of word pairs should be generated so that the condition is not satisfied for any of the word pairs for a non-member and it is satisfied for at least one pair for a member. The approach described in Section 3 is effective in the sense that it can generate such well-crafted list of word pairs for each user. The results of the security experiment, which has been repeated five times with different randomness, are presented in Table 5.

We observe that with label-only access and not training any shadow models, our attack performs impressively. This is the first attack in such a setting to the best of our knowledge. For comparison, the attack in Song and Shmatikov (2019) under label-only access boils down to calculating the average accuracy over each user's data and selecting a threshold to decide membership using shadow model training. Without the latter step, the attack would naturally result in all-member prediction since the attack essentially uses all word pairs in a user's data and each user has at least one word pair satisfying the required condition. This is equivalent to the first baseline in Table 5 where we apply the attack in Algorithm 5 using all word pairs in a user's dataset. We point out another advantage of our method that it is query-efficient in the sense that it uses a curated list of word pairs, therefore, this substantially reduces the number of queries to the model compared to Song and Shmatikov (2019). This means that our attack cannot trivially be mitigated by limiting the number of queries made to the model.

As a second baseline, we try to obtain a curated list of word pairs by excluding common word pairs if they are present in a public dictionary (we use GloVe public dictionary (Pennington et al., 2014b)). As noted in Song and Shmatikov (2019) successful prediction of rare words provide strong signal for membership inference. Although this improved the attack performance as expected, it still substantially falls short of our attack. The results in Table 5 show that our approach indeed generates a well-curated list of word pairs that provides good signal for membership inference. We finally highlight the fact that our attack, which is based on Word2Vec embedding, transfers through the LSTM-based language model, which trains its own embeddings different from Word2Vec. Therefore, we expect our MI attack to generalize to other text-generation models such as ones that are based on large transformer (Vaswani et al., 2017) models for which shadow model training would be highly costly.

# 6 Related work

Embeddings, as a representation of the string format of text, has been one of the key stepping stones to successful machine learning models in NLP applications. Utilizing unsupervised learning on large corpus of text, various methods such as Word2Vec (Mikolov et al., 2013), Glove (Pennington et al., 2014a), fastText (Bojanowski et al., 2017) have been designed to produce a vector space where each word in the corpus is assigned a vector (embedding) in the generated space. These word embeddings can be more compact and maintain semantic similarity by being in close proximity to one other in the vector space, therefore, more favorable than one hot encoded vectors. Embeddings are the first block of the deep neural models such as ones that are based on recurrent neural networks (RNNs) (Mikolov et al., 2010; Sundermeyer et al., 2012) or based on self-attention mechanisms of the transformer (Vaswani et al., 2017). These models have been widely employed in NLP applications and the granularity of the embedding can be word level or sub-word level (e.g. BPE tokenization (Sennrich et al., 2016)) depending on the model architecture.

When machine learning models are trained on sensitive personal data, utility as the performance of the model should not be the only metric of attention. In fact, a wide body of work has demonstrated privacy issues for machine learning models trained on personal data. In general, it is known that deep learning models can achieve perfect accuracy even on randomly labeled data (Zhang et al., 2017). Strong memorization ability may actually be required to achieve near-optimal accuracy on test data when the data distribution is long-tailed as recently shown by Feldman (2020); Brown et al. (2020). There are serious implications of this in the NLP domain that may lead to privacy breaches. For instance,

Carlini et al. (2020) demonstrated that individual training examples from the GPT-2 language model (Radford et al., 2019) can be recovered verbatim. In a transfer learning setup, Zanella-Béguelin et al. (2020) have shown that by having simultaneous black box access to the pre-trained and fine-tuned language models, rare sequences from the typically more sensitive fine-tuning dataset can be extracted successfully. In this regard, recent work (Carlini et al., 2019; Inan et al., 2021; Mireshghallah et al., 2021) have also proposed metrics and mitigations to evaluate generative models from the perspective of privacy.

In case of classification tasks, label-only access may obstruct such a direct leakage from the training data. However, an indirect leakage of what is called membership inference attack (Shokri et al., 2017) can still lead to privacy violations (Murakonda and Shokri, 2020). In a membership inference attack, the goal is to determine if a particular data point or a targeted user belongs to the training set of the model. There has been substantial progress in this area over a wide range of applications under different assumptions/settings (Shokri et al., 2017; Yeom et al., 2018b; Song and Shmatikov, 2019; Nasr et al., 2019; Long et al., 2018; Hayes et al., 2019; Truex et al., 2018; Irolla and Châtel, 2019; Hisamoto et al., 2020; Salem et al., 2018; Sablayrolles et al., 2019; Leino and Fredrikson, 2020; Choquette-Choo et al., 2020).

While the settings in the aforementioned work are different than what we are focusing on in this work, we highlight the article Song and Raghunathan (2020) as it is closest to this work. Song and Raghunathan (2020) studies information leakage from embeddings and we compare it with our work in great detail in the next section.

# 6.1 Comparison with Song and Raghunathan (2020)

In Song and Raghunathan (2020), the authors consider three types of attacks: 1) embedding inversion 2) sensitive attribute inference and 3) membership inference (MI). In this paper, we focus on MI and how it can be further used in downstream tasks. Like in the previous paper, we also focus on text input data. We will compare our attacks with the MI attacks in Song and Raghunathan (2020).

In Song and Raghunathan (2020), the authors address the following question: can an adversary with access to the embeddings extract the encoded sensitive information? The MI attack on word embeddings that the authors propose also has several differences from ours.

We go beyond what Song and Raghunathan (2020) consider as sensitive information: we also consider the question of what kind of sensitive information an adversary can learn given black-box access to a model that uses embeddings as its first layer (rather than direct access to the embedding itself).

Definition of MI The paper points out that unlike supervised learning, word embeddings trivially allows for word level MI: every word in the vocabulary is trivially a member of the training dataset $D_{train}$ . So, to meaningfully talk about MI, the definition of MI needs to be expanded. The way the authors expand on this is the following: They consider that the adversary has a target string of words $[w_{1},\ldots,w_{n}]$ , referred to as a context, and access to the embedding. The adversary tries to decide the membership for the context string.

This definition, while interesting, is a little limited, in that the attack is very tied to the length of the input. For example, it's not clear how it would apply to membership inference for inputs that may have different length. On the other hand our security notion could apply to membership of substrings of specific length, but it can also apply to membership of emails/documents, or of a users' email or document collections, without any assumption that all users' emails/collections must have the same number of words.

Assumption of adversary's knowledge In the MI attack in Song and Raghunathan (2020), the attacker has access to some auxiliary data labeled with membership. Moreover, in their attack, they show that their adversary's advantage increases as the central word in the 5-window context becomes rarer. Therefore, the success of the attack requires the adversary to pick a context whose central word is rare in the training distribution, which implicitly assumes that the adversary has enough knowledge of the training distribution to find these rare words. Moreover, success of their attack depends on finding these specially constructed context strings.

This is in contrast to ours, where the adversary does not have access to any auxiliary labeled data. In fact, we show that, the adversary does not even need sample access to the exact training distribution of the target model; access to a similar distribution can be sufficient. Relaxing the assumption on the adversary's knowledge makes our attack stronger.

Success metric Song and Raghunathan (2020) show that their attack is successful for certain types of contexts (specifically those whose central word is rare). This is interesting in that it says specifically which types of substrings are vulnerable, but it leaves open the question of how common such substrings are in real datasets.

Our analysis, in contrast, shows that our attack is successful for input data drawn from real datasets.

Attack specification The attack defined in Song and Raghunathan (2020) is threshold based: they measure average cosine similarity of all word pairs in the context and sees if it is over a certain threshold. However, in the experiment they do not discuss how they compute the threshold. Deciding on a threshold is non-trivial and probably requires some additional shadow model training.

In contrast to this, in our attacks that use thresholds, we specify exactly how the thresholds would be computed, so this cost is already included in the analysis of our attack. Our text-generative model attack has the advantage that it doesn't require a threshold.

Fundamental attack intuition The attack in Song and Raghunathan (2020) is fundamentally based on the insight that rarer words are memorized more. They check with sliding window of size 5 with varying frequency of the central word and show that adversary's advantage increases as the central word becomes rarer. Their sentence embedding MI uses a similar idea.

This is fundamentally different from the intuition of our attack. We exploit the following property of embedding: a good embedding is expected to capture semantic relationships between words. This is widely believed to be an inherent property of a good embedding function (Schnabel et al., 2015). We combine this with the assumption, inherent in the way that embeddings are trained, that words that are adjacent/close in the training data will be somehow semantically close.

# 7 Conclusion

In this paper we looked at a simple embedding function that lies at the heart of almost any NLP application, namely, word embedding. First we show that word embeddings are vulnerable to black-box membership inference attack. Then we show that this leakage persists through two other major NLP applications: classification and text-generation, even when the embedding layer is not exposed to the attacker. Our attacks exploit a property of word embeddings (preserving semantic relationship of words), which is widely believed to be a property of a good embedding function. Whether this leakage is in some way inherent is a question that requires further investigation.

A secondary contribution of our attack is a cheaper membership inference on text-generative models, which does not require any expensive training of text-generative models as shadow models.

Given the vulnerability of word embeddings and its persistence through other downstream tasks that our work exposes, and the ubiquitous use of word embeddings in almost all NLP tasks, we believe our work raises an important question of how to mitigate such vulnerability. Applying differential privacy (Dwork, 2011) in training (Abadi et al., 2016) will protect the model as a whole, hence including the embedding layer. Therefore, this is the most obvious possible defense here, but differentially private model training is substantially slower in comparison and may affect the utility of the model negatively (Bagdasaryan et al., 2019). While this line of defense is being actively researched, it is also worth investigating other heuristic defenses that could cater to specific applications.

We conclude the article with two possible lines of future work. The first one is to measure the effectiveness of heuristic defenses or differential privacy with realistic privacy budget in defending against our attack. The second is to apply our attack to other models such as those based on transformers.

# References

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. 2016. Deep learning with differential privacy. In ACM CCS.   
Eugene Bagdasaryan, Omid Poursaeed, and Vitaly Shmatikov. 2019. Differential privacy has disparate impact on model accuracy. In NeurIPS 2019.   
Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching Word Vectors with Subword Information. arXiv:1607.04606 [cs.CL]   
Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar. 2020. When is Memorization of Irrelevant Training Data Necessary for High-Accuracy Learning? arXiv preprint arXiv:2012.06421 (2020).   
Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. 2019. The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks. In USENIX Security 2019.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, Alina Oprea, and Colin Raffel. 2020. Extracting Training Data from Large Language Models. arXiv preprint arXiv:2012.07805 (2020).   
Dingfan Chen, Ning Yu, Yang Zhang, and Mario Fritz. 2020. GAN-Leaks: A Taxonomy of Membership Inference Attacks against Generative Models. In Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security (Virtual Event, USA) (CCS '20). Association for Computing Machinery, New York, NY, USA, 343–362. https://doi.org/10.1145/3372297.3417238   
Mia Xu Chen, Benjamin N. Lee, Gagan Bansal, Yuan Cao, Shuyuan Zhang, Justin Lu, Jackie Tsay, Yinan Wang, Andrew M. Dai, Zhifeng Chen, Timothy Sohn, and Yonghui Wu. 2019. Gmail Smart Compose: Real-Time Assisted Writing (KDD '19). New York, NY, USA, 2287–2295.   
Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini, and Nicolas Papernot. 2020. Label-Only Membership Inference Attacks. arXiv preprint arXiv:2007.14321 (2020).   
Cynthia Dwork. 2011. Differential privacy. Encyclopedia of Cryptography and Security (2011).   
Vitaly Feldman. 2020. Does Learning Require Memorization? A Short Tale about a Long Tail. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing (Chicago, IL, USA) (STOC 2020). New York, NY, USA, 954–959.   
Jamie Hayes, Luca Melis, George Danezis, and Emiliano De Cristofaro. 2019. LOGAN: Membership Inference Attacks Against Generative Models. Proceedings on Privacy Enhancing Technologies 2019, 1 (2019), 133 – 152.   
Xinlei He, Rui Wen, Yixin Wu, Michael Backes, Yun Shen, and Yang Zhang. 2021. Node-Level Membership Inference Attacks Against Graph Neural Networks. arXiv:2102.05429 [cs.CR]   
Yang He, Shadi Rahimian, Bernt Schiele, and Mario Fritz. 2020. Segmentations-Leak: Membership Inference Attacks and Defenses in Semantic Image Segmentation. arXiv:1912.09685 [cs.CV]   
Sorami Hisamoto, Matt Post, and Kevin Duh. 2020. Membership Inference Attacks on Sequence-to-Sequence Models: Is My Data In Your Machine Translation System? TACL 8 (2020), 49–63.   
Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long Short-term Memory. Neural computation 9 (1997), 1735–80.   
Hongsheng Hu, Zoran Salcic, Gillian Dobbie, and Xuyun Zhang. 2021. Membership Inference Attacks on Machine Learning: A Survey. arXiv:2103.07853 [cs.LG]   
Huseyin A. Inan, Osman Ramadan, Lukas Wutschitz, Daniel Jones, Victor Rühle, James Withers, and Robert Sim. 2021. Training Data Leakage Analysis in Language Models. arXiv preprint arXiv:2101.05405 (2021).   
Paul Irolla and Grégory Châtel. 2019. Demystifying the Membership Inference Attack. In 2019 12th CMI Conf. on Cybersecurity and Privacy (CMI). 1–7.   
Klas Leino and Matt Fredrikson. 2020. Stolen Memories: Leveraging Model Memorization for Calibrated White-Box Membership Inference. 29th USENIX Security Symposium (2020).   
Zheng Li and Yang Zhang. 2021. Membership Leakage in Label-Only Exposures. arXiv preprint arXiv:2007.15528 (2021).   
Yunhui Long, Vincent Bindschaedler, Lei Wang, Diyue Bu, Xiaofeng Wang, Haixu Tang, Carl A. Gunter, and Kai Chen. 2018. Understanding Membership Inferences on Well-Generalized Learning Models. arXiv preprint arXiv:1802.04889 (2018).   
Microsoft SwiftKey. [n.d.]. https://www.microsoft.com/en-us/swiftkey   
Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781 [cs.CL]   
Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan “Honza” Černocký, and Sanjeev Khudanpur. 2010. Recurrent neural network based language model. In Proceedings of the 11th Annual Conference of the International Speech Communication Association. 1045–1048.   
Fatemehsadat Mireshghallah, Huseyin A. Inan, Marcello Hasegawa, Victor Rühle, Taylor Berg-Kirkpatrick, and Robert Sim. 2021. Privacy Regularization: Joint Privacy-Utility Optimization in Language Models. arXiv preprint arXiv:2103.07567 (2021).   
Fatemehsadat Mireshghallah, Mohammadkazem Taram, Praneeth Vepakomma, Abhishek Singh, Ramesh Raskar, and Hadi Esmaeilzadeh. 2020. Privacy in Deep Learning: A Survey. CoRR abs/2004.12254 (2020). arXiv:2004.12254 https://arxiv.org/abs/2004.12254   
Sasi Kumar Murakonda and Reza Shokri. 2020. ML Privacy Meter: Aiding Regulatory Compliance by Quantifying the Privacy Risks of Machine Learning. arXiv preprint arXiv:2007.09339 (2020).

Milad Nasr, Reza Shokri, and Amir Houmansadr. 2019. Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning. In 2019 IEEE Symposium on Security and Privacy (SP). 739–753.   
Douglas Oard, William Webber, David Kirsch, and Sergey Golitsynskiy. 2015. Avocado Research Email Collection. https://catalog.ldc.upenn.edu/LDC2015T03.   
Iyiola E. Olatunji, Wolfgang Nejdl, and Megha Khosla. 2021. Membership Inference Attack on Graph Neural Networks. arXiv:2101.06570 [cs.LG]   
Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014a. GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Doha, Qatar, 1532–1543. https://doi.org/10.3115/v1/D14-1162   
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014b. GloVe: Global Vectors for Word Representation. In Empirical Methods in Natural Language Processing (EMNLP). 1532–1543. http://www.aclweb.org/anthology/D14-1162   
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language Models are Unsupervised Multitask Learners.   
Xin Rong. 2014. word2vec parameter learning explained. arXiv preprint arXiv:1411.2738 (2014).   
Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Yann Ollivier, and Herve Jegou. 2019. White-box vs Black-box: Bayes Optimal Strategies for Membership Inference. In Proceedings of the 36th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 97). Long Beach, California, USA, 5558–5567.   
Ahmed Salem, Yang Zhang, Mathias Humbert, Mario Fritz, and Michael Backes. 2018. ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models. arXiv preprint arXiv:1806.01246 (2018).   
Tobias Schnabel, Igor Labutov, David Mimno, and Thorsten Joachims. 2015. Evaluation methods for unsupervised word embeddings. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Lisbon, Portugal, 298–307. https://doi.org/10.18653/v1/D15-1036   
Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural Machine Translation of Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Berlin, Germany, 1715–1725.   
Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017. Membership Inference Attacks Against Machine Learning Models. In 2017 IEEE Symposium on Security and Privacy (SP). 3–18.   
Congzheng Song and Ananth Raghunathan. 2020. Information Leakage in Embedding Models. In Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security (Virtual Event, USA) (CCS '20). Association for Computing Machinery, New York, NY, USA, 377–390. https://doi.org/10.1145/3372297.3417270   
Congzheng Song and Vitaly Shmatikov. 2019. Auditing Data Provenance in Text-Generation Models. In KDD.   
Martin Sundermeyer, Ralf Schlüter, and Hermann Ney. 2012. LSTM Neural Networks for Language Modeling. In INTERSPEECH. 194–197.   
Harry Chandra Tanuwidjaja, Rakyong Choi, and Kwangjo Kim. 2019. A Survey on Deep Learning Techniques for Privacy-Preserving. In Machine Learning for Cyber Security, Xiaofeng Chen, Xinyi Huang, and Jun Zhang (Eds.). Springer International Publishing, Cham, 29–46.   
Stacey Truex, Ling Liu, Mehmet Emre Gursoy, Lei Yu, and Wenqi Wei. 2018. Towards Demystifying Membership Inference Attacks. arXiv preprint arXiv:1807.09173 (2018).   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, undefinedukasz Kaiser, and Illia Polosukhin. 2017. Attention is All You Need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (Long Beach, California, USA) (NIPS'17). Red Hook, NY, USA, 6000–6010.   
Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. 2018a. Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. In 31st IEEE Computer Security Foundations Symposium, CSF 2018, Oxford, United Kingdom, July 9-12, 2018. IEEE Computer Society, 268–282. https://doi.org/10.1109/CSF.2018.00027   
Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. 2018b. Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. In 2018 IEEE 31st Computer Security Foundations Symposium (CSF). 268–282.

Santiago Zanella-Béguelin, Lukas Wutschitz, Shruti Tople, Victor Rühle, Andrew Paverd, Olga Ohrimenko, Boris Köpf, and Marc Brockschmidt. 2020. Analyzing Information Leakage of Updates to Natural Language Models. In Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security. 363–375.   
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. 2017. Understanding deep learning requires rethinking generalization. ICLR (2017).