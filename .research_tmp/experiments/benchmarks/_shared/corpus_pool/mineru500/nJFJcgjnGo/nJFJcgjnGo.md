# Evaluating Robustness and Uncertainty of Graph Models Under Structural Distributional Shifts

Gleb Bazhenov\*

HSE University, Yandex Research

Denis Kuznedelev

Yandex Research, Skoltech

Andrey Malinin $^{†}$

Isomorphic Labs

Artem Babenko

Yandex Research, HSE University

Liudmila Prokhorenkova

Yandex Research

# Abstract

In reliable decision-making systems based on machine learning, models have to be robust to distributional shifts or provide the uncertainty of their predictions. In node-level problems of graph learning, distributional shifts can be especially complex since the samples are interdependent. To evaluate the performance of graph models, it is important to test them on diverse and meaningful distributional shifts. However, most graph benchmarks considering distributional shifts for node-level problems focus mainly on node features, while structural properties are also essential for graph problems. In this work, we propose a general approach for inducing diverse distributional shifts based on graph structure. We use this approach to create data splits according to several structural node properties: popularity, locality, and density. In our experiments, we thoroughly evaluate the proposed distributional shifts and show that they can be quite challenging for existing graph models. We also reveal that simple models often outperform more sophisticated methods on the considered structural shifts. Finally, our experiments provide evidence that there is a trade-off between the quality of learned representations for the base classification task under structural distributional shift and the ability to separate the nodes from different distributions using these representations.

# 1 Introduction

Recently, much effort has been put into creating decision-making systems based on machine learning for various high-risk applications, such as financial operations, medical diagnostics, autonomous driving, etc. These systems should comprise several important properties that allow users to rely on their predictions. One such property is robustness, the ability of an underlying model to cope with distributional shifts when the features of test inputs become different from those encountered in the training phase. At the same time, if a model is unable to maintain high performance on a shifted input, it should signal the potential problems by providing some measure of uncertainty. These properties are especially difficult to satisfy in the node-level prediction tasks on graph data since the elements are interdependent, and thus the distributional shifts may be even more complex than for the classic setup with i.i.d. samples.

To evaluate graph models in the node-level problems, it is important to test them under diverse, complex, and meaningful distributional shifts. Unfortunately, most existing graph datasets split the nodes into train and test parts uniformly at random. Rare exceptions include the Open Graph Benchmark (OGB) [12] that creates more challenging non-random data splits by dividing the nodes

according to some domain-specific property. For instance, in the OGB-Arxiv dataset, the papers are divided according to their publication date, which is a realistic setup. Unfortunately, not many datasets contain such meta-information that can be used to split the data. To overcome this issue, Gui et al. [10] propose the Graph OOD Benchmark (GOOD) designed specifically for evaluating graph models under distributional shifts. The authors distinguish between two types of shifts: concept and covariate. However, creating such shifts is non-trivial, while both types of shifts are typically present simultaneously in practical applications. Also, the splitting strategies in GOOD are mainly based on the node features and do not take into account the graph structure. $^{3}$

In our work, we fill this gap and propose a universal method for inducing structural distributional shifts in graph data. Our approach allows for creating diverse, complex, and meaningful node-level shifts that can be applied to any graph dataset. In particular, we introduce the split strategies that focus on such node properties as popularity, locality, and density. Our framework is flexible and allows one to easily extend it with other structural shifts or vary the fraction of nodes available for training and testing. We empirically show that the proposed distributional shifts are quite challenging for existing graph methods. In particular, the locality-based shift appears to be the most difficult in terms of the predictive performance for most considered OOD robustness methods, while the density-based shift is extremely hard for OOD detection by uncertainty estimation methods. Our experiments also reveal that simple models often outperform more sophisticated approaches on structural distributional shifts. In addition, we investigate some modifications of graph model architectures that may improve their OOD robustness or help in OOD detection on the proposed structural shifts. Our experiments provide evidence that there is a trade-off between the quality of learned representations for base classification task under structural distributional shift and the ability to separate the nodes from different distributions using these representations.

# 2 Background

# 2.1 Graph problems with distributional shifts

Several research areas in graph machine learning investigate methods for solving node-level prediction tasks under distributional shifts, and they are primarily different in what problem they seek to overcome. One such area is adversarial robustness, which requires one to construct a method that can handle artificial distributional shifts that are induced as adversarial attacks for graph models in the form of perturbed and contaminated inputs. The related approaches often focus on designing various data augmentations via introducing random or learnable noise $[30, 42, 40, 35, 22]$ .

Another research area is out-of-distribution generalization. The main task is to design a method that can handle real-world distributional shifts on graphs and maintain high predictive performance across different OOD environments. Several invariant learning and risk minimization techniques have been proposed to improve the robustness of graph models to such real-world shifts $[1, 19, 20, 38]$ .

There is also an area of uncertainty estimation, which covers various problems. In error detection, a model needs to provide the uncertainty estimates that are consistent with prediction errors, i.e., assign higher values to potential misclassifications. In the presence of distributional shifts, the uncertainty estimates can also be used for out-of-distribution detection, where a model is required to distinguish the shifted OOD data from the ID data $[24, 36, 2, 27, 18]$ .

The structural distributional shifts proposed in our paper can be used for evaluating OOD generalization, OOD detection, and error detection since they are designed to replicate the properties of realistic graph data.

# 2.2 Uncertainty estimation methods

Depending on the source of uncertainty, it is usually divided into data uncertainty, which describes the inherent noise in data due to the labeling mistakes or class overlap, and knowledge uncertainty, which accounts for the insufficient amount of information for accurate predictions when the distribution of the test data is different from the training one $[5, 24, 23]$ .

General-purpose methods The most simple approaches are standard classification models that predict the parameters of softmax distribution. For these methods, we can define the measure of uncertainty as the entropy of the predictive categorical distribution. This approach, however, does not allow us to distinguish between data and knowledge uncertainties, so it is usually inferior to other methods discussed below.

Ensembling techniques are powerful but expensive approaches that providing decent predictive performance and uncertainty estimation. The most common example is Deep Ensemble [21], which can be formulated as an empirical distribution of model parameters obtained after training several instances of the model with different random seeds for initialization. Another way to construct an ensemble is Monte Carlo Dropout [6], which is usually considered as the baseline in uncertainty estimation literature. However, it has been shown that this technique is commonly inferior to deep ensembles, as the obtained predictions are dependent and thus lack diversity. Importantly, ensembles allow for a natural decomposition of total uncertainty in data and knowledge uncertainty [23].

There is also a family of Dirichlet-based methods. Their core idea is to model the point-wise Dirichlet distribution by predicting its parameters for each input individually using some model. To get the parameters of the categorical distribution, one can normalize the parameters of the Dirichlet distribution by their sum. This normalization constant is called evidence and can be used to express the general confidence of a Dirichlet-based model. Similarly to ensembles, these methods are able to distinguish between data and knowledge uncertainty. There are numerous examples of Dirichlet-based methods, one of the first being Prior Network [24] that induces the behavior of Dirichlet distribution by contrastive learning against the OOD samples. Although this method is theoretically sound, it requires knowing the OOD samples in the training stage, which is a significant limitation. Another approach is Posterior Network [2], where the behavior of the Dirichlet distribution is controlled by Normalizing Flows [15, 13], which estimate the density in latent space and reduce the Dirichlet evidence in the regions of low density without using any OOD samples.

Graph-specific methods Recently, several uncertainty estimation methods have been designed specifically for node-level problems. For example, Graph Posterior Network $[33]$ is an extension of the Posterior Network framework discussed above. To model the Dirichlet distribution, it first encodes the node features into latent representations with a graph-agnostic model and then uses one flow per class for density estimation. Then, the Personalized Propagation scheme $[17]$ is applied to the Dirichlet parameters to incorporate the network effects. Another Dirichlet-based method is Graph-Kernel Dirichlet Estimation $[41]$ . In contrast to Posterior Networks, its main property is a compound training objective, which is focused on optimizing the node classification performance and inducing the behavior of the Dirichlet distribution. The former is achieved via knowledge distillation from a teacher GNN, while the latter is performed as a regularisation against the prior Dirichlet distribution computed via graph kernel estimation. However, as shown by Stadler et al. $[33]$ , this approach is inferior to Graph Posterior Network while having a more complex training procedure and larger computational complexity.

# 2.3 Methods for improving robustness

Improving OOD robustness can be approached from different perspectives, which, however, share the same idea of learning the representations that are invariant to undesirable changes of the input distribution. For instance, the main claim in domain adaptation is that to achieve an effective domain transfer and improve the robustness to distributional shift, the predictions should depend on the features that can not discriminate between the source and target domains. Regarding the branch of invariant learning, these methods decompose the input space into different environments and focus on constructing robust representations that are insensitive to their change.

General-purpose methods A classic approach for domain adaptation is Domain-Adversarial Neural Network [7] that promotes the emergence of features that are discriminative for the main learning task on the source domain but do not allow to detect the distributional shift on other domains. This is achieved by jointly optimizing the underlying representations as well as two predictors operating on them: the main label predictor that solves the base classification task and the domain classifier that discriminates between the source and the target domains during training. Another simple technique in the class of unsupervised domain adaptation methods is Deep Correlation Alignment

[34], which trains to align the second-order statistics of the source and target distributions that are produced by the activation layers of a deep neural model that solves the base classification task.

A representative approach in invariant learning is Invariant Risk Minimization $[1]$ , which searches for data representations providing decent performance across all environments, while the optimal classifier on top of these representations matches for all environments. Another method is Risk Extrapolation $[20]$ , which targets both robustness to covariate shifts and invariant predictions. In particular, it targets the forms of distributional shifts having the largest impact on performance in training domains.

Graph-specific methods More recently, several graph-specific methods for improving OOD robustness have been proposed. One of them is an invariant learning technique Explore-to-Extrapolate Risk Minimization [38], which leverages multiple context explorers that are specified by graph structure editors and adversarially trained to maximize the variance of risks across different created environments. There is also a simple data augmentation technique Mixup [39] that trains neural models on convex combinations of pairs of samples and their corresponding labels. However, devising such a method for solving the node-level problems is not straightforward, as the inputs are connected to each other. Based on this technique, Wang et al. [37] have designed its adaptation for graph data: instead of training on combinations of the initial node features, this method exploits the intermediate representations of nodes and their neighbors that are produced by graph convolutions.

# 3 Structural distributional shifts

# 3.1 General approach

As discussed above, existing datasets for evaluating the robustness and uncertainty of node-level problems mainly focus on feature-based distributional shifts. Here, we propose a universal approach that produces non-trivial yet reasonable structural distributional shifts. For this purpose, we introduce a node-level graph characteristic $\sigma_{i}$ and compute it for every node $i \in V$ . We sort all nodes in ascending order of $\sigma_{i}$ — those with the smallest values of $\sigma_{i}$ are considered to be ID, while the remaining ones are OOD. As a result, we obtain a graph-based distributional shift where ID and OOD nodes have different structural properties. The type of shift depends on the choice of $\sigma_{i}$ , and several possible options are described in Section 3.2 below.

We further split the ID nodes uniformly at random into the following parts:

- Train contains nodes $\mathcal{V}_{\text{train}}$ that are used for regular training of models and represent the only observations that take part in gradient computation.   
- Valid-In enables us to monitor the best model during the training stage by computing the validation loss for nodes $\mathcal{V}_{\mathrm{valid - in}}$ and choose the best checkpoint.   
- Test-In is used for testing on the remaining ID nodes $\mathcal{V}_{\mathrm{test - in}}$ and represents the simplest setup that requires a model to reproduce in-distribution dependencies.

The remaining OOD nodes are split into Valid-Out and Test-Out subsets based on their $\sigma_{i}$ :

- Test-Out is used to evaluate the robustness of models to distributional shifts. It consists of nodes with the largest values of $\sigma_{i}$ and thus represents the most shifted part $\mathcal{V}_{\mathrm{test - out}}$ .   
- Valid-Out contains OOD nodes with smaller values of $\sigma_{i}$ and thus is less shifted than Test-Out. This subset $\mathcal{V}_{\mathrm{valid - out}}$ can be used for monitoring the model performance on a shifted distribution. Our experiments assume a more challenging setup when such shifted data is unavailable during training. However, the presence of this subset allows us to further separate $\mathcal{V}_{\mathrm{test - out}}$ from $\mathcal{V}_{\mathrm{train}}$ — the larger $\mathcal{V}_{\mathrm{valid - out}}$ we consider, the more significant distributional shift is created between the train and OOD test nodes.

Our general framework is quite flexible and allows one to easily vary the size of the training part and the type of distributional shift. Let us now discuss some particular shifts that we propose in this paper.

![](images/328b1e7250cee0f46abe5a9e34a35f812016f20907431e80bcfccbdef34d4e81.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | A |
| 0.2 | 0.85 | A |
| 0.3 | 0.75 | A |
| 0.4 | 0.65 | A |
| 0.5 | 0.55 | A |
| 0.6 | 0.45 | A |
| 0.7 | 0.35 | A |
| 0.8 | 0.25 | A |
| 0.9 | 0.15 | A |
| 0.15 | 0.88 | B |
| 0.25 | 0.78 | B |
| 0.35 | 0.68 | B |
| 0.45 | 0.58 | B |
| 0.55 | 0.48 | B |
| 0.65 | 0.38 | B |
| 0.75 | 0.28 | B |
| 0.85 | 0.18 | B |
| 0.95 | 0.08 | B |
</details>

![](images/8fe721e48aaf11759e4f4f88b995cbe1591654c943b28bbaf29710c600f5b9ad.jpg)

<details>
<summary>scatter</summary>

| x | y | category |
| --- | --- | --- |
| (data not extractable from image) | (data not extractable from image) | Red dots (locally) |
</details>

![](images/79ba14dbd38beb340ff6185b611008f85a8345eca23cc4dce7f8bc8c26f2d211.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (various) | (various) | Red |
| (various) | (various) | Blue |
</details>

Figure 1: Visualization of structural shifts for AmazonPhoto dataset: ID is blue, OOD is red.

# 3.2 Proposed distributional shifts

To define our data splits, we need to choose a node property $\sigma_{i}$ as a splitting factor. We consider diverse graph characteristics covering various distributional shifts that may occur in practice. In a standard non-graph ML setup, shifts typically happen only in the feature space (or, more generally, the joint distribution of features and targets may shift). However, in graph learning tasks, there can be shifts specifically related to the graph structure. We discuss some representative examples below.

Popularity-based The first strategy represents a possible bias towards popularity. In some applications, it is natural to expect the training set to consist of more popular items. For instance, in the web search, the importance of pages in the internet graph can be measured via PageRank [29]. For this application, the labeling of pages should start with important ones since they are visited more often. Similar situations may happen for social networks, where it is natural to start labeling with the most influential users, or citation networks, where the most cited papers should be labeled first. However, when applying a graph model, it is essential to make accurate predictions on less popular items. Motivated by that, we introduce a popularity-based split based on PageRank. The vector of PageRank values $\pi_{i}$ describes the stationary distribution of a random walk with restarts, for which the transition matrix is defined as the normalized adjacency matrix $AD^{-1}$ and the probability of restart is $\alpha$ :

$$
\boldsymbol {\pi} = (1 - \alpha) \mathbf {A D} ^ {- 1} \boldsymbol {\pi} + \alpha \boldsymbol {p}. \tag {1}
$$

The vector of restart probabilities p is called a personalization vector, and $p_{i} = 1/n$ by default, which describes the uniform distribution over nodes.

To construct a popularity-based split, we use PageRank (PR) as a measure of node importance. Thus, we compute PR for every node i and set $\sigma_{i} = -\pi_{i}$ , which means that the nodes with smaller PR values (i.e., less important ones) belong to the OOD subsets. Note that instead of PR, one can potentially use any other measure of node importance, e.g., node degree, or betweenness centrality.

Figure 1a illustrates that the proposed shift separates the most important nodes that belong to the cores of large clusters and the structural periphery, which consists of less important nodes in terms of their PR values. We also observe that such popularity-based split agrees well with some natural distributional shifts. For instance, Figure 11a in Appendix shows the distribution of PageRank in train and test parts of the OGB-Arxiv dataset [12]. Here, the papers are split according to their publication date. Thus, older papers naturally have more citations and, therefore, larger PageRank. Our proposed split allows one to mimic this behavior for datasets without timestamps available. We refer to Appendix A for more details and additional analysis.

Locality-based Our next strategy is focused on a potential bias towards locality, which may happen when labeling is performed by exploring the graph starting from some node. For instance, in web search applications, a crawler has to explore the web graph following the links. Similarly, the information about the users of a social network can usually be obtained via an API, and new users are discovered following the friends of known users. To model such a situation, one could divide nodes based on the shortest path distance to a given node. However, graph distances are discrete, and the number of nodes at a certain distance may grow exponentially with distance. Thus, such an approach

does not provide us with the desired flexibility in varying the size of the train part. Instead, we use the concept of Personalized PageRank (PPR) [29] to define a local neighborhood of a node. PPR is the stationary distribution of a random walk that always restarts from some fixed node j. Thus, the personalization vector p in (1) is set to the one-hot-encoding of j. The associated distributional shift naturally captures locality since a random walk always restarts from the same node.

For our experiments, we select the node j with the highest PR score as a restarting one. Then, we compute the PPR values $\pi_{i}$ for every node i and define the measure $\sigma_{i} = -\pi_{i}$ . The nodes with high PPR, which belong to the ID part, are expected to be close to the restarting node, while far away nodes go to the OOD subset. Figure 1b confirms that the locality is indeed preserved, as the ID part consists of one compact region around the restarting node. Thus, the OOD subset includes periphery nodes as well as some nodes that were previously marked as important in the PR-based split but are far away from the restarting node. Our analysis in Appendix A also provides evidence for this behavior: the PPR-based split strongly affects the distribution of pairwise distances within the ID/OOD parts as the locality bias of the ID part makes the OOD nodes more distant from each other.

While locality-based distributional shifts are natural, we are unaware of publicly available benchmarks focusing on such shifts. We believe that our approach will be helpful for evaluating the robustness of GNNs under such shifts. Our empirical results in Section 4 demonstrate that locality-based splits are the most challenging for graph models and thus may require special attention.

Density-based The next distributional shift we propose is based on density. One of the most simple node characteristics that describe the local density in a graph is the local clustering coefficient. Considering some node i, let $d_{i}$ be its degree and $\gamma_{i}$ be the number of edges connecting the neighbors of i. Then, the local clustering coefficient is defined as the edge density within the one-hop neighborhood:

$$
c _ {i} = \frac {2 \gamma_ {i}}{d _ {i} (d _ {i} - 1)} \tag {2}
$$

For our experiments, we consider nodes with the highest clustering coefficient to be ID, which implies that $\sigma_{i} = -c_{i}$ .

This structural property might be particularly interesting for inducing distributional shifts since it is defined through the number of triangles, the substructures that most standard graph neural networks are unable to distinguish and count $[4]$ . Thus, it is interesting to know how changes in the clustering coefficient affect the predictive performance and uncertainty estimation.

Figure 1c visualizes the density-based split. We see that the OOD part includes both the high-degree central nodes and the periphery nodes of degree one. Indeed, the nodes of degree one naturally have zero clustering coefficient. On the other hand, for the high-degree nodes, the number of edges between their neighbors usually grows slower than quadratically. Thus, such nodes tend to have a vanishing clustering coefficient.

Finally, we note that the existing datasets with realistic splits may often have the local clustering coefficient shifted between the train and test parts. Figures 12c and 13c in Appendix show this for two OGB datasets. Depending on the dataset, the train subset may be biased towards the nodes with either larger (Figure 13c) or smaller (Figure 12c) clustering. In our experiments, we focus on the former scenario.

# 4 Experimental setup

Datasets While our approach can potentially be applied to any node prediction dataset, for our experiments, we pick the following seven homophilous datasets that are commonly used in the literature: three citation networks, including CoraML, CiteSeer [26, 9, 8, 31], and PubMed [28], two co-authorship graphs — CoauthorPhysics and CoauthorCS [32], and two co-purchase datasets — AmazonPhoto and AmazonComputer [25, 32]. Moreover, we consider OGB-Products, a large-scale dataset from the OGB benchmark. Some of the methods considered in our work are not able to process such a large dataset, so we provide only the analysis of structural shifts on this dataset and do not use it for comparing different methods.

For any distributional shift, we split each graph dataset as follows. The half of nodes with the smallest values of $\sigma_{i}$ are considered to be ID and split into Train, Valid-In, and Test-In uniformly at

random in proportion 30% : 10% : 10%. The second half contains the remaining OOD nodes and is split into Valid-Out and Test-Out in the ascending order of $\sigma_{i}$ in proportion 10% : 40%. Thus, in our base setup, the ID to OOD split ratio is 50% : 50%. We have also conducted experiments with other split ratios that involve smaller sizes of OOD subsets, see Appendix B for the details.

Models In our experiments, we apply the proposed benchmark to evaluate the OOD robustness and uncertainty of various graph models. In particular, we consider the following methods for improving the OOD generalisation in the node classification task:

- ERM is the Empirical Risk Minimization technique that trains a simple GNN model by optimizing a standard classification loss;   
- DANN is an instance of Domain Adversarial Network [7] that trains a regular and a domain classifiers to make features indistinguishable across different domains;   
- CORAL is the Deep Correlation Alignment [34] technique that encourages the representations of nodes in different domains to be similar;   
- EERM is the Explore-to-Extrapolate Risk Minimization [38] method based on graph structure editors that creates virtual environments during training;   
- Mixup is an implementation of Mixup from [37] and represents a simple data augmentation technique adapted for the graph learning problems;   
- DE represents a Deep Ensemble [21] of graph models, a strong but expensive method for improving the predictive performance;

In context of OOD detection, we consider the following uncertainty estimation methods:

- SE represents a simple GNN model that is used in the ERM method, for which the measure of uncertainty is Softmax Entropy (i.e., the entropy of predictive distribution);   
- GPN is an implementation of the Graph Posterior Network [33] method for the node-level uncertainty estimation;   
- NatPN is an instance of Natural Posterior Network [3] in which the encoder has the same architecture as in the SE method;   
- DE represents a Deep Ensemble [21] of graph models, which allows to separate the knowledge uncertainty that is used for OOD detection;   
- GPE and NatPE represent the Bayesian Combinations of GPN and NatPN, an approach to construct an ensemble of Dirichlet models [3].

The training details are described in Appendix F. For experiments with the considered OOD robustness methods, including DANN, CORAL, EERM, and Mixup, we use the experimental framework from the GOOD benchmark, $^{4}$ whereas the remaining methods are implemented in our custom experimental framework and can be found in our repository. $^{5}$

Prediction tasks & evaluation metrics To evaluate OOD robustness in the node classification problem, we exploit standard Accuracy. Further, to assess the quality of uncertainty estimates, we treat the OOD detection problem as a binary classification with positive events corresponding to the observations from the OOD subset and use AUROC to measure performance.

# 5 Empirical study

In this section, we show how the proposed approach to creating distributional shifts can be used for evaluating the robustness and uncertainty of graph models. In particular, we compare the types of distributional shifts introduced above and discuss which of them are more challenging. We also discuss how they affect the predictive performance of OOD robustness methods as well as the ability for OOD detection of uncertainty estimation methods.

Table 1: Comparison of structural distributional shifts in terms of OOD robustness and OOD detection. We report the drop in predictive performance of the ERM method measured by Accuracy (left) and the quality of uncertainty estimates of the SE method measured by AUROC (right). 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>AmazonComputer</td><td>-10.01%</td><td>-21.95%</td><td>-7.91%</td></tr><tr><td>AmazonPhoto</td><td>-8.54%</td><td>-30.93%</td><td>-3.58%</td></tr><tr><td>CoauthorCS</td><td>-3.13%</td><td>-1.22%</td><td>-4.86%</td></tr><tr><td>CoauthorPhysics</td><td>-3.42%</td><td>-4.75%</td><td>-1.41%</td></tr><tr><td>CoraML</td><td>-3.94%</td><td>-14.61%</td><td>-17.09%</td></tr><tr><td>CiteSeer</td><td>-0.02%</td><td>-26.51%</td><td>-8.39%</td></tr><tr><td>PubMed</td><td>-3.23%</td><td>-5.71%</td><td>-0.78%</td></tr><tr><td>OGB-Products</td><td>-2.86%</td><td>-2.83%</td><td>-0.12%</td></tr><tr><td>Average</td><td>-4.39%</td><td>-13.56%</td><td>-5.52%</td></tr></table>

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>AmazonComputer</td><td>88.52</td><td>86.48</td><td>44.24</td></tr><tr><td>AmazonPhoto</td><td>92.05</td><td>93.29</td><td>41.08</td></tr><tr><td>CoauthorCS</td><td>83.25</td><td>85.74</td><td>50.91</td></tr><tr><td>CoauthorPhysics</td><td>86.60</td><td>87.73</td><td>37.50</td></tr><tr><td>CoraML</td><td>75.67</td><td>87.13</td><td>81.55</td></tr><tr><td>CiteSeer</td><td>68.01</td><td>89.89</td><td>66.90</td></tr><tr><td>PubMed</td><td>68.60</td><td>66.34</td><td>58.60</td></tr><tr><td>OGB-Products</td><td>88.50</td><td>88.56</td><td>36.00</td></tr><tr><td>Average</td><td>81.40</td><td>85.65</td><td>52.10</td></tr></table>

# 5.1 Analysis of structural distributional shifts

In this section, we analyze and compare the proposed structural distributional shifts.

OOD robustness To investigate how the proposed shifts affect the predictive performance of graph models, we take the most simple ERM method and report the drop in Accuracy between the ID and OOD test subsets in Table 1 (left). It can be seen that the node classification results on the considered datasets are consistently lower when measured on the OOD part, and this drop can reach tens of percent in some cases. The most significant decrease in performance is observed on the locality-based splits, where it reaches 14% on average and more than 30% in the worst case. This fact matches our intuition about how training on local regions of graphs may prevent OOD generalization and create a great challenge for improving OOD robustness. Although the density-based shift does not appear to be as difficult, it is still more challenging than the popularity-based shift, leading to performance drops of 5.5% on average and 17% in the worst case.

OOD detection To analyze the ability of graph models to detect distributional shifts by providing higher uncertainty on the shifted inputs, we report the performance of the most simple SE method for each proposed distributional shift and graph dataset in Table 1 (right). It can be seen that the popularity-based and locality-based shifts can be effectively detected by this method, which is proved by the average performance metrics. In particular, the AUROC values may vary from 68 to 92 points on the popularity-based splits and approximately in the same range for the locality-based splits. Regarding the density-based shifts, one can see that the OOD detection performance is almost the same as for random predictions on average. Only for the citation networks the AUROC exceeds 58 points, reaching a peak of 81 points. This is consistent with the previous works showing that standard graph neural networks are unable to count substructures such as triangles. In our case, this leads to graph models failing to detect changes in density measured as the number of triangles around the central node.

Thus, the popularity-based shift appears to be the simplest for OOD detection, while the density-based is the most difficult. This clearly shows how our approach to creating data splits allows one to vary the complexity of distributional shifts using different structural properties as splitting factors.

# 5.2 Comparison of existing methods

In this section, we compare several existing methods for improving OOD robustness and detecting OOD inputs on the proposed structural shifts. To concisely illustrate the overall performance of the models, we first rank them according to a given performance measure on a particular dataset and then average the results over the datasets. For detailed results of experiments on each graph dataset separately, please refer to Appendix G.

OOD robustness For each model, we measure both the absolute values of Accuracy and the drop in this metric between the ID and OOD subsets in Table 2 (left). It can be seen that a simple data augmentation technique Mixup often shows the best performance. Only on the density-based shift, expensive DE outperforms it on average when tested on the OOD subset. This proves that data

Table 2: Comparison of several graph methods for improving the OOD robustness (left) and detecting the OOD inputs by means of uncertainty estimation (right). For each task, we report the method ranks averaged across different graph datasets (lower is better). 

<table><tr><td rowspan="2"></td><td colspan="2">Popularity</td><td colspan="2">Locality</td><td colspan="2">Density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td>4.0</td><td>4.0</td><td>3.3</td><td>4.1</td><td>3.9</td><td>4.0</td></tr><tr><td>Mixup</td><td>1.4</td><td>2.1</td><td>1.4</td><td>2.4</td><td>1.9</td><td>3.3</td></tr><tr><td>EERM</td><td>3.6</td><td>3.9</td><td>4.4</td><td>3.3</td><td>5.0</td><td>4.3</td></tr><tr><td>DANN</td><td>4.3</td><td>4.3</td><td>5.0</td><td>4.1</td><td>3.0</td><td>3.6</td></tr><tr><td>CORAL</td><td>4.7</td><td>4.1</td><td>4.3</td><td>4.7</td><td>4.1</td><td>3.9</td></tr><tr><td>DE</td><td>3.0</td><td>2.6</td><td>2.6</td><td>2.3</td><td>3.1</td><td>2.0</td></tr></table>

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>1.4</td><td>2.1</td><td>4.0</td></tr><tr><td>GPN</td><td>3.3</td><td>3.9</td><td>4.3</td></tr><tr><td>NatPN</td><td>5.3</td><td>4.1</td><td>2.9</td></tr><tr><td>DE</td><td>2.1</td><td>1.1</td><td>2.7</td></tr><tr><td>GPE</td><td>3.1</td><td>4.3</td><td>3.4</td></tr><tr><td>NatPE</td><td>5.7</td><td>5.4</td><td>3.7</td></tr></table>

Table 3: Comparison of the proposed architecture modifications that are used in the ERM method for OOD robustness (left) and SE method for OOD detection (right). For each task, we report the win/tie/loss counts across graph datasets for the modified GNN architecture against the base one. 

<table><tr><td rowspan="2"></td><td colspan="2">Popularity</td><td colspan="2">Locality</td><td colspan="2">Density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM + mod</td><td>4/2/1</td><td>5/2/0</td><td>4/2/1</td><td>3/2/2</td><td>4/2/1</td><td>5/2/0</td></tr></table>

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE + mod</td><td>0/0/7</td><td>0/1/6</td><td>4/0/3</td></tr></table>

augmentation techniques are beneficial in practice, as they prevent overfitting to the ID structural patterns and improve the OOD robustness. Regarding other OOD robustness methods, a graph-specific method EERM outperforms DANN and CORAL on the popularity-based and locality-based shifts. However, these domain adaptation methods are superior to EERM on the density-based shifts, providing better predictive performance on average for both ID and OOD subsets.

In conclusion, we reveal that the most sophisticated methods that generate virtual environments and predict the underlying domains for improving the OOD robustness may often be outperformed by simpler methods, such as data augmentation.

OOD detection Comparing the quality of uncertainty estimates in Table 2 (right), one can observe that the methods based on the entropy of predictive distribution usually outperform Dirichlet methods. In particular, a natural distinction of knowledge uncertainty in DE enables it to produce uncertainty estimates that are the most consistent with OOD inputs on average, especially when applied to the locality-based and density-based shifts. In general, GPN and the combination of its instances GPE provide higher OOD detection performance than their counterparts based on NatPN when tested on the popularity-based and locality-based splits.

# 5.3 Influence of graph architecture improvements

In this section, we consider several adjustments for the base GNN architecture of such methods as ERM, which is used to evaluate OOD robustness, and SE, which provides the uncertainty estimates for OOD detection. In particular, we reduce the number of graph convolutional layers from 3 to 2, replacing the first one with a pre-processing step based on MLP, apply the skip-connections between graph convolutional layers, and replace the GCN [16] graph convolution with SAGE [11].

These changes are aimed at relaxing the restrictions on the information exchange between a central node and its neighbors and providing more independence in processing the node representations across neural layers. Such a modification is expected to help the GNN model to learn structural patterns that could be transferred to the shifted OOD subset more successfully. Further, we investigate how these changes in the model architecture affect the predictive performance and the quality of uncertainty estimates of the corresponding methods when tested on the proposed structural distributional shifts.

For this, we use the win/tie/loss counts that reflect how many times the modified architecture has outperformed, got a statistically insignificant difference, or lost to the corresponding method, respectively. As can be seen from Table 3 (left), the ERM method supplied with the proposed modifications usually outperforms the baseline architecture both on ID and OOD, which is proved by high win counts. However, as can be seen from Table 3 (right), the same modification in the corresponding SE method leads to consistent performance drops, which is reflected in high loss

counts. It means that, when higher predictive performance is reached on the shifted subset, it becomes more difficult to detect the inputs from this subset as OOD since they appear to be less distinguishable by a GNN model in the context of the base node classification problem. Our observation is very similar to what is required from Invariant Learning techniques, which try to produce node representations invariant to different domains or environments. This may serve as evidence that there is a trade-off between the quality of learned representations for solving the target node classification task under structural distributional shift and the ability to separate the nodes from different distributions based on these representations.

# 6 Conclusion

In this work, we propose and analyze structural distributional shifts for evaluating robustness and uncertainty in node-level graph problems. Our approach allows one to create realistic, challenging, and diverse distributional shifts for an arbitrary graph dataset. In our experiments, we evaluate the proposed structural shifts and show that they can be quite challenging for existing graph models. We also find that simple models often outperform more sophisticated methods on these challenging shifts. Moreover, by applying various modifications for graph model architectures, we show that there is a trade-off between the quality of learned representations for the target classification task under structural distributional shift and the ability to detect the shift using these representations.

Limitations While our methods of creating structural shifts are motivated by real distributional shifts that arise in practice, they are synthetically generated, whereas, for particular applications, natural distributional shifts would be preferable. However, our goal is to address the situations when such natural shifts are unavailable. Thus, we have chosen an approach universally applied to any dataset. Importantly, graph structure is the only common modality of different graph datasets that can be exploited in the same manner to model diverse and complex distributional shifts.

Broader impact Considering the broader implications of our work, we assume that the proposed approach for evaluating robustness and uncertainty of graph models will support the development of more reliable systems based on machine learning. By testing on the presented structural shifts, it should be easier to detect various biases against under-represented groups that may have a negative impact on the resulting performance and interfere with fair decision-making.

Future work In the future, two key areas can be explored based on our work. Firstly, there is a need to develop principled solutions for improving robustness and uncertainty estimation on the proposed structural shifts. Our new approach can assist in achieving this objective by providing an instrument for testing and evaluating such solutions. Additionally, there is a need to create new graph benchmarks that accurately reflect the properties of data observed in real-world applications. This should involve replacing synthetic shifts with realistic ones. By doing so, we may capture the challenges and complexities that might be faced in practice, thereby enabling the development of more effective and applicable graph models.

# References

[1] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.   
[2] B. Charpentier, D. Zügner, and S. Günnemann. Posterior network: Uncertainty estimation without ood samples via density-based pseudo-counts. Advances in Neural Information Processing Systems, 33:1356–1367, 2020.   
[3] B. Charpentier, O. Borchert, D. Zügner, S. Geisler, and S. Günnemann. Natural posterior network: Deep bayesian predictive uncertainty for exponential family distributions. In International Conference on Learning Representations, 2022.   
[4] Z. Chen, L. Chen, S. Villar, and J. Bruna. Can graph neural networks count substructures? Advances in Neural Information Processing Systems, 33:10383–10395, 2020.   
[5] Y. Gal. Uncertainty in deep learning. University of Cambridge, Cambridge, 2016.

[6] Y. Gal and Z. Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning, pages 1050–1059. PMLR, 2016.   
[7] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky. Domain-adversarial training of neural networks. The journal of machine learning research, 17(1):2096–2030, 2016.   
[8] L. Getoor. Link-based classification. In Advanced methods for knowledge discovery from complex data, pages 189–207. Springer, 2005.   
[9] C. L. Giles, K. D. Bollacker, and S. Lawrence. Citeseer: An automatic citation indexing system. In Proceedings of the third ACM conference on Digital libraries, pages 89–98, 1998.   
[10] S. Gui, X. Li, L. Wang, and S. Ji. GOOD: A Graph Out-of-Distribution Benchmark. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022.   
[11] W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. Advances in Neural Information Processing Systems, 30, 2017.   
[12] W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec. Open graph benchmark: Datasets for machine learning on graphs. Advances in Neural Information Processing Systems, 33:22118–22133, 2020.   
[13] C.-W. Huang, D. Krueger, A. Lacoste, and A. Courville. Neural autoregressive flows. In International Conference on Machine Learning, pages 2078–2087. PMLR, 2018.   
[14] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.   
[15] D. P. Kingma, T. Salimans, R. Jozefowicz, X. Chen, I. Sutskever, and M. Welling. Improved variational inference with inverse autoregressive flow. Advances in Neural Information Processing Systems, 29, 2016.   
[16] T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.   
[17] J. Klicpera, A. Bojchevski, and S. Günnemann. Predict then propagate: Graph neural networks meet personalized pagerank. In International Conference on Learning Representations, 2019.   
[18] N. Kotelevskii, A. Artemenkov, K. Fedyanin, F. Noskov, A. Fishkov, A. Petiushko, and M. Panov. Nonparametric Uncertainty Quantification for Deterministic Neural Networks. Advances in Neural Information Processing Systems, 35, 2022.   
[19] M. Koyama and S. Yamaguchi. When is invariance useful in an out-of-distribution generalization problem? arXiv preprint arXiv:2008.01883, 2020.   
[20] D. Krueger, E. Caballero, J.-H. Jacobsen, A. Zhang, J. Binas, D. Zhang, R. Le Priol, and A. Courville. Out-of-Distribution Generalization via Risk Extrapolation (REx). In International Conference on Machine Learning, pages 5815–5826. PMLR, 2021.   
[21] B. Lakshminarayanan, A. Pritzel, and C. Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems, 30, 2017.   
[22] J. Ma, S. Ding, and Q. Mei. Towards more practical adversarial attacks on graph neural networks. Advances in Neural Information Processing Systems, 33:4756–4766, 2020.   
[23] A. Malinin. Uncertainty estimation in deep learning with application to spoken language assessment. PhD thesis, University of Cambridge, 2019.   
[24] A. Malinin and M. Gales. Predictive uncertainty estimation via prior networks. Advances in Neural Information Processing Systems, 31, 2018.

[25] J. McAuley, C. Targett, Q. Shi, and A. Van Den Hengel. Image-based recommendations on styles and substitutes. In Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval, pages 43–52, 2015.   
[26] A. K. McCallum, K. Nigam, J. Rennie, and K. Seymore. Automating the construction of internet portals with machine learning. Information Retrieval, 3(2):127–163, 2000.   
[27] J. Mukhoti, A. Kirsch, J. van Amersfoort, P. H. Torr, and Y. Gal. Deep deterministic uncertainty: A simple baseline. arXiv e-prints, pages arXiv-2102, 2021.   
[28] G. Namata, B. London, L. Getoor, B. Huang, and U. Edu. Query-driven active surveying for collective classification. In 10th International Workshop on Mining and Learning with Graphs, volume 8, page 1, 2012.   
[29] L. Page, S. Brin, R. Motwani, and T. Winograd. The pagerank citation ranking: Bringing order to the web. Technical report, Stanford InfoLab, 1999.   
[30] Y. Rong, W. Huang, T. Xu, and J. Huang. DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. In International Conference on Learning Representations, 2020.   
[31] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher, and T. Eliassi-Rad. Collective classification in network data. AI magazine, 29(3):93–93, 2008.   
[32] O. Shchur, M. Mumme, A. Bojchevski, and S. Günnemann. Pitfalls of graph neural network evaluation. Relational Representation Learning Workshop, NeurIPS 2018, 2018.   
[33] M. Stadler, B. Charpentier, S. Geisler, D. Zügner, and S. Günnemann. Graph posterior network: Bayesian predictive uncertainty for node classification. Advances in Neural Information Processing Systems, 34:18033–18048, 2021.   
[34] B. Sun and K. Saenko. Deep coral: Correlation alignment for deep domain adaptation. In Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14, pages 443–450. Springer, 2016.   
[35] Y. Sun, S. Wang, X. Tang, T.-Y. Hsieh, and V. Honavar. Adversarial attacks on graph neural networks via node injections: A hierarchical reinforcement learning approach. In Proceedings of the Web Conference 2020, pages 673–683, 2020.   
[36] B. Wang, J. Lu, Z. Yan, H. Luo, T. Li, Y. Zheng, and G. Zhang. Deep uncertainty quantification: A machine learning approach for weather forecasting. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2087–2095, 2019.   
[37] Y. Wang, W. Wang, Y. Liang, Y. Cai, and B. Hooi. Mixup for node and graph classification. In Proceedings of the Web Conference 2021, pages 3663-3674, 2021.   
[38] Q. Wu, H. Zhang, J. Yan, and D. Wipf. Handling Distribution Shifts on Graphs: An Invariance Perspective. In International Conference on Learning Representations, 2022.   
[39] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations, 2018.   
[40] X. Zhang and M. Zitnik. GNNGuard: Defending Graph Neural Networks Against Adversarial Attacks. Advances in Neural Information Processing Systems, 33:9263–9275, 2020.   
[41] X. Zhao, F. Chen, S. Hu, and J.-H. Cho. Uncertainty Aware Semi-Supervised Learning on Graph Data. Advances in Neural Information Processing Systems, 33:12827-12836, 2020.   
[42] D. Zügner and S. Günnemann. Adversarial attacks on graph neural networks via meta learning. In International Conference on Learning Representations, 2019.

# A Properties of distributional shifts

This section provides a detailed analysis and comparison of the proposed distributional shifts. For this purpose, we consider three representative real-world datasets AmazonComputer, CoauthorCS, and CoraML, and discuss how different distributional shifts affect the basic properties of data: class balance, degree distribution, and graph distances between nodes within ID and OOD subsets.

![](images/5c5460941948228b9d3a1b5a01ece1bc281927920d2416cbfa6a32ebe12da20d.jpg)

<details>
<summary>bar</summary>

| class | ID    | OOD   |
|-------|-------|-------|
| 1     | 0.03  | 0.03  |
| 2     | 0.13  | 0.18  |
| 3     | 0.13  | 0.07  |
| 4     | 0.04  | 0.04  |
| 5     | 0.36  | 0.39  |
| 6     | 0.03  | 0.01  |
| 7     | 0.04  | 0.03  |
| 8     | 0.06  | 0.06  |
| 9     | 0.15  | 0.16  |
| 10    | 0.02  | 0.02  |
</details>

(a) Popularity

![](images/d9d6a3b192721f4648d8daab3aeff3fa8ef95e38616351f7c15589624dd11544.jpg)

<details>
<summary>bar</summary>

| class | ID | OOD |
|---|---|---|
| 1 | 0.03 | 0.04 |
| 2 | 0.12 | 0.18 |
| 3 | 0.05 | 0.16 |
| 4 | 0.05 | 0.03 |
| 5 | 0.56 | 0.21 |
| 6 | 0.04 | 0.04 |
| 7 | 0.04 | 0.03 |
| 8 | 0.13 | 0.17 |
| 9 | 0.14 | 0.17 |
| 10 | 0.03 | 0.03 |
</details>

(b) Locality

![](images/c6cc14542e09bb73a377ddea2062d9518a4d8d1852ddc4e69ec1e499f6d08bbe.jpg)

<details>
<summary>bar</summary>

| class | ID | OOD |
|---|---|---|
| 1 | 0.03 | 0.03 |
| 2 | 0.15 | 0.16 |
| 3 | 0.07 | 0.14 |
| 4 | 0.03 | 0.05 |
| 5 | 0.43 | 0.32 |
| 6 | 0.02 | 0.02 |
| 7 | 0.03 | 0.04 |
| 8 | 0.05 | 0.07 |
| 9 | 0.16 | 0.15 |
| 10 | 0.03 | 0.01 |
</details>

(c) Density

Figure 2: Class balance for AmazonComputer dataset across different types of shifts.   
![](images/1a4e7f50db041461235938b3e0a0821fd371734419ac101372e82de4b3fe00be.jpg)

<details>
<summary>bar</summary>

| class | ID | OOD |
|---|---|---|
| 1 | 0.035 | 0.045 |
| 2 | 0.115 | 0.025 |
| 3 | 0.08 | 0.07 |
| 4 | 0.125 | 0.11 |
| 5 | 0.02 | 0.025 |
| 6 | 0.055 | 0.045 |
| 7 | 0.045 | 0.04 |
| 8 | 0.09 | 0.075 |
| 9 | 0.025 | 0.125 |
| 10 | 0.22 | 0.235 |
| 11 | 0.05 | 0.045 |
</details>

(a) Popularity

![](images/14dcb1c755d3b3e4c2375de417f512e1109c7223a0a93c9ef91a08e2164ba698.jpg)

<details>
<summary>bar</summary>

| class | ID    | OOD   |
|-------|-------|-------|
| 1     | 0.02  | 0.06  |
| 2     | 0.03  | 0.02  |
| 3     | 0.12  | 0.11  |
| 4     | 0.02  | 0.04  |
| 5     | 0.05  | 0.10  |
| 6     | 0.23  | 0.02  |
| 7     | 0.03  | 0.03  |
| 8     | 0.06  | 0.04  |
| 9     | 0.05  | 0.03  |
| 10    | 0.04  | 0.13  |
| 11    | 0.21  | 0.02  |
| 12    | 0.09  | 0.36  |
| 13    | 0.06  | 0.03  |
</details>

(b) Locality

![](images/a4171f0180344035c0c6e51504c1bb84c08e8240715291d50731dd3c2aa9be80.jpg)

<details>
<summary>bar</summary>

| class | ID | OOD |
|---|---|---|
| 1 | 0.04 | 0.035 |
| 2 | 0.10 | 0.125 |
| 3 | 0.08 | 0.07 |
| 4 | 0.105 | 0.14 |
| 5 | 0.025 | 0.025 |
| 6 | 0.05 | 0.055 |
| 7 | 0.045 | 0.045 |
| 8 | 0.08 | 0.01 |
| 9 | 0.10 | 0.125 |
| 10 | 0.03 | 0.015 |
| 11 | 0.27 | 0.175 |
| 12 | 0.03 | 0.065 |
</details>

(c) Density

Figure 3: Class balance for CoauthorCS dataset across different types of shifts.   
![](images/fb23dc8b9ff6f12ddd601aff9051265747d58c77ccacf3fb4a9af246dea49e21.jpg)

<details>
<summary>bar</summary>

| class | ID | OOD |
|---|---|---|
| 1 | 0.125 | 0.115 |
| 2 | 0.145 | 0.13 |
| 3 | 0.135 | 0.165 |
| 4 | 0.16 | 0.14 |
| 5 | 0.27 | 0.295 |
| 6 | 0.065 | 0.07 |
| 7 | 0.11 | 0.09 |
</details>

(a) Popularity

![](images/f1297a2a9629d06d7069f0eb181323e2d31ff6dc2d2e7e698e63a1bbef5fae0f.jpg)

<details>
<summary>bar</summary>

| class | ID (mitio) | OOD (mitio) |
|---|---|---|
| 1 | 0.09 | 0.14 |
| 2 | 0.13 | 0.13 |
| 3 | 0.29 | 0.01 |
| 4 | 0.07 | 0.22 |
| 5 | 0.18 | 0.39 |
| 6 | 0.05 | 0.07 |
| 7 | 0.17 | 0.03 |
</details>

(b) Locality

![](images/727b5af4cc0bb9b1fdd67f8bc818c5abb018df6f93328c86ac4f34c7ddf205cc.jpg)

<details>
<summary>bar</summary>

| class | ID | OOD |
|---|---|---|
| 1 | 0.11 | 0.12 |
| 2 | 0.13 | 0.14 |
| 3 | 0.15 | 0.14 |
| 4 | 0.16 | 0.13 |
| 5 | 0.23 | 0.34 |
| 6 | 0.06 | 0.07 |
| 7 | 0.14 | 0.06 |
</details>

(c) Density   
Figure 4: Class balance for CoraML dataset across different types of shifts.

Class balance Class balance directly affects the amount of evidence acquired by the graph processing model and used for estimating uncertainty and making predictions. It is especially important for evaluating Dirichlet-based models which exploit normalizing flows, as their density estimates can become irrelevant due to a significant change in class balance.

In Figures 2–4, one can see that the popularity-based split does not create a notable difference in the class balance between the ID and OOD subsets (for the datasets under consideration). Thus, the more important and less important nodes have, on average, the same probability of belonging to a particular class. More noticeable differences are induced by the density-based split. The locality-based split leads to the most significant changes for some classes. This shows that the split strategies based on the structural locality in graph can be very challenging as they also affect such crucial statistics as class balance.

Degree distribution The node degree distribution is one of the basic structural characteristics of graph that describes the local importance of nodes. Degrees are especially important for such graph

![](images/5cdc870346fa02bd5cb6db97bf16a62e61792186f62ef14a7f83eb5818875a55.jpg)

<details>
<summary>scatter</summary>

| degree | count | group |
| ------ | ----- | ----- |
| 10^0   | 10^2  | ID    |
| 10^1   | 10^1  | ID    |
| 10^2   | 10^0  | ID    |
| 10^3   | 10^0  | ID    |
| 10^0   | 10^2  | OOD   |
| 10^1   | 10^2  | OOD   |
| 10^2   | 10^1  | OOD   |
| 10^3   | 10^0  | OOD   |
</details>

(a) Popularity

![](images/09c6c87146c42b5af4c052183ddca8270541730a9e414470c4db70f73150e34c.jpg)

<details>
<summary>scatter</summary>

| degree | count | group |
| ------ | ----- | ----- |
| 10^0   | 10^2  | ID    |
| 10^1   | 10^1  | ID    |
| 10^2   | 10^0  | ID    |
| 10^3   | 10^0  | ID    |
| 10^0   | 10^2  | OOD   |
| 10^1   | 10^1  | OOD   |
| 10^2   | 10^0  | OOD   |
| 10^3   | 10^0  | OOD   |
</details>

(b) Locality

![](images/298e71a8c0119c068ed6276a2a160e839ec089f58470ce88a2fcc96553d5f49f.jpg)

<details>
<summary>scatter</summary>

| degree | count | group |
| ------ | ----- | ----- |
| 10^0   | 10^2  | ID    |
| 10^1   | 10^1  | OOD   |
| 10^2   | 10^0  | ID    |
| 10^3   | 10^0  | OOD   |
</details>

(c) Density

Figure 5: The distribution of node degrees for AmazonComputer dataset across different shifts.   
![](images/a095874d50671a5f92005cdf47a3a997aabb198de600b98080f20eff576c76ef.jpg)

<details>
<summary>scatter</summary>

| degree | count | group |
| ------ | ----- | ----- |
| 10^0   | 10^3  | OOD   |
| 10^1   | 10^2  | ID    |
| 10^2   | 10^0  | ID    |
</details>

(a) Popularity

![](images/49774d053b38803180285fa3bb7dbfb04d0a5d39fefe849767c1e3fa2f9cfe1a.jpg)

<details>
<summary>scatter</summary>

| degree | count | group |
| ------ | ----- | ----- |
| 10^0   | 10^3  | ID    |
| 10^1   | 10^2  | ID    |
| 10^2   | 10^0  | ID    |
| 10^0   | 10^3  | OOD   |
| 10^1   | 10^2  | OOD   |
| 10^2   | 10^0  | OOD   |
</details>

(b) Locality

![](images/a970a9dd6007f38b024f9d5339a51d5a14f7397d805526284900356d41999add.jpg)

<details>
<summary>scatter</summary>

| degree | count | group |
| ------ | ----- | ----- |
| 10^0   | 10^3  | ID    |
| 10^1   | 10^2  | OOD   |
| 10^2   | 10^0  | ID    |
| 10^2   | 10^0  | OOD   |
</details>

(c) Density

Figure 6: The distribution of node degrees for CoauthorCS dataset across different shifts.   
![](images/da32962a44f7927575bcd036b538eca960c2f43eb6838e7755fffb81876e06d3.jpg)

<details>
<summary>scatter</summary>

| degree | count | group |
| ------ | ----- | ----- |
| 1      | 100   | ID    |
| 2      | 50    | ID    |
| 3      | 30    | ID    |
| 4      | 20    | ID    |
| 5      | 15    | ID    |
| 6      | 10    | ID    |
| 7      | 8     | ID    |
| 8      | 6     | ID    |
| 9      | 5     | ID    |
| 10     | 4     | ID    |
| 11     | 3     | ID    |
| 12     | 2     | ID    |
| 13     | 1.5   | ID    |
| 14     | 1.2   | ID    |
| 15     | 1.0   | ID    |
| 16     | 0.8   | ID    |
| 17     | 0.6   | ID    |
| 18     | 0.5   | ID    |
| 19     | 0.4   | ID    |
| 20     | 0.3   | ID    |
| 21     | 0.2   | ID    |
| 22     | 0.1   | ID    |
| 23     | 0.05  | ID    |
| 24     | 0.03  | ID    |
| 25     | 0.02  | ID    |
| 26     | 0.01  | ID    |
| 27     | 0.005 | ID    |
| 28     | 0.003 | ID    |
| 29     | 0.002 | ID    |
| 30     | 0.001 | ID    |
| 31     | 0.0005| ID    |
| 32     | 0.0003| ID    |
| 33     | 0.0002| ID    |
| 34     | 0.0001| ID    |
| 35     | 0.00005| ID   |
| 36     | 0.00003| ID   |
| 37     | 0.00002| ID   |
| 38     | 0.00001| ID   |
| 39     | 0.000005| ID   |
| 40     | 0.000003| ID   |
| 41     | 0.000002| ID   |
| 42     | 0.000001| ID   |
| 43     | 0.0000005| ID   |
| 44     | 0.0000003| ID   |
| 45     | 0.0000002| ID   |
| 46     | 0.0000001| ID   |
| 47     | 0.00000005| ID   |
| 48     | 0.00000003| ID   |
| 49     | 0.00000002| ID   |
| 50     | 0.00000001| ID   |
| 51     | 0.000000005| ID   |
| 52     | 0.000000003| ID   |
| 53     | 0.000000002| ID   |
| 54     | 0.000000001| ID   |
| 55     | 0.0000000005| ID   |
| 56     | 0.0000000003| ID   |
| 57     | 0.0000000002| ID   |
| 58     | 0.0000000001| ID   |
| 59     | 0.00000000005| ID   |
| 60     | 6             | OOD   |
| 61     | 4             | OOD   |
| 62     | 3             | OOD   |
| 63     | 2             | OOD   |
| 64     | 1             | OOD   |
| 65     | 1             | OOD   |
| 66     | 1             | OOD   |
| 67     | 1             | OOD   |
| 68     | 1             | OOD   |
| 69     | 1             | OOD   |
| 70     | 1             | OOD   |
| 71     | 1             | OOD   |
| 72     | 1             | OOD   |
| 73     | 1             | OOD   |
| 74     | 1             | OOD   |
| 75     | 1             | OOD   |
| 76     | 1             | OOD   |
| 77     | 1             | OOD   |
| 78     | 1             | OOD   |
| 79     | 1             | OOD   |
| 80     | 1             | OOD   |
| 81     | 1             | OOD   |
| 82     | 1             | OOD   |
| 83     | 1             | OOD   |
| 84     | 1             | OOD   |
| 85     | 1             | OOD   |
| 86     | 1             | OOD   |
| 87     | 1             | OOD   |
| 88     | 1             | OOD   |
| 89     | 1             | OOD   |
| 90     | 1             | OOD   |
| 91     | 1             | OOD   |
| 92     | 1             | OOD   |
| 93     | 1             | OOD   |
| 94     | 1             | OOD   |
| 95     | 1             | OOD   |
| 96     | 1             | OOD   |
| 97     | 1             | OOD   |
| 98     | 1             | OOD   |
| 99     | 1             | OOD   |
| Note: The actual counts may vary due to the random nature of the data generation. The provided values are just an example from the code execution.
</details>

(a) Popularity

![](images/7b239d70d404f9fb2b6cf19cb1e2a6d3f5f2bb037c48b5811a7e76473b1dd964.jpg)

<details>
<summary>scatter</summary>

| degree | count | group |
| ------ | ----- | ----- |
| 1      | 200   | ID    |
| 2      | 150   | ID    |
| 3      | 120   | ID    |
| 4      | 100   | ID    |
| 5      | 80    | ID    |
| 6      | 60    | ID    |
| 7      | 50    | ID    |
| 8      | 40    | ID    |
| 9      | 30    | ID    |
| 10     | 20    | ID    |
| 11     | 15    | ID    |
| 12     | 10    | ID    |
| 13     | 8     | ID    |
| 14     | 6     | ID    |
| 15     | 5     | ID    |
| 16     | 4     | ID    |
| 17     | 3     | ID    |
| 18     | 2     | ID    |
| 19     | 1     | ID    |
| 20     | 1     | ID    |
| 21     | 1     | ID    |
| 22     | 1     | ID    |
| 23     | 1     | ID    |
| 24     | 1     | ID    |
| 25     | 1     | ID    |
| 26     | 1     | ID    |
| 27     | 1     | ID    |
| 28     | 1     | ID    |
| 29     | 1     | ID    |
| 30     | 1     | ID    |
| 31     | 1     | ID    |
| 32     | 1     | ID    |
| 33     | 1     | ID    |
| 34     | 1     | ID    |
| 35     | 1     | ID    |
| 36     | 1     | ID    |
| 37     | 1     | ID    |
| 38     | 1     | ID    |
| 39     | 1     | ID    |
| 40     | 1     | ID    |
| 41     | 1     | ID    |
| 42     | 1     | ID    |
| 43     | 1     | ID    |
| 44     | 1     | ID    |
| 45     | 1     | ID    |
| 46     | 1     | ID    |
| 47     | 1     | ID    |
| 48     | 1     | ID    |
| 49     | 1     | ID    |
| 50     | 1     | ID    |
| 51     | 1     | ID    |
| 52     | 1     | ID    |
| 53     | 1     | ID    |
| 54     | 1     | ID    |
| 55     | 1     | ID    |
| 56     | 1     | ID    |
| 57     | 1     | ID    |
| 58     | 1     | ID    |
| 59     | 1     | ID    |
| 60     | 1     | ID    |
| 61     | 1     | ID    |
| 62     | 1     | ID    |
| 63     | 1     | ID    |
| 64     | 1     | ID    |
| 65     | 1     | ID    |
| 66     | 1     | ID    |
| 67     | 1     | ID    |
| 68     | 1     | ID    |
| 69     | 1     | ID    |
| 70     | 1     | ID    |
| 71     | 1     | ID    |
| 72     | 1     | ID    |
| 73     | 1     | ID    |
| 74     | 1     | ID    |
| 75     | 1     | ID    |
| 76     | 1     | ID    |
| 77     | 1     | ID    |
| 78     | 1     | ID    |
| 79     | 1     | ID    |
| 80     | 1     | ID    |
| 81     | 1     | ID    |
| 82     | 1     | ID    |
| 83     | 1     | ID    |
| 84     | 1     | ID    |
| 85     | 1     | ID    |
| 86     | 1     | ID    |
| 87     | 1     | ID    |
| 88     | 1     | ID    |
| 89     | 1     | ID    |
| 90     | 1     | ID    |
| 91     | 1     | ID    |
| 92     | 1     | ID    |
| 93     | 1     | ID    |
| 94     | 1     | ID    |
| 95     | 1     | ID    |
| 96     | 1     | ID    |
| 97     | 1     | ID    |
| 98     | 1     | ID    |
| 99     | 1     | ID    |
| Note: The actual counts for each degree are not provided in the code. The actual counts for IDs and OOD are not provided in the code. There is only one data point for the first three data points. The values for IDs and OOD are estimated based on the provided code. There is no additional data series present in the code.
</details>

(b) Locality

![](images/e48c7185486f7aeed3030e7452f4c1270c0be6914d12410aa9e637becd863a63.jpg)

<details>
<summary>scatter</summary>

| degree | count | group |
| ------ | ----- | ----- |
| 10^0   | 10^2  | ID    |
| 10^0   | 10^2  | OOD   |
| 10^1   | 10^1  | ID    |
| 10^1   | 10^1  | OOD   |
| 10^2   | 10^0  | ID    |
| 10^2   | 10^0  | OOD   |
</details>

(c) Density   
Figure 7: The distribution of node degrees for CoraML dataset across different shifts.

processing methods as GNNs since they describe how many channels around the considered node are used for message passing and aggregation.

In Figures 5–7, one can see that the most significant change in the degree distribution appears when the ID and OOD subsets are separated based on PageRank: the ID part contains more high-degree nodes. This is expected since PageRank is a graph characteristic measuring node importance (a.k.a. centrality), and node degree is the simplest centrality measure known to be correlated with PageRank. For the locality-based splits, the difference in degree distribution is smaller but still significant since PPR selects nodes by their relative importance for a particular node, so some high-degree nodes can be less important. Finally, for the density-based splits, the degree distribution also changes, but the extent of shift is usually less significant, and for some datasets (e.g., CoauthorCS), the higher-degree nodes are in the OOD subset.

Graph distance distribution The distance between two nodes in a graph is defined as the length of the shortest path between them. Here, we compute such distances between the nodes in the ID or OOD subset within the original graph, i.e., we consider the whole graph when searching for the shortest path. The distribution of distances illustrates which parts of the datasets are more locally concentrated.

In Figures 8–10, one can observe that the locality-based split leads to the most significant changes in distances, making the OOD nodes nearly twice as far from each other as the ID ones. At the same time, the popularity-based split does not lead to such a difference, revealing almost the same distributions on ID and OOD subsets. This means that the popularity bias in a graph does not prevent one from

![](images/dafcd69908134c7183f2546a0f87acc8c60aff416302f3b8a5465370812e6846.jpg)

<details>
<summary>scatter</summary>

| distance | count (ID) | count (OOD) |
| -------- | ---------- | ----------- |
| 1        | 10^5       | 10^4        |
| 2        | 10^6       | 10^6        |
| 3        | 10^7       | 10^7        |
| 4        | 10^7       | 10^7        |
| 5        | 10^6       | 10^6        |
| 6        | 10^5       | 10^5        |
| 7        | 10^4       | 10^4        |
| 8        | 10^2       | 10^3        |
| 9        | 10^1       | 10^2        |
| 10       | 10^1       | 10^1        |
</details>

(a) Popularity

![](images/57ca11d162a4bc3a8a1a27a0dcaccf9265aae25b5d46efc77c87a134847a3049.jpg)

<details>
<summary>scatter</summary>

| distance | count  | group |
| -------- | ------ | ----- |
| 1        | 10^5   | ID    |
| 2        | 10^6   | ID    |
| 3        | 10^7   | ID    |
| 4        | 10^6   | ID    |
| 5        | 10^5   | ID    |
| 6        | 10^6   | ID    |
| 7        | 10^5   | ID    |
| 8        | 10^4   | ID    |
| 9        | 10^3   | ID    |
| 10       | 10^2   | ID    |
| 1        | 10^4   | OOD   |
| 2        | 10^5   | OOD   |
| 3        | 10^6   | OOD   |
| 4        | 10^7   | OOD   |
| 5        | 10^6   | OOD   |
| 6        | 10^5   | OOD   |
| 7        | 10^4   | OOD   |
| 8        | 10^3   | OOD   |
| 9        | 10^2   | OOD   |
| 10       | 10^1   | OOD   |
</details>

(b) Locality

![](images/2ac5571065a4eb0e0219a6a2b07df3db4248c8a5a3ef5bb57a7433e76cec7d4c.jpg)

<details>
<summary>scatter</summary>

| distance | count  | group |
| -------- | ------ | ----- |
| 1        | 10^4   | ID    |
| 2        | 10^6   | ID    |
| 3        | 10^7   | OOD   |
| 4        | 10^7   | OOD   |
| 5        | 10^6   | OOD   |
| 6        | 10^5   | OOD   |
| 7        | 10^4   | OOD   |
| 8        | 10^3   | ID    |
| 9        | 10^2   | ID    |
| 10       | 10^2   | OOD   |
</details>

(c) Density

Figure 8: The distribution of graph distances for AmazonComputer dataset across different shifts.   
![](images/ae8b03a4812030e995063fdfb621b95062e5cb4645fe5477def45ad365f0b270.jpg)

<details>
<summary>scatter</summary>

| distance | ID     | OOD    |
| -------- | ------ | ------ |
| 0        | 10^4   | 10^3   |
| 5        | 10^6   | 10^5   |
| 10       | 10^5   | 10^4   |
| 15       | 10^4   | 10^3   |
| 20       | 10^3   | 10^2   |
| 25       | 10^1   | 10^0   |
</details>

(a) Popularity

![](images/3fcff645aa8aa004762c459749c02eb561cce7ce8d457633c4b73ef87602500c.jpg)  
(b) Locality

![](images/f69a7b90b7e71a3581333826a846d94d97a151baca61276371d1e8457f8c74cb.jpg)

<details>
<summary>scatter</summary>

| distance | count (ID) | count (OOD) |
| -------- | ---------- | ----------- |
| 0        | 10^4       | 10^4        |
| 1        | 10^5       | 10^5        |
| 2        | 10^6       | 10^6        |
| 3        | 10^6       | 10^6        |
| 4        | 10^6       | 10^6        |
| 5        | 10^7       | 10^7        |
| 6        | 10^7       | 10^7        |
| 7        | 10^6       | 10^6        |
| 8        | 10^5       | 10^5        |
| 9        | 10^5       | 10^5        |
| 10       | 10^5       | 10^5        |
| 11       | 10^4       | 10^4        |
| 12       | 10^4       | 10^4        |
| 13       | 10^4       | 10^4        |
| 14       | 10^4       | 10^4        |
| 15       | 10^4       | 10^4        |
| 16       | 10^4       | 10^4        |
| 17       | 10^3       | 10^3        |
| 18       | 10^3       | 10^3        |
| 19       | 10^2       | 10^2        |
| 20       | 10^2       | 10^2        |
| 21       | 10^1       | 10^1        |
| 22       | 10^1       | 10^1        |
| 23       | 10^0       | 10^0        |
| 24       | 10^0       | 10^0        |
| 25       | 10^0       | 10^0        |
</details>

(c) Density

Figure 9: The distribution of graph distances for CoauthorCS dataset across different shifts.   
![](images/97a90852f2671aaead9ac85b63c8cf4ceea18c38ee03cb5b57e48f228bc37f47.jpg)

<details>
<summary>scatter</summary>

| distance | ID (count) | OOD (count) |
| :--- | :--- | :--- |
| 1 | 1000 | 100 |
| 2 | 10000 | 1000 |
| 3 | 100000 | 10000 |
| 4 | 100000 | 100000 |
| 5 | 100000 | 100000 |
| 6 | 100000 | 100000 |
| 7 | 10000 | 10000 |
| 8 | 1000 | 1000 |
| 9 | 1000 | 1000 |
| 10 | 1000 | 100 |
| 11 | 100 | 10 |
| 12 | 10 | 1 |
| 13 | 1 | 1 |
| 14 | 1 | 1 |
| 15 | 1 | 1 |
| 16 | 1 | 1 |
The chart displays the counts of ID and OOD at different distances. The x-axis represents distance, and the y-axis represents count. The data points are color-coded: light blue for ID and red for OOD. The chart is labeled in English. The title is 'ID' but not explicitly shown in the image.
</details>

(a) Popularity

![](images/8766701e479e9df20de403fbbfa651388021f5fd5780ca76701b6da6b9fc7105.jpg)

<details>
<summary>scatter</summary>

| distance | count  | group |
| -------- | ------ | ----- |
| 2        | 10000  | ID    |
| 3        | 100000 | ID    |
| 4        | 100000 | ID    |
| 5        | 100000 | ID    |
| 6        | 100000 | ID    |
| 7        | 100000 | ID    |
| 8        | 1000   | ID    |
| 9        | 100    | ID    |
| 10       | 1      | ID    |
| 11       | 1      | ID    |
| 12       | 1      | ID    |
| 13       | 1      | ID    |
| 14       | 1      | ID    |
| 15       | 1      | ID    |
| 16       | 1      | ID    |
| 2        | 1000   | OOD   |
| 3        | 10000  | OOD   |
| 4        | 100000 | OOD   |
| 5        | 100000 | OOD   |
| 6        | 100000 | OOD   |
| 7        | 100000 | OOD   |
| 8        | 1000   | OOD   |
| 9        | 100    | OOD   |
| 10       | 1      | OOD   |
| 11       | 1      | OOD   |
| 12       | 1      | OOD   |
| 13       | 1      | OOD   |
| 14       | 1      | OOD   |
| 15       | 1      | OOD   |
| 16       | 1      | OOD   |
</details>

(b) Locality

![](images/6889b44892835d8e9da37de0f79820dd06535f073c5557776791c379e557f7a1.jpg)

<details>
<summary>scatter</summary>

| distance | ID (count) | OOD (count) |
| :--- | :--- | :--- |
| 2 | 10000 | 1000 |
| 3 | 100000 | 10000 |
| 4 | 100000 | 100000 |
| 5 | 100000 | 100000 |
| 6 | 100000 | 100000 |
| 7 | 100000 | 100000 |
| 8 | 10000 | 10000 |
| 9 | 10000 | 1000 |
| 10 | 1000 | 100 |
| 11 | 100 | 10 |
| 12 | 10 | 1 |
| 13 | 1 | 1 |
| 14 | 1 | 1 |
| 15 | 1 | 1 |
| 16 | 1 | 1 |
The chart displays the counts of ID and OOD at different distances. The x-axis represents distance, and the y-axis represents count. The data points are color-coded: light blue for ID and red for OOD. The chart is labeled with the same axes and the legend in English. The title is 'I don't have a specific label' but it's not explicitly labeled.
</details>

(c) Density   
Figure 10: The distribution of graph distances for CoraML dataset across different shifts.

covering the less popular periphery nodes since the most popular nodes may be widespread. Similarly, the density-based split does not induce a significant difference in pairwise distances.

# B Results on other split ratios

The main part of our empirical study is focused on a 50% : 50% split ratio. In this section, we extend our discussion with varying ID to OOD ratios and consider two setups where the fraction of OOD samples is smaller — 70% : 30% and 90% : 10%.

Let us first revisit our analysis of the proposed structural shifts on a 70% : 30% ratio. As can be seen in Table 4, the results in both OOD robustness, which is evaluated in the node classification task, and OOD detection, which is done by means of uncertainty estimation, have not changed much compared to our base setup discussed in the main text — the considered graph models show the average drop in performance of 3% on the popularity-based shift and 6% on the density-based shift, while the OOD detection performance remains at nearly 81 and 54 points, respectively. Despite the fact that the graph models now have access to more diverse structures in terms of popularity and local density, making decisions about unimportant and sparsely surrounded nodes remains as difficult as before.

Further, the drop in predictive performance on the locality-based shift appears to be less significant compared to the original setup and reaches only 5% on average instead of the previous 14%. At the same time, the OOD detection performance on this structural shift drops to 79 points, whereas it was 85 points in the original setup. These results also match our intuition — as the distance between the ID and OOD nodes decreases, the OOD samples become less distinguishable from the ID ones. This makes the OOD detection performance drop on average, while the gap between ID and OOD metrics in standard node classification disappears.

We also may compare the existing graph models based on the results in Table 5. As can be seen, the ranking of methods remains almost the same as in the original setup. Specifically, the most simple data augmentation technique Mixup often shows the best performance in OOD robustness, while DE provides the second best results on most structural shifts. As for OOD detection, the methods based on the entropy of predictive distribution again outperform the Dirichlet ones, and the uncertainty estimates produced by DE are best correlated with the OOD examples. Thus, our observations regarding the performance of graph models are consistent with those reported in the paper.

Table 4: Comparison of structural distributional shifts in terms of OOD robustness and OOD detection. We report the drop in predictive performance of the ERM method measured by Accuracy (left) and the quality of uncertainty estimates of the SE method measured by AUROC (right). The ID to OOD ratio in these experiments is 70% : 30% instead of 50% : 50% as in the main text. 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>AmazonComputer</td><td>-3.80%</td><td>-12.66%</td><td>-14.04%</td></tr><tr><td>AmazonPhoto</td><td>-7.19%</td><td>-3.03%</td><td>-6.79%</td></tr><tr><td>CoauthorCS</td><td>-6.51%</td><td>-1.89%</td><td>-6.64%</td></tr><tr><td>CoauthorPhysics</td><td>-2.58%</td><td>-5.50%</td><td>-2.47%</td></tr><tr><td>CoraML</td><td>-7.56%</td><td>-13.63%</td><td>-9.49%</td></tr><tr><td>CiteSeer</td><td>+3.24%</td><td>+2.50%</td><td>-4.13%</td></tr><tr><td>PubMed</td><td>+0.46%</td><td>-0.67%</td><td>-1.76%</td></tr><tr><td>OGB-Products</td><td>-2.35%</td><td>-2.36%</td><td>-4.02%</td></tr><tr><td>Average</td><td>-3.28%</td><td>-4.65%</td><td>-6.17%</td></tr></table>

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>AmazonComputer</td><td>87.83</td><td>85.77</td><td>50.46</td></tr><tr><td>AmazonPhoto</td><td>91.80</td><td>85.78</td><td>47.27</td></tr><tr><td>CiteSeer</td><td>70.56</td><td>64.55</td><td>58.04</td></tr><tr><td>CoauthorCS</td><td>83.68</td><td>81.13</td><td>63.24</td></tr><tr><td>CoauthorPhysics</td><td>86.70</td><td>82.06</td><td>39.68</td></tr><tr><td>CoraML</td><td>82.06</td><td>84.63</td><td>76.82</td></tr><tr><td>PubMed</td><td>66.64</td><td>63.34</td><td>55.79</td></tr><tr><td>OGB-Products</td><td>82.35</td><td>82.44</td><td>43.50</td></tr><tr><td>Average</td><td>81.45</td><td>78.71</td><td>54.35</td></tr></table>

Table 5: Comparison of several graph methods for improving the OOD robustness (left) and detecting the OOD inputs by means of uncertainty estimation (right). For each task, we report the method ranks averaged across different graph datasets (lower is better). The ID to OOD ratio in these experiments is 70% : 30% instead of 50% : 50% as in the main text. 

<table><tr><td rowspan="2"></td><td colspan="2">Popularity</td><td colspan="2">Locality</td><td colspan="2">Density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td>3.0</td><td>3.4</td><td>2.9</td><td>3.1</td><td>4.3</td><td>4.3</td></tr><tr><td>Mixup</td><td>1.9</td><td>2.6</td><td>1.4</td><td>2.7</td><td>1.4</td><td>3.3</td></tr><tr><td>EERM</td><td>4.9</td><td>3.9</td><td>5.0</td><td>4.0</td><td>4.4</td><td>4.1</td></tr><tr><td>DANN</td><td>4.1</td><td>4.4</td><td>4.3</td><td>4.3</td><td>3.7</td><td>2.6</td></tr><tr><td>CORAL</td><td>4.1</td><td>4.0</td><td>4.4</td><td>4.9</td><td>4.1</td><td>3.7</td></tr><tr><td>DE</td><td>3.0</td><td>2.7</td><td>3.0</td><td>2.0</td><td>3.0</td><td>3.0</td></tr></table>

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>1.3</td><td>2.3</td><td>3.7</td></tr><tr><td>GPN</td><td>3.3</td><td>3.9</td><td>3.9</td></tr><tr><td>NatPN</td><td>5.4</td><td>4.3</td><td>3.7</td></tr><tr><td>DE</td><td>3.0</td><td>1.4</td><td>1.9</td></tr><tr><td>GPE</td><td>3.1</td><td>3.9</td><td>3.4</td></tr><tr><td>NatPE</td><td>4.9</td><td>5.3</td><td>4.4</td></tr></table>

Now let us discuss the setup based on a more extreme 90% : 10% ratio. According to Table 6, the drop in predictive performance under the popularity-based and locality-based shifts reaches more than 6% instead of 3% in the previous setup and almost 11% instead of 6%, respectively. This was expected and can be explained by the fact that graph models are now tested on the nodes with the most extreme structural properties (i.e., nodes with the lowest PageRank or clustering coefficient), and their inaccurate predictions are no longer compensated by the more accurate ones, which were made on the nodes with far less anomalous properties.

At the same time, the performance drop on the locality-based shift appears to be nearly 1% on average. This result is quite reasonable since now graph models have access to the whole variety of graph substructures and node features, which allows them to predict equally well for any graph region regardless of its distance to some particular node. The observed effects are interesting and also important to consider when using our approach for evaluating graph models.

Regarding the OOD detection performance, we observe that results on the locality-based shift keep decreasing and reach 73 points, in contrast to 79 points in the previous setup (and this might happen for the same reason as we discussed before). At the same time, the detection metrics on the remaining structural shifts appear to be nearly the same (79 points instead of the previous 81 on the popularity-based shift) or even better on average (63 points instead of 54 on the density-based shift), which is consistent with the changes in predictive performance discussed above.

As for the ranking of different models in Table 7, it remains almost the same, with the most simple methods providing top performance on the majority of prediction tasks.

Table 6: Comparison of structural distributional shifts in terms of OOD robustness and OOD detection. We report the drop in predictive performance of the ERM method measured by Accuracy (left) and the quality of uncertainty estimates of the SE method measured by AUROC (right). The ID to OOD ratio in these experiments is 90% : 10% instead of 50% : 50% as in the main text. 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>AmazonComputer</td><td>-8.45%</td><td>-7.79%</td><td>-30.98%</td></tr><tr><td>AmazonPhoto</td><td>-8.00%</td><td>+0.12%</td><td>-16.35%</td></tr><tr><td>CoauthorCS</td><td>-9.79%</td><td>-4.42%</td><td>-7.76%</td></tr><tr><td>CoauthorPhysics</td><td>-3.77%</td><td>-4.91%</td><td>-5.32%</td></tr><tr><td>CoraML</td><td>-16.10%</td><td>+5.52%</td><td>-8.03%</td></tr><tr><td>CiteSeer</td><td>-5.14%</td><td>+1.35%</td><td>+1.86%</td></tr><tr><td>PubMed</td><td>+3.33%</td><td>+4.70%</td><td>-2.63%</td></tr><tr><td>OGB-Products</td><td>-2.92%</td><td>-2.75%</td><td>-15.61%</td></tr><tr><td>Average</td><td>-6.36%</td><td>-1.02%</td><td>-10.60%</td></tr></table>

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>AmazonComputer</td><td>83.42</td><td>76.53</td><td>68.97</td></tr><tr><td>AmazonPhoto</td><td>89.01</td><td>82.57</td><td>64.41</td></tr><tr><td>CiteSeer</td><td>75.90</td><td>65.11</td><td>55.48</td></tr><tr><td>CoauthorCS</td><td>83.22</td><td>76.02</td><td>70.63</td></tr><tr><td>CoauthorPhysics</td><td>84.68</td><td>78.60</td><td>52.97</td></tr><tr><td>CoraML</td><td>79.36</td><td>69.64</td><td>69.78</td></tr><tr><td>PubMed</td><td>61.05</td><td>57.14</td><td>55.11</td></tr><tr><td>OGB-Products</td><td>77.92</td><td>78.02</td><td>68.33</td></tr><tr><td>Average</td><td>79.32</td><td>72.95</td><td>63.21</td></tr></table>

Table 7: Comparison of several graph methods for improving the OOD robustness (left) and detecting the OOD inputs by means of uncertainty estimation (right). For each task, we report the method ranks averaged across different graph datasets (lower is better). The ID to OOD ratio in these experiments is 90% : 10% instead of 50% : 50% as in the main text. 

<table><tr><td rowspan="2"></td><td colspan="2">Popularity</td><td colspan="2">Locality</td><td colspan="2">Density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td>3.3</td><td>3.7</td><td>3.4</td><td>3.0</td><td>3.3</td><td>4.4</td></tr><tr><td>Mixup</td><td>1.7</td><td>2.4</td><td>1.7</td><td>2.9</td><td>1.7</td><td>3.0</td></tr><tr><td>EERM</td><td>5.0</td><td>4.1</td><td>5.0</td><td>3.7</td><td>4.9</td><td>3.9</td></tr><tr><td>DANN</td><td>4.3</td><td>4.0</td><td>4.0</td><td>4.3</td><td>3.4</td><td>3.0</td></tr><tr><td>CORAL</td><td>3.9</td><td>3.6</td><td>4.1</td><td>5.1</td><td>4.3</td><td>3.4</td></tr><tr><td>DE</td><td>2.9</td><td>3.1</td><td>2.7</td><td>2.0</td><td>3.4</td><td>3.3</td></tr></table>

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>1.6</td><td>2.6</td><td>3.3</td></tr><tr><td>GPN</td><td>3.1</td><td>3.0</td><td>3.9</td></tr><tr><td>NatPN</td><td>5.0</td><td>4.4</td><td>3.7</td></tr><tr><td>DE</td><td>2.6</td><td>1.7</td><td>1.6</td></tr><tr><td>GPE</td><td>3.6</td><td>4.0</td><td>4.3</td></tr><tr><td>NatPE</td><td>5.1</td><td>5.3</td><td>4.3</td></tr></table>

# C Comparison with the GOOD benchmark from Gui et al. [10]

Our work complements and extends the GOOD benchmark recently proposed by Gui et al. [10]. However, there are several important differences that we discuss in this section.

One of the main properties of the GOOD benchmark is its theoretical distinction between two types of distributional shifts, which are represented through a graphical model. In particular, the authors consider covariate shifts, in which the distribution of features changes while the conditional distribution of targets given features remains the same, and concept shifts, where the opposite situation occurs, i.e., the conditional target distribution changes, while the feature distribution is the same. Although this distinction might be very helpful for understanding the properties of particular GNN models, such exclusively covariate or concept shifts rarely happen in practice where both types of shifts are present at the same time.

To create pure covariate or concept shifts, Gui et al. $[10]$ introduce different subsets of variables that either fully determine the target, create confounding associations with the target, or are completely independent of the target. This has to be properly handled and makes it non-trivial to create distributional shifts on new datasets with this approach. Indeed, the distributional shifts in the GOOD benchmark can be properly implemented only for synthetic graph datasets or via appending synthetic features that either describe various domains as completely independent variables or create the necessary concepts by inducing some spurious correlation with the target. Moreover, the authors claim that, in the case of real-world datasets, one has to perform screening over the available node features to create the required setup of domain or concept shift. This fact implies numerous restrictions on how the data splits can be prepared.

In contrast, our method does not distinguish between covariate and concept shifts and thus can be universally applied to any dataset and does not require any dataset modifications. Importantly, the type of distributional shift and the sizes of all split parts are easily controllable. This flexibility is the main advantage of our approach.

Finally, Gui et al. [10] confirm the importance of using both node features and graph structure. Still, their node-level distributional shifts are mainly based on node features such as the number of words or the year of publication in a citation network, the language of users in a social network, or the name of organizations in a webpage network. As for the graph properties, only node degrees are used in some citation networks. In contrast, we focus on the graph structure and propose diverse structural shifts together with a framework allowing one to easily create splits based on other structural properties.

# D Structural properties of the OGB data splits proposed by Hu et al. [12]

In Figures 11–13, we present how the distribution of structural node characteristics, including PageRank, Personalized PageRank, and clustering coefficient, may change across the train and test parts in some OGB datasets, where the distributional shifts are constructed using some domain-specific node feature. This shows that our approach to creating data splits using structural graph properties can be similar to realistic distributional shifts.

![](images/439095a30e3bdb80ff71e403280bb9129928819f4231df499520181896c2fa6f.jpg)

<details>
<summary>bar</summary>

| property values | train density | test density |
| :--- | :--- | :--- |
| 10^-13 | 0.05 | 0.08 |
| 10^-12 | 0.07 | 0.09 |
| 10^-11 | 0.04 | 0.06 |
| 10^-10 | 0.02 | 0.03 |
| 10^-9 | 0.01 | 0.015 |
| 10^-8 | 0.005 | 0.008 |
| 10^-7 | 0.002 | 0.003 |
| 10^-6 | 0.001 | 0.0015 |
</details>

(a) PageRank

![](images/592d223025ac07ec54aabff9f2f46da4cacf57e3f2b1057a9ce7a03506cee37b.jpg)

<details>
<summary>bar</summary>

| property values | train density | test density |
| :--- | :--- | :--- |
| 10^-14 | 10^0 | 10^0 |
| 10^-13 | 10^-1 | 10^-1 |
| 10^-12 | 10^-1 | 10^-1 |
| 10^-11 | 10^-1 | 10^-1 |
| 10^-10 | 10^-2 | 10^-2 |
| 10^-9 | 10^-3 | 10^-3 |
| 10^-8 | 10^-4 | 10^-4 |
| 10^-7 | 10^-5 | 10^-5 |
| 10^-6 | 10^-6 | 10^-6 |
</details>

(b) Personalized PageRank

![](images/37c80fd6682b5a6dee2504a97488169121c0c12595eb56042a30c96a133e9fd8.jpg)

<details>
<summary>bar</summary>

| property values | train | test |
| --------------- | ----- | ---- |
| 0.0             | 10^1  | 10^1 |
| 0.1             | 10^0  | 10^0 |
| 0.2             | 10^-1 | 10^0 |
| 0.3             | 10^-2 | 10^0 |
| 0.4             | 10^-3 | 10^0 |
| 0.5             | 10^-4 | 10^0 |
| 0.6             | 10^-5 | 10^0 |
| 0.7             | 10^-6 | 10^0 |
| 0.8             | 10^-7 | 10^0 |
| 0.9             | 10^-8 | 10^0 |
| 1.0             | 10^-9 | 10^1 |
</details>

(c) Clustering coefficient

Figure 11: Structural properties across train and test parts in time-based split of OGB-Arxiv.   
![](images/01321461d1b24dc66841ba8d60c76ba914ab0247283dd7c93fafba1f94066b1a.jpg)

<details>
<summary>bar</summary>

| property values | train | test |
| :--- | :--- | :--- |
| 10^-14 | 10^6 | 10^5 |
| 10^-13 | 10^5 | 10^4 |
| 10^-12 | 10^4 | 10^3 |
| 10^-11 | 10^3 | 10^2 |
| 10^-10 | 10^2 | 10^1 |
| 10^-9 | 10^1 | 10^0 |
| 10^-8 | 10^0 | 10^-1 |
| 10^-7 | 10^-1 | 10^-2 |
| 10^-6 | 10^-2 | 10^-3 |
| 10^-5 | 10^-3 | 10^-4 |
| 10^-4 | 10^-4 | 10^-5 |
| 10^-3 | 10^-5 | 10^-6 |
| 10^-2 | 10^-6 | 10^-7 |
| 10^-1 | 10^-7 | 10^-8 |
| 10^0 | 10^-8 | 10^-9 |
| 10^1 | 10^-9 | 10^-10 |
| 10^2 | 10^-10 | 10^-11 |
| 10^3 | 10^-11 | 10^-12 |
| 10^4 | 10^-12 | 10^-13 |
| 10^5 | 10^-13 | 10^-14 |
| 10^6 | 10^-14 | 10^-15 |
| 10^7 | 10^-15 | 10^-16 |
| 10^8 | 10^-16 | 10^-17 |
| 10^9 | 10^-17 | 10^-18 |
| 10^10 | 10^-18 | 10^-19 |
| 10^11 | 10^-19 | 10^-20 |
| 10^12 | 10^-20 | 10^-21 |
| 10^13 | 10^-21 | 10^-22 |
| 10^14 | 10^-22 | 10^-23 |
| 10^15 | 10^-23 | 10^-24 |
| 10^16 | 10^-24 | 10^-25 |
| 10^17 | 10^-25 | 10^-26 |
| 10^18 | 10^-26 | 10^-27 |
| 10^19 | 10^-27 | 10^-28 |
| 10^20 | 10^-28 | 10^-29 |
| 10^21 | 10^-29 | 10^-30 |
| 10^22 | 10^-30 | 10^-31 |
| 10^23 | 10^-31 | 10^-32 |
| 10^24 | 10^-32 | 10^-33 |
| 10^25 | 10^-33 | 10^-34 |
| 10^26 | 10^-34 | 10^-35 |
| 10^27 | 10^-35 | 10^-36 |
| 10^28 | 10^-36 | 10^-37 |
| 10^29 | 10^-37 | 10^-38 |
| 10^30 | 10^-38 | 10^-39 |
| 10^31 | 10^-39 | 10^-40 |
| 2x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x3.5x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4x4.5x4x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x4.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x3.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5x2.5e-6 x - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -<fcel>The data is a histogram comparing the density distributions of property values for two datasets: train and test samples. The x-axis represents property values ranging from ~×× to ~×××, and the y-axis represents density on a logarithmic scale from ~××× to ~×××× (with some values not specified). The chart shows that the density of property values for both train and test samples increases as property values increase, indicating higher density in the dataset at larger property values, with the highest density observed in the train set at approximately ××× (≈×× × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × × = (××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××= (××××××××××××××××××××××××××××××××××××××××××××××××= (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = x + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) * (y = y + y) .<fcel>The data is a histogram showing the density distribution of property values for each dataset, where the x-axis represents property values and the y-axis represents density in log scale, based on the number of occurrences of each dataset within each bin. The chart displays the frequency distribution of property values for each dataset, with the legend indicating the same color coding for each dataset. The data is presented in a vertical format with a color palette and a color bar indicating the color range from light blue to dark red. The chart is created using a bar plot using a color scheme to compare the distribution of property values between train and test datasets across multiple categories, including a single category labeled 'test' and a single category labeled 'train'. The visual pattern indicates that the data is structured by the color scheme, but specific numerical labels are not provided in the image.
</details>

(a) PageRank

![](images/194e4b589cb0576b2c551b434433ec9bda83e1ff564502852f0b6eff8f52c047.jpg)

<details>
<summary>bar</summary>

| property values | train density | test density |
| :--- | :--- | :--- |
| 10^-14 | 10^0 | 10^0 |
| 10^-13 | 10^-1 | 10^-1 |
| 10^-12 | 10^-2 | 10^-2 |
| 10^-11 | 10^-3 | 10^-3 |
| 10^-10 | 10^-4 | 10^-4 |
| 10^-9 | 10^-5 | 10^-5 |
| 10^-8 | 10^-6 | 10^-6 |
| 10^-7 | 10^-7 | 10^-7 |
| 10^-6 | 10^-8 | 10^-8 |
</details>

(b) Personalized PageRank

![](images/6ce9e34aa61827dc679555f516fe566b27c33020c511724b0a87ca8f075dd491.jpg)

<details>
<summary>bar</summary>

| property values | train density | test density |
| --------------- | ------------- | ------------ |
| 0.0             | ~10^0         | ~10^0        |
| 0.1             | ~10^0         | ~10^-1       |
| 0.2             | ~10^0         | ~10^-1       |
| 0.3             | ~10^0         | ~10^-1       |
| 0.4             | ~10^0         | ~10^-1       |
| 0.5             | ~10^0         | ~10^-1       |
| 0.6             | ~10^0         | ~10^-1       |
| 0.7             | ~10^0         | ~10^-1       |
| 0.8             | ~10^0         | ~10^-1       |
| 0.9             | ~10^0         | ~10^-1       |
| 1.0             | ~10^0         | ~10^-1       |
</details>

(c) Clustering coefficient

Figure 12: Structural properties across train and test parts in rank-based split of OGB-Products.   
![](images/ca7c32f19f12e1b0403e829cd2a7ff77026f11414246fe748a24178355a7a532.jpg)

<details>
<summary>bar</summary>

| property values | train | test |
| :--- | :--- | :--- |
| 10^-13.0 | 0.5 | 0.2 |
| 10^-12.5 | 0.4 | 0.3 |
| 10^-12.0 | 0.3 | 0.4 |
| 10^-11.5 | 0.2 | 0.5 |
| 10^-11.0 | 0.1 | 0.6 |
| 10^-10.5 | 0.05 | 0.7 |
| 10^-10.0 | 0.02 | 0.8 |
| 10^-9.5 | 0.01 | 0.9 |
</details>

(a) PageRank

![](images/f28f276b055af33b372de6055197290a79a0777b880c06a275aa53a40942f14b.jpg)

<details>
<summary>bar</summary>

| property values | train density | test density |
| :--- | :--- | :--- |
| 10^-14 | 0.05 | 0.12 |
| 10^-13 | 0.06 | 0.13 |
| 10^-12 | 0.07 | 0.14 |
| 10^-11 | 0.08 | 0.15 |
| 10^-10 | 0.09 | 0.16 |
| 10^-9 | 0.10 | 0.17 |
| 10^-8 | 0.03 | 0.01 |
</details>

(b) Personalized PageRank

![](images/303ac1df58907639fae80cc0d5cbee06b6bb70ca122b92d9154847b9ccaf0de2.jpg)

<details>
<summary>bar_stacked</summary>

| property values | train | test |
| :--- | :--- | :--- |
| 0.0 | 0.5 | 0.1 |
| 0.1 | 0.3 | 1.2 |
| 0.2 | 0.4 | 1.8 |
| 0.3 | 0.6 | 1.5 |
| 0.4 | 0.7 | 1.0 |
| 0.5 | 0.8 | 0.6 |
| 0.6 | 0.9 | 1.1 |
| 0.7 | 0.7 | 0.5 |
| 0.8 | 0.6 | 0.2 |
| 0.9 | 0.5 | 0.3 |
| 1.0 | 0.8 | 0.7 |
</details>

(c) Clustering coefficient   
Figure 13: Structural properties across train and test parts in species-based split of OGB-Proteins.

# E Dataset characteristics

Characteristics of the datasets used in this paper are listed in Table 8.

Table 8: Description of the considered graph datasets for node classification task. 

<table><tr><td>Dataset</td><td># Nodes</td><td># Edges</td><td># Classes</td><td># Features</td></tr><tr><td>AmazonComputer</td><td>13,381</td><td>259,159</td><td>10</td><td>767</td></tr><tr><td>AmazonPhoto</td><td>7,484</td><td>126,530</td><td>8</td><td>745</td></tr><tr><td>CoauthorCS</td><td>18,333</td><td>163,788</td><td>15</td><td>6,805</td></tr><tr><td>CoauthorPhysics</td><td>34,493</td><td>495,924</td><td>5</td><td>8,415</td></tr><tr><td>CoraML</td><td>2,995</td><td>16,316</td><td>7</td><td>2,879</td></tr><tr><td>CiteSeer</td><td>3,327</td><td>4,732</td><td>6</td><td>3,703</td></tr><tr><td>PubMed</td><td>19,717</td><td>44,338</td><td>3</td><td>8,415</td></tr><tr><td>OGB-Products</td><td>2,449,029</td><td>61,859,140</td><td>47</td><td>100</td></tr></table>

# F Method configurations

For our main series of experiments with graph models in Sections 5.1 and 5.2, we consider a graph encoder based on three GCN convolutional layers [16]. On top of this graph encoder, we use a single linear layer as the task-specific head, which is used to predict the parameters of categorical distribution in classification tasks, the parameters of Dirichlet distribution for modeling uncertainty, etc. The hidden dimension of our baseline architecture is 256, and the dropout between hidden layers is p = 0.2. We exploit a standard Adam optimizer [14] with a learning rate of 0.0003 and a weight decay of 0.00001. For an additional series of experiments with an improved GNN architecture, which is discussed in Section 5.3, we reduce the number of graph convolutional layers from 3 to 2, replacing the first one with a pre-processing step based on linear layer, apply the skip-connections between graph convolutional layers, and replace the GCN graph convolution with SAGE [11].

Some of the considered methods for improving the OOD robustness, such as DANN and CORAL, exploit the notion of domains and require the knowledge about which nodes in the training set belong to which domain. Therefore, we need to define domains, which is not straightforward since our node properties are real-valued, not discrete. Thus, we discretize the values into k = 10 non-intersecting domains. After that, we assign a domain index to each node. These indices are treated as labels and used by DANN and CORAL during their training.

# G Detailed experimental results

In this section, we provide detailed experimental results.

- Tables 9 — 15 show the predictive performance of various OOD robustness methods on the proposed structural distributional shifts;   
- Tables 16 — 22 show the OOD detection performance of various uncertainty estimation methods on the proposed structural distributional shifts.

Table 9: Comparison of several graph methods for improving the OOD robustness in terms of their predictive performance across structural shifts on CoraML dataset. 

<table><tr><td rowspan="2"></td><td colspan="2">popularity</td><td colspan="2">locality</td><td colspan="2">density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td> $85.80 \pm 1.10$ </td><td> $82.42 \pm 0.79$ </td><td> $84.47 \pm 0.69$ </td><td> $72.13 \pm 1.74$ </td><td> $93.27 \pm 0.28$ </td><td> $77.33 \pm 1.07$ </td></tr><tr><td>Mixup</td><td> $89.27 \pm 0.57$ </td><td> $83.65 \pm 0.67$ </td><td> $86.27 \pm 0.57$ </td><td> $78.42 \pm 0.84$ </td><td> $88.20 \pm 0.44$ </td><td> $78.98 \pm 2.10$ </td></tr><tr><td>EERM</td><td> $88.40 \pm 1.22$ </td><td> $83.07 \pm 0.76$ </td><td> $88.80 \pm 0.73$ </td><td> $69.96 \pm 6.62$ </td><td> $86.87 \pm 0.45$ </td><td> $79.07 \pm 0.32$ </td></tr><tr><td>DANN</td><td> $86.00 \pm 1.10$ </td><td> $79.01 \pm 0.77$ </td><td> $81.45 \pm 0.65$ </td><td> $65.03 \pm 2.51$ </td><td> $91.33 \pm 0.76$ </td><td> $78.37 \pm 2.07$ </td></tr><tr><td>CORAL</td><td> $85.00 \pm 1.11$ </td><td> $77.65 \pm 0.62$ </td><td> $82.33 \pm 0.25$ </td><td> $66.83 \pm 2.24$ </td><td> $89.55 \pm 0.54$ </td><td> $73.34 \pm 2.28$ </td></tr><tr><td>DE</td><td> $87.00 \pm 0.00$ </td><td> $82.74 \pm 0.00$ </td><td> $84.67 \pm 0.00$ </td><td> $75.40 \pm 0.00$ </td><td> $93.00 \pm 0.00$ </td><td> $78.90 \pm 0.00$ </td></tr><tr><td>ERM + mod</td><td> $85.33 \pm 0.62$ </td><td> $82.55 \pm 0.56$ </td><td> $84.87 \pm 0.61$ </td><td> $72.59 \pm 0.96$ </td><td> $94.13 \pm 0.69$ </td><td> $79.33 \pm 0.37$ </td></tr></table>

Table 10: Comparison of several graph methods for improving the OOD robustness in terms of their predictive performance across structural shifts on CiteSeer dataset. 

<table><tr><td rowspan="2"></td><td colspan="2">popularity</td><td colspan="2">locality</td><td colspan="2">density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td> $72.43 \pm 1.33$ </td><td> $72.42 \pm 0.37$ </td><td> $77.60 \pm 0.66$ </td><td> $57.03 \pm 1.16$ </td><td> $73.75 \pm 0.96$ </td><td> $67.57 \pm 0.49$ </td></tr><tr><td>Mixup</td><td> $72.79 \pm 0.74$ </td><td> $72.13 \pm 0.69$ </td><td> $76.82 \pm 0.59$ </td><td> $56.45 \pm 4.53$ </td><td> $76.88 \pm 1.47$ </td><td> $68.59 \pm 2.12$ </td></tr><tr><td>EERM</td><td> $71.47 \pm 1.01$ </td><td> $70.48 \pm 0.66$ </td><td> $75.31 \pm 0.80$ </td><td> $61.95 \pm 3.30$ </td><td> $76.46 \pm 0.92$ </td><td> $68.18 \pm 0.71$ </td></tr><tr><td>DANN</td><td> $67.57 \pm 0.96$ </td><td> $67.84 \pm 0.65$ </td><td> $71.17 \pm 0.33$ </td><td> $54.90 \pm 2.49$ </td><td> $75.07 \pm 1.48$ </td><td> $61.13 \pm 2.82$ </td></tr><tr><td>CORAL</td><td> $67.87 \pm 1.19$ </td><td> $67.77 \pm 0.38$ </td><td> $71.87 \pm 0.62$ </td><td> $57.16 \pm 1.84$ </td><td> $73.47 \pm 1.22$ </td><td> $58.53 \pm 2.02$ </td></tr><tr><td>DE</td><td> $73.27 \pm 0.00$ </td><td> $72.37 \pm 0.00$ </td><td> $78.38 \pm 0.00$ </td><td> $64.71 \pm 0.00$ </td><td> $74.17 \pm 0.00$ </td><td> $70.35 \pm 0.00$ </td></tr><tr><td>ERM + mod</td><td> $73.75 \pm 0.62$ </td><td> $71.73 \pm 1.02$ </td><td> $76.70 \pm 1.14$ </td><td> $59.86 \pm 2.23$ </td><td> $75.80 \pm 0.96$ </td><td> $68.32 \pm 1.01$ </td></tr></table>

Table 11: Comparison of several graph methods for improving the OOD robustness in terms of their predictive performance across structural shifts on PubMed dataset. 

<table><tr><td rowspan="2"></td><td colspan="2">popularity</td><td colspan="2">locality</td><td colspan="2">density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td> $86.85 \pm 0.12$ </td><td> $84.04 \pm 0.33$ </td><td> $86.75 \pm 0.48$ </td><td> $81.80 \pm 1.07$ </td><td> $85.96 \pm 0.27$ </td><td> $85.29 \pm 0.16$ </td></tr><tr><td>Mixup</td><td> $89.32 \pm 0.24$ </td><td> $88.21 \pm 0.21$ </td><td> $87.94 \pm 0.37$ </td><td> $86.20 \pm 0.75$ </td><td> $88.72 \pm 0.25$ </td><td> $88.23 \pm 0.16$ </td></tr><tr><td>EERM</td><td> $86.83 \pm 0.12$ </td><td> $83.39 \pm 0.10$ </td><td> $85.67 \pm 0.24$ </td><td> $84.79 \pm 0.18$ </td><td> $86.43 \pm 0.22$ </td><td> $84.10 \pm 0.10$ </td></tr><tr><td>DANN</td><td> $86.26 \pm 0.48$ </td><td> $84.49 \pm 0.18$ </td><td> $86.04 \pm 0.30$ </td><td> $85.41 \pm 1.04$ </td><td> $87.07 \pm 0.24$ </td><td> $84.74 \pm 0.14$ </td></tr><tr><td>CORAL</td><td> $86.31 \pm 0.48$ </td><td> $84.48 \pm 0.37$ </td><td> $86.00 \pm 0.32$ </td><td> $85.36 \pm 1.63$ </td><td> $87.24 \pm 0.15$ </td><td> $84.86 \pm 0.27$ </td></tr><tr><td>DE</td><td> $87.17 \pm 0.00$ </td><td> $84.72 \pm 0.00$ </td><td> $87.07 \pm 0.00$ </td><td> $82.82 \pm 0.00$ </td><td> $86.61 \pm 0.00$ </td><td> $85.97 \pm 0.00$ </td></tr><tr><td>ERM + mod</td><td> $88.83 \pm 0.24$ </td><td> $87.39 \pm 0.27$ </td><td> $88.32 \pm 0.51$ </td><td> $87.58 \pm 0.17$ </td><td> $88.56 \pm 0.43$ </td><td> $87.71 \pm 0.22$ </td></tr></table>

Table 12: Comparison of several graph methods for improving the OOD robustness in terms of their predictive performance across structural shifts on AmazonComputer dataset. 

<table><tr><td rowspan="2"></td><td colspan="2">popularity</td><td colspan="2">locality</td><td colspan="2">density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td> $92.43 \pm 0.24$ </td><td> $83.18 \pm 0.58$ </td><td> $93.04 \pm 0.38$ </td><td> $72.62 \pm 0.52$ </td><td> $93.10 \pm 0.27$ </td><td> $85.73 \pm 0.60$ </td></tr><tr><td>Mixup</td><td> $92.27 \pm 0.19$ </td><td> $66.80 \pm 2.84$ </td><td> $93.76 \pm 0.19$ </td><td> $66.94 \pm 2.00$ </td><td> $92.46 \pm 0.44$ </td><td> $74.72 \pm 1.09$ </td></tr><tr><td>EERM</td><td> $89.49 \pm 1.55$ </td><td> $82.08 \pm 0.61$ </td><td> $92.22 \pm 0.37$ </td><td> $61.39 \pm 0.85$ </td><td> $88.09 \pm 0.55$ </td><td> $79.98 \pm 1.28$ </td></tr><tr><td>DANN</td><td> $91.25 \pm 0.37$ </td><td> $83.39 \pm 1.87$ </td><td> $92.30 \pm 0.77$ </td><td> $68.84 \pm 2.53$ </td><td> $91.84 \pm 0.79$ </td><td> $83.89 \pm 1.37$ </td></tr><tr><td>CORAL</td><td> $91.25 \pm 0.54$ </td><td> $84.77 \pm 1.48$ </td><td> $92.47 \pm 0.18$ </td><td> $62.36 \pm 1.87$ </td><td> $91.40 \pm 0.91$ </td><td> $85.35 \pm 1.58$ </td></tr><tr><td>DE</td><td> $92.51 \pm 0.00$ </td><td> $84.17 \pm 0.00$ </td><td> $93.31 \pm 0.00$ </td><td> $73.93 \pm 0.00$ </td><td> $93.24 \pm 0.00$ </td><td> $86.93 \pm 0.00$ </td></tr><tr><td>ERM + mod</td><td> $90.99 \pm 0.16$ </td><td> $85.72 \pm 0.34$ </td><td> $91.74 \pm 0.24$ </td><td> $63.67 \pm 0.35$ </td><td> $92.40 \pm 0.34$ </td><td> $86.17 \pm 0.20$ </td></tr></table>

Table 13: Comparison of several graph methods for improving the OOD robustness in terms of their predictive performance across structural shifts on AmazonPhoto dataset. 

<table><tr><td rowspan="2"></td><td colspan="2">popularity</td><td colspan="2">locality</td><td colspan="2">density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td> $95.82 \pm 0.24$ </td><td> $87.63 \pm 0.65$ </td><td> $93.73 \pm 0.29$ </td><td> $64.73 \pm 3.80$ </td><td> $94.64 \pm 0.38$ </td><td> $91.25 \pm 0.16$ </td></tr><tr><td>Mixup</td><td> $97.99 \pm 0.36$ </td><td> $87.73 \pm 2.39$ </td><td> $95.55 \pm 0.30$ </td><td> $75.36 \pm 2.01$ </td><td> $95.50 \pm 0.41$ </td><td> $90.71 \pm 2.13$ </td></tr><tr><td>EERM</td><td> $95.90 \pm 0.40$ </td><td> $86.84 \pm 1.02$ </td><td> $91.95 \pm 0.21$ </td><td> $54.06 \pm 0.81$ </td><td> $90.46 \pm 0.38$ </td><td> $88.91 \pm 0.31$ </td></tr><tr><td>DANN</td><td> $95.25 \pm 0.26$ </td><td> $85.23 \pm 0.78$ </td><td> $91.81 \pm 0.39$ </td><td> $48.79 \pm 5.04$ </td><td> $93.99 \pm 0.66$ </td><td> $91.87 \pm 0.46$ </td></tr><tr><td>CORAL</td><td> $95.34 \pm 0.27$ </td><td> $85.91 \pm 0.67$ </td><td> $91.85 \pm 0.51$ </td><td> $45.70 \pm 3.34$ </td><td> $93.64 \pm 0.80$ </td><td> $91.99 \pm 0.43$ </td></tr><tr><td>DE</td><td> $95.82 \pm 0.00$ </td><td> $88.82 \pm 0.00$ </td><td> $93.59 \pm 0.00$ </td><td> $69.15 \pm 0.00$ </td><td> $94.38 \pm 0.00$ </td><td> $92.09 \pm 0.00$ </td></tr><tr><td>ERM + mod</td><td> $97.07 \pm 0.35$ </td><td> $89.95 \pm 0.20$ </td><td> $95.01 \pm 0.43$ </td><td> $57.41 \pm 1.79$ </td><td> $95.22 \pm 0.33$ </td><td> $91.97 \pm 0.33$ </td></tr></table>

Table 14: Comparison of several graph methods for improving the OOD robustness in terms of their predictive performance across structural shifts on CoauthorCS dataset. 

<table><tr><td rowspan="2"></td><td colspan="2">popularity</td><td colspan="2">locality</td><td colspan="2">density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td> $93.60 \pm 0.18$ </td><td> $90.67 \pm 0.20$ </td><td> $92.34 \pm 0.14$ </td><td> $91.22 \pm 0.35$ </td><td> $94.30 \pm 0.17$ </td><td> $89.71 \pm 0.21$ </td></tr><tr><td>Mixup</td><td> $94.97 \pm 0.33$ </td><td> $94.05 \pm 0.38$ </td><td> $94.05 \pm 0.13$ </td><td> $91.48 \pm 0.16$ </td><td> $94.97 \pm 0.29$ </td><td> $92.11 \pm 0.15$ </td></tr><tr><td>EERM</td><td> $93.82 \pm 0.08$ </td><td> $91.01 \pm 0.15$ </td><td> $91.22 \pm 0.27$ </td><td> $91.95 \pm 0.21$ </td><td> $93.64 \pm 0.23$ </td><td> $88.42 \pm 0.15$ </td></tr><tr><td>DANN</td><td> $93.89 \pm 0.26$ </td><td> $90.59 \pm 0.12$ </td><td> $91.66 \pm 0.37$ </td><td> $91.65 \pm 1.29$ </td><td> $94.60 \pm 0.16$ </td><td> $89.92 \pm 0.29$ </td></tr><tr><td>CORAL</td><td> $93.80 \pm 0.14$ </td><td> $90.78 \pm 0.13$ </td><td> $91.69 \pm 0.23$ </td><td> $91.37 \pm 0.48$ </td><td> $94.42 \pm 0.07$ </td><td> $89.72 \pm 0.22$ </td></tr><tr><td>DE</td><td> $93.68 \pm 0.00$ </td><td> $90.93 \pm 0.00$ </td><td> $92.64 \pm 0.00$ </td><td> $91.76 \pm 0.00$ </td><td> $94.55 \pm 0.00$ </td><td> $90.05 \pm 0.00$ </td></tr><tr><td>ERM + mod</td><td> $95.35 \pm 0.15$ </td><td> $95.00 \pm 0.06$ </td><td> $94.77 \pm 0.09$ </td><td> $94.06 \pm 0.11$ </td><td> $96.67 \pm 0.11$ </td><td> $93.64 \pm 0.16$ </td></tr></table>

Table 15: Comparison of several graph methods for improving the OOD robustness in terms of their predictive performance across structural shifts on CoauthorPhysics dataset. 

<table><tr><td rowspan="2"></td><td colspan="2">popularity</td><td colspan="2">locality</td><td colspan="2">density</td></tr><tr><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td><td>ID</td><td>OOD</td></tr><tr><td>ERM</td><td> $93.60 \pm 0.18$ </td><td> $90.67 \pm 0.20$ </td><td> $92.34 \pm 0.14$ </td><td> $91.22 \pm 0.35$ </td><td> $94.30 \pm 0.17$ </td><td> $89.71 \pm 0.21$ </td></tr><tr><td>Mixup</td><td> $96.96 \pm 0.11$ </td><td> $94.05 \pm 0.74$ </td><td> $97.27 \pm 0.11$ </td><td> $93.39 \pm 0.95$ </td><td> $96.80 \pm 0.08$ </td><td> $84.57 \pm 0.85$ </td></tr><tr><td>EERM</td><td> $95.93 \pm 0.10$ </td><td> $93.37 \pm 0.08$ </td><td> $94.04 \pm 0.16$ </td><td> $92.32 \pm 0.11$ </td><td> $95.26 \pm 0.03$ </td><td> $93.98 \pm 0.06$ </td></tr><tr><td>DANN</td><td> $96.56 \pm 0.14$ </td><td> $93.73 \pm 0.25$ </td><td> $96.64 \pm 0.23$ </td><td> $91.35 \pm 1.12$ </td><td> $96.08 \pm 0.17$ </td><td> $95.05 \pm 0.11$ </td></tr><tr><td>CORAL</td><td> $96.51 \pm 0.10$ </td><td> $93.58 \pm 0.29$ </td><td> $96.93 \pm 0.08$ </td><td> $86.21 \pm 0.86$ </td><td> $95.97 \pm 0.09$ </td><td> $95.00 \pm 0.17$ </td></tr><tr><td>DE</td><td> $93.68 \pm 0.00$ </td><td> $90.93 \pm 0.00$ </td><td> $92.64 \pm 0.00$ </td><td> $91.76 \pm 0.00$ </td><td> $94.55 \pm 0.00$ </td><td> $90.05 \pm 0.00$ </td></tr><tr><td>ERM + mod</td><td> $95.35 \pm 0.15$ </td><td> $95.00 \pm 0.06$ </td><td> $94.77 \pm 0.09$ </td><td> $94.06 \pm 0.11$ </td><td> $96.67 \pm 0.11$ </td><td> $93.64 \pm 0.16$ </td></tr></table>

Table 16: Comparison of several graph methods for uncertainty estimation in terms of their OOD detection performance across structural shifts on CoraML dataset. 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>75.67 ± 0.85</td><td>87.13 ± 0.74</td><td>81.55 ± 0.45</td></tr><tr><td>NatPN</td><td>47.31 ± 9.44</td><td>81.66 ± 3.05</td><td>69.05 ± 3.05</td></tr><tr><td>GPN</td><td>66.95 ± 1.15</td><td>73.09 ± 2.21</td><td>70.95 ± 1.58</td></tr><tr><td>DE</td><td>70.55 ± 0.00</td><td>93.32 ± 0.00</td><td>84.46 ± 0.00</td></tr><tr><td>NatPE</td><td>37.59 ± 0.00</td><td>78.48 ± 0.00</td><td>65.83 ± 0.00</td></tr><tr><td>GPE</td><td>65.67 ± 0.00</td><td>74.23 ± 0.00</td><td>71.07 ± 0.00</td></tr><tr><td>SE + mod</td><td>55.66 ± 0.25</td><td>72.97 ± 0.71</td><td>68.27 ± 0.58</td></tr></table>

Table 17: Comparison of several graph methods for uncertainty estimation in terms of their OOD detection performance across structural shifts on CiteSeer dataset. 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>68.01 ± 1.23</td><td>89.89 ± 0.56</td><td>66.90 ± 0.41</td></tr><tr><td>NatPN</td><td>31.18 ± 1.54</td><td>91.96 ± 3.14</td><td>59.77 ± 1.54</td></tr><tr><td>GPN</td><td>61.99 ± 1.74</td><td>69.53 ± 5.10</td><td>57.41 ± 1.72</td></tr><tr><td>DE</td><td>56.22 ± 0.00</td><td>98.18 ± 0.00</td><td>70.48 ± 0.00</td></tr><tr><td>NatPE</td><td>28.07 ± 0.00</td><td>89.02 ± 0.00</td><td>58.10 ± 0.00</td></tr><tr><td>GPE</td><td>63.01 ± 0.00</td><td>73.89 ± 0.00</td><td>57.94 ± 0.00</td></tr><tr><td>SE + mod</td><td>54.36 ± 0.56</td><td>78.74 ± 0.55</td><td>61.03 ± 0.73</td></tr></table>

Table 18: Comparison of several graph methods for uncertainty estimation in terms of their OOD detection performance across structural shifts on PubMed dataset. 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>68.60 ± 0.34</td><td>66.34 ± 1.03</td><td>58.60 ± 0.39</td></tr><tr><td>NatPN</td><td>49.61 ± 3.17</td><td>58.91 ± 2.89</td><td>51.00 ± 1.45</td></tr><tr><td>GPN</td><td>72.30 ± 0.29</td><td>69.63 ± 4.84</td><td>62.04 ± 0.68</td></tr><tr><td>DE</td><td>74.31 ± 0.00</td><td>72.19 ± 0.00</td><td>63.04 ± 0.00</td></tr><tr><td>NatPE</td><td>46.94 ± 0.00</td><td>56.41 ± 0.00</td><td>49.54 ± 0.00</td></tr><tr><td>GPE</td><td>72.62 ± 0.00</td><td>66.07 ± 0.00</td><td>62.58 ± 0.00</td></tr><tr><td>SE + mod</td><td>53.68 ± 0.12</td><td>54.86 ± 0.72</td><td>52.25 ± 0.13</td></tr></table>

Table 19: Comparison of several graph methods for uncertainty estimation in terms of their OOD detection performance across structural shifts on AmazonComputer dataset. 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>88.52 ± 0.37</td><td>86.48 ± 0.80</td><td>44.24 ± 0.57</td></tr><tr><td>NatPN</td><td>53.83 ± 15.71</td><td>59.92 ± 6.59</td><td>44.42 ± 3.44</td></tr><tr><td>GPN</td><td>81.23 ± 1.44</td><td>82.84 ± 2.56</td><td>43.92 ± 2.56</td></tr><tr><td>DE</td><td>82.53 ± 0.00</td><td>83.54 ± 0.00</td><td>50.88 ± 0.00</td></tr><tr><td>NatPE</td><td>41.33 ± 0.00</td><td>51.93 ± 0.00</td><td>42.09 ± 0.00</td></tr><tr><td>GPE</td><td>81.35 ± 0.00</td><td>81.97 ± 0.00</td><td>43.81 ± 0.00</td></tr><tr><td>SE + mod</td><td>60.54 ± 0.41</td><td>74.09 ± 0.64</td><td>59.27 ± 0.44</td></tr></table>

Table 20: Comparison of several graph methods for uncertainty estimation in terms of their OOD detection performance across structural shifts on AmazonPhoto dataset. 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>92.05 ± 0.27</td><td>93.29 ± 0.60</td><td>41.08 ± 1.89</td></tr><tr><td>NatPN</td><td>79.72 ± 7.29</td><td>65.66 ± 11.98</td><td>54.88 ± 17.40</td></tr><tr><td>GPN</td><td>80.85 ± 2.64</td><td>90.90 ± 2.82</td><td>49.97 ± 2.24</td></tr><tr><td>DE</td><td>90.72 ± 0.00</td><td>96.66 ± 0.00</td><td>51.33 ± 0.00</td></tr><tr><td>NatPE</td><td>80.43 ± 0.00</td><td>59.74 ± 0.00</td><td>64.69 ± 0.00</td></tr><tr><td>GPE</td><td>82.95 ± 0.00</td><td>91.61 ± 0.00</td><td>51.74 ± 0.00</td></tr><tr><td>SE + mod</td><td>66.93 ± 0.47</td><td>83.09 ± 1.57</td><td>51.36 ± 0.94</td></tr></table>

Table 21: Comparison of several graph methods for uncertainty estimation in terms of their OOD detection performance across structural shifts on CoauthorCS dataset. 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>83.25 ± 0.40</td><td>85.74 ± 1.06</td><td>50.91 ± 0.26</td></tr><tr><td>NatPN</td><td>59.99 ± 5.57</td><td>64.53 ± 12.82</td><td>58.47 ± 8.22</td></tr><tr><td>GPN</td><td>63.43 ± 0.46</td><td>64.84 ± 1.57</td><td>56.81 ± 0.54</td></tr><tr><td>DE</td><td>82.50 ± 0.00</td><td>87.54 ± 0.00</td><td>54.47 ± 0.00</td></tr><tr><td>NatPE</td><td>61.00 ± 0.00</td><td>60.55 ± 0.00</td><td>62.63 ± 0.00</td></tr><tr><td>GPE</td><td>62.12 ± 0.00</td><td>63.76 ± 0.00</td><td>57.15 ± 0.00</td></tr><tr><td>SE + mod</td><td>58.05 ± 0.71</td><td>64.43 ± 1.40</td><td>56.99 ± 0.26</td></tr></table>

Table 22: Comparison of several graph methods for uncertainty estimation in terms of their OOD detection performance across structural shifts on CoauthorPhysics dataset. 

<table><tr><td></td><td>Popularity</td><td>Locality</td><td>Density</td></tr><tr><td>SE</td><td>86.60 ± 0.29</td><td>87.73 ± 0.51</td><td>37.50 ± 0.32</td></tr><tr><td>NatPN</td><td>56.74 ± 3.68</td><td>59.54 ± 23.75</td><td>58.81 ± 5.96</td></tr><tr><td>GPN</td><td>67.48 ± 0.45</td><td>74.05 ± 2.67</td><td>50.99 ± 0.53</td></tr><tr><td>DE</td><td>85.94 ± 0.00</td><td>91.93 ± 0.00</td><td>36.15 ± 0.00</td></tr><tr><td>NatPE</td><td>56.11 ± 0.00</td><td>46.14 ± 0.00</td><td>56.94 ± 0.00</td></tr><tr><td>GPE</td><td>66.29 ± 0.00</td><td>72.16 ± 0.00</td><td>51.89 ± 0.00</td></tr><tr><td>SE + mod</td><td>64.84 ± 0.73</td><td>88.02 ± 0.31</td><td>44.13 ± 0.31</td></tr></table>

# H Visualization of distributional shifts in graph domain

In Figures 14–20, we provide the visualizations of different structural shifts in graph domain for all the considered graph datasets: ID is blue, OOD is red. Some graphs have multiple connected components — in that case, we keep only the largest one.

![](images/a2b034d818e705a38d6eb4a858c7ddc103aea0525c232a9d3c8365ce378aac6d.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(a) Popularity

![](images/b0d1303f02ed6319aaa61affd38c8a842e5e2b75c0f7ebd3fe378ba9ae899eba.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(b) Locality

![](images/6165a7b2eaa64afcc8510645f0ccea92498793035f6e5062f1608345b0105f83.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | A |
| 0.2 | 0.85 | B |
| 0.3 | 0.75 | A |
| 0.4 | 0.65 | B |
| 0.5 | 0.55 | A |
| 0.6 | 0.45 | B |
| 0.7 | 0.35 | A |
| 0.8 | 0.25 | B |
| 0.9 | 0.15 | A |
| 1.0 | 0.05 | B |
</details>

(c) Density

Figure 14: Visualization of structural shifts in graph domain for CoraML dataset.   
![](images/4b5f1009e871c2d5ab2b3435f7a8761f73f6db2f9d2a17f485f384380ed75b80.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(a) Popularity

![](images/62fa43f6b2aa932094ca70817c8af4967692feb6d4282e6f33dcfd370654e887.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | A |
| 0.2 | 0.88 | A |
| 0.3 | 0.75 | A |
| 0.4 | 0.62 | A |
| 0.5 | 0.55 | A |
| 0.6 | 0.48 | A |
| 0.7 | 0.42 | A |
| 0.8 | 0.38 | A |
| 0.9 | 0.35 | A |
| 1.0 | 0.32 | A |
| 0.15 | 0.85 | B |
| 0.25 | 0.78 | B |
| 0.35 | 0.65 | B |
| 0.45 | 0.58 | B |
| 0.55 | 0.52 | B |
| 0.65 | 0.48 | B |
| 0.75 | 0.45 | B |
| 0.85 | 0.42 | B |
| 0.95 | 0.40 | B |
| 1.05 | 0.38 | B |
</details>

(b) Locality

![](images/5689b5366e6c528a1f4273b5141508e938509ece2e89bb974563e082c5cb786f.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | A |
| 0.2 | 0.85 | B |
| 0.3 | 0.75 | A |
| 0.4 | 0.65 | B |
| 0.5 | 0.55 | A |
| 0.6 | 0.45 | B |
| 0.7 | 0.35 | A |
| 0.8 | 0.25 | B |
| 0.9 | 0.15 | A |
| 1.0 | 0.05 | B |
</details>

(c) Density

Figure 15: Visualization of structural shifts in graph domain for CiteSeer dataset.   
![](images/8f0526a14d37f97c692185333185c04c122fcf191835c9385fe50601d4987966.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(a) Popularity

![](images/d43f396925c83f92ab1890d48b4be67ae9f7f07b612d3351e6f636cbd35ab843.jpg)

<details>
<summary>scatter</summary>

| x | y |
| --- | --- |
| (data not extractable) | (data not extractable) |
</details>

(b) Locality

![](images/81951b7b867ba740cd72860a48dcfac7eae71c8907adc3cda8ce0b9f9c414c39.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| (data points not extractable as discrete values) | (data points not extractable as discrete values) |
</details>

(c) Density   
Figure 16: Visualization of structural shifts in graph domain for PubMed dataset.

![](images/2217a6f3b45fd9c269796ffb2cf0b3fcaee80625f0828f4460b3d11eef544c3a.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | A |
| 0.2 | 0.85 | B |
| 0.3 | 0.75 | A |
| 0.4 | 0.65 | B |
| 0.5 | 0.55 | A |
| 0.6 | 0.45 | B |
| 0.7 | 0.35 | A |
| 0.8 | 0.25 | B |
| 0.9 | 0.15 | A |
| 1.0 | 0.05 | B |
</details>

(a) Popularity

![](images/03f259a3dc5ecac3a9c2675e98f0c24fae178b28277f09f6cc1f85613fd7e79d.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| 0.1 | 0.9 |
| 0.2 | 0.85 |
| 0.3 | 0.75 |
| 0.4 | 0.65 |
| 0.5 | 0.55 |
| 0.6 | 0.45 |
| 0.7 | 0.35 |
| 0.8 | 0.25 |
| 0.9 | 0.15 |
| 1.0 | 0.05 |
</details>

(b) Locality

![](images/e4d6f63c5089e09bf90c8199978c97e746e6e333be485b6da70ae915226beb5e.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | A |
| 0.2 | 0.85 | A |
| 0.3 | 0.75 | A |
| 0.4 | 0.65 | A |
| 0.5 | 0.55 | A |
| 0.6 | 0.45 | A |
| 0.7 | 0.35 | A |
| 0.8 | 0.25 | A |
| 0.9 | 0.15 | A |
| 0.15 | 0.88 | B |
| 0.25 | 0.78 | B |
| 0.35 | 0.68 | B |
| 0.45 | 0.58 | B |
| 0.55 | 0.48 | B |
| 0.65 | 0.38 | B |
| 0.75 | 0.28 | B |
| 0.85 | 0.18 | B |
| 0.95 | 0.08 | B |
</details>

(c) Density   
Figure 17: Visualization of structural shifts in graph domain for AmazonPhoto dataset.

![](images/1ad596a3779339555b1ec4ff5941ee5341206d6089b747b12ee90eb72c6d3378.jpg)

<details>
<summary>scatter</summary>

| x | y |
| --- | --- |
| (data not extractable) | (data not extractable) |
</details>

(a) Popularity

![](images/5ad5715d297d38d86e7ff416d516a6fb108d0629747e93a93f2134d92302473b.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| 0.1 | 0.9 |
| 0.2 | 0.85 |
| 0.3 | 0.75 |
| 0.4 | 0.65 |
| 0.5 | 0.55 |
| 0.6 | 0.45 |
| 0.7 | 0.35 |
| 0.8 | 0.25 |
| 0.9 | 0.15 |
| 1.0 | 0.05 |
</details>

(b) Locality

![](images/755ab2d7f7a06c5ebed7315bd001af8c6a8aab897b6e62269eaee88443629cf1.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(c) Density   
Figure 18: Visualization of structural shifts in graph domain for AmazonComputer dataset.

![](images/d2d6662d07d8a7c83c34c586e97ff477f788e1630b2f7fa36185aadfd2bec18d.jpg)

<details>
<summary>natural_image</summary>

Heart-shaped arrangement of red dots forming a heart shape, with no text or symbols present.
</details>

(a) Popularity

![](images/ecaaa90865cfb3fb3913bd022c5c0856e3054d32ea376cb28b74451e149f8d08.jpg)

<details>
<summary>natural_image</summary>

Abstract heart-shaped cluster of red dots forming a ring, with a light blue center (no text or symbols)
</details>

(b) Locality

![](images/937fb97304ebb709e44f17574e7ee347e2207c1c992a9a241c662e060745e8d9.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| (data not extractable) | (data not extractable) |
</details>

(c) Density   
Figure 19: Visualization of structural shifts in graph domain for CoauthorCS dataset.

![](images/f567dce2f17ef8022877e54ac20ccc48653f1f9e5f53041cbf310ddab28d0012.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| 0.0 | 0.0 |
| 0.1 | 0.1 |
| 0.2 | 0.2 |
| 0.3 | 0.3 |
| 0.4 | 0.4 |
| 0.5 | 0.5 |
| 0.6 | 0.6 |
| 0.7 | 0.7 |
| 0.8 | 0.8 |
| 0.9 | 0.9 |
| 1.0 | 1.0 |
</details>

(a) Popularity

![](images/02f2ed222768b794337414eec2e6c48c40e56ea127a0b8b1f3155620d3105412.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| (data points not extractable as discrete values) | (data points not extractable as discrete values) |
</details>

(b) Locality

![](images/c50eeb3a36f11d61a64fa8b73e8e693230fc986bd84ab6d8ccd108ea8bbb5f69.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | A |
| 0.2 | 0.85 | A |
| 0.3 | 0.75 | A |
| 0.4 | 0.65 | A |
| 0.5 | 0.55 | A |
| 0.6 | 0.45 | A |
| 0.7 | 0.35 | A |
| 0.8 | 0.25 | A |
| 0.9 | 0.15 | A |
| 0.15 | 0.88 | B |
| 0.25 | 0.78 | B |
| 0.35 | 0.68 | B |
| 0.45 | 0.58 | B |
| 0.55 | 0.48 | B |
| 0.65 | 0.38 | B |
| 0.75 | 0.28 | B |
| 0.85 | 0.18 | B |
| 0.95 | 0.08 | B |
</details>

(c) Density   
Figure 20: Visualization of structural shifts in graph domain for CoauthorPhysics dataset.

# I Visualization of distributional shifts in node feature space

In Figures 21–27, we provide the visualizations of different structural shifts in node feature space for all the considered graph datasets: ID is blue, OOD is red. The pictures are obtained by reducing the original node feature space into the 2D space of t-SNE representations.

![](images/e17f6c86aa10b5b81df1301458210287f23fd410da9af14d2383cbf3b04b6638.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(a) Popularity

![](images/603128c96824d149ecdd74de9e948b95469477710167d145b87a6af1227525df.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | red |
| 0.2 | 0.85 | blue |
| 0.3 | 0.75 | red |
| 0.4 | 0.65 | blue |
| 0.5 | 0.55 | red |
| 0.6 | 0.45 | blue |
| 0.7 | 0.35 | red |
| 0.8 | 0.25 | blue |
| 0.9 | 0.15 | red |
| 1.0 | 0.05 | blue |
</details>

(b) Locality

![](images/c8515cc3c8516e33c6be6af966ba813b2c5cae81df1ce01efa149fb7e48c72e5.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(c) Density

Figure 21: Visualization of structural shifts in node feature space for CoraML dataset.   
![](images/28ebd1bc4ca384f4928a3bb6ecd057cfe757bf3527b2ee36d6c2627bf4be1a17.jpg)

<details>
<summary>scatter</summary>

| x | y | color |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | (data not extractable) |
</details>

(a) Popularity

![](images/0a98e1a07d158da7b8214a67f8ef46d71608af8be9c479ac96b97ffa63b1f3c8.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | red |
| 0.2 | 0.85 | blue |
| 0.3 | 0.75 | red |
| 0.4 | 0.65 | blue |
| 0.5 | 0.55 | red |
| 0.6 | 0.45 | blue |
| 0.7 | 0.35 | red |
| 0.8 | 0.25 | blue |
| 0.9 | 0.15 | red |
| 1.0 | 0.05 | blue |
</details>

(b) Locality

![](images/fb27079fbb00c008aba5769270385408601b7735bea9e2acad0211733890d1db.jpg)

<details>
<summary>scatter</summary>

| x | y | color |
|---|---|---|
| 0.1 | 0.95 | red |
| 0.2 | 0.85 | blue |
| 0.3 | 0.75 | red |
| 0.4 | 0.65 | blue |
| 0.5 | 0.55 | red |
| 0.6 | 0.45 | blue |
| 0.7 | 0.35 | red |
| 0.8 | 0.25 | blue |
| 0.9 | 0.15 | red |
| 1.0 | 0.05 | blue |
</details>

(c) Density

Figure 22: Visualization of structural shifts in node feature space for CiteSeer dataset.   
![](images/0889d9edf0c8e26710c7eb4aa087cb2d4a40ad83ea8642b3f305031bb93ca886.jpg)

<details>
<summary>bubble</summary>

| X | Y | Size |
|---|---|---|
| 0.1 | 0.95 | 100 |
| 0.2 | 0.85 | 100 |
| 0.3 | 0.75 | 100 |
| 0.4 | 0.65 | 100 |
| 0.5 | 0.55 | 100 |
| 0.6 | 0.45 | 100 |
| 0.7 | 0.35 | 100 |
| 0.8 | 0.25 | 100 |
| 0.9 | 0.15 | 100 |
| 1.0 | 0.05 | 100 |
| 1.1 | 0.98 | 100 |
| 1.2 | 0.88 | 100 |
| 1.3 | 0.78 | 100 |
| 1.4 | 0.68 | 100 |
| 1.5 | 0.58 | 100 |
| 1.6 | 0.48 | 100 |
| 1.7 | 0.38 | 100 |
| 1.8 | 0.28 | 100 |
| 1.9 | 0.18 | 100 |
| 2.0 | 0.92 | 100 |
| 2.1 | 0.82 | 100 |
| 2.2 | 0.72 | 100 |
| 2.3 | 0.62 | 100 |
| 2.4 | 0.52 | 100 |
| 2.5 | 0.42 | 100 |
| 2.6 | 0.32 | 100 |
| 2.7 | 0.22 | 100 |
| 2.8 | 0.12 | 100 |
| 2.9 | 0.96 | 100 |
| 3.0 | 0.86 | 100 |
| 3.1 | 0.76 | 100 |
| 3.2 | 0.66 | 100 |
| 3.3 | 0.56 | 100 |
| 3.4 | 0.46 | 100 |
| 3.5 | 0.36 | 100 |
| 3.6 | 0.26 | 100 |
| 3.7 | 0.16 | 100 |
| 3.8 | 0.94 | 100 |
| 3.9 | 0.84 | 100 |
| 4.0 | 0.74 | 100 |
| 4.1 | 0.64 | 100 |
| 4.2 | 0.54 | 100 |
| 4.3 | 0.44 | 100 |
| 4.4 | 0.34 | 100 |
| 4.5 | 0.24 | 100 |
| 4.6 | 0.14 | 100 |
| 4.7 | 0.98 | 100 |
| 4.8 | 0.88 | 100 |
| 4.9 | 0.78 | 100 |
| 5.0 | 0.68 | 100 |
| Note: The actual values for 'Value' are not provided in the code; they are estimated based on the provided code to calculate the value from the original data source.
</details>

(a) Popularity

![](images/cd0fa5c002f0a4cd02800f5149bac9a3185efb838a2a22669e4489e4afac619e.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(b) Locality

![](images/c3f5cb98a2040672ab45b14464591a4aee4eb2e29565322ca4e2d78bc034a390.jpg)

<details>
<summary>bubble</summary>

| X | Y | Size |
|---|---|---|
| 0.1 | 0.95 | 100 |
| 0.2 | 0.85 | 100 |
| 0.3 | 0.75 | 100 |
| 0.4 | 0.65 | 100 |
| 0.5 | 0.55 | 100 |
| 0.6 | 0.45 | 100 |
| 0.7 | 0.35 | 100 |
| 0.8 | 0.25 | 100 |
| 0.9 | 0.15 | 100 |
| 1.0 | 0.05 | 100 |
| 1.1 | 0.98 | 100 |
| 1.2 | 0.88 | 100 |
| 1.3 | 0.78 | 100 |
| 1.4 | 0.68 | 100 |
| 1.5 | 0.58 | 100 |
| 1.6 | 0.48 | 100 |
| 1.7 | 0.38 | 100 |
| 1.8 | 0.28 | 100 |
| 1.9 | 0.18 | 100 |
| 2.0 | 0.92 | 100 |
| 2.1 | 0.82 | 100 |
| 2.2 | 0.72 | 100 |
| 2.3 | 0.62 | 100 |
| 2.4 | 0.52 | 100 |
| 2.5 | 0.42 | 100 |
| 2.6 | 0.32 | 100 |
| 2.7 | 0.22 | 100 |
| 2.8 | 0.12 | 100 |
| 2.9 | 0.96 | 100 |
| 3.0 | 0.86 | 100 |
| 3.1 | 0.76 | 100 |
| 3.2 | 0.66 | 100 |
| 3.3 | 0.56 | 100 |
| 3.4 | 0.46 | 100 |
| 3.5 | 0.36 | 100 |
| 3.6 | 0.26 | 100 |
| 3.7 | 0.16 | 100 |
| 3.8 | 0.94 | 100 |
| 3.9 | 0.84 | 100 |
| 4.0 | 0.74 | 100 |
| 4.1 | 0.64 | 100 |
| 4.2 | 0.54 | 100 |
| 4.3 | 0.44 | 100 |
| 4.4 | 0.34 | 100 |
| 4.5 | 0.24 | 100 |
| 4.6 | 0.14 | 100 |
| 4.7 | 0.98 | 100 |
| 4.8 | 0.88 | 100 |
| 4.9 | 0.78 | 100 |
| 5.0 | 0.68 | 100 |
| Note: The actual values for 'X' and 'Y' are not provided in the code image, so they are estimated based on the provided code format.
</details>

(c) Density   
Figure 23: Visualization of structural shifts in node feature space for PubMed dataset.

![](images/6986a4afd85f43dc89ac7672e9cdbb16abd697ab2201024bafc4baa1ab186e17.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| 0.1 | 0.95 |
| 0.2 | 0.88 |
| 0.3 | 0.75 |
| 0.4 | 0.62 |
| 0.5 | 0.55 |
| 0.6 | 0.48 |
| 0.7 | 0.42 |
| 0.8 | 0.38 |
| 0.9 | 0.35 |
| 1.0 | 0.32 |
| 1.1 | 0.30 |
| 1.2 | 0.28 |
| 1.3 | 0.25 |
| 1.4 | 0.23 |
| 1.5 | 0.22 |
| 1.6 | 0.21 |
| 1.7 | 0.20 |
| 1.8 | 0.19 |
| 1.9 | 0.18 |
| 2.0 | 0.17 |
| 2.1 | 0.16 |
| 2.2 | 0.15 |
| 2.3 | 0.14 |
| 2.4 | 0.13 |
| 2.5 | 0.12 |
| 2.6 | 0.11 |
| 2.7 | 0.10 |
| 2.8 | 0.09 |
| 2.9 | 0.08 |
| 3.0 | 0.07 |
| 3.1 | 0.06 |
| 3.2 | 0.05 |
| 3.3 | 0.04 |
| 3.4 | 0.03 |
| 3.5 | 0.02 |
| 3.6 | 0.01 |
| 3.7 | 0.01 |
| 3.8 | 0.01 |
| 3.9 | 0.01 |
| 4.0 | 0.01 |
| 4.1 | 0.01 |
| 4.2 | 0.01 |
| 4.3 | 0.01 |
| 4.4 | 0.01 |
| 4.5 | 0.01 |
| 4.6 | 0.01 |
| 4.7 | 0.01 |
| 4.8 | 0.01 |
| 4.9 | 0.01 |
| 5.0 | 0.01 |
| 5.1 | 0.01 |
| 5.2 | 0.01 |
| 5.3 | 0.01 |
| 5.4 | 0.01 |
| 5.5 | 0.01 |
| 5.6 | 0.01 |
| 5.7 | 0.01 |
| 5.8 | 0.01 |
| 5.9 | 0.01 |
| 6.0 | 0.01 |
| 6.1 | 0.01 |
| 6.2 | 0.01 |
| 6.3 | 0.01 |
| 6.4 | 0.01 |
| 6.5 | 0.01 |
| 6.6 | 0.01 |
| 6.7 | 0.01 |
| 6.8 | 0.01 |
| 6.9 | 0.01 |
| 7.0 | 0.01 |
| 7.1 | 0.01 |
| 7.2 | 0.01 |
| 7.3 | 0.01 |
| 7.4 | 0.01 |
| 7.5 | 0.01 |
| 7.6 | 0.01 |
| 7.7 | 0.01 |
| 7.8 | 0.01 |
| 7.9 | 0.01 |
| 8.0 | 0.01 |
| 8.1 | 0.01 |
| 8.2 | 0.01 |
| 8.3 | 0.01 |
| 8.4 | 0.01 |
| 8.5 | 0.01 |
| 8.6 | 0.01 |
| 8.7 | 0.01 |
| 8.8 | 0.01 |
| 8.9 | 0.01 |
| 9.0 | 0.01 |
| 9.1 | 0.01 |
| 9.2 | 0.01 |
| 9.3 | 0.01 |
| 9.4 | 0.01 |
| 9.5 | 0.01 |
| 9.6 | 0.01 |
| 9.7 | 0.01 |
| 9.8 | 0.01 |
| 9.9 | 0.01 |
| Note: The data is randomly generated and may vary in each execution of the code or graph, as it is not explicitly provided in the image to create a table with rows and columns specified in the code format.
</details>

(a) Popularity

![](images/c1058f2108ba26fd60cb02af684d0a70fae0e1d2cdefbef1d7571609bd839c27.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| 0.1 | 0.95 | red |
| 0.2 | 0.85 | blue |
| 0.3 | 0.75 | red |
| 0.4 | 0.65 | blue |
| 0.5 | 0.55 | red |
| 0.6 | 0.45 | blue |
| 0.7 | 0.35 | red |
| 0.8 | 0.25 | blue |
| 0.9 | 0.15 | red |
| 1.0 | 0.05 | blue |
</details>

(b) Locality

![](images/15260d98ae88a3014bffb322bbb19a749aa218100e8205352220e1f4bd5f15d8.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(c) Density   
Figure 24: Visualization of structural shifts in node feature space for AmazonPhoto dataset.

![](images/db9ce652bce0a589ddaf79da68e7ce561da3cae3ddc5ae1f6c40f7cfd227ad6e.jpg)

<details>
<summary>scatter</summary>

| x | y |
| --- | --- |
| (data not extractable) | (data not extractable) |
</details>

(a) Popularity

![](images/e2c5dc870878adc3ea3995b6b568f29b4048a1806df6863f6ebdaee798d9db50.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(b) Locality

![](images/94b93a7785e5f3b327b5289ca7a9ef368180af7c214c96dc9916f8b2714e5ee2.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| (data not extractable) | (data not extractable) |
</details>

(c) Density   
Figure 25: Visualization of structural shifts in node feature space for AmazonComputer dataset.

![](images/a8cd6660243f847311a5f6becef5775fbb838d94e28469524357f6a4dc7f4d8f.jpg)

<details>
<summary>bubble</summary>

| X | Y | Size |
|---|---|---|
| 0.1 | 0.95 | 100 |
| 0.2 | 0.85 | 100 |
| 0.3 | 0.75 | 100 |
| 0.4 | 0.65 | 100 |
| 0.5 | 0.55 | 100 |
| 0.6 | 0.45 | 100 |
| 0.7 | 0.35 | 100 |
| 0.8 | 0.25 | 100 |
| 0.9 | 0.15 | 100 |
| 1.0 | 0.05 | 100 |
| 1.1 | 0.98 | 100 |
| 1.2 | 0.88 | 100 |
| 1.3 | 0.78 | 100 |
| 1.4 | 0.68 | 100 |
| 1.5 | 0.58 | 100 |
| 1.6 | 0.48 | 100 |
| 1.7 | 0.38 | 100 |
| 1.8 | 0.28 | 100 |
| 1.9 | 0.18 | 100 |
| 2.0 | 0.92 | 100 |
| 2.1 | 0.82 | 100 |
| 2.2 | 0.72 | 100 |
| 2.3 | 0.62 | 100 |
| 2.4 | 0.52 | 100 |
| 2.5 | 0.42 | 100 |
| 2.6 | 0.32 | 100 |
| 2.7 | 0.22 | 100 |
| 2.8 | 0.12 | 100 |
| 2.9 | 0.96 | 100 |
| 3.0 | 0.86 | 100 |
| 3.1 | 0.76 | 100 |
| 3.2 | 0.66 | 100 |
| 3.3 | 0.56 | 100 |
| 3.4 | 0.46 | 100 |
| 3.5 | 0.36 | 100 |
| 3.6 | 0.26 | 100 |
| 3.7 | 0.16 | 100 |
| 3.8 | 0.94 | 100 |
| 3.9 | 0.84 | 100 |
| 4.0 | 0.74 | 100 |
| 4.1 | 0.64 | 100 |
| 4.2 | 0.54 | 100 |
| 4.3 | 0.44 | 100 |
| 4.4 | 0.34 | 100 |
| 4.5 | 0.24 | 100 |
| 4.6 | 0.14 | 100 |
| 4.7 | 0.98 | 100 |
| 4.8 | 0.88 | 100 |
| 4.9 | 0.78 | 100 |
| 5.0 | 0.68 | 100 |
| Note: The actual values for 'Value' are not provided in the code; they are estimated based on the provided code to calculate the value from the original data point (e.g., 'Value' = 'Value' * 'Value'). The code does not have explicit labels or additional data series in this image.
</details>

(a) Popularity

![](images/1ed0f231d410786850b25d59d2e931ff66c64270a15570ab225a99a9dd63b75d.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(b) Locality

![](images/74d06a1ee59a1ac2819f371225fc288a56dd04bfb816b1c8ae2731230c971e05.jpg)

<details>
<summary>bubble</summary>

| X | Y | Size |
|---|---|---|
| 0.1 | 0.95 | 100 |
| 0.2 | 0.85 | 100 |
| 0.3 | 0.75 | 100 |
| 0.4 | 0.65 | 100 |
| 0.5 | 0.55 | 100 |
| 0.6 | 0.45 | 100 |
| 0.7 | 0.35 | 100 |
| 0.8 | 0.25 | 100 |
| 0.9 | 0.15 | 100 |
| 1.0 | 0.05 | 100 |
| 1.1 | 0.98 | 100 |
| 1.2 | 0.88 | 100 |
| 1.3 | 0.78 | 100 |
| 1.4 | 0.68 | 100 |
| 1.5 | 0.58 | 100 |
| 1.6 | 0.48 | 100 |
| 1.7 | 0.38 | 100 |
| 1.8 | 0.28 | 100 |
| 1.9 | 0.18 | 100 |
| 2.0 | 0.92 | 100 |
| 2.1 | 0.82 | 100 |
| 2.2 | 0.72 | 100 |
| 2.3 | 0.62 | 100 |
| 2.4 | 0.52 | 100 |
| 2.5 | 0.42 | 100 |
| 2.6 | 0.32 | 100 |
| 2.7 | 0.22 | 100 |
| 2.8 | 0.12 | 100 |
| 2.9 | 0.96 | 100 |
| 3.0 | 0.86 | 100 |
| 3.1 | 0.76 | 100 |
| 3.2 | 0.66 | 100 |
| 3.3 | 0.56 | 100 |
| 3.4 | 0.46 | 100 |
| 3.5 | 0.36 | 100 |
| 3.6 | 0.26 | 100 |
| 3.7 | 0.16 | 100 |
| 3.8 | 0.94 | 100 |
| 3.9 | 0.84 | 100 |
| 4.0 | 0.74 | 100 |
| 4.1 | 0.64 | 100 |
| 4.2 | 0.54 | 100 |
| 4.3 | 0.44 | 100 |
| 4.4 | 0.34 | 100 |
| 4.5 | 0.24 | 100 |
| 4.6 | 0.14 | 100 |
| 4.7 | 0.98 | 100 |
| 4.8 | 0.88 | 100 |
| 4.9 | 0.78 | 100 |
| 5.0 | 0.68 | 100 |
| Note: The actual values for 'Value' are not provided in the code; they are estimated based on the provided code to calculate the value from the original data source (e.g., 'Value' = 'Value' * 'Value'). The code does not have explicit labels or additional data series in this view.
</details>

(c) Density   
Figure 26: Visualization of structural shifts in node feature space for CoauthorCS dataset.

![](images/de5b35341f69d1edbfb026212aa873e42c9fdc1949573dc82ccc4dd65397ff1c.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| 0.0 | 0.0 |
| 0.1 | 0.1 |
| 0.2 | 0.2 |
| 0.3 | 0.3 |
| 0.4 | 0.4 |
| 0.5 | 0.5 |
| 0.6 | 0.6 |
| 0.7 | 0.7 |
| 0.8 | 0.8 |
| 0.9 | 0.9 |
| 1.0 | 1.0 |
</details>

(a) Popularity

![](images/1c506903ecd9a4cec4db01a16e44a6cfb4019eb578332066070d6152ea20846d.jpg)

<details>
<summary>scatter</summary>

| x | y | group |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | Red |
| (data not extractable) | (data not extractable) | Blue |
</details>

(b) Locality

![](images/fa94562287529ebaf7d75218cda3e5aba58652062b76538c0bf1cf02f1229ed2.jpg)

<details>
<summary>scatter</summary>

| x | y |
|---|---|
| 0.0 | 0.0 |
| 0.1 | 0.1 |
| 0.2 | 0.2 |
| 0.3 | 0.3 |
| 0.4 | 0.4 |
| 0.5 | 0.5 |
| 0.6 | 0.6 |
| 0.7 | 0.7 |
| 0.8 | 0.8 |
| 0.9 | 0.9 |
| 1.0 | 1.0 |
</details>

(c) Density   
Figure 27: Visualization of structural shifts in node feature space for CoauthorPhysics dataset.