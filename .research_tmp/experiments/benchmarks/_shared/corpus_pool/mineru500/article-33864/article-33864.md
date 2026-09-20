# Exploring Vacant Classes in Label-Skewed Federated Learning

Kuangpu Guo $^{1,3}$ , Yuhe Ding $^{2,3}$ , Jian Liang $^{3,4*}$ , Ran He $^{3,4}$ , Zilei Wang $^{1}$ , Tieniu Tan $^{5,4,3}$

$^{1}$ University of Science and Technology of China, $^{2}$ Anhui University,

$^{3}$ NLPR & MAIS, Institute of Automation, Chinese Academy of Sciences,

$^{4}$ University of Chinese Academy of Sciences, $^{5}$ Nanjing University

gkp@mail.ustc.edu.cn, {madao3c, liangjian92}@gmail.com, zlwang@ustc.edu.cn, {rhe, tint}@nlpr.ia.ac.cn

# Abstract

Label skews, characterized by disparities in local label distribution across clients, pose a significant challenge in federated learning. As minority classes suffer from worse accuracy due to overfitting on local imbalanced data, prior methods often incorporate class-balanced learning techniques during local training. Although these methods improve the mean accuracy across all classes, we observe that vacant classes—referring to categories absent from a client's data distribution—remain poorly recognized. Besides, there is still a gap in the accuracy of local models on minority classes compared to the global model. This paper introduces FedVLS, a novel approach to label-skewed federated learning that integrates both vacant-class distillation and logit suppression simultaneously. Specifically, vacant-class distillation leverages knowledge distillation during local training on each client to retain essential information related to vacant classes from the global model. Moreover, logit suppression directly penalizes network logits for non-label classes, effectively addressing misclassifications in minority classes that may be biased toward majority classes. Extensive experiments validate the efficacy of FedVLS, demonstrating superior performance compared to previous state-of-the-art (SOTA) methods across diverse datasets with varying degrees of label skews. Our code is available at https://github.com/krumpguo/FedVLS.

# Introduction

Federated learning has emerged as a prominent distributed learning paradigm, lauded for its capability to train a global model without direct access to raw data (Konečný et al. 2016; Li et al. 2020a; Kairouz et al. 2021). The traditional federated learning (FL) algorithm, FedAvg (McMahan et al. 2017), follows an iterative process of refining the global model by aggregating parameters from local models, which are initialized with the latest global model parameters and trained across diverse client devices (Sheller et al. 2020; Li et al. 2020b; Chai et al. 2023; Luo et al. 2022). In real-world scenarios, local client data often originate from diverse populations or organizations, displaying significant label skews that severely undermine the performance of federated learning (Hsu, Qi, and Brown 2019; Yang, Fang, and Liu 2021; Reguieg et al. 2023; Zhang et al. 2023; Ye et al. 2023).

Typically, the local data often consists of majority classes and minority classes, which refer to classes with a large amount of data a small amount of data, respectively (Zhang et al. 2022, 2023). As evidenced in prior studies (Zhang et al. 2022; Chen et al. 2022), the accuracy of minority classes notably decreases after local updates, signaling that the client model is overfitting to local imbalanced data. Consequently, this results in substantial performance degradation of the global model (Yeganeh et al. 2020; Li et al. 2020b; Liu et al. 2022). To address the issue of lower accuracy in minority classes, previous methods often incorporate class-balanced learning techniques during local training (Zhang et al. 2022; Chen et al. 2022; Wang et al. 2023b; Shen, Wang, and Lv 2023). Some works (Zhang et al. 2022; Chen et al. 2022; Shen, Wang, and Lv 2023; Wang et al. 2023b) advocate for calibrating logits according to the client data distribution to balance minority and majority classes. However, previous methods have overlooked vacant classes, which refer to classes without data but have highly versatile applications. For instance, in landmark detection (Weyand et al. 2020), most contributors possess only a subset of landmark categories from places they have lived or traveled. More importantly, these vacant classes can significantly compromise the model's performance, particularly in scenarios with highly skewed label distributions.

For example, we compare the class-wise accuracy of the initial global model and updated local models using both the classic method FedAvg (McMahan et al. 2017) and one SOTA method FedLC (Zhang et al. 2022). As depicted in Figure 1 (b) and (c), the updated local model exhibits a notable decline in accuracy for vacant classes (e.g., categories 0, 1, 2, 4, 6, and 7) compared to the class-wise accuracy of the initial global model. In extreme cases, the accuracy even decreases close to zero (e.g., category 6). We posit this severe decline is attributed to the loss of information about vacant classes in the updated local models. By the way, although FedLC (Zhang et al. 2022) partially alleviates the performance decline for minority classes, particularly in classes 3, a substantial gap remains compared to the global model. These results indicate that ignoring vacant classes can lead to a sharp decline in accuracy for those classes, and previous methods still often misclassify minority classes.

Based on these findings, we believe improving the accuracy of vacant and minority classes is crucial for addressing

![](images/e81fa7722132d4c379a46daca5ce90a1b160a14e0f14a5d32050835baeec4f65.jpg)

<details>
<summary>line</summary>

| Class | Red Line Value | Green Dashed Line Value |
|-------|----------------|-------------------------|
| 0     | 420            | 380                     |
| 1     | 450            | 400                     |
| 2     | 380            | 320                     |
| 3     | 400            | 350                     |
| 4     | 420            | 380                     |
| 5     | 400            | 350                     |
| 6     | 350            | 280                     |
| 7     | 400            | 350                     |
| 8     | 450            | 420                     |
| 9     | 480            | 480                     |
</details>

![](images/d04bc526cd723d75eeb19365718a97aa6a3160b152dec336bcdbd6b9d8dba6e8.jpg)

<details>
<summary>line</summary>

| Class | Red Line Value | Green Line Value |
|-------|----------------|------------------|
| 0     | 300            | 1100             |
| 1     | 350            | 1200             |
| 2     | 300            | 900              |
| 3     | 500            | 1000             |
| 4     | 300            | 1200             |
| 5     | 1400           | 1100             |
| 6     | 700            | 700              |
| 7     | 200            | 1200             |
| 8     | 1500           | 1500             |
| 9     | 1200           | 1400             |
</details>

![](images/328d938a499ae6f68fa5f743449ce3646caab525b0b1529f37e9a1ea66eca50d.jpg)

<details>
<summary>line</summary>

| Class | Red Line Value | Green Line Value |
|-------|----------------|------------------|
| 0     | 100            | 1100             |
| 1     | 200            | 1200             |
| 2     | 300            | 1000             |
| 3     | 600            | 1100             |
| 4     | 400            | 1200             |
| 5     | 1300           | 1100             |
| 6     | 0              | 500              |
| 7     | 300            | 800              |
| 8     | 1500           | 1400             |
| 9     | 1400           | 1400             |
</details>

![](images/19cd21a859ddbd47b4045cdb4fd5309e84010febfaf94f4386750fd8b3b01d48.jpg)

<details>
<summary>line</summary>

| Class | Accuracy |
|-------|----------|
| 0     | 0.9      |
| 1     | 1.2      |
| 2     | 0.8      |
| 3     | 0.9      |
| 4     | 1.3      |
| 5     | 1.1      |
| 6     | 0.5      |
| 7     | 1.5      |
| 8     | 1.4      |
| 9     | 1.3      |
</details>

Sample Num --- Initial Global Acc — Updated Local Acc

Figure 1: Class-wise accuracy of the initial global model and updated local modelS on IID and label-skewed CIFAR10 data distributions. (a) represents the result updating on IID local data with FedAvg (McMahan et al. 2017). (b-d) showcase the results updating on skewed data distribution with FedAvg (McMahan et al. 2017), FedLC (Zhang et al. 2022), and our FedVLS, respectively. The value $(\%)$ in each caption corresponds to the accuracy of the global model aggregated from local models.

the challenges posed by label skews. Therefore, we present FedVLS, a novel approach comprising two pivotal components: vacant-class distillation and logit suppression. The vacant-class distillation aims to address the performance decline related to vacant classes by distilling vital information from the global model for each client during local training. Additionally, FedVLS incorporates logit suppression, which regulates the output logit for non-label classes. This process emphasizes minimizing the predicted logit values linked to the majority class when handling minority samples, amplifying the penalty for the misclassification of minority classes. As shown in Figure 1 (d), FedVLS significantly mitigates the decline in accuracy in both vacant and minority classes of the updated local model. Consequently, FedVLS effectively reduces overfitting in client models, leading to a significant improvement in the global model's performance. Our experiments demonstrate that FedVLS consistently outperforms current SOTA federated learning methods across various settings. Our contributions are summarized as follows:

- We find that prior federated learning methods suffer from vacant classes and propose FedVLS to distill vacant-class-aware knowledge from the global model.   
- FedVLS further presents a logit suppression strategy to address the misclassification of the minority classes, thereby enhancing the generalization of local models.   
- Extensive results validate the effectiveness of both components in FedVLS, outperforming previous state-of-the-art methods across diverse datasets and different degrees of label skews.

# Related Work

# Heterogeneous Federated Learning

Federated learning faces a significant challenge known as data heterogeneity, also referred to as non-identical and independently distributed (Non-IID) data (Kairouz et al. 2021; Luo et al. 2021; Shi et al. 2023b; Guo, Wang, and Geng 2024; Guo et al. 2024). This challenge encompasses issues such as label skews and domain shifts. In this paper, we primarily focus on addressing label skews. The classic federated learning algorithm, FedAvg (McMahan et al. 2017), experiences a significant decline in performance when dealing with label skews (Li et al. 2019; Acar et al. 2021; Luo, Wang, and Wang 2024). Numerous studies have aimed to mitigate the adverse impacts of label skews. For instance, FedProx (Li et al. 2020b) employs a proximal term and SCAFFOLD (Karimireddy et al. 2020) uses a variance reduction approach to constrain the update direction of local models. Additionally, MOON (Li, He, and Song 2021) and FedProc (Mu et al. 2023) utilize contrastive loss to enhance the agreement between local models and the global model. Furthermore, FedConcat (Diao, Li, and He 2024) propose model concatenation, FedMR (Fan et al. 2023) proposes a manifold reshaping approach, FedGELA (Fan et al. 2024) uses simplex Equiangular Tight to initialize the local classifier and FedGF (Lee and Yoon 2024) refine the flat minima searching to alleviate the label skews. However, these methods often fail to address the issue of vacant classes in highly skewed scenarios. We propose FedVLS to effectively mitigate the decline in class-wise accuracy of vacant classes.

# Learning from Imbalanced Data

Imbalanced data distribution is pervasive in real-world scenarios, and numerous methods have been proposed to address its impact on model performance (Cui et al. 2019; Menon et al. 2021; Tan et al. 2020; Li et al. 2022; Ma et al. 2023). Existing approaches generally fall into two categories: re-weighting (Cui et al. 2019) and logit-adjustment (Menon et al. 2021; Tan et al. 2020). However, previous works primarily discuss scenarios with long-tailed distributions (Zeng et al. 2023; Xiao et al. 2023). These methods may not be directly applicable in federated learning due to the diversity of client data distributions. In federated learning, FedLC (Zhang et al. 2022) and Calfat (Chen et al. 2022) introduce logit calibration based on the local data distribution to balance the majority and minority classes. FedLMD (Lu et al. 2023) proposes distillation masks to preserve the information of minority class. However, they often neglect vacant classes and cannot effectively handle the accuracy decrease of the minority class. Our method, on the other hand, addresses both existence of vacant classes and the class imbalance between majority and minority classes,

making it more practical for real-world scenarios.

# Knowledge Distillation in Federated Learning

Knowledge Distillation (KD) has been introduced to federated learning to address issues arising from variations in data distributions and model constructions across clients (Jeong et al. 2018; Itahara et al. 2021; Wu et al. 2023). FedDF (Lin et al. 2020) and FedMD (Li and Wang 2019) leverage KD to transfer the knowledge from multiple local models to the global model. However, these KD methods typically require a public dataset available to all clients on the server, which presents potential practical challenges. Recent methods, such as FEDGEN (Zhu, Hong, and Zhou 2021), DaFKD (Wang et al. 2023a), and DFRD (Luo et al. 2023), propose training a generator on the server or client to enable data-free federated knowledge distillation. However, training the generator adds computational complexity and can often be unstable in cases of extreme label skews (Wu et al. 2023). Additionally, FedNTD (Lee et al. 2022) conducts local-side distillation only for not-true labels to prevent overfitting, while FedHKD (Chen, Vikalo et al. 2023) performs local-side distillation on both logits and class prototypes to align the global and local optimization directions. However, these methods perform knowledge distillation across all classes, which may limit the retention of local models' information about vacant classes. Our method, in contrast, applies knowledge distillation exclusively to vacant classes, preserving vital information about these categories without impacting the learning of other categories or introducing significant computational overhead.

# Method

# Preliminaries

In federated learning, we consider a scenario with N clients, where $D_{i}$ represents the local training data of client i. The combined data $D = \bigcup_{i=1}^{N} D_{i}$ comprises the local data from all clients. These data distributions might differ across clients, encompassing situations where the local training data of some clients only contain samples from a subset of all classes. The overarching goal is to address the optimization problem as follows (McMahan et al. 2017):

$$
\min _ {\boldsymbol {\omega}} \left[ \mathcal {L} (\boldsymbol {\omega}) \stackrel {d e f} {=} \sum_ {i = 1} ^ {N} \frac {| \mathcal {D} _ {i} |}{| \mathcal {D} |} \mathcal {L} _ {i} (\boldsymbol {\omega}) \right], \tag {1}
$$

where $\mathcal{L}_{i}(\boldsymbol{\omega}) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{i}}[\ell_{i}(f(x;\boldsymbol{\omega}), y)]$ is the empirical loss of the i-th client. $f(x; \boldsymbol{\omega})$ is the output of the model when the input x and model parameter $\omega$ are given, and $\ell_{i}$ is the loss function of the i-th client. $|D_{i}|$ is the number of samples on $D_{i}$ , $|D|$ is the number of samples on D. Here, FL expects to learn a global model that can perform well on the entire data D.

# Motivation

When the local training data $\{D_{i}\}_{i=1}^{N}$ exhibit label skews, as illustrated in Figure 3 of the technical appendix, there are variations in the quantity of data for the same category across different clients, leading the client models to excessively fit their respective local data distributions. This overfitting phenomenon causes divergence during model aggregation, subsequently resulting in inferior global performance (Yeganeh et al. 2020; Li et al. 2020b; Liu et al. 2022). To address the imbalance between minority and majority classes, previous methodologies (Zhang et al. 2022; Shen, Wang, and Lv 2023) suggest calibrating logits based on the local data distribution, outlined as follows:

$$
\mathcal {L} _ {\mathrm{cal}} = - \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {i}} \log \left(\frac {p (y) \cdot e ^ {f (x ; \boldsymbol {\omega}) [ y ]}}{\sum_ {c} p (c) \cdot e ^ {f (x ; \boldsymbol {\omega}) [ c ]}}\right), \tag {2}
$$

where $p(y)$ signifies the probability of class $y$ occurring within the client's data distribution, while $f(x; \omega)[c]$ denotes the logit output for the c-th category. The calibration technique weights the outputs for all classes in the denominator by $p(c)$ . However, the probability $p(m)$ for the vacant class in the client data distribution equates to zero, leading to the weighting term for the vacant category, $p(m) \cdot e^{f(x; \omega)[m]}$ , also becoming zero. Consequently, local models prioritize learning the majority and minority classes, gradually disregarding information associated with the vacant categories during local training. This gradual shift causes the updated direction of local models to deviate from that of the global model over time. It's crucial to acknowledge that treating the vacant class merely as a unique minority class is insufficient, an oversight prevalent in prior methodologies. We assert this drawback significantly contributes to severe instances of local overfitting.

Our empirical observations reveal a substantial decrease in class-wise accuracy for vacant classes (such as categories 0, 1, 2, 4, 6, and 7) after the local update, as illustrated in Figure 1 (b) and (c). Notably, in specific cases (such as category 6), this accuracy even drops close to zero. The specific experimental setup and other analyses can be found in the technical appendix. By the way, we find the updated class-wise accuracy for the minority classes (e.g., category 3) continues to display a notable decline, maintaining a significant gap compared to the IID scenario. Through the analysis of the confusion matrix in Figure 2, we find that vacant and minority classes are still frequently misclassified as majority classes. Thus, we aim to develop different objectives to alleviate these two issues, respectively.

# Vacant-class Distillation

Motivated by the above observations and analyses of vacant classes, we propose to prevent the disappearance of information related to vacant classes during local training. The global model harbors valuable insights, particularly regarding the prediction of vacant classes, making it an exceptional teacher for each client. Hence, we introduce vacant-class distillation, aimed at preserving the global perspective of vacant classes for clients through knowledge distillation. To achieve this, we utilize the Kullback-Leibler Divergence loss function, as outlined below:

$$
\mathcal {L} _ {\mathbf {d i s}} = \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {i}} \sum_ {o \in \mathbb {O}} q ^ {g} (o; x) \log \left[ \frac {q (o ; x)}{q ^ {g} (o ; x)} \right], \tag {3}
$$

$$
\text { where } q (o; x) = \frac {\exp (f (x ; \omega) [ o ])}{\sum_ {c \in \mathbb {O}} \exp (f (x ; \omega) [ c ])}
$$

![](images/53246e91b6522ae22a5d22527fa2b73e50645afe5104c2d9c24a2976c9247356.jpg)

<details>
<summary>heatmap</summary>

Confusion Matrix of Client 3
| Truth label | Predict label 0 | Predict label 1 | Predict label 2 | Predict label 3 | Predict label 4 | Predict label 5 | Predict label 6 | Predict label 7 | Predict label 8 | Predict label 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.92 | 0.0 | 0.0 | 0.01 | 0.04 | 0.0 | 0.01 | 0.0 | 0.03 | 0.0 |
| 1 | 0.05 | 0.78 | 0.0 | 0.03 | 0.06 | 0.0 | 0.0 | 0.03 | 0.05 | 0.0 |
| 2 | 0.07 | 0.02 | 0.65 | 0.04 | 0.06 | 0.01 | 0.01 | 0.07 | 0.06 | 0.01 |
| 3 | 0.04 | 0.0 | 0.0 | 0.86 | 0.05 | 0.0 | 0.0 | 0.02 | 0.03 | 0.0 |
| 4 | 0.02 | 0.0 | 0.0 | 0.0 | 0.95 | 0.0 | 0.0 | 0.01 | 0.02 | 0.0 |
| 5 | 0.08 | 0.02 | 0.01 | 0.05 | 0.12 | 0.61 | 0.02 | 0.02 | 0.07 | 0.0 |
| 6 | 0.07 | 0.01 | 0.01 | 0.07 | 0.11 | 0.03 | 0.56 | 0.05 | 0.09 | 0.0 |
| 7 | 0.05 | 0.02 | 0.0 | 0.03 | 0.06 | 0.0 | 0.01 | 0.8 | 0.03 | 0.0 |
| 8 | 0.03 | 0.01 | 0.0 | 0.02 | 0.04 | 0.0 | 0.0 | 0.02 | 0.88 | 0.0 |
| 9 | 0.17 | 0.03 | 0.02 | 0.1 | 0.18 | 0.01 | 0.01 | 0.08 | 0.14 | 0.26 |
</details>

Figure 2: Confusion matrix of client 3 on CIFAR10 dataset with Dirichlet-based label skews ( $\beta = 0.5$ ) using FedLC (Zhang et al. 2022).

denotes the output for the $o$ -th class of the local model using softmax within the vacant classes, and $q^{g}(o;x) = \frac{\exp(f(x;\omega^{\mathbf{g}})[o])}{\sum_{c\in\mathbb{O}}\exp(f(x;\omega^{\mathbf{g}})[c]}$ denotes the same for the global model. $\mathbb{O}$ represents the set that contains all vacant classes within the local client data and $\omega^{g}$ denotes the parameters of the global model.

Unlike FedNTD (Lee et al. 2022), which encourages the client model to closely match the global model's output for not-true labels, thereby limiting the knowledge protection for vacant class, our loss function ensures that the local model replicates the global model's outputs only for vacant-class labels. This approach preserves the predictive capability for vacant categories significantly. Moreover, the computational overhead introduced by this loss function is minimal, enhancing its practical implementation. Additional comparisons with other distillation-based methods and further analyses are provided in the technical appendix.

# Logit Suppression

Previous methods still often suffer from low accuracy in the minority classes of local models, a factor that requires mitigation to enhance the generalization capabilities of these models. To identify the root cause of this issue, we analyzed the confusion matrix of the local model in client 3 on the entire test dataset using FedLC (Zhang et al. 2022), where the training data distribution is shown in the fourth column of Figure 3 (a) in the technical appendix. As shown in Figure 2, minority classes (e.g., categories 2, 5, and 6) are frequently misclassified as majority classes (e.g., categories 0, 4, and 8). It is evident that, in the model's output for minority samples, the majority class tends to have a higher logit value, leading to the misclassification of minority classes. Therefore, we implement regularization on non-label class logits to penalize the majority class output for minority samples. To avoid non-trivial optimization over direct logits, we aim to minimize the following objective for each class:

$$
\mathcal {L} _ {\text { logit }} ^ {c} = \log \left(\mathbb {E} _ {(x, y) \sim \mathcal {D} _ {i}} \mathbb {I} (y \neq c) \cdot e ^ {f (x; \boldsymbol {\omega}) [ c ]}\right), \tag {4}
$$

where $\mathbb{I}$ is an indicator function with value 1 when $y \neq c$ . We use the log function to increase the proportion of loss values for minority categories. Since minority samples are prone to be more frequently misclassified into majority classes, the higher weight should be assigned to the logits of majority categories in non-labeled outputs. Therefore, we weight the loss function $\mathcal{L}_{\mathrm{logit}}^{c}$ using the probability of occurrence $p(c)$ for each class as follows:

$$
\mathcal {L} _ {\text { logit }} = \sum_ {c} p (c) \cdot \mathcal {L} _ {\text { logit }} ^ {c}. \tag {5}
$$

This adaptation prompts the learning process to pay more attention to the penalty for incorrectly classifying minority class samples as majority classes. As a result, the model is encouraged to refine its prediction across diverse classes, thereby improving its overall generalization capability.

# Overall Objective

As of now, we have elaborated extensively on our strategy to tackle the problem of loss of information about vacant classes in previous methods through knowledge distillation. Moreover, we mitigate the decrease in minority classes of the updated local model by regulating non-label logits directly, to further alleviate local overfitting issues. In summary, we propose the comprehensive method named Fed-VLS, whose objective is as follows:

$$
\mathcal {L} (\omega) = \mathcal {L} _ {\mathrm{cal}} + \lambda \cdot \mathcal {L} _ {\mathrm{dis}} + \mathcal {L} _ {\mathrm{logit}}, \tag {6}
$$

where $\lambda$ is a non-negative hyperparameter to control the contribution of vacant-class distillation. In the loss function of our FedVLS, we attain new knowledge from the observed class in local data distribution using the $\mathcal{L}_{\mathrm{cal}}$ and $\mathcal{L}_{\mathrm{logit}}$ . In the meanwhile, we preserve the previous knowledge on the vacant classes by following the global model's perspective using the $\mathcal{L}_{\mathrm{dis}}$ . By combining vacant-class distillation and logit suppression, FedVLS can effectively manipulate various levels of label skews. Algorithm 1 in the technical appendix shows the overflow of our method.

# Experiments

# Setups

Datasets We evaluate the effectiveness of our approach across various image classification datasets, including MNIST (Deng 2012), CIFAR10 (Krizhevsky 2009), CIFAR100 (Krizhevsky 2009), and TinyImageNet (Le and Yang 2015). We partitioned each dataset into distinct training and test sets. Subsequently, the training set undergoes further division into non-overlapping subsets, distributed among different clients. The global model's performance is then assessed on the test set. We follow the settings outlined in (Li et al. 2022) and introduce two prevalent forms of label skews: Dirichlet-based and quantity-based. In the

Table 1: Performance overview for different degrees of Dirichlet-based label skews. All results are (re)produced by us and are averaged over 3 runs (mean ± std). Bold is the best result, underline is the second-best. 

<table><tr><td rowspan="2">Method(venue)</td><td colspan="3">MNIST</td><td colspan="3">CIFAR10</td><td colspan="3">CIFAR100</td><td colspan="3">TinyImageNet</td></tr><tr><td> $\beta = 0.5$ </td><td> $\beta = 0.1$ </td><td> $\beta = 0.05$ </td><td> $\beta = 0.5$ </td><td> $\beta = 0.1$ </td><td> $\beta = 0.05$ </td><td> $\beta = 0.5$ </td><td> $\beta = 0.1$ </td><td> $\beta = 0.05$ </td><td> $\beta = 0.5$ </td><td> $\beta = 0.1$ </td><td> $\beta = 0.06$ </td></tr><tr><td>FedAvg (AISTATS 2017)</td><td>98.96±0.00</td><td>96.69±0.00</td><td>94.77±0.44</td><td>91.46±0.55</td><td>82.00±0.75</td><td>62.90±0.95</td><td>72.22±0.34</td><td>66.18±0.35</td><td>62.13±0.09</td><td>47.02±0.40</td><td>39.90±0.27</td><td>35.21±0.47</td></tr><tr><td>FedProx (MLSys 2020)</td><td>98.93±0.00</td><td>96.42±0.00</td><td>94.95±0.24</td><td>92.24±0.78</td><td>82.65±1.33</td><td>63.14±0.41</td><td>72.65±0.60</td><td>66.61±0.22</td><td>62.23±0.20</td><td>45.76±0.50</td><td>40.26±0.51</td><td>35.22±0.17</td></tr><tr><td>MOON (CVPR 2021)</td><td>99.18±0.01</td><td>96.94±0.12</td><td>93.39±0.21</td><td>92.13±0.35</td><td>83.38±0.43</td><td>61.34±0.77</td><td>72.87±0.11</td><td>66.12±0.32</td><td>60.45±0.41</td><td>42.26±0.36</td><td>36.88±0.53</td><td>33.61±0.35</td></tr><tr><td>FedEXP (ICLR 2023)</td><td>97.57±0.49</td><td>91.59±0.48</td><td>92.54±1.08</td><td>92.31±0.52</td><td>83.48±1.15</td><td>63.22±0.51</td><td>72.41±0.39</td><td>66.74±0.19</td><td>62.24±0.18</td><td>47.00±0.23</td><td>40.58±0.15</td><td>34.95±0.18</td></tr><tr><td>FedLC (ICML 2022)</td><td>98.97±0.01</td><td>95.59±0.05</td><td>85.56±0.18</td><td>91.98±0.63</td><td>82.24±0.53</td><td>57.31±0.97</td><td>72.69±0.30</td><td>66.20±0.20</td><td>59.18±0.11</td><td>48.01±0.21</td><td>41.46±0.37</td><td>35.56±0.58</td></tr><tr><td>FedRS (KDD 2021)</td><td>99.03±0.00</td><td>96.67±0.01</td><td>94.60±0.40</td><td>92.55±0.68</td><td>83.95±0.35</td><td>63.17±0.57</td><td>72.99±0.20</td><td>66.84±0.25</td><td>62.19±0.06</td><td>47.95±0.43</td><td>41.77±0.25</td><td>35.82±0.20</td></tr><tr><td>FedSAM (ICML2022)</td><td>99.21±0.00</td><td>97.24±0.00</td><td>95.17±0.42</td><td>92.37±1.33</td><td>81.19±0.32</td><td>63.11±1.05</td><td>72.96±0.25</td><td>67.50±0.19</td><td>61.32±0.14</td><td>48.43±1.42</td><td>43.96±1.02</td><td>41.14±0.23</td></tr><tr><td>FedNTD (NeurIPS 2022)</td><td>99.15±0.04</td><td>96.67±0.17</td><td>94.30±0.71</td><td>92.46±0.19</td><td>83.23±0.22</td><td>68.71±0.27</td><td>73.43±0.15</td><td>68.00±0.50</td><td>63.71±0.19</td><td>48.02±1.05</td><td>45.11±0.21</td><td>40.65±0.26</td></tr><tr><td>FedMR (TMLR 2023)</td><td>98.95±0.02</td><td>96.73±0.08</td><td>95.34±0.50</td><td>91.98±0.55</td><td>82.09±0.42</td><td>63.54±0.69</td><td>71.94±0.36</td><td>67.57±0.37</td><td>63.75±0.24</td><td>47.21±0.53</td><td>40.35±0.26</td><td>35.94±0.46</td></tr><tr><td>FedLMD (MM 2023)</td><td>99.17±0.03</td><td>97.18±0.12</td><td>95.33±0.53</td><td>92.50±0.34</td><td>83.14±0.19</td><td>70.50±0.29</td><td>73.30±0.30</td><td>68.83±0.35</td><td>64.10±0.19</td><td>48.43±0.48</td><td>44.03±0.25</td><td>41.18±0.27</td></tr><tr><td>FedConcat (AAAI 2024)</td><td>99.04±0.01</td><td>96.99±0.11</td><td>95.02±0.47</td><td>92.45±0.29</td><td>82.83±0.21</td><td>64.30±0.28</td><td>73.27±0.28</td><td>68.57±0.34</td><td>63.74±0.13</td><td>48.45±0.44</td><td>47.32±0.21</td><td>43.44±0.21</td></tr><tr><td>FedGF (ICML 2024)</td><td>99.22±0.00</td><td>97.35±0.00</td><td>95.36±0.28</td><td>92.52±0.22</td><td>82.91±0.16</td><td>69.61±0.47</td><td>73.30±0.25</td><td>68.70±0.20</td><td>64.48±0.08</td><td>48.52±0.23</td><td>47.64±0.16</td><td>44.71±0.20</td></tr><tr><td>FedVLS (Ours)</td><td>99.23±0.00</td><td>97.24±0.00</td><td>95.56±0.12</td><td>92.66±0.14</td><td>84.35±0.04</td><td>75.71±0.28</td><td>73.49±0.80</td><td>69.02±0.18</td><td>65.71±0.01</td><td>48.54±0.12</td><td>47.73±0.13</td><td>45.23±0.15</td></tr></table>

Table 2: Performance overview for quantity-based label skews. s presents the number of shards per client. 

<table><tr><td rowspan="2">Method(venue)</td><td>CIFAR10</td><td>CIFAR100</td><td>TinyImageNet</td></tr><tr><td>s = 2</td><td>s = 20</td><td>s = 40</td></tr><tr><td>FedAvg (AISTATS 2017)</td><td> $44.63 \pm 0.77$ </td><td> $63.14 \pm 0.03$ </td><td> $30.28 \pm 0.12$ </td></tr><tr><td>FedProx (MLSys 2020)</td><td> $48.65 \pm 0.59$ </td><td> $62.10 \pm 0.10$ </td><td> $28.14 \pm 0.93$ </td></tr><tr><td>MOON (CVPR 2021)</td><td> $38.24 \pm 1.00$ </td><td> $57.33 \pm 0.06$ </td><td> $26.25 \pm 0.73$ </td></tr><tr><td>FedEXP (ICLR 2023)</td><td> $41.11 \pm 0.26$ </td><td> $62.61 \pm 0.06$ </td><td> $29.38 \pm 0.19$ </td></tr><tr><td>FedLC (ICML 2022)</td><td> $55.14 \pm 0.26$ </td><td> $61.56 \pm 0.03$ </td><td> $26.29 \pm 1.00$ </td></tr><tr><td>FedRS (KDD 2021)</td><td> $42.20 \pm 1.49$ </td><td> $61.53 \pm 0.03$ </td><td> $28.31 \pm 0.06$ </td></tr><tr><td>FedSAM (ICML2022)</td><td> $36.97 \pm 1.18$ </td><td> $63.50 \pm 0.01$ </td><td> $37.55 \pm 0.10$ </td></tr><tr><td>FedNTD (NeurIPS 2022)</td><td> $67.35 \pm 0.19$ </td><td> $63.74 \pm 0.01$ </td><td> $37.19 \pm 0.07$ </td></tr><tr><td>FedMR (TMLR 2023)</td><td> $46.55 \pm 0.64$ </td><td> $63.55 \pm 0.03$ </td><td> $28.45 \pm 0.10$ </td></tr><tr><td>FedLMD (MM 2023)</td><td> $\underline{68.52} \pm 0.34$ </td><td> $63.51 \pm 0.02$ </td><td> $32.29 \pm 0.08$ </td></tr><tr><td>FedConcat (AAAI 2024)</td><td> $62.00 \pm 0.28$ </td><td> $63.87 \pm 0.01$ </td><td> $42.95 \pm 0.06$ </td></tr><tr><td>FedGF (ICML 2024)</td><td> $66.97 \pm 0.45$ </td><td> $\underline{63.90} \pm 0.02$ </td><td> $\underline{43.55} \pm 0.06$ </td></tr><tr><td>FedVLS (Ours)</td><td> $\underline{68.03} \pm 0.18$ </td><td> $\underline{64.95} \pm 0.01$ </td><td> $\underline{43.97} \pm 0.04$ </td></tr></table>

quantity-based label skews, all training data is grouped by label and allocated into shards with imbalanced quantities. The parameter s signifies the number of shards per client, regulating the level of label skews (Lee et al. 2022). In the Dirichlet-based label skews, clients receive samples for each class based on the Dirichlet distribution (Zhu et al. 2021), denoted as $D(\beta)$ . Here, the parameter $\beta$ controls the degree of label skews, with lower values indicating higher label skews. Notably, each client's training data may encompass majority classes, minority classes, and even vacant classes, which is more practical.

Models and baselines Following a prior study (Shi et al. 2023a), our primary network architecture for all experiments, except MNIST, predominantly relies on MobileNetV2 (Sandler et al. 2018). For the MNIST, we adopt a deep neural network (DNN) containing three fully connected layers as the backbone. Our baseline models encompass conventional approaches to tackle data heterogeneity issues, including FedProx (Li et al. 2020b), MOON (Li, He, and Song 2021), FedSAM (Qu et al. 2022), FedEXP (Divyansh Jhunjhunwala 2023), FedConcat (Diao, Li, and He 2024), and FedGF (Lee and Yoon 2024). To ensure a fair comparison, we also assess our method against FedRS (Li and Zhan 2021), FedLC (Zhang et al. 2022), FedNTD (Lee et al. 2022) and FedLMD (Lu et al. 2023), which also focus on addressing label skews in federated learning.

Implementation details We set the number of clients N to 10 and implement full client participation. We run 100 communication rounds for all experiments on the CIFAR10/100 datasets and 50 communication rounds on the MNIST and TinyImageNet datasets. Within each communication round, local training spans 5 epochs for MNIST and 10 epochs for the other datasets. For FedConcat (Diao, Li, and He 2024) and FedGF (Lee and Yoon 2024), we followed the original paper's settings for communication rounds and local epochs. We employ stochastic gradient descent (SGD) optimization with a learning rate of 0.01, a momentum of 0.9, and a batch size of 64. Weight decay is set to $10^{-5}$ for MNIST and CIFAR10 and $10^{-4}$ for CIFAR100 and TinyImageNet. The hyperparameter $\lambda$ of FedVLS in Equation 6 is set to 0.1 for MNIST and CIFAR10, while it is set to 0.5 for CIFAR100 and TinyImageNet. Following pFedMe (T Dinh, Tran, and Nguyen 2020), we conduct three trials for each experimental setting and report the mean accuracy and standard deviation of the maximum accuracy achieved by the global model during the training process. More implementation details and experimental results can be found in the technical appendix at https://github.com/krumpguo/FedVLS.

# Results

Results under various levels of label skews and datasets Table 1 presents the performance results of various methods with different levels of Dirichlet-based label skews ( $\beta \in \{0.5, 0.1, 0.05\}$ ). Our method consistently achieves notably higher accuracy compared to other SOTA methods. As the degree of label skews increases, competing methods struggle to maintain their performance levels. For instance, FedLC (Zhang et al. 2022) experiences a substantial decline, dropping even below the performance of the classic method FedAvg when $\beta = 0.05$ . This decline stems from each client having numerous vacant classes in extreme cases, a factor overlooked by FedLC (Zhang et al. 2022). Conversely, our method consistently upholds excellent performance, especially in highly skewed label distribution scenarios. For instance, in the case of the CIFAR10 dataset with $\beta = 0.05$ ,

![](images/fac699caeaec83b670268029635e7307801ccd2487f3eef0400e68d55ca0f539.jpg)

<details>
<summary>line</summary>

| Round | FedAvg | FedAvg (Red) | FedAvg (Orange) | FedAvg (Green) |
|-------|--------|--------------|-----------------|----------------|
| 0     | 0.5    | 0.5          | 0.5             | 0.5            |
| 20    | 0.7    | 0.75         | 0.72            | 0.73           |
| 40    | 0.75   | 0.8          | 0.78            | 0.79           |
| 60    | 0.78   | 0.82         | 0.81            | 0.83           |
| 80    | 0.8    | 0.84         | 0.83            | 0.85           |
| 100   | 0.82   | 0.85         | 0.84            | 0.86           |
</details>

![](images/64691ae26e85ff2d83166dc68bcef5b4ed11374db3eed039e458151b0d38bb90.jpg)

<details>
<summary>line</summary>

| Round | iProx | FedLC | FedRS |
|-------|-------|-------|-------|
| 0     | 0.2   | 0.2   | 0.2   |
| 20    | 0.5   | 0.45  | 0.55  |
| 40    | 0.6   | 0.55  | 0.65  |
| 60    | 0.65  | 0.6   | 0.7   |
| 80    | 0.7   | 0.65  | 0.75  |
| 100   | 0.75  | 0.7   | 0.8   |
</details>

![](images/1a8b2ecce4182b6051c7490c09eb8998faaea9e76a398ab58e6d8f211785fc50.jpg)

<details>
<summary>line</summary>

| Round | FedNTD | FedLMD | Series 1 |
|-------|--------|--------|----------|
| 0     | 0.5    | 0.5    | 0.5      |
| 20    | 0.6    | 0.6    | 0.6      |
| 40    | 0.65   | 0.65   | 0.65     |
| 60    | 0.68   | 0.68   | 0.68     |
| 80    | 0.69   | 0.69   | 0.69     |
| 100   | 0.7    | 0.7    | 0.7      |
</details>

![](images/f40df772fb73dadda582a12e02737b2750dd65890c3d34982221758ad0e12e7c.jpg)

<details>
<summary>line</summary>

| Round | FedVLS(Ours) |
|-------|--------------|
| 0     | 0.4          |
| 20    | 0.55         |
| 40    | 0.6          |
| 60    | 0.62         |
| 80    | 0.64         |
| 100   | 0.65         |
</details>

Figure 3: The test accuracy over each communication round during training for different levels of Dirichlet-based label skews ( $\beta \in \{0.1, 0.05\}$ ) on CIFAR10 and CIFAR100 datasets.

![](images/add4aaebf477fe6b552e2a5c994e410ce6a3233d0f0028df14d00c21ba48cb8e.jpg)

<details>
<summary>line</summary>

| Round | FedAvg | FedLC | FedNTD | FedLMD | FedProx | FedRS | FedVLS(Ours) |
|-------|--------|-------|--------|--------|---------|-------|--------------|
| 0     | 0.1    | 0.1   | 0.1    | 0.1    | 0.1     | 0.1   | 0.1          |
| 20    | 0.4    | 0.5   | 0.6    | 0.55   | 0.5     | 0.55  | 0.65         |
| 40    | 0.5    | 0.55  | 0.65   | 0.6    | 0.55    | 0.6   | 0.7          |
| 60    | 0.55   | 0.6   | 0.7    | 0.65   | 0.6     | 0.65  | 0.75         |
| 80    | 0.6    | 0.65  | 0.75   | 0.7    | 0.65    | 0.7   | 0.8          |
| 100   | 0.65   | 0.7   | 0.8    | 0.75   | 0.7     | 0.75  | 0.85         |
</details>

Figure 4: Sensitivity analysis on the client participating rates R, local epochs E, and client numbers N.

our method achieves an impressive test accuracy of 75.71%, surpassing FedAvg by 12.81%. This outcome highlights the efficacy of our approach in addressing the accuracy decline observed in both vacant and minority classes, effectively mitigating instances of overfitting in local data distributions. Additionally, we present the performance of these methods for quantity-based label distribution skews in Table 2, further emphasizing the superiority of our method.

Communication efficiency Figure 3 illustrates the accuracy over each communication round throughout the training process. Our method showcases quicker convergence

Table 3: Results of different methods under various backbones with Dirichlet-based label skews on CIFAR10 dataset. 

<table><tr><td rowspan="2">Method(venue)</td><td colspan="2">ResNet18</td><td colspan="2">ResNet32</td><td colspan="2">MobileNetV2</td></tr><tr><td> $\beta=0.1$ </td><td> $\beta=0.05$ </td><td> $\beta=0.1$ </td><td> $\beta=0.05$ </td><td> $\beta=0.1$ </td><td> $\beta=0.05$ </td></tr><tr><td>FedAvg (AISTATS 2017)</td><td>73.84</td><td>58.54</td><td>79.38</td><td>55.41</td><td>82.00</td><td>62.90</td></tr><tr><td>FedProx (MLSys 2020)</td><td>74.68</td><td>58.14</td><td>80.60</td><td>62.51</td><td>82.65</td><td>63.14</td></tr><tr><td>MOON (CVPR 2021)</td><td>74.04</td><td>55.41</td><td>76.91</td><td>51.85</td><td>83.38</td><td>61.34</td></tr><tr><td>FedEXP (ICLR 2023)</td><td>72.80</td><td>58.04</td><td>78.36</td><td>53.35</td><td>83.48</td><td>63.22</td></tr><tr><td>FedLC (ICML 2022)</td><td>73.15</td><td>48.94</td><td>77.71</td><td>55.41</td><td>82.24</td><td>57.31</td></tr><tr><td>FedRS (KDD 2021)</td><td>76.38</td><td>57.47</td><td>82.03</td><td>66.87</td><td>83.95</td><td>63.17</td></tr><tr><td>FedSAM (ICML2022)</td><td>68.42</td><td>55.42</td><td>75.66</td><td>58.88</td><td>81.19</td><td>63.11</td></tr><tr><td>FedNTD (NeurIPS 2022)</td><td>76.76</td><td>60.01</td><td>79.75</td><td>65.96</td><td>83.23</td><td>68.71</td></tr><tr><td>FedLMD (MM 2023)</td><td>77.02</td><td>65.80</td><td>81.76</td><td>68.04</td><td>83.14</td><td>70.50</td></tr><tr><td>FedConcat (AAAI 2024)</td><td>76.33</td><td>59.83</td><td>70.32</td><td>61.86</td><td>82.83</td><td>64.30</td></tr><tr><td>FedGF (ICML 2024)</td><td>76.74</td><td>64.44</td><td>81.44</td><td>67.83</td><td>82.91</td><td>69.61</td></tr><tr><td>FedVLS (Ours)</td><td>78.00</td><td>68.33</td><td>82.44</td><td>68.84</td><td>84.35</td><td>75.71</td></tr></table>

and higher accuracy when compared to the other six methods. Due to the differences in communication rounds among FedConcat (Diao, Li, and He 2024), FedGF (Lee and Yoon 2024) and our approach, we have not included the convergence curves for these two methods. Unlike its counterparts, our approach displays a more consistent upward trend. Moreover, our method exhibits a significant improvement as the skews in the data distribution increase. These outcomes underscore the substantial communication efficiency of our method compared with other approaches.

# Analysis

Impact of participating rates To begin with, we analyze our model's performance against SOTA methods across varying client participation rates. Unless specified otherwise, our experiments focus on the CIFAR10 dataset with a Dirichlet-based skew parameter of $\beta = 0.05$ . Initially, we set the client participation rate $R$ within the range $\{0.5, 1.0\}$ . As illustrated in the top row of Figure 4, our method consistently outperforms other approaches across all participation rates, showcasing a faster convergence rate. Notably, as the participation rate decreases, several methods display highly unstable convergence. This instability is expected, as a lower client participation rate amplifies the divergence between randomly participating clients and the global model, resulting in erratic convergence. In contrast, our method exhibits a relatively stable convergence trend, highlighting its robustness to varying participation rates.

Table 4: Results under different values of hyperparameter $\lambda$ with Dirichlet-based label skews ( $\beta = 0.05$ ) on CIFAR10 and CIFAR100 datasets. 

<table><tr><td>λ</td><td>0.05</td><td>0.1</td><td>0.25</td><td>0.5</td><td>1</td></tr><tr><td>CIFAR10</td><td>74.70</td><td>75.71</td><td>75.47</td><td>75.29</td><td>74.98</td></tr><tr><td>CIFAR100</td><td>65.49</td><td>65.57</td><td>65.63</td><td>65.71</td><td>65.18</td></tr></table>

Table 5: Effectiveness of each loss function in FedVLS with Dirichlet-based label skews ( $\beta = 0.05$ ) on various datasets. (The value) represents the improvement over the first row. 

<table><tr><td> $\mathcal{L}_{\text{dis}}$ </td><td> $\mathcal{L}_{\text{logit}}$ </td><td>CIFAR10</td><td>CIFAR100</td><td>TinyImageNet</td></tr><tr><td>✗</td><td>✗</td><td>57.31</td><td>59.18</td><td>35.56</td></tr><tr><td>✗</td><td>√</td><td>70.25(+12.94)</td><td>64.24(+5.06)</td><td>39.04(+3.48)</td></tr><tr><td>√</td><td>✗</td><td>71.53(+14.22)</td><td>65.28(+6.10)</td><td>44.90(+9.34)</td></tr><tr><td>√</td><td>√</td><td>75.71(+18.40)</td><td>65.71(+6.53)</td><td>45.23(+9.67)</td></tr></table>

Impact of local epochs In this analysis, we investigate variations in the number of local epochs per communication round, represented as E, considering values from $\{10, 20\}$ . An intriguing observation emerges, particularly noticeable when E equals 20: several methods, notably FedNTD (Lee et al. 2022), exhibit declining accuracy in the later stages of training, as depicted in the second row of Figure 4. This decline is attributed to larger E values, making these models more susceptible to overfitting local data distribution as training progresses. In contrast, our method sustains a consistent and improving performance even with larger E values and consistently outperforms all other methods.

Impact of client numbers To underscore the resilience of our method in scenarios involving an increasing number of clients, we divide the CIFAR10 dataset into 10 and 30 clients, showcasing their convergence curves in the final row of Figure 4. Remarkably, our method consistently outperforms the baseline methods, regardless of the number of clients. An interesting trend emerges where, with the expanding number of clients, many methods exhibit slower and less stable convergence. In contrast, FedVLS maintains a consistent trend of rapid and stable convergence across these varied client numbers. Additional ablation study results concerning participating rates, local epochs, and the number of clients can be found in the technical appendix.

Impact of different backbones Apart from MobileNetV2, we conduct experiments using ResNet18 and ResNet32. The skew parameter, denoted as $\beta$ , is set to 0.1 and 0.05. The results are presented in Table 3, demonstrating our method, FedVLS, consistently outperforms the baseline methods. These experiments underscore the versatility and robustness of FedVLS in real-world federated learning scenarios employing various backbone architectures.

Robustness to hyperparameter $\lambda$ To demonstrate the robustness of our method concerning hyperparameter selection, we conduct experiments using various values of $\lambda$ on the CIFAR10 and CIFAR100 datasets. The findings, presented in Table 4, illustrate that our method exhibits insensitivity to the parameter $\lambda$ . Across $\lambda \in$ $\{0.05, 0.1, 0.25, 0.5, 1\}$ , our method consistently achieves approximately $75\%$ accuracy on CIFAR10 and $65.5\%$ on CIFAR100. This consistent performance highlights our method's ability to deliver stable results regardless of variations in $\lambda$ values, underscoring its robustness to hyperparameter changes.

Table 6: Results of combining FedVLS with other methods under Dirichlet-based label skews ( $\beta = 0.05$ ) across various datasets. (The values) represent the performance gains. 

<table><tr><td>Method(venue)</td><td>CIFAR10</td><td>CIFAR100</td><td>TinyImageNet</td></tr><tr><td>FedLC (ICML 2022)</td><td>57.31</td><td>59.18</td><td>35.56</td></tr><tr><td>+ FedVLS (Ours)</td><td>75.71(+18.40)</td><td>65.71(+6.53)</td><td>45.23(+9.67)</td></tr><tr><td>FedEXP (ICLR 2023)</td><td>63.22</td><td>62.24</td><td>34.95</td></tr><tr><td>+ FedVLS (Ours)</td><td>75.80(+12.58)</td><td>65.94(+3.70)</td><td>44.76(+9.81)</td></tr><tr><td>FedSAM (ICML2022)</td><td>63.11</td><td>61.32</td><td>41.14</td></tr><tr><td>+ FedVLS (Ours)</td><td>75.92(+12.81)</td><td>65.46(+4.14)</td><td>48.12(+6.98)</td></tr></table>

Effectiveness of different objectives Our approach comprises two key objectives: vacant-classes distillation and logit suppression. The results, presented in Table 5, reveal that both vacant-classes distillation and logit suppression contribute to notable performance improvements compared to FedLC (Zhang et al. 2022). These results demonstrate the effectiveness of our two key objectives in enhancing the overall model performance in federated learning scenarios with significant label skews.

Combination with other techniques In this section, we integrate our method with two SOTA methods, FedEXP (Divyansh Jhunjhunwala 2023) and FedSAM (Qu et al. 2022), as detailed in Table 6. The combination of our method with FedEXP (Divyansh Jhunjhunwala 2023) and FedSAM (Qu et al. 2022) results in improved performance. This enhancement is reasonable because FedEXP focuses on optimizing the server update for an improved learning rate, and FedSAM emphasizes local gradient descent to achieve a smoother loss landscape. These elements complement well with our core idea, which finally results in enhanced performance when combined.

# Conclusion

We have observed that existing federated learning methods always perform poorly in vacant and minority classes, under skewed label distribution across clients. To overcome these challenges, we introduce FedVLS—an innovative methodology integrating vacant-class distillation and logit suppression simultaneously. The vacant-class distillation extracts pertinent knowledge regarding vacant classes from the global model for each client, while logit suppression is implemented to directly regularize non-label class logits, addressing the imbalance among majority and minority classes. Extensive results affirm the effectiveness of both components, surpassing previous state-of-the-art methods across diverse datasets and varying degrees of label skews. In future work, we will conduct a theoretical analysis of FedVLS, including convergence, privacy, fairness, and other pertinent considerations.

# Acknowledgments

This work was funded by the National Natural Science Foundation of China under Grants (62276256, U2441251) and the Young Elite Scientists Sponsorship Program by CAST (2023QNRC001).

# References

Acar, D. A. E.; Zhao, Y.; Navarro, R. M.; Mattina, M.; Whatmough, P. N.; and Saligrama, V. 2021. Federated learning based on dynamic regularization. arXiv preprint arXiv:2111.04263.

Chai, D.; Wang, L.; Yang, L.; Zhang, J.; Chen, K.; and Yang, Q. 2023. A Survey for Federated Learning Evaluations: Goals and Measures. arXiv preprint arXiv:2308.11841.

Chen, C.; Liu, Y.; Ma, X.; and Lyu, L. 2022. Calfat: Calibrated federated adversarial training with label skewness. In Proc. NeurIPS.

Chen, H.; Vikalo, H.; et al. 2023. The Best of Both Worlds: Accurate Global and Personalized Models through Federated Learning with Data-Free Hyper-Knowledge Distillation. In Proc. ICLR.

Cubuk, E. D.; Zoph, B.; Mane, D.; Vasudevan, V.; and Le, Q. V. 2019. Autoaugment: Learning augmentation policies from data. In Proc. CVPR.

Cui, Y.; Jia, M.; Lin, T.-Y.; Song, Y.; and Belongie, S. 2019. Class-balanced loss based on effective number of samples. In Proc. CVPR.

Deng, L. 2012. The mnist database of handwritten digit images for machine learning research. Proc. SPM.

Diao, Y.; Li, Q.; and He, B. 2024. Exploiting Label Skews in Federated Learning with Model Concatenation. In Proc. AAAI.

Divyansh Jhunjhunwala, G. J., Shiqiang Wang. 2023. FedExP: Speeding up Federated Averaging Via Extrapolation. In Proc. ICLR.

Fan, Z.; Yao, J.; Han, B.; Zhang, Y.; Wang, Y.; et al. 2024. Federated Learning with Bilateral Curation for Partially Class-Disjoint Data. Proc. NeurIPS.

Fan, Z.; Yao, J.; Zhang, R.; Lyu, L.; Wang, Y.; and Zhang, Y. 2023. Federated Learning under Partially Disjoint Data via Manifold Reshaping. Proc. JMLR.

Guo, S.; Wang, H.; and Geng, X. 2024. Dynamic heterogeneous federated learning with multi-level prototypes. Proc. PR.

Guo, S.; Wang, H.; Lin, S.; Kou, Z.; and Geng, X. 2024. Addressing Skewed Heterogeneity via Federated Prototype Rectification With Personalization. Proc. TNNLS.

Hsu, T.-M. H.; Qi, H.; and Brown, M. 2019. Measuring the effects of non-identical data distribution for federated visual classification. arXiv preprint arXiv:1909.06335.

Huang, W.; Ye, M.; Shi, Z.; Li, H.; and Du, B. 2023. Rethinking federated learning with domain shift: A prototype view. In Proc. CVPR.

Itahara, S.; Nishio, T.; Koda, Y.; Morikura, M.; and Yamamoto, K. 2021. Distillation-based semi-supervised federated learning for communication-efficient collaborative training with non-iid private data. IEEE Transactions on Mobile Computing.   
Jeong, E.; Oh, S.; Kim, H.; Park, J.; Bennis, M.; and Kim, S.-L. 2018. Communication-efficient on-device machine learning: Federated distillation and augmentation under non-iid private data. In Proc. NeurIPS Workshops.   
Kairouz, P.; McMahan, H. B.; Avent, B.; Bellet, A.; Bennis, M.; Bhagoji, A. N.; Bonawitz, K.; Charles, Z.; Cormode, G.; Cummings, R.; et al. 2021. Advances and open problems in federated learning. Foundations and Trends® in Machine Learning.   
Karimireddy, S. P.; Kale, S.; Mohri, M.; Reddi, S.; Stich, S.; and Suresh, A. T. 2020. Scaffold: Stochastic controlled averaging for federated learning. In Proc. ICML.   
Konečný, J.; McMahan, H. B.; Ramage, D.; and Richtárik, P. 2016. Federated optimization: Distributed machine learning for on-device intelligence. arXiv preprint arXiv:1610.02527.   
Krizhevsky, A. 2009. Learning Multiple Layers of Features from Tiny Images. Master's thesis, University of Tront.   
Le, Y.; and Yang, X. 2015. Tiny imagenet visual recognition challenge. CS 231N.   
Lee, G.; Jeong, M.; Shin, Y.; Bae, S.; and Yun, S.-Y. 2022. Preservation of the global knowledge by not-true distillation in federated learning. In Proc. NeurIPS.   
Lee, T.; and Yoon, S. W. 2024. Rethinking the Flat Minima Searching in Federated Learning. In Proc. ICML.   
Li, B.; Schmidt, M. N.; Alstrøm, T. S.; and Stich, S. U. 2023. On the effectiveness of partial variance reduction in federated learning with heterogeneous data. In Proc. CVPR.   
Li, D.; and Wang, J. 2019. Fedmd: Heterogenous federated learning via model distillation. In Proc. NeurIPS Workshops.   
Li, Q.; Diao, Y.; Chen, Q.; and He, B. 2022. Federated learning on non-iid data silos: An experimental study. In Proc. ICDE.   
Li, Q.; He, B.; and Song, D. 2021. Model-contrastive federated learning. In Proc. CVPR.   
Li, T.; Sahu, A. K.; Talwalkar, A.; and Smith, V. 2020a. Federated learning: Challenges, methods, and future directions. Proc. SPM.   
Li, T.; Sahu, A. K.; Zaheer, M.; Sanjabi, M.; Talwalkar, A.; and Smith, V. 2020b. Federated optimization in heterogeneous networks. In Proc. MLSys.   
Li, X.; Huang, K.; Yang, W.; Wang, S.; and Zhang, Z. 2019. On the convergence of fedavg on non-iid data. In Proc. ICLR.   
Li, X.-C.; and Zhan, D.-C. 2021. Fedrs: Federated learning with restricted softmax for label distribution non-iid data. In Proc. KDD.   
Lin, T.; Kong, L.; Stich, S. U.; and Jaggi, M. 2020. Ensemble distillation for robust model fusion in federated learning. In Proc. NeurIPS.

Liu, D.; Bai, L.; Yu, T.; and Zhang, A. 2022. Towards Method of Horizontal Federated Learning: A Survey. In Proc. BigDIA.   
Lu, J.; Li, S.; Bao, K.; Wang, P.; Qian, Z.; and Ge, S. 2023. Federated Learning with Label-Masking Distillation. In Proc. ACM-MM.   
Luo, K.; Wang, S.; Fu, Y.; Li, X.; Lan, Y.; and Gao, M. 2023. DFRD: Data-Free Robustness Distillation for Heterogeneous Federated Learning. In Proc. NeurIPS.   
Luo, M.; Chen, F.; Hu, D.; Zhang, Y.; Liang, J.; and Feng, J. 2021. No fear of heterogeneity: Classifier calibration for federated learning with non-iid data. In Proc. NeurIPS.   
Luo, Z.; Wang, Y.; and Wang, Z. 2024. Federated Local Compact Representation Communication: Framework and Application. Proc. MIR.   
Luo, Z.; Wang, Y.; Wang, Z.; Sun, Z.; and Tan, T. 2022. Disentangled federated learning for tackling attributes skew via invariant aggregation and diversity transferring. arXiv preprint arXiv:2206.06818.   
Ma, Y.; Jiao, L.; Liu, F.; Yang, S.; Liu, X.; and Li, L. 2023. Curvature-Balanced Feature Manifold Learning for Long-Tailed Classification. In Proc. CVPR.   
McMahan, B.; Moore, E.; Ramage, D.; Hampson, S.; and y Arcas, B. A. 2017. Communication-efficient learning of deep networks from decentralized data. In Proc. AISTATS.   
Menon, A. K.; Jayasumana, S.; Rawat, A. S.; Jain, H.; Veit, A.; and Kumar, S. 2021. Long-tail learning via logit adjustment. In Proc. ICLR.   
Mu, X.; Shen, Y.; Cheng, K.; Geng, X.; Fu, J.; Zhang, T.; and Zhang, Z. 2023. Fedproc: Prototypical contrastive federated learning on non-iid data. Proc. FGCS.   
Qu, Z.; Li, X.; Duan, R.; Liu, Y.; Tang, B.; and Lu, Z. 2022. Generalized federated learning via sharpness aware minimization. In Proc. ICML.   
Reguieg, H.; El Hanjri, M.; El Kamili, M.; and Kobbane, A. 2023. A Comparative Evaluation of FedAvg and Per-FedAvg Algorithms for Dirichlet Distributed Heterogeneous Data. In Proc. WINCOM.   
Sandler, M.; Howard, A.; Zhu, M.; Zhmoginov, A.; and Chen, L.-C. 2018. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proc. CVPR.   
Sheller, M. J.; Edwards, B.; Reina, G. A.; Martin, J.; Pati, S.; Kotrotsou, A.; Milchenko, M.; Xu, W.; Marcus, D.; Colen, R. R.; et al. 2020. Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data. Scientific reports.   
Shen, Y.; Wang, H.; and Lv, H. 2023. Federated Learning with Classifier Shift for Class Imbalance. arXiv preprint arXiv:2304.04972.   
Shi, Y.; Liang, J.; Zhang, W.; Tan, V. Y.; and Bai, S. 2023a. Towards Understanding and Mitigating Dimensional Collapse in Heterogeneous Federated Learning. In Proc. ICLR.   
Shi, Y.; Liang, J.; Zhang, W.; Xue, C.; Tan, V. Y.; and Bai, S. 2023b. Understanding and Mitigating Dimensional Collapse in Federated Learning. IEEE Transactions on Pattern Analysis and Machine Intelligence.

T Dinh, C.; Tran, N.; and Nguyen, J. 2020. Personalized federated learning with moreau envelopes. In Proc. NeurIPS.   
Tan, J.; Wang, C.; Li, B.; Li, Q.; Ouyang, W.; Yin, C.; and Yan, J. 2020. Equalization loss for long-tailed object recognition. In Proc. CVPR.   
Wang, H.; Li, Y.; Xu, W.; Li, R.; Zhan, Y.; and Zeng, Z. 2023a. DaFKD: Domain-aware Federated Knowledge Distillation. In Proc. CVPR.   
Wang, Y.; Li, R.; Tan, H.; Jiang, X.; Sun, S.; Liu, M.; Gao, B.; and Wu, Z. 2023b. Federated Skewed Label Learning with Logits Fusion. arXiv preprint arXiv:2311.08202.   
Weyand, T.; Araujo, A.; Cao, B.; and Sim, J. 2020. Google landmarks dataset v2-a large-scale benchmark for instance-level recognition and retrieval. In Proc. CVPR.   
Wu, Z.; Sun, S.; Wang, Y.; Liu, M.; Jiang, X.; and Li, R. 2023. Survey of Knowledge Distillation in Federated Edge Learning. arXiv preprint arXiv:2301.05849.   
Xiao, Z.; Chen, Z.; Liu, S.; Wang, H.; Feng, Y.; Hao, J.; Zhou, J. T.; Wu, J.; Yang, H. H.; and Liu, Z. 2023. Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer. arXiv preprint arXiv:2310.07587.   
Yang, H.; Fang, M.; and Liu, J. 2021. Achieving linear speedup with partial worker participation in non-iid federated learning. In Proc. ICLR.   
Ye, M.; Fang, X.; Du, B.; Yuen, P. C.; and Tao, D. 2023. Heterogeneous federated learning: State-of-the-art and research challenges. ACM Computing Surveys.   
Yeganeh, Y.; Farshad, A.; Navab, N.; and Albarqouni, S. 2020. Inverse distance aggregation for federated learning with non-iid data. In Proc. MICCAI Workshops.   
Zeng, Y.; Liu, L.; Liu, L.; Shen, L.; Liu, S.; and Wu, B. 2023. Global Balanced Experts for Federated Long-Tailed Learning. In Proc. ICCV.   
Zhang, J.; Li, C.; Qi, J.; and He, J. 2023. A Survey on Class Imbalance in Federated Learning. arXiv preprint arXiv:2303.11673.   
Zhang, J.; Li, Z.; Li, B.; Xu, J.; Wu, S.; Ding, S.; and Wu, C. 2022. Federated learning with label distribution skew via logits calibration. In Proc. ICML.   
Zhao, B.; Cui, Q.; Song, R.; Qiu, Y.; and Liang, J. 2022. Decoupled knowledge distillation. In Proc. CVPR.   
Zhu, H.; Xu, J.; Liu, S.; and Jin, Y. 2021. Federated learning on non-IID data: A survey. Neurocomputing.   
Zhu, Z.; Hong, J.; and Zhou, J. 2021. Data-free knowledge distillation for heterogeneous federated learning. In Proc. ICML.

# Technical Appendix

# The Pseudocode of Our Method

Algorithm 1: FedVLS   
Input: number of communication rounds T, number of clients N, client participating rate R, number of local epochs E, batch size B, learning rate $\eta$ .

Output: the global model $\omega^{T}$ 1: initialize $\omega^{0}$ 2: $m \leftarrow \max(\lfloor R \cdot N \rfloor, 1)$ 3: for communication round $t = 1, 2, \cdots, T - 1$ do

4: $M_{t} \leftarrow$ randomly select a subset containing m clients

5: for each client $i \in M_{t}$ do

6: $\omega_{i}^{t} = \omega^{t}$ 7: $\omega_{i}^{t+1} \leftarrow \text{LocalUpdate}(\omega_{i}^{t})$ 8: end for

9: $\omega^{t+1} = \omega^{t} + \sum_{i \in M_{t}} \frac{|\mathcal{D}_{i}|}{|\mathcal{D}|} (\omega_{i}^{t+1} - \omega_{i}^{t})$ 10: end for

11: LocalUpdate( $\omega_{i}^{t}$ ):
12: for epoch $e = 1, 2, \cdots, E$ do

13: for each batch $B_{i} = \{x, y\} \in D_{i}$ do

14: $\mathcal{L}_{\text{cal}}(\omega; \mathcal{B}_{i}) = -\mathbb{E}_{(x,y) \sim \mathcal{B}_{i}} \log \left( \frac{p(y) \cdot e^{f(x;\omega)[y]}}{\sum_{c} p(c) \cdot e^{f(x;\omega)[c]}} \right)$ 15: $\mathcal{L}_{\text{dis}}(\omega; \mathcal{B}_{i}) = \mathbb{E}_{(x,y) \sim \mathcal{B}_{i}} \sum_{o \in O} q^{\mathbf{g}}(o; x) \log \left[ \frac{q(o;x)}{q^{\mathbf{g}}(o;x)} \right]$ 16: $\mathcal{L}_{\text{logit}}^{c}(\omega; \mathcal{B}_{i}) = \log \left( \mathbb{E}_{(x,y) \sim \mathcal{B}_{i}} I(y \neq c) \cdot e^{f(x;\omega)[c]} \right)$ 17: $\mathcal{L}_{\text{logit}}(\omega; \mathcal{B}_{i}) = \sum p(c) \cdot \mathcal{L}_{\text{logit}}^{c}(\omega; \mathcal{B}_{i})$ 18: $\mathcal{L}(\omega_{i}^{t}; \mathcal{B}_{i}) = \mathcal{L}_{\text{cal}}(\omega_{i}^{t}; \mathcal{B}_{i}) + \lambda \cdot \mathcal{L}_{\text{dis}}(\omega_{i}^{t}; \mathcal{B}_{i}) + \mathcal{L}_{\text{logit}}(\omega_{i}^{t}; \mathcal{B}_{i})$ 19: $\omega_{i}^{t} = \omega_{i}^{t} - \eta \nabla L(\omega_{i}^{t}; \mathcal{B}_{i})$ 20: end for

21: end for

22: return $\omega_{i}^{t}$

# Experimental Details

# Data Distribution among Clients

In Figure 1 (a) of the main paper, all clients' data distributions are independent and identically sampled. In Figures1 (b), (c), (d) of the main paper, the data distribution of all clients is shown in Table 7 as follows. We focus on client 0 for analysis, where it is evident that classes 5, 8, and 9 are majority classes, class 3 is a minority class, and the remaining classes are vacant.

In Figure 2 of the main paper, the data distribution for this client is shown in the fourth column of Figure 7 (a). Here, classes 0, 1, 3, and 7 are majority classes, while classes 2, 5, and 6 are minority classes. Figure 6 reveals that minority classes are frequently misclassified as majority classes,

which motivates the introduction of Logit Suppression in the main paper.

In our experiments, we incorporate Dirichlet-based label skews ( $\beta = 0.5, 0.1, 0.05$ ) and quantity-based label skews (s=2) for the CIFAR10 dataset. The data distribution for these skews is illustrated in Figure 7.

Table 7: The data distribution among clients with Dirichlet-based ( $\beta = 0.1$ ) CIFAR10 datasets. 

<table><tr><td>client</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td></tr><tr><td>class 0</td><td>0</td><td>57</td><td>0</td><td>600</td><td>0</td><td>4342</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>class 1</td><td>0</td><td>155</td><td>0</td><td>0</td><td>1</td><td>679</td><td>4153</td><td>0</td><td>11</td><td>1</td></tr><tr><td>class 2</td><td>0</td><td>3</td><td>24</td><td>0</td><td>15</td><td>0</td><td>3536</td><td>1419</td><td>0</td><td>3</td></tr><tr><td>class 3</td><td>141</td><td>99</td><td>3490</td><td>953</td><td>0</td><td>0</td><td>0</td><td>0</td><td>208</td><td>109</td></tr><tr><td>class 4</td><td>0</td><td>0</td><td>98</td><td>1217</td><td>3684</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td></tr><tr><td>class 5</td><td>1471</td><td>0</td><td>3403</td><td>0</td><td>125</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>class 6</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4999</td><td>1</td><td>0</td></tr><tr><td>class 7</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>4998</td><td>0</td></tr><tr><td>class 8</td><td>1360</td><td>35</td><td>0</td><td>0</td><td>3604</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>class 9</td><td>366</td><td>4608</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>26</td></tr></table>

# Implementation Details

The augmentation for all CIFAR and TinyImageNet experiments is the same as existing literature AutoAugment (Cubuk et al. 2019). The specific architecture of MobileNetV2 (Sandler et al. 2018) is shown in Table 8, while the structure of the bottleneck is detailed in Table 9. Since the architectures of ResNet-18 and ResNet-32 are well-known, we do not present their detailed structures here. Hyperparameters for all baseline methods are set according to the configurations specified in the original papers, as detailed in Table 10. All experiments are conducted on a single NVIDIA GeForce RTX 3090 with 24GB of memory.

Table 8: The architecture of MobileNetV2. 

<table><tr><td>Input</td><td>Operator</td><td>t</td><td> $^*C^*$ </td><td> $^*n^*$ </td><td>s</td></tr><tr><td> $224^{2} \times 3$ </td><td>conv2d</td><td>-</td><td>32</td><td>1</td><td>2</td></tr><tr><td> $112^{2} \times 32$ </td><td>bottleneck</td><td>1</td><td>16</td><td>1</td><td>1</td></tr><tr><td> $112^{2} \times 16$ </td><td>bottleneck</td><td>6</td><td>24</td><td>2</td><td>2</td></tr><tr><td> $56^{2} \times 24$ </td><td>bottleneck</td><td>6</td><td>32</td><td>3</td><td>2</td></tr><tr><td> $28^{2} \times 32$ </td><td>bottleneck</td><td>6</td><td>64</td><td>4</td><td>2</td></tr><tr><td> $14^{2} \times 64$ </td><td>bottleneck</td><td>6</td><td>96</td><td>3</td><td>1</td></tr><tr><td> $14^{2} \times 96$ </td><td>bottleneck</td><td>6</td><td>160</td><td>3</td><td>2</td></tr><tr><td> $7^{2} \times 160$ </td><td>bottleneck</td><td>6</td><td>320</td><td>1</td><td>1</td></tr><tr><td> $7^{2} \times 320$ </td><td>conv2d1  $\times$  1</td><td>-</td><td>1280</td><td>1</td><td>1</td></tr><tr><td> $7^{2} \times 1280$ </td><td>avgpool7  $\times$  7</td><td>-</td><td>-</td><td>1</td><td>-</td></tr><tr><td> $1 \times 1 \times 1280$ </td><td>conv2d1  $\times$  1</td><td>-</td><td>k</td><td>-</td><td></td></tr></table>

# Additional Experimental Observations

In Figure 5, the updated local model's performance on classes 5 and 8 surpasses that of the initial global model.

![](images/6bd04c6a1a6e1582d00699a5647230751c7c1fceece5af45b61a558030499caf.jpg)

<details>
<summary>line</summary>

| Class | Number (Red Line) | Number (Green Dashed Line) |
|-------|-------------------|----------------------------|
| 0     | 420               | 380                        |
| 1     | 450               | 400                        |
| 2     | 380               | 320                        |
| 3     | 400               | 350                        |
| 4     | 420               | 380                        |
| 5     | 400               | 350                        |
| 6     | 350               | 280                        |
| 7     | 400               | 350                        |
| 8     | 450               | 420                        |
| 9     | 480               | 480                        |
</details>

![](images/005901416a4bf50ef48a2b462883461b32dd9d0490640d95af0c62e9476e87b2.jpg)

<details>
<summary>line</summary>

| Class | Red Line Value | Green Line Value |
|-------|----------------|------------------|
| 0     | 300            | 1100             |
| 1     | 350            | 1250             |
| 2     | 250            | 950              |
| 3     | 500            | 1000             |
| 4     | 300            | 1200             |
| 5     | 1400           | 700              |
| 6     | 0              | 1500             |
| 7     | 300            | 1300             |
| 8     | 1500           | 1450             |
| 9     | 1200           | 1500             |
</details>

![](images/5b068734bbd4ea521efaaf76265b4fb9a5388ebc9628e40461798f5b89624f0a.jpg)

<details>
<summary>line</summary>

| Class | Red Line Value | Green Line Value |
|-------|----------------|------------------|
| 0     | 100            | 1200             |
| 1     | 200            | 1300             |
| 2     | 300            | 1100             |
| 3     | 400            | 1200             |
| 4     | 600            | 1300             |
| 5     | 1300           | 1200             |
| 6     | 0              | 500              |
| 7     | 300            | 800              |
| 8     | 1500           | 1400             |
| 9     | 1400           | 1500             |
</details>

![](images/b37628911ca838a8d3bb163c48136cc4ab646e9d7d69fcd4de6542247ce67ab1.jpg)

<details>
<summary>bar_line</summary>

(d) FedVLS (Ours): 70.60%
| Class | Bar Value | Line Value |
| :--- | :--- | :--- |
| 0 | 1500 | 0.9 |
| 1 | 1200 | 0.8 |
| 2 | 1000 | 0.7 |
| 3 | 1500 | 0.8 |
| 4 | 1200 | 0.7 |
| 5 | 1500 | 0.8 |
| 6 | 500 | 0.5 |
| 7 | 1500 | 0.9 |
| 8 | 1500 | 1.0 |
| 9 | 350 | 1.0 |
</details>

Sample Num --- Initial Global Acc — Updated Local Acc

Figure 5: Class-wise accuracy of the initial global model and updated local model on IID and label-skewed CIFAR10 data distributions. (a) represents the result updating on IID local data with FedAvg (McMahan et al. 2017). (b-d) showcase the results updating on skewed local data distribution with FedAvg, FedLC (Zhang et al. 2022), and our FedVLS, respectively. The value (%) in each caption corresponds to the accuracy of the global model aggregated from updated local models.   
![](images/50ee66b365a866c91a061130fa0964f553a215e0c10bb05689c4cc73dd40c83a.jpg)

<details>
<summary>heatmap</summary>

Confusion Matrix of Client 3
| Truth label | Predict label 0 | Predict label 1 | Predict label 2 | Predict label 3 | Predict label 4 | Predict label 5 | Predict label 6 | Predict label 7 | Predict label 8 | Predict label 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.92 | 0.0 | 0.0 | 0.01 | 0.04 | 0.0 | 0.01 | 0.0 | 0.03 | 0.0 |
| 1 | 0.05 | 0.78 | 0.0 | 0.03 | 0.06 | 0.0 | 0.0 | 0.03 | 0.05 | 0.0 |
| 2 | 0.07 | 0.02 | 0.65 | 0.04 | 0.06 | 0.01 | 0.01 | 0.07 | 0.06 | 0.01 |
| 3 | 0.04 | 0.0 | 0.0 | 0.86 | 0.05 | 0.0 | 0.0 | 0.02 | 0.03 | 0.0 |
| 4 | 0.02 | 0.0 | 0.0 | 0.0 | 0.95 | 0.0 | 0.0 | 0.01 | 0.02 | 0.0 |
| 5 | 0.08 | 0.02 | 0.01 | 0.05 | 0.12 | 0.61 | 0.02 | 0.02 | 0.07 | 0.0 |
| 6 | 0.07 | 0.01 | 0.01 | 0.07 | 0.11 | 0.03 | 0.56 | 0.05 | 0.09 | 0.0 |
| 7 | 0.05 | 0.02 | 0.0 | 0.03 | 0.06 | 0.0 | 0.01 | 0.8 | 0.03 | 0.0 |
| 8 | 0.03 | 0.01 | 0.0 | 0.02 | 0.04 | 0.0 | 0.0 | 0.02 | 0.88 | 0.0 |
| 9 | 0.17 | 0.03 | 0.02 | 0.1 | 0.18 | 0.01 | 0.01 | 0.08 | 0.14 | 0.26 |
</details>

Figure 6: Confusion matrix of client 3 on CIFAR10 dataset with Dirichlet-based label skews ( $\beta = 0.5$ ) using FedLC (Zhang et al. 2022).

This improvement is due to our proposed loss function, which constrains the local model's output for vacant classes and suppresses the misclassification of minority samples. These adjustments have minimal impact on the learning of majority classes. Consequently, local models continue to acquire category knowledge from majority classes, such as classes 5 and 8, similar to FedAvg, resulting in enhanced classification accuracy for these classes.

Another interesting observation is that both FedLC (Zhang et al. 2022) and our method reduce the accuracy of classes 5 and 8 while increasing the accuracy of the remaining classes. The reason for this behavior is as follows: In FedAvg (McMahan et al. 2017), the local model often misclassifies vacant and minority classes as majority classes. This leads to disproportionately high accuracy for the majority classes and extremely low accuracy for the

Table 9: The architecture of bottleneck. 

<table><tr><td>Input</td><td>Operator</td><td>Output</td></tr><tr><td> $h \times w \times k$ </td><td> $1 \times 1$  conv2d, ReLU6</td><td> $h \times w \times (tk)$ </td></tr><tr><td> $nh \times w \times tk$ </td><td> $3 \times 3$  dwise s=s, ReLU6</td><td> $\frac{h}{s} \times \frac{w}{s} \times (tk) \frac{h}{s} \times \frac{w}{s} \times k'$ </td></tr><tr><td> $n\frac{h}{s} \times \frac{w}{s} \times tk$ </td><td>linear1  $\times$  1 conv2d</td><td> $\frac{h}{s} \times \frac{w}{s} \times (tk) \frac{h}{s} \times \frac{w}{s} \times k'$ </td></tr></table>

Table 10: The hyperparameters for all baseline methods. 

<table><tr><td>FedAvg (AISTATS 2017)</td><td>None</td></tr><tr><td>FedProx (MLSys 2020)</td><td> $\mu=0.01$ </td></tr><tr><td>MOON (CVPR 2021)</td><td> $\mu=0.01, \tau=0.5$ </td></tr><tr><td>FedEXP (ICLR 2023)</td><td> $\epsilon=0.01$ </td></tr><tr><td>FedLC (ICML 2022)</td><td> $\tau=0.5$ </td></tr><tr><td>FedRS (KDD 2021)</td><td> $\alpha=0.7$ </td></tr><tr><td>FedSAM (ICML2022)</td><td> $\rho=0.1, \beta=0.9$ </td></tr><tr><td>FedNTD (NeurIPS 2022)</td><td> $\beta=0.1$ </td></tr><tr><td>FedMR (TMLR 2023)</td><td>deco=4</td></tr><tr><td>FedLMD (MM 2023)</td><td> $\beta=0.1$ </td></tr><tr><td>FedConcat (AAAI 2024)</td><td> $cluster=\{2, 4\}$ </td></tr><tr><td>FedGF (ICML 2024)</td><td> $\rho=0.1, c_{o}s=0.3$ </td></tr></table>

minority and vacant classes.

FedLC (Zhang et al. 2022) employs logit weighting to enhance the learning of minority classes, which can result in some majority class samples being misclassified as similar minority classes. As a result, this method improves accuracy for minority classes while slightly reducing accuracy for majority classes. In contrast, our method introduces vacant-class distillation and logit suppression to substantially mitigate the misclassification of minority and vacant classes as majority classes. This approach improves accuracy for vacant and minority classes but may cause some majority class samples to be misclassified as similar vacant or minority classes. Consequently, while this slightly reduces accuracy for the majority classes, it significantly enhances the overall performance of the local models.

![](images/2af1205dd2458a9ed3fc81eca82e90b2bee0d59fbd47ce3b0d14a46af2ed5238.jpg)

<details>
<summary>heatmap</summary>

| Label | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
|-------|------|------|------|------|------|------|------|------|------|------|
| 0     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 1     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 2     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 3     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 4     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 5     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 6     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 7     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 8     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
| 9     | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    |
</details>

![](images/44fbee42ec4ddb446384a043322bebf7aa8eea69865eebd6637bcf00a405f3cc.jpg)

<details>
<summary>heatmap</summary>

(b) β=0.1
Label | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 0 | 1000 | 0 | 0 | 0 | 0 | 0 | 0 | 4000 | 0 |
| 8 | 0 | 4000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4000 |
| 7 | 0 | 0 | 4000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 2000 | 0 | 0 | 0 | 4000 | 2000 | 0 | 0 |
| 5 | 1000 | 1000 | 2000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 2000 |
| 4 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 3 | 1000 | 2000 | 2000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 2 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 4000 | 2000 | 1000 | 100 |
| 1 | 1500 | 1500 | 1500 | 1500 | 1500 | 1500 | 1500 | 1500 | 1500 | 15 |
| 9 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 4555 |
The heatmap visualizes the distribution of values across Client IDs, with color intensity indicating magnitude based on the value scale. The label 'B' appears in the top-left corner. Values are estimated based on the color bar ranging from ~45 to ~48. The chart is a grid-based visualization with cell colors transitioning from dark blue (high values) to light blue (low values).
</details>

![](images/6ebed8da54c248ddec05cb51dc921ac325f5ec03ad6526150c93ddd931ed1e57.jpg)

<details>
<summary>heatmap</summary>

(c) β=0.05
Label | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 9 | 1000 | 2000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 1000 | 1000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | 1000 | 0 | 0 | 0 | 3000 | 0 | 0 | 0 | 4000 | 0 |
| 6 | 1000 | 0 | 2000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 100 |
| 5 | 1000 | 1000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 20 |
| 4 | 1000 | 1000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 20 |
| 3 | 1000 | 1000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 20 |
| 2 | 1000 | 1000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 20 |
| 1 | 1000 | 1000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 2000 | 20 |
| 9 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 |
The image contains a heatmap with color intensity based on the value scale. The label 'β' is not explicitly labeled in the chart.
</details>

![](images/30d50a11f9a0ef9a9267da5ae9743536765f6d7f5441aaeff4812be675c0d29e.jpg)

<details>
<summary>heatmap</summary>

(d) s=2
| Label | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 500 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 1 | 500 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 2 | 500 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 3 | 500 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 4 | 500 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 5 | 500 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 6 | 500 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 7 | 500 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 8 | 555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 |
| 9 | 555 | 2555 | 2555 | 2555 | 2555 | 2555 | 2555 | 2555 | 2555 | 2555 |
The data is a heatmap of values across the grid, with cell colors indicating magnitude based on the color scale (ranging from ~5 to ~42). The x-axis represents Client ID (ranging from 8 to 9), and the y-axis represents Label (ranging from -9 to -3). The color intensity reflects the value at each cell, with darker blue indicating higher values. The chart is labeled '(d) s=2' in the top-left corner.
</details>

Figure 7: Visualization of the Dirichlet-based ( $\beta = 0.5, 0.1, 0.05$ ) and quantity-based (s=2) label skews of CIFAR10 dataset among 10 clients.

# Additional Experimental Results

# The Experimental Results on the AG\_news Dataset

In this subsection, we add the experimental results on the AG\_news dataset with Dirichlet-based ( $\beta = 0.1$ and $\beta = 0.05$ ) and quantity-based (s=2) label skews, as shown in the Tab 11, demonstrating our method, FedVLS, consistently outperforms the base-line methods. These experiments underscore the versatility and robustness of FedVLS in real-world federated learning scenarios facing text classification.

# Compared to Other Knowledge Distillation Methods

To demonstrate the effectiveness of our vacant-class distillation, we compare it with existing class distillation, normal distillation, DKD (Zhao et al. 2022), and FedNTD (Lee et al. 2022). Similar to FedNTD (Lee et al. 2022), we integrate existing class distillation, normal distillation (KD), and DKD (Zhao et al. 2022) into FedAvg, denoted as FedEKD, FedKD, and FedDKD, respectively. As shown in Table 13, our method consistently outperforms these approaches.

Table 11: Performance overview for our method and baselines on the AG\_news dataset with Dirichlet-based ( $\beta=0.05$ and $\beta=0.1$ ) and quantity-based (s=2) label skews. Bold is the best result. 

<table><tr><td>Method(venue)</td><td> $\beta = 0.1$ </td><td> $\beta = 0.05$ </td><td> $s = 2$ </td></tr><tr><td>FedAvg (AISTATS 2017)</td><td>73.52</td><td>71.08</td><td>62.85</td></tr><tr><td>FedProx (MLSys 2020)</td><td>75.11</td><td>71.92</td><td>64.36</td></tr><tr><td>FedEXP (ICLR 2023)</td><td>78.08</td><td>72.35</td><td>63.01</td></tr><tr><td>FedSAM (ICML2022)</td><td>77.88</td><td>72.46</td><td>66.73</td></tr><tr><td>FedNTD (NeurIPS 2022)</td><td>79.14</td><td>75.60</td><td>69.28</td></tr><tr><td>FedLMD (MM 2023)</td><td>82.14</td><td>77.54</td><td>71.41</td></tr><tr><td>FedConcat (AAAI 2024)</td><td>81.59</td><td>74.84</td><td>68.11</td></tr><tr><td>FedGF (ICML 2024)</td><td>82.76</td><td>77.09</td><td>70.28</td></tr><tr><td>FedVLS (Ours)</td><td>87.31</td><td>83.19</td><td>77.46</td></tr></table>

To investigate the underlying reasons, we further examined the class-wise accuracy of the initial global model and the local models trained using these methods on client 0, whose data distribution is detailed in Table 7. The specific class-wise accuracy results are presented in Table 12. FedEKD shows minimal improvement in majority classes

Table 12: The class-wise accuracy for different knowledge distillation methods with Dirichlet-based ( $\beta = 0.1$ ) CIFAR10 datasets. 

<table><tr><td>class</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>Avg</td></tr><tr><td>global model</td><td>72.20</td><td>90.90</td><td>71.30</td><td>72.10</td><td>84.40</td><td>73.60</td><td>86.20</td><td>73.80</td><td>90.60</td><td>93.80</td><td>80.89</td></tr><tr><td>FedAvg</td><td>0</td><td>0</td><td>0</td><td>44.10</td><td>0</td><td>98.40</td><td>0</td><td>0</td><td>97.90</td><td>95.90</td><td>33.63</td></tr><tr><td>FedEKD</td><td>0</td><td>0</td><td>0</td><td>45.30</td><td>0</td><td>98.80</td><td>0</td><td>0</td><td>98.90</td><td>94.90</td><td>33.79</td></tr><tr><td>FedKD</td><td>1.50</td><td>5.10</td><td>1.80</td><td>51.90</td><td>1.60</td><td>94.80</td><td>1.20</td><td>0</td><td>97.40</td><td>95.50</td><td>35.08</td></tr><tr><td>FedDKD</td><td>3.90</td><td>34.20</td><td>16.60</td><td>52.50</td><td>9.80</td><td>94.10</td><td>14.40</td><td>0.10</td><td>97.60</td><td>96.00</td><td>41.92</td></tr><tr><td>FedNTD</td><td>8.00</td><td>38.80</td><td>22.60</td><td>58.10</td><td>11.10</td><td>96.10</td><td>12.20</td><td>0.20</td><td>98.10</td><td>94.60</td><td>43.98</td></tr><tr><td>Ours</td><td>40.40</td><td>71.20</td><td>39.60</td><td>64.60</td><td>50.07</td><td>83.50</td><td>54.80</td><td>41.30</td><td>92.30</td><td>94.32</td><td>63.21</td></tr></table>

Table 13: Performance overview for different knowledge distillation methods under Dirichlet-based label skews. 

<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR100</td><td colspan="2">TinyImageNet</td></tr><tr><td> $\beta = 0.1$ </td><td> $\beta = 0.05$ </td><td> $\beta = 0.1$ </td><td> $\beta = 0.05$ </td><td> $\beta = 0.1$ </td><td> $\beta = 0.05$ </td></tr><tr><td>FedAvg</td><td>82.00</td><td>62.90</td><td>66.18</td><td>62.13</td><td>39.90</td><td>35.21</td></tr><tr><td>FedEKD</td><td>81.25</td><td>62.26</td><td>67.66</td><td>62.90</td><td>40.95</td><td>36.13</td></tr><tr><td>FedKD</td><td>82.42</td><td>64.16</td><td>67.19</td><td>63.21</td><td>41.77</td><td>36.55</td></tr><tr><td>FedDKD</td><td>82.87</td><td>65.27</td><td>67.70</td><td>63.53</td><td>43.63</td><td>37.23</td></tr><tr><td>FedNTD</td><td>83.23</td><td>68.71</td><td>68.00</td><td>63.71</td><td>45.11</td><td>40.65</td></tr><tr><td>Ours</td><td>84.35</td><td>75.71</td><td>69.02</td><td>65.71</td><td>47.73</td><td>45.23</td></tr></table>

but significantly hinders the learning of vacant classes. FedKD, which uses distillation across all classes, still exhibits low accuracy for vacant classes. FedDKD adjusts distillation weights for true and not-true classes, while Fed-NTD applies distillation to not-true classes. Although these methods improve accuracy for vacant classes, there remains a substantial gap compared to the global model. Based on these observations, we believe that performing distillation on majority and minority classes will weaken the protection of information for vacant classes. Therefore, we use vacant-class distillation. The results in Table 12 further demonstrate that our method significantly enhances the accuracy for vacant classes, finally improving the performance of the local and global models.

# Combined with Methods for Domain Shift

Our method is specifically designed to address label skews, making it complementary to approaches that tackle domain skews. When both domain and label skews are present, our approach can further enhance the performance of methods like FPL (Huang et al. 2023). We have conducted experiments to validate this, with results presented in Table 14 and Table 15. Following the experimental setup in FPL (Huang et al. 2023), we use the Digits dataset and apply Dirichlet sampling to distribute the data for each domain among six clients. Under conditions of both domain and label skews, our method significantly improves the performance of PFL (Huang et al. 2023), demonstrating its effectiveness across different levels of label skews and domain shifts.

Table 14: Performance overview for FPL and our method combined with FPL in Dirichlet-based label skews, $\beta=0.1$ . Bold is the best result. 

<table><tr><td>Method</td><td>MNIST</td><td>USPS</td><td>SVHN</td><td>SYN</td><td>AVG</td></tr><tr><td>FPL</td><td>97.56</td><td>98.73</td><td>85.06</td><td>94.23</td><td>93.89</td></tr><tr><td>FPL + Ours</td><td>98.36</td><td>98.40</td><td>86.66</td><td>95.38</td><td>94.70</td></tr></table>

Table 15: Performance overview for FPL and our method combined with FPL in Dirichlet-based label skews, $\beta=0.05$ . Bold is the best result. 

<table><tr><td>Method</td><td>MNIST</td><td>USPS</td><td>SVHN</td><td>SYN</td><td>AVG</td></tr><tr><td>FPL</td><td>96.82</td><td>96.40</td><td>77.09</td><td>89.96</td><td>90.07</td></tr><tr><td>FPL + Ours</td><td>97.75</td><td>97.07</td><td>82.05</td><td>91.74</td><td>92.15</td></tr></table>

# Impact of Communication Rounds

In real-world scenarios, constraints often limit the number of available communication rounds. To address this, we evaluate the performance of various methods under different communication round limits using the CIFAR10 dataset with skew parameters $\beta = 0.1$ and $\beta = 0.05$ . The results, presented in Table 16, show that as the number of communication rounds decreases, the accuracy of most methods drops significantly. However, our method maintains high accuracy even with fewer communication rounds, demonstrating the robustness and efficiency of FedVLS in environments with restricted communication capabilities.

# Impact of Joining Rates, Local Epochs, and Client Numbers

Due to space constraints, we included only a portion of the ablation studies on joining rates, local epochs, and client numbers in the main paper. Here, we present the complete results. Specifically, we evaluated joining rates of 0.3, 0.5, 0.8, 1.0, local epochs of 5, 10, 15, 20, and client numbers of 10, 20, 30, 50. The experimental results are shown in Figure 8, and the observations are consistent with those presented in the main paper.

As the participation rate decreases, several methods exhibit highly unstable convergence. In contrast, our method

![](images/f2f699103fef3ef06944f6b2a5ce2eaa1bcd491a28d9900febb3b4b4fdbe20f9.jpg)  
Round
FedAvg    FedProx    FedLC    FedRS    FedNTD    FedLMD    FedVLS

Figure 8: Sensitivity analysis on the client participating rate R, local epochs E, and client numbers N. Each figure separately shows the convergence curve with Dirichlet-based label skews ( $\beta = 0.05$ ) on CIFAR10 dataset with R in $\{0.3, 0.5, 0.8, 1.0\}$ , E in $\{5, 10, 15, 20\}$ and N in $\{10, 20, 30, 50\}$ .

demonstrates relatively stable convergence, highlighting its robustness to varying participation rates.

Increasing the number of local epochs leads to declining accuracy in the later stages of training for several methods, notably FedNTD (Lee et al. 2022). However, our method maintains consistency and improves performance with larger E values, consistently outperforming other methods.

With an increasing number of clients, many methods show slower and less stable convergence. This is because the larger the number of clients, the greater the damage to model convergence caused by data heterogeneity among clients. However, our method maintains rapid and stable convergence across varying client numbers, demonstrating the robustness and scalability of our approach.

# Class-wise Accuracy

To evaluate the effectiveness of our approach, we conduct a comparative analysis of class-wise accuracy before and after local updates using our method, the classic method FedAvg (McMahan et al. 2017), and the state-of-the-art method FedLC (Zhang et al. 2022). For a fair comparison, we use the same well-trained federated model as the initial global model, which is then distributed to all clients. We train the local models using FedAvg and FedLC, and our method uses the same local data distribution. As shown in Figure 1 of the main paper, the results align with the observations discussed in the motivation section. Additionally, we compare the average class-wise accuracy for all clients after local updates and the class-wise accuracy for the aggregated global model of our approach with that of FedLC (Zhang et al. 2022), as demonstrated in Figure 9. Our method consistently achieves higher class-wise accuracy compared to FedLC, both after local updates and model aggregation.

Table 16: Results under varying numbers of communication rounds with Dirichlet-based label skews on CIFAR10 dataset. 

<table><tr><td rowspan="2">Method(venue)</td><td colspan="2">40 comm</td><td colspan="2">60 comm</td><td colspan="2">80 comm</td></tr><tr><td> $\beta=0.1$ </td><td> $\beta=0.05$ </td><td> $\beta=0.1$ </td><td> $\beta=0.05$ </td><td> $\beta=0.1$ </td><td> $\beta=0.05$ </td></tr><tr><td>FedAvg (AISTATS 2017)</td><td>74.62</td><td>53.44</td><td>78.59</td><td>56.71</td><td>80.72</td><td>59.10</td></tr><tr><td>FedProx (MLSys 2020)</td><td>78.59</td><td>57.67</td><td>81.63</td><td>61.84</td><td>82.88</td><td>61.96</td></tr><tr><td>MOON (CVPR 2021)</td><td>78.23</td><td>52.84</td><td>81.73</td><td>57.11</td><td>82.91</td><td>61.35</td></tr><tr><td>FedEXP (ICLR 2023)</td><td>75.90</td><td>54.14</td><td>79.69</td><td>55.98</td><td>81.51</td><td>60.01</td></tr><tr><td>FedLC (ICML 2022)</td><td>75.74</td><td>53.06</td><td>77.22</td><td>53.77</td><td>80.22</td><td>55.75</td></tr><tr><td>FedRS (KDD 2021)</td><td>79.10</td><td>60.99</td><td>81.13</td><td>63.16</td><td>82.94</td><td>64.28</td></tr><tr><td>FedSAM (ICML2022)</td><td>69.02</td><td>50.05</td><td>75.42</td><td>55.85</td><td>78.38</td><td>60.79</td></tr><tr><td>FedNTD (NeurIPS 2022)</td><td>81.26</td><td>65.75</td><td>82.23</td><td>66.48</td><td>82.95</td><td>67.91</td></tr><tr><td>FedLMD (MM 2023)</td><td>79.99</td><td>66.72</td><td>81.77</td><td>68.14</td><td>83.01</td><td>69.87</td></tr><tr><td>FedVLS (Ours)</td><td>82.54</td><td>72.90</td><td>83.82</td><td>74.34</td><td>84.30</td><td>75.25</td></tr></table>

These results highlight how our method effectively improves the performance of minority and vacant classes, leading to an overall enhancement in the global model's performance.

![](images/8a9a77f2f4f8cbb47e7159e9643673d101fcb82655754b855f9fa95c25b1c992.jpg)  
FedLC FedVLS

Figure 9: Comparison of class-wise accuracy after local update and after model aggregation with Dirichlet-based label skews ( $\beta = 0.05$ ) on CIFAR10 dataset.

# Model Bias among Clients

Thanks to the Vacant-class Distillation module, the client model will pay more attention to the vacant classes, which is beneficial to alleviate the model bias among clients. To demonstrate this, we conduct experiments to measure the drift diversity across all client models in the final round following (Li et al. 2023). Specially, the drift diversity is defined as follows:

$$
D r i f t = \frac {\sum_ {i = 1} ^ {N} \| m _ {i} \| ^ {2}}{\| \sum_ {i = 1} ^ {N} m _ {i} \| ^ {2}}, m _ {i} = \omega_ {i} ^ {T} - \omega^ {T} \tag {7}
$$

The results are presented in Table 17. It is evident that our approach effectively mitigates model bias among clients, leading to improved global performance.

# The Connection between Equation (2) of The Main Paper and FedLC

Apart from FedLC (Zhang et al. 2022), Fedshift (Shen, Wang, and Lv 2023) also adjusts the logits of model outputs to alleviate model bias caused by imbalanced data distributions. However, they have different forms, so we uniformly represent their loss functions using Eq(2). Nevertheless, during experiments, we train the models according to the original loss function forms as presented in the respective papers. Below, we demonstrate that Eq(2) is positively correlated to the loss function in FedLC (Zhang et al. 2022). In Eq(2),

Table 17: The drift diversity of different method on CIFAR10 datasets with $\beta = 0.1$ . 

<table><tr><td>Method</td><td>FedAvg</td><td>FedNTD</td><td>FedLC</td><td>FedVLS (Ours)</td></tr><tr><td>Drift diversity</td><td>29.73</td><td>17.85</td><td>12.11</td><td>8.37</td></tr></table>

$$
\mathcal {L} _ {\mathbf {c a l}} = - \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {i}} \log \left(\frac {p (y) \cdot e ^ {f (x ; \boldsymbol {\omega}) [ y ]}}{\sum_ {c} p (c) \cdot e ^ {f (x ; \boldsymbol {\omega}) [ c ]}}\right) \tag {8}
$$

$$
= - \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {i}} \log \left(\frac {e ^ {\ln p (y)} \cdot e ^ {f (x ; \boldsymbol {\omega}) [ y ]}}{\sum_ {c} e ^ {\ln p (c)} \cdot e ^ {f (x ; \boldsymbol {\omega}) [ c ]}}\right) \tag {9}
$$

$$
= - \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {i}} \log \left(\frac {e ^ {\ln p (y) + f (x ; \boldsymbol {\omega}) [ y ]}}{\sum_ {c} e ^ {\ln p (c) + f (x ; \boldsymbol {\omega}) [ c ]}}\right), \tag {10}
$$

where $p(y) = \frac{n_{y}}{n}$ , $n_{y}$ is the number of samples of class y in client i, and n is the total number of samples in client i. Therefore, Eq(2) can be rewritten in the following form.

$$
\mathcal {L} _ {\mathbf {c a l}} = - \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {i}} \log \left(\frac {e ^ {\ln \left(\frac {n _ {y}}{n}\right) + f (x ; \boldsymbol {\omega}) [ y ]}}{\sum_ {c} e ^ {\ln \left(\frac {n _ {c}}{n}\right) + f (x ; \boldsymbol {\omega}) [ c ]}}\right) \tag {11}
$$

$$
= - \mathbb {E} _ {(x, y) \sim \mathcal {D} _ {i}} \log \left(\frac {e ^ {f (x ; \boldsymbol {\omega}) [ y ] + \ln n _ {y} - \ln n}}{\sum_ {c} e ^ {f (x ; \boldsymbol {\omega}) [ c ] + \ln n _ {c} + \ln n}}\right) \tag {12}
$$

For different classes within the same client, n remains the same while $n_{y}$ varies. Therefore, the loss functions for different classes lie in $n_{y}$ and the output logits. Compared with the loss function in FedLC,

$$
\mathcal {L} _ {\mathrm{cal}} (y, f (x)) = - \log \left(\frac {e ^ {f _ {y} (x) - \tau \cdot n _ {y} ^ {(- 1 / 4)}}}{\sum_ {c \neq y} e ^ {f _ {c} (x) - \tau \cdot n _ {y} ^ {(- 1 / 4)}}}\right), \tag {13}
$$

$\ln n_{y}$ and $-\tau \cdot n_{y}^{(-1/4)}$ exhibit the same trend as $n_{y}$ changes, therefore they have similar effects on the loss function.