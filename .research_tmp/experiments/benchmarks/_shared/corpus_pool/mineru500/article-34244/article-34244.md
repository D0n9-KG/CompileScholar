# Boosting Test Performance with Importance Sampling—a Subpopulation Perspective

# Hongyu Shen $^{1}$ , Zhizhen Zhao $^{1}$

$^{1}$ Department of Electrical and Computer Engineering,

University of Illinois at Urbana-Champaign, Champaign, IL, 61820, U.S.A.

{hongyu2, zhizhenz}@illinois.edu

# Abstract

Despite empirical risk minimization (ERM) is widely applied in the machine learning community, its performance is limited on data with spurious correlation or subpopulation that is introduced by hidden attributes. Existing literature proposed techniques to maximize group-balanced or worst-group accuracy when such correlation presents, yet, at the cost of lower average accuracy. In addition, many existing works conduct surveys on different subpopulation methods without revealing the inherent connection between these methods, which could hinder the technology advancement in this area. In this paper, we identify important sampling as a simple yet powerful tool for solving the subpopulation problem. On the theory side, we provide a new systematic formulation of the subpopulation problem and explicitly identify the assumptions that are not clearly stated in the existing works. This helps to uncover the cause of the dropped average accuracy. We provide the first theoretical discussion on the connections of existing methods, revealing the core components that make them different. On the application side, we demonstrate a single estimator is enough to solve the subpopulation problem. In particular, we introduce the estimator in both attribute-known and -unknown scenarios in the subpopulation setup, offering flexibility in practical use cases. And empirically, we achieve state-of-the-art performance on commonly used benchmark datasets.

# 1 Introduction

Empirical risk minimization (ERM) often struggles with distribution shifts that manifest when the training and test distributions differ (Bickel, Brückner, and Scheffer 2007; Quionero-Candela et al. 2009; Shimodaira 2000). One ubiquitous type of distribution shift is subpopulation shift, which describes a scenario where the portion of the subpopulations may vary between training and testing sets. See Figure 1 for an example. This consequently leads to degraded performance when a trained model is applied to production/testing environments (Yang et al. 2023). Ensuring that machine learning models are robust against these distribution shifts hence is crucial for their reliability and safe real-world application.

![](images/28902038dc117fae2b6191f9f4da7948eac9895b0db8c5263fe811fbafde0f88.jpg)  
Figure 1: An image example on subpopulation shift. The left panel contains images where digits and colors are correlated, whereas the right panel does not exhibit such correlation.

Existing works proposed different methods in the forms of auxiliary losses (Li et al. 2018; Arjovsky et al. 2019; Alshammari et al. 2022), data augmentations (Zhang et al. 2018; Yao et al. 2022; Han et al. 2022), modeling objectives (Liu et al. 2021; Sagawa et al. 2020; Japkowicz 2000; Wu et al. 2023; Nam et al. 2020; Asgari et al. 2022; Rudner et al. 2024; Han and Zou 2024; Hong et al. 2023; Tsirigotis et al. 2024; Menon et al. 2021; Lin et al. 2017) and data sampling techniques (LaBonte, Muthukumar, and Kumar 2024; Izmailov et al. 2022; Japkowicz 2000). They all exhibit superior performance on worst group accuracy while maintaining high accuracy in the overall set. However, two recent works experimentally observed that most models experience a drop in average accuracy performance compared to the ERM setup despite the high worst group accuracy (Tsirigotis et al. 2024; Yang et al. 2023). Nonetheless, none of the papers is able to provide rigorous explanations on the answer to why. The lack of

clarity in understanding can impede the development of appropriate models and methods, potentially stalling progress in the field.

In this work, we propose a systematic dataset bias analysis (DBA) framework that is rooted in importance sampling. With this framework, we reveal the cause of the lower-than-ERM average accuracy is the mismatch between the learning objective and the testing dataset. Moreover, we identify the flexibility of this framework in interpreting the formulation of some of the existing works that focus primarily on statistical heuristics and do not clearly specify the underlying assumptions of the models or data. The DBA framework, on the other hand, can close the gap, allowing us to explicitly discuss assumptions systematically and compare different existing works with the same language. We believe this analysis offers a comprehensive and theoretically grounded view to people who wish to proceed with the study of subpopulation methods.

Practically, we propose to estimate a single distribution given the conducted analysis using the DBA framework and prove that this is enough for solving the subpopulation problem under certain assumptions. Subsequently, we propose 3 different methods for estimating the distribution given different access levels to data and attributes. Empirically, we demonstrate the framework improves the test performance under subpopulation setups and achieves state-of-the-art (SOTA) results for both average and worst group accuracy while avoiding the lower-to-ERM performance.

# 2 Related Work

In this section, we cover related works about importance sampling, the survey papers of the subpopulation shift, and the associated SOTA methods. Due to space limits, we only provide a concise version here and defer the complete version in Appendix A.

# 2.1 Importance Sampling

DBA interprets distributional shift as a mismatch of the weight function from an importance sampling perspective. Although primarily focused on subpopulation setups, the method's formulation applies broadly to distributional shift problems. Early works on importance sampling (Shimodaira 2000; Huang et al. 2006) address dataset shifts but lack real-world experiments and clarity for subpopulation cases. In contrast, DBA systematically formulates the application to subpopulation problems, explicitly stating assumptions and identifying key components like distributions leading to such issues. Other studies (Kanamori, Hido, and Sugiyama 2009; Fang et al. 2020) propose weight estimation methods requiring partial test set access, unlike DBA. Additionally, DBA considers the weight function as the ratio of joint distributions of $x$ and $y$ , addressing subpopulation and covariate shifts more realistically.

# 2.2 Subpopulation Survey

Yang et al. (2023) provides the first comprehensive experimental study on subpopulation methods. It uses Bayes' theorem to decompose $y|x$ , accounting for attributes (spurious features), and categorizes datasets into four classes with varying label-attribute correlations. The paper benchmarks 20 subpopulation methods across these datasets but lacks statistical quantification of performance differences. Other surveys (Yu et al. 2024; Zhang et al. 2023) cover broader out-of-distribution (OOD) and domain generalization (DG) methods. While Yu et al. (2024) focuses on applications, Zhang et al. (2023) quantifies error inflation due to distribution shifts but doesn't address correction via model design. Our work extends prior studies by providing formal statistical analysis to quantify errors from both data and modeling perspectives. DBA also explains why some methods trade worst-case accuracy for lower average test accuracy.

# 2.3 Subpopulation Method

We categorize subpopulation methods into four classes: auxiliary losses, data augmentations, modeling objectives, and data sampling techniques. Auxiliary loss methods (Li et al. 2018; Arjovsky et al. 2019; Alshammari et al. 2022) aim to mitigate the impact of spurious backgrounds via adversarial training, gradient regularization, or class-balanced adjustments. Data augmentation methods (Zhang et al. 2018; Han et al. 2022; Yao et al. 2022) use convex combinations of samples to reduce background effects. Data sampling methods (LaBonte, Muthukumar, and Kumar 2024; Izmailov et al. 2022) identify class-balanced subsets with independent spurious features for finetuning. Modeling objective methods (Sagawa et al. 2020; Wu et al. 2023; Lin et al. 2017; Rudner et al. 2024) focus on robust feature learning, subpopulation correction, or tailored loss terms like KL divergence or mutual information.

DBA stands out by explicitly stating data assumptions and connecting existing methods under a unified statistical framework (see Sec. 4). For instance, it highlights that augmentation methods (Zhang et al. 2018; Yao et al. 2022; Han et al. 2022) assume conditional similarity across subpopulations. DBA also identifies a universal assumption of identical conditional generative models across methods, which previous works did not explicitly address. Empirically, DBA outperforms SOTA methods on three datasets, confirming its effectiveness and simplicity, and leveraging importance sampling for practical implementation.

# 3 Method

The method section consists of two components: 1. the DBA framework; 2. three estimation methods for the only conditional outlined in the DBA framework, concerning three different scenarios where data assumptions vary. We first describe the framework, followed by the introduction of the proposed methods.

# 3.1 Dataset Bias Analysis Framework

Throughout the paper, we consider the following notations: $x \in X$ and $y \in Y$ indicate the random variables for the data and labels, respectively. X and Y refer to their corresponding spaces. We denote y as a discrete random variable. We use $p(\cdot)$ to denote the probability distribution and $q(\cdot)$ or $\hat{p}(\cdot)$ to represent the estimates. Subscripts “tr”, “va”, and “te” indicate concepts associated with train, validation, and test datasets, respectively. We use D to refer to the datasets. We let $M_{tr} := \{q(\cdot) | q(\cdot) \text{ estimated with data in } D_{\text{tr}}\}$ denote the model spaces for the general learning problem. s denotes the attributes/spurious variables that are present in the datasets. This is also the root of the subpopulation. And I refers to the dataset indicator, which is the abstract variable that has no real values (i.e. $I_{tr}, I_{va}$ , and $I_{te}$ ). We use Supp(·) to indicate the support set. We also use the notation “\~” on two datasets (e.g. $D_{tr} \sim D_{te}$ ) to represent the same data distributions for the given datasets.

The DBA framework is formulated by initially asking the question: Which model do we pick after training? Conventional approaches consider ERM over $D_{tr}$ , stop the training, and choose the model with the lowest loss value on $D_{va}$ . Usually, the losses are implicitly assumed to be identical across $D_{tr}$ , $D_{va}$ , and $D_{te}$ . There are two drawbacks to this inattentive assumption. First, it does not properly characterize the difference across different datasets. Second, it does not naturally take into account how people make choices on the model. As a remedy, we propose the following objective (Eq. (1)) as the foundation for the DBA framework:

$$
\mathbb {E} _ {(x, y) \sim p (x, y \mid I _ {\mathrm{va}})} [ \log q (y \mid x, I _ {\mathrm{tr}}) ]. \tag {1}
$$

The maximization of the objective (Eq. (2)) hence provides an intuitive view of how people choose the final model after the optimization:

$$
\max _ {q \in \mathcal {M} _ {\mathrm{tr}}} \mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{va}})} [ \log q (y | x, I _ {\mathrm{tr}}) ]. \tag {2}
$$

In this paper, we consistently focus on the predictive modeling setup (i.e. $y|x$ ), which is aligned with existing works. Intuitively, Eq. (2) describes the scenario where we find the best conditional predictive model q according to the highest log likelihood measured over $D_{va}$ . Eq. (2) differs from ERM by explicitly considering the inherent difference between different datasets. In most cases, we seek for models to perform well on the unseen $D_{te}$ . To characterize this, we apply a similar logic as in Eq. (1) and focus on measuring the difference between validation and test sets. We make the following universal assumption 1.

Assumption 1. The supports of $x, y$ on $\mathcal{D}_{tr}, \mathcal{D}_{va}$ , and $\mathcal{D}_{te}$ follow the relationship:

$$
\operatorname{Supp} _ {t r} (x, y) \supset \operatorname{Supp} _ {v a} (x, y), \operatorname{Supp} _ {t r} (x, y) \supset \operatorname{Supp} _ {t e} (x, y), a n d \operatorname{Supp} _ {v a} (x, y) \supset \operatorname{Supp} _ {t e} (x, y).
$$

The inclusion relationship described in the Assumption 1 essentially ensures a well-defined weight function (i.e., the denominator of the weight function is not zero) in the importance sampling setup in the proposed DBA framework. With this assumption, we make the following claim on the performance of the picked model (from Eq. (2)) with $D_{te}$ : How does the picked model perform on the test set?

Claim 1. Given Assumption 1 holds and let $q^{*}$ denote the best model obtained from Eq. (2). The likelihood evaluated with the test set $\mathcal{D}_{te}$ for the model $q^{*}$ can be viewed as the importance sampling version of $\mathbb{E}_{(x,y)\sim p(x,y|I_{va})}[z(x,y,I_{va},I_{te})\log q^{*}(y|x,I_{tr})]$ over the validation set with the function $z(\cdot)$ defined below:

$$
z (x, y, I _ {v a}, I _ {t e}) := \frac {p (x , y \mid I _ {t e})}{p (x , y \mid I _ {v a})}. \tag {3}
$$

We defer this and all the following proof details in Appendix B. Claim 1 informs that the only way to guarantee the best testing performance for the picked model $q^{*}$ is to have access to the distribution $p(x,y|I_{\mathrm{te}})$ . This points out a hidden pitfall that commonly exists, yet overlooked, in the current machine learning optimizations with ERM—people choose a model with the best validation performance and report the corresponding testing performance. By Claim 1, we know that this general setup is true only in the case where $p(x,y|I_{\mathrm{va}}) = p(x,y|I_{\mathrm{te}})$ . Otherwise, one needs to provide an accurate estimation on $z(x,y,I_{\mathrm{va}},I_{\mathrm{te}})$ and pick the training model via a weighted likelihood, $\mathbb{E}_{(x,y)\sim p(x,y|I_{\mathrm{va}})}[z(x,y,I_{\mathrm{va}},I_{\mathrm{te}})\log q^{*}(y|x,I_{\mathrm{tr}})]$ , on the validation set, to achieve optimal performance on the test set.

Simply put, Eq. (2) describes the way people pick the model during optimization, and Claim 1 points out the correct picking criterion for maximum test set performance. A natural follow-up question on these two arguments is: Can we combine the notion of training and picking, and directly optimize q to maximize the testing performance? The answer is affirmative under some additional assumptions. To explain, we first claim an optimization equivalence, providing the general form with which the optimization on the training set is identical to the optimization on the testing set (Claim 2). Then we derive another objective in the setup where we obtain a closed-form $g(x, y, I_{\mathrm{tr}}, I_{\mathrm{te}})$ (see Claim 2) after making several assumptions on the structure of the test data (Theorem 1).

Claim 2. Given Assumption 1 holds we obtain the following equality on the objective:

$$
\begin{array}{l} \mathbb {E} _ {(x, y) \sim p (x, y | I _ {t e})} [ \log q (y | x, I _ {t r}) ] \\ = \mathbb {E} _ {(x, y) \sim p (x, y \mid I _ {t r})} [ g (x, y, I _ {t r}, I _ {t e}) \log q (y \mid x, I _ {t r}) ], \tag {4} \\ \end{array}
$$

where the weight function $g(x,y,I_{tr},I_{te}):= \frac{p(x,y|I_{te})}{p(x,y|I_{tr})}$ .

The proof is similar to Claim 1 and can be found in Appendix B. Note that in the language of importance sampling, the weight function $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ consists of the proposal distribution $p(x,y|I_{\mathrm{tr}})$ and the data distribution $p(x,y|I_{\mathrm{te}})$ in our setup. Compared to Eq. (1), Eq. (4) offers an objective that can be optimized with $\mathcal{D}_{\mathrm{tr}}$ as the expectation is taken over the training set—the same space defined for models $q\in \mathcal{M}_{\mathrm{tr}}$ . Claim 2 also confirms that one must know $p(x,y|I_{\mathrm{te}})$ to improve the testing performance of $q$ when optimizing a model.

In this paper, we consider a uniform attribute setup that assumes the uniform distribution on the attribute/spurious variable $s \in S$ , which is a discrete random variable and $\operatorname{Supp}(s) = \operatorname{Supp}(y)$ . s represents the cause of the subpopulation in our study. Formally speaking, this paper considers the following subpopulation shift:

Definition 1. The subpopulation shift is defined as the distributional difference between $p(x,y|I_{\mathrm{tr}})$ and $p(x,y|I_{\mathrm{te}})$ that is introduced by the spurious variable $s$ w.r.t. the response $y$ . Namely, $p(s,y|I_{\mathrm{tr}}) \neq p(s,y|I_{\mathrm{te}})$ .

Specifically, we decompose the joint distribution of x and y through $\sum_{s} p(x,y,s|I_{\mathrm{tr}}) = \sum_{s} p(x|y,s,I_{\mathrm{tr}}) p(y,s|I_{\mathrm{tr}})$ , and $\sum_{s} p(x,y,s|I_{\mathrm{te}}) = \sum_{s} p(x|y,s,I_{\mathrm{te}}) p(y,s|I_{\mathrm{te}})$ . And the difference between datasets is on $p(s,y|I_{\mathrm{tr}}) \neq p(s,y|I_{\mathrm{te}})$ . In the following, we describe several assumptions that lead to the major result of the paper—Theorem 1:

Assumption 2. A universal data generator given the dataset information $I$ , the label $y$ , and the attribute $s$ for the training and test sets: $p(x|y, s, I_{tr}) = p(x|y, s, I_{te})$ .

Assumption 3. The attribute variable $s$ follows a uniform distribution, conditional on $y$ and $I_{te}$ : $p(s|y, I_{te}) = 1 / L$ , where $L$ is the number of outcomes for the discrete random variable $s$ .

Assumption 2 requires identical generative processes for x across training and testing. This can be seen as a specific type of covariate shift, attributing shifts in $p(x,y)$ to variations in $p(y,s)$ given the attribute s, rather than $p(x)$ . Such an assumption is common in conformal analysis and causal inference (Yang, Kuchibhotla, and Tchetgen Tchetgen 2024; Suter et al. 2019; Lei and Candès 2021). Assumption 3 imposes a weaker assumption compared to the literature, where uniformity and independence are generally assumed for both y and s (Tsirigotis et al. 2024). Compared to the existing work, we only assume s to follow a uniform distribution and there is no constraint on the distribution of y. The latter makes this approach applicable to class-imbalanced test data.

We further make two additional assumptions (Assumption 4 and 5) that reflect the nature of the considered subpopulation problems. This starts with studying the composition of the shifted datasets. Specifically, we introduce a random variable m that explicitly describes the substructure of the given data (i.e. $D_{tr}$ , $D_{va}$ , and $D_{te}$ ). Most existing works only consider the attribute variable s and its relation to labels y and data x. However, we realize that simply introducing this attribute is not enough to quantify the subpopulation as different subpopulations may have distinct relationships between s, y, and x. Therefore, the presence of m enables the quantification of such differences, making the proposed framework more flexible.

In particular, we consider m to be a binary random variable that takes values $m_{0}$ or $m_{1}$ . And $m_{0}$ refers to the conceptual minority group in $D_{tr}$ that shares the same statistics for s, y, and x in $D_{te}$ , whereas $m_{1}$ denotes the majority group that has distinct statistics of s, y, and possibly x—this explicitly characterizes the prevalent subpopulation in $D_{tr}$ that causes the underperformance in $D_{te}$ . One may question the soundness of why we claim it is possible to find such a minority group in $D_{tr}$ . An intuitive, yet not strict, proof is to consider the established Assumption 1 that constrains inclusive supports across datasets. With Assumption 1, we can always find a subset of $D_{tr}$ whose data statistics are close to that of $D_{te}$ for any possibly large enough datasets. This leads to the following assumption:

Assumption 4. $p(y|I_{te}) = p(y|m_0, I_{tr}) = p(y|I_{tr})$ .

Assumption 4 describes the scenario where there is no subpopulation on y between $D_{tr}$ and $D_{te}$ . This assumption indicates that the subpopulation is introduced by the association between s and y, or x and y, but not solely by y itself. Since the minority group $m_{0}$ shares same data statistics as y, it is natural to have the equality $p(y|I_{\mathrm{te}})=p(y|m_{0},I_{\mathrm{tr}})$ . It is noteworthy that there is no constraint on the number of groups specified by m. The size of 2 is considered in this paper due to its simplicity and high performance in practice (see Sec. 5).

Assumption 5, on the other hand, quantifies explicitly that there is a portion (i.e. $m_{1}$ ) of samples in $D_{tr}$ whose attributes s are identical to the labels y. Rather than treating it as an assumption, it is more of a characterization on the subpopulation that widely presents in the real-world data (e.g., Waterbirds and ColorMNIST, or others described in (Yang et al. 2023)), where attributes strongly mislead the model prediction by such correlation.

Assumption 5. $p(s|y, m_1, I_{tr}) = \mathbf{1}_{\{y=s\}}$ , where $\mathbf{1}_{\{y=s\}}$ is the indicator function.

With all ingredients, we propose the following theorem on the modeling objective:

Theorem 1. Given Assumption 1, 2, 3, 4, and 5 hold, the optimization of Eq. (4) with the following weight function $g(x,y,I_{tr},I_{te})$ directly maximizes the testing performance:

$$
g (x, y, I _ {t r}, I _ {t e}) ^ {- 1} := p \left(m _ {0} \mid I _ {t r}\right)
$$

$$
+ \frac {p (m _ {1} | I _ {t r}) \cdot \frac {L}{p (y | I _ {t r})} \cdot p (y | m _ {1} , I _ {t r})}{1 + \left[ \frac {p (m _ {0} | I _ {t r}) \cdot p (y | I _ {t r}) / L + p (y | m _ {1} , I _ {t r})}{p (m _ {0} | I _ {t r}) \cdot p (y | I _ {t r}) / L} \right] \cdot \frac {1 - p (s = y | y , x , I _ {t r})}{p (s = y | y , x , I _ {t r})}}, \tag {5}
$$

where $p(y|m_1, I_{tr}) = \frac{p(y|I_{tr}) - p(m_0|I_{tr}) \cdot p(y|I_{tr})}{p(m_1|I_{tr})}$ . $p(m_0|I_{tr})$ and $p(m_1|I_{tr})$ represent the probability of a binary random variable $m$ taking the value $m_0$ or $m_1$ , respectively. Namely, the random variable $m$ denotes the split of $\mathcal{D}_{tr}$ into the majority and minority groups.

The corresponding proof can be found in Appendix B. With this formulation, $p(s = y|y, x, I_{\mathrm{tr}})$ is the only unknown term to be estimated. Theorem 1 provides a closed form objective with which models trained with $D_{tr}$ perform optimally on $D_{te}$ . In the following, we consider 3 different setups on the accessibility of s and the relationship between $D_{tr}$ and $D_{va}$ . In each setup, we provide a method to estimate Eq. (5). We further showcase the performance of the proposed methods in the experiment section (Sec. 5). In Appendix C, we include a discussion on the limitations of this approach concerning the restriction and possible relaxation of the assumptions.

# 3.2 Dataset Bias Correction Method

In this section, we provide 3 different approaches to estimate Eq. (5). We summarize these approaches with a general name: dataset bias correction method (DBCM). The 3 approaches essentially provide different ways of estimating the only missing term $p(s|y, x, I_{\mathrm{tr}})$ in Eq. (5). Once the term is estimated, we employ a universal algorithm (see Algorithm 1) to train the model with $D_{tr}$ . To facilitate the use of this approach in more real-world applications, we describe the scenarios where the three following approaches can be applied in Appendix D.

Attribute s is Known When we have access to the attribute s, we can make a direct estimation on the only unknown term $p(s = y|y, x, I_{\mathrm{tr}})$ using the data $(x, y, s) \in \mathcal{D}_{\mathrm{tr}}$ and apply Algorithm 1 therein. As $p(s = y|y, x, I_{\mathrm{tr}})$ increases, the weight function g decreases (see Eq. (5)), because stronger spurious correlations make $p(s = y|y, x, I_{\mathrm{tr}})$ larger. Down-weighting these samples during training helps performance by reducing reliance on spurious correlations.

Attribute $s$ is Unknown and $\mathcal{D}_{\mathrm{tr}} \sim \mathcal{D}_{\mathrm{va}}$ When we do not have access to the attribute $s$ and $\mathcal{D}_{\mathrm{tr}} \sim \mathcal{D}_{\mathrm{va}}$ , we propose to use the following term to estimate $p(s|y, x, I_{\mathrm{tr}})$ :

$$
\hat {p} (s = y | y, x, I _ {\mathrm{tr}}) \propto \exp \left(\frac {| \log \hat {p} (y | x , I _ {\mathrm{tr}}) - \log \hat {p} (y | x , I _ {\mathrm{va}}) |}{\tau}\right) ^ {- 1}, \tag {6}
$$

where $\hat{p}(y|x,I_{\mathrm{tr}})$ and $\hat{p}(y|x,I_{\mathrm{va}})$ are the predictive models learned with $D_{tr}$ and $D_{va}$ , respectively. And $\tau$ is the temperature hyperparameter. In practice we find $\tau=1$ consistently produces good results. We explicitly introduce $\tau$ to allow flexibility in the control of the estimation in Eq. (6). Specifically, we first overfit two independent predictive models on both $D_{tr}$ and $D_{va}$ and then measure the difference on the two approximate laws with the training data. Note that $p(s=y|y,x,I_{\mathrm{tr}})$ captures how likely s shares the same label as y, which is the only unknown term evaluated in Eq. (5). Therefore, we do not need to recover the full distribution $p(s|y,x,I_{\mathrm{tr}})$ . Instead, we only need to quantify $\hat{p}(s=y|y,x,I_{\mathrm{tr}})$ —“how likely the bias is biased towards the true label y.” This is captured by Eq. (6), as if two models (trained separately on training and validation data) produce similar likelihoods (i.e. the difference in Eq. (6) is smaller) on a given input, then the input must associate with the attribute s that is same as y. To summarize this approach in one line: two overfitted models act as a bias corrector!

Attribute s is Unknown and $D_{tr} \sim D_{va}$ . On the other hand, when $D_{tr} \sim D_{va}$ , we cannot utilize the predictive model estimated with $D_{va}$ . Instead, we propose to use the following term as an alternative,

$$
\hat {p} (s = y | y, x, I _ {\mathrm{tr}}) \propto \exp \left(- \frac {\log \hat {p} (y | x , I _ {\mathrm{tr}})}{\tau}\right) ^ {- 1}. \tag {7}
$$

This is according to the observation that machine learning models tend to learn the correlated attributes s with y easily (Asgari et al. 2022). In our case, we simply use $\hat{p}(y|x, I_{\mathrm{tr}})$ as the proxy to characterize such correlation. In this case, samples with high accuracy should be down-weighted, as the model easily learns spurious correlations.

# 3.3 Choose Models

Similarly, we discuss different approaches for choosing a model. Unlike conventional methods that consistently use $D_{va}$ to decide which model to choose, we propose to consider different ways for choosing a model when relationships between $D_{va}$ and $D_{te}$ are different. When $D_{va} \sim D_{te}$ , according to Eq. (3), $z(x, y, I_{\mathrm{va}}, I_{\mathrm{te}}) = 1$ . This indicates that evaluating models on validation set is equivalent to evaluating on the test set, which corresponds to the conventional approach. However, things change when $D_{va} \sim D_{te}$ . This suggests that $D_{va}$ is not sufficient in measuring the model performance for the test set as $z(x, y, I_{\mathrm{va}}, I_{\mathrm{te}}) \neq 1$ . In this case, we can adopt the similar approach outlined in Sec. 3.2 to estimate $z(x, y, I_{\mathrm{va}}, I_{\mathrm{te}})$ , which focuses on $D_{va}$ and $D_{te}$ , rather than $D_{tr}$ and $D_{te}$ .

# 4 DBA Interpretation on Existing Work

In this section, we showcase how some representative existing works can be related to the DBA framework. Such discussion should complement the existing survey papers on subpopulation/distributional shifts and provide insights on the methodological development in the future. We follow the previously introduced categorization.

Algorithm 1 The universal algorithm for optimizing $q(y|x, I_{\mathrm{tr}})$ .

Input The initialized model $q(y|x, I_{\mathrm{tr}})$ ; dataset $\mathcal{D}_{\mathrm{tr}}$ ; The estimation $\hat{p}(s|y, x, I_{\mathrm{tr}})$ .

Output: the optimized $q(y|x, I_{\mathrm{tr}})$ .

1: Obtain $\hat{g}(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ given $\hat{p}(s = y|y,x,I_{\mathrm{tr}})$ (see Eq. (5)).   
2: Perform the following optimization using $D_{tr}$ :

$$
\max _ {q \in \mathcal {M} _ {\mathrm{tr}}} \mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{tr}})} [ \hat {g} (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) \log q (y | x, I _ {\mathrm{tr}}) ]. \tag {8}
$$

The model objective class: Liu et al. (2021) and Nam et al. (2020) can be viewed as proposing different forms of the $p(s|y,x,I_{\mathrm{tr}})$ estimation, where the former utilizes the classification accuracy and the latter considers generalized cross-entropy. Sec. 3.2 and 3.2 provide rationale on the validity of these terms—essentially they characterize the probability $p(s = y|y,x,I_{\mathrm{tr}})$ . Besides the variants of $\hat{p}(s = y|y,x,I_{\mathrm{tr}})$ , they propose different training schemes to correct. Liu et al. (2021) subsamples the training set with their $\hat{p}(s = y|y,x,I_{\mathrm{tr}})$ and Nam et al. (2020) proposes a parallel model to reweight samples according to $\hat{p}(s = y|y,x,I_{\mathrm{tr}})$ from the generalized cross entropy. Nonetheless, none of them is alike DBCM, which is statistically consistent in directly improving the testing performance.

The data sampling class: The methods in the data sampling class share great similarity to ours, as the proposed DBCM is essentially an importance sampling (reweighing) mechanism. ReWeight and ReSample (Japkowicz 2000) can be treated as variants of the sampling technique. Precisely, ReWeight adjusts each sample weight according to the class ratio, in order to recover the class-balanced setup. Similarly, ReSample bootstraps the dataset with class-balanced weights. Essentially, they can be treated as the direct estimation of $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}}):=\frac{p(x,y|I_{\mathrm{te}})}{p(x,y|I_{\mathrm{tr}})}$ , assuming $p(y|I_{\mathrm{tr}})$ is uniform. When considering the presence of attribute s, $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ becomes,

$$
g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) := \frac {p (x , y \mid I _ {\mathrm{te}})}{p (x , y \mid I _ {\mathrm{tr}})} = \frac {\sum_ {s} p (x \mid y , s , I _ {\mathrm{te}}) p (y , s \mid I _ {\mathrm{te}})}{\sum_ {s} p (x \mid y , s , I _ {\mathrm{tr}}) p (y , s \mid I _ {\mathrm{tr}})}. \tag {9}
$$

Their setups, in this case, further assume $p(y, s|I_{\mathrm{te}})$ is uniform and $p(y, s|I_{\mathrm{tr}}) = p(y, s|I_{\mathrm{te}})$ , which is a stronger assumption compared to the proposed.

The auxiliary loss class: Tsirigotis et al. (2024) and Menon et al. (2021) are commonly used logit adjustment methods. With the DBA framework, they can be viewed as a two-step method. First, both methods propose an estimation of $p(y, s = y|, x, I_{\mathrm{tr}})$ . Then the estimates are used as a penalty term to regularize the ERM of the predictive model $q(y|x, I_{\mathrm{tr}})$ . In the first step, Tsirigotis et al. (2024) applies a similar approach to one described in (Liu et al. 2021). Both share conceptual similarity to the DBCM variant in Sec. 3.2. Menon et al. (2021), on the other hand, simply enforces the uniform class balance assumption. Once $\hat{p}(y, s = y|x, I_{\mathrm{tr}})$ is obtained, they optimize w.r.t.

$$
\mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{tr}})} [ \log q (y | x, I _ {\mathrm{tr}}) + \log \hat {p} (y, s = y | x, I _ {\mathrm{tr}}) ]. \tag {10}
$$

To compare the difference between Eq. (10) and the optimal objective (Eq. (4)), we prove the following theorem with two additional assumptions on the label $y$ .

Assumption 6. The label y given $I_{tr}$ follows a uniform distribution.

Assumption 7. The training set contains only the dominant group $m_1$ : $p(m_1|I_{tr}) = 1$ .

Theorem 2. Given Assumption 1, 2, 3 6, and 7 hold, the optimization of Eq. (4) with the following weight function $g(x,y,I_{tr},I_{te})$ directly maximizes the testing performance:

$$
g (x, y, I _ {t r}, I _ {t e}) ^ {- 1} := L \cdot p (y, s = y | x, I _ {t r}). \tag {11}
$$

And the objective Eq. (4) is of form:

$$
\begin{array}{l} \mathbb {E} _ {(x, y) \sim p (x, y | I _ {t r})} \left[ g (x, y, I _ {t r}, I _ {t e}) \left(\log q (y | x, I _ {t r}) \right. \right. \\ \left. + \log p (y, s = y \mid x, I _ {t r})\right) + g (x, y, I _ {t r}, I _ {t e}) \log L \cdot g (x, y, I _ {t r}, I _ {t e}) ]. \tag {12} \\ \end{array}
$$

The proof is deferred to Appendix B. Eq. (10) differs Eq. (12) by 2 aspects. First, Eq. (10) ignores the weight function $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ before the summation. Second, the regularization $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})\log L\cdot g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ in Eq. (12) is missing. Without these terms, Eq. (12) is not guaranteed to optimize for a class-balance dataset, as indicated in (Tsirigotis et al. 2024; Menon et al. 2021). Consequently, these methods may underperform.

The augmentation class: Despite existing works provide augmentation techniques in the form of linear combination (Zhang et al. 2018; Yao et al. 2022; Han et al. 2022), none of the papers provides statistical interpretation on why

such techniques work better than ERM. We see our DBA framework as the first to provide supports for the soundness of the augmentation technique. In short, the augmentation to combine data samples can be viewed as variations of the direct recovery of $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}}):=\frac{p(x,y|I_{\mathrm{te}})}{p(x,y|I_{\mathrm{tr}})}$ under a different set of assumptions. Specifically, we provide the following Theorem 3 to support this statement. The proof can be found in Appendix B. In the following theorem, $m_{0}$ and $m_{1}$ are identical to the terms introduced in Theorem 1. We first describe the assumptions.

Assumption 8. The data generator of $\mathcal{D}_{tr}$ are conditionally identical given different group information $m$ : $p(x|m_0, I_{tr}) = p(x|m_1, I_{tr}) = p(x|I_{tr})$ .

Assumption 9. The predictive model on $\mathcal{D}_{te}$ shares the same law with the model that is conditioned on the group $m_0$ for $\mathcal{D}_{tr}$ : $p(y|x, I_{te}) = p(y|x, m_0, I_{tr})$ .

Note that Assumption 9 is conceptually similar to the setup for Theorem 1.

Theorem 3. Given Assumption 1, 8, and 9 hold, the weight function $g(x, y, I_{tr}, I_{te})$ has the following form:

$$
\begin{array}{l} g (x, y, I _ {t r}, I _ {t e}) ^ {- 1} := \lambda_ {0} (x, I _ {t r}, I _ {t e}) \cdot p (x | I _ {t r}) \\ + \lambda_ {1} (x, y, I _ {t r}, I _ {t e}) \cdot p (x | I _ {t r}), \tag {13} \\ \end{array}
$$

where $\lambda_{0}(x,I_{\mathrm{tr}},I_{\mathrm{te}}):=\frac{p(m_{0}|I_{\mathrm{tr}})}{p(x|I_{\mathrm{te}})}$ and $\lambda_{1}(x,y,I_{\mathrm{tr}},I_{\mathrm{te}}):=\frac{p(m_{1}|I_{\mathrm{tr}})p(y|x,m_{1},I_{\mathrm{tr}})}{p(y|x,I_{\mathrm{te}})p(x|I_{\mathrm{te}})}$ . This means the weight function $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ is a reweighing of the original $p(x|I_{\mathrm{tr}})$ . The commonly used augmentation can be viewed as a sample-level adjustment to the weight function. From Theorem 3 we know that the sum of $\lambda_{0}$ and $\lambda_{1}$ need not be 1, which is different from some existing augmentation approaches (Zhang et al. 2018; Yao et al. 2022) $^{1}$ . The theorem also offers statistical rationale on why the weighted linear combination works (Han et al. 2022). Since both $\lambda_{0}$ and $\lambda_{1}$ depend on the data statistics from the testing set, methods that utilize sample-independent coefficient (Zhang et al. 2018; Yao et al. 2022) should experience degraded performance. We believe this provides insights into the advancement of augmentation-based techniques in the future.

# 5 Experiment

We compare different DBCM variants (see Sec. 3.2) benchmarking models with three benchmarking datasets. We showcase the SOTA performance of our models, to demonstrate the consistency of the theory developed in Sec. 3. In addition, we provide experimental evidence that complements the theory on explaining why existing works would sacrifice average accuracy for higher worst group accuracy.

<table><tr><td rowspan="2"></td><td colspan="2">ColorMNIST(0.5%)</td><td colspan="2">ColorMNIST(2%)</td><td colspan="2">Waterbirds</td><td colspan="2">CivilComments</td></tr><tr><td>average</td><td>worst</td><td>average</td><td>worst</td><td>average</td><td>worst</td><td>average</td><td>worst</td></tr><tr><td>ERM</td><td>81.69 ± 0.10</td><td>1.14 ± 0.40</td><td>95.23 ± 0.07</td><td>56.82 ± 0.23</td><td>88.25 ± 0.16</td><td>67.76 ± 0.30</td><td>87.59 ± 0.38</td><td>48.17 ± 2.61</td></tr><tr><td>Mixup (Zhang et al. 2018)</td><td>81.12 ± 2.20</td><td>0.00 ± 0.00</td><td>96.09 ± 0.20</td><td>80.00 ± 2.22</td><td>88.52 ± 0.22</td><td>59.97 ± 2.01</td><td>87.67 ± 0.12</td><td>53.10 ± 2.11</td></tr><tr><td>LISA (Yao et al. 2022)</td><td>89.45 ± 1.57</td><td>21.50 ± 8.51</td><td>97.32 ± 0.37</td><td>87.27 ± 6.55</td><td>93.63 ± 0.66</td><td>76.95 ± 4.25</td><td>87.22 ± 0.13</td><td>40.62 ± 4.32</td></tr><tr><td>JTT (Liu et al. 2021)</td><td>81.98 ± 1.17</td><td>2.00 ± 0.08</td><td>95.03 ± 0.10</td><td>56.82 ± 2.21</td><td>88.32 ± 0.20</td><td>68.80 ± 2.99</td><td>87.78 ± 0.29</td><td>47.06 ± 2.94</td></tr><tr><td>Focal Loss (Lin et al. 2017)</td><td>67.37 ± 0.44</td><td>0.00 ± 0.00</td><td>94.62 ± 0.25</td><td>43.00 ± 2.33</td><td>87.75 ± 0.36</td><td>54.67 ± 2.67</td><td>87.74 ± 0.16</td><td>43.73 ± 3.66</td></tr><tr><td>GroupDRO (Sagawa et al. 2020)</td><td>82.88 ± 0.09</td><td>9.00 ± 0.08</td><td>95.19 ± 1.01</td><td>40.91 ± 1.20</td><td>92.03 ± 0.16</td><td>83.64 ± 1.88</td><td>86.78 ± 0.18</td><td>56.51 ± 1.93</td></tr><tr><td>MMD (Li et al. 2018)</td><td>11.35 ± 1.30</td><td>0.00 ± 0.00</td><td>11.35 ± 2.26</td><td>0.00 ± 0.00</td><td>88.33 ± 0.51</td><td>53.58 ± 2.38</td><td>82.08 ± 0.63</td><td>0.00 ± 0.00</td></tr><tr><td>ReSample (Japkowicz 2000)</td><td>94.95 ± 0.19</td><td>66.37 ± 2.33</td><td>98.34 ± 0.23</td><td>92.00 ± 1.66</td><td>93.72 ± 0.22</td><td>80.69 ± 1.86</td><td>84.59 ± 1.23</td><td>62.17 ± 1.72</td></tr><tr><td>ReWeight (Japkowicz 2000)</td><td>92.43 ± 0.21</td><td>57.84 ± 1.78</td><td>97.83 ± 0.19</td><td>91.46 ± 1.80</td><td>93.86 ± 0.30</td><td>81.15 ± 2.20</td><td>87.04 ± 0.74</td><td>58.27 ± 2.14</td></tr><tr><td>DBCM(Sec. 3.2, known s)</td><td>96.67 ± 0.27</td><td>84.62 ± 2.02</td><td>98.76 ± 0.20</td><td>92.31 ± 1.73</td><td>94.01 ± 0.19</td><td>83.18 ± 2.00</td><td>87.85 ± 0.15</td><td>43.33 ± 2.15</td></tr></table>

Table 1: Results on the three benchmarking datasets with accessible attribute s. We report both average and worst group accuracy, with mean and standard deviation (“±”) for each of the considered methods after 3 independent runs. The boldfaced values indicate the highest accuracy in comparison.

# 5.1 Experimental Setup

To ensure a fair comparison, we consider models and datasets prepared by Yang et al. (2023). Specifically, we consider two vision datasets: Waterbirds (Sagawa et al. 2020) and ColorMNIST (Nam et al. 2020; Tsirigotis et al. 2024), and one language dataset: CivilComments (Borkan et al. 2019), in order to cover the two popular data types. We modify the ColorMNIST dataset such that it aligns with the setup in Tsirigotis et al. (2024), which is a harder setup. This is because the vanilla version in Yang et al. (2023) consists of only two types of attributes, whereas the version in Nam et al. (2020); Tsirigotis et al. (2024) contains 10 attributes. The modified ColorMNIST contains a “ratio” indicator that specifies the portion of samples that do not correlate with labels and attributes. In our experiment, we consider ratios 2% and 0.5%, as they are the intermediate and

the hardest setups. In practice, we also treat $p(m_{1}|I_{\mathrm{tr}})$ and $p(m_{0}|I_{\mathrm{tr}})$ serve as prior knowledge/hyperparameters of training composition. Specifically for ColorMNIST, where spurious sample ratio is known, we directly assign 0.5% or 2% for $p(m_{0}|I_{\mathrm{tr}})$ (i.e., $1 - p(m_{1}|I_{\mathrm{tr}})$ ). When the composition ratio is unknown, $p(m_{0}|I_{\mathrm{tr}})$ is treated as a hyperparameter and empirically we identify $p(m_{0}|I_{\mathrm{tr}}) = 0.85$ performed well across datasets.

<table><tr><td rowspan="2"></td><td colspan="2">ColorMNIST(0.5%)</td><td colspan="2">ColorMNIST(2%)</td><td colspan="2">Waterbirds</td><td colspan="2">CivilComments</td></tr><tr><td>average</td><td>worst</td><td>average</td><td>worst</td><td>average</td><td>worst</td><td>average</td><td>worst</td></tr><tr><td>ERM</td><td>81.69 ± 0.10</td><td>1.14 ± 0.40</td><td>95.23 ± 0.07</td><td>56.82 ± 0.23</td><td>88.25 ± 0.16</td><td>67.76 ± 0.30</td><td>87.59 ± 0.38</td><td>48.17 ± 2.61</td></tr><tr><td>Mixup (Zhang et al. 2018)</td><td>81.03 ± 2.30</td><td>0.00 ± 0.00</td><td>95.26 ± 0.17</td><td>42.05 ± 3.61</td><td>90.65 ± 0.30</td><td>67.29 ± 1.93</td><td>87.48 ± 0.11</td><td>54.84 ± 2.13</td></tr><tr><td>LISA (Yao et al. 2022)</td><td>68.09 ± 2.06</td><td>0.00 ± 0.00</td><td>94.46 ± 0.53</td><td>15.91 ± 13.11</td><td>89.80 ± 1.11</td><td>66.82 ± 3.87</td><td>87.18 ± 0.28</td><td>49.21 ± 2.11</td></tr><tr><td>JTT (Liu et al. 2021)</td><td>81.80 ± 0.19</td><td>2.00 ± 0.08</td><td>95.42 ± 0.02</td><td>48.86 ± 1.85</td><td>88.83 ± 0.28</td><td>66.36 ± 3.10</td><td>87.78 ± 3.84</td><td>47.06 ± 8.09</td></tr><tr><td>Focal Loss (Lin et al. 2017)</td><td>67.12 ± 0.50</td><td>0.00 ± 0.00</td><td>94.53 ± 0.32</td><td>37.00 ± 4.10</td><td>89.92 ± 0.43</td><td>61.68 ± 3.01</td><td>87.74 ± 0.12</td><td>50.08 ± 4.10</td></tr><tr><td>ReSample (Japkowicz 2000)</td><td>81.55 ± 0.21</td><td>0.00 ± 0.00</td><td>95.70 ± 0.15</td><td>65.00 ± 1.63</td><td>87.99 ± 0.12</td><td>64.17 ± 1.98</td><td>83.24 ± 1.73</td><td>68.91 ± 4.51</td></tr><tr><td>ReWeight (Japkowicz 2000)</td><td>76.77 ± 0.37</td><td>0.00 ± 0.00</td><td>94.93 ± 0.08</td><td>54.55 ± 0.10</td><td>87.81 ± 0.18</td><td>67.60 ± 1.67</td><td>87.02 ±1.12</td><td>58.73 ± 4.60</td></tr><tr><td>DBCM(Sec. 3.2,  $\mathcal{D}_{\text{tr}} \sim \mathcal{D}_{\text{va}}$ )</td><td>94.63 ± 0.35</td><td>57.95 ± 2.30</td><td>97.64 ± 0.10</td><td>81.00 ± 1.40</td><td>88.25 ± 0.05</td><td>70.56 ± 0.12</td><td>87.86 ± 0.30</td><td>43.41 ± 2.20</td></tr><tr><td>DBCM(Sec. 3.2,  $\mathcal{D}_{\text{tr}} \approx \mathcal{D}_{\text{va}}$ )</td><td>86.12 ± 0.29</td><td>3.41 ± 0.70</td><td>96.08 ± 0.20</td><td>61.36 ± 1.90</td><td>91.04 ± 0.07</td><td>62.77 ± 0.10</td><td>87.62 ± 0.20</td><td>53.89 ± 2.16</td></tr></table>

Table 2: Results on the three benchmarking datasets without accessible attribute s. We report both average and worst group accuracy, with mean and standard deviation (“±”) for each of the considered methods after 3 independent runs. The boldfaced values indicate the highest accuracy in comparison.

The evaluation consists of 8 benchmarking models from Yang et al. (2023) that fall into the 4 different classes (see Sec. 1 and 4): Mixup (Zhang et al. 2018); LISA (Yao et al. 2022); JTT (Liu et al. 2021); Focal Loss (Lin et al. 2017); GroupDRO (Sagawa et al. 2020); MMD (Li et al. 2018); ReSample (Japkowicz 2000); ReWeight (Japkowicz 2000). For each model, we consider two setups, where the first allows the presence of attributes and the second does not. We retrain all the considered models from Yang et al. (2023) and pick the best models according to the average validation accuracy, which is different from the worst-group-accuracy criterion in Yang et al. (2023) to match the objective in Eq. 4 (i.e. the framework considers on average accuracy by design). For model optimization, we consider default optimizers and learning rates in Yang et al. (2023). Details are deferred to Appendix E. Code access: https://github.com/skyve2012/DBA.

Attribute s is Known This section presents results with accessible attribute s. In addition to the 8 benchmarking models, we also include results with ERM as the baseline. We consider the DBCM variant in Sec. 3.2. Results are summarized in Table 1. It is clear that when the attribute s presents, the proposed DBCM(Sec. 3.2, known s) achieves the highest average accuracy among all the considered datasets. And all the accuracy of our model exceeds the ERM baseline. This provides the empirical evidence for Theorem 2 and 1. Although there is no theoretical quantification on the worst group accuracy, DBCM achieves two highest and one competing (i.e., Waterbirds) worst group accuracy.

Attribute s is Unknown This section presents results without accessing the attribute s. We omit results for GroupDRO (Sagawa et al. 2020), MMD (Li et al. 2018) as both methods naturally require the knowledge of s (Yang et al. 2023). DBCM(Sec. 3.2, $D_{tr} \sim D_{va}$ ) and DBCM(Sec. 3.2, $D_{tr} \sim D_{va}$ ) are two variants of the proposed method. Results are summarized in Table 2. We observe that the proposed DBCM variants achieve the highest average accuracy among all the compared datasets, and 3 out of 4 highest worst group accuracy, suggesting the validity of the methods when s is unknown. And in the case of the worst group accuracy for ColorMNIST(0.5%), almost all but DBCM cannot correctly classify the worst group samples (i.e. worst group accuracy = 0), suggesting that DBCM method is robust to the change of spurious association between the attributes and the labels.

# 5.2 Observation on the Degraded Average Accuracy

From Table 1 and 2 we empirically identify an interesting phenomenon—compared to all other methods, DBCM is the only model that consistently outperforms the results of ERM. This observation is aligned with Yang et al. (2023); Tsirigotis et al. (2024). Yet the previous work did not provide systematic reasoning on why. We argue that the cause is an incorrect model objective that is different from the data composition in $D_{te}$ . Specifically, the reduced average accuracy is the result of the misspecified $p(x,y|I_{\mathrm{te}})$ in $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ . For the full explanation, please refer to Appendix F. It is noteworthy that we are the first to provide such a statistical interpretation of the degradation phenomenon.

# 6 Conclusion

In summary, we present the DBA framework to identify the true model objective that improves the test performance. The paper proposes different DBCM variants with weaker assumptions compared to the existing works and demonstrates the SOTA performance. Additionally, we reinterpret the existing work with the proposed framework, which explains the issue of the degraded average accuracy. With the analysis, we convey a message that to achieve decent test performance (even without the access to test during training), one must comprehensively investigate the relationship between those datasets and the model objective. For this purpose, we hope the proposed framework could act as a complementary tool to all the existing work, help people analyze such gaps, and facilitate the development of the corresponding model solutions.

# Acknowledgments

This research was partially supported by Alfred P. Sloan foundation and NSF #1934757.

# References

Alshammari, S.; Wang, Y.-X.; Ramanan, D.; and Kong, S. 2022. Long-tailed recognition via weight balancing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6897–6907.   
Arjovsky, M.; Bottou, L.; Gulrajani, I.; and Lopez-Paz, D. 2019. Invariant risk minimization. arXiv preprint arXiv:1907.02893.   
Asgari, S.; Khani, A.; Khani, F.; Gholami, A.; Tran, L.; Mahdavi Amiri, A.; and Hamarneh, G. 2022. Masktune: Mitigating spurious correlations by forcing to explore. Advances in Neural Information Processing Systems, 35: 23284–23296.   
Bickel, S.; Brückner, M.; and Scheffer, T. 2007. Discriminative learning for differing training and test distributions. In Proceedings of the 24th international conference on Machine learning, 81–88.   
Borkan, D.; Dixon, L.; Sorensen, J.; Thain, N.; and Vasserman, L. 2019. Nuanced metrics for measuring unintended bias with real data for text classification. In Companion proceedings of the 2019 world wide web conference, 491–500.   
Byrd, J.; and Lipton, Z. 2019. What is the effect of importance weighting in deep learning? In International conference on machine learning, 872–881. PMLR.   
Fang, T.; Lu, N.; Niu, G.; and Sugiyama, M. 2020. Rethinking importance weighting for deep learning under distribution shift. Advances in neural information processing systems, 33: 11996–12007.   
Han, Y.; and Zou, D. 2024. Improving Group Robustness on Spurious Correlation Requires Preciser Group Inference. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024.   
Han, Z.; Liang, Z.; Yang, F.; Liu, L.; Li, L.; Bian, Y.; Zhao, P.; Wu, B.; Zhang, C.; and Yao, J. 2022. Umix: Improving importance weighting for subpopulation shift via uncertainty-aware mixup. Advances in Neural Information Processing Systems, 35:37704–37718.   
Hong, F.; Yao, J.; Lyu, Y.; Zhou, Z.; Tsang, I.; Zhang, Y.; and Wang, Y. 2023. On Harmonizing Implicit Subpopulations. In The Twelfth International Conference on Learning Representations.   
Huang, J.; Gretton, A.; Borgwardt, K.; Schölkopf, B.; and Smola, A. 2006. Correcting sample selection bias by unlabeled data. Advances in neural information processing systems, 19.   
Izmailov, P.; Kirichenko, P.; Gruver, N.; and Wilson, A. G. 2022. On feature learning in the presence of spurious correlations. Advances in Neural Information Processing Systems, 35: 38516–38532.   
Japkowicz, N. 2000. The class imbalance problem: Significance and strategies. In Proc. of the Int'l Conf. on artificial intelligence, volume 56, 111–117.   
Kanamori, T.; Hido, S.; and Sugiyama, M. 2009. A least-squares approach to direct importance estimation. The Journal of Machine Learning Research, 10: 1391–1445.   
Kang, J.; and Schafer, J. L. 2007. Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data. Statistical Science, 22: 523–539.   
Kimura, M.; and Hino, H. 2024. A Short Survey on Importance Weighting for Machine Learning. Trans. Mach. Learn. Res., 2024.   
LaBonte, T.; Muthukumar, V.; and Kumar, A. 2024. Towards last-layer retraining for group robustness with fewer annotations. Advances in Neural Information Processing Systems, 36.   
Lei, L.; and Candès, E. J. 2021. Conformal inference of counterfactuals and individual treatment effects. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(5): 911–938.   
Li, H.; Pan, S. J.; Wang, S.; and Kot, A. C. 2018. Domain generalization with adversarial feature learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, 5400–5409.   
Lin, T.-Y.; Goyal, P.; Girshick, R.; He, K.; and Dollár, P. 2017. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, 2980–2988.   
Liu, E. Z.; Haghgoo, B.; Chen, A. S.; Raghunathan, A.; Koh, P. W.; Sagawa, S.; Liang, P.; and Finn, C. 2021. Just train twice: Improving group robustness without training group information. In International Conference on Machine Learning, 6781–6792. PMLR.   
Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight Decay Regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019.   
Menon, A. K.; Jayasumana, S.; Rawat, A. S.; Jain, H.; Veit, A.; and Kumar, S. 2021. Long-tail learning via logit adjustment. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021.   
Nam, J.; Cha, H.; Ahn, S.; Lee, J.; and Shin, J. 2020. Learning from failure: De-biasing classifier from biased classifier. Advances in Neural Information Processing Systems, 33: 20673–20684.

Quionero-Candela, J.; Sugiyama, M.; Schwaighofer, A.; and Lawrence, N. D. 2009. Dataset Shift in Machine Learning. The MIT Press. ISBN 0262170051.   
Rudner, T. G.; Zhang, Y. S.; Wilson, A. G.; and Kempe, J. 2024. Mind the GAP: Improving Robustness to Subpopulation Shifts with Group-Aware Priors. In International Conference on Artificial Intelligence and Statistics, 127–135. PMLR.   
Sagawa, S.; Koh, P. W.; Hashimoto, T. B.; and Liang, P. 2020. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In International Conference on Learning Representations (ICLR).   
Shimodaira, H. 2000. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of statistical planning and inference, 90(2): 227–244.   
Suter, R.; Miladinovic, D.; Schölkopf, B.; and Bauer, S. 2019. Robustly disentangled causal mechanisms: Validating deep representations for interventional robustness. In International Conference on Machine Learning, 6056–6065. PMLR.   
Tsirigotis, C.; Monteiro, J.; Rodriguez, P.; Vazquez, D.; and Courville, A. C. 2024. Group Robust Classification Without Any Group Information. Advances in Neural Information Processing Systems, 36.   
Wu, S.; Yuksekgonul, M.; Zhang, L.; and Zou, J. 2023. Discover and cure: Concept-aware mitigation of spurious correlation. In International Conference on Machine Learning, 37765–37786. PMLR.   
Yang, Y.; Kuchibhotla, A. K.; and Tchetgen Tchetgen, E. 2024. Doubly robust calibration of prediction sets under covariate shift. Journal of the Royal Statistical Society Series B: Statistical Methodology.   
Yang, Y.; Zhang, H.; Katabi, D.; and Ghassemi, M. 2023. Change is Hard: A Closer Look at Subpopulation Shift. In International Conference on Machine Learning.   
Yao, H.; Wang, Y.; Li, S.; Zhang, L.; Liang, W.; Zou, J.; and Finn, C. 2022. Improving out-of-distribution robustness via selective augmentation. In International Conference on Machine Learning, 25407–25437. PMLR.   
Yu, H.; Liu, J.; Zhang, X.; Wu, J.; and Cui, P. 2024. A Survey on Evaluation of Out-of-Distribution Generalization. arXiv preprint arXiv:2403.01874.   
Zhang, H.; Cissé, M.; Dauphin, Y. N.; and Lopez-Paz, D. 2018. mixup: Beyond Empirical Risk Minimization. In ICLR.   
Zhang, S.; Luo, Y.; Wang, Q.; Chi, H.; Chen, X.; Han, B.; and Li, J. 2023. Mixture Data for Training Cannot Ensure Out-of-distribution Generalization. arXiv:arXiv:2312.16243.

# Appendix

# A Supplementary Related Work

# A.1 Importance Sampling

DBA interprets the distributional shift as the mismatch of the weight function from the importance sampling perspective. Albeit DBA primarily focuses on the subpopulation setup, we want to point out that the formulation in the method section is general enough to be applied to any of the distributional shift problems.

The discussion on the use of importance sampling for boosting the testing performance can be traced back to 2000 when the authors of the paper discussed the use of importance sampling to improve the accuracy of the test set (Shimodaira 2000). The paper focuses on correcting the shifts using the weights as a function of the data distribution and provides theoretical quantification on such shifts relative to the specification of the weights. However, the lack of real-world experiments and the requirement of second-order derivative calculation w.r.t. the model weights limit its application in complex models. Another early endeavor considers a similar importance weighting approach for correcting the error caused by the dataset shifts (Huang et al. 2006). However, the paper only presents a general setup without any assumptions about the data or the model, leaving a wide space of uncleanness in the subpopulation use cases. In comparison, we provide a systematic formulation of how this tool can be applied to the subpopulation problems and explicitly state the underlying assumptions. This offers a clear theoretical ground for identifying the root components (e.g. distributions) that lead to the subpopulation problem and the corresponding solutions.

Another line of work such as Kanamori, Hido, and Sugiyama (2009) and Fang et al. (2020), provide different ways to estimate the weight function for correcting the shift. Nonetheless, they require partial access to the test set, distinguishing their methods from the proposed one in this paper.

Besides, Kimura and Hino (2024) offers a comprehensive summary on how important sampling can be applied to solve problems like distributional shift, active learning, model calibration, etc. Byrd and Lipton (2019) studies the importance weighting—a different name but inherently it is just importance sampling—and its impact on the test performance. However, one major limitation of the two papers is that both works consider the weight function only on the data variable x, making it insufficient for cases such as subpopulation and covariate shifts, where the attribute and label information are crucial. In comparison, DBA offers a more realistic way with the consideration of the weight function being the ratio of joint distributions of x and y. We further provide experimental evidence on the soundness of the proposed framework in Sec. 5.

# A.2 Subpopulation Survey

Several survey papers provide the summary of subpopulation methods $^{2}$ . To the best of our knowledge, Yang et al. (2023) is the first and possibly the only survey paper that provides a comprehensive experimental study on existing subpopulation methods. The paper decomposes the distribution of $y|x$ to take into account the effect of attributes (i.e. spurious features) via Bayes' theorem, where $y$ is the random variable for the labels and $x$ indicates the random variable for data (see Eq. (1) in (Yang et al. 2023) for details). In particular, the paper categorizes 12 different datasets into 4 classes that have different correlations between the labels and the attributes, and benchmarks 20 popular subpopulation methods with these datasets, to characterize the performance of the methods given different attribute setups. Nonetheless, the statistical quantification of why such differences exist is still missing in the paper.

Yu et al. (2024) and Zhang et al. (2023), on the other hand, provide a good overview of the existing out-of-distribution (OOD) and domain generalization (DG) methods, which is a superset of the subpopulation methods. While the first paper summarizes the existing works on the application level, the second provides statistical quantification of the error inflation when the distribution shifts are present. One major limitation in the analysis of the second paper is its inability to describe how much inflation can be corrected by model design. It only provides insights on data collection.

Our work, in comparison, can be considered as an extension of the previous work, to offer formal statistical analysis on quantifying the source of error from both modeling and data perspectives. In addition, the proposed DBA can formally answer why the existing methods obtain high worst-case accuracy at the cost of lowering average accuracy on the testing set.

# A.3 Subpopulation Method

As presented earlier, different subpopulation methods can be categorized into 4 general classes (auxiliary losses, data augmentations, modeling objectives, and data sampling techniques). We summarize these methods in this section for a detailed overview.

In the auxiliary loss class, Li et al. (2018) uses maximum discrepancy distance and generative adversarial models to remove the effect of background on the prediction performance. Aiming the same goal, Arjovsky et al. (2019) utilizes a gradient regulation on the last layer of the prediction to enforce consistent perdition across different environments/backgrounds. Alshammari et al. (2022), on the other hand, directly adjusts the inference network with a class-balanced distribution. In the data augmentation class, Han et al. (2022), Zhang et al. (2018) and Yao et al. (2022) employ convex combination between two arbitrarily drawn

data samples, to reduce the effect of the spurious backgrounds. They differ only by the means of generating the coefficient for the convex combination. In the data sampling class, both LaBonte, Muthukumar, and Kumar (2024) and Izmailov et al. (2022) look for a subset of class-balanced samples with independent spurious backgrounds. The models are finetuned with this cleaned dataset for optimal performance. The last modeling objective class contains methods that are significantly different from the previous 3 classes. For instance, Japkowicz (2000) adjust the data by class-balanced weights; Liu et al. (2021) and Nam et al. (2020) train the same model twice, correcting the second model with the incorrect prediction from the first; Sagawa et al. (2020) proposes the ERM with the focus on the worst-class; Wu et al. (2023) proposes a model with two parallel processors, where one discovers the subpopulation and the other corrects it; Asgari et al. (2022) first trains a model, then mask out the learned features from the train model and finetune it the second time, forcing the model to learn robust features during prediction; Rudner et al. (2024) utilizes a prior distribution on the subpopulation to improve model performance; Hong et al. (2023) and Han and Zou (2024) implement different loss terms to mitigate the effect of subpopulation on the model. The former uses mutual information whereas the latter considers KL divergence. Lin et al. (2017) proposes a focusing parameter to the cross entropy, to reduce the class imbalance impact.

Compared to the existing work, DBA merits its niche. First, it acts as a tool to help people understand the existing methods described above. Specifically, we explicitly state the assumptions on the data, revealing that different existing methods require different underlying assumptions (see Sec. 4). For example, methods that utilize data augmentation (Zhang et al. 2018; Yao et al. 2022; Han et al. 2022) assume that training data are conditionally identical given different subpopulation conditionals. Tsirigotis et al. (2024) and Menon et al. (2021) requires a uniformity assumption on the label y in the training data. Interestingly, we also discover that all methods can be viewed as having a universal assumption on the identical conditional generative model of data x across all subpopulations. However, none of the previous works states these assumptions explicitly. We believe such discussion is beneficial for people who proceed with this subpopulation direction as for the first time in the community, we explicitly analyze the connections between methods in the same statistical framework.

In practice, DBA outperforms the SOTA benchmarking methods on 3 datasets with 3 proposed methods in the subpopulation setup. This further confirms that DBA is the right framework for this type of problem. More importantly, they are extremely simple to implement due to the nature of importance sampling.

# B Proofs

# B.1 Claim 1

We evaluate the chosen model $q^{*}$ on the testing set $\mathcal{D}_{\mathrm{te}}$ with the following objective:

$$
\begin{array}{l} \mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{te}})} [ \log q ^ {*} (y | x, I _ {\mathrm{tr}}) ] \\ = \mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{te}})} \left[ \frac {p (x , y | I _ {\mathrm{va}})}{p (x , y | I _ {\mathrm{va}})} \log q ^ {*} (y | x, I _ {\mathrm{tr}}) \right] \\ = \mathbb {E} _ {(x, y) \sim p (x, y \mid I _ {\mathrm{va}})} \left[ \frac {p (x , y \mid I _ {\mathrm{te}})}{p (x , y \mid I _ {\mathrm{va}})} \log q ^ {*} (y \mid x, I _ {\mathrm{tr}}) \right]. \tag {14} \\ \end{array}
$$

Under Assumption 1, let $z(x,y,I_{\mathrm{va}},I_{\mathrm{te}}) := \frac{p(x,y|I_{\mathrm{te}})}{p(x,y|I_{\mathrm{va}})}$ , which is well-defined. This completes the proof.

# B.2 Claim 2

We first realize that the notation of finding the model $(q\in \mathcal{M}_{\mathrm{tr}})$ for the best test set performance can be described by the following optimization problem:

$$
\max _ {q \in \mathcal {M} _ {\mathrm{tr}}} \mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{te}})} [ \log q (y | x, I _ {\mathrm{tr}}) ]. \tag {15}
$$

With the same proof logic as in Sec. B.1, we obtain the following on the objective $\mathbb{E}_{(x,y)\sim p(x,y|I_{\mathrm{te}})}[\log q(y|x,I_{\mathrm{tr}})]$ :

$$
\begin{array}{l} \mathbb {E} _ {(x, y) \sim p (x, y \mid I _ {\mathrm{te}})} [ \log q (y \mid x, I _ {\mathrm{tr}}) ] (16) \\ = \mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{te}})} [ \frac {p (x , y | I _ {\mathrm{r}})}{p (x , y | I _ {\mathrm{tr}})} \log q (y | x, I _ {\mathrm{tr}}) ] \\ = \mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{tr}})} \left[ \frac {p (x , y \mid I _ {\mathrm{te}})}{p (x , y \mid I _ {\mathrm{tr}})} \log q (y \mid x, I _ {\mathrm{tr}}) \right]. (17) \\ \end{array}
$$

Under Assumption 1, let $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}}):=\frac{p(x,y|I_{\mathrm{te}})}{p(x,y|I_{\mathrm{tr}})}$ , which is well-defined. Since Eq. (16) and Eq. (17) are equal, maximizing either one w.r.t. $q\in M_{tr}$ is equivalent to the maximization of the other. This completes the proof.

# B.3 Theorem 1

We prove the form of $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ (Eq. (5)) in this section under Assumption 1, 2, and 3. Without loss of generality, we assume the attribute $s$ is categorical and the density for it is $\frac{1}{\tau}$ (Assumption 3).

We first consider the expansion of $p(x,y|I_{\mathrm{te}})$ and $p(x,y|I_{\mathrm{tr}})$ in $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ .

$$
\begin{array}{l} p (x, y | I _ {\mathrm{te}}) = \sum_ {s} p (x | y, s, I _ {\mathrm{te}}) p (y | I _ {\mathrm{te}}) p (s | y, I _ {\mathrm{te}}) \\ \xlongequal {\text {   Assumption   3   }} \sum_ {s} p (x | y, s, I _ {\mathrm{te}}) p (y | I _ {\mathrm{te}}) p (s |, I _ {\mathrm{te}}) \\ = \frac {p (y | I _ {\mathrm{te}})}{L} \sum_ {s} p (x | y, s, I _ {\mathrm{te}}). \tag {18} \\ \end{array}
$$

$$
p (x, y | I _ {\mathrm{tr}}) = \sum_ {s} p (x | y, s, I _ {\mathrm{tr}}) p (y, s | I _ {\mathrm{tr}}). \tag {19}
$$

We further expand $p(y, s|I_{\mathrm{tr}})$ to introduce $p(m = m_0)$ and $p(m = m_1)$ .

$$
\begin{array}{l} p (y, s | I _ {\mathrm{tr}}) \\ = p (m _ {0}) p (y |, m _ {0}, I _ {\mathrm{tr}}) p (s | y, m _ {0}, I _ {\mathrm{tr}}) \\ + p \left(m _ {1} \mid I _ {\mathrm{tr}}\right) p \left(y \mid m _ {1}, I _ {\mathrm{tr}}\right) p \left(s \mid y, m _ {1}, I _ {\mathrm{tr}}\right) \\ \end{array}
$$

$$
\stackrel {\text {Assumption 4,5}} {=} p \left(m _ {1} \mid I _ {\mathrm{tr}}\right) \frac {p \left(y \mid I _ {\mathrm{tr}}\right)}{L} + p \left(m _ {1} \mid I _ {\mathrm{tr}}\right) p \left(y \mid m _ {1}, I _ {\mathrm{tr}}\right) \mathbf {1} _ {\{y = s \}}. \tag {20}
$$

$$
g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) ^ {- 1} = \frac {p (x , y | I _ {\mathrm{tr}})}{p (x , y | I _ {\mathrm{te}})}
$$

$$
\underline {{\underline {{\text { Eq. (19) ,Eq. (20)}}}}} \sum_ {s _ {0}} \frac {p (x | y , s _ {0} , I _ {\mathrm{tr}}) \left[ \frac {p (y | I _ {\mathrm{tr}})}{L} p (m _ {0} | I _ {\mathrm{tr}}) + p (m _ {1} | I _ {\mathrm{tr}}) p (y | m _ {1} , I _ {\mathrm{tr}}) \mathbf {1} _ {\{y = s \}} \right]}{\frac {p (y | I _ {\mathrm{tr}})}{L} \sum_ {s} p (x | s , y , I _ {\mathrm{te}})}
$$

Then $= \frac{p(m_1|I_{\mathrm{tr}})p(y|m_1,I_{\mathrm{tr}})p(x|y,s_0 = y,I_{\mathrm{tr}})}{\frac{p(y|I_{\mathrm{tr}})}{L}\sum_s p(x|s,y,I_{\mathrm{te}})} + p(m_0|I_{\mathrm{tr}})\sum_{s_0}\frac{p(s|y,s_0,I_{\mathrm{tr}})}{\sum_s p(s|y,s,I_{\mathrm{te}})}}$

$$
\underline {\underline {{\text { Assumption   2 }}}} \frac {p (m _ {1} | I _ {\mathrm{tr}}) p (y | m _ {1} , I _ {\mathrm{tr}}) p (x | y , s _ {0} = y , I _ {\mathrm{tr}})}{\frac {p (y | I _ {\mathrm{tr}})}{L} \sum_ {s} p (x | s , y , I _ {\mathrm{te}})} + p (m _ {0} | I _ {\mathrm{tr}}) \cdot 1
$$

$$
\xlongequal {\text {Assumption 2}} p \left(m _ {1} \mid I _ {\mathrm{tr}}\right) \cdot \frac {p \left(y \mid m _ {1} , I _ {\mathrm{tr}}\right)}{\frac {p \left(y \mid I _ {\mathrm{tr}}\right)}{L} + \frac {p \left(y \mid I _ {\mathrm{tr}}\right)}{L} \sum_ {s \neq s _ {0} , s \neq y , y = s _ {0}} \frac {p \left(x \mid y , s , I _ {\mathrm{te}}\right)}{p \left(x \mid y , s _ {0} = y , I _ {\mathrm{tr}}\right)}} + p \left(m _ {0} \mid I _ {\mathrm{tr}}\right). \tag {21}
$$

From Assumption 2, we can also obtain the following equality:

$$
\begin{array}{l} p (x | y, s, I _ {\mathrm{te}}) = p (x | y, s, I _ {\mathrm{tr}}) \\ = \frac {p (y , s | x , I _ {\mathrm{tr}})}{p (y , s | I _ {\mathrm{tr}})} \cdot p (x | I _ {\mathrm{tr}}) \\ = \frac {p (y , s | x , I _ {\mathrm{tr}}) p (x | I _ {\mathrm{tr}})}{p (m _ {0} | I _ {\mathrm{tr}}) \frac {p (y | I _ {\mathrm{tr}})}{L} + p (m _ {1} | I _ {\mathrm{tr}}) \mathbf {1} _ {\{y = s \}} p (y | m _ {1} , I _ {\mathrm{tr}})}. \tag {22} \\ \end{array}
$$

Combining Eq. (21) and Eq. (22), we have:

$$
\begin{array}{l} g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) ^ {- 1} = p (m _ {1} | I _ {\mathrm{tr}}) \cdot \frac {\frac {L}{p (y | I _ {\mathrm{tr}})} p (y | m _ {1} , I _ {\mathrm{tr}})}1 + \sum_ {s \neq s _ {0}, s \neq y, y = s _ {0}} \frac {\frac {p (y , s | x , I _ {\mathrm{tr}}) p (x | I _ {\mathrm{tr}})}{p (m _ {0} | I _ {\mathrm{tr}}) \frac {p (y | I _ {\mathrm{tr}})}{L} + p (m _ {1} | I _ {\mathrm{tr}}) \mathbf {1} _ {\{y = s \}} p (y | m _ {1} , I _ {\mathrm{tr}})}}{\frac {p (y , s = y | x , I _ {\mathrm{tr}}) p (x | I _ {\mathrm{tr}})}{p (m _ {0} | I _ {\mathrm{tr}}) \frac {p (y | I _ {\mathrm{tr}})}{L} + p (m _ {1} | I _ {\mathrm{tr}}) p (y | m _ {1} , I _ {\mathrm{tr}})}} + p (m _ {0} | I _ {\mathrm{tr}}) \\ = p (m _ {1} | I _ {\mathrm{tr}}) \cdot \frac {\frac {L}{p (y | I _ {\mathrm{tr}})} p (y | m _ {1} , I _ {\mathrm{tr}})}{1 + \sum_ {s \neq s _ {0} , s \neq y , y = s _ {0}} \frac {p (y , s , | x , I _ {\mathrm{tr}}) \cdot [ p (m _ {0} | I _ {\mathrm{tr}}) \frac {p (y | I _ {\mathrm{tr}})}{L} + p (m _ {1} | I _ {\mathrm{tr}}) p (y | m _ {1} , I _ {\mathrm{tr}}) ]}{p (y , s = y , | x , I _ {\mathrm{tr}}) \cdot [ p (m _ {0} | I _ {\mathrm{tr}}) \frac {p (y | I _ {\mathrm{tr}})}{L} ]}} + p (m _ {0} | I _ {\mathrm{tr}}) \\ = p (m _ {1} | I _ {\mathrm{tr}}) \cdot \frac {\frac {L}{p (y | I _ {\mathrm{tr}})} p (y | m _ {1} , I _ {\mathrm{tr}})}{1 + \left[ \frac {p (m _ {0} | I _ {\mathrm{tr}}) \frac {p (y | I _ {\mathrm{tr}})}{L} + p (m _ {1} | I _ {\mathrm{tr}}) p (y | m _ {1} , I _ {\mathrm{tr}})}{p (m _ {0} | I _ {\mathrm{tr}}) \frac {p (y | I _ {\mathrm{tr}})}{L}} \right] \cdot \sum_ {s \neq s _ {0}, s \neq y, y = s _ {0}} \frac {p (y , s , | x , I _ {\mathrm{tr}})}{p (y , s = y , | x , I _ {\mathrm{tr}})}} + p (m _ {0} | I _ {\mathrm{tr}}) \\ = p (m _ {1} | I _ {\mathrm{tr}}) \cdot \frac {\frac {L}{p (y | I _ {\mathrm{tr}})} p (y | m _ {1} , I _ {\mathrm{tr}})}{1 + \left[ \frac {p (m _ {0} | I _ {\mathrm{tr}}) \frac {p (y | I _ {\mathrm{tr}})}{L} + p (m _ {1} | I _ {\mathrm{tr}}) p (y | m _ {1} , I _ {\mathrm{tr}})}{p (m _ {0} | I _ {\mathrm{tr}}) \frac {p (y | I _ {\mathrm{tr}})}{L}} \right] \cdot \sum_ {s \neq s _ {0}, s \neq y, y = s _ {0}} \frac {p (s | y , x , I _ {\mathrm{tr}})}{p (s = y , | , y , x , I _ {\mathrm{tr}})}} + p (m _ {0} | I _ {\mathrm{tr}}) \\ = p \left(m _ {1} \mid I _ {\mathrm{tr}}\right) \cdot \left(\frac {\frac {L}{p \left(y \mid I _ {\mathrm{tr}}\right)} \cdot p \left(y \mid m _ {1} , I _ {\mathrm{tr}}\right)}{1 + \left[ \frac {p \left(m _ {0} \mid I _ {\mathrm{tr}}\right) \cdot p \left(y \mid I _ {\mathrm{tr}}\right) / L + p \left(y \mid m _ {1} , I _ {\mathrm{tr}}\right)}{p \left(m _ {0} \mid I _ {\mathrm{tr}}\right) \cdot p \left(y \mid I _ {\mathrm{tr}}\right) / L} \right] \cdot \frac {1 - p (s = y \mid y , x , I _ {\mathrm{tr}})}{p (s = y \mid y , x , I _ {\mathrm{tr}})}}\right) + p \left(m _ {0} \mid I _ {\mathrm{tr}}\right), \tag {23} \\ \end{array}
$$

which is Eq. (5) in Theorem 1. Because of the following equality under Assumption 4

$$
p (y | I _ {\mathrm{tr}}) = p \left(m _ {1} \mid I _ {\mathrm{tr}}\right) p \left(y \mid m _ {1}, I _ {\mathrm{tr}}\right) + p \left(m _ {0} \mid I _ {\mathrm{tr}}\right) p \left(y \mid m _ {0}, I _ {\mathrm{tr}}\right)
$$

$$
\stackrel {\text {Assumption 4}} {=} p \left(m _ {1} \mid I _ {\mathrm{tr}}\right) p \left(y \mid m _ {1}, I _ {\mathrm{tr}}\right) + p \left(m _ {0} \mid I _ {\mathrm{tr}}\right) p \left(y \mid , I _ {\mathrm{tr}}\right), \tag {24}
$$

we obtain $p(y|m_1, I_{\mathrm{tr}}) = \frac{p(y|I_{\mathrm{tr}}) - p(m_0|I_{\mathrm{tr}}) \cdot p(y|I_{\mathrm{tr}})}{p(m_1|I_{\mathrm{tr}})}$ . This completes the proof.

# B.4 Theorem 2

We prove it using some of the previous results from Sec. B.3. Combine Eq. (21), Eq. (22), and Assumption 6 and 7, we have the fol-

$$
g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) ^ {- 1} = p (m _ {1} | I _ {\mathrm{tr}}) \cdot \frac {p (y | m _ {1} , I _ {\mathrm{tr}})}{\frac {p (y | I _ {\mathrm{tr}})}{L} + \frac {p (y | I _ {\mathrm{tr}})}{L} \sum_ {s \neq s _ {0} , s \neq y , y = s _ {0}} \frac {p (x | y , s , I _ {\mathrm{te}})}{p (x | y , s _ {0} = y , I _ {\mathrm{tr}})}} + p (m _ {0} | I _ {\mathrm{tr}})
$$

$$
\xlongequal {\text {Assumption} 7} \frac {p (y | m _ {1} , I _ {\mathrm{tr}})}{\frac {p (y | I _ {\mathrm{tr}})}{L} + \frac {p (y | I _ {\mathrm{tr}})}{L} \sum_ {s \neq s _ {0} , s \neq y , y = s _ {0}} \frac {p (x | y , s , I _ {\mathrm{te}})}{p (x | y , s _ {0} = y , I _ {\mathrm{tr}})}}
$$

lowing:

$$
\text { Eq. } \underline {{\underline {{2 2 , \text { Assumption   6 }}}}} \cdot \frac {p (y | I _ {\mathrm{tr}})}{\frac {p (y | I _ {\mathrm{tr}})}{L} + \frac {p (y | I _ {\mathrm{tr}})}{L} \cdot \frac {1 - p (y , s = y | x , I _ {\mathrm{tr}})}{p (y , s = y | x , I _ {\mathrm{tr}})}}
$$

$$
= \frac {1}{\frac {1}{L \cdot p (y , s = y | x , I _ {\mathrm{tr}})}}
$$

$$
= L \cdot p (y, s = y | x, I _ {\mathrm{tr}}). \tag {25}
$$

Then we have the following from Eq. (4):

$$
\mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{tr}})} \left[ g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) \log q (y | x, I _ {\mathrm{tr}}) \right]
$$

$$
= \mathbb {E} _ {(x, y) \sim p (x, y \mid I _ {\mathrm{tr}})} \left[ g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) \log \left(q (y \mid x, I _ {\mathrm{tr}}) \cdot \frac {g (x , y , I _ {\mathrm{tr}} , I _ {\mathrm{te}})}{g (x , y , I _ {\mathrm{tr}} , I _ {\mathrm{te}})}\right) \right]
$$

$$
= \mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{tr}})} \left[ g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) \left(\log q (y | x, I _ {\mathrm{tr}}) + \log g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) ^ {- 1}\right) + g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) \log g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) \right]
$$

$$
= \mathbb {E} _ {(x, y) \sim p (x, y | I _ {\mathrm{tr}})} \left[ g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) (\log q (y | x, I _ {\mathrm{tr}}) + \log L \cdot p (y, s = y | x, I _ {\mathrm{tr}})) + g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) \log g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) \right]
$$

$$
= \mathbb {E} _ {(x, y) \sim p (x, y \mid I _ {\mathrm{tr}})} [ g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) (\log q (y \mid x, I _ {\mathrm{tr}}) + \log p (y, s = y \mid x, I _ {\mathrm{tr}})) + g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) \log L \cdot g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) ]. \tag {26}
$$

This completes the proof.

# B.5 Theorem 3

We directly expand the reciprocal of the weight function:

$$
\begin{array}{l} g (x, y, I _ {\mathrm{tr}}, I _ {\mathrm{te}}) ^ {- 1} := \frac {p (x , y | I _ {\mathrm{tr}})}{p (x , y | I _ {\mathrm{te}})} \\ = \frac {p \left(m _ {1} \mid I _ {\mathrm{tr}}\right) p \left(y \mid x , m _ {1} , I _ {\mathrm{tr}}\right) p \left(x \mid m _ {1} , I _ {\mathrm{tr}}\right) + p \left(m _ {0} \mid I _ {\mathrm{tr}}\right) p \left(y \mid x , m _ {0} , I _ {\mathrm{tr}}\right) p \left(x \mid m _ {0} , I _ {\mathrm{tr}}\right)}{p \left(y \mid x , I _ {\mathrm{te}}\right) p \left(x \mid I _ {\mathrm{te}}\right)} \\ \underline {\underline {{\text { Assumption   8 }}}} \frac {p (m _ {1} | I _ {\mathrm{tr}}) p (y | x , m _ {1} , I _ {\mathrm{tr}}) p (x | I _ {\mathrm{tr}}) + p (m _ {0} | I _ {\mathrm{tr}}) p (y | x , m _ {0} , I _ {\mathrm{tr}}) p (x | I _ {\mathrm{tr}})}{p (y | x , I _ {\mathrm{te}}) p (x | I _ {\mathrm{te}})} \\ \underline {{{\underline {{{\text {Assumption 9}}}}}}} \frac {p (m _ {1} | I _ {\mathrm{tr}}) p (y | x , m _ {1} , I _ {\mathrm{tr}})}{p (y | x , I _ {\mathrm{te}}) p (x | I _ {\mathrm{te}})} \cdot p (x | I _ {\mathrm{tr}}) + \frac {p (m _ {0} | I _ {\mathrm{tr}})}{p (x | I _ {\mathrm{te}})} \cdot p (x | I _ {\mathrm{tr}}). \tag {27} \\ \end{array}
$$

Let $\lambda_0(x, I_{\mathrm{tr}}, I_{\mathrm{te}}) := \frac{p(m_0|I_{\mathrm{tr}})}{p(x|I_{\mathrm{te}})}$ and $\lambda_1(x, y, I_{\mathrm{tr}}, I_{\mathrm{te}}) := \frac{p(m_1|I_{\mathrm{tr}}) p(y|x,m_1,I_{\mathrm{tr}})}{p(y|x,I_{\mathrm{te}}) p(x|I_{\mathrm{te}})}$ . This completes the proof.

# C Restriction and Relaxation of Core Assumptions for Theorem 1

This section discusses the limitations of DBCM, aiming to facilitate the research for the future direction. Specifically, we discuss about Assumption 1 to Assumption 5.

Assumption 1: This assumption requires inclusion relationships on the supports of the training, testing, and validation sets, ensuring a well-defined weight function for importance sampling—a fundamental assumption required for this approach.

Assumption 2: this assumption states that the data generation process for x is identical in both the training and testing datasets, which implies that the underlying generative model for x given y remains consistent. This assumption is prevalent in the literature related to disentangled causal processes and conformal inference (Kang and Schafer 2007; Tsirigotis et al. 2024; Yang, Kuchibhotla, and Tchetgen Tchetgen 2024). The assumption can be interpreted as a form that characterizes the covariate shift. Relaxing this assumption would require a precise understanding of the relationship between $p(x|y, s, I_{\mathrm{tr}})$ and $p(x|y, s, I_{\mathrm{te}})$ , which is currently beyond the scope of our work, but could be an interesting avenue for future research.

Assumption 3: This assumption asserts that the label distribution x does not need to be uniformly distributed, which makes our approach applicable to scenarios with class imbalance. Specifically, this means that the test data can have different class proportions compared to the training data, a common situation in real-world settings. If we treat s as a treatment variable in causal language, this assumption is akin to requiring strong ignorability between x and s, with an additional constraint that s is independent of x. This assumption simplifies the weight function in Theorem 1 by making it depend only on $p(s = y|y, x, I_{\mathrm{tr}})$ . If this assumption were relaxed, we would need additional estimation steps, such as estimating $p(s = y|y, I_{\mathrm{te}})$ .

Assumption 4: This assumption requires that there exists a subset of labels y that follows the same distribution across both training and testing datasets. Given that we assume inclusive supports (Assumption 1) for training and testing datasets, this assumption is not restrictive, as we can subsample the training set to approximate the distribution of labels in the test set.

Assumption 5: This assumption characterizes the nature of the subpopulation shift itself by assuming that the shift is caused by a spurious variable influencing Y. Such a scenario is frequently observed in datasets like Waterbirds, CelebA, and CivilComments (Yang et al. 2023), where spurious correlations exist between features and labels, leading to subpopulation shifts. This assumption essentially defines the kind of subpopulation shift we are considering, specifically focusing on shifts due to spurious correlations.

# D Application of DBCM in Difference Scenarios

In this section, we outline the three scenarios (see Sec. 3) for estimating $p(s = y|y, x, I_{\mathrm{tr}})$ , each applicable based on the availability of certain types of information. Below, we provide practical guidance on selecting the appropriate scenario for a given application:

Attribute s is Known: this scenario applies when s, the spurious variable, is known. Since we have access to x and y in the training set, we can directly model $p(s = y|y, x, I_{\mathrm{tr}})$ . This is equivalent to saying that we have perfect knowledge about the source of the subpopulation shift between training and test sets. In such cases, the estimation is straightforward, and the subpopulation shift can be offset using the weight function derived in Theorem 1.

Attribute s is Unknown, $D_{tr} \sim D_{va}$ : if s is unknown, but a validation set is available that follows the same distribution as the training set, we use Eq. (6) for estimating $p(s = y|y, x, I_{\mathrm{tr}})$ . The availability of a validation set allows us to learn about the underlying data distribution and estimate the relevant quantities, even without explicit knowledge of s. Practically speaking, when one suspects a subpopulation shift between train and test sets, a validation set can be used to model the shift and adjust accordingly.

Attribute s is Unknown, $D_{tr} \sim D_{va}$ : when neither s is known nor is a validation set available, and we assume that the train and validation distributions differ, we suggest estimating $p(s = y|y, x, I_{\mathrm{tr}})$ like the logit adjustment technique (Menon et al. 2021) but with a different objective function (Eq. (7)). Although it is used in a different context, the methodology can be adapted for the estimation here, especially when faced with distributional shifts.

# E Training Configuration

For training, we consider stochastic gradient descent (learning rate = 0.001) for the vision datasets and AdamW (Loshchilov and Hutter 2019) (learning rate = 0.0001) for the language dataset. All models are trained with a single 16 GB NVIDIA V100 GPU for 3 independent runs $^{3}$ . We follow the convention and report the mean and the standard deviation on both the average accuracy and the worst group accuracy.

# F Discussion on the Degraded Average Accuracy

From Table 1 and 2 we empirically identify an interesting phenomenon—compared to all other methods, DBCM is the only model that consistently outperforms the results of ERM. This observation is aligned with Yang et al. (2023); Tsirigotis et al. (2024). However, the previous work did not provide systematic reasoning on why.

With the proposed DBA framework, we argue that this phenomenon is introduced by an incorrect model objective that is different from the data composition in $D_{te}$ . Essentially, the DBA framework conveys a single message—we need to optimize for what we want $^{4}$ .

We first make the connection between the ERM and the proposed optimization (Eq. (8)). During optimization, lowering the ERM aims to improve the models' accuracy in the average sense. This is because ERM is an empirical mean estimator of the loss objective at the population level. We can view the proposed as a generalized ERM for an arbitrary weight function $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ . Then in the case of ERM, the assumption $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})=1$ is implicitly made. Different specification on $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ characterizes different relationships between $D_{tr}$ and $D_{te}$ , allowing the correction of the subpopulation shifts therein. Nonetheless, in either case, we do not alter our target on the average accuracy as this is constrained by the empirical risks (regardless of the form of $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ ), meaning the worst group accuracy is never a causal result of the optimization. Instead, it is a result of optimization with misspecification on $D_{te}$ . For example, ReSample (Japkowicz 2000) aims to maximize the average accuracy for a group-balanced $D_{te}$ . In this case, ReSample implicitly assumes that the $D_{te}$ is group-balanced. However, this need not be the case. In the CivilComments case in $D_{te}$ , the groups are still imbalanced, suggesting a misspecification of $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ that ReSample focuses on than what is the groundtruth $g(x,y,I_{\mathrm{tr}},I_{\mathrm{te}})$ in $D_{te}$ . Subsequently, we observe the lowered average accuracy in Table 1.

Broadly speaking, from the analysis in Sec. 4, we know that most benchmarking methods essentially propose different forms of class-balance recovery $^{5}$ . However, since benchmarking datasets $D_{te}$ do not necessarily share the identical class-balance setup (see Table 2 in (LaBonte, Muthukumar, and Kumar 2024)), the objective with which the existing methods optimize may introduce misspecification between the true testing data and the data to which model is optimized. In comparison, since DBCM imposes weaker assumptions, it is reasonable to observe improved performance. It is noteworthy that we are the first to provide such a statistical interpretation of the degradation phenomenon.