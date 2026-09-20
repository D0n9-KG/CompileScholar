# Estimating and Controlling for Equalized Odds via Sensitive Attribute Predictors

Beepul Bharti \*†‡

Paul Yi §¶

Jeremias Sulam $*\dagger\ddagger$

# Abstract

As the use of machine learning models in real world high-stakes decision settings continues to grow, it is highly important that we are able to audit and control for any potential fairness violations these models may exhibit towards certain groups. To do so, one naturally requires access to sensitive attributes, such as demographics, gender, or other potentially sensitive features that determine group membership. Unfortunately, in many settings, this information is often unavailable. In this work we study the well known equalized odds (EOD) definition of fairness. In a setting without sensitive attributes, we first provide tight and computable upper bounds for the EOD violation of a predictor. These bounds precisely reflect the worst possible EOD violation. Second, we demonstrate how one can provably control the worst-case EOD by a new post-processing correction method. Our results characterize when directly controlling for EOD with respect to the predicted sensitive attributes is – and when is not – optimal when it comes to controlling worst-case EOD. Our results hold under assumptions that are milder than previous works, and we illustrate these results with experiments on synthetic and real datasets.

# 1 Introduction

Machine learning (ML) algorithms are increasingly used in high stakes prediction applications that can significantly impact society. For example, ML models have been used to detect breast cancer in mammograms $[33]$ , inform parole and sentencing decisions $[14]$ , and aid in loan approval decisions $[37]$ . While these algorithms often demonstrate excellent overall performance, they can be dangerously unfair and negatively impact under-represented groups $[34]$ . Some of these unforeseen negative consequences can even be fatal when considering under diagnosis biases of deep learning models in chest x-ray diagnosis of diseases, like cancer $[32]$ . Recommendations to ensure that ML systems do not exacerbate societal biases have been raised by several groups, including the White House in a 2016 report on big data, algorithms, and civil rights $[27]$ . It is thus critical to understand how to rigorously evaluate the fairness of ML algorithms and control for unfairness during model development.

These needs have prompted considerable research in the area of Fair ML. While definitions of algorithmic fairness abound $[7]$ , common notions of group fairness consider different error rates of a predictor across different groups: males and females, white and non-white, etc. For example, the equal opportunity criterion requires the true positive rate (TPR) be equal across both groups, while equalized odds requires both TPR and false positive rate (FPR) to be the same across groups $[20]$ . Obtaining predictors that are fair therefore requires enforcing these constraints on error rates across groups during model development, which can be posed as a constrained (or regularized) optimization problem $[31, 38, 2, 13, 15]$ . Alternatively, one can devise post-processing strategies to modify a certain predictor to correct for differences in TPR and FPR $[20, 17, 12, 1]$ , or even include data

pre-processing steps that ensure that unfair models could not be obtained from such data to begin with $[35, 8]$ .

Naturally, all these techniques for estimating or enforcing fairness require access to a dataset with features, X, responses, Y, and sensitive attributes, A. However, in many settings this is difficult or impossible, as datasets often do not include samples that have all these variables. This could be because the sensitive attribute data was withheld due to privacy concerns, which is very common with medical data due to HIPAA federal law requirements, or simply because it was deemed unnecessary to record $[36, 39]$ . A real-world example of this is in the recent Kaggle-hosted RSNA Chest X-Ray Pneumonia Detection Challenge $[29]$ . Even though this dataset of chest x-rays was painstakingly annotated for pneumonia disease by dozens of radiologists, it did not include sensitive attributes (e.g., age, sex, and race), precluding the evaluation of fairness of models developed as part of the challenge. In settings like this, where there is limited or no information on the sensitive attribute of interest, it is still important to be able to accurately estimate the violation of fairness constraints by a ML classifier and to be able to alleviate such biases before deploying it in a sensitive application. This leads to the natural question, how can one assess and control the fairness of a classifier without having access to sensitive attribute data? In other words, how can we measure and potentially control the fairness violations of a classifier for Y with respect to a sensitive attribute A, when we have no data that jointly observes A and Y?

# 1.1 Related Work

Estimating unfairness and, more importantly, developing fair predictors where there is no – or only partial – information about the sensitive attribute has only recently received increasing attention. The recent work by Zhao et al. $[40]$ explores the perspective of employing features that are correlated with the sensitive attribute, and shows that enforcing low correlation with such “fairness related features” can lead to models with lower bias. Although these techniques are promising, they require domain expertise to determine which features are highly correlated with the sensitive attribute. An appealing alternative that has been studied is the use proxy sensitive attributes that are created by a second predictor trained on a different data set that contains only sensitive attribute information $[19, 24, 10, 5]$ . This strategy has been widely adopted in many domains such as healthcare $[16]$ , finance $[6]$ , and politics $[22]$ . While using sensitive attribute predictors has proven to be an effective and practical solution, it must be done with care, as this opens new problems for estimating and controlling for fairness. The work by Prost et al. $[30]$ considers the estimation of fairness in a setting where one develops a predictor for an unobserved covariate, but it does not contemplate predicting the sensitive attribute itself. On the other hand Chen et al. $[10]$ study the sources of error in the estimation of fairness via predicted proxies computed using threshold functions, which are prone to over-estimation.

The closest to our work are the recent results by Kallus et al. [24], and Awasthi et al. [5, 4]. Kallus et al. [24] study the identifiability of fairness violations under general assumptions on the distribution and classifiers. They show that, in the absence of the sensitive attribute, the fairness violation of predictions, $\widehat{Y}$ , is unidentifiable unless strict assumptions $^{1}$ are made or if there is some common observed data over $A$ and $Y$ . Nonetheless, they show not all hope is lost and provide closed form upper and lower bounds of the fairness violations of $\widehat{Y}$ under the assumption that one has two datasets: one that is drawn from the marginal over $(X,A)$ and the other drawn from the marginal over $(X,Y,\widehat{Y})$ . Their analysis, however, does not consider predictors $\widehat{Y}=f(X)$ or $\widehat{A}=h(X)$ , and instead their bounds depend explicitly on the conditional probabilities, $\mathbb{P}(A\mid X)$ and $\mathbb{P}(\widehat{Y},Y\mid X)$ , along with the distribution over the features, $\mathbb{P}(X)$ . Unfortunately, with this forumalation, it is unclear how the bounds would change if the estimation of the conditional probabilities was inaccurate (as is the case when developing predictors $\widehat{A}$ in the real world). Furthermore, in settings where $X$ is

high dimensional (as for image data), calculating such bounds would become intractable. Since the fairness violation cannot be directly modified, they then study when these bounds can be reduced and improved. However, they do so in settings that impose smoothness assumptions over $(X, A, Y, \widehat{Y})$ , which clearly are not-verifiable without data over the complete joint distribution. As a result, their results do not provide any actionable method that could improve the bounds.

Awasthi et al. [5], on the other hand, make progress in understanding properties of the sensitive attribute predictor, $\widehat{A}$ , that are desirable for easy and accurate fairness violation estimation and control of a classifier $\widehat{Y}$ . Assuming that $\widehat{Y} \perp \widehat{A} \mid (A, Y)$ , they demonstrate that the true fairness violation is in fact proportional to the estimated fairness violation (the fairness violation using $\widehat{A}$ in lieu of A). This relationships yields the counter-intuitive result that given a fixed error budget for the sensitive attribute predictor, the optimal attribute predictor for the estimation of the true fairness violation is one with the most unequal distribution of errors across the subgroups of the sensitive attribute. However, one is still unable to actually calculate the true fairness violation - as it is unidentifiable. Nonetheless, the relationship does demonstrates that if one can maintain the assumption above while controlling for fairness with respect to $\widehat{A}$ , then doing so will provably reduce the true fairness violation with respect to $A$ . Unfortunately, while these rather strict assumptions can be met in some limited scenarios (as in [4]), these are not applicable in general - and cannot even be tested without access to data over $(A, Y)$ .

Overall, while progress has been made in understanding how to estimate and control fairness violations in the presence of incomplete sensitive attribute information, these previous results highlight that this can only be done in simple settings (e.g., having access to some data from the entire distribution, or by making strong assumptions of conditional independence). Moreover, it remains unclear whether tight bounds can be obtained that explicitly depend on the properties of the predictor $\widehat{Y}$ , allowing for actionable bounds that can provably mitigate for its fairness violation without having an observable sensitive attribute or making stringent assumptions.

# 1.2 Contributions

The contributions of our work can be summarized as follows:

- We study the well known equalized odds (EOD) definition of fairness in a setting where the sensitive attributes, $A$ , are not observed with the features $X$ and labels $Y$ . We provide tight and computable bounds on the EOD violation of a classifier, $\widehat{Y} = f(X)$ . These bounds represent the worst-case EOD violation of $f$ and employ a predictor for the sensitive attributes, $\widehat{A} = h(X)$ , obtained from a sample over the distribution $(X, A)$   
- We provide a precise characterization of the classifiers that achieve minimal worst-case EOD violations with respect to unobserved sensitive attributes. Through this characterization, we demonstrate when simply correcting for fairness with respect to the proxy sensitive attributes will yield minimal worst-case EOD violations, and when instead it proves to be sub-optimal.   
- We provide a simple and practical post-processing technique that provably yields classifiers that maximize prediction power while achieving minimal worst-case EOD violations with respect to unobserved sensitive attributes.   
- We illustrate our results on a series of simulated and real data of increasing complexity.

# 2 Problem Setting

We work within a binary classification setting and consider a distribution $\mathcal{Q}$ over $(\mathcal{X} \times \mathcal{A} \times \mathcal{Y})$ where $\mathcal{X} \subseteq \mathbb{R}^n$ is the feature space, $\mathcal{Y} = \{0,1\}$ the label space, and $\mathcal{A} = \{0,1\}$ the sensitive attribute space. Furthermore, and adopting the setting of Awasthi et al. [5], we consider 2 datasets, $\mathcal{D}_1$ and

$D_{2}$ . The former is drawn from the marginal over $(\mathcal{X},\mathcal{A})$ of Q while $D_{2}$ is drawn from the marginal $(\mathcal{X},\mathcal{Y})$ of Q. In this way, $D_{1}$ and $D_{2}$ contain the same set of features, $D_{1}$ contains sensitive attribute information and $D_{2}$ contains label information. The drawn samples in $D_{1}$ and $D_{2}$ are i.i.d over their respective marginals, and thus different from one another.

Similar to previous work [10, 30, 5], we place ourselves in a demographically scarce regime where there is designer who has access to $\mathcal{D}_1$ to train a sensitive attribute predictor $h: \mathcal{X} \to \mathcal{A}$ and a developer, who has access to $\mathcal{D}_2$ , the sensitive attribute classifier $h$ , and all computable probabilites $\mathbb{P}(h(X), A)$ that the designer of $h$ can extract from $\mathcal{D}_1$ . In this setting, the goal of the developer is to learn a classifier $f: \mathcal{X} \to \mathcal{Y}$ (from $\mathcal{D}_2$ ) that is fair with respect to $A$ utilizing $\widehat{A}$ . The central idea is to augment every sample in $\mathcal{D}_2$ , $(x_i, y_i)$ , by $(x_i, y_i, \hat{a}_i)$ , where $\hat{a}_i = h(x_i)$ . Intuitively, if the error of the sensitive attribute predictor, denoted herein by $U = \mathbb{P}(h(X) \neq A)$ , is low, we could hope that fairness with respect to the real (albeit unobserved) sensitive attribute can be faithfully estimated. Our goal is to thus estimate the error incurred in measuring and enforcing fairness constraints by means of $\hat{A} = h(X)$ , and potentially alleviate or control for it.

Throughout the remainder of this work we focus on equalized odds (EOD) as our fairness metric [20] of interest as it is one of the most popular notions of fairness. Thus moving forward, the term fairness refers specifically to EOD. We denote $\hat{Y} = f(X)$ for simplicity, and for $i,j\in \{0,1\}$ define the group conditional probabilities

$$
\alpha_ {i, k} = \mathbb {P} (\hat {Y} = 1 \mid A = i, Y = j). \tag {1}
$$

These probabilities quantify the TPR (when j = 1) and FPR (when j = 0), for either protected group (i = 0 or i = 1). We assume that the base rates, $r_{i,j} = \mathbb{P}(A = i, Y = j) > 0$ so that these quantities are not undefined. With these conditionals probabilities, we define the true fairness violation of f, $\Delta(f)$ , as the tuple $\Delta(f) = (\Delta_{\mathrm{TPR}}(f), \Delta_{\mathrm{FPR}}(f))$ , where

$$
\Delta_ {\mathrm{TPR}} (f) = \alpha_ {1, 1} - \alpha_ {0, 1} \quad \text { and } \quad \Delta_ {\mathrm{FPR}} (f) = \alpha_ {1, 0} - \alpha_ {0, 0}. \tag {2}
$$

The quantities $\Delta_{\mathrm{TPR}}(f)$ and $\Delta_{\mathrm{FPR}}(f)$ , respectively quantify the absolute difference in TPR and FPRs among the two protected groups. Throughout this work, we will use $\Delta(f)$ to refer to both of these quantities simultaneously.

We also need to characterize the performance of the sensitive attribute classifier, h. The misclassification error of h can be decomposed as, $U = U_{0} + U_{1}$ where $U_{i} = \mathbb{P}(\widehat{A} = i, A \neq i)$ , for $i \in \{0, 1\}$ . We define the difference in errors to be

$$
\Delta U = U _ {0} - U _ {1}. \tag {3}
$$

In a demographically scarce regime, the rates $r_{i,j}$ , and more importantly the quantities of interest, $\Delta_{\mathrm{TPR}}(f)$ and $\Delta_{\mathrm{FPR}}(f)$ , cannot be computed because samples from A and Y are not jointly observed. However, using the sensitive attribute classifier h, we can predict $\widehat{A}$ on $D_{2}$ and compute

$$
\hat {r} _ {i, j} = \mathbb {P} (\widehat {A} = i, Y = j) \quad \mathrm{and} \quad \widehat {\alpha} _ {i, j} = \mathbb {P} (\widehat {Y} = 1 | \widehat {A} = i, Y = j),
$$

which serve as the estimates for the true base rates and group TPRs and FPRs.

# 3 Theoretical Results

With the setting defined, we will now present our results. The first result provides computable bounds on the true fairness violation of $f$ with respect to the true, but unobserved, sensitive attribute $A$ . The bounds precisely characterize the worst-case fairness violation of $f$ . Importantly, as we will explain later, this first result will provide insight into what properties $f$ must satisfy so that it's worst-case fairness violation is minimal. In turn, these results will lead to a simple post-processing

method that can correct a pretrained classifier f into another one, $\bar{f}$ , that has minimal worst-case fairness violations. Before presenting our findings, we first describe the key underlying assumption we make about the pair of classifiers, h and f, so that the subsequent results are true.

Assumption 1. For $i, j \in \{0,1\}$ , the classifiers $\widehat{Y} = f(X)$ and $\widehat{A} = h(X)$ satisfy

$$
\frac {U _ {i}}{\hat {r} _ {i , j}} \leq \widehat {\alpha} _ {i, j} \leq 1 - \frac {U _ {i}}{\hat {r} _ {i , j}}. \tag {4}
$$

To parse this assumption, it is easy to show that this is met when a) h is accurate enough for the setting, namely that $\mathbb{P}(\widehat{A}=i,A\neq i)\leq\frac{1}{2}\mathbb{P}(\widehat{A}=i,Y=j)$ , and b) the predictive power of h is better than the ability of f to predict the labels, Y – or more precisely, $\mathbb{P}(\widehat{A}=i,A\neq i)\leq\mathbb{P}(\widehat{Y}=j,\widehat{A}=i,Y\neq j)$ . We refer the reader to Appendix A.1 for a thorough explanation for why this is true. While this assumption may seem limiting, this is milder than those in existing results: First, accurate predictors h can be developed [6, 16, 22, 18], thus satisfying the assumption (our numerical results will highlight this fact as well). Second, other works [5, 9], require assumptions on $\widehat{A}$ and $\widehat{Y}$ that are unverifiable in a demographically scarce regime. Our assumption, on the other hand, can always be easily verified because all the quantities are computable.

# 3.1 Bounding Fairness Violations with Proxy Sensitive Attributes

With Assumption 1 in place, we present our main result.

Theorem 1 (Bounds on $\Delta(f)$ ). Under Assumption 1, we have that

$$
\left| \Delta_ {T P R} (f) \right| \leq B _ {T P R} (f) \stackrel {\Delta} {=} \max \left\{\left| B _ {1} + C _ {0, 1} \right|, \left| B _ {1} - C _ {1, 1} \right| \right\} \tag {5}
$$

$$
| \Delta_ {F P R} (f) | \leq B _ {F P R} (f) \stackrel {\Delta} {=} \max \{| B _ {0} + C _ {0, 0} |, | B _ {0} - C _ {1, 0} | \}
$$

where

$$
B _ {j} = \frac {\hat {r} _ {1 , j}}{\hat {r} _ {1 , j} + \Delta U} \widehat {\alpha} _ {1, j} - \frac {\hat {r} _ {0 , j}}{\hat {r} _ {0 , j} - \Delta U} \widehat {\alpha} _ {0, j} \quad a n d \quad C _ {i, j} = U _ {i} \left(\frac {1}{\hat {r} _ {1 , j} + \Delta U} + \frac {1}{\hat {r} _ {0 , j} - \Delta U}\right).
$$

Furthermore, the upper bounds for $|\Delta_{TPR}(f)|$ and $|\Delta_{FPR}(f)|$ are tight.

The proof, along with all others in this work, are included in Appendix A.1. We now make a few remarks on this result. First, the bound is tight in that there exists settings (albeit unlikely) with particular marginal distributions such that the bounds hold with equality. Second, even though $|\Delta_{\mathrm{TPR}}(f)|$ and $|\Delta_{\mathrm{FPR}}(f)|$ cannot be calculated, a developer can still calculate the worst-case fairness violations, $B_{\mathrm{TPR}}(f)$ and $B_{\mathrm{FPR}}(f)$ , because these depend on quantities that are all computable in practice. Thus, if $B_{\mathrm{TPR}}(f)$ and $B_{\mathrm{FPR}}(f)$ are low, then the developer can proceed having a guarantee on the maximal fairness violation of f, even while not observing $\Delta(f)$ . On the other hand, if these bounds are large, this implies a potentially large fairness violation over the protected group A by f. Third, the obtained bounds are linear in the parameters $\widehat{\alpha}_{i,j}$ , which the developer can adjust as they are properties of f: this will become useful shortly.

# 3.2 Optimal Worst-Case Fairness Violations

Given the result above, what properties should classifiers f satisfy such that $B_{\mathrm{TPR}}(f)$ and $B_{\mathrm{FPR}}(f)$ are minimal? Moreover, are the classifiers f that are fair with respect to $\widehat{A}$ , the ones that have smallest $B_{\mathrm{TPR}}(f)$ and $B_{\mathrm{FPR}}(f)$ ? We now answer these questions in the following theorem.

Theorem 2 (Minimizers of $B_{\mathrm{TPR}}(f)$ and $B_{\mathrm{FPR}}(f)$ ). Let $\widehat{A} = h(X)$ be a sensitive attribute classifier with errors $U_0$ and $U_1$ that produces rates $\hat{r}_{i,j} = \mathbb{P}(\widehat{A} = i, Y = j)$ . Let $\mathcal{F}$ be the set of classifiers for $Y$ such that $\forall f \in \mathcal{F}$ , $f$ and $h$ satisfy Assumption 1. Then, $\exists \bar{f} \in \mathcal{F}$ with group conditional probabilities, $\underline{\alpha}_{i,j} = \mathbb{P}(\widehat{Y} = 1 \mid \widehat{A} = i, Y = j)$ that satisfy the following condition,

$$
\frac {\hat {r} _ {0 , j}}{\hat {r} _ {0 , j} - \Delta U} \widehat {\underline {{\alpha}}} _ {0, j} - \frac {\hat {r} _ {1 , j}}{\hat {r} _ {1 , j} + \Delta U} \widehat {\underline {{\alpha}}} _ {1, j} = \frac {\Delta U}{2} \left(\frac {1}{\hat {r} _ {1 , j} + \Delta U} + \frac {1}{\hat {r} _ {0 , j} - \Delta U}\right). \tag {6}
$$

Furthermore, any $\bar{f}$ that satisfies the condition above has minimal bounds, i.e. $\forall f\in \mathcal{F}$ ,

$$
| \Delta_ {T P R} (\bar {f}) | \leq B _ {T P R} (\bar {f}) \leq B _ {T P R} (f) \quad a n d \quad | \Delta_ {F P R} (\bar {f}) | \leq B _ {F P R} (\bar {f}) \leq B _ {F P R} (f). \tag {7}
$$

This result provides a precise characterization of the conditions that lead to minimal worst-case fairness violations. Observe that if $\Delta U \neq 0$ , the classifier with minimal $B_{\mathrm{TPR}}(f)$ and $B_{\mathrm{FPR}}(f)$ involves $\widehat{\alpha}_{i,j}$ such that $\widehat{\alpha}_{1,j} \neq \widehat{\alpha}_{0,j}$ , i.e. it is not fair with respect to $\widehat{A}$ . On the other hand, if the errors of h are balanced ( $\Delta U = 0$ ), then minimal bounds are achieved by being fair with respect to $\widehat{A}$ .

# 3.3 Controlling Fairness Violations with Proxy Sensitive Attributes

Now that we understand what conditions $f$ must satisfy so that it's worst case fairness violations are minimal, what remains is a method to obtain such a classifier. We take inspiration from the post-processing method proposed by Hardt et al. [20], which derives a classifier $\overline{Y} = \bar{f}(X)$ from $\widehat{Y} = f(X)$ that satisfies equalized odds with respect to a sensitive attribute $A$ while minimizing an expected misclassification loss - only applicable if one has access to $A$ , which is not true in our setting. Nonetheless, since our method will generalize this idea, we first briefly comment on this approach. The method they propose works as follows: given a sample with initial prediction $\widehat{Y} = \hat{y}$ and sensitive attribute $A = a$ , the derived predictor $\bar{f}$ , with group conditional probabilities $\underline{\alpha}_{i,j} = \mathbb{P}(\overline{Y} = 1 \mid A = i, Y = j)$ , predicts $\overline{Y} = 1$ with probability $p_{a,\hat{y}} = \mathbb{P}(\overline{Y} = 1 \mid A = a, \widehat{Y} = \hat{y})$ . The four probabilities $p_{0,0}, p_{0,1}, p_{1,0}, p_{1,1}$ can be then calculated so that $\overline{Y}$ satisfies equalized odds and the expected loss between $\overline{Y}$ and labels $Y$ , i.e. $\mathbb{E}[L(\overline{Y}, Y)]$ , is minimized. The fairness constraint, along with the objective to minimize the expected loss, give rise to the linear program:

# Equalized Odds Post-Processing [20]

$$
\min _ {p _ {a, \hat {y}} \in [ 0, 1 ]} \quad \mathbb {E} [ L (\overline {{Y}}, Y) ] \quad \text { subject   to } \quad \underline {{\alpha}} _ {0, j} = \underline {{\alpha}} _ {1, j} \quad \text { for } \quad j \in \{0, 1 \}. \tag {8}
$$

Returning to our setting where we do not have access to A but only proxy variables $\widehat{A}=h(X)$ , we seek classifiers f that are fair with respect to the sensitive attribute A. Since these attributes are not available, (thus rendering the fairness violation to be unidentifiable [24]), a natural alternative is to minimize the worst-case fair violation with respect to A, which can be computed as shown in Theorem 1. Of course, such an approach will only minimize the worst case fairness and one cannot certify that the true fairness violation will decrease because – as explained above – it is unidentifiable. Nonetheless, since we know what properties optimal classifiers must satisfy (as per Theorem 2), we can now modify the above problem to construct a corrected classifier, $\bar{f}$ , as follows. First, we must employ $\widehat{A}$ in place of A, which amounts to employing $\widehat{\alpha}_{i,j}$ in lieu of $\alpha_{i,j}$ . To this end, denote the (corrected) group conditional probabilities of $\overline{Y}$ to be $\widehat{\alpha}_{i,j}$ . Second, the equalized odds constraint is replaced with the constraint in Theorem 2. Lastly, we also enforce the additional constraints detailed in Assumption 1 on the $\widehat{\alpha}_{i,j}$ . With these modifications in place, we present the following generalized linear program:

# Worst-case Fairness Violation Reduction

$$
\min _ {p _ {\hat {a}, \hat {y}} \in [ 0, 1 ]} \quad \mathbb {E} [ L (\overline {{Y}}, Y) ]
$$

subject to

$$
\frac {\hat {r} _ {0 , j}}{\hat {r} _ {0 , j} + \Delta U} \widehat {\alpha} _ {0, j} - \frac {\hat {r} _ {1 , j}}{\hat {r} _ {1 , j} - \Delta U} \widehat {\alpha} _ {1, j} = \frac {\Delta U}{2} \left(\frac {1}{\hat {r} _ {1 , j} + \Delta U} + \frac {1}{\hat {r} _ {0 , j} - \Delta U}\right) \tag {9}
$$

$$
\frac {U _ {i}}{\hat {r} _ {i , j}} \leq \widehat {\alpha} _ {i, j} \leq 1 - \frac {U _ {i}}{\hat {r} _ {i , j}} \quad \mathrm{for} \quad i, j \in \{0, 1 \}.
$$

The solution to this linear program will yield a classifier $\bar{f}$ that satisfies Assumption 1, has minimal $B_{\mathrm{TPR}}(\bar{f})$ and $B_{\mathrm{FPR}}(\bar{f})$ , and has minimal expected loss. Note, that if $\Delta U = 0$ , then the coefficients $\widehat{\alpha}_{0,j}$ and $\widehat{\alpha}_{1,j}$ will equal one, and so the first set of constraints simply reduces to $\widehat{\alpha}_{0,j} = \widehat{\alpha}_{1,j}$ for $j \in \{0,1\}$ and the linear program above is precisely the post-processing method of Hardt et al. [20] with $\widehat{A}$ in place of A (while enforcing Assumption 1).

Let us briefly recap the findings of this section: We have shown that in a demographically scarce regime, one can provide an upper bound on the true fairness violation of a classifier (Theorem 1). Second, we have presented a precise characterization of the classifiers with minimal worst-case fairness violations. Lastly, we have provided a simple and practical post-processing method (a linear program) that utilizes a sensitive attribute predictor to construct classifiers with minimal worst-case fairness violations with respect to the true, unknown, sensitive attribute.

# 4 Experimental Results

# 4.1 Synthetic Data

We begin with a synthetic example that will allow us to showcase different aspects of our results. The data is constructed from 3 features, $X_{1}, X_{2}, X_{3} \in \mathbb{R}$ , sensitive attribute $A \in \{0,1\}$ , and response $Y \in \{0,1\}$ . The features are sampled from $(X_{1}, X_{2}, X_{3}) \sim \mathcal{N}(\mu, \Sigma)$ , where

$$
\mu = \left[ \begin{array}{c} 1 \\ - 1 \\ 0 \end{array} \right] \quad \text {and} \quad \Sigma = \left[ \begin{array}{c c c} 1 & 0. 0 5 & 0 \\ 0. 0 5 & 1 & 0 \\ 0 & 0 & 0. 0 5 \end{array} \right].
$$

The sensitive attribute, A, response Y, and classifier f, are modeled as

$$
A = \mathbb {I} [ (X _ {3} + 0. 1) \geq 0 ],
$$

$$
Y = \mathbb {I} [ S (X _ {1} + X _ {2} + X _ {3} + \epsilon_ {0} (1 - A) + \epsilon_ {1} A) \geq 0. 3 5 ],
$$

$$
f (X; c _ {1}, c _ {2}) = \mathbb {I} (S (c _ {1} X _ {1} + c _ {2} X _ {2} + c _ {3} X _ {3}) \geq 0. 3 5)
$$

where $\mathbb{I}(\cdot)$ is the indicator function, $S(\cdot)$ is the sigmoid function and. $c_{1}, c_{2}, c_{3} \sim \mathcal{N}(1, 0.01)$ . To ensure there is a non trivial fairness violation, $\epsilon_{0} \sim \mathcal{N}(0, 2), \epsilon_{1} \sim \mathcal{N}(0, 1.5)$ are independent noise variables placed on the samples belonging to the groups A = 0 and A = 1 respectively. Specifically, $\operatorname{Var}(\epsilon_{0}) > \operatorname{Var}(\epsilon_{1})$ guarantees that $f(X; c_{1}, c_{2}, c_{3})$ is unfair with respect to A = 0. Lastly, to measure the predictive capabilities of f, we use the loss function, $L(\hat{Y} \neq y, Y = y) = \mathbb{P}(Y = y)$ , as this maximizes the well known Youden's Index $^{2}$ .

Equal Errors ( $\Delta U = 0$ ): We model the sensitive attribute predictor as $h(X; \delta) = \mathbb{I}((X_3 + 0.1 + \delta) \geq 0)$ , where $\delta \sim \mathcal{N}(0, \sigma^2)$ . We choose $\sigma^2$ so that $h(X)$ has a total error $U \approx 0.04$ distributed so that $\Delta U \approx 0$ with $U_0 \approx U_1 \approx 0.02$ . We generate 1000 classifiers f and for each one, calculate $\Delta_{TPR}$ ,

![](images/4747ad2e0cac86c89fdd6742a3f1fdd92c970eff961e025932747de23c6c6d2c.jpg)

<details>
<summary>violin</summary>

| Category | B_TPR Median | B_TPR Min | B_TPR Max | Δ_TPR Median | Δ_TPR Min | Δ_TPR Max |
| -------- | ------------ | --------- | --------- | ------------ | --------- | --------- |
| f        | 0.24         | 0.18      | 0.26      | 0.09         | 0.05      | 0.12      |
| f̂_fair   | 0.16         | 0.08      | 0.17      | 0.01         | 0.00      | 0.01      |
| f_opt    | 0.16         | 0.08      | 0.17      | 0.01         | 0.00      | 0.01      |
</details>

(a) $\Delta_{\mathrm{TPR}}$ and $B_{\mathrm{TPR}}$

![](images/c0543cc358cf032c1ec4c91063f398b692713335d60a119d055f7e611a5919cd.jpg)

<details>
<summary>violin</summary>

| Category | B_FPR | Δ_FPR |
| -------- | ----- | ----- |
| f        | 0.25  | 0.05  |
| f̂_fair   | 0.22  | 0.01  |
| f_opt    | 0.22  | 0.01  |
</details>

(b) $\Delta_{\mathrm{FPR}}$ and $B_{\mathrm{FPR}}$

![](images/9f4caf2c409ae0c604bb3f7100b035afd0dd8d79f71f7a243ae6faeb7f7bac82.jpg)

<details>
<summary>violin</summary>

| Group   | Median | Q1    | Q3    | Min  | Max  |
|---------|--------|-------|-------|------|------|
| f       | 0.28   | 0.27  | 0.29  | 0.27 | 0.34 |
| f̂_fair  | 0.30   | 0.30  | 0.31  | 0.29 | 0.34 |
| f̂_opt   | 0.30   | 0.30  | 0.31  | 0.29 | 0.34 |
</details>

(c) Expected loss

Figure 1: Synthetic data ( $\Delta U = 0$ ): true fairness, worst-case fairness violations, and expected loss for f, $f_{fair}$ , and $f_{opt}$   
![](images/00f226a80ed27cf9a0e4d5a5a8ef912f214b576add8b9970329fb69323971bfc.jpg)

<details>
<summary>violin</summary>

| Category | B_TPR | Δ_TPR |
| -------- | ----- | ----- |
| f        | 0.35  | 0.10  |
| f̂_fair   | 0.25  | 0.00  |
| f_opt    | 0.15  | 0.05  |
</details>

(a) $\Delta_{\mathrm{TPR}}$ and $B_{\mathrm{TPR}}$

![](images/bd5bd2c899fbb4d446c8d6875f1afbf31a1f75f10f51ceaa8637d193c8e6b9da.jpg)

<details>
<summary>violin</summary>

| Category | B_FPR | Δ_FPR |
| -------- | ----- | ----- |
| f        | 0.25  | 0.05  |
| f̄_fair   | 0.25  | 0.01  |
| f_opt    | 0.22  | 0.04  |
</details>

(b) $\Delta_{\mathrm{FPR}}$ and $B_{\mathrm{FPR}}$

![](images/7a3ee810c9d55c3f258b8b7331232f832e98c4b9178b57839e918c9245d21ce8.jpg)

<details>
<summary>violin</summary>

| Category | Expected Loss |
| -------- | ------------- |
| f        | 0.28          |
| f̄_fair   | 0.32          |
| f_opt    | 0.36          |
</details>

(c) Expected loss   
Figure 2: Synthetic data ( $\Delta U \neq 0$ ): true fairness, worst-case fairness violations, and expected loss for f, $f_{fair}$ and $f_{opt}$

$\Delta_{FPR}, B_{TPR}, B_{FPR}$ , and $\mathbb{E}[L(f,Y)]$ . Then, on each f, we run the (naïve) equalized odds post processing algorithm to correct for fairness with respect $\hat{A}$ to yield a classifier $f_{fair}$ , and we also run our post-processing algorithm to yield an optimal classifier $f_{opt}$ . For both sets of classifiers we again calculate the same quantities.

The results in Fig. 1 present the worst-case fairness violations and expected loss for the 3 sets of different classifiers. Observe that both $B_{\mathrm{TPR}}$ and $B_{\mathrm{FPR}}$ are significantly lower for $f_{\widehat{\mathrm{fair}}}$ and $f_{\mathrm{opt}}$ and that these values for both sets of classifiers are approximately the same. This is expected as $U_0 \approx U_1$ and so performing the fairness correction algorithm and our proposed algorithm amount to solving nearly identical linear programs. We also show the true fairness violations, $\Delta_{\mathrm{TPR}}$ and $\Delta_{\mathrm{FPR}}$ for all the classifiers to portray the gap between the bounds and the true values. As mentioned before, these true values cannot be calculated in a real demographically scarce regime. Nonetheless, the developer of $f$ now knows, post correction, that $|\Delta_{\mathrm{TPR}}|, |\Delta_{\mathrm{FPR}}| \lesssim 0.2$ . Lastly, observe that the expected loss for $f_{\widehat{\mathrm{fair}}}$ and $f_{\mathrm{opt}}$ are naturally higher compared to that of $f$ , however the increase in loss is minimal.

Unequal Errors ( $\Delta U \neq 0$ ): We model the sensitive attribute predictor in the same way as in the previous experiment except with $\delta = c$ , for a constant c so that the sensitive attribute classifier still has the same total error of $U \approx 0.04$ but distributed unevenly so that $\Delta U = -0.04$ with $U_{1} \approx 0.04$ and $U_{0} = 0$ . As in the previous experiment we generate classifiers f, and perform the same correction algorithms to yield $f_{fair}$ and $f_{opt}$ and present the same metrics as before.

The results in Fig. 2 depict the worst-case fairness violations and expected loss for the 3 sets of different classifiers. Observe that our correction algorithm yields classifiers, $f_{\mathrm{opt}}$ , that have significantly lower $B_{\mathrm{TPR}}$ and $B_{\mathrm{FPR}}$ . Furthermore, observe that the $f_{\widehat{\mathrm{fair}}}$ that results from performing the naïve fairness correction algorithm in fact have higher $B_{\mathrm{FPR}}$ the original classifier $f!$ . Even though the total error $U$ has remained the same, its imbalance showcases the optimality of our correction method. Lastly, observe that there is a trade-off, albeit slight, in performing our correction algorithm. The expected loss for $f_{\mathrm{opt}}$ is higher than that of $f_{\widehat{\mathrm{fair}}}$ and $f$ .

![](images/03448ba4e6bd988fb1c329501326ab2789239df04565de60d788f58153387816.jpg)

<details>
<summary>scatter</summary>

| Category | B_TPR | Δ_TPR |
| -------- | ----- | ----- |
| f        | 0.09  | 0.02  |
| f_fair   | 0.07  | 0.005 |
| f_opt    | 0.06  | 0.01  |
</details>

(a) $\Delta_{\mathrm{TPR}}$ and $B_{\mathrm{TPR}}$

![](images/e87d00949077be3a6ce9618690c1f34c8959632cca88da5976af5b456a4ecc35.jpg)

<details>
<summary>scatter</summary>

|        | B_FPR  | Δ_FPR  |
| ------ | ------ | ------ |
| f      | 0.32   | 0.04   |
| f̄_fair | 0.34   | 0.01   |
| f_opt  | 0.30   | 0.06   |
</details>

(b) $\Delta_{\mathrm{FPR}}$ and $B_{\mathrm{FPR}}$

![](images/94d85adbc839c9f724de4e2a52314fe2a6132e3ef3b5a18e1e9a7a9ec6b76347.jpg)

<details>
<summary>violin</summary>

| Category | Expected Loss |
| -------- | ------------- |
| f        | 0.208         |
| f̂_fair   | 0.214         |
| f_opt    | 0.226         |
</details>

(c) Expected loss   
Figure 3: CheXpert data: true fairness, worst-case fairness violations, and expected loss for $f$ , $f_{\widehat{\mathrm{fair}}}$ and $f_{\mathrm{opt}}$

# 4.2 Real World Data:

We now move to a real and important problem on assisted diagnosis on medical images and employ the CheXpert dataset [23]. CheXpert is a large public dataset for chest radiograph interpretation, consisting of 224,316 chest radio graphs of 65,240 patients, with labeled annotations for 14 observations (positive, negative, or unlabeled) including cardiomegaly, atelectasis, edema, consolidation, and several others. Each image is also accompanied by the sensitive attribute sex. We consider a binary classification task in which we aim to learn a classifier $f$ to predict if an image contains annotation for any abnormal condition ( $Y = 1$ ) or does not $Y = 0$ . We then wish to measure and correct for any fairness violations that $f$ may exhibit towards the sex attribute assuming we do not have the true sex attribute at the time of measurement and correction. Instead, since the data contains the sex attribute, we aim to learn a sensitive attribute predictor, $h$ , on a withheld subset of the data. To learn both $f$ and $h$ we use a DenseNet121 convolutional neural network architecture. Images are fed into the network with size $320 \times 320$ pixels. We use the Adam optimizer with default $\beta$ -parameters of $\beta_1 = 0.9$ , $\beta_2 = 0.999$ and learning rate $1 \times 10^{-4}$ which is fixed for the duration of the training. Batches are sampled using a fixed batch size of 16 images and we train for 5 epochs. The sex predictor, $h$ , achieves an error of $U = 0.023$ with $U_1 \approx 0.015$ and $U_0 \approx 0.008$ . On a separate subset of the data, we generate our predictions $\hat{Y} = f$ and $\hat{A} = h$ to yield a dataset over $(\hat{A}, Y, \hat{Y})$ . We utilize the bootstrap method to obtain to generate 500 samples of from this dataset and for each sample, perform the same correction algorithms as before to yield $f_{\widehat{\text{fair}}}$ and $f_{\text{opt}}$ and calculate the same metrics as done in the previous experiments.

The results in Fig. 3 show that our proposed correction method performs the best in reducing $B_{\mathrm{TPR}}$ and $B_{\mathrm{FPR}}$ . Even though $U$ is very small, since $U_{1}$ is approximately 2 times $U_{0}$ , simply correcting for fairness with respect to $\hat{A}$ is suboptimal in reducing the worst-case fairness violations. In particular, the results in Fig. 3a are noteworthy, as they depicts how our proposed correction method, and our bounds, allow the user to certify that the obtained classifier has a fairness violation in TPRs of no more than 0.06, without having access to the true sensitive attributes. Moreover, the improvement is significant, since before the correction one had $|\Delta_{\mathrm{TPR}}| \lesssim 0.10$ . In a high-stakes decision setting, such as this one where the model $f$ could be used to aid in diagnosis, this knowledge could be vital. Naturally, the expected loss is highest for $f_{\mathrm{opt}}$ but that the increase is minimal. We make no claim as to whether this (small) increase in loss is reasonable for this particular problem setting, and the precise trade-offs must be defined in the context of a broader discussion involving policy makers, domain experts and other stakeholders.

# 5 Limitations and Broader Impacts

While our results are novel and informative, they come with limitations. First, our results are limited to EOD (and its relaxations) as definitions of fairness. Fairness is highly context-specific and in many scenarios one may be interested in utilizing other definitions of fairness. One can easily extend

our results to other associative definitions of fairness, such as demographic parity, predictive parity, and others. However, extending our results to counter-factual notions of fairness $[25, 11, 28]$ is non trivial and matter of future work. We recommend thoroughly assessing the problem and context in question prior to selecting a definition. It is crucial to ensure that the rationale behind choosing a definition is based on reasoning from both philosophical and political theory, as each definition implicitly make a distinct set of moral assumptions. For example, with EOD, we implicitly assert that all individuals with the same true label have the same effort-based utility $[21]$ . More generally, other statistical definitions of fairness such as demographic parity and equality of accuracy can be thought of as special instances of Rawlsian equality of opportunity and predictive parity, the other hand, can be thought of as an instance of egalitarian equality of opportunity $[21, 26, 3]$ . We refer the reader to Heidari et al. $[21]$ to understand the relationship between definitions of fairness in machine learning and models of Equality of opportunity (EOP) – an extensively studied ideal of fairness in political philosophy.

A second limitation of our results is Assumption 1. This assumption is relatively mild, as it is met for accurate proxy sensitive attributes (as illustrated in the chest X-rays study). Yet, we conjecture that one can do away with this assumption and consider less accurate proxy sensitive attributes with the caveat that the worst case fairness violations will no longer be linear in the TPRs and FPRs. Thus, the characterization of the classifiers with minimal worst-case bounds would be more involved and the method to minimize these violations will likely be more difficult. Furthermore, while we proposed a simple post-processing correction method, it would be of interest to understand how one could train a classifier – from scratch – to have minimal violations. Lastly, in our setting we assume the sensitive attribute predictor and label classifier are trained on marginal distributions from the same joint distribution. As a next step, it would be important to understand how these results extend to settings where these marginal distributions come from (slightly) different joint distributions. All of this constitutes matter of future work.

Finally, we would like to remark the positive and potentially negative societal impacts of this work. Our contribution is focused on a solution to a technical problem – estimating and correcting for fairness violations when the sensitive attribute and responses are not jointly observed. However, we understand that fairness is a complex and multifaceted issue that extends beyond technical solutions and, more importantly, that there can be disconnect between algorithmic fairness and fairness in a broader socio-technical context. Nonetheless, we believe that technical contributions such as ours can contribute to the fair deployment of machine learning tools. In regards to the technical contribution itself, our results rely on predicting missing sensitive attributes. While such a strategy could be seen as controversial – e.g. because it could involve potential negative consequences such as the perpetuation of discrimination or violation of privacy – this is necessary to build classifiers with minimal worst-case fairness violations in a demographically scarce regime. On the one hand, not allowing for such predictions could be seen as one form of “fairness through unawareness”, which has been proven to be an incorrect and misleading strategy in fairness $[20, 10]$ . Moreover, our post-processing algorithm, similar to that of Hardt et al. $[20]$ , admits implementations in a differentially private manner as well, since it only requires aggregate information about the data. As a result, our method, which uses an analogous formulation with different constraints, can also be carried out in a manner that preserves privacy. Lastly, note that if one does not follow our approach of correcting for the worst-case fairness by predicting the sensitive attributes, other models trained on this data can inadvertently learn this sensitive attribute indirectly and base decisions of it with negative and potentially grave consequences. Our methodology prevents this from happening by appropriately correcting models to have minimal worst-case fairness violations.

# 6 Conclusion

In this paper we address the problem of estimating and controlling potential EOD violations towards an unobserved sensitive attribute by means of predicted proxies. We have shown that under mild assumptions (easily satisfied in practice, as demonstrated) the worst-case fairness violations, $B_{TPR}$ and $B_{FPR}$ , have simple closed form solutions that are linear in the estimated group conditional probabilities $\widehat{\alpha}_{i,j}$ . Furthermore, we give an exact characterization of the properties that a classifier must satisfy so that $B_{TPR}$ and $B_{FPR}$ are indeed minimal. Our results demonstrate that, even when the proxy sensitive attributes are highly accurate, simply correcting for fairness with respect to these proxy attributes might be suboptimal in regards to minimizing the worst-case fairness violations. To this end, we present a simple post-processing method that can correct a pre-trained classifier f to yield an optimally corrected classifier, $\bar{f}$ , i.e. one with minimal worst-case fairness violations.

Our experiments on both synthetic and real data illustrate our theoretical findings. We show how, even if the proxy sensitive attributes are highly accurate, the smallest imbalance in $U_{0}$ and $U_{1}$ renders the naïve correction for fairness with respect to the proxy attributes suboptimal. More importantly, our experiments highlight our method's ability to effectively control the worst-case fairness violation of a classifier with minimal decrease in the classifier's overall predictive power. On a final observation on our empirical results, the reader might be tempted to believe that the classifier $f_{\widehat{fair}}$ (referring to, e.g., Fig. 3) is better because it provides a lower “true” fairness than that of $f_{opt}$ . Unfortunately, these true fairness violations are not identifiable in practice, and all one can compute are the provided upper bounds, which $f_{opt}$ minimizes. In conclusion, our contribution aims to provide better and more rigorous control over potential negative societal impacts that arise from unfair machine learning algorithms in settings of unobserved data.

# References

[1] Philip Adler, Casey Falk, Sorelle A. Friedler, Tionney Nix, Gabriel Rybeck, Carlos Scheidegger, Brandon Smith, and Suresh Venkatasubramanian. Auditing black-box models for indirect influence. Knowl. Inf. Syst., 54:95–122, 2018.   
[2] Alekh Agarwal, Alina Beygelzimer, Miroslav Dudik, John Langford, and Hanna Wallach. A reductions approach to fair classification. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pages 60–69. PMLR, 10–15 Jul 2018.   
[3] Larry A. Alexander. Fair equality of opportunity. Philosophy Research Archives, 11:197–208, 1985. doi: 10.5840/pra19851111.   
[4] Pranjal Awasthi, Matthäus Kleindessner, and Jamie Morgenstern. Equalized odds postprocessing under imperfect group information. In International Conference on Artificial Intelligence and Statistics, pages 1770–1780. PMLR, 2020.   
[5] Pranjal Awasthi, Alex Beutel, Matthäus Kleindessner, Jamie Morgenstern, and Xuezhi Wang. Evaluating fairness of machine learning models under uncertain and incomplete information. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 206–214, 2021.   
[6] Arthur P. Baines and Marsha J. Courchane. Fair lending: Implications for the indirect auto finance market. study prepared for the American Financial Services Association, 11 2014.   
[7] Solon Barocas, Moritz Hardt, and Arvind Narayanan. Fairness and machine learning. fairmlbook.org, 2019, 2018.

[8] Flavio Calmon, Dennis Wei, Bhanukiran Vinzamuri, Karthikeyan Natesan Ramamurthy, and Kush R Varshney. Optimized pre-processing for discrimination prevention. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.   
[9] L. Elisa Celis, Lingxiao Huang, Vijay Keswani, and Nisheeth K. Vishnoi. Fair classification with noisy protected attributes: A framework with provable guarantees. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 1349–1361. PMLR, 18–24 Jul 2021.   
[10] Jiahao Chen, Nathan Kallus, Xiaojie Mao, Geoffry Svacha, and Madeleine Udell. Fairness under unawareness: Assessing disparity when protected class is unobserved. In Proceedings of the conference on fairness, accountability, and transparency, pages 339–348, 2019.   
[11] Silvia Chiappa. Path-specific counterfactual fairness. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):7801–7808, Jul. 2019.   
[12] Evgenii Chzhen, Christophe Denis, Mohamed Hebiri, Luca Oneto, and Massimiliano Pontil. Leveraging labeled and unlabeled data for consistent fair binary classification. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.   
[13] Andrew Cotter, Maya Gupta, Heinrich Jiang, Nathan Srebro, Karthik Sridharan, Serena Wang, Blake Woodworth, and Seungil You. Training well-generalizing classifiers for fairness metrics and other data-dependent constraints. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 1397–1405. PMLR, 09–15 Jun 2019.   
[14] Swarup Dhar, Vanessa Massaro, Darakhshan Mir, and Nathan C. Ryan. Uncertainty in criminal justice algorithms: simulation studies of the pennsylvania additive classification tool. CoRR, abs/2112.00301, 2021.   
[15] Michele Donini, Luca Oneto, Shai Ben-David, John S Shawe-Taylor, and Massimiliano Pontil. Empirical risk minimization under fairness constraints. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.   
[16] Marc Elliott, Peter Morrison, Allen Fremont, Daniel Mccaffrey, Philip Pantoja, and Nicole Lurie. Using the census bureau's surname list to improve estimates of race/ethnicity and associated disparities. Health Services and Outcomes Research Methodology, 9:252–253, 06 2009.   
[17] Michael Feldman, Sorelle A. Friedler, John Moeller, Carlos Scheidegger, and Suresh Venkatasubramanian. Certifying and removing disparate impact. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, page 259–268, New York, NY, USA, 2015. Association for Computing Machinery. ISBN 9781450336642.   
[18] Judy Wawira Gichoya, Imon Banerjee, Ananth Reddy Bhimireddy, John L. Burns, Leo Anthony Celi, Li Ching Chen, Ramon Correa, Natalie Dullerud, Marzyeh Ghassemi, Shih Cheng Huang, Po Chih Kuo, Matthew P. Lungren, Lyle J. Palmer, Brandon J. Price, Saptarshi Purkayastha, Ayis T. Pyrros, Lauren Oakden-Rayner, Chima Okechukwu, Laleh Seyyed-Kalantari, Hari Trivedi, Ryan Wang, Zachary Zaiman, and Haoran Zhang. Ai recognition of patient race in medical imaging: a modelling study. The Lancet Digital Health, 4:e406–e414, June 2022. ISSN 2589-7500. doi: 10.1016/S2589-7500(22)00063-2.

[19] Maya R. Gupta, Andrew Cotter, Mahdi Milani Fard, and Serena Wang. Proxy fairness. CoRR, abs/1806.11212, 2018.   
[20] Moritz Hardt, Eric Price, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016.   
[21] Hoda Heidari, Michele Loi, Krishna P. Gummadi, and Andreas Krause. A moral framework for understanding fair ml through economic models of equality of opportunity. In Proceedings of the Conference on Fairness, Accountability, and Transparency, FAT\* '19, page 181–190, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450361255.   
[22] Kosuke Imai and Kabir Khanna. Improving ecological inference by predicting individual ethnicity from voter registration records. Political Analysis, 24:mpw001, 03 2016.   
[23] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong, Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, and Andrew Y. Ng. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In 33rd AAAI Conference on Artificial Intelligence, AAAI 2019, 31st Innovative Applications of Artificial Intelligence Conference, IAAI 2019 and the 9th AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, pages 590–597. AAAI Press, 2019.   
[24] Nathan Kallus, Xiaojie Mao, and Angela Zhou. Assessing algorithmic fairness with unobserved protected class using data combination. Management Science, 68(3):1959–1981, 2022.   
[25] Matt J Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. Counterfactual fairness. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.   
[26] Andrew Mason. 68Rawlsian Fair Equality of Opportunity. In Levelling the Playing Field: The Idea of Equal Opportunity and its Place in Egalitarian Thought. Oxford University Press, 102006. ISBN 9780199264414. doi: 10.1093/acprof:oso/9780199264414.003.0004.   
[27] Cecilia Munoz, DJ Patil, and Megan Smith. Big data: A report on algorithmic systems, opportunity, and civil rights. Executive Office of the President, 2016.   
[28] Razieh Nabi and Ilya Shpitser. Fair inference on outcomes. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, AAAI'18/IAAI'18/EAAI'18. AAAI Press, 2018. ISBN 978-1-57735-800-8.   
[29] Radiological Society of North America. Rsna pneumonia detection challenge, kaggle competition, 2018.   
[30] Flavien Prost, Pranjal Awasthi, Nick Blumm, Aditee Kumthekar, Trevor Potter, Li Wei, Xuezhi Wang, Ed H Chi, Jilin Chen, and Alex Beutel. Measuring model fairness under noisy covariates: A theoretical perspective. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, pages 873–883, 2021.   
[31] Yaniv Romano, Stephen Bates, and Emmanuel Candes. Achieving equalized odds by resampling sensitive attributes. Advances in Neural Information Processing Systems, 33:361-371, 2020.

[32] Laleh Seyyed-Kalantari, Haoran Zhang, Matthew McDermott, Irene Y Chen, and Marzyeh Ghassemi. Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. Nature medicine, 27(12):2176–2182, 2021.   
[33] Li Shen, Laurie R. Margolies, Joseph H. Rothstein, Eugene Fluder, Russell McBride, and Weiva Sieh. Deep learning to improve breast cancer detection on screening mammography. Scientific Reports, 9, 2019.   
[34] Kush R. Varshney and Homa Alemzadeh. On the safety of machine learning: Cyber-physical systems, decision sciences, and data products. Big Data, 5(3):246–255, 2017.   
[35] Dennis Wei, Karthikeyan Natesan Ramamurthy, and Flavio Calmon. Optimized score transformation for fair classification. In Silvia Chiappa and Roberto Calandra, editors, Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pages 1673–1683. PMLR, 26–28 Aug 2020.   
[36] Joel S. Weissman and Romana Hasnain-Wynia. Advancing health care equity through improved data collection. New England Journal of Medicine, 364, 2011.   
[37] Xiaojiao Yu. Machine learning application in online lending risk prediction, 2017.   
[38] Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, and Krishna P. Gummadi. Fairness Constraints: Mechanisms for Fair Classification. In Aarti Singh and Jerry Zhu, editors, Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pages 962–970. PMLR, 20–22 Apr 2017.   
[39] Yan Zhang. Assessing fair lending risks using race/ethnicity proxies. Comparative Political Economy: Regulation eJournal, 2018.   
[40] Tianxiang Zhao, Enyan Dai, Kai Shu, and Suhang Wang. Towards fair classifiers without sensitive attributes: Exploring biases in related features. 2022.

# A Appendix

# A.1 Proofs

# A.1.1 Explanation of Assumption 1

Assumption 1: For $i, j \in \{0,1\}$ , the classifiers $\widehat{Y} = f(X)$ and $\widehat{A} = h(X)$ satisfy

$$
\frac {U _ {i}}{\hat {r} _ {i , j}} \leq \widehat {\alpha} _ {i, j} \leq 1 - \frac {U _ {i}}{\hat {r} _ {i , j}}.
$$

We now expand on the implications of this assumptions. Recall that $U_{i} = \mathbb{P}(\widehat{A} = i, A \neq i)$ , $\hat{r}_{i,j} = \mathbb{P}(\widehat{A} = i, Y = j)$ , and $\widehat{\alpha}_{i,j} = \mathbb{P}(\widehat{Y} = 1 \mid \widehat{A} = i, Y = j)$ . Thus Assumption 1 states,

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {Y} = 1, \widehat {A} = i, Y = j) \leq \mathbb {P} (\widehat {A} = i, Y = j) - \mathbb {P} (\widehat {A} = i, A \neq i). \tag {10}
$$

Immediately, it is clear that Eq. (10) implies

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {A} = i, Y = j) - \mathbb {P} (\widehat {A} = i, A \neq i) \tag {11}
$$

$$
\Longrightarrow \mathbb {P} (\widehat {A} = i, A \neq i) \leq \frac {1}{2} \mathbb {P} (\widehat {A} = i, Y = j) \tag {12}
$$

Now, the left inequality of Eq. (10) states

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {Y} = 1, \widehat {A} = i, Y = j) \tag {13}
$$

and the right inequality of Eq. (10) states

$$
\mathbb {P} (\widehat {Y} = 1, \widehat {A} = i, Y = j) \leq \mathbb {P} (\widehat {A} = i, Y = j) - \mathbb {P} (\widehat {A} = i, A \neq i)
$$

which implies

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {A} = i, Y = j) - \mathbb {P} (\widehat {Y} = 1, \widehat {A} = i, Y = j) \tag {14}
$$

$$
= \mathbb {P} (\widehat {Y} = 0, \widehat {A} = i, Y = j) \tag {15}
$$

If $j = 1$ , then this implies

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {Y} = 1, \widehat {A} = i, Y = 1) \tag {16}
$$

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {Y} = 0, \widehat {A} = i, Y = 1) \tag {17}
$$

and if $j = 0$ ,

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {Y} = 1, \widehat {A} = i, Y = 0) \tag {18}
$$

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {Y} = 0, \widehat {A} = i, Y = 0) \tag {19}
$$

Any reasonable classifier $\widehat{Y}$ would have the properties

$$
\mathbb {P} (\widehat {Y} = 0, \widehat {A} = i, Y = 1) \leq \mathbb {P} (\widehat {Y} = 1, \widehat {A} = i, Y = 1) \tag {20}
$$

$$
\mathbb {P} (\widehat {Y} = 1, \widehat {A} = i, Y = 0) \leq \mathbb {P} (\widehat {Y} = 0, \widehat {A} = i, Y = 0) \tag {21}
$$

Thus, Assumption 1 is met when

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {Y} = j, \widehat {A} = i, Y \neq j) \tag {22}
$$

# A.1.2 Proof of Theorem 1

We only prove the result for $|\Delta_{\mathrm{TPR}}(f)|$ as the proof for $|\Delta_{\mathrm{FPR}}(f)|$ is completely analogous.

Proof. The rules of conditional probability and the law of total probability allow us to decompose $\alpha_{1,1}$ and $\alpha_{0,1}$ in the following manner,

$$
\alpha_ {1, 1} = \mathbb {P} (\hat {Y} = 1 \mid A = 1, Y = 1) \tag {23}
$$

$$
= \frac {\mathbb {P} (\hat {Y} = 1 , A = 1 , Y = 1)}{\mathbb {P} (A = 1 , Y = 1)} \tag {24}
$$

$$
= \frac {\sum_ {i \in \{0 , 1 \}} \mathbb {P} (\hat {Y} = 1 , A = 1 , Y = 1 , \hat {A} = i)}{\sum_ {j \in \{0 , 1 \}} \sum_ {i \in \{0 , 1 \}} \mathbb {P} (\hat {Y} = j , A = 1 , Y = 1 , \hat {A} = i)} \tag {25}
$$

$$
= \frac {\sum_ {i \in \{0 , 1 \}} \mathbb {P} (\hat {Y} = 1 , A = 1 , Y = 1 \mid \hat {A} = i) \cdot \mathbb {P} (\hat {A} = i)}{\sum_ {j \in \{0 , 1 \}} \sum_ {i \in \{0 , 1 \}} \mathbb {P} (\hat {Y} = j , A = 1 , Y = 1 \mid \hat {A} = i) \cdot \mathbb {P} (\hat {A} = i)} \tag {26}
$$

and

$$
\alpha_ {0, 1} = \mathbb {P} (\hat {Y} = 1 \mid A = 0, Y = 1) \tag {27}
$$

$$
= \frac {\mathbb {P} (\hat {Y} = 1 , A = 0 , Y = 1)}{\mathbb {P} (A = 0 , Y = 1)} \tag {28}
$$

$$
= \frac {\mathbb {P} (\hat {Y} = 1 , Y = 1) - \mathbb {P} (\hat {Y} = 1 , A = 1 , Y = 1)}{\mathbb {P} (Y = 1) - \mathbb {P} (A = 1 , Y = 1)} \tag {29}
$$

$$
= \frac {\mathbb {P} (\hat {Y} = 1 , Y = 1) - \left[ \sum_ {i \in \{0 , 1 \}} \mathbb {P} (\hat {Y} = 1 , A = 1 , Y = 1 \mid \hat {A} = i) \cdot \mathbb {P} (\hat {A} = i) \right]}{\mathbb {P} (Y = 1) - \left[ \sum_ {j \in \{0 , 1 \}} \sum_ {i \in \{0 , 1 \}} \mathbb {P} (\hat {Y} = j , A = 1 , Y = 1 \mid \hat {A} = i) \cdot \mathbb {P} (\hat {A} = i) \right]} \tag {30}
$$

Therefore, $\Delta_{\mathrm{TPR}}(f)=\alpha_{1,1}-\alpha_{0,1}$ is a function of the four probabilities given by

$$
\mathbb {P} (\hat {Y} = j, A = 1, Y = 1 \mid \hat {A} = i) \tag {31}
$$

which are unidentifiable in a demographically scarce regime and therefore not computable.

The Fréchet inequalities tell us that for $i, j \in \{0, 1\}$

$$
\mathbb {P} (\hat {Y} = j, A = 1, Y = 1 \mid \hat {A} = i) \geq \max \{\mathbb {P} (\hat {Y} = j, Y = 1 \mid \hat {A} = i) - \mathbb {P} (A = 0 \mid \hat {A} = i), 0 \} \tag {32}
$$

$$
\mathbb {P} (\hat {Y} = j, A = 1, Y = 1 \mid \hat {A} = i) \leq \min \{\mathbb {P} (\hat {Y} = j, Y = 1 \mid \hat {A} = i), \mathbb {P} (A = 1 \mid \hat {A} = i) \}. \tag {33}
$$

Observe that, $\Delta_{\mathrm{TPR}}(f)$ is an increasing function with respect to the two probabilities

$$
\mathbb {P} (\hat {Y} = 1, A = 1, Y = 1 \mid \hat {A} = i) \tag {34}
$$

and a decreasing one with respect to the two probabilities,

$$
\mathbb {P} (\hat {Y} = 0, A = 1, Y = 1 \mid \hat {A} = i). \tag {35}
$$

As a result, $\Delta_{\mathrm{TPR}}(f)$ is maximal when $\mathbb{P}(\hat{Y} = 1, A = 1, Y = 1 \mid \hat{A} = i)$ achieve their maximum values and $\mathbb{P}(\hat{Y} = 0, A = 1, Y = 1 \mid \hat{A} = i)$ achieve their minimum values. On the other hand, $\Delta_{\mathrm{TPR}}(f)$ is minimal when $\mathbb{P}(\hat{Y} = 1, A = 1, Y = 1 \mid \hat{A} = i)$ achieve their minimum values and

$\mathbb{P}(\hat{Y} = 0, A = 1, Y = 1, |\hat{A} = i)$ achieve their maximum values. With these facts, we now provide the upper bound. Recall from Appendix A.1.1 that Assumption 1 implies

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \frac {1}{2} \mathbb {P} (\widehat {A} = i, Y = j) \tag {36}
$$

$$
\mathbb {P} (\widehat {A} = i, A \neq i) \leq \mathbb {P} (\widehat {Y} = j, \widehat {A} = i, Y \neq j) \leq \mathbb {P} (\widehat {Y} = j, \widehat {A} = i, Y = j) \tag {37}
$$

With Assumption 1 we first provide the values of min $[\mathbb{P}(\hat{Y}=0, A=1, Y=1, |\hat{A}=i)]$ . First,

$$
\mathbb {P} (\hat {Y} = 0, Y = 1, | \hat {A} = 1) - \mathbb {P} (A = 0 \mid \hat {A} = 1) = \frac {\mathbb {P} (\hat {Y} = 0 , \hat {A} = 1 , Y = 1)}{P (\hat {A} = 1)} - \frac {U _ {1}}{\mathbb {P} (\hat {A} = 1)} \tag {39}
$$

$$
\geq 0 \tag {40}
$$

because $\mathbb{P}(\hat{Y} = 0, \hat{A} = 1, Y = 1) - U_1 \geq 0$ . Second,

$$
\mathbb {P} (\hat {Y} = 0, Y = 1, | \hat {A} = 0) - \mathbb {P} (A = 0 \mid \hat {A} = 0) = \frac {\mathbb {P} (\hat {Y} = 0 , \hat {A} = 0 , Y = 1)}{P (\hat {A} = 0)} - \frac {\mathbb {P} (A = 0 , \hat {A} = 0)}{\mathbb {P} (\hat {A} = 0)} \tag {41}
$$

$$
= \frac {\mathbb {P} (\hat {Y} = 0 , \hat {A} = 0 , Y = 1)}{P (\hat {A} = 0)} - \frac {\mathbb {P} (\hat {A} = 0) - U _ {0}}{\mathbb {P} (\hat {A} = 0)} \tag {42}
$$

$$
= \frac {\mathbb {P} (\hat {Y} = 0 , \hat {A} = 0 , Y = 1)}{P (\hat {A} = 0)} - \frac {\mathbb {P} (\hat {A} = 0) - U _ {0}}{\mathbb {P} (\hat {A} = 0)} \tag {43}
$$

$$
= \frac {\mathbb {P} (\hat {Y} = 0 , \hat {A} = 0 , Y = 1)}{P (\hat {A} = 0)} + \frac {U _ {0}}{\mathbb {P} (\hat {A} = 0)} - 1 \tag {44}
$$

Now note that,

$$
\mathbb {P} (\hat {Y} = 0, \hat {A} = 0, Y = 1) = \mathbb {P} (\hat {A} = 0, Y = 1) - \mathbb {P} (\hat {Y} = 1, \hat {A} = 0, Y = 1) \tag {45}
$$

$$
\leq \mathbb {P} (\hat {A} = 0, Y = 1) - U _ {0} \tag {46}
$$

where the second equality is due to Assumption 1. As a result

$$
\frac {\mathbb {P} (\hat {Y} = 0 , \hat {A} = 0 , Y = 1)}{P (\hat {A} = 0)} + \frac {U _ {0}}{\mathbb {P} (\hat {A} = 0)} - 1 \leq \frac {\mathbb {P} (\hat {A} = 0 , Y = 1) - U _ {0}}{P (\hat {A} = 0)} + \frac {U _ {0}}{\mathbb {P} (\hat {A} = 0)} - 1 \tag {47}
$$

$$
= \frac {\mathbb {P} (\hat {A} = 0 , Y = 1)}{P (\hat {A} = 0)} - 1 \leq 0 \tag {48}
$$

Therefore,

$$
\min \left[ \mathbb {P} (\hat {Y} = 0, A = 1, Y = 1 \mid \hat {A} = 1) \right] = \mathbb {P} (\hat {Y} = 0, Y = 1, | \hat {A} = 1) - \mathbb {P} (A = 0 \mid \hat {A} = 1) \tag {49}
$$

$$
\min \left[ \mathbb {P} (\hat {Y} = 0, A = 1, Y = 1 \mid \hat {A} = 0) \right] = 0 \tag {50}
$$

Now we provide the values of $\max [\mathbb{P}(\hat{Y} = 1, A = 1, Y = 1, |\hat{A} = i)]$ . First,

$$
\mathbb {P} (A = 1 \mid \hat {A} = 1) = \frac {\mathbb {P} (A = 1 , \hat {A} = 1)}{\mathbb {P} (\hat {A} = 1)} \tag {51}
$$

$$
= \frac {\mathbb {P} (\hat {A} = 1) - U _ {1}}{\mathbb {P} (\hat {A} = 1)} \tag {52}
$$

$$
\geq \frac {\mathbb {P} (\hat {A} = 1 , Y = 1) - U _ {1}}{\mathbb {P} (\hat {A} = 1)} \tag {53}
$$

$$
\geq \frac {\mathbb {P} (\hat {A} = 1 , Y = 1) - \mathbb {P} (\hat {Y} = 0 , \hat {A} = 1 , Y = 1)}{\mathbb {P} (\hat {A} = 1)} \tag {54}
$$

$$
= \frac {\mathbb {P} (\hat {Y} = 1 , \hat {A} = 1 , Y = 1)}{\mathbb {P} (\hat {A} = 1)} = \mathbb {P} (\hat {Y} = 1, Y = 1 \mid \hat {A} = 1) \tag {55}
$$

Second,

$$
\mathbb {P} (A = 1 \mid \hat {A} = 0) = \frac {\mathbb {P} (A = 1 , \hat {A} = 0)}{\mathbb {P} (\hat {A} = 0)} \tag {56}
$$

$$
\leq \frac {\mathbb {P} (\hat {Y} = 1 , \hat {A} = 0 , Y = 1)}{\mathbb {P} (\hat {A} = 0)} = \mathbb {P} (\hat {Y} = 1, Y = 1 \mid \hat {A} = 0) \tag {57}
$$

Therefore,

$$
\max \left[ \mathbb {P} (\hat {Y} = 1, A = 1, Y = 1 \mid \hat {A} = 1) \right] = \mathbb {P} (\hat {Y} = 1, Y = 1 \mid \hat {A} = 1) \tag {58}
$$

$$
\max \left[ \mathbb {P} (\hat {Y} = 1, A = 1, Y = 1 \mid \hat {A} = 0) \right] = \mathbb {P} (A = 1 \mid \hat {A} = 0) \tag {59}
$$

Plugging these 4 values into $\Delta_{TPR}$ will yield the upper bound,

$$
B _ {1} + C _ {0, 1} = \frac {\hat {r} _ {1 , 1}}{\hat {r} _ {1 , 1} + \Delta U} \widehat {\alpha} _ {1, 1} - \frac {\hat {r} _ {0 , 1}}{\hat {r} _ {0 , 1} - \Delta U} \widehat {\alpha} _ {0, 1} + U _ {0} \left(\frac {1}{\hat {r} _ {1 , 1} + \Delta U} + \frac {1}{\hat {r} _ {0 , 1} - \Delta U}\right) \tag {60}
$$

One can similarly use the assumptions to derive the lower bound,

$$
B _ {1} - C _ {1, 1} = \frac {\hat {r} _ {1 , 1}}{\hat {r} _ {1 , 1} + \Delta U} \widehat {\alpha} _ {1, 1} - \frac {\hat {r} _ {0 , 1}}{\hat {r} _ {0 , 1} - \Delta U} \widehat {\alpha} _ {0, 1} - U _ {1} \left(\frac {1}{\hat {r} _ {1 , 1} + \Delta U} + \frac {1}{\hat {r} _ {0 , 1} - \Delta U}\right) \tag {61}
$$

and thus $|\Delta_{\mathrm{TPR}}| \leq \max\{|B_{1} + C_{0,1}|, |B_{1} - C_{1,1}|\}$ . One can use same arguments to derive the upper bound for $|\Delta_{FPR}|$ .

# A.1.3 Proof of Theorem 2

We prove the result for $|\Delta_{\mathrm{TPR}}(f)|$ . We first start by proving the existence part of the theorem.

Let $\widehat{A} = h(X)$ be a sensitive attribute classifier with errors $U_0$ and $U_{1}$ that produces rates $\hat{r}_{i,j} = \mathbb{P}(\widehat{A} = i,Y = j)$ . Let $\mathcal{F}$ be the set of classifiers for $Y$ such that $\forall f\in \mathcal{F}$ , $f$ and $h$ satisfy Assumption 1. Consider any $f\in \mathcal{F}$ with group conditional probabilities, $\widehat{\alpha}_{i,j} = \mathbb{P}(\widehat{Y} = 1\mid \widehat{A} = i,Y = j)$ . Since we are only proving the result for $|\Delta_{\mathrm{TPR}}(f)|$ , set $j = 1$ . Consider the $xy$ plane, with the $x$ -axis being $\widehat{\alpha}_{0,1}$ and the $y$ -axis being $\widehat{\alpha}_{1,1}$ . We know,

$$
\frac {U _ {i}}{\hat {r} _ {i , 1}} \leq \widehat {\alpha} _ {i, 1} \leq 1 - \frac {U _ {i}}{\hat {r} _ {i , 1}} \tag {62}
$$

which implies

$$
\frac {U _ {i}}{\hat {r} _ {i , 1}} \leq \frac {1}{2}. \tag {63}
$$

The two equations above define a rectangular region in the xy plane with a center $\left(\frac{1}{2},\frac{1}{2}\right)$ , meaning any classifier $f\in F$ , has $\widehat{\alpha}_{i,j}$ that are in this region.

Now, denote $\bar{F}$ to be a the set of classifiers for Y, with group conditional probabilities $\widehat{\alpha}_{i,1}$ , that satisfy the condition,

$$
\frac {\hat {r} _ {0 , 1}}{\hat {r} _ {0 , 1} - \Delta U} \widehat {\underline {{\alpha}}} _ {0, 1} - \frac {\hat {r} _ {1 , 1}}{\hat {r} _ {1 , 1} + \Delta U} \widehat {\underline {{\alpha}}} _ {1, 1} = \frac {\Delta U}{2} \left(\frac {1}{\hat {r} _ {1 , 1} + \Delta U} + \frac {1}{\hat {r} _ {0 , 1} - \Delta U}\right). \tag {64}
$$

This condition defines a line in the $xy$ plane meaning any classifier in $\bar{\mathcal{F}}$ has $\underline{\alpha}_{i,1}$ that are on this line. Now observe that the classifier $\bar{f} \in \bar{\mathcal{F}}$ with $\underline{\alpha}_{i,1} = \frac{1}{2}$ , satisfy the above condition because,

$$
\frac {\hat {r} _ {0 , 1}}{\hat {r} _ {0 , 1} - \Delta U} \left(\frac {1}{2}\right) - \frac {\hat {r} _ {1 , 1}}{\hat {r} _ {1 , 1} + \Delta U} \left(\frac {1}{2}\right) = \frac {1}{2} \left(\frac {\hat {r} _ {0 , 1}}{\hat {r} _ {0 , 1} - \Delta U} - \frac {\hat {r} _ {1 , 1}}{\hat {r} _ {1 , 1} + \Delta U}\right) \tag {65}
$$

$$
= \frac {1}{2} \left(\frac {\hat {r} _ {0 , 1} - \Delta U + \Delta U}{\hat {r} _ {0 , 1} - \Delta U} - \frac {\hat {r} _ {1 , 1} + \Delta U - \Delta U}{\hat {r} _ {1 , 1} + \Delta U}\right) \tag {66}
$$

$$
= \frac {1}{2} \left(1 + \frac {\Delta U}{\hat {r} _ {0 , 1} - \Delta U} - 1 + \frac {\Delta U}{\hat {r} _ {1 , 1} + \Delta U}\right) \tag {67}
$$

$$
= \frac {\Delta U}{2} \left(\frac {1}{\hat {r} _ {1 , 1} + \Delta U} + \frac {1}{\hat {r} _ {0 , 1} - \Delta U}\right) \tag {68}
$$

This implies that the line defined by Eq. (64) intersects the rectangular region that Assumption 1 defines. As a result, $\mathcal{F} \cap \bar{\mathcal{F}}$ is not empty, meaning there exists a classifier $\bar{f} \in \mathcal{F}$ with group conditional probabilities $\widehat{\underline{\alpha}}_{i,1}$ that also satisfies the condition,

$$
\frac {\hat {r} _ {0 , 1}}{\hat {r} _ {0 , 1} - \Delta U} \widehat {\underline {{\alpha}}} _ {0, 1} - \frac {\hat {r} _ {1 , 1}}{\hat {r} _ {1 , 1} + \Delta U} \widehat {\underline {{\alpha}}} _ {1, 1} = \frac {\Delta U}{2} \left(\frac {1}{\hat {r} _ {1 , 1} + \Delta U} + \frac {1}{\hat {r} _ {0 , 1} - \Delta U}\right). \tag {69}
$$

Now we prove that such a classifier has minimal bounds. Theorem 1 tells us that for $f \in \mathcal{F}$

$$
| \Delta_ {\mathrm{TPR}} (f) | \leq B _ {\mathrm{TPR}} (f) \stackrel {\Delta} {=} \max \{| B _ {1} + C _ {0, 1} |, | B _ {1} - C _ {1, 1} | \}
$$

Note that $B_{1}$ is linear in $\widehat{\alpha}_{1,1}$ and $\widehat{\alpha}_{0,1}$ and that $C_{0,1}$ and $C_{1,1}$ are constants such that $B_{1} + C_{0,1} \geq B_{1} - C_{1,1}$ simply because $B_{1} + C_{0,1}$ is the upper bound for $\Delta_{TPR}$ and $B_{1} - C_{0,1}$ is the lower bound.

Since these bounds are affine functions shifted by a constant, then $\min\max\{|B_{1}+C_{0,1}|,|B_{1}-C_{1,1}|\}$ necessarily occurs when

$$
B _ {1} + C _ {0, 1} = - B _ {1} - C _ {1, 1} \tag {70}
$$

meaning

$$
2 B _ {1} = - \left(C _ {1, 1} + C _ {0, 1}\right) \tag {71}
$$

have minimal upper bounds on $|\Delta_{TPR}|$ . After rearranging terms, this condition is precisely

$$
\frac {\hat {r} _ {0 , 1}}{\hat {r} _ {0 , 1} - \Delta U} \widehat {\alpha} _ {0, 1} - \frac {\hat {r} _ {1 , 1}}{\hat {r} _ {1 , 1} + \Delta U} \widehat {\alpha} _ {1, 1} = \frac {\Delta U}{2} \left(\frac {1}{\hat {r} _ {1 , 1} + \Delta U} + \frac {1}{\hat {r} _ {0 , 1} - \Delta U}\right). \tag {72}
$$