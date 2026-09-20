# Enhancing Statistical Validity and Power in Hybrid Controlled Trials: A Randomization Inference Approach with Conformal Selective Borrowing

Ke Zhu $^{12}$ Shu Yang $^{1}$ Xiaofei Wang $^{2}$

# Abstract

External controls from historical trials or observational data can augment randomized controlled trials when large-scale randomization is impractical or unethical, such as in drug evaluation for rare diseases. However, non-randomized external controls can introduce biases, and existing Bayesian and frequentist methods may inflate the type I error rate, particularly in small-sample trials where external data borrowing is most critical. To address these challenges, we propose a randomization inference framework that ensures finite-sample exact and model-free type I error rate control, adhering to the “analyze as you randomize” principle to safeguard against hidden biases. Recognizing that biased external controls reduce the power of randomization tests, we leverage conformal inference to develop an individualized test-then-pool procedure that selectively borrows comparable external controls to improve power. Our approach incorporates selection uncertainty into randomization tests, providing valid post-selection inference. Additionally, we propose an adaptive procedure to optimize the selection threshold by minimizing the mean squared error across a class of estimators encompassing both no-borrowing and full-borrowing approaches. The proposed methods are supported by non-asymptotic theoretical analysis, validated through simulations, and applied to a randomized lung cancer trial that integrates external controls from the National Cancer Database.

$^{1}$ Department of Statistics, North Carolina State University, Raleigh, NC 27695, U.S.A. $^{2}$ Department of Biostatistics and Bioinformatics, Duke University, Durham, NC 27710, U.S.A.. Correspondence to: Shu Yang <syang24@ncsu.edu>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

# 1. Introduction

Randomized controlled trials (RCTs) are the gold standard for making causal inferences on the treatment effect of a new treatment relative to a control treatment. However, large RCTs are often infeasible to conduct in practice when the indications of interest involve rare diseases (U.S. Food and Drug Administration, 2022) or common conditions where few patients are willing to participate due to a lack of equipoise (Miller & Joffe, 2011). RCTs in such a context often lack sufficient statistical power to detect realistic treatment effect sizes. Meanwhile, historical studies or large external databases provide real-world data under control conditions, often referred to as external controls (ECs). By integrating RCT with ECs, hybrid controlled trials have garnered significant interest as an effective approach to enhance the power of RCTs with small sample sizes. However, most existing methods for hybrid controlled trials rely on model-based or asymptotic p-values, which can lead to inflated type I error rates when the randomized sample size is small, or the model is misspecified. Moreover, since ECs are not randomized, they may systematically differ from randomized controls, even after adjusting for measured confounders. Directly incorporating these ECs may introduce hidden bias, compromising the validity of the statistical inference. Strictly controlling the type I error rate in hybrid controlled trials, especially with small sample sizes and unmeasured confounding, remains an open problem.

To address this problem, we extend the randomization inference framework to hybrid controlled trials. To utilize ECs, we use a doubly robust estimator of the average treatment effect (ATE) as the test statistic, which incorporates both RCT and EC data and effectively balances the measured confounders between RCT and EC (Li et al., 2023b). Then, Fisher randomization tests (FRTs) are performed using only the randomization in the RCT. In contrast to the asymptotic inference in Li et al. (2023b), which relies on (i) large sample sizes for both the RCT and EC, (ii) correct specification of at least one of the two nuisance models, and (iii) no unmeasured confounders, the FRT strictly controls the type I error rate without requiring any of these conditions, thus achieving model-free, finite-sample exact inference. The validity of the FRT relies solely on the randomization

within the RCT, which is typically well-managed by the study design. Furthermore, we perform a power analysis for FRT in hybrid controlled trials and show that incorporating unbiased ECs with correctly specified models can enhance statistical power. However, EC borrowing is not a free lunch, as including biased ECs may diminish power.

The power issue motivates us to develop a method that selectively incorporates unbiased ECs rather than indiscriminately borrowing all ECs. Unlike observational studies, where the assumption of no unmeasured confounders is untestable, a key advantage of hybrid controlled trials is that the bias in ECs can be identified by comparing EC units to randomized control units. Existing methods mitigate hidden bias by penalized bias estimation and selective borrowing (Gao et al., 2025), where selection consistency depends on asymptotic arguments, potentially leading to inferior performance in small samples.

We propose a novel approach called Conformal Selective Borrowing (CSB), which tests the comparability of ECs and selectively incorporates them using conformal inference (Vovk et al., 2005; Lei et al., 2018). We measure the bias of each EC using a score function that can flexibly accommodate parametric or machine-learning models. We then calibrate this score to a conformal p-value, which test the exchangeability of each EC. These conformal p-values are valid in finite samples, distribution-free, and do not depend on the asymptotic properties of models. CSB offers three advantages: (i) individual borrowing decisions for each EC, (ii) flexibility in using parametric or machine learning models for bias estimation, and (iii) finite-sample guarantees with stable performance in small samples.

In summary, the proposed methods leverage the two key advantages of hybrid controlled trials: (i) randomization within the RCT data allows us to use FRT to control the type I error rate, and (ii) the presence of randomized controls enables us to evaluate bias in ECs using conformal p-values, selectively borrow unbiased ECs, and enhance power. We account for selection uncertainty in FRT and offer valid post-selection inference. Both FRT and CSB are model-free, distribution-free, and maintain finite-sample exact properties, allowing them to flexibly incorporate state-of-the-art machine learning methods while remaining valid for any sample size or data distribution. To ensure robust performance across varying bias magnitudes, we propose a data-adaptive procedure for determining the selection threshold to minimize the MSE of the CSB estimator. Our MSE-guided adaptive threshold offers key advantages: (i) it improves FRT power over RCT-only analysis when EC bias is negligible or detectable; when the bias is non-negligible yet difficult to detect, it may lead to power loss, though FRT still maintains valid Type I error control; (ii) it enables CSB to serve as both a powerful test statistic and an accurate ATE estimator; (iii) the empirical MSE of CSB can be approximated leveraging the RCT-only estimator, making the procedure practically feasible, and we provide a non-asymptotic excess risk bound for its performance. The advantages of our approach are shown via simulations and a lung cancer RCT with ECs from the National Cancer Database.

# 1.1. Related work

Hybrid controlled trials aim to integrate ECs to boost RCT efficiency (Pocock, 1976). For an overview of RCT and RWD integration, see Colnet et al. (2024). A key challenge is biases in ECs, which stem from factors like selection bias, non-concurrency, and measurement error (U.S. Food and Drug Administration, 2023). Statistically, biases are categorized as measured and unmeasured confounding. Measured confounding, or covariate shift, refers to systematic differences in observed covariates between RCs and ECs. To address measured confounding, covariate balancing techniques such as matching, inverse propensity score weighting, calibration weighting, and their augmented counterparts can be employed (Li et al., 2023b; Valancius et al., 2024; Li & Luedtke, 2023). When there is unmeasured confounding between RCT and EC, a rich body of literature addresses the hidden bias through various strategies, including test-thenpool (Viele et al., 2014; Yuan et al., 2019; Li et al., 2020; Ventz et al., 2022; Liu et al., 2022; Yang et al., 2023; Gao & Yang, 2023; Dang et al., 2023), weighted combination (Chen et al., 2020; 2021a; Cheng & Cai, 2021; Li et al., 2022; Oberst et al., 2022; Rosenman et al., 2023; Chen et al., 2023; Karlsson et al., 2024), selective borrowing (Chen et al., 2021b; Li et al., 2023a; Zhai & Han, 2022; Gao et al., 2025; Huang et al., 2023), bias modeling (Stuart & Rubin, 2008; Cheng et al., 2023; Li & Jemielita, 2023; van der Laan et al., 2024; Yang et al., 2024; Gu et al., 2024), control variates or prognostic adjustment (Yang & Ding, 2020; Guo et al., 2022; Schuler et al., 2022; Gagnon-Bartsch et al., 2023), Bayesian methods (Hobbs et al., 2011; Schmidli et al., 2014; Jiang et al., 2023; Kwiatkowski et al., 2024; Alt et al., 2024; Lin et al., 2024; 2025), and sensitivity analysis (Yi et al., 2023). None of them use randomization inference or conformal inference to address unmeasured confounding in hybrid controlled trials with a small sample size.

Randomization inference, introduced by Fisher (1935), provides finite-sample exact p-values for any test statistic and is widely endorsed (Rosenberger et al., 2019; Proschan & Dodd, 2019; Young, 2019; Bind & Rubin, 2020; Carter et al., 2023). Randomization tests are useful for small sample trials or complex designs, including cluster experiments with few clusters (Rabideau & Wang, 2021) and adaptive experiments (Simon & Simon, 2011; Plamadeala & Rosenberger, 2012; Nair & Janson, 2023; Freidling et al., 2024). Randomization tests have appeared in regulatory guidance documents to ensure type I error rate control in adaptive

designs when conventional statistical methods fail (European Medicines Agency, 2015; U.S. Food and Drug Administration, 2019; Carter et al., 2023). For an overview of randomization inference, see Zhang & Zhao (2023) and Ritzwoller et al. (2024). Nevertheless, the randomization inference hasn't been applied to hybrid controlled trials, especially with selective borrowing to address unmeasured confounding.

Conformal inference, or conformal prediction, is a model-free method providing finite-sample valid uncertainty quantification for individual predictions (Vovk et al., 2005), particularly useful in high-stakes scenarios with black-box machine learning models (Angelopoulos & Bates, 2023). Two main applications are most relevant to this paper. The first involves using conformal inference to infer individual treatment effects (Chernozhukov et al., 2021; Lei & Candès, 2021). The second line is in outlier detection (Guan & Tibshirani, 2022; Bates et al., 2023; Liang et al., 2024). These studies inspire us to treat biased ECs as outliers and use conformal p-values to test their exchangeability. Our primary goal, however, is to boost FRT power by selectively borrowing unbiased ECs with conformal p-values. The adaptive selection threshold that minimized the estimator's MSE is also a novel approach.

# 2. Randomization inference framework

# 2.1. Preliminaries

Consider $n_{R}$ patients in the RCT, $n_{E}$ patients in the EC group, and $n = n_{R} + n_{E}$ patients in total. Let S = 1 for patients in RCT and S = 0 for patients in the EC group. Let A denote the binary treatment, where A = 1 stands for treatment and A = 0 stands for control. We denote $T = \{i : A_i = 1, S_i = 1\}$ , $C = \{i : A_i = 0, S_i = 1\}$ , $R = T \cup C$ , and $E = \{i : S_i = 0\}$ . Let X denote the baseline covariates, Y denote the observed outcome, and $Y(0)$ and $Y(1)$ denote the potential outcomes. In an RCT, we randomize $n_{R}$ patients into either the treatment or control groups based on the known propensity score $e(x) = \mathbb{P}(A = 1 \mid X = x, S = 1)$ . This results in $n_1$ patients in the treatment group and $n_0$ patients in the control group. For $n_{E}$ patients in the EC group, since all of them are under control, we have A = 0 for S = 0. Let $\pi(x) = \mathbb{P}(S = 1 \mid X = x)$ denote the sampling score of participating in the RCT. We consider the average treatment effect in the RCT population as our estimand $\tau = \mathbb{E}\{Y(1) - Y(0) \mid S = 1\}$ . For RCT data, the following standard identification assumptions are considered (Imbens & Rubin, 2015).

Assumption 2.1 (RCT identification). (i) (Consistency) $Y = AY(1) + (1 - A)Y(0)$ . (ii) (Positivity) $0 < e(x) < 1$ for all $x$ such that $f_{X|S}(x|1) > 0$ , where $f_{X|S}(x|s)$ is the conditional p.d.f. of $X$ given $S = s$ . (iii) (Randomization)

$$
Y (a) \perp A \mid (X, S = 1), a = 0, 1.
$$

Under Assumption 2.1, $\tau$ is identifiable based on RCT data. We denote the conditional outcome mean functions by $\mu_a(x) = \mathbb{E}(Y \mid X = x, A = a, S = 1)$ , $a = 0, 1$ . We estimate $\mu_a(x)$ and $e(x)$ with only RCT data and denote the estimated functions by $\hat{\mu}_{a,\mathcal{R}}(x)$ and $\hat{e}(x)$ , respectively. An RCT-only doubly robust estimator of $\tau$ is

$$
\begin{array}{l} \hat {\tau} _ {\mathcal {R}} = \frac {1}{n _ {\mathcal {R}}} \sum_ {i = 1} ^ {n} S _ {i} \left[ \hat {\mu} _ {1, \mathcal {R}} (X _ {i}) + \frac {A _ {i}}{\hat {e} (X _ {i})} \left\{Y _ {i} - \hat {\mu} _ {1, \mathcal {R}} (X _ {i}) \right\} \right. \\ \left. - \hat {\mu} _ {0, \mathcal {R}} (X _ {i}) - \frac {1 - A _ {i}}{1 - \hat {e} (X _ {i})} \{Y _ {i} - \hat {\mu} _ {0, \mathcal {R}} (X _ {i}) \} \right], \\ \end{array}
$$

which is referred to as the No Borrowing (NB) approach hereafter. In RCTs, since the propensity score model $e(x)$ is known, $\hat{\tau}_{R}$ is consistent and asymptotically normal regardless of whether $\mu_{a}(x)$ is correctly specified for a = 0, 1. Thus, $\hat{\tau}_{R}$ serves as a model-assisted covariate-adjusted ATE estimator whose asymptotic variance attains the semiparametric efficiency bound if $\mu_{a}(x)$ is correctly specified for a = 0, 1. The efficiency of $\hat{\tau}_{R}$ could be further improved by borrowing information from EC data. To incorporate EC data for estimating $\tau$ , many scholars have considered the following assumption (Li et al., 2023b).

Assumption 2.2 (Mean exchangeability). $\mathbb{E}\{Y(0) \mid X, S = 0\} = \mathbb{E}\{Y(0) \mid X, S = 1\}$ .

Under Assumptions 2.1 and 2.2, $\tau$ could be identified with both RCT and EC data. We estimate $\mu_0(x)$ with RCT and EC data and denote the estimated functions by $\hat{\mu}_{0,\mathcal{R} + \mathcal{E}}(x)$ . Let $\hat{\pi}_{\mathcal{E}}(x)$ denote the estimated sampling score. The variance ratio between randomized controls and ECs is denoted by $r(x) = \mathbb{V}\{Y(0) \mid X = x, A = 0, S = 1\} / \mathbb{V}\{Y(0) \mid X = x, A = 0, S = 0\}$ . Let $\hat{r}_{\mathcal{E}}(x)$ denote the estimated variance ratio. Li et al. (2023b) proposed a doubly robust estimator of $\tau$ :

$$
\begin{array}{l} \hat {\tau} _ {\mathcal {R} + \mathcal {E}} = \frac {1}{n _ {\mathcal {R}}} \sum_ {i = 1} ^ {n} \left[ S _ {i} \hat {\mu} _ {1, \mathcal {R}} (X _ {i}) + S _ {i} \frac {A _ {i}}{\hat {e} (X _ {i})} \{Y _ {i} - \hat {\mu} _ {1, \mathcal {R}} (X _ {i}) \} \right. \\ \left. - S _ {i} \hat {\mu} _ {0, \mathcal {R} + \mathcal {E}} (X _ {i}) - W _ {i} \{Y _ {i} - \hat {\mu} _ {0, \mathcal {R} + \mathcal {E}} (X _ {i}) \} \right], \tag {1} \\ W _ {i} = \hat {\pi} _ {\mathcal {E}} (X _ {i}) \frac {S _ {i} (1 - A _ {i}) + (1 - S _ {i}) \hat {r} _ {\mathcal {E}} (X _ {i})}{\hat {\pi} _ {\mathcal {E}} (X _ {i}) \{1 - \hat {e} (X _ {i}) \} + \{1 - \hat {\pi} _ {\mathcal {E}} (X _ {i}) \} \hat {r} _ {\mathcal {E}} (X _ {i})}. \\ \end{array}
$$

$\hat{\tau}_{\mathcal{R}+\mathcal{E}}$ is referred to as the Full Borrowing (FB) approach hereafter. The term “Full” here refers to incorporating the full set of ECs to construct $\hat{\tau}_{\mathcal{R}+\mathcal{E}}$ , while down-weighting those ECs based on similarity measured by X, thereby addressing bias caused by observed confounders. $\hat{\tau}_{\mathcal{R}+\mathcal{E}}$ is consistent and asymptotically normal if either (i) $\mu_{a}(x)$ is correctly specified for a = 0, 1, or (ii) both $\pi(x)$ and $e(x)$ are correctly specified. If all models for $\mu_{a}(x)$ , a = 0, 1, $\pi(x)$ , and $e(x)$ are correctly specified, $\hat{\tau}_{\mathcal{R}}$ achieves the semi-parametric efficiency bound.

However, asymptotic inference for $\hat{\tau}_{R+E}$ may be invalid due to three main reasons: (i) it assumes $n_{R} \to \infty$ , which contradicts the motivation for EC borrowing, where the sample size of the RCT is typically small; (ii) it relies on the correct specification of at least one of the two nuisance models, which may be violated because sophisticated models are difficult to work with under small sample sizes; and (iii) it depends on Assumption 2.2, which may be violated due to unmeasured confounders. To address these issues, we consider a finite-sample exact randomization inference framework that maintains strict type I error rate control even if all models are misspecified and Assumption 2.2 fails. We consider $\hat{\tau}_{R}$ and $\hat{\tau}_{R+E}$ as candidate test statistics and propose a new class of test statistics in Section 3 to achieve improved power across various scenarios.

# 2.2. Fisher randomization test

In the randomization inference framework, we are conditional on the potential outcomes $Y_{i}(a)$ and covariates $X_{i}$ for $i \in R \cup E$ , and consider the randomized assignment $\boldsymbol{A} = (A_{1}, \ldots, A_{n})$ as the sole source of randomness. Since $A_{i}$ for $i \in R$ is well controlled and known in the RCT, we can leverage this advantage to guarantee the validity of inference without any additional assumptions. Let A denote the set of all possible assignments generated by the actual RCT design. Since all external units are under control, we have $A_{i} = 0$ for $i \in E$ . Randomization inference accommodates not only Bernoulli trials with $A_{i} \stackrel{i.i.d.}{\sim} \text{Bernoulli}(p)$ for $i \in R$ but also complex designs like covariate-adaptive randomization (Rosenberger & Lachin, 2015).

Consider Fisher's sharp null hypothesis $H_0: Y_i(0) = Y_i(1), \forall i \in \mathcal{R}$ , which states no treatment effect for any units in RCT. Based on $H_0$ , we could impute all potential outcomes $Y_i^{\mathrm{imp}}(0) = Y_i^{\mathrm{imp}}(1) = Y_i$ for $i \in \mathcal{R}$ . Let $T(\boldsymbol{A})$ denote the test statistic, which depends on the assignment $\boldsymbol{A} \in \mathcal{A}$ . $T(\boldsymbol{A})$ could be $|\hat{\tau}_{\mathcal{R}}(\boldsymbol{A})|, |\hat{\tau}_{\mathcal{R} + \mathcal{E}}(\boldsymbol{A})|$ , or the estimator introduced in Section 3. The theoretical guarantee of type I error rate control holds for any test statistic, including those involving ECs, even if these ECs have hidden biases. This is one of the key merits of randomization inference. We define the $p$ -value for measuring the extremeness of the observed $T(\boldsymbol{A})$ against $H_0$ as

$$
p ^ {\mathrm{FRT}} = \mathbb {P} _ {\boldsymbol {A} ^ {*}} \left\{T (\boldsymbol {A} ^ {*}) \geq T (\boldsymbol {A}) \right\},
$$

where $A^{*} \in \mathcal{A}$ has the same distribution as $A$ and is independent of $A$ , and $\mathbb{P}_{A^{*}}$ is taken over the distribution of $A^{*}$ .

Theorem 2.3. Under $H_0$ , for $\alpha \in (0,1)$ , we have $\mathbb{P}_A(p^{\mathrm{FRT}} \leq \alpha) \leq \alpha$ , where $\mathbb{P}_A$ is taken over the distribution of $A$ . If we further assume that $T(A)$ takes distinct values for different $A \in A$ , then we have $\mathbb{P}_A(p^{\mathrm{FRT}} \leq \alpha) = [\alpha |A|] / |A| > \alpha - 1 / |A|$ , where $\lfloor x \rfloor$ represents the greatest integer less than or equal to $x$ .

In practice, we use Monte Carlo to approximate $p^{FRT}$ . Based on the RCT's actual randomization, we generate the new assignment $A_{i}^{b}$ for $i \in R$ and set $A_{i}^{b} \equiv 0$ for $i \in E$ since the randomization in the RCT does not affect the assignments of the ECs. A caveat is that the assignment of ECs should not be permuted, as this would violate the “analyze as you randomize” principle and compromise the validity of the FRT. The new assignment vector is denoted as $\boldsymbol{A}^{b} = (A_{1}^{b}, \ldots, A_{n}^{b})$ . We generate assignments for B times and obtain $\hat{p}^{\mathrm{FRT}} = \left[\sum_{b=1}^{B} \mathbb{I}\{T(\boldsymbol{A}^{b}) \geq T(\boldsymbol{A})\} + 1\right]/(B + 1)$ , where the “+1” term accounts for A itself.

Theorem 2.3 shows that FRT exactly controls the type I error rate in finite samples, regardless of Assumption 2.2, because, under $H_{0}$ , the reference distribution is derived from true randomization, which is well-controlled in clinical trials. However, the power of FRT heavily depends on the choice of test statistic, making it the most critical decision in randomization inference.

# 2.3. Model-based power analysis

There are two primary approaches for conducting a power analysis of FRT: model-based or simulation-based (Rosenberger & Lachin, 2015). We first perform a model-based power analysis under Assumption 2.2, highlighting how low variance of a consistent test statistic enhances the power of FRT. In the following section, we conduct a simulation-based power analysis for a more challenging scenario where Assumption 2.2 does not hold, showing that the bias of an inconsistent test statistic reduces the power of FRT.

Let $M$ denote the total number of possible assignments, $F_{1,n,M}(t) = \mathbb{P}_{\boldsymbol{A}}(T(\boldsymbol{A}) \leq t)$ denote the randomization distribution of $T(\boldsymbol{A})$ , and $F_{0,n,M}(t) = \mathbb{P}_{\boldsymbol{A}^*}(T(\boldsymbol{A}^*) \leq t)$ denote the reference distribution of $T(\boldsymbol{A}^*)$ under $H_0$ . Both $F_{1,n,M}$ and $F_{0,n,M}$ are discrete in finite samples. To apply empirical process theory and derive asymptotic rates for testing power, we assume continuous super-population distributions $F_{1,n}$ and $F_{0,n}$ , with $F_{1,n,M}$ and $F_{0,n,M}$ representing the empirical distribution functions based on $M$ independent samples drawn from $F_{1,n}$ and $F_{0,n}$ , respectively. In cases where these assumptions do not hold, FRT still controls the type I error rate, and we will investigate its power through simulation in Section 4. Based on those notations, the $p$ -value and the power can be expressed as $p^{\mathrm{FRT}} = \mathbb{P}_{\boldsymbol{A}^*}\{T(\boldsymbol{A}^*) \geq T(\boldsymbol{A})\} = 1 - F_{0,n,M}(T(\boldsymbol{A}))$ , and $\psi_{n,M} = \mathbb{P}_{\boldsymbol{A}}(p^{\mathrm{FRT}} \leq \alpha) = \mathbb{P}_{\boldsymbol{A}}\{1 - F_{0,n,M}(T(\boldsymbol{A})) \leq \alpha\} = 1 - F_{1,n,M}(F_{0,n,M}^{-1}(1 - \alpha))$ .

# Theorem 2.4. For fixed $n > 0$ , suppose

(a) There are continuous cumulative distribution functions (c.d.f.) $F_{0,n}$ and $F_{1,n}$ , such that $F_{0,n,M}$ and $F_{1,n,M}$ are the empirical distribution functions based on $M$ independent samples drawn from $F_{0,n}$ and $F_{1,n}$ , respectively.

(b) There is $\sigma_{n} > 0$ and a continuous c.d.f. $F$ such that $F_{0,n}(t) = F(t / \sigma_n)$ for all $t\in \mathbb{R}$ .   
(c) For ATE $\tau$ , $F_{1,n}(t) = F_{0,n}(t - \tau) = F\big((t - \tau) / \sigma_n\big)$ , for all $t \in \mathbb{R}$ .

For $0 < \iota < 0.5$ and sufficiently large $M$ ,

$$
\mathbb {E} \left(\psi_ {n, M}\right) \geq 1 - F \left(F ^ {- 1} (1 - \alpha) - \tau / \sigma_ {n}\right) - O \left(M ^ {- 0. 5 + \iota}\right),
$$

where $\mathbb{E}$ is over $M$ independent samples from $F_{0,n}$ and $F_{1,n}$ .

For a given $\tau \neq 0$ , significance level $\alpha$ , and design with possible assignments M, Theorem 2.4 shows that the power of the FRT also depends on the variance of the test statistic, $\sigma_{n}$ . Under Assumptions 2.1 and 2.2, and with all working models correctly specified, $\hat{\tau}_{R+E}$ is consistent and has a variance that is less than or equal to that of $\hat{\tau}_{R}$ (Li et al., 2023b). Thus, when there is no hidden bias, using $\hat{\tau}_{R+E}$ as the test statistic improves the power of the FRT compared to $\hat{\tau}_{R}$ , as shown in subplot (B) of Figure 1.

# 2.4. Simulation-based power analysis

When unmeasured confounding exists between RCT and EC data, Assumption 2.2 is violated, rendering $\hat{\tau}_{\mathcal{R} + \mathcal{E}}$ inconsistent. In such cases, asymptotic inference based on $\hat{\tau}_{\mathcal{R} + \mathcal{E}}$ is invalid and fails to control the type I error rate. In contrast, since Theorem 2.3 holds for any test statistic, FRT can still control the type I error with the inconsistent test statistic $\hat{\tau}_{\mathcal{R} + \mathcal{E}}$ , highlighting a core merit of FRTs. However, the violation of Assumption 2.2 subsequently causes Assumption (c) in Theorem 2.4 to be unfulfilled, rendering FRT with $\hat{\tau}_{\mathcal{R} + \mathcal{E}}$ unable to achieve a power improvement over FRT with $\hat{\tau}_{\mathcal{R}}$ . Furthermore, employing $\hat{\tau}_{\mathcal{R} + \mathcal{E}}$ as the test statistic results in a substantial loss of power compared to using $\hat{\tau}_{\mathcal{R}}$ , as illustrated in subplot (D) of Figure 1.

The trade-off between $\hat{\tau}_{R}$ and $\hat{\tau}_{R+\varepsilon}$ generally arises between a causal estimator that ignores additional information and assumptions and one that incorporates them but risks bias if the assumptions fail (Rothenhäusler, 2020; Rothenhäusler et al., 2021). In the next section, instead of choosing between $\hat{\tau}_{R}$ and $\hat{\tau}_{R+\varepsilon}$ , we construct a class of ATE estimators, $\hat{\tau}_{\gamma}$ , indexed by a tuning parameter $\gamma$ and encompassing $\hat{\tau}_{R}$ and $\hat{\tau}_{R+\varepsilon}$ as special cases. We then propose a data-adaptive procedure to select $\gamma$ that minimizes the MSE of $\hat{\tau}_{\gamma}$ , thereby enhancing the power of FRT by using $\hat{\tau}_{\gamma}$ as the test statistic.

# 3. Conformal Selective Borrowing

# 3.1. A class of estimators

Motivated by heterogeneous scenarios where some ECs satisfy Assumption 2.2 while others do not, we propose an individualized test-then-pool approach that leverages conformal inference to select comparable ECs. The conformal $p$ -value $p_j^* \in (0,1]$ is used to test the exchangeability of each EC $j \in \mathcal{E}$ . The selected EC set is then defined as $\hat{\mathcal{E}}(\gamma) = \{j \in \mathcal{E} : p_j^* > \gamma\}$ , where $\gamma \in [0,1]$ is a selection threshold. Substituting $\mathcal{E}$ with $\hat{\mathcal{E}}(\gamma)$ in (1), we obtain the Conformal Selective Borrowing (CSB) estimator:

$$
\begin{array}{l} \hat {\tau} _ {\gamma} = \frac {1}{n _ {\mathcal {R}}} \sum_ {i = 1} ^ {n} \left[ S _ {i} \hat {\mu} _ {1, \mathcal {R}} (X _ {i}) + S _ {i} \frac {A _ {i}}{\hat {e} (X _ {i})} \{Y _ {i} - \hat {\mu} _ {1, \mathcal {R}} (X _ {i}) \} \right. \\ \left. - S _ {i} \hat {\mu} _ {0, \mathcal {R} + \hat {\mathcal {E}} (\gamma)} (X _ {i}) - V _ {i} \{Y _ {i} - \hat {\mu} _ {0, \mathcal {R} + \hat {\mathcal {E}} (\gamma)} (X _ {i}) \} \right], \tag {2} \\ \end{array}
$$

$$
\begin{array}{l} V _ {i} = \hat {\pi} _ {\hat {\mathcal {E}} (\gamma)} (X _ {i}) \\ \times \frac {S _ {i} (1 - A _ {i}) + (1 - S _ {i}) \mathbb {I} \{i \in \hat {\mathcal {E}} (\gamma) \} \hat {r} _ {\hat {\mathcal {E}} (\gamma)} (X _ {i})}{\hat {\pi} _ {\hat {\mathcal {E}} (\gamma)} (X _ {i}) \{1 - \hat {e} (X _ {i}) \} + \{1 - \hat {\pi} _ {\hat {\mathcal {E}} (\gamma)} (X _ {i}) \} \hat {r} _ {\hat {\mathcal {E}} (\gamma)} (X _ {i})}. \\ \end{array}
$$

CSB represents a class of ATE estimators: when $\gamma = 1$ , no ECs borrowed and $\hat{\mathcal{E}}(1) = \varnothing$ , we have $\hat{\tau}_{1} \equiv \hat{\tau}_{R}$ ; when $\gamma = 0$ , all ECs borrowed and $\hat{\mathcal{E}}(0) = \mathcal{E}$ , we have $\hat{\tau}_{0} = \hat{\tau}_{R + \mathcal{E}}$ . For $0 < \gamma < 1$ , $\hat{\tau}_{\gamma}$ balances the trade-off between borrowing more ECs with a smaller $\gamma$ and discarding more ECs with a larger $\gamma$ . By using $T(\boldsymbol{A}) = |\hat{\tau}_{\gamma}|$ as the test statistic for FRT and allowing $\hat{\mathcal{E}}(\gamma)$ to vary with resampling A in FRT could account for selection uncertainty and provide valid post-selection inference. The following sections introduce various conformal p-values and a data-adaptive procedure for selecting $\gamma$ to minimize the MSE of $\hat{\tau}_{\gamma}$ .

# 3.2. Conformal $p$ -value

Split conformal p-value. We first consider split conformal inference (Papadopoulos et al., 2002). We randomly split C into a calibration set $C_{1}$ and a training set $C \setminus C_{1}$ according to a prespecified sample size ratio, for example, 1:3. We use a score function $s(x,y)$ to measure the “nonconformity” of $(x,y)$ . For example, we can use the absolute residual as the score function: $s_{i} = |Y_{i} - \hat{f}_{-C_{1}}(X_{i})|$ for $i \in C_{1}$ and $s_{j} = |Y_{j} - \hat{f}_{-C_{1}}(X_{j})|$ , where $\hat{f}_{-C_{1}}(x)$ is a prediction model fitted by the training set $C \setminus C_{1}$ . Intuitively, if $(X_{j}, Y_{j})$ is not exchangeable (see Remark 3.2 for a formal definition) with $\{(X_{i}, Y_{i})\}_{i \in C_{1}}$ , $s_{j}$ should be large compared to $\{s_{i}\}_{i \in C_{1}}$ . Thus, we define the split conformal p-value as the proportion of $\{s_{i}\}_{i \in C_{1}}$ that are larger than $s_{j}$ , that is, $p_{j}^{\text{split}} = \{\sum_{i \in C_{1}} \mathbb{I}(s_{i} \geq s_{j}) + 1\}/(|C_{1}| + 1)$ , where I is the indicator function, and the “+1” accounts for including $s_{j}$ itself. If $p_{j}^{split}$ is smaller than a threshold $\gamma$ , we reject the hypothesis of exchangeability and discard EC j. The following theoretical guarantee states that if EC j is exchangeable, the rejection rate is less than $\gamma$ .

Proposition 3.1. For $j \in \mathcal{E}$ , suppose that $(X_j, Y_j)$ and $\{(X_i, Y_i)\}_{i \in \mathcal{C}}$ are exchangeable. For $\gamma \in (0, 1)$ , we have

(A) $H_{0}$ , unbiased ECs   
![](images/847e0328601cae0a5f9f09a676917905e2a94d108729d9a5e55e166cf407b4f3.jpg)

<details>
<summary>bar</summary>

| Type I Error | Exact p-value |
|--------------|---------------|
| 0.048        | 0.00          |
| 0.038        | 0.01          |
</details>

(B) $H_{1}$ , unbiased ECs   
![](images/f04cb50b9b9697b98f48ef4219a4ec84b99229bd3c280c157456d295e740d81a.jpg)

<details>
<summary>histogram</summary>

| Exact p-value | Power |
| ------------- | ----- |
| 0.00          | 0.334 |
| 0.05          | 0.486 |
</details>

(C) $H_{0}$ , biased ECs   
![](images/ecdd55a1bf9eff889fd6e3f0da9524a55c291e15879a20122046a9414dc489b5.jpg)

<details>
<summary>histogram</summary>

| Exact p-value | Type I error |
| ------------- | ------------ |
| 0.00          | 0.048        |
| 0.25          | 0.056        |
</details>

(D) $H_{1}$ , biased ECs   
![](images/daf6c18fcf890aadd98c61ce05b6607cc666a40a6c5bab209415c73e3b2f9a55.jpg)

<details>
<summary>histogram</summary>

| Exact p-value | Power |
| ------------- | ----- |
| 0.00          | 0.334 |
| 0.05          | 0.186 |
</details>

Method □ No Borrow (γ=1) □ Full Borrow (γ=0)

Figure 1. Simulated distributions of $p$ -values under $H_0$ and $H_1$ .

$\mathbb{P}(p_{j}^{\text{split}} \leq \gamma) \leq \gamma.$ If $s_{j}$ and $\{s_{i}\}_{i \in \mathcal{C}}$ have distinct values, we have $\mathbb{P}(p_{j}^{\text{split}} \leq \gamma) = \left\{ \lfloor \gamma(|\mathcal{C}_{1}| + 1) \rfloor \right\} / (|\mathcal{C}_{1}| + 1) > \gamma - 1 / (|\mathcal{C}_{1}| + 1).$

Remark 3.2 (Definition of exchangeability). The random variables $z_{1}, \ldots, z_{n}$ are exchangeable if, for any permutation $\omega$ of $1, \ldots, n$ , the random variables $z_{\omega(1)}, \ldots, z_{\omega(n)}$ have the same joint distribution as $z_{1}, \ldots, z_{n}$ . The i.i.d. assumption is stronger than exchangeability, as the latter can hold with dependence (Shafer & Vovk, 2008). The exchangeability required by conformal inference is stronger than the mean exchangeability (Assumption 2.2), which allows the construction of a statistically valid estimator within the asymptotic inference framework.

CV+ p-value. While split conformal p-values are computationally efficient, they lose statistical efficiency due to data splitting. CV+ (Barber et al., 2021) fully utilize training data and remain computationally feasible. We randomly split C into K disjoint folds: $C = \cup_{k=1}^{K} C_k$ . We use the training set $C \setminus C_k$ to fit prediction models $\hat{f}_{-C_k}(x)$ and use the absolute residual as the score function: $s_i = |Y_i - \hat{f}_{-C_{k(i)}}(X_i)|$ and $s_j^{(i)} = |Y_j - \hat{f}_{-C_{k(i)}}(X_j)|$ for $i \in C$ , where $k(i) \in \{1, \ldots, K\}$ is a function that indicates $i \in C_k$ . Thus, for $i \neq i'$ and $k(i) = k(i')$ , we have $s_j^{(i)} = s_j^{(i')}$ . We define the CV+ p-value as the proportion of $\{s_i\}_{i \in C}$ that are larger than the corresponding $\{s_j^{(i)}\}_{i \in C}$ , that is, $p_j^{\text{cv+}} = \{\sum_{i \in C} \mathbb{I}(s_i \geq s_j^{(i)}) + 1\}/(|C| + 1)$ .

Proposition 3.3. For $j \in \mathcal{E}$ , suppose that $(X_j, Y_j)$ and $\{(X_i, Y_i)\}_{i \in \mathcal{C}}$ are exchangeable. For $\gamma \in (0,1)$ , we have $\mathbb{P}(p_j^{\mathrm{cv} + } \leq \gamma) \leq 2\gamma + \left\{(1 - 2\gamma)(m - 1) - 1\right\} / (|\mathcal{C}| + m) < 2\gamma + (1 - K / |\mathcal{C}|) / (K + 1)$ , where $m = |\mathcal{C}| / K$ is assumed to be an integer for simplicity.

# 3.3. Adaptive selection threshold

Since we construct $p_j^*$ individually and make borrowing decisions collectively, one might consider choosing a selection threshold $\gamma$ that controls the family-wise type I error rate or false discovery rate for testing the exchangeability of all ECs (Bates et al., 2023). However, in our context, the power of the conformal tests is of greater concern. The classical test-then-pool approach has been criticized for its low power in detecting hidden bias, especially with small randomized control sample sizes (Li et al., 2020). Even with effective control of the family-wise type I error rate, low-power conformal tests can allow many biased ECs to be incorrectly borrowed, increasing the MSE of $\hat{\tau}_{\gamma}$ and reducing the power of the FRT. Therefore, we propose a data-adaptive procedure to directly minimize the MSE of $\hat{\tau}_{\gamma}$ .

We decompose $\mathrm{MSE}(\gamma) \equiv \mathbb{E}(\hat{\tau}_{\gamma} - \tau)^{2} = \{\mathbb{E}(\hat{\tau}_{\gamma}) - \tau\}^{2} + \mathbb{V}(\hat{\tau}_{\gamma})$ . The main challenge lies in estimating the squared bias $\{\mathbb{E}(\hat{\tau}_{\gamma} - \tau)\}^{2}$ as the true $\tau$ is unknown. Fortunately, since the NB estimator $\hat{\tau}_{1}$ is consistent for $\tau$ , we approximate $\{\mathbb{E}(\hat{\tau}_{\gamma} - \tau)\}^{2}$ by $\{\mathbb{E}(\hat{\tau}_{\gamma} - \hat{\tau}_{1})\}^{2} = \mathbb{E}(\hat{\tau}_{\gamma} - \hat{\tau}_{1})^{2} - \mathbb{V}(\hat{\tau}_{\gamma} - \hat{\tau}_{1})$ . We then use $(\hat{\tau}_{\gamma} - \hat{\tau}_{1})^{2}$ to estimate $\mathbb{E}(\hat{\tau}_{\gamma} - \hat{\tau}_{1})^{2}$ and apply bootstrap to estimate $\mathbb{V}(\hat{\tau}_{\gamma})$ and $\mathbb{V}(\hat{\tau}_{\gamma} - \hat{\tau}_{1})$ . Combining these provides the estimated MSE for each $\gamma$ over finite grids, and we select the $\gamma$ that minimizes it. The complete procedure is detailed in Algorithm 1.

# Algorithm 1: Adaptive Selection Threshold

Input: Grid $\Gamma = \{0, 0.1, \ldots, 1\}$ ; bootstrap times $L$ .

for $\gamma \in \Gamma$ do

Compute $\hat{\tau}_{\gamma}$ from the original sample.

for $l = 1,\dots ,L$ do

Compute $\hat{\tau}_{\gamma}^{(l)}$ from the $l$ -th bootstrap sample.

for $\gamma \in \Gamma$ do

Compute $\widehat{\mathbb{V}} (\hat{\tau}_{\gamma} - \hat{\tau}_1)$ using $\hat{\tau}_{\gamma}^{(l)} - \hat{\tau}_1^{(l)}$

Compute $\widehat{\mathbb{V}} (\hat{\tau}_{\gamma})$ using $\hat{\tau}_{\gamma}^{(l)}$ .

$\widehat{\mathrm{MSE}} (\gamma) = (\hat{\tau}_{\gamma} - \hat{\tau}_{1})^{2} - \widehat{\mathbb{V}} (\hat{\tau}_{\gamma} - \hat{\tau}_{1}) + \widehat{\mathbb{V}} (\hat{\tau}_{\gamma}).$

Output: $\hat{\gamma} = \arg \min_{\gamma \in \Gamma}\widehat{\mathrm{MSE}} (\gamma)$

We theoretically analyze the procedure from a non-asymptotic perspective (Wainwright, 2019). Decomposing $\hat{\tau}_{\gamma} = \tau + \delta_{\gamma} + \epsilon_{\gamma}$ , where $\delta_{\gamma} \equiv \mathbb{E}(\hat{\tau}_{\gamma}) - \tau$ and $\mathbb{E}(\epsilon_{\gamma}) = 0$ . Let $\kappa_{\gamma}^{2} \equiv \mathbb{V}(\hat{\tau}_{\gamma} - \hat{\tau}_{1}) = \mathbb{V}(\epsilon_{\gamma} - \epsilon_{1})$ and $\sigma_{\gamma}^{2} \equiv \mathbb{V}(\hat{\tau}_{\gamma}) = \mathbb{V}(\epsilon_{\gamma})$ .

Theorem 3.4. For fixed n > 0 and $\gamma \in \Gamma$ , let $\epsilon_{\gamma}$ be a centered sub-Gaussian variable with parameter $\phi_{\gamma} > 0$ , i.e., $\mathbb{E}\exp(\lambda\epsilon_{\gamma}) \leq \exp(\phi_{\gamma}^{2}\lambda^{2}/2)$ for all $\lambda \in R$ . For $\iota > 0$ , there exists c > 0 such that with probability at least $1 - 4\iota$ :

$$
\begin{array}{l} \max _ {\gamma \in \Gamma} \left| \widehat {\mathrm{MSE}} (\gamma) - \mathrm{MSE} (\gamma) \right| \leq c \Delta | \delta_ {1} | + c \Delta \Phi \sqrt {\log (| \Gamma | / \iota)} \\ + \max \left\{c \Phi^ {2} \sqrt {\log (| \Gamma | / \iota)}, c \Phi^ {2} \log (| \Gamma | / \iota) \right\} \\ + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) - \kappa_ {\gamma} ^ {2} | + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - \sigma_ {\gamma} ^ {2} |, \\ \end{array}
$$

where $\Delta = \max_{\gamma \in \Gamma} |\delta_{\gamma}|$ , $\Phi = \max_{\gamma \in \Gamma} \phi_{\gamma}$ , $\delta_{1}$ is the bias of $\hat{\tau}_{1}$ , and $|\Gamma|$ is the cardinality of $\Gamma$ .

Theorem 3.4 shows that the discrepancy between the estimated and true MSE vanishes if $\Delta$ is bounded and the bias of the consistent estimator $\hat{\tau}_{1}$ , the maximum standard deviation proxy $\Phi$ , and the variance estimation errors are sufficiently small.

Theorem 3.5. Under the same assumptions as in Theorem 3.4, for any $\iota > 0$ , there exists a constant $c > 0$ such that, with probability at least $1 - 8\iota$ , the following holds:

$$
\begin{array}{l} (\hat {\tau} _ {\hat {\gamma}} - \tau) ^ {2} - \min _ {\gamma \in \Gamma} (\hat {\tau} _ {\gamma} - \tau) ^ {2} \leq 2 c \Delta | \delta_ {1} | + 2 c \Delta \Phi \sqrt {\log (| \Gamma | / \iota)} \\ + 2 \max \left\{c \Phi^ {2} \sqrt {\log \left(| \Gamma | / \iota\right)}, c \Phi^ {2} \log \left(| \Gamma | / \iota\right) \right\} \\ + 2 \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) - \kappa_ {\gamma} ^ {2} | + 2 \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - \sigma_ {\gamma} ^ {2} |. \\ \end{array}
$$

Theorem 3.5 provides a bound for the excess risk of $\hat{\tau}_{\hat{\gamma}}$ in comparison to the oracle estimator. Although $\hat{\tau}_{\hat{\gamma}}$ generally outperforms $\hat{\tau}_1$ in terms of MSE, it may exhibit excess risk in certain challenging cases, as shown in Figure 2 (C) in the simulation. This phenomenon highlights that $\hat{\tau}_{\hat{\gamma}}$ behaves similarly to the Hodges estimator (Le Cam, 1953) and to integrated estimators in data fusion (Yang et al., 2023; Oberst et al., 2022): improving upon the baseline estimator (here, the No Borrow estimator) in certain regions of the parameter space (where there is no bias in ECs) inevitably leads to worse performance in other regions (where the bias in ECs is difficult to detect). FRT still controls the type I error rate even if excess risk is present or the assumptions in Theorem 3.4 are not satisfied.

# 4. Simulation

We conduct simulations to evaluate the repeated sampling performance of the proposed methods under small sample sizes and varying magnitudes of hidden bias, including challenging cases where separating biased ECs is difficult. Specifically, the sample sizes for the randomized treatment, randomized control, and EC groups are set as $(n_{1}, n_{0}, n_{\mathcal{E}}) = (50, 25, 50)$ . Similar results for a larger EC sample size $(n_{\mathcal{E}} = 300)$ are included in the Appendix. We generate covariates $X \sim \text{Unif}(-2, 2)$ with dimension p = 2. The sampling indicator $S \sim \text{Bernoulli}(\pi(X))$ is generated with $\pi(X) = \{1 + \exp(\eta_{0} + X^{\mathrm{T}}\eta)\}^{-1}$ , where $\eta_{0}$ is chosen to ensure $\mathbb{E}(S) = n_{\mathcal{R}}/n$ , and $\eta = (0.1, 0.1)$ . The assignment is generated by $A \sim \text{Bernoulli}(n_{1}/n_{\mathcal{R}})$ for S = 1 and A = 0 for S = 0. Let $\varepsilon \sim N(0, 1)$ denote the noise. For the RCT sample (S = 1), we generate the potential outcomes as $Y(0) = X^{\mathrm{T}}\beta_{0} + \varepsilon$ with $\beta_{0} = (1, 1)$ , and $Y(1) = \tau_{0} + X^{\mathrm{T}}\beta_{1} + \varepsilon$ with $\tau_{0} = 0.4$ and $\beta_{1} = (2, 2)$ . For the EC sample (S = 0), we consider two scenarios: (i) the scenario without hidden bias, where $Y(0) = X^{\mathrm{T}}\beta_{0} + 0.5\varepsilon$ ; (ii) the scenario where part of the ECs have hidden bias b, where a random proportion $\rho$ of the ECs is biased, with $Y(0) = -b + X^{\mathrm{T}}\beta_{0} + 0.5\varepsilon$ , and the remaining proportion $(1 - \rho)$ are unbiased, with $Y(0) = X^{\mathrm{T}}\beta_{0} + 0.5\varepsilon$ . We consider proportions of biased ECs $\rho = 50\%$ and magnitudes of hidden bias b = 1, 2, …, 8. Note that hidden bias refers to bias that remains due to unmeasured confounders, even after balancing the observed covariates. Under the alternative hypothesis, the observed outcome is $Y = AY(1) + (1 - A)Y(0)$ ; under the sharp null hypothesis, the observed outcome is $Y = Y(0)$ . We consider NB, FB, and CSB with the adaptive selection threshold as estimators of $\tau$ and test statistics for FRT. We also consider Adaptive Lasso Selective Borrowing (ALSB) by Gao et al. (2025). Given its higher computational cost (approximately 10 times slower than CSB), we omit FRTs for this method and instead compare CSB+FRT with ALSB+asymptotic inference in the Appendix. CV+p-values are used with 10 folds. We set B = 5000 to approximate $p^{FRT}$ and replicate the simulation 500 times per scenario.

Figure 2 displays performance metrics for $b = 0,1,\dots ,8$ . In the first case ( $b = 0$ ): (i) all methods exhibit negligible bias; (ii) FB and CSB reduce MSE by $42\%$ and $20\%$ , respectively, compared to NB; (iii) all methods effectively control the type I error rate; and (iv) FB and CSB increase power by $46\%$ and $45\%$ , respectively, compared to NB. In the following eight cases ( $b = 1,\dots ,8$ ): (i) FB exhibits a large bias, approximately $125\% -203\%$ of its standard deviation (SD). The absolute bias of FB decreases with $b$ when $b \geq 3$ because large $b$ values increase $\mathbb{V}\{Y(0)\mid X = x,A = 0,S = 0\}$ , causing FB to downweight ECs with small $\hat{r}(X_i)$ in (1). CSB performs better at bias control, with bias ranging from $0\% -22\%$ of its SD; (ii) compared to NB, FB increases MSE by up to $454\%$ , and CSB decreases MSE by $13\% -16\%$ (except when $b = 1,2$ , where MSE increases by $1\% -18\%$ ). (iii) In line with Theorem 2.3, all methods control the type I error rate well; (iv) compared to NB, FB decreases power by up to $51\%$ . In contrast, CSB increases power by $13\% -36\%$ (except when $b = 2,3,4$ , where

![](images/b47038253b6af1f98ea8531a2ae20740b4af9e9c2ded756a99f6b03639e27eb1.jpg)  
Method • No Borrow (γ=1) • Full Borrow (γ=0) • Conformal Selective Borrow (γ)

Figure 2. Simulation results across different hidden bias magnitudes b.

power decreases by 7%-20%). In challenging cases where $0 < b \leq 4$ , the efficiency loss of CSB occurs because small biases make it hard to distinguish biased ECs from unbiased ECs. Such loss is inevitable when aiming to gain efficiency in scenarios without hidden bias, a phenomenon known in the transfer learning literature as the cost of transferability detection (Cai et al., 2024). This phenomenon also occurs for other data integration estimators under hidden bias (see Figure 2 in Yang et al. (2023), Figure 4 in Oberst et al. (2022), and Figure 2 in Lin et al. (2024)). Finally, we examine the selection performance of CSB. We do not expect CSB to perfectly separate biased ECs from unbiased ones due to (i) the small sample size of randomized controls and (ii) finite sample noise. As shown in Figure 2, CSB discards biased ECs and some unbiased ones that aren’t sufficiently similar to randomized controls, demonstrating satisfactory selection performance.

# 5. Real data application

The CALGB 9633 and NCDB data. We apply the proposed methods to an RCT conducted by the Cancer and Leukemia Group B (CALGB), known as CALGB 9633, which investigated the treatment effect of adjuvant chemotherapy in patients with stage IB non-small-cell lung cancer (Strauss et al., 2008). In CALGB 9633 (S = 1), $n_{1} = 167$ patients were randomized to adjuvant chemotherapy (A = 1), and $n_{0} = 168$ were randomized to observation (A = 0). We extract data for 11,700 patients from the National Cancer Database (NCDB) as the EC sample

$(A = 0, S = 0)$ to improve CALGB 9633's statistical efficiency. The NCDB is a clinical oncology database sourced from hospital registry data, jointly run by the American Cancer Society and the American College of Surgeons, covering 70% of U.S. cancer cases.

RMST and pseudo-observations We use the Restricted Mean Survival Time (RMST), $Y = \min(T, t^{*})$ , as the primary endpoint, where T represents the survival time and $t^{*}$ is the truncation time. RMST measures survival time up to a clinically relevant truncation point and serves as a compelling alternative to the hazard ratio when the proportional hazards assumption is violated (Hernán, 2010). We consider the difference in 3-year RMST between the treatment and control groups for the RCT population $\tau = E\{Y(1) - Y(0) \mid S = 1\}$ as the estimand, where $Y(a) = \min\{T(a), 3\}$ and $T(a)$ is the potential survival time, a = 0, 1. Five baseline covariates in CALGB 9633 and NCDB are considered: sex, age, race, histology, and tumor size. The censoring rates of T in CALGB 9633 and NCDB are 42% and 48%, respectively. We use a “once-for-all” approach to transform right-censored survival times into pseudo-observations for RMST, allowing standard causal inference methods as if outcomes were non-censored (Andersen et al., 2003; Overgaard et al., 2017). To address covariate-dependent censoring, we stratified by sex, race, and histology, applying transformations separately within each dataset (Andersen & Pohar Perme, 2010). The stratified Kaplan–Meier estimator is used to estimate survival functions, with pseudo-observations generated via the jack-

Table 1. Analysis results for CALGB 9633 + NCDB. 

<table><tr><td>Method</td><td>Est</td><td>SE</td><td>CI</td><td>Asym p</td><td>Exact p</td><td>#EC</td></tr><tr><td>No Borrow (Dif-in-Means)</td><td>0.135</td><td>0.072</td><td>(-0.007, 0.276)</td><td>0.062</td><td>0.060</td><td>0</td></tr><tr><td>No Borrow (AIPW)</td><td>0.142</td><td>0.074</td><td>(-0.003, 0.286)</td><td>0.055</td><td>0.051</td><td>0</td></tr><tr><td>Full Borrow</td><td>0.241</td><td>0.061</td><td>(0.122, 0.361)</td><td>&lt;0.001</td><td>0.031</td><td>335</td></tr><tr><td>Conformal Selective Borrow</td><td>0.138</td><td>0.058</td><td>(0.024, 0.252)</td><td>0.018</td><td>0.046</td><td>264</td></tr></table>

“Est” is the estimate. “SE”, “CI”, and “Asym p” are the asymptotic standard error, confidence interval, and p-value, respectively. “Exact p” is the exact p-value. “#EC” is the number of borrowed ECs.

knife method, as implemented in the R package eventglm (Sachs & Gabriel, 2022). We treat the pseudo-observations for 3-year RMST as the outcome hereafter. More details about the real data are provided in Appendix Section D.

Data analysis. We apply NB, FB, and CSB to estimate the ATE and perform FRTs. For comparison, we also apply NB without covariate adjustment, i.e., difference-in-means estimator. In addition to the proposed exact p-value, we also compute the standard error, confidence interval, and p-value based on asymptotic inference for all approaches (Li et al., 2023b). Since the outcome shows a high proportion of truncation at 3 years, resulting in a highly skewed distribution, we apply the conformal quantile regression (Romano et al., 2019) to compute the conformal score. We use the Jackknife+ p-value (Barber et al., 2021) to achieve a better balance between statistical and computational efficiency. Table 1 presents the analysis results. For NB using Dif-in-Means and AIPW, asymptotic and exact p-values range from 0.051 to 0.062. In contrast, FB (using all 335 ECs) gives an asymptotic p-value of < 0.001 and an exact p-value of 0.031, indicating a significantly positive ATE. Similarly, CSB (using 178 ECs) shows an asymptotic p-value of 0.018 and an exact p-value of 0.046, also indicating a significantly positive ATE. The ATE estimate from CSB falls between NB and FB, indicating a trade-off between these two approaches.

# 6. Discussion

This paper proposes using FRT in hybrid controlled trials and introduces CSB for selectively incorporating comparable ECs, mitigating hidden bias. FRT with CSB maintains type I error control and improves power compared to RCT-only analysis. The proposed CSB estimator with an adaptive selection threshold enhances efficiency over the NB approach.

One limitation of our procedure is that, when the bias is non-negligible yet difficult to detect, it may incur some power loss, though it still maintains valid Type I error control. This no-free-lunch limitation is acknowledged in existing papers (Oberst et al., 2022; Lin et al., 2024), which point out that without assuming mean exchangeability of ECs, no method can uniformly and significantly outperform RCT-only analysis across varying levels of hidden bias, although different approaches optimize the risk-reward trade-off from different perspectives. The most challenging scenarios are those where bias is non-negligible but complex to correct or difficult to detect. Our key distinctions from existing literature are twofold: (i) we prioritize exact Type I error control in small samples before seeking power gains; (ii) we optimize the risk-reward trade-off between no borrowing and full borrowing through conformal selective borrowing, motivated by real data in which some ECs are unbiased while others are not.

Heterogeneity among data sources is common in integration and transfer learning, often leading to bias or efficiency loss even after balancing measured confounders. While penalized bias estimation is a common solution, our work demonstrates that conformal inference provides greater stability and flexibility in finite samples. Extending this approach to tasks like developing individual treatment regimes (Chu et al., 2023), exploring treatment effect heterogeneity (Wu & Yang, 2022), and improving experimental design (Ruan et al., 2024) shows great potential.

Beyond the sharp null, FRTs can test the weak null asymptotically using studentized or prepivoted statistics (Wu & Ding, 2021; Cohen & Fogarty, 2022). Randomization-based confidence intervals can be constructed by inverting FRTs (Luo et al., 2021; Zhu & Liu, 2023; Fiksel, 2024), and randomization inference can test bounded nulls and construct confidence intervals for treatment effect quantiles (Caughey et al., 2023). Extending these methods to hybrid controlled trials would be valuable.

# Software and Data

A user-friendly R package, intFRT, is available at: https://github.com/ke-zhu/intFRT.

# Acknowledgment

We thank the anonymous reviewers and meta-reviewers of ICML 2025 for their helpful comments, which significantly

improved the manuscript. This project is supported by the Food and Drug Administration (FDA) of the U.S. Department of Health and Human Services (HHS) as part of a financial assistance award U01FD007934 totaling \$1,674,013 over two years funded by FDA/HHS. It is also supported by the National Institute On Aging of the National Institutes of Health under Award Number R01AG06688, totaling \$1,565,763 over four years. The contents are those of the authors and do not necessarily represent the official views of, nor an endorsement by, FDA/HHS, the National Institutes of Health, or the U.S. Government.

# Impact Statement

This paper presents work aimed at advancing data integration, conformal inference, and their applications in biomedical science. The potential societal impact of this research is substantial, including fostering the reliable and efficient use of real-world data, accelerating drug development processes, improving the understanding of rare diseases, and ultimately enhancing patient outcomes.

# References

Alt, E. M., Chang, X., Jiang, X., Liu, Q., Mo, M., Xia, H. A., and Ibrahim, J. G. LEAP: The latent exchangeability prior for borrowing information from historical data. Biometrics, 80(3):ujae083, 2024.   
Andersen, P. K. and Pohar Perme, M. Pseudo-observations in survival analysis. Statistical Methods in Medical Research, 19(1):71–99, 2010.   
Andersen, P. K., Klein, J. P., and Rosthøj, S. Generalised linear models for correlated pseudo-observations, with applications to multi-state models. Biometrika, 90(1):15–27, 2003.   
Angelopoulos, A. N. and Bates, S. Conformal prediction: A gentle introduction. Foundations and Trends® in Machine Learning, 16(4):494–591, 2023.   
Barber, R. F., Candes, E. J., Ramdas, A., and Tibshirani, R. J. Predictive inference with the jackknife+. The Annals of Statistics, 49(1):486–507, 2021.   
Bates, S., Candès, E., Lei, L., Romano, Y., and Sesia, M. Testing for outliers with conformal p-values. The Annals of Statistics, 51(1):149–178, 2023.   
Bind, M.-A. C. and Rubin, D. B. When possible, report a Fisher-exact P value and display its underlying null randomization distribution. Proceedings of the National Academy of Sciences, 117(32):19151–19158, 2020.

Cai, T., Li, M., and Liu, M. Semi-supervised triply robust inductive transfer learning. Journal of the American Statistical Association, in press, 2024.

Carter, K., Scheffold, A. L., Renteria, J., Berger, V. W., Luo, Y. A., Chipman, J. J., and Sverdlov, O. Regulatory guidance on randomization and the use of randomization tests in clinical trials: A systematic review. Statistics in Biopharmaceutical Research, 16(4):428–440, 2023.

Caughey, D., Dafoe, A., Li, X., and Miratrix, L. Randomisation inference beyond the sharp null: Bounded null hypotheses and quantiles of individual treatment effects. Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(5):1471–1491, 2023.

Chen, C., Wang, M., and Chen, S. An efficient data integration scheme for synthesizing information from multiple secondary datasets for the parameter inference of the main analysis. Biometrics, 79(4):2947–2960, 2023.

Chen, S., Zhang, B., and Ye, T. Minimax rates and adaptivity in combining experimental and observational data. arXiv preprint arXiv:2109.10522, 2021a.

Chen, W.-C., Wang, C., Li, H., Lu, N., Tiwari, R., Xu, Y., and Yue, L. Q. Propensity score-integrated composite likelihood approach for augmenting the control arm of a randomized controlled trial by incorporating real-world data. Journal of Biopharmaceutical Statistics, 30(3):508–520, 2020.

Chen, Z., Ning, J., Shen, Y., and Qin, J. Combining primary cohort data with external aggregate information without assuming comparability. Biometrics, 77(3):1024–1036, 2021b.

Cheng, D. and Cai, T. Adaptive combination of randomized and observational data. arXiv preprint arXiv:2111.15012, 2021.

Cheng, Y., Wu, L., and Yang, S. Enhancing treatment effect estimation: A model robust approach integrating randomized experiments and external controls using the double penalty integration estimator. In Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, volume 216 of Proceedings of Machine Learning Research, pp. 381–390, 2023.

Chernozhukov, V., Wüthrich, K., and Zhu, Y. An exact and robust conformal inference method for counterfactual and synthetic controls. Journal of the American Statistical Association, 116(536):1849–1864, 2021.

Chu, J., Lu, W., and Yang, S. Targeted optimal treatment regime learning using summary statistics. Biometrika, 110(4):913–931, 2023.

Cohen, P. L. and Fogarty, C. B. Gaussian prepivoting for finite population causal inference. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(2):295–320, 2022.   
Colnet, B., Mayer, I., Chen, G., Dieng, A., Li, R., Varo-quaux, G., Vert, J.-P., Josse, J., and Yang, S. Causal inference methods for combining randomized trials and observational studies: A review. Statistical Science, 39(1):165–191, 2024.   
Dang, L. E., Tarp, J. M., Abrahamsen, T. J., Kvist, K., Buse, J. B., Petersen, M., and van der Laan, M. A cross-validated targeted maximum likelihood estimator for data-adaptive experiment selection applied to the augmentation of RCT control arms with external data. arXiv preprint arXiv:2210.05802v3, 2023.   
European Medicines Agency. Guideline on adjustment for baseline covariates in clinical trials. https://www.ema.europa.eu/en/adjustment-baseline-covariates-clinical2015.   
Fiksel, J. On exact randomization-based covariate-adjusted confidence intervals. Biometrics, 80(2):ujae051, 2024.   
Fisher, R. A. The Design of Experiments. Oliver and Boyd, Edinburgh, 1st edition, 1935.   
Freidling, T., Zhao, Q., and Gao, Z. Selective randomization inference for adaptive experiments. arXiv preprint arXiv:2405.07026, 2024.   
Gagnon-Bartsch, J. A., Sales, A. C., Wu, E., Botelho, A. F., Erickson, J. A., Miratrix, L. W., and Heffernan, N. T. Precise unbiased estimation in randomized experiments using auxiliary observational data. Journal of Causal Inference, 11(1):20220011, 2023.   
Gao, C. and Yang, S. Pretest estimation in combining probability and non-probability samples. Electronic Journal of Statistics, 17(1):1492–1546, 2023.   
Gao, C., Yang, S., Shan, M., Ye, W., Lipkovich, I., and Faries, D. Improving randomized controlled trial analysis via data-adaptive borrowing. Biometrika, 112(2):asae069, 2025.   
Gu, Y., Liu, H., and Ma, W. Incorporating external data for analyzing randomized clinical trials: A transfer learning approach. arXiv preprint arXiv:2409.04126, 2024.   
Guan, L. and Tibshirani, R. Prediction and outlier detection in classification problems. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(2):524–546, 2022.

Guo, W., Wang, S. L., Ding, P., Wang, Y., and Jordan, M. Multi-source causal inference using control variates under outcome selection bias. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=CrimIjBa64.   
Hernán, M. A. The hazards of hazard ratios. Epidemiology, 21(1):13–15, 2010.   
Ho, D., Imai, K., King, G., and Stuart, E. A. Matchit: Nonparametric preprocessing for parametric causal inference. Journal of Statistical Software, 42(8):1–28, 2011.   
Ho, D. E., Imai, K., King, G., and Stuart, E. A. Matching as nonparametric preprocessing for reducing model dependence in parametric causal inference. Political Analysis, 15(3):199–236, 2007.   
Hobbs, B. P., Carlin, B. P., Mandrekar, S. J., and Sargent, D. J. Hierarchical commensurate and power prior models for adaptive incorporation of historical information in clinical trials. Biometrics, 67(3):1047–1056, 2011. trials-scientific-guideline,   
Huang, Y., Huang, C.-Y., and Kim, M.-O. Simultaneous selection and incorporation of consistent external aggregate information. Statistics in Medicine, 42(30):5630–5645, 2023.   
Imbens, G. W. and Rubin, D. B. Causal Inference in Statistics, Social, and Biomedical Sciences. Cambridge University Press, 2015.   
Jiang, L., Nie, L., and Yuan, Y. Elastic priors to dynamically borrow information from historical data in clinical trials. Biometrics, 79(1):49–60, 2023.   
Karlsson, R., Wang, G., Krijthe, J. H., and Dahabreh, I. J. Robust integration of external control data in randomized trials. arXiv preprint arXiv:2406.17971, 2024.   
Kwiatkowski, E., Zhu, J., Li, X., Pang, H., Lieberman, G., and Psioda, M. A. Case weighted power priors for hybrid control analyses with time-to-event data. Biometrics, 80(2):ujae019, 2024.   
Le Cam, L. On some asymptotic properties of maximum likelihood estimates and related Bayes estimates. University of California Publications in Statistics, 1:277–330, 1953.   
Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523):1094–1111, 2018.   
Lei, L. and Candès, E. J. Conformal inference of counterfactuals and individual treatment effects. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(5):911–938, 2021.

Li, H., Tiwari, R., and Li, Q. H. Conditional borrowing external data to establish a hybrid control arm in randomized clinical trials. Journal of Biopharmaceutical Statistics, 32(6):954–968, 2022.   
Li, L. and Jemielita, T. Confounding adjustment in the analysis of augmented randomized controlled trial with hybrid control arm. Statistics in Medicine, 42(16):2855–2872, 2023.   
Li, R., Lin, R., Huang, J., Tian, L., and Zhu, J. A frequentist approach to dynamic borrowing. Biometrical Journal, 65(7):2100406, 2023a.   
Li, S. and Luedtke, A. Efficient estimation under data fusion. Biometrika, 110(4):1041–1054, 2023.   
Li, W., Liu, F., and Snavely, D. Revisit of test-then-pool methods and some practical considerations. Pharmaceutical Statistics, 19(5):498–517, 2020.   
Li, X., Miao, W., Lu, F., and Zhou, X.-H. Improving efficiency of inference in clinical trials with external control data. Biometrics, 79(1):394–403, 2023b.   
Liang, Z., Sesia, M., and Sun, W. Integrative conformal p-values for out-of-distribution testing with labelled outliers. Journal of the Royal Statistical Society Series B: Statistical Methodology, pp. qkad138, 2024.   
Lin, X., Tarp, J. M., and Evans, R. J. Data fusion for efficiency gain in ATE estimation: A practical review with simulations. arXiv preprint arXiv:2407.01186, 2024.   
Lin, X., Tarp, J., and Evans, R. Combining experimental and observational data through a power likelihood. Biometrics, 2025.   
Liu, Y., Lu, B., Foster, R., Zhang, Y., Zhong, Z. J., Chen, M.-H., and Sun, P. Matching design for augmenting the control arm of a randomized controlled trial using real-world data. Journal of Biopharmaceutical Statistics, 32(1):124–140, 2022.   
Luo, X., Dasgupta, T., Xie, M., and Liu, R. Y. Leveraging the Fisher randomization test using confidence distributions: Inference, combination and fusion learning. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(4):777–797, 2021.   
Miller, F. and Joffe, S. Equipoise and the dilemma of randomized clinical trials. The New England Journal of Medicine, 364(5):476–480, 2011.   
Nair, Y. and Janson, L. Randomization tests for adaptively collected data. arXiv preprint arXiv:2301.05365, 2023.

Oberst, M., D'Amour, A., Chen, M., Wang, Y., Sontag, D., and Yadlowsky, S. Understanding the risks and rewards of combining unbiased and possibly biased estimators, with applications to causal inference. arXiv preprint arXiv:2205.10467, 2022.   
Overgaard, M., Parner, E. T., and Pedersen, J. Asymptotic theory of generalized estimating equations based on jack-knife pseudo-observations. The Annals of Statistics, 45(5):1988–2015, 2017.   
Papadopoulos, H., Proedrou, K., Vovk, V., and Gammerman, A. Inductive confidence machines for regression. In Machine learning: ECML 2002: 13th European conference on machine learning Helsinki, Finland, August 19–23, 2002 proceedings 13, pp. 345–356. Springer, 2002.   
Plamadeala, V. and Rosenberger, W. F. Sequential monitoring with conditional randomization tests. The Annals of Statistics, 40(1):30–44, 2012.   
Pocock, S. J. The combination of randomized and historical controls in clinical trials. Journal of Chronic Diseases, 29(3):175–188, 1976.   
Proschan, M. A. and Dodd, L. E. Re-randomization tests in clinical trials. Statistics in Medicine, 38(12):2292–2302, 2019.   
Puelz, D., Basse, G., Feller, A., and Toulis, P. A graph-theoretic approach to randomization tests of causal effects under general interference. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(1):174–204, 2022.   
Rabideau, D. J. and Wang, R. Randomization-based confidence intervals for cluster randomized trials. Biostatistics, 22(4):913–927, 2021.   
Ritzwoller, D. M., Romano, J. P., and Shaikh, A. M. Randomization inference: Theory and applications. arXiv preprint arXiv:2406.09521, 2024.   
Romano, Y., Patterson, E., and Candès, E. J. Conformalized quantile regression. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 3543–3553, 2019.   
Rosenberger, W. F. and Lachin, J. M. Randomization in Clinical Trials: Theory and Practice. John Wiley & Sons, 2015.   
Rosenberger, W. F., Uschner, D., and Wang, Y. Randomization: The forgotten component of the randomized clinical trial. Statistics in Medicine, 38(1):1–12, 2019.   
Rosenman, E. T., Basse, G., Owen, A. B., and Baiocchi, M. Combining observational and experimental datasets

using shrinkage estimators. Biometrics, 79(4):2961–2973, 2023.   
Rothenhäusler, D. Model selection for estimation of causal parameters. arXiv preprint arXiv:2008.12892, 2020.   
Rothenhäusler, D., Meinshausen, N., Bühlmann, P., and Peters, J. Anchor regression: Heterogeneous data meet causality. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(2):215–246, 2021.   
Ruan, X., Wang, J., Wang, Y., and Wei, W. Electronic medical records assisted digital clinical trial design. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pp. 2836–2844, 2024.   
Sachs, M. C. and Gabriel, E. E. Event history regression with pseudo-observations: computational approaches and an implementation in R. Journal of Statistical Software, 102:1–34, 2022.   
Schmidli, H., Gsteiger, S., Roychoudhury, S., O'Hagan, A., Spiegelhalter, D., and Neuenschwander, B. Robust meta-analytic-predictive priors in clinical trials with historical control information. Biometrics, 70(4):1023–1032, 2014.   
Schuler, A., Walsh, D., Hall, D., Walsh, J., Fisher, C., for Alzheimer's Disease, C. P., Initiative, A. D. N., and Study, A. D. C. Increasing the efficiency of randomized trial estimates via linear adjustment for a prognostic score. The International Journal of Biostatistics, 18(2):329–356, 2022.   
Shafer, G. and Vovk, V. A tutorial on conformal prediction. Journal of Machine Learning Research, 9(3):371–421, 2008.   
Simon, R. and Simon, N. R. Using randomization tests to preserve type I error with response adaptive and covariate adaptive randomization. Statistics & Probability Letters, 81(7):767–772, 2011.   
Strauss, G. M., Herndon, J. E., Maddaus, M. A., Johnstone, D. W., Johnson, E. A., Harpole, D. H., Gillenwater, H. H., Watson, D. M., Sugarbaker, D. J., Schilsky, R. L., et al. Adjuvant paclitaxel plus carboplatin compared with observation in stage IB non-small-cell lung cancer: CALGB 9633 with the cancer and leukemia group B, radiation therapy oncology group, and north central cancer treatment group study groups. Journal of Clinical Oncology, 26(31):5043–5051, 2008.   
Stuart, E. A. and Rubin, D. B. Matching with multiple control groups with adjustment for group differences. Journal of Educational and Behavioral Statistics, 33(3):279–306, 2008.

U.S. Food and Drug Administration. Adaptive design clinical trials for drugs and biologics guidance for industry. https://www.fda.gov/media/78495/download, 2019.   
U.S. Food and Drug Administration. Rare Diseases at FDA. https://www.fda.gov/patients/rare-diseases-fda, 2022.   
U.S. Food and Drug Administration. Considerations for the design and conduct of externally controlled trials for drug and biological products guidance for industry. https://www.fda.gov/media/164960/download, 2023.   
Valancius, M., Pang, H., Zhu, J., Cole, S. R., Funk, M. J., and Kosorok, M. R. A causal inference framework for leveraging external controls in hybrid trials. Biometrics, 80(4):ujae095, 2024.   
van der Laan, M., Qiu, S., and van der Laan, L. Adaptive-TMLE for the average treatment effect based on randomized controlled trial augmented with real-world data. arXiv preprint arXiv:2405.07186, 2024.   
Ventz, S., Khozin, S., Louv, B., Sands, J., Wen, P. Y., Rahman, R., Comment, L., Alexander, B. M., and Trippa, L. The design and evaluation of hybrid controlled trials that leverage external data and randomization. Nature Communications, 13(1):5783, 2022.   
Viele, K., Berry, S., Neuenschwander, B., Amzal, B., Chen, F., Enas, N., Hobbs, B., Ibrahim, J. G., Kinnersley, N., Lindborg, S., et al. Use of historical control data for assessing treatment effects in clinical trials. Pharmaceutical Statistics, 13(1):41–54, 2014.   
Vovk, V., Gammerman, A., and Shafer, G. Algorithmic Learning in a Random World, volume 29. Springer, 2005.   
Wainwright, M. J. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge University Press, 2019.   
Wu, J. and Ding, P. Randomization tests for weak null hypotheses in randomized experiments. Journal of the American Statistical Association, 116(536):1898–1913, 2021.   
Wu, L. and Yang, S. Integrative R-learner of heterogeneous treatment effects combining experimental and observational studies. In Proceedings of the First Conference on Causal Learning and Reasoning, volume 177 of Proceedings of Machine Learning Research, pp. 904–926, 2022.   
Yang, S. and Ding, P. Combining multiple observational data sources to estimate causal effects. Journal of the

American Statistical Association, 115(531):1540–1554, 2020.   
Yang, S., Gao, C., Zeng, D., and Wang, X. Elastic integrative analysis of randomised trial and real-world data for treatment heterogeneity estimation. Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(3):575–596, 2023.   
Yang, S., Liu, S., Zeng, D., and Wang, X. Data fusion methods for the heterogeneity of treatment effect and confounding function. Bernoulli, in press, 2024.   
Yi, Y., Zhang, Y., Du, Y., and Ye, T. Testing for treatment effect twice using internal and external controls in clinical trials. Journal of Causal Inference, 11(1):20220018, 2023.   
Young, A. Channeling Fisher: Randomization tests and the statistical insignificance of seemingly significant experimental results. The Quarterly Journal of Economics, 134(2):557–598, 2019.   
Yuan, J., Liu, J., Zhu, R., Lu, Y., and Palm, U. Design of randomized controlled confirmatory trials using historical control data to augment sample size for concurrent controls. Journal of biopharmaceutical statistics, 29(3):558–573, 2019.   
Zhai, Y. and Han, P. Data integration with oracle use of external information from heterogeneous populations. Journal of Computational and Graphical Statistics, 31(4):1001–1012, 2022.   
Zhang, Y. and Zhao, Q. What is a randomization test? Journal of the American Statistical Association, 118(544):2928–2942, 2023.   
Zhu, K. and Liu, H. Pair-switching rerandomization. Biometrics, 79(3):2127–2142, 2023.

# A. Additional conformal $p$ -values

Full conformal $p$ -value. Full conformal inference (Vovk et al., 2005) fully utilizes all data in $\mathcal{C}$ for both training and calibration. We can still use the absolute residual as the score function: $s_i = |Y_i - \hat{f}_j(X_i)|$ for $i \in \mathcal{C}$ and $s_j = |Y_j - \hat{f}_j(X_j)|$ , where $\hat{f}_j(x)$ is a prediction model fitted by the augmented set $\mathcal{C} \cup \{j\}$ . To measure the extremeness of observing $s_j$ under the exchangeability, we define the full conformal $p$ -value as the proportion of the elements in $\{s_i\}_{i \in \mathcal{C}}$ that are larger than or equal to $s_j$ , that is, $p_j^{\mathrm{full}} = \{\sum_{i \in \mathcal{C}} \mathbb{I}(s_i \geq s_j) + 1\} / (|\mathcal{C}| + 1)$ .

Proposition A.1. For $j \in E$ , suppose that $(X_{j}, Y_{j})$ and $\{(X_{i}, Y_{i})\}_{i \in C}$ are exchangeable. For $\gamma \in (0, 1)$ , we have $\mathbb{P}(p_{j}^{\mathrm{full}} \leq \gamma) \leq \gamma$ . If $s_{j}$ and $\{s_{i}\}_{i \in C}$ have distinct values, we have $\mathbb{P}(p_{j}^{\mathrm{full}} \leq \gamma) = \left\lfloor \gamma(|C| + 1) \right\rfloor / (|C| + 1) > \gamma - 1 / (|C| + 1)$ .

To compute full conformal $p$ -values for all ECs $j \in \mathcal{E}$ , the prediction model must be refit $n_{\mathcal{E}}$ times, which is time-consuming for large EC samples.

Jackknife+ p-value. Jackknife+ p-values (Barber et al., 2021) is a special case of CV+ with $K = |C|$ . We use the leave-one-out training set $C \setminus \{i\}$ to fit prediction models $\hat{f}_{-i}(x)$ and use the absolute residual as the score function: $s_i = |Y_i - \hat{f}_{-i}(X_i)|$ and $s_j^{(i)} = |Y_j - \hat{f}_{-i}(X_j)|$ for $i \in C$ . We define the Jackknife+ p-value as the proportion of $\{s_i\}_{i \in C}$ that are larger than the corresponding $\{s_j^{(i)}\}_{i \in C}$ , that is, $p_j^{\text{jackknife+}} = \{\sum_{i \in C} \mathbb{I}(s_i \geq s_j^{(i)}) + 1\}/(|C| + 1)$ .

Proposition A.2. For $j \in \mathcal{E}$ , suppose that $(X_j, Y_j)$ and $\{(X_i, Y_i)\}_{i \in \mathcal{C}}$ are exchangeable. For $\gamma \in (0,1)$ , we have $\mathbb{P}(p_j^{\text{jackknife} +} \leq \gamma) \leq 2\gamma - 1 / (|\mathcal{C}| + 1) < 2\gamma$ .

Remark A.3. The factor of 2 cannot be reduced without further assumptions, as shown by pathological cases in Barber et al. (2021), though the empirical error rate is close to $\gamma$ .

# B. Proofs

# B.1. Proof of Theorem 2.3

Proof of Theorem 2.3. Under $H_0$ , the imputed potential outcomes are the same as the true potential outcomes. Thus, the distribution of $T^* \equiv T(\mathbf{A}^*)$ is the same as that of $T \equiv T(\mathbf{A})$ . With simplified notations, we have

$$
\mathbb {P} _ {\boldsymbol {A}} (p ^ {\mathrm{FRT}} \leq \alpha) = \mathbb {P} _ {\boldsymbol {A}} \left\{\mathbb {P} _ {\boldsymbol {A} ^ {*}} \left(T ^ {*} \geq T\right) \leq \alpha \right\}.
$$

In a finite sample, $A$ can take only a finite set of values, which implies that $T$ must also take on a finite set of values. Suppose these values are

$$
T _ {1} > \dots > T _ {m} > \dots > T _ {M},
$$

and

$$
\mathbb {P} _ {\boldsymbol {A}} (T = T _ {m}) = \mathbb {P} _ {\boldsymbol {A} ^ {*}} (T ^ {*} = T _ {m}) = \alpha_ {m}, \quad m = 1, \dots , M.
$$

For $T \in \{T_1, \ldots, T_M\}$ , we have $\alpha_1 \leq \mathbb{P}_{\boldsymbol{A}^*}(T^* \geq T) \leq \sum_{m=1}^{M} \alpha_m = 1$ . If $0 < \alpha < \alpha_1$ , we have

$$
\mathbb {P} _ {\boldsymbol {A}} (p ^ {\mathrm{FRT}} \leq \alpha) = \mathbb {P} _ {\boldsymbol {A}} \left\{\mathbb {P} _ {\boldsymbol {A} ^ {*}} \left(T ^ {*} \geq T\right) \leq \alpha \right\} = 0 \leq \alpha .
$$

If $\alpha_{1} \leq \alpha < 1$ , $\exists \tilde{M} \in \{1, \ldots, M - 1\}$ , such that $\sum_{m=1}^{\tilde{M}} \alpha_{m} \leq \alpha$ and $\sum_{m=1}^{\tilde{M}+1} \alpha_{m} > \alpha$ . Then, we have

$$
\mathbb {P} _ {\boldsymbol {A}} (p ^ {\mathrm{FRT}} \leq \alpha) = \mathbb {P} _ {\boldsymbol {A}} \left\{\mathbb {P} _ {\boldsymbol {A} ^ {*}} \left(T ^ {*} \geq T\right) \leq \alpha \right\} = \mathbb {P} _ {\boldsymbol {A}} \left\{T \in \{T _ {1}, \ldots , T _ {\tilde {M}} \} \right\} = \sum_ {m = 1} ^ {\tilde {M}} \alpha_ {m} \leq \alpha .
$$

If $T(\boldsymbol{A})$ takes distinct values for different $A \in A$ , $p^{FRT}$ is uniformly distributed:

$$
\mathbb {P} _ {\boldsymbol {A}} \left(p ^ {\mathrm{FRT}} = \frac {a}{| \mathcal {A} |}\right) = \frac {1}{| \mathcal {A} |}, \quad a = 1, \dots , | \mathcal {A} |.
$$

Thus, we have

$$
\mathbb {P} _ {\boldsymbol {A}} (p ^ {\mathrm{FRT}} \leq \alpha) = \frac {\lfloor \alpha | \mathcal {A} | \rfloor}{| \mathcal {A} |} > \frac {\alpha | \mathcal {A} | - 1}{| \mathcal {A} |} = \alpha - \frac {1}{| \mathcal {A} |}.
$$

Remark B.1. If $T$ is a continuous random variable, suppose its distribution function is $F(t) = P(T \leq t)$ , then the proof could be simplified as

$$
\begin{array}{l} \mathbb {P} _ {\boldsymbol {A}} \left\{\mathbb {P} _ {\boldsymbol {A} ^ {*}} \left(T ^ {*} \geq T\right) \leq \alpha \right\} = P \left\{1 - F (T) \leq \alpha \right\} \\ = P \left\{T \geq F ^ {- 1} (1 - \alpha) \right\} \\ = 1 - F \{F ^ {- 1} (1 - \alpha) \} \\ = \alpha . \\ \end{array}
$$

However, T is discrete with finite values, and we provide a rigorous proof in the finite-sample setting.

# B.2. Proof of Theorem 2.4

We invoke two lemmas from the Supplementary Material of Puelz et al. (2022).

Lemma B.2 (Lemma 5 in Puelz et al. (2022)). Suppose Assumptions (b) and (c) in Theorem 2.4 hold, for some $r \in (0.5, 1 + O(\log^{-1} M))$ , we have

$$
\mathbb {E} \left(F _ {1, n} \left(q _ {\alpha}\right) - F _ {1, n} \left(q _ {\alpha , M}\right)\right) \geq - O \left(M ^ {- r}\right),
$$

Lemma B.3 (Lemma 4 in Puelz et al. (2022)). Suppose Assumption (a) of Theorem 2.4 holds, for any $0 < \iota < 0.5$ and large enough $M$ , we have

$$
\mathbb {E} \left(F _ {1, n, M} (z) - F _ {1, n} (z)\right) = O \left(M ^ {- 0. 5 + \iota}\right), \quad \text {   for   any   } z \in \mathbb {R}.
$$

Proof of Theorem 2.4. Let $q_{\alpha, M} = F_{0, n, M}^{-1}(1 - \alpha)$ and $q_{\alpha} = F_{0, n}^{-1}(1 - \alpha)$ . Thus, we have

$$
\begin{array}{l} \psi_ {N, M} = 1 - F _ {1, n, M} \left(F _ {0, n, M} ^ {- 1} (1 - \alpha)\right) \\ = 1 - F _ {1, n, M} \left(q _ {\alpha , M}\right) \\ = \underbrace {1 - F _ {1 , n} (q _ {\alpha})} _ {T _ {1}} + \underbrace {F _ {1 , n} (q _ {\alpha}) - F _ {1 , n} (q _ {\alpha , M})} _ {T _ {2}} + \underbrace {F _ {1 , n} (q _ {\alpha , M}) - F _ {1 , n , M} (q _ {\alpha , M})} _ {T _ {3}}. \tag {3} \\ \end{array}
$$

By Assumptions (b) and (c), we have

$$
T _ {1} = 1 - F _ {1, n} \left(F _ {0, n} ^ {- 1} (1 - \alpha)\right) = 1 - F \left(F ^ {- 1} (1 - \alpha) - \tau / \sigma_ {N}\right).
$$

Combined with Lemmas B.2 and B.3, we have

$$
\mathbb {E} (\psi_ {N, M}) \geq 1 - F \left(F ^ {- 1} (1 - \alpha) - \tau / \sigma_ {N}\right) - O (M ^ {- r}) - O (M ^ {- 0. 5 + \iota}).
$$

The result follows from that r > 0.5 > 0.5 - $\iota > 0$ .

# B.3. Proof of Proposition A.1

Proof of Proposition A.1. Since the calibration set $(X_{i},Y_{j})_{i\in \mathcal{C}}$ and external control $(X_{j},Y_{j})$ are exchangeable, we have $(s_i)_{i\in \mathcal{C}}$ and $s_j$ are exchangeable. Thus, we have

$$
\begin{array}{l} \mathbb {P} (p _ {j} ^ {\text { full }} \leq \gamma) = \mathbb {P} \left(\frac {\sum_ {i \in \mathcal {C}} \mathbb {I} (s _ {i} \geq s _ {j}) + 1}{| \mathcal {C} | + 1} \leq \gamma\right) \\ \leq \frac {\lfloor \gamma (| \mathcal {C} | + 1) \rfloor}{| \mathcal {C} | + 1} \\ \leq \gamma , \\ \end{array}
$$

where the first inequality is due to exchangeability and the possibility of ties in $(s_{i})_{i\in\mathcal{C}}$ and $s_{j}$ .

If $s_j$ and $\{s_i\}_{i \in \mathcal{C}}$ have distinct values, $p_j^{\mathrm{full}}$ is uniformly distributed due to exchangeability. That is,

$$
\mathbb {P} \left(p _ {j} ^ {\text { full }} = \frac {a}{| \mathcal {C} | + 1}\right) = \frac {1}{| \mathcal {C} | + 1}, \quad a = 1, \dots , | \mathcal {C} | + 1.
$$

Thus, we have

$$
\mathbb {P} (p _ {j} ^ {\text { full }} \leq \gamma) = \frac {\lfloor \gamma (| \mathcal {C} | + 1) \rfloor}{| \mathcal {C} | + 1} > \frac {\gamma (| \mathcal {C} | + 1) - 1}{| \mathcal {C} | + 1} = \gamma - \frac {1}{| \mathcal {C} | + 1}.
$$

# B.4. Proof of Proposition 3.1

Proof of Proposition 3.1. Since the calibration set $(X_{i},Y_{j})_{i\in \mathcal{C}}$ and external control $(X_{j},Y_{j})$ are exchangeable, we have $(s_i)_{i\in \mathcal{C}_1}$ and $s_j$ are exchangeable. Thus, we have

$$
\begin{array}{l} \mathbb {P} (p _ {j} ^ {\text { split }} \leq \gamma) = \mathbb {P} \left(\frac {\sum_ {i \in \mathcal {C} _ {1}} \mathbb {I} (s _ {i} \geq s _ {j}) + 1}{| \mathcal {C} _ {1} | + 1} \leq \gamma\right) \\ \leq \frac {\left\lfloor \gamma (| \mathcal {C} _ {1} | + 1) \right\rfloor}{| \mathcal {C} _ {1} | + 1} \\ \leq \gamma , \\ \end{array}
$$

where the first inequality is due to exchangeability and the possibility of ties in $(s_{i})_{i\in\mathcal{C}_{1}}$ and $s_{j}$ .

If $s_{j}$ and $\{s_{i}\}_{i\in\mathcal{C}_{1}}$ have distinct values, $p_{j}^{split}$ is uniformly distributed due to exchangeability. That is,

$$
\mathbb {P} \left(p _ {j} ^ {\text { split }} = \frac {a}{| \mathcal {C} _ {1} | + 1}\right) = \frac {1}{| \mathcal {C} _ {1} | + 1}, \quad a = 1, \dots , | \mathcal {C} _ {1} | + 1.
$$

Thus, we have

$$
\mathbb {P} (p _ {j} ^ {\mathrm{split}} \leq \gamma) = \frac {\lfloor \gamma (| \mathcal {C} _ {1} | + 1) \rfloor}{| \mathcal {C} _ {1} | + 1} > \frac {\gamma (| \mathcal {C} _ {1} | + 1) - 1}{| \mathcal {C} _ {1} | + 1} = \gamma - \frac {1}{| \mathcal {C} _ {1} | + 1}.
$$

![](images/f57285eff8d58da128bfd7440ba19beb27caacd1a6c4b2660213396a9f3c0098.jpg)

# B.5. Proof of Proposition A.2

Lemma B.4. Consider a matrix $R \in \mathbb{R}^{(n+1) \times (n+1)}$ with elements $R_{ij}$ . Define the set

$$
\mathcal {S} = \left\{j \in \{1, \dots , n + 1 \}: \sum_ {i = 1} ^ {n + 1} \mathbb {I} (R _ {i j} <   R _ {j i}) \geq (1 - \gamma) (n + 1) \right\}, \quad \gamma \in (0, 1).
$$

Then, we have

$$
s \leq 2 \gamma (n + 1) - 1 <   2 \gamma (n + 1),
$$

where $s = |\mathcal{S}|$ .

Proof. Since

$$
\sum_ {i = 1} ^ {n + 1} \mathbb {I} (R _ {i j} <   R _ {j i}) \geq (1 - \gamma) (n + 1) \quad \Leftrightarrow \quad \sum_ {i = 1} ^ {n + 1} \mathbb {I} (R _ {i j} \geq R _ {j i}) \leq \gamma (n + 1),
$$

by summing over all $j \in S$ , we have

$$
\sum_ {j \in \mathcal {S}} \sum_ {i = 1} ^ {n + 1} \mathbb {I} (R _ {i j} \geq R _ {j i}) \leq s \gamma (n + 1).
$$

For $i \neq j$ , since $\mathbb{I}(R_{ij} \geq R_{ji}) + \mathbb{I}(R_{ji} \geq R_{ij}) \geq 1$ , we have

$$
\begin{array}{l} \sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}} \mathbb {I} (R _ {i j} \geq R _ {j i}) = \sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}, i \neq j} \mathbb {I} (R _ {i j} \geq R _ {j i}) + s \\ \geq \frac {s (s - 1)}{2} + s. \\ \end{array}
$$

By combining these two inequalities, we obtain

$$
\begin{array}{l} \frac {s (s - 1)}{2} + s \leq \sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}} \mathbb {I} (R _ {i j} \geq R _ {j i}) \\ \leq \sum_ {j \in \mathcal {S}} \sum_ {i = 1} ^ {n + 1} \mathbb {I} (R _ {i j} \geq R _ {j i}) \\ \leq s \gamma (n + 1). \\ \end{array}
$$

Thus, we have

$$
\frac {(s - 1)}{2} + 1 \leq \gamma (n + 1) \quad \Rightarrow \quad s \leq 2 \gamma (n + 1) - 1 <   2 \gamma (n + 1).
$$

Proof of Proposition A.2. For $i', j' \in C \cup \{j\}$ , we define

$$
R _ {i ^ {\prime} j ^ {\prime}} = \left\{ \begin{array}{l l} + \infty & i ^ {\prime} = j ^ {\prime}, \\ \left| Y _ {i ^ {\prime}} - \hat {f} _ {- (i ^ {\prime}, j ^ {\prime})} \left(X _ {i ^ {\prime}}\right) \right| & i ^ {\prime} \neq j ^ {\prime}, \end{array} \right.
$$

where $\hat{f}_{-(i',j')}$ is a prediction model fitted by the leave-two-out augmented set $(\mathcal{C} \cup \{j\}) \setminus \{i', j'\}$ . For $i \in \mathcal{C}$ , since $(\mathcal{C} \cup \{j\}) \setminus \{i, j\} = \mathcal{C} \setminus \{i\}$ , we have $\hat{f}_{-i}(x) = \hat{f}_{-(i,j)}(x)$ , thereby,

$$
s _ {i} = | Y _ {i} - \hat {f} _ {- i} (X _ {i}) | = R _ {i j},
$$

$$
s _ {j} ^ {(i)} = | Y _ {j} - \hat {f} _ {- i} (X _ {j}) | = R _ {j i}.
$$

Thus, we have

$$
\begin{array}{l} \mathbb {P} (p _ {j} ^ {\text { jackknife } +} \leq \gamma) = \mathbb {P} \left(\frac {\sum_ {i \in \mathcal {C}} \mathbb {I} (s _ {i} \geq s _ {j} ^ {(i)}) + 1}{| \mathcal {C} | + 1} \leq \gamma\right) \\ = \mathbb {P} \left(\frac {\sum_ {i \in \mathcal {C} \cup \{j \}} \mathbb {I} (R _ {i j} \geq R _ {j i})}{| \mathcal {C} | + 1} \leq \gamma\right) \\ = \mathbb {P} \left(\sum_ {i \in \mathcal {C} \cup \{j \}} \mathbb {I} (R _ {i j} <   R _ {j i}) \geq (1 - \gamma) (| \mathcal {C} | + 1)\right) \\ \leq 2 \gamma - \frac {1}{| \mathcal {C} | + 1} \\ <   2 \gamma , \\ \end{array}
$$

where first inequality is due to exchangeability and Lemma B.4.

# B.6. Proof of Proposition 3.3

Lemma B.5. Suppose $m = n / K$ is an integer, and the $n + m$ units are evenly divided into $K + 1$ sets, denoted by $\mathcal{C}_1, \ldots, \mathcal{C}_{K + 1}$ . Consider a matrix $R \in \mathbb{R}^{(n + m) \times (n + m)}$ with elements $R_{ij} = R_{ji}$ if $i$ and $j$ belong to the same set. Define the set

$$
\mathcal {S} = \left\{j \in \{1, \dots , n + m \}: \sum_ {i = 1} ^ {n + m} \mathbb {I} (R _ {i j} <   R _ {j i}) \geq (1 - \gamma) (n + 1) \right\}, \quad \gamma \in (0, 1).
$$

Then, we have

$$
s \leq 2 \gamma (n + 1) + m - 2,
$$

where $s = |\mathcal{S}|$ .

Proof. For $j \in S$ , by definition, we have

$$
\sum_ {i = 1} ^ {n + m} \mathbb {I} (R _ {i j} \geq R _ {j i}) \leq (n + m) - (1 - \gamma) (n + 1).
$$

Since $R_{ij} = R_{ji}$ if $i$ and $j$ belong to the same set, we have

$$
\begin{array}{l} \sum_ {i = 1} ^ {n + m} \mathbb {I} (R _ {i j} \geq R _ {j i}) = \sum_ {i \notin \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) + \sum_ {i \in \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) \\ = \sum_ {i \notin \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) + m, \\ \end{array}
$$

where $\mathcal{C}_{k(j)}$ is the set containing unit $j$ . Thus, we have

$$
\begin{array}{l} \sum_ {i \notin \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) \leq (n + m) - (1 - \gamma) (n + 1) - m \\ = \gamma (n + 1) - 1. \\ \end{array}
$$

By summing over all $j \in S$ , we have

$$
\sum_ {j \in \mathcal {S}} \sum_ {i \notin \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) \leq s \{\gamma (n + 1) - 1 \}. \tag {4}
$$

On the other hand, for $i \neq j$ , since $\mathbb{I}(R_{ij} \geq R_{ji}) + \mathbb{I}(R_{ji} \geq R_{ij}) \geq 1$ , we have

$$
\sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}, i \neq j} \mathbb {I} (R _ {i j} \geq R _ {j i}) \geq \frac {s (s - 1)}{2}.
$$

Since $R_{ij} = R_{ji}$ if $i$ and $j$ belong to the same set, we have

$$
\begin{array}{l} \sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}, i \neq j} \mathbb {I} (R _ {i j} \geq R _ {j i}) = \sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}, i \notin \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) + \sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}, i \in \mathcal {C} _ {k (j)}, i \neq j} \mathbb {I} (R _ {i j} \geq R _ {j i}) \\ = \sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}, i \notin \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) + \sum_ {k = 1} ^ {K + 1} \frac {s _ {k} (s _ {k} - 1)}{2}, \\ \end{array}
$$

where $s_k = |\mathcal{C}_k\cap \mathcal{S}|$ . Thus, we have

$$
\sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}, i \notin \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) \geq \frac {s (s - 1)}{2} - \sum_ {k = 1} ^ {K + 1} \frac {s _ {k} (s _ {k} - 1)}{2}. \tag {5}
$$

By combining (4) and (5), we have

$$
\begin{array}{l} \frac {s (s - 1)}{2} - \sum_ {k = 1} ^ {K + 1} \frac {s _ {k} (s _ {k} - 1)}{2} \leq \sum_ {j \in \mathcal {S}} \sum_ {i \in \mathcal {S}, i \notin \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) \\ \leq \sum_ {j \in \mathcal {S}} \sum_ {i \notin \mathcal {C} _ {k (j)}} \mathbb {I} (R _ {i j} \geq R _ {j i}) \\ \leq s \{\gamma (n + 1) - 1 \}. \\ \end{array}
$$

Since $s_k \leq m$ , we have

$$
\sum_ {k = 1} ^ {K + 1} \frac {s _ {k} (s _ {k} - 1)}{2} \leq \frac {s (m - 1)}{2}.
$$

Thus, we have

$$
s \leq 2 \gamma (n + 1) + m - 2.
$$

Proof of Proposition 3.3. We consider $m = |\mathcal{C}| / K$ is an integer for simplicity. Let $\mathcal{C}_{K + 1}$ contain $j$ and other $m - 1$ hypothetical points. For $i', j' \in \cup_{k=1}^{K+1} \mathcal{C}_k$ , we define

$$
R _ {i ^ {\prime} j ^ {\prime}} = \left\{ \begin{array}{l l} + \infty & k (i ^ {\prime}) = k (j ^ {\prime}), \\ \left| Y _ {i ^ {\prime}} - \hat {f} _ {- (\mathcal {C} _ {k (i ^ {\prime})}, \mathcal {C} _ {k (j ^ {\prime})})} (X _ {i ^ {\prime}}) \right| & k (i ^ {\prime}) \neq k (j ^ {\prime}), \end{array} \right.
$$

where $\hat{f}_{-(\mathcal{C}_{k(i')},\mathcal{C}_{k(j')})}$ is a prediction model fitted by the leave-two-set-out augmented set $(\cup_{k=1}^{K+1}\mathcal{C}_k)\setminus(\mathcal{C}_{k(i')} \cup \mathcal{C}_{k(j')})$ . Since $\mathcal{C} = \cup_{k=1}^{K}\mathcal{C}_k$ and $\mathcal{C}_{k(j)} = \mathcal{C}_{K+1}$ , we have $(\cup_{k=1}^{K+1}\mathcal{C}_k)\setminus(\mathcal{C}_{k(i)} \cup \mathcal{C}_{k(j)}) = \mathcal{C}\setminus\mathcal{C}_{k(i)}$ for $i \in \mathcal{C}$ . Thus, for $i \in \mathcal{C}$ , we have $\hat{f}_{-\mathcal{C}_{k(i)}}(x) = \hat{f}_{-(\mathcal{C}_{k(i)},\mathcal{C}_{k(j)})}(x)$ , thereby,

$$
s _ {i} = | Y _ {i} - \hat {f} _ {- \mathcal {C} _ {k (i)}} (X _ {i}) | = R _ {i j},
$$

$$
s _ {j} ^ {(i)} = | Y _ {j} - \hat {f} _ {- \mathcal {C} _ {k (i)}} (X _ {j}) | = R _ {j i}.
$$

Thus, we have

$$
\begin{array}{l} \mathbb {P} (p _ {j} ^ {\mathrm{cv} +} \leq \gamma) = \mathbb {P} \left(\frac {\sum_ {i \in \mathcal {C}} \mathbb {I} (s _ {i} \geq s _ {j} ^ {(i)}) + 1}{| \mathcal {C} | + 1} \leq \gamma\right) \\ = \mathbb {P} \left(\frac {\sum_ {i \in \mathcal {C} \cup \{j \}} \mathbb {I} (R _ {i j} \geq R _ {j i})}{| \mathcal {C} | + 1} \leq \gamma\right) \\ = \mathbb {P} \left(\sum_ {i \in \mathcal {C} \cup \{j \}} \mathbb {I} (R _ {i j} <   R _ {j i}) \geq (1 - \gamma) (| \mathcal {C} | + 1)\right) \\ \leq \mathbb {P} \left(\sum_ {i \in \cup_ {k = 1} ^ {K + 1} \mathcal {C} _ {k}} \mathbb {I} (R _ {i j} <   R _ {j i}) \geq (1 - \gamma) (| \mathcal {C} | + 1)\right) \\ \leq \frac {2 \gamma (| \mathcal {C} | + 1) + m - 2}{| \mathcal {C} | + m} \\ \leq 2 \gamma + \frac {(1 - 2 \gamma) (m - 1) - 1}{| \mathcal {C} | + m} \\ \leq 2 \gamma + \frac {1 - K / | \mathcal {C} |}{K + 1}, \\ \end{array}
$$

where the second inequality is due to exchangeability and Lemma B.5.

# B.7. Proof of Theorem 3.4

Proof of Theorem 3.4. Since $\epsilon_{\gamma}$ is a centered sub-Gaussian variable with parameter $\phi_{\gamma}$ , we have $\epsilon_{\gamma} - \epsilon_1$ as a centered sub-Gaussian variable with parameter $2\Phi$ , where $\Phi = \max_{\gamma \in \Gamma} \phi_{\gamma}$ . Moreover, we have $(\epsilon_{\gamma} - \epsilon_1)^2 - \kappa_{\gamma}^2$ is a centered sub-exponential variable with parameters $(c_1\Phi^2, c_1\Phi^2)$ , where $c_1$ is a constant. By $\hat{\tau}_{\gamma} - \hat{\tau}_1 = (\delta_{\gamma} - \delta_1) + (\epsilon_{\gamma} - \epsilon_1)$ and using the concentration inequalities for sub-Gaussian and sub-exponential variables (Wainwright, 2019), it follows that, with probability at least $1 - 4\iota$ ,

$$
\begin{array}{l} \max _ {\gamma \in \Gamma} | (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) ^ {2} - (\delta_ {\gamma} - \delta_ {1}) ^ {2} - \kappa_ {\gamma} ^ {2} | \\ = \max _ {\gamma \in \Gamma} | 2 (\delta_ {\gamma} - \delta_ {1}) (\epsilon_ {\gamma} - \epsilon_ {1}) + (\epsilon_ {\gamma} - \epsilon_ {1}) ^ {2} - \kappa_ {\gamma} ^ {2} | \\ \leq 8 \sqrt {2} \Delta \Phi \sqrt {\log (| \Gamma | / \iota)} + \max \left\{\sqrt {2} c _ {1} \Phi^ {2} \sqrt {\log (| \Gamma | / \iota)}, 2 c _ {1} \Phi^ {2} \log (| \Gamma | / \iota) \right\}, \tag {6} \\ \end{array}
$$

where $\Delta = \max_{\gamma \in \Gamma}|\delta_{\gamma}|$ .

By (6), it follows that, with probability at least $1 - 4\iota$ ,

$$
\begin{array}{l} \max _ {\gamma \in \Gamma} \left| \widehat {\mathrm{MSE}} (\gamma) - \mathrm{MSE} (\gamma) \right| \\ = \max _ {\gamma \in \Gamma} \left| (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) ^ {2} - \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) + \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - \delta_ {\gamma} ^ {2} - \sigma_ {\gamma} ^ {2} \right| \\ \leq \max _ {\gamma \in \Gamma} \left| (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) ^ {2} - \kappa_ {\gamma} ^ {2} + \sigma_ {\gamma} ^ {2} - \delta_ {\gamma} ^ {2} - \sigma_ {\gamma} ^ {2} \right| + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) - \kappa_ {\gamma} ^ {2} | + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - \sigma_ {\gamma} ^ {2} | \\ \leq \max _ {\gamma \in \Gamma} \left| (\delta_ {\gamma} - \delta_ {1}) ^ {2} - \delta_ {\gamma} ^ {2} \right| + c \Delta \Phi \sqrt {\log (| \Gamma | / \iota)} + \max \left\{c \Phi^ {2} \sqrt {\log (| \Gamma | / \iota)}, c \Phi^ {2} \log (| \Gamma | / \iota) \right\} \\ + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) - \kappa_ {\gamma} ^ {2} | + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - \sigma_ {\gamma} ^ {2} | \\ \leq c \Delta | \delta_ {1} | + c \Delta \Phi \sqrt {\log (| \Gamma | / \iota)} + \max \left\{c \Phi^ {2} \sqrt {\log (| \Gamma | / \iota)}, c \Phi^ {2} \log (| \Gamma | / \iota) \right\} \\ + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) - \kappa_ {\gamma} ^ {2} | + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - \sigma_ {\gamma} ^ {2} |, \\ \end{array}
$$

where $c$ is a constant.

![](images/590dd12632f244ff2b975d920ce619f53cbea5f0ddadef5f52ade9513429b837.jpg)

# B.8. Proof of Theorem 3.5

Proof of Theorem 3.5. Since $\epsilon_{\gamma}$ is a centered sub-Gaussian variable with parameter $\phi_{\gamma}$ , we have $\epsilon_{\gamma}^{2} - \sigma_{\gamma}^{2}$ as a centered sub-exponential variable with parameter $(c_2\Phi^2, c_2\Phi^2)$ , where $c_2$ is a constant. By $\hat{\tau}_{\gamma} - \tau = \delta_{\gamma} + \epsilon_{\gamma}$ and using the concentration inequalities for sub-Gaussian and sub-exponential variables (Wainwright, 2019), it follows that, with probability at least $1 - 4\iota$ ,

$$
\begin{array}{l} \max _ {\gamma \in \Gamma} | (\hat {\tau} _ {\gamma} - \tau) ^ {2} - \delta_ {\gamma} ^ {2} - \sigma_ {\gamma} ^ {2} | \\ = \max _ {\gamma \in \Gamma} | 2 \delta_ {\gamma} \epsilon_ {\gamma} + \epsilon_ {\gamma} ^ {2} - \sigma_ {\gamma} ^ {2} | \\ \leq 2 \sqrt {2} \Delta \Phi \sqrt {\log (| \Gamma | / \iota)} + \max \left\{\sqrt {2} c _ {2} \Phi^ {2} \sqrt {\log (| \Gamma | / \iota)}, 2 c _ {2} \Phi^ {2} \log (| \Gamma | / \iota) \right\}. \tag {7} \\ \end{array}
$$

By (6) and (7), it follows that, with probability at least $1 - 8\iota$ ,

$$
\begin{array}{l} \max _ {\gamma \in \Gamma} \left| \widehat {\mathrm{MSE}} (\gamma) - (\hat {\tau} _ {\gamma} - \tau) ^ {2} \right| \\ = \max _ {\gamma \in \Gamma} \left| (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) ^ {2} - \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) + \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - (\hat {\tau} _ {\gamma} - \tau) ^ {2} \right| \\ \leq \max _ {\gamma \in \Gamma} \left| (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) ^ {2} - \kappa_ {\gamma} ^ {2} + \sigma_ {\gamma} ^ {2} - (\hat {\tau} _ {\gamma} - \tau) ^ {2} \right| + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) - \kappa_ {\gamma} ^ {2} | + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - \sigma_ {\gamma} ^ {2} | \\ \leq \max _ {\gamma \in \Gamma} \left| (\delta_ {\gamma} - \delta_ {1}) ^ {2} - \delta_ {\gamma} ^ {2} \right| + c \Delta \Phi \sqrt {\log (| \Gamma | / \iota)} + \max \left\{c \Phi^ {2} \sqrt {\log (| \Gamma | / \iota)}, c \Phi^ {2} \log (| \Gamma | / \iota) \right\} \\ + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) - \kappa_ {\gamma} ^ {2} | + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - \sigma_ {\gamma} ^ {2} | \\ \leq c \Delta | \delta_ {1} | + c \Delta \Phi \sqrt {\log (| \Gamma | / \iota)} + \max \left\{c \Phi^ {2} \sqrt {\log (| \Gamma | / \iota)}, c \Phi^ {2} \log (| \Gamma | / \iota) \right\} \\ + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma} - \hat {\tau} _ {1}) - \kappa_ {\gamma} ^ {2} | + \max _ {\gamma \in \Gamma} | \widehat {\mathbb {V}} (\hat {\tau} _ {\gamma}) - \sigma_ {\gamma} ^ {2} |, \tag {8} \\ \end{array}
$$

where $c$ is a constant.

Since

$$
\left| \min _ {\gamma \in \Gamma} \widehat {\mathrm{MSE}} (\gamma) - \min _ {\gamma \in \Gamma} (\hat {\tau} _ {\gamma} - \tau) ^ {2} \right| \leq \max _ {\gamma \in \Gamma} \left| \widehat {\mathrm{MSE}} (\gamma) - (\hat {\tau} _ {\gamma} - \tau) ^ {2} \right|,
$$

and

$$
\left| \min _ {\gamma \in \Gamma} \widehat {\mathrm{MSE}} (\gamma) - (\hat {\tau} _ {\hat {\gamma}} - \tau) ^ {2} \right| = \left| \widehat {\mathrm{MSE}} (\hat {\gamma}) - (\hat {\tau} _ {\hat {\gamma}} - \tau) ^ {2} \right| \leq \max _ {\gamma \in \Gamma} \left| \widehat {\mathrm{MSE}} (\gamma) - (\hat {\tau} _ {\gamma} - \tau) ^ {2} \right|,
$$

![](images/ded000daa85a889aa5a841b896224ccc464f75c8f085e3c2544ebc185fb3764c.jpg)

<details>
<summary>line</summary>

| τ | No Hidden Bias - No Borrow (γ = 1) | No Hidden Bias - Full Borrow (γ = 0) | No Hidden Bias - Conformal Selective Borrow (ξ̂) | Half of ECs Exhibit Hidden Bias - No Borrow (γ = 1) | Half of ECs Exhibit Hidden Bias - Full Borrow (γ = 0) | Half of ECs Exhibit Hidden Bias - Conformal Selective Borrow (ξ̂) |
|---|---|---|---|---|---|---|
| 0.0 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| 0.3 | 0.15 | 0.28 | 0.28 | 0.15 | 0.08 | 0.25 |
| 0.6 | 0.35 | 0.48 | 0.48 | 0.35 | 0.20 | 0.45 |
| 0.9 | 0.78 | 0.82 | 0.82 | 0.78 | 0.52 | 0.82 |
| 1.2 | 0.92 | 0.92 | 0.92 | 0.92 | 0.65 | 0.92 |
</details>

Figure 3. Power curves when $b = 0$ and $b = 8$ .

we have

$$
(\hat {\tau} _ {\hat {\gamma}} - \tau) ^ {2} - \min _ {\gamma \in \Gamma} (\hat {\tau} _ {\gamma} - \tau) ^ {2} \leq 2 \max _ {\gamma \in \Gamma} \left| \widehat {\mathrm{MSE}} (\gamma) - (\hat {\tau} _ {\gamma} - \tau) ^ {2} \right|.
$$

The result follows from (8).

![](images/3eca01c595f7befd10f5454c39595e5ea56d6ff62f43a8594016ab5d1c8f25e6.jpg)

# C. Additional simulation results

# C.1. Power curve

For the scenario where there is no hidden bias $(b = 0)$ and another where half of the ECs exhibit hidden bias with a magnitude of b = 8, we vary $\tau$ to plot the power curve, as shown in Figure 3. CSB outperforms NB in both cases, while FB demonstrates low power in the presence of hidden bias.

# C.2. Adaptivity of the selection threshold

Figure 4 illustrates how $\hat{\gamma}$ changes with the magnitude of b: (i) When there is no bias (b = 0), $\hat{\gamma}$ approaches 0 to borrow all ECs and maximize power; (ii) with moderate bias (b = 1, 2, 3), where distinguishing between biased and unbiased ECs is challenging, $\hat{\gamma}$ increases to help discard the biased ECs; (iii) when the bias is large ( $b \geq 4$ ), $\hat{\gamma}$ decreases but remains non-zero, retaining more unbiased ECs, while easily discarding the biased ones.

# C.3. Various selection thresholds

Figure 5 shows the performance of the fixed selection threshold $\gamma$ and the adaptive selection threshold $\hat{\gamma}$ when $n_{\mathcal{E}} = 50$ . As discussed in Section 3.3, smaller $\gamma$ selects more ECs but risks greater bias when distinguishing between biased and unbiased ECs is difficult. This creates a power trade-off across different bias levels, similar to MSE simulation results in data integration (Yang et al., 2023; Oberst et al., 2022; Lin et al., 2024). We find that (i) CSB with $\gamma = 0.6$ improves power compared to NB, except in extreme cases like $b = 2, 3$ , where it decreases power slightly, and (ii) CSB with $\hat{\gamma}$ further improves power but also risks power loss in difficult scenarios. The power trade-off does not compromise the Type I error rate, which remains controlled with all selection thresholds.

# C.4. Comparison to Adaptive Lasso Selective Borrowing

Figure 6 presents the simulation results for ALSB with asymptotic inference. Unlike CSB + FRT, ALSB with asymptotic inference fails to control the type I error rate in this small sample size scenario. Additionally, CSB demonstrates better estimation and selection performance in most cases.

![](images/f86a03e3033057ab6068678d098335705ef07aa42e5a312eba750f0f89998549.jpg)

<details>
<summary>bar</summary>

| b   | count | value |
| --- | ----- | ----- |
| 0   | 0     | 1.00  |
| 0   | 50    | 0.75  |
| 0   | 100   | 0.50  |
| 0   | 150   | 0.25  |
| 0   | 200   | 0.00  |
| 1   | 0     | 1.00  |
| 1   | 50    | 0.75  |
| 1   | 100   | 0.50  |
| 1   | 150   | 0.25  |
| 1   | 200   | 0.00  |
| 2   | 0     | 1.00  |
| 2   | 50    | 0.75  |
| 2   | 100   | 0.50  |
| 2   | 150   | 0.25  |
| 2   | 200   | 0.00  |
| 3   | 0     | 1.00  |
| 3   | 50    | 0.75  |
| 3   | 100   | 0.50  |
| 3   | 150   | 0.25  |
| 3   | 200   | 0.00  |
| 4   | 0     | 1.00  |
| 4   | 50    | 0.75  |
| 4   | 100   | 0.50  |
| 4   | 150   | 0.25  |
| 4   | 200   | 0.00  |
| 5   | 0     | 1.00  |
| 5   | 50    | 0.75  |
| 5   | 100   | 0.50  |
| 5   | 150   | 0.25  |
| 5   | 200   | 0.00  |
| 6   | 0     | 1.00  |
| 6   | 50    | 0.75  |
| 6   | 100   | 0.50  |
| 6   | 150   | 0.25  |
| 6   | 200   | 0.00  |
| 7   | 0     | 1.00  |
| 7   | 50    | 0.75  |
| 7   | 100   | 0.50  |
| 7   | 150   | 0.25  |
| 7   | 200   | 0.00  |
| 8   | 0     | 1.00  |
| 8   | 50    | 0.75  |
| 8   | 100   | 0.50  |
| 8   | 150   | 0.25  |
| 8   | 200   | 0.00  |
</details>

Figure 4. $\hat{\gamma}$ versus $b$ when $n_{\mathcal{E}} = 50$ .   
![](images/98f80c10e674f209b110d19629caec7da217f51f2b121735a5793995c2470725.jpg)  
No Borrow ( $\gamma=1$ )   
• Conformal Selective Borrow ( $\gamma=0.8$ )   
• Conformal Selective Borrow ( $\gamma=0.6$ )   
• Conformal Selective Borrow ( $\gamma=0.4$ )

Figure 5. Simulation results for various selection threshold $\gamma$ 's when $n_{\mathcal{E}} = 50$ .

We further compared CSB with asymptotic inference to ALSB with asymptotic inference. Figure 7 shows that CSB+Asym Inf generally achieves better Type I error control than ALSB+Asym Inf, while performing comparably when b = 1.

We did not compare to ALSB + FRT because, while CSB is compatible with FRT, ALSB is not readily applicable due to its computational complexity. This highlights an advantage of CSB when exact finite-sample inference is desired.

# C.5. A larger sample size of ECs

Figures 8, 9, 10, 11, and 12 show the simulation results for $n_{\mathcal{E}} = 300$ . The conclusion is similar to that in the main text.

# C.6. Dependent covariates with p = 5

We additionally consider p = 5 and $X \sim N(0, \Sigma)$ , where $\Sigma$ is a Toeplitz matrix with $\rho = 0.6$ to introduce dependence among the coordinates of X. We did not consider larger p since the sample size is small, with only 25 RCT controls. The simulation results (see Figure 13) show similar conclusions and demonstrate the robustness of our method.

# D. More details about the real data

Pseudo-observations. Figure 14 shows the pseudo-observations versus censored times for CALGB 9633 and NCDB, illustrating that (i) all pseudo-observations are less than or equal to the truncation time of 3 years; (ii) when an event occurs before 3 years, pseudo-observations are generally equal to the event time; and (iii) when censoring occurs before 3 years, pseudo-observations are typically greater than the censored time.

Matching. We use nearest-neighbor matching to mitigate the covariate imbalance between CALGB 9633 and NCDB. Tumor size was imputed for eight missing values in CALGB 9633 using the median of 4. NCDB samples with missing values

(A) Absolute Bias   
![](images/e86a975a510e2548e8e211cc09d5853374263aaab8f268bf4e1ecd78ee891e1b.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Blue Line | Orange Dashed Line |
| ------------------------ | --------- | ------------------ |
| 0                        | 0.00      | 0.00               |
| 1                        | 0.08      | 0.06               |
| 2                        | 0.04      | 0.09               |
| 3                        | 0.01      | 0.05               |
| 4                        | 0.00      | 0.03               |
| 5                        | 0.00      | 0.01               |
| 6                        | 0.00      | 0.01               |
| 7                        | 0.00      | 0.01               |
| 8                        | 0.00      | 0.01               |
</details>

(B) Variance   
![](images/11949daa1a18a52e927b32ad84c54a6b5d0a74349752022fd5821fb0bf230088.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Line 1 | Line 2 |
| ------------------------ | ------ | ------ |
| 0                        | 0.085  | 0.102  |
| 1                        | 0.120  | 0.097  |
| 2                        | 0.108  | 0.103  |
| 3                        | 0.095  | 0.105  |
| 4                        | 0.092  | 0.103  |
| 5                        | 0.091  | 0.104  |
| 6                        | 0.091  | 0.105  |
| 7                        | 0.091  | 0.106  |
| 8                        | 0.091  | 0.105  |
</details>

(C) MSE   
![](images/2324f909b85da153744b8d5bce68bc378f34d65a35e041bdbf7b82ddb57c2930.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Line 1 | Line 2 |
| ------------------------ | ------ | ------ |
| 0                        | 0.08   | 0.10   |
| 1                        | 0.13   | 0.10   |
| 2                        | 0.11   | 0.11   |
| 3                        | 0.095  | 0.105  |
| 4                        | 0.09   | 0.105  |
| 5                        | 0.09   | 0.105  |
| 6                        | 0.09   | 0.105  |
| 7                        | 0.09   | 0.105  |
| 8                        | 0.09   | 0.105  |
</details>

(D) Type I Error Rate   
![](images/3dc1cd8533fe633b53be891bef5327d0b4352b5ebc20d95cadbd50ace4910393.jpg)

<details>
<summary>scatter</summary>

| Magnitude of Hidden Bias | Series 1 | Series 2 |
| ------------------------ | -------- | -------- |
| 0                        | 0.2      | 0.05     |
| 1                        | 0.2      | 0.05     |
| 2                        | 0.2      | 0.05     |
| 3                        | 0.2      | 0.05     |
| 4                        | 0.15     | 0.05     |
| 5                        | 0.15     | 0.05     |
| 6                        | 0.15     | 0.05     |
| 7                        | 0.15     | 0.05     |
| 8                        | 0.1      | 0.05     |
</details>

(E) Power   
![](images/c6778ebc76fcfe8396eff70f3dd0999d0921458ac9f677fa8917d08bbc9b2d66.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Line 1 | Line 2 |
| ------------------------ | ------ | ------ |
| 0                        | 0.5    | 0.5    |
| 2                        | 0.6    | 0.35   |
| 4                        | 0.55   | 0.3    |
| 6                        | 0.5    | 0.35   |
| 8                        | 0.45   | 0.35   |
</details>

Method • No Borrow ( $\gamma=1$ ) • Adaptive Lasso Selective Borrow • Conformal Selective Borrow ( $\hat{\gamma}$ )

Figure 6. Comparison of CSB + FRT and ALSB + asymptotic inference when $n_{E} = 50$ .   
![](images/609176212bd3ed0429d83c8b2ae40f8629a79a81e2bcd2f034423262d1433c2c.jpg)

<details>
<summary>bar</summary>

| Magnitude of Hidden Bias | Series 1 (Orange) | Series 2 (Blue) | Series 3 (Green) |
| ------------------------ | ----------------- | --------------- | ---------------- |
| 0                        | 0.22              | 0.10            | 0.06             |
| 1                        | 0.19              | 0.24            | 0.06             |
| 2                        | 0.21              | 0.13            | 0.06             |
| 3                        | 0.19              | 0.09            | 0.06             |
| 4                        | 0.15              | 0.09            | 0.06             |
| 5                        | 0.13              | 0.09            | 0.06             |
| 6                        | 0.13              | 0.09            | 0.06             |
| 7                        | 0.13              | 0.09            | 0.06             |
| 8                        | 0.11              | 0.09            | 0.06             |
</details>

Method • No Borrow ( $\gamma=1$ ) + Asym Inf • Adaptive Lasso Selective Borrow + Asym Inf • Conformal Selective Borrow ( $\hat{\gamma}$ ) + Asym Inf

![](images/ceee0a47a01e2d76157f787d54d9fdcbdf4022459f21b0e7f488f993fbd08628.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Power (Blue Line) | Power (Orange Dashed Line) | Power (Green Dotted Line) |
| ------------------------ | ----------------- | -------------------------- | ------------------------- |
| 0                        | 0.48              | 0.53                       | 0.34                      |
| 1                        | 0.53              | 0.59                       | 0.34                      |
| 2                        | 0.50              | 0.61                       | 0.34                      |
| 3                        | 0.46              | 0.57                       | 0.34                      |
| 4                        | 0.45              | 0.53                       | 0.34                      |
| 5                        | 0.45              | 0.50                       | 0.34                      |
| 6                        | 0.45              | 0.50                       | 0.34                      |
| 7                        | 0.45              | 0.49                       | 0.34                      |
| 8                        | 0.45              | 0.48                       | 0.34                      |
</details>

Figure 7. Comparison of CSB + asymptotic inference and ALSB + asymptotic inference when $n_{\varepsilon} = 50$ .

or covariates outside the CALGB 9633 range were excluded, leaving 10,241 samples. We perform 1:1 nearest-neighbor matching using MatchIt (Ho et al., 2011), treating the sampling indicator S as a “treatment” and targeting the average treatment effect on the treated (ATT). This preserves all RCT samples and matches 335 NCDB samples. Distributional balance for the baseline covariates and the estimated sampling score $\hat{\mathbb{P}}(S=1|X)$ improves significantly after matching, with a visual comparison in Figure 15. However, certain covariates, such as tumor size, remain imbalanced, which could not be addressed by matching without resorting to methods that would undesirably discard RCT samples. This motivates the use of the doubly robust estimator in Sections 2.1 and 3. Notably, while a doubly robust estimator alone can address covariate imbalance, matching as a pre-processing step reduces reliance on correct model specification (Ho et al., 2007). A summary table of the pre-processed data is in Table 2.

Selection performance. Figure 16 shows that, given the observed confounder X, CSB tends to select ECs whose outcomes are more similar to randomized controls, reducing hidden bias that cannot be addressed by balancing X alone.

(A) Absolute Bias   
![](images/14283d33848f4ce85d562f8ebb54556ba11117d849b33da4389567d531cf59c3.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Value |
| ------------------------ | ----- |
| 0                        | 0.0   |
| 1                        | 0.4   |
| 2                        | 0.9   |
| 3                        | 1.2   |
| 4                        | 1.4   |
| 5                        | 1.5   |
| 6                        | 1.5   |
| 7                        | 1.5   |
| 8                        | 1.5   |
</details>

(B) Variance   
![](images/1838e35f437be56d0383baae0866e98691afc38c5e3df2933d211869311cb66d.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Blue Line | Red Dotted Line |
| ------------------------ | --------- | --------------- |
| 0                        | 0.075     | 0.065           |
| 2                        | 0.120     | 0.065           |
| 4                        | 0.075     | 0.085           |
| 6                        | 0.075     | 0.130           |
| 8                        | 0.075     | 0.175           |
</details>

(C) MSE   
![](images/aed8362bdc620a2553bfd96cb4ce25cddefbe355a7e0f376964f4f82ef41b314.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Red Dashed Line | Green Dotted Line | Blue Solid Line |
| ------------------------ | --------------- | ----------------- | --------------- |
| 0                        | 0.05            | 0.1               | 0.05            |
| 1                        | 0.3             | 0.1               | 0.1             |
| 2                        | 0.9             | 0.1               | 0.1             |
| 3                        | 1.4             | 0.1               | 0.1             |
| 4                        | 1.8             | 0.1               | 0.1             |
| 5                        | 2.0             | 0.1               | 0.1             |
| 6                        | 2.2             | 0.1               | 0.1             |
| 7                        | 2.3             | 0.1               | 0.1             |
| 8                        | 2.4             | 0.1               | 0.1             |
</details>

(D) Type I Error Rate   
![](images/8a8f63f8bf47e59cfcef9db31ccca67fa7c2eac339371ed2cdde47558aa0dbd2.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Value |
| ------------------------ | ----- |
| 0                        | 0.05  |
| 1                        | 0.05  |
| 2                        | 0.05  |
| 3                        | 0.05  |
| 4                        | 0.05  |
| 5                        | 0.05  |
| 6                        | 0.05  |
| 7                        | 0.05  |
| 8                        | 0.05  |
</details>

(E) Power   
![](images/a5e7ace2d2746138513f28ae2e01e7ea7a27b48db404113ab2158edfc836a6b7.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Series 1 | Series 2 |
| ------------------------ | -------- | -------- |
| 0                        | 0.50     | 0.50     |
| 2                        | 0.25     | 0.50     |
| 4                        | 0.25     | 0.50     |
| 6                        | 0.50     | 0.50     |
| 8                        | 0.50     | 0.50     |
</details>

Method • No Borrow ( $\gamma=1$ ) • Full Borrow ( $\gamma=0$ ) • Conformal Selective Borrow ( $\hat{\gamma}$ )

Figure 8. Simulation results when $n_{\mathcal{E}} = 300$ . ALSB's exact $p$ -value is unavailable due to computation.   
![](images/533edc4cfcb5513c036ae0bdb8353d975382fe5170ce2453d483260e9e53eff0.jpg)

Figure 9. Selection performance of CSB ( $\hat{\gamma}$ ) when $n_{\varepsilon} = 300$ .   
![](images/b33f73cdf5d0dd2528213563c23b3736023f0f757acfa70ecee2f2c5fa59b5cf.jpg)

<details>
<summary>bar</summary>

| b   | count 0 | count 5 | count 100 | count 200 | count 300 | count 400 | count 500 | count 600 | count 700 | count 800 |
|-----|---------|---------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 0   | 1.00    | 0.95    | 0.90      | 0.85      | 0.80      | 0.75      | 0.70      | 0.65      | 0.60      | 0.55      |
| 1   | 0.75    | 0.70    | 0.65      | 0.60      | 0.55      | 0.50      | 0.45      | 0.40      | 0.35      | 0.30      |
| 2   | 0.50    | 0.45    | 0.40      | 0.35      | 0.30      | 0.25      | 0.20      | 0.15      | 0.10      | 0.05      |
| 3   | 0.25    | 0.20    | 0.15      | 0.10      | 0.05      | 0.00      | 0.05      | 0.15      | 0.25      | 0.35      |
| 4   | 0.10    | 0.15    | 0.20      | 0.25      | 0.30      | 0.35      | 0.40      | 0.45      | 0.50      | 0.60      |
| 5   | 0.15    | 0.20    | 0.25      | 0.30      | 0.35      | 0.40      | 0.45      | 0.50      | 0.55      | 0.70      |
| 6   | 0.25    | 0.30    | 0.35      | 0.40      | 0.45      | 0.50      | 0.55      | 0.60      | 0.65      | 0.80      |
| 7   | 0.35    | 0.45    | 0.50      | 0.55      | 0.60      | 0.65      | 0.70      | 0.75      | 0.80      | 1.15      |
| 8   | 1.15    | 1.25    | 1.35      | 1.45      | 1.55      | 1.65      | 1.75      | 1.85      | 1.95      | 2.25      |
</details>

Figure 10. $\hat{\gamma}$ versus $b$ when $n_{\mathcal{E}} = 300$ .

(A) Absolute Bias   
![](images/a741a24af86c407d50243db20c743b35dabbba66128f90c2d2085582008bf2d6.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Series 1 | Series 2 | Series 3 |
| ------------------------ | -------- | -------- | -------- |
| 0                        | 0.00     | 0.00     | 0.00     |
| 2                        | 0.17     | 0.10     | 0.03     |
| 4                        | 0.05     | 0.01     | 0.01     |
| 6                        | 0.01     | 0.01     | 0.01     |
| 8                        | 0.01     | 0.01     | 0.01     |
</details>

(B) Variance   
![](images/df9a1324e283f7ec0b78921219fa88ce46e27fb8ec6caab7cd19176d07fdf064.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Series 1 | Series 2 | Series 3 |
| ------------------------ | -------- | -------- | -------- |
| 0                        | 0.07     | 0.08     | 0.105    |
| 1                        | 0.085    | 0.09     | 0.105    |
| 2                        | 0.08     | 0.085    | 0.105    |
| 3                        | 0.075    | 0.08     | 0.105    |
| 4                        | 0.07     | 0.08     | 0.105    |
| 5                        | 0.07     | 0.08     | 0.105    |
| 6                        | 0.07     | 0.08     | 0.105    |
| 7                        | 0.07     | 0.08     | 0.105    |
| 8                        | 0.07     | 0.08     | 0.105    |
</details>

(C) MSE   
![](images/7aa587c9e4edc0e65b2be4ffbf22090b6e9a28d3da08c4f33085cadda5fee4b8.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Series 1 | Series 2 | Series 3 |
| ------------------------ | -------- | -------- | -------- |
| 0                        | 0.07     | 0.08     | 0.10     |
| 1                        | 0.11     | 0.10     | 0.10     |
| 2                        | 0.08     | 0.08     | 0.10     |
| 3                        | 0.07     | 0.08     | 0.10     |
| 4                        | 0.07     | 0.08     | 0.10     |
| 5                        | 0.07     | 0.08     | 0.10     |
| 6                        | 0.07     | 0.08     | 0.10     |
| 7                        | 0.07     | 0.08     | 0.10     |
| 8                        | 0.07     | 0.08     | 0.10     |
</details>

(D) Type I Error Rate   
![](images/cce33ca0a7c53d3aaf75344a8edf765bcf128a69a740e983409096903adf95ca.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Value |
| ------------------------ | ----- |
| 0                        | 0.0   |
| 1                        | 0.05  |
| 2                        | 0.03  |
| 3                        | 0.04  |
| 4                        | 0.02  |
| 5                        | 0.06  |
| 6                        | 0.01  |
| 7                        | 0.07  |
| 8                        | 0.04  |
</details>

(E) Power   
![](images/89abbf5d29fa477ff5a868793444bd7d6b8b17c50a0d1c442e2275056894ce63.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Series 1 | Series 2 | Series 3 | Series 4 |
| ------------------------ | -------- | -------- | -------- | -------- |
| 0                        | 0.58     | 0.48     | 0.48     | 0.48     |
| 2                        | 0.28     | 0.48     | 0.48     | 0.48     |
| 4                        | 0.22     | 0.52     | 0.52     | 0.52     |
| 6                        | 0.56     | 0.56     | 0.56     | 0.56     |
| 8                        | 0.58     | 0.58     | 0.58     | 0.58     |
</details>

No Borrow ( $\gamma=1$ )   
- Conformal Selective Borrow ( $\gamma = 0.8$ ) - Conformal Selective Borrow ( $\gamma = 0.4$ )

Figure 11. Simulation results for various selection threshold $\gamma$ 's when $n_{\mathcal{E}} = 300$ .

![](images/1b5945da6c2920850254a1249772898b1be1c57bf689abab46eb147747468a08.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Absolute Bias |
| ------------------------ | ------------- |
| 0                        | 0.0           |
| 1                        | 0.25          |
| 2                        | 0.4           |
| 3                        | 0.45          |
| 4                        | 0.4           |
| 5                        | 0.2           |
| 6                        | 0.05          |
| 7                        | 0.0           |
| 8                        | 0.0           |
</details>

![](images/4fafa15d3f537e14ad23558cd3760533d63dfbcff147573c01cad9d814c9ceb2.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Variance (Blue Line) | Variance (Orange Dashed Line) |
| ------------------------ | --------------------- | ------------------------------ |
| 0                        | 0.075                 | 0.105                          |
| 1                        | 0.115                 | 0.108                          |
| 2                        | 0.095                 | 0.112                          |
| 3                        | 0.080                 | 0.108                          |
| 4                        | 0.078                 | 0.102                          |
| 5                        | 0.078                 | 0.108                          |
| 6                        | 0.078                 | 0.085                          |
| 7                        | 0.078                 | 0.075                          |
| 8                        | 0.078                 | 0.072                          |
</details>

![](images/25c5f8dba4e566d1118b4c6520a97c249e1d768441721dd053ee607149215c76.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | MSE (Blue Line) | MSE (Orange Dashed Line) | MSE (Green Dotted Line) |
| ------------------------ | --------------- | ------------------------ | ----------------------- |
| 0                        | 0.05            | 0.09                     | 0.10                    |
| 1                        | 0.13            | 0.17                     | 0.10                    |
| 2                        | 0.09            | 0.29                     | 0.10                    |
| 3                        | 0.06            | 0.30                     | 0.10                    |
| 4                        | 0.05            | 0.26                     | 0.10                    |
| 5                        | 0.05            | 0.16                     | 0.10                    |
| 6                        | 0.05            | 0.09                     | 0.10                    |
| 7                        | 0.05            | 0.05                     | 0.10                    |
| 8                        | 0.05            | 0.04                     | 0.10                    |
</details>

![](images/717c69689522c0891a88fb08c43d3810ba062f1cd0ef8f1e22a1c9cec533b283.jpg)

<details>
<summary>scatter</summary>

(D) Type I Error Rate
| Magnitude of Hidden Bias | Type I Error Rate |
| :--- | :--- |
| 0 | 0.25 |
| 1 | 0.4 |
| 2 | 0.58 |
| 3 | 0.57 |
| 4 | 0.49 |
| 5 | 0.26 |
| 6 | 0.11 |
| 7 | 0.06 |
| 8 | 0.06 |
</details>

![](images/9a4ed9429aaa6d20f29692dfe167e27d4fe9e01a49fd6447e81ce9e3012c49a2.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Power (Orange Dashed Line) | Power (Green Dotted Line) |
| ------------------------ | -------------------------- | ------------------------- |
| 0                        | 0.70                       | 0.50                      |
| 2                        | 0.90                       | 0.50                      |
| 4                        | 0.95                       | 0.50                      |
| 6                        | 0.75                       | 0.50                      |
| 8                        | 0.60                       | 0.50                      |
</details>

Figure 12. Comparison of CSB + FRT and ALSB + asymptotic inference when $n_{\mathcal{E}} = 300$ .   
![](images/10d6d459e2cc111185c886bf357652b5195376a706b533aa5223d4d8295fb897.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Absolute Bias |
| ------------------------ | ------------- |
| 0                        | 0.0           |
| 1                        | 0.35          |
| 2                        | 0.5           |
| 3                        | 0.58          |
| 4                        | 0.57          |
| 5                        | 0.55          |
| 6                        | 0.52          |
| 7                        | 0.5           |
| 8                        | 0.48          |
</details>

![](images/cd74d2a12a5fceeccc905a4dcca68c74a0458835e205e1e74b9d460a2d195dbb.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Variance (Blue Line) | Variance (Red Dashed Line) |
| ------------------------ | -------------------- | -------------------------- |
| 0                        | 0.125                | 0.125                      |
| 2                        | 0.155                | 0.125                      |
| 4                        | 0.135                | 0.165                      |
| 6                        | 0.140                | 0.185                      |
| 8                        | 0.135                | 0.200                      |
</details>

![](images/857f3a0f3514f1b312018a8e5c7d53b5e1261b585030c1ef5990e3f65033062b.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | MSE (Red Dashed Line) | MSE (Blue Solid Line) |
| ------------------------ | --------------------- | --------------------- |
| 0                        | 0.0                   | 0.0                   |
| 1                        | 0.23                  | 0.16                  |
| 2                        | 0.37                  | 0.16                  |
| 3                        | 0.49                  | 0.16                  |
| 4                        | 0.49                  | 0.12                  |
| 5                        | 0.48                  | 0.12                  |
| 6                        | 0.47                  | 0.12                  |
| 7                        | 0.46                  | 0.12                  |
| 8                        | 0.45                  | 0.12                  |
</details>

![](images/34fe638cc7ffeaf692ac176ba093ea769f5a7249cbce8c3bf9a7c96f0bdd935d.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Type I Error Rate |
| ------------------------ | ----------------- |
| 0                        | 0.10              |
| 1                        | 0.05              |
| 2                        | 0.08              |
| 3                        | 0.07              |
| 4                        | 0.13              |
| 5                        | 0.06              |
| 6                        | 0.12              |
| 7                        | 0.13              |
| 8                        | 0.13              |
</details>

![](images/0d934891884baaead5941aaeefcc8b06edb214673c25d1dca968f94959cb7498.jpg)

<details>
<summary>line</summary>

| Magnitude of Hidden Bias | Blue Line | Red Dashed Line |
| ------------------------ | --------- | --------------- |
| 0                        | 0.5       | 0.5             |
| 2                        | 0.25      | 0.35            |
| 4                        | 0.2       | 0.15            |
| 6                        | 0.25      | 0.1             |
| 8                        | 0.35      | 0.1             |
</details>

Method No Borrow ( $\gamma=1$ ) Full Borrow ( $\gamma=0$ ) Conformal Selective Borrow ( $\hat{\gamma}$ )

Figure 13. Simulation results across different hidden bias magnitudes b for dependent covariates with p = 5.   
Table 2. Summary statistics of the pre-processed data. 

<table><tr><td></td><td>C9633 Treated $(n_1 = 167)$ </td><td>C9633 Controlled $(n_0 = 168)$ </td><td>NCDB Controlled $(n_\varepsilon = 335)$ </td></tr><tr><td colspan="4">Sex</td></tr><tr><td>Male</td><td>109 (65.3%)</td><td>106 (63.1%)</td><td>219 (65.4%)</td></tr><tr><td>Female</td><td>58 (34.7%)</td><td>62 (36.9%)</td><td>116 (34.6%)</td></tr><tr><td colspan="4">Age (years)</td></tr><tr><td>Mean (SD)</td><td>60.4 (10.2)</td><td>61.2 (9.28)</td><td>60.8 (9.69)</td></tr><tr><td>Median [Min, Max]</td><td>61.0 [34.0, 78.0]</td><td>62.0 [40.0, 81.0]</td><td>61.0 [34.0, 80.0]</td></tr><tr><td colspan="4">Race</td></tr><tr><td>White</td><td>151 (90.4%)</td><td>148 (88.1%)</td><td>300 (89.6%)</td></tr><tr><td>Non-white</td><td>16 (9.6%)</td><td>20 (11.9%)</td><td>35 (10.4%)</td></tr><tr><td colspan="4">Histology</td></tr><tr><td>Squamous</td><td>66 (39.5%)</td><td>65 (38.7%)</td><td>131 (39.1%)</td></tr><tr><td>Other</td><td>101 (60.5%)</td><td>103 (61.3%)</td><td>204 (60.9%)</td></tr><tr><td colspan="4">Tumor Size (cm)</td></tr><tr><td>Mean (SD)</td><td>4.60 (2.04)</td><td>4.56 (2.05)</td><td>4.77 (1.42)</td></tr><tr><td>Median [Min, Max]</td><td>4.00 [1.00, 12.0]</td><td>4.00 [1.00, 12.0]</td><td>4.50 [3.10, 12.0]</td></tr><tr><td colspan="4">Outcome: 3-year RMST*</td></tr><tr><td>Mean (SD)</td><td>2.77 (0.596)</td><td>2.64 (0.720)</td><td>2.43 (0.947)</td></tr><tr><td>Median [Min, Max]</td><td>3.00 [0.383, 3.00]</td><td>3.00 [0.181, 3.00]</td><td>3.00 [0.0242, 3.00]</td></tr></table>

\*Pseudo-observations transformed from censored survival time.

![](images/e257d2bd6ca91546755882502143b785f11fd7c7206626348553e5ec6345195e.jpg)  
Censored Yes No

Figure 14. Pseudo-observation vs. Censored Time for CALGB 9633 and NCDB datasets.

![](images/848a5e8a9149494f439f38911bc3481fb74f58818440da54ab856e0157fcc593.jpg)

<details>
<summary>bar</summary>

| Sex | Unmatched | Matched |
| --- | --- | --- |
| 0 | 0.45 | 0.35 |
| 1 | 0.55 | 0.65 |
</details>

![](images/39b6924502ebd0481c296482f6df40225f5f5685dc7735c59b918e031da5ed37.jpg)

<details>
<summary>area</summary>

| Age Range | Unmatched Density | Matched Density |
|-----------|-------------------|-----------------|
| 40-50     | ~0.01             | ~0.01           |
| 50-60     | ~0.03             | ~0.03           |
| 60-70     | ~0.04             | ~0.04           |
| 70-80     | ~0.02             | ~0.02           |
</details>

![](images/3f890efbfaa609881c1a378e9a1a6d14284adb479d9e32abbe67bf55a1d4a050.jpg)

<details>
<summary>bar</summary>

| Race | Unmatched | Matched |
| ---- | --------- | ------- |
| 0    | 0.12      | 0.12    |
| 1    | 0.85      | 0.85    |
</details>

![](images/c83c28bcbbc27ec65d39b808777128a424cfe329782c98ec062db00fdbbeb3fe.jpg)

<details>
<summary>bar</summary>

| Histology | Unmatched | Matched |
| --------- | --------- | ------- |
| 0         | 0.6       | 0.6     |
| 1         | 0.4       | 0.4     |
</details>

![](images/7632c0550436dca118de49a580676c135ce3c3ab6040bea6bceb75e5f72918cf.jpg)

<details>
<summary>area</summary>

| Tumor Size | Unmatched Density | Matched Density |
| ---------- | ----------------- | --------------- |
| 2.5        | 0.0               | 0.0             |
| 5.0        | 0.4               | 0.35            |
| 7.5        | 0.1               | 0.1             |
| 10.0       | 0.0               | 0.0             |
| 12.5       | 0.0               | 0.0             |
</details>

![](images/f008ff3f0326542bec0ccd76cd1cc604d99509b1fd63e151f43321dddab486f7.jpg)

<details>
<summary>area</summary>

| Sampling Score | Unmatched Density | Matched Density |
| -------------- | ----------------- | --------------- |
| 0.0            | 25                | 16              |
| 0.1            | 10                | 8               |
| 0.2            | 2                 | 1               |
</details>

Sample □ NCDB (S = 0) □ CALGB 9633 (S = 1)

Figure 15. Distributional balance (unmatched and matched) between CALGB 9633 ( $S = 1$ ) and NCDB ( $S = 0$ ) for baseline covariates and the estimated sampling score $\hat{\mathbb{P}}(S = 1|X)$ .   
![](images/cd91647afd3f154b8b7d2bfa88e3f2295b23c037e8d8701a255a4a17bab02cc6.jpg)  
Figure 16. 3-year RMST (Outcome) vs. Sampling Score estimated by 5 covariates. The shaded area is constructed using quantile regression on the CALGB 9633 controlled data.