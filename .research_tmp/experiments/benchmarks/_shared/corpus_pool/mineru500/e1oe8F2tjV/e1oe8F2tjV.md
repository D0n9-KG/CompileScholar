# MULTINOMIAL LOGISTIC REGRESSION: ASYMPTOTIC NORMALITY ON NULL COVARIATES IN HIGH-DIMENSIONS

KAI TAN AND PIERRE C. BELLEC

ABSTRACT. This paper investigates the asymptotic distribution of the maximum-likelihood estimate (MLE) in multinomial logistic models in the high-dimensional regime where dimension and sample size are of the same order. While classical large-sample theory provides asymptotic normality of the MLE under certain conditions, such classical results are expected to fail in high-dimensions as documented for the binary logistic case in the seminal work of Sur and Candès [2019]. We address this issue in classification problems with 3 or more classes, by developing asymptotic normality and asymptotic chi-square results for the multinomial logistic MLE (also known as cross-entropy minimizer) on null covariates. Our theory leads to a new methodology to test the significance of a given feature. Extensive simulation studies on synthetic data corroborate these asymptotic results and confirm the validity of proposed p-values for testing the significance of a given feature.

# 1. INTRODUCTION

Multinomial logistic modeling has become a cornerstone of classification problems in machine learning, as witnessed by the omnipresence of both the cross-entropy loss (multinomial logistic loss) and the softmax function (gradient of the multinomial logistic loss) in both applied and theoretical machine learning. We refer to Cramer [2002] for an account of the history and early developments of logistic modeling.

Throughout, we consider a classification problem with $K + 1$ possible labels where K is a fixed constant. This paper tackles asymptotic distributions of multinomial logistic estimates (or cross-entropy minimizers) in generalized linear models with moderately high-dimensions, where sample size n and dimension p have the same order, for instance $n, p \to +\infty$ simultaneously while the ratio p/n converges to a finite constant. Throughout the paper, let $[n] = \{1, 2, \ldots, n\}$ for all $n \in N$ , and $I\{statement\}$ be the 0-1 valued indicator function, equal to 1 if statement is true and 0 otherwise (e.g., $I\{y_i = 1\}$ in the next paragraph equals 1 if $y_i = 1$ holds and 0 otherwise).

The case of binary logistic regression. Let $\rho(t) = \log(1 + e^t)$ be the logistic loss and $\rho'(t) = 1/(1 + e^{-t})$ be its derivative, often referred to as the sigmoid function. In the current moderately-high dimensional regime where $n, p \to +\infty$ with $p/n \to \kappa > 0$ for some constant $\kappa$ , recent works [Candès and Sur, 2020, Sur and Candès, 2019, Zhao et al., 2022] provide a detailed theoretical understanding of the behavior of the logistic Maximum Likelihood Estimate (MLE) in binary logistic regression models. Observing independent observations $(x_i, y_i)_{i \in [n]}$ from a logistic model defined as $\mathbb{P}(y_i = 1|x_i) = \rho'(x_i^T\beta)$ where $x_i \sim N(\mathbf{0}, n^{-1}I_p)$ , and $\lim_{n \to \infty} \| \beta\|^2 / n = \gamma^2$ for a constant $\gamma$ for the limiting squared norm of the unknown regression vector $\beta$ . These works prove that the behavior of the MLE $\hat{\beta} = \arg\min_{b \in \mathbb{R}^p} \sum_{i=1}^{n} \rho(x_i^T b) - I\{y_i = 1\} x_i^T b$ is summarized by the solution ( $\alpha_*, \sigma*, \lambda_*$ )

of the system of three equations

$$
\left\{ \begin{array}{l l} \sigma^ {2} & = \frac {1}{\kappa^ {2}} \mathbb {E} [ 2 \rho^ {\prime} (\gamma Z _ {1}) (\lambda \rho^ {\prime} (\mathrm{prox} _ {\lambda \rho} (- \alpha \gamma Z _ {1} + \sqrt {\kappa} \sigma Z _ {2}))) ^ {2} ] \\ 0 & = \mathbb {E} [ \rho^ {\prime} (\gamma Z _ {1}) \lambda \rho^ {\prime} (\mathrm{prox} _ {\lambda \rho} (- \alpha \gamma Z _ {1} + \sqrt {\kappa} \sigma Z _ {2})) ] \\ 1 - \kappa & = \mathbb {E} [ 2 \rho^ {\prime} (\gamma Z _ {1}) / (1 + \lambda \rho^ {\prime \prime} (\mathrm{prox} _ {\lambda \rho} (- \alpha \gamma Z _ {1} + \sqrt {\kappa} \sigma Z _ {2})) ] \end{array} , \right. \tag {1.1}
$$

where $(Z_{1}, Z_{2})$ are i.i.d. $N(0,1)$ random variables and the proximal operator is defined as $\operatorname{prox}_{\lambda\rho}(z) = \arg\min_{t \in \mathbb{R}} \left\{ \lambda \rho(t) + (t - z)^{2}/2 \right\}$ . The system (1.1) characterize, among others, the following behavior of the MLE $\hat{\beta}$ : for almost any $(\gamma, \kappa)$ , the system admits a solution if and only if $\hat{\beta}$ exists with probability approaching one and in this case, $\|\hat{\beta}\|^{2}/n$ and $\|\hat{\beta} - \beta\|^{2}/n$ both have finite limits that may be expressed as simple functions of $(\alpha_{*}, \sigma_{*}, \lambda_{*})$ , and for any feature $j \in [p]$ such that $\beta_{j} = 0$ (i.e., j is a null covariate), the j-th coordinate of the MLE satisfies

$$
\hat {\beta} _ {j} \xrightarrow {\mathrm{d}} N (0, \sigma_ {*} ^ {2}).
$$

The proofs in Sur and Candès [2019] are based on approximate message passing (AMP) techniques; we refer to Berthier et al. [2020], Feng et al. [2022], Gerbelot and Berthier [2021] and the references therein for recent surveys and general results. More recently, Zhao et al. [2022] extended the result of Sur and Candès [2019] from isotropic design to Gaussian covariates with an arbitrary covariance structure: if now $x_{i} \sim N(\mathbf{0}, \Sigma)$ for some positive definite $\Sigma$ and $\lim_{n, p \to +\infty} \beta^{T} \Sigma \beta = \kappa$ , null covariates $j \in [p]$ (in the sense that $y_{i}$ is independent of $x_{ij}$ given $(x_{ik})_{k \in [p] \setminus \{j\}}$ ) of the MLE satisfy

$$
(n / \Omega_ {j j}) ^ {1 / 2} \hat {\beta} _ {j} \xrightarrow {\mathrm{d}} N (0, \sigma_ {*} ^ {2}), \tag {1.2}
$$

where $\sigma_{*}$ is the same solution of (1.1) and $\Omega = \Sigma^{-1}$ . Zhao et al. [2022] also obtained asymptotic normality results for non-null covariates, that is, features $j \in [p]$ such that $\beta_{j} \neq 0$ . The previous displays can be used to test the null hypothesis $H_{0}: y_{i}$ is independent of $x_{ij}$ given $(x_{ik})_{k \in [p] \setminus \{j\}}$ and develop the corresponding p-values if $\sigma_{*}$ is known; in this binary logistic regression model the ProbeFrontier [Sur and Candès, 2019] and SLOE Yadlowsky et al. [2021] give means to estimate the solutions $(\alpha_{*}, \sigma_{*}, \lambda_{*})$ of system (1.1) without the knowledge of $\gamma$ . Mai et al. [2019] studied the performance of Ridge regularized binary logistic regression in mixture models. Salehi et al. [2019] extended Sur and Candès [2019] to separable penalty functions. Bellec [2022] derived asymptotic normality results similar to (1.2) in single-index models including binary logistic regression without resorting to the system (1.1), showing that for a null covariate $j \in [p]$ in the unregularized case that

$$
(n / \Omega_ {j j}) ^ {1 / 2} (\hat {v} / \hat {r}) \hat {\beta} _ {j} \xrightarrow {\mathrm{d}} N (0, 1) \tag {1.3}
$$

where $\hat{v} = \frac{1}{n}\sum_{i = 1}^{n}\rho ''(x_i^T\hat{\beta}) - \rho ''(x_i^T\hat{\beta})^2 x_i^T [\sum_{l = 1}^{n}x_l\rho ''(x_l^T\hat{\beta})x_l^T ]^{-1}x_i$ is scalar and so is $\hat{r}^2 = \frac{1}{n}\sum_{i = 1}^{n}(I\{y_i = 1\} -\rho '(x_i^T\hat{\beta}))^2$ . In summary, in this high dimensional binary logistic model,

(i) The phase transition from Candès and Sur [2020] splits the $(\gamma, \kappa)$ plane into two connected components: in one component the MLE does not exist with high probability, in the other component the MLE exists and $\| \Sigma^{1/2} \hat{\beta} \|^2$ is bounded with high probability (boundedness is a consequence of the fact that $\| \Sigma^{1/2} \hat{\beta} \|^2$ or $\| \Sigma^{1/2} (\hat{\beta} - \beta) \|^2$ admit finite limits);

(ii) In the component of the $(\gamma, \kappa)$ plane where the MLE exists, for any null covariate $j \in [p]$ , the asymptotic normality results (1.2)-(1.3) holds.

Multiclass classification. The goal of this paper is to develop a theory for the asymptotic normality of the multinomial logistic regression MLE (or cross-entropy minimizer) on null covariates when the number of classes, $K+1$ , is greater than 2 and n, p are of the same order. In other words, we aim to generalize results such as (1.2) or (1.3) for three or more classes. Classification datasets with 3 or more classes are ubiquitous in machine learning (MNIST, CIFAR to name a few), which calls for such multiclass generalizations. In Gaussian mixtures and logistic models, Thrampoulidis et al. [2020] derived characterizations of the performance of least-squares and class-averaging estimators, excluding cross-entropy minimizers or minimizers of non-linear losses. Loureiro et al. [2021] extended Sur and Candès [2019], Zhao et al. [2022], Salehi et al. [2019] to multiclass classification problems in a Gaussian mixture model, and obtained the fixed-point equations that characterize the performance and empirical distribution of the minimizer of the cross-entropy loss plus a convex regularizer. In the same vein as Loureiro et al. [2021], Cornacchia et al. [2022] studied the limiting fixed-point equations in a multiclass teacher-student learning model where labels are generated by a noiseless channel with response $\arg\min_{k\in\{1,\ldots,K\}}x_{i}^{T}\beta_{k}$ where $\beta_{k}\in R^{p}$ is unknown for each class k. These two aforementioned works assume a multiclass Gaussian mixture model, which is different than the normality assumption for $x_{i}$ used in the present paper. More importantly, these results cannot be readily used for the purpose testing significant covariates (cf. (1.10) below) since solving the fixed-point equations require the knowledge of several unknown parameters, including the limiting spectrum of the mixture covariances and empirical distributions of the mixture means (cf. for instance Corollary 3 in Loureiro et al. [2021]). In the following sections, we fill this gap with a new methodology to test the significance of covariates. This is made possible by developing new asymptotic normality results for cross-entropy minimizers that generalize (1.3), without relying on the low-dimensional fixed-point equations.

Notation. Throughout, $I_p \in \mathbb{R}^{p \times p}$ is the identity matrix, for a matrix $A \in \mathbb{R}^{m \times n}$ , $A^T$ denotes the transpose of $A$ , $A^\dagger$ denotes the Moore-Penrose inverse of $A$ . If $A$ is psd, $A^{1/2}$ denotes the unique symmetric square root, i.e., the unique positive semi-definite matrix such that $(A^{1/2})^2 = A$ . The symbol $\otimes$ denotes the Kronecker product of matrices. Given two matrices $A \in \mathbb{R}^{n \times k}$ , $B \in \mathbb{R}^{n \times q}$ with the same number or rows, $(A, B) \in \mathbb{R}^{n \times (k + q)}$ is the matrix obtained by stacking the columns of $A$ and $B$ horizontally. If $v \in \mathbb{R}^n$ is a column vector with dimension equal to the number of rows in $A$ , we construct $(A, v) \in \mathbb{R}^{n \times (k + 1)}$ similarly. We use $\mathbf{0}_n$ and $\mathbf{1}_n$ to denote the all-zeros vector and all-ones vector in $\mathbb{R}^n$ , respectively; we do not bold vectors and matrices other than $\mathbf{0}_n$ and $\mathbf{1}_n$ . We may omit the subscript giving the dimension if clear from context; e.g., in $I_{K+1} - \frac{\mathbf{11}^T}{K+1}$ the vector $\mathbf{1}$ is in $\mathbb{R}^{K+1}$ . The Kronecker product between two matrices is denoted by $\otimes$ and $\mathrm{vec}(M) \in \mathbb{R}^{nd}$ is the vectorization operator applied to a matrix $M \in \mathbb{R}^{n \times d}$ . For an integer $K \geq 2$ and $\alpha \in (0, 1)$ , the quantile $\chi_K^2(\alpha)$ is the unique real number satisfying $\mathbb{P}(W > \chi_K^2(\alpha)) = \alpha$ where $W$ has a chi-square distribution with $K$ degrees of freedom. The symbols $\xrightarrow{\mathrm{d}}$ and $\xrightarrow{\mathrm{p}}$ denote convergence in distribution and in probability.

Throughout, classical asymptotic regime refers to the scenario where the feature dimension $p$ is fixed and the sample size $n$ goes to infinity. In contrast, the term high-dimensional regime refers to the situation where $n$ and $p$ both tend to infinity with the ratio $p / n$ converging to a limit smaller than 1.

1.1. Multinomial logistic regression. Consider a multinomial logistic regression model with $K+1$ classes. We have n i.i.d. data samples $\{(x_{i},y_{i})\}_{i=1}^{n}$ , where $x_{i}\in R^{p}$ is the feature vector and $y_{i}=(y_{i1},...,y_{i(K+1)})^{T}\in\mathbb{R}^{K+1}$ is the response. Each response $y_{i}$ is the one-hot encoding of a single label, i.e., $y_{i}\in\{0,1\}^{K+1}$ with $\sum_{k=1}^{K+1}y_{ik}=1$ such that $y_{ik}=1$ if and

only if the label for i-th observation is k. A commonly used generative model for $y_{i}$ is the multinomial regression model, namely

$$
\mathbb {P} (\mathbf {y} _ {i k} = 1 | x _ {i}) = \frac {\exp (x _ {i} ^ {T} \mathsf {B} ^ {*} e _ {k})}{\sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp (x _ {i} ^ {T} \mathsf {B} ^ {*} e _ {k ^ {\prime}})}, \quad k \in \{1, 2, \ldots , K + 1 \} \tag {1.4}
$$

where $\mathsf{B}^{*}\in\mathbb{R}^{p\times(K+1)}$ is an unknown logistic model parameter and $e_{k}\in\mathbb{R}^{K+1},e_{k^{\prime}}\in\mathbb{R}^{K+1}$ are the k-th and $k^{\prime}$ -th canonical basis vectors. The MLE for $B^{*}$ in the model (1.4) is any solution that minimizes the cross-entropy loss,

$$
\hat {\mathsf {B}} \in \arg \min _ {\mathsf {B} \in \mathbb {R} ^ {p \times (K + 1)}} \sum_ {i = 1} ^ {n} \mathsf {L} _ {i} (\mathsf {B} ^ {T} x _ {i}), \tag {1.5}
$$

where $L_{i}: R^{K+1} \to R$ is defined as $\mathsf{L}_{i}(\mathsf{u}) = -\sum_{k=1}^{K+1} \mathsf{y}_{ik} \mathsf{u}_{k} + \log \sum_{k'=1}^{K+1} \exp(\mathsf{u}_{k'})$ . If the solution set in (1.5) is non-empty, we define for each observation $i \in [n]$ the vector of predicted probabilities $\hat{\mathsf{p}}_{i} = (\hat{\mathsf{p}}_{i1}, ..., \hat{\mathsf{p}}_{i(K+1)})^{T}$ with

$$
\hat {\mathfrak {p}} _ {i k} \stackrel {{\text {def}}} {{:=}} \mathbb {P} (\hat {\mathbf {y}} _ {i k} = 1) = \frac {\exp (x _ {i} ^ {T} \hat {\mathsf {B}} e _ {k})}{\sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp (x _ {i} ^ {T} \hat {\mathsf {B}} e _ {k ^ {\prime}})} \qquad \text {for each} k \in \{1,..., K + 1 \}. \tag {1.6}
$$

Our results will utilize the gradient and Hessian of $L_{i}$ evaluated at $\hat{B}^{T}x_{i}$ , denoted by

$$
\mathbf {g} _ {i} \stackrel {{\text {def}}} {{:=}} \nabla \mathrm{L} _ {i} (\hat {\mathrm{B}} ^ {T} x _ {i}) = - \mathbf {y} _ {i} + \hat {\mathbf {p}} _ {i}, \qquad \mathrm{H} _ {i} \stackrel {{\text {def}}} {{:=}} \nabla^ {2} \mathrm{L} _ {i} (\hat {\mathrm{B}} ^ {T} x _ {i}) = \operatorname{diag} (\hat {\mathbf {p}} _ {i}) - \hat {\mathbf {p}} _ {i} \hat {\mathbf {p}} _ {i} ^ {T}. \tag {1.7}
$$

The quantities $(\hat{\mathsf{B}},\hat{\mathsf{p}}_{i},\mathsf{g}_{i},\mathsf{H}_{i})$ can be readily computed from the data $\{(x_{i},\mathsf{y}_{i})\}_{i=1}^{n}$ . To be specific, the MLE $\hat{B}$ in (1.5) can be obtained by invoking a multinomial regression solver (e.g., sklearn.linear\_model.LogisticRegression from Pedregosa et al. [2011]), and the quantities $\hat{p}_{i}, g_{i}, H_{i}$ can be further computed from eqs. (1.6) and (1.7) by a few matrix multiplications and application of the softmax function.

Log-odds model and reference class. The matrix $B^{*}$ in (1.4) is not identifiable since the conditional distribution of $y_{i}|x_{i}$ in the model (1.4) remains unchanged if we replace columns of $B^{*}$ by $B^{*}-b1_{K+1}^{T}$ for any $b\in R^{p}$ . In order to obtain an identifiable model, a classical and natural remedy is to model the log-odds, here with the class $K+1$ as the reference class:

$$
\log \frac {\mathbb {P} (\mathrm{y} _ {i k} = 1 | x _ {i})}{\mathbb {P} (\mathrm{y} _ {i (K + 1)} = 1 | x _ {i})} = x _ {i} ^ {T} A ^ {*} e _ {k}, \qquad \forall k \in [ K ] \tag {1.8}
$$

where $e_{k}$ is the k-th canonical basis vector of $R^{K}$ , and $A^{*} \in R^{p \times K}$ is the unknown parameter. The matrix $A^{*} \in R^{p \times K}$ in log-odds model (1.8) is related to $\mathsf{B}^{*} \in \mathbb{R}^{p \times (K+1)}$ in the model (1.4) by $A^{*} = \mathsf{B}^{*}(I_{K}, -\mathbf{1}_{K})^{T}$ . This log-odds model has two benefits: First it is identifiable since the unknown matrix $A^{*}$ is uniquely defined. Second, the matrix $A^{*}$ lends itself well to interpretation as its k-th column represents the contrast coefficient between class k and the reference class $K + 1$ .

The MLE $\hat{A}$ of $A^{*}$ in (1.8) is $\hat{A} = \arg \min_{A\in \mathbb{R}^{p\times K}}\sum_{i = 1}^{n}\mathsf{L}_{i}((A,\mathbf{0}_{p})^{T}x_{i})$ . If the solution set in (1.5) is non-empty, $\hat{A}$ is related to any solution $\hat{\mathsf{B}}$ in (1.5) by $\hat{A} = \hat{\mathsf{B}} (I_K, - \mathbf{1}_K)^T$ . Equivalently,

$$
\hat {A} _ {j k} = \hat {\mathsf {B}} _ {j k} - \hat {\mathsf {B}} _ {j (K + 1)} \tag {1.9}
$$

for each $j \in [p]$ and $k \in [K]$ .

If there are three classes (i.e. $K + 1 = 3$ ), this parametrization allows us to draw scatter plots of realizations of $\sqrt{n}e_{j}^{T}\hat{A} = (\sqrt{n}\hat{A}_{j1}, \sqrt{n}\hat{A}_{j,2})$ as in Figure 1.

1.2. Hypothesis testing for the j-th feature and classical asymptotic normality for MLE.

Hypothesis testing for the j-th feature. Our goal is to develop a methodology to test the significance of the j-th feature. Specifically, for a desired confidence level $(1-\alpha)\in(0,1)$ (say, $1-\alpha=0.95$ ) and a given feature $j\in[p]$ of interest, our goal is to test

(1.10) $H_0:y_i$ is conditionally independent of $x_{ij}$ given $(x_{ij'})_{j'\in [p]\setminus \{j\}}$ .

Namely, we want to test whether the $j$ -th variable is independent from the response given all other explanatory variables $(x_{ij'}, j' \in [p] \setminus \{j\})$ . Assuming normally distributed $x_i$ and a multinomial model as in (1.4) or (1.8), it is equivalent to test

(1.11) $H_0:e_j^T A^* = \mathbf{0}_K^T$ versus $H_{1}:e_{j}^{T}A^{*}\neq \mathbf{0}_{K}^{T},$

where $e_{j} \in R^{p}$ is the j-th canonical basis vector.

If the MLE $\hat{B}$ in (1.5) exists in the sense that the solution set in (1.5) is nonempty, the conjecture that rejecting $H_{0}$ when $e_{j}^{T}\hat{B}$ is far from $O_{K+1}$ is a reasonable starting point. The important question, then, is to determine a quantitative statement for the informal “far from $O_{K+1}$ ”, similarly to (1.2) or (1.3) in binary logistic regression.

Classical theory with $p$ fixed. If $p$ is fixed and $n \to \infty$ in model (1.8), classical maximum likelihood theory [Van der Vaart, 1998, Chapter 5] provides the asymptotic distribution of the MLE $\hat{A}$ , which can be further used to test (1.11).

Briefly, if $x$ has the same distribution as any $x_{i}$ , the MLE $\hat{A}$ in the multinomial logistic model is asymptotically normal with

$$
\sqrt {n} (\mathrm{vec} (\hat {A}) - \mathrm{vec} (A ^ {*})) \stackrel {\mathrm{d}} {\longrightarrow} N (\mathbf {0}, \mathcal {I} ^ {- 1}) \quad \mathrm{where} \quad \mathcal {I} = \mathbb {E} [ (x x ^ {T}) \otimes (\mathrm{diag} (\pi^ {*}) - \pi^ {*} \pi^ {* T}) ]
$$

is the Fisher information matrix evaluated at the true parameter $A^{*}$ , $\operatorname{vec}(\cdot)$ is the usual vectorization operator, and $\pi^{*} \in R^{K}$ has random entries

$$
\pi_ {k} ^ {*} = \exp (x ^ {T} A ^ {*} e _ {k}) / (1 + \sum_ {k ^ {\prime} = 1} ^ {K} \exp (x ^ {T} A ^ {*} e _ {k ^ {\prime}}))
$$

for each $k \in [K]$ . In particular, under $H_0: e_j^T A^* = \mathbf{0}_K^T$ ,

(1.12) $\sqrt{n}\hat{A}^T e_j\xrightarrow{\mathrm{d}}N(\mathbf{0},S_j)$

where $S_{j} = (e_{j}^{T}\otimes I_{K})\mathcal{I}^{-1}(e_{j}\otimes I_{K}) = e_{j}^{T}(\mathrm{cov}(x))^{-1}e_{j}[\mathbb{E}\left(\mathrm{diag}(\pi^{*}) - \pi^{*}\pi^{*T}\right)]^{-1}$ . When (1.12) holds, by the delta method we also have $\sqrt{n} S_j^{-1/2}\hat{A}^T e_j\xrightarrow{\mathrm{d}} N(\mathbf{0},I_K)$ and

(1.13) $n\| S_j^{-1 / 2}\hat{A}^T e_j\| ^2\xrightarrow{\mathrm{d}}\chi_K^2.$

where the limiting distribution is chi-square with K degrees of freedom. This further suggests the size $\alpha$ test that rejects $H_{0}$ when $T_{n}^{j}(X,Y) > \chi_{K}^{2}(\alpha)$ , where $T_{n}^{j}(X,Y) = n\|S_{j}^{-1/2}\hat{A}^{T}e_{j}\|^{2}$ is the test statistic. If (1.13) holds, this test is guaranteed to have a type I error converging to $\alpha$ . The p-value of this test is given by

(1.14) $\int_{T_n^j (X,Y)}^{+\infty}f_{\chi_K^2}(t)dt,$

where $f_{\chi_K^2}(\cdot)$ is the density of the chi-square distribution with $K$ degrees of freedom.

As discussed in the introduction, Sur and Candès [2019] showed that in binary logistic regression, classical normality results for the MLE such as (1.12) fail in the high-dimensional regime because the variance in (1.12) underestimates the variability of the MLE even for null covariates; see also the discussion surrounding (1.2). Our goal is to develop, for classification problems with $K + 1 \geq 3$ classes, a theory that correctly characterize the asymptotic distribution of $\hat{A}^T e_j$ for a null covariate $j \in [p]$ in the high-dimensional regime.

We present first some motivating simulations that demonstrate the failure of classical normal approximation (1.12) in finite samples. These simulations are conducted for

various configurations of $(n,p)$ with $K+1=3$ classes. We fix the true parameter $A^{*}$ and obtain 1000 realizations of $(\hat{A}_{j1},\hat{A}_{j2})$ by independently resampling the data $\{(x_{i},y_{i})\}_{i=1}^{n}$ 1000 times. If the result (1.12) holds, then $\mathbb{P}(\sqrt{n}\hat{A}^{T}e_{j}\in\mathcal{C}_{\alpha}^{j})\to1-\alpha$ , where $\mathcal{C}_{\alpha}^{j}=\{u\in\mathbb{R}^{K}:\|S_{j}^{-1/2}u\|\leq\chi_{K}^{2}(\alpha)\}$ . Figure 1 displays scatter plots of $\sqrt{n}(\hat{A}_{j1},\hat{A}_{j2})$ along with the boundary of 95% confidence set $C_{\alpha}^{j}$ with $\alpha=0.05$ . We observe that, across the three different configurations of $(n,p)$ , the 95% confidence sets from our theory (Theorem 2.2 presented in next section) cover around 95% of the realizations, while the set $C_{\alpha}^{j}$ from classical theory only covers approximately 30% of the points, which is significantly lower than the desired coverage rate of 95%. Intuitively and by analogy with results in binary classification [Sur and Candès, 2019], this is because the classical theory (1.12) underestimates the variation of the MLE in the high-dimensional regime. Motivated by this failure of classical MLE theory and the results in binary classification [Sur and Candès, 2019, among others], the goal of this paper is to develop a theory for multinomial logistic regression that achieves the following objectives:

- Establish asymptotic normality of the multinomial MLE $\hat{A}^T e_j$ for null covariates as $n, p \to +\infty$ simultaneously with a finite limit for $n/p$ .   
- Develop a valid methodology for hypothesis testing of (1.10) in this regime, i.e., testing for the presence of an effect of a feature $j \in [p]$ on the multiclass response.

The contribution of this paper is two-fold: (i) For a null covariate $j \in [p]$ , we establish asymptotic normality results for $\hat{A}^{T}e_{j}$ that are valid in the high-dimensional regime where n and p have the same order; (ii) we propose a user-friendly test for assessing the significance of a feature in multiclass classification problems.

![](images/02370247c2e77d2d93a798f8686ec6df18f7077ea55fb58bf90cfba0bf5c75fd.jpg)

<details>
<summary>scatter</summary>

| x    | y    | class     |
| ---- | ---- | --------- |
| -10  | 5    | classical  |
| 5    | 0    | classical  |
| 15   | -5   | classical  |
| -5   | -10  | classical  |
| 0    | -15  | classical  |
| 10   | -20  | classical  |
| -15  | -15  | modern    |
| 0    | -10  | modern    |
| 10   | -5   | modern    |
| -5   | 0    | modern    |
| 5    | 5    | modern    |
| 15   | 10   | modern    |
| -10  | 10   | modern    |
| 0    | 15   | modern    |
| 10   | 20   | modern    |
| -5   | 20   | modern    |
| 5    | 25   | modern    |
| 15   | 30   | modern    |
| -10  | -20  | modern    |
| 0    | -25  | modern    |
| 10   | -30  | modern    |
| -5   | -30  | modern    |
| 5    | -35  | modern    |
| 15   | -40  | modern    |
| -10  | -40  | modern    |
| 0    | -45  | modern    |
| 10   | -50  | modern    |
| -5   | -50  | modern    |
| 5    | -55  | modern    |
| 15   | -60  | modern    |
| -10  | -60  | modern    |
| 0    | -65  | modern    |
| 10   | -70  | modern    |
| -5   | -70  | modern    |
| 5    | -75  | modern    |
| 15   | -80  | modern    |
| -10  | -80  | modern    |
| 0    | -85  | modern    |
| 10   | -90  | modern    |
| -5   | -90  | modern    |
| 5    | -95  | modern    |
| 15   | -100 | modern    |
| -10  | -100 | modern    |
| 0    | -105 | modern    |
| 10   | -110 | modern    |
| -5   | -110 | modern    |
| 5    | -115 | modern    |
| 15   | -120 | modern    |
| -10  | -120 | modern    |
| 0    | -125 | modern    |
| 10   | -130 | modern    |
| -5   | -130 | modern    |
| 5    | -135 | modern    |
| 15   | -140 | modern    |
| -10  | -140 | modern    |
| 0    | -145 | modern    |
| 10   | -150 | modern    |
| -5   | -150 | modern    |
| 5    | -155 | modern    |
| 15   | -160 | modern    |
| -10  | -160 | modern    |
| 0    | -165 | modern    |
| 10   | -170 | modern    |
| -5   | -170 | modern    |
| 5    | -175 | modern    |
| 15   | -180 | modern    |
| -10  | -180 | modern    |
| 0    | -185 | modern    |
| 10   | -190 | modern    |
| -5   | -190 | modern    |
| 5    | -195 | modern    |
| 15   | -200 | modern    |
| -10  | -200 | modern    |
| 0    | -205 | modern    |
| 10   | -210 | modern    |
| -5   | -210 | modern    |
| 5    | -215 | modern    |
| 15   | -220 | modern    |
| -10  | -220 | modern    |
| 0    | -225 | modern    |
| 10   | -230 | modern    |
| -5   | -230 | modern    |
| 5    | -235 | modern    |
| 15   | -240 | modern    |
| -10  | -240 | modern    |
| 0    | -245 | modern    |
| 10   | -250 | modern    |
| -5   | -250 | modern    |
| 5    | -255 | modern    |
| 15   | -260 | modern    |
| -10  | -260 | modern    |
| 0    | -265 | modern    |
| 10   | -270 | modern    |
| -5   | -270 | modern    |
| 5    | -275 | modern    |
| 15   | -280 | modern    |
| -10  | -280 | modern    |
| 0    | -285 | modern    |
| 10   | -290 | modern    |
| -5   | -290 | modern    |
| 5    | -295 | modern    |
| 15   | -300 | modern    |
| -10  | -300 | modern    |
| 0    | -305 | modern    |
| 10   | -310 | modern    |
| -5   | -310 | modern    |
| 5    | -315 | modern    |
| 15   | -320 | modern    |
| -10  | -320 | modern    |
| 0    | -325 | modern    |
| 10   | -330 | modern    |
| -5   | -330 | modern    |
| 5    | -335 | modern    |
| 15   | -340 | modern    |
| -10  | -340 | modern    |
| 0    | -345 | modern    |
| 10   | -350 | modern    |
| -5   | -350 | modern    |
| 5    | -355 | modern    |
| 15   | -360 | modern    |
| -10  | -360 | modern    |
| 0    | -365 | modern    |
| 10   | -370 | modern    |
| -5   | -370 | modern    |
| 5    | -375 | modern    |
| 15   | -380 | modern    |
| -10  | -380 | modern    |
| 0    | -385 | modern    |
| 10   | -390 | modern    |
| -5   | -390 | modern    |
| 5    | -395 | modern    |
| 15   | -400 | modern    |
| -10  | -400 | modern    |
| 0    | -405       | modern      |
| 10   | -410       | modern      |
| -5   | -410       | modern      |
| 5    | -415       | modern      |
| 15   | -420       | modern      |
| -10  | -420       | modern      |
| 0    | -425       | modern      |
| 10   | -430       | modern      |
| -5   | -430       | modern      |
| 5    | -435       | modern      |
| 15   | -440       | modern      |
| -10  | -440       | modern      |
| 0    | -445       | modern      |
| 10   | -450       | modern      |
| -5   | -450       | modern      |
| 5    | -455       | modern      |
| 15   | -460       | modern      |
| -10  | -460       | modern      |
| 0    | -465       | modern      |
| 10   | -470       | modern      |
| -5   | -470       | modern      |
| 5    | -475       | modern      |
| 15   | -480       | modern      |
| -10  | -480       | modern      |
| 0    \text{mod}^* (Coverage=3.6%) are estimated based on visual scale for comparison. The data provided in the image is a sample of those values. The actual data points are not explicitly labeled in the code. The chart type is a scatter plot with a color legend. The labels above the plots indicate 'classical' or 'modern'.
</details>

(A) $(n,p)=(2000,600)$

![](images/593d24d1d2f377b89c953e454b6d40e5b384f847e30e913a1fbd878cdb46f02e.jpg)

<details>
<summary>scatter</summary>

| Class       | Coverage |
|-------------|----------|
| classical   | 35.5%    |
| modern      | 95.3%    |
</details>

(B) $(n,p) = (3500,1000)$

![](images/2b0dcca1c4dcc0f1d0b7fc8462bf5babe1bbc187ec43cb24756834b68dcc0e2b.jpg)

<details>
<summary>scatter</summary>

| Model Type | Coverage |
|------------|----------|
| classical  | 28.8%    |
| modern     | 95.1%    |
</details>

(C) $(n,p) = (5000,1500)$   
FIGURE 1. Scatter plot of pairs $(\sqrt{n}\hat{A}_{j1},\sqrt{n}\hat{A}_{j2})$ with K=2 over 1000 repetitions. The blue ellipsoid is the boundary of the 95% confidence set for $\sqrt{n}\hat{A}^{T}e_{j}$ under $H_{0}$ from the classical MLE theory (1.12)-(1.13) based on the Fisher information, the dashed red ellipsoids are the boundaries of the 95% confidence set for $\sqrt{n}\hat{A}^{T}e_{j}$ under $H_{0}$ from this paper (cf. (2.3) below). Each of the 1000 repetition gives a slightly different dashed ellipsoid. The solid red ellipsoid is the average of these 1000 dashed ellipsoids. Each row of X is i.i.d. sampled from $N(\mathbf{0},\Sigma)$ with $\Sigma=(0.5^{|i-j|})_{p\times p}$ . The first $\lceil p/4\rceil$ rows of $A^{*}$ are i.i.d. sampled from $N(\mathbf{0},I_{K})$ while other rows are set to zeros. We further normalize $A^{*}$ such that $A^{*T}\Sigma A^{*}=I_{K}$ . The last coordinate j=p is used as the null coordinate.

2. MAIN RESULT: ASYMPTOTIC NORMALITY OF $\hat{\mathbf{B}}^T e_j$ AND $\hat{A}^T e_j$ ON NULL COVARIATES

In this section, we present the main theoretical results of our work and discuss their significance. We work under the following assumptions.

Assumption 2.1. For constants $\delta > 1$ , assume that $n, p \to \infty$ with $p / n \leq \delta^{-1}$ , and that the design matrix $X \in \mathbb{R}^{n \times p}$ has $n$ i.i.d. rows $(x_i)_{i \in [n]} \sim N(\mathbf{0}, \Sigma)$ for some invertible $\Sigma \in \mathbb{R}^{p \times p}$ . The observations $(x_i, y_i)_{i \in [n]}$ are i.i.d. and each $y_i$ is of the form $y_i = f(U_i, x_i^T B^*)$ for some deterministic function $f$ , deterministic matrix $B^* \in \mathbb{R}^{p \times (K + 1)}$ such that $B^* 1_{K + 1} = 0_p$ , and latent random variable $U_i$ independent of $x_i$ .

Assumption 2.2 (One-hot encoding). The response matrix Y is in $\mathbb{R}^{n\times(K+1)}$ . Its i-th row $y_{i}$ is a one-hot encoded vector, that is, valued in $\{0,1\}^{K+1}$ with $\sum_{k=1}^{K+1}y_{ik}=1$ for each $i\in[n]$ .

The model $y_{i} = f(U_{i}, x_{i}^{T} \mathsf{B}^{*})$ for some deterministic f and $B^{*}$ and latent random variable $U_{i}$ in Assumption 2.1 is more general than a specific generative model such as the multinomial logistic conditional probabilities in (1.4), as broad choices for f are allowed. In words, the model $y_{i} = f(U_{i}, x_{i}^{T} \mathsf{B}^{*})$ with $B^{*} \mathbf{1}_{K+1} = 0_{p}$ means that $y_{i}$ only depends on $x_{i}$ through a K dimensional projection of $x_{i}$ (the projection on the row-space of $B^{*}$ ). The assumption $p/n \leq \delta^{-1}$ is more general than assuming a fixed limit for the ratio p/n; this allows us to cover low-dimensional settings satisfying $p/n \to 0$ as well.

The following assumption requires the labels to be “balanced”: we observe each class at least $\gamma n$ times for some constant $\gamma > 0$ . If $(\mathbf{y}_{i})_{i\in[n]}$ are i.i.d. as in Assumption 2.1 with distribution independent of n, p, by the law of large numbers this assumption is equivalent to $\min_{k\in[K+1]}\mathbb{P}(\mathbf{y}_{ik}=1) > 0$ .

Assumption 2.3. There exists a constant $\gamma\in(0,\frac{1}{K+1}]$ , such that for each $k\in[K+1]$ , with probability approaching one at least $\gamma n$ observations $i\in[n]$ are such that $y_{ik}=1$ . In other words, $\mathbb{P}(\sum_{i=1}^{n}I(y_{ik}=1)\geq\gamma n)\to1$ for each $k\in[K+1]$ .

As discussed in item list (i) on page 2, in binary logistic regression, Candès and Sur [2020], Sur and Candès [2019] show that the plane $\left(\frac{p}{n},\|\Sigma^{1/2}\beta^{*}\|\right)$ is split by a smooth curve into two connected open components: in one component the MLE does not exist with high probability, while in the other component, with high probability the MLE exists and is bounded in the sense that $\|\Sigma^{1/2}\hat{\beta}\|^{2}<\tau'$ or equivalently $\frac{1}{n}\|X\hat{\beta}\|^{2}<\tau$ for constants $\tau,\tau'$ independent of n,p. The next assumption requires the typical situation of the latter component, in the current multiclass setting: $\hat{B}$ in (1.5) exists in the sense that the minimization problem has solutions, and at least one solution is bounded.

Assumption 2.4. Assume $\mathbb{P}(\hat{\mathsf{B}}$ exists and $\| X\hat{\mathsf{B}}(I_{K + 1} - \frac{\mathbf{11}^T}{K + 1})\| _F^2\leq n\tau)\to 1$ as $n,p\rightarrow +\infty$ for some large enough constant $\tau$ .

Note that the validity of Assumption 2.4 can be assessed using the data at hand; if a multinomial regression solver (e.g. sklearn.linear\_model.LogisticRegression) converges and $\frac{1}{n}\| X\hat{\mathbf{B}}(I_{K + 1} - \frac{\mathbf{11}^T}{K + 1})\| _F^2$ is no larger than a predetermined large constant $\tau$ , then we know Assumption 2.4 holds. Otherwise the algorithm does not converge or produces an unbounded estimate: we know Assumption 2.4 fails to hold and we need collect more data.

Our first main result, Theorem 2.1, provides the asymptotic distribution of $\hat{\mathbf{B}}^T e_j$ where $j\in [p]$ is a null covariate, where $\hat{\mathbf{B}}$ is any minimizer $\hat{\mathbf{B}}$ of (1.5). Throughout, we denote by $\Omega$ the precision matrix defined as $\Omega = \Sigma^{-1}$ .

Theorem 2.1. Let Assumptions 2.1 to 2.4 be fulfilled. Then for any $j \in [p]$ such that $H_0$ in (1.10) holds, and any minimizer $\hat{\mathsf{B}}$ of (1.5), we have (2.1)

$$
\underbrace {\sqrt {\frac {n}{\Omega_ {j j}}}} _ {s c a l a r} \Big (\underbrace {\Big (\frac {1}{n} \sum_ {i = 1} ^ {n} (\mathsf {y} _ {i} - \hat {\mathsf {p}} _ {i}) (\mathsf {y} _ {i} - \hat {\mathsf {p}} _ {i}) ^ {T} \Big) ^ {1 / 2}} _ {s q u a r e r o o t p s e u d o - i n v e r s e \mathbb {R} ^ {(K + 1) \times (K + 1)}} \Big) ^ {\dagger} \underbrace {\Big (\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {V} _ {i} \Big)} _ {\mathbb {R} ^ {(K + 1) \times (K + 1)}} \underbrace {\hat {\mathsf {B}} ^ {T} e _ {j}} _ {\mathbb {R} ^ {K + 1}} \xrightarrow {\mathrm{d}} N \Big (\mathbf {0}, \underbrace {I _ {K + 1} - \frac {\mathbf {1 1} ^ {T}}{K + 1}} _ {c o v. \mathbb {R} ^ {(K + 1) \times (K + 1)}} \Big),
$$

where $\mathsf{V}_i = \mathsf{H}_i - (\mathsf{H}_i\otimes x_i^T)[\sum_{l = 1}^n\mathsf{H}_l\otimes (x_lx_l^T)]^\dagger (\mathsf{H}_i\otimes x_i)$ .

The proof of Theorem 2.1 is given in Supplementary Section S3. Theorem 2.1 establishes that under $H_0$ , $\hat{\mathsf{B}}^T e_j$ converges to a singular multivariate Gaussian distribution in $\mathbb{R}^{K+1}$ . In (2.1), the two matrices $\frac{1}{n} \sum_{i=1}^{n} (\mathbf{y}_i - \hat{\mathbf{p}}_i)(\mathbf{y}_i - \hat{\mathbf{p}}_i)^T$ and $\frac{1}{n} \sum_{i=1}^{n} \mathsf{V}_i$ are symmetric with kernel being the linear span of $\mathbf{1}_{K+1}$ , and similarly, if a solution exists, we may replace $\hat{\mathsf{B}}$ by $\hat{\mathsf{B}}(I_{K+1} - \frac{\mathbf{11}^T}{K+1})$ which is also solution in (1.5). In this case, all matrix-matrix and matrix-vector multiplications, matrix square root and pseudo-inverse in (2.1) happen with row-space and column space contained in the orthogonal component of $\mathbf{1}_{K+1}$ , so that the limiting Gaussian distribution in $\mathbb{R}^{K+1}$ is also supported on this $K$ -dimensional subspace.

Since the distribution of the left-hand side of (2.1) is asymptotically pivotal for all null covariates $j \in [p]$ , Theorem 2.1 opens the door of statistical inference for multinomial logistic regression in high-dimensional settings. By construction, the multinomial logistic estimate $\hat{A} \in R^{p \times K}$ in (1.9) ensures $(\hat{A}, \mathbf{0}_{p})$ is a minimizer of (1.5). Therefore, we can deduce the following theorem from Theorem 2.1.

Theorem 2.2. Define the matrix $R = (I_K, \mathbf{0}_K)^T \in \mathbb{R}^{(K+1) \times K}$ using block matrix notation. Let Assumptions 2.1 to 2.4 be fulfilled. For $\hat{A}$ in (1.9) and any $j \in [p]$ such that $H_0$ in (1.10) holds, (2.2)

$$
\underbrace {\Big (I _ {K} + \frac {\mathbf {1} _ {K} \mathbf {1} _ {K} ^ {T}}{\sqrt {K + 1} + 1} \Big) R ^ {T}} _ {m a t r i x \mathbb {R} ^ {K \times (K + 1)}} \underbrace {\sqrt {\frac {n}{\Omega_ {j j}}}} _ {s c a l a r} \Big (\underbrace {\Big (\frac {1}{n} \sum_ {i = 1} ^ {n} \mathfrak {g} _ {i} \mathfrak {g} _ {i} ^ {T} \Big) ^ {1 / 2}} _ {m a t r i x \mathbb {R} ^ {(K + 1) \times (K + 1)}} \Big) ^ {\dagger} \underbrace {\Big (\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {V} _ {i} R \Big)} _ {\mathbb {R} ^ {(K + 1) \times K}} \underbrace {\hat {A} ^ {T} e _ {j}} _ {\mathbb {R} ^ {K}} \xrightarrow {\mathrm{d}} N (\mathbf {0} _ {K}, I _ {K})
$$

where $\mathbf{g}_i$ is defined in (1.7) and $\mathsf{V}_i$ is defined in Theorem 2.1. Furthermore, for the same $j\in [p]$ , (2.3)

$$
\mathcal {T} _ {n} ^ {j} (X, Y) \stackrel {{d e f}} {{:=}} \frac {n}{\Omega_ {j j}} \Big \| \Big (\Big (\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {g} _ {i} \mathsf {g} _ {i} ^ {T} \Big) ^ {1 / 2} \Big) ^ {\dagger} \Big (\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {V} _ {i} \Big) R \hat {A} ^ {T} e _ {j} \Big \| ^ {2} s a t i s f i e s \mathcal {T} _ {n} ^ {j} (X, Y) \xrightarrow {\mathrm{d}} \chi_ {K} ^ {2}.
$$

Theorem 2.2 is proved in Supplementary Section S4. To the best of our knowledge, Theorem 2.2 is the first result that characterizes the distribution of null MLE coordinate $\hat{A}^T e_j$ in high-dimensional multinomial logistic regression with 3 or more classes. It is worth mentioning that the quantities $(\mathbf{g}_i, \mathsf{V}_i, \hat{A})$ used in Theorem 2.2 can be readily computed from the data $(X, Y)$ . Therefore, Theorem 2.2 lets us test the significance of a specific feature: for testing $H_0$ , this theorem suggests the test statistic $\mathcal{T}_n^j(X, Y)$ in (2.3) and the rejection region $\mathcal{E}_\alpha^j \stackrel{\mathrm{def}}{:=}\left\{(X, Y) : \mathcal{T}_n^j(X, Y) \geq \chi_K^2(\alpha)\right\}$ . Under the null hypothesis $H_0$ in (1.10), Theorem 2.2 guarantees $\mathbb{P}\big((X, Y) \in \mathcal{E}_\alpha^j\big) \to \alpha$ . In other words, the test that rejects $H_0$ if $(X, Y) \in \mathcal{E}_\alpha^j$ has type I error converging to $\alpha$ . The p-value of this test is

$$
\text { p - value } = \int_ {\mathcal {T} _ {n} ^ {j} (X, Y)} ^ {+ \infty} f _ {\chi_ {K} ^ {2}} (t) d t, \tag {2.4}
$$

where $f_{\chi_K^2}(\cdot)$ is the density of the chi-square distribution with $K$ degrees of freedom.

Unknown $\Omega_{jj}=e_{j}^{T}\Sigma^{-1}e_{j}$ . If $\Sigma$ is unknown, we describe a consistent estimate of the quantity $\Omega_{jj}$ appearing in (2.1), (2.2), and (2.3). Under the Gaussian Assumption 2.1, the quantity $\Omega_{jj}$ is the reciprocal of the conditional variance $\operatorname{Var}(x_{ij}|x_{i,-j})$ , which is also the noise variance in the linear model of regressing $Xe_{j}$ onto $X_{-j}$ (the submatrix of X excluding the j-th column). According to standard results in linear models, we have $\Omega_{jj}\|[I_{n}-X_{-j}(X_{-j}^{T}X_{-j})^{-1}X_{-j}^{T}]Xe_{j}\|^{2}\sim\chi_{n-p+1}^{2}$ . Since $\chi_{n-p+1}^{2}/(n-p+1)\to1$ almost surely by the strong law of large numbers,

$$
\hat {\Omega} _ {j j} = (n - p + 1) \big / \| [ I _ {n} - X _ {- j} (X _ {- j} ^ {T} X _ {- j}) ^ {- 1} X _ {- j} ^ {T} ] X e _ {j} \| ^ {2} \tag {2.5}
$$

is a consistent estimator of $\Omega_{jj}$ . Therefore, the previous asymptotic results in Theorems 2.1 and 2.2 still hold by Slutsky's theorem if we replace $\Omega_{jj}$ by the estimate $\hat{\Omega}_{jj}$ in (2.5).

# 3. NUMERICAL EXPERIMENTS

This section presents simulations to examine finite sample properties of the above results and methods.

# Simulation settings.

We set p = 1000 and consider different combinations of $(n, K)$ . The covariance matrix $\Sigma$ is specified to be the correlation matrix of an AR(1) model with parameter $\rho = 0.5$ , that is, $\Sigma = (0.5^{|i-j|})_{p \times p}$ . We generate the regression coefficients $A^{*} \in R^{p \times K}$ once and for all as follows: sample $A_{0} \in R^{p \times K}$ with first $\lceil p/4 \rceil$ rows being i.i.d. $N(\mathbf{0}, I_{K})$ , and set the remaining rows to 0. We then scale the coefficients by defining $A^{*} = A_{0}(A_{0}^{T} \Sigma A_{0})^{-1/2}$ so that $A^{*T} \Sigma A^{*} = I_{K}$ . With this construction, the p-th variable is always a null covariate, and we use this null coordinate j = p to demonstrate the effectiveness of our theoretical results presented in Theorem 2.2 and the suggested test for testing $H_{0}$ as described in (1.10). Using the above settings, we generate the design matrix $X \in R^{n \times p}$ from $N(0, \Sigma)$ , and then simulate the labels from a multinomial logistic model as given in (1.8), using the coefficients $A^{*} \in R^{p \times K}$ . For each simulation setting, we perform 5,000 repetitions.

Assessment of $\chi^{2}$ approximations. To assess the $\chi^{2}$ approximation (2.3) from this paper and that of the classical theory (1.13), we compute the two $\chi_{K}^{2}$ test statistics for each sample $(x_{i},y_{i})_{i=1}^{n}$ . Figure 2 shows the empirical quantiles of the two statistics versus the $\chi_{K}^{2}$ distribution quantiles. The results demonstrate that the quantiles (in blue) from our high-dimensional theory closely match the 45-degree line (in red), whereas the quantiles (in orange) from the classical theory significantly deviate from the 45-degree line. These findings highlight the accuracy of our proposed $\chi^{2}$ approximation (2.3) over the classical result (1.13) when p is not sufficiently small compared to n.

Uniformity of null p-values. Recall that the p-value from the classical test (1.13) is given by (1.14), while the p-value from this paper taking into account high-dimensionality is given by (2.4). Figure 3 displays the histograms of these two sets of p-values out of 5000 repetitions. The results in Figure 3 show that the p-values obtained from the classical test deviate significantly from the uniform distribution, with a severe inflation in the lower tail. This indicates that the classical test tends to produce large type I errors due to the excess of p-values close to 0. In contrast, the p-values proposed in this paper exhibit a uniform distribution, further confirming the effectiveness and applicability of the theory in Theorem 2.2 for controlling type I error when testing for null covariates with (1.10).

Unknown $\Omega_{jj}$ . In the situation where the covariance matrix $\Sigma$ is unknown, we can estimate the diagonal element $\Omega_{jj} = e_{j}^{T}\Sigma^{-1}e_{j}$ by $\hat{\Omega}_{jj}$ defined in (2.5). To evaluate the accuracy of the normal and chi-square approximations and the associated test with $\Omega_{jj}$ replaced by $\hat{\Omega}_{jj}$ , we conduct simulations similar to those in Figures 2 and 3, but we replace $\Omega_{jj}$ with its estimate $\hat{\Omega}_{jj}$ . The results are presented in Figure S1. The plots are

![](images/5c368e1bd46436f12067534a83b991079725c8e6c2ff4eeab2ed2abadbcbb428.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | sample quantiles (classical) | sample quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0.0                   | 0.0                          | 0.0                       |
| 2.5                   | 12.5                         | 2.5                       |
| 5.0                   | 12.5                         | 5.0                       |
| 7.5                   | 12.5                         | 7.5                       |
| 10.0                  | 12.5                         | 10.0                      |
| 12.5                  | 12.5                         | 12.5                      |
</details>

(A) $(n,K) = (4000,2)$

![](images/6257ff3c7b3891df552cebe61f3f4775717ebce7cf8fa45164289b260a142682.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | sample quantiles (classical) | sample quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0                     | 0                            | 0                         |
| 5                     | 16                           | 6                         |
| 10                    | 12                           | 10                        |
| 15                    | 16                           | 14                        |
</details>

(B) $(n,K) = (5000,3)$

![](images/2656355561ab119e92268a2ce01edbd0af60b2207b5fcfa2ffe1f029c6c5fecf.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | sample quantiles (classical) | sample quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0                     | 0.0                          | 0.0                       |
| 5                     | 17.5                         | 5.0                       |
| 10                    | 17.5                         | 10.0                      |
| 15                    | 17.5                         | 15.0                      |
| 20                    | 17.5                         | 17.5                      |
</details>

(C) $(n,K) = (6000,4)$

FIGURE 2. Q-Q plots of the test statistic in the left-hand side of (1.13) (in orange) and in the left-hand side of (2.3) (in blue) for different $(n, K)$ and p = 1000.   
![](images/df6cf8ad4671f16ba781037ede0cd116ef6f707897754c621b5b505ebe89f36c.jpg)

<details>
<summary>bar</summary>

| p-values | classical | modern |
|---|---|---|
| 0.0 | 2800 | 500 |
| 0.1 | 500 | 450 |
| 0.2 | 350 | 500 |
| 0.3 | 250 | 450 |
| 0.4 | 200 | 450 |
| 0.5 | 150 | 450 |
| 0.6 | 150 | 450 |
| 0.7 | 150 | 500 |
| 0.8 | 150 | 500 |
| 0.9 | 100 | 500 |
| 1.0 | 50 | 450 |
</details>

(A) $(n,K) = (4000,2)$

![](images/4c24b74521757ff7883f6e9da61ea2e87789606cf78f11e0482bc45eaf83cb7d.jpg)

<details>
<summary>bar</summary>

| p-values | classical | modern |
|---|---|---|
| 0.0 | 2950 | 500 |
| 0.1 | 500 | 500 |
| 0.2 | 350 | 500 |
| 0.3 | 250 | 500 |
| 0.4 | 200 | 500 |
| 0.5 | 150 | 500 |
| 0.6 | 100 | 500 |
| 0.7 | 100 | 500 |
| 0.8 | 100 | 500 |
| 0.9 | 50 | 500 |
| 1.0 | 50 | 500 |
</details>

(B) $(n,K) = (5000,3)$

![](images/b81d2005f5d5de645d4c0becdea82eaf6f7940f56ff3ed539e81819473481688.jpg)

<details>
<summary>bar</summary>

| p-values | classical | modern |
|---|---|---|
| 0.0 | 3000 | 480 |
| 0.1 | 550 | 470 |
| 0.2 | 350 | 520 |
| 0.3 | 250 | 520 |
| 0.4 | 180 | 490 |
| 0.5 | 140 | 490 |
| 0.6 | 130 | 510 |
| 0.7 | 100 | 490 |
| 0.8 | 70 | 460 |
| 0.9 | 50 | 480 |
| 1.0 | 30 | 490 |
</details>

(C) $(n,K) = (6000,4)$   
FIGURE 3. Histogram for p-values of the classical test (1.14) (in orange) and of the proposed test (2.4) (in blue) under $H_{0}$ in simulated data with different $(n, K)$ and p = 1000.

visually indistinguishable from the plots using $\Omega_{jj}$ . These confirm that the chi-square approximation and the associated test using $\hat{\Omega}_{jj}$ are accurate.

Non-Gaussian covariates and unknown $\Omega_{jj}$ . Although our theory assumes Gaussian covariates, we expect that the same results hold for other distributions with sufficiently light tails. To illustrate this point, we consider the following two types of non-Gaussian covariates: (i) The design matrix X has i.i.d. Rademacher entries, i.e., $\mathbb{P}(x_{ij} = \pm1) = \frac{1}{2}$ , (ii) Each $x_{ij}$ takes on values 0, 1 and 2 with respectively probabilities $a_{j}^{2}, 2a_{j}(1 - a_{j})$ , and $(1 - a_{j})^{2}$ , where $a_{j}$ varies in [0.25, 0.75]. Each columns of X are then centered and normalized to have 0 mean and unit variance. This generation of non-Gaussian covariates is adopted from single-nucleotide poly-morphisms (SNPs) example in Sur and Candès [2019]. For these two types of non-Gaussian covariates, we further rescale the feature vectors to ensure that $x_{i}$ has the same covariance as in the Gaussian case at the beginning of Section 3, that is $\Sigma = (0.5^{|i-j|})_{p \times p}$ . We present the Q-Q plots in Figure S2 using the same settings as in Figure S1, with the only difference being that the covariates in Figure S2 are non-Gaussian distributed. The Q-Q plots of $\mathcal{T}_{n}^{j}(X, Y)$ in (2.3) plotted in Figure S2 still closely match the diagonal line. These empirical successes suggest that the

normal and $\chi_{K}^{2}$ approximations (2.1)-(2.3) apply to a wider range of covariate distributions beyond normally distributed data.

# 4. DISCUSSION AND FUTURE WORK

Multinomial logistic regression estimates and their p-values are ubiquitous throughout the sciences for analyzing the significance of explanatory variables on multiclass responses. Following the seminal work of Sur and Candès [2019] in binary logistic regression, this paper develops the first valid tests and p-values for multinomial logistic estimates when p and n are of the same order. For 3 or more classes, this methodology and the corresponding asymptotic normality results in Theorems 2.1 and 2.2 are novel and provide new understanding of multinomial logistic estimates (also known as cross-entropy minimizers) in high-dimensions. We expect similar asymptotic normality and chi-square results to be within reach for loss functions different than the cross-entropy or a different model for the response $y_{i}$ ; for instance Section S1 provides an extension to the q-repeated measurements model, where q responses are observed for each feature vector $x_{i}$ .

Let us point a few follow-up research directions that we leave open for future work. A first open problem regards extensions of our methodology to confidence sets for $e_{j}^{T}B^{*}$ when $H_{0}$ in (1.10) is violated for the j-th covariate. This would require more stringent assumptions on the generative model than Assumption 2.1 as $B^{*}$ there is not identifiable (e.g., modification of both $B^{*}$ and $f(\cdot,\cdot)$ in Assumption 2.1 is possible without changing $y_{i}$ ). A second open problem is to relate this paper's theory to the fixed-point equations and limiting Gaussian model obtained in multiclass models, e.g., Loureiro et al. [2021]. While it may be straightforward to obtain the limit of $\frac{1}{n}\sum_{i=1}^{n}g_{i}g_{i}^{T}$ and of the empirical distribution of the rows of $\hat{A}$ in this context (e.g., using Corollary 3 in Loureiro et al. [2021]), the relationship between the fixed-point equations and the matrix $\frac{1}{n}\sum_{i=1}^{n}V_{i}$ appearing in (2.1) is unclear and not explained by typical results from this literature. A third open problem is to characterize the exact phase transition below which the multinomial logistic MLE exists and is bounded with high-probability (Assumption 2.4); while this is settled for two classes [Candès and Sur, 2020] and preliminary results are available for 3 or more classes [Loureiro et al., 2021, Kini and Thrampoulidis, 2021], a complete understanding of this phase transition is currently lacking. A last interesting open problem is to prove that our theory extend to non-Gaussian data, as observed in simulations. This challenging problem is often referred to as “universality” and has received intense attention recently [Montanari and Saeed, 2022, Gerace et al., 2022, Pesce et al., 2023, Dandi et al., 2023], showing that in several settings of interest (although none exactly the one considered here), the asymptotic behavior of the minimizers is unchanged if the distribution of the covariates is modified from normal to another distribution with the same covariance.

# REFERENCES

Pierre C Bellec. Observable adjustments in single-index models for regularized m-estimators. arXiv preprint arXiv:2204.06990, 2022.   
Pierre C. Bellec and Cun-Hui Zhang. Second-order Stein: SURE for SURE and other applications in high-dimensional inference. Ann. Statist., 49(4):1864–1903, 2021. ISSN 0090-5364. URL .   
Raphael Berthier, Andrea Montanari, and Phan-Minh Nguyen. State evolution for approximate message passing with non-separable functions. Information and Inference: A Journal of the IMA, 9(1):33–79, 2020.

Emmanuel J. Candès and Pragya Sur. The phase transition for the existence of the maximum likelihood estimate in high-dimensional logistic regression. The Annals of Statistics, 48(1):27 - 42, 2020. doi: 10.1214/18-AOS1789. URL .   
Elisabetta Cornacchia, Francesca Mignacco, Rodrigo Veiga, Cédric Gerbelot, Bruno Loureiro, and Lenka Zdeborová. Learning curves for the multi-class teacher-student perceptron. arXiv preprint arXiv:2203.12094, 2022.   
Jan Salomon Cramer. The origins of logistic regression. Tinbergen Institute Working Paper, 2002.   
Yatin Dandi, Ludovic Stephan, Florent Krzakala, Bruno Loureiro, and Lenka Zdeborová. Universality laws for gaussian mixtures in generalized linear models. arXiv preprint arXiv:2302.08933, 2023.   
Kenneth R Davidson and Stanislaw J Szarek. Local operator theory, random matrices and banach spaces. Handbook of the geometry of Banach spaces, 1(317-366):131, 2001.   
Oliver Y Feng, Ramji Venkataramanan, Cynthia Rush, Richard J Samworth, et al. A unifying tutorial on approximate message passing. Foundations and Trends® in Machine Learning, 15(4):335–536, 2022.   
Federica Gerace, Florent Krzakala, Bruno Loureiro, Ludovic Stephan, and Lenka Zdeborová. Gaussian universality of linear classifiers with random labels in high-dimension. arXiv preprint arXiv:2205.13303, 2022.   
Cédric Gerbelot and Raphaël Berthier. Graph-based approximate message passing iterations. arXiv preprint arXiv:2109.11905, 2021.   
Ganesh Ramachandra Kini and Christos Thrampoulidis. Phase transitions for one-vs-one and one-vs-all linear separability in multiclass gaussian mixtures. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4020–4024. IEEE, 2021.   
Bruno Loureiro, Gabriele Sicuro, Cédric Gerbelot, Alessandro Pacco, Florent Krzakala, and Lenka Zdeborová. Learning gaussian mixtures with generalized linear models: Precise asymptotics in high-dimensions. Advances in Neural Information Processing Systems, 34:10144–10157, 2021.   
Xiaoyi Mai, Zhenyu Liao, and Romain Couillet. A large scale analysis of logistic regression: Asymptotic performance and new insights. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3357–3361. IEEE, 2019.   
Andrea Montanari and Basil N Saeed. Universality of empirical risk minimization. In Conference on Learning Theory, pages 4310–4312. PMLR, 2022.   
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.   
Luca Pesce, Florent Krzakala, Bruno Loureiro, and Ludovic Stephan. Are gaussian data all you need? extents and limits of universality in high-dimensional generalized linear estimation. arXiv preprint arXiv:2302.08923, 2023.   
Fariborz Salehi, Ehsan Abbasi, and Babak Hassibi. The impact of regularization on high-dimensional logistic regression. Advances in Neural Information Processing Systems, 32, 2019.   
Pragya Sur and Emmanuel J. Candès. A modern maximum-likelihood theory for high-dimensional logistic regression. Proc. Natl. Acad. Sci. USA, 116(29):14516–14525, 2019. ISSN 0027-8424. doi: 10.1073/pnas.1810420116. URL .

Christos Thrampoulidis, Samet Oymak, and Mahdi Soltanolkotabi. Theoretical insights into multiclass classification: A high-dimensional asymptotic view. Advances in Neural Information Processing Systems, 33:8907–8920, 2020.   
Aad W Van der Vaart. Asymptotic statistics, volume 3. Cambridge university press, 1998.   
J Leo van Hemmen and Tsuneya Ando. An inequality for trace ideals. Communications in Mathematical Physics, 76:143-148, 1980.   
Steve Yadlowsky, Taedong Yun, Cory Y McLean, and Alexander D'Amour. Sloe: A faster method for statistical inference in high-dimensional logistic regression. Advances in Neural Information Processing Systems, 34:29517–29528, 2021.   
Qian Zhao, Pragya Sur, and Emmanuel J Candes. The asymptotic distribution of the mle in high-dimensional logistic models: Arbitrary covariance. Bernoulli, 28(3):1835–1861, 2022.   
Ji Zhu and Trevor Hastie. Classification of gene microarrays by penalized logistic regression. Biostatistics, 5(3):427-443, 2004.

# Supplementary Material of “Multinomial Logistic Regression: Asymptotic Normality on Null Covariates in High-Dimensions”

Let us define some standard notation that will be used in the rest of this supplement. For a vector $v \in R^{n}$ , let $\|v\|_{\infty} = \max_{i \in [n]} |v_i|$ denote the infinity norm of vector v. If A is symmetric, we define $\lambda_{\min}(A)$ and $\|A\|_{op}$ as the minimal and maximal eigenvalues of A, respectively. For two symmetric matrices A, B of the same size, we write $A \preceq B$ if and only if B - A is positive semi-definite.

# DIAGRAM: ORGANIZATION OF THE PROOFS

The following diagram summarizes the different theorems and lemmas, and the relationships between them.

![](images/75ec4e2d642f5f76c612f45dc86ce6951d44130760fa9af36cfee5dd9ea3265a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Theorem 2.2\nAsymptotic normality for \(\hat{A}^T e_j\) on null covariates, where \(\hat{A} \in \mathbb{R}^{p \times K}\) is the multinomial logistic MLE with class \(K+1\) fixed as the reference class (see (1.9).)"] --> B["Theorem 2.1\nAsymptotic normality for \(\hat{B}^T e_j\) on null covariates, where \(\hat{B} \in \mathbb{R}^{p \times (K+1)}\) is the multinomial logistic MLE in (1.5)."]
    B --> C["Theorem S3.1\nAsymptotic normality for \(\hat{B}^T e_j\) on null covariates, where \(\hat{B} \in \mathbb{R}^{p \times K}\) is the multinomial logistic MLE using the parameter space from Section S3.1.\nThe proof uses that the conditions in Theorem S5.1 on the loss function are satisfied by the cross-entropy."]
    C --> D["Theorem S5.1\nAsymptotic normality on null covariates for general loss functions, \(\Sigma \neq I_p\).\nDeduced from Theorem S5.2 by rotational invariance."]
    D --> E["Theorem S5.2\nAsymptotic normality on null covariates for general loss functions, \(\Sigma = I_p\)."]
    E --> F["Lemma S5.3\nNormal and \(\chi^2\) approximations for random variables defined as a differentiable function of standard normal vectors."]
    F --> G["Lemma S5.5\nLemma S5.5 computes the derivatives of the minimizer with respect to X, used in the proof of Theorem S5.2."]
    G --> H["Control of g_i and H_i for the cross-entropy loss\nLemmas S6.1 and S6.2 give deterministic arguments to control the gradients and Hessians of the cross-entropy loss. Lemma S6.4 controls the Hessian of the cross-entropy loss at the minimizer, in a specific high-probability event. Lemma S6.3 defines this high-probability event."]
    H --> I["Section S3.1 defines the matrix \(Q\) and discusses a convenient parametrization of the model isometric to the subspace orthogonal to \(\mathbf{1}_{K+1}\)."]
```
</details>

# S1. EXTENSION: q REPEATED MEASUREMENTS

Let integer $q \geq 1$ be a constant independent of n, p. Our results readily extend if q labels are observed for each observed feature vector $x_{i}$ , and the corresponding q one-hot encoded vectors are averaged into $y_{i} \in \{0, \frac{1}{q}, \frac{2}{q}, ..., 1\}^{K+1}$ . Concretely, for each observation $i \in [n]$ , q i.i.d. labels $(Y_{i}^{m})_{m \in [q]}$ are observed with each $Y_{i}^{m} \in \{0, 1\}^{K+1}$ one-hot encoded and $y_{ik} = \frac{1}{q} \sum_{m=1}^{q} Y_{ik}^{m}$ , for instance in a repeated multinomial regression model with $\mathbb{P}(Y_{ik}^{m} = 1 | x_{i})$ equal to right-hand side of (1.4). In this case where $(Y_{i}^{m})_{m \in [q]}$ are i.i.d., Assumption 2.3 is satisfied by the law of large numbers if $\min_{k \in [K+1]} \mathbb{P}(Y_{ik}^{m} = 1) > 0$ since q is constant. For this q repeated measurements model, the negative log-likelihood function of a parameter $B \in R^{p \times (K+1)}$ is

$$
\begin{array}{l} - \sum_ {i = 1} ^ {n} \sum_ {m = 1} ^ {q} \sum_ {k = 1} ^ {K + 1} Y _ {i k} ^ {m} \left[ x _ {i} ^ {T} \mathsf {B e} _ {k} - \log \sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp (x _ {i} ^ {T} \mathsf {B e} _ {k ^ {\prime}}) \right] \\ = q \sum_ {i = 1} ^ {n} \sum_ {k = 1} ^ {K + 1} y _ {i k} \left[ - x _ {i} ^ {T} \mathsf {B} e _ {k} + \log \sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp \left(x _ {i} ^ {T} \mathsf {B} e _ {k ^ {\prime}}\right) \right] \\ = q \sum_ {i = 1} ^ {n} \left[ \sum_ {k = 1} ^ {K + 1} - \mathrm{y} _ {i k} x _ {i} ^ {T} \mathsf {B} e _ {k} + \log \sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp \left(x _ {i} ^ {T} \mathsf {B} e _ {k ^ {\prime}}\right) \right] \\ = q \sum_ {i = 1} ^ {n} \mathsf {L} _ {i} (\mathsf {B} ^ {T} x _ {i}), \\ \end{array}
$$

where the first equality uses $y_{ik} = \frac{1}{q} \sum_{m=1}^{q} Y_{ik}^{m}$ , the second equality uses $\sum_{k=1}^{K+1} y_{ik} = 1$ under the following Assumption S1.1, and the last equality uses the definition of $L_{i}$ after (1.5).

Assumption S1.1. For all $i \in [n]$ , the response $y_{i}$ is in $\{0, 1/q, 2/q, ..., 1\}^{K+1}$ with $\sum_{k=1}^{K+1} y_{ik} = 1$ .

In such repeated measurements model, we replace Assumption 2.2 with Assumption S1.1 under which the following Theorem S1.1 holds.

Theorem S1.1. Let $q \geq 2$ be constant. Let Assumptions S1.1, 2.1, 2.3 and 2.4 be fulfilled. For any $j \in [p]$ such that $H_0$ in (1.10) holds, we have the convergence in distribution (2.1), (2.2) and (2.3).

Proof of Theorem S1.1. Under the assumptions in Theorem S1.1, the MLE $\hat{B}$ for this q repeated measurements model is the minimizer of the optimization problem

$$
\hat {\mathsf {B}} \in \underset {\mathsf {B} \in \mathbb {R} ^ {p \times (K + 1)}} {\arg \min} \sum_ {i = 1} ^ {n} \mathsf {L} _ {i} (\mathsf {B} ^ {T} x _ {i})
$$

as in (1.5). Similar to the non-repeated model, the MLE $\hat{A}$ for the identifiable log-odds model can be expressed as

$$
\hat {A} = \underset {A \in \mathbb {R} ^ {p \times K}} {\arg \min} \sum_ {i = 1} ^ {n} \mathsf {L} _ {i} ((A, \mathbf {0} _ {p}) ^ {T} x _ {i}).
$$

The only difference between this q repeated measurements model and the non-repeated model considered in the main text is that the response $y_{ik}$ for this q repeated measurements model is now valued in $\{0,1/q,2/q,\ldots,1\}$ . Because the proofs of Theorems 2.1 and 2.2 do

not require the value of $y_{ik}$ to be $\{0,1\}$ -valued. Theorem S1.1 can be proved by the same arguments used in the proof of Theorems 2.1 and 2.2. ☐

# S2. IMPLEMENTATION DETAILS AND ADDITIONAL FIGURES

The pivotal quantities in our main results Theorems 2.1 and 2.2 involve only observable quantities that can be computed from the data $(x_{i},\mathbf{y}_{i})_{i\in[n]}$ . In this section we provide an efficient way of computing the matrix $V_{i}$ appearing in Theorems 2.1 and 2.2.

Fast computation of $V_{i}$ . Recall the definition of $V_{i}$ in Theorem 2.1,

$$
\mathsf {V} _ {i} = \mathsf {H} _ {i} - (\mathsf {H} _ {i} \otimes x _ {i} ^ {T}) \Big [ \sum_ {l = 1} ^ {n} \mathsf {H} _ {l} \otimes (x _ {l} x _ {l} ^ {T}) \Big ] ^ {\dagger} (\mathsf {H} _ {i} \otimes x _ {i}).
$$

The majority of computational cost in calculating $V_{i}$ lies in the step of calculating its second term

$$
\left(\mathsf {H} _ {i} \otimes x _ {i} ^ {T}\right) \left[ \sum_ {l = 1} ^ {n} \mathsf {H} _ {l} \otimes \left(x _ {l} x _ {l} ^ {T}\right) \right] ^ {\dagger} \left(\mathsf {H} _ {i} \otimes x _ {i} ^ {T}\right).
$$

Here we provide an efficient way to compute this term using the Woodbury matrix identity. Since $H_{i}1_{K+1}=0_{K+1}$ , we have $\ker(\mathsf{H}_{i}\otimes(x_{i}x_{i}^{T}))$ is the span of $\{1_{K+1}\otimes e_{j}:j\in[p]\}$ , where $1_{K+1}$ is the all-ones vector in $R^{K+1}$ . Therefore, the second term in $V_{i}$ can be rewritten as

$$
\begin{array}{l} \left(\mathsf {H} _ {i} \otimes x _ {i} ^ {T}\right) \left[ \sum_ {l = 1} ^ {n} \mathsf {H} _ {l} \otimes \left(x _ {l} x _ {l} ^ {T}\right) \right] ^ {\dagger} \left(\mathsf {H} _ {i} \otimes x _ {i}\right) \\ = \left(\mathsf {H} _ {i} \otimes x _ {i} ^ {T}\right) \left[ \sum_ {l = 1} ^ {n} \mathsf {H} _ {l} \otimes \left(x _ {l} x _ {l} ^ {T}\right) - \sum_ {j = 1} ^ {p} (\mathbf {1} \otimes e _ {j}) (\mathbf {1} \otimes e _ {j}) ^ {T} \right] ^ {- 1} \left(\mathsf {H} _ {i} \otimes x _ {i}\right). \\ \end{array}
$$

We now apply the Woodbury matrix identity to compute the matrix inversion in the above display. Recall $\mathsf{H}_{i}=\mathrm{diag}(\hat{\mathsf{p}}_{i})-\hat{\mathsf{p}}_{i}\hat{\mathsf{p}}_{i}^{T}$ , we have

$$
\sum_ {i = 1} ^ {n} \mathsf {H} _ {i} \otimes (x _ {i} x _ {i} ^ {T}) = \sum_ {k = 1} ^ {K + 1} (e _ {k} e _ {k} ^ {T}) \otimes (\sum_ {i = 1} ^ {n} \hat {\mathfrak {p}} _ {i k} x _ {i} x _ {i} ^ {T}) - \sum_ {i = 1} ^ {n} (\hat {\mathfrak {p}} _ {i} \otimes x _ {i}) (\hat {\mathfrak {p}} _ {i} \otimes x _ {i}) ^ {T}.
$$

Let $A = \sum_{k=1}^{K+1}(e_k e_k^T) \otimes (\sum_{i=1}^n \hat{\mathfrak{p}}_{ik} x_i x_i^T)$ , and $U \in \mathbb{R}^{p(K+1) \times (n+p)}$ with the first $n$ columns being $(\hat{\mathfrak{p}}_i \otimes x_i)_{i \in [n]}$ and the following $p$ columns $(\mathbf{1} \otimes e_j)_{j \in [p]}$ . Then the term we want to invert is $A - UU^T$ , where $A$ is a block diagonal matrix and can be inverted by inverting each block separately. By the Woodbury matrix identity, we have

$$
(A - U U ^ {T}) ^ {- 1} = A ^ {- 1} - A ^ {- 1} U (- I _ {n + p} + U ^ {T} A ^ {- 1} U) ^ {- 1} U ^ {T} A ^ {- 1}.
$$

The gain of using the above formula is significant for large $K$ : instead of inverting the $p(K + 1) \times p(K + 1)$ matrix $\sum_{l=1}^{n} \mathsf{H}_l \otimes (x_l x_l^T)$ in the left-hand side, the right-hand side only requires to invert a block diagonal matrix $A$ and a $(n+p) \times (n+p)$ matrix $-I_{n+p} + U^T A^{-1}U$ .

![](images/dce1fea087f0738fe6db58f1983f2e01e79e1fb76e259c62cfef135628d2c529.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | sample quantiles (classical) | sample quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0.0                   | 0.0                          | 0.0                       |
| 2.5                   | 12.0                         | 2.5                       |
| 5.0                   | 12.0                         | 5.0                       |
| 7.5                   | 12.0                         | 7.5                       |
| 10.0                  | 12.0                         | 10.0                      |
| 12.5                  | 12.0                         | 12.5                      |
| 13.0                  | 12.0                         | 13.0                      |
</details>

![](images/e4e530d2a65caf833dfce81c94701e76d27a8df5402b5931869ba4642725f0ec.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | classical | modern |
| --------------------- | --------- | ------ |
| 0                     | 0         | 0      |
| 5                     | 16        | 5      |
| 10                    | 16        | 10     |
| 15                    | 16        | 15     |
</details>

![](images/38bc6988086d21ba635a6589758c1dd3a1df6b8dbea2586f46c819cceb6e263d.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | sample Quantiles (classical) | sample Quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0                     | 0.0                          | 0.0                       |
| 5                     | 17.5                         | 5.0                       |
| 10                    | 17.5                         | 10.0                      |
| 15                    | 17.5                         | 15.0                      |
| 20                    | 17.5                         | 17.5                      |
</details>

![](images/34efcc4fb8f7b4b56217dbb7d656307b86b14a41b6e1d4a442815570e0e102ed.jpg)

<details>
<summary>bar</summary>

| p-values | classical | modern |
|---|---|---|
| 0.0 | 2800 | 500 |
| 0.1 | 550 | 500 |
| 0.2 | 350 | 500 |
| 0.3 | 250 | 500 |
| 0.4 | 250 | 500 |
| 0.5 | 200 | 500 |
| 0.6 | 150 | 500 |
| 0.7 | 150 | 500 |
| 0.8 | 150 | 550 |
| 0.9 | 100 | 550 |
| 1.0 | 100 | 500 |
</details>

(A) $(n, K) = (4000, 2)$

![](images/f9f358fa341a78caa93eb28166e9c2a2b2e3b22d3c31a28be57f95635450aa21.jpg)

<details>
<summary>bar</summary>

| p-values | classical | modern |
|---|---|---|
| 0.0 | 2950 | 500 |
| 0.1 | 500 | 500 |
| 0.2 | 350 | 500 |
| 0.3 | 250 | 500 |
| 0.4 | 200 | 500 |
| 0.5 | 150 | 500 |
| 0.6 | 100 | 500 |
| 0.7 | 100 | 500 |
| 0.8 | 100 | 500 |
| 0.9 | 50 | 500 |
| 1.0 | 50 | 500 |
</details>

(B) $(n,K) = (5000,3)$

![](images/ddb7d37eb3a2fdb43a88411211e40542c002c286b45a90558496915ed1e526fe.jpg)

<details>
<summary>bar</summary>

| p-values | classical | modern |
|---|---|---|
| 0.0 | 3000 | 450 |
| 0.1 | 600 | 450 |
| 0.2 | 350 | 500 |
| 0.3 | 250 | 500 |
| 0.4 | 150 | 450 |
| 0.5 | 100 | 450 |
| 0.6 | 100 | 450 |
| 0.7 | 80 | 450 |
| 0.8 | 50 | 450 |
| 0.9 | 30 | 450 |
| 1.0 | 20 | 450 |
</details>

(C) $(n,K) = (6000,4)$   
FIGURE S1. The upper row: Q-Q plots of the test statistics from (2.3) (in blue) and (1.13) (in orange) for different $(n, K)$ and p = 1000 using $\hat{\Omega}_{jj}$ . The lower row: histograms of p-values from classical test and our test for different $(n, K)$ and p = 1000 using $\hat{\Omega}_{jj}$ .

![](images/f54585e1843ef0d570e746db4feee74548fc0bf0a75599fb1dd441f8109d5997.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | sample Quantiles (classical) | sample Quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0.0                   | 0.0                          | 0.0                       |
| 2.5                   | 12.5                         | 2.5                       |
| 5.0                   | 12.5                         | 5.0                       |
| 7.5                   | 12.5                         | 7.5                       |
| 10.0                  | 12.5                         | 10.0                      |
| 12.5                  | 12.5                         | 12.5                      |
</details>

![](images/c8a3c4f784f6ca406843c0ae636c2d9e6ba8974f7421859465fc72b87adb82a9.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | classical | modern |
| --------------------- | --------- | ------ |
| 0                     | 0         | 0      |
| 5                     | 16        | 6      |
| 10                    | 12        | 10     |
| 15                    | 16        | 16     |
</details>

![](images/7b250f91d615e06e7768ae6c30c73b5edf1b973510fd449bb6198a49f4ee676b.jpg)

<details>
<summary>scatter</summary>

| Theoretical Quantiles | Sample Quantiles (classical) | Sample Quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0                     | 0.0                          | 0.0                       |
| 5                     | 17.5                         | 5.0                       |
| 10                    | 17.5                         | 10.0                      |
| 15                    | 17.5                         | 15.0                      |
| 16                    | 17.5                         | 17.5                      |
</details>

![](images/ce339cfa95b50103ee54439235e30510c89f282640cf41d2fbb45c6b7e21b9ee.jpg)

<details>
<summary>scatter</summary>

| Theoretical Quantiles | Sample Quantiles (classical) | Sample Quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0.0                   | 0.0                          | 0.0                       |
| 2.5                   | 12.5                         | 2.5                       |
| 5.0                   | 12.5                         | 5.0                       |
| 7.5                   | 12.5                         | 7.5                       |
| 10.0                  | 12.5                         | 10.0                      |
| 12.5                  | 12.5                         | 12.5                      |
</details>

(A) $(n, K) = (4000, 2)$

![](images/dcfdbeacabf05dd87c723a10eb71fe0d847c552e8c973f636545121f92273aac.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | sample quantiles (classical) | sample quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0                     | 0                            | 0                         |
| 5                     | 16                           | 5                         |
| 10                    | 12                           | 10                        |
| 15                    | 16                           | 15                        |
</details>

(B) $(n,K) = (5000,3)$

![](images/49e23955c82758a9ed207d5df7c9685594c79c5851998681c33c270214839319.jpg)

<details>
<summary>line</summary>

| Theoretical Quantiles | sample Quantiles (classical) | sample Quantiles (modern) |
| --------------------- | ---------------------------- | ------------------------- |
| 0                     | 0.0                          | 0.0                       |
| 5                     | 17.5                         | 5.0                       |
| 10                    | 17.5                         | 10.0                      |
| 15                    | 17.5                         | 15.0                      |
| 20                    | 17.5                         | 17.5                      |
</details>

(C) $(n,K) = (6000,4)$

FIGURE S2. Q-Q plots of the test statistics from (2.3) (in blue) and (1.13) (in orange) for different $(n, K)$ and p = 1000 using $\hat{\Omega}_{jj}$ . The upper row: covariates are sampled from Rademacher distribution. The lower row: covariates are sampled from distribution of SNPs.   
![](images/3ed2f355f8280c25717c413f3154c6f7fcfbd0908fed619dae0a3995e108b4a8.jpg)  
(A) q = 1

![](images/85bc89c3c484bfadbf3fc260128b4d84fbee897f16f8402ef8ba0cffa2cac7cb.jpg)

<details>
<summary>scatter</summary>

| x    | y    | class     |
| ---- | ---- | --------- |
| -10  | -5   | classical  |
| -5   | 0    | classical  |
| 0    | 5    | classical  |
| 5    | 0    | classical  |
| 10   | -5   | classical  |
| -10  | -10  | modern    |
| -5   | -5   | modern    |
| 0    | 0    | modern    |
| 5    | 5    | modern    |
| 10   | 0    | modern    |
</details>

(B) q = 2

![](images/84c5055b1124671f7f4f11319cd00f693501243e967ea02b0761b22d8553a8f4.jpg)

<details>
<summary>scatter</summary>

| x    | y    | class         |
| ---- | ---- | ------------- |
| -8   | 8    | classical     |
| -6   | 6    | classical     |
| -4   | 4    | classical     |
| -2   | 2    | classical     |
| 0    | 0    | classical     |
| 2    | -2   | classical     |
| 4    | -4   | classical     |
| 6    | -6   | classical     |
| 8    | -8   | classical     |
| -8   | 8    | modern        |
| -6   | 6    | modern        |
| -4   | 4    | modern        |
| -2   | 2    | modern        |
| 0    | 0    | modern        |
| 2    | -2   | modern        |
| 4    | -4   | modern        |
| 6    | -6   | modern        |
| 8    | -8   | modern        |
</details>

(C) q = 4   
FIGURE S3. Scatter plot of pairs $(\sqrt{n}\hat{A}_{j1},\sqrt{n}\hat{A}_{j2})$ with the same data generating process as in Figure 1 (a) except using different q.

# S3. PROOF OF THEOREM 2.1

Before proving Theorem 2.1, we present another parametrization of the multinomial logistic regression model. The asymptotic theory of MLE for this new parametrized multinomial logistic model will be used to prove Theorem 2.1.

S3.1. Another parametrization of multinomial logistic regression. Recall the symbol “∈” is used in (1.5) to emphasize that the minimizer $\hat{B}$ in (1.5) is not unique: if $\hat{B}$ is a minimizer of (1.5) then $\hat{B}-b1_{K+1}^{T}$ is also a minimizer of (1.5), for any $b\in R^{p}$ and the all-ones vector $1_{K+1}$ in $R^{K+1}$ .

Besides the log-odds model (1.8), here we consider another identifiable parametrization of multinomial logistic regression, whose unknown parameter, denoted by $B^{*}$ , is in $\mathbb{R}^{p \times K}$ . Orthogonal complement. To obtain an identifiable multinomial logistic regression model from (1.4), we consider the symmetric constraint $\mathsf{B}^{*}\mathbf{1} = \sum_{k=1}^{K+1}\mathsf{B}^{*}e_{k} = \mathbf{0}$ as in [Zhu and Hastie, 2004], thus $\mathsf{B}^{*} = \mathsf{B}^{*}(I_{K+1} - \frac{\mathbf{11}^{T}}{K+1})$ , where $\mathbf{1}$ is the all-ones vector in $\mathbb{R}^{K+1}$ . Let $Q \in \mathbb{R}^{(K+1) \times K}$ be any matrix such that

$$
I _ {K + 1} - \frac {1}{K + 1} \mathbf {1 1} ^ {T} = Q Q ^ {T}, \quad Q ^ {T} Q = I _ {K}. \tag {S3.1}
$$

We fix one choice of Q satisfying (S3.1) throughout this supplement. Let $B^{*} = B^{*}Q$ , then $B^{*} = B^{*}Q^{T}$ and the model (1.4) can be parameterized using $B^{*}$ as

$$
\mathbb {P} (\mathbf {y} _ {i k} = 1 | x _ {i}) = \frac {\exp (x _ {i} ^ {T} B ^ {*} Q e _ {k})}{\sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp (x _ {i} ^ {T} B ^ {*} Q e _ {k ^ {\prime}})}, \quad k \in \{1, 2, \ldots , K + 1 \}. \tag {S3.2}
$$

The multinomial logistic MLE of $B^{*}$ in (S3.2) is given by

$$
\hat {B} = \arg \min _ {B \in \mathbb {R} ^ {p \times K}} \sum_ {i = 1} ^ {n} L _ {i} (B ^ {T} x _ {i}), \tag {S3.3}
$$

where $L_{i}: R^{K} \to R$ is defined by $L_{i}(u) = \mathsf{L}_{i}(Qu)$ for all $u \in R^{K}$ . By this construction, we have $\hat{B} = \hat{B}Q$ for any minimizer $\hat{B}$ of (1.5). Furthermore, by the chain rule using the expressions (1.7), the gradient and Hessian of $L_{i}$ evaluated at $\hat{B}^{T}x_{i}$ are

$$
g _ {i} := \nabla L _ {i} (\hat {B} ^ {T} x _ {i}) = Q ^ {T} \mathbf {g} _ {i}, \qquad H _ {i} := \nabla^ {2} L _ {i} (\hat {B} ^ {T} x _ {i}) = Q ^ {T} \mathsf {H} _ {i} Q. \tag {S3.4}
$$

Throughout, we use serif upright letters to denote quantities defined on the unidentifiable parameter space $\mathbb{R}^{p\times (K + 1)}$ :

$$
\mathsf {B} ^ {*}, \hat {\mathsf {B}} \in \mathbb {R} ^ {p \times (K + 1)}, \quad \mathsf {L} _ {i}: \mathbb {R} ^ {K + 1} \to \mathbb {R}, \qquad \hat {\mathsf {p}} _ {i}, \mathsf {y} _ {i}, \mathsf {g} _ {i} \in \mathbb {R} ^ {K + 1}, \qquad \mathsf {H} _ {i} \in \mathbb {R} ^ {(K + 1) \times (K + 1)}
$$

and the normal italic font to denote analogous quantities for the identifiable parameter space $\mathbb{R}^{p\times K}$ :

$$
B ^ {*}, \hat {B} \in \mathbb {R} ^ {p \times K}, \qquad L _ {i}: \mathbb {R} ^ {K} \to \mathbb {R}, \qquad g _ {i} \in \mathbb {R} ^ {K}, \qquad H _ {i} \in \mathbb {R} ^ {K \times K}.
$$

Theorem S3.1 provides the asymptotic normality and the chi-square approximation of null MLE coordinates in high-dimensions where $n, p \to \infty$ with the ratio n/p converging to a finite limit.

Theorem S3.1 (Proof is given on page 31). Let Assumptions 2.1, 2.3 and 2.4 be fulfilled. Assume that either Assumption 2.2 or Assumption S1.1 holds. Then for any $j \in [p]$ such that $H_0$ in (1.10) holds,

$$
\sqrt {n} \Omega_ {j j} ^ {- 1 / 2} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} g _ {i} g _ {i} ^ {T}\right) ^ {- 1 / 2} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} V _ {i}\right) \hat {B} ^ {T} e _ {j} \xrightarrow {\mathrm{d}} N (0, I _ {K}), \tag {S3.5}
$$

where $V_{i} = H_{i} - (H_{i}\otimes x_{i}^{T})[\sum_{l = 1}^{n}H_{l}\otimes (x_{l}x_{l}^{T})]^{-1}(H_{i}\otimes x_{i}).$

A direct consequence of (S3.5) is the $\chi^2$ result,

$$
\| \sqrt {n} \Omega_ {j j} ^ {- 1 / 2} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} g _ {i} g _ {i} ^ {T}\right) ^ {- 1 / 2} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} V _ {i}\right) \hat {B} ^ {T} e _ {j} \| ^ {2} \xrightarrow {\mathrm{d}} \chi_ {K} ^ {2}. \tag {S3.6}
$$

The proof of Theorem S3.1 is deferred to Section S5 and Section S6. In the next subsection, we prove Theorem 2.1 using Theorem S3.1.

# S3.2. Proof of Theorem 2.1. We restate Theorem 2.1 for convenience.

Theorem 2.1. Let Assumptions 2.1 to 2.4 be fulfilled. Then for any $j \in [p]$ such that $H_0$ in (1.10) holds, and any minimizer $\hat{\mathsf{B}}$ of (1.5), we have (2.1)

$$
\underbrace {\sqrt {\frac {n}{\Omega_ {j j}}}} _ {s c a l a r} \Big (\underbrace {\left(\frac {1}{n} \sum_ {i = 1} ^ {n} (\mathsf {y} _ {i} - \hat {\mathsf {p}} _ {i}) (\mathsf {y} _ {i} - \hat {\mathsf {p}} _ {i}) ^ {T}\right) ^ {1 / 2}} _ {s q u a r e r o o t p s e u d o - i n v e r s e \mathbb {R} ^ {(K + 1) \times (K + 1)}} \Big) ^ {\dagger} \underbrace {\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {V} _ {i}\right)} _ {\mathbb {R} ^ {(K + 1) \times (K + 1)} \underbrace {\mathsf {B} ^ {T} e _ {j}} _ {\mathbb {R} ^ {K + 1}}} \xrightarrow {\mathrm{d}} N \Big (\mathbf {0}, \underbrace {I _ {K + 1} - \frac {\mathbf {1 1} ^ {T}}{K + 1}} _ {c o v. \mathbb {R} ^ {(K + 1) \times (K + 1)}} \Big),
$$

where $\mathsf{V}_i = \mathsf{H}_i - (\mathsf{H}_i\otimes x_i^T)[\sum_{l = 1}^n\mathsf{H}_l\otimes (x_lx_l^T)]^\dagger (\mathsf{H}_i\otimes x_i)$ .

The proof of Theorem 2.1 is a consequence of Theorem S3.1. To begin with, we state the following useful lemma.

Lemma S3.2. For $V_{i}$ and $V_{i}$ defined in Theorems 2.1 and S3.1, we have $V_{i}=Q^{T}V_{i}Q$ .

Proof of Lemma S3.2. Since $H_{i}=Q^{T}H_{i}Q$ , we have

$$
\begin{array}{l} V _ {i} = H _ {i} - \left(H _ {i} \otimes x _ {i} ^ {T}\right) \left[ \sum_ {i = 1} ^ {n} H _ {i} \otimes \left(x _ {i} x _ {i} ^ {T}\right) \right] ^ {- 1} \left(H _ {i} \otimes x _ {i}\right) \\ = H _ {i} - \left[ \left(Q ^ {T} \mathsf {H} _ {i} Q\right) \otimes x _ {i} ^ {T} \right] \left[ \sum_ {i = 1} ^ {n} \left(Q ^ {T} \mathsf {H} _ {i} Q\right) \otimes \left(x _ {i} x _ {i} ^ {T}\right) \right] ^ {- 1} \left[ \left(Q ^ {T} \mathsf {H} _ {i} Q\right) \otimes x _ {i} \right] \\ = H _ {i} - Q ^ {T} \left(\mathsf {H} _ {i} \otimes x _ {i} ^ {T}\right) (Q \otimes I _ {p}) \left[ \left(Q ^ {T} \otimes I _ {p}\right) \left[ \sum_ {i = 1} ^ {n} \mathsf {H} _ {i} \otimes \left(x _ {i} x _ {i} ^ {T}\right) \right] (Q \otimes I _ {p}) \right] ^ {- 1} \left(Q ^ {T} \otimes I _ {p}\right) \left(\mathsf {H} _ {i} \otimes x _ {i}\right) Q \\ = Q ^ {T} \mathsf {H} _ {i} Q - Q ^ {T} (\mathsf {H} _ {i} \otimes x _ {i} ^ {T}) \left[ \sum_ {i = 1} ^ {n} \left(\mathsf {H} _ {i} \otimes x _ {i} x _ {i} ^ {T}\right) \right] ^ {\dagger} (\mathsf {H} _ {i} \otimes x _ {i}) Q \\ = Q ^ {T} \mathrm{V} _ {i} Q, \\ \end{array}
$$

where the penultimate equality is proved as follows.

Let $A = Q \otimes I_p$ and $D = \sum_{i=1}^{n} H_i \otimes (x_i x_i^T)$ only in the remaining of this proof. It remains to prove

$$
A [ A ^ {T} \mathsf {D} A ] ^ {- 1} A ^ {T} = \mathsf {D} ^ {\dagger}. \tag {S3.7}
$$

Since $H_{i}1 = 0$ , we have $\mathsf{D}(\mathbf{1} \otimes I_{p}) = 0$ . Since $Q^{T}1 = 0$ by definition of Q, we have $A^{T}(\mathbf{1} \otimes I_{p}) = 0$ . If we write the eigen-decomposition of D as $D = \sum_{i=1}^{pK} \lambda_{i} u_{i} u_{i}^{T}$ , then $u_{i}^{T}(\mathbf{1} \otimes I_{p}) = 0$ . Hence, with $v_{i} = A^{T}u_{i}$ ,

$$
A ^ {T} \mathsf {D} A = \sum_ {i = 1} ^ {p K} \lambda_ {i} v _ {i} v _ {i} ^ {T}.
$$

Since $v_{i}^{T}v_{i'} = u_{i}^{T}AA^{T}u_{i'} = u_{i}^{T}[(I_{K+1} - \frac{11^{T}}{K+1})\otimes I_{p}]u_{i'} = u_{i}^{T}u_{i'} = I(i = i')$ , we have

$$
A [ A ^ {T} \mathsf {D} A ] ^ {- 1} A ^ {T} = A \big (\sum_ {i = 1} ^ {p K} \lambda_ {i} ^ {- 1} v _ {i} v _ {i} ^ {T} \big) A ^ {T} = \sum_ {i = 1} ^ {p K} \lambda_ {i} ^ {- 1} u _ {i} u _ {i} ^ {T} = \mathsf {D} ^ {\dagger},
$$

where the second equality uses $Av_{i} = AA^{T}u_{i} = u_{i}$ . The proof of (S3.7) is complete. ☐

Now we are ready to prove that Theorem 2.1 is a consequence of Theorem S3.1.

Proof of Theorem 2.1. By definition of $\mathbf{g}_i$ and $\mathsf{V}_i$ , we have $\mathbf{1}^T\mathbf{g}_i = 0$ and $\mathbf{1}^T\mathsf{V}_i = \mathbf{0}^T$ . Thus, we have $QQ^{T}\mathbf{g}_{i} = \mathbf{g}_{i}$ and $QQ^{T}\mathsf{V}_{i} = \mathsf{V}_{i}$ . Therefore, we can rewrite the left-hand side of (2.1) (without $\sqrt{n}\Omega_{jj}^{-1/2}$ ) as

$$
\Big (\Big (\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {g} _ {i} \mathsf {g} _ {i} ^ {T} \Big) ^ {1 / 2} \Big) ^ {\dagger} \Big (\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {V} _ {i} \Big) \hat {\mathsf {B}} ^ {T} e _ {j}
$$

$$
= \Big (\Big (\frac {1}{n} \sum_ {i = 1} ^ {n} Q Q ^ {T} \mathsf {g} _ {i} \mathsf {g} _ {i} ^ {T} Q Q ^ {T} \Big) ^ {1 / 2} \Big) ^ {\dagger} \Big (\frac {1}{n} \sum_ {i = 1} ^ {n} Q Q ^ {T} \mathsf {V} _ {i} Q Q ^ {T} \Big) \hat {\mathsf {B}} ^ {T} e _ {j}
$$

$$
= \Big (\Big (\frac {1}{n} \sum_ {i = 1} ^ {n} Q g _ {i} g _ {i} ^ {T} Q ^ {T} \Big) ^ {1 / 2} \Big) ^ {\dagger} \Big (\frac {1}{n} \sum_ {i = 1} ^ {n} Q V _ {i} Q ^ {T} \Big) \hat {\mathsf {B}} ^ {T} e _ {j}
$$

$$
= Q \left(\left(\frac {1}{n} \sum_ {i = 1} ^ {n} g _ {i} g _ {i} ^ {T}\right) ^ {1 / 2}\right) ^ {\dagger} Q ^ {T} Q \left(\frac {1}{n} \sum_ {i = 1} ^ {n} V _ {i} Q ^ {T}\right) \hat {\mathsf {B}} ^ {T} e _ {j}
$$

$$
= Q \left(\left(\frac {1}{n} \sum_ {i = 1} ^ {n} g _ {i} g _ {i} ^ {T}\right) ^ {1 / 2}\right) ^ {\dagger} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} V _ {i}\right) \hat {B} ^ {T} e _ {j},
$$

where the first equality uses $QQ^{T}g_{i}=g_{i}$ and $QQ^{T}V_{i}=V_{i}$ , the second equality uses $g_{i}=Q^{T}g_{i}$ and $V_{i}=Q^{T}V_{i}Q$ from Lemma S3.2, the third equality follows from the same argument of (S3.7), and the last equality uses $Q^{T}Q=I_{K}$ and $\hat{B}=\hat{B}Q$ .

Therefore, Theorem S3.1 implies that the limiting covariance for the left-hand side of (2.1) is $QQ^{T} = I_{K} - \frac{11^{T}}{K+1}$ . This completes the proof. □

# S4. PROOF OF THEOREM 2.2

We restate Theorem 2.2 for convenience.

Theorem 2.2. Define the matrix $R = (I_K, \mathbf{0}_K)^T \in \mathbb{R}^{(K+1) \times K}$ using block matrix notation. Let Assumptions 2.1 to 2.4 be fulfilled. For $\hat{A}$ in (1.9) and any $j \in [p]$ such that $H_0$ in (1.10) holds, (2.2)

$$
\underbrace {\left(I _ {K} + \frac {\mathbf {1} _ {K} \mathbf {1} _ {K} ^ {T}}{\sqrt {K + 1} + 1}\right) R ^ {T}} _ {\text {matrix} \mathbb {R} ^ {K \times (K + 1)}} \underbrace {\sqrt {\frac {n}{\Omega_ {j j}}}} _ {\text {scalar}} \Big (\underbrace {\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {g} _ {i} \mathsf {g} _ {i} ^ {T}\right) ^ {1 / 2}} _ {\text {matrix} \mathbb {R} ^ {(K + 1) \times (K + 1)}} \Big) ^ {\dagger} \underbrace {\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {V} _ {i} R\right)} _ {\mathbb {R} ^ {(K + 1) \times K}} \underbrace {\hat {A} ^ {T} e _ {j}} _ {\mathbb {R} ^ {K}} \xrightarrow {\mathrm{d}} N (\mathbf {0} _ {K}, I _ {K})
$$

where $g_{i}$ is defined in (1.7) and $V_{i}$ is defined in Theorem 2.1. Furthermore, for the same $j \in [p]$ ,
(2.3)

$$
\mathcal {T} _ {n} ^ {j} (X, Y) := \frac {n}{\Omega_ {j j}} \left\| \left(\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \mathrm{g} _ {i} \mathrm{g} _ {i} ^ {T}\right) ^ {1 / 2}\right) ^ {\dagger} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \mathrm{V} _ {i}\right) R \hat {A} ^ {T} e _ {j} \right\| ^ {2} s a t i s f i e s \mathcal {T} _ {n} ^ {j} (X, Y) \xrightarrow {\mathrm{d}} \chi_ {K} ^ {2}.
$$

The proof is a direct consequence of Theorem 2.1.

Proof of Theorem 2.2. By definition of $\hat{A}$ in (1.9), we have $\hat{A} = \hat{\mathsf{B}}(I_K, -\mathbf{1}_K)^T$ and

$$
\hat {A} \left(I _ {K}, \mathbf {0} _ {K}\right) = \hat {\mathsf {B}} (I _ {K}, - \mathbf {1} _ {K}) ^ {T} (I _ {K}, \mathbf {0} _ {K}) = \hat {\mathsf {B}} (I _ {K + 1} - e _ {K + 1} \mathbf {1} ^ {T}) = \hat {\mathsf {B}} - (\hat {\mathsf {B}} e _ {K + 1}) \mathbf {1} ^ {T},
$$

which is of the form $\hat{\mathsf{B}} - b\mathbf{1}^T$ with $b = \hat{\mathsf{B}}e_{K + 1}$ . Therefore, $\hat{A}(I_K,\mathbf{0}_K)$ is also a solution of (1.5). Taking $\hat{\mathsf{B}}$ in Theorem 2.1 to be $\hat{A}(I_K,\mathbf{0}_K) = \hat{A}R^T$ gives the desired $\chi^2$ result (2.3)

and

$$
\sqrt {n} \Omega_ {j j} ^ {- 1 / 2} \left(\left(\frac {1}{n} \sum_ {i = 1} ^ {n} \mathbf {g} _ {i} \mathbf {g} _ {i} ^ {T}\right) ^ {1 / 2}\right) ^ {\dagger} \left(\frac {1}{n} \sum_ {i = 1} ^ {n} \mathsf {V} _ {i}\right) R \hat {A} ^ {T} e _ {j} \xrightarrow {\mathrm{d}} N \left(0, I _ {K + 1} - \frac {\mathbf {1 1} ^ {T}}{K + 1}\right). \tag {S4.1}
$$

Multiplying $(R^{T}(I_{K+1}-\frac{\mathbf{1}\mathbf{1}^{T}}{K+1})R)^{-1/2}R^{T}$ to the left of the above display gives the desired normality result (2.2) by observing $(R^{T}(I_{K+1}-\frac{\mathbf{1}\mathbf{1}^{T}}{K+1})R)^{-1/2}=(I_{K}+\frac{\mathbf{1}_{K}\mathbf{1}_{K}^{T}}{\sqrt{K+1+1}})$ . This completes the proof. □

# S5. PRELIMINARY RESULTS FOR PROVING THEOREM S3.1

S5.1. Results for general loss functions. In this subsection, we will work under the following assumptions with a general convex loss function. Later in Section S6, we will apply the general results of this subsection to the multinomial logistic loss discussed in Section S3.1.

Assumption S5.1. Suppose we have data $(Y,X)$ , where $Y \in \mathbb{R}^{n \times (K+1)}$ with rows $(y_{1},...,y_{n})$ , and $X \in R^{n \times p}$ has i.i.d. rows $(x_{1},...,x_{n})$ with $x_{i} \sim N(\mathbf{0},\Sigma)$ and invertible $\Sigma$ . The observations $(y_{i},x_{i})_{i \in [n]}$ are i.i.d. and $y_{i}$ has the form $y_{i} = f(U_{i},x_{i}^{T}B^{*})$ for some deterministic function f, deterministic $B^{*} \in R^{p \times K}$ , and latent random variable $U_{i}$ independent of $x_{i}$ . Assume $p/n \leq \delta^{-1} < 1$ .

Assumption S5.2. Given data $(Y, X)$ , consider twice continuously differentiable and strictly convex loss functions $(L_{i})_{i \in [n]}$ with each $L_{i} : R^{K} \to R$ depending on $y_{i}$ but not on $x_{i}$ .

Provided that the following minimization problem admits a solution, define

$$
\hat {B} (Y, X) = \underset {B \in \mathbb {R} ^ {p \times K}} {\arg \min} \sum_ {i = 1} ^ {n} L _ {i} (B ^ {T} x _ {i}).
$$

Define for each $i \in [n]$ ,

$$
g _ {i} (Y, X) = \nabla L _ {i} (\hat {B} (Y, X) ^ {T} x _ {i}), H _ {i} (Y, X) = \nabla^ {2} L _ {i} (\hat {B} (Y, X) ^ {T} x _ {i}),
$$

so that $g_{i}(Y,X)\in \mathbb{R}^{K}$ and $H_{i}(Y,X)\in \mathbb{R}^{K\times K}$ . Define

$$
G (Y, X) = \sum_ {i = 1} ^ {n} e _ {i} g _ {i} (Y, X) ^ {T},
$$

$$
V (Y, X) = \sum_ {i = 1} ^ {n} \Bigl (H _ {i} (Y, X) - (H _ {i} (Y, X) \otimes x _ {i} ^ {T}) \Bigl [ \sum_ {l = 1} ^ {n} H _ {l} (Y, X) \otimes (x _ {l} x _ {l} ^ {T}) \Bigr ] ^ {\dagger} (H _ {i} (Y, X) \otimes x _ {i}) \Bigr),
$$

so that $G(Y, X) \in \mathbb{R}^{n \times K}$ and $V(Y, X) \in \mathbb{R}^{K \times K}$ . If the dependence on data $(Y, X)$ is clear from context, we will simply write $\hat{B}$ , $g_i$ , $H_i$ , G, and V.

Theorem S5.1. Let Assumptions S5.1 and S5.2 be fulfilled. Let $c_{*}, m_{*}, m^{*}$ , K be positive constants independent of n, p. Let $U^{*} \subset \mathbb{R}^{p \times (K+1)} \times \mathbb{R}^{n \times p}$ be an open set satisfying

(1) If $\{(Y,X)\in U^{*}\}$ , then the minimizer $\hat{B} (Y,X)$ in Assumption S5.2 exists, $H_{i}\preceq I_{K}$ for each $i\in [n]$ , $\frac{1}{n}\sum_{i = 1}^{n}H_{i}(Y,X)\otimes (x_{i}x_{i}^{T})\succeq c_{*}(I_{K}\otimes \Sigma)$ and $m_*I_K\preceq \frac{1}{n} G(Y,X)^T G(Y,X)\preceq m^* I_K$ .   
(2) For any $\{(Y,X),(Y,\tilde{X})\} \subset U^{*}$ , $\| G(Y,X) - G(Y,\tilde{X})\|_{F} \leq L\| (X - \tilde{X})\Sigma^{-1/2}\|_{F}$ holds for some positive constant $L$ .

Then for any $j \in [p]$ such that $e_j^T B^* = \mathbf{0}_K^T$ , there exists a random variable $\xi \in \mathbb{R}^K$ such that

$$
\mathbb {E} \big [ I \{(Y, X) \in U ^ {*} \} \big \| \frac {(G ^ {T} G) ^ {- 1 / 2} V \hat {B} ^ {T} e _ {j}}{\sqrt {\Omega_ {j j}}} - \xi \big \| ^ {2} \big ] \leq \frac {C}{p - K},
$$

and $\mathbb{P}(\| \xi \| ^2 >\chi_K^2 (\alpha))\leq \alpha$ for all $\alpha \in (0,1)$ , $C$ is a positive constant depending on $(c_{*},m_{*},m^{*},K,L)$ only. If additionally $\mathbb{P}((Y,X)\in U^{*})\to 1$ , then $\xi$ in the previous display satisfies $\xi \xrightarrow{\mathrm{d}} N(\mathbf{0},I_K)$ and

$$
\frac {(G ^ {T} G) ^ {- 1 / 2} V \hat {B} ^ {T} e _ {j}}{\sqrt {\Omega_ {j j}}} \xrightarrow {\mathrm{d}} N (\mathbf {0}, I _ {K}).
$$

The proof of Theorem S5.1 is given in next subsection.

S5.2. Proof of Theorem S5.1. In this subsection and next subsection, we will slightly abuse the notations $A^*$ and $\hat{A}$ , which have different definitions than the definitions in the main text.

Let $\Sigma^{1/2}B^{*} = \sum_{k=1}^{K}s_{k}u_{k}v_{k}^{T}$ be the singular value decomposition of $\Sigma^{1/2}B^{*}$ , where $u_{1},\ldots,u_{K}$ are the left singular vectors and $v_{1},\ldots,v_{k}$ the right singular vectors. If $\Sigma^{1/2}B^{*}$ is of rank strictly less than $K$ , we allow some $s_k$ to be equal to 0 so that $\Sigma^{1/2}B^{*} = \sum_{k=1}^{K}s_{k}u_{k}v_{k}^{T}$ still holds with orthonormal $(u_{1},\ldots,u_{K})$ and orthonormal $(v_{1},\ldots,v_{K})$ . We consider an orthogonal matrix $\tilde{P}\in\mathbb{R}^{p\times p}$ such that

$$
\tilde {P} \tilde {P} ^ {T} = \tilde {P} ^ {T} \tilde {P} = I _ {p}, \quad \tilde {P} \frac {\Sigma^ {- 1 / 2} e _ {j}}{\| \Sigma^ {- 1 / 2} e _ {j} \|} = e _ {1}, \quad \tilde {P} u _ {k} = e _ {p - K + k}, \quad \forall k \in [ K ]. \tag {S5.1}
$$

Since $e_j^T B^* = \mathbf{0}^T$ implies $e_j^T \Sigma^{-1/2} u_k = 0$ , we can always find a matrix $\tilde{P}$ satisfying (S5.1). From now on we fix this matrix $\tilde{P}$ and consider the following change of variable,

$$
Z = X \Sigma^ {- 1 / 2} \tilde {P} ^ {T}, \quad A ^ {*} = \tilde {P} \Sigma^ {1 / 2} B ^ {*}. \tag {S5.2}
$$

It immediately follows that $Z$ has i.i.d. $N(0,1)$ entries and the first $p - K$ rows of $A^*$ are all zeros. Since the response $y_i$ has the expression $y_i = f(U_i, x_i^T B^*)$ , $Y$ is unchanged by the change of variable (S5.2) from $ZA^* = XB^*$ . We now work on the multinomial logistic estimation with data $(Y,Z)$ and the underlying coefficient matrix $A^*$ in (S5.2). Parallel to the estimate $\hat{B}$ of $B^*$ in Assumption S5.2, we define the estimate of $A^*$ using data $(Y,Z)$ as

$$
\hat {A} (Y, Z) = \underset {A \in \mathbb {R} ^ {p \times K}} {\arg \min} \sum_ {i} L _ {i} (A ^ {T} z _ {i}),
$$

where $z_{i} = Z^{T}e_{i}$ is the i-th row of Z. By construction, we have $\hat{A} = \tilde{P}\Sigma^{1/2}\hat{B}$ , hence $Z\hat{A} = X\hat{B}$ and $e_{1}^{T}\hat{A} = e_{j}^{T}\hat{B}/\sqrt{\Omega_{jj}}$ . Furthermore, the quantities depending on $(Y, X\hat{B})$ remain unchanged after the change of variable. In particular, the gradient and Hessian

$$
\nabla L _ {i} (\hat {B} ^ {T} x _ {i}) = \nabla L _ {i} (\hat {A} ^ {T} z _ {i}), \qquad \nabla^ {2} L _ {i} (\hat {B} ^ {T} x _ {i}) = \nabla^ {2} L _ {i} (\hat {A} ^ {T} z _ {i})
$$

are unchanged. It follows that the matrix G and V are unchanged. Therefore, we have

$$
\frac {e _ {j} ^ {T} \hat {B} V (G ^ {T} G) ^ {- 1 / 2}}{\sqrt {\Omega_ {j j}}} = e _ {1} ^ {T} \hat {A} V (G ^ {T} G) ^ {- 1 / 2}.
$$

In conclusion, with the change of variables (S5.2), we only need to prove Theorem S5.1 in the special case, where the design matrix X i.i.d. $N(0,1)$ entries and the response Y is independent of the first p-K columns of X. To this end, we introduce the following Theorem S5.2, and the proof of Theorem S5.1 is a consequence of Theorem S5.2 as it proves the desired result for $e_{1}^{T}\hat{A}V(G^{T}G)^{-1/2}$ .

Theorem S5.2. Let $c_{*}, m_{*}, m^{*}$ , K be constants independent of n, p. Let $Z \in R^{n \times p}$ have i.i.d. rows $(z_{1}, \ldots, z_{n})$ with $z_{i} \sim N(\mathbf{0}, I_{p})$ . Let $y_{1}, \ldots, y_{n} \in \mathbb{R}^{(K+1)}$ such that $(y_{1}, \ldots, y_{n})$ is independent of the first p - K columns of Z. Consider twice continuously differentiable and strictly convex loss functions $(L_{i})_{i=1,\ldots,n}$ with each $L_{i}: R^{K} \to R$ depending on $y_{i}$ but not on $z_{i}$ and define, provided that the minimizer admits a solution,

$$
\hat {A} (Y, Z) = \underset {A \in \mathbb {R} ^ {p \times K}} {\arg \min} \sum_ {i = 1} ^ {n} L _ {i} (A ^ {T} z _ {i}), \quad g _ {i} (Y, Z) = \nabla L _ {i} (\hat {A} (Y, Z) ^ {T} z _ {i}), \quad H _ {i} (Y, Z) = \nabla^ {2} L _ {i} (\hat {A} (Y, Z) ^ {T} z _ {i}),
$$

$G(Y,Z) = \sum_{i=1}^{n} e_i g_i(Y,Z)^T \in \mathbb{R}^{n \times K}$ , and $V(Y,Z) = \sum_{i=1}^{n} (H_i - (H_i \otimes z_i^T)) [\sum_{l=1}^{n} H_l \otimes (z_l z_l^T)]^\dagger (H_i \otimes z_i)) \in \mathbb{R}^{K \times K}$ , where we dropped the dependence of $H_i$ on $(Y,Z)$ for simplicity. Let $O \subset \mathbb{R}^{n \times (K+1)} \times \mathbb{R}^{n \times p}$ be an open set satisfying

- If $(Y,Z) \in O$ , then the minimizer $\hat{A}(Y,Z)$ exists, $H_i \preceq I_K$ for each $i \in [n]$ , $c_*I_{pK} \preceq \frac{1}{n}\sum_{i=1}^{n} H_i(Y,Z) \otimes (z_i z_i^T)$ , and $m_*I_K \preceq \frac{1}{n}\sum_{i=1}^{n} G(Y,Z)^T G(Y,Z) \preceq m^*I_K$ .   
- With the notation $G(Y,Z) = \sum_{i=1}^{n} e_i g_i(Y,Z)^T$ , we have if two $Z, \tilde{Z} \in \mathbb{R}^{n \times p}$ satisfy $\{(Y,Z),(Y,\tilde{Z})\} \subset O$ then $\|G(Y,Z) - G(Y,\tilde{Z})\| \leq L \|Z - \tilde{Z}\|$ .

For $e_1 \in \mathbb{R}^p$ the first canonical basis vector, there exists a random variable $\xi \in \mathbb{R}^K$ such that

$$
\mathbb {E} \big [ I \{(Y, Z) \in O \} \big \| (G ^ {T} G) ^ {- 1 / 2} V \hat {A} ^ {T} e _ {1} - \xi \big \| ^ {2} \big ] \leq \frac {C}{p - K},
$$

and $\mathbb{P}(\| \xi \| ^2 >\chi_K^2 (\alpha))\leq \alpha$ for all $\alpha \in (0,1)$ , $C$ is a positive constant depending on $(c_{*},m_{*},m^{*},K,L)$ only. If additionally $\mathbb{P}((Y,Z)\in O)\to 1$ , then $\xi$ in the previous display satisfies $\xi \xrightarrow{\mathrm{d}} N(\mathbf{0},I_K)$ and

$$
e _ {1} ^ {T} \hat {A} V (G ^ {T} G) ^ {- 1 / 2} {\xrightarrow {\mathrm{d}}} N (\mathbf {0}, I _ {K}).
$$

The proof of Theorem S5.2 is presented in Section S5.3.

S5.3. Proof of Theorem S5.2. We first present a few useful lemmas, whose proofs are given at the end of this subsection.

Lemma S5.3 (Proof is given on page 28). Let $z \sim N(\mathbf{0}, \sigma^2 I_n)$ and $F: \mathbb{R}^n \to \mathbb{R}^{n \times K}$ be weakly differentiable with $\mathbb{E}\| F(z)\|_F^2 < \infty$ . Let $\tilde{z}$ be an independent copy of $z$ . Then

$$
\mathbb {E} \Big [ \Big \| z ^ {T} F (z) - \sigma^ {2} \sum_ {i = 1} ^ {n} \frac {\partial e _ {i} ^ {T} F (z)}{\partial z _ {i}} - z ^ {T} F (\tilde {z}) \Big \| ^ {2} \Big ] \leq 3 \sigma^ {4} \mathbb {E} \sum_ {i = 1} ^ {n} \Big \| \frac {\partial F (z)}{\partial z _ {i}} \Big \| _ {F} ^ {2}.
$$

Lemma S5.4 (Proof is given on page 28). If $G, \tilde{G} \in \mathbb{R}^{n \times K}$ satisfy $m_* I_K \preceq \frac{1}{n} G^T G \preceq m^* I_K$ and $m_* I_K \preceq \frac{1}{n} \tilde{G}^T \tilde{G} \preceq m^* I_K$ for some positive constants $m_*$ and $m^*$ . Then

$$
\begin{array}{l} \| (G ^ {T} G) ^ {- 1 / 2} - (\tilde {G} ^ {T} \tilde {G}) ^ {- 1 / 2} \| _ {F} \leq L _ {1} n ^ {- 1} \| G - \tilde {G} \| _ {F}, \\ \| G (G ^ {T} G) ^ {- 1 / 2} - \tilde {G} (\tilde {G} ^ {T} \tilde {G}) ^ {- 1 / 2} \| _ {F} \leq L _ {2} n ^ {- 1 / 2} \| G - \tilde {G} \| _ {F}, \\ \end{array}
$$

where $L_{1}, L_{2}$ are positive constants depending on $(K, m_{*}, m^{*})$ only.

Lemma S5.5 (Proof is given on page 29). Let the assumptions in Theorem S5.2 be fulfilled. Let $Y \in \mathbb{R}^{n \times (K + 1)}$ be fixed. If a minimizer $\hat{A}(Y, Z)$ exists at $Z$ , then $Z \mapsto \hat{A}(Y, Z)$ exists and is differentiable in a neighborhood of $Z$ with derivative

$$
\begin{array}{l} \frac {\partial \operatorname{vec} (\hat {A})}{\partial z _ {i j}} = - M [ g _ {i} \otimes e _ {j} + (H _ {i} \hat {A} ^ {T} e _ {j} \otimes z _ {i}) ], \\ \frac {\partial g _ {l}}{\partial z _ {i j}} = - (H _ {l} \otimes z _ {l} ^ {T}) M [ g _ {i} \otimes e _ {j} + (H _ {i} \hat {A} ^ {T} e _ {j} \otimes z _ {i}) ] + I (l = i) H _ {l} \hat {A} ^ {T} e _ {j}, \\ \end{array}
$$

where $M = [\sum_{i=1}^{n} H_i \otimes (z_i z_i^T)]^{-1}$ . It immediately follows that

$$
\frac {\partial g _ {i}}{\partial z _ {i j}} = [ H _ {i} - (H _ {i} \otimes z _ {i} ^ {T}) M (H _ {i} \otimes z _ {i}) ] \hat {A} ^ {T} e _ {j} - (H _ {i} \otimes z _ {i} ^ {T}) M (g _ {i} \otimes e _ {j}).
$$

Corollary S5.6 (Proof is given on page 30). Under the same conditions of Lemma S5.5, for $G = \sum_{i=1}^{n} e_i g_i^T$ , we have for each $i \in [n], j \in [p]$ ,

$$
\begin{array}{l} \sum_ {i = 1} ^ {n} \frac {\partial e _ {i} ^ {T} G (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} \\ = e _ {j} ^ {T} \hat {A} V _ {i} ^ {T} (G ^ {T} G) ^ {- 1 / 2} + \sum_ {i = 1} ^ {n} \Bigl [ - (g _ {i} ^ {T} \otimes e _ {j} ^ {T}) M (H _ {i} \otimes z _ {i}) (G ^ {T} G) ^ {- 1 / 2} + e _ {i} ^ {T} G \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} \Bigr ]. \\ \end{array}
$$

Now we are ready to prove Theorem S5.2.

Proof of Theorem S5.2. Let $h: O \to \mathbb{R}^{n \times K}$ be $h(Y, Z) = G(Y, Z)(G(Y, Z)^T G(Y, Z))^{-1/2}$ . In most of this proof, we will omit the dependence $(Y, Z)$ on $h, \hat{A}, g_i, H_i, G, V$ to lighten notation. By Lemma S5.4, we know this $h$ is $LL_2 n^{-1/2}$ -Lipschitz in the sense that $\| h(Y, Z) - h(Y, \tilde{Z}) \|_F \leq LL_2 n^{-1/2} \| Z - \tilde{Z} \|_F$ for all $\{(Y, Z), (Y, \tilde{Z})\} \subset O$ . By Kirszbraun theorem, there exists a function $H: \mathbb{R}^{n \times (K+1)} \times \mathbb{R}^{n \times p} \to \mathbb{R}^{n \times K}$ (an extension of $h$ from $O$ to $\mathbb{R}^{n \times (K+1)} \times \mathbb{R}^{n \times p}$ ) such that $H(Y, Z) = h(Y, Z)$ for all $(Y, Z) \in O$ , $\| H(Y, Z) \|_{op} \leq 1$ and $\| H(Y, Z) - H(Y, \tilde{Z}) \|_F \leq LL_2 n^{-1/2} \| Z - \tilde{Z} \|_F$ for all $\{(Y, Z), (Y, \tilde{Z})\} \subset \mathbb{R}^{n \times (K+1)} \times \mathbb{R}^{n \times p}$ .

For each $j \in [p]$ , let $z_{j} = Ze_{j}$ be the j-th column of Z to distinguish it from the notation $z_{i}$ , which means the i-th row of Z. Let $\check{z} \sim N(\mathbf{0}, I_{n})$ be an independent copy of each columns of Z, and $\check{Z}^{j} = Z(I_{p} - e_{j}e_{j}^{T}) + \check{z}e_{j}^{T}$ . That is, $\check{Z}^{j}$ replaces the j-th column of Z by $\check{z}$ . By definition, $z_{1} \perp \check{z}$ and $z_{1} \perp \check{Z}^{1}$ .

Let $\xi = -[H(Y,\check{Z}^1)]^T\mathbf{z}_1\in \mathbb{R}^K$ , then $\| \xi \| ^2\leq \| \mathbf{z}_1\|$ since $\| H(Y,\check{Z}^1)\|_{op}\leq 1$ . It follows that

$$
\mathbb {P} (\| \xi \| ^ {2} > \chi_ {K} ^ {2} (\alpha)) \leq \mathbb {P} (\| z _ {1} \| ^ {2} > \chi_ {K} ^ {2} (\alpha)) = \alpha .
$$

Note that the first p - K columns of Z are exchangeable, because they are i.i.d. and independent of the response Y, we have for each $\ell \in [p - K]$ ,

$$
\begin{array}{l} \mathbb {E} \left[ I \{(Y, Z) \in O \} \left\| (G ^ {T} G) ^ {- 1 / 2} V \hat {A} ^ {T} e _ {1} - \xi \right\| ^ {2} \right] \\ = \mathbb {E} \left[ I \{(Y, Z) \in O \} \| e _ {1} ^ {T} \hat {A} V (G ^ {T} G) ^ {- 1 / 2} + \mathbf {z} _ {1} ^ {T} H (Y, \check {Z} ^ {1}) \| ^ {2} \right] \\ = \mathbb {E} \Big [ I \{(Y, Z) \in O \} \| e _ {\ell} ^ {T} \hat {A} V (G ^ {T} G) ^ {- 1 / 2} + \mathbf {z} _ {\ell} ^ {T} H (Y, \check {Z} ^ {\ell}) \| ^ {2} \Big ], \\ \end{array}
$$

where the last line holds for any $\ell\in[p-K]$ because $(\mathbf{z}_{1},e_{1}^{T}\hat{A},\check{Z})\stackrel{d}{=}(z_{\ell},e_{\ell}^{T}\hat{A},\check{Z}^{\ell})$ . Therefore,

$$
\begin{array}{l} \mathbb {E} \left[ I \{(Y, Z) \in O \} \left\| e _ {1} ^ {T} \hat {A} V (G ^ {T} G) ^ {- 1 / 2} + \mathbf {z} _ {1} ^ {T} H (Y, \check {Z} ^ {1}) \right\| ^ {2} \right] \\ = \frac {1}{p - K} \sum_ {\ell = 1} ^ {p - K} \mathbb {E} \left[ I \{(Y, Z) \in O \} \left\| e _ {\ell} ^ {T} \hat {A} V (G ^ {T} G) ^ {- 1 / 2} + \mathbf {z} _ {\ell} ^ {T} H (Y, \check {Z} ^ {\ell}) \right\| ^ {2} \right] \\ = \frac {1}{p - K} \sum_ {\ell = 1} ^ {p - K} \mathbb {E} \left[ I \{(Y, Z) \in O \} \left\| \sum_ {i} \frac {\partial e _ {i} ^ {T} G (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i \ell}} + \mathbf {z} _ {\ell} ^ {T} H (Y, \check {Z} ^ {\ell}) - \operatorname{Rem} _ {\ell} \right\| ^ {2} \right], \\ \end{array}
$$

where $\mathrm{Rem}_{\ell} = \sum_{i=1}^{n}\left[-(g_i^T\otimes e_\ell^T)M(H_i\otimes z_i)(G^TG)^{-1/2} + e_i^TG\frac{\partial(G^TG)^{-1/2}}{\partial z_{i\ell}}\right]$ from Corollary S5.6, and $M = [\sum_{i=1}^{n}(H_i\otimes z_iz_i^T)]^{-1}$ from Lemma S5.5. Using $(a + b)^2\leq 2a^2 +2b^2$ , the above display can be bounded by sum of two terms, denoted by $(RHS)_1$ and $(RHS)_2$ .

For the first term,

$$
(R H S) _ {1} = \frac {2}{p - K} \sum_ {\ell = 1} ^ {p - K} \mathbb {E} \Big [ I \{(Y, Z) \in O \} \Big \| \sum_ {i} \frac {\partial e _ {i} ^ {T} h (Y , Z)}{\partial z _ {i \ell}} + \mathsf {z} _ {\ell} ^ {T} H (Y, \check {Z} ^ {\ell}) \Big \| ^ {2} \Big ].
$$

Let $F(\mathbf{z}_{\ell}) = H(Y, Z(I - e_{\ell}e_{\ell}^{T}) + \mathbf{z}_{\ell}e_{\ell}^{T}) = H(Y, Z)$ , then $F(\check{\mathbf{z}}) = H(Y, \check{Z}^{\ell})$ . Apply Lemma S5.3 to $F(\mathbf{z}_{\ell})$ conditionally on $Z(I - e_{\ell}e_{\ell}^{T})$ , we obtain

$$
\begin{array}{l} \mathbb {E} \Big [ I \{(Y, Z) \in O \} \Big \| \sum_ {i} \frac {\partial e _ {i} ^ {T} h (Y , Z)}{\partial z _ {i \ell}} + \mathbf {z} _ {\ell} ^ {T} F (\check {\mathbf {z}}) \Big \| ^ {2} \Big ] \\ = \mathbb {E} \Big [ I \{(Y, Z) \in O \} \Big \| \mathbf {z} _ {\ell} ^ {T} F (\mathbf {z} _ {\ell}) - \sum_ {i} \frac {\partial e _ {i} ^ {T} F (\mathbf {z} _ {\ell})}{\partial z _ {i \ell}} - \mathbf {z} _ {\ell} ^ {T} F (\check {\mathbf {z}}) \Big \| ^ {2} \Big ] \\ \leq \mathbb {E} \Big [ \Big \| \mathbf {z} _ {\ell} ^ {T} F (\mathbf {z} _ {\ell}) - \sum_ {i} \frac {\partial e _ {i} ^ {T} F (\mathbf {z} _ {\ell})}{\partial z _ {i \ell}} - \mathbf {z} _ {\ell} ^ {T} F (\check {\mathbf {z}}) \Big \| ^ {2} \Big ] \\ \leq 3 \sum_ {i} \mathbb {E} \left\| \frac {\partial F (\mathbf {z} _ {\ell})}{\partial z _ {i \ell}} \right\| _ {F} ^ {2} \\ = 3 \sum_ {i} \mathbb {E} \left\| \frac {\partial H (Y , Z)}{\partial z _ {i \ell}} \right\| _ {F} ^ {2}, \\ \end{array}
$$

where the first equality uses $\mathbf{z}_{\ell}^{T}h(Y,Z)=0$ from the KKT conditions $Z^{T}G=0$ and $h(Y,Z)=G(G^{T}G)^{-1/2}$ . It follows that

$$
(R H S) _ {1} \leq \frac {6}{p - K} \mathbb {E} \Big [ \sum_ {\ell = 1} ^ {p} \sum_ {i = 1} ^ {n} \left\| \frac {\partial H (Y , Z)}{\partial z _ {i \ell}} \right\| _ {F} ^ {2} \Big ].
$$

Note that the integrand in the last display is actually the squared Frobenius norm of the Jacobian of the mapping from $R^{n\times p}$ to $R^{n\times K}\colon Z\mapsto H(Y,Z)$ . This Jacobian is a matrix with nK rows and np columns, has rank at most nK and operator norm at most $LL_{2}n^{-1/2}$ because $Z\mapsto H(Y,Z)$ is $LL_{2}n^{-1/2}$ -Lipschitz from Lemma S5.4. Using $\|A\|_{F}^{2}\leq\operatorname{rank}(A)\|A\|_{op}^{2}$ , we obtain

$$
(R H S) _ {1} \leq 6 K (L L _ {2}) ^ {2} / (p - K).
$$

For the second term $(RHS)_{2}=\frac{2}{p-K}\sum_{\ell=1}^{p-K}\mathbb{E}\big[I\{(Y,Z)\in O\}\|\operatorname{Rem}_{\ell}\|^{2}\big]$ . By definition of $Rem_{\ell}$ and $(a+b)^{2}\leq2a^{2}+2b^{2}$ , we obtain

$$
\text {(S5.3)} \quad (R H S) _ {2} \leq \frac {4}{p - K} \sum_ {\ell = 1} ^ {p - K} \mathbb {E} \Big [ I \{(Y, Z) \in O \} \Big \| \sum_ {i = 1} ^ {n} (g _ {i} ^ {T} \otimes e _ {\ell} ^ {T}) M (H _ {i} \otimes z _ {i}) (G ^ {T} G) ^ {- 1 / 2} \Big \| ^ {2} \Big ]
$$

$$
+ \frac {4}{p - K} \sum_ {\ell = 1} ^ {p - K} \mathbb {E} \left[ I \{(Y, Z) \in O \} \left\| \sum_ {i = 1} ^ {n} e _ {i} ^ {T} G \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i \ell}} \right\| ^ {2} \right]. \tag {S5.4}
$$

We next bound (S5.3) and (S5.4) one by one. For (S5.3), we focus on the norm without $(G^{T}G)^{-1/2}$ which is $\|\sum_{i=1}^{n}(g_{i}^{T}\otimes e_{\ell}^{T})M(H_{i}\otimes z_{i})\|$ . With $\|a\|=\max_{u:\|u\|=1}a^{T}u$ in mind, let us multiply to the right by a unit vector $u\in R^{K}$ and instead bound

$$
\sum_ {i = 1} ^ {n} (g _ {i} ^ {T} \otimes e _ {\ell} ^ {T}) M (H _ {i} \otimes z _ {i}) u = \mathrm{Tr} \Big [ (I _ {K} \otimes e _ {\ell} ^ {T}) M \sum_ {i} (H _ {i} u \otimes z _ {i}) g _ {i} ^ {T} \Big ] \leq K \| (I _ {K} \otimes e _ {\ell} ^ {T}) M \| _ {o p} \| \sum_ {i} (H _ {i} u \otimes z _ {i}) g _ {i} ^ {T} \| _ {o p}
$$

because the rank of the matrix inside the trace is at most $K$ and $\mathrm{Tr}[\cdot] \leq K \| \cdot \|_{op}$ holds. Then

$$
\| \sum_ {i} (H _ {i} u \otimes z _ {i}) g _ {i} ^ {T} \| _ {o p} = \| (I _ {K} \otimes Z ^ {T}) \sum_ {i} (H _ {i} u \otimes e _ {i}) e _ {i} ^ {T} G \| _ {o p} \leq \| Z \| _ {o p} \| \sum_ {i} (H _ {i} u \otimes e _ {i} e _ {i} ^ {T}) \| _ {o p} \| G \| _ {o p}.
$$

Next, $\| \sum_{i}(H_{i}u\otimes e_{i}e_{i}^{T})\|_{op} = \| \sum_{i}(H_{i}\otimes e_{i}e_{i}^{T})(u\otimes I_{n})\|_{op}\leq 1$ because $H_{i}\preceq I_{K}$ and $\| u\| = 1$ . In summary, the norm in (S5.3) is bounded from above by

$$
K \| (G ^ {T} G) ^ {- 1 / 2} \| _ {o p} \| M \| _ {o p} \| Z \| _ {o p} \| G \| _ {o p}.
$$

To bound (S5.3), since in the event $(Y,Z)\in O$ , $m_{*}I_{K}\preceq\frac{1}{n}G^{T}G\preceq m^{*}I_{K}$ and $\frac{1}{n}\sum_{i=1}^{n}(H_{i}\otimes z_{i}z_{i}^{T})\succeq c_{*}I_{pK}$ , we have $\|G\|_{op}\leq\sqrt{m^{*}n}$ , hence $\|M\|_{op}\leq c_{*}^{-1}$ . Thus, the above display can be bounded by

$$
K (m _ {*} n) ^ {- 1 / 2} c _ {*} ^ {- 1} \| Z \| _ {o p} \sqrt {m ^ {*} n} = (m ^ {*} / m _ {*}) ^ {1 / 2} c _ {*} ^ {- 1} K \| Z \| _ {o p}.
$$

Since $Z \in \mathbb{R}^{n \times p}$ has i.i.d. $N(0,1)$ entries, [Davidson and Szarek, 2001, Theorem II.13] implies that $\mathbb{E}\| Z\|_{op} \leq \sqrt{n} + \sqrt{p} \leq 2\sqrt{n}$ . Therefore,

$$
(\mathrm{S} 5. 3) \leq C (c _ {*}, K, L) n ^ {- 1}.
$$

Now we bound (S5.4). Since

$$
\begin{array}{l} \sum_ {\ell = 1} ^ {p - K} \left\| \sum_ {i = 1} ^ {n} e _ {i} ^ {T} G \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i \ell}} \right\| ^ {2} \\ \leq \sum_ {j = 1} ^ {p} \left\| \sum_ {i = 1} ^ {n} e _ {i} ^ {T} G \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} \right\| ^ {2} \\ = \sum_ {j = 1} ^ {p} \sum_ {k ^ {\prime} = 1} ^ {K} \left(\sum_ {i = 1} ^ {n} \sum_ {k = 1} ^ {K} e _ {i} ^ {T} G e _ {k} e _ {k} ^ {T} \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} e _ {k ^ {\prime}}\right) ^ {2} \\ \leq \sum_ {j = 1} ^ {p} \sum_ {k ^ {\prime} = 1} ^ {K} \left[ \sum_ {i = 1} ^ {n} \sum_ {k = 1} ^ {K} (e _ {i} ^ {T} G e _ {k}) ^ {2} \sum_ {i = 1} ^ {n} \sum_ {k = 1} ^ {K} \left(e _ {k} ^ {T} \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} e _ {k ^ {\prime}}\right) ^ {2} \right] \\ = \| G \| _ {F} ^ {2} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {p} \left\| \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} \right\| ^ {2}. \\ \end{array}
$$

Using $\|G\|_{F}^{2}\leq nK$ , and the mapping $Z\mapsto(G^{T}G)^{-1/2}$ is $LL_{1}n^{-1}$ -Lipschitz on O using Lemma S5.4, we conclude that

$$
(\mathrm{S5.4}) \leq 4 K ^ {3} L L _ {1} / (p - K).
$$

Combining the above bounds on $(RHS)_{1}$ and $(RHS)_{2}$ , we have

$$
\mathbb {E} \Big [ I \{(Y, Z) \in O \} \| (G ^ {T} G) ^ {- 1 / 2} V \hat {A} ^ {T} e _ {1} - \xi \| ^ {2} \Big ] \leq \frac {C (c _ {*} , K , m _ {*} , m ^ {*} , L , L _ {1} , L _ {2})}{p - K}, \tag {S5.5}
$$

where the constant depends on $(c_{*}, K, m_{*}, m^{*}, L)$ only because $L_{1}$ and $L_{2}$ are constants depending on $(K, m_{*}, m^{*})$ only.

If additionally $\mathbb{P}((Y,Z)\in O)\to 1$ , we have $\mathbb{P}((Y,\check{Z}^1)\in O)\to 1$ using $(Y,Z)\stackrel {d}{=}(Y,\check{Z}^{1})$ . Therefore,

$$
\xi = - [ h (Y, \check {Z} ^ {1}) ^ {T} \mathbf {z} _ {1} ] I ((Y, \check {Z} ^ {1}) \in O) - [ H (Y, \check {Z} ^ {1}) ^ {T} \mathbf {z} _ {1} ] I ((Y, \check {Z} ^ {1}) \notin O) \xrightarrow {\mathrm{d}} N (\mathbf {0}, I _ {K}). \tag {S5.6}
$$

By (S5.5), we know $(G^{T}G)^{-1/2}V\hat{A}^{T}e_{1}-\xi\xrightarrow{\mathrm{d}}0$ when $\mathbb{P}((Y,Z)\in O)\to1$ . Hence, we conclude

$$
(G ^ {T} G) ^ {- 1 / 2} V \hat {A} ^ {T} e _ {1} {\xrightarrow {\mathrm{d}}} N (\mathbf {0}, I _ {K}) \quad \mathrm{and} \quad \| (G ^ {T} G) ^ {- 1 / 2} V \hat {A} ^ {T} e _ {1} \| ^ {2} {\xrightarrow {\mathrm{d}}} \chi_ {K} ^ {2}.
$$

![](images/8f7978b2256c555b1373e532d791e2821e6c266177f5957c424f143bc28a70e6.jpg)

We next prove Lemmas S5.3 to S5.5 and corollary S5.6.

Proof of Lemma S5.3. Let $z_0 = (z^T, \tilde{z}^T)^T \in \mathbb{R}^{2n}$ , then $z_0 \sim N(\mathbf{0}, \sigma^2 I_{2n})$ . For each $k \in [K]$ , let $f^{(k)}: \mathbb{R}^{2n} \to \mathbb{R}^{2n}$ be

$$
f ^ {(k)} (z _ {0}) = \binom {[ F (z) - F (\tilde {z}) ] e _ {k}} {0 _ {n}},
$$

so that $z_{0}^{T}f^{(k)}(z_{0}) = z^{T}[F(z) - F(\tilde{z})]e_{k}$ , and $\operatorname{div} f^{(k)}(z_{0}) = \sum_{i=1}^{n} \frac{\partial e_{i}^{T} F(z)e_{k}}{\partial z_{i}}$ . Applying the second order Stein formula [Bellec and Zhang, 2021] to $f^{(k)}$ gives, with Jac denoting the Jacobian,

$$
\begin{array}{l} \mathbb {E} \Big [ \Big (z ^ {T} F (z) e _ {k} - \sigma^ {2} \sum_ {i = 1} ^ {n} \frac {\partial e _ {i} ^ {T} F (z) e _ {k}}{\partial z _ {i}} - z ^ {T} F (\tilde {z}) e _ {k} \Big) ^ {2} \Big ] \\ = \mathbb {E} \left[ \left(z _ {0} ^ {T} f ^ {(k)} (z _ {0}) - \sigma^ {2} \operatorname{div} f ^ {(k)} (z _ {0})\right) ^ {2} \right] \\ = \sigma^ {2} \mathbb {E} \| f ^ {(k)} (z _ {0}) \| ^ {2} + \sigma^ {4} \mathbb {E} \mathrm{Tr} [ (\mathrm{Jac} f ^ {(k)} (z _ {0})) ^ {2} ] \\ = \sigma^ {2} \mathbb {E} \| [ F (z) - F (\tilde {z}) ] e _ {k} \| ^ {2} + \sigma^ {4} \mathbb {E} \operatorname{Tr} \Big [ \left( \begin{array}{c c} \mathrm{Jac} [ F (z) e _ {k} ] & - \mathrm{Jac} [ F (\tilde {z}) e _ {k} ] \\ 0 _ {n \times n} & 0 _ {n \times n} \end{array} \right) ^ {2} \Big ] \\ = 2 \sigma^ {2} \mathbb {E} \| [ F (z) - \mathbb {E} F (z) ] e _ {k} \| ^ {2} + \sigma^ {4} \mathbb {E} \operatorname{Tr} ((\mathrm{Jac} [ F (z) e _ {k} ]) ^ {2}) \\ \leq 3 \sigma^ {4} \mathbb {E} \| \operatorname{Jac} [ F (z) e _ {k} ] \| _ {F} ^ {2}, \\ \end{array}
$$

where the last inequality uses the Gaussian Poincaré inequality, and the Cauchy-Schwarz inequality $\mathrm{Tr}(A^{2}) \leq \|A\|_{F}^{2}$ . Summing over $k \in [K]$ gives the desired inequality. ☐

Proof of Lemma S5.4. We first prove $G \mapsto G^{T}G$ is Lipschitz by noting

$$
\begin{array}{l} \left\| G ^ {T} G - \tilde {G} ^ {T} \tilde {G} \right\| _ {o p} \\ = \left\| (G - \tilde {G}) ^ {T} G + \tilde {G} ^ {T} (G - \tilde {G}) \right\| _ {o p} \\ \leq \| G - \tilde {G} \| _ {o p} (\| G \| _ {o p} + \| \tilde {G} \| _ {o p}) \\ \leq 2 \sqrt {m ^ {*} n} \| G - \tilde {G} \| _ {o p}. \\ \end{array}
$$

Then we show $G^{T}G \mapsto (G^{T}G)^{-1}$ is Lipschitz. Let $A = G^{T}G$ and $\tilde{A} = \tilde{G}^{T}\tilde{G}$ , we have

$$
\begin{array}{l} \| A ^ {- 1} - \tilde {A} ^ {- 1} \| _ {o p} \\ = \| A ^ {- 1} (\tilde {A} - A) \tilde {A} ^ {- 1} \| _ {o p} \\ \leq \| A - \tilde {A} \| _ {o p} \| A ^ {- 1} \| _ {o p} \| \tilde {A} ^ {- 1} \| _ {o p} \\ \leq (m _ {*} n) ^ {- 2} \| A - \tilde {A} \| _ {o p}. \\ \end{array}
$$

We next prove $(G^{T}G)^{-1}\mapsto(G^{T}G)^{-1/2}$ is Lipschitz. Let $S=(G^{T}G)^{-1}$ , $S'=(\tilde{G}^{T}\tilde{G})^{-1}$ , and if u with $\|u\|=1$ is the eigenvector of $\sqrt{S}-\sqrt{\tilde{S}}$ with eigenvalue d, then

$$
\begin{array}{l} u ^ {T} (S - \tilde {S}) u = u ^ {T} (\sqrt {S} - \sqrt {\tilde {S}}) \sqrt {S} u + u ^ {T} \sqrt {\tilde {S}} (\sqrt {S} - \sqrt {\tilde {S}}) u \\ = d u ^ {T} \sqrt {S} u + d u ^ {T} \sqrt {\tilde {S}} u \\ = d u ^ {T} (\sqrt {S} + \sqrt {\tilde {S}}) u. \\ \end{array}
$$

As $d$ can be chosen as $\pm \| \sqrt{S} -\sqrt{\tilde{S}}\|_{op}$ (this argument is a special case of the Hemmen-Ando inequality [van Hemmen and Ando, 1980]), this implies

$$
\| \sqrt {S} - \sqrt {\tilde {S}} \| _ {o p} = \frac {| u ^ {T} (S - \tilde {S}) u |}{u ^ {T} (\sqrt {S} + \sqrt {\tilde {S}}) u} \leq \frac {\| S - \tilde {S} \| _ {o p}}{\lambda_ {\mathrm{min}} (\sqrt {S} + \sqrt {\tilde {S}})} \leq \frac {\| S - \tilde {S} \| _ {o p}}{2 / \sqrt {m ^ {*} n}}.
$$

Combining the above Lipschitz results, we have

$$
\| (G ^ {T} G) ^ {- 1 / 2} - (\tilde {G} ^ {T} \tilde {G}) ^ {- 1 / 2} \| _ {o p} \leq (m ^ {*} n) ^ {1 / 2} (m _ {*} n) ^ {- 2} (m ^ {*} n) ^ {1 / 2} \| G - \tilde {G} \| _ {o p} = \frac {m ^ {*}}{m _ {*} ^ {2}} n ^ {- 1} \| G - \tilde {G} \| _ {o p}.
$$

It immediately follows that

$$
\| (G ^ {T} G) ^ {- 1 / 2} - (\tilde {G} ^ {T} \tilde {G}) ^ {- 1 / 2} \| _ {F} \leq \sqrt {K} \frac {m ^ {*}}{m _ {*} ^ {2}} n ^ {- 1} \| G - \tilde {G} \| _ {F}.
$$

That is, the mapping $G \mapsto (G^T G)^{-1/2}$ is $L_1 n^{-1}$ -Lipschitz, where $L_1 = \sqrt{K} m_*^{-2} m^*$ .

For the second statement, the result follows by

$$
\begin{array}{l} \left\| G (G ^ {T} G) ^ {- 1 / 2} - \tilde {G} (\tilde {G} ^ {T} \tilde {G}) ^ {- 1 / 2} \right\| _ {o p} \\ \leq \| G - \tilde {G} \| _ {o p} \| (G ^ {T} G) ^ {- 1 / 2} \| _ {o p} + \| \tilde {G} \| _ {o p} \| (G ^ {T} G) ^ {- 1 / 2} - (\tilde {G} ^ {T} \tilde {G}) ^ {- 1 / 2} \| _ {o p} \\ \leq \| G - \tilde {G} \| _ {o p} (m _ {*} n) ^ {- 1 / 2} + (m ^ {*} n) ^ {1 / 2} L _ {1} n ^ {- 1} \| G - \tilde {G} \| _ {o p} \\ \end{array}
$$

Hence,

$$
\begin{array}{l} \left\| G \left(G ^ {T} G\right) ^ {- 1 / 2} - \tilde {G} \left(\tilde {G} ^ {T} \tilde {G}\right) ^ {- 1 / 2} \right\| _ {F} \\ \leq \sqrt {K} \big (m _ {*} ^ {- 1 / 2} + (m ^ {*}) ^ {1 / 2} L _ {1} \big) n ^ {- 1 / 2} \| G - \tilde {G} \| _ {F}, \\ \end{array}
$$

where $L_{2}=\sqrt{K}\big(m_{*}^{-1/2}+(m^{*})^{1/2}L_{1}\big)$ .

Proof of Lemma S5.5. Recall the KKT conditions $\sum_{l=1}^{n}z_{l}g_{l}^{T}=0_{p\times K}$ . We look for the derivative with respect to $z_{ij}$ . Denoting derivatives with a dot, we find by the chain rule and product rule

$$
\dot {z} _ {l} = \frac {\partial z _ {l}}{\partial z _ {i j}} = I (l = i) e _ {j},
$$

$$
\dot {g} _ {l} = \frac {\partial g _ {l}}{\partial z _ {i j}} = \frac {\partial g _ {l}}{\partial \hat {A} ^ {\top} z _ {l}} \frac {\partial \hat {A} ^ {\top} z _ {l}}{\partial z _ {i j}} = H _ {l} [ \dot {A} ^ {\top} z _ {l} + I (l = i) \hat {A} ^ {\top} e _ {j} ].
$$

Thus, differentiating the KKT conditions w.r.t. $z_{ij}$ by the product rule gives

$$
\sum_ {l = 1} ^ {n} \left[ I (l = i) e _ {j} g _ {l} ^ {T} + x _ {l} \left(\dot {A} ^ {\top} z _ {l} + I (l = i) \hat {A} ^ {\top} e _ {j}\right) ^ {T} H _ {l} \right] = 0.
$$

That is,

$$
e _ {j} g _ {i} ^ {T} + \sum_ {l = 1} ^ {n} z _ {l} z _ {l} ^ {T} \dot {A} H _ {l} + z _ {i} e _ {j} ^ {T} \hat {A} H _ {i} = 0.
$$

We then move the term involving $\dot{A}$ to one side, and vectorize both sides,

$$
g _ {i} \otimes e _ {j} + (H _ {i} \hat {A} ^ {T} e _ {j} \otimes z _ {i}) = - \sum_ {l = 1} ^ {n} (H _ {l} \otimes z _ {l} z _ {l} ^ {T}) \operatorname{vec} (\dot {A}).
$$

With $M = [\sum_{l=1}^{n}(H_l \otimes z_l z_l^T)]^{-1}$ , we obtain

$$
\mathrm{vec} (\dot {A}) = - M [ g _ {i} \otimes e _ {j} + (H _ {i} \hat {A} ^ {T} e _ {j} \otimes z _ {i}) ].
$$

Hence, using $\mathrm{vec}(H_l\dot{A}^\top z_l) = \mathrm{vec}(z_l^T\dot{A} H_l) = (H_l\otimes z_l^T)\mathrm{vec}(\dot{A})$ gives

$$
\begin{array}{l} \dot {g} _ {l} = (H _ {l} \otimes z _ {l} ^ {T}) \mathrm{vec} (\dot {A}) + I (l = i) H _ {l} \hat {A} ^ {T} e _ {j} \\ = - (H _ {l} \otimes z _ {l} ^ {T}) M [ g _ {i} \otimes e _ {j} + (H _ {i} \hat {A} ^ {T} e _ {j} \otimes z _ {i}) ] + I (l = i) H _ {l} \hat {A} ^ {T} e _ {j}. \\ \end{array}
$$

Thus,

$$
\begin{array}{l} \dot {g} _ {i} = - (H _ {i} \otimes z _ {i} ^ {T}) M [ g _ {i} \otimes e _ {j} + (H _ {i} \hat {A} ^ {T} e _ {j} \otimes z _ {i}) ] + H _ {i} \hat {A} ^ {T} e _ {j} \\ = - (H _ {i} \otimes z _ {i} ^ {T}) M (H _ {i} \hat {A} ^ {T} e _ {j} \otimes z _ {i}) + H _ {i} \hat {A} ^ {T} e _ {j} - (H _ {i} \otimes z _ {i} ^ {T}) M (g _ {i} \otimes e _ {j}) \\ = \left[ H _ {i} - \left(H _ {i} \otimes z _ {i} ^ {T}\right) M \left(H _ {i} \otimes z _ {i}\right) \right] \hat {A} ^ {T} e _ {j} - \left(H _ {i} \otimes z _ {i} ^ {T}\right) M \left(g _ {i} \otimes e _ {j}\right) \\ = V _ {i} \hat {A} ^ {T} e _ {j} - (H _ {i} \otimes z _ {i} ^ {T}) M (g _ {i} \otimes e _ {j}), \\ \end{array}
$$

where $V_{i}=[H_{i}-(H_{i}\otimes z_{i}^{T})M(H_{i}\otimes z_{i})]$ .

![](images/2c531e16acb618b92a636c5f3ac2de35f017f4e7bdbf91a770daee477958da3a.jpg)

Proof of Corollary S5.6. For each $i \in [n], j \in [p]$ , we have by the product rule

$$
\begin{array}{l} \frac {\partial e _ {i} ^ {T} G (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} \\ = \frac {\partial g _ {i} ^ {T}}{\partial z _ {i j}} (G ^ {T} G) ^ {- 1 / 2} + e _ {i} ^ {T} G \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} \\ = \left[ V _ {i} \hat {A} ^ {T} e _ {j} - \left(H _ {i} \otimes z _ {i} ^ {T}\right) M \left(g _ {i} \otimes e _ {j}\right) \right] ^ {T} \left(G ^ {T} G\right) ^ {- 1 / 2} + e _ {i} ^ {T} G \frac {\partial \left(G ^ {T} G\right) ^ {- 1 / 2}}{\partial z _ {i j}} \\ = e _ {j} ^ {T} \hat {A} V _ {i} (G ^ {T} G) ^ {- 1 / 2} + \Big [ - (g _ {i} ^ {T} \otimes e _ {j} ^ {T}) M (H _ {i} \otimes z _ {i}) (G ^ {T} G) ^ {- 1 / 2} + e _ {i} ^ {T} G \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} \Big ]. \\ \end{array}
$$

With $V = \sum_{i=1}^{n} V_i$ , we further have

$$
\begin{array}{l} \sum_ {i = 1} ^ {n} \frac {\partial e _ {i} ^ {T} G (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} \\ = e _ {j} ^ {T} \hat {A} V _ {i} ^ {T} (G ^ {T} G) ^ {- 1 / 2} + \sum_ {i = 1} ^ {n} \Bigl [ - (g _ {i} ^ {T} \otimes e _ {j} ^ {T}) M (H _ {i} \otimes z _ {i}) (G ^ {T} G) ^ {- 1 / 2} + e _ {i} ^ {T} G \frac {\partial (G ^ {T} G) ^ {- 1 / 2}}{\partial z _ {i j}} \Bigr ]. \\ \end{array}
$$

![](images/cf7cdeeb9c0e4247287ef748f7396726a4359715efad444492040a7077d1dcea.jpg)

# S6. PROOF OF THEOREM S3.1

Recall that Theorem S5.1 holds for general loss function $L_{i}: R^{K} \to R$ provided that conditions (1) and (2) in Theorem S5.1 hold. In this section, we consider the multinomial logistic loss function $L_{i}$ defined in Section S3.1. To be specific,

$$
L _ {i} (u) = - \sum_ {k = 1} ^ {K + 1} y _ {i k} e _ {k} ^ {T} Q u + \log \sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp (e _ {k ^ {\prime}} ^ {T} Q u), \quad \forall u \in \mathbb {R} ^ {K}. \tag {S6.1}
$$

In order to apply Theorem S5.1, we need to verify that, when $L_{i}$ in (S6.1) is used, the two conditions (1) and (2) in Theorem S5.1 hold. To this end, we present a few lemmas in the following two subsections, which will be useful for asserting the conditions (1) and (2) when we apply Theorem S5.1 to prove Theorem S3.1.

S6.1. Control of the singular values of the gradients and Hessians. Before stating the lemmas that assert the conditions in Theorem S5.1, define

$$
U = \big \{(Y, X) \in \mathbb {R} ^ {n \times (K + 1)} \times \mathbb {R} ^ {n \times p}: \hat {\mathsf {B}} \mathrm{exists}, \| X \hat {\mathsf {B}} (I _ {K + 1} - \frac {\mathbf {1 1} ^ {T}}{K + 1}) \| _ {F} ^ {2} <   n \tau \big \},
$$

$$
U _ {y} = \Big \{Y \in \mathbb {R} ^ {n \times (K + 1)}: \sum_ {i = 1} ^ {n} I (\mathsf {y} _ {i k} = 1) \geq \gamma n \text {for all} k \in [ K + 1 ] \Big \}.
$$

Lemma S6.1 (deterministic result on gradient). Let $L_{i}$ be defined as in (S6.1). Assume that either Assumption 2.2 or Assumption S1.1 holds. If $Y \in U_{y}$ , for any $M \in R^{n \times K}$ such that $\|MQ^{T}\|_{F}^{2} \leq n\tau$ , we have

$$
m _ {*} I _ {K} \preceq n ^ {- 1} \sum_ {i = 1} ^ {n} \nabla L _ {i} (M ^ {T} e _ {i}) \nabla L _ {i} (M ^ {T} e _ {i}) ^ {T} \preceq K I _ {K},
$$

where $m_*$ is a positive constant depending on $(K, \gamma, \tau)$ only.

Proof of Lemma S6.1. Without loss of generality, let's assume that $\gamma n$ is an integer. Otherwise, we can replace it with the greatest integer less than or equal to $\gamma n$ , denoted as $|\gamma n|$ .

If $Y \in U_y$ , there exists at least $\gamma n$ many disjoint index sets $\{S_1, \ldots, S_{\gamma n}\}$ such that the following hold for each $l \in [\gamma n]$ ,

$$
(i) S _ {l} \subset [ n ]; (i i) | S _ {l} | = K + 1; (i i i) \sum_ {i \in S _ {l}} \mathsf {y} _ {i k} = 1, \quad \forall k \in [ K + 1 ].
$$

Since $S_{l}$ are disjoint and $\cup_{l = 1}^{\gamma n}S_l\subset [n]$ , we have

$$
\sum_ {l = 1} ^ {\gamma n} \sum_ {i \in S _ {l}} \| Q M ^ {T} e _ {i} \| ^ {2} \leq \sum_ {i = 1} ^ {n} \| Q M ^ {T} e _ {i} \| ^ {2} = \| Q M ^ {T} \| _ {F} ^ {2} <   n \tau .
$$

It follows that at most $\alpha n$ many of $l \in \{1, 2, ..., \gamma n\}$ s.t. $\sum_{i \in S_l} \|QM^T e_i\|^2 > \tau/\alpha$ , otherwise the previous display can not hold. In other words, there exists a subset $L^* \subset \{1, 2, ..., \gamma n\}$ with $|L^*| \geq (\gamma - \alpha)n$ s.t. $\sum_{i \in S_l} \|QM^T e_i\|^2 \leq \tau/\alpha$ for all $l \in L^*$ . Define the index set $I = \cup_{l \in L^*} S_l$ , then $|I| \geq (K + 1)n(\gamma - \alpha)$ , and $\|QM^T e_i\|_\infty \leq \sqrt{\tau/\alpha}$ for all $i \in I$ . Let us take $\alpha = \gamma/2$ , then $|L^*| \geq \frac{\gamma}{2}n$ and $|I| \geq \gamma(K + 1)n/2$ . Recall that $L_i(u) = \mathsf{L}_i(Qu)$ , we have $\nabla L_i(u) = Q^T \nabla \mathsf{L}_i(Qu)$ . Thus,

$$
\nabla L _ {i} (M ^ {T} e _ {i}) = Q ^ {T} \nabla \mathsf {L} _ {i} (Q M ^ {T} e _ {i}) = Q ^ {T} (- \mathsf {y} _ {i} + \mathsf {p} _ {i}),
$$

where $p_{i} \in R^{K+1}$ and its k-th entry satisfying

$$
\mathfrak {p} _ {i k} = \frac {\exp (e _ {k} ^ {T} Q M ^ {T} e _ {i})}{\sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp (e _ {k ^ {\prime}} ^ {T} Q M ^ {T} e _ {i})} \in [ c, 1 - c ], \tag {S6.2}
$$

for some constant $c \in (0,1)$ depending on $(\tau, \alpha, K)$ only. Therefore,

$$
\begin{array}{l} n ^ {- 1} \sum_ {i = 1} ^ {n} \nabla L _ {i} (M ^ {T} e _ {i}) \nabla L _ {i} (M ^ {T} e _ {i}) ^ {T} \\ = n ^ {- 1} Q ^ {T} \sum_ {i = 1} ^ {n} \left(\mathrm{y} _ {i} - \mathrm{p} _ {i}\right) \left(\mathrm{y} _ {i} - \mathrm{p} _ {i}\right) ^ {T} Q \\ \succeq n ^ {- 1} Q ^ {T} \sum_ {l = 1} ^ {\gamma n} \sum_ {i \in S _ {l}} (\mathsf {y} _ {i} - \mathsf {p} _ {i}) (\mathsf {y} _ {i} - \mathsf {p} _ {i}) ^ {T} Q \\ \succeq n ^ {- 1} Q ^ {T} \sum_ {l \in L ^ {*}} \sum_ {i \in S _ {l}} (\mathsf {y} _ {i} - \mathsf {p} _ {i}) (\mathsf {y} _ {i} - \mathsf {p} _ {i}) ^ {T} Q \\ := n ^ {- 1} Q ^ {T} \sum_ {l \in L ^ {*}} A _ {l} ^ {T} A _ {l} Q, \\ \end{array}
$$

where $A_{l} \in \mathbb{R}^{(K+1) \times (K+1)}$ has $K + 1$ rows $\{y_{i} - p_{i} : i \in S_{l}\}$ . We further note that $A_{l}$ is of the form $(I_{K+1} - P_{l})$ up to a rearrangement of the columns, where $P_{l} \in \mathbb{R}^{(K+1) \times (K+1)}$ is a stochastic matrix with entries of the form

$$
\frac {\exp (e _ {k} ^ {T} Q M ^ {T} e _ {i})}{\sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp (e _ {k ^ {\prime}} ^ {T} Q M ^ {T} e _ {i})}, \qquad i \in S _ {l}, \quad k \in [ K + 1 ].
$$

By (S6.2), for each $l \in L^*$ , the stochastic matrix $\mathsf{P}_l$ is irreducible and aperiodic and $\ker(I_{K+1} - \mathsf{P}_l)$ is the span of the all-ones vector 1. Therefore,

$$
(I _ {K + 1} - \mathsf {P} _ {l}) Q Q ^ {T} = (I _ {K + 1} - \mathsf {P} _ {l}) (I _ {K + 1} - \frac {\mathbf {1 1} ^ {T}}{K + 1}) = (I _ {K + 1} - \mathsf {P} _ {l}).
$$

It follows that

$$
K = \mathrm{rank} (I _ {K + 1} - \mathsf {P} _ {l}) = \mathrm{rank} ((I _ {K + 1} - \mathsf {P} _ {l}) Q Q ^ {T}) \leq \mathrm{rank} ((I _ {K + 1} - \mathsf {P} _ {l}) Q) \leq K.
$$

We conclude that the rank of $(I_{K + 1} - \mathsf{P}_l)Q$ is $K$ .

If $\mathcal{P}$ denotes the set of matrices $\{\mathsf{P} \in \mathbb{R}^{(K+1) \times (K+1)} : \text{stochastic with entries in } [c, 1 - c]\}$ , and $S^{K-1} = \{a \in \mathbb{R}^K : \|a\| = 1\}$ . By compactness of $\mathcal{P}$ and $S^{K-1}$ , we obtain

$$
\begin{array}{l} \frac {1}{n} \lambda_ {\min} (\sum_ {i = 1} ^ {n} \nabla L _ {i} (M ^ {T} e _ {i}) \nabla L _ {i} (M ^ {T} e _ {i}) ^ {T}) \\ \geq \frac {1}{n} \sum_ {l \in L ^ {*}} \lambda_ {\min} (Q ^ {T} A _ {l} ^ {T} A _ {l} Q) \\ \geq \frac {1}{n} \sum_ {l \in L ^ {*}} \min _ {a \in S ^ {K - 1}} a ^ {T} Q ^ {T} A _ {l} ^ {T} A _ {l} Q a \\ \geq \frac {1}{n} | L ^ {*} | \min _ {a \in S ^ {K - 1}, \mathsf {P} \in \mathcal {P}} a ^ {T} Q ^ {T} (I _ {K + 1} - \mathsf {P}) ^ {T} (I _ {K + 1} - \mathsf {P}) Q a \\ \geq \frac {\gamma}{2} a _ {*} ^ {T} Q ^ {T} (I _ {K + 1} - \mathsf {P} _ {*}) ^ {T} (I _ {K + 1} - \mathsf {P} _ {*}) Q a _ {*} \\ \geq \frac {\gamma}{2} a _ {*} ^ {T} Q ^ {T} (I _ {K + 1} - \mathsf {P} _ {*}) ^ {T} (I _ {K + 1} - \mathsf {P} _ {*}) Q a _ {*} \\ := m _ {*}, \\ \end{array}
$$

where $a_{*} \in S^{K-1}$ , $P_{*} \in P$ , and $m_{*}$ is a positive constant depending on $(K, \gamma, \tau)$ only. The first inequality above uses the property $\lambda_{\min}(A + B) \geq \lambda_{\min}(A) + \lambda_{\min}(B)$ , where A and B are two positive semi-definite matrices.

In other words, $\frac{1}{n}\sum_{i=1}^{n}\nabla L_{i}(M^{T}e_{i})\nabla L_{i}(M^{T}e_{i})^{T}\succeq m_{*}I_{K}$ . For the upper bound, since $\|Q\|_{op}\leq1$ by definition of Q and all the entries of $(\mathbf{y}_{i}-\mathfrak{p}_{i})(\mathbf{y}_{i}-\mathfrak{p}_{i})^{T}$ are between -1 and 1 if Assumption 2.2 or Assumption S1.1 holds, we have

$$
\| \sum_ {i = 1} ^ {n} \nabla L _ {i} (M ^ {T} e _ {i}) \nabla L _ {i} (M ^ {T} e _ {i}) ^ {T} \| _ {o p} = \| Q ^ {T} \sum_ {i = 1} ^ {n} (\mathsf {y} _ {i} - \mathsf {p} _ {i}) (\mathsf {y} _ {i} - \mathsf {p} _ {i}) ^ {T} Q \| _ {o p} \leq n K.
$$

![](images/69c6307dbb19b4fb1af5b267e6abb8a0b5437bfaae86024bedc78293b0f6600c.jpg)

Lemma S6.2 (deterministic result on Hessian). Let $L_{i}$ be defined as in (S6.1). For all $i \in [n]$ , we have $\nabla^{2}L_{i}(u) \preceq I_{K}$ for any $u \in R^{K}$ and

$$
\min _ {u \in \mathbb {R} ^ {K}, \| Q u \| _ {\infty} \leq r} \nabla^ {2} L _ {i} (u) \succeq c _ {*} I _ {K},
$$

where $c_*$ is a positive constant depending on $(K,r)$ only.

Proof of Lemma S6.2. Recall that $L_{i}(u) = \mathsf{L}_{i}(Qu)$ , we have

$$
\nabla^ {2} L _ {i} (u) = Q ^ {T} \nabla^ {2} \mathsf {L} _ {i} (Q u) Q,
$$

where $\nabla^2\mathsf{L}_i(Qu) = \mathrm{diag}(\mathsf{p}_i) - \mathsf{p}_i\mathsf{p}_i^T$ and the $k$ -th entry of $\mathsf{p}_i \in \mathbb{R}^{K+1}$ is defined as $\mathsf{p}_{ik} = \frac{\exp(e_k^T Qu)}{\sum_{k'=1}^{K+1} \exp(e_{k'}^T Qu)}$ for all $i \in [n], k \in [K+1]$ . Thus, $\mathsf{p}_{ik} \leq 1$ for all $i \in [n], k \in [K+1]$ . For any vector $a \in S^{K-1}$ , we have

$$
\begin{array}{l} a ^ {T} \nabla^ {2} L _ {i} (u) a = a ^ {T} Q ^ {T} [ \mathrm{diag} (\mathsf {p} _ {i}) - \mathsf {p} _ {i} \mathsf {p} _ {i} ^ {T} ] Q a \\ \leq a ^ {T} Q ^ {T} \operatorname{diag} (\mathfrak {p} _ {i}) Q a \\ \leq \| Q ^ {T} \operatorname{diag} (\mathsf {p} _ {i}) Q \| _ {o p} \\ \leq 1, \\ \end{array}
$$

where the last inequality uses $\|Q\|_{op} \leq 1$ and $p_{ik} \leq 1$ for any k. Hence, $\nabla^{2}L_{i}(u) \preceq I_{K}$ for any $i \in [n]$ .

Now we prove the lower bound. For any $u \in R^{K}$ such that $\|Qu\|_{\infty} \leq r$ , we have

$$
\mathsf {p} _ {i k} = \frac {\exp (e _ {k} ^ {T} Q u)}{\sum_ {k ^ {\prime} = 1} ^ {K + 1} \exp (e _ {k ^ {\prime}} ^ {T} Q u)} \in [ c, 1 - c ]
$$

for some constant $c$ depending on $(K,r)$ only. For any vector $a\in S^{K - 1}$ , let $\eta = Qa\in \mathbb{R}^{K + 1}$ , then $\mathbf{1}^T\eta = 0$ and

$$
\begin{array}{l} a ^ {T} \big [ \nabla^ {2} L _ {i} (u) \big ] a = a ^ {T} Q ^ {T} [ \mathrm{diag} (\mathsf {p} _ {i}) - \mathsf {p} _ {i} \mathsf {p} _ {i} ^ {T} ] Q a \\ = \eta^ {T} [ \mathrm{diag} (\mathsf {p} _ {i}) - \mathsf {p} _ {i} \mathsf {p} _ {i} ^ {T} ] \eta \\ = \sum_ {k = 1} ^ {K + 1} \mathfrak {p} _ {i k} \eta_ {k} ^ {2} - (\sum_ {k = 1} ^ {K + 1} \mathfrak {p} _ {i k} \eta_ {k}) ^ {2} \\ > \sum_ {k} \mathsf {p} _ {i k} \eta_ {k} ^ {2} - \sum_ {k} \mathsf {p} _ {i k} \eta_ {k} ^ {2} \sum_ {k} \mathsf {p} _ {i k} \\ = \sum_ {k} \mathsf {p} _ {i k} \eta_ {k} ^ {2} (1 - \sum_ {k} \mathsf {p} _ {i k}) \\ = 0, \\ \end{array}
$$

where the last equality uses $\sum_{k=1}^{K+1}p_{ik}=1$ , and the inequality follows by $(\sum_{k}p_{ik}\eta_{k})^{2}=(\sum_{k}\sqrt{p_{ik}}\sqrt{p_{ik}}\eta_{k})^{2}\leq\sum_{k}p_{ik}\sum_{k}p_{ik}\eta_{k}^{2}$ using the Cauchy-Schwarz inequality, and here “=” holds if and only if $\sqrt{p_{ik}}\propto\sqrt{p_{ik}}\eta_{k}$ for each k, which is not true since $p_{ik}\in[c,1-c]$ and $1^{T}\eta=0$ .

Let $\mathcal{H} = \{Q^T (\mathrm{diag}(\mathfrak{p}) - \mathfrak{pp}^T)Q:\mathfrak{p}\in [c,1 - c]^{K + 1}\}$ , then $\mathcal{H}$ is compact and

$$
\min _ {u \in \mathbb {R} ^ {K}, \| Q u \| _ {\infty} \leq r} \lambda_ {\min} (\nabla^ {2} L _ {i} (u)) \geq \min _ {a \in S ^ {K - 1}, H \in \mathcal {H}} a ^ {T} H a = a _ {*} ^ {T} H _ {*} a _ {*} > 0
$$

for some $a_* \in S^{K-1}$ and $H_* \in \mathcal{H}$ . Therefore,

$$
\min _ {u \in \mathbb {R} ^ {K}, \| Q u \| _ {\infty} \leq r} \nabla^ {2} L _ {i} (u)) \succeq c _ {*} I _ {K}
$$

where $c_*$ is a positive constant depending on $(K, r)$ only.

![](images/8f91db15a6a7cc32e5153783b6d3cfef29f3625846af86f9ba47273e15e2d124.jpg)

S6.2. Lipschitz conditions. We first restate the definitions of following sets,

$$
U = \left\{(Y, X) \in \mathbb {R} ^ {n \times (K + 1)} \times \mathbb {R} ^ {n \times p}: \hat {\mathsf {B}} \text {exists}, \| X \hat {\mathsf {B}} (I _ {K + 1} - \frac {\mathbf {1 1} ^ {T}}{K + 1}) \| _ {F} ^ {2} <   n \tau \right\},
$$

$$
U _ {y} = \left\{Y \in \mathbb {R} ^ {n \times (K + 1)}: \sum_ {i = 1} ^ {n} I (\mathrm{y} _ {i k} = 1) \geq \gamma n \text {for all} k \in [ K + 1 ] \right\}
$$

Lemma S6.3. Assume $p / n \leq \delta^{-1} < 1 - \alpha$ for some $\alpha \in (0,1)$ and $\delta > 1$ . Let $\mathcal{I} = \{I \subset [n] : |I| = \lceil n(1 - \alpha) \rceil\}$ and $P_I = \sum_{i \in I} e_i e_i^T$ . Define

$$
U _ {x} = \left\{X \in \mathbb {R} ^ {n \times p}: \min _ {I \in \mathcal {I}} \lambda_ {\min} (\frac {\Sigma^ {- 1 / 2} X ^ {T} P _ {I} X \Sigma^ {- 1 / 2}}{n}) \geq \phi_ {*} ^ {2}, \frac {\| X \Sigma^ {- 1 / 2} \| _ {o p}}{\sqrt {n}} \leq \phi^ {*} \right\} \tag {S6.3}
$$

for some positive constants $\phi_{*},\phi^{*}$ , which depend on $(\delta ,\alpha)$ only. Let $U^{*} = \{(Y,X)\in U: Y\in U_{y},X\in U_{x}\}$ . Then under Assumptions 2.1, 2.3 and 2.4, and if either Assumption 2.2 or Assumption S1.1 holds, we have

(i) $\mathbb{P}((Y,X)\in U^{*})\to 1$ as $n,p\to \infty$   
(ii) Let $G$ be defined in Assumption S5.2. If $\{(Y,X),(Y,\tilde{X})\} \subset U^{*}$ , we have

$$
\| G (Y, X) - G (Y, \tilde {X}) \| _ {F} \leq L \| (X - \tilde {X}) \Sigma^ {- 1 / 2} \| _ {F},
$$

where $L$ is a positive constant depending on $(K,\gamma ,\tau ,\alpha)$ only.

Proof of Lemma S6.3. We first prove statement (i). Under Assumption 2.1, [Bellec, 2022, Lemma 7.7] implies

$$
\mathbb {P} \bigl (\min _ {I \in \mathcal {I}} \lambda_ {\min} \bigl (\frac {\Sigma^ {- 1 / 2} X ^ {T} P _ {I} X \Sigma^ {- 1 / 2}}{n} \bigr) \geq \phi_ {*} ^ {2} \bigr) \to 1
$$

for some positive constant $\phi_{*}$ depending on $(\delta,\alpha)$ only. Furthermore, [Davidson and Szarek, 2001, Theorem II.13] implies

$$
\mathbb {P} \big (\frac {\| X \Sigma^ {- 1 / 2} \| _ {o p}}{\sqrt {n}} \leq \phi^ {*} \big) \to 1
$$

for some positive constant $\phi^{*}$ depending on $\delta$ only. Therefore, $\mathbb{P}(X\in U_x)\to 1$ . Under Assumption 2.3, we have $\mathbb{P}(Y\in U_y)\to 1$ . Under Assumption 2.4, we have $\mathbb{P}((Y,X)\in U)\to 1$ . In conclusion, under Assumptions 2.1, 2.3 and 2.4, we have $\mathbb{P}((Y,X)\in U^{*})\to 1$ as $n,p\to \infty$ .

Now we prove the statement (ii). For a fixed Y, let $(Y,X)$ , $(Y,\tilde{X})\in U^{*}$ , $\hat{B},\tilde{B}$ be their corresponding minimizers of (S3.3), and $G,\tilde{G}$ be their corresponding gradient matrices. We first provide some useful results derived from the KKT conditions. From the KKT conditions $X^{T}G=\tilde{X}^{T}\tilde{G}=0$ , we have

$$
\begin{array}{l} \langle X \hat {B} - \tilde {X} \tilde {B}, G - \tilde {G} \rangle = \langle \hat {B} - \tilde {B}, \tilde {X} ^ {T} \tilde {G} - X ^ {T} G \rangle + \langle X \hat {B} - \tilde {X} \tilde {B}, G - \tilde {G} \rangle \\ = - \langle (X - \tilde {X}) (\hat {B} - \tilde {B}), G \rangle + \langle (X - \tilde {X}) \hat {B}, G - \tilde {G} \rangle . \\ \end{array}
$$

Since $\|\nabla^{2}L_{i}(u)\|_{op}\leq1$ for any $u\in R^{K}$ from Lemma S6.2, $\nabla L_{i}(\cdot)$ is 1-Lipschitz. Thus,

$$
\langle X \hat {B} - \tilde {X} \tilde {B}, G - \tilde {G} \rangle = \sum_ {i = 1} ^ {n} \langle \hat {B} ^ {T} x _ {i} - \tilde {B} ^ {T} \tilde {x} _ {i}, \nabla L _ {i} (\hat {B} ^ {T} x _ {i}) - \nabla L _ {i} (\tilde {B} ^ {T} \tilde {x} _ {i}) \rangle
$$

$$
\geq \sum_ {i = 1} ^ {n} \langle \nabla L _ {i} (\hat {B} ^ {T} x _ {i}) - \nabla L _ {i} (\tilde {B} ^ {T} \tilde {x} _ {i}), \nabla L _ {i} (\hat {B} ^ {T} x _ {i}) - \nabla L _ {i} (\tilde {B} ^ {T} \tilde {x} _ {i}) \rangle
$$

$$
= \| G - \tilde {G} \| _ {F} ^ {2}.
$$

If $(Y,X),(Y,\tilde{X})\in U$ , we have $\| X\hat{B} Q^T\| _F^2 +\| \tilde{X}\tilde{B} Q^T\| _F^2\leq 2n\tau$ . That is,

$$
\sum_ {i = 1} ^ {n} \big (\| Q \hat {B} ^ {T} x _ {i} \| ^ {2} + \| Q \tilde {B} ^ {T} \tilde {x} _ {i} \| ^ {2} \big) \leq 2 n \tau .
$$

Define the index set

$$
I = \{i \in [ n ]: \| Q \hat {B} ^ {T} x _ {i} \| ^ {2} + \| Q \tilde {B} ^ {T} \tilde {x} _ {i} \| ^ {2} \leq \frac {2 \tau}{\alpha} \},
$$

then we have $|I| \geq (1 - \alpha)n$ by Markov's inequality. Thus, for all $i \in I$ , we have $\| Q\hat{B}^T x_i\|_{\infty} \vee \| Q\tilde{B}^T\tilde{x}_i\|_{\infty} \leq \sqrt{\frac{2\tau}{\alpha}}$ .

Applying Lemma S6.2 with $r = \sqrt{\frac{2\tau}{\alpha}}$ gives

$$
\min _ {\| Q u \| _ {\infty} \leq \sqrt {\frac {2 \tau}{\alpha}}} \nabla^ {2} L _ {i} (u) \succeq c _ {*} I _ {K},
$$

where $c_{*}$ is a constant depending on $(K,\tau,\alpha)$ . Therefore,

$$
\begin{array}{l} c _ {*} \| P _ {I} (X \hat {B} - \tilde {X} \tilde {B}) \| _ {F} ^ {2} = c _ {*} \sum_ {i \in I} \| \hat {B} ^ {T} x _ {i} - \tilde {B} ^ {T} \tilde {x} _ {i} \| ^ {2} \\ \leq \sum_ {i \in I} \langle \hat {B} ^ {T} x _ {i} - \tilde {B} ^ {T} \tilde {x} _ {i}, \nabla L _ {i} (\hat {B} ^ {T} x _ {i}) - \nabla L _ {i} (\tilde {B} ^ {T} \tilde {x} _ {i}) \rangle \\ \leq \sum_ {i = 1} ^ {n} \langle \hat {B} ^ {T} x _ {i} - \tilde {B} ^ {T} \tilde {x} _ {i}, \nabla L _ {i} (\hat {B} ^ {T} x _ {i}) - \nabla L _ {i} (\tilde {B} ^ {T} \tilde {x} _ {i}) \rangle \\ = \langle X \hat {B} - \tilde {X} \tilde {B}, G - \tilde {G} \rangle \\ = - \langle (X - \tilde {X}) (\hat {B} - \tilde {B}), G \rangle + \langle (X - \tilde {X}) \hat {B}, G - \tilde {G} \rangle . \\ \end{array}
$$

We next bound the first line from below by expanding the squares,

$$
\begin{array}{l} \| P _ {I} (X \hat {B} - \tilde {X} \tilde {B}) \| _ {F} ^ {2} = \| P _ {I} \tilde {X} (\hat {B} - \tilde {B}) + P _ {I} (X - \tilde {X}) \hat {B} \| _ {F} ^ {2} \\ \geq \| P _ {I} \tilde {X} (\hat {B} - \tilde {B}) \| _ {F} ^ {2} + 2 \langle P _ {I} \tilde {X} (\hat {B} - \tilde {B}), P _ {I} (X - \tilde {X}) \hat {B} \rangle \\ \geq n \phi_ {*} ^ {2} \| \Sigma^ {1 / 2} (\hat {B} - \tilde {B}) \| _ {F} ^ {2} + 2 \langle \tilde {X} (\hat {B} - \tilde {B}), P _ {I} (X - \tilde {X}) \hat {B} \rangle , \\ \end{array}
$$

where in the last inequality we use the constant $\phi^{*}$ in (S6.3). Therefore, we obtain

$$
\begin{array}{l} c _ {*} \phi_ {*} ^ {2} n \| \Sigma^ {1 / 2} (\hat {B} - \tilde {B}) \| _ {F} ^ {2} \\ \leq - \langle (X - \tilde {X}) (\hat {B} - \tilde {B}), G \rangle + \langle (X - \tilde {X}) \hat {B}, G - \tilde {G} \rangle - 2 c _ {*} \langle \tilde {X} (\hat {B} - \tilde {B}), P _ {I} (X - \tilde {X}) \hat {B} \rangle . \\ \end{array}
$$

Together with the inequality that $\|G-\tilde{G}\|_{F}^{2}\leq\langle X\hat{B}-\tilde{X}\tilde{B},G-\tilde{G}\rangle$ , we obtain

$$
\begin{array}{l} c _ {*} \phi_ {*} ^ {2} n \| \Sigma^ {1 / 2} (\hat {B} - \tilde {B}) \| _ {F} ^ {2} + \| G - \tilde {G} \| _ {F} ^ {2} \\ \leq - 2 \langle (X - \tilde {X}) (\hat {B} - \tilde {B}), G \rangle + 2 \langle (X - \tilde {X}) \hat {B}, G - \tilde {G} \rangle - 2 c _ {*} \langle \tilde {X} (\hat {B} - \tilde {B}), P _ {I} (X - \tilde {X}) \hat {B} \rangle \\ \leq (4 + 2 c _ {*} \phi^ {*}) \| (X - \tilde {X}) \Sigma^ {- 1 / 2} \| _ {o p} \big (\| \Sigma^ {1 / 2} (\hat {B} - \tilde {B}) \| _ {F} \vee \frac {\| G - \tilde {G} \| _ {F}}{\sqrt {n}} \big) \big (\| \Sigma^ {1 / 2} \hat {B} \| _ {F} \vee \frac {\| G \| _ {o p}}{\sqrt {n}} \big) \sqrt {n}, \\ \end{array}
$$

where we bound $\langle \tilde{X} (\hat{B} -\tilde{B}),P_I(X - \tilde{X})\hat{B}\rangle$ by definition of $\phi^{*}$

$$
\begin{array}{l} \langle \tilde {X} (\hat {B} - \tilde {B}), P _ {I} (X - \tilde {X}) \hat {B} \rangle \\ = \langle \Sigma^ {1 / 2} (\hat {B} - \tilde {B}), \Sigma^ {- 1 / 2} \tilde {X} ^ {T} P _ {I} (X - \tilde {X}) \hat {B} \rangle \\ \leq \| \Sigma^ {1 / 2} (\hat {B} - \tilde {B}) \| _ {F} \| P _ {I} \tilde {X} \Sigma^ {- 1 / 2} \| _ {o p} \| (X - \tilde {X}) \Sigma^ {- 1 / 2} \| _ {o p} \| \Sigma^ {1 / 2} \hat {B} \| _ {F} \\ \leq \sqrt {n} \phi^ {*} \| \Sigma^ {1 / 2} (\hat {B} - \tilde {B}) \| _ {F} \| (X - \tilde {X}) \Sigma^ {- 1 / 2} \| _ {o p} \| \Sigma^ {1 / 2} \hat {B} \| _ {F}. \\ \end{array}
$$

Now we derive a bound of the form $\| \Sigma^{1 / 2}\hat{B}\| _F\lesssim \| G\| _F / \sqrt{n}$ . To this end, since $\phi_{*}\| \Sigma^{1 / 2}\hat{B}\| _F\leq$

$$
\| P _ {I} X \hat {B} \| _ {F} / \sqrt {n} \leq \| X \hat {B} \| _ {F} / \sqrt {n} = \| X \hat {B} Q ^ {T} \| _ {F} / \sqrt {n} \leq \sqrt {\tau}.
$$

Applying Lemma S6.1 to $M = X\hat{B}$ , we have $\frac{1}{n}\sum_{i=1}^{n}g_{i}g_{i}^{T}\succeq m_{*}I_{K}$ . Therefore,

$$
\frac {1}{n} \| G \| _ {F} ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \| g _ {i} \| ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mathrm{Tr} (g _ {i} g _ {i} ^ {T}) \geq K m _ {*}.
$$

This implies that

$$
\phi_ {*} ^ {2} \| \Sigma^ {1 / 2} \hat {B} \| _ {F} ^ {2} \leq \tau \leq \frac {\tau}{K m _ {*} (I)} \| G \| _ {F} ^ {2} / n.
$$

In conclusion, if $\{(Y,X),(Y,\tilde{X})\} \subset U^{*}$ then

$$
\sqrt {n} \| \Sigma^ {1 / 2} (\hat {B} - \tilde {B}) \| _ {F} + \| G - \tilde {G} \| _ {F} \leq C n ^ {- 1 / 2} \| (X - \tilde {X}) \Sigma^ {- 1 / 2} \| _ {o p} \| G \| _ {F} \leq C K \| (X - \tilde {X}) \Sigma^ {- 1 / 2} \| _ {o p},
$$

where C is a constant depending on $(K,\gamma,\tau,\alpha)$ only. Note that $\|G\|_{F} \leq \sqrt{nK}$ since all entries of G are in $[-1,1]$ . ☐

Lemma S6.4. If $p / n \leq \delta^{-1} < (1 - \alpha)$ for some $\alpha \in (0,1)$ and $\delta > 1$ . If $(Y,X) \in U$ and $X \in U_x$ , where $U_x$ is defined in Lemma S6.3, we have

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} H _ {i} \otimes (x _ {i} x _ {i} ^ {T}) \succeq c _ {1} (I _ {K} \otimes \Sigma),
$$

where $c_{1}$ is a positive constant depending on $(K,\tau ,\alpha ,\phi_{*})$ only.

Proof of Lemma S6.4. If $(Y,X)\in U$ , we have $\| X\hat{B} Q^T\| _F^2\leq n\tau$ . Define the index set

$$
I = \{i \in [ n ]: \| Q \hat {B} ^ {T} x _ {i} \| \leq \frac {\tau}{\alpha} \},
$$

then we have $|I| \geq (1 - \alpha)n$ by Markov's inequality. Therefore, for any $i \in I$ , $\| Q\hat{B}^T x_i\|_{\infty} \leq \frac{\tau}{\alpha}$ . Applying Lemma S6.2 with $u = \hat{B}^T x_i$ and $r = \frac{\tau}{\alpha}$ , we have for any $i \in I$ , $H_i = \nabla^2 L_i(\hat{B}^Tx_i) \succeq c_*I_K$ for some positive constant $c_*$ depending on $(K,\tau ,\alpha)$ only. Therefore, if $(Y,X)\in U$ and $X\in U_x$ , we have

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} H _ {i} \otimes (x _ {i} x _ {i} ^ {T}) \succeq \frac {1}{n} \sum_ {i \in I} H _ {i} \otimes (x _ {i} x _ {i} ^ {T})
$$

$$
\succeq c _ {*} \frac {1}{n} \sum_ {i \in I} I _ {K} \otimes (x _ {i} x _ {i} ^ {T})
$$

$$
= c _ {*} (I _ {K} \otimes \frac {X ^ {T} P _ {I} X}{n})
$$

$$
\succeq c _ {*} \phi_ {*} (I _ {K} \otimes \Sigma)
$$

$$
= c _ {1} (I _ {K} \otimes \Sigma),
$$

where $P_{I} = \sum_{i\in I}e_{i}e_{i}^{T}$ and $c_{1}$ is a positive constant depending on $(K,\tau ,\alpha ,\phi_{*})$ only.

S6.3. Proof of Theorem S3.1. The proof of Theorem S3.1 is a direct consequence of Theorem S5.1 by noting

$$
\sqrt {n} \Omega_ {j j} ^ {- 1 / 2} \Bigl (\frac {1}{n} \sum_ {i = 1} ^ {n} g _ {i} g _ {i} ^ {T} \Bigr) ^ {- 1 / 2} \Bigl (\frac {1}{n} \sum_ {i = 1} ^ {n} V _ {i} \Bigr) \hat {B} ^ {T} e _ {j} = \Omega_ {j j} ^ {- 1 / 2} (G ^ {T} G) ^ {- 1 / 2} V \hat {B} ^ {T} e _ {j},
$$

which is a consequence of the identities $G = \sum_{i=1}^{n} e_i g_i^T$ and $V = \sum_{i=1}^{n} V_i$ .

It thus remains to verify the conditions (1) and (2) in Theorem S5.1 from the assumptions in Theorem S3.1.

Applying Lemma S6.3 with $\alpha$ chosen as $1 - \delta^{-1/2}$ , we have for $\{(Y,X),(Y,\tilde{X})\} \subset U^{*}$ ,

$$
\| G (Y, X) - G (Y, \tilde {X}) \| _ {F} \leq L \| (X - \tilde {X}) \Sigma^ {- 1 / 2} \| _ {F},
$$

where $L$ is a positive constant depending on $(K,\gamma ,\tau ,\delta)$ only.

Apply Lemma S6.4 with the same $\alpha = 1 - \delta^{-1/2}$ , we have for $(Y, X) \in U^{*}$ ,

$$
\frac {1}{n} \sum_ {i = 1} ^ {n} H _ {i} \otimes (x _ {i} x _ {i} ^ {T}) \succeq c _ {*} (I _ {K} \otimes \Sigma),
$$

where $c_*$ is a positive constant depending on $(K, \tau, \delta)$ only.

Applying Lemma S6.1 with $M = X\hat{B}$ , we have for $(Y,X)\in U^{*}$ ,

$$
m _ {*} I _ {K} \preceq n ^ {- 1} \sum_ {i = 1} ^ {n} \nabla L _ {i} (M ^ {T} e _ {i}) \nabla L _ {i} (M ^ {T} e _ {i}) ^ {T} \preceq K I _ {K},
$$

where $m_{*}$ is a positive constant depending on $(K,\gamma,\tau)$ only. Therefore, the conditions (1) and (2) in Theorem S5.1 hold when the multinomial logistic loss is used. This completes the proof of Theorem S3.1.

# S7. OTHER PROOF

S7.1. Proof of Equation (1.12) (Classical asymptotic theory with fixed $p$ ). Here we provide a derivation of the asymptotic distribution of MLE under classical setting, where $p$ is fixed and $n$ tends to infinity.

We first calculate the Fisher information matrix of the multinomial logistic log-odds model (1.8) with covariate $x \sim N(\mathbf{0}, \Sigma)$ and response $y \in \{0, 1\}^{K+1}$ one-hot encoded satisfying $\sum_{k=1}^{K+1} y_k = 1$ . Note that the model (1.8) can be rewritten as

$$
\mathbb {P} (\mathbf {y} _ {k} = 1 | x) = \frac {\exp (x ^ {T} A ^ {*} e _ {k})}{1 + \sum_ {k ^ {\prime} = 1} ^ {K} \exp (x ^ {T} A ^ {*} e _ {k ^ {\prime}})}, \quad \forall k \in \{1, \dots , K \}
$$

$$
\mathbb {P} (\mathsf {y} _ {K + 1} = 1 | x) = \frac {1}{1 + \sum_ {k ^ {\prime} = 1} ^ {K} \exp (x ^ {T} A ^ {*} e _ {k ^ {\prime}})}.
$$

The likelihood function of a parameter $A \in R^{p \times K}$ is

$$
L (A) = \prod_ {k = 1} ^ {K} \Bigl [ \frac {\exp (x ^ {T} A e _ {k})}{1 + \sum_ {k ^ {\prime} = 1} ^ {K} \exp (x ^ {T} A e _ {k ^ {\prime}})} \Bigr ] ^ {y _ {k}} \Bigl [ \frac {1}{1 + \sum_ {k ^ {\prime} = 1} ^ {K} \exp (x ^ {T} A e _ {k ^ {\prime}})} \Bigr ] ^ {y _ {K + 1}}
$$

$$
= \prod_ {k = 1} ^ {K} [ \exp (x ^ {T} A e _ {k}) ] ^ {\mathsf {y} _ {k}} \frac {1}{1 + \sum_ {k ^ {\prime} = 1} ^ {K} \exp (x ^ {T} A e _ {k ^ {\prime}})},
$$

where we used $\sum_{k=1}^{K+1} y_k = 1$ . Thus, the log-likelihood function is

$$
\ell (A) = \sum_ {k = 1} ^ {K} \mathsf {y} _ {k} x ^ {T} A e _ {k} - \log \bigl [ 1 + \sum_ {k ^ {\prime} = 1} ^ {K} \exp (x ^ {T} A e _ {k ^ {\prime}}) \bigr ].
$$

It is more convenient to calculate the Fisher information matrix on the vector space $R^{pK}$ instead of the matrix space $R^{p\times K}$ . To this end, let $\theta = \operatorname{vec}(A^{T})$ , then $x^{T}Ae_{k} = e_{k}^{T}A^{T}x = (x^{T} \otimes e_{k}^{T})\operatorname{vec}(A^{T}) = (x^{T} \otimes e_{k}^{T})\theta$ , and the log-likelihood function parameterized by $\theta$ is

$$
\ell (\theta) = \sum_ {k = 1} ^ {K} y _ {k} (x ^ {T} \otimes e _ {k} ^ {T}) \theta - \log [ 1 + \sum_ {k ^ {\prime} = 1} ^ {K} \exp ((x ^ {T} \otimes e _ {k ^ {\prime}} ^ {T}) \theta) ].
$$

By multivariate calculus, we obtain the Fisher information matrix evaluated at $\theta^{*} = \operatorname{vec}(A^{*T})$ ,

$$
\begin{array}{l} \mathcal {I} \left(\theta^ {*}\right) = - \mathbb {E} \left[ \frac {\partial}{\partial \theta} \frac {\partial \ell (\theta)}{\partial \theta^ {T}} \right] \Big | _ {\theta = \theta^ {*}} \\ = \mathbb {E} [ (x x ^ {T}) \otimes (\mathrm{diag} (\pi^ {*}) - \pi^ {*} \pi^ {* T}) ], \\ \end{array}
$$

where $\pi^{*}\in \mathbb{R}^{K}$ with $k$ -th entry $\pi_k^* = \frac{\exp(x^T A^*e_k)}{1 + \sum_{k' = 1}^{K}\exp(x^T A^*e_{k'})}$ for each $k\in [K]$ .

From classical maximum likelihood theory, for instance [Van der Vaart, 1998, Chapter 5], we have

$$
\sqrt {n} (\hat {\theta} - \theta^ {*}) \xrightarrow {\mathrm{d}} N (\mathbf {0}, \mathcal {I} _ {\theta^ {*}} ^ {- 1}),
$$

where $\hat{\theta} = \mathrm{vec}(\hat{A}^T)$ and $\hat{A}$ is the MLE of $A^{*}$ . Furthermore, if the $j$ -th covariate is independent of the response, we know $e_j^T A^* = \mathbf{0}^T$ , then

$$
\sqrt {n} \hat {A} ^ {T} e _ {j} = \sqrt {n} (\hat {A} ^ {T} e _ {j} - A ^ {* T} e _ {j}) = \sqrt {n} (e _ {j} ^ {T} \otimes I _ {K}) (\hat {\theta} - \theta^ {*}) \xrightarrow {\mathrm{d}} N (\mathbf {0}, S _ {j}),
$$

where

$$
\begin{array}{l} S _ {j} = (e _ {j} ^ {T} \otimes I _ {K}) \mathcal {I} _ {\theta^ {*}} ^ {- 1} (e _ {j} \otimes I _ {K}) \\ = e _ {j} ^ {T} \mathrm{cov} (x) ^ {- 1} e _ {j} [ \mathbb {E} (\mathrm{diag} (\pi^ {*}) - \pi^ {*} \pi^ {* T}) ] ^ {- 1} \\ \end{array}
$$

holds by the independence between the j-th covariate and the response under $H_{0}$ in (1.10). This completes the proof of Equation (1.12).

(Kai Tan) DEPARTMENT OF STATISTICS, RUTGERS UNIVERSITY, PISCATAWAY, NJ 08854, USA.

Email address: kai.tan@rutgers.edu

(Pierre C. Bellec) DEPARTMENT OF STATISTICS, RUTGERS UNIVERSITY, PISCATAWAY, NJ 08854, USA.

Email address: pierre.bellec@rutgers.edu