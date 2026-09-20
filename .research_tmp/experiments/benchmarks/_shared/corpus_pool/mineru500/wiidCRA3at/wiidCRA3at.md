# Stein Π-Importance Sampling

Congye Wang $^{1}$ , Wilson Ye Chen $^{2}$ , Heishiro Kanagawa $^{1}$ , Chris. J. Oates $^{1}$

$^{1}$ Newcastle University, UK

$^{2}$ University of Sydney, Australia

# Abstract

Stein discrepancies have emerged as a powerful tool for retrospective improvement of Markov chain Monte Carlo output. However, the question of how to design Markov chains that are well-suited to such post-processing has yet to be addressed. This paper studies Stein importance sampling, in which weights are assigned to the states visited by a $\Pi$ -invariant Markov chain to obtain a consistent approximation of P, the intended target. Surprisingly, the optimal choice of $\Pi$ is not identical to the target P; we therefore propose an explicit construction for $\Pi$ based on a novel variational argument. Explicit conditions for convergence of Stein $\Pi$ -Importance Sampling are established. For $\approx 70\%$ of tasks in the PosteriorDB benchmark, a significant improvement over the analogous post-processing of P-invariant Markov chains is reported.

# 1 Introduction

Stein discrepancies are a class of statistical divergences that can be computed without access to a normalisation constant. Originally conceived as a tool to measure the performance of sampling methods (Gorham and Mackey, 2015), these discrepancies have since found wide-ranging statistical applications (see the review of Anastasiou et al., 2023). Our focus here is the use of Stein discrepancies for retrospective improvement of Markov chain Monte Carlo (MCMC), and here two main techniques have been proposed: (i) Stein importance sampling (Liu and Lee, 2017; Hodgkinson et al., 2020), and (ii) Stein thinning (Riabiz et al., 2022). In Stein importance sampling (also called black box importance sampling), the samples are assigned weights such that a Stein discrepancy between the weighted empirical measure and the target P is minimised. Stein thinning constructs a sparse approximation to this optimally weighted measure at a lower computational and storage cost. Together, these techniques provide a powerful set of post-processing tools for MCMC, with subsequent authors proposing a range of generalisations and extensions (Teymur et al., 2021; Chopin and Ducrocq, 2021; Hawkins et al., 2022; Fisher and Oates, 2023; Bénard et al., 2023).

The consistency of these algorithms has been established in the setting of approximate, $\Pi$ -invariant MCMC, motivated by challenging inference problems where only approximate sampling can be performed. In these settings, $\Pi$ is implicitly an approximation to P that is as accurate as possible subject to computational budget. However, the critical question of how to design Markov chains that are well-suited to such post-processing has yet to be addressed. This paper provides a solution, in the form of a specific construction for $\Pi$ derived from a novel variational argument. Surprisingly, we are able to demonstrate a substantial improvement using the proposed $\Pi$ , compared to the case where $\Pi$ and P are equal. The paper proceeds as follows: Section 2 presents an abstract formulation of the task and existing results for optimally-weighted empirical measures are reviewed. Section 3 derives our proposed choice of $\Pi$ and establishes that Stein post-processing of samples from a $\Pi$ -invariant Metropolis-adjusted Langevin algorithm (MALA) provides a consistent approximation of P. The approach is stress-tested using the recently released PosteriorDB suite of benchmark tasks in Section 4, before concluding with a discussion in Section 5.

# 2 Background

To properly contextualise our discussion we start with an abstract mathematical description of the task. Let $P$ be a probability measure on a measurable space $\mathcal{X}$ . Let $\mathcal{P}(\mathcal{X})$ be the set of all probability measures on $\mathcal{X}$ . Let $D_P: \mathcal{P}(\mathcal{X}) \to [0, \infty]$ be a statistical divergence for measuring the quality of an approximation $Q$ to $P$ , meaning that $D_P(Q) = 0$ if and only if $Q = P$ . In this work we consider approximations whose support is contained in a finite set $\{x_1, \ldots, x_n\} \subset \mathcal{X}$ , and in particular we consider optimal approximations of the form

$$
P _ {n} ^ {\star} = \sum_ {i = 1} ^ {n} w _ {i} ^ {\star} \delta (x _ {i}), \qquad w ^ {\star} \in \underset {w \geq 0, 1 ^ {\top} w = 1} {\arg \min} D _ {P} \left(\sum_ {i = 1} ^ {n} w _ {i} \delta (x _ {i})\right).
$$

In what follows we restrict attention to statistical divergences for which such approximations can be shown to exist and be well-defined. The question that we then ask is which states $\{x_{1},\ldots,x_{n}\}$ minimise the approximation error $D_{P}(P_{n}^{\star})$ ? Before specialising to Stein discrepancies, it is helpful to review existing results for some standard statistical divergences $D_{P}$ .

# 2.1 Wasserstein Divergence

Optimal quantisation focuses on the r-Wasserstein $(r \geq 1)$ family of statistical divergences $D_{P}(Q) = \inf_{\gamma \in \Gamma(P, Q)} \int \|x - y\|^{r} \mathrm{d}\gamma(x, y)$ , where $\Gamma(P, Q)$ denotes the set of all couplings $^{1}$ of $P, Q \in \mathcal{P}(\mathbb{R}^{d})$ , and the divergence is finite whenever P and Q have finite r-th moment. Assuming the states $\{x_{1}, \ldots, x_{n}\}$ are distinct, the corresponding optimal weights are $w_{i}^{\star} = P(A_{i})$ where $A_{i}$ is the Voronoi neighbourhood $^{2}$ of $x_{i}$ in $R^{d}$ . Optimal states achieve the minimal quantisation error for P;

$$
e _ {n, r} (P) = \inf _ {x _ {1}, \dots , x _ {n} \in \mathbb {R} ^ {d}} D _ {P} \left(\sum_ {i = 1} ^ {n} w _ {i} ^ {\star} \delta \left(x _ {i}\right)\right),
$$

the smallest value of the divergence among optimally-weighted distributions supported on at most n states. Though the dependence of optimal states on n and P can be complicated, we can broaden our perspective to consider asymptotically optimal states, whose asymptotic properties can be precisely characterised. To this end, for $A \subset R^{d}$ , let $\mathcal{U}(A)$ denote the uniform distribution on A, and define the universal constant $C_{r}([0,1]^{d}) = \inf_{n \geq 1} n^{r/d} e_{n,r}(\mathcal{U}([0,1]^{d}))$ . Suppose that P admits a density p on $R^{d}$ . Then the rth quantisation coefficient of P on $R^{d}$ , defined as

$$
C _ {r} (P) = C _ {r} ([ 0, 1 ] ^ {d}) \left(\int p (x) ^ {d / (d + r)} \mathrm{d} x\right) ^ {(d + r) / d},
$$

plays a central role in the classical theory of quantisation, being the rate constant in the asymptotic convergence of the minimal quantisation error; $\lim_{n\to \infty}n^{r / d}e_{n,r}(P) = C_r(P)$ ; see Theorem 6.2 of Graf and Luschgy (2007). This suggests a natural definition; a collection $\{x_{1},\ldots ,x_{n}\}$ is called asymptotically optimal if

$$
\lim _ {n \to \infty} n ^ {r / d} D _ {P} \left(\sum_ {i = 1} ^ {n} w _ {i} ^ {\star} \delta (x _ {i})\right) = C _ {r} (P),
$$

which amounts to $P_{n}^{\star}$ asymptotically attaining the minimal quantisation error $e_{n,r}(P)$ . The main result here is that, if $\{x_{1},\ldots,x_{n}\}$ are asymptotically optimal, then $\frac{1}{n}\sum_{i=1}^{n}\delta(x_{i})\to\Pi_{r}$ , where convergence is in distribution and $\Pi_{r}$ is the distribution whose density is $\pi_{r}(x)\propto p(x)^{d/(d+r)}$ ; see Theorem 7.5 of Graf and Luschgy (2007). This provides us with a key insight; optimal states are over-dispersed with respect to the intended distributional target. The extent of the over-dispersion here depends both on r, a parameter of the statistical divergence, and the dimension d of the space on which distributions are defined.

The r-Wasserstein divergence is, unfortunately, not well-suited for use in the motivating Bayesian context. In particular, computing the optimal weights $w_{i} = P(A_{i})$ requires knowledge of P, which is typically not available when P is implicitly defined via an intractable normalisation constant. On

the other hand, the optimal sampling distribution $\Pi$ is explicit and can be sampled (for example using MCMC); for discussion of random quantisers in this context see Graf and Luschgy (2007, Chapter 9), Cohort (2004, p126) and Sonnleitner (2022, Section 4.5). The simple form of $\Pi$ is a feature of the classical approach to quantisation that we will attempt to mimic in the sequel.

# 2.2 Kernel Discrepancies

The theory of quantisation using kernels is less well-developed. A kernel is a measurable, symmetric, positive-definite function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ . From the Moore-Aronszajn theorem, there is a unique Hilbert space $\mathcal{H}(k)$ for which $k$ is a reproducing kernel, meaning that $k(\cdot, x) \in \mathcal{H}(k)$ for all $x \in \mathcal{X}$ and $\langle f, k(\cdot, x) \rangle_{\mathcal{H}(k)} = f(x)$ for all $f \in \mathcal{H}(k)$ and all $x \in \mathcal{X}$ . Assuming that $\mathcal{H}(k) \subset L^1(P)$ , we can define the weak (or Pettis) integral

$$
\mu_ {P} (\cdot) = \int k (\cdot , x) \mathrm{d} P (x), \tag {1}
$$

called the kernel mean embedding of $P$ in $\mathcal{H}(k)$ . The kernel discrepancy is then defined as the norm of the difference between kernel mean embeddings

$$
D _ {P} (Q) = \left\| \mu_ {Q} - \mu_ {P} \right\| _ {\mathcal {H} (k)} = \sqrt {\iint k (x , y) \mathrm{d} (Q - P) (x) \mathrm{d} (Q - P) (y)} \tag {2}
$$

where, to be consistent with our earlier notation, we adopt the convention that $D_P(Q)$ is infinite whenever $\mathcal{H}(k) \not\subset L^1(Q)$ . The second equality in (2) follows immediately from the stated properties of a reproducing kernel. To satisfy the requirement of a statistical divergence, we assume that the kernel $k$ is characteristic, meaning that $\mu_P = \mu_Q$ if and only if $P = Q$ . In this setting, the properties of optimal states are necessarily dependent on the choice of kernel $k$ , and are in general not well-understood. Indeed, given distinct states $\{x_1, \ldots, x_n\}$ , the corresponding optimal weights $w^\star = (w_1^\star, \ldots, w_n^\star)^\top$ are the solution to the linearly-constrained quadratic program

$$
\underset {w \in \mathbb {R} ^ {d}} {\arg \min} w ^ {\top} K w - 2 z ^ {\top} w \quad \text {   s.t.   } \quad w \geq 0, 1 ^ {\top} w = 1 \tag {3}
$$

where $K_{i,j} = k(x_i, x_j)$ and $z_i = \mu_P(x_i)$ . This program does not admit a closed-form solution, but can be numerically solved. To the best of our knowledge, the only theoretical analysis of approximations based on (3) is due to Hayakawa et al. (2022), who established rates for the convergence of $P_n^\star$ to P in the case where states are independently sampled from P. The question of an optimal sampling distribution was not considered in that work.

Although few results are available concerning (3), relaxations of this program have been well-studied. The simplest relaxation of (3) is to remove both the positivity $w \geq 0$ and normalisation $(1^{\top}w = 1)$ constraints, in which case the optimal weights have the explicit representation $w^{*} = K^{-1}z$ . The analysis of optimal states in this context has developed under the dual strands of kernel cubature and Bayesian cubature, where it has been theoretically or empirically demonstrated that (i) if states are randomly sampled, the optimal sampling distribution will be n-dependent (Bach, 2017) and over-dispersed with respect to the distributional target (Briol et al., 2017), and (ii) space-filling designs are asymptotically optimal for typical stationary kernels on bounded domains $X \subset R^{d}$ (Briol et al., 2019). Analysis of optimal states on unbounded domains appears to be more difficult; see e.g. Karvonen et al. (2021). Relaxation of either the positivity or normalisation constraints results in approximations that behave similarly to kernel cubature (see, respectively, Ehler et al., 2019; Karvonen et al., 2018). However, relaxation of either constraint can result in the failure of $P_{n}^{\star}$ to be an element of $\mathcal{P}(\mathcal{X})$ , limiting the relevance of these results to the posterior approximation task.

Despite relatively little being known about the character of optimal states in this context, kernel discrepancy is widely used. The application of kernel discrepancies to an implicitly defined distributional target, such as a posterior distribution in a Bayesian analysis, is made possible by the use of a Stein kernel; a $P$ -dependent kernel $k = k_{P}$ for which $\mu_P(x) = 0$ for all $x \in \mathcal{X}$ (Oates et al., 2017). The associated kernel discrepancy

$$
D _ {P} (Q) = \left\| \mu_ {Q} \right\| _ {\mathcal {H} \left(k _ {P}\right)} = \sqrt {\iint k _ {P} (x , y) \mathrm{d} Q (x) \mathrm{d} Q (y)} \tag {4}
$$

is called a kernel Stein discrepancy (KSD) (Chwialkowski et al., 2016; Liu et al., 2016; Gorham and Mackey, 2017), and this will be a key tool in our methodological development. The corresponding optimally weighted approximation $P_{n}^{\star}$ is the Stein importance sampling method of Liu and Lee (2017). To retain clarity of presentation in the main text, we defer all details on the construction of Stein kernels to Appendix A.

# 2.3 Sparse Approximation

If the number n of states is large, computation of optimal weights can become impractical. This has motivated a range of sparse approximation techniques, which aim to iteratively construct an approximation of the form $P_{n,m} = \frac{1}{m} \sum_{i=1}^{m} \delta(y_i)$ , where each $y_i$ is an element from $\{x_1, \ldots, x_n\}$ . The canonical example is the greedy algorithm which, at iteration j, selects a state

$$
y _ {j} \in \underset {y \in \{x _ {1}, \dots , x _ {n} \}} {\arg \min} D _ {P} \left(\frac {1}{j} \delta (y) + \frac {1}{j} \sum_ {i = 1} ^ {j - 1} \delta \left(y _ {i}\right)\right) \tag {5}
$$

for which the statistical divergence is minimised. In the context of kernel discrepancy, the greedy algorithm (5) has computational cost $O(m^{2}n)$ , which compares favourably $^{3}$ with the cost of solving (3) when $m \ll n$ . Furthermore, under appropriate assumptions, the sparse approximation converges to the optimally weighted approximation; $D_{P}(P_{n,m}) \to D_{P}(P_{n}^{\star})$ as $m \to \infty$ with n fixed. See Teymur et al. (2021) for full details, where non-myopic and mini-batch extensions of the greedy algorithm are also considered. The greedy algorithm can be viewed as a regularised version of the Frank–Wolfe algorithm (also called herding, or the conditional gradient method), for which a similar asymptotic result can be shown to hold (Chen et al., 2010; Bach et al., 2012; Chen et al., 2018). Related work includes Dwivedi and Mackey (2021, 2022); Shetty et al. (2022); Hayakawa et al. (2022). Since in what follows we aim to retrospectively improve MCMC output, where it is not unusual to encounter $n \approx 10^{4}-10^{6}$ , sparse approximation will be important.

This completes our overview of background material. In what follows we seek to mimic classical quantisation by deriving a choice for $\Pi$ that is straight-forward to sample using MCMC and is appropriately over-dispersed relative to P. This should be achieved while remaining in the framework of kernel discrepancies, so that optimal weights can be explicitly computed, and coupled with a sparse approximation that has low computational and storage cost.

# 3 Methodology

The methods that we consider first sample states $\{x_{1},\ldots,x_{n}\}$ using $\Pi$ -invariant MCMC, then post-process these states using kernel discrepancies (Section 2.2) and sparse approximation (Section 2.3), to obtain an approximation to the target P. A variational argument, which we present in Section 3.1, provides a suitable n-independent choice for $\Pi$ (which agrees with our intuition from Section 2.1 that $\Pi$ should be in some appropriate sense over-dispersed with respect to P). Sufficient conditions for strong consistency of the approximation are established in Section 3.3.

# 3.1 Selecting $\Pi$

Here we present a heuristic argument for a particular choice of $\Pi$ ; rigorous theoretical support for Stein $\Pi$ -Importance Sampling is then provided in Section 3.3. Our setting is that of Section 2.2, and the following will additionally be assumed:

# Assumption 1. It is assumed that

(A1) $C_{1}^{2} := \inf_{x \in \mathcal{X}} k(x, x) > 0$   
(A2) $C_{2} := \int \sqrt{k(x, x)} \, \mathrm{d}P(x) < \infty.$

Note that (A2) implies that $\mathcal{H}(k)\subset L^{1}(P)$ , and thus (1) is in fact a strong (or Bochner) integral.

A direct analysis of the optimal states associated to the optimal weights $w^{\star}$ appears to be challenging due to the fact that the components of $w^{\star}$ are strongly inter-dependent. Our solution here is to instead consider optimal states associated with weights that, while not optimal, can be expected to perform much better than alternatives, with the advantage that their components are only weakly dependent. Specifically, we will be assuming that P is absolutely continuous with respect to $\Pi$ (denoted $P \ll \Pi$ ), and study convergence of self-normalised importance sampling (SNIS), i.e. the approximation $P_{n} = \sum_{i=1}^{n} w_{i} \delta(x_{i})$ , $w_{i} \propto (\mathrm{d}P / \mathrm{d}\Pi)(x_{i})$ , where $x_{1}, \ldots, x_{n} \sim \Pi$ are independent. Since $w \geq 0$ and $1^{\top}w = 1$ , from the optimality of $w^{\star}$ under these constraints we have that $D_{P}(P_{n}^{\star}) \leq D_{P}(\overline{P}_{n})$ . It is emphasised that the SNIS weights are a theoretical device only, and will not be used for computation; indeed, we can demonstrate that the SNIS weights w perform substantially worse than $w^{\star}$ in general.

The analysis of SNIS weights w is tractable when viewed as approximation of the kernel mean embedding $\mu_{P}$ in the Hilbert space $\mathcal{H}(k)$ . Indeed, recall that $D_{P}(P_{n}) = \|\xi_{n}/\sqrt{n}\|_{\mathcal{H}(k)}$ where $\xi_{n} = \sqrt{n}(\mu_{P_{n}} - \mu_{P})$ . Then, following Section 2.3.1 of Agapiou et al. (2017), we observe that

$$
\xi_ {n} = \sqrt {n} \left(\sum_ {i = 1} ^ {n} w _ {i} k (\cdot , x _ {i}) - \mu_ {P}\right) = \frac {\frac {1}{\sqrt {n}} \sum_ {i = 1} ^ {n} \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x _ {i}) [ k (\cdot , x _ {i}) - \mu_ {P} ]}{\frac {1}{n} \sum_ {i = 1} ^ {n} \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x _ {i})}. \tag {6}
$$

The idea is to seek $\Pi$ for which the asymptotic variance of $\xi_{n}$ is small. Supposing that

$$
\int \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x) ^ {2} \mathrm{d} \Pi (x) <   \infty , \tag {S1}
$$

from the weak law of large numbers the denominator in (6) converges in probability to 1. Further supposing that

$$
\int \left\| \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x) [ k (\cdot , x) - \mu_ {P} ] \right\| _ {\mathcal {H} (k)} ^ {2} \mathrm{d} \Pi (x) <   \infty , \tag {S2}
$$

from the Hilbert space central limit theorem the numerator in (6) converges in distribution to a Gaussian $\frac{1}{\sqrt{n}}\sum_{i=1}^{n}(\mathrm{d}P/\mathrm{d}\Pi)(x_i)[k(\cdot,x_i)-\mu_P]\stackrel{\mathrm{d}}{\to}\mathcal{N}(0,\mathcal{C})$ where $\mathcal{C}:\mathcal{H}(k)\to\mathcal{H}(k)$ is the covariance operator defined via

$$
\langle f, \mathcal {C} g \rangle_ {\mathcal {H} (k)} = \int \left\langle f, \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x) [ k (\cdot , x) - \mu_ {P} ] \right\rangle_ {\mathcal {H} (k)} \left\langle g, \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x) [ k (\cdot , x) - \mu_ {P} ] \right\rangle_ {\mathcal {H} (k)} \mathrm{d} \Pi (x),
$$

see Section 10.1 of Ledoux and Talagrand (1991). Thus, from Slutsky's lemma applied to (6), we conclude that $\xi_n \xrightarrow{\mathrm{d}} \mathcal{N}(0, \mathcal{C})$ . Recalling that $nD_P(P_n)^2 = \| \xi_n\|_{\mathcal{H}(k)}^2$ , and noting that the mean square of the limiting Gaussian random variable is $\operatorname{tr}(\mathcal{C})$ , a natural idea is to select the sampling distribution $\Pi$ such that $\operatorname{tr}(\mathcal{C})$ is minimised.

Fortunately, the trace of C can be explicitly computed. It simplifies presentation to restrict attention to a Stein kernel $k = k_{P}$ , for which $\mu_{P} = 0$ , giving $\operatorname{tr}(\mathcal{C}) = \int (\mathrm{d}P / \mathrm{d}\Pi)(x)^{2} k_{P}(x) \, \mathrm{d}\Pi(x)$ , where for convenience we have let $k_{P}(x) := k_{P}(x, x)$ . Assuming that P and $\Pi$ admit densities p and $\pi$ on $X = R^{d}$ , the variational problem we wish to solve is

$$
\underset {\pi \in \mathcal {Q}} {\arg \min} \int \frac {p (x) ^ {2}}{\pi (x)} k _ {P} (x) \mathrm{d} x \quad \text { s.t. } \quad \int \pi (x) \mathrm{d} x = 1, \tag {7}
$$

where Q be the set of positive measures on $R^{d}$ for which (S1-2) are satisfied. To solve this problem, we first relax the constraints (S1-2) and solve the relaxed problem using the Euler–Lagrange equations, which yield

$$
\pi (x) \propto p (x) \sqrt {k _ {P} (x)}. \tag {8}
$$

Note that the normalisation constant of $\pi$ is $C_{2}$ from (1), whose existence we assumed. Then we verify that (S1-2) in fact hold for this choice of $\Pi$ . Indeed,

$$
(\mathrm{S} 1) = \int \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x) ^ {2} \mathrm{d} \Pi (x) = C _ {2} \int \frac {1}{k _ {P} (x)} \mathrm{d} \Pi (x) \leq \frac {C _ {2}}{C _ {1} ^ {2}} <   \infty
$$

$$
(\mathbf {S 2}) = \int \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x) ^ {2} k _ {P} (x) \mathrm{d} \Pi (x) = C _ {2} \int \sqrt {k _ {P} (x)} \mathrm{d} P (x) = C _ {2} ^ {2} <   \infty ,
$$

![](images/f14839db6fc110dd00d556c55dacaeda9fbb3f77d9668f1019a05c87cf201fc4.jpg)

<details>
<summary>line</summary>

| x    | P     | Π (Langevin) | Π (KGM1) | Π (KGM3) |
| ---- | ----- | ------------ | -------- | -------- |
| -3.0 | 0.000 | 0.000        | 0.000    | 0.000    |
| -2.5 | 0.050 | 0.080        | 0.070    | 0.120    |
| -2.0 | 0.150 | 0.250        | 0.240    | 0.350    |
| -1.5 | 0.250 | 0.320        | 0.310    | 0.360    |
| -1.0 | 0.350 | 0.340        | 0.330    | 0.320    |
| -0.5 | 0.400 | 0.310        | 0.290    | 0.280    |
| 0.0  | 0.380 | 0.280        | 0.270    | 0.250    |
| 0.5  | 0.350 | 0.250        | 0.240    | 0.220    |
| 1.0  | 0.300 | 0.220        | 0.210    | 0.190    |
| 1.5  | 0.250 | 0.180        | 0.170    | 0.160    |
| 2.0  | 0.150 | 0.120        | 0.110    | 0.130    |
| 2.5  | 0.050 | 0.060        | 0.070    | 0.140    |
| 3.0  | 0.000 | 0.010        | 0.020    | 0.150    |
</details>

(1.1)

![](images/09dceaf0e31cd2daec66926cadb4c70a045c51852f66286ff7b0980dfac98f98.jpg)

<details>
<summary>line</summary>

| n    | P (Langevin) | II (Langevin) | P (KGM1) | II (KGM1) | P (KGM3) | II (KGM3) |
| ---- | ------------ | ------------- | -------- | --------- | -------- | --------- |
| 10^0 | ~1.0         | ~1.0          | ~1.0     | ~1.0      | ~1.0     | ~1.0      |
| 10^1 | ~0.1         | ~0.1          | ~0.1     | ~0.1      | ~0.1     | ~0.1      |
| 10^2 | ~0.01        | ~0.01         | ~0.01    | ~0.01     | ~0.01    | ~0.01     |
</details>

(1.2)   
Figure 1: Illustrating our choice of $\Pi$ in 1D. (a) The univariate target $P$ (black), and our choice of $\Pi$ based on the Langevin-Stein kernel (purple), the KGM1-Stein kernel (green), and the KGM3-Stein kernel (blue). (b) The mean kernel Stein discrepancy (KSD) for Stein $\Pi$ -Importance Sampling using the Stein kernels from (a); in each case, KSD was computed using the same Stein kernel used to construct $\Pi$ . Solid lines indicate the baseline case of sampling from $P$ , while dashed lines indicate sampling from $\Pi$ . (The experiment was repeated 100 times and standard error bars are plotted.)

which shows that we have indeed solved (7). The sampling distribution $\Pi$ we have obtained is characterised up to a normalisation constant in (8), so just like P we can sample from $\Pi$ using techniques such as MCMC. It is interesting to note that $\Pi$ is also optimal for standard importance sampling (i.e. without self-normalisation); see Lemma 1 of Adachi et al. (2022). The Stein kernel $k_{P}$ determines the extent to which $\Pi$ differs from P, as we illustrate next.

# 3.2 Illustration

For illustration, consider the univariate target $P$ (black curve) in Figure 1.1, a 3-component Gaussian mixture model. Our recommended choice of $\Pi$ in (8) is shown for both the Langevin-Stein kernel (purple curve) and the KGMs-Stein kernels with $s \in \{1,3\}$ (green and blue curves). The Stein discrepancy corresponding to the Langevin-Stein kernel provides control over weak convergence (i.e. convergence of integrals of functions that are continuous and bounded), while the KGMs-Stein kernel provides additional control over the convergence of polynomial moments up to order $s$ ; full details about the construction of Stein kernels are contained in Appendix A. The Langevin and KGM1-Stein kernels have $k_{P}(x) \asymp x^{2}$ , while the KGM3-Stein kernel has $k_{P}(x) \asymp x^{6}$ , in each case as $|x| \to \infty$ , and thus greater over-dispersion results from use of the KGM3-Stein kernel. This over-dispersion is less pronounced $^{4}$ in higher dimensions; see Appendix D.1.

To illustrate the performance of Stein $\Pi$ -Importance Sampling, we generated a sequence $(x_{n})_{n\in\mathbb{N}}$ of independent samples from $\Pi$ . For each $n\in\{1,\ldots,100\}$ , the samples $\{x_{1},\ldots,x_{n}\}$ were assigned optimal weights $w^{\star}$ by solving (3), and the associated KSD was computed. As a baseline, we performed the same calculation using independent samples from $P$ . Figure 1.2 indicates that, for both Stein kernels, substantial improvement results from the use of samples from $\Pi$ compared to the use of samples from $P$ . Interestingly, the KGM3–Stein kernel demonstrated a larger improvement compared to the Langevin–Stein kernel, suggesting that the choice of $\Pi$ may be more critical in settings where KSD enjoys a stronger form of convergence control.

To illustrate a posterior approximation task, consider a simple regression model $y_{i} = f_{i}(x) + \epsilon_{i}$ with $f_{i}(x) = x_{1}(1 + t_{i}x_{2})$ , $t_{i} = i - 5$ , $i = 1, \ldots, 10$ , with $\epsilon_{i}$ independent $\mathcal{N}(0,1)$ . The parameter $x = (x_{1}, x_{2})$ was assigned a prior $\mathcal{N}(0, I)$ . Data were simulated using $x = (0,0)$ . The posterior distribution P is depicted in the leftmost panel of Figure 2, while our choice of $\Pi$ corresponding to the Langevin (centre left), KGM3 (centre right) and Riemann–Stein kernels (right) are also displayed. For the Langevin and KGM3 kernels, the associated $\Pi$ target their mass toward regions where P varies the most. The reason for this behaviour is clearly seen for the Langevin–Stein kernel since

![](images/be8f6177fc8cc4cd3eb41af867da61552e9b9ff6e0297324f5a4dea7432a3ca6.jpg)

<details>
<summary>scatter</summary>

| x    | y    |
| ---- | ---- |
| -2   | 0    |
| 0    | 0    |
| 2    | 0    |
</details>

![](images/af59126b23d6d34b7ba8b9c60d2eaaf57384d66b52ed9db04580702f3d516ac3.jpg)

<details>
<summary>line</summary>

| x    | y    |
| ---- | ---- |
| -2   | -3   |
| 0    | 0    |
| 2    | 3    |
</details>

![](images/d467540eea3d3aea0da5c589a95e833dbf7040883ee5a27a296adfb4867b157d.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| -2   | -3.0  |
| 0    | 0.0   |
| 2    | 3.0   |
</details>

![](images/90ea60bfc7ea91c75acc3ffde4112653f2d1dabbd77a4442660089623445c5dc.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| -2   | -3.0  |
| -1   | -1.0  |
| 0    | 0.0   |
| 1    | 1.0   |
| 2    | 2.0   |
</details>

Figure 2: Illustrating our choice of $\Pi$ in 2D. The bivariate target $P$ (left), together with our choice of $\Pi$ based on the Langevin–Stein kernel (centre left), the KGM3–Stein kernel (centre right), and the Riemann–Stein kernel (right).

Algorithm 1 Π-Invariant Metropolis-Adjusted Langevin Algorithm (MALA)   
Require: $x_{0}$ (initial state), $\epsilon$ (step size), M (preconditioner matrix), n (chain length), $k_{P}$ (Stein kernel)
1: for $i = 1, \ldots, n$ do
2: $x' \leftarrow \underbrace{x_{i-1} + \epsilon M^{-1} \nabla \log p(x_{i-1}) + \frac{\epsilon}{2} M^{-1} \nabla \log k_{P}(x_{i-1})}_{=: \nu(x_{i-1})} + \sqrt{2\epsilon} M^{-1/2} Z_{i} \quad \triangleright Z_{i} \stackrel{\text{IID}}{\sim} \mathcal{N}(0, I)$ 3: $L \leftarrow \log \left( \frac{p(x')}{p(x_{i-1})} \right) + \frac{1}{2} \log \left( \frac{k_{P}(x')}{k_{P}(x_{i-1})} \right) - \frac{1}{4\epsilon} \|x_{i-1} - \nu(x')\|_{M-1}^{2} + \frac{1}{4\epsilon} \|x' - \nu(x_{i-1})\|_{M-1}^{2}$ 4: if $\log(U_{i}) < L$ then $x_{i} \leftarrow x'$ ; else $x_{i} \leftarrow x_{i-1}$ ; end if $\triangleright U_{i} \stackrel{\text{IID}}{\sim} \mathcal{U}([0,1])$ 5: end for

Algorithm 2 Stein $\Pi$ -Importance Sampling (SIIS-MALA)   
Require: $\{x_{1},\ldots,x_{n}\}$ from Algorithm 1, $k_{P}$ (Stein kernel)
1: $w^{\star} \in \arg \min_{w \in R^{d}} \{\langle w, K_{P} w \rangle : w \geq 0, 1^{\top} w = 1\}$ $\triangleright [K_{P}]_{i,j} = k_{P}(x_{i}, x_{j})$

$k_{P}(x)=c_{1}+c_{2}\|\nabla\log p(x)\|^{2}$ for some $c_{1}, c_{2}>0$ ; see Appendix C for detail. The Riemann–Stein kernel can be viewed as a preconditioned form of the Langevin–Stein kernel which takes into account the geometric structure of P; see Appendix A for full detail $^{5}$ . Results in Figure S2 demonstrate that Stein $\Pi$ -Importance Sampling improves upon the default Stein importance sampling method (i.e. with $\Pi$ and P equal) for all choices of kernel.

An additional illustration involving a GARCH model with d = 4 parameters is presented in Appendix D.4, where the effect of varying the order s of the KGM–Stein kernel is explored.

# 3.3 Theoretical Guarantees

The aim of this section is to establish when post-processing of $\Pi$ -invariant MCMC produces a strongly consistent approximation of P, for our recommended choice of $\Pi$ in (8). Our analysis focuses on MALA (Roberts and Stramer, 2002), leveraging the recent work of Durmus and Moulines (2022) to present explicit and verifiable conditions on P for our results to hold. In fact, we consider the more general preconditioned form of MALA, where the symmetric positive definite preconditioner matrix M is to be specified. Our results also allow for (optional) sparse approximation, to circumvent direct solution of (3) (c.f. Section 2.3). The resulting algorithms, which we call Stein $\Pi$ -Importance Sampling (SIIIS-MALA) and Stein $\Pi$ -Thinning (SIIT-MALA), are quite straight-forward and contained, respectively, in Algorithms 2 and 3. The linearly-constrained quadratic programme in Algorithm 2 was solved using the Python v3.10.4 packages qpsolvers v3.4.0 and proxsuite v0.3.7. While it is difficult to analyse the computational complexity associated with these methods, we believe they are at worst $O(n^{3})$ .

# Algorithm 3 Stein Π-Thinning (SΠT-MALA)

Require: $\{x_{1},\ldots ,x_{n}\}$ from Algorithm 1, $m$ (number of samples to retain), $k_{P}$ (Stein kernel)

1: for $i = 1,\dots ,m$ do

2: $y_{i}\gets \arg \min_{y\in \{x_{1},\dots,x_{n}\}}\frac{1}{2} k_{P}(y) + \sum_{j = 1}^{i - 1}k_{P}(y,y_{j})$

3: end for

Let $A \preceq B$ indicate that $A - B$ is a positive semi-definite matrix for $A, B \in \mathbb{R}^{n \times n}$ . For a symmetric positive definite matrix $A$ let $\| z \|_A := \sqrt{z^\top A^{-1}z}$ for $z \in \mathbb{R}^d$ . Let $C^s(\mathbb{R}^d)$ denote the set of $s$ -times continuously differentiable real-valued functions on $\mathbb{R}^d$ .

Theorem 1 (Strong consistency of SIIS- and SIIT-MALA). Let Assumption 1 hold where $k = k_{P}$ is a Stein kernel, and let $D_P: \mathcal{P}(\mathcal{X}) \to [0, \infty]$ denote the associated KSD. Assume also that

(A1) $\nabla \log p \in C^2(\mathbb{R}^d)$ with $\sup_{x \in \mathbb{R}^d} \| \nabla^2 \log p(x) \| < \infty$   
(A2) $\exists b_{1} > 0, B_{1} \geq 0$ such that $-\nabla^{2}\log p(x) \succeq b_{1}I$ for all $\|x\| \geq B_{1}$   
(A3) $k_{P} \in C^{2}(\mathbb{R}^{d})$ .   
(A4) $\exists 0 < b_{2} < 2b_{1}C_{1}^{2}, B_{2} \geq 0$ such that $\nabla^{2}k_{P}(x) \preceq b_{2}I$ for all $\| x\| \geq B_{2}$

Let $P_{n}^{\star} = \sum_{i=1}^{n} w_{i}^{\star} \delta(x_{i})$ be the result of running Algorithm 2 and let $P_{n,m} = \frac{1}{m} \sum_{i=1}^{m} \delta(y_{i})$ be the result of running Algorithm 3. Let $m \leq n$ and $m = \Omega((\log n)^{\delta})$ for some $\delta > 2$ . Then there exists $\epsilon_{0} > 0$ such that, for all step sizes $\epsilon \in (0, \epsilon_{0})$ and all initial states $x_{0} \in \mathbb{R}^{d}$ , $D_{P}(P_{n}^{\star}) \to 0$ , $D_{P}(P_{n,m}) \to 0$ almost surely as $m, n \to \infty$ .

The proof is in Appendix B. Compared to earlier authors $^{6}$ , such as Chen et al. (2019); Riabiz et al. (2022), a major novelty here is that our assumptions are explicit and can often be verified (see also Hodgkinson et al., 2020). (A2) is strong log-concavity of P when $B_{1}=0$ , while for $B_{1}>0$ this condition is slightly stronger than the related distant dissipativity condition assumed in earlier work (Gorham and Mackey, 2017; Riabiz et al., 2022). (A4) holds for the Langevin–Stein kernel (i.e. weak convergence control) and for the KGM1–Stein kernel (i.e. weak convergence control + control over first moments), but not for the higher-order KGM–Stein kernels. Extending our proof strategy to the higher-order KGM–Stein kernels would require further research into the convergence properties of MALA, and this is expected to be difficult.

# 4 Benchmarking on PosteriorDB

The area of Bayesian computation has historically lacked a common set of benchmark problems, with classical examples being insufficiently difficult and case-studies being hand-picked (Chopin and Ridgway, 2017). To introduce objectivity into our assessment, we exploited the recently released PosteriorDB benchmark (Magnusson et al., 2022). This project is an attempt toward standardised benchmarking, consisting of a collection of posteriors to be numerically approximated. Here, we systematically compared the performance of SIIS-MALA against the default Stein importance sampling algorithm (i.e. $\Pi = P$ ; denoted SIS-MALA), and also against unprocessed $P$ -invariant MALA (i.e. uniform weights), reporting results across the breadth of PosteriorDB. The test problems in PosteriorDB are defined in the Stan probabilistic programming language, and so BridgeStan (Roualdes et al., 2023) was used to directly access posterior densities and their gradients as required. For all instances of MALA, an adaptive algorithm was used to learn a suitable preconditioner matrix $M$ during the warm-up period; see Appendix D.3. All experiments that we report can be reproduced using code available at https://github.com/congyewang/Stein-Pi-Importance-Sampling.

Results are reported in Table 1 for $n = 3 \times 10^{3}$ samples from MALA. These focus on the Langevin-Stein kernel, for which our theory holds, and the KGM3-Stein kernel, for which it does not. There was a significant improvement of SIIS-MALA over SIS-MALA in $73\%$ of test problems for the Langevin-Stein kernel and in $65\%$ of test problems for the KGM3-Stein kernel. Compared to

<table><tr><td></td><td></td><td colspan="3">Langevin-Stein Kernel</td><td colspan="3">KGM3-Stein Kernel</td></tr><tr><td>Task</td><td>d</td><td>MALA</td><td>SIS - MALA</td><td>SIIS - MALA</td><td>MALA</td><td>SIS - MALA</td><td>SIIS - MALA</td></tr><tr><td>earnings-earn_height</td><td>3</td><td>1.41</td><td>0.0674</td><td>0.0332</td><td>5.33</td><td>0.656</td><td>0.181</td></tr><tr><td>gp_pois_regr-gp_regr</td><td>3</td><td>0.298</td><td>0.0436</td><td>0.0373</td><td>1.22</td><td>0.385</td><td>0.223</td></tr><tr><td>kidiq-kidscore_momhs</td><td>3</td><td>1.04</td><td>0.109</td><td>0.0941</td><td>4.66</td><td>0.848</td><td>0.476</td></tr><tr><td>kidiq-kidscore_momiq</td><td>3</td><td>5.03</td><td>0.516</td><td>0.358</td><td>25.3</td><td>4.86</td><td>1.55</td></tr><tr><td>mesquite-logmesquite_logvolume</td><td>3</td><td>1.10</td><td>0.179</td><td>0.156</td><td>4.97</td><td>1.70</td><td>0.844</td></tr><tr><td>arma-arma11</td><td>4</td><td>4.47</td><td>1.09</td><td>1.01</td><td>26.0</td><td>8.91</td><td>6.03</td></tr><tr><td>earnings-logearn_logheight_male</td><td>4</td><td>9.46</td><td>1.96</td><td>1.59</td><td>53.9</td><td>15.4</td><td>8.65</td></tr><tr><td>garch-garch11</td><td>4</td><td>0.543</td><td>0.159</td><td>0.130</td><td>4.70</td><td>1.16</td><td>1.01</td></tr><tr><td>kidiq-kidscore_momhsiq</td><td>4</td><td>5.21</td><td>0.982</td><td>0.897</td><td>29.3</td><td>7.25</td><td>5.05</td></tr><tr><td>earnings-logearn_interaction_z</td><td>5</td><td>3.09</td><td>1.36</td><td>1.33</td><td>19.3</td><td>10.4</td><td>8.94</td></tr><tr><td>kidiq-kidscore_interaction</td><td>5</td><td>7.74</td><td>1.65</td><td>1.79</td><td>47.8</td><td>13.2</td><td>10.1</td></tr><tr><td>kidiq_with_mom_work-kidscore_interaction_c</td><td>5</td><td>1.35</td><td>0.659</td><td>0.711</td><td>7.92</td><td>4.05</td><td>4.17</td></tr><tr><td>kidiq_with_mom_work-kidscore_interaction_c2</td><td>5</td><td>1.38</td><td>0.689</td><td>0.699</td><td>8.09</td><td>4.24</td><td>4.25</td></tr><tr><td>kidiq_with_mom_work-kidscore_interaction_z</td><td>5</td><td>1.11</td><td>0.500</td><td>0.499</td><td>6.62</td><td>2.63</td><td>3.25</td></tr><tr><td>kidiq_with_mom_work-kidscore_mom_work</td><td>5</td><td>1.07</td><td>0.507</td><td>0.545</td><td>6.70</td><td>2.63</td><td>3.04</td></tr><tr><td>low_dim_gauss_mix-low_dim_gauss_mix</td><td>5</td><td>5.51</td><td>1.87</td><td>1.76</td><td>37.5</td><td>14.7</td><td>11.3</td></tr><tr><td>mesquite-logmesquite_logva</td><td>5</td><td>1.83</td><td>0.821</td><td>0.818</td><td>12.6</td><td>5.73</td><td>5.59</td></tr><tr><td>hmm_example-hmm_example</td><td>6</td><td>1.99</td><td>0.578</td><td>0.523</td><td>11.6</td><td>4.13</td><td>3.40</td></tr><tr><td>sblrc-blr</td><td>6</td><td>479</td><td>154</td><td>134</td><td>3300</td><td>1100</td><td>854</td></tr><tr><td>sblri-blr</td><td>6</td><td>201</td><td>66.7</td><td>60.3</td><td>1340</td><td>514</td><td>595</td></tr><tr><td>arK-arK</td><td>7</td><td>6.87</td><td>3.39</td><td>3.16</td><td>60.4</td><td>26.4</td><td>23.0</td></tr><tr><td>mesquite-logmesquite_logvash</td><td>7</td><td>1.89</td><td>1.18</td><td>1.23</td><td>15.5</td><td>8.88</td><td>10.1</td></tr><tr><td>bball_drive_event_0-hmm_drive_0</td><td>8</td><td>1.15</td><td>0.679</td><td>0.698</td><td>8.55</td><td>4.72</td><td>3.99</td></tr><tr><td>bball_drive_event_1-hmm_drive_1</td><td>8</td><td>42.9</td><td>11.9</td><td>12.4</td><td>285</td><td>85.6</td><td>67.8</td></tr><tr><td>hudson_lynx_hare-lotka_volterra</td><td>8</td><td>4.62</td><td>2.29</td><td>2.15</td><td>47.4</td><td>18.8</td><td>18.9</td></tr><tr><td>mesquite-logmesquite</td><td>8</td><td>1.46</td><td>1.00</td><td>1.06</td><td>13.3</td><td>8.28</td><td>9.14</td></tr><tr><td>mesquite-logmesquite_logvas</td><td>8</td><td>2.02</td><td>1.31</td><td>1.35</td><td>19.2</td><td>10.8</td><td>12.2</td></tr><tr><td>mesquite-mesquite</td><td>8</td><td>0.429</td><td>0.268</td><td>0.235</td><td>3.71</td><td>2.17</td><td>2.42</td></tr><tr><td>eight_schools-eight_schools_centered</td><td>10</td><td>0.526</td><td>0.100</td><td>0.182</td><td>7.53</td><td>2.15</td><td>215</td></tr><tr><td>eight_schools-eight_schools_noncentered</td><td>10</td><td>0.210</td><td>0.137</td><td>0.137</td><td>43.6</td><td>28.7</td><td>27.5</td></tr><tr><td>nes1972-nes</td><td>10</td><td>6.16</td><td>3.89</td><td>3.45</td><td>72.9</td><td>36.2</td><td>34.4</td></tr><tr><td>nes1976-nes</td><td>10</td><td>6.67</td><td>3.86</td><td>3.53</td><td>77.5</td><td>35.5</td><td>34.4</td></tr><tr><td>nes1980-nes</td><td>10</td><td>4.34</td><td>2.68</td><td>2.57</td><td>49.8</td><td>25.4</td><td>25.7</td></tr><tr><td>nes1984-nes</td><td>10</td><td>6.18</td><td>3.75</td><td>3.43</td><td>71.3</td><td>34.9</td><td>33.6</td></tr><tr><td>nes1988-nes</td><td>10</td><td>7.40</td><td>3.70</td><td>3.27</td><td>81.4</td><td>34.6</td><td>32.4</td></tr><tr><td>nes1992-nes</td><td>10</td><td>7.52</td><td>4.32</td><td>3.84</td><td>89.1</td><td>39.7</td><td>37.3</td></tr><tr><td>nes1996-nes</td><td>10</td><td>6.44</td><td>3.87</td><td>3.53</td><td>74.1</td><td>36.4</td><td>34.3</td></tr><tr><td>nes2000-nes</td><td>10</td><td>3.35</td><td>2.22</td><td>2.20</td><td>38.6</td><td>21.3</td><td>22.8</td></tr><tr><td>diamonds-diamonds</td><td>26</td><td>196</td><td>157</td><td>143</td><td>5120</td><td>2990</td><td>2620</td></tr><tr><td>mcycle_gp-accel_gp</td><td>66</td><td>11.3</td><td>8.25</td><td>9.79</td><td>960</td><td>623</td><td>815</td></tr></table>

Table 1: Benchmarking on PosteriorDB. Here we compared raw output from MALA with the post-processed output provided by the default Stein importance sampling method of Liu and Lee (2017) (SIS-MALA) and the proposed Stein II-Importance Sampling method (SIIS-MALA). Here $d = \dim(P)$ and the number of MALA samples was $n = 3 \times 10^{3}$ . The Langevin and KGM3–Stein kernels were used for SIS-MALA and SIIS-MALA and the associated KSDs are reported. Ten replicates were computed and statistically significant improvement is highlighted in bold.

unprocessed MALA, a significant improvement occurred in 100% and 97% of cases, respectively for each kernel. However, the extent of improvement decreased when the dimension d of the target increased, supporting the intuition that we set out earlier and in Appendix D.1. An in-depth breakdown of results, including varying the number n of samples that were used, and the performance SIIT-MALA, can be found in Appendices D.5 and D.6.

If P and its gradients are cheap to evaluate, the computational cost of MALA is lower than that of SIS-MALA, and one could run more iterations of MALA for an equivalent computational cost. But for more complex P, the computational cost of all algorithms will be gated by the number of times P and its gradients need to be evaluated, making the direct comparison in Table 1 meaningful. Further, if we aim for a compressed representation of P, then some form of post-processing of MALA would be required, which would then entail an additional computational cost.

Our focus is on the development of algorithms for minimisation of KSDs; the properties of KSDs themselves are out of scope for this work $^{7}$ . Nonetheless, there is much interest in better understanding the properties of KSDs, and we therefore also report performance of SIIIS-MALA in terms of 1-Wasserstein divergence in Appendix D.7. The main contrast between these results and the results in Table 1 is that, being score-based, KSDs suffer from the blindness to mixing proportions phenomena which has previously been documented in Wenliang and Kanagawa (2021); Koehler et al. (2022); Liu et al. (2023). Caution should therefore be taken when using algorithms based on Stein discrepancies

in the context of posterior distributions with multiple high probability regions that are spatially separated. This is also a failure mode for MCMC algorithms such as MALA, and yet there are still many problems for which MALA has been successfully used.

The alternative choice $\Pi_{1}$ , with $\pi_{1}(x) \propto p(x)^{d/(d+1)}$ , which provides a generic form of overdispersion and is optimal for approximation in 1-Wasserstein divergence (c.f. Section 2.1), was also considered. Results in Appendix D.8 indicate that, while $\Pi_{1}$ yields an improvement compared to the baseline of using P itself, $\Pi_{1}$ may be less effective than our proposed $\Pi$ when P is skewed.

# 5 Discussion

This paper presented Stein $\Pi$ -Importance Sampling; an algorithm that is simple to implement, admits an end-to-end theoretical treatment, and achieves a significant improvement over existing post-processing methods based on KSD. On the negative side, second order derivatives of the statistical model are required, and we are ultimately bound to the performance of the KSD on which Stein $\Pi$ -Importance Sampling is based. Our analysis focused on MALA, but there is in principle no barrier to deriving sufficient conditions for consistent approximation that are applicable to other sampling algorithms, such as the unadjusted Langevin algorithm. Of course, it remains to be seen whether SIS-MALA or any of its variants will stand the test of time compared to continued development in MCMC methodology, but we believe this line of research merits further investigation. For models for which access to second order derivatives is impractical, our methodology and theoretical analysis are directly applicable to gradient-free KSD (Fisher and Oates, 2023), and this would be an interesting direction for future work. Similarly, alternatives to KSD that are better-suited to high-dimensional P could be considered, such as the sliced KSD of Gong et al. (2021a,b).

Acknowledgements CW was supported by the China Scholarship Council. HK and CJO were supported by EP/W019590/1. The authors are grateful to François-Xavier Briol for feedback on an earlier draft of the manuscript, and to the anonymous Reviewers for their input.

# References

Adachi, M., Hayakawa, S., Jørgensen, M., Oberhauser, H., and Osborne, M. A. (2022). Fast Bayesian inference with batch Bayesian quadrature via kernel recombination. In Proceedings of the 35th Conference on Neural Information Processing Systems.   
Agapiou, S., Papaspiliopoulos, O., Sanz-Alonso, D., and Stuart, A. M. (2017). Importance sampling: Intrinsic dimension and computational cost. Statistical Science, 32(3):405–431.   
Anastasiou, A., Barp, A., Briol, F.-X., Ebner, B., Gaunt, R. E., Ghaderinezhad, F., Gorham, J., Gretton, A., Ley, C., Liu, Q., Mackey, L., Oates, C. J., Reinert, G., and Swan, Y. (2023). Stein's method meets statistics: A review of some recent developments. Statistical Science, (38):120–139.   
Bach, F. (2017). On the equivalence between kernel quadrature rules and random feature expansions. The Journal of Machine Learning Research, 18(1):714–751.   
Bach, F., Lacoste-Julien, S., and Obozinski, G. (2012). On the equivalence between herding and conditional gradient algorithms. In Proceedings of the 29th International Conference on Machine Learning.   
Barbour, A. D. (1988). Stein's method and poisson process convergence. Journal of Applied Probability, 25(A):175–184.   
Barbour, A. D. (1990). Stein's method for diffusion approximations. Probability Theory and Related Fields, 84(3):297-322.   
Barp, A., Oates, C. J., Porcu, E., and Girolami, M. (2022a). A Riemann–Stein kernel method. Bernoulli, 28(4):2181–2208.   
Barp, A., Simon-Gabriel, C.-J., Girolami, M., and Mackey, L. (2022b). Targeted separation and convergence with kernel discrepancies. arXiv preprint arXiv:2209.12835.

Bénard, C., Staber, B., and Da Veiga, S. (2023). Kernel stein discrepancy thinning: A theoretical perspective of pathologies and a practical fix with regularization. arXiv preprint arXiv:2301.13528.   
Briol, F.-X., Oates, C. J., Cockayne, J., Chen, W. Y., and Girolami, M. (2017). On the sampling problem for kernel quadrature. In Proceedings of the 34th International Conference on Machine Learning, pages 586–595.   
Briol, F.-X., Oates, C. J., Girolami, M., Osborne, M. A., and Sejdinovic, D. (2019). Probabilistic integration: A role in statistical computation (with discussion and rejoinder). Statistical Science, 34(1):1–22.   
Carmeli, C., De Vito, E., and Toigo, A. (2006). Vector valued reproducing kernel Hilbert spaces of integrable functions and mercer theorem. Analysis and Applications, 4(04):377–408.   
Chen, W. Y., Barp, A., Briol, F.-X., Gorham, J., Girolami, M., Mackey, L., and Oates, C. J. (2019). Stein point Markov chain Monte Carlo. In Proceedings of the 36th International Conference on Machine Learning, pages 1011–1021.   
Chen, W. Y., Mackey, L., Gorham, J., Briol, F.-X., and Oates, C. J. (2018). Stein points. In Proceedings of the 35th International Conference on Machine Learning, pages 844–853.   
Chen, Y., Welling, M., and Smola, A. (2010). Super-samples from kernel herding. In Proceedings of the 26th Conference on Uncertainty in Artificial Intelligence, pages 109–116.   
Chopin, N. and Ducrocq, G. (2021). Fast compression of MCMC output. Entropy, 23(8):1017.   
Chopin, N. and Ridgway, J. (2017). Leave pima indians alone: Binary regression as a benchmark for Bayesian computation. Statistical Science, 32(1):64–87.   
Chwialkowski, K., Strathmann, H., and Gretton, A. (2016). A kernel test of goodness of fit. In Proceedings of the 33rd International Conference on Machine Learning, pages 2606–2615.   
Cohort, P. (2004). Limit theorems for random normalized distortion. The Annals of Applied Probability, 14(1):118–143.   
Durmus, A. and Moulines, É. (2022). On the geometric convergence for MALA under verifiable conditions. arXiv preprint arXiv:2201.01951.   
Dwivedi, R. and Mackey, L. (2021). Kernel thinning. In Proceedings of 34th Conference on Learning Theory, pages 1753–1753.   
Dwivedi, R. and Mackey, L. (2022). Generalized kernel thinning. In Proceedings of the 10th International Conference on Learning Representations.   
Ehler, M., Gräf, M., and Oates, C. J. (2019). Optimal Monte Carlo integration on closed manifolds. Statistics and Computing, 29(6):1203–1214.   
Fisher, M. A. and Oates, C. J. (2023). Gradient-free kernel Stein discrepancy. In Proceedings of the 37th Conference on Neural Information Processing Systems.   
Girolami, M. and Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2):123–214.   
Gong, W., Li, Y., and Hernández-Lobato, J. M. (2021a). Sliced kernelized Stein discrepancy. In Proceedings of the 9th International Conference on Learning Representations.   
Gong, W., Zhang, K., Li, Y., and Hernández-Lobato, J. M. (2021b). Active slices for sliced Stein discrepancy. arXiv:2102.03159.   
Gorham, J., Duncan, A. B., Vollmer, S. J., and Mackey, L. (2019). Measuring sample quality with diffusions. The Annals of Applied Probability, 29(5):2884–2928.   
Gorham, J. and Mackey, L. (2015). Measuring sample quality with Stein's method. In Proceedings of the 29th Conference on Neural Information Processing Systems, pages 226–234.

Gorham, J. and Mackey, L. (2017). Measuring sample quality with kernels. In Proceedings of the 34th International Conference on Machine Learning, pages 1292–1301.   
Gotze, F. (1991). On the rate of convergence in the multivariate CLT. The Annals of Probability, pages 724–739.   
Graf, S. and Luschgy, H. (2007). Foundations of Quantization for Probability Distributions. Springer.   
Hawkins, C., Koppel, A., and Zhang, Z. (2022). Online, informative MCMC thinning with kernelized Stein discrepancy. arXiv preprint arXiv:2201.07130.   
Hayakawa, S., Oberhauser, H., and Lyons, T. (2022). Positively weighted kernel quadrature via subsampling. In Proceedings of the 35th Conference on Neural Information Processing Systems.   
Hodgkinson, L., Salomone, R., and Roosta, F. (2020). The reproducing Stein kernel approach for post-hoc corrected sampling. arXiv preprint arXiv:2001.09266.   
Kanagawa, H., Gretton, A., and Mackey, L. (2022). Controlling moments with kernel Stein discrepancies. arXiv preprint arXiv:2211.05408 (v1).   
Karvonen, T., Oates, C. J., and Girolami, M. (2021). Integration in reproducing kernel Hilbert spaces of Gaussian kernels. Mathematics of Computation, 90(331):2209–2233.   
Karvonen, T., Oates, C. J., and Sarkka, S. (2018). A Bayes–Sard cubature method. In Proceedings of the 32nd Conference on Neural Information Processing Systems.   
Kent, J. (1978). Time-reversible diffusions. Advances in Applied Probability, 10(4):819–835.   
Koehler, F., Heckett, A., and Risteski, A. (2022). Statistical efficiency of score matching: The view from isoperimetry. In Proceedings of the 36th Conference on Neural Information Processing Systems.   
Ledoux, M. and Talagrand, M. (1991). Probability in Banach Spaces: Isoperimetry and Processes, volume 23. Springer Science & Business Media.   
Liu, Q. and Lee, J. (2017). Black-box importance sampling. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, pages 952–961.   
Liu, Q., Lee, J., and Jordan, M. (2016). A kernelized Stein discrepancy for goodness-of-fit tests. In Proceedings of the 33rd International Conference on Machine Learning, pages 276–284.   
Liu, X., Duncan, A., and Gandy, A. (2023). Using perturbation to improve goodness-of-fit tests based on kernelized Stein discrepancy. In Proceedings of the 40th International Conference on Machine Learning.   
Magnusson, M., Bürkner, P., and Vehtari, A. (2022). PosteriorDB: A set of posteriors for Bayesian inference and probabilistic programming. https://github.com/stan-dev/posteriordb.   
Meyn, S. P. and Tweedie, R. L. (2012). Markov Chains and Stochastic Stability. Springer Science & Business Media.   
Oates, C. J., Girolami, M., and Chopin, N. (2017). Control functionals for Monte Carlo integration. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 79(3):695–718.   
Riabiz, M., Chen, W., Cockayne, J., Swietach, P., Niederer, S. A., Mackey, L., and Oates, C. J. (2022). Optimal thinning of MCMC output. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 84(4):1059–1081.   
Roberts, G. O. and Rosenthal, J. S. (1998). Optimal scaling of discrete approximations to Langevin diffusions. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 60(1):255–268.   
Roberts, G. O. and Stramer, O. (2002). Langevin diffusions and Metropolis–Hastings algorithms. Methodology and Computing in Applied Probability, 4:337–357.

Roualdes, E., Ward, B., Axen, S., and Carpenter, B. (2023). BridgeStan: Efficient in-memory access to Stan programs through Python, Julia, and R. https://github.com/roualdes/bridgestan.   
Shetty, A., Dwivedi, R., and Mackey, L. (2022). Distribution compression in near-linear time. In Proceedings of the 10th International Conference on Learning Representations.   
Sonnleitner, M. (2022). The Power of Random Information for Numerical Approximation and Integration. PhD thesis, University of Passau.   
Teymur, O., Gorham, J., Riabiz, M., and Oates, C. J. (2021). Optimal quantisation of probability measures using maximum mean discrepancy. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, pages 1027–1035.   
Wenliang, L. K. and Kanagawa, H. (2021). Blindness of score-based methods to isolated components and mixing proportions. In Proceedings of “Your Model is Wrong” @ the 35th Conference on Neural Information Processing Systems.

# Appendices

These appendices contains supporting material for the paper Stein II-Importance Sampling. The mathematical background on Stein kernels is contained in Appendix A. The proof of Theorem 1 is contained in Appendix B. For implementation of Stein II-Importance Sampling without the aid of automatic differentiation, various explicit derivatives are required; the relevant calculations can be found in Appendix C. The empirical protocols and additional empirical results are presented in Appendix D.

# A Mathematical Background

This appendix contains mathematical background on reproducing kernels and Stein kernels, as used in the main text. Appendix A.1 introduces matrix-valued reproducing kernels, while Appendix A.2 specialises to Stein kernels by application of a Stein operator to a matrix-valued kernel. A selection of useful Stein kernels are presented in Appendix A.3.

# A.1 Matrix-Valued Reproducing Kernels

A matrix-valued kernel is a function $K:\mathbb{R}^d\times \mathbb{R}^d\to \mathbb{R}^{d\times d}$ , that is both

1. symmetric; $K(x,y) = K(y,x)$ for all $x,y\in \mathbb{R}^d$ , and   
2. positive semi-definite; $\sum_{i=1}^{n} \sum_{j=1}^{n} \langle c_i, K(x_i, x_j) c_j \rangle \geq 0$ for all $x_1, \ldots, x_n \in \mathbb{R}^d$ and $c_1, \ldots, c_n \in \mathbb{R}^d$ .

Let $K_{x} = K(\cdot, x)$ . For vector-valued functions $g, g' : \mathbb{R}^{d} \to \mathbb{R}^{d}$ , defined by $g = \sum_{i=1}^{n} K_{x_i} c_i$ and $g' = \sum_{j=1}^{m} K_{x_j'} c_i'$ , define an inner product

$$
\langle g, g ^ {\prime} \rangle_ {\mathcal {H} (K)} = \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {m} \left\langle c _ {i}, K \left(x _ {i}, x _ {j} ^ {\prime}\right) c _ {j} ^ {\prime} \right\rangle . \tag {9}
$$

There is a unique Hilbert space of such vector-valued functions associated to K, denoted $\mathcal{H}(K)$ ; see Proposition 2.1 of Carmeli et al. (2006). This space is characterised as

$$
\mathcal {H} (K) = \overline {{\operatorname{span}}} \left\{K _ {x} c: x, c \in \mathbb {R} ^ {d} \right\}
$$

where here the closure is taken with respect to the inner product in (9). It can be shown that $\mathcal{H}(K)$ is in fact a reproducing kernel Hilbert space (RKHS) which satisfies the reproducing property

$$
\langle g, K _ {x} c \rangle_ {\mathcal {H} (K)} = \langle g (x), c \rangle
$$

for all $g \in \mathcal{H}(K)$ and $x, c \in \mathbb{R}^d$ . Matrix-valued kernels are the natural starting point for construction of KSDs, as described next.

# A.2 Stein Kernels

A general construction for Stein kernels is to first identify a matrix-valued RKHS $\mathcal{H}(K)$ and an operator $S_{P}:\mathcal{H}(K)\to L^{1}(P)$ for which $\int S_{p}h\,dP=0$ for all $h\in\mathcal{H}(K)$ . Such an operator will be called a Stein operator. The collection $\{S_{p}h:h\in\mathcal{H}(K)\}$ inherits the structure of an RKHS, whose reproducing kernel

$$
k _ {P} (x, y) = \left\langle S _ {P} K _ {x}, S _ {P} K _ {y} \right\rangle_ {\mathcal {H} (K)} \tag {10}
$$

is a Stein kernel, meaning that $\mu_{P}=0$ where $\mu_{P}$ is the kernel mean embedding from (1); see Barp et al. (2022b). Explicit calculations for the Stein kernels considered in this work can be found in Appendix C.

For univariate distributions, Barbour (1988) proposed to obtain Stein operators from infinitesimal generators of P-invariant continuous-time Markov processes; see also Barbour (1990); Gotze (1991).

The approach was extended to multivariate distributions in Gorham and Mackey (2015). The starting point is the $P$ -invariant Itô diffusion

$$
\mathrm{d} X _ {t} = \frac {1}{2} \frac {1}{p (X _ {t})} \nabla \cdot [ p (X _ {t}) M (X _ {t}) ] \mathrm{d} t + M (X _ {t}) ^ {1 / 2} \mathrm{d} W _ {t}, \tag {11}
$$

where $p$ is the density of $P$ , assumed to be positive, $M: \mathbb{R}^d \to \mathbb{R}^{d \times d}$ is a symmetric matrix called the diffusion matrix, and $W_t$ is a standard Wiener process (Kent, 1978; Roberts and Stramer, 2002). Here the notation $[\nabla \cdot A]_i = \nabla \cdot (A_{i,\cdot}^\top)$ indicates the divergence operator applied to each row of the matrix $A(x) \in \mathbb{R}^{d \times d}$ . The infinitesimal generator is

$$
(A _ {P} u) (x) = \frac {1}{2} \frac {1}{p (x)} \nabla \cdot [ p (x) M (x) \nabla u (x) ].
$$

Substituting $h(x)$ for $\frac{1}{2}\nabla u(x)$ , we obtain a Stein operator

$$
(S _ {P} h) (x) = \frac {1}{p (x)} \nabla \cdot [ p (x) M (x) h (x) ] \tag {12}
$$

called the diffusion Stein operator (Gorham et al., 2019). This is indeed a Stein operator, since under mild integrability conditions on K, the divergence theorem gives that $\int S_{p}h \, dP = 0$ for all $h \in \mathcal{H}(K)$ ; for full details and a proof see Barp et al. (2022b).

# A.3 Selecting a Stein Kernel

There are several choices for a Stein kernel, and which we should use depends on what form of convergence we hope to control (Gorham and Mackey, 2017; Gorham et al., 2019; Hodgkinson et al., 2020; Barp et al., 2022b; Kanagawa et al., 2022). Appendix A.3.1 describes the Langevin–Stein kernel for weak convergence control, Appendix A.3.2 describes the KGM–Stein kernels for additional control over moments, and Appendix A.3.3 presents the Riemann–Stein kernel, whose convergence properties have to-date been less well-studied.

All of the kernels that we consider have length scale parameters that need to be specified, and some also have location parameters to be specified. As a reasonably automatic default we define

$$
x _ {\star} \in \arg \max p (x), \qquad \Sigma^ {- 1} = - \nabla^ {2} \log p (x _ {\star})
$$

as a location and a matrix of characteristic length scales for P that will be used throughout. These values can typically be obtained using gradient-based optimisation, which is usually cheaper to perform compared to full approximation of P. It is assumed that $\nabla^{2}\log p(x_{\star})$ is positive definite in the sequel.

# A.3.1 Weak Convergence Control with Langevin–Stein Kernels

The first kernel we consider, which we called the Langevin–Stein kernel in the main text, was introduced by Gorham and Mackey (2017). This Stein kernel was developed for the purpose of controlling the weak convergence of a sequence $(Q_{n})_{n\in\mathbb{N}}\subset\mathcal{P}(\mathbb{R}^{d})$ to P. Recall that a sequence $(Q_{n})_{n\in\mathbb{N}}$ is said to converge weakly (or in distribution) to P if $\int f\,dQ_{n}\to\int f\,dP$ for all continuous bounded functions $f:\mathbb{R}^{d}\to\mathbb{R}$ . This convergence is denoted $Q_{n}\xrightarrow{d}P$ in shorthand.

The problem considered in Gorham and Mackey (2017) was how to select a combination of matrix-valued kernel $K$ (and, implicitly, a diffusion matrix $M$ ) such that the Stein kernel $k_{P}$ in (10) generates a KSD $D_{P}(Q)$ in (4) for which $D_{P}(Q_{n}) \to 0$ implies $Q_{n} \xrightarrow{\mathrm{d}} P$ . Their solution was to combine the inverse multi-quadric kernel with an identity diffusion matrix;

$$
K (x, y) = \left(1 + \| x - y \| _ {\Sigma} ^ {2}\right) ^ {- \beta} I, \quad M (x) = I
$$

for $\beta\in(0,1)$ . Provided that P has a density p for which $\nabla\log p(x)$ is Lipschitz, and that P is distantly dissipative (see Definition 4 of Gorham and Mackey, 2017), the associated KSD enjoys weak convergence control. Technically, the results in Gorham and Mackey (2017) apply only when $\Sigma=I$ , but Theorem 4 in Chen et al. (2019) demonstrated that they hold also for any positive definite $\Sigma$ . Following the recommendation of several previous authors, including Chen et al. (2018, 2019); Riabiz et al. (2022), we take $\beta=\frac{1}{2}$ throughout.

# A.3.2 Moment Convergence Control with KGM-Stein Kernels

Despite its many elegant properties, weak convergence can be insufficient for applications where we are interested in integrals $\int f \, dP$ for which the integrand $f : R^{d} \to R$ is unbounded. In particular, this is the case for moments of the form $f(x) = x_{1}^{\alpha_{1}} \ldots x_{d}^{\alpha_{d}}, 0 \neq \alpha \in \mathbb{N}_{0}^{d}$ . In such situations, we may seek also the stronger property of moment convergence control. The development of KSDs for moment convergence control was recently considered by Kanagawa et al. (2022), and we referred to their construction as the KGM–Stein kernels in the main text. (For convenience, we have adopted the initials of the authors in naming the KGM–Stein kernel.)

A sequence $(Q_{n})_{n\in\mathbb{N}}\subset\mathcal{P}(\mathbb{R}^{d})$ is said to converge to P in the sth order moment if $\int\|x\|^{s}\mathrm{d}Q_{n}(x)\to\int\|x\|^{s}\mathrm{d}P(x)$ . To establish convergence of moments, we need an additional condition on top of weak convergence control: uniform integrability control. A sequence of measures $(Q_{n})_{n\in\mathbb{N}}$ is said to have uniformly integrable sth moments if for any $\varepsilon>0$ , we can take r>0 such that

$$
\sup _ {n \in \mathbb {N}} \int_ {\| x \| > r} \| x \| ^ {s} \mathrm{d} Q _ {n} (x) <   \varepsilon .
$$

This condition essentially states that the tail decay of the measures is well-controlled (so that it has a convergent moment). The KSD convergence $D_{P}(Q_{n}) \to 0$ implies uniform integrability if for any $\varepsilon > 0$ , we can take $r_{\varepsilon} > 0$ and $f_{\varepsilon} \in \mathcal{H}(K)$ such that

$$
S _ {P} f _ {\varepsilon} (x) \geq \| x \| ^ {s} 1 \{\| x \| > r _ {\varepsilon} \} - \varepsilon , \tag {13}
$$

i.e., the Stein-modified RKHS can approximate the (norm-weighted) indicator function arbitrarily well. Such a function $f_{\varepsilon}$ can be explicitly constructed (while not guaranteed to be a member of the RKHS). Specifically, the choice $f_{\varepsilon} = (1 - \iota_{\varepsilon})g$ satisfies (13) under an appropriate dissipativity condition, where $\iota_{\varepsilon}$ is a differentiable indicator function vanishing outside a ball, and $g(x) = -x/\sqrt{1 + \|x\|^{2}}$ . This motivated Kanagawa et al. (2022) to introduce the sth order KGM–Stein kernel, which is based on the matrix-valued kernel and diffusion matrix

$$
K (x, y) = \left[ \phi (\| x - y \| _ {\Sigma}) + \kappa_ {\mathrm{lin}} (x, y) \right] I, \quad M (x) = \left(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {\frac {s - 1}{2}} I,
$$

where $(x,y)\mapsto \phi (\| x - y\|_{\Sigma})$ is a $C_0^1$ universal kernel (see Barp et al., 2022b, Theorem 4.8). For comparability of our results, we take $\phi$ to be the inverse multi-quadric $\phi (r) = (1 + r^2)^{-1 / 2}$ , and

$$
\kappa_ {\mathrm{lin}} (x, y) = \frac {1 + (x - x _ {\star}) ^ {\top} \Sigma^ {- 1} (y - x _ {\star})}{\sqrt {1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}} \sqrt {1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}}}.
$$

Here the normalised linear kernel $\kappa_{lin}$ ensures $g \in \mathcal{H}(K)$ , while the $C_{0}^{1}$ universal kernel $\phi$ allows approximation of $S_{P}\iota_{\varepsilon}g$ ; see Kanagawa et al. (2022).

# A.3.3 Exploiting Geometry with Riemann–Langevin–Stein Kernels

For academic interest only, here we describe the Riemann–Stein kernel that featured in Figure 2 of the main text. This Stein kernel is motivated by the analysis of Gorham et al. (2019), who argued that the use of rapidly mixing Itô diffusions in Stein operators can lead to sharper convergence control. The Riemann–Stein kernel is based on the class of so-called Riemannian diffusions considered in Girolami and Calderhead (2011), who proposed to take the diffusion matrix M in (11) to be $M = (\mathcal{I}_{\mathrm{prior}} + \mathcal{I}_{\mathrm{Fisher}})^{-1}$ , the inverse of the Fisher information matrix, $I_{Fisher}$ , regularised using the Hessian of the negative log-prior, $I_{prior}$ . For the two-dimensional illustration in Section 3.2, this leads to the diffusion matrix

$$
M (x) = \left(I + \sum_ {i = 1} ^ {n} [ \nabla f _ {i} (x) ] [ \nabla f _ {i} (x) ] ^ {\top}\right) ^ {- 1},
$$

where we recall that $y_{i}=f_{i}(x)+\epsilon_{i}$ , where the $\epsilon_{i}$ are independent with $\epsilon_{i}\sim\mathcal{N}(0,1)$ , and the prior is $x\sim\mathcal{N}(0,1)$ . For the presented experiment we paired the above diffusion matrix with the inverse multi-quadric kernel $K(x,y)=(1+\|x-y\|_{\Sigma}^{2})^{-\beta}$ for $\beta=\frac{1}{2}$ . The Riemann–Stein kernel extends naturally to distributions P defined on Riemannian manifolds X; see Barp et al. (2022a) and Example 1 of Hodgkinson et al. (2020).

Unfortunately, the Riemann–Stein kernel is prohibitively expensive in most real applications, since each evaluation of M requires a full scan through the size-n dataset. The computational complexity

of Stein $\Pi$ -Thinning with the Riemann–Stein kernel is therefore $O(m^{2}n^{2})$ , which is unfavourable compared to the $O(m^{2}n)$ complexity in the case where the Stein kernel is not data-dependent. Furthermore, the convergence control properties of the Riemann–Stein kernel have yet to be established. For these reasons we included the Riemann–Stein kernel for illustration only; further groundwork will be required before the Riemann-Stein kernel can be practically used.

# B Proof of Theorem 1

This appendix is devoted to the proof of Theorem 1. The proof is based on the recent work of Durmus and Moulines (2022), on the geometric convergence of MALA, and on the analysis of sparse (greedy) approximation of kernel discrepancies performed in Riabiz et al. (2022); these existing results are recalled in Appendix B.1. An additional technical result on preconditioned MALA is contained in Appendix B.2. The proof of Theorem 1 itself is contained in Appendix B.3.

# B.1 Auxiliary Results

To precisely describe the results on which our analysis is based, we first need to introduce some notation and terminology. Let $V: \mathcal{X} \to [1, \infty)$ and, for a function $f: \mathcal{X} \to \mathbb{R}$ and a measure $\mu$ on $\mathcal{X}$ , let

$$
\| f \| _ {V} := \sup _ {x \in \mathcal {X}} \frac {| f (x) |}{V (x)}, \quad \| \mu \| _ {V} := \sup _ {\| f \| _ {V} \leq 1} \left| \int_ {\mathcal {X}} f \mathrm{d} \mu \right|.
$$

Recall that a $Q$ -invariant Markov chain $(x_{i})_{i\in \mathbb{N}}\subset \mathcal{X}$ with $n^{\mathrm{th}}$ step transition kernel $\mathbf{Q}^n$ is $V$ -uniformly ergodic (see Theorem 16.0.1 of Meyn and Tweedie, 2012) if and only if $\exists R\in [0,\infty),\rho \in (0,1)$ such that

$$
\left\| \mathrm{Q} ^ {n} (x, \cdot) - Q \right\| _ {V} \leq R \rho^ {n} V (x) \tag {14}
$$

for all initial states $x \in \mathcal{X}$ and all $n \in \mathbb{N}$ .

Although MALA (Algorithm 1) is classical (Roberts and Stramer, 2002), until recently explicit sufficient conditions for ergodicity of MALA had not been obtained. The first result we will need is due Durmus and Moulines (2022), who presented the first explicit conditions for V-uniform convergence of MALA. It applies only to standard MALA, meaning that the preconditioning matrix M appearing in Algorithm 1 is the identity matrix. The extension of this result to preconditioned MALA will be handled in Appendix B.2.

Theorem 2. Let $Q \in \mathcal{P}(\mathbb{R}^d)$ admit a density, $q$ , such that

(DM1) there exists $x_0$ with $\nabla \log q(x_0) = 0$   
(DM2) $q$ is twice continuously differentiable with $\sup_{x\in \mathbb{R}^d}\| \nabla^2\log q(x - x_0)\| < \infty$   
(DM3) there exists b > 0 and $B \geq 0$ such that $-\nabla^{2} \log q(x - x_{0}) \succeq bI$ for all $\|x - x_{0}\| \geq B$ .

Then there exists $\epsilon_0 > 0$ such that for all step sizes $\epsilon \in (0, \epsilon_0)$ , standard $Q$ -invariant MALA (i.e. with $M = I$ ) is $V$ -uniformly ergodic for $V(x) = \exp\left(\frac{b}{16} \|x - x_0\|^2\right)$ .

Proof. This is Theorem 1 of Durmus and Moulines (2022).

![](images/8d05ffbba78dbe9f2a7fd5bb9a50efba954d1bdeca1ba278957f902eb27843eb.jpg)

The next result that we will need establishes consistency of the greedy algorithm applied to samples from a Markov chain that is Q-invariant.

Theorem 3. Let $P, Q \in \mathcal{P}(\mathcal{X})$ with $P \ll Q$ . Let $k_P: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ be a Stein kernel and let $D_P: \mathcal{X} \times \mathcal{X} \to [0, \infty]$ denote the associated KSD. Consider a $Q$ -invariant, time-homogeneous Markov chain $(x_i)_{i \in \mathbb{N}} \subset \mathcal{X}$ such that

(R $^{+}$ 1) (x $_{i}$ ) $_{i\in\mathbb{N}}$ is V-uniformly ergodic, such that V(x) ≥ $\frac{\mathrm{d}P}{\mathrm{d}Q}(x)\sqrt{k_{P}(x)}$

$(\mathbf{R}^{+}2)\sup_{i\in \mathbb{N}}\mathbb{E}\left[\frac{\mathrm{d}P}{\mathrm{d}Q} (x_{i})\sqrt{k_{P}(x_{i})} V(x_{i})\right] <   \infty$

$$
(\mathbf {R} ^ {+} 3) \text {there exists} \gamma > 0 \text {such that} \sup _ {i \in \mathbb {N}} \mathbb {E} \left[ \exp \left\{\gamma \max \left(1, \frac {\mathrm{d} P}{\mathrm{d} Q} (x _ {i}) ^ {2}\right) k _ {P} (x _ {i}) \right\} \right] <   \infty .
$$

Let $P_{n,m}$ be the result of running the greedy algorithm in (5). If $m \leq n$ and $\log(n) = O(m^{\gamma/2})$ for some $\gamma < 1$ , then $D_P(P_{n,m}) \to 0$ almost surely as $m, n \to \infty$ .

Proof. This is Theorem 3 of Riabiz et al. (2022).

![](images/3cda20fc63b062281174b24467f04c64d5c68f0e1e9895616020a071d8490a45.jpg)

# B.2 Preconditioned MALA

In addition to the auxiliary results in Appendix B.1, which concern standard MALA (i.e. with $M = I$ ), we require an elementary fact about MALA, namely that preconditioned MALA is equivalent to standard MALA under a linear transformation of the state variable. Recall that the $M$ -preconditioned MALA algorithm is a Metropolis-Hastings algorithm whose proposal is the Euler-Maruyama discretisation of the Itô diffusion (11).

Proposition 1. Let $M(x) \equiv M$ for a symmetric positive definite and position-independent matrix $M \in \mathbb{R}^{d \times d}$ . Let $Q \in \mathcal{P}(\mathbb{R}^d)$ admit a probability density function (PDF) $q$ for which the $Q$ -invariant diffusion $(X_t)_{t \geq 0}$ , given by setting $p = q$ in (11), is well-defined. Then under the change of variables $Y_t := M^{1/2} X_t$ ,

$$
\mathrm{d} Y _ {t} = \frac {1}{2} (\nabla \log \tilde {q}) (Y _ {t}) \mathrm{d} t + \mathrm{d} W _ {t}, \tag {15}
$$

where $\tilde{q}(x) \propto q(M^{-1/2}x)$ for all $x \in R^{d}$ .

Proof. From the chain rule,

$$
(\nabla \log \tilde {q}) (y) = \nabla_ {y} \log q (M ^ {- 1 / 2} y) = M ^ {- 1 / 2} (\nabla \log q) (M ^ {- 1 / 2} y),
$$

and thus, substituting $Y_{t} = M^{1 / 2}X_{t}$ , (15) is equal to

$$
\begin{array}{l} \mathrm{d} X _ {t} = M ^ {- 1 / 2} \left[ \frac {1}{2} M ^ {- 1 / 2} (\nabla \log q) (M ^ {- 1 / 2} M ^ {1 / 2} X _ {t}) + \mathrm{d} W _ {t} \right] \\ = \frac {1}{2} M ^ {- 1} (\nabla \log q) (X _ {t}) + M ^ {- 1 / 2} \mathrm{d} W _ {t}, \\ \end{array}
$$

which is identical to (11) in the case where $M(x) = M$ is constant.

![](images/aa541422817fac5dfcf5d6b5c38228779e576c1f894484e985218a1cd34ae816.jpg)

Let $Q$ and $\tilde{Q}$ be the distributions referred to in Proposition 1, whose PDFs are respectively $q(x)$ and $\tilde{q}(x) \propto q(M^{-1/2}x)$ . Proposition 1 then implies that the $M$ -preconditioned MALA algorithm applied to $Q$ (i.e. Algorithm 1 for $\Pi = Q$ ) is equivalent to the standard MALA algorithm (i.e. $M = I$ ) applied to $\tilde{Q}$ . This fact allows us to generalise the result of Theorem 2 as follows:

Corollary 1. Consider a symmetric positive definite matrix $M \in \mathbb{R}^{d \times d}$ . Assume that conditions (DM1-3) in Theorem 2 are satisfied. Then there exists $\epsilon_0' > 0$ and $b' > 0$ such that for all step sizes $\epsilon \in (0, \epsilon_0')$ , the $M$ -preconditioned $Q$ -invariant MALA is $V$ -uniformly ergodic for $V(x) = \exp\left(\frac{b'}{16} \|x - x_0\|^2\right)$ .

Proof. From Theorem 2 and Proposition 1, the result follows if we can establish (DM1-3) for $\tilde{Q}$ , since $M$ -preconditioned MALA is equivalent to standard MALA applied to $\tilde{Q}$ . For a matrix $A \in \mathbb{R}^{d \times d}$ , let $\lambda_{\min}(A)$ and $\lambda_{\max}(A)$ respectively denote the minimum and maximum eigenvalues of $A$ . For (DM1) we set $y_0 = M^{1/2}x_0$ and observe that

$$
(\nabla \log \tilde {q}) (y _ {0}) = M ^ {- 1 / 2} (\nabla \log q) (x _ {0}) = 0.
$$

For (DM2) we have that

$$
\begin{array}{l} \sup _ {y \in \mathbb {R} ^ {d}} \| \nabla^ {2} (\log \tilde {q}) (y - y _ {0}) \| = \sup _ {y \in \mathbb {R} ^ {d}} \| M ^ {- 1 / 2} (\nabla^ {2} \log q) (M ^ {- 1 / 2} (y - y _ {0})) M ^ {- 1 / 2} \| \\ \leq \lambda_ {\min} (M) ^ {- 1} \sup _ {x \in \mathbb {R} ^ {d}} \| (\nabla^ {2} \log q) (x - x _ {0}) \| <   \infty . \\ \end{array}
$$

For (DM3) we have that

$$
\begin{array}{l} - (\nabla^ {2} \log \tilde {q}) (y - y _ {0}) = - M ^ {- 1 / 2} (\nabla^ {2} \log q) (M ^ {- 1 / 2} (y - y _ {0})) M ^ {- 1 / 2} \\ = - M ^ {- 1 / 2} (\nabla^ {2} \log q) (x - x _ {0}) M ^ {- 1 / 2} \succeq M ^ {1 / 2} (b I) M ^ {1 / 2} = b M ^ {- 1} \succeq b ^ {\prime} I \\ \end{array}
$$

where $b' = b\lambda_{\max}(M)^{-1}$ , which holds for all $\|x - x_{0}\| \geq B$ , and in particular for all $\|y - y_{0}\| \geq B'$ where $B' = B\lambda_{\max}(M)^{1/2}$ . Thus (DM1-3) are established for $\tilde{Q}$ .

Remark 1. The choice $M = \Sigma^{-1}$ , which sets the preconditioner matrix M equal to the inverse of the length scale matrix $\Sigma$ used in the specification of the kernel K (c.f. Appendix A.3), leads to the elegant interpretation that Stein $\Pi$ -Importance Sampling applied to M-preconditioned MALA is equivalent to the Stein $\Pi$ -Importance Sampling applied to standard MALA (i.e. with M = I) for the whitened target $\tilde{P}$ with PDF $\tilde{p}(x) \propto p(M^{-1/2}x)$ . For our experiments, however, the preconditioner matrix M was learned during a warm-up phase of MALA, since in general the curvature of P (captured by $\Sigma$ ) and the curvature of $\Pi$ (captured by $M^{-1}$ ) may be different.

# B.3 Proof of Theorem 1

The route to establishing Theorem 1 has three parts. First, we establish (DM1-3) of Theorem 2 with $Q = \Pi$ , to deduce from Corollary 1 that $\Pi$ -invariant $M$ -preconditioned MALA is $V$ -uniformly ergodic. This in turn enables us to establish conditions $(\mathbb{R}^{+}1 - 3)$ of Theorem 3, again for $Q = \Pi$ , from which the strong consistency $D_P(P_{n,m}) \stackrel{\mathrm{a.s.}}{\to} 0$ of SIIT-MALA is established. Finally, we note that $0 \leq D_P(P_n^\star) \leq D_P(P_{n,m})$ , since the support of $P_{n,m}$ is contained in the support of $P_n^\star$ , and the latter is optimally weighted, whence also the strong consistency of SIIS-MALA.

Establish (DM1-3) First we establish (DM1-3) for $Q = \Pi$ . Fix $x_0 \in \mathbb{R}^d$ . For (DM2), first recall that the range of $k_P$ is $[C_1^2, \infty)$ where $C_1 > 0$ , from Assumption 1. Since $\log(\cdot)$ has bounded second derivatives on $[C_1^2, \infty)$ , there is a constant $C > 0$ such that

$$
\forall x \in \mathbb {R} ^ {d}, \quad \| \nabla^ {2} \log k _ {P} (x) \| \leq C \| \nabla^ {2} k _ {P} (x) \|.
$$

Thus, using compactness of the set $\{x : \|x - x_{0}\| \leq B_{2}\}$ ,

$$
\sup _ {x \in \mathbb {R} ^ {d}} \| \nabla^ {2} \log k _ {P} (x) \| \leq C \max \left(\underbrace {\sup _ {\| x - x _ {0} \| \leq B _ {2}} \| \nabla^ {2} k _ {P} (x) \|} _ {<   \infty \text {   by   (A3) }}, \underbrace {\sup _ {\| x - x _ {0} \| \geq B _ {2}} \| \nabla^ {2} k _ {P} (x) \|} _ {<   b _ {2} \| I \| \text {   by   (A4) }}\right) <   \infty . \tag {16}
$$

Now, $\pi$ is twice differentiable as it is the product of twice differentiable functions $p$ and $k_{P}^{1 / 2}$ from (A1) and (A3), and moreover

$$
\sup _ {x \in \mathbb {R} ^ {d}} \| \nabla^ {2} \log \pi (x - x _ {0}) \| \leq \underbrace {\sup _ {x \in \mathbb {R} ^ {d}} \| \nabla^ {2} \log p (x) \|} _ {<   \infty \text {by (A1)}} + \frac {1}{2} \underbrace {\sup _ {x \in \mathbb {R} ^ {d}} \| \nabla^ {2} \log k _ {P} (x) \|} _ {<   \infty \text {by (16)}} <   \infty ,
$$

so (DM2) is satisfied. For (DM3), first note from the chain and product rules that for all $\|x\|\geq B_{2}$

$$
\nabla^ {2} \log k _ {P} (x - x _ {0}) = \underbrace {\frac {\nabla^ {2} k _ {P} (x - x _ {0})}{k _ {P} (x - x _ {0})}} _ {\preceq \left(b _ {2} / C _ {1} ^ {2}\right) I \text { by   (A4)}} - \underbrace {\frac {[ \nabla k _ {P} (x - x _ {0}) ] [ \nabla k _ {P} (x - x _ {0}) ] ^ {\top}}{k _ {P} (x - x _ {0}) ^ {2}}} _ {\succeq 0} \preceq \frac {b _ {2}}{C _ {1} ^ {2}} I. \tag {17}
$$

Thus, for all $\|x - x_{0}\| \geq B := \|x_{0}\| + \max(B_{1}, B_{2})$ ,

$$
- \nabla^ {2} \log \pi (x - x _ {0}) = \underbrace {- \nabla^ {2} \log p (x - x _ {0})} _ {\succeq b _ {1} I \text {   by   (A2) }} - \frac {1}{2} \underbrace {\nabla^ {2} \log k _ {P} (x - x _ {0})} _ {\preceq (b _ {2} / C _ {1} ^ {2}) I \text {   by   (17) }} \succeq \underbrace {\left(b _ {1} - \frac {b _ {2}}{2 C _ {1} ^ {2}}\right)} _ {=: b > 0} I \tag {18}
$$

as required. The same argument establishes (DM1); from (18) we have $\lim_{\|x\|\to\infty}\pi(x)=0$ , and since $\pi$ is a continuously differentiable density there must exist an $x_{0}$ at which $\pi$ is locally minimised. Thus we have established (DM1-3) for $Q=\Pi$ and we may conclude from Corollary 1 that there is an $\epsilon_{0}^{\prime}>0$ and $b^{\prime}>0$ such that, for all $\epsilon\in(0,\epsilon_{0}^{\prime})$ , the $\Pi$ -invariant M-preconditioned MALA chain $(x_{i})_{i\in\mathbb{N}}$ is V-uniformly ergodic for $V(x)=C_{2}\exp\left(\frac{b^{\prime}}{16}\|x-x_{0}\|^{2}\right)$ (since if a Markov chain is V-uniformly ergodic, then it is also CV-uniformly ergodic).

Establish (R $^{+}$ 1-3) The aim is now to establish conditions (R $^{+}$ 1-3) of Theorem 3 for $Q = \Pi$ . By construction $\mathrm{d}P/\mathrm{d}\Pi = C_2/\sqrt{k_P(x)} < C_2/C_1 < \infty$ , where $C_1$ and $C_2$ were defined in Assumption 1, so that $P \ll \Pi$ . It has already been established that $(x_i)_{i \in \mathbb{N}}$ is $V$ -uniformly ergodic, and further

$$
V (x) = C _ {2} \exp \left(\frac {b ^ {\prime}}{1 6} \| x - x _ {0} \| ^ {2}\right) \geq C _ {2} = \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x) \sqrt {k _ {P} (x)}
$$

for all x, which establishes $(\mathbb{R}^{+}1)$ . Let R and $\rho$ denote constants for which the V-uniform ergodicity property (14) is satisfied. From V-uniform ergodicity, the integral $\int V \, d\Pi$ exists and

$$
\begin{array}{l} \left| \mathbb {E} \left[ \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x _ {i}) \sqrt {k _ {P} (x _ {i})} V (x _ {i}) \right] - C _ {2} \int V \mathrm{d} \Pi \right| = C _ {2} \left| \mathbb {E} [ V (x _ {i}) ] - \int V \mathrm{d} \Pi \right| \\ \leq C _ {2} R \rho^ {n} V (x _ {0}) \rightarrow 0 \\ \end{array}
$$

which establishes $(\mathrm{R}^{+}2)$ . Fix $\gamma > 0$ . By construction $dP/d\Pi \leq C_{2}/C_{1}$ , and thus

$$
\exp \left\{\gamma \max \left(1, \frac {\mathrm{d} P}{\mathrm{d} \Pi} (x) ^ {2}\right) k _ {P} (x) \right\} <   \exp \left\{\tilde {\gamma} k _ {P} (x) \right\}
$$

where $\tilde{\gamma} = \max (1,C_2 / C_1)\gamma$ . Since we have assumed that $k_{P}$ is continuous with, from (A4),

$$
b _ {3} := \operatorname * {l i m s u p} _ {\| x \| \to \infty} \frac {k _ {P} (x)}{\| x \| ^ {2}} <   \infty ,
$$

we may take $\gamma$ such that $\tilde{\gamma} b_{3} < b' / 16$ , so that $\| x \mapsto \exp \{\tilde{\gamma} k_{P}(x)\} \|_{V} < \infty$ and in particular

$$
\left| \mathbb {E} \left[ \exp \{\tilde {\gamma} k _ {P} (x _ {i}) \} \right] - \int \exp \{\tilde {\gamma} k _ {P} (x) \} \mathrm{d} \Pi (x) \right| \leq \| x \mapsto \exp \{\tilde {\gamma} k _ {P} (x) \} \| _ {V} \times R \rho^ {n} V (x _ {0}) \to 0
$$

which establishes $(\mathbb{R}^{+}3)$ . Thus we have established $(\mathbb{R}^{+}1-3)$ for $Q = \Pi$ , so from Theorem 3 we have strong consistency of SIIT-MALA (i.e. $D_{P}(P_{n,m}) \stackrel{\text{a.s.}}{\to} 0$ ) provided that $m \leq n$ with $\log(n) = O(m^{\gamma/2})$ for some $\gamma < 1$ . The latter condition is equivalent to $m = \Omega((\log n)^{\delta})$ for some $\delta > 2$ , which we used for the statement. Since $0 \leq D_{P}(P_{n}^{\star}) \leq D_{P}(P_{n,m})$ , the strong consistency of SIIS-MALA is also established.

# C Explicit Calculation of Stein Kernels

This appendix contains explicit calculations for the Langevin–Stein and KGM–Stein kernels $k_{P}$ , which are sufficient to implement Stein II-Importance Sampling and Stein II-Thinning. These calculations can also be performed using automatic differentiation, but comparison to the analytic expressions is an important step in validation of computer code.

To proceed, we observe that the diffusion Stein operator $S_P$ in (12) applied to a matrix-valued kernel $K$ is equivalent to the Langevin-Stein operator applied to the kernel $C(x,y) = M(x)K(x,y)M(y)^\top$ . In the case of the Langevin-Stein and KGM-Stein kernels we have $K(x,y) = \kappa(x,y)I$ for some $\kappa(x,y)$ and $M(x) = (1 + \|x - x_\star\|_{\Sigma}^2)^{(s - 1)/2}I$ for some $s \in \{0,1,2,\ldots\}$ . Thus $C(x,y) = c(x,y)I$ where

$$
c (x, y) := \left(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {(s - 1) / 2} \left(1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {(s - 1) / 2} \kappa (x, y)
$$

and

$$
\begin{array}{l} k _ {P} (x, y) = \nabla_ {x} \cdot \nabla_ {y} c (x, y) + [ \nabla_ {x} c (x, y) ] \cdot [ \nabla_ {y} \log p (y) ] + [ \nabla_ {y} c (x, y) ] \cdot [ \nabla_ {x} \log p (x) ] \\ + c (x, y) [ \nabla_ {x} \log p (x) ] \cdot [ \nabla_ {y} \log p (y) ], \\ \end{array}
$$

following the calculations in Oates et al. (2017). To evaluate the terms in this formula we start by differentiating $c(x,y)$ , to obtain

$$
\begin{array}{l} \nabla_ {x} c (x, y) = (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {(s - 1) / 2} (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {(s - 1) / 2} \\ \times \left[ \frac {(s - 1) \kappa (x , y) \Sigma^ {- 1} (x - x _ {\star})}{1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}} + \nabla_ {x} \kappa (x, y) \right] \\ \end{array}
$$

$$
\nabla_ {y} c (x, y) = (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {(s - 1) / 2} (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {(s - 1) / 2}
$$

$$
\times \left[ \frac {(s - 1) \kappa (x , y) \Sigma^ {- 1} (y - x _ {\star})}{1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}} + \nabla_ {y} \kappa (x, y) \right]
$$

$$
\nabla_ {x} \cdot \nabla_ {y} c (x, y) = (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {(s - 1) / 2} (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {(s - 1) / 2}
$$

$$
\times \left[ \frac {(s - 1) ^ {2} \kappa (x , y) (x - x _ {\star}) ^ {\top} \Sigma^ {- 2} (y - x _ {\star})}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2})} + \frac {(s - 1) (y - x _ {\star}) ^ {\top} \Sigma^ {- 1} \nabla_ {x} \kappa (x , y)}{(1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2})} \right.
$$

$$
\left. + \frac {(s - 1) (x - x _ {\star}) ^ {\top} \Sigma^ {- 1} \nabla_ {y} \kappa (x , y)}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2})} + \nabla_ {x} \cdot \nabla_ {y} \kappa (x, y) \right].
$$

These expressions involve gradients of $\kappa(x, y)$ , and explicit formulae for these are presented for the choice of $\kappa(x, y)$ corresponding to the Langevin–Stein kernel in Appendix C.1, and to the KGM–Stein kernel in Appendix C.2.

To implement Stein II-Thinning we require access to both $k_{P}(x)$ and $\nabla k_{P}(x)$ , the latter for use in the proposal distribution and acceptance probability in MALA. These quantities will now be calculated. In what follows we assume that $\kappa(x,y)$ is continuously differentiable, so that partial derivatives with respect to x and y can be interchanged. Then

$$
c _ {0} (x) := c (x, x)
$$

$$
= (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s - 1} \kappa (x, x)
$$

$$
c _ {1} (x) := \left. \nabla_ {x} c (x, y) \right| _ {y \to x}
$$

$$
= (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s - 1} \left[ \frac {(s - 1) \kappa (x , x) \Sigma^ {- 1} (x - x _ {\star})}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2})} + \nabla_ {x} \kappa (x, y) | _ {y \rightarrow x} \right]
$$

$$
c _ {2} (x) := \left. \nabla_ {x} \cdot \nabla_ {y} c (x, y) \right| _ {y \rightarrow x}
$$

$$
= (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s - 1} \left[ \frac {(s - 1) ^ {2} \kappa (x , x) (x - x _ {\star}) ^ {\top} \Sigma^ {- 2} (x - x _ {\star})}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {2}} \right.
$$

$$
\left. + \frac {2 (s - 1) (x - x _ {\star}) ^ {\top} \Sigma^ {- 1} \left. \nabla_ {x} \kappa (x , y) \right| _ {y \to x}}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2})} + \left. \nabla_ {x} \cdot \nabla_ {y} \kappa (x, y) \right| _ {y \to x} \right]
$$

so that

$$
k _ {P} (x) := k _ {P} (x, x) = c _ {2} (x) + 2 c _ {1} (x) \cdot \nabla_ {x} \log p (x) + c _ {0} (x) \| \nabla_ {x} \log p (x) \| ^ {2}. \tag {19}
$$

Let $[\nabla_x c_1(x)]_{i,j} = \partial_{x_i}[c_1(x)]_j$ and $[\nabla_x^2\log p(x)]_{i,j} = \partial_{x_i}\partial_{x_j}\log p(x)$ . Now we can differentiate (19) to get

$$
\nabla_ {x} k _ {P} (x) = \nabla_ {x} c _ {2} (x) + 2 [ \nabla_ {x} c _ {1} (x) ] [ \nabla_ {x} \log p (x) ] + 2 [ \nabla_ {x} ^ {2} \log p (x) ] c _ {1} (x)
$$

$$
+ \left[ \nabla_ {x} c _ {0} (x) \right] \| \nabla_ {x} \log p (x) \| ^ {2} + 2 c _ {0} (x) \left[ \nabla_ {x} ^ {2} \log p (x) \right] \left[ \nabla_ {x} \log p (x) \right]. \tag {20}
$$

In what follows we also derive explicit formulae for $c_{0}(x)$ , $c_{1}(x)$ and $c_{2}(x)$ , and hence for $\nabla_{x}c_{0}(x)$ , $\nabla_{x}c_{1}(x)$ and $\nabla_{x}c_{2}(x)$ , for the case of the Langevin–Stein kernel in Appendix C.1, and the KGM–Stein kernel in Appendix C.2.

# C.1 Explicit Formulae for the Langevin-Stein Kernel

The Langevin–Stein kernel from Appendix A.3.1 corresponds to the choice s = 1 and $\kappa(x, y)$ the inverse multi-quadric kernel, so that

$$
\kappa (x, y) = (1 + \| x - y \| _ {\Sigma} ^ {2}) ^ {- \beta}
$$

$$
\nabla_ {x} \kappa (x, y) = - 2 \beta (1 + \| x - y \| _ {\Sigma} ^ {2}) ^ {- \beta - 1} \Sigma^ {- 1} (x - y)
$$

$$
\nabla_ {y} \kappa (x, y) = 2 \beta (1 + \| x - y \| _ {\Sigma} ^ {2}) ^ {- \beta - 1} \Sigma^ {- 1} (x - y)
$$

$$
\nabla_ {x} \cdot \nabla_ {y} \kappa (x, y) = - 4 \beta (\beta + 1) (1 + \| x - y \| _ {\Sigma} ^ {2}) ^ {- \beta - 2} (x - y) ^ {\top} \Sigma^ {- 2} (x - y)
$$

$$
+ 2 \beta \mathrm{tr} (\Sigma^ {- 1}) (1 + \| x - y \| _ {\Sigma} ^ {2}) ^ {- \beta - 1}.
$$

Evaluating on the diagonal:

$$
\kappa (x, x) = 1
$$

$$
\nabla_ {x} \kappa (x, y) | _ {y \to x} = \left. \nabla_ {y} \kappa (x, y) \right| _ {y \to x} = 0
$$

$$
\nabla_ {x} \cdot \nabla_ {y} \kappa (x, y) | _ {y \to x} = 2 \beta \mathrm{tr} (\Sigma^ {- 1}),
$$

so that $c_{0}(x) = 1$ , $c_{1}(x) = 0$ , $c_{2}(x) = 2\beta \mathrm{tr}(\Sigma^{-1})$ . Differentiating these formulae, $\nabla_{x}c_{0}(x) = 0$ , $\nabla_{x}c_{1}(x) = 0$ , $\nabla_{x}c_{2}(x) = 0$ .

# C.2 Explicit Formulae for the KGM-Stein Kernel

The KGM kernel of order $s$ from Appendix A.3.2 corresponds to the choice

$$
\kappa (x, y) = \left(1 + \| x - y \| _ {\Sigma} ^ {2}\right) ^ {- \beta} + \frac {1 + \left(x - x _ {\star}\right) ^ {\top} \Sigma^ {- 1} \left(y - x _ {\star}\right)}{\left(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {s / 2} \left(1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {s / 2}},
$$

for which we have

$$
\begin{array}{l} \nabla_ {x} \kappa (x, y) = - 2 \beta (1 + \| x - y \| _ {\Sigma} ^ {2}) ^ {- \beta - 1} \Sigma^ {- 1} (x - y) \\ + \frac {\Sigma^ {- 1} (y - x _ {\star}) - s [ 1 + (x - x _ {\star}) ^ {\top} \Sigma^ {- 1} (y - x _ {\star}) ] \Sigma^ {- 1} (x - x _ {\star}) (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- 1}}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s / 2} (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s / 2}} \\ \end{array}
$$

$$
\nabla_ {y} \kappa (x, y) = 2 \beta (1 + \| x - y \| _ {\Sigma} ^ {2}) ^ {- \beta - 1} \Sigma^ {- 1} (x - y)
$$

$$
+ \frac {\Sigma^ {- 1} (x - x _ {\star}) - s [ 1 + (x - x _ {\star}) ^ {\top} \Sigma^ {- 1} (y - x _ {\star}) ] \Sigma^ {- 1} (y - x _ {\star}) (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- 1}}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s / 2} (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s / 2}}
$$

$$
\nabla_ {x} \cdot \nabla_ {y} \kappa (x, y) = - 4 \beta (\beta + 1) (1 + \| x - y \| _ {\Sigma} ^ {2}) ^ {- \beta - 2} (x - y) ^ {\top} \Sigma^ {- 2} (x - y) + 2 \beta \mathbf {t r} (\Sigma^ {- 1}) (1 + \| x - y \| _ {\Sigma} ^ {2}) ^ {- \beta - 1}
$$

$$
+ \frac {\left[ \begin{array}{c} \mathsf {t r} (\Sigma^ {- 1}) - s (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- 1} (x - x _ {\star}) ^ {\top} \Sigma^ {- 2} (x - x _ {\star}) \\ - s (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- 1} (y - x _ {\star}) ^ {\top} \Sigma^ {- 2} (y - x _ {\star}) \\ + s ^ {2} [ 1 + (x - x _ {\star}) ^ {\top} \Sigma^ {- 1} (y - x _ {\star}) ] (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- 1} (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- 1} \\ \times (x - x _ {\star}) ^ {\top} \Sigma^ {- 2} (y - x _ {\star}) \end{array} \right]}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s / 2} (1 + \| y - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s / 2}}.
$$

Evaluating on the diagonal:

$$
\kappa (x, x) = 1 + \left(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {- s + 1}
$$

$$
\nabla_ {x} \kappa (x, y) | _ {y \to x} = \left. \nabla_ {y} \kappa (x, y) \right| _ {y \to x} = - (s - 1) \Sigma^ {- 1} (x - x _ {\star}) (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- s}
$$

$$
\nabla_ {x} \cdot \nabla_ {y} \kappa (x, y) | _ {y \to x} = 2 \beta \mathrm{tr} (\Sigma^ {- 1}) + \mathrm{tr} (\Sigma^ {- 1}) (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- s}
$$

$$
+ s (s - 2) \left(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {- s - 1} \left(x - x _ {\star}\right) ^ {\top} \Sigma^ {- 2} \left(x - x _ {\star}\right)
$$

so that

$$
c _ {0} (x) = 1 + \left(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {s - 1}
$$

$$
c _ {1} (x) = (s - 1) \left(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {s - 2} \Sigma^ {- 1} \left(x - x _ {\star}\right)
$$

$$
c _ {2} (x) = \frac {[ (s - 1) ^ {2} (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s - 1} - 1 ] (x - x _ {\star}) ^ {\top} \Sigma^ {- 2} (x - x _ {\star})}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {2}} + \frac {\mathrm{tr} (\Sigma^ {- 1}) [ 1 + 2 \beta (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s} ]}{(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2})}.
$$

Differentiating these formulae:

$$
\begin{array}{l} \nabla_ {x} c _ {0} (x) = 2 (s - 1) (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s - 2} \Sigma^ {- 1} (x - x _ {\star}) \\ \nabla_ {x} c _ {1} (x) = 2 (s - 1) (s - 2) (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s - 3} [ \Sigma^ {- 1} (x - x _ {\star}) ] [ \Sigma^ {- 1} (x - x _ {\star}) ] ^ {\top} \\ + (s - 1) \left(1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}\right) ^ {s - 2} \Sigma^ {- 1} \\ \nabla_ {x} c _ {2} (x) = 2 (s - 1) ^ {2} (s - 3) (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s - 4} [ (x - x _ {\star}) ^ {\top} \Sigma^ {- 2} (x - x _ {\star}) ] \Sigma^ {- 1} (x - x _ {\star}) \\ + 2 (s - 1) ^ {2} (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s - 3} \Sigma^ {- 2} (x - x _ {\star}) \\ + 4 \beta \mathrm{tr} (\Sigma^ {- 1}) (s - 1) (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {s - 2} \Sigma^ {- 1} (x - x _ {\star}) \\ - 2 (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- 2} [ \Sigma^ {- 2} (x - x _ {\star}) + \mathbf {t r} (\Sigma^ {- 1}) \Sigma^ {- 1} (x - x _ {\star}) ] \\ + 4 (1 + \| x - x _ {\star} \| _ {\Sigma} ^ {2}) ^ {- 3} [ (x - x _ {\star}) ^ {\top} \Sigma^ {- 2} (x - x _ {\star}) ] \Sigma^ {- 1} (x - x _ {\star}). \\ \end{array}
$$

These complete the analytic calculations necessary to compute the Stein kernel $k_{P}$ and its gradient.

# D Empirical Assessment

This appendix contains full details of the empirical protocols that were employed and the additional empirical results described in the main text. Appendix D.1 discusses the effect of dimension on our proposed II. Additional illustrative results from Section 3.2 are contained in Appendix D.2. The full details for how MALA was implemented are contained in Appendix D.3. An additional illustration using a generalised auto-regressive moving average (GARCH) model is presented in Appendix D.4. The full results for SIIIS-MALA are contained in Appendix D.5, and in Appendix D.6 the convergence of the sparse approximation provided by SIIT-MALA to the optimal weighted approximation is investigated. Finally, the performance of KSDs is quantified using the 1-Wasserstein divergence in Appendix D.7.

# D.1 The Effect of Dimension on $\Pi$

The improvement of Stein $\Pi$ -Importance Sampling over the default Stein importance sampling algorithm (i.e. $\Pi = P$ ) can be expected to reduce as the dimension $d$ of the target $P$ is increased. To see this, consider the Langevin–Stein kernel

$$
k _ {P} (x) = c _ {1} + c _ {2} \| \nabla \log p (x) \| _ {\Sigma} ^ {2} \tag {21}
$$

for some $c_{1}, c_{2} > 0$ ; see Appendix C. Taking $P = \mathcal{N}(0, I_{d \times d})$ , for which the length scale matrix $\Sigma$ appearing in Appendix A.3 is $\Sigma = I_{d \times d}$ , we obtain

$$
k _ {P} (x) = c _ {1} + c _ {2} \| x \| ^ {2}.
$$

However, the sampling distribution $\Pi$ defined in (8) depends on $k_{P}$ only up to an unspecified normalisation constant; we may therefore equally consider the asymptotic behaviour of $\tilde{k}_{P}(x):=k_{P}(x)/d$ . Let $X\sim P$ . Then $\mathbb{E}[\tilde{k}_{P}(X)]=c_{2}$ is a d-independent constant, and

$$
\left\| \tilde {k} _ {P} - \mathbb {E} [ \tilde {k} _ {P} (X) ] \right\| _ {L ^ {2} (P)} ^ {2} = \int \left[ \frac {k _ {P} (x) - (c _ {1} + c _ {2} d)}{d} \right] ^ {2} \mathrm{d} P (x) = \frac {2 c _ {2} ^ {2}}{d} \rightarrow 0
$$

as $d \to \infty$ . This shows that $\tilde{k}_{P}$ converges to a constant function in $L^{2}(P)$ , and thus for “typical” values of x in the effective support of P,

$$
\pi (x) \propto p (x) \sqrt {\tilde {k} _ {P} (x)} \stackrel {\approx} {\propto} p (x),
$$

so that $\Pi \approx P$ in the $d\to \infty$ limit. This intuition is borne out in simulations involving both the Langevin-Stein kernel (as just discussed) and also the KGM3-Stein kernel. Indeed, Figure S1 shows that as the dimension $d$ is increased, the marginal distributions of $\Pi$ become increasingly similar to those of $P$ .

![](images/c5c93e678fc223217d4611f6b53a4a564624fb30587b4e848e929d7c8dd55b5f.jpg)

<details>
<summary>line</summary>

| x1   | P     | II (d = 1) | II (d = 2) | II (d = 10) |
|------|-------|------------|------------|-------------|
| -3.0 | 0.000 | 0.000      | 0.000      | 0.000       |
| -2.5 | 0.050 | 0.045      | 0.048      | 0.047       |
| -2.0 | 0.150 | 0.145      | 0.148      | 0.147       |
| -1.5 | 0.280 | 0.275      | 0.278      | 0.277       |
| -1.0 | 0.350 | 0.345      | 0.348      | 0.347       |
| -0.5 | 0.390 | 0.385      | 0.388      | 0.387       |
| 0.0  | 0.400 | 0.395      | 0.398      | 0.397       |
| 0.5  | 0.395 | 0.390      | 0.393      | 0.392       |
| 1.0  | 0.350 | 0.345      | 0.348      | 0.347       |
| 1.5  | 0.280 | 0.275      | 0.278      | 0.277       |
| 2.0  | 0.150 | 0.145      | 0.148      | 0.147       |
| 2.5  | 0.050 | 0.045      | 0.048      | 0.047       |
| 3.0  | 0.000 | 0.000      | 0.000      | 0.000       |
</details>

(S1.1)

![](images/e7ce048c55c274c719927f070b1ad664a08509e9da3121371a7ed2b46f495972.jpg)

<details>
<summary>line</summary>

| x1   | P     | II (d = 1) | II (d = 2) | II (d = 10) |
|------|-------|------------|------------|-------------|
| -3.0 | 0.000 | 0.050      | 0.025      | 0.010       |
| -2.0 | 0.100 | 0.150      | 0.125      | 0.100       |
| -1.0 | 0.300 | 0.200      | 0.225      | 0.250       |
| 0.0  | 0.400 | 0.200      | 0.250      | 0.350       |
| 1.0  | 0.300 | 0.200      | 0.225      | 0.250       |
| 2.0  | 0.100 | 0.150      | 0.125      | 0.100       |
| 3.0  | 0.000 | 0.050      | 0.025      | 0.010       |
</details>

(S1.2)   
Figure S1: The effect of dimension on $\Pi$ : Here $P$ was taken to be the standard Gaussian distribution $\mathcal{N}(0, I_{d \times d})$ in $\mathbb{R}^d$ and the proposed distribution $\Pi$ was computed. The marginal distribution of the first component of $\Pi$ is plotted for $d \in \{1, 2, 10\}$ , for both (a) the Langevin-Stein kernel and (b) the KGM3-Stein kernel.

![](images/7b46dff043fe293c3c89567e2ad7a3acd9870a60866ced45051c3f3ef431bd42.jpg)

<details>
<summary>line</summary>

| n     | P (Langevin) | Π (Langevin) | P (KGM3) | Π (KGM3) | P (Riemann) | Π (Riemann) |
|-------|--------------|--------------|----------|----------|-------------|-------------|
| 10^1  | ~1.0         | ~1.0         | ~2.0     | ~2.0     | ~0.5        | ~0.5        |
| 10^2  | ~0.1         | ~0.1         | ~0.5     | ~0.5     | ~0.05       | ~0.05       |
| 10^3  | ~0.01        | ~0.01        | ~0.1     | ~0.1     | ~0.005      | ~0.005      |
</details>

Figure S2: Assessing the performance of the sampling distributions $\Pi$ shown in Figure 2. The mean kernel Stein discrepancy (KSD) for computation performed using the Langevin–Stein kernel (purple), the KGM3–Stein kernel (blue), and the Riemann–Stein kernel (red); in each case, KSD was computed using the same Stein kernel used to construct $\Pi$ . Solid lines indicate the baseline case of sampling from P, while dashed lines indicate the proposed approach of sampling from $\Pi$ . (The experiment was repeated 10 times and standard error bars are plotted.)

# D.2 2D Illustration from the Main Text

Section 3.2 of the main text contained a 2-dimensional illustration of Stein $\Pi$ -Importance Sampling and presented the distributions $\Pi$ corresponding to different choices of Stein kernel. Here, in Figure S2, we present the mean KSDs for Stein $\Pi$ -Importance Sampling performed using the Langevin-Stein kernel (purple), the KGM3-Stein kernel (blue), and the Riemann-Stein kernel (red), corresponding to the sampling distributions $\Pi$ displayed in Figure 2 of the main text.

For this experiment, exact sampling from both P and $\Pi$ was performed using a fine grid on which all probabilities were calculated and appropriately normalised. Results are in broad agreement with the 1-dimensional illustration contained in the main text, in the sense that in all cases Stein $\Pi$ -Importance Sampling provides a significant improvement over the default Stein importance sampling method with $\Pi$ equal to P.

Algorithm 4 Adaptive MALA   
Require: $x_{0,0}$ (initial state), $\epsilon_{0}$ (initial step size), $M_{0}$ (initial preconditioner matrix), $\{n_{i}\}_{i=0}^{h-1}$ (epoch lengths), $\{\alpha_{i}\}_{i=1}^{h-1}$ (learning schedule), h (number of epochs), $k_{P}$ (Stein kernel)

1: $\{x_{0,1}\ldots,x_{0,n_{0}}\}\leftarrow\text{MALA}(x_{0,0},\epsilon_{0},M_{0},n_{0},k_{P})$ 2: for $i=1,\ldots,h-1$ do

3: $x_{i,0}\leftarrow x_{i-1,n_{i-1}}$ 4: $\rho_{i-1}\leftarrow\frac{1}{n_{i-1}}\sum_{j=1}^{n_{i-1}}1_{x_{i-1,j}\neq x_{i-1,j-1}}$ $\triangleright$ Average acceptance rate for chain i

5: $\epsilon_{i}\leftarrow\epsilon_{i-1}\exp(\rho_{i-1}-0.57)$ $\triangleright$ Update step size

6: $M_{i}\leftarrow\alpha_{i}M_{i}+(1-\alpha_{i})\text{cov}(\{x_{i-1,1}\ldots,x_{i-1,n_{i-1}}\})$ $\triangleright$ Update preconditioner matrix

7: $\{x_{i,1}\ldots,x_{i,n_{i}}\}\leftarrow\text{MALA}(x_{i,0},\epsilon_{i},M_{i},n_{i},k_{P})$ 8: end for

# D.3 Implementation of MALA

For implementation of MALA in Algorithm 4 we are required to specify a step size $\epsilon$ and a preconditioner matrix M. In general, suitable values for both of these parameters will be problem-dependent. Standard practice is to perform some form of manual or automated tuning to arrive at parameter values for which the average acceptance rate is close to 0.57, motivated by the asymptotic analysis of Roberts and Rosenthal (1998). Adaptive MCMC algorithms, which seek to optimise the parameters of MCMC algorithms such as MALA during the warm-up period, provide an appealing solution, and was the approach taken in this work.

The adaptive MALA algorithm which we used is contained in Algorithm 4, where we have let $\mathrm{MALA}(x,\epsilon,M,n,k_{P})$ denote the output from the preconditioned MALA with initial state x, step size $\epsilon$ , preconditioner matrix M, and chain length n, described in Algorithm 1. In Algorithm 4, we use $\mathrm{cov}(\cdot)$ to denote the sample covariance matrix. The algorithm monitors the average acceptance rate and increases or decreases it according to whether it is below or above, respectively, the 0.57 target. For the preconditioner matrix, the sample covariance matrix of samples obtained from the penultimate tuning run of MALA is used. For all experiments that we report using MALA, we set $\epsilon_{0}=1$ , $M_{0}=I_{d}$ , h=10, and $\alpha_{1}=\cdots=\alpha_{9}=0.3$ . The warm-up epoch lengths were $n_{0}=\cdots=n_{8}=1,000$ and the final epoch length was $n_{9}=10^{5}$ . The samples $\{x_{h-1,1},\ldots,x_{h-1,n_{i-1}}\}$ from the final epoch are returned, and constituted output from MALA for our experimental assessment.

To sample from $P$ instead of $\Pi$ , we used Algorithm 4 we formally set $k_{P}(x) = 1$ for all $x \in \mathbb{R}^{d}$ , which recovers $\Pi = P$ as the target.

# D.4 Illustration on a GARCH Model

This appendix contains an additional illustrative experiment, concerning a GARCH model that is a particular instance of a model from the PosteriorDB database discussed in Section 4. The purpose of this illustration is to facilitate an empirical investigation in a slightly higher dimension $(d = 4)$ and to explore the effect of changing the order s of the KGM–Stein kernel defined in Appendix A.3.2.

First we describe the GARCH model that was used. These models are widely-used in econometrics to describe time series data $\{y_{t}\}_{t=1}^{n}$ in settings where the volatility process is assumed to be time-varying (but stationary). In particular, we consider the GARCH(1,1) model

$$
y _ {t} = \phi_ {1} + a _ {t},
$$

$$
a _ {t} = \sigma_ {t} \epsilon_ {t}, \quad \epsilon_ {t} \sim \mathcal {N} (0, 1),
$$

$$
\sigma_ {t} ^ {2} = \phi_ {2} + \phi_ {3} a _ {t - 1} ^ {2} + \phi_ {4} \sigma_ {t - 1} ^ {2},
$$

where $\phi_{2}>0,\phi_{3}>0,\phi_{4}>0$ , and $\phi_{3}+\phi_{4}<1$ are the model parameters, constrained to a subset of $R^{4}$ . For ease of sampling, a change of variables $\tau:(\phi_{1},\phi_{2},\phi_{3},\phi_{4})\mapsto\theta$ is performed in such a way that the parameter $\theta\in R^{4}$ is unconstrained. Assuming an improper flat prior on $\theta$ , the log-posterior density for $\theta$ is given up to an additive constant by

$$
\log p (\theta \mid y _ {1}, \dots , y _ {n}) \stackrel {+ C} {=} \sum_ {t = 1} ^ {n} \left[ - \frac {1}{2} \log \bigl (\sigma_ {t} ^ {2} \bigr) - \frac {y _ {t} ^ {2}}{2 \sigma_ {t} ^ {2}} \right] + \log | J _ {\tau^ {- 1}} (\theta) |,
$$

where $|J_{\tau^{-1}}(\theta)|$ is the Jacobian determinant of $\tau^{-1}$ .

![](images/41887b70b59b70be696aeb04659a8ce5b9474f6cf78629e19ab482cfc58c69e8.jpg)

<details>
<summary>line</summary>

| θ₁   | P     | Π (KGM2) | Π (KGM3) | Π (KGM4) |
|------|-------|----------|----------|----------|
| 4.6  | 0.0   | 0.0      | 0.0      | 0.0      |
| 4.8  | 1.0   | 1.0      | 1.0      | 1.0      |
| 5.0  | 3.3   | 2.5      | 2.4      | 2.4      |
| 5.2  | 2.0   | 2.4      | 2.3      | 2.3      |
| 5.4  | 0.5   | 0.5      | 0.5      | 0.5      |
| 5.6  | 0.0   | 0.0      | 0.0      | 0.0      |
</details>

![](images/21e636f075fdb777ec585375f86b3e4d542b9dc0acba27007ba314d730316c01.jpg)

<details>
<summary>line</summary>

| θ₂    | Series 1 | Series 2 | Series 3 | Series 4 |
| ------ | -------- | -------- | -------- | -------- |
| -1.5   | 0.0      | 0.0      | 0.0      | 0.0      |
| -1.0   | 0.0      | 0.0      | 0.0      | 0.0      |
| -0.5   | 0.2      | 0.3      | 0.4      | 0.5      |
| 0.0    | 0.8      | 0.9      | 1.0      | 0.9      |
| 0.5    | 0.7      | 0.8      | 0.9      | 0.8      |
| 1.0    | 0.3      | 0.4      | 0.5      | 0.6      |
| 1.5    | 0.0      | 0.0      | 0.0      | 0.0      |
| 2.0    | 0.0      | 0.0      | 0.0      | 0.0      |
</details>

![](images/2050c8c49b6622f348e4fe784dc016dc76b56d711d3d03dc7fb185ac316aeb62.jpg)

<details>
<summary>line</summary>

| θ₃   | Density (Blue) | Density (Orange) | Density (Green) | Density (Red) |
|------|----------------|------------------|-----------------|---------------|
| -1.0 | 0.0            | 0.0              | 0.0             | 0.0           |
| -0.5 | 0.3            | 0.2              | 0.1             | 0.1           |
| 0.0  | 0.75           | 0.65             | 0.6             | 0.6           |
| 0.5  | 0.65           | 0.6              | 0.55            | 0.55          |
| 1.0  | 0.4            | 0.4              | 0.4             | 0.4           |
| 1.5  | 0.2            | 0.2              | 0.2             | 0.2           |
| 2.0  | 0.1            | 0.1              | 0.1             | 0.1           |
| 2.5  | 0.05           | 0.05             | 0.05            | 0.05          |
| 3.0  | 0.0            | 0.0              | 0.0             | 0.0           |
</details>

![](images/01d0651aa6f4a69699b3b1cc1fb0f7a5f90ed4f3933eb8dcc206eae4e0ce9b40.jpg)

<details>
<summary>line</summary>

| θ₄   | Blue Line | Orange Line | Green Line | Red Line |
|------|-----------|-------------|------------|----------|
| -5   | 0.00      | 0.00        | 0.00       | 0.00     |
| 0    | 0.37      | 0.30        | 0.20       | 0.12     |
| 5    | 0.15      | 0.12        | 0.10       | 0.14     |
| 10   | 0.00      | 0.00        | 0.00       | 0.00     |
</details>

Figure S3: Illustrating the shape of $\Pi$ based on the KGMs–Stein kernel for a GARCH(1,1) model, controlling convergence of moments up to order $s \in \{2, 3, 4\}$ . The marginal density functions of each distribution were approximated using one-million samples obtained using MCMC.

For this illustration, real data were provided within the model description of PosteriorDB, for which the estimated maximum a posteriori parameter is $\hat{\phi} = (5.04, 1.36, 0.53, 0.31)$ . The marginal distributions of $\Pi$ corresponding to the KGM–Stein kernels of orders $s \in \{2, 3, 4\}$ are compared to the marginals of P in Figure S3. It can be seen that higher orders s correspond to greater overdispersion of $\Pi$ ; this makes intuitive sense since larger s corresponds to a more stringent KSD (controlling the convergence of moments up to order s) which places greater emphasis on how the tails of P are approximated. Further, for the final skewed marginal of P, we note that the distribution $\Pi$ exaggerates the skew, placing more of its mass in the tail of the direction which is positively skewed. Further discussion of skewed targets is contained in Appendix D.8.

# D.5 Stein $\Pi$ -Importance Sampling for PosteriorDB

To introduce objectivity into our assessment, we exploited the PosteriorDB benchmark (Magnusson et al., 2022). This ongoing project is an attempt toward standardised benchmarking, consisting of a collection of posteriors to be numerically approximated. The test problems in PosteriorDB are defined in the Stan probabilistic programming language, and so BridgeStan (Roualdes et al., 2023) was used to directly access posterior densities and their gradients as required. The ambition of PosteriorDB is to provide an extensive set of benchmark tasks; at the time we conducted our research, PosteriorDB was at Version 0.4.0 and contained 149 models, of which 47 came equipped with a gold-standard sample of size $n = 10^{3}$ , generated from a long run of Hamiltonian Monte Carlo (the No-U-Turn sampler in Stan). Of these 47 models, a subset of 40 were found to be compatible with BridgeStan, which was at Version 1.0.2 at the time this research was performed. The version of Stan that we used was Stanc3 Version 2.31.0 (Unix). Thus we used a total of 40 test problems for our empirical assessment.

For each test problem, a total of 10 replicate experiments were performed and standard errors were computed. A sampling method was defined as being significantly better for approximation of a given target, compared to all other methods considered, if had lower mean KSD and the standard error bar did not overlap with the standard error bar of any other method. Table 1 in the main text summarises the performance of SIIS-MALA, fixing the number of samples to be $n = 3 \times 10^{3}$ . In this appendix, full empirical results are provided.

For sampling from MALA, we used the adaptive algorithm described in Appendix D.3 with a final epoch of length $n_{max} = 10^{5}$ . Then, whenever a set of $n \ll n_{max}$ consecutive samples from MALA are required for our experimental assessment, these were obtained by selecting at random a consecutive sequence of length n from the total chain of length $10^{5}$ . This ensures that the performance of unprocessed MALA that we report is not negatively affected by burn-in, in so far as is practical to control.

Full results are presented in Figure S4. These results broadly support the interpretation that SIIS-MALA usually outperforms SIS-MALA, or otherwise both methods provide a similar level of performance, for the sufficiently large sample sizes n considered. The sample size threshold at which SIIS-MALA outperforms SIS-MALA appears to be dimension-dependent. A notable exception is panel 29 of Figure S4, a d = 10 dimensional task for which SIIS-MALA provided a substantially worse approximation in KSD for the range of values of n considered.

Figure S4: Benchmarking on PosteriorDB. Here we compared raw output from MALA (dotted lines) with the post-processed output provided by the default Stein importance sampling method of Liu and Lee (2017) (SIS-MALA; solid lines) and the proposed Stein II-Importance Sampling method (SIIIS-MALA; dashed lines). The Langevin (purple) and KGM3–Stein kernels (blue) were used for SIS-MALA and SIIIS-MALA and the associated KSDs are reported as the number n of iterations of MALA is varied. Ten replicates were computed and standard errors were plotted. The name of each model is shown in the title of the corresponding panel, and the dimension d of the parameter vector is given in parentheses. [Langevin–Stein kernel: ……MALA, ——SIS-MALA, ---- SIIIS-MALA. KGM3–Stein kernel: ……MALA, ——SIS-MALA, ---- SIIIS-MALA.]   
![](images/c2dbd007c099f440d64e9bc28eeeed24faa6d4db42b2588aec8113508bae89fc.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed blue) | E[KSD] (dotted blue) | E[KSD] (dash-dot purple) |
|-------|---------------------|----------------------|----------------------|--------------------------|
| 10^1  | ~10^2               | ~10^2                | ~10^2                | ~10^1                    |
| 10^2  | ~10^1               | ~10^1                | ~10^1                | ~10^0                    |
| 10^3  | ~10^0               | ~10^0                | ~10^0                | ~10^-1                   |
| 10^4  | ~10^-1              | ~10^-1               | ~10^-1               | ~10^-2                   |
</details>

(S4.1)

![](images/ef86c0ef3b9baaa6aaef229c04d301320c046d64e4a7971b47a707b4f844d2e0.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (Solid Line) | E[KSD] (Dashed Line) | E[KSD] (Dotted Line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~10^1               | ~10^0                | ~10^0                |
| 10^2  | ~10^0               | ~10^-1               | ~10^-1               |
| 10^3  | ~10^-1              | ~10^-2               | ~10^-2               |
</details>

(S4.2)

![](images/1471093e57bfaa901145257dbbb7c8719aad9bbda846e12b3bad1c5968e11790.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid purple) | E[KSD] (dashed cyan) | E[KSD] (dotted light blue) |
|-------|------------------------|----------------------|----------------------------|
| 10^1  | ~10^1                  | ~10^2                | ~10^2                      |
| 10^2  | ~10^0.5                | ~10^1.5              | ~10^1.8                    |
| 10^3  | ~10^-0.5               | ~10^0.5              | ~10^0.8                    |
</details>

(S4.3)

![](images/a8026a64b42e6096d250d63fba5135f484e3fd52c5aa4f30409fbd6aa78dda95.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~10^2               | ~10^2                | ~10^2                |
| 10^2  | ~10^1.5             | ~10^1.5              | ~10^1.5              |
| 10^3  | ~10^0.5             | ~10^0.5              | ~10^0.5              |
| 10^4  | ~10^0               | ~10^0                | ~10^0                |
</details>

(S4.4)

![](images/42c409fa50bf0fa22e9e6e975bcd203255784e846c624750e8573a94f19cb2cf.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid purple) | E[KSD] (dashed purple) | E[KSD] (dotted purple) |
|-------|------------------------|-------------------------|-------------------------|
| 10^1  | ~10^1                  | ~10^1                   | ~10^2                   |
| 10^2  | ~10^0.5                | ~10^0.5                 | ~10^1.5                 |
| 10^3  | ~10^-0.5               | ~10^-0.5                | ~10^0.5                 |
| 10^4  | ~10^-1                 | ~10^-1                  | ~10^-0.5                |
</details>

(S4.5)

![](images/359eddd121fb92b1dcb9c985bcc8e82c644a2499d4aa48a655be9d5dd8c62fc1.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (Line 1) | E[KSD] (Line 2) | E[KSD] (Line 3) |
|-------|-----------------|-----------------|-----------------|
| 10^1  | ~10^2           | ~10^2           | ~10^1           |
| 10^2  | ~10^1           | ~10^1           | ~10^0           |
| 10^3  | ~10^0           | ~10^0           | ~10^-1          |
</details>

(S4.6)

![](images/8cca2b790a5cf751b54e4705e68890486b8d669639eb4d12fcfefff55d352bd5.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~10^2               | ~10^2                | ~10^2                |
| 10^2  | ~10^1.5             | ~10^1.5              | ~10^1.5              |
| 10^3  | ~10^1               | ~10^1                | ~10^1                |
| 10^4  | ~10^0.5             | ~10^0.5              | ~10^0.5              |
</details>

(S4.7)

![](images/c1aa0e16ec3e6e32d043a5e2b2bfcab018fdd03e3cc8895fb13a12a93c09322b.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (Line 1) | E[KSD] (Line 2) | E[KSD] (Line 3) | E[KSD] (Line 4) |
|-------|-----------------|-----------------|-----------------|-----------------|
| 10^1  | ~10^2           | ~10^1           | ~10^0           | ~10^-1          |
| 10^2  | ~10^1           | ~10^0           | ~10^-1          | ~10^-2          |
| 10^3  | ~10^0           | ~10^-1          | ~10^-2          | ~10^-3          |
</details>

(S4.8)

![](images/04bd1ca45582e8c51dbaad8b589072c889fa84c19c08418eef750c47c82586bf.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (Line 1) | E[KSD] (Line 2) | E[KSD] (Line 3) |
|-------|-----------------|-----------------|-----------------|
| 10^1  | ~10^2           | ~10^2           | ~10^1           |
| 10^2  | ~10^1.5         | ~10^1.5         | ~10^0.5         |
| 10^3  | ~10^1           | ~10^1           | ~10^0           |
</details>

(S4.9)

![](images/832a2be9ab4d9480edb6d60c9c49cafad3f8ca7eb013bb41b17850c5431f6911.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~10^2               | ~10^2                | ~10^2                |
| 10^2  | ~10^1.5             | ~10^1.8              | ~10^1.7              |
| 10^3  | ~10^1               | ~10^1.2              | ~10^1                |
</details>

(S4.10)

![](images/d96fdfde6c45629fa8dd6a689459055db5e939d2a6bd00ec87fc5d0e39f50347.jpg)

<details>
<summary>line</summary>

| n    | E[KSD] (solid purple) | E[KSD] (dashed purple) | E[KSD] (dotted blue) |
| ---- | --------------------- | ---------------------- | -------------------- |
| 10   | ~100                  | ~100                   | ~200                 |
| 100  | ~30                   | ~20                    | ~100                 |
| 1000 | ~10                   | ~5                     | ~30                  |
</details>

(S4.11)

![](images/0fbce579fccf7e4778106f229c5bcc87f35c5c1dbad4ca891494b36c668a3e05.jpg)

<details>
<summary>line</summary>

| n    | E[KSD] (solid blue) | E[KSD] (dashed blue) | E[KSD] (dotted purple) |
| ---- | ------------------- | -------------------- | ---------------------- |
| 10   | ~100                | ~100                 | ~20                    |
| 100  | ~30                 | ~40                  | ~10                    |
| 1000 | ~10                 | ~15                  | ~5                     |
</details>

(S4.12)

![](images/88b8888fac5750722173f413da8d671c1507e9675b8df0ea1a21725155ff56df.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed blue) | E[KSD] (dotted blue) | E[KSD] (dash-dot purple) |
|-------|---------------------|----------------------|----------------------|--------------------------|
| 10^1  | ~10^2               | ~10^2                | ~10^2                | ~10^1                    |
| 10^2  | ~10^1.5             | ~10^1.5              | ~10^1.5              | ~10^0.5                  |
| 10^3  | ~10^1               | ~10^1                | ~10^1                | ~10^0                    |
</details>

(S4.13)

![](images/d4a1323c29b86f3d1b779fda884e65ed265ff137be783c7e622e5edc8fc40cdc.jpg)

<details>
<summary>line</summary>

| n    | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
| ---- | ------------------- | -------------------- | -------------------- |
| 10   | ~10^2               | ~10^2                | ~10^1                |
| 100  | ~10^1               | ~10^1                | ~10^0                |
| 1000 | ~10^0               | ~10^0                | ~10^-1               |
</details>

(S4.14)

![](images/6e93d477adfa700a07aac71235d8f167fd7505f301710a8ba2c206cfb74f4ecd.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed blue) | E[KSD] (dotted blue) | E[KSD] (dash-dot purple) |
|-------|---------------------|----------------------|----------------------|--------------------------|
| 10^1  | ~10^2               | ~10^2                | ~10^2                | ~10^1                    |
| 10^2  | ~10^1               | ~10^1                | ~10^1                | ~10^0.5                  |
| 10^3  | ~10^0.5             | ~10^0.5              | ~10^0.5              | ~10^0                    |
</details>

(S4.15)

![](images/0376d09f4014d8a6bc71311b6d26075e80051a842f464496c4d25fd492427d5c.jpg)

<details>
<summary>line</summary>

| n    | E[KSD] (solid) | E[KSD] (dashed) | E[KSD] (dotted) |
| ---- | -------------- | --------------- | --------------- |
| 10^1 | ~300           | ~200            | ~150            |
| 10^2 | ~100           | ~50             | ~30             |
| 10^3 | ~30            | ~10             | ~5              |
</details>

(S4.16)

![](images/d545762d745e91ecf7f2ea7f516f62dc47ac708a870fbe2ca95be6721e012f4c.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed blue) | E[KSD] (solid purple) | E[KSD] (dotted purple) |
|-------|---------------------|----------------------|-----------------------|------------------------|
| 10^1  | ~10^3               | ~10^3                | ~10^2                 | ~10^2                  |
| 10^2  | ~10^2               | ~10^2                | ~10^1                 | ~10^1                  |
| 10^3  | ~10^1               | ~10^1                | ~10^0                 | ~10^0                  |
</details>

(S4.17)

![](images/628b7b95cdbd29a0e66dbb5a0b8538834908be5b69ad1254d8e6ca7861e94c1d.jpg)

<details>
<summary>line</summary>

| n    | E[KSD] (solid line) | E[KSD] (dashed line) |
| ---- | ------------------- | -------------------- |
| 10^1 | ~10^2               | ~10^1                |
| 10^2 | ~10^1               | ~10^0.5              |
| 10^3 | ~10^0.5             | ~10^0                |
</details>

(S4.18)

![](images/5733e947e4a16b53695cd58e5a57f6fc0c89ab8f876dc0460bea41195f137fa3.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (Solid Blue) | E[KSD] (Dashed Cyan) | E[KSD] (Dotted Purple) |
|-------|---------------------|----------------------|------------------------|
| 10^1  | ~10^4               | ~10^4                | ~10^3                  |
| 10^2  | ~10^3               | ~10^3                | ~10^2                  |
| 10^3  | ~10^2               | ~10^2                | ~10^1                  |
</details>

(S4.19)

![](images/a138df353bf849c369e9b2ed2c8e8ce8b8394657779cac651130c79c5a5c3cdd.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~10^4               | ~10^4                | ~10^3                |
| 10^2  | ~10^3               | ~10^3                | ~10^2                |
| 10^3  | ~10^2               | ~10^2                | ~10^1                |
</details>

(S4.20)

![](images/3c834f184c29d91e3e3b949facb852a30aded531420e52d8aae8a9cc9f57df8b.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~300                | ~250                 | ~100                 |
| 10^2  | ~150                | ~120                 | ~50                  |
| 10^3  | ~80                 | ~60                  | ~20                  |
| 10^4  | ~40                 | ~30                  | ~10                  |
</details>

(S4.21)

![](images/035d80d8ce4954ddad08e71710b74c15d9f1c3a16680832cb8db0d10fa42f52c.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid) | E[KSD] (dashed) | E[KSD] (dotted) |
|-------|----------------|-----------------|-----------------|
| 10^1  | ~10^2          | ~10^2           | ~10^2           |
| 10^2  | ~10^1          | ~10^1           | ~10^1           |
| 10^3  | ~10^0          | ~10^0           | ~10^0           |
</details>

(S4.22)

![](images/a9d2c2509db7f5151d7d0544f96ac4ba52f5149f0b7db5d7a6f85ea4a5caa751.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed light blue) | E[KSD] (dotted purple) |
|-------|---------------------|----------------------------|------------------------|
| 10^1  | ~10^2               | ~10^2                      | ~10^1                  |
| 10^2  | ~10^1.5             | ~10^1.5                    | ~10^0.5                |
| 10^3  | ~10^1               | ~10^1                      | ~10^0                  |
</details>

(S4.23)

![](images/eee6f6445ae72f44837e1e694fceb81b8219a0c60bed38172ca99b9504a5cb43.jpg)

<details>
<summary>line</summary>

| n     | E [KSD] (Line 1) | E [KSD] (Line 2) | E [KSD] (Line 3) | E [KSD] (Line 4) |
|-------|------------------|------------------|------------------|------------------|
| 10^1  | ~10^3            | ~10^2            | ~10^2            | ~10^1            |
| 10^2  | ~10^2            | ~10^1            | ~10^1            | ~10^0            |
| 10^3  | ~10^1            | ~10^0            | ~10^0            | ~10^-1           |
</details>

(S4.24)

![](images/114ff7ce94c7883a7dae3833e99d885c092d1a546d05d7b45ab275ba5460f35e.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~300                | ~200                 | ~150                 |
| 10^2  | ~150                | ~100                 | ~70                  |
| 10^3  | ~70                 | ~50                  | ~30                  |
</details>

(S4.25)

![](images/ef17439115f4a5f3f2e15d5204269f18ade251aa286c7b474d6effe5cf0cd358.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~10^2               | ~10^2                | ~10^2                |
| 10^2  | ~10^1.5             | ~10^1.5              | ~10^1.5              |
| 10^3  | ~10^0.5             | ~10^0.5              | ~10^0.5              |
</details>

(S4.26)

![](images/e357eabc1d572722a46ea39a8b411a91e6391f3713f6dcdeeba9b44c49b4a016.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed blue) | E[KSD] (dotted purple) |
|-------|---------------------|----------------------|------------------------|
| 10^1  | ~200                | ~300                 | ~150                   |
| 10^2  | ~100                | ~150                 | ~70                    |
| 10^3  | ~50                 | ~70                  | ~30                    |
</details>

(S4.27)

![](images/44b0eafa6d64d51fb89dc88c3e0fffe54dc97d1013ca02c9eb9749d8993edc9e.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed light blue) | E[KSD] (dotted purple) |
|-------|---------------------|----------------------------|------------------------|
| 10^1  | ~10^2               | ~10^2                      | ~10^1                  |
| 10^2  | ~10^1               | ~10^1                      | ~10^0                  |
| 10^3  | ~10^0               | ~10^0                      | ~10^-1                 |
</details>

(S4.28)

![](images/9e2ed23bf4cb7f478e20770a8bc65c77885c2b283cb8fe2d1ecb7fb4a505fe11.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (blue solid) | E[KSD] (blue dashed) | E[KSD] (cyan solid) | E[KSD] (cyan dashed) | E[KSD] (purple solid) | E[KSD] (purple dashed) |
|-------|---------------------|----------------------|---------------------|----------------------|-----------------------|------------------------|
| 100   | ~10^2               | ~10^3                | ~10^1               | ~10^1                | ~10^0                 | ~10^0                  |
| 1000  | ~10^1               | ~10^3                | ~10^0               | ~10^0                | ~10^-1                | ~10^-1                 |
</details>

(S4.29)

![](images/1037c3d0c81b51bc1180bc2d016b1856379ad9585c031b180a4af6f565e5db36.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~300                | ~250                 | ~10                  |
| 10^2  | ~100                | ~80                  | ~1                   |
| 10^3  | ~30                 | ~20                  | ~0.1                 |
</details>

(S4.30)

![](images/efaa214b7786fc9cf159960edc2985d770d7b2bc74e8debc02c65258ba788b62.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed blue) | E[KSD] (dotted blue) | E[KSD] (dash-dot purple) |
|-------|---------------------|----------------------|----------------------|--------------------------|
| 10^1  | ~10^3               | ~10^3                | ~10^3                | ~10^2                    |
| 10^2  | ~10^2               | ~10^2                | ~10^2                | ~10^1.5                  |
| 10^3  | ~10^1.5             | ~10^1.5              | ~10^1.5              | ~10^1                    |
| 10^4  | ~10^1               | ~10^1                | ~10^1                | ~10^0.5                  |
</details>

(S4.31)

![](images/1ef510c54dd65f68bb36e09d6f75e20785bcebc6d280d58cdf51e6507024188a.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (blue solid) | E[KSD] (blue dashed) | E[KSD] (purple solid) | E[KSD] (purple dashed) |
|-------|---------------------|----------------------|-----------------------|------------------------|
| 10^1  | ~10^3               | ~10^2                | ~10^2                 | ~10^1                  |
| 10^2  | ~10^2               | ~10^1                | ~10^1                 | ~10^0                  |
| 10^3  | ~10^1               | ~10^0                | ~10^0                 | ~10^-1                 |
</details>

(S4.32)

![](images/8148279bd7fd01b687590ee7f8ecbf644b20bc193b62f63177817416ec5fd498.jpg)

<details>
<summary>line</summary>

| n    | E[KSD] (solid blue) | E[KSD] (dashed blue) | E[KSD] (dotted blue) | E[KSD] (solid purple) | E[KSD] (dashed purple) |
|------|---------------------|----------------------|----------------------|-----------------------|------------------------|
| 10   | ~10^3               | ~10^3                | ~10^3                | ~10^2                 | ~10^2                  |
| 100  | ~10^2               | ~10^2                | ~10^2                | ~10^1                 | ~10^1                  |
| 1000 | ~10^1               | ~10^1                | ~10^1                | ~10^0                 | ~10^0                  |
</details>

(S4.33)

![](images/dd278c05a08a9fa6471cf3f861f315008fa6cade5c962f974560793db024330e.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~10^3               | ~10^2                | ~10^2                |
| 10^2  | ~10^2               | ~10^1.5              | ~10^1.5              |
| 10^3  | ~10^1.5             | ~10^1                | ~10^1                |
</details>

(S4.34)

![](images/3bf84787da3ecc16a263a183ed8c899b178205981eb113637c2a38ca543f4896.jpg)

<details>
<summary>line</summary>

| n    | E[KSD] (solid blue) | E[KSD] (dotted blue) | E[KSD] (dashed purple) |
| ---- | ------------------- | -------------------- | ---------------------- |
| 10   | ~10^3               | ~10^3                | ~10^2                  |
| 100  | ~10^2.5             | ~10^2.5              | ~10^1.5                |
| 1000 | ~10^2               | ~10^2                | ~10^1                  |
</details>

(S4.35)

![](images/3a62e0e1b86c19dfac3bc751defb51e2ed7b1bb2c3154a7acb0ef14443bcf0de.jpg)

<details>
<summary>line</summary>

| n    | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
| ---- | ------------------- | -------------------- | -------------------- |
| 10   | 1000                | 1000                 | 100                  |
| 100  | 100                 | 100                  | 10                   |
| 1000 | 10                  | 10                   | 1                    |
</details>

(S4.36)

![](images/a0e61e2f48bfe4543e4b99f9ea31d56dd694e45c93c454d411180b37713520ad.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed blue) | E[KSD] (dotted blue) | E[KSD] (dash-dot purple) |
|-------|---------------------|----------------------|----------------------|--------------------------|
| 10^1  | ~10^3               | ~10^3                | ~10^3                | ~10^2                    |
| 10^2  | ~10^2               | ~10^2                | ~10^2                | ~10^1.5                  |
| 10^3  | ~10^1.5             | ~10^1.5              | ~10^1.5              | ~10^1                    |
| 10^4  | ~10^1               | ~10^1                | ~10^1                | ~10^0.5                  |
</details>

(S4.37)

![](images/4454cd752ec7e195f4080defb81b87042f46bec354e645e74b917beaa75548f6.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid line) | E[KSD] (dashed line) | E[KSD] (dotted line) |
|-------|---------------------|----------------------|----------------------|
| 10^1  | ~300                | ~400                 | ~50                  |
| 10^2  | ~150                | ~200                 | ~10                  |
| 10^3  | ~70                 | ~100                 | ~5                   |
</details>

(S4.38)

![](images/c0fc02c9496645e2d7b8b8b28284b52ed8bc60e75f8e8ca8a948dc85f3226e50.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed light blue) | E[KSD] (dotted purple) |
|-------|---------------------|----------------------------|------------------------|
| 10^1  | ~20000              | ~15000                     | ~1000                  |
| 10^2  | ~10000              | ~8000                      | ~500                   |
| 10^3  | ~5000               | ~4000                      | ~200                   |
| 10^4  | ~2000               | ~1500                      | ~100                   |
</details>

(S4.39)

![](images/249315d0c2a9de205af1aa65599d0e4a7c60ed2c07b812de5d673c8f899e633e.jpg)

<details>
<summary>line</summary>

| n     | E[KSD] (solid blue) | E[KSD] (dashed light blue) | E[KSD] (dotted purple) |
|-------|---------------------|----------------------------|------------------------|
| 10^1  | ~10^4               | ~10^4                      | ~10^2                  |
| 10^2  | ~10^3               | ~10^3                      | ~10^1                  |
| 10^3  | ~10^2               | ~10^2                      | ~10^0                  |
</details>

(S4.40)

# D.6 Stein $\Pi$ -Thinning for PosteriorDB

The results presented in the main text concerned $n = 3 \times 10^{3}$ samples from MALA, which is near the limit at which the optimal weights $w^{\star}$ can be computed in a few seconds on a laptop PC. For larger values of n, sparse approximation methods are likely to be required. In the main text we presented Stein II-Thinning, which employs a greedy optimisation perspective to obtain a sparse approximation to the optimal weights at cost $O(m^{2}n)$ , where m are the number of greedy iterations performed. Explicit and verifiable conditions for the strong consistency of the resulting SIIT-MALA algorithm were established in Section 3.3. The purpose of this appendix is to empirically explore the convergence of SIIT-MALA using the PosteriorDB test bed.

In the experiments we report the number of MALA samples was fixed to $n = 10^{3}$ and the number of greedy iterations was varied from m = 1 to $m = 10^{3}$ . The results, in Figure S5, indicate that for most models in PosteriorDB the minimum value of KSD is approximately reached when m is anywhere from $\frac{n}{10}$ to $\frac{n}{2}$ , representing a modest but practically significant reduction in computational cost compared to SIIS-MALA. This agrees with the qualitative findings reported in the original Stein thinning paper of Riabiz et al. (2022).

Figure S5: Benchmarking on PosteriorDB. Here we investigate the convergence of the sparse approximation provided by the proposed Stein $\Pi$ -Thinning method (SIIT-MALA). The Langevin (purple) and KGM3–Stein kernels (blue) were used for SIIT-MALA and the associated KSDs are reported as the number m of iterations of Stein thinning is varied. Ten replicates were computed and standard errors were plotted. The name of each model is shown in the title of the corresponding panel, and the dimension d of the parameter vector is given in parentheses. [Langevin–Stein kernel: SIIT-MALA. KGM3–Stein kernel: SIIT-MALA.]   
![](images/144d011534b4316f8088bc721e2cdbeab10efbfbb29446426ac8bc5b94e956ac.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (Line 1) | E[KSD] (Line 2) |
|-------|-----------------|-----------------|
| 10^0  | ~10^2           | ~10^1           |
| 10^1  | ~10^1           | ~10^0           |
| 10^2  | ~10^0           | ~10^-1          |
| 10^3  | ~10^-1          | ~10^-2          |
</details>

(S5.1)

![](images/027f2508c5e0ca27a633db79508608151ac16223f0ef8b47d9e2ecb5e5947e57.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~10^1                | ~10^1              |
| 10^1  | ~10^0.5              | ~10^0.8            |
| 10^2  | ~10^0                | ~10^0.5            |
| 10^3  | ~10^-0.5             | ~10^0              |
</details>

(S5.2)

![](images/465a92cf86f1135f1432113e33c9f8eb31b8817bdba5c73de1c054d23d843d0a.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (Line 1) | E[KSD] (Line 2) |
|-------|-----------------|-----------------|
| 10^0  | ~10^2           | ~10^1           |
| 10^1  | ~10^1           | ~10^0           |
| 10^2  | ~10^0           | ~10^-1          |
| 10^3  | ~10^-1          | ~10^-2          |
</details>

(S5.3)

![](images/e2b30f7b7e0c37003a4e2f4a4f516eced5b87f1fb08222718786ae03d53fb416.jpg)

<details>
<summary>line</summary>

| m     | E [KSD] (Line 1) | E [KSD] (Line 2) |
|-------|------------------|------------------|
| 10^0  | ~10^2            | ~10^2            |
| 10^1  | ~10^1.5          | ~10^1.8          |
| 10^2  | ~10^1            | ~10^1.2          |
| 10^3  | ~10^0.5          | ~10^0.7          |
</details>

(S5.4)

![](images/8bf5742e2d23a1e4c1c1fe6da19ae675303ff3b7be38aec09878eb2a35e63e7b.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~20                    | ~25                |
| 10^1  | ~10                    | ~15                |
| 10^2  | ~3                     | ~8                 |
| 10^3  | ~1                     | ~4                 |
</details>

(S5.5)

![](images/9461052bf8cfc2699441758f1679e38a94012e5faeb2a58b0c385ce99c396ba8.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (light blue line) |
|-------|----------------------|--------------------------|
| 10^0  | ~10^2                | ~10^2                    |
| 10^1  | ~10^1.5              | ~10^2                    |
| 10^2  | ~10^1                | ~10^1.8                  |
| 10^3  | ~10^0.5              | ~10^1.5                  |
</details>

(S5.6)

![](images/7ce2fd538f1c30c64bdf5133a26cb577b73382cdddf2718f9d3935855889ed03.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (Line 1) | E[KSD] (Line 2) |
|-------|-----------------|-----------------|
| 10^0  | ~300            | ~250            |
| 10^1  | ~150            | ~120            |
| 10^2  | ~80             | ~60             |
| 10^3  | ~40             | ~30             |
</details>

(S5.7)

![](images/e49476119684ab699a360d41660a32c49883df4d2eba565df4dd615630a18da7.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | 10^1                 | 10^1               |
| 10^1  | ~10^0.5              | ~10^0.8            |
| 10^2  | ~10^0                | ~10^0.5            |
| 10^3  | ~10^-0.5             | ~10^0.3            |
</details>

(S5.8)

![](images/baaa08535aa3eea39b26771b06fdfcec1b1707db130c7fd2caed16689bfa7a12.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (Line 1) | E[KSD] (Line 2) |
|-------|-----------------|-----------------|
| 10^0  | ~300            | ~100            |
| 10^1  | ~150            | ~50             |
| 10^2  | ~80             | ~20             |
| 10^3  | ~40             | ~10             |
</details>

(S5.9)

![](images/7d42eebbf38ae2350d35852ba2f152ad92ded76fb41a284aa70d79367650f528.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~10^2                | ~10^2              |
| 10^1  | ~10^1.5              | ~10^1.8            |
| 10^2  | ~10^1                | ~10^1.5            |
| 10^3  | ~10^0.5              | ~10^1.2            |
</details>

(S5.10)

![](images/65cfb99dfde6895356c1f5d12249d1b1318d6909307f9f62a2d2e80caa005d90.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~200                 | ~300               |
| 10^1  | ~50                  | ~150               |
| 10^2  | ~10                  | ~80                |
| 10^3  | ~3                   | ~40                |
</details>

(S5.11)

![](images/f964cfa4346e7e850ebc8a20d242346cfef5e0ea6cc568a0e8a6ab475084999f.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~30                  | ~40                |
| 10^1  | ~15                  | ~25                |
| 10^2  | ~5                   | ~15                |
| 10^3  | ~2                   | ~8                 |
</details>

(S5.12)

![](images/dbf6cf59fc221ced2c21e92d25733b1e75574234ba9338bf2377dcc829778773.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~30                  | ~80                |
| 10^1  | ~10                  | ~40                |
| 10^2  | ~2                   | ~20                |
| 10^3  | ~1                   | ~10                |
</details>

(S5.13)

![](images/631fa463b4484a41b76e117d4cafd23d0d82fd7ef6373470b1f88ec20dca4f10.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~10^2              | ~10^1                |
| 10^1  | ~10^1              | ~10^0.5              |
| 10^2  | ~10^0.5            | ~10^0                |
| 10^3  | ~10^0              | ~10^0                |
</details>

(S5.14)

![](images/b2b90394db3dbea77c4fead399e9da6bbd672061536ede28f055044922803b53.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~10^2              | ~10^1                |
| 10^1  | ~10^1              | ~10^0                |
| 10^2  | ~10^0              | ~10^-1               |
| 10^3  | ~10^-1             | ~10^-2               |
</details>

(S5.15)

![](images/6774f84eaaf1fc8de46fd754436c00a3beb2f3e26487946e02151ce3b38a4843.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~10^2                | ~10^2              |
| 10^1  | ~10^1.5              | ~10^1.8            |
| 10^2  | ~10^1                | ~10^1.5            |
| 10^3  | ~10^0.5              | ~10^1.2            |
</details>

(S5.16)

![](images/be6936aa284c623e330ae430d933fc059ca8c1e26a7c75fd91c71fb3d7f4f434.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~10^2              | ~10^1.5              |
| 10^1  | ~10^1.8            | ~10^1                |
| 10^2  | ~10^1.4            | ~10^0.5              |
| 10^3  | ~10^1.2            | ~10^0                |
</details>

(S5.17)

![](images/01f2c6d4a606321ae8841af0d5f6848e972632d8b5671a3fb69de434dd78fbd0.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~10^2                | ~10^2              |
| 10^1  | ~10^1.5              | ~10^2              |
| 10^2  | ~10^1                | ~10^1.5            |
| 10^3  | ~10^0.5              | ~10^1              |
</details>

(S5.18)

![](images/80eb4269cebba866765487a74d4c312c820c10b43e478e18ec12b4eddf23633d.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (light blue line) |
|-------|----------------------|--------------------------|
| 10^0  | ~5000                | ~10000                   |
| 10^1  | ~2500                | ~7500                    |
| 10^2  | ~1000                | ~5000                    |
| 10^3  | ~500                 | ~2500                    |
</details>

(S5.19)

![](images/831ba5e1932adcf58010a6316b6376fa668bc5f0c2aebc6d416d06676e34f19b.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~10^4              | ~10^3                |
| 10^1  | ~5×10^3            | ~10^2.5              |
| 10^2  | ~2×10^3            | ~10^2                |
| 10^3  | ~1×10^3            | ~10^1.5              |
</details>

(S5.20)

![](images/7b0fbbcc48d20b5aaf104af5b10254b5da792262389b97d6b4d93b0ffd601529.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~10^2                | ~10^3              |
| 10^1  | ~10^1.5              | ~10^2.5            |
| 10^2  | ~10^1                | ~10^2              |
| 10^3  | ~10^0.5              | ~10^1.5            |
</details>

(S5.21)

![](images/d1fe96ad9ed0088b305b8db97511a630ca2c7b790b4d7eaa383a917b6fd4bcb3.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~10^2              | ~10^1.5              |
| 10^1  | ~10^1.8            | ~10^1                |
| 10^2  | ~10^1.5            | ~10^0.5              |
| 10^3  | ~10^1.3            | ~10^0.2              |
</details>

(S5.22)

![](images/1103223921029668ea915e5bb50625bc24a6e93f86cfd2abe6a98741d365097e.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (blue line) |
|-------|----------------------|--------------------|
| 10^0  | ~20                  | ~30                |
| 10^1  | ~10                  | ~25                |
| 10^2  | ~5                   | ~15                |
| 10^3  | ~3                   | ~10                |
</details>

(S5.23)

![](images/76c10f41b78cfc9331bd9dcfb50938717383074742a8fd5f5038cb88452aa90f.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (light blue) | E[KSD] (purple) |
|-------|---------------------|-----------------|
| 10^0  | ~3.5                | ~400            |
| 10^1  | ~2.5                | ~200            |
| 10^2  | ~1.5                | ~50             |
| 10^3  | ~1.0                | ~20             |
</details>

(S5.24)

![](images/7cf1b7e5ea43ed3a1e06104fad8a583844aa86b977fe919a14ad1db5f7bced96.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (light blue line) |
|-------|----------------------|--------------------------|
| 10^0  | ~10^2                | ~10^2                    |
| 10^1  | ~10^1.5              | ~10^1.8                  |
| 10^2  | ~10^1                | ~10^1.5                  |
| 10^3  | ~10^0.5              | ~10^1.2                  |
</details>

(S5.25)

![](images/872d898ca8027dcc4010b390f9cae6a3d9c975828c8d618811ddbe68d91a4eb4.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | 10^2               | 10^1                 |
| 10^1  | ~5×10^1            | ~3×10^1              |
| 10^2  | ~2×10^1            | ~5×10^-1             |
| 10^3  | ~1×10^1            | ~1×10^-2             |
</details>

(S5.26)

![](images/29a0deb8f4afaa51385cd579b8d6a0884eb5aff979a37bdcb5cafe9ca0cd3ede.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~10^2              | ~10^1                |
| 10^1  | ~10^1.5            | ~10^0.5              |
| 10^2  | ~10^1              | ~10^0                |
| 10^3  | ~10^0.8            | ~10^-0.5             |
</details>

(S5.27)

![](images/bec918572f72979eac9f4fc782fcc1e4d9232668444398e9cd06b175deeeedd1.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (Line 1) | E[KSD] (Line 2) |
|-------|-----------------|-----------------|
| 10^0  | ~10^1           | ~10^1           |
| 10^1  | ~10^0.5         | ~10^0.3         |
| 10^2  | ~10^-0.5        | ~10^-0.7        |
| 10^3  | ~10^-1          | ~10^-1.2        |
</details>

(S5.28)

![](images/aa76d5577d555e457b89409b9fc27a15f59285f34357b2f388aa4c5a9638624d.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (light blue) | E[KSD] (purple) |
|-------|---------------------|-----------------|
| 10^0  | ~30                 | ~5              |
| 10^1  | ~25                 | ~2              |
| 10^2  | ~20                 | ~1              |
| 10^3  | ~18                 | ~0.5            |
</details>

(S5.29)

![](images/d58912f0d063e5bb1b7dc9930ee4a4e6827c338629c3eb892174c45cb121ee8d.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (light blue) | E[KSD] (purple) |
|-------|---------------------|-----------------|
| 10^0  | ~10^2               | ~10^1           |
| 10^1  | ~10^2               | ~10^0.5         |
| 10^2  | ~10^2               | ~10^0           |
| 10^3  | ~10^2               | ~10^0           |
</details>

(S5.30)

![](images/1a965bcf3c271499b11785102614923f186fb3140771031a8c9799b3d10d08f9.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (light blue line) |
|-------|----------------------|--------------------------|
| 10^0  | ~10^2                | ~10^3                    |
| 10^1  | ~10^1.5              | ~10^2.5                  |
| 10^2  | ~10^1                | ~10^2                    |
| 10^3  | ~10^0.5              | ~10^1.5                  |
</details>

(S5.31)

![](images/11051fe5a5f9f6b84b25b65a7fd494a7b26a79ad6d4c506e3bf6c9ee56f83eb4.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~300               | ~150                 |
| 10^1  | ~200               | ~80                  |
| 10^2  | ~120               | ~30                  |
| 10^3  | ~100               | ~10                  |
</details>

(S5.32)

![](images/2c7adbf1044202eda8fb4aaf9f5477c0419c0c95789b0864ef7b702c5142226b.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~300               | 100                  |
| 10^1  | ~200               | ~50                  |
| 10^2  | ~150               | ~10                  |
| 10^3  | ~100               | ~5                   |
</details>

(S5.33)

![](images/34df80fe5ebdfceb7d08ace57eda9a9b9934c77b1672c0c76c5364d994c35db2.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (Line 1) | E[KSD] (Line 2) |
|-------|-----------------|-----------------|
| 10^0  | ~300            | ~150            |
| 10^1  | ~200            | ~70             |
| 10^2  | ~120            | ~30             |
| 10^3  | ~80             | ~10             |
</details>

(S5.34)

![](images/40b9c5c6f8d95fd8b1366e1737f4e93082d36373475c3f1d075285085faa24dd.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~300               | ~150                 |
| 10^1  | ~200               | ~70                  |
| 10^2  | ~150               | ~30                  |
| 10^3  | ~120               | ~10                  |
</details>

(S5.35)

![](images/7c2c875ff098662bbae3c0962ce0379387b7b9e02aff404e036f0a64bc04acc5.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~10^3              | ~10^2                |
| 10^1  | ~10^2.5            | ~10^1.5              |
| 10^2  | ~10^2              | ~10^1                |
| 10^3  | ~10^1.8            | ~10^0.8              |
</details>

(S5.36)

![](images/8618b3d446d83c06e53027a1049b70de0b8e69fa43b9c9a62dc4dc8e2f7317c2.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (purple line) | E[KSD] (light blue line) |
|-------|----------------------|--------------------------|
| 10^0  | ~10^2                | ~10^3                    |
| 10^1  | ~10^1.5              | ~10^2.5                  |
| 10^2  | ~10^1                | ~10^2                    |
| 10^3  | ~10^0.5              | ~10^1.5                  |
</details>

(S5.37)

![](images/349d02657b4745bf36ad2d82a9d868f249ef5d7941ffa3bfd9b63d3857026f9b.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (blue line) | E[KSD] (purple line) |
|-------|--------------------|----------------------|
| 10^0  | ~300               | ~100                 |
| 10^1  | ~150               | ~40                  |
| 10^2  | ~80                | ~10                  |
| 10^3  | ~60                | ~5                   |
</details>

(S5.38)

![](images/e7aa7287e13145d732ac5ed15f4d78e60123c313b50cc23974a90097335b081d.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (light blue) | E[KSD] (purple) |
|-------|---------------------|-----------------|
| 10^0  | ~10^4               | ~10^3           |
| 10^1  | ~5×10^3             | ~10^2.5         |
| 10^2  | ~2×10^3             | ~10^2           |
| 10^3  | ~1.5×10^3           | ~10^2           |
</details>

(S5.39)

![](images/0d0680d9e7466fa9e6f36e9b75dd392f4ad20921b3ec7a18c3d2a6ea0901e1e4.jpg)

<details>
<summary>line</summary>

| m     | E[KSD] (light blue) | E[KSD] (purple) |
|-------|---------------------|-----------------|
| 10^0  | ~10^5               | ~10^2           |
| 10^1  | ~10^4.5             | ~10^1.5         |
| 10^2  | ~10^4               | ~10^1           |
| 10^3  | ~10^4               | ~10^0.5         |
</details>

(S5.40)

# D.7 Performance of Stein Discrepancies

The properties of Stein discrepancies was out of scope for this work. Nonetheless, there is much interest in better understanding the properties of KSDs, and in this appendix the performance of SIIS-MALA in terms of 1-Wasserstein divergence is reported. This was made possible since PosteriorDB supplies a set of posterior samples obtained from a long run of Hamiltonian Monte Carlo (the No-U-Turn sampler in Stan) which we treat as a gold standard.

Full results are presented in Figure S6 and Table 2. Broadly speaking, in most cases the minimisation of KSD seems to be associated with minimisation of 1-Wasserstein distance, and in particular a significant improvement of SIIS-MALA over SIS-MALA is reported for $\approx 63\%$ of tasks in the PosteriorDB benchmark. However there are some scenarios for which minimisation of KSD is loosely, if at all, related to minimisation of 1-Wasserstein divergence. In these cases, we attribute this performance to a combination of two factors: First, the blindness to mixing proportions phenomena, described in Wenliang and Kanagawa (2021); Koehler et al. (2022); Liu et al. (2023), which is a pathology of KSDs in general. Second, the Langevin–Stein kernel cannot be expected to control convergence in 1-Wasserstein, since convergence in 1-Wasserstein is equivalent to weak convergence plus convergence of the first moment. Focusing therefore on the KGM3–Stein kernel only, it is encouraging to note that SIIS-MALA outperforms SIS-MALA on 83% of tasks in PosteriorDB in the 1-Wasserstein metric, as shown in Table 2. However, it is interesting to observe that MALA performed well in the 1-Wasserstein sense across the PosteriorDB test bed.

The development of improved Stein discrepancies is an active area of research, and we emphasise that the methodology developed in this work can be applied to any KSDs, including potentially KSDs with better or more direct control over standard notions of convergence (such as 1-Wasserstein) that in the future may be developed.

Figure S6: Performance of Stein discrepancies on PosteriorDB. Here we compared raw output from MALA (dotted lines) with the post-processed output provided by the default Stein importance sampling method of Liu and Lee (2017) (SIS-MALA; solid lines) and the proposed Stein II-Importance Sampling method (SIIIS-MALA; dashed lines). The Langevin (purple) and KGM3–Stein kernels (blue) were used for SIS-MALA and SIIIS-MALA, and the 1-Wasserstein divergence is reported as the number n of iterations of MALA is varied. Ten replicates were computed and standard errors were plotted. The name of each model is shown in the title of the corresponding panel, and the dimension d of the parameter vector is given in parentheses. [Legend: ……Raw MALA. Langevin–Stein kernel: ——SIS-MALA, ---- SIIIS-MALA. KGM3–Stein kernel: ——SIS-MALA, ---- SIIIS-MALA.]   
![](images/7c6a0586317ac6e3143229300f033b199375a2351dbaa4f5bebf49d3ad9b42b2.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (line 1) | E[WASD] (line 2) | E[WASD] (line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~9500            | ~9500            | ~9500            |
| 10^2  | ~9000            | ~9000            | ~9000            |
| 10^3  | ~7500            | ~7500            | ~7500            |
| 10^4  | ~7000            | ~7000            | ~7000            |
</details>

(S6.1)

![](images/17f728cd272cad1a615e6f92cdc0d892a390db721e0c0750dc0cc3eee3f5f904.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~0.4             | ~0.38            | ~0.35            |
| 10^2  | ~0.2             | ~0.18            | ~0.16            |
| 10^3  | ~0.08            | ~0.07            | ~0.06            |
</details>

(S6.2)

![](images/ec202cb8ee856c741ecad00ddb2e552c1141fe1367021ac6a201773a641ebeea.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed blue) | E[WASD] (solid purple) | E[WASD] (dotted black) |
|-------|----------------------|-----------------------|------------------------|------------------------|
| 10^1  | ~10^1                | ~10^1                 | ~10^1                  | ~10^1                  |
| 10^2  | ~10^0.5              | ~10^0.5               | ~10^0.5                | ~10^0.5                |
| 10^3  | ~10^0.5              | ~10^0.5               | ~10^0.5                | ~10^0.5                |
| 10^4  | ~10^0.5              | ~10^0.5               | ~10^0.5                | ~10^0.5                |
</details>

(S6.3)

![](images/6e383bd205c8df24a15de3ac97ae21680ff1ee7c5f0f66e1bd04d62372b82611.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) | E[WASD] (Line 4) |
|-------|------------------|------------------|------------------|------------------|
| 10^1  | ~10^1            | ~10^1            | ~10^1            | ~10^1            |
| 10^2  | ~10^0.5          | ~10^0.5          | ~10^0.5          | ~10^0.5          |
| 10^3  | ~10^0.5          | ~10^0.5          | ~10^0.5          | ~10^0.5          |
| 10^4  | ~10^0.5          | ~10^0.5          | ~10^0.5          | ~10^0.5          |
</details>

(S6.4)

![](images/4dbf7fcfdd9d1a85d74c497f780a6f36fc0117fe0d9a9aaaecb1487ac69890e8.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~0.1             | ~0.1             | ~0.1             |
| 10^2  | ~0.05            | ~0.06            | ~0.07            |
| 10^3  | ~0.02            | ~0.03            | ~0.04            |
</details>

(S6.5)

![](images/0b519071282a699660c55e4ac7707ee667c4e89ef6a4fbb8643f6c7fa3ba8ec4.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~0.065           | ~0.068           | ~0.070           |
| 10^2  | ~0.035           | ~0.038           | ~0.040           |
| 10^3  | ~0.015           | ~0.018           | ~0.020           |
</details>

(S6.6)

![](images/baf5d99fb1b1a4de9b9a03eef9b2208f71fc57cd22c623176db167b8939ae55d.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid purple) | E[WASD] (dashed purple) | E[WASD] (dotted black) | E[WASD] (dashed cyan) |
|-------|------------------------|-------------------------|------------------------|-----------------------|
| 10^1  | ~3.2e9                 | ~3.0e9                  | ~3.0e9                 | ~3.2e9                |
| 10^2  | ~2.8e9                 | ~2.5e9                  | ~2.7e9                 | ~2.0e9                |
| 10^3  | ~2.0e9                 | ~1.5e9                  | ~1.8e9                 | ~0.5e9                |
| 10^4  | ~1.5e9                 | ~1.0e9                  | ~1.2e9                 | ~0.2e9                |
</details>

(S6.7)

![](images/7a36b11db0d1b809103cf49e60713c07fbf4da3b613c1d19b94d1840e598688d.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~10^9            | ~10^9            | ~10^9            |
| 10^2  | ~10^8            | ~10^8            | ~10^8            |
| 10^3  | ~10^7            | ~10^7            | ~10^7            |
| 10^4  | ~10^6            | ~10^6            | ~10^6            |
</details>

(S6.8)

![](images/6c4bc8a510b44d97d49b776287f1c5015fae875a116ec3b2be1f746e21a951f1.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed purple) | E[WASD] (dotted black) | E[WASD] (dash-dot cyan) |
|-------|----------------------|-------------------------|------------------------|-------------------------|
| 10^1  | ~6.5e0               | ~6.8e0                  | ~6.7e0                 | ~6.2e0                  |
| 10^2  | ~5.5e0               | ~6.0e0                  | ~5.8e0                 | ~4.5e0                  |
| 10^3  | ~3.0e0               | ~2.5e0                  | ~3.5e0                 | ~1.5e0                  |
| 10^4  | ~2.0e0               | ~1.5e0                  | ~2.5e0                 | ~1.0e0                  |
</details>

(S6.9)

![](images/3a2a81e66037a04cae2533d91a662fa89471374206b3840e42420a1f54181c36.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] |
|-------|---------|
| 10^1  | ~0.1    |
| 10^2  | ~0.06   |
| 10^3  | ~0.03   |
</details>

(S6.10)

![](images/9059271896987846b2dafcf93658679d9a7b8140565f0fd4527ab0f58f0717ae.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed purple) | E[WASD] (dotted black) |
|-------|----------------------|-------------------------|------------------------|
| 10^1  | ~2.0                 | ~2.2                    | ~2.0                   |
| 10^2  | ~1.9                 | ~2.0                    | ~1.9                   |
| 10^3  | ~1.7                 | ~1.6                    | ~1.5                   |
| 10^4  | ~1.5                 | ~1.4                    | ~1.3                   |
</details>

(S6.11)

![](images/86d17b834593f6c88830c5ce3c5591e6387a6398cf1090f62872af926ab2ffe8.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed blue) | E[WASD] (solid purple) | E[WASD] (dotted black) |
|-------|----------------------|-----------------------|------------------------|------------------------|
| 10^1  | ~10^1                | ~10^1                 | ~10^1                  | ~10^1                  |
| 10^2  | ~10^0.5              | ~10^0.5               | ~10^0.5                | ~10^0.5                |
| 10^3  | ~10^0.2              | ~10^0.2               | ~10^0.2                | ~10^0.2                |
| 10^4  | ~10^0.1              | ~10^0.1               | ~10^0.1                | ~10^0.1                |
</details>

(S6.12)

![](images/7436bb587a1cb15550ef8a84f9aac88ebc410d90c62dd50c8451cf1580ad2751.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed blue) | E[WASD] (dotted black) |
|-------|----------------------|-----------------------|------------------------|
| 10^1  | ~10^9                | ~10^9                 | ~10^9                  |
| 10^2  | ~10^8                | ~10^8                 | ~10^7                  |
| 10^3  | ~10^7                | ~10^6                 | ~10^5                  |
| 10^4  | ~10^6                | ~10^5                 | ~10^4                  |
</details>

(S6.13)

![](images/ae765692ff8eee0a5eb59d41d3ede959dc1f2f02dc1b6c6d90281b8d78c361c2.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid line) | E[WASD] (dashed line) |
|-------|----------------------|------------------------|
| 10^1  | ~4.5e9               | ~4.8e9                 |
| 10^2  | ~2.2e9               | ~2.5e9                 |
| 10^3  | ~1.2e9               | ~1.5e9                 |
| 10^4  | ~7.0e8               | ~1.0e9                 |
</details>

(S6.14)

![](images/93e70914a0e00aca6e13871440cf9044373680c1f55f4c66604ebcf7e0a458cd.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid black) | E[WASD] (dashed blue) | E[WASD] (dash-dot cyan) | E[WASD] (dotted black) |
|-------|------------------------|------------------------|-------------------------|-------------------------|
| 10^1  | ~5.5e6                 | ~5.8e6                 | ~5.7e6                  | ~5.6e6                  |
| 10^2  | ~2.5e6                 | ~3.0e6                 | ~2.8e6                  | ~2.7e6                  |
| 10^3  | ~1.0e6                 | ~1.5e6                 | ~1.4e6                  | ~1.3e6                  |
</details>

(S6.15)

![](images/1eff7485a6a41148199eb8c80712e9d918de6ebf7a187a3d3b5a2efcd8e688ee.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid line) | E[WASD] (dashed line) | E[WASD] (dotted line) |
|-------|----------------------|-----------------------|-----------------------|
| 10^1  | ~0.9                 | ~0.8                  | ~0.9                  |
| 10^2  | ~0.5                 | ~0.4                  | ~0.5                  |
| 10^3  | ~0.2                 | ~0.2                  | ~0.2                  |
</details>

(S6.16)

![](images/46a81184906889b8598dc4822eedb73498fab83d3bed0c0d7485ac3b07c52a9d.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~0.3             | ~0.3             | ~0.3             |
| 10^2  | ~0.15            | ~0.15            | ~0.15            |
| 10^3  | ~0.05            | ~0.05            | ~0.05            |
</details>

(S6.17)

![](images/2c081738f1f435346061a7ec2ddb50cbef352528f7d0b3b99da24a5353a8a167.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | 0.5              | 0.45             | 0.4              |
| 10^2  | 0.25             | 0.2              | 0.15             |
| 10^3  | 0.15             | 0.1              | 0.08             |
| 10^4  | 0.1              | 0.08             | 0.05             |
</details>

(S6.18)

![](images/c67712a88cc732c8c01cc520db5ad8303b094351a0ba0f383863eada63607074.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Solid Blue) | E[WASD] (Dashed Light Blue) | E[WASD] (Dotted Purple) |
|-------|----------------------|-----------------------------|-------------------------|
| 10^1  | 6.4 × 10^-2          | 6.3 × 10^-2                 | 6.0 × 10^-2             |
| 10^2  | 6.3 × 10^-2          | 6.2 × 10^-2                 | 5.9 × 10^-2             |
| 10^3  | 6.2 × 10^-2          | 6.1 × 10^-2                 | 5.8 × 10^-2             |
</details>

(S6.19)

![](images/9eb53c71cc251bb4fcedba521986d5b596f6c0db3050a5b854d193a070186d10.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed purple) | E[WASD] (dotted red) |
|-------|----------------------|-------------------------|----------------------|
| 10^1  | ~0.075               | ~0.085                  | ~0.075               |
| 10^2  | ~0.075               | ~0.085                  | ~0.075               |
| 10^3  | ~0.075               | ~0.085                  | ~0.075               |
| >10^3 | ~0.075               | ~0.085                  | ~0.075               |
</details>

(S6.20)

![](images/9d64b8a80c0be5e4c2ea173fd0e6f8842c960a81ffbee77f26aede49038c8487.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~0.2             | ~0.18            | ~0.15            |
| 10^2  | ~0.1             | ~0.09            | ~0.08            |
| 10^3  | ~0.06            | ~0.055           | ~0.05            |
| 10^4  | ~0.04            | ~0.035           | ~0.03            |
</details>

(S6.21)

![](images/60792a115552192dd10cdf6ef6ba16c6372be202816a0a2529e9a393d6eceb08.jpg)

<details>
<summary>line</summary>

| n    | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
| ---- | ---------------- | ---------------- | ---------------- |
| 10   | 0.6              | 0.55             | 0.5              |
| 100  | 0.35             | 0.3              | 0.25             |
| 1000 | 0.15             | 0.1              | 0.08             |
</details>

(S6.22)

![](images/2840662e0bdc4f567f1e709e5e5c3e17c2008c7ec34fcb353a136e478d71d546.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~0.7             | ~0.8             | ~0.9             |
| 10^2  | ~0.4             | ~0.5             | ~0.6             |
| 10^3  | ~0.2             | ~0.3             | ~0.4             |
</details>

(S6.23)

![](images/1581b1f705f4096ca2ac8f175fee8ab3ce1be39260db1c44dd3f2cf0c4e2eda4.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid line) | E[WASD] (dashed line) | E[WASD] (dotted line) |
|-------|----------------------|-----------------------|-----------------------|
| 10^1  | ~9.5                 | ~10.5                 | ~9.3                  |
| 10^2  | ~9.2                 | ~10.0                 | ~9.0                  |
| 10^3  | ~8.5                 | ~8.8                  | ~8.6                  |
| 10^4  | ~8.2                 | ~8.2                  | ~8.1                  |
</details>

(S6.24)

![](images/33ea99cb12d2baf23fc045e7e380d8910054fb69c937a573c0192763e8b84fee.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed blue) | E[WASD] (dotted black) | E[WASD] (dash-dot purple) |
|-------|----------------------|-----------------------|------------------------|---------------------------|
| 10^1  | ~0.35                | ~0.34                 | ~0.36                  | ~0.37                     |
| 10^2  | ~0.25                | ~0.23                 | ~0.28                  | ~0.29                     |
| 10^3  | ~0.15                | ~0.13                 | ~0.16                  | ~0.17                     |
| 10^4  | ~0.10                | ~0.09                 | ~0.11                  | ~0.12                     |
</details>

![](images/51e0be78909f6c5737b9b76fa0d3f8744e560071c103257ae5b975e141067754.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] |
|-------|---------|
| 10^1  | 0.6     |
| 10^2  | 0.35    |
| 10^3  | 0.2     |
</details>

![](images/ef6db82c9238fb863e9e194455a5b797ad617b96f6596080491e825f1bcd9603.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~0.65            | ~0.60            | ~0.55            |
| 10^2  | ~0.40            | ~0.38            | ~0.35            |
| 10^3  | ~0.25            | ~0.23            | ~0.22            |
| >10^3 | ~0.20            | ~0.19            | ~0.18            |
</details>

![](images/95409df002bc6833dfacbb31cbb07f3c0ae03823266f0170b8ad8d2b668cf5ef.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid line) | E[WASD] (dashed line) |
|-------|----------------------|------------------------|
| 10^1  | ~400                 | ~400                   |
| 10^2  | ~250                 | ~250                   |
| 10^3  | ~100                 | ~100                   |
</details>

![](images/27f601a6cbb3b8d732d09e633413c8d5ac58c96237ca0631fd3debf1f74e5c41.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed blue) | E[WASD] (solid purple) |
|-------|----------------------|-----------------------|------------------------|
| 10^1  | ~20                  | ~25                   | ~25                    |
| 10^2  | ~10                  | ~15                   | ~15                    |
| 10^3  | ~5                   | ~10                   | ~10                    |
| 10^4  | ~2                   | ~5                    | ~5                     |
</details>

![](images/3b7c78223af10f32311dc7b4d717f931d30247198e933c83ea4e56af60dd5266.jpg)

<details>
<summary>line</summary>

| n    | E[WASD] (solid line) | E[WASD] (dashed line) |
| ---- | -------------------- | --------------------- |
| 10   | 1.82 × 10¹           | 1.80 × 10¹            |
| 30   | 1.76 × 10¹           | 1.74 × 10¹            |
| 100  | 1.72 × 10¹           | 1.70 × 10¹            |
| 300  | 1.69 × 10¹           | 1.68 × 10¹            |
| 1000 | 1.67 × 10¹           | 1.66 × 10¹            |
| 3000 | 1.65 × 10¹           | 1.64 × 10¹            |
</details>

![](images/8f97766ead0574bd5761b8aae299424128e016756e75548f7ad51ce51b617e8c.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid purple) | E[WASD] (dashed blue) | E[WASD] (dotted black) |
|-------|------------------------|------------------------|------------------------|
| 10^1  | ~0.6                   | ~0.55                  | ~0.6                   |
| 10^2  | ~0.4                   | ~0.35                  | ~0.4                   |
| 10^3  | ~0.25                  | ~0.2                   | ~0.25                  |
| 10^4  | ~0.15                  | ~0.1                   | ~0.15                  |
</details>

![](images/39796b9ceadee0d76cc8ccd29fffeb47cc4ae9eae701c4b232a959248e9001f3.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed purple) | E[WASD] (dotted black) |
|-------|----------------------|-------------------------|------------------------|
| 10^1  | ~0.6                 | ~0.58                   | ~0.6                   |
| 10^2  | ~0.45                | ~0.4                    | ~0.5                   |
| 10^3  | ~0.2                 | ~0.2                    | ~0.25                  |
</details>

![](images/beaefe5cf3fb5a1ec0580099cce8cbd0ce9e176a71db8e04bd972df163050774.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid line) | E[WASD] (dashed line) |
|-------|----------------------|------------------------|
| 10^1  | ~0.65                | ~0.62                  |
| 10^2  | ~0.45                | ~0.42                  |
| 10^3  | ~0.25                | ~0.23                  |
</details>

![](images/a582ab1265c48d9af13b56f25963a1c09dd0f375220601a915bd02438a24c5c1.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid purple) | E[WASD] (dashed red) | E[WASD] (dotted black) | E[WASD] (dashed blue) |
|-------|------------------------|----------------------|------------------------|-----------------------|
| 10^1  | ~0.65                  | ~0.63                | ~0.64                  | ~0.58                 |
| 10^2  | ~0.50                  | ~0.48                | ~0.47                  | ~0.42                 |
| 10^3  | ~0.25                  | ~0.23                | ~0.24                  | ~0.18                 |
</details>

![](images/53a526d536053b9c47ff9be333f5762507902565d3e38a36054720a58c91e92f.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid purple) | E[WASD] (dashed purple) | E[WASD] (solid cyan) | E[WASD] (dashed cyan) |
|-------|------------------------|-------------------------|----------------------|-----------------------|
| 10^1  | ~0.65                  | ~0.58                   | ~0.63                | ~0.60                 |
| 10^2  | ~0.45                  | ~0.38                   | ~0.42                | ~0.35                 |
| 10^3  | ~0.25                  | ~0.20                   | ~0.28                | ~0.22                 |
| 10^4  | ~0.18                  | ~0.15                   | ~0.20                | ~0.17                 |
</details>

![](images/a3553826826a2822bec8ae7eaa51f4f48ba39822840ad4dfb34e1779025ef783.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dotted blue) | E[WASD] (dashed blue) | E[WASD] (dash-dot blue) |
|-------|----------------------|-----------------------|-----------------------|-------------------------|
| 10^1  | ~0.6                 | ~0.6                  | ~0.55                 | ~0.55                   |
| 10^2  | ~0.38                | ~0.4                  | ~0.35                 | ~0.35                   |
| 10^3  | ~0.2                 | ~0.2                  | ~0.15                 | ~0.15                   |
</details>

![](images/571d9a9e3f682c84a7bce38c3ab314be904befa0162bde22c2ce570319ba0f8e.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (Line 1) | E[WASD] (Line 2) | E[WASD] (Line 3) |
|-------|------------------|------------------|------------------|
| 10^1  | ~0.65            | ~0.60            | ~0.68            |
| 10^2  | ~0.45            | ~0.40            | ~0.48            |
| 10^3  | ~0.25            | ~0.22            | ~0.28            |
| 10^4  | ~0.15            | ~0.13            | ~0.17            |
</details>

(S6.37)

![](images/f1ee84c5073c490774ec5aea2e28662e88aaeff5b2b31ac096fb0426d94d386b.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] |
|-------|---------|
| 10^1  | 1.0     |
| 10^2  | 0.5     |
| 10^3  | 0.3     |
</details>

(S6.38)

![](images/40c091e17180e61a78ce438ce53794e684191a983c843682e3c3ff8ea7980a04.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid line) | E[WASD] (dashed line) |
|-------|----------------------|------------------------|
| 10^3  | 0.66                 | 0.68                   |
| 10^2  | 0.65                 | 0.67                   |
| 10^3  | 0.64                 | 0.65                   |
</details>

(S6.39)

![](images/ecb9baf0dc885ba6e6b54fc2564497c4e8661a283cc5aa53808017790cddee29.jpg)

<details>
<summary>line</summary>

| n     | E[WASD] (solid blue) | E[WASD] (dashed purple) |
|-------|----------------------|-------------------------|
| 10^3  | ~1.59 × 10^1         | ~1.60 × 10^1            |
| 10^2  | ~1.58 × 10^1         | ~1.59 × 10^1            |
| 10^3  | ~1.55 × 10^1         | ~1.57 × 10^1            |
| >10^3 | ~1.54 × 10^1         | ~1.56 × 10^1            |
</details>

(S6.40)

# D.8 Investigation for a Skewed Target

This final appendix contrasts the 1-Wasserstein optimal sampling distribution $\Pi_1$ (c.f. Section 2.1), with the choice of $\Pi$ that we recommended in (8). In particular, we focus on the KGM3-Stein kernel under a heavily skewed $P$ , for which $\Pi_1$ and $\Pi$ can be markedly different.

For this investigation a bivariate skew-normal target was constructed, where the density is given by $p(x_{1}, x_{2}) = 4\phi(x_{1})\Phi(6x_{1})\phi(x_{2})\Phi(-3x_{2})$ , with $\phi$ and $\Phi$ respectively denoting the density and distribution functions of a standard Gaussian. The density p of P, together with the marginal densities of $\Pi_{1}$ and $\Pi$ , are plotted in Figure S7. It can be seen that, while both $\Pi_{1}$ and $\Pi$ are over-dispersed with respect to P, our recommended $\Pi$ assigns proportionally more mass to the tail that is positively skewed.

The performance of Stein $\Pi$ -Importance Sampling based on $\Pi_{1}$ and $\Pi$ is compared in Figure S8. Though both choices lead to an improvement relative to Stein importance sampling algorithm with $\Pi = P$ , the use of $\Pi$ leads to a significant further reduction (on average) in KSD compared to $\Pi_{1}$ . Based on our investigations, this finding seems general; the use of $\Pi_{1}$ does not realise the full potential of Stein $\Pi$ -Imporance sampling when the target is skewed.

<table><tr><td></td><td></td><td colspan="3">Langevin-Stein Kernel</td><td colspan="3">KGM3-Stein Kernel</td></tr><tr><td>Task</td><td>d</td><td>MALA</td><td>SIS - MALA</td><td>SIIS - MALA</td><td>MALA</td><td>SIS - MALA</td><td>SIIS - MALA</td></tr><tr><td>earnings-earn_height</td><td>3</td><td>6420.0</td><td>7230.0</td><td>9440.0</td><td>6420.0</td><td>7280.0</td><td>7390.0</td></tr><tr><td>gp_pois_regr-gp_regr</td><td>3</td><td>0.0779</td><td>0.0758</td><td>0.0724</td><td>0.0779</td><td>0.0865</td><td>0.0778</td></tr><tr><td>kidiq-kidscore_momhs</td><td>3</td><td>0.282</td><td>0.581</td><td>0.508</td><td>0.282</td><td>0.950</td><td>0.604</td></tr><tr><td>kidiq-kidscore_momiq</td><td>3</td><td>0.826</td><td>2.19</td><td>2.01</td><td>0.826</td><td>2.53</td><td>1.82</td></tr><tr><td>mesquite-logmesquite_logvolume</td><td>3</td><td>0.0236</td><td>0.0253</td><td>0.0238</td><td>0.0236</td><td>0.0311</td><td>0.0234</td></tr><tr><td>arma-arma11</td><td>4</td><td>0.0127</td><td>0.0132</td><td>0.0131</td><td>0.0127</td><td>0.0144</td><td>0.0133</td></tr><tr><td>earnings-logearn_logheight_male</td><td>4</td><td>1.46</td><td>1.75</td><td>1.34</td><td>1.46</td><td>1.56</td><td>0.725</td></tr><tr><td>garch-garch11</td><td>4</td><td>0.260</td><td>0.241</td><td>0.243</td><td>0.260</td><td>0.365</td><td>0.263</td></tr><tr><td>kidiq-kidscore_momhsiq</td><td>4</td><td>1.86</td><td>2.39</td><td>1.72</td><td>1.86</td><td>2.51</td><td>1.54</td></tr><tr><td>earnings-logearn_interaction_z</td><td>5</td><td>0.0245</td><td>0.0240</td><td>0.0240</td><td>0.0245</td><td>0.0252</td><td>0.0251</td></tr><tr><td>kidiq-kidscore_interaction</td><td>5</td><td>13.9</td><td>14.5</td><td>15.3</td><td>13.9</td><td>14.4</td><td>20.4</td></tr><tr><td>kidiq_with_mom_work-kidscore_interaction_c</td><td>5</td><td>0.289</td><td>0.251</td><td>0.237</td><td>0.289</td><td>0.584</td><td>0.371</td></tr><tr><td>kidiq_with_mom_work-kidscore_interaction_c2</td><td>5</td><td>0.258</td><td>0.269</td><td>0.252</td><td>0.258</td><td>0.638</td><td>0.430</td></tr><tr><td>kidiq_with_mom_work-kidscore_interaction_z</td><td>5</td><td>0.914</td><td>0.904</td><td>0.889</td><td>0.914</td><td>1.46</td><td>1.17</td></tr><tr><td>kidiq_with_mom_work-kidscore_mom_work</td><td>5</td><td>1.01</td><td>1.05</td><td>1.06</td><td>1.01</td><td>1.74</td><td>1.43</td></tr><tr><td>low_dim_gauss_mix-low_dim_gauss_mix</td><td>5</td><td>0.0215</td><td>0.0206</td><td>0.0207</td><td>0.0215</td><td>0.0214</td><td>0.0212</td></tr><tr><td>mesquite-logmesquite_logva</td><td>5</td><td>0.0715</td><td>0.0672</td><td>0.0681</td><td>0.0715</td><td>0.0833</td><td>0.0775</td></tr><tr><td>hmm_example-hmm_example</td><td>6</td><td>0.0708</td><td>0.102</td><td>0.105</td><td>0.0708</td><td>0.161</td><td>0.12</td></tr><tr><td>sblrc-blr</td><td>6</td><td>0.0613</td><td>0.0614</td><td>0.0581</td><td>0.0613</td><td>0.0613</td><td>0.0605</td></tr><tr><td>sbli-blr</td><td>6</td><td>0.0729</td><td>0.0733</td><td>0.0843</td><td>0.0729</td><td>0.0722</td><td>0.0926</td></tr><tr><td>arK-arK</td><td>7</td><td>0.0544</td><td>0.0533</td><td>0.0525</td><td>0.0544</td><td>0.0618</td><td>0.0557</td></tr><tr><td>mesquite-logmesquite_logvash</td><td>7</td><td>0.165</td><td>0.158</td><td>0.161</td><td>0.165</td><td>0.180</td><td>0.171</td></tr><tr><td>bball_drive_event_0-hmm_drive_0</td><td>8</td><td>0.203</td><td>0.183</td><td>0.186</td><td>0.203</td><td>0.242</td><td>0.216</td></tr><tr><td>bball_drive_event_1-hmm_drive_1</td><td>8</td><td>0.812</td><td>0.825</td><td>0.811</td><td>0.812</td><td>0.814</td><td>0.754</td></tr><tr><td>hudson_lynx_hare-lotka_volterra</td><td>8</td><td>0.113</td><td>0.112</td><td>0.111</td><td>0.113</td><td>0.135</td><td>0.120</td></tr><tr><td>mesquite-logmesquite</td><td>8</td><td>0.212</td><td>0.204</td><td>0.209</td><td>0.212</td><td>0.214</td><td>0.212</td></tr><tr><td>mesquite-logmesquite_logvas</td><td>8</td><td>0.197</td><td>0.192</td><td>0.196</td><td>0.197</td><td>0.208</td><td>0.203</td></tr><tr><td>mesquite-mesquite</td><td>8</td><td>123.0</td><td>122.0</td><td>114.0</td><td>123.0</td><td>123.0</td><td>121.0</td></tr><tr><td>eight_schools-eight_schools_centered</td><td>10</td><td>9.17</td><td>9.50</td><td>8.14</td><td>9.17</td><td>7.00</td><td>10.3</td></tr><tr><td>eight_schools-eight_schools_noncentered</td><td>10</td><td>16.8</td><td>16.7</td><td>16.7</td><td>16.8</td><td>16.7</td><td>16.7</td></tr><tr><td>nes1972-nes</td><td>10</td><td>0.193</td><td>0.189</td><td>0.172</td><td>0.193</td><td>0.198</td><td>0.172</td></tr><tr><td>nes1976-nes</td><td>10</td><td>0.190</td><td>0.189</td><td>0.179</td><td>0.190</td><td>0.202</td><td>0.177</td></tr><tr><td>nes1980-nes</td><td>10</td><td>0.229</td><td>0.228</td><td>0.222</td><td>0.229</td><td>0.256</td><td>0.233</td></tr><tr><td>nes1984-nes</td><td>10</td><td>0.202</td><td>0.198</td><td>0.197</td><td>0.202</td><td>0.210</td><td>0.184</td></tr><tr><td>nes1988-nes</td><td>10</td><td>0.212</td><td>0.210</td><td>0.185</td><td>0.212</td><td>0.214</td><td>0.185</td></tr><tr><td>nes1992-nes</td><td>10</td><td>0.172</td><td>0.169</td><td>0.159</td><td>0.172</td><td>0.182</td><td>0.168</td></tr><tr><td>nes1996-nes</td><td>10</td><td>0.191</td><td>0.187</td><td>0.173</td><td>0.191</td><td>0.199</td><td>0.179</td></tr><tr><td>nes2000-nes</td><td>10</td><td>0.275</td><td>0.273</td><td>0.274</td><td>0.275</td><td>0.306</td><td>0.284</td></tr><tr><td>diamonds-diamonds</td><td>26</td><td>0.648</td><td>0.647</td><td>0.643</td><td>0.648</td><td>0.648</td><td>0.658</td></tr><tr><td>mcycle_gp-accel_gp</td><td>66</td><td>15.5</td><td>15.5</td><td>15.6</td><td>15.5</td><td>15.5</td><td>15.8</td></tr></table>

Table 2: Benchmarking on PosteriorDB. Here we compared raw output from MALA with the post-processed output provided by the default Stein importance sampling method of Liu and Lee (2017) (SIS-MALA) and the proposed Stein II-Importance Sampling method (SIIS-MALA). Here $d = \dim(P)$ and the number of MALA samples was $n = 3 \times 10^{3}$ . The Langevin and KGM3–Stein kernels were used for SIS-MALA and SIIS-MALA and the associated 1-Wasserstein distances are reported. Ten replicates were computed and statistically significant improvement is highlighted in bold.

![](images/20506f19dfaf9585b356b4ddbf64fe5fc2b9ce25407f9f1653759a8b94561bab.jpg)

<details>
<summary>line</summary>

| x1   | P     | Π (1Wass.) | Π (KGM3) |
|------|-------|------------|----------|
| -1   | 0.000 | 0.000      | 0.000    |
| 0    | 0.750 | 0.600      | 0.330    |
| 1    | 0.450 | 0.400      | 0.420    |
| 2    | 0.200 | 0.250      | 0.380    |
| 3    | 0.050 | 0.100      | 0.250    |
| 4    | 0.010 | 0.020      | 0.100    |
| 5    | 0.005 | 0.010      | 0.050    |
| 6    | 0.002 | 0.005      | 0.025    |
</details>

![](images/191a6120af2b149329eace9b9ac863c6eaecb7070e031b0aacc94f61eef3acac.jpg)

<details>
<summary>line</summary>

| x2   | Blue Line | Orange Line | Green Line |
|------|-----------|-------------|------------|
| -6.0 | 0.0000    | 0.0000      | 0.0000     |
| -5.0 | 0.0000    | 0.0000      | 0.0000     |
| -4.0 | 0.0000    | 0.0000      | 0.0000     |
| -3.0 | 0.0500    | 0.0300      | 0.0200     |
| -2.0 | 0.2500    | 0.2000      | 0.1500     |
| -1.0 | 0.5500    | 0.4500      | 0.3500     |
| 0.0  | 0.6500    | 0.5200      | 0.4300     |
| 1.0  | 0.1000    | 0.1500      | 0.1200     |
| 2.0  | 0.0000    | 0.0000      | 0.0000     |
</details>

Figure S7: Comparing the proposed distribution $\Pi$ (KGM3; based on the KGM3–Stein kernel) to $\Pi_{1}$ (1Wass.; the optimal choice for 1-Wasserstein quantisation from Section 2.1) for a bivariate skew-normal target (d = 2). The marginal density functions of each distribution were approximated using $10^{6}$ samples from MCMC.

![](images/65c85d87e45614fe138dffe071ff3a3531ae2643c775481631f4a58e6a3f0cc2.jpg)

<details>
<summary>line</summary>

| n    | P       | Π (1Wass.) | Π (KGM3) |
| ---- | ------- | ---------- | -------- |
| 10   | ~5      | ~8         | ~12      |
| 100  | ~2      | ~2         | ~2       |
| 1000 | ~0.5    | ~0.3       | ~0.2     |
</details>

Figure S8: Comparing the performance of using the proposed distribution $\Pi$ (KGM3; based on the KGM3–Stein kernel) to $\Pi_{1}$ (1Wass.; the optimal choice for 1-Wasserstein quantisation from Section 2.1) for a bivariate skew-normal target (d = 2). The mean kernel Stein discrepancy (KSD) for Stein $\Pi$ -Importance Sampling was estimated; in each case, the KSD based on the KGM3–Stein kernel was computed. Solid lines indicate the baseline case of sampling from P, while dashed lines indicate sampling from $\Pi$ . (The experiment was repeated 10 times and standard error bars are plotted.)