# Improved Sample Complexity for Private Nonsmooth Nonconvex Optimization

Guy Kornowski $^{*}$ Daogao Liu $^{\dagger}$ Kunal Talwar $^{\ddagger}$

# Abstract

We study differentially private (DP) optimization algorithms for stochastic and empirical objectives which are neither smooth nor convex, and propose methods that return a Goldstein-stationary point with sample complexity bounds that improve on existing works. We start by providing a single-pass $(\varepsilon, \delta)$ -DP algorithm that returns an $(\alpha, \beta)$ -stationary point as long as the dataset is of size $\widetilde{\Omega}(\sqrt{d} / \alpha \beta^3 + d / \varepsilon \alpha \beta^2)$ , which is $\Omega(\sqrt{d})$ times smaller than the algorithm of Zhang et al. [2024] for this task, where $d$ is the dimension. We then provide a multi-pass polynomial time algorithm which further improves the sample complexity to $\widetilde{\Omega}\left(d / \beta^2 + d^{3/4} / \varepsilon \alpha^{1/2} \beta^{3/2}\right)$ , by designing a sample efficient ERM algorithm, and proving that Goldstein-stationary points generalize from the empirical loss to the population loss.

# 1 Introduction

We consider optimization problems in which the loss function is stochastic or empirical, of the form

$$
F (x) := \underset {\xi \sim \mathcal {P}} {\mathbb {E}} [ f (x; \xi) ], \quad \text {(stochastic)}
$$

$$
\widehat {F} ^ {\mathcal {D}} (x) := \frac {1}{n} \sum_ {i = 1} ^ {n} f (x; \xi_ {i}), \quad (\text { ERM })
$$

where P is the population distribution from which we sample a dataset $\mathcal{D} = (\xi_{1}, \ldots, \xi_{n}) \sim \mathcal{P}^{n}$ , and the component functions $f(\cdot; \xi) : \mathbb{R}^{d} \to \mathbb{R}$ may be neither smooth nor convex. Such problems are ubiquitous throughout machine learning, where losses given by deep-learning based models give rise to highly nonsmooth nonconvex (NSNC) landscapes.

Due to its fundamental importance in modern machine learning, the field of nonconvex optimization has received substantial attention in recent years. Moving away from the classical regime of convex optimization, many works aimed at understanding the complexity of producing approximate-stationary points (with small gradient norm) for smooth nonconvex functions [Ghadimi and Lan, 2013, Fang et al., 2018, Carmon et al., 2020, Arjevani et al., 2023]. However, smoothness rarely holds in modern practice, posing a major challenge for large and highly expressive models such as deep neural networks. Indeed, using ReLUs and MaxPool layers is common practice [LeCun et al., 2015], and moreover, stochastic gradient descent (SGD) with a large batch size tends to converge to sharp minima [Keskar et al., 2017].

As it turns out, without smoothness, it is actually impossible to directly minimize the gradient norm without suffering from an exponential-dimension dependent runtime in the worst case [Kornowski and Shamir, 2022]. Nonetheless, a nuanced notion coined as Goldstein-stationarity [Goldstein, 1977], has been shown in recent years to enable favorable guarantees. Roughly speaking, a point $x \in \mathbb{R}^d$ is called an $(\alpha, \beta)$ -Goldstein stationary point (or simply $(\alpha, \beta)$ -stationary) if there exists a convex combination of gradients in the $\alpha$ -ball around $x$ whose norm is at most $\beta$ . Following the groundbreaking work of Zhang et al. [2020], a surge of works study NSNC optimization through the lens of Goldstein stationarity, with associated finite-time guarantees [Davis et al., 2022, Lin et al., 2022, Cutkosky et al., 2023, Jordan et al., 2023, Kong and Lewis, 2023, Grimmer and Jia, 2024, Kornowski and Shamir, 2024, Tian and So, 2024].

In this work, we study NSNC optimization problems under the additional constraint of differential privacy (DP) [Dwork et al., 2006]. With the ever-growing deployment of ML models in various domains, the privacy of the data on which models are trained is a major concern. Accordingly, DP optimization is an extremely well-studied problem, with a vast literature focusing on functions that are assumed to be either convex or smooth [Bassily et al., 2014, Wang et al., 2017, Bassily et al., 2019, Wang et al., 2019, Feldman et al., 2020, Gopi et al., 2022, Arora et al., 2023, Carmon et al., 2023, Liu et al., 2024].

The fundamental investigation in this literature is the privacy-utility trade-off, that is, assessing the minimal dataset size n (referred to as the sample complexity) required in order to optimize the loss up to some error under DP. Being able to improve utility while using less samples has significant consequences, as in applications the amount of available data is a serious bottleneck, or arguably soon to become one [Villalobos et al., 2024].

For NSNC DP optimization, to the best of our knowledge the only existing result is by Zhang et al. [2024], which provided a zero-order algorithm (namely, utilizing only function value evaluations of $f(\cdot ;\xi)$ ) that preforms a single pass over the dataset and returns an $(\alpha ,\beta)$ -stationary point of $F$ under $(\varepsilon ,\delta)$ -DP as long as

$$
n = \widetilde {\Omega} \left(\frac {d}{\alpha \beta^ {3}} + \frac {d ^ {3 / 2}}{\varepsilon \alpha \beta^ {2}}\right). \tag {1}
$$

# 1.1 Our contributions

In this paper, we provide several new algorithms for NSNC DP optimization, which improve the previously best-known sample complexity for this task. Equivalently, given the same amount of data, they provide better utility. For consistency with the previous result by Zhang et al. [2024], throughout most of this paper we propose and analyze zero-order algorithms, yet we later generalize our results to accommodate first-order (i.e. gradient) oracles.

Our contributions, summarized in Table 1, are as follows:

1. Improved single-pass algorithm (Theorem 3.1): We provide an $(\varepsilon, \delta)$ -DP algorithm that preforms a single pass over that dataset, and returns an $(\alpha, \beta)$ -stationary point as long as

$$
n = \widetilde {\Omega} \left(\frac {\sqrt {d}}{\alpha \beta^ {3}} + \frac {d}{\varepsilon \alpha \beta^ {2}}\right), \tag {2}
$$

<table><tr><td>Sample complexity</td><td>empirical</td><td>stochastic</td></tr><tr><td>[Zhang et al., 2024] (single-pass)</td><td colspan="2"> $\frac{d}{\alpha\beta^3} + \frac{d^{3/2}}{\varepsilon\alpha\beta^2}$ </td></tr><tr><td>Theorem 3.1 (single-pass)</td><td colspan="2"> $\frac{\sqrt{d}}{\alpha\beta^3} + \frac{d}{\varepsilon\alpha\beta^2}$ </td></tr><tr><td>Theorem 4.1 (multi-pass)</td><td> $\frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}$ </td><td> $\frac{d}{\beta^2} + \frac{d^{3/4}}{\varepsilon\alpha^{1/2}\beta^{3/2}}$ </td></tr></table>

Table 1: Main results (ignoring dependence on Lipschitz constant, initialization, and log terms).

which is $\Omega(\sqrt{d})$ times smaller than (1). Notably, the “non-private” term $\sqrt{d}/\alpha\beta^{3}$ has sublinear dimension dependence, as opposed to (1), which was erroneously claimed impossible by previous work (see Remark 3.2).

2. Better multi-pass algorithm (Theorem 4.1): In order to further improve the sample complexity, we move to consider ERM algorithms that go over the data polynomially many times, which we will later argue generalize to the population loss. To that end, we provide an $(\varepsilon,\delta)$ -DP ERM algorithm, that returns an $(\alpha,\beta)$ -Goldstein stationary point of $\widehat{F}^{D}$ as long as

$$
n = \widetilde {\Omega} \left(\frac {d ^ {3 / 4}}{\varepsilon \alpha^ {1 / 2} \beta^ {3 / 2}}\right). \tag {3}
$$

Notably, (3) substantially improves (2) (and thus, (1)) in parameter regimes of interest (small $\varepsilon,\alpha,\beta$ , large d) with respect to the dimension and accuracy parameters, and in particular is the first algorithm to perform private ERM with sublinear dimension-dependent sample complexity for NSNC objectives.

In order to utilize our empirical algorithm for stochastic objectives, we must argue that Goldstein-stationarity generalizes from the ERM to the population. No such result appears in the literature, thus we proceed to prove it:

\- Additional contribution: generalizing from ERM to population in NSNC optimization (Proposition 5.1). We show that with high probability, any $(\alpha, \widehat{\beta})$ -stationary point of $\widehat{F}^{\mathcal{D}}$ is an $(\alpha, \beta)$ -stationary point of $F$ , for $\beta = \widehat{\beta} + \widetilde{O}(\sqrt{d/n})$ . Hence, the empirical guarantee (3) generalizes to stochastic losses with an additional $d/\beta^{2}$ additive term in $n$ (up to log terms).

3. First-order algorithm with reduced oracle complexity (Theorem 6.1): We provide a first-order (i.e., gradient-based) algorithm with the same sample complexity derived thus far, which requires $\widetilde{\Omega}(d^{2})$ times less oracle calls compared to the zero-order case. Overall, this establishes the best-known algorithm for NSNC DP both in terms of sample efficiency and oracle efficiency.

# 1.2 Our techniques

One of the main techniques we use is constructing a better gradient estimator with an improved effective sensitivity. We consider the zero-order gradient estimator

$$
\tilde {\nabla} f _ {\alpha} (x; \xi) = \frac {1}{m} \sum_ {j = 1} ^ {m} \frac {d}{2 \alpha} (f (x + \alpha y _ {j}; \xi) - f (x - \alpha y _ {j}; \xi)) \tag {4}
$$

for $(y_{i})_{i = 1}^{m}\stackrel {iid}{\sim}\mathrm{Unif}(\mathbb{S}^{d - 1})$ , which is an unbiased estimator of the smoothed gradient $\nabla f_{\alpha}(x;\xi)$ (cf. Section 2 for details), to which we then apply variance reduction. Zhang et al. [2024] considered this oracle specifically with $m = d$ , for which it is easy to bound the estimator's sensitivity over neighboring minibatches $\xi_{1:B},\xi_{1:B}^{\prime}$ of size $B$ by

$$
\left\| \frac {1}{B} \sum_ {i = 1} ^ {B} \tilde {\nabla} f _ {\alpha} (x; \xi_ {i}) - \frac {1}{B} \sum_ {i = 1} ^ {B} \tilde {\nabla} f _ {\alpha} (x; \xi_ {i} ^ {\prime}) \right\| \leq \frac {L d}{B}. \tag {5}
$$

Our key observation is that while this is indeed the worst-case sensitivity, we can get substantially lower sensitivity with high probability: For sufficiently large m, standard sub-Gaussian concentration bounds ensure that $\tilde{\nabla}f_{\alpha}(x;\xi_{i})\approx\nabla f_{\alpha}(x;\xi_{i})$ with high probability, and hence under this event we show the sensitivity over a mini-batch can be decreased to an order of $\frac{L}{B}$ . This is a factor of d smaller than (5), thus we can add significantly less noise in order to privatize, leading to faster convergence to stationarity.

# 2 Preliminaries

Notation. We denote by $\langle\cdot,\cdot\rangle$ , $\|\cdot\|$ the standard Euclidean dot product and its induced norm. For $x\in R^{d}$ and $\alpha>0$ , we denote by $\mathbb{B}(x,\alpha)$ the closed ball of radius $\alpha$ centered at x, and further denote $\mathbb{B}_{\alpha}:=\mathbb{B}(0,\alpha)$ . $S^{d-1}\subset R^{d}$ denotes the unit sphere. We make standard use of O-notation to hide absolute constants, $\widetilde{O},\widetilde{\Omega}$ to hide poly-logarithmic factors, and also let $f\lesssim g$ denote $f=O(g)$ .

Nonsmooth optimization. A function $h : R^{d} \to R$ is called L-Lipschitz if for all $x, y \in R^{d}$ : $|h(x) - h(y)| \leq L \|x - y\|$ . We call h H-smooth, if h is differentiable and $\nabla h$ is H-Lipschitz with respect to the Euclidean norm. For Lipschitz functions, the Clarke subgradient set [Clarke, 1990] can be defined as

$$
\partial h (x) := \mathrm{conv} \{g: g = \lim _ {n \to \infty} \nabla h (x _ {n}), x _ {n} \to x \},
$$

namely the convex hull of all limit points of $\nabla h(x_{n})$ over sequences of differentiable points (which are a full Lebesgue-measure set by Rademacher's theorem), converging to $x$ . For $\alpha \geq 0$ , the Goldstein $\alpha$ -subdifferential [Goldstein, 1977] is further defined as

$$
\partial_ {\alpha} h (x) := \mathrm{conv} (\cup_ {y \in \mathbb {B} (x, \alpha)} \partial h (y)),
$$

and we denote the minimum-norm element of the Goldstein $\alpha$ -subdifferential by

$$
\overline {{\partial}} _ {\alpha} h (x) := \arg \min _ {g \in \partial_ {\alpha} h (x)} \| g \|.
$$

Definition 2.1. A point $x \in \mathbb{R}^d$ is called an $(\alpha, \beta)$ -Goldstein stationary point of $h$ if $\| \overline{\partial}_{\alpha} h(x) \| \leq \beta$ .

Throughout the paper we impose the following standard Lipschitz assumption:

Assumption 2.2. For any $\xi$ , $f(\cdot; \xi): \mathbb{R}^d \to \mathbb{R}$ is $L$ -Lipschitz (hence, so is $F$ ).

Randomized smoothing. Given any function $h : R^{d} \to R$ , we denote its randomized smoothing $h_{\alpha}(x) := \mathbb{E}_{y \sim \mathbb{B}_{\alpha}} h(x + y)$ . We recall the following standard properties of randomized smoothing [Flaxman et al., 2005, Yousefian et al., 2012, Duchi et al., 2012, Shamir, 2017].

Fact 2.3 (Randomized smoothing). Suppose $h: \mathbb{R}^d \to \mathbb{R}$ is L-Lipschitz. Then:

(i) $h_{\alpha}$ is L-Lipschitz.   
(ii) $|h_{\alpha}(x) - h(x)| \leq L\alpha$ for any $x \in \mathbb{R}^d$ .   
(iii) $h_{\alpha}$ is $O(L\sqrt{d} /\alpha)$ -smooth.

(iv) $\nabla h_{\alpha}(x) = \mathbb{E}_{y\sim \mathbb{B}_{\alpha}}[\nabla h(x + y)] = \mathbb{E}_{y\sim \mathbb{S}^{d - 1}}[\frac{d}{2\alpha} (h(x + \alpha y) - h(x - \alpha y))y].$

The following result shows that in order to find a Goldstein-stationary point of a function, it suffices to find a Goldstein-stationary point of its randomized smoothing:

Lemma 2.4 (Kornowski and Shamir, 2024, Lemma 4). Any $(\alpha, \beta)$ -stationary point of $h_{\alpha}$ is a $(2\alpha, \beta)$ -stationary point of $h$ .

Differential privacy. Two datasets $\mathcal{D},\mathcal{D}'\in \mathrm{supp}(\mathcal{P})^n$ are said to be neighboring if they differ in only one data point. A randomized algorithm $\mathcal{A}:\mathcal{Z}^n\to \mathcal{R}$ is called $(\varepsilon ,\delta)$ differentially private (or $(\varepsilon ,\delta)$ -DP) for $\varepsilon ,\delta >0$ if for any two neighboring datasets $\mathcal{D},\mathcal{D}'$ and measurable $E\subseteq \mathcal{R}$ in the algorithm's range, it holds that $\operatorname {Pr}[\mathcal{A}(\mathcal{D})\in E]\leq e^{\varepsilon}\operatorname {Pr}[\mathcal{A}(\mathcal{D}')\in E] + \delta$ [Dwork et al., 2006].

Tree mechanism. Next, we revisit the well-known tree mechanism, detailed in Algorithm 1. In essence, the tree mechanism enables the privatization of cumulative sums $\sum_{j=1}^{t} g_j$ for all $t \in [\Sigma]$ , which, in our context, correspond to summed gradient estimators. A naive approach would add independent Gaussian noise $\zeta_j$ to each $g_j$ , leading to an error that scales as $\sqrt{\Sigma}$ . In contrast, the tree mechanism reduces this error to $O(\log \Sigma)$ by introducing correlated Gaussian noise. Broadly speaking, the mechanism generates a set of independent Gaussian noise values and organizes them in a tree-like structure, where each node corresponds to a specific noise value. To compute any cumulative sum $\sum_{j=1}^{t} g_j$ , the mechanism selects at most $O(\log \Sigma)$ nodes (Function NODE in Algorithm 1) from the tree and privatizes the sum using the aggregated noise from these nodes (i.e., final noise $\mathrm{TREE}(t)$ generated by Algorithm 1). The formal guarantee associated with this mechanism is provided below.

Proposition 2.5 (Tree Mechanism Dwork et al., 2010, Chan et al., 2011, Zhang et al., 2024). Let $\mathcal{Z}_1, \cdots, \mathcal{Z}_{\Sigma}$ be dataset spaces, suppose $\mathcal{X} \subseteq \mathbb{R}^d$ , and let $\mathcal{M}_i: \mathcal{X}^{i-1} \times \mathcal{Z}_i \to \mathcal{X}$ be a sequence of algorithms for $i \in [\Sigma]$ . Let ALG: $\mathcal{Z}^{(1:\Sigma)} \to \mathcal{X}^{\Sigma}$ be the algorithm that given a dataset $Z_{1:\Sigma} \in \mathcal{Z}^{(1:\Sigma)}$ , sequentially computes $X_i = \sum_{j=1}^i \mathcal{M}_j(X_{1:j-1}, Z_j) + \text{TREE}(i)$ for $i \in [\Sigma]$ , and then outputs $X_{1:\Sigma}$ . Suppose for all $i \in [\Sigma]$ , and neighboring $Z_{1:\Sigma}, Z_{1:\Sigma}' \in \mathcal{Z}^{(1:\Sigma)}$ , $\|\mathcal{M}_i(X_{1:i-1}, Z_i) - \mathcal{M}_i(X_{1:i-1}, Z_i')\| \leq s$ for all auxiliary inputs $X_{1:i-1} \in \mathcal{X}^{i-1}$ . Then setting $\sigma = 4s\sqrt{\log\Sigma\log(1/\delta)}/\varepsilon$ , Algorithm 1 is $(\varepsilon, \delta)$ -DP. Furthermore, for any $t \in [\Sigma]$ : $\mathbb{E}[\text{TREE}(t)] = 0$ and $\mathbb{E}\|\text{TREE}(t)\|^2 \lesssim d\log(\Sigma)\sigma^2$ .

In our case, the “dataset spaces” $Z_{i}$ will be collections of possible minibatches of some determined size, and $M_{i}$ will be a gradient estimator with respect to the sampled minibatch at some current iterate.

Algorithm 1 Tree Mechanism   
1: Input: Noise parameter $\sigma$ , sequence length $\Sigma$ 2: Define $\mathcal{T} := \{(u, v) : u = j \cdot 2^{\ell-1} + 1, v = (j + 1) \cdot 2^{\ell-1}, 1 \leq \ell \leq \log \Sigma, 0 \leq j \leq \Sigma / 2^{\ell-1} - 1\}$ 3: Sample and store $\zeta_{(u,v)} \sim \mathcal{N}(0, \sigma^2)$ for all $(u, v) \in \mathcal{T}$ 4: for $t = 1, \cdots, \Sigma$ do
5: Let $\text{TREE}(t) \leftarrow \sum_{(u,v) \in \text{NODE}(t)} \zeta_{(u,v)}$ 6: end for
7: Return: $\text{TREE}(t)$ for each $t \in [\Sigma]$ 8: Function NODE:
9: Input: index $t \in [\Sigma]$ 10: Initialize $S = \{\}$ and k = 0
11: for $i = 1, \cdots, \lceil \log \Sigma \rceil$ while k < t do
12: Set $k' = k + 2^{\lceil \log \Sigma \rceil - i}$ 13: if $k' \leq t$ then
14: $S \leftarrow S \cup \{(k + 1, k')\}$ , $k \leftarrow k'$ 15: end if
16: end for

Algorithm 2 Nonsmooth Nonconvex Algorithm (based on O2NC [Cutkosky et al., 2023])   
1: Input: Oracle $O : R^{d} \to R^{d}$ , initialization $x_{0} \in R^{d}$ , clipping parameter D > 0, step size $\eta > 0$ , averaging length $M \in N$ , iteration budget $T \in N$ .
2: Initialize: $\Delta_{1} = 0$ 3: for $t = 1, \ldots, T$ do
4: Sample $s_{t} \sim \text{Unif}[0, 1]$ 5: $x_{t} = x_{t-1} + \Delta_{t}$ 6: $z_{t} = x_{t-1} + s_{t}\Delta_{t}$ 7: $\tilde{g}_{t} = \mathcal{O}(z_{t})$ 8: $\Delta_{t+1} = \min\{1, \frac{D}{\|\Delta_{t}-\eta\tilde{g}_{t}\|}\} \cdot (\Delta_{t} - \eta\tilde{g}_{t})$ 9: end for
10: $K = \left\lfloor \frac{T}{M} \right\rfloor$ 11: for $k = 1, \ldots, K$ do
12: $\overline{x}_{k} = \frac{1}{M} \sum_{m=1}^{M} z_{(k-1)M+m}$ 13: end for
14: Sample $x^{out} \sim Unif\{\overline{x}_{1}, \ldots, \overline{x}_{K}\}$ 15: Output: $x^{out}$ .

# 2.1 Base algorithm: O2NC

Similar to Zhang et al. [2024], our general algorithm is based on the so-called “Online-to-Non-Convex conversion” (O2NC) of Cutkosky et al. [2023], which is generally an optimal method for finding Goldstein-stationary points (without the privacy constraint). We slightly modify previous proofs by disentangling the role of the variance of the gradient estimator vs. its second order moment, as follows:

Proposition 2.6 (O2NC). Suppose that $\mathcal{O}(\cdot)$ is a stochastic gradient oracle of some differentiable function $h:\mathbb{R}^d\to \mathbb{R}$ , so that for all $z\in \mathbb{R}^d$ : $\mathbb{E}\| \mathcal{O}(z) - \nabla h(z)\|^2\leq G_0^2$ and $\mathbb{E}\| \mathcal{O}(z)\|^2\leq G_1^2$ . Then

running Algorithm 2 with $\eta = \frac{D}{G_1\sqrt{M}}$ , $MD\leq \alpha$ , uses $T$ calls to $\mathcal{O}(\cdot)$ , and satisfies

$$
\mathbb {E} \left\| \overline {{\partial}} _ {\alpha} h (x ^ {\mathrm{out}}) \right\| \leq \frac {h (x _ {0}) - \inf h}{D T} + \frac {3 G _ {1}}{2 \sqrt {M}} + G _ {0}.
$$

We provide a proof of Proposition 2.6 in Appendix A. Recalling that by Lemma 2.4 any $(\alpha,\beta)$ -stationary point of $F_{\alpha}$ is a $(2\alpha,\beta)$ -stationary point of F, we see that it is enough to design a private stochastic gradient oracle O of $\nabla F_{\alpha}$ , while controlling its variance $G_{0}$ and second moment $G_{1}$ . In the next sections, we show how to construct such private oracles and derive the corresponding guarantees through Proposition 2.6. As previously remarked, throughout most of the paper, our oracles will be based on zero-order queries of the component functions $f(\cdot,\xi)$ , yet in Section 6 we construct oracles with the same sample complexity using first-order queries, leading to a lower oracle complexity overall.

# 3 Single-pass algorithm

In this section, we consider Algorithm 3, which provides an oracle to be used in Algorithm 2. Algorithm 3 is such that throughout T calls, it uses each data point once, and hence, privacy is maintained with no need for composition. The main theorem in this section is the following:

Theorem 3.1 (Single-pass algorithm). Suppose $F(x_0) - \inf_x F(x) \leq \Phi$ , that Assumption 2.2 holds, and let $\alpha, \beta, \delta, \varepsilon > 0$ such that $\alpha \leq \frac{\Phi}{L}$ . Then setting $B_1 = \Sigma$ , $B_2 = 1$ , $M = \alpha / 4D$ , $m = \widetilde{O}(d^2 B_1^2 + \frac{d\alpha^2 B_2^2}{D^2})$ , $\sigma = \widetilde{O}(\frac{L}{B_1\varepsilon} + \frac{LD\sqrt{d}}{\alpha B_2\varepsilon})$ , $\Sigma = \widetilde{\Theta}((\frac{\alpha}{\varepsilon D})^{2/3} + \frac{\alpha}{Dd^{1/2}})$ , $D = \widetilde{\Theta}(\min\{(\frac{\Phi^2\alpha}{L^2T^2})^{1/3}, (\frac{\Phi\alpha\varepsilon}{dLT})^{1/2}, (\frac{\Phi^3\alpha^2\varepsilon}{d^3/2L^3T^3})^{1/5}, (\frac{\Phi^2\alpha}{L^2T^2\sqrt{d}})^{1/3}\})$ , $T = \Theta(n)$ , and running Algorithm 2 with Algorithm 3 as the oracle subroutine, is $(\varepsilon, \delta)$ -DP. Furthermore, its output satisfies $\mathbb{E}\| \overline{\partial}_{2\alpha}F(x^{\mathrm{out}})\| \leq \beta$ as long as

$$
n = \widetilde {\Omega} \left(\frac {\Phi L ^ {2} \sqrt {d}}{\alpha \beta^ {3}} + \frac {\Phi L d}{\varepsilon \alpha \beta^ {2}}\right).
$$

Remark 3.2. It is interesting to note that the “non-private” term $\Phi L^{2}\sqrt{d}/\alpha\beta^{3}$ in Theorem 3.1 has sublinear dependence on the dimension d. Not only is this the first such result, this was even (erroneously) claimed impossible by Zhang et al. [2024]. The reason for this confusion is that while the optimal zero-order oracle complexity is $d/\alpha\beta^{3}$ [Kornowski and Shamir, 2024], and in particular must scale linearly with the dimension [Duchi et al., 2015], the sample complexity might not.

Remark 3.3. Since Algorithm 2 uses T calls to $\mathcal{O}(\cdot)$ , and it easy to see that the amortized oracle complexity of $\mathcal{O}(\cdot)$ is $O(m)$ , the overall oracle complexity we get is $O(Tm)$ . As previously mentioned, we set m to reduce the sensitivity of $\mathcal{O}(\cdot)$ , leading to an improvement in sample complexity. More generally, our analysis allows trading-off sample and oracle complexities in a Pareto-front fashion. We further use this observation in Section 6, where we show that first-order oracles allow setting a substantially smaller m while maintaining the same reduced sensitivity.

In the rest of the section, we will present the basic properties of this oracle in terms of sensitivity (implying the privacy), variance and second moment. We will then plug these into Algorithm 2, which enables proving Theorem 3.1. Corresponding proofs are deferred to Section 7.

Algorithm 3 Single-pass instantiation of $\mathcal{O}(z_t)$ in Line 7 of Algorithm 2   
1: Input: Current iterate $z_{t}$ , time $t \in N$ , period length $\Sigma \in N$ , accuracy parameter $\alpha > 0$ , batch sizes $B_{1}, B_{2} \in N$ , gradient validation size $m \in N$ , noise level $\sigma > 0$ .
2: if t mod $\Sigma = 1$ then
3: Sample minibatch $S_{t}$ of size $B_{1}$ from unused samples
4: for each sample $\xi_{i} \in S_{t}$ do
5: Sample $y_{1}, \ldots, y_{m} \stackrel{iid}{\sim} \text{Unif}(\mathbb{S}^{d-1})$ 6: $\tilde{\nabla}f(z_{t}; \xi_{i}) = \frac{1}{m}\sum_{j \in [m]} \frac{d}{2\alpha}(f(z_{t} + \alpha y_{j}; \xi_{i}) - f(z_{t} - \alpha y_{j}; \xi_{i}))y_{j}$ 7: end for
8: $g_{t} = \frac{1}{B_{1}}\sum_{\xi_{i} \in S_{t}} \tilde{\nabla}f(z_{t}; \xi_{i})$ 9: else
10: Sample minibatch $S_{t}$ of size $B_{2}$ from unused samples
11: for each sample $\xi_{i} \in S_{t}$ do
12: Sample $y_{1}, \ldots, y_{2m} \stackrel{iid}{\sim} \text{Unif}(\mathbb{B}_{\alpha})$ 13: $\tilde{\nabla}f(z_{t}; \xi_{i}) = \frac{1}{m}\sum_{j \in [m]} \frac{d}{2\alpha}(f(z_{t} + \alpha y_{j}; \xi_{i}) - f(z_{t} - \alpha y_{j}; \xi_{i}))y_{j}$ 14: $\tilde{\nabla}f(z_{t-1}; \xi_{i}) = \frac{1}{m}\sum_{j=m+1}^{2m}\frac{d}{2\alpha}(f(z_{t-1} + \alpha y_{j}; \xi_{i}) - f(z_{t-1} - \alpha y_{j}; \xi_{i}))y_{j}$ 15: end for
16: $g_{t} = g_{t-1} + \frac{1}{B_{2}}\sum_{\xi_{i} \in S_{t}} (\tilde{\nabla}f(z_{t}; \xi_{i}) - \tilde{\nabla}f(z_{t-1}; \xi_{i}))$ 17: end if
18: Return $\tilde{g}_{t} = g_{t} + \text{TREE}(\sigma, \Sigma)(t \mod \Sigma)$

Lemma 3.4 (Sensitivity). Consider the gradient oracle $\mathcal{O}(\cdot)$ in Algorithm 3 when acting on two neighboring minibatches $S_{t}$ and $S_{t}^{\prime}$ , and correspondingly producing $g_{t}$ and $g_{t}^{\prime}$ , respectively. If $t$ mod $\Sigma = 1$ , then it holds with probability at least $1 - \delta /2$ that

$$
\| g _ {t} - g _ {t} ^ {\prime} \| \lesssim \frac {L}{B _ {1}} + \frac {L d \sqrt {\log (d B _ {1} / \delta)}}{\sqrt {m}}.
$$

Otherwise, conditioned on $g_{t-1} = g_{t-1}'$ , we have with probability at least $1 - \delta/2$ :

$$
\left\| g _ {t} - g _ {t} ^ {\prime} \right\| \lesssim \frac {L \sqrt {d} D}{\alpha B _ {2}} + \frac {L d \sqrt {\log (d B _ {1} / \delta)}}{\sqrt {m}}.
$$

With the sensitivity bound given by Lemma 3.4, we easily derive the privacy guarantee of our oracle from the Tree Mechanism (Proposition 2.5).

Lemma 3.5 (Privacy). Running Algorithm 3 with $m = O\left(\log (dB_2 / \delta)(d^2 B_1^2 +\frac{d\alpha^2B_2^2}{D^2})\right)$ and $\sigma = O\left(\frac{L\sqrt{\log(1 / \delta)}}{B_1\varepsilon} +\frac{LD\sqrt{d\log(1 / \delta)}}{\alpha B_2\varepsilon}\right)$ is $(\varepsilon ,\delta)$ -DP.

We next analyze the variance and second moment of the gradient oracle.

Lemma 3.6 (Variance). In Algorithm 3, for all $t \in [T]$ it holds that

$$
\mathbb {E} \left\| \tilde {g} _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} \lesssim \frac {L ^ {2}}{B _ {1}} + \frac {L ^ {2} d ^ {2}}{B _ {1} m} + \frac {L ^ {2} d D ^ {2} \Sigma}{\alpha^ {2} B _ {2}} + \sigma^ {2} d \log \Sigma + \frac {L ^ {2} d ^ {2} \Sigma}{m B _ {2}},
$$

$$
\mathbb {E} \left\| \tilde {g} _ {t} \right\| ^ {2} \lesssim L ^ {2} + \frac {L ^ {2} d ^ {2}}{B _ {1} m} + \frac {L ^ {2} d D ^ {2} \Sigma}{\alpha^ {2} B _ {2}} + \sigma^ {2} d \log \Sigma + \frac {L ^ {2} d ^ {2} \Sigma}{m B _ {2}}.
$$

Combining the ingredients that we have set up, we can derive Theorem 3.1.

Proof of Theorem 3.1. The privacy guarantee follows directly from Lemma 3.5, by noting that our parameter assignment implies $B_{1}T/\Sigma + B_{2}T = O(n)$ , which allows letting $T = \Theta(n)$ while never re-using samples (hence no privacy composition is required). Therefore, it remains to show the utility bound. By applying Lemma 2.4 and Proposition 2.6, we get that

$$
\mathbb {E} \left\| \overline {{\partial}} _ {2 \alpha} F (x ^ {\mathrm{out}}) \right\| \leq \mathbb {E} \left\| \overline {{\partial}} _ {\alpha} F _ {\alpha} (x ^ {\mathrm{out}}) \right\| \leq \frac {F _ {\alpha} (x _ {0}) - \inf F _ {\alpha}}{D T} + \frac {3 G _ {1}}{2 \sqrt {M}} + G _ {0} \leq \frac {2 \Phi}{D T} + \frac {3 G _ {1}}{2 \sqrt {M}} + G _ {0}, (6)
$$

where the last inequality used the fact that Assumption 2.2 and Fact 2.3 together imply that $F_{\alpha}(x_0) - \inf F_{\alpha} \leq F(x_0) - \inf F + L\alpha \leq \Phi + L\alpha \leq 2\Phi$ . Under our parameter assignment, Lemma 3.6 yields

$$
G _ {1} \lesssim G _ {0} + L, \tag {7}
$$

which plugged into (6) gives

$$
\mathbb {E} \left\| \overline {{\partial}} _ {2 \alpha} F (x ^ {\mathrm{out}}) \right\| = O \left(\frac {\Phi}{D T} + \frac {L}{\sqrt {M}} + G _ {0}\right). \tag {8}
$$

Moreover, under our parameter assignment, Lemma 3.6 also gives the bound

$$
G _ {0} \lesssim \frac {L}{\sqrt {B _ {1}}} + \frac {L D \sqrt {d \Sigma}}{\alpha \sqrt {B _ {2}}} + \sigma \sqrt {d \log \Sigma} = \widetilde {O} \left(\frac {L}{\sqrt {\Sigma}} + \frac {L D d ^ {1 / 2} \Sigma^ {1 / 2}}{\alpha} + \frac {L d ^ {1 / 2}}{\Sigma \varepsilon} + \frac {L D d}{\alpha \varepsilon}\right), \tag {9}
$$

which propagated into (8) and recalling that $M = \Theta(\alpha/D)$ shows that

$$
\mathbb {E} \left\| \overline {{{\partial}}} _ {2 \alpha} F (x ^ {\mathrm{out}}) \right\| = \widetilde {O} \bigg (\frac {\Phi}{D T} + \frac {L D ^ {1 / 2}}{\alpha^ {1 / 2}} + \frac {L D d ^ {1 / 2} \Sigma^ {1 / 2}}{\alpha} + \frac {L}{\sqrt {\Sigma}} + \frac {L d ^ {1 / 2}}{\Sigma \varepsilon} + \frac {L D d}{\alpha \varepsilon} \bigg).
$$

Plugging our assignments of $\Sigma$ and D, and recalling that $n = \Theta(T)$ , a straightforward calculation simplifies the bound above to

$$
\mathbb {E} \left\| \overline {{\partial}} _ {2 \alpha} F (x ^ {\mathrm{out}}) \right\| = \widetilde {O} \bigg (\Big (\frac {\Phi L ^ {2} \sqrt {d}}{n \alpha} \Big) ^ {1 / 3} + \Big (\frac {\Phi d L}{n \alpha \varepsilon} \Big) ^ {1 / 2} + \Big (\frac {\Phi^ {2} L ^ {3} d ^ {3 / 2}}{n ^ {2} \alpha^ {2} \varepsilon} \Big) ^ {1 / 5} \bigg).
$$

Bounding by $\beta$ and solving for n results in

$$
n = \widetilde {\Omega} \left(\frac {\Phi L ^ {2} \sqrt {d}}{\alpha \beta^ {3}} + \frac {\Phi L d}{\varepsilon \alpha \beta^ {2}} + \frac {\Phi L ^ {3 / 2} d ^ {3 / 4}}{\varepsilon^ {1 / 2} \alpha \beta^ {5 / 2}}\right).
$$

To complete the proof, we simply note that $\frac{\Phi L^{3/2}d^{3/4}}{\varepsilon^{1/2}\alpha\beta^{5/2}}\leq \frac{\Phi L^2\sqrt{d}}{\alpha\beta^3} +\frac{\Phi Ld}{\varepsilon\alpha\beta^2}$ by the AM-GM inequality, and so the third term above is negligible.

□

Algorithm 4 Multi-pass instantiation of $\mathcal{O}(z_t)$ in Line 7 of Algorithm 2   
1: Input: Current iterate $z_{t}$ , time $t \in N$ , period length $\Sigma \in N$ , accuracy parameter $\alpha > 0$ , gradient validation size $m \in N$ , noise levels $\sigma_{1}, \sigma_{2} > 0$ .
2: if t mod $\Sigma = 1$ then
3: for each sample $\xi_{i} \in D$ do
4: Sample $y_{1}, \ldots, y_{m} \stackrel{iid}{\sim} \text{Unif}(\mathbb{S}^{d-1})$ 5: $\tilde{\nabla} f(z_{t}; \xi_{i}) = \frac{1}{m} \sum_{j \in [m]} \frac{d}{2\alpha} (f(z_{t} + \alpha y_{j}; \xi_{i}) - f(z_{t} - \alpha y_{j}; \xi_{i})) y_{j}$ 6: end for
7: $g_{t} = \frac{1}{n} \sum_{\xi_{i} \in D} \tilde{\nabla} f(z_{t}; \xi_{i})$ 8: Return: $\tilde{g}_{t} = g_{t} + \chi_{t}$ , where $\chi_{t} \sim \mathcal{N}(0, \sigma_{1}^{2} I_{d})$ 9: else
10: for each sample $\xi_{i} \in D$ do
11: Sample $y_{1}, \ldots, y_{2m} \stackrel{iid}{\sim} \text{Unif}(\mathbb{B}_{\alpha})$ 12: $\tilde{\nabla} f(z_{t}; \xi_{i}) = \frac{1}{m} \sum_{j=1}^{m} \frac{d}{2\alpha} (f(z_{t} + \alpha y_{j}; \xi_{i}) - f(z_{t} - \alpha y_{j}; \xi_{i})) y_{j}$ 13: $\tilde{\nabla} f(z_{t-1}; \xi_{i}) = \frac{1}{m} \sum_{j=m+1}^{2m} \frac{d}{2\alpha} (f(z_{t-1} + \alpha y_{j}; \xi_{i}) - f(z_{t-1} - \alpha y_{j}; \xi_{i})) y_{j}$ 14: end for
15: $g_{t} = \tilde{g}_{t-1} + \frac{1}{n} \sum_{\xi_{i} \in D} (\tilde{\nabla} f(z_{t}; \xi_{i}) - \tilde{\nabla} f(z_{t-1}; \xi_{i}))$ 16: Return: $\tilde{g}_{t} = g_{t} + \chi_{t}$ , where $\chi_{t} \sim N(0, \sigma_{2}^{2} I_{d})$ .
17: end if

# 4 Multi-pass algorithm

In this section, we consider a different oracle construction given by Algorithm 4, to be used in Algorithm 2. The main difference from the previous section is that this oracle reuses data points a polynomial number of times, and therefore cannot directly guarantee generalization to the stochastic objective. Instead, in this section we analyze the empirical objective $\widehat{F}^{\mathcal{D}}(x):=\frac{1}{n}\sum_{i=1}^{n}f(x;\xi_{i})$ . After establishing ERM results, in Section 5 we show that any empirical Goldstein-stationarity guarantee generalizes to the population loss.

Similarly to the single-pass oracle (Algorithm 3), we use randomized smoothing and variance reduction. A difference in the oracle construction is that we replace the tree mechanism with the Gaussian mechanism and apply advanced composition for the privacy analysis (since now samples are reused). The main theorem for this section is the following:

Theorem 4.1 (Multi-pass ERM). Suppose $\widehat{F}^{\mathcal{D}}(x_0) - \inf_x\widehat{F}^{\mathcal{D}}(x)\leq \Phi$ , Assumption 2.2 holds, and let $\alpha ,\beta ,\delta ,\varepsilon >0$ such that $\alpha \leq \frac{\Phi}{L}$ . Then setting $m = \frac{L^2d\Sigma}{n\sigma_1^2} +\frac{L^2d}{n\sigma_2^2}$ , $\sigma_{1} = O(\frac{L\sqrt{T\log(1 / \delta) / \Sigma}}{n\varepsilon})$ , $\sigma_{2} = O(\frac{LD\sqrt{Td\log(1 / \delta)}}{\alpha n\varepsilon})$ , $\Sigma = \tilde{\Theta} (\frac{\alpha}{D\sqrt{d}})$ , $D = \tilde{\Theta} (\frac{\alpha^{2}\beta^{2}}{L^{2}}),T = \tilde{\Theta} (\frac{\Phi L^2}{\alpha^2\beta^3})$ , and running Algorithm 2 with Algorithm 4 as the oracle subroutine is $(\varepsilon ,\delta)$ -DP. Furthermore, its output satisfies $\mathbb{E}\| \overline{\partial}_{2\alpha}\widehat{F}^{\mathcal{D}}(x^{\mathrm{out}})\| \leq \beta$ as long as

$$
n = \widetilde {\Omega} \left(\frac {\sqrt {\Phi} L d ^ {3 / 4}}{\varepsilon \alpha^ {1 / 2} \beta^ {3 / 2}}\right).
$$

Remark 4.2. As we will show in Section 5, Theorem 4.1 also provides the same population guarantee for $\|\overline{\partial}_{2\alpha}F(x^{\mathrm{out}})\|$ with an additional $L^{2}d/\beta^{2}$ term (up to log factors) to the sample complexity.

To prove Theorem 4.1, we analyze the properties of the oracle given by Algorithm 4. The

sensitivity of $g_{t}$ in Algorithm 4 directly follows from Lemma 3.4. $^{2}$ By the standard composition results of the Gaussian mechanism (e.g., Abadi et al. 2016, Mironov 2017, Kulkarni et al. 2021), we have the following privacy guarantee:

Lemma 4.3 (Privacy). Calling Algorithm 4 T times with $m = \frac{L^2d\Sigma}{n\sigma_1^2} + \frac{L^2d}{n\sigma_2^2}$ , $\sigma_1 = O(\frac{L\sqrt{T\log(1/\delta)/\Sigma}}{n\varepsilon})$ and $\sigma_2 = O(\frac{LD\sqrt{Td\log(1/\delta)}}{\alpha n\varepsilon})$ is $(\varepsilon, \delta)$ -DP.

In terms of the oracle's variance, we show:

Lemma 4.4 (Variance). In Algorithm 4, for any $t \in [T]$ , we have

$$
\begin{array}{l} \mathbb {E} \left\| \tilde {g} _ {t} - \nabla F _ {\alpha} ^ {\mathcal {D}} (z _ {t}) \right\| ^ {2} \lesssim \frac {L ^ {2} d ^ {2} \Sigma}{m n} + \sigma_ {1} ^ {2} d + \sigma_ {2} ^ {2} d \Sigma , \\ \mathbb {E} \left\| \tilde {g} _ {t} \right\| ^ {2} \lesssim L ^ {2} + \frac {L ^ {2} d ^ {2} \Sigma}{m n} + \sigma_ {1} ^ {2} d + \sigma_ {2} ^ {2} d \Sigma . \\ \end{array}
$$

The proof of Theorem 4.1, which we defer to Section 7, is a combination of the two lemmas and Proposition 2.6.

# 5 Empirical to population Goldstein- stationarity

In this section, we provide a generalization result, showing that our ERM algorithm from the previous section also guarantees Goldstein-stationarity in terms of the population loss. We prove the following more general statement:

Proposition 5.1. Under Assumption 2.2, suppose $\mathcal{D} \sim \mathcal{P}^n$ , and consider running an algorithm on $\widehat{F}^{\mathcal{D}}$ whose (possibly randomized) output $x^{\mathrm{out}} \in \mathcal{X} \subset \mathbb{R}^d$ is supported over a set $\mathcal{X}$ of diameter $\leq R$ . Then with probability at least $1 - \zeta: \| \overline{\partial}_{\alpha} F(x^{\mathrm{out}}) \| \leq \| \overline{\partial}_{\alpha} \widehat{F}^{\mathcal{D}}(x^{\mathrm{out}}) \| + \widetilde{O}\left(L \sqrt{d \log(R / \zeta) / n}\right)$ .

We remark that in all algorithms of interest, the output is known to lie in some predefined set, such as a sufficiently large ball around the initialization. As long as the diameter $R$ is polynomial in the problem parameters, the $\log (R)$ in the result above is therefore negligible. For instance, Algorithm 2 is easily verified to output a point $x^{\mathrm{out}}\in \mathbb{B}(x_0,DT)$ (since $\| x_{t + 1} - x_t\| \leq D$ for all $t$ ). Hence, in our use case, Proposition 5.1 ensures $\| \overline{\partial}_{2\alpha}F(x^{\mathrm{out}})\| \leq \| \overline{\partial}_{2\alpha}\widehat{F}^{\mathcal{D}}(x^{\mathrm{out}})\| +\beta$ for $n = \widetilde{O} (d / \beta^{2})$ .

# 6 Improved efficiency with gradients

In this section, our goal is to show that the zero-order algorithms presented thus far can be replaced by first-order algorithms with the same sample complexity, and improved oracle complexity.

The idea is to replace the zero-order gradient estimator from Eq. (4) by the smoothed first-order estimator

$$
\tilde {\nabla} f _ {\alpha} (x; \xi) = \frac {1}{m} \sum_ {j = 1} ^ {m} \nabla f (x + \alpha y _ {j}; \xi) \tag {10}
$$

for $(y_{i})_{i = 1}^{m}\stackrel {iid}{\sim}\mathrm{Unif}(\mathbb{S}^{d - 1})$ . While this estimator has the same expectation as the zero-order variant, the key difference lies in the fact that its sub-Gaussian norm is substantially smaller, and in particular, it does not depend on $d$ . Hence, smaller $m$ suffices for similar concentration. This observation

Algorithm 5 First-order instantiation of $\mathcal{O}(z_t)$ in Line 7 of Algorithm 2   
1: Input: Current iterate $z_{t}$ , time $t \in N$ , period length $\Sigma \in N$ , accuracy parameter $\alpha > 0$ , batch sizes $B_{1}, B_{2} \in N$ , gradient validation size $m \in N$ , noise level $\sigma > 0$ .
2: if t mod $\Sigma = 1$ then
3: Sample minibatch $S_{t}$ of size $B_{1}$ from unused samples
4: Sample $y_{1}, \ldots, y_{B_{1}} \stackrel{iid}{\sim} \text{Unif}(\mathbb{B}_{\alpha})$ 5: $g_{t} = \frac{1}{B_{1}} \sum_{\xi_{i} \in S_{t}} \nabla f(z_{t} + y_{i}; \xi_{i})$ 6: else
7: Sample minibatch $S_{t}$ of size $B_{2}$ from unused samples
8: for each sample $\xi_{i} \in S_{t}$ do
9: Sample $y_{1}, \ldots, y_{2m} \stackrel{iid}{\sim} \text{Unif}(\mathbb{B}_{\alpha})$ 10: $\tilde{\nabla} f(z_{t}; \xi_{i}) = \frac{1}{m} \sum_{j=1}^{m} \nabla f(z_{t} + y_{j}; \xi_{i})$ 11: $\tilde{\nabla} f(z_{t-1}; \xi_{i}) = \frac{1}{m} \sum_{j=m+1}^{2m} \nabla f(z_{t-1} + y_{j}; \xi_{i})$ 12: end for
13: $g_{t} = g_{t-1} + \frac{1}{B_{2}} \sum_{\xi_{i} \in S_{t}} (\tilde{\nabla} f(z_{t}; \xi_{i}) - \tilde{\nabla} f(z_{t-1}; \xi_{i}))$ 14: end if
15: Return $\tilde{g}_{t} = g_{t} + \text{TREE}(\sigma, \Sigma)(t \mod \Sigma)$

enables reducing the oracle complexity, while ensuring the same sample complexity guarantee as earlier.

We fully analyze here a single-pass first-order oracle presented in Algorithm 5, which can be used in Algorithm 2, similarly to Section 3. We note that the same analysis can be applied to the multi-pass oracle of Section 4, once again by replacing (4) by (10), which we omit here for brevity. The main result in this section is the following:

Theorem 6.1 (First-order). Suppose $F(x_0) - \inf_x F(x) \leq \Phi$ , that Assumption 2.2 holds, and let $\alpha, \beta, \delta, \varepsilon > 0$ such that $\alpha \leq \frac{\Phi}{L}$ . Then setting $B_1 = \Sigma$ , $B_2 = 1$ , $M = \alpha / 4D$ , $m = \widetilde{O}(\frac{B_2^2\alpha^2}{D^2d})$ , $\sigma = \widetilde{O}(\frac{L}{B_1\varepsilon} + \frac{LD\sqrt{d}}{\alpha B_2\varepsilon})$ , $\Sigma = \widetilde{\Theta}((\frac{\alpha}{\varepsilon D})^{2/3} + \frac{\alpha}{Dd^{1/2}})$ , $D = \widetilde{\Theta}(\min\{(\frac{\Phi^2\alpha}{L^2T^2})^{1/3}, (\frac{\Phi\alpha\varepsilon}{dLT})^{1/2}, (\frac{\Phi^3\alpha^2\varepsilon}{d^3/2L^3T^3})^{1/5}, (\frac{\Phi^2\alpha}{L^2T^2\sqrt{d}})^{1/3}\})$ , $T = \Theta(n)$ , and running Algorithm 2 with Algorithm 5 as the oracle subroutine, is $(\varepsilon, \delta)$ -DP. Furthermore, its output satisfies $\mathbb{E}\| \overline{\partial}_{2\alpha}F(x^{\mathrm{out}})\| \leq \beta$ as long as

$$
n = \widetilde {\Omega} \left(\frac {\Phi L ^ {2} \sqrt {d}}{\alpha \beta^ {3}} + \frac {\Phi L d}{\varepsilon \alpha \beta^ {2}}\right).
$$

Remark 6.2. Compared to the analogous zero-order result given by Theorem 3.1, we see that the number of calls to $\mathcal{O}(\cdot)$ , namely T, is on the same order, and that in both cases the amortized oracle complexity of $\mathcal{O}(\cdot)$ is $O(m)$ . The difference between the settings is that the first-order oracle instantiation sets m to be $\widetilde{\Omega}(d^{2})$ times smaller than its zero-order counterpart, and hence we gain this multiplicative factor in the overall oracle complexity. It is interesting to compare this gain to non-private optimization, where the ratio between zero- and first-order oracle complexities is $\Theta(d)$ [Duchi et al., 2015, Kornowski and Shamir, 2024], whereas here we obtain an even larger gap in favor of gradient-based optimization.

As in Section 3, we will present the basic properties of this oracle. We will then plug these into Algorithm 2, leading to the main result of this section, Theorem 6.1. Corresponding proofs are deferred to Section 7.

Lemma 6.3 (Sensitivity). Consider the gradient oracle $\mathcal{O}(\cdot)$ in Algorithm 5 when acting on two neighboring minibatches $S_{t}$ and $S_{t}^{\prime}$ , and correspondingly producing $g_{t}$ and $g_{t}^{\prime}$ , respectively. If $t$ mod $\Sigma = 1$ , then

$$
\| g _ {t} - g _ {t} ^ {\prime} \| \leq \frac {L}{B _ {1}}.
$$

Otherwise, conditioned on $g_{t-1} = g'_{t-1}$ , we have with probability at least $1 - \delta/2$ :

$$
\| g _ {t} - g _ {t} ^ {\prime} \| \lesssim \frac {L \sqrt {d} D}{\alpha B _ {2}} + \frac {L \sqrt {\log (d B _ {2} / \delta)}}{\sqrt {m}}.
$$

With the sensitivity bound given by Lemma 6.3, we easily derive the privacy guarantee of our algorithm from the Tree Mechanism (Proposition 2.5).

Lemma 6.4 (Privacy). Running Algorithm 5 with $m = O(\log (dB_2 / \delta)\frac{B_2^2\alpha^2}{D^2d})$ and $\sigma = O\big(\frac{L\sqrt{\log(1 / \delta)}}{B_1\varepsilon} + \frac{LD\sqrt{d\log(1 / \delta)}}{\alpha B_2\varepsilon}\big)$ is $(\varepsilon ,\delta)$ -DP.

Proof. By Lemma 6.3 and our assignment of m, we know that with probability at least $1 - \delta/2$ , for any t, we have

$$
\| g _ {t} - g _ {t} ^ {\prime} \| \lesssim \frac {L}{B _ {1}} + \frac {L \sqrt {d} D}{\alpha B _ {2}}.
$$

Then the privacy guarantee follows from the Tree Mechanism (Proposition 2.5).

![](images/f1910ce52dca41924a47ef93a54326fc515ed1ee3bb7e18749aae444f8d4176d.jpg)

We next provide the required oracle variance bound.

Lemma 6.5 (Variance). In Algorithm 5, for all t it holds that

$$
\mathbb {E} \left\| \tilde {g} _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} \lesssim \frac {L ^ {2}}{B _ {1}} + \frac {L ^ {2} d D ^ {2} \Sigma}{\alpha^ {2} B _ {2}} + \sigma^ {2} d \log \Sigma + \frac {L ^ {2} \Sigma}{m B _ {2}},
$$

$$
\mathbb {E} \left\| \tilde {g} _ {t} \right\| ^ {2} \lesssim L ^ {2} + \frac {L ^ {2} d D ^ {2} \Sigma}{\alpha^ {2} B _ {2}} + \sigma^ {2} d \log \Sigma + \frac {L ^ {2} \Sigma}{m B _ {2}}.
$$

Having set up the required bounds, we can prove our main result for the first-order setting.

Proof of Theorem 6.1. The privacy guarantee follows directly from Lemma 6.4, by noting that our parameter assignment implies $B_{1}T/\Sigma + B_{2}T = O(n)$ , hence it allows letting $T = \Theta(n)$ while never re-using samples.

As to the sample complexity, note that our parameter assignment ensures that

$$
G _ {1} = O \left(G _ {0} + L\right),
$$

$$
G _ {0} = \widetilde {O} \left(\frac {L}{\sqrt {\Sigma}} + \frac {L D d ^ {1 / 2} \Sigma^ {1 / 2}}{\alpha} + \frac {L d ^ {1 / 2}}{\Sigma \varepsilon} + \frac {L D d}{\alpha \varepsilon}\right),
$$

similarly to (7) and (9) in the proof of Theorem 3.1. The rest of the proof is therefore exactly the same as for Theorem 3.1.

# 7 Proofs

# 7.1 Proofs from Section 3

Proof of Lemma 3.4. Note that for any $y \in \mathrm{Unif}(\mathbb{S}^{d-1}) : \| \frac{d}{2\alpha}(f(z + \alpha y; \xi) - f(z - \alpha y; \xi))y \| \leq Ld$ due to the Lipschitz assumption. Hence, for any $\xi \in S_t$ , by a standard sub-Gaussian bound (Theorem C.2) we have

$$
\operatorname * {P r} \left[ \| \tilde {\nabla} f (z _ {t}; \xi) - \nabla f _ {\alpha} (z _ {t}; \xi) \| \leq \frac {L d \sqrt {\log (8 d B _ {1} / \delta)}}{\sqrt {m}} \right] \geq 1 - \delta / 8 B _ {1}. \tag {11}
$$

If $t \bmod \Sigma = 1$ , then

$$
\begin{array}{l} \| g _ {t} - g _ {t} ^ {\prime} \| = \left\| \frac {1}{B _ {1}} (\sum_ {\xi \in S _ {t}} \tilde {\nabla} f (z _ {t}; \xi) - \sum_ {\xi^ {\prime} \in S _ {t} ^ {\prime}} \tilde {\nabla} f (z _ {t}; \xi^ {\prime})) \right\| \\ \leq \left\| \frac {1}{B _ {1}} (\sum_ {\xi \in S _ {t}} \tilde {\nabla} f (z _ {t}; \xi) - \nabla f _ {\alpha} (z _ {t}; \xi)) \right\| + \left\| \frac {1}{B _ {1}} (\sum_ {\xi \in S _ {t}} \nabla f _ {\alpha} (z _ {t}; \xi) - \sum_ {\xi^ {\prime} \in S _ {t} ^ {\prime}} \nabla f _ {\alpha} (z _ {t}; \xi^ {\prime})) \right\| \\ + \left\| \frac {1}{B _ {1}} (\sum_ {\xi^ {\prime} \in S _ {t} ^ {\prime}} \tilde {\nabla} f (z _ {t}; \xi^ {\prime}) - \nabla f _ {\alpha} (z _ {t}; \xi^ {\prime})) \right\|. \\ \end{array}
$$

Further note that $\|\frac{1}{B_{1}}(\sum_{\xi\in S_{t}}\nabla f_{\alpha}(z_{t};\xi)-\sum_{\xi^{\prime}\in S_{t}^{\prime}}\nabla f_{\alpha}(z_{t};\xi^{\prime}))\|\leq2L/B_{1}$ , hence by Equation (11) and the union bound,

$$
\operatorname * {P r} \left[ \left\Vert \frac {1}{B _ {1}} (\sum_ {\xi \in S _ {t}} \tilde {\nabla} f (z _ {t}; \xi) - \sum_ {\xi^ {\prime} \in S _ {t} ^ {\prime}} \tilde {\nabla} f (z _ {t}; \xi^ {\prime})) \right\Vert \geq \frac {2 L}{B _ {1}} + \frac {L d \sqrt {\log (8 d B _ {1} / \delta)}}{\sqrt {m}} \right] \leq 1 - \delta / 8,
$$

which proves the claim in the case when $t \mod \Sigma = 1$ . The other case follows from the same argument.

Proof of Lemma 3.5. By Lemma 3.4 and our assignment of m, we know that with probability at least $1 - \delta/2$ , the sensitivity of all t is bounded by $O\left(\frac{L}{B_{1}} + \frac{L\sqrt{dD}}{\alpha B_{2}}\right)$ , namely for all t :

$$
\| g _ {t} - g _ {t} ^ {\prime} \| \lesssim \frac {L}{B _ {1}} + \frac {L \sqrt {d} D}{\alpha B _ {2}}.
$$

Then the privacy guarantee follows from the Tree Mechanism (Proposition 2.5).

Proof of Lemma 3.6. First, note that by Proposition 2.5 and the facts that $\mathbb{E}[g_t] = \nabla F_\alpha(z_t)$ and $\| \nabla F_{\alpha}(z_t)\| \leq L$ , we get

$$
\mathbb {E} \left\| \tilde {g} _ {t} \right\| ^ {2} \lesssim \mathbb {E} \left\| g _ {t} \right\| ^ {2} + d \sigma^ {2} \log \Sigma \lesssim \mathbb {E} \left\| g _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} + L ^ {2} + d \sigma^ {2} \log \Sigma ,
$$

and also

$$
\mathbb {E} \left\| \tilde {g} _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} \lesssim \mathbb {E} \left\| \tilde {g} _ {t} - g _ {t} \right\| ^ {2} + \mathbb {E} \left\| g _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} \lesssim d \sigma^ {2} \log \Sigma + \mathbb {E} \left\| g _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2}.
$$

Therefore, we see that in order to obtain both claimed bounds, it suffices to bound $E\|g_{t}-\nabla F_{\alpha}(z_{t})\|^{2}$ . To that end, denote by $t_{0}\leq t$ the largest integer such that $t_{0}\mod\Sigma=1$ , and note that $t-t_{0}<\Sigma$ . Further denote $\Delta_{j}:=g_{j}-g_{j-1}$ . Then we have

$$
\begin{array}{l} \mathbb {E} \left\| g _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} = \mathbb {E} \left\| g _ {t _ {0}} + \sum_ {j = t _ {0} + 1} ^ {t} \Delta_ {j} - \left(\sum_ {j = t _ {0} + 1} ^ {t} (\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})) + \nabla F _ {\alpha} (z _ {t _ {0}})\right) \right\| ^ {2} \\ = \underbrace {\mathbb {E} \left\| g _ {t _ {0}} - \nabla F _ {\alpha} (z _ {t _ {0}}) \right\| ^ {2}} _ {(I)} + \sum_ {j = t _ {0} + 1} ^ {t} \underbrace {\mathbb {E} \left\| \Delta_ {j} - (\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})) \right\| ^ {2}} _ {(I I)}, \tag {12} \\ \end{array}
$$

where the last equality is due to the cross terms having zero mean. We further see that

$$
\begin{array}{l} (I) \lesssim \mathbb {E} \left\| g _ {t _ {0}} - \frac {1}{B _ {1}} \sum_ {\xi_ {i} \in S _ {t _ {0}}} \nabla f _ {\alpha} (z _ {t _ {0}}; \xi_ {i}) \right\| ^ {2} + \mathbb {E} \left\| \frac {1}{B _ {1}} \sum_ {\xi_ {i} \in S _ {t _ {0}}} \nabla f _ {\alpha} (z _ {t _ {0}}; \xi_ {i}) - \nabla F _ {\alpha} (z _ {t _ {0}}) \right\| ^ {2} \\ \lesssim \frac {L ^ {2} d ^ {2}}{B _ {1} m} + \frac {L ^ {2}}{B _ {1}}, \tag {13} \\ \end{array}
$$

as well as

$$
\begin{array}{l} (I I) = \mathbb {E} \left\| \frac {1}{B _ {2}} \sum_ {\xi_ {i} \in S _ {t}} (\tilde {\nabla} f (z _ {j}; \xi_ {i}) - \tilde {\nabla} f (z _ {j - 1}; \xi_ {i})) - (\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})) \right\| ^ {2} \\ = \frac {1}{B _ {2} ^ {2}} \sum_ {\xi_ {i} \in S _ {t}} \mathbb {E} \| (\tilde {\nabla} f (z _ {j}; \xi_ {i}) - \tilde {\nabla} f (z _ {j - 1}; \xi_ {i})) - (\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})) \| ^ {2} \\ \lesssim \frac {1}{B _ {2} ^ {2}} \sum_ {\xi_ {i} \in S _ {t}} \left(\mathbb {E} \| \tilde {\nabla} f (z _ {j}; \xi_ {i}) - \nabla f _ {\alpha} (z _ {j}; \xi_ {i}) \| ^ {2} + \mathbb {E} \| \tilde {\nabla} f (z _ {j - 1}; \xi_ {i}) - \nabla f _ {\alpha} (z _ {j - 1}; \xi_ {i}) \| ^ {2} \right. \\ \left. + \mathbb {E} \left\| \left(\nabla f _ {\alpha} \left(z _ {j}; \xi_ {i}\right) - \nabla f _ {\alpha} \left(z _ {j - 1}; \xi_ {i}\right)\right) - \left(\nabla F _ {\alpha} \left(z _ {j}\right) - \nabla F _ {\alpha} \left(z _ {j - 1}\right)\right) \right\| ^ {2}\right) \\ \lesssim \frac {L ^ {2} d ^ {2}}{m B _ {2}} + \frac {d L ^ {2} D ^ {2}}{\alpha^ {2} B _ {2}}. \tag {14} \\ \end{array}
$$

Plugging (13) and (14) into (12) and recalling that $t - t_0 < \Sigma$ completes the proof.

# 7.2 Proofs from Section 4

Proof of Lemma 4.4. First, it suffices to prove the first bound, as

$$
\mathbb {E} \left\| \tilde {g} _ {t} \right\| ^ {2} \lesssim \mathbb {E} \left\| \tilde {g} _ {t} - \nabla F _ {\alpha} ^ {\mathcal {D}} (z _ {t}) \right\| ^ {2} + \mathbb {E} \left\| \nabla F _ {\alpha} ^ {\mathcal {D}} (z _ {t}) \right\| ^ {2} \leq \mathbb {E} \left\| \tilde {g} _ {t} - \nabla F _ {\alpha} ^ {\mathcal {D}} (z _ {t}) \right\| ^ {2} + L ^ {2}.
$$

To that end, let $t_0 \leq t$ be the largest integer such that $t_0 \mod \Sigma \equiv 1$ , and note that $t - t_0 < \Sigma$ . Define $\Delta_j := \frac{1}{n} \sum_{\xi_i \in \mathcal{D}} (\tilde{\nabla} f(z_j; \xi_i) - \tilde{\nabla} f(z_{j-1}; \xi_i))$ . It holds that

$$
\mathbb {E} \left\| \tilde {g} _ {t} - \nabla F _ {\alpha} ^ {\mathcal {D}} (z _ {t}) \right\| ^ {2} \leq \underbrace {\mathbb {E} \left\| g _ {t _ {0}} - \nabla F _ {\alpha} ^ {\mathcal {D}} (z _ {t _ {0}}) \right\| ^ {2}} _ {(I)} + \sum_ {j = t _ {0}} ^ {t} \underbrace {\mathbb {E} \left\| \Delta_ {j} - (\nabla F _ {\alpha} ^ {\mathcal {D}} (z _ {j}) - \nabla F _ {\alpha} ^ {\mathcal {D}} (z _ {j - 1})) \right\| ^ {2}} _ {(I I)} + \underbrace {\sum_ {j = t _ {0}} ^ {t} \mathbb {E} \left\| \chi_ {j} \right\| ^ {2}} _ {(I I I)}.
$$

Similar to the proof of Lemma 3.6, we have that

$$
(I) = \mathbb {E} \left\| g _ {t _ {0}} - \frac {1}{n} \sum_ {\xi_ {i} \in \mathcal {D}} \nabla f _ {\alpha} (z _ {t _ {0}}; \xi_ {i}) \right\| ^ {2} \lesssim \frac {L ^ {2} d ^ {2}}{n m},
$$

$$
(I I) = \frac {1}{n ^ {2}} \mathbb {E} \| \sum_ {\xi_ {i} \in \mathcal {D}} (\tilde {\nabla} f (z _ {j}; \xi_ {i}) - \tilde {\nabla} f (z _ {j - 1}; \xi_ {i})) - (\nabla \widehat {F} _ {\alpha} ^ {\mathcal {D}} (z _ {j}) - \nabla \widehat {F} _ {\alpha} ^ {\mathcal {D}} (z _ {j - 1})) \| ^ {2}
$$

$$
\lesssim \frac {1}{n ^ {2}} \sum_ {\xi_ {i} \in \mathcal {D}} \left(\mathbb {E} \| \tilde {\nabla} f (z _ {j}; \xi_ {i}) - \nabla f _ {\alpha} (z _ {j}; \xi_ {i}) \| ^ {2} + \mathbb {E} \| \tilde {\nabla} f (z _ {j - 1}; \xi_ {i}) - \nabla f _ {\alpha} (z _ {j - 1}; \xi_ {i}) \| ^ {2}\right)
$$

$$
\lesssim \frac {L ^ {2} d ^ {2}}{m n},
$$

$$
(I I I) \leq d \sigma_ {1} ^ {2} + d \sigma_ {2} ^ {2} (\Sigma - 1),
$$

overall completing the proof.

Proof of Theorem 4.1. Setting $m = \frac{L^2d\Sigma}{n\sigma_1^2} +\frac{L^2d}{n\sigma_2^2}$ , $\sigma_{1} = O(\frac{L\sqrt{T\log(1 / \delta) / \Sigma}}{n\varepsilon})$ and $\sigma_{2} = O(\frac{LD\sqrt{Td\log(1 / \delta)}}{\alpha n\varepsilon})$ , the privacy guarantee follows from Lemma 4.3. Moreover, by our parameter settings, we have

$$
G _ {0} ^ {2} := \mathbb {E} \left\| \tilde {g} _ {t} - \nabla F _ {\alpha} ^ {\mathcal {D}} (z _ {t}) \right\| ^ {2} \lesssim \frac {L ^ {2} d T \log (1 / \delta) / \Sigma}{n ^ {2} \varepsilon^ {2}} + \frac {L ^ {2} D ^ {2} T d ^ {2} \Sigma \log (1 / \delta)}{\alpha^ {2} n ^ {2} \varepsilon^ {2}},
$$

$$
G _ {1} ^ {2} := \mathbb {E} \left\| \tilde {g} _ {t} \right\| ^ {2} \lesssim L ^ {2} + \frac {L ^ {2} d T \log (1 / \delta) / \Sigma}{n ^ {2} \varepsilon^ {2}} + \frac {L ^ {2} D ^ {2} T d ^ {2} \Sigma \log (1 / \delta)}{\alpha^ {2} n ^ {2} \varepsilon^ {2}}.
$$

Therefore, setting $\Sigma = \widetilde{\Theta} (\frac{\alpha}{D\sqrt{d}})$ , we see that $G_0 = \widetilde{O} (\frac{L\sqrt{DT}d^{3 / 4}}{n\varepsilon\sqrt{\alpha}})$ and $G_{1}\lesssim L + G_{0}$ . By Proposition 2.6, we also know that

$$
\mathbb {E} \left\| \overline {{\partial}} _ {2 \alpha} \widehat {F} ^ {\mathcal {D}} (x ^ {\mathrm{out}}) \right\| \leq \mathbb {E} \left\| \overline {{\partial}} _ {\alpha} \widehat {F} _ {\alpha} ^ {\mathcal {D}} (x ^ {\mathrm{out}}) \right\| \leq \frac {F _ {\alpha} (x _ {0}) - \inf F _ {\alpha}}{D T} + \frac {3 G _ {1}}{2 \sqrt {M}} + G _ {0}
$$

$$
\leq \frac {2 \Phi}{D T} + \frac {3 G _ {1}}{2 \sqrt {M}} + G _ {0}.
$$

Recalling that $M = \Theta(\alpha/D)$ and setting $D = \tilde{\Theta}(\frac{\alpha^2\beta^2}{L^2})$ , $T = \tilde{\Theta}(\frac{\Phi L^2}{\alpha^2\beta^3})$ , we have

$$
\mathbb {E} \left\| \overline {{\partial}} _ {\alpha} \widehat {F} _ {\alpha} ^ {\mathcal {D}} (x ^ {\mathrm{out}}) \right\| = \widetilde {O} \left(\frac {\Phi}{D T} + \frac {L \sqrt {D}}{\sqrt {\alpha}} + \frac {L \sqrt {D T} d ^ {3 / 4}}{n \varepsilon \sqrt {\alpha}}\right) = \frac {\beta}{2} + \widetilde {O} \left(\frac {L d ^ {3 / 4} \sqrt {\Phi}}{n \varepsilon \sqrt {\alpha \beta}}\right).
$$

The latter is bounded by $\beta$ for $n = \widetilde{\Omega}\left(\frac{L\sqrt{\Phi}d^{3 / 4}}{\varepsilon\alpha^{1 / 2}\beta^{3 / 2}}\right)$ , hence completing the proof.

# 7.3 Proofs from Section 5

Proof of Proposition 5.1. Applying a gradient uniform convergence bound for Lipschitz objectives over a bounded domain [Mei et al., 2018, Theorem 1], shows that with probability at least $1 - \zeta$ , for any differentiable $x \in \mathcal{X}$ :

$$
\left\| \nabla \widehat {F} ^ {\mathcal {D}} (x) - \nabla F (x) \right\| = \widetilde {O} \left(L \sqrt {\frac {d \log (R / \zeta)}{n}}\right). \tag {15}
$$

Therefore, given any $x \in X$ , let $y_{1}, \ldots, y_{k} \in \mathbb{B}(x, \alpha)$ be points satisfying $\overline{\partial}_{\alpha} \widehat{F}^{\mathcal{D}}(x) = \sum_{i=1}^{k} \lambda_{i} \nabla \widehat{F}^{\mathcal{D}}(y_{i})$ for coefficients $(\lambda_{i})_{i=1}^{k} \geq 0, \sum_{i=1}^{k} \lambda_{i} = 1$ — note that such points exist by definition of the Goldstein subdifferential. Noting that $\sum_{i=1}^{k} \lambda_{i} \nabla F(y_{i}) \in \partial_{\alpha} F(x)$ , and recalling that $\overline{\partial}_{\alpha} F(x)$ is the minimal norm element of $\partial_{\alpha} F(x)$ , we get that

$$
\left\| \overline {{\partial}} _ {\alpha} F (x) \right\| \leq \left\| \sum_ {i = 1} ^ {k} \lambda_ {i} \nabla F (y _ {i}) \right\| = \left\| \sum_ {i = 1} ^ {k} \lambda_ {i} (\nabla \widehat {F} ^ {\mathcal {D}} (y _ {i}) + v _ {i}) \right\| = (\star)
$$

where $v_{i} := \nabla F(y_{i}) - \nabla \widehat{F}^{\mathcal{D}}(y_{i})$ satisfy $\|v_{i}\| = \widetilde{O}\left(L\sqrt{\frac{d\log(R/\zeta)}{n}}\right)$ for all $i \in [k]$ by (15). Hence

$$
\begin{array}{l} (\star) \leq \left\| \sum_ {i = 1} ^ {k} \lambda_ {i} \nabla \widehat {F} ^ {\mathcal {D}} (y _ {i}) \right\| + \left\| \sum_ {i = 1} ^ {k} \lambda_ {i} v _ {i} \right\| \\ \leq \left\| \overline {{\partial}} _ {\alpha} \widehat {F} ^ {\mathcal {D}} (x) \right\| + \sum_ {i = 1} ^ {k} \lambda_ {i} \| v _ {i} \| \\ \leq \left\| \overline {{\partial}} _ {\alpha} \widehat {F} ^ {\mathcal {D}} (x) \right\| + \widetilde {O} \left(L \sqrt {\frac {d \log (R / \zeta)}{n}}\right). \\ \end{array}
$$

![](images/842cbd0c2e414e8381a2c679c176abb6774fb5beea029088dda9f8a585164baf.jpg)

# 7.4 Proofs from Section 6

Proof of Lemma 6.3. The case when $t$ mod $\Sigma = 1$ trivially follows the Lipschitz assumption. Thus we will consider the more involved case. For any $\xi \in S_t$ , by a standard sub-Gaussian bound (Theorem C.2) we have

$$
\operatorname * {P r} \left[ \| \tilde {\nabla} f (z _ {t}; \xi) - \nabla f _ {\alpha} (z _ {t}; \xi) \| \leq \frac {L \sqrt {\log (8 d B _ {2} / \delta)}}{\sqrt {m}} \right] \geq 1 - \delta / 8 B _ {2},
$$

so by the union bound, we get that with probability at least $1 - \delta/8$ , for all $\xi_{i} \in S_{t}$ :

$$
\left\| \tilde {\nabla} f \left(z _ {t}; \xi\right) - \nabla f _ {\alpha} \left(z _ {t}; \xi\right) \right\| \leq \frac {L \sqrt {\log \left(8 d B _ {2} / \delta\right)}}{\sqrt {m}}. \tag {16}
$$

Hence,

$$
\begin{array}{l} \left\| g _ {t} - g _ {t} ^ {\prime} \right\| \leq \left\| \frac {1}{B _ {2}} \sum_ {\xi \in S _ {t}} \left(\left(\tilde {\nabla} f (z _ {t}; \xi) - \tilde {\nabla} f (z _ {t - 1}; \xi_ {i})\right) - \left(\nabla f _ {\alpha} (z _ {t}; \xi)\right) - \nabla f _ {\alpha} (z _ {t - 1}; \xi)\right) \right\| \\ + \left\| \frac {1}{B _ {2}} \sum_ {\xi \in S _ {t}} \left((\nabla f _ {\alpha} (z _ {t}; \xi) - \nabla f _ {\alpha} (z _ {t - 1}; \xi)) - \sum_ {\xi^ {\prime} \in S _ {t} ^ {\prime}} (\nabla f _ {\alpha} (z _ {t}; \xi^ {\prime}) - \nabla f _ {\alpha} (z _ {t}; \xi^ {\prime}))\right) \right\| \\ + \left\| \frac {1}{B _ {2}} \sum_ {\xi^ {\prime} \in S _ {t} ^ {\prime}} \Big ((\tilde {\nabla} f (z _ {t}; \xi^ {\prime}) - \tilde {\nabla} f (z _ {t - 1}; \xi^ {\prime})) - (\nabla f _ {\alpha} (z _ {t}; \xi^ {\prime}) - \nabla f _ {\alpha} (z _ {t - 1}; \xi^ {\prime})) \Big) \right\| \\ \lesssim \frac {L \sqrt {d} D}{\alpha B _ {2}} + \frac {L \sqrt {\log (d B _ {2} / \delta)}}{\sqrt {m}}, \\ \end{array}
$$

where the last inequality step is due to the smoothness of $f_{\alpha}$ (Fact 2.3) combined with the fact that $\| z_t - z_{t-1}\| \leq 2D$ , and (16).

![](images/2e5e63d7c8040992fffa00cef51e087f0c0cac0fd11d7290aee266460dec03dc.jpg)

Proof of Lemma 6.5. Applying by Proposition 2.5, we have

$$
\mathbb {E} \left\| \tilde {g} _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} \lesssim \mathbb {E} \left\| \tilde {g} _ {t} - g _ {t} \right\| ^ {2} + \mathbb {E} \left\| g _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} \lesssim d \sigma^ {2} \log \Sigma + \mathbb {E} \left\| g _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2},
$$

and also since $\mathbb{E}[g_t] = \nabla F_\alpha (z_t)$ and $\| \nabla F_{\alpha}(z_{t})\| \leq L$ , we have

$$
\mathbb {E} \left\| \tilde {g} _ {t} \right\| ^ {2} \lesssim \mathbb {E} \left\| g _ {t} \right\| ^ {2} + d \sigma^ {2} \log \Sigma \lesssim \mathbb {E} \left\| g _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} + L ^ {2} + d \sigma^ {2} \log \Sigma .
$$

We therefore see that both claimed bounds will follow from bounding $\mathbb{E}\left\| g_t - \nabla F_\alpha (z_t)\right\|^2$ .

To that end, denote by $t_{0} \leq t$ the largest integer such that $t_{0} \mod \Sigma = 1$ , and note that $t - t_{0} < \Sigma$ . Further denote $\Delta_{j} := g_{j} - g_{j-1}$ . Then we have

$$
\begin{array}{l} \mathbb {E} \left\| g _ {t} - \nabla F _ {\alpha} (z _ {t}) \right\| ^ {2} = \mathbb {E} \left\| g _ {t _ {0}} + \sum_ {j = t _ {0} + 1} ^ {t} \Delta_ {j} - \left(\sum_ {j = t _ {0} + 1} ^ {t} (\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})) + \nabla F _ {\alpha} (z _ {t _ {0}})\right) \right\| ^ {2} \\ = \mathbb {E} \left\| g _ {t _ {0}} - \nabla F _ {\alpha} (z _ {t _ {0}}) \right\| ^ {2} + \sum_ {j = t _ {0}} ^ {t} \mathbb {E} \left\| \Delta_ {j} - (\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})) \right\| ^ {2}, \\ \lesssim \frac {L ^ {2}}{B _ {1}} + \sum_ {j = t _ {0}} ^ {t} \underbrace {\mathbb {E} \left\| \Delta_ {j} - \left(\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})\right) \right\| ^ {2}} _ {(\star)} \tag {17} \\ \end{array}
$$

where the second equality is due to the cross terms having zero mean. Moreover, we have

$$
\begin{array}{l} (\star) = \mathbb {E} \left\| \frac {1}{B _ {2}} \sum_ {\xi \in S _ {t}} (\tilde {\nabla} f (z _ {j}; \xi) - \tilde {\nabla} f (z _ {j - 1}; \xi)) - (\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})) \right\| ^ {2} \\ = \frac {1}{B _ {2} ^ {2}} \sum_ {\xi \in S _ {t}} \mathbb {E} \left\| (\tilde {\nabla} f (z _ {j}; \xi) - \tilde {\nabla} f (z _ {j - 1}; \xi)) - (\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})) \right\| ^ {2} \\ \lesssim \frac {1}{B _ {2} ^ {2}} \sum_ {\xi \in S _ {t}} \Big (\mathbb {E} \| \tilde {\nabla} f (z _ {j}; \xi) - \nabla f _ {\alpha} (z _ {j}; \xi) \| ^ {2} + \mathbb {E} \| \tilde {\nabla} f (z _ {j - 1}; \xi) - \nabla f _ {\alpha} (z _ {j - 1}; \xi) \| ^ {2} \\ \left. + \mathbb {E} \left\| (\nabla f _ {\alpha} (z _ {j}; \xi) - \nabla f _ {\alpha} (z _ {j - 1}; \xi)) - (\nabla F _ {\alpha} (z _ {j}) - \nabla F _ {\alpha} (z _ {j - 1})) \right\|\right) ^ {2} \\ \lesssim \frac {L ^ {2}}{m B _ {2}} + \frac {d L ^ {2} D ^ {2}}{\alpha^ {2} B _ {2}}, \\ \end{array}
$$

which plugged into (17) completes the proof by recalling that $t - t_{0} \leq \Sigma$ .

![](images/6cded681b44f87a8bf4d669439cbb7bdac23395f6b8f714a2bc91f3900e2fc9e.jpg)

# 8 Discussion

In this paper, we studied nonsmooth nonconvex optimization, and proposed differentially private algorithms for this task which return Goldstein-stationary points, improving the previously known sample complexity for this task.

Our single-pass algorithm reduces the sample complexity by a multiplicative $\Omega(\sqrt{d})$ factor compared to the previous such result by Zhang et al. [2024]. Furthermore, our result has a sublinear dimension-dependent “non-private” term, which was previously claimed impossible. Moreover, we propose a multi-pass algorithm which preforms sample-efficient ERM with sublinear dimension dependence, and show that it further generalizes to the population.

It is interesting to note that our guarantees are in terms of so-called “approximate” $(\varepsilon,\delta)$ -DP, whereas Zhang et al. [2024] derive a Rényi-DP guarantee [Mironov, 2017]. This is in fact inherent to our techniques, since we condition on a highly probable event in order to substantially decrease the effective sensitivity of our gradient estimators. Further examining this potential gap between approximate- and Rényi-DP for nonsmooth nonconvex optimization is an interesting direction for future research.

Another important problem that remains open is establishing tight lower bounds for DP non-convex optimization and perhaps further improving the sample complexities obtained in this paper. We note that the current upper and lower bounds do not fully match even in the smooth setting. In Appendix B, we provide evidence that our upper bound can be further improved, by proposing a computationally-inefficient algorithm, which converges to a relaxed notion of stationarity, using even fewer samples than the algorithms we presented in this work.

# Acknowledgements

The authors would like to thank the anonymous reviewers for their helpful suggestions, and in particular, for spotting a miscalculation in the previous version of this work. GK is supported by an Azrieli Foundation graduate fellowship.

# References

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pages 308–318, 2016.   
Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. Lower bounds for non-convex stochastic optimization. Mathematical Programming, 199(1):165–214, 2023.   
Raman Arora, Raef Bassily, Tomás González, Cristóbal A Guzmán, Michael Menart, and Enayat Ullah. Faster rates of convergence to stationary points in differentially private optimization. In International Conference on Machine Learning, pages 1060–1092. PMLR, 2023.   
Hédy Attouch and Dominique Aze. Approximation and regularization of arbitrary functions in hilbert spaces by the lasry-lions method. In Annales de l'Institut Henri Poincaré C, Analyse non linéaire, volume 10, 3, pages 289–312. Elsevier, 1993.   
Raef Bassily, Adam Smith, and Abhradeep Thakurta. Private empirical risk minimization: Efficient algorithms and tight error bounds. In 2014 IEEE 55th annual symposium on foundations of computer science, pages 464–473. IEEE, 2014.   
Raef Bassily, Vitaly Feldman, Kunal Talwar, and Abhradeep Guha Thakurta. Private stochastic convex optimization with optimal rates. Advances in neural information processing systems, 32, 2019.

Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points i. Mathematical Programming, 184(1):71–120, 2020.   
Yair Carmon, Arun Jambulapati, Yujia Jin, Yin Tat Lee, Daogao Liu, Aaron Sidford, and Kevin Tian. Resqueing parallel and private stochastic convex optimization. In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS), pages 2031–2058. IEEE, 2023.   
T-H Hubert Chan, Elaine Shi, and Dawn Song. Private and continual release of statistics. ACM Transactions on Information and System Security (TISSEC), 14(3):1–24, 2011.   
F. H. Clarke. Optimization and Nonsmooth Analysis. SIAM, 1990.   
Ashok Cutkosky, Harsh Mehta, and Francesco Orabona. Optimal stochastic non-smooth non-convex optimization through online-to-non-convex conversion. In International Conference on Machine Learning, pages 6643–6670. PMLR, 2023.   
Damek Davis, Dmitriy Drusvyatskiy, Yin Tat Lee, Swati Padmanabhan, and Guanghao Ye. A gradient sampling method with complexity guarantees for lipschitz functions in high and low dimensions. Advances in neural information processing systems, 35:6692–6703, 2022.   
John C Duchi, Peter L Bartlett, and Martin J Wainwright. Randomized smoothing for stochastic optimization. SIAM Journal on Optimization, 22(2):674–701, 2012.   
John C Duchi, Michael I Jordan, Martin J Wainwright, and Andre Wibisono. Optimal rates for zero-order convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory, 61(5):2788–2806, 2015.   
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography Conference, pages 265–284. Springer, 2006.   
Cynthia Dwork, Moni Naor, Toniann Pitassi, and Guy N Rothblum. Differential privacy under continual observation. In Proceedings of the forty-second ACM symposium on Theory of computing, pages 715–724, 2010.   
Cong Fang, Chris Junchi Li, Zhouchen Lin, and Tong Zhang. Spider: Near-optimal non-convex optimization via stochastic path-integrated differential estimator. Advances in neural information processing systems, 31, 2018.   
Vitaly Feldman, Tomer Koren, and Kunal Talwar. Private stochastic convex optimization: optimal rates in linear time. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing, pages 439–449, 2020.   
Abraham D Flaxman, Adam Tauman Kalai, and H Brendan McMahan. Online convex optimization in the bandit setting: gradient descent without a gradient. In Proceedings of the sixteenth annual ACM-SIAM symposium on Discrete algorithms, pages 385–394, 2005.   
Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM journal on optimization, 23(4):2341–2368, 2013.   
Allen A Goldstein. Optimization of lipschitz continuous functions. Mathematical Programming, 13: 14-22, 1977.   
Sivakanth Gopi, Yin Tat Lee, and Daogao Liu. Private convex optimization via exponential mechanism. In Conference on Learning Theory, pages 1948–1989. PMLR, 2022.

Benjamin Grimmer and Zhichao Jia. Goldstein stationarity in lipschitz constrained optimization. Optimization Letters, pages 1–11, 2024.   
Elad Hazan. Introduction to online convex optimization. Foundations and Trends® in Optimization, 2(3-4):157–325, 2016.   
Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M Kakade, and Michael I Jordan. A short note on concentration inequalities for random vectors with subgaussian norm. arXiv preprint arXiv:1902.03736, 2019.   
Michael Jordan, Guy Kornowski, Tianyi Lin, Ohad Shamir, and Manolis Zampetakis. Deterministic nonsmooth nonconvex optimization. In The Thirty Sixth Annual Conference on Learning Theory, pages 4570–4597. PMLR, 2023.   
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In International Conference on Learning Representations, 2017.   
Siyu Kong and AS Lewis. The cost of nonconvexity in deterministic nonsmooth optimization. Mathematics of Operations Research, 2023.   
Guy Kornowski and Ohad Shamir. Oracle complexity in nonsmooth nonconvex optimization. Journal of Machine Learning Research, 23(314):1–44, 2022.   
Guy Kornowski and Ohad Shamir. An algorithm with optimal dimension-dependence for zero-order nonsmooth nonconvex stochastic optimization. Journal of Machine Learning Research, 25(122):1–14, 2024.   
Janardhan Kulkarni, Yin Tat Lee, and Daogao Liu. Private non-smooth erm and sco in subquadratic steps. Advances in Neural Information Processing Systems, 34:4053–4064, 2021.   
Jean-Michel Lasry and Pierre-Louis Lions. A remark on regularization in hilbert spaces. Israel Journal of Mathematics, 55:257-266, 1986.   
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436–444, 2015.   
Tianyi Lin, Zeyu Zheng, and Michael Jordan. Gradient-free methods for deterministic and stochastic nonsmooth nonconvex optimization. Advances in Neural Information Processing Systems, 35:26160–26175, 2022.   
Daogao Liu, Arun Ganesh, Sewoong Oh, and Abhradeep Guha Thakurta. Private (stochastic) non-convex optimization revisited: Second-order stationary points and excess risks. Advances in Neural Information Processing Systems, 36, 2024.   
Andrew Lowy, Jonathan Ullman, and Stephen Wright. How to make the gradients small privately: Improved rates for differentially private non-convex optimization. In Forty-first International Conference on Machine Learning, 2024.   
Song Mei, Yu Bai, and Andrea Montanari. The landscape of empirical risk for nonconvex losses. The Annals of Statistics, 46(6A):2747–2774, 2018.   
Ilya Mironov. Rényi differential privacy. In 2017 IEEE 30th computer security foundations symposium (CSF), pages 263–275. IEEE, 2017.

Ohad Shamir. An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. Journal of Machine Learning Research, 18(52):1–11, 2017.   
Lai Tian and Anthony Man-Cho So. No dimension-free deterministic algorithm computes approximate stationarities of lipschitzians. Mathematical Programming, pages 1–24, 2024.   
Pablo Villalobos, Anson Ho, Jaime Sevilla, Tamay Besiroglu, Lennart Heim, and Marius Hobbhahn. Position: Will we run out of data? limits of llm scaling based on human-generated data. In Forty-first International Conference on Machine Learning, 2024.   
Di Wang, Minwei Ye, and Jinhui Xu. Differentially private empirical risk minimization revisited: Faster and more general. Advances in Neural Information Processing Systems, 30, 2017.   
Di Wang, Changyou Chen, and Jinhui Xu. Differentially private empirical risk minimization with non-convex loss functions. In International Conference on Machine Learning, pages 6526–6535. PMLR, 2019.   
Farzad Yousefian, Angela Nedić, and Uday V Shanbhag. On stochastic gradient and subgradient methods with adaptive steplength sequences. Automatica, 48(1):56–67, 2012.   
Jingzhao Zhang, Hongzhou Lin, Stefanie Jegelka, Suvrit Sra, and Ali Jadbabaie. Complexity of finding stationary points of nonconvex nonsmooth functions. In International Conference on Machine Learning, pages 11173–11182. PMLR, 2020.   
Qinzi Zhang, Hoang Tran, and Ashok Cutkosky. Private zeroth-order nonsmooth nonconvex optimization. In The Twelfth International Conference on Learning Representations, 2024.

# A Proof of Proposition 2.6 (O2NC)

We start by noting that the update rule for $\Delta_{t}$ which is given by

$$
\Delta_ {t + 1} = \min \left\{1, \frac {D}{\| \Delta_ {t} - \eta \tilde {g} _ {t} \|} \right\} \cdot (\Delta_ {t} - \eta \tilde {g} _ {t})
$$

is precisely the online project gradient descent update rule, with respect to linear losses of the form $\ell_{t}(\cdot)=\langle\tilde{g}_{t},\cdot\rangle$ , over the ball of radius D around the origin. Accordingly, recalling that $E\|\tilde{g}_{t}-\nabla h(z_{t})\|^{2}\leq G_{1}^{2}$ , combining the linearity of expectation with the standard regret analysis of online linear optimization (cf. Hazan, 2016) gives the following:

Lemma A.1. By setting $\eta=\frac{D}{G_{1}\sqrt{M}}$ , for any $u\in R^{d}$ with $\|u\|\leq D$ it holds that

$$
\underset {\tilde {g} _ {1}, \dots , \tilde {g} _ {M}} {\mathbb {E}} \left[ \sum_ {m = 1} ^ {M} \langle \tilde {g} _ {m}, \Delta_ {m} - u \rangle \right] \leq \frac {3}{2} D G _ {1} \sqrt {M}.
$$

Back to analyzing Algorithm 2, since $x_{t} = x_{t - 1} + \Delta_t$ it holds that

$$
\begin{array}{l} h (x _ {t}) - h (x _ {t - 1}) = \int_ {0} ^ {1} \langle \nabla h (x _ {t - 1} + s \Delta_ {t}), \Delta_ {t} \rangle d s \\ = \underset {s _ {t} \sim \mathrm{Unif} [ 0, 1 ]} {\mathbb {E}} \left[ \langle \nabla h (x _ {t - 1} + s _ {t} \Delta_ {t}), \Delta_ {t} \rangle \right] = \underset {s _ {t}} {\mathbb {E}} \left[ \langle \nabla h (z _ {t}), \Delta_ {t} \rangle \right]. \\ \end{array}
$$

Note that $\langle \nabla h(z_t),\Delta_t\rangle = \langle \nabla h(z_t),u\rangle +\langle \tilde{g}_t,\Delta_t - u\rangle +\langle \nabla h(z_t) - \tilde{g}_t,\Delta_t - u\rangle$ , so by summing over $t\in [T] = [K\times M]$ , we get for any fixed sequence $u_{1},\ldots ,u_{K}\in \mathbb{R}^{d}$ :

$$
\begin{array}{l} \inf h \leq h (x _ {T}) \leq h (x _ {0}) + \sum_ {t = 1} ^ {T} \mathbb {E} \left[ \langle \nabla h (z _ {t}), \Delta_ {t} \rangle \right] \\ = h \left(x _ {0}\right) + \sum_ {k = 1} ^ {K} \sum_ {m = 1} ^ {M} \mathbb {E} \left[ \left\langle \tilde {g} _ {(k - 1) M + m}, \Delta_ {(k - 1) M + m} - u _ {k} \right\rangle \right] \\ + \sum_ {k = 1} ^ {K} \sum_ {m = 1} ^ {M} \mathbb {E} \left[ \left\langle \nabla h (z _ {(k - 1) M + m}), u _ {k} \right\rangle \right] + \sum_ {t = 1} ^ {T} \mathbb {E} [ \left\langle \nabla h (z _ {t}) - \tilde {g} _ {t}, \Delta_ {t} - u \right\rangle ] \\ \leq h (x _ {0}) + \frac {3}{2} K D G _ {1} \sqrt {M} + \sum_ {k = 1} ^ {K} \sum_ {m = 1} ^ {M} \mathbb {E} \left[ \left\langle \nabla h (z _ {(k - 1) M + m}), u _ {k} \right\rangle \right] + G _ {0} D T, \\ \end{array}
$$

where the last inequality follows from applying Lemma A.1 to each $M$ consecutive iterates, and combining the bias bound $\mathbb{E}\left\| \tilde{g}_t - \nabla h(z_t)\right\| \leq G_0$ with Cauchy-Schwarz.

Letting $u_{k} := -D\frac{\sum_{m=1}^{M}\nabla h(z_{(k-1)M+m})}{\left\|\sum_{m=1}^{M}\nabla h(z_{(k-1)M+m})\right\|}$ , rearranging and dividing by $DT = DKM$ , we obtain

$$
\frac {1}{K} \sum_ {k = 1} ^ {K} \mathbb {E} \left\| \frac {1}{M} \sum_ {m = 1} ^ {M} \nabla h (z _ {(k - 1) M + m}) \right\| \leq \frac {h (x _ {0}) - \inf h}{D T} + \frac {3 G _ {1}}{2 \sqrt {M}} + G _ {0}. \tag {18}
$$

Finally, note that for all $k \in [K]$ , $m \in [M]$ : $\left\|z_{(k-1)M+m} - \overline{x}_{k}\right\| \leq MD \leq \alpha$ since the clipping operation ensures each iterate is at most of distance D to its predecessor, and therefore $\nabla h(z_{(k-1)M+m}) \in \partial_{\alpha}h(\overline{x}_{k})$ . Since the set $\partial_{\alpha}h(\cdot)$ is convex by definition, we further see that

$$
\frac {1}{M} \sum_ {m = 1} ^ {M} \nabla h (z _ {(k - 1) M + m}) \in \partial_ {\alpha} h (\overline {{x}} _ {k}),
$$

and hence by (18) we get

$$
\mathbb {E} \left\| \overline {{\partial}} _ {\alpha} h (x ^ {\mathrm{out}}) \right\| = \frac {1}{K} \sum_ {k = 1} ^ {K} \mathbb {E} \left\| \overline {{\partial}} _ {\alpha} h (\overline {{x}} _ {k}) \right\| \leq \frac {h (x _ {0}) - \inf h}{D T} + \frac {3 G _ {1}}{2 \sqrt {M}} + G _ {0}.
$$

# B Even better sample complexity via optimal smoothing

In this Appendix, our aim is to provide evidence that the sample complexities of NSNC DP optimization obtained in our work are likely improvable, at least with a computationally inefficient method. This approach is inspired by Lowy et al. [2024], which in the context of smooth optimization, showed significant sample complexity gains using algorithms with exponential runtime. As we will show, a similar phenomenon might hold for nonsmooth optimization. To that end, we propose a slight relaxation of Goldstein-stationarity, and show it can be achieved using less samples via an exponential time algorithm.

# B.1 Relaxation of Goldstein-stationarity

Recall that $x \in R^{d}$ is called an $(\alpha, \beta)$ -Goldstein stationary point of an objective $F(x) = \mathbb{E}_{\xi}[f(x; \xi)]$ if there exist $y_{1}, \ldots, y_{k} \in \mathbb{B}(x, \alpha)$ and convex coefficients $(\lambda_{i})_{i=1}^{k}$ so that $\|\sum_{i \in [k]} \lambda_{i} E_{\xi}[\nabla f(y_{i}; \xi)]\| \leq \beta$ . Arguably, the two most important properties satisfied by this definition are that

(i) If $f(x; \xi)$ are $L$ -smooth, any $(\alpha, \beta)$ -stationary point is $O(\alpha + \beta)$ -stationary.   
(ii) If $\left\| \overline{\partial}_{\alpha} F(x) \right\| \neq 0$ , then $F\left(x - \frac{\alpha}{\left\| \overline{\partial}_{\alpha} F(x) \right\|} \overline{\partial}_{\alpha} F(x)\right) \leq F(x) - \alpha \left\| \overline{\partial}_{\alpha} F(x) \right\|$ .

The first property shows that Goldstein-stationarity reduces to (“classic”) stationarity under smoothness. The second, known as Goldstein’s descent lemma [Goldstein, 1977], is a generalization of the classic descent lemma for smooth functions.

It is easy to see that Goldstein-stationarity is equivalent to the existence of a distribution P supported over $\mathbb{B}(x,\alpha)$ , such that $\|\mathbb{E}_{\xi,y\sim P}[\nabla f(y;\xi)]\| \leq \beta$ . We will now define a relaxation of Goldstein-stationarity that is easily verified to satisfy both of the aforementioned properties.

Definition B.1. We call a point $x \in \mathbb{R}^d$ an $(\alpha, \beta)$ -component-wise Goldstein-stationary point of $F(x) = \mathbb{E}_{\xi}[f(x; \xi)]$ if there exist distributions $P_{\xi}$ supported over $\mathbb{B}(x, \alpha)$ , such that $\| \mathbb{E}_{\xi, y \sim P_{\xi}}[\nabla f(y; \xi)] \| \leq \beta$ .

In other words, the definition above allows the sampled points $y_{1},\ldots,y_{k}$ in the vicinity of x to vary for different components, and as before, the sampled gradient must have small expected norm. We next show that this relaxed stationarity notion allows improving the sample complexity of DP NSNC optimization.

# B.2 Optimal smoothing and faster algorithm

In the previous sections, given an objective f, we used the fact that Goldstein-stationary points of the randomized smoothing $f_{\alpha}$ correspond to Goldstein-stationary point of f, and therefore constructed private gradient oracles of $f_{\alpha}$ , which is $O(\sqrt{d}/\alpha)$ -smooth. Consequently, the sensitivity of the gradient oracle had a $\sqrt{d}$ dimension dependence (as seen in Lemma 3.4), thus affecting the overall sample complexity.

Instead of randomized smoothing, we now consider the Lasry-Lions (LL) smoothing [Lasry and Lions, 1986], a method that smooths Lipschitz functions in a dimension independent manner, which we now recall. Given $h: \mathbb{R}^d \to \mathbb{R}$ , denote the so-called Moreau envelope

$$
M _ {\lambda} (h) (x) := \min _ {y} \left[ h (y) + \frac {1}{2 \lambda} \| y - x \| ^ {2} \right],
$$

and the Lasry-Lions smoothing:

$$
\tilde {h} _ {\lambda \mathrm{LL}} (x) := - M _ {\lambda} (- M _ {2 \lambda} (h)) (x) = \max _ {z} \min _ {y} \left[ h (z) + \frac {1}{4 \lambda} \| z - y \| ^ {2} - \frac {1}{2 \lambda} \| y - x \| ^ {2} \right]. \tag {19}
$$

Fact B.2. [Lasry and Lions, 1986, Attouch and Aze, 1993] Suppose $h: \mathbb{R}^d \to \mathbb{R}$ is $L$ -Lipschitz. Then: (i) $\tilde{h}_{\lambda \mathrm{LL}}$ is $L$ -Lipschitz; (ii) $|\tilde{h}_{\lambda \mathrm{LL}}(x) - h(x)| \leq L\lambda$ for any $x \in \mathbb{R}^d$ ; (iii) $\arg \min \tilde{h}_{\lambda \mathrm{LL}} = \arg \min h$ ; (iv) $\tilde{h}_{\lambda \mathrm{LL}}$ is $O(L / \lambda)$ -smooth.

The key difference between LL-smoothing and randomized smoothing is that the smoothness constant of LL-smoothing is dimension independent. By solving the optimization problem in (19), it is clear that the values, and therefore gradients, of $\tilde{f}_{\lambda\mathrm{LL}}(x;\xi_{i})$ can be obtained up to arbitrarily high accuracy. Notably, it was shown by Kornowski and Shamir [2022] that solving this problem requires, in general, an exponential number of oracle calls to the original function.

Nonetheless, computational considerations aside, a priori it is not even clear that the LL smoothing can help finding Goldstein-stationary points of the original function, which was previously shown for randomized smoothing (Lemma 2.4). This is the purpose of the following result, which we prove:

Lemma B.3. If $h$ is $L$ -Lipschitz, then any $\beta$ -stationary point of $\tilde{h}_{\lambda \mathrm{LL}}$ is a $(3\lambda L, \beta)$ -Goldstein stationary point of $h$ .

Given the lemma above (which we prove later in this appendix), we are able to utilize smooth algorithms for finding stationary points, and convert the guarantee to Goldstein-stationary points of our objective of interest. Specifically, we will invoke the following result.

Proposition B.4 (Lowy et al., 2024). Given an ERM objective $\tilde{F}(x) = \frac{1}{n}\sum_{i=1}^{n}\tilde{f}(x;\xi_i)$ with $L_0$ -Lipschitz and $L_1$ -smooth components, and an initial point $x_0 \in \mathbb{R}^d$ such that $\mathrm{dist}(x_0, \arg \min \tilde{F}) \leq R$ , there's an $(\varepsilon, \delta)$ -DP algorithm that returns $\tilde{x}^{\mathrm{out}}$ with $\mathbb{E}\left\|\nabla\tilde{F}(\tilde{x}^{\mathrm{out}})\right\| = \widetilde{O}\left(\frac{R^{1/3}L_0^{2/3}L_1^{1/3}d^{2/3}}{n\varepsilon} + \frac{L_0\sqrt{d}}{n\varepsilon}\right)$ .

We remark that we assume for simplicity that $\text{dist}(x_{0}, \arg\min \widehat{F}^{\mathcal{D}}) = \text{dist}(x_{0}, \arg\min \tilde{F}) \leq R$ , though the analysis extends to that case where R is the initial distance to a point with sufficiently small loss (e.g., if the infimum is not attained). Overall, by setting $\lambda = \alpha/3L$ , and combining Fact B.2, Lemma B.3 and Proposition B.4, we get the following:

Theorem B.5. Under Assumption 2.2, suppose $\mathrm{dist}(x_0, \arg \min \widehat{F}^{\mathcal{D}}) \leq R$ . Then there is an $(\varepsilon, \delta)$ -DP algorithm that outputs $x^{\mathrm{out}}$ satisfying $(\alpha, \beta)$ -component-wise Goldstein-stationarity (in expectation) as long as

$$
n = \widetilde {\Omega} \left(\frac {R ^ {1 / 3} L ^ {4 / 3} d ^ {2 / 3}}{\varepsilon \alpha^ {1 / 3} \beta}\right).
$$

# Proof of Lemma B.3

Suppose $x$ is a $\beta$ -stationary point of $\tilde{h}_{\lambda \mathrm{LL}}$ . Let $z^{*} \in \mathbb{R}^{d}$ be the solution of the maximization problem defining the LL smoothing. By [Attouch and Aze, 1993, Remark 4.3.e], $z^{*}$ is uniquely defined, and satisfies

$$
\nabla \tilde {h} _ {\lambda \mathrm{LL}} (x) \in \partial (M _ {2 \lambda} (h)) (z ^ {*}). \tag {20}
$$

Further denote $Y^{*} := \arg\min_{y} \left[ h(y) + \frac{1}{4\lambda} \|z^{*} - y\|^{2} \right] \subseteq R^{d}$ . Rearranging the definition of the Moreau envelope by expanding the square, we see that

$$
M _ {2 \lambda} (h) (z ^ {*}) = \frac {1}{4 \lambda} \| z ^ {*} \| ^ {2} - \frac {1}{2 \lambda} \max _ {y} \left[ \langle z ^ {*}, y \rangle - 2 \lambda h (y) - \frac {1}{2} \| y \| ^ {2} \right],
$$

from which we get

$$
\partial M _ {2 \lambda} h (z ^ {*}) = \frac {1}{2 \lambda} z ^ {*} - \frac {1}{2 \lambda} \text {conv} \left\{y ^ {*}: y ^ {*} \in \mathcal {Y} ^ {*} \right\} = \text {conv} \left\{\frac {1}{2 \lambda} (z ^ {*} - y ^ {*}): y ^ {*} \in \mathcal {Y} ^ {*} \right\}. \tag {21}
$$

Furthermore, for any $y^{*}\in \mathcal{Y}^{*}$ , by first-order optimality it holds that

$$
0 \in \partial \left[ h (y ^ {*}) + \frac {1}{4 \lambda} \| y ^ {*} - z ^ {*} \| ^ {2} \right] \subseteq \partial h (y ^ {*}) + \frac {1}{2 \lambda} (y ^ {*} - z ^ {*}),
$$

and therefore

$$
\frac {1}{2 \lambda} (z ^ {*} - y ^ {*}) \in \partial h (y ^ {*}). \tag {22}
$$

By combining (20), (21) and (22) we conclude that

$$
\nabla \tilde {h} _ {\lambda \mathrm{LL}} (x) \in \partial M _ {2 \lambda} h (z ^ {*}) \subseteq \operatorname{conv} \left\{\partial h (y ^ {*}): y ^ {*} \in \mathcal {Y} ^ {*} \right\} \subseteq \partial_ {r} h (x),
$$

where the last holds for $r := \max_{y^* \in \mathcal{Y}^*} \| x - y^* \|$ . Therefore, recalling that $\| \nabla \tilde{h}_{\lambda \mathrm{LL}}(x) \| \leq \beta$ , all that remains is to bound $r$ .

To that end, it clearly holds that $r \leq \|x - z^{*}\| + \max_{y^{*} \in \mathcal{Y}^{*}} \|z^{*} - y^{*}\|$ . Furthermore, by [Attouch and Aze, 1993, Remark 4.3.e] it holds that $z^{*} - x = \lambda \nabla \tilde{h}_{\lambda LL}(x)$ which implies $\|x - z^{*}\| = \lambda \beta$ . As to the second summand, by (21) it holds that $\max_{y^{*} \in \mathcal{Y}^{*}} \|z^{*} - y^{*}\| \leq 2\lambda \cdot \max_{g \in \partial M_{2\lambda} h(z^{*})} \|g\| \leq 2\lambda L$ , by the fact that $M_{2\lambda}(h)$ is L-Lipschitz. Overall $r \leq \lambda \beta + 2\lambda L$ , and as we can assume without loss of generality that $\beta \leq L$ since otherwise the claim is trivially true (note that all points are L stationary), this completes the proof.

# C Concentration lemma for vectors with sub-Gaussian norm

Here we recall a standard concentration bound for vectors with sub-Gaussian norm, which notably applies in particular to bounded random vectors.

Definition C.1 (Norm-sub-Gaussian). We say a random vector $X \in \mathbb{R}^d$ is $\zeta$ -norm-sub-Gaussian for $\zeta > 0$ , if $\operatorname{Pr}[\|X - \mathbb{E}X\| \geq t] \leq 2e^{-t^2 / 2\zeta^2}$ for all $t \geq 0$ .

Theorem C.2 (Hoeffding-type inequality for norm-sub-Gaussian, Jin et al., 2019). Let $X_{1}, \cdots, X_{k} \in R^{d}$ be random vectors, and let $\mathcal{F}_{i} = \sigma(X_{1}, \cdots, X_{i})$ for $i \in [k]$ be the corresponding filtration. Suppose for each $i \in [k]$ , $X_{i} \mid F_{i-1}$ is zero-mean $\zeta_{i}$ -norm-sub-Gaussian. Then, there exists an absolute constant c > 0, such that for any $\gamma > 0$ :

$$
\operatorname * {P r} \left[ \left\| \sum_ {i \in [ k ]} X _ {i} \right\| \geq c \sqrt {\log (d / \gamma) \sum_ {i \in [ k ]} \zeta_ {i} ^ {2}} \right] \leq \gamma .
$$