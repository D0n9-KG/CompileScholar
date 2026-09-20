# Dueling Convex Optimization with General Preferences

Aadirupa Saha\*

Tomer Koren $^{†}$

Yishay Mansour $^{\dagger}$

# Abstract

We address the problem of convex optimization with dueling feedback, where the goal is to minimize a convex function given a weaker form of dueling feedback. Each query consists of two points and the dueling feedback returns a (noisy) single-bit binary comparison of the function values of the two queried points. The translation of the function values to the single comparison bit is through a transfer function. This problem has been addressed previously for some restricted classes of transfer functions, but here we consider a very general transfer function class which includes all functions that can be approximated by a finite polynomial with a minimal degree p. Our main contribution is an efficient algorithm with convergence rate of $\widetilde{O}(\epsilon^{-4p})$ for a smooth convex objective function, and an optimal rate of $\widetilde{O}(\epsilon^{-2p})$ when the objective is smooth and strongly convex.

# 1 Introduction

Convex optimization algorithms are fundamental across many fields, including machine learning. Most commonly, convex optimization is studied in a first-order gradient oracle model, where the optimization algorithm may query gradients of the objective function. A more limited model is that of zero-order oracle access, where the optimization algorithm may only query function values of the objective rather than gradients. Both of these models are extremely well-studied, and the optimal convergence rates in each of them are well known (see, e.g., Nesterov, 2003).

However, there are optimization scenarios where even zero-order access is unavailable or unreliable. Indeed, studies have shown that it is often easier, faster and involves lesser bias to collect feedback on a relative scale rather than asking for reward/loss feedback on an absolute scale. For example to understand the liking for a given pair of items, say (A,B), it is easier for the users to answer preference-based queries like: “Do you prefer item A over B?”, rather than their absolute counterparts: “How much do you score items A and B in a scale of $[0-10]$ ?”. Consequently, relative preference queries are extremely common in domains such as recommendation systems, online merchandises, search engine optimization, crowd-sourcing, drug testing, tournament ranking, social surveys, etc (Hajek et al., 2014; Khetan and Oh, 2016). This motivated the introduction of dueling bandits Yue and Joachims (2009) in the online learning setting.

Drawing motivation from the above, in this paper we study a challenging convex optimization model where the access to the objective function is through a noisy pairwise comparison oracle. Namely, given an underlying convex objective function $f: \mathcal{D} \mapsto \mathbb{R}$ ( $\mathcal{D} \subseteq \mathbb{R}^d$ being a convex decision space), at each step the optimization algorithm is allowed to query two points $\mathbf{w}, \mathbf{w}'$ in the feasible domain, upon which only a noisy 1-bit feedback $o_t \in \{\pm 1\}$ is revealed, whose expected value

indicates their relative function values. More specifically, the feedback signal $o_{t}$ is such that

$$
\mathbf {E} [ o _ {t} \mid \mathbf {w}, \mathbf {w} ^ {\prime} ] = \rho (f (\mathbf {w}) - f (\mathbf {w} ^ {\prime})),
$$

where $\rho: R \mapsto [-1,1]$ is a (possibly nonlinear) transfer function mapping difference in function values to a signed preference signal, and $\rho(f(\mathbf{w}) - f(\mathbf{w}'))$ is interpreted as the degree to which w should be preferred over $w'$ , or vice versa. Provided such access, our goal is to find a feasible point that approximately minimizes the objective f. Borrowing terminology from the literature on dueling bandits, we call our framework General Dueling Convex Optimization (G-DCO) for general transfer functions.

Noisy pairwise comparison access could potentially be significantly weaker than the already weak zero-order access. Indeed, a special case of this framework has been studied by Jamieson et al. (2012) who focused on polynomial transfer functions of the form $\rho(x)=c\mathrm{sign}(x)|x|^{p}$ and gave tight upper and lower bounds in the pairwise comparison model for strongly convex and smooth objectives. Their results indicate that as p grows larger, the best achievable convergence rate degrades quickly, and already when p>1 this rate becomes strictly inferior to that of zero-order optimization. Much more recently, Saha et al. (2021b) considered a similar pairwise comparison model with a different type of a transfer function, namely the sign function $\rho(x)=\mathrm{sign}(x)$ (p=0), and established fast convergence rates for this case exclusively.

Both of these works point us at a some fundamental questions: Can we design algorithms for dueling convex optimization that are able to leverage more general transfer functions? Can we converge to a minimizer even when the transfer is unknown to the algorithm? And what properties of the transfer function dictate the achievable optimization rates? In this paper, we make progress towards answering these questions.

# 1.1 Our contributions

We make the following main contributions:

(i) We formalize a generalized dueling convex optimization setting for convex optimization with pairwise-preference feedback given by a general transfer function $\rho$ , which is only assumed to be well-behaved around the origin (see Section 2.1 for a precise definition of the query model and optimization objective). Our framework generalizes and significantly extends two existing settings of optimization with comparison feedback Jamieson et al. (2012); Saha et al. (2021b) (Section 2).   
(ii) We give a novel algorithm for dueling convex optimization with a general transfer function $\rho$ , called Relative-Gradient-Descent (Algorithm 1), which relies on performing a 'generalized gradient descent' like update on the $p^{th}$ -degree-scaled Gradient of the objective function $f$ (see Definition 2, Lemma 11). Remark 1 explains how $p^{th}$ -degree-scaled Gradient is a generalization of gradient estimate and smoothly interpolates between different types of descent directions. We prove that when the optimization objective function is smooth, our algorithm needs an order of $O(\epsilon^{-4p})$ queries to the pairwise-preference oracle for finding an $\epsilon$ -optimal point (see Theorem 3). Here, $p$ is the minimal non-zero degree in a series expansion of $\rho(x)$ around zero (Section 3).   
(iii) We further show that our algorithm can achieve faster convergence rates when the function is additionally also strongly convex (Algorithm 2, Theorem 6). Concretely, we show that in this case only $O(\epsilon^{-2p})$ pairwise queries are sufficient for $\epsilon$ -convergence. The latter rate is shown to be tight as it matches existing lower bounds (for certain transfer functions) for strongly convex optimization with comparison feedback due to Jamieson et al. (2015) (Section 4).

Our algorithmic results complement those of Saha et al. (2021b), who only considered the sign transfer function. Compared to the results of Jamieson et al. (2012), we are able to handle both the convex and strongly convex cases (while they only deal with the strongly convex case), and we only require the transfer to be well-behaved around the origin (while they rely on its global structure $^{1}$ ). Thus, we are able to encompass a much wider variety of transfer functions whose local behavior around zero is approximated by a polynomial—this includes virtually all functions that admit a series expansion around the origin.

# 1.2 Related work

Dueling Bandits. Due to the widespread applicability and ease of data collection with relative feedback, learning from preferences has gained much popularity in the machine learning community and widely studied as the problem of Dueling-Bandits over last decade (Ailon et al., 2014; Yue et al., 2012; Zoghi et al., 2014a,b, 2015; Saha et al., 2021a; Gajane et al., 2015; Bengs et al., 2021), which is an online learning framework that generalizes the standard multi-armed bandit (MAB) (Auer et al., 2002) setting for identifying a set of ‘good’ arms from a fixed decision-space (set of items) by querying preference feedback of actively chosen item-pairs.

Limitations of Existing Dueling Bandit techniques. Although the relative feedback variants of stochastic MAB problem have been widely studied in the literature, the majority of the existing techniques are restricted to finite decision spaces and stochastic setting which primarily rely on estimating the entries of the underlying preference-matrix. These settings, though important as basic steps, are mostly impractical for all real world scenarios which often involves large (or potentially infinite) decision spaces, where lies one of the primary motivation of this work. On the other hand, from an optimization point of view, our work is a key step towards analyzing the fundamental performance limits of function minimization using the weaker form of 0/1 bit relative preferences. The few existing attempts along this line is discussed in Related Works.

Dueling Bandits in continuous spaces. Surprisingly, following the same spirit of extending standard multi-armed bandits (MAB) to continuous decision spaces (as in linear or GP-bandits), there has not been much work on the continuous extension of the Dueling Bandit problem for large (and structured) decision spaces. The works in Sui et al. (2017); González et al. (2017) did attempt a similar objective, however, without any satisfactory theoretical performance guarantees. In another recent work, Brost et al. (2016) address the problem of regret minimization in continuous Dueling Bandits, however without any finite time regret guarantee of their proposed algorithms. Recently, Oh and Iyengar (2019); Saha (2021) consider the problem of regret minimization from k-subsetwise preference feedback (k = 2 boils down to the dueling setup) on structured decision spaces, although their underlying utility function is assumed to be only linear, unlike any general convex function considered in our work; moreover, their preference model is restricted only to the class of Multinomial Logit (MNL) based random utility model, unlike the general link function based preference feedback class that we considered. Dudík et al. (2015); Saha and Krishnamurthy (2021) represents another line of dueling bandit work, which incorporates context specific dueling preference model. Specifically, their algorithms are designed to compete against an abstract policy set of context to action mappings w.r.t. ‘minimax-regret’. Their algorithms are also designed to handle potentially large decision spaces, although, the regret objectives are focused to identifying

the von-Neumann distribution of the underlying preference models, which is very different from the function minimization with dueling feedback point of view that we considered.

Optimization for dueling feedback. Along the line of optimization for dueling feedback, Yue and Joachims (2009) is the first to address the regret minimization problem for fixed functions f (arm rewards) with preference feedback, although their techniques are majorly restricted to the class of smooth and differentiable preference functions that allows gradient estimation. This is the main reason they could directly apply the classical one-point gradient estimation based Bandit Gradient Descent (BGD) algorithm of Flaxman et al. (2005) for the setting, unlike us. Moreover, another limitation of their framework is their optimization objective is defined in terms of the ‘preferences’ which are directly observable and hence easier to optimize, as opposed to defining it w.r.t. f as considered in this work. Following up Yue and Joachims (2009), Kumagai (2017) considers the similar problem of dueling bandits on continuous arm set but under rather restrictive sets of assumptions: Twice continuously differentiable, Lipschitz, strongly convex and smooth score/reward function, which are often impractical for modeling any real-world preference feedback.

Closest to our work in spirit are Jamieson et al. (2012) and Saha et al. (2021b), both of which precisely focus on function optimization with relative pairwise preference feedback. The latter however is designed to work only under sign based relative feedback which reveals the exact information of which of the two queried points have smaller function value. We instead consider a very general class of polynomial based preference functions (see Section 2) which generalizes the sign-feedback model of Saha et al. (2021b) as a special case. While the first, although gives provably optimal convergence rates, their guarantees are restricted to the 'well behaved' class of strongly-convex and smooth functions (with bounded Lipschitz gradient). The assumptions and consequently their techniques are hence quite restrictive: A major hindrance towards generalizing their algorithmic ideas to a general function class is owning to their line-search based coordinate descent algorithm which is known to fail without strong-convexity. On the other hand, our algorithm is shown to yield optimal convergence guarantees for more general class of smooth-convex functions. Additionally we match the convergence rate of Jamieson et al. (2012) with the additional strong convexity assumption which shows the generality of our analysis for a large class of dueling feedback based optimization (G-DCO) problems. As motivated in our list of contributions, the novelty lies of our analysis lies in the Relative-Gradient-Descent based optimization approach, which smoothly interpolates between different complexity classes of different Dueling Convex Optimization problems based on the degree of the underlying polynomial link function $p$ (see Remark 1). Besides our method is arguably simpler both in terms of implementation and analysis.

# 2 Preliminaries and Problem Setup

Notation. Let $[n] = \{1, 2, \ldots, n\}$ , for any $n \in N$ . Given a set S, for any two items $x, y \in S$ , we denote by $x \succ y$ the event i is preferred over j. For any r > 0, let $\mathcal{B}_{d}(r)$ and $\mathcal{S}_{d}(r)$ denote the ball and the surface of the sphere of radius r in d dimensions respectively. $I_{d}$ denotes the $d \times d$ identity matrix. For any vector $x \in R^{d}$ , $\|x\|_{2}$ denotes the $\ell_{2}$ norm of vector x.

# 2.1 Problem setup

We consider the problem of minimizing a convex and $\beta$ -smooth function $f : D \mapsto R$ defined on a bounded convex domain $D \subseteq R^{d}$ of Euclidean diameter D. We denote by $\mathbf{w}^{*} \in \arg\min_{\mathbf{w} \in \mathcal{D}} f(\mathbf{w})$ a point where f is minimized over D.

Query model: Our access to the objective f is through a noisy comparison oracle that upon a pair of inputs $(\mathbf{w},\mathbf{w}^{\prime})\in\mathcal{D}^{2}$ emits a random binary response $o\in\{\pm1\}$ such that $\mathbf{E}[o\mid\mathbf{w},\mathbf{w}^{\prime}]=\rho(f(\mathbf{w})-f(\mathbf{w}^{\prime}))$ , where $\rho:\mathbb{R}\to[-1,1]$ is a fixed transfer function mapping difference in function values to (signed) preferences, unknown to the algorithm. For example, given $\rho$ the query model could output a random variable o such that $o\sim\mathrm{Ber}^{\pm}\big(\rho(f(\mathbf{w})-f(\mathbf{w}^{\prime}))\big)$ where $Ber^{\pm}$ denotes a signed version of the Bernoulli distribution (such that for a random variable $X\sim\mathrm{Ber}^{\pm}(p)$ , we have $\Pr(X=+1)=1-\Pr(X=-1)=\frac{p+1}{2}$ ).

Transfer function: We will assume throughout that the transfer function $\rho : R \mapsto [-1, 1]$ is fixed and unknown to the algorithm. We make the following assumptions on $\rho$ :

Assumption 1. (i). $\rho$ is differentiable and anti-symmetric (namely, $\rho(-x) = -\rho(x)$ for all $x$ ) and satisfies $\rho(0) = 0$ and $\mathrm{sign}(\rho(x)) = \mathrm{sign}(x)$ for $x \neq 0$ ; (ii) there are constants $p \geq 1$ and $r, c_{\rho} > 0$ such that for all $x \in (-r, r)$ it holds that $\rho'(x) \geq c_{\rho} p | x|^{p-1}$ .

Following gives the intuition behind the practicability of the above set of assumptions: Let us define the function $\tilde{\rho}_{p}: R \mapsto [-1,1]$ such that $\tilde{\rho}_{p}(x) = c_{\rho} \text{sign}(x)|x|^{p}$ . Then note $\rho$ satisfies $\rho'(x) \geq \tilde{\rho}_{p}'(x) = c_{\rho} p|x|^{p-1}$ for all $x \in (0, r]$ . This essentially implies that our class of admissible transfer functions ( $\rho$ ) admit a series expansion of the form $\rho(x) = \sum_{n=p}^{\infty} a_{n} x^{n}$ around close neighborhood of x = 0 with minimal degree $p \geq 1$ (see Lemma 1 for a formal justification). We will henceforth refer $\tilde{\rho}_{p}$ as the ‘p-th order proxy’ of $\rho$ . Note that, one can recover the ‘sign feedback’ of Saha et al. (2021b) for $\rho = \tilde{\rho}_{p}$ with p = 0, $c_{\rho} = 1$ .

It is also important to note that our assumptions imply that $\rho$ is monotonically increasing in a small neighborhood of the origin. While we assume that $\rho$ is unknown to the algorithm, we will implicitly assume that the parameters $p, r, c_{\rho}$ above are known. (This knowledge will be used only for optimally tuning the hyper-parameters of our algorithms.)

Optimization goal: The goal of the optimization process is then, given $\epsilon > 0$ , to find a point w such that $f(\mathbf{w}) - f(\mathbf{w}^{*}) \leq \epsilon$ while minimizing the number of queries to to the comparison oracle.

# 2.2 Admissible Transfer Functions

Our latter assumption on the transfer function $\rho$ is perhaps the most stringent one; however, it is satisfied by a wide variety of natural transfer functions: those that admit a series expansion about the origin with uniformly bounded coefficients.

Lemma 1. Let $\rho$ admit a series expansion $\rho(x) = \sum_{n=p}^{\infty} a_n x^n$ about $x = 0$ with minimal degree $p \geq 1$ and radius of convergence $\delta > 0$ . Then, if $a_p > 0$ and $|na_n| \leq M$ for all $n > p$ , we have that

$$
| \rho^ {\prime} (x) | \geq \frac {1}{2} p a _ {p} | x | ^ {p - 1} \quad f o r \quad | x | <   \min \Bigl \{\delta , \frac {p a _ {p}}{4 M} \Bigr \}.
$$

Note that since we require $\rho(0)=0$ , it must be that $a_{0}=0$ and the assumption $p\geq1$ holds naturally. Further, since we would like $\rho(x)>0$ to hold for x>0, the first nonzero coefficient must be positive, namely $a_{p}>0$ . Thus, the only non-trivial assumption is that the series coefficients are uniformly bounded; however, this condition holds for many natural transfer functions: e.g., for the sigmoidal $\arctan(x)$ , hyperbolic tangent $\tanh(x)$ and for the error function $\operatorname{erf}(x)$ , it holds simply with M=1.

Proof of Lemma 1. On the interval of convergence $(- \delta, \delta)$ we have $\rho'(x) = \sum_{n=p}^{\infty} na_n x^{n-1}$ as one can exchange the order of summation and differentiation. Let us write $\rho'(x) = pa_p x^{p-1} + R(x)$ , where $R(x) = \sum_{n>p} na_n x^{n-1}$ . Then, for $|x| < \delta \leq \frac{1}{2}$ ,

$$
| R (x) | \leq \sum_ {n > p} | n a _ {n} | | x | ^ {n - 1} \leq M | x | ^ {p} \sum_ {n = 0} ^ {\infty} | x | ^ {n} = M | x | ^ {p} \frac {1}{1 - | x |} \leq 2 M | x | ^ {p}.
$$

Thus, when $|x| \leq pa_p / 4M$ we have $|R(x)| \leq \frac{1}{2} pa_p |x|^{p-1}$ . It follows that $|\rho'(x)| \geq pa_p |x|^{p-1} - |R(x)| \geq \frac{1}{2} pa_p |x|^{p-1}$ as claimed.

# 3 Dueling Convex Optimization with General Transfer Functions

In this section we propose an optimization algorithm for our problem (see Objective in Section 2) for any convex and $\beta$ -smooth $f$ ( $\beta > 0$ ). Note the primary difficulty towards designing an efficient algorithm for the purpose lies in the fact that we can not hope to estimate the gradient of $f$ for any general dueling/pairwise preference model (i.e. any general $\rho$ ). Thus we can not apply the standard gradient descent based techniques to address this problem (Boyd et al., 2004; Bubeck, 2014; Hazan, 2019).

We get around with the difficulty by noting that, though one may not be able to estimate the exact gradient of $f$ , $\nabla f(\mathbf{w})$ , at a given point of interest $\mathbf{w} \in \mathcal{D}$ , we can hope to estimate a 'p-th order proxy of $\nabla f(\mathbf{w})$ ', called $p^{th}$ -degree-scaled Gradient of $f$ at $\mathbf{w}$ , from the 1-bit preference feedback generated according to the transfer function (or pairwise preference model) $\rho$ . The following definition and the lemma describes a more formal argument on this.

Definition 2 ( $p^{th}$ -degree-scaled Gradient). Given any function $f: \mathbb{R}^d \mapsto \mathbb{R}$ , we define the $p^{th}$ -degree-scaled Gradient of $f$ at any point $\mathbf{w} \in \mathbb{R}^d$ to be $\nabla f(\mathbf{w}) \| \nabla f(\mathbf{w}) \|^{p-1}$ for any $p \geq 1$ .

Lemma 11, in Appendix B, gives a formal justification of the key characteristics of $p^{th}$ -degree-scaled Gradient estimate. Remark 1 gives a more intuitive explanation of the same and how we exploited it in our optimization algorithm (Algorithm 1).

Remark 1 (Key idea behind Algorithm 1: How it estimates a descent direction in terms of $p^{th}$ -degree-scaled Gradient?). As shown in Lemma 11 (Appendix B), the expected value of our $\mathbf{g}_t$ estimate in Algorithm 1 ( $\mathbf{E}_{\mathbf{u}_t, o_t}[\mathbf{g}_t]$ ), captures the estimated $p^{th}$ -degree-scaled Gradient (up to constant factors): It reflects the direction of the gradient $\nabla f(\mathbf{w})$ (in expectation) but magnitudewise represents the $p$ -order magnitude of that of the true gradient $\| \nabla f(\mathbf{w}) \|$ . Thus $-\mathbf{g}_t$ represents a valid descent direction in expectation, since it points to the negative direction of the gradient (modulo its magnitude is now skewed by the degree $p$ ).

It is important to note that $p^{th}$ -degree-scaled Gradient at any point w is a power generalization of ‘gradient feedback’ at w, $\nabla f(\mathbf{w})$ , which can automatically smoothly interpolate between different scaling orders of descent directions depending on the ‘expressiveness’ of the transfer function $\rho$ (captured through p). Clearly, the best case is attained for p = 1, when our feedback model is equivalent to the zeroth-order or bandit convex optimization feedback model (Flaxman et al., 2005), when $p^{th}$ -degree-scaled Gradient exactly boils down to the gradient estimate $\nabla f(\mathbf{w})$ . Moreover, note if p = 0, our feedback model recovers the sign-feedback model of Saha et al. (2021b) and in this case our gradient estimate also $E[g_{t}]$ roughly captures the normalized gradient $\frac{\nabla f(\mathbf{w})}{\|\nabla f(\mathbf{w})\|}$ (direction of the gradient at w) on expectation, as used in Saha et al. (2021b) as well.

# 3.1 Algorithm Design: Relative-Gradient-Descent

The crux of the idea lies in designing $p^{th}$ -degree-scaled Gradient based algorithm (Algorithm 1), which is a generalized notion of gradient descent based optimization technique: The algorithm proceeds sequentially, where at each step t, it maintains a current point of interest $w_{t} \in D$ , estimate the $p^{th}$ -degree-scaled Gradient of f at point $w_{t}$ using dueling feedback (as indicated in Lemma 11), and take a ‘carefully chosen small’ step in the negative direction of the estimated $p^{th}$ -degree-scaled Gradient to reach the updated point of interest $w_{t+1}$ .

More formally, the algorithm starts from an initial point $w_{1} \in D$ . Now at any round $t = 1, 2, \ldots$ , the algorithm queries the dueling feedback on a pair of points $(\mathbf{w}_{t} + \gamma\mathbf{u}_{t}, \mathbf{w}_{t} - \gamma\mathbf{u}_{t})$ , such that $u_{t} \sim \text{Unif}(\mathcal{S}_{d}(1))$ is any random unit norm d-dimensional vector, $\gamma$ being a carefully tuned perturbation parameter. Upon receiving the 1-bit preference feedback $o_{t} \in \{\pm 1\}$ , it finds a $p^{th}$ -degree-scaled Gradient estimate of f at $w_{t}$ as $g_{t} := o_{t}u_{t}$ which gives a valid descent direction on expectation as shown in Lemma 11 (see Remark 1 for more insights). It then takes an $\eta$ -sized step along the negative direction of $g_{t}$ to obtain the next iterate $w_{t+1} := w_{t} - \eta g_{t}$ (with suitable projection if necessary). The details of the algorithm is presented in Algorithm 1.

Theorem 3 analyses its convergence guarantees which shows that upon iterating through the above steps for at most $O(\epsilon^{-4p})$ rounds, the algorithm should be able to find a desired $\epsilon$ -optimal point.

Algorithm 1 Relative-Gradient-Descent   
1: Input: Initial point: $w_{1} \in D$ , Learning rate $\eta$ , Perturbation parameter $\gamma$ , Query budget T
2: for $t = 1, 2, 3, \ldots, T$ do
3: Sample $u_{t} \sim \text{Unif}(S_{d}(1))$ 4: Set $x_{t}^{\prime} := w_{t} + \gamma u_{t}$ , $y_{t}^{\prime} := w_{t} - \gamma u_{t}$ 5: Play the duel $(\mathbf{x}_{t}^{\prime}, \mathbf{y}_{t}^{\prime})$ , and observe $o_{t} \in \pm 1$ such that $o_{t} \sim \text{Ber}^{\pm}\big(\rho\big(f(\mathbf{x}_{t}^{\prime}) - f(\mathbf{y}_{t}^{\prime})\big)\big)$ .
6: Update $\tilde{w}_{t+1} \leftarrow w_t - \eta g_t$ , where $g_t = o_t u_t$ 7: Project $w_{t+1} = \arg \min_{w \in D} \|w - \tilde{w}_{t+1}\|$ 8: end for

Since our proposed Algorithm 1 is based on an iterative ‘ $p^{th}$ -degree-scaled Gradient-descent’ based approach (Remark 1), the interesting part in it’s convergence analysis was indeed to understand how this can be exploited to gradually descent towards the true minimizer $\mathbf{w}^*$ and reach an $\epsilon$ -optimal point with small enough query complexity. The details are explained more mathematically in the proof of Theorem 3.

# 3.2 Convergence Analysis for Smoothly Convex Functions

Theorem 3. Consider a dueling feedback optimization problem parameterized by any general admissible transfer function $\rho$ with $p$ -th order proxy $\tilde{\rho}_p$ and a $\beta$ smooth convex function $f: \mathcal{D} \mapsto \mathbb{R}$ . Then given any $\epsilon > 0$ , for the choice of $\gamma = \frac{\tilde{c}\epsilon}{\beta\sqrt{d}D}$ and $\eta = \frac{pc_{\rho}\tilde{c}^{2p-1}\epsilon^{2p}}{d^{(2p+1)/2}\beta^{p}D^{2p-1}}$ , there exists at least one $t$ such that $\mathbf{E}[f(\mathbf{w}_t)] - f(\mathbf{w}^*) \leq \epsilon$ , after at most $T = \frac{d^{2p+1}\beta^{2p}D^{4p}}{p^2(\tilde{c}^{2p-1}c_\rho)^2\epsilon^{4p}} + 1$ iterations; i.e. $\min_{t \in [T]} \mathbf{E}[f(\mathbf{w}_t)] - f(\mathbf{w}^*) \leq \epsilon$ , where $\tilde{c} = \frac{1}{20}$ is a universal constant.

Theorem 3 shows that for any general transfer function $\rho$ with a $p$ -th degree $p$ -th order proxy $\tilde{\rho}_p$ , Algorithm 1 gives a convergence rate of $O(\epsilon^{-4p})$ to find an $\epsilon$ -optimal point. However, Theorem 5 shows a improved convergence rate of $O(\epsilon^{-3})$ for linear ( $p = 1$ ) transfer functions which recovers the convergence rate obtained in Saha and Tewari (2011) for smooth convex functions in the Bandit Convex Optimization (1 point feedback setting). Moreover, Theorem 3 also shows how Algorithm 1

can yield a faster convergence rate of $O(\epsilon^{-1})$ for sign transfer functions (p = 0) which matches the convergence rate obtained in Saha et al. (2021b) — in fact, not just the final convergence rate, our algorithm (Relative-Gradient-Descent, Algorithm 1) generalizes the $\beta$ -NGD algorithm of Saha et al. (2021b) since our descent direction estimate ( $g_{t}$ ), exactly behaves like the normalized gradient (gradient direction) at point $w_{t}$ which was the crux of their optimization analysis. Please see the proof of Theorem 5 for more details.

Proof of Theorem 3 (sketch). The complete details of the proofs can be found in Appendix C. We denote by $\mathcal{H}_t$ the history $\{\mathbf{w}_{\tau},\mathbf{u}_{\tau},o_{\tau}\}_{\tau = 1}^{t - 1}\cup \mathbf{w}_t$ till time $t$ . We start by noting that by definition:

$$
\mathbf {E} _ {o _ {t}} [ \mathbf {g} _ {t} | \mathcal {H} _ {t}, \mathbf {u} _ {t} ] = \rho (f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t})) \mathbf {u} _ {t}
$$

Base Case: Let us start with the assumption that $f(\mathbf{w}_{1}) - f(\mathbf{w}^{*}) > \epsilon$ (as otherwise we already have $\min_{t \in [T]} \mathbf{E}[f(\mathbf{w}_{t})] - f(\mathbf{w}^{*}) \leq \epsilon$ and there is nothing to prove).

We proceed with the proof inductively, i.e. given $H_{t}$ and assuming (conditioning on) $f(\mathbf{w}_{t}) - f(\mathbf{w}^{*}) > \epsilon$ , we can show that $w_{t+1}$ always come closer to the minimum $w^{*}$ on expectation in terms of the $\ell_{2}$ -norm. More formally, given $H_{t}$ and assuming $f(\mathbf{w}_{t}) - f(\mathbf{w}^{*}) > \epsilon$ we will show: $E_{t}[\|\mathbf{w}_{t+1} - \mathbf{w}^{*}\|^{2}] \leq \|\mathbf{w}_{t} - \mathbf{w}^{*}\|^{2}$ , where $E_{t}[\cdot] := E_{o_{t}, u_{t}}[\cdot \mid H_{t}]$ denote the expectation with respect to $u_{t}, o_{t}$ given $H_{t}$ . The precise statement can be summarized in the following lemma:

Lemma 4 (Roundwise Progress of Relative-Gradient-Descent). Consider the problem setup of Theorem 3 and also the choice of $\eta$ , $\gamma$ . Then at any time t, during the run of Relative-Gradient-Descent (Algorithm 1), given $H_{t}$ , if $f(\mathbf{w}_{t}) - f(\mathbf{w}^{*}) > \epsilon$ , we can show that:

$$
\mathbf {E} _ {t} [ \| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - \frac {p ^ {2} (\tilde {c} ^ {2 p - 1} c _ {\rho}) ^ {2} \epsilon^ {4 p}}{d ^ {2 p + 1} \beta^ {2 p} D ^ {4 p - 2}} (1)
$$

Proof. We first note that by our update rule,

$$
\begin{array}{l} \mathbf {E} _ {t} [ \| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \mathbf {E} _ {t} [ \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} ] - 2 \eta \mathbf {E} _ {t} [ [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] ] + \eta^ {2} \\ = \left\| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \right\| ^ {2} - 2 \eta \mathbf {E} _ {t} \left[ \left[ \mathbf {g} _ {t} \cdot \left(\mathbf {w} _ {t} - \mathbf {w} ^ {*}\right) \right] \right] + \eta^ {2}. \tag {2} \\ \end{array}
$$

On the other hand, since both $f$ and $\rho$ is convex (by assumption), using Lemma 9 we get:

$$
\begin{array}{l} \mathbf {E} _ {t} [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] = \mathbf {E} _ {\mathbf {u} _ {t}} \left[ \rho (f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t})) \mathbf {u} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) \mid \mathcal {H} _ {t} \right] \\ = \mathbf {E} _ {\mathbf {u} _ {t}} \left[ \rho (f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t})) \cdot \mathbf {u} _ {t} \mid \mathcal {H} _ {t} \right] \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) \\ = \frac {\gamma}{d} \mathbf {E _ {u}} _ {t} \left[ \rho^ {\prime} \big (f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) \big) \big (\nabla f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) + \nabla f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) \big) \mid \mathcal {H} _ {t} \right] \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) \\ = \frac {\gamma}{d} \mathbf {E} _ {\mathbf {u} t} \left[ \rho^ {\prime} \left(| f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) |\right) \left(\nabla f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) + \nabla f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t})\right) | \mathcal {H} _ {t} \right] \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}), \tag {3} \\ \end{array}
$$

where the last equality follows since $\rho(-x) = -\rho(x)$ (see Assumption 1-i). Now, using convexity of f and $\beta$ -smoothness, we can further show that:

$$
\mathbf {E} _ {t} [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] \geq \frac {2 \gamma}{d} \mathbf {E} _ {\mathbf {u} _ {t}} \left[ \rho^ {\prime} \big (| f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | \big) \big (f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*}) - \beta \gamma^ {2} \big) \right], \tag {4}
$$

and also $\mathbf{E_{u_t}}[|f(\mathbf{w}_t + \gamma \mathbf{u}_t) - f(\mathbf{w}_t - \gamma \mathbf{u}_t)|]\geq \mathbf{E_{u_t}}[|2\gamma \mathbf{u}_t\cdot \nabla f(\mathbf{w}_t)|] - \beta \gamma^2 = 2\frac{\tilde{c}\gamma\|\nabla f(\mathbf{w}_t)\|}{\sqrt{d}} -\beta \gamma^2$

where the last equality is due to Lemma 8. Additionally, since by assumption $f(\mathbf{w}_{t}) - f(\mathbf{w}^{*}) > \epsilon$ , i.e. the suboptimality gap to be at least $\epsilon$ , by Lemma 10 we can further derive a lower bound:

$$
\mathbf {E _ {u}} _ {t} [ | f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | ] \geq \frac {2 \tilde {c} \gamma \epsilon}{\sqrt {d} \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \|} - \beta \gamma^ {2} = \frac {2 \tilde {c} \gamma \epsilon}{\sqrt {d} D} - \beta \gamma^ {2}.
$$

Now for the choice of $\gamma = \frac{\tilde{c}\epsilon}{\beta\sqrt{d}D}$ since the lower bound in right hand side is always positive, by monotonicity of $\tilde{\rho}_p'$ in the positive orthant we get:

$$
\begin{array}{l} \rho \left(\mathbf {E} _ {\mathbf {u} _ {t}} \left[ \left| f \left(\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}\right) - f \left(\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}\right) \right| \right]\right) \geq \tilde {\rho} _ {p} ^ {\prime} \left(\mathbf {E} _ {\mathbf {u} _ {t}} \left[ \left| f \left(\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}\right) - f \left(\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}\right) \right| \right]\right) \\ \geq \tilde {\rho} _ {p} ^ {\prime} \left(\frac {\tilde {c} ^ {2} \epsilon^ {2}}{\beta d D ^ {2}}\right) = c _ {\rho} p \left(\frac {\tilde {c} ^ {2} \epsilon^ {2}}{\beta d D ^ {2}}\right) ^ {p - 1}, \tag {5} \\ \end{array}
$$

where the first inequality follows by the definition of $\tilde{\rho}_{p}$ which is p-th order proxy of $\rho$ (see Assumption 1), and the last equality follows since by definition $\rho'(x) = c_{\rho}px^{p-1}$ for any $x \in R_{+}$ . Finally combining Equations (4) and (5), and the choice of $\gamma$ , we can finally derive the lower bound:

$$
\mathbf {E} _ {t} [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] \geq \frac {p c _ {\rho} \tilde {c} ^ {2 p - 1} \epsilon^ {2 p}}{d ^ {(2 p + 1) / 2} \beta^ {p} D ^ {2 p - 1}}, \tag {6}
$$

Combining Equation (2) with Equation (6):

$$
\begin{array}{l} \mathbf {E} _ {t} [ \| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - 2 \eta \mathbf {E} _ {t} [ [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] ] + \eta^ {2} \\ \leq \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - 2 \eta \left(\frac {p c _ {\rho} \tilde {c} ^ {2 p - 1} \epsilon^ {2 p}}{d ^ {(2 p + 1) / 2} \beta^ {p} D ^ {2 p - 1}}\right) + \eta^ {2} \\ \leq \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - \left(\frac {p c _ {\rho} \tilde {c} ^ {2 p - 1} \epsilon^ {2 p}}{d ^ {(2 p + 1) / 2} \beta^ {p} D ^ {2 p - 1}}\right) ^ {2}, \quad \mathrm{setting} \eta = \left(\frac {p c _ {\rho} \tilde {c} ^ {2 p - 1} \epsilon^ {2 p}}{d ^ {(2 p + 1) / 2} \beta^ {p} D ^ {2 p - 1}}\right), \\ \end{array}
$$

which concludes the claim of Lemma 4.

![](images/c6a5faf5faa1820ac52afa3d3a7cec285c3b9642b8ca3c087b31b3352b57e69d.jpg)

Now coming back to the main proof of Theorem 3, note by iteratively taking expectation over $H_{T}$ on both sides of Equation (1) and summing over $t = 1, \ldots, T$ , we get,

$$
\mathbf {E} _ {\mathcal {H} _ {T}} [ \| \mathbf {w} _ {T + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {w} _ {1} - \mathbf {w} ^ {*} \| ^ {2} - \frac {p ^ {2} (\tilde {c} ^ {2 p - 1} c _ {\rho}) ^ {2} \epsilon^ {4 p}}{d ^ {2 p + 1} \beta^ {2 p} D ^ {4 p - 2}} T.
$$

However, note if we set $T = \frac{d^{2p + 1}\beta^{2p}D^{4p}}{p^2(\tilde{c}^{2p - 1}c_\rho)^2\epsilon^{4p}}$ , this implies $\mathbf{E}_{\mathcal{H}_T}[\| \mathbf{w}_{T + 1} - \mathbf{w}^*\|^2 ]\leq 0$ , or equivalently $\mathbf{w}_{T + 1} = \mathbf{w}^*$ , which concludes the claim.

To clarify further, note we show that for any run of Alg. 1 if indeed $f(\mathbf{w}_t) - f(\mathbf{w}^*) > \epsilon$ continues to hold for all $t = 1,2,\ldots T$ , then $\mathbf{w}_{T + 1} = \mathbf{w}^*$ at $T = \frac{d^{2p + 1}\beta^{2p}D^{4p}}{p^2(\tilde{c}^{2p - 1}c_\rho)^2\epsilon^{4p}}$ . If not, there must have been a time $t\in [T]$ such that $f(\mathbf{w}_t) - f(\mathbf{w}^*) < \epsilon$ .

While Theorem 3 gives the convergence rate of Algorithm 1 for any general ‘admissible transfer function’ $\rho$ (see Section 2 for the setup), the following theorem shows that Algorithm 1 can achieve improved convergence rates for certain class of special transfer functions, as remarked in Theorem 5. The proof is given in Appendix C.3.

Theorem 5 (Improved Convergence Rate for Special Transfer Functions.). Algorithm 1 yields improved $\epsilon$ -convergence rate $(T_{\epsilon})$ for some special class of well-defined transfer functions, e.g.:

1. Linear transfer functions $\rho(x) = c_{\rho}x$ , $\forall x \in \mathbb{R}_{+}$ , then we have $T = \frac{2d^2\beta D^2}{c_\rho^2\epsilon^3}$ ;   
2. Sigmoid transfer functions $\rho(x)=\frac{1-e^{-\omega x}}{1+e^{-\omega x}},\;\forall x\in\mathbb{R}_{+},\;\omega>0,$ then we have $T=O\left(\frac{d^{2}\beta D^{2}}{c_{\rho}^{2}\epsilon^{3}}\right)$ .

It is worth noting that, for Linear transfer functions, i.e. when $\rho(x)=c_{\rho}x\ (p=1)$ , our setting is equivalent to the bandit feedback (or zeroth-order) optimization setting (Flaxman et al., 2005) and our proposed algorithm obtains the same $O(d^{2/3}T^{-1/3})$ convergence rate of Saha and Tewari (2011) which is the best known rate till date for zeroth-order smooth convex optimization with gradient descent based algorithms. Moreover, for Sign transfer functions, i.e. for $\rho(x)=\mathrm{sign}(x)$ (p=0), our algorithm can essentially recovers the $\beta$ -NGD algorithm (Algorithm 1) of Saha et al. (2021b) and hence we can obtain the optimal convergence guarantee of $T=O\left(\frac{dD\beta}{\epsilon}\right)$ (see analysis of Case-3 in Appendix C.3 for details). These results thus show the generalizability of our problem framework as well as our algorithmic approach (Algorithm 1).

# 4 Strongly Convex Dueling Optimization

In this section, we analyze an epoch-wise version of Relative-Gradient-Descent (Algorithm 1) which is shown to yield better convergence guarantees for $\alpha$ -strongly convex and $\beta$ -smooth functions. The key idea lies in noting that in order to design an optimal algorithm for $\alpha$ -strongly convex $\beta$ -smooth functions, one can simply iteratively reuse any $\beta$ -smoothly convex optimization routine (e.g. we can use our Alg. 1) by running it as a black-box over a successive number of epoch-wise warm-starts. Our resulting convergence analysis (Theorem 6) shows that, in this case the algorithm can find an $\epsilon$ -optimal point upon querying just $O(\epsilon^{-2p})$ pairwise comparisons (as opposed to the $O(\epsilon^{-4p})$ sample complexity rate for the $\beta$ -smooth case, see Theorem 3). This is possible due to the nice properties of strong convexity, where nearness in the suboptimality gap in terms of the function values, $f(\mathbf{w}) - f(\mathbf{w}^*)$ implies nearness in terms of the $\ell_2$ -distance from $\| \mathbf{w} - \mathbf{w}^* \|$ (see the third property in Lem. 18). In fact the $O(\epsilon^{-2p})$ convergence rate can shown to be information theoretically optimal (see Remark 2).

# 4.1 Algorithm Design: Epoch-RGD

As motivated above, our proposed method $Epoch-RGD$ (Algorithm 2) uses an 'epoch-wise black-boxing of a smooth-convex optimization routine' with 'warm-starting' approach. For our purpose, we use the earlier proposed $Relative-Gradient-Descent$ (Algorithm 1) as the black-box. More formally, the algorithm, starts with some initial point $\mathbf{w}_1$ and runs over a sequence of $k_\epsilon = O\left(\log \frac{\beta D^2}{\epsilon}\right)$ epochs: Inside each epoch $k \in [k_\epsilon]$ , we call the $Relative-Gradient-Descent(\mathbf{w}_k, \eta_k, \gamma_k, t_k)$ subroutine with the initial (warm-start) iterate $\mathbf{w}_k$ , suitably tuned parameters $\eta_k, \gamma_k$ and a query budget of $t_k$ . The decision point returned by $Relative-Gradient-Descent$ after $t_k$ steps is considered to the next iterate, setting $\mathbf{w}_{k+1} \leftarrow Relative-Gradient-Descent(\mathbf{w}_k, \eta_k, \gamma_k, t_k)$ and we proceed to the $(k+1)$ -th epoch, warm-starting it with $\mathbf{w}_{k+1}$ .

The key idea behind the epoch-wise warm-start approach exploits the fact that between any two consecutive epochs, say k and $k+1$ , the $\ell_{2}$ distance of $w_{k}$ from $w^{*}$ gets reduced by a constant fraction on an expectation (Lemma 7). Thus, it can be shown that running the algorithms for roughly $k_{\epsilon}=O\left(\log\frac{\beta\|\mathbf{w}_{1}-\mathbf{w}^{*}\|^{2}}{\epsilon}\right)$ epoch, would lead to $\|w_{k_{\epsilon}}-w^{*}\|^{2}\leq\epsilon$ , which in turn imply the $\epsilon$ -convergence (see the proof of Theorem 6 for details). The formal description of the algorithm is given in Algorithm 2.

# 4.2 Convergence Analysis for Smooth and Strongly Convex Functions

Theorem 6 (Convergence Analysis of Epoch-RGD for Smooth and Strongly convex Functions). Consider a dueling feedback optimization problem parameterized by any general admissible transfer

Algorithm 2 Epoch-RGD(ε)   
1: Input: Error tolerance $\epsilon > 0$ 2: Initialize Initial point: $\mathbf{w}_1 \in \mathbb{R}^d$ such that $\| \mathbf{w}_1 - \mathbf{w}^* \| \leq D$ , Phase count $k_\epsilon := \lceil \log_{4/3} \frac{\beta D^2}{2\epsilon} \rceil$ $D_1 = D$ , $B := \frac{2c_\rho p}{(\alpha + \beta)} \left( \left( \alpha^2 / 4\beta \right)^p \frac{\tilde{c}^{2p-1}}{d^{\frac{2p+1}{2}}} \right)$ , $\tilde{c} \leftarrow$ the universal constant from Lemma 8.
3: for $k = 1, 2, 3, \ldots, k_\epsilon$ do
4: $\eta_k \leftarrow BD_k^{2p+1}$ , $\gamma_k \leftarrow \frac{\tilde{c}\alpha D_k}{2\beta\sqrt{d}}$ , $t_k = \frac{1}{2B^2(D_k^2)^{2p}}$ , $D_{k+1} \leftarrow \sqrt{\frac{3}{4}} D_k$ .
5: Update $\mathbf{w}_{k+1} \leftarrow Relative-Gradient-Descent(\mathbf{w}_k, \eta_k, \gamma_k, t_k)$ 6: end for
7: Return $\mathbf{w}_{k_\epsilon + 1}$

function $\rho$ with $p$ -th order proxy $\tilde{\rho}_p$ and a $\beta$ smooth $\alpha$ -strongly convex function $f: \mathcal{D} \mapsto \mathbb{R}$ . Then given any $\epsilon > 0$ , the final point $\mathbf{w}_{k_\epsilon + 1}$ returned by Algorithm 2 satisfies $\mathbf{E}[f(\mathbf{w}_{t + 1})] - f(\mathbf{w}^*) \leq \epsilon$ , with a sample complexity of at most $O\left(\frac{1}{B^2\epsilon^{2p}}\right)$ pairwise comparisons. (Here the constant $B$ is as defined in Algorithm 2, $\tilde{c} = \frac{1}{20}$ is a universal constant.

Remark 2 (Optimal Convergence of Epoch-RGD for Strongly Convex and Smooth Functions). Note if $\rho (\mathbf{x})$ is exactly of the form $\rho (\mathbf{x}) = \mathrm{sign}(\mathbf{x})|x|^p$ , then our dueling (pairwise preference) feedback model is equivalent to the same used in Jamieson et al. (2012). It is interesting to note that, their derived lower bound sample complexity for the $\epsilon$ -convergence for smooth and strongly convex functions was indeed shown to be $\Omega (\epsilon^{-2p})$ which implies the optimality of Epoch-RGD (Algorithm 2) for $\alpha$ -strongly convex and $\beta$ -smooth functions for any values of $p$ . The line search algorithm proposed by Jamieson et al. (2012) also achieves the same convergence rate for strongly convex functions, modulo some additional multiplicative polylogarithmic terms in $d,\epsilon$ etc, which we do not incur. Also Epoch-RGD is much more modular and simpler to implement as well as relatively easier to analyze. Besides, the application scope of Epoch-RGD is much more general that applies to the class of any general transfer function (as discussed in Section 2) and also works for non-strongly convex functions (see Theorem 3).

Moreover, Theorem 6 shows that Algorithm 2 actually gives optimal rates for the special transfer functions studied earlier, e.g., $O(\epsilon^{-2})$ convergence rate for linear transfer function ( $p = 1$ ) a.k.a. zeroth-order feedback model (as proved in Hazan and Levy (2014)), or sigmoid based preference feedback (see Theorem 5). Besides it also yields the optimal $O(\log \frac{1}{\epsilon})$ convergence rate for sign feedback ( $p = 0$ ) (see Theorem 7 in Saha et al. (2021b)), since note our algorithm is essentially a generalization of Algorithm 2 of Saha et al. (2021b) which we can easily recover with the proper tuning of the algorithm parameters $\eta_k, \gamma_k, t_k$ ( $k \in [k_\epsilon]$ ).

Proof of Theorem 6 (sketch). The complete details of the proofs can be found in Appendix D. The proof of the main theorem is based on a key lemma that shows after every epoch of length $t_k$ , the distance of the resulting point $\mathbf{w}_{k+1}$ from the optimal $\mathbf{w}^*$ must decrease by at least a constant fraction. The formal statement is given below:

Lemma 7 (Epochwise Convergence Guarantee of Epoch-RGD). Consider the problem setup of Theorem 6. Then the point $w_{k+1}$ returned by k-th epoch run of Epoch-RGD (Algorithm 2) starting from the initial point $w_{k}$ , satisfies:

$$
\mathbf {E} [ \| \mathbf {w} _ {k + 1} - \mathbf {w} ^ {*} \| ^ {2} | \mathbf {w} _ {k} ] \leq \frac {3}{4} \| \mathbf {w} _ {k} - \mathbf {w} ^ {*} \| ^ {2},
$$

$\forall k \in [k_{\epsilon}]$ . Where the expectation is taken over the randomness of the algorithm and the dueling feedback received inside the run of Relative-Gradient-Descent $(\mathbf{w}_k, \eta_k, \gamma_k, t_k)$ .

The main part of the proof of Lemma 7 follows along the similar line of argument as of Lemma 4, however we need to carefully apply the properties of $\alpha$ strong-convexity of $f$ in order to achieve the improved convergence rates. The complete details can be found Appendix D.1. Given Lemma 7, claim of Theorem 6 now follows from the following epoch-wise recursion argument:

Let $\mathcal{H}_{[k]} := \{(\mathbf{w}_{k'})_{k' \in [k_\epsilon]}, (\mathbf{u}_{t'}, o_{t'})_{t' \in [\sum_{k'=1}^k t_{k'}]}\} \cup \{\mathbf{w}_{k+1}\}$ denotes the complete history till the end of epoch $k$ starting from the first epoch $\forall k \in [k_\epsilon]$ .

Further, let us denote by $\mathcal{H}_k := \{\mathbf{w}_k, (\mathbf{u}_{t'}, o_{t'})_{t' \in [\sum_{k'=1}^{k-1} t_{k'} + 1, \sum_{k'=1}^{k} t_{k'}]}\} \cup \{\mathbf{w}_{k+1}\}$ be the history only within epoch $k$ .

Proof of Correctness. From Lemma 7, note we have already established

$$
\mathbf {E} _ {\mathcal {H} _ {k}} [ \| \mathbf {w} _ {k + 1} - \mathbf {w} ^ {*} \| ^ {2} | \mathbf {w} _ {k} ] \leq \frac {3}{4} \| \mathbf {w} _ {k} - \mathbf {w} ^ {*} \| ^ {2}.
$$

Applying the argument iteratively over $E$ epochs, and the law of iterated expectations, we have:

$$
\mathbf {E} _ {\mathcal {H} _ {[ E ]}} [ \| \mathbf {w} _ {E + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq (3 / 4) ^ {E} \| \mathbf {w} _ {1} - \mathbf {w} ^ {*} \| ^ {2}. \tag {7}
$$

Thus choosing $E = \lceil \log_{4/3}(\frac{\beta D^2}{2\epsilon}) \rceil$ , where $\| \mathbf{w}_1 - \mathbf{w}^* \| \leq D$ , we have $E \geq \log_{4/3}(\frac{\beta \| \mathbf{w}_1 - \mathbf{w}^* \|^2}{2\epsilon}) \implies (3/4)^E \| \mathbf{w}_1 - \mathbf{w}^* \|^2 \leq \frac{2\epsilon}{\beta}$ .

Thus from (7), we get: $\mathbf{E}_{\mathcal{H}_{[E]}}[\| \mathbf{w}_{E + 1} - \mathbf{w}^{*}\|^{2}] \leq \frac{2\epsilon}{\beta}$ , and further applying $\beta$ -smoothness of $f$ , we get:

$$
\mathbf {E} _ {\mathcal {H} _ {[ E ]}} [ f (\mathbf {w} _ {E + 1}) - f (\mathbf {w} ^ {*}) ] \leq \mathbf {E} _ {\mathcal {H} _ {[ E ]}} [ \frac {\beta}{2} \| \mathbf {w} _ {E + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \epsilon ,
$$

which proves the correctness of Algorithm 2 for the choice of total number of epochs $k_{\epsilon} = E$ .

Proof of Sample Complexity. In order to verify that Algorithm 2 indeed converges to an $\epsilon$ -optimal point in $O(\epsilon^{-2p})$ sample complexity, note we simply need to count the total sample complexity incurred in the $k_{\epsilon}$ epochwise runs of Relative-Gradient-Descent (see Line #5 of Algorithm 2). However, by design of Epoch-RGD (Algorithm 2), since Relative-Gradient-Descent $\left(\mathbf{w}_k,\eta_k,\gamma_k,t_k\right)$ is run for only $t_k = \frac{1}{2B^2(D_k^2)^{2p}}$ iterations, the total sample complexity of Epoch-RGD becomes:

$$
\begin{array}{l} \sum_ {k = 1} ^ {k _ {\epsilon}} t _ {k} = \frac {1}{2 B ^ {2}} \sum_ {k = 1} ^ {k _ {\epsilon}} \frac {1}{(D _ {k} ^ {2}) ^ {2 p}} = \frac {1}{2 B ^ {2} (D ^ {2}) ^ {2 p}} \bigg (1 + \frac {1}{(3 / 4) ^ {2 p}} + \frac {1}{((3 / 4) ^ {2 p}) ^ {2}} + \dots + \frac {1}{((3 / 4) ^ {2 p}) ^ {k _ {\epsilon} - 1}} \bigg) \\ = \frac {1}{2 B ^ {2} (D ^ {2}) ^ {2 p}} \frac {(4 / 3 ^ {2 p}) ^ {k _ {\epsilon}} - 1}{4 / 3 ^ {2 p} - 1} \leq \frac {1}{4 B ^ {2} (D ^ {2}) ^ {2 p}} \Big ((\beta D ^ {2} / 2 \epsilon) ^ {2 p} - 1 \Big) = O \Big (\frac {1}{B ^ {2} \epsilon^ {2 p}} \Big) \\ \end{array}
$$

where the last inequality is since $k_{\epsilon} = \lceil \log_{4/3}(\frac{\beta D^2}{2\epsilon}) \rceil$ by definition. Thus follows the claimed sample complexity of Epoch-RGD in Theorem 6 and this concludes the proof.

# 5 Conclusion and Perspective

We consider the problem of convex optimization under a general class of pairwise preferences (dueling) feedback. Note the primary difficulty towards designing an efficient algorithm for the purpose lies in the fact that we can not hope to estimate the gradient of $f$ for any general dueling/pairwise preference model. Thus we can not apply the standard gradient descent based techniques to address this problem. We get around with the difficulty by estimating a $p$ -th order proxy of the gradient, called $p^{th}$ -degree-scaled Gradient. The crux of the idea lies in designing Relative-Gradient-Descent

based algorithm (Algorithm 1), which is a generalized notion of gradient descent based optimization technique. Using this we design an efficient algorithm with convergence rate of $\widetilde{O}(\epsilon^{-4p})$ for a smooth convex objective function, and an optimal rate of $\widetilde{O}(\epsilon^{-2p})$ when the objective is smooth and strongly convex.

Future work. Although the derived convergence rate for the strongly convex setting is information theoretically tight, the exact convergence lower bound is unclear for the class of smooth functions, which might be an interesting problem to pursue independently. Another open problem is to analyze this problem beyond the smoothness assumption. Considering a regret minimization objective instead of the optimization perspective, as well as understanding the information theoretic regret performance limit would be interesting direction as well. One can also consider generalizing the optimization framework to subsetwise preferences, instead of just pairwise (dueling) feedback. It might also be useful to extend our setup for contextual scenarios, adversarial preferences or non-stationary function sequences and understand the scopes of feasible solutions as well as the impossibility results.

# Acknowledgments

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 882396), the Israel Science Foundation (grant numbers 993/17; 2549/19), Tel Aviv University Center for AI and Data Science (TAD), the Len Blavatnik and the Blavatnik Family foundation, and the Yandex Initiative for Machine Learning at Tel Aviv University.

# References

Nir Ailon, Zohar Shay Karnin, and Thorsten Joachims. Reducing dueling bandits to cardinal bandits. In ICML, volume 32, pages 856–864, 2014.   
Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2-3):235–256, 2002.   
Viktor Bengs, Róbert Busa-Fekete, Adil El Mesaoudi-Paul, and Eyke Hüllermeier. Preference-based online learning with dueling bandits: A survey. Journal of Machine Learning Research, 2021.   
Stephen Boyd, Stephen P Boyd, and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.   
Brian Brost, Yevgeny Seldin, Ingemar J. Cox, and Christina Lioma. Multi-dueling bandits and their application to online ranker evaluation. CoRR, abs/1608.06253, 2016.   
Sébastien Bubeck. Convex optimization: Algorithms and complexity. arXiv preprint arXiv:1405.4980, 2014.   
Miroslav Dudík, Katja Hofmann, Robert E Schapire, Aleksandrs Slivkins, and Masrour Zoghi. Contextual dueling bandits. In Conference on Learning Theory, 2015.   
Abraham D Flaxman, Adam Tauman Kalai, and H Brendan McMahan. Online convex optimization in the bandit setting: gradient descent without a gradient. In Proceedings of the sixteenth annual ACM-SIAM symposium on Discrete algorithms, pages 385–394. Society for Industrial and Applied Mathematics, 2005.   
Roger Fletcher. Practical methods of optimization. John Wiley & Sons, 2013.   
Pratik Gajane, Tanguy Urvoy, and Fabrice Clérot. A relative exponential weighing algorithm for adversarial utility-based dueling bandits. In Proceedings of the 32nd International Conference on Machine Learning, pages 218–227, 2015.   
Javier González, Zhenwen Dai, Andreas Damianou, and Neil D Lawrence. Preferential bayesian optimization. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 1282–1291. JMLR. org, 2017.   
Bruce Hajek, Sewoong Oh, and Jiaming Xu. Minimax-optimal inference from partial rankings. In Advances in Neural Information Processing Systems, pages 1475-1483, 2014.   
Elad Hazan. Introduction to online convex optimization. arXiv preprint arXiv:1909.05207, 2019.   
Elad Hazan and Kfir Levy. Bandit convex optimization: Towards tight bounds. In Advances in Neural Information Processing Systems, pages 784-792, 2014.   
Kevin G Jamieson, Robert Nowak, and Ben Recht. Query complexity of derivative-free optimization. In Advances in Neural Information Processing Systems, pages 2672-2680, 2012.   
Kevin G Jamieson, Sumeet Katariya, Atul Deshpande, and Robert D Nowak. Sparse dueling bandits. In AISTATS, 2015.   
Ashish Khetan and Sewoong Oh. Data-driven rank breaking for efficient rank aggregation. Journal of Machine Learning Research, 17(193):1–54, 2016.

Wataru Kumagai. Regret analysis for continuous dueling bandit. In Advances in Neural Information Processing Systems 30, 2017.   
David G Luenberger, Yinyu Ye, et al. Linear and nonlinear programming, volume 2. Springer, 1984.   
Yurii Nesterov. Introductory lectures on convex optimization: A basic course, volume 87. Springer Science & Business Media, 2003.   
Jorge Nocedal and Stephen J Wright. Numerical optimization. Springer, 1999.   
Min-hwan Oh and Garud Iyengar. Thompson sampling for multinomial logit contextual bandits. In Advances in Neural Information Processing Systems, pages 3145-3155, 2019.   
Aadirupa Saha. Optimal algorithms for stochastic contextual preference bandits. Advances in Neural Information Processing Systems, 34, 2021.   
Aadirupa Saha and Akshay Krishnamurthy. Efficient and optimal algorithms for contextual dueling bandits under realizability. arXiv preprint arXiv:2111.12306, 2021.   
Aadirupa Saha, Tomer Koren, and Yishay Mansour. Adversarial dueling bandits. In International Conference on Machine Learning, 2021a.   
Aadirupa Saha, Tomer Koren, and Yishay Mansour. Dueling convex optimization. In International Conference on Machine Learning, pages 9245–9254. PMLR, 2021b.   
Ankan Saha and Ambuj Tewari. Improved regret guarantees for online smooth convex optimization with bandit feedback. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pages 636–642, 2011.   
Yanan Sui, Vincent Zhuang, Joel Burdick, and Yisong Yue. Multi-dueling bandits with dependent arms. In Conference on Uncertainty in Artificial Intelligence, UAI'17, 2017.   
Yisong Yue and Thorsten Joachims. Interactively optimizing information retrieval systems as a dueling bandits problem. In Proceedings of the 26th Annual International Conference on Machine Learning, pages 1201–1208, 2009.   
Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. The k-armed dueling bandits problem. Journal of Computer and System Sciences, 78(5):1538–1556, 2012.   
Masrour Zoghi, Shimon Whiteson, Remi Munos, Maarten de Rijke, et al. Relative upper confidence bound for the k-armed dueling bandit problem. In JMLR Workshop and Conference Proceedings, pages 10–18. JMLR, 2014a.   
Masrour Zoghi, Shimon A Whiteson, Maarten De Rijke, and Remi Munos. Relative confidence sampling for efficient on-line ranker evaluation. In Proceedings of the 7th ACM international conference on Web search and data mining, pages 73–82. ACM, 2014b.   
Masrour Zoghi, Zohar S Karnin, Shimon Whiteson, and Maarten De Rijke. Copeland dueling bandits. In Advances in Neural Information Processing Systems, pages 307–315, 2015.

# Supplementary:

# Dueling Convex Optimization with General Preferences

# A Useful Results (used in Sections 3, 3.2 and 4.2)

Lemma 8. For a given vector $\mathbf{g} \in \mathbb{R}^d$ and a random unit vector $\mathbf{u}$ drawn uniformly from $S_d(1)$ , we have

$$
\mathbf {E _ {u}} [ | \mathbf {g} \cdot \mathbf {u} | ] = \frac {\tilde {c} \| \mathbf {g} \|}{\sqrt {d}},
$$

for some universal constant $\tilde{c} \in [\frac{1}{20}, 1]$ .

Proof. Without loss of generality we can assume $\|g\|=1$ , since one can divide by $\|g\|$ in both side of Lem. 8 without affecting the claim. Now to bound $E[|g\cdot u|]$ , note that since u is drawn uniformly from $S_{d}(1)$ , by rotation invariance this equals $E[|u_{1}|]$ . For an upper bound, observe that by symmetry $E[u_{1}^{2}]=\frac{1}{d}E[\sum_{i=1}^{d}u_{i}^{2}]=\frac{1}{d}$ and thus

$$
\mathbf {E} [ | u _ {1} | ] \leq \sqrt {\mathbf {E} [ u _ {1} ^ {2} ]} = \frac {1}{\sqrt {d}}.
$$

We turn to prove a lower bound on $E[|g \cdot u|]$ . If u were a Gaussian random vector with i.i.d. entries $u_{i} \sim \mathcal{N}(0,1/d)$ , then from standard properties of the (truncated) Gaussian distribution we would have gotten that $E[|u_{1}|] = \sqrt{2/\pi d}$ . For u uniformly distributed on the unit sphere, $u_{i}$ is distributed as $v_{1}/||v||$ where v is Gaussian with i.i.d. entries $\mathcal{N}(0,1/d)$ . We then can write

$$
\begin{array}{l} \operatorname * {P r} \left(| u _ {1} | \geq \frac {\epsilon}{\sqrt {d}}\right) = \operatorname * {P r} \left(\frac {| v _ {1} |}{\| \mathbf {v} \|} \geq \frac {\epsilon}{\sqrt {d}}\right) \geq \operatorname * {P r} \left(| v _ {1} | \geq \frac {1}{\sqrt {d}} \text {   and   } \| \mathbf {v} \| \leq \frac {1}{\epsilon}\right) \\ \geq 1 - \operatorname * {P r} \left(\left| v _ {1} \right| <   \frac {1}{\sqrt {d}}\right) - \operatorname * {P r} \left(\left\| \mathbf {v} \right\| > \frac {1}{\epsilon}\right). \\ \end{array}
$$

Since $\sqrt{d} v_{1}$ is a standard Normal, we have

$$
\operatorname * {P r} \left(\left| v _ {1} \right| <   \frac {1}{\sqrt {d}}\right) = \operatorname * {P r} \left(- 1 <   \sqrt {d} v _ {1} <   1\right) = 2 \Phi (1) - 1 \leq 0. 7,
$$

and since $\mathbf{E}[\| \mathbf{v}\|^2] = 1$ an application of Markov's inequality gives

$$
\operatorname * {P r} \big (\| \mathbf {v} \| > \frac {1}{\epsilon} \big) = \operatorname * {P r} \big (\| \mathbf {v} \| ^ {2} > \frac {1}{\epsilon^ {2}} \big) \leq \epsilon^ {2} \mathbf {E} [ \| \mathbf {v} \| ^ {2} ] = \epsilon^ {2}.
$$

For $\epsilon = \frac{1}{4}$ this implies that $\Pr(|u_1| \geq 1/4\sqrt{d}) \geq \frac{1}{5}$ , whence $\mathbf{E}[|\mathbf{g} \cdot \mathbf{u}|] = \mathbf{E}[|u_1|] \geq 1/20\sqrt{d}$ .

![](images/3f75e909905949dc1db38867793d44a5ac3a55c41b98ba1640b77c84b11e8cdd.jpg)

Lemma 9. Let $g: \mathbb{R}^d \to \mathbb{R}$ be differentiable and let $\mathbf{u}$ be a random unit vector in $\mathbb{R}^d$ . Then

$$
\mathbf {E} [ g (\mathbf {u}) \mathbf {u} ] = \frac {1}{d} \mathbf {E} [ \nabla g (u) ].
$$

Proof. The above claim follows from Lemma 1 of Flaxman et al. (2005) which shows that for any differentiable function $f: \mathbb{R}^d \to \mathbb{R}$ and any $\mathbf{x} \in \mathbb{R}^d$ ,

$$
\mathbf {E} [ f (\mathbf {x} + \gamma \mathbf {u}) \mathbf {u} ] = \frac {\delta}{d} \nabla \mathbf {E} [ f (\mathbf {x} + \gamma \mathbf {u}) ] = \frac {\delta}{d} \mathbf {E} [ \nabla f (\mathbf {x} + \gamma \mathbf {u}) ].
$$

Fix $\mathbf{x} = 0$ and substitute $f(\mathbf{z}) = g(\frac{1}{\gamma}\mathbf{z})$ . Then $\nabla f(\mathbf{z}) = \frac{1}{\gamma}\nabla g(\frac{1}{\gamma}\mathbf{z})$ , and we obtain:

$$
\mathbf {E} [ g (\mathbf {u}) \mathbf {u} ] = \frac {1}{d} \mathbf {E} [ \nabla g (\mathbf {u}) ].
$$

![](images/a840fddd38b64be5c42953eb1859fc149f1580e97366b21ac0ab92c89a548861.jpg)

Lemma 10. Suppose $f: \mathcal{D} \mapsto \mathbb{R}$ is a convex function for some convex set $\mathcal{D} \subseteq \mathbb{R}^d$ such that for any $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ , $f(\mathbf{x}) - f(\mathbf{y}) > \epsilon$ . Then this implies $\| \nabla f(\mathbf{x}) \| > \frac{\epsilon}{\|\mathbf{x} - \mathbf{y}\|}$ . Further, assuming $D := \max_{\mathbf{x}, \mathbf{y} \in \mathbb{R}^d} \| \mathbf{x} - \mathbf{y} \|_2$ , we get $\| \nabla f(\mathbf{x}) \| > \frac{\epsilon}{D}$ for any $\mathbf{x} \in \mathcal{D}$ .

Proof. The proof simply follows using convexity of $f$ as:

$$
\begin{array}{l} f (\mathbf {x}) - f (\mathbf {y}) > \epsilon \implies \epsilon <   f (\mathbf {x}) - f (\mathbf {y}) \leq \nabla f (\mathbf {x}) (\mathbf {x} - \mathbf {y}) \leq \| \nabla f (\mathbf {x}) \| _ {2} \| \mathbf {x} - \mathbf {y} \| _ {2} \\ \Longrightarrow \| \nabla f (\mathbf {x}) \| \geq \frac {\epsilon}{\| \mathbf {x} - \mathbf {y} \|}. \\ \end{array}
$$

![](images/14c0bf02e0f583e40906013e5d65703f6d2f741993b735cf9c7c4d8c2df6a1b8.jpg)

# B Appendix for Section 3

Lemma 11 (Estimation of $p^{th}$ -degree-scaled Gradient from Dueling Feedback from general transfer function $\rho$ ). Consider any general admissible transfer function $\rho: \mathbb{R} \mapsto [-1,1]$ (that satisfies Assumption 1). Then if $f: \mathcal{D} \mapsto \mathbb{R}$ is L-Lipschitz and $\beta$ -smooth, given any $\mathbf{w} \in \mathcal{D}$ such that $r \geq \frac{2L\|\nabla f(\mathbf{w})\|}{\beta\sqrt{d}}$ , $o \sim \mathrm{Ber}^{\pm}\big(\rho(f(\mathbf{w} + \gamma\mathbf{u}) - f(\mathbf{w} - \gamma\mathbf{u}))\big)$ where $\mathbf{u} \sim \mathrm{Unif}(\mathcal{S}_d(1))$ , and suitably tuned step-size $\gamma > 0$ :

$$
\mathbf {E} _ {\mathbf {u}, o} [ o \mathbf {u} ] \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \geq \left\{ \begin{array}{l l} C (d, \beta , c _ {\rho}, p, r, L) \| \nabla f (\mathbf {w}) \| ^ {2 p - 1} \nabla \tilde {f} (\mathbf {w}) \cdot (\mathbf {w} - \mathbf {w} ^ {*}), & \text {for any p\geq 1}, \\ \frac {4 \gamma c _ {\rho}}{d} \nabla \tilde {f} (\mathbf {w}) \cdot (\mathbf {w} - \mathbf {w} ^ {*}), & \text {for p = 1}. \end{array} \right.
$$

Here $\nabla\tilde{f}(\mathbf{w})$ denotes the gradient of the smoothed function $\tilde{f}$ at w, where $\tilde{f}(\mathbf{w}):=E_{\mathbf{u}\sim Unif(S_{d}(1))}[f(\mathbf{w}+\gamma\mathbf{u})]$ , and $C(d,\beta,c_{\rho},p,r,L)$ is a constant dependent on the problem parameters $d,\beta,c_{\rho},p,r,L$ .

# B.1 Proof of Lemma 11

Lemma 11 (Estimation of $p^{th}$ -degree-scaled Gradient from Dueling Feedback from general transfer function $\rho$ ). Consider any general admissible transfer function $\rho: \mathbb{R} \mapsto [-1,1]$ (that satisfies Assumption 1). Then if $f: \mathcal{D} \mapsto \mathbb{R}$ is L-Lipschitz and $\beta$ -smooth, given any $\mathbf{w} \in \mathcal{D}$ such that $r \geq \frac{2L\|\nabla f(\mathbf{w})\|}{\beta\sqrt{d}}$ , $o \sim \mathrm{Ber}^{\pm}\big(\rho(f(\mathbf{w} + \gamma\mathbf{u}) - f(\mathbf{w} - \gamma\mathbf{u}))\big)$ where $\mathbf{u} \sim \mathrm{Unif}(\mathcal{S}_d(1))$ , and suitably tuned step-size $\gamma > 0$ :

$$
\mathbf {E} _ {\mathbf {u}, o} [ o \mathbf {u} ] \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \geq \left\{ \begin{array}{l l} C (d, \beta , c _ {\rho}, p, r, L) \| \nabla f (\mathbf {w}) \| ^ {2 p - 1} \nabla \tilde {f} (\mathbf {w}) \cdot (\mathbf {w} - \mathbf {w} ^ {*}), & \text {for any p\geq 1}, \\ \frac {4 \gamma c _ {\rho}}{d} \nabla \tilde {f} (\mathbf {w}) \cdot (\mathbf {w} - \mathbf {w} ^ {*}), & \text {for p = 1}. \end{array} \right.
$$

Here $\nabla\tilde{f}(\mathbf{w})$ denotes the gradient of the smoothed function $\tilde{f}$ at w, where $\tilde{f}(\mathbf{w}):=\mathbf{E}_{\mathbf{u}\sim\operatorname{Unif}(\mathcal{S}_{d}(1))}[f(\mathbf{w}+\gamma\mathbf{u})]$ , and $C(d,\beta,c_{\rho},p,r,L)$ is a constant dependent on the problem parameters $d,\beta,c_{\rho},p,r,L$ .

Proof. We start by noting that given w and denoting g = ou:

$$
\begin{array}{l} \mathbf {E} _ {\mathbf {u}, o} [ \mathbf {g} ] = \mathbf {E} _ {\mathbf {u}} [ \mathbf {E} _ {o} [ o \mathbf {u} \mid \mathbf {u} ] ] \\ = \mathbf {E _ {u}} [ \rho (f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u})) \mathbf {u} ] \\ \end{array}
$$

Let us start with the case for $p \geq 1$ . Note since $\rho$ is differentiable by assumption (see Assumption 1-i) as well as $f$ , using Lemma 9 in the first equality, we get:

$$
\begin{array}{l} \mathbf {E} _ {\mathbf {u}} \big [ \rho (f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u})) \mathbf {u} \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \big ] \\ = \frac {\gamma}{d} \mathbf {E} _ {\mathbf {u}} \left[ \rho^ {\prime} (f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u})) (\nabla f (\mathbf {w} + \gamma \mathbf {u}) + \nabla f (\mathbf {w} - \gamma \mathbf {u})) \right] \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \\ \geq \frac {2 \gamma}{d} \mathbf {E} _ {\mathbf {u}} \left[ \tilde {\rho} _ {p} ^ {\prime} \big (| f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u}) | \big) \big (\nabla f (\mathbf {w} + \gamma \mathbf {u}) + \nabla f (\mathbf {w} - \gamma \mathbf {u}) \big) \right] \cdot (\mathbf {w} - \mathbf {w} ^ {*}), \tag {8} \\ \end{array}
$$

where the second equality follows since $\rho'(-x)=\rho'(x)$ by the anti-symmetry of $\rho$ (see Assumption 1-(i)). Note in Equation (8), we further lower bounded $\rho'$ by $\tilde{\rho}_{p}'$ using Assumption 1-(ii) along with the observation that $\nabla f(\mathbf{w}+\gamma\mathbf{u})\cdot(\mathbf{w}-\mathbf{w}^{*})$ is positive for any $\mathbf{u}\in\mathcal{S}_{d}(1)$ by first order optimality conditions.

It is important to note that, to ensure r-proximity to the origin, as needed in Assumption 1-(ii), it needed to satisfy $|f(\mathbf{w} + \gamma\mathbf{u}) - f(\mathbf{w} - \gamma\mathbf{u})| \leq r$ , which we ensured by choosing $\gamma \leq \frac{r}{2L}$ (as recall by assumption f is L-lipschitz).

Case: p = 1. Note in this case from Equation (8) we get:

$$
\begin{array}{l} \mathbf {E} _ {\mathbf {u}} \left[ \rho (f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u})) \mathbf {u} \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \right] \\ \geq \frac {2 \gamma}{d} \mathbf {E _ {u}} \big [ c _ {\rho} \big (\nabla f (\mathbf {w} + \gamma \mathbf {u}) + \nabla f (\mathbf {w} - \gamma \mathbf {u}) \big) \big ] \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \\ = \frac {2 \gamma c _ {\rho}}{d} \mathbf {E _ {u}} \big [ \big (\nabla f (\mathbf {w} + \gamma \mathbf {u}) + \nabla f (\mathbf {w} - \gamma \mathbf {u}) \big) \big ] \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \\ = \frac {2 \gamma c _ {\rho}}{d} \big (\nabla \mathbf {E _ {u}} \big [ f (\mathbf {w} + \gamma \mathbf {u}) \big ] + \nabla \mathbf {E _ {u}} \big [ f (\mathbf {w} - \gamma \mathbf {u}) \big ] \big) \cdot (\mathbf {w} - \mathbf {w} ^ {*}) = \frac {4 \gamma c _ {\rho}}{d} \nabla \tilde {f} (\mathbf {w}) \cdot (\mathbf {w} - \mathbf {w} ^ {*}), \\ \end{array}
$$

where the inequality follows since when p = 1, $\tilde{\rho}_{p}^{\prime}(x) = c_{\rho}$ (independent of $x, \forall x \in R$ ). The last equality follows by exchanging expectation and derivative (since f is differentiable and finite valued by assumptions).

Case: $p \geq 1$ . Now lets consider the case for any general $p \geq 1$ : Now for the first term in Equation (8), again applying the $\beta$ -smoothness of f we get:

$$
\begin{array}{l} \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) - \frac {1}{2} \beta \gamma^ {2} \leq f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w}) \leq \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) + \frac {1}{2} \beta \gamma^ {2} \\ - \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) - \frac {1}{2} \beta \gamma^ {2} \leq f (\mathbf {w} - \gamma \mathbf {u}) - f (\mathbf {w}) \leq - \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) + \frac {1}{2} \beta \gamma^ {2}. \\ \end{array}
$$

Subtracting the inequalities, we get

$$
\begin{array}{l} | f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u}) - 2 \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) | \leq \beta \gamma^ {2} \\ \Longrightarrow 2 \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) - \beta \gamma^ {2} \leq f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u}) \leq 2 \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) + \beta \gamma^ {2}. \\ \end{array}
$$

Note above implies: $|f(\mathbf{w} + \gamma\mathbf{u}) - f(\mathbf{w} - \gamma\mathbf{u})| \geq |2\gamma\mathbf{u} \cdot \nabla f(\mathbf{w})| - \beta\gamma^{2}$ . Now using Lem. 4 of Saha et al. (2021b), we know that:

$$
\mathbf {P} _ {\mathbf {u}} \big (| \mathbf {u} ^ {\top} \nabla f (\mathbf {w}) | \geq \beta \gamma \big) \geq 1 - \lambda , \text {where} \lambda = \inf _ {\gamma^ {\prime} > 0} \left\{\gamma^ {\prime} + \frac {2 \beta \gamma \sqrt {d \log (1 / \gamma^ {\prime})}}{\| \nabla f (\mathbf {w}) \|} \right\}
$$

for any $\mathbf{w} \in \mathbb{R}^d$ . Note choosing $\gamma' = \frac{\beta\gamma\sqrt{d}}{\|\nabla f(\mathbf{w})\|}$ , we get $\lambda \leq \frac{\beta\gamma\sqrt{d}}{\|\nabla f(\mathbf{x})\|} \left(1 + 2\sqrt{\log\frac{\|\nabla f(\mathbf{x})\|}{\sqrt{d}\beta\gamma}}\right)$ . Note if we further choose $\gamma \leq \frac{\|\nabla f(\mathbf{w})\|}{10\beta\sqrt{d}}$ , we have $\lambda \leq 1/2$ .

Thus we have: $\mathbf{P}_{\mathbf{u}}\big(|f(\mathbf{w} + \gamma \mathbf{u}) - f(\mathbf{w} - \gamma \mathbf{u})| \geq \beta \gamma^2\big) \geq 1 - \lambda$ , and using Equation (8), we further get:

$$
\begin{array}{l} \mathbf {E} _ {\mathbf {u}} \left[ \rho (f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u})) \mathbf {u} \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \right] \\ \geq \frac {2 \gamma (1 - \lambda)}{d} \mathbf {E} _ {\mathbf {u}} \left[ \tilde {\rho} _ {p} ^ {\prime} (\beta \gamma^ {2}) (\nabla f (\mathbf {w} + \gamma \mathbf {u}) + \nabla f (\mathbf {w} - \gamma \mathbf {u})) \right] \cdot (\mathbf {w} - \mathbf {w} ^ {*}), \tag {9} \\ \end{array}
$$

Here it is important to note that above lower bound holds since (a) from first order optimality conditions for any $\mathbf{u} \in S_d(1)$ , $f(\mathbf{w} + \gamma\mathbf{u}) \cdot (\mathbf{w} - \mathbf{w}^*) > 0$ , and also (b) $\tilde{\rho}_p'(x) > 0$ for any $x > 0$ by definition of $\tilde{\rho}_p'$ .

Thus from Equation (9) we further get:

$$
\begin{array}{l} \mathbf {E} _ {\mathbf {u}} \left[ \rho (f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u})) \mathbf {u} \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \right] \\ \geq \frac {2 \gamma (1 - \lambda) \tilde {\rho} _ {p} ^ {\prime} (\beta \gamma^ {2})}{d} \mathbf {E} _ {\mathbf {u}} \left[ \left(\nabla f (\mathbf {w} + \gamma \mathbf {u}) + \nabla f (\mathbf {w} - \gamma \mathbf {u})\right) \right] \cdot \left(\mathbf {w} - \mathbf {w} ^ {*}\right) \\ = \frac {2 c _ {\rho} p (1 - \lambda) \gamma^ {2 p - 1} \beta^ {p - 1}}{d} \left(\nabla \tilde {f} (\mathbf {w} + \gamma \mathbf {u}) + \nabla \tilde {f} (\mathbf {w} - \gamma \mathbf {u})\right) \cdot \left(\mathbf {w} - \mathbf {w} ^ {*}\right), \tag {10} \\ \end{array}
$$

The result now follows by noting $\tilde{\rho}_p'(x) = c_\rho px^{p-1}$ for any $x \in \mathbb{R}_+$ , $\nabla \tilde{f}(\mathbf{w}) = \mathbf{E}_{\mathbf{u} \sim \mathrm{Unif}(\mathcal{S}_d(1))}[\nabla f(\mathbf{w} + \gamma \mathbf{u})]$ and choosing $\gamma = \min\left(\frac{r}{2L}, \frac{\|\nabla f(\mathbf{w})\|}{10\beta\sqrt{d}}\right)$ . That concludes the proof for any $p \geq 1$ .

Remark 3. It is also worth noting that, for $p = 0$ , $c_{\rho} = 1$ , our transfer function recovers the sign feedback of Saha et al. (2021b). In this case it can be shown that:

$$
\mathbf {E} _ {\mathbf {u}, o} [ o \mathbf {u} ] \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \geq \frac {1}{4 0 \sqrt {d}} \frac {\nabla f (\mathbf {w}) \cdot (\mathbf {w} - \mathbf {w} ^ {*})}{\| \nabla f (\mathbf {w}) \|},
$$

which essentially recovers the normalized gradient estimate (or the direction of the gradient).

The claim for $p = 0$ simply follows by consecutively applying Lemma 4 and 3 of Saha et al. (2021b) as follows: From Lemma 4 of Saha et al. (2021b) we have:

$$
\mathbf {E} _ {\mathbf {u}} [ \operatorname{sign} (f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u})) \mathbf {u} ] \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \geq (1 - \lambda) \mathbf {E} _ {\mathbf {u}} [ \operatorname{sign} (\nabla f (\mathbf {w}) \cdot \mathbf {u}) \mathbf {u} ] \cdot (\mathbf {w} - \mathbf {w} ^ {*}),
$$

where $\lambda$ is as defined in the Case for $p \geq 1$ above. But using Lemma 3 of Saha et al. (2021b) we further get:

$$
\mathbf {E} _ {\mathbf {u}} [ \text { sign } (f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u})) \mathbf {u} ] \cdot (\mathbf {w} - \mathbf {w} ^ {*}) \geq \frac {1}{4 0 \sqrt {d}} \frac {\nabla f (\mathbf {w}) \cdot (\mathbf {w} - \mathbf {w} ^ {*})}{\| \nabla f (\mathbf {w}) \|},
$$

which concludes the claim choosing $\gamma = \min \left(\frac{r}{2L},\frac{\|\nabla f(\mathbf{w})\|}{10\beta\sqrt{d}}\right)$ (noting that $\lambda \leq 1/2$ for this choice of $\gamma$ as explained in the case for $p \geq 1$ above).

# C Appendix for Section 3.2

Notations. We denote by $\mathcal{H}_t$ the history $\{\mathbf{w}_{\tau},\mathbf{u}_{\tau},o_{\tau}\}_{\tau = 1}^{t - 1}\cup \mathbf{w}_t$ till time $t$ , for all $t\in [T]$ . $\mathbf{E}_t[\cdot ]\coloneqq \mathbf{E}_{o_t,\mathbf{u}_t}[\cdot |\mathcal{H}_t]$ denote the expectation with respect to $\mathbf{u}_t,o_t$ given $\mathcal{H}_t$

# C.1 Proof of Lemma 4

Lemma 4 (Roundwise Progress of Relative-Gradient-Descent). Consider the problem setup of Theorem 3 and also the choice of $\eta$ , $\gamma$ . Then at any time t, during the run of Relative-Gradient-Descent (Algorithm 1), given $H_{t}$ , if $f(\mathbf{w}_{t}) - f(\mathbf{w}^{*}) > \epsilon$ , we can show that:

$$
\mathbf {E} _ {t} \left[ \left\| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \right\| ^ {2} \right] \leq \left\| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \right\| ^ {2} - \frac {p ^ {2} \left(\tilde {c} ^ {2 p - 1} c _ {\rho}\right) ^ {2} \epsilon^ {4 p}}{d ^ {2 p + 1} \beta^ {2 p} D ^ {4 p - 2}} \tag {1}
$$

Complete Proof of Lemma 4. First note by our update rule,

$$
\begin{array}{l} \mathbf {E} _ {t} [ \| \tilde {\mathbf {w}} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} ] = \mathbf {E} _ {t} [ \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} ] - 2 \eta \mathbf {E} _ {t} [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] + \eta^ {2} \mathbf {E} _ {t} \| \mathbf {u} _ {t} \| ^ {2} \\ = \mathbf {E} _ {t} [ \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} ] - 2 \eta \mathbf {E} _ {t} [ [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] ] + \eta^ {2}. \tag {11} \\ \end{array}
$$

But since projection to D reduces distance from $w^{*}$ we have:

$$
\left\| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \right\| ^ {2} \leq \left\| \tilde {\mathbf {w}} _ {t + 1} - \mathbf {w} ^ {*} \right\| ^ {2}
$$

This further implies:

$$
\| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} \leq \| \mathbf {w} _ {t + 1} ^ {\prime} - \mathbf {w} _ {t + 1} \| ^ {2} + \| \mathbf {w} _ {t + 1} ^ {\prime} - \mathbf {w} ^ {*} \| ^ {2} \leq \gamma^ {2} + \| \tilde {\mathbf {w}} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2}.
$$

Applying above in Equation (11) we get:

$$
\begin{array}{l} \mathbf {E} _ {t} [ \| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \mathbf {E} _ {t} [ \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} ] - 2 \eta \mathbf {E} _ {t} [ [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] ] + \eta^ {2} \\ = \left\| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \right\| ^ {2} - 2 \eta \mathbf {E} _ {t} \left[ \left[ \mathbf {g} _ {t} \cdot \left(\mathbf {w} _ {t} - \mathbf {w} ^ {*}\right) \right] \right] + \eta^ {2}. \tag {12} \\ \end{array}
$$

On the other hand, since both $f$ and $\rho$ is convex (by assumption), using Lemma 9 we get:

$$
\begin{array}{l} \mathbf {E} _ {t} \left[ \mathbf {g} _ {t} \cdot \left(\mathbf {w} _ {t} - \mathbf {w} ^ {*}\right) \right] = \mathbf {E} _ {\mathbf {u} _ {t}} \left[ \rho \left(f \left(\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}\right) - f \left(\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}\right)\right) \mathbf {u} _ {t} \cdot \left(\mathbf {w} _ {t} - \mathbf {w} ^ {*}\right) \mid \mathcal {H} _ {t} \right] \\ = \mathbf {E} _ {\mathbf {u} _ {t}} \left[ \rho (f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t})) \cdot \mathbf {u} _ {t} \mid \mathcal {H} _ {t} \right] \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) \\ = \frac {\gamma}{d} \mathbf {E _ {u}} _ {t} \left[ \rho^ {\prime} \big (f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) \big) \big (\nabla f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) + \nabla f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) \big) \mid \mathcal {H} _ {t} \right] \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) \\ = \frac {\gamma}{d} \mathbf {E} _ {\mathbf {u} _ {t}} \left[ \rho^ {\prime} \left(| f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) |\right) \left(\nabla f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) + \nabla f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t})\right) | \mathcal {H} _ {t} \right] \cdot \left(\mathbf {w} _ {t} - \mathbf {w} ^ {*}\right), \tag {13} \\ \end{array}
$$

where the last equality follows since $\rho(-x) = -\rho(x)$ (see Assumption 1-i). Now, since f is convex and $\beta$ -smooth, we have that:

$$
\nabla f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) \geq f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} ^ {*} + \gamma \mathbf {u} _ {t})
$$

$$
\geq f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*}) + \nabla f (\mathbf {w} _ {t}) \cdot (\gamma \mathbf {u} _ {t}) - \nabla f (\mathbf {w} ^ {*}) \cdot (\gamma \mathbf {u} _ {t}) - \beta \| \gamma \mathbf {u} _ {t} \| ^ {2}
$$

$$
= f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*}) + \gamma (\nabla f (\mathbf {w} _ {t}) - \nabla f (\mathbf {w} ^ {*})) \cdot \mathbf {u} _ {t} - \beta \gamma^ {2}.
$$

Likewise, for the term $\nabla f(\mathbf{w}_t - \gamma \mathbf{u}_t) \cdot (\mathbf{w}_t - \mathbf{w}^*)$ we get,

$$
\nabla f \left(\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}\right) \cdot \left(\mathbf {w} _ {t} - \mathbf {w} ^ {*}\right) \geq f \left(\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}\right) - f \left(\mathbf {w} ^ {*} - \gamma \mathbf {u} _ {t}\right)
$$

$$
\begin{array}{l} \geq f \left(\mathbf {w} _ {t}\right) - f \left(\mathbf {w} ^ {*}\right) - \nabla f \left(\mathbf {w} _ {t}\right) \cdot \left(\gamma \mathbf {u} _ {t}\right) + \nabla f \left(\mathbf {w} ^ {*}\right) \cdot \left(\gamma \mathbf {u} _ {t}\right) - \beta \| \gamma \mathbf {u} _ {t} \| ^ {2} \\ = f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*}) - \gamma (\nabla f (\mathbf {w} _ {t}) - \nabla f (\mathbf {w} ^ {*})) \cdot \mathbf {u} _ {t} - \beta \gamma^ {2}. \\ \end{array}
$$

Summing, we thus get:

$$
\left(\nabla f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) + \nabla f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t})\right) \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) \geq 2 (f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*})) - 2 \beta \gamma^ {2}. \tag {14}
$$

Now since by Assumption 1-(2) we have $\rho'(x) > 0$ for any $x \in (0, r]$ , we can claim that

$$
\rho^ {\prime} \big (| f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | \big) > 0;
$$

see Remark 4 for a formal justification (on choices of $\gamma$ to satisfy $|f(\mathbf{w}_t + \gamma \mathbf{u}_t) - f(\mathbf{w}_t - \gamma \mathbf{u}_t)| \leq r$ ).

Remark 4. Recall from Assumption 1, there exists some constants $p, r, c_{\rho} > 0$ such that $|\rho(x)| \geq |\tilde{\rho}_p(x)|$ for all $x \in [-r, r]$ . Clearly we apply inequality (a) for $x = f(\mathbf{w}_t + \gamma\mathbf{u}_t) - f(\mathbf{w}_t - \gamma\mathbf{u}_t)$ . Now to justify indeed $|f(\mathbf{w}_t + \gamma\mathbf{u}_t) - f(\mathbf{w}_t - \gamma\mathbf{u}_t)| \leq r$ , we note that since $f$ is $\beta$ -smooth, it is also locally-lipschitz inside the bounded domain $\mathcal{D}$ and suppose $L$ is the resulting lipschitz constant. Then we have $|f(\mathbf{w}_t + \gamma\mathbf{u}_t) - f(\mathbf{w}_t - \gamma\mathbf{u}_t)| \leq 2L\gamma$ , and to ensure the above condition, we can assume $2L\gamma \leq r$ by choosing small enough $\gamma$ . But since we set $\gamma = \frac{\tilde{c}\epsilon}{\beta D\sqrt{d}}$ , note the constraints are satisfied for any $\epsilon \leq \frac{r\beta D\sqrt{d}}{2L\tilde{c}}$ .

For simplicity, let us denote $E_{u_{t}}[\cdot]:=E_{u_{t}}[\cdot\mid\mathcal{H}_{t}]$ . Recall, since $\rho^{\prime}\big(|f(\mathbf{w}_{t}+\gamma\mathbf{u}_{t})-f(\mathbf{w}_{t}-\gamma\mathbf{u}_{t})|\big)>0$ by Assumption 1, now using Equations (13) and (14), we can write:

$$
\begin{array}{l} \mathbf {E} _ {t} [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] = \frac {\gamma}{d} \mathbf {E} _ {\mathbf {u} _ {t}} \big [ \rho^ {\prime} \big (| f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | \big) \big (\nabla f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) + \nabla f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) \big) \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) \big ] \\ \geq \frac {2 \gamma}{d} \mathbf {E _ {u}} _ {t} \left[ \rho^ {\prime} \big (| f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | \big) \big (f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*}) - \beta \gamma^ {2} \big) \right] \\ \geq \frac {2 \gamma}{d} \mathbf {E _ {u t}} \left[ \rho^ {\prime} \big (| f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | \big) \right] \big (f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*}) - \beta \gamma^ {2} \big) \\ > \frac {2 \gamma}{d} \mathbf {E _ {u t}} \left[ \rho^ {\prime} \big (| f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | \big) \big (\epsilon - \beta \gamma^ {2} \big) \right. \mathrm{(sinceweconditionedon} f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*}) > \epsilon), \\ > \frac {2 \gamma}{d} \mathbf {E} _ {\mathbf {u} _ {t}} \left[ \rho^ {\prime} \big (| f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | \big) \right] \epsilon / 2, \tag {15} \\ \end{array}
$$

where the second last inequality follows since we choose $\gamma = \frac{\tilde{c}\epsilon}{\beta\sqrt{d}D}$ (recall from Theorem 3). However since $\tilde{c} < 1$ , note we have $\gamma \leq \frac{\epsilon}{\beta\sqrt{d}D}$ and further since $D > \sqrt{\frac{2\epsilon}{\beta}}$ , above in turn implies $\gamma < \sqrt{\frac{\epsilon}{2\beta}}$ and hence $(\epsilon - \beta\gamma^{2}) > \frac{\epsilon}{2}$ . Last equality is from Assumption 1-(ii) where recall that $\tilde{\rho}_{p}(x) = c_{\rho}\operatorname{sign}(x)|x|^{p}$ for all denotes the p-th order proxy of $\rho$ and hence $\rho'(x) \geq c_{\rho}px^{p-1} = \tilde{\rho}_{p}'(x)$ for $x \in [0, r]$ . Now applying the $\beta$ -smoothness of f:

$$
\begin{array}{l} \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) - \frac {1}{2} \beta \gamma^ {2} \leq f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w}) \leq \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) + \frac {1}{2} \beta \gamma^ {2} \\ - \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) - \frac {1}{2} \beta \gamma^ {2} \leq f (\mathbf {w} - \gamma \mathbf {u}) - f (\mathbf {w}) \leq - \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) + \frac {1}{2} \beta \gamma^ {2}. \\ \end{array}
$$

Subtracting the inequalities, we get

$$
\begin{array}{l} | f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u}) - 2 \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) | \leq \beta \gamma^ {2} \\ \Longrightarrow 2 \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) - \beta \gamma^ {2} \leq f (\mathbf {w} + \gamma \mathbf {u}) - f (\mathbf {w} - \gamma \mathbf {u}) \leq 2 \gamma \mathbf {u} \cdot \nabla f (\mathbf {w}) + \beta \gamma^ {2}. \\ \end{array}
$$

Note above implies: $|f(\mathbf{w}_t + \gamma \mathbf{u}_t) - f(\mathbf{w}_t - \gamma \mathbf{u}_t)| \geq |2\gamma \mathbf{u}_t \cdot \nabla f(\mathbf{w}_t)| - \beta \gamma^2$ .

Now taking expectation over $\mathbf{u}_t$ in both side:

$$
\mathbf {E _ {u}} _ {t} [ | f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | ] \geq \mathbf {E _ {u}} _ {t} [ | 2 \gamma \mathbf {u} _ {t} \cdot \nabla f (\mathbf {w} _ {t}) | ] - \beta \gamma^ {2} = 2 \frac {\tilde {c} \gamma \| \nabla f (\mathbf {w} _ {t}) \|}{\sqrt {d}} - \beta \gamma^ {2}
$$

where the last equality is due to Lemma 8. Additionally, since we assumed $f(\mathbf{w}_t) - f(\mathbf{w}^*) > \epsilon$ , i.e. the suboptimality gap to be at least $\epsilon$ , by Lemma 10 we can further derive a lower bound:

$$
\mathbf {E _ {u}} _ {t} [ | f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) | ] \geq \frac {2 \tilde {c} \gamma \epsilon}{\sqrt {d} \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \|} - \beta \gamma^ {2} = \frac {2 \tilde {c} \gamma \epsilon}{\sqrt {d} D} - \beta \gamma^ {2}.
$$

And now note that setting $\gamma=\frac{\tilde{c}\epsilon}{\beta\sqrt{dD}}$ , the right hand side is positive. Now, lower bounding $\rho^{\prime}$ by $\tilde{\rho}_{p}^{\prime}$ as per Assumption 1-(ii) and further applying monotonicity of $\tilde{\rho}_{p}^{\prime}(\cdot)$ in the positive orthant, we get:

$$
\rho^ {\prime} \left(\mathbf {E} _ {\mathbf {u} _ {t}} \left[ | f \left(\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}\right) - f \left(\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}\right) | \right]\right) \geq \rho^ {\prime} \left(\mathbf {E} _ {\mathbf {u} _ {t}} \left[ | f \left(\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}\right) - f \left(\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}\right) | \right]\right) \geq \tilde {\rho} _ {p} ^ {\prime} \left(\frac {2 \tilde {c} \gamma \epsilon}{\sqrt {d} D} - \beta \gamma^ {2}\right). \tag {16}
$$

Then from Equation (15):

$\mathbf{E}_t[\mathbf{g}_t \cdot (\mathbf{w}_t - \mathbf{w}^*)] \geq \frac{2\gamma}{d} \tilde{\rho}_p' \left( \frac{2\tilde{c}\gamma\epsilon}{\sqrt{d}D} - \beta\gamma^2 \right) \epsilon/2$ (applying the bound from from (16))

$$
\geq \frac {2 \gamma}{d} \left(c _ {\rho} p \gamma^ {p - 1} \mid \frac {2 \tilde {c} \epsilon}{\sqrt {d} D} - \beta \gamma | ^ {p - 1}\right) \epsilon / 2 (\text { as }, \tilde {\rho} _ {p} ^ {\prime} (x) = c _ {\rho} p x ^ {p - 1} \text {   for   any   } x \in \mathbb {R} _ {+}), \tag {17}
$$

Then for the above choice of $\gamma=\frac{\tilde{c}\epsilon}{\beta\sqrt{d}D}$ , we finally get:

$$
\begin{array}{l} \mathbf {E} _ {t} [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] \geq \frac {2 \gamma \epsilon}{2 d} \big (c _ {\rho} p \gamma^ {p - 1} | \frac {\tilde {c} ^ {2} \epsilon^ {2}}{d D ^ {2}} | ^ {p - 1} \big) \\ = \frac {p c _ {\rho} \tilde {c} ^ {2 p - 1} \epsilon^ {2 p}}{d ^ {(2 p + 1) / 2} \beta^ {p} D ^ {2 p - 1}}, \tag {18} \\ \end{array}
$$

Combining Equation (12) with Equation (18):

$$
\mathbf {E} _ {t} [ \| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - 2 \eta \mathbf {E} _ {t} [ [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] ] + \eta^ {2}
$$

$$
\leq \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - 2 \eta \left(\frac {p c _ {\rho} \tilde {c} ^ {2 p - 1} \epsilon^ {2 p}}{d ^ {(2 p + 1) / 2} \beta^ {p} D ^ {2 p - 1}}\right) + \eta^ {2}
$$

$$
\leq \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - \left(\frac {p c _ {\rho} \tilde {c} ^ {2 p - 1} \epsilon^ {2 p}}{d ^ {(2 p + 1) / 2} \beta^ {p} D ^ {2 p - 1}}\right) ^ {2}, \quad \mathrm{setting} \eta = \left(\frac {p c _ {\rho} \tilde {c} ^ {2 p - 1} \epsilon^ {2 p}}{d ^ {(2 p + 1) / 2} \beta^ {p} D ^ {2 p - 1}}\right),
$$

which concludes the claim of Lemma 4.

![](images/0b4cd9a5144f0124d7a0ddbcf3b6b8b6409d3c539128cc4c9b447366c17d4d33.jpg)

# C.2 Proof of Theorem 3

Theorem 3. Consider a dueling feedback optimization problem parameterized by any general admissible transfer function $\rho$ with $p$ -th order proxy $\tilde{\rho}_p$ and a $\beta$ smooth convex function $f: \mathcal{D} \mapsto \mathbb{R}$ . Then given any $\epsilon > 0$ , for the choice of $\gamma = \frac{\tilde{c}\epsilon}{\beta\sqrt{d}D}$ and $\eta = \frac{pc_{\rho}\tilde{c}^{2p-1}\epsilon^{2p}}{d^{(2p+1)/2}\beta^p D^{2p-1}}$ , there exists at least one $t$ such that $\mathbf{E}[f(\mathbf{w}_t)] - f(\mathbf{w}^*) \leq \epsilon$ , after at most $T = \frac{d^{2p+1}\beta^{2p}D^{4p}}{p^2(\tilde{c}^{2p-1}c_\rho)^2\epsilon^{4p}} + 1$ iterations; i.e. $\min_{t \in [T]} \mathbf{E}[f(\mathbf{w}_t)] - f(\mathbf{w}^*) \leq \epsilon$ , where $\tilde{c} = \frac{1}{20}$ is a universal constant.

Complete Proof of Theorem 3. We denote by $\mathcal{H}_t$ the history $\{\mathbf{w}_{\tau},\mathbf{u}_{\tau},o_{\tau}\}_{\tau = 1}^{t - 1}\cup \mathbf{w}_t$ till time $t$ .

We start by noting that by definition:

$$
\mathbf {E} _ {o _ {t}} [ \mathbf {g} _ {t} | \mathcal {H} _ {t}, \mathbf {u} _ {t} ] = \rho (f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t})) \mathbf {u} _ {t}
$$

We proceed with the proof inductively, i.e. given $H_{t}$ and assuming (conditioning on) $f(\mathbf{w}_{t}) - f(\mathbf{w}^{*}) > \epsilon$ , we can show that $w_{t+1}$ always come closer to the minimum $w^{*}$ on expectation in terms of the $\ell_{2}$ -norm. More formally, given $H_{t}$ and assuming $f(\mathbf{w}_{t}) - f(\mathbf{w}^{*}) > \epsilon$ we will show:

$$
\mathbf {E} _ {t} [ \left\| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \right\| ^ {2} ] \leq \left\| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \right\| ^ {2}
$$

where $E_{t}[\cdot]:=E_{o_{t},u_{t}}[\cdot\mid\mathcal{H}_{t}]$ denote the expectation with respect to $u_{t},o_{t}$ given $H_{t}$ . The precise statement is given by Lemma 4.

Given the statement of Lemma 4, now note that by iteratively taking expectation over $\mathcal{H}_T$ on both sides of Equation (1) and summing over $t = 1,\dots ,T$ , we get,

$$
\mathbf {E} _ {\mathcal {H} _ {T}} [ \| \mathbf {w} _ {T + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {w} _ {1} - \mathbf {w} ^ {*} \| ^ {2} - \frac {p ^ {2} (\tilde {c} ^ {2 p - 1} c _ {\rho}) ^ {2} \epsilon^ {4 p}}{d ^ {2 p + 1} \beta^ {2 p} D ^ {4 p - 2}} T.
$$

However, note if we set $T = \frac{d^{2p + 1}\beta^{2p}D^{4p}}{p^2(\tilde{c}^{2p - 1}c_\rho)^2\epsilon^{4p}}$ , this implies $\mathbf{E}_{\mathcal{H}_T}[\| \mathbf{w}_{T + 1} - \mathbf{w}^*\|^2 ]\leq 0$ , or equivalently $\mathbf{w}_{T + 1} = \mathbf{w}^*$ , which concludes the claim.

To clarify further, note we show that for any run of Alg. 1 if indeed $f(\mathbf{w}_t) - f(\mathbf{w}^*) > \epsilon$ continues to hold for all $t = 1,2,\ldots T$ , then $\mathbf{w}_{T + 1} = \mathbf{w}^*$ at $T = \frac{d^{2p + 1}\beta^{2p}D^{4p}}{p^2(\tilde{c}^{2p - 1}c_\rho)^2\epsilon^{4p}}$ . If not, there must have been a time $t\in [T]$ such that $f(\mathbf{w}_t) - f(\mathbf{w}^*) < \epsilon$ . This concludes the proof with $T_{\epsilon} = T$ .

# C.3 Proof of Theorem 5

Theorem 5 (Improved Convergence Rate for Special Transfer Functions.). Algorithm 1 yields improved $\epsilon$ -convergence rate $(T_{\epsilon})$ for some special class of well-defined transfer functions, e.g.:

1. Linear transfer functions $\rho(x) = c_{\rho}x$ , $\forall x \in \mathbb{R}_{+}$ , then we have $T = \frac{2d^2\beta D^2}{c_\rho^2\epsilon^3}$ ;   
2. Sigmoid transfer functions $\rho(x) = \frac{1 - e^{-\omega x}}{1 + e^{-\omega x}}$ , $\forall x \in \mathbb{R}_+$ , $\omega > 0$ , then we have $T = O\left(\frac{d^2\beta D^2}{c_\rho^2\epsilon^3}\right)$ .

Proof. Case 1. Linear transfer functions: $\rho(x) = c_{\rho}x$ , $\forall x \in \mathbb{R}_{+}$ . So in this case $\rho$ is the $p$ -th order proxy of itself, i.e. $\tilde{\rho}_p = \rho$ with $p = 1$ and any $r \in \mathbb{R}$ .

Here $\rho'(x) = c_{\rho}$ for any $x \in \mathbb{R}$ . Then following the same steps as derived in the proof of Theorem 3, note we have,

$$
\mathbf {E} _ {t} [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) ] \geq \frac {2 \gamma}{d} \mathbf {E} _ {\mathbf {u} _ {t}} \left[ \rho^ {\prime} \big (f (\mathbf {w} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {w} _ {t} - \gamma \mathbf {u} _ {t}) \big) \right] \big (f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*}) - \beta \gamma^ {2} \big)
$$

$$
= \frac {2 \gamma}{d} c _ {\rho} \big (f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*}) - \beta \gamma^ {2} \big)
$$

Moreover, since we conditioned on $f(\mathbf{w}_{t}) - f(\mathbf{w}^{*}) > \epsilon$ , plugging this in above and combining with

$$
\begin{array}{l} \mathbf {E} _ {t} [ \| \mathbf {w} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - \frac {4 \eta \gamma c _ {\rho}}{d} \left(\left(f (\mathbf {w} _ {t}) - f (\mathbf {w} ^ {*})\right) - \beta \gamma^ {2}\right) + \eta^ {2} \\ \leq \left\| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \right\| ^ {2} - \frac {4 \eta \gamma c _ {\rho}}{d} \left(\epsilon - \beta \gamma^ {2}\right) + \eta^ {2} \\ \stackrel {(a)} {=} \left\| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \right\| ^ {2} - \frac {4 \eta c _ {\rho}}{d} \big (\frac {\epsilon^ {3 / 2}}{2 \sqrt {2 \beta}} \big) + \eta^ {2} \\ \stackrel {(b)} {=} \left\| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \right\| ^ {2} - \frac {c _ {\rho} ^ {2} \epsilon^ {3}}{2 \beta d ^ {2}}, \\ \end{array}
$$

where $(a)$ follows by setting $\gamma = \sqrt{\frac{\epsilon}{2\beta}}$ , and $(b)$ follows by setting $\eta = \frac{c_{\rho}\epsilon^{3 / 2}}{d\sqrt{2\beta}}$ .

Then same as the proof of Theorem 3, now iteratively taking expectations over $\mathcal{H}_T$ on both sides of Equation (1) and summing over $t = 1,\dots ,T$ , we get,

$$
\mathbf {E} _ {\mathcal {H} _ {T}} [ \| \mathbf {w} _ {T + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {w} _ {1} - \mathbf {w} ^ {*} \| ^ {2} - \frac {c _ {\rho} ^ {2} \epsilon^ {3}}{2 \beta d ^ {2}} T.
$$

However, note if we set $T = \frac{2d^2\beta D^2}{c_\rho^2\epsilon^3}$ , this implies $\mathbf{E}_{\mathcal{H}_T}[\| \mathbf{w}_{T + 1} - \mathbf{w}^*\| ^2 ]\leq 0$ , or equivalently $\mathbf{w}_{T + 1} = \mathbf{w}^*$ , which concludes the claim (similarly as line of argument we concluded the proof of Theorem 3).

Case 2. Sigmoid transfer functions: $\rho(x) = \frac{1 - e^{-\omega x}}{1 + e^{-\omega x}}, \forall x \in \mathbb{R}_+, \omega > 0$ . In this case the it can be shown that $\rho$ can be approximated by a linear function near the origin, or more specifically, depending on the constant $\omega$ , there exists $c_{\rho}^{\omega}$ and $r^{\omega}$ such that

$$
| \rho (x) | \geq c _ {\rho} ^ {\omega} | x | \forall x \in [ - r ^ {\omega}, r ^ {\omega} ].
$$

The claimed convergence bound now follows similar to the analysis shown for Case 1 above.

Case 3. Sign transfer function: $\rho(x) = \text{sign}(x)$ , $\forall x \in \mathbb{R}_{+}$ .

In this case also, $\rho$ is the p-th order proxy of itself, with p = 0, $c_{\rho} = 1$ and any $r \in R$ . This particular transfer function was considered in the similar optimization setup in Saha et al. (2021b). We show below how our proposed Algorithm 1 (Relative-Gradient-Descent) generalizes their $\beta$ -NGD algorithm and recovers their convergence rate of $O(\epsilon^{-1})$ .

We start by noting that, our algorithm generalizes the $\beta$ -NGD algorithm of Saha et al. (2021b). The convergence rate claim now follows by noting that in this case our descent direction $g_{t}$ at any point $w_{t}$ , becomes the normalized gradient estimate, of $w_{t}$ with high probability (over the random draws of $u_{t}$ ). Roughly speaking it can be show that

$$
\mathbf {E} _ {\mathbf {u} _ {t}} \left[ \mathbf {g} _ {t} \mid \mathbf {w} _ {t} \right] \approx \frac {\tilde {c}}{\sqrt {d}} \left(\nabla f (\mathbf {w} _ {t}) / \| \nabla f (\mathbf {w} _ {t}) \|\right),
$$

using Lemma 2 and 4 of Saha et al. (2021b), or more precisely,

$$
\mathbf {E} _ {\mathbf {u} _ {t}} [ \mathbf {g} _ {t} \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*}) \mid \mathbf {w} _ {t} ] \geq \frac {\tilde {c}}{\sqrt {d}} \left(\frac {\nabla f (\mathbf {w} _ {t}) \cdot (\mathbf {w} _ {t} - \mathbf {w} ^ {*})}{\| \nabla f (\mathbf {w} _ {t}) \|}\right) - 2 \lambda \| \mathbf {w} _ {t} - \mathbf {w} ^ {*} \|,
$$

where $\lambda = \frac{3\beta\delta}{\|\nabla f(\mathbf{w}_t)\|}\sqrt{d\log\frac{\|\nabla f(\mathbf{w}_t)\|}{\sqrt{d}\beta\delta}}$ . Combining the above bound in the proof of Theorem 3 (to lower bound the term $\mathbf{E}_t[\mathbf{g}_t\cdot (\mathbf{w}_t - \mathbf{w}^*)]$ ), the result follows. In fact in this case the proof of Theorem 3 exactly follows the same line of argument as that of the proof of Theorem 5 of Saha et al. (2021b). This shows the generalization ability of our proof analysis for different special class of transfer functions.

# D Appendix for Section 4.2

# D.1 Proof of Lemma 7

Lemma 7 (Epochwise Convergence Guarantee of Epoch-RGD). Consider the problem setup of Theorem 6. Then the point $w_{k+1}$ returned by k-th epoch run of Epoch-RGD (Algorithm 2) starting from the initial point $w_{k}$ , satisfies:

$$
\mathbf {E} [ \| \mathbf {w} _ {k + 1} - \mathbf {w} ^ {*} \| ^ {2} | \mathbf {w} _ {k} ] \leq \frac {3}{4} \| \mathbf {w} _ {k} - \mathbf {w} ^ {*} \| ^ {2},
$$

$\forall k \in [k_{\epsilon}]$ . Where the expectation is taken over the randomness of the algorithm and the dueling feedback received inside the run of Relative-Gradient-Descent $(\mathbf{w}_k, \eta_k, \gamma_k, t_k)$ .

Complete Proof of Lemma 7. The proof relies on analyzing the epochwise performance guarantee of any representative run Relative-Gradient-Descent $\left(\mathbf{w}_{k},\eta_{k},\gamma_{k},t_{k}\right)$ (see Line #5 of Algorithm 2).

For simplicity of notations, for any epoch k, inside the call of Relative-Gradient-Descent $\left(\mathbf{w}_{k},\eta_{k},\gamma_{k},t_{k}\right)$ , let us assume $\mathbf{x}_{1}(=\mathbf{w}_{k})$ denotes the initial point in the run of Relative-Gradient-Descent (Algorithm 1) and let is denote by $D_{0}=\|\mathbf{x}_{1}-\mathbf{w}^{*}\|$ . The goal is to analyze the guarantees on the output point $\mathbf{x}_{T+1}$ of the run of $p^{th}$ -degree-scaled Gradient after $T=t_{k}$ time steps; thus $\mathbf{x}_{T+1}=\mathbf{w}_{k+1}$ . We use the same notations as used in the proof of Theorem 3.

Recall from Equation (12), at any time step t inside the run of $p^{th}$ -degree-scaled Gradient we have:

$$
\mathbf {E} _ {t} [ \| \mathbf {x} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {x} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - 2 \eta \mathbf {E} _ {t} [ \mathbf {g} _ {t} \cdot (\mathbf {x} _ {t} - \mathbf {w} ^ {*}) ] + \eta^ {2}, \tag {19}
$$

and on the other hand, from Equation (13) we have:

$$
\begin{array}{l} \mathbf {E} _ {t} \left[ \mathbf {g} _ {t} \cdot \left(\mathbf {x} _ {t + 1} - \mathbf {w} ^ {*}\right) \right] = \mathbf {E} _ {\mathbf {u} _ {t}} \left[ \rho \left(f \left(\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}\right) - f \left(\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}\right)\right) \mathbf {u} _ {t} \mid \mathcal {H} _ {t} \right] \cdot \left(\mathbf {x} _ {t} - \mathbf {w} ^ {*}\right) \\ = \frac {\gamma}{d} \mathbf {E _ {u}} _ {t} \left[ \rho^ {\prime} \big (| f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) | \big) \big (\nabla f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) + \nabla f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) \big) | \mathcal {H} _ {t} \right] \cdot (\mathbf {x} _ {t} - \mathbf {w} ^ {*}). (2 0) \\ \end{array}
$$

Now since f is convex and $\beta$ -smooth, applying Lemma 19, we know that for any $w \in D$ ,

$$
\left(\nabla f (\mathbf {w}) - \nabla f (\mathbf {w} ^ {*})\right) ^ {\top} (\mathbf {w} - \mathbf {w} ^ {*}) \geq \frac {\alpha \beta}{\alpha + \beta} \| \mathbf {w} - \mathbf {w} ^ {*} \| ^ {2} + \frac {1}{\alpha + \beta} \| \nabla f (\mathbf {w}) - \nabla f (\mathbf {w} ^ {*}) \| ^ {2}.
$$

Moreover since $w^{*}$ is the minimizer of f in D, we have $\nabla f(\mathbf{w}^{*}) \cdot (\mathbf{x}_{t} - \mathbf{w}^{*}) \geq 0$ (from first order optimality conditions, see Lemma 14). Hence from above we further get:

$$
\begin{array}{l} \nabla f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) \cdot (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*}) \geq \left(\nabla f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - \nabla f (\mathbf {w} ^ {*})\right) \cdot (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*}) \\ \geq \frac {\alpha \beta}{\alpha + \beta} \| \mathbf {x} _ {t} + \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \frac {1}{\alpha + \beta} \| \nabla f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - \nabla f (\mathbf {w} ^ {*}) \| ^ {2}. \\ \end{array}
$$

Likewise,

$$
\nabla f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) \cdot (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*}) \geq \frac {\alpha \beta}{\alpha + \beta} \| \mathbf {x} _ {t} - \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \frac {1}{\alpha + \beta} \| \nabla f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) - \nabla f (\mathbf {w} ^ {*}) \| ^ {2}.
$$

Combining the two, we get:

$$
\nabla f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) \cdot (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*}) + \nabla f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) \cdot (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*})
$$

$$
\geq \frac {\alpha \beta}{\alpha + \beta} (\| \mathbf {x} _ {t} + \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \| \mathbf {x} _ {t} - \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*} \| ^ {2})
$$

$$
+ \frac {1}{\alpha + \beta} (\| \nabla f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - \nabla f (\mathbf {w} ^ {*}) \| ^ {2} + \| \nabla f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) - \nabla f (\mathbf {w} ^ {*}) \| ^ {2})
$$

$$
\stackrel {(a)} {\geq} \frac {\alpha \beta}{\alpha + \beta} (\| \mathbf {x} _ {t} + \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \| \mathbf {x} _ {t} - \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*} \| ^ {2}) + \frac {\alpha}{\alpha + \beta} (\| \mathbf {x} _ {t} + \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \| \mathbf {x} _ {t} - \gamma \mathbf {u} _ {t} - \mathbf {w} ^ {*} \| ^ {2})
$$

$$
= \frac {2 (\alpha \beta + \alpha)}{\alpha + \beta} (\| \mathbf {x} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \gamma^ {2} \| \mathbf {u} _ {t} \| ^ {2}) = \frac {2 (\alpha \beta + \alpha)}{\alpha + \beta} (\| \mathbf {x} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \gamma^ {2}),
$$

where the inequality $(a)$ follows due to Lemma 18. Thus we can write:

$$
\begin{array}{l} \nabla f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) \cdot (\mathbf {x} _ {t} - \mathbf {w} ^ {*}) + \nabla f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) \cdot (\mathbf {x} _ {t} - \mathbf {w} ^ {*}) \\ \geq \frac {2 (\alpha \beta + \alpha)}{\alpha + \beta} (\| \mathbf {x} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \gamma^ {2}) - \gamma (\nabla f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - \nabla f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t})) \cdot \mathbf {u} _ {t} \\ \stackrel {(b)} {\geq} \frac {2 (\alpha \beta + \alpha)}{\alpha + \beta} (\| \mathbf {x} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \gamma^ {2}) - \gamma \| \nabla f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - \nabla f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) \| \\ \geq \frac {2 (\alpha \beta + \alpha)}{\alpha + \beta} (\| \mathbf {x} _ {t} - \mathbf {w} ^ {*} \| ^ {2} + \gamma^ {2}) - 2 \beta \gamma^ {2} \\ = \frac {2 \alpha (\beta + 1)}{\beta (\tilde {\kappa} + 1)} \| \mathbf {x} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - \frac {2 (\beta - \tilde {\kappa})}{\tilde {\kappa} + 1} \gamma^ {2}, \tag {21} \\ \end{array}
$$

where $(b)$ follows from Cauchy-Schwarz, the last inequality holds due to Lemma 16, and $\tilde{\kappa} := \frac{\alpha}{\beta}$ .

Recall we assumed $\|x_{1}-w^{*}\|=D_{0}$ . Let $T(=t_{k})$ be the sample complexity of the run Relative-Gradient-Descent $\left(\mathbf{w}_{k},\eta_{k},\gamma_{k},t_{k}\right)$ , i.e. the Relative-Gradient-Descent runs for T time-steps starting from the initial point $x_{1}=w_{k}$ .

Case analysis 1: (Assume $\|x_{t}-w^{*}\|\geq D_{0}/2$ for all $t=[T]$ ). In this case, by assumption, $\|x_{t}-w^{*}\|\geq D_{0}/2$ . So from above we further get:

$$
\mathbf {E _ {u}} _ {t} [ | f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) | ] \geq \frac {2 \tilde {c} \alpha \gamma D _ {0}}{2 \sqrt {d}} - \beta \gamma^ {2}.
$$

And now note that setting $\gamma = \frac{\tilde{c}\alpha D_0}{2\beta\sqrt{d}}$ , the right hand side is positive. Further using Assumption 1-(3) and by the definition of $\tilde{\rho}_p$ , we get:

$$
\begin{array}{l} \mathbf {E _ {u}} _ {t} [ \rho^ {\prime} (| f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) |) ] \geq \mathbf {E _ {u}} _ {t} [ \tilde {\rho} _ {p} ^ {\prime} (| f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) |) ] \\ \geq \tilde {\rho} _ {p} ^ {\prime} (\mathbf {E} _ {\mathbf {u} _ {t}} [ | f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) | ]) \\ \geq \tilde {\rho} _ {p} ^ {\prime} \left(\frac {2 \tilde {c} \alpha \gamma D _ {0}}{2 \sqrt {d}} - \beta \gamma^ {2}\right) = \tilde {\rho} _ {p} ^ {\prime} \left(\frac {\tilde {c} ^ {2} \alpha^ {2} D _ {0} ^ {2}}{4 \beta d}\right). \tag {22} \\ \end{array}
$$

where the second and the third inequalities are respectively due to Jensen's inequality, as by definition $\tilde{\rho}_p'$ is convex (for any $p \geq 1$ ) in the positive orthant and also monotonically increasing. Further by denoting $A = \frac{\alpha(\beta + 1)}{\beta(\tilde{\kappa} + 1)}$ and $A' = \frac{(\beta - \tilde{\kappa})}{\tilde{\kappa} + 1}$ , note that for this choice of $\gamma$ we have

$$
\begin{array}{l} A \| \mathbf {x} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - A ^ {\prime} \gamma^ {2} > \frac {\alpha (\beta + 1)}{\beta (\tilde {\kappa} + 1)} \frac {D _ {0} ^ {2}}{4} - \frac {(\beta - \tilde {\kappa})}{\tilde {\kappa} + 1} \left(\frac {\tilde {c} \alpha D _ {0}}{2 \beta \sqrt {d}}\right) ^ {2} \\ \stackrel {(c)} {\geq} \frac {\alpha (\beta + 1)}{\beta (\tilde {\kappa} + 1)} \frac {D _ {0} ^ {2}}{4} - \frac {(\beta - \tilde {\kappa})}{\tilde {\kappa} + 1} \left(\frac {\alpha^ {2} D _ {0} ^ {2}}{4 \beta^ {2} d}\right) \\ \geq \frac {\alpha (\beta + 1)}{\beta (\tilde {\kappa} + 1)} \frac {D _ {0} ^ {2}}{4} - \frac {\alpha^ {2}}{\beta d (\tilde {\kappa} + 1)} \frac {D _ {0} ^ {2}}{4} \\ \geq \frac {\alpha (\beta + 1)}{\beta (\tilde {\kappa} + 1)} \frac {D _ {0} ^ {2}}{4} - \frac {\alpha^ {2}}{\beta d (\tilde {\kappa} + 1)} \frac {D _ {0} ^ {2}}{4} \\ = \frac {\alpha (\beta + 1 - \alpha / d)}{\beta (\tilde {\kappa} + 1)} \frac {D _ {0} ^ {2}}{4} > \frac {\alpha}{\alpha + \beta} \frac {D _ {0} ^ {2}}{4}, \\ \end{array}
$$

where $(c)$ follows since $\tilde{c} \leq 1$ (as follows from Lemma 8), and the last inequality is due to the fact that by definition $\beta \geq \alpha$ and $d \geq 1$ . Then combining above with Equations (20) and (22) we have:

$$
\begin{array}{l} \mathbf {E} _ {t} [ \mathbf {g} _ {t} \cdot (\mathbf {x} _ {t} - \mathbf {w} ^ {*}) ] \geq \frac {2 \gamma}{d} \tilde {\rho} _ {p} ^ {\prime} \big (\mathbf {E} _ {\mathbf {u} _ {t}} \big [ | f (\mathbf {x} _ {t} + \gamma \mathbf {u} _ {t}) - f (\mathbf {x} _ {t} - \gamma \mathbf {u} _ {t}) | \big ] \big) \left(\frac {\alpha}{\alpha + \beta} \frac {D _ {0} ^ {2}}{4}\right) \\ \geq \frac {2 \alpha D _ {0} ^ {2}}{d (\alpha + \beta) 4} \left(\frac {\tilde {c} \alpha D _ {0}}{2 \beta \sqrt {d}}\right) \rho^ {\prime} \left(\frac {\tilde {c} ^ {2} \alpha^ {2} D _ {0} ^ {2}}{4 \beta d}\right) \text {(from (22), where recall we needed to set \gamma = \frac {\tilde {c} \alpha D _ {0}}{2\beta\sqrt {d}})} \\ = \frac {2 \alpha D _ {0} ^ {2}}{d (\alpha + \beta) 4} \left(\frac {\tilde {c} \alpha D _ {0}}{2 \beta \sqrt {d}}\right) c _ {\rho} p \left(\frac {\tilde {c} ^ {2} \alpha^ {2} D _ {0} ^ {2}}{4 \beta d}\right) ^ {p - 1} (\text {as,} \rho^ {\prime} (x) = c _ {\rho} p x ^ {p - 1} \text {for any} x \in \mathbb {R} _ {+}) \\ = B D _ {0} ^ {2 p + 1}, \left(\text {where we denote by,} B := \frac {2 c _ {\rho} p}{(\alpha + \beta)} \left(\left(\alpha^ {2} / 4 \beta\right) ^ {p} \frac {\tilde {c} ^ {2 p - 1}}{d ^ {\frac {2 p + 1}{2}}}\right)\right). \tag {23} \\ \end{array}
$$

Plugging the above expression in Equation (19):

$$
\begin{array}{l} \mathbf {E} _ {t} [ \| \mathbf {x} _ {t + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {x} _ {t} - \mathbf {w} ^ {*} \| ^ {2} - 2 \eta \mathbf {E} _ {t} [ [ \mathbf {g} _ {t} \cdot (\mathbf {x} _ {t} - \mathbf {w} ^ {*}) ] ] + \eta^ {2} \\ \leq D _ {0} ^ {2} - \eta B (D _ {0} ^ {2 p + 1}) + \eta^ {2} \left(\text {where recall} B := \frac {2 c _ {\rho} p}{(\alpha + \beta)} \Big (\big (\alpha^ {2} / 4 \beta \big) ^ {p} \frac {\tilde {c} ^ {2 p - 1}}{d ^ {\frac {2 p + 1}{2}}} \Big)\right) \\ = D _ {0} ^ {2} - B ^ {2} D _ {0} ^ {2 (2 p + 1)}, \left(\text {by setting} \eta = B D _ {0} ^ {2 p + 1} = \frac {2 c _ {\rho} p}{(\alpha + \beta)} \left(\left(\alpha^ {2} / 4 \beta\right) ^ {p} \frac {\tilde {c} ^ {2 p - 1}}{d ^ {\frac {2 p + 1}{2}}}\right) D _ {0} ^ {2 p + 1}\right). \tag {24} \\ \end{array}
$$

Now let us fix some time-stamp $T$ and let us assume $\| \mathbf{x}_t - \mathbf{w}^* \| \geq D_0 / 2$ for all $t = [T]$ . Then taking expectation over $\mathcal{H}_{T+1}$ on both sides iteratively and summing over $t = 1, \ldots, T$ , note that we get:

$$
\mathbf {E} _ {\mathcal {H} _ {T + 1}} [ \left\| \mathbf {x} _ {T + 1} - \mathbf {w} ^ {*} \right\| ^ {2} ] \leq D _ {0} ^ {2} - B ^ {2} D _ {0} ^ {2 (2 p + 1)} T.
$$

Then if we set $T = \frac{1}{2B^{2}D_{0}^{22p}}$ , at time $T + 1$ we have:

$$
\mathbf {E} _ {\mathcal {H} _ {T + 1}} [ \| \mathbf {x} _ {T + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq D _ {0} ^ {2} / 2 <   3 D _ {0} ^ {2} / 4.
$$

Case analysis 2: (∃ at least an $\tau\in[T]$ such that $\|x_{\tau}-w^{*}\|\leq D_{0}/2$ ). In this case, from Equation (19), after $(T-\tau)$ steps we can have the expected value of $\|x_{T+1}-w^{*}\|^{2}$ can be at most:

$$
\mathbf {E} _ {\mathcal {H} _ {T + 1}} [ \| \mathbf {x} _ {T + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \| \mathbf {x} _ {\tau} - \mathbf {w} ^ {*} \| ^ {2} + (T - \tau) \eta^ {2} \leq D _ {0} ^ {2} / 4 + T \eta^ {2} = 3 D _ {0} ^ {2} / 4,
$$

since recall that we set $\eta = BD_0^{2p + 1} = \frac{2c_\rho p}{(\alpha + \beta)}\left((\alpha^2 / 4\beta)^p\frac{\tilde{c}^{2p - 1}}{d^{\frac{2p + 1}{2}}}\right)D_0^{2p + 1}$ .

This implies that for the above choice of $\gamma$ and $\eta$ we can get constant fraction reduction in the “sub-optimality gap” ( $\|w_{k}-w^{*}\|$ ) after at most $O(\frac{1}{2B^{2}D_{0}^{2p}})$ time steps. ☐

# D.2 Proof of Theorem 6

Theorem 6 (Convergence Analysis of Epoch-RGD for Smooth and Strongly convex Functions). Consider a dueling feedback optimization problem parameterized by any general admissible transfer function $\rho$ with $p$ -th order proxy $\tilde{\rho}_p$ and a $\beta$ smooth $\alpha$ -strongly convex function $f: \mathcal{D} \mapsto \mathbb{R}$ . Then given any $\epsilon > 0$ , the final point $\mathbf{w}_{k_{\epsilon} + 1}$ returned by Algorithm 2 satisfies $\mathbf{E}[f(\mathbf{w}_{t + 1})] - f(\mathbf{w}^*) \leq \epsilon$ , with a sample complexity of at most $O\left(\frac{1}{B^2\epsilon^{2p}}\right)$ pairwise comparisons. (Here the constant $B$ is as defined in Algorithm 2, $\tilde{c} = \frac{1}{20}$ is a universal constant.

Complete Proof of Theorem 6. The proof of Theorem 6 is based on the key claim of Lemma 7 which shows that after every epoch of length $t_{k}$ , the distance of the resulting point $w_{k+1}$ from the optimal $w^{*}$ must decrease by at least a constant fraction. Recall from the statement of Lemma 7, we have:

Lemma 7 (Epochwise Convergence Guarantee of Epoch-RGD). Consider the problem setup of Theorem 6. Then the point $w_{k+1}$ returned by k-th epoch run of Epoch-RGD (Algorithm 2) starting from the initial point $w_{k}$ , satisfies:

$$
\mathbf {E} [ \| \mathbf {w} _ {k + 1} - \mathbf {w} ^ {*} \| ^ {2} | \mathbf {w} _ {k} ] \leq \frac {3}{4} \| \mathbf {w} _ {k} - \mathbf {w} ^ {*} \| ^ {2},
$$

$\forall k \in [k_{\epsilon}]$ . Where the expectation is taken over the randomness of the algorithm and the dueling feedback received inside the run of Relative-Gradient-Descent $(\mathbf{w}_k, \eta_k, \gamma_k, t_k)$ .

The claim of Theorem 6 now follows form the following epoch-wise recursion argument:

Let $\mathcal{H}_{[k]} := \{(\mathbf{w}_{k'})_{k' \in [k_\epsilon]}, (\mathbf{u}_{t'}, o_{t'})_{t' \in [\sum_{k'=1}^k t_{k'}]}\} \cup \{\mathbf{w}_{k+1}\}$ denotes the complete history till the end of epoch $k$ starting from the first epoch $\forall k \in [k_\epsilon]$ .

Further, let us denote by $\mathcal{H}_k := \{\mathbf{w}_k, (\mathbf{u}_{t'}, o_{t'})_{t' \in [\sum_{k'=1}^{k-1} t_{k'} + 1, \sum_{k'=1}^{k} t_{k'}]}\} \cup \{\mathbf{w}_{k+1}\}$ be the history only within epoch $k$ .

Proof of Correctness. From Lemma 7, note we have already established that $E_{\mathcal{H}_k}[\| \mathbf{w}_{k + 1} - \mathbf{w}^*\|^2|\mathbf{w}_k] \leq \frac{3}{4}\| \mathbf{w}_k - \mathbf{w}^*\|^2$ .

Applying the argument iteratively over $E$ epochs, and the law of iterated expectations, we have:

$$
\mathbf {E} _ {\mathcal {H} _ {[ E ]}} \left[ \left\| \mathbf {w} _ {E + 1} - \mathbf {w} ^ {*} \right\| ^ {2} \right] \leq (3 / 4) ^ {E} \left\| \mathbf {w} _ {1} - \mathbf {w} ^ {*} \right\| ^ {2}. \tag {25}
$$

Thus choosing $E = \lceil \log_{4/3}(\frac{\beta D^2}{2\epsilon}) \rceil$ , where $\| \mathbf{w}_1 - \mathbf{w}^* \| \leq D$ , we have

$$
E \geq \log_ {4 / 3} (\frac {\beta \| \mathbf {w} _ {1} - \mathbf {w} ^ {*} \| ^ {2}}{2 \epsilon}) \Rightarrow (3 / 4) ^ {E} \| \mathbf {w} _ {1} - \mathbf {w} ^ {*} \| ^ {2} \leq \frac {2 \epsilon}{\beta}.
$$

Thus from (25), we get:

$$
\mathbf {E} _ {\mathcal {H} _ {[ E ]}} [ \| \mathbf {w} _ {E + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \frac {2 \epsilon}{\beta},
$$

and further applying $\beta$ -smoothness of $f$ , we get:

$$
\mathbf {E} _ {\mathcal {H} _ {[ E ]}} [ f (\mathbf {w} _ {E + 1}) - f (\mathbf {w} ^ {*}) ] \leq \mathbf {E} _ {\mathcal {H} _ {[ E ]}} [ \frac {\beta}{2} \| \mathbf {w} _ {E + 1} - \mathbf {w} ^ {*} \| ^ {2} ] \leq \epsilon ,
$$

which proves the correctness of Algorithm 2 for the choice of total number of epochs $k_{\epsilon} = E$ .

Proof of Sample Complexity. In order to verify that Algorithm 2 indeed converges to an $\epsilon$ -optimal point in $O(\epsilon^{-2p})$ sample complexity, note we simply need to count the total sample complexity incurred in the $k_{\epsilon}$ epochwise runs of Relative-Gradient-Descent (see Line #5 of Algorithm 2). However, by design of Epoch-RGD (Algorithm 2), since Relative-Gradient-Descent $\left(\mathbf{w}_k,\eta_k,\gamma_k,t_k\right)$ is run for only $t_k = \frac{1}{2B^2(D_k^2)^{2p}}$ iterations, the total sample complexity of Epoch-RGD becomes:

$$
\begin{array}{l} \sum_ {k = 1} ^ {k _ {\epsilon}} t _ {k} = \frac {1}{2 B ^ {2}} \sum_ {k = 1} ^ {k _ {\epsilon}} \frac {1}{(D _ {k} ^ {2}) ^ {2 p}} = \frac {1}{2 B ^ {2}} \bigg (\frac {1}{(D _ {1} ^ {2}) ^ {2 p}} + \frac {1}{(D _ {2} ^ {2}) ^ {2 p}} + \dots + \frac {1}{(D _ {k _ {\epsilon}} ^ {2}) ^ {2 p}} \bigg) \\ = \frac {1}{2 B ^ {2}} \bigg (\frac {1}{(D ^ {2}) ^ {2 p}} + \frac {1}{(3 / 4 D ^ {2}) ^ {2 p}} + \frac {1}{((3 / 4) ^ {2} D ^ {2}) ^ {2 p}} + \dots + \frac {1}{((3 / 4) ^ {k _ {\epsilon} - 1} D _ {k _ {\epsilon}} ^ {2}) ^ {2 p}} \bigg) \\ = \frac {1}{2 B ^ {2} (D ^ {2}) ^ {2 p}} \left(1 + \frac {1}{(3 / 4) ^ {2 p}} + \frac {1}{((3 / 4) ^ {2 2 p}} + \dots + \frac {1}{((3 / 4) ^ {k _ {\epsilon} - 1}) ^ {2 p}}\right) \\ = \frac {1}{2 B ^ {2} (D ^ {2}) ^ {2 p}} \left(1 + \frac {1}{(3 / 4) ^ {2 p}} + \frac {1}{((3 / 4) ^ {2 p}) ^ {2}} + \dots + \frac {1}{((3 / 4) ^ {2 p}) ^ {k _ {\epsilon} - 1}}\right) \\ = \frac {1}{2 B ^ {2} (D ^ {2}) ^ {2 p}} \frac {(4 / 3 ^ {2 p}) ^ {k _ {\epsilon}} - 1}{4 / 3 ^ {2 p} - 1} \leq \frac {1}{4 B ^ {2} (D ^ {2}) ^ {2 p}} \Big ((\beta D ^ {2} / 2 \epsilon) ^ {2 p} - 1 \Big) = O \Big (\frac {1}{B ^ {2} \epsilon^ {2 p}} \Big) \\ \end{array}
$$

where the last inequality is since $k_{\epsilon} = \lceil \log_{4/3}(\frac{\beta D^{2}}{2\epsilon}) \rceil$ by definition. Thus follows the claimed sample complexity of Epoch-RGD in Theorem 6 and this concludes the proof. □

# E Standard Results from Convex Optimization

The results covered in this section can be found in Hazan (2019); Luenberger et al. (1984); Boyd et al. (2004); Fletcher (2013); Nesterov (2003); Nocedal and Wright (1999).

Definition 12 (Lipschitz Function). Assume $\mathcal{D} \subseteq \mathbb{R}^d$ be bounded decision space. Then any function $f: \mathcal{D} \mapsto \mathbb{R}$ f is called $L$ -Lipschitz over $\mathcal{D}$ with respect to a norm $\|\cdot\|$ if for all $\mathbf{x}, \mathbf{y} \in \mathcal{D}$ , we have:

$$
| f (\mathbf {x}) - f (\mathbf {y}) | \leq L \| \mathbf {x} - \mathbf {y} \|.
$$

# E.1 Useful properties for Convex Functions

Definition 13 (Convex Function). Assume $\mathcal{D} \subseteq \mathbb{R}^d$ be any convex and bounded decision space. Then any differential function $f: \mathcal{D} \mapsto \mathbb{R}$ is called convex if for all $\mathbf{x}, \mathbf{y} \in \mathcal{D}$ ,

$$
f (\mathbf {x}) - f (\mathbf {y}) \geq \nabla f (\mathbf {y}) ^ {\top} (\mathbf {x} - \mathbf {y}).
$$

Lemma 14 (First Order Optimality Condition (Luenberger et al., 1984; Boyd et al., 2004)). Assume $f: \mathcal{D} \mapsto \mathbb{R}$ is a convex function and $\mathbf{x}^*$ be the minimizer of $f$ . Then for any $\mathbf{x} \in \mathcal{D}$ ,

$$
\nabla f (\mathbf {x} ^ {*}) ^ {\top} (\mathbf {y} - \mathbf {x} ^ {*}) \geq 0
$$

# E.2 Useful properties for $\beta$ -Smooth Convex Functions

Definition 15 ( $\beta$ -Smooth Convex Function). Assume $D \subseteq R^{d}$ be any convex and bounded decision space. Then any differential and convex function $f : D \mapsto R$ is also called $\beta$ -smooth (any $\beta > 0$ ) if for all $x, y \in D$ ,

$$
f (\mathbf {x}) - f (\mathbf {y}) \leq \nabla f (\mathbf {y}) ^ {\top} (\mathbf {x} - \mathbf {y}) + \frac {\beta}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Lemma 16 (Properties of $\beta$ -smooth functions (Hazan, 2019; Bubeck, 2014)). Suppose $f: \mathcal{D} \mapsto \mathbb{R}$ is a $\beta$ -smooth convex function. Then for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ ,

$$
f (\mathbf {x}) - f (\mathbf {y}) \leq \nabla f (\mathbf {x}) ^ {\top} (\mathbf {x} - \mathbf {y}) - \frac {1}{2 \beta} \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| ^ {2}
$$

$$
\| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| \leq \beta \| \mathbf {x} - \mathbf {y} \|.
$$

Further if $\mathbf{x}^*$ is the minimizer of $f$ and $\mathbf{x}^* \in Int(\mathcal{D})$ (i.e. $\mathbf{x}^*$ belong to the interior of $f$ 's domain $\mathcal{D}$ ), then

$$
\left\| \nabla f (\mathbf {x}) \right\| ^ {2} \leq 2 \beta (f (\mathbf {x}) - f (\mathbf {x} ^ {*})
$$

$$
f (\mathbf {x}) - f (\mathbf {x} ^ {*}) \leq \frac {\beta}{2} \| \mathbf {x} - \mathbf {x} ^ {*} \| ^ {2}.
$$

# E.3 Useful properties for $\alpha$ -Strongly Convex Functions

Definition 17 ( $\alpha$ -Strongly Convex Function). Assume $D \subseteq R^{d}$ be any convex and bounded decision space. Then any differential and convex function $f : D \mapsto R$ is also called $\alpha$ -strongly convex (any $\alpha > 0$ ) if for all $x, y \in D$ ,

$$
f (\mathbf {x}) - f (\mathbf {y}) \geq \nabla f (\mathbf {y}) ^ {\top} (\mathbf {x} - \mathbf {y}) + \frac {\alpha}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Lemma 18. If $f: \mathcal{D} \mapsto \mathbb{R}$ is an $\alpha$ -strongly convex function, with $\mathbf{x}^*$ being the minimizer of $f$ . Then for any $\mathbf{x}, \mathbf{y}, \mathbf{z} \in \mathcal{D}$ ,

$$
\| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| \geq \alpha \| \mathbf {x} - \mathbf {y} \|
$$

$$
\| \nabla f (\mathbf {z}) \| \geq \alpha \| \mathbf {z} - \mathbf {x} ^ {*} \|
$$

$$
\frac {\alpha}{2} \| \mathbf {x} ^ {*} - \mathbf {x} \| ^ {2} \leq f (\mathbf {x}) - f (\mathbf {x} ^ {*}).
$$

Proof. This simply follows by the properties of $\alpha$ -strongly convex function. Note by definition of $\alpha$ -strong convexity, for any $x, y \in R$ ,

$$
f (\mathbf {x}) - f (\mathbf {y}) \geq \nabla f (\mathbf {y}) ^ {\top} (\mathbf {x} - \mathbf {y}) + \frac {\alpha}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Similarly,

$$
f (\mathbf {y}) - f (\mathbf {x}) \geq \nabla f (\mathbf {x}) ^ {\top} (\mathbf {y} - \mathbf {x}) + \frac {\alpha}{2} \| \mathbf {x} - \mathbf {y} \| ^ {2}.
$$

Adding we get:

$$
\left(\nabla f (\mathbf {y}) - \nabla f (\mathbf {x})\right) ^ {\top} (\mathbf {y} - \mathbf {x}) \geq \alpha \| \mathbf {x} - \mathbf {y} \| ^ {2}
$$

Now applying Cauchy-Schwarz inequality to the left hand side of the above inequality yields the first result.

To get the second result, let us use y = z and $x = x^{*}$ in the above inequality, which along with the first order optimality yields (Lemma 14):

$$
\nabla f (\mathbf {z}) ^ {\top} \left(\mathbf {z} - \mathbf {x} ^ {*}\right) \geq \alpha \| \mathbf {z} - \mathbf {x} ^ {*} \| ^ {2}
$$

The result now follows by again applying Cauchy-Schwarz inequality to the left hand side of the above inequality. Finally the last part of the proof simply follows setting $y = x^{*}$ and from the first order optimality condition (see Lemma 14).

# E.4 Useful properties for $\alpha$ -Strongly Convex and $\beta$ -Smooth Convex Functions ( $\beta \geq \alpha$ )

Lemma 19 (Properties of $\beta$ -smooth and $\alpha$ -strongly convex functions (Bubeck, 2014)). Suppose $f: \mathcal{D} \mapsto \mathbb{R}$ is a $\beta$ -smooth and $\alpha$ -strongly convex function. Then for all $\mathbf{x}, \mathbf{y} \in \mathcal{D}$

$$
\left(\nabla f (\mathbf {x}) - \nabla f (\mathbf {y})\right) ^ {\top} (\mathbf {x} - \mathbf {y}) \geq \frac {\alpha \beta}{\alpha + \beta} \| \mathbf {x} - \mathbf {y} \| ^ {2} + \frac {1}{\alpha + \beta} \| \nabla f (\mathbf {x}) - \nabla f (\mathbf {y}) \| ^ {2}
$$