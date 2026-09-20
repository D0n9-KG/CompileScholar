# Contextual Multinomial Logit Bandits with General Value Functions

Mengxiao Zhang

Haipeng Luo

University of Southern California

MENGXIAO.ZHANG@USC.EDU

HAIPENGL@USC.EDU

# Abstract

Contextual multinomial logit (MNL) bandits capture many real-world assortment recommendation problems such as online retailing/advertising. However, prior work has only considered (generalized) linear value functions, which greatly limits its applicability. Motivated by this fact, in this work, we consider contextual MNL bandits with a general value function class that contains the ground truth, borrowing ideas from a recent trend of studies on contextual bandits. Specifically, we consider both the stochastic and the adversarial settings, and propose a suite of algorithms, each with different computation-regret trade-off. When applied to the linear case, our results not only are the first ones with no dependence on a certain problem-dependent constant that can be exponentially large, but also enjoy other advantages such as computational efficiency, dimension-free regret bounds, or the ability to handle completely adversarial contexts and rewards.

# 1. Introduction

As assortment recommendation becomes ubiquitous in real-world applications such as online retailing and advertising, the multinomial (MNL) bandit model has attracted great interest in the past decade since it was proposed by Rusmevichientong et al. (2010). It involves a learner and a customer interacting for $T$ rounds. At each round, knowing the reward/profit for each of the $N$ available items, the learner selects a subset/assortment of size at most $K$ and recommend it to the customer, who then purchases one of these $K$ items or none of them according to a multinomial logit model specified by the customer's valuation over the items. The goal of the learner is to learn these unknown valuations over time and select the assortments with high reward.

To better capture practical applications where there is rich contextual information about the items and customers, a sequence of recent works study a contextual MNL bandit model where the customer's valuation is determined by the context via an unknown (generalized) linear function (Cheung and Simchi-Levi, 2017; Ou et al., 2018; Chen et al., 2020; Oh and Iyengar, 2019, 2021; Perivier and Goyal, 2022; Agrawal et al., 2023). However, there are no studies on general value functions, despite many recent breakthroughs for classic contextual multi-armed bandits using a general value function class with much stronger representation power that enables fruitful results in both theory and practice (Agarwal et al., 2014; Foster and Rakhlin, 2020; Xu and Zeevi, 2020; Foster and Krishnamurthy, 2021; Simchi-Levi and Xu, 2021).

Contributions. Motivated by this gap, we propose a contextual MNL bandit model with a general value function class that contains the ground truth (a standard realizability assumption), and develop a suite of algorithms for different settings and with different computation-regret trade-off.

More specifically, in Section 3, we first consider a stochastic setting where the context-reward pairs are i.i.d. samples of an unknown distribution. Following the work by Simchi-Levi and Xu (2021) for contextual bandits, we reduce the problem to an easier offline log loss regression problem

and propose two strategies using an offline regression oracle: one with simple and efficient uniform exploration, and another with more adaptive exploration (and hence improved regret) induced by a novel log-barrier regularized strategy. Our results rely on several new technical findings, including a fast rate regression result (Lemma 1), a “reverse Lipschitzness” for the MNL model (Lemma 3), and a certain “low-regret-high-dispersion” property of the log-barrier regularized strategy (Lemma 6).

Next, in Section 4, we switch to the more challenging adversarial setting where the context-reward pairs can be arbitrarily chosen. We start by following the idea of Foster and Rakhlin (2020); Foster and Krishnamurthy (2021) for contextual bandits and reducing our problem to online log loss regression, and show that it suffices find a strategy with a small Decision-Estimation Coefficient (DEC) (Foster and Rakhlin, 2020; Foster et al., 2021). We then show that, somewhat surprisingly, the same log-barrier regularized strategy we developed for the stochastic setting leads to a small DEC, despite the fact that it is not the exact DEC minimizer (unlike its counterpart for contextual bandits (Foster et al., 2020)). We prove this by using the same aforementioned low-regret-high-dispersion property, which to our knowledge is a new way to bound DEC and reveals why log-barrier regularized strategies work in different settings and for different problems. Finally, we also extend the idea of Feel-Good Thompson Sampling (Zhang, 2022) and propose a variant for our problem that leads to the best regret bounds in some cases, despite its lack of computational efficiency.

Throughout the paper, we use two running examples to illustrate the concrete regret bounds our different algorithms get: the finite class and the linear class. In particular, for the linear class, this leads to five new results, summarized in Table 1 together with previous results. These results all have their own advantages and disadvantages, but we highlight the following:

- While all previous regret bounds depend on a problem-dependent constant $\kappa$ that can be exponentially large in the norm of the weight vector $B$ , none of our results depends on $\kappa$ . In fact, our best results (Corollary 17) even has only logarithmic dependence on $B$ , a potential doubly-exponential improvement compared to prior works. $^{1}$   
- The regret bounds of our two algorithms that reduce contextual MNL bandits to online regression are dimension-free, despite not having the optimal $\sqrt{T}$ -dependence (Corollary 12 and Corollary 15).   
- Our results are the first to handle completely adversarial context-reward pairs. $^{2}$

Related works. The (non-contextual) MNL model was initially studied in Rusmevichientong et al. (2010), followed by a line of improvements (Agrawal et al., 2016, 2017; Chen and Wang, 2018; Agrawal et al., 2019; Peeters et al., 2022). Specifically, Agrawal et al. (2016, 2019) introduced a UCB-type algorithm achieving $\widetilde{\mathcal{O}}(\sqrt{NT})$ regret and proved a lower bound of $\Omega(\sqrt{NT/K})$ . Subsequently, Chen and Wang (2018) enhanced the lower bound to $\Omega(\sqrt{NT})$ , matching the upper bound up to logarithmic factors.

Table 1: Comparisons of results for contextual MNL bandits with T rounds, N items, size-K assortments, and a d-dimensional linear value function class with norm bounded by B. All previous results depend on a problem-dependent constant $\kappa$ that is $\exp(2B)$ in the worst case, while ours (in gray) do not. The notation $\widetilde{\mathcal{O}}(\cdot)$ hides logarithmic dependency on all parameters. In the last column, $\checkmark$ means polynomial runtime in all parameters; $\checkmark$ means polynomial only when K is a constant; and X means not polynomial even for a small K. 

<table><tr><td>Context  $x_t$  &amp; reward  $r_t$ </td><td>Regret</td><td>Efficient?</td></tr><tr><td rowspan="2">Stochastic ( $x_t, r_t$ )</td><td> $\widetilde{\mathcal{O}}((dBNK)^{1/3}T^{2/3})$  (Corollary 5)</td><td>√</td></tr><tr><td> $\widetilde{\mathcal{O}}(K^2\sqrt{dBNT})$  (Corollary 8)</td><td>✕</td></tr><tr><td>Adversarial  $x_t, r_t \equiv 1$ </td><td> $\widetilde{\mathcal{O}}(dK\sqrt{T/\kappa} + d^2K^4\kappa)$  (Perivier and Goyal, 2022)</td><td>✕</td></tr><tr><td rowspan="3">Stochastic  $x_t$ Adversarial  $r_t$ </td><td> $\widetilde{\mathcal{O}}(d\sqrt{T} + d^2K^2\kappa^4)$  (Chen et al., 2020)</td><td>✕</td></tr><tr><td> $\widetilde{\mathcal{O}}(\kappa\sqrt{dT} + \kappa^4)$  (Oh and Iyengar, 2021)</td><td>✕</td></tr><tr><td> $\widetilde{\mathcal{O}}(d\sqrt{\kappa T} + \kappa^2)$  (Oh and Iyengar, 2021)</td><td>√</td></tr><tr><td rowspan="3">Adversarial ( $x_t, r_t$ )</td><td> $\mathcal{O}((NKB)^{1/3}T^{5/6})$  (Corollary 12)</td><td>√</td></tr><tr><td> $\mathcal{O}(K^2\sqrt{NBT}^{3/4})$  (Corollary 15)</td><td>✕</td></tr><tr><td> $\widetilde{\mathcal{O}}(K^{2.5}\sqrt{dNT})$  (Corollary 17)</td><td>✕</td></tr></table>

Cheung and Simchi-Levi (2017) first extended MNL bandits to its contextual version and designed a Thompson sampling based algorithm. Follow-up works consider this problem under different settings, including stochastic context (Chen et al., 2020; Oh and Iyengar, 2019, 2021), adversarial context (Ou et al., 2018; Agrawal et al., 2023), and uniform reward over items (Perivier and Goyal, 2022). However, as mentioned, all these works consider (generalized) linear value functions, and our work is the first to consider contextual MNL bandits under a general value function class.

Our work is also closely related to the recent trend of designing contextual bandits algorithms for a general function class. Due to space limit, we defer the discussion to Appendix A.

# 2. Notations and Preliminary

Notations. Throughout this paper, we denote the set $\{1,2,\ldots,N\}$ for some positive integer N by [N] and $\{0,1,2,\ldots,N\}$ by $[N]_{0}$ . For a vector $u\in R^{N}$ , we use $u_{i}$ to denote its i-th coordinate, and for a matrix $W\in R^{N\times M}$ , we use $W_{j}$ to denote its j-th column. For a set S, we denote by $\Delta(\mathcal{S})$ the set of distributions over S, and by conv(S) the convex hull of S. Finally, for a distribution $\mu\in\Delta([N]_{0})$ and an outcome $i\in[N]_{0}$ , the corresponding log loss is $\ell_{\log}(\mu,i)=-\log\mu_{i}$ .

We consider the following contextual MNL bandit problem that proceeds for T rounds. At each round t, the learner receives a context $x_{t} \in X$ for some arbitrary context space X and a reward vector $r_{t} \in [0,1]^{N}$ which specifics the reward of N items. Then, out of these N items, the learner needs to recommend a subset $S_{t} \subseteq S$ to a customer, where $S \subseteq 2^{[N]}$ is the collection of all subsets of [N] with cardinality at least 1 and at most K for some $K \leq N$ . Finally, the learner observes the customer purchase decision $i_{t} \in S_{t} \cup \{0\}$ , where 0 denotes the no-purchase option, and receives

reward $r_{t,i_{t}}$ , where for notational convenience we define $r_{t,0}=0$ for all t (no reward if no purchase). The customer decision $i_{t}$ is assumed to follow an MNL model:

$$
\operatorname * {P r} [ i _ {t} = i \mid S _ {t}, x _ {t} ] = \left\{ \begin{array}{l l} \frac {f _ {i} ^ {\star} (x _ {t})}{1 + \sum_ {j \in S _ {t}} f _ {j} ^ {\star} (x _ {t})} & \text { if   } i \in S _ {t}, \\ \frac {1}{1 + \sum_ {j \in S _ {t}} f _ {j} ^ {\star} (x _ {t})} & \text { if   } i = 0, \\ 0 & \text { otherwise }, \end{array} \right. \tag {1}
$$

where $f^{\star} : \mathcal{X} \to [0,1]^{N}$ is an unknown value function, specifying the costumer's value for each item under the given context. The MNL model above implicitly assumes a value of 1 for the no-purchase option, making it the most likely outcome. This is a standard assumption that holds in many realistic settings (Agrawal et al., 2019; Dong et al., 2020; Han et al., 2021).

To simplify notation, we define $\mu : \mathcal{S} \times [0,1]^N \to \Delta([N]_0)$ such that $\mu_i(S,v) \propto v_i\mathbf{1}[i \in S \cup \{0\}]$ with the convention $v_0 = 1$ . The purchase decision $i_t$ is thus sampled from the distribution $\mu(S_t, f^\star(x_t))$ . In addition, given a reward vector $r \in [0,1]^N$ (again, with convention $r_0 = 0$ ), we further define the expected reward of choosing subset $S \in \mathcal{S}$ under context $x \in \mathcal{X}$ as

$$
R (S, v, r) = \mathbb {E} _ {i \sim \mu (S, v)} [ r _ {i} ] = \sum_ {i \in S} \mu_ {i} (S, v) r _ {i} = \frac {\sum_ {i \in S} r _ {i} v _ {i}}{1 + \sum_ {i \in S} v _ {i}}.
$$

The goal of the learner is then to minimize her regret, defined as the expected gap between her total reward and that of the optimal strategy with the knowledge of $f^{\star}$ :

$$
\mathbf {R e g} _ {\mathrm{MNL}} = \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \max _ {S \in \mathcal {S}} R (S, f ^ {\star} (x _ {t}), r _ {t}) - \sum_ {t = 1} ^ {T} R (S _ {t}, f ^ {\star} (x _ {t}), r _ {t}) \right].
$$

To ensure that no-regret is possible, we make the following assumption, which is standard in the literature of contextual bandits.

Assumption 1 The learner is given a function class $\mathcal{F} = \{f:\mathcal{X}\to [0,1]^{N}\}$ which contains $f^{\star}$ .

Our hope is thus to design algorithms whose regret is sublinear in T and polynomial in N and some standard complexity measure of the function class F. So far, we have not specified how the context $x_{t}$ and the reward $x_{t}$ are chosen. In the next two sections, we will discuss both the easier stochastic case where $(x_{t}, r_{t})$ is jointly drawn from some fixed and unknown distribution, and the harder adversarial case where $(x_{t}, r_{t})$ can be arbitrarily chosen by an adversary.

# 3. Contextual MNL Bandits with Stochastic Contexts and Rewards

In this section, we consider contextual MNL bandits with stochastic contexts and rewards, where at each round $t \in [T]$ , $x_{t}$ and $r_{t}$ are jointly drawn from a fixed and unknown distribution D. Following the literature of contextual bandits, we aim to reduce the problem to an easier and better-studied offline regression problem and only access the function class F through some offline regression oracle. Specifically, an offline regression oracle $Alg_{off}$ takes as input a set of i.i.d. context-subset-purchase tuples and outputs a predictor from F with low generalization error in terms of log loss, formally defined as follows.

Algorithm 1 Contextual MNL Algorithms with an Offline Regression Oracle   
Input: an offline regression oracle $\mathsf{Alg}_{\mathrm{off}}$ satisfying Assumption 2  
Define: epoch schedule $\tau_0 = 0$ and $\tau_m = 2^{m-1} - 1$ for all $m = 1,2,\ldots$ .  
for epoch $m = 1,2,\ldots$ do  
    Feed $\{x_t,S_t,i_t\}_{t=\tau_{m-1}+1}^{\tau_m}$ to $\mathsf{Alg}_{\mathrm{off}}$ and obtain $f_m$ .  
    Define a stochastic policy $q_m:\mathcal{X}\times[0,1]^N\to\Delta(\mathcal{S})$ via either Eq. (4) or Eq. (5).  
    for $t=\tau_m+1,\cdots,\tau_{m+1}$ do  
    Observe context $x_t\in\mathcal{X}$ and reward vector $r_t\in[0,1]^N$ .  
    Sample $S_t\sim q_m(x_t,r_t)$ and recommend it to the customer.  
    Observe customer's purchase decision $i_t\in S_t\cup\{0\}$ , drawn according to Eq. (1).

Assumption 2 Given n samples $D = \{(x_k, S_k, i_k)\}_{k=1}^n$ where each $(x_k, S_k, i_k) \in \mathcal{X} \times \mathcal{S} \times [N]_0$ is an i.i.d. sample of some unknown distribution $\mathcal{H}$ and the conditional distribution of $i_k$ is $\mu(S_k, f^\star(x_k))$ , with probability at least $1 - \delta$ the offline regression oracle $\text{Alg}_{\text{off}}$ outputs a function $\widehat{f}_D \in \mathcal{F}$ such that:

$$
\mathbb {E} _ {(x, S, i) \sim \mathcal {H}} \left[ \ell_ {\log} (\mu (S, \widehat {f} _ {D} (x)), i) - \ell_ {\log} (\mu (S, f ^ {\star} (x)), i) \right] \leq \mathbf {E r r} _ {\log} (n, \delta , \mathcal {F}), \tag {2}
$$

for some function $\mathbf{Err}_{\log}(n, \delta, \mathcal{F})$ that is non-increasing in $n$ .

Given the similarity between MNL and multi-class logistic regression, assuming such a log loss regression oracle is more than natural. Indeed, in the following lemma, we prove that for both the finite class and a certain linear function class, the empirical risk minimizer (ERM) not only satisfies this assumption, but also enjoys a fast $1/n$ rate. The proof is based on a simple observation that our loss function $\ell_{\log}(\mu(S, f(x)), i)$ , when seen as a function of $f$ , satisfies the so-called strong 1-central condition (Grünwald and Mehta, 2020, Definition 7), which might be of independent interest; see Appendix B.1 for details.

Lemma 1 The ERM strategy $\widehat{f}_D = \operatorname{argmin}_{f\in \mathcal{F}}\sum_{(x,S,i)\in D}\ell_{\log}(\mu (S,f(x)),i)$ satisfies Assumption 2 for the following two cases:

- (Finite class) $\mathcal{F}$ is a finite class of functions with image $[\beta, 1]^N$ for some $\beta \in (0,1)$ and $\mathbf{Err}_{\log}(n, \delta, \mathcal{F}) = \mathcal{O}\left(\frac{\log \frac{K}{\beta} \log \frac{|\mathcal{F}|}{\delta}}{n}\right)$ .   
- (Linear class) $\mathcal{X} \subseteq \{x \in \mathbb{R}^{d \times N} \mid \| x_i \|_2 \leq 1, \forall i \in [N]\}$ , $\mathcal{F} = \{f_{\theta,i}(x) = e^{\theta^\top x_i - B} \mid \| \theta \|_2 \leq B\}$ , and $\operatorname{Err}_{\log}(n, \delta, \mathcal{F}) = \mathcal{O}\big(\frac{dB \log K \log B \log \frac{1}{\delta}}{n}\big)$ , for some $B > 0$ .

Given $Alg_{off}$ , we outline a natural algorithm framework that proceeds in epochs with exponentially increasing length (see Algorithm 1): At the beginning of each epoch m, the algorithm feeds all the context-subset-purchase tuples from the last epoch to the offline regression oracle $Alg_{off}$ and obtains a value predictor $f_{m}$ . Then, it decides in some way using $f_{m}$ a stochastic policy $q_{m}$ , which

maps a context $x$ and a reward vector $r \in [0,1]^N$ to a distribution over $S$ . With such a policy in hand, for every round $t$ within this epoch, the algorithm simply samples a subset $S_t$ according to $q_m(x_t, r_t)$ and recommend it to the customer.

We will specify two concrete stochastic policies $q_{m}$ in the next two subsections. Before doing so, we highlight some key parts of the analysis that shed light on how to design a “good” $q_{m}$ . The first step is an adaptation of Simchi-Levi and Xu (2021, Lemma 7), which quantifies the expected reward difference of any policy under the ground-truth value function $f^{\star}$ versus the estimated value function $f_{m}$ . Specifically, for a deterministic policy $\pi : X \times [0, 1]^{N} \to S$ mapping from a context-reward pair to a subset, we define its true expected reward and its expected reward under $f_{m}$ respectively as (overloading the notation R):

$$
R (\pi) = \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ R (\pi (x, r), f ^ {\star} (x), r) \right], \quad R _ {m} (\pi) = \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ R (\pi (x, r), f _ {m} (x), r) \right]. \tag {3}
$$

Moreover, for any $\rho \in \Delta(S)$ , define $w(\rho) \in [0,1]^N$ such that $w_i(\rho) = \sum_{S \in S: i \in S} \rho(S)$ is the probability of item $i$ being selected under distribution $\rho$ , and for any stochastic policy $q$ , further define a dispersion measure for a deterministic policy $\pi$ as $V(q, \pi) = \mathbb{E}_{(x,r) \sim \mathcal{D}} \left[ \sum_{i \in \pi(x,r)} \frac{1}{w_i(q(x,r))} \right]$ (the smaller $V(q, \pi)$ is, the more disperse the distribution induced by $q$ is). Using the Lipschitzness (in $v$ ) of the reward function $R(S, v, r)$ (Lemma 18), we prove the following.

Lemma 2 For any deterministic policy $\pi : \mathcal{X} \times [0,1]^N \to S$ and any epoch $m \geq 2$ , we have

$$
| R _ {m} (\pi) - R (\pi) | \leq \sqrt {V (q _ {m - 1} , \pi)} \cdot \sqrt {\mathbb {E} _ {(x , r) \sim \mathcal {D} , S \sim q _ {m - 1} (x , r)} \left[ \sum_ {i \in S} (f _ {m , i} (x) - f _ {i} ^ {\star} (x)) ^ {2} \right]}.
$$

If the learner could observe the true value of each item in the selected subset (or its noisy version), then doing squared loss regression on these values would make the squared loss term in Lemma 2 small; this is essentially the case in the contextual bandit problem studied by Simchi-Levi and Xu (2021). However, in our problem, only the purchase decisions are observed but not the true values that define the MNL model. Nevertheless, one of our key technical contributions is to show that the offline log-loss regression, which only relies on observing the purchase decisions, in fact also makes sure that the squared loss above is small.

Lemma 3 For any $S \in S$ and $v, v^{\star} \in [0,1]^{N}$ , we have

$$
\begin{array}{l} \frac {1}{2 (K + 1) ^ {4}} \sum_ {i \in S} (v _ {i} - v _ {i} ^ {\star}) ^ {2} \leq \| \mu (S, v) - \mu (S, v ^ {\star}) \| _ {2} ^ {2} \\ \leq 2 \mathbb {E} _ {i \sim \mu (S, v ^ {\star})} \left[ \ell_ {\log} (\mu (S, v), i) - \ell_ {\log} (\mu (S, v ^ {\star}), i) \right]. \\ \end{array}
$$

The first equality establishes certain “reverse Lipschitzness” of $\mu$ and is proven by providing a universal lower bound on the minimum singular value of its Jacobian matrix, which is new to our knowledge. It implies that if two value vectors induce a pair of close distributions, then they must be reasonably close as well. The second equality, proven using known facts, further states that to control the distance between two distributions, it suffices to control their log loss difference, which is exactly the job of the offline regression oracle.

Therefore, combining Lemma 2 and Lemma 3, we see that to design a good algorithm, it suffices to find a stochastic policy that “mostly” follows $\arg\max_{S} R(S, f_{m}(x_{t}), r_{t})$ , the best decision

according to the oracle's prediction, and at the same time ensures high dispersion for all $\pi$ such that the oracle's predicted reward for any policy is close to its true reward. The design of our two algorithms in the remaining of this section follows exactly this principle.

# 3.1. A Simple and Efficient Algorithm via Uniform Exploration

As a warm-up, we first introduce a simple but efficient $\varepsilon$ -greedy-type algorithm that ensures reasonable dispersion by uniformly exploring all the singleton sets. Specifically, at epoch m, given the value predictor $f_{m}$ from $Alg_{off}$ , $q_{m}(x,r) \in \Delta(S)$ is defined as follows for some $\varepsilon_{m} > 0$ :

$$
q _ {m} (S | x, r) = \left(1 - \varepsilon_ {m}\right) \mathbb {1} \left[ S = \underset {S ^ {\star} \in \mathcal {S}} {\operatorname{argmax}} R \left(S ^ {\star}, f _ {m} (x), r\right) \right] + \frac {\varepsilon_ {m}}{N} \sum_ {i = 1} ^ {N} \mathbb {1} [ S = \{i \} ]. \tag {4}
$$

In other words, with probability $1 - \varepsilon$ , the learner picks the subset achieving the maximum reward based on the reward vector r and the predicted value $f_{m}(x)$ ; with the remaining $\varepsilon$ probability, the learner selects a uniformly random item $i \in [N]$ and recommend only this item, which clearly ensures $V(q_{m}, \pi) \leq \frac{KN}{\varepsilon_{m}}$ for any $\pi$ . Based on our previous analysis, it is straightforward to prove the following regret guarantee.

Theorem 4 Under Assumption 1 and Assumption 2, Algorithm 1 with $q_{m}$ defined in Eq. (4) and the optimal choice of $\varepsilon_{m}$ ensures $\mathbf{Reg}_{\mathsf{MNL}} = \sum_{m=1}^{\lceil \log_2 T \rceil} \mathcal{O}\left(2^{m}(NK\mathbf{Err}_{\log}(2^{m-1}, 1/T^2, \mathcal{F}))^{\frac{1}{3}}\right)$ .

To better interpret this regret bound, we consider the finite class and the linear class discussed in Lemma 1. Combining it with Theorem 4, we immediately obtain the following corollary:

Corollary 5 Under Assumption 1, Algorithm 1 with $q_{m}$ defined in Eq. (4), the optimal choice of $\varepsilon_{m}$ , and ERM as $\mathsf{Alg}_{\mathrm{off}}$ ensures the following regret bounds for the finite class and the linear class discussed in Lemma 1:

- (Finite class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left((NK\log \frac{K}{\beta}\log (|\mathcal{F}|T))^{\frac{1}{3}}T^{\frac{2}{3}}\right);$   
- (Linear class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left((dBNK\log K)^{\frac{1}{3}}T^{\frac{2}{3}}\log B\log T\right)$ .

While these $\widetilde{\mathcal{O}}(T^{2/3})$ regret bounds are suboptimal, Theorem 4 provides the first computationally efficient algorithms for contextual MNL bandits with an offline regression oracle for a general function class. Indeed, computing $\operatorname{argmax}_{S^{\star}\in\mathcal{S}} R(S^{\star}, f_{m}(x), r)$ can be efficiently done in $\mathcal{O}(N^{2})$ time according to Rusmevichientong et al. (2010). Moreover, for the linear case, the ERM oracle can indeed be efficiently (and approximately) implemented because it is a convex optimization problem over a simple ball constraint. Importantly, previous regret bounds for the linear case all depend on a problem-dependent constant $\kappa = \max_{\|\theta\| \leq B, S \in S, i \in S, t \in [T]} \frac{1}{\mu_{i}(S, f_{\theta}(x_{t})) \mu_{0}(S, f_{\theta}(x_{t}))}$ , which is $\exp(2B)$ in the worst case (Chen et al., 2020; Oh and Iyengar, 2021; Perivier and Goyal, 2022), but ours only has polynomial dependence on B.

# 3.2. Better Exploration Leads to Better Regret

Next, we show that a more sophisticated construction of $q_{m}$ in Algorithm 1 leads to better exploration and consequently improved regret bounds. Specifically, $q_{m}$ is defined as (for some $\gamma_{m} > 0$ ):

$$
q _ {m} (x, r) = \underset {\rho \in \Delta (\mathcal {S})} {\operatorname{argmax}} \mathbb {E} _ {S \sim \rho} [ R (S, f _ {m} (x), r) ] - \frac {(K + 1) ^ {4}}{\gamma_ {m}} \sum_ {i = 1} ^ {N} \log \frac {1}{w _ {i} (\rho)}. \tag {5}
$$

The first term of the optimization objective above is the expected reward when one picks a subset according to $\rho$ and the value function is $f_{m}$ , while the second term is a certain log-barrier regularizer applied to $\rho$ , penalizing it for putting too little mass on any single item. This specific form of regularization ensures that $q_{m}$ enjoys a low-regret-high-dispersion guarantee, as shown below.

Lemma 6 For any $x \in X$ and $r \in [0,1]^{N}$ , the distribution $q_{m}(x,r)$ defined in Eq. (5) satisfies:

$$
\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - \mathbb {E} _ {S \sim q _ {m} (x, r)} [ R (S, f _ {m} (x), r) ] \leq \frac {N (K + 1) ^ {4}}{\gamma_ {m}}, \tag {6}
$$

$$
\forall S \in \mathcal {S}, \quad \sum_ {i \in S} \frac {1}{w _ {i} (q _ {m} (x , r))} \leq N + \frac {\gamma_ {m}}{(K + 1) ^ {4}} \left(\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - R (S, f _ {m} (x), r)\right). \tag {7}
$$

Eq. (6) states that following $q_{m}(x,r)$ does not incur too much regret compared to the best subset predicted by the oracle, and Eq. (7) states that the dispersion of $q_{m}(x,r)$ on any subset is controlled by how bad this subset is compared to the best one in terms of their predicted reward — a good subset has a large dispersion while a bad one can have a smaller dispersion since we do not care about estimating its true reward very accurately. Such a refined dispersion guarantee intuitively provides a much more adaptive exploration scheme compared to uniform exploration.

This kind of low-regret-high-dispersion guarantees is in fact very similar to the ideas of Simchi-Levi and Xu (2021) for contextual bandits (which itself is similar to an earlier work by Agarwal et al. (2014)). While Simchi-Levi and Xu (2021) were able to provide a closed-form strategy with such a guarantee for contextual bandits, we do not find a similar closed-form for MNL bandits and instead provide the strategy as the solution of an optimization problem Eq. (5). Unfortunately, we are not aware of an efficient way to solve Eq. (5) with polynomial time complexity, but one can clearly solve it in poly(|S|) = poly(N^K) time since it is a concave problem over $\Delta(S)$ . Thus, the algorithm is efficient when $K$ is small, which we believe is the case for most real-world applications.

Combining Lemma 2 and Lemma 6, we prove the following regret guarantee, which improves the $Err_{log}^{1/3}$ term in Theorem 4 to $Err_{log}^{1/2}$ (proofs deferred to Appendix B).

Theorem 7 Under Assumption 1 and Assumption 2, Algorithm 1 with $q_{m}$ defined in Eq. (5) and the optimal choice of $\gamma_{m}$ ensures $\mathbf{Reg}_{\mathsf{MNL}} = \mathcal{O}\left(\sum_{m=1}^{\lceil \log_2 T \rceil} 2^{m} K^{2} \sqrt{N\mathbf{Err}_{\log}(2^{m-1}, 1/T^{2}, \mathcal{F})}\right)$ .

Similar to Section 3.1, we instantiate Theorem 7 using the following two concrete classes:

Corollary 8 Under Assumption 1, Algorithm 1 with $q_{m}$ calculated via Eq. (5), the optimal choice of $\gamma_{m}$ , and ERM as $\mathrm{Alg}_{\mathrm{off}}$ ensures the following regret bounds for the finite class and the linear class discussed in Lemma 1:

\- (Finite class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left(K^2\sqrt{T\log\frac{K}{\beta}\log(|\mathcal{F}|T)}\right)$ ;

\- (Linear class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left(K^2\sqrt{dBNT\log B\log T}\right)$ .

The dependence on T in these $\mathcal{O}(\sqrt{T})$ regret bounds is known to be optimal (Chen and Wang, 2018; Chen et al., 2020). Once again, in the linear case, we have no exponential dependence on B, unlike previous results.

# 4. Contextual MNL Bandits with Adversarial Contexts and Rewards

In this section, we move on to consider the more challenging case where the context $x_{t}$ and the reward vector $r_{t}$ can both be arbitrarily chosen by an adversary. We propose two different approaches leading to three different algorithms, each with its own pros and cons.

# 4.1. First Approach: Reduction to Online Regression

In the first approach, we follow a recent trend of studies that reduces contextual bandits to online regression and only accesses $\mathcal{F}$ through an online regression oracle (Foster and Rakhlin, 2020; Foster and Krishnamurthy, 2021; Foster et al., 2022; Zhu and Mineiro, 2022; Zhang et al., 2023). More specifically, we assume access to an online regression oracle $\mathrm{Alg}_{\mathrm{on}}$ that follows the protocol below: at each round $t \in [T]$ , $\mathrm{Alg}_{\mathrm{on}}$ outputs a value predictor $f_t \in \mathrm{conv}(\mathcal{F})$ ; then, it receives a context $x_t$ , a subset $S_t$ , and a purchase decision $i_t \in S_t \cup \{0\}$ , all chosen arbitrarily, and suffers log loss $\ell_{\log}(\mu(S_t, f_t(x_t)), i_t)$ . The oracle is assumed to enjoy the following regret guarantee.

Assumption 3 The predictions made by the online regression oracle $\mathsf{Alg}_{\mathsf{on}}$ ensure:

$$
\mathbb {E} \left[ \sum_ {t = 1} ^ {T} \ell_ {\mathrm{log}} (\mu (S _ {t}, f _ {t} (x _ {t})), i _ {t}) - \sum_ {t = 1} ^ {T} \ell_ {\mathrm{log}} (\mu (S _ {t}, f ^ {\star} (x _ {t})), i _ {t}) \right] \leq \mathbf {R e g} _ {\mathrm{log}} (T, \mathcal {F}),
$$

for any $f^{\star} \in F$ and some regret bound $\mathbf{Reg}_{\log}(T, \mathcal{F})$ that is non-decreasing in T.

While most previous works on contextual bandits assume a squared loss online oracle, log loss is more than natural for our MNL model (it was also used by Foster and Krishnamurthy (2021) to achieve first-order regret guarantees for contextual bandits). The following lemma shows that Assumption 3 again holds for the finite class and the linear class.

Lemma 9 For the finite class and the linear class discussed in Lemma 1, the following concrete oracles satisfy Assumption 3:

- (Finite class) Hedge (Freund and Schapire, 1997) with $\mathbf{Reg}_{\log}(T,\mathcal{F}) = \mathcal{O}(\sqrt{T\log|\mathcal{F}|}\log \frac{K}{\beta})$ ;   
- (Linear class) Online Gradient Descent (Zinkevich, 2003) with $\mathbf{Reg}_{\log}(T,\mathcal{F}) = \mathcal{O}(B\sqrt{T})$ .

Unfortunately, unlike the offline oracle, we are not able to provide a “fast rate” (that is, $\tilde{\mathcal{O}}(1)$ regret) for these two cases, because our loss function does not appear to satisfy the standard Vovk’s mixability condition or any other sufficient conditions discussed in Van Erven et al. (2015). This is in sharp contrast to the standard multi-class logistic loss (Foster et al., 2018), despite the similarity

Algorithm 2 Contextual MNL Algorithms via an Online Regression Oracle   
Input: an online regression oracle $\mathsf{Alg}_{\mathrm{on}}$ satisfying Assumption 3.  
for $t = 1,2,\ldots ,T$ do  
    Obtain value predictor $f_{t}$ from oracle $\mathsf{Alg}_{\mathrm{on}}$ .  
    Receive context $x_{t}\in \mathcal{X}$ and reward vector $r_t\in [0,1]^N$ .  
    Calculate $q_{t}\in \Delta (\mathcal{S})$ based on $f(x_{t})$ and $r_t$ , via either Eq. (9) or Eq. (10).  
    Sample $S_{t}\sim q_{t}$ and receive purchase decision $i_t\in S_t\cup \{0\}$ drawn according Eq. (1).  
    Feed the tuple $(x_{t},S_{t},i_{t})$ to the oracle $\mathsf{Alg}_{\mathrm{on}}$ .

between these two models. We leave as an open problem whether fast rates exist for these two classes, which would have immediate consequences to our final MNL regret bounds below.

With this online regression oracle, a natural algorithm framework works as follows: at each round $t$ , the learner first obtains a value predictor $f_{t} \in \mathrm{conv}(\mathcal{F})$ from the regression oracle $\mathrm{Alg}_{\mathrm{on}}$ ; then, upon seeing context $x_{t}$ and reward vector $r_{t}$ , the learner decides in some way a distribution $q_{t} \in \Delta(S)$ based on $f_{t}(x_{t})$ and $r_{t}$ , and samples $S_{t}$ from $q_{t}$ ; finally, the learner observes the purchase decision $i_{t}$ and feeds the tuple $(x_{t}, S_{t}, i_{t})$ to the oracle $\mathrm{Alg}_{\mathrm{on}}$ (see Algorithm 2). To shed light on how to design a good sampling distribution $q_{t}$ , we first show a general lemma that holds for any $q_{t}$ .

Lemma 10 Under Assumption 1 and Assumption 3, Algorithm 2 (with any $q_{t}$ ) ensures

$$
\mathbf {R e g} _ {\mathrm{MNL}} \leq \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \mathsf {d e c} _ {\gamma} (q _ {t}; f _ {t} (x _ {t}), r _ {t}) \right] + 2 \gamma \mathbf {R e g} _ {\log} (T, \mathcal {F})
$$

for any $\gamma > 0$ , where $\operatorname{dec}_{\gamma}(q; v, r)$ is the Decision-Estimation Coefficient (DEC) defined as

$$
\max _ {v ^ {\star} \in [ 0, 1 ] ^ {N}} \max _ {S ^ {\star} \in \mathcal {S}} \left\{R (S ^ {\star}, v ^ {\star}, r) - \mathbb {E} _ {S \sim q} \left[ R (S, v ^ {\star}, r) \right] - \gamma \mathbb {E} _ {S \sim q} \left[ \| \mu (S, v) - \mu (S, v ^ {\star}) \| _ {2} ^ {2} \right] \right\}. \tag {8}
$$

Our DEC adopts the idea of Foster et al. (2021) for general decision making problems: the term $R(S^{\star}, v^{\star}, r) - \mathbb{E}_{S \sim q}[R(S, v^{\star}, r)]$ represents the instantaneous regret of strategy $q$ against the best subset $S^{\star}$ with respect to reward vector $r$ and the worst-case value vector $v^{\star}$ , and the term $\mathbb{E}_{S \sim q}[\| \mu(S, v) - \mu(S, v^{\star})\|_2^2]$ is the expected squared distance between two distributions induced by $v$ and $v^{\star}$ , which, in light of the second inequality of Lemma 3, lower bounds the instantaneous log loss regret of the online oracle. Therefore, a small DEC makes sure that the learner's MNL regret is somewhat close to the oracle's log loss regret $\mathbf{Reg}_{\log}$ , formally quantified by Lemma 10. With the goal of ensuring a small DEC, we again propose two strategies similar to Section 3.

Uniform Exploration. We start with a simple uniform exploration approach that is basically the same as Eq. (4):

$$
q _ {t} (S) = (1 - \varepsilon) \mathbb {1} \left[ S = \underset {S ^ {\star} \in \mathcal {S}} {\operatorname{argmax}} R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) \right] + \frac {\varepsilon}{N} \sum_ {i = 1} ^ {N} \mathbb {1} [ S = \{i \} ]. \tag {9}
$$

where $\varepsilon > 0$ is a parameter specifying the probability of uniformly exploring the singleton sets. We prove the following results for this simple algorithm.

Theorem 11 The strategy defined in Eq. (9) guarantees $\mathsf{dec}_{\gamma}(q_t; f_t(x_t), r_t) = \mathcal{O}(\frac{NK}{\gamma\varepsilon} + \varepsilon)$ . Consequently, under Assumption 1 and Assumption 3, Algorithm 2 with $q_t$ calculated via Eq. (9) and the optimal choice of $\varepsilon$ and $\gamma$ ensures $\mathbf{Reg}_{\mathsf{MNL}} = \mathcal{O}\left((NK\mathbf{Reg}_{\log}(T, \mathcal{F}))^{\frac{1}{3}} T^{\frac{2}{3}}\right)$ .

Combining this with Lemma 9, we immediately obtain the following corollary.

Corollary 12 Under Assumption 1, Algorithm 2 with $q_{t}$ defined in Eq. (9) and the optimal choice of $\varepsilon$ and $\gamma$ ensures the following regret bounds for the finite class (with Hedge as $\mathrm{Alg}_{\mathrm{on}}$ ) and the linear class (with Online Gradient Descent as $\mathrm{Alg}_{\mathrm{on}}$ ) discussed in Lemma 9:

- (Finite class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left((NK\log \frac{N}{\beta})^{\frac{1}{3}}T^{\frac{5}{6}}\right)$ ;   
- (Linear class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left((NKB)^{\frac{1}{3}}T^{\frac{5}{6}}\right)$ .

While these regret bounds have a large dependence on T, the advantage of this algorithm is its computational efficiency as discussed before.

Better Exploration. Can we improve the algorithm via a strategy with an even smaller DEC? In particular, what happens if we take the extreme and let $q_{t}$ be the minimizer of $\mathrm{dec}_{\gamma}(q; f_{t}(x_{t}), r_{t})$ ? Indeed, this is exactly the approach in several prior works that adopt the DEC framework (Foster et al., 2020; Zhang et al., 2023), where the exact minimizer for DEC is characterized and shown to achieve a small DEC value.

On the other hand, for our problem, it appears quite difficult to analyze the exact DEC minimizer. Somewhat surprisingly, however, we show that the same construction in Eq. (5) for the stochastic environment in fact also achieves a reasonably small DEC for the adversarial case:

Theorem 13 The following distribution

$$
q _ {t} = \underset {q \in \Delta (\mathcal {S})} {\operatorname{argmax}} \mathbb {E} _ {S \sim q} [ R (S, f _ {t} (x _ {t}), r _ {t}) ] - \frac {(K + 1) ^ {4}}{\gamma} \sum_ {i = 1} ^ {N} \log \frac {1}{w _ {i} (q)}, \tag {10}
$$

satisfies $\mathsf{dec}_{\gamma}(q_t, f_t(x_t), r_t) \leq \mathcal{O}\left(\frac{NK^4}{\gamma}\right)$ .

A couple of remarks are in order. First, while for some cases such as the contextual bandit problem studied by Foster et al. (2020), this kind of log-barrier regularized strategies is known to be the exact DEC minimizer, one can verify that this is not the case for our DEC. Second, the fact that the same strategy works for both the stochastic and the adversarial environments is similar to the case for contextual bandits where the same inverse gap weighting strategy works for both cases (Foster and Rakhlin, 2020; Simchi-Levi and Xu, 2021), but to our knowledge, the connection between these two cases is unclear since their analysis is quite different. Finally, our proof (in Appendix C) in fact relies on the same low-regret-high-dispersion property of Lemma 6, which is a new way to bound DEC as far as we know. More importantly, this to some extent demystifies the last two points: the reason that such log-barrier regularized strategies work regardless whether they are the exact minimizer or not and regardless whether the environment is stochastic or adversarial is all due to their inherent low-regret-high-dispersion property.

Combining Theorem 13 with Lemma 10, we obtain the following improved regret.

Theorem 14 Under Assumption 1 and Assumption 3, Algorithm 2 with $q_{t}$ calculated via Eq. (10) and the optimal choice of $\gamma$ ensures $\mathbf{Reg}_{\mathsf{MNL}} = \mathcal{O}\left(K^{2}\sqrt{NT\mathbf{Reg}_{\log}(T,\mathcal{F})}\right)$ .

Corollary 15 Under Assumption 1, Algorithm 2 with $q_t$ defined in Eq. (10) and the optimal choice of $\gamma$ ensures the following regret bounds for the finite class (with Hedge as $\mathsf{Alg}_{\mathsf{on}}$ ) and the linear class (with Online Gradient Descent as $\mathsf{Alg}_{\mathsf{on}}$ ) discussed in Lemma 9:

- (Finite class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left(K^2\sqrt{N\log\frac{K}{\beta}} T^{\frac{3}{4}}(\log |\mathcal{F}|)^{\frac{1}{4}}\right)$ ;   
- (Linear class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left(K^2\sqrt{NB} T^{\frac{3}{4}}\right)$ .

We remark that if the “fast rate” discussed after Lemma 9 exists, we would have obtained the optimal $\sqrt{T}$ -regret here. Despite having worse dependence on T, however, our result for the linear case enjoys three advantages compared to prior work (Chen et al., 2020; Oh and Iyengar, 2021; Perivier and Goyal, 2022): 1) no exponential dependence on B (as in all our other results), 2) no dependence at all on the dimension d, and 3) valid even when contexts and rewards are completely adversarial. We refer the reader to Table 1 again for detailed comparisons.

# 4.2. Second Approach: Feel-Good Thompson Sampling

The second approach we take to derive an algorithm for adversarial contextual MNL bandits is inspired by the Feel-Good Thompson Sampling algorithm of Zhang (2022) for contextual bandits. Specifically, the algorithm maintains a distribution $p_{t}$ over the value function class F, and at each round t, it samples $f_{t}$ from $p_{t}$ and selects the subset $S_{t}$ that maximizes the expected reward with respect to the value function $f_{t}$ and the reward vector $r_{t}$ . After receiving the purchase decision $i_{t}$ , the algorithm constructs a loss estimator $\widehat{\ell}_{t,f}$ for each $f \in F$ as defined in Eq. (29), and updates the distribution $p_{t}$ using a standard multiplicative update with learning rate $\eta$ . Due to space limit, the pseudocode is deferred to the appendix; see Algorithm 3 in Appendix D.

The idea of the loss estimator Eq. (29) is as follows. The first term measures how accurate f is via the squared distance between the multinomial distribution induced by f and the true outcome. The second term, which is the highest expected reward one could get if the value function was f, is subtracted from the first term to serve as a form of optimism (the “feel-good” part), encouraging exploration for those f’s that promise a high reward.

We extend the analysis of Zhang (2022) and combine it with our technical lemmas (such as Lemma 3 and Lemma 18) to prove the following regret guarantee (see Appendix D for the proof).

Theorem 16 Under Assumption 1, Algorithm 3 with learning rate $\eta \leq 1$ ensures $\mathbf{Reg}_{\mathsf{MNL}} \leq 12\eta NK(K + 1)^4 T + 4\eta T + \frac{Z_T}{\eta}$ , where $Z_T = -\mathbb{E}[\log \mathbb{E}_{f \sim p_1}[\exp(-\eta \sum_{t=1}^T (\widehat{\ell}_{t,f} - \widehat{\ell}_{t,f^*}))]]$ .

The term $Z_{T}$ should be interpreted as a certain complexity measure for the class F. To better understand this regret bound, we again instantiate it for the two classes below.

Corollary 17 Under Assumption 1, Algorithm 3 with the optimal choice of $\eta$ ensures the following regret bounds for finite class and the linear class:

\- (Finite class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left(K^{2.5}\sqrt{NT\log|\mathcal{F}|}\right)$ ;

$$
\bullet \text {(Linear class)} \mathbf {R e g} _ {\mathrm{MNL}} = \mathcal {O} \left(K ^ {2. 5} \sqrt {d N T \log (B T)}\right).
$$

In terms of the dependence on T, Algorithm 3 achieves the best (and in fact optimal) regret bounds among all our results. For the linear case, it even has only logarithmic dependence on B, a potential doubly-exponential improvement compared to prior works. The caveat is that there is no efficient way to implement the algorithm even for the linear case and even when K is a constant (unlike all our other algorithms), because sampling from $p_{t}$ , a distribution that does not enjoy log-concavity or other nice properties, is generally hard. We leave the question of whether there exists a computationally efficient algorithm (even only for small K) with a $\sqrt{T}$ -regret bound that has no exponential dependence on B as a key future direction.

# References

Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. International Conference on Machine Learning, 2014.   
Priyank Agrawal, Theja Tulabandhula, and Vashist Avadhanula. A tractable online learning algorithm for the multinomial logit contextual bandit. European Journal of Operational Research, 2023.   
Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. A near-optimal exploration-exploitation approach for assortment selection. Proceedings of the ACM Conference on Economics and Computation, 2016.   
Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Thompson sampling for the mnl-bandit. Conference on Learning Theory, 2017.   
Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Mnl-bandit: A dynamic learning approach to assortment selection. Operations Research, 67(5), 2019.   
Xi Chen and Yining Wang. A note on a tight lower bound for capacitated mnl-bandit assortment selection models. Operations Research Letters, 46(5), 2018.   
Xi Chen, Yining Wang, and Yuan Zhou. Dynamic assortment optimization with changing contextual information. The Journal of Machine Learning Research, 21(1), 2020.   
Wang Chi Cheung and David Simchi-Levi. Thompson sampling for online personalized assortment optimization problems with multinomial logit choice models. Available at SSRN 3075658, 2017.   
Kefan Dong, Yingkai Li, Qin Zhang, and Yuan Zhou. Multinomial logit bandit with low switching cost. International Conference on Machine Learning, 2020.   
Dylan Foster and Alexander Rakhlin. Beyond ucb: Optimal and efficient contextual bandits with regression oracles. International Conference on Machine Learning, 2020.   
Dylan J Foster and Akshay Krishnamurthy. Efficient first-order contextual bandits: Prediction, allocation, and triangular discrimination. Conference on Advances in Neural Information Processing Systems, 2021.

Dylan J Foster, Satyen Kale, Haipeng Luo, Mehryar Mohri, and Karthik Sridharan. Logistic regression: The importance of being improper. Conference On Learning Theory, 2018.   
Dylan J Foster, Claudio Gentile, Mehryar Mohri, and Julian Zimmert. Adapting to misspecification in contextual bandits. Conference on Neural Information Processing Systems, 2020.   
Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487, 2021.   
Dylan J Foster, Alexander Rakhlin, Ayush Sekhari, and Karthik Sridharan. On the complexity of adversarial decision making. Conference on Advances in Neural Information Processing Systems, 2022.   
Yoav Freund and Robert E Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 1997.   
Peter D Grünwald and Nishant A Mehta. Fast rates for general unbounded loss functions: from erm to generalized bayes. The Journal of Machine Learning Research, 2020.   
Yanjun Han, Yining Wang, and Xi Chen. Adversarial combinatorial bandits with general non-linear reward functions. International Conference on Machine Learning, pages 4030–4039, 2021.   
Min-hwan Oh and Garud Iyengar. Thompson sampling for multinomial logit contextual bandits. Conference on Advances in Neural Information Processing Systems, 32, 2019.   
Min-hwan Oh and Garud Iyengar. Multinomial logit contextual bandits: Provable optimality and practicality. Proceedings of the AAAI conference on artificial intelligence, 35(10), 2021.   
Mingdong Ou, Nan Li, Shenghuo Zhu, and Rong Jin. Multinomial logit bandit with linear utility functions. Proceedings of the International Joint Conference on Artificial Intelligence, 2018.   
Yannik Peeters, Arnoud V den Boer, and Michel Mandjes. Continuous assortment optimization with logit choice probabilities and incomplete information. Operations Research, 70(3), 2022.   
Noemie Perivier and Vineet Goyal. Dynamic pricing and assortment under a contextual mnl demand. Conference on Advances in Neural Information Processing Systems, 35, 2022.   
Paat Rusmevichientong, Zuo-Jun Max Shen, and David B Shmoys. Dynamic assortment optimization with a multinomial logit choice model and capacity constraint. Operations research, 58(6), 2010.   
David Simchi-Levi and Yunzong Xu. Bypassing the monster: A faster and simpler optimal algorithm for contextual bandits under realizability. Mathematics of Operations Research, 2021.   
Tim Van Erven, Peter Grunwald, Nishant A Mehta, Mark Reid, Robert Williamson, et al. Fast rates in statistical and online learning. Journal of Machine Learning Research, 54(6), 2015.   
Yunbei Xu and Assaf Zeevi. Upper counterfactual confidence bounds: a new optimism principle for contextual bandits. arXiv preprint arXiv:2007.07876, 2020.

Mengxiao Zhang and Haipeng Luo. Online learning in contextual second-price pay-per-click auctions. International Conference on Artificial Intelligence and Statistics, 2024.   
Mengxiao Zhang, Yuheng Zhang, Olga Vrousgou, Haipeng Luo, and Paul Mineiro. Practical contextual bandits with feedback graphs. Conference on Neural Information Processing Systems, 2023.   
Tong Zhang. Feel-good thompson sampling for contextual bandits and reinforcement learning. SIAM Journal on Mathematics of Data Science, 4(2), 2022.   
Yinglun Zhu and Paul Mineiro. Contextual bandits with smooth regret: Efficient learning in continuous action spaces. International Conference on Machine Learning, 2022.   
Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. International Conference on Machine Learning, 2003.

# Appendix A. Additional Related Works

As mentioned, our work is closely related to the recent trend of designing contextual bandits algorithms for a general function class. Specifically, under stochastic context, Xu and Zeevi (2020); Simchi-Levi and Xu (2021) designed algorithms based on an offline squared loss regression oracle and achieved optimal regret guarantees. Under adversarial context, there are two lines of works. The first one reduces the contextual bandit problem to online regression (Foster and Rakhlin, 2020; Foster and Krishnamurthy, 2021; Foster et al., 2021; Zhu and Mineiro, 2022; Zhang et al., 2023), while the second one is based on the ability to sample from a certain distribution over the function class using Markov chain Monte Carlo methods (Zhang, 2022; Zhang and Luo, 2024). We follow and greatly extend the ideas of all these approaches to design algorithms for contextual MNL bandits.

# Appendix B. Omitted Details in Section 3

# B.1. Offline Regression Oracle

We start by proving Lemma 1, which shows that ERM strategy satisfies Assumption 2 for the finite class and the linear function class.

Proof [of Lemma 1] We first show that our log loss function $\ell_{\log}(\mu(S, f(x)), i)$ satisfies the so-called strong 1-central condition (Definition 7 of Grünwald and Mehta (2020)), which states that there exists $f_0 \in \mathcal{F}$ , such that for any $f \in \mathcal{F}$ ,

$$
\mathbb {E} _ {(x, S, i) \sim \mathcal {H}} \left[ \exp (\ell_ {\log} (\mu (S, f (x)), i) - \ell_ {\log} (\mu (S, f _ {0} (x)), i)) \right] \leq 1.
$$

Indeed, by picking $f_{0} = f^{\star}$ , we know that

$$
\begin{array}{l} \mathbb {E} _ {(x, S, i) \sim \mathcal {H}} \left[ \exp (\ell_ {\log} (\mu (S, f (x), i)) - \ell_ {\log} (S, f ^ {\star} (x), i)) \right] \\ = \mathbb {E} _ {(x, S)} \mathbb {E} _ {i \sim \mu (S, f ^ {\star} (x))} \left[ \frac {\mu_ {i} (S , f)}{\mu_ {i} (S , f ^ {\star})} \right] \\ = \mathbb {E} _ {(x, S)} \left[ \sum_ {i \in S \cup \{0 \}} \mu_ {i} (S, f) \right] = 1, \\ \end{array}
$$

certifying the strong 1-central condition.

Now, we first consider the case where $\mathcal{F}$ is finite. Since $f_{i}(x) \geq \beta$ for all $x \in \mathcal{X}$ and $i \in [N]$ , we know that for any $i \in [N]_0$ , we have (defining $f_{0}(x) = 1$ )

$$
\ell_ {\log} (\mu (S, f (x)), i) = \log \frac {1 + \sum_ {j \in S} f _ {j} (x)}{f _ {i} (x)} \leq \log \frac {K + 1}{\beta}.
$$

Therefore, according to Theorem 7.6 of (Van Erven et al., 2015), we know that given n i.i.d samples $D = \{(x_k, S_k, i_k)\}_{k \in [n]}$ , ERM predictor $\widehat{f}_D$ guarantees that with probability $1 - \delta$ :

$$
\mathbb {E} _ {(x, S, i) \sim \mathcal {D}} \left[ \ell_ {\log} (\mu (S, \widehat {f} _ {D} (x)), i) \right] \leq \mathbb {E} _ {(x, S, i) \sim \mathcal {D}} \left[ \ell_ {\log} (\mu (S, f ^ {\star} (x)), i) \right] + \mathcal {O} \left(\frac {\log \frac {K}{\beta} \log \frac {| \mathcal {F} |}{\delta}}{n}\right).
$$

Next, we consider the linear function class. In this case, we know that $x_{i}^{\top}\theta - B \in [-2B, 0]$ for all $x_{i}$ . Therefore, $\ell_{\log}(\mu(S, f(x)), i)$ is bounded by $2B + 2\ln N$ for all $x \in X$ , $f \in F$ , $S \in S$ and $i \in [N]$ since

$$
\ell_ {\mathrm{log}} (\mu (S, f (x)), i) = \log \frac {1 + \sum_ {j \in S} \exp (x _ {j} ^ {\top} \theta)}{\exp (x _ {i} ^ {\top} \theta)} \leq \log \frac {1 + K}{e ^ {- 2 B}} \leq 2 B + 2 \log K,
$$

and the same bound clearly holds as well for $i = 0$ . Moreover, since

$$
\left\| \nabla_ {\theta} \log \frac {1 + \sum_ {j \in S} \exp (x _ {j} ^ {\top} \theta)}{\exp (x _ {i} ^ {\top} \theta)} \right\| _ {2} = \left\| \frac {\sum_ {j \in S} \exp (\theta^ {\top} x _ {j}) x _ {j}}{1 + \sum_ {j \in S} \exp (\theta^ {\top} x _ {j})} - x _ {i} \right\| _ {2} \leq 2,
$$

we know that the $\varepsilon$ -covering number of $\ell_{\log}\circ\mathcal{F}$ is bounded by $\left(\frac{16B}{\varepsilon}\right)^{d}$ . Therefore, according to Theorem 7.7 of (Van Erven et al., 2015), we know that given n i.i.d samples $D=\{(x_{k},S_{k},i_{k})\}_{k\in[n]}$ , ERM predictor $\widehat{f}_{D}$ guarantees that with probability $1-\delta$ :

$$
\mathbb {E} _ {(x, S, i) \sim \mathcal {D}} \left[ \ell_ {\log} (\mu (S, \widehat {f} _ {D} (x)), i) \right] \leq \mathbb {E} _ {(x, S, i) \sim \mathcal {D}} \left[ \ell_ {\log} (\mu (S, f ^ {\star} (x)), i) \right] + \mathcal {O} \left(\frac {d B \log K \log B \log \frac {1}{\delta}}{n}\right).
$$

# B.2. Analysis of Algorithm 1

We first prove the following lemma, which shows that the expected reward function $R(S, v, r)$ is 1-Lipschitz in the value vector v.

Lemma 18 Given $r \in [0,1]^N$ and $S \subseteq [N]$ , function $R(S,v,r) = \frac{\sum_{i \in S} r_i v_i}{1 + \sum_{i \in S} v_i}$ satisfies that for any $v', v \in [0,\infty)^N$ , $|R(S,v,r) - R(S,v',r)| \leq \sum_{i \in S} |v_i - v_i'|$ .

Proof Taking derivative with respect to $v_{j}$ for $j \in S$ , we know that

$$
\left| \nabla_ {v _ {j}} R (S, v, r) \right| = \left| \frac {r _ {j} (1 + \sum_ {i \in S} v _ {i}) - \sum_ {j \in S} r _ {j} v _ {j}}{(1 + \sum_ {i \in S} v _ {i}) ^ {2}} \right| \leq \max \left\{\frac {r _ {j}}{1 + \sum_ {i \in S} v _ {i}}, \frac {\sum_ {i \in S} v _ {i}}{(1 + \sum_ {i \in S} v _ {i}) ^ {2}} \right\} \leq 1,
$$

where both inequalities are because $r_{j} \in [0, 1]$ . This finishes the proof.

Next, we restate and prove Lemma 2.

Lemma 2 For any deterministic policy $\pi : \mathcal{X} \times [0,1]^N \to S$ and any epoch $m \geq 2$ , we have

$$
| R _ {m} (\pi) - R (\pi) | \leq \sqrt {V (q _ {m - 1} , \pi)} \cdot \sqrt {\mathbb {E} _ {(x , r) \sim \mathcal {D} , S \sim q _ {m - 1} (x , r)} \left[ \sum_ {i \in S} (f _ {m , i} (x) - f _ {i} ^ {\star} (x)) ^ {2} \right]}.
$$

Proof We proceed as:

$$
\begin{array}{l} \left| R _ {m} (\pi) - R (\pi) \right| \\ = \left| \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ R (\pi (x, r), f _ {m} (x), r) - R (\pi (x, r), f ^ {\star} (x), r) \right] \right| \\ \leq \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \sum_ {i = 1} ^ {N} \mathbb {1} \{i \in \pi (x, r) \} | f _ {m, i} (x) - f _ {i} ^ {\star} (x) | \right] \tag {11} \\ \leq \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \sqrt {\sum_ {i = 1} ^ {N} \frac {\mathbb {1} \{i \in \pi (x , r) \}}{w _ {i} (q _ {m - 1} | x , r)} \sum_ {i = 1} ^ {N} w _ {i} (q _ {m - 1} | x , r) \left(f _ {m , i} (x) - f _ {i} ^ {\star} (x)\right) ^ {2}} \right] \\ \end{array}
$$

(Cauchy–Schwarz inequality)

$$
\leq \sqrt {\mathbb {E} _ {(x , r) \sim \mathcal {D}} \left[ \sum_ {i = 1} ^ {N} \frac {\mathbb {1} \{i \in \pi (x , r) \}}{w _ {i} (q _ {m - 1} | x , r)} \right]} \cdot \sqrt {\mathbb {E} _ {(x , r) \sim \mathcal {D}} \left[ \sum_ {i = 1} ^ {N} w _ {i} (q _ {m - 1} | x , r) \left(f _ {m , i} (x) - f _ {i} ^ {\star} (x)\right) ^ {2} \right]}
$$

(Cauchy–Schwarz inequality)

$$
\begin{array}{l} = \sqrt {V (q _ {m - 1} , \pi)} \cdot \sqrt {\mathbb {E} _ {(x , r) \sim \mathcal {D}} \left[ \sum_ {i = 1} ^ {N} w _ {i} (q _ {m - 1} | x , r) \left(f _ {m , i} (x) - f _ {i} ^ {\star} (x)\right) ^ {2} \right]} \\ = \sqrt {V (q _ {m - 1} , \pi)} \cdot \sqrt {\mathbb {E} _ {(x , r) \sim \mathcal {D} , S \sim q _ {m - 1} (x , r)} \left[ \sum_ {i = 1} ^ {N} \left(f _ {m , i} (x) - f _ {i} ^ {\star} (x)\right) ^ {2} \right]}, \tag {12} \\ \end{array}
$$

where the first inequality uses the convexity of the absolute value function and Lemma 18.

Next, to prove Lemma 3, we first prove the following key technical lemma (where 1 denotes the all-one vector).

Lemma 19 Let $h(a) = \frac{a}{1 + \mathbf{1}^{\top}a}$ for $a \in [0,1]^d$ . Then, for any $a, b \in [0,1]^d$ , we have

$$
\frac {1}{2 (d + 1) ^ {4}} \| a - b \| _ {2} ^ {2} \leq \| h (a) - h (b) \| _ {2} ^ {2}.
$$

Proof The Jacobian matrix of $h$ is

$$
H (a) = \frac {1}{1 + \mathbf {1} ^ {\top} a} \mathbf {I} - \frac {\mathbf {1} a ^ {\top}}{(1 + \mathbf {1} ^ {\top} a) ^ {2}}.
$$

Therefore, there exists $z \in \mathrm{conv}(\{a, b\})$ such that $\| h(a) - h(b) \|_2 = \| H(z)(a - b) \|_2$ . It thus remains to figure out the minimum singular value of $H(z)$ , which is equal to the reciprocal of the spectral norm of $H(z)^{-1}$ . By Sherman-Morrison formula, we know that

$$
H (z) ^ {- 1} = \left(1 + \mathbf {1} ^ {\top} z\right) \left(\mathbf {I} + \mathbf {1} z ^ {\top}\right).
$$

Therefore, we have

$$
\begin{array}{l} H (z) ^ {- 1} H (z) ^ {- \top} = (1 + \mathbf {1} ^ {\top} z) ^ {2} (\mathbf {I} + \mathbf {1} z ^ {\top}) (\mathbf {I} + \mathbf {1} z ^ {\top}) ^ {\top} \\ = (1 + \mathbf {1} ^ {\top} z) ^ {2} (\mathbf {I} + \mathbf {1} z ^ {\top} + z \mathbf {1} ^ {\top} + z ^ {\top} z \mathbf {1 1} ^ {\top}). \\ \end{array}
$$

Note that for any $u$ that is perpendicular to the subspace spanned by $\{z, 1\}$ , we have $H(z)^{-1}H(z)^{-\top}u = (1 + 1^{\top}z)^{2}u$ . Therefore, there are $d - 2$ identical eigenvalues 1 for the matrix $\frac{1}{(1 + 1^{\top}z)^{2}} H(z)^{-1}H(z)^{-\top}$ . Let the remaining two eigenvalues of $\frac{1}{(1 + 1^{\top}z)^{2}} H(z)^{-1}H(z)^{-\top}$ be $\lambda_{1}$ and $\lambda_{2}$ . Note that

$$
\lambda_ {1} \lambda_ {2} = \det \left((\mathbf {I} + \mathbf {1} z ^ {\top}) (\mathbf {I} + z \mathbf {1} ^ {\top})\right) = (1 + \mathbf {1} ^ {\top} z) ^ {2},
$$

$$
\lambda_ {1} + \lambda_ {2} = \operatorname{Trace} (\mathbf {I} + \mathbf {1} z ^ {\top} + z \mathbf {1} ^ {\top} + z ^ {\top} z \mathbf {1 1} ^ {\top}) - (d - 2)
$$

$$
= 2 + 2 \mathbf {1} ^ {\top} z + z ^ {\top} z \mathbf {1} ^ {\top} \mathbf {1}
$$

$$
= 2 + 2 \mathbf {1} ^ {\top} z + d \cdot z ^ {\top} z
$$

$$
\leq 2 + 2 d + d ^ {2}.
$$

Therefore, we know that $\max\{\lambda_{1},\lambda_{2}\}\leq\lambda_{1}+\lambda_{2}\leq2+2d+d^{2}$ , meaning that

$$
\| H (z) ^ {- 1} H (z) ^ {- \top} \| _ {2} \leq 2 (1 + \mathbf {1} ^ {\top} z) ^ {2} (1 + d + d ^ {2}) \leq 2 (1 + d) ^ {2} (d ^ {2} + d + 1) \leq 2 (d + 1) ^ {4}.
$$

This further means that the minimum singular value of $H(z)$ is at least $\frac{1}{\sqrt{2}(d+1)^{2}}$ . Therefore, we can conclude that

$$
\| h (a) - h (b) \| _ {2} \geq \frac {1}{\sqrt {2} (d + 1) ^ {2}} \| a - b \| _ {2},
$$

leading to

$$
\| h (a) - h (b) \| _ {2} ^ {2} \geq \frac {1}{2 (d + 1) ^ {4}} \| a - b \| _ {2} ^ {2}.
$$

Next, we restate and prove Lemma 3.

Lemma 3 For any $S \in S$ and $v, v^{\star} \in [0,1]^{N}$ , we have

$$
\begin{array}{l} \frac {1}{2 (K + 1) ^ {4}} \sum_ {i \in S} (v _ {i} - v _ {i} ^ {\star}) ^ {2} \leq \| \mu (S, v) - \mu (S, v ^ {\star}) \| _ {2} ^ {2} \\ \leq 2 \mathbb {E} _ {i \sim \mu (S, v ^ {\star})} \left[ \ell_ {\log} (\mu (S, v), i) - \ell_ {\log} (\mu (S, v ^ {\star}), i) \right]. \\ \end{array}
$$

Proof The first inequality follows directly from Lemma 19 using the fact that $|S| \leq K$ for all $S \in S$ . Consider the second inequality. For any $\mu, \mu' \in \Delta([K])$ , by definition of $\ell_{\log}(\mu, i)$ , we know that

$$
\mathbb {E} _ {i \sim \mu} \left[ \ell_ {\log} (\mu^ {\prime}, i) - \ell_ {\log} (\mu , i) \right] = \mathbb {E} _ {i \sim \mu} \left[ \log \frac {\mu_ {i} ^ {\prime}}{\mu_ {i}} \right] = \mathrm{KL} (\mu , \mu^ {\prime}) \geq \frac {1}{2} \| \mu - \mu^ {\prime} \| _ {1} ^ {2} \geq \frac {1}{2} \| \mu - \mu^ {\prime} \| _ {2} ^ {2},
$$

where the first inequality is due to Pinsker's inequality.

# B.3. Omitted Details in Section 3.1

In this section, we show omitted details in Section 3.1. For ease of presentation, we assume that the distribution over context-reward pair D has finite support. All our results can be directly generalized to the case with infinite support following a similar argument in Appendix A.7 of (Simchi-Levi and Xu, 2021). Define $\Psi: X \times [0,1]^{N} \mapsto S$ as the set of all deterministic policy. Following Lemma 3 in (Simchi-Levi and Xu, 2021), we know that for any context $x \in X$ and reward vector $r \in [0,1]^{N}$ , and any stochastic policy $q: X \times [0,1]^{N} \mapsto \Delta(S)$ , there exists an equivalent randomized policy $Q \in \Delta(\Psi)$ such that for all $S \in S$ , $x \in X$ , and $r \in [0,1]^{N}$ ,

$$
q (S | x, r) = \sum_ {\pi \in \Psi} \mathbb {1} \{\pi (x, r) = S \} Q (\pi).
$$

Let $Q_{m}$ be the randomized policy induced by $q_{m}$ . Define $\operatorname{Reg}(\pi)$ and $\operatorname{Reg}_{m}(\pi)$ as:

$$
\operatorname{Reg} (\pi) = R \left(\pi_ {f ^ {\star}}\right) - R (\pi), \quad \operatorname{Reg} _ {m} (\pi) = R _ {m} \left(\pi_ {f _ {m}}\right) - R _ {m} (\pi), \tag {13}
$$

where $R(\pi)$ and $R_{m}(\pi)$ are defined in Eq. (3) and $\pi_f$ is the policy that maps each $(x,r)$ to the one-hot distribution supported on $\operatorname{argmax}_{S\in S}R(S,f(x),r)$ .

Following the analysis in (Simchi-Levi and Xu, 2021), we show that to analyze our algorithm's expected regret, we only need to analyze the induced randomized policies' implicit regret.

Lemma 20 Fix any epoch $m$ . For any round $t$ in this epoch, we have

$$
\mathbb {E} _ {(x _ {t}, r _ {t}) \sim \mathcal {D}, S _ {t} \sim q _ {m} (x _ {t}, r _ {t})} \left[ R (\pi_ {f ^ {\star}} (x _ {t}, r _ {t}), f ^ {\star} (x), r _ {t}) - R (S _ {t}, f ^ {\star} (x), r _ {t}) \right] = \sum_ {\pi \in \Psi} Q _ {m} (\pi) \mathrm{Reg} (\pi).
$$

Proof Direct calculation shows that

$$
\begin{array}{l} \mathbb {E} _ {(x _ {t}, r _ {t}) \sim \mathcal {D}, S _ {t} \sim q _ {m} (x _ {t}, r _ {t})} \left[ R (\pi_ {f ^ {\star}} (x _ {t}, r _ {t}), f ^ {\star} (x), r _ {t}) - R (S _ {t}, f ^ {\star} (x), r _ {t}) \right] \\ = \mathbb {E} _ {(x _ {t}, r _ {t}) \sim \mathcal {D}} \left[ R (\pi_ {f ^ {\star}} (x _ {t}, r _ {t}), f ^ {\star} (x), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {m} (S | x _ {t}, r _ {t}) R (S, f ^ {\star} (x), r _ {t}) \right] \\ = \mathbb {E} _ {(x _ {t}, r _ {t}) \sim \mathcal {D}} \left[ R (\pi_ {f ^ {\star}} (x _ {t}, r _ {t}), f ^ {\star} (x), r _ {t}) - \sum_ {S \in \mathcal {S}} \sum_ {\pi \in \Psi} \mathbb {1} \{\pi (x _ {t}, r _ {t}) = S \} Q _ {m} (\pi) R (S, f ^ {\star} (x), r _ {t}) \right] \\ = \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \sum_ {S \in \mathcal {S}} \sum_ {\pi \in \Psi} \mathbb {1} \{\pi (x, r) = S \} Q _ {m} (\pi) \left(R \left(\pi_ {f ^ {*}} (x, r), f ^ {\star} (x), r\right) - R (S, f ^ {\star} (x), r)\right) \right] \\ = \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \sum_ {\pi \in \Psi} Q _ {m} (\pi) \left(R (\pi_ {f ^ {*}} (x, r), f ^ {\star} (x), r) - R (\pi (x, r), f ^ {\star} (x), r)\right) \right] \\ = \sum_ {\pi \in \Psi} Q _ {m} (\pi) \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ R \left(\pi_ {f ^ {*}} (x, r), f ^ {\star} (x), r\right) - R \left(\pi (x, r), f ^ {\star} (x), r\right) \right] \\ = \sum_ {\pi \in \Psi} Q _ {m} (\pi) \mathrm{Reg} (\pi), \\ \end{array}
$$

which finishes the proof.

To prove our main results for Algorithm 1, we define the following good event:

Event 1 For all epoch $m \geq 2$ , $f_{m}$ satisfies

$$
\mathbb {E} _ {(x, r) \sim \mathcal {D}, S \sim q _ {m - 1} (x, r), i \sim \mu (S, f ^ {\star} (x))} \left[ \ell_ {\log} (\mu (S, f _ {m} (x)), i) - \ell_ {\log} (\mu (S, f ^ {\star} (x)), i) \right]
$$

$$
\leq \mathbf {E r r} _ {\log} (\tau_ {m} - \tau_ {m - 1}, 1 / T ^ {2}, \mathcal {F}).
$$

According to Assumption 2, Event 1 happens with probability at least $1 - \frac{1}{T}$ since there are at most $T$ epochs.

Although now we have all ingredients to analyze our $\varepsilon$ -greedy-type algorithm defined Eq. (4), to get the exact result in Theorem 4, we will in fact need a refined version of Lemma 2, which eventually provides a tighter regret guarantee.

Lemma 21 Suppose that Event 1 holds. Algorithm 1 with $q_{t}$ defined in Eq. (4) satisfies that for any deterministic policy $\pi \in \Psi$ and any epoch $m \geq 2$ , we have

$$
| R _ {m} (\pi) - R (\pi) | \leq 8 \sqrt {\frac {N K}{\varepsilon_ {m - 1}}} \cdot \sqrt {\mathbf {E r r} _ {\log} (2 ^ {m - 2} , 1 / T ^ {2} , \mathcal {F})}.
$$

Proof Following Eq. (11) in the proof of Lemma 2, we know that

$$
| R _ {m} (\pi) - R (\pi) |
$$

$$
\leq \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \sum_ {i = 1} ^ {N} \mathbb {1} \{i \in \pi (x, r) \} | f _ {m, i} (x) - f _ {i} ^ {\star} (x) | \right]
$$

$$
\leq \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \sqrt {\sum_ {i = 1} ^ {N} \frac {N \mathbb {1} \{i \in \pi (x , r) \}}{\varepsilon_ {m - 1}} \sum_ {i = 1} ^ {N} \frac {\varepsilon_ {m - 1}}{N} \left(f _ {m , i} (x) - f _ {i} ^ {\star} (x)\right) ^ {2}} \right]
$$

(Cauchy–Schwarz inequality)

$$
\leq \sqrt {\mathbb {E} _ {(x , r) \sim \mathcal {D}} \left[ \sum_ {i = 1} ^ {N} \frac {N \mathbb {1} \{i \in \pi (x , r) \}}{\varepsilon_ {m - 1}} \right]} \cdot \sqrt {\mathbb {E} _ {(x , r) \sim \mathcal {D}} \left[ \sum_ {i = 1} ^ {N} \frac {\varepsilon_ {m - 1}}{N} \left(f _ {m , i} (x) - f _ {i} ^ {\star} (x)\right) ^ {2} \right]}
$$

(Cauchy–Schwarz inequality)

$$
\leq \sqrt {\frac {N K}{\varepsilon_ {m - 1}}} \cdot \sqrt {\mathbb {E} _ {(x , r) \sim \mathcal {D}} \left[ \sum_ {i = 1} ^ {N} \frac {\varepsilon_ {m - 1}}{N} \left(f _ {m , i} (x) - f _ {i} ^ {\star} (x)\right) ^ {2} \right]}. \tag {14}
$$

Since $f_{m}$ is the output of $Alg_{off}$ with i.i.d tuples $\{(x_{t}, S_{t}, i_{t})\}_{t=\tau_{m-1}+1}^{\tau_{m}}$ , according to Lemma 3 and Event 1, we know that

$$
6 4 \mathbf {E r r} _ {\log} (\tau_ {m} - \tau_ {m - 1}, 1 / T ^ {2}, \mathcal {F})
$$

$$
\geq 3 2 \mathbb {E} _ {(x, r) \sim \mathcal {D}, S \sim q _ {m - 1} (x, r)} \left[ \| \mu (S, f _ {m} (x)) - \mu (S, f ^ {\star} (x)) \| _ {2} ^ {2} \right] \tag {Lemma3}
$$

$$
\geq \frac {3 2 \varepsilon_ {m - 1}}{N} \sum_ {i = 1} ^ {N} \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \| \mu (\{i \}, f _ {m} (x)) - \mu (\{i \}, f ^ {\star} (x)) \| _ {2} ^ {2} \right] \quad \text {(according to Eq. (4))}
$$

$$
\geq \frac {\varepsilon_ {m - 1}}{N} \sum_ {i = 1} ^ {N} \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \sum_ {i = 1} ^ {N} (f _ {m, i} (x) - f _ {i} ^ {\star} (x)) ^ {2} \right]. \quad \text {(using Lemma 19 with $d = 1$)}
$$

Plugging the above inequality back to Eq. (14) and noticing that $\tau_{m}=2^{m-1}-1$ , we know that

$$
| R _ {m} (\pi) - R (\pi) | \leq 8 \sqrt {\frac {N K}{\varepsilon_ {m - 1}} \cdot \mathbf {E r r} _ {\log} (2 ^ {m - 2} , 1 / T ^ {2} , \mathcal {F})}.
$$

Now we are ready to prove Theorem 4

Theorem 4 Under Assumption 1 and Assumption 2, Algorithm 1 with $q_{m}$ defined in Eq. (4) and the optimal choice of $\varepsilon_{m}$ ensures $\mathbf{Reg}_{\mathsf{MNL}} = \sum_{m=1}^{\lceil \log_2 T \rceil} \mathcal{O}\left(2^{m}(NK\mathbf{Err}_{\log}(2^{m-1}, 1/T^2, \mathcal{F}))^{\frac{1}{3}}\right)$ .

Proof Consider the regret within epoch $m \geq 2$ . Under Event 1, we know that for any $\pi \in \Psi$ ,

$$
\begin{array}{l} \operatorname{Reg} (\pi) = R \left(\pi_ {f ^ {\star}}\right) - R (\pi) \\ = \left(R (\pi_ {f ^ {\star}}) - R _ {m} (\pi_ {f _ {m}})\right) - \left(R _ {m} (\pi) - R _ {m} (\pi_ {f _ {m}})\right) + \left(R _ {m} (\pi) - R (\pi)\right) \\ \leq (R (\pi_ {f ^ {\star}}) - R _ {m} (\pi_ {f ^ {\star}})) + (R _ {m} (\pi_ {f _ {m}}) - R _ {m} (\pi)) + (R _ {m} (\pi) - R (\pi)) \\ \leq (R _ {m} (\pi_ {f _ {m}}) - R _ {m} (\pi)) + 1 6 \sqrt {\frac {N K}{\varepsilon_ {m - 1}} \cdot \mathbf {E r r} _ {\log} (2 ^ {m - 2} , 1 / T ^ {2} , \mathcal {F})}, \tag {15} \\ \end{array}
$$

where the first inequality is because $R_{m}(\pi_{f_{m}}) \geq R_{m}(\pi_{f^{\star}})$ by definition and the second inequality is due to Lemma 21. Taking summation over all rounds within epoch m and picking $\varepsilon_{m} = (NK)^{\frac{1}{3}}\mathbf{Err}_{\log}^{\frac{1}{3}}(2^{m-2}, 1/T^{2}, \mathcal{F})$ , we know that

$$
\begin{array}{l} \mathbb {E} \left[ \sum_ {t = \tau_ {m} + 1} ^ {\tau_ {m + 1}} \left(\max _ {S \in \mathcal {S}} R (S, x _ {t}, f ^ {\star} (x _ {t})) - R (S _ {t}, x _ {t}, f ^ {\star} (x _ {t}))\right) \right] \\ = (\tau_ {m + 1} - \tau_ {m}) \mathbb {E} \left[ \sum_ {\pi \in \Psi} Q _ {m} (\pi) \mathrm{Reg} (\pi) \right] \tag {Lemma20} \\ \end{array}
$$

$$
\stackrel {(i)} {\leq} 2 ^ {m - 1} \cdot \mathbb {E} \left[ ((1 - \varepsilon_ {m}) \mathrm{Reg} (\pi_ {f _ {m}}) + \varepsilon_ {m}) \right]
$$

$$
\stackrel {(i i)} {\leq} \frac {2 ^ {m - 1}}{T} + 2 ^ {m - 1} \mathbb {E} \left[ ((1 - \varepsilon_ {m}) \mathrm{Reg} (\pi_ {f _ {m}}) + \varepsilon_ {m})   \bigg |   \text {Event 1 holds} \right]
$$

$$
\stackrel {(i i i)} {\leq} \frac {2 ^ {m - 1}}{T} + 2 ^ {m - 1} \left(\varepsilon_ {m} + 1 6 \sqrt {\frac {N K}{\varepsilon_ {m - 1}} \cdot \mathbf {E r r} _ {\log} (2 ^ {m - 2} , 1 / T ^ {2} , \mathcal {F})}\right)
$$

$$
\stackrel {(i v)} {=} \frac {2 ^ {m - 1}}{T} + \mathcal {O} \left(2 ^ {m - 1} \left(N K \mathbf {E r r} _ {\log} (2 ^ {m - 2}, 1 / T ^ {2}, \mathcal {F})\right) ^ {\frac {1}{3}}\right),
$$

where $(i)$ is due to $\tau_{m}=2^{m-1}-1$ and the construction of $q_{m}(x,r)$ defined in Eq. (4); $(ii)$ is because Event 1 holds with probability at least $1-\frac{1}{T}$ ; $(iii)$ uses Eq. (15); and $(iv)$ is due to the choice of $\varepsilon_{m}$ . Taking summation over all $m=2,3,\ldots\lceil\log_{2}T\rceil+1$ epochs, we can obtain that

$$
\mathbf {R e g} _ {\mathrm{MNL}} = \sum_ {m = 1} ^ {\lceil \log_ {2} T \rceil} \mathcal {O} \left(2 ^ {m} \left(N K \mathbf {E r r} _ {\log} (2 ^ {m - 1}, 1 / T ^ {2}, \mathcal {F})\right) ^ {\frac {1}{3}}\right).
$$

# B.4. Omitted Details in Section 3.2

First, we restate and prove Lemma 6, which shows that $q_{m}$ defined in Eq. (5) enjoys a low-regret-high-dispersion guarantee.

Lemma 6 For any $x \in X$ and $r \in [0,1]^{N}$ , the distribution $q_{m}(x,r)$ defined in Eq. (5) satisfies:

$$
\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - \mathbb {E} _ {S \sim q _ {m} (x, r)} [ R (S, f _ {m} (x), r) ] \leq \frac {N (K + 1) ^ {4}}{\gamma_ {m}}, \tag {6}
$$

$$
\forall S \in \mathcal {S}, \quad \sum_ {i \in S} \frac {1}{w _ {i} (q _ {m} (x , r))} \leq N + \frac {\gamma_ {m}}{(K + 1) ^ {4}} \left(\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - R (S, f _ {m} (x), r)\right). (7)
$$

Proof It is direct to see that solving Eq. (5) is equivalent to solving the following optimization problem:

$$
\underset {\rho \in \Delta (\mathcal {S})} {\operatorname{argmin}} \mathbb {E} _ {S \sim \rho} \left[ \underset {S ^ {\star} \in \mathcal {S}} {\max} R (S ^ {\star}, f _ {m} (x), r) - R (S, f _ {m} (x), r) \right] + \frac {(K + 1) ^ {4}}{\gamma_ {m}} \sum_ {i = 1} ^ {N} \log \frac {1}{w _ {i} (\rho)}. \tag {16}
$$

Moreover, relaxing the constraint $\rho$ from $\Delta(S)$ to $\{\rho\in[0,1]^S:\sum_{S\in S}\rho(S)\leq 1\}$ in Eq. (16) does not change the solution, since for any $\rho\in[0,1]^S$ such that $\sum_{S\in S}\rho(S)<1$ , putting the remaining $1-\sum_{S\in S}\rho(S)$ probability mass on $\operatorname{argmax}_{S^{\star}\in S}R(S^{\star},f_m(x),r)$ can only make the objective smaller.

Now, consider the Lagrangian form of Eq. (16) over this relaxed constraint and set the derivative with respect to $\rho(S)$ to zero. We obtain

$$
\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - R (S, f _ {m} (x), r) - \frac {(K + 1) ^ {4}}{\gamma_ {m}} \sum_ {i: i \in S} \frac {1}{w _ {i} (\rho)} - \lambda (S) + \lambda = 0, \tag {17}
$$

where $\lambda\geq0$ and $\lambda(S)\geq0$ , $S\in S$ are the Lagrangian multipliers. Let $\rho^{\star}\in\Delta(S)$ be the optimal solution of Eq. (16). Replacing $\rho$ by $\rho^{\star}$ in Eq. (17), multiplying Eq. (17) by $\rho^{\star}(S)$ for each $S\in S$ , and taking the summation over $S\in S$ , we know that

$$
\begin{array}{l} \sum_ {S \in \mathcal {S}} \rho^ {\star} (S) \left(\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - R (S, f _ {m} (x), r)\right) \\ - \frac {(K + 1) ^ {4}}{\gamma_ {m}} \sum_ {S \in \mathcal {S}} \rho^ {\star} (S) \sum_ {i: i \in S} \frac {1}{w _ {i} (\rho^ {\star})} - \sum_ {S \in \mathcal {S}} \rho^ {\star} (S) \lambda (S) + \lambda = 0. \\ \end{array}
$$

Rearranging the terms, we know that

$$
\begin{array}{l} \sum_ {S \in \mathcal {S}} \rho^ {\star} (S) \left(\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - R (S, f _ {m} (x), r)\right) \\ = \frac {(K + 1) ^ {4}}{\gamma_ {m}} \sum_ {S \in \mathcal {S}} \rho^ {\star} (S) \sum_ {i: i \in S} \frac {1}{w _ {i} (\rho^ {\star})} + \sum_ {S \in \mathcal {S}} \rho^ {\star} (S) \lambda (S) - \lambda \\ = \frac {(K + 1) ^ {4}}{\gamma_ {m}} \sum_ {i = 1} ^ {N} \frac {1}{w _ {i} (\rho^ {\star})} \sum_ {S \in \mathcal {S}: i \in S} \rho^ {\star} (S) - \lambda \quad \text {(complementary slackness)} \\ \end{array}
$$

$$
= \frac {N (K + 1) ^ {4}}{\gamma_ {m}} - \lambda \leq \frac {N (K + 1) ^ {4}}{\gamma_ {m}},
$$

proving Eq. (6). The above also implies that $\lambda \leq \frac{N(K + 1)^4}{\gamma_m}$ since

$$
\sum_ {S \in \mathcal {S}} \rho^ {\star} (S) \left(\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - R (S, f _ {m} (x), r)\right) \geq 0.
$$

Therefore, Eq. (17) implies that for any $S \in \mathcal{S}$ ,

$$
\begin{array}{l} \sum_ {i: i \in S} \frac {1}{w _ {i} (\rho^ {\star})} = \frac {\gamma_ {m}}{(K + 1) ^ {4}} \left(\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - R (S, f _ {m} (x), r) - \lambda_ {S} + \lambda\right) \\ \leq \frac {\gamma_ {m}}{(K + 1) ^ {4}} \left(\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {m} (x), r) - R (S, f _ {m} (x), r)\right) + N, \\ \end{array}
$$

where the last inequality uses the fact that $\lambda \leq \frac{N(K + 1)^4}{\gamma_m}$ and $\lambda_S \geq 0$ . This proves Eq. (7).

Now, to prove Theorem 7, we first prove the following lemma, which shows that the regret with respect to the true value function $f^{\star}$ and the one respect to the value predictor $f_{m}$ is within a factor of 2 plus an additional term of order $\frac{N(K + 1)^4}{\gamma_m}$ .

Lemma 22 Suppose that Event 1 holds. For all epochs $m \geq 2$ , all rounds $t$ in this epoch, and all policies $\pi \in \Psi$ , with $\gamma_{m} = \max \left\{1, \sqrt{\frac{N(K + 1)^4}{\mathbf{Err}_{\log}(2^{m - 2}, 1 / T^2, \mathcal{F})}}\right\}$ and $\lambda = 33$ , we have

$$
\operatorname{Reg} (\pi) \leq 2 \cdot \operatorname{Reg} _ {m} (\pi) + \frac {\lambda N (K + 1) ^ {4}}{\gamma_ {m}},
$$

$$
\mathrm{Reg} _ {m} (\pi) \leq 2 \cdot \mathrm{Reg} (\pi) + \frac {\lambda N (K + 1) ^ {4}}{\gamma_ {m}}.
$$

Proof We prove this by induction. The base case holds trivially. Suppose that this holds for all epochs with index less than m. Consider epoch m. We first show that $\operatorname{Reg}(\pi) \leq 2\operatorname{Reg}_{m}(\pi) + \frac{\lambda N(K+1)^{4}}{\gamma_{m}}$ for all deterministic policy $\pi \in \Psi$ . This holds trivially if $\sqrt{\frac{N(K+1)^{4}}{\mathbf{Err}_{\log}(2^{m-2},1/T^{2},\mathcal{F})}} \leq 1$ since $\operatorname{Reg}(\pi) \leq 1$ . Consider the case in which $\gamma_{m} = \sqrt{\frac{N(K+1)^{4}}{\mathbf{Err}_{\log}(2^{m-2},1/T^{2},\mathcal{F})}}$ . Specifically, we have

$$
\operatorname{Reg} (\pi) - \operatorname{Reg} _ {m} (\pi)
$$

$$
= \left(R (\pi_ {f ^ {\star}}) - R (\pi)\right) - \left(R _ {m} (\pi_ {f _ {m}}) - R _ {m} (\pi)\right)
$$

$$
\stackrel {(i)} {\leq} (R (\pi_ {f ^ {\star}}) - R (\pi)) - (R _ {m} (\pi_ {f ^ {\star}}) - R _ {m} (\pi))
$$

$$
\leq \left| R _ {m} (\pi_ {f ^ {\star}}) - R (\pi_ {f ^ {\star}}) \right| + \left| R _ {m} (\pi) - R (\pi) \right|
$$

$$
\stackrel {(i i)} {\leq} \sqrt {V (q _ {m - 1} , \pi_ {f ^ {\star}}) \cdot \mathbb {E} _ {(x , r) \sim \mathcal {D} , S \sim q _ {m - 1} (x , r)} \left[ \sum_ {i \in S} (f _ {m , i} (x) - f _ {i} ^ {\star} (x)) ^ {2} \right]}
$$

$$
+ \sqrt {V (q _ {m - 1} , \pi) \cdot \mathbb {E} _ {(x , r) \sim \mathcal {D} , S \sim q _ {m - 1} (x , r)} \left[ \sum_ {i \in S} (f _ {m , i} (x) - f _ {i} ^ {\star} (x)) ^ {2} \right]}, \tag {18}
$$

where (i) is because $R_{m}(\pi_{f_{m}}) \geq R_{m}(\pi_{f^{\star}})$ by definition and (ii) follows Lemma 2. Next, using Lemma 3 and Lemma 19, since Event 1 holds, we know that

$$
\begin{array}{l} 4 (K + 1) ^ {4} \mathbf {E r r} _ {\log} (2 ^ {m - 2}, 1 / T ^ {2}, \mathcal {F}) \\ \geq 2 (K + 1) ^ {4} \mathbb {E} _ {(x, r) \sim \mathcal {D}, S \sim q _ {m - 1} (x, r)} \left[ \| \mu (S, f _ {m} (x)) - \mu (S, f ^ {\star} (x)) \| _ {2} ^ {2} \right] (Lemma3) \\ \geq \mathbb {E} _ {(x, r) \sim \mathcal {D}, S \sim q _ {m - 1} (x, r)} \left[ \sum_ {i \in S} (f _ {m, i} (x) - f _ {i} ^ {\star} (x)) ^ {2} \right]. (Lemma19) \\ \end{array}
$$

Plugging the above back to Eq. (18), we obtain that

$$
\begin{array}{l} \operatorname{Reg} (\pi) - \operatorname{Reg} _ {m} (\pi) (19) \\ \leq 2 (K + 1) ^ {2} \sqrt {V (q _ {m - 1} , \pi_ {f ^ {\star}}) \mathbf {E r r} _ {\log} (2 ^ {m - 2} , 1 / T ^ {2} , \mathcal {F})} \\ + 2 (K + 1) ^ {2} \sqrt {V (q _ {m - 1} , \pi) \mathbf {E r r} _ {\log} (2 ^ {m - 2} , 1 / T ^ {2} , \mathcal {F})} \\ \leq \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi_ {f ^ {\star}})}{8 \gamma_ {m}} + \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi)}{8 \gamma_ {m}} + 1 6 \gamma_ {m} \mathbf {E r r} _ {\log} (2 ^ {m - 2}, 1 / T ^ {2}, \mathcal {F}) \\ = \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi_ {f ^ {\star}})}{8 \gamma_ {m}} + \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi)}{8 \gamma_ {m}} + \frac {1 6 N (K + 1) ^ {4}}{\gamma_ {m}}, (20) \\ \end{array}
$$

(AM-GM inequality)

where the last equality is because $\gamma_{m} = \sqrt{\frac{N(K + 1)^{4}}{\mathbf{Err}_{\log}(2^{m - 2},1 / T^{2},\mathcal{F})}}$ . According to Lemma 6, we know that for all $\pi \in \Psi$ ,

$$
\begin{array}{l} V (q _ {m - 1}, \pi) = \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \sum_ {i \in \pi (x, r)} \frac {1}{w _ {i} (q _ {m - 1} | x , r)} \right] \\ \leq \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ N + \frac {\gamma_ {m - 1}}{(K + 1) ^ {4}} \left(\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, r, f _ {m - 1} (x)) - R (S, r, f _ {m - 1} (x))\right) \right] \\ = N + \frac {\gamma_ {m - 1}}{(K + 1) ^ {4}} \mathrm{Reg} _ {m - 1} (\pi). \tag {21} \\ \end{array}
$$

Using Eq. (21), we bound the first and the second term in Eq. (20) as follows

$$
\begin{array}{l} \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi)}{8 \gamma_ {m}} \leq \frac {N (K + 1) ^ {4}}{8 \gamma_ {m}} + \frac {\gamma_ {m - 1} \mathrm{Reg} _ {m - 1} (\pi)}{8 \gamma_ {m}} \\ \leq \frac {N (K + 1) ^ {4}}{8 \gamma_ {m}} + \frac {\gamma_ {m - 1} \left(2 \mathrm{Reg} (\pi) + \frac {\lambda N (K + 1) ^ {4}}{\gamma_ {m - 1}}\right)}{8 \gamma_ {m}} \\ \leq \frac {1}{4} \operatorname{Reg} (\pi) + \frac {\lambda + 1}{8 \gamma_ {m}} \cdot N (K + 1) ^ {4}, \quad \text {(since} \gamma_ {m - 1} \leq \gamma_ {m}) \\ \end{array}
$$

$$
\frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi_ {f ^ {\star}})}{8 \gamma_ {m}} \leq \frac {N (K + 1) ^ {4}}{8 \gamma_ {m}} + \frac {\gamma_ {m - 1} \mathrm{Reg} _ {m - 1} (\pi_ {f ^ {\star}})}{8 \gamma_ {m}}
$$

$$
\leq \frac {N (K + 1) ^ {4}}{8 \gamma_ {m}} + \frac {\gamma_ {m - 1} \left(2 \mathrm{Reg} (\pi_ {f ^ {\star}}) + \frac {\lambda N (K + 1) ^ {4}}{\gamma_ {m - 1}}\right)}{8 \gamma_ {m}}
$$

$$
\leq \frac {\lambda + 1}{8 \gamma_ {m}} \cdot N (K + 1) ^ {4}. \quad (\text { since   } \operatorname{Reg} (\pi_ {f ^ {*}}) = 0 \text {   and   } \gamma_ {m - 1} \leq \gamma_ {m})
$$

Plugging back to Eq. (20), we know that

$$
\mathrm{Reg} (\pi) - \mathrm{Reg} _ {m} (\pi) \leq \frac {1}{4} \mathrm{Reg} (\pi) + \frac {1 6 N (K + 1) ^ {4}}{\gamma_ {m}} + \frac {\lambda + 1}{4 \gamma_ {m}} N (K + 1) ^ {4}.
$$

Rearranging the terms, we know that

$$
\begin{array}{l} \operatorname{Reg} (\pi) \leq \frac {4}{3} \operatorname{Reg} _ {m} (\pi) + \frac {1 2 N (K + 1) ^ {4}}{\gamma_ {m}} + \frac {\lambda + 1}{3 \gamma_ {m}} N (K + 1) ^ {4} \\ \leq 2 \operatorname{Reg} _ {m} (\pi) + \frac {\lambda N (K + 1) ^ {4}}{\gamma_ {m}}, \tag {22} \\ \end{array}
$$

where the last inequality uses $\lambda = 33$ .

For the other direction, similar to Eq. (20), we know that

$$
\begin{array}{l} \mathrm{Reg} _ {m} (\pi) - \mathrm{Reg} (\pi) \\ = \left(R _ {m} (\pi_ {f _ {m}}) - R _ {m} (\pi)\right) - \left(R (\pi_ {f ^ {\star}}) - R (\pi)\right) \\ \leq \left(R (\pi_ {f _ {m}}) - R (\pi)\right) - \left(R (\pi_ {f _ {m}}) - R (\pi)\right) \\ \leq | R _ {m} (\pi_ {f _ {m}}) - R (\pi_ {f _ {m}}) | + | R _ {m} (\pi) - R (\pi) | \\ \leq 2 (K + 1) ^ {2} \sqrt {V (q _ {m - 1} , \pi_ {f _ {m}}) \mathbf {E r r} _ {\log} (2 ^ {m - 2} , 1 / T ^ {2} , \mathcal {F})} \\ + 2 (K + 1) ^ {2} \sqrt {V (q _ {m - 1} , \pi) \mathbf {E r r} _ {\log} (2 ^ {m - 2} , 1 / T ^ {2} , \mathcal {F})} \\ \leq \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi_ {f _ {m}})}{8 \gamma_ {m}} + \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi)}{8 \gamma_ {m}} + 1 6 \gamma_ {m} \mathbf {E r r} _ {\log} (2 ^ {m - 2}, 1 / T ^ {2}, \mathcal {F}) \\ \end{array}
$$

(AM-GM inequality)

$$
\stackrel {(i)} {=} \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi_ {f _ {m}})}{8 \gamma_ {m}} + \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi)}{8 \gamma_ {m}} + \frac {1 6 N (K + 1) ^ {4}}{\gamma_ {m}}, \tag {23}
$$

where (i) is again because $\gamma_{m} = \sqrt{\frac{N(K + 1)^{4}}{\mathbf{Err}_{\log}(2^{m - 2},1 / T^{2},\mathcal{F})}}$ . Applying Eq. (21) to the first term in Eq. (23), we know that

$$
\begin{array}{l} \frac {(K + 1) ^ {4} V (q _ {m - 1} , \pi_ {f _ {m}})}{8 \gamma_ {m}} \\ \leq \frac {N (K + 1) ^ {4}}{8 \gamma_ {m}} + \frac {\gamma_ {m - 1} \mathrm{Reg} _ {m - 1} (\pi_ {f _ {m}})}{8 \gamma_ {m}} \\ \leq \frac {N (K + 1) ^ {4}}{8 \gamma_ {m}} + \frac {\gamma_ {m - 1} \left(2 \mathrm{Reg} (\pi_ {f _ {m}}) + \frac {\lambda N (K + 1) ^ {4}}{\gamma_ {m - 1}}\right)}{8 \gamma_ {m}} \\ \stackrel {(i)} {\leq} \frac {\lambda + 1}{8 \gamma_ {m}} \cdot N (K + 1) ^ {4} + \frac {1}{4} \left(2 \operatorname{Reg} _ {m} (\pi_ {f _ {m}}) + \frac {\lambda N (K + 1) ^ {4}}{\gamma_ {m}}\right) \\ \stackrel {(i i)} {=} \frac {1 + 3 \lambda}{8 \gamma_ {m}} N (K + 1) ^ {4}, \\ \end{array}
$$

where $(i)$ is because $\gamma_{m-1} \leq \gamma_{m}$ and Eq. (22), and $(ii)$ is due to $\operatorname{Reg}_{m}(\pi_{f_{m}}) = 0$ . Plugging the above back to Eq. (23), we obtain that

$$
\begin{array}{l} \mathrm{Reg} _ {m} (\pi) \leq \mathrm{Reg} (\pi) + \frac {2 + 4 \lambda}{8 \gamma_ {m}} N (K + 1) ^ {4} + \frac {1}{4} \mathrm{Reg} (\pi) + \frac {1 6 N (K + 1) ^ {4}}{\gamma_ {m}} \\ \leq 2 \operatorname{Reg} (\pi) + \frac {\lambda N (K + 1) ^ {4}}{\gamma_ {m}}, \quad (\text { since } \lambda = 3 3) \\ \end{array}
$$

which finishes the proof.

Now we are ready to prove Theorem 7.

Theorem 7 Under Assumption 1 and Assumption 2, Algorithm 1 with $q_{m}$ defined in Eq. (5) and the optimal choice of $\gamma_{m}$ ensures $\mathbf{Reg}_{\mathsf{MNL}} = \mathcal{O}\left(\sum_{m=1}^{\lceil \log_2 T \rceil} 2^{m} K^{2} \sqrt{N\mathbf{Err}_{\log}(2^{m-1}, 1/T^2, \mathcal{F})}\right)$ .

Proof Choose $\gamma_{m} = \max \left\{1, \sqrt{\frac{N(K + 1)^{4}}{\mathbf{Err}_{\log}(2^{m - 2},1 / T^{2},\mathcal{F})}}\right\}$ for all $m \geq 2$ . Consider the regret within epoch $m \geq 2$ . We first show that $\sum_{\pi \in \Psi} Q_{m}(\pi)\mathrm{Reg}_{m}(\pi) \leq \frac{N(K + 1)^{4}}{\gamma_{m}}$ . Concretely, according to Lemma 6 and Lemma 20, we know that

$$
\begin{array}{l} \sum_ {\pi \in \Psi} Q _ {m} (\pi) \mathrm{Reg} _ {m} (\pi) \\ = \mathbb {E} _ {(x, r) \sim \mathcal {D}} \left[ \sum_ {S \in \mathcal {S}} q _ {m} (S | x, r) \left(\max _ {S ^ {\star} \in \mathcal {S}} R \left(S ^ {\star}, f _ {m} (x), r\right) - R \left(S, f _ {m} (x), r\right)\right) \right] \leq \frac {N (K + 1) ^ {4}}{\gamma_ {m}}. \tag {24} \\ \end{array}
$$

Now consider the regret within epoch $m$ . Since Event 1 holds with probability at least $1 - \frac{1}{T}$ , we know that

$$
\begin{array}{l} \mathbb {E} \left[ \sum_ {t = \tau_ {m} + 1} ^ {\tau_ {m + 1}} \left(\max _ {S \in \mathcal {S}} R (S, x _ {t}, f ^ {\star} (x _ {t})) - R (S _ {t}, x _ {t}, f ^ {\star} (x _ {t}))\right) \right] \\ = (\tau_ {m + 1} - \tau_ {m}) \mathbb {E} \left[ \sum_ {\pi \in \Psi} Q _ {m} (\pi) \mathrm{Reg} (\pi) \right] \\ \leq \frac {\tau_ {m + 1} - \tau_ {m}}{T} + (\tau_ {m + 1} - \tau_ {m}) \mathbb {E} \left[ \sum_ {\pi \in \Psi} Q _ {m} (\pi) \operatorname{Reg} (\pi) \Bigg | \text { Event   1   holds } \right] \\ \end{array}
$$

(since Event 1 holds with probability at least $1 - \frac{1}{T}$ )

$$
\stackrel {(i)} {\leq} \frac {\tau_ {m + 1} - \tau_ {m}}{T} + (\tau_ {m + 1} - \tau_ {m}) \mathbb {E} \left[ \sum_ {\pi \in \Psi} Q _ {m} (\pi) \left(2 \mathrm{Reg} _ {m} (\pi) + \frac {3 3 N (K + 1) ^ {4}}{\gamma_ {m}}\right) \Bigg | \mathrm{Event1holds} \right]
$$

$$
\leq \frac {\tau_ {m + 1} - \tau_ {m}}{T} + (\tau_ {m + 1} - \tau_ {m}) \cdot \frac {3 5 N (K + 1) ^ {4}}{\gamma_ {m}} \tag {usingEq.(24)}
$$

$$
= \mathcal {O} \left(\frac {\tau_ {m + 1} - \tau_ {m}}{T} + 2 ^ {m - 1} K ^ {2} \sqrt {N \mathbf {E r r} _ {\mathrm{log}} (2 ^ {m - 2} , 1 / T ^ {2} , \mathcal {F})}\right),
$$

where (i) uses Lemma 22. Taking summation over $m = 2,3,\ldots ,\lceil \log_2T + 1\rceil$ , we conclude that

$$
\mathbf {R e g} _ {\mathrm{MNL}} = \mathcal {O} \left(\sum_ {m = 1} ^ {\lceil \log_ {2} T \rceil} 2 ^ {m} K ^ {2} \sqrt {N \mathbf {E r r} _ {\log} (2 ^ {m - 1} , 1 / T ^ {2} , \mathcal {F})}\right).
$$

# Appendix C. Omitted Details in Section 4.1

In this section, we show omitted details in Section 4.1.

# C.1. Online Regression Oracle

We first show that there exists efficient online regression oracle for the finite class and the linear class.

Lemma 9 For the finite class and the linear class discussed in Lemma 1, the following concrete oracles satisfy Assumption 3:

- (Finite class) Hedge (Freund and Schapire, 1997) with $\mathbf{Reg}_{\log}(T, \mathcal{F}) = \mathcal{O}(\sqrt{T \log |\mathcal{F}|} \log \frac{K}{\beta})$ ;   
- (Linear class) Online Gradient Descent (Zinkevich, 2003) with $\mathbf{Reg}_{\log}(T, \mathcal{F}) = \mathcal{O}(B\sqrt{T})$ .

Proof We first consider the finite function class. Since for any $S \in S$ , $i \in S \cup \{0\}$ , and $x \in \mathcal{X}$ , we have $f_i(x) \geq \beta$ , we know that $\ell_{\log}(\mu(S, f(x)), i) \leq \log \frac{K + 1}{\beta}$ . Therefore, Hedge (Freund and Schapire, 1997) guarantees that $\mathbf{Reg}_{\log}(T, \mathcal{F}) = \mathcal{O}\left(\log \frac{K}{\beta} \sqrt{T \log |\mathcal{F}|}\right)$ .

For the linear class, we first prove that given $S \in \mathcal{S}$ , $i \in S \cup \{0\}$ and $x \in \mathbb{R}^{d \times N}$ , for any $f_{\theta} \in \mathcal{F}$ , $\ell_{\log}(\mu(S, f_{\theta}(x)), i)$ is convex in $\theta$ . Specifically, for $u \in \mathbb{R}^d$ , $h(u) = \log (\sum_{i=1}^d e^{u_i})$ is convex in $u$ since for any $\alpha \in \mathbb{R}^d$ ,

$$
\begin{array}{l} \alpha^ {\top} \nabla_ {u} ^ {2} h (u) \alpha = \alpha^ {\top} \left(\frac {1}{\mathbf {1} ^ {\top} u} \operatorname{diag} (u) - \frac {1}{(\mathbf {1} ^ {\top} u) ^ {2}} u u ^ {\top}\right) \alpha \\ = \frac {(\sum_ {k = 1} ^ {d} u _ {k} \alpha_ {k} ^ {2}) (\sum_ {k = 1} ^ {d} u _ {k}) - (\sum_ {k = 1} ^ {d} u _ {k} \alpha_ {k}) ^ {2}}{(\mathbf {1} ^ {\top} u) ^ {2}} \geq 0, \\ \end{array}
$$

where the last inequality is due to Cauchy-Schwarz inequality. Define $x_{0} = 0 \in R^{d}$ to be the d-dimensional all-zero vector. Then, we know that $\ell_{\log}(\mu(S, f_{\theta}(x), i)) = \log \left(e^{\theta^{\top} x_{0}} + \sum_{j \in S} e^{\theta^{\top} x_{j} - B}\right) - (\theta^{\top} x_{i} - B) \cdot \mathbb{1}\{i \neq 0\}$ is convex in $\theta$ . Moreover, direct calculation shows that

$$
\| \nabla_ {\theta} \ell_ {\log} (\mu (S, f _ {\theta} (x)), i) \| _ {2} = \left\| \frac {\sum_ {j \in S} e ^ {\theta^ {\top} x _ {j} - B} \cdot x _ {j}}{1 + \sum_ {j \in S} e ^ {\theta^ {\top} x _ {j} - B}} - x _ {i} \cdot \mathbb {1} \{i \neq 0 \} \right\| _ {2} \leq 2.
$$

Therefore, Online Gradient Descent (Zinkevich, 2003) guarantees that $\mathbf{Reg}_{\log}(T,\mathcal{F}) = \mathcal{O}(B\sqrt{T})$ , since $\| \theta \|_2 \leq B$ .

For completeness, we restate and prove Lemma 10, which is extended from the analysis in (Foster et al., 2021, 2022).

Lemma 10 Under Assumption 1 and Assumption 3, Algorithm 2 (with any $q_{t}$ ) ensures

$$
\mathbf {R e g} _ {\mathrm{MNL}} \leq \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \mathsf {d e c} _ {\gamma} (q _ {t}; f _ {t} (x _ {t}), r _ {t}) \right] + 2 \gamma \mathbf {R e g} _ {\log} (T, \mathcal {F})
$$

for any $\gamma > 0$ , where $\operatorname{dec}_{\gamma}(q; v, r)$ is the Decision-Estimation Coefficient (DEC) defined as

$$
\max _ {v ^ {\star} \in [ 0, 1 ] ^ {N}} \max _ {S ^ {\star} \in \mathcal {S}} \left\{R (S ^ {\star}, v ^ {\star}, r) - \mathbb {E} _ {S \sim q} [ R (S, v ^ {\star}, r) ] - \gamma \mathbb {E} _ {S \sim q} \left[ \| \mu (S, v) - \mu (S, v ^ {\star}) \| _ {2} ^ {2} \right] \right\}. \tag {8}
$$

Proof Following the regret decomposition in (Foster et al., 2021, 2022), we decompose $Reg_{MNL}$ as follows:

RegMNL

$$
= \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \max _ {S ^ {\star} \in \mathcal {S}} R (S, f ^ {\star} (x _ {t}), r _ {t}) - \sum_ {t = 1} ^ {T} q _ {t} (S) R (S, f ^ {\star} (x _ {t}), r _ {t}) \right]
$$

$$
= \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f ^ {\star} (x _ {t}), r _ {t}) - \sum_ {t = 1} ^ {T} q _ {t} (S) R (S, f ^ {\star} (x _ {t}), r _ {t}) \right.
$$

$$
\left. - \gamma \sum_ {S \in \mathcal {S}} q _ {t} (S) \| \mu (S, f _ {t} (x _ {t})) - \mu (S, f ^ {\star} (x _ {t})) \| _ {2} ^ {2} \right]
$$

$$
+ \gamma \mathbb {E} \left[ \sum_ {S \in \mathcal {S}} q _ {t} (S) \| \mu (S, f _ {t} (x _ {t})) - \mu (S, f ^ {\star} (x _ {t})) \| _ {2} ^ {2} \right]
$$

$$
\leq \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \max _ {S ^ {\star} \in \mathcal {S}, v ^ {\star} \in [ 0, 1 ] ^ {N}} \left\{R (S ^ {\star}, v ^ {\star}, r _ {t}) - \sum_ {t = 1} ^ {T} q _ {t} (S) R (S, v ^ {\star}, r _ {t}) - \right. \right.
$$

$$
\left. \gamma \sum_ {S \in \mathcal {S}} q _ {t} (S) \| \mu (S, f _ {t} (x _ {t})) - \mu (S, v ^ {\star}) \| _ {2} ^ {2} \right\}
$$

$$
+ \gamma \cdot \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \| \mu (S _ {t}, f _ {t} (x _ {t})) - \mu (S _ {t}, f ^ {\star} (x _ {t})) \| _ {2} ^ {2} \right]
$$

$$
= \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \mathrm{dec} _ {\gamma} (q _ {t}; f _ {t} (x _ {t}), r _ {t}) \right] + \gamma \cdot \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \| \mu (S _ {t}, f _ {t} (x _ {t})) - \mu (S _ {t}, f ^ {\star} (x _ {t})) \| _ {2} ^ {2} \right], \tag {25}
$$

where the last equality is by the definition of $\mathrm{dec}_{\gamma}(q_t; f_t(x_t), r_t)$ . According to Lemma 3, we know that

$$
\begin{array}{l} \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \| \mu (S _ {t}, f _ {t} (x _ {t})) - \mu (S _ {t}, f ^ {\star} (x _ {t})) \| _ {2} ^ {2} \right] \\ \leq 2 \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \ell_ {\log} (\mu (S _ {t}, f _ {t} (x _ {t})), i _ {t}) - \sum_ {t = 1} ^ {T} \ell_ {\log} (\mu (S _ {t}, f ^ {\star} (x _ {t})), i _ {t}) \right] \leq 2 \mathbf {R e g} _ {\log} (T, \mathcal {F}). \tag {26} \\ \end{array}
$$

Combining Eq. (25) and Eq. (26) finishes the proof.

# C.2. Proof of Theorem 11

Next, we prove Theorem 11, which shows that similar to the stochastic environment, a simple but efficient $\varepsilon$ -greedy strategy achieves $\mathcal{O}\left(T^{2/3}(NK\mathbf{Reg}_{\log}(T,\mathcal{F}))^{1/3}\right)$ expected regret.

Theorem 11 The strategy defined in Eq. (9) guarantees $\mathsf{dec}_{\gamma}(q_t; f_t(x_t), r_t) = \mathcal{O}(\frac{NK}{\gamma\varepsilon} + \varepsilon)$ . Consequently, under Assumption 1 and Assumption 3, Algorithm 2 with $q_t$ calculated via Eq. (9) and the optimal choice of $\varepsilon$ and $\gamma$ ensures $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left((NK\mathbf{Reg}_{\log}(T, \mathcal{F}))^{\frac{1}{3}} T^{\frac{2}{3}}\right)$ .

Proof We first prove that $q_{t}$ defined in Eq. (9) guarantees $\operatorname{dec}_{\gamma}(q_{t}; f_{t}(x_{t}), r_{t}) \leq \mathcal{O}\left(\frac{NK}{\gamma\varepsilon} + \varepsilon\right)$ . Specifically, for any $S^{\star} \in S$ and $v^{\star} \in [0,1]^{N}$ , we know that

$$
\begin{array}{l} R (S ^ {\star}, v ^ {\star}, r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, v ^ {\star}, r _ {t}) - \gamma \sum_ {S \in \mathcal {S}} q _ {t} (S) \| \mu (S, f _ {t} (x _ {t})) - \mu (S, f ^ {\star} (x _ {t})) \| _ {2} ^ {2} \\ \stackrel {(i)} {\leq} \sum_ {i \in S ^ {\star}} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | + \sum_ {S \in \mathcal {S}} q _ {t} (S) \sum_ {i \in S} | \mu_ {i} (S, v _ {i} ^ {\star}) - \mu_ {i} (S, f _ {t} (x _ {t})) | \\ + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, f _ {t} (x _ {t}), r _ {t}) - \gamma \sum_ {S \in \mathcal {S}} q _ {t} (S) \| \mu (S, f _ {t} (x _ {t})) - \mu (S, v ^ {\star}) \| _ {2} ^ {2} \\ \stackrel {(i i)} {\leq} \sum_ {i \in S ^ {\star}} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | + \frac {2 K}{\gamma} \\ + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, f _ {t} (x _ {t}), r _ {t}) - \frac {\gamma}{2} \sum_ {S \in \mathcal {S}} q _ {t} (S) \| \mu (S, f _ {t} (x _ {t})) - \mu (S, v ^ {\star}) \| _ {2} ^ {2} \\ \stackrel {(i i i)} {\leq} \sum_ {i \in S ^ {\star}} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | + \frac {2 K}{\gamma} + \varepsilon + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \max _ {S \in \mathcal {S}} R (S, f _ {t} (x _ {t}), r _ {t}) \\ - \frac {\gamma \varepsilon}{2 N} \sum_ {i = 1} ^ {N} \| \mu (\{i \}, f _ {t} (x _ {t})) - \mu (\{i \}, v ^ {\star}) \| _ {2} ^ {2} \\ \stackrel {(i v)} {\leq} \sum_ {i \in S ^ {\star}} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | + \frac {2 K}{\gamma} + \varepsilon + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \max _ {S \in \mathcal {S}} R (S, f _ {t} (x _ {t}), r _ {t}) \\ - \frac {\gamma \varepsilon}{6 4 N} \sum_ {i = 1} ^ {N} (f _ {t, i} (x _ {t}) - v _ {i} ^ {\star}) ^ {2} \\ \stackrel {(v)} {\leq} \frac {1 6 N K}{\gamma \varepsilon} + \frac {2 K}{\gamma} + \varepsilon \\ \leq \mathcal {O} \left(\frac {N K}{\gamma \varepsilon} + \varepsilon\right), \\ \end{array}
$$

where (i) uses Lemma 18, (ii) is due to AM-GM inequality and $|S| \leq K$ , (iii) is according to the construction of $q_t$ and $R(S, v, r) \in [0,1]$ , (iv) uses Lemma 19 with $d = 1$ , and (v) is uses AM-GM inequality and the fact that $|S^{\star}| \leq K$ . Taking maximum over all $S^{\star} \in S$ and $v^{\star} \in [0,1]^N$ proves that $\mathrm{dec}_{\gamma}(q_t; f_t(x_t), r_t) \leq \mathcal{O}\left(\frac{NK}{\gamma\varepsilon} + \varepsilon\right)$ .

Combining the above result with Lemma 10, we know that

$$
\mathbf {R e g} _ {\mathrm{MNL}} = \mathcal {O} \left(\frac {N K T}{\gamma \varepsilon} + \varepsilon T + \gamma \mathbf {R e g} _ {\log} (T, \mathcal {F})\right).
$$

Picking $\gamma$ and $\varepsilon$ optimally finishes the proof.

# C.3. Proof of Theorem 13 and Theorem 14

In this section, we restate and prove Theorem 13, which proves that $q_{t}$ calculated via Eq. (10) guarantees that $\mathrm{dec}_{\gamma}(q_t; f_t(x_t), r_t) \leq \mathcal{O}\left(\frac{NK^4}{\gamma}\right)$ .

Theorem 13 The following distribution

$$
q _ {t} = \underset {q \in \Delta (\mathcal {S})} {\operatorname{argmax}} \mathbb {E} _ {S \sim q} [ R (S, f _ {t} (x _ {t}), r _ {t}) ] - \frac {(K + 1) ^ {4}}{\gamma} \sum_ {i = 1} ^ {N} \log \frac {1}{w _ {i} (q)}, \tag {10}
$$

satisfies $\mathsf{dec}_{\gamma}(q_t, f_t(x_t), r_t) \leq \mathcal{O}\left(\frac{NK^4}{\gamma}\right)$ .

Proof Since the construction of $q_{t}$ is the same as Eq. (5) with $f_{m}$ replaced by $f_{t}$ and $\gamma_{m}$ replaced by $\gamma$ , according to Lemma 6, we know that $q_{t}$ satisfies that

$$
\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) \cdot R (S, f _ {t} (x _ {t}), r _ {t}) \leq \frac {N (K + 1) ^ {4}}{\gamma}, \tag {27}
$$

$$
\forall S \in \mathcal {S}, \sum_ {i \in S} \frac {1}{w _ {i} (q)} \leq N + \frac {\gamma}{(K + 1) ^ {4}} \left(\max _ {S ^ {\star} \in \mathcal {S}} R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - R (S, f _ {t} (x _ {t}), r _ {t})\right). \tag {28}
$$

Using Eq. (27) and Eq. (28), we know that for any $S^{\star} \in S$ and $v^{\star} \in [0,1]^{N}$ ,

$$
\begin{array}{l} R (S ^ {\star}, v ^ {\star}, r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, v ^ {\star}, r _ {t}) - \gamma \sum_ {S \in \mathcal {S}} q _ {t} (S) \| \mu (S, f _ {t} (x _ {t})) - \mu (S, f ^ {\star} (x _ {t})) \| _ {2} ^ {2} \\ \leq \sum_ {i \in S ^ {\star}} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | + \sum_ {S \in \mathcal {S}} q _ {t} (S) \sum_ {i \in S} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | \tag {accordingtoLemma18} \\ + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, f _ {t} (x _ {t}), r _ {t}) - \gamma \sum_ {S \in \mathcal {S}} q _ {t} (S) \| \mu (S, f _ {t} (x _ {t})) - \mu (S, v ^ {\star}) \| _ {2} ^ {2} \\ \leq \sum_ {i \in S ^ {\star}} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | + \sum_ {i = 1} ^ {N} w _ {i} (q _ {t}) \cdot | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | \quad (\text { by   definition   of } w _ {i} (q)) \\ + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, f _ {t} (x _ {t}), r _ {t}) - \gamma \sum_ {S \in \mathcal {S}} q _ {t} (S) \| \mu (S, f _ {t} (x _ {t})) - \mu (S, v ^ {\star}) \| _ {2} ^ {2}. \\ \leq \sum_ {i \in S ^ {\star}} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | + \sum_ {i = 1} ^ {N} w _ {i} (q _ {t}) \cdot | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | \\ + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, f _ {t} (x _ {t}), r _ {t}) - \frac {\gamma}{2 (K + 1) ^ {4}} \sum_ {S \in \mathcal {S}} q _ {t} (S) \sum_ {i \in S} (v _ {i} ^ {\star} - f _ {t, i} (x _ {t})) ^ {2} \\ \end{array}
$$

$$
\begin{array}{l} \leq \sum_ {i \in S ^ {\star}} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | + \sum_ {i = 1} ^ {N} w _ {i} (q _ {t}) \cdot | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | \\ + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, f _ {t} (x _ {t}), r _ {t}) - \frac {\gamma}{2 (K + 1) ^ {4}} \sum_ {S \in \mathcal {S}} q _ {t} (S) \sum_ {i \in S} (v _ {i} ^ {\star} - f _ {t, i} (x _ {t})) ^ {2} \\ \end{array}
$$

(according to Lemma 3)

$$
\begin{array}{l} = \sum_ {i \in S ^ {\star}} | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | + \sum_ {i = 1} ^ {N} w _ {i} (q _ {t}) \cdot | v _ {i} ^ {\star} - f _ {t, i} (x _ {t}) | \\ + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, f _ {t} (x _ {t}), r _ {t}) - \frac {\gamma}{2 (K + 1) ^ {4}} \sum_ {i = 1} ^ {N} w _ {i} (q _ {t}) (v _ {i} ^ {\star} - f _ {t, i} (x _ {t})) ^ {2} \\ \leq \frac {N (K + 1) ^ {4}}{\gamma} + \sum_ {i \in S ^ {\star}} \frac {(K + 1) ^ {4}}{\gamma w _ {i} (q _ {t})} + R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, f _ {t} (x _ {t}), r _ {t}) \\ = \frac {N (K + 1) ^ {4}}{\gamma} + \sum_ {i \in S ^ {\star}} \frac {(K + 1) ^ {4}}{\gamma w _ {i} (q _ {t})} - \left(\max _ {S _ {0} \in \mathcal {S}} R (S _ {0}, f _ {t} (x _ {t}), r _ {t}) - R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t})\right) \\ + \max _ {S _ {0} \in \mathcal {S}} R (S _ {0}, f _ {t} (x _ {t}), r _ {t}) - \sum_ {S \in \mathcal {S}} q _ {t} (S) R (S, f _ {t} (x _ {t}), r _ {t}) \\ \leq \frac {N (K + 1) ^ {4}}{\gamma} + \frac {(K + 1) ^ {4}}{\gamma} \left(N + \frac {\gamma}{(K + 1) ^ {4}} \left(\max _ {S _ {0} \in \mathcal {S}} R \left(S _ {0}, f _ {t} \left(x _ {t}\right), r _ {t}\right) - R \left(S ^ {\star}, f _ {t} \left(x _ {t}\right), r _ {t}\right)\right)\right) \\ \left. - \left(\max _ {S _ {0} \in \mathcal {S}} R (S _ {0}, f _ {t} (x _ {t}), r _ {t}) - R (S ^ {\star}, f _ {t} (x _ {t}), r _ {t})\right) + \frac {N (K + 1) ^ {4}}{\gamma} \right. \\ = \frac {3 N (K + 1) ^ {4}}{\gamma}. \\ \end{array}
$$

(AM-GM inequality)

(according to Eq. (27) and Eq. (28))

Taking maximum over all $S^{\star} \in S$ and $v^{\star} \in [0,1]^{N}$ finishes the proof.

Combining Lemma 10 and Theorem 13, we are able to prove Theorem 14.

Theorem 14 Under Assumption 1 and Assumption 3, Algorithm 2 with $q_{t}$ calculated via Eq. (10) and the optimal choice of $\gamma$ ensures $\mathbf{Reg}_{\mathsf{MNL}} = \mathcal{O}\left(K^{2}\sqrt{NT\mathbf{Reg}_{\log}(T,\mathcal{F})}\right)$ .

Proof Combining Lemma 10 and Theorem 13, we know that Algorithm 2 with $q_{t}$ calculated via Eq. (10) satisfies that $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left(\frac{NK^4}{\gamma} + \gamma \mathbf{Reg}_{\log}(T, \mathcal{F})\right)$ . Picking $\gamma = K^2 \sqrt{\frac{NT}{\mathbf{Reg}_{\log}(T, \mathcal{F})}}$ finishes the proof.

# Appendix D. Omitted Details in Section 4.2

In this section, we show omitted details in Section 4.2. Specifically, we restate and prove Theorem 16 as follows.

Theorem 16 Under Assumption 1, Algorithm 3 with learning rate $\eta \leq 1$ ensures $\mathbf{Reg}_{\mathsf{MNL}} \leq 12\eta NK(K + 1)^4 T + 4\eta T + \frac{Z_T}{\eta}$ , where $Z_T = -\mathbb{E}[\log \mathbb{E}_{f \sim p_1}[\exp(-\eta \sum_{t=1}^T (\widehat{\ell}_{t,f} - \widehat{\ell}_{t,f^{\star}}))]]$ .

Proof First, we decompose the regret as follows:

$$
\mathbf {R e g} _ {\mathrm{MNL}} = \mathbb {E} \left[ \sum_ {t = 1} ^ {T} (\max _ {S \in \mathcal {S}} R (S, f ^ {\star} (x _ {t}), r _ {t}) - R (S _ {t}, f ^ {\star} (x _ {t}), r _ {t})) \right]
$$

Algorithm 3 Feel-Good Thompson Sampling for Contextual MNL bandits   
Input: a learning rate $\eta > 0$ .  
Initialize $p_1 \in \Delta(\mathcal{F})$ to be the uniform distribution over $\mathcal{F}$ .  
for $t = 1, 2, \ldots, T$ do  
    Sample a value function $f_t$ from $p_t$ .  
    Receive context $x_t$ and reward vector $r_t \in [0, 1]^N$ .  
    Select $S_t = \arg\max_{S \in S} R(S, f_t(x), r_t)$ and receive feedback $i_t \in S_t \cup \{0\}$ .  
    Define the loss estimator $\widehat{\ell}_{t,f}$ for each $f \in \mathcal{F}$ as $\widehat{\ell}_{t,f} = \frac{1}{8\eta K} \sum_{i \in S_t} (\mu_i(S_t, f(x_t)) - \mathbb{1}[i = i_t])^2 - \max_{S \in S} R(S, f(x_t), r_t)$ . (29)  
    Update $p_{t+1,f} \propto p_{t,f} \cdot \exp(-\eta \widehat{\ell}_{t,f})$ .

$$
\begin{array}{l} = \mathbb {E} \left[ \sum_ {t = 1} ^ {T} (R (S _ {t}, f _ {t} (x _ {t}), r _ {t}) - R (S _ {t}, f ^ {\star} (x _ {t}), r _ {t})) \right] \\ - \mathbb {E} \left[ \sum_ {t = 1} ^ {T} (R (S _ {t}, f _ {t} (x _ {t}), r _ {t}) - \max _ {S \in \mathcal {S}} R (S, f ^ {\star} (x _ {t}), r _ {t})) \right] \\ \stackrel {(i)} {=} \mathbb {E} \left[ \sum_ {t = 1} ^ {T} (R (S _ {t}, f _ {t} (x _ {t}), r _ {t}) - R (S _ {t}, f ^ {\star} (x _ {t}), r _ {t})) \right] \\ - \mathbb {E} \left[ \sum_ {t = 1} ^ {T} (\underbrace {\max _ {S \in \mathcal {S}} R (S , f _ {t} (x _ {t}) , r _ {t}) - \max _ {S \in \mathcal {S}} R (S , f ^ {\star} (x _ {t}) , r _ {t}))} _ {\triangleq \mathrm{FG} _ {t}}) \right] \\ \end{array}
$$

$$
\stackrel {(i i)} {\leq} \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \sum_ {i \in S _ {t}} | f _ {t, i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t}) | \right] - \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \mathrm{FG} _ {t} \right]. \tag {30}
$$

where (i) is because $S_{t} = \arg\max_{S \in S} R(S, f_{t}(x_{t}), r_{t})$ according to Algorithm 3 and (ii) is using Lemma 18. Here, “Feel-Good” term $FG_{t}$ measures the difference between the expected reward of the best subset given the value predictor $f_{t}$ and that of the true value predictor $f^{\star}$ .

Next, we analyze the first term $\sum_{t=1}^{T} \sum_{i \in S_t} |f_{t,i}(x_t) - f_i^\star(x_t)|$ . Given any context $x \in \mathcal{X}$ , reward vector $r_t \in [0,1]^N$ , and a value predictor $f \in \mathcal{F}$ , let $S(f(x), r) = \operatorname{argmax}_{S \in \mathcal{S}} R(S, f(x), r)$ . According to Algorithm 3, we have $S_t = S(\theta_t, x_t)$ . With a slight abuse of notation, for distribution $p_t$ over $\mathcal{F}$ , let $w_{t,i} = \mathbb{E}_{f \sim p_t}[\mathbb{1}\{i \in S(f(x_t), r_t)\}]$ be the probability that item $i$ is included in the selected set at round $t$ . Let $q_t \in \Delta(\mathcal{S})$ be the distribution over $\mathcal{S}$ induced by $p_t$ , meaning that $q_t(S) = \mathbb{E}_{f \sim p_t}[\mathbb{1}\{S(f(x_t), r_t) = S\}]$ . Then, for each $i \in [N]$ , for any $\mu > 0$ ,

$$
\begin{array}{l} \mathbb {E} _ {f \sim p _ {t}} \left[ | f _ {i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t}) | \cdot \mathbb {1} \{i \in S (f (x _ {t}), r _ {t}) \} \right] \\ \leq \mathbb {E} _ {f \sim p _ {t}} \left[ \frac {\mathbb {1} \{i \in S (f (x _ {t}) , r _ {t}) \}}{4 \mu w _ {t , i}} + w _ {t, i} (f _ {i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t})) ^ {2} \right] \quad (\text {AM - GM inequality}) \\ \end{array}
$$

$$
= \frac {1}{4 \mu} + \mu w _ {t, i} \mathbb {E} _ {f \sim p _ {t}} \left[ (f _ {i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t})) ^ {2} \right]. \tag {31}
$$

Taking a summation over all $i \in [N]$ , we know that for any $\mu > 0$ ,

$$
\begin{array}{l} \mathbb {E} \left[ \sum_ {i \in S _ {t}} | f _ {t, i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t}) | \right] \\ = \mathbb {E} \left[ \sum_ {i = 1} ^ {N} \left| f _ {t, i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t}) \right| \cdot \mathbb {1} \{i \in S (f _ {t} (x _ {t}), r _ {t}) \} \right] \\ = \mathbb {E} \left[ \sum_ {i = 1} ^ {N} \mathbb {E} _ {f \sim p _ {t}} \left[ | f _ {i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t}) | \cdot \mathbb {1} \{i \in S (f (x _ {t}), r _ {t}) \} \right] \right] \\ \stackrel {(i)} {\leq} \frac {N}{4 \mu} + \mu \mathbb {E} \left[ w _ {t, i} \mathbb {E} _ {f \sim p _ {t}} \left[ \sum_ {i = 1} ^ {N} (f _ {i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t})) ^ {2} \right] \right] \\ \end{array}
$$

$$
\stackrel {(i i)} {=} \frac {N}{4 \mu} + \mu \mathbb {E} _ {S _ {t} \sim q _ {t}} \mathbb {E} _ {f \sim p _ {t}} \left[ \sum_ {i \in S _ {t}} (f _ {i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t})) ^ {2} \right], \tag {32}
$$

where $(i)$ uses Eq. (31) and $(ii)$ is by definition of $w_{t,i}$ and $q_{t}$ .

Let $\mathrm{LS}_t = \sum_{i\in S_t}(f_i(x_t) - f_i^\star (x_t))^2$ ("Least Squares"). Combining Eq. (30) with Eq. (32), we know that

$$
\mathbf {R e g} _ {\mathrm{MNL}} \leq \frac {N T}{4 \mu} + \mu \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \mathbb {E} _ {S _ {t} \sim q _ {t}} \mathbb {E} _ {f \sim p _ {t}} [ \mathrm{LS} _ {t} ] \right] - \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \mathrm{FG} _ {t} \right]. \tag {33}
$$

To bound the last two terms in Eq. (33), using Lemma 19 and the fact that $i_t$ is a drawn from the distribution $\mu(S_t, f^{\star}(x_t), r_t)$ and, we show in Lemma 23 that

$$
\frac {1}{4 8 \eta K (K + 1) ^ {4}} \mathbb {E} _ {f \sim p _ {t}} [ \mathrm{LS} _ {t} ] - \mathbb {E} _ {f _ {t} \sim q _ {t}} [ \mathrm{FG} _ {t} ] \leq - \frac {1}{\eta} \log \mathbb {E} _ {i _ {t} | x _ {t}, S _ {t}} \mathbb {E} _ {f \sim p _ {t}} \left[ \exp (- \eta (\widehat {\ell} _ {t, f} - \widehat {\ell} _ {t, f ^ {\star}})) \right] + 4 \eta . \tag {34}
$$

Therefore, picking $\mu = \frac{1}{48\eta K(K + 1)^4}$ and combining Eq. (33) and Eq. (34), we know that

Reg $_{MNL}$

$$
\leq 1 2 \eta N K (K + 1) ^ {4} T + 4 \eta T - \frac {1}{\eta} \mathbb {E} \left[ \sum_ {t = 1} ^ {T} \log \mathbb {E} _ {i _ {t} | x _ {t}, S _ {t}} \mathbb {E} _ {f \sim p _ {t}} \left[ \exp \left(- \eta (\widehat {\ell} _ {t, f} - \widehat {\ell} _ {t, f ^ {*}})\right) \right] \right] \tag {35}
$$

To bound the last term in Eq. (35), we use the exponential weight update dynamic of $p_t$ . Following a classic analysis of exponential weight update, we show in Lemma 24 that

$$
- \mathbb {E} \left[ \log \mathbb {E} _ {i _ {t} | x _ {t}, S _ {t}} \mathbb {E} _ {f \sim p _ {t}} \left[ \exp (- \eta (\widehat {\ell} _ {t, f} - \widehat {\ell} _ {t, f ^ {*}})) \right] \right] \leq Z _ {t} - Z _ {t - 1}, \tag {36}
$$

where $Z_{t} \triangleq -\mathbb{E}\left[\log \mathbb{E}_{f \sim p_1}\left[\exp \left(-\eta \sum_{\tau=1}^{t} \left(\widehat{\ell}_{t,f} - \widehat{\ell}_{t,f^*}\right)\right)\right]\right]$ . Combining Eq. (35) and Eq. (36), we arrive at

$$
\mathbf {R e g} _ {\mathrm{MNL}} \leq 1 2 \eta N K (K + 1) ^ {4} T + 4 \eta T + \frac {1}{\eta} \sum_ {t = 1} ^ {T} (Z _ {t} - Z _ {t - 1})
$$

$$
\leq 1 2 \eta N K (K + 1) ^ {4} T + 4 \eta T + \frac {Z _ {T}}{\eta},
$$

where the last inequality uses the fact that $Z_{0}=0$ .

Lemma 23 Suppose that $\eta \leq 1$ . For any distribution $p_{t}$ over F, we have

$$
\frac {1}{4 8 \eta K (K + 1) ^ {4}} \mathbb {E} _ {f \sim p _ {t}} [ \mathrm{LS} _ {t} ] - \mathbb {E} _ {f _ {t} \sim q _ {t}} [ \mathrm{FG} _ {t} ] \leq - \frac {1}{\eta} \log \mathbb {E} _ {i _ {t} | x _ {t}, S _ {t}} \mathbb {E} _ {f \sim p _ {t}} \left[ \exp (- \eta (\widehat {\ell} _ {t, f} - \widehat {\ell} _ {t, f ^ {\star}})) \right] + 4 \eta ,
$$

where $\mathrm{LS}_t$ and $\mathrm{FG}_t$ are defined in the proof of Theorem 16, and $\widehat{\ell}_{t,f}$ is defined in Eq. (29).

Proof For notational convenience, define $c_{t,i} = \mathbb{1}\{i = i_t\}$ for all $i \in [N]$ . Let $\varepsilon_{t,i} = c_{t,i} - \mu_i(S_t, f^\star(x_t))$ for all $i \in S_t$ . Consider the term $-\eta\left(\widehat{\ell}_{t,f} - \widehat{\ell}_{t,f^{\star}}\right)$ for an arbitrary $f \in \mathcal{F}$ .

$$
\begin{array}{l} - \eta (\widehat {\ell} _ {t, f} - \widehat {\ell} _ {t, f ^ {\star}}) \\ = - \frac {1}{8 K} \sum_ {i \in S _ {t}} (\mu_ {i} (S _ {t}, f (x _ {t})) - c _ {t, i}) ^ {2} + \frac {1}{8 K} \sum_ {i \in S _ {t}} (\mu_ {i} (S _ {t}, f ^ {\star} (x _ {t})) - c _ {t, i}) ^ {2} \\ + \eta \cdot \max _ {S \in \mathcal {S}} R (S, f (x _ {t}), r _ {t}) - \eta \cdot \max _ {S \in \mathcal {S}} R (S, f ^ {\star} (x _ {t}), r _ {t}) \\ = - \frac {1}{8 K} \sum_ {i \in S _ {t}} (\mu_ {i} (S _ {t}, f (x _ {t})) - \mu_ {i} (S _ {t}, f ^ {\star} (x _ {t}))) (2 c _ {t, i} - \mu_ {i} (S _ {t}, f (x _ {t})) - \mu_ {i} (S _ {t}, f ^ {\star} (x _ {t}))) \\ + \eta \cdot \max _ {S \in \mathcal {S}} R (S, f (x _ {t}), r _ {t}) - \eta \cdot \max _ {S \in \mathcal {S}} R (S, f ^ {\star} (x _ {t}), r _ {t}) \\ = - \frac {1}{8 K} \underbrace {\sum_ {i \in S _ {t}} (\mu_ {i} (S _ {t} , f (x _ {t})) - \mu_ {i} (S _ {t} , f ^ {\star} (x _ {t}))) (\mu_ {i} (S _ {t} , f ^ {\star} (x _ {t})) - \mu_ {i} (S _ {t} , f (x _ {t})) + 2 \varepsilon_ {t , i})} _ {\widehat {\mathrm{LS}} _ {t}} \\ + \eta \mathrm{FG} _ {t} (f), \\ \end{array}
$$

where we define the first term as $\widehat{\mathrm{LS}}_t$ , which we will show later how this term is related to $\mathrm{LS}_t$ , and the second term $\mathrm{FG}_t(f) = \max_{S \in S} R(S, f(x_t), r_t) - \max_{S \in S} R(S, f^*(x_t), r_t)$ (so $\mathrm{FG}_t = \mathrm{FG}_t(f_t)$ ). Consider the log of the expectation of the exponent on both sides.

$$
\begin{array}{l} \log \mathbb {E} _ {f \sim p _ {t}} \mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \left[ \exp \left(- \eta \left(\widehat {\ell} _ {t, f} - \widehat {\ell} _ {t, f ^ {\star}}\right)\right) \right] \\ = \log \mathbb {E} _ {f \sim p _ {t}} \mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \left[ \exp \left(- \frac {1}{8 K} \widehat {\mathrm{LS}} _ {t} + \eta \mathrm{FG} _ {t} (f)\right) \right] \\ \leq \frac {1}{2} \log \mathbb {E} _ {f \sim p _ {t}} \left(\mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \left[ \exp \left(- \frac {1}{8 K} \widehat {\mathrm{LS}} _ {t}\right) \right] ^ {2}\right) + \frac {1}{2} \log \mathbb {E} _ {f \sim p _ {t}} \left[ \exp \left(2 \eta \mathrm{FG} _ {t} (f)\right) \right] \\ \leq \frac {1}{2} \log \mathbb {E} _ {f \sim p _ {t}} \left(\mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \left[ \exp \left(- \frac {1}{4 K} \widehat {\mathrm{LS}} _ {t}\right) \right]\right) + \frac {1}{2} \log \mathbb {E} _ {f \sim p _ {t}} \left[ \exp (2 \eta \mathrm{FG} _ {t} (f)) \right], \tag {37} \\ \end{array}
$$

where the first inequality is by Cauchy-Schwarz inequality and the second inequality is because $E[x]^{2} \leq E[x^{2}]$ . Next, we consider bounding each of the two terms. For the first term, since

$$
\left| \frac {1}{4 K} \sum_ {i \in S _ {t}} (\mu_ {i} (S _ {t}, f (x _ {t})) - \mu_ {i} (S _ {t}, f ^ {\star} (x _ {t}))) (\mu_ {i} (S _ {t}, f ^ {\star} (x _ {t})) - \mu_ {i} (S _ {t}, f (x _ {t})) + 2 \varepsilon_ {t, i}) \right|
$$

$$
\leq \frac {1}{4 K} \cdot 2 | S _ {t} | \leq \frac {1}{2},
$$

using the fact that $\exp (x)\leq 1 + x + \frac{2}{3} x^2$ when $x\leq \frac{1}{2}$ , we know that

$$
\mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \left[ \exp \left(- \frac {1}{4 K} \widehat {\mathrm{LS}} _ {t}\right) \right]
$$

$$
\leq 1 - \frac {1}{4 K} \mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \left[ \sum_ {i \in S _ {t}} (\mu_ {i} (S _ {t}, f (x _ {t})) - \mu_ {i} (S _ {t}, f ^ {\star} (x _ {t}))) (\mu_ {i} (S _ {t}, f ^ {\star} (x _ {t})) - \mu_ {i} (S _ {t}, f (x _ {t})) + 2 \varepsilon_ {t, i}) \right]
$$

$$
+ \frac {1}{2 4 K ^ {2}} \mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \left[ \left(\sum_ {i \in S _ {t}} (\mu_ {i} (S _ {t}, f (x _ {t})) - \mu_ {i} (S _ {t}, f ^ {\star} (x _ {t}))) (\mu_ {i} (S _ {t}, f ^ {\star} (x _ {t})) - \mu_ {i} (S _ {t}, f (x _ {t})) + 2 \varepsilon_ {t, i})\right) ^ {2} \right]
$$

$$
\stackrel {(i)} {\leq} 1 - \frac {1}{4 K} \| \mu (S _ {t}, f (x _ {t})) - \mu (S _ {t}, f ^ {\star} (x _ {t}) \| _ {2} ^ {2}
$$

$$
+ \frac {1}{2 4 K} \mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \left[ \left(\sum_ {i \in S _ {t}} (\mu_ {i} (S _ {t}, f (x _ {t})) - \mu_ {i} (S _ {t}, f ^ {\star} (x _ {t}))) ^ {2} (\mu_ {i} (S _ {t}, f ^ {\star} (x _ {t})) - \mu_ {i} (S _ {t}, f (x _ {t})) + 2 \varepsilon_ {t, i}) ^ {2}\right) \right]
$$

$$
\stackrel {(i i)} {\leq} 1 - \frac {1}{1 2 K} \| \mu (S _ {t}, f (x _ {t})) - \mu (S _ {t}, f ^ {\star} (x _ {t}) \| _ {2} ^ {2}
$$

$$
\stackrel {(i i i)} {\leq} 1 - \frac {1}{2 4 K (K + 1) ^ {4}} \sum_ {i \in S _ {t}} \left(f _ {i} (x _ {t}) - f _ {i} ^ {\star} (x _ {t})\right) ^ {2}
$$

$$
= 1 - \frac {1}{2 4 K (K + 1) ^ {4}} \mathrm{LS} _ {t},
$$

where (i) is due to Cauchy-Schwarz inequality, (ii) is because $\left|\mu_{i}(S_{t},f^{\star}(x_{t}))-\mu_{i}(S_{t},f(x_{t}))+2\varepsilon_{t,i}\right|\leq2$ , and (iii) is because Lemma 19. Further using the fact that $\log(1+x)\leq x$ for all $x\geq-1$ , we have

$$
\frac {1}{2} \log \mathbb {E} _ {f \sim p _ {t}} \left(\mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \left[ \exp \left(- \frac {1}{4 K} \widehat {\mathrm{LS}} _ {t}\right) \right]\right) \leq - \frac {1}{4 8 K (K + 1) ^ {4}} \mathrm{LS} _ {t}. \tag {38}
$$

Consider the second term in Eq. (37). Since $\eta \leq 1$ and $|\mathrm{FG}_t(f)| \leq 1$ , using $e^x \leq 1 + x + 2x^2$ for $x \leq 1$ , we know that

$$
\begin{array}{l} \frac {1}{2} \log \mathbb {E} _ {f \sim q _ {t}} [ \exp (2 \eta \mathrm{FG} _ {t} (f)) ] \leq \frac {1}{2} \log \left(1 + 2 \eta \mathbb {E} _ {f \sim q _ {t}} [ \mathrm{FG} _ {t} (f) ] + 2 (2 \eta) ^ {2}\right) \\ \leq \eta \mathbb {E} _ {f \sim q _ {t}} [ \mathrm{FG} _ {t} (f) ] + 4 \eta^ {2} \quad (\log (1 + x) \leq x) \\ = \eta \mathbb {E} _ {f _ {t} \sim q _ {t}} [ \mathrm{FG} _ {t} ] + 4 \eta^ {2}. \quad (f _ {t} \text {   is   drawn   from   } q _ {t}) \\ \end{array}
$$

Plugging the last bound and Eq. (38) into Eq. (37) and rearranging finishes the proof.

The next lemma follows the classic analysis of multiplicative weight update algorithm.

Lemma 24 Algorithm 3 guarantees that for each $t \in [T]$ ,

$$
- \mathbb {E} \left[ \mathbb {E} _ {S _ {t} \sim q _ {t}} \log \mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \mathbb {E} _ {f \sim p _ {t}} \left[ \exp (- \eta (\widehat {\ell_ {t , f}} - \widehat {\ell_ {t , f ^ {*}}})) \right] \right] \leq Z _ {t} - Z _ {t - 1},
$$

where $Z_{t} = -\mathbb{E}\left[\log \mathbb{E}_{f\sim p_1}\left[\exp \left(-\eta \sum_{\tau = 1}^{t}\left(\widehat{\ell}_{t,f} - \widehat{\ell}_{t,f^*}\right)\right)\right]\right]$ and $q_{t}\in \Delta (\mathcal{S})$ satisfies that $q_{t}(S) = \mathbb{E}_{f\sim p_{t}}[\mathbb{1}\{S = \mathrm{argmax}_{S^{\prime}\in \mathcal{S}}R(S^{\prime},f(x_{t}),r_{t})\} ]$ .

Proof Let $G_{t,f} \triangleq \exp \left(-\eta \sum_{\tau=1}^{t} \left(\widehat{\ell}_{t,f} - \widehat{\ell}_{t,f^*}\right)\right)$ . According to Algorithm 3, we know that

$$
p _ {t, f} = \frac {\exp \left(- \eta \sum_ {\tau = 1} ^ {t - 1} \widehat {\ell} _ {\tau , f}\right)}{\int_ {f ^ {\prime} \in \mathcal {F}} \exp \left(- \eta \sum_ {\tau = 1} ^ {t - 1} \widehat {\ell} _ {\tau , f ^ {\prime}}\right) d f ^ {\prime}} = \frac {G _ {t - 1 , f}}{\int_ {f ^ {\prime} \in \mathcal {F}} G _ {t - 1 , f ^ {\prime}} d f ^ {\prime}}.
$$

Then, according to the definition of $Z_{t}$ , we have

$$
\begin{array}{l} Z _ {t - 1} - Z _ {t} \\ = \mathbb {E} \left[ \log \frac {\int_ {f \in \mathcal {F}} G _ {t , f} d f}{\int_ {f \in \mathcal {F}} G _ {t - 1 , f} d f} \right] \\ = \mathbb {E} \left[ \log \frac {\int_ {f \in \mathcal {F}} G _ {t - 1 , f} \exp (- \eta (\widehat {\ell} _ {t , f} - \widehat {\ell} _ {t , f ^ {*}})) d f}{\int_ {f \in \mathcal {F}} G _ {t - 1 , f} d f} \right] \\ = \mathbb {E} \left[ \log \mathbb {E} _ {f \sim p _ {t}} \left[ \exp (- \eta (\widehat {\ell} _ {t, f} - \widehat {\ell} _ {t, f ^ {*}}) \right] \right] \\ \leq \mathbb {E} \left[ \mathbb {E} _ {S _ {t} \sim q _ {t}} \log \mathbb {E} _ {c _ {t} | x _ {t}, S _ {t}} \mathbb {E} _ {f \sim p _ {t}} \left[ \exp (- \eta (\widehat {\ell} _ {t, f} - \widehat {\ell} _ {t, f ^ {*}}) \right] \right], \\ \end{array}
$$

where the last inequality is due to Jensen's inequality. Rearranging the terms finishes the proof.

Next, we restate and prove Corollary 17.

Corollary 17 Under Assumption 1, Algorithm 3 with the optimal choice of $\eta$ ensures the following regret bounds for finite class and the linear class:

- (Finite class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left(K^{2.5}\sqrt{NT\log|\mathcal{F}|}\right)$ ;   
- (Linear class) $\mathbf{Reg}_{\mathrm{MNL}} = \mathcal{O}\left(K^{2.5}\sqrt{dNT\log(BT)}\right)$ .

Proof For a finite function class F, since $q_{1}$ is uniform, we have

$$
\begin{array}{l} Z _ {T} = - \mathbb {E} \left[ \log \sum_ {f \in \mathcal {F}} \frac {1}{| \mathcal {F} |} \exp \left(- \eta \sum_ {t = 1} ^ {T} \left(\widehat {\ell} _ {t, f} - \widehat {\ell} _ {t, f ^ {\star}}\right)\right) \right] \\ \leq - \mathbb {E} \left[ \log \frac {1}{| \mathcal {F} |} \exp \left(- \eta \sum_ {t = 1} ^ {T} \left(\widehat {\ell} _ {t, f ^ {\star}} - \widehat {\ell} _ {t, f ^ {\star}}\right)\right) \right] = \log | \mathcal {F} |. \\ \end{array}
$$

Combining with Theorem 16 and picking $\eta = \frac{1}{K^{2.5}}\sqrt{\frac{N\log|\mathcal{F}|}{T}}$ , we prove the first conclusion.

To prove our results for the linear class, we first show a more general results for parametrized Lipschitz function class. Suppose that $\mathcal{F}$ is a $d$ -dimensional parametrized function class defined as:

$$
\mathcal {F} = \left\{f _ {\theta}: \mathcal {X} \mapsto [ 0, 1 ] ^ {N}, \| \theta \| _ {2} \leq B, f _ {\theta , i} \text {   is   } \alpha \text {-Lipschitz   with   respect   to   } \| \cdot \| _ {2} \text {   for   all   } i \in [ N ] \right\}. \tag {39}
$$

Direct calculation shows that the linear function class we consider is an instance of Eq. (39) with $\alpha = 1$ . For function class satisfying Eq. (39), we aim to show that $Z_{T} = \mathcal{O}(K\eta + d\log(\alpha BT))$ . Specifically, we consider a small $\ell_{2}$ -ball around the true parameter $\theta^{\star}$ : $\Omega_{T} = \{\theta : \|\theta - \theta^{*}\|_{2} \leq \frac{1}{\alpha T}\}$ . Since F is $\alpha$ -Lipschitz with respect to $\|\cdot\|_{2}$ , we know that for any $x \in X$ , and any $i \in [N]$ ,

$$
\left| f _ {\theta , i} (x) - f _ {\theta^ {*}, i} (x) \right| \leq \frac {1}{T}. \tag {40}
$$

Therefore, for any $\theta \in \Omega_T$ ,

$$
\begin{array}{l} - \eta (\widehat {\ell} _ {t, f _ {\theta}} - \widehat {\ell} _ {t, f _ {\theta^ {*}}}) \\ = - \frac {1}{8 K} \sum_ {i \in S _ {t}} (f _ {\theta , i} (x _ {t}) - c _ {t, i}) ^ {2} + \frac {1}{8 K} \sum_ {i \in S _ {t}} (f _ {\theta , i} (x _ {t}) - c _ {t, i}) ^ {2} \\ + \eta \cdot \max _ {S \in \mathcal {S}} R (S, f _ {\theta} (x _ {t}), r _ {t}) - \eta \cdot \max _ {S \in \mathcal {S}} R (S, f _ {\theta^ {\star}} (x _ {t}), r _ {t}) \\ \geq - \frac {1}{4 K} \sum_ {i \in S _ {t}} | f _ {\theta , i} (x _ {t}) - f _ {\theta^ {*}, i} (x _ {t}) | + \eta \cdot \max _ {S \in \mathcal {S}} R (S, f _ {\theta} (x _ {t}), r _ {t}) - \eta \cdot \max _ {S \in \mathcal {S}} R (S, f _ {\theta^ {*}} (x _ {t}), r _ {t}). \tag {41} \\ \end{array}
$$

Let $S(f_{\theta^{\star}}(x_t), r_t) = \arg\max_{S \in \mathcal{S}} R(S, f_{\theta^{\star}}(x_t), r_t)$ . Then, we can further lower bound Eq. (41) as follows:

$$
\begin{array}{l} - \eta (\widehat {\ell} _ {t, f _ {\theta}} - \widehat {\ell} _ {t, f _ {\theta^ {*}}}) \\ \geq - \frac {1}{4 K} \sum_ {i \in S _ {t}} | f _ {\theta , i} (x _ {t}) - f _ {\theta^ {\star}, i} (x _ {t}) | + \eta R (S (f _ {\theta^ {\star}} (x _ {t}), r _ {t}), f _ {\theta} (x _ {t}), r _ {t}) - \eta \cdot \max _ {S \in \mathcal {S}} R (S, f _ {\theta^ {\star}} (x _ {t}), r _ {t}) \\ \end{array}
$$

$$
\stackrel {(i)} {\geq} - \frac {1}{4 K} \sum_ {i \in S _ {t}} | f _ {\theta , i} (x _ {t}) - f _ {\theta^ {\star}, i} (x _ {t}) | - \eta \sum_ {i \in S (f _ {\theta^ {\star}} (x _ {t}), r _ {t})} | f _ {\theta , i} (x _ {t}) - f _ {\theta^ {\star}, i} (x _ {t}) |
$$

$$
\stackrel {(i i)} {\geq} - \frac {1}{4 T} - \frac {\eta K}{T},
$$

where $(i)$ is because Lemma 18 and $(ii)$ uses Eq. (40). This means that

$$
\begin{array}{l} Z _ {T} = - \mathbb {E} \left[ \log \mathbb {E} _ {f \sim q _ {1}} \exp \left(- \eta \sum_ {t = 1} ^ {T} \left(\widehat {\ell} _ {t, f _ {\theta}} - \widehat {\ell} _ {t, f _ {\theta^ {*}}}\right)\right) \right] \\ \leq - \mathbb {E} \left[ \log (\alpha B T) ^ {- d} \inf _ {\theta \in \Omega_ {T}} \exp \left(- \eta \sum_ {t = 1} ^ {T} \left(\widehat {\ell} _ {t, f _ {\theta}} - \widehat {\ell} _ {t, f _ {\theta^ {*}}}\right)\right) \right] \\ \leq d \log (\alpha B T) + \frac {1}{4} + K \eta = \mathcal {O} (K \eta + d \log (\alpha B T)). \\ \end{array}
$$

With the optimal choice of $\eta = \frac{1}{K^{2.5}}\sqrt{\frac{Nd\log(BT)}{T}}$ , Theorem 16 shows that Algorithm 3 guarantees that for linear function class

$$
\mathbf {R e g} _ {\mathrm{MNL}} = \mathcal {O} \left(K ^ {2. 5} \sqrt {d N T \log (B T)}\right).
$$