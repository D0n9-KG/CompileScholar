# DIVERSE PROJECTION ENSEMBLES FOR DISTRIBUTIONAL REINFORCEMENT LEARNING

Moritz A. Zanger Wendelin Böhmer Matthijs T. J. Spaan

Delft University of Technology, The Netherlands

{m.a.zanger, j.w.bohmer, m.t.j.spaan}@tudelft.nl

# ABSTRACT

In contrast to classical reinforcement learning (RL), distributional RL algorithms aim to learn the distribution of returns rather than their expected value. Since the nature of the return distribution is generally unknown a priori or arbitrarily complex, a common approach finds approximations within a set of representable, parametric distributions. Typically, this involves a projection of the unconstrained distribution onto the set of simplified distributions. We argue that this projection step entails a strong inductive bias when coupled with neural networks and gradient descent, thereby profoundly impacting the generalization behavior of learned models. In order to facilitate reliable uncertainty estimation through diversity, we study the combination of several different projections and representations in a distributional ensemble. We establish theoretical properties of such projection ensembles and derive an algorithm that uses ensemble disagreement, measured by the average 1-Wasserstein distance, as a bonus for deep exploration. We evaluate our algorithm on the behavior suite benchmark and VizDoom and find that diverse projection ensembles lead to significant performance improvements over existing methods on a variety of tasks with the most pronounced gains in directed exploration problems.

# 1 INTRODUCTION

In reinforcement learning (RL), agents interact with an unknown environment, aiming to acquire policies that yield high cumulative rewards. In pursuit of this objective, agents must engage in a trade-off between information gain and reward maximization, a dilemma known as the exploration/exploitation trade-off. In the context of model-free RL, many algorithms designed to address this problem efficiently rely on a form of the optimism in the face of uncertainty principle (Auer, 2002) where agents act according to upper confidence bounds of value estimates. When using high-capacity function approximators (e.g., neural networks) the derivation of such confidence bounds is non-trivial. One popular approach fits an ensemble of approximations to a finite set of observations (Dietterich, 2000; Lakshminarayanan et al., 2017). Based on the intuition that a set of parametric solutions explains observed data equally well but provides diverse predictions for unobserved data, deep ensembles have shown particularly successful at quantifying uncertainty for novel inputs. An exploring agent may, for example, seek to reduce this kind of uncertainty by visiting unseen state-action regions sufficiently often, until ensemble members converge to almost equal predictions. This notion of reducible uncertainty is also known as epistemic uncertainty (Hora, 1996; Der Kiureghian and Ditlevsen, 2009).

A concept somewhat orthogonal to epistemic uncertainty is aleatoric uncertainty, that is the uncertainty associated with the inherent irreducible randomness of an event. The latter is the subject of the recently popular distributional branch of RL (Bellemare et al., 2017), which aims to approximate the distribution of returns, as opposed to its mean. While distributional RL naturally lends itself to risk-sensitive learning, several results show significant improvements over classical RL even when distributions are used only to recover the mean (Bellemare et al., 2017; Dabney et al., 2018b; Rowland et al., 2019; Yang et al., 2019; Nguyen-Tang et al., 2021). In general, the probability distribution of the random return may be arbitrarily complex and difficult to represent, prompting many recent advancements to rely on novel methods to project the unconstrained return distribution onto a set of representable distributions.

In this paper, we study the combination of different projections and representations in an ensemble of distributional value learners. In this setting, agents who seek to explore previously unseen states and

actions can recognize such novel, out-of-distribution inputs by the diversity of member predictions: through learning, these predictions align with labels in frequently visited states and actions, while novel regions lead to disagreement. For this, the individual predictions for unseen inputs, hereafter also referred to as generalization behavior, are required to be sufficiently diverse. We argue that the projection step in distributional RL imposes an inductive bias that leads to such diverse generalization behaviors when joined with neural function approximation. We thus deem distributional projections instrumental to the construction of diverse ensembles, capable of effective separation of epistemic and aleatoric uncertainty. To illustrate the effect of the projection step in the function approximation setting, Fig. 1 shows a toy regression problem where the predictive distributions differ visibly for inputs x not densely covered by training data depending on the choice of projection.

Our main contributions are as follows:

(1) We introduce distributional projection ensembles and analyze their properties theoretically. In our setting, each model is iteratively updated toward the projected mixture over ensemble return distributions. We describe such use of distributional ensembles formally through a projection mixture operator and establish several of its properties, including contractivity and residual approximation errors.

(2) When using shared distributional temporal difference (TD) targets, ensemble disagreement is biased to represent distributional TD errors rather than errors w.r.t. the true return distribution. To this end, we derive a propagation scheme for epistemic uncertainty that relates absolute deviations from the true value function to distributional TD errors. This insight allows us to devise an optimism-based exploration algorithm that leverages a learned bonus for directed

(3) We implement these algorithmic elements in a deep RL setting and evaluate the resulting agent on the behavior suite (Osband et al., 2020), a benchmark collection of 468 environments, and a set of hard exploration problems in the visual domain VizDoom (Kempka et al., 2016). Our experiments show that projection ensembles aid reliable uncertainty estimation and exploration, outperforming baselines on most tasks, even when compared to significantly larger ensemble sizes.

![](images/113f1ab510e19872100f2be77a63340709ee038a66948c986898996c82aed8cf.jpg)

<details>
<summary>line</summary>

| x    | Categorical | Quantile |
| ---- | ----------- | -------- |
| -1   | -0.1        | -0.1     |
| 0    | 0.1         | 0.1      |
| 1    | 0.0         | 0.0      |
</details>

Figure 1: Toy 1D-regression: Black dots are training data with inputs x and labels y. Two models have been trained to predict the distribution $p(y|x)$ using a categorical projection (l.h.s.) and a quantile projection (r.h.s.). We plot contour lines for the $\tau = [0.1, ..., 0.9]$ quantiles of the predictive distributions over the interval $x \in [-1.5, 1.5]$ .

# 2 RELATED WORK

Our work builds on a swiftly growing body of literature in distributional RL (Morimura et al., 2010; Bellemare et al., 2017). In particular, several of our theoretical results rely on works by Rowland et al. (2018) and Dabney et al. (2018b), who first provided contraction properties with categorical and quantile projections in distributional RL respectively. Numerous recently proposed algorithms (Dabney et al., 2018a; Rowland et al., 2019; Yang et al., 2019; Nguyen-Tang et al., 2021) are based on novel representations and projections, typically with an increased capacity to represent complex distributions. In contrast to our approach, however, these methods have no built-in functionality to estimate epistemic uncertainty. To the best of our knowledge, our work is the first to study the combination of different projection operators and representations in the context of distributional RL.

Several works, however, have applied ensemble techniques to distributional approaches. For example, Clements et al. (2019), Eriksson et al. (2022), and Hoel et al. (2023) use ensembles of distributional models to derive aleatoric and epistemic risk measures. Lindenberg et al. (2020) use an ensemble of agents in independent environment instances based on categorical models to drive performance and stability. Jiang et al. (2024) leverage quantile-based ensembles to drive exploration in contextual MDPs, while Nikolov et al. (2019) combine a deterministic Q-ensemble with a distributional categorical model for information-directed sampling. In a broader sense, the use of deep ensembles for value estimation and exploration is widespread (Osband et al., 2016; 2019; Flennerhag et al., 2020; Fellows et al., 2021; Chen et al., 2017). A notable distinction between such algorithms is whether ensemble members are trained independently or whether joint TD backups are used. Our work falls into the latter category which typically requires a propagation mechanism to estimate value uncertainty rather

than uncertainty in TD targets (Janz et al., 2019; Fellows et al., 2021; Moerland et al., 2017). Our proposed propagation scheme establishes a temporal consistency between distributional TD errors and errors w.r.t. the true return distribution. In contrast to the related uncertainty Bellman equation (O'Donoghue et al., 2018), our approach applies to the distributional setting and devises uncertainty propagation from the perspective of error decomposition, rather than posterior variance.

# 3 BACKGROUND

Throughout this work, we consider a finite Markov Decision Process (MDP) (Bellman, 1957) of the tuple $(\mathcal{S},\mathcal{A},\mathcal{R},\gamma ,P,\mu)$ as the default problem framework, where $\mathcal{S}$ is the finite state space, $\mathcal{A}$ is the finite action space, $\mathcal{R}:\mathcal{S}\times \mathcal{A}\to \mathcal{P}(\mathbb{R})$ is the immediate reward distribution, $\gamma \in [0,1]$ is the discount factor, $P:\mathcal{S}\times \mathcal{A}\rightarrow \mathcal{P}(\mathcal{S})$ is the transition kernel, and $\mu :\mathcal{P}(\mathcal{S})$ is the start state distribution. Here, we write $\mathcal{P}(\mathcal{X})$ to indicate the space of probability distributions defined over some space $\mathcal{X}$ . Given a state $S_{t}$ at time $t$ , agents draw an action $A_{t}$ from a stochastic policy $\pi :\mathcal{S}\to \mathcal{P}(\mathcal{A})$ to be presented the random immediate reward $R_{t}\sim \mathcal{R}(\cdot |S_{t},A_{t})$ and the successor state $S_{t + 1}\sim P(\cdot |S_t,A_t)$ . Under policy $\pi$ and transition kernel $P$ , the discounted return is a random variable given by the discounted cumulative sum of random rewards according to $Z^{\pi}(s,a) = \sum_{t = 0}^{\infty}\gamma^{t}R_{t}$ , where $S_0 = s,A_0 = a$ . Note that our notation will generally use uppercase letters to indicate random variables. Furthermore, we write $\mathcal{D}(Z^{\pi}(s,a))\in \mathcal{P}(\mathbb{R})$ to denote the distribution of the random variable $Z^{\pi}(\mathrm{s},\mathrm{a})$ , that is a state-action-dependent distribution residing in the space of probability distributions $\mathcal{P}(\mathbb{R})$ . For explicit referrals, we label this distribution $\eta^{\pi}(s,a) = \mathcal{D}(Z^{\pi}(s,a))$ . The expected value of $Z^{\pi}(s,a)$ is known as the state-action value $Q^{\pi}(s,a) = \mathbb{E}[Z^{\pi}(s,a)]$ and adheres to a temporal consistency condition described by the Bellman equation (Bellman, 1957)

$$
Q ^ {\pi} (s, a) = \mathbb {E} _ {P, \pi} [ R _ {0} + \gamma Q ^ {\pi} (S _ {1}, A _ {1}) | S _ {0} = s, A _ {0} = a ], \tag {1}
$$

where $E_{P,\pi}$ indicates that successor states and actions are drawn from P and $\pi$ respectively. Moreover, the Bellman operator $T^{\pi}Q(s,a):=\mathbb{E}_{P,\pi}[R_{0}+\gamma Q(S_{1},A_{1})|S_{0}=s,A_{0}=a]$ has the unique fixed point $Q^{\pi}(s,a)$ .

# 3.1 DISTRIBUTIONAL REINFORCEMENT LEARNING

The distributional Bellman operator $T^{\pi}$ (Bellemare et al., 2017) is a probabilistic generalization of $T^{\pi}$ and considers return distributions rather than their expectation. For notational convenience, we first define $P^{\pi}$ to be the transition operator according to

$$
P ^ {\pi} Z (s, a) := \stackrel {{D}} {{=}} Z (S _ {1}, A _ {1}), \quad \text { where } \quad S _ {1} \sim P (\cdot | S _ {0} = s, A _ {0} = a), \quad A _ {1} \sim \pi (\cdot | S _ {1}), \tag {2}
$$

and $\stackrel{D}{=}$ indicates an equality in distributional law (White, 1988). In this setting, the distributional Bellman operator is defined as

$$
\mathcal {T} ^ {\pi} Z (s, a) := R _ {0} + \gamma P ^ {\pi} Z (s, a). \tag {3}
$$

Similar to the classical Bellman operator, the distributional counterpart $\mathcal{T}^{\pi}:\mathcal{P}(\mathbb{R})^{\mathcal{S}\times \mathcal{A}}\to \mathcal{P}(\mathbb{R})^{\mathcal{S}\times \mathcal{A}}$ has the unique fixed point $\mathcal{T}^{\pi}Z^{\pi} = Z^{\pi}$ , that is the true return distribution $Z^{\pi}$ . In the context of iterative algorithms, we will also refer to the identity $\mathcal{T}^{\pi}Z(s,a)$ as a bootstrap of the distribution $Z(s,a)$ . For the analysis of many properties of $\mathcal{T}^{\pi}$ , it is helpful to define a distance metric over the space of return distributions $\mathcal{P}(\mathbb{R})^{\mathcal{S}\times \mathcal{A}}$ . Here, the supremum $p$ -Wasserstein metric $\bar{w}_p:\mathcal{P}(\mathbb{R})^{\mathcal{S}\times \mathcal{A}}\times \mathcal{P}(\mathbb{R})^{\mathcal{S}\times \mathcal{A}}\to [0,\infty ]$ has proven particularly useful. In the univariate case, $\bar{w}_p$ is given by

$$
\bar {w} _ {p} \left(\nu , \nu^ {\prime}\right) = \sup _ {s, a \in \mathcal {S} \times \mathcal {A}} \left(\int_ {0} ^ {1} \left| F _ {\nu (s, a)} ^ {- 1} (\tau) - F _ {\nu^ {\prime} (s, a)} ^ {- 1} (\tau) \right| ^ {p} d \tau\right) ^ {\frac {1}{p}}, \tag {4}
$$

where $p \in [1, \infty)$ , $\nu, \nu'$ are any two state-action return distributions, and $F_{\nu(s,a)} : \mathbb{R} \to [0,1]$ is the cumulative distribution function (CDF) of $\nu(s,a)$ . For notational brevity, we will use the notation $w_p(\nu(s,a), \nu'(s,a)) = w_p(\nu, \nu')(s,a)$ for the p-Wasserstein distance between distributions $\nu, \nu'$ , evaluated at $(s,a)$ . One of the central insights of previous works in distributional RL is that the operator $T^\pi$ is a $\gamma$ -contraction in $\bar{w}_p$ (Bellemare et al., 2017), meaning that we have $\bar{w}_p(\mathcal{T}^\pi \nu, \mathcal{T}^\pi \nu') \leq \gamma \bar{w}_p(\nu, \nu')$ , a property that allows us (in principle) to construct convergent value iteration schemes in the distributional setting.

# 3.2 CATEGORICAL AND QUANTILE DISTRIBUTIONAL RL

In general, we can not represent arbitrary probability distributions in $\mathcal{P}(\mathbb{R})$ and instead resort to parametric models capable of representing a subset $\mathcal{F}$ of $\mathcal{P}(\mathbb{R})$ . Following Bellemare et al. (2023),

we refer to $\mathcal{F}$ as a representation and define it to be the set of parametric distributions $P_{\theta}$ with $\mathcal{F} = \{P_{\theta} \in \mathcal{P}(\mathbb{R}) : \theta \in \Theta\}$ . Furthermore, we define the projection operator $\Pi : \mathcal{P}(\mathbb{R}) \to \mathcal{F}$ to be a mapping from the space of probability distributions $\mathcal{P}(\mathbb{R})$ to the representation $\mathcal{F}$ . Recently, two particular choices for representation and projection have proven highly performant in deep RL: the categorical and quantile model.

The categorical representation (Bellemare et al., 2017; Rowland et al., 2018) assumes a weighted mixture of $K$ Dirac deltas $\delta_{z_k}$ with support at evenly spaced locations $z_k \in [z_1, \dots, z_K]$ . The categorical representation is then given by $\mathcal{F}_C = \{\sum_{k=1}^{K} \theta_k \delta_{z_k} | \theta_k \geq 0, \sum_{k=1}^{K} \theta_k = 1\}$ . The corresponding categorical projection operator $\Pi_C$ maps a distribution $\nu$ from $\mathcal{P}(\mathbb{R})$ to a distribution in $\mathcal{F}_C$ by assigning probability mass inversely proportional to the distance to the closest $z_k$ in the support $[z_1, \dots, z_K]$ for every point in the support of $\nu$ . For example, for a single Dirac distribution $\delta_x$ and assuming $z_k \leq x \leq z_{k+1}$ the projection is given by

$$
\Pi_ {C} \delta_ {x} = \frac {z _ {k + 1} - x}{z _ {k + 1} - z _ {k}} \delta_ {z _ {k}} + \frac {x - z _ {k}}{z _ {k + 1} - z _ {k}} \delta_ {z _ {k + 1}}. \tag {5}
$$

The corner cases are defined such that $\Pi_{C}\delta_{x} = \delta_{z_{1}}\forall x \leq z_{1}$ and $\Pi_{C}\delta_{x} = \delta_{z_{K}}\forall x \geq z_{K}$ . It is straightforward to extend the above projection step to finite mixtures of Dirac distributions through $\Pi_{C}\sum_{k}p_{k}\delta_{z_{k}} = \sum_{k}p_{k}\Pi_{C}\delta_{z_{k}}$ . The full definition of the projection $\Pi_{C}$ is deferred to Appendix A.5.

The quantile representation (Dabney et al., 2018b), like the categorical representation, comprises mixture distributions of Dirac deltas $\delta_{\theta_k}(z)$ , but in contrast parametrizes their locations rather than probabilities. This yields the representation $\mathcal{F}_Q = \{\sum_{k=1}^{K} \frac{1}{K} \delta_{\theta_k}(z) | \theta_k \in \mathbb{R}\}$ . For some distribution $\nu \in \mathcal{P}(\mathbb{R})$ , the quantile projection $\Pi_Q \nu$ is a mixture of $K$ Dirac delta distributions with the particular choice of locations that minimizes the 1-Wasserstein distance between $\nu \in \mathcal{P}(\mathbb{R})$ and the projection $\Pi_Q \nu \in \mathcal{F}_Q$ . The parametrization $\theta_k$ with minimal 1-Wasserstein distance is given by the evaluation of the inverse of the CDF, $F_\nu^{-1}$ , at midpoint quantiles $\tau_k = \frac{2k-1}{2K}$ , $k \in [1, ..., K]$ , s.t. $\theta_k = F_\nu^{-1}(\frac{2k-1}{2K})$ . Equivalently, $\theta_k$ is the minimizer of the quantile regression loss (QR) (Koenker and Hallock, 2001), which is more amenable to gradient-based optimization. The loss is given by

$$
\mathcal {L} _ {Q} (\theta_ {k}, \nu) = \mathbb {E} _ {Z \sim \nu} [ \rho_ {\tau_ {k}} (Z - \theta_ {k}) ], \tag {6}
$$

where $\rho_{\tau}(u) = u(\tau -\mathbb{1}_{\{u\leq 0\}}(u))$ is an error function that assigns asymmetric weight to over- or underestimation errors and $\mathbb{1}$ denotes the indicator function.

# 4 EXPLORATION WITH DISTRIBUTIONAL PROJECTION ENSEMBLES

This paper is foremost concerned with leveraging ensembles with diverse generalization behaviors induced by different representations and projection operators. To introduce the concept of distributional projection ensembles and their properties, we describe the main components in a formal setting that foregoes sample-based stochastic approximation and function approximation, before moving to a more practical deep RL setting in Section 5. We begin by outlining the projection mixture operator and its contraction properties. While this does not inform an exploration algorithm in its own right, it lays a solid algorithmic foundation for the subsequently derived exploration framework. Consider an ensemble $E = \{\eta_i(s,a) \mid i \in [1,\dots,M]\}$ of $M$ member distributions $\eta_i(s,a)$ , each associated with a representation $\mathcal{F}_i$ and a projection operator $\Pi_i$ . In this setting, we assume that each member distribution $\eta_i(s,a) \in \mathcal{F}_i$ is an element of the associated representation $\mathcal{F}_i$ and the projection operator $\Pi_i: \mathcal{P}(\mathbb{R}) \to \mathcal{F}_i$ maps any distribution $\nu \in \mathcal{P}(\mathbb{R})$ to $\mathcal{F}_i$ such that $\Pi_i\nu \in \mathcal{F}_i$ . The set of representable uniform mixture distributions over $E$ is then given by $\mathcal{F}_E = \{\eta_E(s,a) \mid \eta_E(s,a) = \frac{1}{M}\sum_i\eta_i(s,a),\eta_i(s,a)\in \mathcal{F}_i,i\in [1,\dots,M]\}$ . We can now define a central object in this paper, the projection mixture operator $\Omega_M: \mathcal{P}(\mathbb{R})\to \mathcal{F}_E$ , as follows:

$$
\Omega_ {M} \eta (s, a) = \frac {1}{M} \sum_ {i = 1} ^ {M} \Pi_ {i} \eta (s, a). \tag {7}
$$

Joining $\Omega_{M}$ with the distributional Bellman operator $T^{\pi}$ yields the combined operator $\Omega_{M}T^{\pi}$ . Fig. 2 illustrates the intuition behind the operator $\Omega_{M}T^{\pi}$ : the distributional Bellman operator $T^{\pi}$ is applied to a return distribution $\eta$ (Fig. 2 a and b), then projects the resulting distribution with the individual projection operators $\Pi_{i}$ onto M different representations $\eta_{i} = \Pi_{i}T^{\pi}\eta \in F_{i}$ (Fig. 2 c and d), and finally recombines the ensemble members into a mixture model in $F_{E}$ (Fig. 2 e). In connection with iterative algorithms, we are often interested in the contractivity of the combined operator $\Omega_{M}T^{\pi}$ to establish convergence. Proposition 1 delineates conditions under which we can combine individual projections $\Pi_{i}$ such that the resulting combined operator $\Omega_{M}T^{\pi}$ is a contraction mapping.

![](images/f7f7443ba5fb242f0b3059f7d65a8e030d5b8f50230e45de417d0e9c21619388.jpg)

![](images/94a619be58e3d1f57c344c5f6f31ffd5070cbaef18db0bd4beb7ad10b9c8746e.jpg)

![](images/913d0fdca48a95b32e3cb4e5de6f1bf4827451706f7e978f81e67216116b16a5.jpg)

![](images/c99afb39bc2ccc1bc608777a39a1be91cc6062f29b39a7e5c5f2c728ea5d7ff6.jpg)

![](images/943f9e7cc1aff7845d986e2ba19fcd6d724e36da202430ef9c5c8ea7dff78c04.jpg)  
Figure 2: Illustration of the projection mixture operator with quantile and categorical projections.

Proposition 1 Let $\Pi_i$ , $i \in [1, ..., M]$ be projection operators $\Pi_i: \mathcal{P}(\mathbb{R}) \to \mathcal{F}_i$ mapping from the space of probability distributions $\mathcal{P}(\mathbb{R})$ to representations $\mathcal{F}_i$ and denote the projection mixture operator $\Omega_M: \mathcal{P}(\mathbb{R}) \to \mathcal{F}_E$ as defined in Eq. 7. Furthermore, assume that for some $p \in [1, \infty)$ each projection $\Pi_i$ is bounded in the $p$ -Wasserstein metric in the sense that for any two return distributions $\eta, \eta'$ we have $w_p(\Pi_i \eta, \Pi_i \eta')(s, a) \leq c_i w_p(\eta, \eta')(s, a)$ for a constant $c_i$ . Then, the combined operator $\Omega_M \mathcal{T}^\pi$ is bounded in the supremum $p$ -Wasserstein distance $\bar{w}_p$ by

$$
\bar {w} _ {p} (\Omega_ {M} \mathcal {T} ^ {\pi} \eta , \Omega_ {M} \mathcal {T} ^ {\pi} \eta^ {\prime}) \leq \bar {c} _ {p} \gamma \bar {w} _ {p} (\eta , \eta^ {\prime})
$$

and is accordingly a contraction so long as $\bar{c}_p\gamma < 1$ , where $\bar{c}_p = (\sum_{i=1}^{M}\frac{1}{M} c_i^p)^{1/p}$ .

The proof is deferred to Appendix A. The contraction condition in Proposition 1 is naturally satisfied for example if all projections $\Pi_i$ are non-expansions in a joint metric $w_p$ . It is, however, more permissive in the sense that it only requires the joint modulus $\bar{c}_p$ to be limited, allowing for expanding operators in the ensemble for finite $p$ . A contracting combined operator $\Omega_M\mathcal{T}^\pi$ allows us to formulate a simple convergent iteration scheme where in a sequence of steps $k$ , ensemble members are moved toward the projected mixture distribution according to $\hat{\eta}_{i,k+1} = \Pi_i\mathcal{T}^\pi\hat{\eta}_{E,k}$ , yielding the $(k+1)$ -th mixture distribution $\hat{\eta}_{E,k+1} = \frac{1}{M}\sum_{i=1}^{M}\hat{\eta}_{i,k+1}$ . This procedure can be compactly expressed by

$$
\hat {\eta} _ {E, k + 1} = \Omega_ {M} \mathcal {T} ^ {\pi} \hat {\eta} _ {E, k}, \quad \text { for } \quad k = [ 0, 1, 2, 3,... ] \tag {8}
$$

and has a unique fixed point which we denote $\eta_{E}^{\pi} = \hat{\eta}_{E,\infty}$ .

# 4.1 FROM DISTRIBUTIONAL APPROXIMATIONS TO OPTIMISTIC BOUNDS

We proceed to describe how distributional projection ensembles can be leveraged for exploration. Our setting considers exploration strategies based on the upper-confidence-bound (UCB) algorithm (Auer, 2002). In the context of model-free RL, provably efficient algorithms often rely on the construction of a bound, that overestimates the true state-action value with high probability (Jin et al., 2018; 2020). In other words, we are interested in finding an optimistic value $\hat{Q}^{+}(s,a)$ such that $\hat{Q}^{+}(s,a) \geq Q^{\pi}(s,a)$ with high probability. To this end, Proposition 2 relates an estimate $\hat{Q}(s,a)$ to the true value $Q^{\pi}(s,a)$ through a distributional error term.

Proposition 2 Let $\hat{Q}(s,a) = \mathbb{E}[\hat{Z}(s,a)]$ be a state-action value estimate where $\hat{Z}(s,a) \sim \hat{\eta}(s,a)$ is a random variable distributed according to an estimate $\hat{\eta}(s,a)$ of the true state-action return distribution $\eta^{\pi}(s,a)$ . Further, denote $Q^{\pi}(s,a) = \mathbb{E}[Z^{\pi}(s,a)]$ the true state-action, where $Z^{\pi}(s,a) \sim \eta^{\pi}(s,a)$ . We have that $Q^{\pi}(s,a)$ is bounded from above by

$$
\hat {Q} (s, a) + w _ {1} \left(\hat {\eta}, \eta^ {\pi}\right) (s, a) \geq Q ^ {\pi} (s, a) \quad \forall (s, a) \in \mathcal {S} \times \mathcal {A},
$$

where $w_{1}$ is the 1-Wasserstein distance metric.

The proof follows from the definition of the Wasserstein distance and is given in Appendix A. Proposition 2 implies that, for a given distributional estimate $\hat{\eta}(s,a)$ , we can construct an optimistic upper bound on $Q^{\pi}(s,a)$ by adding a bonus of the 1-Wasserstein distance between an estimate $\hat{\eta}(s,a)$ and the true return distribution $\eta^{\pi}(s,a)$ , which we define as $b^{\pi}(s,a) = w_{1}(\hat{\eta},\eta^{\pi})(s,a)$ in the following. By adopting an optimistic action-selection with this guaranteed upper bound on $Q^{\pi}(s,a)$

according to

$$
a = \underset {a \in \mathcal {A}} {\arg \max} [ \hat {Q} (s, a) + b ^ {\pi} (s, a) ], \tag {9}
$$

we maintain that the resulting policy inherits efficient exploration properties of known optimism-based exploration methods. Note that in a convergent iteration scheme, we should expect the bonus $b^{\pi}(s,a)$ to almost vanish in the limit of infinite iterations. We thus refer to $b^{\pi}(s,a)$ as a measure of the epistemic uncertainty of the estimate $\hat{\eta}(s,a)$ .

# 4.2 PROPAGATION OF EPISTEMIC UNCERTAINTY THROUGH DISTRIBUTIONAL ERRORS

By Proposition 2, an optimistic policy for efficient exploration can be derived from the distributional error $b^{\pi}(s,a)$ . However, since we do not assume knowledge of the true return distribution $\eta^{\pi}(s,a)$ ,

this error term requires estimation. The primary purpose of this section is to establish such an estimator by propagating distributional TD errors. This is necessary because the use of TD backups prohibits a consistent uncertainty quantification in values (described extensively in the Bayesian setting for example by Fellows et al. 2021). The issue is particularly easy to see by considering the backup in a single $(s,a)$ tuple: even if every estimate $\hat{\eta}_{i}(s,a)$ in an ensemble fits the backup $\mathcal{T}^{\pi}\hat{\eta}_{E}(s,a)$ accurately, this does not imply $\hat{\eta}_{i}(s,a)=\eta^{\pi}(s,a)$ as the TD backup may have been incorrect. Even a well-behaved ensemble (in the sense that its disagreement reliably measures prediction errors) in this case quantifies errors w.r.t. the bootstrapped target $\Omega_{M}\mathcal{T}^{\pi}\hat{\eta}_{E}(s,a)$ , rather than the true return distribution $\eta^{\pi}(s,a)$ .

To establish a bonus estimate that allows for optimistic action selection in the spirit of Proposition 2, we now derive a propagation scheme for epistemic uncertainty in the distributional setting. More specifically, we find that an upper bound on the bonus $b^{\pi}(s,a)$ satisfies a temporal consistency condition, similar to the Bellman equations, that relates the total distributional error $w_{1}(\hat{\eta},\eta_{E}^{\pi})(s,a)$ to a one-step error $w_{1}(\hat{\eta},\Omega_{M}\mathcal{T}^{\pi}\hat{\eta})(s,a)$ that is more amenable to estimation.

Theorem 3 Let $\hat{\eta}(s,a)\in\mathcal{P}(\mathbb{R})$ be an estimate of the true return distribution $\eta^{\pi}(s,a)\in\mathcal{P}(\mathbb{R})$ , and denote the projection mixture operator $\Omega_{M}:\mathcal{P}(\mathbb{R})\to\mathcal{F}_{E}$ with members $\Pi_{i}$ and bounding moduli $c_{i}$ and $\bar{c}_{p}$ as defined in Proposition 1. Furthermore, assume $\Omega_{M}\mathcal{T}^{\pi}$ is a contraction mapping with fixed point $\eta_{E}^{\pi}$ . We then have for all $(s,a)\in\mathcal{S}\times\mathcal{A}$

$$
w _ {1} \big (\hat {\eta}, \eta_ {E} ^ {\pi} \big) (s, a) \leq w _ {1} \big (\hat {\eta}, \Omega_ {M} \mathcal {T} ^ {\pi} \hat {\eta} \big) (s, a) + \bar {c} _ {1} \gamma \mathbb {E} \big [ w _ {1} \big (\hat {\eta}, \eta_ {E} ^ {\pi} \big) (S _ {1}, A _ {1}) | S _ {0} = s, A _ {0} = a \big ],
$$

where $S_{1}\sim P(\cdot |S_{0} = s,A_{0} = a)$ and $A_{1}\sim \pi (\cdot |S_{1})$

The proof is given in Appendix A and exploits the triangle inequality property of the Wasserstein distance. It may be worth noting that Theorem 3 is a general result that is not restricted to the use of projection ensembles. It is, however, a natural complement to the iteration described in Eq. 8 in that it allows us to reconcile the benefits of bootstrapping diverse ensemble mixtures with optimistic action selection for directed exploration. To this end, we devise a separate iteration procedure aimed at finding an approximate upper bound on $w_{1}(\hat{\eta},\eta_{E}^{\pi})(s,a)$ . Denoting the $k$ -th iterate of the bonus estimate $\hat{b}_k(s,a)$ , we have by Theorem 3 that the iteration

$$
\hat {b} _ {k + 1} (s, a) = w _ {1} \big (\hat {\eta}, \Omega_ {M} \mathcal {T} ^ {\pi} \hat {\eta} \big) (s, a) + \bar {c} _ {1} \gamma \mathbb {E} _ {P, \pi} \big [ \hat {b} _ {k} (S _ {1}, A _ {1}) | S _ {0} = s, A _ {0} = a \big ]   \forall (s, a) \in \mathcal {S} \times \mathcal {A},
$$

converges to an upper bound on $w_{1}(\hat{\eta}, \eta_{E}^{\pi})(s, a)^{1}$ . Notably, this iteration requires only a local error estimate $w_{1}(\hat{\eta}, \Omega_{M} \mathcal{T}^{\pi} \hat{\eta})(s, a)$ and is more amenable to estimation through our ensemble.

We conclude this section with the remark that the use of projection ensembles may clash with the intuition that epistemic uncertainty should vanish in convergence. This is because each member inherits irreducible approximation errors from the projections $\Pi_{i}$ . In Appendix A, we provide general bounds for these errors and show that residual errors can be controlled through the number of atoms K in the specific example of an ensemble based on the quantile and categorical projections.

# 5 DEEP DISTRIBUTIONAL RL WITH PROJECTION ENSEMBLES

Section 4 has introduced the concept of projection ensembles in a formal setting. In this section, we aim to transcribe the previously derived algorithmic components into a deep RL algorithm that departs from several of the previous assumptions. Specifically, this includes 1) control with a greedy policy, 2) sample-based stochastic approximation, 3) nonlinear function approximation, and 4) gradient-based optimization. While this sets the following section apart from the theoretical setting considered in Section 4, we hypothesize that diverse projection ensembles bring to bear several advantages in this scenario. The underlying idea is that distributional projections and the functional constraints they entail offer an effective tool to impose diverse generalization behaviors on an ensemble, yielding a more reliable tool for out-of-distribution sample detection. In particular, we implement the above-described algorithm with a neural ensemble comprising the models of the two popular deep RL algorithms quantile regression deep Q network (QR-DQN) (Dabney et al., 2018b) and C51 (Bellemare et al., 2017).

# 5.1 DEEP QUANTILE AND CATEGORICAL PROJECTION ENSEMBLES FOR EXPLORATION

In this section, we propose Projection Ensemble DQN (PE-DQN), a deep RL algorithm that combines the quantile and categorical projections (Dabney et al., 2018b; Bellemare et al., 2017) into a diverse ensemble to drive exploration and learning stability. Our parametric model consists of the mixture distribution $\eta_{E,\theta}$ parametrized by $\theta$ . We construct $\eta_{E,\theta}$ as an equal mixture between a quantile and a categorical representation, each parametrized through a NN with $K$ output logits where we use the notation $\theta_{ik}$ to mean the $k$ -th logit of the network parametrized by the parameters $\theta_i$ of the $i$ -th model in the ensemble. We consider a sample transition $(s,a,r,s',a')$ where $a'$ is chosen greedily according to $\mathbb{E}_{Z\sim \eta_{E,\theta}(s',a')}[Z]$ . Dependencies on $(s,a)$ are hereafter dropped for conciseness by writing $\theta_{ik} = \theta_{ik}(s,a)$ and $\theta_{ik}' = \theta_{ik}(s',a')$ .

Projection losses. Next, we assume that bootstrapped return distributions are generated by a set of delayed parameters $\tilde{\theta}$ , as is common (Mnih et al., 2015). The stochastic (sampled) version of the distributional Bellman operator $\hat{\mathcal{T}}^{\pi}$ , applied to the target ensemble's mixture distribution $\eta_{E,\tilde{\theta}}$ yields

$$
\hat {\mathcal {T}} ^ {\pi} \eta_ {E, \tilde {\theta}} = \frac {1}{2} \sum_ {i = 1} ^ {M = 2} \sum_ {k = 1} ^ {K} p (\tilde {\theta} _ {i k} ^ {\prime})   \delta_ {r + \gamma z (\tilde {\theta} _ {i k} ^ {\prime})}. \tag {10}
$$

Instead of applying the projection mixture $\Omega_M$ analytically, as done in Section 4, the parametric estimates $\eta_{E,\theta}$ are moved incrementally towards a projected target distribution through gradient descent on a loss function. In the quantile representation, we augment the classical quantile regression loss (Koenker and Hallock, 2001) with an importance-sampling ratio $Kp(\tilde{\theta}_{ij}^{\prime})$ to correct for the nonuniformity of atoms from the bootstrapped distribution $\hat{T}^{\pi}\eta_{E,\tilde{\theta}}$ . For a set of fixed quantiles $\tau_k$ , the loss $\mathcal{L}_1$ is given by

$$
\mathcal {L} _ {1} \big (\eta_ {\theta_ {1}}, \Pi_ {Q} \hat {\mathcal {T}} ^ {\pi} \eta_ {E, \tilde {\theta}} \big) = \sum_ {i = 1} ^ {M = 2} \sum_ {k, j = 1} ^ {K} K p (\tilde {\theta} _ {i j} ^ {\prime}) \Big (\rho_ {\tau_ {k}} \big (r + \gamma z (\tilde {\theta} _ {i j} ^ {\prime}) - \theta_ {1 k} \big) \Big). \tag {11}
$$

The categorical model minimizes the Kullback-Leibler (KL) divergence between the projected bootstrap distribution $\Pi_{C}\hat{T}^{\pi}\eta_{E,\tilde{\theta}}$ and an estimate $\eta_{\theta_{2}}$ . The corresponding loss is given by

$$
\mathcal {L} _ {2} \left(\eta_ {\theta_ {2}}, \Pi_ {C} \hat {\mathcal {T}} ^ {\pi} \eta_ {E, \tilde {\theta}}\right) = D _ {K L} \left(\Pi_ {C} \hat {\mathcal {T}} ^ {\pi} \eta_ {E, \tilde {\theta}} \| \eta_ {\theta_ {2}}\right). \tag {12}
$$

As $\hat{\mathcal{T}}^{\pi}\eta_{E,\tilde{\theta}}$ is a mixture of Dirac distributions, the definition of the projection $\Pi_C$ according to Eq. 5 can be applied straightforwardly to obtain the projected bootstrap distribution $\Pi_C\hat{\mathcal{T}}^{\pi}\eta_{E,\tilde{\theta}}$ .

Uncertainty Propagation. We aim to estimate a state-action dependent bonus $b_{\phi}(s,a)$ in the spirit of Theorem 3 and the subsequently derived iteration with a set of parameters $\phi$ . For this, we estimate the local error estimate $w_{1}(\eta_{E,\theta},\Omega_{M}\hat{\mathcal{T}}^{\pi}\eta_{E,\theta})(s,a)$ as the average ensemble disagreement $w_{\mathrm{avg}}(s,a)=1/(M(M-1))\sum_{i,j=1}^{M}w_{1}(\eta_{\theta_{i}},\eta_{\theta_{j}})(s,a)$ . The bonus $b_{\phi}(s,a)$ can then be learned in the same fashion as a regular value function with the local uncertainty estimate $w_{\mathrm{avg}}(s,a)$ as an intrinsic reward. This yields the exploratory action-selection rule

$$
a _ {\epsilon} = \underset {a \in \mathcal {A}} {\arg \max} \left(\mathbb {E} _ {Z \sim \eta_ {E, \theta} (s, a)} [ Z ] + \beta b _ {\phi} (s, a)\right), \tag {13}
$$

where $\beta$ is a hyperparameter to control the policy's drive towards exploratory actions. Further details on our implementation and an illustration of the difference between local error estimates and bonus estimates in practice are given in Appendix B.2 and Appendix B.3.

# 6 EXPERIMENTAL RESULTS

Our experiments are designed to provide us with a better understanding of how PE-DQN operates, in comparison to related algorithms as well as in relation to its algorithmic elements. To this end, we aimed to keep codebases and hyperparameters between all implementations equal up to algorithm-specific parameters, which we optimized with a grid search on a selected subsets of problems. Further details regarding the experimental design and implementations are provided in Appendix B.

We outline our choice of baselines briefly: Bootstrapped DQN with prior functions (BDQN+P) (Osband et al., 2019) approximates posterior sampling of a parametric value function by combining statistical bootstrapping with additive prior functions in an ensemble of DQN agents. Information-directed sampling (IDS-C51) (Nikolov et al., 2019) builds on the BDQN+P architecture but acts

according to an information-gain ratio for which Nikolov et al. (2019) estimate aleatoric uncertainty (noise) with the categorical C51 model. In contrast, Decaying Left-Truncated Variance (DLTV) QR-DQN (Mavrin et al., 2019) uses a distributional value approximation based on the quantile representation and follows a decaying exploration bonus of the left-truncated variance.

# 6.1 Do DIFFERENT PROJECTIONS LEAD TO DIFFERENT GENERALIZATION BEHAVIOR?

First, we examine empirically the influence of the projection step in deep distributional RL on generalization behaviors. For this, we probe the influence of the quantile and categorical projections on generalization through an experiment that evaluates exploration in a reward-free setting. Specifically, we equip agents with an action-selection rule that maximizes a particular statistic $S[Z]$ of the predictive distribution $\hat{\eta}(s,a)$ according to

$$
a = \operatorname * {a r g   m a x} _ {a \in \mathcal {A}} \left(\mathbb {S} [ Z ]\right), Z \sim \hat {\eta} (s, a).
$$

The underlying idea is that this selection rule leads to exploration of novel state-action regions only if high values of the statistic are correlated with high epistemic uncertainty. For example, if we choose a quantile representation with $\mathbb{S}[Z]$ to be the variance of the distribution, we recover a basic form of the exploration algorithm DLTV-QR (Mavrin et al., 2019). Fig. 3 shows the results of this study for the first four statistical moments on the deep exploration benchmark deep sea with size 50. Except for the mean (the greedy policy), the choice of projection influences significantly whether the statistic-maximizing policy leads to more exploration, implying that the generalization behaviour of the 2nd to 4th moment of the predictive distributions is shaped distinctly by the employed projection.

![](images/2fc585c4d88bfd8a010a9e8eca74a0029da9fd74bd2ec53b17e92351458d30ca.jpg)

<details>
<summary>violin</summary>

| Statistic | Categorical | Quantile |
| --------- | ----------- | -------- |
| Mean      | 0.3         | 0.4      |
| Var       | 0.7         | 0.8      |
| Skew      | 0.6         | 0.9      |
| Kurt      | 0.8         | 0.9      |
</details>

Figure 3: Deep-sea exploration with different statistics. Higher means more exploration. Bars represent medians and interquartile ranges of 30 seeds.

# 6.2 THE BEHAVIOUR SUITE

In order to assess the learning process of agents in various aspects on a wide range of tasks, we evaluate PE-DQN on the behavior suite (bsuite) (Osband et al., 2020), a battery of benchmark problems constructed to assess key properties of RL algorithms. The suite consists of 23 tasks with up to 22 variations in size or seed, totaling 468 environments.

Comparative evaluation. Fig. 4 (a) shows the results of the entire bsuite experiment, summarized in seven core capabilities. These capability scores are computed as proposed by Osband et al. (2020) and follow a handcrafted scoring function per environment. For example, exploration capability is scored by the average regret in the sparse-reward environments deep sea, stochastic deep sea, and cartpole swingup. The full set of results is provided in Appendix B. Perhaps unsurprisingly, PE-DQN has its strongest performance in the exploration category but we find that it improves upon baselines in several more categories. Note here that PE-DQN uses substantially fewer models than the baselines, with a total of 4 distributional models compared to the 20 DQN models used in the ensembles of both BDQN+P and IDS, where the latter requires an additional C51 model.

# 6.3 THE DEEP-SEA ENVIRONMENT AND ABLATIONS

Deep sea is a hard exploration problem in the behavior suite and has recently gained popularity as an exploration benchmark (Osband et al., 2019; Janz et al., 2019; Flennerhag et al., 2020). It is a sparse reward environment where agents can reach the only rewarding state at the bottom right of an $N \times N$ grid through a unique sequence of actions in an exponentially growing trajectory space. We ran an additional experiment on deep sea with grid sizes up to 100; double the maximal size in the behavior suite. Fig. 4 (b) shows a summary of this experiment where we evaluated episodic regret, that is the number of non-rewarding episodes with a maximum budget of 10000 episodes. PE-DQN scales more gracefully to larger sizes of the problem than the baselines, reducing the median regret by roughly half. The r.h.s. plot in Fig. 4 (b) shows the results of ablation studies designed to provide a more nuanced view of PE-DQN's performance; the baselines labeled PE-DQN[QR/QR] and PE-DQN[C51/C51] use the same bonus estimation step as PE-DQN except that ensemble members consist of equivalent models with the same projections and representations. Conversely, PE-DQN [Ind.] uses PE-DQN's diverse projection ensemble and employs an optimistic action-selection directly with the ensemble disagreement $w_{\mathrm{avg}}(s,a)$ but trains models independently and accordingly does not make use of an

![](images/36a1cf7df480a5f96fc34c1b35e86abde78a3e081a381c07c9b08b149341955d.jpg)

Figure 4: (a) Summary of bsuite experiments. Wide is better. (b) Median episodic regret for deep sea sizes up to 100. Low is better. Shaded regions are the interquartile range of 10 seeds.   
![](images/702199ea4ae710ab117e2f4d9100f7309651e8a9df3445a499c8cae4a3dd5a81.jpg)  
Figure 5: (a) Visual observation in the VizDoom environment (Kempka et al., 2016). (b) Mean learning curves in different variations of the MyWayHome VizDoom environment. Shaded regions are $90\%$ Student's t confidence intervals from 10 seeds.

uncertainty propagation scheme in the spirit of Theorem 3. Both components lead to a pronounced difference in exploration capability and rendered indispensable to PE-DQN's overall performance.

# 6.4 THE VIZDOOM ENVIRONMENT

We investigate PE-DQN's behavior in a high-dimensional visual domain. The VizDoom environment MyWayHome (Kempka et al., 2016) tasks agents with finding a (rewarding) object by navigating in a maze-like map with ego-perspective pixel observations as seen in Fig. 5 (a). Following work by Pathak et al. (2017), we run three variations of this experiment where the reward sparsity is increased by spawning the player further away from the goal object. Learning curves for all algorithms are shown in Fig. 5 (b). Among the tested algorithms, only PE-DQN finds the object across all 10 seeds in all environments, indicating particularly reliable novelty detection. Interestingly, the sparse domain proved harder to baseline algorithms which we attribute to the “forkedness” of the associated map (see Appendix B). This result moreover shows that diverse projection ensembles scale gracefully to high-dimensional domains while using significantly fewer models than the ensemble-based baselines.

# 7 CONCLUSION

In this work, we have introduced projection ensembles for distributional RL, a method combining models based on different parametric representations and projections of return distributions. We provided a theoretical analysis that establishes convergence conditions and bounds on residual approximation errors that apply to general compositions of such projection ensembles. Furthermore, we introduced a general propagation method that reconciles one-step distributional TD errors with optimism-based exploration. PE-DQN, a deep RL algorithm, empirically demonstrates the efficacy of diverse projection ensembles on exploration tasks and showed performance improvements on a wide range of tasks. We believe our work opens up a number of promising avenues for future research. For example, we have only considered the use of uniform mixtures over distributional ensembles in this work. A continuation of this approach may aim to use a diverse collection of models less conservatively, aiming to exploit the strengths of particular models in specific regions of the state-action space.

# 8 ACKNOWLEDGEMENTS

We thank Max Weltevrede, Pascal van der Vaart, Miguel Suau, and Yaniv Oren for fruitful discussions and remarks. The project has received funding from the EU Horizon 2020 programme under grant number 964505 (Epistemic AI). The computational resources for empirical work were provided by the Delft High Performance Computing Centre (DHPC) and the Delft Artificial Intelligence Cluster (DAIC).

# REFERENCES

T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019.   
P. Auer. Using confidence bounds for exploitation-exploration trade-offs. Journal of machine learning research, 3, 2002.   
M. G. Bellemare, W. Dabney, and R. Munos. A distributional perspective on reinforcement learning. In International conference on machine learning. PMLR, 2017.   
M. G. Bellemare, W. Dabney, and M. Rowland. Distributional reinforcement learning. MIT Press, 2023.   
R. Bellman. A Markovian decision process. Journal of mathematics and mechanics, 6, 1957.   
Y. Burda, H. Edwards, D. Pathak, A. J. Storkey, T. Darrell, and A. A. Efros. Large-scale study of curiosity-driven learning. In International Conference on Learning Representations, ICLR, 2019a.   
Y. Burda, H. Edwards, A. J. Storkey, and O. Klimov. Exploration by random network distillation. In International conference on learning representations, ICLR, 2019b.   
R. Y. Chen, S. Sidor, P. Abbeel, and J. Schulman. UCB exploration via Q-ensembles. arXiv preprint arXiv:1706.01502, 2017.   
W. R. Clements, B. Van Delft, B.-M. Robaglia, R. B. Slaoui, and S. Toth. Estimating risk and uncertainty in deep reinforcement learning. arXiv preprint arXiv:1905.09638, 2019.   
W. Dabney, G. Ostrovski, D. Silver, and R. Munos. Implicit quantile networks for distributional reinforcement learning. In International conference on machine learning. PMLR, 2018a.   
W. Dabney, M. Rowland, M. Bellemare, and R. Munos. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018b.   
Delft Artificial Intelligence Cluster (DAIC), 2024.   
Delft High Performance Computing Centre (DHPC). DelftBlue Supercomputer (Phase 1), 2022.   
A. Der Kiureghian and O. Ditlevsen. Aleatory or epistemic? Does it matter? Structural safety, 31, 2009.   
T. G. Dietterich. Ensemble methods in machine learning. In Multiple classifier systems: First international workshop, MCS. Springer, 2000.   
H. Eriksson, D. Basu, M. Alibeigi, and C. Dimitrakakis. Sentinel: Taming uncertainty with ensemble based distributional reinforcement learning. In Uncertainty in artificial intelligence. PMLR, 2022.   
L. Espeholt, H. Soyer, R. Munos, K. Simonyan, V. Mnih, T. Ward, Y. Doron, V. Firoiu, T. Harley, I. Dunning, et al. Impala: Scalable distributed deep-RL with importance weighted actor-learner architectures. In International conference on machine learning. PMLR, 2018.   
M. Fellows, K. Hartikainen, and S. Whiteson. Bayesian Bellman operators. Advances in neural information processing systems, 34, 2021.

S. Flennerhag, J. X. Wang, P. Sprechmann, F. Visin, A. Galashov, S. Kapturowski, D. L. Borsa, N. Heess, A. Barreto, and R. Pascanu. Temporal difference uncertainties as a signal for exploration. arXiv preprint arXiv:2010.02255, 2020.   
K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, 2015.   
C.-J. Hoel, K. Wolff, and L. Laine. Ensemble quantile networks: Uncertainty-aware reinforcement learning with applications in autonomous driving. IEEE Transactions on intelligent transportation systems, 2023.   
S. C. Hora. Aleatory and epistemic uncertainty in probability elicitation with an example from hazardous waste management. Reliability engineering & system safety, 54, 1996.   
O. Ibe. Fundamentals of Applied Probability and Random Processes. Elsevier Science, 2014.   
D. Janz, J. Hron, P. Mazur, K. Hofmann, J. M. Hernández-Lobato, and S. Tschitschek. Successor uncertainties: Exploration and uncertainty in temporal difference learning. Advances in neural information processing systems, 32, 2019.   
Y. Jiang, J. Z. Kolter, and R. Raileanu. On the importance of exploration for generalization in reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.   
C. Jin, Z. Allen-Zhu, S. Bubeck, and M. I. Jordan. Is Q-learning provably efficient? Advances in neural information processing systems, 31, 2018.   
C. Jin, Z. Yang, Z. Wang, and M. I. Jordan. Provably efficient reinforcement learning with linear function approximation. In Proceedings of Thirty Third Conference on Learning Theory, volume 125. PMLR, 2020.   
M. Kempka, M. Wydmuch, G. Runc, J. Toczek, and W. Jaśkowski. ViZDoom: A Doom-based AI research platform for visual reinforcement learning. In IEEE Conference on computational intelligence and games. IEEE, 2016.   
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In Y. Bengio and Y. LeCun, editors, International conference on learning representations, ICLR, 2015.   
R. Koenker and K. F. Hallock. Quantile regression. Journal of economic perspectives, 15, 2001.   
B. Lakshminarayanan, A. Pritzel, and C. Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.   
B. Lindenberg, J. Nordqvist, and K.-O. Lindahl. Distributional reinforcement learning with ensembles. Algorithms, 13, 2020.   
E. Mariucci and M. Reiß. Wasserstein and total variation distance between marginals of Lévy processes. Electronic journal of statistics, 12, 2018.   
B. Mavrin, H. Yao, L. Kong, K. Wu, and Y. Yu. Distributional reinforcement learning for efficient exploration. In International conference on machine learning. PMLR, May 2019.   
V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518, 2015.   
T. M. Moerland, J. Broekens, and C. M. Jonker. Efficient exploration with double uncertain value networks. arXiv:1711.10789 [cs, stat], Nov. 2017.   
T. Morimura, M. Sugiyama, H. Kashima, H. Hachiya, and T. Tanaka. Nonparametric return distribution approximation for reinforcement learning. In International conference on machine learning. PMLR, 2010.   
T. Nguyen-Tang, S. Gupta, and S. Venkatesh. Distributional reinforcement learning via moment matching. Proceedings of the AAAI conference on artificial intelligence, 35, May 2021.

N. Nikolov, J. Kirschner, F. Berkenkamp, and A. Krause. Information-directed exploration for deep reinforcement learning. In International conference on learning representations, ICLR, 2019.   
I. Osband, C. Blundell, A. Pritzel, and B. Van Roy. Deep exploration via bootstrapped DQN. Advances in neural information processing systems, 29, 2016.   
I. Osband, B. Van Roy, D. J. Russo, Z. Wen, et al. Deep exploration via randomized value functions. Journal of machine learning research, 20, 2019.   
I. Osband, Y. Doron, M. Hessel, J. Aslanides, E. Sezener, A. Saraiva, K. McKinney, T. Lattimore, C. Szepesvári, S. Singh, B. V. Roy, R. S. Sutton, D. Silver, and H. van Hasselt. Behaviour suite for reinforcement learning. In International conference on learning representations, ICLR, 2020.   
B. O'Donoghue, I. Osband, R. Munos, and V. Mnih. The uncertainty Bellman equation and exploration. In International conference on machine learning. PMLR, 2018.   
D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning. PMLR, 2017.   
M. Rowland, M. Bellemare, W. Dabney, R. Munos, and Y. W. Teh. An analysis of categorical distributional reinforcement learning. In International conference on artificial intelligence and statistics. PMLR, 2018.   
M. Rowland, R. Dadashi, S. Kumar, R. Munos, M. G. Bellemare, and W. Dabney. Statistics and samples in distributional reinforcement learning. In International conference on machine learning. PMLR, 2019.   
T. Schaul, J. Quan, I. Antonoglou, and D. Silver. Prioritized experience replay. In Y. Bengio and Y. LeCun, editors, International conference on learning representations, ICLR, 2016.   
D. Schmidt and T. Schmied. Fast and data-efficient training of rainbow: An experimental study on Atari. arXiv preprint arXiv:2111.10247, 2021.   
D. J. White. Mean, variance, and probabilistic criteria in finite Markov decision processes: A review. Journal of optimization theory and applications, 56, 1988.   
D. Yang, L. Zhao, Z. Lin, T. Qin, J. Bian, and T.-Y. Liu. Fully parameterized quantile function for distributional reinforcement learning. Advances in neural information processing systems, 32, 2019.

# A APPENDIX

This section provides proofs for the theoretical claims and establishes further results on the residual approximation error incurred by our method.

# A.1 PROOF OF PROPOSITION 1

Before stating supporting lemmas and proofs of the results in Section 4, we recall several basic properties of the p-Wasserstein distances which we will find useful in the subsequent proofs. Derivations of these properties can for example be found in an overview by Mariucci and Reiß (2018).

P.1 The $p$ -Wasserstein distances satisfy the triangle inequality, that is

$$
w _ {p} (X, Y) \leq w _ {p} (X, Z) + w _ {p} (Z, Y).
$$

P.2 For random variables $X$ and $Y$ and an auxiliary variable $Z$ independent of $X$ and $Y$ , the $p$ -Wasserstein metric satisfies the inequality

$$
w _ {p} (X + Z, Y + Z) \leq w _ {p} (X, Y).
$$

P.3 For a real-valued scalar $a \in \mathbb{R}$ , we have

$$
w _ {p} (a X, a Y) = | a | w _ {p} (X, Y).
$$

Lemma 4 Let $\nu = \sum_{i=1}^{M} \frac{1}{M} \nu_i$ , $\nu' = \sum_{i=1}^{M} \frac{1}{M} \nu_i'$ be two mixture distributions $\nu, \nu' \in \mathcal{P}(\mathbb{R})$ . Furthermore denote $w_p(\nu, \nu')$ the $p$ -Wasserstein metric between $\nu$ and $\nu'$ . Then $w_p^p$ satisfies

$$
w _ {p} ^ {p} (\nu , \nu^ {\prime}) \leq \frac {1}{M} \sum_ {i = 1} ^ {M} w _ {p} ^ {p} (\nu_ {i}, \nu_ {i} ^ {\prime}).
$$

Proof. The Wasserstein distance in its general form is expressed in terms of couplings between the probability measures $\nu$ and $\nu'$ according to

$$
w _ {p} (\nu , \nu^ {\prime}) = \inf _ {\mu \in \Gamma (\nu , \nu^ {\prime})} \mathbb {E} _ {(x, y) \sim \mu} [ | x - y | ^ {p} ] ^ {1 / p},
$$

where $\Gamma (\nu ,\nu^{\prime})$ is the set of all couplings between $\nu$ and $\nu^{\prime}$ , i.e. joint distributions on $\mathcal{P}(\mathbb{R}^2)$ with marginals $\nu$ and $\nu^{\prime}$ . Now suppose for each $i$ we have a coupling $\mu_i(x,y)\in \Gamma (\nu_i,\nu_i')$ such that

$$
\mathbb {E} _ {(x, y) \sim \mu_ {i}} [ | x - y | ^ {p} ] = \inf _ {\mu \in \Gamma (\nu_ {i}, \nu_ {i} ^ {\prime})} \mathbb {E} _ {(x, y) \sim \mu} [ | x - y | ^ {p} ] = w _ {p} ^ {p} (\nu_ {i}, \nu_ {i} ^ {\prime}).
$$

Since by definition $\mu_i(x,y)$ is a coupling of $\nu_i$ and $\nu_i'$ , the mixture of couplings $\bar{\mu}(x,y) = \sum_{i=1}^{M} \frac{1}{M} \mu_i(x,y)$ is then a valid coupling of $\nu$ and $\nu'$ , as $\int \bar{\mu}(x,y) dy = \nu(x)$ and $\int \bar{\mu}(x',y) dx' = \nu'(x)$ . We can thus write

$$
\begin{array}{l} w _ {p} ^ {p} (\nu , \nu^ {\prime}) = \inf _ {\mu \in \Gamma (\nu , \nu^ {\prime})} \mathbb {E} _ {(x, y) \sim \mu} [ | x - y | ^ {p} ] \\ \leq \mathbb {E} _ {(x, y) \sim \bar {\mu}} [ | x - y | ^ {p} ] \\ = \sum_ {i = 1} ^ {M} \frac {1}{M} \mathbb {E} _ {(x, y) \sim \mu_ {i}} [ | x - y | ^ {p} ] \\ = \sum_ {i = 1} ^ {M} \frac {1}{M} w _ {p} ^ {p} (\nu_ {i}, \nu_ {i} ^ {\prime}). \\ \end{array}
$$

Proposition 1 Let $\Pi_i$ , $i \in [1, \dots, M]$ be projection operators $\Pi_i: \mathcal{P}(\mathbb{R}) \to \mathcal{F}_i$ mapping from the space of probability distributions $\mathcal{P}(\mathbb{R})$ to representations $\mathcal{F}_i$ and denote the projection mixture operator $\Omega_M: \mathcal{P}(\mathbb{R}) \to \mathcal{F}_E$ as defined in Eq. 7. Furthermore, assume that for some $p \in [1, \infty)$ each projection $\Pi_i$ is bounded in the $p$ -Wasserstein metric in the sense that for any two return distributions $\eta, \eta'$ we have $w_p(\Pi_i \eta, \Pi_i \eta')(s, a) \leq c_i w_p(\eta, \eta')(s, a)$ for a constant $c_i$ . Then, the combined operator $\Omega_M \mathcal{T}^\pi$ is bounded in the supremum $p$ -Wasserstein distance $\bar{w}_p$ by

$$
\bar {w} _ {p} (\Omega_ {M} \mathcal {T} ^ {\pi} \eta , \Omega_ {M} \mathcal {T} ^ {\pi} \eta^ {\prime}) \leq \bar {c} _ {p} \gamma \bar {w} _ {p} (\eta , \eta^ {\prime})
$$

and is accordingly a contraction so long as $\bar{c}_p\gamma < 1$ , where $\bar{c}_p = (\sum_{i=1}^{M}\frac{1}{M} c_i^p)^{1/p}$ .

Proof. Due to the assumption of the proposition, we have $w_{p}(\Pi_{i}\nu,\Pi_{i}\nu') \leq c_{i}w_{p}(\nu,\nu')$ . With Lemma 4 and the $\gamma$ -contractivity of $T^{\pi}$ , it follows that

$$
\begin{array}{l} \bar {w} _ {p} ^ {p} (\Omega_ {M} \mathcal {T} ^ {\pi} \eta , \Omega_ {M} \mathcal {T} ^ {\pi} \eta^ {\prime}) = \bar {w} _ {p} ^ {p} (\sum_ {i = 1} ^ {M} \frac {1}{M} \Pi_ {i} \mathcal {T} ^ {\pi} \eta , \sum_ {i = 1} ^ {M} \frac {1}{M} \Pi_ {i} \mathcal {T} ^ {\pi} \eta^ {\prime}) \\ \leq \frac {1}{M} \sum_ {i = 1} ^ {M} \bar {w} _ {p} ^ {p} (\Pi_ {i} \mathcal {T} ^ {\pi} \eta , \Pi_ {i} \mathcal {T} ^ {\pi} \eta^ {\prime}) \\ \leq \frac {1}{M} \sum_ {i = 1} ^ {M} c _ {i} ^ {p} \bar {w} _ {p} ^ {p} (\mathcal {T} ^ {\pi} \eta , \mathcal {T} ^ {\pi} \eta^ {\prime}) \\ \leq \frac {1}{M} \sum_ {i = 1} ^ {M} c _ {i} ^ {p} \gamma^ {p} \bar {w} _ {p} ^ {p} (\eta , \eta^ {\prime}) \\ = \gamma^ {p} \bar {w} _ {p} ^ {p} (\eta , \eta^ {\prime}) \frac {1}{M} \sum_ {i = 1} ^ {M} c _ {i} ^ {p}. \\ \end{array}
$$

The state then finally follows by taking the $p$ -th root, yielding the joint modulus $\bar{c}_p = (\sum_{i=1}^{M} \frac{1}{M} c_i^p)^{1/p}$ .

# A.2 PROOF OF PROPOSITION 2

Proposition 2 Let $\hat{Q}(s,a) = \mathbb{E}[\hat{Z}(s,a)]$ be a state-action value estimate where $\hat{Z}(s,a) \sim \hat{\eta}(s,a)$ is a random variable distributed according to an estimate $\hat{\eta}(s,a)$ of the true state-action return distribution $\eta^{\pi}(s,a)$ . Further, denote $Q^{\pi}(s,a) = \mathbb{E}[Z^{\pi}(s,a)]$ the true state-action, where $Z^{\pi}(s,a) \sim \eta^{\pi}(s,a)$ . We have that $Q^{\pi}(s,a)$ is bounded from above by

$$
\hat {Q} (s, a) + w _ {1} \big (\hat {\eta}, \eta^ {\pi} \big) (s, a) \geq Q ^ {\pi} (s, a) \quad \forall (s, a) \in \mathcal {S} \times \mathcal {A},
$$

where $w_{1}$ is the 1-Wasserstein distance metric.

Proof. We begin by stating a property that relates the expected value $\mathbb{E}[X]$ to the CDF of $X$ under the condition that the expectation $\mathbb{E}[X]$ is well-defined and finite. The property is an extension to the property of the expectation of nonnegative variables which itself is a consequence of Fubini's Theorem (see for example Ibe 2014 for this). Let $X \sim \nu$ and write $F_{\nu}$ for the CDF of $\nu$ , then:

$$
\mathbb {E} [ X ] = \int_ {0} ^ {\infty} \left(1 - F _ {\nu} (x)\right) d x - \int_ {- \infty} ^ {0} F _ {\nu} (x) d x.
$$

Now, suppose an auxiliary variable $X'$ is distributed according to the law $\nu'$ . It then follows that

$$
\begin{array}{l} \left| \mathbb {E} [ X ] - \mathbb {E} [ X ^ {\prime} ] \right| = \Big | \int_ {0} ^ {\infty} \big (F _ {\nu^ {\prime}} (x) - F _ {\nu} (x) \big) d x - \int_ {- \infty} ^ {0} \big (F _ {\nu} - F _ {\nu^ {\prime}} (x) \big) d x \Big | \\ = \left| \int_ {- \infty} ^ {\infty} F _ {\nu^ {\prime}} (x) - F _ {\nu} (x) d x \right| \\ \leq \int_ {- \infty} ^ {\infty} \left| F _ {\nu^ {\prime}} (x) - F _ {\nu} (x) \right| d x \\ = w _ {1} (\nu , \nu^ {\prime}), \\ \end{array}
$$

where the last step was obtained by a change of variables in the definition of the 1-Wasserstein distance:

$$
\begin{array}{l} w _ {1} (\nu , \nu^ {\prime}) = \int_ {0} ^ {1} | F _ {\nu} ^ {- 1} (\tau) - F _ {\nu^ {\prime}} ^ {- 1} (\tau) | d \tau \\ = \int_ {\mathbb {R}} | F _ {\nu} (x) - F _ {\nu^ {\prime}} (x) | d x. \\ \end{array}
$$

The result of Proposition 2 is obtained by rearranging.

# A.3 PROOF OF THEOREM 3

Before stating the proof of Theorem 3, we formalize the notion of a pushforward distribution which will be useful in a more explicit description of the distributional Bellman operator $\mathcal{T}^{\pi}$ . Our notation here follows the detailed exposition by Bellemare et al. (2023).

Definition 5 For a function $f: \mathbb{R} \to \mathbb{R}$ and a random variable $Z$ with distribution $\nu = \mathcal{D}(Z)$ , $\nu \in \mathcal{P}(\mathbb{R})$ , the pushforward distribution $f_{\#}\nu \in \mathcal{P}(\mathbb{R})$ of $\nu$ through $f$ is defined as

$$
f _ {\#} \nu (B) = \nu (f ^ {- 1} (B)), \quad \forall B \in \mathcal {B} (\mathbb {R}),
$$

where B are the Borel subsets of R.

Equivalently to Definition 5, we may write $f_{\#} \nu = \mathcal{D}(f(Z))$ . By defining a bootstrap transformation $b_{r,\gamma} : \mathbb{R} \to \mathbb{R}$ with $b_{r,\gamma} = r + \gamma x$ , we can state a more explicit definition of the distributional Bellman operator $\mathcal{T}^{\pi}$ according to Definition 6.

Definition 6 [Distributional Bellman Operator (Bellemare et al., 2017)] The distributional Bellman operator $\mathcal{T}^{\pi}:\mathcal{P}(\mathbb{R})^{S\times\mathcal{A}}\to\mathcal{P}(\mathbb{R})^{S\times\mathcal{A}}$ is given by

$$
\left(\mathcal {T} ^ {\pi} \eta\right) (s, a) = \mathbb {E} \big [ \big (b _ {R _ {0}, \gamma} \big) _ {\#} \eta (S _ {1}, A _ {1}) \big | S _ {0} = s, A _ {0} = a \big ],
$$

where $S_{1}\sim P(\cdot |S_{0} = s,A_{0} = a),\quad A_{1}\sim \pi (\cdot |S_{1}).$

Lemma 7 Let $(b_{r,\gamma})_{\#}\nu \in \mathcal{P}(\mathbb{R})$ be the pushforward distribution of $\nu \in \mathcal{P}(\mathbb{R})$ through $b_{r,\gamma}:\mathbb{R}\to \mathbb{R}$ . Then we have for two distributions $\nu, \nu'$ and the 1-Wasserstein distance $w_{1}$ that

$$
w _ {1} \big ((b _ {r, \gamma}) _ {\#} \nu , (b _ {r, \gamma}) _ {\#} \nu^ {\prime} \big) = \gamma w _ {1} (\nu , \nu^ {\prime}).
$$

Proof. The proof follows from the definition of the 1-Wasserstein distance. Let $Z \sim \nu$ and $Z' \sim \nu'$ be two independent random variables, then

$$
\begin{array}{l} w _ {1} \big ((b _ {r, \gamma}) _ {\#} \nu , (b _ {r, \gamma}) _ {\#} \nu^ {\prime} \big) = w _ {1} \big (\mathcal {D} (r + \gamma Z), \mathcal {D} (r + \gamma Z ^ {\prime}) \big) \\ = \int_ {0} ^ {1} \left| F _ {(b _ {0, \gamma}) _ {\#} \nu} ^ {- 1} (\tau) - F _ {(b _ {0, \gamma}) _ {\#} \nu^ {\prime}} ^ {- 1} (\tau) \right| d \tau \\ = | \gamma | w _ {1} (\nu , \nu^ {\prime}). \\ \end{array}
$$

Theorem 3 Let $\hat{\eta}(s,a)\in\mathcal{P}(\mathbb{R})$ be an estimate of the true return distribution $\eta^{\pi}(s,a)\in\mathcal{P}(\mathbb{R})$ , and denote the projection mixture operator $\Omega_{M}:\mathcal{P}(\mathbb{R})\to\mathcal{F}_{E}$ with members $\Pi_{i}$ and bounding moduli $c_{i}$ and $\bar{c}_{p}$ as defined in Proposition 1. Furthermore, assume $\Omega_{M}\mathcal{T}^{\pi}$ is a contraction mapping with fixed point $\eta_{E}^{\pi}$ . We then have for all $(s,a)\in\mathcal{S}\times\mathcal{A}$

$$
w _ {1} \big (\hat {\eta}, \eta_ {E} ^ {\pi} \big) (s, a) \leq w _ {1} \big (\hat {\eta}, \Omega_ {M} \mathcal {T} ^ {\pi} \hat {\eta} \big) (s, a) + \bar {c} _ {1} \gamma \mathbb {E} \big [ w _ {1} \big (\hat {\eta}, \eta_ {E} ^ {\pi} \big) (S _ {1}, A _ {1}) | S _ {0} = s, A _ {0} = a \big ],
$$

where $S_{1}\sim P(\cdot |S_{0} = s,A_{0} = a)$ and $A_{1}\sim \pi (\cdot |S_{1})$

Proof. Since $\eta_E^\pi (s,a)$ is the fixed point of the combined operator $\Omega_M\mathcal{T}^{\pi}$ , we have that $\Omega_M\mathcal{T}^\pi \eta_E^\pi (s,a) = \eta_E^\pi (s,a)$ . From the triangle inequality it follows that

$$
w _ {1} (\hat {\eta}, \eta_ {E} ^ {\pi}) (s, a) \leq w _ {1} (\hat {\eta}, \Omega_ {M} \mathcal {T} ^ {\pi} \hat {\eta}) (s, a) + w _ {1} \left(\Omega_ {M} \mathcal {T} ^ {\pi} \hat {\eta}, \Omega_ {M} \mathcal {T} ^ {\pi} \eta_ {E} ^ {\pi}\right) (s, a). \tag {14}
$$

Furthermore, for the second term on the r.h.s. in Eq. 14 the following holds:

$$
\begin{array}{l} w _ {1} \big (\Omega_ {M} \mathcal {T} ^ {\pi} \hat {\eta}, \Omega_ {M} \mathcal {T} ^ {\pi} \eta_ {E} ^ {\pi} \big) (s, a) = w _ {1} \big (\frac {1}{M} \sum_ {i = 1} ^ {M} \Pi_ {i} \mathcal {T} ^ {\pi} \hat {\eta}, \frac {1}{M} \sum_ {i = 1} ^ {M} \Pi_ {i} \mathcal {T} ^ {\pi} \eta_ {E} ^ {\pi} \big) (s, a) \\ \leq \frac {1}{M} \sum_ {i = 1} ^ {M} c _ {i} w _ {1} \left(\mathcal {T} ^ {\pi} \hat {\eta}, \mathcal {T} ^ {\pi} \eta_ {E} ^ {\pi}\right) (s, a) \\ = \bar {c} _ {1} w _ {1} \big (\mathcal {T} ^ {\pi} \hat {\eta}, \mathcal {T} ^ {\pi} \eta_ {E} ^ {\pi} \big) (s, a). \\ \end{array}
$$

Under slight abuse of the assumptions in Section 3, we here consider an immediate reward distribution with finite support on $\mathcal{R}$ to simplify the following derivation. In this case, we can write out the expectation in Definition 6 as

$$
\big (\mathcal {T} ^ {\pi} \hat {\eta} \big) (s, a) = \sum_ {r \in \mathcal {R}} \sum_ {s ^ {\prime} \in \mathcal {S}} \sum_ {a ^ {\prime} \in \mathcal {A}} P r (R _ {0} = r, A _ {1} = a ^ {\prime}, S _ {1} = s ^ {\prime} | S _ {0} = s, A _ {0} = a) \big ((b _ {r, \gamma}) _ {\#} \hat {\eta} (s ^ {\prime}, a ^ {\prime}) \big),
$$

where $Pr(\cdot)$ is the joint probability distribution given by the transition kernel $P(\cdot | s, a)$ , the immediate reward distribution $\mathcal{R}(\cdot | s, a)$ , and the policy $\pi(\cdot | S')$ . Thus, by Lemma 4 and Lemma 7 it follows that

$$
\begin{array}{l} \bar {c} _ {1} w _ {1} \left(\mathcal {T} ^ {\pi} \hat {\eta}, \mathcal {T} ^ {\pi} \eta_ {E} ^ {\pi}\right) (s, a) \\ \leq \bar {c} _ {1} \mathbb {E} \left[ w _ {1} \left(\left(b _ {R _ {0}, \gamma}\right) _ {\#} \hat {\eta} \left(S _ {1}, A _ {1}\right), \left(b _ {R _ {0}, \gamma}\right) _ {\#} \eta_ {E} ^ {\pi} \left(S _ {1}, A _ {1}\right)\right) \mid S _ {0} = s, A _ {0} = a \right] \\ = \bar {c} _ {1} \gamma \mathbb {E} \left[ w _ {1} (\hat {\eta}, \eta_ {E} ^ {\pi}) (S _ {1}, A _ {1}) \mid S _ {0} = s, A _ {0} = a \right], \\ \end{array}
$$

where $S_{1} \sim P(\cdot | S_{0} = s, A_{0} = a)$ and $A_{1} \sim \pi(\cdot | S')$ . The proof is completed by rearranging.

# A.4 RESIDUAL EPISTEMIC UNCERTAINTY

Due to a limitation to finite-dimensional representations and the use of varying projections, our algorithm incurs residual approximation errors which may not vanish even in convergence. In the context of epistemic uncertainty quantification, this is unfortunate as it can frustrate exploration or lead to overconfident predictions. Specifically, the undesired properties are twofold: 1) Even in convergence, the fixed point $\eta_{E}^{\pi}$ does not equal the true return distribution (bias). 2) Even in the fixed point $\eta_{E}^{\pi}$ , the ensemble disagreement $w_{avg}$ does not vanish. Often, however, we may be able to upper bound and control the error incurred due to the projections $\Pi_{i}$ . In this case, Propositions 8 and 9 provide upper bounds on both types of errors as a function of bounded projection errors.

Proposition 8 Let $\Omega_M$ be a projection mixture operator with individual projections $\Pi_i$ defined as in Eq. (7). Further, assume each projection $\Pi_i$ is upper bounded by $w_p(\Pi_i\nu, \nu) \leq d_i$ for some $p \in [1, \infty)$ . Then, the $p$ -Wasserstein distance between the fixed point $\eta_E^\pi(s, a) = \Omega_M \mathcal{T}^\pi \eta_E^\pi(s, a)$ and the true return distribution $\eta^\pi(s, a) = \mathcal{T}^\pi \eta^\pi(s, a)$ satisfies

$$
w _ {p} (\eta_ {E} ^ {\pi}, \eta^ {\pi}) (s, a) \leq \frac {\bar {d} _ {p}}{1 - \bar {c} _ {p} \gamma} \quad \forall (s, a) \in \mathcal {S} \times \mathcal {A}, \qquad w h e r e \qquad \bar {d} _ {p} = (\sum_ {i = 1} ^ {M} \frac {1}{M} d _ {i} ^ {p}) ^ {1 / p}.
$$

Proof. To show the desired property, we will make use of Proposition 1 and Lemma 4. We omitted the dependency on $(s, a)$ in this section for brevity. It follows then from the triangle inequality that

$$
\begin{array}{l} w _ {p} \left(\eta_ {E} ^ {\pi}, \eta^ {\pi}\right) \leq w _ {p} \left(\Omega_ {M} \mathcal {T} ^ {\pi} \eta_ {E} ^ {\pi}, \Omega_ {M} \eta^ {\pi}\right) + w _ {p} \left(\Omega_ {M} \eta^ {\pi}, \eta^ {\pi}\right) \\ = w _ {p} (\Omega_ {M} \mathcal {T} ^ {\pi} \eta_ {E} ^ {\pi}, \Omega_ {M} \mathcal {T} ^ {\pi} \eta^ {\pi}) + w _ {p} (\Omega_ {M} \eta^ {\pi}, \eta^ {\pi}) \\ \leq \bar {c} _ {p} \gamma w _ {p} (\eta_ {E} ^ {\pi}, \eta^ {\pi}) + w _ {p} (\frac {1}{M} \sum_ {i = 1} ^ {M} \Pi_ {i} \eta^ {\pi}, \eta^ {\pi}) \\ \leq \bar {c} _ {p} \gamma w _ {p} (\eta_ {E} ^ {\pi}, \eta^ {\pi}) + (\frac {1}{M} \sum_ {i = 1} ^ {M} w _ {p} ^ {p} (\Pi_ {i} \eta^ {\pi}, \eta^ {\pi})) ^ {1 / p}. \\ \end{array}
$$

Per the assumption of Proposition 8 and by rearranging we obtain the desired result.

Proposition 9 Let $w_{avg}$ be the average ensemble disagreement given by $\frac{1}{M(M-1)}\sum_{i,j=1}^{M}w_{p}(\hat{\eta}_{i},\hat{\eta}_{j})$ and assume individual projections $\Pi_{i}$ are bounded by $w_{p}(\Pi_{i}\nu,\nu)\leq d_{i}$ . For an ensemble E whose mixture distribution equals exactly the fixed point $\eta_{E}^{\pi}(s,a)=\Omega_{M}\mathcal{T}^{\pi}\eta_{E}^{\pi}(s,a)$ , the average ensemble disagreement $w_{avg}$ satisfies the inequality

$$
w _ {a v g} (s, a) \leq \frac {2 M}{M - 1} \bar {d} \quad \forall (s, a) \in \mathcal {S} \times \mathcal {A}, \quad w h e r e \quad \bar {d} = \frac {1}{M} \sum_ {i = 1} ^ {M} d _ {i}.
$$

Proof. In the fixed point $\eta_E^\pi (s,a) = \Omega_M\mathcal{T}^\pi \eta_E^\pi (s,a)$ , the distributional error estimated by $w_{\mathrm{avg}}(s,a)$ does not vanish, unlike the ground truth error $w_{1}\big(\eta_{E}^{\pi},\Omega_{M}\mathcal{T}^{\pi}\eta_{E}^{\pi}\big)(s,a) = 0$ . The shown property upper bounds this mismatch and is a direct consequence of the assumption $w_{p}(\Pi_{i}\nu ,\nu)\leq d_{i}$ which postulates an upper bound on the error introduced by the projection $\Pi_i$ in terms of the $p$ -Wasserstein distance. The average disagreement is given by

$$
w _ {\text { avg }} (s, a) = \frac {1}{M (M - 1)} \sum_ {i, j = 1} ^ {M} w _ {p} (\hat {\eta} _ {i}, \hat {\eta} _ {j}) (s, a).
$$

The proof is given by applying the triangle inequality and the assumption of the proposition with

$$
\begin{array}{l} w _ {p} (\hat {\eta} _ {i}, \hat {\eta} _ {j}) = w _ {p} (\Pi_ {i} \eta_ {E} ^ {\pi}, \Pi_ {j} \eta_ {E} ^ {\pi}) \\ \leq w _ {p} (\Pi_ {i} \eta_ {E} ^ {\pi}, \eta_ {E} ^ {\pi}) + w _ {p} (\eta_ {E} ^ {\pi}, \Pi_ {j} \eta_ {E} ^ {\pi}) \\ \leq d _ {i} + d _ {j}. \\ \end{array}
$$

Plugging in and rearranging yields the desired result.

Lemma 10 [Projection error of the categorical projection (Rowland et al., 2018)] For any distribution $\nu \in \mathcal{P}([z_{\mathrm{min}}, z_{\mathrm{max}}])$ with support on the interval $[z_{\mathrm{min}}, z_{\mathrm{max}}]$ and a categorical projection as defined in Eq. (5) with $K$ atoms $z_k \in [z_1, ..., z_K]$ s.t. $z_1 \geq Z_{\mathrm{min}}$ and $z_K \leq z_{\mathrm{max}}$ , the error incurred by the projection $\Pi_C$ is upper bounded in the 1-Wasserstein distance by the identity

$$
w _ {1} (\Pi_ {C} \nu , \nu) \leq \left[ \sup _ {1 \leq k \leq K} (z _ {k + 1} - z _ {k}) \right].
$$

Proof (restated). The proof uses the duality between the 1-Wasserstein distance and the 1-Cramér distance stating

$$
l _ {1} (\nu , \nu^ {\prime}) = \int_ {\mathbb {R}} | F _ {\nu} (x) - F _ {\nu^ {\prime}} (x) | d x = \int_ {0} ^ {1} | F _ {\nu} ^ {- 1} (\tau) - F _ {\nu^ {\prime}} ^ {- 1} (\tau) | d \tau = w _ {1} (\nu , \nu^ {\prime}),
$$

and can be obtained by a change of variables. The $l_{1}$ formulation simplifies the analysis of the categorical projection, yielding

$$
\begin{array}{l} w _ {1} (\Pi_ {C} \nu , \nu) = \int_ {\mathbb {R}} | F _ {\Pi_ {C} \nu} (x) - F _ {\nu} (x) | d x \\ \leq \sum_ {k = 1} ^ {K - 1} (z _ {k + 1} - z _ {k}) | F _ {\Pi_ {C} \nu} (z _ {k}) - F _ {\nu} (z _ {k}) | \\ \leq \sum_ {k = 1} ^ {K - 1} (z _ {k + 1} - z _ {k}) | F _ {\nu} (z _ {k + 1}) - F _ {\nu} (z _ {k}) | \\ \leq \left[ \sup _ {1 \leq k \leq K} (z _ {k + 1} - z _ {k}) \right] \sum_ {k = 1} ^ {K - 1} | F _ {\nu} (z _ {k + 1}) - F _ {\nu} (z _ {k}) | \\ \leq \left[ \sup _ {1 \leq k \leq K} \left(z _ {k + 1} - z _ {k}\right) \right]. \\ \end{array}
$$

Lemma 11 [Projection error of the quantile projection (Dabney et al., 2018b)] For any distribution $\nu \in \mathcal{P}([z_{\mathrm{min}}, z_{\mathrm{max}}])$ with support on the interval $[z_{\mathrm{min}}, z_{\mathrm{max}}]$ and a quantile projection defined according to Eq. (6) with $K$ equally weighted locations $\theta_k \in [\theta_1, \dots, \theta_K]$ , the error incurred by the projection $\Pi_Q$ is bounded in the 1-Wasserstein distance by the identity

$$
w _ {1} (\Pi_ {Q} \nu , \nu) \leq \frac {z _ {\mathrm{max}} - z _ {\mathrm{min}}}{K}.
$$

Proof (restated). The projection $\Pi_Q$ is given by

$$
\Pi_ {Q} \nu = \frac {1}{K} \sum_ {k = 1} ^ {K} \delta_ {F _ {\nu} ^ {- 1} (\tau_ {k})}, \quad \text { where } \tau_ {k} = \frac {2 k - 1}{2 K}.
$$

The desired identity $w_{1}(\Pi_{Q}\nu, \nu)$ is accordingly given by the continuous integral

$$
w _ {1} (\Pi_ {Q} \nu , \nu) = \int_ {0} ^ {1} | F _ {\Pi_ {Q} \nu} ^ {- 1} (\tau) - F _ {\nu} ^ {- 1} (\tau) | d \tau ,
$$

and can be rewritten in terms of a sum of piecewise expectations

$$
w _ {1} \left(\Pi_ {Q} \nu , \nu\right) = \sum_ {k = 1} ^ {K} \frac {1}{K} \mathbb {E} _ {X \sim \nu} \left[ \left| X - F _ {\nu} ^ {- 1} \left(\frac {2 k - 1}{2 K}\right) \right| \mid F _ {\nu} ^ {- 1} \left(\frac {k - 1}{K}\right) <   X \leq F _ {\nu} ^ {- 1} \left(\frac {k}{K}\right) \right].
$$

From this, it follows that

$$
w _ {1} (\Pi_ {Q} \nu , \nu) \leq \frac {1}{K} (F _ {\nu} ^ {- 1} (1) - F _ {\nu} ^ {- 1} (0))
$$

$$
\leq \frac {z _ {\mathrm{max}} - z _ {\mathrm{min}}}{K}.
$$

Corollary 12 Let $\eta_E^\pi (s,a)$ be the fixed point return distribution for an ensemble of the categorical and quantile projections with the mixture operator $\Omega_M\eta (s,a) = 1 / 2\Pi_Q\eta (s,a) + 1 / 2\Pi_C\eta (s,a)$ . Furthermore, suppose the return distribution $\eta_E^\pi (s,a)$ has bounded support on the interval $(R_{\mathrm{max}} - R_{\mathrm{min}}) / (1 - \gamma)$ where $R_{\mathrm{max}}$ and $R_{\mathrm{min}}$ denote the maximum and minimum immediate reward of the MDP. The average ensemble disagreement $w_{avg}(s,a)$ is then bounded by

$$
w _ {a v g} (s, a) \leq \frac {4 (R _ {\max} - R _ {\min})}{(1 - \gamma) K}.
$$

Proof. The result follows straightforwardly from Proposition 9 and Lemmas 10, 11.

Table 1: Hyperparameter search space for bsuite 

<table><tr><td>Hyperparameter</td><td>Values</td></tr><tr><td>Neural net architecture</td><td>[[64, 64], [128, 128], [512]]</td></tr><tr><td>Learning rate</td><td> $[5 \times 10^{-5}, 1 \times 10^{-4}, 5 \times 10^{-4}, 1 \times 10^{-3}]$ </td></tr><tr><td>Prior function scale</td><td>[0.0, 5.0, 20.0]</td></tr><tr><td>Heads K</td><td>[51, 101]</td></tr><tr><td>Initial bonus β</td><td>[0.5, 5.0, 50.0]</td></tr></table>

Table 2: Hyperparameter search space for VizDoom 

<table><tr><td>Hyperparameter</td><td>Values</td></tr><tr><td>Learning rate</td><td> $[1.25 \times 10^{-5}, 2.5 \times 10^{-5}, 3.75 \times 10^{-5}, 5 \times 10^{-5}, 6.25 \times 10^{-5}, 7.5 \times 10^{-5}]$ </td></tr><tr><td>Prior function scale</td><td> $[1.0, 3.0, 5.0]$ </td></tr><tr><td>Initial bonus  $\beta$ </td><td> $[0.05, 0.1, 0.5, 1.0, 5.0]$ </td></tr></table>

# A.5 THE CATEGORICAL PROJECTION

The full definition of the categorical (or also Cramér) projection as stated by Rowland et al. (2018) is given below.

Definition 13 [Categorical projection (Rowland et al., 2018)] For a set of fixed locations $z_1, \ldots, z_K$ where $z_1 < z_2 < \ldots < z_K$ , let $h_{z_k} : \mathbb{R} \to [0,1]$ be the hat function centered around $z_k$ for $k = 1, \ldots, K$ given by

$$
h _ {z _ {k}} (x) = \left\{ \begin{array}{l l} \frac {z _ {k + 1} - x}{z _ {k + 1} - z _ {k}} & \text {   for   } x \in [ z _ {k}, z _ {k + 1} ] \quad \text { and   } 1 \leq k <   K, \\ \frac {x - z _ {k - 1}}{z _ {k} - z _ {k - 1}} & \text {   for   } x \in [ z _ {k - 1}, z _ {k} ] \quad \text { and   } 1 <   k \leq K, \\ 1 & \text {   for   } x \leq z _ {1} \quad \text { and   } k = 1, \\ 1 & \text {   for   } x \geq z _ {K} \quad \text { and   } k = K, \\ 0 & \text {   otherwise. } \end{array} \right.
$$

Furthermore, let the categorical representation $\mathcal{F}_C$ be defined as a finite mixture of Dirac deltas $\mathcal{F}_C = \{\sum_{k=1}^{K}\theta_k\delta_{z_k}|\theta_k\geq 0,\sum_{k=1}^{K}\theta_k = 1\}$ . The categorical projection operator $\Pi_C:\mathcal{P}(\mathbb{R})\to \mathcal{F}_C$ of a distribution $\nu \in \mathcal{P}(\mathbb{R})$ is then defined as

$$
\Pi_ {C} \nu = \sum_ {k = 1} ^ {K} \mathbb {E} _ {\omega \sim \nu} [ h _ {z _ {k}} (\omega) ] \delta_ {z _ {k}}.
$$

# B EXPERIMENTAL DETAILS

We provide a detailed exposition of our experimental setup, including the hyperparameter search procedure, hyperparameter settings, algorithmic details, and the full bsuite experimental results.

# B.1 HYPERPARAMETER SETTINGS

In our experiments, we aimed to keep most hyperparameters between different implementations equal to maintain comparability between the analyzed methods. Algorithm-specific hyperparameters were optimized over a search space of hyperparameters using Optuna (Akiba et al., 2019). The total search space for bsuite and VizDoom are given in Table 1 and Table 2 respectively, where the Heads K parameter only applies to distributional algorithms. C51 requires us to define return ranges, which we defined manually and can be found in the online code repository. All algorithms use the Adam optimizer (Kingma and Ba, 2015).

Bsuite. For bsuite, the hyperparameter search was conducted on a subselection of environments of the bsuite, as shown in Table 3. For each environment, we evaluate a set of hyperparameters by

Table 3: Hyperparameter search environments 

<table><tr><td>Environment ID</td><td>Horizon in no. of episodes</td><td>Scoring function f</td></tr><tr><td>deep_sea/20</td><td>500</td><td> $\sum_{(s,a)} \mathbb{1}_{\text{visited}} (s, a)$ </td></tr><tr><td>deep_sea_stochastic/20</td><td>1500</td><td> $\sum_{(s,a)} \mathbb{1}_{\text{visited}} (s, a)$ </td></tr><tr><td>mountain_car/19</td><td>100</td><td> $\sum_{0}^{t} (-1)$ </td></tr></table>

means of a scoring function. A particular set of hyperparameters is evaluated every T/5 episodes with a maximum training horizon of T episodes. The “continuous” scoring functions make the hyperparameter search more amenable to pruning, for which we use the median pruner of Optuna, reducing the computational burden of the combinatorial search space significantly.

Here, $\sum_{(s,a)}\mathbb{1}_{\text{visited}}(s,a)$ is the count of visited state-action tuples and $\sum_{0}^{t}(-1)$ is simply the negative number of total environment interactions. For every hyperparameter configuration $\zeta_{i}$ , the scores $f(\zeta_{i})$ are calibrated to facilitate a meaningful comparison between different environments. The calibrated score function we use is given by

$$
f _ {c} (\zeta_ {i}) = \exp \left(0. 6 9 3 \frac {f (\zeta_ {i}) - \mu_ {\zeta}}{\sup _ {i} f (\zeta_ {i}) - \mu_ {\zeta}}\right), \tag {15}
$$

where $\mu_{\zeta}$ is the average score of all hyperparameter configurations $\mu_{\zeta} = \sum_{i}^{N} 1/N f(\zeta_{i})$ , and $\sup_{i} f(\zeta_{i})$ is the maximal score achieved. The calibration function in Eq. (15) was chosen heuristically to have an intuitive interpretation: it assigns a score of 1 to the best-performing hyperparameter configuration, 0.5 to configurations that achieve exact average performance, and decays exponentially according to score. The final score assigned to a hyperparameter configuration $\zeta_{i}$ is the sum of all scores of the tested environments. Table 4 shows the full set of hyperparameters used for every algorithm.

VizDoom. For the VizDoom domain, the hyperparameter search was conducted on the MyWayHomeSparse-v0 variation with a training budget of 5 million frames, where final configurations were chosen by achieved return at the end of training. Due to the sparsity of the problem, we did not make use of a pruning algorithm. The specific difference between the different variations of the VizDoom environment MyWayHome are shown in Fig. 6, where the sparsity of the problem is increased by changing the agents spawning location to a room further from the goal position. The network architecture is based to a large extent on the rainbow network proposed by Schmidt and Schmied (2021) who in turn base their architecture on IMPALA (Espeholt et al., 2018). The specific algorithm configuration for VizDoom is given in Table 5 with a schematic of the network architecture shown in Fig. 8. Table 6 shows our preprocessing pipeline used for the VizDoom environments.

![](images/2d159bbf118a5beee5dc72d4531f7dbbd9baa8b1939c922b0718d2095ab14827.jpg)

<details>
<summary>text_image</summary>

Room 13
sparse
Goal
Room 17
very sparse
</details>

Figure 6: Map for the VizDoom MyWayHome environment. Agents are spawned in the sparse and very sparse locations to vary the exploration difficulty.

# B.2 IMPLEMENTATION DETAILS

Parametric model. Our parametric model is a mixture distribution $\eta_{E,\theta}$ parametrized by $\theta$ . We construct $\eta_{E,\theta}$ as an equal mixture between a quantile and a categorical representation, each parametrized through a NN with K output logits where we use the notation $\theta_{ik}$ to mean the k-th logit of the network parametrized by the parameters $\theta_{i}$ of the i-th model in the ensemble. We consider a sample transition $(s,a,r,s',a')$ where $a'$ is chosen greedily according to $\mathbb{E}_{Z\sim\eta_{E,\theta}(s',a')}[Z]$ . Dependencies

Table 4: Hyperparameter settings bsuite 

<table><tr><td>Hyperparameter</td><td>BDQN+P</td><td>DLTV</td><td>IDS</td><td>PE-DQN</td></tr><tr><td>Net architecture</td><td>[64, 64]</td><td>[512]</td><td>[64, 64] / [512]</td><td>[512]</td></tr><tr><td>Adam Learning rate</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td><td> $10^{-3}/5 \times 10^{-4}$ </td><td> $5 \times 10^{-4}$ </td></tr><tr><td>Prior function scale</td><td>5.0</td><td>20.0</td><td>20.0 / 5.0</td><td>20.0 / 0.0</td></tr><tr><td>Heads K</td><td>1</td><td>101</td><td>1 / 101</td><td>101/101</td></tr><tr><td>Ensemble size</td><td>20</td><td>1</td><td>20/1</td><td>2/2</td></tr><tr><td>Initial bonus  $\beta_{init}$ </td><td>n/a</td><td>5.0</td><td>5.0</td><td>5.0</td></tr><tr><td>Final bonus  $\beta_{final}$ </td><td>n/a</td><td>n/a</td><td>5.0</td><td>5.0</td></tr><tr><td>Bonus decay (in eps)</td><td>n/a</td><td> $10^{3}/N_{episodes}$ </td><td> $0.33 \times N_{episodes}$ </td><td> $0.33 \times N_{episodes}$ </td></tr><tr><td>Discount</td><td></td><td></td><td>0.99</td><td></td></tr><tr><td>Buffer size</td><td></td><td></td><td>10,000</td><td></td></tr><tr><td>Adam epsilon</td><td></td><td></td><td>0.001/batch size</td><td></td></tr><tr><td>Initialization</td><td></td><td></td><td>He truncated normal (He et al., 2015)</td><td></td></tr><tr><td>Update frequency</td><td></td><td></td><td>1</td><td></td></tr><tr><td>Target update step size</td><td></td><td></td><td>1.0</td><td></td></tr><tr><td>Target update frequency</td><td></td><td></td><td>4</td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>128</td><td></td></tr></table>

Table 5: Hyperparameter settings VizDoom 

<table><tr><td>Hyperparameter</td><td>BDQN+P</td><td>DLTV</td><td>IDS</td><td>PE-DQN</td></tr><tr><td>Adam Learning rate</td><td> $2.5 \times 10^{-5}$ </td><td> $7.5 \times 10^{-5}$ </td><td> $2.5 \times 10^{-5}$ </td><td> $6.25 \times 10^{-5}$ </td></tr><tr><td>Prior function scale</td><td>1.0</td><td>3.0</td><td>1.0</td><td>3.0</td></tr><tr><td>Heads K</td><td>1</td><td>101</td><td>1 / 101</td><td>101/101</td></tr><tr><td>Ensemble size</td><td>10</td><td>1</td><td>10/1</td><td>2/2</td></tr><tr><td>Initial bonus  $\beta_{init}$ </td><td>n/a</td><td>0.5</td><td>0.1</td><td>5.0</td></tr><tr><td>Final bonus  $\beta_{final}$ </td><td>n/a</td><td>n/a</td><td>0.01</td><td>0.01</td></tr><tr><td>Bonus decay (in frames)</td><td>n/a</td><td> $10^{3}/N_{frames}$ </td><td> $0.33 \times N_{frames}$ </td><td> $0.33 \times N_{frames}$ </td></tr><tr><td>Loss function</td><td>Huber</td><td>QR-Huber</td><td>Huber/C51</td><td>QR-Huber/C51</td></tr><tr><td>Initial  $\epsilon$  in  $\epsilon$ -greedy</td><td></td><td></td><td>1.0</td><td></td></tr><tr><td>Final  $\epsilon$  in  $\epsilon$ -greedy</td><td></td><td></td><td>0.01</td><td></td></tr><tr><td> $\epsilon$  decay time</td><td></td><td></td><td>500,000</td><td></td></tr><tr><td>Training starts</td><td></td><td></td><td>100,000</td><td></td></tr><tr><td>Discount</td><td></td><td></td><td>0.997</td><td></td></tr><tr><td>Buffer size</td><td></td><td></td><td>1,000,000</td><td></td></tr><tr><td>Batch size</td><td></td><td></td><td>512</td><td></td></tr><tr><td>Parallel Envs</td><td></td><td></td><td>32</td><td></td></tr><tr><td>Adam epsilon</td><td></td><td></td><td>0.005/batch size</td><td></td></tr><tr><td>Initialization</td><td></td><td></td><td>He uniform (He et al., 2015)</td><td></td></tr><tr><td>Gradient clip norm</td><td></td><td></td><td>10</td><td></td></tr><tr><td>Regularization</td><td></td><td></td><td>spectral normalization</td><td></td></tr><tr><td>Double DQN</td><td></td><td></td><td>Yes</td><td></td></tr><tr><td>Update frequency</td><td></td><td></td><td>1</td><td></td></tr><tr><td>Target update step size</td><td></td><td></td><td>1.0</td><td></td></tr><tr><td>Target update frequency</td><td></td><td></td><td>8000</td><td></td></tr><tr><td>PER  $\beta_{0}$ </td><td></td><td></td><td>0.45</td><td></td></tr><tr><td>n-step returns</td><td></td><td></td><td>10</td><td></td></tr></table>

Table 6: VizDoom Preprocessing 

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Grayscale</td><td>Yes</td></tr><tr><td>Frame-skipping</td><td>No</td></tr><tr><td>Frame-stacking</td><td>6</td></tr><tr><td>Resolution</td><td>42 × 42</td></tr><tr><td>Max. Episode Length</td><td>2100</td></tr></table>

on $(s, a)$ are dropped for conciseness by writing $\theta_{ik}(s, a) = \theta_{ik}$ and $\theta_{ik}(s', a') = \theta_{ik}'$ . The full mixture model $\eta_{E,\theta}$ is then given by

$$
\eta_ {E, \theta} = \frac {1}{2} \sum_ {i = 1} ^ {M = 2} \sum_ {k = 1} ^ {K} p (\theta_ {i k}) \delta_ {z (\theta_ {i k})}, \quad \text { with } \quad \begin{array}{l} p (\theta_ {1 k}) = \frac {1}{K}, z (\theta_ {1 k}) = \theta_ {1 k}, \\ p (\theta_ {2 k}) = \sigma (\theta_ {2 k}), z (\theta_ {2 k}) = z _ {k}, \end{array} \tag {16}
$$

where $\sigma(x_{i}) = e^{x_{i}} / \sum_{j} e^{x_{j}}$ is the softmax transfer function. Consequently, this representation comprises a total of 2K atoms, K of which parametrize locations in the quantile model, and the remaining K parametrizing probabilities in the categorical representation. The losses used for each projection method are as provided in the main text.

Distributional estimation of bonuses. For the parametric bonus estimate $b_{\phi}(s,a)$ we use the same procedure for learning a distributional projection ensemble as with extrinsic rewards. Note that it is not necessary for our method to learn a distributional estimate of the bonus but we find that diverse projection ensembles are good value learners in general and simply reuse the existent function approximation machinery for an intrinsic reward instead of the extrinsic reward. We thus have a model of parameters $\phi$ trained with an alternate tuple $(s,a,w_{\mathrm{avg}},s',a'_{\epsilon})$ , where we replaced the immediate reward with the ensemble disagreement $w_{avg}$ and $a'_{\epsilon}$ is an exploratory action chosen according to the rule

$$
a _ {\epsilon} ^ {\prime} = \underset {a \in \mathcal {A}} {\arg \max} \left(\mathbb {E} _ {Z \sim \eta_ {E, \theta} (s, a)} [ Z ] + \beta b _ {\phi} (s, a)\right), \quad \text { where } \quad b _ {\phi} (s, a) = \mathbb {E} _ {B \sim \eta_ {E, \phi} (s, a)} [ B ]. \tag {17}
$$

Here, $\beta$ is a hyperparameter to control the policy's drive towards exploratory actions.

Pseudocode. We provide pseudocode for a basic version of PE-DQN where we have simplified details such as the previously described distributional estimation of $b_{\phi}(s,a)$ , prioritized replay, double Q-learning, and prior functions for clarity.

Randomized prior functions are added to all baselines and PE-DQN. Specifically, we add the output of a fixed, randomly initialized neural network of the same architecture as the main net, scaled by a hyperparameter, to the main network's logits. In the case of C51, the prior function is added pre softmax. To the best of our knowledge, DLTV-QR does not use prior functions in its original formulation but we find it to be crucial in improving exploration performance. Fig. 7 (b) shows an experiment assessing the exploration performance of DLTV-QR with randomized prior functions and prior scale 20 (DLTV [rpf20]) compared to the vanilla implementation without priors (DLTV [rpf0]).

Information-gain in our IDS implementation for bsuite is computed in a slightly modified way compared to the vanilla version. Nikolov et al. (2019) compute the information gain function $I(s, a)$ with

$$
I (s, a) = \log \left(1 + \frac {\sigma^ {2} (s , a)}{\rho^ {2} (s , a)}\right) + \epsilon_ {2},
$$

where $\sigma^{2}(s,a)$ is the empirical variance of BDQN+P predictions, $\epsilon_{2}=1\times10^{-5}$ is a zero-division protection, and $\rho^{2}(s,a)$ is the clipped action-space normalized return variance

$$
\rho (s, a) ^ {2} = \max \left(\frac {\operatorname{Var} (Z (s , a))}{\frac {1}{| \mathcal {A} |} \sum_ {a \in \mathcal {A}} \operatorname{Var} (Z (s , a))}, 0. 2 5\right). \tag {18}
$$

Algorithm 1 PE-DQN   
1: quantile model with parameters $\theta_1$ , target parameters $\tilde{\theta}_1$ , and $K$ heads
2: categorical model with parameters $\theta_2$ , target parameters $\tilde{\theta}_2$ , $K$ heads, and grid $[z_1, \ldots, z_K]$ 3: bonus estimation model with parameters $\phi$ , and target parameters $\tilde{\phi}$ 4: exploration parameter $\beta$ , learning rate $\alpha$ 5: initialize Buffer $\mathcal{B}$ 6: sample initial state $s_0$ 7: for $t = 0, \ldots, T$ do
8:    predict locations $[\theta_{11}, \ldots, \theta_{1K}](s_t, a)$ and probabilities $[\theta_{21}, \ldots, \theta_{2K}](s_t, a)$ 9: $Q(s_t, a) := \frac{1}{2} \sum_{k=1}^{K} \theta_{1k}(s_t, a) \frac{1}{K} + \theta_{2k}(s_t, a) z_k$ 10:    predict bonus $b_\phi(s_t, a)$ 11: $a_t \leftarrow \arg \max_{a \in \mathcal{A}} \{Q(s_t, a) + \beta b_\phi(s_t, a)\}$ 12:    for $j = 0, \ldots, N_{\text{trainsteps}}$ do
13:    sample transition tuple $(s_j, a_j, r_j, s_j') \sim \mathcal{B}$ 14:    predict locations $[\theta_{11}, \ldots, \theta_{1K}](s_j, a_j)$ and probabilities $[\theta_{21}, \ldots, \theta_{2K}](s_j, a_j)$ 15:    predict target locations $[\tilde{\theta}_{11}, \ldots, \tilde{\theta}_{1K}](s_j', a)$ and probabilities $[\tilde{\theta}_{21}, \ldots, \tilde{\theta}_{2K}](s_j', a)$ 16: $Q(s_j', a) := \frac{1}{2} \sum_{k=1}^{K} \theta_{1k}(s_j', a) \frac{1}{K} + \theta_{2k}(s_j', a) z_k$ 17: $a_j' \leftarrow \arg \max_{a \in \mathcal{A}} \{Q(s_j', a)\}$ 18:    mixture target $\tilde{\eta}_M'\leftarrow \frac{1}{2} \sum_{k=1}^{K} \frac{1}{K} \delta_{r_j + \gamma \tilde{\theta}_{1k}(s_j', a_j')} + \tilde{\theta}_{2k}(s_j', a_j') \delta_{r_j + \gamma z_k}$ 19:    quantile loss $l_1 \leftarrow \mathcal{L}_Q(\theta_1, \tilde{\eta}_M')$ 20:    categorical loss $l_2 \leftarrow \mathcal{L}_C(\theta_2, \tilde{\eta}_M')$ 21:    wasserstein distance $r_{\text{intr}} \leftarrow w_1(\sum_{k=1}^{K} \frac{1}{K} \delta_{\theta_{1k}(s_j, a_j)}, \sum_{k=1}^{K} \theta_{2k}(s_j, a_j)) \delta_{z_k}$ 22:    bonus estimation target $\tilde{b}' \leftarrow r_{\text{intr}} + \gamma b_\tilde{\phi}(s_j', a_j')$ 23:    bonus estimation loss $l_3 \leftarrow MSE(b_\phi(s_j, a_j), \tilde{b}')$ 24: $[\theta_1, \theta_2, \phi]^T \leftarrow [\theta_1, \theta_2, \phi]^T + \alpha \nabla_{\theta_1, \theta_2, \phi}(l_1 + l_2 + l_3)$ 25:    end for
26:    execute $a_t$ and store $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{B}$ 27: end for

Table 7: VizDoom wall clock time comparisons 

<table><tr><td>Environment</td><td>BDQN+P</td><td>DLTV</td><td>IDS</td><td>PE-DQN</td></tr><tr><td>MyWayHome - Dense</td><td>14h 35m</td><td>14h 22m</td><td>16h 49m</td><td>17h 3m</td></tr><tr><td>MyWayHome - Sparse</td><td>14h 29m</td><td>13h 49m</td><td>16h 11m</td><td>16h 11m</td></tr><tr><td>MyWayHome - Very Sparse</td><td>21h 27m</td><td>21h 12m</td><td>23h 3m</td><td>23h 3m</td></tr></table>

$\operatorname{Var}(Z(s, a))$ here is the variance of the distributional estimate provided by C51. We replace the clipping in Eq. (18) by adding a small constant $\epsilon_1 = 1 \times 10^{-4}$ to $\operatorname{Var}(Z(s, a))$ , s.t.

$$
\rho_ {\epsilon} (s, a) ^ {2} = \frac {\operatorname{Var} (Z (s , a)) + \epsilon_ {1}}{\epsilon_ {1} + \frac {1}{| \mathcal {A} |} \sum_ {a \in \mathcal {A}} \operatorname{Var} (Z (s , a))}.
$$

Fig. 7 (b) shows the effect of clipping as in the vanilla version (IDS-C51 [clip]) compared to our variation (IDS-C51 [noclip]) on the deep sea environment.

Intrinsic reward priors are a computational method we implement with PE-DQN, which leverages the fact that we can compute the one-step uncertainty estimate $w_{\mathrm{avg}}(s,a)$ deterministically from a parametric ensemble given a state-action tuple. This obviates the need to learn it explicitly in the bonus estimation step. We thus add $w_{\mathrm{avg}}(s,a)$ automatically to the forward pass of the bonus estimator $b_{\phi}(s,a)$ as a sort of “prior” mechanism according to

$$
b _ {\phi} (s, a) := b _ {\phi} ^ {\text { raw }} (s, a) + w _ {\text { avg }} (s, a),
$$

where $b_{\phi}^{raw}$ is the raw output of the bonus estimator NN of parameters $\phi$ . In the VizDoom environment, we follow the default pipeline suggested by Burda et al. (2019b) and subsequent works (Burda et al., 2019a) that normalize intrinsic rewards by a running estimate of its marginal standard deviation.

Bonus decay is the decaying of the exploratory bonus during action selection. It is well-known that the factor $\beta$ is a sensitive parameter for UCB-type exploration algorithms, enabling efficient exploration when chosen correctly but simultaneously preventing proper convergence when chosen wrongly. Due to the variety of tasks included in the bsuite and VizDoom, we opted for a fixed schedule by which $\beta$ is linearly decayed to 0.0 over one third of the total training horizon. In the bsuite experiments, we apply this schedule to all tested baselines where applicable and chose the initial $\beta_{init}$ value according to the hyperparameter search. Since the decay rate is a central part of the DLTV algorithm, we here do not use our linearly deacying schedule but adopt the original decay rate of $\beta = \beta_{0} * \sqrt{\log(\alpha t)/\alpha t}$ where $\alpha$ is a scaling parameter.

Ensembles and their size are a central parameter in IDS and BDQN+P. For the bsuite experiments, we used a size of 20 as in the implementation by Osband et al. (2020), who find that increasing the ensemble size beyond 20 did not lead to significant performance improvements on the bsuite. Fig. 7(a) shows a comparison of the influence of ensemble size in BDQN+P compared to PE-DQN. For VizDoom, we used 10 models in accordance with Nikolov et al. (2019) for their Atari experiments. Here, we follow the original implementations and let the ensembles used in BDQN+P and IDS-C51 (and also PE-DQN) share a network body for feature extraction to save computation.

Replay buffer In the VizDoom environment, all our algorithms make use of prioritized experience replay (Schaul et al., 2016).

The computational resources we used to conduct the bsuite experiments were supplied by Delft High Performance Computing Centre (DHPC) and Delft Artificial Intelligence Cluster (DAIC). We deployed bsuite environments in 16 parallel jobs to be executed on 8 NVIDIA Tesla V100S 32GB GPUs, 16 Intel XEON E5-6248R 24C 3.0GHz CPUs, and 64GB of memory in total. In this setup, the execution of one seed on the entire suite experiment took approximately 38 hours for DLTV, 72 hours for PE-DQN, and 80 hours for IDS. Due to the narrower network architecture of BDQN+P, we in this case parallelized environments over 64 Intel XEON E5-6248R 24C 3.0GHz CPUs, taking approximately 76 hours wall-clock time for the entire suite. In the VizDoom environments, we deployed 32 parallel environments for each agent on the same hardware. In this case, computation for $10 \times 10^{6}$ took approximately 24 hours per seed per environment and did not differ significantly between any of the tested methods. Table 7 shows the average wall clock time for the VizDoom experiments.

![](images/bc72a0bc757f2fe48b989f98f690c938bb7e36a9f155936bcb8ff48bb568afef.jpg)

<details>
<summary>radar</summary>

| Category       | A: PE-DQN | B: BDQN+P[20] | C: BDQN+P[7] | D: BDQN+P[5] | E: BDQN+P[2] |
| -------------- | --------- | ------------- | ------------ | ------------ | ------------ |
| Exploration    | 80        | 75            | 70           | 65           | 60           |
| Credit Assignment | 70      | 65            | 60           | 55           | 50           |
| Basic          | 90        | 85            | 80           | 75           | 70           |
| Scale          | 75        | 70            | 65           | 60           | 55           |
| Noise          | 60        | 55            | 50           | 45           | 40           |
| Memory         | 50        | 45            | 40           | 35           | 30           |
| Generalization | 40        | 35            | 30           | 25           | 20           |
</details>

![](images/38759104a4caeeabe82b3c35ba63e360c1782dc7e8e1f417ce55101a557ed185.jpg)

<details>
<summary>line</summary>

| x    | IDS-C51 [noclip] - Ours | IDS-C51 [clip] - Vanilla | DLTV-QR [rpf20] - Ours | DLTV-QR [rpf0] - Vanilla |
| ---- | ------------------------ | ------------------------- | ---------------------- | ------------------------- |
| 0.0  | 1.0                      | 1.0                       | 1.0                    | 1.0                       |
| 1.0  | 2.5                      | 1.0                       | 2.5                    | 1.5                       |
| 2.0  | 2.5                      | 1.0                       | 2.5                    | 1.5                       |
| 3.0  | 2.5                      | 1.0                       | 2.5                    | 1.5                       |
</details>

No. episodes (in 1e3)   
Figure 7: (a) Summary of bsuite experiments. Comparison between BDQN+P with different ensemble sizes and PE-DQN (total ensemble size 4). (b) Deep sea comparison between our implementations and vanilla implementations of baseline algorithms. Shown are median state-action visitation counts over number of episodes on the deep sea environment with size 50. Shaded regions represent the interquartile range of 10 seeds. Higher is better.

![](images/8fa9091bd659109cd625716054856e839bf85cd5bf81c3bd23d59f9a7d9cb38a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["K × |A|"] --> B["ReLU"]
    C["K × |A|"] --> B
    D["K × |A|"] --> B
    B --> E["FC 256"]
    E --> F["ReLU"]
    F --> G["Residual Block"]
    G --> H["Max 3 × 3, stride 2"]
    H --> I["Conv. 3 × 3, stride 1"]
    I --> J["/255"]
    J --> K["×6"]
    K --> L["42 × 42"]
    M["× Ensemble Size"] -.-> N["Residual Block"]
    N --> O["Conv. 3 × 3, stride 1"]
    N --> P["Conv. 3 × 3, stride 1"]
    N --> Q["ReLU"]
    Q --> R["+"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#ccf,stroke:#333
    style O fill:#ccf,stroke:#333
    style P fill:#ccf,stroke:#333
    style Q fill:#ccf,stroke:#333
```
</details>

Figure 8: Schematic of the architecture used for VizDoom environments. Based on the architecture used by Espeholt et al. (2018).

![](images/99b2f848875aef4ffd1c1929380b90a06c59a6e3b3b237b6717911200a8d35b8.jpg)  
Figure 9: A comparison of inverse counts (top row), ensemble disagreement (mid row), and bonus estimates (bottom row) on the deep sea environment. t indicates total environment interactions. Each image depicts the state-space of deep sea, where only the lower triangle (including the diagonal) is reachable. For each state, the plotted values indicate the maximum of two actions. At t = 32000, the agent has discovered the goal-state at the bottom right.

# B.3 ADDITIONAL EXPERIMENTAL RESULTS

Fig. 9 illustrates a comparison of the uncertainty estimates used in PE-DQN for the deep sea environment. Every plot shows the entire state-space of the deep sea environment. In deep sea, the agent starts at the top left entry in a matrix and, depending on his action, moves to the left or right column while descending one row. The upper right triangular matrix above the diagonal is thus not reachable to the agent. The goal, i.e., the rewarding final state is located at the bottom right of the matrix.

For different time steps t (total environment interactions) during training, we evaluate the entire state-space and compare three quantities:

- Inverse counts are the inverse of visitations to each state-action $\frac{1}{N(s,a)+0.1}$ . For every state, we plot the maximum of both actions.   
- Ensemble disagreement defined as $w_{\mathrm{avg}}(s, a) = 1 / (M(M - 1)) \sum_{i,j=1}^{M} w_1(\eta_{\theta_i}, \eta_{\theta_j})(s, a)$ . For every state, we plot the maximum of both actions.   
- Bonus estimates $b_{\phi}(s, a)$ as defined in Section 5.1. For every state, we plot the maximum of both actions.

In the top row, the agent has explored an increasing fraction of the state space with increasing time. The number of states with high inverse counts thus decreases. The ensemble disagreement $w_{\mathrm{avg}}(s,a)$ behaves similarly to inverse counts, a result in line with the notion that $w_{\mathrm{avg}}(s,a)$ serves as an estimate of the local TD error $w_{1}(\eta_{E,\theta},\Omega_{M}\hat{\mathcal{T}}^{\pi}\eta_{E,\theta})(s,a)$ , which is expected to decrease with number of visits. In contrast to this, we expect bonus estimates $b_{\phi}(s,a)$ to quantify errors w.r.t the true value, that is $w_{1}(\hat{\eta},\eta^{\pi})(s,a)$ . As a result, $b_{\phi}(s,a)$ should not, for example, vanish prematurely for the initial state at the top left, even after many visitations, since its value can only be assessed upon having explored the entire state space. The bottom row of Fig. 9 is closely in line with this intuition. At t=32000, the agent has discovered the reward at the bottom right.

![](images/566d9eaf50acbcd6094c4eccc24b6edae2c0558cd1edceb28cd82f2e976c5c61.jpg)  
Figure 10: Averaged episodic return for all 23 bsuite tasks.

# B.4 FULL RESULTS OF BSUITE EXPERIMENTS

Fig. 10 shows the averaged undiscounted episodic return for all bsuite tasks. Each curve represents the average over approximately 20 variations of the same task (Osband et al. (2020) provide a detailed account of the task variations) where results were taken from a separate evaluation episode using a greedy action-selection rule. In the “scale” environments, evaluation results were rescaled to the original reward range to maintain a sensible average. Bold titles indicate environments tagged as hard exploration tasks.