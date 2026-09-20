# Harnessing Density Ratios for Online Reinforcement Learning

Philip Amortila $^{*}$ philipa4@illinois.edu

Dylan J. Foster
dylanfoster@microsoft.com

Nan Jiang
nanjiang@illinois.edu

Ayush Sekhari
sekhari@mit.edu

Tengyang Xie
tengyangxie@microsoft.com

# Abstract

The theories of offline and online reinforcement learning, despite having evolved in parallel, have begun to show signs of the possibility for a unification, with algorithms and analysis techniques for one setting often having natural counterparts in the other. However, the notion of density ratio modeling, an emerging paradigm in offline RL, has been largely absent from online RL, perhaps for good reason: the very existence and boundedness of density ratios relies on access to an exploratory dataset with good coverage, but the core challenge in online RL is to collect such a dataset without having one to start.

In this work we show—perhaps surprisingly—that density ratio-based algorithms have online counterparts. Assuming only the existence of an exploratory distribution with good coverage, a structural condition known as coverability (Xie et al., 2023), we give a new algorithm (GLOW) that uses density ratio realizability and value function realizability to perform sample-efficient online exploration. GLOW addresses unbounded density ratios via careful use of truncation, and combines this with optimism to guide exploration.

GLOW is computationally inefficient; we complement it with a more efficient counterpart, HYGLOW, for the Hybrid RL setting (Song et al., 2023) wherein online RL is augmented with additional offline data. HYGLOW is derived as a special case of a more general meta-algorithm that provides a provable black-box reduction from hybrid RL to offline RL, which may be of independent interest.

# 1 Introduction

A fundamental problem in reinforcement learning (RL) is to understand what modeling assumptions and algorithmic principles lead to sample-efficient learning guarantees. Investigation into algorithms for sample-efficient reinforcement learning has primarily focused on two separate formulations: Offline reinforcement learning, where a learner must optimize a policy from logged transitions and rewards, and online reinforcement learning, where the learner can gather new data by interacting with the environment; both formulations share the common goal of learning a near-optimal policy. For the most part, the bodies of research on offline and online reinforcement have evolved in parallel, but they exhibit a number of curious similarities. Algorithmically, many design principles for offline RL (e.g., pessimism) have online counterparts (e.g., optimism), and statistically efficient algorithms for both frameworks typically require similar representation conditions (e.g., ability to model state-action value functions). Yet, the frameworks have notable differences: online RL algorithms require exploration conditions to address the issue of distribution shift (Russo and Van Roy, 2013; Jiang et al., 2017; Sun et al., 2019; Wang et al., 2020c; Du et al., 2021; Jin et al., 2021a; Foster et al., 2021), while offline RL algorithms require conceptually distinct coverage conditions to ensure the data logging distribution sufficiently covers the state space (Munos, 2003; Antos et al., 2008; Chen and Jiang, 2019; Xie and Jiang, 2020, 2021; Jin et al., 2021b; Rashidinejad et al., 2021; Foster et al., 2022; Zhan et al., 2022).

Recently, Xie et al. (2023) exposed a deeper connection between online and offline RL by showing that coverability—that is, existence of a data distribution with good coverage for offline RL—is itself a sufficient condition that enables sample-efficient exploration in online RL, even when the learner has no prior knowledge

of said distribution. This suggests the possibility of a theoretical unification of online and offline RL, but the picture remains incomplete, and there are many gaps in our understanding. Notably, a promising emerging paradigm in offline RL makes use of the ability to model density ratios (also referred to as marginalized importance weights or simply weight functions) for the underlying MDP. Density ratio modeling offers an alternative to classical value function approximation (or, approximate dynamic programming) methods (Munos, 2007; Munos and Szepesvári, 2008; Chen and Jiang, 2019), as it avoids instability and typically succeeds under weaker representation conditions (requiring only realizability conditions as opposed to Bellman completeness-type assumptions). Yet despite extensive investigation into density ratio methods for offline RL—both in theory (Liu et al., 2018; Uehara et al., 2020; Yang et al., 2020; Uehara et al., 2021; Jiang and Huang, 2020; Xie and Jiang, 2020; Zhan et al., 2022; Chen and Jiang, 2022; Rashidinejad et al., 2023; Ozdaglar et al., 2023) and practice (Nachum et al., 2019; Kostrikov et al., 2019; Nachum and Dai, 2020; Zhang et al., 2020; Lee et al., 2021)—density ratio modeling has been conspicuously absent in the online reinforcement learning. This leads us to ask:

Can online reinforcement learning benefit from the ability to model density ratios?

Adapting density ratio-based methods to the online setting with provable guarantees presents a number of conceptual and technical challenges. First, since the data distribution in online RL is constantly changing, it is unclear what densities one should even attempt to model. Second, most offline reinforcement learning algorithms require relatively stringent notions of coverage for the data distribution. In online RL, it is unreasonable to expect data gathered early in the learning process to have good coverage, and naive algorithms may cycle or fail to explore as a result. As such, it may not be reasonable to expect density ratio modeling to benefit online RL in the same fashion as offline.

Our contributions. We show that in spite of these challenges, density ratio modeling enables guarantees for online reinforcement learning that were previously out of reach.

- Density ratios for online RL. We show (Section 3) that for any MDP with low coverability, density ratio realizability and realizability of the optimal state-action value function are sufficient for sample-efficient online RL. This result is obtained through a new algorithm, GLOW, which addresses the issue of distribution shift via careful use of truncated density ratios, which it combines with optimism to drive exploration. This complements Xie et al. (2023), who gave sample complexity guarantees for coverable MDPs under a stronger Bellman completeness assumption for the value function class.   
- Density ratios for hybrid RL. Our algorithm for online RL is computationally inefficient. We complement it (Section 4) with a more efficient counterpart, HYGLOW, for the hybrid RL framework (Song et al., 2023), in which the learner has access to additional offline data that covers a high-quality policy.   
- Hybrid-to-offline reductions. To achieve the result above, we investigate a broader question: when can offline RL algorithms be adapted as-is to online settings? We provide a new meta-algorithm, $H_{2}O$ , which reduces hybrid RL to offline RL by repeatedly calling a given offline RL algorithm as a black box. We show that $H_{2}O$ enjoys low regret whenever the black-box offline algorithm satisfies certain conditions, and demonstrate that these conditions are satisfied by a range of existing offline algorithms, thus lifting them to the hybrid RL setting.

While our results are theoretical in nature, we are optimistic that they will lead to further investigation into the power of density ratio modeling in online RL and inspire practical algorithms.

Paper organization. Section 2 contains necessary background, introducing density ratio modeling and the notion of coverability. Section 3 presents our main results for the online reinforcement learning framework, and Section 4 presents our main results for the hybrid framework. We conclude with discussion in Section 5. Proofs, examples, and additional discussion are deferred to the appendix.

# 1.1 Preliminaries

Markov Decision Processes. We consider an episodic reinforcement learning setting. A Markov Decision Process (MDP) is a tuple $\mathcal{M} = (\mathcal{X}, \mathcal{A}, P, R, H, d_{1})$ , where X is the (large/potentially infinite) state space, A is

the action space, $H \in N$ is the horizon, $R = \{R_h\}_{h=1}^H$ is the reward function (where $R_h : X \times A \to \Delta([0,1])$ ), $P = \{P_h\}_{h \leq 1}$ is the transition distribution (where $P_h : X \times A \to \Delta(X)$ ), and $d_1$ is the initial state distribution. A randomized policy is a sequence of functions $\pi = \{\pi_h : X \to \Delta(A)\}_{h=1}^H$ . When a policy is executed, it generates a trajectory $(x_1, a_1, r_1), \ldots, (x_H, a_H, r_h)$ via the process $a_h \sim \pi_h(x_h), r_h \sim R_h(x_h, a_h), x_{h+1} \sim P_h(x_h, a_h)$ , initialized at $x_1 \sim d_1$ (we use $x_{H+1}$ to denote a deterministic terminal state with zero reward). We write $P^\pi[\cdot]$ and $E^\pi[\cdot]$ to denote the law and corresponding expectation for the trajectory under this process.

For a policy $\pi$ , the expected reward for is given by $J(\pi) := \mathbb{E}^{\pi}\left[\sum_{h=1}^{H} r_h\right]$ , and the value functions given by

$$
V _ {h} ^ {\pi} (x) := \mathbb {E} ^ {\pi} \left[ \sum_ {h ^ {\prime} = h} ^ {H} r _ {h ^ {\prime}} \mid x _ {h} = x \right], \quad \text { and } \quad Q _ {h} ^ {\pi} (x, a) := \mathbb {E} ^ {\pi} \left[ \sum_ {h ^ {\prime} = h} ^ {H} r _ {h ^ {\prime}} \mid x _ {h} = x, a _ {h} = a \right].
$$

We write $\pi^{\star} = \{\pi_h^\star\}_{h=1}^H$ to denote an optimal deterministic policy, which maximizes $V^{\pi}$ at all states. We let $\mathcal{T}_h$ denote the Bellman (optimality) operator for layer $h$ , defined via

$$
[ \mathcal {T} _ {h} f ] (x, a) = \mathbb {E} \Big [ r _ {h} + \max _ {a ^ {\prime}} f (x _ {h + 1}, a ^ {\prime}) \mid x _ {h} = x, a _ {h} = a \Big ]
$$

for $f:\mathcal{X}\times \mathcal{A}\to \mathbb{R}.$

Online RL. In the online reinforcement learning framework, the learner repeatedly interacts with an unknown MDP by executing a policy and observing the resulting trajectory. The goal is to maximize total reward. Formally, the protocol proceeds in N rounds, where at each round $t \in [N]$ , the learner selects a policy $\pi^{(t)} = \{\pi_{h}^{(t)}\}_{h=1}^{H}$ in the (unknown) underlying MDP $M^{\star}$ and observes the trajectory $\{(x_{h}^{(t)}, a_{h}^{(t)}, r_{h}^{(t)})\}_{h=1}^{H}$ . Our results are most naturally stated in terms of PAC guarantees. Here, after the N rounds of interaction conclude, the learner can use all of the data collected to produce a final policy $\widehat{\pi}$ , with the goal of minimizing

$$
\mathbf {R i s k} := \mathbb {E} _ {\widehat {\pi} \sim p} [ J (\pi^ {\star}) - J (\widehat {\pi}) ], \tag {1}
$$

where $p \in \Delta(\Pi)$ denotes a distribution that the algorithm can use to randomize the final policy.

Offline RL. In offline reinforcement learning, the learner does not directly interact with $M^{\star}$ , and is instead given a dataset of tuples $(x_{h}, a_{h}, r_{h}, x_{h+1})$ collected i.i.d. according to $(x_{h}, a_{h}) \sim \mu_{h}$ , $r_{h} \sim R_{h}(x_{h}, a_{h})$ , $x_{h+1} \sim P_{h}(x_{h}, a_{h})$ , where $\mu_{h}$ is the offline data distribution for layer h. Based on the dataset, the offline algorithm produces a policy $\widehat{\pi}$ whose performance is measured by its risk, as in Equation (1); we write $Risk_{off}$ when we are in the offline interaction protocol.

Additional definitions and assumptions. We assume that rewards are normalized such that $\sum_{h=1}^{H} r_{h} \in [0,1]$ almost surely for all trajectories (Jiang and Agarwal, 2018; Wang et al., 2020a; Zhang et al., 2021; Jin et al., 2021a). To simplify presentation, we assume that X and A are countable; we expect that our results extend to handle continuous variables with an appropriate measure-theoretic treatment. We define the occupancy measure for policy $\pi$ via $d_{h}^{\pi}(x,a) := \mathbb{P}^{\pi}[x_{h} = x, a_{h} = a]$ .

# 2 Problem Setup: Density Ratio Modeling and Coverability

To investigate the power of density ratio modeling in online RL, we make use of function approximation, and aim to provide sample complexity guarantees with no explicit dependence on the size of the state space. We begin by appealing to value function approximation, a standard approach in online and offline reinforcement learning, and assume access to a value-function class $\mathcal{F} \subset (\mathcal{X} \times \mathcal{A} \times [H] \to [0,1])$ that can realize the optimal value function $Q^{\star}$ .

Assumption 2.1 (Value function realizability). We have $Q^{\star} \in F$ .

For $f \in F$ , we define the greedy policy $\pi_{f}$ via $\pi_{f,h}(x) = \arg\max_{a} f_{h}(x, a)$ , with ties broken in an arbitrary consistent fashion. For the remainder of the paper, we define $\Pi := \{\pi_{f} \mid f \in F\}$ as the policy class induced by F, unless otherwise specified.

Density ratio modeling. While value function approximation is a natural modeling approach, prior works in both online (Du et al., 2020; Weisz et al., 2021; Wang et al., 2021) and offline RL (Wang et al., 2020b; Zanette, 2021; Foster et al., 2022) have shown that value function realizability alone is not sufficient for statistically tractable learning in many settings. As such, value function approximation methods in online (Zanette et al., 2020; Jin et al., 2021a; Xie et al., 2023) and offline RL (Antos et al., 2008; Chen and Jiang, 2019) typically require additional representation conditions that may not be satisfied in practice, such as the stringent Bellman completeness assumption (i.e., $\mathcal{T}_h\mathcal{F}_{h+1} \subseteq \mathcal{F}_h$ ).

In offline RL, a promising emerging paradigm that goes beyond pure value function approximation is to model density ratios (or, marginalized important weights), which typically take the form

$$
\frac {d _ {h} ^ {\pi} (x , a)}{\mu_ {h} (x , a)} \tag {2}
$$

for a policy $\pi$ , where $\mu_{h}$ denotes the offline data distribution. A recent line of work (Xie and Jiang, 2020; Jiang and Huang, 2020; Zhan et al., 2022) shows, that given access to a realizable value function class and a weight function class W that can realize the ratio (2) (typically either for all policies $\pi$ , or for the optimal policy $\pi^{\star}$ ), one can learn a near-optimal policy offline in a sample-efficient fashion; such results sidestep the need for stringent value function representation conditions like Bellman completeness. To explore whether density ratio modeling has similar benefits in online RL, we make the following assumption.

Assumption 2.2 (Density ratio realizability). The learner has access to a weight function class $\mathcal{W} \subset (\mathcal{X} \times \mathcal{A} \times [H] \to \mathbb{R}_{+})$ such that for any policy pair $\pi, \pi' \in \Pi$ , and $h \in [H]$ , we have<sup>1</sup>

$$
w _ {h} ^ {\pi ; \pi^ {\prime}} (x, a) := \frac {d _ {h} ^ {\pi} (x , a)}{d _ {h} ^ {\pi^ {\prime}} (x , a)} \in \mathcal {W}.
$$

Assumption 2.2 does not assume that the density ratios under consideration are finite. That is, we do not assume boundedness of the weights, and our results do not pay for their range; our algorithm will only access certain clipped versions of the weight functions (in fact, it is sufficient to only realize certain “clipped” weight functions; cf. Remark B.1).

Compared to density ratio approaches in the offline setting, which typically require either realizability of $d_{h}^{\pi}/\mu_{h}$ for all $\pi\in\Pi$ or realizability of $d_{h}^{\pi^{\star}}/\mu_{h}$ , where $\mu$ is the offline data distribution, we require realizability of $d_{h}^{\pi}/d_{h}^{\pi'}$ for all pairs of policies $\pi,\pi'\in\Pi$ . This assumption is natural because it facilitates transfer between historical data (which is algorithm-dependent) and future policies. Notably, it is weaker than assuming realizability of $d_{h}^{\pi}/\nu_{h}$ for all $\pi\in\Pi$ and any fixed distribution $\nu$ (Remark B.2), and is also weaker than model-based realizability. We refer the reader to Appendix B for a detailed comparison to alternative assumptions.

Coverability. In addition to realizability assumptions, online RL methods require exploration conditions (Russo and Van Roy, 2013; Jiang et al., 2017; Sun et al., 2019; Wang et al., 2020c; Du et al., 2021; Jin et al., 2021a; Foster et al., 2021) that allow deliberately designed algorithms to control distribution shift or extrapolate to unseen states. Towards lifting density ratio modeling from offline to online RL, we make use of coverability (Xie et al., 2023), an exploration condition inspired by the notion of coverage in the offline setting.

Definition 2.1 (Coverability coefficient (Xie et al., 2023)). The coverability coefficient $C_{\mathrm{cov}} > 0$ for a policy class $\Pi$ is given by

$$
C _ {\mathsf {c o v}} := \inf _ {\mu_ {1}, \ldots , \mu_ {H} \in \Delta (\mathcal {X} \times \mathcal {A})} \sup _ {\pi \in \Pi , h \in [ H ]} \left\| \frac {d _ {h} ^ {\pi}}{\mu_ {h}} \right\| _ {\infty}.
$$

We refer to the distribution $\mu_h^\star$ that attains the minimum for $h$ as the coverability distribution.

Coverability is a structural property of the underlying MDP, and can be interpreted as the best value one can achieve for the concentrability coefficient $C_{\mathrm{conc}}(\mu) := \sup_{\pi \in \Pi, h \in [H]} \|d_h^\pi / \mu_h\|_\infty$ (a standard coverage parameter in offline RL (Munos, 2007; Munos and Szepesvári, 2008; Chen and Jiang, 2019)) by optimally designing the offline data distribution $\mu$ . However, in our setting the agent has no prior knowledge of $\mu^{\star}$ and no way

to explicitly search for it. Examples that admit low coverability include tabular MDPs and Block MDPs (Xie et al., 2023), linear/low-rank MDPs (Huang et al., 2023), and analytically sparse low-rank MDPs (Golowich et al., 2023); see Appendix C for further examples.

Concretely, we aim for sample complexity guarantees scaling as $\text{poly}(H, C_{\text{cov}}, \log|\mathcal{F}|, \log|\mathcal{W}|, \varepsilon^{-1})$ , where $\varepsilon$ is the desired bound on the risk in Eq. (1). Such a guarantee complements Xie et al. (2023), who achieved similar sample complexity under the Bellman completeness assumption, and parallels the fashion in which density ratio modeling allows one to remove completeness in offline RL. To simplify presentation as much as possible, we assume finiteness of F and W, but our results extend to infinite classes via standard uniform convergence arguments. Likewise, we do not require exact realizability, and an extension to misspecified classes is given in Appendix E.

Additional notation. For $n \in \mathbb{N}$ , we write $[n] = \{1, \dots, n\}$ . For a countable set $\mathcal{Z}$ , we write $\Delta(\mathcal{Z})$ for the set of probability distributions on $\mathcal{Z}$ . We adopt standard big-oh notation, and use $\widetilde{O}(\cdot)$ and $\widetilde{\Omega}(\cdot)$ to suppress factors polylogarithmic in $H, T, \varepsilon^{-1}, \log|\mathcal{F}|, \log|\mathcal{W}|$ , and other problem parameters. For each $h \in [H]$ , we define $\mathcal{F}_h = \{f_h \mid f \in \mathcal{F}\}$ and $\mathcal{W}_h = \{w_h \mid w \in \mathcal{W}\}$ . For any function $u: \mathcal{X} \times \mathcal{A} \mapsto \mathbb{R}$ and distribution $\rho \in \Delta(\mathcal{X} \times \mathcal{A})$ , we define the norms $\|u\|_{1,\rho} = \mathbb{E}_{(x,a) \sim \rho}[|u(x,a)|]$ and $\|u\|_{2,\rho} = \sqrt{\mathbb{E}_{(x,a) \sim \rho}[u^2(x,a)]}$ .

# 3 Online RL with Density Ratio Realizability

This section presents our main results for the online RL setting. We first introduce our main algorithm. GLOW (Algorithm 1), and explain the intuition behind its design (Section 3.1). We then show (Section 3.2) that GLOW obtains polynomial sample complexity guarantees (Theorems 3.1 and 3.2) under density ratio realizability and coverability, and conclude with a proof sketch (Section 3.3).

# 3.1 Algorithm and Key Ideas

Our algorithm, GLOW (Algorithm 1), is based on the principle of optimism in the face of uncertainty. For each iteration $t \leq T \in \mathbb{N}$ , the algorithm uses the density ratio class $\mathcal{W}$ to construct a confidence set (or version space) $\mathcal{F}^{(t)} \subseteq \mathcal{F}$ with the property that $Q^{\star} \in \mathcal{F}^{(t)}$ . It then chooses a policy $\pi^{(t)} = \pi_{f(t)}$ based on the value function $f^{(t)} \in \mathcal{F}^{(t)}$ with the most optimistic estimate $\mathbb{E}[f_1(x_1, \pi_{f,1}(x_1))]$ for the initial value. Then, it uses the policy $\pi^{(t)}$ to gather $K \in \mathbb{N}$ trajectories, which are used to update the confidence set for subsequent iterations.

Within the scheme above, the main novelty to our approach lies in the confidence set construction. GLOW appeals to global optimism (Jiang et al., 2017; Zanette et al., 2020; Du et al., 2021; Jin et al., 2021a; Xie et al., 2023), and constructs the confidence set $\mathcal{F}^{(t)}$ by searching for value functions $f\in \mathcal{F}$ that satisfy certain Bellman residual constraints for all layers $h\in [H]$ simultaneously. For MDPs with low coverability, previous such approaches (Jin et al., 2021a; Xie et al., 2023) make use of constraints based on squared Bellman error which requires Bellman completeness. The confidence set construction in GLOW (Eq. (4)) departs from this approach, and aims to find $f\in \mathcal{F}$ such that the average Bellman error is small for all weight functions. At the population level, this (informally) corresponds to requiring that for all $h\in [H]$ and $w\in \mathcal{W},^2$

$$
\mathbb {E} _ {\bar {d} ^ {(t)}} \left[ w _ {h} (x _ {h}, a _ {h}) (f _ {h} (x _ {h}, a _ {h}) - [ \mathcal {T} _ {h} f _ {h + 1} ] (x _ {h}, a _ {h})) \right] - \alpha^ {(t)} \cdot \mathbb {E} _ {\bar {d} ^ {(t)}} \left[ (w _ {h} (x _ {h}, a _ {h})) ^ {2} \right] \leq \beta^ {(t)}. \tag {3}
$$

where $\bar{d}_{h}^{(t)} := \frac{1}{t-1} \sum_{i < t} d_{h}^{\pi^{(t)}}$ is the historical data distribution and $\alpha^{(t)} > 0$ and $\beta^{(t)} > 0$ are algorithm parameters; this is motivated by the fact that the optimal value function satisfies

$$
\mathbb {E} ^ {\pi} \big [ w _ {h} (x _ {h}, a _ {h}) (Q _ {h} ^ {\star} (x _ {h}, a _ {h}) - \big [ \mathcal {T} _ {h} Q _ {h + 1} ^ {\star} \big ] (x _ {h}, a _ {h})) \big ] = 0
$$

for all functions $w$ and policies $\pi$ . Our analysis uses that Eq. (3) holds for the weight function $w_h^{(t)} := d_h^{\pi(t)} / \bar{d}_h^{(t)}$ which allows to transfer bounds on the off-policy Bellman error for the historical distribution $\bar{d}^{(t)}$ to the on-policy Bellman error for $\pi^{(t)}$ .

Algorithm 1 GLOW: Global Optimism via Weight Function Realizability   
input: Value function class $\mathcal{F}$ , Weight function class $\mathcal{W}$ , Parameters $T, K \in \mathbb{N}, \gamma \in [0,1]$ .

1: // For Theorem 3.1, set $T = \widetilde{\Theta}((H^2 C_{\mathrm{cov}} / \varepsilon^2) \cdot \log(|\mathcal{F}||\mathcal{W}|/\delta))$ , $K = 1$ , and $\gamma = \sqrt{C_{\mathrm{cov}}/(T \log(|\mathcal{F}||\mathcal{W}|/\delta))}$ .

2: // For Theorem 3.2, set $T = \widetilde{\Theta}(H^2 C_{\mathrm{cov}} / \varepsilon^2)$ , $K = \widetilde{\Theta}(T \log(|\mathcal{F}||\mathcal{W}|/\delta))$ , and $\gamma = \sqrt{C_{\mathrm{cov}}/T}$ .

3: Set $\gamma^{(t)} = \gamma \cdot t$ , $\alpha^{(t)} = 8 / \gamma^{(t)}$ and $\beta^{(t)} = (36\gamma^{(t)} / K(t-1)) \cdot \log(6|\mathcal{F}||\mathcal{W}|TH/\delta)$ .

4: Initialize $\mathcal{D}_h^{(1)} = \varnothing$ for all $h \leq H$ .

5: for $t = 1, \ldots, T$ do

6: Define confidence set based on (regularized) minimax average Bellman error: $\mathcal{F}^{(t)} = \left\{f \in \mathcal{F} \mid \forall h : \sup_{w \in \mathcal{W}_h} \widehat{\mathbb{E}}_{\mathcal{D}_h^{(t)}} \left[([ \widehat{\Delta}_h f](x,a,r,x') \cdot \widetilde{w}_h(x,a) - \alpha^{(t)} \cdot (\widetilde{w}_h(x,a))^2] \leq \beta^{(t)}\right\},$ (4)

where $\widetilde{w} := \text{clip}_{\gamma^{(t)}}[w]$ and $[\widehat{\Delta}_h f](x,a,r,x') := f_h(x,a) - r - \max_{a'} f_{h+1}(x',a')$ .

7: Compute optimistic value function and policy: $f^{(t)} := \arg\max_{f \in \mathcal{F}^{(t)}} \widehat{\mathbb{E}}_{x_1 \sim \mathcal{D}_1^{(t)}}[f_1(x_1,\pi_f(x_1))],$ and $\pi^{(t)} := \pi_{f^{(t)}}$ .

8: // Online data collection.

9: Initialize $\mathcal{D}_h^{(t+1)} \leftarrow \mathcal{D}_h^{(t)}$ for $h \in [H]$ .

10: for $k = 1, \ldots, K$ do

11: Collect a trajectory $(x_1, a_1, r_1), \ldots, (x_H, a_H, r_H)$ by executing $\pi^{(t)}$ .

12: Update $\mathcal{D}_h^{(t+1)} \leftarrow \mathcal{D}_h^{(t+1)} \cup \{(x_h, a_h, r_h, x_{h+1})\}$ for each $h \in [H]$ .

13: output: policy $\widehat{\pi} = \text{Unif}(\pi^{(1)}, \ldots, \pi^{(T)})$ . // For PAC guarantee only.

Remark 3.1. Among density ratio-based algorithms for offline reinforcement learning (Jiang and Huang, 2020; Xie and Jiang, 2020; Zhan et al., 2022; Chen and Jiang, 2022; Rashidinejad et al., 2023), the constraint (3) is most directly inspired by the Minimax Average Bellman Optimization (MABO) algorithm (Xie and Jiang, 2020), which uses a similar minimax approximation to the average Bellman error.

Partial coverage and clipping. Compared to the offline setting, much extra work is required to handle the issue of partial coverage. Early in the learning process, the ratio $w_{h}^{(t)} := d_{h}^{\pi(t)} / \bar{d}_{h}^{(t)}$ may be unbounded, which prevents the naive empirical approximation to Eq. (3) from concentrating. To address this issue, GLOW carefully truncates the weight functions under consideration.

Definition 3.1 (Clipping operator). For any $w: \mathcal{X} \times \mathcal{A} \to \mathbb{R} \cup \{\infty\}$ and $\gamma \in \mathbb{R}$ , we define the clipped weight function (at scale $\gamma$ ) via

$$
\operatorname{clip} _ {\gamma} [ w ] (x, a) := \min \{w (x, a), \gamma \}.
$$

Within GLOW, we replace the weight functions in Eq. (3) with clipped counterparts given by $\check{w}(x,a):=\text{clip}_{\gamma^{(t)}}[w](x,a)$ , where $\gamma^{(t)}:=\gamma\cdot t$ for a parameter $\gamma\in[0,1]$ . For a given iteration t, clipping in this fashion may render Eq. (3) a poor approximation to the on-policy Bellman error. The crux of our analysis is to show—via coverability—that on average across all iterations, the approximation error is small.

An important difference relative to MABO is that the weighted Bellman error in Eq. (3) incorporates a quadratic penalty $-\alpha^{(t)} \cdot \mathbb{E}_{\bar{d}^{(t)}}[(w_{h}(x_{h}, a_{h}))^{2}]$ for the weight function. This is not essential to derive polynomial sample complexity guarantees, but is critical to attain the $1/\varepsilon^{2}$ -type rates we achieve under our strongest realizability assumption. Briefly, regularization is beneficial because it allows us to appeal to variance-dependent Bernstein-style concentration; our analysis shows that while the variance of the weight functions under consideration may not be small on a per-iteration basis, it is small on average across all iterations (again, via coverability). Interestingly, similar quadratic penalties have been used within empirical offline RL algorithms based on density ratio modeling (Yang et al., 2020; Lee et al., 2021), as well as recent theoretical results (Zhan et al., 2022), but for considerations seemingly unrelated to concentration.

# 3.2 Main Result: Sample Complexity Bound for GLOW

We now present the main sample complexity guarantees for GLOW. The first result we present, which gives the tightest sample complexity bound, is stated under a form of density ratio realizability that strengthens Assumption 2.2. Concretely, we assume that the class W can realize density ratios for certain mixtures of policies. For $t \in N$ , we write $\pi^{(1:t)}$ as a shorthand for a sequence of policies $(\pi^{(1)}, \cdots, \pi^{(t)})$ , where $\pi^{(i)} \in \Pi$ , and let $d^{\pi^{(1:t)}} := \frac{1}{t} \sum_{i=1}^{t} d^{\pi^{(i)}}$ .

Assumption 2.2' (Density ratio realizability, mixture version). Let $T$ be the parameter to GLOW (Algorithm 1). For all $h \in [H]$ , $\pi \in \Pi$ , $t \leq T$ , and $\pi^{(1)}, \ldots, \pi^{(t)} \in \Pi$ , we have

$$
w _ {h} ^ {\pi ; \pi^ {(1: t)}} (x, a) := \frac {d _ {h} ^ {\pi} (x , a)}{d _ {h} ^ {\pi^ {(1 : t)}} (x , a)} \in \mathcal {W}.
$$

This assumption directly facilitates transfer from the algorithm's historical distribution $\bar{d}_{h}^{(t)} := \frac{1}{t-1} \sum_{i < t} d_{h}^{\pi(t)}$ to on-policy error. Naturally, it is implied by the stronger-but-simpler-to-state assumption that we can realize density ratios $d_{h}^{\pi}/d_{h}^{\rho}$ for all $\pi \in \Pi$ and all mixture policies $\rho \in \Delta(\Pi)$ . Under Assumption 2.2', we show that GLOW obtains $1/\varepsilon^{2}$ -PAC sample complexity and $\sqrt{T}$ -regret.

Theorem 3.1 (Risk bound for GLOW under strong density ratio realizability). Let $\varepsilon > 0$ be given, and suppose that Assumptions 2.1 and 2.2' hold. Then, GLOW, with hyperparameters $T = \widetilde{\Theta}\big((H^2 C_{\mathrm{cov}} / \varepsilon^2) \cdot \log(|\mathcal{F}||\mathcal{W}|/\delta)\big)$ , $K = 1$ , and $\gamma = \sqrt{C_{\mathrm{cov}} / (T\log(|\mathcal{F}||\mathcal{W}|/\delta))}$ returns an $\varepsilon$ -suboptimal policy $\widehat{\pi}$ with probability at least $1 - \delta$ after collecting

$$
N = \widetilde {O} \left(\frac {H ^ {2} C _ {\mathrm{cov}}}{\varepsilon^ {2}} \log (| \mathcal {F} | | \mathcal {W} | / \delta)\right) \tag {6}
$$

trajectories. Additionally, for any $T \in \mathbb{N}$ , with the same choice for $K$ and $\gamma$ as above, GLOW enjoys the regret bound

$$
\mathbf {R e g} := \sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) = \widetilde {O} \left(H \sqrt {C _ {\mathrm{cov}} T \log \left(| \mathcal {F} | | \mathcal {W} | / \delta\right)}\right).
$$

Next, we provide our main result, which gives a sample complexity guarantee under density ratio realizability for pure policies (Assumption 2.2). To obtain the result, we begin with a class W that satisfies Assumption 2.2, then expand it to obtain an augmented class $\overline{W}$ that satisfies mixture realizability (Assumption 2.2'). This reduction increases $\log|\mathcal{W}|$ by a T factor, which we offset by increasing the batch size K; this leads to a polynomial increase in sample complexity. $^{4}$

Theorem 3.2 (Risk bound for GLOW under weak density ratio realizability). Let $\varepsilon > 0$ be given, and suppose that Assumptions 2.1 and 2.2 hold for the classes $\mathcal{F}$ and $\mathcal{W}$ . Then, GLOW, when executed with a modified class $\overline{\mathcal{W}}$ defined in Eq. (34) in Appendix E, with hyperparameters $T = \widetilde{\Theta}(H^2 C_{\mathrm{cov}} / \varepsilon^2)$ , $K = \widetilde{\Theta}(T\log (|\mathcal{F}||\mathcal{W}|/\delta))$ , and $\gamma = \sqrt{C_{\mathrm{cov}} / T}$ , returns an $\varepsilon$ -suboptimal policy $\widehat{\pi}$ with probability at least $1 - \delta$ after collecting $N$ trajectories, for

$$
N = \widetilde {O} \left(\frac {H ^ {4} C _ {\text { cov }} ^ {2}}{\varepsilon^ {4}} \log (| \mathcal {F} | | \mathcal {W} | / \delta)\right). \tag {7}
$$

Theorems 3.1 and 3.2 show for the first time that value function realizability and density ratio realizability alone are sufficient for sample-efficient online RL under coverability. In particular, the sample complexity and regret bound in Theorem 3.1 match the coverability-based guarantees obtained in Xie et al. (2023, Theorem 1) under the complementary Bellman completeness assumption, with the only difference being that they scale with $\log(|\mathcal{F}||\mathcal{W}|)$ instead of $\log|\mathcal{F}|$ ; as discussed in Xie et al. (2023), this rate is tight for the special case of contextual bandits (H = 1). Interesting open questions include (i) whether the sample complexity

for learning with density ratio realizability for pure policies can be improved to $1/\varepsilon^{2}$ , and (ii) whether value realizability and coverability alone are sufficient for sample-efficient RL. Extensions to Theorems 3.1 and 3.2 under misspecification are given in Appendix E. We further refer to Appendix C for examples instantiating these results. In particular, our results establish a positive result for a generalized class of Block MDPs with coverable latent spaces, while only requiring (for the first time) function approximation conditions that concern the latent space (Example C.2).

Like other algorithms based on global optimism (Jiang et al., 2017; Zanette et al., 2020; Du et al., 2021; Jin et al., 2021a; Xie et al., 2023), GLOW is not computationally efficient. As a step toward developing practical online RL algorithms based on density ratio modeling, we give a more efficient counterpart for the hybrid RL model in the Section 4.

Remark 3.2 (Connection to GOLF). Prior work (Xie et al., 2023) analyzed the GOLF algorithm of Jin et al. and established positive results under coverability and Bellman completeness. We remark that by allowing for weight functions that take negative values, $^{5}$ GLOW can be viewed as a generalization of GOLF, and can be configured to obtain comparable results. Indeed, given a value function class F that satisfies Bellman completeness, the weight function class $W := \{f - f' \mid f, f' \in F\}$ leads to a confidence set construction at least as tight as that of GOLF. To see this, observe that if we set $\gamma \geq 2$ so that no clipping occurs, our construction for $\mathcal{F}^{(t)}$ (Eq. (4)) implies (after standard concentration arguments) that $Q^{\star} \in \mathcal{F}^{(t)}$ and that in-sample squared Bellman errors are small with high probability. These ingredients are all that is required to repeat the analysis of GOLF from Xie et al. (2023).

# 3.3 Proof Sketch

We now give a proof sketch for Theorem 3.1, highlighting the role of truncated weight functions in addressing partial coverage. We focus on the regret bound; the sample complexity bound in Eq. (6) is an immediate consequence.

By design, the constraint in Eq. (4) ensures that $Q^{\star} \in \mathcal{F}^{(t)}$ for all $t \leq T$ with high probability. Thus, by a standard regret decomposition for optimistic algorithms (Lemma D.4 in the appendix), we have

$$
\mathbf {R e g} = \sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) \lesssim \sum_ {t = 1} ^ {T} \sum_ {h = 1} ^ {H} \underbrace {\mathbb {E} _ {d _ {h} ^ {(t)}} \left[ f _ {h} ^ {(t)} (x _ {h} , a _ {h}) - [ \mathcal {T} f _ {h + 1} ^ {(t)} ] (x _ {h} , a _ {h}) \right]} _ {\text { On - policy   Bellman   error   for } f ^ {(t)} \text { under } \pi^ {(t)}}, \tag {8}
$$

up to lower-order terms, where we abbreviate $d^{(t)} = d^{\pi^{(t)}}$ . Defining $[\Delta_{h}f^{(t)}](x,a) := f_{h}^{(t)}(x,a) - [\mathcal{T}_{h}f_{h+1}^{(t)}](x,a)$ , it remains to bound the on-policy expected bellman error $\mathbb{E}_{d_{h}^{(t)}}[[\Delta_{h}f^{(t)}](x_{h},a_{h})]$ . To do so, a natural approach is to relate this quantity to the weighted off-policy Bellman error under $\bar{d}^{(t)} := \frac{1}{t-1}\sum_{i<t} d^{\pi^{(i)}}$ by introducing the weight function $d^{(t)}/\bar{d}^{(t)} \in \mathcal{W}$ :

$$
\mathbb {E} _ {d _ {h} ^ {(t)}} [ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) ] \approx \mathbb {E} _ {\overline {{d}} _ {h} ^ {(t)}} \left[ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) \cdot \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\overline {{d}} _ {h} ^ {(t)} (x _ {h} , a _ {h})} \right].
$$

Unfortunately, this equality is not true as-is because the ratio $d^{(t)}/\bar{d}^{(t)}$ can be unbounded. We address this by replacing $\bar{d}^{(t)}$ by $\bar{d}^{(t+1)}$ throughout the analysis (at the cost of small approximation error), and work with the weight function $w_{h}^{(t)} := d_{h}^{(t)}/\bar{d}_{h}^{(t+1)} \in \mathcal{W}$ , which is always bounded in magnitude t. However, while boundedness is a desirable property, the range t is still too large to obtain non-vacuous concentration guarantees. This motivates us to introduce clipped/truncated weight functions via the following decomposition.

$$
\underbrace {\mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \left[ \Delta_ {h} f ^ {(t)} \right] (x _ {h} , a _ {h}) \right]} _ {\text {On - policy Bellman error}} \leq \underbrace {\mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left[ \Delta_ {h} f ^ {(t)} \right] (x _ {h} , a _ {h}) \cdot \mathsf {c l i p} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} \right] (x _ {h} , a _ {h}) \right]} _ {(A _ {t}) \colon \text {Clipped off - policy Bellman error}} + \underbrace {\mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \mathbb {I} \big \{w _ {h} ^ {(t)} (x _ {h} , a _ {h}) \geq \gamma^ {(t)} \big \} \right]} _ {(B _ {t}) \colon \text {Loss due to clipping}}.
$$

Recall that $\check{w}_{h}^{(t)} := \mathsf{clip}_{\gamma^{(t)}} \left[ w_{h}^{(t)} \right] (x_{h}, a_{h})$ . As $w^{(t)} \in \mathcal{W}$ , it follows from the constraint in Eq. (4) and Freedman-type concentration that the clipped Bellman error in term $(A_{t})$ has order $\alpha^{(t)} \cdot \mathbb{E}_{\tilde{d}^{(t+1)}} \left[ (\check{w}_{h}^{(t)})^{2} \right] + \beta^{(t)}$ , so that

$\sum_{t=1}^{T} A_t \leq \sum_{t=1}^{T} \alpha^{(t)} \cdot \mathbb{E}_{\bar{d}^{(t+1)}} \left[ (\check{w}_h^{(t)})^2 \right] + \beta^{(t)}$ . Since we clip to $\gamma^{(t)} = \gamma t$ , we have $\sum_{t=1}^{T} \beta^{(t)} \lesssim \gamma \cdot T \log (|\mathcal{F}||\mathcal{W}|/\delta)$ ; bounding the sum of weight functions requires a more involved argument that we defer for a moment.

We now focus on bounding the terms $(B_{t})$ . Each term $(B_{t})$ captures the extent to which the weighted off-policy Bellman error at iteration $t$ fails to approximate the true Bellman error due to clipping. This occurs when $\bar{d}^{(t + 1)}$ has poor coverage relative to $d^{(t)}$ , which happens when $\pi^{(t)}$ visits a portion of the state space not previously covered. We begin by applying Markov's inequality ( $\mathbb{I}\{u\geq v\} \leq u / v$ for $u,v\geq 0$ ) to bound

$$
B _ {t} \leq \frac {1}{\gamma^ {(t)}} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ w _ {h} ^ {(t)} (x _ {h}, a _ {h}) \right] = \frac {1}{\gamma^ {(t)}} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\widetilde {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right] = \frac {1}{\gamma} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\widetilde {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right], \tag {9}
$$

where the equality uses that $\gamma^{(t)} := \gamma \cdot t$ and $\widetilde{d}^{(t+1)} := \bar{d}^{(t+1)} \cdot t$ . Our most important insight is that even though each term in Eq. (9) might be large on a given iteration t (if a previously unexplored portion of the state space is visited), coverability implies that on average across all iterations the error incurred by clipping must be small. In particular, using a variant of a coverability-based potential argument from Xie et al. (2023) (Lemma D.5), we show that

$$
\sum_ {t = 1} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\widetilde {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right] \leq O (C _ {\mathrm{cov}} \cdot \log (T)),
$$

so that $\sum_{t=1}^{T} B_t \leq \widetilde{O}(C_{\mathrm{cov}} / \gamma)$ . To conclude the proof, we use an analogous potential argument to show the sum of weight functions in our bound on $\sum_{t=1}^{T} A_t$ also satisfies $\sum_{t=1}^{T} \alpha^{(t)} \cdot \mathbb{E}_{\bar{d}^{(t+1)}}[(\check{w}_h^{(t)})^2] \leq \widetilde{O}(C_{\mathrm{cov}} / \gamma)$ . The intuition is similar: the squared weight functions (corresponding to variance of the weighted Bellman error) may be large in a given round, but cannot be large for all rounds under coverability. Altogether, combining the bounds on $A_t$ and $B_t$ gives

$$
\mathbf {R e g} = \widetilde {O} \left(H \left(\frac {C _ {\text { cov }}}{\gamma} + \gamma \cdot T \log (| \mathcal {F} | | \mathcal {W} | H T \delta^ {- 1})\right)\right). \tag {10}
$$

The final result follows by choosing $\gamma > 0$ to balance the terms.

We find it interesting that the way in which this proof makes use of coverability—to handle the cumulative loss incurred by clipping—is quite different from the analysis in Xie et al. (2023), where it more directly facilitates a change-of-measure argument.

# 4 Efficient Hybrid RL with Density Ratio Realizability

Our results in the prequel show that density ratio realizability and coverability suffice for sample-efficient online RL. However, like other algorithms for sample-efficient exploration under general function approximation (Jiang et al., 2017; Du et al., 2021; Jin et al., 2021a), GLOW is not computationally efficient. Toward overcoming the challenges of intractable computation in online exploration, a number of recent works show that including additional offline data in online RL can lead to computational benefits in theory (e.g., Xie et al., 2021b; Wagenmaker and Pacchiano, 2023; Song et al., 2023; Zhou et al., 2023) and in practice (e.g., Cabi et al., 2020; Nair et al., 2020; Ball et al., 2023; Song et al., 2023; Zhou et al., 2023). Notably, combining offline and online data can enable algorithms that provably explore without having to appeal to optimism or pessimism, both of which are difficult to implement efficiently under general function approximation.

Song et al. (2023) formalize a version of this setting—in which online RL is augmented with offline data—as hybrid reinforcement learning. Formally, in hybrid RL, the learner interacts with the MDP online (as in Section 1.1) but is additionally given an offline dataset $D_{off}$ collected from a data distribution $\nu$ . The data distribution $\nu$ is typically assumed to provide coverage for the optimal policy $\pi^{\star}$ (formalized in Definition 4.4), but not on all policies, and thus additional online exploration is required (see ?? for further discussion).

# 4.1 $H_{2}O$ : A Provable Black-Box Hybrid-to-Offline Reduction

Interestingly, many of the above approaches for the hybrid setting simply apply offline algorithms (with relatively little modification) on a mixture of online and offline data (e.g., Cabi et al., 2020; Nair et al., 2020; Ball et al., 2023). This raises the question: when can we use a given offline algorithm as a black box to solve the problem of hybrid RL (or, more generally, of online RL?). To answer this, we give a general meta-algorithm, $\mathrm{H}_2\mathrm{O}$ , which provides a provable black-box reduction to solve the hybrid RL problem by repeatedly invoking a given offline RL algorithm on a mixture of offline data and freshly gathered online trajectories. We instantiate the meta-algorithm using a simplified offline counterpart to GLOW as a black box to obtain HYGLOW, a density ratio-based algorithm for the hybrid RL setting that improves upon the computational efficiency of GLOW (Section 4.2). To present the result, we first describe the class of offline RL algorithms with which it will be applied.

Offline RL and partial coverage. We refer to a collection of distributions $\mu=\{\mu_{h}\}_{h=1}^{H}$ , where $\mu_{h}\in\Delta(\mathcal{X}\times\mathcal{A})$ , as a data distribution, and we say that a dataset $D=\{D_{h}\}_{h=1}^{H}$ has $H\cdot n$ samples from data distributions $\mu^{(1)},\ldots,\mu^{(n)}$ if $\mathcal{D}_{h}=\{(x_{h}^{(i)},a_{h}^{(i)},r_{h}^{(i)},x_{h+1}^{(i)})\}_{i=1}^{n}$ where $(x_{h}^{(i)},a_{h}^{(i)})\sim\mu_{h}^{(i)}$ , $r_{h}^{(i)}\sim R_{h}(x_{h}^{(i)},a_{h}^{(i)})$ , $x_{h+1}^{(i)}\sim P_{h}(x_{h}^{(i)},a_{h}^{(i)})$ . We denote the mixture distribution via $\mu^{(1:n)}=\{\mu_{h}^{(1:n)}\}_{h=1}^{H}$ , where $\mu_{h}^{(1:n)}:=\frac{1}{n}\sum_{i=1}^{n}\mu_{h}^{(i)}$ .

Definition 4.1 (Offline RL algorithm). An offline RL algorithm $\mathbf{Alg}_{\mathrm{off}}$ takes as input a dataset $\mathcal{D} = \{\mathcal{D}_h\}_{h=1}^H$ of $H \cdot n$ samples from $\mu^{(1)}, \ldots, \mu^{(n)}$ and outputs a policy $\pi = \{\pi_h\}_{h=1}^H$ . We allow $\mu^{(1)}, \ldots, \mu^{(n)}$ to be adaptively chosen, i.e. each $\mu^{(i+1)}$ may be a function of the samples generated from $\mu^{(1)} \ldots \mu^{(i)}$ .

An immediate problem with directly invoking offline RL algorithms in the hybrid model is that typical algorithms—particularly, those that do not make use of pessimism (e.g., Xie and Jiang, 2020)—require relatively uniform notions of coverage (e.g., coverage for all policies as opposed to just coverage for $\pi^{\star}$ ) to provide guarantees, leading one to worry that their behaviour might be completely uncontrolled when applied with non-exploratory datasets. Fortunately, we will show that for a large class algorithms whose risk scales with a measure of coverage we refer to as clipped concentrability, this phenomenon cannot occur. Below, for any distribution $\rho \in \Delta(\mathcal{X} \times \mathcal{A})$ , we write $\| \cdot \|_{1,\rho}$ and $\| \cdot \|_{2,\rho}$ for the $L_1(\rho)$ and $L_2(\rho)$ norms.

Definition 4.2 (Clipped concentrability coefficient). The clipped concentrability coefficient (at scale $\gamma \in \mathbb{R}_+$ ) for $\pi \in \Pi$ relative to a data distribution $\mu = \{\mu_h\}_{h=1}^H$ , where $\mu_h \in \Delta(\mathcal{X} \times \mathcal{A})$ , is defined as

$$
\mathsf {C C} _ {h} (\pi , \mu , \gamma) := \left\| \operatorname{clip} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi}}{\mu_ {h}} \right] \right\| _ {1, d _ {h} ^ {\pi}}.
$$

This coefficient should be thought of as a generalization of the standard (squared) $L_{2}(\mu)$ concentrability coefficient $C_{\mathrm{conc},2,h}^{2}(\pi,\mu):=\|d_{h}^{\pi}/\mu_{h}\|_{2,\mu_{h}}^{2}=\|d_{h}^{\pi}/\mu_{h}\|_{1,d_{h}^{\pi}}$ , a fundamental object in the analysis of offline RL algorithms (e.g., Farahmand et al., 2010), but incorporates clipping to better handle partial coverage. We consider offline RL algorithms with the property that for any offline distribution $\mu$ , the algorithm's risk can be bounded by the clipped concentrability coefficients for (i) the output policy $\widehat{\pi}$ , and (ii) the optimal policy $\pi^{\star}$ . For the following definition, we recall the notation $\gamma^{(n)}:=\gamma\cdot n$ .

Definition 4.3 (CC-bounded offline RL algorithm). We say that an offline algorithm $\mathbf{Alg}_{\mathrm{off}}$ is CC-bounded at scale $\gamma \in \mathbb{R}_+$ under an assumption $\mathbf{Assumption}(\cdot)$ if there exists scalars $\mathfrak{a}_{\gamma}, \mathfrak{b}_{\gamma}$ such that for all $n \in \mathbb{N}$ and data distributions $\mu^{(1)}, \ldots, \mu^{(n)}$ , $\mathbf{Alg}_{\mathrm{off}}$ outputs a distribution $p \in \Delta(\Pi)$ satisfying

$$
\mathbf {R i s k} _ {\text { off }} = \mathbb {E} _ {\widehat {\pi} \sim p} [ J (\pi^ {\star}) - J (\widehat {\pi}) ] \leq \sum_ {h = 1} ^ {H} \frac {\mathfrak {a} _ {\gamma}}{n} \left(\mathbb {C C} _ {h} (\pi^ {\star}, \mu^ {(1: n)}, \gamma^ {(n)}) + \mathbb {E} _ {\widehat {\pi} \sim p} [ \mathbb {C C} _ {h} (\widehat {\pi}, \mu^ {(1: n)}, \gamma^ {(n)}) ]\right) + \mathfrak {b} _ {\gamma} \tag {11}
$$

with probability at least $1-\delta$ , when given a dataset of $H\cdot n$ samples from $\mu^{(1)},\ldots,\mu^{(n)}$ such that $\text{Assumption}(\mu^{(1:n)},M^{\star})$ is satisfied.

This definition does not automatically imply that the offline algorithm has low offline risk, but simply that the risk can be bounded in terms of clipped coverage (which may be large if the dataset has poor coverage). In the sequel, we will show that many natural offline RL algorithms have this property (Appendix F.1). Examples of assumptions for Assumption include value function completeness (e.g., for FQI (Chen and Jiang, 2019)) and realizability of value functions and density ratios (e.g., for MABO (Xie and Jiang, 2020)).

Offline RL algorithms based on pessimism (Jin et al., 2021b; Rashidinejad et al., 2021; Xie et al., 2021a) typically enjoy risk bounds that only require coverage for $\pi^{\star}$ . Crucially, by allowing the risk bound in Definition 4.3 to scale with coverage for $\widehat{\pi}$ in addition to $\pi^{\star}$ , we can accommodate non-pessimistic offline RL algorithms such as FQI and MABO that are weaker statistically, yet more computationally efficient.

Remark 4.1. While the bound in Eq. (11) might seem to suggest a $^{1/n}$ -type rate, it will typically lead to a $^{1/\sqrt{n}}$ -type rate after choosing $\gamma > 0$ to balance the $\frac{\mathfrak{a}_{\gamma}}{n}$ and $\mathfrak{b}_{\gamma}$ terms.

The $H_{2}O$ algorithm. Our reduction, $H_{2}O$ , is given in Algorithm 2. For any dataset D, we will write $D|_{1:t}$ for the subset consisting of its first t elements. The algorithm is initialized with an offline dataset $D_{off} = \{D_{off,h}\}_{h=1}^{H}$ , and at each iteration $t \in [T]$ invokes the black-box offline RL algorithm $Alg_{off}$ with a dataset $D_{hybrid} = \{D_{hybrid,h}\}_{h=1}^{H}$ that mixes the first t elements of $D_{off,h}$ with all of the online data gathered so far. This produces a policy $\pi^{(t)}$ , which is executed to gather trajectories that are then added to the online dataset and used at the next iteration.

$\mathrm{H}_2\mathrm{O}$ is inspired by empirical methods for the hybrid setting (e.g., Cabi et al., 2020; Nair et al., 2020; Ball et al., 2023). The total computational cost is simply that of running the base algorithm $\mathbf{Alg}_{\mathrm{off}}$ for $T$ rounds, and in particular the meta-algorithm is efficient whenever $\mathbf{Alg}_{\mathrm{off}}$ is.

Algorithm 2 $\mathrm{H}_2\mathrm{O}$ : Hybrid-to-Offline Reduction   
input: Parameter $T \in N$ , offline algorithm $Alg_{off}$ , offline datasets $D_{off} = \{D_{off,h}\}_{h}$ each of size T.
1: Initialize $\mathcal{D}_{\mathrm{on},h}^{(1)} = \mathcal{D}_{\mathrm{hybrid},h}^{(1)} = \varnothing$ for all $h \in [H]$ .
2: for $t = 1, \ldots, T$ do
3: Get policy $\pi^{(t)}$ from $Alg_{off}$ on dataset $\mathcal{D}_{\mathrm{hybrid}}^{(t)} = \{\mathcal{D}_{\mathrm{hybrid},h}^{(t)}\}_{h}$ .
4: Collect trajectory $(x_{1}, a_{1}, r_{1}), \ldots, (x_{H}, a_{H}, r_{H})$ using $\pi^{(t)}$ ; $\mathcal{D}_{\mathrm{on},h}^{(t+1)} := \mathcal{D}_{\mathrm{on},h}^{(t)} \cup \{(x_{h}, a_{h}, r_{h}, x_{h+1})\}$ .
5: Aggregate offline and online data: $\mathcal{D}_{\mathrm{hybrid},h}^{(t+1)} := \mathcal{D}_{\mathrm{off},h}|_{1:t} \cup \mathcal{D}_{\mathrm{on},h}^{(t+1)}$ for all $h \in [H]$ .
6: output: policy $\widehat{\pi} = \operatorname{Unif}(\pi^{(1)}, \ldots, \pi^{(T)})$ .

Main risk bound for $H_{2}O$ . We now present the main result for this section: a risk bound for the $H_{2}O$ reduction. Our bound depends on the coverability parameter for the underlying MDP, as well as the quality of the offline data distribution $\nu$ , quantified by single-policy concentrability.

Definition 4.4 (Single-policy concentrability). A data distribution $\nu = \{\nu_h\}_{h=1}^H$ satisfies $C_{\star}$ -single-policy concentrability if

$$
\max _ {h} \left\| \frac {d _ {h} ^ {\pi^ {\star}}}{\nu_ {h}} \right\| _ {\infty} \leq C _ {\star}.
$$

Theorem 4.1 (Risk bound for $\mathrm{H}_2\mathrm{O}$ ). Let $T \in \mathbb{N}$ be given, let $\mathcal{D}_{\mathrm{off}}$ consist of $H \cdot T$ samples from data distribution $\nu$ , and suppose that $\nu$ satisfies $C_{\star}$ -single-policy concentrability. Let $\mathbf{Alg}_{\mathrm{off}}$ be CC-bounded at scale $\gamma \in (0,1)$ under Assumption $(\cdot)$ , with parameters $\mathfrak{a}_{\gamma}$ and $\mathfrak{b}_{\gamma}$ . Suppose that for all $t \in [T]$ and $\pi^{(1)}, \ldots, \pi^{(t)} \in \Pi$ , Assumption $(\mu^{(t)}, M^{\star})$ holds for $\mu^{(t)} := \{1/2(\nu_h + 1/t \sum_{i=1}^t d_h^{\pi^{(i)}})\}_{h=1}^H$ . Then, with probability at least $1 - \delta T$ , the risk of $\mathrm{H}_2\mathrm{O}$ (Algorithm 2) with inputs $T$ , $\mathbf{Alg}_{\mathrm{off}}$ , and $\mathcal{D}_{\mathrm{off}}$ is bounded as

$$
\mathbf {R i s k} \leq \widetilde {O} \left(H \left(\frac {\mathfrak {a} _ {\gamma} (C _ {\star} + C _ {\mathrm{cov}})}{T} + \mathfrak {b} _ {\gamma}\right)\right). \tag {12}
$$

For the algorithms we consider, one can take $\mathfrak{a}_{\gamma} \propto \mathfrak{a}/_{\gamma}$ and $\mathfrak{b}_{\gamma} \propto \mathfrak{b}\gamma$ for scalar-valued problem parameters $\mathfrak{a}, \mathfrak{b} > 0$ , so that choosing $\gamma$ optimally gives $\mathbf{Risk} \leq \widetilde{O}\big(H\sqrt{(C_{\star} + C_{\mathrm{cov}})\mathfrak{ab}/T}\big)$ and a sample complexity of

$$
\widetilde {O} \left(H ^ {2} (C _ {\star} + C _ {\mathrm{cov}}) \mathfrak {a b} / \varepsilon^ {2}\right) \text {   to   find   an   } \varepsilon \text {-optimal   policy   (see   Corollary   F.1). } ^ {8}
$$

The basic idea behind the proof of Theorem 4.1 is as follows: using a standard regret decomposition based on average Bellman error, we can bound the risk of $\mathrm{H}_2\mathrm{O}$ by the average of the two clipped concentrability terms in (11) across all iterations. Coverage for $\pi^{\star}$ is automatically handled by Definition 4.4, and we use a potential argument similar to the online setting (cf. Section 3.3) to show that the $\widehat{\pi}$ -coverage terms can be controlled by coverability. This is similar in spirit to the analysis of Song et al. (2023), with coverability taking the place of bilinear rank (Du et al., 2021).

Our result is stated as a bound on the risk to the optimal policy $\pi^{\star}$ , but extends to give a bound on the risk of any comparator $\pi^{c}$ with Definition 4.3 and Definition 4.4 replaced by coverage for $\pi^{c}$ . This is a special case of a more general result, Theorem F.6, which handles the general case where $\nu$ need not satisfy single-policy concentrability.

# 4.2 Applying the Reduction: HYGLOW

We now apply $H_{2}O$ to give a hybrid counterpart to GLOW (Algorithm 1), using a variant of MABO (Xie and Jiang, 2020) as the black box offline RL algorithm $Alg_{off}$ in $H_{2}O$ . Further examples, which apply Fitted Q-Iteration (FQI) and Model-Based Maximum Likelihood Estimation as the black box, are deferred to Appendix F.1.

As discussed in Section 3, the construction for the confidence set of GLOW (Eq. (4)) bears some resemblance to the MABO algorithm (Xie and Jiang, 2020) from offline RL, save for the important additions of clipping and regularization. For our main example, the offline RL algorithm we consider is a variant of MABO that incorporates clipping and regularization in the same fashion, which we call MABO.CR. Our algorithm takes as input a dataset $\mathcal{D} = \{\mathcal{D}_h\}$ with $H \cdot n$ samples, has parameters consisting of a value function class $\mathcal{F}$ , a weight function class $\mathcal{W}$ , and a clipping scale $\gamma$ , and computes the following estimator:

$$
\widehat {f} \in \underset {f \in \mathcal {F}} {\arg \min} \max _ {w \in \mathcal {W}} \sum_ {h = 1} ^ {H} \left| \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \check {w} _ {h} (x _ {h}, a _ {h}) [ \widehat {\Delta} _ {h} f ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \right] \right| - \alpha^ {(n)} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \check {w} _ {h} ^ {2} (x _ {h}, a _ {h}) \right], \tag {13}
$$

where $\alpha^{(n)} := 8/\gamma^{(n)}$ and $\check{w}_{h} := \mathsf{clip}_{\gamma^{(n)}}[w_{h}]$ . We will show that this algorithm is CC-bounded under $Q^{\star}$ -realizability and a density ratio realizability assumption.

Theorem 4.2 (MABO.CR is CC-bounded). Let $\mathcal{D} = \{\mathcal{D}_h\}_{h=1}^H$ consist of $H \cdot n$ samples from $\mu^{(1)}, \ldots, \mu^{(n)}$ . For any $\gamma \in \mathbb{R}_+$ , the MABO.CR algorithm (Eq. (13)) with parameters $\mathcal{F}$ , augmented class $\overline{\mathcal{W}}$ defined in Eq. (38) in Appendix F.1.1, and $\gamma$ is CC-bounded at scale $\gamma$ under the Assumption that $Q^\star \in \mathcal{F}$ and that for all $\pi \in \Pi$ and $h \in [H]$ , $d_h^\pi / \mu_h^{(1:n)} \in \mathcal{W}$ .

We will show that this algorithm is CC-bounded under $Q^{\star}$ -realizability and a density ratio realizability assumption. We remark that, following the same arguments in Remark B.1, it suffices to instead only realize the clipped density ratios for the optimal scale $\gamma$ . By instantiating $H_{2}O$ with this algorithm, we obtain a density ratio-based algorithm for the hybrid RL setting that is statistically efficient and improves the computational efficiency of GLOW by removing the need for optimism. We call the end-to-end hybrid algorithm HYGLOW, and the full pseudocode can be found in Algorithm 3.

Corollary 4.1 (HYGLOW Risk bound). Let $\varepsilon > 0$ be given, $\mathcal{D}_{\mathrm{off}}$ consist of $H \cdot T$ samples from data distribution $\nu$ , where $\nu$ satisfies $C_{\star}$ -single-policy concentrability. Suppose that $Q^{\star} \in \mathcal{F}$ and that for all $t \in [T]$ , $\pi \in \Pi$ , and $h \in [H]$ , we have $d_h^{\pi} / \mu_h^{(t)} \in \mathcal{W}$ , where $\mu_h^{(t)} := 1/2 (\nu_h + 1/t \sum_{i=1}^{t} d_h^{\pi^{(i)}})$ . Then, HYGLOW with inputs $T = \widetilde{\Theta}((H^4(C_{\mathrm{cov}} + C_{\star}) / \varepsilon^2) \cdot \log(|\mathcal{F}||\mathcal{W}|/\delta))$ , $\mathcal{F}$ , augmented $\overline{\mathcal{W}}$ defined in Eq. (38), $\gamma = \widetilde{\Theta}\left(\sqrt{(C_{\star} + C_{\mathrm{cov}})/TH^2 \log(|\mathcal{F}||\mathcal{W}|/\delta)}\right)$ , and $\mathcal{D}_{\mathrm{off}}$ returns an $\varepsilon$ -suboptimal policy with probability at least $1 - \delta T$ after collecting

$$
N = \widetilde {O} \bigg (\frac {H ^ {2} (C _ {\mathrm{cov}} + C _ {\star})}{\varepsilon^ {2}} \log (| \mathcal {F} | | \mathcal {W} | / \delta) \bigg)
$$

trajectories.

Algorithm 3 HYGLOW: $\mathrm{H}_2\mathrm{O} + \mathrm{MABO.CR}$   
input: Parameter $T \in N$ , value function class F, weight function class W, parameter $\gamma \in [0,1]$ , offline datasets $D_{off} = \{D_{off,h}\}_{h}$ each of size T.

1: // For Corollary 4.1, set $T = \widetilde{\Theta}((H^{4}(C_{\text{cov}} + C_{\star}) / \varepsilon^{2}) \cdot \log(|F||W|/\delta))$ , and $\gamma = \widetilde{\Theta}\left(\sqrt{(C_{\star} + C_{\text{cov}})/TH^{2} \log(|F||W|/\delta)}\right)$ .

2: Set $\gamma^{(t)} = \gamma \cdot t$ , and $\alpha^{(t)} = 8/\gamma^{(t)}$ .

3: Initialize $D_{on,h}^{(1)} = D_{hybrid,h}^{(1)} = \varnothing$ for all $h \in [H]$ .

4: for $t = 1, \ldots, T$ do

5: Compute value function $f^{(t)}$ such that $f^{(t)} \in \arg\min_{f \in \mathcal{F}} \max_{w \in \mathcal{W}} \sum_{h=1}^{H} \left| \widehat{\mathbb{E}}_{\mathcal{D}_{\text{hybrid},h}^{(t)}} \left[ \check{w}_{h}(x_{h}, a_{h})[\widehat{\Delta}_{h} f](x_{h}, a_{h}, r_{h}, x'_{h+1}) \right] \right| - \alpha^{(t)} \widehat{\mathbb{E}}_{\mathcal{D}_{\text{hybrid},h}^{(t)}} \left[ \check{w}_{h}^{2}(x_{h}, a_{h}) \right]$ , (14)

where $\check{w} := \text{clip}_{\gamma^{(t)}}[w]$ and $[\widehat{\Delta}_{h} f](x, a, r, x') := f_{h}(x, a) - r - \max_{a'} f_{h+1}(x', a')$ .

6: Compute policy $\pi^{(t)} \leftarrow \pi_{f^{(t)}}$ .

7: Collect trajectory $(x_{1}, a_{1}, r_{1}), \ldots, (x_{H}, a_{H}, r_{H})$ using $\pi^{(t)}; D_{on,h}^{(t+1)} := D_{on,h}^{(t)} \cup \{(x_{h}, a_{h}, r_{h}, x_{h+1})\}$ .

8: Aggregate offline and online data: $D_{hybrid,h}^{(t+1)} := D_{off,h}|_{1:t} \cup D_{on,h}^{(t+1)}$ for all $h \in [H]$ .

9: output: policy $\widehat{\pi} = \text{Unif}(\pi^{(1)}, \ldots, \pi^{(T)})$ .

The realizability assumptions parallel those required that GLOW to obtain the analogous $C_{cov}/\varepsilon^{2}$ -type sample complexity for the purely online setting (cf. Assumption 2.2'). While the sample complexity matches that of GLOW, the computational efficiency is improved because we remove the need for global optimism. More specifically, note that when clipping and the absolute value signs are removed from Eq. (13), the optimization problem is concave-convex in the function class F and weight function class W, a desirable property shared by standard density-ratio based offline algorithms (Xie and Jiang, 2020; Zhan et al., 2022). $^{9}$ Thus, if F and W are parameterized as linear functions, (i.e., $\mathcal{F} = \{(x, a) \mapsto \langle \phi(x, a), \theta \rangle \mid \theta \in \Theta_{\mathcal{F}}\}$ and $\mathcal{W} = \{(x, a) \mapsto \langle \psi(x, a), \theta \rangle \mid \theta \in \Theta_{\mathcal{W}}\}$ for feature maps $\phi$ and $\psi)$ it can be solved in polynomial time using standard tools for minimax optimization (e.g., Nemirovski, 2004). To accommodate clipping and the absolute value signs efficiently, we note that Eq. (13) can be written as a convex-concave program in which the max player optimizes over the set $\widetilde{W}_{\gamma,h} := \{\pm clip_{\gamma n}[w_h] \mid w_h \in W_h\}$ . While this set may be complex, the result continues to hold if the max player optimizes over any expanded weight function class $W'$ that satisfies $\widetilde{W}_{\gamma} \subseteq W'$ and $\|w\|_{\infty} \leq \gamma n$ for all $w \in W'$ , thus allowing for the use of, e.g., convex relaxations or alternate parameterizations. We defer the details to Appendix F.1.2.

Remark 4.2 (Comparison to offline RL). It is instructive to compare the performance of HYGLOW to existing results for purely offline RL, which assume access to a data distribution $\nu$ with single-policy concentrability (Definition 4.4). Let us write $w^{\pi} := d^{\pi}/\nu$ , $w^{\star} := d^{\pi^{\star}}/\nu$ and $V^{\star}$ for the optimal value function. The most relevant work is the PRO-RL algorithm of Zhan et al. (2022). Their algorithm is computationally efficient and enjoys a polynomial sample complexity bound under the realizability of certain regularized versions of $w^{\star}$ and $V^{\star}$ . By contrast, our result requires $Q^{\star}$ -realizability and the density ratio realizability of $w^{\pi}$ for all $\pi \in \Pi$ , but for the unregularized problem. These assumptions are not comparable, and thus these results are best thought of as complementary. $^{10}$ However, our approach requires additional online access, while their algorithm does not.

To the best of our knowledge, all other algorithms for the purely offline setting that only require single-policy concentrability either need stronger representation conditions (such as value-function completeness (Xie et al., 2021a)), or are not known to be computationally efficient in the general function approximation setting due to the need for implementing pessimism (e.g., Chen and Jiang, 2022).

# 4.3 Generic Reductions from Online to Offline RL?

Our hybrid-to-offline reduction $H_{2}O$ and the CC-boundedness definition also shed light on the question of when offline RL methods can be lifted to the purely online setting. Indeed, observe that any offline algorithm which satisfies CC-boundedness (Definition 4.3) with only a $\widehat{\pi}$ -coverage term, namely which satisfies an offline risk bound of the form

$$
\mathbf {R i s k} _ {\text { off }} \leq \sum_ {h = 1} ^ {H} \frac {\mathfrak {a} _ {\gamma}}{n} \mathbb {E} _ {\widehat {\pi} \sim p} [ \mathrm{CC} _ {h} (\widehat {\pi}, \mu^ {(1: n)}, \gamma n) ] + \mathfrak {b} _ {\gamma}, \tag {15}
$$

can be repeatedly invoked within $H_{2}O$ (with $D_{off} = \varnothing$ ) to achieve a small $\sqrt{C_{cov}/T}$ -type risk bound for the purely online setting, with no hybrid data. This can be seen immediately by inspecting our proof for the hybrid setting (Theorem F.6 and Theorem 4.1).

We can think of algorithms satisfying Eq. (15) as optimistic offline RL algorithms, since their risk only scales with a term depending on their own output policy; this is typically achieved using optimism. In particular, it is easy to see, that GLOW and GOLF (Jin et al., 2021a; Xie et al., 2023) can be interpreted as repeatedly invoking such an optimistic offline RL algorithm within the $H_{2}O$ reduction. This class of algorithms has not been considered in the offline RL literature since they inherit both the computational drawbacks of pessimistic algorithms and the statistical drawbacks of “neutral” (i.e. non-pessimistic) algorithms (at least, when viewed only in the context of offline RL).

In more detail, as with pessimism, optimism is often not computationally efficient, although it furthermore requires all-policy concentrability (as opposed to single-policy concentrability) to obtain low offline risk. On the other hand, neutral (non-pessimistic) algorithms such as FQI (Chen and Jiang, 2019) and MABO (Xie and Jiang, 2020) also require all-policy concentrability, but are more computationally efficient. However, our reduction shows that these algorithms might merit further investigation. In particular, it uncovers that they can automatically solve the online setting (without hybrid data) under coverability and when repeatedly invoked on datasets generated from their previous policies. We find that this reduction advances the fundamental understanding of sample-efficient algorithms in both the online and offline settings, and are optimistic that this understanding can be used for future algorithm design.

# 5 Discussion

Our work shows for the first time that density ratio modeling has provable benefits for online reinforcement learning, and serves as step in a broader research program that aims to clarify connections between online and offline reinforcement learning. To this end, we highlight some exciting directions for future research.

Realizability. While our results show that density ratio realizability allows for sample complexity guarantees based on coverability that do not require Bellman completeness, the question of whether value function realizability alone is sufficient still remains.

Generic reductions from online to offline RL. Our hybrid-to-offline reduction, $H_{2}O$ , sheds light on the question of when and how existing offline RL methods can be adapted as-is to the hybrid setting. Are there more general principles under which offline RL methods can be adapted to online settings?

Practical and efficient online algorithms. Beyond the theoretical directions above, we are excited to explore the possibility of developing practical and computationally efficient online reinforcement learning algorithms based on density ratio modeling.

# Acknowledgements

AS thanks Sasha Rakhlin for useful discussions. AS acknowledges support from the Simons Foundation and NSF through award DMS-2031883, as well as from the DOE through award DE-SC0022199. Nan Jiang acknowledges funding support from NSF IIS-2112471 and NSF CAREER IIS-2141781.

# References

Alekh Agarwal, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. Taming the monster: A fast and simple algorithm for contextual bandits. In International Conference on Machine Learning, pages 1638–1646, 2014.   
Alekh Agarwal, Sham Kakade, Akshay Krishnamurthy, and Wen Sun. Flambe: Structural complexity and representation learning of low rank mdps. Advances in neural information processing systems, 33:20095–20107, 2020.   
András Antos, Csaba Szepesvári, and Rémi Munos. Learning near-optimal policies with bellman-residual minimization based fitted policy iteration and a single sample path. Machine Learning, 71(1):89–129, 2008.   
Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning with offline data. In International Conference on Machine Learning, pages 1577–1594. PMLR, 2023.   
Serkan Cabi, Sergio Gómez Colmenarejo, Alexander Novikov, Ksenia Konyushkova, Scott E. Reed, Rae Jeong, Konrad Zolna, Yusuf Aytar, David Budden, Mel Vecerík, Oleg Sushkov, David Barker, Jonathan Scholz, Misha Denil, Nando de Freitas, and Ziyu Wang. Scaling data-driven robotics with reward sketching and batch reinforcement learning. In Robotics: Science and Systems XVI, Virtual Event / Corvalis, Oregon, USA, July 12-16, 2020, 2020.   
Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning, 2019.   
Jinglin Chen and Nan Jiang. Offline reinforcement learning under value and density-ratio realizability: the power of gaps. In Uncertainty in Artificial Intelligence, pages 378–388. PMLR, 2022.   
Simon Du, Akshay Krishnamurthy, Nan Jiang, Alekh Agarwal, Miroslav Dudik, and John Langford. Provably efficient RL with rich observations via latent state decoding. In International Conference on Machine Learning, pages 1665–1674. PMLR, 2019.   
Simon S Du, Sham M Kakade, Ruosong Wang, and Lin F Yang. Is a good representation sufficient for sample efficient reinforcement learning? In International Conference on Learning Representations, 2020.   
Simon S Du, Sham M Kakade, Jason D Lee, Shachar Lovett, Gaurav Mahajan, Wen Sun, and Ruosong Wang. Bilinear classes: A structural framework for provable generalization in RL. International Conference on Machine Learning, 2021.   
Amir-massoud Farahmand, Csaba Szepesvári, and Rémi Munos. Error propagation for approximate policy and value iteration. Advances in Neural Information Processing Systems, 23, 2010.   
Yihao Feng, Lihong Li, and Qiang Liu. A kernel loss for solving the bellman equation. Advances in Neural Information Processing Systems, 32, 2019.   
Dylan J Foster, Sham M Kakade, Jian Qian, and Alexander Rakhlin. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487, 2021.   
Dylan J Foster, Akshay Krishnamurthy, David Simchi-Levi, and Yunzong Xu. Offline reinforcement learning: Fundamental barriers for value function approximation. In Conference on Learning Theory, pages 3489–3489. PMLR, 2022.   
Noah Golowich, Dhruv Rohatgi, and Ankur Moitra. Exploring and learning in sparse linear mdps without computationally intractable oracles. arXiv preprint arXiv:2309.09457, 2023.   
Audrey Huang, Jinglin Chen, and Nan Jiang. Reinforcement learning in low-rank mdps with density features. arXiv preprint arXiv:2302.02252, 2023.   
Nan Jiang and Alekh Agarwal. Open problem: The dependence of sample complexity lower bounds on planning horizon. In Conference On Learning Theory, pages 3395-3398. PMLR, 2018.   
Nan Jiang and Jiawei Huang. Minimax value interval for off-policy evaluation and policy optimization. Neural Information Processing Systems, 2020.

Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are pac-learnable. In International Conference on Machine Learning, pages 1704–1713. PMLR, 2017.   
Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory, pages 2137-2143, 2020.   
Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of RL problems, and sample-efficient algorithms. Neural Information Processing Systems, 2021a.   
Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline RL? In International Conference on Machine Learning, pages 5084–5096. PMLR, 2021b.   
Ilya Kostrikov, Ofir Nachum, and Jonathan Tompson. Imitation learning via off-policy distribution matching. In International Conference on Learning Representations, 2019.   
Akshay Krishnamurthy, Alekh Agarwal, and John Langford. PAC reinforcement learning with rich observations. In Advances in Neural Information Processing Systems, pages 1840–1848, 2016.   
Jongmin Lee, Wonseok Jeon, Byungjun Lee, Joelle Pineau, and Kee-Eung Kim. Optidice: Offline policy optimization via stationary distribution correction estimation. In International Conference on Machine Learning, pages 6120–6130. PMLR, 2021.   
Gen Li, Wenhao Zhan, Jason D Lee, Yuejie Chi, and Yuxin Chen. Reward-agnostic fine-tuning: Provable statistical benefits of hybrid reinforcement learning. arXiv preprint arXiv:2305.10282, 2023.   
Fanghui Liu, Luca Viano, and Volkan Cevher. Provable benefits of general coverage conditions in efficient online rl with function approximation. International Conference on Machine Learning, 2023.   
Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou. Breaking the curse of horizon: Infinite-horizon off-policy estimation. Advances in neural information processing systems, 31, 2018.   
Zakaria Mhammedi, Dylan J Foster, and Alexander Rakhlin. Representation learning with multi-step inverse kinematics: An efficient and optimal approach to rich-observation rl. International Conference on Machine Learning (ICML), 2023.   
Dipendra Misra, Mikael Henaff, Akshay Krishnamurthy, and John Langford. Kinematic state abstraction and provably efficient rich-observation reinforcement learning. arXiv preprint arXiv:1911.05815, 2019.   
Rémi Munos. Error bounds for approximate policy iteration. In International Conference on Machine Learning, 2003.   
Rémi Munos. Performance bounds in $\ell_p$ -norm for approximate value iteration. SIAM Journal on Control and Optimization, 2007.   
Rémi Munos and Csaba Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research, 2008.   
Ofir Nachum and Bo Dai. Reinforcement learning via fenchel-rockafellar duality. arXiv preprint arXiv:2001.01866, 2020.   
Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans. Algaedice: Policy gradient from arbitrary experience. arXiv preprint arXiv:1912.02074, 2019.   
Ashvin Nair, Abhishek Gupta, Murtaza Dalal, and Sergey Levine. Awac: Accelerating online reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359, 2020.   
Arkadi Nemirovski. Prox-method with rate of convergence O(1/t) for variational inequalities with Lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization, 15(1):229–251, 2004.   
Gergely Neu and Nneka Okolo. Efficient global planning in large mdps via stochastic primal-dual optimization. In International Conference on Algorithmic Learning Theory, pages 1101-1123. PMLR, 2023.

Gergely Neu and Ciara Pike-Burke. A unifying view of optimism in episodic reinforcement learning. Advances in Neural Information Processing Systems, 33:1392-1403, 2020.   
Asuman E Ozdaglar, Sarath Pattathil, Jiawei Zhang, and Kaiqing Zhang. Revisiting the linear-programming framework for offline rl with general function approximation. In International Conference on Machine Learning, pages 26769–26791. PMLR, 2023.   
Paria Rashidinejad, Banghua Zhu, Cong Ma, Jiantao Jiao, and Stuart Russell. Bridging offline reinforcement learning and imitation learning: A tale of pessimism. Advances in Neural Information Processing Systems, 34:11702–11716, 2021.   
Paria Rashidinejad, Hanlin Zhu, Kunhe Yang, Stuart Russell, and Jiantao Jiao. Optimal conservative offline rl with general function approximation via augmented lagrangian. In The Eleventh International Conference on Learning Representations, 2023.   
Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. In Advances in Neural Information Processing Systems, pages 2256-2264, 2013.   
Yuda Song, Yifei Zhou, Ayush Sekhari, J Andrew Bagnell, Akshay Krishnamurthy, and Wen Sun. Hybrid RL: Using both offline and online data can make RL efficient. International Conference on Learning Representations, 2023.   
Wen Sun, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Model-based RL in contextual decision processes: PAC bounds and exponential improvements over model-free approaches. In Conference on learning theory, pages 2898–2933. PMLR, 2019.   
Masatoshi Uehara, Jiawei Huang, and Nan Jiang. Minimax weight and Q-function learning for off-policy evaluation. In International Conference on Machine Learning, 2020.   
Masatoshi Uehara, Masaaki Imaizumi, Nan Jiang, Nathan Kallus, Wen Sun, and Tengyang Xie. Finite sample analysis of minimax offline reinforcement learning: Completeness, fast rates and first-order efficiency. arXiv:2102.02981, 2021.   
Andrew Wagenmaker and Aldo Pacchiano. Leveraging offline data in online reinforcement learning. In International Conference on Machine Learning, pages 35300–35338. PMLR, 2023.   
Ruosong Wang, Simon S Du, Lin Yang, and Sham Kakade. Is long horizon rl more difficult than short horizon rl? Advances in Neural Information Processing Systems, 33:9075–9085, 2020a.   
Ruosong Wang, Dean Foster, and Sham M Kakade. What are the statistical limits of offline RL with linear function approximation? In International Conference on Learning Representations, 2020b.   
Ruosong Wang, Russ R Salakhutdinov, and Lin Yang. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. Advances in Neural Information Processing Systems, 33, 2020c.   
Yuanhao Wang, Ruosong Wang, and Sham M Kakade. An exponential lower bound for linearly-realizable MDPs with constant suboptimality gap. Neural Information Processing Systems (NeurIPS), 2021.   
Gellért Weisz, Philip Amortila, and Csaba Szepesvári. Exponential lower bounds for planning in MDPs with linearly-realizable optimal action-value functions. In Algorithmic Learning Theory, pages 1237–1264. PMLR, 2021.   
Tengyang Xie and Nan Jiang. Q\* approximation schemes for batch reinforcement learning: A theoretical comparison. In Conference on Uncertainty in Artificial Intelligence, 2020.   
Tengyang Xie and Nan Jiang. Batch value-function approximation with only realizability. In International Conference on Machine Learning, pages 11404-11413. PMLR, 2021.   
Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-consistent pessimism for offline reinforcement learning. Advances in neural information processing systems, 34:6683–6694, 2021a.

Tengyang Xie, Nan Jiang, Huan Wang, Caiming Xiong, and Yu Bai. Policy finetuning: Bridging sample-efficient offline and online reinforcement learning. Advances in neural information processing systems, 34:27395–27407, 2021b.   
Tengyang Xie, Dylan J Foster, Yu Bai, Nan Jiang, and Sham M Kakade. The role of coverage in online reinforcement learning. International Conference on Learning Representations, 2023.   
Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans. Off-policy evaluation via the regularized lagrangian. Advances in Neural Information Processing Systems, 33:6551–6561, 2020.   
Andrea Zanette. Exponential lower bounds for batch reinforcement learning: Batch RL can be exponentially harder than online RL. In International Conference on Machine Learning, 2021.   
Andrea Zanette, Alessandro Lazaric, Mykel Kochenderfer, and Emma Brunskill. Learning near optimal policies with low inherent bellman error. In International Conference on Machine Learning, pages 10978–10989. PMLR, 2020.   
Wenhao Zhan, Baihe Huang, Audrey Huang, Nan Jiang, and Jason Lee. Offline reinforcement learning with realizability and single-policy concentrability. In Conference on Learning Theory, pages 2730–2775. PMLR, 2022.   
Ruiyi Zhang, Bo Dai, Lihong Li, and Dale Schuurmans. Gendice: Generalized offline estimation of stationary values. arXiv preprint arXiv:2002.09072, 2020.   
Xuezhou Zhang, Yuda Song, Masatoshi Uehara, Mengdi Wang, Alekh Agarwal, and Wen Sun. Efficient reinforcement learning in block mdps: A model-free representation learning approach. In International Conference on Machine Learning, pages 26517–26547. PMLR, 2022.   
Zihan Zhang, Xiangyang Ji, and Simon Du. Is reinforcement learning more difficult than bandits? a near-optimal algorithm escaping the curse of horizon. In Conference on Learning Theory, pages 4528-4531. PMLR, 2021.   
Yifei Zhou, Ayush Sekhari, Yuda Song, and Wen Sun. Offline data enhanced on-policy policy gradient with provable guarantees. arXiv preprint arXiv:2311.08384, 2023.

# Contents of Appendix

A Additional Related Work 19   
B Comparing Weight Function Realizability to Alternative Realizability Assumptions 20   
C Examples for GLOW 21   
D Technical Tools 23

D.1 Reinforcement Learning Preliminaries 23

E Proofs from Section 3 (Online RL) 24

E.1 Supporting Technical Results 24   
E.2 Main Technical Result: Bound on Cumulative Suboptimality for GLOW 29   
E.3 Proof of Theorem 3.1 32   
E.4 Proof of Theorem 3.2....33

F Proofs and Additional Results from Section 4 (Hybrid RL) 36

F.1 Examples for $\mathrm{H}_2\mathrm{O}$ 36   
F.2 Proofs for $\mathrm{H}_2\mathrm{O}$ (Theorem 4.1) 39   
F.3 Proofs for $\mathrm{H}_2\mathrm{O}$ Examples (Appendix F.1) 43

# A Additional Related Work

Online reinforcement learning. Xie et al. (2023) introduce the notion of coverability and provide regret bounds for online reinforcement learning under the assumption of access to a value function class F satisfying Bellman completeness. Liu et al. (2023) extend their result to more general coverage conditions under the same Bellman completeness assumption. Our work complements these results by providing guarantees based on coverability that do not require Bellman completeness.

To the best of our knowledge, our work is the first to provide provable sample complexity guarantees for online reinforcement learning that take advantage of the ability to model density ratios, but a number of closely related works bear mentioning. Recent work of Huang et al. (2023) considers the low-rank MDP model and provides an algorithms which takes advantage of a form of occupancy realizability. Occupancy realizability, while related to density ratio realizability, is stronger assumption in general: For example, the Block MDP model (Krishnamurthy et al., 2016; Du et al., 2019; Misra et al., 2019; Zhang et al., 2022; Mhammedi et al., 2023) admits a density ratio class of low complexity, but does not admit a small occupancy class. Overall, however, their results are somewhat complementary, as they do not require any form of value function realizability. A number of other recent works in online reinforcement learning also make use of occupancy measures, but restrict to linear function approximation (Neu and Pike-Burke, 2020; Neu and Okolo, 2023). Lastly, a number of works apply density ratio modeling in online settings with an empirical focus (Feng et al., 2019; Nachum et al., 2019), but do not address the exploration problem.

Offline reinforcement learning. Within the literature on offline reinforcement learning theory, density ratio modeling has been widely used as a means to avoid strong Bellman completeness requirements, with theoretical guarantees for policy evaluation (Liu et al., 2018; Uehara et al., 2020; Yang et al., 2020; Uehara et al., 2021) and policy optimization (Jiang and Huang, 2020; Xie and Jiang, 2020; Zhan et al., 2022; Chen and Jiang, 2022; Rashidinejad et al., 2023; Ozdaglar et al., 2023). A number of additional works investigate density ratio modeling with an empirical focus, and do not provide finite-sample guarantees (Nachum et al., 2019; Kostrikov et al., 2019; Nachum and Dai, 2020; Zhang et al., 2020; Lee et al., 2021).

Hybrid reinforcement learning. Song et al. (2023) were the first to show, theoretically, that the hybrid reinforcement learning model can lead to computational benefits over online and offline RL individually. Our reduction, $H_{2}O$ , can be viewed as generalization of their Hybrid Q-Learning algorithm, with their result corresponding to the special case in which FQI is applied as a base algorithm. Our guarantees under

coverability complement their guarantees based on bilinear rank. Other recent works on hybrid reinforcement learning in specialized settings (e.g., tabular MDPs or linear MDPs) include Wagenmaker and Pacchiano (2023); Li et al. (2023); Zhou et al. (2023).

# B Comparing Weight Function Realizability to Alternative Realizability Assumptions

In this section, we compare the density ratio realizability assumption in Assumption 2.2 to a number of alternative realizability assumptions.

Comparison to Bellman completeness. A traditional assumption in the analysis of value function approximation methods is to assume that $\mathcal{F}$ satisfies a representation condition called Bellman completeness, which asserts that $\mathcal{T}_h\mathcal{F}_{h + 1}\subseteq \mathcal{F}_h$ ; this assumption is significantly stronger than just assuming realizability of $Q^{\star}$ (Assumption 2.1), and has been used throughout offline RL (Antos et al., 2008; Chen and Jiang, 2019), and online RL (Zanette et al., 2020; Jin et al., 2021a; Xie et al., 2023).

Bellman completeness is incomparable to our density ratio realizability assumption. For example, the low-rank MDP model (Jin et al., 2020) in which $P_{h}(x^{\prime} \mid x, a) = \langle \phi_{h}(x, a), \psi_{h}(x^{\prime}) \rangle$ satisfies Bellman completeness when the feature map $\phi$ is known to the learner even if $\psi$ is unknown (but may not satisfy it otherwise), and satisfies weight function realizability when the feature map $\psi$ is known even if $\phi$ is unknown (but may not satisfy it otherwise). Examples C.1 and C.2 in Appendix C give further examples that satisfy weight function realizability but are not known to satisfy Bellman completeness.

Comparison to model-based realizability. Weight function realizability is strictly weaker than model-based realizability (e.g., Foster et al., 2021), in which one assumes access to a model class M of MDPs that contains the true MDP. $^{[11]}$ Since each MDP induces an occupancy for every policy, it is straightforward to see that given such a class, we can construct a weight function class W that satisfies Assumption 2.2 with

$$
\log | \mathcal {W} | \leq O (\log | \mathcal {M} | + \log | \Pi |),
$$

as well as a realizable value function class with $\log|\mathcal{F}|\leq O(\log|\mathcal{M}|)$ . On the other hand, weight function realizability does not imply model-based realizability; a canonical example that witnesses this is the Block MDP.

Example B.1 (Block MDP). In the well-studied Block MDP model (Krishnamurthy et al., 2016; Jiang et al., 2017; Du et al., 2019; Misra et al., 2019; Zhang et al., 2022; Mhammedi et al., 2023), there exists realizable value function class $\mathcal{F}$ and weight function class $\mathcal{W}$ with $\log |\mathcal{F}|, \log |\mathcal{W}| \lesssim \text{poly}(|\mathcal{S}|, |\mathcal{A}|, \log |\Phi|)$ , where $\mathcal{S}$ is the latent state space and $\Phi$ is a class of decoder functions. However, there does not exist a realizable model class with bounded statistical complexity (see discussion in, e.g., Mhammedi et al. (2023)).

Alternative forms of density ratio realizability. The following remarks concern slight variants of the density ratio realizability assumption.

Remark B.1 (Clipped density ratio realizability). Since GLOW only accesses the weight function class W through the clipped weight functions, we can replace Assumption 2.2 with the assumption that for all $\pi, \pi' \in \Pi$ , $t \leq T$ , and $h \in [H]$ , we have

$$
\operatorname{clip} _ {\gamma t} \left[ w _ {h} ^ {\pi ; \pi^ {\prime}} \right] \in \mathcal {W} _ {h},
$$

where $T \in \mathbb{N}$ and $\gamma \in [0,1]$ are chosen as in Theorem 3.2. Likewise, we can replace Assumption 2.2' with the assumption that for all $\pi \in \Pi$ , and for all $t \leq T$ , $h \in [H]$ , and $\pi^{(1:t)} \in \Pi^t$ , we have

$$
\operatorname{clip} _ {\gamma t} \left[ w _ {h} ^ {\pi ; \pi^ {(1: t)}} \right] \in \mathcal {W} _ {h},
$$

for $T\in \mathbb{N}$ and $\gamma \in [0,1]$ chosen as in Theorem 3.1.

Remark B.2 (Density ratio realizability relative to a fixed reference distribution). Assumption 2.2 is weaker than assuming access to a class W that can realize the ratio $d_{h}^{\pi}/\nu_{h}$ (or alternatively $\nu_{h}/d_{h}^{\pi}$ ) for all $\pi \in \Pi$ , where $\nu_{h}$ is an arbitrary fixed distribution (natural choices might include $\nu_{h} = \mu_{h}^{\star}$ or $\nu_{h} = d_{h}^{\pi^{\star}}$ ). Indeed, given access to such a class, the expanded class $W' := \{w/w' \mid w, w' \in W\}$ satisfies Assumption 2.2, and has $\log|\mathcal{W}'| \leq 2\log|\mathcal{W}|$ .

# C Examples for GLOW

In this section, we give two new examples in which our main results for GLOW, Theorems 3.1 and 3.2, can be applied: MDPs with low-rank density features and general value functions, and a class of MDPs we refer to as Generalized Block MDPs.

Example C.1 (Realizable $Q^{\star}$ with low-rank density features). Consider a setting in which (i) $Q^{\star} \in F$ , and (ii), there is a known feature map $\psi_{h}: X \times A \to R^{d}$ with $\|\psi_{h}(x,a)\|_{2} \leq 1$ such that for all $\pi \in \Pi$ , $d_{h}^{\pi}(x,a) = \langle \psi_{h}(x,a), \theta^{\pi} \rangle$ for an unknown parameter $\theta^{\pi} \in R^{d}$ with $\|\theta^{\pi}\|_{2} \leq 1$ . This assumption is sometimes referred to as low occupancy complexity (Du et al., 2021), and has been studied with and without known features. In this case, we have $C_{cov} \leq d$ , and one can construct a weight function class W that satisfies Assumption 2.2' with $\log|\mathcal{W}| \leq O(dH)$ (Huang et al., 2023). $^{12}$ As a result, Theorem 3.1 gives sample complexity $\widetilde{O}\big(H^{3}d^{2}\log|\mathcal{F}|/\varepsilon^{2}\big)$ . Note that while this setup requires that the occupancies themselves have low-rank structure, the class F can consist of arbitrary, potentially nonlinear functions (e.g., neural networks). We remark that when the feature map $\psi$ is not known, but instead belongs to a known class $\Psi$ , the result continues to hold, at the cost of expanding W to have size $\log|\mathcal{W}| \leq \widetilde{O}\big(dH + \log|\Psi|\big)$ .

This example is similar to but complementary to Huang et al. (2023), who give guarantees for reward-free exploration under low-rank occupancies. Their results do not require any form of value realizability, but require a low-rank MDP assumption (which, in particular, implies Bellman completeness). $^{13}$

Our next example concerns a generalization of the well-studied Block MDP framework (Krishnamurthy et al., 2016; Jiang et al., 2017; Du et al., 2019; Misra et al., 2019; Zhang et al., 2022; Mhammedi et al., 2023) that we refer to as the Generalized Block MDP.

Definition C.1 (Generalized Block MDP). A Generalized Block MDP $\mathcal{M} = (\mathcal{X},\mathcal{S},\mathcal{A},P_{\mathrm{latent}},R_{\mathrm{latent}},q,H,d_1)$ is comprised of an observation space $\mathcal{X}$ , latent state space $\mathcal{S}$ , action space $\mathcal{A}$ , latent space transition kernel $P_{\mathrm{latent}}: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$ , and emission distribution $q: \mathcal{S} \to \Delta(\mathcal{X})$ . The latent state space evolves based on the agent's action $a_h \in \mathcal{A}$ via the process

$$
r _ {h} \sim R _ {\text { latent }} (s _ {h}, a _ {h}), \quad s _ {h + 1} \sim P _ {\text { latent }} (\cdot | s _ {h}, a _ {h}), \tag {16}
$$

with $s_1 \sim d_1$ ; we refer to $s_h$ as the latent state. The latent state is not observed directly, and instead we observe observations $x_h \in \mathcal{X}$ generated by the emission process

$$
x _ {h} \sim q (\cdot \mid s _ {h}). \tag {17}
$$

We assume that the emission process satisfies the decodability property:

$$
\operatorname{supp} q (\cdot \mid s) \cap \operatorname{supp} q (\cdot \mid s ^ {\prime}) = \varnothing , \quad \forall s ^ {\prime} \neq s \in \mathcal {S}. \tag {18}
$$

Decodability implies that there exists a (unknown to the agent) decoder $\phi_{\star} \colon \mathcal{X} \to \mathcal{S}$ such that $\phi_{\star}(x_h) = s_h$ a.s. for all $h \in [H]$ , meaning that latent states can be uniquely decoded from observations. Prior work on the Block MDP framework (Krishnamurthy et al., 2016; Jiang et al., 2017; Du et al., 2019; Misra et al., 2019; Zhang et al., 2022; Mhammedi et al., 2023) assumes that the latent space $\mathcal{S}$ and action space $\mathcal{A}$ are finite, but allow the observation space $\mathcal{X}$ to be large or potentially infinite. They provide sample complexity guarantees that scale as poly(| $\mathcal{S}$ |, | $\mathcal{A}$ |, $H$ , log| $\Phi$ |, $\varepsilon^{-1}$ ), where $\Phi$ is a known class of decoders that contains $\phi_{\star}$ .

We use the term Generalized Block MDP to refer to Block MDPs in which the latent space not tabular, and can be arbitrarily large.

Example C.2 (Generalized Block MDPs with coverable latent states). We can use GLOW to give sample complexity guarantees for Generalized Block MDPs in which the latent space is large, but has low coverability. Let $\Pi_{\mathrm{latent}} = (\mathcal{S} \times [H] \to \Delta(\mathcal{A}))$ denote the set of all randomized policy that operate on the latent space. Assume that the following conditions hold:

- We have a value function class $\mathcal{F}_{\mathrm{latent}}$ such that $Q_{\mathrm{latent}}^{\star} \in \mathcal{F}$ , where $Q_{\mathrm{latent}}^{\star}$ is the optimal $Q$ -function for the latent space.   
- We have access to a class of latent space density ratios $\mathcal{W}_{\mathrm{latent}}$ such that for all $h \in [H]$ , and all $\pi, \pi' \in \Pi_{\mathrm{latent}}$ ,

$$
w _ {\mathrm{latent}, h} ^ {\pi , \pi^ {\prime}} (s, a) := \frac {d _ {\mathrm{latent} , h} ^ {\pi} (s , a)}{d _ {\mathrm{latent} , h} ^ {\pi^ {\prime}} (s , a)} \in \mathcal {W} _ {\mathrm{latent}},
$$

where $d_{\mathrm{latent},h}^{\pi} = \mathbb{P}^{\pi}(s_h = s,a_h = a)$ is the latent occupancy measure.

\- The latent coverability coefficient is bounded:

$$
C _ {\text { cov,latent }} := \inf _ {\mu_ {1}, \ldots , \mu_ {H} \in \Delta (\mathcal {S} \times \mathcal {A})} \sup _ {\pi \in \Pi_ {\text { latent }}, h \in [ H ]} \left\| \frac {d _ {\text { latent } , h} ^ {\pi}}{\mu_ {h}} \right\| _ {\infty}.
$$

We claim that whenever these conditions hold, analogous conditions hold in observation space (viewing the Generalized BMDP as a large MDP), allowing GLOW and Theorem 3.2 to be applied. Namely, we have:

- There exists a class $\mathcal{F}$ satisfying Assumption 2.1 in observation space such that $\log |\mathcal{F}| \leq O(\log |\mathcal{F}_{\mathrm{latent}}| + \log |\Phi|)$ .   
- There exists a weight function class $\mathcal{W}$ satisfying Assumption 2.2 in observation space such that $\log |\mathcal{W}| \leq O(\log |\mathcal{W}_{\mathrm{latent}}| + \log |\Phi|)$ .   
- We have $C_{\mathrm{cov}} \leq C_{\mathrm{cov,latent}}$ .

As a result, GLOW attains sample complexity poly( $C_{\text{cov. latent}}$ , $H$ , $\log|\mathcal{F}_{\text{latent}}|$ , $\log|\mathcal{W}_{\text{latent}}|$ , $\log|\Phi|$ , $\varepsilon^{-1}$ ). This generalizes existing results for Block MDPs with tabular latent state spaces, which have $\log|\mathcal{F}_{\text{latent}}|$ , $\log|\mathcal{W}_{\text{latent}}|$ , $C_{\text{cov, latent}} = \text{poly}(|\mathcal{S}|, |\mathcal{A}|, H)$ .

# D Technical Tools

Lemma D.1 (Azuma-Hoeffding). Let $M \in N$ and $(Y_{m})_{m \leq M}$ be a sequence of random variables adapted to a filtration $(\mathcal{F}_{m})_{m \leq M}$ . If $|Y_{m}| \leq R$ almost surely, then with probability at least $1 - \delta$ ,

$$
\left| \sum_ {m = 1} ^ {M} Y _ {m} - \mathbb {E} _ {m - 1} [ Y _ {m} ] \right| \leq R \cdot \sqrt {8 M \log (2 \delta^ {- 1})}.
$$

Lemma D.2 (Freedman's inequality (e.g., Agarwal et al., 2014)). Let $M \in \mathbb{N}$ and $(Y_m)_{m \leq M}$ be a real-valued martingale difference sequence adapted to a filtration $(\mathcal{F}_m)_{m \leq M}$ . If $|Y_m| \leq R$ almost surely, then for any $\eta \in (0,1/R)$ , with probability at least $1 - \delta$ ,

$$
\left| \sum_ {m = 1} ^ {M} Y _ {m} \right| \leq \eta \sum_ {m = 1} ^ {M} \mathbb {E} _ {m - 1} \left[ (Y _ {m}) ^ {2} \right] + \frac {\log (2 \delta^ {- 1})}{\eta}.
$$

The following lemma is a standard consequence of Lemma D.2 (e.g., Foster et al., 2021).

Lemma D.3. Let $M \in \mathbb{N}$ and $(Y_m)_{m \leq M}$ be a sequence of random variables adapted to a filtration $(\mathcal{F}_m)_{m \leq M}$ . If $0 \leq Y_m \leq R$ almost surely, then with probability at least $1 - \delta$ ,

$$
\sum_ {m = 1} ^ {M} Y _ {m} \leq \frac {3}{2} \sum_ {m = 1} ^ {M} \mathbb {E} _ {m - 1} [ Y _ {m} ] + 4 R \log (2 \delta^ {- 1}),
$$

and

$$
\sum_ {m = 1} ^ {M} \mathbb {E} _ {m - 1} [ Y _ {m} ] \leq 2 \sum_ {m = 1} ^ {M} Y _ {m} + 8 R \log (2 \delta^ {- 1}).
$$

# D.1 Reinforcement Learning Preliminaries

Lemma D.4 (Jiang et al. (2017, Lemma 1)). For any value function $f = (f_{1}, \ldots, f_{H})$ ,

$$
\mathbb {E} _ {x _ {1} \sim d _ {1}} [ f _ {1} (x _ {1}, \pi_ {f _ {1}} (x _ {1})) ] - J (\pi_ {f}) = \sum_ {h = 1} ^ {H} \mathbb {E} _ {d _ {h} ^ {\pi_ {f}}} [ f _ {h} (x _ {h}, a _ {h}) - [ \mathcal {T} _ {h} f _ {h + 1} ] (x _ {h}, a _ {h}) ].
$$

Lemma D.5 (Per-state-action elliptic potential lemma; Xie et al. (2023, Lemma 4)). Let $d^{(1)}, \ldots, d^{(T)}$ be an arbitrary sequence of distributions over a set $\mathcal{Z}$ , and let $\mu \in \Delta(\mathcal{Z})$ be a distribution such that $d^{(t)}(z) / \mu(z) \leq C$ for all $z \in \mathcal{Z}$ and $t \in [T]$ . Then, for all $z \in \mathcal{Z}$ , we have

$$
\sum_ {t = 1} ^ {T} \frac {d ^ {(t)} (z)}{\sum_ {i <   t} d ^ {(m)} (z) + C \mu (z)} \leq 2 \log (1 + T).
$$

# E Proofs from Section 3 (Online RL)

This section of the appendix is organized as follows:

- Appendix E.1 provides supporting technical results for GLOW, including concentration guarantees.   
- Appendix E.2 presents our main technical result for GLOW, Lemma E.4, which bounds the cumulative suboptimality of the iterates $\pi^{(1)},\ldots ,\pi^{(T)}$ produced by the algorithm for general choices of the parameters $T$ , $K$ , and $\gamma >0$ .   
- Finally, in Appendices E.3 and E.4, we invoke with specific parameter choices to prove Theorems 3.1 and 3.2, as well as more general results (Theorems $3.1'$ and $3.2'$ ) that allow for misspecification error.

# E.1 Supporting Technical Results

For $x, x' \in \mathcal{X}$ , $a \in \mathcal{A}$ , $r \in [0,1]$ , and $h \in [H]$ , recall the notation

$$
[ \widehat {\Delta} _ {h} f ] (x, a, r, x ^ {\prime}) = f _ {h} (x, a) - r - \max _ {a ^ {\prime}} f _ {h} (x ^ {\prime}, a ^ {\prime}),
$$

$$
[ \Delta_ {h} f ] (x, a) = f _ {h} (x, a) - [ \mathcal {T} _ {h} f _ {h + 1} ] (x, a),
$$

$$
\check {w} _ {h} (x, a) = \operatorname{clip} _ {\gamma^ {(t)}} [ w _ {h} ] (x, a).
$$

Lemma E.1 (Basic concentration for GLOW). Let $\gamma^{(t)} \geq 0$ for $t \in [T]$ . With probability at least $1 - \delta$ , all of the following inequalities hold for all $f \in \mathcal{F}$ , $w \in \mathcal{W}$ , $t \in [T]$ and $h \in [H]$ :

(a) $\left|\widehat{\mathbb{E}}_{\mathcal{D}_{h}^{(t)}}\left[\left[\widehat{\Delta}_{h}f\right](x_{h},a_{h},r_{h},x_{h+1}^{\prime})\cdot\check{w}_{h}(x_{h},a_{h})\right]-\mathbb{E}_{\bar{d}_{h}^{(t)}}\left[\left[\Delta_{h}f\right](x_{h},a_{h})\cdot\check{w}_{h}(x_{h},a_{h})\right]\right| \leq\frac{10}{3\gamma^{(t)}}\mathbb{E}_{\bar{d}_{h}^{(t)}}\left[\left(\check{w}_{h}(x_{h},a_{h})\right)^{2}\right]+\frac{\beta^{(t)}}{12},$   
(b) $\frac{1}{\gamma^{(t)}}\mathbb{E}_{\bar{d}_h^{\left(t\right)}}\big[\check{w}_h^2 (x_h,a_h)\big]\leq \frac{2}{\gamma^{(t)}}\widehat{\mathbb{E}}_{\mathcal{D}_h^{(t)}}\big[\check{w}_h^2 (x_h,a_h)\big] + \frac{2\beta^{(t)}}{9},$   
(c) $\frac{1}{\gamma^{(t)}}\widehat{\mathbb{E}}_{\mathcal{D}_h^{(t)}}\big[\check{w}_h^2 (x_h,a_h)\big]\leq \frac{3}{2\gamma^{(t)}}\mathbb{E}_{\bar{d}_h^{(t)}}\big[\check{w}_h^2 (x_h,a_h)\big] + \frac{\beta^{(t)}}{9},$   
(d) $J(\pi^{\star}) - \mathbb{E}_{x_1\sim d_1}[f_1(x_1,\pi_{f_1}(x_1))]\leq \widehat{\mathbb{E}}_{x_1\sim \mathcal{D}_1^{(t)}}[\max_aQ_1^\star (x_1,a_1) - f_1(x_1,\pi_{f_1}(x_1))] + \sqrt{\frac{8\log(6|\mathcal{F}||\mathcal{W}|TH / \delta)}{K(t - 1)}},$

where $\check{w}_{h} := \mathsf{clip}_{\gamma^{(t)}}[w_{h}]$ and $\beta^{(t)} := \frac{36\gamma^{(t)}}{K(t-1)} \log(6|\mathcal{F}||\mathcal{W}|TH/\delta)$ .

Proof of Lemma E.1. Fix any $h \in [H]$ and $t \in [T]$ . Let $M = K(t-1)$ and recall that the dataset $D_{h}^{t}$ consists of M tuples of the form $\{(x_{h}^{(m)}, a_{h}^{(m)}, r_{h}^{(m)}, x_{h+1}^{(m)})\}_{m \leq M}$ where $x_{h+1}^{(m)} \sim P(\cdot \mid x_{h}^{(m)}, a_{h}^{(m)})$ , and $a_{h}^{(m)} = \pi_{\tau(m)}(x_{h}^{(m)})$ where $\tau(m) = \lceil m/K \rceil$ . Fix any $f \in F$ and $w \in W$ .

Proof of (a). For each $m \in [M]$ , define the random variable

$$
Y _ {m} = [ \widehat {\Delta} _ {h} f ] (x _ {h} ^ {(m)}, a _ {h} ^ {(m)}, r _ {h} ^ {(m)}, x _ {h + 1} ^ {(m)}) \cdot \check {w} _ {h} (x _ {h} ^ {(m)}, a _ {h} ^ {(m)}) - \mathbb {E} _ {d _ {h} ^ {(\tau (m))}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ]
$$

Clearly, $\mathbb{E}_{m - 1}[Y_m] = 0$ and thus $\{Y_m\}_{m\leq M}$ is a martingale difference sequence with

$$
\left| Y _ {m} \right| \leq 3 \sup _ {x _ {h}, a _ {h}} \left| \check {w} _ {h} (x _ {h}, a _ {h}) \right| \leq 3 \gamma^ {(t)},
$$

since $|[ \widehat{\Delta}_h f](x_h^{(m)}, a_h^{(m)}, r_h^{(m)}, x_{h+1}^{(m)})| \leq 2$ and $|[ \Delta_h f](x_h, a_h)| \leq 1$ . Furthermore,

$$
\begin{array}{l} \sum_ {m = 1} ^ {M} Y _ {m} = \sum_ {m = 1} ^ {M} [ \widehat {\Delta} _ {h} f ] (x _ {h} ^ {(m)}, a _ {h} ^ {(m)}, r _ {h} ^ {(m)}, x _ {h + 1} ^ {(m)}) \cdot \check {w} _ {h} (x _ {h} ^ {(m)}, a _ {h} ^ {(m)}) - \sum_ {m = 1} ^ {M} \mathbb {E} _ {d _ {h} ^ {(\tau (m))}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ] \\ = K (t - 1) \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \Big [ [ \widehat {\Delta} _ {h} f ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \Big ] - K \sum_ {\tau = 1} ^ {t - 1} \mathbb {E} _ {d _ {h} ^ {(\tau)}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ] \\ \end{array}
$$

$$
= K (t - 1) \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \left[ [ \widehat {\Delta} _ {h} f ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] - K (t - 1) \mathbb {E} _ {\tilde {d} _ {h} ^ {(t)}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ].
$$

Additionally, we also have that

$$
\begin{array}{l} \mathbb {E} _ {m - 1} \big [ (Y _ {m}) ^ {2} \big ] \leq 2 \mathbb {E} _ {m - 1} \Big [ ([ \widehat {\Delta} _ {h} f ] (x _ {h} ^ {(m)}, a _ {h} ^ {(m)}, r _ {h} ^ {(m)}, x _ {h + 1} ^ {(m)}) \cdot \check {w} _ {h} (x _ {h} ^ {(m)}, a _ {h} ^ {(m)})) ^ {2} \Big ] \\ + 2 \mathbb {E} _ {m - 1} \left[ \left(\mathbb {E} _ {d _ {h} ^ {(\tau (m))}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ]\right) ^ {2} \right] \\ \leq \mathbb {E} _ {m - 1} \left[ 8 \check {w} _ {h} \left(x _ {h} ^ {(m)}, a _ {h} ^ {(m)}\right) ^ {2} + 2 \mathbb {E} _ {d _ {h} ^ {(\tau (m))}} \left[ \left([ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h})\right) ^ {2} \right] \right] \\ \leq \mathbb {E} _ {m - 1} \left[ 8 \check {w} _ {h} \left(x _ {h} ^ {(m)}, a _ {h} ^ {(m)}\right) ^ {2} + 2 \mathbb {E} _ {d _ {h} ^ {(\tau (m))}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] \right] \\ = 1 0 \mathbb {E} _ {d _ {h} ^ {(\tau (m))}} \Big [ \big (\check {w} _ {h} (x _ {h}, a _ {h}) \big) ^ {2} \Big ] \\ \end{array}
$$

where the second line follows since $|[ \widehat{\Delta}_h f](x_h^{(m)}, a_h^{(m)}, r_h^{(m)}, x_{h+1}^{(m)})| \leq 2$ and by using Jensen's inequality, and the third line uses $|[ \Delta_h f](x_h, a_h)| \leq 1$ .

Thus, using Lemma D.2 with $\eta = 1 / 3\gamma^{(t)}$ , we get that with probability at least $1 - \delta'$ ,

$$
\begin{array}{l} \left| \sum_ {m = 1} ^ {M} Y _ {m} \right| = K (t - 1) \Big | \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \Big [ [ \widehat {\Delta} _ {h} f ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \Big ] - \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ] \Big | \\ \leq \frac {1 0 K}{3 \gamma^ {(t)}} \sum_ {\tau = 1} ^ {t - 1} \mathbb {E} _ {d _ {h} ^ {(\tau)}} \left[ \left(\check {w} _ {h} (x _ {h}, a _ {h})\right) ^ {2} \right] + 3 \gamma^ {(t)} \log (2 / \delta^ {\prime}) \\ = \frac {1 0 K (t - 1)}{3 \gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + 3 \gamma^ {(t)} \log (2 / \delta^ {\prime}). \\ \end{array}
$$

The above bound implies that

$$
\begin{array}{l} \left| \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \left[ [ \widehat {\Delta} _ {h} f ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] - \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \left[ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ] \right. \right| \\ \leq \frac {1 0}{3 \gamma^ {(t)}} \mathbb {E} _ {\vec {d} _ {h} ^ {(t)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \frac {3 \gamma^ {(t)}}{K (t - 1)} \log (2 / \delta^ {\prime}). \\ \end{array}
$$

Plugging in the value of $\beta^{(t)}$ gives the desired bound. The final result follows by setting $\delta' = \delta/3|\mathcal{F}||\mathcal{W}|_{TH}$ , and taking another union bound over the choice of f, w, t and h.

Proof of (b) and (c). For each $m \in [M]$ , define the random variable

$$
Y _ {m} = \left(\check {w} _ {h} (x _ {h} ^ {(m)}, a _ {h} ^ {(m)})\right) ^ {2}.
$$

Clearly, the sequence $\{Y_m\}_{m\leq M}$ is adapted to an increasing filtration, with $Y_{t}\geq 0$ and $|Y_{t}| = |(\check{w}_{h}(x_{h}^{(m)},a_{h}^{(m)}))^{2}|\leq (\gamma^{(t)})^{2}$ . Furthermore,

$$
\sum_ {m = 1} ^ {M} \mathbb {E} _ {m - 1} [ Y _ {m} ] = K \sum_ {\tau = 1} ^ {t - 1} \mathbb {E} _ {d _ {h} ^ {(\tau)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ] = K (t - 1) \mathbb {E} _ {\check {d} _ {h} ^ {(t)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ],
$$

and,

$$
\sum_ {m = 1} ^ {M} Y _ {m} = \sum_ {(x, a) \in \mathcal {D} _ {h} ^ {(t)}} (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} = K (t - 1) \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ].
$$

Thus, by Lemma D.3, we have that with probability at least $1 - \delta'$ ,

$$
\mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ] \leq 2 \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ] + \frac {8 (\gamma^ {(t)}) ^ {2} \log (2 / \delta^ {\prime})}{K (t - 1)},
$$

and

$$
\widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ] \leq \frac {3}{2} \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ] + \frac {4 (\gamma^ {(t)}) ^ {2} \log (2 / \delta^ {\prime})}{K (t - 1)}.
$$

The final result follows by setting $\delta' = \delta/3|\mathcal{F}||\mathcal{W}|_{TH}$ , and taking another union bound over the choice of f, w, t and h.

Proof of $(d)$ . For each $m \in [M]$ , define the random variable

$$
Y _ {m} = \max _ {a} Q _ {1} ^ {\star} (x _ {1} ^ {(m)}, a) - f _ {1} (x _ {1} ^ {(m)}, \pi_ {f _ {1}} (x _ {1} ^ {(m)})).
$$

Clearly, $|Y_{m}| \leq 1$ . Thus, using Lemma D.1, we get that with probability at least $1 - \delta'$ ,

$$
\sum_ {m = 1} ^ {M} \mathbb {E} _ {m - 1} [ Y _ {m} ] \leq \sum_ {m = 1} ^ {M} \left(\max _ {a} Q _ {1} ^ {\star} (x _ {1} ^ {(m)}, a) - f _ {1} (x _ {1} ^ {(m)}, \pi_ {f _ {1}} (x _ {1} ^ {(m)}))\right) + \sqrt {8 M \log (2 / \delta^ {\prime})}.
$$

Setting $M = K(t - 1)$ and noting that $\mathbb{E}_{m-1}[Y_m] = J(\pi^\star) - \mathbb{E}_{x_1 \sim d_1}[f_1(x_1, \pi_{f_1}(x_1)]$ since $x_1^{(m)} \sim d_1$ for any $m \in [M]$ , we get that

$$
J (\pi^ {\star}) - \mathbb {E} _ {x _ {1} \sim d _ {1}} [ f _ {1} (x _ {1}, \pi_ {f _ {1}} (x _ {1}) ] \leq \mathbb {E} _ {x \sim \mathcal {D} _ {1} ^ {(t)}} \Bigl [ \max _ {a} Q _ {1} ^ {\star} (x _ {1}, a) - f _ {1} (x _ {1}, \pi_ {f _ {1}} (x _ {1})) \Bigr ] + \sqrt {\frac {8 \log (2 / \delta^ {\prime})}{K (t - 1)}}
$$

The final result follows by setting $\delta' = \delta/3|\mathcal{F}||\mathcal{W}|TH$ , and taking another union bound over the choice of $f, w, t$ and $h$ .

Lemma E.2 (Properties of GLOW confidence set). Let $\gamma^{(t)} \geq 0$ for $t \in [T]$ . With probability at least $1 - \delta$ , all of the following events hold:

(a) For all $t \geq 1$ , $Q^{\star} \in \mathcal{F}^{(t)}$   
(b) For all $t \geq 2$ , $h \in [H]$ , $f \in \mathcal{F}^{(t)}$ , and $w \in \mathcal{W}$ , we have

$$
\mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \left[ \left[ \Delta_ {h} f \right] \left(x _ {h}, a _ {h}\right) \cdot \check {w} _ {h} \left(x _ {h}, a _ {h}\right) \right] \leq \frac {2 0}{\gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] + \frac {7 \beta^ {(t)}}{1 8}.
$$

Furthermore,

$$
\mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left[ \Delta_ {h} f \right] \left(x _ {h}, a _ {h}\right) \cdot \check {w} _ {h} \left(x _ {h}, a _ {h}\right) \right] \leq \frac {4 0}{\gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] + \frac {7 \beta^ {(t)}}{9} + \frac {\gamma^ {(t)}}{1 6 0 t ^ {2}},
$$

(c) For all $t \geq 2$ , we have

$$
\mathbb {E} _ {x _ {1} \sim d _ {1}} \Big [ \max _ {a} Q _ {1} ^ {\star} (x _ {1}, a) - f _ {1} ^ {(t)} (x _ {1}, \pi_ {1} ^ {(t)} (x _ {1})) \Big ] \leq \sqrt {\frac {8 \log (6 | \mathcal {F} | | \mathcal {W} | T H / \delta)}{K (t - 1)}},
$$

where $\check{w}_{h} := clip_{\gamma^{(t)}}[w_{h}]$ and $\beta^{(t)} = \frac{36\gamma^{(t)}}{K(t-1)} \log(6|\mathcal{F}||\mathcal{W}|TH/\delta)$ .

Proof of Lemma E.2. Using Lemma E.1, we have that with probability at least $1 - \delta$ , for all $f \in \mathcal{F}$ , $w \in \mathcal{W}$ , $t \in [T]$ and $h \in [H]$ ,

$$
\begin{array}{l} \left| \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \Big [ [ \widehat {\Delta} _ {h} f ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \Big ] - \mathbb {E} _ {\vec {d} _ {h} ^ {(t)}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ] \right| \\ \leq \frac {1 0}{3 \gamma^ {(t)}} \mathbb {E} _ {\tilde {d} _ {h} ^ {(t)}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] + \frac {\beta^ {(t)}}{1 2}, \tag {19} \\ \end{array}
$$

$$
\frac {1}{\gamma^ {(t)}} \mathbb {E} _ {\tilde {d} _ {h} ^ {(t)}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] \leq \frac {2}{\gamma^ {(t)}} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] + \frac {2 \beta^ {(t)}}{9}, \tag {20}
$$

$$
\frac {1}{\gamma^ {(t)}} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] \leq \frac {3}{2 \gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] + \frac {\beta^ {(t)}}{9}, \tag {21}
$$

and,

$$
\begin{array}{l} J (\pi^ {\star}) - \mathbb {E} _ {d _ {1}} [ f _ {1} (x _ {1}, \pi_ {f _ {1}} (x _ {1}) ] \leq \widehat {\mathbb {E}} _ {\mathcal {D} _ {1} ^ {(t)}} \Bigl [ \max _ {a} Q _ {1} ^ {\star} (x _ {1}, a) - f _ {1} (x _ {1}, \pi_ {f _ {1}} (x _ {1})) \Bigr ] \\ + \sqrt {\frac {8 \log (6 | \mathcal {F} | | \mathcal {W} | T H / \delta)}{K (t - 1)}}. \tag {22} \\ \end{array}
$$

For the rest of the proof, we condition on the event in which (19-22) hold.

Proof of $(a)$ . Consider any $t \in [T]$ , and observe that the optimal state-action value function $Q^{\star}$ satisfies for any $w_{h} \in W$ ,

$$
\mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \Big [ (Q _ {h} ^ {\star} (x _ {h}, a _ {h}) - r _ {h} - \max _ {a ^ {\prime}} Q _ {h + 1} ^ {\star} (x _ {h + 1} ^ {\prime}, a ^ {\prime})) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \Big ] = 0,
$$

where $\check{w}_h := \mathsf{clip}_{\gamma^{(t)}}[w_h]$ . Using the above relation with (19), we get that

$$
\begin{array}{l} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \left[ (Q _ {h} ^ {\star} (x _ {h}, a _ {h}) - r _ {h} - \max _ {a ^ {\prime}} Q _ {h + 1} ^ {\star} (x _ {h + 1} ^ {\prime}, a ^ {\prime})) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \leq \frac {1 0}{3 \gamma^ {(t)}} \mathbb {E} _ {\tilde {d} _ {h} ^ {(t)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \frac {\beta^ {(t)}}{1 2} \\ \leq \frac {4}{\gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \frac {\beta^ {(t)}}{1 2} \\ \leq \frac {8}{\gamma^ {(t)}} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \beta^ {(t)}, \\ \end{array}
$$

where the second-last inequality follows from (20).

Plugging in the values of $\alpha^{(t)}$ and $\beta^{(t)}$ , rearranging the terms, we get that

$$
\widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \Big [ (Q _ {h} ^ {\star} (x _ {h}, a _ {h}) - r _ {h} - \max _ {a ^ {\prime}} Q _ {h + 1} ^ {\star} (x _ {h + 1} ^ {\prime}, a ^ {\prime})) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) - \alpha^ {(t)} (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \Big ] \leq \beta^ {(t)}.
$$

Since the above inequality holds for all $w \in \mathcal{W}$ , we have that $Q^{\star} \in \mathcal{F}^{(t)}$ .

Proof of (b). Fix any $t$ , and note that by the definition of $\mathcal{F}^{(t)}$ , any $f \in \mathcal{F}^{(t)}$ satisfies for any $w \in \mathcal{W}$ , the bound

$$
\widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \left[ (f _ {h} (x _ {h}, a _ {h}) - r _ {h} - \max _ {a ^ {\prime}} f _ {h + 1} (x _ {h + 1} ^ {\prime}, a ^ {\prime})) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \leq \frac {1 0}{\gamma^ {(t)}} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \beta^ {(t)}.
$$

Using the above bound with (19), we get that

$$
\mathbb {E} _ {\vec {d} _ {h} ^ {(t)}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ] \leq \frac {1 0}{3 \gamma^ {(t)}} \mathbb {E} _ {\vec {d} _ {h} ^ {(t)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ] + \frac {1 0}{\gamma^ {(t)}} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h} ^ {(t)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ] + \frac {1 3}{1 2} \beta^ {(t)}.
$$

Plugging the bound from (21) for the second term above, we get that

$$
\mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ] \leq \frac {2 0}{\gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \big [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \big ] + \frac {7 \beta^ {(t)}}{1 8}.
$$

Finally, noting that $\bar{d}^{(t + 1)} = \frac{(t - 1)\bar{d}^{(t)} + d^{(t)}}{t}$ , we can further upper bound as:

$$
\begin{array}{l} \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left[ \Delta_ {h} f \right] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \\ \leq \frac {t - 1}{t} \left(\frac {2 0}{\gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \frac {7 \beta^ {(t)}}{1 8}\right) + \frac {1}{t} \mathbb {E} _ {d _ {h} ^ {(t)}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) ] \\ \leq 2 \left(\frac {2 0}{\gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t)}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] + \frac {7 \beta^ {(t)}}{1 8}\right) + \frac {1}{t} \mathbb {E} _ {d _ {h} ^ {(t)}} [ | \check {w} _ {h} \left(x _ {h}, a _ {h}\right) | ] \\ \leq \frac {4 0}{\gamma^ {(t)}} \mathbb {E} _ {\vec {d} _ {h} ^ {(t)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \frac {7 \beta^ {(t)}}{9} + \frac {4 0}{\gamma^ {(t)}} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \check {w} _ {h} (x _ {h}, a _ {h}) ^ {2} \right] + \frac {\gamma^ {(t)}}{1 6 0 t ^ {2}} \\ = \frac {4 0}{\gamma^ {(t)}} \mathbb {E} _ {\vec {d} _ {h} ^ {(t + 1)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \frac {7 \beta^ {(t)}}{9} + \frac {\gamma^ {(t)}}{1 6 0 t ^ {2}}, \\ \end{array}
$$

where the second-to-last line follows from an application of AM-GM inequality.

Proof of (c). Plugging in $f = f^{(t)}$ in (22) and noting that $\max_{a} Q_{1}^{\star}(x_{1}, a) = Q_{1}^{\star}(x, \pi^{\star}(x))$ for any $x \in \mathcal{X}$ , we get that

$$
J (\pi^ {\star}) - \mathbb {E} _ {x _ {1} \sim d _ {1}} \Big [ f _ {1} ^ {(t)} (x _ {1}, \pi_ {f _ {1} ^ {(t)}} (x _ {1})) \Big ] \leq \widehat {\mathbb {E}} _ {x _ {1} \sim \mathcal {D} _ {1} ^ {(t)}} \Big [ Q _ {1} ^ {\star} (x _ {1}, \pi_ {1} ^ {\star} (x _ {1})) - f _ {1} ^ {(t)} (x _ {1}, \pi_ {f _ {1} ^ {(t)}} (x _ {1})) \Big ] + \sqrt {\frac {8 \log (6 | \mathcal {F} | | \mathcal {W} | T H / \delta)}{K (t - 1)}}.
$$

However, note that by definition, $f^{(t)} \in \arg \max_f \widehat{\mathbb{E}}[f_1(x_1, \pi_{f_1}(x_1)]$ , and using part-(a), $Q^\star \in \mathcal{F}^{(t)}$ . Thus, $\widehat{\mathbb{E}}_{\mathcal{D}_1^{(t)}}\left[Q_1^\star (x_1,\pi_1^\star (x_1)) - f_1^{(t)}(x_1,\pi_{f_1^{(t)}}(x_1))\right] \leq 0$ , which implies that

$$
J (\pi^ {\star}) - \mathbb {E} _ {x _ {1} \sim d _ {1}} \Big [ f _ {1} ^ {(t)} (x _ {1}, \pi_ {f _ {1} ^ {(t)}} (x _ {1})) \Big ] \leq \sqrt {\frac {8 \log (6 | \mathcal {F} | | \mathcal {W} | T H / \delta)}{K (t - 1)}}.
$$

![](images/e3a53851221958bb26b316e283547930c667e1a76ded5bf5b98cb76a766ccd80.jpg)

Lemma E.3 (Coverability potential bound). Let $d^{(1)},\ldots,d^{(T)}$ be an arbitrary sequence of distributions over $X\times A$ , such that there exists a distribution $\mu\in\Delta(\mathcal{X}\times\mathcal{A})$ that satisfies $\|d^{(t)}/\mu\|_{\infty}\leq C$ for all $(x,a)\in\mathcal{X}\times\mathcal{A}$ and $t\in[T]$ . Then,

$$
\sum_ {t = 1} ^ {T} \mathbb {E} _ {(x, a) \sim d ^ {(t)}} \left[ \frac {d ^ {(t)} (x , a)}{\widetilde {d} ^ {(t + 1)} (x , a)} \right] \leq 5 C \log (T),
$$

where recall that $\widetilde{d}^{(t + 1)}\coloneqq \sum_{s = 1}^{t}d^{(t)}$ for all $t\in [T]$

Proof of Lemma E.3. Let $\tau(x, a) = \min\{t \mid \widetilde{d}^{(t+1)}(x, a) \geq C\mu(x, a)\}$ . With this definition, we can bound

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} \mathbb {E} _ {d ^ {(t)}} \left[ \frac {d ^ {(t)} (x , a)}{\widetilde {d} ^ {(t + 1)} (x , a)} \right] = \sum_ {t = 1} ^ {T} \mathbb {E} _ {d ^ {(t)}} \left[ \frac {d ^ {(t)} (x , a)}{\widetilde {d} ^ {(t + 1)} (x , a)} \cdot \mathbb {I} \{t <   \tau (x, a) \} \right] \\ + \sum_ {t = 1} ^ {T} \mathbb {E} _ {d ^ {(t)}} \left[ \frac {d ^ {(t)} (x , a)}{\widetilde {d} ^ {(t + 1)} (x , a)} \cdot \mathbb {I} \{t \geq \tau (x, a) \} \right] \\ \leq \underbrace {\sum_ {t = 1} ^ {T} \mathbb {E} _ {d ^ {(t)}} [ \mathbb {I} \{t <   \tau (x , a) \} ]} _ {\text {(I): burn - in phase}} + \underbrace {\sum_ {t = 1} ^ {T} \mathbb {E} _ {d ^ {(t)}} \left[ \frac {d ^ {(t)} (x , a)}{\widetilde {d} ^ {(t + 1)} (x , a)} \cdot \mathbb {I} \{t \geq \tau (x , a) \} \right]} _ {\text {(II): stable phase}}, \\ \end{array}
$$

where the second line uses that $d^{(t)}(x,a) / \widetilde{d}^{(t + 1)}(x,a) \leq 1$ .

For the burn-in phase, note that

$$
\sum_ {t = 1} ^ {T} \mathbb {E} _ {d ^ {(t)}} [ \mathbb {I} \{t <   \tau (x, a) \} ] = \sum_ {x, a} \sum_ {t <   \tau (x, a)} d ^ {t} (x, a) = \sum_ {x, a} \widetilde {d} ^ {(\tau (x, a))} (x, a) \leq \sum_ {x, a} C \mu (x, a) = C,
$$

where the last inequality uses that by definition, $\tilde{d}^{t}(x,a)\leq C\mu(x,a)$ for all $t\leq\tau(x,a)$ .

For the stable phase, whenever $t \geq \tau(x, a)$ , by definition, we have $\widetilde{d}^{(t+1)}(x, a) \geq C\mu(x, a)$ which implies that $\widetilde{d}^{(t+1)}(x, a) \geq \frac{1}{2}(\widetilde{d}^{(t+1)}(x, a) + C\mu(x, a))$ . Thus,

$$
\begin{array}{l} \text {(II)} \leq 2 \sum_ {t = 1} ^ {T} \mathbb {E} _ {d ^ {(t)}} \left[ \frac {d ^ {(t)} (x , a)}{\widetilde {d} ^ {(t)} (x , a) + C \mu (x , a)} \right] \tag {23} \\ = 2 \sum_ {x, a} \sum_ {t = 1} ^ {T} \frac {d ^ {(t)} (x , a) \cdot d ^ {(t)} (x , a)}{\widetilde {d} ^ {(t)} (x , a) + C \mu (x , a)} \\ \leq 2 \sum_ {x, a} \max _ {t ^ {\prime} \in [ T ]} d ^ {(t ^ {\prime})} (x, a) \max _ {x, a} \left(\sum_ {t = 1} ^ {T} \frac {d ^ {(t)} (x , a)}{\widetilde {d} ^ {(t)} (x , a) + C \mu (x , a)}\right). \\ \end{array}
$$

Using the per-state elliptical potential lemma (Lemma D.5) in the above inequality, we get that

$$
(\mathrm{II}) \leq 4 \log (T + 1) \sum_ {x, a} \max _ {t ^ {\prime} \in [ T ]} d ^ {(t ^ {\prime})} (x, a) \leq 4 C \log (T + 1) \sum_ {x, a} \mu (x, a) = 4 C \log (1 + T), \tag {24}
$$

where the second inequality follows from the fact that $\left\|\frac{d^{(t)}}{\mu}\right\|_{\infty}\leq C$ (by definition), and the last equality uses that $\sum_{x,a}\mu(x,a)=1$ . Combining the above bound, we get that

$$
\sum_ {t = 1} ^ {T} \mathbb {E} _ {d ^ {(t)}} \left[ \frac {d ^ {(t)} (x , a)}{\widetilde {d} ^ {(t + 1)} (x , a)} \right] \leq 5 C \log (T).
$$

![](images/edf870f170c2968d2faf4d7c2ad5feaa53a0737db20de20f864a4ba6f19b2bd1.jpg)

# E.2 Main Technical Result: Bound on Cumulative Suboptimality for GLOW

In this section we prove a key technical lemma, Lemma E.4, which gives a bound on the cumulative suboptimality of the sequence of policies generated by GLOW. Both the proofs of Theorem 3.2 and Theorem 3.1 build on this result. To facilitate more general sample complexity bounds that allow for misspecification error in W. In particular, for each $t \in [T]$ , we define $\xi_{t}$ as the misspecification error of the clipped density ratio $d_{h}^{(t)} / \bar{d}_{h}^{(t+1)}$ in class $W_{h}$ , defined as

$$
\xi_ {t} := \sup _ {h \in [ H ]} \inf _ {w \in \mathcal {W} _ {h}} \sup _ {\pi \in \Pi} \left\| \operatorname{clip} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)}}{\overline {{d}} _ {h} ^ {(t + 1)}} \right] - \operatorname{clip} _ {\gamma^ {(t)}} [ w _ {h} ] \right\| _ {1, \bar {d} _ {h} ^ {\pi}}, \tag {25}
$$

where recall that for any function $u: \mathcal{X} \times \mathcal{A} \mapsto \mathbb{R}$ and distribution $d \in \Delta(\mathcal{X} \times \mathcal{A})$ , the norm $\|u\|_{1,d} := \mathbb{E}_{(x,a) \sim d}[|u(x,a)|]$ . Note that under Assumption 2.2 or 2.2', $\xi_t = 0$ for all $t \in [T]$ .

Lemma E.4 (Bound on cumulative suboptimality). Let $\pi^{(1)},\ldots,\pi^{(T)}$ be the sequence of policies generated by GLOW, when executed on classes F and W with parameters T,K and $\gamma$ . Then the cumulative suboptimality of the sequence of policies $\{\pi^{(t)}\}_{t\in[T]}$ is bounded as

$$
\sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) = O \left(H \left(\frac {C _ {\mathrm{cov}} \log (1 + T)}{\gamma} + \frac {\gamma T \log (| \mathcal {F} | | \mathcal {W} | H T \delta^ {- 1})}{K} + \sum_ {t = 1} ^ {T} \xi_ {t} + \gamma \log (T)\right)\right).
$$

Proof of Lemma E.4. Fix any $t \geq 2$ . We begin by establishing optimism as follows:

$$
\begin{array}{l} J (\pi^ {\star}) - J (\pi^ {(t)}) = \mathbb {E} _ {x _ {1} \sim d _ {1}} \left[ \max _ {a} Q _ {1} ^ {\star} (x _ {1}, a) \right] - J (\pi^ {(t)}) \\ = \mathbb {E} _ {x _ {1} \sim d _ {1}} \left[ \max _ {a} Q _ {1} ^ {\star} (x _ {1}, a) - f _ {1} ^ {(t)} (x _ {1}, \pi^ {(t)} (x _ {1})) \right] + \mathbb {E} _ {x _ {1} \sim d _ {1}} \left[ f _ {1} ^ {(t)} (x _ {1}, \pi^ {(t)} (x _ {1})) \right] - J (\pi^ {(t)}) \\ \leq \sqrt {\frac {8 \log (6 | \mathcal {F} | | \mathcal {W} | T H / \delta)}{K (t - 1)}} + \mathbb {E} _ {x _ {1} \sim d _ {1}} \left[ f _ {1} ^ {(t)} (x _ {1}, \pi^ {(t)} (x _ {1})) \right] - J (\pi^ {(t)}), \\ \end{array}
$$

where the last line follows from Lemma E.2-(c). Using Lemma D.4 for the second term, we get that

$$
J (\pi^ {\star}) - J (\pi^ {(t)}) \leq \sqrt {\frac {8 \log (6 | \mathcal {F} | | \mathcal {W} | T H / \delta)}{K (t - 1)}} + \sum_ {h = 1} ^ {H} \mathbb {E} _ {(x _ {h}, a _ {h}) \sim d _ {h} ^ {(t)}} [ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) ],
$$

where recall that $[\Delta_{h}f^{(t)}](x_{h},a_{h}):=f_{h}^{(t)}(x_{h},a_{h})-[\mathcal{T}_{h}f_{h+1}^{(t)}](x_{h},a_{h})$ . Thus,

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} J \left(\pi^ {\star}\right) - J \left(\pi^ {(t)}\right) \leq J \left(\pi^ {\star}\right) - J \left(\pi^ {(1)}\right) + \sum_ {t = 2} ^ {T} J \left(\pi^ {\star}\right) - J \left(\pi^ {(t)}\right) \\ \leq 1 + \sum_ {t = 2} ^ {T} \sqrt {\frac {8 \log (6 | \mathcal {F} | | \mathcal {W} | T H / \delta)}{K (t - 1)}} + \sum_ {t = 2} ^ {T} \sum_ {h = 1} ^ {H} \mathbb {E} _ {(x _ {h}, a _ {h}) \sim d _ {h} ^ {(t)}} [ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) ], \tag {26} \\ \end{array}
$$

where the second inequality uses that $J(\pi^{\star}) \leq 1$ and $J(\pi^{(1)}) \geq 0$ .

We next bound the expected Bellman error terms that appear in the right-hand-side above. Consider any $t \geq 2$ and $h \in [H]$ , and note that via a straightforward change of measure,

$$
\mathbb {E} _ {d _ {h} ^ {(t)}} [ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) ] = \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \bigg [ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) \cdot \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\bar {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \bigg ]
$$

Since $u \leq \min\{u, v\} + u\mathbb{I}\{u \geq v\}$ for any $u, v$ , we further decompose as

$$
\begin{array}{l} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \left[ \Delta_ {h} f ^ {(t)} \right] (x _ {h}, a _ {h}) \right] \leq \underbrace {\mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left[ \Delta_ {h} f ^ {(t)} \right] (x _ {h} , a _ {h}) \cdot \min \left\{\frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\bar {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} , \gamma^ {(t)} \right\} \right]} _ {\text {(A): Expected clipped Bellman error}} \\ + \underbrace {\mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \mathbb {I} \left\{\frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\overline {{d}} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \geq \gamma^ {(t)} \right\} \right]} _ {\text {(B): clipping violation}}, \\ \end{array}
$$

where in the second term we have changed the measure back to $d_h^{(t)}$ and used that $|[\Delta_h f^{(t)}](x_h, a_h)| \leq 1$ . We bound the terms (A) and (B) separately below.

Bound on expected clipped Bellman error. Let $w_{h}^{(t)}(x_{h}, a_{h}) \in \mathcal{W}$ denote a weight function which satisfies

$$
\sup _ {\pi} \left\| \operatorname{clip} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)}}{\bar {d} _ {h} ^ {(t + 1)}} \right] - \operatorname{clip} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} \right] \right\| _ {1, d _ {h} ^ {\pi}} \leq \xi_ {t}, \tag {27}
$$

which is guaranteed to exist by the definition of $\xi_t$ . Then, we have

$$
\begin{array}{l} (\mathrm{A}) = \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) \cdot \operatorname{clip} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\bar {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right] \right] \\ \leq \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) \cdot \mathsf {c l i p} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} (x _ {h}, a _ {h}) \right] \right] + \left\| \mathsf {c l i p} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)}}{\bar {d} _ {h} ^ {(t + 1)}} \right] - \mathsf {c l i p} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} \right] \right\| _ {1, \bar {d} _ {h} ^ {(t + 1)}} \\ \end{array}
$$

$$
\leq \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) \cdot \mathsf {c l i p} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} (x _ {h}, a _ {h}) \right] \right] + \xi_ {t},
$$

where the second line uses that $|[\Delta_h f^{(t)}](x_h, a_h)| \leq 1$ , and the last line plugs in (27). Next, using Lemma E.2-(b) in the above inequality, we get that

$$
(\mathrm{A}) \leq \frac {4 0}{\gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left(\operatorname{clip} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} (x _ {h}, a _ {h}) \right]\right) ^ {2} \right] + \frac {7 \beta^ {(t)}}{9} + \frac {\gamma^ {(t)}}{1 6 0 t ^ {2}} + \xi_ {t}. \tag {28}
$$

Further splitting the first term, and using that $(a + b)^2 \leq 2a^2 + 2b^2$ , we have that

$$
\begin{array}{l} \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left(\mathsf {c l i p} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} (x _ {h}, a _ {h}) \right]\right) ^ {2} \right] \\ \leq 2 \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left(\mathsf {c l i p} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\bar {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right]\right) ^ {2} \right] + 2 \left\| \mathsf {c l i p} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)}}{\bar {d} _ {h} ^ {(t + 1)}} \right] - \mathsf {c l i p} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} \right] \right\| _ {2, \bar {d} _ {h} ^ {(t + 1)}} ^ {2} \\ \leq 2 \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left(\mathsf {c l i p} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\bar {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right]\right) ^ {2} \right] + 2 \gamma^ {(t)} \left\| \mathsf {c l i p} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)}}{\bar {d} _ {h} ^ {(t + 1)}} \right] - \mathsf {c l i p} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} \right] \right\| _ {1, \bar {d} _ {h} ^ {(t + 1)}} \\ \leq 2 \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left(\mathsf {c l i p} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\bar {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right]\right) ^ {2} \right] + 2 \gamma^ {(t)} \xi_ {t}, \\ \end{array}
$$

where the second line holds since $\|w\|_{2,d}^{2}\leq\|w\|_{\infty}\|w\|_{1,d}$ and $\mathsf{clip}_{\gamma^{(t)}}[w]\leq\gamma^{(t)}$ for any w, and the last line is due to (27). Using the above bound in (28), we get that

$$
\begin{array}{l} \mathrm{(A)} \leq \frac {8 0}{\gamma^ {(t)}} \mathbb {E} _ {\bar {d} _ {h} ^ {(t + 1)}} \left[ \left(\min \left\{\frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\bar {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})}, \gamma^ {(t)} \right\}\right) ^ {2} \right] + 4 \xi_ {t} + 1 0 \beta^ {(t)} + \frac {\gamma^ {(t)}}{8 0 t ^ {2}} \\ \leq \frac {8 0}{\gamma^ {(t)}} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\overline {{d}} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right] + 4 \xi_ {t} + 1 0 \beta^ {(t)} + \frac {\gamma^ {(t)}}{8 0 t ^ {2}} \\ = \frac {8 0}{\gamma} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\widetilde {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right] + 4 \xi_ {t} + 1 0 \beta^ {(t)} + \frac {\gamma}{8 0 t}, \\ \end{array}
$$

where the second line simply follows from a change of measure, and the last line holds since $\gamma^{(t)} = \gamma t$ , and $\widetilde{d}^{(t + 1)} = t\bar{d}^{(t + 1)}$ .

Bound on clipping violation. Since $I\{u \geq v\} \leq \frac{u}{v}$ for any $u, v \geq 0$ , we get that

$$
\mathrm{(B)} \leq \frac {1}{\gamma^ {(t)}}   \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\widetilde {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right] = \frac {1}{\gamma}   \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\widetilde {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right],
$$

where the last line holds since $\gamma^{(t)} = \gamma t$ .

Combining the bounds on the terms (A) and (B) above, and summing over the rounds $t = 2, \ldots, T$ , we get

$$
\sum_ {t = 2} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} [ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) ] \leq \frac {8 1}{\gamma} \sum_ {t = 2} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\widetilde {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right] + 1 0 \sum_ {t = 2} ^ {T} \beta^ {(t)} + 4 \sum_ {t = 2} ^ {T} \xi_ {t} + \frac {\gamma}{8 0}, \tag {29}
$$

For the first term, using Lemma E.3, along with the bound $\left\|d_{h}^{(t)}/\mu_{h}^{\star}\right\|_{\infty}\leq C_{\mathrm{cov}}$ , we get that

$$
\sum_ {t = 2} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\widetilde {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})} \right] \leq 5 C _ {\mathrm{cov}} \log (1 + T).
$$

For the second term, we have

$$
\sum_ {t = 2} ^ {T} \beta^ {(t)} = \sum_ {t = 2} ^ {T} \frac {3 6 \gamma t \log (6 | \mathcal {F} | | \mathcal {W} | H T \delta^ {- 1})}{K (t - 1)} \leq \frac {7 2 \gamma T \log (6 | \mathcal {F} | | \mathcal {W} | H T \delta^ {- 1})}{K}.
$$

Combining these bounds, we get that

$$
\sum_ {t = 2} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} [ [ \Delta_ {h} f ^ {(t)} ] (x _ {h}, a _ {h}) ] = O \left(\frac {C _ {\mathrm{cov}} \log (1 + T)}{\gamma} + \frac {\gamma T \log (| \mathcal {F} | | \mathcal {W} | H T \delta^ {- 1})}{K} + \sum_ {t = 1} ^ {T} \xi_ {t} + \gamma \log (T)\right), \tag {30}
$$

Plugging this bound in to (26) for each $h \in [H]$ gives the desired result.

![](images/b4a8b3d0d21eae1a5c2fdb9749f1b1d9324c274ce5af3a56070ba52e1be5f876.jpg)

# E.3 Proof of Theorem 3.1

In this section, we prove a generalization of Theorem 3.1 that accounts for misspecification error when the class W can only approximately realize the density ratios of mixed policies. Formally, we make the following assumption on the class W:

Assumption 2.2 $^{\dagger}$ (Density ratio realizability, mixture version, with misspecification error). Let T be the parameter to GLOW (Algorithm 1). For all $h \in [H]$ , $\pi \in \Pi$ , $t \in [T]$ , and $\pi^{(1:t)} = (\pi^{(1)}, \ldots, \pi^{(t)}) \in \Pi$ , there exists a weight function $w_{h}^{\pi; \pi^{(1:t)}}(x, a) \in \mathcal{W}_{h}$ such that

$$
\sup _ {\widetilde {\pi} \in \Pi} \left\| \frac {d _ {h} ^ {\pi}}{d _ {h} ^ {\pi^ {(1 : t)}}} - w _ {h} ^ {\pi ; \pi^ {(1: t)}} \right\| _ {1, d _ {h} ^ {\widetilde {\pi}}} \leq \varepsilon_ {\mathrm{apx}}.
$$

Note that setting $\varepsilon_{apx}=0$ above recovers Assumption 2.2' given in the main body.

Theorem 3.1'. Let $\varepsilon > 0$ be given, and suppose that Assumption 2.1 holds. Further, suppose that Assumption 2.2 $^{\dagger}$ (above) holds with $\varepsilon_{\mathrm{apx}} \leq^{\varepsilon} / 18H$ . Then, GLOW, when executed on classes $\mathcal{F}$ and $\mathcal{W}$ with hyperparameters $T = \widetilde{\Theta}\left((H^{2}C_{\mathrm{cov}} / \varepsilon^{2}) \cdot \log(|\mathcal{F}||\mathcal{W}|/\delta)\right)$ , $K = 1$ , and $\gamma = \sqrt{C_{\mathrm{cov}} / (T\log(|\mathcal{F}||\mathcal{W}|/\delta))}$ returns an $\varepsilon$ -suboptimal policy $\widehat{\pi}$ with probability at least $1 - \delta$ after collecting

$$
N = \widetilde {O} \left(\frac {H ^ {2} C _ {\text { cov }}}{\varepsilon^ {2}} \log (| \mathcal {F} | | \mathcal {W} | / \delta)\right) \tag {31}
$$

trajectories. In addition, for any $T \in \mathbb{N}$ , with the same choice for $K$ and $\gamma$ as above, the algorithm enjoys a regret bound of the form

$$
\mathbf {R e g} := \sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) = \widetilde {O} \left(H \sqrt {C _ {\mathrm{cov}} T \log (| \mathcal {F} | | \mathcal {W} | / \delta)} + H T \varepsilon_ {\mathrm{apx}}\right). \tag {32}
$$

Clearly, setting the misspecification error $\varepsilon_{apx}=0$ above recovers Theorem 3.1.

Proof of Theorem 3.1'. First note that by combining Assumption 2.2 $^{\dagger}$ with the fact that $\mathsf{clip}_{\gamma}[z]$ is 1-Lipschitz for any $\gamma > 0$ , we have that for any $h \in [H]$ and $t \in [T]$ , there exists a weight function $w_{h}^{(t)} \in \mathcal{W}_{h}$ such that

$$
\sup _ {\pi \in \Pi} \left\| \operatorname{clip} _ {\gamma^ {(t)}} \left[ \frac {d _ {h} ^ {(t)}}{\overline {{d}} _ {h} ^ {(t + 1)}} \right] - \operatorname{clip} _ {\gamma^ {(t)}} \left[ w _ {h} ^ {(t)} \right] \right\| _ {1, d _ {h} ^ {\pi}} \leq \varepsilon_ {\mathrm{apx}}. \tag {33}
$$

Using this misspecification bound, and setting $K = 1$ in Lemma E.4, we get that with probability at least $1 - \delta$ ,

$$
\mathbf {R e g} = \sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) = O \bigg (\frac {H C _ {\mathrm{cov}} \log (1 + T)}{\gamma} + \gamma H T \log (6 | \mathcal {F} | | \mathcal {W} | H T \delta^ {- 1}) + H T \varepsilon_ {\mathrm{apx}} \bigg).
$$

Further setting $\gamma = \sqrt{C_{\mathrm{cov}} / (T\log(6|\mathcal{F}||\mathcal{W}|HT\delta^{-1}))}$ implies that

$$
\mathbf {R e g} \leq O \Big (H \sqrt {C _ {\mathrm{cov}} T \log (T) \log (6 | \mathcal {F} | | \mathcal {W} | H T \delta^ {- 1})} + H T \varepsilon_ {\mathrm{apx}} \Big).
$$

For the sample complexity bound, note that the returned policy $\widehat{\pi}$ is chosen via $\widehat{\pi} \sim \mathsf{Unif}(\{\pi^{(1)},\ldots,\pi^{(T)}\})$ , and thus

$$
\mathbb {E} [ J (\pi^ {\star}) - J (\widehat {\pi}) ] = \frac {1}{T} \sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) \leq O \left(H \sqrt {\frac {C _ {\mathrm{cov}}}{T} \log (T) \log (6 | \mathcal {F} | | \mathcal {W} | H T \delta^ {- 1})} + H \varepsilon_ {\mathrm{apx}}\right).
$$

Hence, when $\varepsilon_{\mathrm{apx}} \leq O(\varepsilon / H)$ , setting $T = \widetilde{\Theta}\left(\frac{H^2C_{\mathrm{cov}}}{\varepsilon^2} \log (6|\mathcal{F}||\mathcal{W}|HT\delta^{-1})\right)$ implies that the returned policy $\widehat{\pi}$ satisfies

$$
\mathbb {E} \left[ J \left(\pi^ {\star}\right) - J (\widehat {\pi}) \right] \leq \varepsilon .
$$

The total number of trajectories collected to return an $\varepsilon$ -suboptimal policy is given by

$$
T \cdot K \leq \widetilde {O} \bigg (\frac {H ^ {2} C _ {\mathrm{cov}}}{\varepsilon^ {2}} \log (6 | \mathcal {F} | | \mathcal {W} | H \delta^ {- 1}) \bigg).
$$

![](images/563397208a58b9aa84a7181ea48ad6dbd6cc31ffc1f60b6113d594745db75ef9.jpg)

# E.4 Proof of Theorem 3.2

In this section, we prove a generalization of Theorem 3.2 in Theorem 3.2' (below) which accounts for misspecification error when the class W can only approximately realize the density ratios of pure policies. Formally, we make the following assumption on the class W.

Assumption 2.2 $^{\dagger}$ (Density ratio realizability, with misspecification error). For any policy pair $\pi_{1},\pi_{2}\in\Pi$ and $h\in[H]$ , there exists some weight function $w_{h}^{(\pi_{1},\pi_{2})}\in\mathcal{W}_{h}$ such that

$$
\sup _ {\pi} \left\| \frac {d _ {h} ^ {\pi_ {1}}}{d _ {h} ^ {\pi_ {2}}} - w _ {h} ^ {(\pi_ {1}, \pi_ {2})} \right\| _ {1, d _ {h} ^ {\pi}} \leq \varepsilon_ {\mathrm{apx}}.
$$

Setting $\varepsilon_{apx}=0$ above recovers Assumption 2.2 given in the main body.

Note that Assumption 2.2 $^{\dagger}$ only states that density ratios of pure policies are approximately realized by $W_{h}$ . On the other hand, the proof of Lemma E.4, our key tool in sample complexity analysis, requires (approximate) realizability for the ratio $d^{(t)}/\bar{d}^{(t+1)}$ in $W_{h}$ , which involves a mixture of occupancies. We fix this problem by running GLOW on a larger class $\overline{W}$ that is constructed using W and has small misspecification error for $d^{(t)}/\bar{d}^{(t+1)}$ for all $t \leq T$ . Before delving into the proof of Theorem 3.2', we first describe the class $\overline{W}$ .

Construction of the class $\overline{W}$ . Define an operator Mixture that takes in a sequence of weight functions $\{w^{(1)},\ldots,w^{(t)}\}$ and a parameter $t\leq T$ , and outputs a function $[\mathsf{Mixture}(w^{(1)},\ldots,w^{(t)};t)]$ such that for any $x,a\in X\times A$ ,

$$
\left[ \operatorname{Mixture} \left(w ^ {(1)}, \dots , w ^ {(t)}; t\right) \right] _ {h} (x, a) := \frac {1}{\mathbb {E} _ {s \sim \operatorname{Unif} ([ t ])} \left[ w _ {h} ^ {(s)} (x , a) \right]}.
$$

Using the operator Mixture, we define $\overline{\mathcal{W}}^{(t)}$ via

$$
\overline {{\mathcal {W}}} ^ {(t)} = \{\text { Mixture } (w ^ {(1)}, \dots , w ^ {(t)}; t) \mid w ^ {(1)}, \dots , w ^ {(t)} \in \mathcal {W} \},
$$

and then define

$$
\overline {{\mathcal {W}}} = \cup_ {t \leq T} \overline {{\mathcal {W}}} ^ {(t)}. \tag {34}
$$

As a result of this construction, we have that

$$
\left| \overline {{{\mathcal {W}}}} \right| \leq \left(\left| \mathcal {W} \right| + 1\right) ^ {T} \leq (2 \left| \mathcal {W} \right|) ^ {T}. \tag {35}
$$

In addition, we define $\overline{\mathcal{W}}_h = \{w_h\mid w\in \overline{\mathcal{W}}\}$ . The following lemma shows that $\overline{\mathcal{W}}$ has small misspecification error for density ratios of mixture policies.

Lemma E.5. Let $t \geq 0$ be given, and suppose Assumption 2.2 holds. For any sequence of policies $\pi^{(1)}, \ldots, \pi^{(t)} \in \Pi$ , and $h \in [H]$ , there exists a weight function $\bar{w}_h \in \overline{\mathcal{W}}_h^{(t)}$ such that for any $\gamma > 0$ ,

$$
\sup _ {\pi} \left\| \mathsf {c l i p} _ {\gamma} \left[ \frac {d _ {h} ^ {(t)}}{\bar {d} _ {h} ^ {(t + 1)}} \right] - \mathsf {c l i p} _ {\gamma} [ \bar {w} _ {h} ] \right\| _ {1, d _ {h} ^ {\pi}} \leq \gamma^ {2} \varepsilon_ {\mathrm{apx}},
$$

where recall that $\bar{d}_h^{(t + 1)} = \frac{1}{t}\sum_{s = 1}^t d_h^{(s)}$

The following theorem, which is our main sample complexity bound under misspecification error, is obtained by running GLOW on the weight function class $\overline{W}$ .

Theorem 3.2'. Let $\varepsilon > 0$ be given, and suppose that Assumption 2.1 holds. Further, suppose that Assumption 2.2 holds with $\varepsilon_{\mathrm{apx}} \leq \widetilde{O}(\varepsilon^5 / C_{\mathrm{cov}}^3 H^5)$ . Then, GLOW, when executed on classes $\mathcal{F}$ and $\overline{\mathcal{W}}$ (defined in Eq. (34)) with hyperparameters $T = \widetilde{\Theta}(H^2 C_{\mathrm{cov}} / \varepsilon^2)$ , $K = \widetilde{\Theta}(T \log(|\mathcal{F}||\mathcal{W}|/\delta))$ , and $\gamma = \sqrt{C_{\mathrm{cov}} / T}$ returns an $\varepsilon$ -suboptimal policy $\widehat{\pi}$ with probability at least $1 - \delta$ after collecting

$$
N = \widetilde {O} \Big (\frac {H ^ {4} C _ {\mathrm{cov}} ^ {2}}{\varepsilon^ {4}} \log (| \mathcal {F} | | \mathcal {W} | / \delta) \Big).
$$

trajectories.

Setting the misspecification error $\varepsilon_{apx}=0$ above recovers Theorem 3.2 in the main body.

Proof of Theorem 3.2'. Using the misspecification bound from Lemma E.5 in Lemma E.4 implies that with probability at least $1 - \delta$ ,

$$
\sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) = O \Bigg (\frac {H C _ {\mathrm{cov}} \log (1 + T)}{\gamma} + \frac {\gamma H T \log (6 | \mathcal {F} | | \overline {{\mathcal {W}}} | H T \delta^ {- 1})}{K} + H \gamma^ {2} T ^ {3} \varepsilon_ {\mathrm{apx}} + \gamma H \log (T) \Bigg).
$$

Using the relation in Eq. (35), we get that $|\overline{\mathcal{W}}| \leq (2|\mathcal{W}|)^T$ and thus

$$
\sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) = O \Bigg (\frac {H C _ {\mathrm{cov}} \log (1 + T)}{\gamma} + \frac {\gamma H T \log (6 | \mathcal {F} | | \overline {{\mathcal {W}}} | H T \delta^ {- 1})}{K} + H \gamma^ {2} T ^ {3} \varepsilon_ {\mathrm{apx}} + \gamma H \log (T) \Bigg).
$$

Setting $K = 2T\log (6|\mathcal{F}||\mathcal{W}|HT\delta^{-1})$ and $\gamma = \sqrt{\frac{C_{\mathrm{cov}}}{T}}$ in the above bound, we get

$$
\sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) \leq O \Big (\Big (H \sqrt {C _ {\mathrm{cov}} T} \log (T) + H C _ {\mathrm{cov}} T ^ {2} \varepsilon_ {\mathrm{apx}} \Big) \Big).
$$

Finally, observing that the returned policy $\widehat{\pi} \sim \mathsf{Unif}\big(\{\pi^{(1)},\ldots ,\pi^{(T)}\}\big)$ , we get

$$
\sum_ {t = 1} ^ {T} J (\pi^ {\star}) - J (\pi^ {(t)}) \leq O \left(\left(H \sqrt {\frac {C _ {\mathrm{cov}}}{T}} \log (T) + H C _ {\mathrm{cov}} T ^ {2} \varepsilon_ {\mathrm{apx}}\right)\right).
$$

Thus, when $\varepsilon_{\mathrm{apx}} \leq \widetilde{O}(\varepsilon^5 / C_{\mathrm{cov}}^3 H^5)$ , setting $T = \widetilde{\Theta}\left(\frac{C_{\mathrm{cov}}}{\varepsilon^2} \log^2\left(\frac{C_{\mathrm{cov}}}{\varepsilon^2}\right)\right)$ in the above bound implies that

$$
\mathbb {E} \left[ J \left(\pi^ {\star}\right) - J (\widehat {\pi}) \right] \leq \varepsilon .
$$

The total number of trajectories collected to return $\varepsilon$ -suboptimal policy is given by:

$$
T \cdot K = O \bigg (\frac {H ^ {4} C _ {\mathrm{cov}} ^ {2}}{\varepsilon^ {4}} \log (| \mathcal {F} | | \mathcal {W} | H T \delta^ {- 1}) \bigg).
$$

![](images/4e95f094ee2a7fe1e68b93f64bce67fc851c5439a67a11819d6a8d9ad7a57b52.jpg)

Proof of Lemma E.5. Fix any $h \in [H]$ and $t \in [T]$ . Using Assumption 2.2 $^{\ddagger}$ , we have that for any $s \leq t$ , there exists a function $w_h^{(s,t)} \in \mathcal{W}_h$ such that

$$
\sup _ {\pi \in \Pi} \left\| \frac {d _ {h} ^ {(s)}}{d _ {h} ^ {(t)}} - w _ {h} ^ {(s, t)} \right\| _ {1, d _ {h} ^ {\pi}} \leq \varepsilon_ {\mathrm{apx}}. \tag {36}
$$

Let $\bar{w}_h = \left[\mathsf{Mixture}(w^{(1,t)},\dots ,w^{(t,t)};t)\right]_h\in \overline{\mathcal{W}}_h$ , and recall that $\bar{d}^{(t + 1)} = \mathbb{E}_{s\sim \mathrm{Unif}([t])}\left[d_h^{(s)}\right]$ . For any $x,a\in \mathcal{X}\times \mathcal{A}$ define

$$
\zeta (x, a) := \left| \min \left\{\frac {1}{\mathbb {E} _ {s \sim \mathsf {U n i f} ([ t ])} \left[ d _ {h} ^ {(s)} (x , a) / d _ {h} ^ {(t)} (x , a) \right]}, \gamma \right\} - \min \left\{\frac {1}{\mathbb {E} _ {s \sim \mathsf {U n i f} ([ t ])} \left[ w ^ {(s , t)} (x , a) \right]}, \gamma \right\} \right|.
$$

Using that the function $g(z) = \min \left\{\frac{1}{z},\gamma \right\} = \frac{1}{\max\{z,\gamma^{-1}\}}$ is $\gamma^2$ -Lipschitz, we get that

$$
\begin{array}{l} \zeta (x, a) \leq \gamma^ {2} \cdot \left| \mathbb {E} _ {s \sim \operatorname{Unif} ([ t ])} \left[ \frac {d _ {h} ^ {(s)} (x , a)}{d _ {h} ^ {(t)} (x , a)} \right] - \mathbb {E} _ {s \sim \operatorname{Unif} ([ t ])} \left[ w _ {h} ^ {(s, t)} (x, a) \right] \right| \\ \leq \gamma^ {2} \mathbb {E} _ {s \sim \mathrm{Unif} ([ t ])} \left[ \left| \frac {d _ {h} ^ {(s)} (x , a)}{d _ {h} ^ {(t)} (x , a)} - w _ {h} ^ {(s, t)} (x, a) \right| \right], \\ \end{array}
$$

where the last inequality follows from the linearity of expectation, and by using Jensen's inequality.

Thus,

$$
\begin{array}{l} \sup _ {\pi \in \Pi} \mathbb {E} _ {x _ {h}, a _ {h} \sim d _ {h} ^ {\pi}} \left[ \left| \min \left\{\frac {d _ {h} ^ {(t)} (x _ {h} , a _ {h})}{\bar {d} _ {h} ^ {(t + 1)} (x _ {h} , a _ {h})}, \gamma \right\} - \min \{\bar {w} _ {h} (x _ {h}, a _ {h}), \gamma \} \right| \right] \\ = \sup _ {\pi \in \Pi} \mathbb {E} _ {x _ {h}, a _ {h} \sim d _ {h} ^ {\pi}} [ \zeta (x _ {h}, a _ {h}) ] \\ \leq \gamma^ {2} \sup _ {\pi \in \Pi} \mathbb {E} _ {s \sim \mathrm{Unif} ([ t ])} \left[ \left\| \frac {d _ {h} ^ {(s)}}{d _ {h} ^ {(t)}} - w _ {h} ^ {(s, t)} \right\| _ {1, d _ {h} ^ {\pi}} \right] \\ \leq \gamma^ {2} \mathbb {E} _ {s \sim \mathrm{Unif} ([ t ])} \sup _ {\pi \in \Pi} \left[ \left\| \frac {d _ {h} ^ {(s)}}{d _ {h} ^ {(t)}} - w _ {h} ^ {(s, t)} \right\| _ {1, d _ {h} ^ {\pi}} \right] \\ \leq \gamma^ {2} \varepsilon_ {\mathrm{apx}}, \\ \end{array}
$$

where the second-to-last line is due to Jensen's inequality, and the last line follows from the bound in (36). $\square$

# F Proofs and Additional Results from Section 4 (Hybrid RL)

This section is organized as follows:

- Appendix F.1 gives additional details and further examples of offline algorithms with which we can apply $\mathrm{H}_2\mathrm{O}$ , including an example that uses FQI as the base algorithm.   
- Appendix F.2 contains the proof of the main result for $\mathrm{H}_2\mathrm{O}$ , Theorem 4.1.   
- Appendix F.3 contains supporting proofs for the offline RL algorithms (Appendix F.1) we use within $\mathrm{H}_2\mathrm{O}$ .

# F.1 Examples for $\mathrm{H}_2\mathrm{O}$

This section contains additional examples of base algorithms that can be applied within the $H_{2}O$ reduction.

# F.1.1 HYGLOW Algorithm

For completeness, we state full pseudocode for the HYGLOW algorithm described in Section 4.2 as Algorithm 3. As described in the main body, this algorithm simply invokes $H_{2}O$ with a clipped and regularized variant of the MABO algorithm (MABO.CR, Eq. (13)) as the base algorithm.

We will invoke MABO.CR with a certain augmented weight function class $\overline{\mathcal{W}}$ , which we now define. For any $w \in \mathcal{W}$ , let $w^{(h)} \in (\mathcal{X} \times \mathcal{A} \times [H] \to \mathbb{R} \cup \{+\infty\})$ be defined by

$$
w _ {h ^ {\prime}} ^ {(h)} (x, a) := \left\{ \begin{array}{l l} w _ {h} (x, a) & \text { if } h = h ^ {\prime} \\ 0 & \text { if } h \neq h ^ {\prime} \end{array} , \right. \tag {37}
$$

and let $\mathcal{W}^{(h)} = \{w^{(h)}\mid w\in \mathcal{W}\}$ . Define the set

$$
\overline {{{\mathcal {W}}}} := \mathcal {W} \cup (- \mathcal {W}) \cup_ {h \in [ H ]} \left(\mathcal {W} ^ {(h)} \cup (- \mathcal {W} ^ {(h)})\right). \tag {38}
$$

Note that the size satisfies $\overline{W}$ is $|\overline{W}| \leq 2(H + 1)|\mathcal{W}| \leq 4H|\mathcal{W}|$ .

We recall that the MABO.CR algorithm with dataset D and parameters $\gamma, F, W$ is defined via

$$
\widehat {f} \in \underset {f \in \mathcal {F}} {\arg \min} \max _ {w \in \mathcal {W}} \sum_ {h = 1} ^ {H} \left| \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \check {w} _ {h} (x _ {h}, a _ {h}) [ \widehat {\Delta} _ {h} f ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \right] \right| - \alpha^ {(n)} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \check {w} _ {h} ^ {2} (x _ {h}, a _ {h}) \right], \tag {39}
$$

where $\alpha^{(n)} := 8 / \gamma^{(n)}$ and $\check{w}_h := \mathsf{clip}_{\gamma^{(n)}}[w_h]$ .

In Appendix F.3.2 we will prove the following result.

Theorem F.1 (MABO.CR is CC-bounded). Let $D = \{D_h\}_{h=1}^H$ consist of $H \cdot n$ samples from $\underline{\mu}^{(1)}, \ldots, \mu^{(n)}$ . For any $\gamma \in R_+$ , the MABO.CR algorithm (Eq. (13)) with parameters F, augmented class W defined in Eq. (38) in Appendix F.1.1, and $\gamma$ is CC-bounded at scale $\gamma$ under the Assumption that $Q^\star \in F$ and that for all $\pi \in \Pi$ and $h \in [H]$ , $d_h^\pi / \mu_h^{(1:n)} \in W$ .

The risk bound for HYGLOW will follow by Theorem 4.1. Namely, we have the following.

Corollary 4.1 (HYGLOW Risk bound). Let $\varepsilon > 0$ be given, $\mathcal{D}_{\mathrm{off}}$ consist of $H \cdot T$ samples from data distribution $\nu$ , where $\nu$ satisfies $C_{\star}$ -single-policy concentrability. Suppose that $Q^{\star} \in \mathcal{F}$ and that for all $t \in [T]$ , $\pi \in \Pi$ , and $h \in [H]$ , we have $d_h^\pi / \mu_h^{(t)} \in \mathcal{W}$ , where $\mu_h^{(t)} := 1/2 (\nu_h + 1/t \sum_{i=1}^t d_h^{\pi^{(i)}})$ . Then, HYGLOW with inputs $T = \widetilde{\Theta}((H^4(C_{\mathrm{cov}} + C_{\star})/\varepsilon^2) \cdot \log(|\mathcal{F}||\mathcal{W}|/\delta))$ , $\mathcal{F}$ , augmented $\overline{\mathcal{W}}$ defined in Eq. (38), $\gamma = \widetilde{\Theta}\left(\sqrt{(C_{\star} + C_{\mathrm{cov}})/TH^2 \log(|\mathcal{F}||\mathcal{W}|/\delta)}\right)$ , and $\mathcal{D}_{\mathrm{off}}$ returns an $\varepsilon$ -suboptimal policy with probability at least $1 - \delta T$ after collecting

$$
N = \widetilde {O} \left(\frac {H ^ {2} (C _ {\text { cov }} + C _ {\star})}{\varepsilon^ {2}} \log (| \mathcal {F} | | \mathcal {W} | / \delta)\right)
$$

trajectories.

# F.1.2 Computationally efficient implementation for HYGLOW

In the following, we expand on the discussion after Theorem F.1 regarding computationally efficient implementation of the optimization problem in Eq. (13) and Eq. (14) via reparameterization. Let $\gamma > 0$ and n > 0 be given to the learner, and suppose that in addition to the class W, the learner has access to a function class $\mathcal{W}^{(\gamma,n)}$ that satisfies the following assumption.

Assumption F.1. The function class $\mathcal{W}^{(\gamma,n)}$ satisfies

(a) For all $w \in \mathcal{W}^{(\gamma, n)}$ and $h \in [H]$ , $\| w_h \|_{\infty} \leq \gamma n$ .   
(b) For all $h \in [H]$ , $\{\pm \mathsf{clip}_{\gamma n}[w] \mid w \in \mathcal{W}\} \subseteq \mathcal{W}^{(\gamma, n)}$ .   
(c) For all $h \in [H]$ , $\{\pm \mathsf{clip}_{\gamma n}[w^{(h)}] \mid w \in \mathcal{W}\} \subseteq \mathcal{W}^{(\gamma, n)}$ , where $w^{(h)}$ is defined as in Eq. (37).

Note that $\mathcal{W}^{(\gamma,n)}$ also satisfies the density ratio realizability required by MABO.CR (cf. Theorem F.1). We claim that optimizing directly over the class $\mathcal{W}^{(\gamma,n)}$ , which does not involve explicitly clipping, leads to the same guarantee as solving Eq. (13). In more detail, consider the following offline RL algorithm, which given offline datasets $\{D_{h}\}_{h\leq H}$ , returns

$$
\widehat {f} \in \underset {f \in \mathcal {F}} {\arg \min} \max _ {w \in \mathcal {W} ^ {(\gamma , n)}} \sum_ {h = 1} ^ {H} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ w _ {h} (x _ {h}, a _ {h}) [ \widehat {\Delta} _ {h} f ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \right] - \alpha^ {(n)} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ w _ {h} ^ {2} (x _ {h}, a _ {h}) \right]. \tag {40}
$$

Using the next lemma, we show that the function obtained by solving Eq. (40) leads to the same offline RL guarantee as MABO.CR when $|\log(\mathcal{W}^{(\gamma,n)})| = O(\log(|\mathcal{W}|) + \text{poly}(H))$ . In particular, substituting the bound from Lemma F.1 in place of the corresponding bound from Lemma F.6 in the proof of Theorem F.1, while keeping rest of the analysis same, shows that the above described computationally efficient implementation of MABO.CR is also CC-bounded. Using this fact with Theorem 4.1 implies the desired performance guarantee for HYGLOW (similar to Corollary 4.1).

Lemma F.1. Suppose $\mathcal{F}$ and $\mathcal{W}$ satisfy Assumption 2.1 and Assumption F.4. Additionally, let $\gamma \in (0,1)$ and $n > 0$ be given constants, and for $h \in [H]$ , let $\mathcal{D}_h$ be datasets of size $n$ sampled from the offline distribution $\mu_h^{(1:n)}$ . Furthermore, let $\mathcal{W}^{(\gamma,n)}$ be a reparameterized function class that satisfies Assumption F.1 w.r.t. $\mathcal{W}$ . Then, the function $\widehat{f}$ returned by Eq. (40), when executed with datasets $\{\mathcal{D}_h\}_{h \leq H}$ , weight function class $\mathcal{W}^{(\gamma,n)}$ , and parameters $\alpha^{(n)} = \frac{8}{\gamma^n}$ , satisfies with probability at least $1 - \delta$ ,

$$
\sum_ {h = 1} ^ {H} \left| \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ [ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \right| \leq \mathcal {O} \left(\sum_ {h = 1} ^ {H} \frac {1}{\gamma n} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + H ^ {2} \beta^ {(n)}\right),
$$

for all $w \in \mathcal{W}$ , where $\beta^{(n)} = O\left(\frac{\gamma n}{n-1} \log(24|\mathcal{F}||\mathcal{W}|H^2/\delta)\right)$ .

Proof of Lemma F.1. Repeating the same arguments as in the proof of Lemma E.2 with K = 1 (we avoid repeating the arguments for conciseness), along with the fact that $\|w_{h}^{\prime}\|_{\infty} \leq \gamma n$ for all $h \in [H]$ and $w' \in \mathcal{W}^{(\gamma, n)}$ , we get that the returned function $\widehat{f}$ satisfies for all $w' \in \mathcal{W}^{(\gamma, n)}$ ,

$$
\sum_ {h = 1} ^ {H} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left[ \Delta_ {h} \widehat {f} \right] (x _ {h}, a _ {h}) \cdot w _ {h} ^ {\prime} (x _ {h}, a _ {h}) \right] \leq \mathcal {O} \left(\sum_ {h = 1} ^ {H} \frac {1}{\gamma n} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left(w _ {h} ^ {\prime} (x _ {h}, a _ {h})\right) ^ {2} \right] + \frac {H \gamma n}{n - 1} \log (| \mathcal {F} | | \mathcal {W} ^ {(\gamma , n)} | H / \delta)\right).
$$

However, as in the proof of Lemma F.6, since for any $w \in \mathcal{W}$ we have that both $\mathsf{clip}_{\gamma n}[w^{(h)}] \in \mathcal{W}_h^{(\gamma, n)}$ and $-\mathsf{clip}_{\gamma n}[w^{(h)}] \in \mathcal{W}_h^{(\gamma, n)}$ for every $h \in [H]$ , the above inequality immediately implies that for every $w \in \mathcal{W}$ ,

$$
\sum_ {h = 1} ^ {H} \left| \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ [ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \right| \leq \mathcal {O} \left(\sum_ {h = 1} ^ {H} \frac {1}{\gamma n}   \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \frac {H ^ {2} \gamma n}{n - 1} \log (| \mathcal {F} | | \mathcal {W} ^ {(\gamma , n)} | H / \delta)\right),
$$

where we used that the RHS is independent of the sign. The final statement follows by plugging in that $|\log(\mathcal{W}^{(\gamma,n)})| = O(\log(|\mathcal{W}|) + \text{poly}(H))$ .

# F.1.3 Fitted Q-Iteration

In this section, we apply $\mathrm{H}_2\mathrm{O}$ with the Fitted Q-Iteration (FQI) algorithm (Munos, 2007; Munos and Szepesvári, 2008; Chen and Jiang, 2019) as the base algorithm. For an offline dataset $\mathcal{D}$ and value function class $\mathcal{F}$ , the FQI algorithm is defined as follows:

Algorithm F.2 (Fitted Q-Iteration (FQI)).

1. Set $\widehat{f}_{H+1}(x,a)=0$ for all $(x,a)$   
2. For $h = H, \ldots, 1$ :

$$
\widehat {f} _ {h} \in \underset {f _ {h} \in \mathcal {F} _ {h}} {\arg \min} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \Big [ (f _ {h} (x _ {h}, a _ {h}) - r _ {h} - \underset {a ^ {\prime}} {\max} \widehat {f} _ {h + 1} (x _ {h + 1}, a ^ {\prime})) ^ {2} \Big ].
$$

3. Output $\widehat{\pi} = \pi_{\widehat{f}}$ .

We analyze FQI under the following standard Bellman completeness assumption.

Assumption F.2 (Bellman completeness). For all $h \in [H]$ , we have that

$$
\mathcal {T} _ {h} \mathcal {F} _ {h + 1} \subseteq \mathcal {F} _ {h}
$$

Theorem F.3. The FQI algorithm is CC-bounded under Assumption F.2 with scaling functions $a_{\gamma} = \frac{6}{\gamma}$ and $b_{\gamma} = 1024 \log(2n|\mathcal{F}|)\gamma$ , for all $\gamma > 0$ simultaneously. As a consequence, when invoked within $H_{2}O$ , we have Risk $\leq \widetilde{O}\left(H \sqrt{(C_{\star} + C_{\mathrm{cov}}) \log(|\mathcal{F}|\delta^{-1})/T}\right)$ with probability at least $1 - \delta T$ .

The proof can be found in Appendix F.3.3.

The full pseudocode for $H_{2}O$ with FQI is essentially identical to the Hy-Q algorithm of Song et al. (2023), except for a slightly different data aggregation strategy in Line 8. Thus, Hy-Q can be interpreted as a special case of the $H_{2}O$ algorithm when instantiated with FQI as a base algorithm. The risk bounds in Song et al. (2023) are proven under an a structural condition known as bilinear rank (Du et al., 2021), which is complementary to coverability. Our result recovers a special case of a risk bound for Hy-Q given in the follow-up work of Liu et al. (2023), which analyzes Hy-Q under coverability instead of bilinear rank.

# F.1.4 Model-Based MLE

In this section we apply $H_{2}O$ with Model-Based Maximum Likelihood Estimation (MLE) algorithm as the base algorithm. The algorithm is parameterized by a model class $M = \{M_{h}\}_{h=1}^{H}$ , where $M_{h} \subset \{M_{h} : X \times A \to \Delta(\mathbb{R} \times X)\}$ . Each model $M = \{M_{h}\}_{h=1}^{H} \in M$ has the same state space, action space, initial distribution, and horizon, and each $M_{h} \in M_{h}$ is a conditional distribution over rewards and next states for layer h. For a dataset D, the algorithm proceeds as follows.

Algorithm F.4 (Model-Based MLE).

\- For $h \in [H]$ :

\- Compute the maximum likelihood estimator for layer $h$ as

$$
\widehat {M} _ {h} = \underset {M _ {h} \in \mathcal {M} _ {h}} {\arg \max} \sum_ {(x _ {h}, a _ {h}, r _ {h}, x _ {h + 1}) \in \mathcal {D} _ {h}} \log (M _ {h} (r _ {h}, x _ {h + 1} \mid x _ {h}, a _ {h})) \tag {41}
$$

\- Output $\pi_{\widehat{M}}^{\star}$ , the optimal policy for $\widehat{M} = \{\widehat{M}_h\}_{h}$

We analyze Model-Based MLE under a standard realizability assumption.

Assumption F.3 (Model realizability). We have that $M^{\star} \in M$ .

Theorem F.5. The model-based MLE algorithm is CC-bounded under Assumption F.3 for all $\gamma > 0$ simultaneously, with scaling functions $\mathfrak{a}_{\gamma} = \frac{6}{\gamma}$ and $\mathfrak{b}_{\gamma} = 8\log (|\mathcal{M}|H / \delta)\gamma$ . As a consequence, when invoked within $H_{2}O$ , we have $\mathbf{Risk} \leq \widetilde{O}\left(H\sqrt{(C_{\star} + C_{\mathrm{cov}})\log(|\mathcal{M}|H\delta^{-1}) / T}\right)$ with probability at least $1 - \delta T$ .

The proof can be found in Appendix F.3.4.

# F.2 Proofs for $\mathrm{H}_2\mathrm{O}$ (Theorem 4.1)

The following theorem is a slight generalization of Theorem 4.1. In the sequel, we prove Theorem 4.1 as a consequence of this result.

Theorem F.6. Let $T \in \mathbb{N}$ be given, let $\mathcal{D}_{\mathrm{off}}$ consist of $H \cdot T$ samples from data distribution $\nu$ . Let $\mathbf{Alg}_{\mathrm{off}}$ be CC-bounded at scale $\gamma \in [0,1]$ under Assumption(·), with parameters $\mathfrak{a}_{\gamma}$ and $\mathfrak{b}_{\gamma}$ . Suppose that for all $t \in [T]$ and $\pi^{(1)}, \ldots, \pi^{(t)} \in \Pi$ , Assumption $(\mu^{(t)}, M^{\star})$ holds for $\mu^{(t)} := \{1/2(\nu_h + 1/t \sum_{i=1}^t d_h^{\pi^{(i)}})\}_{h=1}^H$ . Then, with probability at least $1 - \delta T$ , the risk of $H_2O$ (Algorithm 2) with inputs $T$ , $Alg_{off}$ , and $\mathcal{D}_{\mathrm{off}}$ is bounded as

$$
\mathbf {R i s k} \leq \frac {2 \mathfrak {a} _ {\gamma}}{T} \sum_ {h = 1} ^ {H} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathsf {C C} _ {h} (\pi^ {\star}, \nu , \gamma^ {(t)}) + \underbrace {\widetilde {O} \left(H \left(\frac {C _ {\text {cov}} \mathfrak {a} _ {\gamma}}{N} + \mathfrak {b} _ {\gamma}\right)\right)} _ {=: \text {err} _ {\text {off}}}. \tag {42}
$$

Proof of Theorem F.6. Recall the definitions $d_{h}^{(t)} := d_{h}^{\pi^{(t)}}$ , $\widetilde{d}_{h}^{(t)} := \sum_{s=1}^{t-1} d_{h}^{(t)}$ , and $\bar{d}_{h}^{(t)} := \frac{1}{t-1} \widetilde{d}_{h}^{(t)}$ . Furthermore, let $d_{h}^{\star} := d_{h}^{\pi^{\star}}$ for all $h \leq H$ . Note that the data distribution for $\mathcal{D}_{\text{hybrid}}^{(t)}$ is $\mu^{(t)} = \{\mu_{h}^{(t)}\}_{h=1}^{H}$ where $\mu_{h}^{(t)} = \frac{1}{2} (\nu_{h} + \bar{d}_{h}^{(t)})$ , where $\nu_{h}$ is the offline distribution. As a result of Definition 4.3, the offline algorithm Alg $_{\text{off}}$ invoked on the dataset $\mathcal{D}_{\text{hybrid}}^{(t)}$ outputs a distribution $p_{t} \sim \Delta(\Pi)$ that satisfies the bound:

$$
\mathbb {E} _ {\pi^ {(t)} \sim p _ {t}} \left[ J (\pi^ {\star}) - J (\pi^ {(t)}) \right] \leq \sum_ {h = 1} ^ {H} \frac {\mathfrak {a} _ {\gamma}}{t} \left(\mathsf {C C} _ {h} (\pi^ {\star}, \mu^ {(t)}, \gamma t) + \mathbb {E} _ {\pi^ {(t)} \sim p _ {t}} [ \mathsf {C C} _ {h} (\pi^ {(t)}, \mu^ {(t)}, \gamma t) ]\right) + \mathfrak {b} _ {\gamma}, \tag {43}
$$

with probability at least $1 - \delta$ , where $\mathfrak{a}_{\gamma}$ and $\mathfrak{b}_{\gamma}$ are the scaling functions for which $\mathbf{Alg}_{\mathrm{off}}$ is CC-bounded at scale $\gamma$ . By taking a union bound over $T$ , the number of iterations, we have that the event in Eq. (43) occurs for all $t \leq T$ with probability greater than $1 - \delta T$ .

Plugging in the definition for the clipped concentrability coefficient above and summing over $t$ from $1, \ldots, T$ we get that

$$
\begin{array}{l} \mathbf {R e g} = \mathbb {E} \left[ \sum_ {t = 1} ^ {T} J \left(\pi^ {\star}\right) - J \left(\pi^ {(t)}\right) \right] \\ \leq \sum_ {h = 1} ^ {H} \left(\underbrace {\sum_ {t = 1} ^ {T} \frac {\mathfrak {a} _ {\gamma}}{t} \left\| \operatorname{clip} _ {\gamma t} \left[ \frac {d _ {h} ^ {\pi^ {\star}}}{\mu_ {h} ^ {(t)}} \right] \right\| _ {1 , d _ {h} ^ {\star}}} _ {\text {(I)}} + \underbrace {\sum_ {t = 1} ^ {T} \frac {\mathfrak {a} _ {\gamma}}{t} \mathbb {E} _ {\pi^ {(t)} \sim p _ {t}} \left[ \left\| \operatorname{clip} _ {\gamma t} \left[ \frac {d _ {h} ^ {(t)}}{\mu_ {h} ^ {(t)}} \right] \right\| _ {1 , d _ {h} ^ {(t)}} \right]} _ {\text {(II)}}\right) + H T \mathfrak {b} _ {\gamma}. \\ \end{array}
$$

For each $h \in [H]$ , we bound the two terms I and II separately below.

Term (I). Note that $\mu_{h}^{(t)}(x,a)\geq\nu_{h}(x,a)/2$ for any x,a. Thus,

$$
(I) \leq 2 \mathfrak {a} _ {\gamma} \sum_ {t = 1} ^ {T} \frac {1}{t} \left\| \operatorname{clip} _ {\gamma t} \left[ \frac {d _ {h} ^ {\pi^ {*}}}{\nu_ {h}} \right] \right\| _ {1, d _ {h} ^ {\pi^ {*}}} = 2 \mathfrak {a} _ {\gamma} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {C C} _ {h} (\pi^ {\star}, \nu , \gamma t).
$$

Term (II). We bound this term uniformly for any $\pi^{(t)} \sim p_{t}$ . So, fix $\pi^{(t)}$ and note that

$$
\begin{array}{l} \mathrm{(II)} = \mathfrak {a} _ {\gamma} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \min \Bigl \{\frac {d _ {h} ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)}, \gamma t \Bigr \} \right] \\ = \mathfrak {a} _ {\gamma} \Bigg (\underbrace {\sum_ {t = 1} ^ {T} \frac {1}{t}   \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} \mathbb {I} \left\{\frac {d ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} \leq \gamma t \right\} \right]} _ {\text {(II.A)}} + \underbrace {\sum_ {t = 1} ^ {T} \frac {1}{t}   \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \gamma t \cdot \mathbb {I} \left\{\frac {d ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} > \gamma t \right\} \right]} _ {\text {(II.B)}} \Bigg), \\ \end{array}
$$

where the second line holds since $\min\{u,v\}=u\mathbb{I}\{u\leq v\}+v\mathbb{I}\{v<u\}$ for all $u,v\in R$ .

In order to bound the two terms appearing above, we use certain properties of coverability, similar to the analysis of GLOW (Appendix E). For a parameter $\lambda \in (0,1)$ , let us define a burn-in time

$$
\tau_ {h} ^ {(\lambda)} (x, a) = \min \left\{t \mid \widetilde {d} _ {h} ^ {(t)} (x, a) \geq \frac {C _ {\mathrm{cov}} \cdot \mu_ {h} ^ {\star} (x , a)}{\lambda} \right\}, \tag {44}
$$

and observe that

$$
\sum_ {t = 1} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} [ \mathbb {I} \{t <   \tau_ {h} ^ {(\lambda)} (x, a) \} ] = \sum_ {x, a} \sum_ {t <   \tau_ {h} ^ {(\lambda)} (x, a)} d _ {h} ^ {(t)} (x, a) \leq \frac {2 C _ {\mathrm{cov}}}{\lambda}, \tag {45}
$$

which holds for any $\lambda \in (0,1)$ . This bound can be derived by noting that

$$
\begin{array}{l} \sum_ {x, a} \sum_ {t <   \tau_ {h} ^ {(\lambda)} (x, a)} d _ {h} ^ {(t)} (x, a) = \sum_ {x, a} \tilde {d} ^ {(\tau_ {h} ^ {(\lambda)} (x, a))} (x, a) \\ = \sum_ {x, a} \tilde {d} ^ {\left(\tau_ {h} ^ {(\lambda)} (x, a) - 1\right)} (x, a) + \sum_ {x, a} d ^ {\left(\tau_ {h} ^ {(\lambda)} (x, a)\right)} (x, a) \\ \leq \sum_ {x, a} \frac {C _ {\mathrm{cov}}}{\lambda} \mu_ {h} ^ {\star} (x, a) + \sum_ {x, a} C _ {\mathrm{cov}} \mu_ {h} ^ {\star} (x, a) \\ \leq C _ {\mathrm{cov}} \left(\frac {1}{\lambda} + 1\right) \leq \frac {2 C _ {\mathrm{cov}}}{\lambda}. \\ \end{array}
$$

We also recall the follow bound, which is a corollary of the elliptical potential lemma (Lemma D.5):

$$
\sum_ {t = 1} ^ {T} \sum_ {x, a} d _ {h} ^ {(t)} (x, a) \frac {d _ {h} ^ {(t)} (x , a)}{\widetilde {d} ^ {(t)} (x , a)} \mathbb {I} \left\{t > \tau_ {h} ^ {(\lambda)} (x, a) \right\} \leq \sum_ {t = 1} ^ {T} \sum_ {x, a} d _ {h} ^ {(t)} (x, a) \frac {d _ {h} ^ {(t)} (x , a)}{\widetilde {d} ^ {(t)} (x , a)} \mathbb {I} \left\{t > \tau_ {h} ^ {(1)} (x, a) \right\} \stackrel {\text {(i)}} {\leq} 5 \log (T) C _ {\text {cov}}. \tag {46}
$$

The inequality (i) can be seen derived by noting that, under the event in the indicator, we have $\widetilde{d}_h^{(t)}(x,a)\geq C_{\mathrm{cov}}\mu_h^\star (x,a)$ and thus $\widetilde{d}_h^{(t)}(x,a)\geq \frac{1}{2} (C_{\mathrm{cov}}\mu_h^\star (x,a) + \widetilde{d}^{(t)}(x,a))$ . This gives

$$
\sum_ {t = 1} ^ {T} \sum_ {x, a} d _ {h} ^ {(t)} (x, a) \frac {d _ {h} ^ {(t)} (x , a)}{\widetilde {d} ^ {(t)} (x , a)} \mathbb {I} \bigl \{t > \tau_ {h} ^ {(1)} (x, a) \bigr \} \leq 2 \sum_ {t = 1} ^ {T} \sum_ {x, a} d _ {h} ^ {(t)} (x, a) \frac {d _ {h} ^ {(t)} (x , a)}{\widetilde {d} ^ {(t)} (x , a) + C _ {\mathrm{cov}} \mu_ {h} ^ {\star} (x , a)},
$$

from which we can repeat the steps from Eq. (23) to Eq. (24).

Term (II.A). To bound this term, we introduce a split according to the burn-in time $\tau_{h}^{(1)}(x,a)$ , i.e.

$$
\text {(II.A)} = \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} \mathbb {I} \left\{\frac {d _ {h} ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} \leq \gamma t \right\} \left(\mathbb {I} \{t \leq \tau_ {h} ^ {(1)} (x, a) \} + \mathbb {I} \{t > \tau_ {h} ^ {(1)} (x, a) \}\right) \right].
$$

The first term is bounded via

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} \mathbb {I} \bigg \{\frac {d _ {h} ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} \leq \gamma t \bigg \} \mathbb {I} \big \{t \leq \tau_ {h} ^ {(1)} (x, a) \big \} \right] \leq \gamma \sum_ {t = 1} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \mathbb {I} \big \{t \leq \tau_ {h} ^ {(1)} (x, a) \big \} \right] \\ \leq 2 \gamma C _ {\mathrm{cov}}, \\ \end{array}
$$

by Eq. (45) with $\lambda = 1$ . The second term is bounded via:

$$
\sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} \mathbb {I} \Bigl \{\frac {d ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} \leq \gamma t \Bigr \} \mathbb {I} \bigl \{t > \tau_ {h} ^ {(1)} (x, a) \bigr \} \right] \leq 2 \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x , a)}{\overline {{d}} _ {h} ^ {(t)} (x , a)} \mathbb {I} \bigl \{t > \tau_ {h} ^ {(1)} (x, a) \bigr \} \right]
$$

$$
\leq 2 \sum_ {t = 1} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \frac {d _ {h} ^ {(t)} (x , a)}{\widetilde {d} _ {h} ^ {(t)} (x , a)} \mathbb {I} \{t > \tau_ {h} ^ {(1)} (x, a) \} \right]
$$

$$
\leq 1 0 \log (T) C _ {\mathrm{cov}},
$$

by using that $\mu_h^{(t)}(x,a)\geq \frac{\bar{d}_h^{(t)}(x,a)}{2}$ and Equation (46).

Adding these two terms together gives us the upper bound (II.A) $\leq 2\gamma C_{\mathrm{cov}} + 10\log(T)C_{\mathrm{cov}}$ .

Term (II.B). We have

$$
\sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \gamma t \mathbb {I} \left\{\frac {d _ {h} ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} > \gamma t \right\} \right] = \gamma \sum_ {t = 1} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \mathbb {I} \left\{\frac {d _ {h} ^ {(t)} (x , a)}{\mu_ {h} ^ {(t)} (x , a)} > \gamma t \right\} \right]
$$

$$
\stackrel {\mathrm{(i)}} {\leq} \gamma \sum_ {t = 1} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \mathbb {I} \bigg \{\frac {C _ {\mathrm{cov}} \mu_ {h} ^ {\star} (x , a)}{\widetilde {d} _ {h} ^ {(t)} (x , a)} > \frac {\gamma}{2} \bigg \} \right]
$$

$$
\stackrel {\text {(ii)}} {=} \gamma \sum_ {t = 1} ^ {T} \mathbb {E} _ {d _ {h} ^ {(t)}} \left[ \mathbb {I} \bigl \{t \leq \tau_ {h} ^ {(\gamma / 2)} (x, a) \bigr \} \right]
$$

$$
\leq \gamma \cdot \frac {4 C _ {\mathrm{cov}}}{\gamma} = 4 C _ {\mathrm{cov}},
$$

where the inequality (i) follows from applying the upper bounds $d_{h}^{(t)}(x,a)\leq C_{\mathrm{cov}}\mu_{h}^{\star}(x,a)$ and $\mu_{h}^{(t)}(x,a)\geq\frac{1}{2}\bar{d}_{h}^{(t)}(x,a)$ , and the inequality (ii) follows from the definition of the burn-in time (Eq. (44)) with $\lambda=\gamma/2$ .

Combining all the bounds above, we get that

$$
(\mathbf {I I}) \leq 2 \mathfrak {a} _ {\gamma} (2 \gamma C _ {\text { cov }} + 1 0 \log (T) C _ {\text { cov }} + 2 C _ {\text { cov }})
$$

$$
= 4 \mathfrak {a} _ {\gamma} C _ {\mathrm{cov}} (\gamma + 1 0 \log (T) + 1).
$$

Adding together the terms so far, we can conclude the regret bound:

$$
\mathbf {R e g} \leq 2 \mathfrak {a} _ {\gamma} \sum_ {h = 1} ^ {H} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {C C} _ {h} (\pi^ {\star}, \nu , \gamma t) + H \left(4 \mathfrak {a} _ {\gamma} C _ {\text { cov }} (\gamma + 1 0 \log (T) + 1) + T \mathfrak {b} _ {\gamma}\right).
$$

It follows that the policy $\widehat{\pi} = \mathsf{Unif}(\pi^{(1)},\ldots ,\pi^{(T)})$ satisfies the risk bound

$$
\mathbf {R i s k} \leq \frac {2 \mathfrak {a} _ {\gamma}}{T} \sum_ {h = 1} ^ {H} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathsf {C C} _ {h} (\pi^ {\star}, \nu , \gamma t) + \underbrace {H \left(4 \frac {\mathfrak {a} _ {\gamma} C _ {\mathrm{cov}}}{T} (\gamma + 1 0 \log (T) + 1) + \mathfrak {b} _ {\gamma}\right)} _ {:= \operatorname{err} _ {\text {off}}}
$$

$$
= \frac {2 \mathfrak {a} _ {\gamma}}{T} \sum_ {h = 1} ^ {H} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathsf {C C} _ {h} (\pi^ {\star}, \nu , \gamma t) + \widetilde {O} \bigg (H \bigg (\frac {C _ {\mathrm{cov}} \mathfrak {a} _ {\gamma}}{T} + \mathfrak {b} _ {\gamma} \bigg) \bigg),
$$

with probability at least $1 - \delta T$ , where in the last line we have used that $\gamma \in [0, 1]$ .

We now prove Theorem 4.1 as a consequence of Theorem F.6.

Theorem 4.1 (Risk bound for $\mathrm{H}_2\mathrm{O}$ ). Let $T \in \mathbb{N}$ be given, let $\mathcal{D}_{\mathrm{off}}$ consist of $H \cdot T$ samples from data distribution $\nu$ , and suppose that $\nu$ satisfies $C_{\star}$ -single-policy concentrability. Let $\mathbf{Alg}_{\mathrm{off}}$ be CC-bounded at scale $\gamma \in (0,1)$ under Assumption $(\cdot)$ , with parameters $\mathfrak{a}_{\gamma}$ and $\mathfrak{b}_{\gamma}$ . Suppose that for all $t \in [T]$ and $\pi^{(1)}, \ldots, \pi^{(t)} \in \Pi$ , Assumption $(\mu^{(t)}, M^{\star})$ holds for $\mu^{(t)} := \{1/2(\nu_h + 1/t \sum_{i=1}^t d_h^{\pi^{(i)}})\}_{h=1}^H$ . Then, with probability at least $1 - \delta T$ , the risk of $\mathrm{H}_2\mathrm{O}$ (Algorithm 2) with inputs $T$ , $\mathbf{Alg}_{\mathrm{off}}$ , and $\mathcal{D}_{\mathrm{off}}$ is bounded as

$$
\mathbf {R i s k} \leq \widetilde {O} \left(H \left(\frac {\mathfrak {a} _ {\gamma} (C _ {\star} + C _ {\text { cov }})}{T} + \mathfrak {b} _ {\gamma}\right)\right). \tag {12}
$$

Proof of Theorem 4.1. Under the assumptions in the theorem statement, Theorem F.6 implies that

$$
\mathbf {R i s k} \leq \frac {2 \mathfrak {a} _ {\gamma}}{T} \sum_ {h = 1} ^ {H} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {C C} _ {h} (\pi^ {\star}, \nu , \gamma t) + H \left(4 \frac {\mathfrak {a} _ {\gamma} C _ {\mathrm{cov}}}{T} (\gamma + 1 0 \log (T) + 1) + \mathfrak {b} _ {\gamma}\right).
$$

Since $\max_h\max_{x,a}\left|\frac{d_h^{\pi^\star}(x,a)}{\nu_h(x,a)}\right| \leq C_\star$ , we have $\mathbb{CC}_h(\pi^\star, \nu, \gamma t) \leq C_\star$ , so we can simplify the first term above to

$$
\frac {2 \mathfrak {a} _ {\gamma}}{T} \sum_ {h = 1} ^ {H} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {C C} _ {h} (\pi^ {\star}, \nu , \gamma t) \leq \frac {2 \mathfrak {a} _ {\gamma} C _ {\star}}{T} H \sum_ {t = 1} ^ {T} \frac {1}{t} \leq \frac {6 \mathfrak {a} _ {\gamma} C _ {\star}}{T} H \log (T),
$$

using the bound on the harmonic number $\sum_{t=1}^{T} 1/t \leq 3\log(T)$ . Combining with the remainder of the risk bound in Theorem F.6 gives us

$$
\mathbf {R i s k} \leq \frac {H \mathfrak {a} _ {\gamma}}{T} (6 C _ {\star} \log (T) + 4 C _ {\mathrm{cov}} (\gamma + 1 0 \log (T) + 1)) + H \mathfrak {b} _ {\gamma} = \widetilde {O} \left(H \left(\frac {\mathfrak {a} _ {\gamma} (C _ {\star} + C _ {\mathrm{cov}})}{T} + \mathfrak {b} _ {\gamma}\right)\right),
$$

where we have used the fact that $\gamma \in [0,1]$ .

Corollary F.1. Let $T \in N$ and $D_{off}$ consist of $H \cdot T$ samples from data distribution $\nu$ . Suppose that for all $t \in [T]$ and $\pi^{(1)}, \ldots, \pi^{(t)} \in \Pi$ , $\text{Assumption}(\mu^{(t)}, M^{\star})$ holds for $\mu^{(t)} := \{1/2(\nu_h + 1/t \sum_{i=1}^t d_h^{\pi^{(i)}})\}_{h=1}^H$ . Let $Alg_{off}$ be CC-bounded at scale $\gamma \in [0,1]$ under Assumption( $\cdot$ ) and with parameters $a_\gamma = \frac{a}{\gamma}$ and $b_\gamma = b\gamma$ . Consider the $H_2O$ algorithm with inputs T, $Alg_{off}$ , and $D_{off}$ . Then,

\- If $\mathbf{Alg}_{\mathrm{off}}$ is CC-bounded at scale $\gamma = \widetilde{\Theta}\left(\sqrt{\mathfrak{a}C_{\mathrm{cov}} / \mathfrak{b}T}\right)$ and $T$ is such that $\gamma \in [0,1]$ , we have

$$
\mathbf {R i s k} \leq 2 \sqrt {\frac {\mathfrak {a b}}{T C _ {\mathrm{cov}}}} \sum_ {h = 1} ^ {H} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathbb {C C} _ {h} (\pi^ {\star}, \nu , \gamma^ {(t)}) + \widetilde {O} \left(H \sqrt {\frac {C _ {\mathrm{cov}} \mathfrak {a b}}{T}}\right).
$$

with probability greater than $1 - \delta T$ .

\- If $\nu$ satisfies $C_{\star}$ -single-policy concentrability and $\mathbf{Alg}_{\mathrm{off}}$ is CC-bounded at scale $\gamma = \widetilde{\Theta}\left(\sqrt{\mathfrak{a}(C_{\star} + C_{\mathrm{cov}}) / \mathfrak{b}T}\right)$ and $T$ is such that $\gamma \in [0,1]$ , then

$$
\mathbf {R i s k} \leq \widetilde {O} \left(H \sqrt {\left(C _ {\star} + C _ {\mathrm{cov}}\right) \mathfrak {a b} / T}\right),
$$

with probability greater than $1 - \delta T$ .

Proof of Corollary F.1. We start with the first case. We recall the definition of $err_{off}$ appearing in Theorem F.6:

$$
\operatorname{err} _ {\text {off}} := H \left(\frac {4 \mathfrak {a} _ {\gamma} C _ {\text {cov}}}{T} (\gamma + 8 \log (T) + 1) + \mathfrak {b} _ {\gamma}\right) = \widetilde {O} \left(H \left(\frac {C _ {\text {cov}} \mathfrak {a} _ {\gamma}}{T} + \mathfrak {b} _ {\gamma}\right)\right).
$$

and the risk bound from Theorem F.6.

$$
\mathbf {R i s k} \leq \frac {2 \mathfrak {a} _ {\gamma}}{T} \sum_ {h = 1} ^ {H} \sum_ {t = 1} ^ {T} \frac {1}{t} \mathsf {C C} _ {h} (\pi^ {\star}, \nu , \gamma^ {(t)}) + \mathsf {e r r} _ {\text { off }}. \tag {47}
$$

Plugging in $a_{\gamma} = \frac{a}{\gamma}$ and $b_{\gamma} = b\gamma$ into $err_{off}$ gives

$$
\operatorname{err} _ {\text { off }} = H \left(4 \frac {\mathfrak {a}}{\gamma T} C _ {\text { cov }} (\gamma + 8 \log (T) + 1) + \mathfrak {b} \gamma\right).
$$

The above is optimized by picking $\gamma = 2\sqrt{\frac{\mathfrak{a}}{\mathfrak{b}}\frac{C_{\mathrm{cov}}}{T}(8\log(T) + 1)}$ . Plugging this in gives us

$$
\operatorname{err} _ {\text { off }} = H \left(4 \sqrt {\frac {\mathfrak {a b} C _ {\text { cov }} (8 \log (T) + 1)}{T}} + 4 \frac {\mathfrak {a} C _ {\text { cov }}}{T}\right) = \widetilde {O} \left(H \sqrt {\frac {\mathfrak {a b} C _ {\text { cov }}}{T}}\right),
$$

as desired. For the second result, recall from Theorem 4.1 that the risk bound when $\nu$ satisfied $C_{\star}$ -policy-concentrability is

$$
\mathbf {R i s k} \leq \frac {H \mathfrak {a} _ {\gamma}}{T} (6 C _ {\star} \log (T) + 4 C _ {\operatorname{cov}} (\gamma + 8 \log (T) + 1)) + H \mathfrak {b} _ {\gamma}.
$$

Plugging in $\mathfrak{a}_{\gamma} = \frac{\mathfrak{a}}{\gamma}$ and $\mathfrak{b}_{\gamma} = \mathfrak{b}\gamma$ gives us

$$
\mathbf {R i s k} \leq \frac {H \mathfrak {a}}{T \gamma} (6 C _ {\star} \log (T) + 4 C _ {\operatorname{cov}} (\gamma + 8 \log (T) + 1)) + H \mathfrak {b} \gamma .
$$

This expression is optimized by $\gamma = \sqrt{\frac{\mathfrak{a}(6C_{\star}\log(T) + 4C_{\mathrm{cov}}(\gamma + 8\log(T) + 1))}{T\mathfrak{b}}}$ , which when substituted gives us the risk bound

$$
\mathbf {R i s k} \leq 2 H \sqrt {\frac {\mathfrak {a b} (6 C _ {\star} \log (T) + 4 C _ {\mathrm{cov}} (\gamma + 8 \log (T) + 1))}{T}} = \widetilde {O} \left(H \left(\sqrt {\frac {(C _ {\star} + C _ {\mathrm{cov}}) \mathfrak {a b}}{T}}\right)\right),
$$

as desired.

![](images/6c32863bcd60ca3b108a1710a6b14c11b7bee7595bdf092f0b907de1e3eea925.jpg)

# F.3 Proofs for $H_{2}O$ Examples (Appendix F.1)

# F.3.1 Supporting Technical Results

Lemma F.2 (Telescoping Performance Difference (Xie and Jiang (2020, Theorem 2); Jin et al. (2021b, Lemma 3.1))). For any $f \in \mathcal{F}$ , we have that

$$
J (\pi^ {\star}) - J (\pi_ {f}) \leq \sum_ {h = 1} ^ {H} \mathbb {E} _ {d _ {h} ^ {\pi^ {\star}}} [ \mathcal {T} _ {h} f _ {h + 1} (x _ {h}, a _ {h}) - f _ {h} (x _ {h}, a _ {h}) ] + \mathbb {E} _ {d _ {h} ^ {\pi_ {f}}} [ f _ {h} (x _ {h}, a _ {h}) - \mathcal {T} _ {h} f _ {h + 1} (x _ {h}, a _ {h}) ].
$$

This bound follows from a straightforward adaptation of the proof of Xie and Jiang (2020, Theorem 2) to the finite horizon setting.

Lemma F.3. For all policy $\pi \in \Pi$ , value function $f \in \mathcal{F}$ , timestep $h \in [H]$ , data distribution $\mu = \{\mu_h\}_{h=1}^H$ where $\mu_h \in \Delta(\mathcal{X} \times \mathcal{A})$ , and $\gamma \in \mathbb{R}_+$ , we have

$$
\mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \left[ \Delta_ {h} f \right] \left(x _ {h}, a _ {h}\right) \right] \leq \mathbb {E} _ {\mu_ {h}} \left[ \left[ \Delta_ {h} f \right] \left(x _ {h}, a _ {h}\right) \cdot \operatorname{clip} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi} \left(x _ {h} , a _ {h}\right)}{\mu_ {h} \left(x _ {h} , a _ {h}\right)} \right] \right] + 2 \mathbb {P} ^ {\pi} \left[ \frac {d _ {h} ^ {\pi} \left(x _ {h} , a _ {h}\right)}{\mu_ {h} \left(x _ {h} , a _ {h}\right)} > \gamma \right].
$$

Similarly,

$$
\mathbb {E} _ {d _ {h} ^ {\pi}} \left[ - \left[ \Delta_ {h} f \right] \left(x _ {h}, a _ {h}\right) \right] \leq \mathbb {E} _ {\mu_ {h}} \left[ \left(- \left[ \Delta_ {h} f \right] \left(x _ {h}, a _ {h}\right)\right) \cdot \operatorname{clip} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi} \left(x _ {h} , a _ {h}\right)}{\mu_ {h} \left(x _ {h} , a _ {h}\right)} \right] \right] + 2 \mathbb {P} ^ {\pi} \left[ \frac {d _ {h} ^ {\pi} \left(x _ {h} , a _ {h}\right)}{\mu_ {h} \left(x _ {h} , a _ {h}\right)} > \gamma \right],
$$

where recall that $[\Delta_h f](x, a) := f_h(x, a) - [\mathcal{T}_h f_{h+1}](x, a)$ .

Proof of Lemma F.3. In the following, we prove the first inequality. The second inequality follows similarly. Using that $|[ \Delta_h f](x, a)| \leq 1$ for any $x, a \in \mathcal{X} \times \mathcal{A}$ , we have

$$
\mathbb {E} _ {d _ {h} ^ {\pi}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) ] \leq \mathbb {E} _ {d _ {h} ^ {\pi}} [ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \mathbb {I} \{\mu_ {h} (x _ {h}, a _ {h}) \neq 0 \} ] + \mathbb {E} _ {d _ {h} ^ {\pi}} [ \mathbb {I} \{\mu_ {h} (x _ {h}, a _ {h}) = 0 \} ].
$$

For the second term, for any $\gamma > 0$ ,

$$
\mathbb {E} _ {d _ {h} ^ {\pi}} [ \mathbb {I} \{\mu_ {h} (x _ {h}, a _ {h}) = 0 \} ] \leq \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \mathbb {I} \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} > \gamma \right\} \right] = \mathbb {P} ^ {\pi} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} > \gamma \right].
$$

For the first term, using that $u \leq \min \{u, v\} + u\mathbb{I}\{u \geq v\}$ for all $u, v \geq 0$ , and that $|[\Delta_h f](x, a)| \leq 1$ for any $x, a \in \mathcal{X} \times \mathcal{A}$ , we get that

$$
\begin{array}{l} \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \left[ \Delta_ {h} f \right] (x _ {h}, a _ {h}) \cdot \mathbb {I} \{\mu_ {h} (x _ {h}, a _ {h}) \neq 0 \} \right] \\ = \mathbb {E} _ {\mu_ {h}} \left[ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \mathbb {I} \{\mu (x _ {h}, a _ {h}) \neq 0 \} \right] \\ \leq \mathbb {E} _ {\mu_ {h}} \left[ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \operatorname{clip} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \right] \mathbb {I} \{\mu_ {h} (x _ {h}, a _ {h}) \neq 0 \} \right] \\ + \mathbb {E} _ {\mu_ {h}} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \mathbb {I} \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} > \gamma \right\} \mathbb {I} \left\{\mu_ {h} (x _ {h}, a _ {h}) \neq 0 \right\} \right] \\ = \mathbb {E} _ {\mu_ {h}} \left[ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \operatorname{clip} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \right] \mathbb {I} \{\mu_ {h} (x _ {h}, a _ {h}) \neq 0 \} \right] + \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \mathbb {I} \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} > \gamma \right\} \right]. \\ \end{array}
$$

Furthermore, also note that

$$
\begin{array}{l} \mathbb {E} _ {\mu_ {h}} \left[ [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \mathsf {c l i p} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \right] \mathbb {I} \{\mu_ {h} (x _ {h}, a _ {h}) = 0 \} \right] \\ = \sum_ {(x _ {h}, a _ {h}) \text {s.t.} \mu_ {h} (x _ {h}, a _ {h}) = 0} \mu_ {h} (x _ {h}, a _ {h}) \cdot [ \Delta_ {h} f ] (x _ {h}, a _ {h}) \cdot \mathsf {c l i p} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \right] = 0. \\ \end{array}
$$

The final bound follows by combining the above three terms.

Lemma F.4. For any policy $\pi$ , data distribution $\mu = \{\mu_h\}_{h=1}^H$ where $\mu_h \in \Delta(\mathcal{X} \times \mathcal{A})$ , scale $\gamma \in \mathbb{R}_+$ and horizon $h \in [H]$ , we have

$$
\mathbb {P} ^ {\pi} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} > \gamma \right] \leq \frac {2}{\gamma} \left\| \mathsf {c l i p} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi}}{\mu_ {h}} \right] \right\| _ {1, d _ {h} ^ {\pi}}.
$$

Proof of Lemma F.4. Note that

$$
\begin{array}{l} \mathbb {P} ^ {\pi} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} > \gamma \right] = \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \mathbb {I} \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} > \gamma \right\} \right] \\ \leq \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \mathbb {I} \bigg \{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x , a) + \gamma^ {- 1} d _ {h} ^ {\pi} (x _ {h} , a _ {h})} > \frac {\gamma}{2} \bigg \} \right] \\ \leq \frac {2}{\gamma} \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h}) + \gamma^ {- 1} d _ {h} ^ {\pi} (x _ {h} , a _ {h})} \right] \\ \leq \frac {2}{\gamma} \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \mathsf {c l i p} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \right] \right] = \frac {2}{\gamma} \left\| \mathsf {c l i p} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi}}{\mu_ {h}} \right] \right\| _ {1, d _ {h} ^ {\pi}}, \\ \end{array}
$$

where the first inequality follows from

$$
\gamma^ {- 1} d _ {h} ^ {\pi} (x, a) > \mu_ {h} (x, a) \implies \gamma^ {- 1} d _ {h} ^ {\pi} (x, a) > \frac {1}{2} (\mu_ {h} (x, a) + \gamma^ {- 1} d _ {h} ^ {\pi} (x, a)),
$$

and the second inequality follows by Markov's inequality.

Lemma F.5. For any policy $\pi$ , data distribution $\mu = \{\mu_h\}_{h=1}^H$ where $\mu_h \in \Delta(\mathcal{X} \times \mathcal{A})$ , scale $\gamma \in \mathbb{R}_+$ , and horizon $h \in [H]$ , we have

$$
\left\| \mathsf {c l i p} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi}}{\mu_ {h}} \right] \right\| _ {2, \mu_ {h}} ^ {2} \leq 2 \left\| \mathsf {c l i p} _ {\gamma} \left[ \frac {d _ {h} ^ {\pi}}{\mu_ {h}} \right] \right\| _ {1, d _ {h} ^ {\pi}}.
$$

Proof of Lemma F.5. Beginning with the left-hand side, we have,

$$
\begin{array}{l} \mathbb {E} _ {\mu_ {h}} \left[ \min \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})}, \gamma \right\} ^ {2} \right] \stackrel {\mathrm{(i)}} {\leq} 2 \mathbb {E} _ {\mu_ {h}} \left[ \left(\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})}\right) ^ {2} \mathbb {I} \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \leq \gamma \right\} \right] \\ + 2 \mathbb {E} _ {\mu_ {h}} \left[ \gamma \cdot \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \cdot \mathbb {I} \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} > \gamma \right\} \right] \\ \stackrel {\text {(ii)}} {\leq} 2 \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \mathbb {I} \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} \leq \gamma \right\} \right] + 2 \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \gamma \cdot \mathbb {I} \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})} > \gamma \right\} \right] \\ \stackrel {\mathrm{(iii)}} {\leq} 2 \mathbb {E} _ {d _ {h} ^ {\pi}} \left[ \min \left\{\frac {d _ {h} ^ {\pi} (x _ {h} , a _ {h})}{\mu_ {h} (x _ {h} , a _ {h})}, \gamma \right\} \right]. \\ \end{array}
$$

In (i), we have used that for all $u, v \in \mathbb{R}^+$ , $\min \{u, v\} \leq u\mathbb{I}\{u \leq v\} + \sqrt{uv}\mathbb{I}\{v < u\}$ and that $(u + v)^2 \leq 2(u^2 + v^2)$ , thus that $\min \{u, v\}^2 \leq 2(u^2\mathbb{I}\{u \leq v\} + uv\mathbb{I}\{v < u\})$ . In (ii), we have done a change of measure from $\mu_h$ to $d_h^\pi$ . In (iii), we have used that $\min \{u, v\} = u\mathbb{I}\{u \leq v\} + v\mathbb{I}\{v < u\}$ .

![](images/534094f31a31191c284d0888c1948a896ad8003fd5c4a2ba00c3923a8cb0be77.jpg)

# F.3.2 Proofs for MABO.CR (Proof of Theorem F.1)

Suppose the dataset $D = \{D_{h}\}_{h \leq H}$ is sampled from the offline distribution $\mu^{(1:n)}$ . In this section, we analyze our regularized and clipped variant of MABO (Eq. (13)). We analyze MABO.CR under the following density ratio assumption. Recall for any sequence of data distributions $\mu^{(1)}, \ldots, \mu^{(n)}$ , we denote the mixture distribution by $\mu^{(1:n)} = \{\mu_{h}^{(1:n)}\}_{h=1}^{H}$ , defined by $\mu_{h}^{(1:n)} := \frac{1}{n} \sum_{i=1}^{n} \mu_{h}^{(i)}$ .

Assumption F.4. For a given sequence of data distributions $\mu^{(1)},\ldots \mu^{(n)}$ , we have that for all $\pi \in \Pi$ , and for all $h\in [H]$ ,

$$
\frac {d _ {h} ^ {\pi}}{\mu_ {h} ^ {(1 : n)}} \in \mathcal {W}.
$$

We first note the following bound for the hypothesis $\widehat{f}$ returned by MABO.CR.

Lemma F.6. Let $D = \{D_h\}_{h=1}^H$ be a dataset consisting of $H \cdot n$ samples from $\mu^{(1)}, \ldots, \mu^{(n)}$ . Suppose that F satisfies Assumption 2.1 and that W satisfies Assumption F.4 with respect to $\mu^{(1:n)}$ . Let $\widehat{f}$ be the function returned by MABO.CR, given in Eq. (13), when executed on $\{D_h\}_{h \leq H}$ with parameters $\gamma$ , F, and the augmented weight function class $\overline{W}$ defined in Eq. (38). Then, with probability at least $1 - \delta$ , we have

$$
\sum_ {h = 1} ^ {H} \left| \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left[ \Delta_ {h} \widehat {f} \right] \left(x _ {h}, a _ {h}\right) \cdot \check {w} _ {h} \left(x _ {h}, a _ {h}\right) \right] \right| \leq \sum_ {h = 1} ^ {H} \frac {2 0}{\gamma^ {(n)}} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left(\check {w} _ {h} \left(x _ {h}, a _ {h}\right)\right) ^ {2} \right] + \frac {7}{1 8} H ^ {2} \beta^ {(n)}, \tag {48}
$$

for all $w \in W$ , where $\beta^{(n)} := \frac{36\gamma^{(n)}}{n-1} \log(24|\mathcal{F}||\mathcal{W}|H^{2}/\delta)$ .

Proof of Lemma F.6. Repeating the argument of Lemma E.2 (a), we can establish that $Q^{\star}$ satisfies the following bound for all $h \in [H]$ , $w \in \overline{\mathcal{W}}$ :

$$
\widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \left([ \widehat {\Delta} _ {h} Q ^ {\star} ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime})\right) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \leq \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \alpha^ {(n)} \cdot \left(\check {w} _ {h} (x _ {h}, a _ {h})\right) ^ {2} \right] + \beta^ {(n)}, \tag {49}
$$

with probability at least $1 - \delta$ , where $\alpha^{(n)} = 8 / \gamma^{(n)}$ and $\beta^{(n)} = 36\gamma^{(n)} / n - 1\log (6|\mathcal{F}||\overline{\mathcal{W}} |H / \delta)\leq 36\gamma^{(n)} / n - 1\log (24|\mathcal{F}||\mathcal{W}|H^2 /\delta)$ . Going forward, we condition on the event that this holds. Now, since the right-hand side of Eq. (49) is independent of the sign of $\check{w}_h$ , we can apply this to $-\check{w}\in \overline{\mathcal{W}}$ to conclude that

$$
\left| \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \left([ \widehat {\Delta} _ {h} Q ^ {\star} ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime})\right) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \right| \leq \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \alpha^ {(n)} \cdot \left(\check {w} _ {h} (x _ {h}, a _ {h})\right) ^ {2} \right] + \beta^ {(n)}.
$$

Summing over $h \in [H]$ and taking the max over $w \in \overline{W}$ , we can conclude that:

$$
\max _ {w \in \overline {{\mathcal {W}}}} \sum_ {h = 1} ^ {H} \left| \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \left([ \widehat {\Delta} _ {h} Q ^ {\star} ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime})\right) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \right| - \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \alpha^ {(n)} \cdot \left(\check {w} _ {h} (x _ {h}, a _ {h})\right) ^ {2} \right] \leq H \beta^ {(n)}.
$$

By Assumption 2.1 and the definition of the hypothesis $\widehat{f}$ returned by MABO.CR, we have:

$$
\max _ {w \in \overline {{\mathcal {W}}}} \sum_ {h = 1} ^ {H} \Big | \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \big ([ \widehat {\Delta} _ {h} \widehat {f} ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \big) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \Big | - \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \left[ \alpha^ {(n)} \cdot \big (\check {w} _ {h} (x _ {h}, a _ {h}) \big) ^ {2} \right] \leq H \beta^ {(n)},
$$

and in particular the bound that for all $w \in W$

$$
\sum_ {h = 1} ^ {H} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \Big [ \big ([ \widehat {\Delta} _ {h} \widehat {f} ] (x _ {h}, a _ {h}, r _ {h}, x _ {h + 1} ^ {\prime}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \big) \Big ] \leq \sum_ {h = 1} ^ {H} \widehat {\mathbb {E}} _ {\mathcal {D} _ {h}} \Big [ \alpha^ {(n)} \cdot \big (\check {w} _ {h} (x _ {h}, a _ {h}) \big) ^ {2} \Big ] + H \beta^ {(n)}.
$$

Repeating the argument for Lemma E.2 (b), we can conclude that for all $w \in \overline{\mathcal{W}}$

$$
\sum_ {h = 1} ^ {H} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ [ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \leq \sum_ {h = 1} ^ {H} \frac {2 0}{\gamma^ {(n)}} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \frac {7 H \beta^ {(n)}}{1 8}.
$$

Applying this to $w^{(h)} \in \overline{\mathcal{W}}$ and to $-w^{(h)} \in \overline{\mathcal{W}}$ , and again noting that the right-hand side is independent of the sign of $\check{w}$ , we can conclude that for each $h \in [H]$ , $w \in \overline{W}$ :

$$
\left| \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left[ \Delta_ {h} \widehat {f} \right] (x _ {h}, a _ {h}) \cdot \check {w} _ {h} (x _ {h}, a _ {h}) \right] \right| \leq \frac {2 0}{\gamma^ {(n)}} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} \right] + \frac {7 H \beta^ {(n)}}{1 8},
$$

Summing over $h \in [H]$ gives the desired bound.

Theorem F.1 (MABO.CR is CC-bounded). Let $D = \{D_h\}_{h=1}^H$ consist of $H \cdot n$ samples from $\mu^{(1)}, \ldots, \mu^{(n)}$ . For any $\gamma \in R_+$ , the MABO.CR algorithm (Eq. (13)) with parameters F, augmented class W defined in Eq. (38) in Appendix F.1.1, and $\gamma$ is CC-bounded at scale $\gamma$ under the Assumption that $Q^\star \in F$ and that for all $\pi \in \Pi$ and $h \in [H]$ , $d_h^\pi / \mu_h^{(1:n)} \in W$ .

Proof of Theorem F.1. Let $\widehat{\pi} = \pi_{\widehat{f}}$ . Using the performance difference lemma (Lemma F.2), we note that

$$
J (\pi^ {\star}) - J (\widehat {\pi}) \leq \sum_ {h = 1} ^ {H} \mathbb {E} _ {d _ {h} ^ {\pi^ {\star}}} [ - [ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h}) ] + \sum_ {h = 1} ^ {H} \mathbb {E} _ {d _ {h} ^ {\widehat {\pi}}} [ [ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h}) ]. \tag {50}
$$

However, note that due to Lemma F.3 and Lemma F.4, we have that

$$
\mathbb {E} _ {d _ {h} ^ {\widehat {\pi}}} \left[ \left(\left[ \Delta_ {h} \widehat {f} \right] \left(x _ {h}, a _ {h}\right)\right) \right] \leq \underbrace {\mathbb {E} _ {\mu_ {h} ^ {(1 : n)}} \left[ \left(\left[ \Delta_ {h} \widehat {f} \right] \left(x _ {h} , a _ {h}\right)\right) \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}} \left(x _ {h} , a _ {h}\right)}{\mu_ {h} ^ {(1 : n)} \left(x _ {h} , a _ {h}\right)} \right] \right]} _ {(I)} + \frac {4}{\gamma n} \left\| \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\widehat {\pi}}}, \tag {51}
$$

and,

$$
\mathbb {E} _ {d _ {h} ^ {\pi^ {\star}}} \left[ \left(- \left[ \Delta_ {h} \widehat {f} \right] \left(x _ {h}, a _ {h}\right)\right) \right] \leq \underbrace {\mathbb {E} _ {\mu_ {h} ^ {(1 : n)}} \left[ \left(- \left[ \Delta_ {h} \widehat {f} \right] \left(x _ {h} , a _ {h}\right)\right) \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\pi^ {\star}} \left(x _ {h} , a _ {h}\right)}{\mu_ {h} ^ {(1 : n)} \left(x _ {h} , a _ {h}\right)} \right] \right]} _ {\text {(II)}} + \frac {4}{\gamma n} \left\| \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\pi^ {\star}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\pi^ {\star}}}. \tag {52}
$$

We bound the terms (I) and (II) separately below. Before we delve into these bounds, note that using Lemma F.6, we have with probability at least $1 - \delta$ ,

$$
\max _ {w \in \mathcal {W}} \sum_ {h = 1} ^ {H} \left(\left| \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \check {w} _ {h} ([ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h})) \right] \right| - \frac {2 0}{\gamma^ {(n)}} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} [ (\check {w} _ {h} (x _ {h}, a _ {h})) ^ {2} ]\right) \leq \frac {7}{1 8} H ^ {2} \beta^ {(n)}, \tag {53}
$$

where $\beta^{(n)} := \frac{36\gamma n}{n-1} \log(24|\mathcal{F}||\mathcal{W}|H^{2}/\delta)$ . Moving forward, we condition on the event under which Eq. (53) holds.

Bound on Term (I). Define w via $w_{h} := \frac{d_{h}^{\hat{\pi}}}{\mu_{h}^{(1:n)}}$ and $\check{w}_{h} := \operatorname{clip}_{\gamma n} \left[ \frac{d_{h}^{\hat{\pi}}}{\mu_{h}^{(1:n)}} \right]$ , and note that due to Assumption F.4, we have that $w \in W$ . Thus, using (53), we get that

$$
\begin{array}{l} \sum_ {h = 1} ^ {H} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} [ (- [ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h})) \check {w} _ {h} (x _ {h}, a _ {h}) ] \leq \sum_ {h = 1} ^ {H} \left| \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} [ (- [ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h})) \check {w} _ {h} (x _ {h}, a _ {h}) ] \right| \\ \leq \sum_ {h = 1} ^ {H} \frac {2 0}{\gamma^ {(n)}} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left(\check {w} _ {h} (x _ {h}, a _ {h})\right) ^ {2} \right] + \frac {7}{1 8} H ^ {2} \beta^ {(n)} \\ = \sum_ {h = 1} ^ {H} \frac {2 0}{\gamma^ {(n)}} \left\| \operatorname{clip} _ {\gamma n} [ w _ {h} ] \right\| _ {2, \mu_ {h} ^ {(1: n)}} ^ {2} + \frac {7}{1 8} H ^ {2} \beta^ {(n)} \\ \leq \sum_ {h = 1} ^ {H} \frac {4 0}{\gamma^ {(n)}} \left\| \operatorname{clip} _ {\gamma n} [ w _ {h} ] \right\| _ {1, d _ {h} ^ {\widehat {\pi}}} + \frac {7}{1 8} H ^ {2} \beta^ {(n)}, \\ \end{array}
$$

where the last step follows by Lemma F.5 and the definition of $w_{h}$ .

Bound on Term (II). Define $w_{h}^{\star} := \frac{d_{h}^{\pi^{\star}}}{\mu_{h}^{(1:n)}}$ and $\check{w}_{h}^{\star} := \mathsf{clip}_{\gamma n} \left[ \frac{d_{h}^{\pi^{\star}}}{\mu_{h}^{(1:n)}} \right]$ , and again note that due to Assumption F.4, we have that $w_{h} \in W_{h}$ . Thus, using (53), and repeating the same arguments as above, we get that

$$
\text {(II)} \leq \sum_ {h = 1} ^ {H} \frac {4 0}{\gamma^ {(n)}} \left\| \operatorname{clip} _ {\gamma n} [ w _ {h} ^ {\star} ] \right\| _ {1, d _ {h} ^ {\pi^ {\star}}} + \frac {7}{1 8} H ^ {2} \beta^ {(n)},
$$

Plugging the two bounds above in (55) and (56), and then further in (54), and using the definitions for $w_h$ and $w_h^\star$ , we get that with probability at least $1 - \delta$ ,

$$
\begin{array}{l} J (\pi^ {\star}) - J (\widehat {\pi}) \leq \sum_ {h = 1} ^ {H} \frac {4 0}{\gamma n} \left\| \mathsf {c l i p} _ {\gamma n} \left[ \frac {d _ {h} ^ {\pi^ {\star}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\pi^ {\star}}} + \sum_ {h = 1} ^ {H} \frac {5}{\gamma n} \left\| \mathsf {c l i p} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\widehat {\pi}}} + \frac {1 4}{1 8} H ^ {2} \beta^ {(n)} \\ = \sum_ {h = 1} ^ {H} \frac {4 0}{\gamma^ {(n)}} \left(\left\| \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\pi^ {\star}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\pi^ {\star}}} + \left\| \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\widehat {\pi}}}\right) + \frac {1 4}{1 8} H ^ {2} \beta^ {(n)} \\ = \sum_ {h = 1} ^ {H} \frac {4 0}{\gamma^ {(n)}} \left(\mathbb {C C} _ {h} \left(\pi^ {\star}, \mu^ {(1: n)}, \gamma n\right) + \mathbb {C C} _ {h} \left(\widehat {\pi}, \mu^ {(1: n)}, \gamma n\right)\right) + 2 8 H ^ {2} \gamma \frac {n}{n - 1} \log (2 4 | \mathcal {F} | | \mathcal {W} | H ^ {2} / \delta) \\ \leq \sum_ {h = 1} ^ {H} \frac {4 0}{\gamma n} \left(\mathbb {C C} _ {h} (\pi^ {\star}, \mu^ {(1: n)}, \gamma n) + \mathbb {C C} _ {h} (\widehat {\pi}, \mu^ {(1: n)}, \gamma n)\right) + 5 6 H ^ {2} \gamma \log (2 4 | \mathcal {F} | | \mathcal {W} | H ^ {2} / \delta), \\ \end{array}
$$

which establishes that the algorithm is CC-bounded for scale $\gamma$ , with scaling functions $a_{\gamma} = \frac{40}{\gamma}$ and $b_{\gamma} = 56H^{2}\gamma \log(24|\mathcal{F}||\mathcal{W}|H^{2}/\delta)$ .

□

Corollary 4.1 (HYGLOW Risk bound). Let $\varepsilon > 0$ be given, $\mathcal{D}_{\mathrm{off}}$ consist of $H \cdot T$ samples from data distribution $\nu$ , where $\nu$ satisfies $C_{\star}$ -single-policy concentrability. Suppose that $Q^{\star} \in \mathcal{F}$ and that for all $t \in [T]$ , $\pi \in \Pi$ , and $h \in [H]$ , we have $d_h^\pi / \mu_h^{(t)} \in \mathcal{W}$ , where $\mu_h^{(t)} := 1/2 (\nu_h + 1/t \sum_{i=1}^t d_h^{\pi^{(i)}})$ . Then, HYGLOW with inputs $T = \widetilde{\Theta}((H^4(C_{\mathrm{cov}} + C_{\star})/\varepsilon^2) \cdot \log(|\mathcal{F}||\mathcal{W}|/\delta))$ , $\mathcal{F}$ , augmented $\overline{\mathcal{W}}$ defined in Eq. (38), $\gamma = \widetilde{\Theta}\left(\sqrt{(C_{\star} + C_{\mathrm{cov}})/TH^2 \log(|\mathcal{F}||\mathcal{W}|/\delta)}\right)$ , and $\mathcal{D}_{\mathrm{off}}$ returns an $\varepsilon$ -suboptimal policy with probability at least $1 - \delta T$ after collecting

$$
N = \widetilde {O} \bigg (\frac {H ^ {2} (C _ {\mathrm{cov}} + C _ {\star})}{\varepsilon^ {2}} \log (| \mathcal {F} | | \mathcal {W} | / \delta) \bigg)
$$

trajectories.

Proof of Corollary 4.1. This follows by combining Theorem F.1 with Corollary F.1.

![](images/e5cbc01d095355073c70d0ee7bf18c579ed67e642ab67656fb6e00593a2dbe4f.jpg)

# F.3.3 Proofs for Fitted Q-Iteration (FQI)

In this section we prove Theorem F.3.

We quote the following generalization bound for least squares regression in the adaptive setting.

Lemma F.7 (Least squares generalization bound; Song et al. (2023, Lemma 3)). Let $R > 0, \delta \in (0,1)$ , and $\mathcal{H}: \mathcal{X} \mapsto [-R,R]$ a class of real-valued functions. Let $\mathcal{D} = \{(x_1,y_1)\dots(x_T,y_T)\}$ be a dataset of $T$ points where $x_t \sim \rho_t(x_{1:t-1},y_{1:t-1})$ and $y_t = h^\star(x_t) + \varepsilon_t$ for some realizable $h^\star \in \mathcal{H}$ and $\varepsilon_t$ is conditionally mean-zero, i.e. $\mathbb{E}[y_t \mid x_t] = h^\star(x_t)$ . Suppose $\max_t |y_t| \leq R$ and $\max_x |h^\star(x)| \leq R$ . Then the least squares solution $\widehat{h} \in \arg\min_{h \in \mathcal{H}} \sum_{t=1}^T (h(x_t) - y_t)^2$ satisfies that with probability at least $1 - \delta$ ,

$$
\sum_ {t = 1} ^ {T} \mathbb {E} _ {x \sim \rho_ {t}} \left[ (\widehat {h} (x) - h ^ {\star} (x)) ^ {2} \right] \leq 2 5 6 R ^ {2} \log (2 | \mathcal {H} | / \delta).
$$

Using the above theorem, we can show the following concentration result for FQI.

Lemma F.8 (Concentration bound for FQI). With probability at least $1 - \delta$ , we have that for all $h \in [H]$ ,

$$
\begin{array}{l} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left([ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h})\right) ^ {2} \right] = \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {(x _ {h}, a _ {h}) \sim \mu_ {h} ^ {(i)}} \left[ \left(\widehat {f} _ {h} (x _ {h}, a _ {h}) - [ \mathcal {T} _ {h} \widehat {f} _ {h + 1} ] (x _ {h}, a _ {h})\right) ^ {2} \right] \\ \leq 1 0 2 4 \frac {\log (2 | \mathcal {F} | H / \delta)}{n}. \\ \end{array}
$$

Proof of Lemma F.8. Fix $h + 1$ . Consider the regression problem induced by the dataset $\mathcal{D}_h = \{(z_h^{(i)},y_h^{(i)})\}_{i=1}^n$ where $z_h^{(i)} = (x_h^{(i)},a_h^{(i)})\sim \mu_h^{(i)}$ and $y_h^{(i)} = r^{(i)} + \max_{a'}\widehat{f}_{h+1}(x_{h+1}^{(i)},a')$ . This problem is realizable via the regression function $\mathbb{E}[y_h^{(i)}\mid z_h^{(i)}] = h^\star (z_h^{(i)}) = \mathcal{T}\widehat{f}_{h+1}(z_h^{(i)})\in \mathcal{F}$ , and satisfies that $|y_h^{(i)}|\leq 2,|h^\star (z_h^{(i)})|\leq 2$ . In this regression problem, the least squares solution from Lemma F.7 is precisely the FQI solution, so by Lemma F.7 we have

$$
\begin{array}{l} \mathbb {E} _ {(x _ {h}, a _ {h}) \sim \mu_ {h} ^ {(1: n)}} \left(\widehat {f} _ {h} (x _ {h}, a _ {h}) - [ \mathcal {T} \widehat {f} _ {h + 1} ] (x _ {h}, a _ {h}))\right) ^ {2} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbb {E} _ {(x _ {h}, a _ {h}) \sim \mu_ {h} ^ {(i)}} \left(\widehat {f} _ {h} (x _ {h}, a _ {h}) - [ \mathcal {T} \widehat {f} _ {h + 1} ] (x _ {h}, a _ {h})\right) ^ {2} \\ \leq 1 0 2 4 \frac {\log (2 | \mathcal {F} | / \delta)}{n}, \\ \end{array}
$$

with high probability. Taking a union bound over $h \in [H]$ gives the desired result.

![](images/c3712bfe5140ea281dbc5520edfc4e72dbaaa74096ca7ccdc866c48fd3e78461.jpg)

Theorem F.3. The FQI algorithm is CC-bounded under Assumption F.2 with scaling functions $a_{\gamma} = \frac{6}{\gamma}$ and $b_{\gamma} = 1024 \log(2n|\mathcal{F}|)\gamma$ , for all $\gamma > 0$ simultaneously. As a consequence, when invoked within $H_{2}O$ , we have Risk $\leq \widetilde{O}\left(H \sqrt{(C_{\star} + C_{\mathrm{cov}}) \log(|\mathcal{F}|\delta^{-1})/T}\right)$ with probability at least $1 - \delta T$ .

Proof of Theorem F.3. Let $\widehat{\pi} = \pi_{\widehat{f}}$ . Using the performance difference lemma (given in Lemma F.2), we note that

$$
J (\pi^ {\star}) - J (\widehat {\pi}) \leq \sum_ {h = 1} ^ {H} \mathbb {E} _ {d _ {h} ^ {\pi^ {\star}}} [ - [ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h}) ] + \sum_ {h = 1} ^ {H} \mathbb {E} _ {d _ {h} ^ {\widehat {\pi}}} [ [ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h}) ]. \tag {54}
$$

However, note that due to Lemma F.3 and Lemma F.4, we have that

$$
\mathbb {E} _ {d _ {h} ^ {\widehat {\pi}}} \left[ \left(\left[ \Delta_ {h} \widehat {f} \right] \left(x _ {h}, a _ {h}\right)\right) \right] \leq \underbrace {\mathbb {E} _ {\mu_ {h} ^ {(1 : n)}} \left[ \left(\left[ \Delta_ {h} \widehat {f} \right] \left(x _ {h} , a _ {h}\right)\right) \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}} \left(x _ {h} , a _ {h}\right)}{\mu_ {h} ^ {(1 : n)} \left(x _ {h} , a _ {h}\right)} \right] \right]} _ {(I)} + \frac {4}{\gamma n} \left\| \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\widehat {\pi}}}, \tag {55}
$$

and,

$$
\mathbb {E} _ {d _ {h} ^ {\pi^ {\star}}} \left[ \left(- \left[ \Delta_ {h} \widehat {f} \right] \left(x _ {h}, a _ {h}\right)\right) \right] \leq \underbrace {\mathbb {E} _ {\mu_ {h} ^ {(1 : n)}} \left[ \left(- \left[ \Delta_ {h} \widehat {f} \right] \left(x _ {h} , a _ {h}\right)\right) \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\pi^ {\star}} \left(x _ {h} , a _ {h}\right)}{\mu_ {h} ^ {(1 : n)} \left(x _ {h} , a _ {h}\right)} \right] \right]} _ {\text {(II)}} + \frac {4}{\gamma n} \left\| \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\pi^ {\star}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\pi^ {\star}}}. \tag {56}
$$

Bound on Term (I). Note that

$$
\begin{array}{l} \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ ([ \Delta_ {h} \widehat {f} ] (x _ {h}, a _ {h})) \mathrm{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}} (x _ {h} , a _ {h})}{\mu_ {h} ^ {(1 : n)} (x _ {h} , a _ {h})} \right] \right] \leq \sqrt {\mathbb {E} _ {\mu_ {h} ^ {(1 : n)}} \left[ \left([ \Delta_ {h} \widehat {f} ] (x _ {h} , a _ {h})\right) ^ {2} \right] \mathbb {E} _ {\mu_ {h} ^ {(1 : n)}} \left[ \left(\mathrm{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}} (x _ {h} , a _ {h})}{\mu_ {h} ^ {(1 : n)} (x _ {h} , a _ {h})} \right]\right) ^ {2} \right]} \\ \leq \sqrt {2 0 4 8 \frac {\log (2 | \mathcal {F} | H)}{n}} \left\| \mathsf {c l i p} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {2, \mu_ {h} ^ {(1: n)}}, \\ \end{array}
$$

where the second line follows from Cauchy-Schwarz and the last line follows by Lemma F.8. AM-GM inequality implies that

$$
\begin{array}{l} \sqrt {\frac {2 0 4 8 \log (2 | \mathcal {F} | H)}{n}} \left\| \operatorname{clip} _ {n} \left[ \frac {d _ {h} ^ {\widehat {\pi}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {2, \mu_ {h} ^ {(1: n)}} \leq \frac {1}{2} \left(\frac {1}{\gamma n} \left\| \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {2, \mu_ {h} ^ {(1: n)}} ^ {2} + 2 0 4 8 \log (2 | \mathcal {F} | H) \gamma\right) \\ \leq \frac {1}{2} \left(\frac {4}{\gamma n} \left\| \mathsf {c l i p} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\widehat {\pi}}} + 2 0 4 8 \log (2 | \mathcal {F} | H) \gamma\right), \\ \end{array}
$$

where the last line is due to Lemma F.5.

Bound on Term (II). Repeating the same argument above for $\pi^{\star}$ , we get that

$$
\text {(II)} \leq \frac {2}{\gamma n} \left\| \operatorname{clip} _ {\gamma n} \left[ \frac {d _ {h} ^ {\pi^ {*}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\pi^ {*}}} + 1 0 2 4 \log (2 | \mathcal {F} | H) \gamma .
$$

Combining the above bounds implies that

$$
\begin{array}{l} J (\pi^ {\star}) - J (\widehat {\pi}) \leq \sum_ {h = 1} ^ {H} \frac {2}{\gamma n} \left(\left\| \mathsf {c l i p} _ {\gamma n} \left[ \frac {d _ {h} ^ {\pi^ {\star}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\pi^ {\star}}} + \left\| \mathsf {c l i p} _ {\gamma n} \left[ \frac {d _ {h} ^ {\widehat {\pi}}}{\mu_ {h} ^ {(1 : n)}} \right] \right\| _ {1, d _ {h} ^ {\widehat {\pi}}}\right) + 2 0 4 8 \log (2 | \mathcal {F} | H) \gamma \\ = \sum_ {h = 1} ^ {H} \frac {2}{\gamma n} \left(\mathbb {C C} _ {h} (\pi^ {\star}, \mu^ {(1: n)}, \gamma) + \mathbb {C C} _ {h} (\widehat {\pi}, \mu^ {(1: n)}, \gamma)\right) + 2 0 4 8 \log (2 | \mathcal {F} | H) \gamma , \\ \end{array}
$$

which shows that FQI is CC-bounded at scale $\gamma$ under Assumption F.2, with scaling functions $a_{\gamma} = \frac{2}{\gamma}$ and $b_{\gamma} = 2048 \log(2|\mathcal{F}|H)\gamma$ .

It remains to show the stated risk bound when FQI is applied within $H_{2}O$ . This follows by applying Corollary F.1.

# F.3.4 Proofs for Model-Based MLE

In this section we prove Theorem F.5. We quote the following generalization bound for maximum likelihood estimation (MLE) in the adaptive setting.

Lemma F.9 (MLE generalization bound; Agarwal et al. (2020, Theorem 18)). Consider a sequential conditional probability estimation setting with an instance space $\mathcal{Z}$ and a target space $\mathcal{Y}$ and conditional density $p(y \mid z)$ . We are given $\mathcal{F} : (\mathcal{Z} \times \mathcal{Y}) \to \mathbb{R}$ . We are given a dataset $D = \{(z_t, y_t)\}_{t=1}^T$ where $z_t \sim \mathcal{D}_t = \mathcal{D}_t(z_{1:t-1}, y_{1:t-1})$ and $y_t \sim p(\cdot \mid z_t)$ . Fix $\delta \in (0,1)$ , assume $|\mathcal{F}| < \infty$ and there exists $f^\star \in \mathcal{F}$ such that $f^\star(y \mid z) = p(y \mid z)$ . Then, with probability at least $1 - \delta$ ,

$$
\sum_ {t = 1} ^ {T} \mathbb {E} _ {z \sim \rho_ {t}} \left\| \widehat {f} (\cdot | z) - f ^ {\star} (\cdot | z) \right\| _ {\mathrm{tv}} ^ {2} \leq 2 \log (| \mathcal {F} | / \delta),
$$

where

$$
\widehat {f} \in \arg \max _ {f \in \mathcal {F}} \sum_ {t = 1} ^ {T} \log f (y _ {t} \mid z _ {t}).
$$

Recall the notation that each model $M \in \mathcal{M}$ defines a conditional probability distribution of the form $M(r, x' \mid x, a)$ . We apply the above generalization bound to our setting to obtain the following.

Corollary F.2. Under Assumption F.3, we have that the model-based maximum likelihood estimator of Equation (41) satisfies

$$
\forall h \in [ H ]: \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left\| \widehat {M _ {h}} (\cdot | x _ {h}, a _ {h}) - M _ {h} ^ {\star} (\cdot | x _ {h}, a _ {h}) \right\| _ {\mathrm{tv}} ^ {2} \right] \leq 2 \frac {\log (| \mathcal {M} | H / \delta)}{n}.
$$

Note that we have taken an extra union bound over $H$ so that the MLE succeeds at each layer.

This is easily seen to imply an error bound between the associated Bellman operators.

Corollary F.3. Let $[\widehat{T}f](x,a)=\mathbb{E}_{(r,x^{\prime})\sim\widehat{M}(\cdot|x,a)}[r+\max_{a^{\prime}}f(x^{\prime},a^{\prime})]$ denote the Bellman optimality operator of $\widehat{M}$ , and T denote the Bellman optimality operator of $M^{\star}$ . Then we have that for all $h\in[H]$ and for all $f:X\times A\to[0,1]$ ,

$$
\mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left([ \widehat {\mathcal {T}} _ {h} f ] (x _ {h}, a _ {h}) - [ \mathcal {T} _ {h} f ] (x _ {h}, a _ {h})\right) ^ {2} \right] \leq 8 \frac {\log (| \mathcal {M} | H / \delta)}{n}.
$$

Proof of Corollary F.3. Notice that

$$
\begin{array}{l} [ \widehat {\mathcal {T}} _ {h} f ] (x, a) - [ \mathcal {T} _ {h} f ] (x, a) = \mathbb {E} _ {(r, x ^ {\prime}) \sim \widehat {M} (x, a)} \Big [ r + \max _ {a ^ {\prime}} f (x ^ {\prime}, a ^ {\prime}) \Big ] - \mathbb {E} _ {(r, x ^ {\prime}) \sim M ^ {\star} (x, a)} \Big [ r + \max _ {a ^ {\prime}} f (x ^ {\prime}, a ^ {\prime}) \Big ] \\ \leq 2 \left\| \widehat {M} _ {h} (\cdot | x, a) - M _ {h} ^ {\star} (\cdot | x, a) \right\| _ {\mathrm{tv}}. \\ \end{array}
$$

Theorem F.5. The model-based MLE algorithm is CC-bounded under Assumption F.3 for all $\gamma > 0$ simultaneously, with scaling functions $a_{\gamma} = \frac{6}{\gamma}$ and $b_{\gamma} = 8 \log(|M|H/\delta)\gamma$ . As a consequence, when invoked within $H_{2}O$ , we have Risk $\leq \widetilde{O}\left(H \sqrt{(C_{\star} + C_{\mathrm{cov}}) \log(|M|H\delta^{-1})/T}\right)$ with probability at least $1 - \delta T$ .

Proof of Theorem F.5. We note that Corollary F.2 implies a squared Bellman error bound for $Q_{\widehat{M}}^{\star}$ , the optimal value function for $\widehat{M}$ , since

$$
\forall x, a: \left([ \widehat {\mathcal {T}} _ {h} Q _ {\widehat {M}, h + 1} ^ {\star} ] (x, a) - [ \mathcal {T} _ {h} Q _ {\widehat {M}, h + 1} ^ {\star} ] (x, a)\right) ^ {2} = \left(Q _ {\widehat {M}, h} ^ {\star} (x, a) - [ \mathcal {T} _ {h} Q _ {\widehat {M}, h + 1} ^ {\star} ] (x, a)\right) ^ {2},
$$

by the optimality equation for $Q_{\widehat{M}}^{\star}$ . Thus we have

$$
\forall h \in [ H ]: \mathbb {E} _ {\mu_ {h} ^ {(1: n)}} \left[ \left(Q _ {\widehat {M}, h} ^ {\star} (x _ {h}, a _ {h}) - [ \mathcal {T} _ {h} Q _ {\widehat {M}, h + 1} ^ {\star} ] (x _ {h}, a _ {h})\right) ^ {2} \right] \leq 8 \frac {\log (| \mathcal {M} | H / \delta)}{n}. \tag {57}
$$

This is enough to repeat the proof of CC-boundedness for FQI (Theorem F.3), with $Q_{\widehat{M}}^{\star}$ taking the place of the FQI solution $\widehat{f}$ . Indeed, the only algorithmic property that we used for FQI was Lemma F.8, which also holds for $Q_{\widehat{M}}^{\star}$ by Equation (57). Tracking the slightly different constants resulting gives us the desired values for $a_{\gamma}$ and $b_{\gamma}$ .