# Combinatorial Reinforcement Learning with Preference Feedback

Joongkyu Lee $^{1}$ Min-hwan Oh $^{1}$

# Abstract

In this paper, we consider combinatorial reinforcement learning with preference feedback, where a learning agent sequentially offers an action—an assortment of multiple items—to a user, whose preference feedback follows a multinomial logistic (MNL) model. This framework allows us to model real-world scenarios, particularly those involving long-term user engagement, such as in recommender systems and online advertising. However, this framework faces two main challenges: (1) the unknown value of each item, unlike traditional MNL bandits that only address single-step preference feedback, and (2) the difficulty of ensuring optimism while maintaining tractable assortment selection in the combinatorial action space with unknown values. In this paper, we assume a contextual MNL preference model, where the mean utilities are linear, and the value of each item is approximated by a general function. We propose an algorithm, MNL-VQL, that addresses these challenges, making it both computationally and statistically efficient. As a special case, for linear MDPs (with the MNL preference feedback), we establish the first regret lower bound in this framework and show that MNL-VQL achieves nearly minimax-optimal regret. To the best of our knowledge, this is the first work to provide statistical guarantees in combinatorial RL with preference feedback.

# 1. Introduction

We first formally state the concept of Combinatorial Reinforcement Learning (RL), which we refer to as a class of RL problems where the action space is combinatorial, meaning that the agent selects a combination or subset of base actions from a set of possible base actions. Although $^{1}$ Seoul National University, Seoul, Korea. Correspondence to: Min-hwan Oh <minoh@snu.ac.kr>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

some previous studies have addressed problems within this setting—particularly in deep RL (Sunehag et al., 2015; He et al., 2016; Swaminathan et al., 2017; Metz et al., 2017; Ryu et al., 2019; Ie et al., 2019; Delarue et al., 2020; McInerney et al., 2020; Vlassis et al., 2021; Chaudhari et al., 2024), with less emphasis on theoretical RL—to the best of our knowledge, it appears that no prior work has formally and theoretically defined the concept of combinatorial RL. $^{1}$ This framework is especially relevant for real-world applications such as recommender systems and online advertising, where multiple items (base actions) must be selected simultaneously, such as a set of products to recommend or advertisements to display. The challenge in combinatorial RL lies in the exponentially large action space and the need to efficiently optimize the agent’s action selection while balancing exploration and exploitation (a challenge even for single action selection), while considering the long-term effects of these actions.

One of the most widely encountered settings in combinatorial RL is preference feedback over combinatorial actions, commonly seen in streaming services, online retail, and similar platforms. Despite the broad applicability of this setting, theoretical studies have predominantly focused on the multinomial logistic (MNL) bandit model (Rusmevichientong et al., 2010; Sauré & Zeevi, 2013; Agrawal et al., 2017; 2019; Oh & Iyengar, 2019; 2021; Perivier & Goyal, 2022; Agrawal et al., 2023; Zhang & Sugiyama, 2024; Lee & Oh, 2024). The MNL bandit framework focuses on assortment (a set of items) selection by selecting subsets of items and receiving feedback on chosen items, modeled by the MNL model (McFadden, 1977). However, these studies take a myopic approach, optimizing for immediate, known rewards without considering the long-term impact on user behavior.

While MNL bandits have been widely studied, the myopic approach is limiting in many real-world scenarios. For example in recommender systems, incorporating the long-term impact of recommendations opens the door to balancing short-term engagement with long-term user satisfaction. For instance, recommending junk product or content might lead to high immediate reward but it can decrease user satis-

faction over time due to fatigue. This trade-off between immediate and long-term outcomes is not captured by traditional MNL bandit. See Appendix A for a more details.

On the empirical side, several studies have explored long-term user engagement in recommendation systems, particularly using deep RL (Swaminathan et al., 2017; Ie et al., 2019; McInerney et al., 2020; Vlassis et al., 2021; Chaudhari et al., 2024). However, there is a significant gap in the theoretical understanding of combinatorial actions with preference feedback, particularly within the RL framework. To the best of our knowledge, no theoretical work has yet explored this important problem setting.

In this paper, we aim to address this gap by rigorously studying combinatorial RL with preference feedback and developing a provably efficient algorithm that maximizes long-term user engagement by incorporating state transitions (e.g., historical behavior) into decision-making. We consider a setting where the Q-function of an assortment is decomposed into two components: the preference model and the item values, inspired by Ie et al. (2019). Specifically, we focus on the contextual MNL preference model with linear mean utilities (Agrawal & Goyal, 2013; Cheung & Simchi-Levi, 2017; Agrawal et al., 2019; Oh & Iyengar, 2019; 2021; Amani & Thrampoulidis, 2021; Perivier & Goyal, 2022; Agrawal et al., 2023; Zhang & Sugiyama, 2024; Lee & Oh, 2024) and use general function approximation to estimate item values (Jiang et al., 2017; Wang et al., 2020; Jin et al., 2021; Du et al., 2021; Foster et al., 2021; Agarwal et al., 2023; Zhao et al., 2023). The key challenges in this framework are: (1) the unknown long-term value of each item due to the stochastic nature of rewards and transitions, (2) the difficulty of selecting an assortment that ensures optimism while considering tractable assortment optimization in the combinatorial action space, given the unknown values, and (3) achieving a tighter regret bound (for MNL preference model) than simply summing over H MNL bandit regrets.

Technical novelties. To tackle challenge (1), we estimate the optimistic item values by employing point-wise optimism under general function approximation (Agarwal et al., 2023). Based on these optimistic item values (which incorporate uncertainty), we then select an assortment that ensures sufficient exploration and guarantees optimism. This step is the most challenging part of our framework. Since the true value of each item is unknown, directly applying techniques from MNL bandits (which assume known true values) is not feasible. Thus, to address challenge (2), we propose a novel method to estimate the optimistic preference model by carefully alternating between optimistic and pessimistic utilities (for the preference model), using the optimistic item values (Equation (7)). Additionally, proving optimism (Lemma D.15) and other related results (Lem mas D.5, D.9, and D.13) requires fundamentally more sophisticated analytical techniques. Finally, we avoid naive combinatorial enumeration when selecting the assortment (Equation (8)) by reformulating the optimization problem as a linear program (LP), inspired by Davis et al. (2013). Finally, for challenge (3), instead of naively summing over H MNL bandit regrets, we bound the regret of the MNL preference model in terms of the sum of the variances of value functions and apply the law of total variance (Lattimore & Hutter, 2012; Gheshlaghi Azar et al., 2013). This approach reduces MNL regret by a factor of $\sqrt{H}$ compared to directly summing H MNL bandit regrets. Moreover, in the special case of linear MDPs with preference feedback, we achieve a nearly minimax-optimal regret.

Our main contributions are summarized as follows:

- We propose a MNL-VQL, which achieves a regret upper bound of $\tilde{\mathcal{O}}\big(d\sqrt{HK} +\sqrt{\dim(\mathcal{F})KH\log|\mathcal{F}|}\big)$ while maintaining computational efficiency (Theorem 5.1). Here, $H$ is the horizon length, $K$ is the total number of episodes, $d$ is the feature dimension of the MNL preference model, and $\dim (\mathcal{F})$ is the generalized Eluder dimension (see Definition 3.5) of the function class $\mathcal{F}$ . To the best of our knowledge, this is the first theoretical regret guarantee in combinatorial RL with preference feedback.   
- For the special case of linear MDPs (with preference feedback), MNL-VQL obtains a regret upper bound of $\tilde{\mathcal{O}}(d\sqrt{HK} + d^{\mathrm{lin}}\sqrt{HK})$ , where $d^{\mathrm{lin}}$ is the feature dimension of the linear MDPs (Theorem 5.2). Furthermore, we establish a matching regret lower bound of $\Omega(d\sqrt{HK} + d^{\mathrm{lin}}\sqrt{HK})$ , show the minimax-optimality of our algorithm in linear MDPs (Theorem 5.3).

# 2. Related Work

MNL bandits. The MNL bandits were initially studied in Rusmevichientong et al. (2010), followed by a line of improvements (Filippi et al., 2010; Rusmevichientong et al., 2010; Agrawal et al., 2017; Oh & Iyengar, 2019; Faury et al., 2020; Abeille et al., 2021; Faury et al., 2022; Oh & Iyengar, 2021; Perivier & Goyal, 2022; Agrawal et al., 2023; Lee & Oh, 2024). In MNL bandits, the goal is to offer an assortment that maximizes the expected rewards, which are adaptively learned based on user preference feedback from the offered assortment. However, there are no state transitions, and it is assumed that the value of each item is known, with the value of the outside option fixed at zero. Our study extends this by not only estimating the MNL model but also the long-term item values.

Combinatorial RL with preference feedback. Recently, several studies have demonstrated the empirical success of combinatorial RL with preference feedback (Swaminathan

et al., 2017; Ie et al., 2019; McInerney et al., 2020; Vlassis et al., 2021; Chaudhari et al., 2024), where a set of items is offered to a user, and (relative) choice feedback along with a reward is received, leading to a transition to the next state. However, theoretical results quantifying the benefits of such methods are still few and far between. A closely related work is cascading RL (Du et al., 2024), which also involves selecting a set of items. However, in cascading RL, items are offered to the user one by one, and the user decides only whether to choose the currently offered item. As a result, this framework does not capture relative preference feedback across multiple items. Furthermore, in cascading RL, the probability of choosing each item is independent of the others, which is not the case in our framework.

Another related line of work is preference-based RL (PbRL) (Akrour et al., 2012; Wirth et al., 2017; Christiano et al., 2017; Ouyang et al., 2022; Saha et al., 2023; Zhu et al., 2023; Zhan et al., 2023), where the policy is optimized based on relative, rather than absolute, preference feedback. However, our framework differs from PbRL in that our goal is not to offer just a single item, but to offer multiple items (a combinatorial base action).

RL with non-linear function approximation. With the limitations of the linear models (e.g., as shown in Lee & Oh (2023)), RL under non-linear function approximation has gained attention (Jiang et al., 2017; Wang et al., 2020; Jin et al., 2021; Du et al., 2021; Foster et al., 2021; Ishfaq et al., 2021; Agarwal et al., 2023; Zhao et al., 2023) for modeling complex function spaces like neural networks. Among these, Agarwal et al. (2023); Zhao et al. (2023) achieved the best-known regret guarantees under general function approximation by introducing the concept of generalized Eluder dimension to handle weighted regression. Inspired by their work, we estimate the value of items (referred to as item-level Q-value) using general function approximation in this paper.

# 3. Problem Setting

# 3.1. Combinatorial MDPs with Preference Feedback

In this paper, we consider a episodic combinatorial Markov decision processes (MDPs) with preference feedback, $\mathcal{M}(\mathcal{S},\mathcal{I},\mathcal{A},M,\{\mathcal{P}_{h}\}_{h=1}^{H},\{\mathbb{P}_{h}\}_{h=1}^{H},\{r_{h}\}_{h=1}^{H},H)$ . Here, S is the set of states. Each state $s \in S$ reflects the user's status, capturing both relatively static user features (e.g., demographics, interests) and relevant user history or past behavior (e.g., past recommendations, items purchased or clicked). $I := \{a_{1},\ldots,a_{N},a_{0}\}$ is the ground set of items (base actions), where $a_{1},\ldots,a_{N}$ are items and $a_{0}$ refers to the “outside option”, meaning the user has chosen none of the items from the offered set of items (referred to as an “assortment” throughout the paper). It is included in every assortment by default. $\mathcal{A}$ is the set of candidate assortments that always include the outside option $a_0$ , contain at least one item (other than $a_0$ ), and have at most $M$ items (including $a_0$ ), i.e., $\mathcal{A} = \{A \subseteq \mathcal{I} : a_0 \in A, 1 \leqslant |A \backslash \{a_0\}| \leqslant M\}$ , where $A$ is an assortment. For any $(s, A) \in \mathcal{S} \times \mathcal{A}$ , we denote $\mathcal{P}_h(a|s, A)$ as the probability of the user choosing on item $a \in A$ (including the outside option $a_0$ ). Furthermore, we let $\mathbb{P}_h : \mathcal{S} \times \mathcal{I} \to \Delta_{\mathcal{S}}$ and $r_h : \mathcal{S} \times \mathcal{I} \times \mathcal{S} \to \mathbb{R}$ characterize the transition kernel and instantaneous reward, respectively, at a given horizon $h \in [H]$ . Throughout this paper, we assume that $\sum_{h=1}^{H} r_h(s_h, a_h, s_{h+1}) \in [0, 1]$ for all possible sequence $(s_1, a_1, \ldots, s_H, a_H, s_{H+1})$ . $H \in \mathbb{Z}_+$ is the length of each episode. A policy $\pi : \mathcal{S} \to \mathcal{A}$ is a mapping from the state space to the assortment space. Since the optimal policy is non-stationary in an episodic MDP, we use $\pi$ to refer to the $H$ -tuple $\{\pi_h\}_{h=1}^H$ .

In each episode $k \in [K]$ , an initial state $s_{1}^{k}$ is picked arbitrarily (e.g., a user arrives at the system). The agent then follows a policy $\pi^{k}$ starting from $s_{1}^{k}$ . At each step $h \in [H]$ , the agent observes the current state $s_{h}^{k}$ (e.g., historical behaviors of the user) and offers an assortment $A_{h}^{k} = \pi_{h}^{k}(s_{h}^{k})$ . The user's preference feedback $a_{h}^{k} \in A_{h}^{k}$ is then observed, which is drawn based on the choice probability $\mathcal{P}_{h}(\cdot | s_{h}^{k}, A_{h}^{k})$ . Next, the system transitions to the next state $s_{h+1}^{k} \sim \mathbb{P}_{h}(\cdot | s_{h}^{k}, a_{h}^{k})$ and receives a reward $r_{h}(s_{h}^{k}, a_{h}^{k}, s_{h+1}^{k})$ . After H steps, the episode terminates, and the agent proceeds to the next.

For any policy $\pi = \{\pi_h\}_{h=1}^H$ , we define the value function of policy $\pi$ , denoted as $V_h^\pi : S \to \mathbb{R}$ , as the expected sum of rewards under the policy $\pi$ until the end of the episode when starting from $s_h = s$ , i.e., $V_h^\pi(s) := \mathbb{E}\left[\sum_{h'=h}^H r_{h'}(s_{h'}, a_{h'}, s_{h'+1}) | s_h = s\right]$ . Moreover, we define the action-value function of policy $\pi$ , $Q_h^\pi : S \times \mathcal{A} \to \mathbb{R}$ , as the expected sum of rewards under policy $\pi$ , starting from step $h$ until the end of the episode after taking action $A$ in state $s$ ; that is, $Q_h^\pi(s, A) := \mathbb{E}\left[\sum_{h'=h}^H r_{h'}(s_{h'}, a_{h'}, s_{h'+1}) | s_h = s, A_h = A\right]$ . Furthermore, we define the item-level $Q$ -value function (also called the $\overline{Q}$ -value) $\overline{Q}_h^\pi(s, a) := \sum_{s'} \mathbb{P}_h(s'|s, a)(r_h(s, a, s') + V_{h+1}^\pi(s'))$ . Then, the Bellman equation for assortment RL is denoted as follows:

$$
Q _ {h} ^ {\pi} (s, A) = \sum_ {a \in A} \mathcal {P} _ {h} (a | s, A) \overline {{Q}} _ {h} ^ {\pi} (s, a).
$$

Similarly, we define the optimal value function $V_h^\star(s) = \sup_\pi V_h^\pi(s)$ and the optimal $Q$ -value function as $Q_h^\star(s, A) = \sum_{a \in A} \mathcal{P}_h(a|s, A) \overline{Q}_h^\star(s, a)$ , where $\overline{Q}^\star := \sum_{s'} \mathbb{P}_h(s'|s, a)(r_h(s, a, s') + V_{h+1}'(s'))$ is the item-level optimal $Q$ -value function. For any $V: S \to \mathbb{R}$ and $h \in [H]$ , we define the item-level Bellman operator of $V$ as $T_hV: S \times I \to R$ , such that for all $(s, a) \in S \times I$ , $T_hV(s, a) := \mathbb{E}_{s' \sim \mathbb{P}_h(\cdot | s, a)}[r_h(s, a, s') + V(s') | s, a]$ . The definition of value functions ensures that they satisfy

the equation $\overline{Q}_h^\star(s, a) = \mathcal{T}_h V_{h+1}^\star(s, a)$ . We also define the second moment item-level Bellman operator of $V$ as $\mathcal{T}_h^2 V: \mathcal{S} \times \mathcal{I} \to \mathbb{R}$ such that for all $(s, a) \in \mathcal{S} \times \mathcal{I}$ , $\mathcal{T}_h^2 V(s, a) := \mathbb{E}_{s' \sim \mathbb{P}_h(\cdot | s, a)}[(r_h(s, a, s') + V(s'))^2 \mid s, a]$ .

Our goal is to minimize the cumulative regret over K episodes $\mathbf{Regret}(\mathcal{M},K):=\sum_{k=1}^{K}V_{1}^{\star}(s_{1}^{k})-V_{1}^{\pi_{k}}(s_{1}^{k}).$

# 3.2. Multinomial Logistic Preference Model

In this paper, we make a structural assumption about the MDP $\mathcal{M}$ , where the user's choice probability $\{\mathcal{P}_h\}_{h=1}^H$ follows multinomial logistic (MNL) model (McFadden, 1977) parameterized by $\{\boldsymbol{\theta}_h^\star\}_{h=1}^H$ . We denote $\mathcal{P}_h(\cdot|s,A;\boldsymbol{\theta}_h^\star)$ as equivalent to $\mathcal{P}_h(\cdot|s,A)$ , explicitly showing the dependence on the parameter $\boldsymbol{\theta}_h^\star$ . Throughout the paper, we use $\mathcal{P}_h(\cdot|s,A)$ and $\mathcal{P}_h(\cdot|s,A;\boldsymbol{\theta}_h^\star)$ interchangeably.

Assumption 3.1 (MNL preference model). Let there exist a known feature map $\phi : \mathcal{S} \times \mathcal{I} \to \mathbb{R}^d$ such that $\| \phi(s, a) \|_2 \leqslant 1$ , and an unknown $\boldsymbol{\theta}_h^\star \in \Theta$ for all $h \in [H]$ , where $\Theta = \left\{\boldsymbol{\theta} \in \mathbb{R}^d : \| \boldsymbol{\theta} \|_2 \leqslant B = \mathcal{O}(1)\right\}$ . Then, for any $(s, A, H) \in \mathcal{S} \times \mathcal{A} \times [H]$ , the probability of choosing any item $a \in A$ is defined as:

$$
\mathcal {P} _ {h} (a | s, A) = \mathcal {P} _ {h} (a | s, A; \boldsymbol {\theta} _ {h} ^ {\star}) = \frac {\exp \left(\phi (s , a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}{\sum_ {a ^ {\prime} \in A} \exp \left(\phi (s , a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}.
$$

Here, without loss of generality, we assume that $\phi(s, a_0) = 0$ for all $s \in \mathcal{S}^2$ , which implies that $\exp(\phi(s, a_0)^\top \boldsymbol{\theta}_h^\star) = 1$ . Thus, the preference model can be equivalently expressed as $\mathcal{P}_h(a|s, A; \boldsymbol{\theta}_h^\star) = \frac{\exp\bigl(\phi(s, a)^\top \boldsymbol{\theta}_h^\star\bigr)}{1 + \sum_{a' \in A \setminus \{a_0\}} \exp\bigl(\phi(s, a')^\top \boldsymbol{\theta}_h^\star\bigr)}$ .

Following the previous MNL bandits (Oh & Iyengar, 2021; Perivier & Goyal, 2022; Zhang & Sugiyama, 2024; Lee & Oh, 2024), we also introduce the following constant:

Definition 3.2 (Problem-dependent constant). There exist $\kappa > 0$ such that, for any $A \in \mathcal{A}, a \in A \setminus \{a_0\}, h \in [H]$ , we have $\min_{\boldsymbol{\theta} \in \Theta} \mathcal{P}_h(a|s, A, \boldsymbol{\theta}) \mathcal{P}_h(a_0|s, A, \boldsymbol{\theta}) \geqslant \kappa$ .

A small $\kappa$ indicates a larger deviation from the linear model. Note that $1 / \kappa$ can be exponentially large, so it is crucial to avoid any dependency on $1 / \kappa$ in our regret bound.

# 3.3. Generalized Function Approximation for $\overline{Q}$

We estimate the item-level $Q$ -functions (referred to as $\overline{Q}$ -values) using general function approximation. Specifically, the agent is given a function class $\mathcal{F} := \{\mathcal{F}_h\}_{h=1}^H$ , where each set $\mathcal{F}_h$ is composed of functions $f_h : \mathcal{S} \times \mathcal{I} \to [0, L]$ ,

where $L = \mathcal{O}(1)$ . Since no reward is collected in the $(H + 1)^{\text{th}}$ steps, we set $f_{H+1} = 0$ . We denote $\mathcal{N}$ as the maximal size of function class, i.e., $\mathcal{N} = \max_{h \in [H]} |\mathcal{F}_h|$ . We assume the completeness and realizability for $\mathcal{F}$ .

Assumption 3.3 (Completeness & Realizability). For each $h \in [H]$ and any $V: \mathcal{S} \to [0,1]$ , we assume that $\overline{Q}_{h}^{\star} \in \mathcal{F}_{h}$ , and there exists $f_{h}, f_{h}^{\prime} \in \mathcal{F}_{h}$ such that, for all $(s, a) \in \mathcal{S} \times \mathcal{I}$ ,

$$
f _ {h} (s, a) = \mathcal {T} _ {h} V (s, a), \quad \text { and } \quad f _ {h} ^ {\prime} (s, a) = \mathcal {T} _ {h} ^ {2} V (s, a).
$$

Remark 3.4. The completeness and realizability assumptions are standard in RL with general function approximation (Wang et al., 2021; Jin et al., 2021; Agarwal et al., 2023; Zhao et al., 2023). Our assumption is the same as those in Agarwal et al. (2023); Zhao et al. (2023), but stronger than those in Wang et al. (2021); Jin et al. (2021), especially, in terms of the second moment completeness. However, this assumption is essential for using point-wise exploration bonuses and achieving a tighter regret bound. Additionally, it naturally holds for both tabular and linear MDPs.

To capture the complexity of exploration in the MDP, we define the generalized Eluder dimension, which is a weighted regression version of the original definition (Russo & Van Roy, 2013).

Definition 3.5 (Generalized Eluder dimension, Agarwal et al. 2023). Let $\rho > 0$ , a sequence of state-item pairs $\mathbf{Z}_k = \{z^\tau\}_{\tau=1}^k$ , where $z^\tau = (s^\tau, a^\tau)$ , and a sequence of positive numbers $\sigma_k = \{\sigma^\tau\}_{\tau=1}^k$ . The generalized Eluder dimension of a function class $\mathcal{F}: \mathcal{S} \times \mathcal{I} \to [0, L]$ with respect to $\rho$ is defined as $\dim_{\nu, K}(\mathcal{F}) := \sum_{\mathbf{Z}_K, \sigma_K: \sigma \geqslant \nu} \dim(\mathcal{F}, \mathbf{Z}, \sigma)$ , where

$$
\dim (\mathcal {F}, \mathbf {Z} _ {K}, \boldsymbol {\sigma} _ {K}) := \sum_ {k = 1} ^ {K} \min \left(1, \frac {D _ {\mathcal {F}} ^ {2} \left(z ^ {k} ; \mathbf {Z} _ {k - 1} , \boldsymbol {\sigma} _ {k - 1}\right)}{\left(\sigma^ {k}\right) ^ {2}}\right).
$$

$$
D _ {\mathcal {F}} ^ {2} \left(z ^ {k}; \mathbf {Z} _ {k - 1}, \boldsymbol {\sigma} _ {k - 1}\right) = \sup _ {f _ {1}, f _ {2}} \frac {\left(f _ {1} (z) - f _ {2} (z)\right) ^ {2}}{\sum_ {\tau = 1} ^ {k - 1} \frac {\left(f _ {1} (z ^ {\tau}) - f _ {2} (z ^ {\tau})\right) ^ {2}}{(\sigma^ {\tau}) ^ {2}} + \rho}.
$$

We write $d_{\nu}:= \frac{1}{H}\sum_{h=1}^{H}\dim_{\nu,K}(\mathcal{F}_{h})$ for simplicity.

By Theorem 4.6 of Zhao et al. (2023), the generalized Eluder dimension is upper bounded by the standard Eluder dimension (Russo & Van Roy, 2013) up to logarithmic terms.

# 4. Algorithm

In this section, we introduce an algorithm, which, to the best of our knowledge, is the first to provide statistical guarantees in combinatorial RL with preference feedback while maintaining computational tractability. Step 1 involves online parameter estimation for the MNL preference model, proposed by Lee & Oh (2024). Steps 2, 3, and 4 implement variance-weighted regression to tighten the regret bound, as outlined in Agarwal et al. (2023). Step 5, which is our main

contribution, ensures optimism in a computationally efficient manner, even with uncertainty in item-level Q-values. Step 6 introduces an exploration step that accounts for estimation errors from the MNL preference model.

Step 1. Online parameter estimation for MNL (Line 6). At episode $k$ and horizon $h$ , given the user's choice feedback $c_h^k \in A_h^k$ , the response for each item $a_{i_m} \in A_h^k$ is defined as $y_h^k(a_{i_m}) := \mathbb{1}(c_h^k = a_{i_m}) \in \{0,1\}$ . Therefore, the response variable $\mathbf{y}_h^k := (y_h^k(a_0), y_h^k(a_{i_1}), \ldots, y_h^k(a_{i_l}))$ , where $l \leqslant M - 1$ , is sampled from a multinomial distribution: $\mathbf{y}_h^k \sim \mathrm{MNL}\{1, \mathcal{P}_h(a_0 | s_h^k, A_h^k; \boldsymbol{\theta}_h^\star), \ldots, \mathcal{P}_h(a_l | s_h^k, A_h^k; \boldsymbol{\theta}_h^\star)\}$ , where the parameter 1 indicates that $\mathbf{y}_h^k$ is a single-trial sample, i.e., $y_h^k(a_0) + \sum_{m=1}^{l} y_h^k(a_{i_m}) = 1$ . Then, for any $(k,h) \in [K] \times [H]$ , the multinomial logistic loss function is defined as:

$$
\ell_ {h} ^ {k} (\pmb {\theta}) := - \sum_ {a \in A _ {h} ^ {k}} y _ {h} ^ {k} (a) \log \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \pmb {\theta}).
$$

Inspired by Zhang & Sugiyama (2024); Lee & Oh (2024), for all $(k, h) \in [K] \times [H]$ , we use the online mirror descent algorithm to estimate the true parameter $\theta_h^\star$ as follows:

$$
\boldsymbol {\theta} _ {h} ^ {k + 1} \in \underset {\boldsymbol {\theta} \in \Theta} {\operatorname{argmin}} \left\langle \nabla \ell_ {h} ^ {k} \left(\boldsymbol {\theta} _ {h} ^ {k}\right), \boldsymbol {\theta} \right\rangle + \frac {1}{2 \eta} \| \boldsymbol {\theta} - \boldsymbol {\theta} _ {h} ^ {k} \| _ {\tilde {\mathbf {H}} _ {h} ^ {k}} ^ {2}, \tag {1}
$$

where $\eta = \mathcal{O}(\log M)$ is the step-size, $\tilde{\mathbf{H}}_h^k := \mathbf{H}_h^k + \eta \nabla^2 \ell_h^k (\boldsymbol{\theta}_h^k)$ , and $\mathbf{H}_h^k := \lambda \mathbf{I}_d + \sum_{\tau=1}^{k-1} \nabla^2 \ell_h^\tau (\boldsymbol{\theta}_h^{\tau+1})$ .

Remark 4.1. The computation cost of the optimization problem in (1) is $\mathcal{O}(Md^{3})$ , which dose not scale with k at all.

Then, with high probability, $\theta_{h}^{\star}$ lies within the following confidence interval (Corollary D.2).

$$
\mathcal {C} _ {h} ^ {k} := \left\{\boldsymbol {\theta} \in \Theta : \left\| \boldsymbol {\theta} - \boldsymbol {\theta} _ {h} ^ {k} \right\| _ {\mathbf {H} _ {h} ^ {k}} \leqslant \alpha_ {h} ^ {k} = \tilde {\mathcal {O}} (\sqrt {d}) \right\}. \tag {2}
$$

Step 2. Weighted regression and optimistic estimation for $\overline{Q}$ (Line 7-9). Using the past dataset, we solve the following (weighted) regression problem to fit $T_{h}V_{h+1}^{k}$ :

$$
\hat {f} _ {h, 1} ^ {k} \in \underset {f _ {h} \in \mathcal {F} _ {h}} {\operatorname{argmin}} \sum_ {\tau = 1} ^ {k - 1} \frac {\left(f _ {h} (s _ {h} ^ {\tau} , a _ {h} ^ {\tau}) - r _ {h} ^ {\tau} - V _ {h + 1 , 1} ^ {k} (s _ {h + 1} ^ {\tau})\right) ^ {2}}{\left(\bar {\sigma} _ {h} ^ {\tau}\right) ^ {2}}, \tag {3}
$$

where $(\bar{\sigma}_{h}^{k})^{2}$ is a variance upper bound, i.e., $(\bar{\sigma}_{h}^{k})^{2} \geqslant \mathbb{V}[r_{h} + V_{h+1,1}^{k}(s_{h+1})|s_{h}^{k}, a_{h}^{k}]$ , which will be specified later. Then, an optimistic $\overline{Q}$ -value estimate at horizon h is defined as $f_{h,1}^{k} := \hat{f}_{h,1}^{k} + b_{h}^{k}$ , where $b_{h}^{k}$ is the optimistic bonus. The bonus is calculated as $b_{h}^{k}(s, a) = \max_{f_{h} \in \mathcal{F}_{h}} f_{h}(s, a) - \min_{f_{h} \in \mathcal{F}_{h}} f_{h}(s, a)$ . In general, this uncertainty bonus has a high complexity, as the maximizing and minimizing functions can differ arbitrarily for each $(s, a) \in \mathcal{S} \times \mathcal{I}$ . To address this, we use a low-complexity bonus oracle B (Agarwal et al., 2023) that approximately dominates the value obtained from the point-wise maximization over $F_{h}$ . With the oracle B, we can efficiently calculate the bonus $b_{h}^{k}$ , with an error of $\epsilon_{b}$ . Due to space constraints, we provide the formal definition of B in Appendix B (Definition B.1).

Step 3. Overly optimistic/pessimistic estimation for $\overline{Q}$ (Line 10-13). For a sharp analysis of the convergence of the optimistic estimate $f_{h,1}^{k}$ , we define an overly optimistic $\overline{Q}$ -value estimate $f_{h,2}^{k}$ , as well as an overly pessimistic $\overline{Q}$ -value estimate $f_{h,-2}^{k}$ . Similarly to $f_{h,1}^{k}$ , they are calculated by solving an unweighted regression problem (Line 10), and by adding (or subtracting) a bonus function, which is the output of the bonus oracle B (Line 12-13).

Step 4. Variance estimation (Line 14 and 21). To calculate $\bar{\sigma}_{h}^{k}$ introduced in (3), we first estimate the second moment by solving the unweighted regression problem:

$$
\hat {g} _ {h} ^ {k} \in \underset {g _ {h} \in \mathcal {F} _ {h}} {\operatorname{argmin}} \sum_ {\tau = 1} ^ {k - 1} \left(g _ {h} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau}) - \left(r _ {h} ^ {\tau} + V _ {h + 1, 1} ^ {k} (s _ {h + 1} ^ {\tau})\right) ^ {2}\right) ^ {2}.
$$

Then, denoting $z_{h}^{k} = (s_{h}^{k}, a_{h}^{k})$ for simplicity, we calculate the estimated variance as follows (this is an informal description; for the precise formulation, see Equations (D.9) and (D.10) in Appendix):

$$
\left(\sigma_ {h} ^ {k}\right) ^ {2} \simeq \hat {g} _ {h} ^ {k} (z _ {h} ^ {k}) - \left(\hat {f} _ {h, - 2} ^ {k} (z _ {h} ^ {k})\right) ^ {2} + \mathfrak {D} _ {k, h} ^ {\mathbf {1}} \cdot \mathcal {O} \left(\sqrt {\log \mathcal {N N} _ {b}}\right)
$$

$$
\bar {\sigma} _ {h} ^ {k} \simeq \max \left\{\sigma_ {h} ^ {k}, \mathcal {O} (\log \mathcal {N N} _ {b}) \right.
$$

$$
\left. \cdot \left(\max \left\{f _ {h, 2} ^ {k} (z _ {h} ^ {k}) - f _ {h, - 2} ^ {k} (z _ {h} ^ {k}), \mathfrak {D} _ {k, h} ^ {\sigma} \right\}\right) \right\}, \tag {4}
$$

where $\mathfrak{D}_{k,h}^{\mathbf{1}} = D_{\mathcal{F}_{h}}(z_{h}^{k};\mathbf{Z}_{k-1},\{\mathbf{1}^{\tau}\}_{\tau=1}^{k-1})$ and $\mathfrak{D}_{k,h}^{\sigma} = D_{\mathcal{F}_{h}}(z_{h}^{k};\mathbf{Z}_{k-1},\boldsymbol{\sigma}_{k-1})$ .

Step 5. Efficient optimistic Q-value estimation based on unknown item values (Line 15-16). In this step, we address our main challenge: selecting an optimistic assortment based on the optimistic $\overline{Q}$ -values, which incorporate uncertainty, while ensuring computational tractability.

To introduce optimism and encourage exploration, we need to solve the following optimization problem using the optimistic (or pessimistic) estimates of the $\overline{Q}$ -values, specifically $f_{h,j}^{k}$ for $j = 1, \pm 2$ :

$$
A _ {h, j} ^ {k} \in \underset {A \in \mathcal {A}} {\operatorname{argmax}} \underset {\boldsymbol {\theta} \in \mathcal {C} _ {h} ^ {k}} {\max} \sum_ {a \in A} \mathcal {P} _ {h} ^ {k} (a | s _ {h} ^ {k}, A; \boldsymbol {\theta}) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a), \tag {5}
$$

where $C_{h}^{k}$ is defined in (2). One naive approach to solving the optimization problem in (5) is to add bonus terms to the estimation for each assortment A and then enumerate all $A \in A$ to find the maximum. However, this approach results in a computational cost of $\mathcal{O}(|\mathcal{I}|^{M})$ .

Algorithm 1 MNL-VQL, MNL Preference Model with Variance-weighted Item-level Q-Learning   
1: Inputs: parameter space $\Theta$ , function class $\{\mathcal{F}_h\}_{h=1}^H$ , consistent bonus oracle $\mathcal{B}$ .
2: Parameters: $\{\alpha_h^k, \beta_{h,1}^k, \beta_{h,2}^k, \bar{\beta}_h^k\}_{(k,h) \in [H] \times [K]}, \{u_k\}_{k=1}^K, \rho$ , bonus error $\epsilon_b, \nu, \delta$ .
3: Initialize: dataset $\mathcal{D}_h^0 = \emptyset$ for all $h \in [H]$ .
4: Generate $\{\mathcal{D}_h^1\}_{h=1}^H$ from initial state $s_1^1$ by random policy and set $\sigma_h^1 = \bar{\sigma}_h^1 = 2$ for all $h \in [H]$ .
5: for episode $k = 2, \cdots, K$ do
6:    for horizon $h = H, H - 1, \ldots, 1$ do
    // CACULATE OPTIMISTIC AND OVERLY OPTIMISTIC, PESSIMISTIC $\overline{Q}$ -VALUES
7: $\hat{f}_{h,1}^k \in \text{argmin}_{f_h \in \mathcal{F}_h} \sum_{\tau=1}^{k-1} \frac{1}{(\bar{\sigma}_h^\tau)^2} \left( f_h(s_h^\tau, a_h^\tau) - r_h^\tau - V_{h+1,1}^k(s_{h+1}^\tau) \right)^2$ .
8: $b_{h,1}^k \leftarrow \mathcal{B} \left( \{\bar{\sigma}_h^\tau\}_{\tau=1}^{k-1}, \mathcal{D}_h^{k-1}, \mathcal{F}_h, \hat{f}_{h,1}^k, \beta_{h,1}^k, \rho, \epsilon_b \right)$ (see Definition B.1).
9:    Update $f_{h,1}^k(\cdot, \cdot) \leftarrow \min \left\{ \hat{f}_{h,1}^k(\cdot, \cdot) + b_{h,1}^k(\cdot, \cdot), 1 \right\}$ .
10: $\hat{f}_{h,j}^k \in \text{argmin}_{f_h \in \mathcal{F}_h} \sum_{\tau=1}^{k-1} \left( f_h(s_h^\tau, a_h^\tau) - r_h^\tau - V_{h+1,j}^k(s_{h+1}^\tau) \right)^2, j = \pm 2$ .
11: $b_{h,2}^k \leftarrow \mathcal{B} \left( \{\mathbf{1}^\tau\}_{\tau=1}^{k-1}, \mathcal{D}_h^{k-1}, \mathcal{F}_h, \hat{f}_{h,2}^k, \beta_{h,2}^k, \rho, \epsilon_b \right)$ (see Definition B.1).
12:    Update $f_{h,2}^k(\cdot, \cdot) \leftarrow \min \left\{ \hat{f}_{h,2}^k(\cdot, \cdot) + 2b_{h,1}^k(\cdot, \cdot) + b_{h,2}^k(\cdot, \cdot), 1 \right\}$ .
13:    Update $f_{h,-2}^k(\cdot, \cdot) \leftarrow \max \left\{ \hat{f}_{h,-2}^k(\cdot, \cdot) - b_{h,2}^k(\cdot, \cdot), 0 \right\}$ .
14: $\hat{g}_h^k \in \text{argmin}_{g_h \in \mathcal{F}_h} \sum_{\tau=1}^{k-1} \left( g_h(s_h^\tau, a_h^\tau) - (r_h^\tau + V_{h+1,1}^k(s_{h+1}^\tau))^2 \right)^2$ .
    // UPDATE VALUES
15:    Update $\widetilde{\mathcal{P}}_{h,j}^k(\cdot|\cdot,\cdot)$ by (7).
16:    Update $Q_{h,j}^k(\cdot,A) \leftarrow \sum_{a\in A} \widetilde{\mathcal{P}}_{h,j}^k(a|\cdot,A)f_{h,j}^k(\cdot,a)$ and $V_{h,j}^k(\cdot) \leftarrow \max_{A\in A} Q_{h,j}^k(\cdot,A), j = 1,\pm 2$ .
17: end for
18: Receive initial state $s_1^k$ .
19: for $h = 1,2,\ldots,H$ do
20:    Offer $A_h^k$ by (9) and receive $a_h^k, r_h^k$ , and $s_{h+1}^k$ .
21:    Update $\mathcal{D}_h^k \leftarrow \mathcal{D}_h^{k-1} \cup \{s_h^k, a_h^k, r_h^k, s_{h+1}^k\}$ and update $\sigma_h^k$ and $\bar{\sigma}_h^k$ by (4).
22:    Update $\hat{\theta}_h^{k+1}$ using online mirror descent (1).
23: end for
24: end for

To avoid this exponential computational cost, inspired by Tran-Dinh et al. (2015), we use optimistic MNL utilities instead of directly adding bonus terms to $\sum_{a\in A}\mathcal{P}_{h}^{k}(a|s_{h}^{k},A;\boldsymbol{\theta}_{h}^{k})f_{h,j}^{k}(s_{h}^{k},a)$ . However, unlike traditional MNL bandits (Oh & Iyengar, 2019; 2021; Lee & Oh, 2024), simply using optimistic utilities does not always guarantee optimism because of $f_{h,j}^{k}$ is not the true values. In MNL bandits, the item values are known, and the value of the outside option $a_{0}$ is fixed at zero. Therefore, increasing the MNL utilities (using the optimistic utilities) lowers the probability of choosing the outside option, which in turn increases the expected value of the item values.

However, in our setting, using the optimistic utilities can decrease the expected value of $f_{h,j}^{k}$ . To explain why: even if the true value of the outside option, $\overline{Q}_{h}^{\star}(s,a_{0})$ , is the lowest, its estimated value, $f_{h,j}^{k}(s,a_{0})$ , can be the highest—i.e., $f_{h,j}^{k}(s,a_{0}) > f_{h,j}^{k}(s,a)$ for all $a \in I \setminus \{a_{0}\}$ —due to uncertainty. In this case, increasing the MNL utilities results in a decrease in the expected value of $f_{h,j}^{k}$ . This challenge arises from the unknown item values, $\overline{Q}_{h}^{\star}$ , which is one of the main difficulties we face in our framework.

To tackle this problem, a more refined approach is required to use the utility based on $f_{h,j}^{k}$ . Given the confidence interval in (2), we define the optimistic utility $\tilde{v}_{h}^{k}(s,a)$ and the pessimistic utility $\check{v}_{h}^{k}(s,a)$ as:

$$
\begin{array}{l} \widetilde {v} _ {h} ^ {k} (s, a) := \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {k} + \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}, \\ \check {v} _ {h} ^ {k} (s, a) := \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {k} - \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}. \tag {6} \\ \end{array}
$$

We then use the optimistic utility when $f_{h,j}^{k}(s,a_{0})$ is not the highest estimate to calculate the optimistic choice probabilities $\widetilde{P}_{h,j}^{k}$ . Formally, let $I_{h,j}^{k}\in\{1,0\}$ indicate whether there exists $a\in\mathcal{I}\backslash\{a_{0}\}$ such that $f_{h,j}^{k}(s,a)\geqslant f_{h,j}^{k}(s,a_{0})$

$(I_{h,j}^{k} = 1$ if such an event occurs). Then, we define

$$
\widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, A) := \left\{ \begin{array}{l l} & \frac {\exp \left(\check {v} _ {h} ^ {k} (s , a)\right)}{\sum_ {a ^ {\prime} \in A} \exp \left(\check {v} _ {h} ^ {k} (s , a ^ {\prime})\right)}, \quad \text { if } I _ {h, j} ^ {k} = 1 \\ & \frac {\exp \left(\check {v} _ {h} ^ {k} (s , a)\right)}{\sum_ {a ^ {\prime} \in A} \exp \left(\check {v} _ {h} ^ {k} (s , a ^ {\prime})\right)}, \quad \text { otherwise }. \end{array} \right. \tag {7}
$$

Equipped with $\widetilde{\mathcal{P}}_{h,j}^k$ , for any $j = 1, \pm 2$ , we select the assortments $A_{h,j}^k$ as follows:

$$
A _ {h, j} ^ {k} \in \underset {A \in \mathcal {A}} {\operatorname{argmax}} \underbrace {\sum_ {a \in A} \widetilde {\mathcal {P}} _ {h , j} ^ {k} (a | s _ {h} ^ {k} , A) f _ {h , j} ^ {k} (s _ {h} ^ {k} , a)} _ {=: Q _ {h, j} ^ {k} (s _ {h} ^ {k}, A)}, \qquad (8)
$$

Here, $Q_{h,j}^{k}(s_{h}^{k}, A) := \sum_{a \in A} \widetilde{\mathcal{P}}_{h,j}^{k}(a | s_{h}^{k}, A) f_{h,j}^{k}(s_{h}^{k}, a)$ is the optimistic Q-values. This construction can induce sufficient exploration and guarantee optimism (Lemma D.15). Furthermore, by using the optimistic (or pessimistic) utilities for each item, instead of calculating bonus terms for each $A \in A$ , the optimization problem in (8) can be solved efficiently (Davis et al., 2013).

Remark 4.2. The optimization problem in (8) can be transformed into a linear programming (LP), making it solvable in polynomial time with respect to $|I|$ (see Appendix C).

Step 6. Exploration policy (Line 20). We then offer the assortment $A_{h}^{k}$ to the user as follows:

$$
A _ {h} ^ {k} = \left\{ \begin{array}{l l} A _ {h, 1} ^ {k} & \text { if } f _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}}) \geqslant f _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}}) - u _ {k}, \\ & \forall a _ {h ^ {\prime}} \in A _ {h ^ {\prime}, 1} ^ {k}, \forall h ^ {\prime} \leqslant h, \\ A _ {h, 2} ^ {k} & \text { otherwise }, \end{array} \right. \tag {9}
$$

where $u_{k}$ is a carefully chosen threshold (see Table D.2 for the exact value). When the optimistic sequence $f_{h,1}^{k}$ and the overly optimistic sequence $f_{h,2}^{k}$ diverge beyond a certain threshold, we offer the assortment $A_{h,2}^{k}$ , which is selected based on $f_{h,2}^{k}$ . This approach ensures that by occasionally using $f_{h,2}^{k}$ , the variance upper bound $\bar{\sigma}_{h}^{k}$ , estimated from $f_{h,2}^{k}$ , does not become overly pessimistic.

# 5. Main Results

# 5.1. Non-linear Function Approximation for $\overline{Q}$

Theorem 5.1 (Informal, Regret upper bound of MNL-VQL). Let $d_{\nu} = \frac{1}{H}\sum_{h=1}^{H}\dim_{\nu,K}(\mathcal{F}_h)$ with $\nu = \sqrt{1/KH}$ , and set $u_k$ as in Equation (D.20). Suppose Assumptions 3.1 and 3.3 hold. Then, with probability at least $1 - \delta$ , the regret of MNL-VQL is upper-bounded by:

$$
\begin{array}{l} \mathbf {R e g r e t} (\mathcal {M}, K) \lesssim \underbrace {d \sqrt {H K} + \frac {1}{\kappa} d ^ {2} H ^ {2}} _ {\text {   regret   from   MNL   model   }} \\ + \underbrace {\sqrt {d _ {\nu} H K \log \mathcal {N}} + d _ {\nu} H ^ {5} \log \mathcal {N} \log^ {2} (\mathcal {N N} _ {b})} _ {\text {regret from general function approximation of \overline {{Q}}}}, \\ \end{array}
$$

where $d$ is the feature dimension of the MNL preference model, $\mathcal{N} = \max_{h\in [H]}|\mathcal{F}_h|$ , and $\mathcal{N}_b$ is the size of the bonus function class, i.e., $\mathcal{N}_b = |\mathcal{W}|$ .

Discussion of Theorem 5.1. The proof is deferred to Appendix D. The first two terms arise from the regret of the MNL preference model, while the other two terms come from the regret associated with the general function approximation for item-level Q-values. When H = 1, reducing our setting to MNL bandits (though not exactly the traditional MNL bandits, as we consider a more general case where item values are unknown and the value of the outside option can be non-zero), the first two terms of our regret simplify to $\tilde{\mathcal{O}}(d\sqrt{K} + \frac{1}{\kappa}d^{2})$ . This matches the known minimax optimal regret established by Lee & Oh (2024). Note that we avoid the detrimental dependence on $\kappa$ in our leading term. The last two terms of our regret, incurred from estimating item-level Q-values using general function approximation, similar to Agarwal et al. (2023) and Zhao et al. (2023).

With respect to computational cost, by using the online sensitivity sub-sampling method (Algorithm B.1), we can efficiently implement the bonus oracle $\mathcal{B}$ with $\log |\mathcal{W}| = \log \mathcal{N}_b = \tilde{\mathcal{O}}\left(\max_{h\in [H]}\dim_{\nu ,K}(\mathcal{F}_h)\log \mathcal{N}\log |\mathcal{S}\times \mathcal{I}|\right)$ . Furthermore, we can avoid the exponential computational cost required to solve the optimization in (8) (see Remark 4.2). As a result, our algorithm is both computationally tractable and statistically efficient.

# 5.2. Technical Comparisons to Related Work

Comparison to Lee & Oh (2024). Our framework addresses a strictly more challenging problem than Lee & Oh (2024) because we consider (1) multiple steps, (2) unknown values, and (3) nonzero values for the outside option. The key challenge in tackling these three aspects lies in ensuring optimism while maintaining computational efficiency, which requires a fundamentally different approach—carefully leveraging optimistic and pessimistic utilities. Moreover, in Theorem 5.1, our regret analysis for the MNL preference model goes beyond merely summing over H MNL bandit regrets. Instead, we introduce a novel regret decomposition, bound the regret in terms of the sum of the variances of value functions, and apply the law of total variance (Lattimore & Hutter, 2012; Gheshlaghi Azar et al., 2013). As a result, we achieve a regret reduction by a

factor of $\sqrt{H}$ compared to directly summing over H MNL bandit regrets.

Comparison to Agarwal et al. (2023). While the regret analysis for general function approximation largely follows the approach of Agarwal et al. (2023), several technical lemmas (e.g., Lemmas D.23 and D.24) and parameters (e.g., $u_{k}$ ) are revised to accommodate the estimation errors specific to MNL models.

Overall, our results cannot be gleaned by simply piecing together prior techniques. Instead, they arise from an involved analysis, leading to stronger theoretical guarantees in more general and new combinatorial RL settings.

# 5.3. Linear MDPs with Preference Feedback

As a special case, we also consider linear MDPs (refer Definition E.1) with preference feedback. To show the dependency on parameters, we denote the linear MDPs as $\mathcal{M}_{\Xi^{\star}}$ , where $\Xi^{\star} = \{\{\boldsymbol{\theta}_h^\star\}_{h=1}^H, \{\boldsymbol{\mu}_h^\star\}_{h=1}^H, \{\mathbf{w}_h^\star\}_{h=1}^H\}$ . Note that the bonus oracle can be easily implemented using the standard elliptical bonus, which satisfies all the necessary properties (refer Appendix E). The proof is deferred to Appendix E.

Theorem 5.2 (Informal, Regret upper bound for linear MDPs). In linear MDPs, under the same conditions as Theorem 5.1, with probability at least $1 - \delta$ , the regret of MNL-VQL is upper-bounded by:

$$
\begin{array}{l} \mathbf {R e g r e t} \left(\mathcal {M} _ {\Xi^ {*}}, K\right) \lesssim d \sqrt {H K} + \frac {1}{\kappa} d ^ {2} H ^ {2} \\ + d ^ {l i n} \sqrt {H K} + (d ^ {l i n}) ^ {6} H ^ {5}. \\ \end{array}
$$

We also establish a matching lower bound by constructing a novel multi-layered (linear) MDP (see Figure F.1) with a preference feedback. The proof is deferred to Appendix F.

Theorem 5.3 (Informal, Regret lower bound for linear MDPs). For any algorithm and sufficiently large K, there exists an episodic linear MDP $M_{\Xi}$ with MNL preference feedback such that the worst-case expected regret is lower bounded as follows:

$$
\sup _ {\Xi} \mathbb {E} _ {\Xi} \left[ \operatorname{Regret} \left(\mathcal {M} _ {\Xi}, K\right) \right] = \Omega \left(d \sqrt {H K} + d ^ {l i n} \sqrt {H K}\right).
$$

Discussion of Theorems 5.2 and 5.3. For sufficiently large K, i.e., $K \geqslant \tilde{\mathcal{O}}\big(d^{2}H^{3}/\kappa^{2}+(d^{\mathrm{lin}})^{10}H^{9}\big)$ , the regret upper bound for linear MDPs scales as $\tilde{\mathcal{O}}(d\sqrt{HK} + d^{\mathrm{lin}}\sqrt{HK})$ , which matches the lower bound up to logarithmic factors. Note that if we rescale the rewards to be 1/H in the lower bounds of Zhou et al. (2021a) to align with our setting, their regret bound matches the second term of our regret bound, $\Omega(d^{\mathrm{lin}}\sqrt{HK})$ . To the best of our knowledge, this is the first theoretical result proving minimax-optimality in linear MDPs with preference feedback.

# 6. Numerical Experiment

In this section, we empirically evaluate the performance of our algorithm, MNL-VQL, in two settings: a synthetic environment (Subsection 6.1) and a real-world dataset (Subsection 6.2).

We compare our algorithm against two baselines: Myopic and LSVI-UCB (Jin et al., 2020). Myopic is a variant of OFU-MNL+ (Lee & Oh, 2024) adapted for unknown rewards. It selects assortments based only on immediate rewards, ignoring long-term effects. LSVI-UCB (Jin et al., 2020) treats each assortment as a single atomic action, requiring enumeration of all possible assortments. We also include the optimal policy (Optimal) as a reference.

# 6.1. Synthetic Environment

Setup. We consider an online shopping with budget environment (Figure G.1), modeled as a linear MDP with an MNL preference model. Let $S = \{s_{1}, \ldots, s_{|S|}\}$ denote the set of states and $I = \{a_{1}, \ldots, a_{N}, a_{0}\}$ the set of items, where $a_{0}$ represents the outside option (no purchase). Each state $s_{j} \in S$ corresponds to a user's budget level, with larger indices indicating a higher budget. The initial state is set to the middle budget level, $s_{[|S|/2]}$ . The transition probabilities $P_{h}$ , rewards $r_{h}$ , and preference model $P_{h}$ remain constant across all time steps $h \in [H]$ , so we omit the subscript h.

At state $s_{j}$ , the agent offers an assortment $A \in A$ with a maximum size of M. The user either purchases an item $a_{i} \in A$ or does not $(a_{0} \in A)$ . If the user buys item $a_{i}$ , the agent receives a reward $r(s_{j}, a_{i}) = \left(\frac{i}{100N} + \frac{j}{|S|}\right)/H$ and the state transitions according to $\mathbb{P}(s_{\min(j+1, |S|)} | s_{j}, a_{i}) = 1 - \frac{i}{N}$ , and $\mathbb{P}(s_{\max(j-1,0)} | s_{j}, a_{i}) = \frac{i}{N}$ . If the user does not buy anything $(a_{0})$ , the reward is $r(s_{j}, a_{0}) = 0$ , and the state transitions as $\mathbb{P}(s_{\min(j+1, |S|)} | s_{j}, a_{0}) = 1$ . For the MNL preference model, the true parameter $\theta^{\star} \in R^{d}$ and the feature vector $\phi(s, a) \in R^{d}$ are randomly sampled from a d-dimensional uniform distribution for each instance.

Results. Figure 1 demonstrates that our algorithm significantly outperforms other baseline algorithms. Remarkably, Myopic converges to a suboptimal solution, highlighting the importance of accounting for long-term values. Moreover, in Appendix Table G.1, we show that our algorithm is much faster than others, especially when the total number of assortments $|A|$ is large. Due to the extremely slow runtime of LSVI-UCB, we could not include its performance results for N = 20 and N = 40. For more details, see Appendix G.

# 6.2. Real-World MovieLens Experiment

Setup. The MovieLens dataset contains 25 million ratings on a 5-star scale for 62,000 movies (base items a) provided by 162,000 users (u). We define the state s as the number

![](images/824b5b6259b8ad1139e1070f23e09d77c46acf02bf392e9e0930db5e3164ad2e.jpg)

Figure 1: Synthetic experiment: Episodic returns averaged over 10 runs. Dotted lines indicate estimated returns for incomplete runs due to excessive runtime. Shading denotes $\pm1$ standard deviation.   
![](images/b7d8cd27e0fe58351994196f833610ac7d48d9da7105618d147055039085c01a.jpg)  
Figure 2: MovieLens experiment: The dotted lines represent estimated (virtual) episodic returns for cases that could not be run due to excessively long runtimes. Shaded regions represent $\pm1$ standard deviation.

of movies a user u has watched after entering the system, denoted by $s = (u, n)$ , where $n \in \{0, \ldots, H - 1\}$ is the number of movies watched during the session. We interpret the ratings as representing MNL utilities.

In each episode k, a user $(u_{k})$ is randomly sampled and arrives at the recommender system, initiating the state $s_{1}^{k} = (u_{k}, 0)$ . The agent offers a set of items with a maximum size of M. If the user clicks on an item, they receive a reward of 1 and transition to the next state $s_{2}^{k} = (u_{k}, 1)$ . If no item is clicked, the user receives no reward and remains in the current state $(s_{2}^{k} = s_{1}^{k})$ . In addition, certain junk items—such as those with a provocative title and poster but poor content—can cause users to leave the system immediately. This is modeled as a transition to an absorbing state, where no further rewards are received and the state remains unchanged regardless of future actions. We believe the presence of such junk items is quite natural and reflective of real-world recommendation environments.

For our experiments, we use a subset of the dataset containing $1.1 \times 10^{3}$ users and a varying number of movies, $N \in \{50, 100, 200\}$ . To construct MNL features, we follow a similar experimental setup as in Li et al. (2019), employing low-rank matrix factorization. For linear MDP features, we apply the same approach as used in our synthetic data experiments. We set the parameters as follows: $K = 10000, H = 3, M = 4, |\mathcal{S}| = 100 * (H + 1) = 400$ (including the absorbing state), $d = 26$ (MNL feature dimension), $d^{\mathrm{lin}} = 204$ (Linear MDP feature dimension), $N \in \{50, 100, 200\}$ (number of base items) and $|\mathcal{A}| = \sum_{m=1}^{M-1} \binom{N}{m} \in \{20875, 166750, 1333500\}$ . The proportion of junk items is set to $30\%$ .

Results. Consistent with the synthetic experiment results, Figure 2 shows that our algorithm substantially outperforms the baseline methods on the real-world dataset. This demonstrates the robustness of our approach and its practical effectiveness in realistic settings.

# 7. Conclusion

In this work, we study combinatorial RL with preference feedback, extending MNL bandit problems to account for the influence of user states and state transitions in applications like recommendation systems. Under an MNL preference model with linear utilities and general function approximation for item values, we propose an efficient algorithm, MNL-VQL, which, to the best of our knowledge, provides the first statistical guarantee. As a special case, in linear MDPs, we show the minimax-optimality of MNL-VQL by establishing matching upper and lower bounds.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# Acknowledgements

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. RS-2022-NR071853 and RS-2023-00222663) and by AI-Bio Research Grant through Seoul National University.

# References

Abeille, M., Faury, L., and Calauzènes, C. Instance-wise minimax-optimal algorithms for logistic bandits. In International Conference on Artificial Intelligence and Statistics, pp. 3691–3699. PMLR, 2021.   
Agarwal, A., Jin, Y., and Zhang, T. Vo q 1: Towards optimal regret in model-free rl with nonlinear function approximation. In The Thirty Sixth Annual Conference on Learning Theory, pp. 987–1063. PMLR, 2023.   
Agrawal, P., Tulabandhula, T., and Avadhanula, V. A tractable online learning algorithm for the multinomial logit contextual bandit. European Journal of Operational Research, 310(2):737–750, 2023.   
Agrawal, S. and Goyal, N. Thompson sampling for contextual bandits with linear payoffs. In International Conference on Machine Learning, pp. 127–135. PMLR, 2013.   
Agrawal, S., Avadhanula, V., Goyal, V., and Zeevi, A. Thompson sampling for the mnl-bandit. In Conference on learning theory, pp. 76–78. PMLR, 2017.   
Agrawal, S., Avadhanula, V., Goyal, V., and Zeevi, A. Mnl-bandit: A dynamic learning approach to assortment selection. Operations Research, 67(5):1453–1485, 2019.   
Akrour, R., Schoenauer, M., and Sebag, M. April: Active preference learning-based reinforcement learning. In Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2012, Bristol, UK, September 24-28, 2012. Proceedings, Part II 23, pp. 116–131. Springer, 2012.   
Amani, S. and Thrampoulidis, C. Ucb-based algorithms for multinomial logistic regression bandits. Advances in Neural Information Processing Systems, 34:2913–2924, 2021.

Chaudhari, S., Arbour, D., Theocharous, G., and Vlassis, N. Distributional off-policy evaluation for slate recommendations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 8265–8273, 2024.

Chen, K. D. and Hausman, W. H. Mathematical properties of the optimal product line selection problem using choice-based conjoint analysis. Management Science, 46(2):327–332, 2000.

Chen, W., Wang, Y., and Yuan, Y. Combinatorial multi-armed bandit: General framework and applications. In International conference on machine learning, pp. 151–159. PMLR, 2013.

Cheung, W. C. and Simchi-Levi, D. Thompson sampling for online personalized assortment optimization problems with multinomial logit choice models. Available at SSRN 3075658, 2017.

Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.

Combes, R., Talebi Mazraeh Shahi, M. S., Proutiere, A., et al. Combinatorial bandits revisited. Advances in neural information processing systems, 28, 2015.

Cooper, A. C. W. et al. Programming with linear fractional functionals. Naval Research logistics quarterly, 9(3):181–186, 1962.

Davis, J., Gallego, G., and Topaloglu, H. Assortment planning under the multinomial logit model with totally unimodular constraint structures. department of ieor, columbia university, 2013.

Delarue, A., Anderson, R., and Tjandraatmadja, C. Reinforcement learning with combinatorial actions: An application to vehicle routing. Advances in Neural Information Processing Systems, 33:609–620, 2020.

Du, S., Kakade, S., Lee, J., Lovett, S., Mahajan, G., Sun, W., and Wang, R. Bilinear classes: A structural framework for provable generalization in rl. In International Conference on Machine Learning, pp. 2826–2836. PMLR, 2021.

Du, Y., Srikant, R., and Chen, W. Cascading reinforcement learning. arXiv preprint arXiv:2401.08961, 2024.

Faury, L., Abeille, M., Calauzènes, C., and Fercoq, O. Improved optimistic algorithms for logistic bandits. In International Conference on Machine Learning, pp. 3052–3060. PMLR, 2020.

Faury, L., Abeille, M., Jun, K.-S., and Calauzènes, C. Jointly efficient and optimal algorithms for logistic bandits. In

International Conference on Artificial Intelligence and Statistics, pp. 546–580. PMLR, 2022.   
Filippi, S., Cappé, O., Garivier, A., and Szepesvári, C. Parametric bandits: The generalized linear case. In Proceedings of the 23rd International Conference on Neural Information Processing Systems - Volume 1, NIPS'10, pp. 586–594, Red Hook, NY, USA, 2010. Curran Associates Inc.   
Foster, D. J., Kakade, S. M., Qian, J., and Rakhlin, A. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487, 2021.   
Gheshlaghi Azar, M., Munos, R., and Kappen, H. J. Minimax pac bounds on the sample complexity of reinforcement learning with a generative model. Machine learning, 91:325–349, 2013.   
He, J., Ostendorf, M., He, X., Chen, J., Gao, J., Li, L., and Deng, L. Deep reinforcement learning with a combinatorial action space for predicting popular reddit threads. arXiv preprint arXiv:1606.03667, 2016.   
Ie, E., Jain, V., Wang, J., Narvekar, S., Agarwal, R., Wu, R., Cheng, H.-T., Chandra, T., and Boutilier, C. Slateq: A tractable decomposition for reinforcement learning with recommendation sets. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19, pp. 2592–2599, 2019.   
Ishfaq, H., Cui, Q., Nguyen, V., Ayoub, A., Yang, Z., Wang, Z., Precup, D., and Yang, L. Randomized exploration in reinforcement learning with general value function approximation. In International Conference on Machine Learning, volume 139, pp. 4607–4616. PMLR, 2021.   
Jiang, N., Krishnamurthy, A., Agarwal, A., Langford, J., and Schapire, R. E. Contextual decision processes with low bellman rank are pac-learnable. In International Conference on Machine Learning, pp. 1704–1713. PMLR, 2017.   
Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. Is q-learning provably efficient? In Advances in Neural Information Processing Systems, volume 31, pp. 4868–4878, 2018.   
Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory, pp. 2137–2143. PMLR, 2020.   
Jin, C., Liu, Q., and Miryoosefi, S. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms. Advances in neural information processing systems, 34:13406–13418, 2021.

Kong, D., Salakhutdinov, R., Wang, R., and Yang, L. F. Online sub-sampling for reinforcement learning with general function approximation. arXiv preprint arXiv:2106.07203, 2021.   
Kveton, B., Szepesvari, C., Wen, Z., and Ashkan, A. Cascading bandits: Learning to rank in the cascade model. In International conference on machine learning, pp. 767–776. PMLR, 2015a.   
Kveton, B., Wen, Z., Ashkan, A., and Szepesvari, C. Combinatorial cascading bandits. Advances in Neural Information Processing Systems, 28, 2015b.   
Lattimore, T. and Hutter, M. Pac bounds for discounted mdps. In Algorithmic Learning Theory: 23rd International Conference, ALT 2012, Lyon, France, October 29-31, 2012. Proceedings 23, pp. 320–334. Springer, 2012.   
Lee, J. and Oh, M.-h. Demystifying linear mdps and novel dynamics aggregation framework. In The Twelfth International Conference on Learning Representations, 2023.   
Lee, J. and Oh, M.-h. Nearly minimax optimal regret for multinomial logistic bandit. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.   
Li, S., Lattimore, T., and Szepesvári, C. Online learning to rank with features. In International Conference on Machine Learning, pp. 3856–3865. PMLR, 2019.   
McFadden, D. Modelling the choice of residential location. 1977.   
McInerney, J., Brost, B., Chandar, P., Mehrotra, R., and Carterette, B. Counterfactual evaluation of slate recommendations with sequential reward interactions. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1779–1788, 2020.   
Metz, L., Ibarz, J., Jaitly, N., and Davidson, J. Discrete sequential prediction of continuous actions for deep rl. arXiv preprint arXiv:1705.05035, 2017.   
Mhammedi, Z., Koolen, W. M., and Van Erven, T. Lipschitz adaptivity with multiple learning rates in online learning. In Conference on Learning Theory, pp. 2490–2511. PMLR, 2019.   
Oh, M.-h. and Iyengar, G. Thompson sampling for multinomial logit contextual bandits. Advances in Neural Information Processing Systems, 32, 2019.   
Oh, M.-h. and Iyengar, G. Multinomial logit contextual bandits: Provable optimality and practicality. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 9205–9213, 2021.

Orabona, F. A modern introduction to online learning. arXiv preprint arXiv:1912.13213, 2019.   
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
Perivier, N. and Goyal, V. Dynamic pricing and assortment under a contextual mnl demand. Advances in Neural Information Processing Systems, 35:3461–3474, 2022.   
Rusmevichientong, P., Shen, Z.-J. M., and Shmoys, D. B. Dynamic assortment optimization with a multinomial logit choice model and capacity constraint. Operations research, 58(6):1666–1680, 2010.   
Russo, D. and Van Roy, B. Eluder dimension and the sample complexity of optimistic exploration. In Advances in Neural Information Processing Systems, pp. 2256–2264, 2013.   
Ryu, M., Chow, Y., Anderson, R., Tjandraatmadja, C., and Boutilier, C. Caql: Continuous action q-learning. arXiv preprint arXiv:1909.12397, 2019.   
Saha, A., Pacchiano, A., and Lee, J. Dueling rl: Reinforcement learning with trajectory preferences. In International Conference on Artificial Intelligence and Statistics, pp. 6263–6289. PMLR, 2023.   
Sauré, D. and Zeevi, A. Optimal dynamic assortment planning with demand learning. Manufacturing & Service Operations Management, 15(3):387–404, 2013.   
Sunehag, P., Evans, R., Dulac-Arnold, G., Zwols, Y., Visentin, D., and Coppin, B. Deep reinforcement learning with attention for slate markov decision processes with high-dimensional states and actions. arXiv preprint arXiv:1512.01124, 2015.   
Swaminathan, A., Krishnamurthy, A., Agarwal, A., Dudik, M., Langford, J., Jose, D., and Zitouni, I. Off-policy evaluation for slate recommendation. Advances in Neural Information Processing Systems, 30, 2017.   
Tran-Dinh, Q., Li, Y.-H., and Cevher, V. Composite convex minimization involving self-concordant-like cost functions. In Modelling, Computation and Optimization in Information Systems and Management Sciences: Proceedings of the 3rd International Conference on Modelling, Computation and Optimization in Information Systems and Management Sciences-MCO 2015-Part I, pp. 155–168. Springer, 2015.

Vlassis, N., Chandrashekar, A., Amat, F., and Kallus, N. Control variates for slate off-policy evaluation. Advances in Neural Information Processing Systems, 34:3667–3679, 2021.   
Wang, R., Salakhutdinov, R. R., and Yang, L. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. Advances in Neural Information Processing Systems, 33, 2020.   
Wang, Y., Wang, R., Du, S. S., and Krishnamurthy, A. Optimism in reinforcement learning with generalized linear function approximation. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=CBmJwzneppz.   
Wirth, C., Akrour, R., Neumann, G., and Fürnkranz, J. A survey of preference-based reinforcement learning methods. Journal of Machine Learning Research, 18(136):1–46, 2017.   
Yang, L. and Wang, M. Sample-optimal parametric q-learning using linearly additive features. In International Conference on Machine Learning, pp. 6995–7004. PMLR, 2019.   
Zhan, W., Uehara, M., Kallus, N., Lee, J. D., and Sun, W. Provable offline preference-based reinforcement learning. arXiv preprint arXiv:2305.14816, 2023.   
Zhang, Y.-J. and Sugiyama, M. Online (multinomial) logistic bandit: Improved regret and constant computation cost. Advances in Neural Information Processing Systems, 36, 2024.   
Zhao, H., He, J., and Gu, Q. A nearly optimal and low-switching algorithm for reinforcement learning with general function approximation. arXiv preprint arXiv:2311.15238, 2023.   
Zhou, D., Gu, Q., and Szepesvari, C. Nearly minimax optimal reinforcement learning for linear mixture markov decision processes. In Conference on Learning Theory, pp. 4532–4576. PMLR, 2021a.   
Zhou, D., He, J., and Gu, Q. Provably efficient reinforcement learning for discounted mdps with feature mapping. In International Conference on Machine Learning, pp. 12793–12802. PMLR, 2021b.   
Zhu, B., Jordan, M., and Jiao, J. Principled reinforcement learning with human feedback from pairwise or k-wise comparisons. In International Conference on Machine Learning, pp. 43037–43067. PMLR, 2023.

# Appendix

# A. Illustrative Explanation for Combinatorial RL with Preference Feedback

![](images/255e148917b68e70ebb413a3bb81b8db5b15d5445a0eedecedb5f2188ea690e0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Items 1"] --> B["A"]
    C["Items 2"] --> B
    D["Items 3"] --> B
    E["Items 4"] --> B
    F["Items 5"] --> B
    G["Items 6"] --> B
    B --> H["State 1"]
    style H fill:#d4edda,stroke:#333
```
</details>

![](images/4107f55ed9e7b34c3dd28d48bda405f4b447c563cc2035de7ede7c9c1313a45e.jpg)

<details>
<summary>text_image</summary>

Items
1
2
3
4
5
6
A
2
4
5
State 1
Item 2 is chosen by the user.
</details>

![](images/2fe62a980ac0e5f4eb25bb12b22a8d85a2f045e0ac88c8f87b72d67e2b619dad.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Items 1-6"] --> B["A"]
    B --> C["State 2: 2, 4, 5"]
    D["Receive a reward and Transition to the next state."] --> E["+0 Symbol"]
```
</details>

Figure A.1: Illustration of combinatorial RL with preference feedback.

In this section, we provide additional explanation of our framework, combinatorial RL with preference feedback, for better clarity. In this framework, at each episode, as a user arrives at the system (starting in the initial state, e.g., high loyalty), a learning agent selects an assortment A (a set of items) and offers it to the user (the first figure in Figure A.1). The user then chooses an item from the assortment A (the second figure in Figure A.1). The agent receives a reward, along with preference (or choice) feedback, and transitions to the next state (e.g., lower loyalty) (the last figure in Figure A.1). This process repeats until the episode concludes.

The key advantage of this framework is its ability to capture the long-term value of choosing an item by considering state transitions and avoiding myopic decisions. For instance, in Figure A.1, a user may choose a junk item that provides a high immediate reward. However, repeatedly recommending such items can lead to user fatigue, resulting in a transition to a state of lower satisfaction or loyalty to the system, ultimately leading to a lower cumulative reward.

We compare our framework with other related works.

vs MNL bandits. Our framework can be considered as a multi-step extension of MNL bandits (Rusmevichientong et al., 2010; Sauré & Zeevi, 2013; Agrawal et al., 2017; 2019; Oh & Iyengar, 2019; 2021; Perivier & Goyal, 2022; Agrawal et al., 2023; Zhang & Sugiyama, 2024; Lee & Oh, 2024). In MNL bandits, there are no state transitions; thus, in Figure F.1, the user exits the system immediately after receiving a reward.

Another important difference is that, in MNL bandits, the value (reward) of choosing an item is assumed to be known, and the value of choosing the outside option $a_{0}$ is always assumed to be zero. In contrast, in our framework, the value of choosing an item is unknown due to the stochastic nature of rewards and transition probabilities. Additionally, we allow the value of choosing the outside option $a_{0}$ to be non-zero.

vs Cascading RL. In cascading RL (Du et al., 2024), the agent also selects a set of items, and state transitions are taken into account when making decisions. However, these items are offered to the user one by one, and the user decides whether to choose the currently offered item.

Cascading RL fundamentally differs from our framework because the user does not compare multiple items at once, so it does not involve relative preference feedback. Another key distinction is that in cascading RL, the probability of choosing each item is independent of the others in the chosen set of items In contrast, in our MNL preference model, the choice probability of an item is influenced by the other items in the assortment.

vs PbRL. In preference-based RL (PbRL) (Akrour et al., 2012; Wirth et al., 2017; Christiano et al., 2017; Ouyang et al.,

2022; Saha et al., 2023; Zhu et al., 2023; Zhan et al., 2023), the agent learns not from explicit numerical rewards, but through preferences as feedback. The user is presented with two (or sometimes multiple) items and chooses a preferred one.

In our framework, if we treat the reward signal generated by the user's choice as a preference signal instead of a numerical reward, we can learn the policy based on user preferences, similar to PbRL, without relying on explicit rewards. However, our framework differs fundamentally from PbRL because our goal is not just to offer a single item, but multiple items—a combinatorial (base) action—at each timestep.

# B. Efficient Bonus Oracle B using Online-subsampling

The guarantees of Algorithm 1 rely on a consistent bonus oracle, $\mathcal{B}$ , that satisfies Definition B.1.

Definition B.1 (Oracle $\mathcal{B}$ , Agarwal et al. 2023). For any $(h,k)\in [H]\times [K]$ , sequence of $\{\bar{\sigma}_h^\tau \}_{\tau = 1}^{k - 1}$ and dataset $\mathcal{D}_h^{k - 1} = \{(s_h^\tau ,a_h^\tau ,r_h^\tau ,s_{h + 1}^\tau)\}_{\tau = 1}^{k - 1}$ , function class $\mathcal{F}_h$ with $\hat{f}_h\in \mathcal{F}_h$ , $\beta_h,\rho >0$ , error parameter $\epsilon_b\geqslant 0$ , the bonus oracle $\mathcal{B}(\{\bar{\sigma}_h^\tau \}_{\tau = 1}^{k - 1},\mathcal{D}_h^{k - 1},\mathcal{F}_h,\hat{f}_h,\beta_h,\rho ,\epsilon_b)$ outputs a bonus function $b_{h}(\cdot)$ such that, for any $z_{h} = (s_{h},a_{h})\in \mathcal{S}\times \mathcal{I}$ , we have

- $b_h: \mathcal{S} \times \mathcal{I} \to \mathbb{R}_+$ belongs to a bonus function class $\mathcal{W}$ and denote $\mathcal{N}_b = |\mathcal{W}|$ .   
- $b_{h}(z_{h}) \geqslant \max \left\{ |f_{h}(z_{h}) - \hat{f}_{h}(z_{h})|, f_{h} \in \mathcal{F}_{h} : \sum_{\tau=1}^{k-1} \frac{1}{(\bar{\sigma}_{h}^{\tau})^{2}} \left( f_{h}(z_{h}^{\tau}) - \hat{f}_{h}^{k}(z_{h}^{\tau}) \right)^{2} \leqslant (\beta_{h})^{2} \right\}.$   
- $b_{h}(z_{h}) \leqslant C \cdot \left( \mathcal{D}_{\mathcal{F}_{h}} \left( z_{h}; \{ z_{h}^{\tau} \}_{\tau=1}^{k-1}, \{\bar{\sigma}_{h}^{\tau} \}_{\tau=1}^{k-1} \right) \cdot \sqrt{(\beta_{h})^{2} + \rho} + \epsilon_{b} \cdot \beta_{h} \right)$ with $0 < C < \infty$ .

Further we say the oracle $\mathcal{B}$ is consistent if for any $k < k'$ with consistent $\{\bar{\sigma}_h^\tau\}_{\tau=1}^{k-1} \subseteq \{\bar{\sigma}_h^\tau\}_{\tau=1}^{k'-1}, \mathcal{D}_h^{k-1} \subseteq \mathcal{D}_h^{k'-1}, \beta_h^k$ non-decreasing in $k$ for each $h \in [H]$ and $\hat{f}_h^k$ as defined in (3), it holds that $\mathcal{B}(\{\bar{\sigma}_h^\tau\}_{\tau=1}^{k-1}, \mathcal{D}_h^{k-1}, \mathcal{F}_h, \hat{f}_h^k, \beta_h^k, \rho, \epsilon_b) \geqslant \mathcal{B}(\{\bar{\sigma}_h^\tau\}_{\tau=1}^{k'-1}, \mathcal{D}_h^{k'-1}, \mathcal{F}_h, \hat{f}_h^{k'}, \beta_h^{k'}, \rho, \epsilon_b)$ element-wise.

With the oracle B, we can efficiently calculate the optimistic $\overline{Q}$ -value estimate $f_{h}^{k}$ with an error of $\epsilon_{b}$ .

To implement this oracle, we use the online sensitivity sub-sampling approach described by Agarwal et al. (2023), which builds on the original sensitivity sub-sampling method proposed by Kong et al. (2021) and Wang et al. (2020).

For completeness, we include the sub-sampling procedure in Algorithm B.1 and show its guarantees in Proposition B.3. Let $z = (s, a) \in \mathcal{S} \times \mathcal{I}$ . We first define the weighted data set Z, where each element is $(z, \bar{\sigma}(z))$ , and introduce the weighted sensitivity score as follows:

$$
\mathrm{sensitivity} _ {\mathcal {Z}, \mathcal {F}, \gamma , \nu} (z) = \min \left\{\sup _ {f, f ^ {\prime} \in \mathcal {F}} \frac {\frac {1}{\bar {\sigma} ^ {2} (z)} \left(f (z) - f ^ {\prime} (z)\right) ^ {2}}{\min \left\{\sum_ {z ^ {\prime} \in \mathcal {Z}} \frac {1}{\bar {\sigma} ^ {2} (z ^ {\prime})} \left(f (z ^ {\prime}) - f ^ {\prime} (z ^ {\prime})\right) ^ {2} , \frac {K (H + 1) ^ {2}}{\nu^ {2}} \right\} + \gamma^ {2}}, 1 \right\}.
$$

Now we introduce the sub-sampling procedure.

Algorithm B.1 Online Sensitivity Sub-sampling with Weights   
1: Inputs: function class $\mathcal{F}$ , current sub-sampled dataset $\hat{\mathcal{Z}} \subseteq \mathcal{S} \times \mathcal{I}$ , new state-action pair $s, a$ , parameter $\gamma$ , threshold $\nu > 0$ , failure probability $\delta$ .
2: Parameter: $1 \leqslant C < \infty$ 3: Let $p_z$ be the smallest real number such that $1/p_z$ is an integer and $p_z \geqslant \min \left\{1, C \cdot \text{sensitivity}_{\hat{\mathcal{Z}}, \mathcal{F}, \gamma, \nu}(z) \cdot \log(K\mathcal{N}/\delta)\right\}$ .
4: Independently add $1/p_z$ copies of $(z, \bar{\sigma}(z))$ into $\hat{\mathcal{Z}}$ with probability $p_z$ .
5: Return: $\hat{\mathcal{Z}}$ .

For the weighted dataset $\mathcal{Z}_{h}^{k-1} = \{(s_{h}^{\tau}, a_{h}^{\tau}), \bar{\sigma}_{h}^{\tau}\}_{\tau=1}^{k-1}$ , we defined $\|f\|_{\mathcal{Z}_{h}^{k-1}}^{2} = \sum_{z \in \mathcal{Z}_{h}^{k-1}} \frac{1}{\bar{\sigma}^{2}(z)} f^{2}(z)$ , i.e., weighted sum of $\ell_{2}$ -norm square. We denote $\hat{Z}_{h}^{k-1}$ as the dataset sub-sampled from $Z_{h}^{k-1}$ . At every $(k, h) \in [K] \times [H]$ , we call Algorithm B.1 with the current sub-sampled dataset $\hat{Z}_{h}^{k-1}$ and the new state action-pair $z_{h}^{k} = (s_{h}^{k}, a_{h}^{k})$ to generate the next sub-sampled dataset $\hat{Z}_{h}^{k}$ .

The following proposition shows that the distance of any two functions measured by the historical dataset $Z_{h}^{k-1}$ is well approximated by the subsamping dataset $\hat{Z}_{h}^{k-1}$ . Additionally, it shows that both the number of distinct elements in $|\hat{Z}_{h}^{k-1}|$ and its total size (counting repetitions) do not scale with poly(S).

Proposition B.2 (Guarantees of online sensitivity sub-sampling, Proposition 13 of Agarwal et al. 2023). Let $z = (s, a) \in S \times \mathcal{I}$ . When $\bar{\sigma}(z) \geqslant \nu$ for any $z$ , then with probability at least $1 - \delta$ , it holds that

$$
\begin{array}{l} \sup _ {f _ {1}, f _ {1}: \| f _ {1} - f _ {2} \| _ {\mathcal {Z} _ {h} ^ {k}} ^ {2} \leqslant \gamma^ {2}} | f _ {1} (z) - f _ {2} (z) | \leqslant \sup _ {f _ {1}, f _ {1}: \| f _ {1} - f _ {2} \| _ {\hat {\mathcal {Z}} _ {h} ^ {k}} ^ {2} \leqslant 1 0 ^ {2} \gamma^ {2}} | f _ {1} (z) - f _ {2} (z) | \\ \leqslant \sup _ {f _ {1}, f _ {1}: \| f _ {1} - f _ {2} \| _ {\mathcal {Z} _ {h} ^ {k}} ^ {2} \leqslant 1 0 ^ {4} \gamma^ {2}} | f _ {1} (z) - f _ {2} (z) |. \\ \end{array}
$$

Further, for any $(k,h)\in[K]\times[H]$ , the number of distinct elements in sub-sampled dataset $\hat{Z}_{h}^{k}$ is always bounded by $\mathcal{O}\left(\log\frac{KN}{\delta}\cdot\max_{h\in[H]}\dim_{\nu,K}(\mathcal{F}_{h})\right)$ and the total size of $\hat{Z}_{h}^{k}$ is bounded by $\mathcal{O}(K^{3}/\delta)$ .

We can assert that the predictive differences between the functions are preserved up to constant factors, while requiring significantly less data. Then, the size of the bonus class W in Definition B.1 is bounded as follows:

Proposition B.3 (Implementing $\mathcal{B}$ using online-subsampling, Corollary 14 of Agarwal et al. 2023). There exists an algorithm (see Algorithm B.1) such that, with probability at least $1 - \delta$ , implements a consistent bonus oracle $\mathcal{B}$ with $\epsilon_b = 0$ for all $(k, h) \in [K] \times [H]$ , where

$$
\log | \mathcal {W} | \leqslant \mathcal {O} \left(\max _ {h \in [ H ]} \dim_ {\nu , K} (\mathcal {F} _ {h}) \cdot \log \frac {K \mathcal {N}}{\delta} \log \frac {K | \mathcal {S} \times \mathcal {I} |}{\delta}\right).
$$

# C. Efficient combinatorial optimization

In this section, we explain how to solve the combinatorial optimization problem in (8), following the method outlined in Davis et al. (2013); Ie et al. (2019).

To find an assortment $A \in A$ that maximizes the optimistic Q-value, a fundamental step in Q-learning and crucial for inducing exploration, we must solve the following combinatorial optimization problem:

$$
\max _ {A \in \mathcal {A}} \sum_ {a \in A} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a), \tag {C.1}
$$

where $\widetilde{P}_{h,j}^{k}$ is the optimistic choice probability as defined in (7) (also in (D.17)), and $f_{h,j}^{k}$ is the $\overline{Q}$ -value estimate (item-level Q-values) as defined in (D.15).

Fix $(k,h,s,j)\in[K]\times[H]\times\mathcal{S}\times\{1,2,-2\}$ . For simplicity, we will abbreviate these indices. Accordingly, we denote $w_{a}=\exp\left(\widetilde{v}_{h}^{k}(s_{h}^{k},a)\right)$ or $w_{a}=\exp\left(\check{v}_{h}^{k}(s_{h}^{k},a)\right)$ , depending on the value of j. Additionally, let $\tilde{f}_{a}=f_{h,j}^{k}(s_{h}^{k},a)$ for simplicity.

We can then express the optimization problem in (C.1) in terms of $w$ as fractional mixed-integer program (MIP), with binary variables $x_{a} \in \{0,1\}$ for each item $a \in \mathcal{I} \backslash \{a_0\}$ , indicating whether $a$ is included in the assortment $A$ :

$$
\max \frac {w _ {a _ {0}} \tilde {f} _ {a _ {0}} + \sum_ {a \in \mathcal {I} \backslash \{a _ {0} \}} x _ {a} w _ {a} \tilde {f} _ {a}}{w _ {a _ {0}} + \sum_ {a ^ {\prime} \in \mathcal {I} \backslash \{a _ {0} \}} x _ {a ^ {\prime}} w _ {a ^ {\prime}}} \tag {C.2}
$$

$$
\text { s.t. } \sum_ {a \in \mathcal {I} \setminus \{a _ {0} \}} x _ {a} \leqslant M - 1;
$$

$$
x _ {a} \in \{0, 1 \}, \forall a \in \mathcal {I} \backslash \{a _ {0} \}.
$$

By Chen & Hausman (2000), the binary indicator in the MIP can be relaxed, resulting in the following fractional linear program (LP):

$$
\max \frac {w _ {a _ {0}} \tilde {f} _ {a _ {0}} + \sum_ {a \in \mathcal {I} \backslash \{a _ {0} \}} x _ {a} w _ {a} \tilde {f} _ {a}}{w _ {a _ {0}} + \sum_ {a ^ {\prime} \in \mathcal {I} \backslash \{a _ {0} \}} x _ {a ^ {\prime}} w _ {a ^ {\prime}}} \tag {C.3}
$$

$$
\text { s.t. } \sum_ {a \in \mathcal {I} \setminus \{a _ {0} \}} x _ {a} \leqslant M - 1;
$$

$$
0 \leqslant x _ {a} \leqslant 1, \forall a \in \mathcal {I} \backslash \{a _ {0} \}.
$$

Since this relaxed problem is a fractional linear program (LP), using the Charnes-Cooper method (Cooper et al., 1962), it can be transformed into a (non-fractional) LP. To achieve this, we introduce additional variables:

$$
t = \frac {1}{w _ {a _ {0}} + \sum_ {a ^ {\prime} \in \mathcal {I} \setminus \{a _ {0} \}} x _ {a ^ {\prime}} w _ {a ^ {\prime}}}, y _ {a} = \frac {x _ {a}}{w _ {a _ {0}} + \sum_ {a ^ {\prime} \in \mathcal {I} \setminus \{a _ {0} \}} x _ {a ^ {\prime}} w _ {a ^ {\prime}}}.
$$

Then, we can obtain the following LP:

$$
\max \sum_ {a \in \mathcal {I} \backslash \{a _ {0} \}} \tilde {f} _ {a} w _ {a} y _ {a} + \tilde {f} _ {a _ {0}} w _ {a _ {0}} t \tag {C.4}
$$

$$
\text { s.t. } \sum_ {a \in \mathcal {I} \setminus \{a _ {0} \}} w _ {a} y _ {a} + w _ {a _ {0}} t = 1;
$$

$$
\sum_ {a \in \mathcal {I} \backslash \{a _ {0} \}} y _ {a} \leqslant (M - 1) t; \quad t \geqslant 0.
$$

The optimal solution $(y_{a_{1}}^{\star},\ldots,y_{a_{N}}^{\star},t^{\star})$ to this LP in (C.4) provides the optimal $x_{i}$ values for the fractional LP in (C.3) by setting $x_{a}=y_{a}^{\star}/t^{\star}$ . This, in turn, determines the optimal assortment in the original fractional MIP (Equation (C.2)) by including any item where $y_{a}^{\star}>0$ . Thus, the optimization problem is proven to be solvable in polynomial time.

# D. Proof of Theorem 5.1

# D.1. Notations and Preliminaries

In this subsection, for easy reference, we introduce notations and definitions used throughout the proof. The key notations are summarized in Table D.1, and the specific parameter choices are listed in Table D.2.

Online parameter update and confidence interval for MNL preference model. We define the multinomial logistic loss function at $(k, h) \in [K] \times [H]$ as follows:

$$
\ell_ {h} ^ {k} (\boldsymbol {\theta}) := - \sum_ {a \in A _ {h} ^ {k}} y _ {h} ^ {k} (a) \log \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta}). \tag {D.1}
$$

To achieve constant-time parameter estimation, we use the online mirror descent algorithm to estimate the true parameter $\theta_{h}^{\star}$ :

$$
\boldsymbol {\theta} _ {h} ^ {k + 1} \in \underset {\boldsymbol {\theta} \in \Theta} {\operatorname{argmin}} \left\langle \nabla \ell_ {h} ^ {k} \left(\boldsymbol {\theta} _ {h} ^ {k}\right), \boldsymbol {\theta} \right\rangle + \frac {1}{2 \eta} \| \boldsymbol {\theta} - \boldsymbol {\theta} _ {h} ^ {k} \| _ {\mathbf {H} _ {h} ^ {k}} ^ {2}, \quad \text { where } \Theta = \left\{\boldsymbol {\theta} \in \mathbb {R} ^ {d}: \| \boldsymbol {\theta} \| _ {2} \leqslant B \right\}, \tag {D.2}
$$

where $\eta = \frac{1}{2} \log(M + 1) + B + 1$ is the step-size parameter, and the related matrices are defined as:

$$
\mathbf {H} _ {h} ^ {k} := \lambda \mathbf {I} _ {d} + \sum_ {\tau = 1} ^ {k - 1} \nabla^ {2} \ell_ {h} ^ {\tau} (\boldsymbol {\theta} _ {h} ^ {\tau + 1}),
$$

$$
\tilde {\mathbf {H}} _ {h} ^ {k} := \mathbf {H} _ {h} ^ {k} + \eta \nabla^ {2} \ell_ {h} ^ {k} (\boldsymbol {\theta} _ {h} ^ {k}), \tag {D.3}
$$

where

$$
\begin{array}{l} \nabla^ {2} \ell_ {h} ^ {k} (\boldsymbol {\theta}) = \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta}) \phi (s _ {h} ^ {k}, a) \phi (s _ {h} ^ {k}, a) ^ {\top} \\ - \sum_ {a \in A _ {h} ^ {k}} \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta}) \mathcal {P} _ {h} (a ^ {\prime} | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta}) \phi (s _ {h} ^ {k}, a) \phi (s _ {h} ^ {k}, a ^ {\prime}) ^ {\top}. \\ \end{array}
$$

Table D.1: Summary of notations 

<table><tr><td>Notation</td><td>Meaning</td><td>Remark</td></tr><tr><td> $\mathcal{S},\mathcal{A},\mathcal{I}$ </td><td>state space, action (assortment) space, item set</td><td></td></tr><tr><td> $k,h$ </td><td> $k\in[K]$  episode,  $h\in[H+1]$  horizon</td><td></td></tr><tr><td> $r_{h}^{k},s_{h}^{k},A_{h}^{k},a_{h}^{k}$ </td><td>reward, state, action and item at  $k,h$ </td><td></td></tr><tr><td> $r_{h},s_{h},A_{h},a_{h}$ </td><td>random reward, state, action and item  $h$ </td><td></td></tr><tr><td> $z$ </td><td>shorthand for state-item pair  $(s,a)$ </td><td></td></tr><tr><td> $\overline{Q}: \mathcal{S}\times\mathcal{I}\rightarrow\mathbb{R}$ </td><td>item-level Q-value function</td><td></td></tr><tr><td> $\mathcal{T}_{h},\mathcal{T}_{h}^{2}$ </td><td>Bellman operator and second-moment operator</td><td></td></tr><tr><td> $\mathcal{D}_{h}^{k-1}$ </td><td>data set  $\{(s_{h}^{\tau},a_{h}^{\tau},r_{h}^{\tau},s_{h+1}^{\tau})\}_{\tau=1}^{k-1}$ </td><td></td></tr><tr><td> $\mathcal{F}_{h}$ </td><td>function class for horizon  $h\in[H]$ </td><td>Ass. 3.3</td></tr><tr><td> $\mathcal{F}_{h}^{\text{lin}}$ </td><td>linear function class for horizon  $h\in[H]$ </td><td>Eqn. E.1</td></tr><tr><td> $\mathcal{F}_{h}^{\text{lin}}(\epsilon_{c})$ </td><td> $\epsilon_{c}$ -cover of linear function class  $\mathcal{F}_{h}^{\text{lin}}$ </td><td></td></tr><tr><td> $\mathcal{W}$ </td><td>bonus function class defined for bonus oracle  $\mathcal{B}$ </td><td>Def. B.1</td></tr><tr><td> $\epsilon_{b}$ </td><td>error parameter for bonus oracle</td><td></td></tr><tr><td> $\mathcal{N}$ </td><td>maximal size of function class, i.e.,  $\max_{h\in[H]}|\mathcal{F}_{h}|$ </td><td></td></tr><tr><td> $\mathcal{N}_{b}$ </td><td>size of bonus function class  $|\mathcal{W}|$ </td><td></td></tr><tr><td> $D_{\mathcal{F}}^{2}(z;\{z^{\tau}\}_{\tau=1}^{k-1},\{\sigma^{\tau}\}_{\tau=1}^{k-1})$ </td><td>:=  $\sup_{f_{1},f_{2},\in\mathcal{F}}\frac{(f_{1}(z)-f_{2}(z))^{2}}{\sum_{\tau=1}^{k-1}\frac{1}{(\sigma^{\tau})^{2}}(f_{1}(z^{\tau})-f_{2}(z^{\tau}))^{2}+\rho}$ </td><td> $\rho$  param.</td></tr><tr><td> $\dim_{\nu,K}(\mathcal{F})$ </td><td>generalized Eluder dimension defined in Definition 3.5</td><td> $\nu$  param.</td></tr><tr><td> $d_{\nu}$ </td><td>:=  $\frac{1}{H}\sum_{h=1}^{H}\dim_{\nu,K}(\mathcal{F}_{h})$  (Definition 3.5)</td><td> $\nu$  param.</td></tr><tr><td> $\ell_{h}^{k}(\boldsymbol{\theta})$ </td><td> $-\sum_{a\in A_{h}^{k}}y_{h}^{k}(a)\log\mathcal{P}_{h}(a|s_{h}^{k},A_{h}^{k};\boldsymbol{\theta})$ , loss for MNL model at  $k,h$ </td><td>Eqn. D.1</td></tr><tr><td> $\mathbf{H}_{h}^{k},\check{\mathbf{H}}_{h}^{k}$ </td><td>=  $\lambda\mathbf{I}_{d}+\sum_{\tau=1}^{k-1}\nabla^{2}\ell_{h}^{\tau}(\boldsymbol{\theta}_{h}^{\tau+1})$ , =  $\mathbf{H}_{h}^{k}+\eta\nabla^{2}\ell_{h}^{k}(\boldsymbol{\theta}_{h}^{k})$ , respectively</td><td>Eqn. D.3</td></tr><tr><td> $\mathcal{C}_{h}^{k}$ </td><td>confidence interval for MNL model at  $k,h$ </td><td>Eqn. D.5</td></tr><tr><td> $f_{h,1}^{k}$ </td><td>optimistic  $\overline{Q}$  at  $k,h$ </td><td></td></tr><tr><td> $\hat{f}_{h,1}^{k}$ </td><td>solution of fitting weighted regression at  $k,h$ </td><td>Eqn. D.7</td></tr><tr><td> $\mathcal{F}_{h,1}^{k}$ </td><td>version space of optimistic  $\overline{Q}$  at  $k,h$ </td><td>Eqn. D.8</td></tr><tr><td> $f_{h,\pm2}^{k}$ </td><td>overly optimistic (pessimistic)  $\overline{Q}$  at  $k,h$ </td><td></td></tr><tr><td> $\hat{f}_{h,\pm2}^{k}$ </td><td>solution of fitting unweighted regression at  $k,h$ </td><td>Eqn. D.11</td></tr><tr><td> $\mathcal{F}_{h,\pm2}^{k}$ </td><td>version space of overly optimistic (pessimistic)  $\overline{Q}$  at  $k,h$ </td><td>Eqn. D.12</td></tr><tr><td> $\hat{g}_{h}^{k}$ </td><td>solution of fitting second-moment regression at  $k,h$ </td><td>Eqn. D.13</td></tr><tr><td> $\mathcal{G}_{h}^{k}$ </td><td>version space of second-moment estimates at  $k,h$ </td><td>Eqn. D.14</td></tr><tr><td> $\mathcal{E}^{\theta}$ </td><td>event that  $\{\boldsymbol{\theta}_{h}^{\star}\in\mathcal{C}_{h}^{k} \text{ for all } k\geqslant 1 \text{ and all } h\in[H]\}$ </td><td></td></tr><tr><td> $\mathcal{E}_{h}^{k}$ </td><td>event that  $\{\mathcal{T}_{h}V_{h+1,j}^{k}\in\mathcal{F}_{h,j}^{k} \text{ for } j=1,\pm2 \text{ and }\mathcal{T}_{h}^{2}V_{h+1,1}^{k}\in\mathcal{G}_{h}^{k}\}$ </td><td>Ass. 3.3</td></tr><tr><td> $\mathcal{E}_{\leqslant k}$ </td><td>joint event that  $\bigcap_{\tau=1}^{k}\bigcap_{h=1}^{H}\mathcal{E}_{h}^{\tau}$ </td><td></td></tr><tr><td> $\tilde{v}_{h}^{k}(s_{h}^{k},a),\breve{v}_{h}^{k}(s_{h}^{k},a)$ </td><td>optimistic (pessimistic) utility defined in (6)</td><td></td></tr><tr><td> $\widetilde{\mathcal{P}}_{h}^{k}(a|s,A)$ </td><td>optimistic choice probability defined in (7)</td><td></td></tr><tr><td> $Q_{h,j}^{k}(s,A)$ </td><td>:=  $\sum_{a\in A}\widetilde{\mathcal{P}}_{h,j}^{k}(a|s,A)f_{h,j}^{k}(s,a)$  for  $j=1,\pm2$ </td><td></td></tr><tr><td> $V_{h,j}^{k}(s)$ </td><td>:=  $\max_{A\in\mathcal{A}}Q_{h,j}^{k}(s,A)$  for  $j=1,\pm2$ </td><td></td></tr><tr><td> $A_{h,j}^{k}$ </td><td>∈  $\arg\max_{A\in\mathcal{A}}\sum_{a\in A}\widetilde{\mathcal{P}}_{h,j}^{k}(a|s_{h}^{k},A)f_{h,j}^{k}(s_{h}^{k},a)$  for  $j=1,\pm2$ </td><td></td></tr><tr><td> $A_{h}^{k}$ </td><td>chosen assortment at  $k,h$  by assortment selection rule in (9)</td><td></td></tr><tr><td> $Q_{h}^{k}(s,A),V_{h}^{k}(s)$ </td><td>realized optimistic values determined by (D.18)</td><td>Eqn. 9</td></tr><tr><td> $h_{k}$ </td><td>random  $h$  when first taking action  $A_{h,2}^{k}$  at  $k$ , i.e.,  $A_{h}^{k}=A_{h,2}^{k}$ </td><td>Eqn. 9</td></tr><tr><td> $\mathcal{K}_{o},\mathcal{K}_{oo}$ </td><td>disjoint subsets of [K] when  $h_{k}=H+1$  or  $h_{k}\in[H]$ </td><td>Eqn. 9</td></tr><tr><td> $b_{h,j}^{k}$ </td><td>bonus term obtained in Line 8 and 11 using  $\mathcal{B}$ </td><td>Def. B.1</td></tr><tr><td> $\mathbb{E}_{\mathbb{P}}[\cdot|s_{h}^{k},a_{h}^{k}],\mathbb{V}_{\mathbb{P}}[\cdot|s_{h}^{k},a_{h}^{k}]$ </td><td> $\mathbb{E}_{s_{h+1}\sim\mathbb{P}_{h}(\cdot|s_{h}^{k},a_{h}^{k})}[\cdot|s_{h}^{k},a_{h}^{k}],\mathbb{V}_{s_{h+1}\sim\mathbb{P}_{h}(\cdot|s_{h}^{k},a_{h}^{k})}[\cdot|s_{h}^{k},a_{h}^{k}]$ </td><td></td></tr><tr><td> $\mathbb{E}_{\mathcal{P}}[\cdot|s_{h}^{k},A_{h}^{k}]$ </td><td> $\mathbb{E}_{a_{h}\sim\mathcal{P}_{h}(\cdot|s_{h}^{k},A_{h}^{k})}[\cdot|s_{h}^{k},A_{h}^{k}]$ </td><td></td></tr></table>

Table D.2: Summary of parameter choices 

<table><tr><td>Notation</td><td>Choice</td><td>Remark</td></tr><tr><td> $\delta$ </td><td> $\delta \in (0,1/(H^{2}+15))$ </td><td></td></tr><tr><td> $\delta_{h}^{k}$ </td><td> $\delta/((K+1)(H+1))$ </td><td></td></tr><tr><td> $\eta$ </td><td> $\frac{1}{2}\log(M+1)+B+1$ , step-size parameter for OMD</td><td>Eqn. D.2</td></tr><tr><td> $\lambda$ </td><td> $84\sqrt{2}d\eta$ , regularization parameter</td><td></td></tr><tr><td> $\alpha_{h}^{k}$ </td><td> $\mathcal{O}(\sqrt{d}\log k\log M)$ , confidence radius of  $\mathcal{C}_{h}^{k}$ </td><td>Eqn. D.6</td></tr><tr><td> $\epsilon_{c}$ </td><td>error due to taking covering of function class</td><td></td></tr><tr><td> $\nu$ </td><td> $\sqrt{1/KH}$ </td><td>Def. 3.5</td></tr><tr><td> $\rho$ </td><td>1</td><td>Def. 3.5</td></tr><tr><td> $o(\delta)$ </td><td> $\sqrt{\log\frac{\mathcal{N}^{2}(2\log(4LK/\nu)+2)(\log(8L/\nu^{2})+2)}{\delta}}$ </td><td></td></tr><tr><td> $\iota(\delta)$ </td><td> $3\sqrt{\log\frac{\mathcal{N}\mathcal{N}_{b}(2\log(4LK/\nu)+2)(\log(8L/\nu^{2})+2)}{\delta}}$ </td><td></td></tr><tr><td> $\beta_{h,1}^{k}$ </td><td> $\sqrt{(6\sqrt{\rho}+156)\cdot\log\frac{\mathcal{N}^{2}(K+1)(H+1)(2\log\frac{4LK}{\nu}+2)(\log\frac{8L}{\nu^{2}}+2)}{\delta}}$ ,confidence radius of  $\mathcal{F}_{h,1}^{k}$ </td><td>Eqn. D.8</td></tr><tr><td> $\iota'(\delta)$ </td><td> $\sqrt{2\log\frac{\mathcal{N}\mathcal{N}_{b}(2\log(18LK)+2)(\log(18L)+2)}{\delta}}$ </td><td></td></tr><tr><td> $\beta_{h,2}^{k}$ </td><td> $\sqrt{2(24L+21)(\iota'(\delta_{h}^{k}))^{2}}$ , confidence radius of  $\mathcal{F}_{h,\pm 2}^{k}$ </td><td>Eqn. D.12</td></tr><tr><td> $\iota''(\delta)$ </td><td> $\sqrt{2\log\frac{\mathcal{N}\mathcal{N}_{b}(2\log(32LK)+2)(\log(32L)+2)}{\delta}}$ </td><td></td></tr><tr><td> $\bar{\beta}_{h}^{k}$ </td><td> $\sqrt{8(11L+9)(\iota''(\delta_{h}^{k}))^{2}}$ , confidence radius of  $\mathcal{G}_{h}^{k}$ </td><td>Eqn. D.14</td></tr><tr><td> $(\sigma_{h}^{k})^{2}$ </td><td> $\min\left\{4,\hat{g}_{h}^{k}(z_{h}^{k})-\left(\hat{f}_{h,-2}^{k}(z_{h}^{k})\right)^{2}\right.$  $+D_{\mathcal{F}_{h}}^{2}\left(z_{h}^{k};\{z_{h}^{\tau}\}_{\tau=1}^{k-1},\{1^{\tau}\}_{\tau=1}^{k-1}\right)\cdot\left(\sqrt{\left(\bar{\beta}_{h}^{k}\right)^{2}+\rho}+2L\sqrt{\left(\beta_{h,2}^{k}\right)^{2}+\rho}\right)\right\}$ </td><td></td></tr><tr><td> $\bar{\sigma}_{h}^{k}$ </td><td> $\max\left\{\sigma_{h}^{k},\nu,\sqrt{2}\iota(\delta_{h}^{k})\sqrt{f_{h,2}^{k}(z_{h}^{k})-f_{h,-2}^{k}(z_{h}^{k})},\right.$  $\left.\left.\begin{array}{l}2\left(\sqrt{o(\delta_{h}^{k})}+\iota(\delta_{h}^{k})\right)\cdot\sqrt{D_{\mathcal{F}_{h}}\left(z_{h}^{k};\{z_{h}^{\tau}\}_{\tau=1}^{k-1},\{\bar{\sigma}_{h}^{\tau}\}_{\tau=1}^{k-1}\right)}\end{array}\right\}\right.$ </td><td>Eqn. D.7</td></tr><tr><td> $u_{k}$ </td><td> $C\cdot\left(\sqrt{\log\frac{\mathcal{N}KH}{\nu\delta}}\cdot\left(\log\frac{\mathcal{N}\mathcal{N}_{b}KH}{\nu\delta}\cdot H^{5/2}\sqrt{d_{\nu}}+\sqrt{k}H\epsilon_{b}\right)+dH^{5/2}\log K\log M\sqrt{\log\frac{\mathcal{N}\mathcal{N}_{b}KH}{\nu\delta}}\right)/\sqrt{k} \text{ for } C>0$ </td><td>Eqn. 9</td></tr></table>

By a standard online mirror descent formulation (Orabona, 2019), (D.2) can be solved using a single projected gradient step through the following equivalent formula:

$$
\bar {\boldsymbol {\theta}} _ {h} ^ {k + 1} = \boldsymbol {\theta} _ {h} ^ {k} - \eta (\tilde {H} _ {h} ^ {k}) ^ {- 1} \nabla \ell_ {h} ^ {k} (\boldsymbol {\theta} _ {h} ^ {k}), \quad \text { and } \quad \boldsymbol {\theta} _ {h} ^ {k + 1} \in \underset {\boldsymbol {\theta} \in \Theta} {\operatorname{argmin}} \| \boldsymbol {\theta} - \bar {\boldsymbol {\theta}} _ {h} ^ {k + 1} \| _ {\tilde {H} _ {h} ^ {k}}, \tag {D.4}
$$

which enjoys a computational cost of only $\mathcal{O}(Md^{3})$ , completely independent of k (Mhammedi et al., 2019; Zhang & Sugiyama, 2024; Lee & Oh, 2024).

We define the confidence interval at $(k, h) \in [K] \times [H]$ as follows:

$$
\mathcal {C} _ {h} ^ {k} := \left\{\boldsymbol {\theta} \in \Theta : \left\| \boldsymbol {\theta} - \boldsymbol {\theta} _ {h} ^ {k} \right\| _ {\mathbf {H} _ {h} ^ {k}} \leqslant \alpha_ {h} ^ {k} \right\}, \tag {D.5}
$$

where the radius of the confidence interval $C_{h}^{k}$ is as follows:

$$
\alpha_ {h} ^ {k} = \sqrt {2 \eta \left(1 1 \cdot (3 \log (1 + (M + 1) k) + B + 2) \log \left(\frac {2 \sqrt {1 + 2 k}}{\delta}\right) + 2 + \frac {7 \sqrt {6}}{6} d \eta \log \left(1 + \frac {k + 1}{2 \lambda}\right) + 2\right) + 4 \lambda B ^ {2}} \tag {D.6}
$$

Then, we define the optimistic and pessimistic utility as follows:

$$
\widetilde {v} _ {h} ^ {k} (s, a) := \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {k} + \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}, \quad \check {v} _ {h} ^ {k} (s, a) := \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {k} - \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}},
$$

Regression and confidence intervals for item-level functions. In this paper, we define $N := \max_{h \in [H]} |F_h|$ as the maximum size of the function classes $F_1, \ldots, F_H$ and $N_b = |W|$ as the size of the bonus function class W.

For all $(k, h) \in [K] \times [H]$ , the weighted regression problem for fitting the optimistic item-level $Q$ -functions, $\overline{Q}$ , along with the version space of these functions, is defined as:

$$
\hat {f} _ {h, 1} ^ {k} \in \underset {f _ {h} \in \mathcal {F} _ {h}} {\operatorname{argmin}} \sum_ {\tau = 1} ^ {k - 1} \frac {1}{\left(\bar {\sigma} _ {h} ^ {\tau}\right) ^ {2}} \left(f _ {h} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau}) - r _ {h} ^ {\tau} - V _ {h + 1, 1} ^ {k} (s _ {h + 1} ^ {\tau})\right) ^ {2}, \tag {D.7}
$$

$$
\mathcal {F} _ {h, 1} ^ {k} := \left\{f _ {h} \in \mathcal {F} _ {h}: \sum_ {\tau = 1} ^ {k - 1} \frac {1}{\left(\bar {\sigma} _ {h} ^ {\tau}\right) ^ {2}} \left(f _ {h} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau}) - \hat {f} _ {h, 1} ^ {k} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau})\right) ^ {2} \leqslant \left(\beta_ {h, 1} ^ {k}\right) ^ {2} \right\}. \tag {D.8}
$$

Let $z_{h}^{k} = (s_{h}^{k}, a_{h}^{k})$ . The parameters are as follows (for $k \geqslant 2$ ):

$$
\begin{array}{l} \left(\sigma_ {h} ^ {k}\right) ^ {2} := \min \left\{4, \hat {g} _ {h} ^ {k} (z _ {h} ^ {k}) - \left(\hat {f} _ {h, - 2} ^ {k} (z _ {h} ^ {k})\right) ^ {2} \right. \\ \left. + D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\mathbf {1} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}\right) \cdot \left(\sqrt {\left(\bar {\beta} _ {h} ^ {k}\right) ^ {2} + \rho} + \sqrt {\left(\beta_ {h , 2} ^ {k}\right) ^ {2} + \rho}\right) \right\}, \tag {D.9} \\ \end{array}
$$

$$
\begin{array}{l} \bar {\sigma} _ {h} ^ {k} := \max \left\{\sigma_ {h} ^ {k}, \nu , \sqrt {2} \iota (\delta_ {h} ^ {k}) \sqrt {f _ {h , 2} ^ {k} (z _ {h} ^ {k}) - f _ {h , - 2} ^ {k} (z _ {h} ^ {k})}, \right. \\ 2 \left(\sqrt {o (\delta_ {h} ^ {k})} + \iota (\delta_ {h} ^ {k})\right) \cdot \sqrt {D _ {\mathcal {F} _ {h}} \big (z _ {h} ^ {k} ; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1} , \{\bar {\sigma} _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1} \big)} \Bigg \}, \tag {D.10} \\ \end{array}
$$

$$
\beta_ {h, 1} ^ {k} := \sqrt {(6 \sqrt {\rho} + 1 5 6) \cdot \log \frac {\mathcal {N} ^ {2} (K + 1) (H + 1) (2 \log \frac {4 L K}{\nu} + 2) (\log \frac {8 L}{\nu^ {2}} + 2)}{\delta}},
$$

$$
o (\delta_ {h} ^ {k}) := \sqrt {\log \frac {\mathcal {N} ^ {2} (2 \log (4 L K / \nu) + 2) (\log (8 L / \nu^ {2}) + 2)}{\delta_ {h} ^ {k}}}, \quad \delta_ {h} ^ {k} := \frac {\delta}{(K + 1) (H + 1)},
$$

$$
\iota (\delta_ {h} ^ {k}) := 3 \sqrt {\log \frac {\mathcal {N N} _ {b} (2 \log (4 L K / \nu) + 2) (\log (8 L / \nu^ {2}) + 2)}{\delta_ {h} ^ {k}}}.
$$

For all $(k,h)\in[K]\times[H]$ , the unweighted regression for fitting overly optimistic and overly pessimistic item-level Q-functions, along with the version space of them, is defined as follows:

$$
\hat {f} _ {h, \pm 2} ^ {k} \in \underset {f _ {h} \in \mathcal {F} _ {h}} {\operatorname{argmin}} \sum_ {\tau = 1} ^ {k - 1} \left(f _ {h} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau}) - r _ {h} ^ {\tau} - V _ {h + 1, \pm 2} ^ {k} (s _ {h + 1} ^ {\tau})\right) ^ {2}, \tag {D.11}
$$

$$
\mathcal {F} _ {h, \pm 2} ^ {k} := \left\{f _ {h} \in \mathcal {F} _ {h}: \sum_ {\tau = 1} ^ {k - 1} \left(f _ {h} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau}) - \hat {f} _ {h, \pm 2} ^ {k} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau})\right) ^ {2} \leqslant \left(\beta_ {h, 2} ^ {k}\right) ^ {2} \right\}. \tag {D.12}
$$

We choose the parameters as follows:

$$
\beta_ {h, 2} ^ {k} := \sqrt {2 (2 4 L + 2 1) (\iota^ {\prime} (\delta_ {h} ^ {k})) ^ {2}},
$$

$$
\iota^ {\prime} (\delta_ {h} ^ {k}) := \sqrt {2 \log \frac {\mathcal {N N} _ {b} (2 \log (1 8 L K) + 2) (\log (1 8 L) + 2)}{\delta_ {h} ^ {k}}}, \quad \delta_ {h} ^ {k} := \frac {\delta}{(K + 1) (H + 1)}.
$$

For all $(k, h) \in [K] \times [H]$ , the unweighted regression for fitting second-moment function values to item-level Q-functions, and their version space, is as follows:

$$
\hat {g} _ {h} ^ {k} \in \underset {g _ {h} \in \mathcal {F} _ {h}} {\operatorname{argmin}} \sum_ {\tau = 1} ^ {k - 1} \left(g _ {h} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau}) - \left(r _ {h} ^ {\tau} + V _ {h + 1, 1} ^ {k} (s _ {h + 1} ^ {\tau})\right) ^ {2}\right) ^ {2}, \tag {D.13}
$$

$$
\mathcal {G} _ {h} ^ {k} := \left\{g _ {h} \in \mathcal {F} _ {h}: \sum_ {\tau = 1} ^ {k - 1} \left(g _ {h} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau}) - \hat {g} _ {h} ^ {k} (s _ {h} ^ {\tau}, a _ {h} ^ {\tau})\right) ^ {2} \leqslant \left(\bar {\beta} _ {h} ^ {k}\right) ^ {2} \right\}. \tag {D.14}
$$

We choose the parameters as follows:

$$
\bar {\beta} _ {h} ^ {k} := \sqrt {8 (1 1 L + 9) (\iota^ {\prime \prime} (\delta_ {h} ^ {k})) ^ {2}},
$$

$$
\iota^ {\prime \prime} (\delta_ {h} ^ {k}) := \sqrt {2 \log \frac {\mathcal {N N} _ {b} (2 \log (3 2 L K) + 2) (\log (3 2 L) + 2)}{\delta}}, \quad \delta_ {h} ^ {k} := \frac {\delta}{(K + 1) (H + 1)}.
$$

Given the center of the constructed confidence intervals, $\hat{f}_{h,j}^{k}$ for $j = 1, \pm2$ , we define the optimistic, overly optimistic, and overly pessimistic $\overline{Q}$ -values as follows:

$$
f _ {h, 1} ^ {k} (\cdot , \cdot) := \min \left\{\hat {f} _ {h, 1} ^ {k} (\cdot , \cdot) + b _ {h, 1} ^ {k} (\cdot , \cdot), 1 \right\},
$$

$$
f _ {h, 2} ^ {k} (\cdot , \cdot) := \min \left\{\hat {f} _ {h, 2} ^ {k} (\cdot , \cdot) + 2 b _ {h, 1} ^ {k} (\cdot , \cdot) + b _ {h, 2} ^ {k} (\cdot , \cdot), 1 \right\},
$$

$$
f _ {h, - 2} ^ {k} (\cdot , \cdot) := \max \left\{\hat {f} _ {h, - 2} ^ {k} (\cdot , \cdot) - b _ {h, 2} ^ {k} (\cdot , \cdot), 0 \right\}. \tag {D.15}
$$

Good events. We define the following good events:

$$
\mathcal {E} ^ {\theta} := \left\{\forall k \geqslant 1, \forall h \in [ H ]: \boldsymbol {\theta} _ {h} ^ {\star} \in \mathcal {C} _ {h} ^ {k} \right\}, \tag {D.16}
$$

$$
\mathcal {E} _ {\leqslant K} := \bigcap_ {k = 1} ^ {K} \bigcap_ {h = 1} ^ {H} \mathcal {E} _ {h} ^ {k},
$$

$$
\mathcal {E} _ {h} ^ {k} := \mathcal {E} _ {h, 1} ^ {k} \bigcap \mathcal {E} _ {h, 2} ^ {k} \bigcap \mathcal {E} _ {h, - 2} ^ {k} \bigcap \bar {\mathcal {E}} _ {h} ^ {k},
$$

where $\mathcal{E}_{h,j}^{k} := \left\{\mathcal{T}_{h} V_{h+1,1}^{k} \in \mathcal{F}_{h,j}^{k}\right\}$ for $j = 1, \pm 2$ , and $\bar{\mathcal{E}}_{h}^{k} := \left\{\mathcal{T}_{h}^{2} V_{h+1,1}^{k} \in \mathcal{G}_{h}^{k}\right\}$ .

Optimistic $Q$ -values. For $(k, h, s, A) \in [K] \times [H] \times S \times \mathcal{A}$ and for $j = 1, \pm 2$ , we define the optimistic choice probability as follows:

$$
\widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, A) := \left\{ \begin{array}{l l} \frac {\exp \left(\check {v} _ {h} ^ {k} (s , a)\right)}{\sum_ {a ^ {\prime} \in A} \exp \left(\check {v} _ {h} ^ {k} (s , a ^ {\prime})\right)}, & \text { if } \exists a \in \mathcal {I} \backslash \{a _ {0} \} \text { s.t. } f _ {h, j} ^ {k} (s, a) \geqslant f _ {h, j} ^ {k} (s, a _ {0}) \\ \frac {\exp \left(\check {v} _ {h} ^ {k} (s , a)\right)}{\sum_ {a ^ {\prime} \in A} \exp \left(\check {v} _ {h} ^ {k} (s , a ^ {\prime})\right)}, & \text { otherwise }, \end{array} \right. \tag {D.17}
$$

where

$$
\widetilde {v} _ {h} ^ {k} (s, a) := \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {k} + \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}, \check {v} _ {h} ^ {k} (s, a) := \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {k} - \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}.
$$

Next, we define the optimistic $Q$ -values for $j = 1, \pm 2$ , each constructed using $f_{h,j}^{k}$ and $\widetilde{\mathcal{P}}_{h,j}^{k}$ :

$$
Q _ {h, j} ^ {k} (s, A) = \sum_ {a \in A} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, A) f _ {h, j} ^ {k} (s, a), \quad V _ {h, j} ^ {k} (s) = \max _ {A \in \mathcal {A}} Q _ {h, j} ^ {k} (s, A).
$$

For convenience, we also define the realized optimistic value functions at $(k, h) \in [K] \times [H]$ as follows:

$$
Q _ {h} ^ {k} (s, A) := \left\{ \begin{array}{l l} Q _ {h, 1} ^ {k} (s, A) & \text { if } A _ {h} ^ {k} = A _ {h, 1} ^ {k}, \\ Q _ {h, 2} ^ {k} (s, A) & \text { otherwise }, \end{array} \quad V _ {h} ^ {k} (s) = \max _ {A \in \mathcal {A}} Q _ {h} ^ {k} (s, A), \right. \tag {D.18}
$$

where $A_{h}^{k}$ is the assortment offered to the user by the assortment selection rule in (9) (or equivalently in (D.19)). Therefore, we write $\pi_{h}^{k}(s_{h}^{k}) = \operatorname{argmax}_{A \in \mathcal{A}} Q_{h}^{k}(s_{h}^{k}, A)$ .

Design of exploration policy. At each episode k, the agent collects data using both $A_{h,1}^{k}$ and $A_{h,2}^{k}$ , where $A_{h,j}^{k} \in \arg\max_{A \in A} \sum_{a \in A} \tilde{\mathcal{P}}_{h,j}^{k}(a | s_{h}^{k}, A) f_{h,j}^{k}(s_{h}^{k}, a)$ for j = 1, 2. Given a sequence of pre-specified $\{u_{k}\}_{k=1}^{K}$ , at episode k, the agent select an assortment based on the following rule:

$$
A _ {h} ^ {k} = \left\{ \begin{array}{l l} A _ {h, 1} ^ {k} & \text { if } f _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}}) \geqslant f _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}}) - u _ {k}, \quad \forall a _ {h ^ {\prime}} \in A _ {h ^ {\prime}, 1} ^ {k}, \forall h ^ {\prime} \leqslant h. \\ A _ {h, 2} ^ {k} & \text { otherwise }, \end{array} \right. \tag {D.19}
$$

where

$$
u _ {k} = \mathcal {O} \left(\frac {\sqrt {\log \frac {\mathcal {N} K H}{\nu \delta}} \cdot \left(\log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\nu \delta} \cdot H ^ {5 / 2} \sqrt {d _ {\nu}} + \sqrt {k} H \epsilon_ {b}\right) + d H ^ {5 / 2} \log K \log M \sqrt {\log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\nu \delta}}}{\sqrt {k}}\right). \tag {D.20}
$$

We denote $h_{k} \in [H + 1]$ as the (random) horizon at which the agent first begins offering the assortment $A_{h,2}^{k}$ . More formally, for $h \leqslant h_{k}$ , the assortment offered is $A_{h}^{k} = A_{h,1}^{k}$ , and for $h \geqslant h_{k}$ , the assortment offered is $A_{h}^{k} = A_{h,2}^{k}$ . We then divide the set of episodes [K] into two disjoint subsets: $K_{0}$ and $K_{oo}$ , such that

$$
\mathcal {K} _ {\mathrm{o}} := \{k \in [ K ]: h _ {k} = H + 1 \}, \quad \text { and } \quad \mathcal {K} _ {\mathrm{oo}} := \{k \in [ K ]: h _ {k} \leqslant H \}.
$$

Later in the proof, we separately bound the regret for each case.

Other notations. Throughout the proof, we use $z = (s, a)$ , $z_{h} = (s_{h}, a_{h})$ and $z_{h}^{k} = (s_{h}^{k}, a_{h}^{k})$ interchangeably. We sometimes use $\mathcal{P}_{h}(\cdot | s, A; \boldsymbol{\theta}_{h}^{\star})$ instead of $\mathcal{P}_{h}(\cdot | s, A)$ to explicitly indicate the dependence on the parameter $\theta_{h}^{\star}$ . For simplicity, we denote $\mathbb{E}_{\mathbb{P}}[\cdot | s_{h}^{k}, a_{h}^{k}] = \mathbb{E}_{s_{h+1} \sim \mathbb{P}_{h}(\cdot | s_{h}^{k}, a_{h}^{k})}[\cdot | s_{h}^{k}, a_{h}^{k}]$ , $V_{P}[\cdot | s_{h}^{k}, a_{h}^{k}] = V_{s_{h+1} \sim \mathbb{P}_{h}(\cdot | s_{h}^{k}, a_{h}^{k})}[\cdot | s_{h}^{k}, a_{h}^{k}]$ , and $E_{\mathcal{P}}[\cdot | s_{h}^{k}, A_{h}^{k}] = E_{a_{h} \sim \mathcal{P}_{h}(\cdot | s_{h}^{k}, A_{h}^{k})}[\cdot | s_{h}^{k}, A_{h}^{k}]$ .

# D.2. Confidence Intervals and good events

In this subsection, we show that, given the construction of confidence intervals $C_{h}^{k}$ and $F_{h,j}^{k}$ for $j = 1, \pm2$ , the good events $E^{\theta}$ and $E_{\leqslant K}$ occurs with high probability.

Proposition D.1 (Online parameter confidence interval, Lemma 1 of Lee & Oh 2024). Let $\delta \in (0,1)$ . Under Assumption 3.1, for the confidence interval defined in (2) with

$$
\alpha_ {h} ^ {k} = \sqrt {2 \eta \left(1 1 \cdot (3 \log (1 + (M + 1) k) + B + 2) \log \left(\frac {2 \sqrt {1 + 2 k}}{\delta}\right) + 2 + \frac {7 \sqrt {6}}{6} d \eta \log \left(1 + \frac {k + 1}{2 \lambda}\right) + 2\right) + 4 \lambda B ^ {2}}
$$

$$
\left. + 1 6 \left(\log \left(\frac {2 \sqrt {1 + 2 k}}{\delta}\right)\right) ^ {2}\right) + \frac {7 \sqrt {6}}{6} d \eta \log \left(1 + \frac {k + 1}{2 \lambda}\right) + 2\left. \right) + \lambda B ^ {2} \bigg ] ^ {1 / 2}
$$

$$
= \mathcal {O} (\sqrt {d} \log k \log M),
$$

$\eta = \frac{1}{2}\log (M + 1) + B + 1$ and $\lambda = 84\sqrt{2} d\eta$ , and for any $h\in [H]$ , we have

$$
\operatorname * {P r} \bigl [ \forall k \geqslant 1, \pmb {\theta} _ {h} ^ {\star} \in \mathcal {C} _ {h} ^ {k} \bigr ] \geqslant 1 - \delta .
$$

Now, we define the good event for the preference model $E^{\theta}$ as follows:

$$
\mathcal {E} ^ {\theta} := \left\{\forall k \geqslant 1, \forall h \in [ H ]: \boldsymbol {\theta} _ {h} ^ {\star} \in \mathcal {C} _ {h} ^ {k} \right\}.
$$

Then, by applying proposition D.1 and using a union bound over $h \in [H]$ , we obtain the following corollary:

Corollary D.2 (Good event for MNL preference model). Under the same assumption and settings as in Proposition D.1, for $\delta\in(0,1)$ , with probability at least $1-\delta$ , the good event for the preference model $E^{\theta}$ happens, i.e., $\theta_{h}^{\star}\in C_{h}^{k}$ for all $k\geqslant1$ and all $h\in[H]$ .

The following proposition shows that $\mathcal{T}_hV_{h + 1,j}^k$ for $j = 1,\pm 2$ , and $\mathcal{T}_h^2 V_{h + 1,1}^k$ lie within their respective confidence intervals.

Proposition D.3 (Good event for general functions, Proposition 33 of Agarwal et al. 2023). Suppose Algorithm 1 uses a consistent bonus oracle satisfying Definition B.1. Let $\delta \in (0,1/5)$ . Then, with probability at least $1 - 5\delta$ , the good event $\mathcal{E}_{\leqslant K} = \bigcap_{k=1}^{K} \bigcap_{h=1}^{H} \mathcal{E}_h^k$ happens, that is, $\mathcal{T}_h V_{h+1,1}^k \in \mathcal{F}_{h,1}^k$ , $\mathcal{T}_h V_{h+1,\pm 2}^k \in \mathcal{F}_{h,\pm 2}^k$ , and $\mathcal{T}_h^2 V_{h+1,1}^k \in \mathcal{G}_h^k$ for all $(k,h) \in [K] \times [H]$ .

# D.3. Bound for MNL Preference Model

In this subsection, we provide proofs for several properties of the MNL preference model.

The following lemma presents both the optimistic and pessimistic utilities.

Lemma D.4. For any $(k,h,s,a)\in[K]\times[H]\times\mathcal{S}\times\mathcal{I}$ , let $\widetilde{v}_{h}^{k}(s,a):=\phi(s,a)^{\top}\boldsymbol{\theta}_{h}^{k}+\alpha_{h}^{k}\|\phi(s,a)\|_{(\mathbf{H}_{h}^{k})^{-1}}$ and $\check{v}_{h}^{k}(s,a)=\phi(s,a)^{\top}\boldsymbol{\theta}_{h}^{k}-\alpha_{h}^{k}\|\phi(s,a)\|_{(\mathbf{H}_{h}^{k})^{-1}}$ . Under the good event $E^{\theta}$ defined in (D.16), it holds that

$$
0 \leqslant \widetilde {v} _ {h} ^ {k} (s, a) - \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} \leqslant 2 \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}},
$$

$$
a n d 0 \leqslant \phi (s, a) ^ {\top} \pmb {\theta} _ {h} ^ {\star} - \check {v} _ {h} ^ {k} (s, a) \leqslant 2 \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}.
$$

Proof of Lemma D.4. Conditioning on the good event $E^{\theta}$ holds, we have

$$
\left| \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {k} - \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} \right| \leqslant \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \left\| \boldsymbol {\theta} _ {h} ^ {k} - \boldsymbol {\theta} _ {h} ^ {\star} \right\| _ {\mathbf {H} _ {h} ^ {k}} \leqslant \alpha_ {h} ^ {k} \left\| \phi (s, a) \right\| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}},
$$

where the first inequality holds by Hölder's inequality, and the last inequality holds by Corollary D.2. Therefore, it follows that

$$
\widetilde {v} _ {h} ^ {k} (s, a) - \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} = \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {k} - \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} + \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}
$$

$$
\leqslant 2 \alpha_ {h} ^ {k} \left\| \phi (s, a) \right\| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}.
$$

Furthermore, since $\phi(s,a)^{\top}\boldsymbol{\theta}_{h}^{k}-\phi(s,a)^{\top}\boldsymbol{\theta}_{h}^{\star}\geqslant-\alpha_{h}^{k}\left\|\phi(s,a)\right\|_{\left(\mathbf{H}_{h}^{k}\right)^{-1}}$ , we also get

$$
\widetilde {v} _ {h} ^ {k} (s, a) - \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} = \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {k} - \phi (s, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} + \alpha_ {h} ^ {k} \| \phi (s, a) \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}} \geqslant 0.
$$

The second statement directly follows from the results mentioned above.

Lemma D.5 is useful for proving optimism (Lemma D.15) and bounding the approximation error of the optimistic $\overline{Q}$ (Lemma D.20).

Lemma D.5. For all $(k,h,s,A)\in[K]\times[H]\times\mathcal{S}\times\mathcal{A}$ and any $j\in\{1,2\}$ , under the good event $E^{\theta}$ defined in (D.16), there exists a subset $\tilde{A}\subseteq A$ such that $\tilde{A}\in\mathcal{A}$ and

$$
\max \left\{\sum_ {a \in A} \mathcal {P} _ {h} (a | s, A) f _ {h, j} ^ {k} (s, a), \sum_ {a \in A} \widetilde {\mathcal {P}} _ {h, j ^ {\prime}} ^ {k} (a | s, A) f _ {h, j} ^ {k} (s, a) \right\} \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a),
$$

where $j' \neq j$ .

Proof of Lemma D.5. First, recall that, without loss of generality, we assumed that $\phi(s, a_0) = 0$ for all $s \in S$ . Therefore,

the true preference model and the optimistic preference model can be written as

$$
\mathcal {P} _ {h} (a | s, A) = \frac {\exp \left(\phi (s , a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}{1 + \sum_ {a ^ {\prime} \in A \backslash \{a _ {0} \}} \exp \left(\phi (s , a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)},
$$

$$
\widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, A) = \left\{ \begin{array}{l l} \frac {\exp \left(\check {v} _ {h} ^ {k} (s , a)\right)}{1 + \sum_ {a ^ {\prime} \in A \backslash \{a _ {0} \}} \exp \left(\check {v} _ {h} ^ {k} (s , a ^ {\prime})\right)}, & \text { if } \exists a \in \mathcal {I} \backslash \{a _ {0} \} \text {   s.t.   } f _ {h, j} ^ {k} (s, a) \geqslant f _ {h, j} ^ {k} (s, a _ {0}) \\ \frac {\exp \left(\check {v} _ {h} ^ {k} (s , a)\right)}{1 + \sum_ {a ^ {\prime} \in A \backslash \{a _ {0} \}} \exp \left(\check {v} _ {h} ^ {k} (s , a ^ {\prime})\right)}, & \text { otherwise. } \end{array} \right. \tag {D.21}
$$

Fix $s \in \mathcal{S}$ and $A \in \mathcal{A}$ . We now present the proof by considering two cases: (i) $f_{h,j}^{k}(s, a_0) > f_{h,j}^{k}(s, a)$ for all $a \in A$ and (ii) $\exists a \in A \setminus \{a_0\}$ such that $f_{h,j}^{k}(s, a_0) \leqslant f_{h,j}^{k}(s, a)$ .

Case (i) $f_{h,j}^{k}(s, a_0) > f_{h,j}^{k}(s, a)$ for all $a \in A$ .

We denote $\tilde{a} \in \operatorname{argmax}_{a \in A \setminus \{a_0\}} f_{h,j}^k(s, a)$ . Let $\tilde{A} = \{a_0, \tilde{a}\}$ . Since $f_{h,j}^k(s, a_0) > f_{h,j}^k(s, a)$ for all $a \in A$ and $a_0$ is always included in $A$ , removing any item $a \in A \setminus \{a_0\}$ from $A$ increases the expected value of $f_{h,j}^k$ . Thus, we get

$$
\sum_ {a \in A} \mathcal {P} _ {h} (a | s, A) f _ {h, j} ^ {k} (s, a) \leqslant \sum_ {a \in \tilde {A}} \mathcal {P} _ {h} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a)
$$

$$
\text { and } \sum_ {a \in A} \widetilde {\mathcal {P}} _ {h, j ^ {\prime}} ^ {k} (a | s, A) f _ {h, j} ^ {k} (s, a) \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j ^ {\prime}} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a). \tag {D.22}
$$

By the definition of $\widetilde{\mathcal{P}}_{h,j}^{k}$ in (7), we use the pessimistic utility $\check{v}_h^k(s,a)$ in this case. Since $\check{v}_h^k(s,a) \leqslant \phi(s,a)^\top \boldsymbol{\theta}_h^\star$ by Lemma D.4, using this utility, $\check{v}_h^k(s,a)$ , reduces the probability of selecting $\tilde{a}$ (compared to the true choice probability $\mathcal{P}_h$ ). Moreover, we know that $f_h^k(s,a_0) \geqslant f_h^k(s,\tilde{a})$ , we have

$$
\sum_ {a \in \tilde {A}} \mathcal {P} _ {h} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a) \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a). \tag {D.23}
$$

Furthermore, if $\widetilde{P}_{h,j'}^{k}$ is constructed using the pessimistic utility $\check{v}_{h}^{k}(s,a)$ , then, $\widetilde{P}_{h,j'}^{k} = \widetilde{P}_{h,j}^{k}$ . However, if $\widetilde{P}_{h,j'}^{k}$ is constructed using the optimistic utility $\widetilde{v}_{h}^{k}(s,a)$ , replacing $\widetilde{v}_{h}^{k}(s,a)$ with $\check{v}_{h}^{k}(s,a)$ (which is equivalent to replacing $\widetilde{P}_{h,j'}^{k}$ with $\widetilde{P}_{h,j}^{k}$ ) decreases the probability of choosing $\tilde{a}$ , meaning increase the expected value of $f_{h,j}^{k}$ . Thus, we get

$$
\sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j ^ {\prime}} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a) \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a). \tag {D.24}
$$

Combining (D.22), (D.23), and (D.24), we have

$$
\max \left\{\sum_ {a \in A} \mathcal {P} _ {h} (a | s, A) f _ {h, j} ^ {k} (s, a), \sum_ {a \in A} \widetilde {\mathcal {P}} _ {h, j ^ {\prime}} ^ {k} (a | s, A) f _ {h, j} ^ {k} (s, a) \right\} \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a).
$$

Case (ii) $\exists a\in A\backslash \{a_0\}$ such that $f_{h,j}^{k}(s,a_{0})\leqslant f_{h,j}^{k}(s,a)$ .

Let $\tilde{A} = \left\{A' \subseteq A : f_{h,j}^k(s, a) \geqslant f_{h,j}^k(s, a_0), \forall a \in A'\right\}$ . Note that $|\tilde{A}| \geqslant 2$ and $a_0 \in \tilde{A}$ by definition of action space $\mathcal{A}$ . By selecting $\tilde{A}$ instead of $A$ , we exclude items with the small values of $f_{h,j}^k(s, a)$ , thereby increasing the expected value of $f_{h,j}^k$ , i.e.,

$$
\sum_ {a \in A} \mathcal {P} _ {h} (a | s, A) f _ {h, j} ^ {k} (s, a) \leqslant \sum_ {a \in \tilde {A}} \mathcal {P} _ {h} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a)
$$

$$
\text { and } \sum_ {a \in A} \widetilde {\mathcal {P}} _ {h, j ^ {\prime}} ^ {k} (a | s, A) f _ {h, j} ^ {k} (s, a) \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j ^ {\prime}} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a). \tag {D.25}
$$

By the definition of $\widetilde{\mathcal{P}}_h^k$ in (7), we use the optimistic utility $\widetilde{v}_h^k(s,a)$ in this case. Since $\phi(s,a)^\top\boldsymbol{\theta}_h^\star \leqslant \widetilde{v}_h^k(s,a)$ by Lemma D.4, this choice of utility increases the probability of choosing item $a \neq \tilde{A} \backslash \{a_0\}$ compared to the true $\mathcal{P}_h$ ), implying

that

$$
\sum_ {a \in \tilde {A}} \mathcal {P} _ {h} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a) \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a). \tag {D.26}
$$

Moreover, if $\widetilde{\mathcal{P}}_{h,j'}^k$ is constructed using the pessimistic utility $\check{v}_h^k(s,a)$ , replacing $\check{v}_h^k(s,a)$ with $\widetilde{v}_h^k(s,a)$ (which is equivalent to replacing $\widetilde{\mathcal{P}}_{h,j'}^k$ with $\widetilde{\mathcal{P}}_{h,j}^k$ ) increases the probability of choosing item $a \neq \tilde{A} \backslash \{a_0\}$ . However, if $\widetilde{\mathcal{P}}_{h,j'}^k$ is constructed using the optimistic utility $\check{v}_h^k(s,a)$ , we have $\widetilde{\mathcal{P}}_{h,j}^k = \widetilde{\mathcal{P}}_{h,j'}^k$ . To this end, we get

$$
\sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j ^ {\prime}} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a) \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a). \tag {D.27}
$$

Combining (D.25), (D.26), and (D.27), we have

$$
\max \left\{\sum_ {a \in A} \mathcal {P} _ {h} (a | s, A) f _ {h, j} ^ {k} (s, a), \sum_ {a \in A} \widetilde {\mathcal {P}} _ {h, j ^ {\prime}} ^ {k} (a | s, A) f _ {h, j} ^ {k} (s, a) \right\} \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, \tilde {A}) f _ {h, j} ^ {k} (s, a).
$$

This concludes the proof of Lemma D.5.

![](images/d56b4fba59fe2cadb2ac33562de574c3990cfca233663b9502276ab1f5ee8478.jpg)

Denote $J(k, h) \in \{1, 2\}$ as the chosen index of function $f_{h,j}^{k}$ at $(k, h)$ . Then, we show that the value estimates for the chosen assortment, $f_{h,J(k,h)}^{k}(s_{h}^{k}, a)$ for all $a \in A_{h}^{k}$ , are greater than equal to $\sum_{a \in A_{h}^{k}} \mathcal{P}_{h,J(k,h)}^{k}(a|s, A_{h}^{k}) f_{h,J(k,h)}^{k}(s_{h}^{k}, a)$ .

Lemma D.6. For any $(k,h)\in[K]\times[H]$ , let $J(k,h):\mathcal{K}\times[H]\to\{1,2\}$ be the one-to-one function that maps from $K\times[H]$ to the index set $\{1,2\}$ such that $A_{h}^{k}=A_{h,J(k,h)}^{k}\in\operatorname{argmax}_{A\in\mathcal{A}}\sum_{a\in A}\widetilde{\mathcal{P}}_{h,J(k,h)}^{k}(a|s_{h}^{k},A)f_{h,J(k,h)}^{k}(s_{h}^{k},a)$ . Then, under the good event $E^{\theta}$ defined in (D.16), we have

$$
f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \geqslant \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a), \quad \forall a \in A _ {h} ^ {k}.
$$

Proof of Lemma D.6. By Lemma D.5, there exists a subset $\tilde{A} \subseteq A_h^k$ and $\tilde{A} \in \mathcal{A}$ such that

$$
\begin{array}{l} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \leqslant \sum_ {a \in \tilde {A}} \widetilde {\mathcal {P}} _ {h, J (k, h)} ^ {k} (a | s _ {h} ^ {k}, \tilde {A}) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \\ \leqslant \sum_ {a \in A _ {h} ^ {k}} \widetilde {\mathcal {P}} _ {h, J (k, h)} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a), \\ \end{array}
$$

where the second inequality holds the assortment selection rule. Thus, it is sufficient to show that $f_{h,J(k,h)}^{k}(s_{h}^{k},a) \geqslant \sum_{a \in A_{h}^{k}} \tilde{\mathcal{P}}_{h,J(k,h)}^{k}(a|s_{h}^{k},A_{h}^{k})f_{h,J(k,h)}^{k}(s_{h}^{k},a)$ for all $a \in A_{h}^{k}$ .

We prove this by contradiction. Suppose there exists an item $a \in A_{h}^{k}$ such that

$$
f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) <   \sum_ {a \in A _ {h} ^ {k}} \widetilde {\mathcal {P}} _ {h, J (k, h)} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a).
$$

If we remove item $a$ from the assortment $A_h^k$ , it would result in higher value. This contradicts the optimality of $A_h^k$ . Hence, we conclude

$$
f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \geqslant \sum_ {a \in A _ {h} ^ {k}} \widetilde {\mathcal {P}} _ {h, J (k, h)} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a),
$$

which completes the proof.

![](images/994161ac528efe74fe247138daa7b5dad68b91937abe79e85e3d4f6b6e287ea6.jpg)

Lemma D.7 is an elliptical potential lemma used for bounding the regret incurred from the MNL preference model (Lemma D.10 and D.13).

Lemma D.7 (Elliptical potential lemma, Lemma E.2 and H.3 of Lee & Oh 2024). Assume that $\lambda \geqslant 2$ and $\phi(s, a_0) = 0$ for all $s \in S$ . For any $(k, h, a) \in [K] \times [H] \times \mathcal{I}$ , we define $\widetilde{\phi}(s_h^k, a) = \phi(s_h^k, a) - \mathbb{E}_{a' \sim \mathcal{P}_h(\cdot | s_h^k, A_h^k; \theta_h^{k+1})}[\phi(s_h^k, a)]$ . Then, for $\mathbf{H}_h^k$ defined in (D.3), and for any $h \in [H]$ , the following statements hold true:

$$
\begin{array}{l} \sum_ {\tau = 1} ^ {k} \sum_ {a \in A _ {h} ^ {\tau}} \mathcal {P} _ {h} \big (a | s _ {h} ^ {\tau}, A _ {h} ^ {\tau}; \boldsymbol {\theta} _ {h} ^ {\tau + 1} \big) \mathcal {P} _ {h} \big (a _ {0} | s _ {h} ^ {\tau}, A _ {h} ^ {\tau}; \boldsymbol {\theta} _ {h} ^ {\tau + 1} \big) \| \phi (s _ {h} ^ {\tau}, a) \| _ {\big (\mathbf {H} _ {h} ^ {\tau} \big) ^ {- 1}} ^ {2} \leqslant 2 d \log \left(1 + \frac {k}{d \lambda}\right), \\ \sum_ {\tau = 1} ^ {k} \sum_ {a \in A _ {h} ^ {\tau}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {\tau}, A _ {h} ^ {\tau}; \boldsymbol {\theta} _ {h} ^ {\tau + 1}\right) \| \widetilde {\phi} (s _ {h} ^ {\tau}, a) \| _ {\left(\mathbf {H} _ {h} ^ {\tau}\right) ^ {- 1}} ^ {2} \leqslant 2 d \log \left(1 + \frac {k}{d \lambda}\right), \\ \sum_ {\tau = 1} ^ {k} \max \left\{\max _ {a \in A _ {h} ^ {\tau}} \| \phi (s _ {h} ^ {\tau}, a) \| _ {(\mathbf {H} _ {h} ^ {\tau}) ^ {- 1}} ^ {2}, \max _ {a \in A _ {h} ^ {\tau}} \| \widetilde {\phi} (s _ {h} ^ {\tau}, a) \| _ {(\mathbf {H} _ {h} ^ {\tau}) ^ {- 1}} ^ {2} \right\} \leqslant \frac {2}{\kappa} d \log \left(1 + \frac {k}{d \lambda}\right). \\ \end{array}
$$

Lemma D.8 is used to derive the tight bound for the second-order regret term of the MNL preference model (LemmaD.13).

Lemma D.8 (Lemma E.3 of Lee & Oh 2024). Let $M \in \mathbf{Z}^{+}$ . Define $R: \mathbb{R}^{M} \to \mathbb{R}$ , such that for any $\boldsymbol{v} = (v_{1}, \ldots, v_{M}) \in \mathbb{R}^{M}$ , $R(\boldsymbol{v}) = \sum_{m=1}^{M} \frac{\exp(v_{m})}{1 + \sum_{l=1}^{M} \exp(v_{l})}$ . Let $p_{m}(\boldsymbol{v}) = \frac{\exp(v_{m})}{1 + \sum_{l=1}^{M} \exp(v_{l})}$ . Then, for all $m \in [M]$ , we have

$$
\left| \frac {\partial^ {2} R}{\partial m \partial n} \right| \leqslant \left\{ \begin{array}{l l} 3 p _ {m} (\boldsymbol {v}) & \text { if } m = n, \\ 2 p _ {m} (\boldsymbol {v}) p _ {n} (\boldsymbol {v}) & \text { if } m \neq n. \end{array} \right.
$$

Lemma D.9 is crucial for deriving the $\kappa$ -improved bound for the MNL preference model (LemmaD.13), enabling the analysis.

Lemma D.9 (Overly optimistic choice probability). We define

$$
\tilde {\mathcal {P}} _ {h, j} ^ {k} (a | s, A) = \left\{ \begin{array}{l l} \frac {\exp \left(\phi (s , a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} + 2 \alpha_ {h} ^ {k} \| \phi (s , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}\right)}{\sum_ {a ^ {\prime} \in A} \exp \left(\phi (s , a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} + 2 \alpha_ {h} ^ {k} \| \phi (s , a ^ {\prime}) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}\right)}, & i f \exists a \in \mathcal {I} \backslash \{a _ {0} \} s. t. \\ \frac {\exp \left(\phi (s , a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} - 2 \alpha_ {h} ^ {k} \| \phi (s , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}\right)}{\sum_ {a ^ {\prime} \in A} \exp \left(\phi (s , a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} - 2 \alpha_ {h} ^ {k} \| \phi (s , a ^ {\prime}) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}\right)}, & o t h e r w i s e. \end{array} \right. \tag {D.28}
$$

Let $A_{h,j}^{k}\in \mathrm{argmax}_{A\in \mathcal{A}}\sum_{a\in A}\widetilde{\mathcal{P}}_{h,j}^{k}(a|s_{h}^{k},A)f_{h,j}^{k}(s_{h}^{k},a)$ , where $j\in \{1,2\}$ . Then, under the good event $\mathcal{E}^{\theta}$ , for all $(k,h,j)\in [K]\times [H]\times \{1,2\}$ , we have

$$
\sum_ {a \in A _ {h, j} ^ {k}} \tilde {\mathcal {P}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h, j} ^ {k}) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a) \leqslant \sum_ {a \in A _ {h, j} ^ {k}} \tilde {\bar {\mathcal {P}}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h, j} ^ {k}) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a).
$$

Proof of Lemma D.9. Fix $(k,h,j)\in[K]\times[H]\times\{1,2\}$ . We consider the two cases: (i) $f_{h,j}^{k}(s_{h}^{k},a_{0})>f_{h,j}^{k}(s_{h}^{k},a)$ for all $a\in I$ and (ii) $\exists a\in\mathcal{I}\backslash\{a_{0}\}$ such that $f_{h,j}^{k}(s_{h}^{k},a_{0})\leqslant f_{h,j}^{k}(s_{h}^{k},a)$ .

Case (i) $f_{h,j}^{k}(s_{h}^{k},a_{0}) > f_{h,j}^{k}(s_{h}^{k},a)$ for all $a\in \mathcal{I}$ .

Recall that, by the definition of $\widetilde{P}_{h,j}^{k}$ in (7), we use the pessimistic utility $\check{v}_{h}^{k}(s,a)$ to construct $\widetilde{P}_{h,j}^{k}$ in this case. Note that the outside option $a_{0}$ must be included in the assortment, i.e., $a_{0}\in A_{h}^{k}$ . Moreover, under the event $E^{\theta}$ , by Lemma D.4, we have

$$
\check {v} _ {h} ^ {k} (s _ {h} ^ {k}, a) \geqslant \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} - 2 \alpha_ {h} ^ {k} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}.
$$

Thus, since we assume, without loss of generality, that $\phi(s_h^k, a_0) = 0$ (refer (D.21)), using $\tilde{\mathcal{P}}_{h,j}^k$ instead of $\tilde{\mathcal{P}}_{h,j}^k$ decreases the probability of choosing any item $a \in A_h^k \backslash \{a_0\}$ . As a result, the expected value of $f_{h,j}^k$ increases, since $f_{h,j}^k(s, a_0) \geqslant$

$f_{h,j}^{k}(s,a)$ for all $a\in A_h^k$ . Formally, we have

$$
\sum_ {a \in A _ {h, j} ^ {k}} \tilde {\mathcal {P}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h, j} ^ {k}) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a) \leqslant \sum_ {a \in A _ {h, j} ^ {k}} \tilde {\bar {\mathcal {P}}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h, j} ^ {k}) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a).
$$

Case (ii) $\exists a\in \mathcal{I}\backslash \{a_0\}$ such that $f_{h,j}^{k}(s_{h}^{k},a_{0})\leqslant f_{h,j}^{k}(s_{h}^{k},a)$ .

First, we show that for all $a \in A_{h}^{k} \backslash \{a_{0}\}$ , we have $f_{h,j}^{k}(s_{h}^{k}, a) \geqslant \sum_{a \in A_{h,j}^{k}} \widetilde{\mathcal{P}}_{h,j}^{k}(a | s_{h}^{k}, A_{h,j}^{k}) f_{h,j}^{k}(s_{h}^{k}, a)$ . Suppose that there exists $a \in A_{h}^{k} \backslash \{a_{0}\}$ for which $f_{h,j}^{k}(s_{h}^{k}, a) < \sum_{a \in A_{h,j}^{k}} \widetilde{\mathcal{P}}_{h,j}^{k}(a | s_{h}^{k}, A_{h,j}^{k}) f_{h,j}^{k}(s_{h}^{k}, a)$ . Then, removing item a from the assortment $A_{h}^{k}$ results in the increase in the expected value of $f_{h,j}^{k}$ . Consequently, this contradicts the optimality of $A_{h}^{k}$ . Hence, we get

$$
f _ {h, j} ^ {k} (s _ {h} ^ {k}, a) \geqslant \sum_ {a \in A _ {h, j} ^ {k}} \widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h, j} ^ {k}) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a), \quad \forall a \in A _ {h} ^ {k} \backslash \{a _ {0} \}.
$$

On the other hand, recall that, by the definition of $\widetilde{P}_{h,j}^{k}$ in (7), we use the pessimistic utility $\widetilde{v}_{h}^{k}(s,a)$ to construct $\widetilde{P}_{h,j}^{k}$ in this case. Furthermore, by Lemma D.4, we know that

$$
\widetilde {v} _ {h} ^ {k} (s _ {h} ^ {k}, a) \leqslant \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} + 2 \alpha_ {h} ^ {k} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}.
$$

If we increase $\widetilde{v}_h^k (s_h^k,a)$ to $\phi (s_h^k,a)^{\top}\pmb{\theta}_h^\star +2\alpha_h^k\| \phi (s_h^k,a)\|_{\left(\mathbf{H}_h^k\right)^{-1}}$ for all $a\in A_h^k\backslash \{a_0\}$ , the probability of choosing the outside option decreases (because $\phi (s_h^k,a_0) = 0$ ). In other words, the sum of probabilities of choosing $a\in A_h^k\backslash \{a_0\}$ increases. Since $f_{h,j}^{k}(s_{h}^{k},a)\geqslant \sum_{a\in A_{h,j}^{k}}\widetilde{\mathcal{P}}_{h,j}^{k}(a|s_{h}^{k},A_{h,j}^{k})f_{h,j}^{k}(s_{h}^{k},a)$ for all $a\in A_h^k\backslash \{a_0\}$ , the expected value of $f_{h,j}^{k}$ increases. Formally, we get

$$
\sum_ {a \in A _ {h, j} ^ {k}} \tilde {\mathcal {P}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h, j} ^ {k}) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a) \leqslant \sum_ {a \in A _ {h, j} ^ {k}} \tilde {\bar {\mathcal {P}}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h, j} ^ {k}) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a).
$$

This concludes the proof.

Lemma D.10 will be used to carefully bound the sum of $b_{h,1}^{k}$ (Lemma D.23). Note that the following MNL bandit regret improves upon the one proposed in (Oh & Iyengar, 2021) by a factor of $1/\sqrt{\kappa}$ , which can be exponentially large.

Lemma D.10 (Crude bound for MNL bandits). For any $h \in [H]$ , $j = \{1, 2, -2\}$ and subset $K \in [K]$ , under the good event $E^{\theta}$ defined in (D.16), we have

$$
\sum_ {k \in \mathcal {K}} \left| \sum_ {a \in A _ {h} ^ {k}} \Bigl (\widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) \Bigr) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a) \right| \leqslant \mathcal {O} \left(\frac {1}{\sqrt {\kappa}} d \sqrt {| \mathcal {K} |} \cdot (\log K) ^ {3 / 2} \log M\right),
$$

where M is the maximum size of the assortment.

Proof of Lemma D.10. We denote $M_h^k$ as the size of the assortment at horizon $h$ in episode $k$ , i.e., $M_h^k = |A_h^k|$ . For any $j \in \{1,2,-2\}$ , we define a function $R_j: \mathbb{R}^{M_h^k} \to \mathbb{R}$ such that, for all $\boldsymbol{v} \in \mathbb{R}^{M_h^k}$ , $R_j(\boldsymbol{v}) = \sum_{m=1}^{M_h^k} \frac{\exp(v_m)f_{h,j}^{k}(s_h^k,a_{im})}{1 + \sum_{l=1}^{M_h^k}\exp(v_l)}$ .

For simplicity, we denote $v_{h,j}^{k}(s,a)$ as the utility, which can represent either the optimistic utility $\widetilde{v}_{h}^{k}(s,a)$ or the pessimistic utility $\check{v}_{h}^{k}(s,a)$ , as determined by (7), depending on $f_{h,j}^{k}$ . Let $\boldsymbol{v}_{h,j}^{k}(s_{h}^{k}) = \left(v_{h,j}^{k}(s_{h}^{k},a)\right)_{a \in A_{h}^{k}} \in \mathbb{R}^{M_{h}^{k}}$ and $\boldsymbol{v}_{h}^{\star}(s_{h}^{k}) = \left(\phi(s_{h}^{k},a)^{\top}\boldsymbol{\theta}_{h}^{\star}\right)_{a \in A_{h}^{k}} \in \mathbb{R}^{M_{h}^{k}}$ . Then, by the mean value theorem, there exists a vector $\bar{\boldsymbol{v}}_{h,j}^{k}(s_{h}^{k})$ , which is a convex combination of $\boldsymbol{v}_{h,j}^{k}(s_{h}^{k})$ and $\boldsymbol{v}_{h}^{\star}(s_{h}^{k})$ , such that

$$
\begin{array}{l} \sum_ {k \in \mathcal {K}} \left| \sum_ {a \in A _ {h} ^ {k}} \Bigl (\widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) \Bigr) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a) \right| = \sum_ {k \in \mathcal {K}} \left| R _ {j} \left(\boldsymbol {\upsilon} _ {h, j} ^ {k} (s _ {h} ^ {k})\right) - R _ {j} \left(\boldsymbol {\upsilon} _ {h} ^ {\star} (s _ {h} ^ {k})\right) \right| \\ = \sum_ {k \in \mathcal {K}} \left| \nabla R _ {j} \left(\bar {\boldsymbol {v}} _ {h, j} ^ {k} (s _ {h} ^ {k})\right) ^ {\top} \left(\boldsymbol {v} _ {h, j} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) \right|. \\ \end{array}
$$

Therefore, we get

$$
\begin{array}{l} \sum_ {k \in \mathcal {K}} \left| \nabla R _ {j} \left(\bar {\boldsymbol {v}} _ {h, j} ^ {k} (s _ {h} ^ {k})\right) ^ {\top} \left(\boldsymbol {v} _ {h, j} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) \right| \\ = \sum_ {k \in \mathcal {K}} \left| \sum_ {a \in A _ {h} ^ {k}} \frac {\exp \left(\bar {v} _ {h , j} ^ {k} (s _ {h} ^ {k} , a)\right) f _ {h , j} ^ {k} (s _ {h} ^ {k} , a)}{\sum_ {a ^ {\prime \prime} \in A _ {h} ^ {k}} \exp \left(\bar {v} _ {h , j} ^ {k} (s _ {h} ^ {k} , a ^ {\prime \prime})\right)} \left(v _ {h, j} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \right. \\ - \sum_ {a \in A _ {h} ^ {k}} \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \frac {\exp \left(\bar {v} _ {h , j} ^ {k} (s _ {h} ^ {k} , a)\right) f _ {h , j} ^ {k} (s _ {h} ^ {k} , a) \exp \left(\bar {v} _ {h , j} ^ {k} (s _ {h} ^ {k} , a ^ {\prime})\right)}{\left(\sum_ {a ^ {\prime \prime} \in A _ {h} ^ {k}} \exp \left(\bar {v} _ {h , j} ^ {k} (s _ {h} ^ {k} , a ^ {\prime \prime})\right)\right) ^ {2}} \left(v _ {h, j} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) - \phi (s _ {h} ^ {k}, a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \Bigg | \\ = \sum_ {k \in \mathcal {K}} \left| \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \bar {\boldsymbol {v}} _ {h, j} ^ {k} (s _ {h} ^ {k})\right) \left(v _ {h, j} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \right. \\ \cdot \left(f _ {h, j} ^ {k} (s _ {h} ^ {k}, a) - \mathbb {E} _ {a ^ {\prime} \sim \mathcal {P} _ {h} \big (\cdot | s _ {h} ^ {k}, A _ {h} ^ {k}; \tilde {\boldsymbol {v}} _ {h, j} ^ {k} (s _ {h} ^ {k}) \big)} \left[ f _ {h, j} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) \right]\right) \\ \leqslant 2 \sum_ {k \in \mathcal {K}} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \bar {\boldsymbol {v}} _ {h, j} ^ {k} (s _ {h} ^ {k})\right) \left| \left(\upsilon_ {h, j} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a)\right) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} \right|, \tag {D.29} \\ \end{array}
$$

where the inequality is from $f_{h,j}^{k} \leqslant 1$ . Recall that $v_{h,j}^{k}(s_{h}^{k}, a)$ can be either $\widetilde{v}_{h}^{k}(s_{h}^{k}, a)$ or $\check{v}_{h}^{k}(s_{h}^{k}, a)$ . Then, by Lemma D.4, we have $\left|\left(v_{h,j}^{k}(s_{h}^{k}, a) - \phi(s_{h}^{k}, a)^{\top}\boldsymbol{\theta}_{h}^{\star}\right| \leqslant 2\alpha_{h}^{k}\|\phi(s_{h}^{k}, a)\|_{\left(\mathbf{H}_{h}^{k}\right)^{-1}}\right.$ . Hence, we can further bound the right-hand side of (D.29).

$$
\begin{array}{l} 2 \sum_ {k \in \mathcal {K}} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \bar {\boldsymbol {v}} _ {h, j} ^ {k} (s _ {h} ^ {k})\right) \left| \left(v _ {h, j} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a)\right) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} \right| \\ \leqslant 4 \alpha_ {h} ^ {K} \sum_ {k \in \mathcal {K}} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \bar {\boldsymbol {v}} _ {h, j} ^ {k} (s _ {h} ^ {k})\right) \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \\ \leqslant 4 \alpha_ {h} ^ {K} \sqrt {\sum_ {k \in \mathcal {K}} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \bar {\boldsymbol {v}} _ {h , j} ^ {k} (s _ {h} ^ {k})\right)} \sqrt {\sum_ {k \in \mathcal {K}} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \bar {\boldsymbol {v}} _ {h , j} ^ {k} (s _ {h} ^ {k})\right) \| \phi (s _ {h} ^ {k} , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}} \\ \leqslant 4 \alpha_ {h} ^ {K} \sqrt {| \mathcal {K} |} \cdot \sqrt {\sum_ {k = 1} ^ {K} \sum_ {a \in A _ {h} ^ {k}} \| \phi (s _ {h} ^ {k} , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}} \\ = 4 \alpha_ {h} ^ {K} \sqrt {| \mathcal {K} |} \cdot \sqrt {\sum_ {k = 1} ^ {K} \sum_ {a \in A _ {h} ^ {k}} \frac {\mathcal {P} _ {h} (a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}) \mathcal {P} _ {h} (a _ {0} | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1})}{\mathcal {P} _ {h} (a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}) \mathcal {P} _ {h} (a _ {0} | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k+ 1})} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}} \\ \leqslant 4 \alpha_ {h} ^ {K} \sqrt {| \mathcal {K} |} \cdot \sqrt {\frac {1}{\kappa} \cdot \sum_ {k = 1} ^ {K} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}) \mathcal {P} _ {h} (a _ {0} | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}) \| \phi (s _ {h} ^ {k} , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}} \\ \leqslant 4 \alpha_ {h} ^ {K} \sqrt {| \mathcal {K} |} \cdot \sqrt {\frac {1}{\kappa} \cdot 2 d \log \left(1 + \frac {K}{d \lambda}\right)}, \tag {D.30} \\ \end{array}
$$

where the first inequality holds because $\alpha_{h}^{1}\leqslant\cdots\leqslant\alpha_{h}^{K}$ , the second inequality follows from the Cauchy-Schwarz inequality, the second-to-the last inequality holds due to the definition of $\kappa$ , and the last inequality holds by Lemma D.7.

Combining (D.29) and (D.30), and plugging in the value of $\alpha_{h}^{k}$ , we derive that

$$
\sum_ {k \in \mathcal {K}} \left| \sum_ {a \in A _ {h} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h, j} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k})\right) f _ {h, j} ^ {k} (s _ {h} ^ {k}, a) \right| = \mathcal {O} \left(\frac {1}{\sqrt {\kappa}} d \sqrt {| \mathcal {K} |} \cdot (\log K) ^ {3 / 2} \log M\right).
$$

Lemma D.11. For any $(k,h)\in[K]\times[H]$ , $\theta_{1},\theta_{2}\in C_{h}^{k}$ , and $\omega_{h}^{k}(a)\geqslant0$ , under the event $E^{\theta}$ defined in (D.16), we have

$$
\sum_ {a \in A _ {h} ^ {k}} \left| \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {1}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {1}) \right| \omega_ {h} ^ {k} (a) \leqslant 4 \alpha_ {h} ^ {k} \max _ {a \in A _ {h} ^ {k}} \omega_ {h} ^ {k} (a) \max _ {a \in A _ {h} ^ {k}} \| \phi (s _ {h} ^ {k}, a) \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}}.
$$

Proof of Lemma D.11. By the mean value theorem, there exists $\pmb{\xi} = (1 - c)\pmb{\theta}_1 + c\pmb{\theta}_2$ for some $c \in (0,1)$ such that

$$
\begin{array}{l} \sum_ {a \in A _ {h} ^ {k}} \left| \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \pmb {\theta} _ {1}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \pmb {\theta} _ {1}) \right| \omega_ {h} ^ {k} (a) = \sum_ {a \in A _ {h} ^ {k}} \left| \nabla \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \pmb {\xi}) ^ {\top} (\pmb {\theta} _ {1} - \pmb {\theta} _ {2}) \right| \omega_ {h} ^ {k} (a) \\ = \sum_ {a \in A _ {h} ^ {k}} \left| \left(\mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\xi}) \phi (s _ {h} ^ {k}, a) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\xi}) \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a ^ {\prime} | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\xi}) \phi (s _ {h} ^ {k}, a ^ {\prime})\right) ^ {\top} (\boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {2}) \right| \omega_ {h} ^ {k} (a) \\ \leqslant \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\xi}) \left| \phi (s _ {h} ^ {k}, a) ^ {\top} (\boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {2}) \right| \omega_ {h} ^ {k} (a) + \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\xi}) \omega_ {h} ^ {k} (a) \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a ^ {\prime} | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\xi}) \left| \phi (s _ {h} ^ {k}, a) ^ {\top} (\boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {2}) \right| \\ \leqslant 2 \alpha_ {h} ^ {k} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\xi}) \| \phi (s _ {h} ^ {k}, a) \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}} \omega_ {h} ^ {k} (a) + \max _ {a \in A _ {h} ^ {k}} \omega_ {h} ^ {k} (a) \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\xi}) \| \phi (s _ {h} ^ {k}, a) \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}} \\ \leqslant 4 \alpha_ {h} ^ {k} \max _ {a \in A _ {h} ^ {k}} \omega_ {h} ^ {k} (a) \max _ {a \in A _ {h} ^ {k}} \| \phi (s _ {h} ^ {k}, a) \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}}, \\ \end{array}
$$

where the second-to-last inequality holds under the good event $E^{\theta}$ defined in (D.16).

![](images/9e83c017d6c5b2facda290b3f800ab140d5645b7aa7c4edef103c99bd602a921.jpg)

Lemma D.12 pertains to the law of total variance (Lattimore & Hutter, 2012; Gheshlaghi Azar et al., 2013) that the variance of the value function is smaller than its magnitude by a factor $\sqrt{H}$ .

Lemma D.12 (Total variance lemma, Lemma C.5 of Jin et al. 2018). Let $f_{h,j}^{k} \in [0,1]$ . Then, with probability at least $1 - \delta$ , we have

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \left[ \mathbb {V} _ {h} f _ {h, j} ^ {k} \right] \left(s _ {h} ^ {k}\right) = \mathcal {O} \left(K + H \log (1 / \delta)\right).
$$

Lemma D.13 is crucial for obtaining a $\kappa$ -independent regret in our leading term. While the proof is largely inspired by Lee & Oh (2024), extending their result to our setting is non-trivial because the unknown item values $f_{h,j}^{k}$ add complexity to the analysis. Moreover, thanks to Lemma D.12, we can obtain a tight bound by a factor of $\sqrt{H}$ , instead of naively summing over H MNL bandit regrets.

Lemma D.13 ( $\kappa$ -improved bound for MNL bandits). For any subset $K \in [K]$ , let $J(k,h) : \mathcal{K} \times [H] \to \{1,2\}$ be the one-to-one function that maps from $K \times [H]$ to the index set $\{1,2\}$ such that $A_{h}^{k} = A_{h,J(k,h)}^{k} \in \operatorname{argmax}_{A \in \mathcal{A}} \sum_{a \in A} \widetilde{\mathcal{P}}_{h,J(k,h)}^{k}(a|s_{h}^{k}, A)f_{h,J(k,h)}^{k}(s_{h}^{k}, a)$ . Then, under the good event $E^{\theta}$ defined in (D.16), with probability at least $1 - \delta$ , we have

$$
\begin{array}{l} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \Bigl (\widetilde {\mathcal {P}} _ {h, J (k, h)} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) \Bigr) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \\ = \mathcal {O} \left(d \sqrt {H | \mathcal {K} |} (\log K) ^ {3 / 2} \log M + \frac {1}{\kappa} d ^ {2} H (\log K) ^ {3} (\log M) ^ {2}\right). \\ \end{array}
$$

Proof of Lemma D.13. We begin by defining $\tilde{\mathcal{P}}_{h,j}^k (a|s,A)$ as given in D.28. Then, by Lemma D.9, we have

$$
\begin{array}{l} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h, J (k, h)} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k})\right) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \\ \leqslant \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \Bigl (\widetilde {\tilde {\mathcal {P}}} _ {h, J (k, h)} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) \Bigr) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a). \\ \end{array}
$$

We denote $M_{h}^{k}$ as the size of the assortment at horizon h in episode k, i.e., $M_{h}^{k} = |A_{h}^{k}|$ . We define a function $\tilde{R}: R^{M_{h}^{k}} \to R$ such that for all $v \in R^{M_{h}^{k}}$ , $\tilde{R}(v) = \sum_{m=1}^{M_{h}^{k}} \frac{\exp(v_{m}) f_{h,J(k,h)}^{k}(s_{h}^{k}, a_{im})}{1 + \sum_{l=1}^{M_{h}^{k}} \exp(v_{l})}$ .

For any $(k,h)\in\mathcal{K}\times[H]$ and all $a\in I$ , we denote $v_{h}^{k}(s_{h}^{k},a)$ as the utility, which can be either $\phi(s_{h}^{k},a)^{\top}\boldsymbol{\theta}_{h}^{\star}+2\alpha_{h}^{k}\|\phi(s_{h}^{k},a)\|_{(\mathbf{H}_{h}^{k})^{-1}}$ or $\phi(s_{h}^{k},a)^{\top}\boldsymbol{\theta}_{h}^{\star}-2\alpha_{h}^{k}\|\phi(s_{h}^{k},a)\|_{(\mathbf{H}_{h}^{k})^{-1}}$ , determined deterministically based on the history up to $(k,h)$ :

$$
v _ {h} ^ {k} (s _ {h} ^ {k}, a) = \left\{ \begin{array}{l l} \phi (s _ {h} ^ {k}, a) ^ {\top} \pmb {\theta} _ {h} ^ {\star} + 2 \alpha_ {h} ^ {k} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}, & \text {if} \exists a \in \mathcal {I} \backslash \{a _ {0} \} \text {s.t.} f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \geqslant f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a _ {0}) \\ \phi (s _ {h} ^ {k}, a) ^ {\top} \pmb {\theta} _ {h} ^ {\star} - 2 \alpha_ {h} ^ {k} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}, & \text {if} \forall a \in \mathcal {I} \backslash \{a _ {0} \} f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) <   f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a _ {0}). \end{array} \right.
$$

Let $\pmb{v}_h^k (s_h^k) = \left(v_h^k (s_h^k,a)\right)_{a\in A_h^k}\in \mathbb{R}^{M_h^k}$ and $\pmb{v}_h^\star (s_h^k) = \left(\phi (s_h^k,a)^\top \pmb{\theta}_h^\star\right)_{a\in A_h^k}\in \mathbb{R}^{M_h^k}$ . Thanks to exact second-order Taylor expansion, we obtain that

$$
\begin{array}{l} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \Bigl (\widetilde {\mathcal {P}} _ {h, J (k, h)} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) \Bigr) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \\ = \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \tilde {R} (\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k})) - \tilde {R} (\boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})) \\ = \underbrace {\sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \nabla \tilde {R} (\boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})) ^ {\top} \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right)} _ {\text {(A)}} + \underbrace {\frac {1}{2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) ^ {\top} \nabla^ {2} \tilde {R} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})) \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right)} _ {\text {(B)}}, \tag {D.31} \\ \end{array}
$$

where $\bar{\boldsymbol{v}}_h^k (s_h^k) = \left(\bar{v}_h^k (s_h^k,a)\right)_{a\in A_h^k}\in \mathbb{R}^{M_h^k}$ is the convex combination of $\boldsymbol{v}_h^k (s_h^k)$ and $\boldsymbol{v}_h^\star (s_h^k)$ .

We first bound the term (A) in (D.31).

$$
\begin{array}{l} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \nabla \tilde {R} (\boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})) ^ {\top} \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) \\ = \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \left(\sum_ {a \in A _ {h} ^ {k}} \frac {\exp \left(\phi (s _ {h} ^ {k} , a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) f _ {h , J (k , h)} ^ {k} (s _ {h} ^ {k} , a)}{\sum_ {a ^ {\prime \prime} \in A _ {h} ^ {k}} \exp \left(\phi (s _ {h} ^ {k} , a ^ {\prime \prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)} \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)\right) \\ - \sum_ {a \in A _ {h} ^ {k}} \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \frac {\exp \left(\phi (s _ {h} ^ {k} , a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) f _ {h , J (k , h)} ^ {k} (s _ {h} ^ {k} , a) \exp \left(\phi (s _ {h} ^ {k} , a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}{\left(\sum_ {a ^ {\prime \prime} \in A _ {h} ^ {k}} \exp \left(\phi (s _ {h} ^ {k} , a ^ {\prime \prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)\right) ^ {2}} \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) - \phi (s _ {h} ^ {k}, a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \\ = \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \\ \cdot \left(\left(v _ {h} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) - \sum_ {a ' \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a ^ {\prime} | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) - \phi (s _ {h} ^ {k}, a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)\right). \tag {D.32} \\ \end{array}
$$

We bound the right-hand side of (D.32) by examining two separate cases. For any fixed $h \in [H]$ , let $\mathcal{K}_{(i)}$ denote the set of episodes where Case (i) holds, and $\mathcal{K}_{(ii)}$ denote the set of episodes where Case (ii) holds. More formally, we define:

$$
\mathcal {K} _ {(i)} = \left\{k \in \mathcal {K}: v _ {h} ^ {k} (s _ {h} ^ {k}, a) = \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} + 2 \alpha_ {h} ^ {k} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \right\} \tag {Case(i)}
$$

$$
\mathcal {K} _ {(i i)} = \left\{k \in \mathcal {K}: v _ {h} ^ {k} (s _ {h} ^ {k}, a) = \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} - 2 \alpha_ {h} ^ {k} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \right\}. \tag {Case(ii)}
$$

$\mathbf{Case (i)}$ For $(k,h)\in \mathcal{K}\times [H]$ such that $v_{h}^{k}(s_{h}^{k},a) = \phi (s_{h}^{k},a)^{\top}\pmb{\theta}_{h}^{\star} + 2\alpha_{h}^{k}\| \phi (s_{h}^{k},a)\|_{\left(\mathbf{H}_{h}^{k}\right)^{-1}}.$

Denoting $\mathbb{E}_{\boldsymbol{\theta}}[\cdot] = \mathbb{E}_{a' \sim \mathcal{P}_h(\cdot | s_h^k, A_h^k; \boldsymbol{\theta})}[\cdot]$ and $\bar{\alpha}_K := \max_{h \in [H]} \alpha_h^K$ for simplicity, we get

$$
\sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \nabla \tilde {R} (\boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})) ^ {\top} \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right)
$$

$$
= \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} 2 \alpha_ {h} ^ {k} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \left(\| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \| \phi (s _ {h} ^ {k}, a ^ {\prime}) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \right]\right)
$$

$$
\leqslant 2 \bar {\alpha} _ {K} \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \left(\| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \| \phi (s _ {h} ^ {k}, a ^ {\prime}) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \right]\right)
$$

$$
= 2 \bar {\alpha} _ {K} \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \left(f _ {h, J (k, h)} ^ {k} \left(s _ {h} ^ {k}, a\right) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h, J (k, h)} ^ {k} \left(s _ {h} ^ {k}, a ^ {\prime}\right) \right]\right) \left(\| \phi \left(s _ {h} ^ {k}, a\right) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \| \phi \left(s _ {h} ^ {k}, a ^ {\prime}\right) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \right]\right) \right]
$$

$$
\leqslant 2 \bar {\alpha} _ {K} \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \left(\underbrace {f _ {h , J (k , h)} ^ {k} (s _ {h} ^ {k} , a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h , J (k , h)} ^ {k} (s _ {h} ^ {k} , a ^ {\prime}) \right]} _ {\geqslant 0}\right) \| \phi (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} [ \phi (s _ {h} ^ {k}, a ^ {\prime}) ] \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}} \right], \tag {D.33}
$$

where, in the first inequality, we use the fact that $\alpha_{h}^{K}$ is non-decreasing with respect to k, and by Lemma D.6, we have

$$
\begin{array}{l} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \Bigg (\| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \| \phi (s _ {h} ^ {k}, a ^ {\prime}) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \right] \Bigg) \\ = \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \Bigg (f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) - \underbrace {\mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h , J (k , h)} ^ {k} (s _ {h} ^ {k} , a ^ {\prime}) \right]} _ {= \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a ^ {\prime} | s _ {h} ^ {k}, A _ {h} ^ {k}) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a ^ {\prime})} \Bigg) \\ \geqslant 0. \\ \end{array}
$$

And the last inequality of Equation (D.33) holds because

$$
\left\| \phi (s _ {h} ^ {k}, a) \right\| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \left\| \phi (s _ {h} ^ {k}, a ^ {\prime}) \right\| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \right] \leqslant \left\| \phi (s _ {h} ^ {k}, a) \right\| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} - \left\| \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \phi (s _ {h} ^ {k}, a ^ {\prime}) \right] \right\| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}
$$

$$
\leqslant \left\| \phi (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \big [ \phi (s _ {h} ^ {k}, a ^ {\prime}) \big ] \right\| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}},
$$

where the first inequality holds by Jensen's inequality and the last inequality holds due to the fact that $\| \mathbf{a}\| = \| \mathbf{a} - \mathbf{b} + \mathbf{b}\| \leqslant \| \mathbf{a} - \mathbf{b}\| +\| \mathbf{b}\|$ for any vectors $\mathbf{a},\mathbf{b}\in \mathbb{R}^d$

We further decompose the right-hand side of (D.33) as follows:

$$
\begin{array}{l} \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \left(f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) \right]\right) \| \phi (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} [ \phi (s _ {h} ^ {k}, a ^ {\prime}) ] \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \right] \\ = \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \sqrt {\mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star}\right) \mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}\right)} \left(f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) \right]\right) \\ \cdot \left\| \phi (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {k + 1}} [ \phi (s _ {h} ^ {k}, a ^ {\prime}) ] \right\| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \\ + \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \left(\sqrt {\mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star}\right)} - \sqrt {\mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}\right)}\right) \sqrt {\mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star}\right)} \\ \cdot \left(f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) \right]\right) \| \phi (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {k + 1}} [ \phi (s _ {h} ^ {k}, a ^ {\prime}) ] \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \\ + \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) \left(f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) \right]\right) \\ \cdot \left(\| \phi (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {*}} [ \phi (s _ {h} ^ {k}, a ^ {\prime}) ] \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}} - \| \phi (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {k + 1}} [ \phi (s _ {h} ^ {k}, a ^ {\prime}) ] \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}}\right). \tag {D.34} \\ \end{array}
$$

For simplicity, let $\bar{\phi}(s_h^k,a) = \phi (s_h^k,a) - \mathbb{E}_{\pmb{\theta}_h^\star}\left[\phi (s_h^k,a')\right]$ and $\widetilde{\phi} (s_h^k,a) = \phi (s_h^k,a) - \mathbb{E}_{\pmb{\theta}_h^{k + 1}}\left[\phi (s_h^k,a')\right]$ . Now, we bound the terms on the right-hand side of (D.34) individually. For the first term, with probability at least $1 - \delta$ , we get

$$
\begin{array}{l} \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \sqrt {\mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star}\right) \mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}\right)} \left(f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) \right]\right) \| \widetilde {\phi} (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \\ \leqslant \sqrt {\underbrace {\sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \underbrace {\sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star}\right) \left(f _ {h , J (k , h)} ^ {k} (s _ {h} ^ {k} , a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h , J (k , h)} ^ {k} (s _ {h} ^ {k} , a ^ {\prime}) \right]\right) ^ {2}}} _ {=: [ \mathbb {V} _ {h} f _ {h, J (k, h)} ^ {k} ] (s _ {h} ^ {k})}} \\ \cdot \sqrt {\sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}\right) \| \widetilde {\phi} (s _ {h} ^ {k} , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}} \\ \leqslant \sqrt {\sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \left[ \mathbb {V} _ {h} f _ {h , J (k , h)} ^ {k} \right] \left(s _ {h} ^ {k}\right)} \sqrt {2 d H \log \left(1 + \frac {K}{d \lambda}\right)} \\ = \mathcal {O} \left(\sqrt {| \mathcal {K} | + H \log (1 / \delta)}\right) \cdot \sqrt {2 d H \log \left(1 + \frac {K}{d \lambda}\right)}, \tag {D.35} \\ \end{array}
$$

where the first inequality follows from the Cauchy-Schwarz inequality, the second-to-last inequality holds by Lemma D.7, and the last equality holds by Lemma D.12. Additionally, the second term in (D.34) can be bounded as follows:

$$
\begin{array}{l} \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \left(\sqrt {\mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star}\right)} - \sqrt {\mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}\right)}\right) \sqrt {\mathcal {P} _ {h} \left(a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star}\right)} \\ \cdot \left(f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) \right]\right) \| \widetilde {\phi} (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \\ \leqslant \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \frac {| \mathcal {P} _ {h} (a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1}) |}{\sqrt {\mathcal {P} _ {h} (a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star})} + \sqrt {\mathcal {P} _ {h} (a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {k + 1})}} \sqrt {\mathcal {P} _ {h} (a | s _ {h} ^ {k} , A _ {h} ^ {k} ; \boldsymbol {\theta} _ {h} ^ {\star})} \| \widetilde {\phi} (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \\ \leqslant \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} | \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {k + 1}) | \| \widetilde {\phi} (s _ {h} ^ {k}, a) \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}} \\ \leqslant 4 \bar {\alpha} _ {K} \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \max _ {a \in A _ {h} ^ {k}} \| \widetilde {\phi} (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \max _ {a \in A _ {h} ^ {k}} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \\ \leqslant 4 \bar {\alpha} _ {K} \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \max _ {a \in A _ {h} ^ {k}} \| \widetilde {\phi} (s _ {h} ^ {k} , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}} \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \max _ {a \in A _ {h} ^ {k}} \| \phi (s _ {h} ^ {k} , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}} \\ \leqslant \frac {4}{\sqrt {\kappa}} \bar {\alpha} _ {K} \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} (a | s _ {h} ^ {k} , A _ {h} ^ {k} , ; \boldsymbol {\theta} _ {h} ^ {k + 1}) \| \widetilde {\phi} (s _ {h} ^ {k} , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}} \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \max _ {a \in A _ {h} ^ {k}} \| \phi (s _ {h} ^ {k} , a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}} \\ \leqslant \frac {8}{\sqrt {\kappa}} \bar {\alpha} _ {K} d H \log \left(1 + \frac {K}{d \lambda}\right), \tag {D.36} \\ \end{array}
$$

where the third inequity holds by Lemma D.11 and by the definition of $\bar{\alpha}_{K} = \max_{h \in [H]} \alpha_{h}^{K}$ , the second-to-last inequality holds by the definition of $\kappa$ , and the last inequality holds by Lemma D.7.

Finally, we bound the last term in (D.34). Using the inequality $\|a\| - \|b\| \leqslant \|a - b\|$ for any vectors $a, b \in R^{d}$ , we have

$$
\sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) \left(\underbrace {f _ {h , J (k , h)} ^ {k} (s _ {h} ^ {k} , a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h , J (k , h)} ^ {k} (s _ {h} ^ {k} , a ^ {\prime}) \right]} _ {\geqslant 0}\right) \left(\| \widetilde {\phi} (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} - \| \bar {\phi} (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}\right)
$$

$$
\leqslant \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) \left\| \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \left(\mathcal {P} _ {h} \left(a ^ {\prime} | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) - \mathcal {P} _ {h} \left(a ^ {\prime} | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {k + 1}\right)\right) \phi (s _ {h} ^ {k}, a ^ {\prime}) \right\| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}
$$

$$
\leqslant \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \left| \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) - \mathcal {P} _ {h} \left(a | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {k + 1}\right) \right| \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}}
$$

$$
\leqslant 4 \bar {\alpha} _ {K} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \max _ {a \in A _ {h} ^ {k}} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} ^ {2}
$$

$$
\leqslant \frac {8}{\kappa} \bar {\alpha} _ {K} d H \log \left(1 + \frac {K}{d \lambda}\right), \tag {D.37}
$$

where the third inequity holds by Lemma D.11 and by the definition of $\bar{\alpha}_{K} = \max_{h \in [H]} \alpha_{h}^{K}$ and the last inequality holds by Lemma D.7.

By plugging (D.35), (D.36), and (D.37) into (D.34), and ccombining the result with (D.33), we obtain

$$
\begin{array}{l} \sum_ {k \in \mathcal {K} _ {(i)}} \sum_ {h = 1} ^ {H} \nabla \tilde {R} (\boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})) ^ {\top} \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) \\ \leqslant \mathcal {O} \left(\sqrt {| \mathcal {K} | + H \log (1 / \delta)}\right) \cdot 2 \bar {\alpha} _ {K} \sqrt {2 d H \log \left(1 + \frac {K}{d \lambda}\right)} + \frac {3 2}{\kappa} \bar {\alpha} _ {K} ^ {2} d H \log \left(1 + \frac {K}{d \lambda}\right) \\ = \mathcal {O} \left(d \sqrt {H | \mathcal {K} |} (\log K) ^ {3 / 2} \log M + \frac {1}{\kappa} d ^ {2} H (\log K) ^ {3} (\log M) ^ {2}\right). \tag {D.38} \\ \end{array}
$$

Now, we consider the second case to bound the term (A) in Equation (D.31).

Case (ii) For $(k,h)\in \mathcal{K}\times [H]$ such that $v_{h}^{k}(s_{h}^{k},a) = \phi (s_{h}^{k},a)^{\top}\pmb{\theta}_{h}^{\star} - 2\alpha_{h}^{k}\| \phi (s_{h}^{k},a)\|_{\left(\mathbf{H}_h^k\right)^{-1}}.$

In this case, we know that $f_{h,J(k,h)}^k (s_h^k,a) < f_{h,J(k,h)}^k (s_h^k,a_0)$ for all $a\in \mathcal{I}\backslash \{a_0\}$ . This implies that $|A_h^k| = 2$ , since adding any item $a\in \mathcal{I}\backslash \{a_0\}$ to the set $\{a_0\}$ always decreases the expected value of $f_{h,J(k,h)}^k$ . Furthermore, since we assume $\phi (s_h^k,a_0) = 0$ (which also implies $v_{h}^{k}(s_{h}^{k},a_{0}) = 0$ ), and denoting $A_{h}^{k} = \{a_{0},\tilde{a}_{h}^{k}\}$ , we have:

$$
\begin{array}{l} \sum_ {k \in \mathcal {K} _ {(i i)}} \sum_ {h = 1} ^ {H} \nabla \tilde {R} (\boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})) ^ {\top} \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) \\ = \sum_ {k \in \mathcal {K} _ {(i i)}} \sum_ {h = 1} ^ {H} 2 \alpha_ {h} ^ {k} \mathcal {P} _ {h} \left(\tilde {a} _ {h} ^ {k} | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, \tilde {a} _ {h} ^ {k}) \Bigg (\| \phi (s _ {h} ^ {k}, \tilde {a} _ {h} ^ {k}) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} - \mathcal {P} _ {h} \left(\tilde {a} _ {h} ^ {k} | s _ {h} ^ {k}, A _ {h} ^ {k}; \boldsymbol {\theta} _ {h} ^ {\star}\right) \left[ \| \phi (s _ {h} ^ {k}, \tilde {a} _ {h} ^ {k}) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \right] \Bigg) \\ \leqslant 2 \bar {\alpha} _ {K} \sum_ {k \in \mathcal {K} _ {(i i)}} \sum_ {h = 1} ^ {H} \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ \left(f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} \left[ f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) \right]\right) \| \phi (s _ {h} ^ {k}, a) - \mathbb {E} _ {\boldsymbol {\theta} _ {h} ^ {\star}} [ \phi (s _ {h} ^ {k}, a ^ {\prime}) ] \| _ {(\mathbf {H} _ {h} ^ {k}) ^ {- 1}} \right], \tag {D.39} \\ \end{array}
$$

where in the inequality, we use the definition $\bar{\alpha}_{K} := \max_{h \in [H]} \alpha_{h}^{K}$ . The rest of the analysis is similar to that in Case (i). Therefore, we derive

$$
\sum_ {k \in \mathcal {K} _ {(i i)}} \sum_ {h = 1} ^ {H} \nabla \tilde {R} (\boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})) ^ {\top} \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) = \mathcal {O} \left(d \sqrt {H | \mathcal {K} |} (\log K) ^ {3 / 2} \log M + \frac {1}{\kappa} d ^ {2} H (\log K) ^ {3} (\log M) ^ {2}\right). \tag {D.40}
$$

Now, we bound the term (B) in (D.31). Let $p_a(\bar{\boldsymbol{v}}_h^k(s_h^k)) = \frac{\exp\big(\bar{v}_h^k(s_h^k,a)\big)}{1 + \sum_{a'' \in A_h^k} \exp\big(\bar{v}_h^k(s_h^k,a'')\big)}$ .

$$
\begin{array}{l} \frac {1}{2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) ^ {\top} \nabla^ {2} \tilde {R} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})) \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) \\ = \frac {1}{2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \frac {\partial^ {2} \tilde {R} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k}))}{\partial a \partial a ^ {\prime}} \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) - \phi (s _ {h} ^ {k}, a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \\ = \frac{1}{2}\sum_{k\in \mathcal{K}}\sum_{h = 1}^{H}\sum_{a\in A_{h}^{k}}\sum_{\substack{a^{\prime}\in A_{h}^{k}\\ a^{\prime}\neq a}}\left(\upsilon_{h}^{k}(s_{h}^{k},a) - \phi (s_{h}^{k},a)^{\top}\boldsymbol{\theta}_{h}^{\star}\right)\frac{\partial^{2}\tilde{R}(\bar{\boldsymbol{v}}_{h}^{k}(s_{h}^{k}))}{\partial a\partial a^{\prime}}\left(\upsilon_{h}^{k}(s_{h}^{k},a^{\prime}) - \phi (s_{h}^{k},a^{\prime})^{\top}\boldsymbol{\theta}_{h}^{\star}\right) \\ + \frac {1}{2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \pmb {\theta} _ {h} ^ {\star}\right) ^ {2} \frac {\partial^ {2} \tilde {R} (\bar {\pmb {v}} _ {h} ^ {k} (s _ {h} ^ {k}))}{\partial a \partial a} \\ \leqslant \sum_{k\in \mathcal{K}}\sum_{h = 1}^{H}\sum_{a\in A^{k}_{h}}\sum_{\substack{a^{\prime}\in A^{k}_{h}\\ a^{\prime}\neq a}}\left|v^{k}_{h}(s^{k}_{h},a) - \phi (s^{k}_{h},a)^{\top}\boldsymbol{\theta}^{\star}_{h}\right|p_{a}(\bar{\boldsymbol{v}}^{k}_{h}(s^{k}_{h}))p_{a^{\prime}}(\bar{\boldsymbol{v}}^{k}_{h}(s^{k}_{h}))\left|v^{k}_{h}(s^{k}_{h},a^{\prime}) - \phi (s^{k}_{h},a^{\prime})^{\top}\boldsymbol{\theta}^{\star}_{h}\right| \\ + \frac {3}{2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) ^ {2} p _ {a} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})), \tag {D.41} \\ \end{array}
$$

where the inequality holds by Lemma D.8 and $f_{j,J(k,h)}^k \leqslant 1$ . To bound the first term in (D.41), by applying the AM-GM inequality, we get

$$
\begin{array}{l} \sum_{k\in \mathcal{K}}\sum_{h = 1}^{H}\sum_{a\in A^{k}_{h}}\sum_{\substack{a^{\prime}\in A^{k}_{h}\\ a^{\prime}\neq a}}\left|\upsilon^{k}_{h}(s^{k}_{h},a) - \phi (s^{k}_{h},a)^{\top}\pmb{\theta}^{\star}_{h}\right|p_{a}(\bar{\pmb{v}}^{k}_{h}(s^{k}_{h}))p_{a^{\prime}}(\bar{\pmb{v}}^{k}_{h}(s^{k}_{h}))\left|\upsilon^{k}_{h}(s^{k}_{h},a^{\prime}) - \phi (s^{k}_{h},a^{\prime})^{\top}\pmb{\theta}^{\star}_{h}\right| \\ \leqslant \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \left| v _ {h} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} \right| p _ {a} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})) p _ {a ^ {\prime}} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})) \left| v _ {h} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) - \phi (s _ {h} ^ {k}, a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star} \right| \\ \leqslant \frac {1}{2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \pmb {\theta} _ {h} ^ {\star}\right) ^ {2} p _ {a} (\bar {\pmb {v}} _ {h} ^ {k} (s _ {h} ^ {k})) p _ {a ^ {\prime}} (\bar {\pmb {v}} _ {h} ^ {k} (s _ {h} ^ {k})) \\ + \frac {1}{2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \sum_ {a ^ {\prime} \in A _ {h} ^ {k}} p _ {a} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})) p _ {a ^ {\prime}} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})) \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a ^ {\prime}) - \phi (s _ {h} ^ {k}, a ^ {\prime}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) ^ {2} \\ = \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) ^ {2} p _ {a} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})). \tag {D.42} \\ \end{array}
$$

Plugging (D.42) into (D.41), we have

$$
\begin{array}{l} \frac {1}{2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) ^ {\top} \nabla^ {2} \tilde {R} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})) \left(\boldsymbol {v} _ {h} ^ {k} (s _ {h} ^ {k}) - \boldsymbol {v} _ {h} ^ {\star} (s _ {h} ^ {k})\right) \\ \leqslant \frac {5}{2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} p _ {a} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})) \left(v _ {h} ^ {k} (s _ {h} ^ {k}, a) - \phi (s _ {h} ^ {k}, a) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) ^ {2} \\ \leqslant 1 0 \left(\bar {\alpha} _ {K}\right) ^ {2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} p _ {a} (\bar {\boldsymbol {v}} _ {h} ^ {k} (s _ {h} ^ {k})) \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} \\ \leqslant 1 0 \left(\bar {\alpha} _ {K}\right) ^ {2} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \max _ {a \in A _ {h} ^ {k}} \| \phi (s _ {h} ^ {k}, a) \| _ {\left(\mathbf {H} _ {h} ^ {k}\right) ^ {- 1}} = \mathcal {O} \left(\frac {1}{\kappa} d ^ {2} H (\log K) ^ {3} (\log M) ^ {2}\right), \tag {D.43} \\ \end{array}
$$

where the last inequality holds by Lemma D.7.

Combining (D.38), (D.40) and (D.43), we obtain

$$
\begin{array}{l} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \sum_ {a \in A _ {h} ^ {k}} \Bigl (\widetilde {\mathcal {P}} _ {h, J (k, h)} ^ {k} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k}) \Bigr) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a) \\ = \mathcal {O} \left(d \sqrt {H | \mathcal {K} |} (\log K) ^ {3 / 2} \log M + \frac {1}{\kappa} d ^ {2} H (\log K) ^ {3} (\log M) ^ {2}\right). \\ \end{array}
$$

This conclude the proof of Lemma D.13.

![](images/4472ebb941a779ec55f05c88b675db8ce3f71c0b4d1e11d184b283c47268fe12.jpg)

# D.4. Optimism

In this subsection, we prove the optimism of our value estimates $V_{h}^{k}$ .

Lemma D.14 (Point-wise monotonicity, Lemma 31 of Agarwal et al. 2023). Suppose Algorithm 1 uses a consistent bonus oracle satisfying Definition B.1. For any fixed $(k,h)\in [K]\times [H]$ , conditioning on events $\mathcal{E}_{\leqslant k - 1}\bigcap \left(\bigcap_{h^{\prime} = h}^{H}\mathcal{E}_{h^{\prime}}^{k}\right)$ , for all $(s_h,a_h)\in \mathcal{S}\times \mathcal{I}$ , we have

1. $\overline{Q}_h^\star (s_h,a_h)\leqslant f_{h,1}^k (s_h,a_h);$   
2. $f_{h, - 1}^{k}(s_{h},a_{h})\leqslant \overline{Q}_{h}^{\star}(s_{h},a_{h});$   
3. $f_{h,2}^{\tau}(s_h,a_h)\geqslant \max \left\{\mathcal{T}_hV_{h + 1,1}^k (s_h,a_h),f_{h,1}^k (s_h,a_h)\right\} ,\quad \forall \tau \in [k].$

Lemma D.15 (Optimism). Let $V_h^k$ be the realized optimistic value function defined in (D.18). Suppose Algorithm 1 uses a consistent bonus oracle satisfying Definition B.1. On the even conditioning on the good event $\mathcal{E}^\theta \bigcap \mathcal{E}_{\leqslant K}$ , for all $(k, h) \in [K] \times [H]$ , we have

$$
V _ {h} ^ {k} (s _ {h} ^ {k}) \geqslant V _ {h} ^ {\star} (s _ {h} ^ {k}).
$$

Proof of Lemma D.15. We denote $A_h^{k,\star} \in \operatorname{argmax}_A \sum_{a \in A} \mathcal{P}_h(a|s_h^k, A) \overline{Q}_h^\star(s_h^k, a)$ . If $A_h^k = A_{h,1}^k$ , by the definition of the

optimal value function $V_{h}^{\star}$ , we have

$$
\begin{array}{l} V _ {h} ^ {\star} (s _ {h} ^ {k}) = \max _ {A \in \mathcal {A}} \sum_ {a \in A} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A) \overline {{Q}} _ {h} ^ {\star} (s _ {h} ^ {k}, a) \\ = \sum_ {a \in A _ {h} ^ {k, \star}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k, \star}) \overline {{Q}} _ {h} ^ {\star} (s _ {h} ^ {k}, a) \\ \leqslant \sum_ {a \in A _ {h} ^ {k, \star}} \mathcal {P} _ {h} (a | s _ {h} ^ {k}, A _ {h} ^ {k, \star}) f _ {h, 1} ^ {k} (s _ {h} ^ {k}, a) \\ \leqslant \sum_ {a \in \tilde {A} _ {h} ^ {k}} \widetilde {\mathcal {P}} _ {h, 1} ^ {k} (a | s _ {h} ^ {k}, \tilde {A} _ {h} ^ {k}) f _ {h, 1} ^ {k} (s _ {h} ^ {k}, a) \\ \leqslant \sum_ {a \in A _ {h, 1} ^ {k}} \widetilde {\mathcal {P}} _ {h, 1} ^ {k} (a | s _ {h} ^ {k}, A _ {h, 1} ^ {k}) f _ {h, 1} ^ {k} (s _ {h} ^ {k}, a) \\ = \max _ {A \in \mathcal {A}} \sum_ {a \in A} \widetilde {\mathcal {P}} _ {h, 1} ^ {k} (a | s _ {h} ^ {k}, A) f _ {h, 1} ^ {k} (s _ {h} ^ {k}, a) = V _ {h} ^ {k} (s _ {h} ^ {k}), \\ \end{array}
$$

where the first inequality holds by Lemma D.14, in the second inequality, we use the fact that, by Lemma D.5, there exists a subset $\tilde{A}_{h}^{k} \subseteq A_{h}^{k,\star}$ with $\tilde{A}_{h}^{k} \in A$ such that the inequality holds, and the last inequality holds by the definition of $A_{h,1}^{k}$ .

The case where $A_{h}^{k} = A_{h,2}^{k}$ can be proven using the same reasoning.

# D.5. Variances

In this subsection, we present properties related to variances.

Lemma D.16 (Upper bound of variance estimator, Lemma 34 of Agarwal et al. 2023). Let $z_h^k = (s_h^k, a_h^k)$ . We denote $\mathbb{E}_{\mathbb{P}}[\cdot | s_h^k, a_h^k] = \mathbb{E}_{s_{h+1} \sim \mathbb{P}_h(\cdot | s_h^k, a_h^k)}[\cdot | s_h^k, a_h^k]$ and $\mathbb{V}_{\mathbb{P}}[\cdot | s_h^k, a_h^k] = \mathbb{V}_{s_{h+1} \sim \mathbb{P}_h(\cdot | s_h^k, a_h^k)}[\cdot | s_h^k, a_h^k]$ , where the expectation in only taken over $s_{h+1}$ due to the model transition for shorthand. Suppose Algorithm 1 uses a consistent bonus oracle satisfying Definition B.1. For any episode $k \geqslant 2$ conditioning on the good event $\mathcal{E}_{\leqslant K}$ , the variance estimator $\sigma_h^k$ satisfies

$$
\begin{array}{l} \left(\sigma_ {h} ^ {k}\right) ^ {2} \leqslant \mathbb {V} \left[ r _ {h} + V _ {h + 1, 1} ^ {k} (s _ {h + 1}) \mid z _ {h} ^ {k} \right] + 4 \left(f _ {h, 2} ^ {k} (z _ {h} ^ {k}) - f _ {h, - 2} ^ {k} (z _ {h} ^ {k})\right) \\ + 4 \min \left\{1, D _ {\mathcal {F} _ {h}} \big (z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\mathbf {1} ^ {\tau} \} _ {\tau = 1} ^ {k - 1} \big) \cdot \left(2 \sqrt {\big (\bar {\beta} _ {h} ^ {k} \big) ^ {2} + \rho} + 4 L \sqrt {\big (\beta_ {h , 2} ^ {k} \big) ^ {2} + \rho}\right) \right\}. \\ \end{array}
$$

Lemma D.17 (Sum of variances, Corollary 50 of Agarwal et al. 2023). Let $z_h^k = (s_h^k, a_h^k)$ . We denote $\mathbb{E}_{\mathbb{P}}[\cdot | s_h^k, a_h^k] = \mathbb{E}_{s_{h+1} \sim \mathbb{P}_h(\cdot | s_h^k, a_h^k)}[\cdot | s_h^k, a_h^k]$ and $\mathbb{V}_{\mathbb{P}}[\cdot | s_h^k, a_h^k] = \mathbb{V}_{s_{h+1} \sim \mathbb{P}_h(\cdot | s_h^k, a_h^k)}[\cdot | s_h^k, a_h^k]$ , where the expectation in only taken over $s_{h+1}$ due to the model transition for shorthand. When $L = \mathcal{O}(1)$ , with probability at least $1 - \delta$ , we have

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \mathbb {V} \left[ r _ {h} + V _ {h + 1, 1} ^ {k} (s _ {h + 1}) | z _ {h} ^ {k} \right] \\ \leqslant \mathcal {O} \left(H \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \left(f _ {h, 2} ^ {k} (z _ {h} ^ {k}) - f _ {h, - 2} ^ {k} (z _ {h} ^ {k})\right) + K + K H ^ {2} \delta + H ^ {2} | \mathcal {K} _ {o o} | + H ^ {4} \log^ {2} \frac {K H}{\delta}\right). \\ \end{array}
$$

# D.6. Approximation Error of Optimistic, Overly Optimistic (Pessimistic) $\overline{Q}$ -values

In this section, we provide some inequalities for bounding the optimistic values, overly optimistic values, and overly pessimistic values sequence, which are useful for the proofs in Subsection D.7.

Lemma D.18 (Approximation error of overly pessimistic $\overline{Q}$ ). Suppose Algorithm 1 uses a consistent bonus oracle satisfying

Definition B.1. Conditioning on the good event $\mathcal{E}_{\leqslant K}$ , for any $(k,h)\in [K]\times [H]$ , it holds that

$$
\begin{array}{l} (f _ {h, - 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \geqslant \sum_ {h ^ {\prime} = h + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, - 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, - 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ - 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + \sum_ {h ^ {\prime} = h + 1} ^ {H} \zeta_ {h ^ {\prime}, - 2} ^ {k} + \sum_ {h ^ {\prime} = h + 1} ^ {H} \dot {\zeta} _ {h ^ {\prime}, - 2} ^ {k}, \\ \end{array}
$$

where $\zeta_{h,-2}^{k}$ := $\mathbb{E}_{\mathbb{P}}\left[(V_{h,-2}^{k}-V_{h}^{\pi_{k}})(s_{h})\mid s_{h-1}^{k},a_{h-1}^{k}\right]$ - $(V_{h,-2}^{k}$ - $V_{h}^{\pi_{k}})(s_{h}^{k})$ and $\dot{\zeta}_{h,-2}^{k}$ :=

$$
\mathbb {E} _ {\mathcal {P}} \left[ \left(f _ {h, - 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}\right) (s _ {h} ^ {k}, a _ {h}) \mid s _ {h} ^ {k}, A _ {h} ^ {k} \right] - \left(f _ {h, - 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}).
$$

Proof of Lemma D.18. Under the event $\mathcal{E}_{\leqslant K}$ , we have

$$
\begin{array}{l} (f _ {h, - 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \\ = (f _ {h, - 2} ^ {k} - \mathcal {T} _ {h} V _ {h + 1, - 2} ^ {k}) (s _ {h} ^ {k}, a _ {h} ^ {k}) + (\mathcal {T} _ {h} V _ {h + 1, - 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \\ \geqslant - 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \mathbb {E} _ {\mathbb {P}} \left[ (r _ {h} - r _ {h}) + (V _ {h + 1, - 2} ^ {k} - V _ {h + 1} ^ {\pi_ {k}}) (s _ {h + 1}) | s _ {h} ^ {k}, a _ {h} ^ {k} \right] \\ = - 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + (V _ {h + 1, - 2} ^ {k} - V _ {h + 1} ^ {\pi_ {k}}) (s _ {h + 1} ^ {k}) + \zeta_ {h + 1, - 2} ^ {k} \\ \geqslant - 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + (Q _ {h + 1, - 2} ^ {k} - Q _ {h + 1} ^ {\pi_ {k}}) (s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) + \zeta_ {h + 1, - 2} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, - 2} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) f _ {h + 1, - 2} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) \overline {{Q}} _ {h + 1} ^ {\pi_ {k}} (s _ {h + 1} ^ {k}, a ^ {\prime})\right) \\ - 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, - 2} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, - 2} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k})\right) f _ {h + 1, - 2} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) \\ + \mathbb {E} _ {\mathcal {P}} \left[ \left(f _ {h + 1, - 2} ^ {k} - \overline {{Q}} _ {h + 1} ^ {\pi_ {k}}\right) (s _ {h + 1} ^ {k}, a _ {h + 1}) | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k} \right] - 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, - 2} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, - 2} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k})\right) f _ {h + 1, - 2} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) \\ + \left(f _ {h + 1, - 2} ^ {k} - \overline {{Q}} _ {h + 1} ^ {\pi_ {k}}\right) (s _ {h + 1} ^ {k}, a _ {h + 1} ^ {k}) - 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, - 2} ^ {k} + \dot {\zeta} _ {h + 1, - 2} ^ {k}, \\ \end{array}
$$

where the first inequality holds because $T_{h}V_{h+1,-2}^{k}\in\mathcal{F}_{h,-2}^{k}$ under the event $E_{\leqslant K}$ and definition of $b_{h,2}^{k}$ , and the last inequality holds since $V_{h+1,-2}^{k}(s_{h+1}^{k})\geqslant Q_{h+1,-2}^{k}(s_{h+1}^{k},A_{h+1}^{k})$ .

Hence, by recursion we obtain that

$$
\begin{array}{l} (f _ {h, - 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \geqslant \sum_ {h ^ {\prime} = h + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, - 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, - 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ - 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + \sum_ {h ^ {\prime} = h + 1} ^ {H} \zeta_ {h ^ {\prime}, - 2} ^ {k} + \sum_ {h ^ {\prime} = h + 1} ^ {H} \dot {\zeta} _ {h ^ {\prime}, - 2} ^ {k}. \\ (f _ {h, - 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \geqslant \sum_ {h ^ {\prime} = h + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, - 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, - 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ - 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + \sum_ {h ^ {\prime} = h + 1} ^ {H} \zeta_ {h ^ {\prime}, - 2} ^ {k} + \sum_ {h ^ {\prime} = h + 1} ^ {H} \dot {\zeta} _ {h ^ {\prime}, - 2} ^ {k}. \\ \end{array}
$$

Lemma D.19 (Approximation error of overly optimistic $\overline{Q}$ ). Suppose Algorithm 1 uses a consistent bonus oracle satisfying Definition B.1. Conditioning on the good event $E_{\leqslant K}$ , for any $k \in [K]$ and any $h \geqslant h_{k}$ , it holds that

$$
\begin{array}{l} (f _ {h, 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \leqslant \sum_ {h ^ {\prime} = h + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + \sum_ {h ^ {\prime} = h + 1} ^ {H} \zeta_ {h ^ {\prime}, 2} ^ {k} + \sum_ {h ^ {\prime} = h + 1} ^ {H} \dot {\zeta} _ {h ^ {\prime}, 2} ^ {k}, \\ \end{array}
$$

$$
\begin{array}{l} \text {where} \quad \zeta_ {h, 2} ^ {k} := \mathbb {E} _ {\mathbb {P}} \left[ (V _ {h, 2} ^ {k} - V _ {h} ^ {\pi_ {k}}) (s _ {h}) \mid s _ {h - 1} ^ {k}, a _ {h - 1} ^ {k} \right] \\ \mathbb {E} _ {\mathcal {P}} \left[ \left(f _ {h, 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}\right) (s _ {h} ^ {k}, a _ {h}) \mid s _ {h} ^ {k}, A _ {h} ^ {k} \right] - \left(f _ {h, 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}). \end{array} \quad - \quad (V _ {h, 2} ^ {k} - V _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}) \quad a n d \quad \dot {\zeta} _ {h, 2} ^ {k} :=
$$

Proof of Lemma D.19. Under the event $\mathcal{E}_{\leqslant K}$ , at $h \geqslant h_k$ , we have

$$
\begin{array}{l} (f _ {h, 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) = (f _ {h, 2} ^ {k} - \mathcal {T} _ {h} V _ {h + 1, 2} ^ {k}) (s _ {h} ^ {k}, a _ {h} ^ {k}) + (\mathcal {T} _ {h} V _ {h + 1, 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \\ \leqslant 2 b _ {h, 1} ^ {k} \left(s _ {h} ^ {k}, a _ {h} ^ {k}\right) + 2 b _ {h, 2} ^ {k} \left(s _ {h} ^ {k}, a _ {h} ^ {k}\right) + \mathbb {E} \left[ \left(V _ {h + 1, 2} ^ {k} - V _ {h + 1} ^ {\pi_ {k}}\right) \left(s _ {h + 1}\right) \mid s _ {h} ^ {k}, a _ {h} ^ {k} \right] \\ = 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + (V _ {h + 1, 2} ^ {k} - V _ {h + 1} ^ {\pi_ {k}}) (s _ {h + 1} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} \\ = 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + (Q _ {h + 1, 2} ^ {k} - Q _ {h + 1} ^ {\pi_ {k}}) (s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, 2} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) f _ {h + 1, 2} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) \overline {{Q}} _ {h + 1} ^ {\pi_ {k}} (s _ {h + 1} ^ {k}, a ^ {\prime})\right) \\ + 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, 2} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k})\right) f _ {h + 1, 2} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) \\ + \mathbb {E} _ {\mathcal {P}} \left[ \left(f _ {h + 1, 2} ^ {k} - \overline {{Q}} _ {h + 1} ^ {\pi_ {k}}\right) (s _ {h + 1} ^ {k}, a _ {h + 1}) | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k} \right] + 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, 2} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k})\right) f _ {h + 1, 2} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) \\ + \left(f _ {h + 1, 2} ^ {k} - \overline {{Q}} _ {h + 1} ^ {\pi_ {k}}\right) (s _ {h + 1} ^ {k}, a _ {h + 1} ^ {k}) + 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + 2 b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} + \dot {\zeta} _ {h + 1, 2} ^ {k}, \\ \end{array}
$$

where the first inequality holds based on the assumption that $T_{h}V_{h+1,2}^{k} \in F_{h,2}^{k}$ and definition of $b_{h,2}^{k}$ , and the third equality holds because for $h \geqslant h_{k}$ , we know that $A_{h+1}^{k} \in \arg\max_{A} Q_{h+1}^{k}(s_{h+1}^{k}, A) = \arg\max_{A} \sum_{a \in A} \tilde{\mathcal{P}}_{h+1,2}^{k}(a | s_{h+1}^{k}, A) f_{h+1,2}^{k}(s_{h+1}^{k}, a)$ by the data collection policy in (9).

Therefore, by recursion we get

$$
\begin{array}{l} (f _ {h, 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \leqslant \sum_ {h ^ {\prime} = h + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \Big (\widetilde {\mathcal {P}} _ {h ^ {\prime}, 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) \Big) f _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + \sum_ {h ^ {\prime} = h + 1} ^ {H} \zeta_ {h ^ {\prime}, 2} ^ {k} + \sum_ {h ^ {\prime} = h + 1} ^ {H} \dot {\zeta} _ {h ^ {\prime}, 2} ^ {k}. \\ \end{array}
$$

![](images/0ce88d949db3d0a0db408441c1a3becc88765aaa9b1c297ba04448b5d070b1ba.jpg)

Lemma D.20 (Approximation error of optimistic $\overline{Q}$ ). Suppose Algorithm 1 uses a consistent bonus oracle satisfying Definition B.1. Conditioning on the good event $E^{\theta} \bigcap E_{\leqslant K}$ , for any $k \in [K]$ and any $h \leqslant h_{k}$ , we have

$$
\begin{array}{l} (f _ {h, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \\ \leqslant \sum_ {h ^ {\prime} = h + 1} ^ {h _ {k} - 1} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, 1} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + \sum_ {h ^ {\prime} = h _ {k}} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + 2 \sum_ {h ^ {\prime} = h _ {k}} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + \sum_ {h ^ {\prime} = h + 1} ^ {h _ {k} - 1} (\zeta_ {h ^ {\prime}, 1} ^ {k} + \dot {\zeta} _ {h ^ {\prime}, 1} ^ {k}) + \sum_ {h ^ {\prime} = h _ {k}} ^ {H} (\zeta_ {h ^ {\prime}, 2} ^ {k} + \dot {\zeta} _ {h ^ {\prime}, 2} ^ {k}), \\ \end{array}
$$

where

$$
\zeta_ {h, 1} ^ {k} := \mathbb {E} _ {\mathbb {P}} \left[ (V _ {h, 1} ^ {k} - V _ {h} ^ {\pi_ {k}}) (s _ {h}) \mid s _ {h - 1} ^ {k}, a _ {h - 1} ^ {k} \right] - (V _ {h, 1} ^ {k} - V _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}),
$$

$$
\dot {\zeta} _ {h, 1} ^ {k} := \mathbb {E} _ {\mathcal {P}} \left[ \left(f _ {h, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}\right) (s _ {h} ^ {k}, a _ {h}) \mid s _ {h} ^ {k}, A _ {h} ^ {k} \right] - \left(f _ {h, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}),
$$

$$
\zeta_ {h, 2} ^ {k} := \mathbb {E} _ {\mathbb {P}} \left[ (V _ {h, 2} ^ {k} - V _ {h} ^ {\pi_ {k}}) (s _ {h}) \mid s _ {h - 1} ^ {k}, a _ {h - 1} ^ {k} \right] - (V _ {h, 2} ^ {k} - V _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}),
$$

$$
\dot {\zeta} _ {h, 2} ^ {k} := \mathbb {E} _ {\mathcal {P}} \left[ \left(f _ {h, 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}\right) (s _ {h} ^ {k}, a _ {h}) \mid s _ {h} ^ {k}, A _ {h} ^ {k} \right] - \left(f _ {h, 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}).
$$

Here, following the convention, we use the empty sum notation, i.e., $\sum_{i=a}^{b} x_i = 0$ , when $b \leqslant a$ .

Proof of Lemma D.20. Under the event $\mathcal{E}_{\leqslant K}$ , by Lemma D.14 it holds that $f_{h,1}^{k}(s,a)\leqslant f_{h,2}^{k}(s,a)$ for all $(s,a)\in \mathcal{S}\times \mathcal{I}$ . Therefore, for $h = h_k$ , by Lemma D.19, we get

$$
\begin{array}{l} (f _ {h, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \leqslant (f _ {h, 2} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \\ \leqslant \sum_ {h ^ {\prime} = h + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + \sum_ {h ^ {\prime} = h + 1} ^ {H} \zeta_ {h ^ {\prime}, 2} ^ {k} + \sum_ {h ^ {\prime} = h + 1} ^ {H} \dot {\zeta} _ {h ^ {\prime}, 2} ^ {k}. \tag {D.44} \\ \end{array}
$$

For $h = h_k - 1$ , we have

$$
\begin{array}{l} (f _ {h, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) = (f _ {h, 1} ^ {k} - \mathcal {T} _ {h} V _ {h + 1, 1} ^ {k}) (s _ {h} ^ {k}, a _ {h} ^ {k}) + (\mathcal {T} _ {h} V _ {h + 1, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \\ \leqslant 2 b _ {h, 1} ^ {k} \left(s _ {h} ^ {k}, a _ {h} ^ {k}\right) + \mathbb {E} \left[ \left(V _ {h + 1, 1} ^ {k} - V _ {h} ^ {\pi_ {k}}\right) \left(s _ {h + 1}\right) \mid s _ {h} ^ {k}, a _ {h} ^ {k} \right] \\ \leqslant 2 b _ {h, 1} ^ {k} \left(s _ {h} ^ {k}, a _ {h} ^ {k}\right) + \mathbb {E} \left[ \left(V _ {h + 1, 2} ^ {k} - V _ {h} ^ {\pi_ {k}}\right) \left(s _ {h + 1}\right) \mid s _ {h} ^ {k}, a _ {h} ^ {k} \right] \\ = 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + (V _ {h + 1, 2} ^ {k} - V _ {h} ^ {\pi_ {k}}) (s _ {h + 1} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} \\ = 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + (Q _ {h + 1, 2} ^ {k} - Q _ {h} ^ {\pi_ {k}}) (s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} \\ \leqslant \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, 2} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) f _ {h + 1, 2} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) \overline {{Q}} _ {h + 1} ^ {\pi_ {k}} (s _ {h + 1} ^ {k}, a ^ {\prime})\right) \\ + 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, 2} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k})\right) f _ {h + 1, 2} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) \\ + \mathbb {E} _ {\mathcal {P}} \left[ \left(f _ {h + 1, 2} ^ {k} - \overline {{Q}} _ {h + 1} ^ {\pi_ {k}}\right) (s _ {h + 1} ^ {k}, a _ {h + 1}) | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k} \right] + 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, 2} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k})\right) f _ {h + 1, 2} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) \\ \end{array}
$$

$$
+ \left(f _ {h + 1, 2} ^ {k} - \overline {{{Q}}} _ {h + 1} ^ {\pi_ {k}}\right) (s _ {h + 1} ^ {k}, a _ {h + 1} ^ {k}) + 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, 2} ^ {k} + \dot {\zeta} _ {h + 1, 2} ^ {k}, \tag {D.45}
$$

where the first inequality holds based on the assumption that $\mathcal{T}_hV_{h + 1,1}^k\in \mathcal{F}_{h,1}^k$ and definition of $b_{h,1}^{k}$ , and the second inequality holds because for any $s_{h + 1}\in S$ , we have

$$
\begin{array}{l} V _ {h + 1, 1} ^ {k} (s _ {h + 1}) = \sum_ {a ^ {\prime} \in A _ {h + 1, 1}} \widetilde {\mathcal {P}} _ {h + 1, 1} ^ {k} (a ^ {\prime} | s _ {h + 1}, A _ {h + 1, 1}) f _ {h + 1, 1} ^ {k} (s _ {h + 1}, a ^ {\prime}) \\ \leqslant \sum_ {a ^ {\prime} \in A _ {h + 1, 1}} \widetilde {\mathcal {P}} _ {h + 1, 1} ^ {k} (a ^ {\prime} | s _ {h + 1}, A _ {h + 1, 1}) f _ {h + 1, 2} ^ {k} (s _ {h + 1}, a ^ {\prime}) \\ \leqslant \sum_ {a ^ {\prime} \in \tilde {A} _ {h + 1, 1}} \widetilde {\mathcal {P}} _ {h + 1, 2} ^ {k} (a ^ {\prime} | s _ {h + 1}, \tilde {A} _ {h + 1, 1}) f _ {h + 1, 2} ^ {k} (s _ {h + 1}, a ^ {\prime}) \\ \leqslant \sum_ {a ^ {\prime} \in A _ {h + 1, 2} ^ {k}} \widetilde {\mathcal {P}} _ {h + 1, 2} ^ {k} (a ^ {\prime} | s _ {h + 1}, A _ {h + 1, 2} ^ {k}) f _ {h + 1, 2} ^ {k} (s _ {h + 1}, a ^ {\prime}) = V _ {h + 1, 2} ^ {k} (s _ {h + 1}), \\ \end{array}
$$

where in the first equality, we denote $A_{h + 1,1} \in \operatorname{argmax}_A \sum_{a' \in A} \widetilde{\mathcal{P}}_{h + 1,1}^k (a'|s_{h + 1}, A)f_{h + 1,1}^k (s_{h + 1}, a')$ , for the first inequality, we use the fact that $f_{h + 1,1}^k (s,a) \leqslant f_{h + 1,2}^k (s,a)$ (Lemma D.14), the second inequality holds for some $\tilde{A}_{h + 1,1} \subseteq A_{h + 1,1}$ (Lemma D.5), and the last inequality follows from the definition of $A_{h + 1,2}^k$ . Moreover, the third equality of (D.45) holds because for $h + 1 = h_k$ , we know that $A_{h + 1}^k = A_{h + 1,2}^k$ by the data collection policy in (9).

Therefore, by recursion, we have

$$
\begin{array}{l} (f _ {h, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \\ \leqslant \sum_ {h ^ {\prime} = h + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + 2 \sum_ {h ^ {\prime} = h + 1} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + \sum_ {h ^ {\prime} = h + 1} ^ {H} \zeta_ {h ^ {\prime}, 2} ^ {k} + \sum_ {h ^ {\prime} = h + 1} ^ {H} \dot {\zeta} _ {h ^ {\prime}, 2} ^ {k} \tag {D.46} \\ \end{array}
$$

Finally, we consider the case where $h < h_k - 1$ .

$$
\begin{array}{l} (f _ {h, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) = (f _ {h, 1} ^ {k} - \mathcal {T} _ {h} V _ {h + 1, 1} ^ {k}) (s _ {h} ^ {k}, a _ {h} ^ {k}) + (\mathcal {T} _ {h} V _ {h + 1, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \\ \leqslant 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \mathbb {E} \left[ (V _ {h + 1, 1} ^ {k} - V _ {h + 1} ^ {\pi_ {k}}) (s _ {h + 1}) | s _ {h} ^ {k}, a _ {h} ^ {k} \right] \\ = 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + (V _ {h + 1, 1} ^ {k} - V _ {h + 1} ^ {\pi_ {k}}) (s _ {h + 1} ^ {k}) + \zeta_ {h + 1, 1} ^ {k} \\ = 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + (Q _ {h + 1, 1} ^ {k} - Q _ {h + 1} ^ {\pi_ {k}}) (s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) + \zeta_ {h + 1, 1} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, 1} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) f _ {h + 1, 1} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) \overline {{Q}} _ {h + 1} ^ {\pi_ {k}} (s _ {h + 1} ^ {k}, a ^ {\prime})\right) \\ + 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, 1} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, 1} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k})\right) f _ {h + 1, 1} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) \\ + \mathbb {E} _ {\mathcal {P}} \left[ \left(f _ {h + 1, 1} ^ {k} - \overline {{Q}} _ {h + 1} ^ {\pi_ {k}}\right) (s _ {h + 1} ^ {k}, a _ {h + 1}) | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k} \right] + 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, 1} ^ {k} \\ = \sum_ {a ^ {\prime} \in A _ {h + 1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h + 1, 1} ^ {k} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k}) - \mathcal {P} _ {h + 1} (a ^ {\prime} | s _ {h + 1} ^ {k}, A _ {h + 1} ^ {k})\right) f _ {h + 1, 1} ^ {k} (s _ {h + 1} ^ {k}, a ^ {\prime}) \\ + \left(f _ {h + 1, 1} ^ {k} - \overline {{Q}} _ {h + 1} ^ {\pi_ {k}}\right) (s _ {h + 1} ^ {k}, a _ {h + 1} ^ {k}) + 2 b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \zeta_ {h + 1, 1} ^ {k} + \dot {\zeta} _ {h + 1, 1} ^ {k}, \\ \end{array}
$$

where the first inequality holds based on the assumption that $T_{h}V_{h+1,1}^{k}\in\mathcal{F}_{h,1}^{k}$ and definition of $b_{h,1}^{k}$ and the third equality holds because for $h+1<h_{k}$ , we have $A_{h+1}^{k}=A_{h+1,1}^{k}$ by the data collection policy in (9).

Hence, by recursion we have

$$
\begin{array}{l} (f _ {h, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (s _ {h} ^ {k}, a _ {h} ^ {k}) \leqslant \sum_ {h ^ {\prime} = h + 1} ^ {h _ {k} - 1} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, 1} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + \sum_ {h ^ {\prime} = h _ {k}} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + 2 \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + 2 \sum_ {h ^ {\prime} = h _ {k}} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) + \sum_ {h ^ {\prime} = h + 1} ^ {h _ {k} - 1} (\zeta_ {h ^ {\prime}, 1} ^ {k} + \dot {\zeta} _ {h ^ {\prime}, 1} ^ {k}) \\ + \sum_ {h ^ {\prime} = h _ {k}} ^ {H} (\zeta_ {h ^ {\prime}, 2} ^ {k} + \dot {\zeta} _ {h ^ {\prime}, 2} ^ {k}). \tag {D.47} \\ \end{array}
$$

Combining (D.44), (D.46), and (D.47), we conclude the proof.

# D.7. Bounds on bonuses and $|\mathcal{K}_{\mathbf{00}}|$

In this subsection, we provide proofs for the bounds on the sum of bonuses (Lemma D.21, Lemma D.22, and Lemma D.23) as well as the bound on the size of $K_{oo}$ (Lemma D.24).

Lemma D.21 (Crude bound on $b_{h,1}^{k}$ , Lemma 39 of Agarwal et al. 2023). Let $z_{h}^{k} = (s_{h}^{k}, a_{h}^{k})$ . Given $b_{h,1}^{k}(\cdot) \leqslant C \cdot \left(D_{\mathcal{F}_{h}}\left(\because \{z_{h}^{\tau}\}_{\tau=1}^{k-1}, \{\bar{\sigma}_{h}^{\tau}\}_{\tau=1}^{k-1}\right) \cdot \sqrt{\left(\beta_{h,1}^{k}\right)^{2} + \rho} + \epsilon_{b}\beta_{h,1}^{k}\right)$ , when $\rho = 1, \nu \leqslant 1$ , it holds that for any subset $K \in [K]$ , we have

$$
\begin{array}{l} \sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \min \left\{1 + L, b _ {h, 1} ^ {k} (z _ {h} ^ {k}) \right\} \\ = \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\nu \delta}} \cdot \left(\sqrt {\log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\nu \delta}} \cdot H \sqrt {d _ {\nu} | \mathcal {K} |} + \log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\nu \delta} \cdot d _ {\nu} H + | \mathcal {K} | H \epsilon_ {b}\right)\right). \\ \end{array}
$$

Lemma D.22 (Crude bound on $b_{h,2}^{k}$ , Lemma 38 of Agarwal et al. 2023). Let $z_{h}^{k} = (s_{h}^{k}, a_{h}^{k})$ . Given $b_{h,2}^{k}(\cdot) \leqslant C \cdot \left(D_{\mathcal{F}_{h}}\left(\because \{z_{h}^{\tau}\}_{\tau=1}^{k-1}, \{1^{\tau}\}_{\tau=1}^{k-1}\right) \cdot \sqrt{\left(\beta_{h,2}^{k}\right)^{2} + \rho} + \epsilon_{b}\beta_{h,1}^{k}\right)$ , when $\rho = 1, \nu \leqslant 1$ , it holds that for any subset $K \in [K]$ , we have

$$
\sum_ {k \in \mathcal {K}} \sum_ {h = 1} ^ {H} \min \left\{1 + L, b _ {h, 2} ^ {k} (z _ {h} ^ {k}) \right\} = \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \left(H \sqrt {d _ {\nu} | \mathcal {K} |} + d _ {\nu} H + | \mathcal {K} | H \epsilon_ {b}\right)\right).
$$

Lemma D.23 (Fine-grained bound on $b_{h,1}^{k}$ ). Let $z_{h}^{k} = (s_{h}^{k}, a_{h}^{k})$ . Recall that the bonus oracle B outputs a bonus function such that $b_{h,1}^{k}(\cdot) \leqslant C \cdot \left( D_{\mathcal{F}_{h}} \left( \cdot; \{z_{h}^{\tau}\}_{\tau=1}^{k-1}, \{\bar{\sigma}_{h}^{\tau}\}_{\tau=1}^{k-1} \right) \cdot \sqrt{\left( \beta_{h,1}^{k} \right)^{2} + \rho} + \epsilon_{b} \beta_{h,1}^{k} \right)$ . When $\rho = 1, \nu = 1/\sqrt{KH}$ , $\delta < (0, 1/7)$ and the event $E_{\leqslant K}$ holds, with probability at least $1 - 7\delta$ , we have

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1 + L, b _ {h, 1} ^ {k} (z _ {h} ^ {k}) \right\} \\ = \mathcal {O} \left(\sqrt {d _ {\nu} H K \cdot \log \frac {\mathcal {N} K H}{\delta}} + \frac {1}{\sqrt {\kappa}} d H ^ {7 / 2} \sqrt {d _ {\nu}} (\log K) ^ {3 / 2} \log M \cdot \log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\delta} \cdot \sqrt {\log \frac {\mathcal {N} K H}{\delta}}\right) \\ + \mathcal {O} \left(d _ {\nu} H ^ {7 / 2} \log \frac {\mathcal {N} K H}{\delta} \cdot \left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {3 / 2} + \sqrt {\log \frac {\mathcal {N} K H}{\delta}} \cdot \left(K H \epsilon_ {b} + \sqrt {d _ {\nu} K H ^ {3} \delta}\right)\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\delta} \log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \left(\sqrt {H ^ {2} \sum_ {k \in \mathcal {K} _ {o}} u _ {k}} + \sqrt {H ^ {2} | \mathcal {K} _ {o o} |}\right)\right). \\ \end{array}
$$

Proof of Lemma D.23. By the definition of the oracle B (Definition B.1), we have

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1 + L, b _ {h, 1} ^ {k} (z _ {h} ^ {k}) \right\} \\ = \mathcal {O} \left(\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1, D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \left\{z _ {h} ^ {\tau} \right\} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \cdot \sqrt {\left(\beta_ {h , 1} ^ {k}\right) ^ {2} + \rho} \right\} + K H \epsilon_ {b} \cdot \max _ {k, h} \beta_ {h, 1} ^ {k}\right) \\ = \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\delta}} \cdot \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1, D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \left\{z _ {h} ^ {\tau} \right\} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \right\} + K H \epsilon_ {b}\right), \tag {D.48} \\ \end{array}
$$

where the last equality holds by the definition of $\beta_{h,1}^{k}$ .

Now, we bound the summation terms

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1, D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \right\} \\ = \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1, \bar {\sigma} _ {h} ^ {k} \cdot \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 1} D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \left\{z _ {h} ^ {\tau} \right\} _ {\tau = 1} ^ {k - 1}, \left\{\bar {\sigma} \right\} _ {\tau = 1} ^ {k - 1}\right) \right\} \\ \end{array}
$$

by dividing into the following cases:

$$
\mathcal {I} _ {1} = \left\{(k, h) \in [ K ] \times [ H ]: (\bar {\sigma} _ {h} ^ {k}) ^ {- 1} D _ {\mathcal {F} _ {h}} (z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}) \geqslant 1 \right\},
$$

$$
\mathcal {I} _ {2} = \left\{(k, h) \in [ K ] \times [ H ]: \bar {\sigma} _ {h} ^ {k} = \nu , (k, h) \neq \mathcal {I} _ {1} \right\},
$$

$$
\mathcal {I} _ {3} = \Big \{(k, h) \in [ K ] \times [ H ]: \bar {\sigma} _ {h} ^ {k} = 2 \left(\sqrt {o (\delta_ {h} ^ {k})} + \iota (\delta_ {h} ^ {k})\right) \cdot \sqrt {D _ {\mathcal {F} _ {h}} (z _ {h} ^ {k} ; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1} , \{\bar {\sigma} _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1})},
$$

$$
(k, h) \neq \mathcal {I} _ {1} \Big \},
$$

$$
\mathcal {I} _ {4} = \left\{(k, h) \in [ K ] \times [ H ]: \bar {\sigma} _ {h} ^ {k} = \sqrt {2} \iota (\delta_ {h} ^ {k}) \sqrt {f _ {h , 2} ^ {k} (z _ {h} ^ {k}) - f _ {h , - 2} ^ {k} (z _ {h} ^ {k})}, (k, h) \neq \mathcal {I} _ {1} \right\}
$$

$$
\mathcal {I} _ {5} = \left\{(k, h) \in [ K ] \times [ H ]: \bar {\sigma} _ {h} ^ {k} = \sigma_ {h} ^ {k}, (k, h) \neq \mathcal {I} _ {1} \right\},.
$$

For the case of $I_{1}$ , we have

$$
\begin{array}{l} \sum_ {(k, h) \in \mathcal {I} _ {1}} \min \left\{1, \bar {\sigma} _ {h} ^ {k} \cdot \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 1} D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \right\} \\ \leqslant \sum_ {(k, h) \in \mathcal {I} _ {1}} \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 1} D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \leqslant \sum_ {h = 1} ^ {H} \dim_ {\nu , K} (\mathcal {F} _ {h}) = d _ {\nu} H. \tag {D.49} \\ \end{array}
$$

For $I_{2}$ , we use the Cauchy-Schwarz inequality to get

$$
\begin{array}{l} \sum_ {(k, h) \in \mathcal {I} _ {2}} \min \left\{1, \bar {\sigma} _ {h} ^ {k} \cdot (\bar {\sigma} _ {h} ^ {k}) ^ {- 1} D _ {\mathcal {F} _ {h}} (z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}) \right\} \\ \leqslant \sqrt {\nu^ {2} K H} \cdot \sqrt {\sum_ {(k , h) \in \mathcal {I} _ {2}} \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 2} D _ {\mathcal {F} _ {h}} ^ {2} \left(z _ {h} ^ {k} ; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1} , \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right)} \leqslant \sqrt {\sum_ {h = 1} ^ {H} \dim_ {\nu , K} (\mathcal {F} _ {h})} \quad = \sqrt {d _ {\nu} H}. \tag {D.50} \\ \end{array}
$$

For $\mathcal{I}_3$ , we have

$$
\begin{array}{l} \sum_ {(k, h) \in \mathcal {I} _ {3}} \min \left\{1, \bar {\sigma} _ {h} ^ {k} \cdot \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 1} D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \left\{z _ {h} ^ {\tau} \right\} _ {\tau = 1} ^ {k - 1}, \left\{\bar {\sigma} \right\} _ {\tau = 1} ^ {k - 1}\right) \right\} \\ \leqslant \sum_ {(k, h) \in \mathcal {I} _ {3}} \left(8 o (\delta_ {h} ^ {k}) + \iota^ {2} (\delta_ {h} ^ {k})\right) \cdot \min \left\{1, \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 2} D _ {\mathcal {F} _ {h}} ^ {2} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \right\} \\ = \mathcal {O} \left(\left(\sqrt {\log \frac {\mathcal {N} K H}{\delta}} + \log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) \cdot \sum_ {h = 1} ^ {H} \dim_ {\nu , K} (\mathcal {F} _ {h})\right) \\ = \mathcal {O} \left(\left(\sqrt {\log \frac {\mathcal {N} K H}{\delta}} + \log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\delta}\right) d _ {\nu} H\right), \tag {D.51} \\ \end{array}
$$

where the inequality holds because, by dividing both sides of $\bar{\sigma}_h^k = 2\left(\sqrt{o(\delta_h^k)} + \iota(\delta_h^k)\right) \cdot \sqrt{D_{\mathcal{F}_h}(z_h^k; \{z_h^\tau\}_{\tau=1}^{k-1}, \{\bar{\sigma}_h^\tau\}_{\tau=1}^{k-1})}$ by $\sqrt{\bar{\sigma}_h^k}$ and rearranging terms, we get:

$$
\bar {\sigma} _ {h} ^ {k} \leqslant \left(8 o (\delta_ {h} ^ {k}) + \iota^ {2} (\delta_ {h} ^ {k})\right) D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right).
$$

We also use the property that $\left(\bar{\sigma}_h^k\right)^{-1}D_{\mathcal{F}_h}\big(z_h^k;\{z_h^\tau \}_{\tau = 1}^{k - 1},\{\bar{\sigma}\}_{\tau = 1}^{k - 1}\big)\leqslant 1$ for $(k,h)\in \mathcal{I}_3$ , which follows directly from the definition of $\mathcal{I}_3$ .

For $\mathcal{I}_4$ , we have

$$
\begin{array}{l} \sum_ {(k, h) \in \mathcal {I} _ {4}} \min \left\{1, \bar {\sigma} _ {h} ^ {k} \cdot \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 1} D _ {\mathcal {F} _ {h}} \big (z _ {h} ^ {k}, \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1} \big) \right\} \\ \leqslant \sum_ {(k, h) \in \mathcal {I} _ {4}} \bar {\sigma} _ {h} ^ {k} \cdot \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 1} D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \\ = \sum_ {(k, h) \in \mathcal {I} _ {4}} \sqrt {2} \iota (\delta_ {h} ^ {k}) \sqrt {f _ {h , 2} ^ {k} (z _ {h} ^ {k}) - f _ {h , - 2} ^ {k} (z _ {h} ^ {k})} \cdot \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 1} D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \\ \leqslant \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} f _ {h , 2} ^ {k} (z _ {h} ^ {k}) - f _ {h , - 2} ^ {k} (z _ {h} ^ {k})} \cdot \sqrt {d _ {\nu} H}\right), \tag {D.52} \\ \end{array}
$$

where the last inequality holds by the Cauchy-Schwarz inequality together with the definition of $\iota(\delta_{h}^{k})$ .

Lastly, restricting on $\mathcal{I}_5$ , if the event $\mathcal{E}_{\leqslant K}$ holds, we have

$$
\begin{array}{l} \sum_ {(k, h) \in \mathcal {I} _ {5}} \min \left\{1, \bar {\sigma} _ {h} ^ {k} \cdot (\bar {\sigma} _ {h} ^ {k}) ^ {- 1} D _ {\mathcal {F} _ {h}} (z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}) \right\} \\ \leqslant \sum_ {(k, h) \in \mathcal {I} _ {5}} \bar {\sigma} _ {h} ^ {k} \cdot \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 1} D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \\ \leqslant \sqrt {\sum_ {(k , h) \in \mathcal {I} _ {5}} \left(\sigma_ {h} ^ {k}\right) ^ {2}} \cdot \sqrt {\sum_ {(k , h) \in \mathcal {I} _ {5}} \left(\bar {\sigma} _ {h} ^ {k}\right) ^ {- 2} D _ {\mathcal {F} _ {h}} ^ {2} \left(z _ {h} ^ {k} ; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1} , \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right)} \\ \leqslant \mathcal {O} \left(\sqrt {\sum_ {k , h} \mathbb {V} \left[ r _ {h} + V _ {h + 1 , 1} ^ {k} (s _ {h + 1}) \mid z _ {h} ^ {k} \right] + \sum_ {k , h} \left(f _ {h , 2} ^ {k} (z _ {h} ^ {k}) - f _ {h , - 2} ^ {k} (z _ {h} ^ {k})\right)} \cdot \sqrt {d _ {\nu} H}\right) \\ + \mathcal {O} \left(\sqrt {\sum_ {k , h} \min \left\{1 , D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k} ; \left\{z _ {h} ^ {\tau} \right\} _ {\tau = 1} ^ {k - 1} , \left\{\mathbf {1} ^ {\tau} \right\} _ {\tau = 1} ^ {k - 1}\right) \right\} \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}}} \cdot \sqrt {d _ {\nu} H}\right), \tag {D.53} \\ \end{array}
$$

where the second inequality holds by the Cauchy-Schwarz inequality, the last inequality holds by Lemma D.16 and the definition of Eluder dimension.

To further bound the first term on the right-hand side of (D.53), we apply Lemma D.17. Therefore, with probability at least $1 - \delta$ , we have

$$
\begin{array}{l} \sum_ {(k, h) \in \mathcal {I} _ {5}} \min \left\{1, \bar {\sigma} _ {h} ^ {k} \cdot (\bar {\sigma} _ {h} ^ {k}) ^ {- 1} D _ {\mathcal {F} _ {h}} (z _ {h} ^ {k}, \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}) \right\} \\ \leqslant \mathcal {O} \left(\sqrt {H \sum_ {k , h} \left(f _ {h , 2} ^ {k} (z _ {h} ^ {k}) - f _ {h , - 2} ^ {k} (z _ {h} ^ {k})\right) + K + K H ^ {2} \delta + H ^ {2} | \mathcal {K} _ {\mathrm{oo}} | + H ^ {4} \log^ {2} \frac {K H}{\delta}} \cdot \sqrt {d _ {\nu} H}\right) \\ + \mathcal {O} \left(\sqrt {\sum_ {k , h} \left(f _ {h , 2} ^ {k} (z _ {h} ^ {k}) - f _ {h , - 2} ^ {k} (z _ {h} ^ {k})\right)} \cdot \sqrt {d _ {\nu} H}\right) + \mathcal {O} \left(\sqrt {\left(d _ {\nu} H + H \sqrt {d _ {\nu} K}\right) \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}}} \cdot \sqrt {d _ {\nu} H}\right) \\ \leqslant \mathcal {O} \left(\sqrt {K + K H ^ {2} \delta + H ^ {2} | \mathcal {K} _ {\mathrm{oo}} | + H ^ {4} \log^ {2} \frac {K H}{\delta}} \cdot \sqrt {d _ {\nu} H} + \sqrt {H \sum_ {k , h} \left(f _ {h , 2} ^ {k} (z _ {h} ^ {k}) - f _ {h , - 2} ^ {k} (z _ {h} ^ {k})\right)} \cdot \sqrt {d _ {\nu} H}\right) \\ + \mathcal {O} \left(d _ {\nu} H ^ {1. 5} \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}}\right), \tag {D.54} \\ \end{array}
$$

where the first inequality holds by the fact that

$$
\sum_ {k, h} \min \left\{1, D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\mathbf {1} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}\right) \right\} \leqslant d _ {\nu} H
$$

$$
\sum_ {k, h} \min \left\{1, D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\mathbf {1} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}\right) \right\} \leqslant \sqrt {K H} \sqrt {\sum_ {k , h} D _ {\mathcal {F} _ {h}} ^ {2} \left(z _ {h} ^ {k} ; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1} , \{\mathbf {1} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}\right)} \leqslant H \sqrt {d _ {\nu} K},
$$

and the last inequality holds by the AM-GM inequality such that

$$
H \sqrt {K \cdot d _ {\nu} \log \frac {\mathcal {N N} _ {b} K H}{\delta}} \leqslant K + H ^ {2} d _ {\nu} \log \frac {\mathcal {N N} _ {b} K H}{\delta}.
$$

Combining Equation D.49, D.50, D.51, D.52, and D.54, we get

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1, D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \right\} \\ \leqslant \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \sqrt {H \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} f _ {h , 2} ^ {k} (z _ {h} ^ {k}) - f _ {h , - 2} ^ {k} (z _ {h} ^ {k})}\right) \\ + \mathcal {O} \left(\sqrt {K + K H ^ {2} \delta + H ^ {2} | \mathcal {K} _ {\mathrm{oo}} | + H ^ {4} \log^ {2} \frac {K H}{\delta}} \cdot \sqrt {d _ {\nu} H} + d _ {\nu} H ^ {1. 5} \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}}\right). \tag {D.55} \\ \end{array}
$$

Now, we bound the term $\sum_{k,h}\left(f_{h,2}^{k}(z_{h}^{k}) - f_{h,-2}^{k}(z_{h}^{k})\right)$ . For $k\in \mathcal{K}_{\mathrm{oo}}$ , we have

$$
\sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} \sum_ {h = 1} ^ {H} \left(f _ {h, 2} ^ {k} (z _ {h} ^ {k}) - f _ {h, - 2} ^ {k} (z _ {h} ^ {k})\right) = \mathcal {O} (| \mathcal {K} _ {\mathrm{oo}} | H).
$$

Otherwise, for episodes $k \in K_{0}$ , we know that it holds true that $f_{h,1}^{k}(z_{h}^{k}) \geqslant f_{h,2}^{k}(z_{h}^{k}) - u_{k}$ by (9). Therefore, under the event $E_{\leqslant K}$ , we have

$$
\begin{array}{l} \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} \sum_ {h = 1} ^ {H} \left(f _ {h, 2} ^ {k} (z _ {h} ^ {k}) - f _ {h, - 2} ^ {k} (z _ {h} ^ {k})\right) \leqslant \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} \sum_ {h = 1} ^ {H} \left(f _ {h, 1} ^ {k} (z _ {h} ^ {k}) - f _ {h, - 2} ^ {k} (z _ {h} ^ {k})\right) + H \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} u _ {k} \\ = \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} \sum_ {h = 1} ^ {H} \left((f _ {h, 1} ^ {k} - \overline {{Q}} _ {h} ^ {\pi_ {k}}) (z _ {h} ^ {k}) + (\overline {{Q}} _ {h} ^ {\pi_ {k}} - f _ {h, - 2} ^ {k}) (z _ {h} ^ {k})\right) + H \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} u _ {k} \\ \leqslant \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} \sum_ {h = 1} ^ {H} \sum_ {h ^ {\prime} = h + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, 1} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ - \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} \sum_ {h = 1} ^ {H} \sum_ {h ^ {\prime} = h + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, - 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}} ^ {k})\right) f _ {h ^ {\prime}, - 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + 2 \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} \sum_ {h = 1} ^ {H} \min \left\{1 + L, \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) \right\} + 2 \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} \sum_ {h = 1} ^ {H} \min \left\{1 + L, \sum_ {h ^ {\prime} = h} ^ {H} b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) \right\} \\ + \underbrace {\sum_ {k \in \mathcal {K} _ {\mathrm{o}}} \sum_ {h = 1} ^ {H} \sum_ {h ^ {\prime} = h + 1} ^ {H} \left(\zeta_ {h ^ {\prime} , 1} ^ {k} + \dot {\zeta} _ {h ^ {\prime} , 1} ^ {k} - \zeta_ {h ^ {\prime} , - 2} ^ {k} - \dot {\zeta} _ {h ^ {\prime} , - 2} ^ {k}\right)} _ {\text {martingale difference sequences (MDSs)}} + H \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} u _ {k}, \tag {D.56} \\ \end{array}
$$

where the second inequality holds by Lemma D.18 and D.20.

To further the right-hand side of (D.56), we apply Lemma D.10 (which holds with probability at least $1 - 2\delta$ ) to the first and the second terms, Lemma D.21 to the third term, Lemma D.22 to the forth term, and we bound the fifth term using the

Azuma-Hoeffding inequality (which holds with probability at least $1 - 4\delta$ ). As a result, absorbing the low-order terms, we obtain that

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \left(f _ {h, 2} ^ {k} (z _ {h} ^ {k}) - f _ {h, - 2} ^ {k} (z _ {h} ^ {k})\right) \\ \leqslant \mathcal {O} \left(\frac {1}{\sqrt {\kappa}} d H ^ {2} \sqrt {K} \cdot (\log K) ^ {3 / 2} \log M\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\delta}} \cdot \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot H ^ {2} \sqrt {d _ {\nu} K} + \log \frac {\mathcal {N N} _ {b} K H}{\delta} \cdot d _ {\nu} H ^ {2}\right)\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot K H ^ {2} \epsilon_ {b} + H \sqrt {K H \log \frac {K H}{\delta}} + | \mathcal {K} _ {\mathrm{oo}} | H + H \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} u _ {k}\right), \tag {D.57} \\ \end{array}
$$

where the last inequality holds by the AM-GM inequality.

Plugging (D.57) to (D.55), we have

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1, \bar {\sigma} _ {h} ^ {k} \cdot (\bar {\sigma} _ {h} ^ {k}) ^ {- 1} D _ {\mathcal {F} _ {h}} \left(z _ {h} ^ {k}; \{z _ {h} ^ {\tau} \} _ {\tau = 1} ^ {k - 1}, \{\bar {\sigma} \} _ {\tau = 1} ^ {k - 1}\right) \right\} \\ \leqslant \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \left(\sqrt {\frac {1}{\sqrt {\kappa}} d H ^ {3} (\log K) ^ {3 / 2} \log M \cdot \sqrt {K}} + H ^ {2} \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} u _ {k}\right)\right) \\ + \mathcal {O} \left(\left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {3 / 4} \cdot \sqrt {d _ {\nu} H} \cdot \sqrt {\sqrt {\log \frac {\mathcal {N K H}}{\delta}} \cdot \left(H ^ {3} \sqrt {d _ {\nu} K} + \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot d _ {\nu} H ^ {3}\right)}\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \sqrt {\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot K H ^ {3} \epsilon_ {b} + H ^ {2} \sqrt {K H \log \frac {K H}{\delta}} + | \mathcal {K} _ {\mathrm{oo}} | H ^ {2}}\right) \\ + \mathcal {O} \left(\sqrt {K + K H ^ {2} \delta + H ^ {4} \log^ {2} \frac {K H}{\delta}} \cdot \sqrt {d _ {\nu} H} + d _ {\nu} H ^ {1. 5} \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}}\right) \\ \leqslant \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \sqrt {\frac {K}{\log \frac {\mathcal {N N} _ {b} K H}{\delta}} + \frac {1}{\kappa} d ^ {2} H ^ {6} \left((\log K) ^ {3 / 2} \log M\right) ^ {2} \cdot \log \frac {\mathcal {N N} _ {b} K H}{\delta}}\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \sqrt {d _ {\nu} H ^ {6} \log \frac {\mathcal {N} K H}{\delta} \left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {2} + H ^ {2} \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} u _ {k}}\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \sqrt {H ^ {2} | \mathcal {K} _ {\mathrm{oo}} |} + K H \epsilon_ {b} + \sqrt {d _ {\nu} H} \cdot \sqrt {K H ^ {2} \delta}\right) \\ = \mathcal {O} \left(\sqrt {d _ {\nu} H K} + \frac {1}{\sqrt {\kappa}} d H ^ {7 / 2} \sqrt {d _ {\nu}} (\log K) ^ {3 / 2} \log M \cdot \log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) \\ + \mathcal {O} \left(d _ {\nu} H ^ {7 / 2} \sqrt {\log \frac {\mathcal {N} K H}{\delta}} \cdot \left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {3 / 2}\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \left(\sqrt {H ^ {2} \sum_ {k \in \mathcal {K} _ {0}} u _ {k}} + \sqrt {H ^ {2} | \mathcal {K} _ {0 0} |}\right) + K H \epsilon_ {b} + \sqrt {d _ {\nu} K H ^ {3} \delta}\right), \tag {D.58} \\ \end{array}
$$

where the second inequality holds by applying the AM-GM inequality and absorbing the lower-order terms.

Finally, plugging (D.58) to (D.48), we derive that

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1 + L, b _ {h, 1} ^ {k} (z _ {h} ^ {k}) \right\} \\ \leqslant \mathcal {O} \left(\sqrt {d _ {\nu} H K \cdot \log \frac {\mathcal {N} K H}{\delta}} + \frac {1}{\sqrt {\kappa}} d H ^ {7 / 2} \sqrt {d _ {\nu}} (\log K) ^ {3 / 2} \log M \cdot \log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\delta} \cdot \sqrt {\log \frac {\mathcal {N} K H}{\delta}}\right) \\ + \mathcal {O} \left(d _ {\nu} H ^ {7 / 2} \log \frac {\mathcal {N} K H}{\delta} \cdot \left(\log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\delta}\right) ^ {3 / 2} + \sqrt {\log \frac {\mathcal {N} K H}{\delta}} \cdot \left(K H \epsilon_ {b} + \sqrt {d _ {\nu} K H ^ {3} \delta}\right)\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\delta} \log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \left(\sqrt {H ^ {2} \sum_ {k \in \mathcal {K} _ {0}} u _ {k}} + \sqrt {H ^ {2} | \mathcal {K} _ {0 0} |}\right)\right). \\ \end{array}
$$

This concludes the proof of Lemma D.23.

![](images/75cdc0f0a056593484b44149fd16d2f4bbeda0f8490cf0ec65653bef6b5c301a.jpg)

Lemma D.24 (Bounding size of $K_{oo}$ ). Suppose $\nu \leqslant 1$ and we set

$$
u _ {k} \geqslant C \cdot \left(\frac {\sqrt {\log \frac {\mathcal {N K H}}{\nu \delta}} \cdot \left(\log \frac {\mathcal {N N} _ {b} K H}{\nu \delta} \cdot H ^ {5 / 2} \sqrt {d _ {\nu}} + \sqrt {k} H \epsilon_ {b}\right)}{\sqrt {k}} + \frac {d H ^ {5 / 2} (\log K) ^ {3 / 2} \log M \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}}}{\sqrt {k}}\right),
$$

for some large enough constant $0 < C < \infty$ , when the event $E_{\leq K}$ holds true, then with probability at least $1 - 2\delta$ , it holds that

$$
| \mathcal {K} _ {o o} | \leqslant \mathcal {O} \left(\frac {K}{H ^ {3} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}} + \frac {d _ {\nu}}{H ^ {3}} + \frac {d ^ {4} ((\log K) ^ {3 / 2} \log M) ^ {4}}{\kappa^ {2} d _ {\nu} H ^ {3} \cdot \log \frac {\mathcal {N K H}}{\nu \delta} \left(\log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}\right) ^ {2}}\right).
$$

Proof of Lemma D.24. By the definition of $h_k$ , for each $k \in \mathcal{K}_{\mathrm{oo}}$ , we have $f_{h_k,2}^k (s_{h_k}^k,a_{h_k}^k) \geqslant f_{h_k,1}^k (s_{h_k}^k,a_{h_k}^k) + u_k$ , which implies that

$$
\begin{array}{l} \sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} (f _ {h _ {k}, 2} ^ {k} - f _ {h _ {k}, 1} ^ {k}) (s _ {h _ {k}} ^ {k}, a _ {h _ {k}} ^ {k}) \geqslant \sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} u _ {k} \\ \geqslant C \cdot \left(\sqrt {\log \frac {\mathcal {N} K H}{\nu \delta}} \cdot \left(\log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\nu \delta} \cdot H ^ {5 / 2} \sqrt {d _ {\nu}} \cdot \frac {| \mathcal {K} _ {\mathrm{oo}} |}{\sqrt {K}} + | \mathcal {K} _ {\mathrm{oo}} | H \epsilon_ {b}\right) \right. \\ + d H ^ {5 / 2} (\log K) ^ {3 / 2} \log M \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}} \cdot \left. \frac {| \mathcal {K} _ {\mathrm{oo}} |}{\sqrt {K}}\right). \tag {D.59} \\ \end{array}
$$

Furthermore, under the event $E_{\leqslant K}$ , by Lemma D.14, it holds that $f_{h,1}^{k}(s_{h},a_{h}) \geqslant \overline{Q}_{h}^{\star}(s_{h},a_{h}) \geqslant \overline{Q}_{h}^{\pi_{k}}(s_{h},a_{h})$ for all $(s_{h},a_{h}) \in \mathcal{S} \times \mathcal{I}$ . Thus, we get

$$
\begin{array}{l} \sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} (f _ {h _ {k}, 2} ^ {k} - f _ {h _ {k}, 1} ^ {k}) (s _ {h _ {k}} ^ {k}, a _ {h _ {k}} ^ {k}) \leqslant \sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} (f _ {h _ {k}, 2} ^ {k} - \overline {{Q}} _ {h _ {k}} ^ {\pi_ {k}}) (s _ {h _ {k}} ^ {k}, a _ {h _ {k}} ^ {k}) \\ \leqslant \sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} \sum_ {h ^ {\prime} = h _ {k} + 1} ^ {H} \sum_ {a ^ {\prime} \in A _ {h ^ {\prime}, 2} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h ^ {\prime}, 2} ^ {k} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}, 2} ^ {k}) - \mathcal {P} _ {h ^ {\prime}} (a ^ {\prime} | s _ {h ^ {\prime}} ^ {k}, A _ {h ^ {\prime}, 2} ^ {k})\right) f _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a ^ {\prime}) \\ + 2 \sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} \sum_ {h ^ {\prime} = h _ {k}} ^ {H} \min \left\{1 + L, b _ {h ^ {\prime}, 1} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) \right\} + 2 \sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} \sum_ {h ^ {\prime} = h _ {k}} ^ {H} \min \left\{1 + L, b _ {h ^ {\prime}, 2} ^ {k} (s _ {h ^ {\prime}} ^ {k}, a _ {h ^ {\prime}} ^ {k}) \right\} + \sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} \sum_ {h ^ {\prime} = h _ {k} + 1} ^ {H} \left(\zeta_ {h ^ {\prime}, 2} ^ {k} + \dot {\zeta} _ {h ^ {\prime}, 2} ^ {k}\right) \\ \leqslant \mathcal {O} \left(d H \sqrt {| \mathcal {K} _ {\mathrm{oo}} |} \cdot (\log K) ^ {3 / 2} \log M + \frac {1}{\kappa} d ^ {2} H ((\log K) ^ {3 / 2} \log M) ^ {2} + \sqrt {| \mathcal {K} _ {\mathrm{oo}} | H \log \frac {K H}{\delta}}\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\nu \delta}} \cdot \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}} \cdot H \sqrt {d _ {\nu} | \mathcal {K} _ {\mathrm{oo}} |} + \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta} \cdot d _ {\nu} H + | \mathcal {K} _ {\mathrm{oo}} | H \epsilon_ {b}\right)\right), \tag {D.60} \\ \end{array}
$$

where in the first inequality, we note that $A_{h'}^{k} = A_{h',2}^{k}$ for $h' \geqslant h_{k}$ , the second inequality follows from D.19. And for the third inequality, we apply Lemma D.13 to the first term, Lemma D.21 to the second term, Lemma D.22 to the third term. Finally, we bound the last term using the Azuma-Hoeffding inequality with probability at least $1 - 2\delta$ .

Thus, in order for the two inequalities D.59 and D.60 to hold simultaneously, the following condition must be satisfied:

$$
| \mathcal {K} _ {\mathrm{oo}} | \leqslant \mathcal {O} \Bigg (\max \left\{\frac {K}{H ^ {3} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}}, \frac {\left(d _ {\nu} \sqrt {\log \frac {\mathcal {N K H}}{\nu \delta}} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta} + \frac {1}{\kappa} d ^ {2} ((\log K) ^ {3 / 2} \log M) ^ {2}\right) \cdot \sqrt {K}}{H ^ {3 / 2} \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}} \left(\sqrt {d _ {\nu} \log \frac {\mathcal {N K H}}{\nu \delta} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}} + d (\log K) ^ {3 / 2} \log M\right)} \right\} \Bigg).
$$

Using the AM-GM inequality, we can further bound the second term inside the max operation.

$$
\mathcal {O} \left(\frac {\left(d _ {\nu} \sqrt {\log \frac {\mathcal {N} K H}{\nu \delta}} \log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\nu \delta} + \frac {1}{\kappa} d ^ {2} ((\log K) ^ {3 / 2} \log M) ^ {2}\right) \cdot \sqrt {K}}{H ^ {3 / 2} \sqrt {\log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\nu \delta}} \left(\sqrt {d _ {\nu} \log \frac {\mathcal {N} K H}{\nu \delta} \log \frac {\mathcal {N} \mathcal {N} _ {b} K H}{\nu \delta}} + d (\log K) ^ {3 / 2} \log M\right)}\right)
$$

$$
\leqslant \mathcal {O} \left(\frac {K}{H ^ {3} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}} + \left(\frac {d _ {\nu} \sqrt {\log \frac {\mathcal {N K H}}{\nu \delta}} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta} + \frac {1}{\kappa} d ^ {2} ((\log K) ^ {3 / 2} \log M) ^ {2}}{H ^ {3 / 2} \sqrt {d _ {\nu} \log \frac {\mathcal {N K H}}{\nu \delta}} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}}\right) ^ {2}\right)
$$

$$
\leqslant \mathcal {O} \left(\frac {K}{H ^ {3} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}} + \left(\sqrt {\frac {d _ {\nu}}{H ^ {3}}} + \frac {d ^ {2} ((\log K) ^ {3 / 2} \log M) ^ {2}}{\kappa \cdot H ^ {3 / 2} \sqrt {d _ {\nu} \log \frac {\mathcal {N K H}}{\nu \delta}} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}}\right) ^ {2}\right)
$$

$$
\leqslant \mathcal {O} \left(\frac {K}{H ^ {3} \log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}} + \frac {d _ {\nu}}{H ^ {3}} + \frac {d ^ {4} ((\log K) ^ {3 / 2} \log M) ^ {4}}{\kappa^ {2} d _ {\nu} H ^ {3} \cdot \log \frac {\mathcal {N K H}}{\nu \delta} (\log \frac {\mathcal {N N} _ {b} K H}{\nu \delta}) ^ {2}}\right),
$$

where the second inequality also follows from the AM-GM inequality and the last inequality holds due to the fact that $(a + b)^2 \leqslant 2a^2 + 2b^2$ for any $a, b \in \mathbb{R}^+$ . This concludes the proof.

# D.8. Proof of Theorem 5.1

Now, we are ready to provide the proof of Theorem 5.1. To start, we formally restate the theorem.

Theorem D.25 (Restatement of Theorem 5.1, Regret upper bound of MNL-VQL). Suppose Assumptions 3.1 and 3.3 hold. We assume that we have the generalized Eluder dimension $\dim_{\nu,K}(\mathcal{F}_h)$ , for $h \in [H]$ , as defined in Definition 3.5 with $\rho = 1$ , and access to a consistent bonus oracle $\mathcal{B}$ satisfying Definition B.1 with $\epsilon_b = \mathcal{O}(1/KH)$ . Let $d_\nu = \frac{1}{H} \sum_{h=1}^{H} \dim_{\nu,K}(\mathcal{F}_h)$ with $\nu = \sqrt{1/KH}$ , and set $u_k = \mathcal{O}\left(\sqrt{\log \mathcal{N}} \cdot (\log \mathcal{NN}_b \cdot H^{5/2} \sqrt{d_\nu} + dH^{5/2} \sqrt{\log \mathcal{NN}_b}) / \sqrt{K}\right)$ . Then, for any $\delta < 1/(H^2 + 15)$ , with probability at least $1 - \delta$ , the regret of MNL-VQL is upper-bounded by:

$$
\mathbf {R e g r e t} (\mathcal {M}, K) = \mathcal {O} \left(d \sqrt {H K} (\log K) ^ {3 / 2} \log M + \sqrt {d _ {\nu} H K \cdot \log \frac {\mathcal {N} K H}{\delta}} + \frac {1}{\kappa} d ^ {2} H ^ {2} (\log K) ^ {3} (\log M) ^ {2}\right)
$$

$$
+ \mathcal {O} \left(d _ {\nu} H ^ {5} \log \frac {\mathcal {N} K H}{\delta} \cdot \left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {2} + \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot K H \epsilon_ {b}\right),
$$

where $d$ is the feature dimension of the MNL preference model, $\mathcal{N}$ is the maximum size of the function class, i.e., $\mathcal{N} = \max_{h\in [H]}\left|\mathcal{F}_h\right|$ , and $\mathcal{N}_b$ is the size of the bonus function class, i.e., $\mathcal{N}_b = |\mathcal{W}|$ .

Proof of Theorem 5.1. When the event $\mathcal{E}^{\theta}\bigcap \mathcal{E}_{\leqslant K}$ happens (with probability at least $1 - 2\delta$ ), we can bound the regret as

follows:

$$
\begin{array}{l} \mathbf {R e g r e t} (\mathcal {M}, K) = \sum_ {k = 1} ^ {K} \left(V _ {1} ^ {\star} - V _ {1} ^ {\pi_ {k}}\right) \left(s _ {1} ^ {k}\right) \leqslant \sum_ {k = 1} ^ {K} \left(V _ {1} ^ {k} - V _ {1} ^ {\pi_ {k}}\right) \left(s _ {1} ^ {k}\right) = \sum_ {k = 1} ^ {K} \left(Q _ {1} ^ {k} - Q _ {1} ^ {\pi_ {k}}\right) \left(s _ {1} ^ {k}, A _ {1} ^ {k}\right) \\ \leqslant \mathcal {O} (1) + \sum_ {k = 2} ^ {K} \left(Q _ {1} ^ {k} - Q _ {1} ^ {\pi_ {k}}\right) \left(s _ {1} ^ {k}, A _ {1} ^ {k}\right) \\ = \mathcal {O} (1) + \sum_ {k \in \mathcal {K} _ {\mathrm{o}} \backslash \{1 \}} \left(Q _ {1} ^ {k} - Q _ {1} ^ {\pi_ {k}}\right) \left(s _ {1} ^ {k}, A _ {1} ^ {k}\right) + \sum_ {k \in \mathcal {K} _ {\mathrm{oo}} \backslash \{1 \}} \left(Q _ {1} ^ {k} - Q _ {1} ^ {\pi_ {k}}\right) \left(s _ {1} ^ {k}, A _ {1} ^ {k}\right), \tag {D.61} \\ \end{array}
$$

where the first inequality holds by Lemma D.15.

For $k \in \mathcal{K}_0 \backslash \{1\}$ , recall that $Q_h^k(s, A_h^k) = \sum_{a_1 \in A_h^k} \widetilde{\mathcal{P}}_{h,1}^k(a_1 | s, A_h^k)f_{h,1}^k(s_h^k, a_1)$ for all $h \in [H]$ , as defined in (D.18). Therefore, we have

$$
\begin{array}{l} \left(Q _ {1} ^ {k} - Q _ {1} ^ {\pi_ {k}}\right) \left(s _ {1} ^ {k}, A _ {1} ^ {k}\right) = \sum_ {a _ {1} \in A _ {1} ^ {k}} \widetilde {\mathcal {P}} _ {1, 1} ^ {k} (a _ {1} | s _ {1} ^ {k}, A _ {1} ^ {k}) f _ {1, 1} ^ {k} (s _ {h} ^ {k}, a _ {1}) - \sum_ {a _ {1} \in A _ {1} ^ {k}} \mathcal {P} _ {1} (a _ {1} | s _ {1} ^ {k}, A _ {1} ^ {k}) \overline {{Q}} _ {1} ^ {\pi_ {k}} (s _ {1} ^ {k}, a _ {1}) \\ = \sum_ {a _ {1} \in A _ {1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {1, 1} ^ {k} (a _ {1} | s _ {1} ^ {k}, A _ {1} ^ {k}) - \mathcal {P} _ {1} (a _ {1} | s _ {1} ^ {k}, A _ {1} ^ {k})\right) f _ {1, 1} ^ {k} (s _ {h} ^ {k}, a _ {1}) \\ + \sum_ {a _ {1} \in A _ {1} ^ {k}} \mathcal {P} _ {1} (a _ {1} | s _ {1} ^ {k}, A _ {1} ^ {k}) \left(f _ {1, 1} ^ {k} - \overline {{Q}} _ {1} ^ {\pi_ {k}}\right) (s _ {1} ^ {k}, a _ {1}) \\ = \sum_ {a _ {1} \in A _ {1} ^ {k}} \left(\widetilde {\mathcal {P}} _ {1, 1} ^ {k} (a _ {1} | s _ {1} ^ {k}, A _ {1} ^ {k}) - \mathcal {P} _ {1} (a _ {1} | s _ {1} ^ {k}, A _ {1} ^ {k})\right) f _ {1, 1} ^ {k} (s _ {h} ^ {k}, a _ {1}) + \left(f _ {1, 1} ^ {k} - \overline {{Q}} _ {1} ^ {\pi_ {k}}\right) (s _ {1} ^ {k}, a _ {1}) \\ + \mathbb {E} _ {\mathcal {P}} \left[ \left(f _ {1, 1} ^ {k} - \overline {{Q}} _ {1} ^ {\pi_ {k}}\right) (s _ {1} ^ {k}, a _ {1}) \mid s _ {1} ^ {k}, A _ {1} ^ {k} \right] - \left(f _ {1, 1} ^ {k} - \overline {{Q}} _ {1} ^ {\pi_ {k}}\right) (s _ {1} ^ {k}, a). \\ \end{array}
$$

Then, by applying Lemma D.20 with $h_{k} = H + 1$ , we have

$$
\begin{array}{l} \left(Q _ {1} ^ {k} - Q _ {1} ^ {\pi_ {k}}\right) (s _ {1} ^ {k}, A _ {1} ^ {k}) \leqslant \sum_ {h = 1} ^ {H} \sum_ {a _ {h} \in A _ {h} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h, 1} ^ {k} (a _ {h} | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a _ {h} | s _ {h} ^ {k}, A _ {h} ^ {k})\right) f _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h}) \\ + 2 \sum_ {h = 1} ^ {H} b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \sum_ {h = 1} ^ {H} \dot {\zeta} _ {h, 1} ^ {k} + \sum_ {h = 2} ^ {H} \zeta_ {h, 1} ^ {k}, \tag {D.62} \\ \end{array}
$$

where $\zeta_{h,1}^{k} = \mathbb{E}_{\mathbb{P}}\left[(V_{h,1}^{k} - V_{h}^{\pi_{k}})(s_{h})\mid s_{h - 1}^{k},a_{h - 1}^{k}\right] - (V_{h,1}^{k} - V_{h}^{\pi_{k}})(s_{h}^{k})$ and $\dot{\zeta}_{h,1}^{k} = \mathbb{E}_{\mathcal{P}}\left[\left(f_{h,1}^{k} - \overline{Q}_{h}^{\pi_{k}}\right)(s_{h}^{k},a_{h})\mid s_{h}^{k},A_{h}^{k}\right] -$ $\left(f_{h,1}^{k} - \overline{Q}_{h}^{\pi_{k}}\right)(s_{h}^{k},a_{h}^{k}).$

Now, we consider the case where $k \in \mathcal{K}_{\infty} \backslash \{1\}$ . In this cases, note that $h_k \in [H]$ . Similar to the above analysis, by Lemma D.20, we get

$$
\begin{array}{l} \left(Q _ {1} ^ {k} - Q _ {1} ^ {\pi_ {k}}\right) (s _ {1} ^ {k}, A _ {1} ^ {k}) \leqslant \sum_ {h = 1} ^ {h _ {k} - 1} \sum_ {a _ {h} \in A _ {h} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h, 1} ^ {k} (a _ {h} | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a _ {h} | s _ {h} ^ {k}, A _ {h} ^ {k})\right) f _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h}) \\ + \sum_ {h = h _ {k}} ^ {H} \sum_ {a _ {h} \in A _ {h} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h, 2} ^ {k} (a _ {h} | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a _ {h} | s _ {h} ^ {k}, A _ {h} ^ {k})\right) f _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h}) \\ + 2 \sum_ {h = 1} ^ {H} b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + 2 \sum_ {h = h _ {k}} ^ {H} b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) + \sum_ {h = 1} ^ {h _ {k} - 1} \dot {\zeta} _ {h, 1} ^ {k} + \sum_ {h = 2} ^ {h _ {k} - 1} \zeta_ {h, 1} ^ {k} + \sum_ {h = h _ {k}} ^ {H} \dot {\zeta} _ {h, 2} ^ {k} + \sum_ {h = h _ {k}} ^ {H} \zeta_ {h, 2} ^ {k}, \tag {D.63} \\ \end{array}
$$

where $\zeta_{h,2}^{k} = \mathbb{E}_{\mathbb{P}}\left[(V_{h,2}^{k} - V_{h}^{\pi_{k}})(s_{h})\mid s_{h - 1}^{k},a_{h - 1}^{k}\right] - (V_{h,2}^{k} - V_{h}^{\pi_{k}})(s_{h}^{k})$ and $\dot{\zeta}_{h,2}^{k} = \mathbb{E}_{\mathcal{P}}\left[\left(f_{h,2}^{k} - \overline{Q}_{h}^{\pi_{k}}\right)(s_{h}^{k},a_{h})\mid s_{h}^{k},A_{h}^{k}\right] -$ $\left(f_{h,2}^{k} - \overline{Q}_{h}^{\pi_{k}}\right)(s_{h}^{k},a_{h}^{k}).$

Plugging (D.62) and (D.63) into (D.61), and denoting $J(k,h):[K]\times [H]\to \{1,2\}$ as the one-to-one function that maps from $[K]\times [H]$ to the index set $\{1,2\}$ such that $A_h^k = A_{h,J(k,h)}^k\in \mathrm{argmax}_{A\in \mathcal{A}}\sum_{a\in A}\widetilde{\mathcal{P}}_{h,J(k,h)}^k (a|s_h^k,A)f_{h,J(k,h)}^k (s_h^k,a)$ , we obtain that

Regret(M, K)

$$
\leqslant \mathcal {O} (1) + \sum_ {k = 2} ^ {K} \sum_ {h = 1} ^ {H} \sum_ {a _ {h} \in A _ {h} ^ {k}} \left(\widetilde {\mathcal {P}} _ {h, J (k, h)} ^ {k} (a _ {h} | s _ {h} ^ {k}, A _ {h} ^ {k}) - \mathcal {P} _ {h} (a _ {h} | s _ {h} ^ {k}, A _ {h} ^ {k})\right) f _ {h, J (k, h)} ^ {k} (s _ {h} ^ {k}, a _ {h})
$$

$$
+ 2 \sum_ {k = 2} ^ {K} \sum_ {h = 1} ^ {H} \min \left\{1 + L, b _ {h, 1} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) \right\} + 2 \sum_ {k \in \mathcal {K} _ {\mathrm{oo}}} \sum_ {h = h _ {k}} ^ {H} \min \left\{1 + L, b _ {h, 2} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}) \right\}
$$

$$
+ \sum_ {k = 2} ^ {K} \sum_ {h = 1} ^ {h _ {k} - 1} \dot {\zeta} _ {h, 1} ^ {k} + \sum_ {k = 2} ^ {K} \sum_ {h = 2} ^ {h _ {k} - 1} \zeta_ {h, 1} ^ {k} + \sum_ {k = 2} ^ {K} \sum_ {h = h _ {k}} ^ {H} \dot {\zeta} _ {h, 2} ^ {k} + \sum_ {k = 2} ^ {K} \sum_ {h = h _ {k}} ^ {H} \zeta_ {h, 2} ^ {k}.
$$

Now, by applying the results from Lemma D.13 to bound the first term (which holds with probability at least $1 - 2\delta$ ), Lemma D.23 for the second term (which holds with probability at least $1 - 7\delta$ ), Lemma D.22 for the third term, and applying the Azuma-Hoeffding inequality to the remaining terms (which holds with probability at least $1 - 4\delta$ ), we get

Regret(M, K)

$$
\leqslant \mathcal {O} \left(d \sqrt {H K |} (\log K) ^ {3 / 2} \log M + \frac {1}{\kappa} d ^ {2} H (\log K) ^ {3} (\log M) ^ {2}\right)
$$

$$
+ \mathcal {O} \left(\sqrt {d _ {\nu} H K \cdot \log \frac {\mathcal {N} K H}{\delta}} + \frac {1}{\sqrt {\kappa}} d H ^ {7 / 2} \sqrt {d _ {\nu}} (\log K) ^ {3 / 2} \log M \cdot \log \frac {\mathcal {N N} _ {b} K H}{\delta} \cdot \sqrt {\log \frac {\mathcal {N} K H}{\delta}}\right)
$$

$$
+ \mathcal {O} \left(d _ {\nu} H ^ {7 / 2} \log \frac {\mathcal {N} K H}{\delta} \cdot \left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {3 / 2} + \sqrt {\log \frac {\mathcal {N} K H}{\delta}} \cdot \left(K H \epsilon_ {b} + \sqrt {d _ {\nu} K H ^ {3} \delta}\right)\right)
$$

$$
+ \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\delta} \log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \left(\sqrt {H ^ {2} \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} u _ {k}} + \sqrt {H ^ {2} | \mathcal {K} _ {\mathrm{oo}} |}\right)\right)
$$

$$
+ \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \left(H \sqrt {d _ {\nu} | \mathcal {K} _ {\mathrm{oo}} |} + | \mathcal {K} _ {\mathrm{oo}} | H \epsilon_ {b}\right)\right)
$$

$$
\leqslant \mathcal {O} \left(d \sqrt {H K} | (\log K) ^ {3 / 2} \log M + \frac {1}{\kappa} d ^ {2} H (\log K) ^ {3} (\log M) ^ {2}\right)
$$

$$
+ \mathcal {O} \left(\sqrt {d _ {\nu} H K \cdot \log \frac {\mathcal {N} K H}{\delta}} + \frac {1}{\sqrt {\kappa}} d H ^ {7 / 2} \sqrt {d _ {\nu}} (\log K) ^ {3 / 2} \log M \cdot \log \frac {\mathcal {N N} _ {b} K H}{\delta} \cdot \sqrt {\log \frac {\mathcal {N} K H}{\delta}}\right)
$$

$$
+ \mathcal {O} \left(d _ {\nu} H ^ {7 / 2} \log \frac {\mathcal {N} K H}{\delta} \cdot \left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {3 / 2} + \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \left(K H \epsilon_ {b} + \sqrt {d _ {\nu} K H ^ {3} \delta}\right)\right)
$$

$$
+ \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\delta} \log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \sqrt {H ^ {2} \sum_ {k \in \mathcal {K} _ {0}} u _ {k}}\right), \tag {D.64}
$$

where the second inequality holds by Lemma D.24 with probability at least $1 - 2\delta$ , and use the fact that $|K_{oo}|H\epsilon_{b} \leqslant KH\epsilon_{b}$ . Now, we apply the AM-GM inequality to the term $\mathcal{O}\left(\frac{1}{\sqrt{\kappa}} dH^{7/2}\sqrt{d_{\nu}}(\log K)^{3/2} \log M \cdot \log \frac{\mathcal{N}\mathcal{N}_{b}KH}{\delta} \cdot \sqrt{\log \frac{\mathcal{N}KH}{\delta}}\right)$ , thus we get

$$
\mathcal {O} \left(\frac {1}{\sqrt {\kappa}} d H ^ {7 / 2} \sqrt {d _ {\nu}} (\log K) ^ {3 / 2} \log M \cdot \log \frac {\mathcal {N N} _ {b} K H}{\delta} \cdot \sqrt {\log \frac {\mathcal {N} K H}{\delta}}\right)
$$

$$
\leqslant \mathcal {O} \left(\frac {1}{\kappa} d ^ {2} H ^ {2} (\log K) ^ {3} (\log M) ^ {2} + d _ {\nu} H ^ {5} \left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {2} \cdot \log \frac {\mathcal {N} K H}{\delta}\right). \tag {D.65}
$$

Furthermore, by substituting the chosen values of $u_{k}$ and applying the AM-GM inequality, we get

$$
\begin{array}{l} \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\delta} \log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot \sqrt {d _ {\nu} H} \cdot \sqrt {H ^ {2} \sum_ {k \in \mathcal {K} _ {\mathrm{o}}} u _ {k}}\right) \\ = \mathcal {O} \left(\sqrt {d _ {\nu} H K \cdot \log \frac {\mathcal {N} K H}{\delta}} + d ^ {2} H ^ {2} (\log K) ^ {3} (\log M) ^ {2} + d _ {\nu} H ^ {5} \log \frac {\mathcal {N} K H}{\delta} \cdot \left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {2}\right) \\ + \mathcal {O} \left(\sqrt {\log \frac {\mathcal {N} K H}{\delta}} \cdot K H \epsilon_ {b}\right). \tag {D.66} \\ \end{array}
$$

Then, by plugging (D.65) and (D.66) into (D.64), and setting $\delta < \frac{1}{H^{2}+15}$ , we derive that

$$
\mathbf {R e g r e t} (\mathcal {M}, K) = \mathcal {O} \left(d \sqrt {H K} (\log K) ^ {3 / 2} \log M + \sqrt {d _ {\nu} H K \cdot \log \frac {\mathcal {N} K H}{\delta}} + \frac {1}{\kappa} d ^ {2} H ^ {2} (\log K) ^ {3} (\log M) ^ {2}\right)
$$

$$
+ \mathcal {O} \left(d _ {\nu} H ^ {5} \log \frac {\mathcal {N} K H}{\delta} \cdot \left(\log \frac {\mathcal {N N} _ {b} K H}{\delta}\right) ^ {2} + \sqrt {\log \frac {\mathcal {N N} _ {b} K H}{\delta}} \cdot K H \epsilon_ {b}\right).
$$

We conclude the proof of Theorem 5.1.

![](images/ff1dfccab26f43ed94ecb92227b4a0dc3351f98eda323c0b8d8d8855e84d8724.jpg)

# E. Proof of Theorem 5.2

In this section, we introduce several properties of linear function class. We formally define the linear MDP as follows:

Definition E.1 (Linear MDPs, Yang & Wang 2019; Jin et al. 2020). An MDP $\mathcal{M}$ is a linear MDP if we have a known feature mapping $\psi : \mathcal{S} \times \mathcal{I} \to \mathbb{R}^{d^{\mathrm{lin}}}$ , and there exist $d^{\mathrm{lin}}$ unknown (signed) measures $\boldsymbol{\mu}_h^\star = (\mu_h^{(1)}, \ldots, \mu_h^{(d^{\mathrm{lin}})})$ over $\mathcal{S}$ and unknown vector $\mathbf{w}_h^\star \in \mathbb{R}^{d^{\mathrm{lin}}}$ , such that for any $(s, a) \in \mathcal{S} \times \mathcal{I}$ , we have $\mathbb{P}_h(\cdot | s, a) = \langle \psi(s, a), \boldsymbol{\mu}_h^\star(\cdot) \rangle$ and $r_h(s, a) = \langle \psi(s, a), \mathbf{w}_h^\star\rangle$ . We assume that $\sup_{(s, a) \in \mathcal{S} \times \mathcal{I}} \| \psi(s, a) \|_2 \leqslant 1$ , $\max\{\| \sum_{s \in \mathcal{S}} |\boldsymbol{\mu}_h^\star(s)| \|_2, \| \mathbf{w}_h^\star \|_2\} \leqslant \sqrt{d^{\mathrm{lin}}}$ for all $h \in [H]$ .

In this proof, to explicitly indicate the dependency on parameters, we denote the linear MDPs as $M_{\theta^{\star},\mu^{\star},w^{\star}}$ , where $\theta^{\star}=\{\theta_{h}^{\star}\}_{h=1}^{H}$ , $\mu^{\star}=\{\mu_{h}^{\star}\}_{h=1}^{H}$ , and $w^{\star}=\{w_{h}^{\star}\}_{h=1}^{H}$ .

We also assume that $\sum_{h=1}^{H} r_h \in [0, 1]$ . Proposition 2.3 of Jin et al. (2020) shows that linear MDPs satisfy Assumption 3.3 under the linear function class $\mathcal{F}_h^{\mathrm{lin}}$ defined as follows:

$$
\mathcal {F} _ {h} ^ {\mathrm{lin}} := \left\{\langle \psi (\cdot , \cdot), \boldsymbol {\omega} _ {h} \rangle : \boldsymbol {\omega} _ {h} \in \mathbb {R} ^ {d ^ {\mathrm{lin}}}, \| \boldsymbol {\omega} _ {h} \| _ {2} \leqslant 2 \sqrt {d ^ {\mathrm{lin}}} \right\}, \quad \text { for   any } h \in [ H ]. \tag {E.1}
$$

For linear MDPs, let $\mathcal{F}_{h}^{\mathrm{lin}}(\epsilon_{c})$ be an $\epsilon_{c}$ -cover of $F_{h}^{lin}$ under the $\ell_{\infty}$ norm, so that

$$
\log \left| \mathcal {F} _ {h} ^ {\text { lin }} (\epsilon_ {c}) \right| = \mathcal {O} \left(d ^ {\text { lin }} \log \frac {2 \sqrt {d ^ {\text { lin }}}}{\epsilon_ {c}}\right) = \tilde {\mathcal {O}} \left(d ^ {\text { lin }}\right). \tag {E.2}
$$

Then, the definition of generalized Eluder dimension for the linear function class $\mathcal{F}_h^{\mathrm{lin}}$ can be expressed as:

Lemma E.2 (Lemma 3 of Agarwal et al. 2023). For the class $\mathcal{F}_h^{\mathrm{lin}}$ defined in (E.1), letting $\mathcal{F}_h^{\mathrm{lin}}(\epsilon_c)$ be the $\epsilon_c$ -cover of $\mathcal{F}_h^{\mathrm{lin}}$ for some $\epsilon_c > 0$ , we have

$$
\dim_ {\nu , K} (\mathcal {F} _ {h} ^ {l i n} (\epsilon_ {c})) \leqslant \dim_ {\nu , K} (\mathcal {F} _ {h} ^ {l i n}) = \mathcal {O} \left(d ^ {l i n} \log \left(1 + \frac {K}{\nu^ {2} \rho}\right)\right) = \tilde {\mathcal {O}} (d ^ {l i n}).
$$

The bonus oracle for linear MDPs can be easily instantiated using the standard elliptical bonus, and, as demonstrated in the next lemma, satisfies all the required properties for a bonus oracle.

Lemma E.3 (Bonus oracle $\mathcal{B}$ for linear MDPs, Lemma 7 of Agarwal et al. 2023). Given $K$ , $H \in \mathbb{Z}_+$ , suppose all $\beta_h^k \leqslant \beta$ and $\beta_h^k$ is non-decreasing in $k \in [K]$ for each $h \in [H]$ . For any $k \geqslant 1$ , $h \in [H]$ , variances $\{\bar{\sigma}_h^\tau\}_{\tau=1}^h$ satisfying $\bar{\sigma}_h^\tau \geqslant \nu$ for some $\nu > 0$ , dataset $\mathcal{D}_h^{k-1} = \{\psi(s_h^\tau, a_h^\tau), a_h^\tau, r_h^\tau, \psi(s_{h+1}^\tau, a_{h+1}^\tau)\}_{\tau=1}^{k-1}$ , function class $\mathcal{F}_h^k$ and $\hat{f}_h^k \in \mathcal{F}_h^k$ defined via weighted regression in (3), and parameters $\rho, \epsilon_c > 0$ , let $\mathcal{B}(\{\bar{\sigma}_h^\tau\}_{\tau=1}^h, \mathcal{D}_h^{k-1}, \mathcal{F}_h^k, \hat{f}_h^k, \beta_h^k, \rho, \epsilon_c) = \| \psi(s, a) \|_{(\Sigma_h^k)^{-1}} \sqrt{(\beta_h^k)^2 + \rho}$ , where $\Sigma_h^k = \frac{\rho}{16d} I + \sum_{\tau=1}^{k-1} \frac{1}{(\bar{\sigma}_h^\tau)^2} \psi(s_h^\tau, a_h^\tau) \psi(s_h^\tau, a_h^\tau)^\top$ . For any choice of covering radius $\epsilon_c \leqslant \nu \sqrt{\rho/8K}$ , the oracle satisfies all the properties of Definition B.1 with

$$
\log \mathcal {N} _ {b} = \log | \mathcal {W} | = \mathcal {O} \left((d ^ {l i n}) ^ {2} \log \left(1 + d ^ {l i n} \sqrt {d ^ {l i n}} \beta / (\rho \epsilon_ {c} ^ {2})\right)\right) = \tilde {\mathcal {O}} \left((d ^ {l i n}) ^ {2}\right).
$$

Theorem E.4 (Formal version of Theorem 5.2, Regret upper bound of MNL-VQL for linear MDPs). Under the same conditions with Theorem 5.1, suppose that the underlying MDP has linear transition probabilities and rewards, so that the function class for linear MDPs, $\mathcal{F}_h^{\mathrm{lin}}$ , satisfies Assumption 3.3. Let $\mathcal{F}_h^{\mathrm{lin}}(\epsilon_c)$ be an $\epsilon_c$ -cover of $\mathcal{F}_h^{\mathrm{lin}}$ under $\ell_{\infty}$ norm. We set $\rho = 1$ , $u_k = \tilde{\Theta}((d^{\mathrm{lin}})^3 H^{5/2} + d(d^{\mathrm{lin}})^{3/2}H^{5/2}) / \sqrt{K})$ , $\nu = \sqrt{1/HK}$ , $\epsilon_b = \epsilon_c \leqslant 1/(8HK)$ and $\delta < 1/(H^2 + 15)$ . Then, with probability at least $1 - \delta$ , the cumulative regret of MNL-VQL, with bonus oracle defined in Lemma E.3, is upper-bounded by

$$
\mathbf {R e g r e t} \left(\mathcal {M} _ {\boldsymbol {\theta} ^ {\star}, \boldsymbol {\mu} ^ {\star}, \mathbf {w} ^ {\star}}, K\right) = \tilde {\mathcal {O}} \Bigg (\underbrace {d \sqrt {H K} + \frac {1}{\kappa} d ^ {2} H ^ {2}} _ {\text {regret from MNL model}} + \underbrace {d ^ {l i n} \sqrt {H K} + (d ^ {l i n}) ^ {6} H ^ {5}} _ {\text {regret from linear MDPs}} \Bigg).
$$

Proof of Theorem 5.2. We apply the above results to linear MDPs $\mathcal{M}_{\theta^{\star},\mu^{\star},\mathbf{w}^{\star}}$ with function class $\mathcal{F}_h^{\mathrm{lin}}(\epsilon_c)$ , $h\in [H]$ , and bonus oracle $\mathcal{B}$ . From (E.2), we know that $\mathcal{N} = \tilde{\mathcal{O}}(d^{\mathrm{lin}})$ . Additionally, Lemma E.2 shows that $d_{\nu} = \tilde{\mathcal{O}}(d^{\mathrm{lin}})$ . Therefore, by combining these results with Theorem 5.1 and Lemma E.3, we can establish the upper regret bounds for linear MDPs.

$$
\mathbf {R e g r e t} \left(\mathcal {M} _ {\boldsymbol {\theta} ^ {\star}, \boldsymbol {\mu} ^ {\star}, \mathbf {w} ^ {\star}}, K\right) = \tilde {\mathcal {O}} \Big (d \sqrt {H K} + d ^ {\mathrm{lin}} \sqrt {H K} + \frac {1}{\kappa} d ^ {2} H ^ {2} + (d ^ {\mathrm{lin}}) ^ {6} H ^ {5} \Big),
$$

where we set $\rho = 1$ , $u_{k} = \tilde{\Theta}\left((d^{\mathrm{lin}})^{3}H^{5 / 2} + d(d^{\mathrm{lin}})^{3 / 2}H^{5 / 2}\right) / \sqrt{K}$ , $\nu = \sqrt{1 / HK}$ , $\epsilon_{b} = \epsilon_{c} \leqslant 1 / (8HK)$ and $\delta < 1 / (H^2 + 15)$ .

# F. Proof of Theorem 5.3

In this section, we provide a regret lower bound for linear MDPs with preference model. We construct a hard instance $\mathcal{M}(\mathcal{S},\mathcal{I},\mathcal{A},M,\{\mathcal{P}_{h}\}_{h=1}^{H},\{\mathbb{P}_{h}\}_{h=1}^{H},\{r_{h=1}^{H}\},H)$ , illustrated as in Figure F.1. This instance is based on an $H+1$ -layered structure, where each layer is a variation of the hard-to-learn MDPs introduced in Zhou et al. (2021b).

Without loss of generality, we assume that $d^{lin} \geqslant 6$ and that $d^{lin} - 5$ is divisible by 2. $^{3}$ Let $i \in [H + 2]$ represent the layer index. For each layer $i \in [H + 2]$ , there are H - i + 3 states, denoted as $x_{i}^{(i)}, \ldots, x_{H+2}^{(i)}$ , where $x_{H+2}^{(i)}$ is the absorbing state. Furthermore, there is a global absorbing state $x_{0}$ , which can only be reached at any state and horizon through the user's choice of the outside option $a_{0}$ (not choosing any item in the assortment). Thus, there are $(H + 1)(H + 2)/2 + 1$ states in total in the set of states S. There are $2^{(d^{\text{lin}} - 5)/2} + 1$ items, so the item set is $\mathcal{I} = \{-1, 1\}^{(d^{\text{lin}} - 5)/2} \cup \{a_{0}\}$ . The set of candidate assortments follows the definition in Section 3, i.e., $A = \{A \subseteq I : a_{0} \in A, 1 \leqslant |A \setminus \{a_{0}\}| \leqslant M\}$ .

# F.1. Construction of linear transitions and rewards

At each episode $k \in [K]$ , the agent starts from the fixed initial state $x_1^{(1)}$ . We define $\mathbf{a}_h^\star$ as an item such that $\mathbf{a}_h^\star \in \operatorname{argmax}_{\mathbf{a} \in \mathcal{I} \setminus \{\mathbf{a}_0\}} \langle \boldsymbol{\mu}_h, \mathbf{a} \rangle$ , where $\boldsymbol{\mu}_h \in \{-\Delta, \Delta\}^{(d^{\mathrm{lin}} - 5)/2}$ with $\Delta = \sqrt{\delta/K}/(4\sqrt{2})$ and $\delta = 1/H$ .

If the state is $x_{h}^{(i)}$ with $i \in [H + 1]$ and $h \in [i, H + 1]$ , and the user chooses the item $a_{h}^{\star}$ , the agent remains in the same layer i and receives a reward of $\gamma^{i-1}/H$ , where $\gamma = \frac{H}{1+H}$ . The next state will be either $x_{h+1 \wedge H+1}^{(i)}$ or $x_{H+2}^{(i)}$ , with probabilities $1 - (\delta + \langle \boldsymbol{\mu}_{h}, \mathbf{a} \rangle)$ and $\delta + \langle \boldsymbol{\mu}_{h}, \mathbf{a} \rangle$ , respectively. If the user chooses an item $a \neq a_{0}$ , $a_{h}^{\star}$ in the state $x_{h}^{(i)}$ with $i \in [H + 1]$ and $h \in [i, H + 1]$ , the agent obtains a reward of $\gamma^{i}/H$ and transitions to $x_{h+1 \wedge H+1}^{(i+1)}$ or $x_{H+2}^{(i+2 \wedge H+2)}$ , with probabilities

![](images/4ee44741bbeb4fc02bc9a0e77e7f2547a84f2f9748ccf2cc9203ee77fe815184.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Layer 1
        x0["x0"] --> x1["1(1)"]
        x0 --> x2["1-δ - ⟨μh, ah*⟩, r = 1/H"]
        x0 --> x3["1-δ - ⟨μh, ah*⟩, r = γ/H"]
        x0 --> xH2["..."]
        x0 --> xH21["..."]
        x0 --> xH22["..."]
        x0 --> xH23["..."]
        x0 --> xH24["..."]
    end

    subgraph Layer 2
        x0 --> x2["1(2)"]
        x0 --> x3["1-δ - ⟨μh, ah*⟩, r = 1/H"]
        x0 --> xH2["..."]
        x0 --> xH22["..."]
        x0 --> xH23["..."]
        x0 --> xH24["..."]
        x0 --> xH24["..."]
    end

    subgraph Layer 3
        x0 --> x3["1(3)"]
        x0 --> xH2["..."]
        x0 --> xH22["..."]
        x0 --> xH23["..."]
        x0 --> xH24["..."]
        x0 --> xH24["..."]
    end

    subgraph Layer 4
        x0 --> ...[...]
        x0 --> ...[...]
        ... --> ...[...]
        ... --> ...[...]
        ... --> ...[...]
        ... --> ...[...]
        ... --> ...[...]
    end

    subgraph Layer 5
        x0 --> ...2["..."]
        ... --> ...2["..."]
        ...2 --> ...2["..."]
        ...2 --> ...2["..."]
        ...2 --> ...2["..."]
        ...2 --> ...2["..."]
        ...2 --> ...2["..."]
    end

    subgraph Layer 6
        x0 --> ...3["..."]
        ...3 --> ...3["..."]
        ...3 --> ...3["..."]
        ...3 --> ...3["..."]
        ...3 --> ...3["..."]
        ...3 --> ...3["..."]
        ...3 --> ...3["..."]
    end

    subgraph Layer 7
        x0 --> ...4["..."]
        ...4 --> ...4["..."]
        ...4 --> ...4["..."]
        ...4 --> ...4["..."]
        ...4 --> ...4["..."]
        ...4 --> ...4["..."]
    end

    subgraph Layer 8
        x0 --> ...5["..."]
        ...5 --> ...5["..."]
        ...5 --> ...5["..."]
        ...5 --> ...5["..."]
        ...5 --> ...5["..."]
    end

    subgraph Layer 9
        x0 --> ...6["..."]
        ...6 --> ...6["..."]
        ...6 --> ...6["..."]
        ...6 --> ...6["..."]
    end

    subgraph Layer 10
        x0 --> ...7["..."]
        ...7 --> ...7["..."]
        ...7 --> ...7["..."]
        ...7 --> ...7["..."]
    end

    subgraph Layer 11
        x0 --> ...8["..."]
        ...8 --> ...8["..."]
        ...8 --> ...8["..."]
    end

    subgraph Layer 12
        x0 --> ...9["..."]
        ...9 --> ...9["..."]
        ...9 --> ...9["..."]
    end

    subgraph Layer 13
        x0 --> ...10["..."]
        ...10 --> ...10["..."]
        ...10 --> ...10["..."]
    end

    subgraph Layer 14
        x0 --> ...11["..."]
        ...11 --> ...11["..."]
        ...11 --> ...11["..."]
    end

    subgraph Layer 15
        x0 --> ...12["..."]
        ...12 --> ...12["..."]
        ...12 --> ...12["..."]
    end

    subgraph Layer 16
        x0 --> ...13["..."]
        ...13 --> ...13["..."]
        ...13 --> ...13["..."]
    end

    subgraph Layer 17
        x0 --> ...14["..."]
        ...14 --> ...14["..."]
        ...14 --> ...14["..."]
    end

    subgraph Layer 18
        x0 --> ...15["..."]
        ...15 --> ...15["..."]
        ...15 --> ...15["..."]
    end

    subgraph Layer 19
        x0 --> ...16["..."]
        ...16 --> ...16["..."]
        ...16 --> ...16["..."]
    end

    subgraph Layer 20
        x0 --> ...17["..."]
        ...17 --> ...17["..."]
        ...17 --> ...17["..."]

    Note: The diagram shows a hierarchical structure with nodes labeled by indices and parameters such as r, μ, and a.
```
</details>

Figure F.1: Inhomogeneous, hard-to-learn linear MDPs with MNL preference model. The solid line indicates the transition caused by the user choosing the item $a_{h}^{\star}$ (with a reward of $r_{h} = \gamma^{i-1}/H$ ), the dashed line shows the transition caused by the user choosing any item $a \neq a_{h}^{\star}, a_{0}$ (with a reward of $r_{h} = \gamma^{i}/H$ ), and the dotted line represents the transition caused by the user choosing the outside option $a_{0}$ (with a reward of $r_{h} = 0$ ). The blue solid line indicates a transition from the absorbing state back to itself, caused by the user choosing any item (with a reward of $r_{h} = \gamma^{i-1}/H$ ), and the red dotted line indicates a transition from the global absorbing state back to itself, caused by the user choosing any item (with a reward of $r_{h} = 0$ ).

$1 - (\delta + \langle \boldsymbol{\mu}_{h}, \mathbf{a} \rangle)$ and $\delta + \langle \boldsymbol{\mu}_{h}, \mathbf{a} \rangle$ , respectively. If the user does not choose any item, i.e., chooses the outside option $\mathbf{a}_{0}$ , in the state $x_{h}^{(i)}$ with $i \in [H + 1]$ and $h \in [i, H + 1]$ , the agent will deterministically transition to the global absorbing state $x_{0}$ and receive no reward.

If the agent is in any of the absorbing states- $x_{H+2}^{(i)}$ for $i \in [H + 2]$ -the agent will remain in the same state and receive a reward of $\gamma^{i-1}/H$ , regardless of which item (including the outside option) the user chooses.

Formally, we construct transition probabilities $\mathbb{P}_{h}(s^{\prime}|s,\mathbf{a})=\langle\psi(s,\mathbf{a}),\boldsymbol{\mu}_{h}^{\star}(s^{\prime})\rangle$ , with

$$
\psi (s, \mathbf {a}) = \left\{ \begin{array}{l l} (\alpha , \beta \mathbf {a} ^ {\top}, 0, \mathbf {0}, 0, 0, \frac {\gamma^ {i - 1}}{\sqrt {2}}) ^ {\top}, & s = x _ {h} ^ {(i)}, \mathbf {a} = \mathbf {a} _ {h} ^ {\star}, i \in [ H + 1 ], h \in [ i, H + 1 ]; \\ (0, \mathbf {0}, \alpha , \beta \mathbf {a} ^ {\top}, 0, 0, \frac {\gamma^ {i}}{\sqrt {2}}) ^ {\top}, & s = x _ {h} ^ {(i)}, \mathbf {a} \neq \mathbf {a} _ {h} ^ {\star}, \mathbf {a} _ {0}, i \in [ H + 1 ], h \in [ i, H + 1 ]; \\ (0, \mathbf {0} ^ {\top}, 0, \mathbf {0} ^ {\top}, 0, 1, 0) ^ {\top}, & s = x _ {h} ^ {(i)}, \mathbf {a} = \mathbf {a} _ {0}, i \in [ H + 1 ], h \in [ i, H + 1 ]; \\ (0, \mathbf {0} ^ {\top}, 0, \mathbf {0} ^ {\top}, \frac {1}{\sqrt {2}}, 0, \frac {\gamma^ {i - 1}}{\sqrt {2}}) ^ {\top}, & s = x _ {H + 2} ^ {(i)}, i \in [ H + 2 ], \end{array} \right. \tag {F.1}
$$

and

$$
\boldsymbol {\mu} _ {h} ^ {\star} (s ^ {\prime}) = \left\{ \begin{array}{l l} \left(\frac {1 - \delta}{\alpha}, - \frac {\boldsymbol {\mu} _ {h} ^ {\top}}{\beta}, 0, \mathbf {0}, 0, 0, 0\right) ^ {\top}, & s ^ {\prime} = x _ {h + 1 \wedge H + 1} ^ {(i)}; \\ \left(\frac {\delta}{\alpha}, \frac {\boldsymbol {\mu} _ {h} ^ {\top}}{\beta}, 0, \mathbf {0}, \sqrt {2}, 0, 0\right) ^ {\top}, & s ^ {\prime} = x _ {H + 2} ^ {(i)}; \\ \left(0, \mathbf {0}, \frac {1 - \delta}{\alpha}, - \frac {\boldsymbol {\mu} _ {h} ^ {\top}}{\beta}, 0, 0, 0\right) ^ {\top}, & s ^ {\prime} = x _ {h + 1} ^ {(i + 1)}; \\ \left(0, \mathbf {0}, \frac {\delta}{\alpha}, \frac {\boldsymbol {\mu} _ {h} ^ {\top}}{\beta}, 0, 0, 0\right) ^ {\top}, & s ^ {\prime} = x _ {H + 2} ^ {(i + 2 \wedge H + 2)}; \\ \left(0, \mathbf {0} ^ {\top}, 0, \mathbf {0}, 0, 1, 0\right) ^ {\top}, & s ^ {\prime} = x _ {0}; \\ \left(0, \mathbf {0} ^ {\top}, 0, \mathbf {0}, 0, 0, 0\right) ^ {\top}, & \text {otherwise}, \end{array} \right. \tag {F.2}
$$

where we denote $\mathbf{0}\in\mathbb{R}^{(d^{\mathrm{lin}}-5)/2}$ as the zero vector of dimension $(d^{\mathrm{lin}}-5)/2$ , and set $\gamma=\frac{H}{H+1}$ as the discount factor for transitioning to the next layer. Additionally, we choose $\delta=1/H$ , $\mu_{h}\in\{-\Delta,\Delta\}^{(d^{\mathrm{lin}}-5)/2}$ with $\Delta=\sqrt{\delta/K}/(4\sqrt{2})$ , $\alpha=\sqrt{1/(2+\Delta\cdot(d^{\mathrm{lin}}-5))}$ , and $\beta=\sqrt{\Delta/(2+\Delta\cdot(d^{\mathrm{lin}}-5))}$ .

And the parameter vectors for the linear rewards $r_h(s, \mathbf{a}) = \langle \psi(s, \mathbf{a}), \mathbf{w}_h^\star \rangle$ are as follows:

$$
\mathbf {w} _ {h} ^ {\star} = (0, \mathbf {0} ^ {\top}, 0, \mathbf {0}, 0, 0, \sqrt {2} / H) ^ {\top},
$$

which ensures that the reward function satisfies:

$$
r _ {h} \left(s, \mathbf {a}\right) = \left\{ \begin{array}{l l} \gamma^ {i - 1} / H, & s = x _ {h} ^ {(i)}, \mathbf {a} = \mathbf {a} _ {h} ^ {\star}, i \in [ H + 1 ], h \in [ i, H + 1 ]; \\ \gamma^ {i} / H, & s = x _ {h} ^ {(i)}, \mathbf {a} \neq \mathbf {a} _ {h} ^ {\star}, \mathbf {a} _ {0}, i \in [ H + 1 ], h \in [ i, H + 1 ]; \\ 0, & s = x _ {h} ^ {(i)}, \mathbf {a} = \mathbf {a} _ {0}, i \in [ H + 1 ], h \in [ i, H + 1 ]; \\ \gamma^ {i - 1} / H, & s = x _ {H + 2} ^ {(i)}, i \in [ H + 2 ], \end{array} \right.
$$

where $0 < \gamma \leqslant \frac{H}{H + 1}$ is the discount factor for transitioning to the next layer.

This parameter setting satisfies the boundedness assumption of linear MDPs (refer Definition E.1). First, we show that $\|\psi(s,\mathbf{a})\|_{2}\leqslant1$ :

$$
\| \psi (s, \mathbf {a}) \| _ {2} ^ {2} \leqslant \alpha^ {2} + \frac {d ^ {\mathrm{lin}} - 5}{2} \beta^ {2} + \frac {1}{2} = 1, \quad (\text { the   first   and   second   cases   of   (F.1) }),
$$

$$
\| \psi (s, \mathbf {a}) \| _ {2} ^ {2} = 1, \quad (\text { the   third   case   of   (F.1) }),
$$

$$
\| \psi (s, \mathbf {a}) \| _ {2} ^ {2} \leqslant \frac {1}{2} + \frac {1}{2} = 1, \quad (\text { the   fourth   case   of   (F.1) }),
$$

Moreover, since $d^{\mathrm{lin}} \geqslant 6$ and $K \geqslant 13(d^{\mathrm{lin}} - 5)^2 / H$ , we ensure that $\max \left\{\| \sum_{s \in S} \boldsymbol{\mu}_h(s) \|_2, \| \mathbf{w}_h^\star \|_2\right\} \leqslant \sqrt{d^{\mathrm{lin}}}$ :

$$
\begin{array}{l} \left\| \sum_ {s \in S} | \boldsymbol {\mu} _ {h} (s) | \right\| _ {2} ^ {2} = \frac {2 (1 - \delta) ^ {2} + 2 \delta^ {2}}{\alpha^ {2}} + \frac {\| \boldsymbol {\mu} _ {h} \| _ {2}}{\beta^ {2}} + 3 \\ \leqslant 2 (2 + \Delta \cdot (d ^ {\text { lin }} - 5)) + 2 \Delta \cdot (d ^ {\text { lin }} - 5) (2 + \Delta \cdot (d ^ {\text { lin }} - 5)) \\ \leqslant \left(2 + 2 \Delta \cdot (d ^ {\mathrm{lin}} - 5)\right) ^ {2} \leqslant d ^ {\mathrm{lin}}, \\ \end{array}
$$

$$
\text { and } \quad \| \mathbf {w} _ {h} ^ {\star} \| _ {2} ^ {2} \leqslant \frac {2}{H ^ {2}} \leqslant d ^ {\mathrm{lin}}.
$$

# F.2. Construction of MNL preference model

Inspired by the lower bound proposed in Lee & Oh (2024), we construct an adversarial setting for the MNL preference model.

We assume that $d \geqslant 2$ and that $d - 1$ is divisible by 4 (without loss of generality). Let $\epsilon \in \left(0, \frac{1}{(d - 1)\sqrt{d - 1}}\right)$ be a small positive parameter. Throughout the proof, we set $\epsilon = \sqrt{\frac{d - 1}{144C \cdot K} \cdot \frac{(H + 1)^2}{H}}$ , for some $C > 0$ . For every subset $W \subseteq [d - 1]$ , we define the corresponding parameter $\boldsymbol{\theta}_W \in \mathbb{R}^{d - 1}$ as $[\boldsymbol{\theta}_W]_j = \epsilon$ for all $j \in W$ , and $[\boldsymbol{\theta}_W]_j = 0$ for all $j \notin W$ .

Next, for any $h \in [H]$ , we define the parameter set as:

$$
\begin{array}{l} \boldsymbol {\theta} _ {h} ^ {\star} \in \Theta := \left\{\left(\boldsymbol {\theta} _ {W} ^ {\top}, - \log H\right) ^ {\top}: W \in \mathcal {W} _ {(d - 1) / 4} \right\} \\ = \left\{\left(\boldsymbol {\theta} _ {W} ^ {\top}, - \log H\right) ^ {\top}: W \subseteq [ d - 1 ], | W | = (d - 1) / 4 \right\}, \\ \end{array}
$$

where $W_{k}$ denotes the class of all subsets of $[d-1]$ of size k.

The feature vector $\phi(s, \mathbf{a})$ is invariant across the state $s$ . For each $U \in \mathcal{W}_{(d-1)/4}$ , we define vectors $z_U \in \mathbb{R}^{d-1}$ as follows:

$$
\left[ z _ {U} \right] _ {j} = 1 / \sqrt {d - 1} \quad \text { for } j \in U; \quad \left[ z _ {U} \right] _ {j} = 0 \quad \text { for } j \notin U.
$$

Let $\mathcal{Z} := \{z_U : U \in \mathcal{W}_{(d-1)/4}\}$ . We define the function $Z : \mathcal{I} \to \mathcal{Z}$ , so that $Z(\mathbf{a}) \in \mathcal{Z}$ . Then, the feature vector $\phi(s, \mathbf{a})$ is constructed as follows:

$$
\phi (s, \mathbf {a}) = \left\{ \begin{array}{l l} (Z (\mathbf {a}) ^ {\top}, 0) ^ {\top}, & \mathbf {a} \neq \mathbf {a} _ {0}; \\ (\mathbf {0}, 1) ^ {\top}, & \mathbf {a} = \mathbf {a} _ {0}, \end{array} \right.
$$

where $0 \in R^{d-1}$ . For all $V \in V_{d/4}$ and $(s, \mathbf{a}) \in \mathcal{S} \times \mathcal{I}$ , it can be verified that $\theta_{V}$ and $\phi(s, \mathbf{a})$ satisfy the boundedness in Assumption 3.1 as follows:

$$
\begin{array}{l} \left\| \phi (s, \mathbf {a}) \right\| _ {2} \leqslant \sqrt {(d - 1) \cdot 1 / (d - 1)} = 1, \\ \| \boldsymbol {\theta} _ {h} ^ {\star} \| _ {2} \leqslant \sqrt {(d - 1) \epsilon^ {2} + (- \log H) ^ {2}} \leqslant \sqrt {2} \log H =: B. \\ \end{array}
$$

Let $\mathbf{a}_h^\star$ (defined in the previous subsection) also have the maximum utility, i.e., $\mathbf{a}_h^\star \in \operatorname{argmax}_{a \in \mathcal{I} \setminus \{\mathbf{a}_0\}} \langle \boldsymbol{\theta}_h^\star, \phi(s, \mathbf{a}) \rangle$ (note that $\phi(\cdot, \cdot)$ is identical for all $s \in S$ ).

# F.3. Proof of Theorem 5.3

A good policy is one that quickly reaches the state $x_{H + 2}^{(i)}$ while remaining in the lower layers (i.e., with lower $i$ ). Recall that the item $\mathbf{a}_h^\star$ has the highest utility and, therefore, the highest choice probability. It also has the best chance of quickly reaching the state $x_{H + 2}^{(i)}$ while staying within the same layer. In other words, a good policy encourages the user to frequently select the item $\mathbf{a}_h^\star \in \operatorname{argmax}_{a\in \mathcal{I}\setminus \{\mathbf{a}_0\}}\langle \pmb {\mu}_h,\mathbf{a}\rangle = \operatorname{argmax}_{a\in \mathcal{I}\setminus \{\mathbf{a}_0\}}\langle \pmb{\theta}_h^\star ,\phi (s,\mathbf{a})\rangle$ . Note that $\mathbf{a}_h^\star$ is unique due to the way the action space and transition probabilities are constructed.

We formally restate Theorem 5.3 as follows.

Theorem F.1 (Restatement of Theorem 5.3, Regret lower bound for linear MDPs with preference feedback). Suppose that $d \geqslant 2$ , $d^{lin} \geqslant 6$ , $H \geqslant 3$ , and $K \geqslant \max \{C \cdot (d^{lin} - 5)^2 H(H + 1)^2, C' \cdot (d - 1)^4 (1 + H) / H\}$ for some constant $C, C' > 0$ . Then, for any algorithm, there exists an episodic linear MDP $\mathcal{M}_{\theta, \mu, \mathbf{w}}$ with MNL preference feedback such that the worst-case expected regret is lower bounded as follows:

$$
\sup _ {\boldsymbol {\theta}, \boldsymbol {\mu}, \mathbf {w}} \mathbb {E} _ {\boldsymbol {\theta}, \boldsymbol {\mu}, \mathbf {w}} \left[ \operatorname{Regret} \left(\mathcal {M} _ {\boldsymbol {\theta}, \boldsymbol {\mu}, \mathbf {w}}, K\right) \right] = \Omega \left(d \sqrt {H K} + d ^ {l i n} \sqrt {H K}\right).
$$

Proof of Theorem 5.3. Fix $\pmb{\theta}$ and $\pmb{\mu}$ so that we can omit the parameter dependency of $\mathbb{P}$ and $\mathcal{P}$ throughout the proof. Based on the construction of the hard instance $\mathcal{M}$ discussed in the previous subsections, the following lemma shows that the optimal assortment at horizon $h\in [H]$ is $\{\mathbf{a}_0,\mathbf{a}_h^\star \}$ .

Lemma F.2. For any $h \in [H]$ , we have $A_h^\star = \{\mathbf{a}_0, \mathbf{a}_h^\star\}$ .

Furthermore, we can bound the expected value of $\overline{Q}^{\star}$ for any assortment as follows:

Lemma F.3. For any $(A,i,h)\in\mathcal{A}\times[H]\times[H]$ , let $\tilde{\mathbf{a}}_{h}^{(i)}\in\operatorname{argmax}_{\mathbf{a}\in A\setminus\{\mathbf{a}_{0}\}}\phi(x_{h}^{(i)},\mathbf{a})^{\top}\boldsymbol{\theta}_{h}^{\star},\;\tilde{A}_{h}^{(i)}=\{\tilde{\mathbf{a}}_{h}^{(i)},\mathbf{a}_{0}\}$ , and $\bar{\mathbf{a}}_{h}^{(i)}\in\operatorname{argmax}_{\mathbf{a}\in A\setminus\{\mathbf{a}_{0}\}}\overline{Q}_{h}^{\pi}(x_{h}^{(i)},\mathbf{a})$ . For any $a^{\prime}\neq a_{0}$ , we define

$$
\begin{array}{l} \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}, \mathbf {a} ^ {\prime}) \\ := \left\{ \begin{array}{l l} \frac {\gamma^ {i - 1}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) V _ {h + 1} ^ {\pi} (x _ {h + 1} ^ {(i)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \frac {(H - h) \gamma^ {i - 1}}{H}, & \mathbf {a} ^ {\prime} = \mathbf {a} _ {h} ^ {\star}, \\ \frac {\gamma^ {i - 1}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) V _ {h + 1} ^ {\pi} (x _ {h + 1} ^ {(i)}) + \mathbb {P} _ {h} (X _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \frac {(H - h) \gamma^ {i - 1}}{H}, & \mathbf {a} ^ {\prime} \neq \mathbf {a} _ {h} ^ {\star}. \end{array} \right. \\ \end{array}
$$

Then, for any policy $\pi$ , if $K \geqslant 4(d^{lin} - 5)^2 H(H + 1)^2$ , we have

$$
\sum_ {\mathbf {a} \in A} \mathcal {P} _ {h} (\mathbf {a} | x _ {h} ^ {(i)}, A) \overline {{Q}} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a}) \leqslant \mathcal {P} _ {h} (\bar {\mathbf {a}} _ {h} ^ {(i)} | x _ {h} ^ {(i)}, \tilde {A} _ {h} ^ {(i)}) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}, \bar {\mathbf {a}} _ {h} ^ {(i)}).
$$

Now, we are ready to provide the proof of Theorem 5.3.

For any $h \in [H]$ and any $A_h \in A$ , let $\tilde{a}_h \in \arg\max_{\mathbf{a} \in A_h \setminus \{\mathbf{a}_0\}} \phi(x_h^{(i)}, \mathbf{a})^\top \boldsymbol{\theta}_h^*$ . We also denote $\tilde{A}_h = \{\tilde{a}_h, a_0\}$ and $\bar{a}_h \in \arg\max_{\mathbf{a} \in A \setminus \{\mathbf{a}_0\}} \overline{Q}_h^\pi(x_h, \mathbf{a})$ . Recall that the index i can be omitted for $\tilde{a}_h$ because the transition and choice probabilities are identical across all $x_h^{(1)}, \ldots, x_h^{(H)}$ given $a \in I$ . The change in layer i only affects the scaling of rewards, and consequently the $\overline{Q}$ -values, but the item that maximizes $\overline{Q}(x_h^{(i)}, \mathbf{a})$ remains the same across layers.

By applying Lemma F.3, the value of policy $\pi$ in state $x_{1}^{(1)}$ can be bounded as follows:

$$
V _ {1} ^ {\pi} (x _ {1} ^ {(1)}) = \sum_ {\mathbf {a} \in A _ {1}} \mathcal {P} _ {1} (\mathbf {a} | x _ {1} ^ {(1)}, A _ {1}) \overline {{Q}} _ {1} ^ {\pi} (x _ {1} ^ {(1)}, \mathbf {a}) \leqslant \mathcal {P} _ {1} (\tilde {\mathbf {a}} _ {1} | x _ {1} ^ {(1)}, \tilde {A} _ {1}) \widetilde {Q} _ {h} ^ {\pi} (x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}, \bar {\mathbf {a}} _ {1}). \tag {F.3}
$$

Moreover, according to Lemma F.2, the optimal assortment for horizon $h \in [H]$ is $A_{h}^{\star} = \{a_{0}, a_{h}^{\star}\}$ . Thus, the optimal value function in state $x_{1}^{(1)}$ can be written as follows:

$$
V _ {1} ^ {\star} (x _ {1} ^ {(1)}) = \sum_ {\mathbf {a} \in A _ {1} ^ {\star}} \mathcal {P} _ {1} (\mathbf {a} | x _ {1} ^ {(1)}, A _ {1} ^ {\star}) \overline {{Q}} _ {1} ^ {\star} (x _ {1} ^ {(1)}, \mathbf {a}) = \mathcal {P} _ {1} (\mathbf {a} _ {1} ^ {\star} | x _ {1} ^ {(1)}, A _ {1} ^ {\star}) \overline {{Q}} _ {1} ^ {\star} (x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}),
$$

where the last equality holds because $\overline{Q}_{h}^{\star}(x_{h}^{(i)},\mathbf{a}_{0})=0$ . We denote $s_{H+2}$ can be either $x_{H+2}^{(1)}$ or $x_{H+2}^{(3)}$ , depending on whether the item (for transition) is $a_{h}^{\star}$ or any $a\neq a_{h}^{\star},a_{0}$ . Then, we have

$$
\begin{array}{l} (V _ {1} ^ {\star} - V _ {1} ^ {\pi}) (x _ {1} ^ {(1)}) \geqslant \mathcal {P} _ {1} (\mathbf {a} _ {1} ^ {\star} | x _ {1} ^ {(1)}, A _ {1} ^ {\star}) \overline {{Q}} _ {1} ^ {\star} (x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) - \mathcal {P} _ {1} (\tilde {\mathbf {a}} _ {1} | x _ {1} ^ {(1)}, \tilde {A} _ {1}) \widetilde {Q} _ {h} ^ {\pi} (x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}, \bar {\mathbf {a}} _ {1}) \\ = \left(\mathcal {P} _ {1} (\mathbf {a} _ {1} ^ {\star} | x _ {1} ^ {(1)}, A _ {1} ^ {\star}) - \mathcal {P} _ {1} (\tilde {\mathbf {a}} _ {1} | x _ {1} ^ {(1)}, \tilde {A} _ {1})\right) \overline {{Q}} _ {1} ^ {\star} (x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) \\ + \mathcal {P} _ {1} (\tilde {\mathbf {a}} _ {1} | x _ {1} ^ {(1)}, \tilde {A} _ {1}) \left(\overline {{Q}} _ {1} ^ {\star} (x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) - \widetilde {Q} _ {h} ^ {\pi} (x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}, \bar {\mathbf {a}} _ {1})\right) \\ = \left(\mathcal {P} _ {1} (\mathbf {a} _ {1} ^ {\star} | x _ {1} ^ {(1)}, A _ {1} ^ {\star}) - \mathcal {P} _ {1} (\tilde {\mathbf {a}} _ {1} | x _ {1} ^ {(1)}, \tilde {A} _ {1})\right) \overline {{Q}} _ {1} ^ {\star} (x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) \\ + \mathcal {P} _ {1} (\tilde {\mathbf {a}} _ {1} | x _ {1} ^ {(1)}, \tilde {A} _ {1}) \left(\frac {1}{H} + \mathbb {P} _ {1} (x _ {2} ^ {(1)} | x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) V _ {2} ^ {\star} (x _ {2} ^ {(1)}) + \mathbb {P} _ {1} (x _ {H + 2} ^ {(1)} | x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) \frac {(H - 1)}{H} \right. \\ - \left(\frac {1}{H} + \mathbb {P} _ {1} (x _ {2} ^ {(1)} | x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) V _ {2} ^ {\pi} (x _ {2} ^ {(1)}) + \mathbb {P} _ {1} (s _ {H + 2} | x _ {1} ^ {(1)}, \bar {\mathbf {a}} _ {1}) \frac {(H - 1)}{H}\right) \\ = \left(\mathcal {P} _ {1} (\mathbf {a} _ {1} ^ {\star} | x _ {1} ^ {(1)}, A _ {1} ^ {\star}) - \mathcal {P} _ {1} (\tilde {\mathbf {a}} _ {1} | x _ {1} ^ {(1)}, \tilde {A} _ {1})\right) \overline {{Q}} _ {1} ^ {\star} (x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) \\ + \mathcal {P} _ {1} (\tilde {\mathbf {a}} _ {1} | x _ {1} ^ {(1)}, \tilde {A} _ {1}) \mathbb {P} _ {1} (x _ {2} ^ {(1)} | x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) (V _ {2} ^ {\star} - V _ {2} ^ {\pi}) (x _ {2} ^ {(1)}) \\ + \mathcal {P} _ {1} (\tilde {\mathbf {a}} _ {1} | x _ {1} ^ {(1)}, \tilde {A} _ {1}) \left(\mathbb {P} _ {1} (x _ {H + 2} ^ {(1)} | x _ {1} ^ {(1)}, \mathbf {a} _ {1} ^ {\star}) - \mathbb {P} _ {1} (s _ {H + 2} | x _ {1} ^ {(1)}, \bar {\mathbf {a}} _ {1})\right) \frac {(H - 1)}{H}, \tag {F.4} \\ \end{array}
$$

where the first inequality holds by (F.3). Note that, by construction, for any $h \in [H]$ , we have

$$
\overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(1)}, \mathbf {a} _ {h} ^ {\star}) = \frac {H - h + 1}{H},
$$

$$
\mathcal {P} _ {h} \left(\tilde {\mathbf {a}} _ {h} \mid x _ {h} ^ {(1)}, \tilde {A} _ {h}\right) \geqslant \frac {1}{1 / H + 1} = \frac {H}{1 + H},
$$

$$
\mathbb {P} _ {h} (x _ {h + 1} ^ {(1)} | x _ {h} ^ {(1)}, \mathbf {a} _ {h} ^ {\star}) = 1 - \delta - (d ^ {\mathrm{lin}} - 5) \Delta ,
$$

$$
\mathbb {P} _ {h} (x _ {H + 2} ^ {(1)} | x _ {h} ^ {(1)}, \mathbf {a} _ {h} ^ {\star}) - \mathbb {P} _ {h} (s _ {H + 2} | x _ {h} ^ {(1)}, \bar {\mathbf {a}} _ {h}) = (d ^ {\mathrm{lin}} - 5) \Delta - \langle \boldsymbol {\mu} _ {h}, \bar {\mathbf {a}} _ {h} \rangle . \tag {F.5}
$$

Hence, by plugging (F.5) into (F.4) and applying recursion, we get

$$
\begin{array}{l} \left(V _ {1} ^ {\star} - V _ {1} ^ {\pi}\right) \left(x _ {1} ^ {(1)}\right) \\ \geqslant \sum_ {h = 1} ^ {H} \left(\mathcal {P} _ {h} (\mathbf {a} _ {h} ^ {\star} | x _ {h} ^ {(1)}, A _ {h} ^ {\star}) - \mathcal {P} _ {h} (\tilde {\mathbf {a}} _ {h} | x _ {h} ^ {(1)}, \tilde {A} _ {h})\right) \frac {H - h + 1}{H} \cdot \left(\frac {H}{H + 1}\right) ^ {h - 1} \cdot \left(\left(1 - \delta - (d ^ {\mathrm{lin}} - 5) \Delta\right)\right) ^ {h - 1} \\ + \sum_ {h = 1} ^ {H} \left((d ^ {\text { lin }} - 5) \Delta - \langle \boldsymbol {\mu} _ {h}, \bar {\mathbf {a}} _ {h} \rangle\right) \frac {H - h}{H} \cdot \left(\frac {H}{H + 1}\right) ^ {h} \cdot \left(\left(1 - \delta - (d ^ {\text { lin }} - 5) \Delta\right)\right) ^ {h - 1} \\ - \sum_ {h = 1} ^ {H} \frac {(H - h)}{H \sqrt {K}} \cdot \left(\frac {H}{H + 1}\right) ^ {h} \cdot \left(\left(1 - \delta - (d ^ {\text { lin }} - 5) \Delta\right)\right) ^ {h - 1}. \tag {F.6} \\ \end{array}
$$

Furthermore, since $H \geqslant 3$ and $3(d^{\mathrm{lin}} - 5)\Delta \leqslant \delta = 1 / H$ , we have

$$
\left(\frac {H}{H + 1}\right) ^ {h} \geqslant \left(\frac {H}{H + 1}\right) ^ {H + 1} \geqslant \frac {3}{1 0},
$$

$$
\left(\left(1 - \delta - (d ^ {\mathrm{lin}} - 5) \Delta\right)\right) ^ {h - 1} \geqslant \left(1 - \frac {4 \delta}{3}\right) ^ {H} \geqslant \frac {1}{3}. \tag {F.7}
$$

Therefore, by substituting (F.7) into (F.6), and considering the terms where $h \geqslant H / 2$ , we obtain

$$
\begin{array}{l} \left(V _ {1} ^ {\star} - V _ {1} ^ {\pi}\right) \left(x _ {1} ^ {(1)}\right) \geqslant \frac {1}{2 0} \sum_ {h = 1} ^ {H / 2} \left(\mathcal {P} _ {h} \left(\mathbf {a} _ {h} ^ {\star} \mid x _ {h} ^ {(1)}, A _ {h} ^ {\star}\right) - \mathcal {P} _ {h} \left(\tilde {\mathbf {a}} _ {h} \mid x _ {h} ^ {(1)}, \tilde {A} _ {h}\right)\right) + \frac {1}{2 0} \sum_ {h = 1} ^ {H / 2} \left((d ^ {\text { lin }} - 5) \Delta - \langle \boldsymbol {\mu} _ {h}, \bar {\mathbf {a}} _ {h} \rangle\right) \\ = \frac {1}{2 0} \sum_ {h = 1} ^ {H / 2} \underbrace {\left(\mathcal {P} _ {h} (\mathbf {a} _ {h} ^ {\star} | x _ {h} ^ {(1)} , A _ {h} ^ {\star}) - \mathcal {P} _ {h} (\tilde {\mathbf {a}} _ {h} | x _ {h} ^ {(1)} , \tilde {A} _ {h})\right)} _ {\text { MNL   bandit   regret }} + \frac {1}{2 0} \sum_ {h = 1} ^ {H / 2} \underbrace {\left(\max _ {\mathbf {a} \in \mathcal {I}} \langle \boldsymbol {\mu} _ {h} , \mathbf {a} \rangle - \langle \boldsymbol {\mu} _ {h} , \bar {\mathbf {a}} _ {h} \rangle\right)} _ {\text { linear   bandit   regret }}. \tag {F.8} \\ \end{array}
$$

On the right-hand side of (F.8), the first term corresponds to an MNL bandit problem. Recall that $|A_h^\star| = |\tilde{A}_h| = 2$ and, by construction, we have

$$
\mathcal {P} _ {h} (\mathbf {a} _ {h} ^ {\star} | x _ {h} ^ {(1)}, A _ {h} ^ {\star}) = \frac {\exp \left(\phi (x _ {h} ^ {(1)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \pmb {\theta} _ {h} ^ {\star}\right)}{1 / H + \exp \left(\phi (x _ {h} ^ {(1)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \pmb {\theta} _ {h} ^ {\star}\right)}, \quad \mathcal {P} _ {h} (\tilde {\mathbf {a}} _ {h} | x _ {h} ^ {(1)}, \tilde {A} _ {h}) = \frac {\exp \left(\phi (x _ {h} ^ {(1)} , \tilde {\mathbf {a}} _ {h}) ^ {\top} \pmb {\theta} _ {h} ^ {\star}\right)}{1 / H + \exp \left(\phi (x _ {h} ^ {(1)} , \tilde {\mathbf {a}} _ {h}) ^ {\top} \pmb {\theta} _ {h} ^ {\star}\right)}.
$$

Hence, this corresponds to an MNL bandit problem with a maximum assortment size of $M = 2$ , where the attraction parameter for the outside option (the constant in the denominator) is $1 / H$ .

Furthermore, the second term on the right-hand side of (F.8) represents a linear bandit problem. To sum up, the learning problem is not harder than minimizing the regret on $\Omega(H/2)$ MNL and linear bandit problems.

To bound each term of (F.8), we introduce the following propositions:

Proposition F.4 (Regret lower bound of MNL bandits, Lee & Oh 2024). Let $v_0$ denote the attraction parameter for the outside option. Let $d$ be divisible by 4. Suppose $K \geqslant C \cdot d^4 M / (M - 1)$ for some constant $C > 0$ . Then, in the uniform reward setting (where rewards are identical) with the reward for the outside option being zero, for any policy and the MNL preference model parameterized by $\theta$ , there exists a worst-case problem instance such that the worst-case expected regret is lower bounded as follows:

$$
\sup _ {\boldsymbol {\theta}} \mathbb {E} _ {\boldsymbol {\theta}} \left[ \text { MNLBanditRegret } (\boldsymbol {\theta}, K) \right] = \Omega \left(\frac {\sqrt {v _ {0} (M - 1)}}{v _ {0} + M - 1} \cdot d \sqrt {K}\right).
$$

Proposition F.5 (Lemma C.8 in Zhou et al. 2021a). Fix $0 < \delta < 1/3$ . Consider the linear bandit problem parameterized with a vector $\boldsymbol{\mu} \in \{-\Delta, \Delta\}^d$ and action set $\mathcal{I} = \{-1, 1\}^d$ . And the reward distribution for taking action $a \in \mathcal{I}$ is a Bernoulli distribution denoted as $B(\delta + \langle \boldsymbol{\mu}, \mathbf{a} \rangle)$ . Let $K$ be the number of time steps playing this bandit problem. Assume $K \geqslant d^2/(2\delta)$

and $\Delta = \sqrt{\delta / K} / (4\sqrt{2})$ . Then, for any bandit algorithm $\mathcal{B}$ , there exists $\pmb{\mu}$ such that the expected pseudo-regret of $\mathcal{B}$ over $K$ steps is lower bounded as follows:

$$
\mathbb {E} _ {\boldsymbol {\mu}} [ \text {LinearBanditRegret} (\boldsymbol {\mu}, K) ] \geqslant \frac {d \sqrt {K \delta}}{8 \sqrt {2}}.
$$

where the expectation is with respect to the reward distribution that depends on $\mu$ .

Now, by using Proposition F.4 and F.5, we can bound the regret as follows:

$$
\begin{array}{l} \sup _ {\boldsymbol {\theta}, \boldsymbol {\mu}, \mathbf {w}} \mathbb {E} _ {\boldsymbol {\theta}, \boldsymbol {\mu}, \mathbf {w}} \left[ \mathbf {R e g r e t} \left(\mathcal {M} _ {\boldsymbol {\theta}, \boldsymbol {\mu}, \mathbf {w}}, K\right) \right] \geqslant \frac {1}{2 0} \sum_ {h = 1} ^ {H / 2} \sup _ {\boldsymbol {\theta}} \mathbb {E} _ {\boldsymbol {\theta}} \left[ \sum_ {k = 1} ^ {K} \left(\mathcal {P} _ {h} (\mathbf {a} _ {h} ^ {\star} | x _ {h} ^ {(1)}, A _ {h} ^ {\star}) - \mathcal {P} _ {h} (\tilde {\mathbf {a}} _ {h} | x _ {h} ^ {(1)}, \tilde {A} _ {h})\right) \right] \\ + \frac {1}{2 0} \sum_ {h = 1} ^ {H / 2} \sup _ {\boldsymbol {\mu}} \mathbb {E} _ {\boldsymbol {\mu}} \left[ \sum_ {k = 1} ^ {K} \left(\max _ {\mathbf {a} \in \mathcal {I}} \langle \boldsymbol {\mu} _ {h}, \mathbf {a} \rangle - \langle \boldsymbol {\mu} _ {h}, \bar {\mathbf {a}} _ {h} \rangle\right) \right] \\ = \Omega \left(d \sqrt {H K} + d ^ {\mathrm{lin}} \sqrt {H K}\right), \\ \end{array}
$$

where, in the last equality, we use $v_{0} = 1 / H$ , $M = 2$ , and $\delta = 1 / H$ . This concludes the proof of Theorem 5.3.

![](images/8196f36e5e786e14f1d4ac3281c00346b84347a36903d35119cf9da1088fd764.jpg)

# F.4. Proof of Lemmas for Theorem 5.3

# F.4.1. PROOF OF LEMMA F.2

Proof of Lemma F.2. For any $i \in [H]$ , we can write the optimal $\overline{Q}$ -value in state $x_h^{(i)}$ at horizon $h \in [H]$ as follows:

$$
\overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a}) = \left\{ \begin{array}{l l} \frac {\gamma^ {i - 1}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a}) V _ {h + 1} ^ {\star} (x _ {h + 1} ^ {(i)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a}) \frac {(H - h) \gamma^ {i - 1}}{H}, & \mathbf {a} = \mathbf {a} _ {h} ^ {\star}; \\ \frac {\gamma^ {i}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i + 1)} | x _ {h} ^ {(i)}, \mathbf {a}) V _ {h + 1} ^ {\star} (x _ {h + 1} ^ {(i + 1)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a}) \frac {(H - h) \gamma^ {i + 1}}{H}, & \mathbf {a} = \mathbf {a} _ {h} ^ {\star}, \mathbf {a} _ {0}; \\ 0, & \mathbf {a} = \mathbf {a} _ {0}. \end{array} \right.
$$

First, we show that for any $(i,h)\in [H]\times [H]$ , we have

$$
\overline {{{Q}}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a}) \geqslant \sum_ {\mathbf {a} ^ {\prime} \in A _ {h} ^ {\star}} \mathcal {P} _ {h} (\mathbf {a} ^ {\prime} | x _ {h} ^ {(i)}, A _ {h} ^ {\star}) \overline {{{Q}}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}), \quad \forall \mathbf {a} \in A _ {h} ^ {\star} \backslash \{\mathbf {a} _ {0} \}. \tag {F.9}
$$

We prove this by contradiction. Suppose there exists $\mathbf{a} \in A_h^\star$ such that $\overline{Q}_h^\star(x_h^{(i)}, \mathbf{a}) < \sum_{\mathbf{a}' \in A_h^\star} \mathcal{P}_h(\mathbf{a}' | x_h^{(i)}, A_h^\star) \overline{Q}_h^\star(x_h^{(i)}, \mathbf{a}')$ . In that case, removing the item $\mathbf{a}$ from the assortment $A_h^\star$ results in a higher expected value of $\overline{Q}_h^\star$ . This contradicts the optimality of $A_h^\star$ . Therefore, (F.9) must hold.

By the definition of $\overline{Q}_h^\star (x_h^{(i)},\mathbf{a})$ , for any $\mathbf{a}\in \mathcal{I}\backslash \{\mathbf{a}_h^\star \}$ , we have

$$
\begin{array}{l} \overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a}) \leqslant \frac {\gamma^ {i - 1}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i + 1)} | x _ {h} ^ {(i)}, \mathbf {a}) V _ {h + 1} ^ {\star} (x _ {h + 1} ^ {(i)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a}) \frac {(H - h) \gamma^ {i - 1}}{H} \\ \leqslant \frac {\gamma^ {i - 1}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) V _ {h + 1} ^ {\star} (x _ {h + 1} ^ {(i)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) \frac {(H - h) \gamma^ {i - 1}}{H} \\ = \overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}), \\ \end{array}
$$

where the first inequality holds since $V_{h+1}^{\star}(x_{h+1}^{(i+1)}) \leqslant V_{h+1}^{\star}(x_{h+1}^{(i)})$ , and the second inequality holds due to the fact that $V_{h+1}^{\star}(x_{h+1}^{(i)}) \leqslant \frac{(H - h)\gamma^{i-1}}{H}$ and $\mathbb{P}_h(x_{H+2}^{(i+2)}|x_h^{(i)},\mathbf{a}) \leqslant \mathbb{P}_h(x_{H+2}^{(i)}|x_h^{(i)},\mathbf{a}_h^{\star})$ .

Since $\overline{Q}_h^\star (x_h^{(i)},\mathbf{a}_h^\star)$ has the highest value among all items, the optimal assortment $A_{h}^{\star}$ should include $\mathbf{a}_h^\star$ . Thus, we have $\mathbf{a}_h^\star, \mathbf{a}_0\in A_h^\star$ . In other words, when $A_{h}^{\star} = \{\mathbf{a}_{h}^{\star},\mathbf{a}_{0}\}$ , the condition in (F.9) is satisfied. Thus, we begin with $A_{h}^{\star} = \{\mathbf{a}_{h}^{\star},\mathbf{a}_{0}\}$ and check if there exist an item $\mathbf{a}\neq \mathbf{a}_h^\star, \mathbf{a}_0$ that can increase the expected value of $\overline{Q}_h^\star$ . To this end, for $A_{h}^{\star} = \{\mathbf{a}_{h}^{\star},\mathbf{a}_{0}\}$ , we get

$$
\sum_ {\mathbf {a} ^ {\prime} \in A _ {h} ^ {\star}} \mathcal {P} _ {h} (\mathbf {a} ^ {\prime} | x _ {h} ^ {(i)}, A _ {h} ^ {\star}) \overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) = \mathcal {P} _ {h} (\mathbf {a} _ {h} ^ {\star} | x _ {h} ^ {(i)}, A _ {h} ^ {\star}) \overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) \geqslant \gamma \overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}), \tag {F.10}
$$

where the equality holds since $\overline{Q}_{h}^{\star}(x_{h}^{(i)}, a_{0}) = 0$ , and the inequality holds by the definition of $\gamma$ :

$$
\begin{array}{l} \gamma = \frac {H}{1 + H} = \frac {1}{1 / H + 1} \leqslant \min _ {h \in [ H ]} \min _ {s \in \mathcal {S}} \min _ {A \in \mathcal {A}} \min _ {\mathbf {a} \in A \backslash \{a _ {0} \}} \frac {\exp (\phi (s , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star})}{1 / H + \exp (\phi (s , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star})} \\ \leqslant \frac {\exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}{1 / H + \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)} = \mathcal {P} _ {h} (\mathbf {a} _ {h} ^ {\star} | x _ {h} ^ {(i)}, A _ {h} ^ {\star}). \tag {F.11} \\ \end{array}
$$

Here, we rely on the fact that the sigmoid function is monotonically increasing to establish the inequalities.

On the other hand, for any item $a \neq a_{h}^{\star}, a_{0}$ , we have

$$
\begin{array}{l} \overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a}) = \frac {\gamma^ {i}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i + 1)} | x _ {h} ^ {(i)}, \mathbf {a}) V _ {h + 1} ^ {\star} (x _ {h + 1} ^ {(i + 1)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a}) \frac {(H - h) \gamma^ {i + 1}}{H} \\ \leqslant \frac {\gamma^ {i}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i + 1)} | x _ {h} ^ {(i)}, \mathbf {a}) V _ {h + 1} ^ {\star} (x _ {h + 1} ^ {(i + 1)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a}) \frac {(H - h) \gamma^ {i}}{H} \\ \leqslant \frac {\gamma^ {i}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) V _ {h + 1} ^ {\star} (x _ {h + 1} ^ {(i + 1)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) \frac {(H - h) \gamma^ {i}}{H} \\ = \gamma \cdot \left(\frac {\gamma^ {i - 1}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) V _ {h + 1} ^ {\star} (x _ {h + 1} ^ {(i)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) \frac {(H - h) \gamma^ {i - 1}}{H}\right) \\ = \gamma \overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}), \tag {F.12} \\ \end{array}
$$

where the second inequality holds because $V_{h+1}^{\star}(x_{h+1}^{(i+1)}) \leqslant \frac{(H-h)\gamma^{i}}{H}, \mathbb{P}_{h}(x_{H+2}^{(i+2)}|x_{h}^{(i)}, \mathbf{a}) \leqslant \mathbb{P}_{h}(x_{H+2}^{(i)}|x_{h}^{(i)}, \mathbf{a}_{h}^{\star})$ , and the second equality follows from the fact that $\gamma V_{h+1}^{\star}(x_{h+1}^{(i)}) = V_{h+1}^{\star}(x_{h+1}^{(i+1)})$ by construction.

Combining (F.10) and (F.12), when $A_{h}^{\star} = \{a_{h}^{\star}, a_{0}\}$ , for any item $a \neq a_{h}^{\star}, a_{0}$ , we get

$$
\sum_ {\mathbf {a} ^ {\prime} \in A _ {h} ^ {\star}} \mathcal {P} _ {h} (\mathbf {a} ^ {\prime} | x _ {h} ^ {(i)}, A _ {h} ^ {\star}) \overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \geqslant \overline {{Q}} _ {h} ^ {\star} (x _ {h} ^ {(i)}, \mathbf {a}).
$$

Since $\overline{Q}_{h}^{\star}(x_{h}^{(i)},\mathbf{a})$ for $a\neq a_{h}^{\star},a_{0}$ is not greater than the expected value of $\overline{Q}_{h}^{\star}$ for $A_{h}^{\star}$ , adding any item $a\neq a_{h}^{\star},a_{0}$ to $A_{h}^{\star}$ does not increase the expected value of $\overline{Q}_{h}^{\star}$ . This confirms the optimality of $A_{h}^{\star}$ .

# F.4.2. PROOF OF LEMMA F.3

Proof of Lemma F.3. For any $i \in [H]$ , we can write the $\overline{Q}$ -value for the policy $\pi$ in state $x_{h}^{(i)}$ at horizon $h \in [H]$ as follows:

$$
\overline {{Q}} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a}) = \left\{ \begin{array}{l l} \frac {\gamma^ {i - 1}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a}) V _ {h + 1} ^ {\pi} (x _ {h + 1} ^ {(i)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a}) \frac {(H - h) \gamma^ {i - 1}}{H}, & \mathbf {a} = \mathbf {a} _ {h} ^ {\star}; \\ \frac {\gamma^ {i}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i + 1)} | x _ {h} ^ {(i)}, \mathbf {a}) V _ {h + 1} ^ {\pi} (x _ {h + 1} ^ {(i + 1)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a}) \frac {(H - h) \gamma^ {i + 1}}{H}, & \mathbf {a} = \mathbf {a} _ {h} ^ {\star}, \mathbf {a} _ {0}; \\ 0, & \mathbf {a} = \mathbf {a} _ {0}. \end{array} \right.
$$

We provide a proof by considering the following cases:

Case (i) $a_{h}^{\star} \in A$ .

Recall that, by (F.11), we have

$$
\gamma = \frac {H}{1 + H} \leqslant \frac {\exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}{1 / H + \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}. \tag {F.13}
$$

By multiplying $\widetilde{Q}_h^\pi (x_h^{(i)},\mathbf{a}_h^\star ,\mathbf{a}^\prime)$ on both sides of (F.13), we get

$$
\begin{array}{l} \gamma \cdot \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}, \mathbf {a} ^ {\prime}) \leqslant \frac {\exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star} , \mathbf {a} ^ {\prime})}{1 / H + \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)} \\ \Leftrightarrow \left(\sum_ {\mathbf {a} \in A \backslash \{\mathbf {a} _ {h} ^ {\star}, \mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)\right) \gamma \cdot \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}, \mathbf {a} ^ {\prime}) \left(1 / H + \exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)\right) \\ \leqslant \left(\sum_ {\mathbf {a} \in A \backslash \{\mathbf {a} _ {h} ^ {\star}, \mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)\right) \exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}, \mathbf {a} ^ {\prime}) \\ \Leftrightarrow \left(\exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) + \sum_ {\mathbf {a} \in A \setminus \{\mathbf {a} _ {h} ^ {\star}, \mathbf {a} _ {0} \}} \gamma \cdot \exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)\right) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}, \mathbf {a} ^ {\prime}) \\ \cdot \left(1 / H + \exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)\right) \\ \leqslant \left(1 / H + \exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) + \sum_ {\mathbf {a} \in A \setminus \{\mathbf {a} _ {h} ^ {\star}, \mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)\right) \exp \left(\phi (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \\ \cdot \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}, \mathbf {a} ^ {\prime}) \\ \Leftrightarrow \frac {\exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star} , \mathbf {a} ^ {\prime}) + \sum_ {\mathbf {a} \in A \setminus \{\mathbf {a} _ {h} ^ {\star} , \mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \gamma \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star} , \mathbf {a} ^ {\prime})}{1 / H + \sum_ {\mathbf {a} \in A \setminus \{\mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)} \\ \leqslant \frac {\exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star} , \mathbf {a} ^ {\prime})}{1 / H + \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}. \tag {F.14} \\ \end{array}
$$

On the other hand, by the definition of $\widetilde{Q}_h^\pi (x_h^{(i)},\mathbf{a}_h^\star ,\mathbf{a}')$ , for any $\mathbf{a}'\neq \mathbf{a}_h^\star ,\mathbf{a}_0$ , we have

$$
\begin{array}{l} \gamma \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}, \mathbf {a} ^ {\prime}) = \gamma \cdot \left(\frac {\gamma^ {i - 1}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) V _ {h + 1} ^ {\pi} (x _ {h + 1} ^ {(i)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \frac {(H - h) \gamma^ {i - 1}}{H}\right) \\ = \frac {\gamma^ {i}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) V _ {h + 1} ^ {\pi} (x _ {h + 1} ^ {(i + 1)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \frac {(H - h) \gamma^ {i}}{H} \\ \geqslant \frac {\gamma^ {i}}{H} + \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) V _ {h + 1} ^ {\pi} (x _ {h + 1} ^ {(i + 1)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \frac {(H - h) \gamma^ {i + 1}}{H} \\ = \overline {{{Q}}} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}), \tag {F.15} \\ \end{array}
$$

where the second equality holds since $\gamma V_{h + 1}^{\pi}(x_{h + 1}^{(i)}) = V_{h + 1}^{\pi}(x_{h + 1}^{(i + 1)})$ , and the inequality holds because, for $K\geqslant 4(d^{\mathrm{lin}} - 5)^2 H(H + 1)^2$ , the following inequality holds:

$$
\begin{array}{l} \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) V _ {h + 1} ^ {\pi} (x _ {h + 1} ^ {(i + 1)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \frac {(H - h) \gamma^ {i + 1}}{H} \\ \leqslant \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}) V _ {h + 1} ^ {\pi} (x _ {h + 1} ^ {(i + 1)}) + \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \frac {(H - h) \gamma^ {i}}{H} \\ \Leftrightarrow \left(\mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) - \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star})\right) V _ {h + 1} ^ {\pi} \leqslant \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \frac {(H - h)}{H} \left(\gamma^ {i} - \gamma^ {i + 1}\right). \\ \end{array}
$$

Specifically, if the upper bound of the left-hand side is less than or equal to the lower bound of the right-hand side, the inequality holds. To demonstrate this, we have:

$$
\left(\mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) - \mathbb {P} _ {h} (x _ {h + 1} ^ {(i)} | x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star})\right) V _ {h + 1} ^ {\pi} \leqslant 2 (d ^ {\mathrm{lin}} - 5) \Delta \cdot \frac {(H - h)}{H}, \tag {F.16}
$$

and, since $\gamma^i = \left(\frac{H}{H + 1}\right)^i\geqslant \left(\frac{H}{H + 1}\right)^{H + 1}\geqslant \frac{3}{10}$ , we get

$$
\begin{array}{l} \mathbb {P} _ {h} (x _ {H + 2} ^ {(i + 2)} | x _ {h} ^ {(i)}, \mathbf {a} ^ {\prime}) \frac {(H - h)}{H} (\gamma^ {i} - \gamma^ {i + 1}) \geqslant (\delta - (d ^ {\mathrm{lin}} - 5) \Delta) \frac {(H - h)}{H} \gamma^ {i} (1 - \gamma) \\ \geqslant \left(\frac {1}{H} - (d ^ {\text { lin }} - 5) \Delta\right) \frac {(H - h)}{H} \cdot \frac {3}{1 0} \cdot \frac {1}{H + 1}. \tag {F.17} \\ \end{array}
$$

Combining (F.16) and (F.17), and rearranging the terms, we get

$$
(d ^ {\mathrm{lin}} - 5) \Delta \cdot \left(2 + \frac {3}{1 0 (H + 1)}\right) \leqslant \frac {3}{1 0 H (H + 1)},
$$

which holds when $K \geqslant 4(d^{\mathrm{lin}} - 5)^{2}H(H + 1)^{2}$ . This explains how the inequality in (F.15) is satisfied.

Let $\bar{\mathbf{a}}_h^{(i)}\in \mathrm{argmax}_{\mathbf{a}\in A_h\setminus \{\mathbf{a}_0\}}\overline{Q}_h^\pi (x_h^{(i)},\mathbf{a}) = \mathbf{a}_h^\star$ . Note that $\bar{\mathbf{a}}_h^{(i)}$ is unique due to the way the action space and transition probabilities are constructed. Then, by combining (F.14) and (F.15), and using the fact that $\overline{Q}_h^\pi (x_h^{(i)},\mathbf{a}_0) = 0$ , we obtain that

$$
\begin{array}{l} \sum_ {\mathbf {a} \in A} \mathcal {P} _ {h} (\mathbf {a} | x _ {h} ^ {(i)}, A) \overline {{Q}} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a}) \leqslant \sum_ {\mathbf {a} \in A} \mathcal {P} _ {h} (\mathbf {a} | x _ {h} ^ {(i)}, A) \overline {{Q}} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \bar {\mathbf {a}} _ {h} ^ {(i)}) \\ \leqslant \frac {\exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star} , \bar {\mathbf {a}} _ {h} ^ {(i)})}{1 / H + \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)} \\ = \frac {\max _ {\mathbf {a} \in A \backslash \{\mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star} , \bar {\mathbf {a}} _ {h} ^ {(i)})}{1 / H + \max _ {\mathbf {a} \in A \backslash \{\mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}, \\ \end{array}
$$

where the first inequality holds since $\bar{\mathbf{a}}_h^{(i)}$ is the action that maximizes the $\overline{Q}$ -value. The second inequality follows from (F.14) and (F.15), and from the fact that $\overline{Q}_h^\pi (x_h^{(i)},\bar{\mathbf{a}}_h^{(i)}) = \overline{Q}_h^\pi (x_h^{(i)},\mathbf{a}_h^\star) = \widetilde{Q}_h^\pi (x_h^{(i)},\mathbf{a}_h^\star ,\bar{\mathbf{a}}_h^{(i)})$ . Finally, the last equality holds by the definition of $\mathbf{a}_h^\star$ .

Case (ii) $\mathbf{a}_h^\star \notin A$ .

Again, by (F.11), for any $A \in \mathcal{A}$ , we have

$$
\begin{array}{l} \gamma \leqslant \min _ {\mathbf {a} \in A \backslash \{a _ {0} \}} \frac {\exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}{1 / H + \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)} \\ \leqslant \frac {\max _ {\mathbf {a} \in A \backslash \{a _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}{1 / H + \max _ {\mathbf {a} \in A \backslash \{a _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)} \\ \leqslant \frac {\max _ {\mathbf {a} \in A \setminus \{a _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}{1 / H + \max _ {\mathbf {a} \in A \setminus \{a _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)} \cdot \frac {1 / H + \sum_ {\mathbf {a} \in A \setminus \{\mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}{\sum_ {\mathbf {a} \in A \setminus \{\mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}, \\ \end{array}
$$

where the second inequality holds since the sigmoid function is a monotonically increasing function.

We denote $\bar{\mathbf{a}}_h^{(i)}\in \operatorname{argmax}_{\mathbf{a}\in A_h\setminus \{\mathbf{a}_0\}}\overline{Q}_h^\pi (x_h^{(i)},\mathbf{a})$ . Then, multiplying $\widetilde{Q}_h^\pi (x_h^{(i)},\mathbf{a}_h^\star ,\bar{\mathbf{a}}_h^{(i)})$ (note that $\bar{\mathbf{a}}_h^{(i)}\neq \mathbf{a}_h^\star ,\mathbf{a}_0$ ) on both sides and rearranging terms, we get

$$
\begin{array}{l} \frac {\sum_ {\mathbf {a} \in A \backslash \{\mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \gamma \cdot \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star} , \bar {\mathbf {a}} _ {h} ^ {(i)})}{1 / H + \sum_ {\mathbf {a} \in A \backslash \{\mathbf {a} _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)} \\ \leqslant \frac {\max _ {\mathbf {a} \in A \backslash \{a _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star} , \bar {\mathbf {a}} _ {h} ^ {(i)})}{1 / H + \max _ {\mathbf {a} \in A \backslash \{a _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}. \\ \end{array}
$$

Recall that for any $a^{\prime} \neq a_{h}^{\star}, a_{0}$ , we have $\gamma \cdot \widetilde{Q}_{h}^{\pi}(x_{h}^{(i)}, a_{h}^{\star}, a^{\prime}) \geqslant \overline{Q}_{h}^{\pi}(x_{h}^{(i)}, a^{\prime})$ by (F.15). Thus, we get

$$
\begin{array}{l} \sum_ {\mathbf {a} \in A} \mathcal {P} _ {h} (\mathbf {a} | x _ {h} ^ {(i)}, A) \overline {{Q}} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a}) \leqslant \sum_ {\mathbf {a} \in A} \mathcal {P} _ {h} (\mathbf {a} | x _ {h} ^ {(i)}, A) \overline {{Q}} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \bar {\mathbf {a}} _ {h} ^ {(i)}) \\ \leqslant \sum_ {\mathbf {a} \in A} \mathcal {P} _ {h} (\mathbf {a} | x _ {h} ^ {(i)}, A) \gamma \cdot \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)}, \mathbf {a} _ {h} ^ {\star}, \bar {\mathbf {a}} _ {h} ^ {(i)}) \\ \leqslant \frac {\max _ {\mathbf {a} \in A \backslash \{a _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right) \widetilde {Q} _ {h} ^ {\pi} (x _ {h} ^ {(i)} , \mathbf {a} _ {h} ^ {\star} , \bar {\mathbf {a}} _ {h} ^ {(i)})}{1 / H + \max _ {\mathbf {a} \in A \backslash \{a _ {0} \}} \exp \left(\phi (x _ {h} ^ {(i)} , \mathbf {a}) ^ {\top} \boldsymbol {\theta} _ {h} ^ {\star}\right)}. \\ \end{array}
$$

This concludes the proof of Lemma F.3.

![](images/fdbbc362e07d0fccbcf2987e4bc2ca9153b7648fbc0b33f53ee0fcde851513e8.jpg)

# G. Numerical Experiments

![](images/dd8d25dffcc6acbefce75c1ef75913fe13530e1336ef1043ef4b64ff3df998b9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    s1 -->|1 - i/N| s2
    s2 -->|1 - i/N| s3
    s3 -->|1 - i/N| s4
    s4 -->|1 - i/N| s5
    s5 -->|1 - i/N| s1
    s1 -.->|i/N| s2
    s2 -.->|i/N| s3
    s3 -.->|i/N| s4
    s4 -.->|i/N| s5
    s5 -.->|i/N| s1
```
</details>

Figure G.1: The “online shopping with budget” environment with $|S| = 5$ . Each state represents the user’s budget level of 1, 2, 3, 4, or 5. The solid line indicates the transition when the user purchases an actual item $a_{i}$ (with a reward of $(i/100N + j/|S|)/H$ ), and the dashed line shows the transition when the user does not purchase any item (with a reward of 0). The initial state is $s_{3}$ .

In this section, we empirically evaluate the performance of our algorithm, MNL-VQL, in linear MDPs. We consider an online shopping with budget (refer Figure G.1) environment under linear MDPs and an MNL user preference model. We denote the set of states as $S = \{s_{1}, \ldots, s_{|S|}\}$ and the set of items as $I = \{a_{1}, \ldots, a_{N}, a_{0}\}$ ( $a_{0}$ denotes the outside option). Each state $s_{j} \in S$ corresponds to a user's budget level, where a larger index j indicates a higher budget (e.g., $s_{|S|}$ represents the state with the largest budget). The initial state is set to the medium budget state $s_{[|S|/2]}$ . Furthermore, we let the transition probabilities $P_{h}$ , rewards $r_{h}$ , and preference model $P_{h}$ be the same for all $h \in [H]$ , and thus we omit the subscript h.

At state $s_j$ , the agent offers an assortment $A \in \mathcal{A}$ with a maximum size of $M$ . The user then either purchases an item $a_i \in A$ or opts not to buy anything, represented by the outside option $a_0 \in A$ . Then, the reward is defined as follows:

- If the user purchases an item $a_i \in A$ , the reward is: $r(s_j, a_i) = \left( \frac{i}{100N} + \frac{j}{|S|} \right) / H$ .   
- If the user does not buy anything $(a_0)$ , the reward is: $r(s_j, a_0) = 0$ .

The reward can be regarded as the user's rating of the purchased item. It is reasonable to assume that, at higher budget states, users tend to be more generous in their ratings, leading to higher ratings (rewards). And the transition probability is defined as follows:

\- If the user purchases an item $a_{i} \in A$ , the transition probability is:

$$
\mathbb {P} (s _ {\min (j + 1, | \mathcal {S} |)} | s _ {j}, a _ {i}) = 1 - \frac {i}{N}, \quad \text { and } \mathbb {P} (s _ {\max (j - 1, 0)} | s _ {j}, a _ {i}) = \frac {i}{N}.
$$

\- If the user does not buy anything $(a_0)$ , the transition probability is:

$$
\mathbb {P} \big (s _ {\min (j + 1, | \mathcal {S} |)} | s _ {j}, a _ {0} \big) = 1
$$

If the user does purchase an item, the budget level decreases with a certain probability that depends on the chosen item. Conversely, if the user does not purchase any item $(a_{0})$ , the budget level increases deterministically.

<table><tr><td></td><td>Myoptic</td><td>LSVI-UCB</td><td>MNL-VQL (ours)</td></tr><tr><td>N=10,|A|=637</td><td>0.089 s</td><td>0.136 s</td><td>0.463 s</td></tr><tr><td>N=20,|A|=21,699</td><td>0.097 s</td><td>4.861 s</td><td>0.526 s</td></tr><tr><td>N=40,|A|=760,098</td><td>0.113 s</td><td>453.641 s</td><td>0.620 s</td></tr></table>

Table G.1: Average runtime (seconds) per episode for M = 6.

We construct the feature map $\psi(s, a)$ (for linear MDPs) using SVD. Specifically, the transition kernel $\mathbb{P}(\cdot|\cdot,\cdot)\in\mathbb{R}^{|\mathcal{S}||\mathcal{I}|\times|\mathcal{S}|}$ has at most $|\mathcal{S}|$ singular values, and the reward vector $r(\cdot,\cdot)\in\mathbb{R}^{|\mathcal{S}||\mathcal{I}|}$ has one singular value. Consequently, the feature map $\psi(s,a)\in\mathbb{R}^{d_{lin}}$ lies in a space of dimension $|\mathcal{S}|+1$ , i.e., $d_{lin}=|\mathcal{S}|+1$ .

For MNL preference model, the true parameter $\theta^{\star} \in R^{d}$ , and the feature $\phi(s, a) \in \mathbb{R}^{d}$ (for MNL preference model) are randomly sampled from a d-dimensional uniform distribution in each instance.

We set $K = 30000$ , $H = 5$ , $M = 6$ , $|\mathcal{S}| = 5$ , $d = 5$ (feature dimension for MNL preference model), $d^{lin} = 6$ (feature dimension for linear MDP), $N \in \{10, 20, 40\}$ (the number of items), and $|\mathcal{A}| = \sum_{m' = 1}^{M - 1}\binom{N}{m} \in \{637, 21699, 760098\}$ (the number of assortments). Moreover, for simplicity, we set $\bar{\sigma}_h^k = 1$ in our algorithm. As a result, we use unweighted regression to estimate the $\overline{Q}$ -values.

We compare our algorithm with two baselines: Myopic and LSVI-UCB (Jin et al., 2020). Myopic is a variant of OFU-MNL+ (Lee & Oh, 2024) adapted for unknown rewards. It is a myopic algorithm that selects assortments based only on immediate rewards, ignoring state transitions. LSVI-UCB (Jin et al., 2020) treats each assortment as a single, atomic (holistic) action, requiring enumeration of all possible assortments. To demonstrate the effectiveness of our approach, we also include the performance of the optimal policy (Optimal) to highlight that our algorithm is converging toward optimality. We run the algorithms on 10 independent instances and report the episodic return across all episodes.

Figure 1 demonstrates that our algorithm significantly outperforms other baseline algorithms. And Table G.1 shows that our algorithm maintains robust runtime performance even as the total number of assortments $|A|$ increases. Although the runtime of Myopic is approximately 5.3 times faster than ours, its performance is substantially worse, converging to a suboptimal solution. This underscores a key limitation of the myopic strategy—it can completely fail in certain environments, highlighting the importance of accounting for long-term outcomes. Additionally, the runtime of LSVI-UCB increases exponentially as N grows, because it requires enumerating all possible assortments. Due to the extremely slow runtime of LSVI-UCB, we did not include its performance results for N = 20 and N = 40. Instead, for these cases, we used dotted lines to represent the average episodic return observed for N = 10. Even for the smaller case of N = 10, LSVI-UCB demonstrated the worst performance. Based on this observation, we suspect that its performance is unlikely to improve as N increases.