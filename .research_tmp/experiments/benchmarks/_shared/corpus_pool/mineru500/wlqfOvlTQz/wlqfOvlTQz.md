# Reinforcement Learning with Lookahead Information

Nadav Merlis

FairPlay Joint Team, CREST, ENSAE Paris

nadav.merlis@ensae.fr

# Abstract

We study reinforcement learning (RL) problems in which agents observe the reward or transition realizations at their current state before deciding which action to take. Such observations are available in many applications, including transactions, navigation and more. When the environment is known, previous work shows that this lookahead information can drastically increase the collected reward. However, outside of specific applications, existing approaches for interacting with unknown environments are not well-adapted to these observations. In this work, we close this gap and design provably-efficient learning algorithms able to incorporate lookahead information. To achieve this, we perform planning using the empirical distribution of the reward and transition observations, in contrast to vanilla approaches that only rely on estimated expectations. We prove that our algorithms achieve tight regret versus a baseline that also has access to lookahead information – linearly increasing the amount of collected reward compared to agents that cannot handle lookahead information.

# 1 Introduction

In reinforcement learning (RL), agents sequentially interact with a changing environment, aiming to collect as much reward as possible. While performing actions that yield immediate rewards is enticing, agents must also bear in mind that actions influence the state of the environment, affecting the potential reward that could be collected in future steps. When the environment is unknown, agents also need to balance reward maximization based on previous data and exploration – gathering of data that might improve future reward collection.

In the standard interaction model, at each timestep, agents first choose an action and only then observe its outcome on the rewards and state dynamics. As such, agents can only maximize the expected rewards, collected through the expected dynamics. Yet, in many applications, some information on the immediate outcome of actions is known before actions are performed. For example, when agents interact through transactions, prices and traded goods are usually agreed upon before performing any exchange ('reward information'). Alternatively, in navigation problems, nearby traffic information is known to the agent before choosing which path to go through ('transition information').

In a recent work, Merlis et al. [2024] shows that even for agents with full statistical knowledge of the environment, such ‘lookahead’ information can drastically increase the reward collected by agents – by a multiplicative factor of up to AH when immediate rewards are revealed in advance and $A^{H/2}$ when observing the immediate future transitions. $^{1}$ Intuitively, agents do not only gain from instantaneously using this information – they can also adapt their planning to account for lookahead information being revealed in subsequent states, significantly increasing their future values. However, the work of Merlis et al. [2024] only tackles planning settings in which the model is known and does not provide algorithms or guarantees when interacting with unknown environments.

In this work, we aim to design provably-efficient agents that learn how to interact when given immediate ('one-step lookahead') reward or transition information before choosing an action, under the episodic tabular Markov Decision Process model. While such information can always be embedded into the state of the environment, the state space becomes exponential at best, and continuous at worst, rendering most theoretically-guaranteed approaches both computationally and statistically intractable. To alleviate this, we start by deriving dynamic programming ('Bellman') equations in the original state space that characterize the optimal lookahead policies. Inspired by these update rules, we present two variants to the MVP algorithm [Zhang et al., 2021b] that allow incorporating either reward or transition lookahead. In particular, we suggest a planning procedure that uses the empirical distribution of the reward/transition observations (instead of the estimated expectations), which might also be applied to other complex settings. We prove that these algorithms achieve tight regret bounds of $\tilde{\mathcal{O}}\left(\sqrt{H^3 SAK}\right)$ and $\tilde{\mathcal{O}}\left(\sqrt{H^2 SK} (\sqrt{H} +\sqrt{A})\right)$ after $K$ episodes (for reward and transition lookahead, respectively), compared to a stronger baseline that also has access to lookahead information. As such, they can collect significantly more rewards than vanilla RL algorithms.

Outline. We formally define RL problems with reward/transition lookahead in Section 2 and further discuss the differences between our setting and standard RL problems in Section 3. Then, we present our results in two complementary sections: Section 4 analyzes reward lookahead while Section 5 analyzes transition lookahead. We end with conclusions and future directions in Section 6.

Related Work. Problems with varying lookahead information have been extensively studied in control, with model predictive control [MPC, Camacho et al., 2007] as the most notable example. Conceptually, when interacting with an environment that might be too complex or hard to model, it is oftentimes convenient to use a simpler model that allows accurately predicting its behavior just in the near future. MPC uses such models to repeatedly update its policy using short-term planning. In some cases, the utilized future predictions consist of additive perturbations to the dynamics [Yu et al., 2020], while other cases involve more general future predictions on the model behavior [Li et al., 2019, Zhang et al., 2021a, Lin et al., 2021, 2022]. To the best of our knowledge, these studies focus on comparing the performance of the controller to one with full future information (and thus, linear regret is inevitable), sometimes also considering prediction errors. They do not, however, attempt to learn the predictions. In contrast, we estimate the reward/transition distributions and leverage them to better plan, thus increasing the value gained by the agent. In addition, these works focus on continuous (mostly linear) control problems, whereas we study tabular settings; results from any one of these settings cannot be directly applied to the other.

In RL, lookahead is mostly used as a planning tool; namely, agents test the possible outcomes after performing multiple steps to decide which actions to take or to better estimate the value [Tamar et al., 2017, Efroni et al., 2019a, 2020, Moerland et al., 2020, Rosenberg et al., 2023, El Shar and Jiang, 2020, Biedenkapp et al., 2021, Huang et al., 2019]. Specifically, the future value at the end of the lookahead is often estimated using rollouts, and a longer lookahead is more robust to suboptimality of the rollout policy [Bertsekas, 2023]. However, when agents actually interact with the environment, no additional lookahead information is observed. One notable exception is [Merlis et al., 2024], which analyzes the potential value increase due to multi-step reward lookahead information (and briefly mentions transition lookahead). However, they only tackle planning settings, where the model is known, and do not study learning. In this work, we continue a long line of literature on regret analysis for tabular RL [Jaksch et al., 2010, Jin et al., 2018, Dann et al., 2019, Zanette and Brunskill, 2019, Efroni et al., 2019b, 2021, Simchowitz and Jamieson, 2019, Zhang et al., 2021b, 2023]. Yet, we are not aware of any existing results on regret minimization with reward or transition lookahead information.

Finally, various applications that involve one-step lookahead information have been previously studied. The most notable ones are prophet problems [Correa et al., 2019], where one-step reward lookahead is obtained, and the Canadian traveler problem with resampling [Nikolova and Karger, 2008], which can be formulated through one-step transition lookahead. We discuss the relation to these problems and the relevant existing results when analyzing each type of feedback, and also discuss the relation between transition lookahead and stochastic action sets [Boutilier et al., 2018].

# 2 Setting and Notations

We study episodic tabular Markov Decision Processes (MDPs), defined by the tuple $\mathcal{M} = (\mathcal{S},\mathcal{A},H,P,\mathcal{R})$ , where $\mathcal{S}$ is the state space (of size $S$ ), $\mathcal{A}$ is the action space (of size $A$ ) and $H$ is the

interaction horizon. At each timestep $h \in \{1, \ldots, H\} \triangleq [H]$ of an episode $k \in [K]$ , an agent, located in state $s_{h}^{k} \in S$ , chooses an action $a_{h}^{k} \in A$ and obtains a reward $R_{h}^{k} = R_{h}(s_{h}^{k}, a_{h}^{k}) \sim \mathcal{R}_{h}(s_{h}^{k}, a_{h}^{k})$ . We assume that the rewards are supported by [0, 1] and of expectations $r_{h}(s, a)$ . Afterward, the environment transitions to a state $s_{h+1}^{k} \sim P_{h}(\cdot | s_{h}^{k}, a_{h}^{k})$ and the interaction continues until the end of the episode. We use the notation $\boldsymbol{R} \sim \mathcal{R}_{h}(s)$ (or $s' \sim P_{h}(s)$ ) to denote reward (next-state) samples for all actions simultaneously at step h and state s and assume independence between different timesteps. $^{2}$ On the other hand, samples from different actions at a specific state/timestep are not necessarily independent.

Reward Lookahead. With one-step reward lookahead at timestep h and state s, agents first observe the rewards for all actions $\boldsymbol{R}_{h}(s) \triangleq \{R_{h}(s,a)\}_{a \in \mathcal{A}}$ and only then choose an action to perform. Formally, we define the set of reward lookahead policies as $\Pi^{R} = \left\{\pi : [H] \times S \times [0,1]^{A} \mapsto \Delta_{\mathcal{A}}\right\}$ , where $\Delta_{A}$ is the probability simplex, and denote $a_{h} = \pi_{h}(s_{h}, \boldsymbol{R}_{h})$ . The value of a reward lookahead agent is the cumulative rewards gathered by it starting at timestep h and state s, denoted by

$$
V _ {h} ^ {R, \pi} (s) = \mathbb {E} \left[ \sum_ {t = h} ^ {H} R _ {t} (s _ {t}, \pi_ {t} (s _ {t}, \boldsymbol {R} _ {t} (s _ {t})) | s _ {h} = s \right].
$$

We also define the optimal reward lookahead value to be $V_{h}^{R,*}(s)=\max_{\pi\in\Pi^{R}}V_{h}^{R,\pi}(s)$ . When interacting with an unknown environment for K episodes, agents sequentially choose reward lookahead policies $\pi^{k}\in\Pi^{R}$ based on all historical information and are measured by their regret,

$$
\mathrm{Reg} ^ {R} (K) = \sum_ {k = 1} ^ {K} \Bigl (V _ {1} ^ {R, *} (s _ {1} ^ {k}) - V _ {1} ^ {R, \pi^ {k}} (s _ {1} ^ {k}) \Bigr).
$$

We allow the initial state of each episode $s_{1}^{k}$ to be arbitrarily chosen.

Transition Lookahead. Denoting $s_{h+1}^{\prime}(s,a)$ , the future state when playing action a at step h and state s, one-step transition lookahead agents observe $\boldsymbol{s}_{h+1}^{\prime}(s)\triangleq\left\{s_{h+1}^{\prime}(s,a)\right\}_{a\in\mathcal{A}}$ before acting. The set of transition lookahead agents is denoted by $\Pi^{T}=\left\{\pi:[H]\times\mathcal{S}\times\mathcal{S}^{A}\mapsto\Delta_{\mathcal{A}}\right\}$ with values

$$
V _ {h} ^ {T, \pi} (s) = \mathbb {E} \left[ \sum_ {t = h} ^ {H} R _ {t} (s _ {t}, \pi_ {t} (s _ {t}, \boldsymbol {s} _ {t + 1} ^ {\prime} (s _ {t}))) | s _ {h} = s \right].
$$

The optimal value is $V_h^{T,*}(s) = \max_{\pi \in \Pi^T} V_h^{T,\pi}(s)$ , and we similarly define the regret versus optimal transition lookahead agents as $\mathrm{Reg}^T(K) = \sum_{k=1}^{K} \left( V_1^{T,*}(s_1^k) - V_1^{T,\pi^k}(s_1^k) \right)$ .

When the type of lookahead is clear from the context, we sometimes denote values by $V_h^\pi$ and $V_h^*$ .

Other Notations. For any $p \in \Delta_n$ and $V \in \mathbb{R}^n$ , we define $\mathrm{Var}_p(V) = \sum_{i=1}^{n} p_i V_i^2 - (\sum_{i=1}^{n} p_i V_i)^2$ . Also, given a transition kernel $P$ and a vector $V \in \mathbb{R}^S$ , we let $PV(s, a) = \sum_{s' \in S} P(s'|s, a)V(s')$ and similarly define it for value or transition kernel differences. We denote by $n_h^k(s, a)$ , the number of times the pair $(s, a)$ was visited at timestep $h$ up to episode $k$ (inclusive) and similarly denote $n_h^k(s) = \sum_{a \in \mathcal{A}} n_h^k(s, a)$ . We also let $\hat{r}_h^k(s, a) = \frac{1}{n_h^k(s, a)} \sum_{k'=1}^k \mathbb{1}\left\{ s_h^{k'} = s, a_h^{k'} = a \right\} R_h^{k'}$ and $\hat{P}_h(s'|s, a) = \frac{1}{n_h^k(s, a)} \sum_{k'=1}^k \mathbb{1}\left\{ s_h^{k'} = s, a_h^{k'} = a, s_{h+1}' = s' \right\}$ be the empirical expected rewards and transition kernel at $(s_h, a_h) = (s, a)$ using data up to episode $k$ and assume they are initialized to be zero. Finally, we denote by $\hat{\mathcal{R}}_h^k(s)$ , the empirical reward distribution across all actions, and use $\hat{P}_h^k(s)$ to denote the empirical joint next-state distribution for all actions. In particular, if $k_i$ is the $i^{th}$ episode where $s$ was visited at step $h$ , to sample $R \sim \hat{\mathcal{R}}_h^k(s)$ , we uniformly sample $i \sim U\left( [n_h^k(s)] \right)$ and return $R = \left\{ R_h^{k_i}(s, a) \right\}_{a \in \mathcal{A}}$ . A sample $s' \sim \hat{P}_h^k(s)$ similarly returns $s' = \left\{ s_{h+1}'^{k_i}(s, a) \right\}_{a \in \mathcal{A}}$ .

When we want to indicate the distribution used to calculate an expectation, we sometimes state it in a subscript, e.g., write $E_{\mathcal{R}_{h}(s)}[R(a)]$ to indicate that $R(a) \sim \mathcal{R}_{h}(s, a)$ or use $E_{M}$ to emphasize

that all distributions are according to an environment $\mathcal{M}$ . In this paper, $\mathcal{O}$ -notation only hides absolute constants while $\tilde{\mathcal{O}}$ hides factors of polylog $(S, A, H, K, \delta)$ . We also use the notation $a \vee b = \max\{a, b\}$ .

# 3 Comparing the Values of Lookahead Agents and Vanilla RL agents

In the classic RL formulation [e.g., Azar et al., 2017], agents only observe the reward and transition after performing an action and aim to maximize the 'no-lookahead' value, defined by

$$
V _ {h} ^ {\pi} (s) = \mathbb {E} \left[ \sum_ {t = h} ^ {H} r _ {t} \left(s _ {t}, \pi_ {t} \left(s _ {t}\right) \mid s _ {h} = s \right. \right],
$$

where $\pi \in \Pi^{\mathcal{M}} = \{\pi :[H]\times \mathcal{S}\mapsto \Delta_{\mathcal{A}}\}$ is a Markovian policy. The optimal value is $V_{h}^{no}(s) = \max_{\pi \in \Pi^{\mathcal{M}}}V_{h}^{\pi}(s)$ and the regret is classically defined as $\operatorname {Reg}(K) = \sum_{k = 1}^{K}\Big(V_1^{no}(s_1^k) - V_1^{\pi^k}(s_1^k)\Big)$ .

By definition, the set of lookahead policies also includes all Markovian policies (since agents are not obliged to use reward/transition information), so the optimal lookahead values are always larger than their no-lookahead counterpart. In other words, denoting the value gain due to lookahead information by $G^{R}(s) = V_{1}^{R,*}(s) - V_{1}^{no}(s)$ and $G^{T}(s) = V_{1}^{T,*}(s) - V_{1}^{no}(s)$ , it holds that $G^{R}(s), G^{T}(s) \geq 0$ . In terms of regret, for any fixed algorithm, we can also write

$$
\operatorname{Reg} (K) = \operatorname{Reg} ^ {R} (K) - \sum_ {k = 1} ^ {K} G ^ {R} (s _ {1} ^ {k}) = \operatorname{Reg} ^ {T} (K) - \sum_ {k = 1} ^ {K} G ^ {T} (s _ {1} ^ {k}).
$$

As the value gains are non-negative, it directly implies that any regret bound w.r.t. the lookahead value also leads to the same bound for the standard regret. Even more so, in most cases, lookahead information leads to a strict improvement in the value, that is, $G^{R}(s)$ , $G^{T}(s) \geq G_{0} > 0$ . When this happens, any algorithm with sub-linear lookahead regret enjoys a negative linear standard regret:

$$
I f \operatorname{Reg} ^ {R} (K) = o (K) a n d G ^ {R} (s _ {1} ^ {k}) \geq G _ {0} f o r a l l k \in [ K ], t h e n \operatorname{Reg} (K) \leq - G _ {0} K + o (K).
$$

The same also holds for transition lookahead. Conversely, any agent that suffers positive standard regret will suffer linear regret compared to the best lookahead agent, i.e.,

$$
I f \operatorname{Reg} (K) \geq 0 a n d G ^ {R} (s _ {1} ^ {k}) \geq G _ {0} f o r a l l k \in [ K ], t h e n \operatorname{Reg} ^ {R} (K) \geq G _ {0} K.
$$

Notably, any agent that does not use lookahead information will suffer linear lookahead regret in any such environment. We now present two illustrative examples for environments where the lookahead value gain is significant, one for reward lookahead and another for transition lookahead.

Reward lookahead. Consider a simple 2-state environment, depicted in Figure 1. Starting at $s_i$ , agents can either stay there by playing $a_1$ , earning no reward, or play any other action and move to the absorbing $s_f$ , obtaining a Bernoulli reward $Ber(1/(A-1)H)$ . Actions in the terminal state $s_f$ yield no reward. Without observing the rewards, agents will arbitrarily move from $s_i$ to $s_f$ , obtaining a reward $V^{no} = 1/(A-1)H$ in expectation. On the other hand, when agents observe the rewards before acting, they should move from $s_i$ to $s_f$ only if a reward was realized for some action (and otherwise, stay in $s_i$ by playing $a_1$ ). Such agents will have $(A-1)H$ opportunities to observe a unit reward across all timesteps and actions, collecting in expectation $V^{R,*} = (1 - 1/(A-1)H)^{(A-1)H} \geq 1 - 1/e$ . In other words, just by observing the rewards before acting, the agent's value multiplicatively increases by almost $V^{R,*}/V^{no} \approx AH$ . Moreover, the additive value gain is $G^{R} \approx 1 - \frac{1}{e}$ , so sub-linear lookahead regret with reward information results with a negatively-linear standard regret of $\text{Reg}(K) \lesssim -(1 - \frac{1}{e})K$ .

Transition lookahead. Consider a chain of H/2 states (also described in further detail at Appendix C.9 and depicted at Figure 2). In each state, one action deterministically keeps the agent in its

![](images/2f6b56cd31ea6065b908b3b961e64ca06cfd74a5ac2785285500d3c3c44de349.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["s_i"] -->|a1\nR = 0| B["s_f"]
    B -->|∀a\nR = 0| A
    A -->|R(a) ~ Ber\left(\frac{1}{(A-1)H}\right)| B
```
</details>

Figure 1: Two-state prophet-like problem

current state, while all other actions move the agent one state forward w.p. 1/A, but lead to a terminal non-rewarding state otherwise. If the reward is located at the end of the chain, any standard RL agent can collect it only at an exponentially low probability. On the other hand, transition lookahead agents would move forward only if there is an action that allows it while staying at their current state otherwise; such agents will collect the rewards at the end of the chain with constant probability. More specifically, any no-lookahead agent can collect at most $V^{no} = \mathcal{O}(HA^{-H/2})$ rewards, while transition lookahead agents can collect $V^{T,*} = \Omega(H)$ ; as such, lookahead agents achieve exponential increase in value, and sublinear regret versus the best lookahead agent will yield a standard regret of $\text{Reg}(K) \lesssim -HK$ .

In the following sections, we will present agents that are guaranteed to always achieve sublinear regret compared to the best lookahead agent.

# 4 Planning and Learning with One-Step Reward Lookahead

In this section, we analyze RL settings with one-step reward lookahead, in which immediate rewards are observed before choosing an action. One well-known example of this situation is the prophet problem [Correa et al., 2019], where an agent sequentially observes values from known distributions. Upon observing a value, the agent decides whether to take it as a reward and stop the interaction, or discard it and continue to observe more values. This problem has numerous applications and extensions concerning auctions and posted-price mechanisms [Correa et al., 2017]. As shown in [Merlis et al., 2024], it is critical to observe the distribution values before taking a decision; otherwise, the agent's revenue can decrease by a factor of $H$ . Notably, the example presented in Figure 1 is a small variant of the prophet problem, where the agent can either take one of $A - 1$ values and finish the interaction or discard them and continue playing by staying at $s_i$ ; we showed that for this example, the lookahead information increases the value by a factor of $V^{R,*} / V^{no} \approx AH$ .

The most natural way to tackle this setting is to extend (augment) the state space to contain the observed rewards; this way, we transition from a state and reward observations to a new state with new reward observations and return to the vanilla MDP formulation. However, this comes at a great cost. Even for Bernoulli rewards, there are $2^{A}$ possible reward combinations at any given state, and the augmentation increases the state space by this factor – leading to an exponentially-large state space. Even worse, for continuous rewards, the augmented state space becomes continuous, and any performance guarantees that depend on the size of the state space immediately become vacuous. Hence, algorithms that naively use this reduction are expected to be both computationally and statistically intractable. We refer to Appendix B.2 for further details on one such augmentation.

We take a different approach and derive Bellman equations for this setting in the original state space.

Proposition 1. The optimal value of one-step reward lookahead agents satisfies

$$
V _ {H + 1} ^ {R, *} (s) = 0, \quad \forall s \in \mathcal {S},
$$

$$
V _ {h} ^ {R, *} (s) = \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \left[ \max _ {a \in \mathcal {A}} \left\{R _ {h} (s, a) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, a) V _ {h + 1} ^ {R, *} (s ^ {\prime}) \right\} \right], \quad \forall s \in \mathcal {S}, h \in [ H ].
$$

Also, given reward observations $\mathbf{R} = \{R(a)\}_{a\in \mathcal{A}}$ at state $s$ and step $h$ , the optimal policy is

$$
\pi_ {h} ^ {*} (s, \boldsymbol {R}) \in \underset {a \in \mathcal {A}} {\arg \max} \Bigg \{R (a) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, a) V _ {h + 1} ^ {R, *} (s ^ {\prime}) \Bigg \}.
$$

We prove Proposition 1 in Appendix B.2, where we present an equivalent environment with extended state space in which one could apply the standard Bellman equations [Puterman, 2014] to calculate the value with reward lookahead. In contrast to the previously discussed augmentation approach, we find it more convenient to divide the augmentation into two steps - at odd steps $2h - 1$ , the augmented environment would be in a state $s_h \times \mathbf{0}$ , while at even steps $2h$ , the state is $s_h \times \mathbf{R}_h$ . Doing so creates an overlap between the values of the original and augmented environments at odd steps, simplifying the proofs. We also use this augmentation to prove a variant of the law of total variance [LTV, e.g. Azar et al., 2017] and a value-difference lemma [e.g. Efroni et al., 2019b].

We remark that calculating the exact value is not always tractable - even for $S = H = 1$ (bandit problems) and Gaussian rewards, Proposition 1 requires calculating the expectation of the maximum

Algorithm 1 Monotonic Value Propagation with Reward Lookahead (MVP-RL)   
1: Require: $\delta \in (0,1)$ , bonuses $b_{k,h}^{r}(s), b_{k,h}^{p}(s,a)$ 2: for $k = 1,2,\ldots$ do
3: Initialize $\bar{V}_{H+1}^{k}(s) = 0$ 4: for $h = H, H - 1,..,1$ do
5: Calculate the truncated values for all $s \in S$ $\bar{V}_{h}^{k}(s) = \min\left\{\mathbb{E}_{\boldsymbol{R}\sim\hat{\mathcal{R}}_{h}^{k-1}(s)}\left[\max_{a \in \mathcal{A}}\left\{R(a) + b_{k,h}^{p}(s,a) + \hat{P}_{h}^{k-1}\bar{V}_{h+1}^{k}(s,a)\right\}\right] + b_{k,h}^{r}(s), H\right\}$ 6: end for
7: for $h = 1,2,\ldots H$ do
8: Observe $s_{h}^{k}$ and $R_{h}^{k}(s_{h}^{k}, a)$ for all $a \in \mathcal{A}$ 9: Play an action $a_{h}^{k} \in \arg \max_{a \in \mathcal{A}} \left\{R_{h}^{k}(s_{h}^{k}, a) + b_{k,h}^{p}(s_{h}^{k}, a) + \hat{P}_{h}^{k-1}\bar{V}_{h+1}^{k}(s_{h}^{k}, a)\right\}$ 10: Collect the reward $R_{h}^{k}(s_{h}^{k}, a_{h}^{k})$ and transition to the next state $s_{h+1}^{k} \sim P_{h}(\cdot | s_{h}^{k}, a_{h}^{k})$ 11: end for
12: end for

of Gaussian random variables, which does not admit any simple closed-form solution. On the other hand, these equations allow approximating the value by using reward samples – in the following, we show that it can be used to achieve tight regret bounds when the environment is unknown.

# 4.1 Regret-Minimization with Reward Lookahead

We now present a tractable algorithm that achieves tight regret bounds with one-step reward lookahead. Specifically, we modify the Monotonic Value Propagation (MVP) algorithm [Zhang et al., 2021b] to perform planning using the empirical reward distributions – instead of using the empirical reward expectations. To compensate for transition uncertainty, we add a transition bonus that uses the variance of the optimistic next-state values (w.r.t. the empirical transition kernel), designed to be monotone in the future value. Such construction permits using the variance of optimistic values for the bonus calculation while being able to later replace it with the variance of the optimal value (see discussion in Zhang et al. 2021b). A reward bonus is used for the value calculation, but does not affect the action choice in the current state. Intuitively, this is because we get the same amount of information for all the actions of a state, so they have the same level of uncertainty – there is no need for bonuses to encourage reward exploration at the action level.

A high-level description of the algorithm is presented in Algorithm 1, while the full algorithm and its bonuses are stated in Appendix B.3. Notice that the planning requires calculating the expected maximum using the empirical distribution, whose support always contains at most K elements, so both the memory and computations are polynomial. The algorithm ensures the following guarantees:

Theorem 1. When running MVP-RL, with probability at least $1 - \delta$ uniformly for all $K \geq 1$ , it holds that $\operatorname{Reg}^R(K) \leq \mathcal{O}\left(\sqrt{H^3 SAK} \ln \frac{SAHK}{\delta} + H^3 S^2 A\left(\ln \frac{SAHK}{\delta}\right)^2\right)$ .

See proof in Appendix B.7. Remarkably, our upper bound matches the standard lower bound for episodic RL of $\Omega\left(\sqrt{H^3 SAK}\right)$ [Domingues et al., 2021] up to log-factors; this lower bound is proved for known deterministic rewards, so in particular, it also holds for problems with reward lookahead.

To our knowledge, the only comparable bounds in settings with reward lookahead were proven to prophet problems; as agents observe (up to) n distributions at a fixed order, it can be formulated as a deterministic chain-like MDP, with H = n, $S = n + 1$ and A = 2. Agents start at the head of the chain and can either advance without collecting a reward or collect the observed reward and move to a terminal non-rewarding state (for more details, see Merlis et al. 2024). For this problem, [Gatmiry et al., 2024] proved a regret bound of $\tilde{\mathcal{O}}(n^{3}\sqrt{K})$ (albeit requiring a weaker form of feedback), and [Agarwal et al., 2023] proved a bound of $\tilde{\mathcal{O}}(n\sqrt{T})$ – slightly better than ours, but heavily relies on the ability to control which distributions to observe, which is a specific instance of deterministic transitions. We are unaware of any previous results that cover general Markovian dynamics.

# 4.2 Proof Concepts

When analyzing the regret of RL algorithms, a key step usually involves bounding the difference between the value of a policy in two different environments ('value-difference lemma'). In particular, for a given policy $\pi^{k}$ , many algorithms maintain a confidence interval on the value $V_{h}^{\pi^{k}}(s) \in [V_{h}^{k}(s), \bar{V}_{h}^{k}(s)]$ , calculated based on optimistic and pessimistic MDPs that use the empirical model with bonuses/penalties [Dann et al., 2019, Zanette and Brunskill, 2019, Efroni et al., 2021]. Then, the instantaneous regret (without lookahead) is bounded using the optimistic values by

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s _ {h}) - V _ {h} ^ {\pi^ {k}} (s _ {h}) = \left(\hat {r} _ {h} ^ {k - 1} (s _ {h}, a _ {h}) - r _ {h} (s _ {h}, a _ {h})\right) + \left(\hat {P} _ {h} ^ {k - 1} - P _ {h}\right) \bar {V} _ {h} ^ {k} (s _ {h}, a _ {h}) \\ + P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) \left(s _ {h}, a _ {h}\right) + \text { bonuses }, \\ \end{array}
$$

while the pessimistic values are used either as part of the bonuses or while bounding them. However, when trying to perform a similar decomposition with reward lookahead, we do not have the difference of expected rewards, but rather terms of the form

$$
\mathbb {E} _ {\boldsymbol {R} \sim \hat {\mathcal {R}} _ {h} ^ {k - 1} (s _ {h})} \left[ R (\pi_ {h} ^ {k} (s _ {h}, \boldsymbol {R})) \right] - \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s _ {h})} \left[ R (\pi_ {h} ^ {k} (s _ {h}, \boldsymbol {R})) \right]
$$

(see, e.g., the last term of Lemma 4 in the appendix). As the action can be an arbitrary function of the reward realization, this term is extremely challenging to bound. For example, one could couple both distributions while trying to relate this error term to a Wasserstein distance between the empirical and real reward distribution; however, such distances exhibit much slower error rates than standard mean estimation [Fournier and Guillin, 2015]. Instead, we follow a different approach and show that uniformly for all possible expected next-state values $\hat{P}V\in[0,H]^A$ (as a function of the action at a given state), it holds w.h.p. that

$$
\begin{array}{l} \left| \mathbb {E} _ {\boldsymbol {R} \sim \hat {\mathcal {R}} _ {h} ^ {k - 1} (s)} \Bigl [ \max _ {a} \Bigl \{R (a) + \hat {P} V (s, a) \Bigr \} \Bigr ] - \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \Bigl [ \max _ {a} \Bigl \{R (a) + \hat {P} V (s, a) \Bigr \} \Bigr ] \right| \\ \lesssim \sqrt {\frac {A \ln \frac {1}{\delta}}{n _ {h} ^ {k - 1} (s) \vee 1}}. \tag {1} \\ \end{array}
$$

Throughout the proof, whenever we face an expectation w.r.t. the empirical rewards, we reformulate the expression to fit the form of Equation (1) and use it as a ‘change of measure’ tool. We remark that while this confidence interval admits an extra A-factor compared to standard bounds, the counts only depend on the visits to the state (and not to the state-action), which compensates for this factor.

The choice of MVP for the bonus is similarly motivated – unlike some other bonuses (e.g., Zanette and Brunskill 2019), MVP does not require pessimistic values – either in the bonus itself or in its analysis. In contrast to the optimistic ones, the pessimistic values are not calculated via value iteration, but rather by following the policy $\pi^{k}$ in the pessimistic environment. As such, they cannot be easily manipulated to fit the form in Equation (1).

The analysis of the transitions adapts the techniques in [Efroni et al., 2021], while requiring extra care in handling the dependence of actions in the rewards.

# 5 Reinforcement Learning with One-Step Transition Lookahead

We now move to analyzing problems with one-step transition lookahead, where the resulting next state due to playing any of the actions is revealed before deciding which action to play. For example, consider the stochastic Canadian traveler problem with resampling [Nikolova and Karger, 2008, Boutilier et al., 2018]. In this problem, an agent wants to navigate on a graph as fast as possible from a source to a target, but observes which edges at a node are available only upon reaching this node. When edge availability is stochastic and resampled every time a node is visited, this is a clear case of one-step transition lookahead, as the information on the availability of edges is given before trying to traverse them. The example in Section 3 and Appendix C.9 is one possible formulation of this problem on a chain – agents are awarded for arriving at the end of the chain as fast as possible, but trying to use a non-existing edge results with termination. We showed that in this particular instance, the lookahead value is exponentially larger than the standard value, and any lookahead agent with low regret would greatly surpass no-lookahead agents.

As with reward lookahead, the future states for all actions can be embedded into the state, but doing so increases the size of the state space by a factor of $S^{A}$ , again making this approach intractable (see Appendix C.2 for an example for such an extension). We once more show that this is not necessary; the transition-lookahead optimal values can be calculated using the following Bellman equations:

Proposition 2. The optimal value of one-step transition lookahead agents satisfies

$$
V _ {H + 1} ^ {T, *} (s) = 0, \quad \forall s \in \mathcal {S},
$$

$$
V _ {h} ^ {T, *} (s) = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \max _ {a \in \mathcal {A}} \left\{r _ {h} (s, a) + V _ {h + 1} ^ {T, *} \left(s ^ {\prime} (s, a)\right) \right\} \right], \quad \forall s \in \mathcal {S}, h \in [ H ].
$$

Also, given next-state observations $\mathbf{s}' = \{s'(a)\}_{a \in \mathcal{A}}$ at state $s$ and step $h$ , the optimal policy is

$$
\pi_ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \in \operatorname * {a r g   m a x} _ {a \in \mathcal {A}} \Bigl \{r _ {h} (s, a) + V _ {h + 1} ^ {T, *} (s ^ {\prime} (a)) \Bigr \}.
$$

The proof can be found at Appendix C.2 and again relies on augmenting the state space to incorporate the transitions; this time, we divide the episode into odd steps whose extended state is $s_h \times s_0'$ (for an arbitrary fixed $s_0' \in S^A$ ) and even steps with the state $s_h \times s_{h+1}'$ . Beyond planning, this again allows proving a variant of the LTV and of a value-difference lemma.

One important insight is that the policy $\pi_{h}^{*}(s,s^{\prime})$ admits the form of a list. Namely, consider the values $V_{h}^{*}(s,s^{\prime},a)=r_{h}(s,a)+V_{h+1}^{T,*}(s^{\prime})$ and assume some ordering of next-state-action pairs $\{(s_{i}^{\prime},a_{i})\}_{i=1}^{SA}$ such that $V_{h}^{*}(s,s_{1}^{\prime},a_{1})\geq\cdots\geq V_{h}^{*}(s,s_{SA}^{\prime},a_{SA})$ . Then, an optimal policy would look at all realized pairs $(s^{\prime}(a),a)$ and play the action with the highest location in this list. We refer the readers to Appendix C.4 for an additional discussion on list representations in transition lookahead.

Similar results could be achieved through a reduction to RL problems with stochastic action sets [Boutilier et al., 2018]. There, at every round, a subset of base actions is sampled, and only these actions are available to the agent. In particular, one could sample $A$ actions of the form $(s', a) \in S \times A$ and impose a deterministic transition to $s'$ given this extended action. However, since every original action must be sampled exactly once, this sampling procedure creates a dependence between pairs even when next-states at different actions are independent, adding unnecessary complications. We show that when transitions are independent between states, the expectation in Proposition 2 can be efficiently calculated (see Appendix C.4.1 for details), and otherwise, it can be approximated through sampling, as we do in learning settings.

# 5.1 Regret-Minimization with Transition Lookahead

Relying on similar principals as with reward lookahead, we now present MVP-TL, an adaptation of MVP to settings with one-step transition lookahead (summarized in Algorithm 2; the full details can be found at Appendix C.3). This time, we estimate the empirical expected reward and add a standard Hoeffding-like reward bonus, while performing planning using samples from the empirical joint distribution of the next-state for all the actions simultaneously. A variance-based transition bonus is added to the values; though this time, the variance also incorporates the rewards, namely

$$
b _ {k, h} ^ {p} (s) \approx \sqrt {\frac {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} (\bar {V} _ {h} ^ {k} (s , \boldsymbol {s} ^ {\prime}))}{n _ {h} ^ {k - 1} (s) \vee 1}}, \quad \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) = \max _ {a \in \mathcal {A}} \Bigl \{\hat {r} _ {h} ^ {k - 1} (s, a) + b _ {k, h} ^ {r} (s, a) + \bar {V} _ {h + 1} ^ {k} (s ^ {\prime} (a) \Bigr \}.
$$

The motivation for this modification is the technical challenges described in Section 4.2, in the context of reward lookahead. For reward lookahead, we analyzed a value term that included both the rewards and next-state values, and used concentration arguments to move from the empirical reward distribution to the real one. For transition lookahead, similar values are analyzed, but we require variance-based concentration to obtain tighter regret bounds [Azar et al., 2017], so this variance naturally arises. The bonus is again designed to be monotone, as in the original MVP algorithm, and does not affect the immediate action choice – only the optimistic lookahead value. As before, the planning relies on sampling the next-state observations at previous episodes, and so it is polynomial, even if the precise joint distribution is complex. The algorithm enjoys the following regret bounds:

Theorem 2. When running MVP-TL, with probability at least $1 - \delta$ uniformly for all $K \geq 1$ , it holds that $\operatorname{Reg}^T(K) \leq \mathcal{O}\left(\sqrt{H^2SK}\left(\sqrt{H} + \sqrt{A}\right)\ln \frac{SAHK}{\delta} + H^3S^4A^3\left(\ln \frac{SAHK}{\delta}\right)^2\right)$ .

Algorithm 2 Monotonic Value Propagation with Transition Lookahead (MVP-TL)   
1: Require: $\delta\in(0,1)$ , bonuses $b_{k,h}^{r}(s,a)$ , $b_{k,h}^{p}(s)$ 2: for $k=1,2,\ldots$ do
3: Initialize $\bar{V}_{H+1}^{k}(s)=0$ 4: for $h=H,H-1,\ldots,1$ do
5: Calculate the truncated values for all $s\in S$ $\bar{V}_{h}^{k}(s)=\min\left\{\mathbb{E}_{\boldsymbol{s}^{\prime}\sim\hat{P}_{h}^{k-1}(s)}\left[\max_{a\in\mathcal{A}}\left\{\hat{r}_{h}^{k-1}(s,a)+b_{k,h}^{r}(s,a)+\bar{V}_{h+1}^{k}(s^{\prime}(a))\right\}\right]+b_{k,h}^{p}(s),H\right\}$ 6: end for
7: for $h=1,2,\ldots,H$ do
8: Observe $s_{h}^{k}$ and $s_{h+1}^{\prime k}(s_{h}^{k},a)$ for all $a\in A$ 9: Play an action $a_{h}^{k}\in\arg\max_{a\in\mathcal{A}}\left\{\hat{r}_{h}^{k-1}(s_{h}^{k},a)+b_{k,h}^{r}(s_{h}^{k},a)+\bar{V}_{h+1}^{k}(s_{h+1}^{\prime k}(s_{h}^{k},a))\right\}$ 10: Collect the reward $R_{h}^{k}\sim\mathcal{R}_{h}(s_{h}^{k},a_{h}^{k})$ and transition to the next state $s_{h+1}^{k}=s_{h+1}^{\prime k}(s_{h}^{k},a_{h}^{k})$ 11: end for
12: end for

See proof in Appendix C.8. For transition lookahead, the regret bounds we provide exhibit two rates, both corresponding to a natural adaptation of known lower bounds to transition lookahead.

1. 'Bandit rate' $\mathcal{O}(\sqrt{H^2 SAK})$ : this is the rate due to reward stochasticity. Consider a problem where at odd timesteps $2h - 1$ and across all states, all actions have rewards of mean $^{1/2} - \epsilon$ , except for one action of mean $^{1/2}$ . Assuming that the state-distribution is uniform, each such timestep forms a hard instance of a contextual bandit problem with $S$ contexts, exhibiting a regret of $\Omega(\sqrt{SAK})$ [Auer et al., 2002, Bubeck et al., 2012]. Since there are $H/2$ odd steps and we can design each step independently, the total regret would be $\Omega(H\sqrt{SAK})$ . The even steps can be used to 'remove' the lookahead and create a uniform state distribution. To do so, we set that when taking an action at odd steps, we always transition to a fixed state $s_d$ . From this state, one action $a_1$ leads uniformly to all states, while the rest of the actions lead to an absorbing non-rewarding state - rendering them strictly suboptimal. Thus, no-regret agents will only play $a_1$ , regardless of the lookahead information, and the state distribution at odd timesteps will be uniform.   
2. ‘Transition learning rate’ $\mathcal{O}(\sqrt{H^{3}SK})$ : recall that the vanilla RL lower bound designs a tree with $\Omega(S)$ leaves, to which agents need to navigate at the right timing (with $\Omega(H)$ options) and take the right action (out of A). While all leaves might transition agents to a rewarding state, one combination of state-action-timing has a slightly higher probability of doing so [Domingues et al., 2021]. This roughly creates a bandit problem with SAH arms, constructed such that the maximal reward is $\Omega(H)$ , yielding a total regret of $H\sqrt{HSAK}$ . Now consider the following simple modification where in each leaf, only one action can lead to a reward (and the rest of the actions are ‘useless’ – never lead to rewards). Thus, the agent still needs to test all leaves at all timings, and so there are still SH ‘arms’ with a corresponding regret of $\sqrt{H^{3}SK}$ . Moreover, to test a leaf at a certain timing, we must navigate to it, and since the agent is going to play the single useful action at the leaf, transition lookahead does not provide any additional information.

As discussed before, transition lookahead can be formulated as an RL instance with stochastic action sets. While Boutilier et al. [2018] prove that with stochastic action sets, Q-learning asymptotically converges, they provide no learning algorithm nor regret bounds. Therefore, to our knowledge, our result is the first to achieve sublinear regret with transition lookahead.

# 5.2 Proof Concepts

Transition lookahead causes similar issues as reward lookahead. Hence, it is natural to apply a similar analysis approach – first, formulate the value as the expectation w.r.t. the next-state observations of the maximum of action-observation dependent values; then use uniform concentration as a ‘change of measure’ tool between the empirical and real next-state distribution. In particular, if $V(s, s', a)$ represents the value starting from state s, performing a and transitioning to $s'$ , one can show that for

all $V(s,\cdot ,\cdot)\in [0,H]^{SA}$ (see Lemma 19),

$$
\begin{array}{l} \left| \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \left[ \max _ {a} V (s, s ^ {\prime} (a), a) \right] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \max _ {a} V (s, s ^ {\prime} (a), a) \right] \right| \\ \lesssim \sqrt {\frac {S A \ln \frac {1}{\delta} \operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \max _ {a} V (s , s ^ {\prime} (a) , a)}{n _ {h} ^ {k - 1} (s) \vee 1}}, \tag {2} \\ \end{array}
$$

where the variance term stems from using a Bernstein-like concentration bound. However, in contrast to the reward lookahead, the $\sqrt{SA}$ -factor propagates to the dominant term of the regret, so pursuing this approach would lead to a worse regret bound of $\tilde{\mathcal{O}}\left(\sqrt{H^3 S^2 AK}\right)$ .

To avoid this, we pinpoint the two locations where this change of measure is needed – the proof that $\bar{V}_{h}^{k}$ is optimistic and the regret decomposition – and make sure to perform this change of measure only on a single value $V_{h}^{*}(s,s^{\prime},a)=r_{h}(s,a)+V_{h+1}^{*}(s^{\prime})$ , mitigating the need to cover all possible values and removing the additional $\sqrt{SA}$ -factor. However, doing so leaves us with a residual term. Defining $V_{h}^{*}(s,s^{\prime})=\max_{a\in\mathcal{A}}\{V_{h}^{*}(s,s^{\prime}(a),a)\}$ and assuming a similar optimistic value $\bar{V}_{h}^{k}(s,s^{\prime})$ , this term is of the form

$$
\mathbb {E} _ {\pmb {s ^ {\prime}} \sim \hat {P} _ {h} ^ {k - 1} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \pmb {s} ^ {\prime}) - V _ {h} ^ {*} (s, \pmb {s} ^ {\prime}) \big ] - \mathbb {E} _ {\pmb {s ^ {\prime}} \sim P _ {h} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \pmb {s} ^ {\prime}) - V _ {h} ^ {*} (s, \pmb {s} ^ {\prime}) \big ].
$$

While similar terms have been analyzed before [e.g., Zanette and Brunskill, 2019, Efroni et al., 2021], the analysis leads to a constant regret term that depends on the support of the distribution in question; in our case, it is the distribution over all possible next-states – of cardinality $S^{A}$ . Therefore, following the same derivation would lead to an exponential additive regret term.

We overcome it by utilizing the fact that both the optimistic policy and the optimal one decide which action to take according to a list of next-state-actions $(s', a)$ . In other words, instead of looking at the next-state $s'$ (with $S^{A}$ possible values) to determine a value, we look at the highest-ranked realized pair $(s', a)$ in the list that corresponds to the policy that induces the value (with SA possible rankings). Since we have two values, we need to calculate the probability of being at a certain list location for both $\pi^{k}$ and $\pi^{*}$ , but the cardinality of this space is $(SA)^{2}$ : polynomial and not exponential.

# 6 Conclusions and Future Work

In this work, we presented an RL setting in which immediate rewards or transitions are observed before actions are chosen. We showed how to design provably and computationally efficient algorithms for this setting that achieve tight regret bounds versus a strong baseline that also uses lookahead information. Our algorithms rely on estimating the distribution of the reward or transition observations, a concept that might be utilized in other settings. In particular, we believe that our techniques for transition lookahead could be extended to RL problems with stochastic action sets [Boutilier et al., 2018], but leave this for future work.

One natural extension to our work would be to consider multi-step lookahead information - observing the transition/rewards $L$ steps in advance. We conjecture that from a statistical point of view, a similar algorithmic approach that samples from the empirical observation distribution would be efficient. However, it is not clear how to perform efficient planning with such feedback.

Another possible direction would be to derive model-free algorithms [Jin et al., 2018], with the aim to improve the computation efficiency of the solutions; our model-based algorithms require at most $\mathcal{O}(KS^{2}AH)$ computations per episode due to the planning stage, while model-free algorithms might potentially allow just $\mathcal{O}(AH)$ computations per episode.

On the practical side, previous works presented RL algorithms that utilize/estimate a world model with multi-step lookahead to perform planning and learning [Schrittwieser et al., 2020, Chung et al., 2024], aiming to achieve the optimal no-lookahead value. For some of these approaches, it is quite natural to replace the simulated world behavior with lookahead information on the real future realization. We leave this adaptation and evaluation to future studies.

Finally, the notion of lookahead could be studied in various other decision-making settings (e.g., linear MDPs Jin et al. 2020) and can also be generalized to situations where lookahead information can be queried under some budget constraints [Efroni et al., 2021] or when agents only observe noisy lookahead predictions; we leave these problems for future research.

# Acknowledgements

We thank Alon Cohen and Austin Stromme for the helpful discussions. This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101034255.

# References

Arpit Agarwal, Rohan Ghuge, and Viswanath Nagarajan. Semi-bandit learning for monotone stochastic optimization. arXiv preprint arXiv:2312.15427, 2023.   
Peter Auer, Nicolo Cesa-Bianchi, Yoav Freund, and Robert E Schapire. The nonstochastic multiarmed bandit problem. SIAM journal on computing, 32(1):48–77, 2002.   
Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning, pages 263–272. PMLR, 2017.   
Dimitri Bertsekas. A course in reinforcement learning. Athena Scientific, 2023.   
André Biedenkapp, Raghu Rajan, Frank Hutter, and Marius Lindauer. Temporl: Learning when to act. In International Conference on Machine Learning, pages 914–924. PMLR, 2021.   
Craig Boutilier, Alon Cohen, Avinatan Hassidim, Yishay Mansour, Ofer Meshi, Martin Mladenov, and Dale Schuurmans. Planning and learning with stochastic action sets. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, pages 4674–4682, 2018.   
Sébastien Bubeck, Nicolo Cesa-Bianchi, et al. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends® in Machine Learning, 5(1):1–122, 2012.   
Eduardo F Camacho, Carlos Bordons, Eduardo F Camacho, and Carlos Bordons. Model predictive control. Springer, 2007.   
Stephen Chung, Ivan Anokhin, and David Krueger. Thinker: learning to plan and act. Advances in Neural Information Processing Systems, 36, 2024.   
José Correa, Patricio Foncea, Ruben Hoeksma, Tim Oosterwijk, and Tjark Vredeveld. Posted price mechanisms for a random stream of customers. In Proceedings of the 2017 ACM Conference on Economics and Computation, pages 169–186, 2017.   
Jose Correa, Patricio Foncea, Ruben Hoeksma, Tim Oosterwijk, and Tjark Vredeveld. Recent developments in prophet inequalities. ACM SIGecom Exchanges, 17(1):61–70, 2019.   
Christoph Dann, Lihong Li, Wei Wei, and Emma Brunskill. Policy certificates: Towards accountable reinforcement learning. In International Conference on Machine Learning, pages 1507–1516, 2019.   
Omar Darwiche Domingues, Pierre Ménard, Emilie Kaufmann, and Michal Valko. Episodic reinforcement learning in finite mdps: Minimax lower bounds revisited. In Algorithmic Learning Theory, pages 578–598. PMLR, 2021.   
Yonathan Efroni, Gal Dalal, Bruno Scherrer, and Shie Mannor. How to combine tree-search methods in reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 3494–3501, 2019a.   
Yonathan Efroni, Nadav Merlis, Mohammad Ghavamzadeh, and Shie Mannor. Tight regret bounds for model-based reinforcement learning with greedy policies. In Advances in Neural Information Processing Systems, pages 12224–12234, 2019b.   
Yonathan Efroni, Mohammad Ghavamzadeh, and Shie Mannor. Online planning with lookahead policies. Advances in Neural Information Processing Systems, 33:14024–14033, 2020.

Yonathan Efroni, Nadav Merlis, Aadirupa Saha, and Shie Mannor. Confidence-budget matching for sequential budgeted learning. In International Conference on Machine Learning, pages 2937–2947. PMLR, 2021.   
Ibrahim El Shar and Daniel Jiang. Lookahead-bounded q-learning. In International Conference on Machine Learning, pages 8665–8675. PMLR, 2020.   
Nicolas Fournier and Arnaud Guillin. On the rate of convergence in wasserstein distance of the empirical measure. Probability theory and related fields, 162(3):707–738, 2015.   
Khashayar Gatmiry, Thomas Kesselheim, Sahil Singla, and Yifan Wang. Bandit algorithms for prophet inequality and pandora's box. In Proceedings of the 2024 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 462–500. SIAM, 2024.   
Yunhan Huang, Veeraruna Kavitha, and Quanyan Zhu. Continuous-time markov decision processes with controlled observations. In 2019 57th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 32–39. IEEE, 2019.   
Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11(Apr):1563–1600, 2010.   
Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? Advances in neural information processing systems, 31, 2018.   
Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on learning theory, pages 2137–2143. PMLR, 2020.   
Yingying Li, Xin Chen, and Na Li. Online optimal control with linear dynamics and predictions: Algorithms and regret analysis. Advances in Neural Information Processing Systems, 32, 2019.   
Yiheng Lin, Yang Hu, Guanya Shi, Haoyuan Sun, Guannan Qu, and Adam Wierman. Perturbation-based regret analysis of predictive control in linear time varying systems. Advances in Neural Information Processing Systems, 34:5174–5185, 2021.   
Yiheng Lin, Yang Hu, Guannan Qu, Tongxin Li, and Adam Wierman. Bounded-regret mpc via perturbation analysis: Prediction error, constraints, and nonlinearity. Advances in Neural Information Processing Systems, 35:36174–36187, 2022.   
Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample variance penalization. In Conference on learning theory, 2009.   
Nadav Merlis, Dorian Baudry, and Vianney Perchet. The value of reward lookahead in reinforcement learning. arXiv preprint arXiv:2403.11637, 2024.   
Thomas M Moerland, Anna Deichler, Simone Baldi, Joost Broekens, and Catholijn M Jonker. Think neither too fast nor too slow: The computational trade-off between planning and reinforcement learning. In Proceedings of the International Conference on Automated Planning and Scheduling (ICAPS), Nancy, France, pages 16–20, 2020.   
Evdokia Nikolova and David R Karger. Route planning under uncertainty: The canadian traveller problem. In AAAI, pages 969–974, 2008.   
Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.   
Aviv Rosenberg, Assaf Hallak, Shie Mannor, Gal Chechik, and Gal Dalal. Planning and learning with adaptive lookahead. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 9606–9613, 2023.   
Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.

Max Simchowitz and Kevin G Jamieson. Non-asymptotic gap-dependent regret bounds for tabular mdps. In Advances in Neural Information Processing Systems, pages 1153–1162, 2019.   
Aviv Tamar, Garrett Thomas, Tianhao Zhang, Sergey Levine, and Pieter Abbeel. Learning from the hindsight plan—episodic mpc improvement. In 2017 IEEE International Conference on Robotics and Automation (ICRA), pages 336–343. IEEE, 2017.   
Chenkai Yu, Guanya Shi, Soon-Jo Chung, Yisong Yue, and Adam Wierman. The power of predictions in online control. Advances in Neural Information Processing Systems, 33:1994–2004, 2020.   
Andrea Zanette and Emma Brunskill. Tighter problem-dependent regret bounds in reinforcement learning without domain knowledge using value function bounds. In International Conference on Machine Learning, pages 7304–7312. PMLR, 2019.   
Runyu Zhang, Yingying Li, and Na Li. On the regret analysis of online lqr control with predictions. In 2021 American Control Conference (ACC), pages 697–703. IEEE, 2021a.   
Zihan Zhang, Xiangyang Ji, and Simon Du. Is reinforcement learning more difficult than bandits? a near-optimal algorithm escaping the curse of horizon. In Conference on Learning Theory, pages 4528–4531. PMLR, 2021b.   
Zihan Zhang, Yuxin Chen, Jason D Lee, and Simon S Du. Settling the sample complexity of online reinforcement learning. arXiv preprint arXiv:2307.13586, 2023.

# Table of Contents

A Structure of the Appendix 14

B Proofs for Reward Lookahead 15

B.1 Data Generation Process 15   
B.2 Extended MDP for Reward Lookahead 15   
B.3 Full Algorithm Description for Reward Lookahead 20   
B.4 The First Good Event – Concentration 21   
B.5 Optimism of the Upper Confidence Value Functions 22   
B.6 The Second Good Event – Martingale Concentration ..... 23   
B.7 Regret Analysis 25

C Proofs for Transition Lookahead 30

C.1 Data Generation Process 30   
C.2 Extended MDP for Transition Lookahead 30   
C.3 Full Algorithm Description for Transition Lookahead ..... 34   
C.4 Additional Notations and List Representation 35   
C.5 The First Good Event – Concentration 37   
C.6 Optimism of the Upper Confidence Value Functions 39   
C.7 The Second Good Event - Martingale Concentration 40   
C.8 Regret Analysis 41   
C.9 Example: Value Gain due to Transition Lookahead 46

D Auxiliary Lemmas 47

D.1 Concentration results 47   
D.2 Count-Related Lemmas 51   
D.3 Analysis of Variance terms 52

E Existing Results 53

# A Structure of the Appendix

Both reward and transition lookahead appendices share the following structure. First, we describe our assumption on the data generation process and analyze general properties of reward and transition lookahead. This is done by looking at an extended MDP that incorporates the lookahead information into the state. Then, we present the full algorithm and describe the relevant probabilistic events that ensure the concentration of all the empirical quantities. For transition lookahead, we require some additional notions for the event definitions (including the list representation of values and policies), which are explained in a separate subsection.

Given the concentration-related good event, we can prove that the planning procedure in the algorithm is optimistic, which we do in the subsequent subsection. Then, we define an additional good event that allows adding and removing conditional expectations in a way that will be needed for the proof.

At this point, we provided all (almost all) the results required for the regret analysis, and the proof of the main theorems is stated. The proofs also require some additional analysis for the bonuses (and especially variance terms), which is located at the end of the regret analysis.

For transition lookahead, the appendix includes one more part that further analyzes the example presented in Section 3.

At the end of the appendix, we state and prove several lemmas that will be used throughout our analysis, while also stating several existing results that will be of use.

# B Proofs for Reward Lookahead

# B.1 Data Generation Process

To simplify the proofs, we assume the following 'tabular' data-generation process: Before the game starts, a set of $K$ samples from the transition probabilities and rewards is generated for all $(s, a, h)$ . Once a state $s$ at step $h$ is visited for the $i^{th}$ time, the $i^{th}$ sample from the reward distribution $\mathcal{R}_h(s)$ is the reward realization for all action $a \in \mathcal{A}$ . When a state-action pair is visited for the $i^{th}$ time, the $i^{th}$ sample from the transition kernel $P_h(\cdot | s, a)$ determines the next-state realization. In particular, it implies that the reward samples from the first $i$ visits to a state are i.i.d., and the same for the next-states samples and state-action visitations. Throughout this appendix, we use the notation $\boldsymbol{R}_h^k = \left\{ R_h^k(s_h^k, a) \right\}_{a \in \mathcal{A}}$ to denote the reward observation at episode $k$ and timestep $h$ for all the actions.

For the proof, we define the following three filtrations. Let

$$
F _ {k, h} = \sigma \bigg (\left\{s _ {t} ^ {1}, a _ {t} ^ {1}, \boldsymbol {R} _ {t} ^ {1} \right\} _ {t \in [ H ]}, \ldots , \left\{s _ {t} ^ {k - 1}, a _ {t} ^ {k - 1}, \boldsymbol {R} _ {t} ^ {k - 1} \right\} _ {t \in [ H ]}, \left\{s _ {t} ^ {k}, a _ {t} ^ {k}, \boldsymbol {R} _ {t} ^ {k} \right\} _ {t \in [ h ]}, s _ {h + 1} ^ {k} \bigg),
$$

$$
F _ {k, h} ^ {R} = \sigma \bigg (\left\{s _ {t} ^ {1}, a _ {t} ^ {1}, \boldsymbol {R} _ {t} ^ {1} \right\} _ {t \in [ H ]}, \ldots , \left\{s _ {t} ^ {k - 1}, a _ {t} ^ {k - 1}, \boldsymbol {R} _ {t} ^ {k - 1} \right\} _ {t \in [ H ]}, \left\{s _ {t} ^ {k}, a _ {t} ^ {k}, \boldsymbol {R} _ {t} ^ {k} \right\} _ {t \in [ h + 1 ]} \bigg),
$$

the filtrations that contains all information until episode $k$ and step $h$ , as well as the state at timestep $h + 1$ , or all information of time $h + 1$ , respectively. We make this distinction so that $F_{k,h - 1}$ contains only $s_h^k$ , while $F_{k,h - 1}^R$ also contains $a_h^k$ . We also define

$$
F _ {k} = \sigma \bigg (\left\{s _ {t} ^ {1}, a _ {t} ^ {1}, \boldsymbol {R} _ {t} ^ {1} \right\} _ {t \in [ H ]}, \ldots , \left\{s _ {t} ^ {k}, a _ {t} ^ {k}, \boldsymbol {R} _ {t} ^ {k} \right\} _ {t \in [ H ]}, s _ {1} ^ {k + 1} \bigg),
$$

which contains all information up to the end of the $k^{th}$ episode, as well as the initial state at episode $k + 1$ .

# B.2 Extended MDP for Reward Lookahead

In this appendix, we present an alternative formulation of the one-step reward lookahead that falls under the vanilla (no-lookahead) model and would be helpful for the analysis.

Throughout the section, we study the relations between MDPs with and without reward lookahead, and between different MDPs with lookahead. Therefore, for clarity, we state the concerning MDP in the value, e.g. $V^{R,\pi}(s|\mathcal{M})$ . Specifically in this subsection, we distinguish between values without lookahead (denoted $V^{\pi}$ ) and values with lookahead (denoted $V^{R,\pi}$ ). In the following subsections, unless stated otherwise, we will only consider lookahead values; for brevity, and with some abuse of notations, we will then omit the R in the value notation.

For any MDP $\mathcal{M} = (\mathcal{S},\mathcal{A},H,P,\mathcal{R})$ , define an equivalent extended MDP $\mathcal{M}^R$ of horizon $2H$ that separates the state transition and reward generation as follows:

1. Assume w.l.o.g. that $\mathcal{M}$ starts at some initial state $s_1$ . The extended environment starts at a state $s_1 \times \mathbf{0}$ , where $\mathbf{0} \in \mathbb{R}^A$ is the zeros vector.   
2. For any $h \in [H]$ , at timestep $2h - 1$ , the environment $\mathcal{M}^R$ transitions from state $s_h \times \mathbf{0}$ to $s_h \times \mathbf{R}$ , where $\mathbf{R} \sim \mathcal{R}_h(s)$ is a vector containing the rewards for all actions $a \in \mathcal{A}$ . This transition occurs regardless of the action that was played. At timestep $2h$ , given an action $a_h$ the environment transitions from $s_h \times \mathbf{R}$ to $s_{h+1} \times \mathbf{0}$ , where $s_{h+1} \sim P_h(\cdot | s_h, a_h)$ .   
3. The reward at a state $s \times R$ when playing an action $a$ is $R(a)$ , namely, the reward is deterministic and only obtained on even timesteps.

We emphasize that throughout the section, we assume that $\mathcal{M}$ and $\mathcal{M}^R$ are coupled; that is, assume that under a policy $\pi$ in $\mathcal{M}$ , the agent visits a state $s_h$ , observes $\boldsymbol{R}_h$ , plays an action $a_h$ and transitions to $s_{h+1}$ . Then, in $\mathcal{M}^R$ , the agent starts from $s_h \times \mathbf{0}$ , transitions to $s_h \times \boldsymbol{R}$ (regardless of the action it played), takes the action $a_h$ and finally transitions to $s_{h+1} \times \mathbf{0}$ .

Since the reward is embedded into the state, any state-dependent policy in $M^{R}$ is a one-step reward lookahead policy in the original MDP. Moreover, the policy at the odd steps of M does not affect

the value, and assuming that the policy at the even steps in $\mathcal{M}^R$ is the same as the policy in $\mathcal{M}$ , we trivially get the following relation between the values

$$
V _ {2 h} ^ {\pi} (s, \pmb {R} | \mathcal {M} ^ {R}) = \mathbb {E} \left[ \sum_ {t = h} ^ {H} R _ {t} (s _ {t}, a _ {t}) | s _ {h} = s, R _ {h} (s, \cdot) = \pmb {R}, \pi \right] \triangleq V _ {h} ^ {R, \pi} (s, \pmb {R} | \mathcal {M}),
$$

$$
V _ {2 h - 1} ^ {\pi} (s, \mathbf {0} | \mathcal {M} ^ {R}) = \mathbb {E} \left[ \sum_ {t = h} ^ {H} R _ {t} (s _ {t}, a _ {t}) | s _ {h} = s, \pi \right] = V _ {h} ^ {R, \pi} (s | \mathcal {M}). \tag {3}
$$

While $M^{R}$ has a continuous state space, which generally makes algorithm design impractical, this representation permits applying classic results on MDPs to environments with one-step lookahead.

As a remark, rewards could be directly embedded into the state without separating the state and reward updates. However, this creates unnecessary complications when analyzing the relations between similar environments. This is because we are mainly interested in the value given the state – in expectation over the realized rewards. In particular, value-difference are analyzed assuming a shared initial state, but in our case, we do not want to assume the same reward realization, but rather also account for the distance between reward distributions, which the step separation enables. For similar reasons, this representation also simplifies the proof of the law of total variance [Azar et al., 2017].

Proposition 1. The optimal value of one-step reward lookahead agents satisfies

$$
V _ {H + 1} ^ {R, *} (s) = 0, \quad \forall s \in \mathcal {S},
$$

$$
V _ {h} ^ {R, *} (s) = \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \left[ \max _ {a \in \mathcal {A}} \left\{R _ {h} (s, a) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, a) V _ {h + 1} ^ {R, *} (s ^ {\prime}) \right\} \right], \quad \forall s \in \mathcal {S}, h \in [ H ].
$$

Also, given reward observations $\boldsymbol{R} = \{R(a)\}_{a\in \mathcal{A}}$ at state $s$ and step $h$ , the optimal policy is

$$
\pi_ {h} ^ {*} (s, \boldsymbol {R}) \in \underset {a \in \mathcal {A}} {\arg \max} \Bigg \{R (a) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, a) V _ {h + 1} ^ {R, *} (s ^ {\prime}) \Bigg \}.
$$

Proof. We prove the result in the extended MDP $M^{R}$ and remind the reader that in this formulation, the policy only uses state information, as in the standard RL formulation. In particular, it implies that there exists a Markovian optimal policy that uniformly maximizes the value (in the extended state space), and the optimal value is given through the dynamic-programming equations [Puterman, 2014]

$$
V _ {2 H + 1} ^ {*} (s, \boldsymbol {R} | \mathcal {M} ^ {R}) = 0, \quad \forall s \in \mathcal {S}, \boldsymbol {R} \in \mathbb {R} ^ {A},
$$

$$
V _ {2 h} ^ {*} (s, \boldsymbol {R} | \mathcal {M} ^ {R}) = \max _ {a} \Bigg \{R (a) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, a) V _ {2 h + 1} ^ {*} (s ^ {\prime}, \mathbf {0} | \mathcal {M} ^ {R}) \Bigg \}, \quad \forall h \in [ H ], s \in \mathcal {S}, \boldsymbol {R} \in \mathbb {R} ^ {A},
$$

$$
V _ {2 h - 1} ^ {*} (s, \mathbf {0} | \mathcal {M} ^ {R}) = \mathbb {E} _ {\mathcal {R} _ {h} (s)} \left[ V _ {2 h} ^ {*} (s, \boldsymbol {R} | \mathcal {M} ^ {R}) \right], \quad \forall h \in [ H ], s \in \mathcal {S}. \tag {4}
$$

By the equivalence between $\mathcal{M}$ and $\mathcal{M}^R$ for all policies, this is also the optimal value in $\mathcal{M}$ . Specifically, combining both recursion equations and substituting the relation between the original and extended values of Equation (3), we get the desired value recursion for any $h\in [H]$ and $s\in S$ :

$$
\begin{array}{l} V _ {h} ^ {R, *} (s | \mathcal {M}) = V _ {2 h - 1} ^ {*} (s, \mathbf {0} | \mathcal {M} ^ {R}) \\ = \mathbb {E} _ {\mathcal {R} _ {h} (s)} \left[ V _ {2 h} ^ {*} (s, \boldsymbol {R} | \mathcal {M} ^ {R}) \right] \\ = \mathbb {E} _ {\mathcal {R} _ {h} (s)} \left[ \max _ {a} \left\{R (a) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, a) V _ {2 h + 1} ^ {*} (s ^ {\prime}, \mathbf {0} | \mathcal {M} ^ {R}) \right\} \right] \\ = \mathbb {E} _ {\mathcal {R} _ {h} (s)} \left[ \max _ {a} \left\{R (a) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, a) V _ {h + 1} ^ {R, *} (s | \mathcal {M}) \right\} \right]. \\ \end{array}
$$

Similarly, for any $h \in [H]$ , $s \in S$ and $R \in \mathbb{R}^A$ , the optimal policy at the even stages of the extended MDP is

$$
\pi_ {2 h} ^ {*} (s, \boldsymbol {R}) \in \underset {a \in \mathcal {A}} {\arg \max} \Bigg \{R (a) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, a) V _ {2 h + 1} ^ {*} (s ^ {\prime}, \mathbf {0} | \mathcal {M} ^ {R}) \Bigg \},
$$

alongside arbitrary actions at odd steps. Playing this policy in the original MDP will lead to an optimal one-step reward lookahead policy, as it achieves the optimal value of the original MDP. This policy directly translates to the optimal policy in the statement, by the equivalence between the original and extended MDPs and the relation $V_{2h+1}^{*}(s', \mathbf{0}|\mathcal{M}^{R}) = V_{h+1}^{R,*}(s'|\mathcal{M})$ . ☐

Remark 1. As in Equation (4), one could also write the dynamic programming equations for any policy $\pi \in \Pi^R$ , namely

$$
V _ {2 h} ^ {\pi} (s, \pmb {R} | \mathcal {M} ^ {R}) = R (\pi_ {h} (s, \pmb {R})) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, \pi_ {h} (s, \pmb {R})) V _ {2 h + 1} ^ {\pi} (s ^ {\prime}, \pmb {0} | \mathcal {M} ^ {R}), \quad \forall h \in [ H ], s \in \mathcal {S}, \pmb {R} \in \mathbb {R} ^ {A},
$$

$$
V _ {2 h - 1} ^ {\pi} (s, \mathbf {0} | \mathcal {M} ^ {R}) = \mathbb {E} _ {\mathcal {R} _ {h} (s)} \big [ V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} ^ {R}) \big ], \quad \forall h \in [ H ], s \in \mathcal {S}.
$$

In particular, following the notation of Equation (3), one can also write

$$
V _ {h} ^ {R, \pi} (s, \boldsymbol {R} | \mathcal {M}) = R (\pi_ {h} (s, \boldsymbol {R})) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, \pi_ {h} (s, \boldsymbol {R})) V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M}), \quad a n d,
$$

$$
\begin{array}{l} V _ {h} ^ {R, \pi} (s | \mathcal {M}) = \mathbb {E} _ {\mathcal {R} _ {h} (s)} \Big [ V _ {h} ^ {R, \pi} (s, \boldsymbol {R} | \mathcal {M}) \Big ] \\ = \mathbb {E} _ {\mathcal {R} _ {h} (s)} \left[ R (\pi_ {h} (s, \boldsymbol {R})) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, \pi_ {h} (s, \boldsymbol {R})) V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M})) \right]. \\ \end{array}
$$

We will use this notation in some of the proofs.

Another useful application of the extended MDP is a variation of the law of total variance (LTV), which will be useful in our analysis

Lemma 3. For any deterministic one-step reward lookahead policy $\pi \in \Pi^{R}$ , it holds that

$$
\mathbb {E} \left[ \sum_ {h = 1} ^ {H} \operatorname{Var} _ {P _ {h} (\cdot | s _ {h}, a _ {h})} (V _ {h + 1} ^ {R, \pi} (s _ {h + 1})) | \pi , s _ {1} \right] \leq \mathbb {E} \left[ \left(\sum_ {h = 1} ^ {H} R _ {h} (s _ {h}, a _ {h}) - V _ {1} ^ {R, \pi} (s _ {1})\right) ^ {2} | \pi , s _ {1} \right].
$$

Proof. We apply the law of total variance (Lemma 27) in the extended MDP; there, the rewards are deterministic and equal to either 0 (at odd steps) or $R_{h}(s_{h},a_{h})$ (at even steps), so the total expected rewards are $\sum_{h = 1}^{H}R_{h}(s_{h},a_{h})$ .

$$
\mathbb {E} \left[ \left(\sum_ {h = 1} ^ {H} R _ {h} (s _ {h}, a _ {h}) - V _ {1} ^ {\pi} (s _ {1}, \mathbf {0} | \mathcal {M} ^ {R})\right) ^ {2} | \pi , s _ {1} \right]
$$

$$
= \mathbb {E} \left[ \underbrace {\sum_ {h = 1} ^ {H} \operatorname{Var} (V _ {2 h} ^ {\pi} (s _ {h} , \boldsymbol {R} _ {h} (s _ {h}) | \mathcal {M} ^ {R}) | (s _ {h} , \boldsymbol {0}))} _ {\text {Odd steps}} + \underbrace {\sum_ {h = 1} ^ {H} \operatorname{Var} (V _ {2 h + 1} ^ {\pi} (s _ {h + 1} , \boldsymbol {0} | \mathcal {M} ^ {R}) | (s _ {h} , \boldsymbol {R} _ {h} (s _ {h})))} _ {\text {Even steps}} | \pi , s _ {1} \right]
$$

$$
\geq \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \operatorname{Var} (V _ {2 h + 1} ^ {\pi} (s _ {h + 1}, \mathbf {0} | \mathcal {M} ^ {R}) | (s _ {h}, \boldsymbol {R} _ {h} (s _ {h}))) | \pi , s _ {1} \right]
$$

$$
= \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \operatorname{Var} _ {P _ {h} (\cdot | s _ {h}, a _ {h})} (V _ {2 h + 1} ^ {\pi} (s _ {h + 1}, \mathbf {0} | \mathcal {M} ^ {R})) | \pi , s _ {1} \right]
$$

$$
= \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \operatorname{Var} _ {P _ {h} (\cdot | s _ {h}, a _ {h})} (V _ {h + 1} ^ {R, \pi} (s _ {h + 1} | \mathcal {M})) | \pi , s _ {1} \right].
$$

Noting that $V_{1}^{\pi}(s_{1},\mathbf{0}|\mathcal{M}^{R}) = V_{1}^{R,\pi}(s_{1}|\mathcal{M})$ concludes the proof.

Finally, though not needed in our analysis, we use the extended MDP to prove the following value-difference lemma, which could be of further use in follow-up works. While we prove decomposition just using the next-step values, one could recursively apply the formula until the end of the episode to immediately get another formula that does not depend on the next value.

Lemma 4 (Value-Difference Lemma with Reward Lookahead). Let $\mathcal{M}_1 = (\mathcal{S},\mathcal{A},H,P^1,\mathcal{R}^1)$ and $\mathcal{M}_2 = (\mathcal{S},\mathcal{A},H,P^2,\mathcal{R}^2)$ be two environments. For any deterministic one-step reward lookahead policy $\pi \in \Pi^R$ , any $h\in [H]$ and $s\in \mathcal{S}$ , it holds that

$$
\begin{array}{l} V _ {h} ^ {R, \pi} (s | \mathcal {M} _ {1}) - V _ {h} ^ {R, \pi} (s | \mathcal {M} _ {2}) \\ = \mathbb {E} _ {\mathcal {M} _ {1}} \Big [ V _ {h + 1} ^ {R, \pi} (s _ {h + 1} | \mathcal {M} _ {1}) - V _ {h + 1} ^ {R, \pi} (s _ {h + 1} | \mathcal {M} _ {2}) | s _ {h} = s \Big ] \\ + \mathbb {E} _ {\mathcal {M} _ {1}} \left[ \sum_ {s ^ {\prime} \in \mathcal {S}} \left(P _ {h} ^ {1} (s ^ {\prime} | s _ {h}, \pi_ {h} (s _ {h}, \boldsymbol {R} _ {h})) - P _ {h} ^ {2} (s ^ {\prime} | s _ {h}, \pi_ {h} (s _ {h}, \boldsymbol {R} _ {h}))\right) V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M} _ {2}) | s _ {h} = s \right] \\ + \mathbb {E} _ {\mathcal {M} _ {1}} \Big [ \mathbb {E} _ {\mathcal {R} _ {h} ^ {1} (s)} \Big [ V _ {h} ^ {R, \pi} (s _ {h}, \pmb {R} | \mathcal {M} _ {2}) \Big ] - \mathbb {E} _ {\mathcal {R} _ {h} ^ {2} (s)} \Big [ V _ {h} ^ {R, \pi} (s _ {h}, \pmb {R} | \mathcal {M} _ {2}) \Big ] | s _ {h} = s \Big ], \\ \end{array}
$$

where $V_{h}^{R,\pi}(s, \mathbf{R}|\mathcal{M})$ is the value at a state given the reward realization, defined in Equation (3) and given in Remark 1.

Proof. We again work with the extended MDPs $M_{1}^{R}, M_{2}^{R}$ . Since under the extension, both the environments and the policy are Markovian, all values obey the following Bellman equations:

$$
V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} ^ {R}) = R (\pi_ {h} (s, \boldsymbol {R})) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} (s ^ {\prime} | s, \pi (s, \boldsymbol {R})) V _ {2 h + 1} ^ {\pi} (s ^ {\prime}, \mathbf {0} | \mathcal {M} ^ {R}), \quad \forall h \in [ H ], s \in \mathcal {S}, \boldsymbol {R} \in \mathbb {R} ^ {A}
$$

$$
V _ {2 h - 1} ^ {\pi} (s, \mathbf {0} | \mathcal {M} ^ {R}) = \mathbb {E} _ {\mathcal {R} _ {h} (s)} \big [ V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} ^ {R}) \big ], \quad \forall h \in [ H ], s \in \mathcal {S}.
$$

Using the relation between the value of the original and extended MDP (eq. (3)) and the Bellman equations of the extended MDP, for any $h \in [H]$ , we have

$$
\begin{array}{l} V _ {h} ^ {R, \pi} (s | \mathcal {M} _ {1}) - V _ {h} ^ {R, \pi} (s | \mathcal {M} _ {2}) \\ = V _ {2 h - 1} ^ {\pi} (s, \mathbf {0} | \mathcal {M} _ {1} ^ {R}) - V _ {2 h - 1} ^ {\pi} (s, \mathbf {0} | \mathcal {M} _ {2} ^ {R}) \\ = \mathbb {E} _ {\mathcal {R} _ {h} ^ {1} (s)} \left[ V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} _ {1} ^ {R}) \right] - \mathbb {E} _ {\mathcal {R} _ {h} ^ {2} (s)} \left[ V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} _ {2} ^ {R}) \right] \\ = \mathbb {E} _ {\mathcal {R} _ {h} ^ {1} (s)} \big [ V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} _ {1} ^ {R}) - V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} _ {2} ^ {R}) \big ] + \mathbb {E} _ {\mathcal {R} _ {h} ^ {1} (s)} \big [ V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} _ {2} ^ {R}) \big ] - \mathbb {E} _ {\mathcal {R} _ {h} ^ {2} (s)} \big [ V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} _ {2} ^ {R}) \big ] \\ = \mathbb {E} _ {\mathcal {R} _ {h} ^ {1} (s)} \left[ V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} _ {1} ^ {R}) - V _ {2 h} ^ {\pi} (s, \boldsymbol {R} | \mathcal {M} _ {2} ^ {R}) \right] + \mathbb {E} _ {\mathcal {R} _ {h} ^ {1} (s)} \left[ V _ {h} ^ {R, \pi} (s, \boldsymbol {R} | \mathcal {M} _ {2}) \right] - \mathbb {E} _ {\mathcal {R} _ {h} ^ {2} (s)} \left[ V _ {h} ^ {R, \pi} (s, \boldsymbol {R} | \mathcal {M} _ {2}) \right] \\ = \mathbb {E} _ {\mathcal {M} _ {1}} \big [ V _ {2 h} ^ {\pi} (s _ {h}, \boldsymbol {R} _ {h} | \mathcal {M} _ {1} ^ {R}) - V _ {2 h} ^ {\pi} (s _ {h}, \boldsymbol {R} _ {h} | \mathcal {M} _ {2} ^ {R}) | s _ {h} = s \big ] \\ + \mathbb {E} _ {\mathcal {R} _ {h} ^ {1} (s)} \left[ V _ {h} ^ {R, \pi} (s, \boldsymbol {R} | \mathcal {M} _ {2}) \right] - \mathbb {E} _ {\mathcal {R} _ {h} ^ {2} (s)} \left[ V _ {h} ^ {R, \pi} (s, \boldsymbol {R} | \mathcal {M} _ {2}) \right]. \tag {5} \\ \end{array}
$$

We now focus on the first term. Denoting $a_h = \pi_h(s_h, \mathbf{R}_h)$ the action taken by the agent at environment $\mathcal{M}_1$ , We have

$$
\begin{array}{l} V _ {2 h} ^ {\pi} (s _ {h}, \boldsymbol {R} _ {h} | \mathcal {M} _ {1} ^ {R}) - V _ {2 h} ^ {\pi} (s _ {h}, \boldsymbol {R} _ {h} | \mathcal {M} _ {2} ^ {R}) \\ = \left(R _ {h} (a _ {h}) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} ^ {1} (s ^ {\prime} | s _ {h}, a _ {h}) V _ {2 h + 1} ^ {\pi} (s ^ {\prime}, \mathbf {0} | \mathcal {M} _ {1} ^ {R})\right) \\ - \left(R _ {h} (a _ {h}) + \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} ^ {2} (s ^ {\prime} | s _ {h}, a _ {h}) V _ {2 h + 1} ^ {\pi} (s ^ {\prime}, \mathbf {0} | \mathcal {M} _ {2} ^ {R})\right) \\ = \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} ^ {1} (s ^ {\prime} | s _ {h}, a _ {h}) V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M} _ {1}) - \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} ^ {2} (s ^ {\prime} | s _ {h}, a _ {h}) V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M} _ {2}) \\ = \sum_ {s ^ {\prime} \in \mathcal {S}} P _ {h} ^ {1} (s ^ {\prime} | s _ {h}, a _ {h}) \Big (V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M} _ {1}) - V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M} _ {2}) \Big) \\ + \sum_ {s ^ {\prime} \in \mathcal {S}} \bigl (P _ {h} ^ {1} (s ^ {\prime} | s _ {h}, a _ {h}) - P _ {h} ^ {2} (s ^ {\prime} | s _ {h}, a _ {h}) \bigr) V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M} _ {2}) \\ = E _ {\mathcal {M} _ {1}} \left[ V _ {h + 1} ^ {R, \pi} (s _ {h + 1} | \mathcal {M} _ {1}) - V _ {h + 1} ^ {R, \pi} (s _ {h + 1} | \mathcal {M} _ {2}) | s _ {h}, a _ {h} \right] \\ + \sum_ {s ^ {\prime} \in \mathcal {S}} \bigl (P _ {h} ^ {1} (s ^ {\prime} | s _ {h}, a _ {h}) - P _ {h} ^ {2} (s ^ {\prime} | s _ {h}, a _ {h}) \bigr) V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M} _ {2}). \\ \end{array}
$$

Substituting this back into Equation (5), we have

$$
\begin{array}{l} V _ {h} ^ {\pi} (s | \mathcal {M} _ {1}) - V _ {h} ^ {\pi} (s | \mathcal {M} _ {2}) \\ = \mathbb {E} _ {\mathcal {M} _ {1}} \left[ E _ {\mathcal {M} _ {1}} \left[ V _ {h + 1} ^ {R, \pi} (s _ {h + 1} | \mathcal {M} _ {1}) - V _ {h + 1} ^ {R, \pi} (s _ {h + 1} | \mathcal {M} _ {2}) | s _ {h}, a _ {h} \right] | s _ {h} = s \right] \\ + \mathbb {E} _ {\mathcal {M} _ {1}} \left[ \sum_ {s ^ {\prime} \in \mathcal {S}} \left(P _ {h} ^ {1} (s ^ {\prime} | s _ {h}, a _ {h}) - P _ {h} ^ {2} (s ^ {\prime} | s _ {h}, a _ {h})\right) V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M} _ {2}) | s _ {h} = s \right] \\ + \mathbb {E} _ {\mathcal {R} _ {h} ^ {1} (s)} \left[ V _ {h} ^ {R, \pi} (s, \boldsymbol {R} | \mathcal {M} _ {2}) \right] - \mathbb {E} _ {\mathcal {R} _ {h} ^ {2} (s)} \left[ V _ {h} ^ {R, \pi} (s, \boldsymbol {R} | \mathcal {M} _ {2}) \right] \\ = \mathbb {E} _ {\mathcal {M} _ {1}} \Big [ V _ {h + 1} ^ {R, \pi} (s _ {h + 1} | \mathcal {M} _ {1}) - V _ {h + 1} ^ {R, \pi} (s _ {h + 1} | \mathcal {M} _ {2}) | s _ {h} = s \Big ] \\ + \mathbb {E} _ {\mathcal {M} _ {1}} \left[ \sum_ {s ^ {\prime} \in \mathcal {S}} \left(P _ {h} ^ {1} (s ^ {\prime} | s _ {h}, \pi_ {h} (s _ {h}, \boldsymbol {R} _ {h})) - P _ {h} ^ {2} (s ^ {\prime} | s _ {h}, \pi_ {h} (s _ {h}, \boldsymbol {R} _ {h}))\right) V _ {h + 1} ^ {R, \pi} (s ^ {\prime} | \mathcal {M} _ {2}) | s _ {h} = s \right] \\ + \mathbb {E} _ {\mathcal {M} _ {1}} \left[ \mathbb {E} _ {\mathcal {R} _ {h} ^ {1} (s)} \left[ V _ {h} ^ {R, \pi} \left(s _ {h}, \boldsymbol {R} \mid \mathcal {M} _ {2}\right) \right] - \mathbb {E} _ {\mathcal {R} _ {h} ^ {2} (s)} \left[ V _ {h} ^ {R, \pi} \left(s _ {h}, \boldsymbol {R} \mid \mathcal {M} _ {2}\right) \right] \mid s _ {h} = s \right]. \\ \end{array}
$$

![](images/3b4690a938e35ffb3b06d3295cac1df66fca98fb5298c9bd93261084c42a2efd.jpg)

Algorithm 3 Monotonic Value Propagation with Reward Lookahead (MVP-RL)   
1: Require: $\delta \in (0,1)$ , bonuses $b_{k,h}^{r}(s), b_{k,h}^{p}(s,a)$ 2: for $k = 1,2,\ldots$ do
3: Initialize $\bar{V}_{H+1}^{k}(s) = 0$ 4: for $h = H, H - 1,..,1$ do
5:    for $s \in S$ do
6:    if $n_{h}^{k-1}(s) = 0$ then
7: $\bar{V}_{h}^{k}(s) = H$ 8:    else
9:    Calculate the truncated values $\bar{V}_{h}^{k}(s) = \min\left\{\frac{1}{n_{h}^{k-1}(s)} \sum_{t=1}^{n_{h}^{k-1}(s)} \max_{a \in \mathcal{A}} \left\{ R_{h}^{k_{h}^{t}(s)}(s,a) + b_{k,h}^{p}(s,a) + \hat{P}_{h}^{k-1} \bar{V}_{h+1}^{k}(s,a) \right\} + b_{k,h}^{r}(s), H \right\}$ 10:    end if
11:    For any vector $\boldsymbol{R} \in \mathbb{R}^{A}$ , define the policy $\pi^{k}$ $\pi_{h}^{k}(s, \boldsymbol{R}) \in \arg\max_{a \in \mathcal{A}} \left\{ R(a) + b_{k,h}^{p}(s,a) + \hat{P}_{h}^{k-1} \bar{V}_{h+1}^{k}(s,a) \right\}$ 12:    end for
13:    end for
14:    for $h = 1,2,\ldots H$ do
15:    Observe $s_{h}^{k}$ and $\boldsymbol{R}_{h}^{k} = \left\{ R_{h}^{k}(s_{h}^{k}, a) \right\}_{a \in \mathcal{A}}$ 16:    Play an action $a_{h}^{k} = \pi_{h}^{k}(s_{h}^{k}, \boldsymbol{R}_{h}^{k})$ 17:    Collect the reward $R_{h}^{k}(s_{h}^{k}, a_{h}^{k})$ and transition to the next state $s_{h+1}^{k} \sim P_{h}(\cdot | s_{h}^{k}, a_{h}^{k})$ 18:    end for
19:    Update the empirical estimators and counts for all visited state-actions
20: end for

We use a variant of the MVP algorithm [Zhang et al., 2021b] while adapting their proof and the one from [Efroni et al., 2021]. The algorithm is described in Algorithm 3 and uses the following bonuses:

$$
b _ {k, h} ^ {r} (s) = 3 \sqrt {\frac {A L _ {\delta} ^ {k}}{2 (n _ {h} ^ {k - 1} (s) \vee 1)}},
$$

$$
b _ {k, h} ^ {p} (s, a) = \min \left\{\frac {2 0}{3} \sqrt {\frac {\mathrm{Var} _ {\hat {P} _ {h} ^ {k - 1} (\cdot | s , a)} (\bar {V} _ {h + 1} ^ {k}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {4 0 0}{9} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}, H \right\}
$$

where $L_{\delta}^{k} = \ln \frac{144S^{2}AH^{2}k^{3}(k+1)}{\delta}$ , and for brevity, we shorten $\operatorname{Var}_{\hat{P}_{h}^{k-1}(\cdot|s,a)}(\bar{V}_{h+1}^{k}(s'))$ to $\operatorname{Var}_{\hat{P}_{h}^{k-1}(\cdot|s,a)}(\bar{V}_{h+1}^{k})$ (omitting the state from the value).

For the optimistic value iteration, we use the notation $k_{h}^{t}(s)$ to represent the $t^{th}$ episode where the state s was visited at the $h^{th}$ timestep. Thus, line 9 of Algorithm 3 is the expectation w.r.t. the empirical reward distribution $\hat{\mathcal{R}}_{h}^{k-1}(s)$ (when defining its realization to be zero when $n_{h}^{k-1}(s) = 0$ ). Since the bonuses are larger than H when $n_{h}^{k-1}(s) = 0$ , one could write the update in more concisely as

$$
\bar {V} _ {h} ^ {k} (s) = \min \Biggl \{\mathbb {E} _ {\boldsymbol {R} \sim \hat {\mathcal {R}} _ {h} ^ {k - 1} (s)} \biggl [ \max _ {a \in \mathcal {A}} \Bigl \{R (a) + b _ {k, h} ^ {p} (s, a) + \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, a) \Bigr \} \biggr ] + b _ {k, h} ^ {r} (s), H \Biggr \}.
$$

We will often use this representation in our analysis.

# B.4 The First Good Event – Concentration

We now define the first good event, which ensures that all empirical quantities are well-concentrated. For the transitions, we require each element to concentrate well, as well as both the inner product and the variance w.r.t. the optimal value function. For the reward, we make sure that the maximum of the rewards to concentrate well (with any possible bias, that will later correspond with the next-state values). Formally, for any fixed vector $u \in R^{A}$ , denote

$$
m _ {h} (s, u) = \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \Big [ \max _ {a} \{R _ {h} (a) + u (a) \} \Big ],
$$

$$
\hat {m} _ {h} ^ {k} (s, u) = \mathbb {E} _ {\boldsymbol {R} \sim \hat {\mathcal {R}} _ {h} ^ {k} (s)} \Big [ \max _ {a} \{R _ {h} (a) + u (a) \} \Big ]
$$

with the convention that $\hat{m}_h^k (s,u) = \max_a u(a)$ if $n_h^k (s) = 0$ . We define the following good events:

$$
E ^ {p} (k) = \left\{\forall s, s ^ {\prime}, a, h: | P _ {h} (s ^ {\prime} | s, a) - \hat {P} _ {h} ^ {k - 1} (s ^ {\prime} | s, a) | \leq \sqrt {\frac {2 P (s ^ {\prime} | s , a) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \right\}
$$

$$
E ^ {p v 1} (k) = \left\{\forall s, a, h: \Big | \Big (\hat {P} _ {h} ^ {k - 1} - P _ {h} \Big) V _ {h + 1} ^ {*} (s, a) \Big | \leq \sqrt {\frac {2 \operatorname{Var} _ {P _ {h} (\cdot | s , a)} (V _ {h + 1} ^ {*}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \right\}
$$

$$
E ^ {p v 2} (k) = \left\{\forall s, a, h: \left| \sqrt {\operatorname{Var} _ {P _ {h} (\cdot | s , a)} (V _ {h + 1} ^ {*})} - \sqrt {\operatorname{Var} _ {\hat {P} _ {h} ^ {k - 1} (\cdot | s , a)} (V _ {h + 1} ^ {*})} \right| \leq 4 H \sqrt {\frac {L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} \right\}
$$

$$
E ^ {r} (k) = \left\{\forall s, h, \forall u \in [ 0, 2 H ] ^ {A}: \left| m _ {h} (s, u) - \hat {m} _ {h} ^ {k - 1} (s, u) \right| \leq 3 \sqrt {\frac {A L _ {\delta} ^ {k}}{2 (n _ {h} ^ {k - 1} (s) \vee 1)}} \right\}
$$

where we again use $L_{\delta}^{k} = \ln \frac{144S^{2}AH^{2}k^{3}(k + 1)}{\delta}$ . Then, we define the first good event as

$$
\mathbb {G} _ {1} = \bigcap_ {k \geq 1} E ^ {r} (k) \bigcap_ {k \geq 1} E ^ {p} (k) \bigcap_ {k \geq 1} E ^ {p v 1} (k) \bigcap_ {k \geq 1} E ^ {p v 2} (k),
$$

for which, the following holds:

Lemma 5 (The First Good Event). The good event $\mathbb{G}_1$ holds w.p. $\Pr (\mathbb{G}_1)\geq 1 - \delta /2$ .

Proof. The proof of the first three events uses standard concentration arguments (see, e.g., Efroni et al. 2021) and is stated for completeness. For any fixed $k \geq 1, s, a, h$ and number of visits $n \in [k]$ , we utilize Lemma 16 w.r.t. the transition kernel $P_h(\cdot | s, a)$ , the value $V_{h+1}^* \in [0, H]$ and probability $\delta' = \frac{\delta}{8SAHk^2(k+1)}$ ; notice that by the assumption that samples are generated i.i.d. before the game starts, given the number of visits, all samples are i.i.d., so standard concentration could be applied. By taking the union bound over all $n \in [k]$ and slightly increasing the constants to ensure that $n = 0$ trivially holds, we get that the events also hold for any number of visit $n_h^{k-1}(s, a) \in \{0 \dots, k\}$ , and taking another union bound over all $k \geq 1, s, a, h$ ensures that each of the events $\cap_{k \geq 1} E^p(k), \cap_{k \geq 1} E^{pv1}(k)$ and $\cap_{k \geq 1} E^{pv2}(k)$ holds w.p. at least $1 - \frac{\delta}{8}$

We now focus on bounding the probability of the event $\cap_{k}E^{r}(k)$ . For any fixed k, h and s, observe that the event trivially holds if $n_{h}^{k}=0$ , then the event trivially holds, since for all $u\in[0,2H]^{A}$ ,

$$
\left| m _ {h} (s, u) - \hat {m} _ {h} ^ {k - 1} (s, u) \right| = \left| \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \left[ \max _ {a} \{R _ {h} (s, a) + u (a) \} \right] - \max _ {a} \{u (a) \} \right| \stackrel {(*)} {\leq} 1 \leq 3 \sqrt {\frac {A L _ {\delta} ^ {k}}{2}},
$$

where (\*) uses the boundedness of the rewards in $[0,1]$ . Next, recall that for any fixed $n_{h}^{k-1}=n\in[k]$ , the rewards samples at state s and step h are i.i.d. vectors on $[0,1]^{A}$ . Therefore, by Lemma 18,

$$
\operatorname * {P r} \left\{n _ {h} ^ {k - 1} (s) = n, \forall u \in [ 0, 2 H ] ^ {A}: \left| m _ {h} (s, u) - \hat {m} _ {h} ^ {k - 1} (s, u) \right| > 3 \sqrt {\frac {A L _ {\delta} ^ {k}}{2 \left(n _ {h} ^ {k - 1} (s) \vee 1\right)}} \right\} \leq \frac {\delta}{8 S A H k ^ {2} (k + 1)}.
$$

Taking a union bound on all possible values of $n \in [k]$ , $s$ and $h$ , we get

$$
\operatorname * {P r} \left\{E ^ {r} (k) \right\} \geq 1 - S A k \cdot \frac {\delta}{8 S A H k ^ {2} (k + 1)} \geq 1 - \frac {\delta}{8 k (k + 1)}.
$$

By summing over all $k \geq 1$ , the event $\cap_{k} E^{r}(k)$ holds with a probability of at least $1 - \delta/8$ . Finally, taking the union bound with the other three events leads to the desired result of $\Pr(\mathbb{G}_{1}) \geq 1 - \delta/2$ . ☐

# B.5 Optimism of the Upper Confidence Value Functions

In this subsection, we prove that under the good event $\mathbb{G}_1$ , the values $\bar{V}^k$ that MVP-RL produces are optimistic.

Lemma 6 (Optimism). Under the first good event $\mathbb{G}_1$ , for all $k \in [K]$ , $h \in [H]$ and $s \in S$ , it holds that $V_h^*(s) \leq \bar{V}_h^k(s)$ .

Proof. The proof follows by backward induction on H; see that the claim trivially holds for $h = H + 1$ , where both values are defined to be zero.

Now assume by induction that for some $k \in [K]$ and $h \in [H]$ , the desired inequalities hold at timestep $h + 1$ for all $s \in S$ ; we will show that this implies that they also hold at timestep h.

At this point, we also assume w.l.o.g. that $\bar{V}_h^k (s) < H$ , and in particular, the value is not truncated; otherwise, by the boundedness of the rewards, $V_{h}^{*}(s)\leq H = \bar{V}_{h}^{k}(s)$ . For similar reasons, we assume w.l.o.g. that $b_{k,h}^{p}(s,a) < H$ , so that it is also not truncated.

By the optimism of the value at step $h + 1$ due to the induction hypothesis and the monotonicity of the bonus (Lemma 23), under the good event, we have for all $s \in S$ and $a \in A$ that

$$
\begin{array}{l} \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, a) + b _ {k, h} ^ {p} (s, a) \\ \geq \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, a) + \max \left\{\frac {2 0}{3} \sqrt {\frac {\operatorname{Var} _ {\hat {P} _ {h} ^ {k - 1} (\cdot | s , a)} (\bar {V} _ {h + 1} ^ {k}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}}, \frac {4 0 0}{9} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \right\} \\ \geq \hat {P} _ {h} ^ {k - 1} V _ {h + 1} ^ {*} (s, a) + \max \left\{\frac {2 0}{3} \sqrt {\frac {\operatorname{Var} _ {\hat {P} _ {h} ^ {k - 1} (\cdot | s , a)} \left(V _ {h + 1} ^ {*}\right) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}}, \frac {4 0 0}{9} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \right\} \tag {Lemma23} \\ \geq \hat {P} _ {h} ^ {k - 1} V _ {h + 1} ^ {*} (s, a) + \frac {1 0}{3} \sqrt {\frac {\operatorname{Var} _ {\hat {P} _ {h} ^ {k - 1} (\cdot | s , a)} (V _ {h + 1} ^ {*}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {2 0 0}{9} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \\ \geq \hat {P} _ {h} ^ {k - 1} V _ {h + 1} ^ {*} (s, a) + \frac {1 0}{3} \sqrt {\frac {\operatorname{Var} _ {P _ {h} (\cdot | s , a)} \left(V _ {h + 1} ^ {*}\right) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {8 H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \quad (\text { Under } E ^ {p v 2} (k)) \\ \geq P _ {h} V _ {h + 1} ^ {*} (s, a). \quad (\text { Under   } E ^ {p v 1} (k)) \\ \end{array}
$$

Thus, under the good event and the induction hypothesis, we have that

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s) = \mathbb {E} _ {\boldsymbol {\mathbf {R}} \sim \hat {\mathcal {R}} _ {h} (s)} \bigg [ \max _ {a \in \mathcal {A}} \Bigl \{R (a) + b _ {k, h} ^ {p} (s, a) + \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, a) \Bigr \} \bigg ] + b _ {k, h} ^ {r} (s) \\ \geq \mathbb {E} _ {\boldsymbol {R} \sim \hat {\mathcal {R}} _ {h} (s)} \left[ \max _ {a \in \mathcal {A}} \bigl \{R (a) + P _ {h} V _ {h + 1} ^ {*} (s, a) \bigr \} \right] + b _ {k, h} ^ {r} (s). \\ \end{array}
$$

In particular, using Proposition 1, we get

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s) - V _ {h} ^ {*} (s) \geq \mathbb {E} _ {\boldsymbol {\mathbf {R}} \sim \hat {\mathcal {R}} _ {h} (s)} \biggl [ \max _ {a \in \mathcal {A}} \bigl \{R (a) + P _ {h} V _ {h + 1} ^ {*} (s, a) \bigr \} \biggr ] + b _ {k, h} ^ {r} (s) \\ - \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \bigg [ \max _ {a \in \mathcal {A}} \bigl \{R (a) + P _ {h} V _ {h + 1} ^ {*} (s, a) \bigr \} \bigg ] \\ \geq 0, \\ \end{array}
$$

where the last inequality holds under the event $E^{r}(k)$ with $u(a)=P_{h}V_{h+1}^{*}(s,a)\in[0,H]^{A}$ .

![](images/886ca69de354fe5a292015699f9c311b7dfbb2b64dcbb1e037831419b6b9dd33.jpg)

# B.6 The Second Good Event – Martingale Concentration

In this subsection, we present four good events that will allow us to replace the expectation over the randomizations inside each episode with their realization.

Define the following bonus-like term that will later appear in the proof due to value concentration:

$$
b _ {k, h} ^ {p v 1} (s, a) = \min \Bigg \{\sqrt {\frac {2 \mathrm{Var} _ {P _ {h} (\cdot | s , a)} (V _ {h + 1} ^ {*}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {4 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}, H \Bigg \},
$$

and let

$$
Y _ {1, h} ^ {k} := \bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k}),
$$

$$
Y _ {2, h} ^ {k} = \mathrm{Var} _ {P _ {h} (\cdot | s _ {t, h}, a _ {t, h})} (V _ {h + 1} ^ {\pi^ {k}}),
$$

$$
Y _ {3, h} ^ {k} = b _ {k, h} ^ {p} (s _ {h} ^ {k}, a _ {h} ^ {k}) + b _ {k, h} ^ {p v 1} (s _ {h} ^ {k}, a _ {h} ^ {k}).
$$

The second good event is the intersection of the events $\mathbb{G}_2 = E^{\mathrm{diff1}}\cap E^{\mathrm{diff2}}\cap E^{\mathrm{Var}}\cap E^{bp}$ defined as follows.

$$
E ^ {\text {diff1}} = \left\{\forall h \in [ H ], K \geq 1: \sum_ {k = 1} ^ {K} \mathbb {E} [ Y _ {1, h} ^ {k} | F _ {k, h - 1} ] \leq \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} Y _ {1, h} ^ {k} + 1 8 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta} \right\},
$$

$$
E ^ {\mathrm{diff2}} = \left\{\forall h \in [ H ], K \geq 1: \sum_ {k = 1} ^ {K} \mathbb {E} [ Y _ {1, h} ^ {k} | F _ {k, h - 1} ^ {R} ] \leq \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} Y _ {1, h} ^ {k} + 1 8 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta} \right\},
$$

$$
E ^ {\mathrm{Var}} = \Bigg \{K \geq 1: \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} Y _ {2, h} ^ {k} \leq 2 \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \mathbb {E} [ Y _ {2, h} ^ {k} | F _ {k - 1} ] + 4 H ^ {3} \ln \frac {8 H K (K + 1)}{\delta} \Bigg \},
$$

$$
E ^ {b p} = \left\{\forall h \in [ H ], K \geq 1: \sum_ {k = 1} ^ {K} \mathbb {E} [ Y _ {3, h} ^ {k} | F _ {k, h - 1} ] \leq 2 \sum_ {k = 1} ^ {K} Y _ {3, h} ^ {k} + 5 0 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta} \right\},
$$

We define the good event $\mathbb{G} = \mathbb{G}_1\cap \mathbb{G}_2$

Lemma 7. The good event G holds with a probability of at least $1 - \delta$ .

Proof. The proof follows similarly to Lemmas 15 and 21 of [Efroni et al., 2021].

First, define the random process $W_{k} = \mathbb{1}\left\{\bar{V}_{h}^{k}(s) - V_{h}^{\pi^{k}}(s)\in [0,H],\forall h\in [H],s\in \mathcal{S}\right\}$ and define $\tilde{Y}_{1,h}^{k} = W_{k}Y_{1,h}^{k}$ , which is bounded in $[0,H]$ . Also observe that $W_{k}$ is $F_{k - 1}$ measurable, since both values and policies are calculated based on data up to the episode $k - 1$ , and in particular, it is $F_{k,h - 1}$ measurable and $\tilde{Y}_{1,h}^{k}$ is $F_{k,h}$ measurable. thus, by Lemma 25, for any $k\in [K]$ and $h\in [H]$ , we have w.p. at least $1 - \frac{\delta}{8HK(K + 1)}$ that

$$
\sum_ {k = 1} ^ {K} \mathbb {E} [ \tilde {Y} _ {1, h} ^ {k} | F _ {k, h - 1} ] \leq \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} \tilde {Y} _ {1, h} ^ {k} + 1 8 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta}.
$$

Since $W_{k}$ is $F_{k,h - 1}$ measurable, we can write the event as

$$
\sum_ {k = 1} ^ {K} W _ {k} \mathbb {E} [ Y _ {1, h} ^ {k} | F _ {k, h - 1} ] \leq \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} W _ {k} Y _ {1, h} ^ {k} + 1 8 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta},
$$

and taking the union bound over all $h \in [H]$ and $K \geq 1$ , we get w.p. at least $1 - \frac{\delta}{8}$ that the event

$$
\tilde {E} ^ {\text {diff1}} = \left\{\forall h \in [ H ], K \geq 1: \sum_ {k = 1} ^ {K} W _ {k} \mathbb {E} [ Y _ {1, h} ^ {k} | F _ {k, h - 1} ] \leq \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} W _ {k} Y _ {1, h} ^ {k} + 1 8 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta} \right\}.
$$

Importantly, by optimism (Lemma 6), under $\mathbb{G}_1$ , it holds that $W_{k} = 1$ for all $k \geq 1$ , so we immediately get that $\mathbb{G}_1 \cap \tilde{E}^{\mathrm{diff}1} = \mathbb{G}_1 \cap E^{\mathrm{diff}1}$ .

Following the exact same proof just with the filtration $F_{k,h}^{R}$ and defining the equivalent $\tilde{E}^{\mathrm{diff2}}$ , we get that this event also holds w.p. $1 - \frac{\delta}{8}$ and is the desired event when $\mathbb{G}_1$ holds.

Next, we prove that the other two events also hold w.p. at least $1 - \frac{\delta}{8}$ .

By the assumptions of our setting, we know that $V_{h}^{\pi^{k}}(s) \in [0,H]$ , and so

$$
\sum_ {h = 1} ^ {H} Y _ {2, h} ^ {k} = \sum_ {h = 1} ^ {H} \operatorname{Var} _ {P _ {h} (\cdot | s _ {t, h}, a _ {t, h})} (V _ {h + 1} ^ {\pi^ {k}}) \in [ 0, H ^ {3} ].
$$

In particular, applying Lemma 25 (w.r.t. the filtration $F_{k}$ ) with $C = H^{3}$ and any fixed $K$ , we get w.p. $1 - \frac{\delta}{8HK(K + 1)}$ that

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} Y _ {2, h} ^ {k} \leq 2 \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \mathbb {E} [ Y _ {2, h} ^ {k} | F _ {k - 1} ] + 4 H ^ {3} \ln \frac {8 H K (K + 1)}{\delta}.
$$

Taking the union bound on all possible values of $K \geq 1$ proves that $E^{Var}$ holds w.p. at least $1 - \frac{\delta}{8}$ .

Similarly, by definition, we have that $Y_{3,h}^{k} = b_{k,h}^{p}(s_{h}^{k},a_{h}^{k}) + b_{k,h}^{pv1}(s_{h}^{k},a_{h}^{k}) \in [0,2H]$ and is $F_{k,h}$ measurable. Thus, for any fixed $k \geq 1$ and $h \in [H]$ , using Lemma 25, we have w.p. $1 - \frac{\delta}{8HK(K + 1)}$ that

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \mathbb {E} [ Y _ {3, h} ^ {k} | F _ {k, h - 1} ] \leq \left(1 + \frac {1}{4 H}\right) \sum_ {k = 1} ^ {K} Y _ {3, h} ^ {k} + 5 0 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta} \\ \leq 2 \sum_ {k = 1} ^ {K} Y _ {3, h} ^ {k} + 5 0 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta}, \\ \end{array}
$$

applying the union bound on all $K \geq 1$ , the event $E^{bp}$ holds w.p. $1 - \frac{\delta}{8}$ .

To summarize, we have that the event $G_{1}$ holds w.p. $1 - \frac{\delta}{2}$ (Lemma 5), and we proved that the events $\tilde{E}^{diff1}, \tilde{E}^{diff2}, E^{Var}, E^{bp}$ hold each w.p. $1 - \frac{\delta}{8}$ , so we also have that the event

$$
\begin{array}{l} \mathbb {G} = \mathbb {G} _ {1} \cap \mathbb {G} _ {2} \\ = \mathbb {G} _ {1} \cap E ^ {\text { diff1 }} \cap E ^ {\text { diff2 }} \cap E ^ {\text { Var }} \cap E ^ {b p} \\ = \mathbb {G} _ {1} \cap \tilde {E} ^ {\text { diff1 }} \cap \tilde {E} ^ {\text { diff2 }} \cap E ^ {\text { Var }} \cap E ^ {b p} \\ \end{array}
$$

holds w.p. at least $1 - \delta$ .

![](images/1b7521abdf06cbd53bcc050555153f190d0ff2c344fcc948de9abaa5fd95e771.jpg)

# B.7 Regret Analysis

We finally analyze the regret of the algorithm

Theorem 1. When running MVP-RL, with probability at least $1 - \delta$ uniformly for all $K \geq 1$ , it holds that $\operatorname{Reg}^R(K) \leq \mathcal{O}\left(\sqrt{H^3 SAK} \ln \frac{SAHK}{\delta} + H^3 S^2 A\left(\ln \frac{SAHK}{\delta}\right)^2\right)$ .

Proof. Assume that the good events $\mathbb{G}$ holds, which by Lemma 7, happens with probability at least $1 - \delta$ . Then, by optimism (Lemma 6), for any $k \in [K]$ , $h \in [H]$ and $s \in S$ , it holds that $V_h^*(s) \leq \bar{V}_h^k(s)$ . Moreover, we can lower bound the value of the policy $\pi^k$ as follows (see Remark 1):

$$
\begin{array}{l} V _ {h} ^ {\pi^ {k}} (s) = \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \Big [ R (\pi_ {h} ^ {k} (s, \boldsymbol {R})) + P _ {h} V _ {h + 1} ^ {\pi^ {k}} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) \Big ] \\ = \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \Big [ R (\pi_ {h} ^ {k} (s, \boldsymbol {R})) + \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) + b _ {k, h} ^ {p} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) \Big ] \\ + \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \Big [ P _ {h} V _ {h + 1} ^ {\pi^ {k}} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) - \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) - b _ {k, h} ^ {p} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) \Big ] \\ \stackrel {(1)} {=} \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \left[ \max _ {a \in \mathcal {A}} \left\{R (a) + \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, a) + b _ {k, h} ^ {p} (s, a) \right\} \right] \\ + \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \left[ P _ {h} V _ {h + 1} ^ {\pi^ {k}} \left(s, \pi_ {h} ^ {k} (s, \boldsymbol {R})\right) - \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} \left(s, \pi_ {h} ^ {k} (s, \boldsymbol {R})\right) - b _ {k, h} ^ {p} \left(s, \pi_ {h} ^ {k} (s, \boldsymbol {R})\right) \right] \\ \stackrel {(2)} {\geq} \mathbb {E} _ {\boldsymbol {R} \sim \hat {\mathcal {R}} _ {h} ^ {k - 1} (s)} \left[ \max _ {a \in \mathcal {A}} \left\{R (a) + \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, a) + b _ {k, h} ^ {p} (s, a) \right\} \right] - b _ {k, h} ^ {r} (s) \\ + \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \Big [ P _ {h} V _ {h + 1} ^ {\pi^ {k}} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) - \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) - b _ {k, h} ^ {p} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) \Big ] \\ \stackrel {(3)} {\geq} \bar {V} _ {h} ^ {k} (s) - 2 b _ {k, h} ^ {r} (s) \\ + \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \left[ P _ {h} V _ {h + 1} ^ {\pi^ {k}} \left(s, \pi_ {h} ^ {k} (s, \boldsymbol {R})\right) - \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} \left(s, \pi_ {h} ^ {k} (s, \boldsymbol {R})\right) - b _ {k, h} ^ {p} \left(s, \pi_ {h} ^ {k} (s, \boldsymbol {R})\right) \right]. \tag {6} \\ \end{array}
$$

Relation (1) is by the definition of $\pi^{k}$ (see Algorithm 3), while (2) holds under the good event $E^{r}(k)$ with $u(a)=\hat{P}_{h}^{k-1}\bar{V}_{h+1}^{k}(s,a)+b_{k,h}^{p}(s,a)\in[0,2H]$ (due to the value and bonus truncation). Finally, (3) is by the definition of $\bar{V}_{h}^{k}(s)$ , where the inequality also accounts for its possible truncation.

To further bound this, we need to bound

$$
\begin{array}{l} \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, a) - P _ {h} V _ {h + 1} ^ {\pi^ {k}} (s, a) = P _ {h} \Big (\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}} \Big) (s, a) + \Big (\hat {P} _ {h} ^ {k - 1} - P _ {h} \Big) \bar {V} _ {h + 1} ^ {k} (s, a) \\ = P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) \\ + \Big (\hat {P} _ {h} ^ {k - 1} - P _ {h} \Big) V _ {h + 1} ^ {*} (s, a) + \Big (\hat {P} _ {h} ^ {k - 1} - P _ {h} \Big) \big (\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {*} \big) (s, a). \\ \end{array}
$$

The first error term can be bounded under the good event, while the second using Lemma 24. More formally, under the good event $E^{pv1}(k)$ , we have

$$
\Big | \Big (\hat {P} _ {h} ^ {k - 1} - P _ {h} \Big) V _ {h + 1} ^ {*} (s, a) \Big | \leq \sqrt {\frac {2 \mathrm{Var} _ {P _ {h} (\cdot | s , a)} (V _ {h + 1} ^ {*}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1},
$$

and by Lemma 24 with $\alpha = 4H$ (using and $P_{1} = P_{h}$ , $P_{2} = \hat{P}_{h}^{k - 1}$ , under $E^{p}(k)$ ),

$$
\begin{array}{l} \Big | \Big (\hat {P} _ {h} ^ {k - 1} - P _ {h} \Big) \big (\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {*} \big) (s, a) \Big | \leq \frac {1}{4 H} \mathbb {E} _ {P _ {h} (\cdot | s, a)} \big [ \bar {V} _ {h + 1} ^ {k} (s ^ {\prime}) - V _ {h + 1} ^ {*} (s ^ {\prime}) \big ] + \frac {H S L _ {\delta} ^ {k} (1 + 4 H \cdot 2 / 4)}{n _ {h} ^ {k - 1} (s , a) \vee 1} \\ \leq \frac {1}{4 H} \mathbb {E} _ {P _ {h} (\cdot | s, a)} \left[ \bar {V} _ {h + 1} ^ {k} (s ^ {\prime}) - V _ {h + 1} ^ {\pi^ {k}} (s ^ {\prime}) \right] + \frac {3 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \\ = \frac {1}{4 H} P _ {h} \Big (\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}} \Big) (s, a) + \frac {3 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}, \\ \end{array}
$$

where the second inequality is since the value of $\pi^{k}$ cannot exceed the optimal value.

Since under the good event by Lemma 6, we have $0 \leq V_{h+1}^{\pi^k}(s') \leq V_{h+1}^*(s') \leq \bar{V}_{h+1}^k(s') \leq H$ , we can trivially bound the error by $H$ and bound

$$
\begin{array}{l} \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, a) - P _ {h} V _ {h + 1} ^ {\pi^ {k}} (s, a) \\ \leq \min \left\{\left(1 + \frac {1}{4 H}\right) \underbrace {P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s , a)} _ {\geq 0} + \frac {3 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} + \sqrt {\frac {2 \operatorname{Var} _ {P _ {h} (\cdot | s , a)} (V _ {h + 1} ^ {*}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}, H \right\} \\ \leq \left(1 + \frac {1}{4 H}\right) P _ {h} \Big (\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}} \Big) (s, a) + \min \left\{\sqrt {\frac {2 \mathrm{Var} _ {P _ {h} (\cdot | s , a)} (V _ {h + 1} ^ {*}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {4 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}, H \right\} \\ \triangleq \left(1 + \frac {1}{4 H}\right) P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) + b _ {k, h} ^ {p v 1} (s, a). \\ \end{array}
$$

Substituting back to Equation (6) while writing the linear operation $P_{h}V(s,a)$ as an expectation and letting the action be $a_{h} = \pi_{h}^{k}(s,\boldsymbol{R})$ , we get under $\mathbb{G}$ for all $k\in [K], h\in [H]$ and $s\in S$ that

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s) - V _ {h} ^ {\pi^ {k}} (s) \\ \leq \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \left[ \hat {P} _ {h} ^ {k - 1} \bar {V} _ {h + 1} ^ {k} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) - P _ {h} V _ {h + 1} ^ {\pi^ {k}} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) + b _ {k, h} ^ {p} (s, \pi_ {h} ^ {k} (s, \boldsymbol {R})) \right] + 2 b _ {k, h} ^ {r} (s) \\ \leq \mathbb {E} _ {\boldsymbol {R} \sim \mathcal {R} _ {h} (s)} \bigg [ \bigg (1 + \frac {1}{4 H} \bigg) \mathbb {E} \Big [ \bar {V} _ {h + 1} ^ {k} (s _ {h + 1}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1}) | s _ {h} = s, a _ {h} \Big ] (s, a) + b _ {k, h} ^ {p v 1} (s, a _ {h}) + b _ {k, h} ^ {p} (s, a _ {h}) \bigg ] + 2 b _ {k, h} ^ {r} (s) \\ = \mathbb {E} \bigg [ \bigg (1 + \frac {1}{4 H} \bigg) \Big (\bar {V} _ {h + 1} ^ {k} (s _ {h + 1}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1}) \Big) + b _ {k, h} ^ {p} (s _ {h}, a _ {h}) + b _ {k, h} ^ {p v 1} (s _ {h}, a _ {h}) | s _ {h} = s, \pi^ {k} \bigg ] + 2 b _ {k, h} ^ {r} (s). \\ \end{array}
$$

Next, taking $s = s_{h}^{k}$ , the action $a_{h} = \pi_{h}^{k}(s, \boldsymbol{R})$ becomes $a_{h}^{k}$ , and summing on all k, we can rewrite

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \bar {V} _ {h} ^ {k} (s _ {h} ^ {k}) - V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k}) \\ \leq \sum_ {k = 1} ^ {K} \mathbb {E} \bigg [ \bigg (1 + \frac {1}{4 H} \bigg) \Big (\bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k}) \Big) + b _ {k, h} ^ {p} (s _ {h} ^ {k}, a _ {h} ^ {k}) + b _ {k, h} ^ {p v 1} (s _ {h} ^ {k}, a _ {h} ^ {k}) | F _ {k, h - 1} \bigg ] + 2 \sum_ {k = 1} ^ {K} b _ {k, h} ^ {r} (s _ {h} ^ {k}) \\ \stackrel {(1)} {\leq} \left(1 + \frac {1}{2 H}\right) \left(1 + \frac {1}{4 H}\right) \sum_ {k = 1} ^ {K} \left(\bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k})\right) \\ + 2 \sum_ {k = 1} ^ {K} \Bigl (b _ {k, h} ^ {p} (s _ {h} ^ {k}, a _ {h} ^ {k}) + b _ {k, h} ^ {p v 1} (s _ {h} ^ {k}, a _ {h} ^ {k}) \Bigr) + 2 \sum_ {k = 1} ^ {K} b _ {k, h} ^ {r} (s _ {h} ^ {k}) + 6 8 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta} \\ \stackrel {(2)} {\leq} \left(1 + \frac {1}{2 H}\right) \left(1 + \frac {1}{4 H}\right) \sum_ {k = 1} ^ {K} \left(\bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k})\right) + \frac {1}{4 H} \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} \left(\bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k})\right) \\ + 1 8 \sum_ {k = 1} ^ {K} \sqrt {\frac {\mathrm{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k} , a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} + \sum_ {k = 1} ^ {K} \frac {1 6 2 0 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1} + 6 8 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta} + 2 \sum_ {k = 1} ^ {K} b _ {k, h} ^ {r} (s _ {h} ^ {k}) \\ \leq \left(1 + \frac {1}{2 H}\right) ^ {2} \sum_ {k = 1} ^ {K} \Bigl (\bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k}) \Bigr) + 1 8 \sum_ {k = 1} ^ {K} \frac {\sqrt {L _ {\delta} ^ {k} \mathrm{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k} , a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}})}}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} \\ + \sum_ {k = 1} ^ {K} \frac {1 7 0 0 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1} + 6 \sum_ {k = 1} ^ {K} \sqrt {\frac {A L _ {\delta} ^ {k}}{2 n _ {h} ^ {k - 1} (s) \vee 1}} \\ \end{array}
$$

where inequality (1) holds when both $E^{\mathrm{diff1}}$ and $E^{bp}$ occur and inequality (2) is by Lemma 8. In the last inequality, we also substituted the definition of the reward bonus. Recursively applying this

inequality up to $h = H + 1$ (where both values are zero), w.p. at least $1 - \delta$ , we get

$$
\begin{array}{l} \operatorname{Reg} ^ {R} (K) \leq \sum_ {k = 1} ^ {K} \left(V _ {1} ^ {*} \left(s _ {1} ^ {k}\right) - V _ {1} ^ {\pi^ {k}} \left(s _ {1} ^ {k}\right)\right) \\ \leq \sum_ {k = 1} ^ {K} \left(\bar {V} _ {1} ^ {k} (s _ {1} ^ {k}) - V _ {1} ^ {\pi^ {k}} (s _ {1} ^ {k})\right) \tag {Lemma6} \\ \leq 18\bigg(1 + \frac{1}{2H}\bigg)^{2H}\sum_{k = 1}^{K}\frac{\sqrt{L_{\delta}^{k}\mathrm{Var}_{P_{h}(\cdot|s_{h}^{k},a_{h}^{k})}(V_{h + 1}^{\pi^{k}})}}{\sqrt{n_{h}^{k - 1}(s_{h}^{k},a_{h}^{k})\vee 1}}\\ +\bigg(1 + \frac{1}{2H}\bigg)^{2H}\sum_{k = 1}^{K}\frac{1700H^{2}SL_{\delta}^{k}}{n_{h}^{k - 1}(s_{h}^{k},a_{h}^{k})\vee 1} \\ + 6 \left(1 + \frac {1}{2 H}\right) ^ {2 H} \sum_ {k = 1} ^ {K} \sqrt {\frac {A L _ {\delta} ^ {k}}{2 n _ {h} ^ {k - 1} (s) \vee 1}} \\ \stackrel {(*)} {\leq} 1 0 0 \sqrt {H ^ {3} S A K} L _ {\delta} ^ {K} + 5 0 \sqrt {2 S A} H ^ {2} \left(L _ {\delta} ^ {K}\right) ^ {1. 5} \\ + 5 0 0 0 H ^ {2} S L _ {\delta} ^ {K} \cdot S A H (2 + \ln (K)) + 1 2 \sqrt {A L _ {\delta} ^ {K}} \left(S H + 2 \sqrt {S H ^ {2} K}\right) \\ = \mathcal {O} \left(\sqrt {H ^ {3} S A K} L _ {\delta} ^ {K} + H ^ {3} S ^ {2} A \left(L _ {\delta} ^ {K}\right) ^ {2}\right). \\ \end{array}
$$

$$
\begin{array}{l} \stackrel {(*)} {\leq} 1 0 0 \sqrt {H ^ {3} S A K} L _ {\delta} ^ {K} + 5 0 \sqrt {2 S A} H ^ {2} \left(L _ {\delta} ^ {K}\right) ^ {1. 5} \\ + 5 0 0 0 H ^ {2} S L _ {\delta} ^ {K} \cdot S A H (2 + \ln (K)) + 1 2 \sqrt {A L _ {\delta} ^ {K}} \left(S H + 2 \sqrt {S H ^ {2} K}\right) \\ = \mathcal {O} \left(\sqrt {H ^ {3} S A K} L _ {\delta} ^ {K} + H ^ {3} S ^ {2} A \left(L _ {\delta} ^ {K}\right) ^ {2}\right). \\ \end{array}
$$

Relation (\*) is by Lemma 9 and Lemma 20.

![](images/eb50ac106d56f54ac7382468aa489968b926e39b768110436b5a116fd629d9b0.jpg)

# B.7.1 Lemmas for Bounding Bonus Terms

Lemma 8. Conditioned on the good event G, for any $h \in [H]$ , it holds that

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \Bigl (b _ {k, h} ^ {p} (s _ {h} ^ {k}, a _ {h} ^ {k}) + b _ {k, h} ^ {p v 1} (s _ {h} ^ {k}, a _ {h} ^ {k}) \Bigr) \leq \frac {1}{8 H} \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} \Bigl (\bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k}) \Bigr) \\ + 9 \sum_ {k = 1} ^ {K} \sqrt {\frac {\mathrm{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k} , a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} + \sum_ {k = 1} ^ {K} \frac {8 1 0 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}. \\ \end{array}
$$

Proof. We start by analyzing each of the terms separately. First, we apply Lemma 22 with $\alpha = \frac{20}{3} \cdot 32HL_{\delta}^{k}$ , noting that under the good event (by Lemma 6), $0 \leq V_{h+1}^{\pi^k}(s) \leq V_{h+1}^*(s) \leq \bar{V}_{h+1}^k(s) \leq H$ and using the event $E^{pv}$ ; doing so yields

$$
\begin{array}{l} b _ {k, h} ^ {p} (s, a) \leq \frac {2 0}{3} \sqrt {\frac {\operatorname{Var} _ {\hat {P} _ {h} ^ {k - 1} (\cdot | s , a)} (\bar {V} _ {h + 1} ^ {k}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {4 0 0}{9} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \\ \leq \frac {2 0 \sqrt {L _ {\delta} ^ {k} \operatorname{Var} _ {P _ {h} (\cdot | s , a)} \left(V _ {h + 1} ^ {\pi^ {k}}\right)}}{3 \sqrt {n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {1}{3 2 H} P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) + \frac {1}{3 2 H} \hat {P} _ {h} ^ {k - 1} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) \\ + \frac {6 4 0 0 H ^ {2} L _ {\delta} ^ {k}}{9 n _ {h} ^ {k - 1} (s , a) \vee 1} + \frac {2 0}{3} \frac {4 H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} + \frac {4 0 0}{9} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \\ \end{array}
$$

Using Lemma 24 with $\alpha = 1$ , under the good event $E^p(k)$ and for any $s, a$ , we can further bound

$$
\begin{array}{l} \hat {P} _ {h} ^ {k - 1} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) \\ = P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) + \left(\hat {P} _ {h} ^ {k - 1} - P _ {h}\right) \left(\bar {V} _ {h + 1} ^ {k} \left(s ^ {\prime}\right) - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) \\ \leq P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) + P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) + \frac {H S L _ {\delta} ^ {k} (1 + 2 \cdot 1 / 4)}{n _ {h} ^ {k - 1} (s , a) \vee 1} \tag {Lemma24} \\ \leq 2 P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) + \frac {1 . 5 H S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \\ \end{array}
$$

Thus, we get the overall bound

$$
b _ {k, h} ^ {p} (s, a) \leq \frac {2 0 \sqrt {L _ {\delta} ^ {k} \mathrm{Var} _ {P _ {h} (\cdot | s , a)} (V _ {h + 1} ^ {\pi^ {k}})}}{3 \sqrt {n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {3}{3 2 H} P _ {h} \Big (\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}} \Big) (s, a) + \frac {7 8 5 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}
$$

For the second bonus, we apply Lemma 21 w.r.t. $V_{h + 1}^{\pi^k}(s) \leq V_{h + 1}^* (s)$ and $\alpha = 32\sqrt{2L_{\delta}^{k}} H$ and get

$$
\begin{array}{l} b _ {k, h} ^ {p v 1} (s, a) \leq \sqrt {\frac {2 \operatorname{Var} _ {P _ {h} (\cdot | s , a)} \left(V _ {h + 1} ^ {*}\right) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {4 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \\ \leq \sqrt {\frac {2 \operatorname{Var} _ {P _ {h} (\cdot | s , a)} (V _ {h + 1} ^ {\pi^ {k}}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {1}{3 2 H} P _ {h} \left(V _ {h + 1} ^ {*} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) + \frac {1 6 H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a)} + \frac {4 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \\ \leq \sqrt {\frac {2 \operatorname{Var} _ {P _ {h} (\cdot | s , a)} (V _ {h + 1} ^ {\pi^ {k}}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} + \frac {1}{3 2 H} P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}}\right) (s, a) + \frac {2 0 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1} \\ \end{array}
$$

where we again used the optimism. Combining both and summing over all k, we get

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \Big (b _ {k, h} ^ {p} (s _ {h} ^ {k}, a _ {h} ^ {k}) + b _ {k, h} ^ {p v 1} (s _ {h} ^ {k}, a _ {h} ^ {k}) \Big) \leq 9 \sum_ {k = 1} ^ {K} \sqrt {\frac {\mathrm{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k} , a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}}) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} + \frac {1}{8 H} \sum_ {k = 1} ^ {K} P _ {h} \Big (\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}} \Big) (s _ {h} ^ {k}, a _ {h} ^ {k}) \\ + \sum_ {k = 1} ^ {K} \frac {8 0 5 H ^ {2} S L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1} \\ \end{array}
$$

Finally, under the good event $E^{\mathrm{diff2}}$ , it holds that

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} P _ {h} \Bigl (\bar {V} _ {h + 1} ^ {k} - V _ {h + 1} ^ {\pi^ {k}} \Bigr) (s _ {h} ^ {k}, a _ {h} ^ {k}) = \sum_ {k = 1} ^ {K} \mathbb {E} \Bigl [ \bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k}) | F _ {k, h - 1} ^ {R} \Bigr ] \\ \leq \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} \Bigl (\bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k}) \Bigr) + 1 8 H ^ {2} \ln \frac {8 H K (K + 1)}{\delta}. \\ \end{array}
$$

Substituting this relation back concludes the proof.

![](images/5d09519468013472f0189ffec8e3bc9b697e5a55dea5ee394906c6e5b009af44.jpg)

Lemma 9. Under the event $E^{Var}$ it holds that

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {\sqrt {\operatorname{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k} , a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}})}}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} \leq 2 \sqrt {H ^ {3} S A K L _ {\delta} ^ {K}} + \sqrt {8 S A H ^ {2}} L _ {\delta} ^ {K}.
$$

Proof. Following Lemma 24 of [Efroni et al., 2021], by Cauchy-Schwartz inequality, it holds that

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {\sqrt {\mathrm{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k} , a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}})}}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} \leq \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \mathrm{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k} , a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}})} \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}}.
$$

The second term can be bounded by Lemma 20, namely,

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1} \leq S A H (2 + \ln (K)).
$$

We further focus on bounding the first term. Under $E^{Var}$ , we have

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \mathrm{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k}, a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}}) \\ \leq 2 \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \operatorname{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k}, a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}}) | F _ {k - 1} \right] + 4 H ^ {3} \ln \frac {8 H K (K + 1)}{\delta} \quad (\text { Under   } E ^ {\text { Var }}) \\ \leq 2 \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \left(\sum_ {h = 1} ^ {H} R _ {h} (s _ {h} ^ {k}, a _ {h} ^ {k}) - V _ {1} ^ {\pi^ {k}} (s _ {1} ^ {k})\right) ^ {2} | F _ {k - 1} \right] + 4 H ^ {3} \ln \frac {8 H K (K + 1)}{\delta} \quad (\text { By   Lemma   3 }) \\ \leq 2 H ^ {2} K + 4 H ^ {3} \ln \frac {8 H K (K + 1)}{\delta}, \\ \end{array}
$$

where the last inequality is since both the values and cumulative rewards are bounded in $[0, H]$ . Combining both, we get

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {\sqrt {\operatorname{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k} , a _ {h} ^ {k})} (V _ {h + 1} ^ {\pi^ {k}})}}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} \leq \sqrt {2 H ^ {2} K + 4 H ^ {3} \ln \frac {8 H K (K + 1)}{\delta}} \sqrt {S A H (2 + \ln (K))} \\ \leq \sqrt {2 H ^ {2} K + 4 H ^ {3} \ln \frac {8 H K (K + 1)}{\delta}} \sqrt {2 S A H \ln \frac {8 H K (K + 1)}{\delta}} \\ \leq 2 \sqrt {H ^ {3} S A K L _ {\delta} ^ {K}} + \sqrt {8 S A} H ^ {2} L _ {\delta} ^ {K}. \\ \end{array}
$$

![](images/b4495e44f33f2c5ebfd1bf487ab130fc07dde9ff442b316edd7a7a4ef9ab98d3.jpg)

# C Proofs for Transition Lookahead

# C.1 Data Generation Process

As for the reward transition, we also assume that all data was generated before the game starts for all state-action-timesteps, and it is given to the agent when the relevant $(s, a, h)$ is visited. Thus, the rewards and next-state from the first $i^{th}$ visits at a state (or a state-action pair) at a certain timestep are i.i.d.

Throughout this appendix, we use the notation $s_{h+1}^{\prime k} = \left\{ s_{h+1}^{\prime k}(s_{h}^{k}, a) \right\}_{a \in \mathcal{A}}$ to denote the next-state observations at episode k and timestep h for all the actions, and use the equivalent filtrations to the ones defined at Appendix B.1, namely

$$
F _ {k, h} = \sigma \Big (\big \{s _ {t} ^ {1}, a _ {t} ^ {1}, \pmb {s} _ {t + 1} ^ {\prime 1}, R _ {t} ^ {1} \big \} _ {t \in [ H ]}, \ldots , \big \{s _ {t} ^ {k - 1}, a _ {t} ^ {k - 1}, \pmb {s} _ {t + 1} ^ {\prime k - 1}, R _ {t} ^ {k - 1} \big \} _ {t \in [ H ]}, \big \{s _ {t} ^ {k}, a _ {t} ^ {k}, \pmb {s} _ {t + 1} ^ {\prime k}, R _ {t} ^ {k} \big \} _ {t \in [ h ]} \Big),
$$

$$
F _ {k} = \sigma \Big (\big \{s _ {t} ^ {1}, a _ {t} ^ {1}, \boldsymbol {s} _ {t + 1} ^ {\prime 1} \big \} _ {t \in [ H ]}, \ldots , \big \{s _ {t} ^ {k}, a _ {t} ^ {k}, \boldsymbol {s} _ {t + 1} ^ {\prime k}, R _ {t} ^ {k} \big \} _ {t \in [ H ]}, s _ {1} ^ {k + 1} \Big).
$$

In particular, notice that since both $s_{h+1}'^k$ and $a_h^k$ are $F_{k,h}$ measurable, then so does $s_{h+1}^k$ .

# C.2 Extended MDP for Transition Lookahead

In this appendix, we present an equivalent extended MDP that embeds the lookahead into the state to fall under the vanilla MDP model, similarly to Appendix B.2. We use this equivalence to apply various existing results on MDPs without the need to reprove them. We follow the same conventions as Appendix B.2 while denoting transition lookahead values by $V^{T,\pi}(s|\mathcal{M})$ (and again, the superscript T will be omitted in subsequent subsections).

For any MDP $\mathcal{M} = (\mathcal{S},\mathcal{A},H,P,\mathcal{R})$ , let $\mathcal{M}^T$ be an MDP of horizon $2H$ and state space $\mathcal{S}^{A + 1}$ that separates the state transition and next-state generation as follows:

1. Assume w.l.o.g. that $\mathcal{M}$ starts at some initial state $s_1$ . The extended environment starts at a state $s_1 \times s_0'$ , where $s_0' \in S^A$ is a vector of $A$ copies of some arbitrary state $s_0 \in S$ .   
2. For any $h \in [H]$ , at timestep $2h - 1$ , the environment $\mathcal{M}^T$ transitions from state $s_h \times \boldsymbol{s}_0'$ to $s_h \times \boldsymbol{s}_{h+1}'$ , where $\boldsymbol{s}_{h+1}' \sim P_h(s)$ is a vector containing the next state for all actions $a \in \mathcal{A}$ ; this transition happens regardless of the action that the agent played. At timestep $2h$ , given an action $a_h$ , the environment transitions from $s_h \times \boldsymbol{s}_{h+1}'$ to $s_{h+1}'(a) \times \boldsymbol{s}_0'$ .   
3. The rewards at odd steps $2h - 1$ are zero, while the rewards at even steps $2h$ are $R_{h}(s_{h},a_{h})\sim \mathcal{R}_{h}(s_{h},a_{h})$ of expectation $r_h(s_h,a_h)$ .

As before, since the next state is embedded into the extended state space, any state-dependent policy in $M^{T}$ is a one-step transition lookahead policy in the original MDP. Also, the policy at even timesteps does not affect either the rewards or transitions, so it does not affect the value in any way. We again couple the two environments to have the exact same randomness, so assuming that the policy at the even steps in $M^{T}$ is the same as the policy in M, we trivially get the following relation between the values

$$
\begin{array}{l} V _ {2 h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} ^ {T}) = \mathbb {E} \left[ \sum_ {t = h} ^ {H} R _ {t} (s _ {t}, a _ {t}) | s _ {h} = s, s _ {h + 1} ^ {\prime} (s, \cdot) = \boldsymbol {s} ^ {\prime}, \pi \right] \triangleq V _ {h} ^ {T, \pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M}), \\ V _ {2 h - 1} ^ {\pi} (s, \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}) = \mathbb {E} \left[ \sum_ {t = h} ^ {H} R _ {t} (s _ {t}, a _ {t}) | s _ {h} = s, \pi \right] = V _ {h} ^ {T, \pi} (s | \mathcal {M}). \tag {7} \\ \end{array}
$$

While $M^{T}$ is finite, it is exponential in size, so applying any standard algorithm in this environment would lead to exponentially-bad performance bounds. Nonetheless, as with the extended-reward environment, we use this representation to prove useful results on one-step transition lookahead.

Proposition 2. The optimal value of one-step transition lookahead agents satisfies

$$
V _ {H + 1} ^ {T, *} (s) = 0, \quad \forall s \in \mathcal {S},
$$

$$
V _ {h} ^ {T, *} (s) = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \bigg [ \max _ {a \in \mathcal {A}} \Bigl \{r _ {h} (s, a) + V _ {h + 1} ^ {T, *} (s ^ {\prime} (s, a)) \Bigr \} \bigg ], \qquad \forall s \in \mathcal {S}, h \in [ H ].
$$

Also, given next-state observations $\mathbf{s}' = \{s'(a)\}_{a \in \mathcal{A}}$ at state $s$ and step $h$ , the optimal policy is

$$
\pi_ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \in \operatorname * {a r g   m a x} _ {a \in \mathcal {A}} \Bigl \{r _ {h} (s, a) + V _ {h + 1} ^ {T, *} (s ^ {\prime} (a)) \Bigr \}.
$$

Proof. We prove the result in the extended MDP $M^{T}$ , in which (as with reward lookahead) the optimal value can be calculated using the Bellman equations as follows [Puterman, 2014]

$$
V _ {2 H + 1} ^ {T} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} ^ {T}) = 0, \quad \forall s \in \mathcal {S}, \boldsymbol {s} ^ {\prime} \in \mathcal {S} ^ {A},
$$

$$
V _ {2 h} ^ {*} (s, \pmb {s} ^ {\prime} | \mathcal {M} ^ {T}) = \max _ {a} \bigl \{r _ {h} (s, a) + V _ {2 h + 1} ^ {*} (s ^ {\prime} (a), \pmb {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}) \bigr \}, \quad \forall h \in [ H ], s \in \mathcal {S}, \pmb {s} ^ {\prime} \in \mathcal {S} ^ {A},
$$

$$
V _ {2 h - 1} ^ {*} (s, \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}) = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ V _ {2 h} ^ {*} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} ^ {T}) \right], \quad \forall h \in [ H ], s \in \mathcal {S}. \tag {8}
$$

By the equivalence between $\mathcal{M}$ and $\mathcal{M}^T$ for all policies, this is also the optimal value in $\mathcal{M}$ . Combining both recursion equations and substituting Equation (7) leads to the stated value calculation for all $h\in [H]$ and $s\in S$ :

$$
\begin{array}{l} V _ {h} ^ {T, *} (s | \mathcal {M}) = V _ {2 h - 1} ^ {*} (s, \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}) \\ = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ V _ {2 h} ^ {*} (s, \boldsymbol {s} _ {h + 1} ^ {\prime} | \mathcal {M} ^ {T}) \right] \\ = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \Big [ \max _ {a} \bigl \{r _ {h} (s, a) + V _ {2 h + 1} ^ {*} (s _ {h + 1} ^ {\prime} (a), \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}) \bigr \} \Bigr ] \\ = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \Big [ \max _ {a} \Bigl \{r _ {h} (s, a) + V _ {h + 1} ^ {T, *} (s _ {h + 1} ^ {\prime} (a) | \mathcal {M}) \Bigr \} \Bigr ]. \\ \end{array}
$$

In addition, a given state $s$ and next-state observations $s'$ , the optimal policy at the even stages of the extended MDP is

$$
\pi_ {2 h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \in \underset {a \in \mathcal {A}} {\arg \max} \bigl \{r _ {h} (s, a) + V _ {2 h + 1} ^ {*} (s ^ {\prime} (a)) \bigr \},
$$

alongside arbitrary actions at odd steps. Playing this policy in the original MDP will lead to the optimal one-step transition lookahead policy, as it achieves the optimal value of the original MDP. By the value relations between the two environments $(V_{2h+1}^{*}(s,\boldsymbol{s}_{0}^{\prime}|\mathcal{M}^{T})=V_{h+1}^{T,*}(s|\mathcal{M}))$ , this is equivalent to the stated policy. □

Remark 2. As in Remark 1, one could write the dynamic programming equations for any policy $\pi \in \Pi^T$ , and not just to the optimal one, namely

$$
V _ {2 h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} ^ {T}) = r _ {h} (s, \pi (s, \boldsymbol {s} ^ {\prime})) + V _ {2 h + 1} ^ {*} (s ^ {\prime} (\pi_ {h} (s, \boldsymbol {s} ^ {\prime})), \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}), \quad \forall h \in [ H ], s \in \mathcal {S}, \boldsymbol {s} ^ {\prime} \in \mathcal {S} ^ {A},
$$

$$
V _ {2 h - 1} ^ {\pi} (s, \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}) = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ V _ {2 h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} ^ {T}) \big ], \quad \forall h \in [ H ], s \in \mathcal {S}.
$$

In particular, following the notation of Equation (7), we can write

$$
\begin{array}{l} V _ {h} ^ {T, \pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M}) = r _ {h} (s, \pi_ {h} (s, \boldsymbol {s} ^ {\prime})) + V _ {h + 1} ^ {T, \pi} (s ^ {\prime} (\pi_ {h} (s, \boldsymbol {s} ^ {\prime})) | \mathcal {M}), \qquad a n d, \\ V _ {h} ^ {T, \pi} (s | \mathcal {M}) = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \Big [ V _ {h} ^ {T, \pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M}) \Big ] \\ = \mathbb {E} _ {\pmb {s ^ {\prime}} \sim P _ {h} (s)} \Big [ r _ {h} \big (s, \pi_ {h} (s, \pmb {s ^ {\prime}}) \big) + V _ {h + 1} ^ {T, \pi} (s ^ {\prime} (\pi_ {h} (s, \pmb {s ^ {\prime}})) | \mathcal {M}) \Big ], \\ \end{array}
$$

a notation that will be extensively used for transition lookahead.

We also prove a variation of the law of total variance (LTV) for transition lookahead:

Lemma 10. For any one-step transition lookahead policy $\pi \in \Pi^{T}$ , it holds that

$$
\mathbb {E} \left[ \sum_ {h = 1} ^ {H} \mathrm{Var} _ {\pmb {s} ^ {\prime} \sim P _ {h} (s _ {h})} (V _ {h} ^ {T, \pi} (s _ {h}, \pmb {s} ^ {\prime})) | \pi , s _ {1} \right] \leq \mathbb {E} \left[ \left(\sum_ {h = 1} ^ {H} r _ {h} (s _ {h}, a _ {h}) - V _ {1} ^ {T, \pi} (s _ {1})\right) ^ {2} | \pi , s _ {1} \right].
$$

Proof. We apply the law of total variance in the extended MDP; there, the expected rewards are either 0 (at odd steps) or $r_{h}(s_{h}, a_{h})$ (at even steps), so the total expected rewards are $\sum_{h=1}^{H} r_{h}(s_{h}, a_{h})$ . Hence, by Lemma 27,

$$
\mathbb {E} \left[ \left(\sum_ {h = 1} ^ {H} r _ {h} (s _ {h}, a _ {h}) - V _ {1} ^ {\pi} (s _ {1}, \pmb {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T})\right) ^ {2} | \pi , s _ {1} \right]
$$

$$
= \mathbb {E} \left[ \underbrace {\sum_ {h = 1} ^ {H} \mathrm{Var} (V _ {2 h} ^ {\pi} (s _ {h} , \boldsymbol {s} _ {h + 1} ^ {\prime} | \mathcal {M} ^ {T}) | (s _ {h} , \boldsymbol {s} _ {0} ^ {\prime}))} _ {\text {Odd steps}} + \underbrace {\sum_ {h = 1} ^ {H} \mathrm{Var} (V _ {2 h + 1} ^ {\pi} (s _ {h + 1} , \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}) | (s _ {h} , \boldsymbol {s} _ {h + 1} ^ {\prime}))} _ {\text {Even steps}}   | \pi , s _ {1} \right]
$$

$$
\geq \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \operatorname{Var} (V _ {2 h} ^ {\pi} (s _ {h}, \boldsymbol {s} _ {h + 1} | \mathcal {M} ^ {T}) | (s _ {h}, \boldsymbol {s} _ {0} ^ {\prime})) | \pi , s _ {1} \right]
$$

$$
= \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h})} (V _ {2 h} ^ {\pi} (s _ {h}, \boldsymbol {s} ^ {\prime} | \mathcal {M} ^ {T})) | \pi , s _ {1} \right]
$$

$$
= \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h})} \left(V _ {h} ^ {T, \pi} \left(s _ {h}, \boldsymbol {s} ^ {\prime} | \mathcal {M}\right)\right) | \pi , s _ {1} \right].
$$

Using again the identity $V_{1}^{\pi}(s_{1},\pmb{s}_{0}^{\prime}|\mathcal{M}^{T}) = V_{1}^{T,\pi}(s_{1}|\mathcal{M})$ leads to the desired result.

![](images/e54d27bc015e18b7cd85c137a171c885bdfd5a6cb5785b079b2ecbbecb7497ca.jpg)

Finally, prove a value-difference lemma also for transition lookahead

Lemma 11 (Value-Difference Lemma with Transition Lookahead). Let $\mathcal{M}_1 = (\mathcal{S},\mathcal{A},H,P^1,\mathcal{R}^1)$ and $\mathcal{M}_2 = (\mathcal{S},\mathcal{A},H,P^2,\mathcal{R}^2)$ be two environments. For any deterministic one-step transition lookahead policy $\pi \in \Pi^T$ , any $h\in [H]$ and $s\in \mathcal{S}$ , it holds that

$$
\begin{array}{l} V _ {h} ^ {T, \pi} (s | \mathcal {M} _ {1}) - V _ {h} ^ {T, \pi} (s | \mathcal {M} _ {2}) \\ = \mathbb {E} _ {\mathcal {M} _ {1}} \big [ r _ {h} ^ {1} (s _ {h}, \pi_ {h} (s _ {h}, \pmb {s} _ {h + 1} ^ {\prime})) - r _ {h} ^ {2} (s _ {h}, \pi_ {h} (s _ {h}, \pmb {s} _ {h + 1} ^ {\prime})) | s _ {h} = s \big ] \\ + \mathbb {E} _ {\mathcal {M} _ {1}} \Big [ V _ {h + 1} ^ {T, \pi} (s _ {h + 1} | \mathcal {M} _ {1}) - V _ {h + 1} ^ {T, \pi} (s _ {h + 1} | \mathcal {M} _ {2}) | s _ {h} = s \Big ] \\ + \mathbb {E} _ {\mathcal {M} _ {1}} \Big [ \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {1} (s _ {h})} \Big [ V _ {h} ^ {T, \pi} (s _ {h}, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {2}) \Big ] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {2} (s _ {h})} \Big [ V _ {h} ^ {T, \pi} (s _ {h}, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {2}) \Big ] | s _ {h} = s \Big ]. \\ \end{array}
$$

where $V_{h}^{T,\pi}(s,s^{\prime}|\mathcal{M})$ is the value at a state given the reward realization, defined in Equation (7) and given in Remark 2.

Proof. We again work with the extended MDPs $M_{1}^{T}$ , $M_{2}^{T}$ and use their Bellman equations, namely,

$$
V _ {2 h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} ^ {T}) = r _ {h} (s, \pi (s, \boldsymbol {s} ^ {\prime})) + V _ {2 h + 1} ^ {*} (s ^ {\prime} (\pi_ {h} (s, \boldsymbol {s} ^ {\prime})), \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}), \quad \forall h \in [ H ], s \in \mathcal {S}, \boldsymbol {s} ^ {\prime} \in \mathcal {S} ^ {A},
$$

$$
V _ {2 h - 1} ^ {\pi} (s, \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} ^ {T}) = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ V _ {2 h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} ^ {T}) \big ], \quad \forall h \in [ H ], s \in \mathcal {S}.
$$

Using the relation between the value of the original and extended MDP (eq. (7)) and the Bellman equations of the extended MDP, for any $h \in [H]$ , we have

$$
\begin{array}{l} V _ {h} ^ {T, \pi} (s | \mathcal {M} _ {1}) - V _ {h} ^ {T, \pi} (s | \mathcal {M} _ {2}) \\ = V _ {2 h - 1} ^ {\pi} (s, \pmb {s} _ {0} ^ {\prime} | \mathcal {M} _ {1} ^ {T}) - V _ {2 h - 1} ^ {\pi} (s, \pmb {s} _ {0} ^ {\prime} | \mathcal {M} _ {2} ^ {T}) \\ = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {1} (s)} \left[ V _ {2 h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {1} ^ {T}) \right] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {2} (s)} \left[ V _ {2 h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {2} ^ {T}) \right] \\ = \mathbb {E} _ {\pmb {s ^ {\prime}} \sim P _ {h} ^ {1} (s)} \big [ V _ {2 h} ^ {\pi} (s, \pmb {s ^ {\prime}} | \mathcal {M} _ {1} ^ {T}) - V _ {2 h} ^ {\pi} (s, \pmb {s ^ {\prime}} | \mathcal {M} _ {2} ^ {T}) \big ] + \mathbb {E} _ {\pmb {s ^ {\prime}} \sim P _ {h} ^ {1} (s)} \big [ V _ {2 h} ^ {\pi} (s, \pmb {s ^ {\prime}} | \mathcal {M} _ {2} ^ {T}) \big ] - \mathbb {E} _ {\pmb {s ^ {\prime}} \sim P _ {h} ^ {2} (s)} \big [ V _ {2 h} ^ {\pi} (s, \pmb {s ^ {\prime}} | \mathcal {M} _ {2} ^ {T}) \big ] \\ = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {1} (s)} \left[ V _ {2 h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {1} ^ {T}) - V _ {2 h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {2} ^ {T}) \right] + \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {1} (s)} \left[ V _ {h} ^ {T, \pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {2}) \right] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {2} (s)} \left[ V _ {h} ^ {T, \pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {2}) \right] \\ = \mathbb {E} _ {\mathcal {M} _ {1}} \big [ V _ {2 h} ^ {\pi} (s _ {h}, \pmb {s} _ {h + 1} ^ {\prime} | \mathcal {M} _ {1} ^ {T}) - V _ {2 h} ^ {\pi} (s _ {h}, \pmb {s} _ {h + 1} ^ {\prime} | \mathcal {M} _ {2} ^ {T}) | s _ {h} = s \big ] \\ + \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {1} (s)} \left[ V _ {h} ^ {T, \pi} \left(s, \boldsymbol {s} ^ {\prime} \mid \mathcal {M} _ {2}\right) \right] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {2} (s)} \left[ V _ {h} ^ {T, \pi} \left(s, \boldsymbol {s} ^ {\prime} \mid \mathcal {M} _ {2}\right) \right]. \tag {9} \\ \end{array}
$$

Denoting $a_{h} = \pi_{h}(s_{h},\pmb{s}_{h + 1}^{\prime})$ the action taken by the agent at environment $\mathcal{M}_1$ , We have

$$
\begin{array}{l} V _ {2 h} ^ {\pi} (s _ {h}, \boldsymbol {s} _ {h + 1} ^ {\prime} | \mathcal {M} _ {1} ^ {T}) - V _ {2 h} ^ {\pi} (s _ {h}, \boldsymbol {s} _ {h + 1} ^ {\prime} | \mathcal {M} _ {2} ^ {T}) \\ = \left(r _ {h} ^ {1} (s _ {h}, a _ {h}) + V _ {2 h + 1} ^ {\pi} (s _ {h + 1} ^ {\prime} (a _ {h}), \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} _ {1} ^ {T})\right) - \left(r _ {h} ^ {2} (s _ {h}, a _ {h}) + V _ {2 h + 1} ^ {\pi} (s _ {h + 1} ^ {\prime} (a _ {h}), \boldsymbol {s} _ {0} ^ {\prime} | \mathcal {M} _ {2} ^ {T})\right) \\ = r _ {h} ^ {1} (s _ {h}, a _ {h}) - r _ {h} ^ {2} (s _ {h}, a _ {h}) + V _ {h + 1} ^ {T, \pi} (s _ {h + 1} ^ {\prime} (a _ {h}) | \mathcal {M} _ {1}) - V _ {h + 1} ^ {T, \pi} (s _ {h + 1} ^ {\prime} (a _ {h}) | \mathcal {M} _ {2}), \\ \end{array}
$$

when taking the expectation w.r.t. $\mathcal{M}_1$ , it holds that $s_{h + 1}^{\prime}(a_h) = s_{h + 1}$ ; substituting this back into Equation (9), we get

$$
\begin{array}{l} = \mathbb {E} _ {\mathcal {M} _ {1}} \Big [ r _ {h} ^ {1} (s _ {h}, a _ {h}) - r _ {h} ^ {2} (s _ {h}, a _ {h}) + V _ {h + 1} ^ {T, \pi} (s _ {h + 1} ^ {\prime} (a _ {h}) | \mathcal {M} _ {1}) - V _ {h + 1} ^ {T, \pi} (s _ {h + 1} ^ {\prime} (a _ {h}) | \mathcal {M} _ {2}) | s _ {h} = s \Big ] \\ + \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {1} (s)} \Big [ V _ {h} ^ {T, \pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {2}) \Big ] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {2} (s)} \Big [ V _ {h} ^ {T, \pi} (s, \boldsymbol {s} ^ {\prime} | \mathcal {M} _ {2}) \Big ] \\ = \mathbb {E} _ {\mathcal {M} _ {1}} \left[ r _ {h} ^ {1} (s _ {h}, \pi_ {h} (s _ {h}, \boldsymbol {s} _ {h + 1} ^ {\prime})) - r _ {h} ^ {2} (s _ {h}, \pi_ {h} (s _ {h}, \boldsymbol {s} _ {h + 1} ^ {\prime})) | s _ {h} = s \right] \\ + \mathbb {E} _ {\mathcal {M} _ {1}} \Big [ V _ {h + 1} ^ {T, \pi} (s _ {h + 1} | \mathcal {M} _ {1}) - V _ {h + 1} ^ {T, \pi} (s _ {h + 1} | \mathcal {M} _ {2}) | s _ {h} = s \Big ] \\ + \mathbb {E} _ {\mathcal {M} _ {1}} \left[ \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {1} (s _ {h})} \left[ V _ {h} ^ {T, \pi} \left(s _ {h}, \boldsymbol {s} ^ {\prime} \mid \mathcal {M} _ {2}\right) \right] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} ^ {2} (s _ {h})} \left[ V _ {h} ^ {T, \pi} \left(s _ {h}, \boldsymbol {s} ^ {\prime} \mid \mathcal {M} _ {2}\right) \right] \mid s _ {h} = s \right]. \\ \end{array}
$$

$$
V _ {h} ^ {\pi} (s | \mathcal {M} _ {1}) - V _ {h} ^ {\pi} (s | \mathcal {M} _ {2})
$$

□

Algorithm 4 Monotonic Value Propagation with Transition Lookahead (MVP-TL)   
1: Require: $\delta \in (0,1)$ , bonuses $b_{k,h}^{r}(s,a), b_{k,h}^{p}(s)$ 2: for $k = 1,2,\ldots$ do
3: Initialize $\bar{V}_{H+1}^{k}(s) = 0$ 4: for $h = H, H-1,..,1$ do
5:    for $s \in S$ do
6:    if $n_{h}^{k-1}(s) = 0$ then
7: $\bar{V}_{h}^{k}(s) = H$ 8:    else
9:    Calculate the truncated values $\bar{V}_{h}^{k}(s) = \min\left\{\frac{1}{n_{h}^{k-1}(s)} \sum_{t=1}^{n_{h}^{k-1}(s)} \max_{a \in A} \left\{ \hat{r}_{h}^{k-1}(s,a) + b_{k,h}^{r}(s,a) + \bar{V}_{h+1}^{k}(s'_{h+1}^{t}(s),a)) \right\} + b_{k,h}^{p}(s), H \right\}$ 10:    end if
11:    For any set of next-states $s' \in S^{A}$ , define the policy $\pi^{k}$ $\pi_{h}^{k}(s,s') \in \arg\max_{a \in A} \left\{ \hat{r}_{h}^{k-1}(s,a) + b_{k,h}^{r}(s,a) + \bar{V}_{h+1}^{k}(s'(a)) \right\}$ 12:    end for
13:    end for
14:    for $h = 1,2,\ldots,H$ do
15:    Observe $s_{h}^{k}$ and $s_{h+1}'^{rk} = \left\{ s_{h+1}'^{rk}(s_{h}^{k},a) \right\}_{a \in A}$ 16:    Play an action $a_{h}^{k} = \pi_{h}^{k}(s_{h}^{k},s_{h}'^{rk})$ 17:    Collect the reward $R_{h}^{k} \sim \mathcal{R}_{h}(s_{h}^{k},a_{h}^{k})$ and transition to the next state $s_{h+1}^{k} = s_{h+1}'^{rk}(s_{h}^{k},a_{h}^{k})$ 18:    end for
19:    Update the empirical estimators and counts for all visited state-actions
20: end for

As with reward lookahead, we again use a variant of the MVP algorithm [Zhang et al., 2021b], described in Algorithm 4. For the bonuses, we use the notation

$$
\bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) = \max _ {a \in \mathcal {A}} \left\{\hat {r} _ {h} ^ {k - 1} (s, a) + b _ {k, h} ^ {r} (s, a) + \bar {V} _ {h + 1} ^ {k} (s ^ {\prime} (a) \right\}
$$

and define the following bonuses:

$$
\begin{array}{l} b _ {k, h} ^ {r} (s, a) = \min \Bigg \{\sqrt {\frac {L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}}, 1 \Bigg \}, \\ b _ {k, h} ^ {p} (s) = \frac {2 0}{3} \sqrt {\frac {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} (\bar {V} _ {h} ^ {k} (s , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {4 0 0}{3} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}, \\ \end{array}
$$

where $L_{\delta}^{k} = \ln \frac{16S^{3}A^{2}Hk^{2}(k + 1)}{\delta}$ and

$$
\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} (\bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime})) = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) ^ {2} \big ] - \left(\mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) \big ]\right) ^ {2}.
$$

The notation $k_{h}^{t}(s)$ again represents the $t^{th}$ episode where the state s was visited at the $h^{th}$ timestep; in particular, line 9 of the algorithm is the expectation w.r.t. the empirical reward distribution $\hat{P}_{h}^{k-1}(s)$ . Since the transition bonus is larger than H when $n_{h}^{k-1}(s) = 0$ , we can arbitrarily define the expectation w.r.t. $\hat{P}_{h}^{k-1}(s)$ when $n_{h}^{k-1}(s) = 0$ to be 0, and one could write the update in a more concise way as

$$
\bar {V} _ {h} ^ {k} (s) = \min \Bigl \{\mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \bigl [ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) \bigr ] + b _ {k, h} ^ {p} (s), H \Bigr \}.
$$

# C.4 Additional Notations and List Representation

In this subsection, we present additional notations for both values and transition distributions that will be helpful in the analysis. In particular, we show that instead of looking at the distribution over all combinations of next state $s' \in S^{A}$ , we can look at a ranking of all the next-state-actions and represent important quantities using the effective distribution on these ranks – this moves the problem from being $S^{A}$ -dimensional to a dimension of SA.

We start by defining the values starting from state $s \in S$ , playing $a \in \mathcal{A}$ and transitioning to $s' \in S$ , denoted by

$$
V _ {h} ^ {\pi} (s, s ^ {\prime}, a) = r _ {h} (s, a) + V _ {h + 1} ^ {\pi} (s ^ {\prime}),
$$

$$
V _ {h} ^ {*} (s, s ^ {\prime}, a) = r _ {h} (s, a) + V _ {h + 1} ^ {*} (s ^ {\prime}),
$$

$$
\bar {V} _ {h} ^ {k} (s, s ^ {\prime}, a) = \hat {r} _ {h} ^ {k - 1} (s, a) + b _ {k, h} ^ {r} (s, a) + \bar {V} _ {h + 1} ^ {k} (s ^ {\prime}),
$$

We similarly define (consistently with Remark 2)

$$
V _ {h} ^ {\pi} (s, \boldsymbol {s} ^ {\prime}) = V _ {h} ^ {\pi} (s, s ^ {\prime} (\pi_ {h} (s, \boldsymbol {s} ^ {\prime})), \pi_ {h} (s, \boldsymbol {s} ^ {\prime})),
$$

$$
V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) = \max _ {a} V _ {h} ^ {*} (s, s ^ {\prime} (a), a), \quad \text { and },
$$

$$
\bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) = \max _ {a} \bar {V} _ {h} ^ {k} (s, s ^ {\prime} (a), a).
$$

List representation. We now move to defining lists of next-state-actions and distributions with respect to such lists. Let $\ell$ be a list that orders all next-state-action pairs from $(s'_{\ell(1)}, a_{\ell(1)})$ to $(s'_{\ell(SA)}, a_{\ell(SA)})$ and define the set of all possible lists to be L (with $|\mathcal{L}| = (SA)!$ ). Also, define $\ell^{u}$ , the list induced by a function $u : S \times A \mapsto R$ such that $u(s'_{\ell^{u}(1)}, a_{\ell^{u}(1)}) \geq \cdots \geq u(s'_{\ell^{u}(SA)}, a_{\ell^{u}(SA)})$ , where ties are broken in any fixed arbitrary way. From this point forward, for brevity and when clear from the context, we omit the list from the indexing, e.g., write the list $\ell$ by $(s'_{1}, a_{1}), \ldots, (s'_{SA}, a_{SA})$ .

We now define the probability of list elements. Denote by $E_{i}^{\ell}$ the event that the highest-ranked realized element in the list is element $i$ , namely

$$
E _ {i} ^ {\ell} = \left\{\boldsymbol {s} ^ {\prime} \in \mathcal {S} ^ {A}: s ^ {\prime} (a _ {i}) = s _ {i} ^ {\prime} \text {   and   } \forall j <   i, s ^ {\prime} (a _ {j}) \neq s _ {j} ^ {\prime} \right\}. \tag {10}
$$

Then, for a probability measure $P$ on $S^A$ , define $\mu(i|\ell, P) = P(s' \in E_i^\ell)$ . Notably, when the list is induced by $u$ and element $i$ is the realized highest-ranked elements, we can write $\max_a u(s'(a), a) = u(s_i', a_i)$ , so we have that (e.g. by Lemma 17 with $f(s') = \max_a u(s'(a), a)$ )

$$
\mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \Big [ \max _ {a} \{u (s ^ {\prime} (a), a) \} \Big ] = \mathbb {E} _ {i \sim \mu (\cdot | \ell , P _ {h} (s))} [ u (s _ {i} ^ {\prime}, a _ {i}) ]
$$

We also denote by $\hat{\mu}_h^k (i|s;\ell) = \frac{1}{n_h^k(s)\vee 1}\sum_{t = 1}^{K}\mathbb{1}\big\{s_h^t = s,\pmb{s}_{h + 1}^{\prime t}\in E_i^\ell \big\}$ , the empirical probability for a list location $i$ to be the highest-realized ranking according to a list $\ell$ at state $s$ and step $h$ , based on samples up to episode $k$ ; We have by Lemma 17 that $\hat{\mu}_h^k (i|s;\ell) = \hat{P}_h^k (E_i^\ell |s)$ and

$$
\mathbb {E} _ {\pmb {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \Big [ \max _ {a} \{u (s ^ {\prime} (a), a) \} \Big ] = \mathbb {E} _ {i \sim \hat {\mu} _ {h} ^ {k - 1} (\cdot | s; \ell^ {u})} [ u (s _ {i} ^ {\prime}, a _ {i}) ].
$$

Similarly, we will require the distribution probability w.r.t. two lists - the probability that the top element w.r.t. list $\ell$ is $i$ and the top element w.r.t. list $\ell'$ is $j$ ; we denote the real and empirical probability distributions by $\mu(i,j|\ell,\ell',P)$ and $\hat{\mu}_h^k (i,j|s;\ell,\ell')$ , respectively. This allows, for example, using Lemma 17 to write for any $u,v:\mathcal{S}\times \mathcal{A}\mapsto \mathbb{R}$ ,

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \Big [ \max _ {a} \{u (s ^ {\prime} (a), a) \} - \max _ {a} \{v (s ^ {\prime} (a), a) \} \Big ] \\ = \mathbb {E} _ {i, j \sim \mu (\cdot | \ell^ {u}, \ell^ {v}, P _ {h} (s))} \left[ u \left(s _ {\ell^ {u} (i)} ^ {\prime}, a _ {\ell^ {u} (i)}\right) - v \left(s _ {\ell^ {v} (j)} ^ {\prime}, a _ {\ell^ {v} (j)}\right) \right], \\ \end{array}
$$

$$
\mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \left[ \max _ {a} \{u (s ^ {\prime} (a), a) \} - \max _ {a} \{v (s ^ {\prime} (a), a) \} \right]
$$

$$
= \mathbb {E} _ {i, j \sim \hat {\mu} _ {h} ^ {k} (\cdot | s; \ell^ {u}, \ell^ {v})} \left[ u (s _ {\ell^ {u} (i)} ^ {\prime}, a _ {\ell^ {u} (i)}) - v (s _ {\ell^ {v} (j)} ^ {\prime}, a _ {\ell^ {v} (j)}) \right]. \tag {11}
$$

Finally, we say that a policy $\pi_h(s, s')$ is induced by lists $\ell_h(s)$ if it chooses an action $a$ such that its next-state $s'(a)$ is ranked higher in $\ell$ than all other realized next-state-action pairs. In particular, the policy $\pi^k$ and the optimal policy $\pi^*$ (defined in Proposition 2) are such policies w.r.t. the lists $\overline{\ell}_h^k(s)$ and $\ell_h^*(s) - \text{induced by } V_h^k(s, s', a)$ and $V_h^*(s, s', a)$ , respectively. As such, for any probability measure $P_h(s)$ , function $u: \mathcal{S} \times \mathcal{S} \times \mathcal{A} \mapsto \mathbb{R}$ and a policy $\pi$ induced by a list $\ell$ , it holds that

$$
\mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} [ u (s, s ^ {\prime} (\pi (a)), \pi (a)) ] = \mathbb {E} _ {i \sim \mu (\cdot | \ell_ {h} (s), P _ {h} (s))} [ u (s, s _ {i} ^ {\prime}, a _ {i}) ]. \tag {12}
$$

# C.4.1 Planning with Transition Lookahead

We have already seen the optimal policy is induced by a list $\ell_h^*(s)$ , and in particular, we can write the dynamic programming equations of Proposition 2 as

$$
V _ {h} ^ {*} (s) = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \max _ {a \in \mathcal {A}} \left\{r _ {h} (s, a) + V _ {h} ^ {T, *} \left(s ^ {\prime} (a)\right) \right\} \right]
$$

$$
= \mathbb {E} _ {i \sim \mu (\cdot | \ell_ {h} ^ {*} (s), P _ {h} (s))} \big [ r _ {h} (s, a _ {i}) + V _ {h + 1} ^ {*} (s ^ {\prime} (a _ {i})) \big ].
$$

Therefore, one way to perform the planning is to build a list $\ell_h^*(s)$ of $(s', a)$ s.t. the values

$$
V _ {h} ^ {*} (s, s ^ {\prime}, a) = r _ {h} (s, a) + V _ {h + 1} ^ {*} (s ^ {\prime})
$$

are sorted in a non-increasing order and calculate the probability of any pair in the list to be the highest-realized pair:

$$
\mu (i | \ell , P _ {h} (s)) = P _ {h} (E _ {i} ^ {\ell}) = \operatorname * {P r} \bigl (s _ {h + 1} ^ {\prime} (a _ {i}) = s _ {i} ^ {\prime} \text { and } \forall j <   i, s _ {h + 1} ^ {\prime} (a _ {j}) \neq s _ {j} ^ {\prime} | s _ {h} = s \bigr).
$$

In general, calculating this distribution is intractable, and one must resort to approximating it by sampling (as done in Algorithm 4. Nonetheless, if next states are generated independently between actions, this distribution could be efficiently calculated as follows:

$$
\mu (i | \ell , P _ {h} (s)) = \operatorname * {P r} \bigl (s _ {h + 1} ^ {\prime} (a _ {i}) = s _ {i} ^ {\prime} \text {and} \forall j <   i, s _ {h + 1} ^ {\prime} (a _ {j}) \neq s _ {j} ^ {\prime} | s _ {h} = s \bigr)
$$

$$
\stackrel {(1)} {=} \operatorname * {P r} \left\{s ^ {\prime} \left(a _ {i}\right) = s _ {i} ^ {\prime} \text {and} \forall j <   i \text {s.t.} a _ {j} \neq a _ {i}, s ^ {\prime} \left(a _ {j}\right) \neq s _ {j} ^ {\prime} \mid s _ {h} = s \right\}
$$

$$
\stackrel {(2)} {=} \operatorname * {P r} \{s ^ {\prime} (a _ {i}) = s _ {i} ^ {\prime} | s _ {h} = s \} \prod_ {a \neq a _ {i}} \operatorname * {P r} \bigl \{\forall j <   i \text {   s.t.   } a _ {j} = a, s ^ {\prime} (a) \neq s _ {j} ^ {\prime} | s _ {h} = s \bigr \}
$$

$$
\stackrel {(3)} {=} P _ {h} (s _ {i} ^ {\prime} | s, a _ {i}) \prod_ {a \neq a _ {i}} \left(1 - \sum_ {j = 1} ^ {i - 1} \mathbb {1} \{a _ {j} = a \} P _ {h} (s _ {j} ^ {\prime} | s, a)\right).
$$

Relation (1) holds since if $s'(a_i) = s_i'$ , it cannot get any previous value of the same action in the list, so these events can be removed. Relation (2) is by the independence and (3) directly calculates the probabilities.

# C.5 The First Good Event – Concentration

Next, we define the events that ensure the concentration of all empirical measures. For rewards, an event handles the convergence of the empirical rewards to their mean. For the transitions, we want the Bellman operator, applied on the optimal value with the empirical model, to concentrate well, and we require the variance of values w.r.t. the empirical and real model to be close. Finally, the empirical measure $\hat{\mu}_{h}^{k}(i,j|s;\ell,\ell_{h}^{*}(s))$ must concentrate well around its mean for any list $\ell$ – this will allow the change-of-measure argument described in the proof sketch.

Formally, define the following good events:

$$
E ^ {r} (k) = \left\{\forall s, a, h: | r _ {h} (s, a) - \hat {r} _ {h} ^ {k - 1} (s, a) | \leq \sqrt {\frac {L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}} \right\}
$$

$$
\begin{array}{l} E ^ {\ell} (k) = \left\{\forall s, h, \forall \ell \in \mathcal {L}, \forall i, j \in [ S A ]: \left| \hat {\mu} _ {h} ^ {k - 1} (i, j | s; \ell , \ell_ {h} ^ {*} (s)) - \mu (i, j | \ell , \ell_ {h} ^ {*} (s); P _ {h} (s)) \right| \right. \\ \leq \sqrt {\frac {4 S A L _ {\delta} ^ {k} \mu (i , j | s ; \ell , \ell_ {h} ^ {*} (s) ; P _ {h} (s))}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {2 S A L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \Bigg \} \\ \end{array}
$$

$$
E ^ {p v 1} (k) = \left\{\forall s, h: \left| \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \big ] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \big [ V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \big ] \right| \leq \sqrt {\frac {2 \mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} (V _ {h} ^ {*} (s , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \right\}
$$

$$
E ^ {p v 2} (k) = \left\{\forall s, h: \left| \sqrt {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} (V _ {h} ^ {*} (s , \boldsymbol {s} ^ {\prime}))} - \sqrt {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} (V _ {h} ^ {*} (s , \boldsymbol {s} ^ {\prime}))} \right| \leq 4 H \sqrt {\frac {L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} \right\}
$$

where we again use $L_{\delta}^{k} = \ln \frac{16S^3A^2Hk^2(k + 1)}{\delta}$ . We define the first good event as

$$
\mathbb {G} _ {1} = \bigcap_ {k \geq 1} E ^ {r} (k) \bigcap_ {k \geq 1} E ^ {\ell} (k) \bigcap_ {k \geq 1} E ^ {p v 1} (k) \bigcap_ {k \geq 1} E ^ {p v 2} (k),
$$

for which the following holds:

Lemma 12 (The First Good Event). It holds that $\Pr(\mathbb{G}_{1}) \geq 1 - \delta/2$ .

Proof. We prove that each of the events holds w.p. at least $1 - \delta/8$ . The result then directly follows by the union bound. We also remark that due to the domain of the variables and their estimators (e.g., [0, 1] for the rewards), all bounds trivially hold when the counts equal zero, so w.l.o.g., we only prove the results for cases in which states/state-actions were already previously visited.

Event $\cap_{k\geq 1}E^{r}(k)$ . Fix $k\geq 1,s,a,h$ and visits $n\geq 1$ . Given all of these, the reward observations are i.i.d. random variables supported by [0, 1]. Denoting the empirical mean based on these $n$ samples by $\hat{r}_h(s,a,n)$ , by Hoeffding's inequality, it holds w.p. $1 - \frac{\delta}{8SAHk^2(k + 1)}$ that

$$
| r _ {h} (s, a) - \hat {r} _ {h} (s, a, n) | \leq \sqrt {\frac {\ln \frac {1 6 S A H k ^ {2} (k + 1)}{\delta}}{2 n}} \leq \sqrt {\frac {L _ {\delta} ^ {k}}{n}}.
$$

Taking the union bound over all $n \in [k]$ at timestep $k$ , we get that w.p. $1 - \frac{\delta}{8SAHk(k + 1)}$

$$
| r _ {h} (s, a) - \hat {r} _ {h} ^ {k - 1} (s, a) | \leq \sqrt {\frac {L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s , a) \vee 1}},
$$

and another union bound over all possible values of $s, a, h$ and $k \geq 1$ implies that $\cap_{k \geq 1} E^r(k)$ holds w.p. at least $1 - \delta/8$ .

The event $\cap_{k\geq 1}E^{\ell}(k)$ . For any fixed $k\geq 1,s,h$ , a list $\ell \in \mathcal{L}$ and number of visits $n\in [k]$ , we utilize Lemma 16 (event $E^{p}$ ) w.r.t. the distribution $\mu (i,j|\ell ,\ell_h^* (s),P)$ (whose support is of size $M = (SA)^2$ ). When applying the lemma, notice that given the number of visits $n\geq 1$ , the empirical distribution $\hat{\mu}_h^{k - 1}(i,j|s;\ell ,\ell_h^* (s))$ is the average of $n = n_h^{k - 1}(s)$ i.i.d samples, so that for all $i,j\in [SA]$ ,

$$
\begin{array}{l} \left| \hat {\mu} _ {h} ^ {k - 1} (i, j | s; \ell , \ell_ {h} ^ {*} (s)) - \mu (i, j | \ell , \ell_ {h} ^ {*} (s); P _ {h} (s)) \right| \leq \sqrt {\frac {2 \mu (i , j | \ell , \ell_ {h} ^ {*} (s) ; P _ {h} (s)) \ln \frac {2 (S A) ^ {2}}{\delta^ {\prime}}}{n}} + \frac {2 \ln \frac {2 (S A) ^ {2}}{\delta^ {\prime}}}{3 n} \\ \leq \sqrt {\frac {4 \mu (i , j | \ell , \ell_ {h} ^ {*} (s) ; P _ {h} (s)) \ln \frac {2 S A}{\delta^ {\prime}}}{n}} + \frac {2 \ln \frac {2 S A}{\delta^ {\prime}}}{n} \\ \end{array}
$$

w.p. $1 - \delta'$ . Choosing $\delta' = \frac{\delta}{8|\mathcal{L}|SHk^2(k + 1)}$ (such that $\ln \frac{2SA}{\delta'} \leq SA\ln \frac{16S^3A^2Hk^2(k + 1)}{\delta}$ since $|\mathcal{L}| \leq (SA)^{SA}$ ), while taking the union bound on all $n \in [k]$ , all $s, h$ and all lists $\ell \in \mathcal{L}$ implies that $\cap_{k \geq 1} E^\ell(k)$ holds w.p. at least $1 - \frac{\delta}{8}$ .

Events $\cap_{k\geq1}E^{pv1}(k)$ and $\cap_{k\geq1}E^{pv2}(k)$ . We repeat the arguments stated in Lemma 5. For any fixed $k\geq1$ , s, h and number of visits $n\in[k]$ , we utilize Lemma 16 w.r.t. the next-state distribution for all actions $P_{h}(s)$ , the value $V_{h}^{*}(s,s^{\prime})\in[0,H]$ and probability $\delta'=\frac{\delta}{8SHk^{2}(k+1)}$ ; we yet again remind that given the number of visits, samples are i.i.d.

As before, the events $\cap_{k\geq 1}E^{pv1}(k)$ and $\cap_{k\geq 1}E^{pv2}(k)$ hold w.p. at least $1 - \frac{\delta}{8}$ through the union bound first on $n\in [k]$ (to get the empirical quantities) and then on $s,h$ and $k\geq 1$ . This proves that each of the events in $\mathbb{G}_1$ holds w.p. at least $1 - \frac{\delta}{8}$ , so $\mathbb{G}_1$ holds w.p. at least $1 - \frac{\delta}{2}$ .

# C.6 Optimism of the Upper Confidence Value Functions

We now prove that under the event $G_{1}$ , the values that MVP-TL outputs are optimistic.

Lemma 13 (Optimism). Under the first good event $\mathbb{G}_1$ , for all $k \in [K]$ , $h \in [H]$ , $a \in \mathcal{A}$ and $s, s' \in \mathcal{S}$ , it holds that $V_h^*(s, s', a) \leq \bar{V}_h^k(s, s', a)$ . Moreover, for all $\boldsymbol{s}' \in \mathcal{S}^A$ , $V_h^*(s, \boldsymbol{s}') \leq \bar{V}_h^k(s, \boldsymbol{s}')$ and also $V_h^*(s) \leq \bar{V}_h^k(s)$ .

Proof. The proof of all claims follows by backward induction on H; the base case naturally holds for $h = H + 1$ , where all values are defined to be zero.

Assume by induction that for some $k \in [K]$ and $h \in [H]$ , the inequality $V_{h+1}^{*}(s) \leq \bar{V}_{h+1}^{k}(s)$ holds for all $s \in S$ ; we will show that this implies that all stated inequalities also hold at timestep h. At this point, we also assume w.l.o.g. that $\hat{V}_{h}^{k}(s) < H$ (namely, not truncated), since otherwise, by the boundedness of the rewards, $V_{h}^{*}(s) \leq H = \bar{V}_{h}^{k}(s)$ . In particular, under the good event $E^{r}(k)$ , for all s and a, it holds that $\hat{r}_{h}^{k-1}(s,a) + b_{k,h}^{r}(s,a) \geq r_{h}(s,a)$ , so for all s, a and $s'$ , we have

$$
\bar {V} _ {h} ^ {k} (s, s ^ {\prime}, a) = \hat {r} _ {h} ^ {k - 1} (s, a) + b _ {k, h} ^ {r} (s, a) + \bar {V} _ {h + 1} ^ {k} (s ^ {\prime}) \geq r _ {h} (s, a) + V _ {h + 1} ^ {*} (s ^ {\prime}) = V _ {h} ^ {*} (s, s ^ {\prime}, a).
$$

where the inequality also uses the induction hypothesis. This proves the first part of the lemma. Moreover, it implies that

$$
\bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) = \max _ {a \in \mathcal {A}} \left\{\bar {V} _ {h} ^ {k} (s, s ^ {\prime} (a), a) \right\} \geq \max _ {a \in \mathcal {A}} \left\{V _ {h} ^ {*} (s, s ^ {\prime} (a), a) \right\} = V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}), \tag {13}
$$

and proves the second part of the statement.

To prove the last claim of the lemma, we use the monotonicity of the bonus, relying on Lemma 23. This lemma can be used when applied to the empirical distribution of all possible next-states $\hat{P}_h^{k - 1}(s)$ ; indeed, the non-truncated optimistic value can be written as

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s) = \mathbb {E} _ {\pmb {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \biggl [ \max _ {a \in \mathcal {A}} \bigl \{\hat {r} _ {h} ^ {k - 1} (s, a) + b _ {k, h} ^ {r} (s, a) + \bar {V} _ {h + 1} ^ {k} (s ^ {\prime} (a)) \bigr \} \biggr ] + b _ {k, h} ^ {p} (s) \\ \geq \mathbb {E} _ {\pmb {s ^ {\prime}} \sim \hat {P} _ {h} ^ {k - 1} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \pmb {s ^ {\prime}}) \big ] + \max \Bigg \{\frac {2 0}{3} \sqrt {\frac {\mathrm{Var} _ {\pmb {s ^ {\prime}} \sim \hat {P} _ {h} ^ {k - 1} (s)} (\bar {V} _ {h} ^ {k} (s , \pmb {s ^ {\prime}})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}}, \frac {4 0 0}{9} \frac {3 H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \Bigg \}, \\ \end{array}
$$

which is exactly the required form in Lemma 23, w.r.t. the distribution $\hat{P}_h^{k - 1}(s)$ and the values $\bar{V}_h^k (s,s')$ (while noticing that due to the truncation of the values and bonuses, $\bar{V}_h^k (s,s')\in [0,3H]$ ). Thus, the lemma guarantees monotonicity in the value, so by Equation (13),

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s) \geq \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \big [ V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \big ] + \max \left\{\frac {2 0}{3} \sqrt {\frac {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} (V _ {h} ^ {*} (s , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}}, \frac {4 0 0}{9} \frac {3 H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \right\} \\ \geq \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \big [ V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \big ] + \frac {1 0}{3} \sqrt {\frac {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} (V _ {h} ^ {*} (s , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {2 0 0}{3} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \\ \geq \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \left[ V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \right] + \frac {1 0}{3} \sqrt {\frac {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left(V _ {h} ^ {*} (s , \boldsymbol {s} ^ {\prime})\right) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {5 0 H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \quad (\text { Under } E ^ {p v 2} (k)) \\ \geq \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ V _ {h} ^ {*} \left(s, \boldsymbol {s} ^ {\prime}\right) \right] \quad (\text { Under   } E ^ {p v 1} (k)) \\ = V _ {h} ^ {*} (s). \\ \end{array}
$$

□

# C.7 The Second Good Event – Martingale Concentration

In this subsection, we present three good events that allow replacing the expectation over the randomizations inside each episode by their realization. Let

$$
Y _ {1, h} ^ {k} := \bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k})
$$

$$
Y _ {2, h} ^ {k} = \mathrm{Var} _ {\pmb {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k}, \pmb {s} ^ {\prime}))
$$

$$
Y _ {3, h} ^ {k} = b _ {k, h} ^ {r} (s _ {h} ^ {k}, a _ {h} ^ {k}).
$$

The second good event is the intersection of the events $\mathbb{G}_2 = E^{\mathrm{diff}}\cap E^{\mathrm{Var}}\cap E^{br}$ defined as follows.

$$
E ^ {\mathrm{diff}} = \left\{\forall h \in [ H ], K \geq 1: \sum_ {k = 1} ^ {K} \mathbb {E} [ Y _ {1, h} ^ {k} | F _ {k, h - 1} ] \leq \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} Y _ {1, h} ^ {k} + 1 8 H ^ {2} \ln \frac {6 H K (K + 1)}{\delta} \right\},
$$

$$
E ^ {\mathrm{Var}} = \Bigg \{K \geq 1: \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} Y _ {2, h} ^ {k} \leq 2 \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \mathbb {E} [ Y _ {2, h} ^ {k} | F _ {k - 1} ] + 4 H ^ {3} \ln \frac {6 H K (K + 1)}{\delta} \Bigg \},
$$

$$
E ^ {b r} = \left\{\forall h \in [ H ], K \geq 1: \sum_ {k = 1} ^ {K} \mathbb {E} \left[ Y _ {3, h} ^ {k} \mid F _ {k, h - 1} \right] \leq 2 \sum_ {k = 1} ^ {K} Y _ {3, h} ^ {k} + 1 8 \ln \frac {6 H K (K + 1)}{\delta} \right\},
$$

We define the good event $\mathbb{G} = \mathbb{G}_1\cap \mathbb{G}_2$

Lemma 14. The good event $\mathbb{G}$ holds with a probability of at least $1 - \delta$ .

Proof. The analysis of the first event follows $E^{\mathrm{diff}}$ exactly as the one of $E^{\mathrm{diff1}}$ in Lemma 7: define $W_{k} = \mathbb{1}\left\{\bar{V}_{h}^{k}(s) - V_{h}^{\pi^{k}}(s)\in [0,H],\forall h\in [H],s\in \mathcal{S}\right\}$ (which happens a.s. under $\mathbb{G}_1$ due to the optimism in Lemma 13 and truncation) and $\tilde{Y}_{1,h}^{k} = W_{k}Y_{1,h}^{k}$ , which is bounded in $[0,H]$ and $F_{k,h}$ -measurable. The corresponding event w.r.t. this modified variables $\tilde{E}^{\mathrm{diff}}$ then holds w.p. $1 - \frac{\delta}{6}$ by Lemma 25, and as in Lemma 7, we can use the fact that $\mathbb{G}_1\cap \tilde{E}^{\mathrm{diff}} = \mathbb{G}_1\cap E^{\mathrm{diff}}$ to conclude this part of the proof.

Moving to the second event, since $V_{h}^{\pi^{k}}(s, s') \in [0, H]$ , then $\sum_{h=1}^{H} Y_{2,h}^{k} \in [0, H^{3}]$ . Therefore, by Lemma 25 (w.r.t. the filtration $F_{k}$ ) with $C = H^{3}$ and any fixed K, we get w.p. $1 - \frac{\delta}{6HK(K+1)}$ that

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} Y _ {2, h} ^ {k} \leq 2 \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \mathbb {E} [ Y _ {2, h} ^ {k} | F _ {k - 1} ] + 4 H ^ {3} \ln \frac {6 H K (K + 1)}{\delta}.
$$

Taking the union bound on all possible values of $K \geq 1$ proves that $E^{Var}$ holds w.p. at least $1 - \frac{\delta}{6}$ . Finally, by definition, we have that $Y_{3,h}^{k} = b_{k,h}^{r}(s_{h}^{k}, a_{h}^{k}) \in [0, 1]$ and is $F_{k,h}$ -measurable. Thus, for any fixed $k \geq 1$ and $h \in [H]$ , using Lemma 25, we have w.p. $1 - \frac{\delta}{6HK(K+1)}$ that

$$
\sum_ {k = 1} ^ {K} \mathbb {E} [ Y _ {3, h} ^ {k} | F _ {k, h - 1} ] \leq \left(1 + \frac {1}{2}\right) \sum_ {k = 1} ^ {K} Y _ {3, h} ^ {k} + 1 8 \ln \frac {6 H K (K + 1)}{\delta} \leq 2 \sum_ {k = 1} ^ {K} Y _ {3, h} ^ {k} + 1 8 \ln \frac {6 H K (K + 1)}{\delta},
$$

so that due to the union bound, $E^{br}$ holds w.p. $1 - \frac{\delta}{6}$ .

To conclude, $G_{1}$ holds w.p. $1 - \frac{\delta}{2}$ (Lemma 5) and the events $\tilde{E}^{diff}, E^{Var}, E^{br}$ each hold w.p. $1 - \frac{\delta}{6}$ . As before, when accounting to the fact that $\tilde{E}^{diff}$ and $E^{diff}$ are identical under $G_{1}$ , the event $G = G_{1} \cap G_{2}$ holds w.p. at least $1 - \delta$ .

# C.8 Regret Analysis

Theorem 2. When running MVP-TL, with probability at least $1 - \delta$ uniformly for all $K \geq 1$ , it holds that $\operatorname{Reg}^T(K) \leq \mathcal{O}\left(\sqrt{H^2SK}\left(\sqrt{H} + \sqrt{A}\right)\ln \frac{SAHK}{\delta} + H^3S^4A^3\left(\ln \frac{SAHK}{\delta}\right)^2\right)$ .

Proof. Assume that the event G holds, which by Lemma 14, happens with probability at least $1 - \delta$ . In particular, throughout the proof, we use optimism (Lemma 13), which implies that $0 \leq V_{h}^{\pi^{k}}(s, s') \leq V_{h}^{*}(s, s') \leq \bar{V}_{h}^{k}(s, s') \leq 3H$ (the upper bound is also by the truncation), as well as $0 \leq V_{h}^{\pi^{k}}(s) \leq V_{h}^{*}(s) \leq \bar{V}_{h}^{k}(s) \leq H$ .

We first focus on lower-bounding the value of the policy $\pi^k$ : by Remark 2, we have

$$
\begin{array}{l} V _ {h} ^ {\pi^ {k}} (s) = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ r _ {h} \left(s, \pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right) + V _ {h + 1} ^ {\pi^ {k}} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) \right] \\ = \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \hat {r} _ {h} ^ {k - 1} \left(s, \pi_ {h} ^ {k} \left(s, \boldsymbol {s}\right)\right) + \bar {V} _ {h + 1} ^ {k} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) + b _ {k, h} ^ {r} \left(s, \pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right) \right] \\ + \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ r _ {h} \big (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) \big) - \hat {r} _ {h} ^ {k - 1} \big (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) \big) - b _ {k, h} ^ {r} \big (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) \big) \big ] \\ + \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ V _ {h + 1} ^ {\pi^ {k}} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) - \bar {V} _ {h + 1} ^ {k} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) \right] \\ \end{array}
$$

$$
\begin{array}{l} \stackrel {(1)} {=} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \max _ {a \in \mathcal {A}} \left\{\hat {r} _ {h} ^ {k - 1} (s, a) + \bar {V} _ {h + 1} ^ {k} \left(s ^ {\prime} (a)\right) + b _ {k, h} ^ {r} (s, a) \right\} \right] \\ + \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ r _ {h} \left(s, \pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right) - \hat {r} _ {h} ^ {k - 1} \left(s, \pi_ {h} ^ {k} \left(s, \boldsymbol {s}\right)\right) - b _ {k, h} ^ {r} \left(s, \pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right) \right] \\ + \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ V _ {h + 1} ^ {\pi^ {k}} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) - \bar {V} _ {h + 1} ^ {k} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) \right] \\ \stackrel {(2)} {\geq} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) \big ] - 2 \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ b _ {k, h} ^ {r} (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime})) \big ] \\ \left. \right. - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h + 1} ^ {k} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) - V _ {h + 1} ^ {\pi^ {k}} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right)\right] \\ \end{array}
$$

where (1) is by the definition of $\pi^k$ and (2) uses the reward concentration event. Thus, we can write

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s) - V _ {h} ^ {\pi^ {k}} (s) \leq \mathbb {E} _ {\pmb {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \pmb {s} ^ {\prime}) \big ] - \mathbb {E} _ {\pmb {s} ^ {\prime} \sim P _ {h} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \pmb {s} ^ {\prime}) \big ] + 2 \mathbb {E} _ {\pmb {s} ^ {\prime} \sim P _ {h} (s)} \big [ b _ {k, h} ^ {r} (s, \pi_ {h} ^ {k} (s, \pmb {s} ^ {\prime})) \big ] \\ + \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h + 1} ^ {k} (s ^ {\prime} (\pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}))) - V _ {h + 1} ^ {\pi^ {k}} (s ^ {\prime} (\pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}))) \right] + b _ {k, h} ^ {p} (s) \\ = \underbrace {\mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \left[ \bar {V} _ {h} ^ {k} \left(s , \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {*} \left(s , \boldsymbol {s} ^ {\prime}\right) \right] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h} ^ {k} \left(s , \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {*} \left(s , \boldsymbol {s} ^ {\prime}\right) \right] + b _ {k , h} ^ {p} (s)} _ {(i)} \\ + \underbrace {\mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} [ V _ {h} ^ {*} (s , \boldsymbol {s} ^ {\prime}) ] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} [ V _ {h} ^ {*} (s , \boldsymbol {s} ^ {\prime}) ]} _ {(i i)} + 2 \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ b _ {k, h} ^ {r} (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime})) \right] \\ + \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h + 1} ^ {k} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) - V _ {h + 1} ^ {\pi^ {k}} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) \right] \tag {14} \\ \end{array}
$$

Bounding term (ii): using the concentration event $E^{pv1}(k)$ , we have

$$
(i i) \leq \sqrt {\frac {2 \operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left(V _ {h} ^ {*} \left(s , \boldsymbol {s} ^ {\prime}\right)\right) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}
$$

$$
\stackrel {(1)} {\leq} \sqrt {\frac {2 \operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left(V _ {h} ^ {\pi^ {k}} \left(s , \boldsymbol {s} ^ {\prime}\right)\right) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {1}{8 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ V _ {h} ^ {\pi^ {k}} \left(s, \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {\pi_ {k}} \left(s, \boldsymbol {s} ^ {\prime}\right) \right] + \frac {4 H ^ {2} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} + \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}
$$

$$
\stackrel {(2)} {\leq} \sqrt {\frac {2 \operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left(V _ {h} ^ {\pi^ {k}} \left(s , \boldsymbol {s} ^ {\prime}\right)\right) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {1}{8 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {\pi_ {k}} \left(s, \boldsymbol {s} ^ {\prime}\right) \right] + \frac {5 H ^ {2} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}. \tag {15}
$$

Relation (1) uses Lemma 21 with the values $0 \leq V_h^{\pi^k}(s, \pmb{s}') \leq V_h^*(s, \pmb{s}') \leq H$ with $\alpha = 8H \cdot \sqrt{2L_{\delta}^{k}}$ and (2) is by optimism.

Bounding term (i): We first focus on the transition bonus; to bound it, we apply Lemma 22 w.r.t. $\hat{P}_h^{k-1}(\boldsymbol{s}'|s), P_h(\boldsymbol{s}'|s)$ , the values $0 \leq V_h^{\pi^k}(s, \boldsymbol{s}') \leq V_h^*(s, \boldsymbol{s}') \leq \bar{V}_h^k(s, \boldsymbol{s}') \leq 3H$ (by optimism), under the event $E^{pv2}(k)$ and with $\alpha = 8H \cdot \frac{20}{3}\sqrt{L_\delta^k}$ :

$$
\begin{array}{l} b _ {k, h} ^ {p} (s) = \frac {2 0}{3} \sqrt {\frac {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} (\bar {V} _ {h} ^ {k} (s , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {4 0 0}{3} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \\ \leq \frac {1}{8 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \left[ \bar {V} _ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {*} \left(s, \boldsymbol {s} ^ {\prime}\right) \right] + \frac {1}{8 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ V _ {h} ^ {*} \left(s, \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {\pi_ {k}} \left(s, \boldsymbol {s} ^ {\prime}\right) \right] \\ + \frac {2 0}{3} \sqrt {\frac {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left(V _ {h} ^ {\pi^ {k}} \left(s , \boldsymbol {s} ^ {\prime}\right)\right) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {1 6 0 0 H ^ {2}}{3 n _ {h} ^ {k - 1} (s) \vee 1} + \frac {2 0}{3} \frac {4 H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} + \frac {4 0 0}{3} \frac {H L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \\ \leq \frac {1}{8 H} \left(\mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \left[ \bar {V} _ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {*} \left(s, \boldsymbol {s} ^ {\prime}\right) \right] - E _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {*} \left(s, \boldsymbol {s} ^ {\prime}\right) \right]\right) \\ + \frac {1}{8 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) - V _ {h} ^ {\pi_ {k}} (s, \boldsymbol {s} ^ {\prime}) \big ] + \frac {2 0}{3} \sqrt {\frac {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} (V _ {h} ^ {\pi^ {k}} (s , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {7 0 0 H ^ {2}}{n _ {h} ^ {k - 1} (s) \vee 1}. \\ \end{array}
$$

Substituting back to term $(i)$ , we now have

$$
\begin{array}{l} (i) \leq \left(1 + \frac {1}{8 H}\right) \left(\mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \left[ \bar {V} _ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {*} \left(s, \boldsymbol {s} ^ {\prime}\right) \right] - E _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {*} \left(s, \boldsymbol {s} ^ {\prime}\right) \right]\right) \\ + \frac {1}{8 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) - V _ {h} ^ {\pi_ {k}} (s, \boldsymbol {s} ^ {\prime}) \big ] + \frac {2 0}{3} \sqrt {\frac {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} (V _ {h} ^ {\pi^ {k}} (s , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {7 0 0 H ^ {2} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}. \\ \end{array}
$$

The next step in the proof involves bounding the first term of (i). At this point, we remind that both values can be written as $\bar{V}_{h}^{k}(s,\boldsymbol{s}^{\prime})=\max_{a}\bar{V}_{h}^{k}(s,s^{\prime}(a),a)$ and $V_{h}^{*}(s,\boldsymbol{s}^{\prime})=\max_{a}V_{h}^{*}(s,s^{\prime}(a),a)$ , inducing the lists $\bar{\ell}=\bar{\ell}_{h}^{k}(s)$ and $\ell^{*}=\ell_{h}^{*}(s)$ , respectively; thus the expectations can be written as (see Appendix C.4 for further details on the list representation, and in particular, Equation (11)):

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim \hat {P} _ {h} ^ {k - 1} (s)} \left[ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) - V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \right] - \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) - V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \right] \\ \stackrel {(1)} {=} \mathbb {E} _ {i, j \sim \hat {\mu} _ {h} ^ {k} (\cdot | s; \bar {\ell}, \ell^ {*})} \left[ \bar {V} _ {h} ^ {k} (s, s _ {\bar {\ell} (i)} ^ {\prime}, a _ {\bar {\ell} (i)}) - V _ {h} ^ {*} (s, s _ {\ell^ {*} (j)} ^ {\prime}, a _ {\ell^ {*} (j)}) \right] \\ - \mathbb {E} _ {i, j \sim \mu (\cdot | \bar {\ell}, \ell^ {*}, P _ {h} (s))} \left[ \bar {V} _ {h} ^ {k} (s, s _ {\bar {\ell} (i)} ^ {\prime}, a _ {\bar {\ell} (i)}) - V _ {h} ^ {*} (s, s _ {\ell^ {*} (j)} ^ {\prime}, a _ {\ell^ {*} (j)}) \right] \\ \stackrel {(2)} {\leq} \frac {1}{8 H} \mathbb {E} _ {i, j \sim \mu (\cdot | \bar {\ell}, \ell^ {*}, P _ {h} (s))} \left[ \bar {V} _ {h} ^ {k} (s, s _ {\bar {\ell} (i)} ^ {\prime}, a _ {\bar {\ell} (i)}) - V _ {h} ^ {*} (s, s _ {\ell^ {*} (j)} ^ {\prime}, a _ {\ell^ {*} (j)}) \right] + \frac {3 H (S A) ^ {2} L _ {\delta} ^ {k} (2 S A + 8 H \cdot 4 S A / 4)}{n _ {h} ^ {k - 1} (s) \vee 1} \\ \stackrel {(1)} {\leq} \frac {1}{8 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) - V _ {h} ^ {*} (s, \boldsymbol {s} ^ {\prime}) \right] + \frac {3 0 H ^ {2} (S A) ^ {3} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \\ \leq \frac {1}{8 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) - V _ {h} ^ {\pi^ {k}} (s, \boldsymbol {s} ^ {\prime}) \right] + \frac {3 0 H ^ {2} (S A) ^ {3} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \\ \end{array}
$$

Relations (1) formulate the expectation using the list representations and backward, as done in Equation (11). For inequality (2) we rely on Lemma 24 with $\alpha = 8H$ under the event $E^{\ell}(k)$ and the optimism, which ensures that the value difference is bounded in $[0,3H]$ . We also remark that the support of the distributions is of size $(SA)^{2}$ ; were we to use the same result on the distributions $\hat{P}_h^{k - 1}(s)$ and $P_{h}(s)$ , the support would be of size $S^A$ , which would lead to an exponential additive factor. And so, we finally have a bound of

$$
(i) \leq \frac {3}{8 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right) - V _ {h} ^ {\pi_ {k}} \left(s, \boldsymbol {s} ^ {\prime}\right) \right] + \frac {2 0}{3} \sqrt {\frac {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left(V _ {h} ^ {\pi^ {k}} \left(s , \boldsymbol {s} ^ {\prime}\right)\right) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {7 3 5 H ^ {2} (S A) ^ {3} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}. \tag {16}
$$

Combining both terms. Substituting this and Equation (15) into Equation (14), we have

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s) - V _ {h} ^ {\pi^ {k}} (s) \leq \frac {1}{2 H} \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}) - V _ {h} ^ {\pi_ {k}} (s, \boldsymbol {s} ^ {\prime}) \big ] + 9 \sqrt {\frac {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} (V _ {h} ^ {\pi^ {k}} (s , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {7 5 0 H ^ {2} (S A) ^ {3} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} \\ + 2 \mathbb {E} _ {\boldsymbol {s ^ {\prime}} \sim P _ {h} (s)} \big [ b _ {k, h} ^ {r} (s, \pi_ {h} ^ {k} (s, \boldsymbol {s ^ {\prime}})) \big ] + \mathbb {E} _ {\boldsymbol {s ^ {\prime}} \sim P _ {h} (s)} \Big [ \bar {V} _ {h + 1} ^ {k} (s ^ {\prime} (\pi_ {h} ^ {k} (s, \boldsymbol {s ^ {\prime}}))) - V _ {h + 1} ^ {\pi^ {k}} (s ^ {\prime} (\pi_ {h} ^ {k} (s, \boldsymbol {s ^ {\prime}}))) \Big ]. \\ \end{array}
$$

and further bounding (using the concentration event $E^{r}(k)$ )

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime})) - V _ {h} ^ {\pi^ {k}} (s, \boldsymbol {s} ^ {\prime}) = \hat {r} _ {h} ^ {k - 1} (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime})) + b _ {k, h} ^ {r} (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime})) + \bar {V} _ {h + 1} ^ {k} (s ^ {\prime} (\pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}))) \\ - r _ {h} ^ {k - 1} (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime})) - V _ {h + 1} ^ {\pi^ {k}} (s ^ {\prime} (\pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}))) \\ \leq \bar {V} _ {h + 1} ^ {k} (s ^ {\prime} (\pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}))) - V _ {h + 1} ^ {\pi^ {k}} (s ^ {\prime} (\pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime}))) + 2 b _ {k, h} ^ {r} (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime})), \\ \end{array}
$$

we finally get the decomposition

$$
\begin{array}{l} \bar {V} _ {h} ^ {k} (s) - V _ {h} ^ {\pi^ {k}} (s) \leq \left(1 + \frac {1}{2 H}\right) \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \left[ \bar {V} _ {h + 1} ^ {k} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) - V _ {h + 1} ^ {\pi^ {k}} \left(s ^ {\prime} \left(\pi_ {h} ^ {k} \left(s, \boldsymbol {s} ^ {\prime}\right)\right)\right) \right] \\ + 9 \sqrt {\frac {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} (V _ {h} ^ {\pi^ {k}} (s , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1}} + \frac {7 5 0 H ^ {2} (S A) ^ {3} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s) \vee 1} + 3 \mathbb {E} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s)} \big [ b _ {k, h} ^ {r} (s, \pi_ {h} ^ {k} (s, \boldsymbol {s} ^ {\prime})) \big ]. \\ \end{array}
$$

At this point, we choose to take $s = s_h^k$ and sum over all $k \in [K]$ ; specifically, for $s' = s_{h+1}'^k$ , the action becomes $\pi_h^k(s, s') = a_h^k$ and $s'(\pi_h^k(s, s')) = s_{h+1}^k$ . Formally, we can write the bound as

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \bar {V} _ {h} ^ {k} (s _ {h} ^ {k}) - V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k}) \leq \left(1 + \frac {1}{2 H}\right) \sum_ {k = 1} ^ {K} \mathbb {E} \Big [ \bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k}) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k}) | F _ {k, h - 1} \Big ] \\ + 3 \sum_ {k = 1} ^ {K} \mathbb {E} \big [ b _ {k, h} ^ {r} (s _ {h} ^ {k}, a _ {h} ^ {k}) | F _ {k, h - 1} \big ] + 9 \sum_ {k = 1} ^ {K} \sqrt {\frac {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k} , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}} \\ + \sum_ {k = 1} ^ {K} \frac {7 5 0 H ^ {2} (S A) ^ {3} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}. \\ \end{array}
$$

and, in particular, under the events $E^{diff}$ and $E^{br}$ , it holds that

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \bar {V} _ {h} ^ {k} (s _ {h} ^ {k}) - V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k}) \leq \left(1 + \frac {1}{2 H}\right) ^ {2} \sum_ {k = 1} ^ {K} \Bigl (\bar {V} _ {h + 1} ^ {k} (s _ {h + 1} ^ {k})) - V _ {h + 1} ^ {\pi^ {k}} (s _ {h + 1} ^ {k}) \Bigr) + 3 6 H ^ {2} \ln \frac {6 H K (K + 1)}{\delta} \\ + 3 \sum_ {k = 1} ^ {K} b _ {k, h} ^ {r} (s _ {h} ^ {k}, a _ {h} ^ {k}) + 5 4 \ln \frac {6 H K (K + 1)}{\delta} \\ + 9 \sum_ {k = 1} ^ {K} \sqrt {\frac {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k} , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}} + \sum_ {k = 1} ^ {K} \frac {7 5 0 H ^ {2} (S A) ^ {3} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}. \\ \end{array}
$$

To conclude the proof, we recursively apply this formula from $h = 1$ to $h = H + 1$ (where the values are zero) and use the optimism. This yields

$$
\begin{array}{l} \operatorname{Reg} ^ {T} (K) = \sum_ {k = 1} ^ {K} V _ {1} ^ {*} (s _ {h} ^ {k}) - V _ {1} ^ {\pi^ {k}} (s _ {h} ^ {k}) \\ \leq \sum_ {k = 1} ^ {K} \bar {V} _ {1} ^ {k} (s _ {h} ^ {k}) - V _ {1} ^ {\pi^ {k}} (s _ {h} ^ {k}) \tag {Optimism} \\ \stackrel {(1)} {\leq} 9 \left(1 + \frac {1}{2 H}\right) ^ {2 H} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {\sqrt {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k} , \boldsymbol {s} ^ {\prime})) L _ {\delta} ^ {k}}}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}} \\ + 3 \left(1 + \frac {1}{2 H}\right) ^ {2 H} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \sqrt {\frac {L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} \\ + \left(1 + \frac {1}{2 H}\right) ^ {2 H} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {7 5 0 H ^ {2} (S A) ^ {3} L _ {\delta} ^ {k}}{n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1} + 9 0 H ^ {3} \left(1 + \frac {1}{2 H}\right) ^ {2 H} \ln \frac {6 H K (K + 1)}{\delta} \\ \stackrel {(2)} {\leq} 5 0 \sqrt {H ^ {3} S K} L _ {\delta} ^ {K} + 5 0 \sqrt {2 S} H ^ {2} \left(L _ {\delta} ^ {K}\right) ^ {1. 5} \\ + 9 \sqrt {L _ {\delta} ^ {K}} \left(S A H + 2 \sqrt {S A H ^ {2} K}\right) + 2 0 5 0 H ^ {3} S ^ {4} A ^ {3} L _ {\delta} ^ {K} (2 + \ln (K)) + 2 5 0 H ^ {3} L _ {\delta} ^ {K} \\ = \mathcal {O} \left(\sqrt {H ^ {2} S K} (\sqrt {H} + \sqrt {A}) L _ {\delta} ^ {K} + H ^ {3} S ^ {4} A ^ {3} (L _ {\delta} ^ {K}) ^ {2}\right). \\ \end{array}
$$

Relation (1) is the recursive application of the difference alongside substitution of the reward bonuses, while relation (2) is by Lemma 15 and Lemma 20.

# C.8.1 Lemmas for Bounding Bonus Terms

Lemma 15. Under the event $E^{\mathrm{Var}}$ it holds that

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {\sqrt {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k} , \boldsymbol {s} ^ {\prime}))}}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}} \leq 2 \sqrt {H ^ {3} S K L _ {\delta} ^ {K}} + \sqrt {8 S} H ^ {2} L _ {\delta} ^ {K}.
$$

Proof. Similar to Lemma 9, we again rely on the lookahead version of the law of total variation to prove this bound. First, by Cauchy-Schwartz inequality, it holds that

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {\sqrt {\operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k} , \boldsymbol {s} ^ {\prime}))}}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}} \leq \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k} , \boldsymbol {s} ^ {\prime}))} \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}}.
$$

We use Lemma 20 to bound the second term by

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1} \leq S H (2 + \ln (K))
$$

and focus on bounding the first term. Under $E^{Var}$ , we have

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k}, \boldsymbol {s} ^ {\prime})) \\ \leq 2 \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \sum_ {h = 1} ^ {H} \operatorname{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k}, \boldsymbol {s} ^ {\prime})) | F _ {k - 1} \right] + 4 H ^ {3} \ln \frac {6 H K (K + 1)}{\delta} \quad (\text { Under   } E ^ {\text { Var }}) \\ = 2 \sum_ {k = 1} ^ {K} \mathbb {E} \left[ \left(\sum_ {h = 1} ^ {H} r _ {h} (s _ {h} ^ {k}, a _ {h} ^ {k}) - V _ {1} ^ {\pi^ {k}} (s _ {1} ^ {k})\right) ^ {2} | F _ {k - 1} \right] + 4 H ^ {3} \ln \frac {6 H K (K + 1)}{\delta} \quad (\text { By   Lemma   10 }) \\ \leq 2 H ^ {2} K + 4 H ^ {3} \ln \frac {6 H K (K + 1)}{\delta}, \\ \end{array}
$$

where the last inequality is since both the values and cumulative rewards are bounded in $[0, H]$ . Combining both, we get

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {\sqrt {\mathrm{Var} _ {\boldsymbol {s} ^ {\prime} \sim P _ {h} (s _ {h} ^ {k})} (V _ {h} ^ {\pi^ {k}} (s _ {h} ^ {k} , \boldsymbol {s} ^ {\prime}))}}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}} \leq \sqrt {2 H ^ {2} K + 4 H ^ {3} \ln \frac {6 H K (K + 1)}{\delta}} \sqrt {S H (2 + \ln (K))} \\ \leq \sqrt {2 H ^ {2} K + 4 H ^ {3} \ln \frac {6 H K (K + 1)}{\delta}} \sqrt {2 S H \ln \frac {6 H K (K + 1)}{\delta}} \\ \leq 2 \sqrt {H ^ {3} S K L _ {\delta} ^ {K}} + \sqrt {8 S} H ^ {2} L _ {\delta} ^ {K}. \\ \end{array}
$$

![](images/880234d0314d395889e38e07933fec0493f9a58d9329eb4567e6fa7555300930.jpg)

# C.9 Example: Value Gain due to Transition Lookahead

![](images/abdc284ecb8a95b77318ec7a8a889673e5e44ed42db55094de80543966d76867.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["r = 0"] -->|w.p. 1 - 1/(A-1)| B["r = 0"]
    B -->|w.p. 1 - 1/(A-1)| C["r = 0"]
    C -->|w.p. 1 - 1/(A-1)| D["r = 0"]
    D -->|w.p. 1 - 1/(A-1)| E["r = 0"]
    E -->|w.p. 1 - 1/(A-1)| F["r = 0"]
    F -->|w.p. 1 - 1/(A-1)| G["r = 0"]
    G -->|w.p. 1 - 1/(A-1)| H["r = 0"]
    H -->|w.p. 1 - 1/(A-1)| I["r = 0"]
    I -->|w.p. 1 - 1/(A-1)| J["r = 0"]
    J -->|w.p. 1 - 1/(A-1)| K["r = 0"]
    K -->|w.p. 1 - 1/(A-1)| L["r = 0"]
    L -->|w.p. 1 - 1/(A-1)| M["r = 0"]
    M -->|w.p. 1 - 1/(A-1)| N["r = 0"]
    N -->|w.p. 1 - 1/(A-1)| O["r = 0"]
    O -->|w.p. 1 - 1/(A-1)| P["r = 0"]
    P -->|w.p. 1 - 1/(A-1)| Q["r = 0"]
    Q -->|w.p. 1 - 1/(A-1)| R["r = 0"]
    R -->|w.p. 1 - 1/(A-1)| S["r = 0"]
    S -->|w.p. 1 - 1/(A-1)| T["r = 0"]
    T -->|w.p. 1 - 1/(A-1)| U["r = 0"]
    U -->|w.p. 1 - 1/(A-1)| V["r = 0"]
    V -->|w.p. 1 - 1/(A-1)| W["r = 0"]
    W -->|w.p. 1 - 1/(A-1)| X["r = 0"]
    X -->|w.p. 1 - 1/(A-1)| Y["r = 0"]
    Y -->|w.p. 1 - 1/(A-1)| Z["r = 0"]
    Z -->|w.p. 1 - 1/(A-1)| AA["r = 0"]
    AA -->|w.p. 1 - 1/(A-1)| AB["r = 0"]
    AB -->|w.p. 1 - 1/(A-1)| AC["r = 0"]
    AC -->|w.p. 1 - 1/(A-1)| AD["r = 0"]
    AD -->|w.p. 1 - 1/(A-1)| AE["r = 0"]
    AE -->|w.p. 1 - 1/(A-1)| AF["r = 0"]
    AF -->|w.p. 1 - 1/(A-1)| AG["r = 0"]
    AG -->|w.p. 1 - 1/(A-1)| AH["r = 0"]
    AH -->|w.p. 1 - 1/(A-1)| AI["r = 0"]
    AI -->|w.p. 1 - 1/(A-1)| AJ["r = 0"]
    AJ -->|w.p. 1 - 1/(A-1)| AK["r = 0"]
    AK -->|w.p. 1 - 1/(A-1)| AL["r = 0"]
    AL -->|w.p. 1 - 1/(A-1)| AM["r = 0"]
    AM -->|w.p. 1 - 1/(A-1)| AN["r = 0"]
    AN -->|w.p. 1 - 1/(A-1)| AO["r = 0"]
    AO -->|w.p. 1 - 1/(A-1)| AP["r = 0"]
    AP -->|w.p. 1 - 1/(A-1)| AQ["r = 0"]
    AQ -->|w.p. 1 - 1/(A-1)| AR["r = 0"]
    AR -->|w.p. 1 - 1/(A-1)| AS["r = 0"]
    AS -->|w.p. 1 - 1/(A-1)| AT["r = 0"]
    AT -->|w.p. 1 - 1/(A-1)| AU["r = 0"]
    AU -->|w.p. 1 - 1/(A-1)| AV["r = 0"]
    AV -->|w.p. 1 - 1/(A-1)| AW["r = 0"]
    AW -->|w.p. 1 - 1/(A-1)| AX["r = 0"]
    AX -->|w.p. 1 - 1/(A-1)| AY["r = 0"]
    AY -->|w.p. 1 / (A-2) | AZ["r = 0"]
    AZ -->|w.p. 2 / (A-3) | BA["r = 0"]
    BA -->|w.p. 3 / (A-4) | BB["r = 0"]
    BB -->|w.p. 4 / (A-5) | BC["r = 0"]
    BC -->|w.p. 5 / (A-6) | BD["r = 0"]
    BD -->|w.p. 6 / (A-7) | BE["r = 0"]
    BE -->|w.p. 7 / (A-8) | BF["r = 0"]
    BF -->|w.p. 8 / (A-9) | BG["r = 0"]
    BG -->|w.p. 9 / (A-10) | BH["r = 0"]
    BH -->|w.p. 10 / (A-11) | BI["r = 0"]
    BI -->|w.p. 11 / (A-12) | BJ["r = 0"]
    BJ -->|w.p. 12 / (A-13) | BK["r = 0"]
    BK -->|w.p. 13 / (A-14) | BL["r = 0"]
    BL -->|w.p. 14 / (A-15) | BM["r = 0"]
    BM -->|w.p. 15 / (A-16) | BN["r = 0"]
    BN -->|w.p. 16 / (A-17) | BO["r = 0"]
    BO -->|w.p. 17 / (A-18) | BP["r = 0"]
    BP -->|w.p. 18 / (A-19) | BQ["r = 0"]
    BQ -->|w.p. 2/ (B-A) | BR["r = 0"]
```
</details>

Figure 2: Random chain: agents start at the left side and must reach its right side to collect a reward.

We now present in further detail the example described at Section 3. This example is inspired by the one in Appendix C.3 in [Merlis et al., 2024], greatly simplifying it and achieving similar behavior for a much smaller environment.

Agents start at the left side of a chain of length H/2 (depicted in Figure 2) and have two options:

1. Play a safe action $a_{1}$ that leaves the agent in the same state (in green), or,   
2. play one of the $A - 1$ risky actions $a_2, \ldots, a_A$ (in red). Each of these actions moves the agent forward in the chain w.p. $\frac{1}{A - 1}$ , but leads to a terminal non-rewarding state w.p. $1 - \frac{1}{A - 1}$ .

At the end of the chain, the last state is an absorbing state with a unit reward.

Without lookahead, all agents can do is try to randomly reach the end of the chain, succeeding with probability $(A-1)^{-H/2}$ . In particular, such agents cannot collect more than $V^{no} \leq H(A-1)^{-H/2}$ . On the other hand, with transition lookahead, agents observe whether the risky actions allow moving forward in the chain or lead to the bad terminal state. If one action allows progressing in the chain (which happens w.p. $p = 1 - \left(1 - \frac{1}{A-1}\right)^{A-1} \geq 1 - \frac{1}{e}$ ), a lookahead agent would take it, and otherwise, they will use $a_{1}$ to remain in the same state. In other words, optimal lookahead agents reach the reward after H/2-1 successful 'progression steps' with probability p each. The probability of reaching the end of the chain using less than 5H/6 steps is at least

$$
\operatorname * {P r} \left(\operatorname{Bin} \left(\frac {5 H}{6} - 1, 1 - \frac {1}{e}\right) > \frac {H}{2}\right) \geq c _ {0}, \quad \text { for   some   absolute } c _ {0} > 0.
$$

Under this event, the agent collects $\frac{H}{6}$ rewards, so the lookahead value is at least $V^{T,*}\geq\frac{c_{0}H}{6}=\Omega(H)$ .

To summarize, for this example, no lookahead optimal value is at most $\approx HA^{-H/2}$ , while transition lookahead agents can collect a value of $\approx H$ : transition lookahead increases the value by an exponential multiplicative factor. The difference between the two values is $G^{T} = \Omega(H)$ , and following the discussion in Section 3, a sublinear transition lookahead regret would imply a negatively linear standard regret of $\operatorname{Reg}(K) \lesssim -HK$ .

Remark 3. The chain length was chosen to be H/2 for simplicity – similar conclusions can be achieved for a length of $\approx 1 - \frac{1}{e}$ . Then, the multiplicative increase in value due to transition lookahead would be $\approx (A - 1)^{(1 - \frac{1}{e})H}$ , matching Proposition 2 in [Merlis et al., 2024]. In fact, setting the transition from the last state of the chain to the terminal state (rendering it possible to earn only one unit of reward), the analysis coincides with the one in [Merlis et al., 2024]. Following their exact derivation, the value with lookahead information is multiplicatively larger than its no-lookahead factor by an exponential factor of $\Theta\left((A - 1)^{\min\left\{(1 - \frac{1}{e})H - 1,S\right\} - 2}\right)$ . This significantly improves the result in [Merlis et al., 2024], that only holds if $S \geq A^{\left(1 - \frac{1}{e}\right)H}$ .

# D Auxiliary Lemmas

In this appendix, we prove various auxiliary lemma that will be used throughout our proofs.

# D.1 Concentration results

We first present and reprove a set of well-known concentration results.

Lemma 16. Let P be a distribution over a discrete set X of size $|X| = M$ and let $X, X_{1}, \ldots, X_{n}$ be independent samples from this distribution. Also, let $U : X \mapsto [0, C]$ for some C > 0 and define the empirical distribution $\hat{P}_{n}(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}\{x_{i} = x\}$ . Then, for any $\delta \in (0, 1)$ , each of the following events hold w.p. at least $1 - \delta$ :

$$
\begin{array}{l} E ^ {p} = \left\{\forall x \in \mathcal {X}, | P (x) - \hat {P} _ {n} (x) | \leq \sqrt {\frac {2 P (x) \ln \frac {2 M}{\delta}}{n}} + \frac {2 \ln \frac {2 M}{\delta}}{3 n} \right\} \\ E ^ {p v 1} = \left\{\left| \sum_ {x \in \mathcal {X}} \Bigl (\hat {P} _ {n} (x) - P (x) \Bigr) U (x) \right| \leq \sqrt {\frac {2 \mathrm{Var} _ {P} (U (X)) \ln \frac {2}{\delta}}{n}} + \frac {2 C \ln \frac {2}{\delta}}{3 n} \right\} \\ E ^ {p v 2} = \left\{\left| \sqrt {\mathrm{Var} _ {\hat {P} _ {n}} (U (X))} - \sqrt {\mathrm{Var} _ {P} (U (X))} \right| \leq 4 C \sqrt {\frac {\ln \frac {2}{\delta}}{n \vee 1}} \right\}, \\ \end{array}
$$

where $\operatorname{Var}_P(U(X)) = \sum_{x\in \mathcal{X}}P(x)U(x)^2 -\left(\sum_{x\in \mathcal{X}}P(x)U(x)\right)^2.$

Proof. All the results require standard probability arguments and are stated for completeness.

For the first event $E^p$ , notice that each of the components $\hat{P}_n(x)$ is the empirical mean of independent Bernoulli random variables $X_i(x)$ of mean $P(x)$ . Therefore, by Bernstein's inequality, recalling that the variance of the variable $Ber(p)$ is $p(1 - p)$ , we get w.p. at least $1 - \frac{\delta}{M}$ that

$$
| P (x) - \hat {P} _ {n} (x) | \leq \sqrt {\frac {2 P (x) (1 - P (x)) \ln \frac {2 M}{\delta}}{n}} + \frac {2 \ln \frac {2 M}{\delta}}{3 n} \leq \sqrt {\frac {2 P (x) \ln \frac {2 M}{\delta}}{n}} + \frac {2 \ln \frac {2 M}{\delta}}{3 n}.
$$

Taking the union bound over all $x \in X$ implies that $E^{p}$ holds w.p. at least $1 - \delta$ .

For the second event $E^{pv1}$ , we apply Bernstein's inequality on the variables $Y_{i} = U(X_{i})$ . The empirical mean is given by $\hat{Y}_n = \frac{1}{n}\sum_i U(X_i) = \sum_{x\in \mathcal{X}}\hat{P}_n(x)U(x)$ and its average is $\mathbb{E}[Y] = \sum_{x\in \mathcal{X}}P(x)U(x)$ . Similarly, the variance of the random variables is $\mathrm{Var}(Y) = \mathrm{Var}_P(U(X))$ . Thus, by Bernstein's inequality, w.p. at least $1 - \delta$ ,

$$
\left| \hat {Y} _ {n} - \mathbb {E} [ Y ] \right| \leq \sqrt {\frac {2 \mathrm{Var} (Y) \ln \frac {2}{\delta}}{n}} + \frac {2 C \ln \frac {2}{\delta}}{3 n}.
$$

Stating the bounds in terms of $X_{i}$ leads to the second event.

For the last event, we follow the analysis of [Efroni et al., 2021, Lemma 19], which in turn, relies on [Maurer and Pontil, 2009, Theorem 10]. Define $V_{n} = \frac{1}{2n(n - 1)}\sum_{i,j = 1}^{n}(U(X_{i}) - U(X_{j}))^{2}$ . This is a well-known unbiased variance estimator, namely, $\mathbb{E}[V_n] = \mathrm{Var}_P(U(X))$ , and by [Maurer and Pontil, 2009, Theorem 10], for any $\delta >0$ it holds w.p. at least $1 - \delta$ that

$$
\left| \sqrt {V _ {n}} - \sqrt {\operatorname{Var} _ {P} (U (X))} \right| \leq C \sqrt {\frac {2 \ln \frac {2}{\delta}}{n - 1}},
$$

where we scaled the bound by C to account for the values being in $[0, C]$ .

Next, we relate $V_{n}$ to the empirical variance. By elementary algebra, we have

$$
\begin{array}{l} V _ {n} = \frac {1}{2 n (n - 1)} \sum_ {i, j = 1} ^ {n} \left(U (X _ {i}) - U (X _ {j})\right) ^ {2} \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} U (X _ {i}) ^ {2} - \frac {1}{n (n - 1)} \sum_ {i \neq j} U (X _ {i}) U (X _ {j}) \\ = \frac {1}{n} \sum_ {i = 1} ^ {n} U (X _ {i}) ^ {2} - \frac {n}{(n - 1)} \left(\frac {1}{n} \sum_ {i} U (X _ {i})\right) ^ {2} + \frac {1}{n (n - 1)} \sum_ {i = 1} ^ {n} U (X _ {i}) ^ {2} \\ = \sum_ {x \in \mathcal {X}} \hat {P} _ {n} (x) U (x) ^ {2} - \left(\sum_ {x \in \mathcal {X}} \hat {P} _ {n} (x) U (x)\right) ^ {2} + \frac {1}{n (n - 1)} \sum_ {i = 1} ^ {n} U (X _ {i}) ^ {2} - \frac {1}{n ^ {2} (n - 1)} \left(\sum_ {i = 1} ^ {n} U (X _ {i})\right) ^ {2}. \\ \end{array}
$$

The first two terms are exactly the variance w.r.t. the empirical distribution; therefore, using the inequality $\left|\sqrt{a} - \sqrt{b}\right| \leq \sqrt{|a - b|}$ for positive numbers, we have

$$
\left| \sqrt {V _ {n}} - \sqrt {\operatorname{Var} _ {\hat {P} _ {n}} (U (X))} \right| \leq \sqrt {\left| \frac {1}{n (n - 1)} \sum_ {i = 1} ^ {n} U (X _ {i}) ^ {2} - \frac {1}{n ^ {2} (n - 1)} \left(\sum_ {i = 1} ^ {n} U (X _ {i})\right) ^ {2} \right|} \leq \sqrt {\frac {C ^ {2}}{n - 1}}.
$$

Combining both inequalities and recalling the trivial bound of $C$ on the difference, we get that w.p. at least $1 - \delta$ ,

$$
\left| \sqrt {\operatorname{Var} _ {\hat {P} _ {n}} (U (X))} - \sqrt {\operatorname{Var} _ {P} (U (X))} \right| \leq \min \left\{C \sqrt {\frac {2 \ln \frac {2}{\delta}}{n - 1}} + \sqrt {\frac {C ^ {2}}{n - 1}}, C \right\} \leq 4 C \sqrt {\frac {\ln \frac {2}{\delta}}{n \vee 1}}.
$$

![](images/f1b409c9466554cc09d3c6b1d78e261582385d3802a93df109ad4800aadbd343.jpg)

Next, we present a short lemma that allows moving between different spaces of probabilities.

Lemma 17. Let $\mathcal{X}$ be a finite set and let $X_{1},\ldots ,X_{n}\in \mathcal{X}$ . Also, let $E_1,\ldots ,E_m\subseteq \mathcal{X}$ be a partition of the set $\mathcal{X}$ , namely, for all $i\neq j$ , $E_{i}\cap E_{j} = \emptyset$ and $\cup_{i = 1}^{m}E_{i} = \mathcal{X}$ . Finally, let $f:\mathcal{X}\mapsto \mathbb{R}$ such that for all $i\in [m]$ and $x\in E_i$ , it holds that $f(x) = f(i)$ , and define

$$
\hat {P} _ {n} (x) = \frac {1}{n} \sum_ {\ell = 1} ^ {n} \mathbb {1} \{X _ {\ell} = x \}, \quad a n d, \quad \hat {Q} _ {n} (i) = \frac {1}{n} \sum_ {\ell = 1} ^ {n} \mathbb {1} \{X _ {\ell} \in E _ {i} \}.
$$

Then, the following hold:

1. $\hat{Q}_n(i) = \hat{P}_n(E_i)\triangleq \sum_{x\in E_i}\hat{P}_n(x)\text{ and, in particular,} \mathbb{E}_{i\sim \hat{Q}_n}[f(i)] = \mathbb{E}_{x\sim \hat{P}_n}[f(x)].$   
2. If $P$ is a distribution over $\mathcal{X}$ and $X_1, \ldots, X_n \in \mathcal{X}$ are i.i.d. samples from $P$ , then $\mathbb{E}[\hat{Q}_n(i)] = P(E_i) \triangleq Q(i)$ . It also holds that $\mathbb{E}_{x \sim P}[f(x)] = \mathbb{E}_{i \sim Q}[f(i)]$ .

Proof. For the first part, we have by definition that

$$
\hat {Q} _ {n} (i) = \frac {1}{n} \sum_ {\ell = 1} ^ {n} \mathbb {1} \left\{X _ {\ell} \in E _ {i} \right\} = \sum_ {x \in \mathcal {X}} \frac {1}{n} \sum_ {\ell = 1} ^ {n} \mathbb {1} \left\{X _ {\ell} = x \right\} \mathbb {1} \left\{x \in E _ {i} \right\} = \sum_ {x \in \mathcal {X}} \hat {P} _ {n} (x) \mathbb {1} \left\{x \in E _ {i} \right\}
$$

$$
= \sum_ {x \in E _ {i}} \hat {P} _ {n} (x) = \hat {P} _ {n} (E _ {i}).
$$

In particular, it holds that

$$
\begin{array}{l} \mathbb {E} _ {i \sim \hat {Q} _ {n}} [ f (i) ] = \sum_ {i = 1} ^ {m} \hat {Q} _ {n} (i) f (i) = \sum_ {i = 1} ^ {m} \sum_ {x \in E _ {i}} \hat {P} _ {n} (x) f (i) \stackrel {(1)} {=} \sum_ {i = 1} ^ {m} \sum_ {x \in E _ {i}} \hat {P} _ {n} (x) f (x) \stackrel {(2)} {=} \sum_ {x \in \mathcal {X}} \hat {P} _ {n} (x) f (x) \\ = \mathbb {E} _ {x \sim \hat {P} _ {n}} [ f (x) ], \\ \end{array}
$$

where (1) is since f is constant inside $E_{i}$ and (2) is since $\{E_{i}\}_{i=1}^{m}$ partition X.

For the second part of the statement, notice that since the samples are i.i.d., it holds that $\mathbb{E}\left[\hat{P}_{n}(x)\right]=P(x)$ , and therefore,

$$
\mathbb {E} [ \hat {Q} _ {n} (i) ] = \mathbb {E} \left[ \sum_ {x \in E _ {i}} \hat {P} _ {n} (x) \right] = \sum_ {x \in E _ {i}} P (x) = P (E _ {i}) = Q (i).
$$

Finally, as in the first part of the statement, it holds that

$$
\begin{array}{l} \mathbb {E} _ {i \sim Q} [ f (i) ] = \sum_ {i = 1} ^ {m} Q (i) f (i) = \sum_ {i = 1} ^ {m} \sum_ {x \in E _ {i}} P (x) f (i) = \sum_ {i = 1} ^ {m} \sum_ {x \in E _ {i}} P (x) f (x) = \sum_ {x \in \mathcal {X}} P (x) f (x) \\ = \mathbb {E} _ {x \sim P} [ f (x) ]. \\ \end{array}
$$

![](images/6449cb044594d81969f6013c642372d8ecea9f28091147c5222661f7c319e5bb.jpg)

Finally, we present two specialized concentration results that are needed for reward and transition lookahead, respectively.

Lemma 18. Let $X, X_1, \ldots, X_n \in \mathbb{R}^d$ be i.i.d. random vectors over $[0,1]$ and let $C \geq 1$ be some constant. Then, for any $\delta \in (0,1)$ , with probability at least $1 - \delta$ ,

$$
\forall u \in [ 0, C ] ^ {d}, \qquad \left| \mathbb {E} \bigg [ \max _ {i \in [ d ]} \{X (i) + u (i) \} \bigg ] - \frac {1}{n} \sum_ {\ell = 1} ^ {n} \max _ {i \in [ d ]} \{X _ {\ell} (i) + u (i) \} \right| \leq 3 \sqrt {\frac {d \ln \frac {9 C n}{\delta}}{2 n}}.
$$

Proof. Denote $m(u) = \mathbb{E}\left[\max_{i \in [d]} \{X(i) + u(i)\}\right]$ and $\hat{m}(u) = \frac{1}{n} \sum_{\ell=1}^{n} \max_{i \in [d]} \{X_{\ell}(i) + u(i)\}$ and fix any $u \in [0, C]^d$ . Since the variables are bounded in [0, 1], their maximum is bounded almost surely in $[\max_i u(i), \max_i u(i) + 1]$ , namely, an interval of unit length. Therefore, by Hoeffding's inequality, for any $\delta' \in (0, 1)$ , w.p. $1 - \delta'$

$$
| m (u) - \hat {m} (u) | \leq \sqrt {\frac {\ln \frac {2}{\delta^ {\prime}}}{2 n}}.
$$

Now, for some $\epsilon \in (0,C]$ , let $u_{\epsilon}$ be the closest vector to $u$ on a grid $\{0,\epsilon ,2\epsilon ,\ldots ,C\}^d$ . Then, it clearly holds that

$$
| m (u) - \hat {m} (u) | \leq | m (u _ {\epsilon}) - \hat {m} (u _ {\epsilon}) | + 2 \epsilon .
$$

Taking the union bound over all $\left(\left\lceil \frac{C}{\epsilon} \right\rceil + 1\right)^d$ possible choices for $u_{\epsilon}$ and fixing $\delta' = \frac{\delta}{\left(\left\lceil \frac{C}{\epsilon} \right\rceil + 1\right)^d}$ , we get w.p. $1 - \delta$ for all $u$ that

$$
| m (u) - \hat {m} (u) | \leq \sqrt {\frac {\ln \frac {2 \left(\lceil \frac {C}{\epsilon} \rceil + 1\right) ^ {d}}{\delta}}{2 n}} + 2 \epsilon \leq \sqrt {\frac {d \ln \frac {6 C}{\epsilon \delta}}{2 n}} + 2 \epsilon .
$$

Now, fixing $\epsilon = \sqrt{\frac{d\ln\frac{6C}{\delta}}{2n}}$ and noting that $\frac{1}{\epsilon} \leq \sqrt{2n}$ for $C \geq 1$ , we get

$$
| m (u) - \hat {m} (u) | \leq \sqrt {\frac {d \ln \frac {6 C \sqrt {2 n}}{\delta}}{2 n}} + 2 \sqrt {\frac {d \ln \frac {6 C}{\delta}}{2 n}} \leq \sqrt {\frac {d \ln \frac {9 C n}{\delta}}{2 n}} + 2 \sqrt {\frac {d \ln \frac {6 C}{\delta}}{2 n}} \leq 3 \sqrt {\frac {d \ln \frac {9 C n}{\delta}}{2 n}}.
$$

![](images/c4644454e3778f4c9cf57377f076c84d266662a7176b2caa4330adcc3c669b35.jpg)

Lemma 19. Let $X, X_1, \ldots, X_n \in \mathbb{R}^d$ be i.i.d. random vectors with components supported over the discrete set $[m]$ and let $C \geq 1$ be some constant. Then, uniformly over all $u \in [0, C]^{dm}$ w.p. $1 - \delta$ :

$$
\begin{array}{l} \left| \mathbb {E} \Big [ \max _ {i} \{u (X (i), i) \} \Big ] - \frac {1}{n} \sum_ {\ell = 1} ^ {n} \max _ {i} \{u (X _ {\ell} (i), i) \} \right| \\ \leq \sqrt {\frac {2 m d \ln \frac {6 n}{\delta} \mathrm{Var} (\max _ {i} \{u (X (i) , i) \})}{n}} + + \frac {8 C m d \left(\ln \frac {6 n}{\delta}\right) ^ {1 . 5}}{n}. \\ \end{array}
$$

Proof. We follow a similar path to Lemma 18 and use a covering argument. Denoting $w(u) = \mathbb{E}[\max_i\{u(X(i),i)\}]$ and $\hat{w}(u) = \frac{1}{n}\sum_{\ell = 1}^{n}\max_i\{u(X_\ell (i),i)\}$ , by Bernstein's inequality, for any $\delta^{\prime}\in (0,1)$ and fixed $u\in [0,C]^{dm}$ , it holds w.p. $1 - \delta^{\prime}$ that

$$
| w (u) - \hat {w} (u) | \leq \sqrt {\frac {2 \operatorname{Var} (\max _ {i} \{u (X (i) , i) \}) \ln \frac {2}{\delta}}{n}} + \frac {2 C \ln \frac {2}{\delta}}{3 n}. \tag {17}
$$

Now, for some $\epsilon \in (0,C]$ , let $u_{\epsilon}$ be the closest matrix to $u$ on a grid $\{0,\epsilon ,2\epsilon ,\ldots ,C\}^{md}$ and denote $Z(u) = \max_i\{u(X(i),i)\}$ with samples $Z_{i}(u)$ . By the smoothness of the max function, it holds that

$$
\left| Z (u) - Z \left(u _ {\epsilon}\right) \right| \leq \epsilon .
$$

In particular, we also have that

$$
\left| \mathbb {E} [ Z (u) ^ {2} ] - \mathbb {E} [ Z (u _ {\epsilon}) ^ {2} ] \right| \leq \epsilon^ {2} + 2 C \epsilon , \quad \text { and } \quad \left| \mathbb {E} [ Z (u) ] ^ {2} - \mathbb {E} [ Z (u _ {\epsilon}) ] ^ {2} \right| \leq \epsilon^ {2} + 2 C \epsilon ,
$$

so we have

$$
\left| \operatorname{Var} \Bigl (\max _ {i} \{u (X (i), i) \} \Bigr) - \operatorname{Var} \Bigl (\max _ {i} \{u _ {\epsilon} (X (i), i) \} \Bigr) \right| = | \operatorname{Var} (Z (u)) - \operatorname{Var} (Z (u _ {\epsilon})) | \leq 2 \epsilon^ {2} + 4 C \epsilon .
$$

Similarly, it holds that

$$
| w (u) - \hat {w} (u) | \leq | w (u _ {\epsilon}) - \hat {w} (u _ {\epsilon}) | + 2 \epsilon .
$$

Taking the union bound over all $\left(\left\lceil\frac{C}{\epsilon}\right\rceil+1\right)^{md}$ possible choices for $u_{\epsilon}$ and fixing $\delta'=\frac{\delta}{\left(\left\lceil\frac{C}{\epsilon}\right\rceil+1\right)^{dm}}$ , we get w.p. $1-\delta$ for all u that

$$
\begin{array}{l} | w (u) - \hat {w} (u) | \leq \sqrt {\frac {2 \operatorname{Var} \left(\max _ {i} \left\{u _ {\epsilon} (X (i) , i) \right\}\right) \ln \frac {2 \left(\lceil \frac {C}{\epsilon} \rceil + 1\right) ^ {m d}}{\delta}}{n}} + \frac {2 C \ln \frac {2 \left(\lceil \frac {C}{\epsilon} \rceil + 1\right) ^ {m d}}{\delta}}{3 n} + 2 \epsilon \\ \leq \sqrt {\frac {2 m d \mathrm{Var} (\max _ {i} \{u _ {\epsilon} (X (i) , i) \}) \ln \frac {6 C}{\epsilon \delta}}{n}} + \frac {2 C m d \ln \frac {6 C}{\epsilon \delta}}{3} + 2 \epsilon \\ \leq \sqrt {\frac {2 m d \ln \frac {6 C}{\epsilon \delta} (\mathrm{Var} (\max _ {i} \{u (X (i) , i) \}) + 2 \epsilon^ {2} + 4 C \epsilon)}{n}} + \frac {2 C m d \ln \frac {6 C}{\epsilon \delta}}{3 n} + 2 \epsilon \\ \leq \sqrt {\frac {2 m d \ln \frac {6 C}{\epsilon \delta} \operatorname{Var} (\max _ {i} \{u (X (i) , i) \})}{n}} + \sqrt {\frac {8 m d C \epsilon \ln \frac {6 C}{\epsilon \delta}}{n}} + \sqrt {\frac {4 m d \epsilon^ {2} \ln \frac {6 C}{\epsilon \delta}}{n}} \\ + \frac {2 C m d \ln \frac {6 C}{\epsilon \delta}}{3 n} + 2 \epsilon . \\ \end{array}
$$

Now, fixing $\epsilon = \frac{C\ln\frac{6n}{\delta}}{n}$ and noticing that $\frac{6C}{\epsilon\delta} \leq \frac{6n}{\delta}$ , we get

$$
\begin{array}{l} | w (u) - \hat {w} (u) | \leq \sqrt {\frac {2 m d \ln \frac {6 n}{\delta} \mathrm{Var} (\max _ {i} \{u (X (i) , i) \})}{n}} + \frac {\sqrt {8 m d} C \ln \frac {6 n}{\delta}}{n} + \frac {\sqrt {4 m d} C \left(\ln \frac {6 n}{\delta}\right) ^ {1 . 5}}{n ^ {1 . 5}} \\ + \frac {2 C m d \ln \frac {6 n}{\delta}}{3 n} + \frac {2 C \ln \frac {6 C}{\delta}}{n} \\ \leq \sqrt {\frac {2 m d \ln \frac {6 n}{\delta} \mathrm{Var} (\max _ {i} \{u (X (i) , i) \})}{n}} + \frac {8 C m d \left(\ln \frac {6 n}{\delta}\right) ^ {1 . 5}}{n}. \\ \end{array}
$$

□

# D.2 Count-Related Lemmas

Lemma 20. The following bounds hold:

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} \leq S A H + 2 \sqrt {S A H ^ {2} K}, \quad \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1} \leq S A H (2 + \ln (K)),
$$

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}} \leq S H + 2 \sqrt {S H ^ {2} K}, \quad \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1} \leq S H (2 + \ln (K)).
$$

Proof. Recall that every time a state (or state-action) is visited, its visitation-count is increased by 1, up to $n_{h}^{K-1}(s,a)$ at the last episode. therefore, we can write

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1}} = \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \sum_ {a \in \mathcal {A}} \sum_ {k = 1} ^ {K} \frac {\mathbb {1} \left\{s _ {h} ^ {k} = s , a _ {h} ^ {k} = a \right\}}{\sqrt {n _ {h} ^ {k - 1} (s , a) \vee 1}} \\ = \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \sum_ {a \in \mathcal {A}} \sum_ {i = 0} ^ {n _ {h} ^ {K - 1} (s, a)} \frac {1}{\sqrt {i \vee 1}} \\ \leq \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \sum_ {a \in \mathcal {A}} \left(1 + 2 \sqrt {n _ {h} ^ {K - 1} (s , a)}\right) \\ \leq S A H + 2 \sqrt {S A H \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \sum_ {a \in \mathcal {A}} n _ {h} ^ {K - 1} (s , a)} (\text { Jensen's   inequality }) \\ \leq S A H + 2 \sqrt {S A H ^ {2} K}. \\ \end{array}
$$

where we bounded the total number of visits by the number of steps HK. Similarly, we also have

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{n _ {h} ^ {k - 1} (s _ {h} ^ {k} , a _ {h} ^ {k}) \vee 1} = \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \sum_ {a \in \mathcal {A}} \sum_ {i = 0} ^ {n _ {h} ^ {K - 1} (s, a)} \frac {1}{i \vee 1} \\ \leq \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \sum_ {a \in \mathcal {A}} \left(2 + \ln \left(n _ {h} ^ {K - 1} (s, a) \vee 1\right)\right) \leq S A H (2 + \ln (K)). \\ \end{array}
$$

We can likewise prove the inequalities for the state counts as follows:

$$
\begin{array}{l} \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{\sqrt {n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1}} = \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \sum_ {k = 1} ^ {K} \frac {\mathbb {1} \left\{s _ {h} ^ {k} = s \right\}}{\sqrt {n _ {h} ^ {k - 1} (s) \vee 1}} \\ = \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \sum_ {i = 0} ^ {n _ {h} ^ {K - 1} (s)} \frac {1}{\sqrt {i \vee 1}} \\ \leq \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \left(1 + 2 \sqrt {n _ {h} ^ {K - 1} (s)}\right) \\ \leq S H + 2 \sqrt {S H \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} n _ {h} ^ {K - 1} (s)} \quad (\text { Jensen's   inequality }) \\ \leq S H + 2 \sqrt {S H ^ {2} K}, \\ \end{array}
$$

and

$$
\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{n _ {h} ^ {k - 1} (s _ {h} ^ {k}) \vee 1} = \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \sum_ {i = 0} ^ {n _ {h} ^ {K - 1} (s)} \frac {1}{i \vee 1} \leq \sum_ {h = 1} ^ {H} \sum_ {s \in \mathcal {S}} \bigl (2 + \ln \bigl (n _ {h} ^ {K - 1} (s) \vee 1 \bigr) \bigr) \leq S H (2 + \ln (K)).
$$

□

# D.3 Analysis of Variance terms

Lemma 21. Let $P$ be a distribution over a finite set $\mathcal{X}$ and let $X \sim P$ . Also, let $V_{1}, V_{2}: \mathcal{X} \mapsto [0, C]$ for some $C > 0$ such that $V_{1}(x) \leq V_{2}(x)$ for all $x \in \mathcal{X}$ . Then, for any $\alpha, n > 0$ , it holds that

$$
\frac {\sqrt {\operatorname{Var} _ {P} (V _ {2} (X))}}{\sqrt {n}} \leq \frac {\sqrt {\operatorname{Var} _ {P} (V _ {1} (X))}}{\sqrt {n}} + \frac {1}{\alpha} \mathbb {E} _ {P} [ V _ {2} (X) - V _ {1} (X) ] + \frac {C \alpha}{4 n}
$$

Proof. By Lemma 26, we have

$$
\begin{array}{l} \sqrt {\operatorname{Var} _ {P} \left(V _ {2} (X)\right)} - \sqrt {\operatorname{Var} _ {P} \left(V _ {1} (X)\right)} \leq \sqrt {\operatorname{Var} _ {P} \left(V _ {2} (X) - V _ {1} (X)\right)} \\ \leq \sqrt {\mathbb {E} _ {P} [ (V _ {2} (X) - V _ {1} (X)) ^ {2} ]} \\ \leq \sqrt {C \mathbb {E} _ {P} [ V _ {2} (X) - V _ {1} (X) ]} \\ \end{array}
$$

where the last inequality is by the boundedness and since $V_{1}(x) \leq V_{2}(x)$ . Thus, we can bound

$$
\begin{array}{l} \frac {\sqrt {\operatorname{Var} _ {P} (V _ {2} (X))} - \sqrt {\operatorname{Var} _ {P} (V _ {1} (X))}}{\sqrt {n}} \leq \frac {\sqrt {C \mathbb {E} _ {P} [ V _ {2} (X) - V _ {1} (X) ]}}{\sqrt {n}} \\ = \sqrt {\mathbb {E} _ {P} [ V _ {2} (X) - V _ {1} (X) ]} \cdot \sqrt {\frac {C}{n}} \\ \leq \frac {1}{\alpha} \mathbb {E} _ {P} [ V _ {2} (X) - V _ {1} (X) ] + \frac {C \alpha}{4 n}, \\ \end{array}
$$

where last inequality is due to Young's inequality ( $ab \leq \frac{1}{\alpha} a^2 + \frac{\alpha}{4} b^2$ for all $\alpha > 0$ ).

![](images/fb2a87ffd94ed70505db718f2dfe4128fec59b070fb270e20d32f5681d7a6d21.jpg)

Lemma 22. Let $P, P'$ be distributions over a finite set $\mathcal{X}$ and let $X \sim P$ . Also, let $V_1, V_2, V_3: \mathcal{X} \mapsto [0, C]$ for some $C > 0$ such that $V_1(x) \leq V_2(x) \leq V_3(x)$ for all $x \in \mathcal{X}$ . Finally, assume that

$$
\left| \sqrt {\operatorname{Var} _ {P} (V _ {2} (X))} - \sqrt {\operatorname{Var} _ {P ^ {\prime}} (V _ {2} (X))} \right| \leq \beta
$$

for some $\beta > 0$ . Then, for any $\alpha, n > 0$ , it holds that

$$
\begin{array}{l} \frac {\sqrt {\operatorname{Var} _ {P ^ {\prime}} (V _ {3} (X))}}{\sqrt {n}} \leq \frac {\sqrt {\operatorname{Var} _ {P} (V _ {1} (X))}}{\sqrt {n}} + \frac {1}{\alpha} \mathbb {E} _ {P ^ {\prime}} [ V _ {3} (X) - V _ {2} (X) ] + \frac {1}{\alpha} \mathbb {E} _ {P} [ V _ {2} (X) - V _ {1} (X) ] + \frac {C \alpha}{2 n} + \frac {\beta}{\sqrt {n}} \\ \leq \frac {\sqrt {\operatorname{Var} _ {P} (V _ {1} (X))}}{\sqrt {n}} + \frac {1}{\alpha} \mathbb {E} _ {P ^ {\prime}} [ V _ {3} (X) - V _ {1} (X) ] + \frac {1}{\alpha} \mathbb {E} _ {P} [ V _ {3} (X) - V _ {1} (X) ] + \frac {C \alpha}{2 n} + \frac {\beta}{\sqrt {n}}. \\ \end{array}
$$

Proof. We decompose the l.h.s. as follows

$$
\begin{array}{l} \frac {\sqrt {\operatorname{Var} _ {P ^ {\prime}} (V _ {3} (X))}}{\sqrt {n}} = \frac {\sqrt {\operatorname{Var} _ {P ^ {\prime}} (V _ {3} (X))} - \sqrt {\operatorname{Var} _ {P ^ {\prime}} (V _ {2} (X))}}{\sqrt {n}} + \frac {\sqrt {\operatorname{Var} _ {P ^ {\prime}} (V _ {2} (X))} - \sqrt {\operatorname{Var} _ {P} (V _ {2} (X))}}{\sqrt {n}} \\ + \frac {\sqrt {\operatorname{Var} _ {P} (V _ {2} (X))} - \sqrt {\operatorname{Var} _ {P} (V _ {1} (X))}}{\sqrt {n}} + \frac {\sqrt {\operatorname{Var} _ {P} (V _ {1} (X))}}{\sqrt {n}} \\ \end{array}
$$

We bound the first and third terms using Lemma 21 and bound the second term with the assumption and get

$$
\begin{array}{l} \frac {\sqrt {\operatorname{Var} _ {P ^ {\prime}} (V _ {3} (X))}}{\sqrt {n}} \leq \frac {1}{\alpha} \mathbb {E} _ {P ^ {\prime}} [ V _ {3} (X) - V _ {2} (X) ] + \frac {C \alpha}{4 n} + \frac {\beta}{\sqrt {n}} \\ + \frac {1}{\alpha} \mathbb {E} _ {P} \left[ V _ {2} (X) - V _ {1} (X) \right] + \frac {C \alpha}{4 n} + \frac {\sqrt {\operatorname{Var} _ {P} \left(V _ {1} (X)\right)}}{\sqrt {n}} \\ = \frac {\sqrt {\operatorname{Var} _ {P} (V _ {1} (X))}}{\sqrt {n}} + \frac {1}{\alpha} \mathbb {E} _ {P ^ {\prime}} [ V _ {3} (X) - V _ {2} (X) ] + \frac {1}{\alpha} \mathbb {E} _ {P} [ V _ {2} (X) - V _ {1} (X) ] + \frac {C \alpha}{2 n} + \frac {\beta}{\sqrt {n}} \\ \leq \frac {\sqrt {\operatorname{Var} _ {P} (V _ {1} (X))}}{\sqrt {n}} + \frac {1}{\alpha} \mathbb {E} _ {P ^ {\prime}} [ V _ {3} (X) - V _ {1} (X) ] + \frac {1}{\alpha} \mathbb {E} _ {P} [ V _ {3} (X) - V _ {1} (X) ] + \frac {C \alpha}{2 n} + \frac {\beta}{\sqrt {n}}, \\ \end{array}
$$

where the last inequality uses the fact that $V_{1}(x) \leq V_{2}(x) \leq V_{3}(x)$ for all $x \in \mathcal{X}$ . The last two bounds are the desired results.

# E Existing Results

Lemma 23 (Monotonic Bonuses,[Zhang et al., 2023], Appendix C.1). For any $p \in \Delta^{S}$ , $v \in \mathbb{R}_{+}^{S}$ s.t. $\| v \|_{\infty} \leq H$ , $\delta' \in (0,1)$ and positive integer $n$ , define the function

$$
f (p, v, n) = p ^ {T} v + \max \Bigg \{\frac {2 0}{3} \sqrt {\frac {\mathrm{Var} _ {p} (v) \ln \frac {1}{\delta^ {\prime}}}{n}}, \frac {4 0 0}{9} \frac {H \ln \frac {1}{\delta^ {\prime}}}{n} \Bigg \}.
$$

Then, the function $f(p, v, n)$ is non-decreasing in each entry of $v$ .

Lemma 24 (Efroni et al. 2021, Lemma 28). Let $Y \in \mathbb{R}^S$ be a vector such that $0 \leq Y(s) \leq H$ for all $s \in S$ . Let $P_1$ and $P_2$ be two transition models and $n \in \mathbb{R}_+^{SA}$ . If

$$
\left\{\forall (s, a, s ^ {\prime}) \in \mathcal {S} \times \mathcal {A} \times \mathcal {S}, h \in [ H ]: | P _ {2, h} (s ^ {\prime} | s, a) - P _ {1, h} (s ^ {\prime} | s, a) | \leq \sqrt {\frac {C _ {1} L _ {\delta} ^ {k} P _ {1 , h} (s ^ {\prime} | s , a)}{n (s , a) \vee 1}} + \frac {C _ {2} L _ {\delta} ^ {k}}{n (s , a) \vee 1} \right\},
$$

for some $C_{1}, C_{2} > 0$ , then, for any $\alpha > 0$ ,

$$
| (P _ {1, h} - P _ {2, h}) Y (s, a) | \leq \frac {1}{\alpha} \mathbb {E} _ {s ^ {\prime} \sim P _ {1, h} (\cdot | s, a)} [ Y (s ^ {\prime}) ] + \frac {H S L _ {\delta} ^ {k} (C _ {2} + \alpha C _ {1} / 4)}{n (s , a) \vee 1},
$$

Lemma 25 (Efroni et al. 2021, Lemma 27). Let $\{Y_t\}_{t \geq 1}$ be a real-valued sequence of random variables adapted to a filtration $\{F_t\}_{t \geq 0}$ . Assume that for all $t \geq 1$ it holds that $0 \leq Y_t \leq C$ a.s., and let $T \in \mathbb{N}$ . Then each of the following inequalities holds with probability greater than $1 - \delta$ .

$$
\sum_ {t = 1} ^ {T} \mathbb {E} [ Y _ {t} | F _ {t - 1} ] \leq \left(1 + \frac {1}{2 C}\right) \sum_ {t = 1} ^ {T} Y _ {t} + 2 (2 C + 1) ^ {2} \ln \frac {1}{\delta},
$$

$$
\sum_ {t = 1} ^ {T} Y _ {t} \leq 2 \sum_ {t = 1} ^ {T} \mathbb {E} [ Y _ {t} | F _ {t - 1} ] + 4 C \ln {\frac {1}{\delta}}.
$$

Lemma 26 (Standard Deviation Differences, e.g., Zanette and Brunskill 2019, lines 48-51). Let $P \in \Delta_d$ be some distribution over [d] and let $V_1, V_2 \in \mathbb{R}^d$ . Then, it holds that

$$
\sqrt {\operatorname{Var} _ {P} (V _ {1})} - \sqrt {\operatorname{Var} _ {P} (V _ {2})} \leq \sqrt {\operatorname{Var} _ {P} (V _ {1} - V _ {2})}.
$$

Lemma 27 (Law of Total Variance, e.g., Zanette and Brunskill 2019, Lemma 15). For any no-lookahead policy $\pi$ , it holds that

$$
\mathbb {E} \left[ \sum_ {h = 1} ^ {H} \mathrm{Var} (V _ {h + 1} ^ {\pi} (s _ {h + 1}) | s _ {h}) | \pi , s _ {1} \right] = \mathbb {E} \left[ \left(\sum_ {h = 1} ^ {H} r _ {h} (s _ {h}, a _ {h}) - V _ {1} ^ {\pi} (s _ {1})\right) ^ {2} | \pi , s _ {1} \right],
$$

where $\operatorname{Var}(V_{h+1}^{\pi}(s_{h+1})|s_{h})$ is the variance of the value at step $s_{h+1}$ given state $s_{h}$ and under the policy $\pi$ , due to the policy randomization and next-state transition probabilities.

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In the abstract, we accurately present the setting and its motivation, as well as a summary of the results, all of which are proved in the appendix.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.   
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The main limitations in this work are a result of the studied setup – some possible extensions and improvement are discussed in the future work section.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
- The authors are encouraged to create a separate "Limitations" section in their paper.   
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.   
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

# Answer: [Yes]

Justification: Proofs for all the stated results are provided in the appendix.

# Guidelines:

- The answer NA means that the paper does not include theoretical results.   
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.   
- All assumptions should be clearly stated or referenced in the statement of any theorems.

\- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

\- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

\- Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

# Answer: [NA]

Justification: The paper does not include experiments.

# Guidelines:

- The answer NA means that the paper does not include experiments.   
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.   
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general, releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

\- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.   
(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.   
(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).   
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: The paper does not include experiments.

# Guidelines:

- The answer NA means that paper does not include experiments requiring code.   
- Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
- While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.   
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).   
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The paper does not include experiments.

# Guidelines:

- The answer NA means that the paper does not include experiments.   
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.   
- The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

# Guidelines:

- The answer NA means that the paper does not include experiments.   
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.   
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

- The assumptions made should be given (e.g., Normally distributed errors).   
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a $96\%$ CI, if the hypothesis of Normality of errors is not verified.   
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.   
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

# 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The paper is purely theoretical and studies a fundamental decision-making model; any ethical issue that might arise would be a core issue in the ethics of applying machine learning, and not tied specifically to this work.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Due to the theoretical nature of the paper and the generality of the model, it is no direct societal impact.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.   
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.   
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: No data or models are released with this paper.

# Guidelines:

- The answer NA means that the paper poses no such risks.   
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

# Guidelines:

- The answer NA means that the paper does not use existing assets.   
- The authors should cite the original paper that produced the code package or dataset.   
- The authors should state which version of the asset is used and, if possible, include a URL.   
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.   
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.   
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.   
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

# 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.   
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
- The paper should discuss whether and how consent was obtained from people whose asset is used.   
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.