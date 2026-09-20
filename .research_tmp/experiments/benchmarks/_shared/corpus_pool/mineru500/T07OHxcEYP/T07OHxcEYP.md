# Differentially Private Reinforcement Learning with Self-Play

Dan Qiao $^{1}$ and Yu-Xiang Wang $^{1}$

$^{1}$ Department of Computer Science, UC Santa Barbara

danqiao@ucsb.edu, yuxiangw@cs.ucsb.edu

# Abstract

We study the problem of multi-agent reinforcement learning (multi-agent RL) with differential privacy (DP) constraints. This is well-motivated by various real-world applications involving sensitive data, where it is critical to protect users' private information. We first extend the definitions of Joint DP (JDP) and Local DP (LDP) to two-player zero-sum episodic Markov Games, where both definitions ensure trajectory-wise privacy protection. Then we design a provably efficient algorithm based on optimistic Nash value iteration and privatization of Bernstein-type bonuses. The algorithm is able to satisfy JDP and LDP requirements when instantiated with appropriate privacy mechanisms. Furthermore, for both notions of DP, our regret bound generalizes the best known result under the single-agent RL case, while our regret could also reduce to the best known result for multi-agent RL without privacy constraints. To the best of our knowledge, these are the first line of results towards understanding trajectory-wise privacy protection in multi-agent RL.

# Contents

# 1 Introduction 3

1.1 Related work 4

# 2 Problem Setup 5

2.1 Markov Games and Regret 5   
2.2 Differential Privacy in Multi-agent RL 6

# 3 Algorithm 8

# 4 Main results 11

# 5 Privatizers for JDP and LDP 11

5.1 Central Privatizer for Joint DP 12   
5.2 Local Privatizer for Local DP 13   
5.3 The post-processing step 14   
5.4 Some discussions 15

# 6 Proof overview 15

# 7 Conclusion 16

# A Extended related works 22

# B Proof of main theorems 22

B.1 Properties of private estimations 22   
B.2 Proof of UCB and LCB 24   
B.3 Proof of Theorem 4.1 27   
B.4 Proof of Theorem 4.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30

# C Missing proof in Section 5 31

# 1 Introduction

This paper considers the problem of multi-agent reinforcement learning (multi-agent RL), wherein several agents simultaneously make decisions in an unfamiliar environment with the goal of maximizing their individual cumulative rewards. Multi-agent RL has been deployed not only in large-scale strategy games like Go [Silver et al., 2017], Poker [Brown and Sandholm, 2019] and MOBA games [Ye et al., 2020], but also in various real-world applications such as autonomous driving [Shalev-Shwartz et al., 2016], negotiation [Bachrach et al., 2020], and trading in financial markets [Shavandi and Khedmati, 2022]. In these applications, the learning agent analyzes users' private feedback in order to refine its performance, where the data from users usually contain sensitive information. Take autonomous driving as an instance, here a trajectory describes the interaction between the cars in a neighborhood during a fixed time window. At each timestamp, given the current situation of each car, the system (central agent) will send a command for each car to take (e.g. speed up, pull over), and finally the system gathers the feedback from each car (e.g. whether the driving is safe, whether the customer feels comfortable) and enhances its policy. Here, (situation, command, feedback) corresponds to (state, action, reward) in a Markov Game where the state and reward of each user are considered as sensitive information. Therefore, leakage of such information is not acceptable. Regrettably, it has been demonstrated that without the implementation of privacy safeguards, learning agents tend to inadvertently memorize details from individual training data points [Carlini et al., 2019], regardless of their relevance to the learning process [Brown et al., 2021]. This susceptibility exposes multi-agent RL agents to potential privacy threats.

To handle the above privacy issue, Differential privacy (DP) [Dwork et al., 2006] has been widely considered. The output of a differentially private reinforcement learning algorithm cannot be discerned from its output in an alternative reality where any specific user is substituted, which effectively mitigates the privacy risks mentioned earlier. However, it is shown [Shariff and Sheffet, 2018] that standard DP will lead to linear regret even under contextual bandits. Therefore, Vietri et al. [2020] considered a relaxed surrogate of DP: Joint Differential Privacy (JDP) [Kearns et al., 2014] for RL. Briefly speaking, JDP protects the information about any specific user even given the output of all other users. Meanwhile, another variant of DP: Local Differential Privacy (LDP) [Duchi et al., 2013] has also been extended to RL by Garcelon et al. [2021] due to its stronger privacy protection. LDP requires that the raw data of each user is privatized before being sent to the agent. Although following works [Chowdhury and Zhou, 2022, Qiao and Wang, 2023c] established near optimal results under these two notions of DP, all of the previous works focused on the single-agent RL setting while the solution to multi-agent RL with differential privacy is still unknown. Therefore it is natural to question:

Question 1.1. Is it possible to design a provably efficient self-play algorithm to solve Markov games while satisfying the constraints of differential privacy?

Our contributions. In this paper, we answer the above question affirmatively by proposing a general algorithm for DP multi-agent RL: DP-Nash-VI (Algorithm 1). Our contributions are threefold and summarized as below.

\- We first extend the definitions of Joint DP (Definition 2.2) and Local DP (Definition 2.3) to the multi-agent RL setting. Both notions of DP focus on protecting the sensitive information of each trajectory, which is consistent with the counterparts under single-agent RL.

<table><tr><td>Algorithms for Markov Games</td><td>Regret without privacy</td><td>Regret under  $\epsilon$ -JDP</td><td>Regret under  $\epsilon$ -LDP</td></tr><tr><td>DP-Nash-VI (Our Algorithm 1)Nash VI [Liu et al., 2021]</td><td> $\tilde{O}(\sqrt{H^{2}SABT})$  $\tilde{O}(\sqrt{H^{2}SABT})^{*}$ </td><td> $\tilde{O}(\sqrt{H^{2}SABT} + H^{3}S^{2}AB/\epsilon)$ N.A.</td><td> $\tilde{O}(\sqrt{H^{2}SABT} + S^{2}AB\sqrt{H^{5}T}/\epsilon)$ N.A.</td></tr><tr><td>Lower bounds</td><td> $\Omega(\sqrt{H^{2}S(A+B)T})$  [Bai and Jin, 2020]</td><td> $\tilde{\Omega}\left(\sqrt{H^{2}S(A+B)T} + \frac{HS(A+B)}{\epsilon}\right)$ </td><td> $\tilde{\Omega}\left(\sqrt{H^{2}S(A+B)T} + \frac{\sqrt{HS(A+B)T}}{\epsilon}\right)$ </td></tr><tr><td>Algorithms for MDPs ( $B = 1$ )</td><td>Regret without privacy</td><td>Regret under  $\epsilon$ -JDP</td><td>Regret under  $\epsilon$ -LDP</td></tr><tr><td>PUCB [Vietri et al., 2020]LDP-OBI [Garcelon et al., 2021]Private-UCB-VI [Chowdhury and Zhou, 2022]DP-UCBVI ‡ [Qiao and Wang, 2023c]</td><td> $\tilde{O}(\sqrt{H^{3}S^{2}AT})$  $\tilde{O}(\sqrt{H^{3}S^{2}AT})$  $\tilde{O}(\sqrt{H^{3}SAT})$  $\tilde{O}(\sqrt{H^{2}SAT})$ </td><td> $\tilde{O}(\sqrt{H^{3}S^{2}AT} + H^{3}S^{2}A/\epsilon)^{*}$ N.A. $\tilde{O}(\sqrt{H^{3}SAT} + H^{3}S^{2}A/\epsilon)$  $\tilde{O}(\sqrt{H^{2}SAT} + H^{3}S^{2}A/\epsilon)$ </td><td>N.A. $\tilde{O}(\sqrt{H^{3}S^{2}AT} + S^{2}A\sqrt{H^{5}T}/\epsilon)^{\dagger}$  $\tilde{O}(\sqrt{H^{3}SAT} + S^{2}A\sqrt{H^{5}T}/\epsilon)$  $\tilde{O}(\sqrt{H^{2}SAT} + S^{2}A\sqrt{H^{5}T}/\epsilon)$ </td></tr></table>

Table 1: Comparison of our results (in blue) to existing work regarding regret without privacy (i.e. the privacy budget is infinity), regret under $\epsilon$ -Joint DP and regret under $\epsilon$ -Local DP. In the above, $S$ is the number of states, $A, B$ are the number of actions for both players, $H$ is the planning horizon and $K$ is the number of episodes ( $T = HK$ is the number of steps). Markov decision processes (MDPs) is a special case of Markov Games where $B = 1$ . $*$ : This result is the best known regret bound when there is no privacy concern. $\star$ : More discussions about this bound can be found in Chowdhury and Zhou [2022]. $\dagger$ : The original regret bound in Garcelon et al. [2021] is derived under the setting of stationary MDP, and can be directly transferred to the bound here by adding $\sqrt{H}$ to the first term. $\ddagger$ : This algorithm achieved the best known results under single-agent MDPs, and our Algorithm 1 can obtain the same regret bounds under this setting.

- We design a new algorithm DP-Nash-VI (Algorithm 1) based on optimistic Nash value iteration and privatization of Bernstein-type bonuses. The algorithm can be combined with any Privatizer (for JDP or LDP) that possesses a corresponding regret bound (Theorem 4.1). Moreover, when there is no privacy constraint (i.e. the privacy budget is infinity), our regret reduces to the best known regret for non-private multi-agent RL.   
- Under the constraint of $\epsilon$ -JDP, DP-Nash-VI achieves a regret of $\widetilde{O}(\sqrt{H^2 SABT} + H^3 S^2 AB / \epsilon)$ (Theorem 5.2). Compared to the regret lower bound (Theorem 5.3), the main term is nearly optimal while the additional cost due to JDP has optimal dependence on $\epsilon$ . Under the $\epsilon$ -LDP constraint, DP-Nash-VI achieves a regret of $\widetilde{O}(\sqrt{H^2 SABT} + S^2 AB\sqrt{H^5 T} /\epsilon)$ (Theorem 5.5), where the dependence on $K, \epsilon$ is optimal according to the lower bound (Theorem 5.6). The pair of results strictly generalizes the best known results for single-agent RL with DP [Qiao and Wang, 2023c].

# 1.1 Related work

We compare our results with existing works on differentially private reinforcement learning [Vietri et al., 2020, Garcelon et al., 2021, Chowdhury and Zhou, 2022, Qiao and Wang, 2023c] and regret minimization under Markov Games [Liu et al., 2021] in Table 1, while more discussions about differentially private learning algorithms are deferred to Appendix A. Notably, all existing DP RL algorithms focus on the single-agent case. In comparison, our algorithm works for the more general two-player setting and our results directly match the best known regret bounds [Qiao and Wang, 2023c] when applied to the single-agent setting.

Recently, several works provide non-asymptotic theoretical guarantees for learning Markov Games. Bai and Jin [2020] developed the first provably-efficient algorithms in MGs based on optimistic value iteration, and the result is improved by Liu et al. [2021] using model-based approach. Meanwhile, model-free approaches are shown to break the curse of multiagency and improve the dependence

on action space [Bai et al., 2020, Jin et al., 2021, Mao et al., 2022, Wang et al., 2023, Cui et al., 2023]. However, all these algorithms base on the original data from users, and thus are vulnerable to various privacy attacks. While several works [Hossain and Lee, 2023, Hossain et al., 2023, Zhao et al., 2023b, Gohari et al., 2023] study the privatization of communications between multiple agents, none of them provide regret guarantees. In comparison, we design algorithms that provably protect the sensitive information in each trajectory, while achieving near-optimal regret bounds simultaneously.

Technically speaking, we follow the idea of optimistic Nash value iteration and privatization of Bernstein-type bonuses. Optimistic Nash value iteration aims to construct both upper bounds and lower bounds for value functions, which could guide the exploration. Such idea has been applied by previous model-based approaches [Bai and Jin, 2020, Xie et al., 2020, Liu et al., 2021] to derive tight regret bounds. To satisfy the privacy guarantees, we are required to construct the UCB and LCB privately. In this work, we privatize the transition kernel estimate and construct a private bonus function for our purpose. Among different bonuses, we generalize the approach in Qiao and Wang [2023c] and directly operate on the Bernstein-type bonus, which could enable tight regret analysis while the privatization is more technically demanding due to the variance term. To handle this, we first privatize the visitation counts such that they satisfy several nice properties, then we use these counts to construct private transition estimates and private bonuses. Lastly, we manage to prove UCB and LCB, and bound the private terms by their non-private counterparts to complete the regret analysis.

# 2 Problem Setup

In this paper, we consider reinforcement learning under Markov Games (MGs) [Shapley, 1953] with the guarantee of Differential Privacy (DP) [Dwork et al., 2006]. Below we introduce MGs and define DP under multi-agent RL.

# 2.1 Markov Games and Regret

Markov Games (MGs) are the generalization of Markov Decision Processes (MDPs) to the multiplayer setting, where each player aims to maximize her own reward. We consider two-player zero-sum episodic MGs, denoted by a tuple $\mathcal{M}\mathcal{G} = (\mathcal{S}, \mathcal{A}, \mathcal{B}, H, \{P_h\}_{h=1}^H, \{r_h\}_{h=1}^H, s_1)$ , where S is the state space with $S = |S|$ , A and B are the action space for the max-player (who aims to maximize the total reward) and the min-player (who aims to minimize the total reward) respectively with $A = |A|$ , $B = |B|$ . Besides, H is the horizon while the non-stationary transition kernel $P_h(\cdot | s, a, b)$ gives the distribution of the next state if action $(a, b)$ is taken at state s and time step h. In addition, we assume that the reward function $r_h(s, a, b) \in [0, 1]$ is deterministic and known $^{1}$ . For simplicity, we assume each episode starts from a fixed initial state $s_1$ . Then at each time step $h \in [H]$ , two players observe $s_h$ and choose their actions $a_h \in A$ and $b_h \in B$ simultaneously, after which both players observe the action of their opponent and receive reward $r_h(s_h, a_h, b_h)$ , the environment will transit to $s_{h+1} \sim P_h(\cdot | s_h, a_h, b_h)$ .

Markov policy, value function. A Markov policy $\mu$ of the max-player can be seen as a series

of mappings $\mu=\{\mu_{h}\}_{h=1}^{H}$ , where each $\mu_{h}$ maps each state $s\in S$ to a probability distribution over actions A, i.e. $\mu_{h}:S\to\Delta(A)$ . A Markov policy $\nu$ for the min-player is defined similarly. Given a pair of policies $(\mu,\nu)$ and time step $h\in[H]$ , the value function $V_{h}^{\mu,\nu}(\cdot)$ is defined as $V_{h}^{\mu,\nu}(s)=\mathbb{E}_{\mu,\nu}[\sum_{t=h}^{H}r_{t}|s_{h}=s]$ while the Q-value function $Q_{h}^{\mu,\nu}(\cdot,\cdot,\cdot)$ is defined as $Q_{h}^{\mu,\nu}(s,a,b)=\mathbb{E}_{\mu,\nu}[\sum_{t=h}^{H}r_{t}|s_{h},a_{h},b_{h}=s,a,b]$ for all s,a,b. According to the definitions, the following Bellman equation holds:

$$
Q _ {h} ^ {\mu , \nu} (s, a, b) = [ r _ {h} + P _ {h} V _ {h + 1} ^ {\mu , \nu} ] (s, a, b),
$$

$$
V _ {h} ^ {\mu , \nu} (s) = [ \mathbb {E} _ {\mu , \nu} Q _ {h} ^ {\mu , \nu} ] (s), \forall (h, s, a, b).
$$

Best responses, Nash equilibrium. For any policy $\mu$ of the max-player, there exists a best response policy $\nu^{\dagger}(\mu)$ of the min-player such that $V_{h}^{\mu,\nu^{\dagger}(\mu)}(s)=\inf_{\nu}V_{h}^{\mu,\nu}(s)$ for all $(s,h)$ . For simplicity, we denote $V_{h}^{\mu,\dagger}:=V_{h}^{\mu,\nu^{\dagger}(\mu)}$ . Also, $\mu^{\dagger}(\nu)$ and $V_{h}^{\dagger,\nu}$ can be defined by symmetry. It is shown [Filar and Vrieze, 2012] that there exists a pair of policies $(\mu^{\star},\nu^{\star})$ that are best responses against each other, i.e.

$$
V _ {h} ^ {\mu^ {\star}, \dagger} (s) = V _ {h} ^ {\mu^ {\star}, \nu^ {\star}} (s) = V _ {h} ^ {\dagger , \nu^ {\star}} (s), \forall (s, h) \in \mathcal {S} \times [ H ].
$$

The pair of policies $(\mu^{\star},\nu^{\star})$ is called the Nash equilibrium of the Markov game, which further satisfies the following minimax property: for all $(s,h)\in\mathcal{S}\times[H]$ ,

$$
\sup _ {\mu} \inf _ {\nu} V _ {h} ^ {\mu , \nu} (s) = V _ {h} ^ {\mu^ {\star}, \nu^ {\star}} (s) = \inf _ {\nu} \sup _ {\mu} V _ {h} ^ {\mu , \nu} (s).
$$

The value functions of $(\mu^{\star},\nu^{\star})$ are called Nash value functions and we denote $V_{h}^{\star}=V_{h}^{\mu^{\star},\nu^{\star}},Q_{h}^{\star}=Q_{h}^{\mu^{\star},\nu^{\star}}$ for simplicity. Intuitively speaking, Nash equilibrium means that no player could gain more from updating her own policy.

Learning objective: regret. Following previous works [Bai and Jin, 2020, Liu et al., 2021], we aim to minimize the regret, which is defined as below:

$$
\operatorname{Regret} (K) = \sum_ {k = 1} ^ {K} \left[ V _ {1} ^ {\dagger , \nu^ {k}} (s _ {1}) - V _ {1} ^ {\mu^ {k}, \dagger} (s _ {1}) \right],
$$

where K is the number of episodes the agent interacts with the environment and $(\mu^{k}, \nu^{k})$ are the policies executed by the agent in the k-th episode. Note that any sub-linear regret bound can be transferred to a PAC guarantee according to the standard online-to-batch conversion [Jin et al., 2018].

# 2.2 Differential Privacy in Multi-agent RL

For RL with self-play, each trajectory corresponds to the interaction between a pair of users and the environment. The interaction generally follows the protocol below. At time step h of the k-th episode, the users send their state $s_{h}^{k}$ to a central agent M, then M sends back a pair of actions ( $a_{h}^{k}, b_{h}^{k}$ ) for the users to take, and finally the users send their reward $r_{h}^{k}$ to M. Following previous works [Vietri et al., 2020, Chowdhury and Zhou, 2022, Qiao and Wang, 2023c], here we let

$\mathcal{U}=(u_{1},\cdots,u_{K})$ denote the sequence of K unique $^{2}$ pairs of users who participate in the above RL protocol. Besides, each pair of users $u_{k}$ is characterized by the $\{s_{h}^{k},r_{h}^{k}\}_{h=1}^{H}$ information they would respond to all $(AB)^{H3}$ possible sequences of actions from the agent. Let $\mathcal{M}(\mathcal{U})=\{(a_{h}^{k},b_{h}^{k})\}_{h,k=1,1}^{H,K}$ denote the whole sequence of actions suggested by the agent M. Then a direct adaptation of differential privacy [Dwork et al., 2006] is defined below, which says that $\mathcal{M}(\mathcal{U})$ and all other pairs excluding $u_{k}$ together will not disclose much information about user $u_{k}$ .

Definition 2.1 (Differential Privacy (DP)). For any $\epsilon > 0$ and $\delta \in [0,1]$ , a mechanism $\mathcal{M} : \mathcal{U} \to (\mathcal{A} \times \mathcal{B})^{KH}$ is $(\epsilon, \delta)$ -differentially private if for any possible user sequences $\mathcal{U}$ and $\mathcal{U}'$ that is different on one pair of users and any subset $E$ of $(\mathcal{A} \times \mathcal{B})^{KH}$ ,

$$
\mathbb {P} [ \mathcal {M} (\mathcal {U}) \in E ] \leq e ^ {\epsilon} \cdot \mathbb {P} [ \mathcal {M} (\mathcal {U} ^ {\prime}) \in E ] + \delta .
$$

If $\delta = 0$ , we say that $\mathcal{M}$ is $\epsilon$ -differentially private ( $\epsilon$ -DP).

Unfortunately, privately recommending actions to the pair of users $u_{k}$ while protecting their own state and reward information is shown to be impractical even for the single-player setting. Therefore, we consider a relaxed version of DP, known as Joint Differential Privacy (JDP) [Kearns et al., 2014]. JDP says that for all pairs of users $u_{k}$ , the recommendation to all other pairs excluding $u_{k}$ will not disclose the sensitive information about $u_{k}$ . Although being weaker than DP, JDP could still provide meaningful privacy protection by ensuring that even if an adversary can observe the interactions between all other users and the environment, it is statistically hard to reconstruct the interaction between $u_{k}$ and the environment. JDP is first studied by Vietri et al. [2020] under single-agent reinforcement learning, and we extend the definition to the two-player setting.

Definition 2.2 (Joint Differential Privacy (JDP)). For any $\epsilon > 0$ , a mechanism $\mathcal{M}:\mathcal{U}\to (\mathcal{A}\times \mathcal{B})^{KH}$ is $\epsilon$ -joint differentially private if for any $k\in [K]$ , any user sequences $\mathcal{U}$ and $\mathcal{U}'$ that is different on the $k$ -th pair of users and any subset $E$ of $(\mathcal{A}\times \mathcal{B})^{(K - 1)H}$ ,

$$
\mathbb {P} [ \mathcal {M} _ {- k} (\mathcal {U}) \in E ] \leq e ^ {\epsilon} \cdot \mathbb {P} [ \mathcal {M} _ {- k} (\mathcal {U} ^ {\prime}) \in E ],
$$

where $\mathcal{M}_{-k}(\mathcal{U})\in E$ means the sequence of actions sent to all pairs of users excluding $u_{k}$ belongs to set E.

In the example of autonomous driving, JDP ensures that even if an adversary observes the interactions between cars within all time windows except one, it is hard to know what happens during the specific time window. While providing strong privacy protection, JDP requires the central agent M to have access to the real trajectories from users. However, in various scenarios the users are not even willing to directly share their data with the agent. To address such circumstances, Duchi et al. [2013] developed a stronger notion of privacy named Local Differential Privacy (LDP). Now that when considering LDP, the agent can not observe the state of users, we consider the following protocol specific for LDP: at the beginning of the k-th episode, the agent M first sends a policy pair $\pi_{k} = (\mu_{k}, \nu_{k})$ to the pair of users $u_{k}$ , after running $\pi_{k}$ and getting a trajectory $X_{k}$ , $u_{k}$ privatizes their trajectory to $X_{k}^{\prime}$ and sends it back to M. We present the definition of Local DP below, which generalizes the LDP under single-agent reinforcement learning by Garcelon et al.

[2021]. Briefly speaking, Local DP ensures that it is impractical for an adversary to reconstruct the whole trajectory of $u_{k}$ even if observing their whole response.

Definition 2.3 (Local Differential Privacy (LDP)). For any $\epsilon > 0$ , a mechanism $\widetilde{\mathcal{M}}$ is $\epsilon$ -local differentially private if for any possible trajectories $X, X'$ and any possible set $E \subseteq \{\widetilde{\mathcal{M}}(X) | X \text{ is any possible trajectory}\}$ ,

$$
\mathbb {P} [ \widetilde {\mathcal {M}} (X) \in E ] \leq e ^ {\epsilon} \cdot \mathbb {P} [ \widetilde {\mathcal {M}} (X ^ {\prime}) \in E ].
$$

In the example of autonomous driving, LDP ensures that the system can only observe a private version of the interactions between cars instead of the raw data.

Remark 2.4. Note that here our definitions of JDP and LDP both provide trajectory-wise privacy protection, which is consistent with previous works [Chowdhury and Zhou, 2022, Qiao and Wang, 2023c]. Moreover, under the special case where the min-player plays a fixed and known deterministic policy (or equivalently, B only contains a single action and B = 1), the Markov Game setting reduces to a single-agent Markov decision process while our JDP and LDP directly matches previous definitions for the MDP setting. Therefore, our setting strictly generalizes previous works and requires novel techniques to handle the min-player.

Remark 2.5. In the following sections we will show that LDP is consistent with sub-linear regret bounds, while it is known that we can not derive sub-linear regret bounds under the constraint of DP. We remark that there is no contradictory since here the RL protocols for DP and LDP are different. As a result, here a guarantee of LDP does not directly imply a guarantee of DP and the two notions are indeed not directly comparable.

# 3 Algorithm

In this part, we introduce DP-Nash-VI (Algorithm 1). Note that the algorithm takes Privatizer as an input. We analyze the regret of Algorithm 1 for all Privatizers satisfying the Assumption 3.1 below, which includes the cases where the Privatizer is chosen as Central (for JDP) or Local (for LDP).

We first introduce the definition of visitation counts, where $N_{h}^{k}(s,a,b)=\sum_{i=1}^{k-1}\mathbb{1}(s_{h}^{i},a_{h}^{i},b_{h}^{i}=s,a,b)$ denotes the visitation count of $(s,a,b)$ at time step h until the beginning of the k-th episode. Similarly, we let $N_{h}^{k}(s,a,b,s')=\sum_{i=1}^{k-1}\mathbb{1}(s_{h}^{i},a_{h}^{i},b_{h}^{i},s_{h+1}^{i}=s,a,b,s')$ be the visitation count of $(h,s,a,b,s')$ before the k-th episode. In multi-agent RL without privacy constraints, such visitation counts are sufficient for estimating the transition kernel $\{P_{h}\}_{h=1}^{H}$ and updating the exploration policy, as in previous model-based approaches [Liu et al., 2021]. However, these counts base on the original trajectories from the users, which could reveal sensitive information. Therefore, with the concern of privacy, we can only incorporate these counts after a privacy-preserving step. In other words, we use a Privatizer to transfer the original counts to the private version $\widetilde{N}_{h}^{k}(s,a,b),\widetilde{N}_{h}^{k}(s,a,b,s')$ . We make the following Assumption 3.1 for Privatizer, which says that the private counts are close to real ones. Privatizers for JDP and LDP that satisfy Assumption 3.1 will be proposed in Section 5.

Assumption 3.1 (Private counts). For any privacy budget $\epsilon > 0$ and failure probability $\beta \in [0,1]$ , there exists some $E_{\epsilon,\beta} > 0$ such that with probability at least $1 - \beta/3$ , for all $(h,s,a,b,s',k) \in$

Algorithm 1 Differentially Private Optimistic Nash Value Iteration (DP-Nash-VI)   
1: Input: Number of episodes K, privacy budget $\epsilon$ , failure probability $\beta$ and a Privatizer (can be either Central or Local).
2: Initialize: Private counts $\widetilde{N}_{h}^{1}(s,a,b)=\widetilde{N}_{h}^{1}(s,a,b,s')=0$ for all $(h,s,a,b,s')$ . Set up the confidence bound $E_{\epsilon,\beta}$ w.r.t the Privatizer, the minimal gap $\Delta=H$ and universal constants $C_{1},C_{2}>0$ . $\iota=\log(30HSABK/\beta)$ .
3: for $k=1,2,\cdots,K$ do
4: $\overline{V}_{H+1}^{k}(\cdot)=\underline{V}_{H+1}^{k}(\cdot)=0.$ 5: for $h=H,H-1,\cdots,1$ do
6: for $(s,a,b)\in\mathcal{S}\times\mathcal{A}\times\mathcal{B}$ do
7: Compute private transition kernel $\widetilde{P}_{h}^{k}(\cdot|s,a,b)$ as in (1).
8: Compute $\gamma_{h}^{k}(s,a,b)=\frac{C_{1}}{H}\cdot\widetilde{P}_{h}^{k}(\overline{V}_{h+1}^{k}-\underline{V}_{h+1}^{k})(s,a,b).$ 9: Compute private bonus $\Gamma_{h}^{k}(s,a,b)=C_{2}\sqrt{\frac{\operatorname{Var}_{\widetilde{P}_{h}^{k}(\cdot|s,a,b)}\left[\left(\frac{\overline{V}_{h+1}^{k}+\underline{V}_{h+1}^{k}}{2}\right)(\cdot)\right]\cdot\iota}{\widetilde{N}_{h}^{k}(s,a,b)}}+\frac{C_{2}HSE_{\epsilon,\beta}\cdot\iota}{\widetilde{N}_{h}^{k}(s,a,b)}+\frac{C_{2}H^{2}S_{\iota}}{\widetilde{N}_{h}^{k}(s,a,b)}.$ 10: UCB $\overline{Q}_{h}^{k}(s,a,b)=\min\{r_{h}(s,a,b)+\sum_{s'}\widetilde{P}_{h}^{k}(s'|s,a,b)\cdot\overline{V}_{h+1}^{k}(s')+\gamma_{h}^{k}(s,a,b)+\Gamma_{h}^{k}(s,a,b),H\}.$ 11: LCB $Q_{h}^{k}(s,a,b)=\max\{r_{h}(s,a,b)+\sum_{s'}\widetilde{P}_{h}^{k}(s'|s,a,b)\cdot\underline{V}_{h+1}^{k}(s')-\gamma_{h}^{k}(s,a,b)-\Gamma_{h}^{k}(s,a,b),0\}.$ 12: end for
13: for $s\in S$ do
14: Compute the policy $\pi_{h}^{k}(\cdot,\cdot|s)=\mathrm{CCE}(\overline{Q}_{h}^{k}(s,\cdot,\cdot),\underline{Q}_{h}^{k}(s,\cdot,\cdot)).$ 15: Compute the value functions $\overline{V}_{h}^{k}(s)=\mathbb{E}_{\pi_{h}^{k}}\overline{Q}_{h}^{k}(s),\quad\underline{V}_{h}^{k}(s)=\mathbb{E}_{\pi_{h}^{k}}\underline{Q}_{h}^{k}(s).$ 16: end for
17: end for
18: Deploy policy $\pi^{k}=(\pi_{1}^{k},\cdots,\pi_{H}^{k})$ and get trajectory $(s_{1}^{k},a_{1}^{k},b_{1}^{k},r_{1}^{k},\cdots,s_{H+1}^{k}).$ 19: Update the private counts to $\widetilde{N}^{k+1}$ via Privatizer.
20: if $(\overline{V}_{1}^{k}-\underline{V}_{1}^{k})(s_{1})<\Delta$ then
21: $\Delta=(\overline{V}_{1}^{k}-\underline{V}_{1}^{k})(s_{1})$ and $\pi^{\mathrm{out}}=\pi^{k}=(\pi_{1}^{k},\cdots,\pi_{H}^{k}).$ 22: end if
23: end for
24: Return: The marginal policies of $\pi^{\mathrm{out}}:(\mu^{\mathrm{out}},\nu^{\mathrm{out}})$ .

$[H] \times \mathcal{S} \times \mathcal{A} \times \mathcal{B} \times \mathcal{S} \times [K]$ , the $\widetilde{N}_h^k(s, a, b, s')$ and $\widetilde{N}_h^k(s, a, b)$ from Privatizer satisfies:

(1) $|\widetilde{N}_h^k (s,a,b,s') - N_h^k (s,a,b,s')|\leq E_{\epsilon ,\beta},|\widetilde{N}_h^k (s,a,b) - N_h^k (s,a,b)|\leq E_{\epsilon ,\beta}.\widetilde{N}_h^k (s,a,b,s') > 0.$   
(2) $\widetilde{N}_h^k (s,a,b) = \sum_{s'\in \mathcal{S}}\widetilde{N}_h^k (s,a,b,s')\geq N_h^k (s,a,b).$

Given the private counts satisfying Assumption 3.1, the private estimate of transition kernel is defined as below.

$$
\widetilde {P} _ {h} ^ {k} (s ^ {\prime} | s, a, b) = \frac {\widetilde {N} _ {h} ^ {k} (s , a , b , s ^ {\prime})}{\widetilde {N} _ {h} ^ {k} (s , a , b)}, \forall (h, s, a, b, s ^ {\prime}, k). \tag {1}
$$

Remark 3.2. Assumption 3.1 is a generalization of Assumption 3.1 of Qiao and Wang [2023c] to the two-player setting. The assumption (2) guarantees that the private transition kernel estimate $\widetilde{P}_h^k (\cdot |s,a,b)$ is a valid probability distribution, which enables our usage of Bernstein-type bonus. Besides, $\widetilde{P}$ is close to the empirical transition kernel based on original visitation counts according to Assumption (1).

Algorithmic design. Following previous non-private approaches [Liu et al., 2021], DP-Nash-VI (Algorithm 1) maintains a pair of value functions $\overline{Q}$ and $Q$ which are the upper bound and lower bound of the Q value of the current policy when facing best responses (with high probability). More specifically, we use private visitation counts $\widetilde{N}_h^k$ to construct a private estimate of transition kernel $\widetilde{P}_h^k$ (line 7) and a pair of private bonus $\gamma_h^k$ (line 8) and $\Gamma_h^k$ (line 9). Intuitively, the first term of $\Gamma_h^k$ is derived from Bernstein's inequality while the second term is the additional bonus due to differential privacy. Next we do value iteration with bonuses to construct the UCB function $\overline{Q}_h^k$ (line 10) and the LCB function $\underline{Q}_h^k$ (line 11). The policy $\pi^k$ for the $k$ -th episode is calculated using the CCE function (discussed below) and we run $\pi^k$ to collect a trajectory (line 14,18). Finally, the Privatizer transfers the non-private counts to private ones for the next episode (line 19). The output policy $\pi^{\mathrm{out}}$ is chosen as the policy $\pi^k$ with minimal gap $(\overline{V}_1^k - \underline{V}_1^k)(s_1)$ (line 21). Decomposing the output policy, the output policy $(\mu^{\mathrm{out}}, \nu^{\mathrm{out}})$ for both players are the marginal policies of $\pi^{\mathrm{out}}$ , i.e. $\mu_h^{\mathrm{out}}(\cdot |s) = \sum_{b \in \mathcal{B}} \pi_h^{\mathrm{out}}(\cdot, b|s)$ and $\nu_h^{\mathrm{out}}(\cdot |s) = \sum_{a \in \mathcal{A}} \pi_h^{\mathrm{out}}(a, \cdot |s)$ for all $(h, s) \in [H] \times S$ .

Coarse Correlated Equilibrium (CCE). Intuitively speaking, CCE of a Markov Game is a potentially correlated policy where no player could benefit from unilateral unconditional deviation. As a computationally friendly relaxation of Nash Equilibrium, CCE has been applied by previous works [Xie et al., 2020, Liu et al., 2021] to design efficient algorithms. Formally, for any two functions $\overline{Q} (\cdot ,\cdot),\underline{Q} (\cdot ,\cdot):\mathcal{A}\times \mathcal{B}\to [0,H]$ , $\mathrm{CCE}(\overline{Q},\underline{Q})$ returns a policy $\pi \in \Delta (\mathcal{A}\times \mathcal{B})$ such that

$$
\mathbb {E} _ {(a, b) \sim \pi} \overline {{Q}} (a, b) \geq \max _ {a ^ {\prime}} \mathbb {E} _ {(a, b) \sim \pi} \overline {{Q}} (a ^ {\prime}, b),
$$

$$
\mathbb {E} _ {(a, b) \sim \pi} \underline {{Q}} (a, b) \leq \min _ {b ^ {\prime}} \mathbb {E} _ {(a, b) \sim \pi} \underline {{Q}} (a, b ^ {\prime}).
$$

Since Nash Equilibrium (NE) is a special case of CCE and a NE always exists, a CCE always exists. Moreover, a CCE can be derived in polynomial time via linear programming. Note that the policies given by CCE can be correlated for the two players, therefore deploying such policy requires the cooperation of both players (line 18).

# 4 Main results

We first state the regret analysis of DP-Nash-VI (Algorithm 1) based on Assumption 3.1, which can be combined with any Privatizers. The proof of Theorem 4.1 is sketched in Section 6 with details in the Appendix. Note that $(\mu^{k}, \nu^{k})$ denote the marginal policies of $\pi^{k}$ for both players.

Theorem 4.1. For any privacy budget $\epsilon > 0$ , failure probability $\beta \in [0,1]$ and any Privatizer satisfying Assumption 3.1, with probability at least $1 - \beta$ , the regret of DP-Nash-VI (Algorithm 1) is bounded by

$$
\operatorname{Regret} (K) = \sum_ {k = 1} ^ {K} \left[ V _ {1} ^ {\dagger , \nu^ {k}} (s _ {1}) - V _ {1} ^ {\mu^ {k}, \dagger} (s _ {1}) \right] \leq \widetilde {O} \left(\sqrt {H ^ {2} S A B T} + H ^ {2} S ^ {2} A B E _ {\epsilon , \beta}\right), \tag {2}
$$

where $K$ is the number of episodes and $T = HK$ .

Under the special case where the privacy budget $\epsilon \to \infty$ (i.e. there is no privacy concern), plugging $E_{\epsilon, \beta} = 0$ in Theorem 4.1 will imply a regret bound of $\widetilde{O}(\sqrt{H^2 SABT})$ . Such result directly matches the best known result for regret minimization without privacy constraints [Liu et al., 2021] and nearly matches the lower bound of $\Omega(\sqrt{H^2 S(A + B)T})$ [Bai and Jin, 2020]. Furthermore, under the special case of single-agent MDP (where $B = 1$ ), our result reduces to $\operatorname{Regret}(K) \leq \widetilde{O}(\sqrt{H^2 SAT} + H^2 S^2 AE_{\epsilon, \beta})$ . Such result matches the best known result under the same set of conditions (Theorem 4.1 of Qiao and Wang [2023c]). Therefore, Theorem 4.1 is a generalization of the best known results under MARL [Liu et al., 2021] and Differentially Private (single-agent) RL [Qiao and Wang, 2023c] simultaneously.

PAC guarantee. Recall that we output a policy $\pi^{out}$ whose marginal policies are $(\mu^{\mathrm{out}},\nu^{\mathrm{out}})$ . We highlight that the output policy for each player is a single Markov policy that is convenient to store and deploy. Moreover, as a corollary of the regret bound, we give a PAC bound for the output policy.

Theorem 4.2. For any privacy budget $\epsilon > 0$ , failure probability $\beta \in [0,1]$ and any Privatizer that satisfies Assumption 3.1, if the number of episodes satisfies that

$K \geq \widetilde{\Omega}\left(\frac{H^{3}SAB}{\alpha^{2}} + \min \left\{K' \mid \frac{H^{2}S^{2}ABE_{\epsilon,\beta}}{K'} \leq \alpha\right\}\right)$ , with probability $1 - \beta$ , ( $\mu^{\text{out}}, \nu^{\text{out}}$ ) is $\alpha$ -approximate Nash, i.e.,

$$
V _ {1} ^ {\dagger , \nu^ {\mathrm{out}}} (s _ {1}) - V _ {1} ^ {\mu^ {\mathrm{out}}, \dagger} (s _ {1}) \leq \alpha . \tag {3}
$$

The proof is deferred to Appendix B.4. Here the second term of the sample complexity bound $^{4}$ ensures that the additional cost due to DP is bounded by $O(\alpha)$ . The detailed PAC guarantees under the special cases where the Privatizer is either Central or Local will be provided in Section 5.

# 5 Privatizers for JDP and LDP

In this section, we propose Privatizers that provide DP guarantees (JDP or LDP) while satisfying Assumption 3.1. The proofs for this section can be found in Appendix C.

# 5.1 Central Privatizer for Joint DP

Given the number of episodes K, the Central Privatizer applies K-bounded Binary Mechanism [Chan et al., 2011] to privatize all the visitation counter streams $N_{h}^{k}(s,a,b)$ , $N_{h}^{k}(s,a,b,s')$ , thus protecting the information of all single users. Briefly speaking, Binary mechanism takes a stream of partial sums as input and outputs a surrogate stream satisfying differential privacy, while the error for each item scales only logarithmically on the length of the stream $^{5}$ . Here in multi-agent RL, for each $(h,s,a,b)$ , the stream $\{N_{h}^{k}(s,a,b)=\sum_{i=1}^{k-1}\mathbb{1}(s_{h}^{i},a_{h}^{i},b_{h}^{i}=s,a,b)\}_{k\in[K]}$ can be considered as the partial sums of $\{\mathbb{1}(s_{h}^{i},a_{h}^{i},b_{h}^{i}=s,a,b)\}$ . Therefore, after observing $\mathbb{1}(s_{h}^{k},a_{h}^{k},b_{h}^{k}=s,a,b)$ at the end of episode k, the Binary Mechanism will output a private version of $\sum_{i=1}^{k}\mathbb{1}(s_{h}^{i},a_{h}^{i},b_{h}^{i}=s,a,b)$ . However, Binary Mechanism alone does not satisfy (2) of Assumption 3.1, and a post-processing step is required. To sum up, we let the Central Privatizer follow the workflow below:

Given the privacy budget for JDP $\epsilon > 0$ ,

(1) For all $(h, s, a, b, s')$ , we apply Binary Mechanism (Algorithm 2 in Chan et al. [2011]) with input parameter $\epsilon' = \frac{\epsilon}{2H \log K}$ to privatize all the visitation counter streams $\{N_{h}^{k}(s, a, b)\}_{k \in [K]}$ and $\{N_{h}^{k}(s, a, b, s')\}_{k \in [K]}$ . We denote the output of Binary Mechanism by $\widehat{N}_{h}^{k}$ .   
(2)The private counts $\widetilde{N}_h^k$ are derived through the procedure in Section 5.3 with $E_{\epsilon ,\beta} = O(\frac{H}{\epsilon}\log (HSABK / \beta)^2)$ .

Our Central Privatizer satisfies the privacy guarantee below.

Lemma 5.1. For any possible $\epsilon, \beta$ , the Central Privatizer satisfies $\epsilon$ -JDP and Assumption 3.1 with $E_{\epsilon, \beta} = \widetilde{O}\left(\frac{H}{\epsilon}\right)$ .

Combining Lemma 5.1 with Theorem 4.1 and Theorem 4.2, we have the following regret & PAC guarantee under $\epsilon$ -JDP.

Theorem 5.2 (Results under JDP). For any possible $\epsilon, \beta$ , with probability $1 - \beta$ , the regret from running DP-Nash-VI (Algorithm 1) instantiated with Central Privatizer satisfies:

$$
\operatorname{Regret} (K) \leq \widetilde {O} \left(\sqrt {H ^ {2} S A B T} + H ^ {3} S ^ {2} A B / \epsilon\right). \tag {4}
$$

Moreover, if the number of episodes $K$ is larger than $\widetilde{\Omega}\left(\frac{H^3 SAB}{\alpha^2} + \frac{H^3 S^2 AB}{\epsilon\alpha}\right)$ , with probability $1 - \beta$ , the output policy $(\mu^{\mathrm{out}}, \nu^{\mathrm{out}})$ is $\alpha$ -approximate Nash.

Similar to the single-agent (MDP) setting $(B=1)$ , the additional cost due to JDP is a lower order term under the most prevalent regime where the privacy budget $\epsilon$ is a constant. When applied to the single-agent case, our regret matches the best known regret $\widetilde{O}\left(\sqrt{H^{2}SAT}+H^{3}S^{2}A/\epsilon\right)$ [Qiao and Wang, 2023c]. Moreover, when compared to the regret lower bound below, our main term is nearly optimal while the lower order term has optimal dependence on $\epsilon$ .

Theorem 5.3. For any algorithm Alg satisfying $\epsilon$ -JDP, there exists a Markov Game such that the expected regret from running Alg for K episodes (T = HK steps) satisfies:

$$
\mathbb {E} \left[ \operatorname{Regret} (K) \right] \geq \widetilde {\Omega} \left(\sqrt {H ^ {2} S (A + B) T} + \frac {H S (A + B)}{\epsilon}\right).
$$

The regret lower bound results from the lower bound for the non-private learning [Bai and Jin, 2020] and an adaptation of the lower bound under JDP guarantees [Vietri et al., 2020] to the multi-player setting. Details are deferred to the appendix.

# 5.2 Local Privatizer for Local DP

At the end of episode $k$ , the Local Privatizer perturbs the statistics calculated from the new trajectory before sending it to the agent. Since the set of original visitation counts $\{\sigma_h^k (s,a,b) = \mathbb{1}(s_h^k,a_h^k,b_h^k = s,a,b)\}_{(h,s,a,b)}$ has $\ell_1$ sensitivity $H$ , we can achieve $\frac{\epsilon}{2}$ -LDP by directly adding Laplace noise, i.e., $\widetilde{\sigma}_h^k (s,a,b) = \sigma_h^k (s,a,b) + \mathrm{Lap}(\frac{2H}{\epsilon})$ . Similarly, repeating the above perturbation to $\{\mathbb{1}(s_h^k,a_h^k,b_h^k,s_{h + 1}^k = s,a,b,s')\}_{(h,s,a,b,s')}$ will lead to identical results. Therefore, the Local Privatizer with budget $\epsilon$ is as below:

(1) We perturb $\sigma_h^k (s,a,b) = \mathbb{1}(s_h^k,a_h^k,b_h^k = s,a,b)$ and $\sigma_h^k (s,a,b,s') = \mathbb{1}(s_h^k,a_h^k,b_h^k,s_{h + 1}^k = s,a,b,s')$ by adding independent Laplace noises: for all $(h,s,a,b,s',k)$ ,

$$
\widetilde {\sigma} _ {h} ^ {k} (s, a, b) = \sigma_ {h} ^ {k} (s, a, b) + \operatorname{Lap} \left(\frac {2 H}{\epsilon}\right), \tag {5}
$$

$$
\widetilde {\sigma} _ {h} ^ {k} (s, a, b, s ^ {\prime}) = \sigma_ {h} ^ {k} (s, a, b, s ^ {\prime}) + \operatorname{Lap} \left(\frac {2 H}{\epsilon}\right).
$$

(2) Then the noisy counts are derived according to

$$
\widehat {N} _ {h} ^ {k} (s, a, b) = \sum_ {i = 1} ^ {k - 1} \widetilde {\sigma} _ {h} ^ {i} (s, a, b), \tag {6}
$$

$$
\widehat {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) = \sum_ {i = 1} ^ {k - 1} \widetilde {\sigma} _ {h} ^ {i} (s, a, b, s ^ {\prime}),
$$

and the private counts $\widetilde{N}_h^k$ are solved through the procedure in Section 5.3 with $E_{\epsilon ,\beta} = O(\frac{H}{\epsilon}\sqrt{K\log(H S A B K / \beta)})$ .

Our Local Privatizer satisfies the privacy guarantee below.

Lemma 5.4. For any possible $\epsilon, \beta$ , the Local Privatizer satisfies $\epsilon$ -LDP and Assumption 3.1 with $E_{\epsilon, \beta} = \widetilde{O}\left(\frac{H}{\epsilon}\sqrt{K}\right)$ .

Combining Lemma 5.4 with Theorem 4.1 and Theorem 4.2, we have the following regret & PAC guarantee under $\epsilon$ -LDP.

Theorem 5.5 (Results under LDP). For any possible $\epsilon, \beta$ , with probability $1 - \beta$ , the regret from running DP-Nash-VI (Algorithm 1) instantiated with Local Privatizer satisfies:

$$
\operatorname{Regret} (K) \leq \widetilde {O} \left(\sqrt {H ^ {2} S A B T} + S ^ {2} A B \sqrt {H ^ {5} T} / \epsilon\right). \tag {7}
$$

Moreover, if the number of episodes $K$ is larger than $\widetilde{\Omega}\left(\frac{H^3 SAB}{\alpha^2} + \frac{H^6 S^4 A^2 B^2}{\epsilon^2 \alpha^2}\right)$ , with probability $1 - \beta$ , the output policy $(\mu^{\mathrm{out}}, \nu^{\mathrm{out}})$ is $\alpha$ -approximate Nash.

Similar to the single-agent case, the additional cost due to LDP is a multiplicative factor to the regret bound. When applied to the single-agent case, our regret matches the best known regret $\widetilde{O}\left(\sqrt{H^2SAT} + S^2A\sqrt{H^5T}/\epsilon\right)$ [Qiao and Wang, 2023c]. Moreover, we state the lower bound below.

Theorem 5.6. For any algorithm Alg satisfying $\epsilon$ -LDP, there exists a Markov Game such that the expected regret from running Alg for $K$ episodes ( $T = HK$ steps) satisfies:

$$
\mathbb {E} \left[ \operatorname{Regret} (K) \right] \geq \widetilde {\Omega} \left(\sqrt {H ^ {2} S (A + B) T} + \frac {\sqrt {H S (A + B) T}}{\epsilon}\right).
$$

The lower bound is adapted from Garcelon et al. [2021]. While our regret has optimal dependence on $\epsilon$ and $K$ , the optimal dependence on $H, S, A, B$ remains open.

# 5.3 The post-processing step

Now we introduce the post-processing step. At the end of episode $k$ , given the noisy counts $\widehat{N}_h^k (s,a,b)$ and $\widehat{N}_h^k (s,a,b,s')$ for all $(h,s,a,b,s')$ , the private visitation counts are constructed as following: for all $(h,s,a,b)$ ,

$$
\left\{\widetilde {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right\} _ {s ^ {\prime} \in \mathcal {S}} = \underset {\{x _ {s ^ {\prime}} \} _ {s ^ {\prime} \in \mathcal {S}}} {\operatorname{argmin}} \max _ {s ^ {\prime} \in \mathcal {S}} \left| x _ {s ^ {\prime}} - \widehat {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right|
$$

such that $\left|\sum_{s' \in S} x_{s'} - \widehat{N}_h^k(s, a, b)\right| \leq \frac{E_{\epsilon,\beta}}{4}$ and $x_{s'} \geq 0$ , $\forall s'$ . (8)

$$
\widetilde {N} _ {h} ^ {k} (s, a, b) = \sum_ {s ^ {\prime} \in \mathcal {S}} \widetilde {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}).
$$

Lastly, we add a constant term to each count to ensure that with high probability, no underestimation will happen.

$$
\widetilde {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) = \widetilde {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) + \frac {E _ {\epsilon , \beta}}{2 S}, \tag {9}
$$

$$
\widetilde {N} _ {h} ^ {k} (s, a, b) = \widetilde {N} _ {h} ^ {k} (s, a, b) + \frac {E _ {\epsilon , \beta}}{2}.
$$

Remark 5.7. Solving problem (8) is equivalent to solving:

min $t$ , s.t. $\left|x_{s'} - \widehat{N}_h^k(s, a, b, s')\right| \leq t$ , $x_{s'} \geq 0$ , $\forall s' \in \mathcal{S}$ ,

$$
\left| \sum_ {s ^ {\prime} \in \mathcal {S}} x _ {s ^ {\prime}} - \widehat {N} _ {h} ^ {k} (s, a, b) \right| \leq \frac {E _ {\epsilon , \beta}}{4},
$$

which is a Linear Programming problem with $O(S)$ variables and $O(S)$ linear constraints. This can be solved in polynomial time [Nemhauser and Wolsey, 1988]. Note that the computation of CCE (line 14 in Algorithm 1) is also a LP problem, therefore the computational complexity of DP-Nash-VI is dominated by $O(\text{HSABK})$ Linear Programming problems, which is computationally friendly.

We summarize the properties of private counts $\widetilde{N}_h^k$ below, which says that the post-processing step ensures that our private transition kernel estimate is a valid probability distribution while only enlarging the error by a constant factor.

Lemma 5.8. Suppose $\widehat{N}_h^k$ satisfies that with probability $1 - \frac{\beta}{3}$ , uniformly over all $(h, s, a, b, s', k)$ it holds that

$$
\left| \widehat {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) - N _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right| \leq \frac {E _ {\epsilon , \beta}}{4},
$$

$$
\left| \widehat {N} _ {h} ^ {k} (s, a, b) - N _ {h} ^ {k} (s, a, b) \right| \leq \frac {E _ {\epsilon , \beta}}{4},
$$

then the $\widetilde{N}_h^k$ derived above satisfies Assumption 3.1.

# 5.4 Some discussions

In this part, we generalize the Privatizers in Qiao and Wang [2023c] (for single-agent case) to the two-player setting, which enables our usage of Bernstein-type bonuses. Such techniques lead to a tight regret analysis and a near-optimal “non-private part” of the regret bound eventually.

Meanwhile, the additional cost due to DP has sub-optimal dependence on parameters regarding the Markov Game. The issue appears even in the single-agent case and is considered to be inherent to model-based algorithms due to the explicit estimation of private transitions [Garcelon et al., 2021]. The improvement requires new algorithmic designs (e.g., private Q-learning) and we leave those as future works.

Lastly, the Laplace Mechanism can be replaced with other mechanisms, such as Gaussian Mechanism [Dwork et al., 2014] with approximate DP guarantee (or zCDP). The regret and PAC guarantees are readily derived by plugging in the corresponding $E_{\epsilon,\beta}$ to Theorem 4.1 and Theorem 4.2.

# 6 Proof overview

In this section, we provide a proof sketch of Theorem 4.1, which can further imply the PAC guarantee (Theorem 4.2) and the regret bounds under JDP (Theorem 5.2) or LDP (Theorem 5.5). The proof consists of the following steps:

(1) Bound the difference between the private statistics and their non-private counterparts.   
(2)Prove that UCB and LCB hold with high probability.   
(3) Bound the regret via telescoping over time steps and replace the private terms by non-private ones.

Below we explain the key steps in detail. Recall that $N_h^k$ denotes the real visitation counts, while $\widetilde{N}_h^k, \widetilde{P}_h^k$ are the private visitation counts and private transition kernel respectively.

Step (1). According to Assumption 3.1 and standard concentration inequalities, we provide high probability upper bounds for $\| \widetilde{P}_h^k (\cdot |s,a,b) - P_h(\cdot |s,a,b)\| _1$ and $|\widetilde{P}_h^k (s'|s,a,b) - P_h(s'|s,a,b)|$ . Besides we upper bound the following key term $|(\widetilde{P}_h^k -P_h)\cdot V_{h + 1}^\star (s,a,b)|$ by

$\widetilde{O}\left(\sqrt{\operatorname{Var}_{\widehat{P}_h^k(\cdot|s,a,b)}V_{h+1}^\star(\cdot)/\widetilde{N}_h^k(s,a,b)} + HSE_{\epsilon,\beta}/\widetilde{N}_h^k(s,a,b)}\right)$ . Details are deferred to Appendix B.1.

Step (2). Then we prove that UCB and LCB hold with high probability via backward induction over timesteps (Appendix B.2). More specifically, the variance term of $\Gamma_{h}^{k}$ is the private Bernstein-type bonus, while the difference between the private variance and its non-private counterpart can be bounded by $\gamma_{h}^{k}$ and the lower order terms in $\Gamma_{h}^{k}$ .

Step (3). Lastly, the regret can be bounded by telescoping:

$$
\begin{array}{l} \operatorname{Regret} (K) \leq \underbrace {O \left(\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \Gamma_ {h} ^ {k} (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k})\right)} _ {\text { bound   by   non - private   terms }} \\ \begin{array}{l} \leq \widetilde {O} \left(\underbrace {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \sqrt {\frac {\mathrm{Var} _ {P _ {h} (\cdot | s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k})} V _ {h + 1} ^ {\pi^ {k}}}{N _ {h} ^ {k} (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k})}}} _ {\text {bound by Cauchy - Schwarz inequality and L.T.V.}} + \underbrace {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {H S E _ {\epsilon , \beta}}{N _ {h} ^ {k} (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k})}} _ {\leq H ^ {2} S ^ {2} A B E _ {\epsilon , \beta^ {\iota}}}\right) \\ \leq \widetilde {O} (\sqrt {H ^ {2} S A B T} + H ^ {2} S ^ {2} A B E _ {\epsilon , \beta}). \end{array} \\ \end{array}
$$

The details about each inequality above and the lower order terms we ignore are deferred to Appendix B.3.

# 7 Conclusion

We take the initial steps to study trajectory-wise privacy protection in multi-agent RL. We extend the definitions of Joint DP and Local DP to the multi-player RL setting. In addition, we design a provably-efficient algorithm: DP-Nash-VI (Algorithm 1) that could satisfy either of the two DP constraints with corresponding regret guarantee. Moreover, our regret bounds strictly generalize the best known results under DP single-agent RL. There are various interesting future directions, such as improving the additional cost due to DP via model-free approaches and considering Markov Games with function approximations. We believe the techniques in this paper could serve as basic building blocks.

# Acknowledgments

The research is partially supported by NSF Awards #2007117 and #2048091.

# References

Alex Ayoub, Zeyu Jia, Csaba Szepesvari, Mengdi Wang, and Lin Yang. Model-based reinforcement learning with value-targeted regression. In International Conference on Machine Learning, pages 463–474. PMLR, 2020.   
Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 263–272. JMLR. org, 2017.

Yoram Bachrach, Richard Everett, Edward Hughes, Angeliki Lazaridou, Joel Z Leibo, Marc Lanc-tot, Michael Johanson, Wojciech M Czarnecki, and Thore Graepel. Negotiating team formation using deep reinforcement learning. Artificial Intelligence, 288:103356, 2020.   
Yu Bai and Chi Jin. Provable self-play algorithms for competitive reinforcement learning. In International Conference on Machine Learning, pages 551–560. PMLR, 2020.   
Yu Bai, Chi Jin, and Tiancheng Yu. Near-optimal reinforcement learning with self-play. Advances in neural information processing systems, 33:2159–2170, 2020.   
Borja Balle, Maziar Gomrokchi, and Doina Precup. Differentially private policy evaluation. In International Conference on Machine Learning, pages 2130–2138. PMLR, 2016.   
Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar. When is memorization of irrelevant training data necessary for high-accuracy learning? In ACM SIGACT Symposium on Theory of Computing, pages 123–132, 2021.   
Noam Brown and Tuomas Sandholm. Superhuman ai for multiplayer poker. Science, 365(6456):885–890, 2019.   
Mark Bun and Thomas Steinke. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In Theory of Cryptography Conference, pages 635–658. Springer, 2016.   
Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating and testing unintended memorization in neural networks. In USENIX Security Symposium (USENIX Security 19), pages 267–284, 2019.   
T-H Hubert Chan, Elaine Shi, and Dawn Song. Private and continual release of statistics. ACM Transactions on Information and System Security (TISSEC), 14(3):1–24, 2011.   
Sayak Ray Chowdhury and Xingyu Zhou. Differentially private regret minimization in episodic markov decision processes. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022.   
Sayak Ray Chowdhury, Xingyu Zhou, and Ness Shroff. Adaptive control of differentially private linear quadratic systems. In 2021 IEEE International Symposium on Information Theory (ISIT), pages 485–490. IEEE, 2021.   
Sayak Ray Chowdhury, Xingyu Zhou, and Nagarajan Natarajan. Differentially private reward estimation with preference feedback. arXiv preprint arXiv:2310.19733, 2023.   
Qiwen Cui, Kaiqing Zhang, and Simon Du. Breaking the curse of multiagents in a large state space: Rl in markov games with independent linear function approximation. In The Thirty Sixth Annual Conference on Learning Theory, pages 2651–2652. PMLR, 2023.   
Chris Cundy and Stefano Ermon. Privacy-constrained policies via mutual information regularized policy gradients. arXiv preprint arXiv:2012.15019, 2020.   
Christoph Dann, Tor Lattimore, and Emma Brunskill. Unifying pac and regret: Uniform pac bounds for episodic reinforcement learning. In Advances in Neural Information Processing Systems, pages 5713–5723, 2017.

John C Duchi, Michael I Jordan, and Martin J Wainwright. Local privacy and statistical minimax rates. In 2013 IEEE 54th Annual Symposium on Foundations of Computer Science, pages 429–438. IEEE, 2013.   
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, pages 265–284. Springer, 2006.   
Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Found. Trends Theor. Comput. Sci., 9(3-4):211–407, 2014.   
Jerzy Filar and Koos Vrieze. Competitive Markov decision processes. Springer Science & Business Media, 2012.   
Evrard Garcelon, Vianney Perchet, Ciara Pike-Burke, and Matteo Pirotta. Local differential privacy for regret minimization in reinforcement learning. Advances in Neural Information Processing Systems, 34, 2021.   
Parham Gohari, Matthew Hale, and Ufuk Topcu. Privacy-engineered value decomposition networks for cooperative multi-agent reinforcement learning. In 2023 62nd IEEE Conference on Decision and Control (CDC), pages 8038–8044. IEEE, 2023.   
Md Tamjid Hossain and John WT Lee. Hiding in plain sight: Differential privacy noise exploitation for evasion-resilient localized poisoning attacks in multiagent reinforcement learning. In 2023 International Conference on Machine Learning and Cybernetics (ICMLC), pages 209–216. IEEE, 2023.   
Md Tamjid Hossain, Hung Manh La, Shahriar Badsha, and Anton Netchaev. Brnes: Enabling security and privacy-aware experience sharing in multiagent robotic and autonomous systems. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 9269–9276. IEEE, 2023.   
Justin Hsu, Zhiyi Huang, Aaron Roth, Tim Roughgarden, and Zhiwei Steven Wu. Private matchings and allocations. In Proceedings of the forty-sixth annual ACM symposium on Theory of computing, pages 21–30, 2014.   
Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research, 11(4), 2010.   
Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? In Advances in Neural Information Processing Systems, pages 4863-4873, 2018.   
Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory, pages 2137-2143. PMLR, 2020.   
Chi Jin, Qinghua Liu, Yuanhao Wang, and Tiancheng Yu. V-learning—a simple, efficient, decentralized algorithm for multiagent rl. arXiv preprint arXiv:2110.14555, 2021.   
Peter Kairouz, Brendan McMahan, Shuang Song, Om Thakkar, Abhradeep Thakurta, and Zheng Xu. Practical and private (deep) learning without sampling or shuffling. In International Conference on Machine Learning, pages 5213–5225. PMLR, 2021.

Michael Kearns, Mallesh Pai, Aaron Roth, and Jonathan Ullman. Mechanism design in large games: Incentives and privacy. In Proceedings of the 5th conference on Innovations in theoretical computer science, pages 403–410, 2014.   
Jonathan Lebensold, William Hamilton, Borja Balle, and Doina Precup. Actor critic with differentially private critic. arXiv preprint arXiv:1910.05876, 2019.   
Chonghua Liao, Jiafan He, and Quanquan Gu. Locally differentially private reinforcement learning for linear mixture markov decision processes. In Asian Conference on Machine Learning, pages 627–642. PMLR, 2023.   
Qinghua Liu, Tiancheng Yu, Yu Bai, and Chi Jin. A sharp analysis of model-based reinforcement learning with self-play. In International Conference on Machine Learning, pages 7001–7010. PMLR, 2021.   
Paul Luyo, Evrard Garcelon, Alessandro Lazaric, and Matteo Pirotta. Differentially private exploration in reinforcement learning with linear representation. arXiv preprint arXiv:2112.01585, 2021.   
Weichao Mao, Lin Yang, Kaiqing Zhang, and Tamer Basar. On improving model-free algorithms for decentralized multi-agent reinforcement learning. In International Conference on Machine Learning, pages 15007-15049. PMLR, 2022.   
George Nemhauser and Laurence Wolsey. Polynomial-time algorithms for linear programming. Integer and Combinatorial Optimization, pages 146-181, 1988.   
Dung Daniel T Ngo, Giuseppe Vietri, and Steven Wu. Improved regret for differentially private exploration in linear mdp. In International Conference on Machine Learning, pages 16529–16552. PMLR, 2022.   
Hajime Ono and Tsubasa Takahashi. Locally private distributed reinforcement learning. arXiv preprint arXiv:2001.11718, 2020.   
Dan Qiao and Yu-Xiang Wang. Near-optimal deployment efficiency in reward-free reinforcement learning with linear function approximation. International Conference on Learning Representations, 2023a.   
Dan Qiao and Yu-Xiang Wang. Offline reinforcement learning with differential privacy. Advances in Neural Information Processing Systems, 2023b.   
Dan Qiao and Yu-Xiang Wang. Near-optimal differentially private reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pages 9914–9940. PMLR, 2023c.   
Dan Qiao and Yu-Xiang Wang. Near-optimal reinforcement learning with self-play under adaptivity constraints. arXiv preprint arXiv:2402.01111, 2024.   
Dan Qiao, Ming Yin, Ming Min, and Yu-Xiang Wang. Sample-efficient reinforcement learning with loglog(T) switching cost. In International Conference on Machine Learning, pages 18031-18061. PMLR, 2022.   
Dan Qiao, Ming Yin, and Yu-Xiang Wang. Logarithmic switching cost in reinforcement learning beyond linear mdps. arXiv preprint arXiv:2302.12456, 2023.

Shai Shalev-Shwartz, Shaked Shammah, and Amnon Shashua. Safe, multi-agent, reinforcement learning for autonomous driving. arXiv preprint arXiv:1610.03295, 2016.   
Lloyd S Shapley. Stochastic games. Proceedings of the national academy of sciences, 39(10):1095-1100, 1953.   
Roshan Shariff and Or Sheffet. Differentially private contextual linear bandits. Advances in Neural Information Processing Systems, 31, 2018.   
Ali Shavandi and Majid Khedmati. A multi-agent deep reinforcement learning framework for algorithmic trading in financial markets. Expert Systems with Applications, 208:118124, 2022.   
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354–359, 2017.   
Imdad Ullah, Najm Hassan, Sukhpal Singh Gill, Basem Suleiman, Tariq Ahamed Ahanger, Zawar Shah, Junaid Qadir, and Salil S Kanhere. Privacy preserving large language models: Chatgpt case study based vision and framework. arXiv preprint arXiv:2310.12523, 2023.   
Giuseppe Vietri, Borja Balle, Akshay Krishnamurthy, and Steven Wu. Private reinforcement learning with pac and regret guarantees. In International Conference on Machine Learning, pages 9754–9764. PMLR, 2020.   
Baoxiang Wang and Nidhi Hegde. Privacy-preserving q-learning with functional noise in continuous spaces. Advances in Neural Information Processing Systems, 32, 2019.   
Yuanhao Wang, Qinghua Liu, Yu Bai, and Chi Jin. Breaking the curse of multiagency: Provably efficient decentralized multi-agent rl with function approximation. arXiv preprint arXiv:2302.06606, 2023.   
Fan Wu, Huseyin A Inan, Arturs Backurs, Varun Chandrasekaran, Janardhan Kulkarni, and Robert Sim. Privately aligning language models with reinforcement learning. arXiv preprint arXiv:2310.16960, 2023a.   
Yulian Wu, Xingyu Zhou, Sayak Ray Chowdhury, and Di Wang. Differentially private episodic reinforcement learning with heavy-tailed rewards. arXiv preprint arXiv:2306.01121, 2023b.   
Qiaomin Xie, Yudong Chen, Zhaoran Wang, and Zhuoran Yang. Learning zero-sum simultaneous-move markov games using function approximation and correlated equilibrium. In Conference on learning theory, pages 3674-3682. PMLR, 2020.   
Tengyang Xie, Philip S Thomas, and Gerome Miklau. Privacy preserving off-policy evaluation. arXiv preprint arXiv:1902.00174, 2019.   
Deheng Ye, Guibin Chen, Wen Zhang, Sheng Chen, Bo Yuan, Bo Liu, Jia Chen, Zhao Liu, Fuhao Qiu, Hongsheng Yu, et al. Towards playing full moba games with deep reinforcement learning. Advances in Neural Information Processing Systems, 33:621–632, 2020.   
Canzhe Zhao, Yanjie Ze, Jing Dong, Baoxiang Wang, and Shuai Li. Differentially private temporal difference learning with stochastic nonconvex-strongly-concave optimization. In Proceedings of

the Sixteenth ACM International Conference on Web Search and Data Mining, pages 985–993, 2023a.   
Canzhe Zhao, Yanjie Ze, Jing Dong, Baoxiang Wang, and Shuai Li. Dpmac: differentially private communication for cooperative multi-agent reinforcement learning. arXiv preprint arXiv:2308.09902, 2023b.   
Fuheng Zhao, Dan Qiao, Rachel Redberg, Divyakant Agrawal, Amr El Abbadi, and Yu-Xiang Wang. Differentially private linear sketches: Efficient implementations and applications. Advances in Neural Information Processing Systems, 35:12691–12704, 2022.   
Xingyu Zhou. Differentially private reinforcement learning with linear function approximation. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 6(1):1–27, 2022.

# A Extended related works

Differentially private reinforcement learning. The stream of research on DP RL started from the offline setting. Balle et al. [2016] first studied privately evaluating the value of a fixed policy from running it for several episodes (the on policy setting). Later, Xie et al. [2019] considered a more general setting of DP off policy evaluation. Recently, Qiao and Wang [2023b] provided the first results for offline reinforcement learning with DP guarantees.

More efforts focused on solving regret minimization. Under the setting of tabular MDP, Vietri et al. [2020] designed PUCB by privatizing UBEV [Dann et al., 2017] to satisfy Joint DP. Besides, under the constraints of Local DP, Garcelon et al. [2021] designed LDP-OBI based on UCRL2 [Jaksch et al., 2010]. Chowdhury and Zhou [2022] designed a general framework for both JDP and LDP based on UCBVI [Azar et al., 2017], and improved upon previous results. Finally, the best known results are obtained by Qiao and Wang [2023c] via incorporating Bernstein-type bonuses. Meanwhile, Wu et al. [2023b] studied the case with heavy-tailed rewards. Under linear MDP, the only algorithm with JDP guarantee: Private LSVI-UCB [Ngo et al., 2022] is a private and low switching $^{6}$ version of LSVI-UCB [Jin et al., 2020], while LDP under linear MDP still remains open. Under linear mixture MDP, LinOpt-VI-Reg [Zhou, 2022] generalized UCRL-VTR [Ayoub et al., 2020] to guarantee JDP, while Liao et al. [2023] also privatized UCRL-VTR for LDP guarantee. In addition, Luyo et al. [2021] provided a unified framework for analyzing joint and local DP exploration.

There are several other works regarding DP RL. Wang and Hegde [2019] proposed privacy-preserving Q-learning to protect the reward information. Ono and Takahashi [2020] studied the problem of distributed reinforcement learning under LDP. Lebensold et al. [2019] presented an actor critic algorithm with differentially private critic. Cundy and Ermon [2020] tackled DP-RL under the policy gradient framework. Chowdhury et al. [2021] considered the adaptive control of differentially private linear quadratic (LQ) systems. Zhao et al. [2023a] studied differentially private temporal difference (TD) learning. Chowdhury et al. [2023] analyzed reward estimation with preference feedback under the constraints of DP. Hossain and Lee [2023], Hossain et al. [2023], Zhao et al. [2023b], Gohari et al. [2023] focused on the privatization of communications between multiple agents in multi-agent RL. For applications, DP RL was applied to protect sensitive information in natural language processing and large language models (LLM) [Ullah et al., 2023, Wu et al., 2023a]. Meanwhile, Zhao et al. [2022] considered linear sketches with DP.

# B Proof of main theorems

In this section, we prove Theorem 4.1 and Theorem 4.2.

# B.1 Properties of private estimations

We begin with some concentration results about our private transition kernel estimate $\widetilde{P}$ that will be useful for the proof. Throughout the paper, let the non-private empirical transition kernel

be:

$$
\widehat {P} _ {h} ^ {k} (s ^ {\prime} | s, a, b) = \frac {N _ {h} ^ {k} (s , a , b , s ^ {\prime})}{N _ {h} ^ {k} (s , a , b)}, \forall (h, s, a, b, s ^ {\prime}, k). \tag {10}
$$

In addition, recall that our private transition kernel estimate is defined as below.

$$
\widetilde {P} _ {h} ^ {k} (s ^ {\prime} | s, a, b) = \frac {\widetilde {N} _ {h} ^ {k} (s , a , b , s ^ {\prime})}{\widetilde {N} _ {h} ^ {k} (s , a , b)}, \forall (h, s, a, b, s ^ {\prime}, k). \tag {11}
$$

Now we are ready to list the properties below. Note that $\iota=\log(30HSABK/\beta)$ throughout the paper.

Lemma B.1. With probability $1 - \frac{\beta}{15}$ , for all $(h, s, a, b, k) \in [H] \times S \times A \times B \times [K]$ , it holds that:

$$
\left\| \widetilde {P} _ {h} ^ {k} (\cdot | s, a, b) - P _ {h} (\cdot | s, a, b) \right\| _ {1} \leq 2 \sqrt {\frac {S \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {2 S E _ {\epsilon , \beta}}{\widetilde {N} _ {h} ^ {k} (s , a , b)}, \tag {12}
$$

$$
\left\| \widetilde {P} _ {h} ^ {k} (\cdot | s, a, b) - \widehat {P} _ {h} ^ {k} (\cdot | s, a, b) \right\| _ {1} \leq \frac {2 S E _ {\epsilon , \beta}}{\widetilde {N} _ {h} ^ {k} (s , a , b)}. \tag {13}
$$

Proof of Lemma B.1. The proof is a direct generalization of Lemma B.2 and Remark B.3 in Qiao and Wang [2023c] to the two-player setting. $\square$

Lemma B.2. With probability $1 - \frac{2\beta}{15}$ , for all $(h, s, a, b, s', k) \in [H] \times S \times A \times B \times S \times [K]$ , it holds that:

$$
\left| \widetilde {P} _ {h} ^ {k} (s ^ {\prime} | s, a, b) - P _ {h} (s ^ {\prime} | s, a, b) \right| \leq 2 \sqrt {\frac {\min \{P _ {h} (s ^ {\prime} | s , a , b) , \widetilde {P} _ {h} ^ {k} (s ^ {\prime} | s , a , b) \} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {2 E _ {\epsilon , \beta^ {\iota}}}{\widetilde {N} _ {h} ^ {k} (s , a , b)}, \tag {14}
$$

$$
\left| \widetilde {P} _ {h} ^ {k} (s ^ {\prime} | s, a, b) - \widehat {P} _ {h} ^ {k} (s ^ {\prime} | s, a, b) \right| \leq \frac {2 E _ {\epsilon , \beta}}{\widetilde {N} _ {h} ^ {k} (s , a , b)}. \tag {15}
$$

Proof of Lemma B.2. The proof is a direct generalization of Lemma B.4 and Remark B.5 in Qiao and Wang [2023c] to the two-player setting. $\square$

Lemma B.3. With probability $1 - \frac{2\beta}{15}$ , for all $(h, s, a, b, k) \in [H] \times S \times A \times B \times [K]$ , it holds that:

$$
\left| \left(\widetilde {P} _ {h} ^ {k} - P _ {h}\right) \cdot V _ {h + 1} ^ {\star} (s, a, b) \right| \leq \min \left\{\sqrt {\frac {2 \operatorname{Var} _ {P _ {h} (\cdot | s , a , b)} V _ {h + 1} ^ {\star} (\cdot) \cdot \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}}, \sqrt {\frac {2 \operatorname{Var} _ {\widehat {P} _ {h} ^ {k} (\cdot | s , a , b)} V _ {h + 1} ^ {\star} (\cdot) \cdot \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} \right\} + \frac {2 H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}, \tag {16}
$$

$$
\left| \left(\widetilde {P} _ {h} ^ {k} - \widehat {P} _ {h} ^ {k}\right) \cdot V _ {h + 1} ^ {\star} (s, a, b) \right| \leq \frac {2 H S E _ {\epsilon , \beta}}{\widetilde {N} _ {h} ^ {k} (s , a , b)}. \tag {17}
$$

Proof of Lemma B.3. The proof is a direct generalization of Lemma B.6 and Remark B.7 in Qiao and Wang [2023c] to the two-player setting. $\square$

According to a union bound, the following lemma holds.

Lemma B.4. Under the high probability event that Assumption 3.1 holds, with probability at least $1 - \frac{\beta}{3}$ , the conclusions in Lemma B.1, Lemma B.2, Lemma B.3 hold simultaneously.

Throughout the proof, we will assume that Assumption 3.1 and Lemma B.4 hold, which will happen with high probability. Before we prove the main theorems, we present the following lemma which bounds the two variances.

Lemma B.5 (Lemma C.5 of Qiao and Wang [2023b]). For any function $V \in R^{S}$ such that $\|V\|_{\infty} \leq H$ , it holds that

$$
\left| \sqrt {\operatorname{Var} _ {\widetilde {P} _ {h} ^ {k} (\cdot | s , a , b)} (V)} - \sqrt {\operatorname{Var} _ {\widehat {P} _ {h} ^ {k} (\cdot | s , a , b)} (V)} \right| \leq \sqrt {3} H \cdot \sqrt {\left\| \widetilde {P} _ {h} ^ {k} (\cdot | s , a , b) - \widehat {P} _ {h} ^ {k} (\cdot | s , a , b) \right\| _ {1}}. \tag {18}
$$

In addition, according to Lemma B.1, the left hand side can be further bounded by

$$
\left| \sqrt {\operatorname{Var} _ {\widetilde {P} _ {h} ^ {k} (\cdot | s , a , b)} (V)} - \sqrt {\operatorname{Var} _ {\widehat {P} _ {h} ^ {k} (\cdot | s , a , b)} (V)} \right| \leq 3 H \sqrt {\frac {S E _ {\epsilon , \beta}}{\widetilde {N} _ {h} ^ {k} (s , a , b)}}. \tag {19}
$$

# B.2 Proof of UCB and LCB

For notational simplicity, for $V \in \mathbb{R}^S$ such that $\| V \|_{\infty} \leq H$ , we define

$$
\widetilde {V} _ {h} ^ {k} V (s, a, b) = \operatorname{Var} _ {\widetilde {P} _ {h} ^ {k} (\cdot | s, a, b)} V (\cdot), \quad V _ {h} V (s, a, b) = \operatorname{Var} _ {P _ {h} (\cdot | s, a, b)} V (\cdot). \tag {20}
$$

Then the bonus term $\Gamma$ can be represented as below ( $C_{2}$ is the universal constant in Algorithm 1).

$$
\Gamma_ {h} ^ {k} (s, a, b) = C _ {2} \sqrt {\frac {\widetilde {V} _ {h} ^ {k} \left(\frac {\overline {{{V}}} _ {h + 1} ^ {k} + \underline {{{V}}} _ {h + 1} ^ {k}}{2}\right) (s , a , b) \cdot \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {C _ {2} H S E _ {\epsilon , \beta} \cdot \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)} + \frac {C _ {2} H ^ {2} S \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}. \tag {21}
$$

We state the following lemma that can bound the lower order term, which is helpful for proving UCB and LCB.

Lemma B.6. Suppose Assumption 3.1 and Lemma B.4 hold, then there exists a universal constant $c_{1} > 0$ such that: if function $g(s)$ satisfies $|g|(s) \leq (\overline{V}_{h + 1}^{k} - \underline{V}_{h + 1}^{k})(s)$ , then it holds that:

$$
\begin{array}{l} \left| \left(\widetilde {P} _ {h} ^ {k} - P _ {h}\right) g (s, a, b) \right| \leq \frac {c _ {1}}{H} \min \left\{P _ {h} \left(\bar {V} _ {h + 1} ^ {k} - \underline {{{V}}} _ {h + 1} ^ {k}\right) (s, a, b), \widetilde {P} _ {h} ^ {k} \left(\bar {V} _ {h + 1} ^ {k} - \underline {{{V}}} _ {h + 1} ^ {k}\right) (s, a, b) \right\} \tag {22} \\ + \frac {c _ {1} H ^ {2} S \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)} + \frac {c _ {1} H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}. \\ \end{array}
$$

Proof of Lemma B.6. If $|g|(s) \leq (\overline{V}_{h+1}^k - \underline{V}_{h+1}^k)(s)$ , it holds that:

$$
\begin{array}{l} \left| (\widetilde {P} _ {h} ^ {k} - P _ {h}) g (s, a, b) \right| \leq \sum_ {s ^ {\prime}} \left| \left(\widetilde {P} _ {h} ^ {k} - P _ {h}\right) (s ^ {\prime} | s, a, b) \right| \cdot | g | (s ^ {\prime}) \\ \leq \sum_ {s ^ {\prime}} \left| \left(\widetilde {P} _ {h} ^ {k} - P _ {h}\right) (s ^ {\prime} | s, a, b) \right| \cdot \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s ^ {\prime}) \\ \leq \sum_ {s ^ {\prime}} \left(2 \sqrt {\frac {P _ {h} (s ^ {\prime} | s , a , b) \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {2 E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}\right) \cdot \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s ^ {\prime}) \tag {23} \\ \leq \sum_ {s ^ {\prime}} \left(\frac {P _ {h} (s ^ {\prime} | s , a , b)}{H} + \frac {H \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)} + \frac {2 E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}\right) \cdot \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s ^ {\prime}) \\ \leq \frac {c _ {1}}{H} P _ {h} (\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}) (s, a, b) + \frac {c _ {1} H ^ {2} S \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)} + \frac {c _ {1} H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}, \\ \end{array}
$$

where the third inequality is because of Lemma B.2. The forth inequality results from AM-GM inequality. The last inequality holds for some universal constant $c_{1}$ .

The empirical part with the R.H.S to be $\widetilde{P}_h^k$ can be proven using identical proof according to (14).

Then we prove that the UCB and LCB functions are actually upper and lower bounds of the best responses. Recall that $\pi^k$ is the (correlated) policy executed in the $k$ -th episode and $(\mu^k, \nu^k)$ for both players are the marginal policies of $\pi^k$ . In other words, $\mu_h^k(\cdot | s) = \sum_{b \in \mathcal{B}} \pi_h^k(\cdot, b | s)$ and $\nu_h^k(\cdot | s) = \sum_{a \in \mathcal{A}} \pi_h^k(a, \cdot | s)$ for all $(h, s) \in [H] \times \mathcal{S}$ .

Lemma B.7. Suppose Assumption 3.1 and Lemma B.4 hold, then there exist universal constants $C_1, C_2 > 0$ (in Algorithm 1) such that for all $(h, s, a, b, k) \in [H] \times S \times A \times B \times [K]$ , it holds that:

$$
\left\{ \begin{array}{l} \overline {{Q}} _ {h} ^ {k} (s, a, b) \geq Q _ {h} ^ {\dagger , \nu^ {k}} (s, a, b) \geq Q _ {h} ^ {\mu^ {k}, \dagger} (s, a, b) \geq \underline {{Q}} _ {h} ^ {k} (s, a, b), \\ \overline {{V}} _ {h} ^ {k} (s) \geq V _ {h} ^ {\dagger , \nu^ {k}} (s) \geq V _ {h} ^ {\mu^ {k}, \dagger} (s) \geq \underline {{V}} _ {h} ^ {k} (s). \end{array} \right. \tag {24}
$$

Proof of Lemma B.7. We prove by backward induction. For each $k \in [K]$ , the conclusion is obvious for $h = H + 1$ . Suppose UCB and LCB hold for Q value functions in the $(h + 1)$ -th time step, we first prove the bounds for V functions in the $(h + 1)$ -th step and then prove the bounds for Q functions in the h-th step. For all $s \in S$ , it holds that

$$
\begin{array}{l} \overline {{V}} _ {h + 1} ^ {k} (s) = \mathbb {E} _ {\pi_ {h + 1} ^ {k}} \overline {{Q}} _ {h + 1} ^ {k} (s) \\ \geq \sup _ {\mu} \mathbb {E} _ {\mu , \nu_ {h + 1} ^ {k}} \overline {{{Q}}} _ {h + 1} ^ {k} (s) \tag {25} \\ \geq \sup _ {\mu} \mathbb {E} _ {\mu , \nu_ {h + 1} ^ {k}} Q _ {h + 1} ^ {\dagger , \nu^ {k}} (s) \\ = V _ {h + 1} ^ {\dagger , \nu^ {k}} (s). \\ \end{array}
$$

The conclusion $\underline{V}_{h + 1}^{k}(s)\leq V_{h + 1}^{\mu^{k},\dagger}(s)$ can be proven by symmetry. Therefore, it holds that

$$
\overline {{V}} _ {h + 1} ^ {k} (s) \geq V _ {h + 1} ^ {\dagger , \nu^ {k}} (s) \geq V _ {h + 1} ^ {\star} (s) \geq V _ {h + 1} ^ {\mu^ {k}, \dagger} (s) \geq \underline {{V}} _ {h + 1} ^ {k} (s). \tag {26}
$$

Next we prove the bounds for Q value functions at the h-th step. For all $(s, a, b)$ , it holds that

$$
\begin{array}{l} \left(\overline {{Q}} _ {h} ^ {k} - Q _ {h} ^ {\dagger , \nu^ {k}}\right) (s, a, b) \geq \min \left\{\left(\widetilde {P} _ {h} ^ {k} \overline {{V}} _ {h + 1} ^ {k} - P _ {h} V _ {h + 1} ^ {\dagger , \nu^ {k}} + \gamma_ {h} ^ {k} + \Gamma_ {h} ^ {k}\right) (s, a, b), 0 \right\} \\ \geq \min \left\{\left(\widetilde {P} _ {h} ^ {k} V _ {h + 1} ^ {\dagger , \nu^ {k}} - P _ {h} V _ {h + 1} ^ {\dagger , \nu^ {k}} + \gamma_ {h} ^ {k} + \Gamma_ {h} ^ {k}\right) (s, a, b), 0 \right\} \\ = \min \left\{\underbrace {\left(\widetilde {P} _ {h} ^ {k} - P _ {h}\right) \left(V _ {h + 1} ^ {\dagger , \nu^ {k}} - V _ {h + 1} ^ {\star}\right) (s , a , b)} _ {\text {(i)}} + \underbrace {\left(\widetilde {P} _ {h} ^ {k} - P _ {h}\right) V _ {h + 1} ^ {\star} (s , a , b)} _ {\text {(ii)}} + \gamma_ {h} ^ {k} (s, a, b) + \Gamma_ {h} ^ {k} (s, a, b), 0 \right\}. \tag {27} \\ \end{array}
$$

The absolute value of term (i) can be bounded as below.

$$
| (\mathrm{i}) | \leq \frac {c _ {1}}{H} \widetilde {P} _ {h} ^ {k} (\overline {{{V}}} _ {h + 1} ^ {k} - \underline {{{V}}} _ {h + 1} ^ {k}) (s, a, b) + \frac {c _ {1} H ^ {2} S \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)} + \frac {c _ {1} H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}, \tag {28}
$$

for some universal constant $c_{1}$ according to Lemma B.6.

The absolute value of term (ii) can be bounded as below.

$$
| (\mathrm{ii}) | \leq \sqrt {\frac {2 \operatorname{Var} _ {\widehat {P} _ {h} ^ {k} (\cdot | s , a , b)} V _ {h + 1} ^ {\star} (\cdot) \cdot \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {2 H S E _ {\epsilon , \beta \iota}}{\widetilde {N} _ {h} ^ {k} (s , a , b)} \leq \sqrt {\frac {2 \operatorname{Var} _ {\widetilde {P} _ {h} ^ {k} (\cdot | s , a , b)} V _ {h + 1} ^ {\star} (\cdot) \cdot \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {8 H S E _ {\epsilon , \beta \iota}}{\widetilde {N} _ {h} ^ {k} (s , a , b)}, \tag {29}
$$

where the first inequality is because of Lemma B.3 while the second inequality holds due to Lemma B.5.

We further bound the term $\mathrm{Var}_{\widetilde{P}_h^k (\cdot |s,a,b)}V_{h + 1}^\star (\cdot)$ as below.

$$
\begin{array}{l} \left| \widetilde {V} _ {h} ^ {k} \left(\frac {\overline {{V}} _ {h + 1} ^ {k} + \underline {{V}} _ {h + 1} ^ {k}}{2}\right) - \widetilde {V} _ {h} ^ {k} V _ {h + 1} ^ {\star} (\cdot) \right| (s, a, b) \\ \leq \left| \widetilde {P} _ {h} ^ {k} \cdot \left(\frac {\overline {{V}} _ {h + 1} ^ {k} + \underline {{V}} _ {h + 1} ^ {k}}{2}\right) ^ {2} - \widetilde {P} _ {h} ^ {k} \cdot \left(V _ {h + 1} ^ {\star}\right) ^ {2} \right| (s, a, b) + \left| \left[ \widetilde {P} _ {h} ^ {k} \cdot \left(\frac {\overline {{V}} _ {h + 1} ^ {k} + \underline {{V}} _ {h + 1} ^ {k}}{2}\right) (s, a, b) \right] ^ {2} - \left[ \widetilde {P} _ {h} ^ {k} V _ {h + 1} ^ {\star} (s, a, b) \right] ^ {2} \right| \\ \leq 4 H \widetilde {P} _ {h} ^ {k} \cdot \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s, a, b). \tag {30} \\ \end{array}
$$

Therefore, the term (ii) can be further bounded as below.

$$
\begin{array}{l} | (\mathrm{ii}) | \leq \sqrt {\frac {2 \operatorname{Var} _ {\widetilde {P} _ {h} ^ {k} (\cdot | s , a , b)} V _ {h + 1} ^ {\star} (\cdot) \cdot \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {8 H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)} \\ \leq \sqrt {\frac {2 \iota \cdot \widetilde {V} _ {h} ^ {k} \left(\frac {\overline {{{V}}} _ {h + 1} ^ {k} + \underline {{{V}}} _ {h + 1} ^ {k}}{2}\right) (s , a , b) + 2 \iota \cdot 4 H \widetilde {P} _ {h} ^ {k} \cdot \left(\overline {{{V}}} _ {h + 1} ^ {k} - \underline {{{V}}} _ {h + 1} ^ {k}\right) (s , a , b)}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {8 H S E _ {\epsilon , \beta^ {\iota}}}{\widetilde {N} _ {h} ^ {k} (s , a , b)} \tag {31} \\ \leq \sqrt {\frac {2 \widetilde {V} _ {h} ^ {k} \left(\frac {\overline {{V}} _ {h + 1} ^ {k} + \underline {{V}} _ {h + 1} ^ {k}}{2}\right) (s , a , b) \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {\widetilde {P} _ {h} ^ {k} \cdot \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s , a , b)}{H} + \frac {2 H ^ {2} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)} + \frac {8 H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}, \\ \end{array}
$$

where the second inequality results from (30) and the third inequality is due to AM-GM inequality. Combining the upper bounds of $|(\mathrm{i})|$ and $|(\mathrm{ii})|$ , there exist universal constants $C_{1}, C_{2} > 0$ such that

$$
(\mathrm{i}) + (\mathrm{ii}) + \gamma_ {h} ^ {k} (s, a, b) + \Gamma_ {h} ^ {k} (s, a, b) \geq 0. \tag {32}
$$

The inequality implies that $\left(\overline{Q}_h^k - Q_h^{\dagger, \nu^k}\right)(s, a, b) \geq 0$ . By symmetry, we have $\left(\underline{Q}_h^k - Q_h^{\mu^k, \dagger}\right)(s, a, b) \leq 0$ . As a result, it holds that $\overline{Q}_h^k(s, a, b) \geq Q_h^{\dagger, \nu^k}(s, a, b) \geq Q_h^\star(s, a, b) \geq Q_h^{\mu^k, \dagger}(s, a, b) \geq \underline{Q}_h^k(s, a, b)$ .

According to backward induction, the conclusion holds for all $(h,s,a,b,k)$ .

![](images/d6a3ea4be28fe9679b53b83fda06b218a3ae0baf2ee9a01fca341bbfff08b8f8.jpg)

# B.3 Proof of Theorem 4.1

Given the UCB and LCB property, we are now ready to prove our main results. We first state the following lemma that controls the error of the empirical variance estimator.

Lemma B.8. Suppose Assumption 3.1 and Lemma B.4 hold, then there exists a universal constant $c_{2} > 0$ such that for all $(h, s, a, b, k) \in [H] \times \mathcal{S} \times \mathcal{A} \times \mathcal{B} \times [K]$ , it holds that

$$
\left| \widetilde {V} _ {h} ^ {k} \left(\frac {\overline {{V}} _ {h + 1} ^ {k} + \underline {{V}} _ {h + 1} ^ {k}}{2}\right) - V _ {h} V _ {h + 1} ^ {\pi^ {k}} \right| (s, a, b) \tag {33}
$$

$$
\leq 4 H P _ {h} \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s, a, b) + \frac {c _ {2} H ^ {2} S E _ {\epsilon , \beta}}{\widetilde {N} _ {h} ^ {k} (s , a , b)} + c _ {2} H ^ {2} \sqrt {\frac {S \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}}.
$$

Proof of Lemma B.8. According to Lemma B.7, $\overline{V}_{h}^{k}(s) \geq V_{h}^{\pi^{k}}(s) \geq \underline{V}_{h}^{k}(s)$ always holds. Then it holds that

$$
\begin{array}{l} \left| \widetilde {V} _ {h} ^ {k} \left(\frac {\overline {{V}} _ {h + 1} ^ {k} + \underline {{V}} _ {h + 1} ^ {k}}{2}\right) - V _ {h} V _ {h + 1} ^ {\pi^ {k}} \right| (s, a, b) \\ \leq \left| \widetilde {P} _ {h} ^ {k} \left(\frac {\overline {{V}} _ {h + 1} ^ {k} + \underline {{V}} _ {h + 1} ^ {k}}{2}\right) ^ {2} - P _ {h} \left(V _ {h + 1} ^ {\pi^ {k}}\right) ^ {2} - \left[ \widetilde {P} _ {h} ^ {k} \left(\frac {\overline {{V}} _ {h + 1} ^ {k} + \underline {{V}} _ {h + 1} ^ {k}}{2}\right) \right] ^ {2} + \left(P _ {h} V _ {h + 1} ^ {\pi^ {k}}\right) ^ {2} \right| (s, a, b) \\ \leq \left| \widetilde {P} _ {h} ^ {k} \left(\overline {{V}} _ {h + 1} ^ {k}\right) ^ {2} - P _ {h} \left(\underline {{V}} _ {h + 1} ^ {k}\right) ^ {2} - \left(\widetilde {P} _ {h} ^ {k} \underline {{V}} _ {h + 1} ^ {k}\right) ^ {2} + \left(P _ {h} \overline {{V}} _ {h + 1} ^ {k}\right) ^ {2} \right| (s, a, b) \tag {34} \\ \leq \underbrace {\left| \left(\widetilde {P} _ {h} ^ {k} - P _ {h}\right) \left(\overline {{V}} _ {h + 1} ^ {k}\right) ^ {2} \right| (s , a , b)} _ {\text {(i)}} + \underbrace {\left| P _ {h} \left[ \left(\overline {{V}} _ {h + 1} ^ {k}\right) ^ {2} - \left(\underline {{V}} _ {h + 1} ^ {k}\right) ^ {2} \right] \right| (s , a , b)} _ {\text {(ii)}} \\ + \underbrace {\left| \left(\widetilde {P} _ {h} ^ {k} \underline {{V}} _ {h + 1} ^ {k}\right) ^ {2} - \left(P _ {h} \underline {{V}} _ {h + 1} ^ {k}\right) ^ {2} \right| (s , a , b)} _ {\text {(iii)}} + \underbrace {\left| \left(P _ {h} \underline {{V}} _ {h + 1} ^ {k}\right) ^ {2} - \left(P _ {h} \overline {{V}} _ {h + 1} ^ {k}\right) ^ {2} \right| (s , a , b)} _ {\text {(iv)}}. \\ \end{array}
$$

The term (i) can be bounded as below due to Lemma B.1.

$$
\text {(i)} \leq 2 H ^ {2} \sqrt {\frac {S \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {2 H ^ {2} S E _ {\epsilon , \beta}}{\widetilde {N} _ {h} ^ {k} (s , a , b)}. \tag {35}
$$

The term (ii) can be directly bounded as below.

$$
\text {(ii)} \leq 2 H P _ {h} \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s, a, b). \tag {36}
$$

The term (iii) can be bounded as below due to Lemma B.1.

$$
\text {(iii)} \leq 2 H \left| \left(\widetilde {P} _ {h} ^ {k} - P _ {h}\right) \underline {{V}} _ {h + 1} ^ {k} \right| (s, a, b) \leq 4 H ^ {2} \sqrt {\frac {S \iota}{\widetilde {N} _ {h} ^ {k} (s , a , b)}} + \frac {4 H ^ {2} S E _ {\epsilon , \beta}}{\widetilde {N} _ {h} ^ {k} (s , a , b)}. \tag {37}
$$

The term (iv) can be directly bounded as below.

$$
\left(\mathrm{iv}\right) \leq 2 H P _ {h} \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s, a, b). \tag {38}
$$

The conclusion holds according the upper bounds of term (i), (ii), (iii) and (iv).

![](images/b7e8ea68e43ce4166b208e65694e7cee1b86b01ac17b181db851960ceca2d5de.jpg)

Finally we prove the regret bound of Algorithm 1.

Proof of Theorem 4.1. Our proof base on Assumption 3.1 and Lemma B.4. We define the following notations.

$$
\left\{ \begin{array}{l} \Delta_ {h} ^ {k} = \left(\overline {{V}} _ {h} ^ {k} - \underline {{V}} _ {h} ^ {k}\right) (s _ {h} ^ {k}), \\ \zeta_ {h} ^ {k} = \Delta_ {h} ^ {k} - \left(\overline {{Q}} _ {h} ^ {k} - \underline {{Q}} _ {h} ^ {k}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}), \\ \xi_ {h} ^ {k} = P _ {h} \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}) - \Delta_ {h + 1} ^ {k}. \end{array} \right. \tag {39}
$$

Then it holds that $\zeta_h^k$ and $\xi_h^k$ are martingale differences bounded by $H$ . In addition, we use the following abbreviations for notational simplicity.

$$
\left\{ \begin{array}{l} \gamma_ {h} ^ {k} = \gamma_ {h} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}), \\ \Gamma_ {h} ^ {k} = \Gamma_ {h} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}), \\ N _ {h} ^ {k} = N _ {h} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}), \\ \widetilde {N} _ {h} ^ {k} = \widetilde {N} _ {h} ^ {k} (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}). \end{array} \right. \tag {40}
$$

Then we have the following analysis about $\Delta_{h}^{k}$ .

$$
\begin{array}{l} \Delta_ {h} ^ {k} = \zeta_ {h} ^ {k} + \left(\overline {{Q}} _ {h} ^ {k} - \underline {{Q}} _ {h} ^ {k}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}) \\ \leq \zeta_ {h} ^ {k} + 2 \gamma_ {h} ^ {k} + 2 \Gamma_ {h} ^ {k} + \widetilde {P} _ {h} ^ {k} \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}) \\ \leq \zeta_ {h} ^ {k} + 2 \Gamma_ {h} ^ {k} + \left(1 + \frac {2 C _ {1}}{H}\right) \cdot \left[ \left(1 + \frac {c _ {1}}{H}\right) \cdot P _ {h} \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}) + \frac {c _ {1} H ^ {2} S \iota}{\widetilde {N} _ {h} ^ {k}} + \frac {c _ {1} H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k}} \right] \\ \leq \zeta_ {h} ^ {k} + \left(1 + \frac {c _ {3}}{H}\right) \cdot P _ {h} \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s _ {h} ^ {k}, a _ {h} ^ {k}, b _ {h} ^ {k}) + \frac {c _ {3} H ^ {2} S \iota}{\widetilde {N} _ {h} ^ {k}} + \frac {c _ {3} H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k}} \\ + c _ {3} \underbrace {\sqrt {\frac {\widetilde {V} _ {h} ^ {k} \left(\frac {\overline {{V}} _ {h + 1} ^ {k} + \underline {{V}} _ {h + 1} ^ {k}}{2}\right) (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k}) \iota}{\widetilde {N} _ {h} ^ {k}}}}, \tag {41} \\ \end{array}
$$

where the first inequality holds because of the definition of $\overline{Q}$ and $Q$ . The second inequality holds due to the definition of $\gamma_h^k$ and Lemma B.6. The last inequality holds for some universal constant $c_3 > 0$ .

The term (i) can be further bounded as below according to Lemma B.8 and AM-GM inequality.

$$
\begin{array}{l} \text {(i)} \leq \sqrt {\frac {V _ {h} V _ {h + 1} ^ {\pi^ {k}} (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k}) \iota}{\widetilde {N} _ {h} ^ {k}}} + \sqrt {\frac {4 H P _ {h} \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k}) \iota}{\widetilde {N} _ {h} ^ {k}}} + \frac {H \sqrt {c _ {2} S E _ {\epsilon , \beta} \iota}}{\widetilde {N} _ {h} ^ {k}} + c _ {2} \sqrt {\frac {\iota}{\widetilde {N} _ {h} ^ {k}}} + \frac {H ^ {2} \iota \sqrt {c _ {2} S}}{\widetilde {N} _ {h} ^ {k}} \\ \leq \sqrt {\frac {V _ {h} V _ {h + 1} ^ {\pi^ {k}} (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k}) \iota}{\widetilde {N} _ {h} ^ {k}}} + \frac {c _ {4} P _ {h} \left(\overline {{V}} _ {h + 1} ^ {k} - \underline {{V}} _ {h + 1} ^ {k}\right) (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k})}{H} + \frac {c _ {4} H ^ {2} \sqrt {S} \iota}{\widetilde {N} _ {h} ^ {k}} + \frac {c _ {4} H \sqrt {S E _ {\epsilon , \beta} \iota}}{\widetilde {N} _ {h} ^ {k}} + c _ {4} \sqrt {\frac {\iota}{\widetilde {N} _ {h} ^ {k}}}, \tag {42} \\ \end{array}
$$

where the first inequality results from Lemma B.8 and AM-GM inequality on the last term of (33). The second inequality holds for some universal constant $c_{4} > 0$ according to AM-GM inequality.

Plugging in the upper bound of term (i), for some universal constant $c_{5} > 0$ , it holds that:

$$
\Delta_ {h} ^ {k} \leq \zeta_ {h} ^ {k} + \left(1 + \frac {c _ {5}}{H}\right) \xi_ {h} ^ {k} + \left(1 + \frac {c _ {5}}{H}\right) \Delta_ {h + 1} ^ {k} + c _ {5} \sqrt {\frac {V _ {h} V _ {h + 1} ^ {\pi^ {k}} (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k}) \iota}{\widetilde {N} _ {h} ^ {k}}} + c _ {5} \sqrt {\frac {\iota}{\widetilde {N} _ {h} ^ {k}}} + \frac {c _ {5} H ^ {2} S \iota}{\widetilde {N} _ {h} ^ {k}} + \frac {c _ {5} H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k}}. \tag {43}
$$

Summing $\Delta_1^k$ over $k\in [K]$ , we have for some universal constant $c_{6} > 0$ , it holds that:

$$
\sum_ {k = 1} ^ {K} \Delta_ {1} ^ {k} \leq \underbrace {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \left(1 + \frac {c _ {5}}{H}\right) ^ {h - 1} \zeta_ {h} ^ {k}} _ {\text {(ii)}} + \underbrace {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \left(1 + \frac {c _ {5}}{H}\right) ^ {h} \xi_ {h} ^ {k}} _ {\text {(iii)}} + c _ {6} \underbrace {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \sqrt {\frac {V _ {h} V _ {h + 1} ^ {\pi^ {k}} (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k}) \iota}{\widetilde {N} _ {h} ^ {k}}}} _ {\text {(iv)}} \tag {44}
$$

$$
+ c _ {6} \underbrace {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \sqrt {\frac {\iota}{\widetilde {N} _ {h} ^ {k}}}} _ {\text {(v)}} + c _ {6} \underbrace {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {H ^ {2} S \iota + H S E _ {\epsilon , \beta} \iota}{\widetilde {N} _ {h} ^ {k}}} _ {\text {(vi)}}.
$$

The term (ii) and term (iii) can be bounded by Azuma-Hoeffding inequality. With probability $1 - \frac{2\beta}{9}$ , it holds that

$$
| (\mathrm{ii}) | \leq O \left(\sqrt {H ^ {3} K \iota}\right), \quad | (\mathrm{iii}) | \leq O \left(\sqrt {H ^ {3} K \iota}\right). \tag {45}
$$

The main term (iv) is bounded as below.

$$
\begin{array}{l} \text {(iv)} \leq \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \sqrt {\frac {V _ {h} V _ {h + 1} ^ {\pi^ {k}} (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k}) \iota}{N _ {h} ^ {k}}} \\ \leq \sqrt {\sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} V _ {h} V _ {h + 1} ^ {\pi^ {k}} (s _ {h} ^ {k} , a _ {h} ^ {k} , b _ {h} ^ {k}) \iota \cdot \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {1}{N _ {h} ^ {k}}} \tag {46} \\ \leq \sqrt {O (H ^ {2} K + H ^ {3} \iota) \iota \cdot O (H S A B \iota)} \\ = \widetilde {O} \left(\sqrt {H ^ {3} S A B K} + H ^ {2} \sqrt {S A B}\right). \\ \end{array}
$$

The first inequality is because $\widetilde{N}_{h}^{k} \geq N_{h}^{k}$ (Assumption 3.1). The second inequality holds due to Cauchy-Schwarz inequality. The third inequality holds with probability $1 - \frac{\beta}{9}$ because of Law of total variance and standard concentration inequalities (for details please refer to Lemma 8 of Azar et al. [2017]).

The term (v) is bounded as below due to pigeon-hole principle.

$$
(\mathrm{v}) \leq \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \sqrt {\frac {\iota}{N _ {h} ^ {k}}} \leq O (\sqrt {H ^ {2} S A B K \iota}), \tag {47}
$$

where the first inequality is because $\widetilde{N}_{h}^{k} \geq N_{h}^{k}$ (Assumption 3.1). The last one results from pigeon-hole principle.

The term (vi) can be bounded as below.

$$
(\mathrm{vi}) \leq \sum_ {k = 1} ^ {K} \sum_ {h = 1} ^ {H} \frac {H ^ {2} S \iota + H S E _ {\epsilon , \beta} \iota}{N _ {h} ^ {k}} \leq O (H ^ {3} S ^ {2} A B \iota^ {2}) + O (H ^ {2} S ^ {2} A B E _ {\epsilon , \beta} \iota^ {2}). \tag {48}
$$

Combining the upper bounds for term $|(\mathrm{ii})|$ , $|(\mathrm{iii})|$ , (iv), (v) and (vi). The regret of Algorithm 1 can be bounded as below.

$$
\operatorname{Regret} (K) = \sum_ {k = 1} ^ {K} \left[ V _ {1} ^ {\dagger , \nu^ {k}} \left(s _ {1}\right) - V _ {1} ^ {\mu^ {k}, \dagger} \left(s _ {1}\right) \right] \leq \sum_ {k = 1} ^ {K} \left[ \bar {V} _ {1} ^ {k} \left(s _ {1}\right) - \underline {{{V}}} _ {1} ^ {k} \left(s _ {1}\right) \right] \tag {49}
$$

$$
= \sum_ {k = 1} ^ {K} \Delta_ {1} ^ {k} \leq \widetilde {O} \left(\sqrt {H ^ {2} S A B T} + H ^ {3} S ^ {2} A B + H ^ {2} S ^ {2} A B E _ {\epsilon , \beta}\right),
$$

where T = HK is the number of steps.

The failure probability is bounded by $\beta$ ( $\frac{\beta}{3}$ for Assumption 3.1, $\frac{\beta}{3}$ for Lemma B.4, $\frac{\beta}{3}$ for terms (ii), (iii) and (iv)). The proof of Theorem 4.1 is complete.

# B.4 Proof of Theorem 4.2

In this part, we provide a proof of the PAC guarantee: Theorem 4.2. The proof directly follows from the proof of the regret bound (Theorem 4.1).

Proof of Theorem 4.2. Recall that we choose $\pi^{\mathrm{out}} = \pi^{\overline{k}}$ such that $\overline{k} = \operatorname{argmin}_k\left(\overline{V}_1^k -\underline{V}_1^k\right)(s_1)$ . Therefore, we have

$$
V _ {1} ^ {\dagger , \nu^ {\mathrm{out}}} (s _ {1}) - V _ {1} ^ {\mu^ {\mathrm{out}}, \dagger} (s _ {1}) \leq \overline {{V _ {1} ^ {\overline {{k}}}}} (s _ {1}) - \underline {{V _ {1} ^ {\overline {{k}}}}} (s _ {1}) \leq \frac {1}{K} \widetilde {O} \left(\sqrt {H ^ {3} S A B K} + H ^ {2} S ^ {2} A B E _ {\epsilon , \beta}\right), \tag {50}
$$

if ignoring the lower order term of the regret bound.

Therefore, choosing $K \geq \widetilde{\Omega} \left( \frac{H^3 SAB}{\alpha^2} + \min \left\{ K' | \frac{H^2 S^2 ABE_{\epsilon,\beta}}{K'} \leq \alpha \right\} \right)$ bounds the R.H.S by $\alpha$ .

# C Missing proof in Section 5

In this section, we provide the missing proof for results in Section 5. Recall that $N_{h}^{k}$ is the real visitation count, $\widehat{N}_{h}^{k}$ is the intermediate noisy count calculated by both Privatizers and $\widetilde{N}_{h}^{k}$ is the final private count after the post-processing step. Note that most of the proof here are generalizations of Appendix D in Qiao and Wang [2023c] to the multi-player setting, and here we state the proof for completeness.

Proof of Lemma 5.1. Due to Theorem 3.5 of Chan et al. [2011] and Lemma 34 of Hsu et al. [2014], the release of $\{\widehat{N}_h^k (s,a,b)\}_{(h,s,a,b,k)}$ satisfies $\frac{\epsilon}{2}$ -DP. Similarly, the release of $\{\widehat{N}_h^k (s,a,b,s')\}_{(h,s,a,b,s',k)}$ also satisfies $\frac{\epsilon}{2}$ -DP. Therefore, the release of the following private counters $\{\widehat{N}_h^k (s,a,b)\}_{(h,s,a,b,k)},$ $\{\widehat{N}_h^k (s,a,b,s')\}_{(h,s,a,b,s',k)}$ satisfy $\epsilon$ -DP. Due to post-processing (Lemma 2.3 of Bun and Steinke [2016]), the release of both private counts $\{\widetilde{N}_h^k (s,a,b)\}_{(h,s,a,b,k)}$ and $\{\widetilde{N}_h^k (s,a,b,s')\}_{(h,s,a,b,s',k)}$ also satisfies $\epsilon$ -DP. Then it holds that the release of all $\pi^k$ is $\epsilon$ -DP according to post-processing. Finally, the guarantee of $\epsilon$ -JDP results from Billboard Lemma (Lemma 9 of Hsu et al. [2014]).

For utility analysis, because of Theorem 3.6 of Chan et al. [2011], our choice $\epsilon' = \frac{\epsilon}{2H\log K}$ in Binary Mechanism and a union bound, with probability $1 - \frac{\beta}{3}$ , for all $(h,s,a,b,s',k)$ ,

$$
\left| \widehat {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) - N _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right| \leq O \left(\frac {H}{\epsilon} \log (H S A B K / \beta) ^ {2}\right), \tag {51}
$$

$$
\left| \widehat {N} _ {h} ^ {k} (s, a, b) - N _ {h} ^ {k} (s, a, b) \right| \leq O \left(\frac {H}{\epsilon} \log (H S A B K / \beta) ^ {2}\right).
$$

Together with Lemma 5.8, the Central Privatizer satisfies Assumption 3.1 with $E_{\epsilon,\beta} = \widetilde{O}\left(\frac{H}{\epsilon}\right)$ .

Proof of Theorem 5.2. The proof directly results from plugging $E_{\epsilon, \beta} = \widetilde{O}\left(\frac{H}{\epsilon}\right)$ into Theorem 4.1 and Theorem 4.2.

Proof of Theorem 5.3. The first term results from the non-private regret lower bound $\Omega (\sqrt{H^2S(A + B)T})$ [Bai and Jin, 2020]. The second term is a direct adaptation of the $\Omega (HSA / \epsilon)$ lower bound for any algorithms with $\epsilon$ -JDP guarantee under single-agent MDP [Vietri et al., 2020].

Proof of Lemma 5.4. The privacy guarantee directly results from properties of Laplace Mechanism and composition of DP [Dwork et al., 2014].

For utility analysis, because of Corollary 12.4 of Dwork et al. [2014] and a union bound, with probability $1 - \frac{\beta}{3}$ , for all possible $(h,s,a,b,s',k)$ ,

$$
\left| \widehat {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) - N _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right| \leq O \left(\frac {H}{\epsilon} \sqrt {K \log (H S A B K / \beta)}\right), \tag {52}
$$

$$
\left| \widehat {N} _ {h} ^ {k} (s, a, b) - N _ {h} ^ {k} (s, a, b) \right| \leq O \left(\frac {H}{\epsilon} \sqrt {K \log (H S A B K / \beta)}\right).
$$

Together with Lemma 5.8, the Local Privatizer satisfies Assumption 3.1 with $E_{\epsilon, \beta} = \widetilde{O}\left(\frac{H}{\epsilon}\sqrt{K}\right)$ .

Proof of Theorem 5.5. The proof directly results from plugging $E_{\epsilon, \beta} = \widetilde{O}\left(\frac{H}{\epsilon}\sqrt{K}\right)$ into Theorem 4.1 and Theorem 4.2.

Proof of Theorem 5.6. The first term results from the non-private regret lower bound $\Omega (\sqrt{H^2S(A + B)T})$ [Bai and Jin, 2020]. The second term is a direct adaptation of the $\Omega (\sqrt{HSAT} /\epsilon)$ lower bound for any algorithms with $\epsilon$ -LDP guarantee under single-agent MDP [Garcelon et al., 2021].

Proof of Lemma 5.8. For clarity, we denote the solution of (8) by $\bar{N}_h^k$ and therefore $\widetilde{N}_h^k(s,a,b,s') = \bar{N}_h^k(s,a,b,s') + \frac{E_{\epsilon,\beta}}{2S}$ , $\widetilde{N}_h^k(s,a,b) = \bar{N}_h^k(s,a,b) + \frac{E_{\epsilon,\beta}}{2}$ .

When the condition (two inequalities) in Lemma 5.8 holds, the original counts $\{N_h^k (s,a,b,s')\}_{s'\in \mathcal{S}}$ is a feasible solution to the optimization problem, which means that

$$
\max _ {s ^ {\prime}} \left| \bar {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) - \widehat {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right| \leq \max _ {s ^ {\prime}} \left| N _ {h} ^ {k} (s, a, b, s ^ {\prime}) - \widehat {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right| \leq \frac {E _ {\epsilon , \beta}}{4}.
$$

Combining with the condition in Lemma 5.8 with respect to $\widehat{N}_h^k (s,a,b,s')$ , it holds that

$$
\left| \bar {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) - N _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right| \leq \left| \bar {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) - \widehat {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right| + \left| \widehat {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) - N _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right| \leq \frac {E _ {\epsilon , \beta}}{2}.
$$

Since $\widetilde{N}_h^k (s,a,b,s') = \bar{N}_h^k (s,a,b,s') + \frac{E_{\epsilon,\beta}}{2S}$ and $\bar{N}_h^k (s,a,b,s')\geq 0$ , we have

$$
\widetilde {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) > 0, \quad \left| \widetilde {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}) - N _ {h} ^ {k} (s, a, b, s ^ {\prime}) \right| \leq E _ {\epsilon , \beta}. \tag {53}
$$

For $\bar{N}_{h}^{k}(s,a,b)$ , according to the constraints in the optimization problem (8), it holds that

$$
\left| \bar {N} _ {h} ^ {k} (s, a, b) - \widehat {N} _ {h} ^ {k} (s, a, b) \right| \leq \frac {E _ {\epsilon , \beta}}{4}.
$$

Combining with the condition in Lemma 5.8 with respect to $\widehat{N}_{h}^{k}(s,a,b)$ , it holds that

$$
\left| \bar {N} _ {h} ^ {k} (s, a, b) - N _ {h} ^ {k} (s, a, b) \right| \leq \left| \bar {N} _ {h} ^ {k} (s, a, b) - \widehat {N} _ {h} ^ {k} (s, a, b) \right| + \left| \widehat {N} _ {h} ^ {k} (s, a, b) - N _ {h} ^ {k} (s, a, b) \right| \leq \frac {E _ {\epsilon , \beta}}{2}.
$$

Since $\widetilde{N}_h^k (s,a,b) = \bar{N}_h^k (s,a,b) + \frac{E_{\epsilon,\beta}}{2}$ , we have

$$
N _ {h} ^ {k} (s, a, b) \leq \widetilde {N} _ {h} ^ {k} (s, a, b) \leq N _ {h} ^ {k} (s, a, b) + E _ {\epsilon , \beta}. \tag {54}
$$

According to the last line of the optimization problem (8), we have $\bar{N}_{h}^{k}(s,a,b)=\sum_{s^{\prime}\in\mathcal{S}}\bar{N}_{h}^{k}(s,a,b,s^{\prime})$ and therefore,

$$
\widetilde {N} _ {h} ^ {k} (s, a, b) = \sum_ {s ^ {\prime} \in \mathcal {S}} \widetilde {N} _ {h} ^ {k} (s, a, b, s ^ {\prime}). \tag {55}
$$

The proof is complete by combining (53), (54) and (55).

![](images/5f9994d20b93daade567715c5b190fd06a5d72e81810e2087f3564a22653a97e.jpg)