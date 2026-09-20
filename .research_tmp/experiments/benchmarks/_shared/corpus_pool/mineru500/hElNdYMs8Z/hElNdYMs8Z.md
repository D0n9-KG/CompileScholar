# A Finite-Sample Analysis of Payoff-Based Independent Learning in Zero-Sum Stochastic Games

Zaiwei Chen $^{1,*}$ , Kaiqing Zhang $^{2}$ , Eric Mazumdar $^{1,\dagger}$ , Asuman Ozdaglar $^{3}$ , and Adam Wierman $^{1,\ddagger}$

$^{1}$ Caltech $*$ zchen458@caltech.edu, $\dagger$ mazumdar@caltech.edu, $\ddagger$ adamw@caltech.edu

$^{2}$ University of Maryland, College Park kaiqing@umd.edu $^{3}$ MIT asuman@mit.edu

# Abstract

We study two-player zero-sum stochastic games, and propose a form of independent learning dynamics called Doubly Smoothed Best-Response dynamics, which integrates a discrete and doubly smoothed variant of the best-response dynamics into temporal-difference (TD)-learning and minimax value iteration. The resulting dynamics are payoff-based, convergent, rational, and symmetric among players. Our main results provide finite-sample guarantees. In particular, we prove the first-known $\tilde{\mathcal{O}}(1/\epsilon^{2})$ sample complexity bound for payoff-based independent learning dynamics, up to a smoothing bias. In the special case where the stochastic game has only one state (i.e., matrix games), we provide a sharper $\tilde{\mathcal{O}}(1/\epsilon)$ sample complexity. Our analysis uses a novel coupled Lyapunov drift approach to capture the evolution of multiple sets of coupled and stochastic iterates, which might be of independent interest.

# 1 Introduction

Recent years have seen remarkable successes of reinforcement learning (RL) in a variety of applications, such as board games (Silver et al., 2017), autonomous driving (Shalev-Shwartz et al., 2016), city navigation (Mirowski et al., 2018), and fusion plasma control (Degrave et al., 2022). A common feature of these applications is that there are multiple decision-makers interacting with each other in an unknown environment. While empirical successes have shown the potential of multi-agent reinforcement learning (MARL) (Busoniu et al., 2008; Zhang et al., 2021a), the training of MARL agents largely relies on heuristics and parameter-tuning, and is not always reliable. In particular, many practical MARL algorithms are heuristically extended from their single-agent counterparts and lack theoretical guarantees.

A growing literature seeks to provide theoretical insights to substantiate the empirical success of MARL and inform the design of efficient, and provably convergent algorithms. Work along these lines can be broadly categorized into work on cooperative MARL such as Arslan and Yüksel (2017); Zhang et al. (2018); Qu et al. (2020); Zhang et al. (2022c) where agents seek to reach a common goal, and work on competitive MARL where agents have individual (and possibly misaligned) objectives (Littman, 1994, 2001; Hu and Wellman, 2003; Daskalakis et al., 2020; Sayin et al., 2021; Bai and Jin, 2020; Xie et al., 2020; Zhang et al., 2021c; Ding et al., 2022; Liu et al., 2021; Jin et al., 2021; Daskalakis et al., 2022). While some earlier work focused on providing guarantees on asymptotic convergence, the more recent ones share an increasing interest in understanding the finite-time/sample behavior. This follows from the line of recent successes in understanding the finite-sample behavior of single-agent RL algorithms, see e.g., Bhandari et al. (2018); Srikant and Ying (2019); Li et al. (2020); Chen et al. (2020) and many others.

In this paper, we focus on the benchmark-setting of two-player $^{1}$ zero-sum matrix and stochastic games, and develop multi-agent learning dynamics with provable finite-sample guarantees. Crucially, our dynamics are independent (requiring no coordination between the agents in learning), rational (each agent will converge to the best response to the opponent if the opponent plays

an (asymptotically) stationary policy (Bowling and Veloso, 2001)), and hence capture the learning in settings with multiple game-theoretic agents. Indeed, game-theoretic agents are self-interested, and ideal learning dynamics should not enforce any communication of information or coordination among agents. In addition, we focus on the more challenging but practically relevant settings of payoff-based learning, where each agent can only observe the realized payoff of itself during learning, without observing the policy or even the action taken by the opponent. For these learning dynamics, we establish for the first time finite-sample guarantees for both two-player zero-sum matrix and stochastic games. We detail our contributions as follows.

# 1.1 Contributions

We take a principled approach to algorithm design: we first construct independent learning dynamics for the special case of zero-sum matrix games. Then, we generalize the dynamics to the setting of Markov games and present the finite-sample guarantees. Further, in both cases, our dynamics are easily implementable and have a simple, intuitive structure that links them to well-known dynamics from the learning in games literature.

# 1.1.1 Independent Learning for Two-Player Zero-Sum Matrix Games

Algorithm Design. We design a new learning dynamics called Doubly Smoothed Best-Response (DSBR) dynamics for solving matrix games. It maintains two sets of iterates on a single time scale: the policies and the local state-action functions (denoted as the q-functions). The policy update can be viewed as a variant of the best-response dynamics, where the best response is constructed by introducing the q-function as an estimate of the payoff marginalized by the opponent's current policy. The name of doubly smoothed follows from two key algorithmic ideas that enable our finite-sample analysis: (1) we introduce a stepsize to smooth the update of the policy, so that it does not change too abruptly during learning; (2) we use a smoothed best-response to the local q-function when updating the policy. Idea (1) is an alternative to independent learning dynamics for matrix games (Leslie and Collins, 2005) that use adaptive stepsizes, enabling us to smooth the variation of the local q-function (and thus policy) in a more controlled way (and thus to establish finite-sample guarantees). Idea (2) has been exploited in the well-known dynamics of smoothed fictitious play (Fudenberg and Kreps, 1993), in order to encourage exploration and make the learning dynamics consistent (Fudenberg and Levine, 1995).

Finite-Sample Analysis. We establish a finite-sample bound (measured in terms of the Nash gap) for our DSBR dynamics when using stepsizes of various decay rates. The best convergence rate is achieved with a stepsize of $\mathcal{O}(1 / k)$ , in which case the learning dynamics enjoys an overall $\mathcal{O}(1 / k)$ rate of convergence to a Nash equilibrium up to a smoothing bias. The smoothing bias arises from the use of softmax policies in the update.

# 1.1.2 Independent Learning for Two-Player Zero-Sum Markov Games

Algorithm Design. Building on the results for matrix games, we design a new learning dynamics for Markov games called Doubly Smoothed Best-Response dynamics with Value Iteration (DSBR-VI), driven by a single trajectory of Markovian samples. The dynamics consist of two loops, and can be viewed as a combination of the DSBR dynamics for an induced auxiliary matrix game (conducted in the inner loop) and an independent way of performing minimax value iteration (conducted in the outer loop). In particular, in the inner loop, the iterate of the outer loop, i.e., the value function, is fixed, and the players learn the approximate Nash equilibrium of an auxiliary matrix game induced by the value function; then the outer loop is updated by approximating the minimax value iteration updates for Markov games, with only local information.

Finite-Sample Analysis. We then establish finite-sample bounds for our DSBR-VI dynamics when using either constant stepsize or diminishing stepsizes, and under weaker assumptions

compared to existing work. Our dynamics achieves an overall $\tilde{\mathcal{O}}(1/\epsilon^{2})$ sample complexity up to a smoothing bias. To the best of our knowledge, this is the first finite-sample analysis of best-response type independent learning dynamics that are convergent and rational for Markov games. Most existing MARL algorithms are either symmetric across players but not payoff-based, e.g., Cen et al. (2021, 2022); Zhang et al. (2022a); Zeng et al. (2022); Erez et al. (2022), or not symmetric and thus not rational, e.g., Daskalakis et al. (2020); Zhao et al. (2021); Zhang et al. (2021b); Alacaoglu et al. (2022), or do not have finite-sample guarantees, e.g., Leslie et al. (2020); Sayin et al. (2021); Baudin and Laraki (2022b).

# 1.2 Challenges & Our Techniques

At a high level, we develop a novel coupled Lyapunov drift argument to establish the finite-sample bounds. Specifically, we design a Lyapunov function for each set of the iterates (i.e., value functions, policies, and q-functions) and establish coupled Lyapunov drift inequalities for each. We then carefully combine the coupled Lyapunov drift inequalities to establish the finite-sample bounds. While a more detailed analysis is provided in Section 4, we briefly give an overview of the main challenges encountered in analyzing the payoff-based independent learning dynamics in Markov games — as well as how we overcome them in our analysis and algorithm design.

Time-Inhomogeneous Markovian Noise. The fact that our algorithm is payoff-based imposes additional challenges in handling the stochastic errors in the update. In particular, due to the best-response nature of the dynamics, the behavior policy for sampling becomes time-varying. In fact, the samples used for learning form a time-inhomogeneous Markov chain. This makes it challenging to establish finite-sample guarantees, as time-inhomogeneity prevents us from directly exploiting the uniqueness of stationary distributions and the fast mixing of Markov chains. Building on existing work Bhandari et al. (2018); Srikant and Ying (2019), we overcome this challenge by tuning the algorithm design and developing a refined conditioning argument.

Possible Non-Smoothness of the Lyapunov Function. To use a Lyapunov argument to study the convergence rate of discrete and stochastic dynamics, existing work, e.g., Chen et al. (2020) shows that the smoothness $^{2}$ of the Lyapunov function plays an important role. However, the Lyapunov function proposed to study the continuous-time smoothed best-response dynamics is not a smooth function on the joint probability simplex. Our smoothed best-response update comes to the rescue: we show that it ensures the policies generated by our learning dynamics are naturally uniformly bounded away from zero. By restricting our analysis to the interior of the joint probability simplex (which does not contain any extreme points), we are able to establish the smoothness of the Lyapunov function, thereby making way for our Lyapunov approach.

Non-Zero-Sum Payoffs due to Independent Learning. As illustrated in Section 1.1.2, the inner loop of DSBR-VI is designed to learn the Nash equilibrium of an auxiliary matrix game induced by the value functions $v_{t}^{i}$ and $v_{t}^{-i}$ , where $t$ is the outer-loop iteration index. Importantly, $v_{t}^{i}$ and $v_{t}^{-i}$ are maintained individually by players $i$ and $-i$ , and hence do not necessarily satisfy $v_{t}^{i} + v_{t}^{-i} = 0$ due to independent learning. As a result, the auxiliary matrix game from the inner loop does not necessarily admit a zero-sum structure. See Section 3.1 for more details. The error induced from such non-zero-sum structure appears in existing work Sayin et al. (2021, 2022a), and was handled by designing a novel truncated Lyapunov function. However, the truncated Lyapunov function was sufficient to establish the asymptotic convergence, but did not provide the explicit rate at which the induced error goes to zero. To facilitate finite-sample analysis, in addition to the standard Lyapunov functions used to analyze the $q$ -functions, the policies, and the $v$ -functions, we introduce $\| v_{t}^{i} + v_{t}^{-i}\|_{\infty}$ as an additional Lyapunov function to capture the behavior of the induced error from the non-zero-sum structure of the inner-loop auxiliary matrix game.

Coupled Lyapunov Drift Inequalities. When using Lyapunov arguments for finite-sample analysis, once the Lyapunov drift inequality is established, the finite-sample bound follows straightforwardly by repeatedly invoking the result. However, since our learning dynamics maintains multiple sets of iterates (the value functions, the policies, and the $q$ -functions) and updates them in a coupled manner, the Lyapunov drift inequalities we establish are also highly coupled. Decoupling the Lyapunov drift inequalities without compromising the convergence rate is a major challenge. We develop a systematic strategy for decoupling, which crucially relies on a bootstrapping argument where we first establish a crude bound of the Lyapunov function and then substitute the bound back into the Lyapunov drift inequalities to obtain a tighter one.

# 1.3 Related Work

Before presenting our problem formulations and analysis, we first briefly summarize related and prior work in single-agent RL, MARL, and learning in games.

Single-Agent RL. The most related works (in single-agent RL) to our paper are those that perform finite-sample analysis for RL in infinite-horizon discounted Markov decision processes following a single trajectory of Markovian samples (Even-Dar and Mansour, 2003; Bhandari et al., 2018; Zou et al., 2019; Srikant and Ying, 2019; Li et al., 2020; Chen et al., 2020; Qu and Wierman, 2020; Chen et al., 2021b; Lan, 2022; Yan et al., 2022). In particular, Bhandari et al. (2018); Srikant and Ying (2019) establish finite-sample bounds for TD-learning (with linear function approximation), and Li et al. (2020); Qu and Wierman (2020); Chen et al. (2021b) establish finite-sample bounds for Q-learning. In both cases, the behavior policy for sampling is some stationary policy. For non-stationary behavior policies as we consider, Zou et al. (2019) establishes finite-sample bounds for SARSA, an on-policy RL algorithm, with additional assumptions that control the varying rate of the non-stationary policy.

Sample-Efficient MARL. There has been increasing study of MARL with sample efficiency guarantees recently (Bai and Jin, 2020; Bai et al., 2020; Liu et al., 2021; Xie et al., 2020; Jin et al., 2021; Song et al., 2022; Mao et al., 2022; Daskalakis et al., 2022; Cui et al., 2023). Most of them focus on the finite-horizon episodic setting with online explorations and perform regret analysis, which differs from our finite-sample analysis. Additionally, these algorithms are episodic due to the finite-horizon nature of the setting, and are not best-response type independent learning dynamics that are repeatedly run for infinitely long, which can be viewed as a non-equilibrating adaptation process. In fact, the primary focus of this line of work is a self-play setting where all the players can be controlled to perform centralized learning (Wei et al., 2017; Bai and Jin, 2020; Bai et al., 2020; Liu et al., 2021; Xie et al., 2020). Beyond the online setting, finite-sample efficiency has also been established for MARL using a generative model (Zhang et al., 2020; Li et al., 2022) or offline datasets (Cui and Du, 2022b,a; Zhong et al., 2022; Yan et al., 2022). These algorithms tend to be centralized in nature and focus on equilibrium computation, and thus do not perform independent learning.

Finite-sample complexity has also been established for policy gradient methods, a popular RL approach, when applied to solving zero-sum stochastic games (Daskalakis et al., 2020; Zhao et al., 2021; Zhang et al., 2021b; Alacaoglu et al., 2022). However, to ensure convergence, these methods are asymmetric in that the players update their policies at different timescales, e.g., one player updates faster than the other with larger stepsizes; or one player fixes its policy while waiting for the other to update. Such asymmetric policy gradient methods are not completely independent, as some implicit coordination is required to enable such a timescale separation across agents. This style of implicit coordination is also required for the finite-sample analysis of decentralized learning in certain general-sum stochastic games, e.g., Gao et al. (2021), which improves the asymptotic convergence in Arslan and Yüksel (2017).

Independent Learning in Games. Independent learning has been well-studied in the literature on learning in matrix games. Fictitious play (FP) (Brown, 1951) may be viewed as the earliest of this kind, and its convergence analysis for the zero-sum setting is provided in Robinson (1951). In FP, each player chooses the best response to its estimate of the opponent's strategy via the history of the play, an idea we also follow. Smoothed versions of FP have been developed (Fudenberg and Kreps, 1993; Hofbauer and Sandholm, 2002) to make the learning dynamics consistent (Fudenberg and Levine, 1995, 1998). Moreover, no-regret learning algorithms, extensively studied in online learning, can also be used as independent learning dynamics for matrix games (Cesa-Bianchi and Lugosi, 2006). It is known that they are both convergent and rational by the definition of Bowling and Veloso (2001), and are usually implemented in a symmetric way. See Cesa-Bianchi and Lugosi (2006) for a detailed introduction to no-regret learning in games.

For stochastic games, independent and symmetric policy gradient methods have been developed in recent years, mostly for the case of potential games (Zhang et al., 2021c; Ding et al., 2022; Leonardos et al., 2022). The zero-sum case is more challenging since there is no off-the-shelf Lyapunov function, which the potential function in the potential game case serves as. For non-potential game settings, symmetric variants of policy gradient methods have been proposed, but have only been studied under the full-information setting without finite-sample guarantees (Cen et al., 2021, 2022; Pattathil et al., 2022; Zhang et al., 2022a; Zeng et al., 2022; Erez et al., 2022), with the exception of Wei et al. (2021); Chen et al. (2021a). However, the learning algorithm in Wei et al. (2021) requires some coordination between the players when sampling, and is thus not completely independent; that in Chen et al. (2021a) is extragradient-based and not best-response-type, and needs some stage-based sampling process that also requires coordination across players.

Best-response type independent learning for stochastic games has attracted increasing attention lately (Leslie et al., 2020; Sayin et al., 2021, 2022a,b; Baudin and Laraki, 2022b,a; Maheshwari et al., 2022), with Sayin et al. (2021, 2022a); Baudin and Laraki (2022b,a) tackling the zero-sum setting. However, only asymptotic convergence was established in these works.

# 2 Independent Learning for Zero-Sum Matrix Games

As a warm-up, we begin by considering zero-sum matrix games. This setting introduces both algorithmic and technical ideas that are important for the stochastic game setting, which may be of independent interest.

Let $\mathcal{A}^1$ (respectively, $\mathcal{A}^2$ ) be the finite action-space of player 1 (respectively, player 2), and let $\mathcal{R}^1 \in \mathbb{R}^{|\mathcal{A}^1| \times |\mathcal{A}^2|}$ (respectively, $\mathcal{R}^2 = -(\mathcal{R}^1)^\top$ ) be the payoff matrix of player 1 (respectively, player 2). The decision variables here are the policies $\pi^i \in \Delta^{|\mathcal{A}^i|}$ , $i \in \{1,2\}$ , where $\Delta^{|\mathcal{A}^i|}$ denotes the $|\mathcal{A}^i|$ -dimensional probability simplex. We assume without loss of generality that $\max_{a^i, a^{-i}} |\mathcal{R}^i(a^i, a^{-i})| \leq 1$ , and denote $A_{\max} = \max(|\mathcal{A}^1|, |\mathcal{A}^2|)$ . In what follows, we use $-i$ as the index of player $i$ 's opponent.

Definition 2.1 (Nash Gap in Matrix Games). Given a joint policy $\pi = (\pi^i, \pi^{-i})$ , the Nash gap is defined as

$$
\mathrm{NG} (\pi^ {i}, \pi^ {- i}) := \sum_ {i = 1, 2} \max _ {\hat {\pi} ^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \bigl (\hat {\pi} ^ {i} - \pi^ {i} \bigr) ^ {\top} \mathcal {R} ^ {i} \pi^ {- i}.
$$

# 2.1 Algorithm: Doubly Smoothed Best-Response Dynamics

The high-level idea behind our proposed dynamics is to use a discrete and smoothed variant of the best-response dynamics, where the players construct approximations of the best response to the opponent's policy using a local $q$ -function update that is in the spirit of temporal-difference (TD)-learning in RL (Sutton, 1988). Importantly, while the learning dynamics maintains two sets of iterates (the policies and the $q$ -functions), they are updated on a single time scale with only a multiplicative constant difference in their stepsizes. The details of the learning dynamics are

Algorithm 1 Doubly Smoothed Best-Response Dynamics   
1: Input: Integer K, initializations $q_{0}^{i} = 0 \in R^{|S||A^{i}|}$ and $\pi_{0}^{i} \sim \text{Unif}(A^{i})$ .
2: for $k = 0, 1, \cdots, K - 1$ do
3: $\pi_{k+1}^{i} = \pi_{k}^{i} + \beta_{k} (\sigma_{\tau}(q_{k}^{i}) - \pi_{k}^{i})$ 4: Play $A_{k}^{i} \sim \pi_{k+1}^{i}(\cdot)$ (against $A_{k}^{-i}$ ), and receive reward $\mathcal{R}^{i}(A_{k}^{i}, A_{k}^{-i})$ 5: $q_{k+1}^{i}(a^{i}) = q_{k}^{i}(a^{i}) + \alpha_{k}\mathbb{1}_{\{a^{i}=A_{k}^{i}\}}\left(\mathcal{R}^{i}(A_{k}^{i}, A_{k}^{-i}) - q_{k}^{i}(A_{k}^{i})\right)$ for all $a^{i} \in A^{i}$ 6: end for
7: Output: $\pi_{K}^{i}$

summarized in Algorithm 1, where $\sigma_{\tau}:\mathbb{R}^{|\mathcal{A}^i|}\mapsto \mathbb{R}^{|\mathcal{A}^i|}$ stands for the softmax function with temperature $\tau >0$ . Specifically, we define $[\sigma_{\tau}(q^{i})](a^{i}) = \exp (q^{i}(a^{i}) / \tau) / \sum_{\tilde{a}^{i}}\exp (q^{i}(\tilde{a}^{i}) / \tau)$ for all $a^i\in \mathcal{A}^i$ and $q^{i}\in \mathbb{R}^{|\mathcal{A}^{i}|}$ .

To motivate the algorithm design, we start with the discrete best-response dynamics:

$$
\pi_ {k + 1} ^ {i} = \pi_ {k} ^ {i} + \frac {1}{k + 1} (\mathbf {b r} (\pi_ {k} ^ {- i}) - \pi_ {k} ^ {i}), \quad \mathbf {b r} (\pi_ {k} ^ {- i}) \in \arg \max _ {a ^ {i}} [ \mathcal {R} ^ {i} \pi_ {k} ^ {- i} ] (a ^ {i}), i = 1, 2, \tag {1}
$$

where $e(a^i)$ is the $a^i$ -th unit vector in $\mathbb{R}^{|\mathcal{A}^i|}$ . In Eq. (1), each player updates its randomized policy $\pi_k^i$ incrementally towards the best response to its opponent's current policy, and chooses an action $A_{k+1}^i \sim \pi_{k+1}^i(\cdot)$ . While the dynamics in Eq. (1) provably converges for zero-sum matrix games, see e.g., (Hofbauer and Sorin, 2006), implementing it requires player $i$ to compute $\arg \max_{a^i} [\mathcal{R}^i\pi_k^{-i}](a^i)$ . Note that $\arg \max_{a^i} [\mathcal{R}^i\pi_k^{-i}](a^i)$ involves the exact knowledge of the opponent's policy, which cannot be accessed in independent learning.

To tackle this issue, suppose for now that we are given a stationary joint policy $\pi = (\pi^{i}, \pi^{-i})$ . The problem of player i estimating $R^{i}\pi^{-i}$ can be viewed as a policy evaluation problem, which is usually solved with TD-learning in reinforcement learning. Specifically, the two players repeatedly play the matrix game with the joint policy $\pi = (\pi^{i}, \pi^{-i})$ and produce a sequence of joint actions $\{(A_{k}^{i}, A_{k}^{-i})\}_{k \geq 0}$ . Then, player i forms an estimate of $R^{i}\pi^{-i}$ through the following iterative algorithm:

$$
q _ {k + 1} ^ {i} (a ^ {i}) = q _ {k} ^ {i} (a ^ {i}) + \alpha_ {k} \mathbb {1} _ {\{a ^ {i} = A _ {k} ^ {i} \}} (\mathcal {R} ^ {i} (A _ {k} ^ {i}, A _ {k} ^ {- i}) - q _ {k} ^ {i} (A _ {k} ^ {i})), \quad \forall a ^ {i} \in \mathcal {A} ^ {i}, \tag {2}
$$

with an arbitrary initialization $q_{0}^{i} \in R^{|A^{i}|}$ , where $\alpha_{k} > 0$ is the stepsize. To understand (2), suppose that $q_{k}^{i}$ converges to some $\bar{q}^{i}$ . Then the update equation (2) should be “stationary” at the limit point $\bar{q}^{i}$ in the sense that

$$
\mathbb {E} _ {A ^ {i} \sim \pi^ {i} (\cdot), A ^ {- i} \sim \pi^ {- i} (\cdot)} [ \mathbb {1} _ {\{a ^ {i} = A ^ {i} \}} (\mathcal {R} ^ {i} (A ^ {i}, A ^ {- i}) - \bar {q} ^ {i} (A ^ {i})) ] = 0
$$

for all $a^{i}$ , which would imply that $\bar{q}^{i} = R^{i}\pi^{-i}$ , as desired. The update equation (2), which can be viewed as a simplification of TD-learning in RL to the stateless case, is promising; however, to use $q_{k}^{i}$ as an estimate of $R^{i}\pi_{k}^{-i}$ in Eq. (1), we need to overcome two challenges as below, which inspire us to develop the “double smoothing” technique in Algorithm 1.

\- While we motivated the use of the update equation (2) in the case when the joint policy $(\pi^i, \pi^{-i})$ is stationary, the joint policy $\pi_k = (\pi_k^i, \pi_k^{-i})$ from the discrete best-response dynamics (1) is time-varying. To make TD-learning (2) work for time-varying target policies, a natural approach is to make sure that the policies evolve at a slower rate compared to that of the $q$ -functions, so that $\pi_k$ is close to being stationary from the perspectives of $q_k^i$ . This represents the first form of smoothing in our algorithm. To implement this smoothing, we view $1/(k+1)$ in Eq. (1) as a stepsize and replace it with a more flexible $\beta_k$ , which is chosen to be smaller (but only by a constant multiplicative factor) than the stepsize $\alpha_k$ (cf. Lines 3 and 5 of Algorithm 1).

\- Now we can view $\pi_k$ as if it is stationary in updating the $q$ -functions. In order for TD-learning (2) to converge, a necessary condition is that the target policy (the value of which we want to

estimate) should ensure exploration (Sutton and Barto, 2018). To see this, suppose that we are evaluating a deterministic policy, which has no exploration components. Then $q_{k}^{i}$ generated by TD-learning (2) clearly cannot converge because essentially only one entry of the vector-valued iterate $q_{k}^{i}$ is updated. To overcome this challenge, we smooth the update by using a softmax instead of a hardmax (cf. Line 3 of Algorithm 1), which prevents the policies we want to evaluate from ever being deterministic. We term this as our second form of smoothing.

Given the two algorithmic ideas described above, we arrive at Algorithm 1 – a payoff-based independent learning dynamics for zero-sum matrix games. While TD-learning (Sutton, 1988; Tsitsiklis and Van Roy, 1997) and best-response dynamics (Hofbauer and Sorin, 2006; Harris, 1998; Leslie et al., 2020) are both extensively studied in isolation, the combined use of them to form independent learning dynamics is less studied. The most related work is Leslie and Collins (2005), in which an individual Q-learning algorithm is proposed. Compared with Algorithm 1, the algorithm in Leslie and Collins (2005) uses stochastic stepsizes (which are adaptively updated based on the algorithm trajectory), and a rapidly time-varying behavior policy. In addition, only asymptotic convergence was shown in Leslie and Collins (2005).

As an aside, the continuous version of Eq. (1), i.e., the best-response dynamics

$$
\dot {\pi} ^ {i} \in \arg \max _ {\hat {\pi} ^ {i} \in \Delta^ {i}} (\hat {\pi} ^ {i}) ^ {\top} \mathcal {R} ^ {i} \pi^ {- i} - \pi^ {i},
$$

is frequently used to analyze the convergence behavior of the celebrated FP dynamics for solving zero-sum matrix games; see Leslie et al. (2020) for more details.

# 2.2 Finite-Sample Analysis

We now present a finite-sample analysis of Algorithm 1, deferring its proofs to Appendix B. We consider stepsizes of the form $\alpha_{k} = \alpha / (k + h)^{z}$ , where $\alpha, h > 0$ and $z \in [0,1]$ . Note that $z = 0$ corresponds to the constant stepsize case. The stepsize $\beta_{k}$ satisfies $\beta_{k} = c_{\alpha,\beta}\alpha_{k}$ for any $k \geq 0$ , where $c_{\alpha,\beta} \in (0,1)$ is a tunable constant. Importantly, the stepsizes $\alpha_{k}$ and $\beta_{k}$ differ only by a multiplicative constant, which makes Algorithm 1 a single time-scale algorithm that is easier to implement than a two time-scale one.

In the following theorem, the parameters $\{c_{j}\}_{0\leq j\leq3}$ are numerical constants, and the parameter $\ell_{\tau}$ (the explicit expression of which is presented in Appendix A) depends only on the temperature $\tau$ and $A_{max}$ .

Theorem 2.1. Suppose that both players follow Algorithm 1 and $c_{\alpha, \beta} \leq \frac{\ell_x^3 \tau^3}{c_0 A_{\max}^2}$ .

(1) When $\alpha_{k} \equiv \alpha$ , we have

$$
\mathbb {E} [ N G (\pi_ {K} ^ {i}, \pi_ {K} ^ {- i}) ] \leq 3 \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {K} + \frac {c _ {1} A _ {\max} ^ {3 / 2}}{c _ {\alpha , \beta}} \alpha + 2 \tau \log (A _ {\max}). \tag {3}
$$

(2) When $\alpha_{k} = \alpha / (k + h)$ with $\alpha > 2 / c_{\alpha, \beta}$ and $h > \alpha$ , we have

$$
\mathbb {E} [ N G (\pi_ {K} ^ {i}, \pi_ {K} ^ {- i}) ] \leq 3 \left(\frac {h}{K + h}\right) ^ {c _ {\alpha , \beta} \alpha / 2} + \frac {c _ {2} A _ {\max} ^ {3 / 2} \alpha}{c _ {\alpha , \beta} \alpha - 2} \frac {\alpha}{K + h} + 2 \tau \log (A _ {\max}).
$$

(3) When $\alpha_{k} = \alpha /(k + h)^{z}$ with $\alpha >0,  z\in (0,1)$ , and $h\geq (\frac{4z}{c_{\alpha,\beta}\alpha})^{\frac{1}{1 - z}}$ , we have

$$
\begin{array}{l} \mathbb {E} [ N G (\pi_ {K} ^ {i}, \pi_ {K} ^ {- i}) ] \leq 3 \exp \left(- \frac {\alpha ((K + h) ^ {1 - z} - h ^ {1 - z})}{2 c _ {\alpha , \beta} (1 - z)}\right) + \frac {c _ {3} A _ {\max} ^ {3 / 2}}{c _ {\alpha , \beta}} \frac {\alpha}{(K + h) ^ {z}} \\ + 2 \tau \log (A _ {\mathrm{max}}). \\ \end{array}
$$

In all the three cases in Theorem 2.1, the bound is a combination of convergence bias, variance, and smoothing bias. The behavior of the convergence bias and the variance agrees with existing literature on stochastic approximation algorithms (Srikant and Ying, 2019; Chen et al., 2021b; Bhandari et al., 2018). In particular, large stepsizes result in smaller convergence bias but larger variance. When using $\mathcal{O}(1/k)$ stepsizes, we achieve the best convergence rate of $\mathcal{O}(1/K)$ . The smoothing bias $2\tau\log(A_{\max})$ arises from using softmax instead of hardmax in Algorithm 1, which can also be viewed as the difference between the Nash distribution (Leslie and Collins, 2005) and a Nash equilibrium.

Importantly, with only bandit feedback, we achieve an $\mathcal{O}(1/K)$ rate of convergence to a Nash equilibrium up to a smoothing bias. In general, for smooth and strongly monotone games, the lower bound for the rate of convergence of payoff-based or zeroth-order algorithms is $\mathcal{O}(1/\sqrt{K})$ (Lin et al., 2021). We have an improved $\mathcal{O}(1/K)$ . convergence rate because our learning dynamics can exploit the bilinear structure of the game. In particular, we are able to use only the bandit feedback to construct an efficient estimator (using the q-functions) of the marginalized payoff $R^{i}\pi_{k}^{-i}$ (which can also be interpreted as the gradient), thereby enjoy the fast $\mathcal{O}(1/K)$ rate of convergence that is comparable to first-order method (Beznosikov et al., 2022).

Based on Theorem 2.1, we next derive the sample complexity in the following corollary.

Corollary 2.1.1 (Sample Complexity). Given $\epsilon > 0$ , to achieve $\mathbb{E}[NG(\pi_K^i, \pi_K^{-i})] \leq \epsilon + 2\tau \log(A_{\max})$ , the sample complexity is $\mathcal{O}(\epsilon^{-1})$ .

Notably, we achieve $\tilde{\mathcal{O}}(1/\epsilon)$ sample complexity up to a smoothing bias. The stepsize ratio appears only as a multiplicative constant in the bound, and does not impact the rate, which is the advantage of using a single time-scale algorithm.

When the opponent does not follow Algorithm 1, but plays with a stationary policy, the following corollary states that we have the same $\mathcal{O}(1/\epsilon)$ sample complexity for the player to find an optimal policy against its opponent.

Corollary 2.1.2 (Rationality). Suppose that player i follows the learning dynamics presented in Algorithm 1, but its opponent follows a stationary policy $\pi^{-i}$ . Then, given $\epsilon > 0$ , to achieve $\mathbb{E}[\max_{\hat{\pi}^{i}}(\hat{\pi}^{i} - \pi_{K}^{i})^{\top}\mathcal{R}^{i}\pi^{-i}] \leq \epsilon + 2\tau \log(A_{\max})$ , the sample complexity is $\mathcal{O}(1/\epsilon)$ .

According to the definition in Bowling and Veloso (2001), a dynamics being rational means that the player following this dynamics will converge to the best response to its opponent when the opponent uses an asymptotically stationary policy. Since we are performing finite-sample analysis, we assume the opponent's policy is stationary, because otherwise, the convergence rate (which may be arbitrary) of the opponent's policy will also impact the bound.

# 3 Independent Learning for Zero-Sum Markov Games

This section presents our main technical and algorithmic contributions. We introduce a payoff-based, single-trajectory, convergent, rational, and independent learning dynamics for zero-sum Markov games. Consider an infinite-horizon two-player zero-sum Markov game $\mathcal{M} = (\mathcal{S},\mathcal{A}^1,\mathcal{A}^2,p,\mathcal{R}^1,\mathcal{R}^2,\gamma)$ , where $\mathcal{S}$ is the finite state-space, $\mathcal{A}^1$ (respectively, $\mathcal{A}^2$ ) is the finite action-space for player 1 (respectively, player 2), $p$ represents the transition probabilities, in particular, $p(s^{\prime}\mid s,a^{1},a^{2})$ is the probability of transitioning to state $s^{\prime}$ after player 1 taking action $a^1$ and player 2 taking action $a^2$ simultaneously at state $s$ , $\mathcal{R}^1:\mathcal{S}\times \mathcal{A}^1\times \mathcal{A}^2\mapsto \mathbb{R}$ (respectively, $\mathcal{R}^2:\mathcal{S}\times \mathcal{A}^2\times \mathcal{A}^1\mapsto \mathbb{R}$ ) is player 1's (respectively, player 2's) reward function, and $\gamma \in [0,1)$ is the discount factor. Note that we have $\mathcal{R}^1 (s,a^1,a^2) + \mathcal{R}^2 (s,a^2,a^1) = 0$ for all $(s,a^{1},a^{2})$ . We assume without loss of generality that $\max_{s,a^1,a^2}|\mathcal{R}^1 (s,a^1,a^2)|\leq 1$ , and denote $A_{\mathrm{max}} = \max (|\mathcal{A}^1|,|\mathcal{A}^2|)$ .

Given a joint stationary policy $\pi = (\pi^i, \pi^{-i})$ , where $\pi^i: \mathcal{S} \mapsto \Delta^{|\mathcal{A}^i|}$ and $\pi^{-i}: \mathcal{S} \mapsto \Delta^{|\mathcal{A}^{-i}|}$ , we define the local $q$ -function $q_\pi^i \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}^i|}$ of player $i$ as

$$
q _ {\pi} ^ {i} (s, a ^ {i}) = \mathbb {E} _ {\pi} \left[ \sum_ {k = 0} ^ {\infty} \gamma^ {i} \mathcal {R} ^ {i} (S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i})   \bigg |   S _ {0} = s, A _ {0} ^ {i} = a ^ {i} \right]
$$

for all $(s, a^i)$ , where we use the notation $\mathbb{E}_{\pi}[\cdot]$ to indicate that the actions are chosen according to the joint policy $\pi$ . In addition, we define the $v$ -function $v_{\pi}^{i} \in \mathbb{R}^{|S|}$ as $v_{\pi}^{i}(s) = \mathbb{E}_{A^i \sim \pi^i (\cdot | s)}[q_{\pi}^{i}(s, A^{i})]$ for all $s$ , and the utility function $U^{i}(\pi^{i}, \pi^{-i}) \in \mathbb{R}$ as $U^{i}(\pi^{i}, \pi^{-i}) = \mathbb{E}_{S \sim p_{o}}[v_{\pi}^{i}(S)]$ , where $p_{o} \in \Delta^{|S|}$ is an arbitrary initial distribution on the states.

Definition 3.1 (Nash Gap in Markov Games). Given a joint policy $\pi = (\pi^i, \pi^{-i})$ , the Nash gap is defined as

$$
N G (\pi^ {i}, \pi^ {- i}) = \sum_ {i = 1, 2} \left(\max _ {\hat {\pi} ^ {i}} U ^ {i} (\hat {\pi} ^ {i}, \pi^ {- i}) - U ^ {i} (\pi^ {i}, \pi^ {- i})\right).
$$

In what follows, we will frequently work with the real vectors in $\mathbb{R}^{|\mathcal{S}||\mathcal{A}^i|}$ , $\mathbb{R}^{|\mathcal{S}||\mathcal{A}^{-i}|}$ , and $\mathbb{R}^{|\mathcal{S}||\mathcal{A}^i||\mathcal{A}^{-i}|}$ . To simplify the notation, for any $Q \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}^i||\mathcal{A}^{-i}|}$ , we use $Q(s)$ to denote the $|\mathcal{A}^i| \times |\mathcal{A}^{-i}|$ matrix with the $(a^i, a^{-i})$ -th entry being $Q(s, a^i, a^{-i})$ . Similarly, for any $q \in \mathbb{R}^{|\mathcal{S}||\mathcal{A}^i|}$ , we use $q(s)$ to denote the $|\mathcal{A}^i|$ -dimensional vector with its $a^i$ -th entry being $q(s, a^i)$ .

# 3.1 Algorithm: Doubly Smoothed Best-Response Dynamics with Value Iteration

Our learning dynamics for Markov games (cf. Algorithm 2) builds on the ideas presented in our algorithm design for matrix games in Section 2.1, with the additional incorporation of minimax value iteration, a well-known approach for zero-sum stochastic games (Shapley, 1953).

Algorithmic Ideas. To motivate the algorithm design, we need to introduce the following notation. For $i \in \{1, 2\}$ , let $T^{i} : R^{|S|} \mapsto R^{|S||A^{i}||A^{-i}|}$ be an operator defined as

$$
\mathcal {T} ^ {i} (v) (s, a ^ {i}, a ^ {- i}) = \mathcal {R} ^ {i} (a, a ^ {i}, a ^ {- i}) + \gamma \mathbb {E} \left[ v (S _ {1}) \mid S _ {0} = s, A _ {0} ^ {i} = a ^ {i}, A _ {0} ^ {- i} = a ^ {- i} \right]
$$

for all $(s, a^i, a^{-i})$ and $v \in \mathbb{R}^{|\mathcal{S}|}$ . We also define $val^i: \mathbb{R}^{|\mathcal{A}^i| \times |\mathcal{A}^{-i}|} \mapsto \mathbb{R}$ to be the following operator

$$
v a l ^ {i} (X) = \max _ {\mu^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \min _ {\mu^ {- i} \in \Delta^ {| \mathcal {A} ^ {- i} |}} \{(\mu^ {i}) ^ {\top} X \mu^ {- i} \} = \min _ {\mu^ {- i} \in \Delta^ {| \mathcal {A} ^ {- i} |}} \max _ {\mu^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \{(\mu^ {i}) ^ {\top} X \mu^ {- i} \}
$$

for all $X \in \mathbb{R}^{|\mathcal{A}^i| \times |\mathcal{A}^{-i}|}$ . Then, the minimax Bellman operator $\mathcal{B}^i: \mathbb{R}^{|\mathcal{S}|} \mapsto \mathbb{R}^{|\mathcal{S}|}$ is defined as

$$
\mathcal {B} ^ {i} (v) (s) = v a l ^ {i} (\mathcal {T} ^ {i} (v) (s))
$$

for all $s \in S$ , where $\mathcal{T}^{i}(v)(s)$ is an $|\mathcal{A}^{i}| \times |\mathcal{A}^{-i}|$ matrix according to our notation. It is known that the operator $\mathcal{B}^{i}(\cdot)$ is a $\gamma$ -contraction mapping with respect to the $\ell_{\infty}$ -norm (Shapley, 1953), hence admits a unique fixed-point, which we denote by $v_{*}^{i}$ .

A common approach for solving Markov games is to first implement the minimax value iteration $v_{t+1}^{i} = \mathcal{B}^{i}(v_{t}^{i})$ until (approximate) convergence to $v_{*}^{i}$ , and then solve the matrix game

$$
\max _ {\mu^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \min _ {\mu^ {- i} \in \Delta^ {| \mathcal {A} ^ {- i} |}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s) \mu^ {- i}
$$

for each state $s$ to obtain an (approximate) Nash equilibrium policy. However, implementing this algorithm requires complete knowledge of the underlying transition probabilities. Moreover, since it is an off-policy algorithm, the output is independent of the opponent's policy. Thus, it is not rational by the definition in Bowling and Veloso (2001). To design a model-free and rational learning dynamics, let us first rewrite the minimax value iteration in the following equivalent way:

$$
\hat {v} (s) = \max _ {\mu^ {i}} \min _ {\mu^ {- i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \mu^ {- i}, \forall s \in \mathcal {S}, \tag {4}
$$

$$
v _ {t + 1} ^ {i} = \hat {v}. \tag {5}
$$

In view of Eqs. (4) and (5), we need to solve a matrix game with payoff matrix $\mathcal{T}^{i}(v_{t}^{i})(s)$ for each state s and then update the value of the game to $v_{t+1}^{i}(s)$ . In light of Algorithm 1, we already know how to solve matrix games with independent learning. Thus, what remains is to combine Algorithm 1 with value iteration, i.e., Eq. (5). This combination yields Algorithm 2.

Algorithm 2 Doubly Smoothed Best-Response Dynamics with Value Iteration   
1: Input: Integers K and T, initializations $v_{0}^{i} = 0 \in R^{|S|}$ , $q_{t,0}^{i} = 0 \in R^{|S||A^{i}|}$ for all t, $\pi_{t,0}^{i}(a^{i}|s) = 1/|\mathcal{A}^{i}|$ for all $(s, a^{i})$ and t, and $S_{0}$ arbitrarily.
2: for $t = 0, 1, \cdots, T$ do
3:    for $k = 0, 1, \cdots, K - 1$ do
4: $\pi_{t,k+1}^{i}(s) = \pi_{t,k}^{i}(s) + \beta_{k}(\sigma_{\tau}(q_{t,k}^{i}(s)) - \pi_{t,k}^{i}(s))$ for all $s \in S$ 5:    Play $A_{k}^{i} \sim \pi_{t,k+1}^{i}(\cdot | S_{k})$ (against $A_{k}^{-i}$ ), and observe $S_{k+1} \sim p(\cdot | S_{k}, A_{k}^{i}, A_{k}^{-i})$ 6: $q_{t,k+1}^{i}(s, a^{i}) = q_{t,k}^{i}(s, a^{i}) + \alpha_{k}\mathbb{1}_{\{(s,a^{i})=(S_{k},A_{k}^{i})\}}(\mathcal{R}^{i}(S_{k}, A_{k}^{i}, A_{k}^{-i}) + \gamma v_{t}^{i}(S_{k+1}) - q_{t,k}^{i}(S_{k}, A_{k}^{i}))$ for all $(s, a^{i})$ 7:    end for
8: $v_{t+1}^{i}(s) = \pi_{t,K}^{i}(s)^{\top} q_{t,K}^{i}(s)$ for all $s \in S$ and set $S_{0} = S_{K}$ 9: end for
10: Output: $\pi_{T,K}^{i}$

Algorithm Details. For each state s, the inner-loop of Algorithm 2 is designed to solve a matrix game with payoff matrices $\mathcal{T}^{i}(v_{t}^{i})(s)$ and $\mathcal{T}^{-i}(v_{t}^{-i})(s)$ , which reduces to Algorithm 1 when (1) the Markov game has only one state, and (2) $v_{t}^{i} = v_{t}^{-i} = 0$ . However, since $v_{t}^{i}$ and $v_{t}^{-i}$ are independently maintained by player i and its opponent, the quantity

$$
\mathcal {T} ^ {i} (v _ {t} ^ {i}) (s, a ^ {i}, a ^ {- i}) + \mathcal {T} ^ {- i} (v _ {t} ^ {- i}) (s, a ^ {- i}, a ^ {i}) = \gamma \sum_ {s ^ {\prime}} p (s ^ {\prime} \mid s, a ^ {i}, a ^ {- i}) (v _ {t} ^ {i} (s) + v _ {t} ^ {- i} (s))
$$

is in general non-zero during learning. As a result, the auxiliary matrix game (with payoff matrices $\mathcal{T}^i(v_t^i)(s)$ and $\mathcal{T}^{-i}(v_t^{-i})(s)$ ) that the inner loop of Algorithm 2 is designed to solve is not necessarily zero-sum, which presents a major challenge in the finite-sample analysis, as illustrated previously in Section 1.2.

The outer loop of Algorithm 2 is an “on-policy” variant of minimax value iteration. To see this, note that ideally we would synchronize $v_{t+1}^{i}(s)$ with $\pi_{t,K}^{i}(s)^{\top}\mathcal{T}^{i}(v_{t}^{i})(s)\pi_{t,K}^{-i}(s)$ , which is an approximation of $val^{i}(\mathcal{T}^{i}(v_{t}^{i})(s))$ by design of our inner loop. However, player i has no access to $\pi_{K}^{-i}$ in independent learning. Fortunately, the q-function $q_{t,K}^{i}$ is precisely constructed as an estimate of $\mathcal{T}^{i}(v_{t}^{i})(s)\pi_{t,K}^{-i}(s)$ , as illustrated in Section 2.1, which leads to the outer loop of Algorithm 2. In Line 8 of Algorithm 2, we set $S_{0}=S_{K}$ to ensure that the initial state of the next inner-loop is the last state of the previous inner-loop, hence Algorithm 2 is driven by a single trajectory of Markovian samples.

# 3.2 Finite-Sample Analysis

We now state our main results, which provide the first finite-sample bounds for best-response type independent learning dynamics in zero-sum Markov games. A detailed analysis is provided in Section 4 and the complete proof are provided in Appendix A. Our results rely on one assumption.

Assumption 3.1. There exists a joint policy $\pi_b = (\pi_b^i, \pi_b^{-i})$ such that the Markov chain $\{S_k\}_{k \geq 0}$ induced by $\pi_b$ is irreducible and aperiodic.

Most, if not all, analyses of RL algorithms driven by time-varying behavior policies assume that the induced Markov chain of any policy, or any policy from the algorithm trajectory, is uniformly geometrically ergodic (Zou et al., 2019; Khodadadian et al., 2022; Chen et al., 2022, 2021a; Xu and Liang, 2021; Wu et al., 2020; Qiu et al., 2021). Assumption 3.1 is weaker, since it assumes only the existence of one policy that induces an irreducible and aperiodic Markov chain.

In the following theorems, we consider using either constant stepsize $\alpha_{k} \equiv \alpha$ , or diminishing stepsize $\alpha_{k} = \alpha/(k+h)$ . In either case, $\beta_{k} = c_{\alpha,\beta}\alpha_{k}$ with $c_{\alpha,\beta} \in (0,1)$ being a tunable constant. The parameters $\{\hat{c}_{j}\}_{0 \leq j \leq 4}$ used to state the following theorem are numerical constants, and $c_{\tau}$ ,

$\ell_{\tau}$ , and $\hat{L}_{\tau}$ are constants that depend on the temperature $\tau$ and $A_{\mathrm{max}}$ . See Appendix A for the explicit expressions of the quantities.

Theorem 3.1 (Constant Stepsize Bound). Suppose that both players follow Algorithm 2, Assumption 3.1 is satisfied, and the stepsize ratio satisfies $c_{\alpha, \beta} \leq \frac{c_{\tau} \tau^3 \ell_{\tau}^2 (1 - \gamma)^2}{\hat{c}_0 |S| A_{\max}^2}$ . Then, there exists a threshold $z_{\beta} = \mathcal{O}(\log(1 / \beta))$ such that the following inequality holds as long as $K \geq z_{\beta}$ :

$$
\begin{array}{l} \mathbb {E} [ N G (\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}) ] \leq \underbrace {\frac {\hat {c} _ {1} | \mathcal {S} | A _ {\max} T}{\tau (1 - \gamma) ^ {3}} \left(\frac {1 + \gamma}{2}\right) ^ {T - 1}} _ {\mathcal {E} _ {1}: \text {   Value   Iteration   Bias }} \\ + \underbrace {\frac {\hat {c} _ {2} (| \mathcal {S} | A _ {\max}) ^ {3 / 2} (K - z _ {\beta}) ^ {1 / 2}}{\tau (1 - \gamma) ^ {5}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}}} _ {\mathcal {E} _ {2}: C o n v e r g e n c e B i a s i n t h e I n n e r - L o o p} \\ + \underbrace {\frac {\hat {c} _ {3} | \mathcal {S} | ^ {2} A _ {\max} ^ {2} \hat {L} _ {\tau}}{c _ {\alpha , \beta} (1 - \gamma) ^ {5}} z _ {\beta} ^ {2} \alpha^ {1 / 2}} _ {\mathcal {E} _ {3}: \text { Variance   in   the   Inner - Loop }} + \underbrace {\frac {\hat {c} _ {4} \tau \log (A _ {\max})}{(1 - \gamma) ^ {2}}} _ {\mathcal {E} _ {4}: \text { Smoothing   Bias }}. \tag {6} \\ \end{array}
$$

As in Theorem 2.1 for matrix games, the bound includes terms for the convergence bias, variance, and smoothing bias. However, now there is an additional term capturing the value iteration bias. More specifically, the first term $E_{1}$ on the right-hand side of Eq. (6) is referred to as the value iteration bias, and would be the only error term if we were able to perform minimax value iteration to solve the game. The terms $E_{2}$ and $E_{3}$ are the counterparts of the first two terms on the right-hand side of Eq. (3) in Theorem 2.1, and capture the convergence bias and the variance in the inner-loop. The term $E_{4}$ represents the smoothing bias resulted from using softmax instead of hardmax in the learning dynamics. Since a Markov game is a sequential decision making problem, the smoothing bias is accumulated over time, and hence is multiplied by a factor depending on the effective horizon of the problem compared to its counterpart in matrix games.

Notably, the terms $E_{2}$ and $E_{3}$ are order-wise larger compared to their matrix game counterparts, which is the (mathematical) reason that Algorithm 2 has a slower convergence rate (or larger sample complexity) compared to that of Algorithm 1. Intuitively, the reason is that the induced auxiliary matrix game (with payoff matrices $\mathcal{T}^{i}(v_{t}^{i})(s)$ and $\mathcal{T}^{-i}(v_{t}^{-i})(s)$ ) that the inner-loop of Algorithm 2 is designed to solve does not necessarily have a zero-sum structure (see the discussion after Algorithm 2). Consequently, the error due to such “non-zero-sum” structure propagates through the algorithm and eventually undermines the rate of convergence.

We next consider using diminishing stepsizes, i.e., $\alpha_{k} = \alpha / (k + h)$ and $\beta_{k} = c_{\alpha, \beta} \alpha_{k}$ . The requirement for choosing $\alpha$ and $h$ are presented in Appendix A. The parameters $\{\hat{c}_j'\}_{0 \leq j \leq 3}$ used in presenting the following theorem are numerical constants.

Theorem 3.2 (Diminishing Stepsizes Bound). Suppose that both players follow the learning dynamics in Algorithm 2, Assumption 3.1 is satisfied, and the stepsize ratio satisfies $c_{\alpha, \beta} \leq \frac{c_{\tau} \tau^3 \ell_\tau^2 (1 - \gamma)^2}{\hat{c}_0' |S| A_{\max}^2}$ . Then there exists a threshold $k_0 > 0$ such that the following inequality holds as long as $K \geq k_0$ :

$$
\begin{array}{l} \mathbb {E} [ N G (\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}) ] \leq \underbrace {\frac {\hat {c} _ {1} ^ {\prime} | \mathcal {S} | A _ {\max} T}{\tau (1 - \gamma) ^ {3}} \left(\frac {\gamma + 1}{2}\right) ^ {T - 1}} _ {\mathcal {E} _ {1} ^ {\prime}} + \underbrace {\frac {\hat {c} _ {2} ^ {\prime} | \mathcal {S} | ^ {2} A _ {\max} ^ {2} \hat {L} _ {\tau}}{\alpha_ {k _ {0}} c _ {\alpha , \beta} (1 - \gamma) ^ {5}} \frac {z _ {K} ^ {2} \alpha^ {1 / 2}}{(K + h) ^ {1 / 2}}} _ {\mathcal {E} _ {2, 3} ^ {\prime}} \\ + \underbrace {\frac {\hat {c} _ {3} ^ {\prime} \tau \log (A _ {\max})}{(1 - \gamma) ^ {2}}} _ {\mathcal {E} _ {4} ^ {\prime}}, w h e r e z _ {K} = \mathcal {O} (\log (K)). \\ \end{array}
$$

The terms $E_{1}^{\prime}$ and $E_{4}^{\prime}$ are quantitatively similar to the terms $E_{1}$ and $E_{4}$ in Eq. (6), and represent the value iteration bias and the smoothing bias. The term $E_{2,3}^{\prime}$ corresponds to $E_{2} + E_{3}$ in Theorem

3.1, and captures the combined error of the convergence bias and the variance in the inner loop. Since we are using diminishing stepsizes, unlike Eq. (6), the convergence bias and the variance are balanced, and are both converging at the same rate.

We next present the sample complexity of Algorithm 2, which does not depend on whether constant or diminishing stepsizes are used.

Corollary 3.2.1 (Sample Complexity). To achieve $\mathbb{E}[NG(\pi_{T,K}^{i},\pi_{T,K}^{-i})]\leq \epsilon +\frac{\hat{c}_4\tau\log(A_{\max})}{(1 - \gamma)^2}$ for some $\epsilon >0$ , the sample complexity is $\tilde{\mathcal{O}} (\epsilon^{-2})$ .

Notably, we achieve an $\tilde{\mathcal{O}}(1/\epsilon^{2})$ sample complexity to find a Nash equilibrium up to a smoothing bias, which is order-wise the same compared with the sample complexity of popular RL algorithms in the single agent setting, such as Q-learning (Qu and Wierman, 2020; Li et al., 2020; Chen et al., 2021b). We want to emphasize that there are no asymptotic bias terms in those single-agent RL algorithms while we have a smoothing bias. An interesting future direction of this work is to investigate the use of a time-varying temperature $\tau_{k}$ and establish a sharp rate of convergence with an asymptotically vanishing smoothing bias. We suspect that this is a much more challenging task as even in single-agent Q-learning (which is arguably one of the most popular and well-studied algorithms), finite-sample analysis under $\epsilon$ -greedy policy with a time-varying $\epsilon$ (or softmax exploration policy with a time-varying temperature) was not performed in the literature.

Finally, we consider the case where the opponent plays with a stationary policy, and provide a sample complexity bound for the player to find the best-response.

Corollary 3.2.2 (Rationality). Suppose that player i follows the learning dynamics presented in Algorithm 2, but its opponent follows a stationary policy $\pi^{-i}$ . Then, given $\epsilon > 0$ , to achieve $\max_{\hat{\pi}^{i}} U^{i}(\hat{\pi}^{i}, \pi^{-i}) - \mathbb{E}[U^{i}(\pi_{T,K}^{i}, \pi^{-i})] \leq \epsilon + \frac{\dot{c}_{4}\tau\log(A_{\max})}{(1-\gamma)^{2}}$ , the sample complexity is $\tilde{\mathcal{O}}(1/\epsilon^{2})$ .

Intuitively, the reason that our algorithm is rational is that it performs the so-called on-policy update in RL. In contrast to an off-policy update, where the behavior policy can be arbitrarily different from the policy being generated during learning (such as in Q-learning and off-policy TD-learning), in the on-policy update for games, each player is actually playing with the policy that is moving towards the best-response to its opponent. As a result, when the opponent's policy is stationary, it reduces to a single-agent problem and the player naturally finds the best response (also up to a smoothing bias). This is also exactly the advantage of symmetric and independent learning dynamics.

# 4 Analyzing the Learning Dynamics in Algorithm 2

In this section, we present the key steps and technical ideas used to prove Theorem 3.1 and Theorem 3.2. The core challenge here is that Algorithm 2 maintains 3 sets of iterates ( $\{q_{t,k}^{i}\}, \{\pi_{t,k}^{i}\}$ , and $\{v_t^i\}$ ), which are coupled. The coupling of their update equations means that it is not possible to separately analyze them. Instead, we develop a coupled Lyapunov drift argument to establish the finite-sample bounds of Algorithm 2. Specifically, we first show that the expected Nash gap can be upper bounded by a sum of properly defined Lyapunov functions, one for each set of the iterates (i.e., the $v$ -functions, the policies, and the $q$ -functions). Then, we establish a set of coupled Lyapunov drift inequalities – one for each Lyapunov function. Finally, we decouple the Lyapunov drift inequalities to establish the overall finite-sample bounds. We outline the key steps in the argument below.

To begin with, we show that the q-functions $\{q_{t,k}^{i}\}$ and the v-functions $\{v_{t}^{i}\}$ generated by Algorithm 2 are uniformly bounded from above in $\ell_{\infty}$ -norm by $1/(1-\gamma)$ (cf. Lemma A.1), and the entries of the policies $\{\pi_{t,k}^{i}\}$ are uniformly bounded below by $\ell_{\tau}>0$ (cf. Lemma A.2). These two results are frequently used in our analysis.

At the core of our argument is the following inequality:

$$
N G (\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}) \leq C _ {0} \left(2 \| v _ {T} ^ {i} + v _ {T} ^ {- i} \| _ {\infty} + \sum_ {i = 1, 2} \| v _ {T} ^ {i} - v _ {*} ^ {i} \| _ {\infty} + \mathcal {L} _ {\pi} (T, K) + \tau \log (A _ {\max})\right), \tag {7}
$$

where $C_{0}$ is a constant, and $\mathcal{L}_{\pi}(\cdot)$ stands for the Lyapunov function we use to analyze the policies (the explicit expression of which is presented in Eq. (15). Eq. (7) follows from Lemma A.3 and Lemma A.4.

# 4.1 Analysis of the Outer Loop: v-Function Update

Motivated by Eq. (7), we need to bound $\|v_{T}^{i} + v_{T}^{-i}\|_{\infty}$ and $\|v_{T}^{i} - v_{*}^{i}\|_{\infty}$ . To do so, we view them as Lyapunov functions and establish Lyapunov drift inequalities for them. Specifically, we show in Lemma A.5 and Lemma A.6 that

$$
\begin{array}{l} \| v _ {t + 1} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \leq \underbrace {\gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty}} _ {\text { Negative   Drift }} \\ + \underbrace {C _ {1} (\| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + \mathcal {L} _ {\pi} (t , K) + \mathcal {L} _ {q} ^ {1 / 2} (t , K) + \tau \log (A _ {\max}))} _ {\text { Additive   Errors }}, (8) \\ \left\| v _ {t + 1} ^ {i} + v _ {t + 1} ^ {- i} \right\| _ {\infty} \leq \underbrace {\gamma \left\| v _ {t} ^ {i} + v _ {t} ^ {- i} \right\| _ {\infty}} _ {\text { Negative   Drift }} + \underbrace {C _ {2} \mathcal {L} _ {q} ^ {1 / 2} (t , K)} _ {\text { Additive   Errors }} (9) \\ \end{array}
$$

for all $t \geq 0$ , where $C_{1}$ and $C_{2}$ are constants, and $\mathcal{L}_{q}(\cdot)$ stands for the Lyapunov function we use to analyze the q-functions (the expression of which is presented in Eq. (15)). If the Additive Errors in the previous two inequalities were only functions of $v_{t}^{i}$ and $v_{t}^{-i}$ , then these two Lyapunov drift inequalities can be repeatedly used to obtain a convergence bound for $\|v_{T}^{i} + v_{T}^{-i}\|_{\infty}$ and $\|v_{T}^{i} - v_{*}^{i}\|_{\infty}$ . However, the coupled nature of Eqs. (8) and (9) requires us to analyze the policies and the q-functions in the inner loop, and establish their Lyapunov drift inequalities.

# 4.2 Analysis of the Inner Loop: Policy Update

As illustrated in Section 2.1 and Section 3.1, for each state s, the update equation of the policies can be viewed as a discrete and stochastic variant of the smoothed best-response dynamics for solving matrix games (Leslie and Collins, 2005). Typically, the following Lyapunov function is used to study such dynamics:

$$
V _ {X} \left(\mu^ {i}, \mu^ {- i}\right) = \sum_ {i = 1, 2} \max _ {\hat {\mu} ^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \left\{\left(\hat {\mu} ^ {i} - \mu^ {i}\right) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu \left(\hat {\mu} ^ {i}\right) - \tau \nu \left(\mu^ {i}\right) \right\}, \tag {10}
$$

where $X_{i}$ and $X_{-i}$ are the payoff matrices for player i and player -i, respectively, and $\nu(\cdot)$ is the entropy function defined as $\nu(\mu^{i}) = -\sum_{a^{i}} \mu^{i}(a^{i}) \log(\mu^{i}(a^{i}))$ . Specialized to our case, given a joint v-function $v = (v^{i}, v^{-i})$ from the outer loop $^{3}$ , and a state $s \in S$ , we would like to use

$$
V _ {v, s} (\pi^ {i} (s), \pi^ {- i} (s)) := \sum_ {i = 1, 2} \max _ {\hat {\mu} ^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \{(\hat {\mu} ^ {i} - \pi^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v ^ {i}) (s) \pi^ {- i} (s) + \tau \nu (\hat {\mu} ^ {i}) - \tau \nu (\pi^ {i} (s)) \}
$$

as our Lyapunov function. Unlike the continuous-time smoothed best-response dynamics $^{4}$ , our policy update equation in Algorithm 2 Line 4 is discrete and stochastic. To use $V_{v,s}(\pi^{i}(s),\pi^{-i}(s))$ as our Lyapunov function to study the policy convergence, we need to show that $V_{v,s}(\cdot,\cdot)$ is a smooth function. However, since the entropy $\nu(\cdot)$ is not a smooth function, the function $V_{v,s}(\cdot,\cdot)$ is in general not smooth on the joint simplex $\Delta^{|\mathcal{A}^{i}|}\times\Delta^{|\mathcal{A}^{-i}|}$ .

To overcome this difficulty, recall that we have shown that all the policies from the algorithm trajectory have uniformly lower-bounded entries, with the lower bound being $\ell_{\tau}$ (cf. Lemma A.2). Therefore, it is enough to only consider $V_{v,s}(\cdot,\cdot)$ on the following proper subset

of the joint probability simplex $\Pi_{\ell_{\tau}} := \{\mu = (\mu^{i}, \mu^{-i}) \in \Delta^{|\mathcal{A}^{i}|} \times \Delta^{|\mathcal{A}^{-i}|} \mid \min_{a^{i}} \mu^{i}(a^{i}) > \ell_{\tau}, \min_{a^{-i}} \mu^{-i}(a^{-i}) > \ell_{\tau}\}$ . Since the extreme points of the joint simplex are excluded, we are able to establish the smoothness of $V_{v,s}(\cdot, \cdot)$ on $\Pi_{\ell_{\tau}}$ , which is key in our Lyapunov approach for analyzing the policies. Eventually, we obtain the following Lyapunov drift inequality for $V_{v,s}(\cdot, \cdot)$ :

$$
\begin{array}{l} \sum_ {s} \mathbb {E} [ V _ {v, s} (\pi_ {k + 1} ^ {i} (s), \pi_ {k + 1} ^ {- i} (s)) ] \leq \underbrace {(1 - C _ {1} ^ {\prime} \beta_ {k}) \sum_ {s} \mathbb {E} [ V _ {v , s} (\pi_ {k} ^ {i} (s) , \pi_ {k} ^ {- i} (s)) ]} _ {\text { Negative   Drift }} \\ + \underbrace {C _ {2} ^ {\prime} (\beta_ {k} ^ {2} + \beta_ {k} \mathbb {E} [ \mathcal {L} _ {q} (k) ] + \beta_ {k} \| v ^ {i} + v ^ {- i} \| _ {\infty} ^ {2})} _ {\text { Additive   Errors }}, \tag {11} \\ \end{array}
$$

where $C_{1}^{\prime}$ and $C_{2}^{\prime}$ are (problem-dependent) constants. To interpret the above, suppose that we were considering the continuous-time smoothed best-response dynamics (which is an ODE). Then, the additive error term would disappear in the sense that the time-derivative of the Lyapunov function $V_{v,s}(\cdot)$ along the trajectory of the ODE is strictly negative. Thus, the three terms in the Additive Errors can be interpreted as (1) the discretization error in the update equation, (2) the stochastic error in the q-function estimate, and (3) the error due to the non-zero-sum structure of the inner-loop auxiliary matrix game; see Section 3.1.

# 4.3 Analysis of the Inner Loop: q-Function Update

Our next focus is the q-function update. The q-function update equation is in the same spirit as TD-learning, and a necessary condition for the convergence of TD-learning is that the behavior policy (i.e., the policy used to collect samples) should enable the agent to sufficiently explore the environment. To achieve this goal, since we have shown that all joint policies from the algorithm trajectory have uniformly lower-bounded entries (with lower bound $\ell_{\tau} > 0$ ), it is enough to restrict our attention to a “soft” policy class $\Pi_{\delta} := \{\pi = (\pi^{i}, \pi^{-i}) \mid \min_{s,a^{i}} \pi^{i}(a^{i}|s) > \delta_{i}, \min_{s,a^{-i}} \pi^{-i}(a^{-i}|s) > \delta_{-i}\}$ , where $(\delta_{i}, \delta_{-i})$ represent the margins. The following lemma, which is an extension of (Zhang et al., 2022c, Lemma 4), establishes a uniform exploration property under Assumption 3.1.

To present the result, we need the following notation. Under Assumption 3.1, the Markov chain induced by the joint policy $\pi_b$ has a unique stationary distribution $\mu_b \in \Delta^{|\mathcal{S}|}$ , the minimum component of which is denoted by $\mu_{b,\min}$ . In addition, there exists $\rho_b \in (0,1)$ such that $\max_{s \in \mathcal{S}} \| P_{\pi_b}^k(s,\cdot) - \mu_b(\cdot) \|_{\mathrm{TV}} \leq 2\rho_b^k$ for all $k \geq 0$ (Levin and Peres, 2017), where $P_{\pi_b}$ is the transition probability matrix of the Markov chain $\{S_k\}$ under $\pi_b$ . We also define the mixing time in the following. Given a joint policy $\pi = (\pi^i, \pi^{-i})$ and an accuracy level $\eta > 0$ , the $\eta-$ mixing time of the Markov chain $\{S_k\}$ induced by $\pi$ is defined as

$$
t _ {\pi , \eta} = \min \left\{k \geq 0: \max _ {s \in \mathcal {S}} \| P _ {\pi} ^ {k} (s, \cdot) - \mu_ {\pi} (\cdot) \| _ {\mathrm{TV}} \leq \eta \right\},
$$

where $P_{\pi}$ is the $\pi$ -induced transition probability matrix and $\mu_{\pi}$ is the stationary distribution of $\{S_k\}$ under $\pi$ , provided that it exists and is unique. When the induced Markov chain mixes at a geometric rate, it is easy to see that $t_{\pi,\eta} = \mathcal{O}(\log (1 / \eta))$ .

Lemma 4.1 (An Extension of Lemma 4 in Zhang et al. (2022c)). Suppose that Assumption 3.1 is satisfied. Then we have the following results.

(1) For any $\pi = (\pi^i, \pi^{-i}) \in \Pi_\delta$ , the Markov chain $\{S_k\}$ induced by the joint policy $\pi$ is irreducible and aperiodic, hence admits a unique stationary distribution $\mu_\pi \in \Delta^{|\mathcal{S}|}$ .   
(2) It holds that $\sup_{\pi\in\Pi_{\delta}}\max_{s\in\mathcal{S}}\|P_{\pi}^{k}(s,\cdot)-\mu_{\pi}(\cdot)\|_{TV}\leq2\rho_{\delta}^{k}$ for any $k\geq0$ , where $\rho_{\delta}=\rho_{b}^{(\delta_{i}\delta_{-i})^{r_{b}}}\mu_{b,\min}$ and $r_{b}:=\min\{k\geq0:P_{\pi_{b}}^{k}(s,s^{\prime})>0,\forall(s,s^{\prime})\}$ . As a result, we have

$$
t (\delta , \eta) := \sup _ {\pi \in \Pi_ {\delta}} t _ {\pi , \eta} \leq \frac {t _ {\pi_ {b} , \eta}}{(\delta_ {i} \delta_ {- i}) ^ {r _ {b}} \mu_ {b , \min}}, \tag {12}
$$

where we recall that $t_{\pi, \eta}$ is the $\eta$ -mixing time of the Markov chain $\{S_k\}$ induced by $\pi$ .

(3) Let $G : R^{|S|A_{\max}} \mapsto R^{|S|}$ be the mapping from a policy $\pi \in \Pi_{\delta}$ to the unique stationary distribution $\mu_{\pi}$ of the Markov chain $\{S_k\}$ induced by $\pi$ . Then $G(\cdot)$ is Lipschitz continuous with respect to $\|\cdot\|_{\infty}$ , with Lipschitz constant $\hat{L}_{\delta} := \frac{2 \log(8|S|/\rho_{\delta})}{\log(1/\rho_{\delta})}$ .

(4) $\mu_{\delta} := \inf_{\pi \in \Pi_{\delta}} \min_{s \in S} \mu_{\pi}(s) > 0.$

Remark. Lemma 4.1 (1), (3), and (4) were previous established in (Zhang et al., 2022c, Lemma 4). Lemma 4.1 (2) enables us to see the explicit dependence of the “uniform mixing time” on the margins $\delta_{i}$ , $\delta_{-i}$ and the mixing time of the benchmark exploration policy $\pi_{b}$ .

In view of Lemma 4.1 (2), we have fast mixing for all policies in $\Pi_{\delta}$ if (i) the margins $\delta_i, \delta_{-i}$ are large, and (ii) the Markov chain $\{S_k\}$ induced by the benchmark exploration policy $\pi_b$ is well-behaved. By “well-behaved” we mean the mixing time is small (i.e., small $t_{\pi_b,\eta}$ ) and the stationary distribution is relatively well-balanced (i.e., large $\mu_{b,\min}$ ). Point (i) agrees with our intuition as large margins encourage more exploration. To make sense of point (ii), since $\pi(a|s) \geq \delta_i \delta_{-i} \pi_b(a|s)$ for all $s$ and $a = (a^i, a^{-i})$ , we can write $\pi$ as a convex combination between $\pi_b$ and some residual policy $\tilde{\pi}$ : $\pi = \delta_i \delta_{-i} \pi_b + (1 - \delta_i \delta_{-i}) \tilde{\pi}$ . Therefore, since any $\pi \in \Pi_{\delta}$ has a portion of the benchmark exploration policy $\pi_b$ in it, it makes intuitive sense that fast mixing of $\{S_k\}$ under $\pi_b$ implies, to some extent, fast mixing of $\{S_k\}$ under $\pi \in \Pi_{\delta}$ . Note that, as the margins $\delta_i, \delta_{-i}$ approach zero, the uniform mixing time in Lemma 4.1 (2) goes to infinity. This is not avoidable in general, as demonstrated by a simple MDP example constructed in Appendix D.

When $\Pi_{\delta} = \Pi_{\ell_{\tau}}$ , we denote $\rho_{\tau} := \rho_{\delta}, \mu_{\tau} := \mu_{\delta}$ , and $\hat{L}_{\tau} := \hat{L}_{\delta}$ . We also define $c_{\tau} := \mu_{\tau}\ell_{\tau}$ . With Lemma 4.1 in hand, we are now able to analyze the behavior of the q-functions. We model the q-function update as a stochastic approximation algorithm driven by time-inhomogeneous Markovian noise, and use the norm-square function

$$
\sum_ {i = 1, 2} \sum_ {s} \| q ^ {i} (s) - \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) \| _ {2} ^ {2}
$$

as the Lyapunov function to study its behavior. The key challenge to establishing a Lyapunov drift inequality is to control a difference of the form

$$
\mathbb {E} [ F ^ {i} (q ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) ] - \mathbb {E} [ F ^ {i} (q ^ {i}, \hat {S}, \hat {A} ^ {i}, \hat {A} ^ {- i}, \hat {S} ^ {\prime}) ] \tag {13}
$$

for any $q^{i} \in R^{|S||A^{i}|}$ , where $F^{i}(\cdot)$ is some appropriately defined operator that captures the dynamics of the update equation; see Appendix A.5 for its definition. In the term (13), the random tuple $(S_{k}, A_{k}^{i}, A_{k}^{-i}, S_{k+1})$ is the k-th sample from the time-inhomogeneous Markov chain $\{(S_{n}, A_{n}^{i}, A_{n}^{-i}, S_{n+1})\}_{n \geq 0}$ generated by the time-varying joint policies $\{\pi_{n}\}_{n \geq 0}$ , and $(\hat{S}, \hat{A}^{i}, \hat{A}^{-i}, \hat{S}')$ is a random tuple such that $S \sim \mu_{k}(\cdot)$ , $A^{i} \sim \pi_{k}^{i}$ , $A^{-i} \sim \pi_{k}^{-i}$ , and $S' \sim p(\cdot | S, A^{i}, A^{-i})$ , where $\mu_{k}(\cdot)$ is the unique stationary distribution of the Markov chain $\{S_{n}\}$ induced by the joint policy $\pi_{k}$ . Due to Lemma 4.1, $\mu_{k}$ exists and is unique.

In existing literature, when $\{(S_{k}, A_{k}^{i}, A_{k}^{-i}, S_{k+1})\}$ is sampled either in an i.i.d. manner or forms an ergodic time-homogeneous Markov chain, there are techniques that successfully handle (13) (Bertsekas and Tsitsiklis, 1996; Srikant and Ying, 2019; Bhandari et al., 2018). To deal with time-inhomogeneous Markovian noise, building upon existing conditioning results (Bhandari et al., 2018; Srikant and Ying, 2019; Zou et al., 2019; Khodadadian et al., 2022) and also Lemma 4.1, we develop a refined conditioning argument to show that

$$
(1 3) = \mathcal {O} \left(z _ {k} \sum_ {n = k - z _ {k}} ^ {k - 1} \beta_ {n}\right),
$$

where $z_{k} = t(\ell_{\tau}, \beta_{k})$ is a uniform upper bound on the $\beta_{k}$ – the mixing time (i.e., the uniform mixing time with accuracy $\beta_{k}$ , see Eq. (12)) of the Markov chain $\{S_{n}\}$ induced by an

arbitrary joint policy from the algorithm trajectory. Suppose we are using diminishing step-sizes $\beta_{k} = \beta/(k + h)$ (similar results hold for constant stepsize). Then, the uniform mixing property from Lemma 4.1 (2) implies that $z_{k} = \mathcal{O}(\log(1/k))$ . As a result, we have $\lim_{k \to \infty}(13) \leq \lim_{k \to \infty} z_{k} \sum_{n=k-z_{k}}^{k-1} \beta_{n} = 0$ , which provides us a way to control (13). After successfully handling (13), we are able to establish a Lyapunov drift inequality of the following form:

$$
\begin{array}{l} \sum_ {i = 1, 2} \mathbb {E} [ \| q _ {k + 1} ^ {i} - \bar {q} _ {k + 1} ^ {i} \| _ {2} ^ {2} ] \leq \underbrace {(1 - C _ {1} ^ {\prime \prime} \alpha_ {k}) \sum_ {i = 1 , 2} \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ]} _ {\text { Negative   Drift }} \\ + \underbrace {C _ {2} ^ {\prime \prime} (\alpha_ {k} ^ {2} + \beta_ {k} \sum_ {s} \mathbb {E} [ V _ {v , s} (\pi_ {k} ^ {i} (s) , \pi_ {k} ^ {- i} (s)) ])} _ {\text { Additive   Error }} \tag {14} \\ \end{array}
$$

where $C_{1}^{\prime\prime}$ and $C_{2}^{\prime\prime}$ are (problem-dependent) constants, $\overline{q}_{k}^{i}(s):=\mathcal{T}^{i}(v^{i})(s)\pi_{k}^{-i}(s)$ for all $s\in S$ , and $V_{v,s}(\cdot,\cdot)$ is the Lyapunov function we used to study the policy convergence.

# 4.4 Solving Coupled Lyapunov Drift Inequalities

Until this point, we have established the Lyapunov drift inequalities for the individual v-functions, the sum of the v-functions, the policies, and the q-functions in Eqs. (8), (9), (11), and (14), respectively. The last challenge is to find a strategic way of using these coupled inequalities to derive the finite-sample bound. To elaborate, we first restate all the Lyapunov drift inequalities in the following. For simplicity of notation, we denote

$$
\mathcal {L} _ {v} (t) = \sum_ {i = 1, 2} \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty}, \quad \mathcal {L} _ {\text { sum }} (t) = \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty},
$$

$$
\mathcal {L} _ {\pi} (t, k) = \sum_ {s} V _ {v _ {t}, s} (\pi_ {t, k} ^ {i} (s), \pi_ {t, k} ^ {- i} (s)), \text { and } \mathcal {L} _ {q} (t, k) = \sum_ {i = 1, 2} \sum_ {s} \| q _ {t, k} ^ {i} (s) - \bar {q} _ {t, k} ^ {i} (s) \| _ {2} ^ {2}. \tag {15}
$$

Then Eqs. (8), (9), (11), and (14) can be compactly written as

$$
\mathcal {L} _ {v} (t + 1) \leq \gamma \mathcal {L} _ {v} (t) + C _ {1} (\mathcal {L} _ {\text { sum }} (t) + \mathcal {L} _ {\pi} (t, K) + \mathcal {L} _ {q} ^ {1 / 2} (t, K) + \tau \log (A _ {\max})), \tag {16}
$$

$$
\mathcal {L} _ {\text { sum }} (t + 1) \leq \gamma \mathcal {L} _ {\text { sum }} (t) + C _ {2} \mathcal {L} _ {q} ^ {1 / 2} (t, K), \tag {17}
$$

$$
\mathbb {E} _ {t} \left[ \mathcal {L} _ {\pi} (t, k + 1) \right] \leq \left(1 - C _ {1} ^ {\prime} \beta_ {k}\right) \mathbb {E} _ {t} \left[ \mathcal {L} _ {\pi} (t, k) \right] + C _ {2} ^ {\prime} \left(\beta_ {k} ^ {2} + \beta_ {k} \mathbb {E} _ {t} \left[ \mathcal {L} _ {q} (t, k) \right] + \beta_ {k} \mathcal {L} _ {\text { sum }} ^ {2} (t)\right), \tag {18}
$$

$$
\mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k + 1) ] \leq (1 - C _ {1} ^ {\prime \prime} \alpha_ {k}) \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k) ] + C _ {2} ^ {\prime \prime} (\alpha_ {k} ^ {2} + \beta_ {k} \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) ]). \tag {19}
$$

where $E_{t}[\cdot]$ stands for conditional expectation conditioned on the history up to the beginning of the t-th outer loop.

A Vanilla Approach. Recall that we have shown that the iterates $\{v_{t}^{i}\}$ and $\{q_{t,k}^{i}\}$ are uniformly bounded (cf. Lemma A.1). As a result, all the Lyapunov functions $\mathcal{L}_{v}(\cdot)$ , $\mathcal{L}_{\mathrm{sum}}(\cdot)$ , $\mathcal{L}_{\pi}(\cdot)$ , and $\mathcal{L}_{q}(\cdot)$ are uniformly bounded too, which provides us a handle to decouple the inequalities. As a clear example, observe that

$$
\begin{array}{l} \mathcal {L} _ {\pi} (t, k) = \sum_ {s} V _ {v _ {t}, s} (\pi_ {t, k} ^ {i} (s), \pi_ {t, k} ^ {- i} (s)) \\ = \sum_ {s} \sum_ {i = 1, 2} \max _ {\hat {\mu} ^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \{(\hat {\mu} ^ {i} - \pi_ {t, k} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, k} ^ {- i} (s) + \tau \nu (\hat {\mu} ^ {i}) - \tau \nu (\pi_ {t, k} ^ {i} (s)) \} \\ \leq \sum_ {s} \sum_ {i = 1, 2} \left(2 \max _ {s, a ^ {i}, a ^ {- i}} | \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s, a ^ {i}, a ^ {- i}) | + \tau \log (A _ {\max})\right) \\ \end{array}
$$

$$
\leq \sum_ {s} \sum_ {i = 1, 2} (2 + 2 \gamma \| v _ {t} ^ {i} \| _ {\infty} + \tau \log (A _ {\max})) \quad (\text { Definition   of } \mathcal {T} ^ {i} (\cdot))
$$

$$
\leq 4 | \mathcal {S} | \left(\frac {1}{1 - \gamma} + \tau \log (A _ {\max})\right) := L _ {\text { bound }},
$$

where the last line follows from the boundedness of the v-functions (cf. Lemma A.1). Therefore, we can replace $\mathcal{L}_{\pi}(t,k)$ in Eq. (19) by its uniform upper bound established in the previous inequality to obtain

$$
\mathbb {E} _ {t} \left[ \mathcal {L} _ {q} (t, k + 1) \right] \leq \left(1 - C _ {1} ^ {\prime \prime} \alpha_ {k}\right) \mathbb {E} _ {t} \left[ \mathcal {L} _ {q} (t, k) \right] + C _ {2} ^ {\prime \prime} \alpha_ {k} ^ {2} + C _ {2} ^ {\prime \prime} L _ {\text {bound}} \beta_ {k}. \tag {20}
$$

Note that Eq. (20) is now a decoupled Lyapunov drift inequality solely for $\mathcal{L}_{q}(\cdot)$ , which can be repeatedly used to derive a finite-sample bound for $\mathbb{E}_{t}[\mathcal{L}_{q}(\cdot)]$ . In particular, when using $\alpha_{k} = \alpha/(k + h)$ with properly chosen $\alpha$ and h, we have

$$
\mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, K) ] \leq \mathcal {O} \left(\frac {1}{K + h}\right) + \mathcal {O} \left(c _ {\alpha , \beta}\right). \tag {21}
$$

With the same decoupling technique, we can establish finite-sample bounds of $\mathcal{L}_{v}(\cdot)$ , $\mathcal{L}_{\mathrm{sum}}(\cdot)$ , and $\mathcal{L}_{\pi}(\cdot)$ . However, in view of Eq. (21), even when using diminishing stepsizes, due to the presence of $\mathcal{O}(c_{\alpha,\beta})$ , we cannot make $\mathbb{E}_{t}[\mathcal{L}_{q}(t,K)]$ arbitrarily small by just increasing the iteration number K. In other words, to make $\mathbb{E}_{t}[\mathcal{L}_{q}(t,K)]$ arbitrarily small, it is necessary to use a diminishing stepsize ratio, which implies $\beta_{k}=o(\alpha_{k})$ and hence making Algorithm 2 a two time-scale learning dynamics. The fact that $\beta_{k}$ has to be order-wise smaller than $\alpha_{k}$ will also undermine the convergence rate. Specifically, with this approach (i.e., using the uniform upper bounds to decouple the Lyapunov drift inequalities and enforcing convergence by introducing another time-scale), the overall sample complexity will be order-wise larger than $\tilde{\mathcal{O}}(\epsilon^{-2})$ . This is not surprising as we essentially use constants (i.e., the uniform upper bounds) to bound quantities that are actually converging to zero.

In general, we observe from existing literature that once an iterative algorithm has multiple time scales, oftentimes the convergence rate is downgraded (Khodadadian et al., 2022; Zhang et al., 2022b).

Our Decoupling Approach. To establish a sharper rate without introducing another time-scale, the high-level ideas are (1) using the Lyapunov drift inequalities in a combined way instead of in a separate manner, and (2) a bootstrapping procedure where we first derive a crude bound and then substitute the crude bound back into the Lyapunov drift inequalities to derive a tighter bound. We next present our approach.

For simplicity of notation, for a scalar-valued quantity W that is a function of k and/or t, we say $W = o_{k}(1)$ if $\lim_{k \to \infty} W = 0$ and $W = o_{t}(1)$ if $\lim_{t \to \infty} W = 0$ . The explicit convergence rates of the $o_{k}(1)$ term and the $o_{t}(1)$ term will be revealed in the complete proof in Appendix A.6, but is not important for the illustration here.

Step 1. Adding up Eq. (18) and (19) and then repeatedly using the resulting inequality, and we obtain:

$$
\mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) ] \leq \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) + \mathcal {L} _ {q} (t, k) ] = o _ {k} (1) + \mathcal {O} (1) \mathcal {L} _ {\text { sum }} ^ {2} (t), \forall t, k. \tag {22}
$$

Step 2. Substituting the bound for $\mathbb{E}_t[\mathcal{L}_\pi (t,k)]$ in Eq. (22) into Eq. (19) and repeatedly using the resulting inequality, and we obtain:

$$
\mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, K) ] = o _ {K} (1) + \mathcal {O} (c _ {\alpha , \beta}) \mathcal {L} _ {\mathrm{sum}} ^ {2} (t), \forall t,
$$

which in turn implies (by first using Jensen's inequality and then taking total expectation) that:

$$
\mathbb {E} [ \mathcal {L} _ {q} ^ {1 / 2} (t, K) ] = o _ {K} (1) + \mathcal {O} (c _ {\alpha , \beta} ^ {1 / 2}) \mathbb {E} [ \mathcal {L} _ {\text { sum }} (t) ], \forall t, \tag {23}
$$

where we recall that $c_{\alpha, \beta} = \beta_k / \alpha_k$ is the stepsize ratio. The fact that we are able to get a factor of $\mathcal{O}(c_{\alpha, \beta}^{1/2})$ in front of $\mathbb{E}[\mathcal{L}_{\mathrm{sum}}(t)]$ is crucial for the decoupling procedure.

Step 3. Taking total expectation on both sides of Eq. (17) and then using the upper bound of $\mathbb{E}[\mathcal{L}_q^{1/2}(t,K)]$ we obtained in Eq. (23), and we obtain

$$
\mathbb {E} \left[ \mathcal {L} _ {\text { sum }} (t + 1) \right] \leq (\gamma + \mathcal {O} \left(c _ {\alpha , \beta} ^ {1 / 2}\right)) \mathbb {E} \left[ \mathcal {L} _ {\text { sum }} (t) \right] + o _ {K} (1), \forall t.
$$

By choosing $c_{\alpha, \beta}$ so that $\mathcal{O}(c_{\alpha, \beta}^{1/2}) \leq (1 - \gamma)/2$ , the previous inequality implies

$$
\mathbb {E} [ \mathcal {L} _ {\text { sum }} (t + 1) ] \leq \left(1 - \frac {1 - \gamma}{2}\right) \mathbb {E} [ \mathcal {L} _ {\text { sum }} (t) ] + o _ {K} (1), \forall t, \tag {24}
$$

which can be repeatedly used to obtain

$$
\mathbb {E} [ \mathcal {L} _ {\text { sum }} (t) ] = o _ {t} (1) + o _ {K} (1). \tag {25}
$$

Substituting the previous bound on $\mathbb{E}[\mathcal{L}_{\mathrm{sum}}(t)]$ into Eq. (22) and we have

$$
\max (\mathbb {E} [ \mathcal {L} _ {\pi} (t, K) ], \mathbb {E} [ \mathcal {L} _ {q} (t, K) ]) = o _ {t} (1) + o _ {K} (1). \tag {26}
$$

Step 4. Substituting the bounds we obtained for $\mathbb{E}[\mathcal{L}_{\pi}(t,K)]$ , $\mathbb{E}[\mathcal{L}_{q}(t,K)]$ , and $\mathbb{E}[\mathcal{L}_{\mathrm{sum}}(t)]$ in Eqs. (25), and (26) into Eq. (16), and then repeatedly using the resulting inequality from t=0 to t=T, we have

$$
\mathbb {E} [ \mathcal {L} _ {v} (T) ] = o _ {T} (1) + o _ {K} (1) + \mathcal {O} (\tau).
$$

Now that we have obtained finite-sample bounds for $\mathbb{E}[\mathcal{L}_v(T)]$ , $\mathbb{E}[\mathcal{L}_{\mathrm{sum}}(T)]$ , $\mathbb{E}[\mathcal{L}_{\pi}(T,K)]$ , and $\mathbb{E}[\mathcal{L}_q(T,K)]$ , using them in Eq. (7) and we finally obtain the desired finite-sample bound for the expected Nash gap.

Looking back at the decoupling procedure, Steps 2 and 3 are crucial. In fact, in Step 1 we already obtain a bound on $\mathbb{E}_t[\mathcal{L}_q(t,k)]$ , where the additive error is $\mathcal{O}(1)\mathbb{E}[\mathcal{L}_{\mathrm{sum}}(t)]$ . However, directly using this bound on $\mathbb{E}_t[\mathcal{L}_q(t,k)]$ in Eq. (17) would result in an expansive inequality for $\mathbb{E}[\mathcal{L}_{\mathrm{sum}}(t)]$ . By performing Step 2, we are able to obtain a tighter bound for $\mathbb{E}_t[\mathcal{L}_q(t,k)]$ , with the additive error being $\mathcal{O}(\sqrt{c_{\alpha,\beta}})\mathbb{E}[\mathcal{L}_{\mathrm{sum}}(t)]$ . Furthermore, we can choose $c_{\alpha,\beta}$ so that after using the bound from Eq. (23) in Eq. (17), the additive error $\mathcal{O}(\sqrt{c_{\alpha,\beta}})\mathbb{E}[\mathcal{L}_{\mathrm{sum}}(t)]$ is dominated by the negative drift in Eq. (24).

# 5 Conclusion

In this work, we consider solving zero-sum matrix games and Markov games with independent learning dynamics. In both settings, we design learning dynamics that are payoff-based, convergent, and rational. In addition, both learning dynamics are intuitive and natural to implement. Our main results provide finite-sample bounds on both learning dynamics, establishing an $\tilde{\mathcal{O}}(1/\epsilon)$ sample complexity in the matrix game setting and an $\tilde{\mathcal{O}}(1/\epsilon^{2})$ sample complexity in the Markov game setting. Our analysis provides a number of new tools that are likely to be of interest more broadly, such as our strategy to handle coupled Lyapunov drift inequalities.

As mentioned in Section 3.2, an immediate future direction is to investigate using a time-varying temperature $\tau_{k}$ and establish a sharp rate of convergence with an asymptotically vanishing smoothing bias. In long term, we are interested to see if the algorithmic ideas and the analysis techniques developed in this work can be used to study other classes of games beyond zero-sum stochastic games.

# References

Alacaoglu, A., Viano, L., He, N., and Cevher, V. (2022). A natural actor-critic framework for zero-sum Markov games. In International Conference on Machine Learning, pages 307–366. PMLR.   
Arslan, G. and Yüksel, S. (2017). Decentralized Q-Learning for Stochastic Teams and Games. IEEE Transactions on Automatic Control, 62(4):1545–1558.   
Bai, Y. and Jin, C. (2020). Provable self-play algorithms for competitive reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning (ICML).   
Bai, Y., Jin, C., and Yu, T. (2020). Near-optimal reinforcement learning with self-play. Advances in Neural Information Processing Systems, 33.   
Baudin, L. and Laraki, R. (2022a). Fictitious play and best-response dynamics in identical interest and zero-sum stochastic games. In International Conference on Machine Learning, pages 1664–1690. PMLR.   
Baudin, L. and Laraki, R. (2022b). Smooth Fictitious Play in Stochastic Games with Perturbed Payoffs and Unknown Transitions. In Advances in Neural Information Processing Systems.   
Beck, A. (2017). First-order methods in optimization, volume 25. SIAM.   
Bertsekas, D. P. and Tsitsiklis, J. N. (1996). Neuro-dynamic programming. Athena Scientific.   
Beznosikov, A., Gorbunov, E., Berard, H., and Loizou, N. (2022). Stochastic gradient descent-ascent: Unified theory and new efficient methods. Preprint arXiv:2202.07262.   
Bhandari, J., Russo, D., and Singal, R. (2018). A Finite Time Analysis of Temporal Difference Learning With Linear Function Approximation. In Conference On Learning Theory, pages 1691–1692.   
Bowling, M. and Veloso, M. (2001). Rational and convergent learning in stochastic games. In International Joint Conference on Artificial Intelligence, volume 17, pages 1021–1026.   
Brown, G. W. (1951). Iterative solution of games by fictitious play. Activity Analysis of Production and Allocation, 13(1):374–376.   
Busoniu, L., Babuska, R., De Schutter, B., et al. (2008). A comprehensive survey of multiagent reinforcement learning. IEEE Transactions on Systems, Man, and Cybernetics, Part C, 38(2):156–172.   
Cen, S., Chi, Y., Du, S. S., and Xiao, L. (2022). Faster last-iterate convergence of policy optimization in zero-sum Markov games. Preprint arXiv:2210.01050.   
Cen, S., Wei, Y., and Chi, Y. (2021). Fast policy extragradient methods for competitive games with entropy regularization. Advances in Neural Information Processing Systems, 34:27952–27964.   
Cesa-Bianchi, N. and Lugosi, G. (2006). Prediction, Learning, and Games. Cambridge University Press.   
Chen, Z., Ma, S., and Zhou, Y. (2021a). Sample efficient stochastic policy extragradient algorithm for zero-sum markov game. In International Conference on Learning Representations.   
Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2020). Finite-Sample Analysis of Contractive Stochastic Approximation Using Smooth Convex Envelopes. Advances in Neural Information Processing Systems, 33.   
Chen, Z., Maguluri, S. T., Shakkottai, S., and Shanmugam, K. (2021b). A Lyapunov Theory for Finite-Sample Guarantees of Asynchronous Q-Learning and TD-Learning Variants. Preprint arXiv:2102.01567.   
Chen, Z., Zhou, Y., Chen, R.-R., and Zou, S. (2022). Sample and communication-efficient decentralized actor-critic algorithms with finite-time analysis. In International Conference on Machine Learning, pages 3794–3834. PMLR.   
Cui, Q. and Du, S. S. (2022a). Provably Efficient Offline Multi-agent Reinforcement Learning via Strategy-wise Bonus. In Advances in Neural Information Processing Systems.

Cui, Q. and Du, S. S. (2022b). When are offline two-player zero-sum Markov games solvable? In Advances in Neural Information Processing Systems.   
Cui, Q., Zhang, K., and Du, S. S. (2023). Breaking the Curse of Multiagents in a Large State Space: RL in Markov Games with Independent Linear Function Approximation. Preprint arXiv:2302.03673.   
Daskalakis, C., Foster, D. J., and Golowich, N. (2020). Independent policy gradient methods for competitive reinforcement learning. Advances in neural information processing systems, 33:5527–5540.   
Daskalakis, C., Golowich, N., and Zhang, K. (2022). The complexity of Markov equilibrium in stochastic games. Preprint arXiv:2204.03991.   
Degrave, J., Felici, F., Buchli, J., Neunert, M., Tracey, B., Carpanese, F., Ewalds, T., Hafner, R., Abdolmaleki, A., de Las Casas, D., et al. (2022). Magnetic control of tokamak plasmas through deep reinforcement learning. Nature, 602(7897):414–419.   
Ding, D., Wei, C.-Y., Zhang, K., and Jovanovic, M. (2022). Independent policy gradient for large-scale markov potential games: Sharper rates, function approximation, and game-agnostic convergence. In International Conference on Machine Learning, pages 5166–5220. PMLR.   
Erez, L., Lancewicki, T., Sherman, U., Koren, T., and Mansour, Y. (2022). Regret minimization and convergence to equilibria in general-sum Markov games. Preprint arXiv:2207.14211.   
Even-Dar, E. and Mansour, Y. (2003). Learning rates for Q-learning. Journal of Machine Learning Research, 5(Dec):1–25.   
Fudenberg, D. and Kreps, D. (1993). Learning mixed equilibria. Games and Economic Behavior, 5:320–367.   
Fudenberg, D. and Levine, D. K. (1995). Consistency and cautious fictitious play. Journal of Economic Dynamics and Control, 19(5-7):1065–1089.   
Fudenberg, D. and Levine, D. K. (1998). The Theory of Learning in Games, volume 2. MIT press.   
Gao, B. and Pavel, L. (2017). On the properties of the softmax function with application in game theory and reinforcement learning. Preprint arXiv:1704.00805.   
Gao, Z., Ma, Q., Başar, T., and Birge, J. R. (2021). Finite-Sample Analysis of Decentralized Q-Learning for Stochastic Games. Preprint arXiv:2112.07859.   
Harris, C. (1998). On the rate of convergence of continuous-time fictitious play. Games and Economic Behavior, 22(2):238–259.   
Hofbauer, J. and Sandholm, W. H. (2002). On the global convergence of stochastic fictitious play. Econometrica, 70(6):2265–2294.   
Hofbauer, J. and Sorin, S. (2006). Best response dynamics for continuous zero-sum games. Discrete and Continuous Dynamical Systems Series B, 6(1):215.   
Hu, J. and Wellman, M. P. (2003). Nash Q-learning for general-sum stochastic games. Journal of Machine Learning Research, 4(Nov):1039–1069.   
Jin, C., Liu, Q., Wang, Y., and Yu, T. (2021). V-learning – A simple, efficient, decentralized algorithm for multiagent RL. Preprint arXiv:2110.14555.   
Khodadadian, S., Doan, T. T., Romberg, J., and Maguluri, S. T. (2022). Finite sample analysis of two-time-scale natural actor-critic algorithm. IEEE Transactions on Automatic Control.   
Lan, G. (2020). First-order and Stochastic Optimization Methods for Machine Learning. Springer.   
Lan, G. (2022). Policy mirror descent for reinforcement learning: Linear convergence, new sampling complexity, and generalized problem classes. Mathematical programming, pages 1–48.   
Leonardos, S., Overman, W., Panageas, I., and Piliouras, G. (2022). Global convergence of multi-agent policy gradient in Markov potential games. In International Conference on Learning Representations.

Leslie, D. S. and Collins, E. J. (2005). Individual Q-learning in normal form games. SIAM Journal on Control and Optimization, 44(2):495–514.   
Leslie, D. S., Perkins, S., and Xu, Z. (2020). Best-response dynamics in zero-sum stochastic games. Journal of Economic Theory, 189:105095.   
Levin, D. A. and Peres, Y. (2017). Markov chains and mixing times, volume 107. American Mathematical Soc.   
Li, G., Chi, Y., Wei, Y., and Chen, Y. (2022). Minimax-optimal multi-agent RL in Markov games with a generative model. In Advances in Neural Information Processing Systems.   
Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2020). Sample Complexity of Asynchronous Q-Learning: Sharper Analysis and Variance Reduction. In Advances in Neural Information Processing Systems, volume 33, pages 7031–7043. Curran Associates, Inc.   
Lin, T., Zhou, Z., Ba, W., and Zhang, J. (2021). Optimal no-regret learning in strongly monotone games with bandit feedback. Preprint arXiv:2112.02856.   
Littman, M. L. (1994). Markov games as a framework for multi-agent reinforcement learning. In Proceedings of the Eleventh International Conference on International Conference on Machine Learning, pages 157–163.   
Littman, M. L. (2001). Friend-or-foe Q-learning in general-sum games. In International Conference on Machine Learning, volume 1, pages 322–328.   
Liu, Q., Yu, T., Bai, Y., and Jin, C. (2021). A sharp analysis of model-based reinforcement learning with self-play. In International Conference on Machine Learning, pages 7001–7010. PMLR.   
Maheshwari, C., Wu, M., Pai, D., and Sastry, S. (2022). Independent and decentralized learning in markov potential games. Preprint arXiv:2205.14590.   
Mao, W., Yang, L., Zhang, K., and Başar, T. (2022). On improving model-free algorithms for decentralized multi-agent reinforcement learning. In International Conference on Machine Learning, pages 15007–15049. PMLR.   
Mirowski, P., Grimes, M., Malinowski, M., Hermann, K. M., Anderson, K., Teplyashin, D., Simonyan, K., Zisserman, A., Hadsell, R., et al. (2018). Learning to navigate in cities without a map. Advances in neural information processing systems, 31.   
Pattathil, S., Zhang, K., and Ozdaglar, A. (2022). Symmetric (optimistic) natural policy gradient for multi-agent learning with parameter convergence. Preprint arXiv:2210.12812.   
Qiu, S., Yang, Z., Ye, J., and Wang, Z. (2021). On finite-time convergence of actor-critic algorithm. IEEE Journal on Selected Areas in Information Theory, 2(2):652–664.   
Qu, G. and Wierman, A. (2020). Finite-Time Analysis of Asynchronous Stochastic Approximation and Q-Learning. In Conference on Learning Theory, pages 3185–3205. PMLR.   
Qu, G., Wierman, A., and Li, N. (2020). Scalable reinforcement learning of localized policies for multi-agent networked systems. In Learning for Dynamics and Control, pages 256–266. PMLR.   
Robinson, J. (1951). An iterative method of solving a game. Annals of Mathematics, pages 296–301.   
Sayin, M., Zhang, K., Leslie, D., Basar, T., and Ozdaglar, A. (2021). Decentralized Q-learning in zero-sum Markov games. Advances in Neural Information Processing Systems, 34:18320–18334.   
Sayin, M. O., Parise, F., and Ozdaglar, A. (2022a). Fictitious play in zero-sum stochastic games. SIAM Journal on Control and Optimization, 60(4):2095–2114.   
Sayin, M. O., Zhang, K., and Ozdaglar, A. (2022b). Fictitious Play in Markov Games with Single Controller. In Proceedings of the 23rd ACM Conference on Economics and Computation, pages 919–936.

Shalev-Shwartz, S., Shammah, S., and Shashua, A. (2016). Safe, multi-agent, reinforcement learning for autonomous driving. Preprint arXiv:1610.03295.   
Shapley, L. S. (1953). Stochastic games. Proceedings of the National Academy of Sciences, 39(10):1095–1100.   
Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. (2017). Mastering the game of go without human knowledge. Nature, 550(7676):354.   
Song, Z., Mei, S., and Bai, Y. (2022). When can we learn general-sum markov games with a large number of players sample-efficiently? In International Conference on Learning Representations.   
Srikant, R. and Ying, L. (2019). Finite-Time Error Bounds For Linear Stochastic Approximation and TD Learning. In Conference on Learning Theory, pages 2803–2830.   
Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. Machine learning, 3(1):9–44.   
Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.   
Tsitsiklis, J. N. and Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. IEEE transactions on automatic control, 42(5):674–690.   
Wei, C.-Y., Hong, Y.-T., and Lu, C.-J. (2017). Online reinforcement learning in stochastic games. In Advances in Neural Information Processing Systems, pages 4987–4997.   
Wei, C.-Y., Lee, C.-W., Zhang, M., and Luo, H. (2021). Last-iterate convergence of decentralized optimistic gradient descent/ascent in infinite-horizon competitive Markov games. In Conference on Learning Theory, pages 4259–4299. PMLR.   
Wu, Y. F., Zhang, W., Xu, P., and Gu, Q. (2020). A finite-time analysis of two time-scale actor-critic methods. Advances in Neural Information Processing Systems, 33:17617–17628.   
Xie, Q., Chen, Y., Wang, Z., and Yang, Z. (2020). Learning zero-sum simultaneous-move Markov games using function approximation and correlated equilibrium. In Conference on Learning Theory, pages 3674-3682. PMLR.   
Xu, T. and Liang, Y. (2021). Sample complexity bounds for two timescale value-based reinforcement learning algorithms. In International Conference on Artificial Intelligence and Statistics, pages 811–819. PMLR.   
Yan, Y., Li, G., Chen, Y., and Fan, J. (2022). The efficacy of pessimism in asynchronous Q-learning. Preprint arXiv:2203.07368.   
Zeng, S., Doan, T. T., and Romberg, J. (2022). Regularized gradient descent ascent for two-player zero-sum Markov games. In Advances in Neural Information Processing Systems.   
Zhang, K., Kakade, S., Başar, T., and Yang, L. (2020). Model-based multi-agent RL in zero-sum Markov games with near-optimal sample complexity. Advances in Neural Information Processing Systems, 33:1166–1178.   
Zhang, K., Yang, Z., and Başar, T. (2021a). Multi-agent reinforcement learning: A selective overview of theories and algorithms. Handbook of Reinforcement Learning and Control, pages 321–384.   
Zhang, K., Yang, Z., Liu, H., Zhang, T., and Başar, T. (2018). Fully decentralized multi-agent reinforcement learning with networked agents. In International Conference on Machine Learning, pages 5867–5876.   
Zhang, K., Zhang, X., Hu, B., and Başar, T. (2021b). Derivative-free policy optimization for linear risk-sensitive and robust control design: Implicit regularization and sample complexity. Advances in Neural Information Processing Systems, 34:2949–2964.   
Zhang, R., Liu, Q., Wang, H., Xiong, C., Li, N., and Bai, Y. (2022a). Policy Optimization for Markov Games: Unified Framework and Faster Convergence. In Advances in Neural Information Processing Systems.

Zhang, R., Ren, Z., and Li, N. (2021c). Gradient play in stochastic games: Stationary points, convergence, and sample complexity. Preprint arXiv:2106.00198.   
Zhang, S., Tachet, R., and Laroche, R. (2022b). Global Optimality and Finite Sample Analysis of Softmax Off-Policy Actor Critic under State Distribution Mismatch. Journal of Machine Learning Research, 23(343):1–91.   
Zhang, Y., Qu, G., Xu, P., Lin, Y., Chen, Z., and Wierman, A. (2022c). Global Convergence of Localized Policy Iteration in Networked Multi-Agent Reinforcement Learning. Preprint arXiv:2211.17116.   
Zhao, Y., Tian, Y., Lee, J. D., and Du, S. S. (2021). Provably Efficient Policy Optimization for Two-Player Zero-Sum Markov Games. Preprint arXiv:2102.08903.   
Zhong, H., Xiong, W., Tan, J., Wang, L., Zhang, T., Wang, Z., and Yang, Z. (2022). Pessimistic minimax value iteration: Provably efficient equilibrium learning from offline datasets. In International Conference on Machine Learning, pages 27117–27142. PMLR.   
Zou, S., Xu, T., and Liang, Y. (2019). Finite-sample analysis for SARSA with linear function approximation. In Advances in Neural Information Processing Systems, pages 8668–8678.

# Appendices

# A Proof of Theorem 3.1 and Theorem 3.2

We first explicitly state the requirement for choosing the stepsizes. For simplicity of notation, given $k_{1} \leq k_{2}$ , we denote $\beta_{k_{1},k_{2}} = \sum_{k=k_{1}}^{k_{2}} \beta_{k}$ and $\alpha_{k_{1},k_{2}} = \sum_{k=k_{1}}^{k_{2}} \alpha_{k}$ . For any $k \geq 0$ , let $z_{k} = t(\ell_{\tau}, \beta_{k})$ , where $t(\cdot, \cdot)$ is the uniform mixing time defined in Lemma 4.1 (2), and $\ell_{\tau}$ is the uniform lower bound of the policies derived in Lemma A.2. When using constant stepsize, $z_{k}$ is not a function of k, and is simply denoted by $z_{\beta}$ . Observe that $z_{k} = \mathcal{O}(\log(k))$ (when using diminishing stepsizes) and $z_{\beta} = \mathcal{O}(\log(1/\beta))$ due to the uniform geometric mixing property established in Lemma 4.1 (2).

Condition A.1. It holds that $\alpha_{k-z_{k},k-1} \leq 1/4$ for all $k \geq z_{k}$ and $c_{\alpha,\beta} \leq \frac{c_{\tau}\ell_{\tau}^{2}\tau^{3}(1-\gamma)^{2}}{512|S|A_{\max}^{2}}$ . When using diminishing stepsizes $\alpha_{k} = \frac{\alpha}{k+h}$ and $\beta_{k} = \frac{\beta}{k+h}$ , we additionally require $\beta > 2$ .

Condition A.1 is easy to satisfy as (1) $z_{k} = \mathcal{O}(\log(1/k))$ while $\alpha_{k} = \mathcal{O}(1/k)$ when using diminishing stepsizes and (2) $z_{k} = \mathcal{O}(\log(1/\alpha))$ and $\alpha_{k} = \alpha$ when using constant stepsize. The parameter $k_{0}$ is defined to be $\min\{k \geq 0 \mid k \geq z_{k}\}$ . Note that $k_{0} = z_{\beta}$ when using constant stepsize.

# A.1 Notation

We begin with a summary of some notation that is used in the proof.

(1) Given a pair of matrices $X_{i} \in \mathbb{R}^{|\mathcal{A}^{i}| \times |\mathcal{A}^{-i}|}$ , $X_{-i} \in \mathbb{R}^{|\mathcal{A}^{-i}| \times |\mathcal{A}^{i}|}$ and a pair of distributions $\mu^{i} \in \Delta^{|\mathcal{A}^{i}|}, \mu^{-i} \in \Delta^{|\mathcal{A}^{-i}|}$ , we define

$$
V _ {X} \left(\mu^ {i}, \mu^ {- i}\right) = \sum_ {i = 1, 2} \max _ {\hat {\mu} ^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \left\{\left(\hat {\mu} ^ {i} - \mu^ {i}\right) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu \left(\hat {\mu} ^ {i}\right) - \tau \nu \left(\mu^ {i}\right) \right\}, \tag {27}
$$

where $\nu (\mu^i) = -\sum_{a^i}\mu^i (a^i)\log (\mu^i (a^i))$ is the entropy function.

(2) Given a pair of $v$ -functions $(v^i, v^{-i})$ and a state $s \in S$ , when $X_i = \mathcal{T}^i(v^i)(s)$ and $X_{-i} = \mathcal{T}^{-i}(v^{-i})(s)$ , we write $V_{v,s}(\cdot, \cdot)$ for $V_X(\cdot, \cdot)$ .

(3) For any $(\pi^i, \pi^{-i})$ and $s$ , define $v_{*,\pi^{-i}}^i(s) = \max_{\hat{\pi}^i} v_{\hat{\pi}^i, \pi^{-i}}^i(s)$ , $v_{\pi^i,*}^i = \min_{\hat{\pi}^{-i}} v_{\pi^i, \hat{\pi}^{-i}}^i(s)$ , $v_{\pi^{-i},*}^{-i}(s) = \min_{\hat{\pi}^i} v_{\pi^{-i}, \hat{\pi}^i}^{-i}(s)$ , and $v_{*,\pi^i}^{-i}(s) = \max_{\hat{\pi}^{-i}} v_{\hat{\pi}^{-i}, \hat{\pi}^i}^{-i}(s)$ . Note that we have $v_{*,\pi^{-i}}^i + v_{\pi^{-i},*}^{-i} = 0$ and $v_{\pi^i,*}^i + v_{*,\pi^i}^{-i} = 0$ because of the zero-sum structure.

(4) Denote $v_{*}^{i}$ (respectively, $v_{*}^{-i}$ ) as the unique fixed-point of the equation $\mathcal{B}^{i}(v^{i}) = v^{i}$ (respectively, $\mathcal{B}^{-i}(v^{-i}) = v^{-i}$ ). Note that we have $v_{*}^{i} + v_{*}^{-i} = 0$ .

# A.2 Boundedness of the Iterates

We first show in the following two lemmas that all the q-functions and v-functions generated by Algorithm 2 are uniformly bounded from above, and the policies are uniformly bounded from below.

Lemma A.1 (Proof in Appendix A.7.1). It holds for all $t, k \geq 0$ and $i \in \{1, 2\}$ that

(1) $\| v_t^i\|_\infty \leq \frac{1}{1 - \gamma},$

(2) $\| q_{t,k}^{i}\|_{\infty}\leq \frac{1}{1 - \gamma}.$

Lemma A.2 (Proof in Appendix A.7.2). It holds for all $t, k \geq 0$ and $(s, a^{i}, a^{-i})$ that

(1) $\pi_{t,k}^{i}(a^{i}|s)\geq \ell_{\tau},$

(2) $\pi_{t,k}^{-i}(a^{-i}|s)\geq \ell_{\tau},$

where $\ell_{\tau} = [1 + (A_{\max} - 1)\exp (2 / [(1 - \gamma)\tau])]^{-1}$ .

# A.3 Analysis of the Outer-Loop: v-Function Update

Our ultimate goal is to bound the expected Nash gap

$$
\mathbb {E} [ N G (\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}) ] = \mathbb {E} \left[ \sum_ {i = 1, 2} \left(\max _ {\pi^ {i}} U ^ {i} (\pi^ {i}, \pi_ {T, K} ^ {- i}) - U ^ {i} (\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i})\right) \right].
$$

We first bound the Nash gap using the value functions of the output policies of Algorithm 2.

Lemma A.3 (Proof in Appendix A.7.4). The following inequality holds:

$$
\sum_ {i = 1, 2} \left(\max _ {\pi^ {i}} U ^ {i} (\pi^ {i}, \pi_ {T, K} ^ {- i}) - U ^ {i} (\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i})\right) \leq \sum_ {i = 1, 2} \left\| v _ {*, \pi_ {T, K} ^ {- i}} ^ {i} - v _ {\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}} ^ {i} \right\| _ {\infty}. \tag {28}
$$

The next lemmas connects the RHS of Eq. (28) to the $v$ -function iterates $\{(v_t^i, v_t^{-i})\}_{t \geq 0}$ of Algorithm 2.

Lemma A.4 (Proof in Appendix A.7.5). It holds for all $t \geq 0$ and i = 1, 2 that

$$
\begin{array}{l} \left\| v _ {*, \pi_ {t, K} ^ {- i}} ^ {i} - v _ {\pi_ {t, K} ^ {i}, \pi_ {t, K} ^ {- i}} ^ {i} \right\| _ {\infty} \leq \frac {2}{1 - \gamma} \bigg (2 \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + 2 \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \\ \left. + \max _ {s} V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\mathrm{max}})\right). \\ \end{array}
$$

In view of Lemma A.4, we need to further bound the terms $\|v_{t}^{i} + v_{t}^{-i}\|_{\infty}$ , $\|v_{t}^{i} - v_{*}^{i}\|_{\infty}$ , and $\max_{s} V_{v_{t},s}(\pi_{t,K}^{i}(s), \pi_{t,K}^{-i}(s))$ . We first consider $\|v_{t}^{i} - v_{*}^{i}\|_{\infty}$ , and establish a one-step Lyapunov drift inequality for it.

Lemma A.5 (Proof in Appendix A.7.6). It holds for all $t \geq 0$ and i = 1, 2 that

$$
\begin{array}{l} \| v _ {t + 1} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \leq \gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} + 2 \max _ {s \in \mathcal {S}} V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 4 \tau \log (A _ {\max}) \\ + \max _ {s \in \mathcal {S}} \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - q _ {t, K} ^ {i} (s) \| _ {\infty} + 2 \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty}. \tag {29} \\ \end{array}
$$

Our next step is to control $\|v_{t}^{i} + v_{t}^{-i}\|_{\infty}$ . Similar to $\|v_{t}^{i} - v_{*}^{i}\|_{\infty}$ , we also establish a one-step Lyapunov drift inequality for $\|v_{t}^{i} + v_{t}^{-i}\|_{\infty}$ in the following lemma.

Lemma A.6 (Proof in Appendix A.7.7). It holds for all $t \geq 0$ that

$$
\| v _ {t + 1} ^ {i} + v _ {t + 1} ^ {- i} \| _ {\infty} \leq \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + \sum_ {i = 1, 2} \max _ {s \in \mathcal {S}} \| q _ {t, K} ^ {i} (s) - \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \| _ {\infty}.
$$

In view of Lemma A.5 and Lemma A.6, our next task is to control the following two terms: $\max_{s\in\mathcal{S}}\|q_{t,K}^{i}(s)-\mathcal{T}^{i}(v_{t}^{i})(s)\pi_{t,K}^{-i}(s)\|_{\infty}$ , and $\max_{s\in\mathcal{S}}V_{v_{t},s}(\pi_{t,K}^{i}(s),\pi_{t,K}^{-i}(s))$ . For ease of exposition, we write down only the inner-loop of Algorithm 2 in the following. All results derived for the q-functions and policies of Algorithm 3 can be directly combined with the outer-loop of Algorithm 2 using a simple conditioning argument together with the Markov property.

Algorithm 3 Inner-Loop of Algorithm 2   
1: Input: Integer K, initializations $q_{0}^{i} = 0 \in R^{|S||A^{i}|}$ and $\pi_{0}^{i}(a^{i}|s) = 1/|A^{i}|$ for all $s \in S$ , and a joint v-function $v = (v^{i}, v^{-i})$ from the outer-loop satisfying $\max(\|v^{-i}\|_{\infty}, \|v^{i}\|_{\infty}) \leq 1/(1 - \gamma)$ .
2: for $k = 0, 1, \cdots, K - 1$ do
3: $\pi_{k+1}^{i}(s) = \pi_{k}^{i}(s) + \beta_{k}(\sigma_{\tau}(q_{k}^{i}(s)) - \pi_{k}^{i}(s))$ for all $s \in S$ 4: Sample $A_{k}^{i} \sim \pi_{k+1}^{i} (\cdot \mid S_{k})$ , and observe $S_{k+1} \sim p (\cdot \mid S_{k}, A_{k}^{i}, A_{k}^{-i})$ 5: $q_{k+1}^{i}(S_{k}, A_{k}^{i}) = q_{k}^{i}(S_{k}, A_{k}^{i}) + \alpha_{k} \left( \mathcal{R}^{i}(S_{k}, A_{k}^{i}, A_{k}^{-i}) + \gamma v^{i}(S_{k+1}) - q_{k}^{i}(S_{k}, A_{k}^{i}) \right)$ 6: end for
7: Output: $q_{K}^{i}$ and $\pi_{K}^{i}$

# A.4 Analysis of the Inner-Loop: Policy Update

We consider $\{(\pi_{k}^{i},\pi_{k}^{-i})\}_{k\geq0}$ generated by Algorithm 3, and use $V_{X}(\cdot,\cdot)$ defined in Eq. (27) as the Lyapunov function to study them. For simplicity of notation, we use $\nabla_{1}V_{X}(\cdot,\cdot)$ (respectively, $\nabla_{2}V_{X}(\cdot,\cdot)$ ) to denote the gradient with respect to the first argument (respectively, the second argument). The following lemma establishes the strongly convexity and the smoothness of $V_{X}(\mu^{i},\mu^{-i})$ . We only state the results regarding the argument $\mu^{i}$ . Similar results also hold for the argument $\mu^{-i}$ .

Lemma A.7 (Proof in Appendix A.7.8). The function $V_{X}(\cdot,\cdot)$ has the following properties.

(1) For any $\mu^{-i} \in \Delta^{|\mathcal{A}^{-i}|}$ , $V_X(\mu^i, \mu^{-i})$ as a function of $\mu^i$ is $\tau$ -strongly convex with respect to $\| \cdot \|_2$ .   
(2) For any $\delta_i > 0$ and $\mu^{-i} \in \Delta^{|\mathcal{A}^{-i}|}$ , $V_X(\mu^i, \mu^{-i})$ as a function of $\mu^i$ is $L_\tau$ - smooth on $\{\mu^i \in \Delta^{|\mathcal{A}^i|} \mid \min_{a^i} \mu^i(a^i) \geq \delta_i\}$ with respect to $\| \cdot \|_2$ , where $L_\tau = \frac{\sigma_{\max}^2(X_{-i})}{\tau} + \frac{\tau}{\delta_i}$ .   
(3) It holds for any $(\mu^i, \mu^{-i})$ that

$$
\langle \nabla_ {1} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \rangle + \langle \nabla_ {2} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \rangle
$$

$$
\leq - \frac {7}{8} V _ {X} (\mu^ {i}, \mu^ {- i}) + \frac {1 6}{\tau} \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} ^ {2}.
$$

(4) For any $u^i \in \mathbb{R}^{|\mathcal{A}^i|}$ , $u^{-i} \in \mathbb{R}^{|\mathcal{A}^{-i}|}$ , we have for all $(\mu^i, \mu^{-i}) \in \{\mu^i \in \Delta^{|\mathcal{A}^i|}, \mu^{-i} \in \Delta^{|\mathcal{A}^{-i}|} | \min_{a^i} \mu^i(a^i) \geq \delta_i, \min_{a^{-i}} \mu^{-i}(a^{-i}) \geq \delta_{-i}\}$ (where $\delta_i, \delta_{-i} > 0$ ) that

$$
\begin{array}{l} \langle \nabla_ {1} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \rangle + \langle \nabla_ {2} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (u ^ {- i}) - \sigma_ {\tau} (X _ {- i} \mu^ {i}) \rangle \\ \leq \left(\frac {\tau}{\delta_ {i}} + \frac {\tau}{\delta_ {- i}} + \| X _ {i} \| _ {2} + \| X _ {- i} \| _ {2}\right) \left[ \frac {2 \bar {c}}{\tau} V _ {X} (\mu^ {i}, \mu^ {- i}) + \frac {1}{\bar {c} \tau^ {2}} \| u ^ {i} - X _ {i} \mu^ {- i} \| _ {2} ^ {2} \right. \\ \left. + \frac {1}{\bar {c} \tau^ {2}} \| u ^ {- i} - X _ {i} \mu^ {i} \| _ {2} ^ {2} \right] \\ \end{array}
$$

where $\bar{c}$ is any positive real number.

With the properties of $V_{X}(\cdot,\cdot)$ established above, we can now use it as a Lyapunov function to study $\pi_{k}^{i}$ and $\pi_{k}^{-i}$ . Specifically, using the smoothness of $V_{X}(\cdot,\cdot)$ , the update equation in Algorithm 3 Line 3, and Lemma A.7 (3) and (4), we have the desired one-step Lyapunov drift inequality for $\sum_{s}V_{v,s}(\pi_{k}^{i}(s),\pi_{k}^{-i}(s))$ , which is presented in the following.

Lemma A.8 (Proof in Appendix A.7.9). The following inequality holds for all $k \geq 0$ :

$$
\sum_ {s} \mathbb {E} [ V _ {v, s} (\pi_ {k + 1} ^ {i} (s), \pi_ {k + 1} ^ {- i} (s)) ] \leq \left(1 - \frac {3 \beta_ {k}}{4}\right) \sum_ {s} \mathbb {E} [ V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) ] + \frac {4 | \mathcal {S} | A _ {\max} ^ {2}}{\ell_ {\tau} (1 - \gamma) ^ {2}} \beta_ {k} ^ {2}
$$

$$
\begin{array}{l} + \frac {2 5 6 A _ {\max} ^ {2} \beta_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3} (1 - \gamma) ^ {2}} \sum_ {i = 1, 2} \sum_ {s} \mathbb {E} [ \| q _ {k} ^ {i} (s) - \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) \| _ {2} ^ {2} ] \\ + \frac {1 6 | \mathcal {S} | A _ {\max} \beta_ {k}}{\tau} \| v ^ {i} + v ^ {- i} \| _ {\infty} ^ {2}. \\ \end{array}
$$

# A.5 Analysis of the Inner-Loop: q-Function Update

In this section, we consider $q_{k}^{i}$ generated by Algorithm 3. We begin by reformulating the update of the q-function as a stochastic approximation algorithm for estimating a time-varying target. Let $F^{i}: R^{|S||A^{i}|} \times S \times A^{i} \times A^{-i} \times S \mapsto R^{|S||A^{i}|}$ be an operator defined as

$$
[ F ^ {i} (q ^ {i}, s _ {0}, a _ {0} ^ {i}, a _ {0} ^ {- i}, s _ {1}) ] (s, a ^ {i}) = \mathbb {1} _ {\{(s, a ^ {i}) = (s _ {0}, a _ {0} ^ {i}) \}} \left(\mathcal {R} ^ {i} (s _ {0}, a _ {0} ^ {i}, a _ {0} ^ {- i}) + \gamma v ^ {i} (s _ {1}) - q ^ {i} (s _ {0}, a _ {0} ^ {i})\right)
$$

for all $(q^{i}, s_{0}, a_{0}^{i}, a_{0}^{-i}, s_{1})$ and $(s, a^{i})$ . Then Algorithm 3 Line 5 can be compactly written as

$$
q _ {k + 1} ^ {i} = q _ {k} ^ {i} + \alpha_ {k} F ^ {i} (q _ {k} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}). \tag {30}
$$

Denote the stationary distribution of the Markov chain $\{S_k\}$ induced by the joint policy $\pi_k = (\pi_k^i, \pi_k^{-i})$ by $\mu_k \in \Delta^{|\mathcal{S}|}$ , the existence and uniqueness of which is guaranteed by Lemma A.2 and Lemma 4.1 (1). Let $\bar{F}_k^i: \mathbb{R}^{|\mathcal{S}||\mathcal{A}^i|} \mapsto \mathbb{R}^{|\mathcal{S}||\mathcal{A}^i|}$ be defined as

$$
\bar {F} _ {k} ^ {i} (q ^ {i}) = \mathbb {E} _ {S _ {0} \sim \mu_ {k} (\cdot), A _ {0} ^ {i} \sim \pi_ {k} ^ {i} (\cdot | S _ {0}), A _ {k} ^ {- i} \sim \pi_ {k} ^ {- i} (\cdot | S _ {0}), S _ {1} \sim p (\cdot | S _ {0}, A _ {0} ^ {i}, A _ {0} ^ {- i})} \left[ F ^ {i} (q ^ {i}, S _ {0}, A _ {0} ^ {i}, A _ {0} ^ {- i}, S _ {1}) \right]
$$

for all $q^{i} \in R^{|S||A^{i}|}$ . Then Eq. (30) can be viewed as a stochastic approximation algorithm for solving the (time-varying) equation $\bar{F}_{k}^{i}(q^{i}) = 0$ with time-inhomogeneous Markovian noise $\{(S_{k}, A_{k}^{i}, A_{k}^{-i}, S_{k+1})\}_{k \geq 0}$ . We next establish the properties of the operators $F^{i}(\cdot)$ and $\bar{F}_{k}^{i}(\cdot)$ in the following lemma.

Lemma A.9 (Proof in Appendix A.7.10). The following inequalities hold:

(1) $\|F^{i}(q_{1}^{i}, s_{0}, a_{0}^{i}, a_{0}^{-i}, s_{1}) - F^{i}(q_{2}^{i}, s_{0}, a_{0}^{i}, a_{0}^{-i}, s_{1})\|_{2} \leq \|q_{1}^{i} - q_{2}^{i}\|_{2} \text{ for any } (q_{1}^{i}, q_{2}^{i}) \text{ and } (s_{0}, a_{0}^{i}, a_{0}^{-i}, s_{1}).$   
(2) $\| F^i (\mathbf{0},s_0,a_0^i,a_0^{-i},s_1)\| _2\leq \frac{1}{1 - \gamma}$ for all $(s_0,a_0^i,a_0^{-i},s_1)$ .   
(3) $\bar{F}_k^i (q^i) = 0$ has a unique solution $\overline{q}_k^i$ , which is explicitly given as $\overline{q}_k^i (s) = \mathcal{T}^i (v^i)(s)\pi_k^{-i}(s)$ for all $s$ .   
(4) $\langle \bar{F}_k^i (q_1^i) - \bar{F}_k^i (q_2^i),q_1^i -q_2^i\rangle \leq -c_{\tau}\| q_1^i -q_2^i\| _2^2$ for all $(q_{1}^{i},q_{2}^{i})$

Using $\| \cdot \|_2^2$ as a Lyapunov function and we have by the equivalent update equation (30) that

$$
\begin{array}{l} \mathbb {E} [ \| q _ {k + 1} ^ {i} - \bar {q} _ {k + 1} ^ {i} \| _ {2} ^ {2} ] \\ = \mathbb {E} [ \| q _ {k + 1} ^ {i} - q _ {k} ^ {i} + q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} + \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \| _ {2} ^ {2} ] \\ = \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ] + \mathbb {E} [ \| q _ {k + 1} ^ {i} - q _ {k} ^ {i} \| _ {2} ^ {2} ] + \mathbb {E} [ \| \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \| _ {2} ^ {2} ] \\ + \alpha_ {k} \mathbb {E} [ \langle F ^ {i} (q _ {k} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}), q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ] + \mathbb {E} [ \langle q _ {k + 1} ^ {i} - q _ {k} ^ {i}, \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \rangle ] \\ + \mathbb {E} [ \langle q _ {k} ^ {i} - \bar {q} _ {k} ^ {i}, \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \rangle ] \\ = \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ] + \alpha_ {k} \underbrace {\mathbb {E} [ \langle \bar {F} _ {k} ^ {i} (q _ {k} ^ {i}) , q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ]} _ {N _ {1}} \\ + \alpha_ {k} \underbrace {\mathbb {E} [ \langle F ^ {i} (q _ {k} ^ {i} , S _ {k} , A _ {k} ^ {i} , A _ {k} ^ {- i} , S _ {k + 1}) - \bar {F} _ {k} ^ {i} (q _ {k} ^ {i}) , q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ]} _ {N _ {2}} \\ + \mathbb {E} [ \| q _ {k + 1} ^ {i} - q _ {k} ^ {i} \| _ {2} ^ {2} ] + \mathbb {E} [ \| \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \| _ {2} ^ {2} ] \\ + \mathbb {E} [ \langle q _ {k + 1} ^ {i} - q _ {k} ^ {i}, \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \rangle ] + \mathbb {E} [ \langle q _ {k} ^ {i} - \bar {q} _ {k} ^ {i}, \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \rangle ]. \tag {31} \\ \end{array}
$$

What remains to do is to bound the terms on the RHS of the previous inequality. Among them, we want to highlight the two terms $N_{1}$ and $N_{2}$ . For the term $N_{1}$ , using Lemma A.9 (4) and we have

$$
N _ {1} = \mathbb {E} [ \langle \bar {F} _ {k} ^ {i} (q _ {k} ^ {i}), q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ] = \mathbb {E} [ \langle \bar {F} _ {k} ^ {i} (q _ {k} ^ {i}) - \bar {F} _ {k} ^ {i} (\bar {q} _ {k} ^ {i}), q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ] \leq - c _ {\tau} \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ], \tag {32}
$$

which provides us the desired negative drift.

The term $N_{2}$ involves the difference between the operator $F^{i}(q_{k}^{i}, S_{k}, A_{k}^{i}, A_{k}^{-i}, S_{k+1})$ and its expected version $\bar{F}_{k}^{i}(q_{k}^{i})$ , and hence can be viewed as the stochastic error due to sampling. The fact that the Markov chain $\{(S_{k}, A_{k}^{i}, A_{k}^{-i}, S_{k+1})\}$ is time-inhomogeneous presents major theoretical challenges in our analysis. To overcome this challenge, observe that: (1) the policy (hence the transition probability matrix of the induced Markov chain) is changing slowly compared to the q-function; see Algorithm 3 Line 3, and (2) the stationary distribution as a function of the policy is Lipschitz (cf. Lemma 4.1 (3)). These two observations together enable us to develop a refined conditioning argument to handle the time-inhomogeneous Markovian noise. The result is presented in following. Similar ideas were previous used in Bhandari et al. (2018); Srikant and Ying (2019); Chen et al. (2021b); Zou et al. (2019); Khodadadian et al. (2022) for finite-sample analysis of single-agent RL algorithms.

Lemma A.10 (Proof in Appendix A.7.11). When $\alpha_{k - z_k,k - 1} \leq 1/4$ for all $k \geq z_k$ , we have for all $k \geq z_k$ that

$$
N _ {2} \leq \frac {3 4 0 | \mathcal {S} | ^ {3 / 2} A _ {\mathrm{max}} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k - z _ {k}, k - 1}.
$$

When using constant stepsize, we have $z_{k}\alpha_{k - z_{k},k - 1} = z_{\beta}^{2}\alpha = \mathcal{O}(\alpha \log^{2}(1 / \beta))$ . Since the two stepsizes $\alpha$ and $\beta$ differ only by a multiplicative constant $c_{\alpha ,\beta}$ , we have $\lim_{\alpha \to 0}z_{\beta}^{2}\alpha = 0$ . Similarly, we also have $\lim_{k\to \infty}z_k\alpha_{k - z_k,k - 1} = 0$ when using diminishing stepsizes. Therefore, Lemma A.10 implies $N_{2} = o(1)$ .

We next bound the rest of terms on the RHS of Eq. (31) in the following lemma.

Lemma A.11 (Proof in Appendix A.7.12). The following inequalities hold for all $k \geq 0$ .

(1) $\mathbb{E}[\|q_{k+1}^{i}-q_{k}^{i}\|_{2}^{2}]\leq\frac{4|\mathcal{S}|A_{\max}\alpha_{k}^{2}}{(1-\gamma)^{2}}.$   
(2) $\mathbb{E}[\| \bar{q}_k^i -\bar{q}_{k + 1}^i \| _2^2 ]\leq \frac{4|\mathcal{S}|A_{\max}\beta_k^2}{(1 - \gamma)^2}.$   
(3) $\mathbb{E}[\langle q_{k + 1}^i -q_k^i,\bar{q}_k^i -\bar{q}_{k + 1}^i\rangle ]\leq \frac{4|\mathcal{S}|A_{\max}\alpha_k\beta_k}{(1 - \gamma)^2}.$   
(4) $\mathbb{E}[\langle q_k^i -\bar{q}_k^i,\bar{q}_k^i -\bar{q}_{k + 1}^i\rangle ]\leq \frac{17A_{\max}^2\beta_k}{\tau(1 - \gamma)^2}\mathbb{E}[\| q_k^i -\bar{q}_k^i\| _2^2 ] + \frac{\beta_k}{16}\sum_s\mathbb{E}[V_{v,s}(\pi_k^i (s),\pi_k^{-i}(s))].$

Using the upper bounds we obtained for all the terms on the RHS of Eq. (31) and we have the one-step Lyapunov drift inequality for $q_{k}^{i}$ . Following the same line of analysis and we also obtain the one-step inequality for $q_{k}^{-i}$ . Both results are presented in the following lemma.

Lemma A.12 (Proof in Appendix A.7.13). The following inequality holds for all $k \geq z_k$ and $i \in \{1, 2\}$ :

$$
\begin{array}{l} \mathbb {E} \left[ \| q _ {k + 1} ^ {i} - \bar {q} _ {k + 1} ^ {i} \| _ {2} ^ {2} \right] \leq \left(1 - c _ {\tau} \alpha_ {k} + \frac {1 7 A _ {\max} ^ {2} \beta_ {k}}{\tau (1 - \gamma) ^ {2}}\right) \mathbb {E} \left[ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} \right] \\ + \frac {3 5 2 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} \alpha_ {k - z _ {k}, k - 1} + \frac {\beta_ {k}}{1 6} \sum_ {s} \mathbb {E} [ V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) ]. \\ \end{array}
$$

# A.6 Solving Coupled Lyapunov Drift Inequalities

We first restate the Lyapunov drift inequalities from previous sections. For simplicity of notation, we denote $\mathcal{L}_{q}(t,k)=\sum_{i=1,2}\|q_{t,k}^{i}-\bar{q}_{t,k}^{i}\|_{2}^{2},\mathcal{L}_{\pi}(t,k)=\sum_{s}V_{v_{t},s}(\pi_{t,k}^{i}(s),\pi_{t,k}^{-i}(s))$ , and $F_{t}$ as the history of Algorithm 2 right before the t-th outer-loop iteration. Note that $v_{t}^{i}$ and $v_{t}^{-i}$ are both measurable with respect to $F_{t}$ . In what follows, we denote $E_{t}[\cdot]$ for $E[\cdot\mid F_{t}]$ .

\- Lemma A.5: It holds for all $t \geq 0$ that

$$
\begin{array}{l} \| v _ {t + 1} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \leq \gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} + 2 \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + 4 \tau \log (A _ {\max}) \\ + 2 \mathcal {L} _ {\pi} (t, K) + \sum_ {i = 1, 2} \| q _ {t, K} ^ {i} - \bar {q} _ {t, K} ^ {i} \| _ {2}. \tag {33} \\ \end{array}
$$

\- Lemma A.6: It holds for all $t \geq 0$ that

$$
\left\| v _ {t + 1} ^ {i} + v _ {t + 1} ^ {- i} \right\| _ {\infty} \leq \gamma \left\| v _ {t} ^ {i} + v _ {t} ^ {- i} \right\| _ {\infty} + \sum_ {i = 1, 2} \left\| q _ {t, K} ^ {i} - \bar {q} _ {t, K} ^ {i} \right\| _ {2}. \tag {34}
$$

\- Lemma A.8: It holds for all $t, k \geq 0$ that

$$
\begin{array}{l} \mathbb {E} _ {t} \left[ \mathcal {L} _ {\pi} (t, k + 1) \right] \leq \left(1 - \frac {3 \beta_ {k}}{4}\right) \mathbb {E} _ {t} \left[ \mathcal {L} _ {\pi} (t, k) \right] + \frac {2 5 6 A _ {\max} ^ {2} \beta_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3} (1 - \gamma) ^ {2}} \mathbb {E} _ {t} \left[ \mathcal {L} _ {q} (t, k) \right] \\ + \frac {1 6 | \mathcal {S} | A _ {\max} \beta_ {k}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} + \frac {4 | \mathcal {S} | A _ {\max} ^ {2} \beta_ {k} ^ {2}}{\ell_ {\tau} (1 - \gamma) ^ {2}}. \tag {35} \\ \end{array}
$$

\- Lemma A.12: It holds for all $t \geq 0$ and $k \geq z_k$ that

$$
\begin{array}{l} \mathbb {E} _ {t} \left[ \mathcal {L} _ {q} (t, k + 1) \right] \leq \left(1 - c _ {\tau} \alpha_ {k} + \frac {1 7 A _ {\max} ^ {2} \beta_ {k}}{\tau (1 - \gamma) ^ {2}}\right) \mathbb {E} _ {t} \left[ \mathcal {L} _ {q} (t, k) \right] \\ + \frac {\beta_ {k}}{1 6} \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) ] + \frac {3 5 2 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} \alpha_ {k - z _ {k}, k - 1}. \tag {36} \\ \end{array}
$$

Adding up Eqs. (35) and (36) and we have

$$
\mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k + 1) + \mathcal {L} _ {q} (t, k + 1) ]
$$

$$
\leq \left(1 - \frac {\beta_ {k}}{2}\right) \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) ] + \left(1 - c _ {\tau} \alpha_ {k} + \frac {2 5 6 A _ {\max} ^ {2} \beta_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3} (1 - \gamma) ^ {2}}\right) \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k) ]
$$

$$
\frac {1 6 | \mathcal {S} | A _ {\max} \beta_ {k}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} + \frac {4 | \mathcal {S} | A _ {\max} ^ {2} \beta_ {k} ^ {2}}{\ell_ {\tau} (1 - \gamma) ^ {2}} + \frac {3 5 2 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} \alpha_ {k - z _ {k}, k - 1}
$$

$$
= \left(1 - \frac {c _ {\alpha , \beta} \alpha_ {k}}{2}\right) \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) ] + \left(1 - c _ {\tau} \alpha_ {k} + \frac {2 5 6 A _ {\mathrm{max}} ^ {2} c _ {\alpha , \beta} \alpha_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3} (1 - \gamma) ^ {2}}\right) \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k) ]
$$

$$
\frac {1 6 | \mathcal {S} | A _ {\max} \beta_ {k}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} + \frac {4 | \mathcal {S} | A _ {\max} ^ {2} \beta_ {k} ^ {2}}{\ell_ {\tau} (1 - \gamma) ^ {2}} + \frac {3 5 2 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} \alpha_ {k - z _ {k}, k - 1}.
$$

Note that Condition A.1 implies that

$$
\frac {2 5 6 A _ {\mathrm{max}} ^ {2} c _ {\alpha , \beta} \alpha_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3} (1 - \gamma) ^ {2}} \leq \frac {c _ {\tau}}{2}.
$$

Therefore, we have

$$
\mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k + 1) + \mathcal {L} _ {q} (t, k + 1) ]
$$

$$
\leq \left(1 - \frac {c _ {\alpha , \beta} \alpha_ {k}}{2}\right) \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) + \mathcal {L} _ {q} (t, k) ] + \frac {1 6 | \mathcal {S} | A _ {\max} c _ {\alpha , \beta} \alpha_ {k}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2}
$$

$$
+ \frac {4 | \mathcal {S} | A _ {\max} ^ {2} c _ {\alpha , \beta} ^ {2} \alpha_ {k} ^ {2}}{\ell_ {\tau} (1 - \gamma) ^ {2}} + \frac {3 5 2 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} \alpha_ {k - z _ {k}, k - 1}. \tag {37}
$$

# A.6.1 Constant Stepsize

When using constant stepsizes, i.e., $\alpha_{k} \equiv \alpha$ , $\beta_{k} \equiv \beta$ , and $\beta = c_{\alpha, \beta}\alpha$ , repeatedly using Eq. (37) from $z_{\beta}$ to $k$ and we have

$$
\begin{array}{l} \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) + \mathcal {L} _ {q} (t, k) ] \\ \leq \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {k - z _ {\beta}} \left(\mathcal {L} _ {\pi} (t, 0) + \mathcal {L} _ {q} (t, 0)\right) \\ + \frac {3 2 | \mathcal {S} | A _ {\max}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} + \frac {8 | \mathcal {S} | A _ {\max} ^ {2} c _ {\alpha , \beta} \alpha}{\ell_ {\tau} (1 - \gamma) ^ {2}} + \frac {7 0 4 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2} c _ {\alpha , \beta}} z _ {\beta} ^ {2} \alpha . \tag {38} \\ \end{array}
$$

We next bound $\mathcal{L}_{\pi}(t,0)+\mathcal{L}_{q}(t,0)$ . For $i\in\{1,2\}$ , since $\pi_{t,0}^{i}$ is initialized at a uniformly random policy and $q_{t,0}^{i}=0$ , we have

$$
\begin{array}{l} \mathcal {L} _ {\pi} (t, 0) = \sum_ {s} V _ {v _ {t}, s} (\pi_ {t, 0} ^ {i} (s), \pi_ {t, 0} ^ {- i} (s)) \\ = \sum_ {s} \sum_ {i = 1, 2} \max _ {\mu^ {i}} \left\{\left(\mu^ {i} - \pi_ {t, 0} ^ {i} (s)\right) ^ {\top} \mathcal {T} ^ {i} \left(v _ {t} ^ {i}\right) (s) \pi_ {t, 0} ^ {- i} (s) + \tau \nu \left(\mu^ {i}\right) - \tau \nu \left(\pi_ {t, 0} ^ {i} (s)\right) \right\} \\ \leq 2 \sum_ {s} \sum_ {i = 1, 2} \max _ {s, a ^ {i}, a ^ {- i}} | \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s, a ^ {i}, a ^ {- i}) | \\ \leq \frac {4 | \mathcal {S} |}{(1 - \gamma)}, \\ \end{array}
$$

and

$$
\mathcal {L} _ {q} (t, 0) = \sum_ {i = 1, 2} \| \bar {q} _ {t, 0} ^ {i} \| _ {2} ^ {2} \leq \frac {2 | \mathcal {S} | A _ {\mathrm{max}}}{(1 - \gamma) ^ {2}}.
$$

Using the previous two bounds in Eq. (38) and we have

$$
\begin{array}{l} \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) + \mathcal {L} _ {q} (t, k) ] \\ \leq \frac {4 | \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {k - z _ {\beta}} + \frac {3 2 | \mathcal {S} | A _ {\max}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} \\ + \frac {8 | \mathcal {S} | A _ {\max} ^ {2} c _ {\alpha , \beta} \alpha}{\ell_ {\tau} (1 - \gamma) ^ {2}} + \frac {7 0 4 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2} c _ {\alpha , \beta}} z _ {\beta} ^ {2} \alpha , \tag {39} \\ \end{array}
$$

which implies

$$
\begin{array}{l} \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) ] \leq \frac {4 | \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {k - z _ {\beta}} + \frac {3 2 | \mathcal {S} | A _ {\max}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} \\ + \frac {8 | \mathcal {S} | A _ {\mathrm{max}} ^ {2} c _ {\alpha , \beta} \alpha}{\ell_ {\tau} (1 - \gamma) ^ {2}} + \frac {7 0 4 | \mathcal {S} | ^ {3 / 2} A _ {\mathrm{max}} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2} c _ {\alpha , \beta}} z _ {\beta} ^ {2} \alpha \\ \leq \frac {4 | \mathcal {S} | A _ {\mathrm{max}}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {k - z _ {\beta}} + \frac {3 2 | \mathcal {S} | A _ {\mathrm{max}}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} \\ + \frac {7 1 2 | \mathcal {S} | ^ {3 / 2} A _ {\mathrm{max}} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2} c _ {\alpha , \beta}} z _ {\beta} ^ {2} \alpha . \\ \end{array}
$$

Substituting the previous inequality on $\mathbb{E}_t[\mathcal{L}_\pi (t,k)]$ into Eq. (36) and we have

$$
\mathbb {E} _ {t} \left[ \mathcal {L} _ {q} (t, k + 1) \right] \leq \left(1 - c _ {\tau} \alpha + \frac {1 7 A _ {\max} ^ {2} \beta}{\tau (1 - \gamma) ^ {2}}\right) \mathbb {E} _ {t} \left[ \mathcal {L} _ {q} (t, k) \right] + \frac {3 5 2 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2}}{(1 - \gamma) ^ {2}} z _ {\beta} ^ {2} \alpha^ {2}
$$

$$
+ \frac {c _ {\alpha , \beta} \alpha}{1 6} \left(\frac {4 | \mathcal {S} | A _ {\mathrm{max}}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {k - z _ {\beta}} + \frac {3 2 | \mathcal {S} | A _ {\mathrm{max}}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} \right.
$$

$$
\left. + \frac {7 1 2 | \mathcal {S} | ^ {3 / 2} A _ {\mathrm{max}} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2} c _ {\alpha , \beta}} z _ {\beta} ^ {2} \alpha\right)
$$

$$
\leq \left(1 - \frac {c _ {\tau} \alpha}{2}\right) \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k) ] + \frac {| \mathcal {S} | A _ {\max} c _ {\alpha , \beta} \alpha}{4 (1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {k - z _ {\beta}}
$$

$$
+ \frac {2 | \mathcal {S} | A _ {\max} c _ {\alpha , \beta} \alpha}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} + \frac {4 5 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {\beta} ^ {2} \alpha^ {2},
$$

where the last line follows from Condition A.1. Repeatedly using the previous inequality from $z_{\beta}$ to k and we have

$$
\mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k) ] \leq \frac {2 | \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\tau} \alpha}{2}\right) ^ {k - z _ {\beta}} + \frac {| \mathcal {S} | A _ {\max} c _ {\alpha , \beta} \alpha (k - z _ {\beta})}{4 (1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {k - z _ {\beta} - 1}
$$

$$
+ \frac {4 | \mathcal {S} | A _ {\max} c _ {\alpha , \beta}}{c _ {\tau} \tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} + \frac {9 0 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{c _ {\tau} (1 - \gamma) ^ {2}} z _ {\beta} ^ {2} \alpha .
$$

The next step is to substitute the previous bound on $\mathbb{E}_t[\mathcal{L}_q(t,k)]$ into Eq. (34). To achieve that, first note that

$$
\sum_ {i = 1, 2} \mathbb {E} _ {t} \left[ \| q _ {t, K} ^ {i} - \bar {q} _ {t, K} ^ {i} \| _ {2} \right] \leq \sum_ {i = 1, 2} \left(\mathbb {E} _ {t} \left[ \| q _ {t, K} ^ {i} - \bar {q} _ {t, K} ^ {i} \| _ {2} ^ {2} \right]\right) ^ {1 / 2} \quad (\text { Jensen's   inequality })
$$

$$
\leq 2 \left(\sum_ {i = 1, 2} \mathbb {E} _ {t} \left[ \| q _ {t, K} ^ {i} - \bar {q} _ {t, K} ^ {i} \| _ {2} ^ {2} \right]\right) ^ {1 / 2} \quad (\sqrt {a} + \sqrt {b} \leq 2 \sqrt {a + b})
$$

$$
\leq 2 \mathbb {E} _ {t} ^ {1 / 2} [ \mathcal {L} _ {q} (t, K) ].
$$

Therefore, we have

$$
\sum_ {i = 1, 2} \mathbb {E} _ {t} \left[ \| q _ {t, K} ^ {i} - \bar {q} _ {t, K} ^ {i} \| _ {2} \right]
$$

$$
\leq 2 \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, K) ] ^ {1 / 2}
$$

$$
\leq \frac {3 \sqrt {| \mathcal {S} | A _ {\mathrm{max}}}}{(1 - \gamma)} \left(1 - \frac {c _ {\tau} \alpha}{2}\right) ^ {\frac {K - z _ {\beta}}{2}} + \frac {\sqrt {| \mathcal {S} | A _ {\mathrm{max}}} c _ {\alpha , \beta} ^ {1 / 2} \alpha^ {1 / 2} (K - z _ {\beta}) ^ {1 / 2}}{(1 - \gamma)} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}}
$$

$$
+ \frac {4 \sqrt {| \mathcal {S} | A _ {\max}} c _ {\alpha , \beta} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} \tau^ {1 / 2}} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + \frac {2 0 | \mathcal {S} | ^ {3 / 4} A _ {\max} ^ {3 / 4} \hat {L} _ {\tau} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} (1 - \gamma)} z _ {\beta} \alpha^ {1 / 2}. \tag {40}
$$

Taking the total expectation on both sides of the previous inequality then using the result in Eq. (34), and we obtain

$$
\begin{array}{l} \mathbb {E} [ \| v _ {t + 1} ^ {i} + v _ {t + 1} ^ {- i} \| _ {\infty} ] \leq \left(\gamma + \frac {4 \sqrt {| \mathcal {S} | A _ {\max}} c _ {\alpha , \beta} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} \tau^ {1 / 2}}\right) \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] \\ + \frac {4 \sqrt {| \mathcal {S} | A _ {\max}} (K - z _ {\beta}) ^ {1 / 2}}{(1 - \gamma)} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}} \\ + \frac {2 0 | \mathcal {S} | ^ {3 / 4} A _ {\max} ^ {3 / 4} \hat {L} _ {\tau} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} (1 - \gamma)} z _ {\beta} \alpha^ {1 / 2} \\ \leq \left(\frac {1 + \gamma}{2}\right) \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] \\ \end{array}
$$

$$
\begin{array}{l} + \frac {4 \sqrt {| \mathcal {S} | A _ {\mathrm{max}} (K - z _ {\beta}) ^ {1 / 2}}}{(1 - \gamma)} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}} \\ + \frac {2 0 | \mathcal {S} | ^ {3 / 4} A _ {\mathrm{max}} ^ {3 / 4} \hat {L} _ {\tau} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} (1 - \gamma)} z _ {\beta} \alpha^ {1 / 2}, \\ \end{array}
$$

where the last line follows from Condition A.1. Since $\|v_{0}^{i} + v_{0}^{-i}\|_{\infty} \leq \frac{2}{1-\gamma}$ , repeatedly using the previous inequality starting from 0 and we have for all $t \geq 0$ that

$$
\begin{array}{l} \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] \leq \frac {2}{1 - \gamma} \left(\frac {1 + \gamma}{2}\right) ^ {t} + \frac {8 \sqrt {| \mathcal {S} | A _ {\max}} (K - z _ {\beta}) ^ {1 / 2}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}} \\ + \frac {4 0 | \mathcal {S} | ^ {3 / 4} A _ {\max} ^ {3 / 4} \hat {L} _ {\tau} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} (1 - \gamma) ^ {2}} z _ {\beta} \alpha^ {1 / 2}. \tag {41} \\ \end{array}
$$

Now we have obtained finite-sample bounds for $\mathcal{L}_{q}(t,k)$ , $\mathcal{L}_{\pi}(t,k)$ , and $\|v_{t}^{i} + v_{t}^{-i}\|_{\infty}$ . The next step is to use them in Eq. (33) to obtain finite-sample bounds for $\|v_{t}^{i} - v_{*}^{i}\|_{\infty}$ . Specifically, we have by Eq. (33), Eq. (39), and Eq. (40) that

$$
\mathbb {E} [ \| v _ {t + 1} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] \leq \gamma \mathbb {E} [ \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] + 2 \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] + 4 \tau \log (A _ {\max})
$$

$$
+ 2 \mathbb {E} [ \mathcal {L} _ {\pi} (t, K) ] + \sum_ {i = 1, 2} \mathbb {E} \| q _ {t, K} ^ {i} - \bar {q} _ {t, K} ^ {i} \| _ {2} ]
$$

$$
\leq \gamma \mathbb {E} [ \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] + 2 \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] + 4 \tau \log (A _ {\mathrm{max}})
$$

$$
+ 2 \mathbb {E} [ \mathcal {L} _ {\pi} (t, K) ] + 2 \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, K) ] ^ {1 / 2}
$$

$$
\leq \gamma \mathbb {E} [ \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] + 2 \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] + 4 \tau \log (A _ {\max})
$$

$$
+ \frac {8 | \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {K - z _ {\beta}} + \frac {6 4 | \mathcal {S} | A _ {\max}}{\tau} \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} ]
$$

$$
+ \frac {1 4 2 4 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2} c _ {\alpha , \beta}} z _ {\beta} ^ {2} \alpha + \frac {3 \sqrt {| \mathcal {S} | A _ {\max}}}{(1 - \gamma)} \left(1 - \frac {c _ {\tau} \alpha}{2}\right) ^ {\frac {K - z _ {\beta}}{2}}
$$

$$
+ \frac {\sqrt {| \mathcal {S} | A _ {\max}} c _ {\alpha , \beta} ^ {1 / 2} \alpha^ {1 / 2} (K - z _ {\beta}) ^ {1 / 2}}{(1 - \gamma)} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}}
$$

$$
+ \frac {4 \sqrt {| \mathcal {S} | A _ {\max}} c _ {\alpha , \beta} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} \tau^ {1 / 2}} \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] + \frac {2 0 | \mathcal {S} | ^ {3 / 4} A _ {\max} ^ {3 / 4} \hat {L} _ {\tau} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} (1 - \gamma)} z _ {\beta} \alpha^ {1 / 2}
$$

$$
\leq \gamma \mathbb {E} [ \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] + \frac {1 3 4 | \mathcal {S} | A _ {\max}}{\tau (1 - \gamma)} \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] + 4 \tau \log (A _ {\max})
$$

$$
+ \frac {1 2 | \mathcal {S} | A _ {\max} (K - z _ {\beta}) ^ {1 / 2}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}}
$$

$$
+ \frac {1 4 4 4 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2} c _ {\alpha , \beta}} z _ {\beta} ^ {2} \alpha^ {1 / 2}
$$

$$
\leq \gamma \mathbb {E} [ \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] + 4 \tau \log (A _ {\max}) + \frac {1 4 4 4 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2} c _ {\alpha , \beta}} z _ {\beta} ^ {2} \alpha^ {1 / 2}
$$

$$
+ \frac {1 2 | \mathcal {S} | A _ {\max} (K - z _ {\beta}) ^ {1 / 2}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}}
$$

$$
+ \frac {1 3 4 | \mathcal {S} | A _ {\max}}{\tau (1 - \gamma)} \left(\frac {2}{1 - \gamma} \left(\frac {1 + \gamma}{2}\right) ^ {t} + \frac {4 0 | \mathcal {S} | ^ {3 / 4} A _ {\max} ^ {3 / 4} \hat {L} _ {\tau} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} (1 - \gamma) ^ {2}} z _ {\beta} \alpha^ {1 / 2} \right.
$$

$$
+ \frac {8 \sqrt {| \mathcal {S} | A _ {\mathrm{max}}} (K - z _ {\beta}) ^ {1 / 2}}{(1 - \gamma) ^ {2}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}}
$$

$$
\begin{array}{l} \leq \gamma \mathbb {E} [ \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] + 4 \tau \log (A _ {\max}) + \frac {2 6 8 | \mathcal {S} | A _ {\max}}{\tau (1 - \gamma) ^ {2}} \left(\frac {1 + \gamma}{2}\right) ^ {t} \\ + \frac {1 0 8 4 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} (K - z _ {\beta}) ^ {1 / 2}}{\tau (1 - \gamma) ^ {3}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}} \\ + \frac {6 8 0 4 | \mathcal {S} | ^ {2} A _ {\mathrm{max}} ^ {2} \hat {L} _ {\tau}}{c _ {\alpha , \beta} (1 - \gamma) ^ {3}} z _ {\beta} ^ {2} \alpha^ {1 / 2}. \\ \end{array}
$$

Repeatedly using the previous inequality from 0 to T - 1 and we have

$$
\begin{array}{l} \mathbb {E} [ \| v _ {T} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] \leq \frac {4 \tau \log (A _ {\max})}{1 - \gamma} + \frac {2 7 0 | \mathcal {S} | A _ {\max} T}{\tau (1 - \gamma) ^ {2}} \left(\frac {1 + \gamma}{2}\right) ^ {T - 1} \\ + \frac {1 0 8 4 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} (K - z _ {\beta}) ^ {1 / 2}}{\tau (1 - \gamma) ^ {4}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}} \\ + \frac {6 8 0 4 | \mathcal {S} | ^ {2} A _ {\mathrm{max}} ^ {2} \hat {L} _ {\tau}}{c _ {\alpha , \beta} (1 - \gamma) ^ {4}} z _ {\beta} ^ {2} \alpha^ {1 / 2}, \\ \end{array}
$$

where we used $\| v_0^i - v_*^i\|_{\infty}\leq 2 / (1 - \gamma)$ .

Our next step is to use the bounds we obtained for $\mathcal{L}_q(t,k)$ , $\mathcal{L}_{\pi}(t,k)$ , $\| v_t^i + v_t^{-i}\|_{\infty}$ , and $\| v_t^i - v_*^i\|_{\infty}$ in Lemma A.4. For simplicity, we use $a \lesssim b$ to mean that there exists a numerical constant $c$ such that $a \leq cb$ . Now, we have by the previous inequality, Eq. (39), and Eq. (41) that

$$
\begin{array}{l} \mathbb {E} \left[ \left\| v _ {*, \pi_ {T, K} ^ {- i}} ^ {i} - v _ {\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}} ^ {i} \right\| _ {\infty} \right] \\ \leq \frac {2}{1 - \gamma} \left(2 \mathbb {E} [ \| v _ {T} ^ {i} + v _ {T} ^ {- i} \| _ {\infty} ] + 2 \mathbb {E} [ \| v _ {T} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] + \mathcal {L} _ {\pi} (T, K) + 2 \tau \log (A _ {\max})\right) \\ \lesssim \frac {\tau \log (A _ {\max})}{(1 - \gamma) ^ {2}} + \frac {| \mathcal {S} | A _ {\max} T}{\tau (1 - \gamma) ^ {3}} \left(\frac {1 + \gamma}{2}\right) ^ {T - 1} \\ + \frac {| \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} (K - z _ {\beta}) ^ {1 / 2}}{\tau (1 - \gamma) ^ {5}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}} + \frac {| \mathcal {S} | ^ {2} A _ {\max} ^ {2} \hat {L} _ {\tau}}{c _ {\alpha , \beta} (1 - \gamma) ^ {5}} z _ {\beta} ^ {2} \alpha^ {1 / 2}, \\ \end{array}
$$

Finally, using the previous inequality in Lemma A.3 and we have

$$
\begin{array}{l} \mathbb {E} [ N G (\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}) ] \lesssim \frac {\tau \log (A _ {\max})}{(1 - \gamma) ^ {2}} + \frac {| \mathcal {S} | A _ {\max} T}{\tau (1 - \gamma) ^ {3}} \left(\frac {1 + \gamma}{2}\right) ^ {T - 1} \\ + \frac {| \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} (K - z _ {\beta}) ^ {1 / 2}}{\tau (1 - \gamma) ^ {5}} \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {\frac {K - z _ {\beta} - 1}{2}} \\ + \frac {| \mathcal {S} | ^ {2} A _ {\mathrm{max}} ^ {2} \hat {L} _ {\tau}}{c _ {\alpha , \beta} (1 - \gamma) ^ {5}} z _ {\beta} ^ {2} \alpha^ {1 / 2}. \\ \end{array}
$$

The proof of Theorem 3.1 is complete.

# A.6.2 Diminishing Stepsizes

Consider using linearly diminishing stepsizes, i.e., $\alpha_{k} = \frac{\alpha}{k+h}$ , $\beta_{k} = \frac{\beta}{k+h}$ , and $\beta = c_{\alpha,\beta}\alpha$ . Repeatedly using Eq. (37) and we have for all $k \geq k_{0}$ that

$$
\mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) + \mathcal {L} _ {q} (t, k) ] \lesssim \frac {4 | \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \underbrace {\prod_ {m = k _ {0}} ^ {k - 1} \left(1 - \frac {c _ {\alpha , \beta} \alpha_ {m}}{2}\right)} _ {\hat {\mathcal {E}} _ {1}}
$$

$$
\begin{array}{l} + \frac {| \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} \underbrace {\sum_ {n = k _ {0}} ^ {k - 1} z _ {n} ^ {2} \alpha_ {n} ^ {2} \prod_ {m = n + 1} ^ {k - 1} \left(1 - \frac {c _ {\alpha , \beta} \alpha_ {m}}{2}\right)} _ {\hat {\mathcal {E}} _ {2}} \\ + \frac {| \mathcal {S} | A _ {\max} c _ {\alpha , \beta}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} \underbrace {\sum_ {n = k _ {0}} ^ {k - 1} \alpha_ {n} \prod_ {m = n + 1} ^ {k - 1} \left(1 - \frac {c _ {\alpha , \beta} \alpha_ {m}}{2}\right)} _ {\hat {\mathcal {E}} _ {3}}. \\ \end{array}
$$

We next provide estimates for the terms $\{\hat{E}_{j}\}_{1\leq j\leq3}$ . Bounds of terms like $\{\hat{E}_{j}\}_{1\leq j\leq3}$ are well-established in existing work studying the convergence rate of iterative algorithms (Srikant and Ying, 2019; Lan, 2020; Chen et al., 2021b). Specifically, we have from (Chen et al., 2021b, Appendix A.2.) that

$$
\hat {\mathcal {E}} _ {1} \leq \left(\frac {k _ {0} + h}{k + h}\right) ^ {c _ {\alpha , \beta} \alpha / 2}, \quad \hat {\mathcal {E}} _ {2} \leq \frac {4 e z _ {k} ^ {2} \alpha^ {2}}{c _ {\alpha , \beta} \alpha / 2 - 1} \frac {1}{k + h}, \text { and } \hat {\mathcal {E}} _ {3} \leq \frac {2}{c _ {\alpha , \beta}}.
$$

It follows that

$$
\begin{array}{l} \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) + \mathcal {L} _ {q} (t, k) ] \\ \lesssim \frac {| \mathcal {S} | A _ {\mathrm{max}}}{(1 - \gamma) ^ {2}} \left(\frac {k _ {0} + h}{k + h}\right) ^ {c _ {\alpha , \beta} \alpha / 2} + \frac {| \mathcal {S} | ^ {3 / 2} A _ {\mathrm{max}} ^ {2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} \frac {z _ {k} ^ {2} \alpha^ {2}}{c _ {\alpha , \beta} \alpha / 2 - 1} \frac {1}{k + h} \\ + \frac {| \mathcal {S} | A _ {\mathrm{max}}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} \\ \lesssim \frac {| \mathcal {S} | A _ {\mathrm{max}}}{(1 - \gamma) ^ {2}} \left(\frac {\alpha_ {k}}{\alpha_ {k _ {0}}}\right) ^ {c _ {\alpha , \beta} \alpha / 2} + \frac {| \mathcal {S} | ^ {3 / 2} A _ {\mathrm{max}} ^ {2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} \frac {z _ {k} ^ {2} \alpha^ {2}}{c _ {\alpha , \beta} \alpha / 2 - 1} \frac {1}{k + h} \\ + \frac {| \mathcal {S} | A _ {\max}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2}, \tag {42} \\ \end{array}
$$

which implies

$$
\begin{array}{l} \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) ] \lesssim \frac {| \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \left(\frac {\alpha_ {k}}{\alpha_ {k _ {0}}}\right) ^ {c _ {\alpha , \beta} \alpha / 2} + \frac {| \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} \frac {z _ {k} ^ {2} \alpha^ {2}}{c _ {\alpha , \beta} \alpha / 2 - 1} \frac {1}{k + h} \\ + \frac {| \mathcal {S} | A _ {\mathrm{max}}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2}. \\ \end{array}
$$

Using the previous bound on $\mathbb{E}_t[\mathcal{L}_\pi (t,k)]$ in Eq. (36) and we have

$$
\begin{array}{l} \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k + 1) ] \leq \left(1 - c _ {\tau} \alpha_ {k} + \frac {1 7 A _ {\max} ^ {2} \beta_ {k}}{\tau (1 - \gamma) ^ {2}}\right) \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k) ] \\ + \frac {\beta_ {k}}{1 6} \mathbb {E} _ {t} [ \mathcal {L} _ {\pi} (t, k) ] + \frac {3 5 2 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} \alpha_ {k - z _ {k}, k - 1} \\ \lesssim \left(1 - \frac {c _ {\tau} \alpha_ {k}}{2}\right) \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k) ] + \frac {| \mathcal {S} | ^ {3 / 2} A _ {\mathrm{max}} ^ {2}}{\alpha_ {k _ {0}} (1 - \gamma) ^ {2}} z _ {k} ^ {2} \alpha_ {k} ^ {2} \\ + \frac {| \mathcal {S} | A _ {\mathrm{max}} c _ {\alpha , \beta} \alpha_ {k}}{\tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2}. \\ \end{array}
$$

Repeatedly using the previous inequality starting from $k_{0}$ and we have

$$
\begin{array}{l} \mathbb {E} _ {t} [ \mathcal {L} _ {q} (t, k) ] \lesssim \frac {| \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \left(\frac {\alpha_ {k}}{\alpha_ {k _ {0}}}\right) ^ {c _ {\tau} \alpha / 2} + \frac {| \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {2}}{\alpha_ {k _ {0}} (1 - \gamma) ^ {2}} z _ {k} ^ {2} \alpha_ {k} \\ + \frac {| \mathcal {S} | A _ {\mathrm{max}} c _ {\alpha , \beta}}{c _ {\tau} \tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2} \\ \end{array}
$$

$$
\lesssim \frac {| \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {2}}{\alpha_ {k _ {0}} (1 - \gamma) ^ {2}} z _ {k} ^ {2} \alpha_ {k} + \frac {| \mathcal {S} | A _ {\max} c _ {\alpha , \beta}}{c _ {\tau} \tau} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ^ {2}
$$

Since $\sum_{i=1,2} \mathbb{E}_t \left[ \| q_{t,K}^i - \bar{q}_{t,K}^i \|_2 \right] \lesssim \mathbb{E}_t [\mathcal{L}_q(t, K)]^{1/2}$ , we have

$$
\sum_ {i = 1, 2} \mathbb {E} _ {t} \left[ \| q _ {t, K} ^ {i} - \bar {q} _ {t, K} ^ {i} \| _ {2} \right] \leq \frac {c _ {1} ^ {\prime} | \mathcal {S} | ^ {3 / 4} A _ {\max}}{\alpha_ {k _ {0}} ^ {1 / 2} (1 - \gamma)} z _ {k} \alpha_ {k} ^ {1 / 2} + \frac {c _ {2} ^ {\prime} \sqrt {| \mathcal {S} | A _ {\max}} c _ {\alpha , \beta} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} \tau^ {1 / 2}} \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty}, \tag {43}
$$

where $c_1'$ and $c_2'$ are numerical constants. Taking total expectation on both sides of the previous inequality and then using the result in Eq. (34), and we have

$$
\begin{array}{l} \mathbb {E} [ \| v _ {t + 1} ^ {i} + v _ {t + 1} ^ {- i} \| _ {\infty} ] \\ \leq \left(\gamma + \frac {c _ {2} ^ {\prime} \sqrt {| \mathcal {S} | A _ {\max}} c _ {\alpha , \beta} ^ {1 / 2}}{c _ {\tau} ^ {1 / 2} \tau^ {1 / 2}}\right) \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] + \frac {c _ {1} ^ {\prime} | \mathcal {S} | ^ {3 / 4} A _ {\max}}{\alpha_ {k _ {0}} ^ {1 / 2} (1 - \gamma)} z _ {k} \alpha_ {k} ^ {1 / 2} \\ \leq \left(\frac {\gamma + 1}{2}\right) \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] + \frac {c _ {1} ^ {\prime} | \mathcal {S} | ^ {3 / 4} A _ {\max}}{\alpha_ {k _ {0}} ^ {1 / 2} (1 - \gamma)} z _ {k} \alpha_ {k} ^ {1 / 2}, \\ \end{array}
$$

where the last line follows from Condition A.1. Repeatedly using the previous inequality starting from 0 and we have

$$
\mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] \lesssim \frac {2}{1 - \gamma} \left(\frac {\gamma + 1}{2}\right) ^ {t} + \frac {| \mathcal {S} | ^ {3 / 4} A _ {\max}}{\alpha_ {k _ {0}} ^ {1 / 2} (1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} ^ {1 / 2}. \tag {44}
$$

The next step is to bound $\|v_{t}^{i}-v_{*}^{i}\|_{\infty}$ . Recall from Eq. (33) that

$$
\begin{array}{l} \mathbb {E} [ \| v _ {t + 1} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] \leq \gamma \mathbb {E} [ \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] + 2 \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] + 4 \tau \log (A _ {\max}) \\ + 2 \mathbb {E} [ \mathcal {L} _ {\pi} (t, K) ] + 2 \mathbb {E} [ \mathcal {L} _ {q} (t, K) ] ^ {1 / 2}. \\ \end{array}
$$

Since Eq. (42) and Eq. (43) imply that

$$
\begin{array}{l} \mathbb {E} [ \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} ] + \tau \log (A _ {\max}) + \mathbb {E} [ \mathcal {L} _ {\pi} (t, K) ] + \mathbb {E} [ \mathcal {L} _ {q} (t, K) ] ^ {1 / 2} \\ \lesssim \frac {| \mathcal {S} | A _ {\mathrm{max}}}{\tau (1 - \gamma) ^ {2}} \left(\frac {\gamma + 1}{2}\right) ^ {t} + \tau \log (A _ {\mathrm{max}}) + \frac {| \mathcal {S} | ^ {2} A _ {\mathrm{max}} ^ {2} \hat {L} _ {\tau}}{\alpha_ {k _ {0}} c _ {\alpha , \beta} (1 - \gamma) ^ {3}} z _ {K} ^ {2} \alpha_ {K} ^ {1 / 2}, \\ \end{array}
$$

we have

$$
\begin{array}{l} \mathbb {E} [ \| v _ {t + 1} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] \leq \gamma \mathbb {E} [ \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] \\ + c ^ {\prime \prime} \left[ \frac {| \mathcal {S} | A _ {\mathrm{max}}}{\tau (1 - \gamma) ^ {2}} \left(\frac {\gamma + 1}{2}\right) ^ {t} + \tau \log (A _ {\mathrm{max}}) + \frac {| \mathcal {S} | ^ {2} A _ {\mathrm{max}} ^ {2} \hat {L} _ {\tau}}{\alpha_ {k _ {0}} c _ {\alpha , \beta} (1 - \gamma) ^ {3}} z _ {K} ^ {2} \alpha_ {K} ^ {1 / 2} \right] \\ \end{array}
$$

for some numerical constant $c''$ . Repeatedly using the previous inequality starting from 0 to $T - 1$ and we have

$$
\mathbb {E} [ \| v _ {T} ^ {i} - v _ {*} ^ {i} \| _ {\infty} ] \lesssim \frac {| \mathcal {S} | A _ {\max} T}{\tau (1 - \gamma) ^ {2}} \left(\frac {\gamma + 1}{2}\right) ^ {T - 1} + \frac {\tau \log (A _ {\max})}{(1 - \gamma)} + \frac {| \mathcal {S} | ^ {2} A _ {\max} ^ {2} \hat {L} _ {\tau}}{\alpha_ {k _ {0}} c _ {\alpha , \beta} (1 - \gamma) ^ {4}} z _ {K} ^ {2} \alpha_ {K} ^ {1 / 2}
$$

Using the previous inequality, Eq. (42), and Eq. (44) in Lemma A.4, and we obtain

$$
\begin{array}{l} \mathbb {E} [ \| v _ {*, \pi_ {T, K} ^ {- i}} ^ {i} - v _ {\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}} ^ {i} \| _ {\infty} ] \lesssim \frac {| \mathcal {S} | A _ {\max} T}{\tau (1 - \gamma) ^ {3}} \left(\frac {\gamma + 1}{2}\right) ^ {T - 1} + \frac {\tau \log (A _ {\max})}{(1 - \gamma) ^ {2}} \\ + \frac {| \mathcal {S} | ^ {2} A _ {\mathrm{max}} ^ {2} \hat {L} _ {\tau}}{\alpha_ {k _ {0}} c _ {\alpha , \beta} (1 - \gamma) ^ {5}} z _ {K} ^ {2} \alpha_ {K} ^ {1 / 2} \\ \end{array}
$$

Finally, we have by the previous inequality and Lemma A.3 that

$$
\begin{array}{l} \mathbb {E} [ N G (\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}) ] \lesssim \frac {| \mathcal {S} | A _ {\max} T}{\tau (1 - \gamma) ^ {3}} \left(\frac {\gamma + 1}{2}\right) ^ {T - 1} + \frac {\tau \log (A _ {\max})}{(1 - \gamma) ^ {2}} \\ + \frac {| \mathcal {S} | ^ {2} A _ {\mathrm{max}} ^ {2} \hat {L} _ {\tau}}{\alpha_ {k _ {0}} c _ {\alpha , \beta} (1 - \gamma) ^ {5}} z _ {K} ^ {2} \alpha_ {K} ^ {1 / 2}. \\ \end{array}
$$

The proof of Theorem 3.2 is complete.

# A.7 Proof of All Supporting Lemmas

# A.7.1 Proof of Lemma A.1

We first show by induction that whenever $\|v_{t}^{i}\|_{\infty}\leq\frac{1}{1-\gamma}$ , we have $\|q_{t,k}^{i}\|_{\infty}\leq\frac{1}{1-\gamma}$ for all $k\geq0$ . Note that $\|q_{t,0}^{i}\|_{\infty}\leq\frac{1}{1-\gamma}$ holds by our initialization. Suppose that $\|q_{t,k}^{i}\|_{\infty}\leq\frac{1}{1-\gamma}$ for some $k\geq0$ . Then we have for all $(s,a^{i})$ that

$$
\begin{array}{l} \left| q _ {t, k + 1} ^ {i} \left(s, a ^ {i}\right) \right| \\ = | q _ {t, k} ^ {i} (s, a ^ {i}) + \alpha_ {k} \mathbb {1} _ {\{(s, a ^ {i}) = (S _ {k}, A _ {k} ^ {i}) \}} (\mathcal {R} ^ {i} (S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}) + \gamma v _ {t} ^ {i} (S _ {k + 1}) - q _ {t, k} ^ {i} (S _ {k}, A _ {k} ^ {i})) | \\ \leq \left(1 - \alpha_ {k} \mathbb {1} _ {\{(s, a ^ {i}) = (S _ {k}, A _ {k} ^ {i}) \}}\right) | q _ {t, k} ^ {i} (s, a ^ {i}) | \\ + \alpha_ {k} \mathbb {1} _ {\{(s, a ^ {i}) = (S _ {k}, A _ {k} ^ {i}) \}} | \mathcal {R} ^ {i} (S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}) + \gamma v _ {t} ^ {i} (S _ {k + 1}) | \\ \leq \left(1 - \alpha_ {k} \mathbb {1} _ {\{(s, a ^ {i}) = (S _ {k}, A _ {k} ^ {i}) \}}\right) \frac {1}{1 - \gamma} + \alpha_ {k} \mathbb {1} _ {\{(s, a ^ {i}) = (S _ {k}, A _ {k} ^ {i}) \}} \left(1 + \frac {\gamma}{1 - \gamma}\right) \tag {45} \\ = \frac {1}{1 - \gamma}, \\ \end{array}
$$

where Eq. (45) follows from the induction hypothesis $\|q_{t,0}^{i}\|_{\infty} \leq \frac{1}{1-\gamma}$ , $\|v_{t}^{i}\|_{\infty} \leq \frac{1}{1-\gamma}$ , and $\max_{s,a^{i},a^{-i}} |\mathcal{R}^{i}(s,a^{i},a^{-i})| \leq 1$ . The induction is now complete and we have $\|q_{t,k}^{i}\|_{\infty} \leq \frac{1}{1-\gamma}$ for all $k \geq 0$ whenever $\|v_{t}^{i}\|_{\infty} \leq \frac{1}{1-\gamma}$ .

We next again use induction to show that $\|v_{t}^{i}\|_{\infty} \leq \frac{1}{1-\gamma}$ for all $t \geq 0$ . Our initialization ensures that $\|v_{0}^{i}\|_{\infty} \leq \frac{1}{1-\gamma}$ . Suppose that $\|v_{t}^{i}\|_{\infty} \leq \frac{1}{1-\gamma}$ for some $t \geq 0$ . Using the update equation for $v_{t+1}^{i}$ (i.e., Algorithm 2 Line 8) and the fact that $\|q_{t,k}^{i}\|_{\infty} \leq \frac{1}{1-\gamma}$ for all $k \geq 0$ , we have for all $s \in S$ that

$$
| v _ {t + 1} ^ {i} (s) | = \left| \sum_ {a ^ {i} \in \mathcal {A} ^ {i}} \pi_ {t, K} ^ {i} (a ^ {i} | s) q _ {t, K} ^ {i} (s, a ^ {i}) \right| \leq \sum_ {a ^ {i} \in \mathcal {A} ^ {i}} \pi_ {t, K} ^ {i} (a ^ {i} | s) \| q _ {t, K} ^ {i} \| _ {\infty} \leq \frac {1}{1 - \gamma}.
$$

The induction for $\{v_t^i\}$ is now complete and we have $\| v_t^i \|_{\infty} \leq \frac{1}{1 - \gamma}$ for all $t \geq 0$ .

# A.7.2 Proof of Lemma A.2

Observe that for any $x \in R^{d}$ and $j \in \{1, 2, \cdots, d\}$ , we have

$$
\begin{array}{l} \frac {\exp (x _ {j})}{\sum_ {\ell = 1} ^ {d} \exp (x _ {\ell})} = \frac {\exp (x _ {j})}{\exp (x _ {j}) + \sum_ {\ell \neq j} \exp (x _ {\ell})} \\ = \frac {1}{1 + \sum_ {\ell \neq j} \exp (x _ {\ell} - x _ {j})} \\ \geq \frac {1}{1 + (d - 1) \exp (2 \| x \| _ {\infty})}. \\ \end{array}
$$

Therefore, since $\| q_{t,k}^i\|_\infty \leq 1 / (1 - \gamma)$ for all $t,k\geq 0$ (cf. Lemma A.1), we have for all $t,k\geq 0$ and $(s,a^{i})$ that

$$
\sigma_ {\tau} (q _ {t, k} ^ {i} (s)) (a ^ {i}) \geq \frac {1}{1 + (A _ {\mathrm{max}} - 1) \exp (2 / [ (1 - \gamma) \tau ])} = \ell_ {\tau}. \tag {46}
$$

We next use induction to show that $\pi_{t,k}^{i}(a^{i}|s) \geq \ell_{\tau}$ for all $t, k \geq 0$ . Given any $t \geq 0$ , our uniform initialization of $\pi_{t,0}^{i}$ ensures that $\pi_{t,0}^{i}(a^{i}|s) \geq \ell_{\tau}$ for all $(s, a^{i})$ . Now suppose that $\pi_{t,k}^{i}(a^{i}|s) \geq \ell_{\tau}$ for all $(s, a^{i})$ for some $k \geq 0$ . Then we have from Algorithm 2 Line 4 that

$$
\begin{array}{l} \pi_ {t, k + 1} ^ {i} (a ^ {i} | s) = (1 - \beta_ {k}) \pi_ {t, k} ^ {i} (a ^ {i} | s) + \beta_ {k} \sigma_ {\tau} (q _ {t, k} ^ {i} (s)) (a ^ {i}) \\ \geq (1 - \beta_ {k}) \ell_ {\tau} + \beta_ {k} \ell_ {\tau} \\ = \ell_ {\tau}, \\ \end{array}
$$

where the inequality follows from Eq. (46) and the induction hypothesis. The induction is now complete and we have $\pi_{t,k}^{i}(a^{i}|s) \geq \ell_{\tau}$ for all $t, k \geq 0$ and $(s, a^{i})$ . Similarly, we also have $\pi_{t,k}^{-i}(a^{-i}|s) \geq \ell_{\tau}$ for all $t, k \geq 0$ and $(s, a^{-i})$ .

# A.7.3 Proof of Lemma 4.1

Lemma 4.1 (1), (3), and (4) are identical to (Zhang et al., 2022c, Proposition 3). We here only prove Lemma 4.1 (2).

Consider the Markov chain $\{S_k\}$ induced by $\pi_b$ . Since $\{S_k\}$ is irreducible and aperiodic, there exists a positive integer $r_b$ such that $P_{\pi_b}^{r_b}$ has strictly positive entries (Levin and Peres, 2017, Proposition 1.7). Therefore, there exists $\delta_b \in (0,1)$ such that

$$
P _ {\pi_ {b}} ^ {r _ {b}} (s, s ^ {\prime}) \geq \delta_ {b} \mu_ {b} (s ^ {\prime})
$$

for all $(s, s')$ . In addition, the constant $\rho_b$ introduced after Assumption 3.1 is explicitly given as $\rho_b = \exp(-\delta_b / r_b)$ . The previous two equations are from the proof of the Markov chain convergence theorem presented in (Levin and Peres, 2017, Section 4.3).

Next we consider the Markov chain $\{S_{k}\}$ induced by an arbitrary $\pi\in\Pi_{\delta}$ . Since

$$
\frac {\pi_ {b} (a | s)}{\pi (a | s)} = \frac {\pi_ {b} ^ {i} (a ^ {i} | s) \pi_ {b} ^ {- i} (a ^ {i} | s)}{\pi^ {i} (a ^ {i} | s) \pi^ {- i} (a ^ {i} | s)} \leq \frac {1}{\delta_ {i} \delta_ {- i}}, \quad \forall a = (a ^ {i}, a ^ {- i}) \text {and} s,
$$

we have for any $s, s' \in S$ and $k \geq 1$ that

$$
\begin{array}{l} P _ {\pi_ {b}} ^ {k} (s, s ^ {\prime}) = \sum_ {s _ {0}} P _ {\pi_ {b}} ^ {k - 1} (s, s _ {0}) P _ {\pi_ {b}} (s _ {0}, s ^ {\prime}) \\ = \sum_ {s _ {0}} P _ {\pi_ {b}} ^ {k - 1} (s, s _ {0}) \sum_ {a \in \mathcal {A}} \pi_ {b} (a | s _ {0}) P _ {a} (s _ {0}, s ^ {\prime}) \\ = \sum_ {s _ {0}} P _ {\pi_ {b}} ^ {k - 1} (s, s _ {0}) \sum_ {a \in \mathcal {A}} \frac {\pi_ {b} (a | s _ {0})}{\pi (a | s _ {0})} \pi (a | s _ {0}) P _ {a} (s _ {0}, s ^ {\prime}) \\ \leq \frac {1}{\delta_ {i} \delta_ {- i}} \sum_ {s _ {0}} P _ {\pi_ {b}} ^ {k - 1} (s, s _ {0}) \sum_ {a \in \mathcal {A}} \pi (a | s _ {0}) P _ {a} (s _ {0}, s ^ {\prime}) \\ \leq \frac {1}{\delta_ {i} \delta_ {- i}} \sum_ {s _ {0}} P _ {\pi_ {b}} ^ {k - 1} (s, s _ {0}) P _ {\pi} (s _ {0}, s ^ {\prime}) \\ = \frac {1}{\delta_ {i} \delta_ {- i}} [ P _ {\pi_ {b}} ^ {k - 1} P _ {\pi} ] (s, s ^ {\prime}). \\ \end{array}
$$

Since the previous inequality holds for all s and $s'$ , we in fact have $\delta_{i}\delta_{-i}P_{\pi_{b}}^{k} \leq P_{\pi_{b}}^{k-1}P_{\pi}$ (which is an entry-wise inequality). Repeatedly using the previous inequality and we obtain

$$
(\delta_ {i} \delta_ {- i}) ^ {k} P _ {\pi_ {b}} ^ {k} \leq P _ {\pi} ^ {k},
$$

which implies

$$
\begin{array}{l} P _ {\pi} ^ {r _ {b}} (s, s ^ {\prime}) \geq (\delta_ {i} \delta_ {- i}) ^ {r _ {b}} P _ {\pi_ {b}} ^ {r _ {b}} (s, s ^ {\prime}) \\ \geq \delta_ {b} (\delta_ {i} \delta_ {- i}) ^ {r _ {b}} \mu_ {b} (s ^ {\prime}) \\ \geq \delta_ {b} (\delta_ {i} \delta_ {- i}) ^ {r _ {b}} \frac {\mu_ {b} (s ^ {\prime})}{\mu_ {\pi} (s ^ {\prime})} \mu_ {\pi} (s ^ {\prime}) \\ \geq \delta_ {b} (\delta_ {i} \delta_ {- i}) ^ {r _ {b}} \mu_ {b, \min} \mu_ {\pi} (s ^ {\prime}). \\ \end{array}
$$

Following the proof of the Markov chain convergence theorem in (Levin and Peres, 2017, Section 4.3) and we have

$$
\left\| P _ {\pi} ^ {k} (s, \cdot) - \mu_ {\pi} (\cdot) \right\| _ {\mathrm{TV}} \leq \left(1 - \delta_ {b} \left(\delta_ {i} \delta_ {- i}\right) ^ {r _ {b}} \mu_ {b, \min}\right) ^ {k / r _ {b} - 1}, \quad \forall s \in \mathcal {S}, \pi \in \Pi_ {\delta}. \tag {47}
$$

Since $A_{\mathrm{max}} \geq 2$ (otherwise there is no decision to make in this Markov game), we have $\delta_i \delta_{-i} \leq \frac{1}{2}$ . It follows that $1 - \delta_b (\delta_i \delta_{-i})^{r_b} \mu_{b,\min} > 1/2$ . Using the previous inequality in Eq. (47) and we have

$$
\begin{array}{l} \sup _ {\pi \in \Pi_ {\delta}} \max _ {s \in \mathcal {S}} \| P _ {\pi} ^ {k} (s, \cdot) - \mu_ {\pi} (\cdot) \| _ {\mathrm{TV}} \leq 2 (1 - \delta_ {b} (\delta_ {i} \delta_ {- i}) ^ {r _ {b}} \mu_ {b, \min}) ^ {k / r _ {b}} \\ \leq 2 \exp \left(- \delta_ {b} (\delta_ {i} \delta_ {- i}) ^ {r _ {b}} \mu_ {b, \min} k / r _ {b}\right) \\ = 2 \rho_ {b} ^ {\left(\delta_ {i} \delta_ {- i}\right) ^ {r _ {b}} \mu_ {b, \min} k} \quad (\text { Recall   that } \rho_ {b} = \exp (- \delta_ {b} / r _ {b})) \\ = 2 \rho_ {\delta} ^ {k}. \\ \end{array}
$$

We next compute the mixing time. Using the previous inequality and the definition of the total variation distance, we have

$$
\sup _ {\pi \in \Pi_ {\delta}} \max _ {s \in \mathcal {S}} \| P _ {\pi} ^ {k} (s, \cdot) - \mu_ {\pi} (\cdot) \| _ {\mathrm{TV}} \leq \eta
$$

as long as

$$
k \geq \frac {\log (2 / \eta)}{\log (1 / \rho_ {\delta})} = \frac {1}{(\delta_ {i} \delta_ {- i}) ^ {r _ {b}} \mu_ {b , \min}} \frac {\log (2 / \eta)}{\log (1 / \rho_ {b})} \geq \frac {t _ {\pi_ {b} , \eta}}{(\delta_ {i} \delta_ {- i}) ^ {r _ {b}} \mu_ {b , \min}}.
$$

# A.7.4 Proof of Lemma A.3

Using the definition of utility functions and we have

$$
\begin{array}{l} \sum_ {i = 1, 2} \left(\max _ {\pi^ {i}} U ^ {i} (\pi^ {i}, \pi_ {T, K} ^ {- i}) - U ^ {i} (\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i})\right) \\ = \sum_ {i = 1, 2} \left(\max _ {\pi^ {i}} \mathbb {E} _ {S \sim p _ {o}} \left[ v _ {\pi^ {i}, \pi_ {T, K} ^ {- i}} ^ {i} (S) - v _ {\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}} ^ {i} (S) \right]\right) \\ \leq \sum_ {i = 1, 2} \left(\mathbb {E} _ {S \sim p _ {o}} \left[ \max _ {\pi^ {i}} v _ {\pi^ {i}, \pi_ {T, K} ^ {- i}} ^ {i} (S) - v _ {\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}} ^ {i} (S) \right]\right) \tag {$Jensen^{\prime$} s i n e q u a l i t y} \\ = \sum_ {i = 1, 2} \left(\mathbb {E} _ {S \sim p _ {o}} \left[ v _ {*, \pi_ {T, K} ^ {- i}} ^ {i} (S) - v _ {\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}} ^ {i} (S) \right]\right) \\ \leq \sum_ {i = 1, 2} \left\| v _ {*, \pi_ {T, K} ^ {- i}} ^ {i} - v _ {\pi_ {T, K} ^ {i}, \pi_ {T, K} ^ {- i}} ^ {i} \right\| _ {\infty}. \\ \end{array}
$$

# A.7.5 Proof of Lemma A.4

For any $t \geq 0$ , $s \in S$ , and $i \in \{1, 2\}$ , we have

$$
\begin{array}{l} 0 \leq \left| v _ {*, \pi_ {t, K} ^ {- i}} ^ {i} (s) - v _ {\pi_ {t, K} ^ {i}, \pi_ {t, K} ^ {- i}} ^ {i} (s) \right| \\ = v _ {*, \pi_ {t, K} ^ {- i}} ^ {i} (s) - v _ {\pi_ {t, K} ^ {i}, \pi_ {t, K} ^ {- i}} ^ {i} (s) \\ \leq v _ {*, \pi_ {t, K} ^ {- i}} ^ {i} (s) - v _ {\pi_ {t, K} ^ {i}, *} ^ {i} (s) \\ = - v _ {\pi_ {t, K} ^ {- i}, *} ^ {- i} (s) - v _ {\pi_ {t, K} ^ {i}, *} ^ {i} (s) \\ = v _ {*} ^ {i} (s) - v _ {\pi_ {t, K} ^ {- i}, *} ^ {- i} (s) + v _ {*} ^ {- i} (s) - v _ {\pi_ {t, K} ^ {i}, *} ^ {i} (s) \\ \leq \left\| v _ {*} ^ {- i} - v _ {\pi_ {t, K} ^ {- i}, *} ^ {- i} \right\| _ {\infty} + \left\| v _ {*} ^ {i} - v _ {\pi_ {t, K} ^ {i}, *} ^ {i} \right\| _ {\infty}. \tag {48} \\ \end{array}
$$

It remains to bound the two terms on the RHS of the previous inequality. For the first term, note that for any $s \in S$ and $t \geq 0$ , we have

$$
\begin{array}{l} 0 \leq v _ {*} ^ {- i} (s) - v _ {\pi_ {t, K} ^ {- i}, *} ^ {- i} (s) \\ = v _ {*, \pi_ {t, K} ^ {- i}} ^ {i} (s) - v _ {*} ^ {i} (s) \\ = \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*, \pi_ {t, K} ^ {- i}} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - \max _ {\mu^ {i}} \min _ {\mu^ {- i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s) \mu^ {- i} \\ = | \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*, \pi_ {t, K} ^ {- i}} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) | \\ + | \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - \max _ {\mu^ {i}} \min _ {\mu^ {- i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s) \mu^ {- i} | \\ \leq \max _ {\mu^ {i}} | (\mu^ {i}) ^ {\top} (\mathcal {T} ^ {i} (v _ {*, \pi_ {t, K} ^ {- i}} ^ {i}) (s) - \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s)) \pi_ {t, K} ^ {- i} (s) | \\ + | \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - \max _ {\mu^ {i}} \min _ {\mu^ {- i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \mu^ {- i} | \\ + \left| \max _ {\mu^ {i}} \min _ {\mu^ {- i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \mu^ {- i} - \max _ {\mu^ {i}} \min _ {\mu^ {- i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s) \mu^ {- i} \right| \\ \leq \underbrace {\max _ {\mu^ {i}} | (\mu^ {i}) ^ {\top} (\mathcal {T} ^ {i} (v _ {* , \pi_ {t , K} ^ {- i}} ^ {i}) (s) - \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s)) \pi_ {t , K} ^ {- i} (s) |} _ {\hat {E} _ {1}} \\ + \underbrace {| \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {*} ^ {i}) (s) \pi_ {t , K} ^ {- i} (s) - \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t , K} ^ {- i} (s) |} _ {\hat {E} _ {2}} \\ + \underbrace {\max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t , K} ^ {- i} (s) - \max _ {\mu^ {i}} \min _ {\mu^ {- i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \mu^ {- i}} _ {\hat {E} _ {3}} \\ + \underbrace {\left| \max _ {\mu^ {i}} \min _ {\mu^ {- i}} \left(\mu^ {i}\right) ^ {\top} \mathcal {T} ^ {i} \left(v _ {t} ^ {i}\right) (s) \mu^ {- i} - \max _ {\mu^ {i}} \min _ {\mu^ {- i}} \left(\mu^ {i}\right) ^ {\top} \mathcal {T} ^ {i} \left(v _ {*} ^ {i}\right) (s) \mu^ {- i} \right\rvert} _ {\hat {E} _ {4}}. \tag {49} \\ \end{array}
$$

We next bound the terms $\{\hat{E}_j\}_{1\leq j\leq 4}$ . For any $v_{1}^{i}, v_{2}^{i} \in \mathbb{R}^{|S|}$ , we have for any $(s, a^{i}, a^{-i})$ that

$$
| \mathcal {T} ^ {i} (v _ {1} ^ {i}) (s, a ^ {i}, a ^ {- i}) - \mathcal {T} ^ {i} (v _ {2} ^ {i}) (s, a ^ {i}, a ^ {- i}) |
$$

$$
= \gamma | \mathbb {E} [ v _ {1} ^ {i} (S _ {1}) - v _ {2} ^ {i} (S _ {1}) \mid S _ {0} = s, A _ {0} ^ {i} = a ^ {i}, A _ {0} ^ {- i} = a ^ {- i} ] |
$$

$$
\leq \gamma \| v _ {1} ^ {i} - v _ {2} ^ {i} \| _ {\infty},
$$

which implies $\|\mathcal{T}^{i}(v_{1}^{i})-\mathcal{T}^{i}(v_{2}^{i})\|_{\infty}\leq\gamma\|v_{1}^{i}-v_{2}^{i}\|_{\infty}$ . As a result, we have

$$
\hat {E} _ {1} \leq \| \mathcal {T} ^ {i} (v _ {*, \pi_ {t, K} ^ {- i}} ^ {i}) - \mathcal {T} ^ {i} (v _ {*} ^ {i}) \| _ {\infty} \leq \gamma \| v _ {*, \pi_ {t, K} ^ {- i}} ^ {i} - v _ {*} ^ {i} \| _ {\infty},
$$

$$
\hat {E} _ {2} \leq \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) - \mathcal {T} ^ {i} (v _ {*} ^ {i}) \| _ {\infty} \leq \gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty},
$$

$$
\hat {E} _ {4} \leq \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) - \mathcal {T} ^ {i} (v _ {*} ^ {i}) \| _ {\infty} \leq \gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty}.
$$

Bounding the term $\hat{E}_3$ requires more effort. First observe that

$$
\begin{array}{l} \hat {E} _ {3} \leq \left| \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - \min _ {\mu^ {- i}} \pi_ {t, K} ^ {i} (s) \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \mu^ {- i} \right| \\ \leq \left| \max _ {\mu^ {- i}} (\mu^ {- i}) ^ {\top} \mathcal {T} ^ {- i} (v _ {t} ^ {- i}) (s) \pi_ {t, K} ^ {i} (s) + \min _ {\mu^ {- i}} (\mu^ {- i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) ^ {\top} \pi_ {t, K} ^ {i} (s) \right| \\ + \left| \sum_ {i = 1, 2} \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ \leq \left| \max _ {\mu^ {- i}} (\mu^ {- i}) ^ {\top} \mathcal {T} ^ {- i} (v _ {t} ^ {- i}) (s) \pi_ {t, K} ^ {i} (s) - \max _ {\mu^ {- i}} (\mu^ {- i}) ^ {\top} [ - \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) ] ^ {\top} \pi_ {t, K} ^ {i} (s) \right| \\ + \left| \sum_ {i = 1, 2} \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ \leq \max _ {\mu^ {- i}} \big | (\mu^ {- i}) ^ {\top} (\mathcal {T} ^ {- i} (v _ {t} ^ {- i}) (s) + [ \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) ] ^ {\top}) \pi_ {t, K} ^ {i} (s) \big | \\ + \left| \sum_ {i = 1, 2} \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ \leq \max _ {a ^ {i}, a ^ {- i}} \left| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s, a ^ {i}, a ^ {- i}) + \mathcal {T} ^ {- i} (v _ {t} ^ {- i}) (s, a ^ {i}, a ^ {- i}) \right| \\ + \left| \sum_ {i = 1, 2} \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ \leq \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + \left| \sum_ {i = 1, 2} \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right|, \\ \end{array}
$$

where the last line follows from

$$
\begin{array}{l} | \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s, a ^ {i}, a ^ {- i}) + \mathcal {T} ^ {- i} (v _ {t} ^ {- i}) (s, a ^ {i}, a ^ {- i}) | \\ = \gamma | \mathbb {E} [ v _ {t} ^ {i} (S _ {1}) + v _ {t} ^ {i} (S _ {1}) \mid S _ {0} = s, A _ {0} ^ {i} = a ^ {i}, A _ {0} ^ {- i} = a ^ {- i} ] | \\ \leq \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty}. \\ \end{array}
$$

In addition, we have

$$
\begin{array}{l} \left| \sum_ {i = 1, 2} \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ = \left| \sum_ {i = 1, 2} \left\{\max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - (\pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right\} \right| \\ + \left| \sum_ {i = 1, 2} (\pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ = \sum_ {i = 1, 2} \left\{\max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - (\pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right\} \\ \end{array}
$$

$$
\begin{array}{l} + \left| \sum_ {i = 1, 2} (\pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ \leq \sum_ {i = 1, 2} \max _ {\mu^ {i}} \left\{\left(\mu^ {i} - \pi_ {t, K} ^ {i} (s)\right) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) + \tau \nu (\mu^ {i}) - \tau \nu (\pi_ {t, K} ^ {i} (s)) \right\} + 2 \tau \log (A _ {\max}) \\ + \left| \sum_ {i = 1, 2} (\pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ = V _ {v _ {t}, s} \left(\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)\right) + 2 \tau \log \left(A _ {\max}\right) + \left| \sum_ {i = 1, 2} \left(\pi_ {t, K} ^ {i} (s)\right) ^ {\top} \mathcal {T} ^ {i} \left(v _ {t} ^ {i}\right) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ \leq V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\max}) \\ + \max _ {a ^ {i}, a ^ {- i}} \left| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s, a ^ {i}, a ^ {- i}) + \mathcal {T} ^ {- i} (v _ {t} ^ {- i}) (s, a ^ {i}, a ^ {- i}) \right| \\ \leq V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\max}) + \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty}. \\ \end{array}
$$

It follows that

$$
\hat {E} _ {3} \leq 2 \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + \max _ {s} V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\max}). \tag {50}
$$

Substituting the upper bounds we obtained for the terms $\{E_j\}_{1\leq j\leq 4}$ into Eq. (49) and we have

$$
\begin{array}{l} \| v _ {*} ^ {- i} - v _ {\pi_ {t, K} ^ {- i}, *} ^ {- i} \| _ {\infty} \leq \gamma \| v _ {*, \pi_ {t, K} ^ {- i}} ^ {i} - v _ {*} ^ {i} \| _ {\infty} + 2 \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + 2 \gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \\ + \max _ {s} V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\mathrm{max}}) \\ = \gamma \| v _ {*} ^ {- i} - v _ {\pi_ {t, K} ^ {- i}, *} ^ {- i} \| _ {\infty} + 2 \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + 2 \gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \\ + \max _ {s} V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\mathrm{max}}), \\ \end{array}
$$

which after rearranging terms implies

$$
\begin{array}{l} \| v _ {*} ^ {- i} - v _ {\pi_ {t, K} ^ {- i}, *} ^ {- i} \| _ {\infty} \leq \frac {1}{1 - \gamma} \bigg (2 \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + 2 \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \\ \left. + \max _ {s} V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\max})\right). \\ \end{array}
$$

Similarly, we also have

$$
\begin{array}{l} \| v _ {*} ^ {i} - v _ {\pi_ {t, K} ^ {i}, *} ^ {i} \| _ {\infty} \leq \frac {1}{1 - \gamma} \bigg (2 \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + 2 \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \\ \left. + \max _ {s} V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\max})\right). \\ \end{array}
$$

Substituting the previous two inequalities into Eq. (48) and we finally obtain

$$
\begin{array}{l} \left\| v _ {*, \pi_ {t, K} ^ {- i}} ^ {i} - v _ {\pi_ {t, K} ^ {i}, \pi_ {t, K} ^ {- i}} ^ {i} \right\| _ {\infty} \leq \frac {2}{1 - \gamma} \left(2 \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + 2 \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \right. \\ \left. + \max _ {s} V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\mathrm{max}})\right). \\ \end{array}
$$

# A.7.6 Proof of Lemma A.5

For any $i \in \{1, 2\}$ , we have by the outer-loop update equation (cf. Line 8) of Algorithm 2 that

$$
\begin{array}{l} v _ {t + 1} ^ {i} (s) = \pi_ {t, K} ^ {i} (s) ^ {\top} q _ {t, K} ^ {i} (s) \\ = v a l ^ {i} (\mathcal {T} ^ {i} (v _ {t} ^ {i}) (s)) + \pi_ {t, K} ^ {i} (s) ^ {\top} q _ {t, K} ^ {i} (s) - v a l ^ {i} (\mathcal {T} ^ {i} (v _ {t} ^ {i}) (s)) \\ \end{array}
$$

Since $val^{i}(\mathcal{T}^{i}(v_{*}^{i})(s)) = \mathcal{B}^{i}(v_{*}^{i})(s) = v_{*}^{i}(s)$ , we have

$$
\begin{array}{l} | v _ {t + 1} ^ {i} (s) - v _ {*} ^ {i} (s) | = | v a l ^ {i} (\mathcal {T} ^ {i} (v _ {t} ^ {i}) (s)) - v a l ^ {i} (\mathcal {T} ^ {i} (v _ {*} ^ {i}) (s)) | \\ + \left| \pi_ {t, K} ^ {i} (s) ^ {\top} q _ {t, K} ^ {i} (s) - v a l ^ {i} (\mathcal {T} ^ {i} (v _ {t} ^ {i}) (s)) \right|. \tag {51} \\ \end{array}
$$

For the first term on the RHS of Eq. (51), we have by the contraction property of the minimax Bellman operator that

$$
\left| v a l ^ {i} \left(\mathcal {T} ^ {i} \left(v _ {t} ^ {i}\right) (s)\right) - v a l ^ {i} \left(\mathcal {T} ^ {i} \left(v _ {*} ^ {i}\right) (s)\right) \right| = \left| \mathcal {B} ^ {i} \left(v _ {t} ^ {i}\right) (s) - \mathcal {B} ^ {i} \left(v _ {*} ^ {i}\right) (s) \right|
$$

$$
\leq \gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty}.
$$

For the second term on the RHS of Eq. (51), we have

$$
\begin{array}{l} \left| \pi_ {t, K} ^ {i} (s) ^ {\top} q _ {t, K} ^ {i} (s) - v a l ^ {i} (\mathcal {T} ^ {i} (v _ {t} ^ {i}) (s)) \right| \leq \underbrace {\left| \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t , K} ^ {- i} (s) - \pi_ {t , K} ^ {i} (s) ^ {\top} q _ {t , K} ^ {i} (s) \right|} _ {T _ {1}} \\ + \underbrace {\left| \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t , K} ^ {- i} (s) - v a l ^ {i} (\mathcal {T} ^ {i} (v _ {t} ^ {i}) (s)) \right|} _ {T _ {2}} \\ \end{array}
$$

For the term $T_{1}$ , we have

$$
\begin{array}{l} T _ {1} \leq \left| \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - (\pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ + \left| (\pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - \pi_ {t, K} ^ {i} (s) ^ {\top} q _ {t, K} ^ {i} (s) \right| \\ \leq \max _ {\mu^ {i}} (\mu^ {i}) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - (\pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \\ + \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - q _ {t, K} ^ {i} (s) \| _ {\infty} \\ \leq \sum_ {i = 1, 2} \left\{\max _ {\mu^ {i}} (\mu^ {i} - \pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right\} \\ + \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - q _ {t, K} ^ {i} (s) \| _ {\infty} \\ \leq \sum_ {i = 1, 2} \left\{\max _ {\mu^ {i}} (\mu^ {i} - \pi_ {t, K} ^ {i} (s)) ^ {\top} \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) + \tau \nu (\mu^ {i}) - \tau \nu (\pi_ {t, K} ^ {i} (s)) \right\} \\ + 2 \tau \log (A _ {\max}) + \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - q _ {t, K} ^ {i} (s) \| _ {\infty} \\ \leq V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\max}) + \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - q _ {t, K} ^ {i} (s) \| _ {\infty}. \\ \end{array}
$$

Note that $T_{2}$ is exactly the term $\hat{E}_3$ we analyzed in proving Lemma A.4. Therefore, we have from Eq. (50) that

$$
T _ {2} \leq 2 \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + \max _ {s} V _ {v _ {t}, s} (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 2 \tau \log (A _ {\mathrm{max}}).
$$

It follows that

$$
\left| \pi_ {t, K} ^ {i} (s) ^ {\top} q _ {t, K} ^ {i} (s) - v a l ^ {i} (\mathcal {T} ^ {i} (v _ {t} ^ {i}) (s)) \right|
$$

$$
\begin{array}{l} \leq T _ {1} + T _ {2} \\ \leq 2 \max _ {s} V (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + \max _ {s} \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - q _ {t, K} ^ {i} (s) \| _ {\infty} \\ + 2 \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + 4 \tau \log (A _ {\mathrm{max}}). \\ \end{array}
$$

Using the upper bounds we obtained for the two terms on the RHS of Eq. (51) and we have

$$
\begin{array}{l} | v _ {t + 1} ^ {i} (s) - v _ {*} ^ {i} (s) | \leq \gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} + 2 \max _ {s \in \mathcal {S}} V (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 4 \tau \log (A _ {\mathrm{max}}) \\ + \max _ {s \in \mathcal {S}} \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - q _ {t, K} ^ {i} (s) \| _ {\infty} + 2 \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty}. \\ \end{array}
$$

Since the RHS of the previous inequality does not depend on $s$ , we have for any $i \in \{1, 2\}$ that

$$
\| v _ {t + 1} ^ {i} - v _ {*} ^ {i} \| _ {\infty} \leq \gamma \| v _ {t} ^ {i} - v _ {*} ^ {i} \| _ {\infty} + 2 \max _ {s \in \mathcal {S}} V (\pi_ {t, K} ^ {i} (s), \pi_ {t, K} ^ {- i} (s)) + 4 \tau \log (A _ {\max})
$$

$$
+ \max _ {s \in \mathcal {S}} \| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) - q _ {t, K} ^ {i} (s) \| _ {\infty} + 2 \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty}.
$$

# A.7.7 Proof of Lemma A.6

Using the outer-loop update equation (cf. Algorithm 2 Line 8) and we have

$$
\begin{array}{l} \left| \sum_ {i = 1, 2} v _ {t + 1} ^ {i} (s) \right| = \sum_ {i = 1, 2} \pi_ {t, K} ^ {i} (s) ^ {\top} q _ {t, K} ^ {i} (s) \\ = \left| \sum_ {i = 1, 2} \pi_ {t, K} ^ {i} (s) ^ {\top} (q _ {t, K} ^ {i} (s) - \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s)) \right| \\ + \left| \sum_ {i = 1, 2} \pi_ {t, K} ^ {i} (s) \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \right| \\ \leq \sum_ {i = 1, 2} \max _ {s \in \mathcal {S}} \| q _ {t, K} ^ {i} (s) - \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \| _ {\infty} \\ + \max _ {(s, a ^ {i}, a ^ {- i})} \left| \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s, a ^ {i}, a ^ {- i}) + \mathcal {T} ^ {- i} (v _ {t} ^ {- i}) (s, a ^ {i}, a ^ {- i}) \right| \\ \leq \sum_ {i = 1, 2} \max _ {s \in \mathcal {S}} \| q _ {t, K} ^ {i} (s) - \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \| _ {\infty} + \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty}. \\ \end{array}
$$

Since the RHS of the previous inequality does not depend on s, we in fact have

$$
\| v _ {t + 1} ^ {i} + v _ {t + 1} ^ {- i} \| _ {\infty} \leq \gamma \| v _ {t} ^ {i} + v _ {t} ^ {- i} \| _ {\infty} + \sum_ {i = 1, 2} \max _ {s \in \mathcal {S}} \| q _ {t, K} ^ {i} (s) - \mathcal {T} ^ {i} (v _ {t} ^ {i}) (s) \pi_ {t, K} ^ {- i} (s) \| _ {\infty}.
$$

# A.7.8 Proof of Lemma A.7

To begin with, observe that

$$
\arg \max _ {\hat {\mu} ^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \left\{(\hat {\mu} ^ {i}) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu (\hat {\mu} ^ {i}) \right\} = \sigma_ {\tau} (X _ {i} \mu^ {- i}).
$$

Therefore, the function $V_{X}(\cdot,\cdot)$ can be equivalently written as

$$
V _ {X} (\mu^ {i}, \mu^ {- i}) = \sum_ {i = 1, 2} \left[ (\sigma_ {\tau} (X _ {i} \mu^ {- i})) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})) - (\mu^ {i}) ^ {\top} X _ {i} \mu^ {- i} - \tau \nu (\mu^ {i}) \right],
$$

which will frequently used in our analysis.

(1) It is clear that the function $V_{X}(\cdot,\cdot)$ is by definition non-negative. The strong convexity follows from the following two observations.

(i) The negative entropy $-\nu(\cdot)$ is $1-$ strongly convex with respect to $\|\cdot\|_{2}$ (Beck, 2017, Example 5.27).

(ii) The following function of $\mu^{i}$

$$
(\sigma_ {\tau} (X _ {- i} \mu^ {i})) ^ {\top} X _ {- i} \mu^ {i} + \tau \nu (\sigma_ {\tau} (X _ {- i} \mu^ {i})) = \max _ {\hat {\mu} ^ {- i} \in \Delta^ {| A ^ {- i} |}} \bigl \{(\hat {\mu} ^ {- i}) ^ {\top} X _ {- i} \mu^ {i} + \tau \nu (\hat {\mu} ^ {- i}) \bigr \}
$$

is the maximum of linear functions in $\mu^{i}$ , and hence is convex.

Therefore, the function $V_{X}(\mu^{i},\mu^{-i})$ is $\tau -$ strongly convex in $\mu^{i}$ with respect to $\| \cdot \|_2$ uniformly for all $\mu^{-i}$ .

(2) The smoothness follows from the following two results.

(i) Since the Hessian matrix of the negative entropy function $-\nu(\mu^{i})$ satisfies

$$
H _ {- \nu} (\mu^ {i}) = \mathrm{diag} \left(\frac {1}{\mu^ {i} (a _ {1} ^ {i})}, \dots , \frac {1}{\mu^ {i} (a _ {n} ^ {i})}\right) \leq \frac {I _ {| \mathcal {A} ^ {i} |}}{\delta_ {i}}
$$

for all $\mu^i\in \Delta^{|\mathcal{A}^i|}$ satisfying $\min_{a^i\in \mathcal{A}^i}\mu^i (a^i)\geq \delta_i$ , we have from the second order characterization of smoothness that $-\nu (\mu^i)$ is a $\frac{1}{\delta_i}$ - smooth function on $\{\mu^i\in \Delta^{|\mathcal{A}^i|}\mid$ $\mu^i (a^i)\geq \delta_i,\forall   a^i\in \mathcal{A}^i\}$ with respect to $\| \cdot \| _2$ .

(ii) Using the optimality condition and we have

$$
\begin{array}{l} \nabla_ {\mu^ {i}} (\sigma_ {\tau} (X _ {- i} \mu^ {i})) ^ {\top} X _ {- i} \mu^ {i} + \tau \nu (\sigma_ {\tau} (X _ {- i} \mu^ {i})) \\ = \nabla_ {\mu^ {i}} \max _ {\hat {\mu} ^ {- i} \in \Delta^ {| \mathcal {A} ^ {- i} |}} \left\{\left(\hat {\mu} ^ {- i}\right) ^ {\top} X _ {- i} \mu^ {i} + \tau \nu (\hat {\mu} ^ {- i}) \right\} \\ = X _ {- i} ^ {\top} \sigma_ {\tau} (X _ {- i} \mu^ {i}). \\ \end{array}
$$

Therefore, using the formula for the gradient of the softmax function (Gao and Pavel, 2017), the Hessian $H(\mu^i)$ of the function $(\sigma_{\tau}(X_{-i}\mu^i))^{\top}X_{-i}\mu^i + \tau \nu (\sigma_{\tau}(X_{-i}\mu^i))$ satisfies

$$
\begin{array}{l} H (\mu^ {i}) = \frac {1}{\tau} X _ {- i} ^ {\top} (\mathrm{diag} (\sigma_ {\tau} (X _ {- i} \mu^ {i})) - \sigma_ {\tau} (X _ {- i} \mu^ {i}) \sigma_ {\tau} (X _ {- i} \mu^ {i}) ^ {\top}) X _ {- i} \\ \leq \frac {1}{\tau} X _ {- i} ^ {\top} \mathrm{diag} (\sigma_ {\tau} (X _ {- i} \mu^ {i})) X _ {- i} \\ \leq \frac {1}{\tau} X _ {- i} ^ {\top} X _ {- i} \\ \leq \frac {\sigma_ {\mathrm{max}} ^ {2} (X _ {- i})}{\tau} I _ {| \mathcal {A} ^ {i} |} \\ \end{array}
$$

Using the second order characterization of smoothness and we conclude that the function $(\sigma_{\tau}(X_{-i}\mu^{i}))^{\top}X_{-i}\mu^{i} + \tau\nu(\sigma_{\tau}(X_{-i}\mu^{i}))$ is $\frac{\sigma_{\max}^{2}(X_{-i})}{\tau} - \text{smooth with respect to } \|\cdot\|_{2}$ .

Combining (i) and (ii) and we conclude that the function $V_{X}(\mu^{i},\mu^{-i})$ is a $\left(\frac{\sigma_{\max}^{2}(X_{-i})}{\tau}+\frac{\tau}{\delta_{i}}\right)$ – smooth function on $\{\mu^{i}\in\Delta^{|A^{i}|}\mid\mu^{i}(a^{i})\geq\delta_{i},\forall a^{i}\in\mathcal{A}^{i}\}$ with respect to $\|\cdot\|_{2}$ uniformly for all $\mu^{-i}$ .

(3) We first compute the gradient $\nabla_{1}V_{X}\left(\mu^{i},\mu^{-i}\right)$ using Danskin's theorem:

$$
\nabla_ {1} V _ {X} (\mu^ {i}, \mu^ {- i}) = - (X _ {i} + X _ {- i} ^ {\top}) \mu^ {- i} - \tau \nabla \nu (\mu^ {i}) + X _ {- i} ^ {\top} \sigma_ {\tau} (X _ {- i} \mu^ {i}). \tag {52}
$$

It follows that

$$
\begin{array}{l} \left\langle \nabla_ {1} V _ {X} \left(\mu^ {i}, \mu^ {- i}\right), \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) - \mu^ {i} \right\rangle \\ = \left\langle - \left(X _ {i} + X _ {- i} ^ {\top}\right) \mu^ {- i} - \tau \nabla \nu \left(\mu^ {i}\right) + X _ {- i} ^ {\top} \sigma_ {\tau} \left(X _ {- i} \mu^ {i}\right), \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) - \mu^ {i} \right\rangle \\ = \langle - (X _ {i} + X _ {- i} ^ {\top}) \mu^ {- i} - \tau \nabla \nu (\mu^ {i}) + X _ {- i} ^ {\top} \sigma_ {\tau} (X _ {- i} \mu^ {i}), \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \rangle \\ + \left\langle X _ {i} \mu^ {- i} + \tau \nabla \nu \left(\sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right)\right), \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) - \mu^ {i} \right\rangle \tag {53} \\ = \tau \langle \nabla \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})) - \nabla \nu (\mu^ {i}), \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \rangle \\ + \left(\sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i}\right) ^ {\top} X _ {- i} (\sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i}). \\ \end{array}
$$

where Eq. (53) is due to the optimality condition $X_{i}\mu^{-i} + \tau\nabla\nu(\sigma_{\tau}(X_{i}\mu^{-i})) = 0$ . To proceed, observe that the concavity of $\nu(\cdot)$ and the optimality condition $X_{i}\mu^{-i} + \tau\nabla\nu(\sigma_{\tau}(X_{i}\mu^{-i})) = 0$ together imply that

$$
\begin{array}{l} \langle \nabla \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})) - \nabla \nu (\mu^ {i}), \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \rangle \\ = \langle \nabla \nu (\mu^ {i}) - \nabla \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})), \mu^ {i} - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \rangle \\ = \left\langle \nabla \nu \left(\mu^ {i}\right), \mu^ {i} - \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) \right\rangle - \left\langle \nabla \nu \left(\sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right)\right), \mu^ {i} - \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) \right\rangle \\ \leq \nu (\mu^ {i}) - \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})) - \langle \nabla \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})), \mu^ {i} - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \rangle \\ = \nu (\mu^ {i}) - \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})) + \frac {1}{\tau} \langle X _ {i} \mu^ {- i}, \mu^ {i} - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \rangle \\ = \frac {1}{\tau} \left[ (\mu^ {i}) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu (\mu^ {i}) - \max _ {\hat {\mu} ^ {i} \in \Delta^ {| \mathcal {A} ^ {i} |}} \bigl \{(\hat {\mu} ^ {i}) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu (\hat {\mu} ^ {i}) \bigr \} \right]. \\ \end{array}
$$

Therefore, we have

$$
\begin{array}{l} \left\langle \nabla_ {1} V _ {X} \left(\mu^ {i}, \mu^ {- i}\right), \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) - \mu^ {i} \right\rangle \\ \leq \left[ (\mu^ {i}) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu (\mu^ {i}) - \max _ {\hat {\mu} ^ {i} \in \Delta | \mathcal {A} ^ {i} |} \bigl \{(\hat {\mu} ^ {i}) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu (\hat {\mu} ^ {i}) \bigr \} \right] \\ + \left(\sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i}\right) ^ {\top} X _ {- i} (\sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i}) \\ \end{array}
$$

Similarly, we also have

$$
\begin{array}{l} \left\langle \nabla_ {2} V _ {X} \left(\mu^ {i}, \mu^ {- i}\right), \sigma_ {\tau} \left(X _ {- i} \mu^ {i}\right) - \mu^ {- i} \right\rangle \\ \leq \left[ (\mu^ {- i}) ^ {\top} X _ {- i} \mu^ {i} + \tau \nu (\mu^ {- i}) - \max _ {\hat {\mu} ^ {- i} \in \Delta^ {| \mathcal {A} ^ {- i} |}} \bigl \{(\hat {\mu} ^ {- i}) ^ {\top} X _ {- i} \mu^ {i} + \tau \nu (\hat {\mu} ^ {- i}) \bigr \} \right] \\ + \left(\sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i}\right) ^ {\top} X _ {i} (\sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i}) \\ \end{array}
$$

Adding up the previous two inequalities and we obtain

$$
\begin{array}{l} \langle \nabla_ {1} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \rangle + \langle \nabla_ {2} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \rangle \\ \leq - V _ {X} \left(\mu^ {i}, \mu^ {- i}\right) + \left(\sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) - \mu^ {i}\right) ^ {\top} \left(X _ {i} + X _ {- i} ^ {\top}\right) \left(\sigma_ {\tau} \left(X _ {- i} \mu^ {i}\right) - \mu^ {- i}\right). \tag {54} \\ \end{array}
$$

To control the second term on the RHS of Eq. (54), observe that

$$
\begin{array}{l} \left(\sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i}\right) ^ {\top} (X _ {i} + X _ {- i} ^ {\top}) (\sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i}) \\ \leq \| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} \| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} \\ \leq \left(\| \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) \| _ {2} + \| \mu^ {i} \| _ {2}\right) \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} \| \sigma_ {\tau} \left(X _ {- i} \mu^ {i}\right) - \mu^ {- i} \| _ {2} \\ \leq 2 \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} \| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} \\ \leq c _ {1} \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} ^ {2} + \frac {1}{c _ {1}} \| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} ^ {2} \quad (\text {This is true for all} c _ {1} > 0) \\ \end{array}
$$

$$
\leq c _ {1} \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} ^ {2} + \frac {1}{c _ {1}} (\| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} ^ {2} + \| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} ^ {2}). \tag {55}
$$

To proceed, note that the function

$$
F _ {X _ {i}} (\mu^ {i}, \mu^ {- i}) := \max _ {\hat {\mu} ^ {i}} \left\{(\hat {\mu} ^ {i} - \mu^ {i}) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu (\hat {\mu} ^ {i}) - \tau \nu (\mu^ {i}) \right\}
$$

is a $\tau$ -strongly convex function of $\mu^i$ uniformly for all $\mu^{-i}$ . Therefore, we have

$$
\begin{array}{l} F _ {X _ {i}} (\mu^ {i}, \mu^ {- i}) = F _ {X _ {i}} (\mu^ {i}, \mu^ {- i}) - F _ {X _ {i}} (\sigma_ {\tau} (X _ {i} \mu^ {- i}), \mu^ {- i}) \\ = F _ {X _ {i}} \left(\mu^ {i}, \mu^ {- i}\right) - \min _ {\mu^ {i}} F _ {X _ {i}} \left(\mu^ {i}, \mu^ {- i}\right) \\ \geq \frac {\tau}{2} \| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} ^ {2}, \\ \end{array}
$$

which is called the quadratic growth property in optimization literature. It follows that

$$
\| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} ^ {2} \leq \frac {2}{\tau} \max _ {\hat {\mu} ^ {i}} \left\{(\hat {\mu} ^ {i} - \mu^ {i}) ^ {\top} X _ {i} \mu^ {- i} + \tau \nu (\hat {\mu} ^ {i}) - \tau \nu (\mu^ {i}) \right\}.
$$

Similarly, we also have

$$
\| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} ^ {2} \leq \frac {2}{\tau} \max _ {\hat {\mu} ^ {- i}} \left\{(\hat {\mu} ^ {- i} - \mu^ {- i}) ^ {\top} X _ {- i} \mu^ {i} + \tau \nu (\hat {\mu} ^ {- i}) - \tau \nu (\mu^ {- i}) \right\}.
$$

Adding up the previous two inequalities and we have

$$
\| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} ^ {2} + \| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} ^ {2} \leq \frac {2}{\tau} V _ {X} (\mu^ {i}, \mu^ {- i}).
$$

Using the previous inequality in Eq. (55) and we have

$$
\begin{array}{l} \left(\sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) - \mu^ {i}\right) ^ {\top} \left(X _ {i} + X _ {- i} ^ {\top}\right) \left(\sigma_ {\tau} \left(X _ {- i} \mu^ {i}\right) - \mu^ {- i}\right) \\ \leq c _ {1} \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} ^ {2} + \frac {1}{c _ {1}} (\| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} ^ {2} + \| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} ^ {2}) \\ \leq c _ {1} \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} ^ {2} + \frac {2}{c _ {1} \tau} V _ {X} (\mu^ {i}, \mu^ {- i}) \\ = \frac {1 6}{\tau} \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} ^ {2} + \frac {1}{8} V _ {X} (\mu^ {i}, \mu^ {- i}), \\ \end{array}
$$

where the last line follows from choosing $c_{1} = 16 / \tau$ . Using the previous inequality in Eq. (54) and we obtain

$$
\begin{array}{l} \langle \nabla_ {1} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \rangle + \langle \nabla_ {2} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \rangle \\ \leq - \frac {7}{8} V _ {X} (\mu^ {i}, \mu^ {- i}) + \frac {1 6}{\tau} \| X _ {i} + X _ {- i} ^ {\top} \| _ {2} ^ {2}. \\ \end{array}
$$

(4) For any $u^i \in \mathbb{R}^{|\mathcal{A}^i|}$ , using the explicit expression of the gradient of $V_X(\cdot, \cdot)$ from Eq. (52) and we have

$$
\begin{array}{l} \langle \nabla_ {1} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \rangle \\ = \left\langle - \left(X _ {i} + X _ {- i} ^ {\top}\right) \mu^ {- i} - \tau \nabla \nu \left(\mu^ {i}\right) + X _ {- i} ^ {\top} \sigma_ {\tau} \left(X _ {- i} \mu^ {i}\right), \sigma_ {\tau} \left(u ^ {i}\right) - \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) \right\rangle \\ = \left\langle - \left(X _ {i} + X _ {- i} ^ {\top}\right) \mu^ {- i} - \tau \nabla \nu \left(\mu^ {i}\right) + X _ {- i} ^ {\top} \sigma_ {\tau} \left(X _ {- i} \mu^ {i}\right), \sigma_ {\tau} \left(u ^ {i}\right) - \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) \right\rangle \\ + \langle X _ {i} \mu^ {- i} + \tau \nabla \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})), \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \rangle \\ = \tau \langle \nabla \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})) - \nabla \nu (\mu^ {i}), \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \rangle \\ \end{array}
$$

$$
\begin{array}{l} + \left(\sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i}\right) ^ {\top} X _ {- i} (\sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i})) \\ \leq \tau \| \nabla \nu (\sigma_ {\tau} (X _ {i} \mu^ {- i})) - \nabla \nu (\mu^ {i}) \| _ {2} \| \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \| _ {2} \\ + \| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} \| X _ {- i} \| _ {2} \| \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \| _ {2} \\ \leq \frac {\tau}{\delta_ {i}} \| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} \| \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \| _ {2} \\ + \| X _ {- i} \| _ {2} \| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} \| \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \| _ {2}, \\ \end{array}
$$

where the last inequality follows from the smoothness of $\nu (\cdot)$ in Lemma A.7 (2). Similarly, we also have for any $u^{-i}\in \mathbb{R}^{|A^{-i}|}$ that

$$
\begin{array}{l} \langle \nabla_ {2} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (u ^ {- i}) - \sigma_ {\tau} (X _ {- i} \mu^ {i}) \rangle \\ \leq \frac {\tau}{\delta_ {- i}} \| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} \| \sigma_ {\tau} (u ^ {- i}) - \sigma_ {\tau} (X _ {- i} \mu^ {i}) \| _ {2} \\ + \| X _ {i} \| _ {2} \| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} \| \sigma_ {\tau} (u ^ {- i}) - \sigma_ {\tau} (X _ {i} \mu^ {i}) \| _ {2}. \\ \end{array}
$$

Adding up the previous two inequalities and we have

$$
\begin{array}{l} \langle \nabla_ {1} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \rangle + \langle \nabla_ {2} V _ {X} (\mu^ {i}, \mu^ {- i}), \sigma_ {\tau} (u ^ {- i}) - \sigma_ {\tau} (X _ {- i} \mu^ {i}) \rangle \\ \leq \left(\frac {\tau}{\delta_ {i}} + \frac {\tau}{\delta_ {- i}} + \| X _ {i} \| _ {2} + \| X _ {- i} \| _ {2}\right) \left(\| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} + \| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2}\right) \\ \times \left(\| \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} \left(X _ {i} \mu^ {- i}\right) \| _ {2} + \| \sigma_ {\tau} (u ^ {- i}) - \sigma_ {\tau} \left(X _ {i} \mu^ {i}\right) \| _ {2}\right) \\ \leq \frac {1}{2} \left(\frac {\tau}{\delta_ {i}} + \frac {\tau}{\delta_ {- i}} + \| X _ {i} \| _ {2} + \| X _ {- i} \| _ {2}\right) \left[ \bar {c} \left(\| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} + \| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2}\right) ^ {2} \right. \\ \left. + \frac {1}{\bar {c}} \left(\| \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \| _ {2} + \| \sigma_ {\tau} (u ^ {- i}) - \sigma_ {\tau} (X _ {i} \mu^ {i}) \| _ {2}\right) ^ {2} \right] \\ \end{array}
$$

(This is true for all $\bar{c} > 0$ )

$$
\begin{array}{l} \leq \left(\frac {\tau}{\delta_ {i}} + \frac {\tau}{\delta_ {- i}} + \| X _ {i} \| _ {2} + \| X _ {- i} \| _ {2}\right) \left[ \bar {c} \| \sigma_ {\tau} (X _ {i} \mu^ {- i}) - \mu^ {i} \| _ {2} ^ {2} + \bar {c} \| \sigma_ {\tau} (X _ {- i} \mu^ {i}) - \mu^ {- i} \| _ {2} ^ {2} \right. \\ \left. + \frac {1}{\bar {c}} \| \sigma_ {\tau} (u ^ {i}) - \sigma_ {\tau} (X _ {i} \mu^ {- i}) \| _ {2} ^ {2} + \frac {1}{\bar {c}} \| \sigma_ {\tau} (u ^ {- i}) - \sigma_ {\tau} (X _ {i} \mu^ {i}) \| _ {2} ^ {2} \right] \\ \end{array}
$$

(This is true because $(a + b)^2 \leq 2(a^2 + b^2)$ for all $a, b \in \mathbb{R}$ )

$$
\begin{array}{l} \leq \left(\frac {\tau}{\delta_ {i}} + \frac {\tau}{\delta_ {- i}} + \| X _ {i} \| _ {2} + \| X _ {- i} \| _ {2}\right) \left[ \frac {2 \bar {c}}{\tau} V _ {X} (\mu^ {i}, \mu^ {- i}) + \frac {1}{\bar {c} \tau^ {2}} \| u ^ {i} - X _ {i} \mu^ {- i} \| _ {2} ^ {2} \right. \\ \left. + \frac {1}{\bar {c} \tau^ {2}} \| u ^ {- i} - X _ {i} \mu^ {i} \| _ {2} ^ {2} \right], \\ \end{array}
$$

where the last line follows from the quadratic growth property of strongly convex functions and the Lipschitz continuity of the softmax function (Gao and Pavel, 2017).

# A.7.9 Proof of Lemma A.8

Since $\min_{i=1,2}\min_{s,a^{i}}\pi_{k}^{i}(a^{i}|s)\geq\ell_{\tau}$ (cf. Lemma A.2), Lemma A.7 (2) implies that the function $V_{v,s}(\mu^{i},\mu^{-i})$ as a function of $\mu^{i}$ is $L_{\tau,i}-smooth$ on $\{\mu^{i}\in\Delta^{|A^{i}|}\mid\min_{a^{i}}\mu^{i}(a^{i})\geq\ell_{\tau}\}$ uniformly for all $\mu^{-i}$ , where

$$
L _ {\tau , i} := \frac {\sigma_ {\max} ^ {2} (\mathcal {T} ^ {- i} (v ^ {- i}) (s))}{\tau} + \frac {\tau}{\ell_ {\tau}}.
$$

We next bound $L_{\tau, i}$ from above. Since $\| v^i \|_\infty \leq 1 / (1 - \gamma)$ and $\| v^{-i} \|_\infty \leq 1 / (1 - \gamma)$ , we have for any $(s, a^i, a^{-i})$ that

$$
\left| \mathcal {T} ^ {- i} \left(v ^ {- i}\right) \left(s, a ^ {- i}, a ^ {i}\right) \right| \leq \left| \mathcal {R} ^ {- i} \left(s, a ^ {- i}, a ^ {i}\right) \right| + \gamma \mathbb {E} \left[ \left| v ^ {- i} \left(S _ {1}\right) \right| \mid S _ {0} = s, A _ {0} ^ {i} = a ^ {i}, A _ {0} ^ {- i} = a ^ {- i} \right]
$$

$$
\begin{array}{l} \leq 1 + \frac {\gamma}{1 - \gamma} \\ = \frac {1}{1 - \gamma}, \\ \end{array}
$$

which implies

$$
\sigma_ {\max} (\mathcal {T} ^ {- i} (v ^ {- i}) (s)) = \| \mathcal {T} ^ {- i} (v ^ {- i}) (s)) \| _ {2} \leq \frac {\sqrt {| \mathcal {A} ^ {i} | | \mathcal {A} ^ {- i} |}}{1 - \gamma} \leq \frac {A _ {\max}}{1 - \gamma}. \tag {56}
$$

As a result, we have by $\tau \leq 1$ and $\ell_{\tau} \leq 1$ that

$$
L _ {\tau , i} = \frac {\sigma_ {\max} ^ {2} (\mathcal {T} ^ {- i} (v ^ {- i}) (s))}{\tau} + \frac {\tau}{\ell_ {\tau}} \leq \frac {A _ {\max} ^ {2}}{\tau (1 - \gamma) ^ {2}} + \frac {\tau}{\ell_ {\tau}} \leq \frac {2 A _ {\max} ^ {2}}{\ell_ {\tau} (1 - \gamma) ^ {2}} := L _ {\tau}.
$$

Similarly, $V_{v,s}(\mu^i,\mu^{-i})$ as a function of $\mu^{-i}$ is also $L_{\tau}$ - smooth on the set $\{\mu^{-i}\in \Delta^{|\mathcal{A}^{-i}|}\mid \min_{a^{-i}}\mu^i (a^{-i})\geq \ell_\tau \}$ uniformly for all $\mu^i$ .

Using the smoothness of $V_{v,s}(\cdot,\cdot)$ established above, for any $s \in S$ , we have by the policy update equation (cf. Algorithm 3 Line 3) that

$$
\begin{array}{l} V _ {v, s} (\pi_ {k + 1} ^ {i} (s), \pi_ {k + 1} ^ {- i} (s)) \\ \leq V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) + \beta_ {k} \langle \nabla_ {2} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)), \sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \pi_ {k} ^ {- i} (s) \rangle \\ + \beta_ {k} \langle \nabla_ {1} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k + 1} ^ {- i} (s)), \sigma_ {\tau} (q _ {k} ^ {i} (s)) - \pi_ {k} ^ {i} (s) \rangle \\ + \frac {L _ {\tau} \beta_ {k} ^ {2}}{2} \| \sigma_ {\tau} (q _ {k} ^ {i} (s)) - \pi_ {k} ^ {i} (s) \| _ {2} ^ {2} + \frac {L _ {\tau} \beta_ {k} ^ {2}}{2} \| \sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \pi_ {k} ^ {- i} (s) \| _ {2} ^ {2} \\ \leq V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) + \underbrace {\beta_ {k} \langle \nabla_ {2} V _ {v , s} (\pi_ {k} ^ {i} (s) , \pi_ {k} ^ {- i} (s)) , \sigma_ {\tau} (\mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s)) - \pi_ {k} ^ {- i} (s) \rangle} _ {\hat {N} _ {1}} \\ + \underbrace {\beta_ {k} \langle \nabla_ {1} V _ {v , s} (\pi_ {k} ^ {i} (s) , \pi_ {k + 1} ^ {- i} (s)) , \sigma_ {\tau} (\mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s)) - \pi_ {k} ^ {i} (s) \rangle} _ {\hat {N} _ {2}} \\ + \underbrace {\beta_ {k} \langle \nabla_ {2} V _ {v , s} (\pi_ {k} ^ {i} (s) , \pi_ {k} ^ {- i} (s)) , \sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \sigma_ {\tau} (\mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s)) \rangle} _ {\hat {N} _ {3}} \\ + \underbrace {\beta_ {k} \langle \nabla_ {1} V _ {v , s} (\pi_ {k} ^ {i} (s) , \pi_ {k + 1} ^ {- i} (s)) , \sigma_ {\tau} (q _ {k} ^ {i} (s)) - \sigma_ {\tau} (\mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s)) \rangle} _ {\hat {N} _ {4}} \\ + 2 L _ {\tau} \beta_ {k} ^ {2}. \tag {57} \\ \end{array}
$$

We next bound the terms $\{\hat{N}_j\}_{1\leq j\leq 4}$ on the RHS of Eq. (57) using Lemma A.7 (3) and (4).

First consider $\hat{N}_1 + \hat{N}_2$ . We have by Lemma A.7 (3) that

$$
\hat {N} _ {1} + \hat {N} _ {2} \leq - \frac {7 \beta_ {k}}{8} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) + \frac {1 6 \beta_ {k}}{\tau} \| \mathcal {T} ^ {i} (v ^ {i}) (s) + \mathcal {T} ^ {- i} (v ^ {- i} (s) ^ {\top} \| _ {2} ^ {2}.
$$

To proceed, note that for any $\mu^{-i} \in \mathbb{R}^{|\mathcal{A}^{-i}|}$ satisfying $\| \mu^{-i}\| _2 = 1$ , we have

$$
\begin{array}{l} \| (\mathcal {T} ^ {i} (v ^ {i}) (s) + \mathcal {T} ^ {- i} (v ^ {- i} (s) ^ {\top}) \mu^ {i} \| _ {2} ^ {2} \\ = \gamma^ {2} \sum_ {a ^ {i}} \left[ \sum_ {a ^ {- i}} \mathbb {E} \left[ v ^ {i} \left(S _ {1}\right) + v ^ {- i} \left(S _ {1}\right) \mid S _ {0} = s, A _ {0} ^ {i} = a ^ {i}, A _ {0} ^ {- i} = a ^ {- i} \right] \mu^ {- i} \left(a ^ {- i}\right) \right] ^ {2} \\ \leq \gamma^ {2} \| v ^ {i} + v ^ {- i} \| _ {\infty} ^ {2} \sum_ {a ^ {i}} \left[ \sum_ {a ^ {- i}} \mu^ {- i} (a ^ {- i}) \right] ^ {2} \\ \leq \gamma^ {2} A _ {\max} \| v ^ {i} + v ^ {- i} \| _ {\infty} ^ {2}, \\ \end{array}
$$

which implies

$$
\| \mathcal {T} ^ {i} (v ^ {i}) (s) + \mathcal {T} ^ {- i} (v ^ {- i} (s) ^ {\top} \| _ {2} ^ {2} \leq \gamma^ {2} A _ {\max} \| v ^ {i} + v ^ {- i} \| _ {\infty} ^ {2}.
$$

It follows that

$$
\hat {N} _ {1} + \hat {N} _ {2} \leq - \frac {7 \beta_ {k}}{8} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) + \frac {1 6 A _ {\mathrm{max}} \beta_ {k}}{\tau} \| v ^ {i} + v ^ {- i} \| _ {\infty} ^ {2}.
$$

We next consider $\hat{N}_{3} + \hat{N}_{4}$ . Since

$$
\max (\| \mathcal {T} ^ {i} (v ^ {i}) (s) \| _ {2}, \| \mathcal {T} ^ {- i} (v ^ {- i}) (s) \| _ {2}) \leq \frac {A _ {\max}}{1 - \gamma}, \tag {SeeEq.(56)}
$$

we have by Lemma A.7 (4) that

$$
\begin{array}{l} \hat {N} _ {3} + \hat {N} _ {4} \\ \leq 2 \beta_ {k} \left(\frac {\tau}{\ell_ {\tau}} + \frac {A _ {\mathrm{max}}}{1 - \gamma}\right) \left[ \frac {2 \bar {c}}{\tau} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) \right. \\ \left. + \frac {1}{\bar {c} \tau^ {2}} \| q _ {k} ^ {i} (s) - \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) \| _ {2} ^ {2} + \frac {1}{\bar {c} \tau^ {2}} \| q _ {k} ^ {- i} (s) - \mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s) \| _ {2} ^ {2} \right] \\ \end{array}
$$

for any $\bar{c} > 0$ . By choosing $\bar{c} = \frac{\tau}{32}\left(\frac{\tau}{\ell_{\tau}} + \frac{A_{\max}}{1 - \gamma}\right)^{-1}$ , we have from the previous inequality that

$$
\begin{array}{l} \hat {N} _ {3} + \hat {N} _ {4} \leq \frac {\beta_ {k}}{8} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) + \frac {6 4 \beta_ {k} \left(\frac {\tau}{\ell_ {\tau}} + \frac {A _ {\max}}{1 - \gamma}\right) ^ {2}}{\tau^ {3}} \| q _ {k} ^ {i} (s) - \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) \| _ {2} ^ {2} \\ + \frac {6 4 \beta_ {k} \left(\frac {\tau}{\ell_ {\tau}} + \frac {A _ {\max}}{1 - \gamma}\right) ^ {2}}{\tau^ {3}} \| q _ {k} ^ {- i} (s) - \mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s) \| _ {2} ^ {2} \\ \leq \frac {\beta_ {k}}{8} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) + \frac {2 5 6 A _ {\max} ^ {2} \beta_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3} (1 - \gamma) ^ {2}} \| q _ {k} ^ {i} (s) - \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) \| _ {2} ^ {2} \\ + \frac {2 5 6 A _ {\mathrm{max}} ^ {2} \beta_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3} (1 - \gamma) ^ {2}} \| q _ {k} ^ {- i} (s) - \mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s) \| _ {2} ^ {2}. \\ \end{array}
$$

Finally, using the upper bounds we obtained for the terms $\hat{N}_1 + \hat{N}_2$ and $\hat{N}_3 + \hat{N}_4$ in Eq. (57) and we have

$$
\begin{array}{l} V _ {v, s} (\pi_ {k + 1} ^ {i} (s), \pi_ {k + 1} ^ {- i} (s)) \\ \leq \left(1 - \frac {3 \beta_ {k}}{4}\right) V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) + \frac {1 6 A _ {\max} \beta_ {k}}{\tau} \| v ^ {i} + v ^ {- i} \| _ {\infty} ^ {2}. \\ + \frac {2 5 6 A _ {\mathrm{max}} ^ {2} \beta_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3} (1 - \gamma) ^ {2}} \| q _ {k} ^ {i} (s) - \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) \| _ {2} ^ {2} \\ + \frac {2 5 6 A _ {\mathrm{max}} ^ {2} \beta_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3} (1 - \gamma) ^ {2}} \| q _ {k} ^ {- i} (s) - \mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s) \| _ {2} ^ {2} \\ + \frac {4 A _ {\mathrm{max}} ^ {2}}{\ell_ {\tau} (1 - \gamma) ^ {2}} \beta_ {k} ^ {2}. \\ \end{array}
$$

Summing up both sides of the previous inequality for all $s$ and then taking expectation, and we have the desired result.

# A.7.10 Proof of Lemma A.9

(1) For any $(q_1^i, q_2^i)$ and $(s_0, a_0^i, a_0^{-i}, s_1)$ , we have

$$
\| F ^ {i} (q _ {1} ^ {i}, s _ {0}, a _ {0} ^ {i}, a _ {0} ^ {- i}, s _ {1}) - F ^ {i} (q _ {2} ^ {i}, s _ {0}, a _ {0} ^ {i}, a _ {0} ^ {- i}, s _ {1}) \| _ {2} ^ {2}
$$

$$
\begin{array}{l} = \sum_ {(s, a ^ {i})} ([ F ^ {i} (q _ {1} ^ {i}, s _ {0}, a _ {0} ^ {i}, a _ {0} ^ {- i}, s _ {1}) ] (s, a ^ {i}) - [ F ^ {i} (q _ {2} ^ {i}, s _ {0}, a _ {0} ^ {i}, a _ {0} ^ {- i}, s _ {1}) ] (s, a ^ {i})) ^ {2} \\ = \sum_ {(s, a ^ {i})} \mathbb {1} _ {\{(s, a ^ {i}) = (s _ {0}, a _ {0} ^ {i}) \}} \left(q _ {1} ^ {i} (s _ {0}, a _ {0} ^ {i}) - q _ {2} ^ {i} (s _ {0}, a _ {0} ^ {i})\right) ^ {2} \\ \leq \| q _ {1} ^ {i} - q _ {2} ^ {i} \| _ {2} ^ {2}. \\ \end{array}
$$

(2) For any $(s_0, a_0^i, a_0^{-i}, s_1)$ , we have

$$
\begin{array}{l} \| F ^ {i} (\mathbf {0}, s _ {0}, a _ {0} ^ {i}, a _ {0} ^ {- i}, s _ {1}) \| _ {2} ^ {2} = \sum_ {(s, a ^ {i})} ([ F ^ {i} (\mathbf {0}, s _ {0}, a _ {0} ^ {i}, a _ {0} ^ {- i}, s _ {1}) ] (s, a ^ {i})) ^ {2} \\ = \sum_ {(s, a ^ {i})} \mathbb {1} _ {\{(s, a ^ {i}) = (s _ {0}, a _ {0} ^ {i}) \}} \left(\mathcal {R} ^ {i} (s _ {0}, a _ {0} ^ {i}, a _ {0} ^ {- i}) + \gamma v ^ {i} (s _ {1})\right) ^ {2} \\ \leq \frac {1}{(1 - \gamma) ^ {2}}, \\ \end{array}
$$

where the last line follows from $\|v^{i}\|_{\infty}\leq1/(1-\gamma)$ and $|\mathcal{R}^{i}(s_{0},a_{0}^{i},a_{0}^{-i})|\leq1.$

(3) We first write down the explicit expression of $\bar{F}_k^i (\cdot)$ . Using the definition of $\mathcal{T}^i (\cdot)$ and we have

$$
\bar {F} _ {k} ^ {i} (q ^ {i}) (s) = \mu_ {k} (s) \Pi_ {k} ^ {i} (s) \left(\mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) - q ^ {i} (s)\right), \forall s \in \mathcal {S},
$$

where $\Pi_k^i (s)\coloneqq \mathrm{diag}(\pi_k^i (s))$ . Since $\min_{0\leq k\leq K - 1}\min_{s\in \mathcal{S}}\mu_k(s) > 0$ (cf. Lemma A.2 and Lemma 4.1 (4)) and $\Pi_k^i (s)$ has strictly positive diagonal entries for all $s$ and $k$ (cf. Lemma A.2), the equation $\bar{F}_k^i (q^i) = 0$ has a unique solution $\bar{q}_k^i\in \mathbb{R}^{|\mathcal{S}||\mathcal{A}^i|}$ , which is explicitly given by

$$
\bar {q} _ {k} ^ {i} (s) = \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s), \forall s \in \mathcal {S}.
$$

(4) Using the explicit expression of $\bar{F}_k^i (\cdot)$ and we have for any $q_{1}^{i}$ and $q_{2}^{i}$ that

$$
\begin{array}{l} \langle \bar {F} _ {k} ^ {i} (q _ {1} ^ {i}) - \bar {F} _ {k} ^ {i} (q _ {2} ^ {i}), q _ {1} ^ {i} - q _ {2} ^ {i} \rangle = - \sum_ {s, a ^ {i}} \mu_ {k} (s) \pi_ {k} ^ {i} (a ^ {i} | s) (q _ {1} ^ {i} (s, a ^ {i}) - q _ {2} ^ {i} (s, a ^ {i})) ^ {2} \\ \leq - \min _ {s, a ^ {i}} \mu_ {k} (s) \pi_ {k} ^ {i} (a ^ {i} | s) \| q _ {1} ^ {i} - q _ {2} ^ {i} \| _ {2} ^ {2} \\ \leq - \mu_ {\tau} \ell_ {\tau} \| q _ {1} ^ {i} - q _ {2} ^ {i} \| _ {2} ^ {2} \quad (\text { Lemma   4.1   (4)   and   Lemma   A.2 }) \\ = - c _ {\tau} \| q _ {1} ^ {i} - q _ {2} ^ {i} \| _ {2} ^ {2}, \\ \end{array}
$$

where we recall that $c_{\tau} = \mu_{\tau}\ell_{\tau}$ .

# A.7.11 Proof of Lemma A.10

For any $k\geq 0$ , we have

$$
\begin{array}{l} N _ {2} = \mathbb {E} [ \langle F ^ {i} (q _ {k} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - \bar {F} _ {k} ^ {i} (q _ {k} ^ {i}), q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ] \\ = \underbrace {\mathbb {E} [ \langle F ^ {i} (q _ {k - z _ {k}} ^ {i} , S _ {k} , A _ {k} ^ {i} , A _ {k} ^ {- i} , S _ {k + 1}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) , q _ {k - z _ {k}} ^ {i} - \bar {q} _ {k - z _ {k}} ^ {i} \rangle ]} _ {N _ {2, 1}} \\ + \underbrace {\mathbb {E} [ \langle F ^ {i} (q _ {k - z _ {k}} ^ {i} , S _ {k} , A _ {k} ^ {i} , A _ {k} ^ {- i} , S _ {k + 1}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) , q _ {k} ^ {i} - q _ {k - z _ {k}} ^ {i} \rangle ]} _ {N _ {2, 2}} \\ + \underbrace {\mathbb {E} [ \langle F ^ {i} (q _ {k - z _ {k}} ^ {i} , S _ {k} , A _ {k} ^ {i} , A _ {k} ^ {- i} , S _ {k + 1}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) , \bar {q} _ {k - z _ {k}} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ]} _ {N _ {2, 3}} \\ \end{array}
$$

$$
\begin{array}{l} + \underbrace {\mathbb {E} [ \langle F ^ {i} (q _ {k} ^ {i} , S _ {k} , A _ {k} ^ {i} , A _ {k} ^ {- i} , S _ {k + 1}) - F ^ {i} (q _ {k - z _ {k}} ^ {i} , S _ {k} , A _ {k} ^ {i} , A _ {k} ^ {- i} , S _ {k + 1}) , q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ]} _ {N _ {2, 4}} \\ + \underbrace {\mathbb {E} [ \langle \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) - \bar {F} _ {k} ^ {i} (q _ {k} ^ {i}) , q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ]} _ {N _ {2, 5}} \\ \end{array}
$$

We next control the terms $\{N_{2,j}\}_{1\leq j\leq5}$ on the RHS of the previous inequality. Before that, the following two lemmas are needed. The proof of Lemma A.13 follows from that of (Srikant and Ying, 2019, Lemma 3) and (Chen et al., 2021b, Lemma A.3). Lemma A.14 is the policy-counterpart of Lemma A.13.

Lemma A.13 (Proof in Appendix A.7.14). Given positive integers $k_{1} \leq k_{2}$ satisfying $\alpha_{k_{1}, k_{2}-1} \leq 1/4$ , we have for any $k \in \{k_{1}, k_{1} + 1, \cdots, k_{2}\}$ that

$$
\left\| q _ {k} ^ {i} - q _ {k _ {1}} ^ {i} \right\| _ {2} \leq \min \left(2 \alpha_ {k _ {1}, k _ {2} - 1}, 1 / 2\right) \left(\left\| q _ {k _ {1}} ^ {i} \right\| _ {2} + \frac {1}{1 - \gamma}\right),
$$

$$
\| q _ {k} ^ {i} - q _ {k _ {1}} ^ {i} \| _ {2} \leq \min (4 \alpha_ {k _ {1}, k _ {2} - 1}, 1) \left(\| q _ {k _ {2}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right).
$$

Lemma A.14 (Proof in Appendix A.7.15). Given positive integers $k_{1} \leq k_{2}$ satisfying $\beta_{k_{1}, k_{2}-1} \leq 1/4$ , we have for any $s \in S$ and $k \in \{k_{1}, k_{1} + 1, \cdots, k_{2}\}$ that

$$
\| \pi_ {k} ^ {i} (s) - \pi_ {k _ {1}} ^ {i} (s) \| _ {2} \leq \min (2 \beta_ {k _ {1}, k _ {2} - 1}, 1 / 2) \left(\| \pi_ {k _ {1}} ^ {i} (s) \| _ {2} + 1\right),
$$

$$
\left\| \pi_ {k} ^ {i} (s) - \pi_ {k _ {1}} ^ {i} (s) \right\| _ {2} \leq \min \left(4 \beta_ {k _ {1}, k _ {2} - 1}, 1\right) \left(\left\| \pi_ {k _ {2}} ^ {i} (s) \right\| _ {2} + 1\right).
$$

We next bound the terms $\{N_{2,j}\}_{1\leq j\leq5}$ . Let $F_{k}$ be the $\sigma$ -algebra generated the sequence of random variables $\{S_{0},A_{0}^{i},A_{0}^{-i},\cdots,S_{k-1},A_{k-1}^{i},A_{k-1}^{-i},S_{k}\}$ .

The Term $N_{2,1}$ . Using the tower property of conditional expectations and we have

$$
\begin{array}{l} N _ {2, 1} \\ = \mathbb {E} [ \langle F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}), q _ {k - z _ {k}} ^ {i} - \bar {q} _ {k - z _ {k}} ^ {i} \rangle ] \\ = \mathbb {E} [ \langle \mathbb {E} [ F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) | \mathcal {F} _ {k - z _ {k}} ] - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}), q _ {k - z _ {k}} ^ {i} - \bar {q} _ {k - z _ {k}} ^ {i} \rangle ] \\ \leq \mathbb {E} [ \| \mathbb {E} [ F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) | \mathcal {F} _ {k - z _ {k}} ] - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2} \| q _ {k - z _ {k}} ^ {i} - \bar {q} _ {k - z _ {k}} ^ {i} \| _ {2} ] \\ \leq \frac {2 \sqrt {| \mathcal {S} | A _ {\max}}}{1 - \gamma} \mathbb {E} [ \underbrace {\| F ^ {i} (q _ {k - z _ {k}} ^ {i} , S _ {k} , A _ {k} ^ {i} , A _ {k} ^ {- i} , S _ {k + 1}) \mid \mathcal {F} _ {k - z _ {k}} ] - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2}} _ {N _ {2, 1, 1}}, \\ \end{array}
$$

where the last line follows from $\|q_{k-z_{k}}^{i}\|_{2}\leq\sqrt{|S|A_{\max}}\|q_{k-z_{k}}^{i}\|_{\infty}\leq\sqrt{|S|A_{\max}}/(1-\gamma)$ and similarly $\|\bar{q}_{k-z_{k}}^{i}\|_{2}\leq\sqrt{|S|A_{\max}}/(1-\gamma)$ . For the term $N_{2,1,1}$ , using triangle inequality and we have

$$
\begin{array}{l} \left\| F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) \mid \mathcal {F} _ {k - z _ {k}} \right] - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2} \\ \leq \| \bar {F} _ {k} ^ {i} (q _ {k - z _ {k}} ^ {i}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2} \\ + \left\| F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) \mid \mathcal {F} _ {k - z _ {k}} \right] - \bar {F} _ {k} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2}. \tag {58} \\ \end{array}
$$

To control the first term on the RHS of Eq. (58), recall that

$$
\bar {F} _ {k} ^ {i} (q ^ {i}) (s) = \mu_ {k} (s) \Pi_ {k} ^ {i} (s) \left(\mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) - q ^ {i} (s)\right), \forall s \in \mathcal {S}, q ^ {i} \in \mathbb {R} ^ {| \mathcal {S} | | \mathcal {A} ^ {i} |},
$$

where $\Pi_k^i (s) = \mathrm{diag}(\pi_k^i (s))$ . Therefore, we have for any $s\in \mathcal{S}$ and $q^{i}\in \mathbb{R}^{|\mathcal{S}||\mathcal{A}^{i}|}$ that

$$
\| \bar {F} _ {k} ^ {i} (q ^ {i}) (s) - \bar {F} _ {k - z _ {k}} ^ {i} (q ^ {i}) (s) \| _ {2}
$$

$$
\leq \| \mu_ {k} (s) \Pi_ {k} ^ {i} (s) \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) - \mu_ {k - z _ {k}} (s) \Pi_ {k - z _ {k}} ^ {i} (s) \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k - z _ {k}} ^ {- i} (s) \| _ {2}
$$

$$
+ \| \left(\mu_ {k} (s) \Pi_ {k} ^ {i} (s) - \mu_ {k - z _ {k}} (s) \Pi_ {k - z _ {k}} ^ {i} (s)\right) q ^ {i} (s) \| _ {2}. \tag {59}
$$

We next bound the two terms on the RHS of Eq. (59). For the first term, we have

$$
\| (\mu_ {k} (s) \Pi_ {k} ^ {i} (s) - \mu_ {k - z _ {k}} (s) \Pi_ {k - z _ {k}} ^ {i} (s)) q ^ {i} (s) \| _ {2}
$$

$$
\leq \frac {A _ {\max} ^ {1 / 2}}{1 - \gamma} \| \mu_ {k} (s) \Pi_ {k} ^ {i} (s) - \mu_ {k - z _ {k}} (s) \Pi_ {k - z _ {k}} ^ {i} (s) \| _ {2} \quad (\| q ^ {i} (s) \| _ {2} \leq \frac {A _ {\max} ^ {1 / 2}}{1 - \gamma})
$$

$$
= \frac {A _ {\max} ^ {1 / 2}}{1 - \gamma} \left(\| \mu_ {k} (s) (\Pi_ {k} ^ {i} (s) - \Pi_ {k - z _ {k}} ^ {i} (s)) \| _ {2} + \| (\mu_ {k} (s) - \mu_ {k - z _ {k}} (s)) \Pi_ {k - z _ {k}} ^ {i} (s) \| _ {2}\right)
$$

$$
\leq \frac {A _ {\max} ^ {1 / 2}}{1 - \gamma} \left(\mu_ {k} (s) \| \Pi_ {k} ^ {i} (s) - \Pi_ {k - z _ {k}} ^ {i} (s) \| _ {2} + | \mu_ {k} (s) - \mu_ {k - z _ {k}} (s) |\right) \quad (\| \Pi_ {k - z _ {k}} ^ {i} (s) \| _ {2} \leq 1)
$$

$$
= \frac {A _ {\max} ^ {1 / 2}}{1 - \gamma} \left(\mu_ {k} (s) \| \pi_ {k} ^ {i} (s) - \pi_ {k - z _ {k}} ^ {i} (s) \| _ {\infty} + | \mu_ {k} (s) - \mu_ {k - z _ {k}} (s) |\right).
$$

For the second term on the RHS of Eq. (59), we have

$$
\| \mu_ {k} (s) \Pi_ {k} ^ {i} (s) \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) - \mu_ {k - z _ {k}} (s) \Pi_ {k - z _ {k}} ^ {i} (s) \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k - z _ {k}} ^ {- i} (s) \| _ {2}
$$

$$
\leq \| (\mu_ {k} (s) \Pi_ {k} ^ {i} (s) - \mu_ {k - z _ {k}} (s) \Pi_ {k - z _ {k}} ^ {i} (s)) \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) \| _ {2}
$$

$$
+ \| \mu_ {k - z _ {k}} (s) \Pi_ {k - z _ {k}} ^ {i} (s) \mathcal {T} ^ {i} (v ^ {i}) (s) (\pi_ {k} ^ {- i} (s) - \pi_ {k - z _ {k}} ^ {- i} (s)) \| _ {2}
$$

$$
\leq \frac {A _ {\max} ^ {1 / 2}}{1 - \gamma} \| \mu_ {k} (s) \Pi_ {k} ^ {i} (s) - \mu_ {k - z _ {k}} (s) \Pi_ {k - z _ {k}} ^ {i} (s) \| _ {2} + \frac {1}{1 - \gamma} \mu_ {k - z _ {k}} (s) \| \pi_ {k} ^ {- i} (s) - \pi_ {k - z _ {k}} ^ {- i} (s) \| _ {\infty}
$$

$$
\leq \frac {A _ {\max} ^ {1 / 2}}{1 - \gamma} \left(\mu_ {k} (s) \| \Pi_ {k} ^ {i} (s) - \Pi_ {k - z _ {k}} ^ {i} (s) \| _ {2} + | \mu_ {k} (s) - \mu_ {k - z _ {k}} (s) |\right)
$$

$$
+ \frac {\mu_ {k - z _ {k}} (s)}{1 - \gamma} \| \pi_ {k} ^ {- i} (s) - \pi_ {k - z _ {k}} ^ {- i} (s) \| _ {\infty}
$$

$$
\leq \frac {A _ {\max} ^ {1 / 2}}{1 - \gamma} \left(| \mu_ {k} (s) - \mu_ {k - z _ {k}} (s) | + (\mu_ {k} (s) + \mu_ {k - z _ {k}} (s)) \| \pi_ {k} ^ {- i} (s) - \pi_ {k - z _ {k}} ^ {- i} (s) \| _ {\infty}\right).
$$

Using the previous two inequalities in Eq. (59) and we have

$$
\left\| \bar {F} _ {k} ^ {i} \left(q ^ {i}\right) (s) - \bar {F} _ {k - z _ {k}} ^ {i} \left(q ^ {i}\right) (s) \right\| _ {2}
$$

$$
\leq \frac {A _ {\max} ^ {1 / 2}}{1 - \gamma} \left(2 | \mu_ {k} (s) - \mu_ {k - z _ {k}} (s) | + (\mu_ {k} (s) + \mu_ {k - z _ {k}} (s)) \| \pi_ {k} ^ {- i} (s) - \pi_ {k - z _ {k}} ^ {- i} (s) \| _ {\infty}\right)
$$

$$
+ \frac {A _ {\mathrm{max}} ^ {1 / 2}}{1 - \gamma} \mu_ {k} (s) \| \pi_ {k} ^ {i} (s) - \pi_ {k - z _ {k}} ^ {i} (s) \| _ {\infty},
$$

which implies

$$
\| \bar {F} _ {k} ^ {i} (q ^ {i}) (s) - \bar {F} _ {k - z _ {k}} ^ {i} (q ^ {i}) (s) \| _ {2} ^ {2}
$$

$$
\leq \frac {3 A _ {\max}}{(1 - \gamma) ^ {2}} \left(4 | \mu_ {k} (s) - \mu_ {k - z _ {k}} (s) | ^ {2} + 2 (\mu_ {k} (s) ^ {2} + \mu_ {k - z _ {k}} (s) ^ {2}) \| \pi_ {k} ^ {- i} (s) - \pi_ {k - z _ {k}} ^ {- i} (s) \| _ {\infty} ^ {2}\right)
$$

$$
+ \frac {3 A _ {\max}}{(1 - \gamma) ^ {2}} \mu_ {k} (s) ^ {2} \| \pi_ {k} ^ {i} (s) - \pi_ {k - z _ {k}} ^ {i} (s) \| _ {\infty} ^ {2}.
$$

It follows that

$$
\| \bar {F} _ {k} ^ {i} (q ^ {i}) - \bar {F} _ {k - z _ {k}} ^ {i} (q ^ {i}) \| _ {2} ^ {2} = \sum_ {s} \| \bar {F} _ {k} ^ {i} (q ^ {i}) (s) - \bar {F} _ {k - z _ {k}} ^ {i} (q ^ {i}) (s) \| _ {2} ^ {2}
$$

$$
\begin{array}{l} \leq \frac {3 A _ {\max}}{(1 - \gamma) ^ {2}} \left(4 \| \mu_ {k} - \mu_ {k - z _ {k}} \| _ {2} ^ {2} + 4 \max _ {s} \| \pi_ {k} ^ {- i} (s) - \pi_ {k - z _ {k}} ^ {- i} (s) \| _ {\infty} ^ {2}\right) \\ + \frac {3 A _ {\mathrm{max}}}{(1 - \gamma) ^ {2}} \max _ {s} \| \pi_ {k} ^ {i} (s) - \pi_ {k - z _ {k}} ^ {i} (s) \| _ {\infty} ^ {2}. \\ \end{array}
$$

Since

$$
\begin{array}{l} \| \mu_ {k} - \mu_ {k - z _ {k}} \| _ {2} ^ {2} \leq | \mathcal {S} | \| \mu_ {k} - \mu_ {k - z _ {k}} \| _ {\infty} ^ {2} \\ \leq | \mathcal {S} | \hat {L} _ {\tau} ^ {2} \| \pi_ {k} ^ {i} - \pi_ {k - z _ {k}} ^ {i} \| _ {\infty} ^ {2} (Lemma4.1(3)) \\ = | \mathcal {S} | \hat {L} _ {\tau} ^ {2} \max _ {s \in \mathcal {S}} \| \pi_ {k} ^ {i} (s) - \pi_ {k - z _ {k}} ^ {i} (s) \| _ {1} ^ {2} \\ \leq | \mathcal {S} | ^ {2} \hat {L} _ {\tau} ^ {2} \max _ {s \in \mathcal {S}} \| \pi_ {k} ^ {i} (s) - \pi_ {k - z _ {k}} ^ {i} (s) \| _ {2} ^ {2} \\ \leq 1 6 | \mathcal {S} | ^ {2} \hat {L} _ {\tau} ^ {2} \beta_ {k - z _ {k}, k - 1} ^ {2} \max _ {s \in \mathcal {S}} (\| \pi_ {k} ^ {i} (s) \| _ {2} + 1) ^ {2} (LemmaA.14) \\ \leq 6 4 | \mathcal {S} | ^ {2} \hat {L} _ {\tau} ^ {2} \beta_ {k - z _ {k}, k - 1} ^ {2} \\ \end{array}
$$

and

$$
\max _ {s} \| \pi_ {k} ^ {i} (s) - \pi_ {k - z _ {k}} ^ {i} (s) \| _ {\infty} ^ {2} \leq \max _ {s} \| \pi_ {k} ^ {i} (s) - \pi_ {k - z _ {k}} ^ {i} (s) \| _ {2} ^ {2}
$$

$$
\leq 1 6 \beta_ {k - z _ {k}, k - 1} ^ {2} \max _ {s} (\| \pi_ {k} ^ {i} (s) \| _ {2} + 1) ^ {2} \tag {LemmaA.14}
$$

$$
\leq 6 4 \beta_ {k - z _ {k}, k - 1} ^ {2}, \quad i \in \{1, 2 \},
$$

we have

$$
\| \bar {F} _ {k} ^ {i} (q ^ {i}) - \bar {F} _ {k - z _ {k}} ^ {i} (q ^ {i}) \| _ {2} ^ {2} \leq \frac {3 A _ {\max}}{(1 - \gamma) ^ {2}} \left(2 5 6 | \mathcal {S} | ^ {2} \hat {L} _ {\tau} ^ {2} \beta_ {k - z _ {k}, k - 1} ^ {2} + 3 2 0 \beta_ {k - z _ {k}, k - 1} ^ {2}\right)
$$

$$
\leq \frac {1 7 2 8 A _ {\mathrm{max}} | \mathcal {S} | ^ {2} \hat {L} _ {\tau} ^ {2} \beta_ {k - z _ {k} , k - 1} ^ {2}}{(1 - \gamma) ^ {2}},
$$

which implies

$$
\left\| \bar {F} _ {k} ^ {i} \left(q ^ {i}\right) - \bar {F} _ {k - z _ {k}} ^ {i} \left(q ^ {i}\right) \right\| _ {2} \leq \frac {4 2 | \mathcal {S} | A _ {\max} ^ {1 / 2} \hat {L} _ {\tau} \beta_ {k - z _ {k} , k - 1}}{1 - \gamma} \tag {60}
$$

for all $q^i \in \mathbb{R}^{|\mathcal{S}| |\mathcal{A}^i|}$ .

We next move on to bound the second term on the RHS of Eq. (58). Recall that we denote $P_{\pi} \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$ as the transition probability matrix of the Markov chain $\{S_k\}$ induced by the joint policy $\pi$ . Using the definition of conditional expectation and we have

$$
\begin{array}{l} \| \mathbb {E} [ F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) | \mathcal {F} _ {k - z _ {k}} ] - \bar {F} _ {k} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2} \\ = \left\| \right. \sum_ {s} \left[\left(\prod_ {j = k + 1} ^ {k + z _ {k}} P _ {\pi_ {j - z _ {k}}}\right) (S _ {k - z _ {k}}, s) - \mu_ {k} (s) \right] \sum_ {a ^ {i}} \pi_ {k} ^ {i} (a ^ {i} | s) \sum_ {a ^ {- i}} \pi_ {k} ^ {- i} (a ^ {- i} | s) \\ \times \sum_ {s ^ {\prime}} p (s ^ {\prime} | s, a ^ {i}, a ^ {- i}) F ^ {i} (q _ {k - z _ {k}} ^ {i}, s, a ^ {i}, a ^ {- i}, s ^ {\prime}) \Bigg \| _ {2} \\ \leq \left| \sum_ {s} \left[ \left(\prod_ {j = k + 1} ^ {k + z _ {k}} P _ {\pi_ {j - z _ {k}}}\right) (S _ {k - z _ {k}}, s) - \mu_ {k} (s) \right] \right| \left(\| q _ {k - z _ {k}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \tag {LemmaA.9} \\ \leq \sum_ {s} \left| \left(\prod_ {j = k + 1} ^ {k + z _ {k}} P _ {\pi_ {j - z _ {k}}}\right) (S _ {k - z _ {k}}, s) - \mu_ {k} (s) \right| \left(\| q _ {k - z _ {k}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \\ \end{array}
$$

$$
\leq \left\{\sum_ {s} \left| \left(\prod_ {j = k + 1} ^ {k + z _ {k}} P _ {\pi_ {j - z _ {k}}}\right) (S _ {k - z _ {k}}, s) - P _ {\pi_ {k}} ^ {z _ {k}} (S _ {k - z _ {k}}, s) \right| + \sum_ {s} \left| P _ {\pi_ {k}} ^ {z _ {k}} (S _ {k - z _ {k}}, s) - \mu_ {k} (s) \right| \right\}
$$

$$
\times \left(\| q _ {k - z _ {k}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right)
$$

$$
\leq \left\{\left\| \prod_ {j = k + 1} ^ {k + z _ {k}} P _ {\pi_ {j - z _ {k}}} - P _ {\pi_ {k}} ^ {z _ {k}} \right\| _ {\infty} + 2 \rho_ {\tau} ^ {z _ {k}} \right\} \left(\left\| q _ {k - z _ {k}} ^ {i} \right\| _ {2} + \frac {1}{1 - \gamma}\right), \tag {61}
$$

where the last line follows from Lemma 4.1 (2) and Lemma A.2. Observe that

$$
\left\| \prod_ {j = k + 1} ^ {k + z _ {k}} P _ {\pi_ {j - z _ {k}}} - P _ {\pi_ {k}} ^ {z _ {k}} \right\| _ {\infty} = \left\| \sum_ {\ell = 1} ^ {z _ {k}} \left(\prod_ {j = k + 1} ^ {k - \ell + 1 + z _ {k}} P _ {\pi_ {j - z _ {k}}} P _ {\pi_ {k}} ^ {\ell - 1} - \prod_ {j = k + 1} ^ {k - \ell + z _ {k}} P _ {\pi_ {j - z _ {k}}} P _ {\pi_ {k}} ^ {\ell}\right) \right\| _ {\infty}
$$

$$
= \left\| \sum_ {\ell = 1} ^ {z _ {k}} \left(\prod_ {j = k + 1} ^ {k - \ell + z _ {k}} P _ {\pi_ {j - z _ {k}}} (P _ {\pi_ {k - \ell + 1}} - P _ {\pi_ {k}}) P _ {\pi_ {k}} ^ {\ell - 1}\right) \right\| _ {\infty}
$$

$$
\leq \sum_ {\ell = 1} ^ {z _ {k}} \left\| \prod_ {j = k + 1} ^ {k - \ell + z _ {k}} P _ {\pi_ {j - z _ {k}}} \right\| _ {\infty} \| P _ {\pi_ {k - \ell + 1}} - P _ {\pi_ {k}} \| _ {\infty} \| P _ {\pi_ {k}} ^ {\ell - 1} \| _ {\infty}.
$$

Since the induced $\ell_{\infty}$ -norm for any stochastic matrix is 1 and $P_{\pi}$ as a function of $\pi$ is 1-Lipschitz continuous with respect to the $\ell_{\infty}$ -norm, we have

$$
\left\| \prod_ {j = k + 1} ^ {k + z _ {k}} P _ {\pi_ {j - z _ {k}}} - P _ {\pi_ {k}} ^ {z _ {k}} \right\| _ {\infty}
$$

$$
\leq \sum_ {\ell = 1} ^ {z _ {k}} \| \pi_ {k - \ell + 1} - \pi_ {k} \| _ {\infty}
$$

$$
= \sum_ {\ell = 1} ^ {z _ {k}} \max _ {s \in \mathcal {S}} \sum_ {a ^ {i}, a ^ {- i}} | \pi_ {k - \ell + 1} ^ {i} (a ^ {i} | s) \pi_ {k - \ell + 1} ^ {- i} (a ^ {- i} | s) - \pi_ {k} ^ {i} (a ^ {i} | s) \pi_ {k} ^ {- i} (a ^ {- i} | s) |
$$

$$
\leq \sum_ {\ell = 1} ^ {z _ {k}} \max _ {s \in \mathcal {S}} \sum_ {a ^ {i}, a ^ {- i}} \pi_ {k - \ell + 1} ^ {i} (a ^ {i} | s) | \pi_ {k - \ell + 1} ^ {- i} (a ^ {- i} | s) - \pi_ {k} ^ {- i} (a ^ {- i} | s) |
$$

$$
+ \sum_ {\ell = 1} ^ {z _ {k}} \max _ {s \in \mathcal {S}} \sum_ {a ^ {i}, a ^ {- i}} | \pi_ {k - \ell + 1} ^ {i} (a ^ {i} | s) - \pi_ {k} ^ {i} (a ^ {i} | s) | \pi_ {k} ^ {- i} (a ^ {- i} | s)
$$

$$
= \sum_ {\ell = 1} ^ {z _ {k}} \max _ {s \in \mathcal {S}} \left(\sum_ {a ^ {- i}} | \pi_ {k - \ell + 1} ^ {- i} (a ^ {- i} | s) - \pi_ {k} ^ {- i} (a ^ {- i} | s) | + \sum_ {a ^ {i}} | \pi_ {k - \ell + 1} ^ {i} (a ^ {i} | s) - \pi_ {k} ^ {i} (a ^ {i} | s) |\right)
$$

$$
= \sum_ {\ell = 1} ^ {z _ {k}} \max _ {s \in \mathcal {S}} \left(\| \pi_ {k - \ell + 1} ^ {- i} (s) - \pi_ {k} ^ {- i} (s) \| _ {1} + \| \pi_ {k - \ell + 1} ^ {i} (s) - \pi_ {k} ^ {i} (s) \| _ {1}\right)
$$

$$
\leq A _ {\max} ^ {1 / 2} \sum_ {\ell = 1} ^ {z _ {k}} \max _ {s \in \mathcal {S}} \left(\| \pi_ {k - \ell + 1} ^ {- i} (s) - \pi_ {k} ^ {- i} (s) \| _ {2} + \| \pi_ {k - \ell + 1} ^ {i} (s) - \pi_ {k} ^ {i} (s) \| _ {2}\right)
$$

$$
\leq 4 z _ {k} \beta_ {k - z _ {k}, k - 1} A _ {\max} ^ {1 / 2} \max _ {s \in \mathcal {S}} \left(\| \pi_ {k} ^ {- i} (s) \| _ {2} + \| \pi_ {k} ^ {i} (s) \| _ {2} + 2\right) \tag {LemmaA.14}
$$

$$
\leq 1 6 z _ {k} \beta_ {k - z _ {k}, k - 1} A _ {\max} ^ {1 / 2}
$$

$$
\leq 1 6 z _ {k} \beta_ {k - z _ {k}, k - 1} A _ {\max} ^ {1 / 2}.
$$

It then follows from the previous inequality that

$$
\| \mathbb {E} [ F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) | \mathcal {F} _ {k - z _ {k}} ] - \bar {F} _ {k} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2}
$$

$$
\leq \left\{\left\| \prod_ {j = k + 1} ^ {k + z _ {k}} P _ {\pi_ {j - z _ {k}}} - P _ {\pi_ {k}} ^ {z _ {k}} \right\| _ {\infty} + 2 \rho_ {\tau} ^ {z _ {k}} \right\} \left(\| q _ {k - z _ {k}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right)
$$

$$
\leq \left(1 6 A _ {\max} ^ {1 / 2} z _ {k} \beta_ {k - z _ {k}, k - 1} + 2 \rho_ {\tau} ^ {z _ {k}}\right) \left(\| q _ {k - z _ {k}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right)
$$

$$
\leq \frac {2 \sqrt {| \mathcal {S} | A _ {\max}}}{1 - \gamma} \left(1 6 A _ {\max} ^ {1 / 2} z _ {k} \beta_ {k - z _ {k}, k - 1} + 2 \rho_ {\tau} ^ {z _ {k}}\right)
$$

$$
\leq \frac {2 \sqrt {| \mathcal {S} | A _ {\max}}}{1 - \gamma} \left(1 6 A _ {\max} ^ {1 / 2} z _ {k} \beta_ {k - z _ {k}, k - 1} + 2 \beta_ {k}\right) \quad (\text { Definition   of } z _ {k})
$$

$$
\leq \frac {3 6 \sqrt {| \mathcal {S} |} A _ {\max} z _ {k} \beta_ {k - z _ {k} , k - 1}}{1 - \gamma}
$$

Substituting the previous inequality and the bound in Eq. (60) into Eq. (58) and we have

$$
N _ {2, 1, 1} \leq \frac {4 2 | \mathcal {S} | A _ {\max} ^ {1 / 2} \hat {L} _ {\tau} \beta_ {k - z _ {k} , k - 1}}{1 - \gamma} + \frac {3 6 \sqrt {| \mathcal {S} |} A _ {\max} z _ {k} \beta_ {k - z _ {k} , k - 1}}{1 - \gamma}
$$

$$
\leq \frac {8 0 | \mathcal {S} | A _ {\mathrm{max}} \hat {L} _ {\tau} z _ {k} \beta_ {k - z _ {k} , k - 1}}{1 - \gamma}.
$$

It follows that

$$
N _ {2, 1} \leq \frac {2 \sqrt {| \mathcal {S} | A _ {\max}}}{1 - \gamma} \mathbb {E} [ N _ {2, 1, 1} ] \leq \frac {1 6 0 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \beta_ {k - z _ {k}, k - 1}.
$$

The Term $N_{2,2}$ . For any $k \geq z_k$ , we have

$$
N _ {2, 2} = \mathbb {E} [ \langle F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}), q _ {k} ^ {i} - q _ {k - z _ {k}} ^ {i} \rangle ]
$$

$$
\leq \mathbb {E} [ \underbrace {\| F ^ {i} (q _ {k - z _ {k}} ^ {i} , S _ {k} , A _ {k} ^ {i} , A _ {k} ^ {- i} , S _ {k + 1}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2}} _ {N _ {2, 2, 1}} \underbrace {\| q _ {k} ^ {i} - q _ {k - z _ {k}} ^ {i} \| _ {2}} _ {N _ {2, 2, 2}} ]
$$

Using Lemma A.9 and we have

$$
N _ {2, 2, 1}
$$

$$
= \| F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2}
$$

$$
= \| F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - F ^ {i} (\mathbf {0}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) + F ^ {i} (\mathbf {0}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) \| _ {2}
$$

$$
+ \| \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) - \bar {F} _ {k - z _ {k}} ^ {i} (\mathbf {0}) + \bar {F} _ {k - z _ {k}} ^ {i} (\mathbf {0}) \| _ {2}
$$

$$
\leq 2 \| q _ {k - z _ {k}} ^ {i} \| _ {2} + \frac {2}{1 - \gamma} \quad (\text { Lemma   A.1   and   Jensen's   inequality })
$$

$$
\leq \frac {2 | \mathcal {S} | ^ {1 / 2} A _ {\max} ^ {1 / 2}}{1 - \gamma} + \frac {2}{1 - \gamma} \tag {LemmaA.1}
$$

$$
\leq \frac {4 | \mathcal {S} | ^ {1 / 2} A _ {\mathrm{max}} ^ {1 / 2}}{1 - \gamma}.
$$

Moreover, we have by Lemma A.13 and Lemma A.1 that

$$
N _ {2, 2, 2} \leq 4 \alpha_ {k - z _ {k}, k - 1} \left(\| q _ {k} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \leq \frac {8 | \mathcal {S} | ^ {1 / 2} A _ {\max} ^ {1 / 2} \alpha_ {k - z _ {k} , k - 1}}{1 - \gamma}.
$$

Therefore, we have

$$
N _ {2, 2} \leq \mathbb {E} [ N _ {2, 2, 1} \times N _ {2, 2, 2} ] \leq \frac {3 2 | \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \alpha_ {k - z _ {k}, k - 1}.
$$

The Term $N_{2,3}$ . For any $k \geq z_k$ , we have

$$
\begin{array}{l} N _ {2, 3} = \mathbb {E} [ \langle F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}), \bar {q} _ {k - z _ {k}} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ] \\ \leq \frac {c ^ {\prime}}{2} \mathbb {E} [ \| F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2} ^ {2} ] + \frac {1}{2 c ^ {\prime}} \mathbb {E} [ \| \bar {q} _ {k - z _ {k}} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ], \\ \end{array}
$$

where $c' > 0$ is any positive real number. Since $N_{2,3,1} = N_{2,2,1}^2$ , we have

$$
N _ {2, 3, 1} \leq \frac {1 6 | \mathcal {S} | A _ {\mathrm{max}}}{(1 - \gamma) ^ {2}}.
$$

To control the term $N_{2,3,2}$ , using the explicit expression of $\bar{q}_k^i$ provided in Lemma A.9 (3) and we have

$$
\| \bar {q} _ {k - z _ {k}} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} = \sum_ {s} \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\pi_ {k - z _ {k}} ^ {- i} (s) - \pi_ {k} ^ {- i} (s)) \| _ {2} ^ {2}.
$$

Since

$$
\begin{array}{l} \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\pi_ {k} ^ {- i} (s) - \pi_ {k - z _ {k}} ^ {- i} (s)) \| _ {2} \leq \frac {A _ {\max}}{1 - \gamma} \| \pi_ {k} ^ {- i} (s) - \pi_ {k - z _ {k}} ^ {- i} (s) \| _ {2} \\ \leq \frac {4 A _ {\max} \beta_ {k - z _ {k} , k - 1}}{1 - \gamma} (\| \pi_ {k} ^ {- i} (s) \| _ {2} + 1) \tag {LemmaA.14} \\ \leq \frac {8 A _ {\mathrm{max}} \beta_ {k - z _ {k} , k - 1}}{1 - \gamma}, \\ \end{array}
$$

we have

$$
N _ {2, 3, 2} \leq \frac {6 4 | \mathcal {S} | A _ {\max} ^ {2} \beta_ {k - z _ {k} , k - 1} ^ {2}}{(1 - \gamma) ^ {2}}
$$

It follows that

$$
\begin{array}{l} N _ {2, 3} = \frac {c ^ {\prime}}{2} \mathbb {E} [ N _ {2, 3, 1} ] + \frac {1}{2 c ^ {\prime}} \mathbb {E} [ N _ {2, 3, 2} ] \\ \leq \frac {8 c ^ {\prime} | \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} + \frac {3 2 | \mathcal {S} | A _ {\max} ^ {2} \beta_ {k - z _ {k} , k - 1} ^ {2}}{c ^ {\prime} (1 - \gamma) ^ {2}} \\ \leq \frac {8 | \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \left(c ^ {\prime} + \frac {4 A _ {\max} \beta_ {k - z _ {k} , k - 1} ^ {2}}{c ^ {\prime}}\right) \\ = \frac {3 2 | \mathcal {S} | A _ {\max} ^ {3 / 2} \beta_ {k - z _ {k} , k - 1}}{(1 - \gamma) ^ {2}}, \\ \end{array}
$$

where the last line follows by choosing $c' = 2A_{\max}^{1/2}\beta_{k - z_k,k - 1}$ .

The Term $N_{2,4}$ . For any $k \geq 0$ , we have

$$
\begin{array}{l} N _ {2, 4} = \mathbb {E} [ \langle F ^ {i} (q _ {k} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}), q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ] \\ \leq \mathbb {E} [ \| F ^ {i} (q _ {k} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - F ^ {i} (q _ {k - z _ {k}} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) \| _ {2} \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ] \\ \leq \mathbb {E} [ \| q _ {k} ^ {i} - q _ {k - z _ {k}} ^ {i} \| _ {2} \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ] (LemmaA.9) \\ \leq 4 \alpha_ {k - z _ {k}, k - 1} \mathbb {E} \left[ \left(\| q _ {k} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} \right] (LemmaA.13) \\ \leq \frac {1 6 | \mathcal {S} | A _ {\max} \alpha_ {k - z _ {k} , k - 1}}{(1 - \gamma) ^ {2}}, \\ \end{array}
$$

where the last line follows from $\|q_{k}^{i}\|_{\infty}\leq\frac{1}{1-\gamma}$ and $\|\bar{q}_{k}^{i}\|_{\infty}\leq\frac{1}{1-\gamma}$ .

The Term $N_{2,5}$ . For any $k \geq 0$ , we have

$$
\begin{array}{l} N _ {2, 5} = \mathbb {E} [ \langle \bar {F} _ {k} ^ {i} (q _ {k} ^ {i}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}), q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \rangle ] \\ \leq \mathbb {E} [ \| \bar {F} _ {k} ^ {i} (q _ {k} ^ {i}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2} \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ] \\ \leq \mathbb {E} [ (\| \bar {F} _ {k} ^ {i} (q _ {k} ^ {i}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k} ^ {i}) \| _ {2} + \| \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k} ^ {i}) - \bar {F} _ {k - z _ {k}} ^ {i} (q _ {k - z _ {k}} ^ {i}) \| _ {2}) \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ] \\ \leq \mathbb {E} [ \| q _ {k} ^ {i} - q _ {k - z _ {k}} ^ {i} \| _ {2} \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ] + \frac {4 2 | \mathcal {S} | A _ {\max} ^ {1 / 2} \hat {L} _ {\tau} \beta_ {k - z _ {k} , k - 1}}{1 - \gamma} \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ] \\ \end{array}
$$

(Lemma A.9 and Eq. (60))

$$
\begin{array}{l} \leq 4 \alpha_ {k - z _ {k}, k - 1} \mathbb {E} \left[ \left(\| q _ {k} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} \right] \tag {LemmaA.13} \\ + \frac {4 2 | \mathcal {S} | A _ {\max} ^ {1 / 2} \hat {L} _ {\tau} \beta_ {k - z _ {k} , k - 1}}{1 - \gamma} \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ] \\ \leq \frac {1 6 | \mathcal {S} | A _ {\max} \alpha_ {k - z _ {k} , k - 1}}{(1 - \gamma) ^ {2}} + \frac {8 4 | \mathcal {S} | ^ {3 / 2} A _ {\max} \hat {L} _ {\tau} \beta_ {k - z _ {k} , k - 1}}{(1 - \gamma) ^ {2}} \\ \leq \frac {1 0 0 | \mathcal {S} | ^ {3 / 2} A _ {\max} \hat {L} _ {\tau} \alpha_ {k - z _ {k} , k - 1}}{(1 - \gamma) ^ {2}}. \\ \end{array}
$$

Finally, combining the upper bounds we derived for the terms $\{N_{2,j}\}_{1\leq j\leq 5}$ and we have

$$
\begin{array}{l} N _ {2} \leq \sum_ {j = 1} ^ {5} N _ {2, j} \\ \leq \frac {1 6 0 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \beta_ {k - z _ {k}, k - 1} + \frac {3 2 | \mathcal {S} | A _ {\max}}{(1 - \gamma) ^ {2}} \alpha_ {k - z _ {k}, k - 1} \\ + \frac {3 2 | \mathcal {S} | A _ {\max} ^ {3 / 2} \beta_ {k - z _ {k} , k - 1}}{(1 - \gamma) ^ {2}} + \frac {1 6 | \mathcal {S} | A _ {\max} \alpha_ {k - z _ {k} , k - 1}}{(1 - \gamma) ^ {2}} \\ + \frac {1 0 0 | \mathcal {S} | ^ {3 / 2} A _ {\max} \hat {L} _ {\tau} \alpha_ {k - z _ {k} , k - 1}}{(1 - \gamma) ^ {2}} \\ \leq \frac {3 4 0 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k - z _ {k}, k - 1}. \\ \end{array}
$$

# A.7.12 Proof of Lemma A.11

(1) For any $k \geq 0$ , using Lemma A.9 and we have

$$
\begin{array}{l} \| q _ {k + 1} ^ {i} - q _ {k} ^ {i} \| _ {2} ^ {2} = \alpha_ {k} ^ {2} \| F ^ {i} (q _ {k} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) \| _ {2} ^ {2} \\ = \alpha_ {k} ^ {2} \| F ^ {i} (q _ {k} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - F ^ {i} (\mathbf {0}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) \\ + F ^ {i} \left(\mathbf {0}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}\right) \| _ {2} ^ {2} \\ \leq \alpha_ {k} ^ {2} \left(\| q _ {k} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) ^ {2} \\ \leq \alpha_ {k} ^ {2} \left(\frac {\sqrt {| \mathcal {S} | A _ {\max}}}{1 - \gamma} + \frac {1}{1 - \gamma}\right) ^ {2} \quad (\| q _ {k} ^ {i} \| _ {\infty} \leq \frac {1}{1 - \gamma}) \\ \leq \frac {4 | \mathcal {S} | A _ {\mathrm{max}} \alpha_ {k} ^ {2}}{(1 - \gamma) ^ {2}}. \\ \end{array}
$$

The result follows by taking expectation on both sides of the previous inequality.

(2) For any $k \geq 0$ , we have by Lemma A.9 that

$$
\begin{array}{l} \| \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \| _ {2} ^ {2} = \sum_ {s} \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\pi_ {k + 1} ^ {- i} (s) - \pi_ {k} ^ {- i} (s)) \| _ {2} ^ {2} \\ = \beta_ {k} ^ {2} \sum_ {s} \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \pi_ {k} ^ {- i} (s)) \| _ {2} ^ {2} \\ \leq \beta_ {k} ^ {2} \sum_ {s} (\| \mathcal {T} ^ {i} (v ^ {i}) (s) \sigma_ {\tau} (q _ {k} ^ {- i} (s)) \| _ {2} + \| \mathcal {T} ^ {i} (v ^ {i}) (s) \pi_ {k} ^ {- i} (s) \| _ {2}) ^ {2} \\ \leq \frac {4 | \mathcal {S} | A _ {\mathrm{max}} \beta_ {k} ^ {2}}{(1 - \gamma) ^ {2}}. \\ \end{array}
$$

The result follows by taking expectation on both sides of the previous inequality.

(3) For any $k \geq 0$ , we have

$$
\langle q _ {k + 1} ^ {i} - q _ {k} ^ {i}, \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \rangle \leq \| q _ {k + 1} ^ {i} - q _ {k} ^ {i} \| _ {2} \| \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \| _ {2} \leq \frac {4 | \mathcal {S} | A _ {\max} \alpha_ {k} \beta_ {k}}{(1 - \gamma) ^ {2}},
$$

where the last inequality follows from Part (1) and Part (2) of this lemma. The result follows by taking expectation on both sides of the previous inequality.

(4) For any $k \geq 0$ , we have

$$
\begin{array}{l} \langle q _ {k} ^ {i} - \bar {q} _ {k} ^ {i}, \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \rangle \\ = \beta_ {k} \sum_ {s} \langle q _ {k} ^ {i} (s) - \bar {q} _ {k} ^ {i} (s), \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \pi_ {k} ^ {- i} (s)) \rangle \\ \leq \beta_ {k} \left(\frac {\hat {c} \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2}}{2} + \frac {\sum_ {s} \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \pi_ {k} ^ {- i} (s)) \| _ {2} ^ {2}}{2 \hat {c}}\right), \tag {62} \\ \end{array}
$$

where $\hat{c}$ is an arbitrary positive real number. We next analyze the second term on the RHS of the previous inequality. For any $s\in S$ , we have

$$
\begin{array}{l} \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \pi_ {k} ^ {- i} (s)) \| _ {2} \\ = \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \sigma_ {\tau} (\bar {q} _ {k} ^ {- i} (s)) + \sigma_ {\tau} (\mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s)) - \pi_ {k} ^ {- i} (s)) \| _ {2} \\ \leq \underbrace {\| \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \sigma_ {\tau} (\bar {q} _ {k} ^ {- i} (s))) \| _ {2}} _ {B _ {1}} \\ + \underbrace {\| \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (\mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s)) - \pi_ {k} ^ {- i} (s)) \| _ {2}} _ {B _ {2}}. \\ \end{array}
$$

Since the softmax operator $\sigma_{\tau}(\cdot)$ is $\frac{1}{\tau} -$ Lipschitz continuous with respect to $\| \cdot \| _2$ (Gao and Pavel, 2017, Proposition 4), we have

$$
\begin{array}{l} B _ {1} \leq \| \mathcal {T} ^ {i} (v ^ {i}) (s) \| _ {2} \| \sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \sigma_ {\tau} (\bar {q} _ {k} ^ {- i} (s)) \| _ {2} \\ \leq \frac {A _ {\max}}{\tau (1 - \gamma)} \| q _ {k} ^ {- i} (s) - \bar {q} _ {k} ^ {- i} (s) \| _ {2}. \\ \end{array}
$$

We next analyze the term $B_{2}$ . Using the quadratic growth property of strongly convex functions and we have

$$
B _ {2} = \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (\mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s)) - \pi_ {k} ^ {- i} (s)) \| _ {2}
$$

$$
\leq \| \mathcal {T} ^ {i} (v ^ {i}) (s) \| _ {2} \| \sigma_ {\tau} (\mathcal {T} ^ {- i} (v ^ {- i}) (s) \pi_ {k} ^ {i} (s)) - \pi_ {k} ^ {- i} (s) \| _ {2}
$$

$$
\leq \frac {\sqrt {2} A _ {\max}}{\sqrt {\tau} (1 - \gamma)} V _ {v, s} ^ {1 / 2} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)).
$$

Combine the upper bounds we obtained for the terms $B_{1}$ and $B_{2}$ and we obtain

$$
\begin{array}{l} \sum_ {s} \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \pi_ {k} ^ {- i} (s)) \| _ {2} ^ {2} \\ \leq \sum_ {s} (B _ {1} + B _ {2}) ^ {2} \\ \leq 2 \sum_ {s} (B _ {1} ^ {2} + B _ {2} ^ {2}) \\ \leq 2 \sum_ {s} \left(\frac {A _ {\max} ^ {2}}{\tau^ {2} (1 - \gamma) ^ {2}} \| q _ {k} ^ {- i} (s)) - \bar {q} _ {k} ^ {- i} (s) \| _ {2} ^ {2} + \frac {2 A _ {\max} ^ {2}}{\tau (1 - \gamma) ^ {2}} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s))\right) \\ = \frac {2 A _ {\mathrm{max}} ^ {2}}{\tau^ {2} (1 - \gamma) ^ {2}} \| q _ {k} ^ {- i} - \bar {q} _ {k} ^ {- i} \| _ {2} ^ {2} + \frac {4 A _ {\mathrm{max}} ^ {2}}{\tau (1 - \gamma) ^ {2}} \sum_ {s} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)). \\ \end{array}
$$

Coming back to Eq. (62), using the previous inequality and we have

$$
\begin{array}{l} \langle q _ {k} ^ {i} - \bar {q} _ {k} ^ {i}, \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \rangle \\ \leq \beta_ {k} \left(\frac {\hat {c} \| q _ {k} ^ {i} - \overline {{q}} _ {k} ^ {i} \| _ {2} ^ {2}}{2} + \frac {\sum_ {s} \| \mathcal {T} ^ {i} (v ^ {i}) (s) (\sigma_ {\tau} (q _ {k} ^ {- i} (s)) - \pi_ {k} ^ {- i} (s)) \| _ {2} ^ {2}}{2 \hat {c}}\right) \\ \leq \beta_ {k} \left(\frac {\hat {c} \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2}}{2} + \frac {A _ {\max} ^ {2}}{\hat {c} \tau^ {2} (1 - \gamma) ^ {2}} \| q _ {k} ^ {- i} - \bar {q} _ {k} ^ {- i} \| _ {2} ^ {2} \right. \\ \left. + \frac {2 A _ {\mathrm{max}} ^ {2}}{\hat {c} \tau (1 - \gamma) ^ {2}} \sum_ {s} V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s))\right). \\ \end{array}
$$

Choosing $\hat{c} = \frac{32A_{\max}^2}{\tau(1 - \gamma)^2}$ in the previous inequality and then taking total expectation, and we obtain

$$
\mathbb {E} [ \langle q _ {k} ^ {i} - \bar {q} _ {k} ^ {i}, \bar {q} _ {k} ^ {i} - \bar {q} _ {k + 1} ^ {i} \rangle ] \leq \frac {1 7 A _ {\max} ^ {2} \beta_ {k}}{\tau (1 - \gamma) ^ {2}} \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ] + \frac {\beta_ {k}}{1 6} \sum_ {s} \mathbb {E} [ V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) ].
$$

# A.7.13 Proof of Lemma A.12

For $i \in \{1, 2\}$ , we have from Eq. (31), Eq. (32), Lemma A.10, and Lemma A.11 that

$$
\begin{array}{l} \mathbb {E} [ \| q _ {k + 1} ^ {i} - \bar {q} _ {k + 1} ^ {i} \| _ {2} ^ {2} ] \\ \leq \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ] - \alpha_ {k} c _ {\tau} \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ] \\ + \frac {4 | \mathcal {S} | A _ {\mathrm{max}}}{(1 - \gamma) ^ {2}} (\alpha_ {k} ^ {2} + \alpha_ {k} \beta_ {k} + \beta_ {k} ^ {2}) \\ + \frac {3 4 0 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} \alpha_ {k - z _ {k}, k - 1} \\ + \frac {1 7 A _ {\mathrm{max}} ^ {2} \beta_ {k}}{\tau (1 - \gamma) ^ {2}} \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ] + \frac {\beta_ {k}}{1 6} \sum_ {s} \mathbb {E} [ V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) ] \\ \leq \left(1 - \alpha_ {k} c _ {\tau} + \frac {1 7 A _ {\mathrm{max}} ^ {2} \beta_ {k}}{\tau (1 - \gamma) ^ {2}}\right) \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ] \\ + \frac {1 2 | \mathcal {S} | A _ {\mathrm{max}}}{(1 - \gamma) ^ {2}} \alpha_ {k} ^ {2} + \frac {3 4 0 | \mathcal {S} | ^ {3 / 2} A _ {\mathrm{max}} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} \alpha_ {k - z _ {k}, k - 1} \\ + \frac {\beta_ {k}}{1 6} \sum_ {s} \mathbb {E} [ V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) ] \\ \end{array}
$$

$$
\begin{array}{l} \leq \left(1 - \alpha_ {k} c _ {\tau} + \frac {1 7 A _ {\max} ^ {2} \beta_ {k}}{\tau (1 - \gamma) ^ {2}}\right) \mathbb {E} [ \| q _ {k} ^ {i} - \bar {q} _ {k} ^ {i} \| _ {2} ^ {2} ] \\ + \frac {3 5 2 | \mathcal {S} | ^ {3 / 2} A _ {\max} ^ {3 / 2} \hat {L} _ {\tau}}{(1 - \gamma) ^ {2}} z _ {k} \alpha_ {k} \alpha_ {k - z _ {k}, k - 1} + \frac {\beta_ {k}}{1 6} \sum_ {s} \mathbb {E} [ V _ {v, s} (\pi_ {k} ^ {i} (s), \pi_ {k} ^ {- i} (s)) ], \\ \end{array}
$$

where the second inequality follows from $\beta_{k} = c_{\alpha ,\beta}\alpha_{k}$ with $c_{\alpha ,\beta}\leq 1$

# A.7.14 Proof of Lemma A.13

For any $k \in [k_1, k_2 - 1]$ , we have

$$
\begin{array}{l} \left\| q _ {k + 1} ^ {i} \right\| _ {2} - \left\| q _ {k} ^ {i} \right\| _ {2} \leq \left\| q _ {k + 1} ^ {i} - q _ {k} ^ {i} \right\| _ {2} \quad (\text { triangle   inequality }) \\ = \alpha_ {k} \| F ^ {i} (q _ {k} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) \| _ {2} \\ \leq \alpha_ {k} \| F ^ {i} (q _ {k} ^ {i}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) - F ^ {i} (\mathbf {0}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) \| _ {2} \\ + \alpha_ {k} \| F ^ {i} (\mathbf {0}, S _ {k}, A _ {k} ^ {i}, A _ {k} ^ {- i}, S _ {k + 1}) \| _ {2} \\ \leq \alpha_ {k} \left(\| q _ {k} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right), \tag {63} \\ \end{array}
$$

where the last inequality follows from Lemma A.9. Adding $1 / (1 - \gamma)$ to both sides of the previous inequality and we have

$$
\left\| q _ {k + 1} ^ {i} \right\| _ {2} + \frac {1}{1 - \gamma} \leq \left(1 + \alpha_ {k}\right) \left(\left\| q _ {k} ^ {i} \right\| _ {2} + \frac {1}{1 - \gamma}\right).
$$

Repeatedly using the previous inequality and we have for all $k \in [k_1, k_2]$ :

$$
\left\| q _ {k} ^ {i} \right\| _ {2} \leq \prod_ {j = k _ {1}} ^ {k - 1} \left(1 + \alpha_ {j}\right) \left(\left\| q _ {k _ {1}} ^ {i} \right\| _ {2} + \frac {1}{1 - \gamma}\right) - \frac {1}{1 - \gamma}.
$$

Since $1 + x \leq e^{x} \leq 1 + 2x$ for all $x \in [0, 1/2]$ and $\alpha_{k_1, k_2 - 1} \leq 1/4$ , we have

$$
\prod_ {j = k _ {1}} ^ {k - 1} \left(1 + \alpha_ {j}\right) \leq \exp \left(\alpha_ {k _ {1}, k - 1}\right) \leq 1 + 2 \alpha_ {k _ {1}, k - 1}.
$$

As a result, we have for all $k \in [k_1, k_2]$ that

$$
\| q _ {k} ^ {i} \| _ {2} \leq (1 + 2 \alpha_ {k _ {1}, k - 1}) \| q _ {k _ {1}} ^ {i} \| _ {2} + \frac {2 \alpha_ {k _ {1} , k - 1}}{1 - \gamma}.
$$

Using the previous inequality in Eq. (63) and we have for any $k \in [k_1, k_2 - 1]$ :

$$
\begin{array}{l} \left\| q _ {k + 1} ^ {i} - q _ {k} ^ {i} \right\| _ {2} \leq \alpha_ {k} \left(\left\| q _ {k} ^ {i} \right\| _ {2} + \frac {1}{1 - \gamma}\right) \\ \leq \alpha_ {k} (1 + 2 \alpha_ {k _ {1}, k - 1}) \| q _ {k _ {1}} ^ {i} \| _ {2} + \frac {2 \alpha_ {k} \alpha_ {k _ {1} , k - 1}}{1 - \gamma} \\ \leq 2 \alpha_ {k} \left(\| q _ {k _ {1}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right), \\ \end{array}
$$

where the last line follows from $\alpha_{k_1,k - 1}\leq 1 / 4$ . Therefore, we have for any $k\in [k_1,k_2]$ :

$$
\left\| q _ {k} ^ {i} - q _ {k _ {1}} ^ {i} \right\| _ {2} \leq \sum_ {j = k _ {1}} ^ {k - 1} \left\| q _ {j + 1} ^ {i} - q _ {j} ^ {i} \right\| _ {2}
$$

$$
\begin{array}{l} \leq 2 \sum_ {j = k _ {1}} ^ {k - 1} \alpha_ {j} \left(\| q _ {k _ {1}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \\ = 2 \alpha_ {k _ {1}, k - 1} \left(\| q _ {k _ {1}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \\ \leq 2 \alpha_ {k _ {1}, k _ {2} - 1} \left(\| q _ {k _ {1}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right), \\ \end{array}
$$

where the last line follows from $\alpha_{k_1,k - 1}\leq \alpha_{k_1,k_2 - 1}$ . This proves the first claimed inequality.

To prove the second claimed inequality, note that

$$
\begin{array}{l} \left\| q _ {k _ {2}} ^ {i} - q _ {k _ {1}} ^ {i} \right\| _ {2} \leq 2 \alpha_ {k _ {1}, k _ {2} - 1} \left(\left\| q _ {k _ {1}} ^ {i} \right\| _ {2} + \frac {1}{1 - \gamma}\right) \\ \leq 2 \alpha_ {k _ {1}, k _ {2} - 1} \left(\| q _ {k _ {1}} ^ {i} - q _ {k _ {2}} ^ {i} \| _ {2} + \| q _ {k _ {2}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \\ \leq \frac {1}{2} \| q _ {k _ {2}} ^ {i} - q _ {k _ {1}} ^ {i} \| _ {2} + 2 \alpha_ {k _ {1}, k _ {2} - 1} \left(\| q _ {k _ {2}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right), \\ \end{array}
$$

we have $\| q_{k_2}^i - q_{k_1}^i\| _2\leq 4\alpha_{k_1,k_2 - 1}(\| q_{k_2}^i\| _2 + 1 / (1 - \gamma))$ . Therefore, we have for any $k\in [k_1,k_2]$

$$
\begin{array}{l} \left\| q _ {k} ^ {i} - q _ {k _ {1}} ^ {i} \right\| _ {2} \leq 2 \alpha_ {k _ {1}, k _ {2} - 1} \left(\left\| q _ {k _ {1}} ^ {i} \right\| _ {2} + \frac {1}{1 - \gamma}\right) \\ \leq 2 \alpha_ {k _ {1}, k _ {2} - 1} \left(\| q _ {k _ {1}} ^ {i} - q _ {k _ {2}} ^ {i} \| _ {2} + \| q _ {k _ {2}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \\ \leq 2 \alpha_ {k _ {1}, k _ {2} - 1} \left(4 \alpha_ {k _ {1}, k _ {2} - 1} \left(\| q _ {k _ {2}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) + \| q _ {k _ {2}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right) \\ \leq 4 \alpha_ {k _ {1}, k _ {2} - 1} \left(\| q _ {k _ {2}} ^ {i} \| _ {2} + \frac {1}{1 - \gamma}\right), \\ \end{array}
$$

where the last inequality follows from $\alpha_{k_1,k_2 - 1}\leq 1 / 4$ . The proof is now complete.

# A.7.15 Proof of Lemma A.14

For any $k\geq 0$ and $s\in \mathcal{S}$ , we have

$$
\begin{array}{l} \| \pi_ {k + 1} ^ {i} (s) \| _ {2} - \| \pi_ {k} ^ {i} (s) \| _ {2} \leq \| \pi_ {k + 1} ^ {i} (s) - \pi_ {k} ^ {i} (s) \| _ {2} \\ = \beta_ {k} \| \sigma_ {\tau} (q _ {k} ^ {i} (s)) - \pi_ {k} ^ {i} (s) \| _ {2} \\ \leq \beta_ {k} (\| \pi_ {k} ^ {i} (s) \| _ {2} + 1). \\ \end{array}
$$

The rest of the proof is identical to that of Lemma A.13 (after Eq. (63)).

# B Proof of Theorem 2.1

Note that Algorithm 1 is a special case of Algorithm 3 when the Markov game has only one state, and the inputs $v^{i}$ and $v^{-i}$ satisfy $v^{i} = v^{-i} = 0$ . Therefore, Lemma A.8 and Lemma A.12 are both applicable, and are restated in the following.

Lemma B.1. The following inequality holds for all $k \geq 0$ .

$$
\begin{array}{l} \mathbb {E} [ V _ {\mathcal {R}} (\pi_ {k + 1} ^ {i}, \pi_ {k + 1} ^ {- i}) ] \leq \left(1 - \frac {3 \beta_ {k}}{4}\right) \mathbb {E} [ V _ {\mathcal {R}} (\pi_ {k} ^ {i}, \pi_ {k} ^ {- i}) ] + \frac {4 A _ {\max} ^ {2} \beta_ {k} ^ {2}}{\ell_ {\tau}} \\ + \frac {2 5 6 A _ {\max} ^ {2} \beta_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3}} \sum_ {i = 1, 2} \mathbb {E} [ \| q _ {k} ^ {i} - \mathcal {R} ^ {i} \pi_ {k} ^ {- i} \| _ {2} ^ {2} ]. \tag {64} \\ \end{array}
$$

Lemma B.2. The following inequality holds for all $k \geq 0$ :

$$
\begin{array}{l} \sum_ {i = 1, 2} \mathbb {E} [ \| q _ {k + 1} ^ {i} - \mathcal {R} ^ {i} \pi_ {k + 1} ^ {- i} \| _ {2} ^ {2} ] \leq \left(1 - \ell_ {\tau} \alpha_ {k} + \frac {1 7 A _ {\max} ^ {2} \beta_ {k}}{\tau}\right) \sum_ {i = 1, 2} \mathbb {E} [ \| q _ {k} ^ {i} - \mathcal {R} ^ {i} \pi_ {k} ^ {- i} \| _ {2} ^ {2} ] \\ + 3 5 2 A _ {\max} ^ {3 / 2} \alpha_ {k} ^ {2} + \frac {\beta_ {k}}{8} \mathbb {E} [ V _ {\mathcal {R}} (\pi_ {k} ^ {i}, \pi_ {k} ^ {- i}) ]. \tag {65} \\ \end{array}
$$

Adding up Eqs. (65) and (64) and we have for any $k \geq 0$ that

$$
\begin{array}{l} \sum_ {i = 1, 2} \mathbb {E} [ \| q _ {k + 1} ^ {i} - \mathcal {R} ^ {i} \pi_ {k + 1} ^ {- i} \| _ {2} ^ {2} ] + \mathbb {E} [ V _ {\mathcal {R}} (\pi_ {k + 1} ^ {i}, \pi_ {k + 1} ^ {- i}) ] \\ \leq \left(1 - \ell_ {\tau} \alpha_ {k} + \frac {2 8 0 A _ {\max} ^ {2} \beta_ {k}}{\ell_ {\tau} ^ {2} \tau^ {3}}\right) \sum_ {i = 1, 2} \mathbb {E} [ \| q _ {k} ^ {i} - \mathcal {R} ^ {i} \pi_ {k} ^ {- i} \| _ {2} ^ {2} ] \\ + \left(1 - \frac {\beta_ {k}}{2}\right) \mathbb {E} [ V _ {\mathcal {R}} (\pi_ {k} ^ {i}, \pi_ {k} ^ {- i}) ] + \frac {4 A _ {\max} ^ {2} \beta_ {k} ^ {2}}{\ell_ {\tau}} + 3 5 2 A _ {\max} ^ {3 / 2} \alpha_ {k} ^ {2} \\ \leq \left(1 - \frac {c _ {\alpha , \beta} \alpha_ {k}}{2}\right) (\mathbb {E} [ \| q _ {k} ^ {i} - \mathcal {R} ^ {i} \pi_ {k} ^ {- i} \| _ {2} ^ {2} ] + \mathbb {E} [ V _ {\mathcal {R}} (\pi_ {k} ^ {i}, \pi_ {k} ^ {- i}) ]) \\ + \frac {4 A _ {\mathrm{max}} ^ {2} c _ {\alpha , \beta} ^ {2} \alpha_ {k} ^ {2}}{\ell_ {\tau}} + 3 5 6 A _ {\mathrm{max}} ^ {3 / 2} \alpha_ {k} ^ {2}, \\ \end{array}
$$

where the last line follows from Condition A.1. Denote $M_{k} = \mathbb{E}[V_{\mathcal{R}}(\pi_{k}^{i}, \pi_{k}^{-i})] + \sum_{i=1,2} \mathbb{E}[\|q_{k}^{i} - \mathcal{R}^{i}\pi_{k}^{-i}\|_{2}^{2}]$ . The previous inequality reads

$$
M _ {k + 1} \leq \left(1 - \frac {c _ {\alpha , \beta} \alpha_ {k}}{2}\right) M _ {k} + 3 5 6 A _ {\max} ^ {3 / 2} \alpha_ {k} ^ {2}.
$$

Repeatedly using the previous inequality and we have for all $k \geq 0$ that

$$
M _ {k} \leq \prod_ {m = 0} ^ {k - 1} \left(1 - \frac {c _ {\alpha , \beta} \alpha_ {m}}{2}\right) M _ {0} + 3 5 6 A _ {\max} ^ {3 / 2} \sum_ {n = 0} ^ {k - 1} \alpha_ {n} ^ {2} \prod_ {m = n + 1} ^ {k - 1} \left(1 - \frac {c _ {\alpha , \beta} \alpha_ {m}}{2}\right). \tag {66}
$$

Due to our initialization, we have

$$
M _ {0} = V _ {\mathcal {R}} \left(\pi_ {0} ^ {i}, \pi_ {0} ^ {- i}\right) + \sum_ {i = 1, 2} \| q _ {0} ^ {i} - \mathcal {R} ^ {i} \pi_ {0} ^ {- i} \| _ {2} ^ {2} \leq 3.
$$

It remains to bound $\prod_{m=0}^{k-1}\left(1 - \frac{c_{\alpha,\beta}\alpha_m}{2}\right)$ and $\sum_{n=0}^{k-1}\alpha_n^2\prod_{m=n+1}^{k-1}\left(1 - \frac{c_{\alpha,\beta}\alpha_m}{2}\right)$ when $\alpha_k$ is explicitly chosen. Using results in (Chen et al., 2021b, Appendix A.2) and we have the following inequalities:

(1) When using constant stepsize, i.e., $\alpha_{k} \equiv \alpha$ , we have for all $k \geq 0$ that

$$
M _ {k} \leq 3 \left(1 - \frac {c _ {\alpha , \beta} \alpha}{2}\right) ^ {k} + \frac {7 1 2 A _ {\max} ^ {3 / 2} \alpha}{c _ {\alpha , \beta}}.
$$

(2) When $\alpha_{k} = \frac{\alpha}{k + h}$ with $\alpha c_{\alpha, \beta} = 4$ and $h$ chosen such that $\alpha / h < 1$ , we have for all $k \geq 0$ that

$$
M _ {k} \leq 3 \left(\frac {h}{k + h}\right) ^ {2} + 3 5 6 A _ {\max} ^ {3 / 2} \frac {1 6 e}{c _ {\alpha , \beta}} \frac {\alpha}{k + h}.
$$

(3) When $\alpha_{k} = \frac{\alpha}{(k + h)^{z}}$ , where $\alpha > 0, z \in (0,1)$ , and $h \geq [4z / (c_{\alpha,\beta}\alpha)]^{1/(1-z)}$ , we have for all $k \geq 0$ that

$$
M _ {k} \leq 3 \exp \left(- \frac {\alpha}{2 c _ {\alpha , \beta} (1 - z)} ((k + h) ^ {1 - z} - h ^ {1 - z})\right) + 3 5 6 A _ {\max} ^ {3 / 2} \frac {4}{c _ {\alpha , \beta}} \frac {\alpha}{(k + h) ^ {z}}.
$$

The result follows by observing that

$$
\begin{array}{l} \mathbb {E} [ \mathbf {N G} (\pi_ {K} ^ {i}, \pi_ {K} ^ {- i}) ] \\ = \mathbb {E} \left[ \sum_ {i = 1, 2} \left(\max _ {\hat {\pi} ^ {i}} (\hat {\pi} ^ {i}) ^ {\top} \mathcal {R} ^ {i} \pi_ {K} ^ {- i} - (\pi_ {K} ^ {i}) ^ {\top} \mathcal {R} ^ {i} \pi_ {K} ^ {- i}\right) \right] \\ \leq \mathbb {E} \left[ \sum_ {i = 1, 2} \left(\max _ {\hat {\pi} ^ {i}} \left\{\left(\hat {\pi} ^ {i}\right) ^ {\top} \mathcal {R} ^ {i} \pi_ {K} ^ {- i} + \tau \nu \left(\hat {\pi} ^ {i}\right) \right\} - \left(\pi_ {K} ^ {i}\right) ^ {\top} \mathcal {R} ^ {i} \pi_ {K} ^ {- i}\right) \right] \\ \leq \mathbb {E} \left[ \sum_ {i = 1, 2} \left(\max _ {\hat {\pi} ^ {i}} \left\{(\hat {\pi} ^ {i}) ^ {\top} \mathcal {R} ^ {i} \pi_ {K} ^ {- i} + \tau \nu (\hat {\pi} ^ {i}) \right\} - (\pi_ {K} ^ {i}) ^ {\top} \mathcal {R} ^ {i} \pi_ {K} ^ {- i} - \tau \nu (\pi_ {K} ^ {i})\right) \right] + \tau \log (A _ {\max} ^ {2}) \\ = \mathbb {E} [ V _ {\mathcal {R}} (\pi_ {K} ^ {i}, \pi_ {K} ^ {- i}) ] + \tau \log (A _ {\max} ^ {2}) \\ \leq M _ {k} + 2 \tau \log (A _ {\max}). \\ \end{array}
$$

# C Proof of Corollary 2.1.2 and Corollary 3.2.2

We first consider Corollary 2.1.2. The following proof idea was previous used in Sayin et al. (2021) to show the rationality of their decentralized Q-learning algorithm.

Observe that Theorem 2.1 can be easily generalized to the case where the reward is corrupted by noise. Specifically, suppose that player $i$ takes action $a^i$ and player $-i$ takes action $a^{-i}$ . Instead of assuming player $i$ receives a deterministic reward $\mathcal{R}^i(a^i,a^{-i})$ , we assume that player $i$ receives a random reward $r^i(a^i,a^{-i},\xi)$ , where $\xi \in \Xi$ (where $\Xi$ is a finite set) is a random variable with distribution $\mu_{\xi}$ , and is independent of everything else. The proof is identical as long as $r^i + r^{-i} = 0$ , and the reward is uniformly bounded, i.e., $\max_{a^i,a^{-i},\xi}|r^i(a^i,a^{-i},\xi)| < \infty$ .

Now consider the case where player $i$ 's opponent follows a stationary policy $\pi^{-i}$ . We incorporate the randomness of player $-i$ 's action into the model and introduce a fictitious opponent with only one action $a^{*}$ . In particular, let $\hat{r}^i(a^i,a^*,a^{-i}) = \mathcal{R}^i(a^i,a^{-i})$ for all $a^i$ and $a^{-i}$ , and let $\hat{p}(s' \mid s,a^i,a^*) = \sum_{\pi^{-i}(a^{-i}|s)} p(s' \mid a^i,a^{-i},s)$ . Now the problem can be reformulated as player $i$ playing against the fictitious player with a single action $a^{*}$ , with reward function $\hat{r}^i (i\in \{1,2\})$ and transition probabilities $\hat{p}$ . Applying Theorem 2.1 and we have the $\mathcal{O}(1 / \epsilon)$ sample complexity for player $i$ to find its best response against $\pi^{-i}$ , up to a smoothing bias.

The proof of Corollary 3.2.2 is identical to that of Corollary 2.1.2.

# D On the Mixing Time of MDPs with Almost Deterministic Policies

Consider an MDP with two states $s_1, s_2$ and two actions $a_1, a_2$ . The transition probability matrix $P_1$ of taking action $a_1$ is the identity matrix $I_2$ , and the transition probability matrix $P_2$ of taking action $a_2$ is $P_2 = [0, 1; 1, 0]$ . Given $\alpha \in (1/2, 1)$ , let $\pi_\alpha$ be a policy such that $\pi(a_1|s) = \alpha$ and $\pi(a_2|s) = 1 - \alpha$ for any $s \in \{s_1, s_2\}$ . Denote $P_\alpha$ as the transition probability matrix under $\pi_\alpha$ . It is easy to see that

$$
P _ {\alpha} = \left[ \begin{array}{c c} \alpha & 1 - \alpha \\ 1 - \alpha & \alpha \end{array} \right].
$$

Since $P_{\alpha}$ is a doubly stochastic matrix, and has strictly positive entries, it has a unique stationary distribution $\mu = \mathbf{1}^{\top} / 2$ .

We next compute a lower bound of the mixing time of the $\pi_{\alpha}$ -induced Markov chain. Let $e_1 = [1,0]^{\top}$ be the initial distribution of the states, and denote $[x_k, 1 - x_k]^{\top}$ as the distribution of the states at time step $k$ . Then we have

$$
x _ {k + 1} = x _ {k} \alpha + (1 - x _ {k}) (1 - \alpha)
$$

$$
\begin{array}{l} = (2 \alpha - 1) x _ {k} + 1 - \alpha \\ = (2 \alpha - 1) ^ {k + 1} x _ {0} + \sum_ {i = 0} ^ {k} (1 - \alpha) (2 \alpha - 1) ^ {k - i} \\ = \frac {1}{2} + \frac {(2 \alpha - 1) ^ {k + 1}}{2}. \\ \end{array}
$$

It follows that

$$
\begin{array}{l} t _ {\pi_ {\alpha}, \eta} = \min _ {k \geq 0} \left\{\max _ {\mu_ {0} \in \Delta^ {2}} \left\| \mu_ {0} ^ {\top} P _ {\alpha} ^ {k} - \mathbf {1} ^ {\top} / 2 \right\| _ {\mathrm{TV}} \leq \eta \right\} \\ \geq \min _ {k \geq 0} \left\{\left\| e _ {1} ^ {\top} P _ {\alpha} ^ {k} - \mathbf {1} ^ {\top} / 2 \right\| _ {\mathrm{TV}} \leq \eta \right\} \\ = \min _ {k \geq 0} \left\{\left(2 \alpha - 1\right) ^ {k} \leq 2 \eta \right\} \\ \geq \frac {\log (1 / 2 \eta)}{\log (1 / (2 \alpha - 1))} - 1, \\ \end{array}
$$

which implies $\lim_{\alpha \to 1} t_{\alpha, \eta} = \infty$ . Therefore, as the policies become deterministic, the mixing time of the associated Markov chain can approach infinity.