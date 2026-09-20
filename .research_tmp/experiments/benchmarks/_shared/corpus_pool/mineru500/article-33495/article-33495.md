# Asymptotic Extinction in Large Coordination Games

Desmond Chan $^{1}$ \*, Bart de Keijzer $^{1}$ , Tobias Galla $^{2}$ , Stefanos Leonardos $^{1}$ , Carmine Ventre $^{1}$

$^{1}$ King's College London, $^{2}$ Institute for Cross-Disciplinary Physics and Complex Systems (IFISC, CSIC-UIB) desmond.chan@kcl.ac.uk, bart.de\_keijzer@kcl.ac.uk, tobias.galla@ifisc.uib-csic.es, stefanos.leonardos@kcl.ac.uk, carmine.ventre@kcl.ac.uk

# Abstract

We study the exploration-exploitation trade-off for large multiplayer coordination games where players strategise via Q-Learning, a common learning framework in multi-agent reinforcement learning. Q-Learning is known to have two shortcomings, namely non-convergence and potential equilibrium selection problems, when there are multiple fixed points, called Quantal Response Equilibria (QRE). Furthermore, whilst QRE have full support for finite games, it is not clear how Q-Learning behaves as the game becomes large. In this paper, we characterise the critical exploration rate that guarantees convergence to a unique fixed point, addressing the two shortcomings above. Using a generating-functional method, we show that this rate increases with the number of players and the alignment of their payoffs. For many-player coordination games with perfectly aligned payoffs, this exploration rate is roughly twice that of p-player zero-sum games. As for large games, we provide a structural result for QRE, which suggests that as the game size increases, Q-Learning converges to a QRE near the boundary of the simplex of the action space, a phenomenon we term asymptotic extinction, where a constant fraction of the actions are played with zero probability at a rate $o(1/N)$ for an N-action game.

# Introduction

Multi-agent systems are an increasingly relevant area in AI research. They typically consist of learning agents trying to coordinate to reach specific outcomes, such scenarios are prevalent in fields ranging from economics (March 1991), robotics and distributed systems (Panait and Luke 2005). A key challenge in these settings is balancing exploration and exploitation in high-dimensional action spaces. Exploration is required for the discovery of optimal strategies; this can come at the expense of short-term rewards. Effectively exploring such complex spaces can be a critical point of failure, preventing convergence to “good” outcomes.

In multi-agent reinforcement learning (MARL), coordination scenarios consisting of interacting agents, can be represented as games. Throughout this work, we will focus on Q-Learning, one of the most widely used methods in MARL, as it provides a framework to analyse the exploitationexploration trade-off algorithmically. The fixed points of Q-Learning are Quantal Response Equilibria (QRE) (Leonardos, Piliouras, and Spendlove 2021; Leonardos and Piliouras 2022), which always assign positive probability to all actions of finite games and for low exploration rates approximate the Nash Equilibria (NE) for the underlying game.

Coordination games are characterised by players' payoffs being aligned in a manner to incentivise picking mutually beneficial actions. In such settings, agents following the Q-Learning algorithm over a fixed game can exhibit two different dynamical behaviours: (i) Convergence to a unique fixed point (at high exploration rates) – where agents reach the same, joint fixed point regardless of initial conditions; and, (ii) Convergence to multiple equilibria (at low exploration rates) – the final strategy profiles agents converge to is dependent on initial conditions. The effectiveness of Q-Learning is influenced by which of these two outcomes emerges during the learning process. This paper investigates the dynamical behaviour of Q-Learning over large, multiplayer coordination games where the payoff matrices are randomly drawn from multivariate Gaussians. In each game, the payoffs matrices are randomly-generated and held fixed. Players are assigned random initial strategies and we study the emerging dynamics.

Related Work and Our Contribution. The study of random competitive games was considered in (Galla and Farmer 2013), which was inspired by replicator models in the context of biological evolution (Opper and Diederich 1992; Galla 2006). Following this line of work, we characterise, through the use of random games, the typical behaviour in complex, coordination games a priori the learning process. Our work complements and extends the results in (Sanders, Farmer, and Galla 2018) to coordination games.

Our theoretical analysis suggest in coordination games with large action sets of size N, a constant, non-zero proportion of actions are played with a frequency of $o(1/N)$ . We call this effect asymptotic extinction. This extinction rate is asymptotic and varies with the model parameters. While simulations cannot fully quantitatively confirm this effect, simulation results are broadly consistent with our theoretical analysis. Taking this effect into account, a minimum exploration rate $T_{crit}$ , which guarantees convergence to a unique fixed point can be found over games with varying number of players and degree of payoff correlation.

# Preliminaries

Multiplayer normal form games. A p player, N action normal form game, G, is defined by a tuple, $\mathcal{G} = (\mathcal{P}, \mathcal{A}, \Pi)$ , where $P := \{1, 2, \ldots, p\}$ is the set of players, and $A := \{1, \ldots, N\}$ is a set of actions. Each player in G chooses an action, resulting in an action profile, i.e., an element $a \in A^{p}$ . Thus, for an action profile a, we write $a_{i}$ to refer to Player i's chosen action in a. Furthermore, we use $a_{-i}$ to refer to vector obtained from a by removing the ith coordinate. The notation $(b, \mathbf{a}_{-i})$ then refers to the vector obtained from a by replacing the value at coordinate i with b (so that $\mathbf{a} = (a_{i}, \mathbf{a}_{-i})$ ). For each action profile, every player experiences a certain payoff which players want to maximise. Payoffs are specified by the payoff function $\Pi : A^{p} \to R^{p}$ , where for $i \in P$ and $a \in A^{p}$ , the payoff for Player i on action profile a is given by $\Pi(\mathbf{a})_{i}$ .

Players can choose their actions probabilistically. This gives rise to the notion of a strategy x, which is a probability distribution over A. Thus, x is a point on the $(N-1)$ -simplex $\Delta_{N} = \{x \in R_{\geq 0}^{n} : x_{1} + \cdots + x_{N} = 1\}$ . Interior points of the simplex correspond to strategies where all actions are played with positive probability $(x_{i} > 0, \forall i)$ . Points not in the interior are known as boundary points. Similar to the notion of an action profile, a strategy profile is a choice of strategy by each of the players, and is hence given by an element $\mathbf{x} = (\mathbf{x}^{1}, \ldots, \mathbf{x}^{p}) \in \Delta_{N}^{p}$ . A strategy profile x, induces a probability distribution over action profiles, and we define the payoff $R(\mathbf{x})^{i}$ of Player i for x from $\Pi$ , as the expected value of the payoff of the random action profile:

$$
R (\mathbf {x}) ^ {i} = \sum_ {\mathbf {a} \in \mathcal {A} ^ {p}} \Pi (\mathbf {a}) _ {i} \prod_ {i \in \mathcal {P}} x _ {a _ {i}} ^ {i}. \tag {1}
$$

Similar to our notation for action profiles, for a strategy profile x we use $x^{-i}$ to refer to the vector obtained from x by removing the strategy of Player i from it. The notation $(y, \mathbf{x}^{-i})$ then refers to the vector obtained from x by replacing $x^{i}$ with $y \in \Delta_{N}$ . Furthermore, we sometimes abuse notation and write $(a, \mathbf{x}^{-i})$ , for an action $a \in A$ , to denote $(\mathbf{e}_{a}, \mathbf{x}^{-i})$ , where $e_{a}$ denotes the vector with a 1 at coordinate a and 0s at all other coordinates.

Constructing payoff matrices. To generate a game, we draw the payoff matrix, $\Pi$ , from a multivariate Gaussian with mean 0 which treats all players symmetrically ${}^{1}$ . The covariance matrix of the distribution is determined by parameter $\Gamma \in (-1, p - 1)$ , which captures the pairwise correlations between the players' payoffs. Fixing all but two players $i, j \in P$ , we have the following pairwise-correlation structure for each action profile $a, b \in A^{p}$ .

$$
\mathbb {E} \left[ \Pi (\mathbf {a}) _ {i} \cdot \Pi (\mathbf {b}) _ {j} \right] = \left\{ \begin{array}{l l} 1, & \text { if } \mathbf {a} = \mathbf {b}, i = j, \\ \Gamma / (p - 1), & \text { if } \mathbf {a} = \mathbf {b}, i \neq j \\ 0, & \text { if } \mathbf {a} \neq \mathbf {b} \end{array} \right.
$$

$\Gamma$ acts as a measure of the level of cooperativeness-competitiveness of a game. For every additional unit of reward that Player i receives by changing their strategy, the sum of all other players' payoffs will change by $\Gamma$ in expectation. $\Gamma = -1$ represents a $p$ -player zero-sum game, while $\Gamma = p - 1$ represents an identical payoff game. In general, $\Gamma < 0$ , corresponds to competitive games where players can only benefit at the expense of others. Conversely, $\Gamma > 0$ , corresponds to coordination games in which the player's payoffs are positively aligned to a degree given by $\Gamma$ .

Q-Learning. Given a game and an initial set of strategies, we wish to analyse how players learn and how their strategies evolve over time. Players following a learning algorithm turn games into dynamical systems with strategies evolving in a state space. Our focus is on the Q-Learning model (Watkins and Dayan 1992). $^{2}$ Here each player $i \in P$ keeps track of a Q-value corresponding to each action $a \in A$ , which estimates the quality of the given action. At each time step t, the Q-value corresponding to action a are updated as follows:

$$
Q _ {a} ^ {i} (t + 1) = \underbrace {(1 - \alpha) Q _ {a} ^ {i} (t)} _ {\text { discounted   previous   Q   -   value }} + \underbrace {R \left(a , \mathbf {x} ^ {- i} (t)\right) ^ {i}} _ {\text { current   reward }} \tag {2}
$$

where $\alpha\in(0,1)$ denotes the discount parameter. $^{3}$ The discount rate is then given by $(1-\alpha)$ which indicates experience (the previous Q-value) is prioritised against the current reward. With the Q-values, player select mixed strategies according to the softmax distribution parameterised by $\beta>0$

$$
x _ {a} ^ {i} (t) = \frac {\exp \left[ \beta Q _ {a} ^ {i} (t) \right]}{\sum_ {b \in \mathcal {A}} \exp \left[ \beta Q _ {b} ^ {i} (t) \right]} \tag {3}
$$

We refer to parameter $T := \alpha/\beta$ as the exploration rate. Taking $\alpha, \beta \to 0$ , but keeping T constant is equivalent to taking smaller step sizes in each update until we reach the continuous limit. See the Appendix for details on how this limit is obtained from the discrete equations (2) (3). Thus, we obtain the continuous Q-Learning equations (Sato and Crutchfield 2003; Tuyls, Hoen, and Vanschoenwinkel 2006):

$$
\frac {\dot {x} _ {a} ^ {i} (t)}{x _ {a} ^ {i} (t)} = R \left(a, \mathbf {x} ^ {- i} (t)\right) ^ {i} - T \ln x _ {a} ^ {i} (t) - \rho^ {i} (t) \tag {4}
$$

where $\rho^{i} := R(\mathbf{x}^{i}(t), \mathbf{x}^{-i}(t))^{i} - T\langle\mathbf{x}^{i}(t), \ln\mathbf{x}^{i}(t)\rangle$ is a normalisation parameter, which ensures strategies stay within the simplex. Here, $\langle\cdot,\cdot\rangle$ denotes the inner product. The fixed points of Q-Learning dynamics (both discrete and continuous variant) are Quantal Response Equilibria.

Definition 1 (Quantal Response Equilibrium (QRE)) A strategy profile $\bar{\mathbf{x}}\in\Delta$ is a Quantal Response Equilibrium (QRE) if, for all players $i\in\mathcal{P}$ and all actions $a\in\mathcal{A}$

$$
\bar {x} _ {a} ^ {i} = \frac {\exp (R (a , \bar {\mathbf {x}} ^ {- i}) ^ {i} / T)}{\sum_ {j \in N} \exp (R (j , \bar {\mathbf {x}} ^ {- i}) ^ {i} / T)}.
$$

where $T \in [0, \infty)$ denotes the exploration rate, which is assumed to be equal for all players.

QRE Interpretation QREs are a natural equilibrium solution concept, which takes into account the risk-reward management of the players, and are related to NE. At T = 0, only the actions which yield the highest payoff are played. Here the QRE corresponds to the NE. For T > 0, players mix actions, with players converging to the uniform distribution as $T \rightarrow \infty$ . Thus, T acts as a risk aversion parameter. Crucially, for any finite game, any initial strategy in the interior of the simplex, continuous Q-Learning (4) will converge to a unique interior fixed point given a sufficiently high T (Hussain, Belardinelli, and Piliouras 2023).

Rescaling of T As we vary the number of actions, N, available to each player, intuitively, we expect the ‘typical action’ $x_{a}^{i}$ to scale at a rate of 1/N. Hence, the expected payoff across different actions scales at a rate of $\sqrt{1/N^{(p-1)}}$ . We dedicate a segment in the Appendix to discuss this rescaling. To facilitate a fair comparison across games of different sizes, we have to take these effects into account and rescale T as follows: $T = \tilde{T}/\sqrt{N^{(p-1)}}$ , where $\tilde{T}$ represents the previous unscaled exploration rate. Thus, we will henceforth be working with the scaled exploration rate T.

Overview of Numerical Results For all values of $\Gamma$ , Q-Learning converges to a unique fixed point at sufficiently high exploration rates T. Below some critical exploration rate $T_{crit}$ , which increases with $\Gamma$ and p, we observe dynamics of varying nature, given in Table 1.

<table><tr><td>Condition</td><td>Dynamical Behaviour at  $T < T_{\text{crit}}$ </td></tr><tr><td> $\Gamma > 0$ </td><td>Convergence to multiple fixed points</td></tr><tr><td> $\Gamma \approx 0$ </td><td>Occasional limit cycles</td></tr><tr><td> $\Gamma < 0$ </td><td>Chaotic behaviour</td></tr></table>

Table 1: Dynamical behaviour at $T < T_{crit}$ for varying $\Gamma$ . A brief overview (and supporting figures from simulation) of the possible dynamical behaviours can be found in the Appendix. A similar overview for this model can be found in (Sanders, Farmer, and Galla 2018).

# Analytic Background

We provide an overview of the generating functional method, which was first introduced in (Galla and Farmer 2013) to study the dynamical behaviour of Q-Learning in the $N \rightarrow \infty$ limit. Instead of focusing on the outcome of a single initialisation of Q-Learning, we study the evolution of ensembles of possible initialisations. Thus, we will work with the distributions of possible Q-Learning trajectories and how they evolve over time. Borrowing methods from dynamical mean-field theory (DMFT) $^{4}$ , we consider the distribution of trajectories in the $N \rightarrow \infty$ limit; here the statistics of the Q-Learning trajectories satisfy a stochastic relation, which we refer to as the effective dynamics. As N increases, the statistics of the Q-learning trajectories obey the effective dynamics with increasing accuracy. Solving for the fixed points of the effective dynamics and its corresponding stability will allow us to identify the critical exploration rate, $T_{crit}$ , required for convergence to a unique fixed point and the rate of extinction in the unique fixed point regime.

Effective Dynamics. Deriving the effective dynamics relation can be broken up into the following two steps: (i) Defining a probability measure over possible trajectories under Q-Learning; (ii) Averaging over all possible payoff matrices, by considering the large action space limit $N \rightarrow \infty$ .

The calculations in each of these steps are lengthy and relies on path integral methods from disordered systems theory and a rescaling of variables, we have relegated the details of derivation from (Sanders, Farmer, and Galla 2018) into the Appendix alongside references. The result of this analysis, which holds for any given value of $\Gamma$ and T, is the following effective dynamics:

$$
\begin{array}{l} \frac {\dot {x} (t)}{x (t)} = \Gamma \int_ {t _ {0}} ^ {t} G (t, t ^ {\prime}) C (t, t ^ {\prime}) ^ {p - 2} x (t ^ {\prime}) \mathrm{d} t ^ {\prime} \\ - T \ln x (t) - \rho (t) + \eta (t). \tag {5} \\ \end{array}
$$

The term $\rho(t)$ is a function corresponding to the normalisation term $\rho^i(t)$ of (4), and $\eta(t)$ is a coloured (i.e., time-correlated) Gaussian random variable satisfying: $\langle \eta(t)\eta(t')\rangle_* = C(t,t')^{p-1}$ for all $t' < t$ , where $\langle\ldots\rangle_*$ denotes the expected value over realisations.

The $\eta$ term can be thought of as the randomness at the fixed point phase originating from the initialisation of the payoff matrices. Lastly, C and G are given by:

$$
C \left(t, t ^ {\prime}\right) = \langle x (t) x \left(t ^ {\prime}\right) \rangle_ {*}, G \left(t, t ^ {\prime}\right) = \left\langle \frac {\delta x (t)}{\delta \eta \left(t ^ {\prime}\right)} \right\rangle_ {*}.
$$

Here, C describes time correlations between strategies and G acts a ‘response’ function that links how strategies are correlated over time and how this varies with $\eta$ .

Equation (5) describes the evolution of the marginal probability of playing a given action $x(t)$ as a stochastic process. It is scaled by a factor of N such that each action is played with mean 1, $\langle x(t)\rangle_{*} = 1$ . We note that (5) does not depend on the player nor the index of the actions. This a due to the a-priori symmetry among players in the initial conditions. By solving for the fixed-point of (5), we are able to find the marginal probability distributions of playing each action at the unique fixed point regime of Q-Learning.

Fixed Points of the Effective Dynamics. For high exploration rates T, Q-Learning converges to a unique fixed point regardless of initial conditions. For large N, this should align with the presence of a stable fixed point for (5). Thus, to check the dynamical behaviour of Q-Learning, given $\Gamma$ and T, we would have to (i) identify the fixed points corresponding to (5) and (ii) analyse the respective stability.

Thus, to find a fixed point of the effective dynamics in a large game, we consider the following. First, a fixed point of (5) is defined as a solution where $\dot{x}(t) = 0$ (for all $t$ ), so that $x(t) = x$ is constant across $t$ for each realisation of

the random variable $\eta$ . In this stationary regime, we note: C becomes a constant; $C(t,t') = \left\langle(x)^{2}\right\rangle_{*} = q$ , and $\eta$ turns out to be a static realisation of a Gaussian $\eta \sim \mathcal{N}(0, q^{p-1})$ , while G is now a function of the time difference; $G(t,t') = G(t-t')$ . We let $z \sim \mathcal{N}(0,1)$ such that $\eta = q^{(p-1)/2}z$ . Thus, we have:

$$
0 = x (z) \left[ \Gamma q ^ {(p - 2)} x (z) \chi - T \ln x (z) - \rho + q ^ {(p - 1) / 2} z \right] \tag {6}
$$

where we note that x in (6) is a function of the Gaussian realisation $z, \chi = \int_{0}^{\infty} G(\tau) d\tau$ is assumed to be finite, and $\rho$ corresponds to $\rho(t)$ in (5), which must be constant in t as well by the requirement that $\dot{x}(t) = 0$ . By the definitions of $q, \chi, C, G$ and $\eta$ , the following self-consistency relations must hold:

$$
\left\langle \frac {\delta x (z)}{\delta z} \right\rangle_ {*} = q ^ {(p - 1) / 2} \chi , \left\langle x (z) ^ {2} \right\rangle_ {*} = q, \langle x (z) \rangle_ {*} = 1 \tag {7}
$$

where $\langle x(z)\rangle_{*}$ can be replaced by the integral $\int_{-\infty}^{\infty}x(z)\exp(-z^{2}/2)/\sqrt{2\pi}dz$ . This gives us a 4-equation, 4-unknown problem (with unknowns $q,\chi,\rho,x$ and Equations (7), and (6)), and solving this problem yields us a fixed point corresponding to (5).

# Beyond Competitive Games

Up to this point, we have summarised previous work by (Galla and Farmer 2013; Sanders, Farmer, and Galla 2018). While the effective dynamics (5) and fixed point equations (6) hold for any $\Gamma$ , only the competitive case ( $\Gamma < 0$ ) has been studied analytically. Now, we will explore the implications of allowing $\Gamma$ to be positive.

The structure is as follows: We show that the theory suggests asymptotic extinction occurs when $\Gamma \geq 0$ , and this is consistent with numerical simulations. We solve $x(z)$ for the fixed point equations (6) using an ansatz that accounts for this effect, determining the frequency of asymptotic extinction. We will then determine $T_{crit}$ , which we will compare against numerical simulations. Additional details, derivations, and supporting figures are provided in the Appendix.

# Asymptotic Extinction

To satisfy (6), we need either of the following to hold for all values of $z$ : $x(z) = 0$ or the bracketed term in (6) = 0, whilst satisfying the self-consistency relations (7). We can classify our fixed points into two distinct cases: i) Interior, where: $\left[\Gamma q^{(p-2)}x(z)\chi - T\ln x(z) - \rho + q^{(p-1)/2}z\right] = 0, \forall z \in \mathbb{R}$ ii) Boundary, where $x(z) = 0$ : for some $z$ .

Points on the boundary correspond to strategies with extinct actions. Our analysis shows that when $\Gamma > 0$ , (6) has no corresponding internal fixed points, only boundary fixed points, even at high exploration rates. How could (6) suggest that Q-Learning would converge to a boundary point?

It is known that strategies on the boundary of the simplex are unstable under Q-Learning in any finitely sized game. However, boundary points, as defined here, are not necessarily unstable since we are working in the large action size limit $N \rightarrow \infty$ . (Recall $x(z)$ is rescaled by a factor of N in the effective dynamics (5)). Simulating games with fixed parameter values ( $\Gamma \geq 0$ and varying N, we find that a proportion of actions are played with near 0 probability. When $T < T_{crit}$ , this proportion can be significant. This occurs to a much smaller extent in the unique fixed point regime $T > T_{crit}$ , typically affecting less than 1% of actions. (See Figure 1, where a probability mass of actions are played with near 0 probability as N increases.) This provides the basis for asymptotic extinction, where points can be internal for any finite game, but asymptotically approaches the boundary in the large action size $N \to \infty$ limit.

![](images/2c4e1ae62e1b2b27b5501cec9d8737d54bb786e31ad818deb1505cd5c80fae78.jpg)

<details>
<summary>line</summary>

| x     | Theoretical extinction rate | N=20   | N=40   | N=80   | N=160  |
|-------|-----------------------------|--------|--------|--------|--------|
| 0.00  | 0.0075                      | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.02  | 0.0075                      | 0.0035 | 0.0065 | 0.0100 | 0.0075 |
| 0.04  | 0.0075                      | 0.0055 | 0.0085 | 0.0105 | 0.0095 |
| 0.06  | 0.0075                      | 0.0075 | 0.0105 | 0.0115 | 0.0115 |
| 0.08  | 0.0075                      | 0.0105 | 0.0125 | 0.0125 | 0.0135 |
| 0.10  | 0.0075                      | 0.0125 | 0.0145 | 0.0135 | 0.0155 |
</details>

Figure 1: Empirical cumulative density plot representing the marginal likelihood of playing an action at unique fixed point for randomly generated games following the Q-Learning dynamic where $\Gamma = 0$ , $T = 1.8$ , $p = 2$ . The plot is zoomed in at the bottom $1\%$ of least played actions and $x$ is rescaled such that $x = 1$ would represent the average likelihood $(1/N)$ . As $N$ increases, a probability mass appears to form near 0, representing actions going asymptotically extinct. The red line represents the theoretical estimate of the extinction rate $(0.74\%)$ in the $N \to \infty$ limit. In this limit, the cumulative density plot would begin on the red line.

The fixed point solution of $x(z)$ allows us to characterise this behaviour in the unique fixed point regime $T > T_{crit}$ . Depending on the sign of $\Gamma$ , the fixed point solution given by $x(z)$ varies significantly (See Table 2). We will provide a case-by-case ansatz of $x(z)^{6}$ , which takes the effect of asymptotic extinction into account for $\Gamma \geq 0$ :

<table><tr><td>Condition</td><td>Stable Interior fixed point?</td></tr><tr><td> $\Gamma > 0$ </td><td>no</td></tr><tr><td> $\Gamma = 0$ </td><td>depends on  $T$ </td></tr><tr><td> $\Gamma < 0$ </td><td>yes</td></tr></table>

Table 2: Existence condition for a stable interior fixed point for finite exploration rates.

Competitive Games ( $\Gamma < 0$ ). In this regime, there exists a unique interior fixed point. At this fixed point we have:

$$
x (z) = K e ^ {b z + a x (z)} \tag {8}
$$

![](images/757e66b344bbe2daf4723c23bafd00d398b69b3ed6e02fddebef46930b5ce751.jpg)

<details>
<summary>line</summary>

| z    | Γ < 0 | Γ = 0 | Γ > 0 |
| ---- | ----- | ----- | ----- |
| Low  | ~0    | ~0    | ~0    |
| Mid  | ~1    | ~2    | ~3    |
| High | ~5    | ~10   | ~15   |
</details>

Figure 2: Sketch of $x(z)$ for different values of $\Gamma$ . The solution for $\Gamma > 0$ is double-valued below a critical z, as seen by the dotted lines. The bottom (solid) branch is of interest here.

where: $a = \Gamma q^{(p-2)} \chi T^{-1} < 0$ , $b = q^{(p-1)/2} T^{-1} > 0$ and K is a normalisation constant, ensuring $\langle x(z) \rangle_{*} = 1$ . These are determined by solving the self-consistency relation (7) and is in agreement with (Sanders, Farmer, and Galla 2018). Cooperative Games ( $\Gamma > 0$ ). For cooperative games $\Gamma > 0$ , Equation (6) does not yield an interior fixed point, but rather a boundary point. Fixed points here take the following form:

$$
x (z) = \left\{ \begin{array}{l l} K e ^ {b z + a x (z)} & , z <   z _ {\text { crit }} \\ 0 & , z \geq z _ {\text { crit }} \end{array} \right. \tag {9}
$$

where: $a = \Gamma q^{(p - 2)}\chi T^{-1} > 0$ and $z_{\mathrm{crit}} = -1 / b(1 + \ln (aK))$ . Values $a, b, K$ are the same as in (8), except $a$ is now positive. We note (9) is double-valued for $x(z)$ below $z_{\mathrm{crit}}$ . This is represented by the two branches in Figure 2: the bottom branch (the solid line) and the top branch (the dotted-line). We take the bottom branch as our value for $x(z)$ .

Uncorrelated Games ( $\Gamma = 0$ ). Uncorrelated games represent a special case, where the existence of an internal fixed point is dependent on T. When, $T \geq \sqrt{3e(p-1)/2}$ , we have an internal fixed point $^{7}$ of the form:

$$
x (z) = K e ^ {b z} \tag {10}
$$

While, when $T < \sqrt{3e(p-1)/2}$ , the fixed point is on the boundary, taking the form:

$$
x (z) = \left\{ \begin{array}{l l} K e ^ {b z} & , z <   z _ {\text { crit }} \\ 0 & , z \geq z _ {\text { crit }} \end{array} \right. \tag {11}
$$

With the disappearance of $a$ , $z_{\mathrm{crit}}$ is to be determined directly from self-consistency (7) as the third unknown.

How likely is extinction? For lower exploration rates $T < T_{crit}$ , the solutions of $x(z)$ are unstable and we are unable to characterise likelihood of extinction. Our solution $x(z)$ to the fixed point relations given by (9) and (10) can only predict the distribution of actions when $T > T_{crit}$ (i.e. when there is a unique fixed point), with $T_{crit}$ and the corresponding regime will be identified in the next section. For now, we will discuss the extinction likelihood obtained by solving $x(z)$ , given by $P(z < z_{\mathrm{crit}})$ , where $z \sim \mathcal{N}(0,1)$ .

Figure 3 displays the theoretical extinction rate for games with $p \in \{2, 3, 5\}$ and varying T and $\Gamma$ . As T is increased beyond $T_{crit}$ , our fixed point relations suggests the likelihood of a randomly chosen strategy going extinct asymptotically decreases drastically. In the large game limit, $N \to \infty$ , for any finite exploration rate T, theory suggests that a non-zero proportion of strategies is expected to go asymptotically extinct in coordination games.

Around the stability boundary, extinctions occur to around 1% to 0.01% of actions. Checking selected parameter combinations of T and $\Gamma$ on the stability boundary for games with more players, we find this roughly holds true for higher values of p. Away from the boundary, the probability of a randomly selected action going extinct becomes very rare. Verifying the likelihood of asymptotic extinction experimentally in this parameter range with experiments is difficult, as extinctions become extraordinarily rare events.

# Stability Analysis

Having found the fixed points distributions, we have to check their corresponding stability to determine if Q-Learning converges to it. We show the following result:

Proposition 1 Q-Learning converges to a unique fixed point when the parameters and corresponding fixed point fulfils the following relation:

$$
\phi \left\langle \left| \frac {T}{x (z)} - \Gamma q ^ {p - 2} \chi \right| ^ {- 2} \right\rangle_ {*} <   ((p - 1) q ^ {p - 2}) ^ {- 1} \tag {12}
$$

where $\phi$ is the proportion of non-extinct strategies, given by $P(z < z_{\text{crit}})$ where $z \sim \mathcal{N}(0,1)$ .

When $\Gamma < 0$ , $\phi = 1$ (as the fixed points are internal); we have the same relation as (Sanders, Farmer, and Galla 2018). What we have done here is added a $\phi$ -term, which takes into effect when $\Gamma \geq 0$ .

Thus, (12) extends the analysis to the coordination setting $\Gamma \geq 0$ . In the unique fixed point regime, only a small fraction (< 1%) of strategies go extinct, thus the relevant $\phi$ s are almost always close to 1. More detailed guidance on obtaining (12) is attached in the Appendix; it is obtained by a somewhat standard procedure ${}^{9}$ used to determine the linearised stability in dynamical systems, as follows: i) linearising the dynamics at the fixed point ii) taking a frequency transform and identifying a stability criterion which guarantees the stability of the whole system under all possible per-

![](images/45e2ba20d88f0297bd0a50188c8749f585f4bfae7b9c111d9de08fda0213d222.jpg)

Figure 3: Theoretical asymptotic extinction rate for varying numbers of players p obtained from estimates from the fixed point relations (9). These estimations are only for the unique fixed point regime (right of the yellow dotted line representing the stability boundary, which is solved in the next segment). There are some numerical instability in the estimations (namely when $\Gamma < 0.1$ , thus the axes not starting at 0), but the figure roughly shows the scale of the expected extinction rate for varying T away from the boundary.   
![](images/6c7125d76155c44fd7d3d8a539d9cd22ef970f28c0882114b22fe4bc695eab38.jpg)

<details>
<summary>line</summary>

| T  | p=2  | p=3  | p=4  | p=5  | p=10 | p=20 | p=50 | p=100 |
|----|------|------|------|------|------|------|------|-------|
| 0  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00  |
| 5  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00  |
| 10 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00  |
| 15 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00  |
| 20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00  |
| 25 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00  |
| 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00  |
| 35 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00  |
| 40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00  |
</details>

Figure 4: Stability boundary obtained by solving (12) for varying values of p, as a function of T for $\hat{\Gamma} > 0$ , where $\hat{\Gamma} = \Gamma/(p - 1)$ . To the right of the boundary, all Q-Learning trajectories converge to a unique fixed point in the large action size limit, $N \to \infty$ . When $\hat{\Gamma} < 0$ , we recover the results from (Sanders, Farmer, and Galla 2018). Our work extends the stability boundary to cover $\hat{\Gamma} > 0$ .

![](images/b10afb8d9cbdcad9a8082248acb67ea0c829e0277e54485e32fe55ce8a8c8dcd.jpg)

<details>
<summary>line</summary>

| T/√(e(p-1)) | p=2   | p=10  | p=20  | p=50  | p=100 | T = (f̂ + 1)√(e(p-1)) |
| ----------- | ----- | ----- | ----- | ----- | ----- | --------------------- |
| 1.0         | 0.00  | 0.00  | 0.00  | 0.00  | 0.00  | 0.00                  |
| 1.5         | 0.50  | 0.60  | 0.65  | 0.70  | 0.75  | 0.55                  |
| 2.0         | 1.00  | 1.00  | 1.00  | 1.00  | 1.00  | 1.00                  |
</details>

Figure 5: Rescaled stability curves for selected values of p. The exploration rate, T, is rescaled by a factor of $\sqrt{e(p-1)}$ and the grey-dashed line represents the straight line given by $T_{\mathrm{crit}} = (\hat{\Gamma} + 1)\sqrt{e(p-1)}$ , which appears to be the limiting behaviour at $p \to \infty$ . The increasing agreement with the grey line for large curves with larger values of p suggests this linear relationship is valid, in the large p limit.

turbation modes $^{10}$ .

Discussion. We are interested in the coordination setting $\Gamma > 0$ . To generate comparison for games of varying number of players, p, we rescale the correlation term as $\hat{\Gamma} = \Gamma/(p - 1)$ . We will be looking at multi-player games for $\hat{\Gamma} \in (0, 1)$ .

Figure 4 displays the stability curves for varying values of p obtained by solving (12). The curve represents the boundary, which separate the multiple fixed point regime from the unique fixed point regime for varying $\hat{\Gamma}$ and T. The key result is as p and $\hat{\Gamma}$ increases, so does the critical exploration rate. What does the boundary look like as p gets larger? Upon a rescaling the exploration rate by $1/\sqrt{e(p-1)}$ in the stability plots, Figure 5 suggests, as p increases, a direct linear relationship between the critical exploration rate $T_{crit}$ and how correlated the game is, $\hat{\Gamma}$ , emerges given by ${}^{11}$ :

$$
T _ {c r i t} = (\hat {\Gamma} + 1) \sqrt {e (p - 1)}. \tag {13}
$$

![](images/15230c04c3d4ee40b1daa4be36d2fdb0eafabc155df9fb34c5365ff414661a33.jpg)

<details>
<summary>heatmap</summary>

| T    | L    | Value |
|------|------|-------|
| 0.0  | 0.0  | Low   |
| 0.0  | 0.5  | Low   |
| 0.0  | 1.0  | Low   |
| 2.0  | 0.0  | Low   |
| 2.0  | 0.5  | Medium-High |
| 2.0  | 1.0  | High    |
| 4.0  | 0.0  | Low   |
| 4.0  | 0.5  | Medium-High |
| 4.0  | 1.0  | High    |
| 6.0  | 0.0  | Low   |
| 6.0  | 0.5  | Medium-High |
| 6.0  | 1.0  | High    |
</details>

![](images/86f87a6b5c1f3c8c943d08fa966f1535674cb777b7948919e8d1f46ea5783c68.jpg)

<details>
<summary>heatmap</summary>

| T    | L1   | Value |
|------|------|-------|
| 0    | 0    | Low   |
| 0    | 1    | Low   |
| 0    | 2    | Low   |
| 2    | 0    | Low   |
| 2    | 1    | Medium|
| 2    | 2    | Medium|
| 4    | 0    | High  |
| 4    | 1    | High  |
| 4    | 2    | High  |
| 6    | 0    | High  |
| 6    | 1    | High  |
| 6    | 2    | High  |
</details>

![](images/930018c3b16f9950fd3005f1c9b75372c3d1c9e60ebbfda7a4e3e4a102753daf.jpg)

<details>
<summary>heatmap</summary>

| T \ L-2 | 0    | 2    | 4    | 6    |
|---------|------|------|------|------|
| 0       | 0.0  | 0.0  | 0.0  | 0.0  |
| 2       | 0.0  | 0.0  | 0.0  | 0.0  |
| 4       | 0.0  | 0.0  | 0.0  | 0.0  |
| 6       | 0.0  | 0.0  | 0.0  | 0.0  |
</details>

Figure 6: Heat maps showing the proportion of 40 independent payoff matrices for which all trajectories of Q-Learning converges to a unique fixed point, for varying parameter values. Dark red corresponds to all initial conditions converging to a unique fixed point, while blue indicates there are multiple equilibria. The yellow dashed line is computed from the generating functional method. It represents the stability boundary, which separates the two regimes in the large action size limit $N \rightarrow \infty$ .

In (Sanders, Farmer, and Galla 2018), it is shown that in the large-p limit of uncorrelated games ( $\hat{\Gamma} = 0$ ) and p-player zero sum games ( $\hat{\Gamma} = 1/(p-1)$ ) the critical exploration rate is given by $T_{\mathrm{crit}} = \sqrt{e(p-1)}$ . This, in combination with (13), suggests the following statement:

Observation 1 The critical exploration rate which guarantees the convergence of Q-Learning to a unique fixed point in pure coordination (identical-payoffs) games is twice that of p-player, zero-sum games in the large p-limit.

Comparison between theory and numerical results. We compare how our theoretical results fare against numerical experiments of finite-sized, coordination games. We used the default SciPy Runga-Kutta 4(5) solver (Virtanen et al. 2020) with max stepsize set to 0.5 to be approximate continuous Q-Learning (4) as closely as possible. A point is classified as fixed when the derivative of (4) drops below $|10^{-8}|$ .

To determine if a given game converges to a unique fixed point, 100 random initial strategies are drawn and simulated for up to 5000 time units, or until it reaches a fixed point. If all 100 final points, are within a relative distance of 0.01 of each other, we assume there is a unique fixed point.

For a p-player, N-action game, we have $p \times N^{p}$ payoff elements. Selecting p = 2, 3, 5 and respectively N = 50, 12, 4, yields games with approximately 5000 payoff elements each. For each of the three $(p, N)$ -pairs, we perform a parameter search or ‘mesh-grid’ evaluation for $\Gamma \geq 0$ and $T \in (0, 6)$ . For each $\Gamma$ and T, 40 independent games are generated, and we record the proportion of games for which Q-Learning converges to a unique fixed point. Figure 6 displays a heat map, displaying the likelihood Q-Learning convergences to a unique fixed point, given the chosen parameters. This is plotted in contrast to the theoretical stability boundary in yellow. We refer to (Sanders, Farmer, and Galla 2018) for a similar comparison between the theoretical and numerical results, for $\Gamma < 0$ .

We can identify a correspondence between the theoretical curve and the simulation results, which validate the generating functional approach. The simulation plots for p = 3 N = 12, p = 5 N = 4 has greater variation between sample rounds than p = 2 N = 50. We suspect increasing the number of actions N should reduce the variation in the dynamics between large games drawn from the same parameters. Similar to previous work on competitive games (Sanders, Farmer, and Galla 2018), the theoretical critical exploration rate, $T_{crit}$ , appears to be an overestimate, especially near $\Gamma = 0$ . We assume (as in previous work) this is a finite-size effect, which disappears as N increases, as the theoretical prediction is in the limit $N \to \infty$ .

# Conclusion

Throughout this paper, we have studied the dynamical behaviour of Q-Learning over large, multi-player coordination games, generated from a multivariate Gaussian. This work builds on the model and analysis introduced in (Sanders, Farmer, and Galla 2018) used to study competitive games, to cover the coordination setting.

Q-Learning in large coordination games exhibits a phenomenon that we call asymptotic extinction, where a nonzero fraction of strategies are played with zero probability in the large action size limit $N \rightarrow \infty$ . Asymptotic extinction is most noticeable at lower exploration rates T, but also occurs at high values of T. Taking this effect into account, a critical exploration rate $T_{crit}$ can be identified above which a unique equilibrium exists, and where all trajectories of Q-Learning from all initial points end up in the same equilibrium.

The problem of choosing the ‘optimum’ exploration rate remains confounding question. Picking the rate $T_{crit}$ , ensures convergence to a unique distribution, avoiding ending up in worst-case scenarios of converging to bad equilibria. $T_{crit}$ could be taken as a reasonable choice of exploration rate because it is the smallest one where such a unique fixed point is guaranteed, but we emphasize that there are further intriguing questions around the topic of determining the ideal T, and there are potentially reasonable alternative choices for T. One can consider the problem of finding the exploration rate that maximises any arbitrary objective function (such as maximising total utility, or maximising the minimum utility among the players). This gives rise to a number of interesting questions to consider for future research.

# Acknowledgments

The authors would like to thank Aamal Hussain and Edward Plumb for the useful discussions throughout the project. This work was partially supported by the UKRI Trustworthy Autonomous Systems Hub (EP/V00784X/1). Partial financial support has been received from the Agencia Estatal de Investigación and Fondo Europeo de Desarrollo Regional (FEDER, UE) under project APASOS (PID2021-122256NB-C21/PID2021-122256NB-C22), and the Maria de Maeztu project CEX2021-001164-M, funded by MCIN/AEI/10.13039/501100011033. Bart de Keijzer was partially supported by EPSRC grant EP/X021696/1.

# References

Bloembergen, D.; Tuyls, K.; Hennes, D.; and Kaisers, M. 2015. Evolutionary dynamics of multi-agent learning: A survey. Journal of Artificial Intelligence Research, 53: 659–697.   
Camerer, C.; and Hua Ho, T. 1999. Experience-weighted attraction learning in normal form games. Econometrica, 67(4): 827–874.   
Drazin, P.; and Reid, W. 2004. Hydrodynamic Stability. Cambridge Mathematical Library. Cambridge University Press. ISBN 9780521525411.   
Galla, T. 2006. Random replicators with asymmetric couplings. Journal of Physics A: Mathematical and General, 39(15): 3853–3869.   
Galla, T. 2018. Dynamically evolved community size and stability of random Lotka-Volterra ecosystems (a). Euro-physics Letters, 123(4): 48004.   
Galla, T. 2024. Generating-functional analysis of random Lotka-Volterra systems: A step-by-step guide. arXiv:2405.14289.   
Galla, T.; and Farmer, J. D. 2013. Complex dynamics in learning complicated games. Proceedings of the National Academy of Sciences, 110(4): 1232–1236.   
Hussain, A. A.; Belardinelli, F.; and Piliouras, G. 2023. Asymptotic convergence and performance of multi-agent q-learning dynamics. arXiv preprint arXiv:2301.09619.   
Kianercy, A.; and Galstyan, A. 2012. Dynamics of Boltzmann Q learning in two-player two-action games. Physical Review E—Statistical, Nonlinear, and Soft Matter Physics, 85(4): 041145.   
Leonardos, S.; and Piliouras, G. 2022. Exploration-exploitation in multi-agent learning: Catastrophe theory meets game theory. Artificial Intelligence, 304: 103653.   
Leonardos, S.; Piliouras, G.; and Spendlove, K. 2021. Exploration-exploitation in multi-agent competition: convergence with bounded rationality. Advances in Neural Information Processing Systems, 34: 26318–26331.   
March, J. G. 1991. Exploration and exploitation in organizational learning. Organization science, 2(1): 71–87.   
Opper, M.; and Diederich, S. 1992. Phase transition and 1/f noise in a game dynamical model. Physical review letters, 69(10): 1616.

Panait, L.; and Luke, S. 2005. Cooperative multi-agent learning: The state of the art. Autonomous agents and multi-agent systems, 11: 387–434.

Pangallo, M.; Sanders, J.; Galla, T.; and Farmer, D. 2017. Towards a taxonomy of learning dynamics in 2 x 2 games. arXiv preprint arXiv:1701.09043.

Peng, R. D. 2024. Advanced Statistical Computing. Manuscript in preparation.

Sanders, J. B.; Farmer, J. D.; and Galla, T. 2018. The prevalence of chaotic dynamics in games with many players. Scientific reports, 8(1): 4902.

Sato, Y.; and Crutchfield, J. P. 2003. Coupled replicator equations for the dynamics of learning in multiagent systems. Physical Review E, 67(1): 015206.

Tao, T.; and Vu, V. 2011. Random matrices: universality of local eigenvalue statistics. Acta Math, 206: 127–204.

Tuyls, K.; Hoen, P. J. T.; and Vanschoenwinkel, B. 2006. An Evolutionary Dynamical Analysis of Multi-Agent Learning in Iterated Games. Autonomous Agents and Multi-Agent Systems, 12(1): 115–153.

Virtanen, P.; Gommers, R.; Oliphant, T. E.; Haberland, M.; Reddy, T.; Cournapeau, D.; Burovski, E.; Peterson, P.; Weckesser, W.; Bright, J.; van der Walt, S. J.; Brett, M.; Wilson, J.; Millman, K. J.; Mayorov, N.; Nelson, A. R. J.; Jones, E.; Kern, R.; Larson, E.; Carey, C. J.; Polat, I.; Feng, Y.; Moore, E. W.; VanderPlas, J.; Laxalde, D.; Perktold, J.; Cimrman, R.; Henriksen, I.; Quintero, E. A.; Harris, C. R.; Archibald, A. M.; Ribeiro, A. H.; Pedregosa, F.; van Mulbregt, P.; and SciPy 1.0 Contributors. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17: 261–272.

Watkins, C. J.; and Dayan, P. 1992. Q-learning. Machine learning, 8: 279–292.

# Appendix for Asymptotic Extinction in Large Coordination Games

This Appendix provides additional details related to the work presented in the main paper, alongside supporting figures. The content of the Appendix is ordered as follows:

1. Motivation for the choice of Gaussian distribution.   
2. Derivation of the continuous Q-Learning equation.   
3. Rescaling of Variables   
4. Derivation of the effective dynamics.   
5. Solving fixed points relations for correlated games. $\Gamma \neq 0$   
6. Solving fixed point relation of uncorrelated games $\Gamma = 0$   
7. Derivation of the stability condition.   
8. Further plots of selected simulations.

# Motivation for the choice of Gaussian distribution

The choice of a Gaussian distribution to describe the distribution of payoff values can be motivated by a maximum-entropy argument. We wish to characterise the outcome of learning in terms of a small number of summary statistics (in our case, the first and second moments of the distribution of payoff matrix elements). Following principles of information theory, one then maximises uncertainty (Shannon entropy) subject to these moments leading to a multivariate Gaussian.

We expect our theory to apply for many non-Gaussian distributions. The ‘universality principle’ in random matrix theory (Tao and Vu 2011) states that the spectra of large non-Gaussian random matrices are identical to those of a Gaussian ensemble with the same first and second moments if higher-order moments fall off sufficiently quickly with the matrix size. The key steps of averaging over the randomness in our calculation is very similar to those in random matrix theory. Hence, we would expect similar universality properties in our model.

# Derivation of the continuous Q-Learning equation

The Q-Learning equations typically given by the following two steps (Watkins and Dayan 1992): i) updating of the Q-values

$$
Q _ {a} ^ {i} (t + 1) = \underbrace {(1 - \alpha) Q _ {a} ^ {i} (t)} _ {\text { discounted   previous   Q   -   value }} + \underbrace {R (a , \mathbf {x} ^ {- i} (t)) ^ {i}} _ {\text { current   reward }} \tag {A.1}
$$

ii) applying a softmax to obtain strategies from the Q-values:

$$
x _ {a} ^ {i} (t) = \frac {\exp \left[ \beta Q _ {a} ^ {i} (t) \right]}{\sum_ {b \in \mathcal {A}} \exp \left[ \beta Q _ {b} ^ {i} (t) \right]} \tag {A.2}
$$

where the discount rate is then given by $(1 - \alpha)$ and the softmax distribution parameterised by $\beta > 0$ Combining (A.1) and (A.2), we obtain the following recursive relation for x:

$$
x _ {a} ^ {i} (t + 1) = \frac {1}{\hat {\rho} ^ {i} (t)} x _ {a} ^ {i} (t) ^ {(1 - \alpha)} \left[ \exp (\beta R (a, \mathbf {x} ^ {- i} (t)) ^ {i}) \right] \tag {A.3}
$$

where $\hat{\rho}^{i}(t):=\sum_{k\in\mathcal{A}}x_{k}^{i}(t)^{1-\alpha}\left[\exp(\beta R(a,\mathbf{x}^{-i}(t))^{i})\right]$ is a normalisation parameter, which ensures strategies stay within the simplex. Typically the step sizes between each update in reinforcement learning is small, this motivates finding the continuous Q-Learning equations, an ODE which represents the small stepsize limit. Dividing both sides sides by $x_{a}^{i}(t)$ and taking the logarithm we obtain the following relation:

$$
\ln \frac {x _ {a} ^ {i} (t + 1)}{x _ {a} ^ {i} (t)} = \beta R (a, \mathbf {x} ^ {- i} (t)) ^ {i}) - \alpha \ln x _ {a} ^ {i} (t) - \ln \hat {\rho} ^ {i} (t) \tag {A.4}
$$

With an abuse of notation, in the small step size limit, the update between each discrete time step is minimal (i.e. we have $t + 1 \approx t$ ). This allows us to make the following approximation:

$$
\frac {\ln x _ {a} ^ {i} (t + 1) - \ln x _ {a} ^ {i} (t)}{(t + 1) - t} \approx \frac {d}{d t} \ln x _ {a} ^ {i} (t) = \frac {\dot {x} _ {a} ^ {i} (t)}{x _ {a} ^ {i} (t)} \tag {A.5}
$$

As discussed in the main paper, we refer to parameter $T := \alpha/\beta$ as the exploration rate. Taking $\alpha, \beta \to 0$ , but keeping T constant is equivalent to taking smaller step size in each update until we reach the continuous limit. Substituting the approximation (A.5) into (A.4) and sending $\alpha, \beta \to 0$ , we obtain the continuous Q-Learning equations (Sato and Crutchfield 2003; Tuyls, Hoen, and Vanschoenwinkel 2006):

$$
\frac {\dot {x} _ {a} ^ {i} (t)}{x _ {a} ^ {i} (t)} = R (a, \mathbf {x} ^ {- i} (t)) ^ {i} - T \ln x _ {a} ^ {i} (t) - \rho^ {i} (t), \tag {A.6}
$$

where $\rho^{i} := R(\mathbf{x}^{i}(t), \mathbf{x}^{-i}(t))^{i} - T\langle\mathbf{x}^{i}(t), \ln\mathbf{x}^{i}(t)\rangle$ is the new normalisation parameter and $\langle\cdot,\cdot\rangle$ denotes the inner product.

# Rescaling of Variables

The equation (A.6) holds for any value of N. In this segment, we provide an overview of the rescaling of variables introduced in (Sanders, Farmer, and Galla 2018). To understand why a rescaling has to be done, we will first write down how the various terms in our equation scale w.r.t with N:

# Rescaling of $T$

- We expect the typical action to be played probability $1 / N$ , $x_{a}^{i}(t) \sim O(1 / N)$ .   
- The typical entry in the payoff matrix is not dependent on $N$ , $\Pi(\mathbf{a})_i \sim O(1)$ .

Let us consider player $i$ 's perspective. They wish to update their strategy using the Q-Learning algorithm. This requires calculating the expected reward of each pure action. Holding player $i$ 's strategy fixed as a pure action, the typical reward contribution from an action profile $\mathbf{a}$ (or each entry in the payoff matrix) given by $\Pi(\mathbf{a})_i \prod_{j \in \mathcal{P}, i \neq j} x_{\mathbf{a}_j}^j$ scales at a rate of $O(1/N^{p-1})$ .

From player $i$ 's point of view, upon fixing their action, the remaining players could play one among $N^{p-1}$ action profiles. Since the reward associated with each action profile is independent, the Q-value, $Q_{a}^{i}(t)$ , and the expected reward for a given action given by: $R(a, \mathbf{x}^{-i}(t))_{i} = \sum_{\mathbf{a} \in \mathcal{A}^{p}, i \neq j} \Pi(\mathbf{a})_{i} \prod_{i \in \mathcal{P}, i \neq j} x_{a_i}^i$ has a standard deviation of $O\left(\sqrt{1/N^{(p-1)}}\right)$ . This means that as $N$ and $p$ increases, the difference in rewards across different actions become less distinguishable, going to 0 in the limit. For exploitation to remain meaningful, the players have to adjust their exploration rate accordingly. Since $Q_{a}^{i}(t)$ has a standard deviation of $O\left(\sqrt{1/N^{(p-1)}}\right)$ , $\beta$ in the soft-max (3) should scale with a factor $\sqrt{N^{(p-1)}}$ . This yields us the change of variables for the exploration rate in the main paper given by:

$$
T = \tilde {T} / \sqrt {N ^ {(p - 1)}}
$$

Rescaling of x We note that this equivalent to keeping T fixed and rescaling x by a factor of N and the matrix elements by $\sqrt{1/N^{(p-1)}}$ . We introduce $\tilde{x}_{a}^{i}=Nx_{a}^{i}\sim O(1)$ and $\tilde{\Pi}(\mathbf{a})_{i}=\Pi(\mathbf{a})_{i}/\sqrt{N^{(p-1)}}\sim O(\sqrt{1/N^{(p-1)}})$ such that the payoff correlations take the following form:

$$
\mathbb {E} \left[ \tilde {\Pi} (\mathbf {a}) _ {i} \cdot \tilde {\Pi} (\mathbf {b}) _ {j} \right] = \left\{ \begin{array}{l l} 1 / N ^ {(p - 1)}, & \text { if } \mathbf {a} = \mathbf {b}, i = j, \\ (p - 1) ^ {- 1} \Gamma / N ^ {(p - 1)}, & \text { if } \mathbf {a} = \mathbf {b}, i \neq j \\ 0, & \text { if } \mathbf {a} \neq \mathbf {b} \end{array} \right.
$$

Now the ‘typical contribution’ for each action profile scales at a rate of $\Pi(\mathbf{a})_{i}\prod_{j\in\mathcal{P},i\neq j}x_{\mathbf{a}_{j}}^{j}\sim O\left(\sqrt{1/N^{(p-1)}}\right)$ and thus, the typical reward and Q values scales with $O(1)$ , meaning we do not have to change T w.r.t N under this particular rescaling. In the derivation of the effective dynamics, we will drop the tildes and use this second set of rescaling on x and $\Pi(\mathbf{a})$ .

# Derivation of the effective dynamics

In this section, we will present the derivation of the effective dynamics presented in the Supplementary of (Sanders, Farmer, and Galla 2018). For a step-by-step approach on this method on replicator models, refer to (Galla 2024). We will break up this segment into the following parts:

- Introduce the generating function induced by Q-Learning.   
- Identify the effective dynamics of ensembles of games under Q-Learning.

Generating Functions We begin with the Characteristic Function of a univariate random variable $X$ , which is given by:

$$
Z _ {X} (t) = \mathbb {E} (e ^ {i t X}) = \int f _ {X} (X) e ^ {i t X} d X \tag {A.7}
$$

where $f_{x}(x)$ is the pdf of $X$ and $t \in \mathbb{R}$ is the parameter of the generating function. It takes Fourier Transform of the probability density function. It has a some very neat properties namely, the n-th order moment can be found by taking the derivatives

$$
\mathbb {E} (X ^ {n}) = i ^ {(- n)} \frac {d ^ {n}}{d t ^ {n}} Z _ {X} (t) \big | _ {t = 0}
$$

For a multivariate time series, $\mathbf{x} = (x_{t}^{i}, t \in [t_{0}, T], \in [0, 1 \ldots n])$ . The characteristic function takes the following form:

$$
Z _ {\mathbf {x}} (\psi) = \mathbb {E} \left[ \exp \left(i \sum_ {i} \int \psi^ {i} (t) x _ {t} ^ {i} d t\right) \right] = \int D \mathbf {x} P (\mathbf {x}) \exp \left(i \sum_ {i} \int \psi^ {i} (t) x _ {t} ^ {i} d t\right) \tag {A.8}
$$

where $P(\mathbf{x})$ is the probability measure over all possible trajectories and $\psi^{i}(\cdot):[t_{0},T)\to\mathbb{R}.\psi^{i}(t)$ replaces t in (A.7) to become the parameter associated with $x^{i}(t)$ . The following moment property describes time correlations across different variables:

$$
\mathbb {E} \left[ x ^ {i} (t) x ^ {j} (t ^ {\prime}) \right] = - \frac {\partial^ {2}}{\partial \psi^ {i} (t) \partial \psi^ {j} (t ^ {\prime})} Z _ {\mathbf {x}} (\psi) \big | _ {\psi^ {i} (t) = \psi^ {j} (t ^ {\prime}) = 0} \tag {A.9}
$$

Moving forward, we want to keep the moment property (A.9) in mind, as we derive a generating function associated with Q-Learning. Consider the following modified continuous Q-Learning equations:

$$
\frac {\dot {x} _ {a} ^ {i} (t)}{x _ {a} ^ {i} (t)} = R (a, \mathbf {x} ^ {- i} (t)) ^ {i} - T \ln x _ {a} ^ {i} (t) - \rho^ {i} (t) + h _ {a} ^ {i} (t) \tag {A.10}
$$

This identical the continuous Q-Learning equations (A.6), except we have included an arbitrary function $h_{a}^{i}(t)$ , this enables us to generate a response function (and which we will be set zero). The generating function for Q-Learning is then given to take the following form:

$$
Z _ {\mathbf {x}} (\psi) = \int D \mathbf {x} \delta (\text { eq.   of   motion }) \exp \left(i \sum_ {a, i} \int d t \psi_ {a} ^ {i} (t) x _ {a} ^ {i} (t)\right) \tag {A.11}
$$

This takes a similar form to (A.8). Here $P(\mathbf{x}) = \delta(\text{eq. of motion})$ describes the probability measure induced by the possible Q-Learning trajectories. Expanding out the $\delta$ -functions with a Fourier Transform we have the generating function induced by Q-Learning:

$$
Z _ {\mathbf {x}} (\psi) = \int D [ \mathbf {x}, \hat {\mathbf {x}} ] \exp \left(i \sum_ {a, i} \int d t \left[ \hat {x} _ {a} ^ {i} (t) \left(\frac {\dot {x} _ {a} ^ {i} (t)}{x _ {a} ^ {i} (t)} - \left(R (a, \mathbf {x} ^ {- i} (t)) ^ {i} - T \ln x _ {a} ^ {i} (t) - \rho^ {i} (t) + h _ {a} ^ {i} (t)\right)\right) \right] + \psi_ {a} ^ {i} (t) x _ {a} ^ {i} (t)\right) \tag {A.12}
$$

Similar to (A.9), (A.12) admits the following time-correlation and response function relations:

$$
\mathbb {E} \left[ x _ {a} ^ {i} (t) x _ {b} ^ {j} (t ^ {\prime}) \right] = - \frac {\partial^ {2}}{\partial \psi_ {a} ^ {i} (t) \partial \psi_ {b} ^ {j} (t ^ {\prime})} Z _ {\mathbf {x}} (\psi) \big | _ {\psi = \mathbf {h} = 0}
$$

$$
\frac {\partial}{\partial h _ {b} ^ {j} (t ^ {\prime})} \mathbb {E} \left[ x _ {a} ^ {i} (t) \right] = - \frac {\partial^ {2}}{\partial \psi_ {a} ^ {i} (t) \partial h _ {b} ^ {j} (t ^ {\prime})} Z _ {\mathbf {x}} (\psi) \big | _ {\psi = \mathbf {h} = 0} \tag {A.13}
$$

Identifying the effective dynamics The randomness associated with (A.12) comes from the entries of the payoff matrix. Separating this from the other elements, we obtain:

$$
\begin{array}{l} Z _ {\mathbf {x}} (\psi) = \int D [ \mathbf {x}, \hat {\mathbf {x}} ] \exp \left(i \sum_ {a, i} \int d t \left[ \hat {x} _ {a} ^ {i} (t) \left(\frac {\dot {x} _ {a} ^ {i} (t)}{x _ {a} ^ {i} (t)} - \left(- T \ln x _ {a} ^ {i} (t) - \rho^ {i} (t) + h _ {a} ^ {i} (t)\right)\right) \right] + \psi_ {a} ^ {i} (t) x _ {a} ^ {i} (t)\right) \\ \times \underbrace {\exp \left(i \sum_ {a , i} \int d t \hat {x} _ {a} ^ {i} (t) R (a , \mathbf {x} ^ {- i} (t)) ^ {i}\right)} _ {\text { randomness   comes   from   here }} \tag {A.14} \\ \end{array}
$$

We will now conduct an averaging over all possible payoff matrices. Let us evaluate the expectation from the final row, which we denote as $Z_{\Pi}$ . To do this, we have to integrate w.r.t to the probability measure of the multivariate Gaussian corresponding to the payoff entries (which we denote as $P(\mathbf{\Pi}(\mathbf{a}))$ ).

$$
\overline {{{Z _ {\Pi}}}} = \mathbb {E} \left[ \exp \left(i \sum_ {a, i} \int d t \hat {x} _ {a} ^ {i} (t) R (a, \mathbf {x} ^ {- i} (t)) ^ {i}\right) \right] = \int \exp \left(i \sum_ {a, i} \int d t \hat {x} _ {a} ^ {i} (t) R (a, \mathbf {x} ^ {- i} (t)) ^ {i}\right) P (\boldsymbol {\Pi} (\mathbf {a})) D [ \boldsymbol {\Pi} (\mathbf {a}) ] \tag {A.15}
$$

Noting the characteristic function of a multivariate Gaussian is given by:

$$
\mathbb {E} \left[ \exp (i t X) \right] = \exp \left(i \mu^ {\mathbf {T}} \mathbf {t} - \frac {1}{2} \mathbf {t} ^ {\mathbf {T}} \boldsymbol {\Sigma} \mathbf {t}\right), X \sim \mathcal {N} (\mu , \Sigma^ {2}) \tag {A.16}
$$

From the rescaling in the previous segment, the rescaled payoff matrix correlations are as follows:

$$
\mathbb {E} \left[ \Pi (\mathbf {a}) _ {i} \cdot \Pi (\mathbf {b}) _ {j} \right] = \left\{ \begin{array}{l l} 1 / N ^ {(p - 1)}, & \text { if } \mathbf {a} = \mathbf {b}, i = j, \\ (p - 1) ^ {- 1} \Gamma / N ^ {(p - 1)}, & \text { if } \mathbf {a} = \mathbf {b}, i \neq j \\ 0, & \text { if } \mathbf {a} \neq \mathbf {b} \end{array} \right.
$$

$\overline{Z_{\Pi}}$ can be rewritten as:

$$
\overline {{Z _ {\Pi}}} = \exp \left(- \frac {N}{2} \int d t d t ^ {\prime} \sum_ {i} \left(L ^ {i} (t, t ^ {\prime}) \prod_ {i \neq j} C ^ {j} (t, t ^ {\prime}) + \Gamma \sum_ {i \neq j} K ^ {i} (t, t ^ {\prime}) K ^ {j} (t, t ^ {\prime}) \prod_ {k \notin (i, j)} C ^ {k} (t, t ^ {\prime})\right)\right) \tag {A.17}
$$

where the following short-hand substitutions are introduced:

$$
C ^ {i} (t, t ^ {\prime}) = \frac {1}{N} \sum_ {a} x _ {a} ^ {i} (t) x _ {a} ^ {i} (t ^ {\prime})
$$

$$
K ^ {i} (t, t ^ {\prime}) = \frac {1}{N} \sum_ {a} x _ {a} ^ {i} (t) \widehat {x} _ {a} ^ {i} (t ^ {\prime})
$$

$$
L ^ {i} (t, t ^ {\prime}) = \frac {1}{N} \sum_ {a} \widehat {x} _ {a} ^ {i} (t) \widehat {x} _ {a} ^ {i} (t ^ {\prime})
$$

We will be substituting (A.17) with the bottom row of (A.14), to complete averaging step:

$$
\overline {{Z _ {\mathbf {x}} (\psi)}} = \int D [ \mathbf {x}, \hat {\mathbf {x}} ] \exp \left(i \sum_ {a, i} \int d t \left[ \hat {x} _ {a} ^ {i} (t) \left(\frac {\dot {x} _ {a} ^ {i} (t)}{x _ {a} ^ {i} (t)} - \left(- T \ln x _ {a} ^ {i} (t) - \rho^ {i} (t) + h _ {a} ^ {i} (t)\right)\right) \right] + \psi_ {a} ^ {i} (t) x _ {a} ^ {i} (t)\right) \times \overline {{Z _ {\Pi}}} \tag {A.18}
$$

We re-express $Z_{\mathbf{x}}(\psi)$ as an integration over the short-hand substitutions terms and their respective conjugates. We seek the following form for $Z_{\mathbf{x}}(\psi)$ ; turning equation (A.12) to the form of (A.19).

$$
Z _ {\mathbf {x}} (\psi) = \int D [ \mathbf {x}, \hat {\mathbf {x}} ] \times \dots \tag {A.12}
$$

$$
\rightarrow \int D [ \mathbf {x}, \hat {\mathbf {x}} ] \times D [ C, L, K, \widehat {C}, \widehat {L}, \widehat {K} ] \times \dots \tag {A.19}
$$

We note:

$$
1 = \int D [ C ^ {i} ] P (C ^ {i}) \delta (0) \tag {A.20}
$$

$$
= \int D [ C ^ {i} ] \prod_ {t, t ^ {\prime}} \delta \left(\underbrace {C ^ {i} (t , t ^ {\prime}) - \frac {1}{N} \sum_ {a} x _ {a} ^ {i} (t) x _ {a} ^ {i} (t ^ {\prime})} _ {= 0}\right) \tag {A.21}
$$

$$
= \int D [ \widehat {C} ^ {i}, C ^ {i} ] \exp \left(i N \int d t d t ^ {\prime} \left(\widehat {C} ^ {i} (t, t ^ {\prime}) C ^ {i} (t, t ^ {\prime}) - \frac {\widehat {C} ^ {i} (t , t ^ {\prime})}{N} \sum_ {a} x _ {a} ^ {i} (t) x _ {a} ^ {i} (t ^ {\prime})\right)\right) \tag {A.22}
$$

where:

- (A.20) states that the integral of the pdf of $C_i = 1$ .   
- (A.21) rewrites (A.20)   
– (A.22) introduces the conjugate integration variable.

Repeating the expansions (A.20) - (A.22) for the other short-hand terms and multiplying onto (A.12) we can express the generating functional in the conjugate form as described in (A.19):

$$
\overline {{{Z _ {\mathbf {x}} (\psi)}}} = \int D [ C, L, K, \widehat {C}, \widehat {L}, \widehat {K} ] \exp (F) \tag {A.23}
$$

where:

$$
F = N (\Psi + \Phi + \Omega)
$$

denotes the terms in the exponent given by:

$$
\Psi = i \sum_ {i} \int d t d t ^ {\prime} \left(\widehat {C} ^ {i} (t, t ^ {\prime}) C ^ {i} (t, t ^ {\prime}) + \widehat {K} ^ {i} (t, t ^ {\prime}) K ^ {i} (t, t ^ {\prime}) + \widehat {L} ^ {i} (t, t ^ {\prime}) L ^ {i} (t, t ^ {\prime})\right) \tag {A.24}
$$

comes from the introduction of the conjugate variables in (A.22).

$$
\Phi = - \frac {1}{2} \sum_ {i} \int d t d t ^ {\prime} \left(L ^ {i} (t, t ^ {\prime}) \prod_ {i \neq j} C ^ {j} (t, t ^ {\prime}) + \Gamma \sum_ {i \neq j} K ^ {i} (t, t ^ {\prime}) K ^ {j} (t, t ^ {\prime}) \prod_ {k \notin (i, j)} C ^ {k} (t, t ^ {\prime})\right) \tag {A.25}
$$

comes from $\overline{Z_{\Pi}}$

$$
\begin{array}{l} \Omega = N ^ {- 1} \sum_ {i, a} \ln \left[ \int D \left[ x _ {a} ^ {i}, \widehat {x} _ {a} ^ {i} \right] p _ {a, t _ {0}} ^ {(i)} \left(x _ {a} ^ {i} (t _ {0})\right) \exp \left(i \int d t \psi_ {a} ^ {i} (t) x _ {a} ^ {i} (t)\right) \right. \\ \times \exp \left(i \sum_ {a, i} \int d t \left[ \hat {x} _ {a} ^ {i} (t) \left(\frac {\dot {x} _ {a} ^ {i} (t)}{x _ {a} ^ {i} (t)} - \left(- T \ln x _ {a} ^ {i} (t) - \rho^ {i} (t) + h _ {a} ^ {i} (t)\right)\right) \right]\right) \\ \times \exp \left(- i \int d t d t ^ {\prime} \left[ \widehat {C} ^ {i} (t, t ^ {\prime}) x _ {a} ^ {i} (t) x _ {a} ^ {i} (t ^ {\prime}) + \widehat {K} ^ {i} (t, t ^ {\prime}) x _ {a} ^ {i} (t) \widehat {x} _ {a} ^ {i} (t ^ {\prime}) + \widehat {L} ^ {i} (t, t ^ {\prime}) \widehat {x} _ {a} ^ {i} (t) \widehat {x} _ {a} ^ {i} (t ^ {\prime}) \right]\right) \Bigg ] \tag {A.26} \\ \end{array}
$$

contains the remaining terms notably, $D[\mathbf{x}, \hat{\mathbf{x}}]$ . We have introduced $p_{a,t_0}^{(i)}(\cdot)$ to represent the initial distribution at $t_0$ . (A.23) takes the form: $I = \int dx\exp (f(x))$ , in the $N\to \infty$ limit the saddle point approximation holds (See (Peng 2024) under Laplace's method) given by:

$$
I \approx \exp (f (\hat {x})) \int d x \exp \left(\frac {1}{2} (x - \hat {x}) ^ {2} f ^ {\prime \prime} (\hat {x})\right) \tag {A.27}
$$

where $\hat{x}$ is the value of x which maximises $f \text{ i.e. } f(\hat{x}) = f_{max}$ . This approximation is found by considering the second-order Taylor expansion around $\hat{x}$ and most of the weight of the integral is near the maxima. Finding the maxima corresponding to the terms in the exponent for (A.23), this involves taking the partial derivatives of F w.r.t $[C, L, K, \widehat{C}, \widehat{L}, \widehat{K}]$ .

$$
0 = \frac {\partial}{\partial C ^ {i}} F = \frac {\partial}{\partial K ^ {i}} F = \frac {\partial}{\partial L ^ {i}} F = \dots
$$

This gives the following relations from taking derivatives of C, K, L:

$$
i \widehat {C} ^ {i} (t, t ^ {\prime}) = \frac {1}{2} \sum_ {j \neq i} \left(L ^ {j} (t, t ^ {\prime}) \prod_ {k \notin (i, j)} C ^ {k} (t, t ^ {\prime}) + \Gamma \sum_ {k \notin (i, j)} K ^ {j} (t, t ^ {\prime}) K ^ {k} (t, t ^ {\prime}) \prod_ {l \notin (i, j, k)} C ^ {l} (t, t ^ {\prime})\right)
$$

$$
i \widehat {K} ^ {i} (t, t ^ {\prime}) = \Gamma \sum_ {j \neq i} K ^ {j} (t, t ^ {\prime}) \prod_ {k \notin (i, j)} C ^ {k} (t, t ^ {\prime})
$$

$$
i \widehat {L} ^ {i} (t, t ^ {\prime}) = \frac {1}{2} \prod_ {j \neq i} C ^ {j} (t, t ^ {\prime}) \tag {A.28}
$$

While taking the derivate w.r.t the conjugate variables $\widehat{C},\widehat{L},\widehat{K}$ yields:

$$
C ^ {i} (t, t ^ {\prime}) = \lim _ {N \rightarrow \infty} N ^ {- 1} \sum_ {a} \left\langle x _ {a} ^ {i} (t) x _ {a} ^ {i} (t ^ {\prime}) \right\rangle_ {\Omega}
$$

$$
K ^ {i} (t, t ^ {\prime}) = \lim _ {N \to \infty} N ^ {- 1} \sum_ {a} \left\langle x _ {a} ^ {i} (t) \widehat {x} _ {a} ^ {i} (t ^ {\prime}) \right\rangle_ {\Omega}
$$

$$
L ^ {i} (t, t ^ {\prime}) = \lim _ {N \rightarrow \infty} N ^ {- 1} \sum_ {a} \left\langle \widehat {x} _ {a} ^ {i} (t) \widehat {x} _ {a} ^ {i} (t ^ {\prime}) \right\rangle_ {\Omega} \tag {A.29}
$$

where the average $\langle \dots \rangle_{\Omega}$ is the average over the probability distribution defined by $\Omega$ (A.26). We note from the properties of the generating functional (A.13), we have the following:

$$
\begin{array}{l} C ^ {i} \left(t, t ^ {\prime}\right) = - \lim _ {N \to \infty} N ^ {- 1} \sum_ {a} \frac {\partial^ {2} Z _ {\mathbf {x}} (\psi)}{\partial \psi_ {a} ^ {i} (t) \partial \psi_ {a} ^ {i} \left(t ^ {\prime}\right)} \Bigg | _ {\boldsymbol {\psi} = \mathbf {h} = 0}, \\ K ^ {i} \left(t, t ^ {\prime}\right) = - \lim _ {N \to \infty} N ^ {- 1} \sum_ {a} \frac {\partial^ {2} Z _ {\mathbf {x}} (\psi)}{\partial \psi_ {a} ^ {i} (t) \partial h _ {a} ^ {i} \left(t ^ {\prime}\right)} \Bigg | _ {\boldsymbol {\psi} = \mathbf {h} = 0}, \\ L ^ {i} \left(t, t ^ {\prime}\right) = - \lim _ {N \to \infty} N ^ {- 1} \sum_ {a} \frac {\partial^ {2} Z _ {\mathbf {x}} (\psi)}{\partial h _ {a} ^ {i} \left(t\right) \partial h _ {a} ^ {i} \left(t ^ {\prime}\right)} \Bigg | _ {\boldsymbol {\psi} = \mathbf {h} = 0} \\ \end{array}
$$

We note the following: $Z_{x}[\psi=0]=1$ , $\forall h$ due to normalisation. Thus $L^{i}(t,t')=0$ , $\forall i,t,t'$ . Due to causality, we have $K^{i}(t,t')=0$ , $\forall t'<t$ . Making the appropriate substitutions, the $\Psi+\Phi$ terms disappear and we are left with $F=N\Omega$ . Assuming a perturbations $h_{a}^{i}(t)=h(t)$ are symmetric with respect to each player and action alongside the initial probability distributions $p_{a,t_{0}}^{(i)}=p_{t_{0}}$ , we drop the dependence of have on a, i to obtain:

$$
\begin{array}{l} \Omega = p \ln \left\{\int \mathcal {D} [ x, \widehat {x} ] p _ {t _ {0}} (x (t _ {0})) \exp \left(\mathrm{i} \int \mathrm{d} t \widehat {x} (t) \left(\frac {\dot {x} (t)}{x (t)} + T \ln x (t) + \rho (t) - h (t)\right)\right) \right. \\ \left. \times \exp \left[ - \int \mathrm{d} t \int \mathrm{d} t ^ {\prime} \left(\Gamma (p - 1) K (t, t ^ {\prime}) C (t, t ^ {\prime}) ^ {p - 2} x (t) \widehat {x} (t ^ {\prime}) + \frac {1}{2} C (t, t ^ {\prime}) ^ {p - 1} \widehat {x} (t) \widehat {x} (t ^ {\prime})\right) \right] \right\} \tag {A.30} \\ \end{array}
$$

Substituting into (A.23) and making the substitution $G(t, t') = -\mathrm{i}K(t, t')$ , we have the following generating function:

$$
\begin{array}{l} Z _ {\text { eff }} = \int \mathcal {D} [ x, \widehat {x} ] p _ {t _ {0}} (x (t _ {0})) \exp \left(\mathrm{i} \int \mathrm{d} t \widehat {x} (t) \left(\frac {\dot {x} (t)}{x (t)} + T \ln x (t) + \rho (t) - h (t)\right)\right) \\ \times \exp \left[ - \int \mathrm{d} t \int \mathrm{d} t ^ {\prime} \left(\mathrm{i} \Gamma G (t, t ^ {\prime}) C (t, t ^ {\prime}) ^ {p - 2} x (t) \widehat {x} (t ^ {\prime}) + \frac {1}{2} C (t, t ^ {\prime}) ^ {p - 1} \widehat {x} (t) \widehat {x} (t ^ {\prime})\right) \right] \tag {A.31} \\ \end{array}
$$

which corresponds to the generating function of the effective dynamic equation

$$
\frac {\dot {x} (t)}{x (t)} = \Gamma \int \mathrm{d} t ^ {\prime} G (t, t ^ {\prime}) C (t, t ^ {\prime}) ^ {p - 2} x (t ^ {\prime}) - T \ln x (t) - \rho (t) + \eta (t) + h (t) \tag {A.32}
$$

where $\eta(t)$ is a coloured (i.e., time-correlated) Gaussian random variable satisfying: $\langle\eta(t)\eta(t')\rangle_{*}=C(t,t')^{p-1}$ . C and G represents the time correlation term and response function respectively and are given by:

$$
C \left(t, t ^ {\prime}\right) = \left\langle x (t) x \left(t ^ {\prime}\right) \right\rangle_ {*}, G \left(t, t ^ {\prime}\right) = \left\langle \frac {\partial x (t)}{\partial h \left(t ^ {\prime}\right)} \right\rangle_ {*}.
$$

where $\langle\ldots\rangle_{*}$ denotes the expected value over realisations of the effective dynamic (A.32). Since $h(t)$ is defined to be zero for all time, we can drop $h(t)$ to finally obtain the effective dynamics as we see in the main paper:

$$
\frac {\dot {x} (t)}{x (t)} = \Gamma \int \mathrm{d} t ^ {\prime} G \left(t, t ^ {\prime}\right) C \left(t, t ^ {\prime}\right) ^ {p - 2} x \left(t ^ {\prime}\right) - T \ln x (t) - \rho (t) + \eta (t) \tag {A.33}
$$

Lastly, the partial derivative of G w.r.t $\eta(t')$ and $h(t')$ are equivalent, we rewrite G as follows:

$$
G \left(t, t ^ {\prime}\right) = \left\langle \frac {\partial x (t)}{\partial \eta \left(t ^ {\prime}\right)} \right\rangle_ {*}
$$

# Solving Fixed points of correlated Games $\Gamma \neq 0$

The fixed point relation (6) and the associated self-consistency relation (7) cannot be solved directly, but rather has to be approximated numerically. We have used a Newton method to find the fixed points by converting the self-consistency relations to loss functions to minimise (do contact the corresponding author for coded implementation in Python). A pseudo-code detailing how we found the fixed point relations for a given coorelation parameter, $\Gamma$ , and exploration rate, T, is given as follows:

Algorithm 1: Fixed point computation for $\Gamma \neq 0$ .   
Input: $\Gamma(\text{Coorelation parameter}), T (\text{Exploration rate}), p (\text{number of players})$ Output: $x : z \to R$ (fixed point distribution of x)
- Let $\hat{x}(z; K, a, b)$ be parameterised by K, a, b, as in (8), (9) from the fixed point relations in the main paper.

We define a general loss function consisting of 3 smaller functions as follows:

$$
\operatorname{Loss} (K, a, b) = \left(\operatorname{loss} _ {1} (K, a, b)\right) ^ {2} + \left(\operatorname{loss} _ {2} (K, a, b)\right) ^ {2} + \left(\operatorname{loss} _ {3} (K, a, b)\right) ^ {2}
$$

where the sub-functions are defined to be:

$$
\begin{array}{l} - \mathrm{loss} _ {1} (K, a, b) = \mathbf {q} ^ {(p - 1) / 2} - \int_ {- \infty} ^ {\infty} D z \frac {\delta \hat {x} (z)}{\delta z} \chi \\ - \operatorname{loss} _ {2} (K, a, b) = \mathbf {q} - \int_ {- \infty} ^ {\infty} D z \hat {x} (z) ^ {2} \\ - \operatorname{loss} _ {3} (K, a, b) = 1 - \int_ {- \infty} ^ {\infty} D z \hat {x} (z) \\ \end{array}
$$

where the integrals are estimated numerically and $\mathbf{q}$ and $\chi$ are determined from $a, b$ . Recall:

$$
\begin{array}{l} - a = \Gamma \mathbf {q} ^ {(p - 2)} \\ - b = \mathbf {q} ^ {(p - 1) / 2} T ^ {- 1} \\ - D z = \exp (- z ^ {2} / 2) / \sqrt {2 \pi} d z \\ \end{array}
$$

Given an initial guess, $K_0, a_0, b_0$ , we use a standard Newton method to find $\mathbf{K}$ , $\mathbf{a}$ , $\mathbf{b}$ which minimise following the general loss function:

$$
\mathbf {K}, \mathbf {a}, \mathbf {b} = \text { minimise } (\text { Loss } (K _ {0}, a _ {0}, b _ {0}))
$$

For almost any input of $T$ , $\Gamma$ and $p$ (in the relevant regime) $^a$ , we can obtain a Loss of almost 0 ( $< 10^{-16}$ ) at the minima. This corresponds to finding parameterisation of $x(z; K, a, b)$ , which almost exactly solves the fixed point relations (6) and the associated self-consistency relation (7).

Return $\hat{x} (z;\mathbf{K},\mathbf{a},\mathbf{b})$

# Solving Fixed points of Uncorrelated Games $\Gamma = 0$

In this section, we will solve the fixed points of uncorrelated games. As discussed in the main paper, fixed points can be internal or on the boundary depending on the exploring rate. We can observe this behaviour change in numerical simulations:

Figures A.7 demonstrate when $\Gamma = 0$ , the fixed points can be internal or on the boundary depending on the exploration rate T. This is illustrated over sample 2-player games with varying T. The asymptotic extinction property of the internal fixed point is demonstrated in A.8, which displays how the frequency of the playing a select 0.5% of actions goes to zero as N increases.

Solving the fixed point distribution of uncorrelated games varies from $\Gamma \neq 0$ , due to the absence of the $a$ term. The solve of this fixed point distribution will proceed in two parts:

- Showing the fixed points are internal when $T \geq \sqrt{3e(p - 1) / 2}$ and solving the fixed point distribution when $T \geq \sqrt{3e(p - 1) / 2}$   
- Solving the fixed point distribution when $T < \sqrt{3e(p - 1) / 2}$

Fixed points are internal when $T \geq \sqrt{3e(p-1)/2}$ . To identify the critical exploration rate for fixed point to remain internal, we need to check the following distribution for $x(z)$ fulfils the self-consistency conditions (7):

$$
x (z) = K e ^ {b z}, \forall z \in \mathbb {R}
$$

The self-consistency condition corresponding to the $\chi$ term disappears. We have to check the remaining to conditions. A substitution of $x(z)$ yields:

$$
\begin{array}{l} q = \int_ {- \infty} ^ {\infty} x ^ {2} (z) D z \\ = \frac {K}{\sqrt {2 \pi}} \int_ {- \infty} ^ {\infty} \exp \left(2 b z - \frac {z ^ {2}}{2}\right) d z \\ = K \exp (2 b ^ {2}) \tag {A.34} \\ \end{array}
$$

$$
\begin{array}{l} q = \int_ {- \infty} ^ {\infty} x ^ {2} (z) D z \\ = \frac {K}{\sqrt {2 \pi}} \int_ {- \infty} ^ {\infty} \exp \left(2 b z - \frac {z ^ {2}}{2}\right) d z \\ = K \exp (2 b ^ {2}) \tag {A.34} \\ \end{array}
$$

Histogram of marginal probability of actions for $\Gamma = 0$ , $T = 1.79$   
![](images/be4f4575e80d540dc1095092a911f94126c61144275316f806edbf4aa6c76708.jpg)

<details>
<summary>histogram</summary>

| Marginal Probability at the fixed point | count |
| --------------------------------------- | ----- |
| 0.0000                                  | 600   |
| 0.0001                                  | 1700  |
| 0.0002                                  | 1800  |
| 0.0003                                  | 1600  |
| 0.0004                                  | 1400  |
| 0.0005                                  | 1200  |
| 0.0006                                  | 1000  |
| 0.0007                                  | 800   |
| 0.0008                                  | 600   |
| 0.0009                                  | 400   |
| 0.0010                                  | 200   |
| 0.0011                                  | 100   |
| 0.0012                                  | 50    |
| 0.0013                                  | 30    |
| 0.0014                                  | 20    |
| 0.0015                                  | 15    |
| 0.0016                                  | 10    |
| 0.0017                                  | 8     |
| 0.0018                                  | 5     |
| 0.0019                                  | 3     |
| 0.0020                                  | 2     |
| 0.0021                                  | 1     |
| 0.0022                                  | 1     |
| 0.0023                                  | 1     |
| 0.0024                                  | 1     |
| 0.0025                                  | 1     |
| 0.0026                                  | 1     |
| 0.0027                                  | 1     |
| 0.0028                                  | 1     |
| 0.0029                                  | 1     |
| 0.0030                                  | 1     |
| 0.0031                                  | 1     |
| 0.0032                                  | 1     |
| 0.0033                                  | 1     |
| 0.0034                                  | 1     |
| 0.0035                                  | 1     |
| 0.0036                                  | 1     |
| 0.0037                                  | 1     |
| 0.0038                                  | 1     |
| 0.0039                                  | 1     |
| 0.0040                                  | 1     |
| 0.0041                                  | 1     |
| 0.0042                                  | 1     |
| 0.0043                                  | 1     |
| 0.0044                                  | 1     |
| 0.0045                                  | 1     |
| 0.0046                                  | 1     |
| 0.0047                                  | 1     |
| 0.0048                                  | 1     |
| 0.0049                                  | 1     |
| 0.0050                                  | 1     |
| 0.0051                                  | 1     |
| 0.0052                                  | 1     |
| 0.0053                                  | 1     |
| 0.0054                                  | 1     |
| 0.0055                                  | 1     |
| 0.0056                                  | 1     |
| 0.0057                                  | 1     |
| 0.0058                                  | 1     |
| 0.0059                                  | 1     |
| 0.0060                                  | 1     |
| 0.0061                                  | 1     |
| 0.0062                                  | 1     |
| 0.0063                                  | 1     |
| 0.0064                                  | 1     |
| 0.0065                                  | 1     |
| 0.0066                                  | 1     |
| 0.0067                                  | 1     |
| 0.0068                                  | 1     |
| 0.0069                                  | 1     |
| 0.0070                                  | 1     |
| 0.0071                                  | 1     |
| 0.0072                                  | 1     |
| 0.0073                                  | 1     |
| 0.0074                                  | 1     |
| 0.0075                                  | 1     |
| 0.0076                                  | 1     |
| 0.0077                                  | 1     |
| 0.0078                                  | 1     |
| 0.0079                                  | 1     |
| 0.0080                                  | 1     |
| Note: The actual values may vary due to the random nature of the data generation process. The provided values are just an example of the actual output from the code execution.
</details>

Histogram of marginal probability of actions for $\Gamma = 0$ , $T = 2.2$   
![](images/83029d57bb7707fa83cce7d394ad64d8baaed7802a3bdf44425ff04a602187c4.jpg)

<details>
<summary>histogram</summary>

| Marginal Probability at the fixed point | count |
| --------------------------------------- | ----- |
| 0.0000                                  | 0     |
| 0.0002                                  | 500   |
| 0.0004                                  | 1500  |
| 0.0006                                  | 1800  |
| 0.0008                                  | 1600  |
| 0.0010                                  | 1400  |
| 0.0012                                  | 1200  |
| 0.0014                                  | 1000  |
| 0.0016                                  | 800   |
| 0.0018                                  | 600   |
| 0.0020                                  | 400   |
| 0.0022                                  | 200   |
| 0.0024                                  | 100   |
| 0.0026                                  | 50    |
| 0.0028                                  | 25    |
| 0.0030                                  | 15    |
| 0.0032                                  | 10    |
| 0.0034                                  | 5     |
| 0.0036                                  | 2     |
| 0.0038                                  | 1     |
| 0.0040                                  | 1     |
| 0.0042                                  | 1     |
| 0.0044                                  | 1     |
| 0.0046                                  | 1     |
| 0.0048                                  | 1     |
| 0.0050                                  | 1     |
| 0.0052                                  | 1     |
| 0.0054                                  | 1     |
| 0.0056                                  | 1     |
| 0.0058                                  | 1     |
| 0.0060                                  | 1     |
| 0.0062                                  | 1     |
| 0.0064                                  | 1     |
| 0.0066                                  | 1     |
| 0.0068                                  | 1     |
| 0.0070                                  | 1     |
| 0.0072                                  | 1     |
| 0.0074                                  | 1     |
| 0.0076                                  | 1     |
| 0.0078                                  | 1     |
| 0.0080                                  | 1     |
| 0.0082                                  | 1     |
| 0.0084                                  | 1     |
| 0.0086                                  | 1     |
| 0.0088                                  | 1     |
| 0.0090                                  | 1     |
| 0.0092                                  | 1     |
| 0.0094                                  | 1     |
| 0.0096                                  | 1     |
| 0.0098                                  | 1     |
| 0.0100                                  | 1     |
</details>

Figure A.7: Histogram of marginal probability of playing a random selected actions from ensemble numerical simulations. These estimate the likelihood distribution of a randomly selected action under Q-Learning. For each plot, 160 2-player, N = 320 action games are simulated with a randomly selected initial strategy until convergence. The probabilities distribution of the actions at the fixed point is then recorded to form a histogram. On the left, the exploration rate $T = 1.79$ remains in the unique fixed point regime, but is set below $\sqrt{3e/2} \approx 2.01$ the threshold and a small fraction of strategies are played near 0. On the right, the exploration rate $T = 2.2$ is above this threshold and there is no probability mass near 0. This indicates the fixed points are internal.

$$
\begin{array}{l} 1 = \int_ {- \infty} ^ {\infty} x (z) D z \\ = \frac {K}{\sqrt {2 \pi}} \int_ {- \infty} ^ {\infty} \exp \left(b z - \frac {z ^ {2}}{2}\right) d z \\ = K \exp \left(\frac {b ^ {2}}{2}\right) \tag {A.35} \\ \end{array}
$$

Raising (A.35) to the $4^{th}$ power, and expression q as a function of T we have:

$$
q (T) = K ^ {- 3} \tag {A.36}
$$

Since there is no dependence on $a$ , solving $q$ will yield us $b, K$ which are sufficient to find $x(z)$ . Substituting $b = \mathbf{q}^{(p - 1) / 2}T^{-1}$ and (A.36) into (A.34), we have:

$$
T = \sqrt {\frac {3}{2} \left(\frac {\ln (q (T))}{q (T) ^ {(p - 1)}}\right)} \tag {A.37}
$$

It can easily be checked that (A.37) has a solution for $q(T)$ when $T \geq \sqrt{3e(p - 1) / 2}$ . Thus, above this critical exploration rate, all fixed points for $\Gamma = 0$ are internal.

Solving for $T < \sqrt{3e(p-1)/2}$ For lower exploration rates, T, the fixed points are internal points. $x(z)$ becomes:

$$
x (z) = \left\{ \begin{array}{l l} K e ^ {b z} & , z <   z _ {\text {crit}} \\ 0 & , z \geq z _ {\text {crit}} \end{array} \right. \tag {A.38}
$$

Now, $z_{crit}$ becomes another parameter that has to be identified. A substitution of (A.38) into the self-consistency relations yield:

$$
\begin{array}{l} q = \int_ {- \infty} ^ {z _ {\mathrm{crit}}} x ^ {2} (z) D z \\ = \frac {K}{\sqrt {2 \pi}} \int_ {- \infty} ^ {z _ {\mathrm{crit}}} \exp \left(2 b z - \frac {z ^ {2}}{2}\right) d z \\ = \frac {K}{2} \exp (2 b ^ {2}) \left(\operatorname{erf} \left(\frac {\sqrt {2} \left(z _ {\text {crit}} - 2 b\right)}{2}\right) + 1\right) \\ = \frac {K}{2} \exp (2 b ^ {2}) Q \tag {A.39} \\ \end{array}
$$

![](images/40b325b54ce9588fb58ed02bb64318569853671bf0f0a947df496d6029b3573a.jpg)

<details>
<summary>line</summary>

| no. of actions(N) | N x 0.5th percentile |
| ----------------- | --------------------- |
| 20                | 10^-2                 |
| 40                | 10^-4                 |
| 80                | 10^-4                 |
| 160               | 10^-8                 |
| 320               | 10^-14                |
</details>

![](images/54177180559c398332e36ac0618042c2249332570112dd8ebf1f84818b79acd0.jpg)

<details>
<summary>line</summary>

| no. of actions(N) | Probability mass of bottom 0.5th percentile |
| ----------------- | ------------------------------------------- |
| 20                | 0.13                                        |
| 40                | 0.145                                       |
| 80                | 0.165                                       |
| 160               | 0.17                                        |
| 320               | 0.165                                       |
</details>

Figure A.8: Plots of the how often the bottom 0.5th percentiles of action are played varying game size N for games with varying exploration rate T = 1.79, 2.2. For games with action sizes N = [10, 20, 40, 80, 160, 320] we run the following number of games: [2560, 1280, 640, 320, 160] until convergence to obtain around 200,000 fixed points. When T = 1.79 (left), we notice the bottom 0.5th percentiles of strategies are played with decreasing probability as the games scale. (Note the log scaling on the y-axes) This corresponds to asymptotic extinction, where actions going extinct in the $N \rightarrow \infty$ limit. When $T = 2.2 \geq \sqrt{3e(p - 1)/2}$ (right), this does not occur. The probability mass of playing the bottom x% of actions does not depend on N

$$
\begin{array}{l} 1 = \int_ {- \infty} ^ {z _ {\text { crit }}} x (z) D z \\ = \frac {K}{\sqrt {2 \pi}} \int_ {- \infty} ^ {z _ {\mathrm{crit}}} \exp \left(b z - \frac {z ^ {2}}{2}\right) d z \\ = \frac {K}{2} \exp \left(\frac {b ^ {2}}{2}\right) \left(\operatorname{erf} \left(\frac {\sqrt {2} (z _ {\text {crit}} - b)}{2}\right) + 1\right) \\ = \frac {K}{2} \exp \left(\frac {b ^ {2}}{2}\right) E \tag {A.40} \\ \end{array}
$$

where:

$$
Q = \operatorname{erf} \left(\frac {\sqrt {2} (z _ {\text {crit}} - 2 b)}{2}\right) + 1
$$

$$
E = \operatorname{erf} \left(\frac {\sqrt {2} (z _ {\text {crit}} - b)}{2}\right) + 1
$$

Substituting (A.39) into (A.40). Noting Q, E and q are dependent on $z_{crit}$ and T, we have the following expression for q:

$$
q = 8 K ^ {- 3} (Q E ^ {- 4}) \tag {A.41}
$$

A substitution into (A.39) yields. The following relation:

$$
q = \exp \left(\frac {3}{2} q ^ {(p - 1)} T ^ {- 2}\right) Q E ^ {- 1} \tag {A.42}
$$

$z_{crit}$ given by the maximum value of z where there exists a solution for q such that the above relation holds. This expression cannot be solved analytically, and has to be solved numerically.

# Derivation of the stability condition

In this segment, we detail the stability analysis to obtain the following stability condition for the fixed points in the main paper:

$$
\phi \left\langle \left| \frac {T}{x ^ {\star}} - \Gamma \mathbf {q} ^ {p - 2} \chi \right| ^ {- 2} \right\rangle_ {*} <   ((p - 1) \mathbf {q} ^ {p - 2}) ^ {- 1}
$$

As discussed, the procedure will be broken up into the following steps:

- Introducing an $\varepsilon$ -perturbation and linearising the dynamics at the fixed point.   
- Taking a Laplace Transform to work in the (complex) frequency domain.   
- Finding a stability criterion as a function of the Q-Learning parameters.

Obtaining the Linearised Equations Beginning with effective dynamics equation (A.32), we introduce a small white noise term $\varepsilon(t)$ and perturb $x(t)$ and $\eta(t)$ slightly at their respective fixed points:

$$
x (t) = \mathbf {x} ^ {\star} + \underbrace {\hat {x} (t)} _ {\mathcal {O} (\varepsilon)}, \eta (t) = \eta^ {\star} + \underbrace {\hat {\eta} (t)} _ {\mathcal {O} (\varepsilon)} \tag {A.43}
$$

Substituting (A.43) into (A.32), keeping only the $\mathcal{O}(\varepsilon)$ terms, we obtain the linearised equations about the fixed point:

$$
\frac {d}{d t} \hat {x} (t) = - T \hat {x} (t) + x ^ {\star} \left[ \Gamma \int \mathrm{d} t ^ {\prime} H (t, t ^ {\prime}) \hat {x} \left(t ^ {\prime}\right) + \hat {\eta} (t) + \varepsilon (t) \right] \tag {A.44}
$$

where:

$$
- H (t, t ^ {\prime}) = G (t, t ^ {\prime}) C (t, t ^ {\prime}) ^ {p - 2}
$$

We note the following Taylor's Expansion result when substituting $x(t)$ into the ln.

$$
\ln (x (t)) = \ln (x ^ {\star} + \hat {x}) = \ln (x ^ {\star}) + \underbrace {\frac {\hat {x}}{x ^ {\star}}} _ {\mathcal {O} (\varepsilon)} + \underbrace {\cdots} _ {\mathcal {O} (\varepsilon^ {2})}
$$

Laplace Transforms Now that we have obtained the linearised equations (A.44), the task is now to solve the following initial value problem: given an initial perturbation, predict the behaviour of $\hat{x}, \hat{\eta} \ldots$ moving forward in time. This is a linear initial value problem and thus, can be solved via the method of Laplace Transform. The Laplace transform is defined as follows:

$$
\tilde {x} (\sigma) = \int_ {0} ^ {\infty} \hat {x} (t) e ^ {- \sigma t} d t
$$

where:

$$
- \sigma \in \mathbb {C}
$$

The idea of the Laplace transform is to seek solution to (A.44) expressed as a superposition of simpler solutions of the following form:

$$
\hat {x} (t) = \tilde {x} (\sigma) e ^ {\sigma t}
$$

where:

\- $\sigma$ can be thought of the growth rate of the perturbation. If $\Re(\sigma) > 0$ , this perturbation grows in time.

Taking the Laplace Transform of (A.44), we go from the time to the frequency domain, and through the following substitutions:

$$
\hat {x} (t) \rightarrow \tilde {x} (\sigma), \hat {\varepsilon} (t) \rightarrow \tilde {\varepsilon} (\sigma),
$$

$$
\hat {\eta} (t) \rightarrow \tilde {\eta} (\sigma), H (t, t ^ {\prime}) \rightarrow \tilde {H} (\sigma), \frac {\partial}{\partial t} \rightarrow \sigma
$$

We obtain the following frequency relation:

$$
A (\sigma , x ^ {\star}) \tilde {x} (\sigma) = \tilde {\eta} (\sigma) + \tilde {\varepsilon} (\sigma) \tag {A.45}
$$

where:

$$
A (\sigma , x ^ {\star}) = \frac {\sigma + T}{x ^ {\star}} - \Gamma \tilde {H} (\sigma)
$$

A distinguishing feature between the Laplace Transform over the Fourier is its ability to handle complex eigenvalues, as it is a one-sided transformation. Taking the square of (A.45) we have:

$$
(\tilde {x} (\sigma)) ^ {2} = (A (\sigma , x ^ {\star})) ^ {- 2} \times \left((\tilde {\eta} (\sigma)) ^ {2} + (\tilde {\varepsilon} (\sigma)) ^ {2} + 2 \tilde {\eta} (\sigma) \tilde {\varepsilon} (\sigma)\right) \tag {A.46}
$$

Integrating (A.46) with respect to the all possible realisations we have:

$$
\left\langle | \tilde {x} (\sigma) | ^ {2} \right\rangle_ {*} = \left\langle | A (\sigma , x ^ {\star}) | ^ {- 2} \right\rangle_ {*} \times \phi \left(\left\langle | \tilde {\eta} (\sigma) | ^ {2} \right\rangle_ {*} + \left\langle | \tilde {\varepsilon} (\sigma) | ^ {2} \right\rangle_ {*}\right) \tag {A.47}
$$

where:

\- $\phi$ is the proportion of strategies which do not go extinct at the fixed point, which is given by $P(z < z_{\mathrm{crit}}), z \sim \mathcal{N}(0,1)$

We note:

- The $\tilde{\eta}(\sigma)\tilde{\varepsilon}(\sigma)$ -term is eliminated as the expectation the product of random variable $\eta$ and an uncorrelated zero-mean variable $\varepsilon$ is 0.   
- $\phi$ takes into account only perturbations on the non-extinct strategies $x^{\star} > 0$ are relevant. We suspect this scalar idea is that if a previously extinct strategy is re-introduced, it would go extinct again (see (Opper and Diederich 1992) for similar calculations). We note $\phi = 1$ when $\Gamma < 0$ , but $\phi < 1$ when $\Gamma > 0$ .

Finding a stability criterion We wish to now manipulate (A.47) to generate a stability criterion. From (A.32), we make a substitution for $\tilde{\eta}$ term. First note:

$$
\langle \eta (t) \eta (t ^ {\prime}) \rangle_ {*} = \langle x (t) x (t ^ {\prime}) \rangle_ {*} ^ {p - 1}
$$

Subbing in (A.43) and matching the $\mathcal{O}(\epsilon^2)$ terms, we have:

$$
\begin{array}{l} \left\langle\left(\eta^ {\star} + \hat {\eta} (t)\right) ^ {2} \right\rangle_ {*} \rightarrow \left\langle | \tilde {\eta} (\sigma) | ^ {2} \right\rangle_ {*} \\ = \left\langle (\mathbf {x} ^ {\star} + \hat {x} (t)) ^ {2} \right\rangle_ {*} ^ {p - 1} \rightarrow (p - 1) \left\langle (\mathbf {x} ^ {\star}) ^ {2} \right\rangle_ {*} ^ {p - 2} \left\langle | \tilde {x} (\sigma) | ^ {2} \right\rangle_ {*} \tag {A.48} \\ \end{array}
$$

Substituting into (A.47) and re-arranging we have:

$$
\left\langle | \tilde {x} (\sigma) | ^ {2} \right\rangle_ {*} = \left\langle | \tilde {\varepsilon} (\sigma) | ^ {2} \right\rangle_ {*} \times \left(\phi \left\langle | A (\sigma , x ^ {\star}) | ^ {- 2} \right\rangle_ {*} ^ {- 1} - (p - 1) \mathbf {q} ^ {p - 2}\right) ^ {- 1}
$$

Since the LHS is strictly positive, to avoid a contradiction, we need:

$$
\phi \left\langle | A (\sigma , x ^ {\star}) | ^ {- 2} \right\rangle_ {*} <   ((p - 1) \left\langle (\mathbf {x} ^ {\star}) ^ {2} \right\rangle_ {*} ^ {p - 2}) ^ {- 1} \tag {A.49}
$$

A fixed point becomes stable when the perturbation decays in time, this corresponds to $\Re\left(\frac{\partial}{\partial t}\right)=\Re(\sigma)<0$ . The boundary of stability is given when $\Re(\sigma)=0$ . Thus, stability occurs where the following holds, $\forall k\in\mathbb{R}$ :

$$
\phi \left\langle \left| \frac {i k + T}{x ^ {\star}} - \Gamma \mathbf {q} ^ {p - 2} \chi \right| ^ {- 2} \right\rangle_ {*} <   ((p - 1) \left\langle (\mathbf {x} ^ {\star}) ^ {2} \right\rangle_ {*} ^ {p - 2}) ^ {- 1} \tag {A.50}
$$

Taking a look at the LHS, we can split it into real and imaginary components. We note it is maximised when k = 0, when the imaginary component of the eigenmode = 0, i.e:

$$
\begin{array}{l} \phi \left\langle \left| \underbrace {\frac {i k}{x ^ {\star}}} _ {\mathfrak {I}} + \underbrace {\frac {T}{x ^ {\star}} - \Gamma \mathbf {q} ^ {p - 2} \chi} _ {\mathfrak {R}} \right| ^ {- 2} \right\rangle_ {*} = \phi \left\langle \left| | \mathfrak {I} | ^ {2} + | \mathfrak {R} | ^ {2} \right| ^ {- 1} \right\rangle_ {*} \\ > \phi \left\langle \left| \frac {T}{x ^ {\star}} - \Gamma \mathbf {q} ^ {p - 2} \chi \right| ^ {- 2} \right\rangle_ {*} \tag {A.51} \\ \end{array}
$$

Thus, we obtain the following stability condition obtained in the main paper:

$$
\phi \left\langle \left| \frac {T}{x ^ {\star}} - \Gamma \mathbf {q} ^ {p - 2} \chi \right| ^ {- 2} \right\rangle_ {*} <   ((p - 1) \mathbf {q} ^ {p - 2}) ^ {- 1}
$$

A Discussion on the stability relation In the main paper, we mentioned there is a subtle difference between (Sanders, Farmer, and Galla 2018) and our derivation. This alongside, other previous work (Galla and Farmer 2013) has used a Fourier Transform and only considered the real eigenmodes at 0, while we allow for complex eigenmodes. On the surface, the difference may appear somewhat irrelevant, as we recover the identical stability condition for $\Gamma < 0$ . This raises the question: Why are we allowed to ignore complex eigenmodes when in transitions to instability? It seems that in numerical simulations holding a competitive game fixed $\Gamma < 0$ and beginning with a high exploration rate in the unique fixed point regime and lowering $T$ , it appears system transitions to instability via a Hopf Bifurcation. The formation of limit cycles also suggests complex eigenvalues are at play in destabilising Q-Learning. Further analysis is required to understand this effect.

# Plots of selected simulations

Player 1's strategy over time, N=20, p=3, Γ=-0.1, T=2.2   
![](images/cc688c13525aa4097c87d16e5658988ebf644a901c8269cab02b0e02da7ae28b.jpg)

<details>
<summary>area</summary>

| time | Probability of each action |
| ---- | -------------------------- |
| 0    | 0.0                        |
| 25   | 0.1                        |
| 50   | 0.2                        |
| 75   | 0.3                        |
| 100  | 0.4                        |
| 125  | 0.5                        |
| 150  | 0.6                        |
| 175  | 0.7                        |
| 200  | 0.8                        |
</details>

Player 1's strategy over time, N=20, p=3, Γ=-0.1, T=1.8   
![](images/974e8d394ecb8c0caba8b2ac05676e8323c8e4585811612a3d0e206891954452.jpg)

<details>
<summary>area_stacked</summary>

| time | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Layer 5 | Layer 6 | Layer 7 | Layer 8 | Layer 9 | Layer 10 |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|
| 0    | 0.0     | 0.0     | 0.0     | 0.0     | 0.0     | 0.0     | 0.0     | 0.0     | 0.0     | 0.0      |
| 500  | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1      |
| 1000 | 0.2     | 0.2     | 0.2     | 0.2     | 0.2     | 0.2     | 0.2     | 0.2     | 0.2     | 0.2      |
| 1500 | 0.3     | 0.3     | 0.3     | 0.3     | 0.3     | 0.3     | 0.3     | 0.3     | 0.3     | 0.3      |
| 2000 | 0.4     | 0.4     | 0.4     | 0.4     | 0.4     | 0.4     | 0.4     | 0.4     | 0.4     | 0.4      |
| 2500 | 0.5     | 0.5     | 0.5     | 0.5     | 0.5     | 0.5     | 0.5     | 0.5     | 0.5     | 0.5      |
| 3000 | 0.6     | 0.6     | 0.6     | 0.6     | 0.6     | 0.6     | 0.6     | 0.6     | 0.6     | 0.6      |
| 3500 | 0.7     | 0.7     | 0.7     | 0.7     | 0.7     | 0.7     | 0.7     | 0.7     | 0.7     | 0.7      |
| 4000 | 0.8     | 0.8     | 0.8     | 0.8     | 0.8     | 0.8     | 0.8     | 0.8     | 0.8     | 0.8      |
| 4500 | 0.9     | 0.9     | 0.9     | 0.9     | 0.9     | 0.9     | 0.9     | 0.9     | 0.9     | 0.9      |
| 5000 | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0     | 1.0      |
</details>

Player 1's strategy over time, N=20, p=3, Γ=-0.5, T=1   
![](images/d7120e8db9cb6c925b77813c2a83a280078fe91e12b867ae0cc9a86c2547c5bc.jpg)

<details>
<summary>line</summary>

| time | Probability of each action |
| ---- | -------------------------- |
| 0    | 0.0                        |
| 250  | 0.0                        |
| 500  | 0.0                        |
| 750  | 0.0                        |
| 1000 | 0.0                        |
| 1250 | 0.0                        |
| 1500 | 0.0                        |
| 1750 | 0.0                        |
| 2000 | 0.0                        |
</details>

Figure A.9: Time series of strategy evolution under what appears to be different types of dynamical behaviour: convergence to a fixed point (top), a limit cycle (middle) and chaotic behaviour (bottom). At any given time, the probability over all actions must sum to 1. Thus, we can represent strategy evolution with a stacked time chart, where each colour represents a unique action and the thickness represents the probability of the action being played. The above plots demonstrate how the strategy of the first player evolves over time for different choices of exploration rate T and $\Gamma$ in a p = 3, N = 20 action game. Limit cycles occur occasionally on the boundary, but this highly depends on the initialisation of the payoffs.

![](images/5d830a978a59f731080df3f8c3053ae31303fe2bfd48ebccaec4870c42cb53b2.jpg)

<details>
<summary>area</summary>

| time | Probability of each action |
| ---- | -------------------------- |
| 0    | 0.0                        |
| 25   | 0.4                        |
| 50   | 0.8                        |
| 75   | 0.6                        |
| 100  | 0.7                        |
| 125  | 0.7                        |
| 150  | 0.7                        |
| 175  | 0.7                        |
| 200  | 0.7                        |
</details>

![](images/381bf28e28e42e163058f9a8bf6303c11eb97482c5a49a55b83bcf16396e2c7b.jpg)

<details>
<summary>area</summary>

| time | Probability of each action |
| ---- | -------------------------- |
| 0    | 0.8                        |
| 25   | 0.6                        |
| 50   | 0.4                        |
| 75   | 0.2                        |
| 100  | 0.0                        |
| 125  | 0.0                        |
| 150  | 0.0                        |
| 175  | 0.0                        |
| 200  | 0.0                        |
</details>

Figure A.10: Convergence to different fixed points for two different initial conditions for the same randomly generated game of p = 3, N = 20 actions, when $\Gamma = 0.5$ , T = 2.5.

![](images/af42d14647fc5ed9553c3a245ee97b849c3ee4a138548d79b57c491a2444f3d1.jpg)

<details>
<summary>histogram</summary>

| Frequency at the fixed point | count |
| ---------------------------- | ----- |
| 0.0000 - 0.0001              | 1800  |
| 0.0001 - 0.0002              | 750   |
| 0.0002 - 0.0003              | 250   |
| 0.0003 - 0.0004              | 150   |
| 0.0004 - 0.0005              | 100   |
| 0.0005 - 0.0006              | 80    |
| 0.0006 - 0.0007              | 60    |
| 0.0007 - 0.0008              | 40    |
| 0.0008 - 0.0009              | 30    |
| 0.0009 - 0.0010              | 25    |
</details>

![](images/7347e86ab015b6cde9b569d3bb0725354a5d522079b4043ada35009993786157.jpg)

<details>
<summary>line</summary>

| no. of actions(N) | Probability mass |
| ----------------- | ---------------- |
| 20                | 1.0              |
| 40                | 0.8              |
| 80                | 0.6              |
| 160               | 0.1              |
| 320               | 0.001            |
</details>

Figure A.11: (Left) Histogram plot of showing the marginal distribution of how often actions are being played from ensemble N = 320-action 2-player games $\Gamma = 0.5, T = 2.2$ at the fixed point from random initial conditions. This corresponds to the multiple fixed point regime. Note the concentration of strategies played with near 0 probability. (Right) Holding the rest of the parameters fixed and varying N, a plot of the probability mass of the bottom 90%. actions is generated. This implies there is a concentration of probabilities on a small fraction of actions at low exploration rates in $\Gamma > 0$ .

![](images/831ac99844d02250af003dccc9a0931481c06e4a9c35dd056678a887f1335be4.jpg)

<details>
<summary>histogram</summary>

| Frequency at the fixed point | count |
| ---------------------------- | ----- |
| 0.0000 - 0.0001              | 0     |
| 0.0001 - 0.0002              | 500   |
| 0.0002 - 0.0003              | 1500  |
| 0.0003 - 0.0004              | 2200  |
| 0.0004 - 0.0005              | 1800  |
| 0.0005 - 0.0006              | 1200  |
| 0.0006 - 0.0007              | 800   |
| 0.0007 - 0.0008              | 400   |
| 0.0008 - 0.0009              | 200   |
| 0.0009 - 0.0010              | 100   |
</details>

![](images/0a7efcc2cfcce4ed0ec7e37d458c9a1baf723a73fb61c2c645cb228437afbb68.jpg)

<details>
<summary>histogram</summary>

| Frequency at the fixed point | count |
| ---------------------------- | ----- |
| 0.0000 - 0.0001              | 0     |
| 0.0001 - 0.0002              | 50    |
| 0.0002 - 0.0003              | 150   |
| 0.0003 - 0.0004              | 250   |
| 0.0004 - 0.0005              | 180   |
| 0.0005 - 0.0006              | 120   |
| 0.0006 - 0.0007              | 80    |
| 0.0007 - 0.0008              | 40    |
| 0.0008 - 0.0009              | 20    |
| 0.0009 - 0.0010              | 10    |
</details>

Figure A.12: Histogram plot of showing the marginal distribution of how often actions are being played from 160 random trajectories from N = 320-action 2-player games $\Gamma = 0.5$ for T = 2.6 (left) and T = 3 (right). Both these parameters are in the unique fixed point regime given by $T > T_{crit} = 2.51$ . It appears strategies only appear to go extinct on the left plot (T = 2.6). As discussed in the main paper, this is likely due to extinctions being extremely rare for parameters away from the boundary. At T = 3, extinctions are expected to be approximately a 1 in $10^{8}$ event, making it difficult to verify.