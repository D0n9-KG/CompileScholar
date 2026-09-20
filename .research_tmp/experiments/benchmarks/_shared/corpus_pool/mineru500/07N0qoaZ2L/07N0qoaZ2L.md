# Improved Analysis for Bandit Learning in Matching Markets

Fang Kong Southern University of Science and Technology kongf@sustech.edu.cn

Zilong Wang Shanghai Jiao Tong University wangzilong@sjtu.edu.cn

Shuai Li<sup>∗</sup> Shanghai Jiao Tong University shuaili8@sjtu.edu.cn

## Abstract

A rich line of works study the bandit learning problem in two-sided matching markets, where one side of market participants (players) are uncertain about their preferences and hope to find a stable matching during iterative matchings with the other side (arms). The state-of-the-art analysis shows that the player-optimal stable regret is of order O(K log T/∆<sup>2</sup>) where K is the number of arms, T is the horizon and ∆ is the players’ minimum preference gap. However, this result may be far from the lower bound Ω(max{N log T /∆<sup>2</sup>, K log T /∆}) since the number K of arms (workers, publisher slots) may be much larger than that N of players (employers in labor markets, advertisers in online advertising, respectively). In this paper, we propose a new algorithm and show that the regret can be upper bounded by $O ( \dot { N } ^ { 2 } \log T / \Delta ^ { 2 } + \breve { K } \log T / \Delta )$ . This result removes the dependence on K in the main order term and improves the state-of-the-art guarantee in common cases where N is much smaller than K. Such an advantage is also verified in experiments. In addition, we provide a refined analysis for the existing centralized UCB algorithm and show that, under α-condition, it achieves an improved O(N log T/∆<sup>2</sup> + K log T/∆) regret.

## 1 Introduction

The two-sided matching market problem has been extensively studied in the literature due to its wide range of applications like labor market, school admission, house allocation, and online advertising [26, 9, 1]. There are two sides of participants in the market, such as the employers and workers in the labor market, advertisers and publishers in online advertising. Each participant on the one side has a preference ranking over the other side. The concept of stability, which characterizes the equilibrium state of the market where no participant wants to break up the current matching relationship and find another partner, has attracted great interest from researchers [26]. Achieving stability is critical for ensuring the long-term viability of the market.

A rich line of works [9, 15, 25] study how to find a stable matching in the market. Most of them assume the preference ranking of each market participant is known beforehand, which we refe to as the offline setting. However, in real applications, the knowledge of the preferences may be uncertain. For example, in the labor market, employers usually do not know the working abilities of workers before being matched, and advertisers also do not know the exact conversion rate of placing the advertisement in a publisher slot. This makes the traditional algorithms unavailable to find an exact stable matching. With the emergence of online market platforms such as the online labor market UpWork and TaskRabbit as well as online advertising platforms where employers or advertisers have many similar tasks, market participants are able to learn their unknown preferences during iterative matchings with the other side of agents.

Multi-armed bandit (MAB) is a classic framework that characterizes the learning process during iterative interactions [3, 19]. It considers the setting with single player on one side and multiple arms on the other side. At each round, the player selects an arm and receives a reward. The player has unknown preferences over arms and would learn this knowledge based on the collected rewards. To accumulate as many rewards as possible, the player faces the dilemma of exploration and exploitation. The former selects arms with less observations while the latter focuses on arms with better historical performances. How to balance the exploration and exploitation trade-off is the key of bandit algorithm design. The upper confidence bound (UCB) [3], Thompson sampling (TS) [14, 2], and explore-then-commit (ETC) [10] are common strategies in MAB to achieve this objective.

Liu et al. [20] introduce the bandit learning problem in matching markets and try to provide theoretical guarantees. Two sides of agents in the market can be modeled as players and arms. Without loss of generality, denote N and K as the number of players and arms, respectively. It is worth noting that this work and all of the following works assume $N \leq K$ to ensure each player has a chance to be matched. In this problem, the objective is to find a stable matching and minimize the stable regret for each player, which is defined as the difference between the reward of the stable arm and that the player receives during the horizon. Since there may be more than one stable matching, they mainly focus on the players’ most preferred one corresponding to the player-optimal stable matching and the least preferred one corresponding to the player-pessimal stable matching. Note that players receive more rewards in the player-optimal stable matching and thus the former objective is the most desirable. Liu et al. [20] first study a centralized setting where a central platform would compute allocations for players to avoid conflicts. Both ETC and UCB-type algorithms are proposed for this setting. The former achieves a player-optimal stable regret guarantee with prior knowledge of players’ minimum preference gap ∆ and the latter can only ensure to reach the playerpessimal stable matching. Motivated by real applications where the central platform may not always exist, a rich line of works then study the decentralized case where no platform coordinates players behavior [21, 28, 4, 18, 22]. This line of works again only achieve guarantees for player-pessimal stable regret [21, 18, 28, 4, 22]. Table 1 compares settings and regrets among these works. Until recently, Zhang et al. [31] and Kong and Li [16] independently derive algorithms that have polynomial player-optimal stable regret and show the upper bound is $\dot { O } ( K$ log $\bar { T / \Delta ^ { 2 } } )$ . However, this result may be still far from the lower bound Ω(max{N log $T / \Delta ^ { 2 }$ , K log $T / \Delta \mathfrak { j } )$ [28] since K is usually much larger than N such as that the number of workers (publisher slots) is usually much larger than that of employers in labor markets (advertisers in online advertising, respectively).

In this paper, we try to provide more efficient algorithms and improve the results over existing works. The detailed contribution can be summarized as follows: (1) We propose an algorithm named adaptively explore-then-Gale-Shapley (AETGS) with elimination. State-of-the-art works [31, 16] explicitly separate the exploration and exploitation processes, which can lead to unnecessary regret, as exploring certain preference rankings may not contribute to the exploitation process. To avoid excessive exploration, our AETGS with elimination algorithm integrates the players’ learning process into the GS steps. Players adaptively switch between exploration and exploitation and promptly eliminate sub-optimal arms. (2) We prove that the player-optimal stable regret of AETGS with elimination can be upper bounded by $O ( \dot { N } ^ { 2 } \log T / \Delta ^ { 2 } + \dot { K } \log \dot { T } / \Delta )$ . This is the first result that removes the dependence on K in the main regret order term and improves existing works in common cases where N is much smaller than K. We also conduct experiments to show the advantages of the algorithm. (3) We refine the analysis of the centralized UCB algorithm in Liu et al. [20] for markets satisfying the α-condition. By investigating the preference hierarchy structure of the α-condition, we demonstrate that the stable matching converges sequentially from player 1 to player N. Through inductive analysis over players, we establish an $\mathrm { \bar { \it O } } ( N _  \mathrm  \bar  \it$ log $T / \bar { \Delta ^ { 2 } } + K$ log $T / \Delta )$ ) regret upper bound, which improves the original result for this algorithm in this specific market.

## 2 Related Work

The problem of bandit learning in matching markets is first introduced by Das and Kamenica [8]. They study the special case where both sides of agents have the same preferences and propose some empirical methods to solve the problem. Liu et al. [20] first theoretically formulate this problem and provide an upper bound for the stable regret of players. They propose a centralized explore-then-commit (ETC) algorithm and upper confidence bound (UCB) algorithm, which obtain an $O \left( K \log T / \Delta ^ { 2 } \right)$ player-optimal stable regret and $O \left( N K \log T / \Delta ^ { 2 } \right)$ player-pessimal stable regret, respectively. It is worth noting that the former ETC algorithm requires knowledge about ∆ to ensure the algorithmic operation. Due to the generality, the following works focus on the decentralized setting. Liu et al. [21] and Kong et al. [18] propose the UCB and TS-type algorithm for general decentralized markets, respectively. Such a setting is much more challenging and both of them only achieve O $( \exp ( N ^ { 4 } ) N ^ { 5 } \bar { K ^ { 2 } } \log ^ { 2 } ( \bar { T } ) / \Delta ^ { 2 } )$ upper bound for the player-pessimal stable regret.

<table><tr><td colspan="2">Regret bound</td><td>Setting</td></tr><tr><td>Liu et al. [20]</td><td> $O \left( K \log T / \Delta ^ { 2 } \right) * \#$   $O \left( N K \log T / \bar { \Delta } ^ { 2 } \right) \#$ </td><td>known ∆, gap1  $\mathrm { g a p } _ { 2 }$ </td></tr><tr><td>Liu et al. [21]</td><td> $O \left( { \frac { N ^ { 5 } K ^ { 2 } \log ^ { 2 } T } { \varepsilon ^ { N ^ { 4 } } \Delta ^ { 2 } } } \right)$ </td><td> $\mathrm { g a p } _ { 2 }$ </td></tr><tr><td>Sankararaman et al. [28]</td><td> $O \left( N K \log T / \Delta ^ { 2 } \right)$   $\Omega \left( \operatorname* { m a x } \left\{ \breve { N } \log T / \Delta ^ { 2 } , K \log T / \Delta \right\} \right)$ </td><td>serial dictatorship, gap1</td></tr><tr><td>Basu et al. [4]</td><td> $O \left( K \log ^ { 1 + \varepsilon } T + 2 ^ { ( \frac { 1 } { \Delta ^ { 2 } } ) ^ { \frac { 1 } { \varepsilon } } } \right) *$   $O \left( \dot { N } K \log T / \Delta ^ { 2 } \right)$ </td><td> $\mathrm { g a p } _ { 2 }$   $\underline { { \alpha \mathrm { - c o n d i t i o n , g a p _ { 1 } } } }$ </td></tr><tr><td>Maheshwari et al. [22]</td><td> $O \left( C N K \log T / \Delta ^ { 2 } \right)$ </td><td>α-reducible condition, gap1</td></tr><tr><td>Kong et al. [18]</td><td> $O \left( { \frac { N ^ { 5 } K ^ { 2 } \log ^ { 2 } T } { \varepsilon ^ { N ^ { 4 } } \Delta ^ { 2 } } } \right)$ </td><td> $\mathrm { g a p } _ { 2 }$ </td></tr><tr><td>Zhang et al. [31]</td><td> $O \left( K \log T / \Delta ^ { 2 } \right) *$ </td><td> $\mathrm { g a p } _ { 2 }$ </td></tr><tr><td>Kong and Li [16]</td><td> $\underline { { O \left( K \log T / \Delta ^ { 2 } \right) } }$  *</td><td> $\mathrm { g a p _ { 3 } }$ </td></tr><tr><td>Ours</td><td> $O \left( N ^ { 2 } \log T / \Delta ^ { 2 } + K \log T / \Delta \right) *$   $O \left( N \log T / \Delta ^ { 2 } + K \log T / \Delta \right) \stackrel { \cdot } { \# }$ </td><td> $\mathrm { g a p _ { 4 } }$   $\alpha { \mathrm { - c o n d i t i o n , g a p _ { 3 } } }$ </td></tr></table>

Table 1: Comparisons of settings and regret bounds with most related works, ∗ represents the playeroptimal stable regret and bounds without labeling ∗ are for player-pessimal stable regret, # represents the centralized setting. N and K are the number of players and arms with $N \leq K , T$ is the total horizon, $\Delta$ corresponds to some preference gap, ε depends on the hyper-parameter of algorithms, and $C$ is related to the unique stable matching condition which can grow exponentially in $N .$ . The definition of $\Delta$ in different works requires particular care. We use $\mathrm { g a p _ { 1 } , g a p _ { 2 } , g a p _ { 3 } , g a p _ { 4 } }$ represent the minimum preference gap between the (player-optimal) stable arm and the next arm after the stable arm in the preference ranking among all players, the minimum preference gap between any different arms among all players, the minimum preference gap between the first $\bar { N + 1 }$ ranked arms among all players, and the minimum preference gap between arms that are more preferred than the next of the player-optimal stable arm among all players, respectively. Based on the property that the player-optimal stable arm of each player must be its first N-ranked (shown in Appendix), there would be gap $\geq \mathrm { g a p _ { 4 } } \geq \mathrm { g a p _ { 3 } } \geq \mathrm { g a p _ { 2 } }$ . So our dependence on $\Delta$ is better than the state-of-the-art works [31, 16] for general markets.

To improve the stable regret guarantee, a line of research studies some special markets with unique stable matching in which case the player-optimal stable matching is equivalent to the playerpessimal one. Sankararaman et al. [28] propose the UCB-D3 algorithm based on the assumption of serial dictatorship, i.e., all arms share the same preferences, and obtain an $O \left( N K \log T / \Delta ^ { 2 } \right)$ regret upper bound. To investigate the problem hardness, they also derive a lower bound Ω  max N log $T / \Delta ^ { 2 }$ , K log $T / \Delta \mathbf { \check { \beta } } )$  under this assumption. Basu et al. [4] consider more general α-condition setting for unique stable matching. They propose the UCB-D4 algorithm and also achieve the O $\left( N K \log T / \Delta ^ { 2 } \right)$ regret bound. Later, Maheshwari et al. [22] study the market satisfying α-reducible condition and proposes a communication-free algorithm. Their regret bound has an exponential dependence on the number of market participants. Recently, researchers have developed algorithms that can achieve player-optimal stable regret guarantees without assuming unique stable matching. Both Zhang et al. [31] and Kong and Li [16] propose ETC-type algorithms that achieve $\mathcal { O } \left( K \log T / \Delta ^ { 2 } \right)$ player-optimal stable regret. Wang and Li [29] studies the matching markets with serial dictatorship and obtains the $O \left( N \log T / \Delta ^ { 2 } + K \log T / \Delta \right)$ regret. Table 1, compares our proposed algorithm with these related works in terms of their corresponding settings and theoretical guarantees.

There are also other works considering unknown preferences in matching markets. Wang et al. [30] study a many-to-one market where an arm can accept multiple players. Jagadeesan et al. [12] consider online matching markets with monetary transfers. Min et al. [23] investigate Markov matching markets where state transitions occur during the matching process and players’ rewards depend on the current state. Other studies have focused on non-stationary rewards, such as Muthirayan et al. [24], Ghosh et al. [11], who propose robust algorithms to mitigate the impact of reward disturbances. Additionally, several studies have explored the problem of offline matching market learning. Dai and Jordan [6, 7] propose approaches that leverage historical data to design optimal matching or recommend participants on both sides.

## 3 Setting

This paper considers the problem of bandit learning in two-sided matching markets. Denote ${ \mathcal { N } } =$ $\left\{ p _ { 1 } , p _ { 2 } , \dotsc , p _ { N } \right\}$ as the player set and ${ \mathcal { K } } = \{ a _ { 1 } , a _ { 2 } , \ldots , a _ { K } \}$ as the arm set. Let N and K be the number of players and arms, respectively. To ensure that each player can be matched with an arm, we follow previous works and assume $\dot { N } \le K [ 2 0 , 2 1 , 2 8 , 4 , 1 \dot { 8 } , \dot { 3 } 1 , 1 6 ]$

For each player $p _ { i } ~ \in \mathcal N .$ , its preference towards arm $a _ { j }$ can be portrayed by an absolute utility $\mu _ { i , j } \in ( 0 , 1 ]$ . For any pair of arms $a _ { j }$ and $a _ { j ^ { \prime } } , \mu _ { i , j } > \mu _ { i , j ^ { \prime } }$ indicates that player $p _ { i }$ prefers arm $a _ { j }$ over $a _ { j ^ { \prime } }$ . Following previous works for matching markets [9, 20, 21, 28, 4, 18, 31, 16], players are assumed to have distinct preferences over different arms, i.e., $\mu _ { i , j } \neq \mu _ { i , j ^ { \prime } }$ for any $a _ { j } \neq a _ { j \prime }$ . In practice, players’ preferences which correspond to workers’ abilities and the publisher’s conversion rates are typically unknown and can be learned through the interactive matching process. On the other side, each arm $a _ { j }$ also has a fixed and distinct preference utility $\pi _ { j , i }$ over each player $p _ { i } \in \mathcal N$ and $\pi _ { j , i } > \pi _ { j , i ^ { \prime } }$ means that arm $a _ { j }$ prefers player $p _ { i }$ over $p _ { i ^ { \prime } }$ . As in labor markets where workers usually know their preferences over employers based on the payments and task types, the preferences of arms are assumed to be known beforehand [20, 21, 18, 28, 4, 31, 16].

At each round $t = 1 , 2 , \dots$ , each player $p _ { i }$ proposes to an arm $A _ { i } ( t )$ . For each arm a , denote $a _ { j }$ $A _ { j } ^ { - 1 } ( t ) \ = \ \{ p _ { i } : A _ { i } ( t ) = a _ { j } \}$ as the set of players who selects arm $a _ { j }$ at round $t .$ . When more than one player selects $a _ { j }$ , it accepts its most-preferred one in $A _ { j } ^ { - 1 } ( t )$ , i.e. $a _ { j }$ will match with $p _ { i } \in \arg \operatorname* { m a x } _ { p _ { i } \in A _ { i } ^ { - 1 } ( t ) } \pi _ { j , i }$ . If a player $p _ { i }$ is successfully matched with arm $A _ { i } ( t )$ , it will receive a random reward $X _ { i } ( t )$ characterizing its matching experience, which we assume is a 1-subgaussian random variable with expectation $\mu _ { i , A _ { i } ( t ) }$ . Otherwise, $p _ { i }$ is rejected by its proposed arm and only gets reward $X _ { i } ( t ) ~ = ~ 0$ . Denote $\bar { A } _ { i } ( t )$ as the final matched arm of player $p _ { i }$ at round t. Then $\mathbf { \bar { \boldsymbol { A } } } _ { i } ( t ) = \mathbf { \boldsymbol { A } } _ { i } ( t ) \mathbf { \boldsymbol { i } } \mathbf { \boldsymbol { f } } _ { p _ { i } }$ is accepted by the arm $A _ { i } ( t )$ and we simply set $\bar { A } _ { i } ( \bar { t } ) \dot { = } \varnothing \mathrm { i f } p _ { i }$ is rejected.

Stability is a key property of a matching in two-sided markets [9, 27, 25]. A matching ${ \bar { A } } ( t ) =$ $\{ ( i , \bar { A } _ { i } \dot { ( } t ) ) : i \stackrel { \cdot } { \in } \top N \} \}$ is stable if no market participant wants to break up its current matching relationship and find a new partner. Formally speaking, there is no player-arm pair $( p _ { i } , a _ { j } )$ such that $\mu _ { i , j } ~ > ~ \mu _ { i , \bar { A } _ { i } ( t ) }$ and $\pi _ { j , i } ~ > ~ \pi _ { j , \bar { A } _ { i } ^ { - 1 } ( t ) }$ It is worth noting that there may be multiple stable matchings in the market. Denoted $M = \{ m : m$ is stable} as the set of all stable matchings. It is shown that there exists a stable matching $\dot { m } ^ { * } \in M$ such that all players are matched with their most preferred stable arm [9], $\mathrm { i . e . , } \ \mu _ { i , m _ { i } ^ { * } } \ \geq \ \mu _ { i , m _ { i } }$ for any m $\in M , i \in [ N ]$ . Given a specified horizon $\bar { T }$ , the learning objective is to minimize the player-optimal stable regret for each player $p _ { i }$ which is defined as the difference between the cumulative reward received by being matched with $m _ { i } ^ { * }$ and the cumulative reward received by $p _ { i }$ over $T$ rounds:

$$
R e g _ { i } ( T ) = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \left( \mu _ { i , m _ { i } ^ { * } } - X _ { i } ( t ) \right) \right] .
$$

Here, the expectation is taken over by the randomness of the reward generation and the randomness inherent in the player’s strategy.

For completeness, we introduce the procedure of the offline Gale-Shapley (GS) algorithm, which would be useful when describing the algorithmic details. The offline GS algorithm is a classic algorithm to find the player-optimal stable matching when both sides of the market participants know their exact preference rankings. Following offline GS, each player proposes to the arm one by one based on its preference ranking. Until no rejection happens, the final matching is exactly the player-optimal stable matching [9]. Specifically, at the first step, all players propose to their most preferred arm. Arms would accept their most preferred player among those who propose to it and reject others. Then players who are rejected at previous steps would then propose to their next preferred arm. And arms still reject the players who propose to it except for their most preferred one. Such a process continues until no rejection happens.

For convenience, we also define some useful notations that quantify the hardness of the learning problem in matching markets and are used in the later analysis.

Definition 3.1. For each player $p _ { i } .$ , denote $\sigma _ { i }$ as $p _ { i } { ' } \mathrm { s }$ preference ranking and let $\sigma _ { i , k }$ as $p _ { i } { ^ { \star } } { \bf s }$ the k-th preferred arm in its ranking. With a little abuse of notation, let $\sigma _ { i } ( a _ { j } )$ represent the rank of arm $a _ { j }$ in $p _ { i } { ' } \mathrm { s }$ preference. For each player $p _ { i }$ and arm $a _ { j } \neq a _ { j ^ { \prime } }$ , let $\Delta _ { i , j , j ^ { \prime } } = | \mu _ { i , j } - \mu _ { i , j ^ { \prime } } |$ be the preference gap of $p _ { i }$ between $a _ { j }$ and $a _ { j ^ { \prime } }$ . Define $\begin{array} { r } { \Delta = \operatorname* { m i n } _ { i , k \in [ \sigma _ { i } ( m _ { i } ^ { * } ) ] } \Delta _ { i , \sigma _ { i , k } , \sigma _ { i , k + 1 } } } \end{array}$ as the minimum preference gap between the arm ranked the first $( \sigma _ { i } ( m _ { i } ^ { * } ) + 1 )$ -th among all players. Further, define $\begin{array} { r } { \Delta _ { N } = \operatorname* { m i n } _ { i , k \in [ N ] } \Delta _ { i , \sigma _ { i , k } , \sigma _ { i , k + 1 } } } \end{array}$ as the minimum preference gap between the arm ranked the first $( N + 1 )$ )-th among all players.

## 4 Algorithm for General Markets

In this section, we propose an algorithm called adaptively explore-then-Gale-Shapley (AETGS) with elimination. For simplicity, we present the centralized version of the algorithm in Algorithm 1 from view of player $p _ { i }$ . The discussion on how to extend it to a decentralized version is deferred to later subsections.

In general, AETGS with elimination is an adaptive version of the GS algorithm. Since players do not know their preference rankings, they need to learn this knowledge by exploring arms (Line 4). To reduce the regret during exploration, players would adaptively eliminate sub-optimal arms (Line 6-8). Until they find their most preferred arm among available arms, they will stop exploration and focus on this arm (Line 9-11). And once the player finds this arm is occupied by a more preferred player, it would re-start exploration to find the next preferred arm (Line 12-19).

Specifically, each player $p _ { i }$ still maintains $\hat { \mu } _ { i , j } ( t )$ and $T _ { i , j } ( t )$ to represent the empirical mean and the number of observations on each arm $a _ { j }$ at the end of round t. To determine whether an arm is more preferred than another, it maintains a confidence interval for each arm $a _ { j }$ with upper bound $\mathrm { U C B } _ { i , j } ( t ) : = \hat { \mu } _ { i , j } ( t ) + \sqrt { 6 \log T / T _ { i , j } ( t ) }$ and lower bound $\mathrm { L C B } _ { i , j } ( t ) : = \hat { \mu } _ { i , j } - \sqrt { 6 \log T / T _ { i , j } ( t ) }$ When $T _ { i , j } = 0 { \mathrm { : } }$ , they will be initialized as +∞ and $- \infty$ , respectively. And once the confidence intervals of the two arms are disjoint, it can regard the arm with a higher empirical mean to be more preferred (Line 1). To be consistent with the offline GS, each player maintains $\mathcal { D } _ { i }$ to represent the set of arms that have rejected $p _ { i }$ during previous steps. In the beginning, it is initialized as an empty set. And we use $A _ { i }$ to represent the available arms with the potential to be the stable arm of $p _ { i } ,$ which is initialized as $\kappa \bar { \langle \mathcal { D } _ { i } }$ . For convenience, denote $\mathrm { E } _ { i }$ as the exploration status of player $p _ { i }$ $\mathrm { E } _ { i } = ^ { \prime }$ True means that $p _ { i }$ still needs to explore arms in $A _ { i }$ to determine its most preferred arm. And E<sub>i</sub> = False means that $p _ { i }$ already finds its most preferred arm and now focuses on this arm (Line 2).

To reduce the regret suffered during exploration, players would update $A _ { i }$ and eliminate sub-optimal arms in real-time (Line 6-8). Here to avoid collision during round-robin exploration, we would maintain $\mathbf { \mathcal { A } } _ { i }$ such that it contains no less than $N$ arms if $p _ { i }$ still has not determined its most preferred one. Thus the union of the available arm set over all players with $\mathrm { E } _ { i } =$ True contains more than $N$ arms and the round-robin exploration over $A _ { i }$ for each such player $p _ { i }$ can be carried out without collisions. For completeness, we defer how to arrange players’ explorations in later discussions. And once there exists an arm in $\mathbf { \mathcal { A } } _ { i }$ that can be regarded to be optimal, $p _ { i }$ will set the exploration status $\mathrm { E } _ { i }$ to be False and update the exploration arm set $\mathbf { \mathcal { A } } _ { i }$ to only contain this optimal arm. For convenience, denote $A _ { i }$ as this arm (Line 9-11).

Algorithm 1 adaptively explore-then-Gale-Shapley with elimination (from the view of player $p _ { i } )$   
Input: $\overline { { N , K , T } } .$   
1: Initialize: $\begin{array} { r } { \hat { \mu } _ { i , j } ( 0 ) = 0 , T _ { i , j } ( 0 ) = 0 , \mathrm { U C B } _ { i , j } ( 0 ) = \infty , \mathrm { L C B } _ { i , j } ( 0 ) = - \infty , \forall j \in [ K ] ; } \end{array}$ ;   
2: Initialize: $\mathcal { D } _ { i } = \emptyset , \mathcal { A } _ { i } = \tilde { \mathcal { K } } , \mathrm { E } _ { i } = \mathrm { T r u e } ;$   
3: for round $t = 1 , 2 , \cdots$ $\mathbf { \nabla } _ { T }$ do   
4: Select $A _ { i } ( t ) \in { \mathcal { A } } _ { i }$ in a round-robin manner;   
5: Update $\hat { \mu } _ { i , j } ( t ) , T _ { i , j } ( t )$ as Algorithm 2; compute $\mathrm { U C B } _ { i , j } ( t ) , \mathrm { L C B } _ { i , j } ( t )$ for any $j \in [ K ]$   
6: $\mathbf { i f } \left| \mathcal { A } _ { i } \right| > N$ and $\begin{array} { r } { \exists j \in \mathcal { A } _ { i } \mathrm { ~ s . t . ~ } \mathrm { U C B } _ { i , j } ( t ) < \operatorname* { i n a x } _ { j ^ { \prime } \in \mathcal { A } _ { i } } \mathrm { L C B } _ { i , j ^ { \prime } } ( t ) } \end{array}$ then   
7: ${ \dot { \mathcal { A } } } _ { i } { \dot { = } } { \mathcal { A } } _ { i } \backslash \{ j \} ;$   
8: end if   
9: j $\mathbf { f } \exists j \in { \mathcal { A } } _ { i } { \mathrm { ~ s . t . ~ L C B } } _ { i , j } ( t ) > { \mathrm { U C B } } _ { i , j ^ { \prime } } ( t )$ for any $j ^ { \prime } \in \mathcal A _ { i }$ and $j ^ { \prime } \ne j$ then   
10: $\begin{array} { r } { \bar { \mathcal { A } } _ { i } = \{ j \} , \mathrm { E } _ { i } = \bar { \mathrm { F a l s e } } , A _ { i } = j ; } \end{array}$   
11: end if   
12: for other player $p _ { i ^ { \prime } }$ with $\mathrm { E } _ { i ^ { \prime } } = \mathrm { F a l s e } , A _ { i ^ { \prime } } \in \mathcal { A } _ { i }$ do   
13: if $\pi _ { A _ { i ^ { \prime } } , i ^ { \prime } } > \pi _ { A _ { i ^ { \prime } } , i }$ then   
14: $\mathcal { D } _ { i } ^ { ' } = \mathcal { D } _ { i } \cup \{ A _ { i ^ { \prime } } \} , A _ { i } = \mathcal { K } \backslash \mathcal { D } _ { i } ;$   
15: if $\mathrm { E } _ { i } = \mathrm { F a l s e }$ and $A _ { i } \in { \mathcal { D } } _ { i }$ then   
16: $\mathrm { E } _ { i } = \mathrm { T r u e } ;$   
17: end if   
18: end $\mathbf { i f }$   
19: end for   
20: end for

The update of the available arm set should not only depend on $p _ { i } { ' } \mathrm { s }$ own observations but also on the other market participants. Specifically, if a player $p _ { i ^ { \prime } }$ determines $A _ { i ^ { \prime } }$ as its most preferred arm, then the final stable player of $A _ { i ^ { \prime } }$ would be the same as or more preferred than $p _ { i ^ { \prime } }$ . So if $A _ { i ^ { \prime } }$ prefers $p _ { i ^ { \prime } }$ than $p _ { i } .$ , then $A _ { i ^ { \prime } }$ would not be the stable arm of $p _ { i }$ and there is no need for $p _ { i }$ to explore $A _ { i ^ { \prime } }$ anymore. In this case, $p _ { i }$ deletes arm $A _ { i ^ { \prime } }$ from its available set and update $A _ { i }$ (Line 12-19). It is worth noting that this operation may incorporate the eliminated arms again in $A _ { i } .$ . This is reasonable as the previously eliminated arm may be more preferred than the current arms in $A _ { i }$ after the deletion operation. And if the deleted arm is $p _ { i } { ' } \mathrm { s }$ current most preferred arm, it will mark $\mathrm { E } _ { i }$ as True and restart exploration to find the next most preferred one (Line 15-17).

## 4.1 Theoretical Results.

Theorem 4.1. Following Algorithm 1, the player-optimal stable regret for each player $p _ { i }$ satisfies

$$
R e g _ { i } ( T ) \leq O \left( N ^ { 2 } \log T / \Delta ^ { 2 } + K \log T / \Delta \right) .
$$

Due to the space limit, the proof of Theorem 4.1 is deferred to Appendix $\mathrm { A } .$ The following are discussions on the detailed implementation as well as the novelty of the result.

Arrangement of the round-robin exploration process. Recall that the number of available arms of each player is always larger than N based on Line 6 and we assume players can explore their available arms in a round-robin manner without conflict. We now propose an arrangement by letting players explore the available arms in units of N to guarantee this property. Specifically, in every 2N rounds, each player selects the N available arms with the fewest observations (randomly breaks ties) and explores them in a round-robin way. It can be shown by contradiction that there exists an assignment such that each player can successfully match with their respective N arms once during these 2N rounds (Lemma B.2 in Appendix), which only doubles the original regret without influencing the regret order. This guarantees that after every 2N rounds, the observation count difference among all available arms is at most 1. Players would perform arm elimination and optimal arm identification (Line 6 and 9) in the end of each 2N rounds. Therefore, compared to the timely eliminating/deleting of arms, this approach ensures that each player will select each arm at most one additional time during each exploration cycle before the player finds the optimal one. Since each player may restart exploration (Line 12-13) up to $N ^ { 2 }$ times (each of $N$ players can focus on N arms), this scenario leads to an additional $O ( N ^ { 2 } K )$ constant regret and does not influence the final regret order.

Extension to the decentralized setting. For simplicity, we present the algorithm in a centralized manner. It can also be extended to the decentralized version where no central platform coordinates players’ selections. Specifically, we divide the total horizon into several phases and the length of each grows exponentially, i.e., the lengths of phases are $2 , 4 , 8 , \cdots \mathrm { A t }$ the end round of each phase, players would decide whether to eliminate sub-optimal arms as Line 6-8, whether to determine one arm as the most preferred one and update the exploration status as Line 9-11. After the end of each phase, players would communicate their current exploration status, update their deletion set and available arm set, re-update the exploration status as Line 12-19, and then communicate their updated available arm set to determine the round-robin exploration process in the next phase. If there exists a player whose exploration status becomes True from False during communication, the phase length would re-start from 2 and grows exponentially since new arms may be required for exploration. If the final exploration status of all players is False and their optimal arms are different, the next phase would continues until the end of the interaction. The detailed implementation of communication is deferred to the next paragraph. Based on the communication, players with $\mathrm { E } _ { i } =$ True can have a pre-agreed protocol to explore arms in their available arm set in a round-robin manner without collision as discussed in the last paragraph. Within each phase, they just round robin explore arms and collect observations but do not make any decision on arms’ optimality. If L observations on a sub-optimal arm are enough to decide its sub-optimality in the centralized version, then this arm would be eliminated at the end of the corresponding phase with the selected time to be at most 2L due to the exponentially increasing phase length. So the regret in this decentralized version is at most two times as that suffered in the centralized version.

This paragraph describes the implementation of the communication procedure. Recall that players need to communicate their exploration status and available arm set (calculated by subtracting the deletion and eliminating set from K) at the end of each phase. For the phase length, recall that it grows exponentially until a player’s exploration status becomes True from False and a player updates $\mathrm { E } _ { i }$ from False as True only when its most preferred arm is occupied by a higher-priority player (Line 15). As shown by Lemma A.3, each player may occupy N arms, so such event happens at most $N ^ { 2 }$ times. And when all players find their unique optimal arm which requires $O ( \dot { N } ^ { 2 } \log T / \Delta ^ { 2 } )$ times, the phase would continue until the end of the interaction. Above all, the total number of phases is of order $O \left( N ^ { 2 } \log \left( N ^ { 2 } \log T / \Delta ^ { 2 } \right) \right)$ . For the detailed communication procedure, as phase 1 in Kong and Li [16], players can first estimate their unique indices and we assume the matching results are public as [16, 18, 21]. During the communication block of each phase, players sequentially transmit their data based on their indices, received by others through matching outcomes. Specifically, in the corresponding round, player p selects the focused arm if $\mathrm { E } _ { i } = \mathrm { F a l s e }$ and nothing otherwise, incurring an $O \left( \breve { N ^ { 3 } } \log \left( \breve { N ^ { 2 } } \log \bar { T / } \Delta ^ { 2 } \right) \right)$ cost for status communication in all phases. For deletion (eliminating) sets, $p _ { i }$ first selects the arm with index $k$ to indicate it will transmit k arms and then sequentially selects these $k$ arms. The communication cost on the arm set size is $O \left( N ^ { 3 } \log \left( N ^ { 2 } \log T / \Delta ^ { 2 } \right) \right)$ . Recall that players delete arms only when a higher-priority player fo cuses on this arm, so $N$ players focus on at most N arms before reaching stability and each player deletes up to N arms. Also, each player can eliminate up to $K - N$ arms during each exploration and would re-start exploration for at most $N ^ { 2 }$ times. Thus the communication cost on the deletion (elim inating) arms is ${ \dot { O } } ( N ^ { 3 } K )$ ) and the total communication cost is ${ \cal O } \left( N ^ { 3 } \log \left( N ^ { 2 } \log T / \Delta ^ { 2 } \right) + \dot { N ^ { 3 } } K \right)$ which is not the main order of the regret.

Key idea of removing the dependence on K. Balancing the exploration-exploitation trade-off is the key to achieving low regret. Previous efforts were devoted to addressing pessimal stable regret [20, 21, 18] and uniqueness assumptions [28, 4] using classic UCB and TS strategies. Until recently, Zhang et al. [31] and Kong and Li [16] show that ETC-type strategies better fit this problem. Specifically, players first uniformly explore arms to learn the complete preference ranking of the top N arms, and then use the GS procedure for exploitation to find the player-optimal stable matching. However, such a method may over-explore and cause unnecessary regret. The reason is that to learn the first N-ranked arms, each sub-optimal arm $a _ { j }$ must be selected $O ( \log T / \Delta _ { i , \sigma _ { i , N } , j } ^ { 2 } )$ times to be distinguished from the N-ranked arm. And each time selecting this arm, the player pays $\Delta _ { i , m _ { i } ^ { * } , j }$ regret. The mismatch between the paid regret and the difference to be figured out results in $O ( K \log T / \Delta _ { N } ^ { 2 } )$ regret.

In contrast to the existing approach, we present a more adaptive perspective that integrates the learning process into each GS step. To avoid additional regret, players do not need to estimate their complete preference. Instead, they would start exploitation once the optimal available arm is identified. And if this arm is occupied by a higher-priority player, this player would restart exploration to find the next preferred one. To avoid the additional cost while exploring the optimal arm, we design a more efficient way to let players promptly eliminate $K - N$ sub-optimal arms and only maintain the remaining $N$ arms to guarantee no collision. For the eliminated arm $a _ { j }$ , the reward difference to be figured out is at least $\Delta _ { i , m _ { i } ^ { * } , j }$ , which matches the regret when selecting this arm. So the regret caused by these eliminated arms is $O ( K \log T / \Delta )$ , avoiding dependence on $K$ in the main order.

Recall Kong and Li [17] propose an adaptively-explore-then-deferred-acceptance (AETDA) algorithm for more general many-to-one markets with responsiveness. Compared with their realization in the one-to-one setting, our algorithm shares the same idea of balancing exploration and exploitation but further introduces the elimination operation to avoid unnecessary selections. Compared with their $O ( N ^ { 2 } K \log T / \Delta ^ { 2 } )$ result in the reduced one-to-one setting, our Theorem 4.1 removes the dependence on K in the main order.

Discussion on the definition of gaps. Recall that our $\Delta$ is defined as the minimum preference difference among arms that ranked in the first $( \sigma _ { i } ( m _ { i } ^ { * } ) + 1 )$ )-th positions $( \mathrm { g a p } _ { 4 } )$ , while the lower bound in Sankararaman et al. [28] for markets with serial dictatorship depends on $\Delta$ that is defined as the minimum preference difference between the arm ranked $\sigma _ { i } ( m _ { i } ^ { * } )$ and the arm ranked $\sigma _ { i } ( m _ { i } ^ { * } ) + 1$ $( \mathrm { g a p } _ { 1 }$ in Table 1, respectively). It is an open problem whether the lower bound should depend on our $\mathrm { g a p _ { 4 } }$ in general markets. Here we would like to discuss that the knowledge of gap is important to learn the true stable matching. Consider a market with 4 players and 5 arms. The preference rankings of players are $p _ { 1 } : a _ { 1 } > a _ { 2 } > a _ { 3 } > a _ { 4 } > a _ { 5 } ; p _ { 2 } : a _ { 2 } > a _ { 3 } > a _ { 1 } > a _ { 4 } > a _ { 5 } ; p _ { 3 }$ $a _ { 3 } > a _ { 1 } > a _ { 2 } > a _ { 4 } > a _ { 5 } ; p _ { 4 } : a _ { 1 } > a _ { 2 } > a _ { 3 } > a _ { 4 } > a _ { 5 }$ and the preference rankings of arms are $a _ { 1 } : p _ { 2 } > p _ { 3 } > p _ { 4 } > p _ { 1 } ; a _ { 2 } : p _ { 3 } > p _ { 4 } > p _ { 1 } > p _ { 2 } ; a _ { 3 } : p _ { 4 } > p _ { 1 } > p _ { 2 } > p _ { 3 } ; a _ { 4 } : p _ { 1 } > p _ { 2 } > p _ { 3 } ; a _ { 4 } : p _ { 1 } > p _ { 2 } : a _ { 3 } : p _ { 2 } .$ $p _ { 2 } > p _ { 3 } > p _ { 4 } ; a _ { 5 } : p _ { 1 } > p _ { 2 } > p _ { 3 } > p _ { 4 }$ In this market, the player-optimal stable matching is $\{ ( p _ { 1 } , a _ { 4 } ) , ( p _ { 2 } , a _ { 1 } ) , ( p _ { 3 } , a _ { 2 } ) , ( p _ { 4 } , a _ { 3 } ) \}$ . However, if player $p _ { 1 }$ has collected enough observations to identify gap but not collected enough observations to identify $\mathrm { g a p _ { 4 } }$ and wrongly estimate the first $\sigma _ { i } ( m _ { i } ^ { * } )$ ranked arms, $\mathrm { i . e . , } p _ { 1 }$ wrongly estimate the preference ranking as $p _ { 1 } : a _ { 1 } > a _ { 2 } >$ $a _ { 4 } > a _ { 3 } > a _ { 5 }$ . Then the computed player-optimal stable matching under this preference ranking is $\{ ( p _ { 1 } , a _ { 4 } ) , ( p _ { 2 } , a _ { 3 } ) , ( p _ { 3 } , a _ { 1 } ) , ( p _ { 4 } , a _ { 2 } ) \}$ , which is not stable in the original market as player $p _ { 1 }$ and arm $a _ { 3 }$ form a blocking pair. This example shows that player $p _ { 1 }$ must identify the gap among the first $\sigma _ { i } ( m _ { i } ^ { * } )$ ranked arms to find a stable matching in the market, which further illustrates the crucial role of $\mathrm { g a p _ { 4 } }$ in learning the true stable matching. We leave the lower bound in general markets as an important future direction.

## 5 Experiments

In this section, we compare our Algorithm 1 (abbreviated as AETGS-E) with baselines ETGS [16], ML-ETC [31] and Phased ETC [4] which also enjoy guarantees for player-optimal stable regret in general decentralized one-to-one markets. To better illustrate the advantages of our algorithm, especially when N is much smaller than $K ,$ , we set $N = 3$ and $K = 1 0$ . The preference rankings for both players and arms are generated as random permutations. The preference gap between any adjacent ranked arms is set as 0.1. The feedback $X _ { i , j } ( t )$ for player $p _ { i }$ on arm $a _ { j }$ at time t is drawn independently from the Gaussian distribution with mean $\mu _ { i , j }$ and variance 1. We report the maximum cumulative player-optimal stable regret among all players and the cumulative player-optimal instability in Figure 1 (a) and (b), respectively. Here the cumulative player-optimal unstability is defined as the number of matchings that are not the player-optimal stable one. All algorithms run for $T = 1 0 0 k$ rounds and all results are averaged over 50 independent runs. The error bars represent standard errors, which are computed as standard deviations divided by $\sqrt { 5 0 }$

As shown in the figure, our AETGS-E algorithm, which only conducts necessary explorations over unknown preferences and promptly eliminates sub-optimal arms, achieves the least cumulative regret and cumulative player-optimal unstability among all baselines. The ML-ETC and ETGS algorithms need to sufficiently explore K arms to estimate the full preference ranking, requiring more exploration time to find the player-optimal stable matching. The PhasedETC algorithm has not yet converged within the displayed rounds due to the cold start problem.

![](images/984393dceb1d0379ac41bb262300da2013216c07f475aeeaa5b11cda08d9877b.jpg)

![](images/efabd04df51665a2a09beb16bf72c9f5f3f3ea5c28b0abe8370c91a4b8ee25ce.jpg)  
Figure 1: Experimental comparisons of our AETGS-E with ETGS, ML-ETC and Phased ETC in one-to-one decentralized markets with N = 3 players and K = 10 arms.

## 6 Centralized UCB Algorithm for Markets with α-condition

In this section, we provide a new analysis for the centralized UCB algorithm in markets satisfying α-condition. The algorithm is first introduced by Liu et al. [20]. For completeness, we present the full algorithm in Algorithm 2. At each round, players submit their UCB rankings to the centralized platform (Line 3). The platform runs the GS algorithm (based on players’ submitted rankings) and returns the partner to each player (Line 4).

```latex
Algorithm 2 centralized UCB
Input: ${ \overline { { N , K } } } .$
1: Initialize: $\begin{array} { r } { \hat { \mu } _ { i , j } ( 0 ) = 0 , T _ { i , j } ( 0 ) = 0 , \mathrm { U C B } _ { i , j } ( 1 ) = \infty , \forall i \in [ N ] , j \in [ K ] . } \end{array}$
2: for round $t = 1 , 2 , \dots , T$ do
3: Receive rankings $\begin{array} { r l r } { \hat { \sigma } } & { { } \mathrel { \mathop : } = } & { \{ \hat { \sigma } _ { i } \} _ { i \in [ N ] } } \end{array}$ according to the decreasing order of
$\{ \mathrm { U C B } _ { i , j } ( t ) \} _ { j \in [ K ] } , \forall i \in [ N ] ;$
4: $A _ { i } ( t )$ ←Gale-Shapley $( \hat { \sigma } , \pi )$ for each player $p _ { i } ;$
5: Observe $X _ { i } ( t )$ , and update $\hat { \mu } _ { i , j } ( t ) , T _ { i , j } \mathrm { ( } t \mathrm { ) , U C B } _ { i , j } ( t + 1 )$ for each $p _ { i } , a _ { j } ;$
6: end for
```

In the following, we introduce the α-condition. Conditions guaranteeing the unique stable matching have been widely studied in the offline setting [5, 13] and also the online setting to improve the learning efficiency [28, 4, 22]. Among these conditions, the α-condition is shown to be the weakest sufficient one [13] and incorporate the conditions studied in existing works [28, 4, 22].

Let $\beta$ denote a pair of permutations of $[ N ]$ and $[ K ]$ . Then $[ N ] _ { \beta } = \{ Q _ { 1 } ^ { ( \beta ) } , . . . , Q _ { N } ^ { ( \beta ) } \}$ and $[ K ] _ { \beta } =$ $\{ q _ { 1 } ^ { ( \beta ) } , \ldots , q _ { K } ^ { ( \beta ) } \}$ denote permutations of the ordered sets [N] and $[ K ]$ , respectively. The j-th player in $[ N ] _ { \beta }$ is the $Q _ { j } ^ { ( \beta ) }$ -th player in $[ N ]$ , and the k-th arm in $[ K ] _ { \beta }$ is the $q _ { k } ^ { ( \beta ) } \ – \mathrm { t h }$ arm in $[ K ]$ . Then we can define the α-condition below.

Definition 6.1. The α-condition is satisfied if there is a stable matching $( \mathbf { j } ^ { * } , \mathbf { i } ^ { * } )$ , a left-order of players and arms s.t. $\forall i \ \in \ [ N ] _ { l } , \forall j \ > \ i , j \ \in \ [ K ] _ { l } : \ \mu _ { i , j _ { i } ^ { * } } \ > \ \mu _ { i , j }$ where $j _ { i } ^ { * }$ is the partner of player $p _ { i }$ in stable matching $( \mathbf { j } ^ { * } , \mathbf { i } ^ { * } )$ , and a (possibly different) right-order of players and arms s.t. $\forall j < i \le N , q _ { j } \in [ K ] _ { r } , Q _ { i } \in [ N ] _ { r } : \pi _ { q _ { j } , Q _ { i _ { a _ { i } } ^ { * } } } > \pi _ { q _ { j } , Q _ { i } }$ . Here similarly, $i _ { q _ { j } } ^ { * }$ is the partner of arm $a _ { q _ { j } }$ <sup>q</sup><sub>j</sub>   
in stable matching $( \mathbf { j } ^ { * } , \mathbf { i } ^ { * } )$

Without loss of generality, we consider the identity of players and arms is just the left order, i.e., $[ N ] = [ N ] _ { l }$ and $[ K ] = [ K ] _ { l }$ . Thus we only deal with player order $Q _ { i } ^ { ( r ) } = Q _ { i }$ and arm order $q _ { j } ^ { ( r ) } = q _ { j }$ , for $i \in [ N ] , j \in [ K ]$ in the rest of the paper. Under α-condition, it is easy to inductively verify that for any $i \in [ N ]$ , the player $p _ { i }$ is matched with arm $a _ { i } .$ and the player $p _ { Q _ { i } }$ is matched with the arm $\boldsymbol { a } _ { q _ { i } }$ in the unique stable matching [4].

## 6.1 Theoretical Results

We analyze the regret for the centralized UCB algorithm under α-condition.

Theorem 6.2. When preferences of participants satisfy α-condition, following Algorithm 2, the stable regretfor each player $p _ { i }$ satisfies

$$
R e g _ { i } ( T ) \le O \left( N \log T / \Delta _ { N } ^ { 2 } + K \log T / \Delta \right) .
$$

The centralized UCB algorithm is proposed by Liu et al. [20] and shown to have $O ( N K \log T / \Delta ^ { 2 } )$ player-pessimal stable regret for general markets. We provide a new analysis for markets satisfying α-condition which removes the dependence of $K$ in the regret. Due to the space limit, we discuss the key idea of the proof below and defer the detailed proof to Appendix C.

We investigate the preference structure of α-condition to obtain the improved analysis. For player $p _ { i } .$ , its regret is due to selecting sub-optimal arm $a _ { k }$ with $\mu _ { i , k } ~ < ~ \mu _ { i , i }$ . Arm $a _ { k }$ will be selected by $p _ { i }$ when its UCB value is higher than $p _ { i } { ' } s$ stable matched arm $a _ { i } .$ , which time is bounded by $\dot { O } ( \log T / \Delta _ { i , i , k } ^ { 2 } )$ , and one $\Delta _ { i , i , k }$ on the denominator can be eliminated when multiplying $\Delta _ { i , i , k }$ to compute regret. This contributes $O \left( K \log T / \Delta \right)$ regret since there are at most $K - 1$ sub-optimal arms. It is worth noting that arm $a _ { k }$ will also be selected by $p _ { i }$ if $p _ { i }$ is rejected by $a _ { i }$ in the GS algorithm. Recall that under α-condition, there is a right order $Q _ { i ^ { \prime } } = i \in [ N ]$ <sub>r</sub> for player $p _ { i }$ , such that $\forall i ^ { \prime \prime } > i ^ { \prime } , Q _ { i ^ { \prime \prime } } \in [ N ] _ { r } : \pi _ { q _ { i ^ { \prime } } , Q _ { i ^ { \prime } } } > \pi _ { q _ { i ^ { \prime } } , Q _ { i ^ { \prime \prime } } } :$ , which means arm $a _ { q _ { i ^ { \prime } } } = a _ { i }$ can only prefer players $p _ { Q _ { 1 } } , p _ { Q _ { 2 } } , \cdot \cdot \cdot , p _ { Q _ { i ^ { \prime } - 1 } }$ than player $p _ { Q _ { i ^ { \prime } } } = p _ { i }$ . Thus $p _ { i }$ is rejected by $a _ { i }$ only when these players select ${ { a } _ { i } } ,$ and $a _ { i }$ is sub-optimal for those players. To bound the regret of $p _ { i }$ when being rejected, we just need to bound the exploration times of these players $p _ { Q _ { 1 } } , p _ { Q _ { 2 } } , \cdots , p _ { Q _ { i ^ { \prime } - } }$ on arm $a _ { i } .$ However, the exploration time of a single player $p _ { Q }$ with $1 \leq \ell \leq i ^ { \prime } - 1$ on $a _ { i }$ can not be trivially bounded by ${ \cal O } ( \log T / \Delta _ { N } ^ { 2 } )$ since $p _ { Q }$ may have to select arm $a _ { i }$ after rejected by its stable arm $a _ { q \ell }$ in offline GS, where $a _ { q \ell }$ might be selected by $p _ { Q _ { 1 } } , \cdots , p _ { Q _ { \ell - 1 } }$ . This leads to a recursion form. We control this term using the fact that when a player is rejected by its stable matched arm in the GS, it can date back to a higher right-order player wrongly over-estimate its preference for a sub-optimal arm. This key observation and the definition of $\Delta$ make it possible to derive the final $O \left( N \log T / \Delta _ { N } ^ { 2 } \right)$ bound.

## 7 Conclusion

In this paper, we investigate the problem of whether a tighter bound can be derived for the bandit learning problem in two-sided matching markets. For the general one-to-one matching markets, we try to improve the learning efficiency of the existing algorithms. By integrating the offline GS procedure into the online learning process and carefully designing the elimination strategy, we show that the player-optimal stable regret can be upper bounded by ${ \cal O } ( \breve { N } ^ { 2 } \log T / \Delta ^ { 2 } + K \log \breve { T } / \Delta )$ . This result removes the dependence on $K$ in the main order term of existing works and improves the stateof-the-art result [31, 16] in common cases where the number of players is much smaller than that of arms. An experiment is conducted to verify its advantage over other baselines in such markets. We also present a novel analysis for the centralized UCB algorithm in markets satisfying α-condition and derive an improved $\bar { O } ( N \log T / \Delta _ { N } ^ { 2 } + K \log T / \Delta )$ regret upper bound.

One significant future direction is to investigate the optimality of algorithms. Although the dependence on $N , K , T$ in Theorem 6.2 matches the lower bound, the definition of $\Delta$ differs. It remains unclear how the upper bound changes with the same $\Delta .$ . Furthermore, since the lower bound provided by [28] applies only to special markets, and the learning problem in general markets is more challenging due to the complex preference structure, determining whether an algorithm can perform better in general markets is still an open problem.

## Acknowledgments and Disclosure of Funding

The corresponding author Shuai Li is supported by National Key Research and Development Program of China (2022ZD0114804) and National Natural Science Foundation of China (62376154). Fang Kong is supported by the Baidu Scholarship.

We thank Yuhao Zhang and Wenqian Wang for valuable discussions and suggestions on the proof of Lemma B.2.

## References

[1] Atila Abdulkadiroglu and Tayfun S ˘ onmez. House allocation with existing tenants. ¨ Journal of Economic Theory, 88(2):233–260, 1999.

[2] Shipra Agrawal and Navin Goyal. Further optimal regret bounds for thompson sampling. In Proceedings of the 16th International Conference on Artificial Intelligence and Statistics, pages 99–107, 2013.

[3] Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine learning, 47(2):235–256, 2002.

[4] Soumya Basu, Karthik Abinav Sankararaman, and Abishek Sankararaman. Beyond $\log ^ { 2 } ( t )$ regret for decentralized bandits in matching markets. In International Conference on Machine Learning, pages 705–715, 2021.

[5] Simon Clark. The uniqueness of stable matchings. Contributions in Theoretical Economics, 6(1), 2006.

[6] Xiaowu Dai and Michael Jordan. Learning in multi-stage decentralized matching markets. In Advances in Neural Information Processing Systems, volume 34, pages 12798–12809, 2021.

[7] Xiaowu Dai and Michael I Jordan. Learning strategies in decentralized matching markets under uncertain preferences. Journal ofMachine Learning Research, 22(1):11806–11855, 2021.

[8] Sanmay Das and Emir Kamenica. Two-sided bandits and the dating market. In International Joint Conference on Artificial Intelligence, pages 947–952, 2005.

[9] David Gale and Lloyd S Shapley. College admissions and the stability of marriage. The American Mathematical Monthly, 69(1):9–15, 1962.

[10] Aurelien Garivier, Tor Lattimore, and Emilie Kaufmann. On explore-then-commit strategies.´ In Advances in Neural Information Processing Systems, volume 29, pages 784–792, 2016.

[11] Avishek Ghosh, Abishek Sankararaman, Kannan Ramchandran, Tara Javidi, and Arya Mazumdar. Decentralized competing bandits in non-stationary matching markets. arXiv preprint arXiv:2206.00120, 2022.

[12] Meena Jagadeesan, Alexander Wei, Yixin Wang, Michael Jordan, and Jacob Steinhardt. Learning equilibria in matching markets from bandit feedback. In Advances in Neural Information Processing Systems, volume 34, 2021.

[13] Alexander Karpov. A necessary and sufficient condition for uniqueness consistency in the stable marriage matching problem. Economics Letters, 178:63–65, 2019.

[14] Emilie Kaufmann, Nathaniel Korda, and Remi Munos. Thompson sampling: An asymptoti-´ cally optimal finite-time analysis. In International Conference on Algorithmic Learning Theory, pages 199–213. Springer, 2012.

[15] Alexander S Kelso Jr and Vincent P Crawford. Job matching, coalition formation, and gross substitutes. Econometrica: Journal ofthe Econometric Society, pages 1483–1504, 1982.

[16] Fang Kong and Shuai Li. Player-optimal stable regret for bandit learning in matching markets. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA). SIAM, 2023.

[17] Fang Kong and Shuai Li. Improved bandits in many-to-one matching markets with incentive compatibility. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 13256–13264, 2024.

[18] Fang Kong, Junming Yin, and Shuai Li. Thompson sampling for bandit learning in matching markets. In International Joint Conference on Artificial Intelligence, 2022.

[19] Tor Lattimore and Csaba Szepesvari. ´ Bandit algorithms. Cambridge University Press, 2020.

[20] Lydia T Liu, Horia Mania, and Michael Jordan. Competing bandits in matching markets. In International Conference on Artificial Intelligence and Statistics, pages 1618–1628. PMLR, 2020.

[21] Lydia T Liu, Feng Ruan, Horia Mania, and Michael I Jordan. Bandit learning in decentralized matching markets. Journal ofMachine Learning Research, 22(211):1–34, 2021.

[22] Chinmay Maheshwari, Eric Mazumdar, and Shankar Sastry. Decentralized, communicationand coordination-free learning in structured matching markets. In Advances in Neural Information Processing Systems, 2022.

[23] Yifei Min, Tianhao Wang, Ruitu Xu, Zhaoran Wang, Michael Jordan, and Zhuoran Yang. Learn to match with no regret: Reinforcement learning in markov matching markets. In Advances in Neural Information Processing Systems, volume 35, pages 19956–19970, 2022.

[24] Deepan Muthirayan, Chinmay Maheshwari, Pramod Khargonekar, and Shankar Sastry. Competing bandits in time varying matching markets. In Learning for Dynamics and Control Conference, pages 1020–1031. PMLR, 2023.

[25] Alvin E Roth and Marilda Sotomayor. Two-sided matching. Handbook of game theory with economic applications, 1:485–541, 1992.

[26] Alvin E Roth. The evolution of the labor market for medical interns and residents: a case study in game theory. Journal of political Economy, 92(6):991–1016, 1984.

[27] Alvin E Roth. Stability and polarization of interests in job matching. Econometrica: Journal of the Econometric Society, pages 47–57, 1984.

[28] Abishek Sankararaman, Soumya Basu, and Karthik Abinav Sankararaman. Dominate or delete: Decentralized competing bandits in serial dictatorship. In International Conference on Artificial Intelligence and Statistics, pages 1252–1260. PMLR, 2021.

[29] Zilong Wang and Shuai Li. Optimal analysis for bandit learning in matching markets with serial dictatorship. Theoretical Computer Science, 1010:114703, 2024.

[30] Zilong Wang, Liya Guo, Junming Yin, and Shuai Li. Bandit learning in many-to-one matching markets. In Proceedings ofthe 31st ACM International Conference on Information & Knowledge Management, pages 2088–2097, 2022.

[31] Yirui Zhang, Siwei Wang, and Zhixuan Fang. Matching in multi-arm bandit with collision. In Advances in Neural Information Processing Systems, 2022.

## A Proof of Theorem 4.1

Define $\begin{array} { r } { \mathcal { F } = \left\{ \exists 1 \leq t \leq T , i \in [ N ] , j \in [ K ] : | \hat { \mu } _ { i , j } ( t ) - \mu _ { i , j } | > \sqrt { \frac { 6 \log T } { T _ { i , j } ( t ) } } \right\} } \end{array}$ as the failure event that the estimated reward is far from the expected reward at some time and some player-arm pair. The regret can be upper bounded as follows.

$$
\begin{array} { r l } & { \boldsymbol { R e g } _ { i } ( t ) = \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \left( \boldsymbol { \mu } _ { i , m _ { i } ^ { * } } - X _ { i } ( t ) \right) \right] } \\ & { \qquad \le \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \left( \boldsymbol { \mu } _ { i , m _ { i } ^ { * } } - X _ { i } ( t ) \right) \mid \boldsymbol { \mathcal { T } } \right] + \mathbb { P } \left( \boldsymbol { \mathcal { F } } \right) \cdot \boldsymbol { T } \cdot \boldsymbol { \mu } _ { i , m _ { i } ^ { * } } } \\ & { \qquad \le \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { a , \boldsymbol { \xi } } \boldsymbol { 1 } \left\{ \bar { A } _ { i } ( t ) = a _ { j } \right\} \cdot \Delta _ { i , m _ { i } ^ { * } , j } \mid \boldsymbol { \mathcal { T } } \right] + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \boldsymbol { 1 } \left\{ \bar { A } _ { i } ( t ) = \emptyset \right\} \cdot \boldsymbol { \mu } _ { i , m _ { i } ^ { * } } \mid \boldsymbol { \mathcal { T } } \right] } \\ & { \qquad + \mathbb { P } \left( \boldsymbol { \mathcal { F } } \right) \cdot \boldsymbol { T } \cdot \boldsymbol { \mu } _ { i , m _ { i } ^ { * } } } \\ & { \qquad \le 9 6 N ^ { 2 } \log T / \Delta ^ { 2 } + 9 6 K \log T / \Delta + 1 9 2 N ^ { 2 } \log T / \Delta ^ { 2 } + 2 N K } \\ & { \qquad = \mathcal { O } \left( N ^ { 2 } \log T / \Delta ^ { 2 } + K \log T / \Delta \right) . } \end{array}\tag{1}
$$

where Eq. (1) holds based on Lemma $\mathrm { A } . 1 , \mathrm { A } . 2$ , and A.4.

Lemma A.1.

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \sum _ { a _ { j } } \mathbb { 1 } \big \{ \bar { A } _ { i } ( t ) = a _ { j } \big \} \cdot \Delta _ { i , m _ { i } ^ { * } , j } \big | ^ { \gamma } \mathcal { F } \right] \leq 9 6 N ^ { 2 } \log T / \Delta ^ { 2 } + 9 6 K \log T / \Delta .
$$

Proof. Recall that player $p _ { i }$ would update the available set $\mathbf { \mathcal { A } } _ { i }$ when other players $p _ { i ^ { \prime } }$ sets $\mathrm { E } _ { i ^ { \prime } } =$ False and $\pi _ { A _ { i ^ { \prime } } , i ^ { \prime } } \ : > \ : \pi _ { A _ { i ^ { \prime } } , i }$ as Line 14. Denote $t _ { s }$ as the round index when this operation happens for the s-th time. Without loss of generality, let $t _ { 0 } = 1$

Recall that at a high level, each time another player $p _ { i ^ { \prime } }$ sets $\mathrm { E } _ { i ^ { \prime } }$ as False, it means that $p _ { i ^ { \prime } }$ learns its most preferred arm in current available set. Combined with $\mathcal { F }$ and Lemma $_ { \mathrm { A . 6 , } }$ the determined arm of players during each exploration would be truly their most preferred one. Thus the AETGS algorithm is an online version of GS and the $s ^ { \prime } { \mathrm { - } } { \mathrm { } } { \mathrm { i h } }$ time player $p _ { i ^ { \prime } }$ sets $\mathrm { E } _ { i ^ { \prime } }$ as False is equivalent to that $p _ { i ^ { \prime } }$ proposes its $s ^ { \prime } .$ -th most preferred arm in the offline GS. According to Lemma $\mathrm { A . } 3 ,$ , at most $N - 1$ arms are proposed by all players before reaching stability. Thus for player $p _ { i } .$ , the operation in Line 14 would happen for at most $N - 1$ times.

Recall that for each s, during time $t _ { s }$ to $t _ { s + 1 }$ , player $p _ { i }$ would explore all available arms in a roundrobin manner, eliminate sub-optimal arms until $N$ arms are in the set, and focuses on the best one among these $N$ when it is identified. For convenience, denote $R _ { s }$ as the set of the remaining $N$ arms that $p _ { i }$ explored in $A _ { i }$ in a round-robin manner until condition Line $^ 9$ is satisfied, $D _ { s }$ as the set of arms that $p _ { i }$ eliminated due to condition Line $^ { 6 , }$ and $j _ { s }$ as the arm that $p _ { i }$ focuses from the time it sets $\mathrm { E } _ { i }$ as False to time $t _ { s + 1 } - 1$ . Then it holds that

$$
\begin{array} { r l } & { \mathbb { E } [ \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { a _ { j } } \mathbb { 1 } \big \{ \bar { A } _ { i } ( t ) = a _ { j } \big \} \cdot \Delta _ { i , m _ { i } ^ { * } , j } \big | \mathcal { T } ] } \\ & { \le \mathbb { E } [ \displaystyle \sum _ { s = 0 } ^ { N - 1 } \displaystyle \sum _ { t = t _ { s } } ^ { t _ { s + 1 } - 1 } \sum _ { a _ { j } } \mathbb { 1 } \big \{ \bar { A } _ { i } ( t ) = a _ { j } \big \} \cdot \Delta _ { i , m _ { i } ^ { * } , j } \big | \ \mathcal { T } ] } \\ & { \le \mathbb { E } [ \displaystyle \sum _ { s = 0 } ^ { N - 1 } \displaystyle \sum _ { t = t _ { s } } ^ { t _ { s + 1 } - 1 } ( \displaystyle \sum _ { a _ { j } \in E _ { B } } \mathbb { 1 } \big \{ \bar { A } _ { i } ( t ) = a _ { j } \big \} \cdot \Delta _ { i , m _ { i } ^ { * } , j } + \displaystyle \sum _ { a _ { j } \in D _ { s } } \mathbb { 1 } \big \{ \bar { A } _ { i } ( t ) = a _ { j } \big \} \cdot \Delta _ { i , m _ { i } ^ { * } , j }   } \\ & { \quad   + \mathbb { 1 } \big \{ \bar { A } _ { i } ( t ) = a _ { j _ { s } } \big \} \cdot \Delta _ { i , m _ { i } ^ { * } , j _ { s } } \big ) \big | \mathcal { T } ] } \end{array}
$$

$$
\leq \mathbb { E } \left[ \sum _ { s = 0 } ^ { N - 1 } \sum _ { t = t _ { s } } ^ { t _ { s + 1 } - 1 } \left( \sum _ { a _ { j } \in R _ { s } } \mathbb { 1 } \big \{ \hat { A } _ { i } ( t ) = a _ { j } \big \} \cdot \Delta _ { i , m _ { i } ^ { * } , j } + \sum _ { a _ { j } \in D _ { s } } \mathbb { 1 } \big \{ \hat { A } _ { i } ( t ) = a _ { j } \big \} \cdot \Delta _ { i , m _ { i } ^ { * } , j } \right) | ^ { \gamma } \right] ,\tag{2}
$$

where Eq. (2) is due to that, based on Lemma A.6 and the offline GS algorithm, the arm $j _ { s }$ before GS stops would be better than $m _ { i } ^ { * }$ , thus the regret caused by selecting these arms is less than 0.

For the first term in Eq. (2), we have

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { s = 0 } ^ { N - 1 } \displaystyle \sum _ { \alpha _ { j } \in R _ { s } } \sum _ { \alpha _ { j } \in R _ { s } } \mathbf { 1 } \left\{ \bar { A } _ { i } ( t ) = a _ { j } \right\} \cdot \Delta _ { i , m _ { i } ^ { * } , j } | ^ { \gamma } \right] } \\ & { = \mathbb { E } \left[ \displaystyle \sum _ { s = 0 } ^ { N - 1 } \displaystyle \sum _ { \alpha _ { j } \in R _ { s } } \sum _ { \zeta = t _ { s } } ^ { t _ { s + 1 } - 1 } \mathbf { 1 } \left\{ \bar { A } _ { i } ( t ) = a _ { j } \right\} \cdot \Delta _ { i , m _ { i } ^ { * } , j } | ^ { \gamma } \right] } \\ & { \le \mathbb { E } \left[ \displaystyle \sum _ { s = 0 } ^ { N - 1 } \displaystyle \sum _ { \alpha _ { j } \in R _ { s } } \frac { 9 6 \log T } { \Delta _ { i , j , \delta _ { j + 1 } } ^ { 2 } } \cdot \Delta _ { i , m _ { i } ^ { * } , j } | ^ { \gamma } \right] } \\ & { \le \frac { 9 6 N ^ { \frac { 2 } { 2 } } \log T } { \Delta ^ { 2 } } . } \end{array}\tag{3}
$$

(4)

(5)

where Eq. (4) is due to Lemma A.5, Eq. (5) holds since $R _ { s }$ contains no more than N arms according to the elimination condition (Line 6).

We now analyze the second term in Eq. (2). For any arm $a _ { j }$ and $s \in \{ 0 , . . . , N - 1 \}$ , denote $T _ { i , j , s }$ as the value of $T _ { i , j }$ at the end of the round $t _ { s + 1 } - 1$ For $s \geq 1$ and arm $a _ { j } ~ \in ~ D _ { s }$ if $\begin{array} { r l } { T _ { i , j , s - 1 } } & { { } \leq } \end{array}$ 96 log $T / \Delta _ { i , j _ { s - 1 } , j } ^ { 2 } ,$ it must hold that $\begin{array} { r } { T _ { i , j , s } : = \sum _ { s ^ { \prime } \leq s } ( T _ { i , j , s ^ { \prime } } - T _ { i , j , s ^ { \prime } - 1 } ) \leq } \end{array}$ 96 log $T / \Delta _ { i , j _ { s } , j } ^ { 2 }$ to ensure arm $a _ { j }$ is eliminated from $\mathbf { \mathcal { A } } _ { i }$ at step s based on Lemma $_ { \mathrm { A . 5 . } }$ . On the other hand, if $T _ { i , j , s - 1 } > 9 6$ log $T / \Delta _ { i , j _ { s - 1 } , j } ^ { 2 } ,$ based on Lemma $\mathrm { A . 5 . }$ , it holds that $T _ { i , j , s } - T _ { i , j , s - 1 } \leq$ 96 log $T / \Delta _ { i , j _ { s } , j } ^ { 2 } \ - \ 9 6$ 6 log $T / \Delta _ { i , j _ { s - 1 } , j } ^ { 2 }$ when $a _ { j }$ is eliminated. For any arm $a _ { j }$ , denote $s _ { j , 1 } : =$ max $\quad : 0 \leq s \leq N - 1 \left\{ s : T _ { i , j , s } \leq 9 6 \log T / \Delta _ { i , j _ { s } , j } ^ { 2 } \right\}$ as the last step when the number of observation times on $a _ { j }$ is less than that threshold. Then the second term in Eq. (2) satisfies

$$
\begin{array} { r l } & { \mathbb { E } [ \frac { \gamma _ { \varepsilon } ( T _ { \varepsilon } ) + \varepsilon - 1 } { \varepsilon } \sum _ { i , j \in \mathcal { E } _ { \varepsilon _ { i } } } \sum _ { \alpha _ { i , j \in \mathcal { E } _ { \varepsilon _ { i } } } } \mathbb { E } _ { i } ( \phi _ { i } - \phi _ { \varepsilon _ { i } } ) \cdot \Delta _ { \alpha _ { i , \alpha _ { i } \in \mathcal { E } _ { \varepsilon _ { i } } } } | \mathcal { F } ] } \\ & { = \mathbb { E } [ \frac { \sum _ { \alpha _ { i , j \in \mathcal { E } _ { \varepsilon _ { i } } } } \sum _ { \alpha _ { i , j \in \mathcal { E } _ { \varepsilon _ { i } } } } \varepsilon - 1 } { \varepsilon } \frac { \varepsilon - \varepsilon - 1 } { \varepsilon } \mathbb { E } _ { i } ( \tilde { \lambda } _ { i } ( \theta _ { i } - \phi _ { i } ) \cdot \Delta _ { \alpha _ { i , \alpha _ { i } \in \mathcal { E } _ { \varepsilon _ { i } } } } | \mathcal { F } ) ] } \\ & { = \mathbb { E } [ \frac { \sum _ { \alpha _ { i , j \in \mathcal { E } _ { \varepsilon _ { i } } } } \sum _ { \alpha _ { i , j \in \mathcal { E } _ { \varepsilon _ { i } } } } \varepsilon - 1 } { \varepsilon } \frac { \varepsilon - 1 } { \varepsilon } \frac { \varepsilon - 1 } { \varepsilon } \Delta _ { \alpha _ { i , j \in \mathcal { E } _ { \varepsilon _ { i } } } } ( \mathbb { E } _ { i } ( \tilde { \lambda } _ { i } ( \theta _ { i } - \phi _ { i } ) \cdot \Delta _ { \alpha _ { i , \alpha _ { i } \in \mathcal { E } _ { \varepsilon _ { i } } } } | \mathcal { F } ) ) ] } \\ &  = \mathbb { E } [ \frac  \sum _ { \alpha _ { i , j \in \mathcal { E } _ { \varepsilon _ { i } } } } \sum _ { \alpha _ { i , j \in \mathcal { E } _ { \varepsilon _ { i } } } } ( \mathbb { E } _ { i } ( \tilde { \lambda } _ { i } ( \theta _ { i } - \phi \end{array}
$$

$$
\leq \sum _ { a _ { j } \in { \mathcal { K } } } { \frac { 9 6 \log T } { \Delta _ { i , m _ { i } ^ { * } , j } ^ { 2 } } } \cdot \Delta _ { i , m _ { i } ^ { * } , j } \leq 9 6 K \log T / \Delta ,
$$

where the second last line is due to the definition of $s _ { j , 1 }$ <sub>1</sub> and the above analysis.

Above all,

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \sum _ { a _ { j } } \mathbf { 1 } \big \{ \bar { A } _ { i } ( t ) = a _ { j } \big \} \cdot \Delta _ { i , m _ { i } ^ { * } , j } \big | \top \right] \leq \mathrm { E q . } ( 2 ) \leq 9 6 N ^ { 2 } \log T / \Delta ^ { 2 } + 9 6 K \log T / \Delta .
$$

Lemma A.2.

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbb { 1 } \big \{ \bar { A } _ { i } ( t ) = \emptyset \big \} \cdot \mu _ { i , m _ { i } ^ { * } } \mid ^ { \top } \mathcal { F } \right] \leq 1 9 2 N ^ { 2 } \log T / \Delta ^ { 2 } .
$$

Proof. Based on the AETGS algorithm, when $\mathrm { E } _ { i } = \mathrm { T r u e }$ , the central platform would assign arms in $A _ { i }$ to player $p _ { i }$ in a round-robin manner. Since the number of arms $| \cup _ { i : \mathrm { E } _ { i } = \mathrm { T r u e } } \mathcal { A } _ { i }$ | to be explored is larger than the number $\begin{array} { r } { \sum _ { i } \mathbb { 1 } \{ \mathrm { E } _ { i } = \mathrm { T r u e } \} } \end{array}$ of players with $\mathrm { E } _ { i } =$ True based on the elimination condition in Line $^ { 6 , }$ we can assume that there is no collision in the exploration phase as discussed in Section 4. So the regret caused by collision only occurs during time with $\mathrm { E } _ { i } = \mathrm { F a l s e }$

Denote $\underline { { t } } _ { s }$ and $\bar { t } _ { s }$ as the round index when $p _ { i }$ sets $\mathrm { E } _ { i }$ as False for the s-th time and as True for the s + 1-th time, respectively. Recall that when $\mathrm { E } _ { i } = \mathrm { F a l s e } , p _ { i }$ will always select arm $A _ { i }$ . Here we use $j _ { s }$ to represent the arm that is selected by $p _ { i }$ from time $\underline { { t } } _ { s } ~ \mathrm { t o } ~ \overline { { t } } _ { s }$

Further, recall that in the AETGS algorithm, each time an arm is added into $\mathcal { D } _ { i }$ (Line 13), the eliminated arms may be contained into $A _ { i }$ again. And only when other players focus on their currently most preferred arm, such operation of adding arms to $\mathcal { D }$ happens. Based on Lemma ${ \mathrm { A } } . 3$ such an operation happens for at most $N$ times. For any player $p _ { i ^ { \prime } }$ , denote $t _ { i ^ { \prime } , r }$ as the round index when $p _ { i ^ { \prime } }$ adds arms to $\mathcal { D } _ { i ^ { \prime } }$ (Line 13) for r-th time. Then $\left\{ { t } _ { i ^ { \prime } , r } \right\} _ { r \in [ N ] }$ further divide $\left\{ \left[ \underline { { t } } _ { s } , \overline { { t } } _ { s } \right] \right\} _ { s \in \left[ N \right] }$ into at most 2N slices. We use $\underline { { t ^ { \prime } } } _ { s } , \overline { { t ^ { \prime } } } _ { s }$ to represent the start round and end round index of the s-th slice, where $s \in [ 2 N ]$ ]. Based on Lemma A.5, $p _ { i ^ { \prime } }$ and $p _ { i }$ would select the same arm for at most 96 log $T / \Delta ^ { 2 }$ times within each slice.

Then the regret satisfies

$$
\begin{array} { r l } { \mathbb { E } [ \displaystyle \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ \hat { \lambda } _ { i } ( t ) = 0 \} \cdot \boldsymbol { \mu } _ { i , i \in \mathbb { N } _ { i } } | \mathcal { V } ] } & { \leq \mathbb { E } [ \displaystyle \sum _ { i = 1 } ^ { n } \boldsymbol { \frac { \lambda } { i } } \{ \hat { \lambda } _ { i } ( t ) = 0 \} \} ^ { - \alpha _ { i } } \mu _ { i } ( t ) - \boldsymbol { \mu } _ { i } ( t ) \cdot \boldsymbol { \mu } _ { i } ( t ) ] } \\ & { = \mathbb { E } [ \displaystyle \sum _ { i = 1 - \alpha _ { i } } ^ { n } \mu _ { i } ( t ) - \boldsymbol { \mu } _ { i } ( t ) - \boldsymbol { \mu } _ { i } ( t ) - \boldsymbol { \mu } _ { i } ( t ) \cdot \boldsymbol { \mu } _ { i } ( t ) ] } \\ & { \leq \mathbb { E } [ \displaystyle \sum _ { i \neq \alpha _ { i } = 1 } ^ { n } \frac { \mathcal { N } } { \alpha _ { i } } \mathbf { 1 } \{ \{ \boldsymbol { \mu } _ { i } ( t ) = \boldsymbol { \lambda } _ { i } ( t ) = \boldsymbol { \lambda } _ { i } ( t ) = \boldsymbol { \lambda } _ { i } \} \cdot \boldsymbol { \mu } _ { i , i \in \mathbb { N } _ { i } } | \mathcal { V } ] } \\ & { \leq \mathbb { E } [ \displaystyle \sum _ { i = \alpha _ { i } = 1 } ^ { n } \frac { \mathcal { N } } { \alpha _ { i } } \mathbf { 1 } \{ \boldsymbol { \mu } _ { i } ( t ) = \boldsymbol { \lambda } _ { i } ( t ) = \boldsymbol { \lambda } _ { i } ( t ) = \boldsymbol { \lambda } _ { i } \cdot \boldsymbol { \mu } _ { i , i \in \mathbb { N } _ { i } } | \mathcal { V } ] } \\ &  \leq \mathbb { E } [ \displaystyle \sum _ { i = \alpha _ { i } = 1 } ^ { n } \sum _ { i = \alpha _ { i } } ^ { \infty } \mathbf { 1 } \{ \boldsymbol { \mu } _ { i } ( t ) = \boldsymbol { \lambda } _ { i } ( \end{array}
$$

Lemma A.3. In the offline GS algorithm, at most $N - 1$ arms have been proposed by players before the algorithm stops.

Proof. Based on the offline GS algorithm, once an arm is proposed, it has a temporary player. By contradiction, once N arms have been proposed, it means that N players are occupied. In this case, each player has a partner and the algorithm stops. □

Lemma A.4.

$$
\mathbb { P } \left( \mathcal { F } \right) \le 2 N K / T .
$$

Proof.

$$
\begin{array} { r l } & { \mathbb { P } \left( \mathcal { F } \right) = \mathbb { P } \left( \exists 1 \leq t \leq T , i \in [ N ] , j \in [ K ] : | \hat { \mu } _ { i , j } ( t ) - \mu _ { i , j } | > \sqrt { \frac { 6 \log T } { T _ { i , j } ( t ) } } \right) } \\ & { \qquad \leq \displaystyle \sum _ { t = 1 } ^ { T } \displaystyle \sum _ { i \in [ N ] } \sum _ { j \in [ K ] } \mathbb { P } \left( \left| \hat { \mu } _ { i , j } ( t ) - \mu _ { i , j } \right| > \sqrt { \frac { 6 \log T } { T _ { i , j } ( t ) } } \right) } \\ & { \qquad \leq \displaystyle \sum _ { t = 1 } ^ { T } \displaystyle \sum _ { i \in [ N ] } \sum _ { j \in [ K ] } \sum _ { s = 1 } ^ { t } \mathbb { P } \left( T _ { i , j } ( t ) = s , | \hat { \mu } _ { i , j } ( t ) - \mu _ { i , j } | > \sqrt { \frac { 6 \log T } { s } } \right) } \\ & { \qquad \leq \displaystyle \sum _ { t = 1 } ^ { T } \displaystyle \sum _ { i \in [ N ] , j \in [ K ] } t \cdot 2 \exp ( - 3 \ln T ) } \\ & { \qquad \leq \displaystyle 2 \sum _ { t = 1 } ^ { T } \sum _ { i \in [ N ] , j \in [ K ] } 1 } \end{array}
$$

where the second last inequality is due to Lemma B.1.

Lemma A.5. For any player $p _ { i } { \mathrm { : } }$ , let $\bar { T } _ { i } = 9 6 \log T / \Delta ^ { 2 }$ . For any two arms ${ j , j ^ { \prime } }$ with $\mu _ { i , j } > \mu _ { i , j ^ { \prime } }$ and $\sigma _ { i } ( a _ { j } ) \in [ 1 , \sigma _ { i } ( m _ { i } ^ { * } ) ] , i f T _ { i } ( t ) : = \operatorname* { m i n } { \{ T _ { i , j } ( t ) , T _ { i , j ^ { \prime } } ( t ) \} } > \bar { T } _ { i }$ , we have $\mathrm { U C B } _ { i , j ^ { \prime } } ( t ) < \mathrm { L C B } _ { i , j } ( t )$ conditioned on $\dot { \neg } \mathcal { F }$

Proof. By contradiction, suppose $\mathrm { U C B } _ { i , j ^ { \prime } } ( t ) \geq \mathrm { L C B } _ { i , j } ( t )$ . According to $\daleth \mathcal { F }$ and the definition of LCB and UCB, we have

$$
\mu _ { i , j } - 2 \sqrt { \frac { 6 \log T } { T _ { i } ( t ) } } \leq \mathrm { L C B } _ { i , j } ( t ) \leq \mathrm { U C B } _ { i , j ^ { \prime } } ( t ) \leq \mu _ { i , j ^ { \prime } } + 2 \sqrt { \frac { 6 \log T } { T _ { i } ( t ) } } .
$$

We can then conclude $\begin{array} { r } { \Delta _ { i , j , j ^ { \prime } } = \mu _ { i , j } - \mu _ { i , j ^ { \prime } } \leq 4 \sqrt { \frac { 6 \log T } { T _ { i } ( t ) } } } \end{array}$ , which implies that $\begin{array} { r } { T _ { i } ( t ) \leq \frac { 9 6 \log T } { \Delta _ { i , j , j ^ { \prime } } ^ { 2 } } \leq } \end{array}$ $\frac { 9 6 \log T } { \Delta ^ { 2 } }$ . This contradicts the fact that $T _ { i } ( t ) > \bar { T } _ { i }$ □

Lemma A.6. Conditioned on $^ { \neg } { \mathcal { F } } ,$ at any time t, $\mathrm { U C B } _ { i , j } ( t ) < \mathrm { L C B } _ { i , j ^ { \prime } } ( t )$ implies $\mu _ { i , j } < \mu _ { i , j ^ { \prime } }$

Proof. According to the definition of LCB and UCB, we have

$$
\mathrm { L C B } _ { i , j } ( t ) = \hat { \mu } _ { i , j } ( t ) - \sqrt { \frac { 6 \log T } { T _ { i , j } ( t ) } } \leq \mu _ { i , j } \leq \hat { \mu } _ { i , j } ( t ) + \sqrt { \frac { 6 \log T } { T _ { i , j } ( t ) } } = \mathrm { U C B } _ { i , j } ( t ) ,
$$

where two inequalities comes from $\daleth \mathcal { F } .$ Thus if $\mathrm { U C B } _ { i , j } ( t ) < \mathrm { L C B } _ { i , j ^ { \prime } } ( t )$ , there would be

$$
\mu _ { i , j } \leq \mathrm { U C B } _ { i , j } ( t ) < \mathrm { L C B } _ { i , j ^ { \prime } } ( t ) \leq \mu _ { i , j ^ { \prime } } .
$$

The lemma can thus be proved.

## B Technical Lemmas

Lemma B.1. (Corollary 5.5 in Lattimore and Szepesvari [´ 19]) Assume that $X _ { 1 } , X _ { 2 } , \ldots , X _ { n }$ are independent, σ-subgaussian random variables centered around µ. Thenfor any $\varepsilon > 0$

$$
\mathbb { P } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i } \ge \mu + \varepsilon \right) \le \exp \left( - \frac { n \varepsilon ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , \mathbb { P } \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i } \le \mu - \varepsilon \right) \le \exp \left( - \frac { n \varepsilon ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .
$$

Lemma B.2. (Arrangement of players’ round-robin exploration) Suppose there are N players who need to explore their respective N arms. There exists an assignment such that during 2N rounds, each player can match with each ofits armfor once.

Proof. Without loss of generality, let’s assume that players assign their respective N arms in 2N rounds one by one, based on the players’ and arms’ indices, aiming to ensure that no arm is assigned to more than one player at the same round. By contradiction, suppose when player $p _ { i }$ assigns its $j \cdot$ -th arm, there is no available round to make this assignment due to conflicting constraints. Given that player $p _ { i }$ is currently assigning the j-th arm, it implies that there are $2 N - j + 1$ rounds where no arm is assigned to player $p _ { i }$ . Since none of these rounds satisfy the conflict constraint, it means that the previous i − 1 players assigned arm $j$ in these $2 N - j + 1$ rounds. This creates a contradiction since the first $i - 1$ players can only occupy $i - 1$ rounds when selecting arm $j ,$ , where $i - 1 < 2 N - j + 1$ with $i \le N , j \le N$ □

## C Proof of Theorem 6.2

In this section, we analyze the regret of the centralized-UCB algorithm under α-condition. Recall that $\begin{array} { r } { \mathcal { F } = \left\{ \exists 1 \leq t \leq T , i \in [ N ] , j \in [ K ] : | \hat { \mu } _ { i , j } ( t ) - \mu _ { i , j } | > \sqrt { \frac { 6 \log T } { T _ { i , j } ( t ) } } \right\} } \end{array}$ is the failure event that the estimated reward is far from the expected reward at some time and some player-arm pair.

For any player $p _ { i }$ with $i \in [ N ]$ , we know that its stable arm is $a _ { i }$ under α-condition. Thus its regret can be decomposed as

$$
R e g _ { i } ( T ) \leq \mathbb { E } \left[ \sum _ { k : \mu _ { i } , k < \mu _ { i , i } } \Delta _ { i , i , k } \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { i } ( t ) = k , \top \mathcal { F } \right\} \right] + T \cdot \mathbb { P } \left( \mathcal { F } \right) .
$$

The first term is the number of selections for sub-optimal arms. The second term is the regret caused by the bad events.

For the arm $a _ { k }$ such that it is sub-optimal for player $p _ { i }$ , i.e., $\mu _ { i , k } < \mu _ { i , i }$ , it will be selected because the preference for arm $a _ { k }$ of player $p _ { i }$ is estimated higher than its stable matched arm $a _ { i } .$ , or player $p _ { i }$ is rejected by arm $a _ { k }$ in the GS algorithm. Note that under α-condition, there is a right order $\mathsf { \bar { Q } } _ { i ^ { \prime } } = \mathsf { \bar { \Phi } } _ { i } \in [ N ] ,$ for player $p _ { i } ,$ , such that $\forall i ^ { \prime } < i ^ { \prime \prime } \leq N , Q _ { i ^ { \prime \prime } } \in [ N ] _ { r } : \pi _ { q _ { i ^ { \prime } } , Q _ { i ^ { \prime } } } > \pi _ { q _ { i ^ { \prime } } , Q _ { i ^ { \prime \prime } } } .$ , which means arm $a _ { q _ { i ^ { \prime } } } = a _ { i }$ can only prefer players $p _ { Q _ { 1 } } , p _ { Q _ { 2 } } , \cdots , p _ { Q _ { i ^ { \prime } - 1 } }$ than player $p _ { Q _ { i ^ { \prime } } } = p _ { i }$ . Denote The right-order mapping for α-condition for player $p _ { i }$ is $l r ( i )$ so that $Q _ { l r ( i ) } = i$ with $Q _ { i }$ defined in Definition 6.1, and $l r ( i ) \leq N$ . For player $p _ { i }$ , denote $\mathcal { G } _ { t , i } : = \{ \forall 1 \leq i ^ { \prime } \leq l r ( i ) - 1 , \bar { A } _ { Q _ { i ^ { \prime } } } ( t ) \neq i \}$ as the event all players preferred by arm $a _ { i }$ do not select $p _ { i }$ at time t. Then the number of selections for sub-optimal arm $a _ { k }$ can be decomposed as

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbf { 1 } \big \{ \bar { A } _ { i } ( t ) = k , { \boldsymbol { \mathcal { F } } } \big \} \right] } \\ & { = \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbf { 1 } \big \{ \bar { A } _ { i } ( t ) = k , \boldsymbol { \mathcal { G } } _ { t , i } , { \boldsymbol { \mathcal { F } } } \big \} \right] + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbf { 1 } \big \{ \bar { A } _ { i } ( t ) = k , { \boldsymbol { \mathcal { I } } } _ { t , i } , { \boldsymbol { \mathcal { F } } } \big \} \right] } \\ & { \leq \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbf { 1 } \big \{ \bar { A } _ { i } ( t ) = k , \mathrm { U C B } _ { i , k } ( t ) > \mathrm { U C B } _ { i , \delta } ( t ) , { \boldsymbol { \mathcal { F } } } \big \} \right] + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbf { 1 } \big \{ \bar { A } _ { i } ( t ) = k , { \boldsymbol { \mathcal { G } } } _ { t , i } , { \boldsymbol { \mathcal { F } } } \big \} \right] } \\ & { \leq \displaystyle \frac { 2 4 \log T } { \Delta _ { i , i , k } ^ { 2 } } + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbf { 1 } \big \{ \bar { A } _ { i } ( t ) = k , { \boldsymbol { \mathcal { G } } } _ { t , i } , { \boldsymbol { \mathcal { F } } } \big \} \right] . } \end{array}
$$

The last inequality is from Lemma C.1.

For the second term in the RHS of the last inequality, $\begin{array} { r } { \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { i } ( t ) = k , \daleth _ { f _ { t , i } , \daleth } \mathcal { F } \right\} \right] } \end{array}$ , we can sum over all sub-optimal arms and it turns out to be

$$
\mathbb { E } \left[ \sum _ { k } \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { i } ( t ) = k , \mathcal { I } _ { t , i } , \mathcal { I } \right\} \right]
$$

$$
\begin{array} { r l } & { \le \mathbb { E } [ \displaystyle \sum _ { i = 1 } ^ { \infty } [ \frac { \displaystyle \sum _ { j = 1 } ^ { n } ( \bar { x } _ { i , j } , \eta _ { j } ) } { \displaystyle \sum _ { i = 1 } ^ { n } ( \bar { x } _ { i , j } , \eta _ { i } ) } ] } \\ & { \le \mathbb { E } [ \displaystyle \sum _ { i = 1 } ^ { \infty } \frac { \displaystyle \sum _ { j = 1 } ^ { n } \mathbb { E } ( \bar { x } _ { i , j } , \eta _ { i } ) - \mathbb { E } ( \bar { x } _ { i , j } , \eta _ { j } ) } { \displaystyle \sum _ { i = 1 } ^ { n } ( \bar { x } _ { i , j } , \eta _ { i } ) } ] } \\ & { \le ( \bar { x } _ { i - 1 } - \displaystyle \sum _ { i = 1 } ^ { n + 1 } \frac { \displaystyle \sum _ { j = 1 } ^ { n } \log \eta _ { i } } { \displaystyle \sum _ { i = 1 } ^ { n } ( \bar { x } _ { i , j } , \eta _ { i } ) } ) } \\ & { \le \displaystyle \sum _ { i = 1 } ^ { \infty } \frac { \displaystyle \sum _ { j = 1 } ^ { n } ( \bar { x } _ { i , j } , \eta _ { j } ) } { \displaystyle \sum _ { i = 1 } ^ { n } ( \bar { x } _ { i , j } , \eta _ { i } ) } } \\ & { \le \displaystyle \sum _ { i = 1 } ^ { \infty } \frac { \displaystyle \sum _ { j = 1 } ^ { n } \sum _ { i = 1 } ^ { n } \frac { \displaystyle \sum _ { j = 1 } ^ { n } \log \eta _ { i } } { \displaystyle \sum _ { i = 1 } ^ { n } ( \bar { x } _ { i , j } , \eta _ { i } ) ^ { 2 } } } { \displaystyle \sum _ { i = 1 } ^ { n } ( \bar { x } _ { i - 1 } , \eta _ { i } ) ^ { 2 } } } \\ &  \le ( i \eta _ { i } + \eta _ { i } ) - 1 ( \displaystyle \sum _ { i = 1 } ^ { n } \frac   \end{array}
$$

where the third inequity is from the Lemma C.2. The fourth inequality is from the definition of $\Delta _ { N }$ Above all, the stable regret of player i can be bounded by

$$
\begin{array} { r l } & { R e g _ { i } ( T ) \leq \mathbb { E } \left[ \displaystyle \sum _ { k : \mu _ { i } , k < \mu _ { i , i } } \Delta _ { i , i , k } \frac { T } { t = 1 } \Im \left\{ \bar { A } _ { i } ( t ) = k , \top \mathcal { F } \right\} \right] + T \cdot \mathbb { P } \left( \mathcal { F } \right) } \\ & { \qquad \leq \displaystyle \sum _ { k : \mu _ { i } , k < \mu _ { i , i } } \Delta _ { i , i , k } \frac { 2 4 \log T } { \Delta _ { i , i , k } ^ { 2 } } + \Delta _ { i , i , k } \left( l r ( i ) - 1 \right) \frac { 5 \pi ^ { 2 } \log T } { \Delta _ { N } ^ { 2 } } + 2 N K } \\ & { \qquad \leq \displaystyle \frac { 2 4 K \log T } { \Delta } + \left( l r ( i ) - 1 \right) \frac { 5 \pi ^ { 2 } \log T } { \Delta _ { N } ^ { 2 } } + 2 N K } \\ & { \qquad \leq \mathcal { O } \left( \frac { K \log T } { \Delta } + \frac { N \log T } { \Delta _ { N } ^ { 2 } } \right) , } \end{array}
$$

where the second inequality is based on Lemma $\mathrm { A . 4 }$

Lemma C.1. Conditioned on $^ { \neg } { \mathcal { F } } ,$ , under the traditional single-player UCB algorithm with single player $p _ { i } ,$ , the expected number of times at which the UCB index of arm $a _ { j ^ { \prime } }$ exceeds that of the better arm $a _ { j }$ , is at most $2 4 \log ( T ) / \Delta _ { i , j , j ^ { \prime } } ^ { 2 }$ by round T.

Proof. Conditioned on $^ { \neg } { \mathcal { F } } ,$ , for any $i , j , t$ we have,

$$
\mu _ { i , j } - \sqrt { \frac { 6 \log ( T ) } { T _ { i , j } ( t - 1 ) } } < \hat { \mu } _ { i , j } ( t - 1 ) < \mu _ { i , j } + \sqrt { \frac { 6 \log ( T ) } { T _ { i , j } ( t - 1 ) } } .\tag{6a}
$$

Recall that the UCB index is:

$$
\mathrm { U C B } _ { i , j } ( t ) = \hat { \mu } _ { i , j } ( t - 1 ) + \sqrt { \frac { 6 \log ( T ) } { T _ { i , j } ( t - 1 ) } } .\tag{6b}
$$

The event that arm $a _ { j ^ { \prime } }$ is successfully selected for player $p _ { i }$ rather than the better arm $a _ { j }$ at time t implies that

$$
\begin{array} { r } { \mathbf { U C B } _ { i , j ^ { \prime } } ( t ) > \mathbf { U C B } _ { i , j } ( t ) . } \end{array}\tag{6c}
$$

Hence,

$$
\begin{array} { r l } & { \mu _ { i , j ^ { \prime } } + 2 \sqrt { \frac { 6 \log ( T ) } { T _ { i , j ^ { \prime } } ( t - 1 ) } } \stackrel { ( 6 a ) } > \hat { \mu } _ { i , j ^ { \prime } } ( t - 1 ) + \sqrt { \frac { 6 \log ( T ) } { T _ { i , j ^ { \prime } } ( t - 1 ) } } } \\ & { \stackrel { ( 6 c ) } > \hat { \mu } _ { i , j } ( t - 1 ) + \sqrt { \frac { 6 \log ( T ) } { T _ { i , j } ( t - 1 ) } } } \\ & { \qquad > \mu _ { i , j } - \sqrt { \frac { 6 \log ( T ) } { T _ { i , j } ( t - 1 ) } } + \sqrt { \frac { 6 \log ( T ) } { T _ { i , j } ( t - 1 ) } } } \\ & { \quad = \mu _ { i , j } , } \end{array}
$$

which leads to

$$
{ T _ { i , j ^ { \prime } } ( t - 1 ) < \frac { 2 4 \log ( T ) } { \Delta _ { i , j , j ^ { \prime } } ^ { 2 } } , }
$$

where $\Delta _ { i , j , j ^ { \prime } }$ is the reward difference between the $\mu _ { i , j ^ { \prime } }$ and $\mu _ { i , j }$

Lemma C.2. For any player $p _ { i }$ with right order $Q _ { l r ( i ) }$ , the following inequality holds:

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { i ^ { \prime } = 1 } ^ { l r ( i ) - 1 } \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { Q _ { i ^ { \prime } } } ( t ) = i , \top \right\} \right] } \\ & { \le \mathbb { E } \left[ \displaystyle \sum _ { u ^ { \prime } = 1 } ^ { l r ( i ) - 1 } \displaystyle \sum _ { u ^ { \prime } = u ^ { \prime } + 1 } ^ { l r ( i ) } \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { Q _ { u ^ { \prime } } } ( t ) = q _ { u ^ { \prime \prime } } , \mathcal { G } _ { t , u ^ { \prime } } , \top \right\} \right] } \\ & { \overset { l r ( i ) - 1 } { \le } \displaystyle \sum _ { u ^ { \prime } = 1 } ^ { l r ( i ) } \displaystyle \sum _ { u ^ { \prime } = u ^ { \prime } + 1 } ^ { l r ( i ) } \displaystyle \frac { 2 4 \log T } { \Delta _ { Q _ { u ^ { \prime } } , q _ { u ^ { \prime } } , q _ { u ^ { \prime } } } ^ { 2 } } . } \end{array}
$$

Proof. For player $p _ { i }$ with right order $Q _ { l r ( i ) }$ , from α-condition we have that its stable matched arm $a _ { i }$ may prefer $p _ { Q _ { 1 } } , p _ { Q _ { 2 } } , \cdots , p _ { Q _ { l r ( i ) - 1 } }$ than player $p _ { Q _ { l r ( i ) } }$ . For any $\textit { i } ^ { \prime } < l r ( i )$ , we know that the number of times player $p _ { Q _ { i } }$ selects arm $a _ { i }$ is decomposed as by

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { Q _ { i ^ { \prime } } } ( t ) = i , \top \right\} \right] } \\ & { = \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { Q _ { i ^ { \prime } } } ( t ) = i , { \mathcal G } _ { t , i ^ { \prime } } , \top \right\} \right] + \mathbb { E } \left[ \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { Q _ { i ^ { \prime } } } ( t ) = i , \top { \mathcal G } _ { t , i ^ { \prime } } , \top \right\} \right] . } \end{array}
$$

The event $\daleth _ { t , i ^ { \prime } }$ implies that there exists another player $p _ { Q _ { i ^ { \prime \prime } } }$ with $i ^ { \prime \prime } < i ^ { \prime }$ that selects the stable arm of $p _ { Q _ { i ^ { \prime } } }$ . This leads to a recursion form. But it is easy to verify that every event $\neg { \mathcal { G } } _ { t , i ^ { \prime } }$ happens only when there exists two players $p _ { Q _ { u ^ { \prime } } } , p _ { Q _ { u ^ { \prime \prime } } }$ with $u ^ { \prime } < \overline { { u ^ { \prime \prime } } } \leq l r ( \overline { { i ^ { \prime } } } )$ , such that player $p _ { Q _ { u ^ { \prime } } }$ explores the stable matched arm $a _ { q _ { u ^ { \prime \prime } } } \mathrm { o f } p _ { Q _ { u ^ { \prime \prime } } } , \mathrm { i . e . , } p _ { Q _ { u ^ { \prime } } }$ selects $\boldsymbol { a } _ { \boldsymbol { q } _ { u ^ { \prime \prime } } }$ conditioned on $\mathcal { G } _ { t , u ^ { \prime } } ^ { }$ . And thus it holds that

$$
\begin{array} { r l } & { \mathbb { E } \left[ \displaystyle \sum _ { i ^ { \prime } = 1 } ^ { l r ( i ) - 1 } \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { Q _ { i ^ { \prime } } } ( t ) = i , \top \right\} \right] } \\ & { \le \mathbb { E } \left[ \displaystyle \sum _ { u ^ { \prime } = 1 } ^ { l r ( i ) - 1 } \displaystyle \sum _ { u ^ { \prime } = u ^ { \prime } + 1 } ^ { l r ( i ) } \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { 1 } \left\{ \bar { A } _ { Q _ { u ^ { \prime } } } ( t ) = q _ { u ^ { \prime \prime } } , \mathcal { G } _ { t , u ^ { \prime } } , \top \right\} \right] } \\ & { \overset { l r ( i ) - 1 } { \le } \displaystyle \sum _ { u ^ { \prime } = 1 } ^ { l r ( i ) } \displaystyle \sum _ { u ^ { \prime } = u ^ { \prime } + 1 } ^ { l r ( i ) } \displaystyle \frac { 2 4 \log T } { \Delta _ { Q _ { u ^ { \prime } } , q _ { u ^ { \prime } } , q _ { u ^ { \prime } } } ^ { 2 } } . } \end{array}
$$

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: Our abstract and introduction clearly describe the scope and outline our main contributions.

Guidelines:

• The answer NA means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 7 discusses the limitation of this paper.

Guidelines:

• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate ”Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: We provide techniques clearly in the paper, and their detailed proofs are in Appendix.

Guidelines:

• The answer NA means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 5 provides the settings and results of our experiments carefully.

## Guidelines:

• The answer NA means that the paper does not include experiments.

• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The codes are uploaded in supplementary material.

Guidelines:

• The answer NA means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: Section 5 provides the settings and results of our experiments carefully.

Guidelines:

• The answer NA means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We provide the error bar in the results of experiments in Section 5.

## Guidelines:

• The answer NA means that the paper does not include experiments.

• The authors should answer ”Yes” if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No]

Justification: We do not provide specific information on the computer resources used, as most experiments on multi-armed bandits are lightweight.

Guidelines:

• The answer NA means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in the paper complies with NeurIPS Code of Ethics.

Guidelines:

• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: There is no societal impact of the work performed.

Guidelines:

• The answer NA means that there is no societal impact of the work performed.

• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

• The answer NA means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

Guidelines:

• The answer NA means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documenta tion provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

• The answer NA means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.