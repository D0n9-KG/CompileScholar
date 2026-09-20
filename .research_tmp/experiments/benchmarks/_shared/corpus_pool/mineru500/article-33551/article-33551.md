# Replicating Electoral Success\*

Kiran Tomlinson $^{1}$ , Tanvi Namjoshi $^{2}$ , Johan Ugander $^{3}$ , and Jon Kleinberg $^{4}$

$^{1}$ Microsoft Research   
$^{2}$ Princeton University   
$^{3}$ Stanford University   
$^{4}$ Cornell University

# Abstract

A core tension in the study of plurality elections is the clash between the classic Hotelling–Downs model, which predicts that two office-seeking candidates should position themselves at the median voter's policy, and the empirical observation that real-world democracies often have two major parties with divergent policies. Motivated in part by this tension and drawing from bounded rationality, we introduce a dynamic model of candidate positioning based on a simple behavioral heuristic: candidates imitate the policy of previous winners. The resulting model is closely connected to evolutionary replicator dynamics and, despite its simplicity, exhibits complex behavior and contrasts considerably with existing modeling approaches. For uniformly-distributed voters, we prove in our model that when there are k = 2, 3, or 4 candidates per election, any symmetric candidate distribution converges over time to a concentration of candidates at the center. With $k \geq 5$ or more candidates per election, however, we prove that the candidate distribution does not converge to the center. For initial distributions of $k \geq 5$ candidates without any extreme candidates, we prove a stronger statement than non-convergence, showing that the density in an interval around the center goes to zero. As a matter of robustness, our conclusions are qualitatively unchanged (though require different analyses) if a small fraction of candidates are not winner-copiers and are instead positioned uniformly at random in each election. Beyond our theoretical analysis, we illustrate our results in extensive simulations; for five or more candidates, we find a tendency towards the emergence of two clusters, a mechanism suggestive of Duverger's Law, the empirical finding that plurality leads to two-party systems. Our simulations also explore several variations of the model, including non-uniform voter distributions, other forms of noise, and replication with memory of earlier rounds of elections. In these simulated variants, we find the same general pattern: convergence to the center with four or fewer candidates, but not with five or more. Finally, we discuss the relationship between our replicator dynamics model and prior work on strategic equilibria of candidate positioning games.

# 1 Introduction

In a democracy, election outcomes determine the trajectory of public policy. A central question in the study of elections is therefore whether we can model which policies are electorally successful. However, elections are extremely complex, with layered interactions between voters, candidates, and the incentives that guide them. To understand the principles governing elections, we therefore need to pare down this complexity and begin with simple models. The literature around this topic traces its roots to Hotelling [33] and Downs [20]. In the Hotelling-Downs model, candidates compete for election in a one-dimensional policy space. Under the assumption that voters prefer candidates closer to them in policy space, two rational office-seeking candidates will adopt the policy of the median voter, since any other position receives strictly fewer

![](images/391bd1d2dbf16a51ac9ec0c5a8a1de211f9159e7a76da6ed72ba32ebf98d0012.jpg)

<details>
<summary>line</summary>

| x    | y (sample) | y (winners) |
| ---- | ---------- | ----------- |
| 0.0  | ●          | ●           |
| 0.5  | ●          | ●           |
| 1.0  | ●          | ●           |
</details>

elections at $t = 1$

![](images/30d004e1c7648bea44dab81fda1a17ba5ace2c07b90befc7a62019e9ea7228f1.jpg)

<details>
<summary>other</summary>

| x    | y     |
| ---- | ----- |
| 0.0  | 0.0   |
| 0.5  | 0.5   |
| 1.0  | 1.0   |
</details>

elections at $t = 2$

![](images/6981c7d319f4e1d42b7ad921dd2da54db48ecf5bd815e5e23ae1ac1dba561212.jpg)

<details>
<summary>line</summary>

| x    | y (green area) | y (black dots) |
| ---- | -------------- | -------------- |
| 0.5  | Peak           | ●              |
| 0.5  | ↓              | ●              |
| 0.5  | ●              | ●              |
| 0.5  | ●              | ●              |
| 0.5  | ●              | ●              |
| 0.5  | ●              | ●              |
| 0.5  | ●              | ●              |
| 0.5  | ●              | ●              |
| 0.5  | ●              | ●              |
| 0.5  | ●              | ●              |
| 0    | —              | —              |
| 1    | —              | —              |
</details>

elections at $t = 3$   
Figure 1: Replicator dynamics for candidate positioning with k = 3 candidates per election. The top row shows the winner distributions $F_{k,t}$ for each generation t, starting from a uniform distribution at t = 0, while the bottom row shows four example elections per generation. In each generation, candidates sample their positions from the winner distribution from the previous generation. Plurality winners (with voters uniform over [0, 1]) are indicated in green.

votes. Thus, the central prediction of the Hotelling–Downs model is that we should expect candidates to espouse near-identical moderate policies; in economic contexts, this is often called the principle of minimum differentiation $[23, 17]$ . However, this is not what we observe in modern democracies: countries using plurality often have two dominant parties with markedly different policies $[56, 30, 57]$ . Decades of research have attempted to reconcile this observed policy divergence with the intuitive arguments of Hotelling and Downs $[30, 53]$ , postulating additional factors like the threat of third-party entry $[54]$ or policy- rather than office-motivated candidates $[72]$ . Subsequent research has also expanded beyond two-candidate analysis to consider k-candidate elections $[16]$ .

The majority of this work has continued under the traditional assumption that candidates are rational and able to make strategically optimal decisions. However, the growing literature on bounded rationality $[65, 66]$ and decision-making heuristics $[69]$ , as well as the complexity of elections, casts doubt on whether this is likely in practice. In a notable exception to the literature on rational candidate positioning, Bendor et al. $[4]$ argue that heuristics play a crucial role in electoral strategy:

Campaigns are of chess like complexity—worse, probably; instead of a fixed board, campaigns are fought out on stages that can change over time, and new players can enter the game. Hence, cognitive constraints (e.g., the inability to look far down the decision tree, to anticipate your opponent's response to your response to their response to your new ad) inevitably matter. [...] Thus, political campaigns, like military ones, are filled with trial and error. A theme is tried, goes badly (or seems to), and is dropped. The staff hurries to find a new one, which seems to work initially and then weakens. A third is tried, and then a fourth. [...] In short, there are good reasons for believing that the basic properties of experiential learning—becoming more likely to use something that has worked in the past and less likely to repeat something that has failed—hold in presidential campaigns. [4, emphasis ours]

Our model. In this paper, we introduce a model of candidate positioning based on the above heuristic: candidates imitate success. We focus on plurality elections, where each voter casts one vote and the candidate with the most votes wins. We assume voters have 1-Euclidean preferences $[15, 24]$ , where voters and candidates occupy points in the unit interval $[0, 1]$ and voters prefer closer candidates. To represent a large voting population, our model uses a continuum of voters and continuous vote shares rather than discrete counts. Diverging from prior work, we model a large number of k-candidate elections that proceed in generations rather than an individual election or election sequence. In each generation, we assume that candidates

![](images/2b1ed6cb62c8c718ee88637ecb51848d8dc959f81abfe56935dae30b13d7bdcb.jpg)  
Figure 2: Replicator dynamics runs for $k = 2, \ldots, 7$ and 200 generations. Each plot shows 50 runs layered on top of each other, where each run simulates 100,000 elections per generation. We also use enhanced symmetry, a trick to keep the symmetry of the analytical model by reflecting copied points across 1/2 (discussed further in Section 5). Darker regions indicate higher candidate density; we use a log-scaled colormap to make low-density regions visible. As our theory establishes, the candidate distribution converges to the center for k = 2, 3, 4, but does not for $k \geq 5$ . The convergence is very fast for k = 2 and 3, but much slower for k = 4.

copy the policy position of a winner from the previous generation, a simple heuristic in line with Bendor et al.'s suggestion that candidates use strategies that worked in the past. This heuristic is also supported by a wealth of political science research arguing that the imitation of policies, especially electorally successful ones, is a major feature of politics $[63, 7, 25]$ . As with voters, our model uses a continuous distribution of candidate positions in each generation, which can be viewed as either capturing the expected behavior of a finite number of elections or as the infinite-election limit.

This simple assumption about candidate behavior (sample a position from the distribution of winners in the last election cycle) yields a mathematical model equivalent to the well-studied replicator dynamics from evolutionary biology $[67, 62]$ , which have also found widespread use in economics $[59, 47]$ . In the classic replicator dynamics, n strategies (or alleles) compete in a population, increasing in prevalence at a rate proportional to their average fitness in pairwise contests drawn from the current population. Our model arises from taking such dynamics and moving to a continuous strategy space with k-way interactions in discrete time (i.e., k-candidate elections), treating the plurality win probability as fitness; we therefore refer to it as replicator dynamics for candidate positioning.

In summary, then, our model operates in a sequence of generations; each generation involves a large number of identically distributed elections, and the candidates in a given generation are drawn from the distribution of winners of the previous generation's elections. Figure 1 provides a schematic visualization of the process with k = 3 candidates. While our model is phrased in terms of a large population of elections—just as the classic replicator dynamics models a population of organisms—there is a deep connection between replicator dynamics and reinforcement learning $[9, 5]$ , so our conclusions are likely to generalize to models of individual-level trial-and-error.

Our results. Our main technical contributions characterize the long-run behavior of the replicator dynamics for different values of k, the number of candidates per election. We find a dramatic qualitative change in the dynamics as the number of candidates k increases. For our analysis, we focus on the case in which the initial distribution of candidates is symmetric and has a continuous CDF and that voters are uniformly distributed over $[0,1]$ , but we find evidence in simulation that the same patterns hold with other symmetric voter distributions. When k = 2, we prove that the candidate distribution converges to a point mass at 1/2 under the replicator dynamics, just like rational candidates in the Hotelling–Downs model. However, we also prove in our model that the candidate distribution converges to the center for k = 3 and 4, in stark contrast to three- and four-candidate extensions of the Hotelling–Downs argument $[16]$ . Given the behavior for k = 2, 3, and 4, one might be tempted to hypothesize that the replicator dynamics always cause the candidate distribution to converge to the center. Surprisingly, we prove that the pattern ends there: for any $k \geq 5$ , we show that the candidate distribution does not converge to 1/2. See Figure 2 for simulations demonstrating the patterns that we characterize theoretically. These simulations reveal a tendency for candidate counts larger than 4 to result in two distinct clusters of policies (around 1/4 and 3/4 with uniform voters).

This is strongly reminiscent of Duverger's Law [22, 57], the observation that plurality elections tend towards two-party systems; it is striking that it emerges here from a model that does not include any explicit reward for clustering at points away from the center or any mechanisms like the threat of third-party candidates [8].

To strengthen this characterization of the long-run replicator dynamics, we show that our convergence results are robust to noise: even if a small fraction of candidates position themselves uniformly at random, we can still show (approximate) convergence to the center for k = 2, 3, 4 and non-convergence for $k \geq 5$ . While we are not able to theoretically derive the asymptotic distribution for $k \geq 5$ in general, we show that when the initial candidate distribution is supported only on $(1/4, 3/4)$ , the candidate density in an interval around 1/2 goes to 0. Additionally, we explore several variants of the model in simulation, including non-uniform voter distributions, noisy position-copying, memory of prior rounds of elections, and mixtures of candidate counts. $^{1}$ Across these variants, we observe the same general pattern: convergence to the center with up to four candidates, but not with five or more. For candidate counts k > 5 we sometimes see complex and chaotic finite-sample effects in simulation. We conclude by relating our replicator dynamics model back to traditional analyses of Nash equilibria in the style of Hotelling and Downs. The close relationship between replicator dynamics fixed points and Nash equilibria is well-known [32], but we argue that ignoring dynamics and focusing only on Nash equilibria leads to brittle conclusions. In particular, we show that different assumptions on voter behavior when candidates occupy the same points lead to dramatically different Nash equilibria than reported in prior work [16]; in contrast, this choice has no effect on our replicator dynamics results.

To summarize, our main finding is that a simple imitation heuristic can cause candidates to either converge to the median voter or to form two distinct parties, depending on how many candidates run in each election. Intuitively, this phenomenon is driven by the bogeyman of one-dimensional plurality elections: being flanked. If a candidate is stuck between two others, they lose votes from both the left and the right. When there are too many candidates all imitating previous moderate winners, only the leftmost or rightmost of them will avoid being flanked, making more extreme candidates more successful. However, with a small enough pool of opponents, the higher vote share a moderate can receive is worth the risk of ending up stuck between two others. This emerges naturally from our dynamics, without the need for strategic forethought. The surprising fact that falls out of our mathematical analysis is that when candidates are imitators rather than optimizers, the tipping point between the Hotelling–Downs centripetal force and the centrifugal force fueled by the problem of flanking occurs between four- and five-candidate elections.

# 1.1 Related work

Before diving into our theoretical analysis, we briefly summarize the literature in relevant areas.

One-shot candidate positioning games. Expanding on the two-candidate Hotelling–Downs foundation, subsequent research has explored higher-dimensional spaces $[55, 34]$ , more than two candidates $[16]$ , policy motivation $[72]$ , uncertainty about voter positions $[12]$ , and candidate valence (i.e., charisma or name recognition) $[31, 10]$ , among many other variations (see Osborne $[53]$ , Kurella $[38]$ for surveys). $^{2}$ Some models allow a third-party candidate to enter the race after the established candidates select their positions, which can lead to non-central two-party equilibria $[54, 70, 8]$ .

Dynamic models of candidate positioning. In addition to the work on one-shot games, there is also a literature on candidate positioning dynamics $[21]$ , although in contrast to our work, the focus of this literature has been on rational two-candidate contests. One notable early paper in this line of work studies a two-party system where the party which lost the previous election is allowed to reformulate its policy to maximize votes in the next election, which can yield predictable trajectories even in higher-dimensional policy

spaces $[37]$ . As in the one-shot literature described above, extensions of this model of two-party dynamics have added a variety of features, including policy motivation $[71, 13]$ , forward-looking parties $[58, 27, 49]$ , and—most closely related to our work—boundedly-rational candidates who are unable to exactly optimize their positions $[35, 36, 4]$ . Our paper is set apart from this prior research on electoral dynamics with bounded rationality in our replicator dynamics approach, and our success deriving analytical results for more than two candidates. We are aware of one paper $[39]$ combining a spatial model of elections and replicator dynamics, but the number of parties is fixed to two and the focus is instead on competition between office- and policy-motivated party members (“opportunists” and “militants”), where opportunists may defect to the other party.

Evolutionary game theory and replicator dynamics. Replicator dynamics $[67, 62, 32]$ were introduced to study the evolution of biological populations, but have since found much broader use. Economists have used evolutionary models—including replicator dynamics—to understand investment behavior $[6]$ , technological innovation $[61, 60]$ , and resource harvesting $[48]$ , among many other phenomena. See Friedman $[28]$ for an introduction to evolutionary game theory from an economic perspective and Nelson et al. $[47]$ , Safarzyńska and van den Bergh $[59]$ for surveys of evolutionary economics. Evolutionary models can even be justified without population-level evolution: models of individual-level learning can give rise to behavior equivalent to replicator dynamics $[9]$ ; see Bloembergen et al. $[5]$ for a survey of the connection between replicator dynamics and reinforcement learning. Evolutionary models are much less common in political science than in economics, but have been used to model the corruption of elected officials $[1]$ , coordination by voters $[43]$ , and party defection $[40]$ . Extensions of the classical replicator dynamics have explored the various modifications found in our model, including multi-way interactions $[29]$ , discrete time $[42]$ , and a continuous strategy space $[52, 14]$ .

Elections with strategic voters. Another line of research around strategic aspects of elections focuses instead on the strategic choices made by voters rather than candidates $[19, 68, 51]$ (with some papers combining both voter and candidate strategy $[26, 46]$ ). Dynamics have featured prominently in the strategic voting literature $[18, 11]$ —in particular, under the framework of iterative voting $[44, 41, 50]$ , where voters are allowed to update their votes in successive rounds until they are satisfied. Intriguingly, this style of voter-dynamics analysis can also produce conclusions paralleling Duverger's Law, where two major candidates emerge, despite using a completely different approach to ours $[45]$ . Evolutionary dynamics have also been applied to voter behavior to explain the paradox of voting (why do people vote when their probability of affecting the outcome is near zero?) $[64]$ .

# 2 Replicator dynamics for candidate positioning

We now formally introduce our model. We consider a one-dimensional policy space represented by the unit interval $[0,1]$ . Candidates and voters reside at points in the interval. To model a large population of voters, we treat the voting population as a continuum; for our theoretical analysis, we assume voters are uniform over $[0,1]$ , but we later relax this assumption in simulation. We assume voters have 1-Euclidean preferences $[24]$ —that is, they vote for the closest candidate. The vote share of a candidate i is the fraction of voters who vote for i. With uniform voters, the vote share of a candidate is equal to half the distance between the candidates to its left and right (a candidate adjacent to a boundary gets the entire vote share on its boundary side). Under plurality voting, the candidate with the largest vote share wins; in the case of tied maximum vote shares, the tie is broken uniformly at random.

Our replicator dynamics model of candidate positioning supposes that elections proceed in generations $t = 1, 2, \ldots$ , with (infinitely) many elections per generation. We assume the number of candidates in each election is fixed at k (later, we relax this assumption in simulation). The core idea of our model is that candidates in generation t chose their policy positions by copying the position of a winner from the previous generation t-1. More formally, let $F_{0}$ be the initial candidate distribution and let $F_{k,t}$ denote the distribution of winner positions in generation t with k candidates per election. We define $F_{k,0} = F_{0}$ for all k, although we

typically write $F_{0}$ since the initial distribution does not depend on k. In generation t, each election consists of k candidates with positions $X_{1,t},\ldots,X_{k,t}\sim F_{k,t-1}$ . We use $F_{k,t}(x)$ to denote the CDF of the winner distribution in generation t and $f_{k,t}(x)$ to denote the PDF. Let $\text{Plurality}(X_{1,t},\ldots,X_{k,t})$ be the position of the plurality winner given candidate positions $X_{1,t},\ldots,X_{k,t}$ and uniformly distributed voters.

Definition 1. Given an initial candidate distribution $F_{0}$ and a candidate count k, the replicator dynamics for candidate positioning (under plurality with uniform 1-Euclidean voters) are, for all t > 0,

$$
F _ {k, t} (x) = \operatorname * {P r} (\text { Plurality } (X _ {1, t}, \dots , X _ {k, t}) \leq x), \tag {1}
$$

$$
X _ {i, t} \sim F _ {k, t - 1}, \forall i = 1, \ldots , k.
$$

Or, in terms of the PDF:

$$
f _ {k, t} (x) = k \cdot \operatorname * {P r} (\text {Plurality} (x, X _ {2, t}, \dots , X _ {k, t}) = x) \cdot f _ {k, t - 1} (x). \tag {2}
$$

This model can be viewed through the lens of evolutionary replicator dynamics $[67, 62, 32]$ , although there are several differences from the classical case. In the classic replicator dynamics, there are n discrete strategies, each of which increases in frequency proportionally to how well that strategy performs against the current population. This proportionality is exactly what Equation (2) captures: strategy x increases in density proportional to its plurality win rate against the current population.

The main question we study is how the candidate distribution evolves over time under the replicator dynamics. We focus on cases where $F_{0}$ is symmetric about 1/2 and contains no point masses (i.e., the initial CDF $F_{0}(x)$ is continuous); we call such distributions symmetric and atomless. This ensures that the probability multiple candidates share the exact same point is 0, so we can ignore these cases for now. Since we assume $F_{0}$ is symmetric, all subsequent winner distributions are also symmetric by the symmetry of plurality with a uniform voter distribution—we lean heavily on this fact in our analysis. Some of our results require an additional assumptions on $F_{0}$ . We say $F_{0}$ is positive near 1/2 if $F_{0}(x) < 1/2$ for all x < 1/2 (equivalently, $f_{0}(x) > 0$ in an interval around 1/2); the symmetry of $F_{0}$ allows us to phrase definitions like this in terms of the left half of the unit interval, and it then applies equivalently to the right half as well. We define F to be the set of all symmetric and atomless distributions over [0,1] and $F^{+} \subset F$ to be the subset of such distributions which are also positive near 1/2.

In this section, we prove our main result piece-by-piece.

Theorem 1. Let $F_{0} \in F^{+}$ . For $k \in \{2, 3, 4\}$ , the candidate distribution converges to a point mass at 1/2 under the replicator dynamics. In contrast, for $k \geq 5$ , the candidate distribution does not converge to a point mass at 1/2.

Theorem 1 follows from Theorems 2 to 5. Our results for $k \in \{2, 3, 4\}$ give fine-grained characterizations of the dynamics, which imply convergence to the center: for k = 2, we derive a closed form for the CDF at generation t (Theorem 2), while for k = 3 and 4 we derive closed-form bounds for the CDF (Theorems 3 and 4). The negative portion of Theorem 1 offers less insight into the dynamics for $k \geq 5$ , only showing non-convergence to the center (Theorem 5), but in Section 4 we prove a stronger result in the special case where $F_{0}$ has no extreme candidates. All proofs omitted from this section for the sake of readability can be found in Appendix B.1.

# 2.1 $k = 2$

Two-candidate plurality with symmetric voters is simple: whichever candidate is closer to 1/2 has the larger vote share and wins. This simplicity allows us to fully characterize the dynamics with k = 2. In particular, we derive a closed form for the CDF $F_{2,t}(x)$ .

Theorem 2. Let $F_0 \in \mathcal{F}$ . For all $x < 1/2$ and $t \geq 0$ , $F_{2,t}(x) = [2 \cdot F_0(x)]^{2^t} / 2$ .

Proof. Let x < 1/2. Since the candidate closer to 1/2 wins with k = 2, $\text{Plurality}(X_{1,t}, X_{2,t}) \notin (x, 1 - x)$ if and only if both $X_{1,t} \notin (x, 1 - x)$ and $X_{2,t} \notin (x, 1 - x)$ , which occurs with probability $(2 \cdot F_{2,t-1}(x))^2$ . By symmetry, we then have $F_{2,t}(x) = \Pr(\text{Plurality}(X_{1,t}, X_{2,t}) \leq x) = (2 \cdot F_{2,t-1}(x))^2 / 2 = 2 \cdot F_{2,t-1}(x)^2$ . We can now prove the claim by induction on t. For the base case t = 0, $(2 \cdot F_0(x))^{2^0} / 2 = F_0(x)$ . For the inductive case $t \geq 1$ , applying the inductive hypothesis yields:

$$
F _ {2, t} (x) = 2 \cdot F _ {2, t - 1} (x) ^ {2} = 2 \left[ (2 \cdot F _ {0} (x)) ^ {2 ^ {t - 1}} / 2 \right] ^ {2} = [ 2 \cdot F _ {0} (x) ] ^ {2 ^ {t}} / 2.
$$

This result shows that for k = 2, the CDF at any point x < 1/2 with $F_{2,0}(x) < 1/2$ rapidly goes to 0—that is (apart from degenerate initial distributions) the candidate distribution converges toward a point mass at 1/2.

Corollary 1. Let $F_{0} \in F^{+}$ . For all x < 1/2, $\lim_{t \to \infty} F_{2,t}(x) = 0$ .

Note that by symmetry, it follows from such a statement that for all x > 1/2, $\lim_{t \to \infty} F_{2,t}(x) = 1$ .

# 2.2 $k = 3$

For k = 3, plurality becomes more complex: the winner need not be the closest to 1/2 (for instance, consider candidates at positioned at 1/3, 1/2, and 2/3). Nonetheless, we can still show that the candidate distribution converges to the center. To do so, we find an upper bound on $F_{3,t}(x)$ which goes to 0. The idea behind the proof is to enumerate cases where a candidate in an inner interval $(x, 1 - x)$ wins and add up the probability of these cases, as a function of $F_{k,t-1}(x)$ . For example, if there are two candidates in $[0, x)$ and one in $(1/2, 1 - x)$ , then the candidate in $(1/2, 1 - x)$ gets vote share greater than 1/2 and wins; this case occurs with probability $\binom{3}{2}F_{3,t-1}(x)^{2} \cdot (1/2 - F_{3,t-1}(x))$ . By symmetry, we can then transform this lower bound on the probability the winner is inside $(x, 1 - x)$ into an upper bound on $F_{3,t}(x)$ , the probability that the winner is in $[0, x]$ .

Theorem 3. Let $F_{0} \in F$ . For all x < 1/2 and t > 0,

$$
F _ {3, t} (x) \leq 3 / 4 \cdot F _ {3, t - 1} (x) + F _ {3, t - 1} (x) ^ {3}. \tag {3}
$$

This can be written as a looser closed form

$$
F _ {3, t} (x) \leq F _ {0} (x) \cdot \left[ 3 / 4 + F _ {0} (x) ^ {2} \right] ^ {t}. \tag {4}
$$

This result reveals that the candidate distribution for k = 3 also converges rapidly to the center. In particular, (4) shows that the CDF at any point x with $F_{0}(x) < 1/2$ decays exponentially towards 0 in t. This can also be seen by analyzing the cubic iterated map suggested by the upper bound (3), which converges to a stable fixed point at 0 for all initial values in $[0, 1/2)$ .

Corollary 2. Let $F_{0} \in F^{+}$ . For all x < 1/2, $\lim_{t \to \infty} F_{3,t}(x) = 0$ .

# 2.3 $k = 4$

As with k = 3, we derive an upper bound on the CDF which converges to 0. However, the bound suggests convergence is much slower for k = 4 than for 2 or 3 (which we will see later confirmed in simulation). The proof follows the same case-enumeration strategy as k = 3, but simple cases only show that the CDF is non-increasing in t for $x \in (1/3, 1/2)$ , giving us the following lemma.

Lemma 1. Let $F_0 \in \mathcal{F}$ . For all $x \in (1/3, 1/2)$ and $t \geq 0$ , $F_{4,t}(x) \leq F_{4,0}(x)$ .

By using Lemma 1, we can strengthen the case analysis with one additional case that tips the recurrence from breaking even to shrinking exponentially towards 0. However, the base of the exponential depends very strongly on x, increasing rapidly towards 1 near 1/2.

Theorem 4. Let $F_{0} \in F$ . For all $x \in (1/3, 1/2)$ and $t \geq 0$ ,

$$
F _ {4, t} (x) \leq F _ {0} (x) \cdot \left[ 1 - 4 (1 / 2 - F _ {0} (x / 3 + 1 / 3)) ^ {3} \right] ^ {t}. \tag {5}
$$

Note that $x/3 + 1/3$ is the point two-thirds of the way from x to 1/2. As long as $F_{0}(x/3 + 1/3) < 1/2$ , which is true for any x < 1/2 if $F_{0}$ is positive near 1/2, this result shows that the CDF left of 1/2 decays to 0 as t grows. That is, the candidate distribution converges to the center again.

Corollary 3. Let $F_{0} \in F^{+}$ . For all x < 1/2, $\lim_{t \to \infty} F_{4,t}(x) = 0$ .

# 2.4 $k\geq 5$

In contrast to k = 2, 3, 4, we now show that for any larger k, the candidate distribution does not converge to the center. The proof is based on the following observation.

Lemma 2. For any k, if all candidates are in $(1/4,3/4)$ , then only the left- or rightmost candidate can win with uniform voters.

Proof. Suppose all candidates are in $(1/4, 3/4)$ . Any candidate between two others gets vote share less than $(1/2)/2 = 1/4$ , since no two candidates are distance 1/2 or greater apart. Meanwhile, the left- and rightmost candidates each get vote share >1/4. ☐

Intuitively, if the candidate distribution starts converging to the center, then all candidates will likely be inside $(1/4, 3/4)$ , at which point only the most extreme candidates can win. When k is sufficiently large (i.e., $\geq 5$ ), the left- and rightmost candidates are likely on opposite sides and farther from 1/2 than the average candidate. This results in a centrifugal force preventing further progress towards the center. Formally, we prove the following theorem.

Theorem 5. Let $F_0 \in \mathcal{F}$ . For any $k \geq 5$ , there exists some $x < 1/2$ such that $\lim_{t \to \infty} F_{k,t}(x) \neq 0$ . That is, the candidate distribution does not converge to a point mass at $1/2$ .

More specifically, our proof assumes for a contradiction that the distribution converges to 1/2, so at some generation $t^{*}$ , the probability mass left of 1/4 must be less than some small $\alpha$ . We then show that the CDF can never decrease at $F_{k,t^{*}}^{-1}(1/4)$ after generation $t^{*}$ , since all candidates will likely be inside (1/4, 3/4), causing only the most extreme candidates to win. This contradicts convergence to the center.

# 3 Replicator dynamics with noise

So far, we have assumed that all candidates copy winner positions from the previous generation. We now show that our results still hold in approximate forms if some of the candidates violate this behavior and instead position themselves uniformly at random. This demonstrates a way in which the convergence of the model is robust to alternative specifications.

Definition 2. Given an initial candidate distribution $F_{0}$ , a candidate count k, and a noise level $\epsilon \in (0,1]$ , the replicator dynamics for candidate positioning with $\epsilon$ -uniform noise (under plurality with uniform 1-Euclidean voters) are, for all t > 0,

$$
F _ {k, t} ^ {\epsilon} (x) = \operatorname * {P r} (\text { Plurality } (X _ {1, t} ^ {\epsilon}, \dots , X _ {k, t} ^ {\epsilon}) \leq x), \tag {6}
$$

$$
F _ {k, 0} ^ {\epsilon} = F _ {0}
$$

$$
X _ {i, t} ^ {\epsilon} \sim \left\{ \begin{array}{l l} \text {Uniform} (0, 1) & \text {w.p.} \epsilon , \\ F _ {k, t - 1} ^ {\epsilon} & \text {w.p.} 1 - \epsilon . \end{array} \right.
$$

As in the noiseless case, we show that the candidate distribution converges to the center under the dynamics with $\epsilon$ -uniform noise for k = 2, 3, 4 but do not for $k \geq 5$ . However, since $\epsilon$ -uniform noise introduces non-central candidates at every t, we need to relax the convergence requirement. The idea behind our notion of approximate convergence is that if we make the noise sufficiently small, then the distribution should get arbitrarily close to a point mass at 1/2. That is, the CDF at any point x < 1/2 eventually goes below any positive threshold c, for sufficiently small $\epsilon > 0$ .

Definition 3. Let $F_0 \in \mathcal{F}$ . The candidate distribution approximately converges to the center under the replicator dynamics with $\epsilon$ -uniform noise if for all $x \in [0,1/2)$ and $c > 0$ , there exists some $\epsilon_{\max} > 0$ such that if $\epsilon \in (0, \epsilon_{\max}]$ , then $\limsup_{t \to \infty} F_{k,t}^{\epsilon}(x) < c$ .

We now give the analogue of our main result with $\epsilon$ -uniform noise. One additional benefit of adding noise is that we no longer need to assume $F_{0}$ is positive near 1/2.

Theorem 6. Let $F_{0} \in F$ . For $k \in \{2, 3, 4\}$ , the candidate distribution approximately converges to the center under replicator dynamics with $\epsilon$ -uniform noise. In contrast, for all $k \geq 5$ , the candidate distribution does not approximately converge to the center.

Theorem 6 follows from Theorems 7 to 10. See Appendix B.2 for proofs omitted from this section.

# 3.1 k = 2

We first show that the replicator dynamics with $\epsilon$ -uniform noise approximately converge to the center with two candidates. In fact, we can exactly characterize the limiting candidate distribution for k=2. As before with k=2, whichever candidate is closer to 1/2 wins, but now these candidates can either be winner-copiers or randomly positioned. The idea behind the proof is to find an iterated map for $F_{2,t}^{\epsilon}(x)$ and find the stable fixed point it converges to for x<1/2, which we show is smaller than $\epsilon$ .

Theorem 7. Let $F_0 \in \mathcal{F}$ . For any $\epsilon \in (0,1)$ and $x \in [0,1/2)$ with $\epsilon$ -uniform noise,

$$
\lim _ {t \to \infty} F _ {2, t} ^ {\epsilon} (x) = \frac {1 - 4 x \epsilon (1 - \epsilon) - \sqrt {1 - 8 \epsilon x (1 - \epsilon)}}{4 (1 - \epsilon) ^ {2}} \leq \epsilon . \tag {7}
$$

Proof. Let x < 1/2 and $p = F_{2,t-1}^{\epsilon}(x)$ . Each candidate in generation t is drawn from $F_{2,t-1}^{\epsilon}$ w.p. $(1-\epsilon)$ and from Uniform(0,1) w.p. $\epsilon$ (call such candidates uniform). Uniform candidates fall outside $(x,1-x)$ w.p. 2x, while non-uniform candidates fall outside $(x,1-x)$ w.p. 2p by symmetry. A winner in generation t is not in $(x,1-x)$ if and only if both candidates fall outside this interval, which thus occurs with probability

$$
\begin{array}{l} \operatorname * {P r} (X _ {1, t} ^ {\epsilon} \notin (x, 1 - x), X _ {2, t} ^ {\epsilon} \notin (x, 1 - x)) = \underbrace {(1 - \epsilon) ^ {2} (2 p) ^ {2}} _ {\text {neither uniform}} + \underbrace {2 \epsilon (1 - \epsilon) (2 x) (2 p)} _ {\text {one uniform}} + \underbrace {\epsilon^ {2} (2 x) ^ {2}} _ {\text {both uniform}} \\ = 4 p ^ {2} (1 - \epsilon) ^ {2} + 8 p x \epsilon (1 - \epsilon) + 4 x ^ {2} \epsilon^ {2}. \\ \end{array}
$$

By symmetry, we then have

$$
\begin{array}{l} F _ {2, t} ^ {\epsilon} (x) = \operatorname * {P r} (X _ {1, t} ^ {\epsilon} \notin (x, 1 - x), X _ {2, t} ^ {\epsilon} \notin (x, 1 - x)) / 2 \\ = 2 p ^ {2} (1 - \epsilon) ^ {2} + 4 p x \epsilon (1 - \epsilon) + 2 x ^ {2} \epsilon^ {2}. \tag {8} \\ \end{array}
$$

The claim then follows from the following technical lemma, proved in Appendix B.2.

Lemma 3. For all initial $p \in [0,1/2]$ , $\epsilon \in (0,1)$ , and $x \in [0,1/2)$ , the quadratic iterated map $p' = 2p^2(1 - \epsilon)^2 + 4px\epsilon(1 - \epsilon) + 2x^2\epsilon^2$ converges to the fixed point $p^* = \frac{1 - 4x\epsilon(1 - \epsilon) - \sqrt{1 - 8\epsilon x(1 - \epsilon)}}{4(1 - \epsilon)^2} \leq \epsilon$ .

![](images/4cc3d09cad0b051e2eacc673fc253b1765bab4cfd1f54622b62671f6d90d8b9b.jpg)

This result implies approximate convergence to the center: we can simply take $\epsilon_{\mathrm{max}} < c$ and we then have $\lim_{t\to \infty}F_{2,t}^{\epsilon}(x)\leq \epsilon_{\mathrm{max}} < c$ .

Corollary 4. Let $F_0 \in \mathcal{F}$ . For $k = 2$ , the candidate distribution approximately converges to the center under replicator dynamics with $\epsilon$ -uniform noise.

# 3.2 k = 3

We now show approximate convergence to the center for k = 3 with $\epsilon$ -uniform noise. As in the noiseless case, we cannot fully characterize the limiting distribution, but we are able to bound it for $\epsilon < 1/3$ . The proof repeats the case analysis from the proof of Theorem 3 but with $\epsilon$ -uniform candidates, which yields a cubic iterated map. We then bound the attracting fixed point of this map, as we did for k = 2.

Theorem 8. Let $F_0 \in \mathcal{F}$ . For any $\epsilon \in (0,1/3)$ and $x \in [0,1/2)$ , $\limsup_{t \to \infty} F_{3,t}^{\epsilon}(x) \leq 1.5\epsilon$ .

As with $k = 2$ , this shows that for any $c > 0$ , we can pick a small enough $\epsilon$ (i.e., $\epsilon < \min\{1/3, 2/3 \cdot c\}$ ) so that $\limsup_{t \to \infty} F_{3,t}^{\epsilon}(x) < c$ .

Corollary 5. Let $F_0 \in \mathcal{F}$ . For $k = 3$ , the candidate distribution approximately converges to the center under replicator dynamics with $\epsilon$ -uniform noise.

# 3.3 k = 4

As with two and three candidates, we can also show approximate convergence to the center for replicator dynamics with $\epsilon$ -uniform noise and four candidates. However, for $k = 4$ , the bound on $\epsilon$ required for convergence depends on the point's distance from $1/2$ —just as the convergence rate did in the noiseless case. We begin with the noisy analogue of Lemma 1.

Lemma 4. Let $F_{0}\in\mathcal{F}$ . With $\epsilon$ -uniform noise, for any $\epsilon\in(0,1]$ , $x\in(1/3,1/2)$ , and $t>0$ ,

$$
F _ {4, t} ^ {\epsilon} (x) \leq \epsilon x + (1 - \epsilon) F _ {4, t - 1} ^ {\epsilon} (x).
$$

Thus, $F_{4,t}^{\epsilon}(x) \leq \max \{x, F_{4,0}^{\epsilon}(x)\}$ .

We can then apply the same strategy as we did in the noiseless case and analyze the resulting iterated map as we have done for k = 2 and 3.

Theorem 9. Let $F_0 \in \mathcal{F}$ . For any $\epsilon \in (0,1]$ and $x \in (1/3,1/2)$ , let $\beta = 1/2 - \epsilon(x/3 + 1/3) - (1 - \epsilon) \max\{x/3 + 1/3, F_0(x/3 + 1/3)\}$ . Then $\beta \in (0,1/2]$ and $\limsup_{t \to \infty} F_{4,t}^{\epsilon}(x) \leq \frac{1}{8\beta^3}\epsilon$ .

As long as we make $\epsilon$ sufficiently small (relative to $8\beta^{3}$ ), the CDF at x < 1/2 eventually goes below any desired threshold c—although the closer x is to 1/2, the smaller $\beta$ becomes, and likewise the required $\epsilon$ .

Corollary 6. Let $F_0 \in \mathcal{F}$ . For $k = 4$ , the candidate distribution approximately converges to the center under replicator dynamics with $\epsilon$ -uniform noise.

# 3.4 $k\geq 5$

Now that we have seen approximate convergence to the center for $k = 2,3,4$ , we prove that this does not happen for any higher $k$ . The argument uses the same idea as in Theorem 5 ( $k \geq 5$ without noise): that when all candidates are in (1/4, 3/4), only the left- and rightmost candidates can win. As long as we make $\epsilon$ sufficiently small, the exact same approach applies, albeit with some added care to account for randomly positioned candidates.

Theorem 10. Let $F_0 \in \mathcal{F}$ . For $k \geq 5$ , the candidate distribution does not approximately converge to the center under replicator dynamics with $\epsilon$ -uniform noise.

# 4 Positive results for $k \geq 5$ with no extreme candidates

In the previous sections, our results for $k \geq 5$ have been negative, showing the candidate distribution does not converge to the center, but without indicating what the distribution converges to instead. While simulations

Table 1: Base of the exponential from Theorem 12 for small k. 

<table><tr><td>k</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td> $k(1/2)^{k-2}$ </td><td>2</td><td>3/2</td><td>1</td><td>5/8</td><td>3/8</td></tr></table>

indicate a tendency towards a two-spike equilibrium, we have not been able to theoretically characterize the limiting distribution in general for $k \geq 5$ , either with or without noise. However, Lemma 2 enables us to analyze the dynamics for $k \geq 5$ (without noise) in the special case that $F_{0}$ has no extreme candidates, with support only on $(1/4, 3/4)$ . In this setting, the dynamics are much simpler, as only the left- and rightmost candidates can win. The same type of argument we used before for $k \geq 5$ then provides a positive result, showing that the candidate distribution converges to one with zero mass in an interval around 1/2. In contrast, our central convergence results for $k \in \{2, 3, 4\}$ still hold in this special case. Proofs of results in this section can be found in Appendix B.3.

Theorem 11. Suppose $F_0 \in \mathcal{F}$ is supported on $(1/4, 3/4)$ . Let $\ell = [1 - \sqrt{3/7}] / 2 = 0.172 \ldots$ . For $k \geq 5$ and $x \in (F_0^{-1}(\ell), 1/2)$ , $\lim_{t \to \infty} F_{k,t}(x) = 1/2$ .

When $F_{0}$ is $\text{Uniform}(1/4,3/4)$ , note that $F_{0}^{-1}(0.172\ldots)=0.336\ldots$ , so Theorem 11 implies that as $t\to\infty$ , the candidate density in [0.34,0.66] goes to 0 for $k\geq5$ . With no extreme candidates, we can also precisely characterize the density of the candidate distribution at 1/2 using a simple argument. Since only the left- or rightmost candidates can win, a candidate i at 1/2 only wins if the other candidates are all on the left or on the right. By symmetry, this occurs with probability $2\cdot(1/2)^{k-1}=(1/2)^{k-2}$ . Accounting for the k-fold symmetry in choosing candidate i and applying an inductive argument based on Equation (2) then gives the following result.

Theorem 12. Suppose $F_0 \in \mathcal{F}$ is supported on $(1/4, 3/4)$ . For any $k \geq 2$ and $t \geq 0$ ,

$$
f _ {k, t} (1 / 2) = f _ {0} (1 / 2) \cdot \left[ k (1 / 2) ^ {k - 2} \right] ^ {t}. \tag {9}
$$

With support on $(1/4,3/4)$ , the behavior of the density at 1/2 therefore depends on whether $k(1/2)^{k-2}$ is smaller or larger than 1. This quantity is smaller than 1 for $k \geq 5$ , larger than 1 for k = 2, 3 and equal to 1 for k = 4 (see Table 1). The larger k is, the more rapidly the density at 1/2 goes to 0. This simple argument reveals a mechanism driving the k < 5 vs $k \geq 5$ divide: $(1/2)^{k-2}$ is exactly the probability that a central candidate is not flanked. This probability decreases rapidly with k and is counterbalanced at first by the increasing number of candidates k who can be at the center—by as soon as $k \geq 5$ , the exponentially low probability of being the left- or rightmost candidate at the center becomes too small.

Theorem 12 is particularly interesting for k = 4, since we know the distribution converges to a point mass at 1/2, but the density at 1/2 stays constant at $f_{0}(1/2)$ when $F_{0}$ is supported on $(1/4, 3/4)$ . These seemingly contradictory facts are both possible since the distribution converges by accumulating more and more mass in two spikes on each side of 1/2 that approach the center arbitrarily closely. Theorem 12 thus highlights how k = 4 is a marginal tipping point which just barely converges to the center—a phenomenon also hinted at by the marginal nature of our k = 4 case analysis in Lemma 1 and Theorem 4: the analysis in Lemma 1 just breaks even, with the low-probability case in Theorem 4 needed to tip the scales. As we saw in Figure 2, this manifests in simulation as slow convergence to the center for k = 4.

# 5 Simulations

Having established our primary theoretical results, we demonstrate them in simulation. $^{3}$ To do so, we use Monte Carlo sampling, simulating a large number of elections per generation (100,000) and using the winners to approximate $F_{k,t}$ . We initialize $F_{0}$ to be uniform. With this basic setup, we observe some effects due purely

![](images/6c07f45a7ada9b86bf75c32cf4b51e1d51ebb6c1c2437a7fd79c411e31fa499b.jpg)

Figure 3: Replicator dynamics runs with 0.01-uniform noise for $k = 2, \ldots, 7$ and 200 generations, using enhanced symmetry, 50 trials per plot, and 100,000 elections per generation. The behavior is qualitatively identical to the model without noise (Figure 2).   
![](images/ce6248f5f7a29af9becc981254d0573ff65eae385590130a04cc73a250ab6f4d.jpg)  
Figure 4: Replicator dynamics runs with no noise (top row) and 0.01-uniform noise (bottom row) for larger candidate counts k and using enhanced symmetry. Other settings are identical to Figure 3, with 50 runs shown in each plot. As the theory predicts, the candidate distribution does not converge to the center; but the exact behavior varies.

to sampling, such as oscillations due to small asymmetries in the Monte Carlo samples. In contrast, our theoretical model revolves around an evolving density which by definition is always symmetric. To preserve symmetry while maintaining the same evolving distribution, we configure our Monte Carlo sampling to use a trick we term enhanced symmetry, mirroring each copied position across 1/2 with probability 1/2 in every generation.

Figure 3 shows 50 aggregated simulation runs for $k = 2, \ldots, 7$ using enhanced symmetry and 0.01-uniform noise (recall Figure 2 for equivalent plots without noise). See Appendix A for additional plots without enhanced symmetry and showing a single trial—these results follow the same general patterns across values of k. The candidate distributions evolve exactly as we would expect from our theory: rapid convergence to the center for k = 2 and 3, slow convergence for k = 4, and non-convergence to the center for $k \geq 5$ ; both with and without $\epsilon$ -uniform noise. Interestingly, k = 5, 6, and 7 show a tendency to converge towards two point masses at 1/4 and 3/4—but this phenomenon is sensitive to sampling asymmetries for k = 6 and 7 (see Figures 8 and 10 in Appendix A). In Section 4, we will see some theoretical justification for this two-spike behavior in a special case when $F_{0}$ has no extreme candidates. In Figure 4, we also show simulations with larger candidate counts, from 8 to 50. In line with Theorem 5, the distributions with large k do not converge to the center. However, we see some surprising differences depending on k; for instance, the asymptotic distribution with k = 8 appears to have four clusters rather than two (mysteriously, all other large values of k we have tested tend towards two clusters, at least with enhanced symmetry). In Appendix A, we provide several additional visualizations: Figure 11 shows simulations with only 50 elections per generation,

![](images/1819161e0f04359349182444f3cd9cee226db4c2e29397d069dd76782b0df79b.jpg)

<details>
<summary>line</summary>

| t   | x = 0.10 | x = 0.40 | x = 0.47 |
| --- | -------- | -------- | -------- |
| 0   | 0.1      | 0.4      | 0.48     |
| 2   | 0.0      | 0.2      | 0.4      |
| 4   | 0.0      | 0.05     | 0.2      |
| 6   | 0.0      | 0.0      | 0.1      |
| 8   | 0.0      | 0.0      | 0.0      |
</details>

![](images/6793406ca257196b99ff8f603925b3928eccf5649e06b9d0b96a9e7c058afa55.jpg)

<details>
<summary>line</summary>

| t  | simulation | Thm 2.5 (bound) |
|----|------------|-----------------|
| 0  | 0.47       | 0.40            |
| 5  | 0.35       | 0.30            |
| 10 | 0.20       | 0.20            |
| 15 | 0.10       | 0.15            |
| 20 | 0.05       | 0.10            |
</details>

![](images/d3e84aa5b733ba7be4ebe7174c31ccd36b72ad7eebb11413100e2f7588c2b44f.jpg)

<details>
<summary>line</summary>

| t  | simulation | Thm 2.8 (bound) |
|----|------------|-----------------|
| 0  | 0.47       | 0.40            |
| 10 | 0.47       | 0.40            |
| 20 | 0.47       | 0.40            |
| 30 | 0.47       | 0.40            |
| 40 | 0.47       | 0.40            |
</details>

Figure 5: Simulations demonstrating our convergence results Theorems 2 to 4, showing the simulated candidate distribution CDF at various points x alongside the theoretical predictions. The simulations use 50 trials with 100,000 elections per generation, no noise, and enhanced symmetry. The theorems get progressively weaker: Theorem 2 provides an exact characterization of the two-candidate dynamics, while Theorems 3 and 4 give upper bounds that converge to 0.   
![](images/884475e576c7e39379b2c21ba1a3cba58c92d5d100ce3d921484bcc43f76834c.jpg)

<details>
<summary>line</summary>

| t    | x=0.4, ε=0.33 | x=0.2, ε=0.33 | x=0.2, ε=0.1 | x=0.1, ε=0.1 |
| ---- | ------------- | ------------- | ------------ | ------------ |
| 0    | ~0.1          | ~0.01         | ~0.001       | ~0.0001      |
| 5    | ~0.05         | ~0.01         | ~0.001       | ~0.0001      |
| 10   | ~0.05         | ~0.01         | ~0.001       | ~0.0001      |
| 15   | ~0.05         | ~0.01         | ~0.001       | ~0.0001      |
</details>

![](images/37ea57d2049998fd48c0ecee93bd7d30d9d62e5429a3456f889d2c44091d64d0.jpg)

<details>
<summary>line</summary>

| t    | x=0.4, ε=0.33 | x=0.2, ε=0.33 | x=0.2, ε=0.1 | x=0.1, ε=0.1 |
| ---- | ------------- | ------------- | ------------ | ------------ |
| 0    | ~10⁻¹         | ~10⁻¹         | ~10⁻¹        | ~10⁻¹        |
| 5    | ~10⁻¹         | ~10⁻²         | ~10⁻³        | ~10⁻³        |
| 10   | ~10⁻¹         | ~10⁻²         | ~10⁻⁴        | ~10⁻⁴        |
| 15   | ~10⁻¹         | ~10⁻²         | ~10⁻⁴        | ~10⁻⁴        |
| 20   | ~10⁻¹         | ~10⁻²         | ~10⁻⁴        | ~10⁻⁴        |
| 25   | ~10⁻¹         | ~10⁻²         | ~10⁻⁴        | ~10⁻⁴        |
| 30   | ~10⁻¹         | ~10⁻²         | ~10⁻⁴        | ~10⁻⁴        |
</details>

![](images/843afbcb29ff5c4fb0cc676c87056dc4e4f07fdceae1e268e0e5f99e6b829bae.jpg)

<details>
<summary>line</summary>

| t  | Thm 3.10 (bound) | simulation |
|----|------------------|----------|
| 0  | 1.0              | 1.0      |
| 20 | ~1e-3            | ~1e-3    |
| 40 | ~1e-5            | ~1e-5    |
| 60 | ~1e-6            | ~1e-6    |
</details>

Figure 6: Simulations demonstrating Theorems 7 to 9, showing the simulated candidate distribution CDF at various points x and various noise levels $\epsilon$ alongside the theoretical asymptotic bounds. The simulations use 50 trials with 100,000 elections per generation and enhanced symmetry. As in the noiseless case, the theorems get progressively weaker as k increases. For k = 3, the asymptotic bounds depend only on $\epsilon$ , while the bounds for k = 4 and the exact limit for k = 2 depend on both $\epsilon$ and x.

demonstrating that our findings still hold in a small-sample setting; Figure 12 shows large-k simulations without enhanced symmetry; finally, Figure 13 demonstrates the no-extremes setting of Theorem 11, starting from Uniform(1/4, 3/4).

In addition to confirming the picture painted by our theory, we also use simulations to explore how tight our bounds are—although our core focus is on characterizing the qualitative behavior of the model rather than achieving the tightest bounds on convergence rate. In Figure 5, we demonstrate the exact result from Theorem 2 and the closed-form upper bounds on the candidate distribution CDF from Theorems 3 and 4. The bounds on convergence rates for k = 3 and 4 are indeed loose, as there are several ways that central candidates can win that are not easily captured by our case analysis. In Figure 6, we demonstrate Theorems 7 to 9. Note that these results are all asymptotic, characterizing or bounding the limit of the candidate distribution CDF as $t \rightarrow \infty$ , whereas the results in Figure 5 hold for finite t. Additionally, the results with $\epsilon$ -uniform noise depend on the value of $\epsilon$ , so we experiment with several different values. Again, we see that our bounds from Theorems 8 and 9 are loose, but nonetheless hold and are non-trivial. Moreover, the exact result in Theorem 7 is nicely confirmed by simulation.

# 6 Variants of the replicator dynamics

We now demonstrate in simulation that the qualitative picture provided by our results from Theorem 1 is robust to different specifications of the model. At a high level, our model of candidate positioning consists of the following components: (1) a fixed voter distribution, (2) a subset of previous candidate positions which new candidates imitate, and (3) a rule for sampling from those previous positions. In the basic model, the voter distribution is uniform, the imitated positions are plurality winners from the previous generation, and the sampling rule chooses uniformly from those winners. Adding $\epsilon$ -uniform noise modifies the sampling rule to sometimes pick uniformly random positions. In simulation, we explore natural variations of each of these modeling components: changing the voter distribution, adding plurality winners from earlier generations or runners-up to the imitation pool, and adding copying errors to the sampling rule or sampling different numbers of candidates across elections. See Appendix C for formal definitions of the variants in this section.

Non-uniform voters. We explore our replicator dynamics with symmetric unimodal and bimodal voter distributions. In Figure 7, we show results with three voter distributions: the unimodal distribution Beta(2,2), the bimodal, extreme voter distribution Beta(0.5,0.5), and a bimodal double Weibull distribution [2] with shape 4, location 0.5, and scale 0.3 (see Figure 14 in Appendix A.1 for visualizations of these distributions). The basic pattern from Theorem 1 continues to hold with these voter distributions. However, when voters are Beta(2,2)-distributed, the two clusters at k = 5 are significantly closer to the center.

Memory. In the basic model, candidates only copy the positions of winners in the previous generation. However, real-world candidates will likely have memory of earlier winners, so in this variant, we allow candidates to sample from winner positions in any of the last m generations. In Figure 7, we see that adding m = 2 generations of memory for candidates still maintains the pattern from Theorem 1. The results for m = 3 are extremely similar (see Figure 15 in Appendix A).

Perturbation noise. With perturbation noise, each candidate slightly deviates from the position they copy, as if their imitation is imperfect. In our simulations, we add Gaussian noise with mean 0 and variance $\sigma^{2}$ to each copied position. Figure 7 shows that the candidate distribution with a small amount of perturbation noise ( $\sigma^{2}=0.005$ ) converges to the center for k=2,3,4 but does not for $k\geq5$ . However, with sufficient noise, higher values of k form a single central cluster; we see this in Figure 7 with k=6 and $\sigma^{2}=0.01$ . Additionally, for $k\geq6$ the behavior varies significantly across runs without enhanced symmetry. We even observe phenomena such as party movement, divergence, and extinction, particularly for higher values of k (see Figure 16 in Appendix A).

Variable candidate counts. In real-world elections, we might expect different numbers of candidates to run in different elections, but our model keeps the candidate count k constant. In this variant, we allow elections in each generation to have a mixture of several candidate counts, where candidates copy from winner positions across all k in the previous generation. We find that our results interpolate smoothly to this setting: when most elections have fewer than five candidates, we see convergence to the center, but not when most elections have $k \geq 5$ (see Figure 7, where we simulate an equal mixture of the listed candidate counts in each generation, with 50,000 elections per k). See Figure 17 in Appendix A for a more fine-grained experiment in which we smoothly vary the proportions of elections with k = 3, 4, 5.

Top-h copying. Finally, we explore a variant where candidates in generation t choose a position to copy from the pool of candidates with the top-h highest vote shares in generation t-1, rather than only winners (i.e., h=1). In simulation, top-h copying is the only variant which strays from the dichotomy we establish in Theorem 1—perhaps unsurprisingly, given that our result is about copying winners. For k=3,4, when h=2 the candidate distribution does not converge to the center and instead ends up as k=5 usually does, with two clusters (see Figure 7, bottom right). For h=3, the candidate distribution does not even appear to converge towards point masses (see Figure 18 in Appendix A.1). These simulations suggest that our

![](images/9695f0bcf4613af2b25e32e036b1e705f9a204a2d5d03a73f246974afdd4a762.jpg)

<details>
<summary>heatmap</summary>

| k   | Beta(2, 2) voters | Beta(0.5, 0.5) voters | dWeibull(4, .5, .3) voters | Memory, m = 2 |
| --- | ----------------- | --------------------- | -------------------------- | -------------- |
| 3   | 0.5               | 0.5                   | 0.5                        | 0.5            |
| 4   | 0.5               | 0.5                   | 0.5                        | 0.5            |
| 5   | 0.5               | 0.5                   | 0.5                        | 0.5            |
| 6   | 0.5               | 0.5                   | 0.5                        | 0.5            |
</details>

![](images/2a15f4fd1c7565d1b9dba7d2440d5b2692e05d5c8b156995a32844781ab24c10.jpg)

<details>
<summary>heatmap</summary>

| k value | t range | Perturb. noise, σ² = 0.005 | Perturb. noise, σ² = 0.01 | Multiple ks | Top-2 copying |
|---------|---------|-----------------------------|-----------------------------|-------------|---------------|
| 3       | 0–200   | ~0.5                        | ~0.5                        | ~0.5        | ~0.5          |
| 4       | 0–200   | ~0.5                        | ~0.5                        | ~0.5        | ~0.5          |
| 5       | 0–200   | ~0.5                        | ~0.5                        | ~0.5        | ~0.5          |
| 6       | 0–200   | ~0.5                        | ~0.5                        | ~0.5        | ~0.5          |
</details>

Figure 7: Variants of the replicator dynamics. Each plot shows 50 trials with no enhanced symmetry. Left column, top to bottom: three different voter distributions and 2 generations of memory. Right column, top to bottom: perturbation noise with $\sigma^{2}=0.005$ and 0.01, variable candidate counts, and top-2 copying. Except for top-2 copying, all of the variants converge to the center for k<5. Additionally, sufficiently high perturbation noise can cause a central cluster to form for high k.

central finding (convergence to the center for k < 5) is a result of copying the positions of plurality winners specifically, and the dynamics under this heuristic.

# 7 Relationship to Nash equilibria of one-shot games

We now take a step back and examine the relationship between our dynamics and prior research on strategic positioning. As we discussed, much of the literature on candidate positioning has focused on one-shot games rather than dynamics $[53, 8, 38]$ , as in the Hotelling–Downs model. In our 1-Euclidean setting with uniform voters, the Hotelling–Downs equilibrium has both candidates positioned at 1/2—which as we showed, is also the attracting distribution of the replicator dynamics with k = 2. Indeed, it is well-known that Nash equilibria of one-shot games are fixed points of the corresponding replicator dynamics $[32]$ , but replicator

dynamics fixed points may not be Nash equilibria. We can see this intuitively in our setting by noting that a distribution F is a (symmetric, mixed-strategy) Nash equilibrium if no strategy does better against F than sampling from F, while F is a fixed point of the replicator dynamics if no strategy drawn from F does better against F than sampling from F. For F with full support, symmetric mixed-strategy Nash equilibria and replicator dynamics fixed points thus coincide $[3]$ . However, Nash equilibria can be unstable under the dynamics—and even if they are attractors, their basins of attraction may be negligible.

Before analyzing Nash equilibria, we first need a brief digression to address what happens when multiple candidates occupy the same point—we call these positional ties. Since we have so far assumed that candidate distributions are atomless, our analyses of the replicator dynamics has avoided this issue: with an atomless candidate distribution, positional ties occur with probability 0. One option for handling positional ties is to suppose that candidates fail to position themselves exactly at the same point and imagine that there is some infinitesimal jitter in their positions which determines a left–right order. Alternatively, we could suppose that candidates are in fact precisely at the same point, forcing voters to make an arbitrary choice between them.

Definition 4. Suppose multiple candidates occupy the same point. Under left-right tie-breaking, one of these candidates (chosen u.a.r.) receives the entire left vote share allocated to that point, while a different candidate (also u.a.r.) receives the entire right vote share. Under equal split tie-breaking, all candidates at a point share the vote share allocated to that point equally. Equivalently, voters randomly choose between equidistant candidates.

Armed with these positional tie-breaking rules, we now provide several results that demonstrate how a static analysis of Nash equilibria yields more fragile conclusions than analyzing the asymptotic behavior of the replicator dynamics. We focus on two types of equilibria: (1) symmetric mixed-strategy Nash equilibria (SMSNEs), since these relate to fixed points of the replicator dynamics; and (2) pure-strategy Nash equilibria (PSNEs), since these are the focus of classical candidate positioning analyses.

We begin by showing there are multiple SMSNEs in the one-shot candidate positioning game, but they are often unstable or have tiny basins of attraction under the dynamics; that is, they are unlikely to be relevant in practice. In contrast, as we have seen in theory and simulation, the replicator dynamics behave in qualitatively similar ways under a range of specifications. Then, we show that PSNEs are very sensitive to the choice of positional tie-breaking rule: we arrive at entirely different conclusions if we adopt left-right versus equal split tie-breaking. In contrast, the positional tie-breaking rule is irrelevant to our analysis with atomless candidate distributions.

# 7.1 Symmetric mixed-strategy Nash equilibria

Since SMSNEs are a subset of the replicator dynamics fixed points, we might hope to understand the dynamics by analyzing SMSNEs of the game where candidates seek to maximize their plurality win probability. However, we find that there are multiple SMSNEs and they can have trivial basins of attraction. For instance, every candidate at 1/2 is a SMSNE and a replicator dynamics fixed point (with left-right tie-breaking $^{4}$ ). But as we have seen, for $k \geq 5$ all symmetric atomless initial distributions do not converge to a point mass at 1/2. On the other hand, if we allow initial distributions with point masses and the mass at 1/2 is sufficiently high, the candidate distribution does indeed approach the all-at-1/2 SMSNE.

Theorem 13. Suppose $F_0$ places probability mass $p$ at $1/2$ . For any $k \geq 2$ , there is some $p_k^* < 1$ such that if $p > p_k^*$ , the candidate distribution converges to a point mass at $1/2$ under the replicator dynamics with left-right tie-breaking. One of the fixed points of $p^k + kp^{k-1}(1 - p)$ is such a $p_k^*$ .

See Appendix B.4 for proofs omitted from this section. Additionally, there is another family of SMSNEs where each candidate randomly picks between the points x and 1 - x (for $x \in (1/4, 1/2)$ ).

Theorem 14. With $k \geq 4$ and left-right tie-breaking, for any $x \in (1/4, 1/2)$ , the strategy where each candidate picks uniformly at random between $x$ and $1 - x$ is a SMSNE.

Just as with the all-at-1/2 equilibrium, this SMSNE is not indicative of the typical behavior of the replicator dynamics. However, we can show as before that for non-atomless distributions, the candidate distribution can converge to this type of equilibrium.

Theorem 15. Suppose $F_0$ places probability mass $p$ at $x$ and at $1 - x$ , for $1/4 < x < 1/2$ . For any $k \geq 5$ , there exists some $p_k^* < 1/2$ such that if $p > p_k^*$ , the candidate distribution converges to point masses at $x$ and $1 - x$ under the replicator dynamics. In particular, one of the fixed points of $(2p)^k / 2 + k(1 - 2p)((2p)^{k-1} - 2p^{k-1}) / 2$ is such a $p_k^*$ .

These results demonstrate the existence of many SMSNEs that alone do not tell us how we should expect the replicator dynamics to behave.

# 7.2 Positional tie-breaking and pure-strategy Nash equilibria

We now demonstrate how ignoring dynamics and focusing on static equilibria can yield results very sensitive to tie-breaking rules. Cox [16] extends the Hotelling-Downs analysis to more than two candidates, characterizing PSNEs of a one-shot candidate positioning game—crucially, with equal split tie-breaking. With uniform voters and $k \geq 3$ candidates, Cox proves that there is no PSNE for odd $k$ and that the only PSNE for even $k$ has evenly-spaced pairs of candidates at $1/k, 3/k, \ldots, (k - 1)/k$ . Clearly, this analysis makes very different predictions than our replicator dynamics. However, we show that Cox's results depend strongly on equal split tie-breaking.

To state Cox's result formally, we need to fully specify the candidate objective. We focus on the objective Cox calls complete plurality maximization, where candidates seek first to maximize their vote margin against their strongest competitor, then second-strongest, etc. We extend this objective to allow stochastic positional tie-breaking, assuming candidates first maximize their win probability, then each of their expected vote margins. We can then state Cox's result.

Theorem 16 (Special case of Theorem 2 from Cox [16]). With uniform voters, $k \geq 3$ complete plurality maximizing candidates, and equal split tie-breaking,

1. if k is odd, there is no PSNE,   
2. if k is even, then the unique PSNE has two candidates at each of the points $1/k, 3/k, \ldots, (k-1)/k$ .

If we instead use left–right tie-breaking, the picture is dramatically different. In particular, all candidates at 1/2 is then a PSNE for all k: any deviant who moves from 1/2 loses with certainty to the center candidate who captures the opposite side of the vote. Left–right tie-breaking also introduces many additional PSNEs; we list some of them in the following theorem.

Theorem 17. The following are (some $^{5}$ of the) PSNEs with uniform voters, complete plurality maximizing candidates, and left-right tie-breaking:

1. Any $k \geq 2$ : all k candidates at 1/2.   
2. Any $k \geq 4$ : for any $x \in (1/4, 1/2)$ , $\lfloor k/2 \rfloor$ candidates at x, $\lfloor k/2 \rfloor$ candidates at 1 - x, and the last candidate (if k is odd) at either x or 1 - x.   
3. Any $k \geq 5$ : $\lfloor(k-1)/2\rfloor$ candidates at 1/4, $\lfloor(k-1)/2\rfloor$ candidates at 3/4, one candidate at 1/2, and the last candidate (if k is even) at either 1/4 or 3/4.   
4. Even $k$ : Cox's equilibrium; two candidates at each of the points $1 / k, 3 / k, \ldots, (k - 1) / k$ .

Thus, the qualitative conclusions we arrive by examining Nash equilibria are very different from Cox's if we make another similarly reasonable assumption. Cox's analysis tells us we should not expect candidates converging to the center for any k > 2, but if we use left–right tie-breaking, we find that central configurations are equilibria for all k. The replicator dynamics reveal when these configurations are stable: only for small k. These results highlight how analyzing Nash equilibria provides a brittle picture of candidate positioning, yielding results that are sensitive to tie-breaking and do not capture iterated play. Even SMSNEs, which are closely related to replicator dynamics fixed points, fail to reveal the typical behavior of the dynamics.

# 8 Discussion

We introduced a replicator dynamics model of one-dimensional candidate positioning in plurality elections based on simple heuristic inspired by bounded rationality. Our theoretical results show that the candidates converge to the center when there are at most four candidates per election, but diverge when there are five or more candidates per election. Simulations confirm that this pattern is robust to a large range of model variations. We contrast our results to prior work that focuses on static equilibria or lacks theoretical results for more than two candidates.

Many open questions remain in the analysis of our model. The foremost is a theoretical characterization of the asymptotic candidate distribution for $k \geq 5$ , although this may be challenging given the complex high-k behavior we observe in simulation. An even larger challenge is posed by expanding beyond symmetric and atomless initial candidate distributions to distributions which have points masses or are asymmetric. As we saw in Theorem 15, allowing atomless distributions means there are infinitely attracting distributions for $k \geq 5$ , so the task becomes one of cataloguing all of the possible long-run candidate distributions. Theoretical results for our model variants would be interesting, such as characterizing which mixtures of candidate counts k lead to convergence to the center, or conditions on voter distributions that result in central convergence for $k \leq 4$ .

While we explored several model variations in simulation, there are many more than can possibly be covered in a single paper. Additional variations of particular interest include policy-motivated candidates, strategic voters, probabilistic voters, and higher-dimensional preferences. Another natural direction would be to explore voting systems other than plurality, like two-round runoff, instant runoff, or Borda count; Condorcet methods are considerably less interesting under our one-dimensional replicator dynamics, since the candidate closest to the median voter always wins, but might exhibit more complex behavior in higher dimensions.

# Acknowledgments

This work was supported in part by ARO MURI, a Simons Collaboration grant, a grant from the MacArthur Foundation, a Vannevar Bush Faculty Fellowship, AFOSR grant FA9550-19-1-0183, and NSF CAREER Award #2143176.

# References

[1] Elvio Accinelli, Filipe Martins, Jorge Oviedo, Alberto Pinto, and Luis Quintas. Who controls the controller? a dynamical model of corruption. The Journal of Mathematical Sociology, 41(4):220–247, 2017.   
[2] N Balakrishnan and Subrahmaniam Kocherlakota. On the double Weibull distribution: order statistics and estimation. Sankhyā: The Indian Journal of Statistics, Series B, pages 161–178, 1985.   
[3] Johann Bauer, Mark Broom, and Eduardo Alonso. The stabilization of equilibria in evolutionary game dynamics through mutation: mutation limits in evolutionary games. Proceedings of the Royal Society A, 475(2231):20190355, 2019.

[4] Jonathan Bendor, Daniel Diermeier, David A Siegel, and Michael M Ting. A behavioral theory of elections. Princeton University Press, 2011.   
[5] Daan Bloembergen, Karl Tuyls, Daniel Hennes, and Michael Kaisers. Evolutionary dynamics of multiagent learning: A survey. Journal of Artificial Intelligence Research, 53:659–697, 2015.   
[6] Lawrence Blume and David Easley. Evolution and market behavior. Journal of Economic Theory, 58(1):9–40, 1992.   
[7] Tobias Böhmelt, Lawrence Ezrow, Roni Lehrer, and Hugh Ward. Party policy diffusion. American Political Science Review, 110(2):397–410, 2016.   
[8] Damien Bol, Arnaud Dellis, and Mandar Oak. Endogenous candidacy in plurality rule elections: Some explanations of the number of candidates and their polarization. Available at SSRN 2704859, 2016.   
[9] Tilman Börgers and Rajiv Sarin. Learning through reinforcement and replicator dynamics. Journal of Economic Theory, 77(1):1-14, 1997.   
[10] Michael Bruter, Robert S Erikson, and Aaron B Strauss. Uncertain candidates, valence, and the dynamics of candidate position-taking. Public Choice, 144:153–168, 2010.   
[11] Steven Callander. Bandwagons and momentum in sequential voting. The Review of Economic Studies, 74(3):653–684, 2007.   
[12] Randall L Calvert. Robustness of the multidimensional voting model: Candidate motivations, uncertainty, and convergence. American Journal of Political Science, pages 69–95, 1985.   
[13] Henry W Chappell and William R Keech. Policy motivation and party differences in a dynamic spatial model of party competition. American Political Science Review, 80(3):881–899, 1986.   
[14] Man-Wah Cheung. Imitative dynamics for games with continuous strategy space. Games and Economic Behavior, 99:206-223, 2016.   
[15] Clyde H Coombs. Psychological scaling without a unit of measurement. Psychological Review, 57(3): 145, 1950.   
[16] Gary W Cox. Electoral equilibrium under alternative voting institutions. American Journal of Political Science, pages 82–108, 1987.   
[17] André De Palma, Victor Ginsburgh, Yorgo Y Papageorgiou, and J-F Thisse. The principle of minimum differentiation holds under sufficient heterogeneity. Econometrica, pages 767–781, 1985.   
[18] Eddie Dekel and Michele Piccione. Sequential voting procedures in symmetric binary elections. Journal of Political Economy, 108(1):34–55, 2000.   
[19] Yvo Desmedt and Edith Elkind. Equilibria of plurality voting with abstentions. In Proceedings of the 11th ACM Conference on Electronic Commerce, pages 347-356, 2010.   
[20] Anthony Downs. An economic theory of democracy. Harper & Row, 1957.   
[21] John Duggan and César Martinelli. The political economy of dynamic elections: Accountability, commitment, and responsiveness. Journal of Economic Literature, 55(3):916–984, 2017.   
[22] Maurice Duverger. Political parties: Their organization and activity in the modern state. Methuen and Wiley, 2nd edition, 1959. Translated by Barbara North and Robert North.   
[23] B Curtis Eaton and Richard G Lipsey. The principle of minimum differentiation reconsidered: Some new developments in the theory of spatial competition. The Review of Economic Studies, 42(1):27–49, 1975.

[24] Edith Elkind, Martin Lackner, and Dominik Peters. Preference restrictions in computational social choice: recent progress. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, pages 4062-4065, 2016.   
[25] Lawrence Ezrow, Tobias Böhmelt, Roni Lehrer, and Hugh Ward. Follow the foreign leader? why following foreign incumbents is an effective electoral strategy. Party Politics, 27(4):716–729, 2021.   
[26] Timothy J Feddersen, Itai Sened, and Stephen G Wright. Rational voting and candidate entry under plurality rule. American Journal of Political Science, pages 1005-1016, 1990.   
[27] Jean Guillaume Forand. Two-party competition with persistent policies. Journal of Economic Theory, 152:64–91, 2014.   
[28] Daniel Friedman. Evolutionary games in economics. Econometrica: Journal of the Econometric Society, pages 637–666, 1991.   
[29] Chaitanya S Gokhale and Arne Traulsen. Evolutionary games in the multiverse. Proceedings of the National Academy of Sciences, 107(12):5500–5504, 2010.   
[30] Bernard Grofman. Downs and two-party convergence. Annual Review of Political Science, 7:25–46, 2004.   
[31] Tim Groseclose. A model of candidate location when one candidate has a valence advantage. American Journal of Political Science, pages 862-886, 2001.   
[32] Josef Hofbauer and Karl Sigmund. Evolutionary game dynamics. Bulletin of the American Mathematical Society, 40(4):479-519, 2003.   
[33] Harold Hotelling. Stability in competition. The Economic Journal, 39(153):41–57, 1929.   
[34] Andreas Irmen and Jacques-François Thisse. Competition in multi-characteristics spaces: Hotelling was almost right. Journal of Economic Theory, 78(1):76–102, 1998.   
[35] Ken Kollman, John H Miller, and Scott E Page. Adaptive parties in spatial elections. American Political Science Review, 86(4):929–937, 1992.   
[36] Ken Kollman, John H Miller, and Scott E Page. Political parties and electoral landscapes. British Journal of Political Science, 28(1):139–158, 1998.   
[37] Gerald H Kramer. A dynamical model of political equilibrium. Journal of Economic Theory, 16(2): 310-334, 1977.   
[38] Anna-Sophie Kurella. Issue Voting and Party Competition, chapter The Evolution of Models of Party Competition, pages 11–25. Springer, 2017.   
[39] Jean-François Laslier and Bilge Ozturk Goktuna. Opportunist politicians and the evolution of electoral competition. Journal of Evolutionary Economics, 26:381–406, 2016.   
[40] Michael Laver and Kenneth Benoit. The evolution of party systems between elections. American Journal of Political Science, 47(2):215–233, 2003.   
[41] Omer Lev and Jeffrey S Rosenschein. Convergence of iterative voting. In Proceedings of the 11th International Conference on Autonomous Agents and Multiagent Systems, pages 611–618, 2012.   
[42] Viktor Losert and Ethen Akin. Dynamics of games and genes: Discrete versus continuous time. Journal of Mathematical Biology, 17:241-251, 1983.

[43] Walter R Mebane Jr. Partisan messages, unconditional strategies and coordination in american elections. Revised version of a paper originally presented at the Annual Meeting of the Political Methodology Society (https://public.websites.umich.edu/\~wmebane/egamesim.pdf), 2005.   
[44] Reshef Meir, Maria Polukarov, Jeffrey Rosenschein, and Nicholas Jennings. Convergence to equilibria in plurality voting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 24, pages 823–828, 2010.   
[45] Reshef Meir, Omer Lev, and Jeffrey S Rosenschein. A local-dominance theory of voting equilibria. In Proceedings of the 15th ACM Conference on Economics and Computation, pages 313-330, 2014.   
[46] Roger B Myerson and Robert J Weber. A theory of voting equilibria. American Political Science Review, 87(1):102–114, 1993.   
[47] Richard R Nelson, Giovanni Dosi, Constance E Helfat, Andreas Pyka, Pier Paolo Saviotti, Keun Lee, Sidney G Winter, Kurt Dopfer, and Franco Malerba. Modern evolutionary economics: An overview. 2018.   
[48] Joille Noailly, Jeroen CJM van den Bergh, and Cees A Withagen. Evolution of harvesting strategies: replicator and resource dynamics. Journal of Evolutionary Economics, 13:183–200, 2003.   
[49] Salvatore Nunnari and Jan Zápal. Dynamic elections and ideological polarization. Political Analysis, 25(4):505–534, 2017.   
[50] Svetlana Obraztsova, Evangelos Markakis, Maria Polukarov, Zinovi Rabinovich, and Nicholas Jennings. On the convergence of iterative voting: how restrictive should restricted dynamics be? In Proceedings of the AAAI Conference on Artificial Intelligence, volume 29, 2015.   
[51] Svetlana Obraztsova, Zinovi Rabinovich, Edith Elkind, Maria Polukarov, and Nicholas R Jennings. Trembling hand equilibria of plurality voting. In Proceedings of the 25th International Joint Conference on Artificial Intelligence, 2016.   
[52] Jörg Oechssler and Frank Riedel. Evolutionary dynamics on infinite strategy spaces. Economic Theory, 17:141-162, 2001.   
[53] Martin J Osborne. Spatial models of political competition under plurality rule: A survey of some explanations of the number of candidates and the positions they take. Canadian Journal of Economics, pages 261-301, 1995.   
[54] Thomas R Palfrey. Spatial equilibrium with entry. The Review of Economic Studies, 51(1):139–156, 1984.   
[55] Charles R Plott. A notion of equilibrium and its possibility under majority rule. The American Economic Review, 57(4):787-806, 1967.   
[56] Keith T Poole and Howard Rosenthal. The polarization of american politics. The Journal of Politics, 46(4):1061–1079, 1984.   
[57] William H Riker. The two-party system and duverger's law: An essay on the history of political science. American Political Science Review, 76(4):753–766, 1982.   
[58] Robert W Rosenthal. A model of far-sighted electoral competition. Mathematical Social Sciences, 2(3): 289-297, 1982.   
[59] Karolina Safarzyńska and Jeroen CJM van den Bergh. Evolutionary models in economics: a survey of methods and building blocks. Journal of Evolutionary Economics, 20:329–373, 2010.

[60] Karolina Safarzynska and Jeroen CJM van den Bergh. Beyond replicator dynamics: Innovation–selection dynamics and optimal diversity. Journal of Economic Behavior & Organization, 78(3):229–245, 2011.   
[61] Pier Paolo Saviotti and GS Mani. Competition, variety and technological evolution: a replicator dynamics model. Journal of Evolutionary Economics, 5:369–392, 1995.   
[62] Peter Schuster and Karl Sigmund. Replicator dynamics. Journal of Theoretical Biology, 100(3):533–538, 1983.   
[63] Charles R Shipan and Craig Volden. The mechanisms of policy diffusion. American Journal of Political Science, 52(4):840–857, 2008.   
[64] Gernot Sieg and Christof Schulz. Evolutionary dynamics in the voting game. Public Choice, 85(1-2): 157–172, 1995.   
[65] Herbert A Simon. A behavioral model of rational choice. The Quarterly Journal of Economics, pages 99-118, 1955.   
[66] Herbert A Simon. Rational decision making in business organizations. The American Economic Review, 69(4):493–513, 1979.   
[67] Peter D Taylor and Leo B Jonker. Evolutionary stable strategies and game dynamics. Mathematical Biosciences, 40(1-2):145–156, 1978.   
[68] David RM Thompson, Omer Lev, Kevin Leyton-Brown, and Jeffrey Rosenschein. Empirical analysis of plurality election equilibria. In Proceedings of the 2013 international conference on Autonomous agents and multi-agent systems, pages 391–398, 2013.   
[69] Amos Tversky and Daniel Kahneman. Judgment under uncertainty: Heuristics and biases. Science, 185(4157):1124–1131, 1974.   
[70] Shlomo Weber. On hierarchical spatial competition. The Review of Economic Studies, 59(2):407-425, 1992.   
[71] Donald Wittman. Candidates with policy preferences: A dynamic model. Journal of Economic Theory, 14(1):180–189, 1977.   
[72] Donald Wittman. Candidate motivation: A synthesis of alternative theories. American Political science review, 77(1):142–157, 1983.

# Appendix for Replicating Electoral Success

# Contents

1 Introduction 1

1.1 Related work 4

2 Replicator dynamics for candidate positioning 5

2.1 $k = 2$ 6   
2.2 $k = 3$ 7   
2.3 $k = 4$ 7   
2.4 $k\geq 5$ 8

3 Replicator dynamics with noise 8

3.1 $k = 2$ 9   
3.2 $k = 3$ 10   
3.3 $k = 4$ 10   
3.4 $k\geq 5$ 10

4 Positive results for $k \geq 5$ with no extreme candidates 10   
5 Simulations 11   
6 Variants of the replicator dynamics 14   
7 Relationship to Nash equilibria of one-shot games 15

7.1 Symmetric mixed-strategy Nash equilibria 16   
7.2 Positional tie-breaking and pure-strategy Nash equilibria 17

8 Discussion 18

A Additional Plots 24

A.1 Additional variant plots 27

B Additional Proofs 30

B.1 Proofs from Section 2 30   
B.2 Proofs from Section 3 33   
B.3 Proofs from Section 4 40   
B.4 Proofs from Section 7 43

C Formal definitions of variants 47

# A Additional Plots

![](images/6df51d15b359b0c6ad07a0bb3928e2df65b839c05ba71439f671ccfa30fb1014.jpg)

Figure 8: Replicator dynamics runs just as in Figure 3, but without enhanced symmetry. For k > 6, the behavior of the Monte Carlo trials becomes inconsistent without enhanced symmetry, particularly without $\epsilon$ -uniform noise. See Figure 3 for more details.   
![](images/461b0a5057c85c8c15977f3eb5705279e27e8629106a7889627288eeca31b6cf.jpg)  
Figure 9: Replicator dynamics runs with enhanced symmetry just as in Figure 3, but showing only a single trial instead of aggregating 50 runs. With enhanced symmetry, the behavior is very consistent across runs.

![](images/c066ba40203754086ee16824122ec4dafb1846577d37e96e98a95b4ae2fbfc80.jpg)

Figure 10: Replicator dynamics runs just as in Figure 8 (no enhanced symmetry), but showing only a single trial instead of aggregating 50 runs to highlight the inconsistent behavior for k = 6 and 7 without enhanced symmetry.   
![](images/e4299eb6d2138d679716cc8119e1c064deadacb50b6b0a3afd2f32f1d1aa252b.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0.5   |
| 200 | 0.5   |
</details>

![](images/0e66e53bbda7932aa930632814ce54781bbdc1f2b23ed8afa53aceeb7e9abe85.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0.5   |
| 100 | 0.5   |
| 200 | 0.5   |
</details>

![](images/ad194fbe249417931b23e06ff8c78f1793816f9963f4ad31e5893da891a91157.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 100 | 0     |
| 200 | 0     |
</details>

![](images/ce72867bada35e5bdf7c008271fcfebbc77464806d0584c2d477cac4dc60cb29.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 50  | 100   |
| 100 | 150   |
| 150 | 200   |
| 200 | 250   |
</details>

![](images/d49373bb66462e24fee33b65e3ddc8eb40b73efcf218481b64fd8ffeedef7169.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 100 | 0     |
| 200 | 0     |
</details>

![](images/3af8edb867638d45bbb4cebc580f5ccc2b389ca0782f22d943dc843e1890977c.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 100 | 0     |
| 200 | 0     |
</details>

![](images/3f6c601eabaa7515a3d672c684c796c4b69d49df4e6034cb02e092342dca72c0.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 50  | 1     |
| 100 | 1     |
| 150 | 1     |
| 200 | 1     |
</details>

![](images/3a5fc232c6bd6624bf756284189aefaad04e32357c79dc14b49a0092b4906e1d.jpg)

<details>
<summary>heatmap</summary>

| t   | Value |
| --- | ----- |
| 0   | Low   |
| 50  | Medium|
| 100 | High  |
| 150 | Medium|
| 200 | High  |
</details>

![](images/2f2d5f78a535d9cda7de184e3b3729bf07863323a1e1a8ba415d51942a7974bb.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 50  | 0     |
| 100 | 0     |
| 150 | 0     |
| 200 | 0     |
</details>

![](images/4bd9cabd0fca2337a0b1c5f421163f31115e12fdfc55c5288825086c59fadd76.jpg)

<details>
<summary>text_image</summary>

k = 6
0 100 200
t
</details>

![](images/82c6dd842e9e830f3217feeb3a47eccda2feed3f768a6c7cde84cccb568d793f.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 50  | 1.5   |
| 100 | 1.5   |
| 150 | 1.5   |
| 200 | 1.5   |
</details>

![](images/fa140c0493f999fe1d0d24bc4327a9ee7e0932f50c8b1a093d5a56e6bd15065b.jpg)

<details>
<summary>heatmap</summary>

| t   | Value |
| --- | ----- |
| 0   | Low   |
| 100 | Medium|
| 200 | High  |
</details>

Figure 11: Replicator dynamics runs with only 50 elections per generation, without enhanced symmetry. Each plot shows 50 trials. The top row has no noise, while the bottom row uses 0.01-uniform noise. Even with a small sample size, our main finding holds.

![](images/fa365fa9292d6506c449a2311dec78b4a60e81333241c058d4514678547cb54c.jpg)

Figure 12: Replicator dynamics runs just as in Figure 4, but without enhanced symmetry. As with smaller values of k, the behavior becomes more chaotic without enhanced symmetry.   
![](images/db156fd8936009afa803e351213a36cbac1255094887fafdd19ebebb5932c5d2.jpg)  
Figure 13: Replicator dynamics with initial candidate distribution Uniform(1/4, 3/4). These plots show 50 trials with 100,000 elections per generation, no noise, and without enhanced symmetry. The dynamics are very well-behaved with $(1/4, 3/4)$ support, removing the need for enhanced symmetry; compare to Figure 8.

# A.1 Additional variant plots

![](images/dacfa99c3475543e410a53c767d63c20cdc3e0366c0f99e06d5f6748f0819543.jpg)

![](images/c94d1f5e5bd9a1b4dfedc03c72f50f409c3a9d7da7fadbffe2a166c1a3da0adf.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0.0  | 1.0   |
| 0.2  | 0.3   |
| 0.4  | 0.2   |
| 0.6  | 0.2   |
| 0.8  | 0.3   |
| 1.0  | 1.0   |
</details>

![](images/ff74e3603e35372d905dc2799b6d8ca1d875b85a8bcd3c08a448368656e644b2.jpg)

<details>
<summary>area</summary>

| x    | y     |
| ---- | ----- |
| 0.0  | 0.0   |
| 0.2  | 1.0   |
| 0.4  | 0.0   |
| 0.6  | 0.0   |
| 0.8  | 1.0   |
| 1.0  | 0.0   |
</details>

Figure 14: PDFs of different voter distributions used in Figure 7.

![](images/997d1d61cd201fd0f2103dc11f9ac74e6c9c5da956212275967cadeaea31d547.jpg)

<details>
<summary>line</summary>

| t   | Memory |
| --- | ------ |
| 0   | 0.0    |
| 100 | 0.5    |
| 200 | 0.5    |
</details>

![](images/2537634a65ac358820a5110e3f76f43a0bd8ad08f6fc3ec9c1eab43e3637650e.jpg)

<details>
<summary>line</summary>

| t   | k = 3 |
| --- | ----- |
| 0   | 0     |
| 100 | 0     |
| 200 | 0     |
</details>

![](images/5380a830b64f4f2009177f505162373c2223bec864ddcbeb852abd00fd267849.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | High  |
| 100 | Medium|
| 200 | Low   |
</details>

![](images/7eb76bcfc25e38ab6f564f1732337a51819784ae5a01499f937b0d5279e7c2f7.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 100 | 0     |
| 200 | 0     |
</details>

![](images/ae6a26401f8305d4270a433898250fa10ee145cfb6d953474812046a89b59219.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 100 | 0     |
| 200 | 0     |
</details>

![](images/4275d46941fa05e503dc21c3b78dfd34911059d0aff686d979aa816e90de86d0.jpg)

<details>
<summary>line</summary>

| t   | Value |
| --- | ----- |
| 0   | 0     |
| 100 | 0     |
| 200 | 0     |
</details>

Figure 15: Replicator dynamics with m = 3 generations of memory, no enhanced symmetry, and 50 trials per plot. There is no qualitative difference between m = 3 and m = 2 (compare to Figure 7).

![](images/512fbf10608bec9ad608e8f75dbc8d854c70116f99641d5d87e559f3f8a3cb4e.jpg)  
Figure 16: Single trials of the replicator dynamics with perturbation noise and 100,000 elections per generation. The first two rows use $\sigma^{2}=0.001$ , the middle two use $\sigma^{2}=0.005$ , and the bottom two use $\sigma^{2}=0.01$ . Perturbation noise combined with Monte-Carlo asymmetries can result in complex and unpredictable branching with higher k.

![](images/04b060a108f8689c6f69385cf722cdb9846788fe0950816c54a1adbafe510f0e.jpg)

<details>
<summary>heatmap</summary>

| k = 3 fraction | k = 4 fraction | mode at t = 100 |
| -------------- | -------------- | --------------- |
| 0.0            | 1.0            | 0.50            |
| 0.1            | 0.9            | 0.45            |
| 0.2            | 0.8            | 0.40            |
| 0.3            | 0.7            | 0.35            |
| 0.4            | 0.6            | 0.30            |
| 0.5            | 0.5            | 0.25            |
| 0.6            | 0.4            | 0.25            |
| 0.7            | 0.3            | 0.25            |
| 0.8            | 0.2            | 0.25            |
| 0.9            | 0.1            | 0.25            |
| 1.0            | 0.0            | 0.25            |
</details>

Figure 17: Heatmap showing the position of the candidate distribution mode at t = 100 when elections have a mixture of k = 3, 4, and 5 candidates each (only modes $\leq$ 1/2 are shown). These simulations use 100,000 elections per generation, with k split between 3, 4, and 5 in different proportions at each point. The fraction of elections with 3 candidates varies along the x axis, while the fraction with 4 candidates varies along the y axis. Any remaining elections have k = 5. For instance, the lower left corner has all 100,000 elections use k = 5, while the point (1/3, 1/3) has an even mix of candidate counts. When either the k = 3 or k = 4 fraction is high enough (but especially k = 3), the distribution converges to the center, with the mode at 1/2. However, with enough k = 5 elections, two clusters emerge, and more k = 5 elections pushes them farther apart.

![](images/b985e5311628adafffc9dba2e4a9f1ad599ee04dcff2dea427dce5fcd0b03877.jpg)

<details>
<summary>heatmap</summary>

| k   | Time Point | Value |
|-----|------------|-------|
| 4   | 0          | 1     |
| 4   | 100        | 1     |
| 4   | 200        | 1     |
| 5   | 0          | 1     |
| 5   | 100        | 1     |
| 5   | 200        | 1     |
| 6   | 0          | 1     |
| 6   | 100        | 1     |
| 6   | 200        | 1     |
| 7   | 0          | 1     |
| 7   | 100        | 1     |
| 7   | 200        | 1     |
| 8   | 0          | 1     |
| 8   | 100        | 1     |
| 8   | 200        | 1     |
| 9   | 0          | 1     |
| 9   | 100        | 1     |
| 9   | 200        | 1     |
| 10  | 0          | 1     |
| 10  | 100        | 1     |
| 10  | 200        | 1     |
| 15  | 0          | 1     |
| 15  | 100        | 1     |
| 15  | 200        | 1     |
| 25  | 0          | 1     |
| 25  | 100        | 1     |
| 25  | 200        | 1     |
| 50  | 0          | 1     |
| 50  | 100        | 1     |
| 50  | 200        | 1     |
</details>

Figure 18: Replicator dynamics with top-h copying where h = 3, no enhanced symmetry, 50 trials per plot, and 100,000 elections per generation.

# B Additional Proofs

# B.1 Proofs from Section 2

Theorem 3. Let $F_{0} \in F$ . For all x < 1/2 and t > 0,

$$
F _ {3, t} (x) \leq 3 / 4 \cdot F _ {3, t - 1} (x) + F _ {3, t - 1} (x) ^ {3}. \tag {3}
$$

This can be written as a looser closed form

$$
F _ {3, t} (x) \leq F _ {0} (x) \cdot \left[ 3 / 4 + F _ {0} (x) ^ {2} \right] ^ {t}. \tag {4}
$$

Proof. Let $x < 1/2$ and define $p = F_{3,t-1}(x)$ . Consider the following cases for the positions of the three candidates $X_{1,t}, X_{2,t}$ , and $X_{3,t}$ . Call candidates in $(x, 1 - x)$ inner.

1. All three candidates in $[0,1/2)$ (and the symmetric case). First suppose all three are in $[0,1/2)$ (the other side is symmetric). If there is at least one inner candidate (w.p. $1/2^{3}-p^{3}$ ), then the winner is inner. Accounting for symmetry, an inner candidate wins in this case w.p. $2(1/2^{3}-p^{3})=1/4-2p^{3}$ .   
2. Two candidates in $(x,1/2)$ and one in $(1/2,1-x)$ (and the symmetric case). Since all candidates are inner, an inner candidate wins. Accounting for symmetry, an inner candidate wins in this case w.p. $2\left[3(1/2-p)^{3}\right]=6(1/2-p)^{3}$ .   
3. Two candidates in $[0, x)$ and one in $(1/2, 1 - x)$ (and the symmetric case). The candidate in $(1/2, 1 - x)$ wins with vote share at least 1/2. Accounting for symmetry, an inner candidate wins in this case w.p. $2\left[3p^{2}(1/2 - p)\right] = 6p^{2}(1/2 - p)$ .   
4. One candidate in $[0,x)$ , one in $(x,1/2)$ , and one in $(1/2,1-x)$ . Label them 1, 2, and 3, respectively. Candidate 3 gets vote share $1-(X_{3}+X_{2})/2=[(1-X_{3})+(1-X_{2})]/2$ , while candidate 1 gets vote share $(X_{1}+X_{2})/2$ . Since $X_{3}<1-x$ , $1-X_{3}>X_{1}$ ; and since $X_{2}<1/2$ , $1-X_{2}>X_{2}$ . Thus candidate 3 has higher vote share than candidate 1 and an inner candidate wins. Accounting for symmetry, an inner candidate wins in this case w.p. $2\left[3\cdot2p(1/2-p)^{2}\right]=12p(1/2-p)^{2}$ .

Adding up these cases yields a lower bound on the probability that an inner candidate wins:

$$
\operatorname * {P r} (x <   \mathrm{Plurality} (X _ {1, t}, X _ {2, t}, X _ {3, t}) <   1 - x) \geq 1 / 4 - 2 p ^ {3} + 6 (1 / 2 - p) ^ {3} + 6 p ^ {2} (1 / 2 - p) + 1 2 p (1 / 2 - p) ^ {2}
$$

$$
= 1 - 3 / 2 \cdot p - 2 p ^ {3}.
$$

By symmetry, this yields the claimed upper bound on the probability a candidate in $[0, x]$ wins:

$$
\begin{array}{l} F _ {3, t} (x) = \operatorname * {P r} (\text { Plurality } (X _ {1, t}, X _ {2, t}, X _ {3, t}) \leq x) \\ = (1 - \operatorname * {P r} (x <   \text { Plurality } (X _ {1, t}, X _ {2, t}, X _ {3, t}) <   1 - x)) / 2 \\ \leq \left[ 1 - (1 - 3 / 2 \cdot p - 2 p ^ {3}) \right] / 2 \\ = 3 / 4 \cdot p + p ^ {3} \\ = 3 / 4 \cdot F _ {3, t - 1} (x) + F _ {3, t - 1} (x) ^ {3}. \\ \end{array}
$$

We now show the closed form bound by induction on $t$ . We'll simultaneously show that $F_{3,t}(x) \leq F_0(x)$ . For the base case $t = 0$ , we have $F_{3,t}(x) \leq F_0(x)$ . Now for $t > 0$ , suppose the claims hold for $t - 1$ . Using the bound above, we know that

$$
\begin{array}{l} F _ {3, t} (x) \leq 3 / 4 \cdot F _ {3, t - 1} (x) + F _ {3, t - 1} (x) ^ {3} \\ = F _ {3, t - 1} (x) \cdot \left[ 3 / 4 + F _ {3, t - 1} (x) ^ {2} \right] \\ \leq F _ {3, t - 1} (x) \cdot \left[ 3 / 4 + F _ {0} (x) ^ {2} \right] (byIH) \\ \leq F _ {0} (x) \cdot \left[ 3 / 4 + F _ {0} (x) ^ {2} \right] ^ {t - 1} \cdot \left[ 3 / 4 + F _ {0} (x) ^ {2} \right] (byIH) \\ = F _ {0} (x) \cdot \left[ 3 / 4 + F _ {0} (x) ^ {2} \right] ^ {t} \\ \end{array}
$$

This is the main claim we wanted to show. We can now also show the supporting fact that $F_{3,t}(x) \leq F_0(x)$ . For $x < 1/2$ , $F_0(x) \leq 1/2$ by symmetry. Thus $3/4 + F_0(x)^2 \leq 3/4 + 1/2^2 = 1$ , so by the inequality above, $F_{3,t}(x) \leq F_0(x) \cdot \left[3/4 + F_0(x)^2\right]^t \leq F_0(x) \cdot 1^t$ .

Lemma 1. Let $F_0 \in \mathcal{F}$ . For all $x \in (1/3, 1/2)$ and $t \geq 0$ , $F_{4,t}(x) \leq F_{4,0}(x)$ .

Proof. Let $x \in (1/3, 1/2)$ and $p = F_{4,t-1}(x)$ . We'll find a lower bound on the probability an inner candidate in $(x, 1 - x)$ wins. Consider the following cases for candidate positions in a $k = 4$ plurality election:

1. All four candidates in $[0,1/2)$ (and the symmetric case). An inner candidate wins if at least one candidate is inner. Accounting for symmetry, an inner candidate wins in this case w.p. $2(1/2^{4}-p^{4})=1/8-2p^{4}$ .   
2. Three candidates in $[0,1/2)$ and one in $(1/2,1-x)$ (and the symmetric case). The candidate on the right has a higher vote share than any outer candidate on the left (as in Theorem 3 Case 4), so an inner candidate wins. Accounting for symmetry, an inner candidate wins in this case w.p. $2(4\cdot1/2^{3}\cdot(1/2-p))=1/2-p$ .   
3. Two candidates in $(x,1/2)$ and two in $(1/2,1 - x)$ . All candidates are inner, so an inner candidate wins. This occurs w.p. $\binom{4}{2} \cdot (1/2 - p)^4 = 6(1/2 - p)^4$ .   
4. Two candidates in $[0, x)$ and two in $(1/2, 1 - x)$ (and the symmetric case). Since x > 1/3, the rightmost candidate gets vote share greater than 1/3. Meanwhile, the leftmost candidate gets vote share less than 1/3. The second-leftmost candidate gets vote share less than $(2/3)/2 = 1/3$ (the candidates flanking it are closer together than 0 and 1 - x < 2/3). Thus an inner candidate wins. Accounting for symmetry, an inner candidate wins in this case w.p. $2\binom{4}{2}p^{2}(1/2 - p)^{2} = 12p^{2}(1/2 - p)^{2}$   
5. Two candidates in $(x,1/2)$ , one in $(1/2,1-x)$ , and one in $(1-x,1]$ (and the symmetric case). Label the candidates 1–4 in left–right order. By symmetry, candidate 3 is farther from 1/2 than candidate 2 with probability 2/3: all $3!=6$ orderings of distance from 1/2 between candidates 1–3 are equiprobable and only the 2 where candidate 3 is closest to 1/2 fail this property. In this scenario, candidate 4 has vote share $1-(X_{3}+X_{4})/2=((1-X_{3})+(1-X_{4}))/2$ and candidate 1 has vote share $(X_{1}+X_{2})/2$ . Since $1-X_{3}<X_{2}$ (candidate 2 is closer to the center than 3) and $1-X_{4}<X_{2}$ (since $X_{2}>x$ and $X_{4}>1-x$ ), candidate 1 has a larger vote share than candidate 4, the only outer candidate. Thus an inner candidate wins. Accounting for symmetry, an inner candidate wins in this case w.p. $2\cdot4\cdot3\cdot2/3\cdot p(1/2-p)^{3}=16p(1/2-p)^{3}$

Combining all five cases gives a lower bound on the probability that an inner candidate wins:

$$
\begin{array}{l} \operatorname * {P r} (x <   \text { Plurality } (X _ {1, t}, X _ {2, t}, X _ {3, t}, X _ {4, t}) <   1 - x) \\ \geq 1 / 8 - 2 p ^ {4} + 1 / 2 - p + 6 (1 / 2 - p) ^ {4} + 1 2 p ^ {2} (1 / 2 - p) ^ {2} + 1 6 p (1 / 2 - p) ^ {3} \\ = 1 - 2 p \\ = 1 - 2 \cdot F _ {4, t - 1} (x). \\ \end{array}
$$

By symmetry, this means

$$
\begin{array}{l} F _ {4, t} (x) = \operatorname * {P r} (\mathrm{Plurality} (X _ {1, t}, X _ {2, t}, X _ {3, t}, X _ {4, t}) \leq x) \\ = \left[ 1 - \operatorname * {P r} (x <   \text { Plurality } (X _ {1, t}, X _ {2, t}, X _ {3, t}, X _ {4, t}) <   1 - x) \right] / 2 \\ \leq \left[ 1 - (1 - 2 \cdot F _ {4, t - 1} (x)) \right] / 2 \\ = F _ {4, t - 1} (x). \\ \end{array}
$$

The claim then follows by induction on t.

Theorem 4. Let $F_{0} \in F$ . For all $x \in (1/3, 1/2)$ and $t \geq 0$ ,

$$
F _ {4, t} (x) \leq F _ {0} (x) \cdot \left[ 1 - 4 (1 / 2 - F _ {0} (x / 3 + 1 / 3)) ^ {3} \right] ^ {t}. \tag {5}
$$

Proof. Let $x \in (1/3, 1/2)$ and $p = F_{4,t-1}(x)$ . By the argument in the proof of Lemma 1, an inner candidate wins with probability at least $1 - 2p$ . We can strengthen this bound using Lemma 1 and one more case omitted from that analysis (which can't easily be used there): three candidates in $(x/3 + 1/3, 1/2)$ and one in $(1 - x, 1]$ (and the symmetric case). Note that $x/3 + 1/3 = x + 2/3 \cdot (1/2 - x)$ is the point two-thirds of the way from $x$ to $1/2$ . The leftmost candidate gets vote share more than $x/3 + 1/3$ . Meanwhile, the lone outer candidate gets vote share less than $x + (1 - x - (x/3 + 1/3))/2 = x/3 + 1/3$ , so an inner candidate wins. By Lemma 1, we know $F_{4,t-1}(x/3 + 1/3) \leq F_{4,0}(x/3 + 1/3)$ . Thus, a candidate is in $(x/3 + 1/3, 1/2)$ with probability $1/2 - F_{4,t-1}(x/3 + 1/3) \geq 1/2 - F_{4,0}(x/3 + 1/3)$ . Therefore, accounting for symmetry, an inner candidate wins in this case w.p. at least $2 \cdot 4 \cdot (1/2 - F_{4,0}(x/3 + 1/3))^3 \cdot p = 8p(1/2 - F_{4,0}(x/3 + 1/3))^3$ .

Combining this new case with the cases from the proof of Lemma 1, an inner candidate wins w.p. at least $1 - 2p + 8p(1/2 - F_{4,0}(x/3 + 1/3))^3$ . By symmetry, this means

$$
\begin{array}{l} F _ {4, t} (x) \leq \left[ 1 - (1 - 2 p + 8 p (1 / 2 - F _ {4, 0} (x / 3 + 1 / 3)) ^ {3}) \right] / 2 \\ = \left[ 2 p - 8 p (1 / 2 - F _ {4, 0} (x / 3 + 1 / 3)) ^ {3} \right] / 2 \\ = p \left[ 1 - 4 (1 / 2 - F _ {4, 0} (x / 3 + 1 / 3)) ^ {3} \right] \\ = F _ {4, t - 1} (x) \cdot \left[ 1 - 4 (1 / 2 - F _ {4, 0} (x / 3 + 1 / 3)) ^ {3} \right]. \\ \end{array}
$$

The claim then follows by induction on $t$ .

![](images/c92a709673e70722f2732540a3cd64a520faac37637d90baacccd957fb2a0189.jpg)

Theorem 5. Let $F_0 \in \mathcal{F}$ . For any $k \geq 5$ , there exists some $x < 1/2$ such that $\lim_{t \to \infty} F_{k,t}(x) \neq 0$ . That is, the candidate distribution does not converge to a point mass at $1/2$ .

Proof. Suppose $F_{k,t-1}(1/4) \leq \alpha$ for some small $\alpha$ . Let $x \in (1/4, 1/2)$ and $F_{k,t-1}(x) = p$ , so $F_{k,t-1}(x) - F_{k,t-1}(1/4) \geq p - \alpha$ . We'll lower bound the probability that the winner is an outer candidate outside of $(x, 1 - x)$ , focusing mainly on cases where all candidates are in $(1/4, 3/4)$ so we can apply Lemma 2.

If all candidates are in $[0,1/2)$ , then an outer candidate only wins if all candidates are left of x, which occurs w.p. $p^{k}$ . Accounting for the symmetric case gives an outer candidate win probability of $2p^{k}$ when all candidates are on the same side. Now suppose there is at least one candidate on each side. If the left-and rightmost candidates are in $(1/4,x)$ and $(1-x,3/4)$ , respectively, then an outer candidate wins by Lemma 2. We can find the probability this occurs as the probability that all candidates are in $(1/4,3/4)$ minus the probability that all candidates are in $(1/4,1-x]$ or in $[x,3/4)$ —since this means there is at least one candidate each in $(1-x,3/4)$ and $(1/4,x)$ . Since $F_{k,t-1}(1/4)\leq\alpha$ , the following is a lower bound on the probability the leftmost candidate is at $X_{1,t}\in(1/4,x)$ and the rightmost is at $X_{k,t}\in(x-1,3/4)$ :

$$
\begin{array}{l} \operatorname * {P r} (X _ {1, t} \in (1 / 4, x), X _ {k, t} \in (x - 1, 3 / 4)) \\ \geq \underbrace {(1 - 2 \alpha) ^ {k}} _ {\text { all   in } (1 / 4, 3 / 4)} - \left[ \underbrace {(1 - \alpha - p) ^ {k}} _ {\text { all   in } (1 / 4, 1 - x ]} + \underbrace {(1 - \alpha - p) ^ {k}} _ {\text { all   in } [ x, 3 / 4)} - \underbrace {(1 - 2 p) ^ {k}} _ {\text { all   in } [ x, 1 - x ]} \right] \quad (\text { by   inclusion - exclusion }) \\ = (1 - 2 \alpha) ^ {k} - 2 (1 - \alpha - p) ^ {k} + (1 - 2 p) ^ {k}. \\ \end{array}
$$

Combining this with the case where all candidates are on the same side (and then dividing by 2 to account for symmetry) yields a lower bound on $F_{k,t}(x)$ :

$$
\begin{array}{l} F _ {k, t} (x) \geq \left[ 2 p ^ {k} + (1 - 2 \alpha) ^ {k} - 2 (1 - \alpha - p) ^ {k} + (1 - 2 p) ^ {k} \right] / 2 \\ = p ^ {k} + (1 - 2 \alpha) ^ {k} / 2 - (1 - \alpha - p) ^ {k} + (1 - 2 p) ^ {k} / 2. \tag {10} \\ \end{array}
$$

We can now use this bound to prove non-convergence. Suppose for a contradiction that $\lim_{t\to \infty}F_{k,t}(x) = 0$ for all $x < 1 / 2$ . Then there exists some $t^*$ such that $F_{k,t}(1 / 4)\leq \alpha = [1 - (499 / 512)^{1 / k}] / 2$ for all $t > t^{*}$ .

But now consider $x^{*} = F_{k,t^{*}}^{-1}(1/4) < 1/2$ . Since $F_{k,t}(1/4) \leq \alpha$ for all $t > t^{*}$ , we can use the fact above to show inductively that $F_{k,t}(x^{*}) \geq 1/4$ for all $t \geq t^{*}$ . For the base case $t = t^{*}$ , the claim is vacuously true: $F_{k,t^{*}}(x^{*}) = 1/4 \geq 1/4$ . Now suppose for $t > t^{*}$ that $F_{k,t-1}(x^{*}) \geq 1/4$ . Then $z = F_{k,t-1}^{-1}(1/4) \leq x^{*}$ . From (10), we then have:

$$
\begin{array}{l} F _ {k, t} (z) \geq 1 / 4 ^ {k} + (1 - 2 \alpha) ^ {k} / 2 - (1 - \alpha - 1 / 4) ^ {k} + (1 - 2 / 4) ^ {k} / 2 \\ > (1 - 2 \alpha) ^ {k} / 2 - (3 / 4 - \alpha) ^ {k} \quad (\text { throw   away   terms }) \\ \geq (1 - 2 \alpha) ^ {k} / 2 - (3 / 4) ^ {5} \quad (\text { since } k \geq 5, \alpha > 0) \\ = \left(1 - 2 \left[ 1 - (4 9 9 / 5 1 2) ^ {1 / k} \right] / 2\right) ^ {k} / 2 - (3 / 4) ^ {5} \quad (\text { plug   in } \alpha) \\ = 4 9 9 / 1 0 2 4 - 2 4 3 / 1 0 2 4 \\ = 1 / 4. \\ \end{array}
$$

By the monotonicity of the CDF, $F_{k,t}(x^{*}) > 1/4$ , since $z \leq x^{*}$ . By induction, $F_{k,t}(x^{*}) \geq 1/4$ for all $t \geq t^{*}$ . This contradicts that $\lim_{t \to \infty} F_{k,t}(x) = 0$ for all $x < 1/2$ .

# B.2 Proofs from Section 3

Our proofs with $\epsilon$ -uniform noise make extensive use of the following lemma, which allows us to translate the convergence of an iterated map bounding a sequence into an eventual bound on the sequence.

Lemma 5. Consider an iterated map $x_{t} = f(x_{t - 1})$ where $f:[0,1 / 2)\to [0,1 / 2)$ is non-decreasing. Suppose $\lim_{t\to \infty}x_t = c$ for all $x_0\in I\subseteq [0,1 / 2)$ .

1. If $y_{t} \leq f(y_{t-1})$ for all t > 0, then $\limsup_{t \to \infty} y_{t} \leq c$ for all $y_{0} \in I$ .   
2. If $y_{t} \geq f(y_{t-1})$ for all $t > 0$ , then $\liminf_{t \to \infty} y_{t} \geq c$ for all $y_{0} \in I$ .

Proof. Let $y_0 \in I$ and define $x_0 = y_0$ . Suppose $y_t \leq f(y_{t-1})$ for all $t > 0$ . We'll show $y_t \leq x_t$ by induction. The base case $t = 0$ holds by the definition of $x_0$ . Suppose for $t > 0$ that $y_{t-1} \leq x_{t-1}$ . Then $y_t \leq f(y_{t-1}) \leq f(x_{t-1}) = x_t$ (since $f$ is non-decreasing), so $y_t \leq x_t$ for all $t$ by induction. Thus, $\limsup_{t \to \infty} y_t \leq \limsup_{t \to \infty} x_t = \lim_{t \to \infty} x_t = c$ . The second claim with $y_t \geq f(y_{t-1})$ follows from the exact same argument with each $\leq$ replaced by $\geq$ and limsup replaced by liminf.

Lemma 3. For all initial $p \in [0,1/2]$ , $\epsilon \in (0,1)$ , and $x \in [0,1/2)$ , the quadratic iterated map $p' = 2p^2(1 - \epsilon)^2 + 4px\epsilon(1 - \epsilon) + 2x^2\epsilon^2$ converges to the fixed point $p^* = \frac{1 - 4x\epsilon(1 - \epsilon) - \sqrt{1 - 8\epsilon x(1 - \epsilon)}}{4(1 - \epsilon)^2} \leq \epsilon$ .

Proof. We begin by looking for the fixed points of the map:

$$
\begin{array}{l} 2 p ^ {2} (1 - \epsilon) ^ {2} + 4 p x \epsilon (1 - \epsilon) + 2 x ^ {2} \epsilon^ {2} = p \\ \Leftrightarrow 2 (1 - \epsilon) ^ {2} p ^ {2} + (4 x \epsilon (1 - \epsilon) - 1) p + 2 x ^ {2} \epsilon^ {2} = 0. \\ \end{array}
$$

Applying the quadratic formula and simplifying yields the two fixed points:

$$
p _ {1} ^ {*} = \frac {1 - 4 x \epsilon (1 - \epsilon) - \sqrt {1 - 8 \epsilon x (1 - \epsilon)}}{4 (1 - \epsilon) ^ {2}}
$$

$$
p _ {2} ^ {*} = \frac {1 - 4 x \epsilon (1 - \epsilon) + \sqrt {1 - 8 \epsilon x (1 - \epsilon)}}{4 (1 - \epsilon) ^ {2}}.
$$

We'll show that $p_1^*$ is stable and that $p_1^* \leq \epsilon$ while $p_2^*$ is unstable and $p_2^* > 1/2$ . To see that $p_1^* \leq \epsilon$ , consider $\epsilon - p_1^*$ :

$$
\epsilon - \frac {1 - 4 x \epsilon (1 - \epsilon) - \sqrt {1 - 8 \epsilon x (1 - \epsilon)}}{4 (1 - \epsilon) ^ {2}} = \frac {4 (1 - \epsilon) ^ {2} \epsilon - 1 + 4 x \epsilon (1 - \epsilon) + \sqrt {1 - 8 \epsilon x (1 - \epsilon)}}{4 (1 - \epsilon) ^ {2}}. \tag {11}
$$

It suffices to show the numerator is non-negative. Taking its derivative with respect to $x$ shows the numerator is decreasing in $x$ :

$$
\frac {\partial}{\partial x} \left[ 4 (1 - \epsilon) ^ {2} \epsilon - 1 + 4 x \epsilon (1 - \epsilon) + (1 - 8 \epsilon x (1 - \epsilon)) ^ {1 / 2} \right] = 4 \epsilon (1 - \epsilon) - \frac {4 \epsilon (1 - \epsilon)}{(1 - 8 \epsilon x (1 - \epsilon)) ^ {1 / 2}}
$$

$$
<   4 \epsilon (1 - \epsilon) - 4 \epsilon (1 - \epsilon)
$$

$$
= 0.
$$

Thus, it suffices to show the function is non-negative when $x = 1/2$ . For $x = 1/2$ ,

$$
4 (1 - \epsilon) ^ {2} \epsilon - 1 + 4 x \epsilon (1 - \epsilon) + \sqrt {1 - 8 \epsilon x (1 - \epsilon)} = 4 (1 - \epsilon) ^ {2} \epsilon - 1 + 2 \epsilon (1 - \epsilon) + \sqrt {1 - 4 \epsilon (1 - \epsilon)}
$$

$$
= 4 \epsilon^ {3} - 1 0 \epsilon^ {2} + 6 \epsilon - 1 + \sqrt {(1 - 2 \epsilon) ^ {2}}
$$

$$
= 4 \epsilon^ {3} - 1 0 \epsilon^ {2} + 6 \epsilon - 1 + | 1 - 2 \epsilon |.
$$

Consider the cases $\epsilon \leq 1/2$ and $\epsilon > 1/2$ . If $\epsilon \leq 1/2$ ,

$$
4 \epsilon^ {3} - 1 0 \epsilon^ {2} + 6 \epsilon - 1 + | 1 - 2 \epsilon | = 4 \epsilon^ {3} - 1 0 \epsilon^ {2} + 4 \epsilon
$$

$$
= 2 \epsilon (2 - \epsilon) (1 - 2 \epsilon)
$$

$$
\geq 0. \quad (\text { since } \epsilon \leq 1 / 2)
$$

If $\epsilon > 1/2$ ,

$$
4 \epsilon^ {3} - 1 0 \epsilon^ {2} + 6 \epsilon - 1 + | 1 - 2 \epsilon | = 4 \epsilon^ {3} - 1 0 \epsilon^ {2} + 8 \epsilon - 2
$$

$$
= 2 (1 - \epsilon) ^ {2} (2 \epsilon - 1)
$$

$$
\geq 0 \quad (\text { since } \epsilon > 1 / 2)
$$

Therefore the numerator in (11) is non-negative, so $p_1^* \leq \epsilon$ . To show $p_1^*$ is stable, consider the derivative of the iterated map in (8):

$$
\frac {\partial}{\partial p} \left[ 2 p ^ {2} (1 - \epsilon) ^ {2} + 4 p x \epsilon (1 - \epsilon) + 2 x ^ {2} \epsilon^ {2} \right] = 4 (1 - \epsilon) ^ {2} p + 4 x \epsilon (1 - \epsilon). \tag {12}
$$

Plugging in $p_1^*$ :

$$
4 (1 - \epsilon) ^ {2} p _ {1} ^ {*} + 4 x \epsilon (1 - \epsilon) = 4 (1 - \epsilon) ^ {2} \frac {1 - 4 x \epsilon (1 - \epsilon) - \sqrt {1 - 8 \epsilon x (1 - \epsilon)}}{4 (1 - \epsilon) ^ {2}} + 4 x \epsilon (1 - \epsilon)
$$

$$
= 1 - \sqrt {1 - 8 \epsilon x (1 - \epsilon)}
$$

$$
<   1 - \sqrt {1 - 4 \epsilon (1 - \epsilon)} \quad (x <   1 / 2)
$$

$$
\leq 1. \quad (\epsilon (1 - \epsilon) \leq 1 / 4)
$$

Thus the derivative of the iterated map at $p_1^*$ has magnitude strictly less than 1, so $p_1^*$ is a stable fixed point.

Now, consider the other fixed point $p_{2}^{*}$ :

$$
p _ {2} ^ {*} = \frac {1 - 4 x \epsilon (1 - \epsilon) + \sqrt {1 - 8 \epsilon x (1 - \epsilon)}}{4 (1 - \epsilon) ^ {2}}
$$

$$
> \frac {1 - 2 \epsilon (1 - \epsilon) + \sqrt {1 - 4 \epsilon (1 - \epsilon)}}{4 (1 - \epsilon) ^ {2}} \quad (x <   1 / 2)
$$

$$
= \frac {1 - 2 \epsilon (1 - \epsilon) + \sqrt {(1 - 2 \epsilon) ^ {2}}}{4 (1 - \epsilon) ^ {2}}
$$

$$
= \frac {1 - 2 \epsilon (1 - \epsilon) + | 1 - 2 \epsilon |}{4 (1 - \epsilon) ^ {2}}.
$$

If $\epsilon \leq 1 / 2$ ,

$$
\begin{array}{l} p _ {2} ^ {*} > \frac {1 - 2 \epsilon (1 - \epsilon) + 1 - 2 \epsilon}{4 (1 - \epsilon) ^ {2}} \\ = \frac {2 (1 - \epsilon) ^ {2}}{4 (1 - \epsilon) ^ {2}} \\ = 1 / 2. \\ \end{array}
$$

If $\epsilon > 1/2$ ,

$$
\begin{array}{l} p _ {2} ^ {*} > \frac {1 - 2 \epsilon (1 - \epsilon) - 1 + 2 \epsilon}{4 (1 - \epsilon) ^ {2}} \\ = \frac {2 \epsilon^ {2}}{4 (1 - \epsilon) ^ {2}} \\ > \frac {2 (1 / 2) ^ {2}}{4 (1 - 1 / 2) ^ {2}} \\ = 1 / 2. \\ \end{array}
$$

In either case, $p_{2}^{*} > 1/2$ . Additionally, plugging $p_{2}^{*}$ into the derivative (12) yields $1 + \sqrt{1 - 8x\epsilon(1 - \epsilon)} > 1$ (for x < 1/2), showing $p_{2}^{*}$ is unstable. Thus, for $p \in [0, 1/2]$ , the quadratic map converges to the stable fixed point $p_{1}^{*} \leq \epsilon$ .

Lemma 6. For any $\epsilon \in (0,1/3)$ , the cubic iterated map given by

$$
p ^ {\prime} = 3 / 4 \cdot [ \epsilon / 2 + (1 - \epsilon) p ] + [ \epsilon / 2 + (1 - \epsilon) p ] ^ {3}
$$

converges to $p^* \leq 1.5\epsilon$ for all initial $p \in [0,1/2)$ . Moreover, the map is non-decreasing in $p$ on $[0,1/2)$ .

Proof. The fixed points of this map can be found using the cubic formula (equivalently, we used Mathematica):

$$
\begin{array}{l} p _ {1} ^ {*} = \frac {1}{4} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} - \frac {1 + 2 \epsilon}{4 (1 - \epsilon)} \\ p _ {2} ^ {*} = - \frac {1}{4} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} - \frac {1 + 2 \epsilon}{4 (1 - \epsilon)} \\ p _ {3} ^ {*} = \frac {1}{2}. \\ \end{array}
$$

We can ignore the negative fixed point $p_2^*$ , since $p$ can never be negative. We'll show that for $\epsilon < 1/3$ , $p_1^* \in [0, 1.5\epsilon]$ , $p_1^*$ is stable, and the cubic map converges to $p_1^*$ for $p \in [0, 1/2)$ . To begin with, we'll show $p_1^* \geq 0$ :

$$
\begin{array}{l} p _ {1} ^ {*} = \frac {1}{4} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} - \frac {1 + 2 \epsilon}{4 (1 - \epsilon)} \\ = \frac {(1 + 1 5 \epsilon) ^ {1 / 2}}{4 (1 - \epsilon) ^ {3 / 2}} - \frac {(1 + 2 \epsilon) (1 - \epsilon) ^ {1 / 2}}{4 (1 - \epsilon) ^ {3 / 2}} \\ = \frac {(1 + 1 5 \epsilon) ^ {1 / 2} - ((1 + 2 \epsilon) ^ {2}) ^ {1 / 2} (1 - \epsilon) ^ {1 / 2}}{4 (1 - \epsilon) ^ {3 / 2}} \\ = \frac {(1 + 1 5 \epsilon) ^ {1 / 2} - ((1 + 2 \epsilon) ^ {2} (1 - \epsilon)) ^ {1 / 2}}{4 (1 - \epsilon) ^ {3 / 2}} \\ = \frac {(1 + 1 5 \epsilon) ^ {1 / 2} - (1 + 3 \epsilon - 4 \epsilon^ {3}) ^ {1 / 2}}{4 (1 - \epsilon) ^ {3 / 2}} \\ \end{array}
$$

$$
\geq 0. \quad (\text { since } 1 + 1 5 \epsilon > 1 + 3 \epsilon - 4 \epsilon^ {3})
$$

Now we'll show that $p_1^* \leq 1.5\epsilon$ . To do this, we'll show $1.5\epsilon - p_1^* \geq 0$ :

$$
\begin{array}{l} 1. 5 \epsilon - p _ {1} ^ {*} = 1. 5 \epsilon - \frac {(1 + 1 5 \epsilon) ^ {1 / 2} - (1 + 3 \epsilon - 4 \epsilon^ {3}) ^ {1 / 2}}{4 (1 - \epsilon) ^ {3 / 2}} \\ = \frac {6 \epsilon (1 - \epsilon) ^ {3 / 2}}{4 (1 - \epsilon) ^ {3 / 2}} - \frac {(1 + 1 5 \epsilon) ^ {1 / 2} - (1 + 3 \epsilon - 4 \epsilon^ {3}) ^ {1 / 2}}{4 (1 - \epsilon) ^ {3 / 2}} \\ = \frac {6 \epsilon (1 - \epsilon) ^ {3 / 2} - (1 + 1 5 \epsilon) ^ {1 / 2} + (1 + 3 \epsilon - 4 \epsilon^ {3}) ^ {1 / 2}}{4 (1 - \epsilon) ^ {3 / 2}}. \\ \end{array}
$$

It suffices to show the numerator is non-negative on $[0, 1/3)$ :

$$
\begin{array}{l} 6 \epsilon (1 - \epsilon) ^ {3 / 2} - (1 + 1 5 \epsilon) ^ {1 / 2} + (1 + 3 \epsilon - 4 \epsilon^ {3}) ^ {1 / 2} \\ = 6 \epsilon (1 - \epsilon) (1 - \epsilon) ^ {1 / 2} - (1 + 1 5 \epsilon) ^ {1 / 2} + (1 + 2 \epsilon) (1 - \epsilon) ^ {1 / 2} \\ = (1 - \epsilon) ^ {1 / 2} [ 6 \epsilon (1 - \epsilon) + (1 + 2 \epsilon) ] - (1 + 1 5 \epsilon) ^ {1 / 2} \\ = (1 - \epsilon) ^ {1 / 2} (1 + 8 \epsilon - 6 \epsilon^ {2}) - (1 + 1 5 \epsilon) ^ {1 / 2} \\ = \left[ (1 - \epsilon) (1 + 8 \epsilon - 6 \epsilon^ {2}) ^ {2} \right] ^ {1 / 2} - (1 + 1 5 \epsilon) ^ {1 / 2} \\ = (1 + 1 5 \epsilon + 3 6 \epsilon^ {2} - 1 4 8 \epsilon^ {3} + 1 3 2 \epsilon^ {4} - 3 6 \epsilon^ {5}) ^ {1 / 2} - (1 + 1 5 \epsilon) ^ {1 / 2}. \\ \end{array}
$$

To show this is non-negative, it suffices to show $36\epsilon^2 - 148\epsilon^3 + 132\epsilon^4 - 36\epsilon^5$ is non-negative. Factoring yields

$$
3 6 \epsilon^ {2} - 1 4 8 \epsilon^ {3} + 1 3 2 \epsilon^ {4} - 3 6 \epsilon^ {5} = 4 \epsilon^ {2} (1 - 3 \epsilon) (9 - 1 0 \epsilon + 3 \epsilon^ {2}).
$$

Finally, we can see this is non-negative for $\epsilon \in (0,1/3)$ , so $p_1^* \leq 1.5\epsilon$ for $\epsilon \in (0,1/3)$ .

Now, to show $p_1^*$ is a stable fixed point, we can take the derivative of the cubic map at $p_2^*$ :

$$
\begin{array}{l} \frac {\partial}{\partial p} \left(3 / 4 \cdot [ \epsilon / 2 + (1 - \epsilon) p ] + [ \epsilon / 2 + (1 - \epsilon) p ] ^ {3}\right) \\ = \frac {\partial}{\partial p} \left[ p ^ {3} (1 - \epsilon) ^ {3} + \frac {3}{2} p ^ {2} \epsilon (1 - \epsilon) ^ {2} + \frac {3}{4} p (1 - \epsilon) (\epsilon^ {2} + 1) + \frac {1}{8} \epsilon^ {3} + \frac {3}{8} \epsilon \right] \\ = 3 (1 - \epsilon) ^ {3} p ^ {2} + 3 \epsilon (1 - \epsilon) ^ {2} p + \frac {3}{4} (1 - \epsilon) (1 + \epsilon^ {2}) \tag {13} \\ \end{array}
$$

Plugging in $p_{1}^{*}$ and simplifying yields

$$
\begin{array}{l} 3 (1 - \epsilon) ^ {3} \left(\frac {1}{4} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} - \frac {1 + 2 \epsilon}{4 (1 - \epsilon)}\right) ^ {2} + 3 \epsilon (1 - \epsilon) ^ {2} \left(\frac {1}{4} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} - \frac {1 + 2 \epsilon}{4 (1 - \epsilon)}\right) + \frac {3}{4} (1 - \epsilon) (1 + \epsilon^ {2}) \\ = 3 (1 - \epsilon) ^ {3} \left(\frac {1 + 1 5 \epsilon}{1 6 (1 - \epsilon) ^ {3}} + \frac {(1 + 2 \epsilon) ^ {2}}{1 6 (1 - \epsilon) ^ {2}} - \frac {1 + 2 \epsilon}{8 (1 - \epsilon)} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}}\right) + \frac {3}{4} \epsilon (1 - \epsilon) ^ {2} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} \\ - \frac {3}{4} \epsilon (1 - \epsilon) (1 + 2 \epsilon) + 3 / 4 (1 - \epsilon) (1 + \epsilon^ {2}) \\ = \frac {3 (1 + 1 5 \epsilon)}{1 6} + \frac {3 (1 - \epsilon) (1 + 2 \epsilon) ^ {2}}{1 6} - \frac {3 (1 - \epsilon) ^ {2} (1 + 2 \epsilon)}{8} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} + \frac {3}{4} \epsilon (1 - \epsilon) ^ {2} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} \\ - \frac {3}{4} \epsilon (1 - \epsilon) (1 + 2 \epsilon) + 3 / 4 (1 - \epsilon) (1 + \epsilon^ {2}) \\ = \frac {3}{1 6} (1 + 1 5 \epsilon) + \frac {3}{1 6} (1 - \epsilon) (1 + 2 \epsilon) ^ {2} + (1 - \epsilon) ^ {2} \left(\frac {3}{4} \epsilon - \frac {3}{8} (1 + 2 \epsilon)\right) \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} - \frac {3}{4} \epsilon (1 - \epsilon) (1 + 2 \epsilon) \\ + 3 / 4 (1 - \epsilon) (1 + \epsilon^ {2}) \\ = - \frac {3}{8} (1 - \epsilon) ^ {2} \sqrt {\frac {1 + 1 5 \epsilon}{(1 - \epsilon) ^ {3}}} + \frac {9}{8} + \frac {1 5}{8} \epsilon \\ = - \frac {3}{8} \sqrt {(1 + 1 5 \epsilon) (1 - \epsilon)} + \frac {9}{8} + \frac {1 5}{8} \epsilon . \\ \end{array}
$$

To see this is positive for $\epsilon < 1/3$ , note that $\frac{3}{8} \sqrt{(1 + 15\epsilon)(1 - \epsilon)} < \frac{3}{8} \sqrt{(1 + 15/3)} \approx 0.92 < 9/8$ . We can also show the derivative of the cubic map at $p_2^*$ is less than 1. To do this, we'll show that 1 minus the derivative at $p_1^*$ is positive:

$$
\begin{array}{l} 1 - \left(- \frac {3}{8} \sqrt {(1 + 1 5 \epsilon) (1 - \epsilon)} + \frac {9}{8} + \frac {1 5}{8} \epsilon\right) = \frac {3}{8} \sqrt {(1 + 1 5 \epsilon) (1 - \epsilon)} - \frac {1}{8} - \frac {1 5}{8} \epsilon \\ = \sqrt {\frac {9}{6 4} (1 + 1 5 \epsilon) (1 - \epsilon)} - \sqrt {\left(\frac {1}{8} + \frac {1 5}{8} \epsilon\right) ^ {2}}. \\ \end{array}
$$

By the monotonicity of square roots, it suffices to show that the following quadratic is positive:

$$
\begin{array}{l} \frac {9}{6 4} (1 + 1 5 \epsilon) (1 - \epsilon) - \left(\frac {1}{8} + \frac {1 5}{8} \epsilon\right) ^ {2} = - \frac {4 5 \epsilon^ {2}}{8} + \frac {3 \epsilon}{2} + \frac {1}{8} \\ = \frac {1}{8} (1 - 3 \epsilon) (1 5 \epsilon + 1). \\ \end{array}
$$

which we can see is positive for $\epsilon \in (0,1/3)$ . Thus, the derivative of the cubic map at $p_1^*$ is positive but less than 1, so $p_1^*$ is a stable fixed point. The fixed point at $1/2$ is unstable, in contrast: plugging $p_3^* = 1/2$ into the derivative (13) and simplifying yields $3/2(1 - \epsilon)$ , which is larger than 1 for $\epsilon < 1/3$ . Thus the cubic map converges to $p_1^*$ for initial values in $[0,1/2)$ .

Finally, to show the map is non-decreasing in $p$ , notice that derivative Equation (13) is non-negative for $p \geq 0$ and $\epsilon \in (0,1]$ .

Theorem 8. Let $F_0 \in \mathcal{F}$ . For any $\epsilon \in (0,1/3)$ and $x \in [0,1/2)$ , $\limsup_{t \to \infty} F_{3,t}^{\epsilon}(x) \leq 1.5\epsilon$ .

Proof. Let x < 1/2 and define $p = F_{3,t-1}^{\epsilon}(x)$ . With $\epsilon$ -uniform noise, $\Pr(X_{i,t} \leq x) = \epsilon x + (1 - \epsilon)p$ . The case analysis from Theorem 3 then proceeds exactly the same way, so we can replace p by $\epsilon x + (1 - \epsilon)p$ in the

bound from Theorem 3 to get the equivalent bound with $\epsilon$ -uniform noise:

$$
F _ {3, t} ^ {\epsilon} (x) \leq 3 / 4 \cdot [ \epsilon x + (1 - \epsilon) p ] + [ \epsilon x + (1 - \epsilon) p ] ^ {3}. \tag {14}
$$

While it would be possible to work directly with the cubic map (14), its fixed points are extremely messy. As such, we instead analyze the upper bound given by x < 1/2 and then use Lemma 5:

$$
F _ {3, t} ^ {\epsilon} (x) <   3 / 4 \cdot [ \epsilon / 2 + (1 - \epsilon) p ] + [ \epsilon / 2 + (1 - \epsilon) p ] ^ {3}. \tag {15}
$$

If $F_0(x) < 1/2$ , then Lemma 6 states that the map (15) upper bounding $F_{3,t}^{\epsilon}(x)$ converges to $p^{*} \leq 1.5\epsilon$ . Thus, applying Lemma 5 gives $\limsup_{t \to \infty} F_{3,t}^{\epsilon}(x) \leq p^{*} \leq 1.5\epsilon$ as claimed. If $F_0(x) = 1/2$ (which is possible since we don't require that $F_0$ is positive near $1/2$ ), then applying (15),

$$
\begin{array}{l} F _ {3, 1} ^ {\epsilon} (x) <   3 / 4 \cdot [ \epsilon / 2 + (1 - \epsilon) / 2 ] + [ \epsilon / 2 + (1 - \epsilon) / 2 ] ^ {3} \\ = 3 / 4 \cdot 1 / 2 + [ 1 / 2 ] ^ {3} \\ = 1 / 2. \\ \end{array}
$$

Thus $F_{3,1}^{\epsilon}(x) < 1/2$ , so we can apply Lemmas 5 and 6 with initial $p = F_{3,1}^{\epsilon}(x)$ rather than $F_0(x)$ .

![](images/073aa9231c333faaa556852d1aaca7a589aa245fa6bd4b314c60c11efc458dff.jpg)

Lemma 4. Let $F_0 \in \mathcal{F}$ . With $\epsilon$ -uniform noise, for any $\epsilon \in (0,1]$ , $x \in (1/3,1/2)$ , and $t > 0$ ,

$$
F _ {4, t} ^ {\epsilon} (x) \leq \epsilon x + (1 - \epsilon) F _ {4, t - 1} ^ {\epsilon} (x).
$$

Thus, $F_{4,t}^{\epsilon}(x) \leq \max \{x, F_{4,0}^{\epsilon}(x)\}$ .

Proof. Let $x \in (1/3, 1/2)$ . Just as in Theorem 8, we can take the bound from Lemma 1 and replace $F_{4,t-1}^{\epsilon}(x)$ with $\epsilon x + (1 - \epsilon)F_{4,t-1}^{\epsilon}(x)$ to get the claimed upper bound with $\epsilon$ -uniform noise. The second part of the claim follows by induction after noting $F_{4,1}^{\epsilon}(x) \leq \epsilon x + (1 - \epsilon)F_{4,0}^{\epsilon}(x) \leq \epsilon \max\{x, F_{4,0}^{\epsilon}(x)\} + (1 - \epsilon) \max\{x, F_{4,0}^{\epsilon}(x)\} = \max\{x, F_{4,0}^{\epsilon}(x)\}$ .

Theorem 9. Let $F_0 \in \mathcal{F}$ . For any $\epsilon \in (0,1]$ and $x \in (1/3,1/2)$ , let $\beta = 1/2 - \epsilon(x/3 + 1/3) - (1 - \epsilon) \max\{x/3 + 1/3, F_0(x/3 + 1/3)\}$ . Then $\beta \in (0,1/2]$ and $\limsup_{t \to \infty} F_{4,t}^{\epsilon}(x) \leq \frac{1}{8\beta^3}\epsilon$ .

Proof. Let $p = F_{4,t-1}^{\epsilon}(x)$ . By Lemma 4 and symmetry, an inner candidate in $(x, 1 - x)$ wins with probability at least $1 - 2p(1 - \epsilon) - 2x\epsilon$ . We'll strengthen this bound in the same way as in Theorem 4, using the case with three candidates in $(x/3 + 1/3, 1/2)$ and one in $(1 - x, 1]$ (and the symmetric case). With $\epsilon$ -uniform noise and accounting for symmetry, the probability this case occurs is

$$
8 [ \epsilon x + (1 - \epsilon) p ] [ 1 / 2 - \epsilon (x / 3 + 1 / 3) - (1 - \epsilon) F _ {4, t - 1} ^ {\epsilon} (x / 3 + 1 / 3) ] ^ {3}
$$

$$
\geq 8 [ \epsilon x + (1 - \epsilon) p ] [ 1 / 2 - \epsilon (x / 3 + 1 / 3) - (1 - \epsilon) \max \{x / 3 + 1 / 3, F _ {0} (x / 3 + 1 / 3) \} ] ^ {3} \quad (\text { by   Lemma4 })
$$

$$
= 8 [ \epsilon x + (1 - \epsilon) p ] \beta^ {3}.
$$

Then, adding this case to the cases implicitly used in Lemma 4 (see Lemma 1 for the list of cases), we find

$$
\operatorname * {P r} (x <   \text { Plurality } (X _ {1, t} ^ {\epsilon}, X _ {2, t} ^ {\epsilon}, X _ {3, t} ^ {\epsilon}, X _ {4, t} ^ {\epsilon}) <   1 - x)
$$

$$
\geq 1 - 2 p (1 - \epsilon) - 2 x \epsilon + 8 [ \epsilon x + (1 - \epsilon) p ] \beta^ {3}.
$$

By symmetry,

$$
F _ {4, t} ^ {\epsilon} (x) = \left[ 1 - \operatorname * {P r} (x <   \text {Plurality} (X _ {1, t} ^ {\epsilon}, X _ {2, t} ^ {\epsilon}, X _ {3, t} ^ {\epsilon}, X _ {4, t} ^ {\epsilon}) <   1 - x) \right] / 2
$$

$$
\leq p (1 - \epsilon) + x \epsilon - 4 [ \epsilon x + (1 - \epsilon) p ] \beta^ {3}
$$

$$
= p (1 - \epsilon) (1 - 4 \beta^ {3}) + \epsilon x (1 - 4 \beta^ {3}). \tag {16}
$$

We'll show that the iterated map (16) upper bounding $F_{4,t}^{\epsilon}(x)$ converges to a fixed point upper bounded by $\frac{1}{8\beta}\epsilon$ and then apply Lemma 5. First, we'll find the fixed point (unique, since this is a linear map):

$$
p ^ {*} (1 - \epsilon) (1 - 4 \beta^ {3}) + \epsilon x (1 - 4 \beta^ {3}) = p ^ {*}
$$

$$
\Leftrightarrow p ^ {*} [ (1 - \epsilon) (1 - 4 \beta^ {3}) - 1 ] + \epsilon x (1 - 4 \beta^ {3}) = 0
$$

$$
\Leftrightarrow p ^ {*} = \frac {\epsilon x (1 - 4 \beta^ {3})}{1 - (1 - \epsilon) (1 - 4 \beta^ {3})}. \tag {17}
$$

To show convergence to $p^*$ , it suffices to show that the slope of the map is in $(-1, 1)$ (any such linear map converges to its unique fixed point, e.g., by the Banach fixed-point theorem). First, we can show $\beta \in (0, 1/2]$ :

$$
\beta = 1 / 2 - \epsilon (x / 3 + 1 / 3) - (1 - \epsilon) \max \{x / 3 + 1 / 3, F _ {0} (x / 3 + 1 / 3) \}
$$

$$
> 1 / 2 - \epsilon (1 / 2) - (1 - \epsilon) \max \{1 / 2, F _ {0} (x / 3 + 1 / 3) \} \quad (\text {since} x <   1 / 2)
$$

$$
= 1 / 2 - \epsilon (1 / 2) - (1 - \epsilon) (1 / 2) \quad (\text { since } F _ {0} (x / 3 + 1 / 3) \leq 1 / 2)
$$

$$
= 0.
$$

Thus, the slope $(1 - \epsilon)(1 - 4\beta^3) \in [0,1)$ , so the map (16) converges to $p^*$ for all initial values $p$ and is non-decreasing in $p$ . Now we can upper bound $p^*$ :

$$
p ^ {*} = \frac {\epsilon x (1 - 4 \beta^ {3})}{1 - (1 - \epsilon) (1 - 4 \beta^ {3})}
$$

$$
<   \frac {\epsilon / 2}{1 - (1 - 4 \beta^ {3})} \quad (\epsilon \geq 0, x <   1 / 2)
$$

$$
= \frac {\epsilon}{8 \beta^ {3}}.
$$

Thus, by Lemma 5, $\limsup_{t\to\infty}F_{4,t}^{\epsilon}(x)\leq p^{*}<\frac{1}{8\beta^{3}}\epsilon$ .

Theorem 10. Let $F_0 \in \mathcal{F}$ . For $k \geq 5$ , the candidate distribution does not approximately converge to the center under replicator dynamics with $\epsilon$ -uniform noise.

Proof. Suppose for a contradiction that the candidate distribution does approximately converge to the center. That is, suppose that for all c > 0 and x < 1/2, there exists some $\epsilon_{max} > 0$ such that with $\epsilon$ -uniform noise, for any $\epsilon \in (0, \epsilon_{\max}]$ , $\limsup_{t \to \infty} F_{k,t}^{\epsilon}(x) < c$ . If $\limsup_{t \to \infty} F_{k,t}^{\epsilon}(x) < c$ , then there is some $t^{*}$ such that for all $t \geq t^{*}$ , $F_{k,t}^{\epsilon}(x) \leq c$ . In particular, let $\epsilon_{max}^{*}$ and $t^{*}$ be the corresponding values for x = 1/4. Then for any $\epsilon$ -uniform noise with $\epsilon \leq \epsilon_{max}^{*}$ , $F_{k,t^{*}}^{\epsilon}(1/4) \leq c$ . Additionally, consider the point $z = (F_{k,t^{*}}^{\epsilon})^{-1}(1/4)$ . By our assumption, there is some $\epsilon_{max}^{\prime}$ and $t^{\prime}$ such that if $\epsilon < \epsilon_{max}^{\prime}$ , then $F_{k,t}^{\epsilon}(z) \leq c$ for all $t \geq t^{\prime}$ . We can make $\epsilon$ and c as small as needed, so we'll pick:

$$
c <   \left[ 1 - (1 2 5 / 1 2 8) ^ {1 / k} \right] / 3 \tag {18}
$$

$$
\epsilon <   \min \left\{\epsilon_ {\max} ^ {*}, \epsilon_ {\max} ^ {\prime}, \left[ 1 - (1 2 5 / 1 2 8) ^ {1 / k} \right] / 3 \right\}. \tag {19}
$$

Note that $[1 - (125 / 128)^{1 / k}] / 3$ is largest at $k = 5$ , when its value is approximately 0.0016. Also note that we must have $t' > t^*$ , since $F_{k,t^*}^\epsilon (z) = 1 / 4 > c$ .

Now, we can apply the same argument as in Theorem 5, finding a lower bound on $F_{4,t^* +1}^\epsilon (x)$ (for $x\in (1 / 4,1 / 2)$ ) given that only a $c$ -fraction of the winners in generation $t^*$ are left of $1 / 4$ (note that this parameter was called $\alpha$ in the proof of Theorem 5). Let $p = F_{k,t^*}^\epsilon (x)$ with $\epsilon$ -uniform noise. For brevity, we will avoid repeating the argument from Theorem 5 and instead substitute directly into the resulting

bound (10). Replacing $p$ with $\Pr(X_{i,t^*}^\epsilon \leq x) = \epsilon x + (1 - \epsilon)p$ and $\alpha$ with $\Pr(X_{i,t^*}^\epsilon \leq 1/4) \leq \epsilon/4 + (1 - \epsilon)c$ in (10) then yields

$$
\begin{array}{l} F _ {4, t ^ {*} + 1} ^ {\epsilon} (x) \geq [ \epsilon x + (1 - \epsilon) p ] ^ {k} + (1 - 2 [ \epsilon / 4 + (1 - \epsilon) c ]) ^ {k} / 2 \\ - (1 - [ \epsilon / 4 + (1 - \epsilon) c ] - [ \epsilon x + (1 - \epsilon) p ]) ^ {k} + (1 - 2 [ \epsilon x + (1 - \epsilon) p ]) ^ {k} / 2. \tag {20} \\ \end{array}
$$

We will now derive a contradiction: that $F_{k,t}^{\epsilon}(z)$ never goes below 1/4 as t increases from $t^{*}$ to $t'$ , when it should go below c. We know z > 1/4 by the monotonicity of the CDF, since $F_{k,t^{*}}^{\epsilon}(1/4) \leq c$ . So, we can apply the lower bound (20) to z (where p = 1/4):

$$
\begin{array}{l} F _ {k, t ^ {*} + 1} ^ {\epsilon} (z) \geq [ \epsilon z + (1 - \epsilon) / 4 ] ^ {k} + (1 - 2 [ \epsilon / 4 + (1 - \epsilon) c ]) ^ {k} / 2 \\ - (1 - [ \epsilon / 4 + (1 - \epsilon) c ] - [ \epsilon z + (1 - \epsilon) / 4 ]) ^ {k} + (1 - 2 [ \epsilon z + (1 - \epsilon) / 4 ]) ^ {k} / 2 \\ > (1 - 2 [ \epsilon / 4 + (1 - \epsilon) c ]) ^ {k} / 2 - (1 - [ \epsilon / 4 + (1 - \epsilon) c ] - [ \epsilon z + (1 - \epsilon) / 4 ]) ^ {k} \\ = (1 - \epsilon / 2 - 2 (1 - \epsilon) c) ^ {k} / 2 - (1 - [ \epsilon / 4 + (1 - \epsilon) c ] - [ \epsilon z + (1 - \epsilon) / 4 ]) ^ {k} \\ > (1 - \epsilon - 2 c) ^ {k} / 2 - (1 - (1 - \epsilon) / 4) ^ {k} \\ \geq (1 - \epsilon - 2 c) ^ {k} / 2 - (3 / 4 + \epsilon / 4) ^ {5}. \tag {21} \\ \end{array}
$$

Now consider each term in (21). By our upper bounds on $c$ and $\epsilon$ ,

$$
\begin{array}{l} (1 - \epsilon - 2 c) ^ {k} / 2 > \left(1 - \left[ 1 - (1 2 5 / 1 2 8) ^ {1 / k} \right] / 3 - 2 \left[ 1 - (1 2 5 / 1 2 8) ^ {1 / k} \right] / 3\right) ^ {k} / 2 \\ = \left(1 - \left[ 1 - (1 2 5 / 1 2 8) ^ {1 / k} \right]\right) ^ {k} / 2 \\ = 1 2 5 / 2 5 6. \\ \end{array}
$$

Meanwhile, $\epsilon$ is also small enough that $(3/4 + \epsilon/4)^{5} < \frac{244}{1024}$ (solving for $\epsilon$ reveals we need $\epsilon < 0.0024$ , which we have ensured). This then means that $(1 - \epsilon - 2c)^{k}/2 - (3/4 + \epsilon/4)^{5} > \frac{125}{256} - \frac{244}{1024} = 1/4$ . Following the above chain of inequalities, this shows that $F_{k,t^{*}+1}^{\epsilon}(z) > 1/4$ .

We will now show by induction that for all $t > t^{*}$ , $F_{k,t}^{\epsilon}(z) > 1/4$ , a contradiction (since by our assumption, $F_{k,t'}^{\epsilon}(z) \leq c$ with $t' > t^{*}$ ). We have just shown the base case $t = t^{*} + 1$ above. Then, suppose as an inductive hypothesis that $F_{k,t}^{\epsilon}(z) \geq 1/4$ for $t > t^{*}$ . Define $w = (F_{k,t}^{\epsilon})^{-1}(1/4)$ ; by the inductive hypothesis, we know $w \leq z$ . By the argument above, $F_{k,t+1}^{\epsilon}(w) > 1/4$ (the argument only requires that $F_{k,t}^{\epsilon}(1/4) \leq c$ and $w \in (1/4, 1/2)$ , which are satisfied here). Thus, by the monotonicity of the CDF, $F_{k,t+1}^{\epsilon}(z) > 1/4$ . By induction, we then have $F_{k,t'}^{\epsilon}(z) > 1/4$ , a contradiction.

# B.3 Proofs from Section 4

Before proving Theorem 11, we need a few supporting lemmas. We begin with a result from the proof of Theorem 5, specialized to the case when all candidates are in $(1/4,3/4)$ .

Lemma 7. Suppose $F_{0} \in F$ is supported on $(1/4, 3/4)$ . For $k \geq 5$ and $x \in (1/4, 1/2)$ ,

$$
F _ {k, t} (x) \geq 1 / 2 + F _ {k, t - 1} (x) ^ {k} - \left(1 - F _ {k, t - 1} (x)\right) ^ {k} + \left(1 - 2 F _ {k, t - 1} (x)\right) ^ {k} / 2.
$$

Proof. We can use the same argument as in Theorem 5 to find a lower bound on $F_{k,t}(x)$ , but now $F_{k,t-1}(x) \leq \alpha = 0$ since $F_0$ (and therefore all subsequent $F_{k,t}$ ) is supported only on $(1/4, 3/4)$ . Plugging $\alpha = 0$ into (10) yields the claim, noting $p = F_{k,t-1}(x)$ .

This gives us an iterated map which bounds $F_{k,t}(x)$ from below. We can show that this map converges to 1/2 in a large interval around 1/2, meaning that the candidate distribution converges to one with no mass in this interval. We cannot give an explicit form for the basin of attraction of this map since it depends on a root of a polynomial of order k, but we can show the interval grows in k and characterize it for k = 5.

Lemma 8. For all $k \geq 5$ , the iterated map given by $p' = 1/2 + p^k - (1 - p)^k + (1 - 2p)^k / 2$ converges to $1/2$ for all initial $p \in ([1 - \sqrt{3/7}] / 2 = 0.172 \ldots, 1/2]$ . Moreover, this map in non-decreasing in $p$ on $[0, 1/2)$ .

Proof. First, we'll show $1/2$ is a stable fixed point of the map. Indeed, $1/2 + (1/2)^k - (1 - 1/2)^k + (1 - 2(1/2))^k / 2 = 1/2 + 1/2^k - 1/2^k = 1/2$ . The stability of this fixed point is determined by the derivative

$$
\frac {\partial}{\partial p} \left(1 / 2 + p ^ {k} - (1 - p) ^ {k} + (1 - 2 p) ^ {k} / 2\right) = k p ^ {k - 1} + k (1 - p) ^ {k - 1} - k (1 - 2 p) ^ {k - 1}. \tag {22}
$$

At $1/2$ , the derivative is $k(1/2)^{k-1} + k(1 - 1/2)^{k-1} - k(1 - 1)^{k-1} = k(1/2)^{k-2}$ . For $k \geq 5$ , $k(1/2)^{k-2} < 1$ , showing the fixed point is stable.

For $k = 5$ , we can find the other fixed points of the map by factoring:

$$
\begin{array}{l} 1 / 2 + p ^ {5} - (1 - p) ^ {5} + (1 - 2 p) ^ {5} / 2 = p \\ \Leftrightarrow 1 / 2 + p ^ {5} - (1 - p) ^ {5} + (1 - 2 p) ^ {5} / 2 - p = 0 \\ \Leftrightarrow p (1 - p) (1 - 2 p) \left(- 7 p ^ {2} + 7 p - 1\right) = 0. \\ \Leftrightarrow p \in \left\{0, (1 - \sqrt {3 / 7}) / 2, 1 / 2, (1 + \sqrt {3 / 7}) / 2, 1 \right\} = \{0, 0. 1 7 2 \dots , 0. 5, 0. 8 2 7 \dots , 1 \}. \\ \end{array}
$$

Plugging in the $k = 5$ fixed point $(1 - \sqrt{3/7})/2 = 0.172 \ldots$ to the derivative (22) yields $\approx 1.43$ , so this fixed point in unstable. Next, note that the map monotonically increases in $p$ for $p \in (0,1/2)$ , since the derivative (22) is positive (as $1 - p > 1 - 2p$ ; similarly, the map is non-increasing on $[0,1/2)$ , as claimed). Thus, for $k = 5$ , the map is larger than $p$ but smaller than $1/2$ for $p$ in $([1 - \sqrt{3/7}]/2, 1/2)$ and initial values in this range converge to the stable fixed point $1/2$ .

The final step is to show the map is increasing in $k$ for $p \in (0,1/2)$ , which means that the basin of attraction only grows in $k$ . To do this, consider the derivative of the map with respect to $k$ :

$$
\frac {\partial}{\partial k} \left(1 / 2 + p ^ {k} - (1 - p) ^ {k} + (1 - 2 p) ^ {k} / 2\right) = \frac {1}{2} (1 - 2 p) ^ {k} \log (1 - 2 p) - (1 - p) ^ {k} \log (1 - p) + p ^ {k} \log p.
$$

We establish this is positive in the following lemma.

Lemma 9. For all $k \geq 3$ and 0 < p < 1/2,

$$
1 / 2 (1 - 2 p) ^ {k} \log (1 - 2 p) - (1 - p) ^ {k} \log (1 - p) + p ^ {k} \log p > 0.
$$

Proof. We thank River Li $^{6}$ for a key idea behind this analysis, based on the following integral trick:

$$
\begin{array}{l} \int_ {0} ^ {1} \frac {x - 1}{1 + t (x - 1)} d t = \log (1 + t (x - 1)) \big | _ {t = 0} ^ {1} \\ = \log (1 + 1 (x - 1)) - \log (1 + 0 (x - 1)) \\ = \log x. \\ \end{array}
$$

Now, apply this identity to the function in question for $k = 3$ :

$$
\begin{array}{l} 1 / 2 (1 - 2 p) ^ {3} \log (1 - 2 p) - (1 - p) ^ {3} \log (1 - p) + p ^ {3} \log p \\ = 1 / 2 (1 - 2 p) ^ {3} \int_ {0} ^ {1} \frac {- 2 p}{1 + t (- 2 p)} d t - (1 - p) ^ {3} \int_ {0} ^ {1} \frac {- p}{1 + t (- p)} d t + p ^ {3} \int_ {0} ^ {1} \frac {p - 1}{1 + t (p - 1)} d t \\ = \int_ {0} ^ {1} \left(- \frac {p (1 - 2 p) ^ {3}}{1 - 2 p t} + \frac {p (1 - p) ^ {3}}{1 - p t} - \frac {p ^ {3} (1 - p)}{1 + p t - t}\right) d t. \\ \end{array}
$$

We'll show that the integrand is positive for all $t \in [0,1]$ , which implies the integral is also positive. Converting to a common denominator,

$$
\begin{array}{l} - \frac {p (1 - 2 p) ^ {3}}{1 - 2 p t} + \frac {p (1 - p) ^ {3}}{1 - p t} - \frac {p ^ {3} (1 - p)}{1 + p t - t} \\ = \frac {- p (1 - 2 p) ^ {3} (1 - p t) (1 + p t - t) + p (1 - p) ^ {3} (1 - 2 p t) (1 + p t - t) - p ^ {3} (1 - p) (1 - 2 p t) (1 - p t)}{(1 - 2 p t) (1 - p t) (1 + p t - t)}. \\ \end{array}
$$

Since $0 < p < 1/2$ and $0 \leq t \leq 1$ , the denominator is positive, so we just need to show the numerator is positive. We can factor:

$$
\begin{array}{l} - p (1 - 2 p) ^ {3} (1 - p t) (1 + p t - t) + p (1 - p) ^ {3} (1 - 2 p t) (1 + p t - t) - p ^ {3} (1 - p) (1 - 2 p t) (1 - p t) \\ = p ^ {2} (1 - 2 p) (3 - 4 p - 4 t + 4 p t + p ^ {2} t + t ^ {2} + p t ^ {2} - 4 p ^ {2} t ^ {2} + 2 p ^ {3} t ^ {2}) \\ = p ^ {2} (1 - 2 p) \left[ (1 + p - 4 p ^ {2} + 2 p ^ {3}) t ^ {2} - (4 - 4 p - p ^ {2}) t + 3 - 4 p \right]. \\ \end{array}
$$

Again, since $p^2$ and $(1 - 2p)$ are positive, we just need to show the right factor is positive. Now, notice that $4 - 4p - p^2 > 0$ (since $p < 1/2$ ), and $t \leq \frac{t^2 + 1}{2}$ , so

$$
\begin{array}{l} (1 + p - 4 p ^ {2} + 2 p ^ {3}) t ^ {2} - (4 - 4 p - p ^ {2}) t + 3 - 4 p \geq (1 + p - 4 p ^ {2} + 2 p ^ {3}) t ^ {2} - (4 - 4 p - p ^ {2}) \frac {t ^ {2} + 1}{2} + 3 - 4 p \\ = \frac {1}{2} \left[ 2 - 4 p + p ^ {2} - (2 - 6 p + 7 p ^ {2} - 4 p ^ {3}) t ^ {2} \right] \\ \end{array}
$$

Now, $2 - 6p + 7p^2 - 4p^3$ is positive for $p \in (0,1/2)$ . We can see this since its derivative, $-6 + 14p - 12p^2$ is negative (achieving a maximum of $-23/12$ at $p = 7/12$ ) and the polynomial has a zero at $p = 1/2$ . Thus, we can shrink the function by replacing $t^2$ by 1:

$$
\begin{array}{l} \frac {1}{2} \left[ 2 - 4 p + p ^ {2} - (2 - 6 p + 7 p ^ {2} - 4 p ^ {3}) t ^ {2} \right] \geq \frac {1}{2} \left[ 2 - 4 p + p ^ {2} - (2 - 6 p + 7 p ^ {2} - 4 p ^ {3}) \right] \\ = p (1 - p) (1 - 2 p). \\ \end{array}
$$

Finally, we see that this is positive for all $p \in (0,1)$ , which implies that

$$
1 / 2 (1 - 2 p) ^ {3} \log (1 - 2 p) - (1 - p) ^ {3} \log (1 - p) + p ^ {3} \log p > 0.
$$

We can now use this as a base case $k = 3$ in an inductive argument. For the inductive case ( $k \geq 3$ ), suppose

$$
1 / 2 (1 - 2 p) ^ {k} \log (1 - 2 p) - (1 - p) ^ {k} \log (1 - p) + p ^ {k} \log p > 0.
$$

Note that the first and third terms are negative, while the middle term is positive (because of the logs). So, let $x = \min \{1/2(1 - 2p)^k \log(1 - 2p), p^k \log p\}$ . We then have

$$
\begin{array}{l} 1 / 2 (1 - 2 p) ^ {k + 1} \log (1 - 2 p) - (1 - p) ^ {k + 1} \log (1 - p) + p ^ {k + 1} \log p \\ \geq (1 - 2 p) x - (1 - p) ^ {k + 1} \log (1 - p) + p x \quad (\text { replace   both   terms   by   their   minimum }) \\ = (1 - p) x - (1 - p) (1 - p) ^ {k} \log (1 - p) \\ \geq (1 - p) 1 / 2 (1 - 2 p) ^ {k} \log (1 - 2 p) - (1 - p) (1 - p) ^ {k} \log (1 - p) + (1 - p) p ^ {k} \log p \\ = (1 - p) \left[ 1 / 2 (1 - 2 p) ^ {k} \log (1 - 2 p) - (1 - p) ^ {k} \log (1 - p) + p ^ {k} \log p \right] \\ > 0. \quad (\text {by IH}) \\ \end{array}
$$

The claim then holds for all $k \geq 3$ by induction.

![](images/847f4ad4fcfb80f8d1bad2b1b1adf1097ad81e598cc4517a1bb03175fbc10b84.jpg)

Therefore, since the map only increases in k, the basin of attraction for the stable fixed point at 1/2 can only grow as k increases from 5. □

Theorem 11. Suppose $F_0 \in \mathcal{F}$ is supported on $(1/4, 3/4)$ . Let $\ell = [1 - \sqrt{3/7}] / 2 = 0.172 \ldots$ . For $k \geq 5$ and $x \in (F_0^{-1}(\ell), 1/2)$ , $\lim_{t \to \infty} F_{k,t}(x) = 1/2$ .

Proof. Applying Lemma 5 to the bound from Lemma 7 and the convergence and monotonicity from Lemma 8 gives $\liminf_{t\to\infty}F_{k,t}(x)\geq1/2$ . Meanwhile, $F_{k,t}(x)\leq1/2$ for all $t$ by symmetry, so $\limsup_{t\to\infty}F_{k,t}(x)\leq1/2$ . Therefore $\lim_{t\to\infty}F_{k,t}(x)=1/2$ .

Theorem 12. Suppose $F_{0} \in F$ is supported on $(1/4, 3/4)$ . For any $k \geq 2$ and $t \geq 0$ ,

$$
f _ {k, t} (1 / 2) = f _ {0} (1 / 2) \cdot \left[ k (1 / 2) ^ {k - 2} \right] ^ {t}. \tag {9}
$$

Proof. By Lemma 2, only the left- or rightmost candidate can win. Thus, if a candidate at $1/2$ is the winner, all other candidates must either be left of $1/2$ or right of $1/2$ . Moreover, if all other candidates are on one side, then a candidate at $1/2$ wins. Thus, a candidate at $1/2$ wins if and only if all other candidates fall on the left or the right. Note that multiple candidates are at $1/2$ with probability 0, since the candidate distribution is atomless. By symmetry, this occurs with probability $2 \cdot (1/2)^{k-1} = (1/2)^{k-2}$ . Therefore $\Pr(\text{Plurality}(1/2, X_{2,t}, \ldots, X_{k,t})) = (1/2)^{k-2}$ . By Equation (2), we then have

$$
f _ {k, t} (1 / 2) = k \cdot f _ {k, t - 1} (1 / 2) \cdot \operatorname * {P r} (\text { Plurality } (1 / 2, X _ {2, t}, \dots , X _ {k, t}))
$$

$$
= k \cdot f _ {k, t - 1} (1 / 2) \cdot (1 / 2) ^ {k - 2}.
$$

We can now prove the claim by induction on t. For t = 0, indeed $f_{k,0}(1/2) = f_{k,0}(1/2) \cdot \left[k(1/2)^{k-2}\right]^{0}$ . For $t \geq 1$ , applying the inductive hypothesis to the above inequality yields

$$
f _ {k, t} (1 / 2) = k \cdot f _ {k, t - 1} (1 / 2) \cdot (1 / 2) ^ {k - 2}
$$

$$
= k \cdot f _ {k, 0} (1 / 2) \cdot \left[ k (1 / 2) ^ {k - 2} \right] ^ {t - 1} \cdot (1 / 2) ^ {k - 2}
$$

$$
= f _ {k, 0} (1 / 2) \cdot \left[ k (1 / 2) ^ {k - 2} \right] ^ {t}.
$$

# B.4 Proofs from Section 7

Theorem 13. Suppose $F_0$ places probability mass $p$ at $1/2$ . For any $k \geq 2$ , there is some $p_k^* < 1$ such that if $p > p_k^*$ , the candidate distribution converges to a point mass at $1/2$ under the replicator dynamics with left-right tie-breaking. One of the fixed points of $p^k + kp^{k-1}(1 - p)$ is such a $p_k^*$ .

Proof. Let $p'$ denote the mass at 1/2 in generation $t + 1$ . If all $k$ candidates are at 1/2, then so is the winner. Similarly, if all but one candidate are at 1/2, then the lone deviant loses with vote share less than 1/2 (with left-right tie-breaking). Thus, $p' \geq p^k + kp^{k-1}(1 - p)$ . For any $k$ , this lower bound is larger than $p$ for $p$ sufficiently close to 1. To see this, take the derivative at $p = 1$ : $\frac{d}{dp} [p^k + kp^{k-1}(1 - p)] = kp^{k-1} + k(k - 1)p^{k-2} - k^2p^{k-1}$ , which is 0 at $p = 1$ . Thus, for any small enough $\epsilon$ , $p^k + kp^{k-1}(1 - p)$ is larger than $1 - \epsilon$ when evaluated at $1 - \epsilon$ . Thus, $p$ will converge to 1 by the monotone convergence theorem.

Theorem 14. With $k \geq 4$ and left-right tie-breaking, for any $x \in (1/4, 1/2)$ , the strategy where each candidate picks uniformly at random between x and 1 - x is a SMSNE.

Proof. By symmetry, every candidate has a $1 / k$ win probability if they all follow this strategy. Suppose a deviant chooses a distribution that is supported on a point besides $x$ and $1 - x$ . If they choose a point between $x$ and $1 - x$ , they lose unless all other candidates pick the same side, which occurs w.p. $2(1 / 2)^{k - 1} = 1 / 2^{k - 2}$ . For $k \geq 4$ , this is at most $1 / k$ (and strictly less for $k > 4$ ), so sampling points in $(x, 1 - x)$ does not increase with probability. Alternatively, if the deviant samples a point in $[0, x)$ (or symmetrically, $(1 - x, 1))$ , they certainly lose unless no candidates pick $x$ , which occurs with probability $1 / 2^{k - 1}$ —smaller than $1 / k$ for $k \geq 4$ .

Thus deviating to a point left of $x$ only hurts. Combining the above findings, a deviant does not benefit by sampling any point other than $x$ or $1 - x$ . Finally, a deviant does not benefit by changing the probability with which they sample either point by symmetry of the other candidates' choices. Since no deviation is beneficial, the strategy is a Nash equilibrium.

Theorem 15. Suppose $F_0$ places probability mass $p$ at $x$ and at $1 - x$ , for $1/4 < x < 1/2$ . For any $k \geq 5$ , there exists some $p_k^* < 1/2$ such that if $p > p_k^*$ , the candidate distribution converges to point masses at $x$ and $1 - x$ under the replicator dynamics. In particular, one of the fixed points of $(2p)^k / 2 + k(1 - 2p)((2p)^{k-1} - 2p^{k-1}) / 2$ is such a $p_k^*$ .

Proof. Let $p'$ denote the mass at x in generation $t+1$ . If all candidates are at x or $1-x$ (w.p. $(2p)^{k}$ ), then a candidate at x wins with probability 1/2 by symmetry. Alternatively, suppose all but one candidate are at x or 1-x. The probability that there is at least one candidate at both x and 1-x and a wildcard is $k(1-2p)((2p)^{k-1}-2p^{k-1})$ . In such a case, the wildcard loses if they are in the middle (since they get vote share less than 1/4) and they lose if they are on the outside (to the opposite outside candidate). Thus, $p'\geq(2p)^{k}/2+k(1-2p)((2p)^{k-1}-2p^{k-1})/2$ . For $k\geq5$ , this is larger than p for p sufficiently close to 1/2. To see this, take the derivative at p=1/2: $f'(p)=\frac{d}{dp}\left[(2p)^{k}/2+k(1-2p)((2p)^{k-1}-2p^{k-1})/2\right]$ . We then find $f'(1/2)=2^{2-k}k$ , which is smaller than 1 for $k\geq5$ . Thus, p will converge to 1/2 by the monotone convergence theorem. □

Theorem 17. The following are (some $^{7}$ of the) PSNEs with uniform voters, complete plurality maximizing candidates, and left–right tie-breaking:

1. Any $k \geq 2$ : all $k$ candidates at $1/2$ .   
2. Any $k \geq 4$ : for any $x \in (1/4, 1/2)$ , $\lfloor k/2 \rfloor$ candidates at $x$ , $\lfloor k/2 \rfloor$ candidates at $1 - x$ , and the last candidate (if $k$ is odd) at either $x$ or $1 - x$ .   
3. Any $k \geq 5$ : $\lfloor (k - 1) / 2 \rfloor$ candidates at $1/4$ , $\lfloor (k - 1) / 2 \rfloor$ candidates at $3/4$ , one candidate at $1/2$ , and the last candidate (if $k$ is even) at either $1/4$ or $3/4$ .   
4. Even $k$ : Cox's equilibrium; two candidates at each of the points $1 / k, 3 / k, \ldots, (k - 1) / k$ .

Proof. We show in each case than no deviation is beneficial.

1. If all k candidates are at 1/2, then the winner is chosen uniformly from the leftmost and rightmost candidate at 1/2, who each get vote share 1/2. If any one candidate moves to some point away from 1/2, they get vote share strictly less than 1/2, while the middle candidate opposite them gets vote share 1/2 and wins. Thus, no candidate can benefit by deviating.   
2. Since $k \geq 4$ , both points x and 1 - x have at least two candidates. The candidates who end up being the outermost at x and 1 - x each get vote share x, while the innermost candidates get vote share 1/2 - x, which is strictly smaller since x > 1/4. Any candidate who moves towards the edge gets vote share strictly less than x and loses to the other side. Any candidate who moves into $(x, 1 - x)$ gets vote share 1/2 - x and loses. Finally, no candidate benefits by moving from x to 1 - x (or vice-versa), since they will always be in a lottery to be the outermost which has at least as many candidates as their original point (even if k is odd and a deviant moves from the more populated point).   
3. Since $k \geq 5$ , both points 1/4 and 3/4 have at least two candidates. There is a three-way tie with vote share 1/4 between the leftmost candidate, the rightmost candidate, and the one at 1/2—the inner candidates at 1/4 and 3/4 get vote share strictly less than 1/4. As in the previous case, every candidate certainly loses if they move left of 1/4 or right of 3/4. The side candidates also lose if they move into $(1/4, 3/4)$ . As before, there is no benefit to switching from 1/4 to 3/4 or vice-versa. Finally, the candidate at 1/2 only shrinks their win probability by moving to 1/4 or 3/4 (and worsens their

margin against a competitor by moving to any other point in $(1/4,3/4)$ . Thus, no candidate benefits by deviating.

4. Every candidate gets vote share 1/k and has a chance to win. If any candidate moves, their partner will get vote share more than 1/k and the deviant will still have vote share at most 1/k, so no one can deviate beneficially.

![](images/bc186655f4790a976e737b7e3465dabcd22b083422eeab9142acfbbfd1d5ce31.jpg)

Lemma 10. Any PSNE with uniform voters, complete plurality mixmizing candidates, and left-right tie-breaking must satisfy the following properties:

(a) Any point occupied by one candidate cannot be between another occupied point and a boundary.   
(b) Any point with at least three candidates must be adjacent to a boundary.   
(c) Any point with two candidates not adjacent to a boundary must have the same vote share on both sides.   
(d) In any two-point equilibrium, the points must be equidistant from $1/2$ .

Proof. (a) Otherwise, the candidate can move away from the boundary to increase their vote share and decrease an opponent's vote share.

(b) Otherwise, one of the candidates could move distance $\epsilon$ either to the right or left of the point to guarantee the maximum possible vote share (instead of having probability $< 1/3$ of being on that side). This only decreases other vote shares—except the new left- or rightmost candidate created, which only has vote share $\epsilon/2$ (note this requires at least three candidates; with only two, moving increases the vote share of the partner). When adjacent to a boundary, this doesn't work—moving $\epsilon$ towards a boundary would decrease the vote share achieved (even if it's guaranteed), which could create a plurality loss, as in Equilibria 2 and 3 from Theorem 17.

(c) If not, then one of the candidates can move $\epsilon$ towards the side with higher vote share to guarantee it. For small enough $\epsilon$ , the deviant will have higher vote share than their former partner. This also decreases the vote share of the bordering candidates the deviant moved towards. Thus, this either increases the plurality win probability of the deviant or at least decreases the expected margin against the winner.

(d) Suppose not, and call the points x and y. Assume without loss of generality that x < y and x < 1 - y. Let $z = (y - x)/2$ be the vote share inner candidates get. If $z \geq y$ , then a candidate at x can move right by $\epsilon$ to improve their winning chances, getting certain vote share z rather than a chance at it. If z < 1 - y, then a candidate at y can move right by $\epsilon$ to guarantee a win. Thus the points must be equidistant from 1/2.

![](images/4970ca952999148a2c562ae08403b42f34f87d0154d783c94f405bdfdf73ecaf.jpg)

Theorem 18. The following is a complete list of the PSNEs with uniform voters, complete plurality maximizing candidates, and left-right tie-breaking for small k:

$$
\begin{array}{l} k = 2 \colon (1 / 2, 1 / 2) \\ k = 3 \colon (1 / 2, 1 / 2, 1 / 2) \\ k = 4: (a) (1 / 2, 1 / 2, 1 / 2, 1 / 2) \\ (b) (1 / 4, 1 / 4, 3 / 4, 3 / 4) \\ (c) (x, x, 1 - x, 1 - x), \text {   for   any   } x \in (1 / 4, 1 / 2) \\ k = 5: (a) (1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2) \\ \end{array}
$$

(b) (1/4, 1/4, 1/2, 3/4, 3/4)   
(c) $(x, x, 1 - x, 1 - x, 1 - x)$ , for any $x \in (1/4, 1/2)$   
(d) $(x,x,x,1 - x,1 - x)$ , for any $x\in (1 / 4,1 / 2)$ .

Proof. We know by Theorem 17 that these are all Nash equilibria, so we only need to show no other equilibria exist.

$k = 2$ : If a candidate is at a point other than $1/2$ , then they can move to $1/2$ and do strictly better (regardless of their opponent's position), so no other equilibrium is possible.

k = 3: We know no point with one candidate can be adjacent to a boundary in equilibrium by Lemma 10. So all candidates must be at the same point. If that point is anything other than 1/2, it would not be an equilibrium, so $(1/2, 1/2, 1/2)$ must be the unique equilibrium.

k = 4: There is no way to have a single-candidate point not adjacent to a boundary, since no partition of 4 that includes a 1 has two numbers larger than 1 to flank the single-candidate point. Thus, any equilibrium either has two points with two candidates each or one point with all four candidates. The latter type of equilibrium must be at 1/2, so we only need to characterize the two-point equilibria.

We know by Lemma 10 that in two-point equilibria, the points must be equidistant from $1/2$ and so can be written as $x$ and $1 - x$ . Now, we can show that we must have $x \in [1/4, 1/2)$ . If $x < 1/4$ , then a candidate at $x$ can move right by $\epsilon$ to guarantee the winning inner vote share rather than a $1/2$ chance at it. Thus, the only two point equilibria are those claimed.

k = 5: A single-point equilibrium must be at 1/2.

A two-point equilibrium cannot be a 1–4 split since the lone candidate would be adjacent to a boundary, so any two-point equilibrium must be a 2–3 split. By Lemma 10, the points must be equidistant from 1/2, so call them x and 1 - x. We cannot have x < 1/4, or else a candidate at x would move right by $\epsilon$ to guarantee a winning vote share. Unlike for k = 4, we also cannot have x = 1/4. If we did, consider the point with 3 candidates. One of them could move to 1/2 to guarantee vote share 1/4, which would be tied for the winning share, whereas they only had a 2/3 chance of getting that vote share before. Thus the only two-point equilibria are those claimed.

We cannot have four- or five-point equilibria, since we would then be forced to place single-candidate points adjacent to the boundary. However, we can have a three-point equilibrium with a 2-1-2 split (a 3-1-1 is impossible for the same boundary reason). So we only need to show that the claimed 2-1-2 equilibrium is the only one. First, the lone candidate must be at the midpoint of the two outer points to optimize its most competitive margin. Next, we'll show the outer points must be equidistant from the boundaries. Suppose not: say the outer points are $x$ and $y$ with $x < 1 - y$ . If the inner vote share at $x$ ( $(y - x)/4$ ) is smaller than $x$ , then a candidate at $x$ has no chance of winning. But by moving to $x - \epsilon$ for some small $\epsilon$ , they can guarantee the larger vote share and reduce their expected losing margin. If the inner vote share at $x$ is larger than the outer vote share, then a candidate at $x$ loses to the lone inner candidate; but again, they can move to $x + \epsilon$ improve their expected losing margin. The only remaining option is that the inner and outer shares at $x$ are equal (so the inner share is $x$ and the middle candidate gets vote share $2x$ ). In that case, consider subcases based on $1 - y$ . If $1 - y < 2x$ , then the middle candidate always wins. Since $1 - y > x$ , a candidate at $y$ can move to $y + \epsilon$ to reduce their expected losing margin against the middle candidate. If $1 - y > 2x$ , then a candidate at $y$ can move to $y + \epsilon$ to guarantee a win rather than a $1/2$ chance. If $1 - y = 2x$ , then a candidate at $x$ can move to $y$ , giving it a chance to enter the winning lottery for vote share $2x$ (note that the candidate they leave behind at $x$ now also gets vote share $2x$ ).

Now that we know the outer points are equidistant from the boundaries, the middle candidate must then be at $1/2$ . We can now show that the only possible outer points $x$ and $1 - x$ are given

by x = 1/4. If x > 1/4, then the middle candidate cannot win; but they could move to x to join a lottery for the winning vote share. If x < 1/4, then a candidate at x cannot win. If the inner vote share at x is larger than x, a candidate at x can move to $x + \epsilon$ to reduce their expected losing margin against the middle candidate. Symmetrically, if x is larger than the inner vote share, then a candidate at x can move to $x - \epsilon$ to reduce their expected losing margin. Finally, consider the case where the inner and outer vote shares are equal (x = 1/6). A candidate at x can move into (1/6, 1/2), keeping the same vote share 1/6 while reducing the vote share of the winning candidate at 1/2, thus improving their losing margin. Therefore, the only three-point equilibrium is the one claimed with x = 1/4.

![](images/91ddc1dbed4174f8e71d785d60fe52a5ac4aebfb8ef088ef0569e0d73d966675.jpg)

# C Formal definitions of variants

To handle non-uniform voter distributions, we define $\text{Plurality}_{V}(x_{1},\ldots,x_{k})$ to be the position of the plurality winner among $x_{1},\ldots,x_{k}$ if the voter distribution is V.

Definition 5. Given an initial candidate distribution $F_{0}$ and a candidate count k, and a distribution of voters V, the replicator dynamics for candidate positioning with voter distribution V are, for all t > 0,

$$
F _ {k, t} (x) = \operatorname * {P r} (\text { Plurality } _ {V} (X _ {1, t}, \dots , X _ {k, t}) \leq x),
$$

$$
X _ {i, t} \sim F _ {k, t - 1}, \forall i = 1, \dots , k.
$$

Definition 6. Given an initial candidate distribution $F_{0}$ , a candidate count k, and m generations of memory, the replicator dynamics for candidate positioning with m generations of memory are, for all t > 0,

$$
F _ {k, t} (x) = \operatorname * {P r} (\text { Plurality } (X _ {1, t}, \dots , X _ {k, t}) \leq x),
$$

$$
X _ {i, t} \sim \left\{ \begin{array}{l l} F _ {k, t - 1} & \text {w.p.} \frac {1}{m} \\ ... \\ F _ {k, t - m} & \text {w.p.} \frac {1}{m}. \end{array} \right.
$$

Definition 7. Given an initial candidate distribution $F_{0}$ , a candidate count k, and a variance $\sigma^{2} \in [0,1]$ , the replicator dynamics for candidate positioning with $\sigma^{2}$ -perturbation noise are, for all t > 0,

$$
F _ {k, t} (x) = \operatorname * {P r} (\text { Plurality } (X _ {1, t}, \dots , X _ {k, t}) \leq x),
$$

$$
X _ {i, t} \sim \min (1, \max (0, F _ {k, t - 1} + \mathcal {N} (0, \sigma^ {2}))), \forall i = 1, \ldots , k.
$$

Definition 8. Given an initial candidate distribution $F_{0}$ , and candidate count proportions $p_{2}, p_{3}, \ldots, p_{k_{\max}}$ , the replicator dynamics for candidate positioning with variable candidate counts are, for all t > 0,

$$
F _ {t} (x) = \sum_ {k = 2} ^ {k _ {\max}} p _ {k} \cdot \operatorname * {P r} (\text { Plurality } (X _ {1, t}, \ldots , X _ {k, t}) \leq x),
$$

$$
X _ {i, t} \sim F _ {t - 1}.
$$

Let $F_{k,t}^{(i)}$ denote the distribution of the $i$ -th place candidate generation $t$ with $k$ candidates per election, where $i \leq k$ . We define $F_{k,0}^{(i)} = F_0$ for all $k$ and all $i$ , although we typically write $F_0$ since the initial distribution does not depend on $k$ . Under this notation $F_{k,t}^{(1)} = F_{k,t}$ where $F_{k,t}$ is the CDF of the winner distribution. Then,

Definition 9. Given an initial candidate distribution $F_{0}$ , a candidate count k, and $h \leq k$ , the replicator dynamics for candidate positioning with top-h copying are, for all t > 0,

$$
F _ {k, t} (x) = \operatorname * {P r} (\mathrm{Plurality} (X _ {1, t}, \ldots , X _ {k, t}) \leq x),
$$

$$
X _ {i, t} \sim \left\{ \begin{array}{l l} F _ {k, t} ^ {(1)} & \text {w.p.} \frac {1}{h} \\ ... \\ F _ {k, t} ^ {(h)} & \text {w.p.} \frac {1}{h}. \end{array} \right.
$$