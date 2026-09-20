# Plant-and-Steal: Truthful Fair Allocations via Predictions\*

Ilan Reuven Cohen $\dagger$

Alon Eden $^{\ddagger}$

Talya Eden§

Arsen Vasilyan 1

June 12, 2024

# Abstract

We study truthful mechanisms for approximating the Maximin-Share (MMS) allocation of agents with additive valuations for indivisible goods. Algorithmically, constant factor approximations exist for the problem for any number of agents. When adding incentives to the mix, a jarring result by Amanatidis, Birmpas, Christodoulou, and Markakis [EC 2017] shows that the best possible approximation for two agents and m items is $\left\lfloor\frac{m}{2}\right\rfloor$ . We adopt a learning-augmented framework to investigate what is possible when some prediction on the input is given. For two agents, we give a truthful mechanism that takes agents' ordering over items as prediction. When the prediction is accurate, we give a 2-approximation to the MMS (consistency), and when the prediction is off, we still get an $\left\lceil\frac{m}{2}\right\rceil$ -approximation to the MMS (robustness). We further show that the mechanism's performance degrades gracefully in the number of "mistakes" in the prediction; i.e., we interpolate (up to constant factors) between the two extremes: when there are no mistakes, and when there is a maximum number of mistakes. We also show an impossibility result on the obtainable consistency for mechanisms with finite robustness. For the general case of $n \geq 2$ agents, we give a 2-approximation mechanism for accurate predictions, with relaxed fallback guarantees. Finally, we give experimental results which illustrate when different components of our framework, made to insure consistency and robustness, come into play.

# 1 Introduction

Allocating items among self interested agents in a “fair” way is an age-old problem, with many applications such as splitting inheritance and allocating courses to students. As a starting point, consider the case of two agents. When the items are divisible, the famous cut-and-choose procedure achieves fairness in two senses. Firstly, no agent wants to switch their allocation with the other; i.e., there is no envy among the agents. Secondly, each agent gets a bundle of items which they value at least as much as their value for all the items divided by 2; that is, each one gets their “fair share”. When moving to the case of indivisible goods, which is relevant to scenarios such as splitting inheritance and allocating courses, things get trickier. For instance, if there’s a single item, the agent that does not receive that item does not get an envy-free allocation, nor do they get their “fair share” according to the previous definitions. Therefore, it is clear that some fairness needs to be sacrificed in this case.

The study of fair allocations with indivisible goods has been a fruitful research direction, with many meaningful notions of fairness studied (see survey by Amanatidis et al. [10]). In this paper, we focus on the notion of the Maximin Share, or MMS, introduced by Budish [18]. For two agents, this notion captures the value an agent will ensure if we implement the cut-and-choose procedure. That is, assume Alice splits the items into two bundles, and then Bob takes one of them (adversarially), and Alice gets the second one. The MMS captures exactly how much value Alice can guarantee for herself. Generalizing the notion for $n$ agents is pretty straightforward — the MMS is the minimum value Alice can guarantee for herself when she partitions the items into $n$ bundles, assuming $n - 1$ bundles are taken adversarially.

We study the case where agents have additive valuations over goods. $^{1}$ For the case of two agents, the allocation produced by the cut-and-choose procedure guarantees each of the agents their MMS value. For more than two agents, the existence of such an allocation is not longer guaranteed. Indeed, Kurokawa et al. [30] show an instance of three agents, where in every allocation, at least one of the agents does not get their MMS value. Since allocating all the agents their MMS value is not always feasible, various papers studied the existence of approximately optimal allocation. An allocation is an $\alpha$ -approximate MMS allocation for $\alpha > 1$ if every agents gets at least an $1/\alpha$ fraction of their MMS value. Feige et al. [22] introduce an instance where one cannot find an $\alpha$ -approximate allocation for $\alpha < \frac{40}{39}$ . On the other hand, [30] show there always exists $\frac{3}{2}$ -approximation. The $\frac{3}{2}$ factor was gradually improved [16, 24, 23, 8, 4, 3, 5], where the state-of-the-art algorithm achieves an approximation of 959/720 > 4/3 [3]. It is worth noting that simple variants of Round-Robin and water-filling algorithms already achieve 2-approximation. When adding incentives to the mix, matters become even more complicated.

Amanatidis et al. [7] study the case of two additive agents and $m$ items, where the algorithm (or mechanism) does not know the values of the agents. Thus, the algorithm's designer is faced with the task of devising an allocation rule such that (i) agents will maximize their allocated value by bidding truthfully, and (ii) the resulting allocation is an $\alpha$ -approximate MMS allocation for an $\alpha$ as close to 1 as possible. [7] show that the cost of dealing with self-interested agents might be dire. Namely, they show that no incentive-compatible algorithm can approximate the MMS to a factor better than $\lfloor \frac{m}{2} \rfloor$ , and this is matched by the following trivial mechanism — first agent picks their favorite item, and the second agent gets the rest. We note that although allocating each

agent all items with probability 1/2 gives each agent an expected value which is at least as large as their MMS, this solution is not deemed fair, as one agent might end up with no items at all, while their counterpart will receive all items. Thus, the fair division literature mainly considers ex-post guarantees.

For 2 < n < m, $^{2}$ a trivial truthful algorithm that lets the first n - 1 agents pick a single item in some order and gives the last agent the rest achieves an $\left\lfloor \frac{m-n+2}{2} \right\rfloor$ -approximation, and no better mechanism is known. It is conjectured that one cannot drop the dependence in m for n > 2. We are left with a stark disparity. On the one hand, assuming agents values are public information, good approximate solutions are known. On the other hand, when considering private values, it seems that only trivial approximations are possible. The goal of this paper is to bridge these two regimes using predictions.

We study the problem of truthful allocations that approximate the MMS, taking a learning-augmented point of view. In the learning-augmented framework, the algorithm designer aims to tackle some intrinsic hardness of the problem at hand, which might arise due to computational constraints, space constraints, input arriving piecemeal online, or incentive constraints, among others. To help the designer overcome these constraints, the algorithm is given some side information which is a function of the input, or a prediction, in order to improve the algorithm's performance. The hope is that if the prediction is accurate, then the performance is greatly improved over the performance without the prediction (termed consistency). On the other end, if the prediction is inaccurate then the performance of the algorithm is comparable to the performance of the best algorithm that is not given access to predictions (termed robustness). The learning-augmented framework has proven useful in bypassing impossibilities that arise due to incentive issues $[14, 1, 25, 15, 39, 32, 13]$ .

When designing a learning-augmented mechanism, one should think of realistic predictions. For instance, predicting the entire valuation profile of all agents seems to be a strong assumption. A more plausible assumption is to have some ordinal ranking over the items of the agents. Indeed, it seems unlikely that the algorithm can accurately predict Alice's value for a car, but it is plausible that the algorithm can guess that Alice values the car more than she values the table. Ideally, the algorithm's performance should remain robust if the predicted ordering is almost perfect, with only a few pairs of items whose real ordering is swapped in the prediction. Another desired property is to make the prediction as space-efficient as possible, following the intuition that smaller predictions are easier to observe. In this paper we devise learning-augmented truthful mechanisms for the problem of approximate-MMS allocations, while taking into considerations the issues mentioned above.

# 1.1 Our Results and Techniques

We start by studying the two agent case. Recall that in the two agent case, [7] show that no truthful mechanism gets a better approximation than $\lceil \frac{m}{2} \rceil$ to the MMS. We aim at getting:

1. Constant consistency: when the predictions are accurate, we want to get a constant approximation to the MMS.   
2. Near-optimal robustness: when the predictions are off, we want to get as close as possible to the optimal $\lceil \frac{m}{2} \rceil$ -approximation we can obtain by truthful mechanisms.

Plant-and-Steal Framework. In Section 3 we present a framework for devising learning-augmented mechanisms for approximating the MMS with two agents. The intuition behind the framework is as follows — in order to get better approximation guarantees, one must use the predictions in order to get a good allocation. But in case the predictions are off, only using the predictions cannot guarantee any finite approximation to the MMS. Therefore, in case the predictions are off, we must use the reports to ensure each agent gets at least one valuable item. In doing so, the mechanism should still maintains a nearly optimal allocation according to the predictions.

Our framework, which we term Plant-and-Steal is given the set of goods, an allocation procedure A, the prediction p and reports v. The framework operates as follows:

1. It first applies A on the predictions p to divide the set of goods into two bundles $A_{1}, A_{2}$ . The procedure A should be an allocation procedure with good MMS guarantees. We use different allocation procedures depending on the type of prediction given and on the consistency-robustness tradeoffs we are aiming for.   
2. Planting phase: For each agent i, it picks i's favorite item in set $A_{i}$ according to prediction, and “plants” this item in the bundle $A_{j}$ of the other agent $j \neq i$ . Let $T_{1}, T_{2}$ denote the sets that result in this planting phase.   
3. Stealing phase: To obtain the final allocation, each agent i now “steals” back their favorite item from set $T_{j}$ of agent $j \neq i$ according to reports. Notice this is the first and only place where we use agents’ reports.

This procedure is trivially truthful because the only step where we use agents' reports is the one where they pick exactly one item to steal back from $T_{j}$ , and this $T_{j}$ only depends on predictions, and not reports (Lemma 3.1). To obtain robustness, we notice that each agent gets one of their two favorite items according to their true valuations (Lemma 3.2). This implies a robustness of $m - 1$ . We show that if the allocations produced by $\mathcal{A}$ are more balanced, we get improved robustness guarantees (Lemma 3.4).

Ordering Predictions. In Section 4, we study learning-augmented mechanisms when the predictions given are the preference orders over items of the agents (rather than the values). In the case where the predictions are preference orders, we instantiate the Plant-and-Steal framework with a Round-Robin-based allocation procedure. [8] show that preceding the Round-Robin procedure with an initial allocation of large items (of worth greater than $\mu_i / 2$ ) gives a 2-approximation to the MMS. We observe that in the case of two agents, one can run the Round-Robin procedure as is, without the initial allocation phase, and still obtain the 2-approximation. The gain in using the standard procedure is that the allocation is as balanced as possible. To show consistency, we notice that by the properties of the Round-Robin procedure, each agent $i$ values her favorite item more then any item in the other agent's set $A_j$ , except for the other agent's favorite item. Since $T_j$ is obtained by adding $i$ 's favorite item to $T_j$ and removing $j$ 's favorite item from it, in case the predictions are accurate, $i$ takes back the item the mechanism planted in $T_j$ , and vice-versa. Thus, we end up with the original allocation ( $A_1, A_2$ ), obtaining a consistency of 2. Since each agent gets at least $\lfloor \frac{m}{2} \rfloor$ items, including one of their top two items, we can show that we obtain a robustness guarantee of $\lceil \frac{m}{2} \rceil$ . This almost completely matches the $\lfloor \frac{m}{2} \rfloor$ lower bound from [7].

Amanatidis et al. [6] study truthful mechanisms when the agents' rankings are global. For two agents, they were able to show that slightly modifying the Round-Robin procedure, to let

the second agent choose two items each time, obtains an improved approximation ratio of $\frac{3}{2}$ to the MMS. When using the modified Round-Robin as the allocation procedure A in the Plant-and-Steal framework, we get an improved consistency of $\frac{3}{2}$ , but since the final allocation is less balanced, our robustness guarantee becomes $\left\lfloor\frac{2m}{3}\right\rfloor$ .

We then study the performance of the Plant-and-Steal framework when using the Round-Robin procedure, when the prediction given is not fully accurate, but accurate to some degree. To quantify the prediction's accuracy, we adopt the Kendall tau distance measure (or the bubble-sort distance). The Kendall tau distance counts the number of pairs of elements swapped in the two orderings. For our purpose, we consider the Kendall tau distance between the predicted preference order and the order induced by the true valuations. In order to simplify the analysis, we apply the zero-one principle. By the zero-one principle, it is enough to show that our mechanism achieves the desired approximation guarantee in instances where the values for the items are either 1's or 0's. We first show that for such instances, the initial allocation of the Round-Robin procedure, $(A_{1}, A_{2})$ , achieves an additive approximation to the MMS (this is also true for the mechanisms with global rankings from [6]). This does not guarantee, however, any multiplicative approximation. Thus, we must leverage the fact that the agents get to “steal back” an item according to their true valuations. We therefore are able to show that combining the Plant-and-Steal framework with a Round-Robin allocation procedure obtains $O(\sqrt{d})$ -approximation to the MMS when the Kendall tau distance between the predictions and the valuations is d. Since d goes from 0 to $\binom{m}{2} = \Theta(m^{2})$ , we recover the constant consistency when there are no errors, and the $O(m)$ robustness when the number of errors is maximal.

General Predictions. In Section 5, we study the two-agent case where the mechanism is given access to predictions which are not necessarily the preference order of the agents. We first show that for any prediction given to the learning-augmented mechanism, no mechanism can simultaneously be $\alpha$ -consistent while maintaining finite robustness for $\alpha < 6/5$ . For the proof, we leverage the characterization of two-agent truthful mechanisms by [7].

We then study small-space predictions. The Round-Robin-based mechanisms described above require an $\Omega(m)$ -bit prediction (to describe an arbitrary allocation of items). We first notice that we can implement a water-filling type allocation procedure using $O(\log m)$ -bit predictions. This already achieves a constant consistency along with $O(m)$ robustness. We then devise a more refined allocation procedure, which requires $O(\log m/\epsilon)$ -bit predictions, and achieves $2+\epsilon$ consistency along with $\left\lceil\frac{m}{2}\right\rceil$ robustness.

Note that the work of [20] showed how to learn an $(1 + \epsilon / 2)$ -approximate MMS allocation in the context of the model of [31] in which the valuations $v_{i}(j)$ are sampled i.i.d. from a distribution $D_{i,j}$ under a small item assumption. We remark that combining this learned allocation with our Plant-and-Steal framework immediately gives a truthful, $(2 + \epsilon)$ -consistent and $m$ -robust mechanism.

General number of agents n. Finally, in Section 6, we devise a learning-augmented truthful mechanism for $n \geq 2$ additive agents. We obtain a 2-consistent mechanism, while relaxing the robustness guarantees of the mechanism. We take a similar approach to the works of [18, 27, 28, 2, 5], who compete against a relaxed benchmark of the MMS value for $\hat{n} > n$ agents, and try to minimize $\hat{n}$ . We obtain a $(m - \lceil 3n/2 \rceil - 1)$ -approximation to the MMS for $\hat{n} = \lceil \frac{3n}{2} \rceil$ agents when the predictions are off. Our mechanism uses the modified Round-Robin procedure from [8] to determine the initial allocation using the predictions. It then applies a recursive plant-and-steal procedure

where in each stage of the recursion, agents are partitioned into two sets. For each set of agents, the mechanism “plants” their current favorite item according to prediction in the combined bundle of items of the other set, and “steals” back an item according to her reports. In order to ensure consistency, the internal order in which each set of agents steal should be the same as their order in the corresponding Round-Robin round. In order to get our robustness guarantee, we carefully choose the order at each Round-Robin round. We then show each agent gets at least their $\left\lceil\frac{3n}{2}\right\rceil$ th most preferred item according to their true valuation.

Experiments. Finally, In Section 7, we demonstrate how several components in our design come into play when experimenting with synthetic data. We run different variants of mechanism on two player instances, and show that when predictions are accurate, then only using predictions is nearly optimal, if predictions are noisy, then the stealing component ensures robustness, and our Plant-and-Steal framework achieves best-of-both-worlds guarantees.

We summarize the known bounds for learning-augmented truthful mechanisms for MMS approximation in Table 1.

<table><tr><td>Setting</td><td>Consistency</td><td>Robustness</td><td>Reference</td></tr><tr><td rowspan="4">Ordering predictions, n = 2</td><td>2</td><td> $\lceil m/2 \rceil$ </td><td>Section 4</td></tr><tr><td>3/2</td><td> $\lfloor 2m/3 \rfloor$ </td><td>Section 4</td></tr><tr><td>Any</td><td> $\geq \lfloor m/2 \rfloor$ </td><td>[7]</td></tr><tr><td> $\geq 5/4$ </td><td>Any</td><td>[6]</td></tr><tr><td>Arbitrary predictions, n = 2</td><td> $\geq 6/5$ </td><td>Bounded</td><td>Section 5.1</td></tr><tr><td>log n + 1 space</td><td>4</td><td>m - 1</td><td>Section 5.2</td></tr><tr><td>O(log(n/ $\epsilon$ ) space</td><td>2 +  $\epsilon$ </td><td> $\lceil m/2 \rceil$ </td><td>Section 5.3</td></tr><tr><td>n &gt; 2</td><td>2</td><td>m -  $\lceil 3n/2 \rceil - 1$  for  $\hat{n} = \lceil 3n/2 \rceil$ </td><td>Section 6</td></tr></table>

Table 1: Known bounds for truthful learning-augmented MMS mechanisms.

# 1.2 Related Works

The notion of the maximin share allocation was introduced by Budish $[18]$ as an ordinal notion, and extended to the notion we adopt by Bouveret and Lemaître $[17]$ . Using machine learning advice in algorithm design was used in theory $[21, 37]$ and practice $[29]$ . The learning-augmented framework of studying consistency-robustness tradeoffs was introduced by Lykouris and Vassilvitskii $[33]$ . $[34, 38]$ studied the performance of algorithms using imprecise predictions.

Fair division with incentives. The two closest papers to ours are Amanatidis et al. [6, 7]. In [6], they initiate the study of truthful mechanisms for approximating the MMS value for agents with additive valuations. They show that no truthful mechanism can get an approximation better than $1/2$ for the MMS in the case of 2 agents and 4 items. They give the best known approximation guarantee for $n$ agents and $m$ items of $\lfloor \frac{m - n + 2}{2} \rfloor$ . Finally they consider the public ranking model, where the ranking over items is public information. Using this, they are able to obtain a $\frac{n + 1}{2}$ -approximation algorithm. One can view this as an algorithm that is given a prediction over the

input, but does not provide robustness guarantees. [7] Fully characterize truthful mechanism for 2 agents with additive valuations. They use this characterization to provide a strong lower bound of $\lfloor \frac{m}{2} \rfloor$ for any truthful mechanism.

[12] design truthful mechanisms for dichotomous submodular valuations that maximize welfare, along with desirable fairness properties such as EFX and NSW. For additive binary valuations, they also maximize the MMS in a truthful manner. [26] bypass the impossibilities imposed by [7, 36] for truthful fair allocations with indivisible and divisible goods by considering Bayesian Incentive Compatible mechanisms with symmetric priors. They are able to obtain EF-1 allocations for indivisible goods and proportional allocations for indivisible goods.

Finally, $[9]$ study the Nash equilibrium for simple mechanisms for agents with additive valuations. They show that for every number of agents, the Pure Nash equilibrium of the Round-Robin procedure produces an EF-1 allocation. For two agents, they show that the Pure Nash equilibrium of Plaut and Roughgarden $[35]$ cut-and-choose procedure produces an EFX and MMS allocation.

1-out-of-k. As stated above, the MMS value of an agent is defined by the highest value an agent can guarantee for themselves when partitioning the items into n different bundles, where n is the number of agents, and then getting the lowest valued bundle. Thus, an agent gets a value larger than the worst one-out-of-n bundles that define the MMS.

Noticing that finding an allocation that satisfies the MMS value of each agent is a demanding task (which was shown to be infeasible in some cases by Kurokawa et al. [30]), Budish [18] relaxed the notion and defined the 1-out-of- $n + 1$ MMS to be the worst bundle out of the bundles that define the MMS when partitioning the items using an additional bundle. [18] showed it is possible to achieve this benchmark when adding a small number of access goods. There has been an effort to find the smallest $k$ for which an allocation that guarantees a 1-out-of- $k$ MMS for each agent exists. [2] were able to show the existence for $k = 2n - 2$ , [27, 28] achieved $k = \lceil \frac{3n}{2} \rceil$ , and recently, [5] showed the smallest up-to-date $k = \lceil \frac{4n}{3} \rceil$ . In our $n$ -agent mechanism, our robustness guarantee approximates this relaxed benchmark for $k = \lceil \frac{3n}{2} \rceil$ .

Learning Augmented Mechanisms. Agrawal et al. $[1]$ and Xu and Lu $[39]$ first explored the learning augmented framework in a mechanism design setting, where $[1]$ studied the facility location problem while $[39]$ applied the framework to several settings such as revenue-maximization, path auctions, scheduling and two-facility games. $[14]$ give nearly optimal consistency-robustness tradeoffs to the strategyproof scheduling with unrelated machines. $[25]$ use predictions to design mechanisms with improved Price of Anarchy bounds. $[32, 19]$ study revenue maximization auctions with predictions, and $[13]$ devise bicriteria mechanisms.

# 2 Preliminaries

In the setting we study, there is a set $N$ of $n$ agents and a set $M$ of $m$ indivisible items. Each agent has a private additive valuation over the items, unknown to the mechanism designer, where the value of agent $i$ for item $j$ is $v_{ij}$ (also denoted as $v_i(j)$ ). For a bundle $S \subseteq M$ of items, $v_i(S) = \sum_{j \in S} v_{ij}$ .

The fairness notion we focus on is the following.

Definition 2.1 (Maximin Share). The Maximin Share (MMS) of agent i with valuation $v_{i}$ and n agents is

$$
\mu_ {i} ^ {n} = \max _ {S _ {1} \bigcup \dots \bigcup S _ {n} = M} \min _ {j \in [ n ]} v _ {i} (S _ {j});
$$

that is, if i were to partition the items into n bundles, and then n - 1 of those bundles are taken adversarially, what is the value i can guarantee for themselves. When clear from the context, we omit n and use $\mu_{i}$ to denote the MMS of i with n agents.

We are interested in mechanisms that produce approximately optimal allocations, as defined next.

Definition 2.2 ((γ, k)-approximate MMS Allocation). An allocation $X = (X_1, \ldots, X_n)$ is (γ, k)-approximate MMS allocation for γ > 1 and a natural number k if for every agent i,

$$
v _ {i} (X _ {i}) \geq \mu_ {i} ^ {k} / \gamma .
$$

When $k = n$ , we say the allocation is a $\gamma$ -approximate MMS allocation.

We study mechanism that get some prediction on the input.

Definition 2.3 (Learning Augmented Mechanism). A learning-augmented mechanism takes agents' reports $\mathbf{r} = (r_1, \ldots, r_n)$ and predictions $\mathbf{p}$ in some prediction space $\mathcal{P}$ , and outputs a partition of the items

$$
X (\mathbf {r}, \mathbf {p}) = (X _ {1} (\mathbf {r}, \mathbf {p}), X _ {2} (\mathbf {r}, \mathbf {p}), \ldots , X _ {n} (\mathbf {r}, \mathbf {p})), \quad X _ {1} (\mathbf {r}, \mathbf {p}) \bigcup \cdot \bigcup X _ {2} (\mathbf {r}, \mathbf {p}) \bigcup \cdot \ldots \bigcup \cdot X _ {n} (\mathbf {r}, \mathbf {p}) = M,
$$

where agent $i$ gets $X_{i}(\mathbf{r},\mathbf{p})$ .

For learning-augmented mechanisms, truthfulness should hold for any possible prediction p.

Definition 2.4. A learning-augmented mechanism is truthful if for every agent i and every possible report of other agents $r_{-i}$ and every possible prediction p,

$$
v _ {i} (X _ {i} (v _ {i}, \mathbf {r} _ {- i}, \mathbf {p})) \geq v _ {i} (X _ {i} (r _ {i}, \mathbf {r} _ {- i}, \mathbf {p}))
$$

for every $r_i$ .

We next define the consistency and robustness measures according to which we measure the performance of our mechanisms.

Definition 2.5 ( $\alpha$ -consistency). Consider a prediction function $f_{P}$ which takes a valuation profile and outputs a prediction in prediction space P. A learning-augmented mechanism is $\alpha$ -consistent for $\alpha > 1$ and prediction function $f_{P}$ if for every valuation profile v and every prediction $\mathbf{p} = f_{\mathcal{P}}(\mathbf{v})$ , $X(\mathbf{v}, \mathbf{p})$ is an $\alpha$ -approximate MMS allocation.

Definition 2.6 ((β, k)-robust). A learning-augmented mechanism is (β, k)-robust for β > 1 and natural number k if for every valuation profile v and every prediction p, X(v, p) is an (β, k)-approximate MMS allocation. If k = n, we say the mechanism is β-robust.

For ease of presentation, for valuation $v_{i}$ , report $r_{i}$ and prediction $p_{i}$ , we use $v_{i}^{\ell}, r_{i}^{\ell}, p_{i}^{\ell}$ to denote both the $\ell$ th highest good according to the valuation/report/prediction and its value. Note that, we may use $v_{i}^{\ell}$ for $\ell > m$ , in this case, $v_{i}^{\ell} = 0$ . For $\ell = 1$ , i.e., the highest good we use $v_{i}^{*}, r_{i}^{*}, p_{i}^{*}$ .

# 2.1 Ordering Predictions and Kendall tau Distance

Most of our mechanisms use predictions which take the form of an ordering over agents items. That is, $f_{\mathcal{P}}(\mathbf{v})$ outputs a vector of orderings $\mathbf{p} = (p_1, \ldots, p_n)$ , where $p_i^\ell$ is the $\ell$ th highest valued item of i in M according to p. Accordingly, for agent i, let $v_i^\ell$ be the $\ell$ th highest valued item according to v. For two items $j \neq j'$ , We use $j \succ_{p_i} j'$ to denote that j is higher ranked than $j'$ according to p.

When studying imprecise predictions, we want to quantify the degree to which the prediction is inaccurate. For this, we use the following measure. For an agent i, we define our noise level with respect to the Kendall tau distance (also known as bubble-sort distance) between v and p.

Definition 2.7 (Kendall tau distance). The Kendall tau distance counts the number of pairwise disagreements between two orders. For $i$ 's valuation $v_i$ and predicted preference order $p_i$ , we define

$$
K _ {d} (v _ {i}, p _ {i}) = | \{j \succ_ {p _ {i}} j ^ {\prime}: v _ {i} (j) <   v _ {i} (j ^ {\prime}) \}.
$$

That is, the number of pairs of items where the prediction got their relative ordering wrong. We also denote $K_{d}(\mathbf{v},\mathbf{p})=\max\{K_{d}(v_{1},p_{1}),K_{d}(v_{2},p_{2})\}$ .

We note that the Kendall tau distance between $v_{i}$ and $p_{i}$ , $K_{d}(v_{i}, p_{i})$ , can go from 0 to $\binom{m}{2}$ .

# 3 Plant-and-Steal Framework

In this section, we present the framework which is used to devise learning-augmented mechanisms for two agents. The ideas presented here also inspire the highly complex learning-augmented mechanism for n > 2 agents. As described in Section 1.1, the Plant-and-Steal framework takes an allocation procedure A, as well agents' predictions and reports. It first uses A on the predictions to derive an initial allocation $(A_{1}, A_{2})$ . Then, it "plants" agent i's favorite item of set $A_{i}$ according predictions in set $A_{j}$ , $j \neq i$ . Let $(T_{1}, T_{2})$ be the sets resulting from the planting phase. Finally, each agent i "steals" back their favorite item in $T_{j}$ , $j \neq i$ , according to reports.

For $S \subseteq M$ , and agent i, let $v_{i}^{*}(S)$ ( $p_{i}^{*}(S), r_{i}^{*}(S)$ ) be the max valued item in S according to $v_{i}$ ( $p_{i}, r_{i}$ ). for $g \in M$ and $S \subseteq M$ , denote $S + g := S \cup \{g\}$ and $S - g = S \setminus \{g\}$ . The Plant-and-Steal framework is presented in Mechanism 1.

We now show that for any allocation function A and predictions p given to the framework, the resulting mechanism is truthful.

Lemma 3.1 (Truthfulness Lemma). For any allocation procedure A, Plant-and-Steal mechanism using A is truthful.

Proof. We show that agent 1 is better off reporting their true valuation, a symmetric argument holds for agent 2. First, notice that sets $T_{1}$ and $T_{2}$ are determined using predictions, ignoring the reports. Next, notice that the item $\tilde{j}_{2}$ is chosen only using agent 2's report. Therefore, the only way agent 1 can affect their allocation is by choosing which item in $T_{2}$ is allocated to them. agent 1 gets their favorite item in $T_{2}$ according to their report. Therefore, it is clear that the agent maximize their utility by reporting their true value.

Since the framework is truthful, from now on, we assume that r = v. Next, we show that the Plant-and-Steal mechanism ensures that for each agent, an item is allocated with a value that is at least as good as their second-best option according to their value.

MECHANISM 1: Two agent Plant-and-Steal Framework   
Input : Allocation Procedure A, set of items M, predictions p and reports r
Output: Allocations $X_{1} \cup X_{2} = M$ /* Find an initial allocation by applying A on the predictions */ $(A_{1}, A_{2}) := \mathcal{A}(M, N, \mathbf{p})$ /* Plant favorite items according to predictions */ $\hat{j}_{1} \leftarrow p_{1}^{*}(A_{1})$ $\hat{j}_{2} \leftarrow p_{2}^{*}(A_{2})$ $T_{1} \leftarrow A_{1} + \hat{j}_{2} - \hat{j}_{1}$ $T_{2} \leftarrow A_{2} + \hat{j}_{1} - \hat{j}_{2}$ /* Steal according to report */ $\tilde{j}_{1} \leftarrow r_{1}^{*}(T_{2})$ $\tilde{j}_{2} \leftarrow r_{2}^{*}(T_{1})$ $X_{1} \leftarrow T_{1} + \tilde{j}_{1} - \tilde{j}_{2}$ $X_{2} \leftarrow T_{2} + \tilde{j}_{2} - \tilde{j}_{1}$

Lemma 3.2. Consider the allocation $(X_{1}, X_{2})$ returned by Plant-and-Steal with some allocation procedure A. For any agent i, then $v_{i}^{1} \in X_{i}$ or $v_{i}^{2} \in X_{i}$ .

Proof. Consider some agent i. We claim for every partition of the items into two non-empty sets, $T_{1}, T_{2}, i$ is always guaranteed to have one of their two favorite items according to their true valuation $v_{i}$ in $X_{i}$ . This is because either (1) i has one of their two favorite items in $T_{\ell}, \ell \neq i$ , and i gets their favorite item from $T_{\ell}$ ; or (2) i's two favorite items are in $T_{i}$ , and in this case, i gets all items from $T_{i}$ but one, so i is guaranteed one of them. □

We next claim that if $i$ gets one of their two favorite items and any $k - 1$ additional items, $i$ 's value is an $m - k$ -approximation to $\mu_i$ .

Lemma 3.3. For any agent $i$ , let $S \subseteq M$ be a subset of the items of size $|S| = k$ and $v_i^1 \in S$ or $v_i^2 \in S$ then

$$
v _ {i} (S) \geq \mu_ {i} / (m - k).
$$

Proof. Let $g \in S \cap \{v_i^1, v_i^2\}$ , by the definition of $S$ such $g$ exists. Let $S' = S \setminus \{g\}$ , by the definition of $S$ , we have $|S'| = k - 1$ and $v_i(S) \geq v_i^2 + v_i(S')$ . Consider a partition

$$
(S _ {1}, S _ {2}) \in \arg \max _ {(T _ {1}, T _ {2}): T _ {1} \cup T _ {2} = M} \min _ {j \in \{1, 2 \}} v _ {i} (T _ {j}).
$$

By definition, $\mu_{i} = \min_{j\in \{1,2\}}v_{i}(S_{j})$ . We have,

$$
\begin{array}{l} \frac {\mu_ {i}}{v _ {i} (S)} \leq \frac {\mu_ {i}}{v _ {i} ^ {2} + v _ {i} (S ^ {\prime})} \\ = \frac {\min _ {j \in \{1 , 2 \}} v _ {i} (S _ {j})}{v _ {i} ^ {2} + v _ {i} (S ^ {\prime})} \\ \leq \frac {\min _ {j \in \{1 , 2 \}} v _ {i} (S _ {j}) - v _ {i} (S ^ {\prime})}{v _ {i} ^ {2}} \\ \end{array}
$$

$$
\begin{array}{l} \leq \frac {\min _ {j \in \{1 , 2 \}} v _ {i} (S _ {j} \setminus S ^ {\prime})}{v _ {i} ^ {2}} \\ \leq \frac {\min _ {j \in \{1 , 2 \}} \{| S _ {j} \setminus S ^ {\prime} | \cdot \max \{v _ {i} (\ell) : \ell \in S _ {j} \setminus S ^ {\prime} \} \}}{v _ {i} ^ {2}}. \\ \leq \frac {(m - k) \cdot \min _ {j \in \{1 , 2 \}} \max \left\{v _ {i} (\ell) : \ell \in S _ {j} \right\}}{v _ {i} ^ {2}} \\ \leq m - k. \\ \end{array}
$$

where the before last inequality is since if $S_{j} \subseteq S'$ for some $j$ , then $v_{i}^{2} + v_{i}(S') \geq \mu_{i}$ ; therefore $S_{1} \setminus S'$ and $S_{2} \setminus S'$ are two disjoint non empty subsets and $|S_{1} \setminus S'| + |S_{2} \setminus S'| = m - k + 1$ , hence the maximum number of elements in one of these subsets is $m - k$ .

We immediately get the following.

Lemma 3.4 (Robustness Lemma). Let A be an allocation rule guaranteeing $\min\{|A_{1}|, |A_{2}|\} \geq k$ , then when Plant-and-Steal uses A, the resulting mechanism is $(m - k)$ -robust.

Proof. By Lemma 3.2, we are guaranteed that each agent gets one of their two favorite items according to their report. Combining with the condition on A and Lemma 3.3, the proof is finished.

![](images/ba6e7cedf50d41c8636b3d221d240f1aabc6febb1db658128d89327102ee451b.jpg)

# 4 Ordering Predictions

In this section, we consider the case of two agents, where the predictions (and in fact, also the reports) given to the mechanism are preference orders of agents over items. Our mechanisms makes use of the Plant-and-Steal framework instantiated by Round-Robin based allocation procedures. In Section 4.1 we present our two round-robin allocation procedures, and give their approximation guarantees when the input is accurate. In Section 4.2 we prove the robustness and consistency guarantees. In Section 4.3 we quantify the accuracy of the predictions using the Kendall tau distance, and obtain fine-grained approximation results, where the approximation smoothly degrades in the accuracy.

Amanatidis et al. [6] studied mechanisms where the preference orders of the agents over items are public (while valuations are private). They showed that no truthful mechanism can achieve a better approximation than 5/4 in this setting. This implies that when the predictions are preference orders, no learning-augmented mechanism can obtain consistency better than 5/4, no matter if the robustness is bounded or not.

Proposition 4.1 (Corollary of Amanatidis et al. [6]). No mechanism that is given preference orders as predictions can obtain consistency $5/4 - \epsilon$ for any $\epsilon > 0$ .

# 4.1 Round-Robin Allocation Procedures

The two allocation procedures we use to instantiate the Plant-and-Steal framework take as input preference orders of agents over items:

\- Balanced-Round-Robin: the agents take turns, and at each turn, an agent takes their highest ranked remaining item. This results in a balanced allocation.

\- 1-2-Round-Robin: the agents take turns, where we compensate the second agent, who might not get their favorite item, to take two items each turn.

Consider the allocation procedure depicted in Algorithm 2.

ALGORITHM 2: Balanced-Round-Robin   
Input : Preference orders of agents over items $\mathbf{v} = (v_{1}, v_{2})$ .
Output: An allocation $A_{1} \cup A_{2} = M$ . $A_{i} \leftarrow \emptyset$ for every agent $i \in \{1, 2\}$ for $r = 1, \ldots, \lceil |M|/2 \rceil$ do $A_{1} \leftarrow A_{1} + v_{1}^{*}(M \setminus A_{1} \setminus A_{2})$ $A_{2} \leftarrow A_{2} + v_{2}^{*}(M \setminus A_{1} \setminus A_{2})$

Notice that to implement the allocation procedure of Balanced-Round-Robin, it only needs to receive preference orders over items. Let $A_{i} = (a_{i}^{1},\dots ,a_{i}^{|A_{i}|})$ be agent $i$ 's allocation by the algorithm, where $a_{i}^{k}$ is the $k$ 'th choice of agent $i$ . We observe the following.

Observation 4.1. The output $(A_{1}, A_{2})$ of the Balanced-Round-Robin procedure, satisfies:

1. $|A_1| = \lceil \frac{m}{2} \rceil, |A_2| = \lfloor \frac{m}{2} \rfloor$ .   
2. For each agent $i$ and round $k$ , $a_{i}^{k} \in \{v_{i}^{\ell}\}_{\ell \in [2k]}$ ; that is, in round $k$ an agent gets one of their top $2k$ items.

Amanatidis et al. [8] show that first allocating large items to agents, and then using a Round-Robin to allocate the remaining items to the remaining agents, gives a 2-approximation to the MMS. We observe that for two agents, Round-Robin as is, without the initial step, achieves this approximation guarantee. The proof of the following Lemma is deferred to Appendix B.

Lemma 4.1. Let $(A_{1}, A_{2})$ be the allocation of Balanced-Round-Robin. For every agent i, $v_{i}(A_{i}) \geq \mu_{i}/2$ .

One can show that the agent that picks first actually gets a value at least as large as their MMS, while for the second agent this analysis is indeed tight. $^{3}$ In order to compensate agent 2, 1-2-Round-Robin lets this agent pick two items each round. See Algorithm 3 for details.

ALGORITHM 3: 1-2-Round-Robin   
Input : Preference orders of agents over items $\mathbf{v} = (v_{1}, v_{2})$ .
Output: An allocation $A_{1} \cup A_{2} = M$ . $A_{i} \leftarrow \emptyset$ , for every agent $i \in N$ for $r = 1, \ldots, \lceil |M|/3 \rceil$ : do $A_{1} \leftarrow A_{1} + v_{1}^{*}(M \setminus A_{1} \setminus A_{2})$ $A_{2} \leftarrow A_{2} + v_{2}^{*}(M \setminus A_{1} \setminus A_{2})$ $A_{2} \leftarrow A_{2} + v_{2}^{*}(M \setminus A_{1} \setminus A_{2})$

Let $a_{i}^{k}$ be agent $i$ 's $k$ th choice in 1-2-Round-Robin, we observe the following.

Observation 4.2. The output $(A_{1}, A_{2})$ of the 1-2-Round-Robin procedure, satisfies:

1. $|A_{1}| = \left\lceil \frac{m}{3} \right\rceil$ and $|A_{2}| = \left\lfloor \frac{2m}{3} \right\rfloor$ .   
2. $a_1^k\in \{v_1^\ell \}_{\ell \in [3k - 2]}, a_2^{2k - 1}\in \{v_2^\ell \}_{\ell \in [3k - 1]}$ and $a_2^{2k}\in \{v_2^\ell \}_{\ell \in [3k]}.$

Amanatidis et al. [6] show that 1-2-Round-Robin guarantees each agent 2/3 of their MMS.

Lemma 4.2 (Amanatidis et al. [6]). Let $(A_{1}, A_{2})$ be the allocation of 1-2-Round-Robin. For every agent i, $v_{i}(A_{i}) \geq 2\mu_{i}/3$ .

For completeness, We provide the proof of the approximation in Appendix B.

We next use the two allocation procedures to instantiate the Plant-and-Steal framework.

# 4.2 Round-Robin-Based Mechanisms

We analyze the two mechanisms:

- B-RR-Plant-and-Steal: The mechanism which results from instantiating Plant-and-Steal with Balanced-Round-Robin as $\mathcal{A}$ .   
- 1-2-RR-Plant-and-Steal: The mechanism which results from instantiating Plant-and-Steal with 1-2-Round-Robin as $\mathcal{A}$ .

We first show that if the predictions correspond to the preference orders of the real valuations, then both B-RR-Plant-and-Steal and 1-2-RR-Plant-and-Steal output the same allocation as Balanced-Round-Robin and 1-2-Round-Robin.

Lemma 4.3. When predictions correspond to actual values, B-RR-Plant-and-Steal (1-2-RR-Plant-and-Steal) outputs the same allocation as Balanced-Round-Robin (1-2-Round-Robin).

Proof. We prove the claim for B-RR-Plant-and-Steal. The proof for 1-2-RR-Plant-and-Steal is identical.

Let $j_{1}$ be the first item assigned in Balanced-Round-Robin to agent 1. By definition, $j_{1}$ is agent 1's favorite item in M according to $p_{1}$ . Clearly, in Plant-and-Steal, $j_{1}$ is also agent 1's favorite item in $A_{1} \subseteq M$ according to $p_{1}$ . Hence, $\hat{j}_{1} = j_{1}$ . By the definition of Plant-and-Steal, $j_{1} \in T_{2}$ . Since we assume the prediction corresponds to agent 1's actual value, $j_{1}$ is also agent 1's favorite item in $T_{2} \subseteq M$ , which implies $\tilde{j}_{1} = j_{1}$ .

Similarly Let $j_2$ be the first item assigned in Balanced-Round-Robin to agent 2. By definition, $j_2$ is agent 2's favorite item in $M \setminus \{j_1\}$ according to $p_2$ . Since $j_1 \in A_1$ , $A_2 \subseteq M \setminus \{j_1\}$ . Therefore, $j_2$ is also agent 2's favorite item in $A_2$ according to $p_2$ . Hence, $\hat{j}_2 = j_2$ . Since we established that $\hat{j}_1 = j_1$ , we have that $T_1 \subseteq M \setminus \{j_1\}$ and $j_2 \in T_1$ . Since we assume the prediction corresponds to agent 1's actual value, $j_2$ is also agent 1's favorite item in $T_1$ , implying $\tilde{j}_2 = j_2$ . We get that $X_1 = A_1$ and $X_2 = A_2$ as required.

We are now ready to prove the performance guarantees of our mechanisms.

Theorem 4.1. Mechanism B-RR-Plant-and-Steal is truthful, 2-consistent and $\left\lceil\frac{m}{2}\right\rceil$ -robust.

Proof. By Lemma 3.1, the mechanism is truthful. By Observation 4.1, each agent receives at least $\lfloor m/2\rfloor$ items; combining with Lemma 3.4, we get that the mechanism is $\lceil\frac{m}{2}\rceil$ -robust. Finally, if predictions correspond to valuations, by Lemma 4.1 and Lemma 4.3, the allocation is a 2-approximation to the MMS. Thus, the mechanism is 2-consistent. □

We note that by Amanatidis et al. [7], our robustness guarantee matches the optimal obtainable approximation by any truthful mechanism (up to the rounding).

We next show that in 1-2-RR-Plant-and-Steal we are able to achieve a better consistency, while slightly weakening the robustness guarantee. Due to similarity to the proof of Theorem 4.1, we defer the proof of the following Theorem to Appendix B.

Theorem 4.2. Mechanism 1-2-RR-Plant-and-Steal is truthful, 3/2-consistent and $\left\lfloor\frac{2m}{3}\right\rfloor$ -robust.

# 4.3 Noisy Predictions

We now analyze Mechanism B-RR-Plant-and-Steal's performance under varying levels of noise. Consider the case where the Kendall tau distance between $\mathbf{v}$ and $\mathbf{p}$ is at most $d$ . Our goal is to relate the value agent $i$ gets from the allocation, $v_{i}(X_{i})$ to their maximin share $\mu_{i}$ . To simplify the analysis, we compare what $i$ gets to the worst possible set of items $i$ might get when running the Round-Robin procedure using the agents' real preferences $R_{i} = \{v_{i}^{2j}\}_{j\in \{1,\dots,\lfloor m / 2\rfloor \}}$ . In Eq. (10) of Lemma 4.1, we show that

$$
v _ {i} (R _ {i}) \geq \mu_ {i} / 2. \tag {1}
$$

We further simplify the analysis by applying the zero-one principle $^{4}$ . The zero-one principle basically let's us reduce to instances where the values are either 0's or 1's. For threshold $\tau \geq 0$ , let

$$
h _ {\tau} (q) = \left\{ \begin{array}{l l} 1 & \quad q \geq \tau \\ 0 & \quad \text {otherwise} \end{array} \right..
$$

Accordingly, let $v_{i}^{\tau}(S) = \sum_{j\in S}h_{\tau}(v_{i}(j))$ .

By the zero-one principle, for two sets $S, T \subseteq M$ , in order to show that $v_{i}(S)$ approximates $v_{i}(T)$ , it is enough to show that $v_{i}^{\tau}(S)$ approximates $v_{i}^{\tau}(T)$ for every threshold $\tau \geq 0$ .

Lemma 4.4. For c > 1 and for any two sets $S, T \subseteq M$ , if for every threshold $\tau \geq 0$ , $v_{i}^{\tau}(S) \geq v_{i}^{\tau}(T)/c$ , then $v_{i}(S) \geq v_{i}(T)/c$ .

Proof. Let $S = \{s_1, \ldots, s_k\} (|S| = k)$ and $T = \{t_1, \ldots, t_\ell\} (|T| = \ell)$ . We have the following.

$$
\begin{array}{l} v _ {i} (S) = \sum_ {j = 1} ^ {k} v _ {i} (s _ {j}) = \sum_ {j = 1} ^ {k} \int_ {0} ^ {\infty} h _ {\tau} (v _ {i} (s _ {j})) d \tau = \int_ {0} ^ {\infty} \sum_ {j = 1} ^ {k} h _ {\tau} (v _ {i} (s _ {j})) d \tau = \int_ {0} ^ {\infty} v _ {i} ^ {\tau} (S) d \tau \\ \geq \int_ {0} ^ {\infty} v _ {i} ^ {\tau} (T) / c d \tau = \frac {1}{c} \int_ {0} ^ {\infty} \sum_ {j = 1} ^ {\ell} h _ {\tau} (t _ {j}) d \tau = \frac {1}{c} \sum_ {j = 1} ^ {\ell} \int_ {0} ^ {\infty} h _ {\tau} (t _ {j}) d \tau = \frac {1}{c} \sum_ {j = 1} ^ {\ell} t _ {j} = v _ {i} (T) / c, \\ \end{array}
$$

where we use the identity $\int_0^\infty h_\tau (q)d\tau = q$

![](images/6c9301beff72bb8002bfbca71d65720ed04374bb02dc02bf59a29f247b46d087.jpg)

Thus, we will show that when the Kendall tau distance is d, for every threshold $\tau \geq 0$ , $v_{i}^{\tau}(X_{i}) \geq v_{i}^{\tau}(R_{i})/c$ for some $c = O(\sqrt{d})$ . Recall that $A_{i}$ is the set of items assigned to i after running the Round-Robin procedure on the predictions p. We first show that for Kendall tau distance d, the additive approximation $v_{i}^{\tau}(A_{i})$ gives to $v_{i}^{\tau}(R_{i})$ is $\sqrt{d}$ .

Lemma 4.5. If the Kendall tau distance between p and v is at most d, then for any threshold $\tau \geq 0$ , we have that $v_{i}^{\tau}(A_{i}) \geq v_{i}^{\tau}(R_{i}) - \sqrt{d}$ .

Proof. Let $\left\lfloor\frac{m}{2}\right\rfloor\leq m_{i}\leq\left\lceil\frac{m}{2}\right\rceil$ be the number of items agent i gets by Mechanism B-RR-Plant-and-Steal. Let $A_{i}=\{a_{i}^{1},a_{i}^{2},\ldots,a_{i}^{m_{i}}\}$ be the items assigned to agent i in the Round-Robin according to the predicted orderings p, where $a_{i}^{\ell}$ is the item allocated to i in the $\ell$ th round of Round-Robin. First, by Observation 4.1, we have:

$$
a _ {i} ^ {\ell} \in \{p _ {i} ^ {j} \} _ {j \in \{1, \dots , 2 \ell \}}. \tag {2}
$$

For a fixed $\tau\geq0$ , let $L_{\tau}=v_{i}^{\tau}(R_{i})$ be the number of values larger than threshold $\tau$ in $R_{i}$ . We show that if the Kendall tau distance is at most d, then it must be the case that

$$
v _ {i} ^ {\tau} (A _ {i}) \geq L _ {\tau} - \sqrt {d}. \tag {3}
$$

Note that $h_{\tau}(v_{i}^{k}) = 1$ for $k \leq 2 \cdot L_{\tau}$ since $R_{i}$ gets every second item by the sorted values of agent i. This implies that if $h_{\tau}(v_{1}(a_{i}^{\ell})) = 0$ then $a_{i}^{\ell} = v_{i}^{k}$ for $k > 2 \cdot L_{\tau}$ . Moreover, if $\ell \leq L_{\tau}$ then $a_{i}^{\ell} = p_{i}^{k}$ for $k \leq 2 \cdot L_{\tau}$ by Eq. (2). Thus, if $\sum_{k=1}^{L_{\tau}} h_{\tau}(v_{1}(a_{i}^{k})) < L_{\tau} - \sqrt{d}$ there are strictly more than $\lceil \sqrt{d} \rceil$ items whose rank according to the true valuation is at most $2 \cdot L_{\tau}$ , and their rank according to the prediction is at least $2 \cdot L_{\tau} + 1$ . We show that this implies that the Kendall tau distance is larger than d, yielding a contradiction. Formally, let

$$
G _ {1} = \{v _ {1} ^ {k} \} _ {k \in \{1, \dots , 2 \cdot L _ {\tau} \}} \setminus \{p _ {1} ^ {k} \} _ {k \in \{1, \dots , 2 \cdot L _ {\tau} \}}
$$

be the set of items whose rank is at most $2 \cdot L_{\tau}$ according to the real values but not according to the predictions, and let

$$
G _ {2} = \{v _ {1} ^ {k} \} _ {k \in \{2 \cdot L _ {\tau} + 1, \dots , m \}} \setminus \{p _ {1} ^ {k} \} _ {k \in \{2 \cdot L _ {\tau} + 1, \dots , m \}}
$$

be the set of items whose rank is strictly larger than $2 \cdot L_{\tau}$ according to the real values but not according to the predictions. By the above, $|G_{1}| = |G_{2}| > \lceil \sqrt{d} \rceil$ , and for each pair $j \in G_{1}, j' \in G_{2}$ ,

1. $j$ rank according to $v_{i}$ is at most $2 \cdot L_{\tau}$ and $j'$ rank according to $v_{i}$ is at least $2 \cdot L_{\tau} + 1$ ;   
2. $j'$ rank according to $p_{i}$ is at most $2 \cdot L_{\tau}$ and j rank according to $p_{i}$ is at least $2 \cdot L_{\tau} + 1$ .

That is, j and $j'$ are ordered oppositely in the ordering according to $p_{i}$ and $v_{i}$ . Since there are $|G_{1}|\cdot|G_{2}|>d$ such pairs, we get that the Kendall tau distance is strictly greater than d, a contradiction.

We note that although $v_{i}^{\tau}(A_{i})$ gives an additive approximation to $v_{i}^{\tau}(R_{i})$ , it can still be the case that the Kendall tau distance is constant, yet $v_{i}(A_{i})$ does not give any multiplicative approximation to $\mu_{i}$ .⁵ Therefore, we must use the fact that agent i gets to “steal” an item according to their true valuation in the Plant-and-Steal procedure in order to get our approximation guarantee. We now prove our approximation guarantees.

Theorem 4.3. Consider a prediction p and valuations v such that $K_{d}(\mathbf{v},\mathbf{p}) = d$ , then Mechanism B-RR-Plant-and-Steal gives a $(2\sqrt{d} + 6)$ -approximation to the MMS.

Proof. We use the zero-one principle to show that Lemma 4.4 holds for sets $X_{i}$ and $R_{i}$ with $c = \sqrt{d} + 3$ . The proof then follows by Eq. (1).

Notice that $|A_{i} \setminus X_{i}| \leq 2$ , because in the “stealing” phase, agent i might not take the “planted” item from $A_{i}$ back, and the other agent might take one item from $A_{i}$ . $^{6}$ Moreover, by Lemma 3.2, either $v_{i}^{1}$ or $v_{i}^{2}$ are in $X_{i}$ . Therefore, for every threshold $\tau \geq 0$ ,

$$
\begin{array}{l} v _ {i} ^ {\tau} (X _ {i}) \geq \max \{h _ {\tau} (v _ {i} ^ {2}), v _ {i} ^ {\tau} (A _ {i}) - 2 \} \\ \geq \max \left\{h _ {\tau} \left(v _ {i} ^ {2}\right), v _ {i} ^ {\tau} \left(R _ {i}\right) - \sqrt {d} - 2 \right\}, \tag {4} \\ \end{array}
$$

where the inequality follows Lemma 4.5.

If $h_{\tau}(v_{i}^{2}) = 0$ , then $v_{i}^{\tau}(R_{i}) \leq |R_{i}| \cdot h_{\tau}(v_{i}^{2}) = 0$ , and Lemma 4.4 holds with c = 0. Therefore, the interesting case is when $h_{\tau}(v_{i}^{2}) = 1$ . Consider the ratio $\frac{v_{i}^{\tau}(R_{i})}{v_{i}^{\tau}(X_{i})}$ which we want to bound. Since $v_{i}^{\tau}(X_{i}) \geq h_{\tau}(v_{i}^{2}) = 1$ , $v_{i}^{\tau}(R_{i}) \in [1, \sqrt{d} + 3]$ implies that

$$
\frac {v _ {i} ^ {\tau} (R _ {i})}{v _ {i} ^ {\tau} (X _ {i})} \leq v _ {i} ^ {\tau} (R _ {i}) \leq \sqrt {d} + 3.
$$

On the other hand, by Eq. (4), setting $v_{i}^{\tau}(R_{i}) = \sqrt{d} + 3 + \delta$ for $\delta > 0$ implies that $v_{i}^{\tau}(X_{i}) \geq v_{i}^{\tau}(R_{i}) - \sqrt{d} - 2 \geq 1 + \delta$ , which yields

$$
\frac {v _ {i} ^ {\tau} (R _ {i})}{v _ {i} ^ {\tau} (X _ {i})} \leq \frac {\sqrt {d} + 3 + \delta}{1 + \delta} \leq \sqrt {d} + 3.
$$

We get that Lemma 4.4 holds for $X_{i}$ and $R_{i}$ with $c = \sqrt{d} +3$ . Thus,

$$
v _ {i} (X _ {i}) \geq v _ {i} (R _ {i}) / (\sqrt {d} + 3) \geq \mu_ {i} / (2 \sqrt {d} + 6),
$$

where the last inequality follows Eq. (1).

We note that a similar analysis for Mechanism 1-2-RR-Plant-and-Steal will show a similar dependence in $\sqrt{d}$ (up to constant factors).

# 5 Non-ordering Predictions

In this Section, we consider the case where predictions are not necessarily preference orders over items. In Section 5.1, we show that for any prediction the mechanism might get, consistency is bounded away from 1. Sections 5.2, 5.3, we study succinct predictions, i.e. predictions about general structure of the preferences of two agents. Section 5.2 presents a 4-consistent and $\lceil m / 2 \rceil$ -robust mechanism, whose consistency relies on the correctness of only a log $m$ -bit prediction about the preferences of the two agents. In Section 5.3, we show that a $2 + \epsilon$ -consistent and $\lceil m / 2 \rceil$ -robust mechanism exists, whose consistency relies on correctly predicting only $O(\log m / \epsilon)$ bit about the preferences of the two agents.

# 5.1 No Mechanism with < 6/5 Consistency and Bounded Robustness

In Appendix C.1, we show that no mechanism can simultaneously achieve a consistency guarantee strictly lower than 6/5 and any bounded robustness guarantee no matter which prediction is given. We use the elegant characterization of [7] for 2-agent mechanisms and show that in any truthful mechanism with finite approximation ratio, if the valuations are identical, then each agent gets at least one of the two largest items. Thus in the instance where the prediction is $p_1 = p_2 = (1/2, 1/2, 1/3, 1/3, 1/3)$ , each agent gets one item of value 1/2. This implies that there is an agent with an allocation of value $1/2 + 1/3 = 5/6$ (and an agent with value 7/6), while $\mu_1 = \mu_2 = 1$ .

Theorem 5.1. For any $\epsilon > 0$ , there is no truthful a mechanism with consistency $6/5 - \epsilon$ and bounded robustness.

# 5.2 4-Consistent, $(m-1)$ -Robust Mechanism Using a $\log m+1$ -Space Prediction

Let us formally define a mechanism that uses a space- $s$ prediction

Definition 5.1. A learning-augmented mechanism is a space-s mechanism if the prediction space P can be represented by the elements of $\{0,1\}^{s}$ .

We first give a simple mechanism that only requires $\log m + 1$ bits of information about the valuations $v_{1}$ and $v_{2}$ . It will only need to know an index $j_{0}$ in [m] together with a bit b. The mechanism will utilize the Plant-and-Steal framework in conjunction with the well-known water-filling allocation procedure:

MECHANISM 4: Water-Filling   
Input : Preference orders of agents over items $\mathbf{v} = (v_{1}, v_{2})$ on a set of items M = [m]
Output: Allocations $A_{1} \cup A_{2} = M$ $j \leftarrow 1$ for $j = 1, \ldots, m$ : do
    if $\frac{v_{1}([j])}{v_{1}([m])} \geq \frac{1}{2}$ then Output $(A_{1}, A_{2}) \leftarrow ([j], [m] \setminus [j])$ and terminate
    if $\frac{v_{2}([j])}{v_{2}([m])} \geq \frac{1}{2}$ then Output $(A_{1}, A_{2}) \leftarrow ([m] \setminus [j], [j])$ and terminate

We see that, in order to predict the behaviour of the mechanism above, one only needs to predict accurately the index $j_{0}$ on which the mechanism terminates, as well as a bit $b \in \{1, 2\}$ that encodes whether the algorithm terminates due to the condition $\frac{v_{1}([j])}{v_{1}([m])} \geq \frac{1}{2}$ being satisfied or due to the condition $\frac{v_{2}([j])}{v_{2}([m])} \geq \frac{1}{2}$ being satisfied. This can be encoded using $\log m + 1$ bits.

We also see that the The Plant-and-Steal framework when used with the Water-Filling allocation procedure gives a truthful 4-consistent and a $m - 1$ -robust $^7$ allocation mechanism. The truthfulness and robustness follow immediately from Lemmas 3.1 and 3.4 respectively.

The 4-consistency holds for the following reason. It is a well-known fact (see i.e. [10]) that the partition $(A_{1}, A_{2})$ given by the water-filling algorithm satisfies $v_{1}(A_{1}) \geq \mu_{1} / 2$ and $v_{2}(A_{2}) \geq \mu_{2} / 2$ . By inspecting the Plant-and-Steal framework (Algorithm 1), we see that both agent 1 and agent

2 will either (i) retain their most preferred item in $A_1$ and $A_2$ respectively or (ii) Lose this item, but obtain an item that they prefer even more. Overall, this implies that in the worst case the difference $v_1(A_1) - v_1(X_1)$ will equal to the value of the second-most favorite item of Agent 1 in $A_1$ . This implies that $v_1(X_1) \geq \frac{1}{2} v_1(A_1) \geq \frac{\mu_1}{4}$ . Analogously, we see that $v_1(X_2) \geq \frac{1}{2} v_1(A_2) \geq \frac{\mu_2}{4}$ .

# 5.3 $2 + \epsilon$ -Consistent, $\lceil \frac{m}{2} \rceil$ -Robust Mechanism Using a $O(\log m / \epsilon)$ -Space Prediction

We now show that a better consistency of $2 + \epsilon$ can be achieved at the cost predicting $O(\log m / \epsilon)$ bits of information about the valuations $v_{1}$ and $v_{2}$ . We will also obtain a better robustness of $\lceil \frac{m}{2} \rceil$ . To do this, we will use the Plant-and-Steal framework in conjunction with the Cut-and-Balance allocation procedure. We first explain how the mechanisms above can be implemented by only

ALGORITHM 5: Cut-and-Balance   
Output: Allocations $A_1 \cup A_2 = M$ Consider a partition $S_1 \cup S_2 = M$ satisfying $|S_1| \geq |S_2|$ and $\min_{j \in \{1,2\}} v_1(S_j) \geq (1 - \epsilon) \max_{T_1 \cup T_2 = M} \min_{j \in \{1,2\}} v_1(T_j) = (1 - \epsilon) \mu_1$

Let $S' \subset S_1$ be a set of $\lfloor m/2 \rfloor - |S_2|$ items satisfying

• $v_{1}(S') \leq v_{1}(S_{1})/2$   
- if $|S_2| > 1$ additionally satisfying $v_1(S') \leq v_1(S_1 \setminus \{\hat{j}, \hat{j}'\}) / 2$ , for some $\hat{j} \in \arg \max_{j \in S_1} v_1(j)$ and $\hat{j}' \in \arg \max_{\ell \in S_1 \setminus \hat{j}} v_1(\hat{j})$

Set $\tilde{S}_1 \leftarrow S_1 \setminus S'$ and $\tilde{S}_2 \leftarrow S_2 \cup S'$ Let $i_2 \leftarrow \arg \max_{i \in \{1,2\}} p_2(\tilde{S}_i)$ and let $i_1$ be the index of the other bundle

Set $A_1 \leftarrow S_{i_1}$ and $A_2 \leftarrow S_{i_2}$ , and output the allocation $(A_1, A_2)$

obtaining $O(\log m / \epsilon)$ bits of information about the valuations $v_{1}$ and $v_{2}$ . This follows from the following proposition, the proof of which is given in Appendix C.2.

Proposition 5.1. Suppose $M = [m]$ . There is a partition $M = L_{1} \bigcup L_{2} \bigcup S$ and indices $\alpha_{1}, \beta_{1}, \alpha_{2}$ and $\beta_{2}$ with $|L_{1}| + |L_{2}| \leq O\left(\frac{1}{\epsilon}\right)$ , such that the partition $M = S_{1} \bigcup S_{2}$ defined as $S_{1} = L_{1} \bigcup (S \bigcap [\alpha_{1}, \beta_{1}])$ and $S_{2} = L_{2} \bigcup (S \bigcap [\alpha_{2}, \beta_{2}])$ satisfies $|S_{1}| \geq |S_{2}|$ and $\min(v_{1}(S_{1}), v_{1}(S_{2})) \geq (1 - \epsilon / 4)\mu_{1}$ .

Additionally, there exist integers $\alpha_{3},\beta_{3},\alpha_{4}$ and $\beta_{4}$ such that the set $S^{\prime} = S\bigcap ([\alpha_{3},\beta_{3}]\bigcup [\alpha_{4},\beta_{4}])$ satisfies $|S^{\prime}| = \lfloor m / 2\rfloor -|S_{2}|$ , $S^{\prime}\subset S_{1}$ , $v_{1}(S^{\prime})\leq v_{1}(S_{1}) / 2$ and if $|S_2| > 1$ then $S^{\prime}$ also satisfies $v_{1}(S^{\prime})\leq v_{1}(S_{1}\setminus \{\hat{j},\hat{j}^{\prime}\}) / 2$ , where $\hat{j}\in \arg \max_{j\in S_1}v_1(j)$ and $\hat{j}^{\prime}\in \arg \max_{j\in S_1\setminus \hat{j}}v_1(j)$ .

The main ideas for proving Proposition 5.1 are: (i) using the sets $L_{1}$ and $L_{2}$ to handle elements $x$ whose value $v(x)$ is large, and separate the remaining items into the set $S$ (ii) Showing that the remaining items can be separated into well-behaved subsets of the form $S \bigcap [\alpha_i, \beta_i]$ .

The proposition above implies that the sets $S_{1}, S_{2}$ and $S'$ can be represented exactly via sets $L_{1}$ and $L_{2}$ , together with the indices $\{\alpha_{1}, \cdots, \alpha_{4}, \beta_{1}, \cdots, \beta_{4}\}$ . We will also need to know the index $i_{2} \in \{1, 2\}$ . Since the sets $L_{1}$ and $L_{2}$ have a size of $O(1/\epsilon)$ , all this information amounts to $O(\log m/\epsilon)$ bits as claimed.

The following proposition implies the truthfulness, the robustness and the consistency of the mechanism that combines the Cut-and-Balance allocation procedure with the Plant-and-Steal framework.

Theorem 5.2. The Plant-and-Steal framework, when used with Cut-and-Balance allocation procedure, gives a truthful, $2 + \epsilon$ -consistent and a $\lceil m/2 \rceil$ -robust allocation mechanism.

Proof. Truthfulness follows from Lemma 3.1. Since the sets $A_{1}$ and $A_{2}$ both have size at most $[m / 2]$ , the robustness follows via Lemma 3.4.

The proof of $(2 + \epsilon)$ -consistency is deferred to Appendix C.3. The main challenge for showing the bound on consistency is the fact that both the Cut-and-Balance allocation procedure and the Plant-and-Steal framework can reduce the consistency by a factor of 2. Naively, one would expect the overall consistency to be close to 4, given that each stage can lose a factor of 2 in consistency. However, our insight is that for the instances, on which the Cut-and-Balance allocation procedure loses a factor of 2 in consistency, the Plant-and-Steal framework will have consistency close to 1, and vice versa. This allows us to prove a tighter bound of $2 + \epsilon$ on the consistency of our overall algorithm.

# 6 Mechanisms for $n$ agents

In this section we provide a learning-augmented mechanism for n > 2 agents, Learning-Augmented-MMS-for-n-Agents. The mechanism we devise ensures that if the predictions are accurate, then each agent gets an allocation with value at least $\mu_{i}^{n}/2$ (2 consistency). On the other hand, we show that for any prediction, every agent gets at least $\mu_{i}^{\lceil3n/2\rceil}/\alpha$ for $\alpha = m - \lceil3n/2\rceil - 1$ (robustness).

Theorem 6.1. The Learning-Augmented-MMS-for-n-Agents Mechanism (Mechanism 6) is truthful, 2-consistent and $\mu_i^{\lceil 3n / 2\rceil} / \alpha$ -robust for $\alpha = m - \lceil 3n / 2\rceil - 1$ .

# 6.1 An Overview

The Mechanism. The mechanism works in three phases. In the first phase, it uses the predictions in order to obtain a partial allocation to agents with high predicted items (which are then removed from the set of active agents, so that we can now that for all agents, all predicted values are small). Then, in the second stage, the mechanism uses the predictions in order to obtain a tentative allocation, by running a Round-Robin procedure, where items are tentatively allocated to agents according to their predictions. In the third and final phase, the tentative allocation is used to implement a recursive plant and steal procedure, where the “planting” is done from the tentative allocations according to predictions, but the “stealing” is done according to the agents’ reports and results in a final allocation.

MECHANISM 6: Learning-Augmented-MMS-for-n-Agents   
Input : Set of agents N, set of items M, reports $r_{N}$ , predictions $p_{N}$ Output: A partition of the items $\bigcup_{i\in N} X_{i}$ Invoke Algorithm 7, $X \leftarrow \text{Allocate-Large}(N, M, \mathbf{r}_{N}, \mathbf{p}_{N})$ Invoke Algorithm 9, $A \leftarrow \text{Tentative-Allocation-Round-Robin}(N, M, \mathbf{p}_{N})$ Invoke Algorithm 10, $X \leftarrow \text{Split-Plant-Steal-Recurse}(N, A, \text{first-level-flag} = \text{True}, X, \mathbf{r}_{N}, \mathbf{p}_{N})$

Consistency. In the case the predictions are accurate, the initial allocation phase will take care of agents with high valued items (of value larger than $\mu_{i}^{n}/2$ ). Then, in the second phase, the

![](images/77acd23907ab5d88a396903f310bd3eecff967a74de0cd9f96f332caf3d6dbe6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph N0
        A["(a₁¹, a₁², ...)"]
        B["(a₃¹, a₃², ...)"]
        C["..."]
        D["(aₙ₋₁¹, aₙ₋₁², ...)"]
    end
    subgraph N1
        E["(a₂¹, a₂², ...)"]
        F["(a₄¹, a₄², ...)"]
        G["..."]
        H["(aₙ¹, aₙ², ...)"]
    end
    A --> E
    B --> F
    C --> H
    D --> E
    E --> F
    style N0 fill:#fff,stroke:#000
    style N1 fill:#fff,stroke:#000
```
</details>

After tentative allocation.   
Each i plants $p_{i}^{*}(A_{i})$ in the tentative set of the corresponding agent

![](images/24d83e9244831f9921c3ab2276352118f65758c8b70637c7669cdaadffb328a0.jpg)

<details>
<summary>text_image</summary>

(a₁₂, a₁², ...)
(a₄¹, a₃², ...)
⋮
(aₙ¹, aₙ², ...)
N₀

(a₁¹, a₂², ...)
(a₂¹, a₄², ...)
⋮
(aₙ⁻¹, aₙ², ...)
N₁
</details>

After planting,   
before stealing

![](images/8b2f9d4cca854ab386ddf5dd5cd731b49b1602b422fe9eaf4b7e52341442dc07.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph_N0["Stealing"]
        A["(a₁², a₁², ...)"]
        B["(a₄¹, a₃², ...)"]
        C["..."]
        D["(aₙ¹, aₙ², ...)"]
    end
    subgraph_N1["Stealing"]
        E["(a₁¹, a₂¹, ...)"]
        F["(a₃¹, a₄¹, ...)"]
        G["..."]
        H["(aₙ¹, aₙ², ...)"]
    end
    A --> E
    B --> F
    C --> H
    D --> E
    E --> F
    style N0 fill:#fff,stroke:#333
    style N1 fill:#fff,stroke:#333
```
</details>

Each agent $i\in N_0$ steals $r_i^* (A_{N_1})$   
Each $i\in N_1$ steals $r_i^* (A_{N_0})$

![](images/e382e75cf7a874ad9fd0c5e769ffbb974a73096515917f8773c7ac7d412ceac7.jpg)

<details>
<summary>text_image</summary>

(a₁², ... ) {a₁¹}
(a₃², ... ) {a₃¹}
⋮
(aₙ₋₁², ... ) {aₙ₋₁¹}
N₀
(a₂², ... ) {a₂¹}
(a₄², ... ) {a₄¹}
⋮
(aₙ², ... ) {aₙ¹}
N₁
</details>

Partial allocation (purple)   
after first round of stealing   
Figure 1: Illustration of a single round of the recursive planting and stealing phase (Algorithm 10), for the case where predictions are accurate (so that each agent steals back their planted item). Note that the stealing is done from the union of items of agents in the opposite set (and not just from the corresponding agent).

tentative allocation will be exactly identical to a Round-Robin allocation (made according to true valuations). Finally, in the third phase, since agents steal in the same order they were allocated the items in the Round-Robin allocation, and since the predictions are accurate, the agents “steal” back the same item the mechanism plants. Since a Round-Robin allocation achieves $\mu_{i}^{n}/2$ when there are no agents with high valued items [8], correctness follows.

Robustness. In the case the predictions are inaccurate, we show that every agent still gets at least $\mu_{i}^{\lceil3n/2\rceil}/\alpha$ . Here we rely on the plant-and-steal phase to ensure that each agent gets at least their $\lceil3n/2\rceil$ highest-valued item according to their true valuation. This property provides our robustness guarantee. We notice that reversing the order between the first and subsequent rounds of the Round-Robin procedure (and thus, the stealing phases) gives an enhanced robustness guarantee.

Prediction. In the description of the mechanism, we assume the mechanism is given a prediction of agents valuations. We note that in order to implement the mechanism it is enough to be given access to agents' preference order over items, and an additional information indicating which items are worth more than $\mu_{i}^{n}/2$ for each agent i.

Below, we first give a detailed description of the mechanism, and then we conclude by proving Theorem 6.1.

# 6.2 Implementation Details

As discussed, in order to utilize the Round-Robin mechanism, we first allocate a single item to each agent with a high predicted value.

ALGORITHM 7: Allocate-Large   
Input : Set of agents N, set of items M, reports $r_{N}$ , predictions $p_{N}$ Output: A partial allocation $\bigcup_{i\in B} X_{i}$ , updated sets of agents and items N, M, respectively

foreach $i \in N$ do

Compute $\mu_{i}^{n}$ based on $p_{i}$ while exists $i \in N$ such that $p_{i}^{*}(M) \geq \mu_{i}^{n}/2$ do $X_{i} \leftarrow \{r_{i}^{1}(M)\}$ $M \leftarrow M \setminus X_{i}$ $N \leftarrow N \setminus \{i\}$

Before describing the tentative allocation mechanism, we first give a procedure, Allocate-Best, which performs a single round of Round-Robin according to a specific order, and preferences (either predictions or reports), denote o.

PROCEDURE 8: Allocate-Best (One-Round-RR)   
Input : Ordered set of agents N, set of items M, valuation $v_{N}$ Output: $|N|$ singletons $X_{i} \in M$ foreach $i \in N$ do $X_{i} \leftarrow v_{i}^{*}(M)$ $M \leftarrow M \setminus X_{i}$

The tentative allocation mechanism repeatedly invokes Allocate-Best according to given predictions, until all items are tentatively allocated. As previously mentioned, the first round of the tentative allocation is performed according to the given order, and in all subsequent rounds, the order is reversed (recall that reversing the order enhances the robustness guarantees).

The final phase in the mechanism is a recursive plant and steal algorithm. The input to this algorithm is an ordered set of agents N, along with their predictions, reports, and a tentative allocation for each agent. At each recursive invocation, the algorithm splits the set of agents into two (almost) equal-size ordered sets $N_{0}$ and $N_{1}$ . Then the mechanism “plants” for the $i^{th}$ agent in each set $N_{b}$ their highest (according to predictions) valued item in their tentative allocation in the tentative set of the $i^{th}$ agent in $N_{\neg b}$ . Then we perform one round of Round-Robin, where the items available to the agents of set $N_{b}$ are those tentatively allocated to the agents of $N_{\neg b}$ (after the planting phase), and the allocations are determined according to agents reports. See Figure 1 for an illustration of a single round of plant and steal. The algorithm then recurses on each of the sets

ALGORITHM 9: Tentative-Allocation-Round-Robin   
Input : Ordered set of agents $N = (i_{1}, \ldots, i_{|N|})$ , set of items M, predictions $p_{N}$ Output: A tentative allocation $\bigcup_{i \in N} A_{i} = M$ $A \leftarrow \text{Allocate-Best}(N, M, \mathbf{p}_{N})$ $M \leftarrow M \setminus \cup_{i \in N} A_{i}$ /* Reverse the order for the allocation of the rest of the items */ $N^{r} = (i_{|N|}, \ldots, i_{1})$ for $k = 2, \ldots, \lceil m/n \rceil$ do $\tilde{A} = \text{Allocate-Best}(N^{r}, M, \mathbf{p}_{N^{r}})$ $A_{i} \leftarrow A_{i} \cup \tilde{A}_{i}$ for $i \in N$ $M \leftarrow M \setminus \cup_{i \in N} \tilde{A}_{i}$

$N_{0}$ and $N_{1}$ , until all sets are of size 1. At this point, the single agent in the set is further allocated its remaining tentatively allocated items, and the process terminates.

Given the above implementation details, it remains to prove Theorem 6.1 regarding truthfulness, consistency and robustness of the mechanism. The proof is given below.

# 6.3 Proof of Theorem 6.1

In this section we prove Theorem 6.1, which we now recall.

Theorem 6.1. The Learning-Augmented-MMS-for-n-Agents Mechanism (Mechanism 6) is truthful, 2-consistent and $\mu_i^{\lceil 3n / 2\rceil} / \alpha$ -robust for $\alpha = m - \lceil 3n / 2\rceil - 1$ .

First, we give a simple observation regarding Algorithm 7.

Observation 6.1. The followings hold for Algorithm Allocate-Large.

1. If the reports equal the true valuations, and agent $i$ is allocated an item $j$ , then $v_{i}(j) \geq v_{i}^{n} / 2$ .   
2. After the algorithm completes its run, there are no remaining agents in $N$ with large predicted values for the remaining items in $M$ .

We continue to prove each of the properties specified in Theorem 6.1 separately, starting with truthfulness.

Lemma 6.1 (Truthfulness). Mechanism Learning-Augmented-MMS-for-n-agents (Mechanism 6) is truthful.

Proof. Algorithm Tentative-Allocation-Round-Robin (Algorithm 9) only depends on agents predictions and not their reports. Hence, we only need to consider the use of the reports in Algorithms 9 and 10.

For every agent i, either they are allocated a single item in Algorithms 9, or i participates in the recursive plant ant steal, and this is determined according to the predictions, so in particular $r_{i}$ has no affect on this. Thus, we can consider the two independent events separately. In the first case, where i is allocated a single item, it is the item that maximizes their report over remaining items at that point, so that i has no incentive to lie.

In the second case, i participates in the plant and steal phase. Observe that in this case, whenever i chooses an item from some set $A'$ , it will have no future interaction with this set. That

ALGORITHM 10: Split-Plant-Steal-Recurse   
Input : Ordered set of agents $N = (i_1, \ldots, i_{|N|})$ , tentative allocations A, partial allocations $X_N$ , first-level-flag indicating if this is the first level of the recursion, reports $r_N$ , predictions $p_N$ /* Halting condition - Allocate all remaining items */

if $N = \{i\}$ then set $X_i = X_i \cup A_i$ and halt

/* Split the agents into two almost-equal parts */

par = N mod 2 $N_0 \leftarrow (i_1, i_3, \ldots, i_{N-1+par})$ $N_1 \leftarrow (i_2, i_4, \ldots, i_{N-par})$ /* Plant according to predictions */

for $i = 1, \ldots, \lfloor |N|/2 \rfloor$ do

Let $i^0, i^1$ denote the $i^{th}$ agent in $N_0, N_1$ respectively. $j_0^* = p_{i^0}^*(A_{i_0})$ $j_1^* = p_{i^1}^*(A_{i_1})$ $A_{i^0} = A_{i^0} + j_1^* - j_0^*$ $A_{i^1} = A_{i^1} + j_0^* - j_1^*$ /* Plant $i_n$ 's favorite item in a tentative set */

if par = 1 then $i^0 = i_n, i^1 = i_2$ $j_0^* = p_{i^0}^*(A_{i^0})$ $A_{i^1} = A_{i^1} + j_0^*, A_{i^0} = A_{i^0} - j_0^*$ /* Steal from the opposite set according to reports */

foreach $b \in \{0, 1\}$ do $\hat{X} = \text{Allocate-Best}(N_b, A_{N_{-b}}, r)$ foreach $i \in N$ do $X_i \leftarrow X_i \cup \hat{X}_i$ /* Reverse the order after the first level of recursion */

if first-level-flag then $N_0 \leftarrow (i_{N-1+par}, \ldots, i_3, i_1)$ $N_1 \leftarrow (i_{N-par}, \ldots, i_4, i_2)$ /* Recursively invoke Split-Plant-Steal-Recurse on each set */

foreach $b \in \{0, 1\}$ do

Split-Plant-Steal-Recurse( $N_b, A_{N_b}, X_{N_b}, first-level-flag = False$ )

is, fix a recursive call and assume without loss of generality that $i \in N_{0}$ . Then after the planting step, i is allocated the item in $A_{N_{1}}$ that maximizes their reports. Then, in following recursive steps, i only continues to interact with items in $A_{N_{0}}$ , so i's choice does not affect the identity of the items from which i will be able to choose from in future rounds. Hence, i's only incentive is to maximize the value of its allocated value in each round, implying truthfulness. ☐

Due to the above lemma, from now on we assume agents report truthfully, i.e., that for every agent i, $r_{i} = v_{i}$ . We turn to show the mechanism is consistent, we rely on the following theorem.

Theorem 6.2 (Lemma 2 in [10] (based on Theorem 3.5 in [8])). If for every $i \in N$ and $j \in M$ , $v_i(j) \leq \frac{1}{2}\mu_i^n$ , then the Round-Robin algorithm returns an allocation that is MMS/2.

Furthermore, their analysis holds when changing the order of allocation between the different rounds of the Round-Robin.

We are now ready to prove the mechanism is consistent.

Lemma 6.2 (Consistency). If the set of predictions is accurate, then for every i, $v_{i}(X_{i}) \geq \mu_{i}^{n}/2$ .

Proof. First consider agents that were allocated an item in Algorithm Allocate-Large (Algorithm 7). If the predictions are accurate, then each such agent i is allocated an item j such that $v_{i}(j) \geq \mu_{i}^{n}/2$ and so the statement holds. Moreover, at the end of this step, there are no remaining agents with large predicted values, hence, no agents with large values remain.

If the set of predictions is accurate, then the tentative allocation determined according to agents' predictions in Algorithm Tentative-Allocation-Round-Robin (Algorithm 9) is identical to a Round-Robin mechanism according to valuations, with reversing the order between the first and all subsequent rounds. Furthermore, by the above, there are no agents with large values when the Round-Robin is invoked. Therefore, by Theorem 6.2, it holds that for every $i$ , $v_{i}(A_{i}) \geq \mu_{i}^{n}/2$ . We shall prove that for every agent $i$ , its final allocation equals its tentative allocation, $X_{i} = A_{i}$ , concluding the proof.

We prove that in depth k of the recursion, every agent i is allocated the $k^{th}$ item in $A_{i}$ . We prove the claim by induction on the depth k of the recursion, and the $\ell^{th}$ agent in that round that is allocated some value.

We first prove for k = 1, $\ell = 1$ . In the plant phase, $\ell^{0}(=1)$ plants $j = p_{\ell^{0}}^{*}(A_{\ell^{0}})$ in $A_{\ell^{1}}$ . Then, in the stealing phase, during the invocation of Algorithm 8, agent $\ell^{0}$ is the first to choose an item from $A_{N_{1}}$ , which in particular contains j. Hence, the first item in $A_{1}$ is allocated into $X_{1}$ . We now assume the claim holds for k = 1 and $\ell - 1$ and prove it for $\ell$ . Assume without loss of generality that $\ell$ is odd so that $i_{\ell} \in N_{0}$ .

In step $\ell$ of the planting phase, the mechanism plants $\ell^0$ 's (the proof for $\ell^1$ is identical) first (according to value $p_{\ell^0}$ ) item in $A_{\ell^0}$ . Then, during the tentative allocation phase, agent $\ell^0$ is the $\ell^{\text{th}}$ to choose among the items in $A_{N_1}$ minus the items that were allocated to the $\ell - 1$ agents that were before her in the tentative Round-Robin. By the induction hypothesis, every agent preceding her chose the item the mechanism planted for them previously in that round. Therefore, the item $j$ that the mechanism planted for agent $\ell^0$ is still available. Moreover, let $M^{\ell-1}$ denote the set of items after $\ell - 1$ rounds of the tentative Round-Robin in Algorithm 9. Further let $A_{N_1}^{\ell-1}$ denote the set of items after $\ell - 1$ rounds of the Allocate-Best algorithm invoked in the stealing phase with the set $N_0$ , i.e., $A_{N_1}^{\ell-1} = A_{N_1} \setminus \bigcup_{j \in N_0, j < \ell} \{X_j\}$ . Since the order in which the agents plant and steal in each round of the recursion is equivalent to the order in which the corresponding tentative allocation round was performed, it holds that $A_{N_1}^{\ell-1} \subset M^{\ell-1}$ . Since $j = p_{\ell^0}^*(M^{\ell-1})$ , and $p_{\ell^0} = r_{\ell^0}$ , it holds that $r_{\ell^0}^*(A_{N_1}^{\ell-1})$ equals $j$ . Therefore $\ell^0$ will choose $j$ to $X_{\ell^0}$ as claimed.

Proving the claim for a general k is almost identical. At the planting phase of the $k^{th}$ round, the mechanism plants for every agent $\ell^{0} \in N_{0}^{k}$ their $k^{th}$ item of $A_{i}$ in $A_{N_{1}^{k}}$ and vice versa. A similar argument to the one above, shows that this item will remain available until its turn to choose an item for allocation, as by the recursion hypothesis, all agents preceding i in the Round-Robin will select the items the mechanism planted for them. Hence, the $k^{th}$ item in $A_{\ell^{0}}$ will be allocated to $X_{\ell^{0}}$ .

Finally, once the set agent i belongs to becomes a singleton, by our halting condition, $X_{i} \leftarrow X_{i} \cup A_{i}$ , so together with the previous argument, we get that for every $\ell$ , $X_{i} = A_{i}$ as needed. ☐

We continue to prove that the mechanism is robust. We first prove in Lemma 6.3 that for each agent $i$ , $v_{i}(X_{i}) \geq v_{i}^{\lceil 3n / 2\rceil}$ , and then prove in Lemma 6.5 that the value of this item is not too small compared to $\mu_i^{\lceil 3n / 2\rceil}$ .

Lemma 6.3. For every agent i, $v_{i}(X_{i}) \geq v_{i}^{\lceil 3n/2 \rceil}$ .

Proof. We first prove the claim for agents that were allocated a value during the invocation of Algorithm 7. By the definition of the algorithm and its truthfulness when agent i is allocated an item, at most n-1 items were previously allocated to other agents. Hence, she can always choose her $n^{th}$ highest valued item. Therefore, we have $v_{i}(X_{i}) \geq v_{i}^{n} \geq v_{i}^{\lceil 3n/2 \rceil}$ , as claimed.

We continue to prove the claim for the set of agents with no large predicted values. Consider the $\ell^{th}$ agent in N, $i_{\ell}$ , and consider the following coloring process. Initially, color all items in M black. We will then color all items $i_{\ell}$ was able to choose from green, and items allocated before she had the chance to choose from gray (note that these colors are unrelated to the ones in the figure). Note that an item turns green when it belongs to the tentative allocation of opposite set to $i_{\ell}$ 's and has not been taken by agents preceding her in the allocation order. We claim that by the time no black items remain, at most $\lceil3n/2\rceil-1$ have turned gray, implying that at some point during the recursion, $i_{\ell}$ could have chosen their $\lceil\frac{3n}{2}\rceil^{th}$ highest valued item (according to $r_{i_{\ell}}$ ).

We let $N^{k}$ denote the set of agents to which $i_{\ell}$ belongs to at depth k of the recursion, starting with $N^{1}=N$ . At each recursive call, $N^{k}$ is partitioned into $N_{0}^{k}, N_{1}^{k}$ . We further let $b^{k}\in\{0,1\}$ denote the index of the set to which $i_{\ell}$ belongs to: $i_{\ell}\in N_{b^{k}}^{k}$ . We will separately bound the number of items turned gray due to agents in $N_{b_{k}}^{k}$ and $N_{\neg b_{k}}^{k}$ .

In the first iteration, for k = 1, let $A_{N_{b^{0}}^{1}}$ , $A_{N_{-b^{0}}^{1}}$ denote the tentative sets allocated to the agents of $N_{0}^{1}$ and $N_{1}^{1}$ after the planting phase (i.e., at the beginning of the stealing phase).

The number of items that turn gray due to agents in $N_{b^{1}}^{1}$ is $G_{b^{1}}^{1} = \lceil \ell/2 \rceil - 1$ , since $i_{\ell}$ has access to all items in $A_{N_{-b^{1}}}^{1}$ excluding the $\lceil \ell/2 \rceil - 1$ items that were allocated to the agents in her set preceding her in the ordering. (The rest of the items in $A_{N_{-b^{1}}}^{1}$ turn green.)

Turning to $G_{\neg b^{1}}^{1}$ , each agent in the opposite set to hers, $N_{\neg b^{1}}^{1}$ , is allocated a single item (from $A_{N_{b^{1}}^{1}}^{1}$ ) before continuing to the next round of the recursion. Therefore, $G_{\neg b^{1}}^{1} = |N_{\neg b^{1}}^{1}|$ (and no item turns green).

The recursion then continues with $N^{2} = N_{b^{1}}^{1}$ and in reversed order (due to the order being reversed). Therefore, at the beginning of the second iteration, $i_{\ell}$ is in location $|N_{b^{1}}^{1}| - \lceil \ell/2\rceil$ in $N^{2}$ . After the partition phase, $i_{\ell}$ is in set $N_{b^{2}}^{2}$ and in location $\lceil \frac{|N_{b^{1}}^{1}| - \lceil \frac{\ell}{2}\rceil}{2}\rceil$ . Hence, $G_{b^{1}}^{1} = \lceil \frac{|N_{b_{1}}^{1}| - \lceil \frac{\ell}{2}\rceil}{2}\rceil - 1$ due to agents in her set preceding here in the ordering. Also, $G_{\neg b^{1}}^{1} = |N_{\neg b^{1}}^{1}|$ due to allocations to agents in the opposite set to hers.

From now on, the order is preserved, so for every $k \geq 3$ , $G_{b^k}^k = |N_{b^k}^k|$ and $G_{\neg b^k}^k = \lceil \frac{|N_b^1| - \lceil \ell / 2\rceil}{2^{k-1}} \rceil - 1$ .

We continue by bounding $\sum_{k=1}^{\lceil \log n \rceil} G_{\neg b^k}^k = \sum_{k=1}^{\lceil \log n \rceil} |N_{\neg b^k}^k|$ . Observe that if $N^k$ is even then $N_{b^k}^k = N_{\neg b^k}^k = N^k / 2$ , and if $N^k$ is odd, then either $N_{b^k}^k$ is odd and $N_{\neg b^k}^k$ is even or vice versa. In the first case, $G_{\neg b^k}^k = \lceil N^k / 2 \rceil$ and we recurse with $N_{b^k}^k$ which is of size $\lfloor N^K / 2 \rfloor$ . In the second case, $G_{\neg b^k}^k = \lfloor N^K / 2 \rfloor$ and we recurse with $N_{b^k}^k$ of size $\lceil N^k / 2 \rceil$ . Hence, we have the following recursion formula. For even $\ell$ , $T(\ell) = \ell / 2 + T(\ell / 2)$ , and for odd $\ell$ , either (a) $T(\ell) = \lceil \ell / 2 \rceil + T(\lfloor \ell / 2 \rfloor)$ or (b) $T(\ell) = \lfloor \ell / 2 \rfloor + T(\lceil \ell / 2 \rceil)$ . In Claim 6.4 below, we prove that for such a function, if it also holds that $T(1) = 0$ and $T(2) = 1$ , then $T(\ell) \leq \ell - 1$ . Therefore, we get that $\sum_{k=1}^{\lceil \log n \rceil} G_{\neg b^k}^k \leq n - 1$ .

We continue to bound $\sum_{k=2}^{\lceil\log n\rceil}G_{-b^{k}}^{k}=\sum_{k=2}^{\lceil\log n\rceil}\lceil\frac{|N_{b1}^{1}|-\lceil\ell/2\rceil}{2^{k-1}}\rceil-1$ . The sum $\sum_{k=1}^{\lceil\log X\rceil}\lceil\frac{X}{2^{k}}\rceil$ can be bounded by $\left(\sum_{k=1}^{\lceil\log X\rceil}\frac{X}{2^{k}}\right)+L$ , where L is the number of indices k for which the fraction $X/2^{k}$ is rounded up. Observe that for every X, L can be bounded above by $\lceil\log X\rceil$ as L exactly equals the number of 1 bits in the binary representation of X. Hence, the overall number of items that turn gray can be bounded as follows:

$$
G ^ {\lceil \log n \rceil} = \sum_ {k = 1} ^ {\lceil \log n \rceil} \left(G _ {\neg b ^ {k}} ^ {k} + G _ {b ^ {k}} ^ {k}\right) \tag {5}
$$

$$
\leq n - 1 + \lceil \ell / 2 \rceil - 1 + \sum_ {k = 2} ^ {\lceil \log n \rceil} \left(\left\lceil \frac {\lceil n / 2 \rceil - \lceil \ell / 2 \rceil}{2 ^ {k - 1}} \right\rceil - 1\right) \tag {6}
$$

$$
\leq n - 1 + \lceil \ell / 2 \rceil - 1 + \lceil n / 2 \rceil - \lceil \ell / 2 \rceil + \lceil \log n \rceil - \lceil \log n \rceil + 1 \tag {7}
$$

$$
\leq \lceil 3 n / 2 \rceil - 1 \tag {8}
$$

Therefore, the number of items that turn gray by the end of the recursion is at most $\lceil 3n / 2\rceil -1$ , and so $i_{\ell}$ get their $\lceil 3n / 2\rceil$ highest valued item $v_{i_{\ell}}^{\lceil 3n / 2\rceil}$ .

We now prove the claim regarding the cost of the recursion that was used in the previous lemma.

Lemma 6.4. Let $T(n)$ be such that $T(n) = n / 2 + T(n / 2)$ if $n$ is even and either (a) $T(n) = \lceil n / 2 \rceil + T(\lfloor n / 2 \rfloor)$ or (b) $T(n) = \lfloor n / 2 \rfloor + T(\lceil n / 2 \rceil)$ for odd $n$ . Also assume $T(1) = 0, T(2) = 1$ . Then $T(n) \leq n - 1$ .

Proof. We prove the claim by induction on $n$ . By $T(1) = 0$ and $T(2) = 1$ so the induction basis holds. We now assume correctness for all values smaller than $n$ and prove for $n$ .

If $n$ is even then $T(n) = n / 2 + T(n / 2) \leq n / 2 + n / 2 - 1 = n - 1$ , so the claim holds.

If $n$ is odd, then in case (a), $T(n) = \lceil n/2 \rceil + T(\lfloor n/2 \rfloor) \leq \lceil n/2 \rceil + \lfloor n/2 \rfloor - 1 = n - 1$ , and in case (b), $T(n) = \lfloor n/2 \rfloor + T(\lceil n/2 \rceil) - 1 \leq \lfloor n/2 \rfloor + \lceil n/2 \rceil - 1 = n - 1$ .

Finally, we prove that the highest valued item allocated to each agent i is not too small compared to their MMS.

Lemma 6.5. Consider an MMS for agent $i$ , and let $j^*$ be the highest valued item of $i$ in her allocation. Then

$$
v _ {i} (j ^ {*}) \geq \mu_ {i} ^ {\lceil 3 n / 2 \rceil} / \alpha \quad f o r \quad \alpha = m - \lceil 3 n / 2 \rceil - 1.
$$

Proof. Consider an MMS allocation of $M$ for $k = \lceil 3n / 2 \rceil$ , and let $A_i$ be the set such that $v_i(A_i) = \mu_i^k$ . By the assumption on $j^*$ , its value is higher then the highest valued item in $A_i$ , $v_i(j^*) \geq v_i^1(A_i)$ . Therefore, $v_i(A_i) \leq |A_i| \cdot v_i(j^*)$ , implying $v_i(j^*) \geq v_i(A_i) / |A_i| = \mu_i^k / |A_i|$ . Since $|A_i| \leq m - k - 1$ (as at least $k - 1$ items must be allocated to the $k - 1$ additional agents, it holds that $v_i(j^*) \geq \mu_i^{3n/2} / (m - \lceil 3n / 2 \rceil - 1)$ .

Proof of Theorem 6.1. The theorem follows by Lemmas 6.1, 6.2, 6.3, and 6.5.

# 7 Experimental Results

In this section, we give experiments which illustrate the role of different components of our framework for two players under various noise levels of the predictions. $^{8}$ The predictions we use for our experiments are the predicted values of the items. The noise we introduce permutes the vectors of values to match the instance's Kendall tau distance, and uses the permuted vector as prediction. We show that our framework is almost optimal for small amounts of noise while still showing resilience for higher noise levels. Moreover, we study the performance of variants which only use specific components of our framework.

When using predictions, our initial allocation procedure is a cut-and-choose procedure, implemented as follows:

- We use the first player's prediction to implement a water filling algorithm which sorts the items by values, and then partitions the items into two sets using a greedy procedure that assigns each item to the set with current lowest value.   
- We use the second player's prediction to allocated the agent the set with the higher predicted value of the two.

This allocation ensures that the second agent obtains their MMS value according to the prediction. In the data we generate, we observe that in a sampled valuation, the two sets chosen by the water filling algorithm gives the two sets the same value, up to 0.5%, which ensures that the lowest valued set obtains a 1.026-approximation to the MMS.

We inspect the following mechanisms:

1. Random: a mechanism that ignores reports and predictions and randomly partitions the items into two sets of size $m / 2$ .   
2. Random-Steal: a mechanism that ignores predictions, randomly partitions the items into two sets of size $m / 2$ , and then implements the stealing phase where each player takes their favorite item from the other player's set according to reports.   
3. Partition: a mechanism that ignores reports, and partitions the items according to predictions, using the cut-and-choose procedure described above.   
4. Partition-Steal: a mechanism that partitions the items according to predictions, using the cut-and-choose procedure described above, and then implements the stealing phase where each player takes their favorite item from the other player's set according to reports.   
5. Partition-Plant-Steal: a mechanism that implements the Plant-and-Steal framework. partitions the items according to predictions, using the cut-and-choose procedure described above, “plants” each player’s favorite item according to predictions, and then “steals” each player’s favorite item from the other player’s set according to reports.

Experiments. We consider two-player scenarios with m = 100 items. For each distance measure, we generate 1000 valuation profiles. For each pair of valuation profiles and corresponding Kendall tau distance, we generate 100 predictions based on the distance. We then assess the performance of the mechanisms described earlier on these instances. We examine two distinct cases regarding the relationship between the players' preference orders: the Correlated case, where both players have identical preference orders, although their valuation magnitudes differ, and the Uncorrelated case, where the preference orders of the players are generated independently and chosen uniformly at random. Further details on the procedures used to generate the valuations and predictions are provided in Appendix A.

Benchmark. We plot the percentage of these instances where both players get at least $(1-\epsilon)$ of their MMS value for $\epsilon=0.1,0.05,0.02$ .

Results. The results are shown in Figure 2. We first examine the performance of the two mechanisms that do not use predictions, Random and Random-Steal. Scenarios with correlated values perform significantly worse, as there is a non-negligible probability of an unbalanced partition of the relatively few high and medium valued items in a random partition. For $\epsilon$ values of 0.02, 0.05, 0.1, the Random strategy success rate is 11%, 25%, and 43%, respectively, under correlated preferences, compared to 33%, 43%, and 60% under uncorrelated preferences. Moreover, adding the stealing component significantly improves the success rate only in the uncorrelated case, as Random-Steal achieves success rates of 66%, 75%, and 87%. In the correlated case, as each player has a highly valuable item stolen, their obtained value is not expected to increase.

In the mechanisms that use predictions, Partition, Partition-Steal and Partition-Plant-Steal, the performance degrades as a function of noise, as expected. When comparing the performance of Partition, which only relies on the prediction component of our framework, and Random-Steal, which only relies on the stealing component of our framework, we notice that in the uncorrelated case, for small amount of noise guarantee a higher success rate, while as the noise increases, the stealing component becomes more instrumental to the performance. This is in tact with the theoretical results, where using the prediction is crucial to achieve the consistency guarantees, which take place when the prediction is accurate, while stealing is important to achieve robustness guarantees in case the prediction is inaccurate. As described above, in the case where the valuations are correlated, stealing is not expected to help. Interestingly, on fully noisy input, even Random outperforms Partition as Partition might partition the items into unequally-sized sets, which performs worse than the equally-sized sets Random outputs.

Our experiments show that Partition-Plant-Steal performs as well as the Partition strategy for small amounts of noise and outperforms it on uncorrelated instances for large amounts of noise. Moreover, for any amount of noise, it outperforms Random-Steal and converges to it for a fully noisy input. This illustrates the “best of both worlds” tradeoff obtained by our framework.

Finally, when comparing the Partition-Plant-Steal strategy to the Partition-Steal strategy, we observe that Partition-Plant-Steal outperforms Partition-Steal in the correlated case with a small amount of noise (worst-case scenario) for $\epsilon = 0.02$ , as planting guarantees your favorite items would not be taken. In other scenarios, Partition-Steal outperforms Partition-Plant-Steal because “planting” removes a valuable item from the player’s set that might be taken otherwise, especially in the uncorrelated case.

![](images/14c620a5eaf61555a1756426e9b52457a33591d2d16ad2abc9675c31e0043609.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Blue Line | Green Line | Light Blue Line |
| ---- | -------- | --------- | ---------- | --------------- |
| 1    | 1.0      | 1.0       | 0.75       | 0.1             |
| 5    | 0.85     | 0.85      | 0.7        | 0.1             |
| 10   | 0.75     | 0.75      | 0.6        | 0.1             |
| 40   | 0.5      | 0.5       | 0.45       | 0.1             |
| 160  | 0.35     | 0.3       | 0.3        | 0.1             |
| 640  | 0.2      | 0.15      | 0.2        | 0.1             |
| 2,560| 0.1      | 0.1       | 0.1        | 0.1             |
</details>

![](images/b36eb13d1af06ae55f2bf5439e7278e769accfb2fd031326f308fb1ed781a056.jpg)

<details>
<summary>line</summary>

| x    | Red   | Blue  | Green | Cyan  | Yellow |
| ---- | ----- | ----- | ----- | ----- | ------ |
| 1    | 1.0   | 1.0   | 1.0   | 0.67  | 0.33   |
| 5    | 0.9   | 0.9   | 1.0   | 0.67  | 0.33   |
| 10   | 0.85  | 0.85  | 1.0   | 0.67  | 0.33   |
| 40   | 0.65  | 0.75  | 0.95  | 0.67  | 0.33   |
| 160  | 0.6   | 0.7   | 0.9   | 0.67  | 0.33   |
| 640  | 0.45  | 0.7   | 0.8   | 0.67  | 0.33   |
| 2560 | 0.25  | 0.67  | 0.65  | 0.67  | 0.33   |
</details>

![](images/c82d33adf653e91c7d8054c659ade631def50e8a4d82fcbc6c337183acefc4a3.jpg)

<details>
<summary>line</summary>

| x    | Series 1 | Series 2 | Series 3 | Series 4 |
| ---- | -------- | -------- | -------- | -------- |
| 1    | 1.0      | 1.0      | 1.0      | 0.3      |
| 5    | 1.0      | 1.0      | 1.0      | 0.3      |
| 10   | 0.98     | 0.98     | 0.98     | 0.3      |
| 40   | 0.9      | 0.85     | 0.85     | 0.3      |
| 160  | 0.75     | 0.75     | 0.75     | 0.3      |
| 640  | 0.5      | 0.5      | 0.5      | 0.3      |
| 2,560| 0.3      | 0.3      | 0.3      | 0.3      |
</details>

![](images/abbff16574585efbc36f7d373fcf091582a91a108764b3b86950e2e416e7ba2c.jpg)

<details>
<summary>line</summary>

| x    | Green Line | Blue Line | Red Line | Cyan Line | Yellow Line |
| ---- | ---------- | --------- | -------- | --------- | ----------- |
| 1    | 1.0        | 1.0       | 1.0      | 0.75      | 0.4         |
| 5    | 1.0        | 0.98      | 0.98     | 0.75      | 0.4         |
| 10   | 1.0        | 0.95      | 0.95     | 0.75      | 0.4         |
| 40   | 0.98       | 0.88      | 0.85     | 0.75      | 0.4         |
| 160  | 0.95       | 0.82      | 0.75     | 0.75      | 0.4         |
| 640  | 0.9        | 0.8       | 0.55     | 0.75      | 0.4         |
| 2,560| 0.75       | 0.75      | 0.35     | 0.75      | 0.4         |
</details>

![](images/f30dbc3a706a789421dfcc67092120f04d7db4c89466ef6fd9f05d1b8d5a3925.jpg)

<details>
<summary>line</summary>

| x    | Series 1 | Series 2 | Series 3 | Series 4 |
| ---- | -------- | -------- | -------- | -------- |
| 1    | 1.0      | 1.0      | 0.55     | 0.52     |
| 5    | 1.0      | 1.0      | 0.55     | 0.52     |
| 10   | 1.0      | 1.0      | 0.55     | 0.52     |
| 40   | 1.0      | 1.0      | 0.55     | 0.52     |
| 160  | 0.98     | 0.98     | 0.55     | 0.52     |
| 640  | 0.7      | 0.85     | 0.55     | 0.52     |
| 2,560| 0.6      | 0.6      | 0.55     | 0.52     |
</details>

KT dist

![](images/7c6563ecc48bb1a666cbb74b26b72d4811ddb9570c48853a317d16901b435c54.jpg)

<details>
<summary>line</summary>

| x    | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 |
| ---- | -------- | -------- | -------- | -------- | -------- |
| 1    | 1.0      | 1.0      | 0.87     | 0.6      | 0.6      |
| 5    | 1.0      | 1.0      | 0.87     | 0.6      | 0.6      |
| 10   | 1.0      | 1.0      | 0.88     | 0.6      | 0.6      |
| 40   | 0.98     | 0.98     | 0.87     | 0.6      | 0.6      |
| 160  | 0.95     | 0.95     | 0.87     | 0.6      | 0.6      |
| 640  | 0.9      | 0.9      | 0.87     | 0.6      | 0.6      |
| 2560 | 0.85     | 0.85     | 0.87     | 0.6      | 0.6      |
</details>

KT dist   
Figure 2: Mechanism: Random (yellow), Random-Steal(cyan), Partition(red), Partition-Steal(green), Partition-Plant-Steal(blue), for the correlated case (first column) and the uncorrelated case (second column) for epsilons: 0.98 (first row), 0.95( second row) and 0.9 (third row).

# References

[1] Priyank Agrawal, Eric Balkanski, Vasilis Gkatzelis, Tingting Ou, and Xizhi Tan. Learning-augmented mechanism design: Leveraging predictions for facility location. In David M. Pennock, Ilya Segal, and Sven Seuken, editors, EC '22: The 23rd ACM Conference on Economics and Computation, Boulder, CO, USA, July 11 - 15, 2022, pages 497–528. ACM, 2022. doi:10.1145/3490486.3538306. URL https://doi.org/10.1145/3490486.3538306.   
[2] Elad Aigner-Horev and Erel Segal-Halevi. Envy-free matchings in bipartite graphs and their applications to fair division. Inf. Sci., 587:164–187, 2022. doi: 10.1016/J.INS.2021.11.059. URL https://doi.org/10.1016/j.ins.2021.11.059.   
[3] Hannaneh Akrami and Jugal Garg. Breaking the 3/4 barrier for approximate maximin share. CoRR, abs/2307.07304, 2023. doi: 10.48550/ARXIV.2307.07304. URL https://doi.org/10.48550/arXiv.2307.07304.   
[4] Hannaneh Akrami, Jugal Garg, Eklavya Sharma, and Setareh Taki. Simplification and improvement of MMS approximation. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China, pages 2485–2493. ijcai.org, 2023. doi: 10.24963/IJCAI.2023/276. URL https://doi.org/10.24963/ijcai.2023/276.   
[5] Hannaneh Akrami, Jugal Garg, and Setareh Taki. Improving approximation guarantees for maximin share. CoRR, abs/2307.12916, 2023. doi: 10.48550/ARXIV.2307.12916. URL https://doi.org/10.48550/arXiv.2307.12916.   
[6] Georgios Amanatidis, Georgios Birmpas, and Evangelos Markakis. On truthful mechanisms for maximin share allocations. In Subbarao Kambhampati, editor, Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI 2016, New York, NY, USA, 9-15 July 2016, pages 31–37. IJCAI/AAAI Press, 2016. URL http://www.ijcai.org/Abstract/16/012.   
[7] Georgios Amanatidis, Georgios Birmpas, George Christodoulou, and Evangelos Markakis. Truthful allocation mechanisms without payments: Characterization and implications on fairness. In Constantinos Daskalakis, Moshe Babaioff, and Hervé Moulin, editors, Proceedings of the 2017 ACM Conference on Economics and Computation, EC '17, Cambridge, MA, USA, June 26-30, 2017, pages 545–562. ACM, 2017. doi: 10.1145/3033274.3085147. URL https://doi.org/10.1145/3033274.3085147.   
[8] Georgios Amanatidis, Evangelos Markakis, Afshin Nikzad, and Amin Saberi. Approximation algorithms for computing maximin share allocations. ACM Trans. Algorithms, 13(4):52:1-52:28, 2017. doi: 10.1145/3147173. URL https://doi.org/10.1145/3147173.   
[9] Georgios Amanatidis, Georgios Birmpas, Federico Fusco, Philip Lazos, Stefano Leonardi, and Rebecca Reiffenhäuser. Allocating indivisible goods to strategic agents: Pure nash equilibria and fairness. In Michal Feldman, Hu Fu, and Inbal Talgam-Cohen, editors, Web and Internet Economics - 17th International Conference, WINE 2021, Potsdam, Germany, December

14-17, 2021, Proceedings, volume 13112 of Lecture Notes in Computer Science, pages 149–166. Springer, 2021. doi: 10.1007/978-3-030-94676-0\_9. URL https://doi.org/10.1007/978-3-030-94676-0\_9.   
[10] Georgios Amanatidis, Haris Aziz, Georgios Birmpas, Aris Filos-Ratsikas, Bo Li, Hervé Moulin, Alexandros A. Voudouris, and Xiaowei Wu. Fair division of indivisible goods: Recent progress and open questions. Artif. Intell., 322:103965, 2023. doi: 10.1016/J.ARTINT.2023.103965. URL https://doi.org/10.1016/j.artint.2023.103965.   
[11] Yossi Azar and Yossi Richter. The zero-one principle for switching networks. In László Babai, editor, Proceedings of the 36th Annual ACM Symposium on Theory of Computing, Chicago, IL, USA, June 13-16, 2004, pages 64–71. ACM, 2004. doi: 10.1145/1007352.1007369. URL https://doi.org/10.1145/1007352.1007369.   
[12] Moshe Babaioff, Tomer Ezra, and Uriel Feige. Fair and truthful mechanisms for dichotomous valuations. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 5119–5126. AAAI Press, 2021. doi: 10.1609/AAAI.V35I6.16647. URL https://doi.org/10.1609/aaai.v35i6.16647.   
[13] Maria-Florina Balcan, Siddharth Prasad, and Tuomas Sandholm. Bicriteria multidimensional mechanism design with side information. CoRR, abs/2302.14234, 2023. doi: 10.48550/ARXIV.2302.14234. URL https://doi.org/10.48550/arXiv.2302.14234.   
[14] Eric Balkanski, Vasilis Gkatzelis, and Xizhi Tan. Strategyproof scheduling with predictions. In Yael Tauman Kalai, editor, 14th Innovations in Theoretical Computer Science Conference, ITCS 2023, January 10-13, 2023, MIT, Cambridge, Massachusetts, USA, volume 251 of LIPIcs, pages 11:1–11:22. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2023. doi:10.4230/LIPICS.ITCS.2023.11. URL https://doi.org/10.4230/LIPIcs.ITCS.2023.11.   
[15] Eric Balkanski, Vasilis Gkatzelis, Xizhi Tan, and Cherlin Zhu. Online mechanism design with predictions. CoRR, abs/2310.02879, 2023. doi: 10.48550/ARXIV.2310.02879. URL https://doi.org/10.48550/arXiv.2310.02879.   
[16] Siddharth Barman and Sanath Kumar Krishnamurthy. Approximation algorithms for maximin fair division. ACM Trans. Economics and Comput., 8(1):5:1–5:28, 2020. doi: 10.1145/3381525. URL https://doi.org/10.1145/3381525.   
[17] Sylvain Bouveret and Michel Lemaître. Characterizing conflicts in fair division of indivisible goods using a scale of criteria. Auton. Agents Multi Agent Syst., 30(2):259–290, 2016. doi:10.1007/S10458-015-9287-3. URL https://doi.org/10.1007/s10458-015-9287-3.   
[18] Eric Budish. The combinatorial assignment problem: Approximate competitive equilibrium from equal incomes. Journal of Political Economy, 119(6):1061–1103, 2011.   
[19] Ioannis Caragiannis and Georgios Kalantzis. Randomized learning-augmented auctions with revenue guarantees. CoRR, abs/2401.13384, 2024. doi: 10.48550/ARXIV.2401.13384. URL https://doi.org/10.48550/arXiv.2401.13384.

[20] Ilan Reuven Cohen and Debmalya Panigrahi. A General Framework for Learning-Augmented Online Allocation. In 50th International Colloquium on Automata, Languages, and Programming (ICALP 2023), volume 261 of Leibniz International Proceedings in Informatics (LIPIcs), pages 43:1–43:21, 2023. ISBN 978-3-95977-278-5. doi: 10.4230/LIPIcs.ICALP.2023.43.   
[21] Nikhil R. Devanur and Thomas P. Hayes. The adwords problem: online keyword matching with budgeted bidders under random permutations. In John Chuang, Lance Fortnow, and Pearl Pu, editors, Proceedings 10th ACM Conference on Electronic Commerce (EC-2009), Stanford, California, USA, July 6–10, 2009, pages 71–78. ACM, 2009. doi: 10.1145/1566374.1566384. URL https://doi.org/10.1145/1566374.1566384.   
[22] Uriel Feige, Ariel Sapir, and Laliv Tauber. A tight negative example for MMS fair allocations. In Michal Feldman, Hu Fu, and Inbal Talgam-Cohen, editors, Web and Internet Economics - 17th International Conference, WINE 2021, Potsdam, Germany, December 14-17, 2021, Proceedings, volume 13112 of Lecture Notes in Computer Science, pages 355–372. Springer, 2021. doi: 10.1007/978-3-030-94676-0\_20. URL https://doi.org/10.1007/978-3-030-94676-0\_20.   
[23] Jugal Garg, Peter McGlaughlin, and Setareh Taki. Approximating maximin share allocations. In Jeremy T. Fineman and Michael Mitzenmacher, editors, 2nd Symposium on Simplicity in Algorithms, SOSA 2019, January 8-9, 2019, San Diego, CA, USA, volume 69 of OASIcs, pages 20:1–20:11. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2019. doi: 10.4230/OASICS.SOSA.2019.20. URL https://doi.org/10.4230/OASIcs.SOSA.2019.20.   
[24] Mohammad Ghodsi, Mohammad Taghi Hajiaghayi, Masoud Seddighin, Saeed Seddighin, and Hadi Yami. Fair allocation of indivisible goods: Improvements and generalizations. In Éva Tardos, Edith Elkind, and Rakesh Vohra, editors, Proceedings of the 2018 ACM Conference on Economics and Computation, Ithaca, NY, USA, June 18-22, 2018, pages 539–556. ACM, 2018. doi: 10.1145/3219166.3219238. URL https://doi.org/10.1145/3219166.3219238.   
[25] Vasilis Gkatzelis, Kostas Kollias, Alkmini Sgouritsa, and Xizhi Tan. Improved price of anarchy via predictions. In David M. Pennock, Ilya Segal, and Sven Seuken, editors, EC '22: The 23rd ACM Conference on Economics and Computation, Boulder, CO, USA, July 11 - 15, 2022, pages 529–557. ACM, 2022. doi: 10.1145/3490486.3538296. URL https://doi.org/10.1145/3490486.3538296.   
[26] Vasilis Gkatzelis, Alexandros Psomas, Xizhi Tan, and Paritosh Verma. Getting more by knowing less: Bayesian incentive compatible mechanisms for fair division. CoRR, abs/2306.02040, 2023. doi: 10.48550/ARXIV.2306.02040. URL https://doi.org/10.48550/arXiv.2306.02040.   
[27] Hadi Hosseini and Andrew Searns. Guaranteeing maximin shares: Some agents left behind. In Zhi-Hua Zhou, editor, Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021, pages 238–244. ijcai.org, 2021. doi: 10.24963/IJCAI.2021/34. URL https://doi.org/10.24963/ijcai.2021/34.   
[28] Hadi Hosseini, Andrew Searns, and Erel Segal-Halevi. Ordinal maximin share approximation for goods (extended abstract). In Proceedings of the Thirty-Second International

Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China, pages 6894–6899. ijcai.org, 2023. doi: 10.24963/IJCAI.2023/778. URL https://doi.org/10.24963/ijcai.2023/778.   
[29] Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean, and Neoklis Polyzotis. The case for learned index structures. In Gautam Das, Christopher M. Jermaine, and Philip A. Bernstein, editors, Proceedings of the 2018 International Conference on Management of Data, SIGMOD Conference 2018, Houston, TX, USA, June 10-15, 2018, pages 489–504. ACM, 2018. doi:10.1145/3183713.3196909. URL https://doi.org/10.1145/3183713.3196909.   
[30] David Kurokawa, Ariel D. Procaccia, and Junxing Wang. Fair enough: Guaranteeing approximate maximin shares. J. ACM, 65(2):8:1–8:27, 2018. doi: 10.1145/3140756. URL https://doi.org/10.1145/3140756.   
[31] T Lavastida, B Moseley, R Ravi, and C Xu. Learnable and instance-robust predictions for online matching, flows and load balancing. In European Symposium on Algorithms, 2021.   
[32] Pinyan Lu, Zongqi Wan, and Jialin Zhang. Competitive auctions with imperfect predictions. CoRR, abs/2309.15414, 2023. doi: 10.48550/ARXIV.2309.15414. URL https://doi.org/10.48550/arXiv.2309.15414.   
[33] Thodoris Lykouris and Sergei Vassilvitskii. Competitive caching with machine learned advice. J. ACM, 68(4):24:1–24:25, 2021. doi: 10.1145/3447579. URL https://doi.org/10.1145/3447579.   
[34] Michael Mitzenmacher. How useful is old information? IEEE Trans. Parallel Distributed Syst., 11(1):6–20, 2000. doi: 10.1109/71.824633. URL https://doi.org/10.1109/71.824633.   
[35] Benjamin Plaut and Tim Roughgarden. Almost envy-freeness with general valuations. SIAM J. Discret. Math., 34(2):1039–1068, 2020. doi: 10.1137/19M124397X. URL https://doi.org/10.1137/19M124397X.   
[36] Biaoshuai Tao. On existence of truthful fair cake cutting mechanisms. In David M. Pennock, Ilya Segal, and Sven Seuken, editors, EC '22: The 23rd ACM Conference on Economics and Computation, Boulder, CO, USA, July 11 - 15, 2022, pages 404-434. ACM, 2022. doi:10.1145/3490486.3538321. URL https://doi.org/10.1145/3490486.3538321.   
[37] Erik Vee, Sergei Vassilvitskii, and Jayavel Shanmugasundaram. Optimal online assignment with forecasts. In David C. Parkes, Chrysanthos Dellarocas, and Moshe Tennenholtz, editors, Proceedings 11th ACM Conference on Electronic Commerce (EC-2010), Cambridge, Massachusetts, USA, June 7-11, 2010, pages 109–118. ACM, 2010. doi: 10.1145/1807342.1807360. URL https://doi.org/10.1145/1807342.1807360.   
[38] Adam Wierman and Misja Nuyens. Scheduling despite inexact job-size information. In Zhen Liu, Vishal Misra, and Prashant J. Shenoy, editors, Proceedings of the 2008 ACM SIGMETRICS International Conference on Measurement and Modeling of Computer Systems, SIGMETRICS 2008, Annapolis, MD, USA, June 2-6, 2008, pages 25–36. ACM, 2008. doi:10.1145/1375457.1375461. URL https://doi.org/10.1145/1375457.1375461.

[39] Chenyang Xu and Pinyan Lu. Mechanism design with predictions. In Luc De Raedt, editor, Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pages 571–577. ijcai.org, 2022. doi: 10.24963/IJCAI.2022/81. URL https://doi.org/10.24963/ijcai.2022/81.

# A Experimental Supplement

Generating valuations. To generate interesting valuations for the players, we use a multi-step function to generate item values, since if the values are close together, any balanced partition obtains good MMS guarantees, without considering reports and predictions. Specifically, we consider a three-step (High-Med-Low) random valuation function, where each player has a high valuation with a probability of 8/m, a medium valuation with a probability of 1/4, and a low valuation with a probability of 1/2. The high valuation is U[1000, 2000], the medium valuation are U[400, 800] and the low valuations are U[100, 200] the rest of the values are U[1, 2]. Figure 3 shows the value distribution generated by our process for two players. we generate values over m = 100 items.

We generate valuations satisfying one of the two types of relations between players' preferences:

- Correlated: the two preference orders are identical (but not the values).   
- Uncorrelated: Both preference orders are chosen independently and uniformly at random.

Generating predictions. To generate predictions, we take valuations and permute elements randomly to create noise. We generate predictions under varying noise levels according to the Kendall tau distance between the valuations and the predictions. We very the Kendall tau distance between 1 to 2560, where 2560 corresponds to the expected noise level of a random permutation of 100 items. To randomly choose a permutation of a certain noise level, we start with the ordered permutation and then choose two indices j < k u.a.r. and swap items r and $r+1$ for $r \in \{j, \ldots, k-1\}$ if it increases the Kendall Tau distance by one. We repeat this process until the distance of the resulting permutation equals the desired value.

# B Deferred proofs from Section 4

Proof of Lemma 4.1. By Observation 4.1, we have $v_{i}(a_{i}^{k}) \geq v_{i}^{2k}$ , therefore

$$
v _ {i} (A _ {i}) = \sum_ {k = 1} ^ {| A _ {i} |} a _ {i} ^ {k} \geq \sum_ {k = 1} ^ {\lfloor m / 2 \rfloor} v _ {i} ^ {2 k}. \tag {9}
$$

Since $i$ 's favorite item must be absent from some set of the sets defining the MMS value,

$$
\sum_ {k = 2} ^ {m} v _ {i} ^ {k} \geq \mu_ {i}.
$$

Since the $v_{i}^{k}$ are ordered, $v_{i}^{2k} \geq v_{i}^{2k+1}$ , hence $\sum_{k=1}^{\lfloor m/2\rfloor} v_{i}^{2k} \geq \sum_{k=1}^{\lfloor m/2\rfloor} v_{i}^{2k+1}$ . Therefore,

$$
\sum_ {k = 1} ^ {\lfloor m / 2 \rfloor} v _ {i} ^ {2 k} \geq \mu_ {i} / 2 \tag {10}
$$

![](images/71a72b2573711215f5b3a00b42075d27acbe379ae62c0f41db609b13d95265b8.jpg)

<details>
<summary>scatter</summary>

| index | values |
| ----- | ------ |
| 0     | 2000   |
| 5     | 1800   |
| 10    | 1600   |
| 15    | 1400   |
| 20    | 1200   |
| 25    | 1000   |
| 30    | 800    |
| 35    | 600    |
| 40    | 400    |
| 45    | 300    |
| 50    | 200    |
| 55    | 150    |
| 60    | 100    |
| 65    | 80     |
| 70    | 60     |
| 75    | 40     |
| 80    | 20     |
| 85    | 10     |
| 90    | 5      |
| 95    | 2      |
| 100   | 1      |
</details>

Figure 3: Plotting randomly sampled valuations for two players, where the values are sorted such that lower indexed items have higher values.

By Equations (9),(10), we have:

$$
v _ {i} (A _ {i}) \geq \sum_ {k = 1} ^ {\lfloor m / 2 \rfloor} v _ {i} ^ {2 k} \geq \mu_ {i} / 2.
$$

![](images/4312d60b88df8e1a740c5846d6c648d4e326252357f4ca29c560a7b23ba552c8.jpg)

Proof of Lemma 4.2. We first prove the approximation for player 1 (the first player to be allocated). First, observe that $v_{1}(M) \geq 2\mu_{1}$ . Let $I_{1} = \{v_{1}^{3k - 2} : k = 1, \dots, \lceil m / 3 \rceil\}$ be the worst possible allocation agent 1 might get in the 1-2-Round-Robin allocation. Notice that $v_{1}(I_{1}) \geq v_{1}(M) / 3 \geq 2\mu_{1} / 3$ . By Observation 4.2, $v_{1}(a_{1}^{k}) \geq v_{1}^{3k - 2}$ . Therefore, $v_{1}(A_{1}) \geq v_{1}(I_{1}) \geq 2\mu_{1} / 3$ .

Now consider player 2. As stated in the proof of Lemma 4.1, $v_{2}(M \setminus v_{2}^{1}) \geq \mu_{2}$ . Let

$$
I _ {2} ^ {a} = \{v _ {2} ^ {3 k - 1}: k \in \mathbb {N} _ {> 0} \wedge 3 k - 1 \leq m \} \text {and} I _ {2} ^ {b} = \{v _ {2} ^ {3 k}: k \in \mathbb {N} _ {> 0} \wedge 3 k \leq m \}.
$$

First, notice that

$$
v _ {2} (I _ {2} ^ {a} \cup I _ {2} ^ {b})   \geq   2 v _ {2} (M \setminus v _ {2} ^ {1}) / 3   \geq   2 \mu_ {2} / 3.
$$

Moreover, by Observation 4.2, we have, $v_{2}(a_{2}^{2k - 1}) \geq v_{2}^{3k - 1}$ , and $v_{2}(a_{2}^{2k}) \geq v_{2}^{3k}$ . Therefore, $v_{2}(A_{2}) \geq v_{2}(I_{2}^{a} \cup I_{2}^{b}) \geq 2\mu_{2} / 3$ .

![](images/fd3a7952a0de1233d28a41e02abba530328de9f9888f131ca95a0ba2a31cfeb9.jpg)

Proof of Theorem 4.2. By Lemma 3.1, the mechanism is truthful. By Observation 4.1, each agent receives at least $\lceil m / 3\rceil$ items; combining with Lemma 3.4, we get that the mechanism is $\lfloor \frac{2m}{3}\rfloor$ -robust. Finally, if predictions correspond to valuations, by Lemma 4.2 and Lemma 4.3, the allocation is $3/2$ -approximation to the MMS. Thus, the mechanism is $2/3$ -consistent.

# C Deferred proofs from Section 5

# C.1 No Mechanism with Bounded Robustness and Consistency < 6/5

In [7] they define the following family of mechanisms.

Definition C.1 (Singleton Picking-Exchange Mechanisms [7]). A mechanism $X$ is a singleton picking-exchange mechanism if for each $i \in \{1,2\}$ , there is exactly one of two sets: either $N_i \subseteq M$ , or $E_i = \{\ell_i\}$ for a single item $\ell_i \in M$ . If $N_i$ is non-empty, then the mechanism lets player $j \neq i$ pick item $\ell \in N_i$ that maximizes $v_j(\ell)$ , and $i$ gets $N_i \setminus \{\ell\}$ . If both $E_1, E_2$ are non-empty, then the agents exchange the two items $\ell_1 \in E_1$ and $\ell_2 \in E_2$ if $v_1(\ell_2) > v_1(\ell_1)$ and $v_2(\ell_1) > v_1(\ell_2)$ . Notice that if $m > 2$ , either $E_1$ or $E_2$ is empty and there will be no exchange.

[7] showed the following.

Lemma C.1. In order for a mechanism to be truthful and have a bounded approximation, it has to be a singleton picking-exchange mechanism

We make use of this characterization in our impossibility.

Theorem 5.1. For any $\epsilon > 0$ , there is no truthful a mechanism with consistency $6/5 - \epsilon$ and bounded robustness.

Proof. Consider the case where $p_{1}=p_{2}=(1/2,1/2,1/3,1/3,1/3)$ . Notice that for the predictions, $\mu_{1}=\mu_{2}=1$ . We show that for any singleton-picking-exchange mechanism, no agent obtains both large items (of value 1/2). Consider agent 1 (the argument is symmetric for agent 2). If $N_{1}$ is non-empty, then if both large items are in $N_{1}$ , surely 1 will only get one of them. If both large items are in $N_{2}$ , then agent 2 will surely pick one of them, and agent 1 will only get one of them. If one large item is in $N_{1}$ and the other is in $N_{2}$ , each agent i will pick the large item in $N_{i}$ . If agent 2 has a large item in $E_{2}$ , then since $N_{1}$ is non-empty, $E_{1}$ is empty and agent 2 will keep the large item. Now consider the case where $E_{1}$ is non-empty. In this case, $E_{1}$ contains one item, and $N_{1}$ is empty. Since $E_{2}$ can contain at most one item, and there are more than 2 items, in this case, $E_{2}=\emptyset$ , and $|N_{2}|=4$ . Therefore, $N_{2}$ contains at least one large item. Since agent 2 will always pick the large item, agent one only gets one large item. We conclude that for any singleton picking-exchange mechanism, the large items are split among the agents. Since there are 3 small items, there must be an agent that gets at most one small item, and this agent has an overall value of at most $1/2+1/3=5/6$ , while the MMS is 1. Thus the claim follows. ☐

# C.2 Proof of Proposition 5.1

We first show the following, which implies the first half of Proposition 5.1.

Proposition C.1. There exists a partition $M = L_{1} \cup L_{2} \cup S$ and indices $\alpha_{1}, \alpha_{2}, \beta_{1}, \beta_{2}$ in [m] such that $M = [\alpha_{1}, \beta_{1}] \cup [\alpha_{2}, \beta_{2}]$ , for the sets $S_{1} = L_{1} \cup (S \cap [\alpha_{1}, \beta_{1}])$ and $S_{2} = L_{2} \cup (S \cap [\alpha_{2}, \beta_{2}])$ we have

- $\min\{v_1(S_1), v_1(S_2)\} \geq (1 - \epsilon/8)\mu_1$   
- $|L_1| + |L_2| \leq \lceil \frac{8}{\epsilon} \rceil + 2$

- $|S_1| \geq |S_2|$ .   
- For every $x$ in $L_{1}$ and $y$ in $S_{1}$ we have $v_{1}(x) > v_{1}(y)$ . Analogously, for every $x$ in $L_{2}$ and $y$ in $S_{2}$ we have $v_{1}(x) > v_{1}(y)$   
- There are $\hat{j},\hat{j}^{\prime}\in L_{1}$ satisfying $\hat{j}\in \arg \max_{\ell \in S_1}v_1(\ell)$ and $\hat{j}^{\prime}\in \arg \max_{\ell \in S_{1}\setminus \hat{j}}v_{1}(\ell),$

We do this by inspecting two types of items, large items, with value greater than $\epsilon\mu_{1}/4$ , and small items with value at most $\epsilon\mu_{1}/4$ . We first show that there are $O(1/\epsilon)$ large items, therefore, separating these items into two bundles require at most $O(1/\epsilon)$ intervals. Moreover, we can find a separation of the largest items into two sets, $L_{1}, L_{2}$ , and a single index $j \in [m]$ such that all small items to the left of j (including) together with $L_{1}$ form $S_{1}$ , and all items to the right of j (excluding) together with $L_{2}$ form $S_{2}$ , such that $S_{1}, S_{2}$ satisfy the approximation requirement. It is easy to see that this increases the number of intervals by at most 1.

We start by showing there are not too many large items.

Lemma C.2. There are at most $\left\lceil\frac{8}{\epsilon}\right\rceil$ items with value strictly greater than $\epsilon\mu_{1}$ for agent 1.

Proof. Let items with value greater than $\epsilon\mu_{1}/4$ be the large items. Suppose there are at least $\lceil\frac{8}{\epsilon}\rceil+1$ large items. If $\lceil\frac{8}{\epsilon}\rceil$ is even, consider a partition $(S_{1},S_{2})$ such that each $S_{i}$ gets at least $\lceil\frac{8}{\epsilon}\rceil/2$ large items and the rest are allocated arbitrarily. If $\lceil\frac{8}{\epsilon}\rceil$ is odd, consider the allocation in which each $S_{i}$ gets $(\lceil\frac{8}{\epsilon}\rceil+1)/2$ large items and the rest are allocated arbitrarily. In either case, each $S_{i}$ gets at least $\lceil\frac{8}{\epsilon}\rceil/2\geq\frac{4}{\epsilon}$ large items. Thus, $\min\{v_{1}(S_{1}),v_{1}(S_{2})\}>\epsilon\mu_{1}/4\cdot\frac{4}{\epsilon}=\mu_{1}$ , a contradiction. ☐

We are now ready to prove Proposition C.1.

Proof of Proposition C.1. Consider the set of large items, $L = \{j \in [m] : v_1(j) > \epsilon \mu_1 / 4\}$ , and let $S = M \setminus L$ be the set of small items.

We give a constructive proof which finds both sets $L_{1}, L_{2}$ and an index j satisfying the condition stated in the lemma. Let

$$
(L _ {1}, L _ {2}) \in \arg \max _ {(T _ {1}, T _ {2}): T _ {1} \cup T _ {2} = L} \min _ {j \in \{1, 2 \}} v _ {1} (S _ {j}).
$$

We use the following procedure to find $j$ .

1. Let $j_{\ell}=0$ and $j_{r}=m$ .

2. While $j_{\ell} \neq j_{r}$ :

(a) Let $S_{\ell} = L_1 \cup \{j' \in S : j' \leq j_{\ell}\}$ and $S_r = L_2 \cup \{j' \in S : j' > j_r\}$ .

(b) If $v_{1}(S_{\ell}) < v_{1}(S_{r})$ :

\- $j_{\ell} := j_{\ell} + 1$ .

(c) Else:

\- $j_{r} := j_{r} - 1$ .

3. Set $j := j_{\ell} = j_{r}$ .

We consider two cases:

Case 1: $j = 0$ (or symmetrically, $j = m$ ). Without loss of generality, suppose that $j = m$ . We first show that if $v_{1}(S_{1}) < v_{1}(S_{2})$ then $\min\{v_{1}(S_{1}), v_{1}(S_{2})\} = \mu_{1}$ . Notice that since $S_{1}$ gets all the small items, it must be the case that $v_{1}(L_{1}) < v_{1}(L_{2})$ . Suppose there's a different partition $T_{1} \cup T_{2}$ such that $\min\{v_{1}(T_{1}), v_{1}(T_{2})\} > \min\{v_{1}(S_{1}), v_{1}(S_{2})\}$ . Without loss of generality, let $v_{1}(T_{1} \cap L) \leq v_{1}(T_{2} \cap L)$ (otherwise, we can rename both bundles). By the definition of $L_{1}, L_{2}$ , it must be the case that $v_{1}(L_{1}) \geq v_{1}(T_{1} \cap L)$ . Thus, Since $T_{1} \setminus (T_{1} \cap L) \subseteq S$ , it must be that

$$
v _ {1} (S _ {1}) = v _ {1} (L _ {1}) + v _ {1} (S) \geq v _ {1} (T _ {1} \cap L) + v _ {1} (T _ {1} \setminus (T _ {1} \cap L)) = v _ {1} (T _ {1}) \geq \min \{v _ {1} (T _ {1}), v _ {1} (T _ {2}) \},
$$

a contradiction.

On the other hand, if $v_{1}(S_{1}) \geq v_{1}(S_{2}) = v_{1}(L_{2})$ , by condition 2b of the above procedure, it must be the case that when $j_{\ell}$ was equal m - 1,

$$
v _ {1} (S _ {\ell}) <   v _ {1} (S _ {r}) = v _ {1} (L _ {2}) = v _ {1} (S _ {2}).
$$

Thus,

$$
v _ {1} (S _ {1}) = v _ {1} (S _ {\ell}) + v _ {1} (m) <   v _ {1} (S _ {2}) + \epsilon \mu_ {1} / 4.
$$

We get that

$$
v _ {1} (S _ {2}) \geq v _ {1} (S _ {1}) - \epsilon \mu_ {1} / 4 \geq 2 \mu_ {1} - v _ {1} (S _ {2}) - \epsilon \mu_ {1} / 4 \Rightarrow
$$

$$
\min \{v _ {1} (S _ {1}), v _ {1} (S _ {2}) \} = v _ {1} (S _ {2}) \geq (1 - \epsilon / 8) \mu_ {1}, \tag {11}
$$

where the second inequality follows since $2\mu_{1} \leq v_{1}(S_{1}) + v_{1}(S_{2})$ .

Case 2: $0 < j < m$ . In this case, since both $j_{\ell}$ and $j_{r}$ were moved, there were some values of $j_{\ell}$ and $j_{r}$ such that $v_{1}(S_{\ell}) \leq v_{1}(S_{r})$ and some values such that $v_{1}(S_{\ell}) > v_{1}(S_{r})$ . Assume initially that $v_{1}(S_{\ell}) \leq v_{1}(S_{r})$ . Since at each step of the procedure, the lower-valued bundle can increase by at most $\epsilon \mu_1 / 4$ , when the first item is added to $S_{\ell}$ such that $v_{1}(S_{\ell}) > v_{1}(S_{r})$ , it must be the case that $v_{1}(S_{\ell}) \leq v(S_{r}) + \epsilon \mu_1 / 4$ . It is easy to see that the invariant where $|v_{1}(S_{\ell}) - v_{1}(S_{r})| \leq \epsilon \mu_1 / 4$ is kept throughout the run of the procedure. Therefore, this also holds for the final $S_{1}$ and $S_{2}$ . Thus, we can use the same reasoning of Eq. (11) to conclude that $\min \{v_{1}(S_{1}), v_{1}(S_{2})\} \geq (1 - \epsilon / 8)\mu_{1}$ .

Thus, the sets $S_{1}$ and $S_{2}$ have a form $S_{1}=L_{1}\cup(S\cap[1,j])$ and $S_{2}=L_{2}\cup(S\cap[j+1,m])$ and have the form required. If $|S_{1}|<|S_{2}|$ we can swap our definitions for the sets $S_{1}$ and $S_{2}$ , thus ensuring that $|S_{1}|>|S_{2}|$ . Due to our definitions of $L_{1}$ and $L_{2}$ we have for every x in $L_{1}$ and y in $S_{1}$ we have $v_{1}(x)>v_{1}(y)$ . Analogously, for every x in $L_{2}$ and y in $S_{2}$ we have $v_{1}(x)>v_{1}(y)$ .

We can ensure that There are $\hat{j},\hat{j}^{\prime}\in L_{1}$ satisfying

$$
\hat {j} \in \arg \max _ {\ell \in S _ {1}} v _ {1} (\ell) \text { and } \hat {j} ^ {\prime} \in \arg \max _ {\ell \in S _ {1} \setminus \hat {j}} v _ {1} (\ell),
$$

by adding such values from $S \cap [\alpha_1, \beta_1]$ to $L_1$ (we see that after this all other properties still hold). Overall, we see that $|L_1| + |L_2| \leq \lceil \frac{8}{\epsilon} \rceil + 2$ , as required.

Now, we proceed to proving the second half of Proposition 5.1. We will need the following lemma.

Lemma C.3. Let $k_{1}$ and $k_{2}$ be positive integers satisfying $k_{1} > k_{2}$ , and let f be a function mapping $[k_{1}]$ to non-negative real numbers. Then, there exist a pair of integers $\alpha, \beta, \alpha'$ and $\beta'$ in $[k_{1}]$ such that $|\left[\alpha, \beta\right] \cup [\alpha', \beta']| = k_{2}$ and

$$
\frac {\sum_ {i \in [ \alpha , \beta ] \cup [ \alpha^ {\prime} , \beta^ {\prime} ]} f (i)}{k _ {2}} \leq \frac {\sum_ {i \in [ k _ {1} ]} f (i)}{k _ {1}}
$$

Proof. We prove the lemma using the probabilistic method. Let j be a uniformly random integer in $[k_{1}]$ , and choose $\alpha, \beta, \alpha'$ and $\beta'$ such that

$$
[ \alpha , \beta ] \cup [ \alpha^ {\prime}, \beta^ {\prime} ] = \{j, j + 1 \mod k _ {1}, \dots , j + k _ {2} - 1 \mod k _ {1} \}.
$$

We see that indeed a set chosen as above can be represented as a union of two intervals. Now, since j is chosen uniformly at random form $[k_{1}]$ , we see that for every element i in $[k_{1}]$ we have

$$
\operatorname * {P r} _ {j \sim [ k _ {1} ]} [ i \in \{j, j + 1 \mod k _ {1}, \dots , j + k _ {2} - 1 \mod k _ {1} \} ] = \frac {k _ {2}}{k _ {1}}.
$$

Thus via linearity of expectation we have:

$$
\mathbb {E} _ {j \sim [ k _ {1} ]} \left[ \frac {1}{k _ {2}} \sum_ {{i \in \{j, j + 1 \mod k _ {1}, \dots , j + k _ {2} - 1 \mod k _ {1} \}}} f (i) \right] = \frac {1}{k _ {1}} \sum_ {i \in [ k _ {1} ]} f (i).
$$

Thus, since $f(i)$ is non-negative for all values of i, we see that for some specific choice of j it has to be the case that

$$
\frac{1}{k_{2}}\sum_{\substack{i\in \{j,j + 1\mod k_{1},\dots ,j + k_{2} - 1\mod k_{1}\} \\ }}f(i)\leq \frac{1}{k_{1}}\sum_{i\in [k_{1}]}f(i),
$$

which finishes the proof.

Now, we apply the lemma above. If $m < 4\lceil\frac{t}{\epsilon}\rceil + 2$ we can satisfy Proposition C.1 by:

1. First choosing a partition $M = S_{1} \cup S_{2}$ such that $\min(v_{1}(S_{1}), v_{1}(S_{2})) \geq \mu_{1}$ and $|R_{1}| \geq |R_{2}|$ .   
2. Define $L_{2} := S_{2}$ , put the smallest $\lfloor m/2 \rfloor - |S_{2}$ elements of $S_{1}$ into S, and define $L_{1}$ to contain the rest of elements in $S_{1}$ .   
3. Define $\alpha_{1}=\alpha_{3}=1,\beta_{1}=\beta_{3}=m,\alpha_{2}=\beta_{2}=\alpha_{4}=\beta_{4}=m+1.$

Overall, this allocates $S'$ to be the bottom $\lfloor m/2\rfloor$ elements of $S_{1}$ . We see that this suffices to guarantee the properties that $S'$ needs to satisfy in Proposition 5.1.

Therefore, henceforth we can assume that $m > 4\lceil \frac{t}{\epsilon}\rceil +2$ . Since $|S_1|\geq m / 2$ and $|L_{1}|\leq \frac{8}{\epsilon} +2$ , and $S_{1} = L_{1}\cup (S\cap [\alpha_{1},\beta_{1}])$ this implies that $|S\cap [\alpha_1,\beta_1])| > 3\left\lceil \frac{t}{\epsilon}\right\rceil >m / 2$ . Thus, we can ensure that $|S^{\prime}| = \lfloor m / 2\rfloor -|S_{2}|$ using a subset $S^{\prime}\subset S\cap [\alpha_{1},\beta_{1}]$ .

If $|S_2| = 1$ we only need choose $S'$ to satisfy $|S'| = \lfloor m / 2 \rfloor - |S_2|$ and $v_1(S') \leq v_1(S_1 \setminus \{\hat{j}, \hat{j}'\}) / 2$ . First of all, since every element in $L_1$ is larger than any element in $S \cap [\alpha_1, \beta_1]$ , we see that this is also true in average

$$
\frac {\sum_ {\ell \in S _ {1}} v _ {1} (\ell)}{| S _ {1} |} \leq \frac {\sum_ {\ell \in S \cap [ \alpha_ {1} , \beta_ {1} ])} v _ {1} (\ell)}{| S \cap [ \alpha_ {1} , \beta_ {1} ]) |} \tag {12}
$$

Then, applying Lemma C.3 to the set $S \cap [\alpha_1, \beta_1]$ we see that there exist disjoint subsets $[\alpha_3, \beta_3]$ and $[\alpha_4, \beta_4]$ of $[\alpha_1, \beta_1]$ such that $|S \cap ([\alpha_3, \beta_3] \bigcup [\alpha_4, \beta_4]))| = \lfloor m/2 \rfloor - |S_2|$

$$
\frac {\sum_ {\ell \in S \cap [ \alpha_ {1} , \beta_ {1} ])} v _ {1} (\ell)}{| S \cap [ \alpha_ {1} , \beta_ {1} ]) |} \leq \frac {\sum_ {\ell \in S \cap ([ \alpha_ {3} , \beta_ {3} ] \bigcup [ \alpha_ {4} , \beta_ {4} ]))} v _ {1} (\ell)}{| S \cap ([ \alpha_ {3} , \beta_ {3} ] \bigcup [ \alpha_ {4} , \beta_ {4} ])) |} \tag {13}
$$

Combing Equations 12 and 13, we see that taking $S' = S \cap ([\alpha_3, \beta_3] \bigcup [\alpha_4, \beta_4])$ satisfies $v_1(S') \leq v_1(S_1)/2$ and the other requirements of Proposition 5.1.

Now, suppose $|S_2| > 1$ . Since by Proposition C.1, the set $S$ does not contain the two largest elements $\hat{j}$ and $\hat{j}'$ of $S_1$ , as well as the fact that every element in $L_1$ is at least as large as any element in $S \cap [\alpha_1, \beta_1]$ , we see that every every element in $S_1 \setminus \{\hat{j}, \hat{j}'\}$ is either in $S \cap [\alpha_1, \beta_1]$ or greater than every element in $S \cap [\alpha_1, \beta_1]$ . This implies that:

$$
\frac {\sum_ {\ell \in S _ {1} \backslash \{\hat {j} , \hat {j} ^ {\prime} \}} v _ {1} (\ell)}{| S _ {1} | - 2} \leq \frac {\sum_ {\ell \in S \cap [ \alpha_ {1} , \beta_ {1} ])} v _ {1} (\ell)}{| S \cap [ \alpha_ {1} , \beta_ {1} ]) |} \tag {14}
$$

Then, we can again apply applying Lemma C.3 to the set $S \cap [\alpha_1, \beta_1]$ ) we see that there exist disjoint subsets $[\alpha_3, \beta_3]$ and $[\alpha_4, \beta_4]$ of $[\alpha_1, \beta_1]$ such that $|S \cap ([\alpha_3, \beta_3] \bigcup [\alpha_4, \beta_4]))| = \lfloor m/2 \rfloor - |S_2|$ .

$$
\frac {\sum_ {\ell \in S \cap \left[ \alpha_ {1} , \beta_ {1} \right]}) v _ {1} (\ell)}{| S \cap \left[ \alpha_ {1} , \beta_ {1} \right]) |} \leq \frac {\sum_ {\ell \in S \cap \left(\left[ \alpha_ {3} , \beta_ {3} \right] \bigcup \left[ \alpha_ {4} , \beta_ {4} \right]\right))} v _ {1} (\ell)}{| S \cap \left(\left[ \alpha_ {3}, \beta_ {3} \right] \bigcup \left[ \alpha_ {4} , \beta_ {4} \right]\right)) |} \tag {15}
$$

Combing Equations 14 and 15, we see that taking $S' = S \cap ([\alpha_3, \beta_3] \bigcup [\alpha_4, \beta_4])$ satisfies $v_1(S') \leq v_1(S_1 \setminus \{\hat{j}, \hat{j}'\}) / 2$ as required in Proposition 5.1. Note that this also implies that $v_1(S') \leq v_1(S_1\}) / 2$ since $\hat{j}, \hat{j}'$ have the top two largest values of $v_1$ in $S_1$ . Overall, this finishes the proof of Proposition 5.1.

# C.3 Proof of $(2 + \epsilon)$ -consistency.

It remains to show that the algorithm is $2 + \epsilon$ -consistent. We will be referencing the variables $\hat{j}_1, \hat{j}_2, \tilde{j}_1, \tilde{j}_2, T_1$ and $T_2$ within the Plant-And-Steal framework (Algorithm 1).

We first reason about agent 2. First, notice that since agent 2 has a higher value for $\tilde{S}_{i_2}$ ,

$$
v _ {2} \left(\tilde {S} _ {i _ {2}}\right) \geq \mu_ {2}.
$$

Since the mechanism had a chance to pick item $\hat{j}_2$ from $T_1$ as $\tilde{j}_2$ , it must be the case that $v_2(\tilde{j}_2) \geq v_2(\hat{j}_2)$ (and possibly $\tilde{j}_2 = \hat{j}_2$ ). If $\tilde{j}_1 = \hat{j}_1$ , then $T_2 \setminus \tilde{j}_1 = \tilde{S}_{i_2} \setminus \hat{j}_2$ , and

$$
\mu_ {2} \leq v _ {2} (\tilde {S} _ {i _ {2}}) = v _ {2} (\tilde {S} _ {i _ {2}} \backslash \hat {j} _ {2}) + v _ {2} (\hat {j} _ {2}) \leq v _ {2} (T _ {2} \backslash \tilde {j} _ {1}) + v _ {2} (\tilde {j} _ {2}) = v _ {2} (X _ {2}).
$$

Otherwise, $\tilde{j}_1\in \tilde{S}_{i_2}$ , and

$$
\tilde {S} _ {i _ {2}} \backslash \hat {j} _ {2} \backslash \tilde {j} _ {1} \subset T _ {2} \backslash \tilde {j} _ {1} \Rightarrow v _ {2} \left(\tilde {S} _ {i _ {2}} \backslash \hat {j} _ {2} \backslash \tilde {j} _ {1}\right) \leq v _ {2} \left(T _ {2} \backslash \tilde {j} _ {1}\right). \tag {16}
$$

Since $\hat{j}_2$ is the item with the highest value for agent 2 in $\tilde{S}_{i_2}$ , $v_2(\tilde{j}_2) \geq v_2(\hat{j}_2) \geq v_2(k_1)$ . Combining with Eq. (16), we get that

$$
v _ {2} (T _ {2} \setminus \tilde {j} _ {1} \cup \{\tilde {j} _ {2} \}) \geq v _ {2} (\tilde {S} _ {i _ {2}} \setminus \hat {j} _ {2}).
$$

Moreover,

$$
v _ {2} (T _ {2} \setminus \tilde {j} _ {1} \cup \{\tilde {j} _ {2} \}) \geq v _ {2} (\tilde {j} _ {2}) \geq v _ {2} (\hat {j} _ {2}).
$$

Thus,

$$
v _ {2} (X _ {2}) = v _ {2} (T _ {2} \setminus \tilde {j} _ {1} \cup \{\tilde {j} _ {2} \}) \geq v _ {2} (\tilde {S} _ {i _ {2}}) / 2 = \mu_ {2} / 2,
$$

as desired.

It is left to show that $v_{1}(X_{1}) \geq \mu_{1} / (2 + \epsilon)$ . If $i_{1} = 2$ , then

$$
v _ {1} (\tilde {S} _ {i _ {1}}) = v _ {1} (\tilde {S} _ {2}) \geq v _ {1} (S _ {2}) \geq (1 - \epsilon / 4) \mu_ {1}.
$$

In this case, the same exact arguments used for agent 2 can be harnessed to show that $v_{1}(X_{1}) \geq (1 - \epsilon /4)\mu_{1} / 2 \geq \mu_{1} / (2 + \epsilon)$ . Thus, it is left to consider the case where $i_1 = 1$ .

Consider the $(S_{1}, S_{2})$ partition that is set in the first step of Cut-and-Balance-and-Choose. Since $v_{1}(S') \leq v_{1}(S_{1})/2$ , we have

$$
v _ {1} (\tilde {S} _ {1}) \geq v _ {1} (S _ {1}) / 2 \geq (1 - \epsilon / 4) \mu_ {1} / 2 \geq \frac {\mu_ {1}}{2 + \epsilon}.
$$

If $\tilde{j}_2 = \hat{j}_2$ , we have that

$$
v _ {1} (X _ {1}) = v _ {1} (\tilde {S} _ {1} \cup \{\tilde {j} _ {1} \} \setminus \{\hat {j} _ {1} \}) \geq v _ {1} (\tilde {S} _ {1}) \geq \frac {\mu_ {1}}{2 + \epsilon},
$$

where the first inequality follows since $v_{1}(\tilde{j}_{1}) \geq v_{1}(\hat{j}_{1})$ .

Note also that if $|S_{2}| = 1$ i.e., $S_{2} = \{a\}$ , if $\hat{j}_{2} \neq a$ then $v_{1}(X_{1}) \geq v_{1}(S_{2})$ since $a \in T_{2}$ , similarly if $\tilde{j}_{2} \neq a$ then $v_{1}(X_{1}) \geq v_{1}(S_{2})$ , finally we have $\hat{j}_{2} = k_{2} = a$ and $v_{1}(X_{1}) \geq \mu_{1}/(2 + \epsilon)$ an in the first case.

Therefore, we assume $\tilde{j}_2 \neq \hat{j}_2$ and $|S_2| > 1$ , and let $\hat{j}_1' \in \arg \max_{j \in \tilde{S}_1 \setminus \{\hat{j}_1\}} v_1(j)$

$$
\begin{array}{l} v _ {1} (X _ {1}) = v _ {1} \left(T _ {1} \cup \{\tilde {j} _ {1} \} \backslash \{\tilde {j} _ {2} \}\right) \\ = v _ {1} (T _ {1}) + v _ {1} (\tilde {j} _ {1}) - v _ {1} (k _ {2}) \\ \geq v _ {1} (\tilde {S} _ {1} \cup \{\hat {j} _ {2} \} \setminus \{\hat {j} _ {1} \}) + v _ {1} (\hat {j} _ {1}) - v _ {1} (\tilde {j} _ {2}) \\ \geq v _ {1} (\tilde {S} _ {1} \setminus \{\hat {j} _ {1} \}) + v _ {1} (\hat {j} _ {1}) - v _ {1} (\tilde {j} _ {2}) \\ = v _ {1} (S _ {1} \setminus S ^ {\prime} \setminus \{\hat {j} _ {1} \}) + v _ {1} (\hat {j} _ {1}) - v _ {1} (\tilde {j} _ {2}) \\ \geq v _ {1} (S _ {1} \setminus S ^ {\prime} \setminus \{\hat {j} _ {1} \}) + v _ {1} (\hat {j} _ {1}) - v _ {1} (\hat {j} _ {1} ^ {\prime}) \\ = v _ {1} (S _ {1} \setminus S ^ {\prime} \setminus \{\hat {j} _ {1}, \hat {j} _ {1} ^ {\prime} \}) + v _ {1} (\hat {j} _ {1}), \\ \end{array}
$$

where the first inequality is since, $v_{1\tilde{j}_1} = \max_{j \in T_2} v_{1j} \geq v_{1\hat{j}_1}$ . The second inequality is since $v_1(\hat{j}_2) \geq 0$ , the third inequality is by $\hat{j}_1'$ definition since $\tilde{j}_2 \in \tilde{S}_1 \setminus \hat{j}_1$ by our assumption that $k_2 \neq \hat{j}_2$ . Finally, we have have $|S_1 \setminus S' \setminus \{\hat{j}_1, \hat{j}_1'\}| \geq |S'|$ since

$$
| S _ {1} | - 2 - | S ^ {\prime} | = | S _ {1} | - 2 - (m / 2 - | S _ {2} |) = | S _ {1} | - 2 - (m / 2 - (m - | S _ {1} |)) = m / 2 - 2 \geq | S ^ {\prime} |,
$$

where the last inequality is since $|S_{2}| > 1$ . Since we handle the case $|S_{2}| = 1$ earlier, we can here assume $|S_{2}| > 1$ in which case the set $S'$ is required to satisfy $v_{1}(S') \leq v_{1}(S_{1} \setminus \{\hat{j}, \hat{j}'\}) / 2$ . Therefore, we have $v_{1}(S_{1} \setminus S' \setminus \{\hat{j}_{1}, \hat{j}_{1}'\} \geq v_{1}(S')$ .

$$
\begin{array}{l} (1 - \epsilon / 4) \mu_ {1} \leq v _ {1} (S _ {1}) = v _ {1} (S _ {1} \setminus S ^ {\prime} \setminus \{\hat {j} _ {1}, \hat {j} _ {1} ^ {\prime} \}) + v _ {1} (\hat {j} _ {1}) + v _ {1} (\hat {j} _ {1} ^ {\prime}) + v _ {1} (S ^ {\prime}) \\ \leq v _ {1} \left(S _ {1} \backslash S ^ {\prime} \backslash \{\hat {j} _ {1}, \hat {j} _ {1} ^ {\prime} \}\right) + 2 \cdot v _ {1} (\hat {j} _ {1}) + v _ {1} \left(S ^ {\prime}\right) \\ \leq 2 \cdot v _ {1} (S _ {1} \setminus S ^ {\prime} \setminus \{\hat {j} _ {1}, \hat {j} _ {1} ^ {\prime} \}) + 2 \cdot v _ {1} (\hat {j} _ {1}) \\ \leq 2 \cdot v _ {1} (X _ {1}), \\ \end{array}
$$

which implies that $v_{1}(X_{1}) \geq \frac{\mu_{1}}{2 + \epsilon}$ , finishing the proof.