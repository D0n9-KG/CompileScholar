# The Complexity of Extending Fair Allocations of Indivisible Goods

Argyrios Deligkas $^{1}$ , Eduard Eiben $^{1}$ , Robert Ganian $^{2}$ , Tiger-Lily Goldsmith $^{1}$ and Stavros D. Ioannidis $^{1}$

$^{1}$ Department of Computer Science, Royal Holloway, University of London $^{2}$ Algorithms and Complexity group, TU Wien, Austria

# Abstract

We initiate the study of computing envy-free allocations of indivisible items in the extension setting, i.e., when some part of the allocation is fixed and the task is to allocate the remaining items. Given the known NP-hardness of the problem, we investigate whether—and under which conditions—one can obtain fixed-parameter algorithms for computing a solution in settings where most of the allocation is already fixed. Our results provide a broad complexity-theoretic classification of the problem which includes: (a) fixed-parameter algorithms tailored to settings with few distinct types of agents or items; (b) lower bounds which exclude the generalization of these positive results to more general settings. We conclude by showing that—unlike when computing allocations from scratch—the non-algorithmic question of whether more relaxed EFX allocations exist can be completely resolved in the extension setting.

# 1 Introduction

Finding a “fair” allocation of indivisible items or resources to a provided set of agents, each with their own preferences over the resources, is one of the central tasks arising in the area of computational social choice. The arguably most classical and established notion of fairness used in these settings is envy-freeness, where we ask for an allocation $\pi: M \to N$ from the set M of items to the set N of agents such that for each pair of agents $i, j \in N$ , i prefers $\pi^{-1}(i)$ to $\pi^{-1}(j)$ . In other words, no agent envies another agent. It is well known that an envy-free allocation need not exist, and in fact, determining whether one exists is NP-complete [BL08].

The aforementioned intractability has led to a flurry of research to circumvent this issue. One approach that has been proposed is to consider “less restrictive” versions of envy-freeness instead, with the aim of not only ensuring that the computation of an assignment is tractable but—even more desirably—that one always exists. Notably, it is known that one can always compute, in polynomial time, an allocation which is envy-free up to one item (EF1) [AMN20, Bud11, IKSY24]. However, such allocations may sometimes be considered very far from “fair”. A recently proposed intermediate notion between envy-free and EF1 allocations is envy-free up to any item (EFX) $[CKM^{+}19, FMP24]$ , but it is not known whether EFX allocations always exist and their polynomial-time computability remains open as well.

Another notable approach to tackling the problem of computing envy-free allocations is to identify precise conditions under which it can be solved efficiently. This is typically done by investigating the problem through the lens of parameterized complexity [DF13, $\mathrm{CFK}^{+}15]$ —a refinement of the classical complexity paradigm where inputs are analyzed not only with respect to their size $n$ , but also to a numerical parameter $k$ which measures some well-defined quantity. While we cannot hope to obtain an $n^{\mathcal{O}(1)}$ algorithm for computing envy-free allocations in general, in this setting one aims to design so-called fixed-parameter tractable (or FPT) algorithms for the problem, that is, algorithms running in

time $f(k) \cdot n^{\mathcal{O}(1)}$ for some computable function f. The simplest parameters considered in past works on fair division include, for example, the number of agents or items; however, such parameterizations place strong restrictions on the input instances, and so more recent works have focused on developing algorithms parameterized by the number of agent types (where two agents have the same type if they have identical preferences over the items), or the number of item types (where two items have the same type if they are valued the same by every agent) [DEG $^{+}$ 21, EGHO23, NR23], see also [BLM16, GOR19].

While the aforementioned approaches have by now provided a fairly detailed understanding of computing an envy-free allocation from scratch, in this article we turn to the problem of extending a partial allocation—that is, computing an envy-free allocation when part (or even most) of the allocation is already fixed. This envy-free allocation extension problem arises naturally whenever one needs to deal with resources that have already been assigned or must be assigned to certain agents—consider, e.g., the case where a few new items are made available after an allocation has been fixed and we are not allowed to take items from agents, or the setting where most employees in a company already have fixed tasks, but we need to distribute a set of new tasks to recently hired employees. It is worth noting that while the complexity of extending partial solutions has been extensively researched in settings as diverse as data completion [GKOS18, KFN21, $\mathrm{EGK^{+}23}$ , KFN23] and graph drawing $[\mathrm{ABF^{+}15}$ , ADP19, $\mathrm{EGH^{+}20}$ , $\mathrm{BGK^{+}23}]$ , this has not yet been systematically studied in classical resource allocation in spite of the many situations in which one may need to deal with some items being pre-assigned. $^{1}$

# 1.1 Our Contribution

We study the envy-free allocation extension problem in the classical setting of additive utilities, as outlined below.

ENVY-FREE ALLOCATION EXTENSION 

<table><tr><td>Input:</td><td>A set M of indivisible items, a set N of n agents with additive valuations, and a partial allocation γ :  $M' \to N$  of some subset  $M' \subseteq M$  of items.</td></tr><tr><td>Question:</td><td>Does there exist an extension allocation π :  $(M \setminus M') \to N$  of the remaining (“open”) items such that γ ∪ π is envy-free?</td></tr></table>

Notice that ENVY-FREE ALLOCATION EXTENSION coincides with the usual problem of obtaining an envy-free allocation from scratch in the special case where $\gamma = \emptyset$ , and hence is necessarily NP-hard. However, this intractability does not properly reflect typical usage scenarios of the problem: in many cases of interest, one may be dealing with allocations $\gamma$ which are “almost complete”. Hence, we turn to the aforementioned parameterized paradigm and ask which parameterizations of the partial assignment $\gamma$ allow us to achieve fixed-parameter traceability, i.e., obtain a fixed-parameter algorithm.

We begin our investigation by considering the complexity of the problem with the number $k = |M \setminus M'|$ of yet-to-be-allocated or open items as the parameter. It is worth noting that ENVY-FREE ALLOCATION EXTENSION can be solved via a trivial $n^{\mathcal{O}(k)}$ algorithm (a so-called XP algorithm), and that computing an envy-free allocation from scratch is trivially fixed-parameter tractable w.r.t. k—hence, one could have expected ENVY-FREE ALLOCATION EXTENSION to be fixed-parameter tractable w.r.t. k as well. However it turns out that this is not the case: we will later show (in Theorem 2) that the problem is W[1]-hard when parameterized by k, which excludes fixed-parameter tractability under well-established complexity assumptions.

Unsatisfied by this outcome, we ask whether one can at least obtain a fixed-parameter algorithm under some mild restrictions on the input instance. For example instances with a bounded number of item or agent types arise naturally in large-scale models: one often aggregates similar items into categories (depending on the setting, these may be, e.g., “cars” and “houses”, or “PhD students” and “Postdocs”), and the inherent limitations of how preferences are collected may result in many agents being represented as having the same preferences. Inspired by related works on resource allocation $[DEG^{+}21, EGHO23, NR23]$ , we consider additionally parameterizing by the number $n_{t}$ of the agent types or the number $m_{t}$ of item types. As our first positive result, we obtain an algorithm when we bound the number of agent types.

Theorem 1. ENVY-FREE ALLOCATION EXTENSION is fixed-parameter tractable when parameterized by the number k of open items plus the number $n_{t}$ of agent types.

Turning to the analogous question for item types, we show that the situation is entirely different: allocating k open items is intractable even when there are only a few item types.

Theorem 2. ENVY-FREE ALLOCATION EXTENSION is W[1]-hard when parameterized by the number k of open items plus the number $m_{t}$ of item types.

The proof of Theorem 1 relies on a combination of branching techniques with insights into the structure of hypothetical solutions; while it is not trivial, in this case it is the lower bound—Theorem 2—which presents a much more surprising result. Indeed, at first glance the problem seems far from intractable: we only need to assign k items to a set of agents, and for each open item i, we can easily test whether assigning it to an agent j would make any of the other agents in N envy j w.r.t. their current items. The only caveat is that we do not know which of the other agents will receive the remaining open items—and in spite of there being only at most k open items in total, we show that the uncertainty of how these will be allocated can be used to obtain a highly non-trivial reduction from the classical W[1]-hard MULTICOLORED CLIQUE problem [DF13].

In the second part of our article, we take aim at the setting where many items may be open, but we are only allowed to allocate these items to at most p-many agents. This is precisely the situation that arises when, e.g., allocating tasks to a few new employees, or needing to distribute new items to individuals while adhering to constraints on shipping costs, or how many of the individuals should be contacted. Formally, this is captured by asking for an allocation $\pi$ which assigns all open items to a set S of at most p-many agents. We will distinguish between whether the set S of potential recipients is provided in the input (in which case we speak of RESTRICTED ENVY-FREE ALLOCATION EXTENSION, or simply REFAE) or can be selected along with $\pi$ (giving rise to FREE ENVY-FREE ALLOCATION EXTENSION, or simply FEFAE). Distinguishing these two variants of our problem is not only necessary to differentiate between distinct usage scenarios, but will also turn out to have an impact on the problem's complexity.

We note that while both variants are in NP, attempting to solve either of them when parameterized by p (the number of agents that receive items) alone is entirely hopeless—and this holds even in the strong setting, i.e., with unary-encoded valuations.

Theorem 3. Both REFAE and FEFAE are strongly NP-hard even when $p \leq 2$ .

This once again raises the question of whether one can use a bound on $n_{t}$ or $m_{t}$ to obtain efficient algorithms. Given Theorem 1 and Theorem 2, one might expect that agent types would once again allow us to obtain fixed-parameter algorithms in this case. And yet, we show that when dealing with p the situation is reversed: unexpectedly, here it is the number $m_{t}$ of item types that provides the key to tractability. Indeed, one can establish strong W[1]-hardness as well as weak paraNP-hardness of REFAE and FEFAE when parameterized by the total number of agents in the instance (which upper-bounds both $n_{t}$ and p) directly from existing results in the literature [BBN16]. Before proceeding, we at least show that the former lower bound is tight: the problems admit XP-algorithms in the case where the utilities are not encoded in binary.

<table><tr><td rowspan="2" colspan="2">Few items k</td><td colspan="2">Few Recipients p</td></tr><tr><td>REFAE</td><td>FEFAE</td></tr><tr><td></td><td>W[1]-hard (Theorem 2)</td><td colspan="2">NP-hard,  $p \leq 2$  (Theorem 3)</td></tr><tr><td>Agent types  $n_t$ </td><td>FPT (Theorem 1)</td><td colspan="2">XP, unary valuations (Theorem 4)</td></tr><tr><td>Item types  $m_t$ </td><td>W[1]-hard (Theorem 2)</td><td>FPT (Theorem 5)</td><td>XP (Theorem 5)</td></tr></table>

Table 1: An overview of our results. The W[1]-hardness results of Theorem 2 are complemented by the naive, brute force, XP algorithms. The XP algorithms for REFAE and FEFAE when parameterized by $n_{t}$ are essentially tight, since matching strong W[1]-hardness and weak paraNP-hardness lower bounds follow from the literature [BBN16].

Theorem 4. If the valuations are encoded in unary, both REFAE and FEFAE are in XP when parameterized by the number p of recipients plus the number $n_{t}$ of agent types.

Turning to the setting with few item types, we provide a fixed-parameter algorithm for the restricted case (i.e., when the set S of recipients is already given) and an XP algorithm for the free case (when the task is to also find S on its own). Here, the latter result is the best one could hope for at this point, as fixed-parameter tractability is immediately excluded by Theorem 2. In contrast to Theorem 4, both of these algorithms can be applied regardless of whether the utilities are encoded in unary or binary.

Theorem 5. When parameterized by p plus the number $m_{t}$ of item types, REFAE and FEFAE are fixed-parameter tractable and XP-tractable, respectively.

An overview of our complete complexity-theoretic classification of ENVY-FREE ALLOCATION EXTENSION under the considered restrictions is provided in Table 1.

While not part of our main technical contributions, in the final section of the article, we also make a few observations about the behavior of relaxed variants of envy-freeness in the extension setting that may be of general interest to the community. In particular, we prove that the known result guaranteeing the existence of an EF1 allocation can be strengthened to also hold when extending any provided partial allocation. Moreover, this result is tight: there exist partial EFX allocations which cannot be extended to a full EF1 allocation. Similarly, while the question of whether an EFX allocation “from scratch” is guaranteed to exist remains one of the main open questions in the field, we show that there exist partial envy-free allocations which cannot be extended to a full EFX allocation.

# 2 Preliminaries

For an integer $\ell$ , we use $[\ell]$ as shorthand for the set $\{1,\ldots,\ell\}$ . We also use the standard $O^{*}$ notation to suppress polynomial factors of the input size in the running time.

Our instances include a set of indivisible items $M = \{a_{1}, \ldots, a_{m}\}$ and a set of n agents N = [n]. Every agent $i \in N$ has an additive valuation function $v_{i}$ that assigns a non-negative value $v_{i}(a)$ for every item $a \in M$ and for every subset, or bundle, of items $B \subseteq M$ we denote $v_{i}(B) := \sum_{j \in B} v_{i}(a_{j})$ . If $v_{i}(a) = v_{j}(a)$ for every $a \in M$ , then we say that agents i and j are of the same type. We will use $n_{t}$ to denote the number of different agent types in N. If for two items a and $a'$ we have that $v_{i}(a) = v_{i}(a')$ for every agent i, then we say that a and $a'$ are of the same type. We will use $m_{t}$ to denote the number of different item types in M.

Partial and Extended Allocations. We will assume that M is partitioned into a set of given items $M'$ and a set of open items A. Formally, $M = M' \cup A$ and $M' \cap A = \emptyset$ . A partial allocation $\gamma = (\gamma_{1}, \gamma_{2}, \ldots, \gamma_{n})$ is a partition of $M'$ into n (potentially empty) sets, where $\gamma_{i}$ is the bundle of given

items to agent i. An allocation of open items $\pi = (\pi_{1}, \pi_{2}, \ldots, \pi_{n})$ partitions A into n (again, potentially empty) sets, where agent i gets bundle $\pi_{i}$ . Given a partial allocation $\gamma$ and an allocation $\pi$ , we get an extended allocation $\gamma \cup \pi$ , where agent i gets allocated bundle $\gamma_{i} \cup \pi_{i}$ .

Fairness Concepts. We will consider three different fairness notions. An extended allocation $\gamma \cup \pi$ is said to be:

- envy-free, denoted EF, if for any pair of agents $i$ and $j$ it holds that $v_{i}(\gamma_{i} \cup \pi_{i}) \geq v_{i}(\gamma_{j} \cup \pi_{j})$ ;   
- envy-free up to any item, denoted EFX, if for any pair of agents $i$ and $j$ and any item $a \in \gamma_j \cup \pi_j$ it holds that $v_i(\gamma_i \cup \pi_i) \geq v_i(\gamma_j \cup \pi_j \setminus a)$ ;   
- envy-free up to one item, denoted EF1, if for any pair of agents $i$ and $j$ there exists an item $a \in \gamma_j \cup \pi_j$ such that $v_i(\gamma_i \cup \pi_i) \geq v_i(\gamma_j \cup \pi_j \setminus a)$ .

Observe that EF1 is a more relaxed fairness notion compared to EFX, which in turn is more relaxed than EF.

Fair Allocation Extension Problems. We are interested in the computation of an allocation $\pi$ of open items, such that the extended allocation is fair according to some of the above-mentioned criteria; we define the problems for envy-free solutions, but they could naturally be extended for EF1 and EFX. The input to each version of our problem is a set of agents with their valuations and a partial allocation $\gamma$ . In the first version, termed ENVY-FREE ALLOCATION EXTENSION and already introduced in the Introduction, we do not constrain the extension allocation in any way. In the other two versions of the problem, the set of recipients is restricted in some way. One option is to restrict the set of agents that are allowed to receive an open item under $\pi$ , which for brevity we denote REFAE.

RESTRICTED ENVY-FREE ALLOCATION EXTENSION (REFAE) 

<table><tr><td>Input:</td><td>A set M of indivisible items, a set N of n agents, a partial allocation γ : M&#x27; → N of some subset M&#x27; ⊆ M of items, and a set S ⊆ N.</td></tr><tr><td>Question:</td><td>Does there exist an allocation π : (M \ M&#x27;) → N of the open items such that: (a) πi = ∅ for every i∉S; (b) γ ∪ π is envy-free?</td></tr></table>

A different option is to restrict just the number of recipients. We term this problem FEFAE.

FREE ENVY-FREE ALLOCATION EXTENSION (FEFAE) 

<table><tr><td>Input:</td><td>A set M of indivisible items, a set N of n agents, a partial allocation γ:M&#x27;→N of some subset M&#x27;⊆M of items, and p∈N.</td></tr><tr><td>Question:</td><td>Does there exist an allocation π:(M\M&#x27;)→N of the open items such that:(a)πi≠∅ for at most p agents;(b)γ∪π is envy-free?</td></tr></table>

While we formally study the decision variants of these problems for complexity-theoretic reasons, every algorithm obtained in this article is constructive and can also output a suitable allocation if one exists.

Parameterized complexity. We refer to the standard books for a basic overview of parameterized complexity theory $[CFK^{+}15, DF13]$ . At a high level, parameterized complexity studies the complexity of a problem with respect to its input size, n, and the size of a parameter k. A problem is fixed parameter tractable by k, if it can be solved in time $f(k) \cdot \text{poly}(n)$ , where f is a computable function. A less favorable, but still positive, outcome is an XP algorithm, which has running-time $\mathcal{O}(n^{f(k)})$ ; problems admitting such algorithms belong to the class XP. Showing that a problem is W[t]-hard rules out the existence of a fixed-parameter algorithm under the well-established assumption that W[t]-hard ≠ FPT.

# 3 Parameterizing by the Number of Open Items

We start our investigation by considering the complexity of ENVY-FREE ALLOCATION EXTENSION when the number k of open items is included in the parameterization. In other words, we ask under which conditions one can efficiently solve the extension problem for only a few open items.

As our baseline, we observe that the problem admits a trivial XP algorithm: one can enumerate all possible assignments of the open items to agents in $\mathcal{O}(n^{k})$ time and check whether any of these is envy-free. However, such algorithms are considered highly inefficient in the community [DF13, CFK $^{+}$ 15], and the central question tackled by this section is whether (or under which conditions) one can achieve fixed-parameter tractability.

Before settling the problem when parameterized by k alone, we first provide a fixed-parameter algorithm for ENVY-FREE ALLOCATION EXTENSION when the parameterization also includes the number of agent types. Intuitively, this provides a positive result for the case where there are only a few open items and the agents can be partitioned into a few groups with identical preferences. We remark that this setting is not trivial, as agents which we consider to have identical preferences (e.g., due to polling limitations) can and will often have different pre-assigned items.

The following observation will be useful in the proof and follows directly from the definition of agent types and envy-free allocations.

Observation 1. Assume we are given an envy-free partial allocation $\gamma$ such that no pair of agents in agent type X envy each other. If a solution $\pi$ assigns a set Q of positively valued items to an agent $j \in X$ , then $\pi$ must assign items of the same value as $v_{j}(Q)$ to all other agents in X.

Theorem 1. ENVY-FREE ALLOCATION EXTENSION is fixed-parameter tractable when parameterized by the number k of open items plus the number $n_{t}$ of agent types.

Proof. As the very first step, we observe that if there is an agent $i$ envying agent $j$ , then $\pi$ must assign some open item to $i$ . Hence, we begin by exhaustively branching, i.e., guessing, for each such agent $i$ , which of the at most $k$ items will be assigned to $i$ , and restart our considerations with the instance updated accordingly. For the following, we hence consider that no agent envies another. If we can't do this, meaning $k$ reached 0, the algorithm stops.

We begin by partitioning the set of agent types into small and large types. An agent type is classed as small when it contains at most $k$ agents. We define $Z$ as the set of agents in all of the small types, the size of $Z$ is upper-bounded by $k \cdot n_t$ .

Secondly, we exhaustively branch to determine the following information about the allocation of the k items to agents. First of all, we branch over all partitions of the open items into bundles, of which there are $k^{k}$ , where we will assume that $\pi$ assigns each of the bundles to distinct agents. Next, for each of the at most k bundles, we branch to determine either: which of the agents in Z will receive it; or that it will not be assigned to any agent in Z, i.e., the bundle goes to an agent from one of the large agent types. This gives us $k \cdot n_{t} + 1$ many possible agents to assign a bundle to. The overall branching factor up to this point is upper-bounded by $k^{k} \cdot (k \cdot n_{t} + 1)^{k}$ .

At this point, we check whether any agent from Z envies another agent; if that is the case, then we can correctly reject the current branch, as we may assume to have precisely guessed the bundles assigned to all agents in Z. Similarly, no pair of agents outside of Z may envy each other due to the exhaustive procedure carried out in the first paragraph of the proof. At this point, we recall that by Observation 1 and the definition of large agent types, each agent outside of Z must receive a bundle that they value as 0. Hence, if an agent outside of Z were to envy an agent in Z, we may also correctly reject the current branch.

Now, it remains to assign the remaining open items to the agents outside of Z, the large agent types, without creating envy. For each of the bundles, we can determine whether assigning it to an agent $i \notin Z$ creates envy from any other agent in the instance (where for agents in Z we assume them to receive the

bundles specified in our branching, while for agents outside of $Z$ , we assume that nothing changed). To complete the proof, we construct an auxiliary bipartite graph $G$ where:

- one side contains the set of all remaining bundles (i.e., those not assigned to agents in $Z$ ),   
- the other side contains the set of all agents outside of $Z$ , and,   
- there is an edge between bundle $b$ and agent $i$ if and only if $b$ can be assigned to $i$ without creating envy from any other agent.

We compute a maximum matching for this bipartite graph in polynomial time.

The matching outputs an allocation of bundles to agents. It suffices to check whether $G$ admits a matching that saturates all bundles, i.e., all items have been assigned to some agent. Note that, if not all bundles are assigned, then there exists some bundle we couldn't assign to an agent without causing envy. Thus, if we did find such a matching, then there is an envy-free solution (by assigning the bundles according to the matching).

Conversely, if there is a solution for ENVY-FREE ALLOCATION EXTENSION, these 3 steps are guaranteed to find it. This holds because in the case of the small agent types we enumerated their possible bundles, and there is no solution where the large agent types receive items that they value positively, because of Observation 1. The overall running time can be upper-bounded by $(k \cdot n_{t})^{\mathcal{O}(k)} \cdot n^{2}$ . ☐

Turning back to ENVY-FREE ALLOCATION EXTENSION parameterized by k alone, our next result shows that the aforementioned trivial XP algorithm can be viewed as “optimal” in the sense of the problem not admitting any fixed-parameter algorithm under the well-established complexity assumption of W[1] $\not\subseteq$ FPT. Naturally, one would then ask whether fixed-parameter tractability can be achieved at least when the parameterization is enriched by the number $m_{t}$ of item types—a setting that can be seen as complementary to the one settled in Theorem 1. Surprisingly, we exclude fixed-parameter algorithms for the problem, even in this significantly more restrictive setting via a highly involved reduction.

Theorem 2. ENVY-FREE ALLOCATION EXTENSION is W[1]-hard when parameterized by the number k of open items plus the number $m_{t}$ of item types.

# 3.1 Proof of Theorem 2

We reduce from the classical W[1]-hard MULTICOLORED CLIQUE problem: Given a q-partite graph, where each part is assigned a unique color, decide whether the graph contains a clique of size q. Let $\mathcal{I} = ((\mathrm{V}, \mathrm{E}), q)$ be an instance of MULTICOLORED CLIQUE, where q is the number of colors. Without loss of generality, assume that E only contains edges adjacent to vertices of different color and that for any pair of colors, there exist at least one pair of adjacent vertices (otherwise trivially a multicolored clique does not exist). Let $V = \bigcup_{1 \leq k \leq q} V_k$ , where $V_k$ is the set that contains the vertices of color k and let $E = \bigcup_{1 \leq i < j \leq q} E_{ij}$ , where $E_{ij}$ is the set that contains the edges of E that are adjacent to vertices that belong to sets $V_i$ and $V_j$ respectively. We will construct an instance $I'$ of ENVY-FREE ALLOCATION EXTENSION and prove that an envy-free allocation extension for $I'$ exists if and only if I contains a clique of size q.

Construction. For each vertex set $V_{i}$ we assume a vertex-agent group $V_{i} = \{\alpha_{j}^{i} | j \in V_{i}\}$ and for each edge set $E_{ij}$ an edge-agent group $E_{ij} = \{\eta_{z}^{ij} | z \in E_{ij}\}$ . To avoid notational overflow, whenever it is clear from the context we omit the superscript in the notation of an agent. Moreover we assume that the agents in each group are provided in an arbitrary order, i.e., the agents in the vertex-agent group $V_{i}$ have the form $\alpha_{1}, \ldots, \alpha_{|V_{i}|}$ and analogously the agents in the edge-agent group $E_{ij}$ have the form of $\eta_{1}, \ldots, \eta_{|E_{ij}|}$ .

A notion that will be useful in the subsequent arguments is that of adjacent agents:

Definition 1 (Adjacent agents). Let $z = (x, y)$ be an edge in $\mathcal{I}$ and let $\eta_z, \alpha_x$ and $\alpha_y$ be the corresponding agents in $\mathcal{I}'$ . We refer to any pair of these three agents as adjacent agents.

For each vertex-agent group $V_{i}$ , the construction will make use of pre-allocated items from the following three item types $\{\Box^{i},\triangle^{i},\star^{i}\}$ , while for each edge-agent group $E_{ij}$ , the construction will also make use of pre-allocated items from the following three item types $\{\Box^{ij},\triangle^{ij},\star^{ij}\}$ . The partial allocation $\gamma$ will provide the agents of the vertex-agent group $V_{i}$ with some combination of items from “their” types $\{\Box^{i},\triangle^{i},\star^{i}\}$ , while the agents of the edge-agent group $E_{ij}$ will receive items not only from “their” types $\{\Box^{ij},\triangle^{ij},\star^{ij}\}$ but also items of the types $\{\Box^{i},\triangle^{i},\Box^{j},\triangle^{j}\}$ .

Partial allocation. We provide the details of how the partial allocation $\gamma$ is constructed.

• Each agent $\alpha_{x}$ where $x \in [|V_{i}|]$ , within a vertex-agent group $V_{i}$ holds a bundle consisting of:

- $x$ copies of $\square^i$ ,   
- $2|V_i|^2 - x^2 - x$ copies of $\triangle^i$ ,   
- one copy of $\star^{i}$ .

Overall the bundle of the agent $\alpha_{x}$ in $V_{i}$ has the following form $\{\underbrace{\square^{i},\ldots,\square^{i}}_{x},\underbrace{\triangle^{i},\ldots,\triangle^{i}}_{2|V_{i}|^{2}-x^{2}-x},\star^{i}\}$ .

\- Each agent $\eta_z$ , within the edge-agent group $E_{ij}$ that is adjacent to some agents $\alpha_x \in V_i$ and $\alpha_y \in V_j$ , holds a bundle consisting of

- $z$ copies of $\square^{ij}$ ,   
- $2|E_{ij}|^2 - z^2 - z$ copies of $\triangle^{ij}$ ,   
- one copy of $\star^{ij}$ ,   
- precisely the same number of $\Box^i (\Box^j)$ and $\triangle^i (\triangle^j)$ items as agent $\alpha_{x}(\alpha_{y})$ . That is $x$ copies of $\Box^i$ , $y$ copies of $\Box^j$ , $2|V_i|^2 - x^2 - x$ copies of $\triangle^i$ and $2|V_j|^2 - y^2 - y$ copies of $\triangle^j$ .

Overall the bundle of an agent $\eta_{z}$ in $E_{ij}$ has the following form

$$
\{\underbrace {\square^ {i j} , \ldots , \square^ {i j}} _ {z}, \underbrace {\triangle^ {i j} , \ldots \triangle^ {i j}} _ {2 | E _ {i j} | ^ {2} - z ^ {2} - z}, \star^ {i j}, \underbrace {\square^ {i} , \ldots , \square^ {i}} _ {x}, \underbrace {\triangle^ {i} , \ldots \triangle^ {i}} _ {2 | V _ {i} | ^ {2} - x ^ {2} - x}, \underbrace {\square^ {j} \ldots , \square^ {j}} _ {y}, \underbrace {\triangle^ {j} \ldots , \triangle^ {j}} _ {2 | V _ {j} | ^ {2} - y ^ {2} - y} \}.
$$

We furthermore assume precisely $q + \binom{q}{2}$ pairwise-distinct open items. A set $S = \{s_{1}, \ldots, s_{q}\}$ of vertex items, containing one item tailored for each vertex-agent group (or equivalently color). A set $T = \{\tau_{12}, \tau_{13}, \cdots, \tau_{(q-1)(q)}\}$ of edge items, containing one item tailored for each edge-agent group $E_{ij}$ .

Combining the open items with the pre-allocated items, we get that the total number of item types used in the generated instance $I'$ is $4q + 4\binom{q}{2}$ .

Agents' valuations. We proceed to define the valuation function of the agents. Each agent will have a unique valuation function defined as follows:

• Each agent $\alpha_{x}$ within a vertex-agent group $V_{i}$ values:

- $\square^{i}$ as $2x + 1$ ,   
- $\triangle^i$ as 1,   
- $\star^i$ as 0.

This describes the valuation for items allocated to $\alpha_{x}$ in $\gamma$ . Next, for the item types that are not pre-assigned to $\alpha_{x}$ , the agent uses the following valuation:

- Each $\triangle^{\circ}$ and each $\square^{\circ}$ for $\circ \neq i$ has a value of 0,   
- For each $j \neq i$ , $\star^{ij}$ has a value of 0,   
- Each $\star^{\circ}$ whose valuation is not defined up to now, has a value that is precisely equal to the value of $\alpha_{x}$ 's bundle in the partial allocation $\gamma$ ”, i.e., $2|V_{i}|^{2} + x^{2}$ (see Lemma 1).

We observe that the construction at this point ensures that agent $\alpha_{x}$ values his own bundle identically to that of any agent from a different vertex-agent group. Moreover, $\alpha_{x}$ values his own bundle identically to that of edge agents that are adjacent to $\alpha_{x}$ , while for the remaining agents in those edge-agent groups, he prefers his own bundle.

Regarding the open items, $\alpha_{x}$ values $s_{i}$ as 1 and for all $j \neq i$ the items $t_{ij}$ as 1, and all the remaining open items as 0.

• Each agent $\eta_{z}$ within an edge-agent group $E_{ij}$ values:

- $\square^{ij}$ as $2z + 1$ ,   
- $\triangle^{ij}$ as 1,   
- $\star^{ij}$ as well as all remaining items included in its bundle by $\gamma$ as 0.

Next, for the item types that are not pre-assigned to $\eta_z$ , the agent uses the following valuation:

- each $\triangle^{\circ}$ and each $\square^{\circ}$ for $\circ \neq ij$ has a value of 0,   
- each $\star^{\circ}$ where $\circ \neq ij$ has a value that is precisely equal to the value of $\eta_z$ 's bundle as assigned by $\gamma$ , i.e., $2|E_{ij}| + z^2$ .

We observe that by construction, $\eta_{z}$ will value its bundle exactly the same as that of every agent outside his group. Regarding the open items, $\eta_{z}$ values only the open item $t_{ij}$ as 1 and all others as 0.

We proceed by establishing two lemmas that must hold in any envy-free allocation extension. The statements of these lemmas arise from the way we defined the partial allocation $\gamma$ and set the agents' valuations.

First we show that we can allocate a single open item to an agent of some group, without creating envy among the agents within that group. We will prove that in the partial allocation $\gamma$ , no envy exists among agents within the same agent group. In fact every agent strictly prefers its own bundle to the bundle of any other agent within the same group.

Lemma 1. Let $V_{i}$ be a vertex-agent group and consider any two vertex agents $\alpha_{x}$ and $\alpha_{y}$ within that group. It holds that $v_{\alpha_{x}}(\gamma_{\alpha_{x}}) > v_{\alpha_{x}}(\gamma_{\alpha_{y}})$ .

Proof. From the way we defined the partial allocation $\gamma$ and set the valuations, we get that:

$$
\begin{array}{l} v _ {\alpha_ {x}} (\gamma_ {\alpha_ {x}}) = x \cdot v _ {\alpha_ {x}} (\square^ {i}) + (2 | V _ {i} | ^ {2} - x ^ {2} - x) \cdot v _ {\alpha_ {x}} (\triangle^ {i}) + v _ {\alpha_ {x}} (\star^ {i}) \\ = x \cdot (2 x + 1) + (2 | V _ {i} | ^ {2} - x ^ {2} - x) \\ = 2 | V _ {i} | ^ {2} + x ^ {2}. \\ \end{array}
$$

The valuation of agent $\alpha_{x}$ for the bundle of agent $\alpha_{y}$ in $\gamma$ is:

$$
\begin{array}{l} v _ {\alpha_ {x}} (\gamma_ {\alpha_ {y}}) = y \cdot v _ {\alpha_ {x}} (\Box^ {i}) + (2 | V _ {i} | ^ {2} - y ^ {2} - y) \cdot v _ {\alpha_ {x}} (\triangle^ {i}) + v _ {\alpha_ {x}} (\star^ {i}) \\ = y \cdot (2 x + 1) + (2 | V _ {i} | ^ {2} - y ^ {2} - y) \\ = 2 | V _ {i} | ^ {2} + 2 x y - y ^ {2}. \\ \end{array}
$$

Since the inequality $2|V_{i}|^{2} + x^{2} > 2|V_{i}|^{2} + 2xy - y^{2}$ holds, we conclude that $v_{\alpha_{x}}(\gamma_{\alpha_{x}}) > v_{\alpha_{x}}(\gamma_{\alpha_{y}})$ . Furthermore, since all valuations are integer values, even assigning one item that is valued 1 by every agent in a group, to some agent within that group, does not create envy among the agents in that group.

![](images/516b0e80c47c61276a1874c79a2c1b5808edd1464bf25cf8a3a5fa1c9631af9d.jpg)

Using the exact same arguments we can prove the following lemma for edge-agent groups.

Lemma 2. Let $E_{ij}$ be an edge-agent group and consider any two edge agents $\eta_x$ and $\eta_y$ within that group. It holds that $v_{\eta_x}(\gamma_{\eta_x}) > v_{\eta_x}(\gamma_{\eta_y})$ .

The next lemma establishes that for an extension for $\gamma$ to be envy-free, adjacent agents must receive triplets of open items that correspond to adjacent vertices in I.

Lemma 3. In any envy-free allocation extension for $\gamma$ , every triplet of open items $s_{i}, \tau_{ij}$ and $s_{j}$ must be allocated to adjacent agents belonging within the groups $V_{i}, E_{ij}$ and $V_{j}$ respectively.

Proof. First we prove that any open edge-item $\tau_{ij}$ must be allocated to some agent within the edge-agent group $E_{ij}$ . If we allocate $\tau_{ij}$ to some agent within another edge-agent group say $E_{\kappa\lambda}$ , then since all agents in $E_{ij}$ value item $\star^{\kappa\lambda}$ (which appears in the bundle of every agent in $E_{\kappa\lambda}$ ) precisely equal to their bundle in $\gamma$ , they become envious. Similarly, if we allocate $\tau_{ij}$ to a vertex-agent group $V_{i}$ , then since all agents in $E_{ij}$ value item $\star^{i}$ (which appears in the bundle of every agent in $V_{i}$ ) equal to their bundle in $\gamma$ , they all become envious. Combining this argument with the fact that edge-agents are solely interested in open edge-items, shows that the edge-item $\tau_{ij}$ must be allocated to some agent in the group $E_{ij}$ .

Additionally, the open vertex-item $s_{i}$ must be allocated to an agent within the vertex-agent group $V_{i}$ . If we allocate item $s_{i}$ to some agent within another vertex-agent group $V_{j}$ , then from the way we defined the valuations each agent in $V_{i}$ values item $\star^{j}$ (which is owned by every agent in $V_{j}$ ) precisely equal to their bundle as assigned in $\gamma$ . Thus allocating $s_{i}$ to some agent within $V_{j}$ makes all agents in $V_{i}$ envious. Similarly, if we allocate $s_{i}$ to some agent within the edge-agent group $E_{\kappa\lambda}$ , then since all agents in $V_{i}$ value item $\star^{\kappa\lambda}$ precisely equal to their bundle in $\gamma$ they all become envious.

If $s_{i}$ is assigned to some agent $\eta_{z}$ within the edge-agent group $E_{ij}$ , then there must exist an adjacent agent $\alpha_{x}$ that belongs to the vertex-agent group $V_{i}$ , that becomes envious of $\eta_{z}$ . That holds because $\alpha_{x}$ values the bundle of $\eta_{z}$ equally to his bundle in the partial allocation $\gamma$ . Then, we cannot allocate any other open item in such a way to reduce this envy.

Finally we can show that every triplet of open items $s_{i}, \tau_{ij}$ and $s_{j}$ must be allocated to adjacent agents within the groups $V_{i}, E_{ij}$ and $V_{j}$ respectively. Indeed, from the above arguments we know that $\tau_{ij}$ will be allocated to some agent, say $\eta_{z}$ , within $E_{ij}$ . From the way we defined the valuations, only the two adjacent agents to $\eta_{z}$ , that belong to vertex-agent groups $V_{i}$ and $V_{j}$ become envious, since both these agents value the bundle of $\eta_{z}$ precisely the same as their bundle in $\gamma$ . Thus, to eliminate envy we must allocate the open items $s_{i}$ and $s_{j}$ to these agents respectively. Notice that from Lemmas 1 and 2 we do not induce envy among agents within the same group since we allocate only one item to each group. ☐

We proceed to show that a q-clique in the initial instance I exists if and only if an envy-free allocation extension for $\gamma$ exists in the constructed instance $I'$ .

From I to $I'$ . Assume that I contains a clique of size q. We extend the partial allocation $\gamma$ of $I'$ by allocating the set of open items as follows: Pick an edge $z = (x, y)$ of the clique and without loss of generality assume that vertex x has color i and vertex y has color j. Moreover let $\eta_{z}, \alpha_{x}$ and $\alpha_{y}$ be the corresponding adjacent agents in $I'$ . Allocate the triplet of items $s_{i}, \tau_{ij}$ and $s_{j}$ to the adjacent agents $\alpha_{x}, \eta_{z}$ and $\alpha_{y}$ respectively. From Lemmas 1 and 2 we get that no envy emerges among agents of the same agent group. Additionally from Lemma 3, we get that no envy emerges among agents of different agent groups. Consequently allocating the items that way produces an envy-free allocation extension for $\gamma$ .

From $I'$ to I. Assume that the partial allocation $\gamma$ of $I'$ extends to a complete envy-free allocation, say $\gamma'$ . As argued above, the statement of Lemma 3 must be satisfied in order for $\gamma'$ to be envy-free. That means that every triplet of open items $s_{i}, \tau_{ij}$ and $s_{j}$ is allocated to adjacent agents within the groups $V_{i}, E_{ij}$ and $V_{j}$ respectively. Moreover notice that only one agent within every vertex/edge-agent group receives an open vertex/edge item. Combining these facts, we get that any two agents of the vertex-agent groups that receive open items in $\gamma'$ , must be adjacent agents. This implies that in I, the vertices that correspond to the agents of the vertex-agent groups that receive the items are adjacent vertices of different color. Consequently these vertices form a multicolor clique of size q.

# 4 Parameterizing by the Number of Recipients

In this section, we address the more complex situation of needing to allocate (a possibly large number of) open items to at most p recipients. As mentioned in the Introduction, here it will be important to distinguish whether the set of these recipients is fixed and provided on the input (REFAE), or whether the task also includes the identification of this set (FEFAE). Moreover—and unlike in the case studied in the previous section—the existence of efficient algorithms will sometimes depend on whether we may assume the valuations to be encoded in unary, or whether they are encoded in binary. It will be useful to recall that lower bounds achieved in the former (latter) setting are called strong (weak).

As regards the complexity of these problems when parameterized by p alone, it is easy to observe that both problems are weakly NP-hard already when $p \leq 2$ as this directly generalizes both the previously-studied setting of all items being open [BBN16] and SUBSET SUM, both of which are weakly NP-hard. On the other hand, the problem is trivial for p = 1. Below, we show that in the extension setting studied here, NP-hardness holds for p = 2 even in the unary (i.e., strong) setting, contrasting the known existence of an XP algorithm for that case when all items are open [BBN16].

Theorem 3. Both REFAE and FEFAE are strongly NP-hard even when $p \leq 2$ .

Proof. We reduce from INDEPENDENT SET that is known to be NP-hard. An instance I of the INDEPENDENT SET problem consists of a graph $(V, E)$ and an integer $\ell$ . The task is to decide whether the graph contains a subset $V' \subseteq V$ where no two vertices in $V'$ are adjacent and $|V'| = \ell$ . We construct an instance $I'$ of ENVY-FREE ALLOCATION EXTENSION and prove that an envy-free allocation extension for $I'$ with only two recipients exists if and only if instance I contains an independent set of size $\ell$ .

Construction. We assume a set $N = \{1, \cdots, |E| + 1, |E| + 2\}$ of $|E| + 2$ agents, consisting of a corresponding agent for each edge in E and two additional agents labeled for simplicity as $|E| + 1$ and $|E| + 2$ . We assume a set of items partitioned into the sets $G = \{g_{1}, \cdots, g_{|E|+1}, g_{|E|+2}\}$ , that contains one item for each agent in N and the set $A = \{a_{1}, \cdots, a_{|V|}\}$ , that contains one item for each vertex of V. Finally, we assume the partial allocation $\gamma$ , where for each item $g_{i} \in G$ and for each agent $i \in N$ , $\gamma(g_{i}) = i$ , i.e., agent i's bundle in the partial allocation $\gamma$ is item $g_{i}$ , while the items in A are open and yet-to-be allocated.

The valuations for each agent $i \in N \setminus \{|E| + 1, |E| + 2\}$ are defined as:

• $v_{i}(g_{i}) = |V|$ .   
- $v_{i}(g_{|E| + 1}) = |V| - 1$ and $v_{i}(g_{|E| + 2}) = 0$ .   
- $v_{i}(a_{j}) = 1$ , if the vertex $j$ that corresponds to item $a_{j}$ is adjacent to the edge represented by agent $i$ , else 0.

The valuation for agent $|E| + 1$ , is defined as:

$$
\bullet v _ {| E | + 1} (g _ {| E | + 1}) = | V | - 2 \ell .
$$

- $v_{|E| + 1}(g) = 0, \forall g \in G \setminus \{g_{|E| + 1}\}$ .   
• $v_{|E|+1}(a_j) = 1, \forall a_j \in A.$

Finally, the valuation for agent $|E| + 2$ is defined as:

- $v_{|E| + 2}(g_{|E| + 2}) = |V|$ and $v_{|E| + 2}(g_{|E| + 1}) = 0$ .   
- $v_{|E| + 2}(g) = 2|V| - \ell$ , for all $g \in G \setminus \{g_{|E| + 1}, g_{|E| + 2}\}$ .   
• $v_{|E|+2}(a_j)=1,\forall a_j\in A.$

From $\mathcal{I}$ to $\mathcal{I}'$ . Assume that instance $\mathcal{I}$ contains an independent set of size $\ell$ . Let $\mathrm{IS} \subset A$ be the subset of open items $a_j \in A$ , that correspond to vertices that belong to an independent set of $\mathcal{I}$ . We will show that by allocating bundle IS to agent $|E| + 1$ and bundle $A \setminus \mathrm{IS}$ to agent $|E| + 2$ , we get an envy-free allocation extension for $\mathcal{I}'$ . Thus, in the complete allocation $\gamma' = \gamma \cup \pi$ , it holds that $\gamma'(g_{|E|+1} \cup \mathrm{IS}) = |E| + 1$ and $\gamma'(g_{|E|+2} \cup (\mathrm{A} \setminus \mathrm{IS})) = |E| + 2$ .

We will argue that in the complete allocation $\gamma'$ , no agent envies another agent. For agent $|E|+1$ it holds that $v_{|E|+1}(g_{|E|+1} \cup \mathrm{IS}) = v_{|E|+1}(g_{|E|+1}) + v_{|E|+1}(\mathrm{IS}) = |V| - 2\ell + \ell = |V| - \ell$ , which holds because we assume that valuations are additive, agent $|E| + 1$ values any item in IS equal to 1, and by the hypothesis each item in IS corresponds to some vertex in the independent set of size $\ell$ .

Agent $|E| + 1$ does not envy any agent in $N \setminus \{|E| + 2\}$ , since he is indifferent towards items in $G \setminus \{g_{|E| + 1}\}$ . Finally, the valuation of $|E| + 1$ for the bundle of agent $|E| + 2$ is $v_{|E| + 1}(g_{|E| + 2} \cup (\mathrm{A} \setminus \mathrm{IS})) = v_{|E| + 1}(g_{|E| + 2}) + v_{|E| + 1}(A \setminus \mathrm{IS}) = 0 + |V| - \ell = |V| - \ell$ . Consequently, $|E| + 1$ does not envy agent $|E| + 2$ either.

Agent's $|E| + 2$ valuation for his bundle is $v_{|E|+2}(g_{|E|+2} \cup (\mathbf{A} \setminus \mathbf{IS})) = v_{|E|+2}(g_{|E|+2}) + v_{|E|+2}(\mathbf{A} \setminus \mathbf{IS}) = |V| + |V| - \ell = 2|V| - \ell$ , since the valuations are additive and there exist $|V| - \ell$ items in the bundle $\mathbf{A} \setminus \mathbf{IS}$ , each valued 1. There exist no envy towards agent $|E| + 1$ since $v_{|E|+2}(g_{|E|+1} \cup \mathbf{IS}) = v_{|E|+2}(g_{|E|+1}) + v_{|E|+2}(\mathbf{IS}) = \ell < 2|V| - \ell$ . Finally agent $|E| + 2$ values any item in $G \setminus \{g_{|E|+1}, g_{|E|+2}\}$ by $2|V| - \ell$ , thus he does not envy any agent in $N \setminus \{|E| + 1, |E| + 2\}$ .

None of the agents $i \in N \setminus \{|E| + 1, |E| + 2\}$ , receive an open item from A, while they value their bundle by $v_i(g_i) = |V|$ . Notice that no two items in IS can correspond to adjacent vertices of $\mathcal{I}$ . This holds from the way IS is defined and the assumption that $\mathcal{I}$ contains an independent set of size $\ell$ . Consequently the valuation of any agent $i \in N \setminus \{|E| + 1, |E| + 2\}$ for IS is at most 1. Thus their valuation for the bundle of agent $|E| + 1$ is $v_i(g_{|E|+1} \cup \mathrm{IS}) = v_i(g_{|E|+1}) + v_i(\mathrm{IS}) \leq |V|$ , meaning that no envy exists towards agent $|E| + 1$ .

Finally no agent $i \in N \setminus \{|E| + 1, |E| + 2\}$ is interested in item $g_{|E| + 2}$ , while $i's$ value for the bundle of agent $|E| + 2$ in $\gamma'$ is $v_i(g_{|E| + 2} \cup (\mathbf{A} \setminus \mathbf{IS})) \leq 2 \leq |V|$ , meaning that no envy towards agent $|E| + 2$ exists.

The above arguments show that by allocating the bundle IS to agent $|E| + 1$ and the bundle A\IS to agent $|E| + 2$ , we get an envy-free allocation extension $\gamma'$ of the partial allocation $\gamma$ for the instance $T'$ .

From $\mathcal{I}'$ to $\mathcal{I}$ . Assume that there exists an envy-free allocation extension $\gamma'$ of $\gamma$ for $\mathcal{I}'$ . First, we will argue that in any such extension, the open items in A are allocated only to the agents $|E| + 1$ and $|E| + 2$ . In the partial allocation $\gamma$ , it holds that $v_{|E| + 2}(g_{|E| + 2}) = |V|$ , but agent $|E| + 2$ envies all agents $i \in N \setminus \{|E| + 1, |E| + 2\}$ , since $v_{|E| + 2}(g_i) = 2|V| - \ell$ . Consequently, since agent $|E| + 2$ values any open item by 1, in allocation $\pi$ he must receive at least $|V| - \ell$ open items. In turn agent $|E| + 1$ has a valuation of $v_{|E| + 1}(g_{|E| + 1}) = |V| - 2\ell$ and the only way not to envy agent $|E| + 2$ is to get exactly $\ell$ open items. Otherwise, since $|E| + 1$ values any open item by 1 and agent $|E| + 2$ must take at least $|V| - \ell$ open items if agent $|E| + 1$ is allocated less than $\ell$ open items, then the valuation he receives will be lower than $|V| - \ell$ which is the lower bound of his valuation for the bundle of agent $|E| + 2$ in any

envy-free extension. Consequently in $\pi$ , agents $|E| + 1$ and $|E| + 2$ must receive exactly $\ell$ and $|V| - \ell$ respectively open items from set A.

Let $A_{|E|+1}$ be the subset of $\ell$ items of A that are allocated to agent $|E| + 1$ and $A_{|E|+2}$ be the subset of $|V| - \ell$ items of A that are allocated to agent $|E| + 2$ in allocation $\pi$ . Notice that no agent $i \in N \setminus \{|E| + 1, |E| + 2\}$ envies agent $|E| + 2$ since $v_i(g_{|E|+2} \cup A_{|E|+2}) \leq 2 \leq |V| = v_i(g_i)$ . Moreover, since no agent can be envious in $\pi$ , that must mean that for each $i \in N \setminus \{|E| + 1, |E| + 2\}$ , $v_i(g_{|E|+1} \cup A_{|E|+1}) \leq v_i(g_i)$ which consequently means that $v_i(A_{|E|+1}) \leq 1$ . This can hold only if no two items in $A_{|E|+1}$ correspond to vertices that are adjacent to the same edge in $\mathcal{I}$ . Consequently, since $\gamma'$ is an envy-free allocation extension of the partial allocation $\gamma$ for $\mathcal{I}'$ , it must be the case that the items in the subset $A_{|E|+1}$ correspond to vertices of instance $\mathcal{I}$ that form an independent set of size $\ell$ .

We remark that Theorem 3 is tight in the sense that both problems are also in NP. Indeed, in both cases one can verify whether a provided complete allocation is envy-free in polynomial time. Having settled the intractability of REFAE and FEFAE w.r.t. p alone, we now ask whether one can achieve tractability at least in settings with a small number of agent or item types.

We begin by considering the former parameterization, which was the one that yielded fixed-parameter tractability for the setting investigated in the previous section. Here, we can immediately exclude fixed-parameter tractability for all of the problem variants considered in this section. This holds due to the aforementioned fact that finding an envy-free allocation of a set of open items to agents is known to be strongly W[1]-hard and weakly paraNP-hard when parameterized by the number of agents [BBN16]. Below, we at least show that both problems admit XP algorithms in the unary-valuation case, complementing the former lower bound.

Theorem 4. If the valuations are encoded in unary, both REFAE and FEFAE are in XP when parameterized by the number p of recipients plus the number $n_{t}$ of agent types.

Proof. Firstly, recall that for REFAE we know exactly which of the agents are the p recipients. However, for FEFAE, we first start by branching to determine which of the at most p agents will be the recipients, requiring a branching factor of at most $n^{p}$ .

Having the set of recipients as part of the input, we denote the set of recipients as $[p]$ . Let us fix some arbitrary ordering $1, \ldots, m$ in which we will assign items in $A$ (the set of open items). We now construct a table $T_t$ for $t \in [A]$ to store how much agents envy one another under different configurations. A configuration $x$ at step $t$ is a possible way of giving out the first $t$ items to the recipients. Each entry $t$ has a vector $c_t^x$ , the configuration $x$ of $t$ many items, for each configuration $x$ .

For each configuration $x$ , we store for each recipient $i \in [p]$ and each agent type $Z$ the valuation of the bundle assigned to $i$ from the perspective of agents of type $Z$ , denoted as $v_i^Z$ . Observe that this includes the valuation agent $i$ has for their bundle. It is sufficient to compute valuations from each agent type $Z$ against each of the $p$ recipient agents because all agents in $Z$ see the bundles of the recipient agents the same, and the bundles of the non-recipient agents won't change. We note that each table entry consists of $p \cdot n_t$ unary-encoded values.

We have just one vector for the starting table entry $T_{0}$ , as none of A has been assigned yet. Hence, this is exactly the valuations $V_{i}^{Z}$ for in the partial allocation $\gamma$ (which is EF). At $T_{1}$ we now start creating configurations of items and then computing the valuations from each agent type Z towards each recipient agent $i \in [p]$ . Note also that in $T_{1}$ we will have p many vectors; we are assigning the first item to each of the possible recipients. Additionally observe that we are not storing the configurations (how we allocate the items), just the valuations induced by the configuration. Hence, we can also discard duplicate entries.

Going forward, we compute each entry $T_{t+1}$ by giving out the next item, the $t + 1$ th item, to each of the recipients and updating the table accordingly. Crucially, each entry will have $n_{t} \cdot p$ many values (the number of valuations we compute), and so—since the valuations are encoded in unary—after assigning all open items, the table will be storing at most $|I|^{n_{t} \cdot p}$ many entries. This yields a total running time of $|I|^{\mathcal{O}(n_{t} \cdot p)} \cdot |I|$ .

At the end of this subroutine, we loop through each of the at most $|\mathcal{I}|^{\mathcal{O}(n_{t}\cdot p)}$ entries in the table, and notice that each such entry provides us with complete information about how each agent values each bundle in I in all possible assignments corresponding to that table entry. Hence, it suffices to check, in polynomial time, whether at least one such table entry results in an envy-free assignment in I. If there does not exist such an entry in the final table, we can conclude there is no solution because this means among all configurations of the open items no extension exists which is EF. □

Surprisingly, when dealing with a small number of recipients, we prove that it is the number $m_{t}$ of item types which yields better tractability results for the considered problems—a situation that is entirely opposite to that of the previous section. In particular, when parameterizing by $p + n_{t}$ we obtain a fixed-parameter algorithm for REFAE and an XP algorithm for FEFAE. We remark that the latter result is the best one could have at this point hoped for due to Theorem 2 ruling out fixed-parameter algorithms even in a strictly more restrictive setting.

Theorem 5. When parameterized by p plus the number $m_{t}$ of item types, REFAE and FEFAE are fixed-parameter tractable and XP-tractable, respectively.

Proof. For FEFAE, we start by branching on which agents will receive items; this requires at most $n^{p}$ branches. Let us fix some set of recipient agents [p]. We skip this first step for REFAE. The remainder of the proof is the same for both REFAE and FEFAE.

We now define an INTEGER LINEAR PROGRAMMING (ILP) instance such that the ILP instance is satisfiable if and only if there exists an envy-free complete allocation consistent with $\gamma$ .

We define a variable $x_{i}^{t}$ for each agent $i \in [p]$ and each item type $t \in T$ which denotes the number of items of type t that agent i receives. We first make sure that any satisfying assignment for ILP assigns the correct number of items of each type by including for each type t the equality (encoded as two inequalities)

$$
\sum_ {i \in [ p ]} x _ {i} ^ {t} = | t |, \tag {1}
$$

where $|t|$ denotes the number of items of type $|t|$ .

In addition, we require that no pair of agents that receive items envy each other after the extension. Formally, for all pairs of agents $i \in [p]$ and $j \in [p]$ we add the following inequality to the ILP instance

$$
v _ {i} (\pi_ {i}) + \sum_ {t \in T} x _ {i} ^ {t} \cdot v _ {i} (t) \geq v _ {i} (\pi_ {j}) + \sum_ {t \in T} v _ {i} (t) \cdot x _ {j} ^ {t}, \tag {2}
$$

This inequality says that agent $i$ is not envious of $j$ after the extension; i.e., $i$ getting $x_{i}^{t}$ many of type $t$ and $j$ getting $x_{j}^{t}$ many does not create envy from $i$ to $j$ .

We also create an inequality which handles agents not receiving any items (agents not in $p$ ) to ensure they do not become envious. In particular, for $j \notin [p]$ we construct the following inequality:

$$
v _ {i} (\pi_ {j}) \geq v _ {i} (\pi_ {i}) + \sum_ {t \in T} x _ {i} ^ {t} \cdot v _ {i} (t). \tag {3}
$$

Observe, the number of variables is upper-bounded by $p \cdot m_{t}$ . Hence, we can solve the resulting instance in time $(p \cdot m_{t})^{\mathcal{O}(p \cdot m_{t})} \cdot |\mathcal{I}|^{\mathcal{O}(1)}$ [Len83, Dad12]. Thus, this is FPT-time for REFAE and XP-time for FEFAE as we have $\mathcal{O}(n^{p})$ different ILP instances to check.

To complete the proof, it remains to argue that we have an envy-free solution if and only if we have a solution to the constructed ILP.

Assume we have a solution for the ILP, this solution describes to us what agents will receive which bundles. More precisely, for each agent $i \in [p]$ and item type $t \in T$ we assign arbitrary $x_{i}^{t}$ many items of type $t$ to agent $i$ . First note that Equality (1) ensures that such an assignment is possible and all items

have been assigned to some agent. Assume that this solution is not an envy-free extension to $\gamma$ . This means that there must exist a pair of agents i and j such that i envies j. If i is a recipient agent and j is also a recipient, this would mean that Inequality (2) does not hold for the pair i, j. On the other hand, if j was not a recipient, it is not possible i envies j because $\gamma$ , the partial allocation, is envy-free, all items have non-negative values for all agents, and j has not received anything in the extension. If agent i was not a recipient and j is a recipient, this means Inequality (3) would not hold for this pair. Thus, the solution for the ILP must be an envy-free solution.

Now, given an solution to ENVY-FREE ALLOCATION EXTENSION we will argue that this is also gives us a solution to the ILP. To obtain the solution for the ILP, for every agent $i \in [p]$ and every item type $t \in T$ , we let $x_{i}^{t}$ be the number of items of type t the agent i received. Equality (1) holds for each item type $t \in T$ , because all items of the type have been assigned. Inequalities (2) and (3) directly describe that the agent i does not envy the agent j for the particular pair of agents i, j, and so all of these inequalities have to hold as well. This concludes the proof.

# 5 Extending Allocations Beyond Envy-Freeness

While it seems that the algorithmic upper and lower bounds presented in the main contribution of our article could be translated also to the analogous problems of extending EF1 and EFX allocations, in this section we turn towards a different, highly studied aspect of such allocations—specifically, the question of whether such allocations are guaranteed to exist [Bud11, CKM $^{+}$ 19]. We show that in the extension setting, this question can be completely resolved based on the level of fairness we assume in the partial allocation.

First of all, it is obvious that if no fairness guarantees are provided for the partial allocation, then one cannot hope to guarantee an extension to an EF1 or EFX allocation (since, e.g., there might not be any open items left at all). On the other hand, if we assume the partial allocation to be envy-free, we obtain the following.

Proposition 1. Every envy-free partial allocation of items to agents can be extended to an EF1 allocation. At the same time, there exists an envy-free partial allocation of items to 2 agents which cannot be extended to an EFX allocation.

Proof. The first part of the claim follows almost immediately from the fact that the utilities are additive. Let $\gamma$ be an envy-free allocation. This means that $v_{i}(\gamma_{i}) \geq v_{i}(\gamma_{j})$ for every pair of agents i and j. We compute a partial allocation $\pi$ that is EF1 employing a standard algorithm, like Envy Cycle Elimination [LMMS04], or Round Robin. Then, since $\pi$ is EF1 it holds that $v_{i}(\pi_{i}) \geq v_{i}(\pi_{j}) - \max_{a \in \pi_{j}} v_{i}(a)$ , for every pair of agents i and j. Thus, if we combine all the above we get that $v_{i}(\gamma_{i} \cup \pi_{i}) = v_{i}(\gamma_{i}) + v_{i}(\pi_{i}) \geq v_{i}(\gamma_{j}) + v_{i}(\pi_{j}) - \max_{a \in \pi_{j}} v_{i}(a)$ , for every pair of agents i and j. Thus, the extended allocation $\gamma \cup \pi$ is EF1.

For the second part of the claim consider two agents with identical valuations. Under $\gamma$ , each agent gets an item that he values 1 and there is one open item that each agent values 2. No matter which agent agent receives the open item, the other agent will still envy him after removing the item of value 1. Hence, both possible extended allocations do not satisfy the conditions of EFX. ☐

Interestingly, the latter non-existence result contrasts with the known fact that EFX allocations from scratch always exist for 2 agents [PR20].

Moreover, rather than starting with an envy-free partial allocation, we can also consider some other “reasonable” partial allocations. For example, if we start with a partial allocation which is EF1 and the envy-free graph is acyclic [LMMS04], there is a polynomial-time algorithm to find an EF1 complete allocation [PIV24]. On another note, it is, in fact, NP-complete to find a complete allocation which is EF1 and PO, by starting from an EF1 partial allocation even when agents have identical valuations [PIV24].

The final remaining question is whether a partial EFX allocation extends to a complete EF1 allocation. We resolve this in the negative below.

Proposition 2. There exists an EFX partial allocation of items to 2 agents which cannot be extended to an EF1 allocation.

Proof. Consider the following instance with two agents and three items x, y, z. The valuations of the agents are as follows: $v_{1}(x) = 10$ , $v_{1}(y) = 0$ , $v_{1}(z) = 1$ ; $v_{2}(x) = 0$ , $v_{2}(y) = 10$ , $v_{2}(z) = 1$ . Consider now the partial allocation that gives x to agent 2 and y to agent 1. Trivially, this is an EFX allocation since envy can be eliminated by removing the only item the other agent receives. Now, no matter which agent receives item z the constructed solution will not be EF1. Indeed, assume that agent 1 gets z. Observe that the value of agent 2 for $\{y, z\}$ is $10 + 1$ , while for their bundle is 0. Hence after the removal of any item from the bundle of agent 1, agent 2 will still have a positive value for it and thus the envy is not eliminated. □

# 6 Concluding Remarks

Our paper initiates the study of fairly extending partial allocations of indivisible items and the frontiers of tractability for this problem. The presented results showcase that the complexity of this task varies as it strongly depends on the chosen parameters—with the number of agent types and item types each leading to tractability in different settings. This naturally gives rise to the question of what other parameterizations can yield fixed-parameter algorithms for this natural but previously overlooked problem. For instance, can one obtain such algorithms when simultaneously parameterizing by the number $m_{t}$ of item types and the number $n_{t}$ of agent types? If this combination of parameters is not sufficient for tractability, can the inclusion of the number of recipients, p, in addition to $n_{t}$ and $m_{t}$ lead to tractability?

A different direction that has proven fruitful recently, is to consider the problem when agents form a social network [BKN22, EGHO23]. In this setting, each agent compares their bundle against the bundles of their “friends”. What are the graph structures on the social network that make the problem tractable?

# Acknowledgments

Argyrios Deligkas and Stavros D. Ioannidis acknowledge the support of the EPSRC grant EP/X039862/1. Robert Ganian acknowledges support from the Austrian Science Fund (FWF, projects 10.55776/Y1329 and 10.55776/COE12) and the Vienna Science Foundation (WWTF, project 10.47379/ICT22029).

# References

[ABF $^{+}$ 15] Patrizio Angelini, Giuseppe Di Battista, Fabrizio Frati, Vít Jelínek, Jan Kratochvíl, Maurizio Patrignani, and Ignaz Rutter. Testing planarity of partially embedded graphs. ACM Trans. Algorithms, 11(4):32:1–32:42, 2015.   
[ADP19] Alan Arroyo, Martin Derka, and Irene Parada. Extending simple drawings. In Daniel Archambault and Csaba D. Tóth, editors, Graph Drawing and Network Visualization - 27th International Symposium, GD 2019, Prague, Czech Republic, September 17-20, 2019, Proceedings, volume 11904 of Lecture Notes in Computer Science, pages 230–243. Springer, 2019.   
[AMN20] Georgios Amanatidis, Evangelos Markakis, and Apostolos Ntokos. Multiple birds with one stone: Beating 1/2 for EFX and GMMS via envy cycle elimination. Theoretical Computer Science, 841:94–109, 2020.

[BBN16] Bernhard Bliem, Robert Bredereck, and Rolf Niedermeier. Complexity of efficient and envy-free resource allocation: Few agents, resources, or utility levels. In Subbarao Kambhampati, editor, Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI 2016, New York, NY, USA, 9-15 July 2016, pages 102–108. IJCAI/AAAI Press, 2016.   
[BGK $^{+}$ 23] Sujoy Bhore, Robert Ganian, Liana Khazaliya, Fabrizio Montecchiani, and Martin Nöllenburg. Extending orthogonal planar graph drawings is fixed-parameter tractable. In Erin W. Chambers and Joachim Gudmundsson, editors, 39th International Symposium on Computational Geometry, SoCG 2023, June 12-15, 2023, Dallas, Texas, USA, volume 258 of LIPIcs, pages 18:1–18:16. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2023.   
[BKN22] Robert Bredereck, Andrzej Kaczmarczyk, and Rolf Niedermeier. Envy-free allocations respecting social networks. Artif. Intell., 305:103664, 2022.   
[BL08] Sylvain Bouveret and Jérôme Lang. Efficiency and envy-freeness in fair division of indivisible goods: Logical representation and complexity. J. Artif. Intell. Res., 32:525–564, 2008.   
[BLM16] Simina Brânzei, Yuezhou Lv, and Ruta Mehta. To give or not to give: Fair division for single minded valuations. In Subbarao Kambhampati, editor, Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI 2016, New York, NY, USA, 9-15 July 2016, pages 123–129. IJCAI/AAAI Press, 2016.   
[Bud11] Eric Budish. The combinatorial assignment problem: Approximate competitive equilibrium from equal incomes. Journal of Political Economy, 119(6):1061–1103, 2011.   
[CFK $^{+}$ 15] Marek Cygan, Fedor V. Fomin, Lukasz Kowalik, Daniel Lokshtanov, Dániel Marx, Marcin Pilipczuk, Michal Pilipczuk, and Saket Saurabh. Parameterized Algorithms. Springer, 2015.   
[CKM $^{+}$ 19] Ioannis Caragiannis, David Kurokawa, Hervé Moulin, Ariel D. Procaccia, Nisarg Shah, and Junxing Wang. The unreasonable fairness of maximum nash welfare. ACM Trans. Economics and Comput., 7(3):12:1–12:32, 2019.   
[Dad12] Daniel Dadush. Integer Programming, Lattice Algorithms, and Deterministic Volume Estimation. PhD thesis, Georgia Institute of Technology, 2012.   
[DEG $^{+}$ 21] Argyrios Deligkas, Eduard Eiben, Robert Ganian, Thekla Hamm, and Sebastian Ordyniak. The parameterized complexity of connected fair division. In Zhi-Hua Zhou, editor, Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27 August 2021, pages 139–145. ijcai.org, 2021.   
[DF13] Rodney G. Downey and Michael R. Fellows. Fundamentals of Parameterized Complexity. Texts in Computer Science. Springer, 2013.   
[EGH $^{+}$ 20] Eduard Eiben, Robert Ganian, Thekla Hamm, Fabian Klute, and Martin Nöllenburg. Extending partial 1-planar drawings. In Artur Czumaj, Anuj Dawar, and Emanuela Merelli, editors, 47th International Colloquium on Automata, Languages, and Programming, ICALP 2020, July 8-11, 2020, Saarbrücken, Germany (Virtual Conference), volume 168 of LIPIcs, pages 43:1–43:19. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2020.   
[EGHO23] Eduard Eiben, Robert Ganian, Thekla Hamm, and Sebastian Ordyniak. Parameterized complexity of envy-free resource allocation in social networks. Artif. Intell., 315:103826, 2023.

[EGK $^{+}$ 23] Eduard Eiben, Robert Ganian, Iyad Kanj, Sebastian Ordyniak, and Stefan Szeider. On the parameterized complexity of clustering problems for incomplete data. J. Comput. Syst. Sci., 134:1–19, 2023.   
[FMP24] Michal Feldman, Simon Mauras, and Tomasz Ponitka. On optimal tradeoffs between EFX and nash welfare. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 9688–9695. AAAI Press, 2024.   
[GKOS18] Robert Ganian, Iyad A. Kanj, Sebastian Ordyniak, and Stefan Szeider. Parameterized algorithms for the matrix completion problem. In Jennifer G. Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 1642–1651. PMLR, 2018.   
[GOR19] Robert Ganian, Sebastian Ordyniak, and C. S. Rahul. Group activity selection with few agent types. In Michael A. Bender, Ola Svensson, and Grzegorz Herman, editors, 27th Annual European Symposium on Algorithms, ESA 2019, September 9-11, 2019, Munich/Garching, Germany, volume 144 of LIPIcs, pages 48:1–48:16. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2019.   
[IKSY24] Ayumi Igarashi, Naoyuki Kamiyama, Warut Suksompong, and Sheung Man Yuen. Reachability of fair allocations via sequential exchanges. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, pages 9773–9780. AAAI Press, 2024.   
[KFN21] Tomohiro Koana, Vincent Froese, and Rolf Niedermeier. Binary matrix completion under diameter constraints. In Markus Bläser and Benjamin Monmege, editors, 38th International Symposium on Theoretical Aspects of Computer Science, STACS 2021, March 16-19, 2021, Saarbrücken, Germany (Virtual Conference), volume 187 of LIPIcs, pages 47:1–47:14. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2021.   
[KFN23] Tomohiro Koana, Vincent Froese, and Rolf Niedermeier. The complexity of binary matrix completion under diameter constraints. J. Comput. Syst. Sci., 132:45–67, 2023.   
[Len83] Hendrik W. Lenstra, Jr. Integer programming with a fixed number of variables. Mathematics of Operations Research, 8(4):538–548, 1983.   
[LMMS04] Richard J Lipton, Evangelos Markakis, Elchanan Mossel, and Amin Saberi. On approximately fair allocations of indivisible goods. In Proceedings of the 5th ACM Conference on Electronic Commerce, pages 125–131, 2004.   
[NR23] Trung Thanh Nguyen and Jörg Rothe. Complexity results and exact algorithms for fair division of indivisible items: A survey. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China, pages 6732–6740. ijcai.org, 2023.   
[PIV24] Vishwa Prakash, Ayumi Igarashi, and Rohit Vaish. Fair and efficient completion of indivisible goods. CoRR, abs/2406.09468, 2024.

[PR20] Benjamin Plaut and Tim Roughgarden. Almost envy-freeness with general valuations. SIAM Journal on Discrete Mathematics, 34(2):1039–1068, 2020.