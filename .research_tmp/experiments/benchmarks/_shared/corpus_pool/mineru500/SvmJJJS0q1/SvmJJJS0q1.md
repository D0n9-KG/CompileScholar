# Detecting and Measuring Confounding Using Causal Mechanism Shifts

Abbavaram Gowtham Reddy and Vineeth N Balasubramanian

Indian Institute of Technology Hyderabad, India

{cs19resch11002,vineethnb}@iith.ac.in

# Abstract

Detecting and measuring confounding effects from data is a key challenge in causal inference. Existing methods frequently assume causal sufficiency, disregarding the presence of unobserved confounding variables. Causal sufficiency is both unrealistic and empirically untestable. Additionally, existing methods make strong parametric assumptions about the underlying causal generative process to guarantee the identifiability of confounding variables. Relaxing the causal sufficiency and parametric assumptions and leveraging recent advancements in causal discovery and confounding analysis with non-i.i.d. data, we propose a comprehensive approach for detecting and measuring confounding. We consider various definitions of confounding and introduce tailored methodologies to achieve three objectives: (i) detecting and measuring confounding among a set of variables, (ii) separating observed and unobserved confounding effects, and (iii) understanding the relative strengths of confounding bias between different sets of variables. We present useful properties of a confounding measure and present measures that satisfy those properties. Empirical results support the theoretical analysis.

# 1 Introduction

Understanding the underlying causal generative process of a set of variables is crucial in many scientific studies for applications in treatment and policy designs $[44]$ . While randomized controlled trials (RCTs) and causal inference through active interventions are ideal choices for understanding the underlying causal model $[19, 12, 13, 55]$ , RCTs and/or active interventions are often impossible/infeasible, and some times unethical $[50, 6]$ . Research efforts in causal inference hence rely on observational data to study causal relationships $[44, 59, 65, 18, 41]$ . However, recovering the underlying causal model purely from observational data is challenging without further assumptions; this challenge is further exacerbated in the presence of unmeasured confounding variables.

A confounding variable is a variable that causes two other variables, resulting in a spurious association between those two variables. As exemplified with Simpson's paradox [58] and many other studies [20, 1, 31], the presence of confounding variables is an important quantitative explanation for why correlation does not imply causation. It is challenging to observe and measure all confounding variables in a scientific study [60, 44]. Identifying latent or unobserved confounding variables is even more challenging, and misinterpretation presents various challenges in downstream applications, such as discovering causal structures from observational data. Numerous methods operate under the assumption of causal sufficiency [45, 4, 60, 8, 51, 65], implying the non-existence of unobserved confounding variables. Causal sufficiency presupposes that all pertinent variables required for causal inference have been observed. However, this may not be a practical or testable assumption.

The study of confounding has various applications, chief among them being causal discovery - identifying the causal relationships among variables [38, 40, 63]. It is also useful for determining whether a set of observed confounding variables is sufficient to adjust for estimating causal effects [29], measuring the extent to which statistical correlation between variables can be attributed to confound-

ing [24, 25, 62], and verifying the comparability of treatment and control groups in non-randomized interventional studies [16].

A fundamental problem in causal inference tasks lies in detecting hidden confounding variables from observational data alone. However, this is non-trivial and poses various challenges. For example, a key issue is that given a marginal distribution over observed variables, there are infinitely many joint distributions corresponding to causal graphs involving unobserved variables $[56]$ . To tackle such challenges, recent endeavors show that using data from different environments helps in improved causal discovery $[40, 38, 33, 45, 23]$ , detecting causal mechanism shifts $[36]$ , and detecting unobserved confounding $[29, 38]$ . However, such recent efforts often subsume confounding detection under causal discovery, focusing primarily on identifying confounding factors while overlooking other useful information, such as the relative strength of confounding between variable sets and the distinction between observed and unobserved confounding within a variable set. We seek to address these gaps in this work.

We focus exclusively on the problem of studying confounding from multiple perspectives, including (i) detecting and measuring confounding among a set of variables, (ii) assessing the relative strengths of confounding among different sets of observed variables, and (iii) distinguishing between observed and unobserved confounding among a set of variables. The primary focus of causal inference often lies in verifying the presence or absence of confounding rather than determining the exact value of the measured confounding. However, we leverage the measured confounding to assess the relative strengths of confounding between sets of variables. To achieve the above objectives, we utilize data from various contexts, where each context results from shifts in the causal mechanisms of a set of variables $[38, 45]$ . This allows us to propose different measures of confounding based on the available context information. Our contributions can be summarized as follows.

- For various definitions of confounding, we propose corresponding measures of confounding and present useful properties of the proposed measures. To our knowledge, this is the first comprehensive study that examines various aspects of observed and unobserved confounding using data from multiple contexts without making parametric or causal sufficiency assumptions.   
- We study pair-wise confounding, confounding among multiple variables, how to separate unobserved confounding from overall confounding, and present ways to assess relative confounding.   
- We present an algorithm for detecting and measuring confounding using data from multiple contexts. Experimental results are performed to verify theoretical analysis.

# 2 Related Work

The study of confounding has typically been embedded as part of causal discovery algorithms in most existing work. Causal discovery methods can be categorized according to several criteria, including the type of data utilized (observational versus interventional/experimental), parametric versus non-parametric approaches, or whether they relax causal sufficiency assumptions $[65, 59]$ . Considering our focus in this work on studying confounding comprehensively by going beyond observed confounding variables, we discuss literature that are directed towards methods that relax the causal sufficiency assumption and rely on experimental data.

Causal Discovery via Observational Data, Relaxing Causal Sufficiency: Constraint based causal discovery algorithms produce equivalence class of graphs that satisfy a set of conditional independence constraints $[60, 11, 9, 42]$ . Other methods such as $[2, 28, 27]$ reduce the problem complexity by assuming a parametric form of the underlying causal model (e.g., variables are jointly Gaussian in Chandrasekaran et al. $[7]$ ), thereby returning unique causal graphs. Nested Markov Models (NMMs) $[56, 57, 49, 14]$ allow identifiability of causal models with latent factors by using (pairwise) Verma constraints. A recent approach using differentiable causal discovery $[2]$ combines NMMs with the differentiable constraint $[66]$ to discover a partially directed causal network and likely confounded nodes. Unlike these methods, our focus in this work is on detecting and measuring confounding under various settings, instead of recovering the entire causal graph or equivalence class.

Causal Discovery Using Data From Multiple Environments: Given access to a set of observed confounding variables, very recent work $[29]$ presented testable conditional independence tests that are violated only when there is unobserved confounding. However, their analysis is focused towards the downstream causal effect estimation. We aim to provide a unified framework for studying and measuring confounding under different types of contextual information available.

Other methods $[33, 23]$ learn an equivalence class of graphs when data from observational and interventional distribution are available. Confounding has also shown to be detected in linear models with non-Gaussian variables $[20]$ . In linear models, a spectral analysis method was proposed in $[25]$ to understand to what extent the statistical correlation between a set of variables on a target variable can be attributed to confounding. See Tab. 4 of $[40]$ for an overview of causal discovery methods that use data from multiple environments or contexts. Under the specific assumptions of causal sufficiency and sparse mechanism shift, a method was proposed in $[45]$ to reduce the size of a given Markov equivalence class using mechanism shift score. A differentiable causal discovery method was proposed in $[4]$ to use interventional data to recover interventional Markov equivalence class. While these methods use data from different contexts, they assume the absence of unobserved confounding variables; we instead focus on capturing both observed and unobserved confounding.

Measuring and Interpreting Confounding: Earlier efforts in the field have studied different measures for observed confounding, each tailored to address specific challenges $[44, 15, 35, 3, 39, 30, 43, 34]$ . Such measures have also been refined to address specific issues $[24, 54]$ ; for e.g., a method to correct the non-linearity effect present in confounding estimates via the exposure–outcome association with and without adjustment for confounding was proposed in $[24]$ . In contrast, we measure the effects of both observed and unobserved confounding. Motivated from the ignorability property in potential outcomes framework $[61, 26]$ , the divergence between nominal and complete propensity density has been considered as an indicative of hidden confounding $[26]$ . To the best of our knowledge, the efforts closest to ours are $[38, 40]$ , which study confounding using data from multiple contexts without the causal sufficiency assumption. However, they do not measure confounding and detect confounding only as a step to discover the causal graph. Ours is a more general framework for studying and measuring confounding from multiple perspectives.

In regression models, certain difference threshold between the coefficients of treatment variable before and after adjusting for the possible confounding is considered as the indication for the presence of confounding. This process of choosing a threshold is also called change-in-estimate criterion. Typical threshold used in literature is 10% [54, 32, 5].

# 3 Background and Problem Setup

Let X be a set of observed variables and Z be a set of unobserved or latent variables. The values of X, Z can be real, discrete, or mixed. Let G be the underlying directed acyclic graph (DAG) among the variables $V = X \cup Z$ . Directed edges among the variables in V indicate direct causal influences. Assume that the set of unobserved variables Z are jointly independent and are exogenous to X (i.e., $Z_{i} \perp Z_{j}$ and $X_{k} \not\to Z_{j} \forall i, j, k$ ). In this setting, any two nodes $X_{i}, X_{j} \in X$ sharing a common parent $Z_{k} \in Z$ are said to be confounded, and $Z_{k}$ is said to be a confounding variable. For a node $X_{i} \in X$ , $PA_{i} = \{X_{j} \in X | X_{j} \to X_{i}\} \cup \{Z_{j} \in Z | Z_{j} \to X_{i}\}$ denotes the set of parents of $X_{i}$ .

For a node $X_{i}$ , $\mathbb{P}(X_i|\mathbf{PA}_i)$ is called the causal mechanism of $X_{i}$ . The causal mechanism $\mathbb{P}(X_i|\mathbf{PA}_i)$ encodes how the variable $X_{i}$ is influenced by its parents $\mathbf{PA}_i$ . Following earlier work [22, 38, 45, 21, 46, 52], we make the following general assumption about the underlying causal mechanisms of data.

Assumption 3.1. (Independent Causal Mechanisms [44, 47]) A change in $\mathbb{P}(X_i|\mathbf{PA}_i)$ has no effect on and provides no information about $\mathbb{P}(X_j|\mathbf{PA}_j)\forall j\neq i$ .

Identifying confounding from only observational data is challenging without further assumptions $[28]$ . Hence, following earlier work $[38, 29, 40]$ , we assume that the data over the variables X is observed over multiple contexts or environments. While there are various ways of formulating/constructing contexts, in this paper, we assume that each context is created as a result of either hard (a.k.a. structural) interventions or soft (a.k.a. parametric) interventions on a subset $V_{S} \subseteq V$ of variables where S is a set of indices. Performing hard intervention on a variable $V_{i}$ is the same as setting the value of $V_{i}$ to a value $v_{i}$ . Hard intervention on a variable $V_{i}$ removes the influence of its parents $PA_{i}$ on $V_{i}$ . Performing soft intervention on a variable $V_{i}$ is the same as changing the causal mechanism of $V_{i}$ , $\mathbb{P}(V_{i}|\mathbf{PA}_{i})$ , with a new causal mechanism $\tilde{\mathbb{P}}(V_{i}|\mathbf{PA}_{i})$ . Soft intervention on a variable $V_{i}$ does not remove the influence of its parents $PA_{i}$ on $V_{i}$ . The idea of explicitly considering context information and using different contexts as context variables to create extended causal graphs has been studied in the literature. Context variables are also called as policy variables, decision variables, regime variables, domain variables, environment variables, etc. $[40, 45, 17, 22]$ .

Let $\mathbf{C} = \{c_1, c_2, \ldots, c_n\}$ be the set of $n$ contexts and let $\mathbb{P}^c(\mathbf{X}), c \in \mathbf{C}$ , denotes the probability distribution of the observed variables $\mathbf{X}$ in the context $c$ . Let $\mathbf{C}_{S \wedge R}$ , where $S, R$ are sets of indices, be

the set of contexts in which we observe mechanism changes for the set of variables $X_{S\cup R}$ . Similarly, let $C_{S\wedge\neg R}$ be the set of contexts in which we observe mechanism changes for the set of variables $X_{S}$ but not for the variables $X_{R}$ . We say that the causal mechanism of a variable $X_{i}$ changes between two contexts c, $c'$ if $\mathbb{P}^{c}(X_{i}|\mathbf{PA}_{i}) \neq \mathbb{P}^{c'}(X_{i}|\mathbf{PA}_{i})$ . Given the data over observed variables in each context, there exist methods for detecting mechanism shifts of each variable between the contexts [36, 38, 45, 37]. For example, the p-value $(\mathbb{P}^{c}(X_{i}|\mathbf{PA}_{i}^{o}) \neq \mathbb{P}^{c'}(X_{i}|\mathbf{PA}_{i}^{o}))$ where $PA_{i}^{o}$ is the set of observed parents of $X_{i}$ can be used to detect mechanism change for $X_{i}$ between the contexts c, $c'$ [38, 36]. Hence, we focus on detecting and measuring confounding among a set of variables, assuming that the causal mechanism shifts are observed among that set of variables.

Context information is not very useful if there is no restriction on how causal mechanisms are changed between the contexts $[45, 38]$ . For example, the causal mechanisms of $X_{i}$ and $X_{j}$ both differing across all (or no) contexts would trivially satisfy Assumption 3.1, but reveal no information about the underlying causal mechanisms $[10, 38]$ . Hence, following earlier work $[45, 38, 17]$ , we make the following assumptions.

Assumption 3.2. (Sparse Causal Mechanism Shift [53]) Causal mechanisms of variables change sparsely across contexts, i.e., if $p := (\mathbb{P}^c(X_i|\mathbf{PA}_i) \neq \mathbb{P}^{c'}(X_i|\mathbf{PA}_i))$ , then $0 < p < 0.5$ ; $\forall c, c' \in \mathbf{C}$ .

Assumption 3.2 implies that the causal mechanisms change infrequently across contexts. This assumption is more general because, in many scientific studies, for any given context, interventions typically affect only a few variables [53].

Assumption 3.3. (Markov Property under Mechanism Shifts [17]) The distribution $\mathbb{P}(\mathbf{V})$ is given by $\mathbb{P}(\mathbf{V}) = \int \mathbb{P}^C (\mathbf{V})d\mathbb{P}(C) = \int \Pi_i\mathbb{P}^C (V_i|\mathbf{PA}_i)d\mathbb{P}(C)$ . In other words, variables $\mathbf{V}$ are assumed to be conditionally exchangeable, so that the same graph $\mathcal{G}$ applies in every context $c\in \mathbf{C}$ .

Assumption 3.4. (Causal Sufficiency Over $X \cup Z$ ) All common parents of any pair of observed nodes belong to the set $X \cup Z$ . In other words, all relevant variables for detecting confounding and the unobserved confounding variables are already present in $X \cup Z$ .

Problem Statement: Given data over the observed variables X in multiple contexts, each context resulting from a sparse causal mechanism shift of variables in V, (i) can we identify which pairs or sets of variables are confounded and can we measure the confounding strength? (ii) can we isolate the confounding effects of observed and unobserved confounding variables? and (iii) can we study the relative strengths of confounding among different sets of variables?

To address the above problem, in the next section, we consider various definitions of confounding and present appropriate confounding measures depending on the context information available.

# 4 Detecting and Measuring Confounding

In this section, we present methods for detecting and measuring confounding for various scenarios in which shifts in causal mechanisms are observed. Considering any three observed variables $X_{i}, X_{j}, X_{o} \in X$ and an unobserved confounding variable $Z \in Z$ , we present measures of confounding depending on the information about mechanism shifts of $X_{i}, X_{j}, X_{o}, Z$ . Each of the following subsections includes: (i) a definition of confounding, (ii) a corresponding definition of the confounding measure, (iii) a method for isolating the unobserved confounding measure from the overall confounding, (iv) an extension of the confounding measure to more than two variables, and (v) key properties of the proposed confounding measures. See Tab. 1 and Fig. 1 for an overview.

<table><tr><td>Settings</td><td>Confounding Definition Based On</td><td>Required Context Information</td><td>Type of Intervention</td></tr><tr><td>1</td><td>Directed Information [48] &amp; Noncollapsibility [15, 43, 54]</td><td> $\mathbf{C}_{\{i\} \land \neg P_{ij}}$  $\mathbf{C}_{\{j\} \land \neg P_{ji}}$ </td><td>Hard / Structural</td></tr><tr><td>2 &amp; 3</td><td>Mutual Information</td><td> $\mathbf{C}_{\{i\} \land \{j\}}$ </td><td>Soft / Parametric</td></tr></table>

Table 1: Summary of the various settings for detecting and measuring confounding between $X_{i}, X_{j}$ . Here $P_{ij}$ is the set of node indices that belong to a path from $X_{i}$ to $X_{j}$ including j.

# 4.1 Setting 1: Measuring Confounding Using Directed Information Between $X_{i}, X_{j}$ .

In this setting, we use the fact that directed information does not vanish in the presence of a confounding variable $[64, 48]$ . To this end, we leverage the interventional effects of $X_{i}, X_{j}$ on each other to define a measure of confounding.

![](images/d54711a8a97d052146455b824b5abc735f761877fb598185eb0f89e01102ef27.jpg)  
Setting 1

![](images/94b672d5d81202e565892d3580656f276b994dc0850ad598d698f014bfd8ef4f.jpg)  
Setting 2

![](images/93fb92ae37a4573deb3bb753929fc75cbc3fdd5a4d090f1a007267c412fbd37c.jpg)

Setting 3   
![](images/3c0dd017d68c27784ad47e9c6b32df911eb89fd27d1f80df091f2c72ae58b2eb.jpg)  
Figure 1: Setting 1: When contexts $\mathbf{C}_{\{i\} \wedge \neg P_{ij}}$ and $\mathbf{C}_{\{j\} \wedge \neg P_{ji}}$ are known where $P_{ij}$ is the set of node indices that belong to a path from $X_i$ to $X_j$ including $J$ , we leverage directed information from $X_i$ to $X_j$ and from $X_j$ to $X_i$ to define a measure of confounding (Defn. 4.4). Setting 2: Causal mechanism changes in $Z$ introduces dependencies on the observed distributions of $X_i, X_j$ . We leverage such dependencies to measure confounding when contexts $\mathbf{C}_{\{i\} \wedge \{j\}}$ are known (Defn. 4.6). Setting 3: If we know that there is a causal path from $X_i$ to $X_j$ , we leverage dependencies between the pairs $(X_i, X_j)$ and $(Z, X_j)$ to measure confounding. Similarly, if we know that there is a causal path from $X_j$ to $X_i$ , we leverage dependencies between the pairs $(X_i, X_j)$ and $(Z, X_i)$ to measure confounding (Defn. 4.7). Dashed arrows from $Z$ indicate that $Z$ is unobserved.

Definition 4.1. (Directed Information [48]). The directed information $I(X_i \to X_j)$ from $X_i \in \mathbf{X}$ to $X_j \in \mathbf{X}$ is defined as the conditional Kullback-Leibler divergence between the distributions $\mathbb{P}(X_i|X_j), \mathbb{P}(X_i|do(X_j))$ . That is:

$$
I (X _ {i} \rightarrow X _ {j}) := D _ {K L} (\mathbb {P} (X _ {i} | X _ {j}) | | \mathbb {P} (X _ {i} | d o (X _ {j}))) := \mathbb {E} _ {\mathbb {P} (X _ {i}, X _ {j})} \log \frac {\mathbb {P} (X _ {i} | X _ {j})}{\mathbb {P} (X _ {i} | d o (X _ {j}))} \tag {1}
$$

Definition 4.2. (No Confounding [44]) When measuring the causal effect of a (treatment) variable $X_{i}$ on a (target) variable $X_{j}$ , the ordered pair $(X_{i}, X_{j})$ is unconfounded if and only if the directed information from $X_{j}$ to $X_{i}$ : $I(X_{j} \to X_{i})$ is zero. Equivalently, $\mathbb{P}(X_{j}|X_{i}) = \mathbb{P}(X_{j}|do(X_{i}))$ .

A similar definition of confounding that relates the conditional distribution $\mathbb{P}(X_{i}|X_{j})$ and interventional distribution $\mathbb{P}(X_{i}|do(X_{j}))$ is defined as follows.

Definition 4.3. (Noncollapsibility) [15, 43, 54] The statistical association between two variables $X_{i}$ and $X_{j}$ is said to be noncollapsible if the association strength differs in each level/strata of other variable $X_{k}$ . That is, if $X_{k}$ is a confounding variable between $X_{i}, X_{j}$ , we have $\mathbb{P}(X_{j}|X_{i}) \neq \mathbb{P}(X_{j}|do(X_{i})) = \mathbb{E}_{X_{k}}(\mathbb{P}(X_{j}|X_{i}, X_{k}))$ .

From Defns. 4.1 and 4.2, for a pair of variables $(X_{i}, X_{j})$ , observing $I(X_{j} \to X_{i}) > 0$ and $I(X_{i} \to X_{j}) > 0$ implies that $\mathbb{P}(X_{j}|do(X_{i})) \neq \mathbb{P}(X_{j}|X_{i})$ and hence the presence of confounding (see Tab. 2). Using the above properties of directed information, we measure confounding as follows.

Definition 4.4. (Confounding Measure 1) When causal mechanism shifts of two variables $X_{i}, X_{j} \in X$ are observed, resulting in different contexts, under the Assumptions 3.2-3.4, the measure of confounding $CNF-1(X_{i}, X_{j})$ between $X_{i}$ and $X_{j}$ is defined as follows.

$$
C N F - 1 \left(X _ {i}, X _ {j}\right) := 1 - e ^ {- \min \left(I \left(X _ {i} \rightarrow X _ {j}\right), I \left(X _ {j} \rightarrow X _ {i}\right)\right)} \tag {2}
$$

For all the confounding measures, we use exponential transformation to limit the range of the measure between 0 and 1. Note that in a DAG, one of $I(X_{i} \to X_{j})$ , $I(X_{j} \to X_{i})$ is zero under no confounding (see Tab. 2 for a simple example with two and three node graphs). Hence $CNF - 1(X_{i}, X_{j})$ outputs zero when there is no confounding between $X_{i}, X_{j}$ . Similarly $CNF - 1(X_{i}, X_{j})$ outputs positive real value in the range (0, 1] when there is confounding. We leverage data from multiple contexts to evaluate $\mathbb{P}(X_{i}|X_{j})$ and $\mathbb{P}(X_{i}|do(X_{j}))$ as follows. In this setting, we assume each context is generated as a result of hard interventions on a subset of variables. Let $P_{ij}$ be the set of node indices that belong to a path from $X_{i}$ to $X_{j}$ including j, we use the contexts $C_{\{i\}\wedge\neg P_{ij}}$ to evaluate $\mathbb{P}(X_{j}|do(X_{i}))$ as $\mathbb{P}(X_{j}|do(X_{i}))=\mathbb{E}_{c\in\mathbf{C}_{\{i\}\wedge\neg P_{ij}}}\left[\mathbb{P}^{c}(X_{j}|X_{i})\right]$ . Intuitively, to compute the interventional effects of $X_{i}$ on $X_{j}$ , we need to observe mechanism changes only for $X_{i}$ to account for the potential causal influence from $X_{i}$ to $X_{j}$ . In addition, none of the nodes in a causal path from $X_{i}$ to $X_{j}$ should be intervened. We use observational data to evaluate $\mathbb{P}(X_{j}|X_{i})$ .

<table><tr><td></td><td>Graph</td><td> $I(X_{i} \to X_{j})$ </td><td> $I(X_{j} \to X_{i})$ </td></tr><tr><td rowspan="2">Uncnf.</td><td> $X_{i} \to X_{j}$ </td><td> $>0$ </td><td> $=0$ </td></tr><tr><td> $X_{j} \to X_{i}$ </td><td> $=0$ </td><td> $>0$ </td></tr><tr><td rowspan="2">Confounded</td><td> $X_{i} \to X_{j}$  $Z \to X_{i}, Z \to X_{j}$ </td><td> $>0$ </td><td> $>0$ </td></tr><tr><td> $X_{j} \to X_{i}$  $Z \to X_{i}, Z \to X_{j}$ </td><td> $>0$ </td><td> $>0$ </td></tr></table>

Table 2: Directed information values in two and three node graphs. If $X_{i}, X_{j}$ are confounded by Z, we observe positive directed information from both directions.

Proposition 4.1. (Identifiability of $\mathbb{P}(X_j|do(X_i))$ ) $\mathbb{P}(X_j|do(X_i))$ is identifiable from the set of contexts $\mathbf{C}_{\{i\} \wedge \neg P_{ij}}$ . To detect and measure confounding between a pair of nodes $X_i, X_j$ , it is enough to observe two sets of contexts $\mathbf{C}_{\{i\} \wedge \neg P_{ij}}$ and $\mathbf{C}_{\{j\} \wedge \neg P_{ji}}$ . Thus, $n$ sets of contexts are needed to detect and measure confounding between $\binom{n}{2}$ distinct pairs of nodes in a causal DAG with $n$ nodes.

When a confounding variable $X_{o}$ between $X_{i}, X_{j}$ is observed, and there may exist an unobserved confounding variable Z, it is crucial to detect and measure unobserved confounding effect [29]. We utilize conditional directed information to define the measure of unobserved confounding.

Definition 4.5. (Conditional Directed Information [48]). The conditional directed information $I(X_{i} \to X_{j}|X_{o})$ from $X_{i}$ to $X_{j}$ conditioned on $X_{o}$ is defined as the conditional Kullback-Leibler divergence between the distributions $\mathbb{P}(X_i|X_j,X_o),\mathbb{P}(X_i|do(X_j),X_o)$ as follows.

$$
I \left(X _ {i} \rightarrow X _ {j} \mid X _ {o}\right) := D _ {K L} \left(\mathbb {P} \left(X _ {i} \mid X _ {j}, X _ {o}\right) \mid \mid \mathbb {P} \left(X _ {i} \mid d o \left(X _ {j}\right), X _ {o}\right)\right) := \underset {\mathbb {P} \left(X _ {i}, X _ {j}, X _ {o}\right)} {\mathbb {E}} \log \frac {\mathbb {P} \left(X _ {i} \mid X _ {j} , X _ {o}\right)}{\mathbb {P} \left(X _ {i} \mid d o \left(X _ {j}\right) , X _ {o}\right)} (3)
$$

This measure can trivially be extended to the case where there exist multiple observed and unobserved confounding variables. The expression $\mathbb{P}(X_{i}|do(X_{j}), X_{o})$ means conditioning on $X_{o}$ in the interventional distribution $\mathbb{P}(X_{i}|do(X_{j}))$ . Now, the conditional confounding can be measured as:

$$
C N F - 1 (X _ {i}, X _ {j} | X _ {o}) := 1 - e ^ {- \min (I (X _ {i} \to X _ {j} | X _ {o}), I (X _ {j} \to X _ {i} | X _ {o}))} \tag {4}
$$

Intuitively, by conditioning on an observed confounding variable $X_{o}$ , we control the association between $X_{i}, X_{j}$ flowing via $X_{o}$ and measure the influence via the unobserved confounding variables.

Beyond Pairwise Confounding: We now study when a set $X_{S}$ of variables where $|X_{S}| > 2$ are jointly confounded i.e., share a common confounding variable and how to measure the joint confounding among the variables $X_{S}$ .

Theorem 4.1. A set of observed variables $X_{S}$ are jointly unconfounded if and only if there exists three variables $X_{i}, X_{j}, X_{k} \in X_{S}$ such that $I(X_{i} \to X_{j} | X_{k}) = I(\{X_{i}, X_{k}\} \to X_{j})$ .

We now define the measure of confounding among the variables in $X_{S}$ as follows.

$$
C N F - 1 (\mathbf {X} _ {S}) = \sum_ {i \in S} C N F - 1 (\mathbf {X} _ {S \setminus \{i \}}, X _ {i}) \tag {5}
$$

Conditional confounding among a set of variables can be defined similarly to Eqn. 4. We now study some useful properties of the measure $CNF - 1$ .

Theorem 4.2. For any three observed variables $X_{i}, X_{j}, X_{o}$ and an unobserved confounding variable Z, the following statements are true for the measure CNF-1.

1. (Reflexivity and Symmetry.) $CNF - 1(X_i, X_i|X_o) = 0$ , $CNF - 1(X_i, X_j|X_o) = CNF - 1(X_j, X_i|X_o)$ .   
2. (Positivity.) $CNF - 1(X_i, X_j) > 0$ if and only if $X_i, X_j$ are confounded. Given an observed confounding variable $X_o$ between $X_i, X_j$ , $CNF - 1(X_i, X_j | X_o) > 0$ if and only if there exists an unobserved confounding variable $Z$ between $X_i, X_j$ .   
3. (Monotonicity.) $CNF - 1(X_{i}, X_{j}) > CNF - 1(X_{k}, X_{l})$ implies that the pair of variables $X_{i}, X_{j}$ are more strongly confounded than the pair of variables $X_{k}, X_{l}$ in the sense of Defns. 4.2 and 4.3.

# 4.2 Setting 2: Detecting and Measuring Confounding Using the Mechanism Shifts of Z.

The previous setting utilizes the interventional effects of $X_{i}(X_{j})$ on $X_{j}(X_{i})$ to define a measure of confounding between $X_{i}, X_{j}$ . In this setting, we utilize the association between the observed marginal distributions of $X_{i}, X_{j}$ under causal mechanism shifts of Z to measure confounding. To this end, similar to [38], we make the following assumption.

Assumption 4.1. (Shift Faithfulness [38]) Let Z be a common parent for a set of variables $X_{S} \subseteq X$ . Then each causal mechanism shift in Z between two contexts c, $c'$ entails a causal mechanism change in each $X_{i} \in X_{S}$ between the same contexts c, $c'$ .

One consequence of the Assumption 4.1 is that a change in the causal mechanism of Z induces correlations between the expectations of $X_{i}, X_{j}$ in different contexts. To understand this, consider the following structural equations.

$$
Z \sim \mathcal {N} (\mu (c), \sigma^ {2} (c)) \quad X _ {i} := \alpha Z + \epsilon_ {i} \quad X _ {j} := \beta X _ {i} + \gamma Z + \epsilon_ {j} \tag {6}
$$

Where $c$ denotes the context and $\epsilon_x$ and $\epsilon_y$ are noise variables with zero mean and have no additional restriction on the underlying probability distribution. The causal graph corresponding to this model has the nodes $X_i, X_j, Z$ and edges: $Z \to X_i, Z \to X_j, X_i \to X_j$ . It is easy to see that $\mathbb{E}(X_i) = \alpha \mu(c)$ and $\mathbb{E}(X_j) = (\alpha \beta + \gamma) \mu(c)$ . Following Assumption 4.1, whenever there is a change in causal mechanism of $Z$ (e.g., $c$ changes to $\tilde{c}$ in Eqn. 6), there is a change in both $\mathbb{E}(X_i), \mathbb{E}(X_j)$ . Additionally, since $Z$ is a common cause of both $X_i, X_j$ , there is a spurious association between $\mathbb{E}(X_i), \mathbb{E}(X_j)$ . Subsequently, in the set of contexts $\mathbf{C}_{\{i\} \wedge \{j\}}$ the values $\mathbb{E}(X_i), \mathbb{E}(X_i)$ are spuriously associated. Under Assumptions 3.2 and 4.1, restricting our analysis to $\mathbf{C}_{\{i\} \wedge \{j\}}$ ensures that with high probability, the association between $\mathbb{E}(X_i), \mathbb{E}(X_i)$ is due to the confounding variable $Z$ . In this example, the association between $\mathbb{E}(X_i), \mathbb{E}(X_j)$ exists even if $\beta = 0$ , i.e., $X_i \not\to X_j$ . To define confounding measure, we create two random variables $E_i^C, E_j^C$ which we define as $E_i^C = \mathbb{E}_{X_i \sim \mathbb{P}^c(X_i)}(X_i), E_j^C = \mathbb{E}_{X_j \sim \mathbb{P}^c(X_j)}(X_j)$ respectively where $c \in \mathbf{C}_{\{i\} \wedge \{j\}}$ . Relying on the context information $\mathbf{C}_{\{i\} \wedge \{j\}}$ and utilizing the association between $E_i^C$ and $E_j^C$ , we define a confounding measure as follows.

Proposition 4.2. (Confounding Based on Mutual Information) If two variables $X_{i}, X_{j}$ are confounded by a variable $Z$ , the induced random variables $E_{i}^{C}, E_{j}^{C}$ as described above have non zero mutual information $I(E_{i}^{C}; E_{j}^{C})$ .

Definition 4.6. (Confounding Measure 2) When the causal mechanism shifts are observed for $X_{i}, X_{j}$ in different contexts and the contexts $\mathbf{C}_{\{i\} \wedge \{j\}}$ are known, under the Assumptions 3.2-4.1, the measure of confounding $CNF - 2(X_{i}, X_{j})$ between $X_{i}$ and $X_{j}$ is defined as

$$
C N F - 2 \left(X _ {i}, X _ {j}\right) := 1 - e ^ {- I \left(E _ {i} ^ {C}; E _ {j} ^ {C}\right)} \tag {7}
$$

To measure the unobserved confounding strength when we already observe a confounding variable $X_{o}$ , we condition on the observed confounding variable $X_{o}$ to define $CNF - 2(X_{i},X_{j}|X_{o})$ as follows.

$$
C N F - 2 \left(X _ {i}, X _ {j} \mid X _ {o}\right) := 1 - e ^ {- I \left(E _ {i} ^ {C}; E _ {j} ^ {C} \mid X _ {o}\right)} \tag {8}
$$

Beyond Pairwise Confounding: Following earlier work [38], we utilize total correlation among triplets $(E_i^C, E_j^C, E_k^C)$ of random variables in $\{E_i^C\}_{i \in S}$ to verify whether a set of variables $\mathbf{X}_S$ are jointly confounded. By Assumption 4.1, we know that the variables in $\mathbf{X}_S$ jointly confounded only if each pair $X_i, X_j$ ; $i, j \in S$ is pairwise confounded. If all three variables share the same latent confounding variable $Z$ , then knowing about one of $E_i^C, E_j^C, E_k^C$ explains away some of the association between the other two, so that we have $I(E_i^C, E_j^C | E_k^C) < I(E_i^C, E_j^C)$ . However, for a triplet $(X_i, X_j, X_k)$ , it is possible that, rather than jointly confounded, there may be three disjoint confounding variables $Z_{12}, Z_{13}, Z_{23}$ confounding each of the individual pairs: $(X_i, X_j), (X_j, X_k), (X_k, X_i)$ . In general, for a set of variables of size $s$ to permit such an equivalent explanation, we would need to have a total of $\binom{s}{2}$ confounding variables with $s(s - 1)$ outgoing edges to obtain the same structure of pairwise confounding [38]. While this may plausibly occur for small sets of variables that appear to be pairwise correlated, we assume the true graph $\mathcal{G}$ to be causally minimal in the following sense.

Assumption 4.2. (Confounder Minimality [38]) For every subset $\mathbf{X}_S$ of at least $|S| \geq 4$ variables, there are at most $2|S|$ edges incoming into $\mathbf{X}_S$ from latent confounding variables with at least three children in $\mathbf{X}_S$ .

Assumption 4.2 ensures that variables that appear to be jointly confounded are indeed confounded. In other words, when a small number of latent variables suffice to explain the observed correlations, there should indeed exist only few confounding variables. With this assumption, we can guarantee that joint confounding can be identified from the total correlation.

Theorem 4.3. Let $\mathbf{X}_S$ be a set of variables such that all $X_i, X_j \in \mathbf{X}_S$ are pairwise confounded. Then $\mathbf{X}_S$ is jointly confounded if and only if for each triple $X_i, X_j, X_k \in \mathbf{X}_S$ we have $I(E_i^C; E_j^C | E_k^C) < I(E_i^C; E_j^C)$ .

Now, the measure of joint confounding among a set of variables $X_{S}$ can be defined using total correlation $T(E_{i}^{C},\ldots,E_{|S|}^{C})$ as follows. To evaluate the following expression, we need to use the contexts $C_{\{1\}\cup\ldots\cup\{|S|\}}$ to ensure that with high probability, the association among the variables in $X_{S}$ is due to the joint confounding variable Z.

$$
C N F - 2 (\mathbf {X} _ {S}) = 1 - e ^ {- T (E _ {i} ^ {C}, \dots , E _ {| S |} ^ {C})} \tag {9}
$$

Theorem 4.4. For any three observed variables $X_{i}, X_{j}, X_{o}$ and an unobserved confounding variable $Z$ , the following statements are true for the measure CNF-2.

1. (Reflexivity and Symmetry.) $CNF - 2(X_i, X_i|X_o) = 1 - e^{-H(E_i^C |X_o)} \forall i$ where $H(.|.)$ denotes conditional entropy and $CNF - 2(X_i, X_j|X_o) = CNF - 2(X_j, X_i|X_o)$ .   
2. (Positivity.) $CNF - 2(X_i, X_j) > 0$ if and only if $X_i, X_j$ are confounded. Given an observed confounding variable $X_o$ between $X_i, X_j$ , $CNF - 2(X_i, X_j | X_o) > 0$ if and only if there exists an unobserved confounding variable $Z$ between $X_i, X_j$ .   
3. (Monotonicity.) $CNF - 2(X_{i}, X_{j}) > CNF - 2(X_{k}, X_{l})$ implies that the pair of variables $X_{i}, X_{j}$ are more strongly confounded than the pair of variables $X_{k}, X_{l}$ in the sense of Defn. 4.2.

# 4.3 Setting 3: Observing the Causal Mechanism Shifts in Z and Known Causal Path Direction Between $X_{i}$ and $X_{j}$

Similar to the previous settings, we utilize marginal and conditional distributions of $X_{i}, X_{j}$ to define a measure of confounding. By prior knowledge, if we know the direction of causal path between $X_{i}, X_{j}$ , we can utilize the causal direction to measure confounding as explained below. In addition to the notations $E_{i}^{C}, E_{j}^{C}$ introduced in the previous setting, let us denote for each $c \in \mathbf{C}_{\{i\} \wedge \{j\}}, \mathbb{E}_{X_{i} \sim \mathbb{P}^{c}(X_{i}|X_{j})}(X_{i}|X_{j}), \mathbb{E}_{X_{j} \sim \mathbb{P}^{c}(X_{j}|X_{i})}(X_{j}|X_{i})$ with $E_{ij}^{C}, E_{ji}^{C}$ respectively. We now leverage dependency among these variables to define the measure of confounding. Intuitively, if $X_{i} \to X_{j}$ and if we observe a change in the causal mechanisms of both $X_{i}, X_{j}$ due to the causal mechanism changes in Z, we also observe a change in the causal mechanism $\mathbb{P}(X_{j}|X_{i})$ .

Definition 4.7. (Confounding Measure 3) When the causal mechanism shifts are observed for $X_{i}, X_{j}$ and the causal direction between the nodes $X_{i}, X_{j}$ is known, under the Assumptions 3.2-4.1, the measure of confounding CNF-3( $X_{i}, X_{j}$ ) between $X_{i} \in X$ and $X_{j} \in X$ is defined as

$$
C N F - 3 \left(X _ {i}, X _ {j}\right) := \left\{\begin{array}{l l}1 - e ^ {- I \left(E _ {j i} ^ {C}; E _ {j} ^ {C}\right)}&i f \quad X _ {i} \rightarrow \dots \rightarrow X _ {j}\\1 - e ^ {- I \left(E _ {i j} ^ {C}; E _ {i} ^ {C}\right)}&i f \quad X _ {j} \rightarrow \dots \rightarrow X _ {i}\\C N F - 2 \left(X _ {i}, X _ {j}\right)&O t h e r w i s e\end{array}\right. \tag {10}
$$

To measure the unobserved confounding strength in the presence of an observed confounding variable $X_{o}$ , similar to setting 2, we can modify Eqn. 10 to condition on the variable $X_{o}$ .

Beyond Pairwise Confounding: Using the Assumption 4.2, we have the following.

Theorem 4.5. Let $\mathbf{X}_S$ be a set of variables such that all $X_i, X_j \in \mathbf{X}_S$ are pairwise confounded and the causal relationships among each pair $X_i, X_j$ . Then $\mathbf{X}_S$ is jointly confounded if and only if for each triple $X_i, X_j, X_k \in \mathbf{X}_S$ we have $I(E_{ij}^C; E_{jk}^C | E_j^C) < I(E_{ij}^C; E_{jk}^C)$ .

Since we have access to random variables $E_{ij}^{C}$ in addition to $E_{i}^{C}, E_{j}^{C}$ , it is not straightforward to use all of them to measure joint confounding. To keep the measure simple, we let the measure of joint confounding among the variables $X_{S}$ be the same as $CNF-2(\mathbf{X}_{S})$ . That is, $CNF-3(\mathbf{X}_{S}) = CNF-2(\mathbf{X}_{S})$ . Setting 3 is an alternative to Setting 2 when we know the direction of the causal path between $X_{i}, X_{j}$ . Settings 2 and 3 act as complementary to each other in validating the correctness of our analysis.

Theorem 4.6. For any three observed variables $X_{i}, X_{j}, X_{o}$ and an unobserved confounding variable $Z$ , the following statements are true for the measure CNF-3.

1. (Reflexivity and Symmetry.) $CNF-3(X_i, X_i | X_o) = 1 - e^{-H(E_i^C | X_o)} \forall i$ where $H(.|.)$ denotes conditional entropy and $CNF-3(X_i, X_j | X_o) = CNF-3(X_j, X_i | X_o)$ .   
2. (Positivity.) $CNF-3(X_{i}, X_{j}) > 0$ if and only if $X_{i}, X_{j}$ are confounded. Given an observed confounding variable $X_{o}$ between $X_{i}, X_{j}$ , $CNF-3(X_{i}, X_{j} | X_{o}) > 0$ if and only if there exists an unobserved confounding variable Z between $X_{i}, X_{j}$ .   
3. (Monotonicity.) $CNF - 3(X_i, X_j) > CNF - 3(X_k, X_l)$ implies that the pair of variables $X_i, X_j$ are more strongly confounded than the pair of variables $X_k, X_l$ in the sense of Defn. 4.2.

# 5 Algorithm

Algorithm 1 outlines the procedures to measure confounding in all three settings and can be extended to the case where we evaluate conditional confounding and evaluating confounding among multiple variables. We present two real-world examples where our methods can be applied in Appendix § B.

Algorithm 1: Algorithm for evaluating pairwise CNF-1, CNF-2, CNF-3   
Data: Context information $C_{\{i\}\wedge\neg P_{ij}}$ , $C_{\{j\}\wedge\neg P_{ji}}$ , $C_{\{i\}\wedge\{j\}}$ , Contextual Datasets $\{D^{c}\}_{c\in C}$ .

Result: $CNF-1(X_{i},X_{j})$ , $CNF-2(X_{i},X_{j})$ , $CNF-3(X_{i},X_{j})$ Step 1: Evaluate $\mathbb{P}(X_{i}|X_{j}),\mathbb{P}(X_{j}|X_{i})$ using observational data;

Step 2: Evaluate $\mathbb{P}(X_{i}|do(X_{j}))$ using $\{D^{c}\}_{c\in C_{\{j\}\wedge\neg P_{ji}}}$ ;

Step 3: Evaluate $\mathbb{P}(X_{j}|do(X_{i}))$ using $\{D^{c}\}_{c\in C_{\{i\}\wedge\neg P_{ij}}}$ ;

Step 4: Evaluate $I(X_{i}\to X_{j})$ , $I(X_{j}\to X_{i})$ ;

Step 5: $CNF-1(X_{i},X_{j})=1-e^{-\min(I(X_{i}\to X_{j}),I(X_{j}\to X_{i}))}$ ;

Step 6: Evaluate $E_{i}^{C},E_{j}^{C}$ using $\{D^{c}\}_{c\in C_{\{i\}\wedge\{j\}}}$ ;

Step 7: $CNF-2(X_{i},X_{j})=1-e^{-I(E_{i}^{c};E_{j}^{c})}$ ;

Step 8: Evaluate $E_{ij}^{C},E_{ji}^{C}$ using $\{D^{c}\}_{c\in C_{\{i\}\wedge\{j\}}}$ ;

Step 9: compute $CNF-3(X_{i},X_{j})$ according to Defn. 4.7;

return $CNF-1(X_{i},X_{j})$ , $CNF-2(X_{i},X_{j})$ , $CNF-3(X_{i},X_{j})$

# 6 Experiments and Results

We perform simulation studies to verify the correctness of the proposed measures. All the experiments are run on a CPU. We report the mean and standard deviation of results taken over five random seeds. Code to reproduce the results is presented in the supplementary material. Code is available at https://github.com/gautam0707/CD\_CNF.

Measuring Confounding: In this set of experiments, we consider the following four causal structures made of three nodes $X_{i}, X_{j}, Z$ : $G_{1}$ : Empty graph over $Z, X_{i}, X_{j}$ i.e., nodes are isolated in the graph, $G_{2}: X_{i} \to X_{j}, G_{3}: Z \to X_{i}, Z \to X_{j}, G_{4}: Z \to X_{i}, Z \to X_{j}, X_{i} \to X_{j}$ . In $G_{1}, G_{2}$ , there is no confounding between $X_{i}, X_{j}$ and in $G_{3}, G_{4}$ there is confounding effect of Z on $X_{i}$ and $X_{j}$ . Results in Fig. 2 show that our measures output zero when there is no confounding between $X_{i}, X_{j}$ and output positive values when $X_{i}, X_{j}$ are confounded by a confounding variable Z.

![](images/a4b9554da7a121e4370cc36a9b5f7fa26fb975aa85dc120e9deba003878f7dab.jpg)  
Figure 2: Measure of confounding between a pair of variables $X_{i}, X_{j}$ . Our measures output zero when there is no confounding between $X_{i}, X_{j}$ and output positive values when $X_{i}, X_{j}$ are confounded.

Measuring Conditional Confounding: We consider the following two causal structures. $G_{5}: Z_{1} \rightarrow X_{i}, Z_{1} \rightarrow X_{j}, Z_{2} \rightarrow X_{i}, Z_{2} \rightarrow X_{j}, X_{i} \rightarrow X_{j}$ . $G_{6}: Z \rightarrow X_{i}, Z \rightarrow X_{j}, X_{i} \rightarrow X_{j}$ . In $G_{5}, X_{i}$ and $X_{j}$ are confounded by two variables $Z_{1}, Z_{2}$ . We measure conditional confounding between $X_{i}, X_{j}$ conditioned on $\emptyset, Z_{1}$ , and $Z_{2}$ respectively. Since confounding still exists in all of the above conditioning settings, CNF-2 correctly returns positive confounding value in all three cases (see Fig. 3 left). On the other hand, in $G_{6}$ , we measure conditional confounding

![](images/fa8e61b10064719fc3201b772ddebba5dd692c6dcb8b4eba85259139f8c82390.jpg)

![](images/9c406bbcad1c1a2fb506d6e637d2fe34fe935a66ed3c4f9458f788ffc1eccb64.jpg)  
Figure 3: Left: Conditioning on one of $\emptyset, Z_1, Z_2$ will not remove confounding between $X_i, X_j$ in $\mathcal{G}_5$ . Hence $CNF-2$ returns positive values. Right: In $\mathcal{G}_6$ , conditioning on $\emptyset$ does not remove the confounding effect of $Z$ on $X_i, X_j$ . Hence, we observe a positive value for $CNF-2(X_i, X_j|\emptyset)$ . Conditioning on $Z$ will block the confounding between $X_i, X_j$ . Hence $CNF-2$ is closer to zero.

between $X_{i}, X_{j}$ conditioning on empty set and Z. Since conditioning on Z will block the confounding association between $X_{i}, X_{j}, CNF-2$ returns confounding value closer to zero. However, the unconditioned confounding (conditioning on empty set) value is still large. These results empirically validate the correctness of the proposed measures.

Downstream Causal Effect Estimation: For the causal graphs $G_{3}$ , $G_{4}$ , we examine the impact of controlling for nodes identified using our method. We measure the causal effect of $X_{i}$

<table><tr><td rowspan="2">Causal Graph</td><td colspan="5">Not Controlling Confounding</td><td colspan="5">Controlling Confounding</td></tr><tr><td>1000</td><td>2000</td><td>3000</td><td>4000</td><td>5000</td><td>1000</td><td>2000</td><td>3000</td><td>4000</td><td>5000</td></tr><tr><td> $\mathcal{G}_{3}$ </td><td>0.55</td><td>0.57</td><td>0.55</td><td>0.52</td><td>0.52</td><td>0.06</td><td>0.02</td><td>0.007</td><td>0.03</td><td>0.009</td></tr><tr><td> $\mathcal{G}_{4}$ </td><td>0.24</td><td>0.26</td><td>0.23</td><td>0.24</td><td>0.23</td><td>0.04</td><td>0.05</td><td>0.06</td><td>0.02</td><td>0.05</td></tr></table>

Table 3: Downstream application of causal effect estimation.

on $X_{j}$ with and without controlling for the detected confounding variable and report the absolute difference between the true and estimated causal effects in Tab. 3. The results show that controlling for the variables identified by our method reduces the bias in the estimated causal effects.

Binary Data - Erdős-Rényi Causal Graphs: To verify the performance of our method on a large scale, similar to [38], we generate causal graphs of various number nodes using Erdős-Rényi model. In these experiments, each context is a result of intervention on one node. This is the reason for having the same value for number of nodes $N$ and number of contexts $|C|$ . Sample size denotes the number of data points used in each context. We detect and measure whether each pair of nodes is confounded or not. We then calculate the Precision, Recall, and F1 scores. Our confounding measures obtain good results across all settings.

<table><tr><td colspan="2"></td><td colspan="3">Setting 1</td><td colspan="3">Setting 2</td><td colspan="3">Setting 3</td></tr><tr><td>N, |C|</td><td>Sample Size</td><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>10</td><td>100</td><td>0.64</td><td>0.97</td><td>0.77</td><td>0.67</td><td>0.83</td><td>0.74</td><td>0.64</td><td>0.72</td><td>0.68</td></tr><tr><td>10</td><td>200</td><td>0.64</td><td>1.0</td><td>0.78</td><td>0.67</td><td>0.83</td><td>0.74</td><td>0.70</td><td>0.79</td><td>0.74</td></tr><tr><td>10</td><td>300</td><td>0.64</td><td>1.0</td><td>0.78</td><td>0.67</td><td>0.83</td><td>0.74</td><td>0.65</td><td>0.76</td><td>0.70</td></tr><tr><td>10</td><td>400</td><td>0.64</td><td>1.0</td><td>0.78</td><td>0.67</td><td>0.83</td><td>0.74</td><td>0.67</td><td>0.83</td><td>0.74</td></tr><tr><td>10</td><td>500</td><td>0.64</td><td>1.0</td><td>0.78</td><td>0.67</td><td>0.83</td><td>0.74</td><td>0.67</td><td>0.83</td><td>0.74</td></tr><tr><td>15</td><td>100</td><td>0.81</td><td>0.95</td><td>0.88</td><td>0.80</td><td>0.85</td><td>0.82</td><td>0.80</td><td>0.79</td><td>0.80</td></tr><tr><td>15</td><td>200</td><td>0.82</td><td>1.0</td><td>0.90</td><td>0.80</td><td>0.85</td><td>0.82</td><td>0.80</td><td>0.85</td><td>0.82</td></tr><tr><td>15</td><td>300</td><td>0.82</td><td>1.0</td><td>0.90</td><td>0.80</td><td>0.85</td><td>0.82</td><td>0.80</td><td>0.85</td><td>0.82</td></tr><tr><td>15</td><td>400</td><td>0.82</td><td>1.0</td><td>0.90</td><td>0.80</td><td>0.85</td><td>0.82</td><td>0.80</td><td>0.85</td><td>0.82</td></tr><tr><td>15</td><td>500</td><td>0.82</td><td>1.0</td><td>0.90</td><td>0.80</td><td>0.85</td><td>0.82</td><td>0.80</td><td>0.84</td><td>0.82</td></tr><tr><td>20</td><td>100</td><td>0.68</td><td>0.95</td><td>0.80</td><td>0.68</td><td>0.88</td><td>0.77</td><td>0.69</td><td>0.84</td><td>0.76</td></tr><tr><td>20</td><td>200</td><td>0.69</td><td>1.0</td><td>0.82</td><td>0.68</td><td>0.88</td><td>0.77</td><td>0.68</td><td>0.87</td><td>0.76</td></tr><tr><td>20</td><td>300</td><td>0.69</td><td>1.0</td><td>0.82</td><td>0.68</td><td>0.88</td><td>0.77</td><td>0.67</td><td>0.86</td><td>0.75</td></tr><tr><td>20</td><td>400</td><td>0.69</td><td>1.0</td><td>0.82</td><td>0.68</td><td>0.88</td><td>0.77</td><td>0.68</td><td>0.87</td><td>0.76</td></tr><tr><td>20</td><td>500</td><td>0.69</td><td>1.0</td><td>0.82</td><td>0.68</td><td>0.88</td><td>0.77</td><td>0.68</td><td>0.87</td><td>0.76</td></tr><tr><td>25</td><td>100</td><td>0.83</td><td>0.96</td><td>0.89</td><td>0.83</td><td>0.91</td><td>0.87</td><td>0.83</td><td>0.89</td><td>0.86</td></tr><tr><td>25</td><td>200</td><td>0.83</td><td>1.0</td><td>0.91</td><td>0.83</td><td>0.91</td><td>0.87</td><td>0.82</td><td>0.90</td><td>0.86</td></tr><tr><td>25</td><td>300</td><td>0.83</td><td>1.0</td><td>0.91</td><td>0.83</td><td>0.91</td><td>0.87</td><td>0.83</td><td>0.91</td><td>0.87</td></tr><tr><td>25</td><td>400</td><td>0.83</td><td>1.0</td><td>0.91</td><td>0.83</td><td>0.92</td><td>0.87</td><td>0.83</td><td>0.91</td><td>0.87</td></tr><tr><td>25</td><td>500</td><td>0.83</td><td>1.0</td><td>0.91</td><td>0.83</td><td>0.91</td><td>0.87</td><td>0.83</td><td>0.91</td><td>0.87</td></tr></table>

Table 4: Results on synthetic datasets for settings 1,2,3.

# 7 Conclusions, Limitations, and Future Work

In this paper, based on the known causal mechanism shifts of observed variables, we propose three measures of confounding along with their conditional and multivariate variants. We also study key properties of these measures. Our measures complement each other depending on the available context information. We propose algorithms to compute the proposed measures and empirically verify their correctness. However, for the same confounded pair of variables, our metrics may yield different results depending on the chosen measure. As discussed in the introduction, the measures are intended to assess the relative strengths of confounding rather than for point-to-point comparison. The number of contexts required to evaluate the measure can be large because many contexts without changes in particular mechanisms are discarded. Identifying appropriate real-world datasets and applying the proposed measures to those datasets is an interesting area for future work, as is developing measures that efficiently use context information. Additionally, devising new definitions for confounding and proposing corresponding confounding measures is also an interesting future direction. We aim to pursue these ideas.

# Acknowledgments

This work was partly supported by the Prime Minister's Research Fellowship (PMRF) program and an Adobe Research Gift. We are grateful to the anonymous reviewers for their valuable feedback, which improved the presentation of the paper.

# References

[1] John Aldrich. Correlations genuine and spurious in pearson and yule. Statistical science, pages 364–376, 1995.   
[2] Rohit Bhattacharya, Tushar Nagarajan, Daniel Malinsky, and Ilya Shpitser. Differentiable causal discovery under unmeasured confounding. In International Conference on Artificial Intelligence and Statistics, pages 2314–2322, 2021.   
[3] Norman E Breslow, Nicholas E Day, and Elisabeth Heseltine. Statistical methods in cancer research. 1980.   
[4] Philippe Brouillard, Sébastien Lachapelle, Alexandre Lacoste, Simon Lacoste-Julien, and Alexandre Drouin. Differentiable causal discovery from interventional data. In Advances in Neural Information Processing Systems, volume 33, pages 21865–21877, 2020.   
[5] Esben Budtz–Jørgensen, Niels Keiding, Philippe Grandjean, and Pal Weihe. Confounder selection in environmental epidemiology: Assessment of health effects of prenatal mercury exposure. Annals of Epidemiology, 17(1):27–35, 2007.   
[6] Timothy A Carey and William B Stiles. Some problems with randomized controlled trials and some viable alternatives. Clinical Psychology & Psychotherapy, 23(1):87–95, 2016.   
[7] Venkat Chandrasekaran, Pablo A Parrilo, and Alan S Willsky. Latent variable graphical model selection via convex optimization. In 2010 48th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 1610–1613. IEEE, 2010.   
[8] David Maxwell Chickering. Learning equivalence classes of bayesian-network structures. The Journal of Machine Learning Research, 2:445–498, 2002.   
[9] Diego Colombo, Marloes H Maathuis, Markus Kalisch, and Thomas S Richardson. Learning high-dimensional directed acyclic graphs with latent and selection variables. The Annals of Statistics, pages 294–321, 2012.   
[10] Shai Ben David, Tyler Lu, Teresa Luu, and Dávid Pál. Impossibility theorems for domain adaptation. In AISTATS, pages 129–136, 2010.   
[11] Mirthe Maria Van Diepen, Ioan Gabriel Bucur, Tom Heskes, and Tom Claassen. Beyond the markov equivalence class: Extending causal discovery under latent confounding. In 2nd Conference on Causal Learning and Reasoning, 2023.   
[12] Frederick Eberhardt and Richard Scheines. Interventions and causal inference. Philosophy of science, 74(5):981–995, 2007.   
[13] Frederick Eberhardt, Clark Glymour, and Richard Scheines. On the number of experiments sufficient and in the worst case necessary to identify all causal relations among n variables. arXiv preprint arXiv:1207.1389, 2012.   
[14] Robin J Evans and Thomas S Richardson. Smooth, identifiable supermodels of discrete dag models with latent variables. Bernoulli, 25:848–876, 2019.   
[15] Sander Greenland and Hal Morgenstern. Confounding in health research. Annual review of public health, 22(1):189–212, 2001.   
[16] R.H.H. Groenwold, E. Hak, and A.W. Hoes. Quantitative assessment of unobserved confounding is mandatory in nonrandomized intervention studies. Journal of Clinical Epidemiology, 62(1):22–28, 2009.

[17] Siyuan Guo, Viktor Tóth, Bernhard Schölkopf, and Ferenc Huszár. Causal de finetti: On the identification of invariant causal structure in exchangeable data. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
[18] Gemma Hammerton and Marcus R Munafò. Causal inference with observational data: the need for triangulation of evidence. Psychological medicine, 51(4):563–578, 2021.   
[19] Alain Hauser and Peter Bühlmann. Two optimal strategies for active learning of causal models from interventional data. International Journal of Approximate Reasoning, 55(4):926–939, 2014.   
[20] Patrik O Hoyer, Shohei Shimizu, Antti J Kerminen, and Markus Palviainen. Estimation of causal effects using linear non-gaussian causal models with hidden variables. International Journal of Approximate Reasoning, 49(2):362–378, 2008.   
[21] Biwei Huang, Kun Zhang, Jiji Zhang, Ruben Sanchez-Romero, Clark Glymour, and Bernhard Schölkopf. Behind distribution shift: Mining driving forces of changes and causal arrows. In 2017 IEEE International Conference on Data Mining (ICDM), pages 913–918. IEEE, 2017.   
[22] Biwei Huang, Kun Zhang, Jiji Zhang, Joseph Ramsey, Ruben Sanchez-Romero, Clark Glymour, and Bernhard Schölkopf. Causal discovery from heterogeneous/nonstationary data. Journal of Machine Learning Research, 21(89):1–53, 2020.   
[23] Amin Jaber, Murat Kocaoglu, Karthikeyan Shanmugam, and Elias Bareinboim. Causal discovery from soft interventions with unknown targets: Characterization and learning. Advances in neural information processing systems, 33:9551–9561, 2020.   
[24] Holly Janes, Francesca Dominici, and Scott Zeger. On quantifying the magnitude of confounding. Biostatistics, 11(3):572–582, 2010.   
[25] Dominik Janzing and Bernhard Schölkopf. Detecting confounding in multivariate linear models via spectral analysis. Journal of Causal Inference, 6(1):20170013, 2018.   
[26] Andrew Jesson, Alyson Rose Douglas, Peter Manshausen, Maëlys Solal, Nicolai Meinshausen, Philip Stier, Yarin Gal, and Uri Shalit. Scalable sensitivity and uncertainty analyses for causal-effect estimates of continuous-valued interventions. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.   
[27] David Kaltenpoth and Jilles Vreeken. Nonlinear causal discovery with latent confounders. In International Conference on Machine Learning, pages 15639–15654, 2023.   
[28] David Kaltenpoth and Jilles Vreeken. Causal discovery with hidden confounders using the algorithmic Markov condition. In Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, pages 1016–1026, 2023.   
[29] Rickard Karlsson and Jesse Krijthe. Detecting hidden confounding in observational data using multiple environments. Advances in Neural Information Processing Systems, 36, 2023.   
[30] David G Kleinbaum, Kevin M Sullivan, and Nancy D Barker. A pocket guide to epidemiology. Springer, 2007.   
[31] Charles Ksir and Carl L Hart. Correlation still does not imply causation. The Lancet Psychiatry, 3(5):401, 2016.   
[32] Paul H Lee. Is a cutoff of 10% appropriate for the change-in-estimate criterion of confounder identification? Journal of epidemiology, 24(2):161–167, 2014.   
[33] Adam Li, Amin Jaber, and Elias Bareinboim. Causal discovery from observational and interventional data across multiple environments. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
[34] George Maldonado and Sander Greenland. Simulation study of confounder-selection strategies. American journal of epidemiology, 138(11):923–936, 1993.

[35] George Maldonado and Sander Greenland. Estimating causal effects. International journal of epidemiology, 31(2):422–429, 2002.   
[36] Sarah Mameche, David Kaltenpoth, and Jilles Vreeken. Discovering invariant and changing mechanisms from data. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1242–1252, 2022.   
[37] Sarah Mameche, David Kaltenpoth, and Jilles Vreeken. Learning causal models under independent changes. Advances in Neural Information Processing Systems, 36, 2023.   
[38] Sarah Mameche, Jilles Vreeken, and David Kaltenpoth. Identifying confounding from causal mechanism shifts. In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS). PMLR, 2024.   
[39] Olli S Miettinen and E Francis Cook. Confounding: essence and detection. American journal of epidemiology, 114(4):593–603, 1981.   
[40] Joris M Mooij, Sara Magliacane, and Tom Claassen. Joint causal inference from multiple contexts. Journal of Machine Learning Research, 21:1–108, 2020.   
[41] Austin Nichols. Causal inference with observational data. The Stata Journal, 7(4):507–541, 2007.   
[42] Juan Miguel Ogarrio, Peter Spirtes, and Joe Ramsey. A hybrid causal search algorithm for latent variable models. In Proceedings of the Eighth International Conference on Probabilistic Graphical Models, pages 368–379, 2016.   
[43] Menglan Pang, Jay S Kaufman, and Robert W Platt. Studying noncollapsibility of the odds ratio with marginal structural and logistic regression models. Statistical methods in medical research, 25(5):1925–1937, 2016.   
[44] Judea Pearl. Causality. Cambridge university press, 2009.   
[45] Ronan Perry, Julius Von Kügelgen, and Bernhard Schölkopf. Causal discovery in heterogeneous environments under the sparse mechanism shift hypothesis. In Advances in Neural Information Processing Systems, pages 10904–10917, 2022.   
[46] Jonas Peters, Joris M Mooij, Dominik Janzing, and Bernhard Schölkopf. Causal discovery with continuous additive noise models. The Journal of Machine Learning Research, 15(1):2009–2053, 2014.   
[47] Jonas Peters, Dominik Janzing, and Bernhard Schlkopf. Elements of Causal Inference: Foundations and Learning Algorithms. The MIT Press, 2017.   
[48] Maxim Raginsky. Directed information and pearl's causal calculus. In 2011 49th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 958–965, 2011.   
[49] Thomas S Richardson, Robin J Evans, James M Robins, and Ilya Shpitser. Nested markov properties for acyclic directed mixed graphs. The Annals of Statistics, 51(1):334–361, 2023.   
[50] Robert William Sanson-Fisher, Billie Bonevski, Lawrence W. Green, and Cate D'Este. Limitations of the randomized controlled trial in evaluating population-based health interventions. American Journal of Preventive Medicine, 33(2):155–161, 2007.   
[51] Mauro Scanagatta, Cassio P de Campos, Giorgio Corani, and Marco Zaffalon. Learning bayesian networks with thousands of variables. Advances in neural information processing systems, 28, 2015.   
[52] B Schölkopf, D Janzing, J Peters, E Sgouritsa, K Zhang, and J Mooij. On causal and anticausal learning. In 29th International Conference on Machine Learning (ICML 2012), pages 1255-1262. International Machine Learning Society, 2012.

[53] Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. Proceedings of the IEEE, 109(5):612–634, 2021.   
[54] Noah A Schuster, Jos WR Twisk, Gerben Ter Riet, Martijn W Heymans, and Judith JM Rijnhart. Noncollapsibility and its role in quantifying confounding bias in logistic regression. BMC medical research methodology, 21:1–9, 2021.   
[55] Karthikeyan Shanmugam, Murat Kocaoglu, Alexandros G Dimakis, and Sriram Vishwanath. Learning causal graphs with small interventions. Advances in Neural Information Processing Systems, 28, 2015.   
[56] Ilya Shpitser, Robin J Evans, Thomas S Richardson, and James M Robins. Introduction to nested markov models. \*Behaviormetrika\*, 41:3-39, 2014.   
[57] Ilya Shpitser, Robin J Evans, and Thomas S Richardson. Acyclic linear sems obey the nested markov property. In Uncertainty in artificial intelligence: proceedings of the... conference. Conference on Uncertainty in Artificial Intelligence, volume 2018. NIH Public Access, 2018.   
[58] Edward H Simpson. The interpretation of interaction in contingency tables. Journal of the Royal Statistical Society: Series B (Methodological), 13(2):238–241, 1951.   
[59] Peter Spirtes and Kun Zhang. Causal discovery and inference: concepts and recent methodological advances. In Applied informatics, volume 3, pages 1–28. Springer, 2016.   
[60] Peter Spirtes, Clark N Glymour, and Richard Scheines. Causation, prediction, and search. 2000.   
[61] Zhiqiang Tan. A distributional approach for causal inference using propensity scores. Journal of the American Statistical Association, 101(476):1619–1637, 2006.   
[62] Tyler J VanderWeele and Ilya Shpitser. On the definition of a confounder. Annals of statistics, 41(1):196, 2013.   
[63] Y. Samuel Wang and Mathias Drton. Causal discovery with unobserved confounding and non-gaussian data. Journal of Machine Learning Research, 24(271):1–61, 2023.   
[64] Aleksander Wieczorek and Volker Roth. Information theoretic causal effect quantification. Entropy, 21(10), 2019.   
[65] Alessio Zanga, Elif Ozkirimli, and Fabio Stella. A survey on causal discovery: Theory and practice. International Journal of Approximate Reasoning, 151:101–129, 2022.   
[66] Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric P Xing. Dags with no tears: Continuous optimization for structure learning. Advances in neural information processing systems, 31, 2018.

# Appendix

# A Proofs

Proposition 4.1. (Identifiability of $\mathbb{P}(X_j|do(X_i))$ ) $\mathbb{P}(X_j|do(X_i))$ is identifiable from the set of contexts $\mathbf{C}_{\{i\} \wedge \neg P_{ij}}$ . To detect and measure confounding between a pair of nodes $X_i, X_j$ , it is enough to observe two sets of contexts $\mathbf{C}_{\{i\} \wedge \neg P_{ij}}$ and $\mathbf{C}_{\{j\} \wedge \neg P_{ji}}$ . Thus, $n$ sets of contexts are needed to detect and measure confounding between $\binom{n}{2}$ distinct pairs of nodes in a causal DAG with $n$ nodes.

Proof. Since the set of contexts $C_{\{i\}\wedge P_{ij}}$ consist of data with all possible interventions on $X_{i}$ , if a context c is generated by performing intervention on $X_{i}$ with the value $x_{i}$ , the expression $\mathbb{P}(X_{j}|do(X_{i}=x_{i}))$ is equal to the expression $\mathbb{P}(X_{j}|X_{i}=x_{i})$ in that context c.

From Defn. 4.4, to detect and measure confounding between the pair of variables $X_{i}, X_{j}$ , we need to evaluate $\mathbb{P}(X_{j}|do(X_{i}))$ and $\mathbb{P}(X_{i}|do(X_{j}))$ . To this end, from the previous paragraph, we need two sets of contexts $\mathbf{C}_{\{i\} \wedge P_{ij}}$ and $\mathbf{C}_{\{j\} \wedge P_{ji}}$ . Following these observations, it is enough to have $n$ sets of contexts to detect and measure confounding between $\binom{n}{2}$ distinct pairs of nodes.

Theorem 4.1. A set of observed variables $X_{S}$ are jointly unconfounded if and only if there exists three variables $X_{i}, X_{j}, X_{k} \in X_{S}$ such that $I(X_{i} \to X_{j} | X_{k}) = I(\{X_{i}, X_{k}\} \to X_{j})$ .

Proof. Consider three variables $X_{i}, X_{j}, X_{k}$ in the underlying causal graph. Consider the conditional directed information between $X_{i}, X_{j}$ given $X_{k}$ and the subsequent manipulations as follows.

$$
\begin{array}{l} I (X _ {i} \to X _ {j} | X _ {k}) := \mathbb {E} _ {\mathbb {P} (X _ {i}, X _ {j}, X _ {k})} \log \frac {\mathbb {P} (X _ {i} | X _ {j} , X _ {k})}{\mathbb {P} (X _ {i} | d o (X _ {j}) , X _ {k})} \\ = \mathbb {E} _ {\mathbb {P} (X _ {i}, X _ {j}, X _ {k})} \log \left(\frac {\mathbb {P} (X _ {i} , X _ {k} | X _ {j})}{\mathbb {P} (X _ {i} , X _ {k} | d o (X _ {j}))} \times \frac {\mathbb {P} (X _ {k} | d o (X _ {j}))}{\mathbb {P} (X _ {k} | X _ {j})}\right) \\ = \mathbb {E} _ {\mathbb {P} (X _ {i}, X _ {j}, X _ {k})} \log \frac {\mathbb {P} (X _ {i} , X _ {k} | X _ {j})}{\mathbb {P} (X _ {i} , X _ {k} | d o (X _ {j}))} - \mathbb {E} _ {\mathbb {P} (X _ {j}, X _ {k})} \log \frac {\mathbb {P} (X _ {k} | X _ {j})}{\mathbb {P} (X _ {k} | d o (X _ {j}))} \\ = I (\{X _ {i} X _ {k} \} \rightarrow X _ {j}) - I (X _ {k} \rightarrow X _ {j}) \\ \end{array}
$$

Since $I(X_k \to X_j) \geq 0$ , we have $I(X_i \to X_j | X_k) \leq I(\{X_i X_k\} \to X_j)$ . Equality holds only when $X_k, X_j$ are unconfounded.

Theorem 4.2. For any three observed variables $X_{i}, X_{j}, X_{o}$ and an unobserved confounding variable Z, the following statements are true for the measure CNF-1.

1. (Reflexivity and Symmetry.) $CNF - 1(X_i, X_i|X_o) = 0$ , $CNF - 1(X_i, X_j|X_o) = CNF - 1(X_j, X_i|X_o)$ .   
2. (Positivity.) $CNF - 1(X_i, X_j) > 0$ if and only if $X_i, X_j$ are confounded. Given an observed confounding variable $X_o$ between $X_i, X_j$ , $CNF - 1(X_i, X_j | X_o) > 0$ if and only if there exists an unobserved confounding variable $Z$ between $X_i, X_j$ .   
3. (Monotonicity.) $CNF - 1(X_{i}, X_{j}) > CNF - 1(X_{k}, X_{l})$ implies that the pair of variables $X_{i}, X_{j}$ are more strongly confounded than the pair of variables $X_{k}, X_{l}$ in the sense of Defns. 4.2 and 4.3.

Proof. Reflexivity: From the definition of directed information, $I(X_{i} \rightarrow X_{i}|X_{o}) = \mathbb{E}_{\mathbb{P}(X_{i}, X_{j}, X_{o})} \log \frac{\mathbb{P}(X_{i}|X_{o})}{\mathbb{P}(X_{i}|X_{o})} = 0$ and hence CNF-1( $X_{i}, X_{j}|X_{o}$ ) = 1 - $e^{0} = 0$ .

Symmetry: Even if $I(X_{i} \rightarrow X_{j}|X_{o})$ is not symmetric, the expression ‘ $\min(I(X_{i} \rightarrow X_{j}|X_{o}), I(X_{j} \rightarrow X_{i}|X_{o}))$ ’ is symmetric and hence CNF-1( $X_{i}, X_{j}|X_{o}$ ) is symmetric.

Positivity: If $X_{i}, X_{j}$ are confounded, irrespective of the direction of the causal path between $X_{i}$ and $X_{j}$ , we have $\mathbb{P}(X_{i}|X_{j}) \neq \mathbb{P}(X_{i}|do(X_{j}))$ and $\mathbb{P}(X_{j}|X_{i}) \neq \mathbb{P}(X_{j}|do(X_{i}))$ . Hence $I(X_{i} \to X_{j}) > 0$ and $I(X_{j} \to X_{i}) > 0$ . We now have CNF-1( $X_{i}, X_{j}$ ) > 0. The above statement is true even if there is no causal path between the nodes $X_{i}, X_{j}$ . The above statements are valid even after conditioning on an observed confounding variable $X_{o}$ if there is an unobserved confounding between $X_{i}, X_{j}$ .

Monotonicity: Without loss of generality, assume that the inequality $CNF-1(X_{i},X_{j}) > CNF-1(X_{k},X_{l})$ is a result of $I(X_{i} \to X_{j}) > I(X_{k} \to X_{l})$ . That is, the KL divergence between $\mathbb{P}(X_{i}|X_{j})$ and $\mathbb{P}(X_{i}|do(X_{j}))$ is greater than the kl divergence between $\mathbb{P}(X_{k}|X_{l})$ and $\mathbb{P}(X_{k}|do(X_{l}))$ . That is, the pair of distributions $\mathbb{P}(X_{k}|X_{l})$ and $\mathbb{P}(X_{k}|do(X_{l}))$ are closer to each other compared to the pair $\mathbb{P}(X_{i}|X_{j})$ and $\mathbb{P}(X_{i}|do(X_{j}))$ . As a result, $X_{k}, X_{l}$ are closer to being not confounded in the sense of Defns. 4.2 and 4.3.

Proposition 4.2. (Confounding Based on Mutual Information) If two variables $X_{i}, X_{j}$ are confounded by a variable Z, the induced random variables $E_{i}^{C}, E_{j}^{C}$ as described above have non zero mutual information $I(E_{i}^{C}; E_{j}^{C})$ .

Proof. There are two sources of dependency between $E_{i}^{C}, E_{j}^{C}$ . If $X_{i}, X_{j}$ are causally related in the underlying causal model generating the data, there will be a dependency between $E_{i}^{C}, E_{j}^{C}$ in the context $C_{\{i\}\wedge\{j\}}$ as the interventions are soft. On the other hand, as per the Assumption 4.1, any shift in the causal mechanism of Z leads to a change in both the mechanisms of $X_{i}, X_{j}$ leading to a dependency. Hence the random variables $E_{i}^{C}, E_{j}^{C}$ have non-zero mutual information. □

Theorem 4.3. Let $\mathbf{X}_S$ be a set of variables such that all $X_i, X_j \in \mathbf{X}_S$ are pairwise confounded. Then $\mathbf{X}_S$ is jointly confounded if and only if for each triple $X_i, X_j, X_k \in \mathbf{X}_S$ we have $I(E_i^C; E_j^C | E_k^C) < I(E_i^C; E_j^C)$ .

Proof. Following the Assumption 4.2, when three variables $X_{i}, X_{j}, X_{k}$ are confounded by as single confounding variable $Z$ , conditioning on one of $E_{i}^{C}, E_{j}^{C}, E_{k}^{C}$ explains away some of the dependency between other two. Hence we have $I(E_{i}^{C}; E_{j}^{C}|E_{k}^{C}) < I(E_{i}^{C}; E_{j}^{C})$ for all triples $i, j, k$ .

Theorem 4.4. For any three observed variables $X_{i}, X_{j}, X_{o}$ and an unobserved confounding variable $Z$ , the following statements are true for the measure CNF-2.

1. (Reflexivity and Symmetry.) $CNF - 2(X_i, X_i | X_o) = 1 - e^{-H(E_i^C | X_o)} \forall i$ where $H(.|.)$ denotes conditional entropy and $CNF - 2(X_i, X_j | X_o) = CNF - 2(X_j, X_i | X_o)$ .   
2. (Positivity.) $CNF - 2(X_i, X_j) > 0$ if and only if $X_i, X_j$ are confounded. Given an observed confounding variable $X_o$ between $X_i, X_j$ , $CNF - 2(X_i, X_j | X_o) > 0$ if and only if there exists an unobserved confounding variable $Z$ between $X_i, X_j$ .   
3. (Monotonicity.) $CNF - 2(X_{i}, X_{j}) > CNF - 2(X_{k}, X_{l})$ implies that the pair of variables $X_{i}, X_{j}$ are more strongly confounded than the pair of variables $X_{k}, X_{l}$ in the sense of Defn. 4.2.

Proof. Reflexivity: from the definition of mutual information, $I(E_{i}^{C}; E_{i}^{C}|X_{o}) = H(E_{i}^{C}|X_{o}) - H(E_{i}^{C}|E_{i}^{C}, X_{o}) = H(E_{i}^{C}|X_{o})$ . Substituting in the definition of CNF-2( $X_{i}, X_{j}$ ), result follows.

Symmetry: The result follows from the ‘symmetry’ property of mutual information.

Positivity: If $X_{i}, X_{j}$ are confounded, from the Assumption 4.1, $E_{i}^{C}, E_{j}^{C}$ are dependent random variables. Hence the mutual information is positive. The result follows after substituting some positive value for $I(E_{i}^{C}; E_{j}^{C})$ in the definition of CNF-2( $X_{i}, X_{j}$ ). The same argument goes for conditional confounding.

Monotonicity: from the definition of $CNF-2(X_{i}, X_{j})$ , $CNF-2(X_{i}, X_{j}) > CNF-2(X_{k}, X_{l})$ implies $I(E_{i}^{C}; E_{j}^{C}) > I(E_{k}^{C}; E_{l}^{C})$ . From the Defn. 4.2, $X_{i}, X_{j}$ have higher mutual information than the pair $X_{k}, X_{l}$ and hence $X_{i}, X_{j}$ are more strongly confounded than $X_{k}, X_{l}$ . ☐

Theorem 4.5. Let $\mathbf{X}_S$ be a set of variables such that all $X_i, X_j \in \mathbf{X}_S$ are pairwise confounded and the causal relationships among each pair $X_i, X_j$ . Then $\mathbf{X}_S$ is jointly confounded if and only if for each triple $X_i, X_j, X_k \in \mathbf{X}_S$ we have $I(E_{ij}^C; E_{jk}^C | E_j^C) < I(E_{ij}^C; E_{jk}^C)$ .

Proof. Following the Assumption 4.2, when three variables $X_{i}, X_{j}, X_{k}$ are confounded by as single confounding variable $Z$ , conditioning on $E_{k}^{C}$ explains away some of the dependency between $E_{ij}^{C}, E_{jk}^{C}$ . Hence we have $I(E_{ij}^{C}; E_{jk}^{C}|E_{j}^{C}) < I(E_{ij}^{C}; E_{jk}^{C})$ for all triples $i, j, k$ .

Theorem 4.6. For any three observed variables $X_{i}, X_{j}, X_{o}$ and an unobserved confounding variable $Z$ , the following statements are true for the measure CNF-3.

1. (Reflexivity and Symmetry.) $CNF - 3(X_i, X_i|X_o) = 1 - e^{-H(E_i^C |X_o)} \forall i$ where $H(.|.)$ denotes conditional entropy and $CNF - 3(X_i, X_j|X_o) = CNF - 3(X_j, X_i|X_o)$ .   
2. (Positivity.) $CNF-3(X_{i}, X_{j}) > 0$ if and only if $X_{i}, X_{j}$ are confounded. Given an observed confounding variable $X_{o}$ between $X_{i}, X_{j}$ , $CNF-3(X_{i}, X_{j}|X_{o}) > 0$ if and only if there exists an unobserved confounding variable Z between $X_{i}, X_{j}$ .   
3. (Monotonicity.) $CNF - 3(X_i, X_j) > CNF - 3(X_k, X_l)$ implies that the pair of variables $X_i, X_j$ are more strongly confounded than the pair of variables $X_k, X_l$ in the sense of Defn. 4.2.

Proof. Reflexivity: from the definition of mutual information, $I(E_{ii}^{C}; E_{i}^{C}|X_{o}) = I(E_{i}^{C}; E_{i}^{C}|X_{o}) = H(E_{i}^{C}|X_{o}) - H(E_{i}^{C}|E_{i}^{C}, X_{o}) = H(E_{i}^{C}|X_{o})$ . Substituting in the definition of CNF-3( $X_{i}, X_{j}$ ), result follows.

Symmetry: Since we rely on the direction of the causal path between $X_{i}, X_{j}$ , for a given pair of nodes $X_{i}, X_{j}$ , we have $CNF-3(X_{i}, X_{j}) = CNF-3(X_{j}, X_{i})$ from Defn. 4.7.

Positivity: If $X_{i}, X_{j}$ are confounded and $X_{i} \to X_{j}$ , from the Assumption 4.1, $E_{ji}^{C}, E_{j}^{C}$ are dependent random variables. Hence the mutual information $E_{ji}^{C}, E_{j}^{C}$ is positive. The result follows after substituting positive value for $I(E_{ji}^{C}; E_{j}^{C})$ in the definition of CNF-3( $X_{i}, X_{j}$ ). The same argument goes for conditional confounding.

Monotonicity: from the definition of $CNF-3(X_{i}, X_{j})$ , without loss of generality, $CNF-3(X_{i}, X_{j}) > CNF-3(X_{k}, X_{l})$ implies $I(E_{ji}^{C}; E_{j}^{C}) > I(E_{lk}^{C}; E_{l}^{C})$ . From the Defn. 4.2, $X_{i}, X_{j}$ have higher mutual information and hence are more strongly confounded than $X_{k}, X_{l}$ . ☐

# B Real-world Examples

![](images/e044ceab62d924df38d122977e07e214f2c5b9b29fbad0a94df8ad47aee78bb8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Pro"] --> B["Exp"]
    A --> C["Lab"]
    D["Edu"] --> E["Wag"]
    D --> F["Inv"]
```
</details>

Figure 4: Two real-world examples where our method can be applied. Here Pro: Production Volume, Exp: Exports, Lab: Total Labor Required, Edu: Education, Wag: Wages, Inv: Investments. We can perform interventions on the above variables and any combination thereof to obtain context-specific data. We can use such data to identify and measure confounding by applying our methods.