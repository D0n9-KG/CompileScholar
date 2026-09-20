# A VERSATILE CAUSAL DISCOVERY FRAMEWORK TO ALLOW CAUSALLY-RELATED HIDDEN VARIABLES

Xinshuai Dong $^{*1}$ Biwei Huang $^{*2}$ Ignavier Ng $^{1}$ Xiangchen Song $^{1}$ Yujia Zheng $^{1}$ Songyao Jin $^{4}$ Roberto Legaspi $^{3}$ Peter Spirtes $^{1}$ Kun Zhang $^{1,4}$

$^{1}$ Carnegie Mellon University   
$^{2}$ University of California San Diego   
$^{3}$ KDDI Research   
$^{4}$ Mohamed bin Zayed University of Artificial Intelligence

# ABSTRACT

Most existing causal discovery methods rely on the assumption of no latent confounders, limiting their applicability in solving real-life problems. In this paper, we introduce a novel, versatile framework for causal discovery that accommodates the presence of causally-related hidden variables almost everywhere in the causal network (for instance, they can be effects of observed variables), based on rank information of covariance matrix over observed variables. We start by investigating the efficacy of rank in comparison to conditional independence and, theoretically, establish necessary and sufficient conditions for the identifiability of certain latent structural patterns. Furthermore, we develop a Rank-based Latent Causal Discovery algorithm, RLCD, that can efficiently locate hidden variables, determine their cardinalities, and discover the entire causal structure over both measured and hidden ones. We also show that, under certain graphical conditions, RLCD correctly identifies the Markov Equivalence Class of the whole latent causal graph asymptotically. Experimental results on both synthetic and real-world personality data sets demonstrate the efficacy of the proposed approach in finite-sample cases. Our code will be publicly available.

# 1 INTRODUCTION AND RELATED WORK

Causal discovery aims at finding causal relationships from observational data and has received successful applications in many fields (Spirtes et al., 2000; 2010; Pearl, 2019). However, traditional methods, such as PC (Spirtes et al., 2000), GES (Chickering, 2002b), and LiNGAM (Shimizu et al., 2006b), generally assume that there are no latent confounders in the graph, which hardly holds in many real-world scenarios. Extensive efforts have been dedicated to addressing this issue for causal structure learning. One line of research focuses on inferring the causal structure among the observed variables, despite the possible existence of latent confounders. Notable approaches include FCI and its variants (Spirtes et al., 2000; Pearl, 2000; Colombo et al., 2012; Akbari et al., 2021) that leverage conditional independence tests, and over-complete ICA-based techniques (Hoyer et al., 2008; Salehkaleybar et al., 2020) that further leverage non-Gaussianity.

Another line of thought focuses more on uncovering the causal structure among latent variables, by assuming observed variables are not directly adjacent. This includes Tetrad condition-based (Silva et al., 2006; Kummerfeld & Ramsey, 2016), high-order moments-based (Shimizu et al., 2009; Cai et al., 2019; Xie et al., 2020; Adams et al., 2021; Chen et al., 2022), matrix decomposition-based (Anandkumar et al., 2013), and mixture oracles-based (Kivva et al., 2021) approaches. Recently, Huang et al. (2022) propose an approach that makes use of rank constraints to identify general latent hierarchical structures, and yet observed variables can only be leaf nodes. Although Chandrasekaran et al. (2012) allow direct causal influences within observed variables and the existence of latent variables, it cannot recover the causal relationships among latent variables and has strong graphical constraints. For a more detailed discussion of related work, please refer to our Appx. C.1.

![](images/ea72ba9190d2d36a00aeec5bc6d37df2d6dace57a49d7a63e2349430e912720e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> X1
    L2 --> X2
    L2 --> L3
    L2 --> X3
    L2 --> X4
    L2 --> X5
    X2 --> X6
    X2 --> X7
    X2 --> X8
    X2 --> X9
    L3 --> X10
    L3 --> X11
    L3 --> X12
    L3 --> X13
    L3 --> X14
    L3 --> X15
    L3 --> X16
    X3 --> L4
    X4 --> L5
```
</details>

Figure 1: An illustrative graph that we aim to handle, where latent variables are denoted by L and observed variables are denoted by X. The latent variables can act as a cause, effect, or mediator for both observed variables and other latent variables. See Appx. B.1 for a comparison of graphs that each method can handle.

In this paper, we aim to handle a more general scenario for causal discovery with latent variables, where observed variables are allowed to be directly adjacent, and latent variables to be flexibly related to all the other variables. That is, hidden variables can serve as confounders, mediators, or effects of latent or observed variables, and even form a hierarchical structure (see an illustrated example in Figure 1). This setting is rather general and practically meaningful to deal with many real-world problems.

To address such a challenging problem, we are confronted with three fundamental questions: (i) What information and constraints can be discovered from the observed variables to reveal the underlying causal structure? (ii) How can we effectively and efficiently search for these constraints? (iii) What graphical conditions are needed to uniquely locate latent variables and ascertain the complete causal structure? Remarkably, these questions can be addressed by harnessing the power of rank deficiency constraints on the covariance of observed variables. By carefully identifying and utilizing rank properties in specific ways, we are able to determine the Markov equivalence class of the entire graph. Our contributions are mainly three-fold:

- We investigate the efficacy of rank in comparison to conditional independence in latent causal graph discovery, and theoretically introduce necessary and sufficient conditions for the identifiability of certain latent structural properties. For instance, the condition we proposed for nonadjacency generalizes the counterpart in Spirtes et al. (2000) to graphs with latent variables.   
- We develop RLCD, an efficient three-phase causal discovery algorithm that is able to locate latent variables, determine their cardinalities, and identify the whole causal structure involving measured and latent variables, by properly leveraging rank properties. In the special case with no latent variables, it asymptotically returns the same graph as the PC algorithm (Spirtes et al., 2000) does.   
- We provide a set of graphical conditions that are sufficient for RLCD to asymptotically identify the correct Markov Equivalence Class of the latent causal graph; notably, these graphical conditions are significantly weaker than those in previous works. Our empirical study on both synthetic and real-world datasets validates RLCD on finite samples.

# 2 PROBLEM SETTING

In this paper, we aim to identify the causal structure of a latent linear causal model defined as follows.

Definition 1. (Latent Linear Causal Models) Suppose a directed acyclic graph $\mathcal{G} := (\mathbf{V}_{\mathcal{G}}, \mathbf{E}_{\mathcal{G}})$ , where each variable $V_i \in \mathbf{V}_{\mathcal{G}}$ is generated following a linear causal structural model:

$$
\mathsf {V} _ {i} = \sum_ {\mathsf {V} _ {j} \in P a _ {\mathcal {G}} (\mathsf {V} _ {i})} a _ {i j} \mathsf {V} _ {j} + \varepsilon_ {\mathsf {V} _ {i}}, \tag {1}
$$

where $V_{G} := L_{G} \cup X_{G}$ contains a set of $n + m$ random variables, with m latent variables $L_{G} := \{L_{i}\}_{i=1}^{m}$ , and n observed variables $X_{G} := \{X_{i}\}_{i=1}^{n}$ . $Pa_{\mathcal{G}}(V_{i})$ denotes the parent set of $V_{i}$ , $a_{ij}$ the causal coefficient from $V_{j}$ to $V_{i}$ , and $\varepsilon_{V_{i}}$ represents the noise term.

We further have a basic assumption for latent linear causal models given as follows.

Assumption 1 (Basic Assumptions for Latent Linear Causal Models). (i) Leaf nodes are observed; or equivalently, a latent variable should have at least one observed descendant. (ii) Rank faithfulness. A probability distribution p is rank faithful to G if every rank constraint on a sub-covariance matrix that holds in p is entailed by every linear structural model with respect to G.

Table 1: Graphical notations used throughout this paper. 

<table><tr><td>Pa: Parents</td><td>V: Variables</td><td>V: Variable</td><td>G: The underlying graph</td></tr><tr><td>Ch: Children</td><td>L: Latent variables</td><td>L: Latent variable</td><td> $\mathcal{G}'$ : Output Graph</td></tr><tr><td>PCh: Pure children</td><td>X: Observed variables</td><td>X: Observed variable</td><td>S: Set of covers</td></tr><tr><td>Sib: Siblings</td><td>MDe: Measured descendants</td><td>PDe: Pure descendants</td><td>S: Set of sets of covers</td></tr></table>

The inclusion of Assumption 1 does not compromise the generality. If (i) does not hold, we can simply remove the latent variables that lack observed descendants, since they provide no information that can be inferred for any other variable. Further, (ii) is the classical faithfulness assumption that is critical and prevalent in causal discovery (Spirtes et al., 2000; Huang et al., 2022); it holds generically on infinite data, as the set of values of the SCM's free parameters for which rank is not faithful is of Lebesgue measure 0 (Spirtes, 2013). On the other hand, if faithfulness is violated (an example in Appx. B.2), even classical methods like PC cannot guarantee asymptotic correctness.

Our objective is to identify the underlying causal structure $\mathcal{G}$ over all the variables $\mathbf{L}_{\mathcal{G}} \cup \mathbf{X}_{\mathcal{G}}$ (detailed in Sec. 5) that are generated according to a latent linear causal model, given i.i.d. samples of observed variables $\mathbf{X}_{\mathcal{G}}$ only. To address this challenging problem, traditional wisdom often relies on strong graphical constraints (Pearl, 1988; Zhang, 2004; Huang et al., 2022; Maeda & Shimizu, 2020)(detailed in Appx. B.1 with illustrative graphs). In contrast, Definition 1 allows all the variables including observed and latent variables to be very flexibly related. We basically allow the presence of edges between any two variables such that a node $V$ , no matter whether it is observed or not, can act as a cause, effect, or mediator for both observed and latent variables.

A summary of notations is in Tab. 1. The rest of the paper is organized as follows. In Sec. 3, we motivate the use of rank and propose conditions for nonadjacency and the existence of latent variables. In Sec. 4, we establish the minimal identifiable substructure of a linear latent graph, based on which we propose RLCD for latent variable causal discovery. In Sec. 5, we introduce the identifiability of RLCD. In Sec. 6, we validate our method using both synthetic and real-life data.

# 3 WHY USE RANK INFORMATION?

In this section, we first motivate the use of rank constraints for causal discovery in the presence of latent variables, and then establish some fundamental theories about what rank implies graphically.

# 3.1 PRELIMINARIES ABOUT TREKS AND RANK

When there is no latent variable, a common approach for causal discovery is to use conditional independence (CI) relationships to identify d-separations in a graph; see, e.g., the PC algorithm (Spirtes et al., 2000). The following theorem illustrates this idea.

Theorem 1 (Conditional Independence and D-separation (Pearl, 1988)). Under the Markov and faithfulness assumption, for disjoint sets of variables A, B and C, C d-separates A and B in graph G, iff A ⊥ B|C holds for every distribution in the graphical model associated to G.

As for latent linear causal models, trek-separations (t-separations) provide more information than d-separations (for readers who are not very familiar with treks and t-separations, kindly refer to Appx. A.2 for examples). The definitions of treks and t-separation are given as follows, together with Theorem 2 showing the relations between t-separations and d-separations.

Definition 2 (Treks (Sullivan et al., 2010)). In G, a trek from X to Y is an ordered pair of directed paths $(P_{1}, P_{2})$ where $P_{1}$ has a sink X, $P_{2}$ has a sink Y, and both $P_{1}$ and $P_{2}$ have the same source Z.

Definition 3 (T-separation (Sullivan et al., 2010)). Let A, B, $C_{A}$ , and $C_{B}$ be four subsets of $V_{G}$ in graph G (not necessarily disjoint). $(\mathbf{C}_{\mathbf{A}}, \mathbf{C}_{\mathbf{B}})$ t-separates A from B if for every trek $(P_{1}, P_{2})$ from a vertex in A to a vertex in B, either $P_{1}$ contains a vertex in $C_{A}$ or $P_{2}$ contains a vertex in $C_{B}$ .

Theorem 2 (T- and D-sep (Di, 2009)). For disjoint sets A, B and C, C d-separates A and B in graph G, iff there is a partition $C = C_{A} \cup C_{B}$ such that $(\mathbf{C}_{\mathbf{A}}, \mathbf{C}_{\mathbf{B}})$ t-separates $A \cup C$ from $B \cup C$ .

![](images/84911331e7db1d20afd74fe43334d47a0de1d807f0431839f8c02eab772aab16.jpg)  
(a) $G_{1}$ .

![](images/80bcf005b2ed29a970e85f5242f5849636fc848fc5436e73e78d629e1cc4f036.jpg)  
(b) CI skeleton of $\mathcal{G}_1$ .

![](images/4cae23ff9776cef43a3ab23d597c6620bb0ce72995d81e93796e4e7c7c30b7e1.jpg)  
(c) $\mathcal{G}_2$

![](images/421bd1556a5a73195af28cb5219c552664e9ccb5679b4e0fca7983aa56c0c37b.jpg)  
(d) CI skeleton of $\mathcal{G}_2$ .

![](images/9fe170be922b82e3221b9bd8eb35b5f710c945fbc877d42908667d67d3a64cb2.jpg)  
(e) $\mathcal{G}_3$ .

![](images/c74f1aa010bcbb247a9e15456e80efbc8609507120d2d1317ad277dbe13e8798.jpg)  
(f) CI skeleton of $\mathcal{G}_3$ .   
Figure 2: Examples that illustrate the basic intuition for a latent to be identifiable.

The theorem above reveals that all d-sep can be reformulated by t-sep, and thus, t-sep encompass d-sep information. Just as we use CI tests to find d-sep, t-sep can be identified by the rank of cross-covariance matrix over specific combinations of variables, which is formally stated as follows.

Theorem 3 (Rank and T-separation (Sullivan et al., 2010)). Given two sets of variables A and B from a linear model with graph G, we have rank( $\Sigma_{A,B}$ ) = min{|C\_A| + |C\_B| : (C\_A, C\_B) t-separates A from B in G}, where $\Sigma_{A,B}$ is the cross-covariance over A and B.

With some abuse of notation, sometimes we also use $\Sigma_{A,B}$ to refer to cross-covariance over sets of sets (see Appx. A.1 for definition and examples). Notably, when all variables are observed, rank and conditional independence are equally informative about the underlying DAG. However, in the presence of latent variables, t-separations, which can be inferred from rank by Theorem 3, offer more graphical information compared to d-separations. Therefore, we next demonstrate how rank constraints play a pivotal role in identifying latent causal structures.

# 3.2 RANK: AN INFORMATIVE GRAPHICAL INDICATOR FOR LATENT VARIABLES

In the presence of latent variables, CI is not enough: FCI and its variants make full use of CI but only recover a representation which is not informative enough about the latent confounders. Fortunately, leveraging rank information can naturally make causal discovery results more informative.

An example highlighting the greater informativeness of rank compared to CI is as follows. Consider the graph $G_{1}$ in Fig. 7, where $\{X_{1}, X_{2}\}$ and $\{X_{3}, X_{4}\}$ are d-separated by $L_{1}$ , but we cannot infer that from a CI test (i.e., whether $\{X_{1}, X_{2}\} \perp \perp \{X_{3}, X_{4}\}|L_{1}$ ), as $L_{1}$ is not observed. In contrast, with rank information, we can infer that $\text{rank}(\Sigma_{\{X_{1}X_{2}\},\{X_{3}X_{4}\}})=1$ , which implies $\{X_{1}, X_{2}\}$ and $\{X_{3}, X_{4}\}$ are t-separated by one latent variable. The rationale behind is that the t-sep of A, B by $(\mathbf{C}_{\mathbf{A}}, \mathbf{C}_{\mathbf{B}})$ can be deduced through rank (as in Theorem 3) without observing any element in $(\mathbf{C}_{\mathbf{A}}, \mathbf{C}_{\mathbf{B}})$ .

With this intuition in mind, below we present three theorems that characterize the graphical implications of rank constraints, in scenarios where latent variables might exist: Theorem 4 gives conditions for observed variables to be nonadjacent, illustrated by Example 6; Theorem 5 gives conditions for the existence of latent variables, illustrated by Example 1; Theorem 6 implies how to utilize pure children as surrogates for calculating rank, illustrated with Example 7. All proofs are in Appendix.

Theorem 4 (Condition for Nonadjacency). Consider a latent linear causal model. Two observed variables $\mathsf{X}_1, \mathsf{X}_2 \in \mathbf{X}_{\mathcal{G}}$ are not adjacent, if there exist two sets $\mathbf{A}, \mathbf{B} \subseteq \mathbf{X}_{\mathcal{G}} \setminus \{\mathsf{X}_1, \mathsf{X}_2\}$ that are not necessarily disjoint, such that $\text{rank}(\Sigma_{\mathbf{A} \cup \{\mathsf{X}_1\}, \mathbf{B} \cup \{\mathsf{X}_2\}}) = \text{rank}(\Sigma_{\mathbf{A}, \mathbf{B}})$ and $\text{rank}(\Sigma_{\mathbf{A} \cup \{\mathsf{X}_1, \mathsf{X}_2\}, \mathbf{B} \cup \{\mathsf{X}_1, \mathsf{X}_2\}}) = \text{rank}(\Sigma_{\mathbf{A}, \mathbf{B}}) + 2$ .

Remark 1. Theorem 4 presents a sufficient condition for determining nonadjacency between two observed variables. In the absence of latent variables, this condition transforms into a necessary and sufficient one. Note that A and B may have overlapping variables. SGS and PC (Spirtes et al., 2000) also introduced a necessary and sufficient condition for determining two variables not being adjacent when latent variables do not exist: there exist a set of observed variables $\mathbf{C} \subseteq \mathbf{X}_{\mathcal{G}}$ , $X_1, X_2 \notin \mathbf{C}$ , such that $X_1 \perp X_2|\mathbf{C}$ . Interestingly, this condition can be expressed in the form of Theorem 4, with $\mathbf{C} = \mathbf{A} = \mathbf{B}$ ; thus Theorem 4 generalizes PC's condition to scenarios where latent variables may be present (this claim is detailed in Appx. A.8).

Theorem 5 (Condition for Existence of Latent Variable). Suppose a latent linear causal model with graph $\mathcal{G}$ and observed variables $\mathbf{X}_{\mathcal{G}}$ . If there exist three disjoint sets of variables $\mathbf{A},\mathbf{B},\mathbf{C} \subseteq \mathbf{X}_{\mathcal{G}}$ , such that (i) $|\mathbf{B}| \geq |\mathbf{A}| \geq 2$ , (ii) $\forall$ distinct $\mathsf{A}_1, \mathsf{A}_2 \in \mathbf{A}$ , $\mathsf{A}_1, \mathsf{A}_2$ are adjacent in the CI skeleton over $\mathbf{X}_{\mathcal{G}}$ with CI skeleton defined in Appx. A.3), (iii) $\forall \mathsf{A} \in \mathbf{A}$ , $\mathsf{B} \in \mathbf{B}$ , $\mathsf{A}$ , $\mathsf{B}$ are adjacent in the CI skeleton over $\mathbf{X}_{\mathcal{G}},(iv)$ $\mathbf{C} \subseteq \{\mathsf{X} : \exists \mathsf{Y} \in \mathbf{A} \cup \mathbf{B}\text{s.t.} \mathsf{X},\mathsf{Y}\text{ are adjacent in the CI skeleton}\}$ (i.e., all elements in

C are neighbours of an element in $A \cup B$ in the CI skeleton), (v) $\text{rank}(\Sigma_{\mathbf{A} \cup \mathbf{C}, \mathbf{B} \cup \mathbf{C}}) < |\mathbf{A}| + |\mathbf{C}|$ , then there must exist at least one latent variable in one of the treks between A, B.

Remark 2. Theorem 5 provides a sufficient condition for determining the existence of latent variables. The underlying intuition is that, in the absence of latent variables, rank information should align with what CI skeleton provides; if not, then there must exist at least one latent variable. Furthermore, we will show in Theorem 9 that, with further graphical constraints the condition in Theorem 5 becomes both necessary and sufficient.

Moreover, it can be shown that observed children, or even descendants of latent variables can be used as surrogates to calculate rank as stated in Theorem 6 with the definition of pure children below.

Definition 4 (Pure Children). Y are pure children of variables X in graph G, iff $Pa_{\mathcal{G}}(\mathbf{Y}) = \cup_{Y_i \in \mathbf{Y}} Pa_{\mathcal{G}}(Y_i) = \mathbf{X}$ and $X \cap Y = \emptyset$ . We denote the pure children of X in G by $PCh_{\mathcal{G}}(\mathbf{X})$ .

Theorem 6 (Pure Children as Surrogate for Rank Estimation). Let $\mathbf{C} \subseteq PCh_{\mathcal{G}}(\mathbf{A})$ be a subset of pure children of $\mathbf{A}$ , and $\mathbf{B}$ be a set of variables such that for all $\mathsf{B} \in \mathbf{B}$ , $\mathsf{B} \notin De_{\mathcal{G}}(\mathbf{C})$ . We have rank $(\Sigma_{\mathbf{A},\mathbf{B}}) \geq \text{rank}(\Sigma_{\mathbf{C},\mathbf{B}})$ . Moreover, if rank $(\Sigma_{\mathbf{A},\mathbf{C}}) = |\mathbf{A}|$ , then rank $(\Sigma_{\mathbf{A},\mathbf{B}}) = \text{rank}(\Sigma_{\mathbf{C},\mathbf{B}})$ .

Remark 3. Theorem 6 informs us that under certain conditions, we can estimate the rank of covariance involving latent variables by using their pure children as surrogates. Even when the pure children are not observed one can recursively examine the children's children until reaching observed descendants (defined in Appx. A.1). This enables us to deduce graphical information associated with latent variables through the use of observed ones as surrogates.

# 4 DISCOVERING LATENT STRUCTURE THROUGH RANK CONSTRAINTS

In this section, we begin with the concept of atomic cover and explore its rank deficiency, and then develop an efficient algorithm based on rank-deficiency for latent causal discovery, as in Alg. 1.

# 4.1 ATOMIC COVER AND RANK DEFICIENCY

Below, we introduce atomic covers and their associated rank deficiency properties, which allows us to define the minimal identifiable substructure of a graph. We start with an example in Figure 2 that motivates the conditions for a latent variable to be identifiable.

Example 1. Consider $\mathcal{G}_1$ in Fig. 2 (a). It can be shown that the latent variable $\mathsf{L}_1$ in $\mathcal{G}_1$ is not identifiable. We can easily find a graph $\mathcal{G}_1'$ with no latent variable, e.g., as in Fig. 10 (b), such that $\mathcal{G}_1'$ shares the same skeleton as Fig. 2 (a), but all the observational rank information entailed by $\mathcal{G}_1$ is the same as by $\mathcal{G}_1'$ - they are indistinguishable. However, if $\mathsf{X}_4$ becomes the children of $\mathsf{L}_1$ , as in $\mathcal{G}_2$ given in Fig. 2 (c), the whole structure becomes identifiable. Specifically, the conditions in Theorem 5 hold when we take $\mathbf{A} = \{\mathsf{X}_1, \mathsf{X}_2\}$ , $\mathbf{B} = \{\mathsf{X}_3, \mathsf{X}_4\}$ , and $\mathbf{C} = \emptyset$ , which informs the existence of latent variables. The same conditions also hold for Fig. 2 (e) and thus $\mathsf{L}_1$ in $\mathcal{G}_3$ is also identifiable, though $\mathsf{L}_1$ has only two children together with another two neighbors.

The intuition is that, for a latent variable to be identifiable, it should have enough children and enough neighbors. We next formalize this intuition into the concept of atomic cover as the minimal identifiable unit in a graph. The formal definition of an atomic cover is given in Definition 5, where effective cardinality of a set of covers V is defined as $||V|| = |(\cup_{\mathbf{V} \in \mathcal{V}} \mathbf{V})|$ .

Definition 5 (Atomic Cover). Let $\mathbf{V}$ be a set of variables in $\mathcal{G}$ with $|\mathbf{V}| = k$ , where $t$ of the $k$ variables are observed, and the remaining $k - t$ are latent. $\mathbf{V}$ is an atomic cover if $\mathbf{V}$ contains a single observed variable (i.e., $k = t = 1$ ), or if the following conditions hold:

(i) There exists a set of atomic covers $\mathcal{C}$ , with $||\mathcal{C}|| \geq k + 1 - t$ , such that $\cup_{\mathbf{C} \in \mathcal{C}} \mathbf{C} \subseteq PCh_{\mathcal{G}}(\mathbf{V})$ and $\forall \mathbf{C}_1, \mathbf{C}_2 \in \mathcal{C}, \mathbf{C}_1 \cap \mathbf{C}_2 = \emptyset$ .   
(ii) There exists a set of covers $\mathcal{N}$ , with $||\mathcal{N}|| \geq k + 1 - t$ , such that every element in $\cup_{\mathbf{N} \in \mathcal{N}} \mathbf{N}$ is a neighbour of $\mathbf{V}$ and $(\cup_{\mathbf{N} \in \mathcal{N}} \mathbf{N}) \cap (\cup_{\mathbf{C} \in \mathcal{C}} \mathbf{C}) = \emptyset$ .   
(iii) There does not exist a partition of $V = V_{1} \cup V_{2}$ such that both $V_{1}$ and $V_{2}$ are atomic covers.

In the definition above, each observed variable is treated as an atomic cover, e.g., $\{\mathsf{X}_1\}$ in Figure 1. We define the minimal identifiable unit as an atomic cover with the rationale that, when two or more

latent variables share exactly the same set of neighbors (e.g., $L_{1}$ and $L_{2}$ in Fig. 6), they can never be distinguished from observational information. Hence, it is more convenient and unified to consider them together in an atomic cover. Examples of atomic covers can be found in Appx. B.3.

Based on the definition of atomic covers, we define a cluster as the set of pure children of an atomic cover, and refer k-cluster to a cluster whose parents' cardinality is k. We further define an operator $\operatorname{Sep}(\mathbf{X}) = \cup_{\mathsf{X} \in \mathbf{X}} \{\{\mathsf{X}\}\}$ . We next show that every atomic cover possesses a useful rank deficiency property, which is formally stated in the following theorem (proof is given in Appx. A.11).

Theorem 7 (Rank Deficiency of an Atomic Cover). Let $\mathbf{V} = \mathbf{X} \cup \mathbf{L}$ be an atomic cover where $|\mathbf{X}| = t$ variables are observed and $|\mathbf{L}| = k - t$ are latent. Let $\mathcal{X} = \text{Sep}(\mathbf{X})$ , $\mathcal{X}_{\mathcal{G}} = \text{Sep}(\mathbf{X}_{\mathcal{G}})$ , and a set of atomic covers $\mathcal{C}$ , satisfying (i) in Definition 5. Then $\text{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, \mathcal{X} \cup \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus MDe_{\mathcal{G}}(\mathcal{C})}) = k$ and $k < \min(||\mathcal{C} \cup \mathcal{X}||, ||\mathcal{X} \cup \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus MDe_{\mathcal{G}}(\mathcal{C})||)$ ( $MDe_{\mathcal{G}}$ denotes measured descendants).

Example 2 (Example for atomic cover and rank deficiency). Consider an atomic cover in Fig. 4 (d): $\mathbf{V} = \{\mathsf{X}_2, \mathsf{L}_2\}$ . $\mathbf{V}$ is an atomic cover, because $\mathbf{V}$ has at least 2 pure children and has additional 3 neighbors, satisfying the conditions in Definition 5. If we take $\mathcal{C} = \{\{\mathsf{X}_4\}, \{\mathsf{X}_5\}\}$ , and $\mathcal{X} = \{\{\mathsf{X}_2\}\}$ , we have $\text{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, \mathcal{X} \cup \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus MDe_{\mathcal{G}}(\mathcal{C})}) = \text{rank}(\Sigma_{\{\mathsf{X}_4, \mathsf{X}_5, \mathsf{X}_2\}, \{\mathsf{X}_1, \mathsf{X}_2, \mathsf{X}_3, \mathsf{X}_6, \mathsf{X}_7, \mathsf{X}_8\}) = 2 = |\mathbf{V}|$ . Noted that both $||\mathcal{C} \cup \mathcal{X}||$ , $||\mathcal{X} \cup \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus MDe_{\mathcal{G}}(\mathcal{C})|| > 2$ , so here the rank is deficient.

Theorem 7 establishes the rank-deficiency property of an atomic cover. Furthermore, if we can build a unique connection between rank deficiency and atomic covers under certain conditions, then we can exploit rank deficiency to identify atomic covers in a graph. In the following, we present the graphical conditions to achieve identifiability and Theorem 8 delineates under what conditions the uniqueness of the rank-deficiency property can be ensured.

Condition 1 (Basic Graphical Conditions for Identifiability). A graph G satisfies the basic graphical condition for identifiability, if $\forall L \in L_{G}$ , L belongs to at least one atomic cover in G and no latent variable is involved in any triangle structure (whose definition is in Appx. D.3).

Theorem 8 (Uniqueness of Rank Deficiency). Suppose a graph G satisfies Condition 1. We further assume (i) all the atomic covers with cardinality $k' < k$ have been discovered and recorded, and (ii) there is no collider in G. If there exists a set of observed variables X and a set of atomic covers C satisfying $\mathcal{X} = \text{Sep}(\mathbf{X})$ , $C \cap X = \emptyset$ , and $||C|| + ||X|| = k + 1$ , such that (i) For all recorded $k'$ cluster $C'$ , $||C \cap C'|| \leq |Pa_{G}(C')|$ , (ii) $\text{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, \mathcal{X} \cup \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus MDe_{\mathcal{G}}(\mathcal{C})) = k$ , then there exists an atomic cover $V = L \cup X$ in G, with $X = \cup_{X' \in X} X'$ , $|L| = k - |X|$ , and $\cup_{C \in C} C \subseteq PCh_{G}(V)$ .

For a better understanding, we provide an illustrative example in Appx. A.12. Basically, Theorem 8 says that under certain conditions, we can build a unique connection between rank deficiency and atomic covers, and thus we can identify atomic covers in a graph by searching for combinations of $\mathcal{C}$ and $\mathcal{X}$ that induce rank deficient property. We further introduce Theorem 9, which is useful in that it provides necessary and sufficient conditions for the existence of latent variables, under Condition 1.

Theorem 9 (Necessary and Sufficient Condition for Existence of Latent Variable). If a graph satisfies Condition 1, then the sufficient condition for the existence of latent variables in Theorem 5 becomes both necessary and sufficient. That is, the "if" in Theorem 5 becomes "if and only if".

With the theoretical guarantees of Theorem 8 and Theorem 9, the next question is how to design a search procedure that strives to fulfill these conditions, in order to cash out the theorems for existence of latent variable and the uniqueness of rank-deficiency, for identifying atomic covers in a graph, and consequently the whole latent causal structure.

# 4.2 METHOD - RANK-BASED LATENT CAUSAL DISCOVERY

In this section, we propose a computationally efficient and scalable search algorithm to identify the latent causal structure, referred to as Rank-based Latent Causal Discovery (RLCD), that leverages the connection between graph structures and rank deficiency, as we discussed in previous sections. The search process mainly comprises three phases: (1) Phase 1: FindCISkeleton, (2) Phase 2: Find-CausalClusters, and (3) Phase 3: RefineCausalClusters, as outlined in Alg. 1.

Specifically, Phase 1 is to find the CI skeleton deduced by conditional independence tests over $\mathbf{X}_{\mathcal{G}}$ , and Phase 2 and Phase 3 are designed such that the conditions in Theorem 8 are satisfied to the largest extent, in order to make use of the unique rank deficiency property to identify the latent structure.

![](images/57bd8594e9a3a2a4976bf3945c6ea4e22e2e51d87608e15057c342901d720449.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> X1
    L1 --> L2
    L1 --> X2
    L1 --> X3
    X1 --> X4
    X1 --> X5
    X2 --> X6
    X2 --> X7
    X3 --> X8
    X3 --> X9
    X3 --> X10
    X3 --> X11
    X3 --> X12
    L3 --> X9
    L3 --> X10
    L3 --> X11
    L3 --> X12
```
</details>

(a) The underlying graph $\mathcal{G}$ , all the observed variables of which are taken as input to Phase 1.   
![](images/d3cfaa00f2cdaf0a7c46bbef5e9340c71d6bf784ae4f01e53fefffa70214f3f0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> X1
    L1 --> L2
    L1 --> X2
    L1 --> X3
    X1 --> X4
    X2 --> X5
    X2 --> X6
    X3 --> X7
    X4 --> X8
    X5 --> X6
    X6 --> X7
    X7 --> X8
    X8 --> X9
    X8 --> X10
    X8 --> X11
    X8 --> X12
```
</details>

(c) Take variables from the first dashed area to Phase 2 and 3, use the result to update the CI skeleton.

![](images/8121a13d83a7aa9c5c3aa44ca5f1a54f5004a5edccf57ec274647f5147146cfd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1["X₁"] --> X2["X₂"]
    X4["X₄"] --> X5["X₅"] --> X6["X₆"] --> X7["X₇"]
    X2 --> X3["X₃"]
    X8["X₈"] --> X9["X₉"] --> X10["X₁₀"] --> X11["X₁₁"] --> X12["X₁₂"]
    X8 -.-> X9
    X8 -.-> X10
    X8 -.-> X11
    X8 -.-> X12
```
</details>

(b) Given the CI skeleton from Phase 1, variables are partitioned into two groups, shown in the two dashed areas.   
![](images/b0833986602d85bee31309e4d837c14091a175e4ac8059e10fba82f14900e84d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> X1
    L1 --> L2
    L1 --> X2
    L1 --> X3
    X1 --> X4
    X2 --> X5
    X2 --> X6
    X3 --> X7
    X3 --> X8
    X4 --> L3
    X5 --> L3
    X6 --> L3
    X7 --> L3
    X8 --> L3
    L3 --> X9
    L3 --> X10
    L3 --> X11
    L3 --> X12
```
</details>

(d) Do the same thing as in (c) for the second dashed area and update all the remaining directions.

Figure 3: An illustrative example of the overall search procedure in Alg. 1.   
![](images/e72a43261a479c8215fc096aefdc6db3447ae2a2823d4628900b745fe46cf37f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
```mermaid
graph TD
    subgraph (a)
        A1["X₁"] --> B1["X₂"]
        A2["X₄"] --> B2["X₅"]
        A3["X₆"] --> B3["X₇"]
        A4["X₇"] --> B4["X₈"]
    end

    subgraph (b)
        A5["X₁"] --> B5["X₂"]
        A6["X₄"] --> B6["X₅"]
        A7["X₆"] --> B7["X₇"]
        A8["X₈"] --> B8["X₃"]
    end

    subgraph (c)
        A9["X₁"] --> B10["L₂"]
        A10["X₄"] --> B11["L₂"]
        A12["X₅"] --> B12["X₂"]
        A13["X₇"] --> B13["X₃"]
        A14["X₈"] --> B14["X₄"]
        A15["X₆"] --> B16["X₅"]
        A16["X₇"] --> B17["X₆"]
    end

    subgraph (d)
        A10 --> B18["L₁"]
        A11 --> B19["L₂"]
        A12 --> B20["X₃"]
        A13 --> B21["X₄"]
        A14 --> B22["X₅"]
        A15 --> B23["X₆"]
        A16 --> B24["X₇"]
        A17 --> B25["X₈"]
        A18 --> B26["X₈"]
    end

    note1["Take 𝒳 = {{X₃}} and 𝒔 = {{X₈}}."]
    note2["Take 𝒳 = {{X₂,X₃}} and 𝒔 = {{X₇}},{X₈}."]
```
</details>

Figure 4: An illustrative example of the process of Phase 2 in Alg. 3.

We initiate the process by finding the CI skeleton first, for the reason as follows. According to Theorem 9, latent variables exist iff conditions (i)-(v) in Theorem 5 are satisfied, while conditions (i)-(iv) can be directed inferred from the CI skeleton. Therefore, there is no need to consider all variables in $\mathbf{X}_{\mathcal{G}}$ as inputs for Phases 2 and 3. Instead, we make use of the CI skeleton over $\mathbf{X}_{\mathcal{G}}$ and select some groups of observed variables as inputs into Phases 2 and 3, where variables in each group together have the potential to satisfy conditions (i)-(iv). This also benefits the computational efficiency as the Phase 2 and 3 for different groups can be done in parallel.

An example of the entire search process is illustrated in Fig. 3. After Phase 1, variables are partitioned into groups as shown in Fig. 3 (b). For each group (dashed area in Fig. 3 (b)), we conduct Phases 2 and 3, and have the final result shown in Fig. 3 (d). Details of each phase are given below.

# 4.3 PHASE 1: FINDING CI SKELETON

The objective of Phase 1 is to find the CI skeleton over observed variables $\mathbf{X}_{\mathcal{G}}$ by utilizing conditional independence relations. To this end, we employ the first stage of the PC algorithm (Spirtes et al., 2000), with the difference that we replace all CI tests with rank tests, according to the following Lemma 10 (proof in Appx. A.15).

Lemma 10 (D-separation by Rank Test). Suppose a linear latent causal model with graph $\mathcal{G}$ . For disjoint $\mathbf{A},\mathbf{B},\mathbf{C}\in \mathbf{X}_{\mathcal{G}}$ , $\mathbf{C}$ $d$ -separates $\mathbf{A}$ and $\mathbf{B}$ in graph $\mathcal{G}$ , if and only if $\mathrm{rank}(\Sigma_{\mathbf{A}\cup \mathbf{C},\mathbf{B}\cup \mathbf{C}) = |\mathbf{C}|$ .

We summarize the procedure of Phase 1 in Alg. 2 in the appendix. Although, asymptotically, using CI and rank information will provide the same d-separation result over observed variables, we use rank instead of CI in Phase 1 just for the purpose of having a unified causal discovery framework with rank constraints (as Phases 2 and 3 are also based on rank).

Given the CI skeleton $\mathcal{G}'$ (result from Phase 1), the next step is to find the substructures in $\mathcal{G}'$ that might contain latent variables. Specifically, Theorem 9 informs us that latent variable exists iff (i)-(v) in Theorem 5 holds, where (i)-(iv) can be directly inferred from the CI skeleton. Specifically, we consider all the maximal cliques $\mathbf{Q}$ in $\mathcal{G}'$ (a clique is a set of variables that are fully connected

Algorithm 1: The overall procedure for Rank-based Latent Causal Discovery (RLCD).   
Input : Samples from all n observed variables $X_{G}$ Output: Markov equivalence class $G'$ def LatentVariableCausalDiscovery( $X_{G}$ ):
    Phase 1: $G' = \text{FindCISkeleton}(X_{G})$ (Algorithm 2);
    for Each Q, a group of overlapping maximal cliques, in $G'$ do
    Set an empty graph $G''$ , $X_{Q} = \cup_{Q \in Q} Q$ , $N_{Q} = \{N : \exists X \in X_{Q} \text{ s.t. } N, X \text{ are adjacent in } G'\}$ ;
    Phase 2: $G'' = \text{FindCausalClusters}(G'', X_{Q} \cup N_{Q})$ (Algorithm 3);
    Phase 3: $G'' = \text{RefineCausalClusters}(G'', X_{Q} \cup N_{Q})$ (Algorithm 5);
    Transfer the estimated DAG $G''$ to the Markov equivalence class and update $G'$ by $G''$ ;
    Orient remaining causal directions that can be inferred from v structures;
    return $G'$

Algorithm 2: Phase1: FindCISkeleton (Stage 1 of PC (Spirtes et al., 2000))   
Input : Samples from n observed variables $X_{G}$ Output: CI skeleton $G'$ def Stage1PC( $X_{G}$ ):
    Initialize a complete undirected graph $G'$ on $X_{G}$ ;
    repeat

    repeat
    Select an ordered pair X, Y that are adjacent in $G'$ , s.t., $|Adj_{G'}(X)\setminus\{Y\}| \geq n$ ;
    Select a subset $S \subseteq Adj_{G'}(X)\setminus\{Y\}$ s.t., $|S| = n$ ;
    If rank( $\Sigma_{\{X\}\cup C,\{Y\}\cup C}$ ) = |C|, delete the edge between X and Y from $G'$ and record S in Sepset(X, Y) and Sepset(Y, X);
    until all X, Y s.t., $|Adj_{G'}(X)\setminus\{Y\}| \geq n$ and all $S \subseteq Adj_{G'}(X)\setminus\{Y\}$ , $|S| = n$ , tested.;
    n:=n+1;
    until no adjacent X, Y s.t., $|Adj_{G'}(X)\setminus\{Y\}| < n$ ;
    return $G'$

and a maximal clique is a clique that cannot be extended), s.t., $|\mathbf{Q}| \geq 3$ . We then partition these cliques into groups such that two cliques $\mathbf{Q}_1, \mathbf{Q}_2$ are in the same group if $|\mathbf{Q}_1 \cap \mathbf{Q}_2| \geq 2$ . Finally, for each group of cliques $\mathcal{Q}$ (as in line 3 in Alg. 1), we combine them to form a set of variables $\mathbf{X}_{\mathcal{Q}} = \cup_{\mathbf{Q} \in \mathcal{Q}} \mathbf{Q}$ . We further determine the neighbour set for each $\mathbf{X}_{\mathcal{Q}}$ , as $\mathbf{N}_{\mathcal{Q}} = \{\mathsf{N} : \exists \mathsf{X} \in \mathbf{X}_{\mathcal{Q}} \text{s.t.} \mathsf{N}, \mathsf{X}\text{ are adjacent in CI skeleton } \mathcal{G}'\}$ . It can be shown that variables that satisfy (i)-(iv) will be in the same set with $\mathbf{X}_{\mathcal{Q}} \cup \mathbf{N}_{\mathcal{Q}}$ (examples and proof in Appx. B.4). Therefore our next step is to take each $\mathbf{X}_{\mathcal{Q}} \cup \mathbf{N}_{\mathcal{Q}}$ separately as input to our Phases 2 and 3, detailed as follows.

# 4.4 PHASE 2: FINDING CAUSAL CLUSTERS

In this section, we introduce the second phase of our algorithm, FindCausalClusters, summarized in Alg. 3 in the appendix and illustrated in Fig. 4. The objective here is to design an effective search procedure to find combinations of sets of covers $\mathcal{C}$ and $\mathcal{X}$ (as defined in Theorem 8), such that rank deficiency holds and all the conditions required in Theorem 8 are satisfied. To be specific, given Condition 1, Theorem 8 further requires that (i) when we are searching for $k$ -clusters, all the $k'$ -clusters, $k' < k$ have been found and recorded, and that (ii) there is no collider in $\mathcal{G}$ . We next introduce the key designs for that end, accompanied by examples.

As for the requirement (i), we design our search procedure such that it starts with k = 1. If a k-cluster is found, we update the graph and reset k to 1; otherwise, we increase k by 1 (as in line 5 Alg 3). This ensures to a large extent that when searching for a k-cluster, all $k'$ clusters such that $k' < k$ , can be found, in order to fulfill the requirement (i).

Regarding the requirement (ii), we need to ensure that all the rank deficiencies are from atomic covers, rather than from colliders. One could directly assume the absence of colliders in the underlying G, but this would impose rather strong structural constraints and thus limit the applicability of a discovery method. Therefore, we add a collider check function NoCollider defined in Alg. 4, together with the designed search procedure to allow incorporating colliders timely such that they

Algorithm 3: Phase2: FindCausalClusters   
Input : Samples from n observed variables $X_{G}$ Output: Graph $G'$ def FindCausalClusters( $G'$ , $X_{G}$ ):
    Active set $S \leftarrow X_{G} = \{\{X_{1}\}, ..., \{X_{n}\}\}, k \leftarrow 1$ ; // S is a set of covers
    repeat $G'$ , S, found = Search( $G'$ , S, $X_{G}$ , k); // Only when nothing can be found
    If found = 1 then $k \leftarrow 1$ else $k \leftarrow k + 1$ ; // udner current k do we add k by 1
    until k is sufficiently large;
    return $G'$ ;

def Search( $G'$ , S, $X_{G}$ , k):
    Rank deficiency set D = {}; // To store rank deficient combinations
    for T ∈ PowerSet(S) (from S to 0) do $S' \leftarrow (S \setminus T) \cup (\cup_{T \in T} P Ch_{G'}(T))$ ; // Unfold S to get $S'$ for t = k to 0 do
    repeat

    Draw a set of t observed covers $X \subset S' \cap X_{G}$ ;
    repeat

    Draw a set of covers $C \subset S' \setminus X$ , s.t., $||C|| = k - t + 1$ and get $N \leftarrow S' \setminus (X \cup C)$ ;
    if $rank(\Sigma_{C \cup X, N \cup X}) = k$ and NoCollider( $C, X, N$ ) then Add C to D;

    until all C exhausted;
    if D ≠ 0 then
    for $D_{i} \in D$ do
    if $|Pa_{G'}(D_{i}) \cup X| = k$ then P ← $Pa_{G'}(D_{i}) \cup X$ ;
    else Create new latent variables L, s.t., P ← L ∪ $Pa_{G'}(D_{i}) \cup X$ and
    |L| = k - |Pa_{G'}(D_{i}) ∪ X|;
    Update $G'$ by taking elements of $D_{i}$ as the pure children of P;
    if P is atomic then Update $S \leftarrow (S \setminus D_{i}) \cup P$ ;
    return $G'$ , S, True; // Return to search with k = 1
    until all X exhausted;
    return $G'$ , S, False; // Return to search with $k \leftarrow k + 1$

Algorithm 4: Function: NoCollider   
Input : C, X, N
Output: Whether there exists O ∈ C s.t., O is a collider of C\{O} and N
def NoCollider(C, X, N):
    for c = 1 to |C| - 1 do
    Draw C' ⊂ C s.t., |C'| = c;
    repeat
    if rank(ΣC'∪X,N∪X) < ||C' ∪ X|| then return False;
    until all C' exhausted;
    return True

will not induce unexpected rank deficiency anymore. With these two designs, we only rely on a much weaker condition about colliders, as in Condition 2, under which it can be guaranteed that our search procedure will not be affected by the existence of colliders (proof in Appx. A.17).

Condition 2 (Graphical condition on colliders for identifiability). In a latent graph G, if (i) there exists a set of variables C such that every variable in C is a collider of two atomic covers $V_{1}$ , $V_{2}$ , and denote by A the minimal set of variables that d-separates $V_{1}$ from $V_{2}$ , (ii) there is a latent variable in $V_{1}$ , $V_{2}$ , C or A, then we must have $|C| + |A| \geq |V_{1}| + |V_{2}|$ .

We summarize the whole process of Phase 2 in Alg. 3 and provide an illustration in Fig. 4, where the input variables are from the left dash area in Fig. 3 (b). For a better understanding, please refer to Appx. B.5 for a detailed description of key steps and illustrative examples.

# 4.5 PHASE 3: REFINING CAUSAL CLUSTERS

In Phase 2, we strive to fulfill all required conditions such that we can correctly identify causal clusters and related structures. However, there still exist some rare cases where our search cannot ensure the requirement (i) in Theorem 8. In this situation, Phase 2 might produce a big cluster in

Algorithm 5: Phase3: RefineCausalClusters   
Input : Graph $\mathcal{G}'$ Output: Refined graph $\mathcal{G}'$ def RefineCausalCLusters( $\mathcal{G}'$ , $\mathbf{X}_{\mathcal{G}}$ ):
repeat
Draw an atomic cover $\mathbf{V}$ from $\mathcal{G}'$ ;
Delete $\mathbf{V}$ , neighbours of $\mathbf{V}$ that are latent, and all relating edges from $\mathcal{G}'$ to get $\hat{\mathcal{G}}$ ; $\mathcal{G}' = \text{FindCausalClusters}(\hat{\mathcal{G}}, \mathbf{X}_{\mathcal{G}})$ ;
until No more $\mathbf{V}$ found and all $\mathbf{V}$ exhausted;
return $\mathcal{G}'$

Table 2: F1 scores (mean (standard deviation)) of compared methods on different types of latent graphs. 

<table><tr><td colspan="2"></td><td colspan="6">F1 score for skeleton among all variables  $V_{\mathcal{G}}$  (both  $X_{\mathcal{G}}$  and  $L_{\mathcal{G}}$ )</td></tr><tr><td colspan="2">Algorithm</td><td>Ours</td><td>Hier. rank</td><td>PC</td><td>FCI</td><td>GIN</td><td>RCD</td></tr><tr><td rowspan="3">Latent+tree</td><td>2k</td><td>0.84(0.11)</td><td>0.58 (0.01)</td><td>0.36 (0.01)</td><td>0.37 (0.01)</td><td>0.37 (0.03)</td><td>0.24 (0.04)</td></tr><tr><td>5k</td><td>0.92(0.05)</td><td>0.60 (0.01)</td><td>0.37 (0.00)</td><td>0.37 (0.01)</td><td>0.41 (0.03)</td><td>0.33 (0.00)</td></tr><tr><td>10k</td><td>0.98(0.02)</td><td>0.60 (0.01)</td><td>0.37 (0.00)</td><td>0.38 (0.02)</td><td>0.41 (0.03)</td><td>0.33 (0.01)</td></tr><tr><td rowspan="3">Latent+measm</td><td>2k</td><td>0.81(0.12)</td><td>0.52 (0.05)</td><td>0.44 (0.01)</td><td>0.38 (0.02)</td><td>0.40 (0.02)</td><td>0.26 (0.03)</td></tr><tr><td>5k</td><td>0.88(0.11)</td><td>0.52 (0.05)</td><td>0.49 (0.01)</td><td>0.40 (0.01)</td><td>0.46 (0.03)</td><td>0.29 (0.01)</td></tr><tr><td>10k</td><td>0.91(0.09)</td><td>0.53 (0.05)</td><td>0.49 (0.01)</td><td>0.40 (0.01)</td><td>0.47 (0.05)</td><td>0.34 (0.04)</td></tr><tr><td rowspan="3">Latent general</td><td>2k</td><td>0.66(0.01)</td><td>0.44 (0.02)</td><td>0.31 (0.01)</td><td>0.25 (0.02)</td><td>0.30 (0.04)</td><td>0.32 (0.03)</td></tr><tr><td>5k</td><td>0.72(0.03)</td><td>0.45 (0.03)</td><td>0.32 (0.01)</td><td>0.28 (0.02)</td><td>0.38 (0.04)</td><td>0.34 (0.02)</td></tr><tr><td>10k</td><td>0.80(0.05)</td><td>0.45 (0.04)</td><td>0.32 (0.01)</td><td>0.28 (0.02)</td><td>0.35 (0.01)</td><td>0.36 (0.01)</td></tr></table>

the resulting $G'$ that should be split into smaller ones (see examples in Appx. B.6). Fortunately, the incorrect cluster will not do harm to the identification of other substructures in the graph, and thus we can employ Phase 3 to characterize and refine the incorrect ones, by making use of the following Theorem 11 (proof in Appx. A.16).

Theorem 11 (Refining Clusters). Denote by $\mathcal{G}'$ the output from FindCausalClusters and by $\mathcal{G}$ the true graph. For an atomic cover $\mathbf{V}$ in $\mathcal{G}'$ , if $\mathbf{V}$ is not a correct cluster in $\mathcal{G}$ but consists of some smaller clusters, then $\mathbf{V}$ can be refined into correct ones by FindCausalClusters( $\hat{\mathcal{G}}, \mathbf{X}$ ), where $\hat{\mathcal{G}}$ is got by deleting $\mathbf{V}$ , all neighbors of $\mathbf{V}$ that are latent, and all relating edges of them, from $\mathcal{G}'$ .

To be specific, we search through all the atomic covers $\mathbf{V}$ in $\mathcal{G}'$ , the output of Phase 2, and then perform FindCausalClusters $(\hat{\mathcal{G}},\mathbf{X})$ , where $\hat{\mathcal{G}}$ is defined as in Theorem 11. With this procedure, we can make sure that all the found clusters in $\mathcal{G}'$ are correct as in $\mathcal{G}$ . We summarized Phase 3 in Alg. 5 and provide illustrative examples in Appx. B.6.

# 5 IDENTIFIABILITY THEORY OF CAUSAL STRUCTURE

Here we show the identifiability of the proposed RLCD algorithm. Specifically, RLCD asymptotically produces the correct Markov equivalence class of the causal graph over both observed and latent variables under certain graphical conditions, up to the minimal-graph operator $\mathcal{O}_{\mathrm{min}}(\cdot)$ and skeleton operator $\mathcal{O}_{\mathrm{s}}(\cdot)$ (defined in Appx. A.4 following Huang et al. (2022)). $\mathcal{O}_{\mathrm{min}}(\cdot)$ is to absorb redundant latent variables under certain conditions and $\mathcal{O}_{\mathrm{s}}(\cdot)$ is to introduce edges involving latent variables if certain conditions hold, and we note that the observational rank information is invariant to these two graph operators (examples in Appx. B.12). We summarize the identifiability result in Theorem 12, along with Corollary 1 (proof in Appx. A.17).

Theorem 12 (Identifiability of the Proposed Alg. 1). Suppose G is a DAG associated with a Linear Latent Causal Model that satisfies Condition 1 and Condition 2. Algorithm 1 can asymptotically identify the Markov equivalence class of $\mathcal{O}_{\min}(\mathcal{O}_{s}(\mathcal{G}))$ .

Corollary 1. Assume linear causal models. Asymptotically, when no latent variable exists, Alg. 1's output is the same as that of PC; as another special case, when there is no edge between observed variables, the output of Alg. 1 is the same as that of Hier. rank (Huang et al., 2022).

Table 3: F1 scores (mean (standard deviation)) of compared methods on different types of latent graphs. F1 score is calculated only for edges between observed variables $X_{G}$ in this graph. Hier.rank and GIN assume that observed variables are not directly adjacent so their performance is reported as -. 

<table><tr><td colspan="2"></td><td colspan="6">F1 score for skeleton among  $X_G$ </td></tr><tr><td colspan="2">Algorithm</td><td>Ours</td><td>Hier. rank</td><td>PC</td><td>FCI</td><td>GIN</td><td>RCD</td></tr><tr><td rowspan="3">Latent+tree</td><td>2k</td><td>0.79 (0.16)</td><td>-</td><td>0.46 (0.02)</td><td>0.00 (0.00)</td><td>-</td><td>0.30 (0.03)</td></tr><tr><td>5k</td><td>0.86 (0.10)</td><td>-</td><td>0.44 (0.00)</td><td>0.03 (0.04)</td><td>-</td><td>0.38 (0.01)</td></tr><tr><td>10k</td><td>0.97 (0.04)</td><td>-</td><td>0.44 (0.00)</td><td>0.18 (0.07)</td><td>-</td><td>0.39 (0.02)</td></tr><tr><td rowspan="3">Latent+measm</td><td>2k</td><td>0.84 (0.11)</td><td>-</td><td>0.50 (0.02)</td><td>0.00 (0.00)</td><td>-</td><td>0.30 (0.02)</td></tr><tr><td>5k</td><td>0.93 (0.08)</td><td>-</td><td>0.49 (0.01)</td><td>0.05 (0.03)</td><td>-</td><td>0.32 (0.02)</td></tr><tr><td>10k</td><td>0.95 (0.05)</td><td>-</td><td>0.48 (0.02)</td><td>0.03 (0.05)</td><td>-</td><td>0.42 (0.09)</td></tr><tr><td rowspan="3">Latent general</td><td>2k</td><td>0.68 (0.02)</td><td>-</td><td>0.44 (0.01)</td><td>0.27 (0.09)</td><td>-</td><td>0.39 (0.06)</td></tr><tr><td>5k</td><td>0.71 (0.03)</td><td>-</td><td>0.45 (0.01)</td><td>0.31 (0.10)</td><td>-</td><td>0.44 (0.05)</td></tr><tr><td>10k</td><td>0.78 (0.06)</td><td>-</td><td>0.45 (0.01)</td><td>0.32 (0.05)</td><td>-</td><td>0.44 (0.01)</td></tr></table>

![](images/7a004b8c5e4f970d1c29b484e496a070c5c3428c2acb6c4b12ec1d839232e09f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Openness"] --> B["L4"]
    A --> C["L5"]
    A --> D["L6"]
    A --> E["L3"]
    A --> F["L1"]
    A --> G["L2"]
    A --> H["Conscientiousness"]
    A --> I["Agreeableness"]
    
    B --> J["[O3"] I have a vivid imagination.]
    C --> K["[O7"] I am quick to understand things.]
    D --> L["[O1"] I have a rich vocabulary.]
    E --> M["[E1"] I am the life of the party.]
    F --> N["[A10"] I make people feel at ease.]
    G --> O["[A9"] I feel others' emotions.]
    H --> P["[N7"] I change my mood a lot.]
    
    L --> Q["[C5"] I get chores done right away.]
    M --> R["[E8"] I don't like to draw attention to myself.]
    N --> S["[N9"] I get upset equally.]
    O --> T["[N10"] I often feel blue.]
    
    P --> U["[C8"] I shirk my duties.]
    Q --> V["[C4"] I make a mess of things.]
    R --> W["[C7"] I like order.]
    S --> X["[C1"] I am always prepared.]
    T --> Y["[C10"] I am exactly in my work.]
    
    U --> Z["[C2"] I leave my belongings around.]
    V --> AA["[C6"] I often forget to put things back in their proper place.]
    
    W --> AB["[C3"] I pay attention to details.]
    X --> AC["[C9"] I follow a schedule.]
    
    Y --> AD["[A1"] I feel little concern for others.]
    Z --> AE["[A4"] I sympathize with others' feelings.]
    
    subgraph Extraversion
        B --> F
        C --> F
        D --> F
        E --> F
        F --> G
        G --> H
        H --> I
        I --> J
        J --> K
        K --> L
        L --> M
        M --> N
        N --> O
        O --> P
        P --> Q
        Q --> R
        R --> S
        S --> T
        T --> U
        U --> V
        V --> W
        W --> X
        X --> Y
        Y --> Z
        Z --> AA
        AA --> AB
        AB --> AC
        AC --> AD
        AD --> AE
        AE --> AF["Extraversion reflecting on things."]
        AF --> AG["[O9"] I spend time reflecting on things.]
        AF --> AH["[E10"] I am quiet around strangers.]
        AF --> AI["[E2"] I don't talk a lot.]
        AF --> AJ["[E7"] I talk to a lot of different people at parties.]
        AF --> AK["[E3"] I feel comfortable around people.]
        AF --> AL["[E5"] I start conversations.]
        AF --> AM["[E4"] I keep in the background.]
    
    end
    
    subgraph Neuroticism
        B --> N
        N --> O
        O --> P
        P --> Q
        Q --> R
        R --> S
        S --> T
        T --> U
        U --> V
        V --> W
        W --> X
        X --> Y
        Y --> Z
    
    end
    
    subgraph Agreeableness
        H --> AA
        AA --> AB
        AB --> AC
        AC --> AD
        AD --> AE
        AE --> AF
    
    style Extraversion fill:#f9f,stroke:#333,stroke-width:2px
    style Neuroticism fill:#bbf,stroke:#333,stroke-width:2px
    style Agreeableness fill:#bfb,stroke:#333,stroke-width:2px
    
    %% Legend
    L4,L5,L6,L7,L8,L9,L10,L11,L12,N13,N14,N15,N16,N17,N18,N19,N20,N21,N22,N23,N24,N25,N26,N27,N28,N29,N30,N31,N32,N33,N34,N35,N36,N37,N38,N39,N40,N41,N42,N43,N44,N45,N46,N47,N48,N49,N50,N51,N52,N53,N54,N55,N56,N57,N58,N59,N60,N61,N62,N63,N64,N65,N66,N67,N68,N69,N70,N71,N72,N73,N74,N75,N76,N77,N78,N79,N80,N81,N82,N83,N84,N85,N86,N87,N88,N89,N90,N91,N92,N93,N94,N95,N96,N97,N98,N99,N100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,
```
</details>

Figure 5: Identified latent graph on Big Five personality dataset.

# 6 EXPERIMENTS

We validate our method using both synthetic and real-life data. In finite sample cases, we employ canonical correlations (Anderson, 1984) to estimate the rank (detailed in Appx. A.18).

Synthetic Data Specifically, we considered different types of latent graphs: (i) latent tree models (Appx. B.8), (ii) latent measurement models (Appx. B.9), and (iii) general latent models (Fig. 1 and Appx. B.10). The causal strength is uniformly from $[-10, 10]$ , and the noise is either Gaussian or uniform (which is for RCD and GIN). We propose to use the following two metrics for comparisons: (1) F1 score of skeleton among observed variables $X_{G}$ , (2) F1 score of skeleton among all variables $V_{G}$ . We consider combinations and permutations of latent variables during evaluation (detailed in Appx. B.15 together with specific definition of F1 score).

We compared with many competitive baselines, including (i) Hier. rank (Huang et al., 2022), (ii) PC (Spirtes et al., 2000), (iii) FCI (Spirtes et al., 2013), (iv) RCD (Maeda & Shimizu, 2020), and (v) GIN (Xie et al., 2020). The results are reported in Tab. 2 and Tab. 3, where we run experiments with different random seeds and sample sizes $2k$ , $5k$ , and $10k$ . Our proposed RLCD gives the best results on all types of graphs, in terms of both metrics, with a clear margin. This result serves as strong empirical support for the identifiability of latent linear causal graphs by our proposed method.

Real-World Data To further verify our proposed method, we employed a real-world Big Five Personality dataset https://openpsychometrics.org/. It consists of 50 personality indicators and close to 20,000 data points. Each Big Five personality dimension, namely, Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism (O-C-E-A-N), are measured with their own 10 indicators. Data is processed to have zero mean and unit variance. We employ the

proposed method to determine the Markov equivalence class and employ GIN (Xie et al., 2020) to further decide other directions between latent variables (more details in Appendix B.17).

We analyzed the data using RLCD, producing a causal graph in Fig 5 that exhibits interesting psychological properties. First, most of the variables related to the same Big Five dimension are in the same cluster. Strikingly, our result reconciles two currently deemed distinct theories of personality: latent personality dimensions and network theory (Cramer et al., 2012; Wright, 2017). We see groups of closely connected items that are predictable from latent dimensions (L1, L2, L3), interactions among latents (L1→L6→L3, L1→L2→L3), and latents influencing the same indicators (L1, L3). We also observe plausible causal links between indicators (e.g., O2→O4 and O1→O8). We argue that our findings are consistent with pertinent personality literature, but more importantly, offer new, plausible explanations as to the nature of human personality (detailed analysis in Appx. B.18).

# 7 DISCUSSION AND CONCLUSION

We developed a versatile causal discovery approach that allows latent variables to be causally-related in a flexible way, by making use of rank information. We showed the proposed method can asymptotically identify the Markov equivalence class of the underlying graph under mild conditions. One limitation of our method is that it cannot directly handle cyclic graphs, and as future work, we will extend this line of thought of using rank information to cyclic graphs. Another line of future research is to extend the idea to handle nonlinear causal relations.

# REFERENCES

Jeffrey Adams, Niels Hansen, and Kun Zhang. Identification of partially observed linear causal models: Graphical conditions for the non-gaussian and heterogeneous cases. Advances in Neural Information Processing Systems, 34, 2021.   
Raj Agrawal, Chandler Squires, Neha Prasad, and Caroline Uhler. The decamfounder: Non-linear causal discovery in the presence of hidden variables. arXiv preprint arXiv:2102.07921, 2021.   
Sina Akbari, Ehsan Mokhtarian, AmirEmad Ghassami, and Negar Kiyavash. Recursive causal structure learning in the presence of latent variables and selection bias. In Advances in Neural Information Processing Systems, volume 34, pp. 10119–10130, 2021.   
Animashree Anandkumar, Daniel Hsu, Adel Javanmard, and Sham Kakade. Learning linear bayesian networks with latent variables. In International Conference on Machine Learning, pp. 249–257, 2013.   
T. W. Anderson. An Introduction to Multivariate Statistical Analysis. 2nd ed. John Wiley & Sons, 1984.   
Ruichu Cai, Feng Xie, Clark Glymour, Zhifeng Hao, and Kun Zhang. Triad constraints for learning causal structure of latent variables. In Advances in Neural Information Processing Systems, pp. 12863–12872, 2019.   
Chandler Squires. causaldag: creation, manipulation, and learning of causal models, 2018. URL https://github.com/uhlerlab/causaldag.   
V. Chandrasekaran, S. Sanghavi, P. A. Parrilo, and A. S. Willsky. Rank-sparsity incoherence for matrix decomposition. SIAM Journal on Optimization, 21(2):572–596, 2011.   
V. Chandrasekaran, P. A. Parrilo, and A. S. Willsky. Latent variable graphical model selection via convex optimization. Annals of Statistics, 40(4):1935–1967, 2012.   
Zhengming Chen, Feng Xie, Jie Qiao, Zhifeng Hao, Kun Zhang, and Ruichu Cai. Identification of linear latent variable model with arbitrary distribution. In Proceedings 36th AAAI Conference on Artificial Intelligence (AAAI), 2022.   
David Maxwell Chickering. Learning equivalence classes of bayesian-network structures. The Journal of Machine Learning Research, 2:445–498, 2002a.

David Maxwell Chickering. Optimal structure identification with greedy search. Journal of machine learning research, 3(Nov):507–554, 2002b.   
David Maxwell Chickering. A transformational characterization of equivalent bayesian network structures. arXiv preprint arXiv:1302.4938, 2013.   
Diego Colombo, Marloes H Maathuis, Markus Kalisch, and Thomas S Richardson. Learning high-dimensional directed acyclic graphs with latent and selection variables. The Annals of Statistics, pp. 294–321, 2012.   
Angélique OJ Cramer, Sophie Van der Sluis, Arjen Noordhof, Marieke Wichers, Nicole Geschwind, Steven H Aggen, Kenneth S Kendler, and Denny Borsboom. Dimensions of normal personality as networks in search of equilibrium: You can't like parties if you don't like people. European Journal of Personality, 26(4):414–431, 2012.   
Petru Lucian Curșeu, Remus Ilies, Delia Vîrgă, Laurențiu Maricuțoiu, and Florin A Sava. Personality characteristics that are valued in teams: Not always “more is better”? International Journal of Psychology, 54(5):638–649, 2019.   
Haoyue Dai, Peter Spirtes, and Kun Zhang. Independence testing-based approach to causal discovery under measurement error and linear non-gaussian models. Advances in Neural Information Processing Systems, 35:27524–27536, 2022.   
Yanming Di. t-separation and d-separation for directed acyclic graphs. preprint, 2009.   
Doris Entner and Patrik O Hoyer. Discovering unconfounded causal relationships using linear non-gaussian models. In JSAI International Symposium on Artificial Intelligence, pp. 181–195. Springer, 2010.   
Thomas Gatzka. Aspects of openness as predictors of academic achievement. Personality and Individual Differences, 170:110422, 2021.   
William G Graziano and Renée M Tobin. Agreeableness: Dimension of personality or social desirability artifact? Journal of personality, 70(5):695–728, 2002.   
Pierce J Howard and Jane Mitchell Howard. The owner's manual for personality at work: How the Big Five personality traits affect performance, communication, teamwork, leadership, and sales. Center for Applied Cognitive Studies (CentACS), 2010.   
Patrik O Hoyer, Shohei Shimizu, Antti J Kerminen, and Markus Palviainen. Estimation of causal effects using linear non-gaussian causal models with hidden variables. International Journal of Approximate Reasoning, 49(2):362–378, 2008.   
Patrik O Hoyer, Dominik Janzing, Joris M Mooij, Jonas Peters, and Bernhard Schölkopf. Nonlinear causal discovery with additive noise models. In Advances in neural information processing systems, pp. 689–696, 2009.   
B. Huang\*, K. Zhang\*, J. Zhang, R. Sanchez-Romero, C. Glymour, and B. Schölkopf. Causal discovery from heterogeneous/nonstationary data. In JMLR, volume 21(89), 2020.   
Biwei Huang, Charles Jia Han Low, Feng Xie, Clark Glymour, and Kun Zhang. Latent hierarchical causal structure discovery with rank constraints. arXiv preprint arXiv:2210.01798, 2022.   
Bohdan Kivva, Goutham Rajendran, Pradeep Ravikumar, and Bryon Aragam. Learning latent causal graphs via mixture oracles. Advances in Neural Information Processing Systems, 34, 2021.   
Erich Kummerfeld and Joseph Ramsey. Causal clustering for 1-factor measurement models. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1655–1664. ACM, 2016.   
Huy Le, In-Sue Oh, Steven B Robbins, Remus Ilies, Ed Holland, and Paul Westrick. Too much of a good thing: curvilinear relationships between personality traits and job performance. Journal of Applied Psychology, 96(1):113, 2011.

Wendy Lord. NEO PI-R: A guide to interpretation and feedback in a work context. Hogrefe, 2007.   
Takashi Nicholas Maeda and Shohei Shimizu. Rcd: Repetitive causal discovery of linear non-gaussian acyclic models with latent confounders. In International Conference on Artificial Intelligence and Statistics, pp. 735–745. PMLR, 2020.   
Christopher Meek. Causal inference and causal explanation with background knowledge. arXiv preprint arXiv:1302.4972, 2013.   
J. Pearl. Probabilistic reasoning in intelligent systems: Networks of plausible inference. 1988.   
Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, New York, NY, USA, 2000. ISBN 0-521-77362-8.   
Judea Pearl. The seven tools of causal inference, with reflections on machine learning. Communications of the ACM, 62(3):54–60, 2019.   
Saber Salehkaleybar, AmirEmad Ghassami, Negar Kiyavash, and Kun Zhang. Learning linear nongaussian causal models in the presence of latent variables. Journal of Machine Learning Research, 21(39):1–24, 2020.   
Shohei Shimizu, Patrik O. Hoyer, Aapo Hyvärinen, and Antti Kerminen. A linear non-gaussian acyclic model for causal discovery. J. Mach. Learn. Res., 7:2003–2030, December 2006a. ISSN 1532-4435.   
Shohei Shimizu, Patrik O Hoyer, Aapo Hyvärinen, Antti Kerminen, and Michael Jordan. A linear non-gaussian acyclic model for causal discovery. Journal of Machine Learning Research, 7(10), 2006b.   
Shohei Shimizu, Patrik O Hoyer, and Aapo Hyvärinen. Estimation of linear non-gaussian acyclic models for latent factors. Neurocomputing, 72(7-9):2024–2027, 2009.   
Ricardo Silva, Richard Scheine, Clark Glymour, and Peter Spirtes. Learning the structure of linear latent variable models. Journal of Machine Learning Research, 7(Feb):191–246, 2006.   
Peter Spirtes. An anytime algorithm for causal inference. In International Workshop on Artificial Intelligence and Statistics, pp. 278–285. PMLR, 2001.   
Peter Spirtes. Calculation of entailed rank constraints in partially non-linear and cyclic models. In Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence, pp. 606–615. AUAI Press, 2013.   
Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. Causation, prediction, and search. MIT press, 2000.   
Peter Spirtes, Clark Glymour, Richard Scheines, and Robert Tillman. Automated search for causal relations: Theory and practice. 2010.   
Peter L Spirtes, Christopher Meek, and Thomas S Richardson. Causal inference in the presence of latent variables and selection bias. arXiv preprint arXiv:1302.4983, 2013.   
Seth Sullivan, Kelli Talaska, and Jan Draisma. Trek separation for gaussian graphical models. arXiv:0812.1938., 2010.   
Tatsuya Tashiro, Shohei Shimizu, Aapo Hyvärinen, and Takashi Washio. ParceLiNGAM: a causal ordering method robust against latent confounders. Neural Computation, 26(1):57–83, 2014.   
Sofia Triantafillou and Ioannis Tsamardinos. Constraint-based causal discovery from multiple interventions over overlapping variable sets. The Journal of Machine Learning Research, 16(1):2147–2205, 2015.   
S. Wang. Causal clustering for 1-factor measurement models on data with various types. arXiv preprint arXiv:2009.08606, 2020.

Aidan GC Wright. Factor analytic support for the five-factor model. The Oxford handbook of the five factor model, pp. 217–242, 2017.   
Feng Xie, Ruichu Cai, Biwei Huang, Clark Glymour, Zhifeng Hao, and Kun Zhang. Generalized independent noise condition for estimating latent variable causal graphs. In Advances in Neural Information Processing Systems, pp. 14891–14902, 2020.   
Kun Zhang and Aapo Hyvärinen. On the identifiability of the post-nonlinear causal model. In Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence, pp. 647–655. AUAI Press, 2009.   
Nevin L Zhang. Hierarchical latent class models for cluster analysis. The Journal of Machine Learning Research, 5:697–723, 2004.   
Yujia Zheng, Biwei Huang, Wei Chen, Joseph Ramsey, Mingming Gong, Ruichu Cai, Shohei Shimizu, Peter Spirtes, and Kun Zhang. Causal-learn: Causal discovery in python. arXiv preprint arXiv:2307.16405, 2023.

# Appendix

# Organization of Appendices:

• Section A: Definitions, Examples, and Proofs

– Section A.1: Subcovariance Matrix and Definition of Descendants.   
- Section A.2: Treks, T-separations, and Examples.   
– Section A.3: Definition of CI Skeleton   
- Section A.4: Definition of Rank-invariant Graph Operator.   
- Section A.5: Max-Flow-Min-Cut Lemma 13 for Treks   
- Section A.6: Example for Theorem 4.   
- Section A.7: Example for Theorem 6.   
- Section A.8: Proof of Theorem 4.   
- Section A.9: Proof of Theorem 5.   
- Section A.10: Proof of Theorem 6.   
- Section A.11: Proof of Theorem 7.   
- Section A.12: Example for Theorem 8.   
- Section A.13: Proof of Theorem 8.   
- Section A.14: Proof of Theorem 9.   
- Section A.15: Proof of Lemma 10.   
- Section A.16: Proof of Theorem 11.   
- Section A.17: Proof of Theorem 12.   
- Section A.18: Description of the rank test that we employ.

• Section B: Graphs, Illustrations of algorithms, and more information on datasets

- Section B.1: Examples of Grpahs that Each Method Can Handle.   
– Section B.2: Example of Violation of faithfulness   
- Section B.3: Example of Atomic Cover.   
– Section B.4: Example for Phase 1   
– Section B.5: Detailed Description and Example for Phase 2.   
– Section B.6: Example for Phase 3.   
- Section B.7: Graph examples for model that has only observed variables.   
- Section B.8: Graph examples for latent tree model.   
- Section B.9: Graph examples for latent measurement model.   
- Section B.10: Graph examples for general latent model.   
- Section B.11: Example for considering colliders in Phase 1.   
- Section B.12: Examples for graph operators.   
- Section B.13: Examples for graphical relations between covers.   
- Section B.14: Discussions on checking colliders completely.   
- Section B.15: Evaluation metric details.   
- Section B.16: More details of experiments on synthetic data.   
– Section B.17: Detailed information of the Big Five personality dataset.   
- Section B.18: Detailed analysis of the result from the Big Five dataset.

• Section C: Related work

\- Section C.1: Related work.

• Section D: Additional Information (added during rebuttal)

# A DEFINITIONS, EXAMPLES, AND PROOFS

# A.1 SUBCOVARIANCE MATRIX AND DEFINITION OF DESCENDANTS

$\Sigma_{\mathbf{A},\mathcal{B}}$ refers to subcovariance over $\mathbf{A}$ and $\mathbf{B}$ . E.g., $\Sigma_{\{\mathsf{X}_1,\mathsf{X}_2\},\{\mathsf{X}_3\}}$ is a $2\times 1$ matrix whose entry at (1,1) is $\mathrm{Cov}(\mathsf{X}_1,\mathsf{X}_3)$ and entry at (2,1) is $\mathrm{Cov}(\mathsf{X}_2,\mathsf{X}_3)$ . $\Sigma_{\mathcal{A},\mathcal{B}}$ refers to subcovariance over $\mathcal{A}$ and $\mathcal{B}$ . Specifically, $\Sigma_{\mathcal{A},\mathcal{B}} = \Sigma_{\cup_{\mathbf{A}\in\mathcal{A}}\mathbf{A},\cup_{\mathbf{B}\in\mathcal{B}}\mathbf{B}}$ . E.g., $\Sigma_{\{\{\mathsf{X}_1\},\{\mathsf{X}_2\}\},\{\{\mathsf{X}_3\}\}}$ is also a $2\times 1$ matrix whose entry at (1,1) is $\mathrm{Cov}(\mathsf{X}_1,\mathsf{X}_3)$ and entry at (2,1) is $\mathrm{Cov}(\mathsf{X}_2,\mathsf{X}_3)$ .

As for the definition of descendants, a descendant is a node that can be reached by following one or more directed edges from a given node, and this definition excludes the node itself.

# A.2 TREKS, T-SEPARATIONS, AND EXAMPLES

![](images/5b8b5752f1a60c48bf48f8095d952ed17226c1876618313ef6cd73e12473d2ba.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1["X₁"] --> L1["L₁"]
    X1 --> L2["L₂"]
    X1 --> X2["X₂"]
    L1 --> X3["X₃"]
    L1 --> X4["X₄"]
    L2 --> X5["X₅"]
    L2 --> X6["X₆"]
    L2 --> X7["X₇"]
```
</details>

Figure 6: An example graph to show treks, t-separations, and rank of subcovariance matrix.

Example 3 (Example of Treks). In Figure 6, there are four treks between $X_{3}$ and $X_{4}$ : (i) $(P_{1}, P_{2}) = (\mathsf{X}_{3} \leftarrow \mathsf{L}_{1}, \mathsf{L}_{1} \rightarrow \mathsf{X}_{4})$ , (ii) $(P_{1}, P_{2}) = (\mathsf{X}_{3} \leftarrow \mathsf{L}_{2}, \mathsf{L}_{2} \rightarrow \mathsf{X}_{4})$ , (iii) $(P_{1}, P_{2}) = (\mathsf{X}_{3} \leftarrow \mathsf{L}_{1} \leftarrow \mathsf{X}_{1}, \mathsf{X}_{1} \rightarrow \mathsf{L}_{2} \rightarrow \mathsf{X}_{4})$ , (iv) $(P_{1}, P_{2}) = (\mathsf{X}_{3} \leftarrow \mathsf{L}_{2} \leftarrow \mathsf{X}_{1}, \mathsf{X}_{1} \rightarrow \mathsf{L}_{1} \rightarrow \mathsf{X}_{4})$ . For adjacent variables such as $X_{1}$ and $X_{2}$ , there must exist at least one trek $(P_{1}, P_{2}) = (\mathsf{X}_{1}, \mathsf{X}_{1} \rightarrow \mathsf{X}_{2})$ between them.

Example 4 (Example of Trek-separations). In Figure 6, $X_{3}$ and $X_{4}$ can be t-separated by $(\{L_{1}, L_{2}\}, \emptyset)$ , as for all the treks (i)-(iv) in Example 3, either $P_{1}$ contains a vertex in $\{L_{1}, L_{2}\}$ or $P_{2}$ contains a vertex in $\emptyset$ . Similarly, $X_{3}$ and $X_{4}$ can also be t-separated by $(\emptyset, \{L_{1}, L_{2}\})$ . However, the most simple way to t-separate $X_{3}$ and $X_{4}$ is by $(\{X_{3}\}, \emptyset)$ or $(\emptyset, \{X_{4}\})$ .

Example 5 (Example of calculating rank). As shown in Example 4, the minimal way to t-separate $X_{3}$ and $X_{4}$ is by $(\mathbf{C}_{\mathbf{A}}, \mathbf{C}_{\mathbf{B}}) = (\{\mathsf{X}_{3}\}, \emptyset)$ or $(\emptyset, \{\mathsf{X}_{4}\})$ , and thus $\min |\mathbf{C}_{\mathbf{A}}| + |\mathbf{C}_{\mathbf{B}}| = 1$ . Therefore, $\text{rank}(\Sigma_{\{\mathsf{X}_{3}\}, \{\mathsf{X}_{4}\}}) = 1$ . Now suppose we want to calculate $\text{rank}(\Sigma_{\{\mathsf{X}_{3}, \mathsf{X}_{4}, \mathsf{X}_{5}\}, \{\mathsf{X}_{1}, \mathsf{X}_{6}, \mathsf{X}_{7}\})$ . As the minimal way to t-separate $\{X_{3}, X_{4}, X_{5}\}$ and $\{X_{1}, X_{6}, X_{7}\}$ is by $(\{L_{1}, L_{2}\}, \emptyset)$ (or $(\emptyset, \{L_{1}, L_{2}\})$ ), the rank is $|\{L_{1}, L_{2}\}| + |\emptyset| = 2$ .

# A.3 DEFINITION OF CI SKELETON

Definition 6. A CI skeleton of $\mathbf{X}_{\mathcal{G}}$ is an undirected graph where the edge between $X_{1}$ and $X_{2}$ exists iff there does not exist a set of observed variables $\mathbf{C}$ such that $X_{1}, X_{2} \notin \mathbf{C}$ and $X_{1} \perp X_{2}|\mathbf{C}$ .

Examples of CI skeleton can be found in Example 2

# A.4 DEFINITION OF RANK-INVARIANT GRAPH OPERATOR

The definitions are as follows with examples in Appx. B.12.

Definition 7 (Minimal-Graph Operator (Huang et al., 2022)). Given two atomic covers L, P in G, we can merge L to P if the following conditions hold: (i) L is the pure children of P, (ii) all elements of L and P are latent and $|L| = |P|$ , and (iii) the pure children of L form a single atomic cover, or the siblings of L form a single atomic cover. We denote such an operator as minimal-graph operator $\mathcal{O}_{\min}(\mathcal{G})$ .

![](images/5ab4063bcedbb914faa096e26c78a16143dd419fea771d7ef0e3cc390ac5d335.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1["X₁"] --> L1["L₁"]
    X2["X₂"] --> L1
    X3["X₃"] --> L1
    X4["X₄"] --> L1
```
</details>

Figure 7: An illustrative example that highlights the motivation for using rank. When using CI, we cannot deduce that $\{X_{1}, X_{2}\}$ and $\{X_{3}, X_{4}\}$ are d-separated by $L_{1}$ as $L_{1}$ is latent, while by using rank we can.

Definition 8 (Skeleton Operator (Huang et al., 2022)). Given an atomic covers V in a graph G, for all $V \in V$ , V is latent, and all $C \in PCh_{G}(V)$ , such that V and C are not adjacent in G, we can draw an edge from V to C. We denote such an operator as skeleton operator $\mathcal{O}_{s}(\mathcal{G})$ .

# A.5 MAX-FLOW-MIN-CUT LEMMA FOR TREKS

Lemma 13 (Max-Flow-Min-Cut for Treks (Sullivan et al., 2010)). The minimal $|C_{A}| + |C_{B}|$ s.t., $(\mathbf{C}_{\mathbf{A}}|, |\mathbf{C}_{\mathbf{B}}|)$ t-separates A from B, equals to the maximum number of non-overlapping (no sided intersection) treks from A to B.

# A.6 EXAMPLE FOR THEOREM 4

![](images/9d96517e81fc079df9f23b2ee11642d2f72e873ada816c0589dff54c4caada9e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L₁"] --> X1["X₁"]
    L1 --> X2["X₂"]
    L1 --> X3["X₃"]
    L1 --> X4["X₄"]
```
</details>

(a) Illustrative graph $\mathcal{G}_1$ for Theorem 4.

![](images/16202e795dbaf897d6c9017369a6c5dd3df51e46f6ac3982014d0399c2ab5102.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L₁"] --> X1["X₁"]
    L1 --> X2["X₂"]
    L1 --> X3["X₃"]
    L1 --> X4["X₄"]
```
</details>

(b) Illustrative graph $\mathcal{G}_2$ for Theorem 4.

![](images/26f6e09f305b47ae5f9f31e9661d1c6c60b17d38ba85a8d298368aed9bc9d6bf.jpg)  
(c) Illustrative graph $\mathcal{G}_3$ for Theorem 4.   
Figure 8: Illustrative figures for Theorem 4.

Example 6. In Figure 8 (a), we can employ Theorem 4 to check whether $X_{1}$ and $X_{2}$ are adjacent. Specifically, let $A = \{X_{3}\}$ and $B = \{X_{4}\}$ , then the condition in Theorem 4 is satisfied, i.e., $\text{rank}(\Sigma_{\mathbf{A} \cup \{\mathsf{X}_{1}\}, \mathbf{B} \cup \{\mathsf{X}_{2}\}}) = \text{rank}(\Sigma_{\mathbf{A}, \mathbf{B}})$ and $\text{rank}(\Sigma_{\mathbf{A} \cup \{\mathsf{X}_{1}, \mathsf{X}_{2}\}, \mathbf{B} \cup \{\mathsf{X}_{1}, \mathsf{X}_{2}\}}) = \text{rank}(\Sigma_{\mathbf{A}, \mathbf{B}}) + 2$ . Therefore $X_{1}$ and $X_{2}$ are not adjacent.

When it comes to Figure 8 (b), the condition does not hold, and thus we cannot conclude that $X_{1}$ and $X_{2}$ are not adjacent.

Figure 8 (c) is an example to show that $\text{rank}(\Sigma_{\mathbf{A}\cup\{\mathsf{X}_{1},\mathsf{X}_{2}\},\mathbf{B}\cup\{\mathsf{X}_{1},\mathsf{X}_{2}\}) = \text{rank}(\Sigma_{\mathbf{A},\mathbf{B}}) + 2$ in the condition is important in the theorem. We need this condition to ensure that treks between A and B do not rely on $X_{1}$ or $X_{2}$ . Otherwise, as in Figure 8 (c), if we only check $\text{rank}(\Sigma_{\mathbf{A}\cup\{\mathsf{X}_{1}\},\mathbf{B}\cup\{\mathsf{X}_{2}\}) = \text{rank}(\Sigma_{\mathbf{A},\mathbf{B}})$ , we will found that it holds if we take $A = \{X_{3}\}$ and $B = \{X_{4}\}$ . Therefore, we will mistakenly conclude that $X_{1}$ and $X_{2}$ are not adjacent.

![](images/81ec59913fc86cdb8be08fd0799956398705dbd28b26f155bed8f746094d7a62.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L₁"] --> L2["L₂"]
    L1 --> L3["L₃"]
    L2 --> X3["X₃"]
    L2 --> X4["X₄"]
    L3 --> X5["X₅"]
    L3 --> X6["X₆"]
    L3 --> X7["X₇"]
    X1["X₁"] --> L2
    X2["X₂"] --> L3
```
</details>

(a) $\mathcal{G}_1$

![](images/904fd8f6e3c3101abd6d874f5dd81f4d47f3ece054146434f1eff1609ac37922.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L₁"] --> L2["L₂"]
    L1 --> L3["L₃"]
    L1 --> X1["X₁"]
    L1 --> X2["X₂"]
    L2 --> X3["X₃"]
    X1 --> X3
    X2 --> X3
```
</details>

(b) $G_{2}$ .   
Figure 9: Illustrative figures for Theorem 6.

![](images/2c35695e3ade92f03ca620537a0cff32f6c5e6b5f2688d72e97673ec27da5d9b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L₁"] --> X1["X₁"]
    L1 --> X2["X₂"]
    L1 --> X3["X₃"]
    L1 --> X4["X₄"]
```
</details>

(a) $G_{1}$

![](images/f6d875fc15f693a35bf9e9dc37d0c0416a9eb0f5a7db06ac0d5b1a865d0792e8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> X2
    X1 --> X3
    X2 --> X3
    X3 --> X4
    X3 --> X2
```
</details>

(b) $G_{1}^{\prime}$

![](images/362c4d32a927d1712b306338cb2792396ede75ffb88a71d20606bf8e97fe8281.jpg)  
(c) $\mathcal{G}_1^{\prime \prime}$ .   
Figure 10: Examples to show that $\mathcal{G}_1$ cannot be identified as $\mathcal{G}_1\mathcal{G}_1',\mathcal{G}_1''$ have the same observational rank information.

# A.7 EXAMPLE FOR THEOREM 6

Example 7. Take Figure 9 (a) as an example. Suppose we aim to calculate $\text{rank}(\Sigma_{\mathbf{A},\mathbf{B}})$ , where $A = \{L_2, L_3\}$ and $B = \{X_1, X_2\}$ . We can employ the pure children of A, $C = \{X_3, ..., X_7\}$ to do so. Specifically, we have $\text{rank}(\Sigma_{\mathbf{A},\mathbf{B}}) = \text{rank}(\Sigma_{\mathbf{C},\mathbf{B}}) = \text{rank}(\Sigma_{\{X_3, ..., X_7\}, \{X_1, X_2\}}) = 1$ .

However, in Figure 9 (b), it is not the case. this is because the number of pure children of $\{\mathsf{L}_2,\mathsf{L}_3\}$ is not enough and thus rank $(\Sigma_{\mathbf{A},\mathbf{C}})\neq |\mathbf{A}|$ . In this case $\mathbf{C} = \{\mathsf{X}_3\}$ cannot work as a surrogate for calculating rank $(\Sigma_{\mathbf{A},\mathbf{B}})$ .

# A.8 PROOF OF THEOREM 4

Proof. Assume that the condition holds and suppose $\text{rank}(\Sigma_{\mathbf{A},\mathbf{B}}) = t$ , which means that there exist t non-overlapping treks between A and B. By $\text{rank}(\Sigma_{\mathbf{A}\cup\{\mathsf{X}_{1},\mathsf{X}_{2}\},\mathbf{B}\cup\{\mathsf{X}_{1},\mathsf{X}_{2}\}) = \text{rank}(\Sigma_{\mathbf{A},\mathbf{B}}) + 2$ , we further have that these t treks do not necessarily travel across $X_{1}$ or $X_{2}$ . As such, if $X_{1}$ and $X_{2}$ are adjacent, then we must have $\text{rank}(\Sigma_{\mathbf{A}\cup\{\mathsf{X}_{1}\},\mathbf{B}\cup\{\mathsf{X}_{2}\}}) = t + 1 \neq \text{rank}(\Sigma_{\mathbf{A},\mathbf{B}})$ , which contradicts with the condition. Therefore $X_{1}$ and $X_{2}$ cannot be adjacent. □

Here we show that the condition generalizes PC's condition. SGS and PC (Spirtes et al., 2000) proposed: there exist a set of observed variables $\mathbf{C} \subseteq \mathbf{X}_{\mathcal{G}}, \mathsf{X}_1, \mathsf{X}_2 \notin \mathbf{C}$ , such that $\mathsf{X}_1 \perp \mathsf{X}_2|\mathbf{C}$ . This condition can be expressed in the form of Theorem 4, with $\mathbf{C} = \mathbf{A} = \mathbf{B}$ , because we have $\mathsf{X}_1 \perp \mathsf{X}_2|\mathbf{C}$ iff $\mathrm{rank}(\Sigma_{\mathbf{A} \cup \{\mathsf{X}_1\}, \mathbf{B} \cup \{\mathsf{X}_2\}}) = \mathrm{rank}(\Sigma_{\mathbf{A}, \mathbf{B}})$ . Plus, $\mathrm{rank}(\Sigma_{\mathbf{A} \cup \{\mathsf{X}_1, \mathsf{X}_2\}, \mathbf{B} \cup \{\mathsf{X}_1, \mathsf{X}_2\}}) = \mathrm{rank}(\Sigma_{\mathbf{A}, \mathbf{B}}) + 2$ is always true when $\mathbf{A} = \mathbf{B}$ .

# A.9 PROOF OF THEOREM 5

We first introduce Lemma 14 as follows to show that when there is no latent variable, the rank information should be aligned with what CI skeleton provides.

Lemma 14. When there is no latent variable in the graph, rank and CI are equally informative about the underlying structure, i.e., the rank-equivalence class and the Markov equivalence class are the same when there is no latent variable.

Proof. (i) As all d-sep can be stated by rank according to Lemma 10, using rank information is able to arrive at the markov equivalence class. (ii) Every element in the markov equivalence class are distributionally equivalent in terms of second order statistics. Therefore, using information from the rank of the covariance matrix cannot differentiate elements in the markove equivalence class. Taking (i) and (ii) together, we have that the rank-equivalence class and the Markov equivalence class are the same when there is no latent variable. $\square$

# Bellow is the proof of Theorem 5.

Proof. First we assume that there is no latent variable and we have that the CI skeleton among observed variables is also the skeleton of the true underlying graph G. By (ii) and (iii), we have (ii') $\forall$ distinct $A_{1}, A_{2} \in A, A_{1}, A_{2}$ are adjacent in G, (iii') $\forall A \in A, B \in B, A, B$ are adjacent in G. By (ii') and (iii'), it must hold that (iv') $\text{rank}(\Sigma_{\mathbf{A} \cup \mathbf{C}, \mathbf{B} \cup \mathbf{C}}) = |\mathbf{A}| + |\mathbf{C}|$ . However, (iv') contradicts with (iv). Therefore, there must exist at least one latent variable. □

# A.10 PROOF OF THEOREM 6

Proof. By Theorem 3 and Lemma 13, we have $\text{rank}(\Sigma_{\mathbf{A},\mathbf{B}})$ equals the maximum number of non-overlapping trek paths from A to B. As C are pure children of A and no element in B are descendants of C, every trek from C to B must travel across A. Therefore, $\text{rank}(\Sigma_{\mathbf{A},\mathbf{B}}) \geq \text{rank}(\Sigma_{\mathbf{C},\mathbf{B}})$ . If we further have $\text{rank}(\Sigma_{\mathbf{A},\mathbf{C}}) = |\mathbf{A}|$ , then the maximum number of non-overlapping trek paths from A to B equals the maximum number of non-overlapping trek paths from C to B, and thus $\text{rank}(\Sigma_{\mathbf{A},\mathbf{B}}) = \text{rank}(\Sigma_{\mathbf{C},\mathbf{B}})$ . ☐

# A.11 PROOF OF THEOREM 7

Proof. As none of elements in $X_{G} \setminus C$ are descendants of C, by Theorem 3, the minimal way to block every trek path between $C \cup X$ and $(\mathcal{X}_{\mathcal{G}} \setminus \mathcal{C}) \cup \mathcal{X}$ is by blocking all the elements of the atomic cover V and all the elements in X. As X is a subset of V, we have $\text{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, (\mathcal{X}_{\mathcal{G}} \setminus \mathcal{C}) \cup \mathcal{X}}) = |\mathbf{V}| = k$ . ☐

# A.12 EXAMPLE FOR THEOREM 8

Example 8 (Example for the uniqueness of rank deficiency in Theorem8). In Figure 1, if the current $k = 2$ , and we assume that all $k = 1$ clusters are found, and no $v$ -structure exists, the rank deficiency would uniquely map to a $k = 2$ cluster. E.g., if we take $\mathcal{C} = \{\{\mathrm{X}_6\}, \{\mathrm{X}_7\}\}$ and $\mathcal{X} = \{\{\mathrm{X}_2\}\}$ , we have $\text{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, \mathcal{X} \cup \mathcal{X}_G \setminus \mathcal{C} \setminus MDe_G(\mathcal{C})}) = \text{rank}(\Sigma_{\{\mathrm{X}_6, \mathrm{X}_7, \mathrm{X}_2\}, \{\mathrm{X}_1, \ldots, \mathrm{X}_5, \mathrm{X}_8, \ldots, \mathrm{X}_{16}\}) = 2$ . In this case, this rank deficiency uniquely relates to a cover $\mathbf{V} = \mathbf{X} \cup \mathbf{L}$ , where $\mathbf{X} = \cup_{\mathbf{X}' \in \mathcal{X}} \mathbf{X}' = \{\mathrm{X}_2\}$ , $\mathbf{L}$ is latent variable to be added with $|\mathbf{L}| = k - |\mathbf{X}| = 1$ , and $\mathcal{C} = \{\{\mathrm{X}_6\}, \{\mathrm{X}_7\}\}$ are the pure children of $\mathbf{V}$ .

In contrast, if we are searching for $k = 2$ and a 1-cluster $\mathsf{L}_4 \to \{\mathsf{X}_{14},\mathsf{X}_5\}$ has not been identified, then the condition for Theorem 8 is not satisfied and thus the uniqueness of rank deficiency does not hold: e.g., by taking $\mathcal{C} = \{\mathsf{X}_{13},\mathsf{X}_{14},\mathsf{X}_{15}\}$ and $\mathcal{X} = \{\}$ , we have rank $(\Sigma_{\{\mathsf{X}_{13},\mathsf{X}_{14},\mathsf{X}_{15}\},\{\mathsf{X}_1,\dots,\mathsf{X}_{12},\mathsf{X}_{16}\}}) = 2$ , which is deficient, and yet $\{\mathsf{X}_{13},\mathsf{X}_{14},\mathsf{X}_{15}\}$ are not from a $k = 2$ cluster. This is because this rank deficiency is not from a $k = 2$ cluster, rather, it is from the 1-cluster $\{\mathsf{X}_{14},\mathsf{X}_{15}\}$ with parent $\mathsf{L}_4$ that has not been found yet.

# A.13 PROOF OF THEOREM 8

Proof. We first show that (a) if $||\mathcal{X}|| = t = 0$ and, elements from $\mathcal{C}$ are pure children of two or more atomic covers, then we must have $\mathrm{rank}(\Sigma_{\mathcal{C}\cup \mathcal{X},\mathcal{X}\cup \mathcal{X}_{\mathcal{G}}\setminus \mathcal{C}\setminus \mathrm{MDe}_{\mathcal{G}}(\mathcal{C})}) = k + 1$ .

Proof of (a). As there is no collider between atomic covers, we have that all the elements from $\mathcal{C}$ are pure children of atomic covers. Suppose $\mathcal{C}$ can be partitioned into $\mathcal{C}_1, \ldots, \mathcal{C}_N$ , where each $\mathcal{C}_i$ are the pure children of a distinct atomic cover in $\mathcal{G}$ . If $\mathcal{C}_i$ are the pure children of an atomic cover with cardinality $k' < k$ , then we have $||\mathcal{C}_i|| \leq k'$ . If $\mathcal{C}_i$ are the pure children of an atomic cover with cardinality $k' \geq k \geq ||\mathcal{C}_i||$ ( $k \geq ||\mathcal{C}_i||$ because if $k < ||\mathcal{C}_i||$ then all elements of $\mathcal{C}$ are from the same cluster), so we also have $||\mathcal{C}_i|| \leq k'$ . Therefore, by Lemma 13 and the fact that each atomic cover has $k + 1 - t$ pure children and $k + 1$ additional neighbors, we have $\text{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, \mathcal{X} \cup \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus \text{MDe}_{\mathcal{G}}(\mathcal{C})}) = \text{rank}(\Sigma_{\mathcal{C}, \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus \text{MDe}_{\mathcal{G}}(\mathcal{C})}) = \sum_1^N ||\mathcal{C}_i|| = k + 1$ . Therefore, when $||\mathcal{X}|| = t = 0$ , the rank deficiency property does not hold when elements of $\mathcal{C}$ are from different clusters.

(b) When $||\mathcal{X}|| = t \neq 0$ , we consider a new graph $\mathcal{G}''$ , where all variables from $||\mathcal{X}||$ are removed (as well as related edges). Assume elements of $\mathcal{C}$ are from different clusters and elements of $\mathcal{X}$ are from the same atomic cover. Thus by (a), we have that the maximum number of non-overlapping treks in $\mathcal{G}''$ between $\mathcal{C}$ and $\mathcal{X}_{\mathcal{G}''} \setminus \mathcal{C} \setminus \mathrm{MDe}_{\mathcal{G}''}(\mathcal{C})$ is $k + 1 - t$ . Then we add $\mathcal{X}$ with relating edges back to the graph and thus we will have $||\mathcal{X}|| = t$ additional non-overlapping treks between $\mathcal{C} \cup \mathcal{X}$ and $\mathcal{X} \cup \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus \mathrm{MDe}_{\mathcal{G}}(\mathcal{C})$ . Therefore, we also have $\mathrm{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, \mathcal{X} \cup \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus \mathrm{MDe}_{\mathcal{G}}(\mathcal{C})}) = k + 1$ , when $||\mathcal{X}|| = t \neq 0$ and elements of $\mathcal{C}$ are from different clusters. Similarly, if elements of $\mathcal{C}$ are from the same cluster but not all elements of $\mathcal{X}$ are the parents of that cluster, we also have $\mathrm{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, \mathcal{X} \cup \mathcal{X}_{\mathcal{G}} \setminus \mathcal{C} \setminus \mathrm{MDe}_{\mathcal{G}}(\mathcal{C})}) = k + 1$ .

Taking (a) and (b) together, we have that rank deficiency holds only if all elements of $\mathcal{C}$ are from the same cluster and all elements of $\mathcal{X}$ are the parents of that cluster.

# A.14 PROOF OF THEOREM 9

In the proof of Theorem 5, we have already shown the 'if' direction. Now we are going to show the sketch of the proof for the 'only if' direction.

Proof. Suppose there is a latent variable in G. According to Condition 1, it must belong to an atomic cover, say V and we suppose that V contains n latent variables in total, rest of which are observed X. According to the definition of atomic cover, V has at least $n+1$ pure children and $n+1$ neighbours that are distinct with the $n+1$ pure children. Assume that all of them are observed. Then we can simply take A as the pure children, B as the neighbours, C as X, and thus conditions (i)-(v) will be all satisfied. If some of the pure children or neighbors of V are latent, we can simply use their pure children instead (if the pure children are still latent, use the pure children of the pure children and finally we will find enough observed pure children/descendants, as latent variables cannot be leaf nodes). Thus the conditions (i)-(v) can also be satisfied. Therefore, if there is at least a latent variable, then there must exist disjoint A,B, and C, such that (i)-(v) hold. □

# A.15 PROOF OF LEMMA 10

Proof. By Theorem 2 we have that for disjoint A,B and C, C d-separates A from B, iff there is a partition $\mathbf{C} = \mathbf{C}_{\mathbf{A}} \cup \mathbf{C}_{\mathbf{B}}$ such that $(\mathbf{C}_{\mathbf{A}}, \mathbf{C}_{\mathbf{B}})$ t-separates $\mathbf{A} \cup \mathbf{C}$ from $\mathbf{B} \cup \mathbf{C}$ . By Theorem 3, we have that $\operatorname{rank}(\Sigma_{\mathbf{A} \cup \mathbf{C}, \mathbf{B} \cup \mathbf{C}}) \leq |\mathbf{C}|$ . Plus, by the definition of treks, $\operatorname{rank}(\Sigma_{\mathbf{A} \cup \mathbf{C}, \mathbf{B} \cup \mathbf{C}}) \geq |\mathbf{C}|$ . Therefore, C d-separates A from B, iff $\operatorname{rank}(\Sigma_{\mathbf{A} \cup \mathbf{C}, \mathbf{B} \cup \mathbf{C}}) = |\mathbf{C}|$ .

![](images/c768cfb5dbad1a66f30fe690c71d38f79295b7683846edf84f5d927b75fd77d9.jpg)

# A.16 PROOF OF THEOREM 11

Proof. The sketch of the proof is as follows.

We first show that a fake cover will not influence all other found structures except itself and its neighbors in the result $G'$ . By Lemma 11 in (Huang et al., 2022), we have that a fake cover with observed descendants X in $G'$ implies a bond set in G (whose definition can be found in Huang et al. (2022)), and there is a partition of the rest of the observed variables $X_{G} \setminus X$ into two groups A and B such that A and B are d-separated by the bond set. Suppose this faker cover is V that corresponds to a set of n latent covers $\{L_{1}, ..., L_{n}\}$ in G. Then $\{L_{1}, ..., L_{n}\}$ d-separates A and B, and thus during the search we will generate two dummy covers that interact with A and B respectively, during which the rank information will not be mistaken. Then we show that in Phase 3, the fake cover can be corrected. When we refine the fake cover V, we will delete it together with its neighbours and thus the two dummy covers will be deleted. Suppose now we are searching for k clusters, as this time all the remaining $k'$ -clusters s.t. $k' < k$ have been found, the FindCausalCluster function will not generate fake cluster anymore. Therefore, the output would be corrected.

![](images/2523f990591d3a81c50f9d298b66b15054461ce5232ad48cc917c162c1e8f85e.jpg)

# A.17 PROOF OF THEOREM 12

First, we show an extension of Theorem 6 to atomic covers.

Lemma 15 (Pure Children as Surrogate for Atomic Covers). Let $\mathbf{A} = \mathbf{L} \cup \mathbf{X}$ be an atomic cover, $\mathcal{C} \subseteq PCh_{\mathcal{G}}(\mathbf{A})$ be a subset of pure children of $\mathbf{A}$ , and $\mathbf{B}_1, \mathbf{B}_2$ be two sets of variables such that for all $\mathsf{B} \in \mathbf{B}_1 \cup \mathbf{B}_2$ , $\mathsf{B} \notin De_{\mathcal{G}}(\mathcal{C})$ . We have rank $(\Sigma_{\{\mathbf{A}\} \cup \{\mathbf{B}_1\}, \mathbf{B}_2}) = \text{rank}(\Sigma_{\mathcal{C} \cup \{\mathbf{X}\} \cup \{\mathbf{B}_1\}, \mathbf{B}_2})$ , if $||\mathcal{C}|| + |\mathbf{X}| \geq |\mathbf{A}|$ .

Proof. By Lemma 13, $\mathrm{rank}(\Sigma_{\{\mathbf{A}\} \cup \{\mathbf{B}_1\},\mathbf{B}_2})$ is the max number of non-overlapping treks between $\{\mathbf{A}\} \cup \{\mathbf{B}_1\}$ and $\mathbf{B}_2$ . If $||\mathcal{C}|| + |\mathbf{X}| \geq |\mathbf{A}|$ holds, all the treks starting from $\{\mathbf{A}\} \cup \{\mathbf{B}_1\}$ can be extended to treks that start from $\mathcal{C} \cup \{\mathbf{X}\} \cup \{\mathbf{B}_1\}$ , and the max number of non-overlapping treks is the same, which means $\mathrm{rank}(\Sigma_{\{\mathbf{A}\} \cup \{\mathbf{B}_1\},\mathbf{B}_2}) = \mathrm{rank}(\Sigma_{\mathcal{C} \cup \{\mathbf{X}\} \cup \{\mathbf{B}_1\},\mathbf{B}_2})$ .

This lemma informs us that if we correctly found a cover $\mathbf{A} = \mathbf{L} \cup \mathbf{X}$ by our rules in Algorithm 2, we can calculate the rank relating to $\mathbf{A}$ by using its pure children $\mathcal{C}$ together with part of the observation

of A, i.e., X as surrogates, even though part of A, i.e., L, cannot be observed. Note that $||C||+|X|\geq|A|$ always holds as it is required when we are searching for clusters in Algorithm 2.

Next, we show that when we are searching for combinations of C and X in Algorithm 2, by leveraging the checking function NoCollider defined in Algorithm 4, the correctness of Algorithm 2 will not be influenced even though there might exist colliders in C.

Lemma 16 (Colliders in C do not harm). Suppose there exist some collider structures in graph G, e.g., there exist two atomic covers $V_{1}$ and $V_{2}$ , with A the minimal set of variables that d-separates $V_{1}$ from $V_{2}$ , and C as a collider of $V_{1}$ , $V_{2}$ . The correctness of Algorithm 2 will not be influenced by the existence of colliders in C.

Proof. In Algorithm 2, we check whether different combinations of $\mathcal{C}$ , $\mathcal{X}$ , and $\mathcal{N}$ induce rank deficiency. Suppose we take $\mathcal{C} = \{\{\mathbf{V}'_1\}, \{\mathbf{V}'_2\}, \{\mathbf{C}'\}\}$ , where $\mathbf{V}'_1 \subseteq \mathbf{V}_1$ , $\mathbf{V}'_2 \subseteq \mathbf{V}_2$ , and $\mathbf{C}' \subseteq \mathbf{C}$ and let $\mathbf{R}$ be $\mathbf{V}_1 \cup \mathbf{V}_2 \setminus \mathbf{V}'_1 \setminus \mathbf{V}'_2$ .

(i) If $|\mathbf{C}'| \leq |\mathbf{R}|$ , and rank deficiency holds, we have $|\mathbf{V}'_1 \cup \mathbf{V}'_2 \cup \mathbf{C}'| = 1 + |\mathbf{C}'| + |\mathbf{A}|$ . Therefore, we can detect $\mathbf{C}'$ in $\mathcal{C}$ by the checking function NoCollider, because by removing $\mathbf{C}'$ we have $|\mathbf{V}'_1 \cup \mathbf{V}'_2| = 1 + |\mathbf{A}|$ .

(ii) If $|\mathbf{C}'| > |\mathbf{R}|$ , and rank deficiency holds when checking $k$ , we have $|\mathbf{V}'_1 \cup \mathbf{V}'_2 \cup \mathbf{C}'| = 1 + |\mathbf{R}| + |\mathbf{A}| = k + 1$ , which means $|\mathbf{V}_1 \cup \mathbf{V}_2| = |\mathbf{V}'_1| + |\mathbf{V}'_2| + |\mathbf{R}| = 1 + |\mathbf{R}| + |\mathbf{A}| + |\mathbf{R}| - |\mathbf{C}'| \leq k$ . Therefore, by the unfolding order in Algorithm 2, $\mathbf{C}$ will be taken as the children of $\mathbf{V}_1$ and $\mathbf{V}_2$ first and thus will not induce incorrect rank deficiency.

Further, under Condition 2, we can show that the correctness of Algorithm 2 will not be influenced even though there might exist colliders in N, which is summarized in the following Lemma.

Lemma 17 (Under Condition 2, colliders in N do not harm). If Condition 2 holds, i.e., for every collider structures $V_{1}$ , $V_{2}$ , C, and A, we have $|C| + |A| \geq |V_{1}| + |V_{2}|$ , then the correctness of Algorithm 2 will not be influenced by the existence of colliders in N.

Proof. Consider a collider structure $V_{1}$ , $V_{2}$ , C, and A. The potential existence of C in N will not cause rank deficiency unless it is when $k = |C| + |A|$ . But under Condition 2, we have $|C| + |A| \geq |V_{1}| + |V_{2}|$ , so in Algorithm 2, the colliders C will be taken as the pure children of $V_{1}$ and $V_{2}$ first. This holds for every collider structure and thus the correctness of Algorithm 2 will not be influenced. □

Now we are ready to prove Theorem 12, as follows.

Proof. As the existence of colliders in X will enable more non-overlapping treks, the existence of colliders in X will not induce incorrect rank deficiency. Taking this and the above two Lemmas 16,17 into consideration, under Condition 1 and 2, the existence of colliders between atomic covers will not influence the correctness of Algorithm 2. During the search process of Phase 2, Lemma 15 allows us to test the rank involving partially observed atomic covers and thus we are able to iteratively find all clusters in G by making use of Theorem 8, with the direction of some edges undetermined. Further, Theorem 11 allows us to correct clusters induced by the violation of the assumption that when searching k clusters all $k' < k$ clusters have been found, by Phase 3 in Algorithm 5. Therefore, our Algorithm 1 including Phase 1, 2, 3 can identify the Markov equivalence class of G, up to rank invariant operations $O_{min}$ and $O_{s}$ . □

Corollary 1 is directly from Theorem 12. When there is no latent, the Markov equivalence of $\mathcal{O}_{\min}(\mathcal{O}_{s}(\mathcal{G}))$ is the Markov equivalence of G so asymptotically the output of RLCD is the same as that of PC.

# A.18 RANK TEST

We employ canonical correlations (Anderson, 1984) to calculate the rank of covariance matrices. Denote by $\alpha_{i}$ the $i$ -th canonical correlation coefficient between two sets of variables $\mathbf{A}$ and $\mathbf{B}$ , under the null hypothesis $\mathrm{rank}(\Sigma_{\mathbf{A},\mathbf{B}}) \leq r$ with $N$ sample size, the statistics $-(N - (p + q + 3)/2)\sum_{i=r+1}^{\max(|\mathbf{A}|,|\mathbf{B}|)} \log(1 - \alpha_i^2)$ is approximately $\chi^2$ distributed with $(|\mathbf{A}| - r)(|\mathbf{B}| - r)$ degrees of freedom.

![](images/f0707ab776511d7b99512e0a44760a9ac36901a4c23de93142640ea6a6c0e503.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L₁"] --> L2["L₂"]
    L1 --> L3["L₃"]
    L2 --> X2["X₂"]
    L2 --> X3["X₃"]
    L2 --> X4["X₄"]
    L3 --> X5["X₅"]
    L3 --> X6["X₆"]
    L3 --> X7["X₇"]
```
</details>

(a) Illustrative graph allowed by Pearl (1988); Zhang (2004).

![](images/c917450691368277910adcdc193b6dbffd2fcb6bcb069686b12c98a6814e13ba.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> L3
    L1 --> L4
    L1 --> X1
    L2 --> X2
    L2 --> X3
    L2 --> X4
    L3 --> X5
    L3 --> X6
    L4 --> X7
    L4 --> X8
    L4 --> X9
    L4 --> X10
```
</details>

(b) Illustrative graph allowed by Huang et al. (2022).

![](images/bd27595884df58ba26d62e798f95227b921582c1e966c0b868e0d0a3dafaeee7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1[" L₁ "] --> X1_1["X₁"]
    L1 --> X2_1["X₂"]
    L1 --> X5_1["X₅"]
    L1 --> X6_1["X₆"]
    L1 --> X7_1["X₇"]
    L2[" L₂ "] --> X3_1["X₃"]
    L2 --> X4_1["X₄"]
    L2 --> X8_1["X₈"]
    L2 --> X9_1["X₉"]
    L2 --> X10_1["X₁₀"]
    X1_1 --> X3_2["X₃"]
    X2_1 --> X3_2
    X3_1 --> X4_2["X₄"]
    X4_1 --> X8_2["X₈"]
    X5_1 --> X8_2
    X6_1 --> X8_2
    X7_1 --> X8_2
```
</details>

(c) Illustrative graph allowed by Maeda & Shimizu (2020).

![](images/ab292b5011933d9fdedff03c19ff2983632ad4e60d35e1d44256d91ae6b47747.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> X1
    L2 --> X6
    L2 --> X7
    L2 --> X8
    L2 --> X9
    L2 --> X10
    L2 --> X11
    L2 --> X12
    L2 --> X13
    L2 --> X14
    L2 --> X15
    L2 --> X16
    L2 --> X2
    L2 --> X3
    L2 --> X4
    L2 --> X5
    L2 --> X6
    L2 --> X7
    L2 --> X8
    L2 --> X9
    L2 --> X10
    L2 --> X11
    L2 --> X12
    L2 --> X13
    L2 --> X14
    L2 --> X15
    L2 --> X16
    L3 --> X3
    L3 --> X4
    L3 --> X5
    L3 --> X6
    L3 --> X7
    L3 --> X8
    L3 --> X9
    L3 --> X10
    L3 --> X11
    L3 --> X12
    L3 --> X13
    L3 --> X14
    L3 --> X15
    L3 --> X16
```
</details>

(d) Illustrative graph allowed by the proposed method.   
Figure 11: Examples of graphs that are allowed by each method.

# B ILLUSTRATIONS OF ALGORITHMS AND MORE DETAILS ABOUT DATASETS

# B.1 EXAMPLES OF GRPAHS THAT EACH METHOD CAN HANDLE

For causal discovery in the presence of latent variables, traditional wisdom often relies on strong graphical constraints for achieving the identifiability of the structure. E.g., Pearl (1988); Zhang (2004) assume that the underlying graph only follows a tree structure; Huang et al. (2022) assumes a more general latent hierarchical structures but edges among observed variables are not allowed; Maeda & Shimizu (2020) allows observed variables to be adjacent but requires that all latent variables are mutually independent.

Illustrative graphs allowed by each method are shown in Figure 11. To be specific, (a) is the illustrative graph allowed by Pearl (1988); Zhang (2004) where each cluster has only one latent variable and observed variables are not allowed to be directly related to each other. (b) is the graph allowed by Huang et al. (2022) where each cluster can have multiple latent variables. However, observed variable cannot be adjacent to each other and observed variables cannot be cause of latent variables. (c) is the graph allowed by Maeda & Shimizu (2020), where all latent variables are required to be mutually independent. (d) is the graph allowed by the proposed method, where all variables are allowed to be very flexibly related to each other.

# B.2 EXAMPLE OF VIOLATION OF FAITHFULNESS

Below, we provide a special example where faithfulness does not hold. Suppose the true underlying graph $\mathcal{G}$ is $X \xrightarrow{a} Y \xrightarrow{b} Z$ and $X \xrightarrow{c} Z$ . If the corresponding SCM is parameterized with $ab + c = 0$ , then the faithfulness assumption is violated. This is because, when $ab + c = 0$ , there exists another SCM with a graph $\mathcal{G}' : X \xrightarrow{a'} Y \xleftarrow{b'} Z$ that can generate exactly the same observational distribution as that of $\mathcal{G}$ , and thus from observational data it is impossible to differentiate $\mathcal{G}$ and $\mathcal{G}'$ (which results in $\mathrm{rank}(\Sigma_{X,Z}) = 0$ ). We note that such scenarios are very rare and classical methods like PC (Spirtes et al., 2000) cannot handle these situations either.

# B.3 EXAMPLE OF ATOMIC COVER

![](images/4796b7d9cdf2d0c73f8e12bf7b0d2f6156117fe516acd0c25f6c132e5cf99a63.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L₁"] --> L2["L₂"]
    L1 --> L3["L₃"]
    L1 --> X1["X₁"]
    L1 --> X2["X₂"]
    L2 --> X3["X₃"]
    L2 --> X4["X₄"]
    L3 --> X5["X₅"]
    L3 --> X6["X₆"]
    L3 --> X7["X₇"]
```
</details>

Figure 12: An example graph for showing atomic covers.

Example 9. Take Figure 12 as an example. Here $\{\mathsf{X}_1\}, \{\mathsf{X}_2\}, \{\mathsf{X}_3\}, \{\mathsf{X}_4\}, \{\mathsf{X}_5\}, \{\mathsf{X}_6\}$ are all atomic covers as they only contain a single observed variable. We also have $\mathbf{V} = \{\mathsf{L}_2, \mathsf{L}_3\}$ as an atomic cover. To show this, lets check whether conditions (i)-(iii) in Definition 5 are satisfied. For (i), we can let $\mathcal{C} = \{\{\mathsf{X}_3\}, \{\mathsf{X}_4\}, \{\mathsf{X}_5\}\}$ , and $||\mathcal{C}|| = 3 \geq k + 1 - t = 3$ . For (ii), we can let $\mathcal{N} = \{\{\mathsf{L}_1\}, \{\mathsf{X}_6\}, \{\mathsf{X}_7\}\}$ , and $||\mathcal{N}|| = 3 \geq k + 1 - t = 3$ . For (iii), it can be shown that both $\{\mathsf{L}_2\}$ and $\{\mathsf{L}_3\}$ cannot be an atomic cover. Therefore, $\{\mathsf{L}_2, \mathsf{L}_3\}$ is an atomic cover.

Next, let's show $\{\mathsf{L}_1\}$ is also an atomic cover. As for (i) we can take $\mathcal{C} = \{\{\mathsf{L}_2,\mathsf{L}_3\}\}$ with $||\mathcal{C}|| = 2\geq k - 1 + t = 2$ , for (ii) we can take $\mathcal{N} = \{\{\mathsf{X}_1\},\{\mathsf{X}_2\}\}$ with $||\mathcal{N}|| = 2\geq k - 1 + t = 2$ , and (iii) naturally holds as $\{\mathsf{L}_1\}$ has a single element. Therefore $\{\mathsf{L}_1\}$ is an atomic cover.

From the example it is natural to see that when we take an atomic cover (a set) as the unit, we need to use a set of covers, i.e., C, to capture the pure children of a cover.

# B.4 EXAMPLE FOR PHASE 1

We take Figure 3 (a) as an example. After Phase 1, we will find the CI skeleton $\mathcal{G}'$ . In $\mathcal{G}'$ , we have three maximal cliques that have cardinality $\geq 3$ . They are $\{\mathrm{X}_1, \mathrm{X}_2, \mathrm{X}_3, \mathrm{X}_4, \mathrm{X}_5, \mathrm{X}_6\}$ , $\{\mathrm{X}_2, \mathrm{X}_3, \mathrm{X}_7\}$ , and $\{\mathrm{X}_9, \mathrm{X}_{10}, \mathrm{X}_{11}, \mathrm{X}_{12}\}$ . Then we partition them into groups such that two cliques $\mathbf{Q}_1, \mathbf{Q}_2$ are in the same group if $|\mathbf{Q}_1 \cap \mathbf{Q}_2| \geq 2$ . Thus we have two groups of cliques $\mathcal{Q}_1 = \{\{\mathrm{X}_1, \mathrm{X}_2, \mathrm{X}_3, \mathrm{X}_4, \mathrm{X}_5, \mathrm{X}_6\}, \{\mathrm{X}_2, \mathrm{X}_3, \mathrm{X}_7\}\}$ and $\mathcal{Q}_2 = \{\{\mathrm{X}_9, \mathrm{X}_{10}, \mathrm{X}_{11}, \mathrm{X}_{12}\}\}$ . Given $\mathcal{Q}_1$ and $\mathcal{Q}_2$ , we get $\mathbf{X}_{\mathcal{Q}_1} = \cup_{\mathbf{Q} \in \mathcal{Q}_1} \mathbf{Q} = \{\mathrm{X}_1, \mathrm{X}_2, \mathrm{X}_3, \mathrm{X}_4, \mathrm{X}_5, \mathrm{X}_6, \mathrm{X}_7\}$ and $\mathbf{X}_{\mathcal{Q}_2} = \cup_{\mathbf{Q} \in \mathcal{Q}_2} \mathbf{Q} = \{\mathrm{X}_9, \mathrm{X}_{10}, \mathrm{X}_{11}, \mathrm{X}_{12}\}$ , the corresponding input to Phase 2 and 3 will be $\mathbf{X}_{\mathcal{Q}_1} \cup \mathbf{N}_{\mathcal{Q}_1} = \{\mathrm{X}_1, \mathrm{X}_2, \mathrm{X}_3, \mathrm{X}_4, \mathrm{X}_5, \mathrm{X}_6, \mathrm{X}_7, \mathrm{X}_8\}$ and $\mathbf{X}_{\mathcal{Q}_2} \cup \mathbf{N}_{\mathcal{Q}_2} = \{\mathrm{X}_9, \mathrm{X}_{10}, \mathrm{X}_{11}, \mathrm{X}_{12}, \mathrm{X}_8\}$ respectively.

We next show that if there exist disjoint $\mathbf{A},\mathbf{B},\mathbf{C}$ , s.t., (i)-(iv) in Theorem 5 hold, then $\mathbf{A} \cup \mathbf{B} \cup \mathbf{C} \subseteq \mathbf{X}_{\mathcal{Q}}$ .

If there exist disjoint A, B, C, s.t., (i)-(iv) in Theorem 5 hold. Then A itself is a clique, and for all $B \in B$ , $A \cup \{B\}$ is also a clique. Plus A and $A \cup \{B\}$ have at least two common elements as $|A| \geq 2$ . Thus, after our processing, A, B will both be subsets of a same $X_{Q}$ . Plus, by (iv), C will be a subset of $N_{Q}$ . Therefore, we have $A \cup B \cup C \subseteq X_{Q}$ .

# B.5 DETAILED DESCRIPTION AND EXAMPLE FOR PHASE 2

Our search starts with k = 1 and an input graph $G'$ (could be empty) over observed variables. Every time we successfully found rank deficiency with rank = k, we update the graph and reset k to 1. On the other hand, if no rank deficiency can be found with current k, we add k by 1 (line 5, Alg 3), as we want to ensure all $k'$ clusters, $k' < k$ , have been found when searching for k, as in Theorem 8.

During the search procedure, we maintain an active set S, which is a set of covers and is initialized as the set of observed covers $\{\{X_{1}\},...,\{X_{n}\}\}$ (line 2, Alg. 3). We test the rank deficiency over different combinations of C and X, drawn from $S'$ , where $S'$ is generated from S by unfolding some existing clusters (lines 10-11, Alg. 2). Introducing S and $S'$ has several merits: (i) When we found a new atomic cover, we want to explore its relation with existing ones. This can be achieved by adding the newly found atomic cover to the active set S (illustrated in Example 10). (ii) According to Theorem 8, we do not want any descendants of C to be on the right side of the cross-covariance matrix when testing the rank. This can be achieved by removing all the children of a newly found atomic cover from the active set S. Taking (i) and (ii) together, we update the active set by $S \leftarrow (\mathcal{S} \backslash \mathcal{C}) \cup \mathbf{P}$ (line 24, Alg. 3). (iii) We unfold S to get $S'$ , and draw combinations C and X from $S'$ instead of S. This allows the children of existing clusters to re-appear in C and X, to facilitate finding new clusters that share parents with existing ones, which will be discussed in detail later.

In addition, to establish a unique connection between rank deficiency and atomic covers, we need to avoid colliders and their descendants to appear in $S'$ , as the existence of colliders in $\mathcal{C}$ or $\mathcal{N}$ ( $\mathcal{N} \leftarrow S' \setminus (\mathcal{X} \cup \mathcal{C})$ ) might induce rank deficiency that does not indicate a correct cluster (example in Appx. B.11). To this end, we take two steps: (i) Every time we found rank deficiency, we further check whether there is a collider in $\mathcal{C}$ (by the NoCollider function described in Alg. 4), i.e., check whether there exists $\mathbf{O} \in \mathcal{C}$ s.t., $\mathbf{O}$ is a collider of $\mathcal{C} \setminus \{\mathbf{O}\}$ and $\mathcal{N}$ (line 5 in Alg. 4). If there is a collider, then we ignore the corresponding combination of $\mathcal{C}$ and $\mathcal{X}$ (line 17 in Alg. 3). (ii) We perform unfolding on $S$ to get $S'$ . That is, every time we consider $T$ , a subset of $S$ , and get $S' \leftarrow (\mathcal{S} \setminus T) \cup (\cup_{\mathbf{T} \in T} \mathrm{PCh}_{\mathcal{G}'}(\mathbf{T}))$ (lines 10-11 in Alg. 3). This allows us to reconsider the pure children of existing atomic covers when choosing combinations of $\mathcal{C}$ and $\mathcal{X}$ , and thus, colliders can be identified (illustrated in Example 10). Taking these two steps together, under the Condition 2, it can be guaranteed that our search procedure will not be affected by the existence of colliders (proof in Appx. A.17).

Example 10 (Example for Phase 2). Consider the graph in Figure 4. We start with finding atomic covers with k = 1, and we can find that $\{X_{3}\}$ is a parent of $\{X_{8}\}$ , as in Figure 4(a). At this point, no more k = 1 clusters can be found, so next we search for k = 2 clusters. Then to identify collider $\{X_{7}\}$ , we only need to consider $\{X_{7}\}$ and $\{X_{8}\}$ together as the children of $\{X_{2}, X_{3}\}$ . After finding such a relationship, we arrive at Figure 4(b), and from now on the collider $\{X_{7}\}$ will not induce unfavorable rank deficiency anymore (as it is recorded). The next step is to find the relation of $\{X_{4}\}, \{X_{5}\}, \{X_{6}\}$ with $\{L_{2}, X_{2}\}$ , by taking $X = \{\{X_{2}\}\}$ and $C = \{\{X_{4}\}, \{X_{5}\}\}$ or $\{\{X_{4}\}, \{X_{6}\}\}$ or $\{\{X_{5}\}, \{X_{6}\}\}$ , and thus conclude $\{\{X_{4}\}, \{X_{5}\}, \{X_{6}\}\}$ as the pure children of $\{L_{2}, X_{2}\}$ , as in Fig 4(c). Finally we are able to find the relationship of $\{X_{1}\}, \{L_{2}, X_{2}\}, \{X_{3}\}$ with $\{L_{1}\}$ , by taking $X = \{\}$ and C as $\{\{X_{1}\}, \{L_{2}, X_{2}\}\}, \{\{X_{1}\}, \{X_{3}\}\}$ , or $\{\{L_{2}, X_{2}\}, \{X_{3}\}\}$ , as in Figure 4(d).

We here give a more detailed example to show the procedure of Phase 2, with the underlying graph $\mathcal{G}$ showed in Figure 4(d).

Step 1. Initialize active set $S$ as $\{\{X_1\}, \ldots, \{X_8\}\}$ , $k$ as 1.

Step 2. Get $S'$ by unfolding $S$ . Currently, $S' = \{ \{ X_1 \}, ..., \{ X_8 \} \}$ . Now let $k = 1$ and $t = 1$ . Draw a set of $t$ observed covers $X \subset S' \cap X_G$ , and draw a set of covers $C \subset S' \setminus X$ , s.t., $||C|| = k - t + 1$ , and check whether rank deficiency holds. We will find that when $X = \{ \{ X_3 \} \}$ and $C = \{ \{ X_8 \} \}$ , rank deficiency holds and there is no collider detected. Therefore we draw a link from $X_3$ to $X_8$ in $G'$ , as shown in Figure 4(a). Now update the active set $S$ as $\{ \{ X_1 \}, ..., \{ X_7 \} \}$ .

Step 3. Continue searching with $k = 1$ , but no more rank deficiency can be found. Therefore, we add $k$ by 1.

Step 4. Unfold S and get $S' = \{\{X_{1}\}, ..., \{X_{8}\}\}$ . Now, k = 2 and t = 2. By drawing X and C, we will find that when $X = \{\{X_{2}\}, \{X_{3}\}\}$ and $C = \{\{X_{7}\}\}$ , rank deficiency holds and there is no collider detected. Therefore, we draw links from $X_{2}X_{3}$ to $X_{7}$ in $G'$ , as shown in Figure 4(b). Now update the active set S as $\{\{X_{1}\}, ..., \{X_{6}\}\}$ .

Step 5. Reset $k = 1$ and search for rank deficiency. No more rank deficiency can be found with $k = 1$ , and thus we add $k$ by 1.

Step 6. Get $S'$ by unfolding $S$ , and $S' = \{\{X_1\}, \ldots, \{X_6\}\}$ . When $k = 2$ and $t = 2$ , no more rank deficiency can be found. Therefore, we try $k = 2$ and $t = 1$ . By drawing $X$ and $C$ , we will find

![](images/46df516e4f24845597bdeef3da253235a85a9ab192df866ebff88216e3546bfe.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L2["L2"] --> L3["L3"]
    L2 --> L4["L4"]
    L3 --> L5["L5"]
    L3 --> L6["L6"]
    L3 --> L7["L7"]
    L3 --> X1["X1"]
    L3 --> X2["X2"]
    L3 --> X3["X3"]
    L3 --> X4["X4"]
    L3 --> L8["L8"]
    L3 --> L9["L9"]
    L3 --> L10["L10"]
    L4 --> L5
    L4 --> L6
    L4 --> L7
    L4 --> X11["X11"]
    L4 --> X21["X2"]
    L4 --> X31["X3"]
    L4 --> X41["X4"]
    L4 --> X51["X5"]
    L4 --> X61["X6"]
    L4 --> X71["X7"]
    L4 --> X81["X8"]
    L4 --> X91["X9"]
    L4 --> X101["X10"]
```
</details>

(a) The ground truth graph $\mathcal{G}$ .

![](images/dd21290bdc8556713c464eadcc62095f2b02dd7a6957fdb7dc9bcf0d002727c0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L'1"] --> L4["L'4"]
    L2["L'2"] --> L3["L'3"]
    L3["L'3"] --> L6["L'6"]
    L4["L'4"] --> L7["L'7"]
    L3["L'3"] --> L8["L'8"]
    L6["L'6"] --> L9["L'9"]
    L7["L'7"] --> X5["X5"]
    L8["L'8"] --> X7["X7"]
    L9["L'9"] --> X8["X8"]
    L6["L'6"] --> X10["X10"]
    L7["L'8"] --> X11["X11"]
    L8["L'9"] --> X12["X12"]
    L9["L'9"] --> X13["X13"]
    L6["L'6"] --> X14["X14"]
    L7["L'8"] --> X15["X15"]
    L8["L'9"] --> X16["X16"]
    L9["L'9"] --> X17["X17"]
    L6["L'6"] --> X18["X18"]
    L7["L'8"] --> X19["X19"]
    L8["L'9"] --> X20["X20"]
    L9["L'9"] --> X21["X21"]
    L6["L'6"] --> X22["X22"]
    L7["L'8"] --> X23["X23"]
    L8["L'9"] --> X24["X24"]
    L9["L'9"] --> X25["X25"]
    L6["L'6"] --> X26["X26"]
    L7["L'8"] --> X27["X27"]
    L8["L'9"] --> X28["X28"]
    L9["L'9"] --> X29["X29"]
    L6["L'6"] --> X30["X30"]
    L7["L'8"] --> X31["X31"]
    L8["L'9"] --> X32["X32"]
    L9["L'9"] --> X33["X33"]
    L6["L'6"] --> X34["X34"]
    L7["L'8"] --> X35["X35"]
    L8["L'9"] --> X36["X36"]
    L9["L'9"] --> X37["X37"]
```
</details>

(b) Algorithm output after phase 2, taken as input of RefineCausalClusters.

![](images/1c0180df8f772d0c5a849290cfb19215d18b1ea02dae4f32eabdf8782caa93ed.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["L'7"] --> B["X5"]
    A --> C["X2"]
    A --> D["X9"]
    A --> E["X7"]
    A --> F["X5"]
    A --> G["X10"]
    
    H["L'8"] --> I["X5"]
    H --> J["X2"]
    H --> K["X9"]
    H --> L["X7"]
    H --> M["X5"]
    H --> N["X10"]
    
    O["L'9"] --> P["X5"]
    O --> Q["X2"]
    O --> R["X9"]
    O --> S["X7"]
    O --> T["X5"]
    O --> U["X10"]
    
    V["X1"] --> W["X2"]
    V --> X["X3"]
    V --> Y["X4"]
    V --> Z["L'10"]
    V --> AA["L'11"]
    V --> AB["L'12"]
    V --> AC["L'7"]
    V --> AD["L'8"]
    V --> AE["L'9"]
    V --> AF["L'10"]
    
    AG["L'13"] --> AH["X4"]
    AG --> AI["L'10"]
    AG --> AJ["L'13"]
    AG --> AK["L'12"]
    
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
    style V fill:#f9f,stroke:#333
    style AG fill:#f9f,stroke:#333
```
</details>

(c) Remove $\{L_1', L_2', L_3'\}$ and its neighbours that (d) During FindCausalClusters performed at (c) are latent. Then perform FindCausalClusters. we first find $\{L_{13}'\}$ .

![](images/71b0ea9e592b2c334f594accc0241b49eb5d55283936d616bd7818416b95f685.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Left_Ch-------------
        L7["L'7"] --> X5["X5"]
        L8["L'8"] --> X7["X7"]
        L9["L'9"] --> X2["X2"]
        L10["L'10"] --> X10["X10"]
    end
    subgraph Right_Ch-------------
        L14["L'14"] --> X3["X3"]
        L13["L'13"] --> X4["X4"]
        L10["L'10"] --> X7["X7"]
        L11["L'11"] --> X7["X7"]
        L12["L'12"] --> X7["X7"]
        L12["L'12"] --> X7["X7"]
        L12["L'12"] --> X7["X7"]
    end
    subgraph Right_Ch-------------
        L16["L'16"] --> X1["X1"]
        L15["L'15"] --> X2["X2"]
        L14["L'14"] --> X3["X3"]
        L13["L'13"] --> X4["X4"]
        L10["L'10"] --> X7["X7"]
        L11["L'11"] --> X7["X7"]
        L12["L'12"] --> X7["X7"]
        L12["L'12"] --> X7["X7"]
    end
```
</details>

(e) During FindCausalClusters performed at (c), (f) During FindCausalClusters performed at (c), we we then find $\{L_{14}^{\prime}\}$ . finally find $\{L_{15}^{\prime}, L_{16}^{\prime}\}$ .   
Figure 13: Example in subfigure (a) is the real graph G. After phase 2 the output graph still contains a fake cluster, as in (b). After phase 3, the output graph will be correct, as in (f).

that when $\mathcal{X} = \{\{\mathrm{X}_2\}\}$ and $\mathcal{C} = \{\{\mathrm{X}_4\}, \{\mathrm{X}_5\}\}$ or $\{\{\mathrm{X}_4\}, \{\mathrm{X}_6\}\}$ or $\{\{\mathrm{X}_5\}, \{\mathrm{X}_6\}\}$ , rank deficiency holds and there is no collider detected. Therefore, we conclude that there is an atomic cover. As $k = 2$ but $||\mathcal{X}|| = 1$ , we need one additional latent variable to explain this atomic cover. Thus, we add a new node $\mathsf{L}_2$ to $\mathcal{G}'$ (the subscript index for $\mathsf{L}$ can be rather arbitrary as long as it is not the same as an existing one), and draw links from $\mathsf{L}_2\mathsf{X}_2$ to $\mathsf{X}_4\mathsf{X}_5\mathsf{X}_6$ in $\mathcal{G}'$ , as shown in Figure 4(c). Now, update the active set $\mathcal{S}$ as $\{\{\mathrm{X}_1\}, \{\mathrm{X}_2\}, \{\mathrm{X}_3\}, \{\mathsf{L}_2, \mathsf{X}_2\}\}$ .

Step 7. Reset $k = 1$ and search for rank deficiency. Unfold $S$ and get $S' = \{\{X_1\}, \{X_2\}, \{X_3\}, \{L_2, X_2\}\}$ . When $k = 1$ and $t = 1$ , no more rank deficiency can be found. Therefore we try $k = 1$ and $t = 0$ . By drawing $\mathcal{X}$ and $\mathcal{C}$ , we will find that when $\mathcal{X} = \{\}$ and $\mathcal{C} = \{\{X_1\}, \{X_2\}\}$ or $\{\{X_2\}, \{X_3\}\}$ or $\{\{X_1\}, \{X_3\}\}$ or $\{\{L_2, X_2\}\}$ , rank deficiency holds and there is no collider detected. Therefore, we conclude that there is an atomic cover. All the possible $\mathcal{C}$ will be merged. As $k = 1$ but $||\mathcal{X}|| = 0$ , we need one additional latent variable to explain this atomic cover. Thus, we add a new node $L_1$ to $G'$ , and draw links from $L_1$ to $X_1L_2X_2X_3$ in $G'$ , as shown in Figure 4(d), and update the active set $S$ as $\{\{L_1\}\}$ .

Step 8. From now on, no more rank deficiency can be found, and when k is sufficiently large the procedure ends. Output $G'$ , as in Figure 4(d).

# B.6 EXAMPLE FOR PHASE 3

Here, we give an example (see Figure 13) where Phase 2 may result in incorrect latent covers, and thus we need Phase 3 to characterize and refine these incorrect latent covers. Specifically, as in Figure 13), when we look for $k = 3$ clusters, none of the atomic covers $\{L_1\}, \{L_4\}, \{L_2, L_3\}$ has been discovered. Therefore, in Phase 2, when looking for $k = 3$ clusters, we will find a combination of $\mathcal{C} = \{\{X_1\}, \{X_2\}, \{X_3\}, \{X_4\}\}$ and $\mathcal{X} = \{\}$ that causes rank deficiency, and thus we will mistakenly create an atomic cover $\{L_1', L_2', L_3'\}$ with their pure children $\{X_1\}, \{X_2\}, \{X_3\}, \{X_4\}$ , as in Figure 13(b).

Fortunately, this incorrect cluster will not affect the identification of other clusters in the graph: e.g., in Figure 13(b), the covers $\{L_7', L_8', L_9'\}, \{L_{10}', L_{11}', L_{12}'\}$ are correctly found, except that the neighbors

![](images/057943a70079c5aa6899c16946c7dc25c88078d48eee0b5ccfeb2de46bfa1b5d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> L3
    L1 --> L4
    L2 --> X2
    L2 --> X3
    L2 --> X4
    L2 --> X5
    L3 --> X6
    L3 --> X7
    L4 --> X6
    L4 --> X7
```
</details>

(a) The original graph $\mathcal{G}$ .

![](images/be403bdef9aa990fb7082e60b0a9780671e3212ee74657a44e11239030f8a716.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> L3
    L1 --> X1
    L2 --> X2
    L2 --> X3
    L2 --> X4
    L2 --> X5
    L3 --> X6
    L3 --> X7
```
</details>

(b) After operators $\mathcal{O}_{\mathrm{min}}$ and $\mathcal{O}_s$ .

Figure 14: Example to show graph operators $\mathcal{O}_{\mathrm{min}}$ and $\mathcal{O}_s$ .   
![](images/6022f14941a0b57d5e00bde436db87f4525bb989d7e5aa735f1b8fff8db4ee12.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> X2
    X1 --> X3
    X1 --> X4
    X2 --> X5
    X2 --> X6
    X2 --> X7
    X3 --> X8
    X3 --> X9
    X3 --> X10
    X3 --> X11
    X4 --> X12
    X4 --> X13
    X4 --> X14
    X4 --> X15
    X4 --> X16
    X4 --> X17
    X5 --> X12
    X5 --> X13
    X6 --> X12
    X6 --> X13
    X7 --> X12
    X7 --> X13
    X8 --> X12
    X8 --> X13
    X8 --> X14
    X8 --> X15
    X8 --> X16
    X8 --> X17
    X9 --> X18
```
</details>

(a) Example 1.

![](images/a0c9c517ea13d17ca5bfeb27ec895d329ceef74a49ee5200256eaa81b72ee4ad.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X4 --> X3
    X5 --> X3
    X1 --> X2
    X3 --> X6
    X6 --> X7
    X6 --> X8
    X9 --> X10
    X10 --> X9
```
</details>

(b) Example 2.

![](images/7430c1c464f19213489dd35ccc3ecd4089ca2311b7d50139b0b1bbdf53df0b83.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> X3
    X1 --> X4
    X1 --> X5
    X1 --> X6
    X1 --> X7
    X1 --> X8
    X1 --> X9
    X1 --> X10
    X3 --> X11
    X3 --> X12
    X3 --> X13
    X3 --> X14
    X3 --> X15
    X3 --> X16
    X4 --> X17
    X4 --> X18
    X4 --> X19
    X4 --> X20
    X4 --> X21
    X4 --> X22
    X5 --> X17
    X5 --> X18
    X5 --> X19
    X5 --> X20
    X5 --> X21
    X5 --> X22
    X6 --> X17
    X6 --> X18
    X6 --> X19
    X6 --> X20
    X6 --> X21
    X6 --> X22
    X7 --> X17
    X7 --> X18
    X7 --> X19
    X7 --> X20
    X7 --> X21
    X7 --> X22
    X8 --> X17
    X8 --> X18
    X8 --> X19
    X8 --> X20
    X8 --> X21
    X8 --> X22
```
</details>

(c) Example 3.   
Figure 15: Examples of graphs that have only observed variables.

of the wrong atomic cover $\{L_{1}^{\prime}, L_{2}^{\prime}, L_{3}^{\prime}\}$ could be incorrect. This allows us to take a further look into the incorrect cluster and refine it based on Theorem 11 (the proof of which is in Appendix A.16).

As shown in Figure 13, the subfigure (a) is the underlying graph $\mathcal{G}$ . After phase 2 the output graph $\mathcal{G}'$ in (b) contains incorrect cover $\mathbf{V} = \{\mathsf{L}_1',\mathsf{L}_2',\mathsf{L}_3'\}$ . In (c), we first calculate $\hat{\mathcal{G}}$ , which is got by deleting $\mathbf{V}$ , all neighbours of $\mathbf{V}$ that are latent, and all relating edges of them from $\mathcal{G}'$ . The resulting $\hat{\mathcal{G}}$ is shown in Figure 13 (c). After that, we perform FindCausalClusters( $\hat{\mathcal{G}}$ , $\mathbf{X}$ ), and then the clusters $\{\mathsf{X}_1,\mathsf{X}_2\},\{\mathsf{X}_3\}$ , and $\{\mathsf{X}_4\}$ can be correctly found, as shown in Figure 13 (d)(e)(f).

# B.7 GRAPH EXAMPLES WITH VARIABLES ALL OBSERVED

Please refer to Figure 15.

# B.8 GRAPH EXAMPLES FOR LATENT TREE MODELS

Please refer to Figure 18.

# B.9 GRAPH EXAMPLES FOR LATENT MEASUREMENT MODELS

Please refer to Figure 17.

# B.10 GRAPH EXAMPLES FOR GENERAL LATENT MODELS

Please refer to Figure 19.

# B.11 ILLUSTRATIVE EXAMPLE OF CONSIDERING COLLIDERS IN PHASE 2

For example, in Figure 16, suppose that we have already found the cover $\mathsf{L}_1$ as the parent of cluster $\mathsf{X}_1\mathsf{X}_2$ , and $\mathsf{L}_2$ as the parent of cluster $\mathsf{X}_6\mathsf{X}_7$ . Next, we search for $k = 2$ clusters and take $\mathcal{C} = \{\{\mathsf{X}_3\}, \{\mathsf{X}_4\}, \{\mathsf{L}_2\}\}$ and $\mathcal{X} = \{\}$ , and then we have rank deficiency $\mathrm{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, \mathcal{N} \cup \mathcal{X}}) = 2$ . However, this rank deficiency does not imply a correct cluster as there is a set of collider $\{\mathsf{L}_2\}$ inside $\mathcal{C}$ . Fortunately, it can be detected by Algorithm 4. Specifically, if we take $\mathcal{C}' = \{\{\mathsf{X}_3\}, \{\mathsf{X}_4\}\}$ , we can find that $\mathrm{rank}(\Sigma_{\mathcal{C}' \cup \mathcal{X}, \mathcal{N} \cup \mathcal{X}}) = \mathrm{rank}(\Sigma_{\mathsf{X}_3\mathsf{X}_4,\mathsf{X}_1\mathsf{X}_2\mathsf{X}_5\mathsf{X}_6}) = 1$ (line 5 in Algorithm 4), which means there exists a smaller group of rank deficiency caused by removing the collider in $\mathcal{C}$ . Thus,

![](images/6fc0747858d6a31c47489b2211845b9e05988b5af250955038d2f7e42191831a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> L1
    X2 --> L1
    X3 --> L1
    X4 --> L1
    X5 --> L1
    X6 --> L1
    X7 --> L2
    X8 --> L2
    L1 --> L2
    L2 --> L2
```
</details>

Figure 16: Example of checking colliders in N.

we conclude that $\mathcal{C} = \{\{\mathrm{X}_3\}, \{\mathrm{X}_4\}, \{\mathrm{L}_2\}\}$ and $\mathcal{X} = \{\}$ is not a correct combination and will not consider them for forming a cluster (as in line 1 in Algorithm 2).

# B.12 EXAMPLES FOR GRAPH OPERATORS

Suppose a graph $\mathcal{G}$ of a latent linear model in Figure 14(a) is $\mathcal{G}$ . After applying $\mathcal{O}_{\min}(\mathcal{O}_s(\mathcal{G}))$ , we have the graph in Figure 14(b). Specifically, the $\mathcal{O}_s$ operator adds an edge from $L_2$ to $X_5$ and the $\mathcal{O}_{\min}$ operator delete $L_4$ and add an edge from $L_1$ directly to $X_6$ and $X_7$ . For $\mathcal{G}$ , such two operators will not change the rank in the infinite sample case.

# B.13 GRAPHICAL RELATIONS BETWEEN COVERS AND SET OF COVERS

The relation between covers naturally follows the relation between a set of variables. For example, in Figure 13(a), the pure children of $\{\mathsf{L}_4,\mathsf{L}_5\}$ is $\{\mathsf{X}_5,\mathsf{X}_6,\mathsf{X}_7,\mathsf{X}_8\}$ . For the relation between sets of covers, it also follows the relationship between variables. E.g., in Figure 13(a), the parents of $\{\{\mathsf{L}_4\},\{\mathsf{L}_5\}\}$ is a set of nodes $\{\mathsf{L}_1\}$ .

# B.14 DISCUSSIONS ON CHECKING COLLIDERS COMPLETELY

With our search procedure that checks colliders in Algorithm 4, we can make sure that the existence of colliders between atomic covers in $\mathcal{C}$ will not induce incorrect clustering results. However, we note that there are still chances that colliders are in $\mathcal{N}$ . If Condition 2 holds, then we can make sure that the existence of colliders in $\mathcal{N}$ will not induce fake clusters. In fact, there is a way to further check whether there exist colliders in $\mathcal{N}$ . Specifically, in the line 17 of Algorithm 2, if we have $\mathrm{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, \mathcal{N} \cup \mathcal{X}}) = k$ , and $\mathrm{NoCollider}(\mathcal{C}, \mathcal{X}, \mathcal{N})$ returns True, we can further check whether there exist a set of covers $\mathcal{N}' \subseteq \mathcal{N}$ such that $\mathcal{N}'$ consists of all the colliders between $\mathcal{C}$ and $\mathcal{N} \setminus \mathcal{N}'$ . To this end, we just enumerate all the possible subsets $\mathcal{N}'$ of $\mathcal{N}$ . If $\mathcal{N}'$ is the set of all the colliders, then it must be that (i) $\mathrm{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X}, (\mathcal{N} \setminus \mathcal{N}') \cup \mathcal{X}}) = k' < k$ , and (ii) $\mathrm{rank}(\Sigma_{\mathcal{C} \cup \mathcal{X} \cup \mathcal{N}', (\mathcal{N}) \cup \mathcal{X}}) > k' + ||\mathcal{N}'||$ .

Take Figure 16 as an example. First, we check whether Condition 2 holds. As $|\mathbf{C}| + |\mathbf{A}| = |\{\mathsf{L}_1\} | + |\{\mathsf{L}_2\}| = 2 < |\mathbf{V}_1| + |\mathbf{V}_2| = |\{\mathsf{X}_3,\mathsf{X}_4\} | + |\{\mathsf{X}_5,\mathsf{X}_6\} | = 4$ , Condition 2 does not hold. Therefore, when checking $k = 2$ , if we take $\mathcal{X} = \{\}$ , $\mathcal{C} = \{\{\mathsf{X}_3\},\{\mathsf{X}_4\},\{\mathsf{X}_5\}\}$ , and $\mathcal{N} = \{\{\mathsf{X}_1\},\{\mathsf{X}_2\},\{\mathsf{X}_6\},\{\mathsf{X}_7\},\{\mathsf{X}_8\}\}$ in line 17 of Algorithm 2, we will find that $\mathrm{rank}(\Sigma_{\mathcal{C}\cup \mathcal{X},\mathcal{N}\cup \mathcal{X}}) = k = 2$ , which implies an incorrect cluster as the cardinality of parents of $\{\{\mathsf{X}_3\},\{\mathsf{X}_4\},\{\mathsf{X}_5\}\}$ should be only 1. Fortunately, in this scenario, we can detect that $\mathcal{N}' = \{\{\mathsf{X}_7\},\{\mathsf{X}_8\}\} \subseteq \mathcal{N}$ is the set of all the colliders, by finding that (i) $\mathrm{rank}(\Sigma_{\mathcal{C}\cup \mathcal{X},(\mathcal{N}\setminus \mathcal{N}')\cup \mathcal{X}}) = 1 < k = 2$ , and (ii) $\mathrm{rank}(\Sigma_{\mathcal{C}\cup \mathcal{X}\cup \mathcal{N}',(\mathcal{N})\cup \mathcal{X}}) = 4 > 1 + ||\mathcal{N}'|| = 3$ .

As mentioned in Section 5, by adding this check function to our algorithm (specifically to line 17 in Algorithm 2 before adding C to D), we can achieve better identifiability that relies on Condition 1 only. However, that additional checking function is computationally inefficient.

# B.15 EVALUATION METRIC DETAILS

The definition of F1 is as follows. $F1 = \frac{2*Recall*Precision}{Recall+Precision}$ , Recall = $\frac{TP}{TP+FN}$ , and Precision = $\frac{TP}{TP+FP}$ , where TP, FP, and FN denote True Positive, False Positive, and False Negative, respectively.

For a fair comparison, we need to align the latent variables in the output graph $\mathcal{G}'$ of a method with the latent variables in the ground truth graph $\mathcal{G}$ . To this end, we first pad each result by adding latents

![](images/460dd1de18a88333123a7537acd6348c9c749559151ecd501032cfb546b54d1e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> X1
    L1 --> X2
    L1 --> L3
    L2 --> X3
    L2 --> L4
    L3 --> X5
    L3 --> X6
    L3 --> X7
    L4 --> X8
    L4 --> X9
    L4 --> X10
    L4 --> X11
    L4 --> X12
```
</details>

(a) Example 1.

![](images/c6371494e17a688c4a46614a40479bca64e0abd9b3d80b04e194cec6973ac6ee.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X3 --> L1
    X4 --> L1
    X1 --> L2
    X2 --> L2
    L1 --> X7
    L2 --> X5
    L1 --> X6
    L2 --> X6
    L1 --> X8
```
</details>

(b) Example 2.

![](images/85a3b0c2cb01688a72aec7ece02f7c755aa8d7f613261260e96eda07d0bfe57e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> L1
    X2 --> L1
    X3 --> L1
    X4 --> L1
    X5 --> L1
    X6 --> L1
    L1 --> L2
    L2 --> X7
    L2 --> X8
    L2 --> X9
    L2 --> X10
    L2 --> X11
    L2 --> X12
```
</details>

(c) Example 3.

![](images/d3986497bc1391adf116e099d18a69fe3eadd914690accfbeb3cbee8aee8747b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X4 --> X3
    X5 --> X3
    X1 --> X6
    X2 --> X6
    X3 --> X9
    X6 --> X7
    X9 --> X10
    X7 --> X8
    X8 --> X7
```
</details>

(d) Example 4.

Figure 17: Examples of Latent Measurement Graphs.   
![](images/360062ac1e50992820990c2442a795ba94ea3e7f320d63cb0cc32997ff30a6a4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1["X₁"] --> L4["L₄"]
    X1 --> X2["X₂"]
    X1 --> L2["L₂"]
    L4 --> X3["X₃"]
    L4 --> L3["L₃"]
    L4 --> X4["X₄"]
    L4 --> X5["X₅"]
    L4 --> L4L["L₄"]
    L4 --> X6["X₆"]
    L4 --> X7["X₇"]
    L4 --> X8["X₈"]
    L4 --> X9["X₉"]
    L4 --> X10["X₁₀"]
    L4 --> X11["X₁₁"]
    L4 --> X12["X₁₂"]
    L4 --> X13["X₁₃"]
    L4 --> X14["X₁₄"]
    L4 --> X15["X₁₅"]
    L4 --> X16["X₁₆"]
    L4 --> X17["X₁₇"]
    L4 --> X18["X₁₈"]
```
</details>

(a) Example 1.

![](images/0e30e146b96587777715c5671ac94e456dec3ad98c771a03d4ae1564841b0ff4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> L3
    L1 --> L4
    L1 --> L6
    L2 --> X1
    L2 --> X2
    L3 --> X3
    L3 --> X4
    L4 --> X5
    L4 --> X6
    L4 --> X7
    L4 --> X8
    L5 --> X9
    L5 --> X10
    L6 --> X11
    L6 --> X12
    L7 --> X13
    L7 --> X14
    L8 --> X15
    L8 --> X16
    L9 --> X17
    L9 --> X18
```
</details>

(b) Example 2.

![](images/920ec1e5138cde5b8eee54c60560d2cc24faec188bc67d474392631a0cd346a2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1["X₁"] --> L1["L₁"]
    X1 --> L2["L₂"]
    L1 --> X3["X₃"]
    L1 --> L4["L₄"]
    L2 --> X5["X₅"]
    L2 --> X6["X₆"]
    L2 --> X7["X₇"]
    L4 --> X8["X₀"]
    L4 --> X9["X₉"]
    L4 --> X10["X₁₀"]
    L4 --> X11["X₁₁"]
    L4 --> X12["X₁₂"]
    L4 --> X13["X₁₃"]
    L4 --> X14["X₁₄"]
```
</details>

(c) Example 3.

![](images/80da608b5f932daf246c118707821fdcf4e957dc37f0892d7a1cb234fc8c3f84.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> X2
    X1 --> X3
    X1 --> X4
    X2 --> X5
    X2 --> X6
    X3 --> X7
    X4 --> X8
    X5 --> X12
    X6 --> X13
    X7 --> X14
    X8 --> X15
    X8 --> X16
    X8 --> X17
    X8 --> X18
```
</details>

(d) Example 4.   
Figure 18: Examples of Latent Tree Graphs.

that have no edge to any other variables to match the number of latents in the ground truth graph. On the other hand, if the number of latents is more than that of the ground truth G, all different combinations will be tried. Finally, we try all different permutations of latent variables to test the F1 score. For each method, the final F1 score is taken as the best F1 score among all possible combinations and permutations.

# B.16 MORE DETAILS OF EXPERIMENTS ON SYNTHETIC DATA

Our code is implemented with Python 3.7. Asymptotically speaking, if the ground gruth graph is a DAG, then there will be no cycle in our result. However, in the finite sample case, rank test results could be self-contradictory. Therefore in our implementation we explicitly prevent that by checking whether cycles may occur every time before concluding a cluster. As different methods employ different statistical tests that may perform differently, the hyperparameter $\alpha$ is chosen from $\{0.1, 0.05, 0.01, 0.005\}$ in favor of each method to ensure their best performance and thus a fair comparison. For the proposed method we employ $\alpha = 0.005$ for the procedure of finding latent variables, while for the first stage we empirically find that using a rather big $\alpha$ would be better. This is because the first stage of PC is good at deleting edges and thus bad at recall, and the following procedure would expects input with high recall rather than high precision. We conduct all the experiments with single Intel(R) Xeon(R) CPU E5-2470. Our proposed method and GIN (Xie et al., 2020) take around 3 hours to finish all the experiments (three random seeds and three different sample sizes), and Hier. rank (Huang et al., 2022) takes around 1 hour. PC (Spirtes et al., 2000) and FCI (Spirtes et al., 2013) take around 10 minutes, while RCD (Maeda & Shimizu, 2020) takes around two days to finish the experiments. For GIN, RCD, and Hier. rank, we employ their original implementation while for PC and FCI we use the causal-learn python package https://causal-learn.readthedocs.io/en/latest/.

The time complexity of our proposed algorithm is upper bounded by $\mathcal{O}(l\sum_{k=1}^{K}\sum_{t=0}^{k}\binom{n}{t}\binom{n-t}{k+1-t})$ , where n is the number of measured variables, K is the cardinality of the largest cover of the estimated graph, with $K \ll n$ , and l is the number of levels of the estimated graph, with $l \ll n$ .

It is also possible to exhaustively enumerate all possible graphs and check whether one of them may be aligned with observational rank information from data. However, that would be very computationally expensive. Assume that the underlying graph consists of n measured variables and m latent variables. To conduct an exhaustive search for causal clusters, we need to enumerate all possible numbers of latent vars and then enumerate all possible structures, which results in an approximate

![](images/db2d9d8c5a880e46daa7edf0ed812d08d365e63e845bbd11f9b26443de3a4010.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L₁"] --> L3["L₃"]
    L1 --> L4["L₄"]
    L1 --> X1["X₁"]
    L2["L₂"] --> L3
    L2 --> X2["X₂"]
    L2 --> X3["X₃"]
    L2 --> L5["L₅"]
    L2 --> X4["X₄"]
    L2 --> X5["X₅"]
    L3 --> X6["X₆"]
    L3 --> X7["X₇"]
    L4 --> X8["X₈"]
    L4 --> X9["X₉"]
    L4 --> X10["X₁₀"]
    L4 --> X11["X₁₁"]
    X2 --> X12["X₁₂"]
    X2 --> X13["X₁₃"]
    X3 --> X14["X₁₄"]
    X3 --> X15["X₁₅"]
    X4 --> X16["X₁₆"]
    X4 --> X17["X₁₇"]
    X6 --> X10
    X7 --> X10
    X8 --> X10
    X9 --> X10
    X10 --> X11
    X11 --> X12
    X12 --> X13
    X13 --> X14
    X14 --> X15
    X15 --> X16
    X16 --> X17
```
</details>

(a) Example 1.

![](images/fbac750c4bf2d9c23d0d7f761928784043c5e267d6bb279ebfd35ff07d8c7ac9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> X1
    L1 --> X3
    L1 --> L4
    L1 --> X5
    L2 --> X6
    L2 --> X7
    L2 --> X8
    L2 --> X9
    L2 --> X10
    L2 --> X11
    L2 --> X12
    L2 --> X13
    L2 --> X14
    L2 --> X15
    L2 --> X16
    X1 --> L3
    X3 --> L4
    X4 --> L5
    X5 --> L4
    X5 --> X16
```
</details>

(b) Example 2.

![](images/3bf6fad54027fd6b071bf2a47e069ba3f667c8f41e4e65eeac3f14cf943a717d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> X2
    X2 --> X3
    X1 --> L1
    X1 --> L2
    X2 --> X4
    X2 --> X5
    X3 --> X6
    X3 --> L3
    X3 --> X7
    L1 --> X8
    L1 --> X9
    L2 --> X10
    L2 --> X11
    L2 --> X12
    L3 --> X13
    L3 --> X14
    L3 --> X15
    L3 --> X16
```
</details>

(c) Example 3.

![](images/5d6cac83a73ec2e9e57853f7bfd2b42c5cf713d26b80ccd987bae7f9b0601779.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> X3
    X1 --> X4
    X1 --> X5
    X1 --> X6
    X1 --> X7
    X1 --> X8
    X1 --> X9
    X1 --> X10
    X3 --> X11
    X3 --> X12
    X3 --> X13
    X3 --> X14
    X3 --> X15
    X3 --> X16
    X4 --> X11
    X4 --> X12
    X4 --> X13
    X4 --> X14
    X4 --> X15
    X4 --> X16
    X5 --> X11
    X5 --> X12
    X5 --> X13
    X5 --> X14
    X5 --> X15
    X5 --> X16
    X6 --> X17
    X6 --> X18
    X6 --> X19
    X6 --> X20
    X6 --> X21
    X6 --> X22
    X7 --> X17
    X7 --> X18
    X7 --> X19
    X7 --> X20
    X7 --> X21
    X7 --> X22
    X8 --> X17
    X8 --> X18
    X8 --> X19
    X8 --> X20
    X8 --> X21
    X8 --> X22
    X9 --> X20
    X9 --> X21
    X9 --> X22
```
</details>

(d) Example 4.   
Figure 19: Examples of Latent General Graphs.

total of $\sum_{i=1}^{m}3^{(i+n)(i+n-1)/2}$ possible combinations. In our synthetic data, which comprises an average of 15 measured variables and 4 latent variables, the mere act of enumeration already demands $10^{66}$ seconds (in our Python environment), which is computationally unacceptable.

# B.17 DETAILED INFORMATION OF THE BIG FIVE PERSONALITY DATASET

Data was collected through an interactive online personality test https://openpsychometrics.org/. Participants were informed that their responses would be recorded and used for research at the beginning of the test and asked to confirm their consent at the end of the test. Items were rated on a five-point scale: 1=Disagree, 2=Slightly disagree, 3=Neutral, 4=Slightly agree, 5=Agree (0=missed). Datapoints with missing values have been filtered out. Some additional information is also collected including Race, Age, and Gender but are not used in our experiment. The Markov equivalence class of Figure 5 is generated by using our proposed method, while we further apply GIN (Xie et al., 2020) to determine directions between latent variables. The five personality dimensions are Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism (O-C-E-A-N). Below are the raw questions. E.g., E1 denotes the first question for the Extraversion score.

E1 I am the life of the party.

E2 I don't talk a lot.

E3 I feel comfortable around people.

E4 I keep in the background.

E5 I start conversations.

E6 I have little to say.

E7 I talk to a lot of different people at parties.

E8 I don't like to draw attention to myself.

E9 I don't mind being the center of attention.

E10 I am quiet around strangers.

N1 I get stressed out easily.

N2 I am relaxed most of the time.

N3 I worry about things.

N4 I seldom feel blue.

N5 I am easily disturbed.

N6 I get upset easily.

N7 I change my mood a lot.

N8 I have frequent mood swings.

N9 I get irritated easily.

N10 I often feel blue.

A1 I feel little concern for others.   
A2 I am interested in people.   
A3 I insult people.   
A4 I sympathize with others' feelings.   
A5 I am not interested in other people's problems.   
A6 I have a soft heart.   
A7 I am not really interested in others.   
A8 I take time out for others.   
A9 I feel others' emotions.   
A10 I make people feel at ease.   
C1 I am always prepared.   
C2 I leave my belongings around.   
C3 I pay attention to details.   
C4 I make a mess of things.   
C5 I get chores done right away.   
C6 I often forget to put things back in their proper place.   
C7 I like order.   
C8 I shirk my duties.   
C9 I follow a schedule.   
C10 I am exacting in my work.   
O1 I have a rich vocabulary.   
O2 I have difficulty understanding abstract ideas.   
O3 I have a vivid imagination.   
O4 I am not interested in abstract ideas.   
O5 I have excellent ideas.   
O6 I do not have a good imagination.   
O7 I am quick to understand things.   
O8 I use difficult words.   
O9 I spend time reflecting on things.   
O10 I am full of ideas.

# B.18 MORE ANALYSIS OF THE RESULTS FOR THE BIG FIVE

A prevalent theory of personality is that personality dimensions (factors or traits) are latent causes of the responses to personality inventory items, which are indicators of the latent construct. For instance, extraversion yields high scores for the indicators "I like to go to parties" and "I like people." Thus, the responses to inventory items are outcomes of one's position on the latent dimension. However, there is also the suggestion of a network perspective in which personality structure is viewed in terms of microcausal connections in a complex network (Wright, 2017) and personality dimensions emerge out of the connectivity structure (Cramer et al., 2012). This is a radical divergence from the conventional viewpoint that dimensions are causes of the relevant indicators. For instance, a network would show that instead of being two distinct markers of extraversion, one might say "I like to go to parties" because "I like people" (Cramer et al., 2012); or in the case of openness, "I am full of ideas" because "I have a vivid imagination". Our result in Figure 5 indicates, however, that our method adheres to a network perspective while identifying groups of closely connected items that are predictable under a latent dimension model. Further, it can be observed that causal links occur between latent dimensions, between observed indicators, and among latents and indicators. Following are interesting aspects of our results.

(i) L1, L2, L3 and L5. While L1, L2, and L3 clearly delineate conscientiousness, agreeableness, and extraversion as causes of the corresponding item responses, it is different in the case of openness (L5). Openness to experience can lead to excellent ideas brought about by active imagination, reflection, and understanding of things. Moreover, those who develop a rich vocabulary will have the propensity to think critically and read more, behaviors that also birth ideas.   
(ii) L1→L6→L3. Conscientiousness and openness are most frequently associated with achievement (Gatzka, 2021). In our results, those who are organized, efficient at tasks, thorough, systematic, or exacting will comprehend things quickly, demonstrate sophistication in language, or are good at

introspection. These could instill a sense of confidence and assurance that encourages assertive, verbal, and bold behaviors, among other extraversion markers.

(iii) L1→L2→L3. People who score highly on conscientiousness are frequently perceived as perfectionists, high achievers, overly focused on personal goals, preoccupied with flawless task execution, overly demanding, and headstrong (Le et al., 2011)(Curşeu et al., 2019). Our findings suggest that due to such behaviors, highly conscientious individuals would judge other people on their accomplishments and results, without giving consideration to others' feelings. Consequently, they act distant, uncommunicative, or unsociable. These can be reasons why they have been found not to engage in group behaviors that lead to straining relationships and tend to take criticism poorly (Curşeu et al., 2019), as well as refrain from conversing about interpersonal issues because they have no bearing on achieving task objectives (Curşeu et al., 2019). On the other hand, those who are both highly conscientious and agreeable put the needs of others before their own (Lord, 2007), at times to the point of pleasing others by overlooking their mistakes, doing things for others because they cannot say no, not disclosing performance gaps and withholding dissident opinions out of aversion to conflict and a lack of competitiveness (Graziano & Tobin, 2002)(Howard & Howard, 2010)(Curşeu et al., 2019). Thus, they will resolve problems on their own, not to draw attention to themselves, have little to say, or would rather stay in the background in order to be integrated. Those who score low on agreeableness are perceived to be unempathetic, unfriendly, and untrustworthy, consequently leading to introvertive behaviors as well.

(iv) L1 and L3 together as common causes. Conscientious individuals who care about being liked by others despite being focused, detail-oriented, and exacting, will make efforts to make people feel at ease amid these behaviors. Otherwise, conscientious individuals who care not about what others think of them could be quick to lambast others if they perform poorly. Low-conscientious people, those who are disorganized, messy, sloppy, and negligent, will also tend to make people at ease in order to remain in their good graces.

(v) No latent variable for neuroticism. Our method did not discover any latent variable that is supposed to correspond to neuroticism. One possible interpretation would be that the question "[N10]: I often feel blue." is designed so well that it fully captured the sense of neuroticism.

(vi) Responses to indicators influence other responses. It is the question items, not the latent dimensions, that can be perceived to have caused the succeeding responses, as in the case of N10→N8→N7, N10→O9, O2→O4, and O1→O8, all of which are plausible. Mood swings are common with depression, and frequent mood swings cause emotions to fluctuate rapidly and intensely, switching between positive and negative emotions. Some people may find themselves reflecting a lot on things because they are trying to figure out what makes them frequently feel blue and how to cope with it. A person who has trouble understanding abstract concepts is unlikely to be particularly interested in them. Finally, one who has a rich vocabulary will not be constrained from using unusual words that are hard to comprehend.

# C RELATED WORK AND BROADER IMPACTS

# C.1 RELATED WORK

Causal discovery aims to identify causal relationships from observational data. Most existing approaches are based on the assumption that there are no latent confounders (Spirtes et al., 2000; Chickering, 2002b; Shimizu et al., 2006a; Hoyer et al., 2009; Zhang & Hyvärinen, 2009), and yet this assumption barely holds for real-life problems. Thus, causal discovery methods that can handle the existence of latent variables are crucial. Existing causal discovery methods for handling latent variables can be categorized into the following folds.

(i) Conditional independence constraints. The FCI algorithm (Spirtes et al., 2000) and its variants (Colombo et al., 2012; Pearl, 2000; Akbari et al., 2021). This line of work checks conditional independence over observed variables to identify the causal structure over observed variables up to a maximal ancestral graph. They can deal with both linear and nonlinear causal relationships, but there are large indeterminacies in their results, e.g., the existence of an edge and confounders. Plus, they cannot consider causal relationships between latent variables. Based on CI tests, Triantafillou & Tsamardinos (2015) proposes a method that can co-analyze multiple datasets that share common

variables and sort the significance tests to address conflicts from statistical errors. (ii) Tetrad condition. This line of work makes use of the rank constraints of every $2 \times 2$ off-diagonal sub-covariance matrix to locate latent variables and thus find the causal skeleton based on linear relationships between variables (Silva et al., 2006; Kummerfeld & Ramsey, 2016; Wang, 2020; Pearl, 1988). One limitation of this line of work is that they assume each measured variable is influenced by only one latent parent, and each latent variable must have more than three pure measured children. (iii) Matrix decomposition. This line of work proposes to decompose the precision matrix into a low-rank matrix and a sparse matrix, where the former represents the causal structure from latent variables to measured variables and the latter represents the causal structure over measured variables, under certain assumptions (Chandrasekaran et al., 2011; 2012; Anandkumar et al., 2013). E.g., Anandkumar et al. (2013) decomposed the covariance matrix into a low-rank matrix and a diagonal matrix, by assuming three times more measured variables than latent variables. (iv) Over-complete independent component analysis (ICA). Over-complete ICA allows more source signals than observed signals, and thus can be used to learn the causal structure with latent variables (Shimizu et al., 2009), and yet they normally do not consider the causal structure among latent variables. The estimation of over-complete ICA models could be hard to reach global optimum without further assumptions (Entner & Hoyer, 2010; Tashiro et al., 2014). (v) Generalized independent noise (GIN). The GIN condition is an extension of the independent noise condition when latent variables exist. Based on non-gaussianity it leverages higher-order statistics to identify latent structures. E.g., Xie et al. (2020) allows multiple latent parents behind every pair of observed variables and can identify causal directions among latent variables, and yet it requires at least twice measured children as latent variables. Dai et al. (2022) proposes a transformed version of GIN to handle measurement errors. (vi) Mixture oracles-based. Kivva et al. (2021) proposes a mixture oracles-based method to identify the causal structure in the presence of latent variables where the causal relationships can be nonlinear. It is based on assumptions that the latent variables are discrete and each latent variable has measured variables as children. (vii) Rank deficiency. Recently Huang et al. (2022) proposes to leverage rank deficiency of sub-covariance of observed variables to find the underlying causal structure in the presence of latent variables. Our method differs in that we consider a more general setting, i.e., we allow hidden variables can be causally related to each other, form a hierarchical structure (i.e., the children of hidden variables can still be hidden), and even serve as both confounders and intermediate variables for observed variables. Our graphical condition for identifiability also generalizes the condition in (Huang et al., 2022) to cases where edges between observed variables are allowed. (viii) Heterogeneous data. Huang\* et al. (2020) considere a special type of latent confounders that can be represented as a function of domain index or a smooth function of time. This line of work makes use of domain index or time index as a surrogate to remove confounders' influence and consequently identify causal structure over observed variables. (ix) Score based. Agrawal et al. (2021) propose a score-based method for latent variable causal discovery, by assuming additional structure among latent confounders.

The most related work to our method is Hier. Rank Huang et al. (2022). Compared to Hier. Rank, our graphical conditions are not only strictly but also much weaker. Our conditions are strictly weaker in the sense that Hier. Rank can be taken as a special case of the proposed method by disallowing direct edges between observed variables, which is formally captured by our Corollary 1. Our conditions are much weaker in the sense that, basically we allow latent variables and observed variables to be flexibly related and exist everywhere in a graph, which is illustrated in Figure 11 (b) v.s. Figure 11 (d). The reason why we are able to identify these latent structures that Hier. Rank cannot identify, lies in that we utilize the rank constraints in a more flexible and comprehensive fashion. Specifically, Hier. Rank only uses the part of the rank information $\mathrm{rank}(\Sigma_{\mathbf{A},\mathbf{B}})$ where $\mathbf{A} \cap \mathbf{B} = \emptyset$ , while the proposed method uses the $\mathrm{rank}(\Sigma_{\mathbf{A},\mathbf{B}})$ where $\mathbf{A}$ and $\mathbf{B}$ are rather arbitrary and thus more t-separations can be inferred. The extra graphical information allows us to make use of, e.g., Lemma 10 for identifying edges between observed variables, and Theorem 8 to identify atomic covers that are partially hidden partially observed.

Table 4: Structural Hamming Distance (SHD) of compared methods on different types of latent graphs where the values are averaged over three random seeds. The smaller the better. 

<table><tr><td colspan="2"></td><td colspan="6">SHD for skeleton among all variables  $V_G$  (both  $X_G$  and  $L_G$ )</td></tr><tr><td colspan="2">Algorithm</td><td>Ours</td><td>Hier. rank</td><td>PC</td><td>FCI</td><td>GIN</td><td>RCD</td></tr><tr><td rowspan="3">Latent+tree</td><td>2k</td><td>6.9</td><td>9.3</td><td>23.7</td><td>23.7</td><td>20.5</td><td>22.2</td></tr><tr><td>5k</td><td>3.2</td><td>9.0</td><td>25.0</td><td>24.3</td><td>21.2</td><td>23.5</td></tr><tr><td>10k</td><td>0.7</td><td>9.0</td><td>25.1</td><td>24.3</td><td>20.0</td><td>24.0</td></tr><tr><td rowspan="3">Latent+measm</td><td>2k</td><td>4.6</td><td>8.1</td><td>14.3</td><td>14.7</td><td>10.8</td><td>15.4</td></tr><tr><td>5k</td><td>3.8</td><td>7.7</td><td>15.0</td><td>15.0</td><td>9.2</td><td>16.2</td></tr><tr><td>10k</td><td>2.9</td><td>7.4</td><td>15.5</td><td>14.8</td><td>9.2</td><td>16.0</td></tr><tr><td rowspan="3">Latent general</td><td>2k</td><td>27.1</td><td>28.1</td><td>38.0</td><td>37.6</td><td>36.4</td><td>36.5</td></tr><tr><td>5k</td><td>23.0</td><td>26.0</td><td>38.2</td><td>36.8</td><td>33.8</td><td>32.5</td></tr><tr><td>10k</td><td>21.4</td><td>26.0</td><td>39.0</td><td>37.1</td><td>34.1</td><td>36.1</td></tr></table>

# D ADDITIONAL INFORMATION

# D.1 EMPIRICAL RESULT USING SHD

In this section we further show the performance of each method using the Structural Hamming Distance (SHD). As shown in Table 4, The SHD of the proposed RLCD method to the ground truth is consistently smaller than all comparative methods under all the settings, which again validates RLCD in the finite sample cases.

# D.2 VIOLATION OF GRAPHICAL CONDITIONS

In this section, we shall discuss what would the output of the proposed method be when graphical conditions 12 are not satisfied.

(i) The result is correct but uninformative. For instance, in Figure 20 (a), $L_1L_2$ do not belong to any atomic cover, as they do not have enough pure children plus neighbours. Thus, Condition 1 is not satisfied. In this scenario, though the output of the proposed method is not informative, the result is correct. It is not informative in the sense that the result in Figure 20 (b) fails to inform us the existence of latent variables. The result is correct in the sense that it correctly outputs the CI skeleton, which is rank-equivalent to the ground truth $\mathcal{G}_1$ . In other words, the output graph $\mathcal{G}_1'$ and the ground truth $\mathcal{G}_1$ are able to entail the same set of observational rank constraints and no algorithm can differentiate them solely by rank information.

(ii) The result is correct and informative, though it uses a more compact graph as the representation of the rank-equivalence class. An example is given in Figure 20 (c), where $L_{4}$ does not belong to any atomic cover (as no enough pure children plus neighbours), and thus $G_{2}$ does not satisfy Condition 1. However, the proposed RLCD still outputs the correct and informative result $G_{2}^{\prime}$ . The output $G_{2}^{\prime}$ is correct in the sense that $G_{2}^{\prime}$ and $G_{2}$ are rank-equivalent, i.e., they entail the same set of observational rank constraints, and the local violation of Condition 1 does not harm the correctness of other substructures. The output is also informative as $G_{2}^{\prime}$ is a compact representation of the rank-equivalence class that contains $G_{2}$ ; one can easily infer many other members of the class from $G_{2}^{\prime}$ .

(iii) The result is incorrect, but from the result we can infer that the conditions are violated. An example is given in Figure 20 (e) where for the v structure in $\mathcal{G}_3$ we have $|\{\mathrm{L}_1,\mathrm{L}_2\} | + |\{\mathrm{L}_3\} |>$ $|\{\mathrm{X}_1\} | + |\{\mathrm{X}_8\} |$ and thus Condition 2 is not satisfied. In this scenario, RLCD will output $\mathcal{G}_3^{\prime}$ , which is incorrect. However, we can easily infer that this result is abnormal, as the result is not consistent with the CI skeleton: in $\mathcal{G}_3^\prime$ , there are 3 subgroups that are not connected to each other, while the CI skeleton would inform us that all the variables are directly or indirectly connected.

We note that the above analysis holds in the large sample limit. In the finite sample cases, the result of statistical tests could be incorrect and thus self-contradictory.

# D.3 TRIANGLE STRUCTURE

The definition of triangle is as follows. If three variables are mutually adjacent, then they form a triangle. For example, in Figure 21, $\mathsf{L}_1, \mathsf{L}_3, \mathsf{L}_4$ form a triangle and $\mathsf{X}_5, \mathsf{X}_6, \mathsf{X}_7$ form a triangle. Our

![](images/6b6fadec654247f84b178354d0d556296816ac4e4d73fefc74f4c3768384f03c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L1"] --> X1["X1"]
    L1 --> X2["X2"]
    L1 --> X3["X3"]
    L1 --> X4["X4"]
    L1 --> X5["X5"]
    L2["L2"] --> X1
    L2 --> X2
    L2 --> X3
    L2 --> X4
    L2 --> X5
```
</details>

![](images/dcf7bdc8366176172d0e69c09e2b5c45373309259ae8643548a4a0461a7eeba5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> X2
    X1 --> X3
    X1 --> X4
    X1 --> X5
    X2 --> X1
    X3 --> X2
    X4 --> X3
    X5 --> X4
    X5 --> X1
```
</details>

(a) In $\mathcal{G}_1$ , Condition 1 is not satisfied, in (b) Given $\mathcal{G}_1$ , RLCD outputs $\mathcal{G}_1'$ , which the sense that $\mathrm{L}_1\mathrm{L}_2$ do not have enough is not informative but correct as they are pure children and neighbours. rank-equivalent.

![](images/077bfdcbff4b1c3b837ee6cbcbf963e456076c07d7c14dc392aeef0abe966780.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> L3
    L1 --> X1
    L1 --> L4
    L2 --> X2
    L2 --> X3
    L2 --> X4
    L2 --> X5
    L3 --> X2
    L3 --> X3
    L3 --> X4
    L3 --> X5
    L3 --> X6
    L4 --> X6
    L4 --> X7
```
</details>

![](images/71d6f97ee0ff78d4e4584c841151bd3432c27928bd82cb9c11f64d7e747d412f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> L3
    L1 --> X1
    L2 --> X2
    L2 --> X3
    L2 --> X4
    L2 --> X5
    L3 --> X4
    L3 --> X5
    L3 --> X6
    L3 --> X7
```
</details>

(c) In $\mathcal{G}_2$ , Condition 1 is not satisfied, in (d) Given $\mathcal{G}_2$ , RLCD outputs $\mathcal{G}_2'$ , which the sense that $\mathrm{L}_4$ does not have enough is correct and informative, though not expure children and neighbours. Exactly the same as $\mathcal{G}_2$ .

![](images/af8d00fd971f64cd1be4e59bbb847fa232771a36e2b65dc01e409d6cbd2ac568.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1["X₁"] --> L1["L₁"]
    X1 --> L2["L₂"]
    X1 --> L3["L₃"]
    X1 --> X2["X₂"]
    X1 --> X3["X₃"]
    L1 --> X4["X₄"]
    L2 --> X5["X₅"]
    L3 --> X6["X₆"]
    L3 --> X7["X₇"]
    L3 --> X8["X₈"]
    X4 --> X9["X₉"]
    X5 --> X9
    X6 --> X9
    X7 --> X10["X₁₀"]
    X8 --> X10
    X9 --> X10
    X10 --> X8
```
</details>

![](images/8df4349e91fa104915e1969b144bc87d0bf12eafd3059a2135e5fb3412493428.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L4["L4"] --> L1["L1"]
    L4 --> L2["L2"]
    X1["X1"] --> L1
    X1 --> L3["L3"]
    X1 --> X2["X2"]
    X1 --> X3["X3"]
    L1 --> X4["X4"]
    L1 --> X5["X5"]
    L2 --> X6["X6"]
    L2 --> X7["X7"]
    L3 --> X8["X8"]
    X2 --> X9["X9"]
    X3 --> X10["X10"]
    L1 --> X4
    L2 --> X5
    L3 --> X6
    L4 --> X4
    L5 --> X5
    L6 --> X7
    L7 --> X8
    L3 --> X9
    L3 --> X10
```
</details>

(e) In $G_{3}$ , the Condition 2 is not satisfied, (f) Given $G_{3}$ , RLCD outputs $G_{3}^{\prime}$ , which is because $|\{L_{1}, L_{2}\}| + |\{L_{3}\}| > |\{X_{1}\}| +$ incorrect but we can infer from the result $|\{X_{8}\}|$ . that conditions are violated.

Figure 20: Examples to show that even when graphical conditions are not satisfied, the proposed RLCD can provide the correct result (though may be uninformative), or infer that the condition is violated.   
![](images/28dbc132a0135c09d18f9ee08d4a0881a50cdb6cd027a77f24c1464a39955f1c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> L3
    L1 --> L4
    L2 --> X1
    L2 --> X2
    L2 --> X3
    L2 --> X4
    L3 --> X1
    L3 --> X2
    L3 --> X3
    L3 --> X4
    L4 --> X5
    L4 --> X6
    L4 --> X7
```
</details>

Figure 21: An illustrative example to show the triangle structure.

Condition 1 rules out the case of the triangle among $L_{1}, L_{3}, L_{4}$ as it involves latent variables, but does not rule out the triangle among $X_{5}, X_{6}, X_{7}$ .

![](images/07641a2ff204ad61436aa21c0658555e0016deee8fe3d99103a56556e9151777.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> X1
    L2 --> X2
    L2 --> L3
    L2 --> X3
    L2 --> X4
    L2 --> X5
    X2 --> X6
    X2 --> X7
    X2 --> X8
    X2 --> X9
    X2 --> X10
    X3 --> X11
    X3 --> X12
    X3 --> X13
    X3 --> X14
    X3 --> X15
    X4 --> X16
    L3 --> X10
    L3 --> X11
    L3 --> X12
    L4 --> X13
    L4 --> X14
    L4 --> X15
    L5 --> X16
```
</details>

(a) Input $\mathcal{G}$ , a latent general graph.   
![](images/3408bac9dc51ba069c2203256412636dec3c1e809213a307a280f7a2532073cf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1 --> L2
    L1 --> X1
    L1 --> X2
    L2 --> X6
    L2 --> X7
    L2 --> X8
    L2 --> X9
    L2 --> X10
    L2 --> X11
    L2 --> X12
    L2 --> X13
    L2 --> X14
    L2 --> X15
    L2 --> X16
    L2 --> X2
    L2 --> X3
    L2 --> X4
    L2 --> X5
    L3 --> X10
    L3 --> X11
    L3 --> X12
    L3 --> X13
    L3 --> X14
    L3 --> X15
    L3 --> X16
    L4 --> X10
    L4 --> X11
    L4 --> X12
    L4 --> X13
    L4 --> X14
    L4 --> X15
    L4 --> X16
```
</details>

(b) The output of RLCD, $\mathcal{G}_1$ .

![](images/aac7ae14e0d9aac8784f35183bc4a6cbcc56797ecbd24d8613b4bb9472821fde.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    L1["L1"] --> X6["X6"]
    L1 --> X7["X7"]
    L1 --> X8["X8"]
    L1 --> X9["X9"]
    L2["L2"] --> X6
    L2 --> X7
    L2 --> X8
    L2 --> X9
    L3["L3"] --> X4["X4"]
    L3 --> X5["X5"]
    L4["L4"] --> X3["X3"]
    L4 --> X4
    L4 --> X5
    L1 --> X10["X10"]
    L2 --> X10
    L3 --> X10
    L4 --> X10
    L1 --> X11["X11"]
    L2 --> X11
    L3 --> X11
    L4 --> X11
    L1 --> X12["X12"]
    L2 --> X12
    L3 --> X12
    L4 --> X12
    L1 --> X13["X13"]
    L2 --> X13
    L3 --> X13
    L4 --> X13
    L1 --> X14["X14"]
    L2 --> X14
    L3 --> X14
    L4 --> X14
    L1 --> X15["X15"]
    L2 --> X15
    L3 --> X15
    L4 --> X15
    L1 --> X16["X16"]
    L2 --> X16
    L3 --> X16
    L4 --> X16
```
</details>

(c) The output of Hier. Rank, $\mathcal{G}_2$ .

![](images/54dc69da514747f57fd1d9f54dfd8ed57e5acd554da2908f973717c582b43da6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> X2
    X2 --> X6
    X2 --> X7
    X2 --> X8
    X2 --> X9
    X2 --> X10
    X2 --> X11
    X2 --> X12
    X2 --> X13
    X2 --> X14
    X2 --> X15
    X2 --> X16
    X3 --> X4
    X4 --> X5
    X5 --> X16
    X6 --> X10
    X7 --> X10
    X8 --> X10
    X9 --> X10
    X10 --> X11
    X11 --> X12
    X12 --> X13
    X13 --> X14
    X14 --> X15
    X15 --> X16
```
</details>

(d) The output of FCI, $\mathcal{G}_3$ .

![](images/b6e09517ae435c819c0a25fd2ffe5f0891056f35397e76b2835a6efebea7ed1f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    X1 --> L1
    X2 --> L1
    X3 --> L1
    X4 --> L1
    X5 --> L1
    L1 --> L2
    L2 --> L1
    L1 --> L4
    L2 --> L4
    L4 --> L1
    L4 --> L3
    L4 --> L4
    L4 --> L5
    L4 --> L6
    L4 --> L7
    L4 --> L8
    L4 --> L9
    L4 --> L10
    L4 --> L11
    L4 --> L12
    L4 --> L13
    L4 --> L14
    L4 --> L15
    L4 --> L16
```
</details>

(e) The output of GIN, $\mathcal{G}_4$ .   
Figure 22: Illustrative examples of the output of each method given data generated from G. We only show the skeleton here for simplicity.

# D.4 ILLUSTRATIVE OUTPUTS OF EACH METHOD.

Here we give illustrative outputs of each method to give an intuitive understanding of why the proposed RLCD performs the best. Here we use 10000 data points and only show the skeleton for simplicity.

The underlying graph G is shown in Figure 22 (a) and it has 16 observed variables and 4 latent variables. The output of the proposed RLCD, $G_{1}$ is given in Figure 22 (b) and it is nearly the same as the input G. As it is the finite sample case, there are still two edges missing due to the error of the statistic test. However, by increasing the sample size, the missing two edges can also be recovered.

The output of Hier. rank, $G_{2}$ is given in Figure 22 (c). It introduces redundant latent variables compared to the ground truth, and many edges are incorrect. The underlying reason is that Hier. rank cannot handle the scenario where observed variables are directly adjacent and it does not allow edges from an observed variable to a latent variable.

The output of FCI, $G_{3}$ is given in Figure 22 (d). It can neither identify the cardinality nor the location of latent variables. Furthermore, many edges between observed variables are missing or incorrect. This might be due to the fact that FCI relies on CI tests conditioned on multiple variables but tests of independence conditional on large numbers of variables have very low power Spirtes (2001).

The output of GIN, $G_{4}$ is given in Figure 22 (e). The number of latent variables it discovered is correct, but they do not exactly correspond to the latent variables in the ground truth G. Plus, many edges are incorrect, and the reason is similar to that for Hier. rank, i.e., GIN cannot handle direct edges between observed variables and edges from an observed variable to a latent variable

# D.5 DETAILED DISCUSSION ABOUT COMBINING RESULT OF PHASES 2 & 3 WITH PHASE 1.

(i) “Transfer the estimated DAG $G''$ to Markov equivalence class” (the first part of Line 7 in Algorithm 1): Here, we transfer the output of Phases 2 & 3, i.e., $G''$ , to a Completed Partial Directed Acyclic Graph (CPDAG), which represents the corresponding Markov Equivalence Class.

Algorithms for transferring a DAG to CPDAG have been well studied (Chickering, 2002b;a; 2013) with implementations available from e.g., causal-learn (Zheng et al., 2023) or causaldag (Chandler Squires, 2018) python package.

(ii) “Update $G'$ by $G''$ ” (the second part of Line 7 in Algorithm 1): Note that the input to Phases 2 & 3 is $X_{Q} \cup N_{Q}$ , and let $L_{Q}$ be the newly discovered latent variables during Phases 2 & 3. By Theorem 9, latent variables must be in treks between variables in $X_{Q}$ . Therefore, we just need to consider the edges among $X_{Q} \cup L_{Q}$ . Specifically, we first delete all the edges among $X_{Q}$ in graph $G'$ , then add $L_{Q}$ to $G'$ , and finally add all the edges among $X_{Q} \cup L_{Q}$ from $G''$ to $G'$ .   
(iii) “Orient remaining causal directions that can be inferred from v structures” (Line 8 in Algorithm 1): Given $G'$ , a partially directed acyclic graph (PDAG), here we orient all remaining causal directions that can be determined. This can be easily achieved by following stage 2 of PC (Spirtes et al., 2000): we use the CI results detected in Phase 1 to find v-structures among observed variables and apply Meek’s rule (Meek, 2013) to infer the remaining directions that can be decided. After that, we then transfer this PDAG to a CPDAG (well studied in Chickering (2002b;a; 2013) with implementations available) to get the corresponding Markov equivalence class.