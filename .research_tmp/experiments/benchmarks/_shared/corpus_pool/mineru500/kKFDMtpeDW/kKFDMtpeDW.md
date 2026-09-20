# On Learning Necessary and Sufficient Causal Graphs

Hengrui Cai

University of California, Irvine
hengrc1@uci.edu

Yixin Wang

University of Michigan
yixinw@umich.edu

Michael I. Jordan

University of California, Berkeley
jordan@cs.berkeley.edu

Rui Song

North Carolina State University
songray@gmail.com

# Abstract

The causal revolution has stimulated interest in understanding complex relationships in various fields. Most of the existing methods aim to discover causal relationships among all variables within a complex large-scale graph. However, in practice, only a small subset of variables in the graph are relevant to the outcomes of interest. Consequently, causal estimation with the full causal graph—particularly given limited data—could lead to numerous falsely discovered, spurious variables that exhibit high correlation with, but exert no causal impact on, the target outcome. In this paper, we propose learning a class of necessary and sufficient causal graphs (NSCG) that exclusively comprises causally relevant variables for an outcome of interest, which we term causal features. The key idea is to employ probabilities of causation to systematically evaluate the importance of features in the causal graph, allowing us to identify a subgraph relevant to the outcome of interest. To learn NSCG from data, we develop a necessary and sufficient causal structural learning (NSCSL) algorithm, by establishing theoretical properties and relationships between probabilities of causation and natural causal effects of features. Across empirical studies of simulated and real data, we demonstrate that NSCSL outperforms existing algorithms and can reveal crucial yeast genes for target heritable traits of interest.

# 1 Introduction

Causal discovery has gained significant attention in recent years for disentangling complex causal relationships in various fields. Building upon the causal graphical model [see e.g., 23], many causal structural learning algorithms have been developed [see e.g., 35; 7; 34; 14; 4; 27; 46; 44; 48; 5] to infer the causal knowledge (e.g., causal graphs) from observed data. These algorithms are based on the assumption of causal sufficiency (the absence of unmeasured confounders). In real-world applications, to satisfy such an assumption, we strive to learn large-scale causal graphs [see e.g., 20; 6; 38; 21], in the hope of sufficiently describing how an outcome of interest depends on its relevant variables.

In addition to sufficiency, it is also crucial to account for the concept of necessity by excluding redundant variables in explaining the outcome of interest. Failure to do so can result in the inclusion of spurious variables in the learned causal graphs, which are highly correlated but have no causal impact on the outcome. These variables can impede causal estimation with limited data and lead to falsely discovered spurious relationships, leading to poor generalization performance for downstream prediction $[31]$ . For example, it might be observed that men aged 30 to 40 who buy diapers are also likely to buy beer. However, beer purchase is a spurious feature for diaper purchases: their correlation is not necessarily causal, as both purchases might be confounded by a shared cause, such as new

![](images/5f4c59c8d9f929527f12fa327bed34fa5bea69b65c6df8e174fb647debe0de7e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["New Father"] --> B["Beer"]
    C["Diaper"] --> B
```
</details>

![](images/8394c14e68df090c3ff8d8522750279b88c035fc0ec2d1a05f2d68c3db36d998.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A --> B
    B --> Y
    C --> Y
    S --> B
    N --> M
```
</details>

Figure 1: Left: Illustration of the causal relationship between the customer being a new father or not, beer purchasing, and diaper purchasing, where solid lines represent the true model, and the dashed line corresponds to the spurious correlation between beer purchasing and diaper purchasing. Right: Relationship between various causal structures. The nodes A, B, and C belong to the necessary and sufficient causal graph for the desired result Y and are represented within the solid green square. Among them, nodes B and C are members of the Markov blanket of Y, enclosed by the blue square. The node S is the spurious variable for Y, while the nodes N and M are not related to the target.

fathers buying diapers for childcare while also buying beer to alleviate stress. Therefore, simply increasing the availability of diapers or beer will not causally improve the demand for the other (see also Fig. 1(left)).

Furthermore, the number of variables causally relevant to the outcome of interest is often considerably smaller than the number of variables included in estimating a causal graph (see Fig. 1(right)). For example, while an individual's genome may encompass 4 to 5 million single nucleotide polymorphisms (SNPs), only a limited number of non-spurious genes or proteins are found to systematically regulate the expression of the phenotype of interest [e.g., 6]. Similarly, in natural language processing tasks, excluding spurious embeddings such as writing style and dialect can enhance model accuracy and downstream prediction performance [e.g., 10]. Thus, a more parsimonious causal graph is required to unveil the necessary and sufficient causal dependencies.

In this work, we focus on learning necessary and sufficient causal graphs (NSCG) that only contain causally relevant variables (which we term causal features) for an outcome of interest, offering a compact representation of causal graphs for a target outcome. Our contributions are three-fold.

- We propose the notion of NSCG (see an illustration inside the green solid square in the right panel of Fig. 1). The key idea is to leverage the marginal and conditional probabilities of causation (POC) to systematically characterize the importance of variables (a.k.a. features).   
- We establish theoretical properties and relationships between POC and the natural causal effects of features, and derive the conditions under which they are equivalent, with lower bounds provided for identification. The natural causal effects of features have explicit forms under parametric models such as the linear structural equation model, enabling convenient estimation of the POC from observed data.   
- To select necessary and sufficient features in causal graphs, we propose a necessary and sufficient causal structural learning algorithm (NSCSL) to learn an NSCG containing all necessary and sufficient causes without unnecessary spurious features. This enables feature selection for causal discovery.

The proposed method provides concise explanations of causal relationships with high-dimensional data (i.e., with a large number of variables). Empirical studies in simulated datasets show that NSCSL outperforms existing algorithms in distilling relevant subgraphs for outcomes of interest; NSCSL can also identify important quantitative trait loci for the yeast and the causal protein signaling network for single cell data, as demonstrated in real data analyses.

# 1.1 Related Works

The literature on causal structural learning can be broadly classified into three classes. The first class of methods focuses on using local conditional independence tests to identify the causal skeleton and determine the direction of the edges, such as the PC algorithm $[35; 14; 36]$ . The second class of methods uses functional causal models with additional assumptions about the data distribution, including ICA-LiNGAM $[34]$ and the causal additive model (CAM) $[4]$ . The third class, the score-based methods, includes greedy equivalence search (GES) $[7; 27; 11]$ and acyclicity optimization methods $[46]$ . Refer to $[44; 48; 17; 5; 47; 41]$ for additional cutting-edge causal structural learning methods. Yet, these works do not consider the necessity of the variables incorporated in the causal graph, i.e., whether the variables are causally relevant to the outcome. Consequently, such algorithms can often produce a redundant or potentially misleading graph, as depicted in Fig. 1 (right).

Our work also links to feature selections [see an overview in 16]. Despite the extensive literature, only a few studies have examined variable selection in causal graphs. One notable exception is Aliferis et al. [1], which uses the concept of the Markov blanket to construct a local causal graph for the target variable of interest. In this context, a Markov blanket of a variable Y is the minimal variable subset conditioned upon which all other variables become probabilistically independent of Y. Consequently, their algorithm uncovers only direct parents or children in the identified causal graph (such as the blue dotted square in the right panel of Fig. 1) and thereby overlooks the ancestors that contain atavistic information and indirectly influence the outcome. Recent works [18; 19] consider a minimal sufficient action set in bandits. Yet, these methods [also see 12; 13] rely on a true or known graph. We instead propose to simultaneously learn the causal graph and select the causal features.

Lastly, our work is connected to the body of research on probability of causation [e.g., 22; 39; 45], which delineates the necessity and sufficiency of features for the outcome of interest. Recently, Wang & Jordan [42] introduced this concept into representation learning, formulating the non-spuriousness and efficiency of representations by generalizing the probabilities of causation to accommodate low-dimensional representations of high-dimensional data. However, these works primarily concentrate on the identification of probabilities of causation, assuming that the causal graph among the variables under consideration is known with causally independent features. We address this gap in our work by incorporating the notion of probabilities of causation into learning complex causal graphs.

# 2 Framework

Graph terminology. Consider a graph $\mathcal{G} = (\boldsymbol{X}, \boldsymbol{D}_{\boldsymbol{X}})$ with a node set X and an edge set $D_{X}$ that encompasses all edges in G for nodes X. A node $X_{i}$ is said to be a parent of $X_{j}$ if there is a directed edge from $X_{i}$ to $X_{j}$ , i.e., $X_{i}$ is a direct cause of $X_{j}$ . A node $X_{k}$ is said to be an ancestor of $X_{j}$ if there is a directed path from $X_{k}$ to $X_{j}$ regulated by at least one additional node $X_{i}$ for $i \neq k$ and $i \neq j$ , i.e., $X_{k}$ is an indirect cause of $X_{j}$ . Let the set of all parents/ancestors of node $X_{j}$ in G as $\mathrm{PA}_{X_{j}}(\mathcal{G})$ . A directed graph G that does not contain directed cycles is called a directed acyclic graph (DAG). The structural causal model (SCM) characterizes the causal relationship among $|X| = d$ nodes via a DAG G and noises $e_{X} = [e_{X_{1}}, \cdots, e_{X_{d}}]^{\top}$ such that $X_{i} := h_{i}\{\mathrm{PA}_{X_{i}}(\mathcal{G}), e_{X_{i}}\}$ for some unknown $h_{i}$ and $i = 1, \cdots, d$ .

Notations and assumptions. Denote $O = (Z, Y)$ as a collection of nodes that contains features $Z = [Z_{1}, \cdots, Z_{p}]^{\top} \in Z \subset R^{p}$ and a discrete outcome of interest as $Y \in L = \{y_{1}, \cdots, y_{l}\}$ for l different values. Here, the features can be intervened, such as treatment and mediators. Let $Y(Z = z)$ be the potential value of Y that would be observed after setting variable Z as z. This is equivalent to the value of Y by imposing a ‘do-operator’ of $do(Z = z)$ as in Pearl et al. [23]. Similarly, one can define the potential outcome, $Y(Z_{i} = z_{i})$ , by setting an individual variable $Z_{i}$ as $z_{i}$ , while keeping the rest of the model unchanged. Suppose there exists an SCM that characterizes the causal relationship among O, with its DAG as $G_{O}$ . A notation and abbreviation table is provided in App. A. Following the causal inference literature [see e.g., 29; 22; 23; 42], we assume:

(A1). Consistency: $Z = z \leftrightarrow Y(Z = z) = Y, \forall z \in \mathcal{Z}$ .   
(A2). Ignorability: (i) $Y(Z = z) \perp Z, \forall z \in \mathcal{Z}$ ; (ii) $Y(Z_i = z_i) \perp Z_i | \mathrm{PA}_{Z_i \cup Y}(\mathcal{G}_O), \forall z_i \in \mathcal{Z}_i$ .

Here, (A1) implies that the outcome observed for each unit under study with features as z is identical to the outcome we would have observed had that unit been set with features Z = z. In addition, since we include as many confounders as possible, the ignorability assumption in (A2), also known as the no unmeasured confounderness assumption, is satisfied.

# 3 Necessary and Sufficient Causal Graphs

We care about a subset or a function of Z, denoted as $X = [X_{1}, \cdots, X_{d}]^{\top}$ (of d dimension with possibly $d \ll p$ ), which indeed captures the causal relationship between Z and Y. To be specific, let an SCM for causal nodes $V = (X, Y)$ with its DAG as $\mathcal{G}_{\mathbf{V}} = (\mathbf{V}, \mathbf{D}_{\mathbf{V}})$ and $e_{V}$ as a $d + 1$ dimensional independent noise, to characterize the causal relationship between X and Y. Let $P_{G}$ be the mass/density function for an SCM with its DAG G. Following the causal (or disentangled) factorization in the causal graphical model [23], we define the sufficient causal graph as follows.

Definition 3.1. (Sufficient Graph) The graph $\mathcal{G}_V$ is a sufficient causal graph to capture the causal relationship among $Z$ and $Y$ with $X \subset Z$ or $X = f(Z)$ (where $f$ is within a countable or Vapnik-Chervonenkis (VC) class) if $\mathbb{P}_{\mathcal{G}_V}\{Y|\mathrm{PA}_Y(\mathcal{G}_V)\} \prod_{X_i \in \mathrm{PA}_Y(\mathcal{G}_V)} \mathbb{P}_{\mathcal{G}_V}\{X_i|\mathrm{PA}_{X_i}(\mathcal{G}_V)\} = \mathbb{P}_{\mathcal{G}_O}\{Y|\mathrm{PA}_Y(\mathcal{G}_O)\} \prod_{Z_i \in \mathrm{PA}_Y(\mathcal{G}_O)} \mathbb{P}_{\mathcal{G}_O}\{Z_i|\mathrm{PA}_{Z_i}(\mathcal{G}_O)\}$ .

Here, Def. 3.1 refers to a sub-structure $G_{V}$ (from the whole graph $G_{O}$ ) containing all directed edges or paths towards Y, making it sufficient to describe how Y depends on all its ancestors. Then, the causal graph $G_{V}$ is said to be necessary and sufficient if it satisfies the following definition.

Definition 3.2. (Necessary and Sufficient Graph) Suppose $\mathcal{G}_V$ satisfies Def. 3.1, then $\mathcal{G}_V$ is a necessary and sufficient causal graph to capture the causal relationship among $Z$ and $Y$ if for any true subset $W$ of $X$ , i.e., $W \subset X$ or $W = g(X)$ (where $g$ is within a countable or VC class), with $U = (W, Y)$ , we have $\mathbb{P}_{\mathcal{G}_V}\{Y|\mathrm{PA}_Y(\mathcal{G}_V)\} \prod_{X_i \in \mathrm{PA}_Y(\mathcal{G}_V)} \mathbb{P}_{\mathcal{G}_V}\{X_i|\mathrm{PA}_{X_i}(\mathcal{G}_V)\} \neq \mathbb{P}_{\mathcal{G}_U}\{Y|\mathrm{PA}_Y(\mathcal{G}_U)\} \prod_{W_i \in \mathrm{PA}_Y(\mathcal{G}_U)} \mathbb{P}_{\mathcal{G}_U}\{W_i|\mathrm{PA}_{W_i}(\mathcal{G}_U)\}$ , where $\mathcal{G}_U$ is the causal graph for $U$ .

Therefore, by Def. 3.2, we can further identify the minimal sub-structure $\mathcal{G}_{\mathbf{V}}$ which includes only all directed edges or paths leading to $Y$ . The goal is to learn such a necessary and sufficient causal graph (NSCG) $\mathcal{G}_{\mathbf{V}}$ from the observed data denoted as $\{\boldsymbol{o}^{(j)} = (\boldsymbol{z}^{(j)}, y^{(j)})\}_{1 \leq j \leq n}$ with sample size $n$ , by identifying the latent causal features $\mathbf{X}$ . Denote the resulting estimated graph as $\widehat{\mathcal{G}}_{\mathbf{V}}$ .

# 4 Probability of Causation and Causal Effects

Obtaining an NSCG $G_{V}$ directly based on Def. 3.2 poses several challenges, as the latent causal features X driving the causal graph remain unknown. A naïve approach is to search all different combinations of Z for a candidate of X such that Def. 3.2 holds, which yields a complexity of $\mathcal{O}(p^{p})$ . This motivates us to assess the necessity and sufficiency of features in determining the outcome by introducing the concepts of probabilities of causation and causal effects, which will be elaborated on and interconnected in this section.

# 4.1 Probabilities of Causation and Lower Bounds

The probabilities of a feature being necessary and sufficient, known as the probability of causation (POC), have been proposed and studied [see 22; 39; 42]. Specifically, the probability of necessity and sufficiency (PNS) of feature Z is first defined in Tian & Pearl [39] as follows.

Definition 4.1. PNS in Tian & Pearl [39] with a univariate binary feature Z:

$$
P N S \equiv \mathbb {P} \{Y (Z \neq z) \neq y, Y (Z = z) = y \} \underset {b y (A 1)} {=} \mathbb {P} (Z = z, Y = y) \cdot P N + \mathbb {P} (Z \neq z, Y \neq y) \cdot P S,
$$

where the probability of necessity (PN) is $PN = \mathbb{P}\{Y(Z \neq z) \neq y | Z = z, Y = y\}$ , and the probability of sufficiency (PS) is $PS = \mathbb{P}\{Y(Z = z) = y | Z \neq z, Y \neq y\}$ .

The second equation in Def. 4.1 holds under (A1) [see details in 22; 39]. The PN score reflects the necessity of Z by evaluating the probability of the outcome becoming worse if revising the features given the good outcome observed. Similarly, the PS score indicates the sufficiency of Z by evaluating the probability of the outcome becoming better if changing the features given the bad outcome observed. Therefore, the PNS score shows the causal importance of the features by combining necessary and sufficient properties. The above definition can be generalized to multivariate cases for nonbinary features [see e.g., 42] to quantify the POC of an individual feature $Z_{i}$ . Let $Z_{-i} \equiv Z \setminus Z_{i}$ be the set of complementary variables of $Z_{i}$ . In the following, we consider two different POCs for $Z_{i}$ by extending the work of Wang & Jordan [42].

Definition 4.2. Marginal POC (M-POC) for $Z_{i}$ :

$$
\mathrm{M-POC} _ {i} (y) \equiv \mathbb {P} \left\{Y \left(Z _ {i} \neq z _ {i}\right) \neq y, Y \left(Z _ {i} = z _ {i}\right) = y \right\}.
$$

Definition 4.3. Conditional POC (C-POC) for $Z_{i}$ :

$$
\mathrm{C-POC} _ {i} (y) \equiv \mathbb {P} \left\{Y \left(Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}\right) \neq y, Y \left(Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}\right) = y \right\}.
$$

We introduce the marginal POC (M-POC) as a novel quantity in the literature to summarize the overall causal importance of an individual feature in determining the outcome's value. The conditional POC (C-POC) in Def. 4.3 corresponds to the conditional PNS in Wang & Jordan [42], which quantifies the likelihood of an individual feature being a direct and significant cause of the outcome while holding other features constant. As per Section 9.2.3 in Pearl et al. [22], the PNS in Def. 4.1 is not estimable unless additional conditions (monotonicity) are specified. To alleviate such a condition for identifying Defs. 4.2 and 4.3, we derive the lower bounds for the proposed POCs as follows.

Theorem 4.4. (Lower Bound of Probabilities of Causation) Suppose (A1) and (A2) hold. Then

$$
M \text {-} P O C _ {i} (y) \geq \mathbb {P} (Y = y | Z _ {i} = z _ {i}) - \mathbb {P} (Y = y | Z _ {i} \neq z _ {i}),
$$

$$
C \text {-} P O C _ {i} (y) \geq \mathbb {P} (Y = y | Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) - \mathbb {P} (Y = y | Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}).
$$

Table 1: Causal effects from the customer being a new father or not $(X_{F})$ and diaper purchasing $(X_{D})$ on beer purchasing $(X_{B})$ , where $\omega_{1}, \omega_{2} \in (0,1)$ with $\omega_{1} + \omega_{2} = 1$ and $c \to 1$ , in the estimated causal graph $(X_{F} \xrightarrow{\omega_{1}} X_{B})$ , and $X_{F} \xrightarrow{c} X_{D} \xrightarrow{\omega_{2}} X_{B})$ . 

<table><tr><td>Variable Name</td><td>Direct Effect</td><td>Total Effect</td></tr><tr><td>New Father ( $X_{F}$ )</td><td> $\omega_{1}$ </td><td> $\omega_{1} + c\omega_{2} \rightarrow 1 \quad (> \omega_{2})$ </td></tr><tr><td>Diaper ( $X_{D}$ )</td><td> $\omega_{2}$ </td><td> $\omega_{2}$ </td></tr></table>

The proofs of Thm. 4.4 are in App. D. The lower bound equality holds when an additional monotonicity condition is imposed, with details in App. D. The results in Thm. 4.4 allow us to estimate the lower bound of POC from observed data by learning the conditional probability of Y given various combinations of confounders. This, in turn, aids in evaluating the significance of features, with details provided in App. B. Yet, estimating these conditional probabilities of Y based on high-dimensional features is very challenging [e.g., 32; 42], which motivates us to consider the corresponding expected mean outcome given different combinations of the confounders.

# 4.2 Causal Effects and Connection to POCs

To connect the proposed POCs and facilitate the empirical estimation, we introduce the natural total effect (TE) and natural direct effect (DE) for $Z_{i}$ by extending definitions in Pearl et al. [22].

Definition 4.5. Natural Causal Effects for $Z_{i}$ :

$$
T E _ {i} = \mathbb {E} \{Y (Z _ {i} = z _ {i} + 1) \} - \mathbb {E} \{Y (Z _ {i} = z _ {i}) \},
$$

$$
D E _ {i} = \mathbb {E} \{Y (Z _ {i} = z _ {i} + 1, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i} ^ {(z _ {i})}) \} - \mathbb {E} \{Y (Z _ {i} = z _ {i}) \},
$$

where $z_{-i}^{(z_{i})}$ is the value of $Z_{-i}$ if setting $do(Z_{i}=z_{i})$ .

The natural total effect $(TE_{i})$ can be understood as the marginal change in the outcome when increasing $Z_{i}$ by one unit. Similarly, the natural direct effect $(DE_{i})$ represents the conditional change in the outcome when $Z_{i}$ is increased by one unit, with all other features held constant. Indeed, the natural total and direct causal effects delineated in Pearl et al. [22] emerge as particular instances of Def. 4.5 when examining a solitary treatment subject to intervention. By comparing Def. 4.5 with Defs. 4.2 and 4.3, it is natural to establish the relationship between POCs and causal effects below.

Theorem 4.6. (Relation between POCs and Causal Effects) Define $\delta_M(z_i) \equiv \mathbb{E}\{Y|Z_i = z_i\} - \mathbb{E}\{Y|Z_i \neq z_i\}$ and $\delta_C(z_i) \equiv \mathbb{E}\{Y|Z_i = z_i, \mathbf{Z}_{-i} = \mathbf{z}_{-i}\} - \mathbb{E}\{Y|Z_i \neq z_i, \mathbf{Z}_{-i} = \mathbf{z}_{-i}\}$ . Suppose (A1)-(A2) hold, then

$$
\sum_ {y \in \mathcal {L}} y M - P O C _ {i} (y) \geq \delta_ {M} (z _ {i}), \quad \sum_ {y \in \mathcal {L}} y C - P O C _ {i} (y) \geq \delta_ {C} (z _ {i}),
$$

if $Y$ is nonnegative. Further, if $Z_{i}$ is binary, we have

$$
\min \left\{\sum_ {y \in \mathcal {L}} y M - P O C _ {i} (y), | T E _ {i} | \right\} \geq \delta_ {M} \left(z _ {i}\right), \quad \min \left\{\sum_ {y \in \mathcal {L}} y C - P O C _ {i} (y), | D E _ {i} | \right\} \geq \delta_ {C} \left(z _ {i}\right).
$$

The proofs of Thm. 4.6 can be found in App. D, with the lower bound equality holds when an additional monotonicity condition is imposed. We can summarize the findings of Thms. 4.4 and 4.6 in two key aspects. First, the marginal and conditional POCs assess the likelihood of a feature being spurious, while the absolute values of natural causal effects quantify the size of such a spurious effect based on the magnitude of the outcome of interest. Both are lower bounded by the same quantity, that is, the differences in expectations based on the corresponding POC, given non-negative outcomes and binary features. With appropriate data processing, we can transform the outcome to be nonnegative, and thus causal effects become a suitable substitute for POCs. Second, these two approaches exhibit consistency under the monotonicity condition. The natural causal effects of features have explicit forms under parametric models (see details in § 5.1), such as the linear structural equation model, enabling convenient estimation of the necessity and sufficiency of features from observed data. This section concludes with a toy example illustrating the identification of spurious features through the proposed causal effects and the distinction between total and direct causal effects.

Example 4.7. (Recall: Beer and Diaper) Given the causal graph in Fig. 1(left), consider a linear SCM for the customer being a new father or not $(X_{F})$ , diaper purchasing $(X_{D})$ , and beer purchasing $(X_{B})$ : $X_{D} = X_{F} + e_{D}$ and $X_{B} = X_{F} + e_{B}$ , where $e_{D}$ and $e_{B}$ are independent mean zero noises.

Since the true SCM is unknown, fitting a linear model $X_B \sim \omega_1 X_F + \omega_2 X_D$ would result in ambiguous coefficients $\omega_1, \omega_2 \in [0,1]$ with $\omega_1 + \omega_2 = 1$ . By fitting $X_D \sim cX_F$ and obtaining $c$ close to 1, we have the estimated causal graph as $X_F \xrightarrow{\omega_1} X_B$ , and $X_F \xrightarrow{c} X_D \xrightarrow{\omega_2} X_B$ . Based on Def. 4.5, we estimate the corresponding causal effects in Table 1. It can be observed that the total effect of $X_F$ on $X_B$ surpasses the total effect from $X_D$ , indicating the spurious nature of the diaper purchasing, which should be removed to form the desired NSCG. Yet, using the direct effect may not be able to distinguish their differences due to the high correlation between $X_F$ and $X_D$ if $\omega_1 < \omega_2$ .

# 5 Necessary and Sufficient Causal Structural Learning

In this section, we formally present how to learn NSCG. Based on Thms. 4.4 and 4.6, a simple solution is to first use a pre-screening process to find necessary and sufficient features from Z that achieve high scores of causation, and then estimate the causal graph among the selected nodes and Y to approximate $G_{V}$ . This approach works for general SCMs while may suffer from overfitting. Instead of using such a two-step learning, we propose to learn necessary and sufficient features and the causal graph simultaneously through a single-step optimization. To this end, in § 5.1, we first introduce the structural equation model in order to provide the closed-form expressions of the proposed causal quantities. The main algorithm based on causal effects is presented in § 5.2 for the linear model, with the POC-based version available in App. B for the nonlinear model.

# 5.1 Structural Equation Model and Close Form of Causal Effects

Structural equation model. We define a selection function g that maps the feature set Z to a subset, aiming to maintain good interpretability. That is, $g : Z \in \mathbb{R}^{p} \to g(Z) \in \mathbb{R}^{d}$ where $d \ll p$ , and we denote the i-th dimension of $g(Z)$ as $g_{i}(Z)$ . Following the causal structure learning literature [35; 25; 46; 44; 48; 5], we assume the Markov and faithfulness conditions and consider a linear structural equation model (LSEM) such that $\{g(Z), Y\}$ is characterized by the pair $(B, \epsilon)$ as

$$
\left[ \begin{array}{c} g (\boldsymbol {Z}) \\ Y \end{array} \right] \leftarrow \boldsymbol {B} \left[ \begin{array}{c} g (\boldsymbol {Z}) \\ Y \end{array} \right] + \epsilon \equiv \left[ \begin{array}{c c} \boldsymbol {B} _ {g} & 0 \\ \boldsymbol {\theta} & 0 \end{array} \right] \left[ \begin{array}{c} g (\boldsymbol {Z}) \\ Y \end{array} \right] + \left[ \begin{array}{c} \boldsymbol {\epsilon} _ {\boldsymbol {Z}} \\ \epsilon_ {Y} \end{array} \right], \tag {1}
$$

where B is a $(d+1)\times(d+1)$ weighted adjacent matrix that characterizes the causal relationship among $\{g(\mathbf{Z}),Y\}$ , and $\epsilon\equiv[\epsilon_{Z}^{\top},\epsilon_{Y}]^{\top}$ is a $d+1$ dimensional random vector of jointly independent errors. Here, B consists of three components: (1). a $d\times d$ matrix $B_{g}=\{b_{i,j}\}_{1\leq i\leq d,1\leq j\leq d}$ with $b_{i,j}$ as the weight of the edge $g_{i}(\mathbf{Z}_{i})\to g_{j}(\mathbf{Z}_{i})$ if exists and $b_{i,j}=0$ otherwise; (2) a $1\times\bar{d}$ vector $\theta=[\theta_{1},\cdots,\theta_{p}]$ for $\theta_{i}$ presenting the weight of the direct edge $g_{i}(\mathbf{Z})\to Y$ ; and (3). a $(d+1)\times1$ zero vector indicating the outcome of interest Y cannot be any parent of the features. Without further assumptions, the model in (1) given a particular selector g can be identified only up to a Markov equivalence class (MEC) [35; 25]. In the following, we focus on cases where the DAG can be uniquely identifiable, such as LSEM with Gaussian noises of equal variance [35; 26; 24], and linear model with non-Gaussian noise [34; 47]. See more details and extensions to MEC in App. C.

Close form and estimation of causal effects. We next provide the close forms of the causal effects in Def. 4.5 under the model in (1). Recall that $\theta_{i}$ presents the weight of the direct edge $g(Z)_i \to Y$ . According to (1) and Def. 4.5, we have

$$
D E _ {i} (\boldsymbol {B}; g) = \theta_ {i}.
$$

The total causal effect can be quantified by the path method [see e.g., 43; 20]. Specifically, the causal effect of $g_{i}(\mathbf{Z})$ on $g_{j}(\mathbf{Z})$ along a directed path from $g_{i}(\mathbf{Z}) \to g_{j}(\mathbf{Z})$ in G can be calculated by multiplying all edge weights along the path, under LSEM. Denote the set of directed paths that starts with $g_{i}(\mathbf{Z})$ and ends with Y as $\pi_{i} = \{g_{i}(\mathbf{Z}) \to \cdots \to Y\}$ with the size as $m_{i}$ . Then the causal effect of $g_{i}(\mathbf{Z})$ on Y through the directed path $\pi_{i}^{(k)} = \{i, l_{1}, \cdots, l_{\tau_{k}}, d + 1\} \in \pi_{i}$ with length $\tau_{k} + 1$ is $PE\{\pi_{i}^{(k)}\} = b_{i,l_{1}} \cdots b_{l_{\tau_{k}}, (d+1)}$ , by the path method, where $b_{i,j}$ is the weight of the edge $g_{i}(\mathbf{Z}) \to g_{j}(\mathbf{Z})$ if it exists, and $b_{i,j} = 0$ otherwise, for $i, j \in \{1, \cdots, d\}$ , and $b_{l_{\tau_{k}}, (d+1)} = \theta_{l_{\tau_{k}}}$ as the direct edge from $g_{l_{\tau_{k}}}(\mathbf{Z})$ to Y. Thus,

$$
T E _ {i} (\boldsymbol {B}; g) = \sum_ {k = 1} ^ {m _ {i}} P E \left\{\pi_ {i} ^ {(k)} \right\}.
$$

Both $TE_{i}$ and $DE_{i}$ can be explicitly calculated given a matrix B under a selector g. We denote their estimates as $\widehat{TE}_{i}$ and $\widehat{DE}_{i}$ given the estimated matrix $\widehat{B}$ and g.

# 5.2 Learning Algorithm based on Causal Effects

The primary algorithm based on causal effects comprises three steps that quantify two sources of loss and learn the causal graph, specifically: loss of causal structural learning, loss of discovering causal features, and minimizing the overall loss to learn NSCG based on data $\{\boldsymbol{o}^{(j)} = (\boldsymbol{z}^{(j)}, y^{(j)})\}_{1 \leq j \leq n}$ .

Step 1: Form the loss from causal structural learning. To estimate the matrix B in (1), we adopt the acyclicity constraint [44; 46] as $h_{1}(\boldsymbol{B}) \equiv \operatorname{tr}\left[(I_{d+1} + t\boldsymbol{B} \circ \boldsymbol{B})^{d+1}\right] - (d + 1) = 0$ , where $I_{d+1}$ is a $d + 1$ -dimensional identity matrix, and $\operatorname{tr}(\cdot)$ is the trace of a matrix and t is a hyperparameter that depends on the estimated largest eigenvalue of B. The first loss by the augmented Lagrangian is

$$
L _ {1} (\boldsymbol {B}, g, \theta , \lambda_ {1} | \{\boldsymbol {o} ^ {(j)} \}) = f (\boldsymbol {B}, g, \theta | \{\boldsymbol {o} ^ {(j)} \}) + \lambda_ {1} h _ {1} (\boldsymbol {B}), \tag {2}
$$

where $f(B, g, \theta | \{o^{(j)}\})$ is some loss such as the least square error in NOTEARS [46] or the Kullback-Leibler divergence in DAG-GNN [44] with parameters $\theta$ , and $\lambda_{1}$ is the Lagrange multiplier. Other causal structural leaning algorithms [see e.g., 35; 7; 34; 14; 4; 27; 48] can also be applied by formulating the corresponding score or loss function.

Step 2: Constraints for causal relevance and causal identifiability. We next measure the causal relevance of the selection function g by the natural causal effects. We convert the outcome Y to be nonnegative. According to Thm. 4.6, we can avoid the estimation of POCs by using the related causal effects in Def. 4.5 with their explicit expressions. This part of loss thus becomes

$$
L _ {2} ^ {C E} (\boldsymbol {B}, g, \gamma | \{\boldsymbol {o} ^ {(j)} \}) = - \sum_ {i = 1} ^ {d} | \widehat {C E} _ {i} (\boldsymbol {B}; g) | + \sum_ {i = 1} ^ {d + 1} | b _ {i, d + 1} | + \gamma | g |. \tag {3}
$$

Here, $\widehat{CE}_i$ can either take the estimated $DE_i$ or the estimated $TE_i$ given the matrix $\pmb{B}$ and $g$ , as detailed in § 5.1. The second term corresponds to the causal identification constraint on the last column of $\pmb{B}$ , requiring all elements to be zeros as in (1). This constraint restricts the causal structural learning to a smaller class of DAGs. Finally, $|g|$ denotes the number of selected nodes in $g$ , with a penalty $\gamma$ to control the complexity of the selector.

Step 3: Necessary and sufficient causal structural learning. Combining two sources of loss functions in (2) with (3), leads to the objective as

$$
\min _ {\boldsymbol {B}, g} \left[ L _ {1} (\boldsymbol {B}, g, \theta , \lambda_ {1} | \{\boldsymbol {o} ^ {(j)} \}) + \alpha L _ {2} ^ {C E} (\boldsymbol {B}, g, \gamma | \{\boldsymbol {o} ^ {(j)} \}) \right], \tag {4}
$$

where $\alpha$ can be reviewed as a trade-off parameter between two loss functions. Next, we provide a solution for (4) without tuning $\alpha$ . Specifically, based on no unmeasured confounders in (A2), we can calculate the highest causal effects that could be achieved given all variables without any penalty. Denote the estimated highest absolute causal effects in data as $\delta^{*}$ . The goal is to find a subset of Z such that $g(\mathbf{Z})$ achieves a similar level of necessity and sufficiency, i.e., the resulting score is close to $\delta^{*}$ . Hence, we set the second loss as a constraint by comparing the overall causal effects of the selected nodes with the highest reference over the entire observed feature space, i.e.,

$$
h _ {2} (\boldsymbol {B}; g) = \delta^ {*} - \sum_ {i = 1} ^ {d} | \widehat {C E} _ {i} (\boldsymbol {B}; g) | + \sum_ {i = 1} ^ {d + 1} | b _ {i, d + 1} |,
$$

should be approaching 0 given a good selector g. This yields a new objective function as

$$
\min _ {\boldsymbol {B}, g} f (\boldsymbol {B}, g, \theta | \{\boldsymbol {o} ^ {(j)} \}) + \lambda_ {1} h _ {1} (\boldsymbol {B}) + \lambda_ {2} h _ {2} (\boldsymbol {B}; g) + c | h _ {1} (\boldsymbol {B}) | ^ {2} + d | h _ {2} (\boldsymbol {B}; g) | ^ {2} + \gamma | g |, \tag {5}
$$

where $\lambda_{2}$ is the Lagrange multiplier for the new constraint, and c and d are penalty terms. To minimize the loss in (5) and satisfy both $h_{1}(B) \to 0$ and $h_{2}(B; g) \to 0$ , we simultaneously update $\lambda_{1}$ and $\lambda_{2}$ and increase c and d to infinity, by modifying the updating technique [see e.g., 46; 44] for multiple constraints, with the class of functions g specified as the subset of Z and its penalty $\gamma$ as the size of selected nodes in g. Here, the minimization can be solved using a black-box stochastic optimization such as ‘Adam’ [15]. Denote the estimated matrix as $\widehat{B}$ , based on which we can obtain the estimated causal graph as $\widehat{G}_{V}$ consisting of nodes $\widehat{g}(Z)$ . Finally, we name this proposed algorithm as necessary and sufficient causal structural learning (NSCSL). The computational complexity of NSCSL is provided in App. B.3. We next establish the consistency of estimated causal graphs below.

Theorem 5.1. Assume Model (1) holds with independent Gaussian error and equal variance. Suppose the topological ordering of the true bounded matrix B is consistently estimated. Then the estimated matrix $\widehat{B}$ minimizing the loss in (5) converges to B with the probability going to 1 as $n \to \infty$ .

Table 2: Comparison results across S1 to S3 under different sample sizes (n). Methods are evaluated by FDR, TPR, and SHD, with the standard error (SE) reported for each metric, over 50 replications. 

<table><tr><td>Scenario</td><td>Method</td><td>FDR±SE $n_1$  (small)</td><td> $n_2$  (large)</td><td>TPR±SE $n_1$  (small)</td><td> $n_2$  (large)</td><td>SHD±SE $n_1$  (small)</td><td> $n_2$  (large)</td></tr><tr><td>S1</td><td>NSCSL-TE</td><td>0.09±0.03</td><td>0.02±0.01</td><td>0.95±0.02</td><td>1.00±0.00</td><td>0.64±0.20</td><td>0.14±0.06</td></tr><tr><td>p=5</td><td>NSCSL-DE</td><td>0.08±0.03</td><td>0.02±0.01</td><td>0.95±0.03</td><td>1.00±0.00</td><td>0.72±0.20</td><td>0.14±0.06</td></tr><tr><td> $n_1$ =30</td><td>NOTEARS</td><td>0.39±0.01</td><td>0.34±0.00</td><td>0.96±0.02</td><td>1.00±0.00</td><td>2.56±0.14</td><td>2.02±0.01</td></tr><tr><td> $n_2$ =100</td><td>PC</td><td>0.53±0.02</td><td>0.53±0.01</td><td>0.47±0.05</td><td>0.41±0.01</td><td>3.12±0.11</td><td>3.20±0.07</td></tr><tr><td>ER Model</td><td>LiNGAM</td><td>0.31±0.02</td><td>0.33±0.00</td><td>0.78±0.02</td><td>0.98±0.01</td><td>2.30±0.10</td><td>2.00±0.00</td></tr><tr><td>S2</td><td>NSCSL-TE</td><td>0.10±0.03</td><td>0.02±0.01</td><td>1.00±0.00</td><td>0.99±0.01</td><td>0.38±0.14</td><td>0.12±0.06</td></tr><tr><td>p=5</td><td>NSCSL-DE</td><td>0.12±0.04</td><td>0.01±0.01</td><td>0.67±0.04</td><td>0.50±0.00</td><td>1.00±0.12</td><td>1.02±0.01</td></tr><tr><td> $n_1$ =30</td><td>NOTEARS</td><td>0.63±0.01</td><td>0.60±0.00</td><td>1.00±0.00</td><td>1.00±0.00</td><td>3.46±0.13</td><td>3.02±0.01</td></tr><tr><td> $n_2$ =100</td><td>PC</td><td>0.73±0.02</td><td>0.79±0.00</td><td>0.50±0.00</td><td>0.50±0.00</td><td>3.62±0.15</td><td>3.88±0.03</td></tr><tr><td>ER Model</td><td>LiNGAM</td><td>0.62±0.01</td><td>0.60±0.00</td><td>0.98±0.02</td><td>1.00±0.00</td><td>3.18±0.09</td><td>3.00±0.00</td></tr><tr><td>S3</td><td>NSCSL-TE</td><td>0.10±0.02</td><td>0.01±0.00</td><td>0.94±0.02</td><td>1.00±0.00</td><td>0.84±0.16</td><td>0.06±0.02</td></tr><tr><td>p=5</td><td>NSCSL-DE</td><td>0.08±0.02</td><td>0.01±0.00</td><td>0.84±0.03</td><td>0.81±0.00</td><td>1.30±0.14</td><td>1.00±0.00</td></tr><tr><td> $n_1$ =30</td><td>NOTEARS</td><td>0.11±0.02</td><td>0.01±0.00</td><td>0.96±0.02</td><td>1.00±0.00</td><td>0.76±0.16</td><td>0.06±0.02</td></tr><tr><td> $n_2$ =100</td><td>PC</td><td>0.00±0.00</td><td>0.00±0.00</td><td>0.66±0.02</td><td>0.80±0.00</td><td>1.68±0.12</td><td>1.00±0.00</td></tr><tr><td>ER Model</td><td>LiNGAM</td><td>0.03±0.01</td><td>0.00±0.00</td><td>0.98±0.01</td><td>1.00±0.00</td><td>0.24±0.09</td><td>0.02±0.01</td></tr></table>

The proof and detailed conditions for Thm. 5.1 are provided in App. D, which align with those commonly imposed in causal structural learning [e.g., 33]. Our proof follows similar strategies but accounts for the extra penalty term from causal effects. Notice that the explicit forms of causal effects under LSEM are linear combinations of elements of B. This implies our new regulation can similarly vanish away as n goes to infinity.

# 6 Experiments

Experiment design. Scenarios are generated as follows. We consider the dimension of variables in the graph as p = 5 in Scenarios 1 to 3 (S1 to S3), p = 20 in Scenario 4 (S4), and p = 50 in Scenario 5 (S5), to examine the scalability of NSCSL. For each scenario, the DAG that characterizes the causal relationship among variables $O = (Z, Y)$ is generated from the Erdős-Reñyi (ER) model with an expected degree as 2 for S1 to S3 and degree as 5 for S4 to S5. We also generate the graph from the scale-free (SF) model for S5 with the degree of 5 to examine the robustness of the proposed method under diverse synthetic graphs. Each edge is assigned positive weights. We set the last variable as the outcome of interest Y, and generate the data based on LSEM by $Z = B^{\top}Z + \epsilon$ , where $\epsilon$ is a random vector of jointly independent binary variables with equal probability taking value one or zero. Thus, the outcome is nonnegative and discrete. In addition, we also include a nonlinear structural equation model for S4 where $O_{i} := \psi_{i}\{\mathrm{PA}_{O_{i}}(\mathcal{G})\} + e_{O_{i}}$ and $\psi_{i}(x) = \lfloor 2 \log(x + 1) \rfloor$ where $\lfloor x \rfloor$ rounds to nearest integer for x. In S1, the true causal graph contains one spurious node (indexed by 0) and three non-spurious nodes (indexed by 1, 2, and 3), as shown in sub-figures (a) of Fig. E.7, with the associated true NSCG in sub-figures (b) of Fig. E.7. Moreover, we design a balanced setting with half spurious variables and half non-spurious variables in S2, as depicted in Fig. E.8, and a baseline setting without any spurious variables in S3, shown in Fig. E.9. Finally, S4 contains 2 non-spurious variables with 17 spurious variables in Fig. E.10. The experiments are conducted on a Google Cloud Platform virtual machine with 8 processor cores and 32GB memory.

Methods and benchmark specification. We apply the proposed NSCSL based on TE and DE as the criteria of necessity and sufficiency, respectively, to capture the marginal and conditional causal effect of the confounders. Note that we consider fully identifiable models in the experiments so that it is meaningful to evaluate causal effects from the estimated graph. The underlying causal structure learning algorithm is set to NOTEARS [46]. We also compare the proposed method against other methods, including PC [35] and LiNGAM [34] for S1 to S5; DAG-GNN [44], GES with generalized score [GSGES, 11], fast causal inference [FCI, 36], and the causal additive model [CAM, 4], for all high-dimensional settings in S4 and S5. Here, we use a graph threshold of 0.3 (commonly used in the literature [46; 44; 48; 5]) to prune the noise edges for a fair comparison. The training details are provided in Table E.1. The true and estimated graphs with the associated matrix under different approaches are illustrated in Figs. E.2 to E.6 and Figs. E.7 to E.11 in App. E for S1 to S4. The comparison results across different sample sizes (n) are presented in Table 2 for S1 to S3, in Table 3 for S4 to S5 with linear ER model, in Table 4 for S4 with nonlinear ER model, and in Table 5 for S5 with linear SF model. All the results are evaluated by false discovery rate (FDR), true positive

Table 3: Comparison studies under S4 to S5 under different sample sizes (n) and dimensions (p) with the Erdős-Reńyi (ER) model. Methods are evaluated by FDR, TPR, SHD, and runtime (seconds), with standard errors (SE) reported for each metric, over 50 replications. 

<table><tr><td>Scenario</td><td>Method</td><td>FDR±SE $n_1$  (small)</td><td> $n_2$  (large)</td><td>TPR±SE $n_1$  (small)</td><td> $n_2$  (large)</td><td>SHD±SE $n_1$  (small)</td><td> $n_2$  (large)</td><td>Time±SE $n_1$  (small)</td><td> $n_2$  (large)</td></tr><tr><td>S4</td><td>NSCSL-TE</td><td>0.11±0.02</td><td>0.00±0.01</td><td>1.00±0.00</td><td>1.00±0.00</td><td>0.86±0.24</td><td>0.04±0.01</td><td>10.8±0.3</td><td>56.2±1.1</td></tr><tr><td>p=20</td><td>NSCSL-DE</td><td>0.11±0.02</td><td>0.00±0.01</td><td>0.69±0.01</td><td>0.67±0.00</td><td>1.34±0.08</td><td>1.00±0.01</td><td>12.4±0.8</td><td>54.5±1.3</td></tr><tr><td> $n_1$ =100</td><td>NOTEARS</td><td>0.93±0.00</td><td>0.92±0.00</td><td>1.00±0.00</td><td>1.00±0.00</td><td>40.80±0.20</td><td>36.40±0.04</td><td>22.9±6.4</td><td>69.8±8.7</td></tr><tr><td> $n_2$ =1000</td><td>PC</td><td>0.92±0.00</td><td>0.95±0.00</td><td>0.51±0.02</td><td>0.67±0.00</td><td>19.08±0.17</td><td>33.34±0.03</td><td>6.9±0.5</td><td>16.3±0.8</td></tr><tr><td>ER Model</td><td>LiNGAM</td><td>0.92±0.00</td><td>0.93±0.00</td><td>0.99±0.01</td><td>1.00±0.00</td><td>33.00±0.20</td><td>37.10±0.02</td><td>8.1±0.5</td><td>24.6±0.6</td></tr><tr><td rowspan="4">Degree=5</td><td>DAGGNN</td><td>0.93±0.00</td><td>0.93±0.00</td><td>0.97±0.01</td><td>0.97±0.00</td><td>41.34±0.19</td><td>40.10±0.06</td><td>28.32±4.3</td><td>39.12±7.2</td></tr><tr><td>GSGES</td><td>0.98±0.01</td><td>0.98±0.00</td><td>0.22±0.02</td><td>0.20±0.01</td><td>43.20±0.26</td><td>45.70±0.34</td><td>14.92±7.5</td><td>26.32±9.1</td></tr><tr><td>FCI</td><td>0.98±0.01</td><td>0.99±0.00</td><td>0.10±0.02</td><td>0.06±0.01</td><td>22.70±0.17</td><td>32.90±0.02</td><td>6.6±0.3</td><td>12.7±0.2</td></tr><tr><td>CAM</td><td>0.93±0.00</td><td>0.94±0.00</td><td>0.63±0.02</td><td>1.00±0.00</td><td>29.80±0.38</td><td>40.30±0.07</td><td>19.62±18.3</td><td>25.62±23.2</td></tr><tr><td>S5</td><td>NSCSL-TE</td><td>0.03±0.01</td><td>0.02±0.01</td><td>0.86±0.03</td><td>0.93±0.01</td><td>2.18±0.13</td><td>1.58±0.07</td><td>110.1±3.9</td><td>21.12±12.1</td></tr><tr><td>p=50</td><td>NSCSL-DE</td><td>0.02±0.02</td><td>0.01±0.01</td><td>0.29±0.02</td><td>0.28±0.01</td><td>10.08±0.21</td><td>9.76±0.13</td><td>119.0±5.5</td><td>23.52±10.9</td></tr><tr><td> $n_1$ =1000</td><td>NOTEARS</td><td>0.86±0.04</td><td>0.85±0.01</td><td>0.93±0.03</td><td>0.92±0.01</td><td>79.20±1.40</td><td>77.12±0.53</td><td>128.3±8.2</td><td>26.92±15.1</td></tr><tr><td> $n_2$ =3000</td><td>PC</td><td>0.96±0.03</td><td>0.97±0.02</td><td>0.07±0.02</td><td>0.06±0.01</td><td>82.12±1.21</td><td>88.28±1.63</td><td>20.9±0.5</td><td>35.9±3.2</td></tr><tr><td>ER Model</td><td>LiNGAM</td><td>0.86±0.02</td><td>0.86±0.01</td><td>0.97±0.02</td><td>0.99±0.01</td><td>86.12±1.20</td><td>85.70±0.84</td><td>43.1±6.3</td><td>145.3±7.9</td></tr><tr><td rowspan="4">Degree=5</td><td>DAGGNN</td><td>0.87±0.02</td><td>0.88±0.01</td><td>0.93±0.02</td><td>0.94±0.01</td><td>87.50±1.10</td><td>85.62±0.96</td><td>49.32±7.5</td><td>81.12±87.6</td></tr><tr><td>GSGES</td><td>0.89±0.03</td><td>0.93±0.01</td><td>0.19±0.03</td><td>0.12±0.01</td><td>93.54±1.46</td><td>95.70±0.79</td><td>31.22±10.1</td><td>45.32±18.1</td></tr><tr><td>FCI</td><td>0.96±0.02</td><td>0.97±0.01</td><td>0.08±0.01</td><td>0.07±0.01</td><td>84.00±0.80</td><td>88.50±0.60</td><td>14.3±0.8</td><td>17.7±0.5</td></tr><tr><td>CAM</td><td>0.93±0.04</td><td>0.95±0.02</td><td>0.66±0.03</td><td>0.67±0.02</td><td>126.00±3.64</td><td>127.80±2.06</td><td>28.42±31.7</td><td>75.62±63.9</td></tr></table>

rate (TPR), and the structural Hamming distance (SHD) to the true causal graph, with their standard errors, over 50 replications. The average running time of these methods is also reported in Table 2 to Table 5 to reflect the computational complexities. In addition, the sensitivity analyses concerning all hyperparameters in Table E.1 are conducted using S4, as presented in Fig. 2.

Results and conclusion. NSCSL performs the best in discovering NSCG in S1, S2, S4, and S5, and shows comparably best results in S3 (a setting without any spurious variables). To be specific, the benchmark methods for causal structural learning aim to reveal causal relationships in the whole graph (i.e., sub-figures (a) in Figs. E.2 to E.6), which contains spurious effects on the target outcome Y. The proposed algorithm overcomes this drawback by purely identifying the true important causal relationships (i.e., sub-figures (b) in Figs. E.2 to E.6). By comparing the sub-figures (c) and (d) in Figs. E.2 to E.6 as well as Table 2 to Table 5, NSCSL-TE detects all necessary and sufficient causal paths towards the outcome, while NSCSL-DE only extracts direct causal relationships between the features and the outcome, resulting in a slightly lower TPR and slightly higher SHD than NSCSL-TE. Furthermore, from Table 2 to Table 5, the results under the proposed algorithm approach the truth more closely as the sample size increases in all scenarios, regardless of the underlying graph models and data-generating process. In contrast, the benchmark methods all fail to discover NSCG in the high-dimensional setting, exhibiting extremely high FDRs and SHDs. In addition, Table 3 to Table 5 reveal that NSCSL is as fast as the quickest benchmarks such as PC, LiNGAM, and FCI, and significantly faster than others like DAGGNN, GSGES, and CAM. Our method's integration of treatment effects into the optimization adds efficiency and restricts the searching space, making it practical and even beating NOTEARS in computation. The sensitivity analyses in Fig. 2 indicate that our method remains robust to these parameters set within a reasonable range.

Real data analysis - Sachs et al. [30]. We conduct real data analysis using the benchmark data from Sachs et al. [30]. To validate our method's capacity to find the NSCG and align with Def. 3.2, we designated the protein Akt as the target outcome. This designation ensures that NSCG exists (see Fig. 3) and that finding an NSCG is meaningful. Our method (NSCSL-TE) and seven baseline methods were applied and evaluated against the true NSCG associated with the protein Akt. Table 6 shows that our method achieves the best performance in finding the NSCG concerning the protein Akt.

Real data analysis - Brem & Kruglyak [2]. Furthermore, we apply NSCSL to gene expression traits in yeast [2] using a dataset of 104 yeast segregants with diverse genotypes. The goal is to identify genes, known as quantitative trait loci (QTLs), influencing the expression level (nonnegative) of the genetic variant YER124C, a daughter cell-specific protein involved in cell wall metabolism. Following a similar approach as in Chakrabortty et al. [6], we include 492 QTLs by filtering out genes with missing or low variability in expression levels. The total sample size is 262. Given the high-dimensional QTLs, constructing an NSCG with only causally relevant variables for the outcome of interest is essential. We apply NSCSL-TE and compare it with all baseline methods. The summarized results in Table 7 highlight our method's ability to identify relevant genetic influencers

Table 4: Comparison studies under the nonlinear structural equation model for S4 over 50 replications. 

<table><tr><td>Scenario</td><td>Method</td><td>FDR±SE</td><td>TPR±SE</td><td>SHD±SE</td><td>Time±SE</td></tr><tr><td>S4</td><td>NSCSL-TE</td><td>0.03±0.01</td><td>0.83±0.01</td><td>0.60±0.02</td><td>55.8±0.3</td></tr><tr><td>p=20</td><td>NSCSL-DE</td><td>0.03±0.01</td><td>0.50±0.01</td><td>1.60±0.02</td><td>56.8±0.2</td></tr><tr><td>n=1000</td><td>NOTEARS</td><td>0.91±0.01</td><td>0.83±0.01</td><td>35.90±0.04</td><td>56.5±0.7</td></tr><tr><td>ER Model</td><td>PC</td><td>0.99±0.01</td><td>0.12±0.01</td><td>44.12±0.03</td><td>15.7±0.2</td></tr><tr><td rowspan="5">Degree=5</td><td>LiNGAM</td><td>0.93±0.01</td><td>0.70±0.00</td><td>37.30±0.03</td><td>11.6±0.1</td></tr><tr><td>DAGGNN</td><td>0.94±0.01</td><td>0.86±0.01</td><td>34.80±0.06</td><td>41.3 $^{2}$ ±9.8</td></tr><tr><td>GSGES</td><td>0.98±0.01</td><td>0.23±0.01</td><td>52.40±0.38</td><td>22.1 $^{2}$ ±7.5</td></tr><tr><td>FCI</td><td>0.97±0.01</td><td>0.13±0.01</td><td>33.80±0.06</td><td>12.6±0.3</td></tr><tr><td>CAM</td><td>0.95±0.01</td><td>1.00±0.01</td><td>31.60±0.08</td><td>27.9 $^{2}$ ±33.7</td></tr></table>

Table 5: Comparison studies under the scale-free (SF) model for S5 over 50 replications. 

<table><tr><td>Scenario</td><td>Method</td><td>FDR±SE</td><td>TPR±SE</td><td>SHD±SE</td><td>Time±SE</td></tr><tr><td>S5</td><td>NSCSL-TE</td><td>0.02±0.02</td><td>0.78±0.03</td><td>5.08±0.11</td><td>135.7±5.6</td></tr><tr><td>p=50</td><td>NSCSL-DE</td><td>0.02±0.02</td><td>0.51±0.02</td><td>17.20±0.35</td><td>147.0±6.3</td></tr><tr><td>n=1000</td><td>NOTEARS</td><td>0.88±0.04</td><td>0.75±0.03</td><td>123.10±1.50</td><td>160.5±8.1</td></tr><tr><td>SF Model</td><td>PC</td><td>0.97±0.03</td><td>0.06±0.02</td><td>79.34±1.17</td><td>32.1±0.8</td></tr><tr><td rowspan="5">Degree=5</td><td>LiNGAM</td><td>0.91±0.02</td><td>0.98±0.01</td><td>212.00±5.12</td><td>117.7±6.3</td></tr><tr><td>DAGGNN</td><td>0.92±0.02</td><td>0.85±0.02</td><td>203.50±7.80</td><td>57.32±13.6</td></tr><tr><td>GSGES</td><td>0.96±0.03</td><td>0.10±0.03</td><td>98.34±2.15</td><td>35.92±13.5</td></tr><tr><td>FCI</td><td>0.97±0.02</td><td>0.07±0.01</td><td>81.70±1.40</td><td>15.7±1.1</td></tr><tr><td>CAM</td><td>0.98±0.04</td><td>0.24±0.03</td><td>218.00±8.15</td><td>37.12±53.6</td></tr></table>

Table 6: Real data results for the single-cell data by Sachs et al. [30], evaluated by total edges, correct edges, and SHD, based on the true NSCG with respect to the protein Akt. 

<table><tr><td>Method</td><td>NSCSL</td><td>NOTEARS</td><td>PC</td><td>LiNGAM</td><td>DAGGNN</td><td>GSGES</td><td>FCI</td><td>CAM</td></tr><tr><td>Total Edges</td><td>8</td><td>20</td><td>25</td><td>6</td><td>33</td><td>28</td><td>24</td><td>7</td></tr><tr><td>Correct Edges</td><td>4</td><td>2</td><td>2</td><td>1</td><td>4</td><td>2</td><td>2</td><td>1</td></tr><tr><td>SHD</td><td>8</td><td>21</td><td>28</td><td>11</td><td>30</td><td>32</td><td>27</td><td>10</td></tr></table>

![](images/084b302e5b6ebc5f671db48ee1182482aeb33c330ecafce27c01bf7fba938900.jpg)

Figure 2: Sensitivity analyses.   
![](images/387cceeb0e6295bc2b209c1816ded721d482964c6975d47380ad99d97379a295.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    Praf --> plog
    plog --> PIP2
    PAP --> pjnk
    pjnk --> pakts473
    pakts473 --> PKC
    PKC --> PKA
    PKA --> p44/42
    p44/42 --> pinek
    pinek --> plog
    plog --> praf
    style pakts473 fill:#000,stroke:#000,color:#fff
    style PKC fill:#000,stroke:#000,color:#fff
    style p9k fill:#000,stroke:#000,color:#fff
    style p1nk fill:#000,stroke:#000,color:#fff
    style p44/42 fill:#000,stroke:#000,color:#fff
    style PKC fill:#000,stroke:#000,color:#fff
    style p3 fill:#000,stroke:#000,color:#fff
    style PKA fill:#000,stroke:#000,color:#fff
    style p1nk fill:#000,stroke:#000,color:#fff
    style p44/42 fill:#000,stroke:#000,color:#fff
    style PKC fill:#000,stroke:#000,color:#fff
    style p3 fill:#000,stroke:#000,color:#fff
    style p9k fill:#000,stroke:#000,color:#fff
    style p1nk fill:#000,stroke:#000,color:#fff
    style p44/42 fill:#000,stroke:#000,color:#fff
    style PKC fill:#000,stroke:#000,color:#fff
    style P38 fill:#000,stroke:#000,color:#fff
```
</details>

Figure 3: The causal signaling network in Sachs et al. [30], where the blue-colored sub-graph is the true NSCG for protein Akt (purple).

Table 7: Real data results for the yeast gene data, evaluated by total edges, the identified parents/ancestors of the variant YER124C, and the edges towards YER124C. 

<table><tr><td>Method</td><td>NSCSL</td><td>NOTEARS</td><td>PC</td><td>LiNGAM</td><td>DAGGNN</td><td>GSGES</td><td>FCI</td><td>CAM</td></tr><tr><td>Total Edges</td><td>11</td><td>25</td><td>22</td><td>15</td><td>35</td><td>27</td><td>22</td><td>33</td></tr><tr><td># Parents/Ancestors of YER124C</td><td>8</td><td>8</td><td>6</td><td>4</td><td>8</td><td>7</td><td>5</td><td>7</td></tr><tr><td># Edges towards YER124C</td><td>11</td><td>9</td><td>8</td><td>6</td><td>10</td><td>9</td><td>8</td><td>10</td></tr></table>

for the variant YER124C without contamination by irrelevant genes. The estimated causal graph and causal effects as well as more detailed analyses are provided in App. E.2. These observations align with findings from our simulation studies, further supporting NSCSL's superiority in revealing important causal features.

# 7 Limitations and Future Research

In this work, we introduced NSCSL that leverages causal effects/POCs to systematically assess feature importance while learning a causal graph. By identifying a subgraph closely related to the outcome, our method filters irrelevant variables, presenting a significant advancement in the field. Extensive empirical evaluations on simulated and real-world data underscore NSCSL's superior performance over existing algorithms, including important findings on yeast genes and the protein signaling network. However, this promising advancement is not without limitations. First, NSCSL, like most existing causal structural learning methods, assumes no unmeasured confounders (A2) and the causal Markov condition. These assumptions may not hold in practice, leading to biased causal effect estimates and potential errors in the causal graph. Second, NSCSL employs absolute causal effects as a substitute for POCs to facilitate estimation in high-dimensional settings. Although theoretically consistent under certain conditions, examining the differences between these two methods in general feature and outcome spaces is an area for future research.

# Acknowledgments and Disclosure of Funding

Research reported in this publication was supported in part by the Amazon Web Services (AWS) Cloud Research program, the Office of Naval Research under grant N00014-23-1-2590, and the National Science Foundation under grant No. 2231174 and No. 2310831. The authors thank the area chair and the reviewers for their constructive comments which have led to a significant improvement of the earlier version of this article.

# Appendices

# A Notations and Abbreviations 13

# B NSCSL for Nonlinear Models: The POC-based Learning Algorithm 13

B.1 Nonlinear Structural Equation Model and Estimation of POC ..... 13

B.1.1 Nonlinear Structural Equation Model 13   
B.1.2 Estimation of POC 14

B.2 Learning Algorithm based on POCs 14

B.2.1 Step 1: Nonlinear Causal Structural Learning ..... 14   
B.2.2 Step 2: Causal Relevance Measurement Using POCs ..... 14   
B.2.3 Step 3: Necessary and Sufficient Causal Structural Learning ..... 15

B.3 The Computational Complexity of the NSCSL Algorithm ..... 15   
B.4 Discussion on Scale Invariance 15

# C Extension to Markov Equivalence Class 15

C.1 Additional Graph Terminology 15   
C.2 Model Identifiabilities ..... 15   
C.3 Extended Algorithm for Markov Equivalence Class 16

# D Technical Proofs 16

D.1 Proofs of Thm. 4.4 17

D.1.1 Part 1: Lower Bound for M-POC 17   
D.1.2 Part 2: Lower Bound for C-POC ..... 18   
D.1.3 Part 3: Conditions to Achieve Lower Bounds ..... 19

D.2 Proofs of Thm. 4.6 19   
D.3 Proofs of Thm. 5.1 20

# E Additional Simulation Results 23

E.1 Simulation Configurations 23   
E.2 More Real Data Analyses on Yeast Data 23   
E.3 Additional Simulation Results: True and Estimated Matrix 25   
E.4 Additional Simulation Results: True and Estimated Graphs ..... 30

# A Notations and Abbreviations

Table A.1: The table of notations and abbreviation. 

<table><tr><td>Concept</td><td>Description</td></tr><tr><td>Graph  $\mathcal{G}$ </td><td>A graph with a node set  $\boldsymbol{X}$  and edge set  $\boldsymbol{D}_{\boldsymbol{X}}$ </td></tr><tr><td>Node  $X_{i}$ </td><td>A member of the node set  $\boldsymbol{X}$ </td></tr><tr><td>Parent of  $X_{j}$ </td><td>A node  $X_{i}$  such that there is a directed edge from  $X_{i}$  to  $X_{j}$  (i.e.,  $X_{i}$  is a direct cause of  $X_{j}$ )</td></tr><tr><td>Ancestor of  $X_{j}$ </td><td>A node  $X_{k}$  such that there&#x27;s a directed path from  $X_{k}$  to  $X_{j}$  regulated by at least one additional node  $X_{i}$  for  $i \neq k$  and  $i \neq j$  (i.e.,  $X_{k}$  is an indirect cause of  $X_{j}$ )</td></tr><tr><td> $\text{PA}_{X_{j}}(\mathcal{G})$ </td><td>The set of all parents/ancestors of node  $X_{j}$  in  $\mathcal{G}$ </td></tr><tr><td>DAG</td><td>A directed acyclic graph; a directed graph  $\mathcal{G}$  that does not contain directed cycles</td></tr><tr><td>SCM</td><td>The structural causal model characterizes the causal relationship among  $|\boldsymbol{X}| = d$  nodes via a DAG  $\mathcal{G}$  and noises  $e_{\boldsymbol{X}} = [e_{X_{1}}, \cdots, e_{X_{d}}]^{\top}$  such that  $X_{i} := h_{i}\{\text{PA}_{X_{i}}(\mathcal{G}), e_{X_{i}}\}$  for some unknown  $h_{i}$  and  $i = 1, \cdots, d$ </td></tr><tr><td> $Y(Z_{i} = z_{i})$ </td><td>Potential outcome after setting individual variable  $Z_{i}$  to  $z_{i}$ </td></tr><tr><td> $\mathcal{G}_{O}$ </td><td>DAG characterizing the causal relationship among  $O$ </td></tr><tr><td> $\mathbb{P}_{\mathcal{G}}$ </td><td>The mass/density function for an SCM with its DAG  $\mathcal{G}$ </td></tr><tr><td> $\boldsymbol{Z}_{-i} \equiv \boldsymbol{Z} \setminus Z_{i}$ </td><td>The complementary variable set of  $Z_{i}$ </td></tr></table>

# B NSCSL for Nonlinear Models: The POC-based Learning Algorithm

In this section, we outline a method for learning the NSCG for a nonlinear model. Given the lack of an explicit form for both causal effects and POCs in such models, we employ an iterative learning method based on Thm. 4.4 to capture complex nonlinear relationships among variables. The method begins with a pre-screening process, identifying necessary and sufficient features from Z with high causation scores. Following this, the causal graph is estimated among the selected nodes and Y, approximating $G_{V}$ in the nonlinear structural equation model. This iterative approach is applicable for general SCMs and the process is repeated until convergence is achieved. App. B.1 introduces the nonlinear structural equation model and describes the estimation of POC in such a model. The main algorithm based on POCs is presented in App. B.2.

# B.1 Nonlinear Structural Equation Model and Estimation of POC

# B.1.1 Nonlinear Structural Equation Model

While the linear structural equation model has good properties such as easy implementation and nice interpretation, it cannot capture complex nonlinear causal relationships. To address this, we consider the non-linear additive form following [25; 47; 28]. Specifically, for variables $D = \{g(Z), Y\}$ as a $d + 1$ -dimensional vector, we consider replacing the model in (1) as follows,

$$
D _ {i} := \psi_ {i} \left\{\mathrm{PA} _ {D _ {i}} (\mathcal {G}) \right\} + e _ {D _ {i}}, \tag {B.1}
$$

for the i-th element/node $D_{i}$ in D with some unknown nonlinear function $\psi_{i}$ and independent noise $e_{D_{i}}$ , for $i = 1, \cdots, d + 1$ . As mentioned in [47; 28], this model is identifiable from observational data. We further define a new functional matrix $B(\psi) = B(\psi_{1}, \cdots, \psi_{d+1})$ that encodes the unknown causal relationship among variables. The element in the i-th row and j-th column is defined as:

$$
[ \boldsymbol {B} (\psi) ] _ {i j} := | | \partial_ {j} \psi_ {i} | | _ {2}, \tag {B.2}
$$

where $\|\cdot\|_{2}$ presents the $L_{2}$ norm. This matrix thus describes the dependency among D; if $D_{i}$ does not depend on $D_{j}$ , we have $\|\partial_{j}\psi_{i}\|_{2}=0$ . We can also incorporate background knowledge as done in § 5.2 to reflect the causal roles in different types of variables. We do this by specifying the last column of $\boldsymbol{B}(\psi)$ to be all zeros, so that $h_{c}\boldsymbol{B}(\psi)=\sum_{i=1}^{d+1}\left|\left[\boldsymbol{B}(\psi)\right]_{i,d+1}\right|=0$ .

# B.1.2 Estimation of POC

We detail the estimation of POC based on Thm. 4.4 and the model (B.1) as follows. Denote the estimator of the conditional probability of the outcome given the $i$ -th selected feature $g_{i}(\mathbf{Z})$ as $\widehat{\mathbb{P}}(Y = y|g_i(\mathbf{Z}))$ . This can be achieved by either parametric models (such as logistic regression for the binary outcome) or non-parametric models (such as random forest or neural network). Then, the estimated marginal POC across $n$ data points is given by

$$
\widehat {\mathrm{M-POC}} (g _ {i} | \{\boldsymbol {o} ^ {(j)} \}) = \prod_ {j = 1} ^ {n} \left| \widehat {\mathbb {P}} \{Y = y ^ {(j)} | g _ {i} (\boldsymbol {Z}) = g _ {i} (\boldsymbol {z} ^ {(j)}) \} - \widehat {\mathbb {P}} \{Y = y ^ {(j)} | g _ {i} (\boldsymbol {Z}) \neq g _ {i} (\boldsymbol {z} ^ {(j)}) \} \right|,
$$

where $g_{i}(\mathbf{Z})$ is the i-th dimension of $g(\mathbf{Z})$ . Similarly, we can estimate the conditional probability of the outcome given all selected feature $g(\mathbf{Z})$ as $\widehat{\mathbb{P}}(Y = y|g(\mathbf{Z}))$ . Likewise, we have the estimated conditional POC as

$$
\begin{array}{l} \widehat {\mathrm{C-POC}} (g _ {i} | \{\boldsymbol {o} ^ {(j)} \}) = \prod_ {j = 1} ^ {n} \left| \widehat {\mathbb {P}} \{Y = y ^ {(j)} | g _ {i} (\boldsymbol {Z}) = g _ {i} (\boldsymbol {z} ^ {(j)}), g _ {- i} (\boldsymbol {Z}) = g _ {- i} (\boldsymbol {z} _ {- i} ^ {(j)}) \right\} \\ - \widehat {\mathbb {P}} \{Y = y ^ {(j)} | g _ {i} (\pmb {Z}) \neq g _ {i} (\pmb {z} ^ {(j)}), g _ {- i} (\pmb {Z}) = g _ {- i} (\pmb {z} _ {- i} ^ {(j)}) \} \Bigg |, \\ \end{array}
$$

where $g_{-i}(\cdot) \equiv g(\cdot) \setminus g_i(\cdot)$ is the complement of $g_i(\cdot)$ .

# B.2 Learning Algorithm based on POCs

Given the lack of an explicit form for causal quantities in the nonlinear models, we employ an iterative learning method based on Thm. 4.4 to capture complex nonlinear relationships among variables. The method begins with an initialized causal graph and then identifies necessary and sufficient features from Z with high POCs by setting g as a subset selection function. Following this, the causal graph can be updated among the selected nodes and Y, approximating $G_{V}$ in the nonlinear structural equation model based on data $\{\boldsymbol{o}^{(j)} = (\boldsymbol{z}^{(j)}, y^{(j)})\}_{1 \leq j \leq n}$ . This process is repeated until convergence is achieved. The main algorithm based on POCs is presented as follows.

# B.2.1 Step 1: Nonlinear Causal Structural Learning

In the first step, we employ a causal structural learning algorithm for nonlinear models to estimate the functional matrix $\boldsymbol{B}(\psi)$ as presented in (B.2). This estimation is performed considering the selector $g^{k}$ at the k-th iteration, where $g^{1}(\boldsymbol{Z}) := \boldsymbol{Z}$ . Aligning with the main text, we adopt the nonparametric acyclicity constraint on $\boldsymbol{B}(\psi)$ proposed in Zheng et al. [47], represented as $h_{n}(\boldsymbol{B}(\psi)) = 0$ . The loss function, defined by the augmented Lagrangian for the k-th iteration, is then formulated as:

$$
\tilde {L} (\boldsymbol {B} (\psi), \theta , \lambda | g ^ {k}, \{\boldsymbol {o} ^ {(j)} \}) = \tilde {f} (\boldsymbol {B} (\psi), \theta | g ^ {k}, \{\boldsymbol {o} ^ {(j)} \}) + \lambda \{h _ {c} (\boldsymbol {B} (\psi)) + h _ {n} (\boldsymbol {B} (\psi)) \}, \tag {B.3}
$$

where $\tilde{f}(\boldsymbol{B}(\psi),\theta|g^{k},\{\boldsymbol{o}^{(j)}\})$ denotes a nonlinear loss function with parameters $\theta$ , and $\lambda$ represents the Lagrange multiplier. The objective in (B.3) can be solved using existing nonlinear causal structural learning methods (refer to Yu et al. [44], Zhu et al. [48], Zheng et al. [47]) given the selector $g^{k}$ and data $\{\boldsymbol{o}^{(j)}=(\boldsymbol{z}^{(j)},y^{(j)})\}_{1\leq j\leq n}$ . The estimated functional matrix resulting from this process is represented as $\widehat{\boldsymbol{B}}^{k}(\widehat{\psi})$ .

# B.2.2 Step 2: Causal Relevance Measurement Using POCs

Following the estimation of the functional matrix $\widehat{\pmb{B}}^k (\widehat{\psi})$ , we assess the causal relevance of nodes through the probabilities of causation (POCs) to update the selection function $g$ . The loss function relating to the selector is defined as follows:

$$
\tilde {L} ^ {P} (g, \gamma | \widehat {\boldsymbol {B}} ^ {k} (\widehat {\psi}), \boldsymbol {o} ^ {(j)}) = - \sum_ {i = 1} ^ {d} \widehat {\mathrm{P}} (g _ {i} | \boldsymbol {o} ^ {(j)}) + \gamma | g |, \tag {B.4}
$$

where $|g|$ signifies the number of selected nodes in g with a penalty $\gamma$ for controlling the complexity of the selector. The term $g_{i}$ refers to the i-th dimension of g for $i = 1, \cdots, d$ . Here, $\widehat{\mathrm{P}}(\cdot)$ can represent either the estimated M-POC or C-POC as outlined in App. B.1. By optimizing different POCs, we can learn the corresponding best selector g according to the loss function detailed in (B.4) using the CAUSAL-REP algorithm as proposed in Wang & Jordan [42] by considering the selector g as the subset of Z. The resulting selector is denoted as $g^{k+1}$ .

# B.2.3 Step 3: Necessary and Sufficient Causal Structural Learning

Repeat the optimization process for loss functions in (B.3) and (B.4) for $k = 1, \cdots, K$ iterations until either the maximum iteration number K is reached, or the change in loss functions in (B.3) and (B.4) falls below a predefined tolerance level $\tau$ . The resultant estimated matrix is denoted as $\widehat{\boldsymbol{B}}(\widehat{\psi})$ , from which the estimated causal graph, $\widehat{G}_{V}$ , can be derived.

# B.3 The Computational Complexity of the NSCSL Algorithm

The computational complexity of NSCSL comprises two parts: the cost from causal discovery as $G(n,p)$ , and the estimation of causal effects/scores $F(n,p)$ , where n is the data sample size and p is the number of nodes. In the linear case, our method learns the features and causal graph through single-step optimization in (5), with complexity cubic in the number of nodes, $G(n,p) = \mathcal{O}(p^{3})$ following Zheng et al. [46]. Here, the causal effect computation is linear-time and thus is dominated. In the nonlinear case, according to App. B.2, the time complexity depends on the base causal discovery method and the number of max iterations K, yielding $\mathcal{O}[K(G(n,p) + F(n,p))]$ . Supporting runtime details are provided in § 6.

# B.4 Discussion on Scale Invariance

NSCSL is scale-invariant when we appropriately choose the causal discovery base learner and model the treatment effects/POCs. Though NOTEARS lacks scale invariance, our method's flexibility allows for the integration of scale-invariant causal discovery methods like NSCGL with FCI. Additionally, under LSEM, the rescaling will not affect the relative rank of the features based on absolute causal effects. In the nonlinear case, we proposed to use POCs which by their definitions are scale-invariant.

# C Extension to Markov Equivalence Class

Our proposed algorithm can also be extended to manage the Markov equivalence class of partial directed acyclic graphs when the causal graph cannot be uniquely identified from the observational studies.

# C.1 Additional Graph Terminology

A graph $\mathcal{G}$ that contains directed and/or undirected edges is termed as a partially directed graph. If this graph doesn't contain a directed cycle, it's referred to as a partially directed acyclic graph or PDAG. The DAG $\mathcal{G}$ is generally not identifiable from the distribution of $X$ based on conditional independence relationships, as per observational data [22]. This is because multiple DAGs can represent the same conditional independence relationships, and these DAGs form a Markov equivalence class (MEC), denoted as $MEC(\mathcal{G})$ . Two DAGs belong to the same MEC if and only if they have the same skeleton and the same $v$ -structures [22]. A MEC of DAGs can be uniquely symbolized by a completed partially directed acyclic graph (CPDAG) [35; 7], a graph that may contain both directed and undirected edges. A CPDAG adheres to the following: if $X_{i} \to X_{j}$ exists in the CPDAG, then $X_{i} \to X_{j}$ is present in every DAG in the MEC; and if $X_{i} - X_{j}$ exists in the CPDAG, then the MEC contains a DAG for which $X_{i} \to X_{j}$ as well as a DAG for which $X_{j} \to X_{i}$ .

# C.2 Model Identifiabilities

In the absence of further assumptions regarding the form of functions and/or noises, the model in (1) can only be identified up to MEC following the Markov and faithful assumptions [35; 25]. Below, we

explore the conditions for the unique identifiability of the DAG and potential strategies for addressing scenarios involving the MEC.

Initially, we consolidate cases where the DAG is uniquely identifiable. In the context of the LSEM, when the noises $\epsilon$ follow a Gaussian distribution, the resulting model corresponds to the standard linear-Gaussian model class, as investigated in Spirtes et al. [35] and Peters et al. [26]. In instances where the noises $\epsilon$ maintain equal variances, according to Peters & Bühlmann [24], the DAG G can be uniquely identified from observational data. Further, when the functions are linear but the noises are non-Gaussian, one can derive the LiNGAM as described in Shimizu et al. [34], where the true DAG can be uniquely identified under certain favorable conditions. In addition, as cited in Zheng et al. [47]; Rolland et al. [28], the nonlinear additive model can be identified from observational data. Another scenario of note arises when the corresponding MEC encompasses only one DAG; here, the DAG can be inherently identified from observational data. Recent score-based causal discovery algorithms [46; 44; 48; 5] typically take into account synthetic datasets generated from fully identifiable models, which provides practical relevance in evaluating the estimated graph in relation to the true DAG.

In instances where the true DAG is not identifiable, we reference the discussion in App. C. In such cases, a CPDAG uniquely symbolizes a MEC of DAGs that yield the same joint distribution of variables. This CPDAG can be inferred from observational data via a variety of causal discovery algorithms [see e.g., 35; 7; 34; 14; 4; 27]. One feasible approach to dealing with MEC involves enumerating all DAGs in the MEC derived from a given CPDAG [6]. It is conventional to encapsulate a range of potential effects or probabilities by their average or the minimum absolute value [6; 33]. However, such an approach typically proves computationally prohibitive for large graphs, necessitating computational shortcuts to acquire the causal effects or probabilities of causation without enumerating all DAGs in the MEC of the estimated CPDAG.

# C.3 Extended Algorithm for Markov Equivalence Class

In contrast to existing causal discovery algorithms, we consider a causal graph that is necessary and sufficient to portray causal relationships influencing the outcome variable Y. This aim is expressed as a causal identification constraint designed to restrict the causal structural learning to a smaller class of DAGs, as detailed in the ensuing section. Additionally, we employ causal effects or probabilities of causation as another loss function within the objective. This assists in identifying the DAG with the highest score of conditional scores of causation or causal effects, wherein the v-structures of interest are also constrained to have an endpoint at Y. Based on the estimated CPDAG, which involves a significantly smaller number of nodes, we can generate all DAGs within the MEC and prune superfluous nodes following aggregation.

Specifically, the proposed NSCSL algorithm can be adapted to handle MEC of PDAGs when the causal graph is not uniquely identifiable from observational data. Let's recall either the estimated matrix $\widehat{B}$ obtained by NSCSL based on causal effects, as presented in § 5.2 for the linear model, or the estimated functional matrix $\widehat{B}(\widehat{\psi})$ by the POC-based NSCSL in App. B for the nonlinear model. Based on these estimated (functional) matrices, we can derive the estimated causal graph $\widehat{\mathcal{G}}_V$ . This can further lead to estimation under its MEC by averaging over possible DAGs, as follows:

$$
\widehat {\mathcal {G}} _ {\boldsymbol {V}} ^ {*} = \frac {1}{| M E C (\widehat {\mathcal {G}} _ {\boldsymbol {V}}) |} \sum_ {\mathcal {G} _ {i} \in M E C (\widehat {\mathcal {G}} _ {\boldsymbol {V}})} \mathcal {G} _ {i}, \tag {C.1}
$$

where $|MEC(\widehat{\mathcal{G}}_V)|$ is the size of MEC for $\widehat{\mathcal{G}}_V$ . As we have previously noted, the number of nodes in $\widehat{\mathcal{G}}_V$ is much smaller than $p$ , making it feasible to generate all DAGs in the MEC and prune extraneous nodes following aggregation.

# D Technical Proofs

In this section, we provide proofs for Thms. 4.4 and 4.6. Let the symbol $\wedge$ denote the logical connective and, and the symbol $\vee$ denote the logical connective or. For two events $A$ and $B$ , $A \wedge B = \text{True}$ if $A = B = \text{True}$ , and $A \wedge B = \text{False}$ otherwise. Furthermore, $A \vee B = \text{False}$ if $A = B = \text{False}$ , and $A \vee B = \text{True}$ otherwise.

# D.1 Proofs of Thm. 4.4

Consider the marginal probability of causation (M-POC) for $Z_{i}$ in Def. 4.2 such that

$$
\mathrm{M-POC} _ {i} (y) \equiv \mathbb {P} \left\{Y \left(Z _ {i} \neq z _ {i}\right) \neq y, Y \left(Z _ {i} = z _ {i}\right) = y \right\},
$$

and the conditional probability of causation (C-POC) for $Z_{i}$ in Def. 4.3 such that

$$
\mathrm{C-POC} _ {i} (y) \equiv \mathbb {P} \left\{Y \left(Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}\right) \neq y, Y \left(Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}\right) = y \right\}.
$$

Our goal is to establish their lower bounds. In the following, we derive the lower bound for M-POC in Part 1 and the lower bound for C-POC in Part 2. Finally, we discuss conditions for the lower bound equality to be held in Part 3.

# D.1.1 Part 1: Lower Bound for M-POC

We focus on M-POC first. Based on the consistency assumption (A1), we have either $\{Y(Z_{i} \neq z_{i}) = y\}$ or $\{Y(Z_{i} \neq z_{i}) \neq y\}$ holds. Since the events $\{Y(Z_{i} \neq z_{i}) = y\}$ and $\{Y(Z_{i} \neq z_{i}) \neq y\}$ are disjoint, we have

$$
\{Y (Z _ {i} \neq z _ {i}) = y \} \vee \{Y (Z _ {i} \neq z _ {i}) \neq y \} = \text { True }, \tag {D.1}
$$

where the symbol $\vee$ denote the logical connective or, meaning one of the above event holds. Based on this fact, we focus on the second event $\{Y(Z_i = z_i) = y\}$ in M-POC which yields

$$
\{Y (Z _ {i} = z _ {i}) = y \} \tag {D.2}
$$

$$
= \{Y (Z _ {i} = z _ {i}) = y \} \wedge \text { True }
$$

$$
= \left\{Y (Z _ {i} = z _ {i}) = y \right\} \land \left[ \left\{Y (Z _ {i} \neq z _ {i}) = y \right\} \lor \left\{Y (Z _ {i} \neq z _ {i}) \neq y \right\} \right]
$$

$$
= \left[ \left\{Y (Z _ {i} = z _ {i}) = y \right\} \wedge \left\{Y (Z _ {i} \neq z _ {i}) = y \right\} \right] \vee \left[ \left\{Y (Z _ {i} = z _ {i}) = y \right\} \wedge \left\{Y (Z _ {i} \neq z _ {i}) \neq y \right\} \right],
$$

where the first equality is owing to the definition of the logical connective $\wedge$ , the second equality comes from (D.1), and the last equality follows the rule of interchange in the logical connectives, i.e., $A \wedge (B \vee C) = (A \wedge B) \vee (A \wedge C)$ for events $A, B$ , and $C$ . By noticing the last line is the logical connective or of two events, taking the probability on both sides of (D.2) gives

$$
\mathbb {P} \{Y (Z _ {i} = z _ {i}) = y \} \tag {D.3}
$$

$$
\leq \mathbb {P} \left[ \left\{Y \left(Z _ {i} = z _ {i}\right) = y \right\} \wedge \left\{Y \left(Z _ {i} \neq z _ {i}\right) = y \right\} \right] + \mathbb {P} \left[ \left\{Y \left(Z _ {i} = z _ {i}\right) = y \right\} \wedge \left\{Y \left(Z _ {i} \neq z _ {i}\right) \neq y \right\} \right]
$$

$$
= \underbrace {\mathbb {P} \left[ Y (Z _ {i} = z _ {i}) = y , Y (Z _ {i} \neq z _ {i}) = y \right]} _ {\eta_ {1}} + \mathbb {P} \left[ Y (Z _ {i} = z _ {i}) = y, Y (Z _ {i} \neq z _ {i}) \neq y \right]
$$

$$
= \eta_ {1} + \mathrm{M-POC} _ {i} (y),
$$

where the first inequality is owing to $\mathbb{P}(A\lor B)\leq \mathbb{P}(A) + \mathbb{P}(B)$ , and the equalities are due to the definitions of probabilities.

Similarly, by (A1), since the events $\{Y(Z_i = z_i) = y\}$ and $\{Y(Z_i = z_i) \neq y\}$ are disjoint, we have

$$
\left\{Y \left(Z _ {i} = z _ {i}\right) = y \right\} \vee \left\{Y \left(Z _ {i} = z _ {i}\right) \neq y \right\} = \text { True }.
$$

Based on this fact, the first event $\{Y(Z_{i} \neq z_{i}) = y\}$ in M-POC is

$$
\{Y (Z _ {i} \neq z _ {i}) = y \} \tag {D.4}
$$

$$
= \{Y (Z _ {i} \neq z _ {i}) = y \} \wedge \text { True }
$$

$$
= \left\{Y (Z _ {i} \neq z _ {i}) = y \right\} \wedge \left[ \left\{Y (Z _ {i} = z _ {i}) = y \right\} \vee \left\{Y (Z _ {i} = z _ {i}) \neq y \right\} \right]
$$

$$
= \left[ \left\{Y (Z _ {i} \neq z _ {i}) = y \right\} \wedge \left\{Y (Z _ {i} = z _ {i}) = y \right\} \right] \vee \left[ \left\{Y (Z _ {i} \neq z _ {i}) = y \right\} \wedge \left\{Y (Z _ {i} = z _ {i}) \neq y \right\} \right].
$$

By noticing the last line is the logical connective or of two events, taking the probability on both sides of (D.4) gives

$$
\mathbb {P} \{Y (Z _ {i} \neq z _ {i}) = y \} \geq \mathbb {P} \left[ \{Y (Z _ {i} \neq z _ {i}) = y \} \land \{Y (Z _ {i} = z _ {i}) = y \} \right] \tag {D.5}
$$

$$
= \mathbb {P} \left[ Y (Z _ {i} \neq z _ {i}) = y, Y (Z _ {i} = z _ {i}) = y \right] = \eta_ {1},
$$

where the first inequality is owing to $\mathbb{P}(A\lor B)\geq \mathbb{P}(A)$ and the last equality comes from the definition of $\eta_{1}$ .

Combining (D.3) and (D.5), we have

$$
\mathrm{M-POC} _ {i} (y) \tag {D.6}
$$

$$
\text {(by (D.3))} \geq \mathbb {P} \{Y (Z _ {i} = z _ {i}) = y \} - \eta_ {1}
$$

$$
\text {(by (D.5))} \geq \mathbb {P} \{Y (Z _ {i} = z _ {i}) = y \} - \mathbb {P} \{Y (Z _ {i} \neq z _ {i}) = y \}
$$

$$
= \mathbb {P} \{Y = y | Z _ {i} = z _ {i} \} - \mathbb {P} \{Y = y | Z _ {i} \neq z _ {i} \},
$$

where the last equation follows the results that $\mathbb{P}\{Y=y|do(X=x)\}=\mathbb{P}\{Y=y|X=x\}$ under the ignorability assumption (A2) following Rosenbaum & Rubin [29] and Pearl et al. [22, 23]. The proof of the first part thus is completed.

# D.1.2 Part 2: Lower Bound for C-POC

We next show the lower bound of conditional POC in Thm. 4.4 following the same logic as in Part 1. Based on the consistency assumption (A1), we have either $\{Y(Z_i \neq z_i, \mathbf{Z}_{-i} = \mathbf{z}_{-i}) = y\}$ or $\{Y(Z_i \neq z_i, \mathbf{Z}_{-i} = \mathbf{z}_{-i}) \neq y\}$ holds, i.e., these two events are disjoint, thus, we have

$$
\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \} \vee \{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) \neq y \} = \text { True }, \tag {D.7}
$$

where the symbol $\vee$ denote the logical connective $or$ , meaning one of the above event holds. Based on this fact, we focus on the second event $\{Y(Z_i = z_i, \mathbf{Z}_{-i} = \mathbf{z}_{-i}) = y\}$ in C-POC which yields

$$
\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \} \tag {D.8}
$$

$$
= \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \wedge \text { True }
$$

$$
= \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\}
$$

$$
\wedge \left[ \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \vee \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) \neq y \right\} \right]
$$

$$
= \left[ \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \wedge \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \right]
$$

$$
\vee \left[ \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \wedge \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) \neq y \right\} \right],
$$

where the first equality is owing to the definition of the logical connective $\wedge$ , the second equality comes from (D.7), and the last equality follows the rule of interchange in the logical connectives, i.e., $A \wedge (B \vee C) = (A \wedge B) \vee (A \wedge C)$ for events $A, B$ , and $C$ . By noticing the last line is the logical connective or of two events, taking the probability on both sides of (D.8) gives

$$
\mathbb {P} \{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \} \tag {D.9}
$$

$$
\leq \mathbb {P} \left[ \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \wedge \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \right]
$$

$$
+ \mathbb {P} \left[ \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \wedge \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) \neq y \right\} \right]
$$

$$
= \underbrace {\mathbb {P} \left[ Y (Z _ {i} = z _ {i} , \mathbf {Z} _ {- i} = \boldsymbol {z} _ {- i}) = y , Y (Z _ {i} \neq z _ {i} , \mathbf {Z} _ {- i} = \boldsymbol {z} _ {- i}) = y \right]} _ {\eta_ {2}}
$$

$$
+ \mathbb {P} \left[ Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y, Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) \neq y \right]
$$

$$
= \eta_ {2} + \mathbf {C - P O C} _ {i} (y),
$$

where the first inequality is owing to $\mathbb{P}(A\vee B)\leq \mathbb{P}(A) + \mathbb{P}(B)$ , and the equalities are due to the definitions of probabilities. Similarly, by (A1), since the events $\{Y(Z_i = z_i,\mathbf{Z}_{-i} = \mathbf{z}_{-i}) = y\}$ and $\{Y(Z_i = z_i,\mathbf{Z}_{-i} = \mathbf{z}_{-i})\neq y\}$ are disjoint, we have

$$
\left\{Y \left(Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}\right) = y \right\} \vee \left\{Y \left(Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}\right) \neq y \right\} = \text { True }.
$$

Based on this fact, the first event $\{Y(Z_{i} \neq z_{i}, Z_{-i} = z_{-i}) = y\}$ in C-POC is

$$
\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \} \tag {D.10}
$$

$$
= \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \land \text { True }
$$

$$
= \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\}
$$

$$
\wedge \left[ \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \vee \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) \neq y \right\} \right]
$$

$$
= \left[ \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \wedge \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \right]
$$

$$
\vee \left[ \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \wedge \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) \neq y \right\} \right].
$$

By noticing the last line is the logical connective or of two events, taking the probability on both sides of (D.10) gives

$$
\mathbb {P} \{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \} \tag {D.11}
$$

$$
\geq \mathbb {P} \left[ \left\{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \wedge \left\{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right\} \right]
$$

$$
= \mathbb {P} \left[ Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y, Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \right] = \eta_ {2},
$$

where the first inequality is owing to $\mathbb{P}(A\lor B)\geq \mathbb{P}(A)$ and the last equality comes from the definition of $\eta_{2}$ . Combining (D.9) and (D.11), we have

$$
\mathrm{C} - \mathrm{POC} _ {i} (y) \tag {D.12}
$$

$$
\text {(by (D.9))} \geq \mathbb {P} \{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \} - \eta_ {2}
$$

$$
\text {(by (D.11))} \geq \mathbb {P} \{Y (Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \} - \mathbb {P} \{Y (Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}) = y \}
$$

$$
= \mathbb {P} \{Y = y | Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i} \} - \mathbb {P} \{Y = y | Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i} \},
$$

where the last equation follows the results that $\mathbb{P}\{Y = y|do(X = x)\} = \mathbb{P}\{Y = y|X = x\}$ under the ignorability assumption (A2) following Rosenbaum & Rubin [29] and Pearl et al. [22, 23]. The proof of the second part thus is completed.

# D.1.3 Part 3: Conditions to Achieve Lower Bounds

In this part, we discuss the conditions under which the POCs are equal to their corresponding lower bounds. To this end, we introduce the following monotonicity condition.

(C1\*). Monotonicity:

(i) $\{Y(\mathbf{Z} \neq \mathbf{z}) = y\} \land \{Y(\mathbf{Z} = \mathbf{z}) \neq y\} = \text{False};$   
(ii) $\{Y(Z_i\neq z_i) = y\} \wedge \{Y(Z_i = z_i)\neq y\} = \mathrm{False}.$

Here, (C1\*.i) is proposed in Section 9.2.3 in Pearl et al. [22] and also Tian & Pearl [39] to establish the identifiability of the probability of causation. We generalize the condition (C1\*.i) to the condition (C1\*.ii) so that we can extend the results in Theorem 9.2.14 in Pearl et al. [22] for POCs in Defs. 4.2 and 4.3.

We detail the case of M-POC first. By noticing the monotonicity condition in (C1\*.ii) such that $\left[\{Y(Z_i \neq z_i) = y\} \wedge \{Y(Z_i = z_i) \neq y\}\right] = \text{False}$ , we can simplify (D.4) as

$$
\{Y (Z _ {i} \neq z _ {i}) = y \} = [ \{Y (Z _ {i} \neq z _ {i}) = y \} \land \{Y (Z _ {i} = z _ {i}) = y \} ]. \tag {D.13}
$$

Substituting (D.13) into (D.2) yields

$$
\left\{Y (Z _ {i} = z _ {i}) = y \right\} = \left\{Y (Z _ {i} \neq z _ {i}) = y \right\} \vee \left[ \left\{Y (Z _ {i} = z _ {i}) = y \right\} \wedge \left\{Y (Z _ {i} \neq z _ {i}) \neq y \right\} \right]. \tag {D.14}
$$

Based on the consistency assumption (A1), we have either $\{Y(Z_i \neq z_i) = y\}$ or $\{Y(Z_i \neq z_i) \neq y\}$ holds, and thus the events $\{Y(Z_i \neq z_i) = y\}$ and $[\{Y(Z_i = z_i) = y\} \wedge \{Y(Z_i \neq z_i) \neq y\}]$ are disjoint. Therefore, taking the probability on both sides of (D.14) gives

$$
\mathbb {P} \{Y (Z _ {i} = z _ {i}) = y \} = \mathbb {P} \{Y (Z _ {i} \neq z _ {i}) = y \} + \mathbb {P} \{Y (Z _ {i} = z _ {i}) = y, Y (Z _ {i} \neq z _ {i}) \neq y \}. \tag {D.15}
$$

Recall Def. 4.2. Based on (D.15), we have

$$
\begin{array}{l} \mathrm{M-POC} _ {i} (y) = \mathbb {P} \left\{Y \left(Z _ {i} \neq z _ {i}\right) \neq y, Y \left(Z _ {i} = z _ {i}\right) = y \right\} \\ = \mathbb {P} \{Y (Z _ {i} = z _ {i}) = y \} - \mathbb {P} \{Y (Z _ {i} \neq z _ {i}) = y \} \\ = \mathbb {P} \{Y = y | Z _ {i} = z _ {i} \} - \mathbb {P} \{Y = y | Z _ {i} \neq z _ {i} \}, \\ \end{array}
$$

where the last equation follows the results that $\mathbb{P}\{Y=y|do(X=x)\}=\mathbb{P}\{Y=y|X=x\}$ under the Ignorability assumption by Rosenbaum & Rubin [29] and Pearl et al. [22, 23]. Thus, the lower bound equality for M-POC holds when an additional monotonicity condition is imposed. Following the same logic, we can show the lower bound equality for C-POC holds given the monotonicity condition. We omit the details for brevity.

# D.2 Proofs of Thm. 4.6

As a direct result of Thm. 4.4, below we can establish the relationship between POC and the corresponding expected mean outcome given different combinations of the confounders involving $Z_{i}$ . Specifically, we take expectations over Y on both sides of (D.6) and (D.12). When Y is nonnegative, this yields

$$
\begin{array}{l} \sum_ {y \in \mathcal {L}} y \mathrm{M-POC} _ {i} (y) \geq \sum_ {y \in \mathcal {L}} y \left[ \mathbb {P} \{Y (Z _ {i} = z _ {i}) = y \} - \mathbb {P} \{Y (Z _ {i} \neq z _ {i}) = y \} \right] \\ \geq \mathbb {E} \left\{Y \left(Z _ {i} = z _ {i}\right) \right\} - \mathbb {E} \left\{Y \left(Z _ {i} \neq z _ {i}\right) \right\}, \\ \end{array}
$$

and

$$
\sum_ {y \in \mathcal {L}} y \mathrm{C-POC} _ {i} (y) \geq \sum_ {y \in \mathcal {L}} y \left[ \mathbb {P} \{Y (Z _ {i} = z _ {i}, \boldsymbol {Z} _ {- i} = \boldsymbol {z} _ {- i}) = y \} - \mathbb {P} \{Y (Z _ {i} \neq z _ {i}, \boldsymbol {Z} _ {- i} = \boldsymbol {z} _ {- i}) = y \} \right]
$$

$$
\geq \mathbb {E} \left\{Y \left(Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}\right) \right\} - \mathbb {E} \left\{Y \left(Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i}\right) \right\}.
$$

Under the ignorability assumption (A2) following Rosenbaum & Rubin [29] and Pearl et al. [22, 23], we have

$$
\sum_ {y \in \mathcal {L}} y \mathrm{M-POC} _ {i} (y) \geq \delta_ {M} (z _ {i}) \equiv \mathbb {E} \{Y | Z _ {i} = z _ {i} \} - \mathbb {E} \{Y | Z _ {i} \neq z _ {i} \}, \tag {D.16}
$$

$$
\sum_ {y \in \mathcal {L}} y \mathrm{C-POC} _ {i} (y) \geq \delta_ {C} (z _ {i}) \equiv \mathbb {E} \{Y | Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i} \} - \mathbb {E} \{Y | Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i} \},
$$

where $\delta_{M}(z_{i})$ and $\delta_{C}(z_{i})$ are defined as the marginal and conditional causal effects using the differences of expectations based on the corresponding POC. Recall the definitions of natural causal effects for $Z_{i}$ as

$$
T E _ {i} = \mathbb {E} \{Y (Z _ {i} = z _ {i} + 1) \} - \mathbb {E} \{Y (Z _ {i} = z _ {i}) \},
$$

$$
D E _ {i} = \mathbb {E} \{Y (Z _ {i} = z _ {i} + 1, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i} ^ {(z _ {i})}) \} - \mathbb {E} \{Y (Z _ {i} = z _ {i}) \},
$$

where $\pmb{z}_{-i}^{(z_i)}$ is the value of $Z_{-i}$ if setting $do(Z_i = z_i)$ . When $Z_i$ is binary, by comparing the definitions with $Z_i \in \{0,1\}$ , we have

$$
\left| T E _ {i} \right| \geq \delta_ {M} (z _ {i}), \quad \left| D E _ {i} \right| \geq \delta_ {C} (z _ {i}). \tag {D.17}
$$

Therefore, combining (D.16) and (D.17) yields the second conclusion in Thm. 4.6 that

$$
\min \left\{\sum_ {y \in \mathcal {L}} y \mathrm{M} - \mathrm{POC} _ {i} (y), | T E _ {i} | \right\} \geq \mathbb {E} \left\{Y \mid Z _ {i} = z _ {i} \right\} - \mathbb {E} \left\{Y \mid Z _ {i} \neq z _ {i} \right\},
$$

$$
\min \{\sum_ {y \in \mathcal {L}} y \mathrm{C-POC} _ {i} (y), | D E _ {i} | \} \geq \mathbb {E} \{Y | Z _ {i} = z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i} \} - \mathbb {E} \{Y | Z _ {i} \neq z _ {i}, \mathbf {Z} _ {- i} = \mathbf {z} _ {- i} \}.
$$

Further, when $Z_{i}$ is binary, the absolute values of $\delta_{M}(z_{i})$ and $\delta_{C}(z_{i})$ is equal to the absolute values of $TE_{i}$ and $DE_{i}$ , respectively. Combining this with (D.16) yields the second conclusion in Thm. 4.6. Finally, following the same logic in App. D.1.3, we can show the lower bound equality for Thm. 4.6 holds when the monotonicity condition is imposed. We omit the details for brevity. The proof is thus completed.

# D.3 Proofs of Thm. 5.1

We investigate the theoretical consistency of the proposed causal structural learning methods under Model (1) with independent Gaussian error and equal variance, using the score-based method such as NOTEARS that minimizes the loss in (5).

Notations and Conditions: We first detail some notations and the required conditions in Thm. 5.1 below. Denote $W_{j} \in R^{n \times 1}$ as the row vector of a matrix $W \in \mathbb{R}^{(d+1) \times n}$ for $j = 1, \cdots, d + 1$ . Let $\operatorname{supp}(v) = \{i : v_{i} \neq 0\}$ denote the support of a vector v, which is the set of indices of nonzero terms of v. The first condition requires a bounded true matrix B as follows.

Condition D.1. The true matrix $B \in \mathbb{R}^{(d+1) \times (d+1)}$ is bounded such that $\|B\|_{2} = \mathcal{O}(1)$ , and the maximum degree across different rows is less than the number of nodes in the graph, i.e., $s_{0} = \max_{j} \operatorname{supp}(B_{j}) \leq d + 1$ .

Next, we recall the linear structural equation model condition on $\boldsymbol{W} \equiv [g(\boldsymbol{Z})^{\top}, Y]^{\top}$ as follows.

Condition D.2. Suppose Model (1) such that $\pmb{W} = \pmb{B}^{\top}\pmb{W} + \epsilon$ , with independent Gaussian error $\epsilon$ and the error variance is a constant $\sigma^2$ .

Furthermore, define the order as $\nu = [\nu_{1}, \nu_{2}, \cdots, \nu_{d+1}]^{\top}$ is a permutation of indices $\{1, 2, ..., d, d+1\}$ of nodes in the causal graph, such that $\nu_{j} \in \{1, 2, ..., d+1\}$ . If we set the last node to be the target

variable, then the last index is fixed to be $d+1$ . Let $\boldsymbol{B}(\nu)$ as $\boldsymbol{B}(\nu)=\left[\boldsymbol{B}_{1}(\nu),\cdots,\boldsymbol{B}_{d+1}(\nu)\right]^{\top}$ , such that

$$
\boldsymbol {B} _ {j} (\nu) = \underset {\boldsymbol {\beta}: \operatorname{supp} (\boldsymbol {\beta}) \in \{\nu_ {1}, \dots , \nu_ {j - 1} \}} {\arg \min} \mathbb {E} (\boldsymbol {W} _ {\nu_ {j}} - \boldsymbol {\beta} ^ {\top} \boldsymbol {W}) ^ {2}, \quad j = 1, 2, \dots , d + 1.
$$

Note that the true causal graph B that generates W also has a topological order, $\nu^{*}$ , which is called the true topological order (this true order may not be unique). Then the true causal graph B can also be denoted as $\boldsymbol{B}(\nu^{*})$ . Denote the set of all permutations of $\{1,2,\cdots,d+1\}$ as $\Upsilon$ and the set of true order as $\Upsilon^{*}$ such that $\Upsilon^{*}\subset\Upsilon$ . Denote the order of $\widehat{B}$ as $\nu\in\Upsilon$ , so $\widehat{B}$ can also be notated as $\widehat{B}(\nu)$ . Then let $s_{j}(\nu)=\operatorname{supp}(B_{j}(\nu)),\widehat{s}_{j}(\nu)=\operatorname{supp}(\widehat{B}_{j}(\nu))$ . The last condition assumes the consistency of the topological ordering.

Condition D.3. The true topological ordering of B, i.e., $\nu^{*}$ , is consistently estimated.

Condition D.3 is commonly imposed when proving the error bound of causal structural learning results [see Condition (A6) in 33].

Overview of Proof: With the aforementioned three conditions, our proof follows similar strategies of the causal structural learning literature [e.g., 33] but accounts for the extra penalty term from causal effects. Notice that the explicit forms of causal effects under LSEM are linear combinations of elements of B. This implies our new regulation can similarly vanish away as n goes to infinity. For simplicity, in the rest of the proof, we show the consistency given a selection function g for brevity. To start with, proving the consistency of $\widehat{B}$ returned by NSCSL with NOTEARS as baseline is equivalent to showing that the $\widehat{B}$ solving

$$
\widehat {\boldsymbol {B}} = \arg \min \| \boldsymbol {W} - \boldsymbol {B} ^ {\top} \boldsymbol {W} \| _ {2} ^ {2} + \lambda h _ {2} (\boldsymbol {B}), \quad \text { subject   to } h _ {1} (\boldsymbol {B}) = 0. \tag {D.18}
$$

is consistent. Then the proof of Thm. 5.1 follows the same strategy as the proof of Proposition 1 in [33]. The key difference between Thm. 5.1 and Proposition 1 in [33] is that the loss function in (D.18) contains extra penalty term $h_2(B) = \delta^* - \sum_{i=1}^{d} |\widehat{CE}_i(B)| + \sum_{i=1}^{d+1} |b_{i,d+1}|$ , that involves causal effect information compared to the original loss function

$$
\widetilde {\boldsymbol {B}} = \arg \min \| \boldsymbol {W} - \boldsymbol {B} ^ {\top} \boldsymbol {W} \| _ {2} ^ {2}, \quad \text { subject   to } h _ {1} (\boldsymbol {B}) = 0,
$$

in [46]. In this proof, we need to show a main statement and the rest of the proof will follow the same procedure of Steps 2-3 in Appendix Section 9 of [33]. The Main Statement is:

$$
\left\| \widehat {\boldsymbol {B}} (\nu) - \boldsymbol {B} (\nu) \right\| _ {2} = \mathcal {O} \big (\sqrt {\sum_ {j = 1} ^ {d + 1} \frac {\log n}{n} (s _ {j} (\nu) + \widehat {s} _ {j} (\nu))} + \frac {\lambda}{n} \sqrt {s _ {0} (d + 1)} \big),
$$

with $\lambda$ satisfy the bound of $\mathcal{O}((n\log n)^{1 / 2})$ for arbitrary $\nu \in \Upsilon$ .

Proof of Main Statement: Since $\widehat{B} (\nu)$ solves (D.18), we have

$$
\left\| \boldsymbol {W} - \widehat {\boldsymbol {B}} (\nu) ^ {\top} \boldsymbol {W} \right\| _ {2} ^ {2} + \lambda h _ {2} (\widehat {\boldsymbol {B}} (\nu)) \leq \left\| \boldsymbol {W} - \boldsymbol {B} (\nu) ^ {\top} \boldsymbol {W} \right\| _ {2} ^ {2} + \lambda h _ {2} (\boldsymbol {B} (\nu)). \tag {D.19}
$$

Following Theorem 7.1 in Van de Geer & Bühlmann [40] and Shi & Li [33], we can transform (D.19) to

$$
\frac {1}{2} \| \boldsymbol {B} (\nu) ^ {\top} \boldsymbol {W} - \widehat {\boldsymbol {B}} (\nu) ^ {\top} \boldsymbol {W} \| _ {2} ^ {2} + \lambda h _ {2} (\widehat {\boldsymbol {B}} (\nu)) \leq 2 \sum_ {j = 1} ^ {d + 1} \kappa (s _ {j} (\nu) + \widehat {s} _ {j} (\nu)) \log n + \lambda h _ {2} (\boldsymbol {B} (\nu)), \tag {D.20}
$$

for some $\kappa > 0$ . Recall Condition D.2 such that $W = B^{\top}W + \epsilon$ with $\epsilon \sim N(\mathbf{0}^{n \times 1}, I_n \times \sigma^2)$ . From Equation (65) in [33], we have $0 < \kappa^* < \sigma^2$ such that

$$
\kappa^ {*} n \sum_ {j = 1} ^ {d + 1} \| \boldsymbol {B} _ {j} (\nu) - \widehat {\boldsymbol {B}} _ {j} (\nu) \| _ {2} ^ {2} \leq \frac {1}{2} \| \boldsymbol {B} (\nu) ^ {\top} \boldsymbol {W} - \widehat {\boldsymbol {B}} (\nu) ^ {\top} \boldsymbol {W} \| _ {2} ^ {2} \leq \frac {1}{\kappa^ {*}} n \sum_ {j = 1} ^ {d + 1} \| \boldsymbol {B} _ {j} (\nu) - \widehat {\boldsymbol {B}} _ {j} (\nu) \| _ {2} ^ {2}, \forall \nu . \tag {D.21}
$$

Combining (D.21) with (D.20), we have

$$
\kappa^ {*} n \sum_ {j = 1} ^ {d + 1} \| \boldsymbol {B} _ {j} (\nu) - \widehat {\boldsymbol {B}} _ {j} (\nu) \| _ {2} ^ {2} + \lambda h _ {2} (\widehat {\boldsymbol {B}} (\nu)) \leq 2 \sum_ {j = 1} ^ {d + 1} \kappa (s _ {j} (\nu) + \widehat {s} _ {j} (\nu)) \log n + \lambda h _ {2} (\boldsymbol {B} (\nu)). \tag {D.22}
$$

If $\lambda h_2(\widehat{B} (\nu)) > \lambda h_2(B(\nu))$ , the main statement directly holds.

In the other case, we focus on showing the main statement when $\lambda h_2(\widehat{B} (\nu))\leq \lambda h_2(B(\nu))$ . Specifically, we first define a vector

$$
\boldsymbol {\mu} (\nu) = \left[ \widehat {\boldsymbol {B}} _ {1} (\nu) - \boldsymbol {B} _ {1} (\nu), \widehat {\boldsymbol {B}} _ {2} (\nu) - \boldsymbol {B} _ {2} (\nu), \dots , \widehat {\boldsymbol {B}} _ {p} (\nu) - \boldsymbol {B} _ {d + 1} (\nu) \right] ^ {\top} \in \mathbb {R} ^ {(d + 1) \times 1}.
$$

Our goal is to bound $\| \pmb{\mu}(\nu)\| _2 = \sum_{j = 1}^{d + 1}\| \widehat{B}_j(\nu) - B_j(\nu)\| _2^2$ . Define $\mathcal{M}(\nu) = \mathrm{supp}(\pmb {\mu}(\nu))$ and $\mathcal{M}^c (\nu)$ is the complementary set. Then denote $\pmb {\mu}(\nu)_{\mathcal{M}(\nu)}$ as the vector formed by elements of $\pmb {\mu}(\nu)$ in $\mathcal{M}(\nu)$ . Denote $\pmb {\mu}(\nu)_{\mathcal{M}^c (\nu)}$ as the vector formed by elements of $\pmb {\mu}(\nu)$ in $\mathcal{M}^c (\nu)$ . Hence, we can show

$$
\begin{array}{l} \left\| \lambda h _ {2} (\boldsymbol {B} (\nu)) \right\| _ {1} - \left\| \lambda h _ {2} (\widehat {\boldsymbol {B}} (\nu)) \right\| _ {1} \\ \leq \left\| \lambda \sum_ {i = 1} ^ {d} \left\{C E _ {i} (\boldsymbol {B} (\nu)) - \widehat {C E} _ {i} (\widehat {\boldsymbol {B}} (\nu)) \right\} + \lambda \left\{\boldsymbol {B} _ {d + 1} (\nu) - \widehat {\boldsymbol {B}} _ {d + 1} (\nu) \right\} \right\| _ {1} \tag {D.23} \\ \leq \lambda \sum_ {i = 1} ^ {d} \left\| C E _ {i} (\boldsymbol {B} (\nu)) - \widehat {C E} _ {i} (\widehat {\boldsymbol {B}} (\nu)) \right\| _ {1} + \lambda \left\| \boldsymbol {B} _ {d + 1} (\nu) - \widehat {\boldsymbol {B}} _ {d + 1} (\nu) \right\| _ {1}. \\ \end{array}
$$

Recall the close form of causal effects in Def. 4.5 derived in § 5.1 under the LSEM model. We have

$$
D E _ {i} (\boldsymbol {B}; g) = \theta_ {i},
$$

where $\theta_{i}$ presents the weight of the direct edge $g(\mathbf{Z})_{i} \to Y$ according to (1) and Def. 4.5. In addition, the total causal effect can be quantified by the path method [see e.g., 43; 20] as

$$
T E _ {i} (\boldsymbol {B}; g) = \sum_ {k = 1} ^ {m _ {i}} P E \{\pi_ {i} ^ {(k)} \},
$$

where $PE\{\pi_{i}^{(k)}\}=b_{i,l_{1}}\cdots b_{l_{\tau_{k}},(d+1)}$ is the causal effect of $g_{i}(\mathbf{Z})$ on Y through the directed path $\pi_{i}^{(k)}=\{i,l_{1},\cdots,l_{\tau_{k}},d+1\}\in\pi_{i}$ with length $\tau_{k}+1$ , and $b_{i,j}$ is the weight of the edge $g_{i}(\mathbf{Z})\to g_{j}(\mathbf{Z})$ if it exists, and $b_{i,j}=0$ otherwise, for $i,j\in\{1,\cdots,d\}$ , and $b_{l_{\tau_{k}},(d+1)}=\theta_{l_{\tau_{k}}}$ as the direct edge from $g_{l_{\tau_{k}}}(\mathbf{Z})$ to Y. Both $TE_{i}$ and $DE_{i}$ can be explicitly calculated given a matrix B under a selector g. We denote their estimates as $\widehat{TE}_{i}$ and $\widehat{DE}_{i}$ given the estimated matrix $\widehat{B}$ and g. Using the direct causal effects as an example, we have (D.23) be further bounded by

$$
\begin{array}{l} \lambda \sum_ {i = 1} ^ {d} \left\| C E _ {i} (\boldsymbol {B} (\nu)) - \widehat {C E _ {i}} (\widehat {\boldsymbol {B}} (\nu)) \right\| _ {1} + \lambda \left\| \boldsymbol {B} _ {d + 1} (\nu) - \widehat {\boldsymbol {B}} _ {d + 1} (\nu) \right\| _ {1} \\ \leq C _ {1} \lambda \sum_ {i = 1} ^ {d} \left\| \boldsymbol {B} _ {i} (\nu) - \widehat {\boldsymbol {B}} _ {i} (\nu) \right\| _ {1} + \lambda \left\| \boldsymbol {B} _ {d + 1} (\nu) - \widehat {\boldsymbol {B}} _ {d + 1} (\nu) \right\| _ {1} \tag {D.24} \\ \leq \max \{C _ {1}, 1 \} \lambda \sum_ {i = 1} ^ {d + 1} \left\| \boldsymbol {B} _ {i} (\nu) - \widehat {\boldsymbol {B}} _ {i} (\nu) \right\| _ {1}, \\ \end{array}
$$

for some constant $C_1 > 0$ . Recall $\| \pmb{\mu}(\nu) \|_2 = \sum_{j=1}^{d+1} \| \widehat{\pmb{B}}_j(\nu) - \pmb{B}_j(\nu) \|_2^2$ . Combining (D.23) and (D.24), we have

$$
\begin{array}{l} \left\| \lambda h _ {2} (\boldsymbol {B} (\nu)) \right\| _ {1} - \left\| \lambda h _ {2} (\widehat {\boldsymbol {B}} (\nu)) \right\| _ {1} \leq \max \{C _ {1}, 1 \} \lambda \sum_ {i = 1} ^ {d + 1} \left\| \boldsymbol {B} _ {i} (\nu) - \widehat {\boldsymbol {B}} _ {i} (\nu) \right\| _ {1} \\ \leq \max \left\{C _ {1}, 1 \right\} \lambda \| \boldsymbol {\mu} (\nu) \| _ {1} \leq \max \left\{C _ {1}, 1 \right\} \lambda \left(\| \boldsymbol {\mu} (\nu) _ {\mathcal {M} (\nu)} \| _ {1} + \| \boldsymbol {\mu} (\nu) _ {\mathcal {M} ^ {c} (\nu)} \| _ {1}\right) \tag {D.25} \\ \leq 2 \max \left\{C _ {1}, 1 \right\} \lambda \sqrt {s _ {0} (d + 1)} \| \boldsymbol {\mu} (\nu) _ {\mathcal {M} (\nu)} \| _ {2}, \\ \end{array}
$$

where the last inequality follows Equation (67) in Shi & Li [33]. Together with (D.22), we have

$$
\begin{array}{l} \kappa^ {*} n \sum_ {j = 1} ^ {d + 1} \| \boldsymbol {B} _ {j} (\nu) - \widehat {\boldsymbol {B}} _ {j} (\nu) \| _ {2} ^ {2} - 2 \sum_ {j = 1} ^ {d + 1} \kappa (s _ {j} (\nu) + \widehat {s} _ {j} (\nu)) \log n \\ = \kappa^ {*} n \| \boldsymbol {\mu} (\nu) \| _ {2} - 2 \sum_ {j = 1} ^ {d + 1} \kappa (s _ {j} (\nu) + \widehat {s} _ {j} (\nu)) \log n \leq 2 \max \{C _ {1}, 1 \} \lambda \sqrt {s _ {0} (d + 1)} \| \boldsymbol {\mu} (\nu) _ {\mathcal {M} (\nu)} \| _ {2}, \\ \end{array}
$$

and thus

$$
\kappa^ {*} \| \boldsymbol {\mu} (\nu) \| _ {2} \leq 2 \sum_ {j = 1} ^ {d + 1} \kappa (s _ {j} (\nu) + \widehat {s} _ {j} (\nu)) \frac {\log n}{n} + 2 \max \left\{C _ {1}, 1 \right\} \frac {\lambda \sqrt {s _ {0} (d + 1)}}{n} \| \boldsymbol {\mu} (\nu) _ {\mathcal {M} (\nu)} \| _ {2}. \tag {D.26}
$$

Rearranging (D.26) leads to

$$
\| \boldsymbol {\mu} (\nu) \| _ {2} \leq \mathcal {O} \big (\sqrt {\sum_ {j = 1} ^ {d + 1} \frac {\log n}{n} (s _ {j} (\nu) + \widehat {s} _ {j} (\nu))} + \frac {\lambda \sqrt {s _ {0} (d + 1)}}{n} \big),
$$

for arbitrary $\nu \in \Upsilon$ . Hence the proof of the Main Statement is completed. The rest of the proofs follow the similar arguments in Proposition 1 of [33]. For $\lambda$ satisfying the bound of $\mathcal{O}((n\log n)^{1/2})$ , the consistency of the estimated matrix holds for arbitrary $\nu \in \Upsilon$ . The proof of Thm. 5.1 is hence completed.

# E Additional Simulation Results

In this section, we provide additional simulation configurations and results.

# E.1 Simulation Configurations

Table E.1: Hyper-parameters information. 

<table><tr><td>Hyper-parameters</td><td>Values</td></tr><tr><td>Maximum number of dual ascent steps in NOTEARS/NSCSL</td><td>100</td></tr><tr><td>Tolerance  $\tau$  of acyclic constraint  $h_1$  to be violated in NOTEARS/NSCSL</td><td>1e-8</td></tr><tr><td>Maximum of parameters for the hard constraints in NOTEARS/NSCSL</td><td>1e+16</td></tr><tr><td> $L_1$  penalty term  $l$  in NOTEARS/NSCSL</td><td>0</td></tr><tr><td>Conditional independent testing in PC</td><td>“Fisher-Z”</td></tr><tr><td>Pruning threshold for all methods</td><td>0.3</td></tr></table>

# E.2 More Real Data Analyses on Yeast Data

The estimated causal graph among candidate QTLs and the outcome is shown in Fig. E.1 under the proposed method and NOTEARS [46] for illustration. The purple node represents the outcome, the blue nodes indicate QTLs with a positive causal impact, and the red nodes denote QTLs with a negative impact. Grey nodes are noisy QTLs without causal impact. Blue and red arrows represent positive and negative causal links, respectively. Causal effects from candidate genes on the genetic variant YER124C in yeast gene data are summarized using NSCSL in Table E.2. Fig. E.1 demonstrates that the proposed algorithm can discover necessary and sufficient causal relationships with better performance compared to the current causal discovery benchmark. Specifically, all nodes with causal effects (either blue or red nodes in the causal graph) on the outcome are identified under NSCSL. Furthermore, the proposed algorithm identifies an additional gene, 'YLR303W', which is not found in NOTEARS. Here, 'YLR303W' is essential for sulfur amino acid synthesis [3], with an estimated total causal effect of -0.06 on the target gene expression, as shown in Table E.2. And 'YER124C' of interest is a daughter cell-specific protein involved in cell wall metabolism [8]. It has been shown that sulfur amino acid synthesis can influence cell wall metabolism [37; 9]. These indicate that NSCGL which additionally identified 'YLR303W' performs better than NOTEARS. These observations align with findings from our simulation studies, further supporting NSCSL's superiority in revealing important causal features.

Table E.2: Summary of candidate genes that affect the variant YER124C in yeast gene data by NSCSL. 

<table><tr><td>Gene Code</td><td>Gene Function</td><td>Direct Effect</td><td>Total Effect</td></tr><tr><td>YOL058W</td><td>Arginosuccinate synthetase</td><td>-0.20</td><td>-0.22</td></tr><tr><td>YCL020W</td><td>Genotype regulators</td><td>-0.26</td><td>-0.26</td></tr><tr><td>YDR074W</td><td>Trehalose-6-phosphate phosphatase</td><td>0.0</td><td>-0.15</td></tr><tr><td>YMR105C</td><td>Phosphoglucomutase</td><td>-0.28</td><td>-0.28</td></tr><tr><td>YKL178C</td><td>Cell surface a factor receptor</td><td>0.06</td><td>0.07</td></tr><tr><td>YLR303W</td><td>Required for sulfur amino acid synthesis</td><td>0.0</td><td>-0.06</td></tr><tr><td>YCL030C</td><td>Multifunctional enzyme</td><td>0.0</td><td>-0.22</td></tr><tr><td>YER073W</td><td>Mitochondrial aldehyde dehydrogenase</td><td>0.0</td><td>-0.06</td></tr></table>

![](images/f43bbcd30c5b2c3ec6192bd6799d3736c7617f4aaee287c295612a036d67294f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["YMR158C-B"] --> B["YPR002C-A"]
    C["YMR090W"] --> D["YOL106W"]
    E["YDR074W"] --> F["YMR105C"]
    G["YHR092C"] --> H["YCL020W"]
    I["YBR068C"] --> J["YNL036W"]
    K["YOL116W"] --> L["YIL116W"]
    M["YOL088W"] --> N["YCR005C"]
    O["YDL066W"] --> P["YER069W"]
    Q["YOR145W"] --> R["YBR145W"]
    S["YBR256C"] --> T["Outcome"]
    U["YKL178C"] --> V["YBR256C"]
    W["YOR158W"] --> X["YER158W"]
    Y["YBL107W-A"] --> Z["YPR002C-A"]
    AA["Outcome"] --> AB["Outcome"]
```
</details>

(a)

![](images/a8316971c4428034d65f4cffe1087f17e5adeb344e4a216590fc8e792f49591c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["YMR105C"] --> B["Outcome"]
    C["YDR074W"] --> B
    D["YCL020W"] --> B
    E["YKL178C"] --> B
    F["YOL058W"] --> B
    G["YLR303W"] --> B
    H["YER073W"] --> B
    I["YCL030C"] --> B
    J["Red arrow"] --> B
    K["Blue arrow"] --> B
    L["Blue arrow"] --> B
    M["Blue arrow"] --> B
    N["Blue arrow"] --> B
```
</details>

(b)   
Figure E.1: Causal graphs for candidate genes that affect variant YER124C in yeast: (a). the estimated graph $\widehat{G}$ by NOTEARS (benchmark); (b). the estimated graph $\widehat{G}$ by NSCSL using TE.

# E.3 Additional Simulation Results: True and Estimated Matrix

![](images/7da19d3696c66e56bbe9d6f6aaef156c8ba964045a5eafbf8d84efc3a6425a1c.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|-------|
| 1 | 1 | 0.75  |
| 2 | 1 | 0.50  |
| 3 | 1 | -0.75 |
| 3 | 2 | -1.00 |
</details>

(a)

![](images/21baa2286083574cbae7e70a0479a55e7b59bb9a2dd9c3ee8abdead791a786f7.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
The color scale ranges from -1.00 (dark red) to 1.00 (dark blue), indicating the value in each cell corresponds to its corresponding position on the Y-Y plane. There is no explicit title or legend provided in the image.
</details>

(b)

![](images/231b30eb8595ff02af5132285f4836e54731a972e2f6901e5163d6454fbcc8a0.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | -1.00 | -0.75 | -0.50 | -0.25 | 0 |
| 1 | -0.75 | -0.50 | -0.25 | 0.00 | 1 |
| 2 | -0.50 | -0.25 | 0.25 | 0.50 | 2 |
| 3 | -0.25 | 0.00 | 0.50 | 0.75 | 3 |
| 4 | 0.00 | 0.25 | 0.50 | 1.00 | 4 |
</details>

(c)

![](images/94204c9f69408f3a4c12332efbc9bff198621329df4d7d3ad3cae68bda5de6bb.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 4 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 5 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 6 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 7 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 8 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 9 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 10 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 11 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 12 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 13 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 14 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 15 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 16 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 17 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 18 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 19 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 20 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 21 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 22 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 23 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 24 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 25 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| Note: The visual intensity values are estimated based on the color scale from -1.0 to +1.0, but the grid structure is not explicitly labeled in the image — it is a simplified representation of the data being plotted as a heatmap using the color bar (ranging from -1.0 to +1.0). The color legend indicates the value for each cell in the heatmap.
</details>

(d)

![](images/e77eb1609207bde2cef920ced139b9a7df6c99cb213838bc67d4de11bea3d369.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| D | 0.75 | 0.50 | 0.75 | 0.50 | 0.75 |
| 1 | 0.75 | 0.50 | 0.75 | 0.50 | 0.75 |
| 2 | 0.75 | 0.50 | 0.75 | 0.50 | 0.75 |
| 3 | 0.75 | 0.50 | 0.75 | 0.50 | 0.75 |
| Y | 0.75 | 0.50 | 0.75 | 0.50 | 0.75 |
</details>

(e)

![](images/55e09bc8276c4dfea8e0c65f7185041b3abf3251a0504e0265afd157704acdc3.jpg)  
(f)

![](images/67709774e251f112ceb7e3c0f0f1761d19eb633c7f5a4d3820276fc1177e3f6f.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| D | 0.75 | 0.50 | 0.75 | 1.00 | 0.75 |
| 1 | 0.50 | 0.25 | 0.75 | 1.00 | 0.50 |
| 2 | 0.25 | 0.00 | 0.75 | 1.00 | 0.25 |
| 3 | 0.00 | -0.25 | 0.75 | 1.00 | -0.25 |
| Y | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
The color scale ranges from -1.00 (dark blue) to +1.00 (dark red), indicating the value at each coordinate point in the grid. The chart is a single vertical bar for all four rows and columns, which are identical in height but differ in width as indicated by the label 'Y'.
</details>

(g)   
Figure E.2: Estimated matrix under S1 $(n=20)$ : (a). true whole graph; (b). true NSCG; (c). $\widehat{G}$ by NSCSL with TE; (d). $\widehat{G}$ by NSCSL with DE; (e). $\widehat{G}$ by NOTEARS; (f). $\widehat{G}$ by PC; (g). $\widehat{G}$ by LiNGAM.

![](images/83e8e4cb97b80d687079c2940eb5cff6cffa3f8605b05aa99b906ae53a2c57f3.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| Y | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 |
</details>

(a)

![](images/02d1b0731334b1ab73fd65d9323be4076b5b54fcbe8c4bc67039a3801d4fdbf3.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| Y | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
</details>

(b)

![](images/84b44c0611fbd398493dc930252b120124d965c7a0a2a4a3be7adcf4df8683c1.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| Y | 0.5 | 0.75 | 0.50 | 0.25 | 0.00 |
| 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'Y'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'X'x'
```
</details>

(c)

![](images/756377c2649747a4acbe50924329da2a593460b6457f23bd7d70bc6de85a67a2.jpg)  
(d)

![](images/2b262e5c6cc9a3d15132bef7a5f28bda5f534a2616537501ad98922ed3d23710.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.50 | 0.75 | 0.75 | 0.75 | 0.75 |
| 1 | 0.50 | 0.75 | 0.75 | 0.75 | 0.75 |
| 2 | 0.50 | 0.75 | 0.75 | 0.75 | 0.75 |
| 3 | 0.50 | 0.75 | 0.75 | 0.75 | 0.75 |
| Y | 0.50 | 0.75 | 0.75 | 0.75 | 0.75 |
</details>

(e)

![](images/89c13e835490b590d939b110e129b5c30de1f0482dca264359edfa7db753f00f.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| Y | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
</details>

(f)

![](images/e258137985d3f810501a6501cd0b1f7934cfcead95f9c351202bc0a6fc97f0c3.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 4 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 5 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 6 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 7 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 8 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 9 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 10 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 11 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 12 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 13 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 14 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 15 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 16 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 17 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 18 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 19 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 20 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 21 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 22 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 23 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 24 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 25 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| Note: The visual data is a heatmap with color intensity corresponding to the value scale on the right side of the heatmap, but no explicit numerical values are provided in the image.
</details>

(g)   
Figure E.3: Estimated matrix under S2 (n = 20): (a). true whole graph; (b). true NSCG; (c). $\widehat{G}$ by NSCSL with TE; (d). $\widehat{G}$ by NSCSL with DE; (e). $\widehat{G}$ by NOTEARS; (f). $\widehat{G}$ by PC; (g). $\widehat{G}$ by LiNGAM.

![](images/17b6fdf21756f4ac70109b6f9ea2cbe37eb55ec34db92879addac87600180020.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| Y | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
</details>

(a)

![](images/13ce9d96ea1c53396ac0fcc337b8447971e8947412f449081756c001345ae910.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| Y | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
</details>

(b)

![](images/28f9f5f0406cea0b96cb5873b84a34a8e74947143b42cb806d33a0f4c19f131e.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0 |
| Y | 1.00 | 1.00 | 1.00 | 1.00 | 1 |
</details>

(c)

![](images/9517f83b0919c20aec9af1856de656916f4b0826234abf21d21b96b10989603a.jpg)

<details>
<summary>heatmap</summary>

| Y\Y | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 |
</details>

(d)

![](images/669457006ece3c2960442fb2bd0dc1dac800545443790eab190e078d40b43ab9.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 1.00 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| ... | ... | ... | ... | ... | ... |
</details>

(e)

![](images/e3fdc363ae9be04e4e6818d8864b8aa2ac296b18242e39fcdffb1f7d410b1de6.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| Y | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
</details>

(f)

![](images/965803574dca3cab59990ec684eb304f9d4925c121bd42fed6d154b6127ea6f1.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | Y |
|---|---|---|---|---|---|
| 0 | 0.75 | 0.50 | 0.25 | 0.00 | 1.00 |
| 1 | 0.75 | 0.50 | 0.25 | 0.00 | 1.00 |
| 2 | 0.75 | 0.50 | 0.25 | 0.00 | 1.00 |
| 3 | 0.75 | 0.50 | 0.25 | 0.00 | 1.00 |
The color scale ranges from -1.00 (dark blue) to 1.00 (dark red). The chart displays a single vertical pattern of the cells, each labeled with its corresponding value on the right side of the grid.
</details>

(g)   
Figure E.4: Estimated matrix under S3 (n = 20): (a). true whole graph; (b). true NSCG; (c). $\widehat{G}$ by NSCSL with TE; (d). $\widehat{G}$ by NSCSL with DE; (e). $\widehat{G}$ by NOTEARS; (f). $\widehat{G}$ by PC; (g). $\widehat{G}$ by LiNGAM.

![](images/e8343bf4defd8ff76d80e09820123913b6ded4e2ad72e311a78baf422211cadc.jpg)  
(a)

![](images/d771c6e2073e19a604aced89849298e1dc4cf9c7086b1742d9b60c6d70bf4b9c.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|---|
| 1 | 0 | 0.00 |
| 2 | 1 | 0.00 |
| 3 | 2 | 0.00 |
| 4 | 3 | 0.00 |
| 5 | 4 | 0.00 |
| 6 | 5 | 0.00 |
| 7 | 6 | 0.00 |
| 8 | 7 | 0.00 |
| 9 | 8 | 0.00 |
| 10 | 9 | 0.00 |
| 11 | 10 | 0.00 |
| 12 | 11 | 0.00 |
| 13 | 12 | 0.00 |
| 14 | 13 | 0.00 |
| 15 | 14 | 0.00 |
| 16 | 15 | 0.00 |
| 17 | 16 | 0.00 |
| 18 | 17 | 0.00 |
| 19 | 18 | 0.00 |
The values in the table represent the magnitudes of the 'Value' column in each cell. There is no label for the data series.
</details>

(b)

![](images/d5730f870caaf21a798409b18944c8775f4cbb3a964ffbd0bd04e367d8725756.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | Y |
|---|---|---|---|---|---|---|---|---|---|---|----|----|----|----|----|----|----|----|----|---|
| Y | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 18 |
| Y | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | Y |
The color scale ranges from -1.00 (dark red) to 1.00 (dark blue). The chart displays a single data series with no explicit title or axis labels provided in the image.
</details>

(c)

![](images/83e15933c8914e80606add7bd129eb1adfd698a7f39233f05e9858d7d030d335.jpg)  
(d)

![](images/6af93aa201d7b5fb24f5c637261542c7073f546ba984015ade0d1f96a41c996d.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0 .75 | 0.75 | -1.00 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0  | 0   | 0   | -1.00 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0   | 0   | 0   | -1.00 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0    | 0   | 0   | -1.00 |
| 4 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0     | 0   | 0   | -1.00 |
| 5 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 1     | -1   | -1   | -1    |
| ... (repeated) for all rows and columns are integers between -1 and +1 in the heatmap; the values in the heatmap range from -1 to +1, but the color scale is based on the value in the heatmap cells, which is calculated based on the color scale and is derived from the visual data.
</details>

(e)

![](images/b04cf1d4d905085e082114903b720a15d10c5965626555404ccdfe470ff34f6d.jpg)  
(f)

![](images/630ce09d0310bd909fcaddecebbab065fe17ca3ae7124e921e8b04ee46325b77.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | Y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0 .75 | 0.75 | -1.00 |
| 1 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0  | 0   | 0   | -1.00 |
| 2 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0   | 0   | 0   | -1.00 |
| 3 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0    | 0   | 0   | -1.00 |
| 4 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0     | 0   | 0   | -1.00 |
| 5 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 1    | -1   | -1   | -1    |
| ... (additional rows) are not explicitly labeled in the image; the values in the table represent the magnitudes of the data points in each row and column of the image.
</details>

(g)   
Figure E.5: Estimated matrix under S4 (n = 100): (a). true whole graph; (b). true NSCG; (c). $\widehat{G}$ by NSCSL with TE; (d). $\widehat{G}$ by NSCSL with DE; (e). $\widehat{G}$ by NOTEARS; (f). $\widehat{G}$ by PC; (g). $\widehat{G}$ by LiNGAM.

![](images/d8832228f8ac36cee7b617afebaffd479e0d7e08e90a98fe4afd2f60efc003f2.jpg)  
(a)

![](images/ee31eb719ab8f506b3db29ccbad41610fec8054880bef9237df7deede783f9a2.jpg)  
(b)

![](images/08ec1ed9327ef53d430f2baa072eaf5c97c8cd9c0c0238db8b3f1293eb4d50f4.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|---|
| 0 | 0 | 0.00 |
| 1 | 0 | 1.00 |
| 2 | 0 | 0.75 |
| 3 | 0 | 0.50 |
| 4 | 0 | 0.25 |
| 5 | 0 | 0.00 |
| 6 | 0 | -0.25 |
| 7 | 0 | -0.50 |
| 8 | 0 | -0.75 |
| 9 | 0 | -1.00 |
| 10 | 0 | -0.75 |
| 11 | 0 | -0.50 |
| 12 | 0 | -0.25 |
| 13 | 0 | 0.00 |
| 14 | 0 | 0.25 |
| 15 | 0 | 0.50 |
| 16 | 0 | 0.75 |
| 17 | 0 | 1.00 |
| 18 | 0 | 0.75 |
| 19 | 0 | 0.50 |
| 20 | 0 | 0.25 |
| 21 | 0 | 0.00 |
| 22 | 0 | -0.25 |
| 23 | 0 | -0.50 |
| 24 | 0 | -0.75 |
| 25 | 0 | -1.00 |
The values in the table represent the magnitude of a single scalar variable at each coordinate point on the x-axis. There is no label for the data series.
</details>

(c)

![](images/433cb87c86c0eef7bfa81d1da6c391b94fd83ff9125d5cdd1f5795ee66e66d90.jpg)  
(d)

![](images/016e3a67be28e3ca11ac1b14c14a00acf91e3ce007ed718cbfcb30c38bbb7897.jpg)  
(e)

![](images/df9b7e56221617eb1a24e8a52627e9f567efa55b2932b8a5065ee99942205629.jpg)  
(f)

![](images/c82c19e75a45066b22a920380a8101ef3822d6d1eb35b016efe8d2ce7a93cb4d.jpg)  
(g)   
Figure E.6: Estimated matrix under S4 (n = 300): (a). true whole graph; (b). true NSCG; (c). $\widehat{G}$ by NSCSL with TE; (d). $\widehat{G}$ by NSCSL with DE; (e). $\widehat{G}$ by NOTEARS; (f). $\widehat{G}$ by PC; (g). $\widehat{G}$ by LiNGAM.

# E.4 Additional Simulation Results: True and Estimated Graphs

![](images/b2b144332f4f03be4fca5e392d877e67bfb2076b6acfd6e36c9af54c5512f63b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    2 --> 1
    2 --> 0
    2 --> 1
    3 --> 1
    3 --> 0
```
</details>

(a)

![](images/a29b4b0f889efb9b8b4346cfe92d5c900d3cb0a4889e96c45f9ddca55be56709.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    1 --> 2
    1 --> 0
    2 --> 3
    3 --> 0
```
</details>

(b)

![](images/e4ac6b588c31a7c063c6ad90bbd2f8d5bcd0de9d7a102730ea9eb6652e89706e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    1 --> 2
    1 --> 1
    2 --> 1
    3 --> 1
    3 --> 2
```
</details>

(c)

![](images/3e9d4762fc2075e0ddd80af04dd5ad8d69338ef5b8c9899d773f4fc73a0071d7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    1 --> 4
    2 --> 4
    3 --> 4
```
</details>

(d)

![](images/a7fbe82355185957b80a1966ab87ac29de27c08b268789dfd3315879bc56e17a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> 1
    0 --> 2
    0 --> 3
    0 --> 4
    0 --> 5
    0 --> 6
    0 --> 7
    0 --> 8
    1 --> 2
    1 --> 3
    1 --> 4
    1 --> 5
    1 --> 6
    1 --> 7
    2 --> 3
    2 --> 4
    2 --> 5
    2 --> 6
    2 --> 7
    3 --> 4
    3 --> 5
    3 --> 6
    3 --> 7
    4 --> 5
    4 --> 6
    4 --> 7
    5 --> 6
    5 --> 7
    6 --> 7
```
</details>

(e)

![](images/9154007f0367a749563050c660be13142b5dd281bef64e3581493dc3984ff984.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1"] --> B["2"]
    C["3"] --> D["7"]
    E["0"] --> F["4"]
```
</details>

(f)

![](images/1abb09a7da5454e4e556dadd26ac42493227c23dc9eafab8614b88cf353d7f90.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    2 --> 1
    2 --> 3
    2 --> 4
    2 --> 6
    3 --> 1
    3 --> 4
    3 --> 6
    4 --> 1
    4 --> 5
    5 --> 6
    5 --> 7
    6 --> 7
    6 --> 8
    7 --> 8
    8 --> 9
    9 --> 1
```
</details>

(g)   
Figure E.7: Graphs under S1 ( $n = 20$ ): (a). true whole graph; (b). true NSCG; (c). $\widehat{\mathcal{G}}$ by NSCSL with TE; (d). $\widehat{\mathcal{G}}$ by NSCSL with DE; (e). $\widehat{\mathcal{G}}$ by NOTEARS; (f). $\widehat{\mathcal{G}}$ by PC; (g). $\widehat{\mathcal{G}}$ by LiNGAM.

![](images/9b9316946288571976a2114cae61c85a22cf793817d33f37c0fba141c031383a.jpg)  
(e)   
Figure E.8: Graphs under S2 (n = 20): (a). true whole graph; (b). true NSCG; (c). $\widehat{G}$ by NSCSL with TE; (d). $\widehat{G}$ by NSCSL with DE; (e). $\widehat{G}$ by NOTEARS; (f). $\widehat{G}$ by PC; (g). $\widehat{G}$ by LiNGAM.

![](images/74e3d6ca743fdaa480cf8fb090fc04981aa0bdc97cc624441c4bf7f8ff6ef54e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> 1
    0 --> 2
    0 --> 3
    1 --> 2
    1 --> 3
    2 --> 3
    3 --> 0
    3 --> 1
```
</details>

(a)

![](images/498783357e8fac8f27b08558dae3773906493fd9aa6db16e88e46b81abc4ee98.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> y
    1 --> y
    2 --> y
    3 --> y
    y --> 0
    y --> 1
    y --> 2
    y --> 3
```
</details>

(b)

![](images/7a31759dd470d7f42fe4816592444c09c6e2c2948765267601291353c4653f02.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> 1
    0 --> 2
    0 --> 3
    1 --> 2
    1 --> 3
    2 --> 3
    3 --> 1
```
</details>

(c)

![](images/159a091445ea1c68ad81e68460afeaeb18239f3c0eb425b7c9e5b66ecc68492b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> Y
    2 --> Y
    3 --> Y
    1 --> Y
```
</details>

(d)

![](images/37829ce1de1a0e9b4aa5b155b9a2fdc4a2029258bd360fa77dd895f618133245.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    1 --> 2
    1 --> 3
    1 --> 4
    2 --> 3
    2 --> 4
    2 --> 5
    3 --> 4
    3 --> 5
    3 --> 6
    4 --> 5
    4 --> 6
    5 --> 6
```
</details>

(e)

![](images/21cc24d5fedd0dc5bd3c132db80997d70d803e7a60d685b87a0515e6a965a79c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> 1
    0 --> 2
    0 --> 3
    0 --> 4
    1 --> 2
    2 --> 3
    3 --> 4
```
</details>

(f)

![](images/bcb4328465a4c1962df51738b9a52dd82f91251c96bc884aa44fbee71ef03e89.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> 1
    0 --> 2
    0 --> 3
    0 --> 4
    1 --> 2
    1 --> 3
    1 --> 4
    2 --> 3
    2 --> 4
    3 --> 4
```
</details>

(g)   
Figure E.9: Graphs under S3 (n = 20): (a). true whole graph; (b). true NSCG; (c). $\widehat{G}$ by NSCSL with TE; (d). $\widehat{G}$ by NSCSL with DE; (e). $\widehat{G}$ by NOTEARS; (f). $\widehat{G}$ by PC; (g). $\widehat{G}$ by LiNGAM.

![](images/548a60198553cfe23c02a4883a494a20cec2c258810fb35efde3c9444468fb14.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> 1
    0 --> 2
    0 --> 3
    0 --> 4
    0 --> 5
    0 --> 6
    0 --> 7
    0 --> 8
    0 --> 9
    0 --> 10
    0 --> 11
    0 --> 12
    0 --> 13
    0 --> 14
    0 --> 15
    0 --> 16
    0 --> 17
    0 --> 18
    1 --> 2
    1 --> 3
    1 --> 4
    1 --> 5
    1 --> 6
    1 --> 7
    1 --> 8
    1 --> 9
    1 --> 10
    1 --> 11
    1 --> 12
    1 --> 13
    1 --> 14
    1 --> 15
    1 --> 16
    1 --> 17
    1 --> 18
    2 --> 3
    2 --> 4
    2 --> 5
    2 --> 6
    2 --> 7
    2 --> 8
    2 --> 9
    2 --> 10
    2 --> 11
    2 --> 12
    2 --> 13
    2 --> 14
    2 --> 15
    2 --> 16
    2 --> 17
    3 --> 4
    3 --> 5
    3 --> 6
    3 --> 7
    3 --> 8
    3 --> 9
    3 --> 10
    3 --> 11
    3 --> 12
    3 --> 13
    3 --> 14
    3 --> 15
    3 --> 16
    3 --> 17
    4 --> N
    N --> M
    M --> N
    M --> N
    M --> N
    M --> N
    M --> N
    M --> N
    M --> N
    M --> N
    M --> N
    M --> N
    M --> N
    M --> N
```
</details>

(a)

![](images/93db0f010565ab220fb10109f4623abcfa2565f8e10f83a078fa1d8c0f279ce1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["0"] --> B["Y"]
    C["1"] --> B["Y"]
    D["2"] --> B["Y"]
    E["3"] --> B["Y"]
    F["4"] --> B["Y"]
    G["5"] --> B["Y"]
    H["6"] --> B["Y"]
    I["7"] --> B["Y"]
    J["8"] --> B["Y"]
    K["9"] --> B["Y"]
    L["10"] --> B["Y"]
    M["11"] --> B["Y"]
    N["12"] --> B["Y"]
    O["13"] --> B["Y"]
    P["14"] --> B["Y"]
    Q["15"] --> B["Y"]
    R["16"] --> B["Y"]
    S["17"] --> B["Y"]
    T["18"] --> B["Y"]
```
</details>

(b)

![](images/707e6a4893d3dd54ad9da3358b84f2e49e746d585e63ce7e3a152c9e038e6bbb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    8 --> 3
    8 --> 2
    8 --> 1
    8 --> 0
    8 --> Y
    3 --> 1
    2 --> 0
    1 --> Y
    0 --> Y
    1 --> Y
    0 --> Y
    1 --> Y
    2 --> Y
    3 --> Y
    4 --> Y
    5 --> Y
    6 --> Y
    7 --> Y
    8 --> 9
    9 --> 10
    10 --> 11
    11 --> 12
    12 --> 13
    13 --> 14
    14 --> 15
    15 --> 16
    16 --> 17
    17 --> 18
```
</details>

(c)

![](images/0ada1d12dec26c59667223f753ecc6f3fd7d8e401c759a146de689c0cf0981d9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> Y
    1 --> Y
    2 --> Y
    3 --> Y
    4 --> Y
    5 --> Y
    6 --> Y
    7 --> Y
    8 --> Y
    9 --> Y
    10 --> Y
    11 --> Y
    12 --> Y
    13 --> Y
    14 --> Y
    15 --> Y
    16 --> Y
    17 --> Y
    18 --> Y
```
</details>

(d)

![](images/cb403af06cbdacc2051235e25977226959709a78642c67fa7beb56fc676a8785.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1"] --> B["2"]
    A --> C["3"]
    A --> D["4"]
    A --> E["5"]
    A --> F["6"]
    A --> G["7"]
    A --> H["8"]
    A --> I["9"]
    A --> J["10"]
    A --> K["11"]
    A --> L["12"]
    A --> M["13"]
    A --> N["14"]
    A --> O["15"]
    A --> P["16"]
    A --> Q["17"]
    A --> R["18"]
    A --> S["19"]
    A --> T["20"]
    B --> U["3"]
    C --> V["4"]
    D --> W["5"]
    E --> X["6"]
    F --> Y["7"]
    G --> Z["8"]
    H --> AA["9"]
    I --> AB["10"]
    J --> AC["11"]
    K --> AD["12"]
    L --> AE["13"]
    M --> AF["14"]
    N --> AG["15"]
    O --> AH["16"]
    P --> AI["17"]
    Q --> AJ["18"]
    R --> AK["19"]
    S --> AL["20"]
```
</details>

(e)

![](images/d5d8747c1c20d5b571f1353077318e74a920306babd19993bfa2dbbda4479ee1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["6"] --> B["7"]
    A --> C["8"]
    A --> D["9"]
    A --> E["10"]
    A --> F["11"]
    A --> G["12"]
    A --> H["13"]
    A --> I["14"]
    A --> J["15"]
    A --> K["16"]
    A --> L["17"]
    A --> M["18"]
    A --> N["19"]
    A --> O["20"]
    A --> P["21"]
    A --> Q["22"]
    A --> R["23"]
    A --> S["24"]
    A --> T["25"]
    A --> U["26"]
    A --> V["27"]
    A --> W["28"]
    A --> X["29"]
    A --> Y["30"]
    A --> Z["31"]
    A --> AA["32"]
    A --> AB["33"]
    A --> AC["34"]
    A --> AD["35"]
    A --> AE["36"]
    A --> AF["37"]
    A --> AG["38"]
    A --> AH["39"]
    A --> AI["40"]
```
</details>

(f)

![](images/5fce3b181971dc2e1f8e23d00c0f7c6d920a2efa161980a56190ce56e89c9456.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Node 1"] --> B["Node 2"]
    A --> C["Node 3"]
    A --> D["Node 4"]
    A --> E["Node 5"]
    A --> F["Node 6"]
    A --> G["Node 7"]
    A --> H["Node 8"]
    A --> I["Node 9"]
    A --> J["Node 10"]
    B --> K["Node 11"]
    B --> L["Node 12"]
    B --> M["Node 13"]
    B --> N["Node 14"]
    B --> O["Node 15"]
    C --> P["Node 16"]
    C --> Q["Node 17"]
    C --> R["Node 18"]
    C --> S["Node 19"]
    D --> T["Node 20"]
    D --> U["Node 21"]
    D --> V["Node 22"]
    E --> W["Node 23"]
    E --> X["Node 24"]
    E --> Y["Node 25"]
    F --> Z["Node 26"]
    F --> AA["Node 27"]
    F --> AB["Node 28"]
    G --> AC["Node 29"]
    G --> AD["Node 30"]
    G --> AE["Node 31"]
    H --> AF["Node 32"]
    H --> AG["Node 33"]
    H --> AH["Node 34"]
    I --> AI["Node 35"]
    I --> AJ["Node 36"]
    I --> AK["Node 37"]
    J --> AL["Node 38"]
    J --> AM["Node 39"]
    J --> AN["Node 40"]
```
</details>

(g)   
Figure E.10: Graphs under S4 ( $n = 100$ ): (a). true whole graph; (b). true NSCG; (c). $\widehat{\mathcal{G}}$ by NSCSL with TE; (d). $\widehat{\mathcal{G}}$ by NSCSL with DE; (e). $\widehat{\mathcal{G}}$ by NOTEARS; (f). $\widehat{\mathcal{G}}$ by PC; (g). $\widehat{\mathcal{G}}$ by LiNGAM.

![](images/9870240a93032421ff6785efc308cf5f005739fe5409fdd845c1224073bb6e0c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    0 --> 1
    0 --> 2
    0 --> 3
    0 --> 4
    0 --> 5
    0 --> 6
    0 --> 7
    0 --> 8
    0 --> 9
    0 --> 10
    0 --> 11
    0 --> 12
    0 --> 13
    0 --> 14
    0 --> 15
    0 --> 16
    0 --> 17
    0 --> 18
    1 --> 2
    1 --> 3
    1 --> 4
    1 --> 5
    1 --> 6
    1 --> 7
    1 --> 8
    1 --> 9
    1 --> 10
    1 --> 11
    1 --> 12
    1 --> 13
    1 --> 14
    1 --> 15
    1 --> 16
    1 --> 17
    1 --> 18
    2 --> 3
    2 --> 4
    2 --> 5
    2 --> 6
    2 --> 7
    2 --> 8
    2 --> 9
    2 --> 10
    2 --> 11
    2 --> 12
    2 --> 13
    2 --> 14
    2 --> 15
    2 --> 16
    2 --> 17
    3 --> 4
    3 --> 5
    3 --> 6
    3 --> 7
    3 --> 8
    3 --> 9
    3 --> 10
    3 --> 11
    3 --> 12
    3 --> 13
    3 --> 14
    3 --> 15
    3 --> 16
    3 --> 17
    4 --> 5
    4 --> 6
    4 --> 7
    4 --> 8
    4 --> 9
    4 --> 10
    4 --> 11
    4 --> 12
    4 --> 13
    4 --> 14
    4 --> 15
    4 --> 16
    4 --> 17
    5 --> N
    N --> M
    M --> N
    M --> N
```
</details>

(a)

![](images/36c35caf8f29413c5c20610328eccfc0f176361e962ac9ff002b3bbc5c5dc1e9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["0"] --> B["Y"]
    C["1"] --> B["Y"]
    D["2"] --> B["Y"]
    E["3"] --> B["Y"]
    F["4"] --> B["Y"]
    G["5"] --> B["Y"]
    H["6"] --> B["Y"]
    I["7"] --> B["Y"]
    J["8"] --> B["Y"]
    K["9"] --> B["Y"]
    L["10"] --> B["Y"]
    M["11"] --> B["Y"]
    N["12"] --> B["Y"]
    O["13"] --> B["Y"]
    P["14"] --> B["Y"]
    Q["15"] --> B["Y"]
    R["16"] --> B["Y"]
    S["17"] --> B["Y"]
    T["18"] --> B["Y"]
```
</details>

(b)

![](images/b7f6662ad1b1e96729f536b54fbfb80d461cafded5bb87e1a87e314dc244f825.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    8 --> 1
    8 --> 2
    8 --> 3
    8 --> 4
    8 --> 5
    8 --> 6
    8 --> 7
    8 --> 8
    1 --> 0
    2 --> Y
    3 --> Y
    4 --> Y
    5 --> Y
    6 --> Y
    7 --> Y
    8 --> 9
    9 --> 10
    10 --> 11
    11 --> 12
    12 --> 13
    13 --> 14
    14 --> 15
    15 --> 16
    16 --> 17
    17 --> 18
```
</details>

(c)

![](images/a94da8f62e7fed6bc19568061e6ad568e84f965472469e323bb679d893a0ddda.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["8"] --> B["Y"]
    C["9"] --> B
    D["10"] --> B
    E["11"] --> B
    F["12"] --> B
    G["13"] --> B
    H["14"] --> B
    I["15"] --> B
    J["16"] --> B
    K["17"] --> B
    L["18"] --> B
    M["2"] --> N["1"]
    O["3"] --> N
    P["4"] --> N
    Q["5"] --> N
    R["6"] --> N
    S["7"] --> N
    T["8"] --> U["1"]
```
</details>

(d)

![](images/865420d569a4ee2559eff92a6702a412ba27ec3ff56afad64cabb0b6dc594d14.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1"] --> B["2"]
    A --> C["3"]
    A --> D["4"]
    A --> E["5"]
    A --> F["6"]
    A --> G["7"]
    A --> H["8"]
    A --> I["9"]
    A --> J["10"]
    A --> K["11"]
    A --> L["12"]
    A --> M["13"]
    A --> N["14"]
    A --> O["15"]
    A --> P["16"]
    A --> Q["17"]
    A --> R["18"]
    A --> S["19"]
    A --> T["20"]
    A --> U["21"]
    A --> V["22"]
    A --> W["23"]
    A --> X["24"]
    A --> Y["25"]
    A --> Z["26"]
    A --> AA["27"]
    A --> AB["28"]
    A --> AC["29"]
    A --> AD["30"]
    A --> AE["31"]
    A --> AF["32"]
    A --> AG["33"]
    A --> AH["34"]
    A --> AI["35"]
    A --> AJ["36"]
    A --> AK["37"]
    A --> AL["38"]
    A --> AM["39"]
    A --> AN["40"]
    A --> AO["41"]
    A --> AP["42"]
    A --> AQ["43"]
    A --> AR["44"]
    A --> AS["45"]
    A --> AT["46"]
    A --> AU["47"]
    A --> AV["48"]
    A --> AW["49"]
    A --> AX["50"]
```
</details>

(e)

![](images/5751edda211970871115379873fa4e5ed200097e5c100ca6ff1fa9f46ca99d46.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1"] --> B["2"]
    A --> C["3"]
    A --> D["4"]
    A --> E["5"]
    A --> F["6"]
    A --> G["7"]
    A --> H["8"]
    A --> I["9"]
    A --> J["10"]
    A --> K["11"]
    A --> L["12"]
    A --> M["13"]
    A --> N["14"]
    A --> O["15"]
    A --> P["16"]
    A --> Q["17"]
    A --> R["18"]
    A --> S["19"]
    A --> T["20"]
    B --> U["3"]
    C --> V["4"]
    D --> W["5"]
    E --> X["6"]
    F --> Y["7"]
    G --> Z["8"]
    H --> AA["9"]
    I --> AB["10"]
    J --> AC["11"]
    K --> AD["12"]
    L --> AE["13"]
    M --> AF["14"]
    N --> AG["15"]
    O --> AH["16"]
    P --> AI["17"]
    Q --> AJ["18"]
    R --> AK["19"]
    S --> AL["20"]
```
</details>

(f)

![](images/81b0f84fe92ed3892b2d48a208cf6e2ee84030d39095488e7d9122301cc01773.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1"] --> B["2"]
    A --> C["3"]
    A --> D["4"]
    A --> E["5"]
    A --> F["6"]
    A --> G["7"]
    A --> H["8"]
    A --> I["9"]
    A --> J["10"]
    B --> K["11"]
    B --> L["12"]
    B --> M["13"]
    B --> N["14"]
    B --> O["15"]
    B --> P["16"]
    B --> Q["17"]
    B --> R["18"]
    C --> S["19"]
    C --> T["20"]
    C --> U["21"]
    C --> V["22"]
    C --> W["23"]
    C --> X["24"]
    C --> Y["25"]
    D --> Z["26"]
    D --> AA["27"]
    D --> AB["28"]
    D --> AC["29"]
    D --> AD["30"]
    E --> AE["31"]
    E --> AF["32"]
    E --> AG["33"]
    E --> AH["34"]
    E --> AI["35"]
    E --> AJ["36"]
    F --> AK["37"]
    F --> AL["38"]
    F --> AM["39"]
    F --> AN["40"]
    F --> AO["41"]
    G --> AP["42"]
    G --> AQ["43"]
    G --> AR["44"]
    G --> AS["45"]
    G --> AT["46"]
    H --> AU["47"]
    H --> AV["48"]
    H --> AW["49"]
    I --> AX["50"]
    I --> AY["51"]
    I --> AZ["52"]
    J --> BA["53"]
    J --> BB["54"]
    J --> BC["55"]
    K --> BD["56"]
    K --> BE["57"]
    K --> BF["58"]
    L --> BG["59"]
    L --> BH["60"]
    M --> BI["61"]
    M --> BJ["62"]
    N --> BK["63"]
    N --> BL["64"]
    O --> BM["65"]
    O --> BN["66"]
    P --> BO["67"]
    P --> BP["68"]
    Q --> BQ["69"]
    Q --> BR["70"]
    R --> BS["71"]
    R --> BT["72"]
    S --> BU["73"]
    S --> BV["74"]
    T --> BW["75"]
    T --> BX["76"]
    U --> BY["77"]
    U --> BZ["78"]
    V --> CA["79"]
    V --> CB["80"]
```
</details>

(g)   
Figure E.11: Graphs under S4 ( $n = 300$ ): (a). true whole graph; (b). true NSCG; (c). $\widehat{\mathcal{G}}$ by NSCSL with TE; (d). $\widehat{\mathcal{G}}$ by NSCSL with DE; (e). $\widehat{\mathcal{G}}$ by NOTEARS; (f). $\widehat{\mathcal{G}}$ by PC; (g). $\widehat{\mathcal{G}}$ by LiNGAM.

# References

[1] Aliferis, C. F., Statnikov, A., Tsamardinos, I., Mani, S., and Koutsoukos, X. D. Local causal and markov blanket induction for causal discovery and feature selection for classification part i: algorithms and empirical evaluation. Journal of Machine Learning Research, 11(1), 2010.   
[2] Brem, R. B. and Kruglyak, L. The landscape of genetic complexity across 5,700 gene expression traits in yeast. Proceedings of the National Academy of Sciences, 102(5):1572–1577, 2005.   
[3] Brzywczy, J. and Paszewski, A. Role of o-acetylhomoserine sulfhydrylase in sulfur amino acid synthesis in various yeasts. Yeast, 9(12):1335–1342, 1993.   
[4] Bühlmann, P., Peters, J., Ernest, J., et al. Cam: Causal additive models, high-dimensional order search and penalized regression. The Annals of Statistics, 42(6):2526–2556, 2014.   
[5] Cai, H., Song, R., and Lu, W. Anoce: Analysis of causal effects with multiple mediators via constrained structural learning. In International Conference on Learning Representations, 2020.   
[6] Chakrabortty, A., Nandy, P., and Li, H. Inference for individual mediation effects and interventional effects in sparse high-dimensional causal graphical models. arXiv preprint arXiv:1809.10652, 2018.   
[7] Chickering, D. M. Optimal structure identification with greedy search. Journal of machine learning research, 3(Nov):507–554, 2002.   
[8] Colman-Lerner, A., Chin, T. E., and Brent, R. Yeast cbk1 and mob2 activate daughter-specific genetic programs to induce asymmetric cell fates. Cell, 107(6):739–750, 2001.   
[9] de Melo, A. T., Martho, K. F., Roberto, T. N., Nishiduka, E. S., Machado Jr, J., Brustolini, O. J., Tashima, A. K., Vasconcelos, A. T., Vallim, M. A., and Pascon, R. C. The regulation of the sulfur amino acid biosynthetic pathway in cryptococcus neoformans: the relationship of cys3, calcineurin, and gpp2 phosphatases. Scientific Reports, 9(1):11923, 2019.   
[10] Feder, A., Keith, K. A., Manzoor, E., Pryzant, R., Sridhar, D., Wood-Doughty, Z., Eisenstein, J., Grimmer, J., Reichart, R., Roberts, M. E., et al. Causal inference in natural language processing: Estimation, prediction, interpretation and beyond. arXiv preprint arXiv:2109.00725, 2021.   
[11] Huang, B., Zhang, K., Lin, Y., Schölkopf, B., and Glymour, C. Generalized score functions for causal discovery. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1551–1560, 2018.   
[12] Janzing, D., Balduzzi, D., Grosse-Wentrup, M., and Schölkopf, B. Quantifying causal influences. 2013.   
[13] Janzing, D., Minorics, L., and Blöbaum, P. Feature relevance quantification in explainable ai: A causal problem. In International Conference on artificial intelligence and statistics, pp. 2907–2916. PMLR, 2020.   
[14] Kalisch, M. and Bühlmann, P. Estimating high-dimensional directed acyclic graphs with the pc-algorithm. Journal of Machine Learning Research, 8(Mar):613–636, 2007.   
[15] Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
[16] Kumar, V. and Minz, S. Feature selection: a literature review. SmartCR, 4(3):211–229, 2014.   
[17] Lachapelle, S., Brouillard, P., Deleu, T., and Lacoste-Julien, S. Gradient-based neural dag learning. arXiv preprint arXiv:1906.02226, 2019.   
[18] Lee, S. and Bareinboim, E. Structural causal bandits: Where to intervene? Advances in neural information processing systems, 31, 2018.   
[19] Lee, S. and Bareinboim, E. Characterizing optimal mixed policies: Where to intervene and what to observe. Advances in neural information processing systems, 33:8565–8576, 2020.

[20] Nandy, P., Maathuis, M. H., Richardson, T. S., et al. Estimating the effect of joint interventions from observational data in sparse high-dimensional settings. The Annals of Statistics, 45(2):647–674, 2017.   
[21] Niu, Y., Tang, K., Zhang, H., Lu, Z., Hua, X.-S., and Wen, J.-R. Counterfactual vqa: A cause-effect look at language bias. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12700–12710, 2021.   
[22] Pearl, J. et al. Models, reasoning and inference. Cambridge, UK: CambridgeUniversityPress, 19, 2000.   
[23] Pearl, J. et al. Causal inference in statistics: An overview. Statistics surveys, 3:96–146, 2009.   
[24] Peters, J. and Bühlmann, P. Identifiability of gaussian structural equation models with equal error variances. Biometrika, 101(1):219–228, 2014.   
[25] Peters, J., Mooij, J. M., Janzing, D., and Schölkopf, B. Causal discovery with continuous additive noise models. 2014.   
[26] Peters, J., Janzing, D., and Schölkopf, B. Elements of causal inference: foundations and learning algorithms. The MIT Press, 2017.   
[27] Ramsey, J., Glymour, M., Sanchez-Romero, R., and Glymour, C. A million variables and more: the fast greedy equivalence search algorithm for learning high-dimensional graphical causal models, with an application to functional magnetic resonance images. International journal of data science and analytics, 3(2):121–129, 2017.   
[28] Rolland, P., Cevher, V., Kleindessner, M., Russell, C., Janzing, D., Schölkopf, B., and Locatello, F. Score matching enables causal discovery of nonlinear additive noise models. In International Conference on Machine Learning, pp. 18741–18753. PMLR, 2022.   
[29] Rosenbaum, P. R. and Rubin, D. B. The central role of the propensity score in observational studies for causal effects. Biometrika, 70(1):41–55, 1983.   
[30] Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., and Nolan, G. P. Causal protein-signaling networks derived from multiparameter single-cell data. Science, 308(5721):523–529, 2005.   
[31] Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., and Bengio, Y. Toward causal representation learning. Proceedings of the IEEE, 109(5):612–634, 2021.   
[32] Shah, R. D. and Peters, J. The hardness of conditional independence testing and the generalised covariance measure. arXiv preprint arXiv:1804.07203, 2018.   
[33] Shi, C. and Li, L. Testing mediation effects using logic of boolean matrices. Journal of the American Statistical Association, pp. 1–14, 2021.   
[34] Shimizu, S., Hoyer, P. O., Hyvärinen, A., and Kerminen, A. A linear non-gaussian acyclic model for causal discovery. Journal of Machine Learning Research, 7(Oct):2003–2030, 2006.   
[35] Spirtes, P., Glymour, C., Scheines, R., Kauffman, S., Aimale, V., and Wimberly, F. Constructing bayesian network models of gene expression networks from microarray data. 2000.   
[36] Spirtes, P. L., Meek, C., and Richardson, T. S. Causal inference in the presence of latent variables and selection bias. arXiv preprint arXiv:1302.4983, 2013.   
[37] Takahashi, H., Braby, C. E., and Grossman, A. R. Sulfur economy and cell wall biosynthesis during sulfur limitation of chlamydomonas reinhardtii. Plant physiology, 127(2):665–673, 2001.   
[38] Tang, K., Huang, J., and Zhang, H. Long-tailed classification by keeping the good and removing the bad momentum causal effect. arXiv preprint arXiv:2009.12991, 2020.   
[39] Tian, J. and Pearl, J. Probabilities of causation: Bounds and identification. Annals of Mathematics and Artificial Intelligence, 28(1):287–313, 2000.

[40] Van de Geer, S. and Bühlmann, P. $\ell_0$ -penalized maximum likelihood for sparse directed acyclic graphs. 2013.   
[41] Vowels, M. J., Camgoz, N. C., and Bowden, R. D'ya like dags? a survey on structure learning and causal discovery. ACM Computing Surveys (CSUR), 2021.   
[42] Wang, Y. and Jordan, M. I. Desiderata for representation learning: A causal perspective. arXiv preprint arXiv:2109.03795, 2021.   
[43] Wright, S. Correlation and causation. Journal of agricultural research, 20(7):557–585, 1921.   
[44] Yu, Y., Chen, J., Gao, T., and Yu, M. Dag-gnn: Dag structure learning with graph neural networks. In International Conference on Machine Learning, pp. 7154–7163. PMLR, 2019.   
[45] Zhang, W., Wu, T., Wang, Y., Cai, Y., and Cai, H. Towards trustworthy explanation: On causal rationalization. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pp. 41715–41736. PMLR, 2023.   
[46] Zheng, X., Aragam, B., Ravikumar, P. K., and Xing, E. P. Dags with no tears: Continuous optimization for structure learning. In Advances in Neural Information Processing Systems, pp. 9472–9483, 2018.   
[47] Zheng, X., Dan, C., Aragam, B., Ravikumar, P., and Xing, E. Learning sparse nonparametric dags. In International Conference on Artificial Intelligence and Statistics, pp. 3414–3425. PMLR, 2020.   
[48] Zhu, S., Ng, I., and Chen, Z. Causal discovery with reinforcement learning. In International Conference on Learning Representations, 2019.