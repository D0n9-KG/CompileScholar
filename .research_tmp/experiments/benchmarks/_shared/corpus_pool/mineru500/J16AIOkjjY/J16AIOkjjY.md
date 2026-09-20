# Causal Abstraction Learning based on the Semantic Embedding Principle

Gabriele D'Acunto $^{12}$ Fabio Massimo Zennaro $^{3}$ Yorgos Felekis $^{4}$ Paolo Di Lorenzo $^{12}$

# Abstract

Structural causal models (SCMs) allow us to investigate complex systems at multiple levels of resolution. The causal abstraction (CA) framework formalizes the mapping between high- and low-level SCMs. We address CA learning in a challenging and realistic setting, where SCMs are inaccessible, interventional data is unavailable, and sample data is misaligned. A key principle of our framework is semantic embedding, formalized as the high-level distribution lying on a subspace of the low-level one. This principle naturally links linear CA to the geometry of the Stiefel manifold. We present a category-theoretic approach to SCMs that enables the learning of a CA by finding a morphism between the low- and high-level probability measures, adhering to the semantic embedding principle. Consequently, we formulate a general CA learning problem. As an application, we solve the latter problem for linear CA; considering Gaussian measures and the Kullback-Leibler divergence as an objective. Given the nonconvexity of the learning task, we develop three algorithms building upon existing paradigms for Riemannian optimization. We demonstrate that the proposed methods succeed on both synthetic and real-world brain data with different degrees of prior information about the structure of CA.

Code: https://github.com/SPAICOM/calsep

# 1. Introduction

Causal modeling and reasoning are key to trustworthy and responsible AI (Ganguly et al., 2023; Rawal et al., 2024; Qi et al., 2024). Structural causal models (SCMs) provide

$^{1}$ Department of Information Engineering, Electronics and Telecommunications, Sapienza University, Rome, Italy $^{2}$ National Inter-University Consortium for Telecommunications (CNIT), Parma, Italy $^{3}$ Department of Informatics, University of Bergen, Bergen, Norway $^{4}$ Department of Computer Science, University of Warwick, Coventry, UK. Correspondence to: Gabriele D'Acunto <gabriele.dacunto@uniroma1.it>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

# The Semantic Embedding Principle (SEP)

Causal Abstractions must preserve high-level causal knowledge when embedded in the low-level.

![](images/754a31e4d5e244ad0483e5da5d2db2287516fd84eb457b5c46552eae3d7cfa1e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["St(ℓ, h)"] -->|V| B["φ#^Vᵀ(χʰ)"]
    A -->|V| C["φ#^V∘Vᵀ(χʰ)"]
    B -->|χℓ| D["Rˡ"]
    C -->|Id_χh| E["Rʰ"]
    D -->|Vᵀ ∈ Rʰ×ℓ| F["V"]
```
</details>

Figure 1: Pictorial representation of SEP for linear CA. A linear map V belonging to the Stiefel manifold embeds a high-level causal knowledge $\chi^{h}$ into a low-level one, viz. $\chi^{\ell}$ , identifying an embedded causal knowledge $\varphi_{\#}^{\mathbf{V}^{\top}}(\chi^{h})$ . Then, a linear CA $V^{\top}$ abstracts $\varphi_{\#}^{\mathbf{V}^{\top}}(\chi^{h})$ , yielding a causal knowledge identical to $\chi^{h}$ . Notice that the arrow $Id_{\chi^{h}}$ underlines that commutativity holds only in one direction, that is, SEP does not imply $\varphi^{\mathbf{V}^{\top}\circ\mathbf{V}}(\chi^{\ell})=\chi^{\ell}$ .

a widely adopted framework for causal reasoning (Pearl, 2009). While canonical causal theory focuses on a single SCM, scientific research often requires multiple representations of the same system at different levels of resolution. For example, biological processes can be studied at the molecular level (e.g., gene expression), cellular level (e.g., metabolic pathways), or organism level (e.g., physiological responses), each offering a different view of the same underlying system. Causal abstraction (CA) theory (Rubenstein et al., 2017; Beckers & Halpern, 2019) formalizes mappings between SCMs at different abstraction levels, enforcing rigorous consistency requirements. This makes CA a powerful tool for transitioning between resolutions, synthesizing causal evidence, and selecting the most parsimonious representation for a given task. However, CAs are unknown in practice, underscoring the need for advancing CA learning from data (Zennaro et al., 2023).

Related works. Seminal works on CA have focused on defining and assessing given CA maps (Rubenstein et al., 2017; Beckers & Halpern, 2019). Our approach builds on the $\alpha$ -abstraction category-theoretic framework introduced by (Rischel, 2020), which neatly separates the structural and functional components of the CA. From a learning perspective, several methods have been proposed which rely on restrictive assumptions. In our work, we transform them into non-assumptions (NA). (Zennaro et al., 2023) addresses the learning problem under (NA1) complete specification of SCMs, which, in reality, is rarely available. (Felekis et al., 2024) assumes (NA2) knowledge of causal DAGs, which are often unknown in many applications. (Dyer et al., 2024) relies on the (NA3) availability of interventional data, which may be infeasible or unethical to obtain. (Kekić et al., 2024; Massidda et al., 2024) make (NA4) functional assumptions on the SCMs, such as linearity. (Massidda et al., 2024) implicitly assumes (NA5) alignment between data generated by two models, which requires tight coordination in sample collection. Conversely, we work under the realistic and pragmatic assumption that (A1) at least partial prior knowledge of the structure of a CA is available. (A1) is met in different application domains, such as neuroscience. For instance, consider the learning of a CA between two brain SCMs, the first referring to some brain region of interest (ROIs), the second to the brain lobes. A map between ROIs and brain lobes is implicitly defined by the location of ROIs, and so it would be natural to try to exploit such prior knowledge when learning the CA. Finally, we build on top of different continuous optimization frameworks, working in both the Euclidean and Riemannian spaces. Specifically, when dealing with a nonsmooth Riemannian problem, we leverage the manifold alternating direction method of multipliers (MADMM, Kovnatsky et al., 2016) and the manifold proximal gradient (ManPG, Chen et al., 2020). They are the Riemannian counterparts of the ADMM (Boyd et al., 2011) and PG (Parikh et al., 2014). Additionally, when dealing with a smooth, constrained, Riemannian problem, our solution combines the splitting of ortogonality constraints (SOC, Lai & Osher, 2014), the ADMM, and the successive convex approximation (SCA, Nedić et al., 2018) methods.

Related to CA learning are mechanistic interpretability via causal abstraction (Bereska & Gavves, 2024) and causal representation learning (Schölkopf et al., 2021). Regarding the former, Geiger et al. (2021; 2024) use CA to explain DNNs by treating the DNN as a low-level — to be read as black-box — SCM and aligning it via interchange intervention training (IIT) with a high-level — to be read as human-understandable — SCM, built from theoretical and empirical modeling work. This setting where two SCMs share structure but differ only in semantics would not constitute valid macro abstraction under our framework, nor can IIT objectives be directly adapted due to (NA1)-(NA3). Regarding the latter, CRL treats low-level variables (e.g. pixels) as noncausal observations generated by latent high-level causal concepts, and aims to recover those concepts and their causal graph to improve interpretability and performance robustness (Schölkopf et al., 2021; Yang et al., 2021; Komanduri et al., 2022). This goal also aligns with recent developments that integrate causal reasoning into communications, leveraging a functorial formalism as well (Thomas et al., 2023; Thomas & Saad, 2023; 2024). By contrast, CA learning focuses on mappings between SCMs, where the low- and high-level variables are causal and known, to enable causal knowledge transfer across abstraction levels.

Contributions. First, we introduce the semantic embedding principle (SEP) for CA, informally stating that in a well-behaved CA, embedding the high-level (coarser) causal knowledge into the low-level (finer) one and then abstracting it back enables perfect reconstruction of the high-level causal knowledge. Second, to formalize SEP categorically, we present an alternative category-theoretic framework for CA, which allows us to focus on the semantic layer of an SCM. Third, we formulate a general CA learning problem based on SEP and (A1). Fourth, we tackle the linear CA case, showing that SEP naturally links the linear CA to the geometry of the Stiefel manifold, shaping the learning process as a Riemannian optimization problem. As an application, we consider the Gaussian setting with the Kullback-Liebler (KL) divergence as a measure of alignment between the low- and high-level SCMs. Fifth, we formalize and solve nonsmooth and smooth learning problems for linear CAs in this setting. For the former, we present the LinSEPAL-ADMM and LinSEPAL-PG methods; for the latter, the CLinSEPAL one. Our experiments on synthetic and brain data, across different levels of prior knowledge, confirm good performance of the proposed methods.

Our work is a first step to bridging the gap between CA learning methods and real-world applications.

# 2. Background on Causality and Abstraction

This section provides the notation and key concepts related to causal modeling and abstraction theory.

Notation. The set of integers from 1 to $n$ is $[n]$ . The vectors of zeros and ones of size $n$ are $\mathbf{0}_n$ and $\mathbf{1}_n$ . The identity matrix of size $n \times n$ is $\mathbf{I}_n$ . The Frobenius norm is $\|\mathbf{A}\|_{\mathrm{F}}$ . The set of positive definite matrices over $\mathbb{R}^{n \times n}$ is $\mathcal{S}_{++}^n$ . The Hadamard product is $\odot$ . Function composition is $\circ$ . The domain of a function is $\mathbb{D}[\cdot]$ and its kernel ker. Let $\mathcal{M}(\mathcal{X}^n)$ be the set of Borel measures over $\mathcal{X}^n \subseteq \mathbb{R}^n$ . Given a measure $\mu^n \in \mathcal{M}(\mathcal{X}^n)$ and a measurable map $\varphi^{\mathbf{V}}, \mathcal{X}^n \ni \mathbf{x} \xmapsto \mathbf{V}^\top \mathbf{x} \in \mathcal{X}^m$ , we denote by $\varphi_{\#}^{\mathbf{V}}(\mu^n) := \mu^n (\varphi^{\mathbf{V}^{-1}}(\mathbf{x}))$ the pushforward measure $\mu^m \in \mathcal{M}(\mathcal{X}^m)$ .

We now present the standard definition of SCM.

Definition 2.1 (SCM, Pearl, 2009). A (Markovian) structural causal model (SCM) $\mathsf{M}^n$ is a tuple $\langle \mathcal{X},\mathcal{Z},\mathcal{F},\zeta^{\mathcal{Z}}\rangle$ , where $(i)\mathcal{X} = \{X_1,\dots ,X_n\}$ is a set of $n$ endogenous random variables; (ii) $\mathcal{Z} = \{Z_1,\dots,Z_n\}$ is a set of $n$ exogenous variables; (iii) $\mathcal{F}$ is a set of $n$ functional assignments such that $X_{i} = f_{i}(\mathcal{P}_{i},Z_{i}),\forall i\in [n]$ , with $\mathcal{P}_i\subseteq \mathcal{X}\setminus \{X_i\}$ ; (iv) $\zeta^{\mathcal{Z}}$ is a product probability measure over independent exogenous variables $\zeta^{\mathcal{Z}} = \prod_{i\in [n]}\zeta^{i}$ , where $\zeta^i = P(Z_i)$ .

A Markovian SCM induces a directed acyclic graph (DAG) $G_{M^{n}}$ where the nodes represent the variables X and the edges are determined by the structural functions F; $P_{i}$ constitutes then the parent set for $X_{i}$ . Furthermore, we can recursively rewrite the set of structural function F as a set of mixing functions M dependent only on the exogenous variables (cf. App. C). A key feature for studying causality is the possibility of defining interventions on the model:

Definition 2.2 (Hard intervention, Pearl, 2009). Given SCM $\mathsf{M}^n = \langle \mathcal{X},\mathcal{Z},\mathcal{F},\zeta^{\mathcal{Z}}\rangle$ , a (hard) intervention $\iota = \mathrm{do}(\mathcal{X}^{\iota} = \mathbf{x}^{\iota})$ , $\mathcal{X}^{\iota}\subseteq \mathcal{X}$ , is an operator that generates a new post-intervention SCM $\mathsf{M}_{\iota}^{n} = \langle \mathcal{X},\mathcal{Z},\mathcal{F}_{\iota},\zeta^{\mathcal{Z}}\rangle$ by replacing each function $f_{i}$ for $X_{i}\in \mathcal{X}^{\iota}$ with the constant $x_{i}^{\iota}\in \mathbf{x}^{\iota}$ . Graphically, an intervention mutilates $\mathcal{G}_{\mathsf{M}^n}$ by removing all the incoming edges of the variables in $\mathcal{X}^{\iota}$ .

Given multiple SCMs describing the same system at different levels of granularity, CA provides the definition of an $\alpha$ -abstraction map to relate these SCMs:

Definition 2.3 ( $\alpha$ -abstraction, Rischel, 2020). Given low-level $\mathsf{M}^{\ell}$ and high-level $\mathsf{M}^{h}$ SCMs, an $\alpha$ -abstraction is a triple $\alpha = \langle \mathcal{R}, m, \alpha \rangle$ , where (i) $\mathcal{R} \subseteq \mathcal{X}^{\ell}$ is a subset of relevant variables in $\mathsf{M}^{\ell}$ ; (ii) $m: \mathcal{R} \to \mathcal{X}^{h}$ is a surjective function between the relevant variables of $\mathsf{M}^{\ell}$ and the endogenous variables of $\mathsf{M}^{h}$ ; (iii) $\alpha: \mathbb{D}[\mathcal{R}] \to \mathbb{D}[\mathcal{X}^{h}]$ is a modular function $\alpha = \bigotimes_{i \in [n]} \alpha_{X_i^h}$ made up by surjective functions $\alpha_{X_i^h}: \mathbb{D}[m^{-1}(X_i^h)] \to \mathbb{D}[X_i^h]$ from the outcome of low-level variables $m^{-1}(X_i^h) \in \mathcal{X}^{\ell}$ onto outcomes of the high-level variables $X_i^h \in \mathcal{X}^h$ .

Notice that an $\alpha$ -abstraction simultaneously maps variables via the function m and values through the function $\alpha$ . The definition itself does not place any constraint on these functions, although a common requirement in the literature is for the abstraction to satisfy interventional consistency (Rubenstein et al., 2017; Rischel, 2020; Beckers & Halpern, 2019). An important class of such well-behaved abstractions is constructive linear abstraction, for which the following properties hold. By constructivity, (i) $\alpha$ is interventionally consistent; (ii) all low-level variables are relevant $R = X^{\ell}$ ; (iii) in addition to the map $\alpha$ between endogenous variables, there exists a map $\alpha_{U}$ between exogenous variables satisfying interventional consistency (Beckers & Halpern, 2019; Schooltink & Zennaro, 2024). By linearity, $\alpha = \mathbf{V}^{\top}\in \mathbb{R}^{h\times \ell}$ (Massidda et al., 2024). App. C provides formal definitions for interventional consistency, linear and constructive abstraction.

![](images/2194c811d967d3a5297c11d8e0591fe8cd9223176efa493157bc04a8823c6166.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Ind"] --> B["(U^ℓ, Σ_U^ℓ, ζ^ℓ)"]
    A --> C["(V^ℓ, Σ_V^ℓ, ·^ℓ)"]
    B --> D["(U^h, Σ_U^h, ζ^h)"]
    C --> E["(V^h, Σ_V^h, ·^h)"]
    D --> F["(U^h, Σ_U^h, ζ^h)"]
    E --> G["(V^h, Σ_V^h, ·^h)"]
    F --> H["M^h"]
    G --> I["M^l"]
    H --> J["α(U^h, Σ_U^h)"]
    I --> K["α(V^h, Σ_V^h)"]
    J --> L["(U^ℓ, Σ_U^ℓ, ζ^ℓ)"]
    K --> M["(V^ℓ, Σ_V^ℓ, ·^ℓ)"]
    L --> N["α(U^h, Σ_U^h)"]
    M --> O["α(V^h, Σ_V^h)"]
```
</details>

Figure 2: An abstraction as natural transformation, that is, a set of commuting arrows in Prob (dashed black) from $M^{\ell}$ (purple) to $M^{h}$ (cyan).

# 3. Category-theory Formalization

Standard category-theoretic formalization of CA (Rischel, 2020; Otsuka & Saigo, 2022) are based on a functorial semantics (Jacobs et al., 2019) approach mapping the graphical structure of causal models (syntax) onto the discrete distributions of individual variables (semantics). Because of our non-assumption (NA2), no knowledge of the structure of an SCM is available in our setting; thus, we propose a formalization mapping a dyadic structure (syntax) onto the exogenous and the endogenous probability measures implied by an SCM (semantics).

A crucial role in our modelling is that of the mixing functions M, which express the data generation process as a recursive process from the exogenous functions. This allows us to define an SCM $M^{n}$ in measure-theoretic terms as a tuple made up of the probability space of exogenous variables $(\mathcal{U}, \Sigma_{\mathcal{U}}, \zeta)$ , the probability space of the endogenous variables $(\mathcal{V}, \Sigma_{\mathcal{V}}, \chi)$ , and a set of measurable functions M given by the mixing functions (cf. App. C).

We can now rely on this representation to interpret an SCM as a category-theoretic functor from a simple index category $\mathrm{Ind}$ , made up only of a source and a sink object and an edge between them, to the category of probability spaces Prob, where objects $(X, \Sigma_X, p)$ are probability spaces and morphisms $\varphi$ are measurable maps:

Definition 3.1 (Category-theoretic SCM). An SCM is a functor $M^{n}: \operatorname{Ind} \to \operatorname{Prob}$ , mapping the source node of $\operatorname{Ind}$ to $(\mathcal{U}, \Sigma_{\mathcal{U}}, \zeta)$ , the sink node of $\operatorname{Ind}$ to $(\mathcal{V}, \Sigma_{\mathcal{V}}, \chi)$ , and the edge of $\operatorname{Ind}$ to the collection M of measurable maps.

App. B presents basic category-theoretic concepts, whereas App. C.5 deepens Def. 3.1. CA can now be expressed as

a natural transformation between two SCMs, as shown in Fig. 2. This formulation has two important features. First, it highlights the role of exogenous variables in a constructive abstraction showing the commutativity of the paths $\mathcal{M}^h\circ \alpha_{\left(\mathcal{U}^h,\Sigma_{\mathcal{U}^h}\right)}$ and $\alpha (\nu^{h},\Sigma_{\nu^{h}})\circ \mathcal{M}^{\ell}$ . Second, morphisms in Prob relates measure spaces, viz. sets equipped with sigma algebras. Consequently, the natural transformation components are measurable maps with dimensionality determined by the cardinality of $\mathcal{X}^h$ and $\mathcal{X}^{\ell}$ . To ease the notation, we will denote $\alpha_{\left(\mathcal{U}^h,\Sigma_{\mathcal{U}^h}\right)}$ by $\alpha_{\mathcal{Z}}$ and $\alpha_{\left(\nu^{h},\Sigma_{\nu^{h}}\right)}$ by $\alpha_{\mathcal{X}}$ . Then, we can formally recast the $\alpha$ -abstraction in Prob.

Definition 3.2 ( $\alpha$ -abstraction in Prob). Given low-level $\mathsf{M}^{\ell}$ and high-level $\mathsf{M}^{h}$ SCMs, an abstraction $\boldsymbol{\alpha} = \langle \mathcal{R}, \mathcal{Q}, m, \alpha \rangle$ is a tuple, where: (i) $\mathcal{R}$ is the same as in Def. 2.3; (ii) $\mathcal{Q} \subseteq \mathcal{Z}^{\ell}$ is a set of relevant exogenous variables given by the union of the set of exogenous corresponding to the endogenous in $\mathcal{R}$ and those corresponding to their ancestors; (iii) $m = \langle m_{\mathcal{Z}}, m_{\mathcal{X}} \rangle$ is a pair of surjective functions mapping sets, $m_{\mathcal{Z}} : \mathcal{Q} \to \mathcal{Z}^{h}$ and $m_{\mathcal{X}} : \mathcal{R} \to \mathcal{X}^{h}$ , respectively; (iv) $\alpha = \langle \alpha_{\mathcal{Z}}, \alpha_{\mathcal{X}} \rangle$ is a natural transformation made by measurable functions mapping probability spaces, $\alpha_{\mathcal{Z}}$ for the exogenous and $\alpha_{\mathcal{X}}$ for the endogenous, respectively.

As Def. 2.3, Def. 3.2 makes no reference to interventional consistency. App. D explains how intervened SCMs and interventional consistency can be represented categorically.

# 4. Problem Formulation

Within our category-theoretic framework, CA learning amounts to finding the endogenous components $m_{\chi}$ and $\alpha_{\chi}$ from data. We start by formulating a general learning problem working under the non-assumption (NA1)-(NA5), and then decline it to the case of linear CA.

Our problem formulation relies upon three key ingredients. First, we assume that the data generated by a constructive abstraction adheres to the semantic embedding principle. This principle requires that the CA component $\alpha_{\mathcal{X}}$ admits a right-inverse measurable map.

Definition 4.1 (Semantic embedding principle, SEP). Given an $\alpha$ -abstraction as in Def. 3.2, the semantic embedding principle states that $\alpha_{\mathcal{X}}$ has a right-inverse measurable map $\beta_{\mathcal{X}}$ , such that $\alpha_{\mathcal{X}} \circ \beta_{\mathcal{X}} = \mathrm{Id}_{(\mathcal{V}^h, \Sigma_{\mathcal{V}^h}, \chi^h)}$ . Hence, it holds

$$
\chi^ {h} = \varphi_ {\#} ^ {\alpha_ {\mathcal {X}} \circ \beta_ {\mathcal {X}}} (\chi^ {h}). \tag {1}
$$

The SEP implies that going from the high-level model $M^{h}$ to the low-level model $M^{\ell}$ and then abstracting back to $M^{h}$ allows for perfect reconstruction. Notice that SEP only holds in one direction, as suggested by the word embedding; thus, identity on the left inverse is not guaranteed, meaning that the abstraction from the low level to the high level can still shed information, as we would expect in CA.

Second, because of the non-assumption (NA3) only observational data is available. Thus, we can not explicitly use interventional consistency information to drive our learning. Only if we identify the true constructive abstraction, we are guaranteed interventional consistency. In trying to learn the abstraction, we leverage (A1), which is met in application domains as discussed in Sec. 1.

Third, to learn a CA, we look for a distance function quantifying the misalignment between the probability measures $\chi^{\ell}$ and $\chi^{h}$ , given $\alpha_{X}$ . Since the probability measures belong to spaces of different dimensionality, specifically $R^{\ell}$ and $R^{h}$ , we leverage the approach proposed in (Cai & Lim, 2022) to compute the misalignment through an embedding as $D\left(\chi^{h}, \varphi_{\#}^{\alpha_{X}}(\chi^{\ell})\right)$ , where D is an information-theoretic metric (e.g., p-Wasserstein) or $\phi$ -divergence (e.g., Kullback-Leibler). Please refer to App. F for more details. We can now pose the following general learning problem:

# Problem 1. (SEP-based CA Learning)

$\textbf{Input: (i) probability measures } \chi^{\ell} \text { and } \chi^{h}; (ii) prior information about } m_{\mathcal{X}}, \text { and (iii) a distance function } D\left(\chi^{h}, \varphi_{\#}^{\alpha \mathcal{X}}(\chi^{\ell})\right).$

Goal: learn a measurable map $\alpha_{\mathcal{X}}^{\star}$ such that (i) it belongs to $\ker D\left(\chi^{h},\varphi_{\#}^{\alpha_{\mathcal{X}}}\left(\chi^{\ell}\right)\right)$ , (ii) it complies with SEP in Def. 4.1, and (iii) it agrees with the prior information about $m_{\mathcal{X}}$ .

The zeroing of the distance function implies $\chi^{h} = \varphi_{\#}^{\alpha_{\mathcal{X}}^{\star}}(\chi^{\ell})$ , which, together with Eq. (1), yields $\varphi_{\#}^{\alpha_{\mathcal{X}}^{\star}\circ\beta_{\mathcal{X}}}\left(\chi^{h}\right) = \varphi_{\#}^{\alpha_{\mathcal{X}}^{\star}}\left(\chi^{\ell}\right)$ . However, despite solving Prob. 1, there is no guarantee that $\alpha_{X}^{\star}$ coincides with the ground truth CA. In other words, the optimal solution is not unique. For a linear constructive CA, we express $m_{X}$ and $\alpha_{X}$ as $B^{\top} \in \{0,1\}^{h \times \ell}$ and $V^{\top} \in R^{h \times \ell}$ , respectively. In accordance with constructivity, each row of B has a single nonzero entry, and each column has at least one nonzero entry. Importantly, for linear CA, a simple yet principled way to satisfy SEP is via the geometry of the Stiefel manifold:

$$
\operatorname{St} (\ell , h) := \left\{\mathbf {V} \in \mathbb {R} ^ {\ell \times h} \mid \mathbf {V} ^ {\top} \mathbf {V} = \mathbf {I} _ {h} \right\}. \tag {2}
$$

The Stiefel manifold (see App. E for details), is a convenient choice for the following reasons: (i) differently from a generic pseudo-inverse matrix, the orthogonality of V guarantees that the geometry of the high-level space is preserved; (ii) the transpose eases the formulation and ensures numerical stability in optimization. Consequently, we restate SEP for the linear case as follows.

Definition 4.2 (Semantic embedding principle, linear case). Given the linear constructive CA, viz. $\mathbf{V}^{\top}$ , SEP implies that $\mathbf{V} \in \mathrm{St}(\ell, h)$ . From Eq. (1) we get $\chi^{h} = \varphi_{\#}^{\mathbf{V} \circ \mathbf{V}^{\top}}(\chi^{h})$ .

A pictorial representation of Def. 4.2 is provided in Fig. 1. Def. 4.2 shapes our methodology for CA learning, posing it as a Riemannian optimization problem (Boumal, 2023).

As an application, in the sequel, we will tackle an implementation of Prob. 1 for the linear constructive case $\alpha_{\mathcal{X}} = \mathbf{V}^{\top}$ , where (i) $\chi^{h} \sim N(\mathbf{0}_{h}, \boldsymbol{\Sigma}^{h})$ and $\chi^{\ell} \sim N(\mathbf{0}_{\ell}, \boldsymbol{\Sigma}^{\ell})$ ; and (ii) $D\left(\chi^{h}, \varphi_{\#}^{\mathbf{V}}(\chi^{\ell})\right) = D^{\mathrm{KL}}\left(\chi^{h}||\varphi_{\#}^{\mathbf{V}}(\chi^{\ell})\right)$ where $D^{\mathrm{KL}}$ stands for KL divergence. Specifically,

$$
D _ {\mathbf {V}} ^ {\mathrm{KL}} = \operatorname{Tr} \left\{\left(\mathbf {V} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {V}\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} + \log \det \left\{\mathbf {V} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {V} \right\} + C, \tag {3}
$$

where $C = -\log \det\left\{\Sigma^{h}\right\} - h$ is a constant term. Additionally, from Eq. (3) it is immediate to see that both V and -V belong to ker $D_{V}^{KL}$ . Such an application is highly relevant as it is common to deal in practice with Gaussian measures (or quasi) (D'Acunto et al., 2024); also, in causality, such a measure easily arises from the prominent family of linear models (Bollen, 1989; Shimizu et al., 2006) and is investigated in the CA literature (Kekić et al., 2024; Massidda et al., 2024). KL divergence is a common choice in ML and statistics, but notice that any distance vanishes when evaluated at the ground truth.

Spectral properties entailed by SEP. In the setting of zero-mean Gaussian measures, all causal information is encoded in the covariance matrices of $\chi^{\ell}$ and $\chi^{h}$ , which may be geometrically represented as ellipsoids in $\mathbb{R}^{\ell}$ and $\mathbb{R}^{h}$ . Each covariance $\boldsymbol{\Sigma}$ admits an eigendecomposition $\boldsymbol{\Sigma} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\top}$ , where the columns of $\mathbf{U}$ specify the ellipsoid's principal axes and the square roots of the diagonal entries of $\boldsymbol{\Lambda}$ give the corresponding axis lengths. Thus, for the low-level distribution $N(\mathbf{0}_{\ell},\boldsymbol{\Sigma}^{\ell})$ , one obtains an $\ell$ -dimensional ellipsoid with axes $\mathbf{U}^{\ell}$ and lengths $\sqrt{\lambda_i}$ , $i\in [\ell]$ ; similarly, $N(\mathbf{0}_h,\boldsymbol{\Sigma}^h)$ defines an $h$ -dimensional ellipsoid with axes $\mathbf{U}^h$ and lengths $\sqrt{\kappa_j}$ , $j\in [h]$ .

When projecting the low-level ellipsoid into an $h$ -dimensional subspace via an orthonormal map $\mathbf{V} \in \mathrm{St}(\ell, h)$ , the resulting covariance $\mathbf{V}^{\top} \boldsymbol{\Sigma}^{\ell} \mathbf{V} = \mathbf{V}^{\top} \mathbf{U}^{\ell} \boldsymbol{\Lambda} \mathbf{U}^{\ell^{\top}} \mathbf{V} = \mathbf{Q}^{\top} \boldsymbol{\Lambda} \mathbf{Q}$ , with $\mathbf{Q} = \mathbf{U}^{\ell^{\top}} \mathbf{V}$ , yields a projected ellipsoid whose axes are linear combinations of those in $\mathbf{U}^{\ell}$ . Because $\mathbf{V}$ is contractive by SEP, no projection can increase variance. Thus, the length of each axis of the projected ellipsoid lies between the minimum and maximum length of those of the $\ell$ -dimensional ellipsoid. Specifically, by the Ostrowski's theorem for rectangular $\mathbf{V}$ (cf. Th. 3.2 in Higham & Cheng, 1998), the $i$ -th largest axis length of the projected ellipsoid falls between the $i$ -th and $(i + \ell - h)$ -th largest lengths of the $\ell$ -dimensional ellipsoid. Consequently, for the optimal CA case where the projected ellipsoid exactly matches that of $\boldsymbol{\Sigma}^{h}$ , the projected axis align with $\mathbf{U}^{h}$ and the previous interlacing inequalities provide necessary spectral conditions for the existence of a linear CA from $N(\mathbf{0}_{\ell}, \boldsymbol{\Sigma}^{\ell})$ to $N(\mathbf{0}_{h}, \boldsymbol{\Sigma}^{h})$ .

Theorem 4.3. Let $\chi^{\ell} \sim N(\mathbf{0}_{\ell}, \boldsymbol{\Sigma}^{\ell})$ , $\chi^{h} \sim N(\mathbf{0}_{h}, \boldsymbol{\Sigma}^{h})$ , where $\boldsymbol{\Sigma}^{\ell} \in S_{++}^{\ell}$ and $\boldsymbol{\Sigma}^{h} \in S_{++}^{h}$ . Denote by $0 < \lambda_{1} \leq \ldots \leq \lambda_{\ell}$ the eigenvalues of $\boldsymbol{\Sigma}^{\ell}$ , and by $0 < \kappa_{1} \leq \ldots \leq \kappa_{h}$ those of $\boldsymbol{\Sigma}^{h}$ . If a linear CA $\mathbf{V} \in \mathrm{St}(\ell, h)$ complying with SEP from $\chi^{\ell}$ to $\chi^{h}$ exists, then

$$
\lambda_ {i} \leq \kappa_ {i} \leq \lambda_ {i + \ell - h}, \quad \forall i \in [ h ]. \tag {4}
$$

Proof. See App. G.

![](images/f86e29783d4fa1f95a16878e56efe468ce37c1def2a0f97e40af38e4cbe8c97d.jpg)

We now turn to formulating the learning problem. We investigate two approaches for injecting the prior information about $m_{X}$ , encoded in the matrix of prior knowledge B, into our problem. Please notice that in case B is not fully specified, it might not comply with the row and column constraints discussed above. These formulations translate into non-smooth and smooth Riemannian learning problems.

Nonsmooth problem. In the nonsmooth problem we introduce B as a penalty term in the objective function. The rationale is to penalize entries in V corresponding to zeros in B. Let $\mathbf{D} = (\mathbf{1}_{\ell \times h} - \mathbf{B})$ . The problem reads as follows:

Problem 2. Given $\Sigma^{\ell} \in S_{++}^{\ell}$ , $\Sigma^{h} \in S_{++}^{h}$ , $\mathbf{D} \in \{0,1\}^{\ell \times h}$ , and $\lambda \in \mathbb{R}_{+}$ , the CA is the transpose of

$$
\mathbf {V} ^ {\star} = \underset {\mathbf {V} \in \operatorname{St} (\ell , h)} {\arg \min} f (\mathbf {V}) + \lambda \underbrace {\left\| \mathbf {D} \odot \mathbf {V} \right\| _ {1}} _ {h (\mathbf {V})}. \tag {5}
$$

Here, $f(\mathbf{V})$ follows Eq. (3), omitting the constant $C$ .

Please notice that, although appealing in its form, Eq. (5) does not guarantee the constructiveness of the learned CA. Moreover, the penalty term introduces a bias in the learned V in the case of partial prior knowledge.

Smooth problem. In the smooth problem, we introduce $\mathbf{B}$ directly in the objective function $f(\cdot)$ . The CA is now defined as the Hadamard product of $\mathbf{V}$ and the support $(\mathbf{B} \odot \mathbf{S})$ integrating prior $\mathbf{B}$ and learned $\mathbf{S}$ knowledge. This formulation is particularly convenient as it enables us to jointly optimize for $\mathbf{V} \in \mathbb{R}^{\ell \times h}$ and matrix $\mathbf{S} \in [0,1]^{\ell \times h}$ . However, we also need to introduce three constraints:

(i) by SEP, $B \odot S \odot V$ must belong to the Stiefel manifold; (ii) by functionality, the columns of the support $(\mathbf{B} \odot \mathbf{S})^{\top}$ must sum up to one, meaning that they lie on a sphere, defined as

$$
\begin{array}{l} \operatorname{Sp} ^ {\Delta} (h, \ell) := \left\{\mathbf {A} \in \{0, 1 \} ^ {h \times \ell} \mid \| \mathbf {a} _ {j} \| _ {2} = 1 \text {   and   } \right. \\ \left. \sum_ {i = 1} ^ {h} a _ {i j} = 1, \forall j \in [ \ell ] \right\}; \tag {6} \\ \end{array}
$$

(iii) by surjectivity, the rows of the support $(\mathbf{B} \odot \mathbf{S})^{\top}$ must contain at least a one. The problem reads as:

Problem 3. Given $\Sigma^{\ell} \in S_{++}^{\ell}$ , $\Sigma^{h} \in S_{++}^{h}$ , and $\mathbf{B} \in \{0,1\}^{\ell \times h}$ , the linear constructive CA is given by the transpose of the product $\mathbf{B} \odot \mathbf{S} \odot \mathbf{V}$ , where

$$
\mathbf{V}^{\star},\mathbf{S}^{\star} = \operatorname *{arg  min}_{\substack{\mathbf{V}\in \mathbb{R}^{\ell \times h}\\ \mathbf{S}\in [0,1]^{\ell \times h}}}f(\mathbf{V},\mathbf{S});
$$

subject to (i) $\mathbf{B} \odot \mathbf{S} \odot \mathbf{V} \in \mathrm{St}(\ell, h)$ , (7)

(ii) $(\mathbf{B} \odot \mathbf{S})^{\top} \in \mathrm{Sp}^{\Delta}(h, \ell)$ ,

(iii) $\mathbf{1}_h - (\mathbf{B} \odot \mathbf{S})^\top \mathbf{1}_\ell \leq \mathbf{0}_h$ ;

and

$$
\begin{array}{l} f (\mathbf {V}, \mathbf {S}) := \operatorname{Tr} \left\{\left(\left(\mathbf {B} \odot \mathbf {S} \odot \mathbf {V}\right) ^ {\top} \boldsymbol {\Sigma} ^ {\ell} (\mathbf {B} \odot \mathbf {S} \odot \mathbf {V})\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} \\ + \log \det \left\{\left(\mathbf {B} \odot \mathbf {S} \odot \mathbf {V}\right) ^ {\top} \boldsymbol {\Sigma} ^ {\ell} (\mathbf {B} \odot \mathbf {S} \odot \mathbf {V}) \right\}. \tag {8} \\ \end{array}
$$

Constraints (ii) and (iii) further underscore the role of S: it enables learning the support of CA while guaranteeing its constructiveness. Notice that the matrix S does not need to be a logical matrix; it is the product B ⊙ S which must be logical. Also, if B provides full prior knowledge about the structure, we have S ≡ B and we do not need to learn S. This approach guarantees the ground-truth structure for the learned CA. The full prior problem formulation is provided in App. J.6.

Unfortunately, both the Stiefel manifold in Eq. (2) and $D_{V}^{KL}$ in Eq. (3) are nonconvex in V. In the next section we devise methods suitable for this setting.

Remark 1. Learning linear CAs remains challenging even with full prior structural knowledge and jointly sampled data (NA5). Although one could decompose the task into h independent linear regressions, enforcing SEP requires each coefficient vector to satisfy a unitary $\ell_{2}$ -norm constraint, rendering the problem nonconvex.

# 5. Problem Solution

To solve the nonsmooth and smooth Riemannian problems in Sec. 4, we leverage the following:

Proposition 5.1. Consider the function

$$
f (\mathbf {A}) = \operatorname{Tr} \left\{\left(\mathbf {A} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {A}\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} + \log \det \left\{\mathbf {A} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \right\}. \tag {9}
$$

Eq. (9) is smooth for $\mathbf{A} \in \mathrm{St}(\ell, h)$ . Additionally, define $\widetilde{\mathbf{A}} := (\mathbf{A}^{\top} \boldsymbol{\Sigma}^{\ell} \mathbf{A})^{-1}$ . The gradient of $f(\mathbf{A})$ is

$$
\nabla_ {\mathbf {A}} f = 2 \left(\boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \widetilde {\mathbf {A}}\right) \left(\mathbf {I} _ {h} - \boldsymbol {\Sigma} ^ {h} \widetilde {\mathbf {A}}\right), \tag {10}
$$

Proof. See App. G.

# 5.1. Solution of the nonsmooth learning problem

Leveraging Proposition 5.1, we have that Eq. (5) is constituted by a smooth yet nonconvex term, $f(\mathbf{V})$ , and a nonsmooth one, $h(\mathbf{V})$ . Hence we solve Prob. 2 through two different optimization paradigms for nonsmooth Riemannian optimization: MADMM and ManPG. We term the proposed methods LinSEPAL-ADMM and LinSEPAL-PG, where LinSEPAL stands for Linear Semantic Embedding Principle Abstraction Learner. Next we provide a sketch of the solution and provide the full mathematical derivation in App. H and App. I.

LinSEPAL-ADMM. The MADMM framework appeals to our setting given the objective function separating into smooth and nonsmooth terms. To derive the LinSEPAL-ADMM iterative algorithm, we proceed as follows. First, the nonsmooth term $h(\mathbf{V})$ is associated with a splitting variable Y to be optimized over $R^{\ell \times h}$ , obtaining an equivalent problem formulation (cf. Eq. (P2)). LinSEPAL-ADMM proceeds by iteratively minimizing the augmented Lagrangian with respect to the primal variables V and Y, while maximizing w.r.t. the scaled dual variable. Specifically, LinSEPAL-ADMM solves the subproblem for V (cf. Eq. (36)) through standard techniques for smooth optimization on the Stiefel manifold (e.g., conjugate gradient, Edelman et al., 1998). This is the most complex update in the LinSEPAL-ADMM iterative procedure due to the nonconvex objective and the Stiefel manifold. Next, LinSEPAL-ADMM updates Y in closed form through the element-wise soft-thresholding operator (cf. Eq. (37)). Finally, the scaled dual variable is updated by adding the primal residual evaluated at the current solution (cf. Eq. (R1)). The stopping criteria for LinSEPAL-ADMM are established according to primal and dual feasibility optimality conditions (Boyd et al., 2011, cf. App. H). To the best of our knowledge, the convergence guarantee for MADMM in the Riemannian space has not been proven. Consequently, the same holds for LinSEPAL-ADMM. Algorithm 1 summarizes the method.

LinSEPAL-PG. Our LinSEPAL-PG is an iterative algorithm alternating two updates (cf. Eq. (R2)). The first update is the proximal mapping providing a proximal gradient direction $G^{k}$ onto the tangent space to the Stiefel manifold, using the first-order approximation of the objective around the k-th estimate. The second is the update for $V^{k+1}$ , which exploits the canonical retraction (cf. Eq. (23)) technique for projecting back $V^{k} + G^{k}$ from the tangent space to the manifold. The hardest step in the LinSEPAL-PG algorithm is the proximal update (cf. Eq. (42)). We solve it by declining the regularized semi-smooth Newton method (Xiao et al., 2018) to our application (cf. App. I). Following the rationale in (Si et al., 2024), differently from the

original ManPG method which uses the parameterization of the tangent space, we constrain $G^{k}$ to the tangent space by exploiting the basis of the normal space to the manifold (cf. Eq. (44)). This way, we ease the mathematical solution, with benefits from the computational perspective (cf. Si et al., 2024). Next, LinSEPAL-PG updates $V^{k+1}$ in closed form (cf. Eq. (69)) by applying the QR-retraction, employing an Armijo line-search procedure to determine the stepsize. The optimization stops either when a maximum number of iterations is reached, or when $D_{V^{k+1}}^{KL}$ is below a threshold $\tau^{KL} \approx 0$ . LinSEPAL-PG inherits the global convergence of the ManPG framework, established in (Chen et al., 2020). Algorithm 2 summarizes the method.

# 5.2. Solution of the smooth learning problem

We provide a sketch of the solution below and the full mathematical derivation in App. J. In this case, we want to jointly optimize S and V, both being components of the linear CA, viz. $(\mathbf{B} \odot \mathbf{S} \odot \mathbf{V})^{\top}$ . Hence, unlike the nonsmooth case, we constrain to the Stiefel manifold the product $(\mathbf{B} \odot \mathbf{S} \odot \mathbf{V})$ . To solve Prob. 3, we combine the SOC, ADMM, and SCA methods. According to the rationale behind SOC, we add two splitting variables, namely $Y_{1}$ and $Y_{2}$ in $\mathrm{St}(\ell, h)$ (cf. Eq. (72)), to separate the nonconvexity of the objective function from that induced by the manifold. The reason why we have two splitting variables is that we need to take into account the bilinear form of the first constraint in Eq. (7). Additionally, to manage the second constraint in Eq. (7), we introduce another splitting variable $\mathbf{X} \in \mathrm{Sp}^{\Delta}(h, \ell)$ . Starting from the equivalent problem formulation (cf. Eq. (73)), we write the (nonconvex) scaled augmented Lagrangian (cf. Eq. (74)), thus arriving at the update recursion for our proposed method (cf. Eq. (75)). We term the latter CLinSEPAL (Constructive LinSEPAL) to highlight that it returns constructive support for CA. CLinSEPAL proceeds by iteratively minimizing the augmented Lagrangian w.r.t. the primal variables V, S, $Y_{1}$ , $Y_{2}$ , and X; and maximizing it w.r.t. the scaled dual ones. In the subproblems for V (cf. Eq. (76)) and S (cf. Eq. (84)), we adopt the SCA paradigm to manage the nonconvexity of $f(\mathbf{V}, \mathbf{S}^{k})$ and $f(\mathbf{V}^{k+1}, \mathbf{S})$ , respectively. By exploiting the smoothness of $f(\mathbf{V}, \mathbf{S})$ (cf. Corollary J.1), the strongly convex surrogates are derived around the current solution (cf. Eqs. (77) and (85)). CLinSEPAL solves the strongly convex surrogate subproblems (cf. Eqs. (78) and (88)) exactly. Due to the presence of the inequality constraints, the subproblem for S is a constrained quadratic programming problem. CLinSEPAL solves it via standard techniques (e.g., Stellato et al., 2020). These two steps in CLinSEPAL can be seen as an instance of the linearized ADMM framework (Alg.1 in Lu et al., 2021) where each internal update is solved exactly. Next, CLinSEPAL solves in closed-form the updates for the three splitting variables. Indeed, the subproblems for $Y_{1}$ and $Y_{2}$ amount to the closest orthogonal approximation problem (Fan & Hoffman, 1955; Higham, 1986), whose solution is obtained in closed form via polar decomposition. Subsequently, the subproblem for X is solved in closed-form according to Lemma J.3. Finally, the scaled dual variables are updated with the corresponding primal residuals. Empirical convergence for CLinSEPAL is established when the norms of primal (cf. Eq. (97)) and dual (cf. Eq. (98)) residuals vanish, in accordance with absolute and relative tolerances (cf. Eq. (99)). Algorithm 3 summarizes the method. Additionally, App. J.6 details the solution in the special case of full prior knowledge.

# 6. Empirical Assessment on Synthetic Data

This section provides the empirical assessment of LinSEPAL-ADMM, LinSEPAL-PG and CLinSEPAL with different degrees of prior knowledge, from full (fp) to partial (pp). We monitor four metrics to evaluate the learned CA $\widehat{V}^{\top}$ : (i) constructiveness, as required by Def. 4.2; (ii) $D_{\widehat{V}}^{KL}$ evaluating the alignment between $\varphi_{\#}^{\widehat{\mathbf{V}}}(\chi^{\ell})$ and $\chi^{h}$ ; (iii) the Frobenius distance between the absolute value of $\widehat{V}$ and that of the ground truth $V^{\star}$ , normalized by $\|V^{\star}\|_{F}$ to make the settings comparable; (iv) the F1 score computed using the support of the learned CAs and that of $V^{\star}$ to evaluate structural interventional consistency. App. K provides the definition for the above metrics and the hyper-parameters values used in the experiments.

Full prior knowledge. In the fp case, we investigate three different settings $(\ell,h)\in\{(12,2),(12,4),(12,6)\}$ , corresponding to the cases of high, medium-high, and medium coarse-graining. We do not consider the case where $h>\ell/2$ since the abstraction for $h-\ell/2$ nodes of the low-level model would be fully specified due to the availability of full prior knowledge. For each setting, we instantiate S=30 ground truth abstractions $V^{\star}$ , and for each simulation $s\in[S]$ we run all the methods R=50 times, with different initializations. Then, for each s and method, we retain the $\widehat{V}$ minimizing the objective $D^{KL}$ .

Fig. 3 shows the performance of the tested methods. All the methods provide constructive CAs $\forall s\in[S]$ , and reach a good level of alignment in terms of $D_{\widehat{\mathbf{V}}}^{\mathrm{KL}}$ . Recall that, while CLinSEPAL and LinSEPAL-ADMM stop the learning procedure according to primal and dual residuals convergence, LinSEPAL-PG exits when $D_{\widehat{\mathbf{V}}}^{\mathrm{KL}}$ is below a certain threshold $\tau^{\mathrm{KL}}$ (in the experiments $\tau^{\mathrm{KL}} = 10^{-4}$ ). The Frobenius absolute distance shows comparable performances for the three methods, although CLinSEPAL and LinSEPAL-ADMM outperform in case $(\ell,h) = (12,4)$ . This metric tells us that, as $h$ increases, the learned $\widehat{\mathbf{V}}$ tends (in absolute terms) to the ground truth. Interestingly, when $(\ell,h) = (12,2)$ we observe a high distance from $\mathbf{V}^{\star}$ , although the learned $\widehat{\mathbf{V}}$

![](images/c901727f8b96920a4fe061b55e8a70d7f600aaea21714987c95e9d3b50804b1f.jpg)  
Figure 3: Synthetic fp results for all settings $(\ell, h)$ and methods: (i) fraction of learned CAs that are constructive, (ii) $D_{\widehat{\mathbf{V}}}^{\mathrm{KL}}$ , (iii) normalized absolute Frobenius distance from $\mathbf{V}^{\star}$ , and (iv) F1 score.

has the correct structure (cf. F1 score). This suggests that under a high coarse-graining, the size of ker $D^{KL}$ grows, and it is more difficult for our methods to estimate $V^{\star}$ under (NA1)-(NA5). Finally, the F1 score confirms that the methods guarantee the true CA structure of $\hat{V}$ , for all the settings. To sum up, CLinSEPAL and LinSEPAL-ADMM are slightly better choices than LinSEPAL-PG in case of full prior knowledge in our experimental setting.

Partial prior knowledge. In the pp case, we consider the setting $(\ell,h)\in\{(4,2)\}$ and simulate partial prior knowledge by setting to 1 the entries of the 25%, 50%, and 75% of the rows in B. For each setting, we instantiate S=30 ground truth abstractions $V^{\star}$ , and for each simulation $s\in[S]$ we run all the methods R=30 times, with different initializations.

In Fig. 4, the first plot immediately shows that only CLin-SEPAL consistently returns a constructive linear CA, as guaranteed by its formulation in Prob. 3. We decided to consider methods performing under a threshold of 90% to be unreliable in returning constructive CAs and not to report their remaining metrics. For completeness, Fig. 7 in App. L provides the results where the threshold on constructiveness is removed. In the case of a limited drop of prior knowledge (25%) all methods perform well, similarly to the fp case, with CLinSEPAL and LinSEPAL-ADMM slightly outperforming LinSEPAL-PG. With a higher drop (50%), LinSEPAL-PG fails to achieve our constructiveness threshold, while CLinSEPAL and LinSEPAL-ADMM still perform well, although LinSEPAL-ADMM provides a lower fraction of constructive CAs. Finally, with the highest drop (75%) CLinSEPAL succeeds in learning a constructive CA and lowering $D^{KL}$ , even if the Frobenius absolute distance slightly increases. To sum up, for the pp setting only CLin-SEPAL guarantees a constructive abstraction.

![](images/80508b8d7fa7ec4c61e05de66edaae8e571106d1bf70fbccc65a2bfc4e4ceb9c.jpg)

<details>
<summary>line</summary>

| Step | Fraction of learned constructive morphisms |
| ---- | ------------------------------------------ |
| 1    | 1.0                                        |
| 2    | 0.9                                        |
| 3    | 0.5                                        |
| 4    | 0.0                                        |
</details>

![](images/3a253b89c3b1362fb926afaad7196da9dcbd14e72dcc1922524bc6e9e4b64f66.jpg)

<details>
<summary>boxplot</summary>

| Group | Median | Q1 | Q3 | Min | Max |
|-------|--------|----|----|-----|-----|
| 1     | 1e-5   | 1e-7 | 1e-6 | 1e-8 | 1e-4 |
| 2     | 1e-6   | 1e-8 | 1e-7 | 1e-9 | 1e-5 |
| 3     | 1e-7   | 1e-9 | 1e-8 | 1e-10 | 1e-6 |
| 4     | 1e-8   | 1e-10 | 1e-9 | 1e-11 | 1e-7 |
| 5     | 1e-9   | 1e-11 | 1e-10 | 1e-12 | 1e-8 |
| 6     | 1e-10  | 1e-12 | 1e-11 | 1e-13 | 1e-9 |
| 7     | 1e-11  | 1e-13 | 1e-12 | 1e-14 | 1e-10 |
| 8     | 1e-12  | 1e-14 | 1e-13 | 1e-15 | 1e-11 |
| 9     | 1e-13  | 1e-15 | 1e-14 | 1e-16 | 1e-12 |
| 10    | 1e-14  | 1e-16 | 1e-15 | 1e-17 | 1e-13 |
| 11    | 1e-15  | 1e-17 | 1e-16 | 1e-18 | 1e-14 |
| 12    | 1e-16  | 1e-18 | 1e-17 | 1e-19 | 1e-15 |
| 13    | 1e-17  | 1e-19 | 1e-18 | 1e-20 | 1e-16 |
| 14    | 1e-18  | 1e-20 | 1e-19 | 1e-21 | 1e-17 |
| 15    | 1e-19  | 1e-21 | 1e-20 | 1e-22 | 1e-18 |
| 16    | 1e-20  | 1e-22 | 1e-21 | 1e-23 | 1e-19 |
| 17    | 1e-21  | 1e-23 | 1e-22 | 1e-24 | 1e-20 |
| 18    | 1e-22  | 1e-24 | 1e-23 | 1e-25 | 1e-21 |
| 19    | 1e-23  | 1e-25 | 1e-24 | 1e-26 | 1e-22 |
| 20    | 1e-24  | 1e-26 | 1e-25 | 1e-27 | 1e-23 |
| 21    | 1e-25  | 1e-27 | 1e-26 | 1e-28 | 1e-24 |
| 22    | 1e-26  | 1e-28 | 1e-27 | 1e-29 | 1e-25 |
| 23    | 1e-27  | 1e-29 | 1e-28 | 1e-30 | 1e-26 |
| 24    | 1e-28  | 1e-30 | 1e-29 | 1e-31 | 1e-27 |
| 25    | 1e-29  | 1e-31 | 1e-30 | 1e-32 | 1e-28 |
| 26    | 1e-30  | 1e-32 | 1e-31 | 1e-33 | 1e-29 |
| 27    | 1e-31  | 1e-33 | 1e-32 | 1e-34 | 1e-30 |
| 28    | 1e-32  | 1e-34 | 1e-33 | 1e-35 | 1e-31 |
| 29    | 1e-33  | 1e-35 | 1e-34 | 1e-36 | 1e-32 |
| 30    | 1e-34  | 1e-36 | 1e-35 | 1e-37 | 1e-33 |
| Note: The data is not explicitly provided in the code. The box plot visualizes the distribution of KL divergence values for each group. The labels are 'A' (Group A) and 'B' (Group B). There is no additional data series or legend data provided in the code.
</details>

![](images/e409a3f3f65d2b6418701ec056601ae444cc9b2c2035835b699f72097a89905b.jpg)

<details>
<summary>scatter</summary>

| Fraction of unavailable prior knowledge | Frobenius absolute distance |
| -------------------------------------- | --------------------------- |
| 0.25                                   | 0.0                         |
| 0.25                                   | 0.1                         |
| 0.25                                   | 0.2                         |
| 0.25                                   | 0.3                         |
| 0.25                                   | 0.4                         |
| 0.25                                   | 0.5                         |
| 0.25                                   | 0.6                         |
| 0.25                                   | 0.7                         |
| 0.25                                   | 0.8                         |
| 0.25                                   | 0.9                         |
| 0.25                                   | 1.0                         |
| 0.5                                    | 0.0                         |
| 0.5                                    | 0.1                         |
| 0.5                                    | 0.2                         |
| 0.5                                    | 0.3                         |
| 0.5                                    | 0.4                         |
| 0.5                                    | 0.5                         |
| 0.5                                    | 0.6                         |
| 0.5                                    | 0.7                         |
| 0.5                                    | 0.8                         |
| 0.5                                    | 0.9                         |
| 0.5                                    | 1.0                         |
| 0.75                                   | 0.0                         |
| 0.75                                   | 0.1                         |
| 0.75                                   | 0.2                         |
| 0.75                                   | 0.3                         |
| 0.75                                   | 0.4                         |
| 0.75                                   | 0.5                         |
| 0.75                                   | 0.6                         |
| 0.75                                   | 0.7                         |
| 0.75                                   | 0.8                         |
| 0.75                                   | 0.9                         |
| 0.75                                   | 1.0                         |
</details>

![](images/7877fcaa5de6e57812b9368c4eb10a70bdada8e7f9c2f648a15d90960bd68e3c.jpg)

<details>
<summary>scatter</summary>

F1 score
| Fraction of unavailable prior knowledge | F1 score |
| :--- | :--- |
| 0.25 | 0.75 |
| 0.25 | 0.75 |
| 0.5 | 0.75 |
| 0.5 | 0.75 |
| 0.75 | 0.5 |
</details>

Figure 4: Synthetic pp results for setting $(\ell,h)=(4,2)$ , all methods, and prior knowledge amounting to the correct structural mapping for 25%, 50%, or 75% of the nodes. All plots and color legend as in Fig. 3.

# 7. Causal Abstraction of Brain Networks

To show the practical relevance of our approach, we apply CLinSEPAL to resting-state functional magnetic resonance imaging (rs-fMRI) data, using the dataset from (D'Acunto et al., 2024) (refer to the paper for details on the dataset). The data, publicly released as part of the Human Connectome Project (Smith et al., 2013), comprises recordings from 100 healthy adults with a parcellation scheme that divides the brain into 89 regions of interest (ROIs), $K = 44$ for each hemisphere plus the shared vermis region.

We simulate a first investigating team of neuroscientists taking zero-mean stationary time series for the left hemisphere of the first adult in the dataset. They estimate the data covariance matrix using a Gaussian mixture probability model, viz. $\Sigma^{\ell} \in R^{\ell \times \ell}$ , with $\ell = K + 1$ , and interpret it as generated by an underlying, unknown, low-level SCM.

In a first fp scenario, we imagine a second investigating team that has collected data according to their causal network specified on a coarser parcellation of the same brain in h = 14 macro ROIs. We generate the data for the second team using a ground truth linear CA B, $V^{\star} \in \text{St}(45, 14)$

based on the structural mapping in (D'Acunto et al., 2024), and use the data for estimating the covariance matrix $\Sigma^h\in$ $\mathbb{R}^{h\times h}$ . In this scenario it is realistic to assume knowledge of B defining how macro ROIs are mapped to ROIs. Then, to align their models, the two groups run CLinSEPAL to recover the abstraction given $\Sigma^{\ell},\Sigma^{h}$ and B. Fig. 8 (in App. M) shows that CLinSEPAL recovers $\mathbf{V}^{\star}$ .

In a second pp scenario, we imagine that the second investigating team has collected data according to a causal network aggregating ROI time series into $h = 8$ brain functional networks related to different activities (e.g., motor, visual, default mode). Data is generated again through a ground truth linear CA B, $\mathbf{V}^{\star} \in \mathrm{St}(45,8)$ based on groupings in (D'Acunto et al., 2024) and the covariance matrix $\Sigma^{h} \in \mathbb{R}^{h \times h}$ computed. In this scenario, knowledge of B is debatable as different studies in the literature suggest different relations between ROIs and functions; we then express this partial information via uncertainty over B, meaning that some rows of B have more than one entry equal to one. Fig. 9 in App. M shows the B matrix provided as input to CLinSEPAL, as well as the ground truth. The two groups now run CLinSEPAL using $\Sigma^{\ell}, \Sigma^{h}$ and an uncertain B; partial knowledge compounds on an already challenging learning problem due to the high coarse-graining. Fig. 10 and Fig. 11 show results with different levels of uncertainty. For low uncertainty, CLinSEPAL correctly retrieves the structure of the CA, although we observe some variation in the colors w.r.t. $\mathbf{V}^{\star}$ ; additionally, $D_{\widehat{\mathbf{V}}}^{\mathrm{KL}}$ and the Frobenius absolute distance in Fig. 11 show that misalignment is minimized and $\widehat{\mathbf{V}}$ very close to $\mathbf{V}^{\star}$ . For medium and high uncertainty, CLinSEPAL makes some mistakes in terms of structural mapping, but Fig. 11 shows that insights from the method are still valuable.

# 8. Conclusion and Future Works

In this work, we addressed the challenge of CA learning in realistic scenarios, abandoning restrictive assumptions (NA1)-(NA5) that limit the applicability of existing methods. We proposed an alternative category-theoretic framework for SCM and CA, and introduced the semantic embedding principle to learn CAs that meaningfully preserve information. We formulated a general CA learning problem grounded in SEP, under a mild assumption of partial prior knowledge about the structure of CA. For the linear CA setting, we showed how SEP links CA to the geometry of the Stiefel manifold; as an application, we tackled the important case of Gaussian measures, with the KL divergence as a measure of alignment between the low- and high-level SCMs. We pursued two different formulations. For the first, a nonsmooth Riemannian learning problem, we devised the LinSEPAL-ADMM and LinSEPAL-PG methods. For the second, a smooth Riemannian learning problem ensuring the constructiveness of the CA, we developed CLinSEPAL. Our empirical assessment on synthetic data confirmed the effectiveness of our methods, and the application to brain data showcased the potential in real-world problems.

Our work paves the way for several exciting research directions. First, as it emerges from our Gaussian application, linear CAs with different probability measures deserve careful investigation. Second, studying the nonlinear case is a compelling avenue. We believe that deep and reinforcement learning paradigms, such as encoding-decoding and actor-critic architectures, hold promise for modeling nonlinear CA maps. Lastly, we view our work as a foundational step toward observational causal abstraction learning, bridging the gap between CA learning and causal discovery (Spirtes & Zhang, 2016). Our category-theoretic framework underscores the pivotal role of exogenous variables, drawing a path to translate SCM identifiability results into CA identifiability results. This suggests that, in some cases, interventional consistency may be achieved without relying on interventional data.

# Impact Statement

Our work is foundational, aiming at advancing the field of causal abstraction. Our proposed methods can be applied to different application domains, such as neuroscience. As demonstrated by our empirical assessment, the information resulting from their application is high-level and useful for a better understanding. Hence, we believe that the risks associated with improper usage of our techniques are low.

# Acknowledgements

The work of Gabriele D'Acunto and Paolo Di Lorenzo was supported by the SNS JU project 6G-GOALS (Strinati et al., 2024) under the EU's Horizon program Grant Agreement No 101139232. The work of Gabriele D'Acunto was also supported by the European Union under the Italian National Recovery and Resilience Plan (NRRP) of NextGenerationEU, partnership on “Telecommunications of the Future” (PE00000001 - program “RESTART”). The work of Yorgos Felekis was supported by the Onassis Foundation - Scholarship ID: F ZR 063-1/2021-2022.

# References

Absil, P.-A., Mahony, R., and Sepulchre, R. Optimization Algorithms on Matrix Manifolds. Princeton University Press, 2008.

Beckers, S. and Halpern, J. Y. Abstracting causal models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 2678–2685, 2019.

Bereska, L. and Gavves, S. Mechanistic interpretability for ai safety-a review. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=ePUVetPKu6.   
Bollen, K. A. Structural equations with latent variables, volume 210. John Wiley & Sons, 1989.   
Boumal, N. An introduction to optimization on smooth manifolds. Cambridge University Press, 2023.   
Boumal, N., Mishra, B., Absil, P.-A., and Sepulchre, R. Manopt, a Matlab toolbox for optimization on manifolds. Journal of Machine Learning Research, 15(1):1455–1459, 2014.   
Boyd, S. and Vandenberghe, L. Convex Optimization. Cambridge University Press, 2004.   
Boyd, S., Parikh, N., Chu, E., Peleato, B., Eckstein, J., et al. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends® in Machine learning, 3(1):1–122, 2011.   
Brookes, M. The matrix reference manual.
http://www.ee.imperial.ac.uk/hp/
staff/dmb/matrix/intro.html, 2020. Accessed: 2024-01-10.   
Cai, Y. and Lim, L.-H. Distances between probability distributions of different dimensions. IEEE Transactions on Information Theory, 68(6), 2022.   
Chen, S., Ma, S., Man-Cho So, A., and Zhang, T. Proximal gradient method for nonsmooth optimization over the Stiefel manifold. SIAM Journal on Optimization, 30(1):210–239, 2020. doi: 10.1137/18M122457X. URL https://doi.org/10.1137/18M122457X.   
D'Acunto, G., Bonchi, F., Morales, G. D. F., and Petri, G. Extracting the multiscale causal backbone of brain dynamics. In Causal Learning and Reasoning, pp. 265–295. PMLR, 2024.   
Diamond, S. and Boyd, S. CVXPY: A python-embedded modeling language for convex optimization. Journal of Machine Learning Research, 17(83):1–5, 2016.   
Dyer, J., Bishop, N. G., Felekis, Y., Zennaro, F. M., Calinescu, A., Damoulas, T., and Wooldridge, M. J. Interventionally consistent surrogates for complex simulation models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=UtTjgMDTFO.   
Edelman, A., Arias, T. A., and Smith, S. T. The geometry of algorithms with orthogonality constraints. SIAM Journal on Matrix Analysis and Applications, 20(2):303–353, 1998.

Fan, K. and Hoffman, A. J. Some metric inequalities in the space of matrices. Proceedings of the American Mathematical Society, 6(1):111–116, 1955.   
Felekis, Y., Zennaro, F. M., Branchini, N., and Damoulas, T. Causal optimal transport of abstractions. In Causal Learning and Reasoning, pp. 462–498. PMLR, 2024.   
Ganguly, N., Fazlija, D., Badar, M., Fisichella, M., Sikdar, S., Schrader, J., Wallat, J., Rudra, K., Koubarakis, M., Patro, G. K., et al. A review of the role of causality in developing trustworthy AI systems. arXiv preprint arXiv:2302.06975, 2023.   
Geiger, A., Lu, H., Icard, T., and Potts, C. Causal abstractions of neural networks. Advances in Neural Information Processing Systems, 34:9574–9586, 2021.   
Geiger, A., Wu, Z., Potts, C., Icard, T., and Goodman, N. Finding alignments between interpretable causal variables and distributed neural representations. In Causal Learning and Reasoning, pp. 160–187. PMLR, 2024.   
Higham, N. J. Computing the polar decomposition—with applications. SIAM Journal on Scientific and Statistical Computing, 7(4):1160–1174, 1986.   
Higham, N. J. and Cheng, S. H. Modifying the inertia of matrices arising in optimization. Linear Algebra and its Applications, 275:261–279, 1998.   
Jacobs, B., Kissinger, A., and Zanasi, F. Causal inference by string diagram surgery. In International Conference on Foundations of Software Science and Computation Structures, pp. 313–329. Springer, 2019.   
Kekić, A., Schölkopf, B., and Besserve, M. Targeted reduction of causal models. In Uncertainty in Artificial Intelligence, pp. 1953–1980. PMLR, 2024.   
Komanduri, A., Wu, Y., Huang, W., Chen, F., and Wu, X. SCM-VAE: Learning identifiable causal representations via structural knowledge. In 2022 IEEE International Conference on Big Data (Big Data), pp. 1014–1023. IEEE, 2022.   
Kovnatsky, A., Glashoff, K., and Bronstein, M. M. MADMM: A generic algorithm for non-smooth optimization on manifolds. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part V 14, pp. 680–696. Springer, 2016.   
Lai, R. and Osher, S. A splitting method for orthogonality constrained problems. Journal of Scientific Computing, 58:431–449, 2014.

Lu, S., Lee, J. D., Razaviyayn, M., and Hong, M. Linearized ADMM converges to second-order stationary points for non-convex problems. IEEE Transactions on Signal Processing, 69:4859–4874, 2021.   
Mac Lane, S. Categories for the Working Mathematician, volume 5. Springer Science & Business Media, 2013.   
Marzouk, Y., Moselhy, T., Parno, M., and Spantini, A. Sampling via measure transport: An introduction, pp. 1–41. Springer International Publishing, 2016. ISBN 9783319112596. doi: 10.1007/978-3-319-11259-6\_23-1. URL http://dx.doi.org/10.1007/978-3-319-11259-6\_23-1.   
Massidda, R., Magliacane, S., and Bacciu, D. Learning causal abstractions of linear structural causal models. In Uncertainty in Artificial Intelligence, pp. 2486–2515. PMLR, 2024.   
Nedić, A., Pang, J.-S., Scutari, G., Sun, Y., Scutari, G., and Sun, Y. Parallel and distributed successive convex approximation methods for big-data optimization. Multi-Agent Optimization: Cetraro, Italy 2014, pp. 141–308, 2018.   
Otsuka, J. and Saigo, H. On the equivalence of causal models: A category-theoretic approach. In Conference on Causal Learning and Reasoning, pp. 634–646. PMLR, 2022.   
Parikh, N., Boyd, S., et al. Proximal algorithms. Foundations and Trends® in Optimization, 1(3):127–239, 2014.   
Pearl, J. Causality. Cambridge University Press, 2009.   
Perrone, P. Starting Category Theory. World Scientific, 2024.   
Qi, Y., Schölkopf, B., and Jin, Z. Causal responsibility attribution for human-AI collaboration. arXiv preprint arXiv:2411.03275, 2024.   
Rawal, A., Raglin, A., Rawat, D. B., Sadler, B. M., and McCoy, J. Causality for trustworthy artificial intelligence: Status, challenges and perspectives. ACM Computing Surveys, 2024.   
Rischel, E. F. The category theory of causal models. Master's thesis, University of Copenhagen, 2020.   
Rubenstein, P. K., Weichwald, S., Bongers, S., Mooij, J. M., Janzing, D., Grosse-Wentrup, M., and Schölkopf, B. Causal consistency of structural equation models. In 33rd Conference on Uncertainty in Artificial Intelligence (UAI 2017), pp. 808–817. Curran Associates, Inc., 2017.

Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., and Bengio, Y. Toward causal representation learning. Proceedings of the IEEE, 109(5):612–634, 2021.   
Schooltink, W. and Zennaro, F. M. Aligning graphical and functional causal abstractions. arXiv preprint arXiv:2412.17080, 2024.   
Shimizu, S., Hoyer, P. O., Hyvärinen, A., Kerminen, A., and Jordan, M. A linear non-Gaussian acyclic model for causal discovery. Journal of Machine Learning Research, 7(10), 2006.   
Si, W., Absil, P.-A., Huang, W., Jiang, R., and Vary, S. A Riemannian proximal Newton method. SIAM Journal on Optimization, 34(1):654–681, 2024.   
Smith, S. M., Beckmann, C. F., Andersson, J., Auerbach, E. J., Bijsterbosch, J., Douaud, G., Duff, E., Feinberg, D. A., Griffanti, L., Harms, M. P., et al. Resting-state fMRI in the Human Connectome Project. NeuroImage, 80:144–168, 2013.   
Spirtes, P. and Zhang, K. Causal discovery and inference: Concepts and recent methodological advances. In Applied Informatics, volume 3, pp. 1–28. Springer, 2016.   
Stellato, B., Banjac, G., Goulart, P., Bemporad, A., and Boyd, S. OSQP: An operator splitting solver for quadratic programs. Mathematical Programming Computation, 12(4):637–672, 2020. doi: 10.1007/s12532-020-00179-2. URL https://doi.org/10.1007/s12532-020-00179-2.   
Strinati, E. C., Di Lorenzo, P., Sciancalepore, V., Aijaz, A., Kountouris, M., Gündüz, D., Popovski, P., Sana, M., Stavrou, P. A., Soret, B., et al. Goal-oriented and semantic communication in 6G AI-native networks: The 6G-GOALS approach. In 2024 Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit), pp. 1–6. IEEE, 2024.   
Thomas, C. K. and Saad, W. Neuro-symbolic causal reasoning meets signaling game for emergent semantic communications. IEEE Transactions on Wireless Communications, 23(5):4546–4563, 2023.   
Thomas, C. K. and Saad, W. Symbolic logic and category theory for reasoning-enabled semantic communications. In 2024 58th Asilomar Conference on Signals, Systems, and Computers, pp. 121–125. IEEE, 2024.   
Thomas, C. K., Saad, W., and Xiao, Y. Causal semantic communication for digital twins: A generalizable imitation learning approach. IEEE Journal on Selected Areas in Information Theory, 4:698–717, 2023.

Xiao, X., Li, Y., Wen, Z., and Zhang, L. A regularized semi-smooth Newton method with projection steps for composite convex programs. Journal of Scientific Computing, 76:364–389, 2018.   
Yang, M., Liu, F., Chen, Z., Shen, X., Hao, J., and Wang, J. CausalVAE: Disentangled representation learning via neural structural causal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9593–9602, 2021.   
Zennaro, F. M., Drávucz, M., Apachitei, G., Widanage, W. D., and Damoulas, T. Jointly learning consistent causal abstractions over multiple interventional distributions. In 2nd Conference on Causal Learning and Reasoning, 2023.   
Zhang, K. and Hyvärinen, A. On the identifiability of the post-nonlinear causal model. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, pp. 647–655, 2009.

# A. Extended Notation for the Appendix

Below is the notation used throughout the appendices. The set of integers from 1 to $n$ is [n]. The vectors of zeros and ones of size $n$ are $\mathbf{0}_n$ and $\mathbf{1}_n$ . The identity matrix of size $n \times n$ is $\mathbf{I}_n$ . The entry indexed by row $i$ and column $j$ is $a_{ij} = [\mathbf{A}]_{ij}$ , $\mathrm{diag}(\mathbf{a})$ is the diagonal matrix having as diagonal the vector $\mathbf{a}$ , while $\mathrm{diag}(\mathbf{A})$ is the diagonal of the matrix $\mathbf{A}$ . The Frobenious norm is $\| \mathbf{A} \|_{\mathrm{F}}$ . The set of positive definite matrices over $\mathbb{R}^{n \times n}$ is $S_{++}^n$ . That of symmetric ones as $\operatorname{Sym}(p)$ . The column-wise vectorization of a matrix is $\operatorname{vec}()$ . The Hadamard product is $\odot$ . Function composition is $\circ$ .

Let $\mathcal{M}(\mathcal{X}^n)$ be the set of Borel measures over $\mathcal{X}^n\subseteq \mathbb{R}^n$ . Given a measure $\mu^n\in \mathcal{M}(\mathcal{X}^n)$ and a measurable map $\varphi^{\mathbf{V}}$ , $\mathcal{X}^n\ni \mathbf{x}\stackrel {\varphi^{\mathbf{V}}}{\longmapsto}\mathbf{V}^\top \mathbf{x}\in \mathcal{X}^m$ , we denote by $\varphi_{\#}^{\mathbf{V}}(\mu^n)\coloneqq \mu^n (\varphi^{\mathbf{V}^{-1}}(\mathbf{x}))$ the pushforward measure $\mu^m\in \mathcal{M}(\mathcal{X}^m)$ . The proximal mapping of $h$ at $\mathbf{A}$ is $\mathrm{prox}_{\lambda h(\cdot)}(\mathbf{A}) = \arg \min_{\mathbf{V}}h(\mathbf{V}) + 1 / (2\lambda)\| \mathbf{V} - \mathbf{A}\|_{\mathrm{F}}^2,\lambda \in \mathbb{R}^{+}$ . The Euclidean gradient of a smooth $f$ is $\nabla f$ , while the Riemannian one $\widetilde{\nabla} h$ . The Euclidean subgradient of a nonsmooth $h$ is $\partial h$ , the Riemannian instead $\widetilde{\partial} h$ .

# B. Category Theory Essentials

Below are fundamental definitions and examples that are instrumental in providing the necessary background on category theory to understand our work. For a comprehensive overview of category theory see resources such as Mac Lane (2013); Perrone (2024).

Definition B.1 (Category). A category C consists of

- A collection of objects, viz. $X$ in $\mathbb{C}$ ,   
- A collection of morphisms, viz. $f: X \to Y$ in C;

such that:

- Each morphism $f$ has assigned two objects of the category called source and target, respectively,   
• Each object X has an identity morphism $id_{X}: X \to X$ ,   
- Given $f: X \to Y$ and $g: Y \to Z$ , than the composition exists, $g \circ f = h: X \to Z$ .

These structures satisfy the following axioms:

- (Unitality) $\forall f: X \to Y$ , $f \circ \mathrm{id}_X = f$ and $\mathrm{id}_Y \circ f = f$ ;   
- (Associativity) Given $f, g$ , and $h$ such that the compositions hold, then $h \circ (g \circ f) = (h \circ g) \circ f$ .

Example 1. The following are some notable examples of categories:

- Indicate with Poset a partial order set. Poset can be viewed as the category whose objects are the elements $p$ and morphisms are order relations $p \leq p'$ . Notice that there is at most one morphism between two objects;   
- $\mathrm{Vect}_{\mathbb{R}}$ is the category whose objects are real vector spaces and morphisms are linear maps;   
- Prob is the category whose objects are probability measure spaces and morphisms measurable maps.

Arrows between categories are called functors, defined as follows:

Definition B.2 (Functor). Consider C and D categories. A functor $F : C \to D$ consists of the following data:

- For each object $X$ in $\mathbb{C}$ , an object $F(X)$ in $\mathbb{D}$ ;   
- For each object morphism $f: X \to Y$ in $\mathbb{C}$ , a morphism $F(f): F(X) \to F(Y)$ in $\mathbb{D}$ ;

such that the following axioms hold:

- (Unitality) $\forall X$ in $C$ , $F(\mathrm{id}_X) = \mathrm{id}_{F(X)}$ . In other words, the identity in $C$ is mapped into the identity in $D$ .   
- (Compositionality) $\forall f$ and $g$ in $\mathbb{C}$ such that the composition is defined, then $F(g \circ f) = F(g) \circ F(f)$ . In other words, the composition in $\mathbb{C}$ is mapped into the composition in $\mathbb{D}$ .

To ease the notation, in the sequel, we use $F^X$ and $F^f$ to denote $F(X)$ and $F(f)$ , respectively. Finally, we can have arrows

between functors as well, called natural transformations:

Definition B.3 (Natural transformation). Consider two categories C and D, and two functors between them, namely $F : C \rightarrow D$ and $G : C \rightarrow D$ . A natural transformation $\alpha : F \stackrel{\bullet}{\rightarrow} G$ consists of the following data:

- For each object $X$ in $\mathsf{C}$ , a morphism $\alpha_X: F^X \to G^X$ in $\mathcal{D}$ called the component of $\alpha$ at $X$ ;   
- For each morphism $f: X \to X'$ in $\mathbb{C}$ , the following diagram commutes:

$$
\begin{array}{c} F ^ {X} \xrightarrow {F ^ {f}} F ^ {X ^ {\prime}} \\ \alpha_ {X} \Bigg \downarrow \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \alpha_ {X ^ {\prime}} \\ G ^ {X} \xrightarrow [ G ^ {f} ]{} G ^ {X ^ {\prime}} \end{array} \tag {11}
$$

A natural transformation can be thought of as a consistent system of arrows between two functors, invariant with respect to maps between the images of two functors.

# C. Causality and Causal Abstraction

This section provides additional definitions and examples related to SCMs and the CA framework.

# C.1. Mixing functions

A set of structural function in a Markovian SCM can be reduced to a set of mixing functions dependent only on the exogenous variables.

Given an SCM $M^{n}$ , recall that F is a set of n functional assignments which define the values $X_{i} = f_{i}(\mathcal{P}_{i}, Z_{i}), \forall i \in [n]$ , with $P_{i} \subseteq X \setminus \{X_{i}\}$ . Denote by $Z^{A_{i}} \subseteq Z \setminus \{Z_{i}\}$ the set of exogenous variables corresponding to the ancestors of $X_{i}$ , where $A_{i} \subseteq [n] \setminus \{i\}$ . According to F, we can identify a set of mixing functions $M = \{m_{1}, \ldots, m_{n}\}$ such that the values of the endogenous random variables are equivalently expressed as $x_{i} = m_{i} (\{z_{j}\}_{j \in \mathcal{A}_{i}}, z_{i}), \forall i \in [n]$ .

Further, we can also characterize the product probability measure implied by the SCM purely in terms of the exogenous variables, viz. $\chi^{\mathcal{X}} = \prod_{i\in [n]}P\left(X_i|\mathcal{Z}^{\mathcal{A}_i},Z_i\right)$ .

As an example, consider a causal relation $x_{1} \to x_{2}$ . In the linear SCM with additive noise (Bollen, 1989; Shimizu et al., 2006) setting we have

$$
\left\{ \begin{array}{l} x _ {1} = z _ {1}, \\ x _ {2} = c _ {2, 1} x _ {1} + z _ {2} = c _ {2, 1} z _ {1} + z _ {2}. \end{array} \right. \tag {12}
$$

Again, for the post-nonlinear model (Zhang & Hyvärinen, 2009), we get

$$
\left\{ \begin{array}{l} x _ {1} = f _ {1, 1} (z _ {1}) = m _ {1} (z _ {1}), \\ x _ {2} = f _ {2, 2} (f _ {2, 1} (x _ {1}) + z _ {2}) \\ \quad = f _ {2, 2} (f _ {2, 1} \circ f _ {1, 1} (z _ {1}) + z _ {2}) \\ \quad = m _ {2} (z _ {1}, z _ {2}). \end{array} \right. \tag {13}
$$

# C.2. Interventional consistency

A typical requirement imposed on CA maps is that they act in a consistent way with respect to interventions (Rischel, 2020).

Definition C.1 (Interventional consistency). Given an $\alpha$ -abstraction between $\mathsf{M}^{\ell}$ and $\mathsf{M}^h$ and a set $\mathcal{I}$ of hard interventions on $\mathcal{X}_{\mathcal{I}}^{h} \subseteq \mathcal{X}^{h}$ , the abstraction is interventionally consistent if, for any intervention in $\mathcal{I}$ and for every set of target variable $\mathcal{Y}_{\mathcal{I}}^{h} \subseteq \mathcal{X}^{h} \setminus \mathcal{X}_{\mathcal{I}}^{h}$ , the following diagram commutes:

![](images/c00db092eb56bc528ec89a44a333f08e446b2bc39fbf4e722d372eee847a12f0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["D[xᵢ^ℓ"] -->|P(yᵢ^ℓ | do(xᵢ^ℓ))| B["D[yᵢ^ℓ"]]
    A -->|αxᵢ^h| C["D[xᵢ^h"]]
    B -->|αyᵢ^h| D["D[yᵢ^h"]]
    C -->|P(yᵢ^h | do(xᵢ^h))| D
```
</details>

or equivalently,

$$
\alpha_ {\mathcal {Y} _ {\mathcal {I}} ^ {h}} \left(P \left(\mathcal {Y} _ {\mathcal {I}} ^ {\ell} \mid \mathrm{do} \left(\mathcal {X} _ {\mathcal {I}} ^ {\ell}\right)\right)\right) = P \left(\mathcal {Y} _ {\mathcal {I}} ^ {h} \mid \alpha_ {\mathcal {X} _ {\mathcal {I}} ^ {h}} \left(\mathrm{do} \left(\mathcal {X} _ {\mathcal {I}} ^ {h}\right)\right)\right), \tag {14}
$$

where $\mathcal{X}_{\mathcal{I}}^{\ell} = m^{-1}(\mathcal{X}_{\mathcal{I}}^{h})$ and $\mathcal{Y}_{\mathcal{I}}^{\ell} = m^{-1}(\mathcal{Y}_{\mathcal{I}}^{h})$ .

Essentially, commutativity suggests that we obtain equivalent intervention outcomes in two different ways: (i) either by intervening on the low-level model and then abstracting or, (ii) by abstracting to the high-level model and then intervening in an equivalent fashion.

# C.3. Linear abstraction

The class of abstractions may be restricted by an assumption of the form of the abstraction map (Massidda et al., 2024):

Definition C.2 (Linear abstraction). Given an $\alpha$ -abstraction $\alpha = \langle \mathcal{R}, m, \alpha \rangle$ from $M^{\ell}$ to $M^{h}$ , the abstraction is linear if $\alpha = \mathbf{V}^{\top} \in \mathbb{R}^{h \times \ell}$ .

# C.4. Constructive abstraction

A particularly well-behaved form of abstraction is a constructive abstraction. In the context of the $\tau$ -abstraction framework (Beckers & Halpern, 2019), a constructive abstraction is an abstraction such that: (i) the variable mapping defines a clustering of the low-level variables (constructivity); (ii) consistency holds for all high-level interventions (strongness); (iii) the value map is surjective and it implies a map between exogenous values and between interventions ( $\tau$ -abstraction). In the $\alpha$ -framework a few of these properties hold by construction; thus, we define a constructive abstraction as (Schooltink & Zennaro, 2024):

Definition C.3 (Constructive abstraction). Given an $\alpha$ -abstraction $\alpha = \langle R, m, \alpha \rangle$ from $M^{\ell}$ to $M^{h}$ , the abstraction is constructive if the abstraction is interventionally consistent and implies the existence of a map $\alpha_{U} : Z^{\ell} \to Z^{h}$ between exogenous variables.

# C.5. Measure-theoretic definition of an SCM

Any SCM can be defined in terms of the probability measure spaces underlying it:

Definition C.4 (Measure-theoretic SCM). A (Markovian) SCM $M^{n}$ is a triple $\langle(\mathcal{U},\Sigma_{\mathcal{U}},\zeta),(\mathcal{V},\Sigma_{\mathcal{V}},\chi),\mathcal{M}\rangle$ where:

\- $(\mathcal{U}, \Sigma_{\mathcal{U}}, \zeta)$ is a probability space associated with exogenous variables. Specifically, it consists of the product probability measure $\zeta = \zeta_1 \times \ldots \times \zeta_n$ on the product measurable space $(\mathcal{U}, \Sigma_{\mathcal{U}})$ where $\mathcal{U} = \mathcal{U}_1 \times \ldots \times \mathcal{U}_n$ is a product set and $\Sigma_{\mathcal{U}} = \Sigma_{\mathcal{U}_1} \otimes \ldots \otimes \Sigma_{\mathcal{U}_n}$ is a product $\sigma$ -algebra. The probability measure is such that, for each $\mathcal{W}_1 \in \Sigma_{\mathcal{U}_1}, \ldots, \mathcal{W}_n \in \Sigma_{\mathcal{U}_n}$ , we have

$$
\zeta_ {1} \times \dots \times \zeta_ {n} (\mathcal {W} _ {1} \times \dots \times \mathcal {W} _ {n}) = \zeta_ {1} (\mathcal {W} _ {1}) \times \dots \times \zeta_ {n} (\mathcal {W} _ {n}); \tag {15}
$$

- $(\mathcal{V}, \Sigma_{\mathcal{V}}, \chi)$ is a probability space associated with endogenous variables consisting of a joint probability measure $\chi$ on the product measurable space $(\mathcal{V}, \Sigma_{\mathcal{V}}) = (\mathcal{V}_1 \times \ldots \times \mathcal{V}_n, \Sigma_{\mathcal{V}_1} \otimes \ldots \otimes \Sigma_{\mathcal{V}_n})$ ;   
- $\mathcal{M}$ is a set of $n$ mixing measurable maps $\varphi^{m_i}$ (cf. Def. 2.1) such that the joint probability measure $\chi$ factorizes as

$$
\chi = \bigotimes_ {i = 1} ^ {n} \varphi_ {\#} ^ {m _ {i}} \left(\mu_ {i} \left(\mathcal {U} _ {i} \times \mathcal {U} ^ {\mathcal {A} _ {i}}\right)\right); \tag {16}
$$

where $U^{A_{i}} = \times_{j \in A_{i}} U_{j}$ , and, denoting by $\Sigma_{U^{A_{i}}} = \bigotimes_{j \in A_{i}} \Sigma_{U_{j}}$ , $\mu_{i}$ is a probability measure on the product measurable space $(\mathcal{U}_{i} \times \mathcal{U}^{A_{i}}, \Sigma_{\mathcal{U}_{i}} \otimes \Sigma_{\mathcal{U}^{A_{i}}})$ .

![](images/6264566bb10a92a441f5cace72df6a7d6ab36b2a58e417cf4acbc97d327e8ac3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Ind"] --> B["•"]
    A --> C["•"]
    B --> D["(U, Σu, ζ)"]
    C --> E["(V, Σv, χ)"]
    D --> F["M"]
    E --> F
```
</details>

Figure 5: An SCM is a functor (purple arrows) from Ind (left) to Prob (right).

# D. Category-theoretic Formalization

This section extends the category-theoretic formalization introduced in the main paper to intervened models and abstraction.

Recall the category-theoretic definition from the main paper:

Definition D.1 (Category-theoretic SCM). An SCM is a functor $M^{n}: \operatorname{Ind} \to \operatorname{Prob}$ , mapping the source node of Ind to the probability space associated with the exogenous variables $(\mathcal{U}, \Sigma_{\mathcal{U}}, \zeta)$ , the sink node of Ind to the probability space associated with the endogenous variables $(\mathcal{V}, \Sigma_{\mathcal{V}}, \chi)$ , and the only edge of Ind to the measurable map induced by the set F of functional assignments.

Fig. 5 offers a depiction of an SCM as a functor.

In the same vein, we can have a functorial representation for intervened SCMs as well. However, instead of representing directly the post-interventional model $M_{\ell}^{n}$ as in Def. D.1, we will adopt a representation that is closer to the intervention operator itself. First, notice that, whenever the domains of the variables of an SCM are continuous, we can represent an intervention as a measurable map by relying on the truncation formula (Pearl, 2009):

Lemma D.2. Given a continuous Markovian SCM $M^{n} = \langle (\mathcal{U}, \Sigma_{\mathcal{U}}, \zeta), (\mathcal{V}, \Sigma_{\mathcal{V}}, \chi), \mathcal{M} \rangle$ and an intervention $\iota$ on $M^{n}$ , there exists a measurable map $\varphi_{\iota}$ from the probability space of endogenous variables of the pre-interventional SCM $(\mathcal{V}, \Sigma_{\mathcal{V}}, \chi)$ to the probability space of endogenous variables of the post-interventional SCM $(\mathcal{V}_{\iota}, \Sigma_{\mathcal{V}_{\iota}}, \chi_{\iota})$ .

Proof. Given a Markovian SCM $M^{n}$ , the probability measure $\chi$ over the measure space of endogenous variables $(\mathcal{V}, \Sigma_{\mathcal{V}}, \chi)$ can be expressed by through the factorization over the endogenous variables $\chi = \prod_{i \in [n]} P(X_i | \mathcal{P}_i, Z_i)$ , with $\chi_i = P(X_i | \mathcal{P}_i, Z_i)$ . Given intervention $\iota = \text{do}(\mathcal{X}^\iota = \mathbf{x}^\iota)$ on $M^n$ , the new post-interventional measure $\chi^\iota$ can be computed through the truncation formula (Pearl, 2009):

$$
\chi^ {\iota} = \left\{ \begin{array}{l l} \prod_ {i \in [ n ], X _ {i} \notin \mathcal {X} ^ {\iota}} P (X _ {i} | \mathcal {P} _ {i}, Z _ {i}) & \text { if } \mathcal {X} ^ {\iota} = \mathbf {x} ^ {\iota} \\ 0 & \text { if } \mathcal {X} ^ {\iota} \neq \mathbf {x} ^ {\iota} \end{array} \right. \tag {17}
$$

We can now define a measurable map $\varphi^{\iota}$ connecting $(\mathcal{V}, \Sigma_{\mathcal{V}}, \chi)$ and $(\mathcal{V}, \Sigma_{\mathcal{V}}, \chi^{\iota})$ such that $\varphi_{\#}^{\iota}(\chi) = \chi^{\iota}$ . Specifically, for each $X_i \in \mathcal{X}^{\iota}$ , $\varphi(X_i) = x_i^{\iota}$ , thus guaranteeing the distribution on the second line of Eq. (17); for each $X_i \notin \mathcal{X}^{\iota}$ , denoting by $\chi_i^{\iota} = P(X_i|\mathcal{P}_i, Z_i)$ evaluated at $\mathcal{X}^{\iota} = \mathbf{x}^{\iota}$ as in the first line of Eq. (17), we solve a measure transport problem (Marzouk et al., 2016) from $\chi_i$ to $\chi_i^{\iota}$ which, in the continuous case, guarantees a transport map over the domains that satisfies the distribution on the first line of Eq. (17).

We can then encode an intervened model as follows:

Definition D.3 (Category-theoretic post-interventional SCM). A post-interventional SCM is a functor $M_{\iota}^{n}: \operatorname{Ind} \to \operatorname{Prob}$ , where the functor maps the source node of $\operatorname{Ind}$ to the probability space associated with the endogenous variables of the pre-interventional SCM $(\mathcal{V}, \Sigma_{\mathcal{V}}, \chi)$ , the sink node of $\operatorname{Ind}$ to the probability space associated with the endogenous variables of the post-interventional SCM $(\mathcal{V}_{\iota}, \Sigma_{\mathcal{V}_{\iota}}, \chi_{\iota})$ , and the only edge of $\operatorname{Ind}$ to the function $\varphi_{\iota}$ encoding the intervention $\iota$ .

This construction gives rise to the structure in Fig. 6 and an immediate category-theory expression of abstraction equivalent to Def.2.3:

Lemma D.4. An interventionally consistent abstraction is a singular natural transformation $\alpha$ , that is, a morphism $\alpha_{\nu}$ in Prob, that, for all intervention in I guarantees the commutativity of the diagrams constructed from Fig. 2.

![](images/2c180b08004f93c61f4bd7d10969809f32ad1c463ac20243dd5c7eae2f6b153b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Ind"] --> B["(U^ℓ, Σ_U^ℓ, ζ^ℓ)"]
    B --> C["α_U^h"]
    C --> D["(U^h, Σ_U^h, ζ^h)"]
    D --> E["α_V^h"]
    E --> F["(V^ℓ, Σ_V^ℓ, ·^ℓ)"]
    F --> G["α_V^h"]
    G --> H["(V^h, Σ_V^h, ·^h)"]
    H --> I["α_V^h"]
    I --> J["(V^h, Σ_V^h, ·^ℓ)"]
    J --> K["α_V^h"]
    K --> L["(V^ℓ, Σ_V^ℓ, ·^ℓ)"]
    L --> M["α_V^h"]
    M --> N["(V^ℓ, Σ_V^ℓ, ·^ℓ)"]
    N --> O["α_V^h"]
```
</details>

Figure 6: Representation of $M^{\ell}$ (blue), $M_{\iota}^{\ell}$ (cyan), $M^{h}$ (red), $M_{\iota}^{h}$ (orange) as functors. An abstraction is just a natural transformation, that is, a set of commuting arrows in Prob (dashed black). Notice two commuting diagrams in Prob: the first observational one rooted on the exogenous variables ( $M^{h} \circ \alpha_{U^{h}} = \alpha_{V^{h}} \circ M^{\ell}$ ), the second interventional one connecting observational and interventional model ( $M_{\kappa}^{h} \circ \alpha_{V^{h}} = \alpha_{V^{h}} \circ M_{\iota}^{\ell}$ ).

Proof. Recall the definition of interventional consistency in Def. C.1:

$$
\alpha_ {\mathcal {Y} _ {\mathcal {I}} ^ {h}} (P (\mathcal {Y} _ {\mathcal {I}} ^ {\ell} | \mathrm{do} (\mathcal {X} _ {\mathcal {I}} ^ {\ell}))) = P (\mathcal {Y} _ {\mathcal {I}} ^ {h} | \alpha_ {\mathcal {X} _ {\mathcal {I}} ^ {h}} (\mathrm{do} (\mathcal {X} _ {\mathcal {I}} ^ {h}))). \tag {18}
$$

Let us relate this definition to our categorical notation. First, $\alpha_{\mathcal{Y}_I^h}$ and $\alpha_{\mathcal{X}_I^h}$ are components of the abstraction map $\alpha$ ; in the categorical notation, this map corresponds to $\alpha_{\mathcal{V}}$ . The probability distribution $P(\mathcal{Y}_I^\ell | \mathrm{do}(\mathcal{X}_I^\ell))$ is a distribution in the low-level model; with no loss of generality, assuming $\mathcal{Y}_I^\ell$ to encompass all the non-intervened variables, this distribution corresponds to the measure $\chi_\iota^\ell$ ; furthermore, the interventional measure $\chi_\iota^\ell$ can be obtained through the pushforward of the observational measure $\chi^\ell$ via the interventional mixing functions $\mathcal{M}_\iota^\ell$ , as by Lemma D.4. Finally, the probability distribution $P(\mathcal{Y}_I^h | \alpha_{\mathcal{X}_I^h}(\mathrm{do}(\mathcal{X}_I^h))$ is a distribution in the high-level model; again, with no loss of generality, assuming $\mathcal{Y}_I^h$ to encompass all the non-intervened variables, this distribution corresponds to the measure $\chi_\kappa^h$ , where $\kappa$ is the abstraction of the terms in $\iota$ . Also, as before, the interventional measure $\chi_\kappa^h$ can be obtained through the pushforward of the observational measure $\chi^h$ via the interventional mixing functions $\mathcal{M}_\kappa^h$ , thanks to Lemma D.4. We then obtain a rewriting of abstraction as:

$$
\alpha_ {\mathcal {V}} \circ \mathcal {M} _ {\iota} ^ {\ell} = \mathcal {M} _ {\kappa} ^ {h} \circ \alpha_ {\mathcal {V}}. \tag {19}
$$

corresponding to the commutativity of the right diagram in Fig. 6, for all interventions.

# E. Stiefel Manifold

We now provide a short review of the Stiefel manifold, referring the interested reader to (Absil et al., 2008; Boumal, 2023) for a comprehensive discussion.

Given $\ell, h \in \mathbb{N}$ , $h < \ell$ , the Stiefel manifold is the set of $\ell \times h$ matrices with orthonormal columns, mathematically

$$
\operatorname{St} (\ell , h) := \left\{\mathbf {V} \in \mathbb {R} ^ {\ell \times h} \mid \mathbf {V} ^ {\top} \mathbf {V} = \mathbf {I} _ {h} \right\}. \tag {20}
$$

Consider the function $g: \mathbb{R}^{\ell \times h} \to \operatorname{Sym}(h)$ , $g(\mathbf{V}) := \mathbf{V}^{\top} \mathbf{V} - \mathbf{I}^{h}$ . It is well-known that g is a generating function for $\operatorname{St}(\ell, h)$ , thus making it an embedded submanifold of $R^{\ell \times h}$ , with dimension $\dim \mathbb{R}^{\ell \times h} - \dim \operatorname{Sym}(h) = \ell h - h(h + 1)/2$ . Given a point of the manifold V, the tangent space to $\operatorname{St}(\ell, h)$ can be defined implicitly as the kernel of the differential of g at V,

$$
T _ {\mathbf {V}} \operatorname{St} (\ell , h) := \left\{\mathbf {G} \in \mathbb {R} ^ {\ell \times h} \mid \mathbf {V} ^ {\top} \mathbf {G} + \mathbf {G} ^ {\top} \mathbf {V} = 0 \right\}. \tag {21}
$$

We consider the Riemannian metric as the restriction of the Euclidean product between two matrices in $\mathbb{R}^{\ell \times h}$ to $\mathrm{St}(\ell ,h)$ . Accordingly, given $\mathbf{A}$ , $\mathbf{B}\in T_{\mathbf{V}}\mathrm{St}(\ell ,h)$ , we have $\langle \mathbf{A},\mathbf{B}\rangle_{\mathbf{V}} = \mathrm{Tr}\mathbf{A}^{\top}\mathbf{B}$ . The tangent space linearizes the manifold around $\mathbf{V}$ , then, we can move away from $\mathbf{V}$ along the directions in $T_{\mathbf{V}}\mathrm{St}(\ell ,h)$ . However, to make such a movement smooth along the manifold, we employ the retraction map $\mathrm{R}_{\mathbf{V}}(:)T_{\mathbf{V}}\mathrm{St}(\ell ,h)\to \mathrm{St}(\ell ,h)$ . The retraction has to satisfy the following conditions

$$
(i) \mathrm{R} _ {\mathbf {V}} \left(\mathbf {0} _ {\ell \times h}\right) = \mathbf {V}, \quad \text { and } \quad (i i) \lim _ {\mathbf {G} \rightarrow \mathbf {0} _ {\ell \times h}} \frac {\| \mathrm{R} _ {\mathbf {V}} (\mathbf {G}) - (\mathbf {V} + \mathbf {G}) \| _ {\mathrm{F}}}{\| \mathbf {G} \| _ {\mathrm{F}}} = 0. \tag {22}
$$

Among the canonical retractions, we have

$$
\mathrm{R} _ {\mathbf {V}} ^ {\mathrm{QR}} (\mathbf {G}) = \operatorname{qf} (\mathbf {V} + \mathbf {G}), [ \mathrm{QRretraction} ]
$$

$$
\mathrm{R} _ {\mathbf {V}} ^ {\text { Polar }} (\mathbf {G}) = (\mathbf {V} + \mathbf {G}) \left(\mathbf {I} ^ {h} - \mathbf {V} ^ {\top} \mathbf {V}\right) ^ {\frac {1}{2}}, [ \text { Polar   retraction } ] \tag {23}
$$

$$
\mathrm{R} _ {\mathbf {V}} ^ {\mathrm{Caley}} (\mathbf {G}) = (\mathbf {I} ^ {\ell} - \frac {1}{2} \mathbf {W} (\mathbf {G})) ^ {- 1} (\mathbf {I} ^ {\ell} + \frac {1}{2} \mathbf {W} (\mathbf {G})) \mathbf {V}; [ \mathrm{Caleyretraction} ]
$$

where qf indicates the Q factor of the QR decomposition, and $\mathbf{W}(\mathbf{G}) = (\mathbf{I}^{\ell} - \frac{1}{2}\mathbf{V}\mathbf{V}^{\top})\mathbf{G}\mathbf{V}^{\top} - \mathbf{V}\mathbf{G}^{\top}(\mathbf{I}^{\ell} - \frac{1}{2}\mathbf{V}\mathbf{V}^{\top})$ .

Finally, the normal space to the manifold at $\mathbf{V}$ has the following explicit form

$$
N _ {\mathbf {V}} \operatorname{St} (\ell , h) := \left\{\mathbf {V S} \mid \mathbf {S} \in \operatorname{Sym} (h) \right\}. \tag {24}
$$

Starting from Eq. (24), the orthogonal projection to $T_{\mathbf{V}}\mathrm{St}(\ell,h)$ , namely $Proj_{V}$ , has to be such that G - $Proj_{V}G$ lies onto $N_{\mathbf{V}}\mathrm{St}(\ell,h)$ , i.e.,

$$
\mathbf {G} - \operatorname{Proj} _ {\mathbf {V}} \mathbf {G} = \mathbf {V S}. \tag {25}
$$

Plugging Eq. (25) into Eq. (21), it can be derived that

$$
\operatorname{Proj} _ {\mathbf {V}} \mathbf {G} = \left(\mathbf {I} ^ {\ell} - \mathbf {V V} ^ {\top}\right) \mathbf {G} + \mathbf {V} \frac {\left(\mathbf {V} ^ {\top} \mathbf {G} - \mathbf {G} ^ {\top} \mathbf {V}\right)}{2}. \tag {26}
$$

Finally, for $\mathrm{St}(\ell, h)$ (and in general for Riemannian submanifolds) the Riemannian gradient of $f$ at $\mathbf{V}$ is the orthogonal projection of $\nabla_{\mathbf{V}} f$ to $T_{\mathbf{V}} \mathrm{St}(\ell, h)$ . Mathematically, starting from Eq. (26), we have

$$
\widetilde {\nabla} _ {\mathbf {V}} f = \operatorname{Proj} _ {\mathbf {V}} \nabla_ {\mathbf {V}} f. \tag {27}
$$

# F. Information-theoretic Distance on Spaces of Different Dimensionality

Two types of distances can be defined as follows using an affine map $\varphi^{V,b}$ (Cai & Lim, 2022).

Definition F.1 (Embedding and projection distances). Let $\ell, h \in \mathbb{N}$ with $h \leq \ell$ , and let $\varphi^{\mathbf{V},b} = \mathbf{V}^\top x + b: \mathbb{R}^\ell \to \mathbb{R}^h$ be an affine map with $\mathbf{V} \in \mathrm{St}(\ell, h)$ and $b \in \mathbb{R}^\ell$ . For any measures $\chi^h \in \mathcal{M}(\mathbb{R}^h)$ and $\chi^\ell \in \mathcal{M}(\mathbb{R}^\ell)$ , the set of embeddings of $\chi^h$ into $\mathbb{R}^\ell$ is the set of of $\ell$ -dimensional measures, defined as follows:

$$
\Phi^ {+} \left(\chi^ {h}, \ell\right) := \left\{\alpha \in \mathcal {M} \left(\mathbb {R} ^ {\ell}\right): \varphi_ {\#} ^ {\mathbf {V}, b} (\alpha) = \chi^ {h} \right\} \tag {28}
$$

Similarly, the set of projections of $\chi^{\ell}$ into $\mathbb{R}^{h}$ is the set of $h$ -dimensional measures defined as:

$$
\Phi^ {-} \left(\chi^ {\ell}, h\right) := \left\{\beta \in \mathcal {M} \left(\mathbb {R} ^ {h}\right): \varphi_ {\#} ^ {\mathbf {V}, b} \left(\chi^ {\ell}\right) = \beta \right\} \tag {29}
$$

Now, for any given distance measure $D(\cdot,\cdot)$ defined in $\mathcal{M}(\mathbb{R}^{\ell})$ , we can define the embedding distance $D^{+}(\chi^{h},\chi^{\ell}):=\inf_{\alpha\in\Phi^{+}(\chi^{h},\ell)}D(\alpha,\nu)$ and the projection distance $D^{-}(\chi^{h},\chi^{\ell}):=\inf_{\beta\in\Phi^{-}(\chi^{\ell},h)}D(\chi^{h},\beta)$ .

Embedding and projection distances can measure distances between probability measures of different dimensions. Additionally, Theorem I.2 (Cai & Lim, 2022) states that the former two distances are equivalent, that is, for a number of different distance metrics and $\phi$ -divergences, $D^{+}(\chi^{h},\chi^{\ell}) = D^{-}(\chi^{h},\chi^{\ell}) = \hat{D} (\chi^{h},\chi^{\ell})$ , implying that computing the embedding distance or the projection distance yields the same result.

# G. Proofs

Theorem 4.3. Let $\chi^{\ell} \sim N(\mathbf{0}_{\ell}, \boldsymbol{\Sigma}^{\ell})$ , $\chi^{h} \sim N(\mathbf{0}_{h}, \boldsymbol{\Sigma}^{h})$ , where $\boldsymbol{\Sigma}^{\ell} \in S_{++}^{\ell}$ and $\boldsymbol{\Sigma}^{h} \in S_{++}^{h}$ . Denote by $0 < \lambda_{1} \leq \ldots \leq \lambda_{\ell}$ the eigenvalues of $\boldsymbol{\Sigma}^{\ell}$ , and by $0 < \kappa_{1} \leq \ldots \leq \kappa_{h}$ those of $\boldsymbol{\Sigma}^{h}$ . If a linear CA $\mathbf{V} \in \mathrm{St}(\ell, h)$ complying with SEP from $\chi^{\ell}$ to $\chi^{h}$ exists, then

$$
\lambda_ {i} \leq \kappa_ {i} \leq \lambda_ {i + \ell - h}, \quad \forall i \in [ h ]. \tag {4}
$$

Proof. If a linear CA $\mathbf{V} \in \mathrm{St}(\ell, h)$ exists, then $\chi^h = \varphi_{\#}^{\mathbf{V}^\top}(\chi^\ell)$ . This implies that the kernel of any information-theoretic metric or $\phi$ -divergence is nonempty, and also $\mathbf{V}^\top \boldsymbol{\Sigma}^\ell \mathbf{V} = \boldsymbol{\Sigma}^h$ . The latter determines the eigenvalues of $\mathbf{V}^\top \boldsymbol{\Sigma}^\ell \mathbf{V}$ are those of $\boldsymbol{\Sigma}^h$ . By the Ostrowski's theorem for rectangular $\mathbf{V}$ — cf. Th.3.2 in (Higham & Cheng, 1998) — we have

$$
\kappa_ {i} = \vartheta_ {i} \mu_ {i}, \quad \text { with } i \in [ h ], \tag {30}
$$

where

$$
\lambda_ {i} \leq \mu_ {i} \leq \lambda_ {i + \ell - h}; \tag {31}
$$

and

$$
\operatorname{eigvls} (\mathbf {V} ^ {\top} \mathbf {V}) _ {1} \leq \vartheta_ {i} \leq \operatorname{eigvls} (\mathbf {V} ^ {\top} \mathbf {V}) _ {h}. \tag {32}
$$

Since $\mathbf{V}\in\mathrm{St}(\ell,h)$ , by Eq. (32) $\vartheta_{i}=1$ for each $i\in[h]$ . Substituting the latter into Eq. (30), we get $\kappa_{i}=\mu_{i}$ , thus obtaining Eq. (4) by Eq. (31).

Proposition 5.1. Consider the function

$$
f (\mathbf {A}) = \operatorname{Tr} \left\{\left(\mathbf {A} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {A}\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} + \log \det \left\{\mathbf {A} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \right\}. \tag {9}
$$

Eq. (9) is smooth for $\mathbf{A} \in \mathrm{St}(\ell, h)$ . Additionally, define $\widetilde{\mathbf{A}} := \left(\mathbf{A}^{\top} \boldsymbol{\Sigma}^{\ell} \mathbf{A}\right)^{-1}$ . The gradient of $f(\mathbf{A})$ is

$$
\nabla_ {\mathbf {A}} f = 2 \left(\boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \widetilde {\mathbf {A}}\right) \left(\mathbf {I} _ {h} - \boldsymbol {\Sigma} ^ {h} \widetilde {\mathbf {A}}\right), \tag {10}
$$

Proof. Consider the first term $\mathrm{Tr}\left\{\left(\mathbf{A}^{\top}\boldsymbol{\Sigma}^{\ell}\mathbf{A}\right)^{-1}\boldsymbol{\Sigma}^{h}\right\}$ in Eq. (9). We have that $\mathrm{Tr}\left\{\left(\mathbf{A}^{\top}\boldsymbol{\Sigma}^{\ell}\mathbf{A}\right)^{-1}\boldsymbol{\Sigma}^{h}\right\}$ is well-defined and smooth in case $\mathbf{A}^{\top}\boldsymbol{\Sigma}^{\ell}\mathbf{A}$ is positive definite (Boyd & Vandenberghe, 2004). If $\mathbf{A}^{\top}\boldsymbol{\Sigma}^{\ell}\mathbf{A}\in\mathcal{S}_{++}^{h}$ , for all $\mathbf{y}\in\mathbb{R}^{h}$ , $\mathbf{y}\neq\mathbf{0}_{h}$ , it holds $\mathbf{y}^{\top}\mathbf{A}^{\top}\boldsymbol{\Sigma}^{\ell}\mathbf{A}\mathbf{y}>0$ . By defining $\mathbb{R}^{\ell}\ni\mathbf{z}:=\mathbf{A}\mathbf{y}$ , this is equivalent to say $\mathbf{z}^{\top}\boldsymbol{\Sigma}^{\ell}\mathbf{z}>0,\forall\mathbf{y}\neq\mathbf{0}_{h}$ . Since $\boldsymbol{\Sigma}^{\ell}\in\mathcal{S}_{++}^{\ell}$ by assumption, we have to prove that $\mathbf{z}\neq\mathbf{0}_{\ell},\forall\mathbf{y}\neq\mathbf{0}_{h}$ . Consider that exists $\tilde{\mathbf{y}}\neq\mathbf{0}_{h}$ such that $\tilde{\mathbf{z}}=\mathbf{A}\tilde{\mathbf{y}}=\mathbf{0}_{\ell}$ . This means that

$$
\mathbf {0} _ {h} = \mathbf {A} ^ {\top} \tilde {\mathbf {z}} = \mathbf {A} ^ {\top} \mathbf {A} \tilde {\mathbf {y}} = \tilde {\mathbf {y}} \neq \mathbf {0} _ {h}; \tag {33}
$$

which is a contradiction. Hence $A^{\top}\Sigma^{\ell}A\in S_{++}^{h}$ and $\operatorname{Tr}\left\{\left(\mathbf{A}^{\top}\boldsymbol{\Sigma}^{\ell}\mathbf{A}\right)^{-1}\boldsymbol{\Sigma}^{h}\right\}$ is smooth over $\operatorname{St}(\ell,h)$ . Consider now $\log\det\left\{\mathbf{A}^{\top}\boldsymbol{\Sigma}^{\ell}\mathbf{A}\right\}$ in Eq. (8). Since $A^{\top}\Sigma^{\ell}A\in S_{++}^{h}$ , also this latter term is well-defined and smooth.

Let $\widetilde{\mathbf{A}} := (\mathbf{A}^{\top} \mathbf{\Sigma}^{\ell} \mathbf{A})^{-1}$ . The gradient in Eq. (10) follows from the application of the following rules of matrix calculus (Brookes, 2020),

$$
(i) \quad \nabla \operatorname{Tr} \left\{\left(\mathbf {A} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {A}\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} = - 2 \boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \widetilde {\mathbf {A}} \boldsymbol {\Sigma} ^ {h} \widetilde {\mathbf {A}} \quad \text { and } \quad (i i) \quad \nabla \log \det \left\{\mathbf {A} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \right\} = 2 \boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \widetilde {\mathbf {A}}, \tag {34}
$$

# H. LinSEPAL-ADMM

Let us recall below the nonsmooth Riemannian problem we have to solve.

Problem 2. Given $\Sigma^{\ell} \in S_{++}^{\ell}$ , $\Sigma^h \in S_{++}^h$ , $\mathbf{D} \in \{0,1\}^{\ell \times h}$ , and $\lambda \in \mathbb{R}_{+}$ , the CA is the transpose of

$$
\mathbf {V} ^ {\star} = \underset {\mathbf {V} \in \mathrm{St} (\ell , h)} {\arg \min} f (\mathbf {V}) + \lambda \underbrace {\left\| \mathbf {D} \odot \mathbf {V} \right\| _ {1}} _ {h (\mathbf {V})}. \tag {5}
$$

Here, $f(\mathbf{V})$ follows Eq. (3), omitting the constant $C$ .

The structure of the objective in (5), separating into smooth (cf. Proposition 5.1) and nonsmooth terms, makes the alternating direction method of multipliers (ADMM, Boyd et al., 2011) an appealing optimization framework for deriving a solution. This is the rationale behind the general framework manifold ADMM (Kovnatsky et al., 2016), that we decline to our setting in the following, thus obtaining the LinSEPAL-ADMM algorithm.

Starting from (5), we add a splitting variable $\mathbf{Y} \in \mathbb{R}^{\ell \times h}$ to be optimized over the Euclidean space to handle the non-smooth term $h(\mathbf{V})$ :

$$
\min _ {\mathbf {V} \in \operatorname{St} (\ell , h), \mathbf {Y} \in \mathbb {R} ^ {\ell \times h}} \quad \operatorname{Tr} \left\{\left(\mathbf {V} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {V}\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} + \log \det \left\{\mathbf {V} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {V} \right\} + \lambda \| \mathbf {Y} \| _ {1}, \tag {P2}
$$

subject to $Y - D \odot V = 0_{\ell \times h}$ .

At this point, following (Boyd et al., 2011), by denoting by $\mathbf{U} \in \mathbb{R}^{\ell \times h}$ the scaled dual variable, and by $\rho \in \mathbb{R}^{+}$ the ADMM stepsize, the scaled augmented Lagrangian reads as

$$
L _ {\rho} (\mathbf {V}, \mathbf {Y}, \mathbf {U}) = \operatorname{Tr} \left\{\left(\mathbf {V} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {V}\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} + \log \det \left\{\mathbf {V} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {V} \right\} + \lambda \| \mathbf {Y} \| _ {1} + \frac {\rho}{2} \| \mathbf {D} \odot \mathbf {V} - \mathbf {Y} + \mathbf {U} \| _ {\mathrm{F}} ^ {2}. \tag {35}
$$

Starting from Eq. (35), the ADMM updates at the k-th iteration are

$$
\mathbf {V} ^ {k + 1} = \underset {\mathbf {V} \in \mathrm{St} (\ell , h)} {\arg \min} L _ {\rho} \left(\mathbf {V}, \mathbf {Y} ^ {k}, \mathbf {U} ^ {k}\right),
$$

$$
\mathbf {Y} ^ {k + 1} = \underset {\mathbf {Y} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} L _ {\rho} \left(\mathbf {V} ^ {k + 1}, \mathbf {Y}, \mathbf {U} ^ {k}\right), \tag {R1}
$$

$$
\mathbf {U} ^ {k + 1} = \mathbf {U} ^ {k} + \mathbf {D} \odot \mathbf {V} ^ {k + 1} - \mathbf {Y} ^ {k + 1}.
$$

Solution for $\mathbf{V}^{k + 1}$

The update for $V^{k+1}$ in (R1) reads as

$$
\mathbf {V} ^ {k + 1} = \underset {\mathbf {V} \in \operatorname{St} (\ell , h)} {\arg \min} \operatorname{Tr} \left\{\left(\mathbf {V} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {V}\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} + \log \det \left\{\mathbf {V} ^ {\top} \boldsymbol {\Sigma} ^ {\ell} \mathbf {V} \right\} + \frac {\rho}{2} \| \mathbf {D} \odot \mathbf {V} - \mathbf {Y} ^ {k} + \mathbf {U} ^ {k} \| _ {\mathrm{F}} ^ {2} \tag {36}
$$

Eq. (36) is a standard smooth optimization problem over the Stiefel manifold, and it can be solved by standard techniques such as those in (Boumal, 2023). Newton and conjugate gradient methods for the Stiefel manifold are discussed in (Edelman et al., 1998). In our experiments, we use the conjugate gradient implementation in (Boumal et al., 2014).

Solution for $Y^{k+1}$ . The update for $Y^{k+1}$ in (R1) reads as

$$
\mathbf {Y} ^ {k + 1} = \underset {\mathbf {Y} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} \lambda \| \mathbf {Y} \| _ {1} + \frac {\rho}{2} \| \mathbf {D} \odot \mathbf {V} ^ {k + 1} - \mathbf {Y} + \mathbf {U} ^ {k} \| _ {\mathrm{F}} ^ {2} = \tag {37}
$$

$$
= \mathcal {S} _ {\lambda / \rho} \left(\mathbf {D} \odot \mathbf {V} ^ {k + 1} + \mathbf {U} ^ {k}\right);
$$

where $\mathcal{S}_{\delta}(x)=\mathrm{sign}(x)\cdot\max(|x|-\delta,0)$ is the element-wise soft-thresholding operator (Parikh et al., 2014).

Stopping criteria. The empirical convergence of LinSEPAL-ADMM is established according to primal and dual feasibility optimality conditions (Boyd et al., 2011). The primal residual, associated with the equality constraint in Eq. (P2), is

$$
\mathbf {R} _ {p} ^ {k + 1} := \mathbf {Y} ^ {k + 1} - \mathbf {D} \odot \mathbf {V} ^ {k + 1}. \tag {38}
$$

The dual residual, which can be obtained from the stationarity condition, is

$$
\mathbf {R} _ {d} ^ {k + 1} := \rho \mathbf {D} \odot \left(\mathbf {Y} ^ {k + 1} - \mathbf {Y} ^ {k}\right). \tag {39}
$$

As $k \to \infty$ , the norm of the primal and dual residuals should vanish. Hence, the stopping criterion can be set in terms of the norms

$$
(i) d _ {p} ^ {k + 1} = \left\| \mathbf {R} _ {p} ^ {k + 1} \right\| _ {\mathrm{F}} \quad \text { and } \quad (i i) d _ {d} ^ {k + 1} = \left\| \mathbf {R} _ {d} ^ {k + 1} \right\| _ {\mathrm{F}}. \tag {40}
$$

Specifically, given absolute and relative tolerance, namely $\tau^a$ and $\tau^r$ in $\mathbb{R}_+$ , respectively, convergence in practice is established following Boyd et al. (2011) when

$$
(i) d _ {p} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \max \left(\left\| \mathbf {Y} ^ {k + 1} \right\| _ {\mathrm{F}}, \left\| \mathbf {D} \circ \mathbf {V} ^ {k + 1} \right\| _ {\mathrm{F}}\right), \quad \text { and } \quad (i i) d _ {d} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \rho \left\| \mathbf {D} \circ \mathbf {U} ^ {k + 1} \right\| _ {\mathrm{F}}. \tag {41}
$$

The LinSEPAL-ADMM algorithm is summarized in Algorithm 1.

Algorithm 1 LinSEPAL-ADMM   
1: Input: $\Sigma^{\ell}$ , $\Sigma^{h}$ , D, $\lambda$ , $\rho$ , $\tau^{a}$ , $\tau^{r}$ 2: Initialize: $\mathbf{V}^{0} \in \mathrm{St}(\ell, h)$ , $\mathbf{Y}^{0} \in \mathbb{R}^{\ell \times h}$ , $\mathbf{U}^{0} \leftarrow \mathbf{D} \odot \mathbf{V}^{0} - \mathbf{Y}^{0}$ 3: repeat
4: $\mathbf{V}^{k+1} \leftarrow$ Solve Eq. (36) via an off-the-shelf method for smooth Riemannian problems
5: $\mathbf{Y}^{k+1} \leftarrow \mathcal{S}_{\lambda/\rho} (\mathbf{D} \odot \mathbf{V}^{k+1} + \mathbf{U}^{k})$ 6: $\mathbf{U}^{k+1} \leftarrow \mathbf{U}^{k} + \mathbf{D} \odot \mathbf{V}^{k+1} - \mathbf{Y}^{k+1}$ 7: until Eq. (41) is satisfied
8: Output: V, Y, U

# I. LinSEPAL-PG

This method is based upon the manifold proximal gradient (Chen et al., 2020) framework, which generalizes the proximal gradient framework defined in the Euclidean space to the Stiefel manifold. Following Chen et al. (2020), denoting by $\mathbf{V}^k$ the iterate at the step $k$ , the updates recursion for solving (5) reads as

$$
\mathbf {G} ^ {k} = \underset {\mathbf {G} \in T _ {\mathbf {V} ^ {k}} \operatorname{St} (\ell , h)} {\arg \min} \quad \left\langle \nabla f \left(\mathbf {V} ^ {k}\right), \mathbf {G} \right\rangle + \frac {1}{2 \rho} \| \mathbf {G} \| _ {\mathrm{F}} ^ {2} + \lambda \| \mathbf {D} \odot \left(\mathbf {V} ^ {k} + \mathbf {G}\right) \| _ {1}, \tag {R2}
$$

$$
\mathbf {V} ^ {k + 1} = \mathrm{R} _ {\mathbf {V} ^ {k}} \left(\mathbf {G} ^ {k}\right).
$$

In (R2), the first update is the proximal mapping providing a proximal gradient direction $G^{k}$ onto the tangent space to the Stiefel manifold, using the first-order approximation of the objective around the k-th estimate. The second is the update for $V^{k+1}$ , which exploits the canonical retraction (cf. Eq. (23)) technique for projecting back $V^{k} + G^{k}$ from the tangent space to the manifold. Global convergence of the ManPG method has been established in Chen et al. (2020).

Solution for $G^{k}$ . Chen et al. (2020) shows that the first update can be efficiently solved using the regularized semi-smooth Newton method in Xiao et al. (2018). Specifically, according to Eq. (21), the feasible set $T_{\mathbf{V}^{k}}\mathrm{St}(\ell,h)$ translates into a linear constraint. By defining $\mathcal{A}^{k}\left(\mathbf{G}\right):=\mathbf{G}^{\top}\mathbf{V}^{k}+\mathbf{V}^{k^{\top}}\mathbf{G}$ , the update is

$$
\mathbf {G} ^ {k} = \underset {\mathbf {G} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} \quad \left\langle \nabla f (\mathbf {V} ^ {k}), \mathbf {G} \right\rangle + \frac {1}{2 \rho} \| \mathbf {G} \| _ {\mathrm{F}} ^ {2} + \lambda \| \mathbf {D} \odot (\mathbf {V} ^ {k} + \mathbf {G}) \| _ {1}, \tag {42}
$$

subject to $\mathcal{A}^{k}\left(\mathbf{G}\right)=\mathbf{0}_{h\times h}$ .

However, following the rationale in Si et al. (2024), we can force $\mathbf{G}^{k}\in T_{\mathbf{V}^{k}}\mathrm{St}(\ell,h)$ by exploiting the basis $B_{V^{k}}$ of the normal space to the manifold, namely $N_{\mathbf{V}^{k}}\mathrm{St}(\ell,h)$ . To find such $B_{V^{k}}$ , recall the explicit form of $N_{\mathbf{V}}\mathrm{St}(\ell,h)$ in Eq. (24).

The basis of $\operatorname{Sym}(h)$ , having dimension $s = h(h + 1)/2$ , is

$$
\mathcal {E} := \left\{\mathbf {E} _ {i j} \in \{0, 1 \} ^ {h \times h} \mid \mathbf {E} _ {i j} \text {   has   } e _ {i j} = e _ {j i} = 1, 0 \text {   elsewhere }, 1 \leq i \leq j \leq h \right\}. \tag {43}
$$

It follows from Eqs. (24) and (43) that

$$
\mathcal {B} _ {\mathbf {V} ^ {k}} := \left\{\mathbf {B} _ {i j} ^ {k} = \mathbf {V} ^ {k} \mathbf {E} _ {i j}, 1 \leq i \leq j \leq h \right\}. \tag {44}
$$

At this point, the membership to $T_{\mathbf{V}^{k}}\mathrm{St}(\ell,h)$ can be expressed as

$$
\left\langle \mathbf {B} _ {i j} ^ {k}, \mathbf {G} \right\rangle = 0, \quad \forall 1 \leq i \leq j \leq h. \tag {45}
$$

Hence, (42) reads as

$$
\mathbf {G} ^ {k} = \underset {\mathbf {G} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} \quad \left\langle \nabla f (\mathbf {V} ^ {k}), \mathbf {G} \right\rangle + \frac {1}{2 \rho} \| \mathbf {G} \| _ {\mathrm{F}} ^ {2} + \lambda \| \mathbf {D} \odot (\mathbf {V} ^ {k} + \mathbf {G}) \| _ {1}, \tag {46}
$$

subject to $\langle \mathbf{B}_{ij}^{k},\mathbf{G}\rangle = 0,\quad \forall 1\leq i\leq j\leq h$

Consider $h\left(\mathbf{V}^k + \mathbf{G}\right) = \left\| \mathbf{D} \odot \left(\mathbf{V}^k + \mathbf{G}\right) \right\|_1$ and $\mathbb{R}^s \ni \boldsymbol{\mu} = [\mu_{11}, \mu_{12}, \ldots, \mu_{ij}, \ldots, \mu_{hh}]$ , with $1 \leq i \leq j \leq h$ . The Lagrangian for (46) is

$$
L _ {\rho} (\mathbf {G}, \boldsymbol {\mu}) = \left\langle \nabla f \left(\mathbf {V} ^ {k}\right), \mathbf {G} \right\rangle + \frac {1}{2 \rho} \| \mathbf {G} \| _ {\mathrm{F}} ^ {2} + \lambda h \left(\mathbf {V} ^ {k} + \mathbf {G}\right) - \sum_ {1 \leq i \leq j \leq h} \mu_ {i j} \left\langle \mathbf {B} _ {i j} ^ {k}, \mathbf {G} \right\rangle . \tag {47}
$$

Let us define now the matrix $R^{s\times\ell h}\ni B^{k}:=[\operatorname{vec}\left(\mathbf{B}_{11}^{k}\right),\operatorname{vec}\left(\mathbf{B}_{12}^{k}\right),\ldots,\operatorname{vec}\left(\mathbf{B}_{hh}^{k}\right)]^{\top}$ , where $\operatorname{vec}\left(\mathbf{B}_{ij}^{k}\right)\in\mathbb{R}^{\ell h}$ . We can compactly express the s equality constraints as

$$
\mathbf {B} ^ {k} \operatorname{vec} (\mathbf {G}) = \mathbf {0} _ {s}. \tag {48}
$$

Thus, the Karush-Kuhn-Tacker (KKT) conditions of Eq. (42) reads as

$$
(i) \mathbf {0} _ {l \times h} \in \partial_ {\mathbf {G}} L _ {\rho} (\mathbf {G}, \boldsymbol {\mu}), \quad \text { and } \quad (i i) \mathbf {B} ^ {k} \operatorname{vec} (\mathbf {G}) = \mathbf {0} _ {s}. \tag {49}
$$

From the stationarity condition we get

$$
\mathbf {0} _ {l \times h} \in \mathbf {G} + \rho \left(\nabla f \left(\mathbf {V} ^ {k}\right) - \sum_ {1 \leq i \leq j \leq h} \mu_ {i j} \mathbf {B} _ {i j} ^ {k}\right) + \lambda \rho \partial_ {\mathbf {G}} h \left(\mathbf {V} ^ {k} + \mathbf {G}\right). \tag {50}
$$

At this point, recalling the inclusion property of proximal operators, viz. $\mathbf{P} = \text{prox}_{g}(\mathbf{B}) \iff \mathbf{B} - \mathbf{P} \in \partial g(\mathbf{P})$ , we have

$$
\mathbf {0} _ {l \times h} \in \underbrace {\mathbf {V} ^ {k} + \mathbf {G}} _ {\mathbf {P}} - \underbrace {\left(\mathbf {V} ^ {k} - \rho \left(\nabla f (\mathbf {V} ^ {k}) - \sum_ {1 \leq i \leq j \leq h} \mu_ {i j} \mathbf {B} _ {i j} ^ {k}\right)\right)} _ {\mathbf {B} (\boldsymbol {\mu})} + \lambda \rho \partial_ {\mathbf {G}} h \underbrace {(\mathbf {V} ^ {k} + \mathbf {G})} _ {\mathbf {P}}; \tag {51}
$$

from which we get

$$
\mathbf {G} (\boldsymbol {\mu}) = \operatorname{prox} _ {\lambda \rho h (\cdot)} \left(\mathbf {B} (\boldsymbol {\mu})\right) - \mathbf {V} ^ {k}. \tag {52}
$$

At this point, $\operatorname{prox}_{\lambda \rho h(\cdot)}$ can be computed element-wise as

$$
\operatorname{prox} _ {\lambda \rho h (\cdot)} (b _ {i j} (\boldsymbol {\mu})) = \left\{ \begin{array}{l l} b _ {i j} (\boldsymbol {\mu}), & \text { if } d _ {i j} = 0, \\ \mathcal {S} _ {\lambda \rho} (b _ {i j} (\boldsymbol {\mu})), & \text { otherwise }. \end{array} \right. \tag {53}
$$

Substituting Eq. (52) into Eq. (49), we have

$$
\mathbf {B} ^ {k} \operatorname{vec} (\mathbf {G} (\boldsymbol {\mu})) = \mathbf {0} _ {s}. \tag {54}
$$

Here the $r$ -th entry of $\operatorname{vec}(\mathbf{G}(\boldsymbol{\mu}))$ corresponds to the entry of $\mathbf{G}(\boldsymbol{\mu})$ at $\operatorname{row} u = (r - 1) \bmod \ell + 1$ , and column $v = \lfloor (r - 1)/\ell \rfloor + 1$ , $r \in [\ell h]$ .

At this point, we can use the regularized semi-smooth Newton method (Xiao et al., 2018) to solve Eq. (54). Our target function is

$$
F (\boldsymbol {\mu}) = \mathbf {B} ^ {k} \operatorname{vec} \left(\mathbf {G} (\boldsymbol {\mu})\right): \mathbb {R} ^ {s} \rightarrow \mathbb {R} ^ {s}. \tag {55}
$$

By the chain rule of calculus, using Eq. (52), the generalized Jacobian matrix is

$$
\mathbb {R} ^ {s \times s} \ni \mathbf {J} = \frac {\partial F (\boldsymbol {\mu})}{\partial \operatorname{vec} (\mathbf {G} (\boldsymbol {\mu}))} \cdot \frac {\partial \operatorname{vec} (\mathbf {G} (\boldsymbol {\mu}))}{\partial \boldsymbol {\mu}} \tag {56}
$$

$$
= \mathbf {B} ^ {k} \frac {\partial \mathrm{prox} _ {\lambda \rho h (\cdot)} \left(\mathrm{vec} \left(\mathbf {B} (\boldsymbol {\mu})\right)\right)}{\partial \mathrm{vec} \left(\mathbf {B} (\boldsymbol {\mu})\right)} \cdot \frac {\partial \mathrm{vec} \left(\mathbf {B} (\boldsymbol {\mu})\right)}{\partial \boldsymbol {\mu}}.
$$

The proximal-related term is a diagonal matrix $\mathbf{M} \in \mathbb{R}^{\ell h \times \ell h}$ , where

$$
m _ {r r} = \left\{ \begin{array}{l l} 1, & \text { if   } \operatorname{vec} (\mathbf {D}) _ {r} = 0 \text {   or   } (\operatorname{vec} (\mathbf {D}) _ {r} = 1 \text {   and   } | b _ {r} | - \lambda \rho > 0) \\ 0, & \text { otherwise. } \end{array} \right. \tag {57}
$$

Additionally, starting from

$$
\begin{array}{l} b _ {r} (\boldsymbol {\mu}) = \operatorname{vec} \left(\mathbf {V} ^ {k} - \rho \left(\nabla f \left(\mathbf {V} ^ {k}\right) - \sum_ {1 \leq i \leq j \leq h} \mu_ {i j} \left[ \mathbf {B} _ {i j} ^ {k} \right] _ {u v}\right)\right) \tag {58} \\ = \operatorname{vec} \left(\mathbf {V} ^ {k} - \rho \left(\nabla f (\mathbf {V} ^ {k}) - \mathbf {b} _ {u v} ^ {k ^ {\top}} \boldsymbol {\mu}\right)\right). \\ \end{array}
$$

Hence, we get

$$
\mathbb {R} ^ {s} \ni \frac {\partial b _ {r} (\boldsymbol {\mu})}{\partial \boldsymbol {\mu}} = \mathbf {b} _ {u v} ^ {k}. \tag {59}
$$

Consequently, starting from Eq. (56), using Eqs. (57) and (59), we finally have

$$
\mathbf {J} = \mathbf {B} ^ {k} \mathbf {M C}, \quad \text { with } \mathbb {R} ^ {\ell h \times s} \ni \mathbf {C} = \left( \begin{array}{c} \mathbf {b} _ {1 1} ^ {k ^ {\top}} \\ \mathbf {b} _ {2 1} ^ {k ^ {\top}} \\ \vdots \\ \mathbf {b} _ {\ell h} ^ {k ^ {\top}} \end{array} \right). \tag {60}
$$

Following Xiao et al. (2018), denoting with $\nu^k = \alpha^k\| F^k\| _2$ , $\alpha^k\in \mathbb{R}^+$ , we define

$$
r ^ {k} := \left(\mathbf {J} ^ {k - 1} + \nu^ {k - 1} \mathbf {I} _ {s}\right) \mathbf {d} ^ {k} + F ^ {k - 1}. \tag {61}
$$

At each iteration we want to find the step $d^{k}$ by solving Eq. (61) inexactly, such that

$$
\left\| r ^ {k} \right\| _ {2} \leq \tau \min \left(1, \alpha^ {k - 1} \left\| F ^ {k - 1} \right\| _ {2} \left\| \mathbf {d} ^ {k} \right\| _ {2}\right), \quad \tau \in (0, 1); \tag {62}
$$

obtaining a trial point

$$
\mathbf {u} ^ {k} = \boldsymbol {\mu} ^ {k - 1} + \mathbf {d} ^ {k}. \tag {63}
$$

Let $\beta^0 = \| F(\pmb{\mu}^0)\| _2$ and $\gamma \in (0,1)$ . If $\| F(\mathbf{u}^k)\| _2\leq \gamma \beta^{k - 1}$ then we set

$$
\boldsymbol {\mu} ^ {k} = \mathbf {u} ^ {k}, \beta^ {k} = \left\| F (\mathbf {u} ^ {k}) \right\| _ {2}, \text { and } \alpha^ {k} = \alpha^ {k - 1}. [ \text { Newton   step } ] \tag {64}
$$

Otherwise, let

$$
\xi^ {k} = \frac {- F (\mathbf {u} ^ {k}) ^ {\top} \mathbf {d} ^ {k}}{\| \mathbf {d} ^ {\mathbf {k}} \| _ {2} ^ {2}}. \tag {65}
$$

Select $0 < \phi_{1} \leq \phi_{2} < 1$ and $1 < \psi_{1} < \psi_{2}$ . Hence, we make a safeguard step as follows

$$
\boldsymbol {\mu} ^ {k} = \left\{ \begin{array}{l l} \mathbf {v} ^ {k}, & \text { if } \xi^ {k} \geq \phi_ {1} \text { and } \| F (\mathbf {v} ^ {k}) \| _ {2} \leq \| F (\boldsymbol {\mu} ^ {k - 1}) \| _ {2}, [ \text { projection   step } ] \\ \mathbf {w} ^ {k}, & \text { if } \xi^ {k} \geq \phi_ {1} \text { and } \| F (\mathbf {v} ^ {k}) \| _ {2} > \| F (\boldsymbol {\mu} ^ {k - 1}) \| _ {2}, [ \text { fixed - point   step } ] \\ \boldsymbol {\mu} ^ {k - 1}, & \text { if } \xi^ {k} <   \phi_ {1}, \text { unsuccessful   step } \end{array} \right. \tag {66}
$$

where

$$
\mathbf {v} ^ {k} = \boldsymbol {\mu} ^ {k - 1} - \frac {F (\mathbf {u} ^ {k}) ^ {\top} (\boldsymbol {\mu} ^ {k - 1} - \mathbf {u} ^ {k})}{\| F (\mathbf {u} ^ {k}) \| _ {2}} F (\mathbf {u} ^ {k}), \mathbf {w} ^ {k} = \boldsymbol {\mu} ^ {k - 1} - \delta F (\boldsymbol {\mu} ^ {k}), \delta \in \left(0, \frac {1}{\omega}\right); \tag {67}
$$

where $\omega \in (0,1]$ . Finally, denoting $\mathbb{R}^{+} \ni \bar{\alpha} \approx 0$ , the parameters $\beta^{k+1}$ and $\alpha^{k+1}$ are updated as

$$
\beta^ {k} = \beta^ {k - 1}, \quad \alpha^ {k} \in \left\{ \begin{array}{l l} (\bar {\alpha}, \alpha^ {k - 1}), & \text { if } \xi^ {k} \geq \phi_ {2}, \\ [ \alpha^ {k - 1}, \psi_ {1} \alpha^ {k - 1} ], & \text { if } \phi_ {1} \leq \xi^ {k} <   \phi_ {2}, \\ (\psi_ {1} \alpha^ {k - 1}, \psi_ {2} \alpha^ {k - 1} ], & \text { otherwise }. \end{array} \right. \tag {68}
$$

At this point, we set $\mathbf{G}^k = \mathbf{G}^k (\pmb{\mu}^k)$ according to Eq. (52).

Solution for $V^{k+1}$ . Given $V^{k} + G^{k} \in T_{V^{k}}\mathrm{St}(\ell, h)$ , we have to project the point onto the manifold. This can be accomplished via the canonical retractions in Eq. (23). However, as suggested by Chen et al. (2020), our LinSEPAL-PG implementation performs an Armijo line-search procedure to determine the stepsize a. Hence, the update is

$$
\mathbf {V} ^ {k + 1} = \mathrm{R} _ {\mathbf {V} ^ {k}} ^ {\mathrm{QR}} \left(a \mathbf {G} ^ {k}\right). \tag {69}
$$

Stopping criteria. Empirical convergence of the LinSEPAL-PG algorithm is established either when a maximum number of iterations K is reached, or when the $D_{V^{k+1}}^{KL}$ is below a certain threshold $\tau^{KL} \approx 0$ . The LinSEPAL-PG algorithm is summarized in Algorithm 2.

Algorithm 2 LinSEPAL-PG   
1: Input: $\Sigma^{\ell}$ , $\Sigma^{h}$ , D, $\lambda$ , $\rho$ , $\gamma \in (0, 1)$ , $\tau^{\mathrm{KL}}$ , $K$ 2: Initialize: $\mathbf{V}^{0} \in \mathrm{St}(\ell, h)$ , $\mathbf{Y}^{0} \in \mathbb{R}^{\ell \times h}$ , $\mathbf{U}^{0} \in \mathbb{R}^{\ell \times h}$ 3: repeat
4: $\mathbf{G}^{k} \leftarrow$ Solve Eq. (46) via the regularized semi-smooth Newton method
5: $a \leftarrow 1$ 6: repeat
7: $a = \gamma a$ 8: $\bar{\mathbf{V}} = \mathrm{R}_{\mathbf{V}^{k}}^{\mathrm{QR}}(a\mathbf{G}^{k})$ 9: until $D_{\bar{\mathbf{V}}}^{\mathrm{KL}} > D_{\mathbf{V}^{k}}^{\mathrm{KL}} - \frac{a\|\mathbf{G}^{k}\|_{\mathrm{F}}^{2}}{2\rho}$ 10: $\mathbf{V}^{k+1} \leftarrow \bar{\mathbf{V}}$ 11: until $k > K$ or $D_{\mathbf{V}^{k+1}}^{\mathrm{KL}} < \tau^{\mathrm{KL}}$ 12: Output: V

# J. CLinSEPAL

The problem we want to solve is:

Problem 3. Given $\Sigma^{\ell} \in S_{++}^{\ell}$ , $\Sigma^{h} \in S_{++}^{h}$ , and $\mathbf{B} \in \{0,1\}^{\ell \times h}$ , the linear constructive CA is given by the transpose of the product $\mathbf{B} \odot \mathbf{S} \odot \mathbf{V}$ , where

$$
\mathbf{V}^{\star},\mathbf{S}^{\star} = \operatorname *{arg  min}_{\substack{\mathbf{V}\in \mathbb{R}^{\ell \times h}\\ \mathbf{S}\in [0,1]^{\ell \times h}}}f(\mathbf{V},\mathbf{S});
$$

subject to (i) $\mathbf{B} \odot \mathbf{S} \odot \mathbf{V} \in \mathrm{St}(\ell, h)$ , (7)

(ii) $(\mathbf{B}\odot \mathbf{S})^{\top}\in \mathrm{Sp}^{\Delta}(h,\ell)$

(iii) $\mathbf{1}_h - (\mathbf{B} \odot \mathbf{S})^\top \mathbf{1}_\ell \leq \mathbf{0}_h$ ;

and

$$
\begin{array}{l} f (\mathbf {V}, \mathbf {S}) := \operatorname{Tr} \left\{\left(\left(\mathbf {B} \odot \mathbf {S} \odot \mathbf {V}\right) ^ {\top} \boldsymbol {\Sigma} ^ {\ell} (\mathbf {B} \odot \mathbf {S} \odot \mathbf {V})\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} \tag {8} \\ + \log \det \left\{\left(\mathbf {B} \odot \mathbf {S} \odot \mathbf {V}\right) ^ {\top} \boldsymbol {\Sigma} ^ {\ell} (\mathbf {B} \odot \mathbf {S} \odot \mathbf {V}) \right\}. \\ \end{array}
$$

Prob. 3 makes it explicit that the abstraction morphism is given by three key ingredients: (i) the given partial, structural prior information represented by B; (ii) the structural component S to be learned, such that the resulting causal abstraction is

constructive; and (iii) the abstraction coefficients in V determining the linear functional forms of the causal abstraction, which have to be learned as well. Specifically, “partial” means that some rows of B have more than one entry equal to one.

Unfortunately, Prob. 3 is nonconvex because of the objective function and the Stiefel manifold. Additionally, in this case, the CA results in a bilinear form $\mathbf{B} \odot \mathbf{S} \odot \mathbf{V}$ , which is not jointly convex in $\mathbf{S}$ and $\mathbf{V}$ . Consequently, the constraint $\mathbf{B} \odot \mathbf{S} \odot \mathbf{V} \in \mathrm{St}(\ell, h)$ has to be carefully handled.

Regarding the nonconvexity of the objective in Eq. (8), we proceed by leveraging its smoothness. Specifically, we have the following result.

Corollary J.1. The function $f(\mathbf{V}, \mathbf{S})$ in Eq. (8) is smooth. Additionally, define $\mathbf{A} := (\mathbf{B} \odot \mathbf{S} \odot \mathbf{V})$ and $\widetilde{\mathbf{A}} := (\mathbf{A}^{\top} \boldsymbol{\Sigma}^{\ell} \mathbf{A})^{-1}$ . The partial derivatives w.r.t. V and S are

$$
\nabla_ {\mathbf {V}} f = 2 (\mathbf {B} \odot \mathbf {S}) \odot \left(\left(\boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \widetilde {\mathbf {A}}\right) \left(\mathbf {I} _ {h} - \boldsymbol {\Sigma} ^ {h} \widetilde {\mathbf {A}}\right)\right), \tag {70}
$$

and

$$
\nabla_ {\mathbf {S}} f = 2 (\mathbf {B} \odot \mathbf {V}) \odot \left(\left(\boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \widetilde {\mathbf {A}}\right) \left(\mathbf {I} _ {h} - \boldsymbol {\Sigma} ^ {h} \widetilde {\mathbf {A}}\right)\right). \tag {71}
$$

Proof. Smoothness directly follows from Proposition 5.1 by defining $\mathbf{A} = (\mathbf{B} \odot \mathbf{S} \odot \mathbf{V})$ , which is constrained to $\mathrm{St}(\ell, h)$ as given in Eq. (7). The partial derivatives in Eqs. (70) and (71) follow from the application of Eq. (34), together with the chain rule for derivatives.

At this point, we leverage Corollary J.1 to provide a solution which combines ADMM (Boyd et al., 2011) and SCA (Nedić et al., 2018). Specifically, ADMM is suitable to isolate and consequently tackle the nonconvexity in different subproblems. To manage the bilinear form within the first constraint in Eq. (7), we introduce two splitting variables, namely $Y_{1}$ and $Y_{2}$ in $\mathrm{St}(\ell,h)$ , and the corresponding equality constraints

$$
\mathbf {Y} _ {1} - \mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {V} = 0 _ {\ell \times h} \quad \text { and } \quad \mathbf {Y} _ {2} - \mathbf {B} \odot \mathbf {V} ^ {k + 1} \odot \mathbf {S} = 0 _ {\ell \times h}, \text { respectively. } \tag {72}
$$

In this way, given the solution at iteration k within the ADMM framework, we optimize separately over V and S while always tracking $\mathrm{St}(\ell,h)$ . Please notice that we use $V^{k+1}$ since when optimizing over S, V has already been updated. The rationale behind the usage of the splitting variable for handling the Stiefel manifold is the same as the splitting of orthogonality constraints method (SOC, Lai & Osher, 2014). Additionally, to handle $(\mathbf{B}\odot\mathbf{S})^{\top}\in\mathrm{Sp}^{\Delta}(h,\ell)$ , we introduce another splitting variable $\mathbf{X}\in\mathrm{Sp}^{\Delta}(h,\ell)$ , and the corresponding equality constraint $\mathbf{X}-(\mathbf{B}\odot\mathbf{S})^{\top}=\mathbf{0}_{h\times\ell}$ . Thus, starting from Eq. (7), we get the following equivalent minimization problem

$$
\begin{array}{l} \mathbf {V} ^ {\star}, \mathbf {S} ^ {\star}, \mathbf {Y} _ {1} ^ {\star}, \mathbf {Y} _ {2} ^ {\star}, \mathbf {X} ^ {\star} = \arg \min f (\mathbf {V}, \mathbf {S}); \\ \mathbf {V} \in \mathbb {R} ^ {\ell \times h} \\ \mathbf {S} \in [ 0, 1 ] ^ {\ell \times h} \\ \mathbf {Y} _ {1} \in \operatorname{St} (\ell , h) \\ \mathbf {Y} _ {2} \in \operatorname{St} (\ell , h) \\ \mathbf {X} \in \operatorname{Sp} ^ {\Delta} (h, \ell) \\ \text { subject   to } \quad \mathbf {Y} _ {1} - \mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {V} = \mathbf {0} _ {\ell \times h}, \tag {73} \\ \mathbf {Y} _ {2} - \mathbf {B} \odot \mathbf {V} ^ {k + 1} \odot \mathbf {S} = \mathbf {0} _ {\ell \times h}, \\ \mathbf {X} - \left(\mathbf {B} \odot \mathbf {S}\right) ^ {\top} = \mathbf {0} _ {h \times \ell}, \\ \mathbf {1} _ {h} - (\mathbf {B} \odot \mathbf {S}) ^ {\top} \mathbf {1} _ {\ell} \leq \mathbf {0} _ {h}. \\ \end{array}
$$

Starting from Eq. (73), considering the penalty $\rho \in \mathbb{R}_+$ , we introduce the scaled dual variables $\mathbf{U}_1$ and $\mathbf{U}_2$ in $\mathbb{R}^{\ell \times h}$ ; and $\mathbf{W} \in \mathbb{R}^{h \times \ell}$ , and write the scaled augmented Lagrangian

$$
\begin{array}{l} L _ {\rho} (\mathbf {V}, \mathbf {S}, \mathbf {Y} _ {1}, \mathbf {Y} _ {2}, \mathbf {X}, \mathbf {U} _ {1}, \mathbf {U} _ {2}, \mathbf {W}) = f (\mathbf {V}, \mathbf {S}) + \frac {\rho}{2} \| \mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {V} - \mathbf {Y} _ {1} + \mathbf {U} _ {1} \| _ {\mathrm{F}} ^ {2} + \\ + \frac {\rho}{2} \left\| \mathbf {B} \odot \mathbf {V} ^ {k + 1} \odot \mathbf {S} - \mathbf {Y} _ {2} + \mathbf {U} _ {2} \right\| _ {\mathrm{F}} ^ {2} + \frac {\rho}{2} \left\| (\mathbf {B} \odot \mathbf {S}) ^ {\top} - \mathbf {X} + \mathbf {W} \right\| _ {\mathrm{F}} ^ {2}. \tag {74} \\ \end{array}
$$

Now, we can apply ADMM iterative procedure, getting the recursion for updating the primal and scaled dual variables. In detail, denote by $k \in N$ the current iteration. We have

$$
\mathbf {V} ^ {k + 1} = \underset {\mathbf {V} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} L _ {\rho} \left(\mathbf {V}, \mathbf {S} ^ {k}, \mathbf {Y} _ {1} ^ {k}, \mathbf {Y} _ {2} ^ {k}, \mathbf {X} ^ {k}, \mathbf {U} _ {1} ^ {k}, \mathbf {U} _ {2} ^ {k}, \mathbf {W} ^ {k}\right);
$$

$$
\mathbf {S} ^ {k + 1} = \underset {\mathbf {S} \in [ 0, 1 ] ^ {\ell \times h}} {\arg \min} L _ {\rho} \left(\mathbf {V} ^ {k + 1}, \mathbf {S}, \mathbf {Y} _ {1} ^ {k}, \mathbf {Y} _ {2} ^ {k}, \mathbf {X} ^ {k}, \mathbf {U} _ {1} ^ {k}, \mathbf {U} _ {2} ^ {k}, \mathbf {W} ^ {k}\right)  ,
$$

subject to $\mathbf{1}_{h} - (\mathbf{B} \odot \mathbf{S})^{\top} \mathbf{1}_{\ell} \leq \mathbf{0}_{h}$ ;

$$
\mathbf {Y} _ {1} ^ {k + 1} = \underset {\mathbf {Y} _ {1} \in \mathrm{St} (\ell , h)} {\arg \min} L _ {\rho} \left(\mathbf {V} ^ {k + 1}, \mathbf {S} ^ {k + 1}, \mathbf {Y} _ {1}, \mathbf {Y} _ {2} ^ {k}, \mathbf {X} ^ {k}, \mathbf {U} _ {1} ^ {k}, \mathbf {U} _ {2} ^ {k}, \mathbf {W} ^ {k}\right);
$$

$$
\mathbf {Y} _ {2} ^ {k + 1} = \underset {\mathbf {Y} _ {2} \in \operatorname{St} (\ell , h)} {\arg \min} L _ {\rho} \left(\mathbf {V} ^ {k + 1}, \mathbf {S} ^ {k + 1}, \mathbf {Y} _ {1} ^ {k + 1}, \mathbf {Y} _ {2}, \mathbf {X} ^ {k}, \mathbf {U} _ {1} ^ {k}, \mathbf {U} _ {2} ^ {k}, \mathbf {W} ^ {k}\right); \tag {75}
$$

$$
\mathbf {X} ^ {k + 1} = \underset {\mathbf {X} \in \mathrm{Sp} ^ {\Delta} (h, \ell)} {\arg \min} L _ {\rho} \left(\mathbf {V} ^ {k + 1}, \mathbf {S} ^ {k + 1}, \mathbf {Y} _ {1} ^ {k + 1}, \mathbf {Y} _ {2} ^ {k + 1}, \mathbf {X}, \mathbf {U} _ {1} ^ {k}, \mathbf {U} _ {2} ^ {k}, \mathbf {W} ^ {k}\right);
$$

$$
\mathbf {U} _ {1} ^ {k + 1} = \mathbf {U} _ {1} ^ {k} + \left(\mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {V} ^ {k + 1} - \mathbf {Y} _ {1} ^ {k + 1}\right);
$$

$$
\mathbf {U} _ {2} ^ {k + 1} = \mathbf {U} _ {2} ^ {k} + \left(\mathbf {B} \odot \mathbf {V} ^ {k + 1} \odot \mathbf {S} ^ {k + 1} - \mathbf {Y} _ {2} ^ {k + 1}\right);
$$

$$
\mathbf {W} ^ {k + 1} = \mathbf {W} ^ {k} + \left(\mathbf {B} \odot \mathbf {S} ^ {k + 1}\right) ^ {\top} - \mathbf {X} ^ {k + 1}.
$$

Similarly to SOC, we isolate the objective nonconvexity into the first and second (nonconvex) subproblems; and the nonconvexity of the manifold into the third and fourth (nonconvex) ones. Notably, the first and second subproblems can be managed through SCA. Additionally, the third and fourth nonconvex subproblems admit closed-form solutions since they boil down to the closest orthogonal approximation problems (Fan & Hoffman, 1955; Higham, 1986). Thus, the latter nonconvexity is somehow resolved. Finally, we solve the subproblem for $X^{k+1}$ in closed form as well.

# J.1. Update for $V^{k+1}$

Starting from Eqs. (74) and (75), the subproblem we have to solve is

$$
\mathbf {V} ^ {k + 1} = \underset {\mathbf {V} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} f (\mathbf {V}, \mathbf {S} ^ {k}) + \frac {\rho}{2} \| \mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {V} - \mathbf {Y} _ {1} ^ {k} + \mathbf {U} _ {1} ^ {k} \| _ {\mathrm{F}} ^ {2}. \tag {76}
$$

Eq. (76) is nonconvex due to the inherent nonconvexity of $f(\mathbf{V}, \mathbf{S}^{k})$ . However, the latter function is smooth and differentiable w.r.t. V, as given in Corollary J.1. Hence, we apply the SCA framework. In detail, denote by q the SCA iteration and set $V^{0} = V^{k}$ for q = 0. We derive a strongly convex surrogate $\widetilde{f}(\mathbf{V}; \mathbf{V}^{q}, \mathbf{S}^{k})$ around the point $V^{q} - i.e.$ , the solution at the iterate q - exploiting Eq. (70):

$$
\widetilde {f} (\mathbf {V}; \mathbf {V} ^ {q}, \mathbf {S} ^ {k}) := \operatorname{Tr} \left\{\nabla_ {\mathbf {V}} f | _ {(\mathbf {V} ^ {q}, \mathbf {S} ^ {k})} ^ {\top} \left(\mathbf {V} - \mathbf {V} ^ {q}\right) \right\} + \frac {\tau}{2} \| \mathbf {V} - \mathbf {V} ^ {q} \| _ {\mathrm{F}} ^ {2}. \tag {77}
$$

It is immediate to check that Eq. (77) is a proper surrogate satisfying the stationarity condition $\nabla_{V}f|_{V^{q}} = \nabla_{V}\widetilde{f}|_{V^{q}}$ .

Therefore, at each SCA iteration q, we solve a strongly convex problem in closed-form and then apply the usual smoothing operation by using a diminishing stepsize $\gamma^{q} \in R_{+}$ . Specifically,

$$
\mathbf {V} ^ {q + 1} = \underset {\mathbf {V} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} \quad \widetilde {f} (\mathbf {V}; \mathbf {V} ^ {q}, \mathbf {S} ^ {k}) + \frac {\rho}{2} \| \mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {V} - \mathbf {Y} _ {1} ^ {k} + \mathbf {U} _ {1} ^ {k} \| _ {\mathrm{F}} ^ {2}, \quad (\text {Strongly convex problem}) \tag {78}
$$

$$
\mathbf {V} ^ {q + 1} = \mathbf {V} ^ {q} + \gamma^ {k} \left(\mathbf {V} ^ {q + 1} - \mathbf {V} ^ {q}\right). (\text {Smoothing})
$$

The solution of the strongly-convex problem is given element-wise in Lemma J.2.

Lemma J.2. The update for $V^{q+1}$ can be computed element-wise as

$$
v _ {i j} ^ {q + 1} = \frac {1}{\tau + b _ {i j} s _ {i j} ^ {k ^ {2}}} \left(\rho b _ {i j} s _ {i j} ^ {k} y _ {1 _ {i j}} ^ {k} - \rho b _ {i j} s _ {i j} ^ {k} u _ {1 _ {i j}} ^ {k} + \tau v _ {i j} ^ {q} - \left[ \nabla \mathbf {v} f | _ {\left(\mathbf {v} ^ {q}, \mathbf {S} ^ {k} \right.} \right] _ {i j}\right). \tag {79}
$$

Proof. The proof follows by imposing the stationarity condition

$$
\mathbf {0} _ {\ell \times h} = \nabla_ {\mathbf {V}} f | _ {(\mathbf {V} ^ {q}, \mathbf {S} ^ {k}} + \tau (\mathbf {V} - \mathbf {V} ^ {q}) + \rho \mathbf {B} \odot \mathbf {S} ^ {k} \odot (\mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {V} - \mathbf {Y} _ {1} ^ {k} + \mathbf {U} _ {1} ^ {k}), \tag {80}
$$

and solving for $\mathbf{V}$ .

Additionally, the diminishing stepsize $\gamma^{k}$ has to satisfy the classical stochastic approximation conditions (Nedić et al., 2018),

$$
(i) \sum_ {q = 1} ^ {\infty} \gamma^ {q} = \infty \quad \text { and } \quad (i i) \sum_ {q = 1} ^ {\infty} (\gamma^ {q}) ^ {2} <   \infty . \tag {81}
$$

In our experiments, we use the decaying rule

$$
\gamma^ {q + 1} = \gamma^ {q} (1 - \varepsilon \gamma^ {q}), \quad \varepsilon \in (0, 1). \tag {82}
$$

The SCA framework is guaranteed to converge to stationary points of the original nonconvex problem in Eq. (76) (Nedić et al., 2018). Accordingly, we establish convergence for the update when

$$
\left\| \mathbf {V} ^ {q + 1} - \mathbf {V} ^ {q} \right\| _ {\mathrm{F}} \leq \tau^ {c}, \quad \tau^ {c} \approx 0; \tag {83}
$$

and set $\mathbf{V}^{k + 1} = \mathbf{V}^{q + 1}$

# J.2. Update for $\mathbf{S}^{k + 1}$

Starting from Eqs. (74) and (75), the subproblem we have to solve is

$$
\mathbf {S} ^ {k + 1} = \underset {\mathbf {S} \in [ 0, 1 ] ^ {\ell \times h}} {\arg \min} f (\mathbf {V} ^ {k}, \mathbf {S}) + \frac {\rho}{2} \left\| \mathbf {B} \odot \mathbf {V} ^ {k + 1} \odot \mathbf {S} - \mathbf {Y} _ {2} ^ {k} + \mathbf {U} _ {2} ^ {k} \right\| _ {\mathrm{F}} ^ {2} + \frac {\rho}{2} \left\| (\mathbf {B} \odot \mathbf {S}) ^ {\top} - \mathbf {X} ^ {k} + \mathbf {W} ^ {k} \right\| _ {\mathrm{F}} ^ {2}, \tag {84}
$$

subject to $\mathbf{1}_h - (\mathbf{B} \odot \mathbf{S})^\top \mathbf{1}_\ell \leq \mathbf{0}_h$ .

The subproblem above is nonconvex and constrained. Similarly to App. J.1, we apply the SCA framework. Denote by $q$ the SCA iteration and set $\mathbf{S}^0 = \mathbf{S}^k$ for $q = 0$ . Here, the strongly convex surrogate of $f(\mathbf{V}^{k + 1},\mathbf{S})$ reads as

$$
\widetilde {f} (\mathbf {S}; \mathbf {V} ^ {k + 1}, \mathbf {S} ^ {q}) := \operatorname{Tr} \left\{\nabla_ {\mathbf {S}} f | _ {(\mathbf {V} ^ {k + 1}, \mathbf {S} ^ {q})} ^ {\top} \left(\mathbf {S} - \mathbf {S} ^ {q}\right) \right\} + \frac {\tau}{2} \| \mathbf {S} - \mathbf {S} ^ {q} \| _ {\mathrm{F}} ^ {2}, \tag {85}
$$

which satisfies $\nabla_{\mathbf{S}}f|_{(\mathbf{V}^{k + 1},\mathbf{S}^q)} = \nabla_{\mathbf{S}}f|_{(\mathbf{V}^{k + 1},\mathbf{S}^q)}$ . At each SCA iteration $q$ , we solve a constrained quadratic programming (QP) problem and apply the smoothing step by using the stepsize $\gamma^q\in \mathbb{R}_+$ complying with the conditions in Eq. (81). In detail, let $\operatorname {vec}(\mathbf{A})$ be the column-wise vectorization of a given matrix $\mathbf{A}$ and define

$$
\mathbf {Q} = \tau \mathbf {I} _ {\ell h} + \rho \operatorname{diag} \left(\left(\operatorname{vec} (\mathbf {B}) \odot \operatorname{vec} \left(\mathbf {V} ^ {k + 1}\right)\right) \odot \left(\operatorname{vec} (\mathbf {B}) \odot \operatorname{vec} \left(\mathbf {V} ^ {k + 1}\right)\right)\right) + \rho \operatorname{diag} \left(\operatorname{vec} (\mathbf {B}) \odot \operatorname{vec} (\mathbf {B})\right),
$$

$$
\mathbf {c} = \operatorname{vec} \left(\nabla \mathbf {s} f | _ {\left(\mathbf {V} ^ {k + 1}, \mathbf {S} ^ {q}\right)}\right) - \tau \operatorname{vec} \left(\mathbf {S} ^ {q}\right) - \rho \operatorname{vec} \left(\mathbf {Y} _ {2} ^ {k} - \mathbf {U} _ {2} ^ {k}\right) \odot \operatorname{vec} (\mathbf {B}) \odot \operatorname{vec} \left(\mathbf {V} ^ {k + 1}\right) - \rho \operatorname{vec} (\mathbf {B}) \odot \operatorname{vec} \left(\left(\mathbf {X} ^ {k} - \mathbf {W} ^ {k}\right) ^ {\top}\right). \tag {86}
$$

Additionally, recall that $\operatorname{vec}(\mathbf{AC}) = (\mathbf{C}^{\top} \otimes \mathbf{I}_h) \operatorname{vec}(\mathbf{A})$ , with $\mathbf{A} \in \mathbb{R}^{h \times \ell}$ and $\mathbf{C} \in \mathbb{R}^{\ell \times m}$ . Hence, denoting with $\mathbf{K}^{\ell,h}$ the commutation matrix, the inequality constraint can be rewritten as

$$
\begin{array}{l} \mathbf {1} _ {h} - \operatorname{vec} \left(\left(\mathbf {B} \odot \mathbf {S}\right) ^ {\top} \mathbf {1} _ {\ell}\right) = \mathbf {1} _ {h} - \left(\mathbf {1} _ {\ell} ^ {\top} \otimes \mathbf {I} _ {h}\right) \operatorname{vec} \left(\left(\mathbf {B} \odot \mathbf {S}\right) ^ {\top}\right) \\ = \mathbf {1} _ {h} - \left(\mathbf {1} _ {\ell} ^ {\top} \otimes \mathbf {I} _ {h}\right) \mathbf {K} ^ {\ell , h} \operatorname{vec} (\mathbf {B} \odot \mathbf {S}) \\ = \mathbf {1} _ {h} - \left(\mathbf {1} _ {\ell} ^ {\top} \otimes \mathbf {I} _ {h}\right) \mathbf {K} ^ {\ell , h} \operatorname{vec} (\operatorname{diag} (\operatorname{vec} (\mathbf {B})) \operatorname{vec} (\mathbf {S})) \tag {87} \\ = \mathbf {1} _ {h} - \underbrace {\left(\mathbf {1} _ {\ell} ^ {\top} \otimes \mathbf {I} _ {h}\right) \mathbf {K} ^ {\ell , h} \operatorname{diag} (\operatorname{vec} (\mathbf {B}))} _ {\mathbf {G}} \operatorname{vec} (\mathbf {S}) \leq \mathbf {0} _ {h}. \\ \end{array}
$$

At this point, starting from Eq. (84) and exploiting Eqs. (86) and (87), we can pose the SCA recursion:

$$
\operatorname{vec} \left(\mathbf {S}\right) ^ {q + 1} = \underset {\mathbf {S} \in [ 0, 1 ] ^ {\ell \times h}} {\arg \min} \quad \frac {1}{2} \operatorname{vec} \left(\mathbf {S}\right) ^ {\top} \mathbf {Q} \operatorname{vec} \left(\mathbf {S}\right) + \mathbf {c} ^ {\top} \operatorname{vec} \left(\mathbf {S}\right), \quad (\text { QP   problem })
$$

subject to $\mathbf{1}_{h}-\mathbf{G}\operatorname{vec}(\mathbf{S})\leq\mathbf{0}_{h}$ . (88)

$$
\operatorname{vec} (\mathbf {S}) ^ {q + 1} = \operatorname{vec} (\mathbf {S}) ^ {q} + \gamma^ {k} \left(\operatorname{vec} (\mathbf {S}) ^ {q + 1} - \operatorname{vec} (\mathbf {S}) ^ {q}\right). \quad (\text { Smoothing })
$$

The QP problem in Eq. (88) can be solved through off-the-shelf quadratic programming solvers. In our experiments, we use the OSQP (Stellato et al., 2020) implementation available in cvxpy (Diamond & Boyd, 2016). Since the quadratic form involves a diagonal, positive definite matrix $\mathbf{Q}$ , in case a solution exists in the feasible set determined by the inequality constraint, it is also unique. Regarding the smoothing step, $\gamma^q$ follows Eq. (82). Similarly to App. J.1, we determine convergence when

$$
\left\| \operatorname{vec} (\mathbf {S}) ^ {q + 1} - \operatorname{vec} (\mathbf {S}) ^ {q} \right\| _ {\mathrm{F}} \leq \tau^ {c}, \quad \tau^ {c} \approx 0; \tag {89}
$$

and set $S^{k+1} = S^{q+1}$ , where $S^{q+1}$ is the reshaping of $\operatorname{vec}(S)^{q+1}$ in matrix form.

# J.3. Update for $Y_{1}^{k+1}$ and $Y_{2}^{k+1}$

Starting from Eqs. (74) and (75), the subproblem to solve is

$$
\begin{array}{l} \mathbf {Y} _ {1} ^ {k + 1} = \underset {\mathbf {Y} _ {1} \in \operatorname{St} (\ell , h)} {\arg \min} \quad \frac {\rho}{2} \| \mathbf {B} \odot \mathbf {S} ^ {k + 1} \odot \mathbf {V} ^ {k + 1} - \mathbf {Y} _ {1} + \mathbf {U} _ {1} ^ {k} \| _ {\mathrm{F}} ^ {2} \tag {90} \\ = \operatorname{prox} _ {\mathrm{St} (\ell , h)} \left(\widetilde {\mathbf {Y}} _ {1}\right), \quad \text {with} \widetilde {\mathbf {Y}} _ {1} := \mathbf {B} \odot \mathbf {S} ^ {k + 1} \odot \mathbf {V} ^ {k + 1} + \mathbf {U} _ {1} ^ {k}. \\ \end{array}
$$

The evaluation of $\mathrm{prox}_{\mathrm{St}(\ell ,h)}(\widetilde{\mathbf{Y}}_1)$ in Eq. (90) is equivalent to the (unique) solution of the closest orthogonal approximation problem (Fan & Hoffman, 1955; Higham, 1986). Specifically, it is equal to the $\mathbf{U}_{p_1}$ factor of the polar decomposition of the matrix $\widetilde{\mathbf{Y}}_1 = \mathbf{U}_{p_1}\mathbf{P}_{p_1}$ , namely

$$
\mathbf {Y} _ {1} ^ {k + 1} = \mathbf {U} _ {p _ {1}}. \tag {91}
$$

Similarly, defining $\widetilde{Y}_{2} := B \odot S^{k+1} \odot V^{k+1} + U_{2}^{k}$ and considering the polar decomposition $\widetilde{Y}_{2} = U_{p_{2}} P_{p_{2}}$ , we have

$$
\mathbf {Y} _ {2} ^ {k + 1} = \mathbf {U} _ {p _ {2}}. \tag {92}
$$

# J.4. Update for $\mathbf{X}^{k + 1}$

Starting from Eqs. (74) and (75), the subproblem reads as

$$
\begin{array}{l} \mathbf {X} ^ {k + 1} = \underset {\mathbf {X} \in \mathrm{Sp} ^ {\Delta} (h, \ell)} {\arg \min} \frac {\rho}{2} \left\| (\mathbf {B} \odot \mathbf {S} ^ {k + 1}) ^ {\top} - \mathbf {X} + \mathbf {W} ^ {k} \right\| _ {\mathrm{F}} ^ {2} \tag {93} \\ = \operatorname{prox} _ {\mathrm{Sp} ^ {\Delta} (h, \ell)} \left(\left(\mathbf {B} \odot \mathbf {S} ^ {k + 1}\right) ^ {\top} + \mathbf {W} ^ {k}\right). \\ \end{array}
$$

The following result gives the solution.

Lemma J.3. Consider

$$
\mathrm{Sp} ^ {\Delta} (h, \ell) := \left\{\mathbf {A} \in \{0, 1 \} ^ {h \times \ell} \mid \| \mathbf {a} _ {j} \| _ {2} = 1 a n d \sum_ {i = 1} ^ {h} a _ {i j} = 1, \forall j \in [ \ell ] \right\}; \tag {94}
$$

and $\mathbf{A} \in \mathbb{R}^{h \times \ell}$ . The proximal operator

$$
\operatorname{prox} _ {\mathrm{Sp} ^ {\Delta} (h, \ell)} (\mathbf {A}) := \underset {\mathbf {X} \in \mathbb {R} ^ {h \times \ell}} {\arg \min} \| \mathbf {A} - \mathbf {X} \| _ {\mathrm{F}}, \tag {95}
$$

is the matrix $\mathbf{X}^{\star}$ such that

$$
\forall j \in [ \ell ], x _ {i j} ^ {\star} = \left\{ \begin{array}{l l} 1, & \text { if   } a _ {i j} = \arg \min _ {i} | a _ {i j} - 1 |, \\ 0, & \text { otherwise. } \end{array} \right. \tag {96}
$$

Proof. To belong to $\mathrm{Sp}^{\Delta}(h,\ell)$ , $X^{\star}$ must have only a single nonzero entry equal to one for each column $j\in[\ell]$ . Consequently, the objective in Eq. (95) is minimized by setting, for each column $j\in[\ell]$ , $x_{ij}^{\star}=1$ in correspondence of the element $a_{ij}$ whose absolute distance from one is minimum. □

# J.5. Stopping criteria

The empirical convergence of the proposed method is established according to primal and dual feasibility optimality conditions. In this case, the primal residuals associated with the equality constraints in Eq. (73) are

$$
\mathbf {R} _ {p, 1} ^ {k + 1} := \mathbf {Y} _ {1} ^ {k + 1} - \mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {V} ^ {k + 1};
$$

$$
\mathbf {R} _ {p, 2} ^ {k + 1} := \mathbf {Y} _ {2} ^ {k + 1} - \mathbf {B} \odot \mathbf {V} ^ {k + 1} \odot \mathbf {S} ^ {k + 1}; \tag {97}
$$

$$
\mathbf {R} _ {p, 3} ^ {k + 1} := \mathbf {X} ^ {k + 1} - \left(\mathbf {B} \odot \mathbf {S} ^ {k + 1}\right) ^ {\top}.
$$

Additionally, the dual residuals obtained from the stationarity condition are

$$
\mathbf {R} _ {d, 1} ^ {k + 1} := \rho \mathbf {B} \odot \mathbf {S} ^ {k} \odot \left(\mathbf {Y} _ {1} ^ {k + 1} - \mathbf {Y} _ {1} ^ {k}\right);
$$

$$
\mathbf {R} _ {d, 2} ^ {k + 1} := \rho \mathbf {B} \odot \mathbf {V} ^ {k + 1} \odot \left(\mathbf {Y} _ {2} ^ {k + 1} - \mathbf {Y} _ {2} ^ {k}\right); \tag {98}
$$

$$
\mathbf {R} _ {d, 3} ^ {k + 1} := \rho \mathbf {B} \odot \left(\mathbf {X} ^ {k + 1} - \mathbf {X} ^ {k}\right) ^ {\top}.
$$

Following (Boyd et al., 2011), denoting with $\tau^a$ and $\tau^r$ in $\mathbb{R}_+$ the absolute and relative tolerances, respectively, the stopping criteria to be satisfied for empirical convergence are

$$
\left\| \mathbf {R} _ {p, 1} ^ {k + 1} \right\| _ {\mathrm{F}} = d _ {p, 1} ^ {k + 1} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \max \left(\left\| \mathbf {Y} _ {1} ^ {k + 1} \right\| _ {\mathrm{F}}, \left\| \mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {V} ^ {k + 1} \right\| _ {\mathrm{F}}\right),
$$

$$
\left\| \mathbf {R} _ {p, 2} ^ {k + 1} \right\| _ {\mathrm{F}} = d _ {p, 2} ^ {k + 1} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \max \left(\left\| \mathbf {Y} _ {2} ^ {k + 1} \right\| _ {\mathrm{F}}, \left\| \mathbf {B} \odot \mathbf {V} ^ {k + 1} \odot \mathbf {S} ^ {k + 1} \right\| _ {\mathrm{F}}\right),
$$

$$
\left\| \mathbf {R} _ {p, 3} ^ {k + 1} \right\| _ {\mathrm{F}} = d _ {p, 3} ^ {k + 1} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \max \left(\left\| \mathbf {X} ^ {k + 1} \right\| _ {\mathrm{F}}, \left\| \mathbf {B} \odot \mathbf {S} ^ {k + 1} \right\| _ {\mathrm{F}}\right),
$$

$$
\left\| \mathbf {R} _ {d, 1} ^ {k + 1} \right\| _ {\mathrm{F}} = d _ {d, 1} ^ {k + 1} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \rho \left\| \mathbf {B} \odot \mathbf {S} ^ {k} \odot \mathbf {U} _ {1} ^ {k + 1} \right\| _ {\mathrm{F}}, \tag {99}
$$

$$
\left\| \mathbf {R} _ {d, 2} ^ {k + 1} \right\| _ {\mathrm{F}} = d _ {d, 2} ^ {k + 1} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \rho \left\| \mathbf {B} \odot \mathbf {V} ^ {k + 1} \odot \mathbf {U} _ {2} ^ {k + 1} \right\| _ {\mathrm{F}},
$$

$$
\left\| \mathbf {R} _ {d, 3} ^ {k + 1} \right\| _ {\mathrm{F}} = d _ {d, 3} ^ {k + 1} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \rho \left\| \mathbf {B} ^ {\top} \odot \mathbf {W} ^ {k + 1} \right\| _ {\mathrm{F}}.
$$

The CLinSEPAL method is summarized in Algorithm 3

Algorithm 3 CLinSEPAL   
1: Input: $\Sigma^{\ell}, \Sigma^{h}, \mathbf{B}, \rho, \tau, \varepsilon, \tau^{c}, \tau^{a}, \tau^{r}$ 2: Initialize: $\mathbf{V}^{0} \in \mathbb{R}^{\ell \times h}, \mathbf{S}^{0} = \mathbf{B}, \mathbf{Y}_{1}^{0} \in \mathrm{St}(\ell, h), \mathbf{Y}_{2}^{0} \in \mathrm{St}(\ell, h), \mathbf{X}^{0} = \mathbf{B}^{\top}, \mathbf{U}_{1}^{0} \leftarrow \mathbf{B} \odot \mathbf{S}^{0} \odot \mathbf{V}^{0} - \mathbf{Y}_{1}^{0}, \mathbf{U}_{2}^{0} \leftarrow \mathbf{B} \odot \mathbf{S}^{0} \odot \mathbf{V}^{0} - \mathbf{Y}_{2}^{0}, \mathbf{W}^{0} \leftarrow (\mathbf{B} \odot \mathbf{S}^{0})^{\top} - \mathbf{X}^{0}$ 3: repeat
4: $\mathbf{V}^{k+1} \leftarrow$ Apply Eq. (78)
5: $\mathbf{S}^{k+1} \leftarrow$ Apply Eq. (88)
6: $\mathbf{Y}_{1}^{k+1} \leftarrow$ Eq. (91)
7: $\mathbf{Y}_{2}^{k+1} \leftarrow$ Eq. (92)
8: $\mathbf{U}_{1}^{k+1} \leftarrow \mathbf{U}_{1}^{k} + \mathbf{B} \odot \mathbf{S}^{k} \odot \mathbf{V}^{k+1} - \mathbf{Y}_{1}^{k+1}$ 9: $\mathbf{U}_{2}^{k+1} \leftarrow \mathbf{U}_{2}^{k} + \mathbf{B} \odot \mathbf{V}^{k+1} \odot \mathbf{S}^{k+1} - \mathbf{Y}_{2}^{k+1}$ 10: $\mathbf{W}^{k+1} \leftarrow \mathbf{W}^{k} + (\mathbf{B} \odot \mathbf{S}^{k+1})^{\top} - \mathbf{X}^{k+1}$ 11: until Eq. (99) is satisfied
12: Output: V, S, Y $_{1}$ , Y $_{2}$ , X, U $_{1}$ , U $_{2}$ , W

# J.6. Full prior case

Prob. 3 simplifies in case of full prior knowledge of $\mathbf{B}$ . Indeed, it is not needed to learn $\mathbf{S}$ since $\mathbf{S} \equiv \mathbf{B}$ . Accordingly, we get the following.

Problem 4. Given $\Sigma^{\ell} \in S_{++}^{\ell}$ , $\Sigma^h \in S_{++}^h$ , and $\mathbf{B} \in \{0,1\}^{\ell \times h}$ , the linear constructive CA is given by the transpose of the product $\mathbf{B} \odot \mathbf{V}$ , where

$$
\mathbf {V} ^ {\star} = \underset {\mathbf {V} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} f (\mathbf {V}); \tag {100}
$$

subject to $\mathbf{B} \odot \mathbf{V} \in \mathrm{St}(\ell, h)$ ;

and

$$
f (\mathbf {V}) := \operatorname{Tr} \left\{\left((\mathbf {B} \odot \mathbf {V}) ^ {\top} \boldsymbol {\Sigma} ^ {\ell} (\mathbf {B} \odot \mathbf {V})\right) ^ {- 1} \boldsymbol {\Sigma} ^ {h} \right\} + \log \det \left\{(\mathbf {B} \odot \mathbf {V}) ^ {\top} \boldsymbol {\Sigma} ^ {\ell} (\mathbf {B} \odot \mathbf {V}) \right\}. \tag {101}
$$

The solution can be obtained in a similar manner as for the partial prior knowledge case. Below, we report the mathematical derivation for completeness without further comments.

Corollary J.4. The function $f(\mathbf{V})$ in Eq. (101) is smooth. Additionally, define $\mathbf{A} := (\mathbf{B} \odot \mathbf{V})$ and $\widetilde{\mathbf{A}} := (\mathbf{A}^{\top} \boldsymbol{\Sigma}^{\ell} \mathbf{A})^{-1}$ . The gradient is

$$
\nabla_ {\mathbf {V}} f = 2 \mathbf {B} \odot \left(\left(\boldsymbol {\Sigma} ^ {\ell} \mathbf {A} \widetilde {\mathbf {A}}\right) \left(\mathbf {I} _ {h} - \boldsymbol {\Sigma} ^ {h} \widetilde {\mathbf {A}}\right)\right), \tag {102}
$$

Proof. Smoothness directly follows from Proposition 5.1 by defining $\mathbf{A} := (\mathbf{B} \odot \mathbf{V})$ , which is constrained to $\operatorname{St}(\ell, h)$ as given in Eq. (100). The gradient in Eq. (102) follows from the application of Eq. (34), together with the chain rule for derivatives.

Starting from Eq. (100), we get the following equivalent minimization problem

$$
\mathbf {V} ^ {\star}, \mathbf {Y} ^ {\star} = \underset { \begin{array}{c} \mathbf {V} \in \mathbb {R} ^ {\ell \times h} \\ \mathbf {Y} \in \operatorname{St} (\ell , h) \end{array} } {\arg \min} f (\mathbf {V}); \tag {103}
$$

subject to $\mathbf{Y} - \mathbf{B}\odot \mathbf{V} = \mathbf{0}_{\ell \times h}$

Considering the scaled dual variable $U \in R^{\ell \times h}$ , the scaled augmented Lagrangian is

$$
L _ {\rho} (\mathbf {V}, \mathbf {Y}, \mathbf {U}) = f (\mathbf {V}) + \frac {\rho}{2} \| \mathbf {B} \odot \mathbf {V} - \mathbf {Y} + \mathbf {U} \| _ {\mathrm{F}} ^ {2}. \tag {104}
$$

The ADMM recursion is

$$
\begin{array}{l} \mathbf {V} ^ {k + 1} = \underset {\mathbf {V} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} L _ {\rho} \left(\mathbf {V}, \mathbf {Y} ^ {k}, \mathbf {U} ^ {k}\right); \\ \mathbf {Y} ^ {k + 1} = \underset {\mathbf {Y} \in \mathrm{St} (\ell , h)} {\arg \min} L _ {\rho} \left(\mathbf {V} ^ {k + 1}, \mathbf {Y}, \mathbf {U} ^ {k}\right); \tag {105} \\ \mathbf {U} ^ {k + 1} = \mathbf {U} ^ {k} + \left(\mathbf {B} \odot \mathbf {V} ^ {k + 1} - \mathbf {Y} ^ {k + 1}\right). \\ \end{array}
$$

# J.6.1. UPDATE FOR $\mathbf{V}^{k + 1}$

Starting from Eqs. (104) and (105), the subproblem we have to solve is

$$
\mathbf {V} ^ {k + 1} = \underset {\mathbf {V} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} f (\mathbf {V}) + \frac {\rho}{2} \| \mathbf {B} \odot \mathbf {V} - \mathbf {Y} ^ {k} + \mathbf {U} ^ {k} \| _ {\mathrm{F}} ^ {2}. \tag {106}
$$

Eq. (106) is nonconvex due to the inherent nonconvexity of $f(\mathbf{V})$ . However, the latter function is smooth and differentiable w.r.t. $\mathbf{V}$ , as given in Corollary J.4. Hence, we apply the SCA framework. In detail, denote by $q$ the SCA iteration and set $\mathbf{V}^0 = \mathbf{V}^k$ for $q = 0$ . We derive a strongly convex surrogate $\widetilde{f}(\mathbf{V};\mathbf{V}^q)$ around the point $\mathbf{V}^q$ - i.e., the solution at the iterate $q$ - exploiting Eq. (102),

$$
\widetilde {f} (\mathbf {V}; \mathbf {V} ^ {q}) := \operatorname{Tr} \left\{\nabla_ {\mathbf {V}} f | _ {\mathbf {V} ^ {q}} ^ {\top} \left(\mathbf {V} - \mathbf {V} ^ {q}\right) \right\} + \frac {\tau}{2} \| \mathbf {V} - \mathbf {V} ^ {q} \| _ {\mathrm{F}} ^ {2}. \tag {107}
$$

Therefore, at each SCA iteration q, we solve a strongly convex problem in closed-form and then apply the usual smoothing operation by using a diminishing stepsize $\gamma^{q} \in R_{+}$ following Eq. (82) and satisfying Eq. (81). Specifically,

$$
\mathbf {V} ^ {q + 1} = \underset {\mathbf {V} \in \mathbb {R} ^ {\ell \times h}} {\arg \min} \quad \widetilde {f} (\mathbf {V}; \mathbf {V} ^ {q}) + \frac {\rho}{2} \| \mathbf {B} \odot \mathbf {V} - \mathbf {Y} ^ {k} + \mathbf {U} ^ {k} \| _ {\mathrm{F}} ^ {2}, \quad (\text {Strongly convex problem}) \tag {108}
$$

$$
\mathbf {V} ^ {q + 1} = \mathbf {V} ^ {q} + \gamma^ {k} \left(\mathbf {V} ^ {q + 1} - \mathbf {V} ^ {q}\right). \quad (\text { Smoothing })
$$

The solution of the strongly-convex problem is given element-wise in Lemma J.5.

Lemma J.5. The update for $V^{q+1}$ can be computed element-wise as

$$
v _ {i j} ^ {q + 1} = \frac {1}{\tau + b _ {i j}} \left(\rho b _ {i j} y _ {i j} ^ {k} - \rho b _ {i j} u _ {i j} ^ {k} + \tau v _ {i j} ^ {q} - \left[ \nabla \mathbf {v} f | \mathbf {v} ^ {q} \right] _ {i j}\right). \tag {109}
$$

Proof. The proof follows by imposing the stationarity condition

$$
\mathbf {0} _ {\ell \times h} = \nabla_ {\mathbf {V}} f | _ {\mathbf {V} ^ {q}} + \tau (\mathbf {V} - \mathbf {V} ^ {q}) + \rho \mathbf {B} \odot (\mathbf {B} \odot \mathbf {V} - \mathbf {Y} ^ {k} + \mathbf {U} ^ {k}), \tag {110}
$$

and solving for $\mathbf{V}$ .

We establish convergence for the update when

$$
\left\| \mathbf {V} ^ {q + 1} - \mathbf {V} ^ {q} \right\| _ {\mathrm{F}} \leq \tau^ {c}, \quad \tau^ {c} \approx 0; \tag {111}
$$

and set $\mathbf{V}^{k + 1} = \mathbf{V}^{q + 1}$

# J.6.2. UPDATE FOR $\mathbf{Y}^{k + 1}$

Starting from Eqs. (104) and (105), the subproblem to solve is

$$
\begin{array}{l} \mathbf {Y} ^ {k + 1} = \underset {\mathbf {Y} \in \operatorname{St} (\ell , h)} {\arg \min} \quad \frac {\rho}{2} \| \mathbf {B} \odot \mathbf {V} ^ {k + 1} - \mathbf {Y} + \mathbf {U} ^ {k} \| _ {\mathrm{F}} ^ {2} \tag {112} \\ = \operatorname{prox} _ {\mathrm{St} (\ell , h)} (\widetilde {\mathbf {Y}}), \quad \text { with } \widetilde {\mathbf {Y}} := \mathbf {B} \odot \mathbf {V} ^ {k + 1} + \mathbf {U} ^ {k}. \\ \end{array}
$$

Denoting by $U_{p}P_{p}$ the polar decomposition of the matrix $\widetilde{Y}$ , the update is

$$
\mathbf {Y} ^ {k + 1} = \mathbf {U} _ {p}. \tag {113}
$$

# J.6.3. STOPPING CRITERIA

The empirical convergence of the proposed method is established according to primal and dual feasibility optimality conditions (Boyd et al., 2011). The primal residual, associated with the equality constraint in Eq. (103), is

$$
\mathbf {R} _ {p} ^ {k + 1} := \mathbf {Y} ^ {k + 1} - \mathbf {B} \odot \mathbf {V} ^ {k + 1}. \tag {114}
$$

The dual residual, which can be obtained from the stationarity condition, is

$$
\mathbf {R} _ {d} ^ {k + 1} := \rho \mathbf {B} \odot \left(\mathbf {Y} ^ {k + 1} - \mathbf {Y} ^ {k}\right). \tag {115}
$$

As $k \to \infty$ , the norm of the primal and dual residuals should vanish. Hence, the stopping criterion can be set in terms of the norms

$$
(i) d _ {p} ^ {k + 1} = \left\| \mathbf {R} _ {p} ^ {k + 1} \right\| _ {\mathrm{F}} \quad \text { and } \quad (i i) d _ {d} ^ {k + 1} = \left\| \mathbf {R} _ {d} ^ {k + 1} \right\| _ {\mathrm{F}}. \tag {116}
$$

Specifically, given absolute and relative tolerance, namely $\tau^a$ and $\tau^r$ in $\mathbb{R}_+$ , respectively, convergence in practice is established following Boyd et al. (2011), when

$$
(i) d _ {p} ^ {k + 1} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \max \left(\left\| \mathbf {Y} ^ {k + 1} \right\| _ {\mathrm{F}}, \left\| \mathbf {B} \odot \mathbf {V} ^ {k + 1} \right\| _ {\mathrm{F}}\right), \quad \text { and } \quad (i i) d _ {d} ^ {k + 1} \leq \tau^ {a} \sqrt {\ell h} + \tau^ {r} \rho \left\| \mathbf {B} \odot \mathbf {U} ^ {k + 1} \right\| _ {\mathrm{F}}. \tag {117}
$$

The full prior version of CLinSEPAL is summarized in Algorithm 4.

Algorithm 4 CLinSEPAL (full prior case)   
1: Input: $\Sigma^{\ell}$ , $\Sigma^{h}$ , B, $\rho$ , $\tau$ , $\varepsilon$ , $\tau^{c}$ , $\tau^{a}$ , $\tau^{r}$ 2: Initialize: $\mathbf{V}^{0} \in \mathbb{R}^{\ell \times h}$ , $\mathbf{Y}^{0} \in \mathrm{St}(\ell, h)$ , $\mathbf{U}^{0} \leftarrow \mathbf{B} \odot \mathbf{V}^{0} - \mathbf{Y}^{0}$ 3: repeat
4: $\mathbf{V}^{k+1} \leftarrow$ Apply Eq. (108)
5: $\mathbf{Y}^{k+1} \leftarrow$ Eq. (113)
6: $\mathbf{U}^{k+1} \leftarrow \mathbf{U}^{k} + \mathbf{B} \odot \mathbf{V}^{k+1} - \mathbf{Y}^{k+1}$ 7: until Eq. (117) is satisfied
8: Output: V, Y, U

# K. Metrics and Hyper-parameters

This section provides the definition of the metrics monitored in our empirical assessment in Secs. 6 and 7. Additionally, we report the hyper-parameters configuration for Algorithms 1 to 3 used in the experiments.

Metrics. Denote by $\mathbf{V}^{\star}$ and $\widehat{\mathbf{V}}$ the ground-truth and the learned (transpose of the) linear CA, both being matrices in $\mathbb{R}^{\ell \times h}$ . The metrics are defined as follows.

\- Fraction of learned constructive morphisms: We define constructiveness as

$$
\text { constr } = (\text { number   of   rows   with   one   nonzero   entry }) / \ell + (\text { number   of   columns   with   at   least   one   nonzero   entry }) / h. \tag {118}
$$

Then, indicating by $S$ the number of experiments, the metric is given by the number of $\tilde{\mathbf{V}}$ leading to constr = 1 divided by $S$ .

- KL divergence: Eq. (3) evaluated at $\widehat{\mathbf{V}}$ ;   
- Frobenious absolute distance:

$$
\frac {\left\| \left| \mathbf {V} ^ {\star} \right| - \left| \widehat {\mathbf {V}} \right| \right\| _ {\mathrm{F}}}{\left\| \left| \mathbf {V} ^ {\star} \right| \right\| _ {\mathrm{F}}}; \tag {119}
$$

\- F1 score: Given

- True positive rate tpr: (true positive, tp: number of predicted nonzero entries in $\widehat{\mathbf{V}}$ existing in $\mathbf{V}^{\star}$ )/ (number of nonzero entries in $\mathbf{V}^{\star}$ ),   
- False discovery rate fdr: (false positive, fp: number of predicted nonzero entries in $\widehat{\mathbf{V}}$ that do not exist in $\mathbf{V}^{\star}$ )/(tp + fp);

the F1 results in the harmonic mean of tpr and $(1 - fdr)$ .

# Hyper-parameters.

- CLinSEPAL: $\rho = 1$ , $\tau = 10^{-3}$ , $\varepsilon = 0.1$ for the full prior case and $\varepsilon = 0.01$ for the partial prior case, $\tau^c = 10^{-3}$ , $\tau^a = 10^{-4}$ , $\tau^r = 10^{-4}$ . The same hyper-parameters were used in the experiments in Sec. 7;   
- LinSEPAL-ADMM: $\rho = 1, \lambda = 1, \tau^{a} = 10^{-4}, \tau^{r} = 10^{-4}$ ;   
- LinSEPAL-PG: $\lambda = 1$ , $\rho = 1 / \left(2 \| \boldsymbol{\Sigma}^{\ell} \|_{\mathrm{F}}^{2}\right)$ , $\gamma = 0.5$ , $\tau^{\mathrm{KL}} = 10^{-4}$ , $K = 1000$ .

# L. Additional Results on Synthetic Data

![](images/ed2d6b6c842a2d3fa0d84223e6c5093d406fd7e0aab58a365dda0eb2ca57bce4.jpg)  
Figure 7: The figure shows the results in the pp setting where the 90% threshold on constructiveness was removed. Differently from Fig. 4, top left boxplots are for constructiveness values. The remaining plots are as in Fig. 4. From the figure we see that although LinSEPAL-ADMM and LinSEPAL-PG minimize the misalignment between the (abstracted) low- and high-level probability measures (top right), the quality of learned CAs is poor compared to that of learned CAs from CLinSEPAL (bottom plots).

# M. Additional Material for the Causal Abstraction of Brain Networks

This section provides additional material about the full and partial prior applications of CLinSEPAL to brain data, given in Sec. 7. Specifically, Fig. 8 depicts the ground truth linear CA and the learned linear CA by CLinSEPAL for the full prior setting; whereas Fig. 10 the results for the partial prior setting. Regarding the partial prior setting, we also report in Fig. 9 the partial prior received as an input by CLinSEPAL for all the settings, and in Fig. 11 the monitored metrics to better understand the performance of CLinSEPAL with varying degree of uncertainty (low, medium, high), as discussed in Sec. 7. The color coding for the partial prior setting refers to the following classification, reported unaltered from (D'Acunto et al., 2024):

- Red for ROIs corresponding to cognitive functions, attention, emotion, and decision-making;   
- Orange for those related to auditory processing, speech and language processing, and memory;   
- Blue for those concerning memory formation and memory retrieval;   
- Pink for those associated with sensory integration and somatosensory;   
- Purple for the ROIs within the visual network and related to the visual memory;   
- Green for those within the motor network;   
- Yellow for those regarding the motor control and the posture.

![](images/9a150456a346ab92a196a8cc0c7caea5bd95bd35bf758fee89d95c87e6a84eab.jpg)  
Figure 8: The figure shows (top) the ground truth linear CA and (bottom) the learned linear CA for the simulated full prior setting in Sec. 7.

![](images/623602cdd96da153e6de565c63770243449e76b1cd417243308583a193695248.jpg)  
Figure 9: Starting from the top, the figure shows (i) the ground truth structure for linear CA, and the partial prior provided as input to CLinSEPALlearned linear CA for the simulated partial prior in the (ii) low, (iii) medium, and (iv) high uncertainty settings discussed in Sec. 7.

![](images/9b4800d4e699e31ff38f712d6385f6799e4df32d8fc855dd78c2edc975ff4f99.jpg)  
Figure 10: Starting from the top, the figure shows (i) the ground truth linear CA, and the learned linear CA for the simulated partial prior setting with (ii) low, (iii) medium, and (iv) high uncertainty in Sec. 7.

![](images/59545f3d818b4d6bcc3cac2b5643fb2c6ec928315d9c81baa19ce03a0072f72d.jpg)  
Figure 11: Starting from the left, the figure provides the (i) the KL divergence evaluated at the learned $\widehat{V}$ , (ii) the Frobenious absolute distance, (iii) the true positive rate, and (iv) the false discovery rate for the simulated partial prior setting with low, medium, and high uncertainty in Sec. 7.