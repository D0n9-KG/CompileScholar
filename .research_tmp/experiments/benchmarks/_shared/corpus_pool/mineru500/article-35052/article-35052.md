# Predictive Modeling of Homeless Service Assignment: A Representation Learning Approach

Khandker Sadia Rahman, Charalampos Chelmis

Department of Computer Science, University at Albany, SUNY

Albany, New York, USA

{srahman2, cchelmis}@albany.edu

# Abstract

In recent years, there has been growing interest in leveraging machine learning for homeless service assignment. However, the categorical nature of administrative data recorded for homeless individuals hinders the development of accurate machine learning methods for this task. This work asserts that deriving latent representations of such features, while at the same time leveraging underlying relationships between instances is crucial in algorithmically enhancing the existing assignment decision-making process. Our proposed approach learns temporal and functional relationships between services from historical data, as well as unobserved but relevant relationships between individuals to generate features that significantly improve the prediction of the next service assignment compared to the state-of-the-art.

Code — https://github.com/IDIASLab/REPLETE

# 1 Introduction

Machine learning has gained significant attention for its ability to solve complex real-world problems across various domains including criminal justice, e-commerce, healthcare, banking, finance, and social service (Sarker 2021). At the same time, a constant rise in the rate of homelessness has been observed (Dej, Gaetz, and Schwan 2020). In the United States alone, homelessness has risen 12.5% between 2022 and 2023, with a particularly sharp increase of 29.5% in chronic patterns of homelessness (Henry et al. 2023). Chronic pattern of homelessness occurs when individuals experience repeated homelessness over an extended period (e.g., at least two years), or endure continuous homelessness for at least 12 months (Fleury et al. 2021).

To date, various machine learning approaches have been proposed for predicting whether individuals will reenter the homeless system (Gao, Das, and Fowler 2017; Hong et al. 2018; Kube, Das, and Fowler 2019) as well as assessing the risk of chronic homelessness of individuals (VanBerlo et al. 2021; Messier, John, and Malik 2021). However, only limited work has addressed the more challenging problem of homeless service assignment (Rahman and Chelmis 2022; Rahman, Zois, and Chelmis 2023; Pokharel, Das, and Fowler 2024). Specifically, (Pokharel, Das, and Fowler 2024) investigate simple models (e.g., Decision Trees with Short Explainable Rules) to recommend the exact service individuals can benefit from, while (Rahman and Chelmis 2022; Rahman, Zois, and Chelmis 2023) explicitly model the trajectories of individuals within the homeless system. The main limitation of these methods lies in their inability to capture the informative relationships between different feature values (i.e., within-feature interactions) and the complex between-feature interactions.

Administrative data collected by the homeless service providers comprises of (i) service assignments and duration of stay, (ii) housing outcome after service assignments, (iii) demographics (e.g., race, gender, ethnicity), educational history, disabling condition, and (iv) time-variant information (e.g., monthly income, health). Most of these features are categorical, and existing methods use one-hot encoding to handle them, mainly for simplicity. Unfortunately, such treatment misses the rich relationships within the features, while exponentially increasing the dimensionality of the data with growing number of categories (Rodríguez et al. 2018). Instead, we propose a predictive model that learns latent representations for features and services, which are then utilized to capture feature interactions and relationships between individuals. We show that the learned representations significantly improve the accuracy of homeless service assignment compared to the state-of-the-art.

This brings us to the core contribution of our work. Similar to (Rahman and Chelmis 2022; Rahman, Zois, and Chelmis 2023) we study the interactions between services found in individuals' history. However, those studies oversimplify the task of predicting the likelihood of reaching the next service based solely on transitional probabilities between past services, while we assert that many other interactions exist that provide valuable context which can in turn enhance predictive performance. Specifically, subsequent services are often assigned to individuals to serve a particular purpose at a specific time. Understanding such functional and temporal relationship between services can provide insights into why a particular service is assigned at a specific time. Our work extends beyond previous studies by incorporating temporal and functional interaction between services. Furthermore, our model explores the relationships between individuals. Generally, individuals with similar features tend to receive similar services, and leveraging such

information can significantly improve the robustness of predictive models. While (Rahman and Chelmis 2022) incorporates finding the “most similar individual” in their prediction problem, it does so as an information retrieval task; instead our approach clusters individuals and utilizes the collective patterns within clusters to provide better insights and more accurate predictions.

In summary, given the history of services assigned to individuals and their current socio-economic features, we propose an objective function with three distinct components that captures $(i)$ temporal and $(ii)$ functional interactions between services, and $(iii)$ interactions between individuals (instances). Through optimization, we obtain latent representations for services and features, and identify the clusters to which each individual belongs. Next, features derived from these representations are input into a feed-forward neural network for predicting the next service assignment. We state our main contributions as follows:

1. We propose a representation-based predictive model for homeless service assignment that effectively learns the latent representation of services and features.   
2. We design a novel optimization function that incorporates temporal, functional, and instance interactions to enhance the learning of representations.   
3. We utilize the learnt representations to derive features that significantly improve the performance of service assignment predictions.

The rest of the paper is organized as follows. Section 2 summarizes the related work. Section 3 delineates the problem setting. Section 4 introduces the proposed approach. Section 5 describes the data, metrics, and baselines. Section 6 provides detailed discussion of the experimental results. Section 7 concludes with a discussion of the limitations of our study and potential directions for future work.

# 2 Related Work

# Reentry Prediction and Risk of Chronic Homelessness

A considerable body of prior work has focused on predicting reentry into homelessness and assessing the risk of homelessness or chronic homelessness. (Gao, Das, and Fowler 2017; Hong et al. 2018) investigated the application of machine learning models such as logistic regression and random forest to predict whether individuals will reenter homelessness. On the other hand, (VanBerlo et al. 2021; Messier, John, and Malik 2021) explored neural networks to assess the risk of chronic homelessness. Additionally, (Vajiac et al. 2024) investigated various machine learning models to evaluate the risk of eviction–caused homelessness and accurately identify individuals in need of assistance. All of these works are formulated as binary classification problems. Our work, on the other hand, focuses on developing a predictive model for service assignment task, which is a more challenging multi-class classification problem.

# Homeless service assignment

Another body of work focuses on predicting the service assignment at various conditions, such as (Toros and Flaming 2018; Shinn et al. 2013; Greer et al. 2016) develop models that prioritize housing for those at high risk of homelessness. In contrast, our work centers on individuals who have already experienced chronic patterns of homelessness. On the other hand, (Chelmis et al. 2021) explores machine learning models to predict the exact support corresponding to the assigned service. Similary, (Pokharel, Das, and Fowler 2024) employs decision trees with simple explainable rules (SER–DT) algorithm, which divides the problem into multiple binary classification tasks using a one-versus-all prediction approach for service assignment upon entry. While these studies use one-hot encoding for categorical features, in contrast, our approach leverages representation learning to uncover the rich relationships within these features, moving beyond one-hot encoding. Our research aligns with (Rahman and Chelmis 2022) and (Rahman, Zois, and Chelmis 2023) which address the multi-class service assignment task. Specifically, (Rahman and Chelmis 2022) infers a homelessness network and (Rahman, Zois, and Chelmis 2023) utilizes a Bayesian network to predict the next service assignment. In contrast, our work integrates both history of individuals within the homeless system and socio-economic features to predict the next service assignment. A key distinction is that while previous studies explore simple transitional relationships between services, our approach leverages more complex (temporal and functional) relationships between services, and relevant interactions between individuals to predict the next service assignment.

# 3 Problem Setting

We denote the set of individuals with chronic patterns of homelessness as $U = \{u_{1}, \ldots, u_{|\mathcal{U}|}\}$ . The set of homeless services, such as permanent housing, rapid rehousing, day shelter, is denoted by $A = \{a^{1}, a^{2}, \ldots, a^{|\mathcal{A}|}\}$ . For each $u_{i} \in U$ , let $\mathcal{T}_{u_{i}} = \{(a_{t_{1}}, t_{1}), \ldots, (a_{t_{N}}, t_{N})\}$ represent the history of services assigned to individual $u_{i}$ , where $a_{t_{i}}$ signifies assignment to a service at time $t_{i}$ . Next, we denote $X \in R^{|U| \times |F|}$ as the feature matrix which comprises the feature set $F \in R^{d}$ of individuals $u_{i} \in U$ at time $t_{N+1}$ .

Given matrix X and the history of prior service assignments $\mathcal{T}_{u_{i}}=\{(a_{t_{1}},t_{1}),\ldots,(a_{t_{N}},t_{N})\}$ for individuals $u_{i}\in U$ , the goal of this paper is to accurately predict the next service assignment $a_{t_{N+1}}\in Y$ for $\forall u_{i}$ .

# 4 Proposed Approach

In this section, we present a comprehensive overview of REPLETE, a REPresentation Learning-based sEvice assignmentT modEl, which consists of two main components: (i) representation learning framework and (ii) prediction model. The representation learning framework is designed to learn the latent representations of services. The prediction model leverages the learned representations to enhance the homeless service assignment decision-making process. Figure 1 provides an overview of REPLETE.

# 4.1 Representation Learning Framework

To effectively learn the representations of services, we design an optimization function that captures three differ-

![](images/06511ee1d510d12e58737c6ab93a08dd07b107a77cf1a0c071f5af45d8913a0b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph "Representation Learning Framework"
        X["X"] --> C["C"]
        D["D"] --> A["A"]
        T["T"] --> A["A"]
        H["H"] --> A["A"]
        C --> V["V^T"]
        A --> S["S^T"]
        R_p["R_p^T"] --> R_s["R_s"]
        A_T["A^T"] --> R_s
    end

    subgraph "Derived features"
        D1["●"] --> E1["●"]
        D2["●"] --> E2["●"]
        D3["●"] --> E3["●"]
        D4["●"] --> E4["●"]
        D5["●"] --> E5["●"]
        E1 --> E6["●"]
        E2 --> E7["●"]
        E3 --> E8["●"]
        E4 --> E9["●"]
        E5 --> E10["●"]
        E6 --> Output["●"]
        E7 --> Output
        E8 --> Output
        E9 --> Output
    end

    subgraph "Output"
        Output1["●"] --> E10["●"]
        Output2["●"] --> E11["●"]
        Output3["●"] --> E12["●"]
        Output4["●"] --> E13["●"]
        Output5["●"] --> E14["●"]
        Output6["●"] --> E15["●"]
        Output7["●"] --> E16["●"]
        Output8["●"] --> E17["●"]
        Output9["●"] --> E18["●"]
        Output10["●"] --> E19["●"]
        Output11["●"] --> E20["●"]
        Output12["●"] --> E21["●"]
        Output13["●"] --> E22["●"]
        Output14["●"] --> E23["●"]
        Output15["●"] --> E24["●"]
        Output16["●"] --> E25["●"]
        Output17["●"] --> E26["●"]
        Output18["●"] --> E27["●"]
        Output19["●"] --> E28["●"]
        Output20["●"] --> E29["●"]
    end
```
</details>

Figure 1: Overview of REPLETE. Representation learning framework first learns representations A, C, V, S, $R_{p}$ , $R_{s}$ . These are used to derive features, which are subsequently input into FFNN for service assignment prediction.

ent contexts. The first context focuses on the temporal interactions between services, where assignments at different stages in the homeless system are likely to have distinct vector representations. The second context captures the functional interactions between services, where assignments serving similar needs share similar vector representations. The last context models relationships between individuals, where similar individuals are more likely to exhibit similar traits.

Temporal Context Here, our goal is to learn the latent representations of services and time units based on the intuitive observation that services occurring within similar time units should have analogous vector representations. To formalize this, we define $D \in R^{|A| \times |\tau|}$ , where $D_{ij}$ represents the frequency with which service $a^{i}$ appears in time unit $\tau_{j}$ . Here, $\tau$ denotes the set of time units (e.g., weeks, months, or years). To derive standardized representations, we employ Non-negative Matrix Factorization (NMF). Specifically, given D, we learn the non-negative matrices, $A \in R^{|A| \times k}$ for services and $S \in R^{| \tau | \times k}$ for time units by solving

$$
\begin{array}{l} \min _ {\mathbf {A}, \mathbf {S} \geq 0} \| \mathbf {D} - \mathbf {A} \mathbf {S} ^ {\top} \| _ {F} ^ {2} + \alpha \sum_ {j = 1} ^ {| \tau | - 1} \| \mathbf {S} _ {j,:} - \mathbf {S} _ {j - 1,:} \| _ {F} ^ {2} \\ + \lambda (\| \mathbf {A} \| _ {F} ^ {2} + \| \mathbf {S} \| _ {F} ^ {2}), \\ \end{array}
$$

where $\alpha$ , $\lambda$ , and k control the smoothness, sparsity constraints, and dimension of representations, respectively.

Functional Context This context is based on the premise that subsequent services are assigned to individuals because they address needs unmet by the preceding service, meaning, the two services together complement each other in satisfying a need. In this context, we capture the latent relationships between such service pairs. We define a transitional probability matrix $T \in R^{|\mathcal{A}| \times |\mathcal{A}|}$ , where each element $T_{ij}$ denotes the probability of assigning service $a^{j}$ after service $a^{i}$ (Rahman and Chelmis 2022). Consequently, we learn m latent relations between services represented by matrices, $R_{p} \in R^{m \times k}$ for preceding assignment and $R_{s} \in R^{m \times k}$ for succeeding assignments such that for any pair $(a^{i}, a^{j})$ , $T_{ij} = \sum_{l=1}^{m} a^{i} R_{p}[l] + a^{j} R_{s}[l]$ . We perform symmetric NMF (Kuang, Ding, and Park 2012) to obtain the solution to

$$
\begin{array}{l} \min _ {\mathbf {A}, \mathbf {R} _ {\mathbf {p}}, \mathbf {R} _ {\mathbf {s}} \geq 0} \| \mathbf {T} - \mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \| _ {F} ^ {2} + \\ + \lambda (\| \mathbf {A} \| _ {F} ^ {2} + \| \mathbf {R _ {p}} \| _ {F} ^ {2} + \| \mathbf {R _ {s}} \| _ {F} ^ {2}), \\ \end{array}
$$

where A denotes the latent representations of services. Similar to the temporal context, we incorporate a sparsity constraint with $\lambda$ controlling the penalty for overfitting, and k denoting the dimension of the representations.

Individual Context In this context, our objective is to learn the latent representations of services denoted by A and features denoted by $V \in R^{|\mathcal{F}| \times k}$ , while also identifying clusters to which individuals belong (represented by $C \in R^{|\mathcal{U}| \times k}$ ). By clustering individuals with similar features and past service assignments, our goal is to ensure that services assigned to similar individuals (i.e., in the same cluster) have similar representations. We define $H \in R^{|\mathcal{A}| \times |U|}$ , where each element $H_{ij}$ denotes the number of times individual $u_j$ is assigned to service $a^i$ . Since X and H exhibit higher sparsity, $L_{2-1}$ norm is applied to prevent rows with greater sparsity from dominating the objective function. Given matrices H and X, we solve

$$
\begin{array}{l} \min _ {\mathbf {A}, \mathbf {V} \geq 0, \mathbf {C}} \| \mathbf {H} - \mathbf {A C} ^ {\top} \| _ {2, 1} + \| \mathbf {X} - \mathbf {C V} ^ {\top} \| _ {2, 1} + \lambda (\| \mathbf {A} \| _ {F} ^ {2} \\ + \| \mathbf {V} \| _ {F} ^ {2} + \| \mathbf {C} \| _ {F} ^ {2}) + \beta t r (\mathbf {C} ^ {\top} \boldsymbol {\Gamma} \mathbf {C}) \\ s. t. \mathbf {C 1} _ {k} = \mathbf {1} _ {| \mathcal {U} |}, \mathbf {C} \in [ 0, 1 ]. \\ \end{array}
$$

Here, the term $tr(\mathbf{C}^{\top}\mathbf{\Gamma}\mathbf{C})$ ensures that similar individuals are clustered together and $\Gamma\in R^{|\mathcal{U}|\times|\mathcal{U}|}$ records the cosine similarity between individuals in the set U. The constraints on C enforce soft clustering by ensuring that the rules of probability are satisfied. Such soft clustering is appropriate for homelessness, a complex domain where individuals can exhibit similarities with members of different clusters. Finally, $\beta$ controls the strictness of the clusters by determining how tightly or loosely data instances are grouped together.

Overall Optimization Function With all the previously introduced components, we formulate our joint optimization problem as follows:

$$
\begin{array}{l} \min _ {\mathbf {A}, \mathbf {S}, \mathbf {V}, \mathbf {R} _ {\mathbf {p}}, \mathbf {R} _ {\mathbf {s}} \geq 0, C} \| \mathbf {D} - \mathbf {A} \mathbf {S} ^ {\top} \| _ {F} ^ {2} + \alpha \sum_ {j = 1} ^ {| \tau | - 1} \| \mathbf {S} _ {j,:} - \mathbf {S} _ {j - 1,:} \| _ {F} ^ {2} \\ + \left\| \mathbf {H} - \mathbf {A C} ^ {\top} \right\| _ {2, 1} + \left\| \mathbf {X} - \mathbf {C V} ^ {\top} \right\| _ {2, 1} + \beta t r (\mathbf {C} ^ {\top} \boldsymbol {\Gamma} \mathbf {C}) \\ + \left\| \mathbf {T} - \mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \right\| _ {F} ^ {2} + \lambda (\left\| \mathbf {A} \right\| _ {F} ^ {2} + \left\| \mathbf {S} \right\| _ {F} ^ {2} + \left\| \mathbf {V} \right\| _ {F} ^ {2} \\ + \| \mathbf {C} \| _ {F} ^ {2} + \| \mathbf {R _ {p}} \| _ {F} ^ {2} + \| \mathbf {R _ {s}} \| _ {F} ^ {2}) \\ s. t. \mathbf {C} \mathbf {1} _ {k} = \mathbf {1} _ {| \mathcal {U} |}, \mathbf {C} \in [ 0, 1 ]. \tag {1} \\ \end{array}
$$

# 4.2 Optimization Algorithm

Jointly updating the variables in Eq. (1) causes the objective function to become non-convex. We therefore optimize

the objective function using Alternating Direction Method of Multiplier (ADMM) (Boyd et al. 2011), where variables are updated separately. All proofs are provided in the Appendix section.

We begin by relaxing the constraints on C to orthogonality, specifically $C^{\top}C = I$ , $C \geq 0$ (Tang and Liu 2012). We then introduce two auxiliary variables: $P = H - AC^{\top}$ and $Q = X - CV^{\top}$ . Consequently, Eq. (1) is reformulated into the following equivalent problem:

$$
\begin{array}{l} \min _ {\mathbf {A}, \mathbf {S}, \mathbf {V}, \mathbf {R} _ {\mathbf {p}}, \mathbf {R} _ {\mathbf {s}}, \mathbf {C} \geq 0, \mathbf {P}, \mathbf {Q}} \| \mathbf {D} - \mathbf {A} \mathbf {S} ^ {\top} \| _ {F} ^ {2} + \alpha \| \mathbf {S B} ^ {\prime} \| _ {F} ^ {2} + \| \mathbf {P} \| _ {2, 1} \\ + \| \mathbf {Q} \| _ {2, 1} + \beta t r (\mathbf {C} ^ {\top} \mathbf {\Gamma} \mathbf {C}) + \| \mathbf {T} - \mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \| _ {F} ^ {2} \\ + \lambda (\| \mathbf {A} \| _ {F} ^ {2} + \| \mathbf {S} \| _ {F} ^ {2} + \| \mathbf {V} \| _ {F} ^ {2} + \| \mathbf {C} \| _ {F} ^ {2} + \| \mathbf {R _ {p}} \| _ {F} ^ {2} + \| \mathbf {R _ {s}} \| _ {F} ^ {2}) \\ + \left\langle \mathcal {L}, \mathbf {X} - \mathbf {C V} ^ {\top} - \mathbf {Q} \right\rangle + \left\langle \mathcal {K}, \mathbf {H} - \mathbf {A C} ^ {\top} - \mathbf {P} \right\rangle \\ + \left\langle \mathcal {N}, \mathbf {C} ^ {\top} \mathbf {C} - \mathbf {I} \right\rangle + \frac {\mu}{2} \| \mathbf {H} - \mathbf {A C} ^ {\top} - \mathbf {P} \| _ {F} ^ {2} \\ + \frac {\mu}{2} \| \mathbf {X} - \mathbf {C V} ^ {\top} - \mathbf {Q} \| _ {F} ^ {2}, \tag {2} \\ \end{array}
$$

where K, L, and N are Lagrangian multipliers, $\mu$ controls the penalty for violating the equality constraints, $\langle\cdot,\cdot\rangle$ denotes the dot product, and $B'$ represents the matrix equivalent to the smoothing constraint. Next, we summarize the update rules for each variable.

Update P For updating P, we hold the other variables constant and remove the terms that are irrelevant to P. Eq. (2) can be written as:

$$
\min _ {\mathbf {P}} \frac {1}{2} \| \mathbf {P} - (\mathbf {H} - \mathbf {A C} ^ {T} + \frac {1}{\mu} \mathcal {K}) \| _ {F} ^ {2} + \frac {1}{\mu} \| \mathbf {P} \| _ {2, 1} \tag {3}
$$

Lemma 1 provides a closed form solution for Eq. (3).

Lemma 1 (Wang, Tang, and Liu 2015) Given matrix E and a positive scale $\alpha$ , the $i^{th}$ row of the optimal solution $W^{*}$ of $\min_{W}\frac{1}{2}\|W-E\|_{F}^{2}+\alpha\|W\|_{2,1}$ is given by:

$$
w _ {i} ^ {*} = \left\{ \begin{array}{l l} (1 - \frac {\alpha}{\| e _ {i} \|}) e _ {i}, & \| e _ {i} \| > \alpha \\ 0, & o t h e r w i s e \end{array} \right.
$$

Using Lemma 1, the optimal solution $\mathbf{P}^*$ for the above equation is as follows, where $\mathbf{E}^{\mathbf{P}} = \mathbf{H} - \mathbf{A}\mathbf{C}^{T} + \frac{1}{\mu}\mathcal{K}$ :

$$
\mathbf {P} _ {i,:} ^ {*} = \left\{ \begin{array}{l l} (1 - \frac {1}{\mu \| \mathbf {E} _ {i , :} ^ {\mathbf {P}} \|}) \mathbf {E} _ {i,:} ^ {\mathbf {P}}, & \| \mathbf {E} _ {i,:} ^ {\mathbf {P}} \| > \frac {1}{\mu} \\ 0, & \text {otherwise} \end{array} \right.
$$

Update Q The optimal solution $Q^{*}$ is obtained similar to P, using Lemma 1, where $E^{Q} = X - CV^{T} + \frac{1}{\mu}L$ :

$$
\mathbf {Q} _ {i,:} ^ {*} = \left\{ \begin{array}{l l} (1 - \frac {1}{\mu \| \mathbf {E} _ {i , :} ^ {\mathbf {Q}} \|}) \mathbf {E} _ {i,:} ^ {\mathbf {Q}}, & \| \mathbf {E} _ {i,:} ^ {\mathbf {Q}} \| > \frac {1}{\mu} \\ 0, & \text { otherwise } \end{array} \right.
$$

Update S Let $\psi_{S}$ be the Lagrangian multiplier for $S \geq 0$ , the Lagrangian function related to S is, $O_{S} = \min \|D - AS^{T}\|_{F}^{2} + \lambda\|S\|_{F}^{2} + \alpha\|SB'\|_{F}^{2} - tr(\psi_{S}S^{\top})$ . The partial derivative of $O_{S}$ is $\frac{1}{2}\frac{dO_{S}}{dS} = -(D - AS^{\top})^{\top}A +$ $\alpha (\mathbf{S}\mathbf{B}^{\prime})\mathbf{B}^{\prime} + \lambda \mathbf{S} - \psi_{\mathbf{S}}$ . Using the Karush-KuhnTucker (KKT) complementary condition (Boyd and Vandenberghe 2004), i.e., $\psi_{\mathbf{S}}(i,j)\mathbf{S}_{ij} = 0$ , we get $\mathbf{S}_{ij}\gets \mathbf{S}_{ij}\frac{\hat{\mathbf{S}}_{ij}}{\tilde{\mathbf{S}}_{ij}}$ (Zhang and Zhang 2017), where

$$
\hat {\mathbf {S}} _ {i j} = \mathbf {D} ^ {\top} \mathbf {A} + [ \alpha (\mathbf {S} (\mathbf {B} ^ {\prime}) ^ {2}) ] ^ {-}
$$

$$
\tilde {\mathbf {S}} _ {i j} = \mathbf {S A} ^ {\top} \mathbf {A} + \lambda \mathbf {S} + [ \alpha (\mathbf {S} (\mathbf {B} ^ {\prime}) ^ {2}) ] ^ {+}.
$$

Here, for any matrix $\mathbf{M}$ , $(\mathbf{M})^{+} = \frac{ABS(\mathbf{M}) + \mathbf{M}}{2}$ and $(\mathbf{M})^{-} = \frac{ABS(\mathbf{M}) - \mathbf{M}}{2}$ are the positive and negative part of $\mathbf{M}$ , respectively, and $ABS(\mathbf{M})$ consists of the absolute value of elements in $\mathbf{M}$ (Shu, Wang, and Liu 2019).

Update $R_{p}$ and $R_{s}$ . The partial derivative of the Lagrangian objective function w.r.t. $R_{p}$ is $\frac{1}{2}\frac{d\mathcal{O}_{R_{p}}}{dR_{p}} = -TAA^{\top}R_{s} + AR_{p}^{\top}R_{s}A^{\top}AA^{\top}R_{s} + \lambda R_{p} - \psi_{R_{p}}$ . Using the KKT complementary condition, we get $R_{p_{ij}} \leftarrow R_{p_{ij}} \frac{\hat{R}_{p_{ij}}}{\tilde{R}_{p_{ij}}}$ , where

$$
\hat {\mathbf {R}} _ {\mathbf {p} _ {i j}} = \mathbf {T A A} ^ {\top} \mathbf {R} _ {\mathbf {s}} + \left(\mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A A} ^ {\top} \mathbf {R} _ {\mathbf {s}}\right) ^ {-}
$$

$$
\tilde {\mathbf {R}} _ {\mathbf {p} _ {i j}} = \lambda \mathbf {R} _ {\mathbf {p}} + (\mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A} \mathbf {A} ^ {\top} \mathbf {R} _ {\mathbf {s}}) ^ {+}.
$$

Similar to $\mathbf{R}_{\mathbf{p}}$ , we get $\mathbf{R}_{\mathbf{s}_{ij}} \leftarrow \mathbf{R}_{\mathbf{s}_{ij}} \frac{\hat{\mathbf{R}}_{\mathbf{s}_{ij}}}{\tilde{\mathbf{R}}_{\mathbf{s}_{ij}}} \text{ for } \mathbf{R}_{\mathbf{s}}$ , such that:

$$
\hat {\mathbf {R}} _ {\mathbf {s} _ {i j}} = \mathbf {T A A} ^ {\top} \mathbf {R} _ {\mathbf {p}} + (\mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A A} ^ {\top} \mathbf {R} _ {\mathbf {p}}) ^ {-}
$$

$$
\tilde {\mathbf {R}} _ {\mathbf {s} _ {i j}} = \lambda \mathbf {R} _ {\mathbf {s}} + (\mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A} \mathbf {A} ^ {\top} \mathbf {R} _ {\mathbf {p}}) ^ {+}.
$$

Update V The partial derivative of the Lagrangian objective function w.r.t. $\mathbf{V}$ is $\frac{d\mathcal{O}_{\mathbf{V}}}{d\mathbf{V}} = -\mu (\mathbf{X} - \mathbf{C}\mathbf{V}^{\top} - \mathbf{Q})^{\top}\mathbf{C}-$ $\mathcal{L}^{\top}\mathbf{C} + 2\lambda \mathbf{V} - \psi_{\mathbf{V}}$ . Using the KKT complementary condition, we get $\mathbf{V}_{ij}\gets \mathbf{V}_{ij}\frac{\hat{\mathbf{V}}_{ij}}{\tilde{\mathbf{V}}_{ij}}$ , where

$$
\hat {\mathbf {V}} _ {i j} = \mu \mathbf {X} ^ {\top} \mathbf {C} + (\mathcal {L} ^ {\top} \mathbf {C}) ^ {+}
$$

$$
\tilde {\mathbf {V}} _ {i j} = \mu \mathbf {V} \mathbf {C} ^ {\top} \mathbf {C} + \mu \mathbf {Q} ^ {\top} \mathbf {C} + (\mathcal {L} ^ {\top} \mathbf {C}) ^ {-} + 2 \lambda \mathbf {V}.
$$

Update C The partial derivative of the Lagrangian objective function w.r.t. C is $\frac{1}{2}\frac{dO_{C}}{dC} = -\mu(\mathbf{H}-\mathbf{A}\mathbf{C}^{\top}-\mathbf{P})^{\top}\mathbf{A} - \mu(\mathbf{X}-\mathbf{C}\mathbf{V}^{\top}-\mathbf{Q})\mathbf{V} - \mathcal{L}\mathbf{V} - \mathcal{K}^{\top}\mathbf{A} + 2\mathbf{C}\mathcal{N} + 2\beta\mathbf{\Gamma}\mathbf{C} + 2\lambda\mathbf{C} - \psi_{\mathbf{C}}$ . Using the KKT complementary condition, we get $C_{ij} \leftarrow C_{ij}\frac{\hat{C}_{ij}}{C_{ij}}$ , where

$$
\begin{array}{l} \hat {\mathbf {C}} _ {i j} = \mu \mathbf {H} ^ {\top} \mathbf {A} + \mu \mathbf {X V} + (\mathcal {L V}) ^ {+} + (\mathcal {K} ^ {\top} \mathbf {A}) ^ {+} + 2 (\mathbf {C N}) ^ {-} \\ + 2 (\beta (\mathbf {\Gamma C}) ^ {-} \\ \tilde {\mathbf {C}} _ {i j} = \mu \mathbf {C A} ^ {\top} \mathbf {A} + \mu \mathbf {P} ^ {\top} \mathbf {A} + \mu \mathbf {C V} ^ {\top} \mathbf {V} + \mu \mathbf {Q V} + (\mathcal {L V}) ^ {-} \\ + (\mathcal {K} ^ {\top} \mathbf {A}) ^ {-} + 2 (\mathbf {C N}) ^ {+} + 2 \beta (\boldsymbol {\Gamma C}) ^ {+} + 2 \lambda \mathbf {C}. \\ \end{array}
$$

Update A The partial derivative of the Lagrangian objective function w.r.t. A is $\frac{1}{2}\frac{dO_{A}}{dA} = -2(D - AS^{\top})S - 4(T - AR_{p}^{\top}R_{s}A^{\top}) \times AR_{p}^{\top}R_{s} + 2\lambda A - \mu(H - AC^{\top} - P)C - KC - \psi_{A}$ . Using the KKT complementary conditions, we

get $\mathbf{A}_{ij} \leftarrow \mathbf{A}_{ij} \frac{\hat{\mathbf{A}}_{ij}}{\tilde{\mathbf{A}}_{ij}}$ where

$$
\begin{array}{l} \hat {\mathbf {A}} _ {i j} = 2 \mathbf {D S} + 4 \mathbf {T A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} + 4 (\mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}}) ^ {-} \\ + \mu \mathbf {H C} + (\mathbf {K C}) ^ {+} \\ \tilde {\mathbf {A}} _ {i j} = 2 \mathbf {A} \mathbf {S} ^ {\top} \mathbf {S} + 4 (\mathbf {A} \mathbf {R _ {p}} ^ {\top} \mathbf {R _ {s}} \mathbf {A} ^ {\top} \mathbf {A} \mathbf {R _ {p}} ^ {\top} \mathbf {R _ {s}}) ^ {+} + 2 \lambda \mathbf {A} \\ + \mu \mathbf {A} \mathbf {C} ^ {\top} \mathbf {C} + \mu \mathbf {P} \mathbf {C} + (\mathbf {K} \mathbf {C}) ^ {-}. \\ \end{array}
$$

Update L, M, and N After updating the variables, we update the Lagrangian parameters (Wang, Tang, and Liu 2015) as follows:

$$
\mathcal {L} = \mathcal {L} + \mu (\mathbf {X} - \mathbf {C V} ^ {\top} - \mathbf {Q})
$$

$$
\mathcal {M} = \mathcal {M} + \mu (\mathbf {H} - \mathbf {A C} ^ {\top} - \mathbf {P})
$$

$$
\mathcal {N} = \mathcal {N} + \mu (\mathbf {C} ^ {\top} \mathbf {C} - \mathbf {I}).
$$

# 4.3 Service Assignment Prediction

In this section, we delve into the specifics of the service assignment module of REPLETE. Once the representations are obtained by solving Eq. (1), we derive three sets of features: (i) service and feature representations, (ii) feature interactions, and (iii) instance interactions. The overall process is summarized in Algorithm 1.

Service and Feature Representations We utilize the representations of services and features in our prediction model instead of one-hot encoding them. The rational is that by treating services and features as categorical our model would miss the inherent relationships within them. Therefore, we replace the services $\{a_{t_{1}},\ldots,a_{t_{N}}\}$ within the history of services received by an individual, with their respective representations $\{A_{a_{t_{1}}},\ldots,A_{a_{t_{N}}}\}$ , where $A_{a_{t_{i}}}$ denotes the representation of service $a_{t_{i}}$ . We additionally obtain the representations of the socio-economic features for each individual by multiplying X and V, where X is the feature matrix and V is the matrix of learned feature representations.

Feature Interactions Beyond the representations of services and features, we further incorporate two categories of service interactions: temporal and functional (Section 4.1). Given the sequence of services $\{a_{t_1},\ldots ,a_{t_N}\}$ received by individuals, the temporal and functional interactions within each service pair $(a_{t_i},a_{t_j})$ are defined as $\mathbf{A}_{a_{t_i}}\mathbf{S}^\top \mathbf{S}\mathbf{A}_{a_{t_j}}^\top$ and $\mathbf{A}_{a_{t_i}}\mathbf{R}_p^\top \mathbf{R}_s\mathbf{A}_{a_{t_j}}^\top$ , respectively. Matrix $\mathbf{A}_{a_{t_i}}$ denotes the representation of service $a_{t_i}$ , whereas matrices $\mathbf{S}$ , $\mathbf{R}_p$ , and $\mathbf{R}_s$ are the learned representations for time units, preceding assignments, and succeeding assignments, respectively.

Instance Interactions We define instance interaction as the similarity between individuals, which we use to identify clusters within the individuals. Since the relaxed optimization on C (i.e., the matrix denoting the clusters to which individuals belong) does not directly provide probabilities for soft clustering, we normalize C such that each row sums to 1, thereby obtaining these probabilities.

Prediction Model We use $Z \in R^{|\mathcal{U}|\times|\mathcal{F}^{\prime}|}$ , to denote the concatenated feature vector, i.e., service and feature representations, feature interactions, and instance interactions, where U is the set of individuals, and $F'$ is the set of derived features. For the service assignment task, we utilize a single hidden layer feed–forward neural network (FFNN). The rationale behind a single hidden layer architecture is that it balances complexity and computational efficiency.

Predicted service assignments, denoted by $\hat{Y}$ , are calculated as $\hat{\mathbf{Y}} = \sigma(\mathbf{Z}\mathbf{W}_{1} + \mathbf{b}_{1})\mathbf{W}_{2} + \mathbf{b}_{2}$ where, $W_{1}$ and $W_{2}$ are the weight matrices for the hidden and output layers, respectively, $b_{1}$ and $b_{2}$ are the bias vectors for the hidden and output layers, respectively, and $\sigma(\cdot)$ is the ReLU activation function. The model is trained using cross-entropy loss.

Algorithm 1: REPLETE   
Require: X, D, T, H, B', α, β, μ, λ, τ, k
Ensure: $\hat{a}_{t_{N+1}}$ 1: Randomly initialize A, S, C, V, Rp, Rs, P, Q
2: Pre-compute similarity matrix Γ
3: repeat
4: Update A, S, C, V, Rp, Rs, P, Q according to Section 4.2
5: until convergence
6: Normalize C s.t. $\sum_{j} C_{i,j} = 1$ 7: for each $u_i \in U$ do
8: $Z_{u_i} = Z_{u_i} \cup (XV)_{u_i} \cup C_{u_i}$ 9: for each $a_{t_k} \in T_{u_i}$ do
10: $Z_{u_i} = Z_{u_i} \cup A_{a_{t_k}}$ 11: end for
12: for each $(a_{t_m}, a_{t_n}) \in T_{u_i}$ do
13: $Z_{u_i} = Z_{u_i} \cup (A_{a_{t_m}} S^\top S A_{a_{t_n}}^\top)$ 14: $Z_{u_i} = Z_{u_i} \cup (A_{a_{t_m}} R_p^\top R_s A_{a_{t_n}}^\top)$ 15: end for
16: end for
17: Split Z into training $Z_{train}$ and testing set $Z_{test}$ 18: Train FFNN with $Z_{train}$ 19: Input $Z_{test}$ into FFNN
20: $\hat{Y}_{test} \leftarrow$ Output of FFNN
21: return $\hat{Y}_{test}$

# 5 Experimental Setup

Data Description Our analysis utilizes an anonymized dataset from CARES of NY comprising 18,817 records from 6,011 chronically homeless individuals in the Capital Region of New York State, covering the period from 2012 to 2018, including a total of 9 different homeless services. A complete description of the services is available at (United States Department of Housing and Urban Development 2020). Sequence of services longer than N are sampled using a sliding window with a length of N. For sequences shorter than N, missing values are encoded with 0, resulting in a zero vector representation with a length $|\mathcal{A}|$ . After sampling, we denote our dataset as D and perform a random 70 – 30 split to obtain the training set $D_{train}$ and testing set $D_{test}$ , respectively. Finally, we train and evaluate our approach using $D_{train}$ and $D_{test}$ respectively.

Table 1: Performance comparison between REPLETE and the baselines. Parameter N is set to be 3. 

<table><tr><td>Method</td><td>Accuracy</td><td>Recall</td><td>Precision</td><td> $F_1$  score</td></tr><tr><td>TRACE</td><td>0.530</td><td>0.188</td><td>0.205</td><td>0.184</td></tr><tr><td>Transformer</td><td>0.632</td><td>0.368</td><td>0.406</td><td>0.376</td></tr><tr><td>LR</td><td>0.691</td><td>0.406</td><td>0.446</td><td>0.420</td></tr><tr><td> $FFNN_1$ </td><td>0.714</td><td>0.385</td><td>0.458</td><td>0.416</td></tr><tr><td>RF</td><td>0.738</td><td>0.391</td><td>0.504</td><td>0.434</td></tr><tr><td>PREVISE</td><td>0.745</td><td>0.400</td><td>0.575</td><td>0.435</td></tr><tr><td>RNN</td><td>0.800</td><td>0.494</td><td>0.583</td><td>0.511</td></tr><tr><td>LSTM</td><td>0.802</td><td>0.496</td><td>0.590</td><td>0.513</td></tr><tr><td>REPLETE</td><td>0.832</td><td>0.539</td><td>0.615</td><td>0.567</td></tr></table>

Baselines We compare REPLETE with the state-of-the-art: TRACE (Rahman and Chelmis 2022) and PREVISE (Rahman, Zois, and Chelmis 2023). TRACE and PREVISE uses service sequences to predict the next assignment. Additionally, following (Rahman, Zois, and Chelmis 2023), we compare our approach with random forest (RF), logistic regression (LR), transformer, feed forward neural network (FFNN $_{1}$ ), recurrent neural network (RNN), and long short-term memory (LSTM), which incorporate one-hot encoded features for prediction.

Evaluation Metrics We evaluate our approach using accuracy, recall, precision, and $F_{1}$ score.

Implementation Details All analyses were conducted in Jupyter Notebook using Python 3 on a MacBook Air with an 8-core M1 chip (4 performance and 4 efficiency cores) and 8 GB of memory, running macOS Sonoma. Representation learning framework ran 500 iterations, and the prediction model, on an average, ran 20 iterations in each fold of 5-fold cross validation.

# 6 Results and Analysis

Quantitative Analysis Table 1 shows that REPLETE significantly outperforms the baselines in predicting the next service assignment, both in terms of accuracy and $F_{1}$ score. Table 2 confirms that REPLETE significantly outperforms PREVISE (state-of-the-art) across all metrics and LSTM (the best-performing baseline) in accuracy and F-score. This indicates that (i) the representation learning framework effectively captures the rich relationships within the features and instances, and (ii) derived features based on temporal, functional, and individual contexts, effectively carry these relationships to the FFNN. Together, these factors lead to a significant improvement in the performance of our approach.

To analyze the reliability of the representation learning framework, we first examine whether derived features can effectively distinguish between labels (i.e., services) compared to the one-hot encoded features. We utilize the t-distributed Stochastic Neighbor Embedding (t-SNE) method to embed both high-dimensional features into 2-dimensional space. Figure 2 illustrates that distinct clusters for each label are formed using the derived features. This indicates that the derived features offer better separability compared to the one-hot encoded features.

Table 2: Statistical significance comparison between RE-PLETE and the baselines. Parameter N is set to be 3. 

<table><tr><td>t-score (p-value)</td><td>Accuracy</td><td>Recall</td><td>Precision</td><td> $F_1$  score</td></tr><tr><td>w.r.t. PREVISE</td><td>-15.3(2.26 × 10-11)</td><td>-3.52(0.0052)</td><td>-5.37(8.8 × 10-5)</td><td>-4.07(0.0018)</td></tr><tr><td>w.r.t. LSTM</td><td>-5.65(2.3 × 10-5)</td><td>-1.99(0.064)</td><td>-1.86(0.082)</td><td>-2.56(0.020)</td></tr></table>

![](images/09386a8f5f8a5e3b0dc33a6c72899387986d573a693dd81cb0563151667c668f.jpg)

<details>
<summary>scatter</summary>

| x    | y    | label |
| ---- | ---- | ----- |
| -100 | 100  | 1     |
| -50  | 75   | 2     |
| 0    | 50   | 3     |
| 50   | 25   | 4     |
| 100  | 0    | 6     |
| -100 | -25  | 11    |
| -50  | -50  | 12    |
| 0    | -75  | 13    |
| 50   | -100 | 14    |
</details>

(a)

![](images/324f0c39ad6dbfce9a5800c143de83da1e7e7b328c0cf3618b3fe8cf1a8a3a77.jpg)

<details>
<summary>scatter</summary>

| x    | y    | label |
| ---- | ---- | ----- |
| -100 | 0    | 1     |
| -50  | 25   | 2     |
| 0    | 50   | 3     |
| 50   | 75   | 4     |
| 100  | 100  | 6     |
| -100 | -25  | 11    |
| -50  | -50  | 12    |
| 0    | -75  | 13    |
| 50   | -100 | 14    |
</details>

(b)   
Figure 2: Scatterplot of 2-dimensional t-SNE embedding for (a) one-hot encoded features and (b) derived features from learned representations.

Next, we analyze the performance of REPLETE across different history length $N$ . Table 3 shows that our model's performance remain consistent despite variation in the length of the individual's history considered for prediction. This indicates the robustness of our approach. For the rest of our experiments, we set $N = 3$ , as the model achieves the highest accuracy and $F_{1}$ score at this value.

Ablation Study We conducted two ablation studies to evaluate the importance of different model components, focusing on the impact of feature and instance interactions. First, we analyze the impact of feature interactions and instance interactions on model performance. Table 4 shows that service and feature representations have the greatest impact, with feature and instance interactions enhancing performance when coupled with these representations.

Next, we analyse the impact of each component of the optimization function on model performance. As shown in Table 5, each component enhances performance, with the best results achieved by combining all three components.

Hyperparameter Selection The joint optimization function has six hyperparameters, with optimal values being $\lambda = 0.01$ , $\alpha = 0.3$ , $\beta = 0.7$ , $\tau = 52$ weeks, $\mu = 1$ , k = 10.

Bias Evaluation Although our proposed approach does not explicitly model against bias and unfairness (both critical characteristics of a technological solution to a complex social challenge, such as homelessness service provision), we evaluate it for potential bias across three sensitive attributes,

Table 3: Performance of our approach across different history length N. 

<table><tr><td>N</td><td>Accuracy</td><td>Recall</td><td>Precision</td><td> $F_1$  score</td></tr><tr><td>2</td><td>0.828</td><td>0.517</td><td>0.644</td><td>0.547</td></tr><tr><td>3</td><td>0.832</td><td>0.539</td><td>0.615</td><td>0.567</td></tr><tr><td>4</td><td>0.809</td><td>0.499</td><td>0.571</td><td>0.517</td></tr><tr><td>5</td><td>0.805</td><td>0.499</td><td>0.566</td><td>0.516</td></tr><tr><td>6</td><td>0.811</td><td>0.548</td><td>0.607</td><td>0.561</td></tr></table>

Table 4: Ablation study for key components of the derived features, namely service and feature representations (rep), instance interactions (inst), and feature interactions (feat). 

<table><tr><td>inst</td><td>rep</td><td>feat</td><td>Accuracy</td><td> $F_1$  score</td></tr><tr><td>√</td><td></td><td></td><td>0.625</td><td>0.208</td></tr><tr><td></td><td>√</td><td></td><td>0.825</td><td>0.526</td></tr><tr><td></td><td></td><td>√</td><td>0.614</td><td>0.144</td></tr><tr><td></td><td>√</td><td>√</td><td>0.829</td><td>0.547</td></tr><tr><td>√</td><td></td><td>√</td><td>0.717</td><td>0.279</td></tr><tr><td>√</td><td>√</td><td></td><td>0.829</td><td>0.543</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.832</td><td>0.567</td></tr></table>

Table 5: Ablation study for key components of the optimization function, namely temporal (temp), functional (func), and individual (ind). 

<table><tr><td>ind</td><td>temp</td><td>func</td><td>Accuracy</td><td> $F_1$  score</td></tr><tr><td>√</td><td></td><td></td><td>0.798</td><td>0.450</td></tr><tr><td></td><td>√</td><td></td><td>0.789</td><td>0.419</td></tr><tr><td></td><td></td><td>√</td><td>0.714</td><td>0.398</td></tr><tr><td></td><td>√</td><td>√</td><td>0.801</td><td>0.476</td></tr><tr><td>√</td><td></td><td>√</td><td>0.821</td><td>0.534</td></tr><tr><td>√</td><td>√</td><td></td><td>0.827</td><td>0.549</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.832</td><td>0.567</td></tr></table>

namely race, gender, and ethnicity. We measure bias using (i) demographic parity ( $\frac{P(\hat{\mathbf{Y}} = a^i | \text{attr} = 1)}{P(\hat{\mathbf{Y}} = a^i | \text{attr} = 0)}$ ) and (ii) equal opportunity ( $\frac{P(\hat{\mathbf{Y}} = a^i | \text{attr} = 1, \mathbf{Y} = a^i)}{P(\hat{\mathbf{Y}} = a^i | \text{attr} = 0, \mathbf{Y} = a^i)}$ ) (Mehrabi et al. 2021), aiming for values within $80\%$ of the group with the highest rate (Pessach and Shmueli 2022). Figure 3 shows that our approach mitigates bias for high-frequency services, such as emergency shelter (denoted by 1) and day shelter (denoted by 11) for majority of the sensitive attributes, but not as much for lower-frequency services, such as transitional housing (denoted by 2) and homelessness prevention (denoted by 12). Moreover, the state-of-the-art method PREVISE exhibits extreme bias for certain services and attributes, where it completely fails to predict some services for specific attributes. This underscores that our approach, REPLETE, demonstrates a significantly lower level of bias compared to PREVISE, ensuring a more equitable service assignment across various attributes and services. In light of this result, we conclude that our approach replicates the existing assignment decision-making process to a significant extent, providing the basis for developing systems that make “unbiased” and “fair” predictions, as well as to understand and evaluate them ethically (i.e., in experimental settings). We therefore plan to incorporate fairness constraints (e.g., (Zafar et al. 2019)) directly into our objective function, as part of our future work.

![](images/b594d613408fdcfd62b246b30d73cd0c7b6ded4923fd709473b775148801cae2.jpg)

<details>
<summary>radar</summary>

| Angle | Series 1 | Series 2 |
|---|---|---|
| 0 | 0.5 | 0.3 |
| 1 | 0.8 | 0.7 |
| 2 | 1.2 | 1.5 |
| 3 | 1.5 | 2.0 |
| 4 | 1.8 | 1.2 |
| 5 | 2.0 | 0.8 |
| 6 | 2.5 | 0.5 |
| 7 | 3.0 | 0.2 |
| 8 | 3.5 | -0.1 |
| 9 | 4.0 | -0.5 |
| 10 | 4.5 | -1.0 |
| 11 | 5.0 | -1.5 |
| 12 | 5.5 | -2.0 |
| 13 | 6.0 | -2.5 |
| 14 | 6.5 | -3.0 |
| 15 | 7.0 | -3.5 |
| 16 | 7.5 | -4.0 |
| 17 | 8.0 | -4.5 |
| 18 | 8.5 | -5.0 |
| 19 | 9.0 | -5.5 |
| 20 | 9.5 | -6.0 |
| 21 | 10.0 | -6.5 |
| 22 | 10.5 | -7.0 |
| 23 | 11.0 | -7.5 |
| 24 | 11.5 | -8.0 |
| 25 | 12.0 | -8.5 |
| 26 | 12.5 | -9.0 |
| 27 | 13.0 | -9.5 |
| 28 | 13.5 | -10.0 |
| 29 | 14.0 | -10.5 |
| 30 | 14.5 | -11.0 |
| 31 | 15.0 | -11.5 |
| 32 | 15.5 | -12.0 |
| 33 | 16.0 | -12.5 |
| 34 | 16.5 | -13.0 |
| 35 | 17.0 | -13.5 |
| 36 | 17.5 | -14.0 |
| 37 | 18.0 | -14.5 |
| 38 | 18.5 | -15.0 |
| 39 | 19.0 | -15.5 |
| 40 | 19.5 | -16.0 |
| 41 | 20.0 | -16.5 |
| 42 | 20.5 | -17.0 |
| 43 | 21.0 | -17.5 |
| 44 | 21.5 | -18.0 |
| 45 | 22.0 | -18.5 |
| 46 | 22.5 | -19.0 |
| 47 | 23.0 | -19.5 |
| 48 | 23.5 | -20.0 |
| 49 | 24.0 | -20.5 |
| 50 | 24.5 | -21.0 |
| Note: The last row is a duplicate of the first row to close the circle in the radar chart, so it is not included in the original data series for comparison.
</details>

—REPLETE Demographic parity —PREVISE Demographic parity
--REPLETE Equal opportunity --PREVISE Equal opportunity

![](images/ae77574f794149d16917beb909bcfd0a48c5c1dda23ae6b306d93e31bb3021fb.jpg)

<details>
<summary>radar</summary>

| Angle | Value |
|-------|-------|
| 1     | 1     |
| 2     | 5     |
| 3     | 3     |
| 4     | 2     |
| 5     | 1     |
| 6     | 1     |
| 7     | 1     |
| 8     | 1     |
| 9     | 1     |
| 10    | 1     |
| 11    | 1     |
| 12    | 1     |
| 13    | 1     |
| 14    | 1     |
</details>

Figure 3: Demographic parity (solid line) and equal opportunity (dotted line) of sensitive attributes (a) gender and (b) ethnicity for each service using REPLETE (blue) and PREVISE (orange). Plots for the remaining attributes are included in the Appendix section. Numerals are used in lieu of actual service names (United States Department of Housing and Urban Development 2020).

# 7 Conclusion

This paper introduced a novel predictive model for homeless service assignment based on representation learning. The proposed approach modeled explicitly both the temporal and functional relationships between services, and the similarity between individuals based on their features and their prior service assignments, to learn latent representations. Utilizing these representations, the proposed approach was shown to outperform the state-of-the-art in the task of next service assignment prediction, a key task in the service assignment decision making process.

Limitations Our analysis is based on a geographically bounded dataset, specifically limited to the Capital Region of New York state. Additionally, the dataset does not record the availability or capacity of services.

Future Directions Our approach serves as a foundational building block for developing more advanced service assignment models. By exploring temporal and functional relationships, we have laid the groundwork for capturing complex interactions in service assignment. However, additional dimensions remain worth investigating, potentially offering valuable insights and new directions for research. Importantly, explicitly addressing bias and unfairness is critical, as our experiments demonstrated, especially if this approach is to be applied in real-world scenarios. Our work takes the first step in this direction by providing a foundation for developing "unbiased" and "fair" predictive models and evaluating them ethically in experimental settings.

REPLETE provides a framework for homelessness service assignment, learning representations that reflect local

services and population demographics. While not region-specific, our model has been constrained by the lack of access to homelessness data from other geographical regions due to confidentiality and privacy concerns. Beyond homelessness, we are actively evaluating REPLETE's applicability to other socially significant domains.

In collaboration with CARES of NY, we are working to test REPLETE in practice. However, empirical evaluation requires addressing ethical considerations, as recommendations could have unintended consequences for individuals. Such a study would also require additional IRB approval. Key technical challenges include mitigating biases in both training data and model outputs and integrating capacity and availability constraints into the predictions. To address these, we plan to incorporate fairness constraints into the optimization function and embed capacity and availability considerations into the prediction model. Additionally, to safeguard privacy and confidentiality, CARES of NY will act as the steward for the model and data.

These efforts position REPLETE as a promising tool for ethical and effective service assignment, with potential for broader applications in other socially impactful domains. By addressing these challenges, we aim to advance the development of fair and unbiased predictive models that can contribute to more equitable outcomes in critical decision-making processes.

# Acknowledgment

This material is based upon work supported by the National Science Foundation under Grant No. ECCS-1737443.

# Appendix

# A Proofs of Update Rules

In this section, we derive the update rules corresponding to equation 2 in Section 4.2 of the paper. Since $L_{2-1}$ norm is not differentiable, directly applying gradient-based methods for optimization is not feasible. To address this, we leverage Lemma 1 in Section 4.2, which provides a closed-form solution for the updates of the matrices P and Q.

Update P For updating P, we fix the other variables and remove the terms that are irrelevant to P in 2 of Section 4.2. We get $O_{P} = \min_{P} \|P\|_{2,1} + \langle K, H - AC^{T} - P \rangle + \frac{\mu}{2} \|H - AC^{T} - P\|_{F}^{2}$ , where $\langle \cdot, \cdot \rangle$ denotes the dot product. Mathematically, the solution to this function is as follows:

$$
\min _ {\mathbf {P}} \| \mathbf {P} \| _ {2, 1} + \left\langle \mathcal {K}, \mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P} \right\rangle + \frac {\mu}{2} \| \mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P} \| _ {F} ^ {2}
$$

Using the dot product $\langle M, M'\rangle = \sum_{ij} M_{ij} M'_{ij}$ for matrices M and $M'$ and the Frobenius norm $\|M\|_{F}^{2} = \sum_{ij} M_{ij}$ , where $\|M\|_{F}^{2}$ denotes the Frobenius norm of matrix M, we derive the following expression.

$$
\begin{array}{l} \min _ {\mathbf {P}} \sum_ {i, j} (\mathcal {K} _ {i j} (\mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P}) _ {i j} + \frac {\mu}{2} (\mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P}) _ {i j} ^ {2}) \\ + \| \mathbf {P} \| _ {2, 1} \\ \end{array}
$$

Next, we combine the first two terms into perfect square by adding and subtracting $\frac{1}{\mu}\mathcal{K}_{ij}$ . After this adjustment, the term $-\frac{1}{\mu}\mathcal{K}_{ij}$ becomes irrelevant to the optimization of P and is therefore eliminated from the equation.

$$
\begin{array}{l} \min _ {\mathbf {P}} \sum_ {i, j} \frac {\mu}{2} ((\mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P}) _ {i j} ^ {2} + \frac {2}{\mu} \mathcal {K} _ {i j} (\mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P}) _ {i j} \\ \left. + \frac {1}{\mu} \mathcal {K} _ {i j} ^ {2} - \frac {1}{\mu} \mathcal {K} _ {i j} ^ {2}\right) + \| \mathbf {P} \| _ {2, 1} \\ = \min _ {\mathbf {P}} \sum_ {i, j} \frac {\mu}{2} ((\mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P}) _ {i j} ^ {2} + \frac {2}{\mu} \mathcal {K} _ {i j} (\mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P}) _ {i j} \\ \left. + \frac {1}{\mu} \mathcal {K} _ {i j} ^ {2}\right) + \| \mathbf {P} \| _ {2, 1} \\ = \min _ {\mathbf {P}} \sum_ {i, j} \frac {\mu}{2} ((\mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P}) _ {i j} + \frac {1}{\mu} \mathcal {K} _ {i j}) ^ {2} + \| \mathbf {P} \| _ {2, 1} \\ \end{array}
$$

Next, the squared term can be expressed as Frobenius norm.

$$
= \min _ {\mathbf {P}} \frac {\mu}{2} \| \mathbf {H} - \mathbf {A C} ^ {T} - \mathbf {P} + \frac {1}{\mu} \mathcal {K} \| _ {F} ^ {2} + \| \mathbf {P} \| _ {2, 1}
$$

$$
= \min _ {\mathbf {P}} \mu [ \frac {1}{2} \| \mathbf {P} - (\mathbf {H} - \mathbf {A C} ^ {T} + \frac {1}{\mu} \mathcal {K}) \| _ {F} ^ {2} + \frac {1}{\mu} \| \mathbf {P} \| _ {2, 1} ]
$$

$$
= \min _ {\mathbf {P}} \frac {1}{2} \| \mathbf {P} - (\mathbf {H} - \mathbf {A C} ^ {T} + \frac {1}{\mu} \mathcal {K}) \| _ {F} ^ {2} + \frac {1}{\mu} \| \mathbf {P} \| _ {2, 1}
$$

Next, applying Lemma 1, the optimal solution $\mathbf{P}^*$ for the above equation is obtained in closed form, where $\mathbf{E}^{\mathbf{P}} = \mathbf{H}-\mathbf{A}\mathbf{C}^{T} + \frac{1}{\mu}\mathcal{K}$ :

$$
\mathbf {P} _ {i,:} ^ {*} = \left\{ \begin{array}{l l} (1 - \frac {1}{\mu \| \mathbf {E} _ {i , :} ^ {\mathbf {P}} \|}) \mathbf {E} _ {i,:} ^ {\mathbf {P}}, & \| \mathbf {E} _ {i,:} ^ {\mathbf {P}} \| > \frac {1}{\mu} \\ 0, & \text { otherwise } \end{array} \right.
$$

Update Q For updating Q, we fix the other variables and remove the terms that are irrelevant to Q in Eq. (2). We get $\min_{\mathbf{Q}} \|Q\|_{2,1} + \langle L, \mathbf{X} - \mathbf{CV}^{T} \mathbf{Q} \rangle + \frac{\mu}{2} \| \mathbf{X} - \mathbf{CV}^{T} - \mathbf{Q} \|_{F}^{2}$ . Similar to P, using Lemma 1, the optimal solution $Q^{*}$ for the above equation is as follows, where $E^{Q} = X - CV^{T} + \frac{1}{\mu} L$ .

$$
\mathbf {Q} _ {i,:} ^ {*} = \left\{ \begin{array}{l l} (1 - \frac {1}{\mu \| \mathbf {E} _ {i , :} ^ {\mathbf {Q}} \|}) \mathbf {E} _ {i,:} ^ {\mathbf {Q}}, & \| \mathbf {E} _ {i,:} ^ {\mathbf {Q}} \| > \frac {1}{\mu} \\ 0, & \text { otherwise } \end{array} \right.
$$

Update S For updating S, we fix the other variables and remove the terms that are irrelevant to S, we get $O_{S} = \min \|D - AS^{T}\|_{F}^{2} + \lambda\|S\|_{F}^{2} + \alpha\|SB'\|_{F}^{2} - tr(\psi_{S}S^{\top})$ , where $\psi_{S}$ is the Lagrangian multiplier for $S \geq 0$ and $tr(\cdot)$ denotes trace operator. The partial derivative of $O_{S}$ is as follows:

$$
\frac {1}{2} \frac {d \mathcal {O} _ {\mathbf {S}}}{d \mathbf {S}} = - (\mathbf {D} - \mathbf {A S} ^ {\top}) ^ {\top} \mathbf {A} + \alpha (\mathbf {S B} ^ {\prime}) \mathbf {B} ^ {\prime} + \lambda \mathbf {S} - \psi_ {\mathbf {S}}
$$

To obtain the optimal solution, we set the partial derivative equal to zero and get

$$
\psi_ {\mathbf {S}} = - \mathbf {D} ^ {\top} \mathbf {A} + \mathbf {S A} ^ {\top} \mathbf {A} + \alpha (\mathbf {S} (\mathbf {B} ^ {\prime}) ^ {2}) + \lambda \mathbf {S}
$$

Next, we use Karush-KuhnTucker complementary condition, that is, $\psi_{\mathbf{S}}(i,j)\mathbf{S}_{ij}=0$ (Boyd et al. 2011), we get,

$$
(\mathbf {\tilde {S}} _ {i j} - \mathbf {\hat {S}} _ {i j}) \mathbf {S} _ {i j} = 0
$$

$$
\tilde {\mathbf {S}} _ {i j} \mathbf {S} _ {i j} = \hat {\mathbf {S}} _ {i j} \mathbf {S} _ {i j}
$$

$$
\mathbf {S} _ {i j} \leftarrow \mathbf {S} _ {i j} \frac {\hat {\mathbf {S}} _ {i j}}{\tilde {\mathbf {S}} _ {i j}}
$$

Here, $\tilde{S}_{ij}$ and $\hat{S}_{ij}$ consists of the positive and negative terms, respectively, as shown below.

$$
\hat {\mathbf {S}} _ {i j} = \mathbf {D} ^ {\top} \mathbf {A} + [ \alpha (\mathbf {S} (\mathbf {B} ^ {\prime}) ^ {2}) ] ^ {-}
$$

$$
\tilde {\mathbf {S}} _ {i j} = \mathbf {S A} ^ {\top} \mathbf {A} + \lambda \mathbf {S} + [ \alpha (\mathbf {S (B ^ {'})} ^ {2}) ] ^ {+}
$$

Here, for any matrix $\mathbf{M}$ , $(\mathbf{M})^{+} = \frac{ABS((M) + \mathbf{M}}{\mathbf{M}}$ and $(\mathbf{M})^{-} = \frac{ABS((M) - \mathbf{M}}{\mathbf{M}}$ are the positive and negative part of $\mathbf{M}$ , respectively. $ABS(\mathbf{M})$ consists of the absolute value of elements in $\mathbf{M}$ (Shu, Wang, and Liu 2019). We follow similar steps to compute $\mathbf{R}_{\mathbf{p}}, \mathbf{R}_{\mathbf{s}}, \mathbf{V}, \mathbf{C}$ , and $\mathbf{A}$ .

Update $R_{p}$ and $R_{s}$ For updating $R_{p}$ , we fix the other variables and remove the terms that are irrelevant to $R_{p}$ , we get $\mathcal{O}_{R_{p}} = \min\{\|T - AR_{p}^{\top}R_{s}A^{\top}\|_{F}^{2}\} + \lambda\|R_{p}\|_{F}^{2} - tr(\psi_{R_{p}}R_{p}^{\top})$ . The partial derivative of $O_{R_{p}}$ is as follows:

$$
\begin{array}{l} \frac {1}{2} \frac {d \mathcal {O} _ {\mathbf {R} _ {\mathrm{p}}}}{d \mathbf {R} _ {\mathrm{p}}} = - \mathbf {T A A} ^ {\top} \mathbf {R} _ {\mathrm{s}} + \mathbf {A R} _ {\mathrm{p}} ^ {\top} \mathbf {R} _ {\mathrm{s}} \mathbf {A} ^ {\top} \mathbf {A A} ^ {\top} \mathbf {R} _ {\mathrm{s}} \\ + \lambda \mathbf {R _ {p}} - \psi_ {\mathbf {R _ {p}}} \\ \end{array}
$$

$$
\psi_ {\mathbf {R _ {p}}} = - \mathbf {T A A} ^ {\top} \mathbf {R _ {s}} + \mathbf {A R _ {p}} ^ {\top} \mathbf {R _ {s}} \mathbf {A} ^ {\top} \mathbf {A A} ^ {\top} \mathbf {R _ {s}} + \lambda \mathbf {R _ {p}}
$$

Using Karush-KuhnTucker complementary condition (Boyd et al. 2011), that is, $\psi_{\mathbf{R}_{\mathbf{p}}}(i,j)\mathbf{R}_{\mathbf{p}_{ij}} = 0$ , we get,

$$
\mathbf {R _ {p}} _ {i j} \leftarrow \mathbf {R _ {p}} _ {i j} \frac {\hat {\mathbf {R}} _ {\mathbf {p} _ {i j}}}{\tilde {\mathbf {R}} _ {\mathbf {p} _ {i j}}}
$$

$$
\hat {\mathbf {R}} _ {\mathbf {p} _ {i j}} = \mathbf {T A A} ^ {\top} \mathbf {R} _ {\mathbf {s}} + (\mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A A} ^ {\top} \mathbf {R} _ {\mathbf {s}}) ^ {-}
$$

$$
\tilde {\mathbf {R}} _ {\mathbf {p} _ {i j}} = \lambda \mathbf {R} _ {\mathbf {p}} + (\mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A} \mathbf {A} ^ {\top} \mathbf {R} _ {\mathbf {s}}) ^ {+}
$$

Similar to $\mathbf{R}_{\mathbf{p}}$ , we get the following equations for $\mathbf{R}_{\mathbf{s}}$ .

$$
\mathbf {R _ {s}} _ {i j} \leftarrow \mathbf {R _ {s}} _ {i j} \frac {\hat {\mathbf {R}} _ {\mathbf {s} _ {i j}}}{\tilde {\mathbf {R}} _ {\mathbf {s} _ {i j}}}
$$

$$
\hat {\mathbf {R}} _ {\mathbf {s} _ {i j}} = \mathbf {T A A} ^ {\top} \mathbf {R} _ {\mathbf {p}} + (\mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A A} ^ {\top} \mathbf {R} _ {\mathbf {p}}) ^ {-}
$$

$$
\tilde {\mathbf {R}} _ {\mathbf {s} _ {i j}} = \lambda \mathbf {R} _ {\mathbf {s}} + (\mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A} \mathbf {A} ^ {\top} \mathbf {R} _ {\mathbf {p}}) ^ {+}
$$

Update V For updating V, we fix the other variables and remove the terms that are irrelevant to V, we get $O_{V} = \min \frac{\mu}{2} \|X - CV^{\top} - Q\|_{F}^{2} + \langle L, X - CV^{\top} - Q \rangle + \lambda \|V\|_{F}^{2} - tr(\psi_{V} V^{\top})$ . The partial derivative of $O_{V}$ is as follows:

$$
\frac {d \mathcal {O} _ {\mathbf {V}}}{d \mathbf {V}} = - \mu (\mathbf {X} - \mathbf {C V} ^ {\top} - \mathbf {Q}) ^ {\top} \mathbf {C} - \mathcal {L} ^ {\top} \mathbf {C} + 2 \lambda \mathbf {V} - \psi_ {\mathbf {V}}
$$

$$
\psi_ {\mathbf {V}} = - \mu \mathbf {X} ^ {\top} \mathbf {C} + \mu \mathbf {V C} ^ {\top} \mathbf {C} + \mu \mathbf {Q} ^ {\top} \mathbf {C} - \mathcal {L} ^ {\top} \mathbf {C} + 2 \lambda \mathbf {V}
$$

Using Karush-KuhnTucker complementary condition (Boyd et al. 2011), that is, $\psi_{\mathbf{V}}(i,j)\mathbf{V}_{ij} = 0$ , we get,

$$
\mathbf {V} _ {i j} \leftarrow \mathbf {V} _ {i j} \frac {\hat {\mathbf {V}} _ {i j}}{\tilde {\mathbf {V}} _ {i j}}
$$

$$
\hat {\mathbf {V}} _ {i j} = \mu \mathbf {X} ^ {\top} \mathbf {C} + (\mathcal {L} ^ {\top} \mathbf {C}) ^ {+}
$$

$$
\tilde {\mathbf {V}} _ {i j} = \mu \mathbf {V} \mathbf {C} ^ {\top} \mathbf {C} + \mu \mathbf {Q} ^ {\top} \mathbf {C} + (\mathcal {L} ^ {\top} \mathbf {C}) ^ {-} + 2 \lambda \mathbf {V}
$$

Update C For updating C, we fix the other variables and remove the terms that are irrelevant to C, we get $O_{C} = \min \frac{\mu}{2} \|H - AC^{\top} - P\|_{F}^{2} + \frac{\mu}{2} \|X - CV^{\top} - Q\|_{F}^{2} + \langle L, X - CV^{\top} - Q \rangle + \langle K, H - AC^{\top} - P \rangle + \langle N, C^{\top}C - I \rangle + \beta tr(C^{\top}ΓC) + \lambda \|C\|_{F}^{2} - tr(\psi_{C}C^{\top})$ . The partial derivative of $O_{C}$ is as follows:

$$
\begin{array}{l} \frac {1}{2} \frac {d \mathcal {O} _ {\mathbf {C}}}{d \mathbf {C}} = - \mu (\mathbf {H} - \mathbf {A C} ^ {\top} - \mathbf {P}) ^ {\top} \mathbf {A} - \mu (\mathbf {X} - \mathbf {C V} ^ {\top} - \mathbf {Q}) \mathbf {V} \\ - \mathcal {L} \mathbf {V} - \mathcal {K} ^ {\top} \mathbf {A} + 2 \mathbf {C N} + 2 \beta \mathbf {\Gamma C} + 2 \lambda \mathbf {C} - \psi_ {\mathbf {C}} \\ \psi_ {\mathbf {C}} = - \mu \mathbf {H} ^ {\top} \mathbf {A} + \mu \mathbf {C A} ^ {\top} \mathbf {A} + \mu \mathbf {P} ^ {\top} \mathbf {A} - \mu \mathbf {X V} + \mu \mathbf {C V} ^ {\top} \mathbf {V} \\ + \mu \mathbf {Q} \mathbf {V} - \mathcal {L} \mathbf {V} - \mathcal {K} ^ {\top} \mathbf {A} + 2 \mathbf {C N} + 2 \beta \mathbf {\Gamma C} + 2 \lambda \mathbf {C} \\ \end{array}
$$

Using Karush-KuhnTucker complementary condition (Boyd et al. 2011), that is, $\psi_{\mathbf{C}}(i,j)\mathbf{C}_{ij} = 0$ , we get,

$$
\mathbf {C} _ {i j} \leftarrow \mathbf {C} _ {i j} \frac {\hat {\mathbf {C}} _ {i j}}{\tilde {\mathbf {C}} _ {i j}}
$$

$$
\hat {\mathbf {C}} _ {i j} = \mu \mathbf {H} ^ {\top} \mathbf {A} + \mu \mathbf {X V} + (\mathcal {L V}) ^ {+} + (\mathcal {K} ^ {\top} \mathbf {A}) ^ {+} + 2 (\mathbf {C N}) ^ {-}
$$

$$
+ 2 (\beta (\mathbf {\Gamma C}) ^ {-}
$$

$$
\tilde {\mathbf {C}} _ {i j} = \mu \mathbf {C A} ^ {\top} \mathbf {A} + \mu \mathbf {P} ^ {\top} \mathbf {A} + \mu \mathbf {C V} ^ {\top} \mathbf {V} + \mu \mathbf {Q V} + (\mathcal {L V}) ^ {-}
$$

$$
+ \left(\mathcal {K} ^ {\top} \mathbf {A}\right) ^ {-} + 2 (\mathbf {C N}) ^ {+} + 2 \beta (\boldsymbol {\Gamma C}) ^ {+} + 2 \lambda \mathbf {C}
$$

Update A For updating A, we fix the other variables and remove the terms that are irrelevant to A, we get $O_{A} = \|D - AS^{\top}\|_{F}^{2} + \|T - AR_{p}^{\top}R_{s}A^{\top}\|_{F}^{2} + \lambda\|A\|_{F}^{2} + \frac{\mu}{2}\|H - AC^{\top} - P\|_{F}^{2} + \langle K, H - AC^{\top} - P\rangle - tr(\psi_{A}A^{\top}) \min$ . The partial derivative of $O_{A}$ is as follows:

$$
\frac {1}{2} \frac {d \mathcal {O} _ {\mathbf {A}}}{d \mathbf {A}} = - 2 (\mathbf {D} - \mathbf {A S} ^ {\top}) \mathbf {S} - 4 (\mathbf {T} - \mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top}) \mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}}
$$

$$
+ 2 \lambda \mathbf {A} - \mu (\mathbf {H} - \mathbf {A C} ^ {\top} - \mathbf {P}) \mathbf {C} - \mathbf {K C} - \psi_ {\mathbf {A}}
$$

$$
\psi_ {\mathbf {A}} = - 2 \mathbf {D} \mathbf {S} + 2 \mathbf {A} \mathbf {S} ^ {\top} \mathbf {S} - 4 \mathbf {T} \mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}}
$$

$$
+ 4 \mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A} \mathbf {R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} + 2 \lambda \mathbf {A} - \mu \mathbf {H C} + \mu \mathbf {A C} ^ {\top} \mathbf {C}
$$

$$
+ \mu \mathbf {P C} - \mathbf {K C}
$$

Using Karush-KuhnTucker complementary condition (Boyd et al. 2011), that is, $\psi_{\mathbf{A}}(i,j)\mathbf{A}_{ij}=0$ , we get,

$$
\mathbf {A} _ {i j} \leftarrow \mathbf {A} _ {i j} \frac {\hat {\mathbf {A}} _ {i j}}{\tilde {\mathbf {A}} _ {i j}}
$$

$$
\hat {\mathbf {A}} _ {i j} = 2 \mathbf {D} \mathbf {S} + 4 \mathbf {T A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} + 4 (\mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}} \mathbf {A} ^ {\top} \mathbf {A R} _ {\mathbf {p}} ^ {\top} \mathbf {R} _ {\mathbf {s}}) ^ {-}
$$

$$
+ \mu \mathbf {H C} + (\mathbf {K C}) ^ {+}
$$

$$
\tilde {\mathbf {A}} _ {i j} = 2 \mathbf {A} \mathbf {S} ^ {\top} \mathbf {S} + 4 (\mathbf {A} \mathbf {R _ {p}} ^ {\top} \mathbf {R _ {s}} \mathbf {A} ^ {\top} \mathbf {A} \mathbf {R _ {p}} ^ {\top} \mathbf {R _ {s}}) ^ {+} + 2 \lambda \mathbf {A}
$$

$$
+ \mu \mathbf {A} \mathbf {C} ^ {\top} \mathbf {C} + \mu \mathbf {P} \mathbf {C} + (\mathbf {K} \mathbf {C}) ^ {-}
$$

B Further Experimental Results on Bias   
![](images/c42e98b2f4184e212363512b60b53766783b484bf393df08da00829a47a863c0.jpg)

<details>
<summary>radar</summary>

| Angle | Series 1 | Series 2 |
|---|---|---|
| 0 | 1 | 0 |
| 35° | 2 | 3 |
| 60° | 1 | 1 |
| 95° | 0 | 0 |
| 130° | 1 | 2 |
| 165° | 2 | 1 |
| 200° | 1 | 0 |
| 235° | 0 | 0 |
| 270° | 1 | 1 |
| 315° | 2 | 2 |
| 360° | 1 | 3 |
| 405° | 0 | 1 |
| 450° | 1 | 0 |
| 505° | 2 | 1 |
| 550° | 1 | 2 |
| 605° | 0 | 3 |
| 650° | 1 | 4 |
| 705° | 2 | 5 |
| 750° | 1 | 6 |
| 805° | 0 | 7 |
| 850° | 1 | 8 |
| 905° | 2 | 9 |
| 950° | 1 | 10 |
| 1005° | 0 | 11 |
| 1050° | 1 | 12 |
| 1105° | 2 | 13 |
| 1150° | 1 | 14 |
| 1205° | 0 | 15 |
| 1250° | 1 | 16 |
| 1305° | 2 | 17 |
| 1350° | 1 | 18 |
| 1405° | 0 | 19 |
| 1450° | 1 | 20 |
| 1505° | 2 | 21 |
| 1550° | 1 | 22 |
| 1605° | 0 | 23 |
| 1650° | 1 | 24 |
| 1705° | 2 | 25 |
| 1750° | 1 | 26 |
| 1805° | 0 | 27 |
| 1850° | 1 | 28 |
| 1905° | 2 | 29 |
| 1950° | 1 | 30 |
| 2005° | 0 | 31 |
| 2050° | 1 | 32 |
| 2105° | 2 | 33 |
| 2150° | 1 | 34 |
| 2205° | 0 | 35 |
| 2250° | 1 | 36 |
| 2305° | 2 | 37 |
| 2350° | 1 | 38 |
| 2405° | 0 | 39 |
| 2450° | 1 | 40 |
| Note: The last row is a duplicate of the first row to close the circle in the radar chart. The rest of the chart is empty. The values for 'Series' are estimated based on the chart's visual context. There is no additional data series or labels provided in the image.
</details>

(a)

![](images/d7c75edfd21a6b117aa55123fe88b98383cfba8263be9778ed971b54ec6c8f4f.jpg)  
(b)

![](images/3e390b84e6e1d6e05dd4c228f372fa13d269f507fe8c10d4351a2453f43da20d.jpg)

<details>
<summary>radar</summary>

| Angle | Value |
|-------|-------|
| 0     | 5     |
</details>

(c)

![](images/0690c28a714958e29b5366f8c89ed7534266e17e5395bce1de7a83349c447dd8.jpg)

<details>
<summary>radar</summary>

| Angle | Series 1 | Series 2 | Series 3 |
|---|---|---|---|
| 0 | 0.5 | 0.8 | 0.6 |
| 1 | 1.0 | 1.2 | 0.9 |
| 2 | 1.5 | 1.8 | 1.3 |
| 3 | 1.0 | 1.5 | 1.7 |
| 4 | 0.5 | 0.8 | 0.6 |
| 5 | 0.8 | 1.0 | 0.9 |
| 6 | 1.2 | 1.5 | 1.2 |
| 7 | 1.5 | 1.8 | 1.5 |
| 8 | 1.0 | 1.5 | 1.2 |
| 9 | 0.8 | 1.2 | 0.9 |
| 10 | 0.5 | 0.8 | 0.6 |
| 11 | 0.8 | 1.0 | 0.9 |
| 12 | 1.2 | 1.5 | 1.2 |
| 13 | 1.5 | 1.8 | 1.5 |
| 14 | 1.0 | 1.5 | 1.2 |
| 15 | 0.5 | 0.8 | 0.6 |
| 16 | 0.8 | 1.0 | 0.9 |
| 17 | 1.2 | 1.5 | 1.2 |
| 18 | 1.5 | 1.8 | 1.5 |
| 19 | 1.0 | 1.5 | 1.2 |
| 20 | 0.5 | 0.8 | 0.6 |
| 21 | 0.8 | 1.0 | 0.9 |
| 22 | 1.2 | 1.5 | 1.2 |
| 23 | 1.5 | 1.8 | 1.5 |
| 24 | 1.0 | 1.5 | 1.2 |
| 25 | 0.5 | 0.8 | 0.6 |
| 26 | 0.8 | 1.0 | 0.9 |
| 27 | 1.2 | 1.5 | 1.2 |
| 28 | 1.5 | 1.8 | 1.5 |
| 29 | 1.0 | 1.5 | 1.2 |
| 30 | 0.5 | 0.8 | 0.6 |
| Note: The values in the 'Series' table are estimated based on the provided code snippet from the original image 'The Data Table'. The numbers 'A' appear to be the same for all three series in this case.
</details>

(d)

![](images/62d2c0344ae5ab8a5452b1b655f80ef03b441209a512388a6c65f77ad9736a89.jpg)

<details>
<summary>radar</summary>

| Angle | Value |
|-------|-------|
| 1     | 1.0   |
| 2     | 0.5   |
| 3     | 0.8   |
| 4     | 0.6   |
| 5     | 0.7   |
| 6     | 0.9   |
| 7     | 1.2   |
| 8     | 2.5   |
</details>

(e)

![](images/bd93e664a992efe99684bb1f891b861b20bf6c7ed39add9f8493fd95438c0436.jpg)

<details>
<summary>radar</summary>

| Angle | Series 1 | Series 2 | Series 3 |
|-------|----------|----------|----------|
| 0°    | 5        | 0        | 5        |
| 30°   | 5        | 0        | 5        |
| 60°   | 5        | 0        | 5        |
| 90°   | 5        | 0        | 5        |
| 120°  | 5        | 0        | 5        |
| 150°  | 5        | 0        | 5        |
| 180°  | 5        | 0        | 5        |
| 210°  | 5        | 0        | 5        |
| 240°  | 5        | 0        | 5        |
| 270°  | 5        | 0        | 5        |
| 300°  | 5        | 0        | 5        |
| 330°  | 5        | 0        | 5        |
| 360°  | 5        | 0        | 5        |
| 390°  | 5        | 0        | 5        |
| 420°  | 5        | 0        | 5        |
| 450°  | 5        | 0        | 5        |
| 480°  | 5        | 0        | 5        |
| 510°  | 5        | 0        | 5        |
| 540°  | 5        | 0        | 5        |
| 570°  | 5        | 0        | 5        |
| 600°  | 5        | 0        | 5        |
| 630°  | 5        | 0        | 5        |
| 660°  | 5        | 0        | 5        |
| 690°  | 5        | 0        | 5        |
| 720°  | 5        | 0        | 5        |
| 750°  | 5        | 0        | 5        |
| 780°  | 5        | 0        | 5        |
| 810°  | 5        | 0        | 5        |
| 840°  | 5        | 0        | 5        |
| 870°  | 5        | 0        | 5        |
| 900°  | 5        | 0        | 5        |
| Note: The last row is a duplicate of the first row to close the circle in the radar chart. The rest of the other rows are empty. The values for the last row are estimated based on the label 'a'.
</details>

(f)

![](images/6eb9e4ab92d32c6c7afc79888fa620fcaf6b20a2888ffdcfa648ebf098a49b93.jpg)

<details>
<summary>radar</summary>

| Angle | Series 1 | Series 2 | Series 3 |
|-------|----------|----------|----------|
| 0     | 1        | 1        | 1        |
| 1     | 2        | 2        | 2        |
| 2     | 3        | 3        | 3        |
| 3     | 4        | 4        | 4        |
| 4     | 3        | 3        | 3        |
| 5     | 2        | 2        | 2        |
| 6     | 1        | 1        | 1        |
| 7     | 0        | 0        | 0        |
| 8     | 1        | 1        | 1        |
| 9     | 2        | 2        | 2        |
| 10    | 3        | 3        | 3        |
| 11    | 4        | 4        | 4        |
| 12    | 3        | 3        | 3        |
| 13    | 2        | 2        | 2        |
| 14    | 1        | 1        | 1        |
</details>

(g)

![](images/d4ab64ce65d9f429f02ef05512b7c30f39e4b4f0c11df462a067745f7c2ed60a.jpg)  
Figure 4: Demographic parity (solid line) and equal opportunity (dotted line) of sensitive attributes (a) gender, (b) ethnicity, (c) American Indian or Alaskan Native, (d) Black African American, (e) Asian, (f) Native Hawaiian or Other Pacific Islander, and (g) White for each service using REPLETE (blue) and PREVISE (orange). Numerals are used in lieu of actual service names (United States Department of Housing and Urban Development 2020). Values within the range of 0.8 to 1.2 are consider better, with the optimal value being 1 (Pessach and Shmueli 2022).

In Section 6, Figure 3 compares our proposed approach (REPLETE) with the state-of-the-art (PREVISE) w.r.t demographic parity and equal opportunity for two sensitive attributes, namely gender and ethnicity. Here, we plot additional results for more sensitive attributes in our dataset. We observe that PREVISE fails to predict certain services for specific attributes altogether, resulting in either zero or disproportionately high values (closer to 15 or 32) for demographic parity and equal opportunity. A value of zero indicates that no individuals with that specific attribute were assigned to the service, suggesting bias against the minority group. Conversely, higher values suggest that the model is biased against the majority group. In contrast, REPLETE exhibits a more balanced behavior, avoiding these extremes and making relatively unbiased assignments.

# C Further Performance Comparison

To evaluate the significance of the features derived from the representation learning framework of REPLETE (Section 4.1), we compare our model against an additional baseline, where sequences of one-hot encoded features are input into the feed-forward neural network (FFNN $_{1}$ ) instead of the derived features. Table 6 shows that naively training a FFNN on the one-hot encoded features results in a model with a predictive power that is better than TRACE and LR but not as good as RF, and significantly worse than both PREVISE and REPLETE. In fact REPLETE (which uses FFNN with the learned representations) achieves the best predictive performance, demonstrating that the derived features are indeed crucial in accurately predicting the next service assignment.

Table 6: Performance comparison between REPLETE and FFNN $_{1}$ . Parameter N is set to be 3. 

<table><tr><td>Method</td><td>Accuracy</td><td>Recall</td><td>Precision</td><td> $F_1$  score</td></tr><tr><td> $FFNN_1$ </td><td>0.714</td><td>0.385</td><td>0.458</td><td>0.416</td></tr><tr><td>REPLETE</td><td>0.832</td><td>0.539</td><td>0.615</td><td>0.567</td></tr></table>

# D Clarification on Features used to Train the Baselines

Figure 5 provides a visual representation of the inputs used by the baselines. PREVISE and TRACE directly use the sequence of services $T_{u_{i}}$ for each individual $u_{i}$ and leverage the transition between services (Rahman and Chelmis 2022) within a network (Bayesian network (Rahman, Zois, and Chelmis 2023)) for service prediction. On the other hand, RF, LR, and FFNN $_{1}$ all use a concatenated vector of one-hot encoded features (e.g., gender, living situation, disability condition etc.) as their input.

![](images/987bec0dde1ecdab2787471853105a5d8dcb68962f79ec18f66332b1853f3b71.jpg)

<details>
<summary>line</summary>

| x    | accuracy | fscore |
| ---- | -------- | ------ |
| 0.1  | 0.82     | 0.55   |
| 0.3  | 0.82     | 0.56   |
| 0.5  | 0.82     | 0.57   |
| 0.7  | 0.82     | 0.57   |
| 0.9  | 0.80     | 0.46   |
</details>

![](images/26f138b031e1d4a8d0a99c7e43a7aa934dc3562d6108c3a79179a36d22037b6c.jpg)

<details>
<summary>line</summary>

| x    | accuracy | f1score |
| ---- | -------- | ------- |
| 0.1  | 0.84     | 0.57    |
| 0.2  | 0.83     | 0.55    |
| 0.3  | 0.82     | 0.53    |
| 0.4  | 0.81     | 0.51    |
| 0.5  | 0.80     | 0.50    |
| 0.6  | 0.81     | 0.51    |
| 0.7  | 0.82     | 0.52    |
| 0.8  | 0.81     | 0.52    |
| 0.9  | 0.80     | 0.52    |
</details>

![](images/8db558d26d97e243f77cd7e70922ee10263c220166c3d4f694f84f77acc900d3.jpg)

<details>
<summary>line</summary>

| x      | accuracy | fiscore |
| ------ | -------- | ------- |
| 10^-2  | 0.6      | 0.1     |
| 10^-1  | 0.8      | 0.55    |
| 10^0   | 0.8      | 0.57    |
| 10^1   | 0.8      | 0.45    |
</details>

![](images/88823a330de5e6362ef6e98fa6e33afd1362dec90c228d4d3a039165b954988c.jpg)

<details>
<summary>line</summary>

| d   | accuracy | f1score |
| --- | -------- | ------- |
| 20  | 0.81     | 0.54    |
| 30  | 0.80     | 0.48    |
| 40  | 0.82     | 0.54    |
| 50  | 0.82     | 0.55    |
| 60  | 0.82     | 0.55    |
</details>

![](images/0c2b83839139ff67aec7627e266d550a55495702efcad1805a074303a5f77245.jpg)

<details>
<summary>line</summary>

| x       | accuracy | flscore |
| ------- | -------- | ------- |
| 10^-4   | 0.80     | 0.48    |
| 10^-3   | 0.82     | 0.52    |
| 10^-2   | 0.83     | 0.55    |
| 10^-1   | 0.82     | 0.50    |
| 10^0    | 0.81     | 0.48    |
| 10^1    | 0.81     | 0.46    |
| 10^2    | 0.80     | 0.44    |
</details>

![](images/e9892bee2ba766e4017e5a027a6c1e1c5b9039dfdd1ceb46bb4f29d9373e23b8.jpg)

<details>
<summary>line</summary>

| f   | accuracy | flscore |
| --- | -------- | ------- |
| 6   | 0.80     | 0.45    |
| 10  | 0.81     | 0.50    |
| 16  | 0.82     | 0.52    |
| 20  | 0.83     | 0.55    |
</details>

Figure 6: Accuracy and $F_{1}$ score plots for varying (a) $\alpha$ , (b) $\beta$ , (c) $\mu$ , (d) $\tau$ , (e) $\lambda$ , and (f) $k$ .

![](images/7a79d924e9db80a19da30bfc482c16cc352989417caf8bf397c46f0a0d12a737.jpg)  
Figure 5: Input for (a) TRACE and PREVISE, and (b) RF, LR, FFNN $_{1}$ . As defined in Section 3, U, F, and $p_{t_{i}}$ is the set of chronically homeless individuals, set of features, and service assigned at time $t_{i}$ , respectively. Additionally, $F_{i}$ denotes the one–hot encoded vector of feature $F_{i}$ .

# E Hyperparameter Tuning

We assessed their impact on model performance, finding that our approach consistently performs well across different hyperparameter values illustrated in Figure 6.

# References

Boyd, S.; Parikh, N.; Chu, E.; Peleato, B.; Eckstein, J.; et al. 2011. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends® in Machine learning, 3(1): 1–122.   
Boyd, S.; and Vandenberghe, L. 2004. Convex optimization. Cambridge university press.

Chelmis, C.; Qi, W.; Lee, W.; and Duncan, S. 2021. Smart Homelessness Service Provision with Machine Learning. Procedia Computer Science, 185: 9–18.   
Dej, E.; Gaetz, S.; and Schwan, K. 2020. Turning off the tap: a typology for homelessness prevention. The Journal of Primary Prevention, 41(5): 397–412.   
Fleury, M.-J.; Grenier, G.; Sabetti, J.; Bertrand, K.; Clément, M.; and Brochu, S. 2021. Met and unmet needs of homeless individuals at different stages of housing reintegration: A mixed-method investigation. PloS one, 16(1): e0245088.   
Gao, Y.; Das, S.; and Fowler, P. 2017. Homelessness service provision: a data science perspective. In Workshops at the Thirty-First AAAI Conference on Artificial Intelligence.   
Greer, A. L.; Shinn, M.; Kwon, J.; and Zuiderveen, S. 2016. Targeting services to individuals most likely to enter shelter: Evaluating the efficiency of homelessness prevention. Social Service Review, 90(1): 130–155.   
Henry, M.; de Sousa, T.; Roddey, C.; Gayen, S.; Joe Bednar, T.; and Associates, A. 2023. The 2023 Annual Homeless Assessment Report (AHAR) to Congress. Part 1: Point-in-time estimates of homelessness. The US Department of Housing and Urban Development.   
Hong, B.; Malik, A.; Lundquist, J.; Bellach, I.; and Kontokosta, C. E. 2018. Applications of machine learning methods to predict readmission and length-of-stay for homeless families: The case of win shelters in new york city. Journal of Technology in Human Services, 36(1): 89–104.   
Kuang, D.; Ding, C.; and Park, H. 2012. Symmetric nonnegative matrix factorization for graph clustering. In Proceedings of the 2012 SIAM international conference on data mining, 106–117. SIAM.   
Kube, A.; Das, S.; and Fowler, P. J. 2019. Allocating interventions based on predicted outcomes: A case study on homelessness services. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, 622–629.   
Mehrabi, N.; Morstatter, F.; Saxena, N.; Lerman, K.; and Galstyan, A. 2021. A survey on bias and fairness in machine learning. ACM computing surveys (CSUR), 54(6): 1–35.   
Messier, G.; John, C.; and Malik, A. 2021. Predicting Chronic Homelessness: The Importance of Comparing Algorithms using Client Histories. Journal of Technology in Human Services, 1–12.   
Pessach, D.; and Shmueli, E. 2022. A review on fairness in machine learning. ACM Computing Surveys (CSUR), 55(3):1–44.   
Pokharel, G.; Das, S.; and Fowler, P. 2024. Discretionary trees: understanding street-level bureaucracy via machine learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 22303–22312.   
Rahman, K. S.; and Chelmis, C. 2022. Learning to Predict Transitions within the Homelessness System from Network Trajectories. In Proceedings of the 2022 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining, 181–189.   
Rahman, K. S.; Zois, D.-S.; and Chelmis, C. 2023. Bayesian Network Modeling and Prediction of Transitions Within the

Homelessness System. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5. IEEE.   
Rodríguez, P.; Bautista, M. A.; Gonzalez, J.; and Escalera, S. 2018. Beyond one-hot encoding: Lower dimensional target embedding. Image and Vision Computing, 75: 21–31.   
Sarker, I. H. 2021. Machine learning: Algorithms, real-world applications and research directions. SN computer science, 2(3): 160.   
Shinn, M.; Greer, A. L.; Bainbridge, J.; Kwon, J.; and Zuiderveen, S. 2013. Efficient targeting of homelessness prevention services for families. American journal of public health, 103(S2): S324–S330.   
Shu, K.; Wang, S.; and Liu, H. 2019. Beyond news contents: The role of social context for fake news detection. In Proceedings of the twelfth ACM international conference on web search and data mining, 312–320.   
Tang, J.; and Liu, H. 2012. Unsupervised feature selection for linked social media data. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining, 904–912.   
Toros, H.; and Flaming, D. 2018. Prioritizing homeless assistance using predictive algorithms: an evidence-based approach. Cityscape, 20(1): 117–146.   
United States Department of Housing and Urban Development. 2020. HMIS Data Standards Manual. Retrieved May 26, 2021, from https://www.hudexchange.info/resource/3824/hmis-data-dictionary/.   
Vajiac, C.; Frey, A.; Baumann, J.; Smith, A.; Amarasinghe, K.; Lai, A.; Rodolfa, K. T.; and Ghani, R. 2024. Preventing Eviction-Caused Homelessness through ML-Informed Distribution of Rental Assistance. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 22393–22400.   
VanBerlo, B.; Ross, M. A.; Rivard, J.; and Booker, R. 2021. Interpretable machine learning approaches to prediction of chronic homelessness. Engineering Applications of Artificial Intelligence, 102: 104243.   
Wang, S.; Tang, J.; and Liu, H. 2015. Embedded unsupervised feature selection. In Proceedings of the AAAI conference on artificial intelligence, volume 29.   
Zafar, M. B.; Valera, I.; Gomez-Rodriguez, M.; and Gummadi, K. P. 2019. Fairness constraints: A flexible approach for fair classification. Journal of Machine Learning Research, 20(75): 1–42.   
Zhang, L.; and Zhang, S. 2017. A unified joint matrix factorization framework for data integration. arXiv preprint arXiv:1707.08183.