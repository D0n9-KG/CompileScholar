# Learning Superconductivity from Ordered and Disordered Material Structures

Pin Chen $^{1}$ Luoxuan Peng $^{1}$ Rui Jiao $^{2,3}$ Qing Mo $^{1}$ Zhen Wang $^{1}$ Wenbing Huang $^{4,5}$ Yang Liu $^{2,3}$ Yutong Lu $^{1*}$

$^{1}$ National Supercomputer Center in Guangzhou,

School of Computer Science and Engineering, Sun Yat-sen University

$^{2}$ Dept. of Comp. Sci. & Tech., Institute for AI, BNRist Center, Tsinghua University

$^{3}$ Institute for AIR, Tsinghua University

$^{4}$ Gaoling School of Artificial Intelligence, Renmin University of China

$^{5}$ Beijing Key Laboratory of Big Data Management and Analysis Methods, Beijing, China

# Abstract

Superconductivity is a fascinating phenomenon observed in certain materials under certain conditions. However, some critical aspects of it, such as the relationship between superconductivity and materials' chemical/structural features, still need to be understood. Recent successes of data-driven approaches in material science strongly inspire researchers to study this relationship with them, but a corresponding dataset is still lacking. Hence, we present a new dataset for data-driven approaches, namely SuperCon3D, containing both 3D crystal structures and experimental superconducting transition temperature ( $T_{c}$ ) for the first time. Based on SuperCon3D, we propose two deep learning methods for designing high $T_{c}$ superconductors. The first is SODNet, a novel equivariant graph attention model for screening known structures, which differs from existing models in incorporating both ordered and disordered geometric content. The second is a diffusion generative model DiffCSP-SC for creating new structures, which enables high $T_{c}$ -targeted generation. Extensive experiments demonstrate that both our proposed dataset and models are advantageous for designing new high $T_{c}$ superconducting candidates.

# 1 Introduction

The pursuit of high-temperature superconductors is driven by their promising applications in efficient energy transmission, advanced electromagnetics, and quantum computing $[6, 36]$ , yet their design is hindered by the enigmatic nature of high-Tc unconventional superconductivity. Although BCS theory $[21]$ aids in predicting Tc for conventional superconductors through first-principles calculations, these methods are computationally demanding and limited to specific materials, necessitating extensive calculations for electron-phonon coupling. Moreover, the intrinsic disorder in many superconductors poses additional challenges for atomic-level design $[39]$ . Such complexities highlight the need for novel approaches in superconductor research and development.

Benefiting from massive public datasets in materials science, data-driven deep learning has been instrumental in predicting material properties [45], synthesizing structures [13], and more. These methods bypass complex physical theories and are crucial in superconductor research, aiding in $\mathrm{T}_c$ prediction models for database analysis [12] and inverse design models for novel structures [60], underscoring deep learning's impact on accelerating superconducting material discovery and design. Specially, Graph Neural Networks (GNN) have been extensively applied to model ordered crystals

[61, 16, 12, 64], fewer methods exist for representing disordered crystals [9], despite their prevalence in nature and databases like ICSD, where over $50\%$ of structures are disordered. Therefore, developing methods to represent disordered structures in graphs is vital, especially for superconductivity research where $\mathrm{T}_c$ enhancement often involves doping or applying pressure.

Recently, generative model is widely used in Natural Language Processing (NLP), Computer Vision (CV) and natural science. Inspired by non-equilibrium thermodynamics, Diffusion Models (DM) currently produce State-of-the-Art proteins $[55]$ , molecules $[25]$ as well as crystals $[62, 27]$ . However, in the field of crystal structure generation, existing models such as CDVAE utilizes the score matching method for atom coordinates, which does not ensure the translation invariance. DiffCSP focuses on crystal structure prediction tasks, which cannot be applied to design novel periodic materials from scratch.

Given the incomplete understanding of superconducting mechanisms, a data-driven approach shows great promise. Constructing a dataset that captures the structure-to-superconductivity relationship is essential for training AI models aimed at designing superconductors. Hence, we introduce SuperCon3D, a new dataset combining crystal structures and the critical temperature $T_{c}$ from SuperCon and ICSD. Utilizing SuperCon3D, we have developed two deep learning models for superconductor discovery and design. We propose a transformer-based GNN, SODNet, to analyze crystal geometries, including both ordered and disordered structures, potentially screening the entire ICSD. SODNet achieves SE(3)-equivariance through irreducible representation-based vector space features. Additionally, we introduce DiffCSP-SC, a transformer-based equivariant diffusion model for inverse design, capable of generating novel high $T_{c}$ superconductor candidates.

The main contributions of our work can be summarized as follows:

- A new dataset SuperCon3D containing both ordered-and-disordered crystal structures and experimental superconducting critical temperature is built for the first time.   
- We propose two deep learning models to showcase the possible methods for exploring Supercon3D dataset. The experimental result indicate that our proposed models outperform the existing similar methods.   
- Based on our proposed models, we present a list of candidate superconductors for future experimental validation. To the best of our knowledge, this is the first report of the candidate superconductors with disordered structures based on GNN methods.

# 2 Related Work

# 2.1 Superconducting Dataset

The SuperCon database encompasses around 33,000 superconductors, providing only their chemical formulas. Jarvis conducted electron-phonon coupling calculations for 1,058 materials, creating a computational database with BCS superconducting properties [12]. However, BCS theory applies mainly to conventional superconductors, and its predicted $\mathrm{T}_c$ values require experimental validation. The recent S2S dataset includes 1,685 entries with crystal structures and binary superconducting labels for machine learning-based discovery [31], but it's geared towards classification tasks. Diverging from these approaches, we constructed a dataset comprising crystal structures and experimental $\mathrm{T}_c$ values, suited for regression-based deep learning. Additionally, 3DSC [53] is a dataset that includes both $\mathrm{T}_c$ and structural information, comprising over 9,150 data entries obtained through elemental matching and manual doping. In contrast, the data in SuperCon3D is entirely derived from experimental observations in databases.

# 2.2 Crystal Modeling

Crystals are typically depicted as periodic graphs with a repeating minimum unit cell in a 3D lattice. While various equivariant GNN models have been developed for ordered crystal structures [61, 64, 11], research on representing disordered crystals is limited. Disorder, as defined by Müller et al. [38], involves varied orientations of atoms in unit cells, categorized into substitutional and positional disorder. MEGNet models disordered sites as elemental embeddings' linear combinations [9], only suitable for substitutional disorder. Our work aims to establish a comprehensive method for representing disordered graphs in crystals.

# 2.3 Generative Models

Drawing on the concepts of non-equilibrium thermodynamics $[52]$ , diffusion models create links between data and prior distributions through forward and backward Markov chains $[24]$ . This method has made significant strides in image generation $[46, 44]$ . Leveraging equivariant GNNs, diffusion models efficiently generate samples from invariant distributions, finding applications in conformation generation $[51, 63]$ , ab initio molecule design $[26]$ , protein generation $[33]$ , and more. The adaptation of diffusion models for crystal generation has also gained traction recently $[62, 34, 27]$ . In our research, we enhance diffusion generative modeling by incorporating an attention-based approach, aimed at reverse-engineering novel superconducting structures with a focus on $T_{c}$ properties.

# 3 Problem Formulation

# 3.1 From Ordered to Disordered Structures

![](images/1f44716dcc84a76e95c64d79d5ee88ca2cf6239777ecd49ae273491f43d16f55.jpg)  
Figure 1: Illustrations of periodic disorder patterns. The dotted red lines are minimum repeated cells. Grey lines are artificial boundaries to form one possible unit cell that repeats in infinite space for the given crystal. (a)→(b): An illustration of periodic substitutional disorder patterns in 2D space. In this case, a new atomic specie replaces the origin one. (c)→(d): An illustration of periodic positional disorder patterns in 2D space. Here, one site occurs position shift, and break the atomic symmetry in the crystal. The crystals are 3D structures in practice, and we use illustrations in 2D for simplicity.

We represent a 3D crystal as the infinite periodic arrangement of atoms in 3D space, and the smallest repeating unit is called a unit cell, as shown in Fig. 1. A unit cell can be defined as $\mathcal{M} = (\boldsymbol{L},\mathcal{S})$ , where $\boldsymbol{L} = [l_1,l_2,l_3]\in \mathbb{R}^{3\times 3}$ represents a minimum unit cell matrix containing three basic vectors to represent the periodicity of the crystal, and $\mathcal{S} = \{S_1,\dots ,S_N\}$ denotes a set of $N$ sites located in the unit cell. Specifically, a site describes a composition located at a specific position, which can be further defined as a triplet $S_{i} = (\boldsymbol{A}_{i},\boldsymbol{w}_{i},\boldsymbol{x}_{i})$ , where $\boldsymbol{A}_i = [\boldsymbol{a}_{i,1},\dots ,\boldsymbol{a}_{i,m_i}]\in \mathbb{R}^{m_i\times h}$ lists the $h$ -dimension features of the atom species composing the site, $\boldsymbol{w}_i\in \mathbb{R}^{m_i}$ describes the occupancy of each specie, and $\boldsymbol{x}_i\in \mathbb{R}^3$ denotes the Cartesian coordinate of the site. $m_{i}$ denotes the number of atoms in one site. Generally a crystal structure is composed of ordered sites, where $m_{i} = 1$ and $\boldsymbol{w}_i = [1]$ , i.e. each site is completely formed by a single atom specie. Under the influence of factors such as doping, superconductors may exhibit a disordered structure, containing two kinds of disordered sites:

Substitutional Disorder (SD). As illustrated in Fig. 1(a)→(b), SD involves a situation where the site is occupied by more than one atomic species. Specifically, for an SD site $S_{i}$ , we have

$$
\left\{ \begin{array}{l} \boldsymbol {m} _ {i} > 1, \\ \boldsymbol {a} _ {i, 1} \neq \boldsymbol {a} _ {i, 2} \neq \dots \neq \boldsymbol {a} _ {i, m _ {i}}, \\ \boldsymbol {w} _ {i, 1} + \boldsymbol {w} _ {i, 2} + \dots + \boldsymbol {w} _ {i, m _ {i}} = 1 \end{array} \right. \tag {1}
$$

Positional Disorder (PD). In this case, one atom in the unit cell occurs position shift as shown in Fig. 1(c)→(d). For a PD site $S_{i}$ , the atomic specie $\boldsymbol{a}_{i,1}$ partially locates in $\boldsymbol{x}_{i}$ with its occupancy.

$$
\left\{ \begin{array}{l} \boldsymbol {m} _ {i} = 1, \\ \boldsymbol {w} _ {i, 1} <   1. \end{array} \right. \tag {2}
$$

SD+PD (SPD). Specially, when $w_{i,1} + w_{i,2} + \cdots + w_{i,m_i} < 1$ in equation 19, both SD and PD can occur simultaneously.

Typically, there is also the occurrence of interstitial disorder. However, it was not detected in our dataset. Further details are provided in the Appendix B.1.

# 3.2 Superconducting Candidates Designing

In this study, we define the design of novel superconducting candidates in two ways: the first is “known materials repurposing”, where the potential superconducting candidates are screened from known structures. And the second involves designing novel material structures that are potential superconducting candidates. The specific definition of the deep learning task is as follows:

Superconductivity Prediction Task. The task involves predicting the $T_{c}$ values given the crystal structure M. Then, we use the predicting models to screen the big structure database to find candidate superconductors with high $T_{c}$ value.

Inverse Superconductor Generation Task. This task predicts the chemical composition $A$ , the Cartesian coordinates $X$ , and the lattice matrix $L$ targeted on higher $T_c$ values. To reduce the exploration space, we set $w_i = 1$ to generate ordered crystals. Such method can potentially design novel high $T_c$ superconductors.

# 4 The Proposed Method

# 4.1 SODNet

![](images/c23fd3be1d7ff0726bfca1a93a29238d71621b64ce11147bc0c1cb67125d1fbb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input 3D graph"] --> B["Atomic specie"]
    B --> C["k-hot"]
    C --> D["Linear"]
    D --> E["+"]
    F["RBF(||r_ij||)"] --> G["Interaction (YH)"]
    G --> H["Linear"]
    H --> I["Σ"]
    I --> J["f_ij"]
    K["Vector r_ij"] --> G
```
</details>

(a)

![](images/caff42642c2be6324405e39465ff83223d375fb34cba01330014a8994e3dc82e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Scalars f_ij^(0)"] --> B["Leaky ReLU"]
    B --> C["Linear"]
    C --> D["Softmax"]
    D --> E["a_ij"]
    F["Vectors f_ij^(L)"] --> G["Gate"]
    G --> H["Interaction"]
    H --> I["Linear"]
    I --> J["v_ij"]
    J --> K["\vec{r}_{ij}"]
    K --> L["Output"]
```
</details>

(b)   
Figure 2: Illustration of graph representation and equivariant graph attention layer in SODNet. (a). Illustration of node and edge embeddings. (b). The Type-0 and Type-L features operations in equivariant graph attention mechanism. $\oplus$ denotes addition and $\sum$ within a circle stands for summation over all neighbors.

Regarding the importance of symmetry in 3D physics space, it is essential to respect SE(3)-equivariance conditions in neural networks to reduce the model's dependence on data. To explore

the geometric structures with ordered and disordered graphs in SuperCon3D dataset, we propose SODNet, an effective architecture with SE(3)-equivariant graph attention to exploit 3D geometric content. We establish SE(3)-equivariance by utilizing equivariant features derived from vector spaces containing irreducible representations and trainable equivariant operations with the help of e3nn [19]. The core modules of the proposed SODNet is illustrated in Fig. 2. We elaborate the details as follows.

# 4.1.1 Disordered Graph Representation

Considering the presence of disordered structures within SuperCon3D, we design two embedding blocks aimed at enhancing the model's ability to effectively capture these disordered inputs.

Node embedding. In the graph network approach, we apply the k-hot embedding [10] as the feature vector $a_{i,k}$ , which encodes the atomic property corresponding to each atom specie. To extend such scheme to disordered structures, we further represent each site $S_{i}$ as a linear combination of atomic occupancy and atomic encoding as:

$$
\boldsymbol {h} _ {i} = \left\{ \begin{array}{l l} \boldsymbol {a} _ {i, 1}, & S _ {i} \text {   is   ordered }, \\ \sum_ {k} \boldsymbol {w} _ {i, k} \boldsymbol {a} _ {i, k}, & S _ {i} \text {   is   SD   or   SPD }, \\ \boldsymbol {w} _ {i, 1} \boldsymbol {a} _ {i, 1}, & S _ {i} \text {   is   PD }. \end{array} \right. \tag {3}
$$

Edge embedding. Then, we consider 3D geometric features by incorporating interatomic distance as well as vectors $\vec{r}_{ij}$ equipped with spherical harmonics as follows:

$$
\boldsymbol {E} = \boldsymbol {w} _ {i} \boldsymbol {w} _ {j} \boldsymbol {R B F} (\| \vec {r} _ {i j} \|), \tag {4}
$$

$$
\boldsymbol {x} _ {i j} = \varphi (\boldsymbol {h} _ {i}) + \varphi (\boldsymbol {h} _ {j}), \tag {5}
$$

$$
\boldsymbol {f} _ {i j} = \varphi_ {f} (\boldsymbol {x} _ {i j} \otimes_ {c \boldsymbol {E}} ^ {T P} \boldsymbol {S} \boldsymbol {H} (\vec {r} _ {i j})) \tag {6}
$$

where $RBF(\|\vec{r}_{ij}\|)$ is the radial distribution function (RBF) expansion for interatomic bond distance. Specially, we set $w_{i}$ and $w_{j}$ to 1 when i and j sites are ordered. The initial edges are constructed by k-nearest neighbor (kNN) methods from Yan et al. [64]. Here, we remove the close edges when bond distance meets $\|\vec{r}_{ij}\| \leq R_{i} + R_{j}$ to avoid strong interactions caused by disordered sites, where $R_{i}$ and $R_{j}$ are atomic radii. $x_{ij}$ combines the features of target node i and source node j with linear layers to obtain initial message. $\varphi$ represents an MLP. $SH(\vec{r}_{ij})$ is spherical harmonics embeddings (SH) of relative position $\vec{r}_{ij}$ , cE is weights parametrized by E. Finally, we obtain $f_{ij}$ to derive non-linear messages and attention weights.

# 4.1.2 Equivariant Graph Attention

Given $f_{ij}$ containing multiple type-L vectors, which are SE(3)-equivariant irreps features. In the context of learning on 3D atomistic graphs, it is essential that features and learnable functions exhibit SE(3)-equivariance with respect to geometric transformations acting on the position $\vec{r}_{ij}$ . We split $f_{ij}$ into $f_{ij}^{L}$ and $f_{ij}^{0}$ . The $f_{ij}^{0}$ is scalar and independent on inputs. However, the $f_{ij}^{L}$ consists of type-L vectors, which can break equivariance. Inspired by Liao and Smidt [30], we apply different operations to each group of $f_{ij}$ .

Type-0 features. Given $f_{ij}^{0}$ , we adopt the leaky ReLU activation and a softmax operation for $\beta_{ij}$ :

$$
\zeta_ {i j} = \alpha^ {\top} \text { LeakReLU } (\boldsymbol {f} _ {i j} ^ {0}), \tag {7}
$$

$$
\beta_ {i j} = \frac {\exp \left(\zeta_ {i j}\right)}{\sum_ {k \in \mathcal {N} (i)} \exp \left(\zeta_ {i k}\right)} \tag {8}
$$

Where $\alpha$ is a learnable vector of the same dimension as $f_{ij}^{0}$ and $\zeta_{ij}$ is a scalar.

Type-L features. We perform non-linear transformation on $f_{ij}^{L}$ to obtain non-linear message:

$$
\mu_ {i j} = \operatorname{Gate} \left(\boldsymbol {f} _ {i j} ^ {L}\right), \tag {9}
$$

$$
v _ {i j} = \varphi_ {f} (\mu_ {i j} \otimes_ {\omega} ^ {T P} \boldsymbol {S} \boldsymbol {H} (\vec {r} _ {i j})) \tag {10}
$$

We apply the equivariant gate activation as Weiler et al. [59] and present the details in Appendix B.2. Then, the similar method is eq. 6 is used to obtain $v_{ij}$ .

Finally, $\beta_{ij}$ and $v_{ij}$ are further transformed features into scalars by multiplication operation. We perform mean aggregate over all nodes to predict the $T_{c}$ value by:

$$
T _ {c} (i) = \frac {1}{| \mathcal {N} (i) |} \sum_ {j \in \mathcal {N} (i)} \beta_ {i j} \cdot v _ {i j}, \tag {11}
$$

$$
T _ {c} = \frac {1}{| \mathcal {V} |} \sum_ {i \in \mathcal {V}} T _ {c} (i) \tag {12}
$$

Where $\mathcal{N}(i)$ is the neighbors on node i, and V denotes the set of all nodes in the graph.

# 4.2 DiffCSP-SC

Based on DiffCSP [27], we further equip our method with superconductivity guidance for crystal generation. The original DiffCSP proposes a periodic SE(3) equivariant model to jointly optimize lattice matrix L and fractional coordinates $F = L^{-1}X$ in a diffusion-based framework, and additionally utilizes a time-dependent guidance model [2] for property optimization. Here, X denotes the Cartesian coordinates. We extend DiffCSP with a more powerful architecture for SuperCon3D dataset.

# 4.2.1 Transformer-based Architecture

The denoising and guidance model of the original DiffCSP share the same architecture, which is built upon EGNN [47], following the standard message passing neural networks (MPNN) framework [20]. To capture the key features related to superconductivity, we employ a transformer-based model for DiffCSP-SC. Let $\boldsymbol{H}^{(s)} = [\boldsymbol{h}_{1}^{(s)}, \cdots, \boldsymbol{h}_{N}^{(s)}]$ denote the node representations in the s-th layer, where N is the number of nodes. The input feature is given by $\boldsymbol{h}_{i}^{(0)} = \varphi(f_{\mathrm{atom}}(\boldsymbol{a}_{i}), f_{\mathrm{pos}}(t))$ , where $f_{atom}$ and $f_{pos}$ are the atomic embedding and sinusoidal positional encoding [56, 24], respectively. $\varphi$ is a multi-layer perception (MLP).

The output features $\boldsymbol{h}_{i}^{(s)}$ are computed by

$$
\boldsymbol {h} _ {i} ^ {(s)} = \boldsymbol {h} _ {i} ^ {(s - 1)} + \sum_ {j = 1} ^ {N} \theta_ {i j} ^ {(s)} \boldsymbol {v} _ {i j} ^ {(s)} \tag {13}
$$

where $\theta_{ij}$ is matrix capturing the similarity between queries and keys.

$$
\theta_ {i j} ^ {(s)} = \text { Softmax } (\frac {\boldsymbol {q} _ {i} ^ {(s) \top} \boldsymbol {k} _ {i j} ^ {(s)}}{\sqrt {d}}) \tag {14}
$$

Here $d$ is the dimension of the hidden state. The queries, keys and values of $\pmb{q}_i^{(s)}$ , $\pmb{k}_{ij}^{(s)}$ and $\pmb{v}_{ij}^{(s)}$ in attention mechanism are unfolded as follows:

$$
\boldsymbol {q} _ {i} ^ {(s)} = \varphi_ {q} (\boldsymbol {h} _ {i} ^ {(s - 1)}), \tag {15}
$$

$$
\boldsymbol {k} _ {i j} ^ {(s)} = \varphi_ {k} (\boldsymbol {h} _ {i} ^ {(s - 1)}, \boldsymbol {L} ^ {\top} \boldsymbol {L}, \psi_ {\mathrm{FT}} (\boldsymbol {f} _ {j} - \boldsymbol {f} _ {i})), \tag {16}
$$

$$
\boldsymbol {v} _ {i j} ^ {(s)} = \varphi_ {v} (\boldsymbol {h} _ {i} ^ {(s - 1)}, \boldsymbol {L} ^ {\top} \boldsymbol {L}, \psi_ {\mathrm{FT}} (\boldsymbol {f} _ {j} - \boldsymbol {f} _ {i})) \tag {17}
$$

Where $\varphi_{q}$ , $\varphi_{k}$ and $\varphi_{v}$ are MLPs. L is the unit lattice cell. Specially, $L^{\top}L$ is used to ensure O(3)-equivariance in diffusion step. The transform $\psi_{FT}$ is able to extract various frequencies of all relative fractional distances that are helpful for crystal structure modeling, and more importantly, $\psi_{FT}$ is periodic translation invariant, namely, $\psi_{\mathrm{FT}}(w(\boldsymbol{f}_{j}+\boldsymbol{t})-w(\boldsymbol{f}_{i}+\boldsymbol{t}))=\psi_{\mathrm{FT}}(\boldsymbol{f}_{j}-\boldsymbol{f}_{i})$ for any translation t. The part corresponding to original DiffCSP is presented in Appendix B.3.

# 4.2.2 Improved Predictor for Evaluation

After denoising process, we need to predict the $T_{c}$ values of the generated samples. We adopt SODNet as an effective substitute of DFT-based predictors. Similar to CDVAE [62], we calculate the success rate (SR) as the proportion of optimized structures reaching the required thresholds. Given the samples $\tilde{D}$ , SR is defined as

$$
\mathrm{SR} \alpha (\tilde {\mathcal {D}}) = \frac {\| \tilde {\mathcal {M}} | \tilde {\mathcal {M}} \in \tilde {\mathcal {D}} , \varphi (\tilde {\mathcal {M}}) > P _ {1 0 0 - \alpha} (\mathcal {D} _ {\text {train}}) \|}{\| \tilde {\mathcal {D}} \|}, \tag {18}
$$

where $\varphi$ is the SODNet predictor and $P_{100-\alpha}(\mathcal{D}_{\mathrm{train}})$ is the $100 - \alpha$ percentile of the $T_{c}$ values in the training set. Similarly, we define the novelty success rate (NSR) as a metric to assess the generation of novel structures, with detailed definitions and explanations provided in the Appendix D.

# 4.2.3 Pre-training

Considering the SuperCon3D dataset's limited structures, which doesn't fully capture the diversity in atomic species, lattice parameters, and atomic spatial distributions, we pre-trained our model on approximately 1.14 million unique 3D crystals sourced from existing databases, including Materials Project, OQMD, ICSD and Matgen.

# 5 Experiments

# 5.1 Setup

# 5.1.1 SuperCon3D dataset.

We extracted approximately 33,000 superconductors with their chemical formulas and corresponding critical temperatures from SuperCon. After removing duplicates and non-superconductors, we identified 11,949 superconducting materials. Additionally, over 200,000 ordered and disordered crystal structures were gathered from the ICSD database [3]. We then matched these 11,949 SuperCon entries with 208,425 ICSD entries based on chemical composition, space group and lattice parameter. Moreover, $\mathrm{T}_c$ values and structural data for hydrogen-enriched superconductors were collated from various literature sources. This process resulted in 1,578 superconductor data entries, each featuring both $\mathrm{T}_c$ and crystal structure. To ensure the dataset's integrity, all entries were vetted by domain experts and accompanied by referenced literature. Detailed data descriptions are provided in Appendix A.

# 5.1.2 Evaluation Metrics.

We mainly compare our proposals with other crystal property predictors and inverse crystal structure generative models. For property predicting tasks, we mainly employ Mean Absolute Error (MAE) and R-Square $(\mathbb{R}^2)$ for $T_{c}$ prediction. In addition, we also use visualization and interpretable analysis to verify our model. For inverse crystal structure generative task, we calculate the success rate (SR) as the percentage of the 100 optimized structures achieving 10, 30, 50 percentiles of the superconducting property distribution.

# 5.2 Experimental Results and Discussion for Superconductivity Prediction

# 5.2.1 Comparison on Dataset.

We present a summary of comparisons with previous crystal property predictors in Table 1. SODNet consistently outperforms the other competitors both on ordered and disordered structures. For example, SODNet achieves $17.6\%$ reduction on MAE and about $4.4\%$ improvements on $\mathbb{R}^2$ than the second ranked Matformer. When considering the PD disordered structure between SODNet and MEGNet, SODNet gets about $41.7\%$ reduction on MAE and almost $66.1\%$ improvement on $\mathbb{R}^2$ than MEGNet. It is worth noting that when we incorporate disordered structures into the training and validation sets, the metrics of $\mathbb{R}^2$ and MAE both show improvements, indicating that accurately representing disordered structures is beneficial for the prediction of ordered structure properties. This also means that SODNet can be further improved with larger dataset in the future work.

Table 1: Predicting models performance on SuperCon3D dataset. 'O' indicates that using ordered data. ML models with -c and -geo denote composition and structure features. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Data</td><td colspan="2">Performance</td></tr><tr><td>Train</td><td>Test</td><td>MAE (logK)↓</td><td>R2↑</td></tr><tr><td>RF-c</td><td>O</td><td>O</td><td>0.738±0.165</td><td>0.711±0.050</td></tr><tr><td>SVM-c</td><td>O</td><td>O</td><td>0.632±0.094</td><td>0.801±0.041</td></tr><tr><td>RF-geo</td><td>O</td><td>O</td><td>0.741±0.115</td><td>0.759±0.051</td></tr><tr><td>SVM-geo</td><td>O</td><td>O</td><td>0.578±0.114</td><td>0.827±0.042</td></tr><tr><td>SchNet</td><td>O</td><td>O</td><td>0.891±0.041</td><td>0.401±0.032</td></tr><tr><td>CGCNN</td><td>O</td><td>O</td><td>0.879±0.047</td><td>0.405±0.022</td></tr><tr><td>DimeNet++</td><td>O</td><td>O</td><td>0.811±0.058</td><td>0.434±0.092</td></tr><tr><td>SphereNet</td><td>O</td><td>O</td><td>0.762±0.048</td><td>0.467±0.096</td></tr><tr><td>ALIGNN</td><td>O</td><td>O</td><td>0.755±0.049</td><td>0.479±0.090</td></tr><tr><td>Matformer</td><td>O</td><td>O</td><td>0.748±0.043</td><td>0.570±0.135</td></tr><tr><td rowspan="2">MEGNet</td><td>O</td><td>O</td><td>0.794±0.006</td><td>0.497±0.009</td></tr><tr><td>O/SD</td><td>O/SD</td><td>0.889±0.049</td><td>0.431±0.058</td></tr><tr><td rowspan="4">SODNet</td><td>O</td><td>O</td><td>0.622±0.112</td><td>0.595±0.101</td></tr><tr><td>O/SD/PD/SPD</td><td>O</td><td>0.584±0.119</td><td>0.634±0.117</td></tr><tr><td>O/SD</td><td>O/SD</td><td>0.518±0.084</td><td>0.716±0.064</td></tr><tr><td>O/SD/PD/SPD</td><td>O/SD/PD/SPD</td><td>0.505±0.055</td><td>0.748±0.032</td></tr></table>

# 5.2.2 Ablation Study.

We conduct ablation studies to investigate crucial factors that influence the performance of the proposed SODNet. Table 2 shows the experimental results of SODNet with disordered graph representation and equivariant graph attention. When nodes and edges are embedded without atomic occupancy, both the MAE and $R^{2}$ metrics exhibit a decline in performance. Among them, node embedding is more sensitive to disordered graphs, leading to almost half of the performance loss.

Table 2: Ablation studies of SODNet on SuperCon3D. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Performance</td></tr><tr><td>MAE (logK)↓</td><td>R2↑</td></tr><tr><td colspan="3">w/o Occupancy Embedding</td></tr><tr><td>w/o disorder node embedding</td><td>0.990±0.033</td><td>0.365±0.044</td></tr><tr><td>w/o disorder edge embedding</td><td>0.592±0.087</td><td>0.655±0.046</td></tr><tr><td colspan="3">w/o O(3) Equivariance</td></tr><tr><td>w/o equivariant operations</td><td>0.611±0.046</td><td>0.618±0.027</td></tr><tr><td>SODNet</td><td>0.505±0.055</td><td>0.748±0.032</td></tr></table>

Additionally, if we replace the type-L layer with MLPs, the proposed model achieves worse performance, indicating that the type-L features with equivariant activation function plays a crucial role in O(3) invariance for vectors.

# 5.2.3 Real-world Superconductors Validation

Table 3: Recently discovered superconductors (not included in the training data). 

<table><tr><td>Material</td><td>O/SD/PD</td><td> $T_c^{exp}$  (K)</td><td> $T_c^{pred}$  (K)</td><td>Relative Error(%)</td></tr><tr><td> $CaH_6$ </td><td>O</td><td>215 [35]</td><td>242.25</td><td>12.67</td></tr><tr><td>Ti</td><td>O</td><td>26 [66]</td><td>8.50</td><td>67.31</td></tr><tr><td> $CsV_3Sb_5$ </td><td>O</td><td>2.3 [18]</td><td>2.36</td><td>6</td></tr><tr><td> $Cs(V_{0.93}Nb_{0.07})_3Sb_5$ </td><td>SD</td><td>4.45 [29]</td><td>4.71</td><td>5.84</td></tr><tr><td> $Zr_4Rh_2O$ </td><td>O</td><td>3.73 [58]</td><td>4.12</td><td>10.45</td></tr><tr><td>Zr4Pd2O</td><td>O</td><td>2.73 [58]</td><td>2.82</td><td>3.3</td></tr><tr><td> $LaFeSiO_{0.9}$ </td><td>PD</td><td>10 [23]</td><td>7.93</td><td>20.7</td></tr></table>

To assess the model's real-world relevance, we gathered newly discovered superconductors from the last three years, not present in our training data. Table 8 reveals that except for titanium superconductors, other materials' critical temperatures $(\mathrm{T}_c)$ are predicted with low relative error margins (below $21\%$ ). This underscores the model's ability to predict $\mathrm{T}_c$ values beyond its training scope, highlighting its utility in new material discovery. The outlier predictions for titanium could stem from close atomic proximities under extreme pressures (248 GPa), a condition scarcely represented in our training set. More details are presented in Appendix E.1.

# 5.2.4 Potential Superconducting Materials.

Using our model, we screened the ICSD database to identify potential high- $\mathrm{Tc}$ superconductors. Appendix E.2 lists 27 candidates, including cuprate, H-rich, heavy-Fermion, iron-based, and other types. The top three candidates are $\mathrm{Ba}_{1.1432}\mathrm{Co}_{0.1429}\mathrm{O}_{3.0009}\mathrm{Rh}_{0.8574}$ , $\mathrm{ErH}_3$ , and $\mathrm{Ba}_{0.515}\mathrm{Ca}_{0.485}$ , previously unreported. This is the first identification of disordered superconducting candidates from ICSD using a GNN method. Given that most ICSD structures are experimentally synthesized, these candidates are valuable for further research. Our model effectively screens disordered high- $\mathrm{T_c}$ structures, demonstrating its usefulness. Additionally, we highlight four prime high- $\mathrm{T_c}$ candidates with analogous parent structures in Table 10 and 11 of the Appendix E.2. In the Appendix E.3, we provide an interpretation of our SODNet predictor by identifying the features that the model prioritizes when making predictions, using the case of order-and-disorder- $\mathrm{MgB}_2$ as an example. This analysis demonstrates SODNet's ability to capture the correlations between superconducting properties and structural characteristics.

# 5.3 Experimental Results and Discussion for Inverse Crystal Structure Generation

# 5.3.1 Comparison on Dataset.

We summarize the comparisons to previous main generative models in Table 4 and present training details in Appendix C.2. Without pretraining, CDVAE, SyMat and DiffCSP generate poor crystal structures, exhibiting extremely low SR performance. The main reason for this phenomenon may be the vast compound space of superconducting materials, making it difficult to effectively sample the atomic species and atomic spatial coordinates. The DiffCSP-SC model we propose shows a slight performance improvement compared to the two models mentioned above under the same conditions. CDVAE lacks translation invariance for atomic coordinates, which affects the quality of generated structures. This low performance metric is also observed in the DiffCSP [27] and aligns with our findings. Moreover, in comparison to DiffCSP, DiffCSP-SC containing an attention mechanism exhibits higher SR performance, indicating that DiffCSP may capture structural features associated with high $T_{c}$ . We will give more discussions for DiffCSP-SC in section Ablation Study. Notably, DiffCSP-SC consistently achieved the highest performance across the NSR metric, with detailed results presented in the Appendix D.

Table 4: Results for inverse crystal structures generation. "O" and "Pre-training" indicate models trained on SuperCon3D's ordered structures and a collection of 1.14 million stable structures, respectively. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Data</td><td colspan="3">Performance</td></tr><tr><td>SR10</td><td>SR30</td><td>SR50</td></tr><tr><td>CDVAE</td><td>O</td><td>0.03</td><td>0.03</td><td>0.03</td></tr><tr><td>SyMat</td><td>O</td><td>0.03</td><td>0.04</td><td>0.04</td></tr><tr><td>DiffCSP</td><td>O</td><td>0.04</td><td>0.05</td><td>0.05</td></tr><tr><td>DiffCSP-SC</td><td>O</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td>CDVAE</td><td>Pre-training + O</td><td>0.25</td><td>0.25</td><td>0.30</td></tr><tr><td>SyMat</td><td>Pre-training + O</td><td>0.28</td><td>0.28</td><td>0.35</td></tr><tr><td>DiffCSP</td><td>Pre-training + O</td><td>0.30</td><td>0.30</td><td>0.45</td></tr><tr><td>DiffCSP-SC</td><td>Pre-training + O</td><td>0.37</td><td>0.37</td><td>0.50</td></tr></table>

# 5.3.2 Ablation Study.

In ablation studies detailed in Table 5, we examine key components of our DiffCSP-SC model. 1. Assessing the transformer's impact, its removal and reverting to the original DiffCSP approach led to a notable performance drop, especially in SR10 and SR30 metrics, implicating a decrease in high $\mathrm{T}_c$ superconductor generation. This suggests that attention mechanisms in transformers effectively capture the complex atomic compositions of high $\mathrm{T}_c$ superconductors, which often involve multi-component, multi-element structures. 2. The pre-training methodology's significance is highlighted by its ability to manage the vast feature space of atomic species, coordinates, and unit cells in ordered crystals. Without it, as seen when training solely on a limited subset from SuperCon3D, the model's efficacy in generating valid superconductors significantly diminishes.

# 5.3.3 Candidate Superconductors.

Utilizing our model, we aimed to generate novel superconductor candidates with high $T_{c}$ values. Table 12 and 13 in Appendix E.4 displays 32 potential high $T_{c}$ superconducting materials categorized as cuprate, H-rich, heavy-Fermion, iron-based, and other types. We initially assessed the novelty of these structures through similarity calculations with our 1.14 million-structure database. Interestingly, our findings reveal three candidates, index 8, 9, and 12, previously reported for $T_{c}$ using computational methods. Additionally, another candidate, index 21 and 22, demonstrated superconductivity upon doping and pressing. Subsequently, density-functional theory

Table 5: Ablation studies of DiffCSP-SC on SuperCon3D. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Performance</td></tr><tr><td>SR10</td><td>SR30</td><td>SR50</td></tr><tr><td colspan="4">w/o Transformer</td></tr><tr><td>w/o attention</td><td>0.28</td><td>0.28</td><td>0.45</td></tr><tr><td colspan="4">w/o Pre-training</td></tr><tr><td>w/o pre-training</td><td>0.05</td><td>0.05</td><td>0.10</td></tr><tr><td>DiffCSP-SC</td><td>0.37</td><td>0.37</td><td>0.50</td></tr></table>

(DFT) were performed on selected candidates to verify their superconducting properties. Notably, Van Hove singularities (VHS) were observed in the electronic structures of $Ba_{2}CuCl_{2}O_{2}$ , Lu, and $BaFe_{2}Se_{2}$ , as further detailed in Appendix E.5. VHS is a significant aspect in superconductivity research, often explored for its potential influence [5].

# 6 Conclusion and Discussion

In conclusion, a novel dataset has been constructed as a benchmark for future deep learning-based superconductivity research. Utilizing the dataset, we put forth two deep learning approaches for the design of high $T_{c}$ superconductors: a property prediction model for screening the known structures, and a generative model for creating the novel structures. To further validate the efficacy of the model, we apply the predicting model to screen the entire ICSD and identify a list of ordered and disorder superconducting candidates. By employing pretraining on large-scale crystal structures, we have achieved the capability to perform reverse structure design on limited superconducting data points.

Our SuperCon3D dataset, featuring experimental structures and $T_{c}$ values, paves the way for real-world superconductor applications. Combined with SODNet, which addresses disordered graph issues previously overlooked by the AI community, and DiffCSP-SC for novel designs. However, the accuracy of data-driven models remains constrained by the collected superconducting dataset. As Fig. 4 in the Appendix shows, data unevenness and elemental skewness (especially in Cu and O) may bias the model. Additionally, as Table 8 indicates, atomic distributions under extreme pressures contribute to predictive errors. Addressing these, Fig. 8 presents our pipeline, combining DiffCSP-SC and SODNet, to design and validate novel superconductors through wet experiments, iteratively enriching the dataset for improved model training and accuracy.

# 7 Acknowledgement

This work was jointly supported by the following projects: National Science and Technology Major Project (2022ZD0117805), the National Natural Science Foundation of China (No. 61925601, No. 62376276), Beijing Nova Program (20230484278).

# References

[1] M. Azam, M. Manasa, T. Zajarniuk, R. Diduszko, T. Cetner, A. Morawski, A. Wiśniewski, and S. J. Singh. Antimony doping effect on the superconducting properties of smfeas (o, f). IEEE Transactions on Applied Superconductivity, 2023.   
[2] F. Bao, M. Zhao, Z. Hao, P. Li, C. Li, and J. Zhu. Equivariant energy-guided sde for inverse molecular design. arXiv preprint arXiv:2209.15408, 2022.   
[3] G. Bergerhoff, R. Hundt, R. Sievers, and I. D. Brown. The inorganic crystal structure data base. Journal of Chemical Information and Computer Sciences, 23(2):66–69, 1983. ISSN 0095-2338. doi: 10.1021/ci00038a003. URL https://pubs.acs.org/doi/abs/10.1021/ci00038a003.

[4] P. E. Blöchl. Projector augmented-wave method. Physical review B, 50(24):17953, 1994.   
[5] J. Bok and J. Bouvier. Superconductivity and the van hove scenario. Journal of superconductivity and novel magnetism, 25:657–667, 2012.   
[6] A. I. Braginski. Superconductor electronics: status and outlook. Journal of superconductivity and novel magnetism, 32(1):23–44, 2019.   
[7] M. Chan and G. Ceder. Efficient band gap prediction for solids. Physical review letters, 105(19):196403, 2010.   
[8] C. Chen, W. Ye, Y. Zuo, C. Zheng, and S. P. Ong. Graph networks as a universal machine learning framework for molecules and crystals. Chemistry of Materials, 31(9):3564–3572, 2019. ISSN 0897-4756. doi: 10.1021/acs.chemmater.9b01294. URL https://doi.org/10.1021/acs.chemmater.9b01294.   
[9] C. Chen, Y. Zuo, W. Ye, X. Li, and S. P. Ong. Learning properties of ordered and disordered materials from multi-fidelity data. Nature Computational Science, 1(1):46–53, 2021. ISSN 2662-8457. doi: 10.1038/s43588-020-00002-x. URL https://doi.org/10.1038/s43588-020-00002-x.   
[10] P. Chen, J. Chen, H. Yan, Q. Mo, Z. Xu, J. Liu, W. Zhang, Y. Yang, and Y. Lu. Improving material property prediction by leveraging the large-scale computational database and deep learning. The Journal of Physical Chemistry C, 126(38):16297–16305, 2022.   
[11] K. Choudhary and B. DeCost. Atomistic line graph neural network for improved materials property predictions. npj Computational Materials, 7(1):1–8, 2021.   
[12] K. Choudhary and K. Garrity. Designing high-tc superconductors with bcs-inspired screening, density functional theory and deep-learning. arXiv preprint arXiv:2205.00060, 2022.   
[13] K. Choudhary, B. DeCost, C. Chen, A. Jain, F. Tavazza, R. Cohn, C. W. Park, A. Choudhary, A. Agrawal, S. J. Billinge, et al. Recent advances and applications of deep learning methods in materials science. npj Computational Materials, 8(1):59, 2022.   
[14] S. Elfwing, E. Uchibe, and K. Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11, 2018. ISSN 0893-6080. doi: https://doi.org/10.1016/j.neunet.2017.12.012. URL https://www.sciencedirect.com/science/article/pii/S0893608017302976. Special issue on deep reinforcement learning.   
[15] Y. Fu, X. Du, L. Zhang, F. Peng, M. Zhang, C. J. Pickard, R. J. Needs, D. J. Singh, W. Zheng, and Y. Ma. High-pressure phase stability and superconductivity of pnictogen hydrides and chemical trends for compressed hydrides. Chemistry of Materials, 28(6):1746–1755, 2016.   
[16] J. Gasteiger, S. Giri, J. T. Margraf, and S. Günnemann. Fast and uncertainty-aware directional message passing for non-equilibrium molecules. In Machine Learning for Molecules Workshop, NeurIPS, 2020.   
[17] J. Gasteiger, F. Becker, and S. Günnemann. Gemnet: Universal directional graph neural networks for molecules. In Conference on Neural Information Processing Systems (NeurIPS), 2021.   
[18] J. Ge, P. Wang, Y. Xing, Q. Yin, H. Lei, Z. Wang, and J. Wang. Discovery of charge-4e and charge-6e superconductivity in kagome superconductor csv3sb5. arXiv preprint arXiv:2201.10352, 2022.   
[19] M. Geiger and T. E. Smidt. e3nn: Euclidean neural networks. CoRR, abs/2207.09453, 2022. doi: 10.48550/arXiv.2207.09453. URL https://doi.org/10.48550/arXiv.2207.09453.   
[20] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural message passing for quantum chemistry. In D. Precup and Y. W. Teh, editors, Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 1263–1272. PMLR, 2017. URL http://proceedings.mlr.press/v70/gilmer17a.html.

[21] F. Giustino. Electron-phonon interactions from first principles. Reviews of Modern Physics, 89(1):015003, 2017.   
[22] Y.-L. Hai, N. Lu, H.-L. Tian, M.-J. Jiang, W. Yang, W.-J. Li, X.-W. Yan, C. Zhang, X.-J. Chen, and G.-H. Zhong. Cage structure and near room-temperature superconductivity in tbh n (n=1–12). The Journal of Physical Chemistry C, 125(6):3640–3649, 2021.   
[23] M. F. Hansen, J.-B. Vaney, C. Lepoittevin, F. Bernardini, E. Gaudin, V. Nassif, M.-A. Méasson, A. Sulpice, H. Mayaffre, M.-H. Julien, et al. Superconductivity in the crystallogenide lafesio1- $\delta$ with squeezed fesi layers. npj Quantum Materials, 7(1):1–8, 2022.   
[24] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.   
[25] E. Hoogeboom, V. G. Satorras, C. Vignac, and M. Welling. Equivariant diffusion for molecule generation in 3d. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 8867–8887. PMLR, 2022. URL https://proceedings.mlr.press/v162/hoogeboom22a.html.   
[26] E. Hoogeboom, V. G. Satorras, C. Vignac, and M. Welling. Equivariant diffusion for molecule generation in 3d. In International Conference on Machine Learning, pages 8867–8887. PMLR, 2022.   
[27] R. Jiao, W. Huang, P. Lin, J. Han, P. Chen, Y. Lu, and Y. Liu. Crystal structure prediction by joint equivariant diffusion on lattices and fractional coordinates. In Workshop on "Machine Learning for Materials" ICLR 2023, 2023. URL https://openreview.net/forum?id=VPByphdu24j.   
[28] W. Li, J. Zhao, L. Cao, Z. Hu, Q. Huang, X. Wang, Y. Liu, G. Zhao, J. Zhang, Q. Liu, et al. Superconductivity in a unique type of copper oxide. Proceedings of the National Academy of Sciences, 116(25):12156–12160, 2019.   
[29] Y. Li, Q. Li, X. Fan, J. Liu, Q. Feng, M. Liu, C. Wang, J.-X. Yin, J. Duan, X. Li, et al. Tuning the competition between superconductivity and charge order in the kagome superconductor cs (v 1-x nb x) 3 sb 5. Physical Review B, 105(18):L180507, 2022.   
[30] Y.-L. Liao and T. Smidt. Equiformer: Equivariant graph attention transformer for 3d atomistic graphs. In International Conference on Learning Representations, 2023.   
[31] K. Liu, K. Yang, J. Zhang, and R. Xu. S2snet: A pretrained neural network for superconductivity discovery. In L. D. Raedt, editor, Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pages 5101–5107. ijcai.org, 2022. doi: 10.24963/ijcai.2022/708. URL https://doi.org/10.24963/ijcai.2022/708.   
[32] Y. Liu, L. Wang, M. Liu, Y. Lin, X. Zhang, B. Oztekin, and S. Ji. Spherical message passing for 3d molecular graphs. In International Conference on Learning Representations (ICLR), 2022.   
[33] S. Luo, Y. Su, X. Peng, S. Wang, J. Peng, and J. Ma. Antigen-specific antibody design and optimization with diffusion-based generative models. bioRxiv, 2022.   
[34] Y. Luo, C. Liu, and S. Ji. Towards symmetry-aware generation of periodic materials. Advances in Neural Information Processing Systems, 36, 2024.   
[35] L. Ma, K. Wang, Y. Xie, X. Yang, Y. Wang, M. Zhou, H. Liu, X. Yu, Y. Zhao, H. Wang, et al. High-temperature superconducting phase in clathrate calcium hydride cah 6 up to 215 k at a pressure of 172 gpa. Physical Review Letters, 128(16):167001, 2022.   
[36] J. L. MacManus-Driscoll and S. C. Wimbush. Processing and application of high-temperature superconducting coated conductors. Nature Reviews Materials, 6(7):587–604, 2021.   
[37] H. J. Monkhorst and J. D. Pack. Special points for brillouin-zone integrations. Physical review B, 13(12):5188, 1976.

[38] P. Müller, R. Herbst-Irmer, A. L. Spek, T. R. Schneider, and M. R. Sawaya. Crystal Structure Refinement: A Crystallographer's Guide to SHELXL. Oxford University Press, 07 2006. ISBN 9780198570769. doi: 10.1093/acprof:oso/9780198570769.001.0001. URL https://doi.org/10.1093/acprof:oso/9780198570769.001.0001.   
[39] V. D. Neverov, A. E. Lukyanov, A. V. Krasavin, A. Vagov, and M. D. Croitoru. Correlated disorder as a way towards robust superconductivity. Communications Physics, 5(1):177, 2022.   
[40] L. Novakovic, A. Salamat, and K. V. Lawler. Machine learning using structural representations for discovery of high temperature superconductors. arXiv preprint arXiv:2301.10474, 2023.   
[41] J. P. Perdew, K. Burke, and M. Ernzerhof. Generalized gradient approximation made simple. Physical review letters, 77(18):3865, 1996.   
[42] S. Peschke, T. Stürzer, and D. Johrendt. Ba1–xrbxfe2as2 and generic phase behavior of hole-doped 122-type superconductors. Zeitschrift für anorganische und allgemeine Chemie, 640(5):830–835, 2014.   
[43] A. Ptok, K. J. Kapcia, M. Sternik, and P. Piekarz. Superconductivity of kfe2as2 under pressure: Ab initio study of tetragonal and collapsed tetragonal phases. Journal of Superconductivity and Novel Magnetism, 33(8):2347–2354, 2020.   
[44] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
[45] P. Reiser, M. Neubert, A. Eberhard, L. Torresi, C. Zhou, C. Shao, H. Metni, C. van Hoesel, H. Schopmans, T. Sommer, et al. Graph neural networks for materials science and chemistry. Communications Materials, 3(1):93, 2022.   
[46] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent diffusion models, 2021.   
[47] V. G. Satorras, E. Hoogeboom, and M. Welling. E(n) equivariant graph neural networks. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 9323–9332. PMLR, 2021. URL http://proceedings.mlr.press/v139/satorras21a.html.   
[48] V. G. Satorras, E. Hoogeboom, and M. Welling. E (n) equivariant graph neural networks. In International Conference on Machine Learning, pages 9323–9332. PMLR, 2021.   
[49] J. Schön, M. Dorget, F. Beuran, X. Zu, E. Arushanov, C. Deville Cavellin, and M. Lagues. Superconductivity in cacuo2 as a result of field-effect doping. Nature, 414(6862):434–436, 2001.   
[50] K. T. Schütt, H. E. Sauceda, P.-J. Kindermans, A. Tkatchenko, and K.-R. Müller. Schnet—a deep learning architecture for molecules and materials. The Journal of Chemical Physics, 148(24):241722, 2018.   
[51] C. Shi, S. Luo, M. Xu, and J. Tang. Learning gradient fields for molecular conformation generation. In International Conference on Machine Learning, pages 9558–9568. PMLR, 2021.   
[52] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015.   
[53] T. Sommer, R. Willa, J. Schmalian, and P. Friederich. 3dsc-a dataset of superconductors including crystal structures. Scientific Data, 10(1):816, 2023.   
[54] H. Takahashi, N. Môri, M. Azuma, Z. Hiroi, and M. Takano. Effect of pressure on tc of hole-and electron-doped infinite-layer compounds up to 8 gpa. Physica C: Superconductivity, 227(3-4): 395–398, 1994.

[55] B. L. Trippe, J. Yim, D. Tischer, D. Baker, T. Broderick, R. Barzilay, and T. S. Jaakkola. Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=6TxBxqNME1Y.   
[56] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
[57] C. Wang and W. Pickett. Density-functional theory of excitation spectra of semiconductors: application to si. Physical review letters, 51(7):597, 1983.   
[58] Y. Watanabe, A. Miura, C. Moriyoshi, A. Yamashita, and Y. Mizuguchi. Observation of superconductivity and enhanced upper critical field of $\eta$ -carbide-type oxide zr4pd2o. Scientific Reports, 13(1):22458, 2023.   
[59] M. Weiler, M. Geiger, M. Welling, W. Boomsma, and T. Cohen. 3d steerable cnns: Learning rotationally equivariant features in volumetric data. In S. Bengio, H. M. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada, pages 10402–10413, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/488e4104520c6aab692863cc1dba45af-Abstract.html.   
[60] D. Wines, T. Xie, and K. Choudhary. Inverse design of next-generation superconductors using data-driven deep generative models, 2023.   
[61] T. Xie and J. C. Grossman. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical Review Letters, 120(14):145301, 2018. doi: 10.1103/PhysRevLett.120.145301. URL https://link.aps.org/doi/10.1103/PhysRevLett.120.145301.   
[62] T. Xie, X. Fu, O. Ganea, R. Barzilay, and T. S. Jaakkola. Crystal diffusion variational autoencoder for periodic material generation. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=03RLpj-tc\_.   
[63] M. Xu, L. Yu, Y. Song, C. Shi, S. Ermon, and J. Tang. Geodiff: A geometric diffusion model for molecular conformation generation. In International Conference on Learning Representations, 2021.   
[64] K. Yan, Y. Liu, Y. Lin, and S. Ji. Periodic graph transformers for crystal material property prediction. In The 36th Annual Conference on Neural Information Processing Systems, 2022.   
[65] K. Yan, Y. Liu, Y.-C. Lin, and S. Ji. Periodic graph transformers for crystal material property prediction. ArXiv, abs/2209.11807, 2022.   
[66] C. Zhang, X. He, C. Liu, Z. Li, K. Lu, S. Zhang, S. Feng, X. Wang, Y. Peng, Y. Long, et al. Record high t c element superconductivity achieved in titanium. Nature Communications, 13(1):5411, 2022.   
[67] S. Zhang, Y. Wang, J. Zhang, H. Liu, X. Zhong, H.-F. Song, G. Yang, L. Zhang, and Y. Ma. Phase diagram and high-temperature superconductivity of compressed selenium hydrides. Scientific reports, 5(1):15433, 2015.

# Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]

(b) Did you describe the limitations of your work? [Yes]   
(c) Did you discuss any potential negative societal impacts of your work? [N/A]   
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [Yes]   
(b) Did you include complete proofs of all theoretical results? [Yes]

3. If you ran experiments (e.g. for benchmarks)...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes]   
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes]   
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes]   
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]   
(b) Did you mention the license of the assets? [Yes]   
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes]   
(d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [Yes]   
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]   
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]   
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

The appendix is organized as follows: Section A details the collection method and distribution of the SuperCon3D dataset. Section B presents more details of disordered graph and models. Section C elaborates on the implementation specifics of both property prediction and generative models. We present the evaluation metrics and experimental results for novel structure generation in Section D. The identification of potential superconductors and their corresponding DFT computational outcomes are presented in Section E. A systematic approach for the design of practical superconductors is expounded in Section F. Section G provides the repository link for the associated coding resources.

# A SuperCon3D Data Details

We extracted approximately 33,000 superconductors, including their chemical formulas and critical temperatures, from the SuperCon database $^{2}$ . After eliminating duplicates and non-superconductors, we retained 11,949 superconducting materials. Over 200,000 ordered and disordered crystal structures were collected from the ICSD database. We then matched the 11,949 SuperCon entries with 208,425 ICSD entries based on chemical composition, space group and lattice parameter. Specifically, we first performed an initial matching based on chemical composition, which may result in one-to-one or one-to-many matches. We then further refined the matches using additional information provided in the literature, such as space groups and lattice constants. Additionally, $T_{c}$ values and structural data for hydrogen-enriched superconductors were obtained from literature sources. Ultimately, we compiled 1,578 superconductors with both $T_{c}$ and crystal structure information.

![](images/59bad88330b283e6e98a8d9eaedbd1d3e61091c673ae4b75905d18a8f052528f.jpg)  
Figure 4: The data distribution of SuperCon3D dataset. (a). The probability of crystals containing a given element in the dataset. (b). The distribution of ordered and disordered superconductors. (c). The distribution of superconducting types. (d). The distribution of $T_{c}$ values.

We plot the data distribution of SuperCon3D dataset in Fig. 4. In dataset, there are 83 different elements, which encompass most of elemental types found in the periodic table. The most frequent elements are O, Cu, La, Ba, Y as shown in Fig. 4a. Fig. 4b depicts the order and disorder distribution. We classify superconducting materials according to cuprate, H riched, heavy fermion, iron based,

and others, and distribute the types in Fig. 4c. The distribution of the $T_{c}$ values of superconducting materials is shown in Fig. 4d. The SuperCon3D dataset can be obtained from the source code package, and the access address is provide in Sec. G.

# B Methods

# B.1 Interstitial Disorder

Interstitial disorder (ID). ID refers to the presence of atoms occupying interstitial sites within a crystal lattice, which are not part of the regular lattice positions. These interstitial atoms introduce additional disorder into the structure. The total occupancy, including both regular lattice sites and interstitial sites, can be expressed as:

$$
\boldsymbol {w} _ {i, 1} + \boldsymbol {w} _ {i, 2} + \dots + \boldsymbol {w} _ {i, m _ {i}} + \boldsymbol {w} _ {i, i n t e r s t i t i a l} = 1 + \Delta \tag {19}
$$

where $w_{i,m_{i}}$ represents the occupancy weight of $m_{i}$ at site i, $w_{i,interstitial}$ represents the occupancy weight of interstitial atoms at site i, $\Delta$ is the excess occupancy due to interstitial atoms, with $\Delta > 0$ indicating the presence of ID. In this case, our disordered graph encoding method remains effective.

ID mixed with substitutional disorder (SD) and positional disorder (PD), would result in more new types. However, given the lack of observation of ID in the SuperCon3D dataset, we will not elaborate further on it.

# B.2 Gate layer

We employ the gate activation mechanism [59] for the equivariant activation function. Standard activation functions are applied to type-0 vectors. For higher order vectors (L > 0), we achieve equivariance by multiplying them with non-linearly transformed type-0 vectors. Specifically, for an input $x$ comprising non-scalar $C_L$ type-L vectors (where $0 < L \leq L_{\mathrm{max}}$ ) and $(C_0 + P_L \sum_{L=1}^{L_{\mathrm{max}}} C_L)$ type-0 vectors, we apply SiLU [14] to the first $C_0$ type-0 vectors and a sigmoid function to the remaining $P_L \sum_{L=1}^{L_{\mathrm{max}}} C_L$ type-0 vectors. This process generates non-linear weights, which are then used to scale each type-L vector. After gate activation, the number of channels for type-0 vectors is reduced to $C_0$ .

# B.3 The Denoising Method of DiffCSP

We introduce the denoising model $\phi(L, F, A, t)$ as part of the original DiffCSP model, which is related to the Transformer-based Architecture section in the main text.

Node representations in the s-th layer, $\boldsymbol{H}^{(s)} = [\boldsymbol{h}_{1}^{(s)}, \cdots, \boldsymbol{h}_{N}^{(s)}]$ , are initialized as $\boldsymbol{h}_{i}^{(0)} = \psi(f_{\mathrm{atom}}(\boldsymbol{a}_{i}), f_{\mathrm{pos}}(t))$ , combining atomic embeddings $f_{atom}$ and sinusoidal positional encoding $f_{pos}$ [24, 56], processed by MLP $\psi$ .

Incorporating EGNN [48], the message-passing in layer s is:

$$
\boldsymbol {m} _ {i j} ^ {(s)} = \varphi_ {m} (\boldsymbol {h} _ {i} ^ {(s - 1)}, \boldsymbol {h} _ {j} ^ {(s - 1)}, \boldsymbol {L} ^ {\top} \boldsymbol {L}, \psi_ {\mathrm{FT}} (\boldsymbol {f} _ {j} - \boldsymbol {f} _ {i})), \tag {20}
$$

$$
\boldsymbol {m} _ {i} ^ {(s)} = \sum_ {j = 1} ^ {N} \boldsymbol {m} _ {i j} ^ {(s)}, \tag {21}
$$

$$
\boldsymbol {h} _ {i} ^ {(s)} = \boldsymbol {h} _ {i} ^ {(s - 1)} + \varphi_ {h} (\boldsymbol {h} _ {i} ^ {(s - 1)}, \boldsymbol {m} _ {i} ^ {(s)}). \tag {22}
$$

Here, $\varphi_{m}$ and $\varphi_{h}$ are MLPs. $\psi_{\mathrm{FT}}$ executes Fourier Transformation on relative fractional coordinates, ensuring periodic translation invariance.

Following $S$ message-passing layers, lattice noise $\hat{\epsilon}_{L}$ is computed as follows:

$$
\hat {\epsilon} \boldsymbol {L} = \boldsymbol {L} \varphi_ {L} \left(\frac {1}{N} \sum i = 1 ^ {N} \boldsymbol {h} _ {i} ^ {(S)}\right), \tag {23}
$$

with $\varphi_{L}$ shaping output as $3 \times 3$ . For fractional coordinate score $\hat{\epsilon}_{F}$ , we have:

$$
\hat {\epsilon} _ {F} [:, i ] = \varphi_ {F} (\boldsymbol {h} _ {i} ^ {(S)}), \tag {24}
$$

where $\hat{\epsilon}_{F}[:,i]$ is the $i$ -th column, and $\varphi_{F}$ operates on the final layer's output.

The inner product $L^{\top}L$ in Eq.(20) ensures O(3)-invariance, as $(\boldsymbol{QL})^{\top}(\boldsymbol{QL}) = \boldsymbol{L}^{\top}\boldsymbol{L}$ for any orthogonal $Q \in R^{3 \times 3}$ . This guarantees the O(3)-invariance of $\varphi_{L}$ in Eq.(24), and L left-multiplied with $\varphi_{L}$ ensures O(3)-equivariance of $\hat{\epsilon}_{L}$ . Thus, $\phi(\boldsymbol{L}, \boldsymbol{F}, \boldsymbol{A}, t)$ satisfies the proposed properties. More details are described in Jiao et al. [27].

# C Hyper-parameters and Training Details

In this section, we provide the training details of property predicting models and generative models.

# C.1 Property Predicting Models

We employ the codebase from RF and SVM $[53]^{3}$ , SchNet $[50]^{4}$ , CGCNN $[61]^{5}$ , DimNet++ $[16]^{6}$ , SphereNet $[32]^{7}$ , ALIGNN $[11]^{8}$ , Matformer $[65]^{9}$ and MEGNet $[8]^{10}$ for baseline implementations. All models are conducted 10-fold experiments based data split method of 8:1:1. The training details of each model are as follows:

# C.1.1 RF and SVM.

We conducted experiments comparing non-deep learning methods, specifically RF and SVM, using both chemical composition features and combined geometric structure features. The SVM model is configured with the following parameters: kernel is set to 'rbf' for mapping data into a higher-dimensional space, while degree, set to 3, controls the complexity of the polynomial kernel (applicable only when a polynomial kernel is used). gamma, set to 'scale', adjusts the influence of individual data points in the feature space. The RF model uses these parameters: n\_estimators is set to 100, defining the number of submodels in the ensemble. criterion, set to 'squared\_error', evaluates split quality based on mean squared error. min\_samples\_split, set to 2, specifies the minimum number of samples required to split an internal node, and min\_samples\_leaf, set to 1, defines the minimum samples needed at a leaf node. Lastly, max\_features, set to 1.0, determines the proportion of features considered when finding the best split.

# C.1.2 SchNet.

Employing the SchNet framework, our method integrates six 64-dimensional message passing layers. SchNet was trained over 500 epochs, using a 5e-4 learning rate and 64 batch size. We optimized using Adam with 1e-5 weight decay, and a one-cycle learning rate scheduler. Atomic radii were determined by the 12th smallest distance between an atom and its neighbors.

# C.1.3 CGCNN.

A batch size of 64 is employed, and the model consists of three layers of CGCNN message passing layer with 128 hidden dimensions. The training process utilizes the Adam optimizer. Initially, a learning rate of 1e-3 is set for the 200 epochs. During the training, a radius cutoff of 8.0 is applied to all crystals, and the 32 nearest neighbors are selected.

# C.1.4 DimNet++.

In our approach, we apply a radius cutoff of 8.0 to all crystals and select the 12 nearest neighbors. To represent each node, we utilize Gaussian radial basis function (RBF) kernels. This results in a 64-dimensional embedding for each node. To optimize the model, we employ the Adam optimizer with a weight decay of 1e-6. The model is trained for 500 epochs using a batch size of 128.

# C.1.5 SphereNet.

In our method, we utilize multi-graph representations of materials as inputs to SphereNet models. The input embedding size is set to 256, and the output embedding size is set to 64 for both the 8 LB2 and LB blocks. A cutoff distance of 6 is used. For each model, we initially perform a warm-up on the learning rate, starting at 1e-3. Subsequently, two learning rate strategies—ReduceLROnPlateau and StepLR—are employed for training. In the StepLR strategy, the learning rate is decayed by a specified ratio every fixed number of epochs, known as the step size. The batch size is set to 32, and training is conducted for 300 epochs.

# C.1.6 ALIGNN.

ALIGNN is trained for 150 epochs with a learning rate of 5e-4 and a batch size of 64. The model architecture follows the original paper, consisting of four GCN layers and four ALIGNN layers. The atom feature dimension is set to 92, and the edge feature dimension is set to 80. The training process utilize the Adam optimizer with a weight decay of 1e-5. Additionally, a one-cycle learning rate scheduler is employed. For all crystals, a radius cutoff of 8.0 is applied, and the nearest 12 neighbors are selected.

# C.1.7 Matformer.

In constructing the crystal graph, we follow a specific procedure. The radius for the neighborhood of a given atom is determined by the 12-th smallest distance between that atom and its neighboring atoms. All atoms within this radius are considered part of the neighborhood for the given atom. Each node is then represented by mapping its atomic number to a 92-dimensional embedding using the CGCNN atomic embedding. This embedding is further transformed into a 128-dimensional vector through a linear transformation. Similarly, for each edge, we utilize a 128-dimensional embedding mapping of the Euclidean distance. This mapping is achieved by employing 128 radial basis function (RBF) kernels with centers ranging from 0.0 to 8.0. During the training process, we employ the Adam optimizer with a weight decay of 1e-5. Additionally, a one-cycle learning rate scheduler is utilized. A batch size of 64 is employed and trained for 150 epochs.

# C.1.8 MEGNet.

To construct the crystal graph, we employ three layers of the MEGNET message passing with with 64,32,16 hidden units, and utilize the Set2Set readout function. Following the configuration described in the original paper, MEGNET is trained for 200 epochs using a batch size of 64 and a learning rate of 1e-3. The Adam optimizer with a weight decay of 1e-5 is used for optimization, and a one-cycle learning rate scheduler is implemented. A radius of 8.0 is set for all crystals.

Table 7: Hyper-parameters for SODNet. 

<table><tr><td>Hyper-parameters</td><td>Value or description</td></tr><tr><td>Batch size</td><td>32, 64, 128</td></tr><tr><td>Number of epochs</td><td>150, 300</td></tr><tr><td>Number of attention heads</td><td>4, 8</td></tr><tr><td>Dropout rate</td><td>0.0, 0.1, 0.2</td></tr><tr><td>Cutoff radius (Å)</td><td>8, 12, 16</td></tr><tr><td>Number of radial bases</td><td>128</td></tr><tr><td>Number of transformer blocks</td><td>6</td></tr><tr><td>Weigh decay</td><td> $0.5 \times 10^{-3}$ ,  $1 \times 10^{-3}$ </td></tr></table>

Table 8: Recently discovered superconductors (not included in the training data). 

<table><tr><td>No.</td><td>Material</td><td>Type  $T_{c}^{exp}$  (K)</td></tr><tr><td>1</td><td> $CaH_{6}$  @172 GPa</td><td>Order 215 [35]</td></tr><tr><td>2</td><td>Ti @248 GPa</td><td>Order 26 [66]</td></tr><tr><td>3</td><td> $CsV_{3}Sb_{5}$ </td><td>Order 2.3 [18]</td></tr><tr><td>4</td><td> $Cs(V_{0.93}Nb_{0.07})_{3}Sb_{5}$ </td><td>SD 4.45 [29]</td></tr><tr><td>5</td><td> $Zr_{4}Rh_{2}O$ </td><td>Order 3.73 [58]</td></tr><tr><td>6</td><td>Zr4Pd2O</td><td>Order 2.73 [58]</td></tr><tr><td>7</td><td> $LaFeSiO_{0.9}$ </td><td>PD 10 [23]</td></tr></table>

# C.1.9 SODNet

During training, we use a batch size of 64 and trained the model for 150 epochs. A radius of 8.0 is applied to define the neighborhood of each crystal. We utilize 128 basis functions to capture the features of the crystals. To control overfitting, a weight decay of 5e-3 is applied. The learning rate is set to 5e-5, with a minimum learning rate of 1e-6. We employ the AdamW optimizer for efficient optimization. The model architecture consisted of 6 Transformer blocks, each with 8 attention heads. This allowed the model to effectively capture the relationships and dependencies within the crystal structures. Irreps features consist of channels of vectors with degrees up to $L_{\mathrm{max}}$ . We denote $C_L$ type- $L$ vectors as $(C_L, L)$ and $C_{(L,p)}$ type- $(L, p)$ vectors as $(C_{(L,p)}, L, p)$ . Brackets denote concatenations of vectors. We set irreps features containing 512 type-0 vectors and 128 type-1 vectors, which can be expressed as [(512, 0), (128, 1)]. Table 7 summarizes the hyper-parameters for the model.

# C.2 Generative Models

We apply the codebases from CDVAE $[62]^{11}$ , SyMat $[34]^{12}$ and DiffCSP $[27]^{13}$ for baseline implementations. All models are conducted experiments based data split method of 6:2:2. For pretraining, we obtain crystal structures from the databases of Materials Project $^{14}$ , Open Quantum Materials Database $^{15}$ , Matgen $^{16}$ , and ICSD $^{17}$ . Molecular crystals are excluded from the dataset. Subsequently, we perform deduplication on all crystal structures, resulting in approximately 1.14 million unique structures. The training specifics for each model are outlined below:

# C.2.1 CDVAE.

For CDVAE model, We replaced the original DimNet++ [16] with SODNet to ensure a fair comparison with other generation models. Regarding the decoder, we utilize the GemNet-T [17], which consists of 3 layers and 128 hidden states.

# C.2.2 SyMat.

For the SyMat model, the property predictor employs SphereNet, which consists of four message-passing layers with a hidden size of 128. The VAE decoder utilizes MLP models composed of two linear layers with a ReLU activation function between them and a hidden size of 256. During training, we use a learning rate of 0.001, a batch size of 128, and run for 1,000 epochs. We assign different weights to various loss terms: 1.0 for atom type set size, 30.0 for atom types, 1.0 for the number of each atom type, and 10.0 for lattice items. Additionally, we apply a weight of 0.01 for the KL-divergence loss and 10.0 for the denoising score matching loss.

# C.2.3 DiffCSP.

We employ a configuration of 6 layers with 512 hidden states for datasets other than specified ones. The dimension of the Fourier embedding is set to 256. To control the variance of the DDPM (Diffusion-Driven Probabilistic Modeling) process on $L_{t}$ , we utilize the cosine scheduler with 0.008. Additionally, we use an exponential scheduler with $\sigma_{1}=0.005$ , $\sigma_{T}=0.5$ to control the noise scale of the score matching process on $F_{t}$ . The diffusion step is set to 1000. Our model is trained for 1000 epochs, employing the same optimizer and learning rate scheduler as CDVAE.

# C.2.4 DiffCSP-SC.

We utilize SODNet as the property predictor, and the parameter configuration aligns with Table 7. The parameters for the diffusion process also follow the original DiffCSP setup. The difference lies in the message passing layer, where we employ a transformer. Specifically, we use a 512-dimensional hidden state encoding and set the number of heads to 8.

# C.3 Pre-training Dataset

we pre-trained our model on approximately 1.14 million unique 3D crystals sourced from existing databases, including Materials Project, OQMD, Matgen and ICSD.

# D Novel Material Structure Generation

we have introduced a new metric, the "novelty success rate" (NSR), to specifically quantify the proportion of novel structures generated by the model. The NSR is defined as:

$$
N S R _ {\alpha} (\tilde {D}) = \frac {\left\| \tilde {M} \mid \tilde {M} \in \tilde {D} , \varphi (\tilde {M}) > P _ {1 0 0 - \alpha} (D _ {\text {train}}) , \tilde {M} \notin D _ {\text {train}} \right\|}{\| \tilde {D} \|} \tag {25}
$$

This metric focuses on evaluating the model's ability to generate structures that are not present in the training dataset.

We conducted additional experiments using NSR, and the results are summarized in the table below, comparing different models:

Table 9: NSR comparison across different models and data settings 

<table><tr><td>Model</td><td>Data</td><td>NSR10</td><td>NSR30</td><td>NSR50</td></tr><tr><td>CDVAE</td><td>O</td><td>0.02</td><td>0.02</td><td>0.02</td></tr><tr><td>SyMat</td><td>O</td><td>0.02</td><td>0.03</td><td>0.03</td></tr><tr><td>DiffCSP</td><td>O</td><td>0.03</td><td>0.04</td><td>0.04</td></tr><tr><td>DiffCSP-SC</td><td>O</td><td>0.04</td><td>0.04</td><td>0.09</td></tr><tr><td>CDVAE</td><td>Pre-training + O</td><td>0.19</td><td>0.19</td><td>0.25</td></tr><tr><td>SyMat</td><td>Pre-training + O</td><td>0.20</td><td>0.21</td><td>0.26</td></tr><tr><td>DiffCSP</td><td>Pre-training + O</td><td>0.25</td><td>0.25</td><td>0.33</td></tr><tr><td>DiffCSP-SC</td><td>Pre-training + O</td><td>0.31</td><td>0.31</td><td>0.39</td></tr></table>

As shown in table 9, our DiffCSP-SC model outperforms others in generating novel materials, as indicated by higher NSR values across all metrics (NSR10, NSR30, NSR50).

Furthermore, our training dataset includes approximately 1 million material structures, many of which have not been experimentally validated for superconductivity. Even if some generated structures appear in the training data, they may still hold potential superconducting properties, making them valuable for further investigation. By leveraging this large dataset and pre-training strategies, our model demonstrates advantages in generating novel and potentially superconductive structures.

Table 10: The predicted potential candidates of high- $T_{c}$ cuprate and h-rich superconductors. Candidates of high confidence are marked in gray. 

<table><tr><td>Type</td><td>ICSD code</td><td>Chemical formula</td><td colspan="2">O/SD/PDTc(K)</td><td>Reported SC.</td></tr><tr><td rowspan="5">Cuprate</td><td>68675</td><td> $CuO_{2}Sr_{0.075}$ </td><td>PD</td><td>93.42</td><td> $CuO_{2}Sr_{91K}$ [54]</td></tr><tr><td>50774</td><td> $Ca_{0.779}CuO_{2}Y_{0.041}$ </td><td>PD</td><td>65.70</td><td></td></tr><tr><td>50773</td><td> $Ca_{0.82}CuO_{2}$ </td><td>PD</td><td>64.72</td><td> $CaCuO_{2}$ 89K [49]</td></tr><tr><td>68217</td><td> $Ba_{2}CuO_{3}$ </td><td>O</td><td>59.89</td><td> $Ba_{2}CuO_{3.2}$ 70K [28]</td></tr><tr><td>67394</td><td> $Ba_{2}CuIO_{2}$ </td><td>O</td><td>43.80</td><td>-</td></tr><tr><td rowspan="7">H-riched</td><td>187375</td><td> $ErH_{3}$ </td><td>O</td><td>193.03</td><td>-</td></tr><tr><td>635802</td><td> $GdH_{3}$ </td><td>O</td><td>143.19</td><td>-</td></tr><tr><td>623739</td><td> $H_{2.57}Co_{0.14}U_{0.84}$ </td><td>PD</td><td>136.76</td><td>-</td></tr><tr><td>42009</td><td> $TbH_{2.25}$ </td><td>SD</td><td>135.13</td><td>-</td></tr><tr><td>424154</td><td> $H_{6}Mg_{1.02}Ti_{1.98}$ </td><td>O</td><td>134.34</td><td>-</td></tr><tr><td>230140</td><td> $Li_{0.14}Y_{0.86}H_{2.7}$ </td><td>PD</td><td>125.94</td><td>-</td></tr><tr><td>93250</td><td> $YFe_{2}H_{5}$ </td><td>PD</td><td>125.00</td><td>-</td></tr></table>

# E Potential Superconductors

In this section, we initially validate our model using the $T_{c}$ values of superconducting materials reported in recent literature, noting that these data points are not included in the SuperCon3D dataset. Subsequently, we present the potential superconducting materials using property prediction model based on SODNet and generative model based on DiffCSP-SC, respectively.

Table 11: The predicted potential candidates of high- $T_{c}$ heavy-fermion, iron-based and others superconductors. Candidates of high confidence are marked in gray. 

<table><tr><td>Type</td><td>ICSD code</td><td>Chemical formula</td><td colspan="2">O/SD/PDTc(K)</td><td>Reported SC.</td></tr><tr><td rowspan="5">Heavy-Fermion</td><td>168466</td><td> $LaMg_{12}$ </td><td>O</td><td>23.83</td><td>-</td></tr><tr><td>161141</td><td> $LaMg_{11.196}Al_{0.804}$ </td><td>SD</td><td>21.13</td><td>-</td></tr><tr><td>69897</td><td> $C_2Ce_{0.75}U_{0.25}$ </td><td>PD</td><td>11.88</td><td>-</td></tr><tr><td>647197</td><td> $Np_{1.1}Pu_{0.9}$ </td><td>SD</td><td>11.75</td><td>-</td></tr><tr><td>614236</td><td> $TmFe_4B$ </td><td>O</td><td>10.81</td><td>-</td></tr><tr><td rowspan="5">Iron-based</td><td>427163</td><td> $Ba_{0.83}Fe_2Rb_{0.17}As_2$ </td><td>SD</td><td>23.21</td><td> $Ba_{0.6}Fe_2Rb_{0.4}As_2$ 37.5K [42]</td></tr><tr><td>188347</td><td> $BaFe_2As_2$ </td><td>O</td><td>23.27</td><td>-</td></tr><tr><td>39530</td><td> $FeCl_7Te$ </td><td>O</td><td>19.57</td><td>-</td></tr><tr><td>633401</td><td> $FeSb_{0.4}Te_{1.6}$ </td><td>SD</td><td>16.83</td><td>-</td></tr><tr><td>165523</td><td> $As_2Ba_{0.777}Fe_2-K_{0.126}Sn_{0.096}$ </td><td>PD</td><td>15.55</td><td>-</td></tr><tr><td rowspan="5">Others</td><td>96031</td><td> $Ba_{1.1432}Co_{0.1429}-O_{3.0009}Rh_{0.8574}$ </td><td>PD</td><td>202.12</td><td>-</td></tr><tr><td>58639</td><td> $Ba_{0.515}Ca_{0.485}$ </td><td>SD</td><td>160.95</td><td>-</td></tr><tr><td>616160</td><td> $BaSr$ </td><td>SD</td><td>123.51</td><td>-</td></tr><tr><td>106111</td><td> $SrTl_2$ </td><td>O</td><td>63.52</td><td>-</td></tr><tr><td>428028</td><td> $Ge_{0.6}Sb_{0.27}Te$ </td><td>SD</td><td>47.48</td><td>-</td></tr></table>

# E.1 Real-world Superconductors Validation

As shown in Table 8, we have collected the structures of superconducting materials along with their corresponding $T_{c}$ values, as reported in the latest literature over the past three years. This includes a total of seven superconducting materials with both ordered and disordered structures.

# E.2 Screening Based Method

We apply our superconductivity predicting model for screening the entire ICSD database. Potential superconductors are show in Table 10 and 11. To elaborate on the candidates with high confidence, we provide the subsequent details:

1. $CuO_{2}Sr_{0.075}$ and $Ca_{0.82}CuO_{2}$ exhibit disordered structures, and their respective parent compounds demonstrate superconductivity [54, 49]. Consequently, these disordered structures are more likely to be superconducting materials as well.

2. $\mathrm{Ba}_{2}\mathrm{CuO}_{3.2}$ exhibits superconductivity with a $\mathrm{T}_c$ of 70K [28]. Its corresponding parent structure $\mathrm{Ba}_{2}\mathrm{CuO}_{3}$ may also be a superconductor, albeit with a comparatively lower probability.   
3. $\mathrm{Ba_{0.83}Fe_2Rb_{0.17}As_2}$ and $\mathrm{Ba_{0.6}Fe_2Rb_{0.4}As_2}$ share the same parent structure and have closely related compositions. Given that $\mathrm{Ba_{0.6}Fe_2Rb_{0.4}As_2}$ exhibits superconductivity with a $\mathrm{T_c}$ of 37.5K [42], it is highly likely that $\mathrm{Ba_{0.83}Fe_2Rb_{0.17}As_2}$ is also a superconducting material.

Table 12: The novel high- $T_{c}$ cuprate and h-rich superconducting candidates. Candidates of high confidence are marked in gray. 

<table><tr><td>Type</td><td>Index</td><td>Chemical formula</td><td> $T_c$  (K)</td><td>Reported SC.</td></tr><tr><td rowspan="7">Cuprate</td><td>1</td><td> $Ba_2CuCl_2O_2$ </td><td>33.56</td><td>-</td></tr><tr><td>2</td><td> $Tl_2Ca_2Ba_2Cu_3O_{10}$ </td><td>14.09</td><td>-</td></tr><tr><td>3</td><td> $Ba_3CaLa_2GdCu_7O_{17}$ </td><td>10.12</td><td>-</td></tr><tr><td>4</td><td> $YCu_3O_7$ </td><td>9.73</td><td>-</td></tr><tr><td>5</td><td> $BaCaCu_3O_7$ </td><td>9.65</td><td>-</td></tr><tr><td>6</td><td> $Cu_7BO_{16}$ </td><td>7.87</td><td>-</td></tr><tr><td>7</td><td> $CsMgCu_3BiAuO_8$ </td><td>7.82</td><td>-</td></tr><tr><td rowspan="7">H-rich</td><td>8</td><td> $TbH_3$ </td><td>164.33</td><td> $TbH_3$ 20K [22]Calculated by DFT</td></tr><tr><td>9</td><td> $SeH_3$ </td><td>139.89</td><td> $SeH_3$ 113K [40]Predicted by ML</td></tr><tr><td>10</td><td> $CaGe_2H_9$ </td><td>103.55</td><td>-</td></tr><tr><td>11</td><td> $Ca_2MnCrH_6$ </td><td>58.07</td><td>-</td></tr><tr><td>12</td><td> $SbH_3$ </td><td>46.42</td><td> $SbH_3$ 20K [15]Calculated by DFT</td></tr><tr><td>13</td><td> $MgCoCuH_{42}CS_2N_{16}$ </td><td>44.27</td><td>-</td></tr><tr><td>14</td><td> $Rb_2Ca_2H_4$ </td><td>13.05</td><td>-</td></tr></table>

# E.3 Interpretability on SODNet

We attempt to interpret our SODNet predictor by determining which feature(s) a given model weighs most heavily when making the prediction. As shown in Fig. 5, we extract the node embedding of the whole graph in the last layer of SODNet, and present the contributions of each atom to $T_{c}$ values. We can observe that the B sites contributes more significantly to the property of $T_{c}$ compared to the Mg site in Fig. 5 (a-d). Moreover, conducting atomic doping and atomic translation on the cation Mg results in a decrease in $T_{c}$ with $39.0\ K \rightarrow 38.4\ K \rightarrow 34.3\ K$ . This phenomenon demonstrates that attempting to enhance the $T_{c}$ value by disrupting the symmetry of Mg site within the lattice may be not workable. Another case of cuprate superconductor has shown in Fig. 5 (e-f), there are three types of oxygen sites that contribute significantly to the $T_{c}$ value: Hg-O-Hg (PD disorder), Cu-O-Cu (order), and Hg-O-Ba (order). Among them, the contribution of disordered Hg-O-Hg is the greatest, indicating that disrupting the symmetry of oxygen atoms within the lattice might potentially further enhance the property of $T_{c}$ .

# E.4 Generative Superconducting Candidates

We apply our generative model for generating new superconducting candidates. We present the crystal structures of the 20 superconducting candidate materials from Table 6 in Fig. 6. Additionally, we display the 32 superconducting candidate materials in Table 12 and 13, arranged in descending order of predicted $T_{c}$ values. The structures of all superconducting candidate materials can be obtained in the source code package. We collected superconducting materials that have been reported and observed that five candidates are more likely to be superconducting materials. Among them, four candidates obtained $T_{c}$ through theoretical calculations, and another material displayed superconducting properties through doping. Specific descriptions are as follows:

1. $\mathrm{SeH}_3$ exhibited a $\mathrm{T}_c$ of 113K as predicted by machine learning [40], corroborated by DFT calculations indicating 110K [67].   
2. DFT methods calculated the $T_{c}$ of $TbH_{3}$ , $SbH_{3}$ , and $KFe_{2}As_{2}$ as 20K [22, 15, 43]. Since H-riched materials belong to conventional superconductors and show high $T_{c}$ under high pressure, but the conditions for wet experimental synthesis are very stringent. Therefore,

![](images/1a5b51e306c0fc2561c33aa51bcebdb215f785d4db2488f2890432a2ae0b967f.jpg)

<details>
<summary>bar</summary>

| Site | SD | PD | Order |
|---|---|---|---|
| B | 0.42 | 0.43 | 0.43 |
| B | 0.42 | 0.43 | 0.43 |
| Mg | 0.15 | 0.13 | 0.12 |
</details>

(a)

![](images/a0c047be43097682920d74e9d4e6faf13af9669cc6074692d53cbbd4e77ab7da.jpg)

<details>
<summary>bar</summary>

| Category | Relative importance |
|---|---|
| Hg-O-Hg | 28 |
| Cu-O-Cu | 25 |
| Cu-O-Cu | 25 |
| Hg-O-Ba | 13 |
| Hg-O-Ba | 13 |
| Hg | 5 |
| Cu | 3 |
| Ba | 0 |
| Ba | 0 |
</details>

(e)

![](images/25def101ca979a6d9005b932490cc6ed479c3674d96230c54d51d156f889cf9a.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing interconnected atoms with green and pink spheres representing different elements
</details>

(b)

![](images/eacd50cf508e0cd432bb0c427c095a0a705bd9311ce4ab8501fb5471ae5a9b64.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a central atom bonded to multiple surrounding atoms, with green and pink spheres representing different elements.
</details>

(c)

![](images/17fff65b8999c479bf1aab3f8a61faf335698499840c9ba0de7bfaba6a041348.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing interconnected atoms with green and pink spheres representing different elements
</details>

(d)

![](images/0a191ad3269b8132162197658feb9e93e0668bd77ea2d670b74f3e4178273f71.jpg)

<details>
<summary>chemical</summary>

Crystal lattice structure diagram showing green, red, and blue atoms in a cubic arrangement
</details>

(f)   
Figure 5: Contribution of each atom to $T_{c}$ value. (a). Feature relative importance of each site in three type $MgB_{2}$ superconductors. Snapshots of (b) SD $Mg_{0.9}Al_{0.1}B_{2}$ , (c) PD $Mg_{0.98}B_{2}$ , and (d) ordered $MgB_{2}$ crystals. Here, B Mg and Al sites are colored by light pink, atrovirens and dark atrovirens. (e). Feature relative importance of $Ba_{2}CuHgO_{4.27}$ superconductor. (f). Snapshot of $Ba_{2}CuHgO_{4.27}$ superconductor (Ba: green, Cu: blue, Hg: pink, O: red).

it can further verify whether superconducting materials are superconducting materials by combining DFT methods, and reduce the research and development cycle of superconducting materials.

3. The parent compound SmFeAsO underwent a superconducting $T_{c}$ around 54 K [1], following fluorine (F) doping at the O-site in the SmO layer. This case can provide us with a method that we can use DiffCSP-SC's generative model to generate superconducting parent structures, and then improve the $T_{c}$ of materials by doping, or transform materials without superconducting properties into superconducting materials.

Table 13: The novel high- $T_{c}$ heavy-fermion, iron-based and others superconducting candidates. Candidates of high confidence are marked in gray. 

<table><tr><td>Type</td><td>Index</td><td>Chemical formula</td><td> $T_c$ (K)</td><td>Reported SC.</td></tr><tr><td rowspan="5">Heavy-Fermion</td><td>15</td><td>Th</td><td>43.61</td><td>-</td></tr><tr><td>16</td><td> $Ba_3Pu$ </td><td>44.81</td><td>-</td></tr><tr><td>17</td><td> $ThC_3$ </td><td>17.96</td><td>-</td></tr><tr><td>18</td><td>Lu</td><td>4.86</td><td>-</td></tr><tr><td>19</td><td>Yb3In</td><td>1.04</td><td>-</td></tr><tr><td rowspan="5">Iron-based</td><td>20</td><td> $BaFe_2Se_2$ </td><td>11.99</td><td>-</td></tr><tr><td>21</td><td>SmFeAsO</td><td>4.42</td><td> $SmFeAsO_{0.8}F0.254K$ [1]</td></tr><tr><td>22</td><td> $KFe_2As_2$ </td><td>4.23</td><td> $KFe_2As_2@30GPa20K$ [43]Calculated by DFT</td></tr><tr><td>23</td><td>NdFeAsF</td><td>4.13</td><td>-</td></tr><tr><td>24</td><td>FeSe</td><td>3.36</td><td>-</td></tr><tr><td rowspan="8">Others</td><td>25</td><td> $Ba_3Ca$ </td><td>80.04</td><td>-</td></tr><tr><td>26</td><td> $Ba_2Se$ </td><td>60.70</td><td>-</td></tr><tr><td>27</td><td>Ba</td><td>52.26</td><td>-</td></tr><tr><td>28</td><td> $Mg_3B$ </td><td>43.96</td><td>-</td></tr><tr><td>29</td><td> $BaCl_2O$ </td><td>35.72</td><td>-</td></tr><tr><td>30</td><td> $Ba_2CaB$ </td><td>32.77</td><td>-</td></tr><tr><td>31</td><td> $Sb_2Ba_4$ </td><td>22.70</td><td>-</td></tr><tr><td>32</td><td> $V_3Si_{11}$ </td><td>16.28</td><td>-</td></tr></table>

![](images/87d07006cd4c4574a38c00b2021bdfd81f54eaf1a492cecab738cbc7fcfe0df7.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing green, red, and blue atoms in a lattice arrangement
</details>

$Ba_{2}CuCl_{2}O_{2}$

![](images/60fdaa275d140439b42d222f6699c1c078188f0f8021992668a3fd071b7cca07.jpg)  
$\mathrm{Tl}_{2}\mathrm{Ca}_{2}\mathrm{Ba}_{2}\mathrm{Cu}_{3}\mathrm{O}_{10}$

![](images/2ad6ca8dee5b950c5b7690c52a026f3da8c57cea99bae9217e6258742b7384f6.jpg)

$\mathrm{Ba}_{3} \mathrm{CaLa}_{2} \mathrm{GdCu}_{7} \mathrm{O}_{17} \mathrm{YCu}_{3} \mathrm{O}_{7}$   
![](images/c9ff7165df8fc577b5e06d0915697de7674062087549d2272baf57b850f28692.jpg)  
$\mathrm{YCu}_{3} \mathrm{O}_{7}$

![](images/53379338f04cafb80d3f0a5002ad8868ac0085f0f463539842ae1fc27ac17fd2.jpg)  
$\mathrm{TbH}_{3}$

![](images/a7b1a5bd5e70f4cdc29116e4e96df9c5a20cbc969f7aca01d67f169c95a83ac5.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing green atoms and red spheres in a cubic arrangement
</details>

$\mathrm{H}_{3}\mathrm{Se}$

![](images/581f3ab01b37d84fcb41701bad71c865c051ff15405e1a7d0474d334b69eae2e.jpg)  
$\mathrm{CaGe}_{2} \mathrm{H}_{9}$

![](images/b265ba166a5034bac30e20f6fc83e11ef1580d9e3ae4fb33d95f5c0cf33ce936.jpg)  
$Ca_{2}MnCrH_{6}$

![](images/2ae45320bfa6b2be06953ed22402cf163bb1615fd8be43af1649ccbe808287cd.jpg)  
$\mathrm{Ba}_{3} \mathrm{Pu}$

![](images/cccaaf13e07e631cef84af1c50ec8e2965e5c6bfac8ff1dd5cb0a720ec480c2d.jpg)  
Th

![](images/7c2dfa618f037172b162597895d49321c144e018a90a490885d4710ac0721d15.jpg)  
ThC $_{3}$

![](images/839571199247dcc4adae2a12d7cbac5d63b041cd38b2e6f4408a143009169800.jpg)  
Lu

![](images/a820d43d2b33335267957021ec19b38592227831e7330833be8c6e62a959002f.jpg)  
$\mathrm{BaFe}_{2}\mathrm{Se}_{2}$

![](images/917009f3974048a8645365bd7acb1feafe9b8c1607146c17a346c63e9a4a4f37.jpg)  
SmFeAsO

![](images/ab6e1a46f1c3a684fadd3839e7a3e53afd8450d3b5c2a503d7aedbd4cafb8885.jpg)  
$\mathrm{KFe}_{2}\mathrm{As}_{2}$

![](images/8a17953233985960875f86e22c2846c7298cee0dba38b20caf3e18201ba8915e.jpg)

<details>
<summary>chemical</summary>

Molecular structure diagram showing a crystal lattice with orange, gray, and green atoms
</details>

NdFeAsF

![](images/8969307064d53bfb8c10e6fa0795d5196f8047c316a3a97c8ec0ba86836d9f22.jpg)  
$\mathrm{Ba}_{3} \mathrm{Ca}$

![](images/a48dde29a9eec3f0d0de20aacd4dd3a8e87cd401895a1c649befb375f3169d23.jpg)

<details>
<summary>chemical</summary>

Molecular lattice structure diagram showing green spheres connected by bonds, with one highlighted unit cell
</details>

${\mathrm{{Ba}}}_{2}\mathrm{{Se}}$

![](images/0e2806462d074e1540fc003107f9bf166d5ed882d94057ccd56a5e08c77d341c.jpg)  
Ba

![](images/0badb0a609c36d7a1c5d9b849c363c452066e9d871135a8ebf0f8b9082f8bdf3.jpg)  
$\mathrm{Mg}_{3}\mathrm{B}$   
Figure 6: The geometric structures of novel superconducting candidates in Table 6.

![](images/f8992c25cb9421213fd26172fcb1d336af2ad16c7fa042b4a1d9c806e69de4eb.jpg)  
$\mathrm{BaFe}_{2} \mathrm{Se}_{2}$

![](images/b7613404815899823717c2bdb86399fc8c9cab9176b6853ed1429d8a303837ba.jpg)  
$\mathrm{BaCaCu}_{3} \mathrm{O}_{7}$

![](images/5caa31ed785dcbb95cf1e5e05c24aa33cd8d03bd3eaab6e31bf3987bc777b94e.jpg)  
Lu

![](images/0d46382026c450d3d2f06d60466c52446063c21614b671d31f682e19df07b3a5.jpg)  
SmFeAsO

![](images/e9027be09901252605628bb615aff881420b4b46880b7be8977f175aa02b37d4.jpg)  
NdFeAsF

![](images/4c38182f5a7f411542bd13788ddb9cce1b3ef6fd23029f138de6aff5c6a7cc5f.jpg)

<details>
<summary>line</summary>

| Wavevector k | total | Ba   | Cd   |
| ------------ | ----- | ---- | ---- |
| G            | -2.0  | -1.5 | -1.0 |
| X | -1.5  | -1.0 | -0.5 |
| Y           | -1.0  | -0.5 | 0.0  |
| Q           | -0.5  | 0.0  | 0.5  |
| R₂           | 0.0   | 0.5  | 1.0  |
| G            | 0.5   | 1.0  | 1.5  |
| T₂ | 1.0   | 1.5  | 2.0  |
| U₂           | 1.5   | 2.0  | 2.5  |
| G            | 2.0   | 2.5  | 3.0  |
| V₂           | 2.5   | 3.0  | 3.5  |
| V₂           | -3.0  | -2.5 | -2.0 |
The top panel displays a zoomed-in view of the DOS plot for the total and Ba components, while the bottom panel shows the corresponding values for the Cd component. The legend indicates the color: blue for total, red for Cd.
</details>

$Ba_{2}Cd_{2}$

![](images/dd9b035d12d73337e1d73bece5785936299a0d8464b3e519f9f6061e96d373c2.jpg)  
$Ba_{2}CuCl_{2}O_{2}$

![](images/868219ae9f4b5a3b422204fc4870e0f1dbf383d48c6bb4897a97099b7bd0cc19.jpg)

<details>
<summary>line</summary>

| Wavevector k | total (eV) | Er (eV) | Zn (eV) | Bi (eV) | O (eV) |
|---|---|---|---|---|---|
| G | ~3.0 | ~2.5 | ~2.0 | ~1.5 | ~1.0 |
| X | ~2.5 | ~2.0 | ~1.5 | ~1.0 | ~0.5 |
| Y | ~2.0 | ~1.5 | ~1.0 | ~0.5 | ~0.0 |
| R | ~1.5 | ~1.0 | ~0.5 | ~0.0 | ~-0.5 |
| G | ~3.0 | ~2.5 | ~2.0 | ~1.5 | ~1.0 |
| T | ~2.5 | ~2.0 | ~1.5 | ~1.0 | ~0.5 |
| U | ~2.0 | ~1.5 | ~1.0 | ~0.5 | ~0.0 |
| Q | ~1.5 | ~1.0 | ~0.5 | ~0.0 | ~-0.5 |
| G | ~3.0 | ~2.5 | ~2.0 | ~1.5 | ~1.0 |
| V | ~2.5 | ~2.0 | ~1.5 | ~1.0 | ~0.5 |
The chart displays the spin-up and spin-down states of an electronic device in a 3D coordinate system (e.g., E - E_F, π), with the top panel showing the spin-up state and the bottom panel showing the spin-down state, alongside a zoomed-in inset illustrating the spatial distribution of the spin-up and spin-down states along the DOS axis.
</details>

$Er_{2}Zn_{2}Bi_{2}O_{6}$

![](images/8b19c7c635427f7b63e1112e19baa0a9c4be4acce89f2c825d473a8df57f5696.jpg)

<details>
<summary>line</summary>

| Wavevector k | E-E_F, eV (Ba) | E-E_F, eV (Sb) |
| ------------ | -------------- | -------------- |
| G            | ~3.0           | ~-2.0          |
| X            | ~2.5           | ~-1.5          |
| Y            | ~2.0           | ~-1.0          |
| GZ           | ~1.5           | ~-0.5          |
| R            | ~1.0           | ~0.0           |
| G            | ~0.5           | ~0.5           |
| T            | ~0.0           | ~1.0           |
| U            | ~0.5           | ~1.5           |
| G            | ~1.0           | ~2.0           |
| V            | ~1.5           | ~2.5           |
| DOS          | Total          | Sb             |
</details>

$\mathbf{Ba}_4\mathbf{Sb}_2$   
Figure 7: The electronic structures of novel superconducting candidates.

# E.5 DFT Calculations

We conduct DFT calculation using the Vienna ab initio package (VASP) [57, 7]. The structures are fully relaxed using the generalized gradient approximation (GGA) [41] of the SCAN meta-GGA functional, employing the pseudopotentials of the projector augmented wave (PAW) method [4]. A plane wave cutoff of $500\mathrm{eV}$ is employed for all simulations. Brillouin-zone integrations are performed using the $\tau$ -centered Monkhorst-Pack (MP) scheme [37]. We initiate the calculations with a k-point meth featuring a dense sampling density of $2\pi \times 0.04$ . The convergence criteria for energy and force is set to $0.1\mathrm{meV}$ and $0.001\mathrm{eV} / \mathring{\mathrm{A}}$ , respectively.

The van Hove singularity (VHS) is a notable occurrence in condensed matter physics, specifically in the density of states (DOS) of a material. It manifests as a distinct peak or divergence in the DOS at a particular energy level. We select materials from Table 12 and 13 for DFT calculations and display their band structures and density of states (DOS) in Fig. 7. From the density of states (DOS) plot, we can observe the van Hove singularity (VHS) phenomenon. Additionally, we can also observe the presence of flat bands in the band structures of materials such as $Ba_{2}CuCl_{2}O_{2}$ , Lu, $Ba_{4}Sb_{2}$ , and others. The integration of flat bands in the electronic architecture, along with the Van Hove Singularities (VHS) in the Density of States (DOS), markedly amplifies the likelihood of these candidates being superconducting materials.

# F Pipeline for Designing Real-world Superconductors.

Fig. 8 presents a pipeline for designing SC., validating our dataset and models for real-world scenarios. We initially generate potential, ordered superconducting structures using the DiffCSP-SC model trained on the SuperCon3D database. Candidate materials are selected based on $T_{c}$ values predicted by SODNet, followed by DFT verification to confirm the presence of superconducting electronic structures, such as VHS. Subsequently, selected candidates undergo wet lab synthesis, with $T_{c}$ values characterized and recorded in the SuperCon3D database. Further, if a superconductor is discovered, methods such as doping, which may transform ordered structures into disordered ones, are explored to enhance the $T_{c}$ value. SODNet is employed to investigate the relationship between disordered structures and doping ratios, aiming to design optimal doping proportions for experimental verification. These experimental outcomes are also recorded in the database. Continuous expansion of the database will incrementally improve the accuracy of the DiffCSP-SC and SODNet models trained on this dataset, creating a reinforcing cycle of enhancement.

![](images/a4100bec45fc3d55317bc0ef764d120279b35ebadd60a256c51a18a92a7ff9ff.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["SuperCon3D"] --> B["DiffCSP-SC"]
    B --> C["Order Cand. Screening"]
    C --> D["SODNet Score"]
    D --> E["DFT filter"]
    E --> F["Top-k"]
    F --> G["Wet Exp."]
    G --> H["SC"]
    I["Improving Tc by Doping"] --> J["Disorder Cand. Screening"]
    J --> K["SODNet Score"]
    K --> L["Top-k"]
    L --> G
```
</details>

Figure 8: Flowchart for designing novel SC materials.

# G Code

We have made the source code for SODNet and DiffCSP-SC, as mentioned in this article, available on GitHub. The repositories can be accessed at: https://github.com/pincher-chen/SODNet and https://github.com/pincher-chen/DiffCSP-SC.