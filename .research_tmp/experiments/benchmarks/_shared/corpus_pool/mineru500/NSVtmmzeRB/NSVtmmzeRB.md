# UNIFIED GENERATIVE MODELING OF 3D MOLECULES VIA BAYESIAN FLOW NETWORKS

Yuxuan Song $^{1*}$ Jingjing Gong $^{1*}$ Yanru Qu $^{2}$ Hao Zhou $^{1}$ Mingyue Zheng $^{3}$ Jingjing Liu $^{1}$ & Wei-Ying Ma $^{1}$

$^{1}$ Institute of AI Industry Research (AIR), Tsinghua University   
$^{2}$ University of Illinois Urbana-Champaign   
$^{3}$ Shanghai Institute of Materia Medica, Chinese Academy of Sciences   
{songyuxuan, gongjingjing, zhouhao, maweiying}@air.tsinghua.edu

# ABSTRACT

Advanced generative model (e.g., diffusion model) derived from simplified continuity assumptions of data distribution, though showing promising progress, has been difficult to apply directly to geometry generation applications due to the multimodality and noise-sensitive nature of molecule geometry. This work introduces Geometric Bayesian Flow Networks (GeoBFN), which naturally fits molecule geometry by modeling diverse modalities in the differentiable parameter space of distributions. GeoBFN maintains the SE-(3) invariant density modeling property by incorporating equivariant inter-dependency modeling on parameters of distributions and unifying the probabilistic modeling of different modalities. Through optimized training and sampling techniques, we demonstrate that GeoBFN achieves state-of-the-art performance on multiple 3D molecule generation benchmarks in terms of generation quality (90.87% molecule stability in QM9 and 85.6% atom stability in GEOM-DRUG $^{1}$ ). GeoBFN can also conduct sampling with any number of steps to reach an optimal trade-off between efficiency and quality (e.g., 20× speedup without sacrificing performance).

# 1 INTRODUCTION

Molecular geometries can be represented as three-dimensional point clouds, characterized by their Cartesian coordinates in space and enriched with descriptive features. For example, proteins can be represented as proximity spatial graphs (Jing et al., 2021) and molecules as atomic graphs in 3D (Schütt et al., 2017). Thus, learning geometric generative models has the potential to benefit scientific discoveries such as material and drug design. Recent progress in deep generative modeling has paved the way for geometric generative modeling. For example, Gebauer et al. (2019); Luo & Ji (2021) and Satorras et al. (2021a) use autoregressive models and flow-based models, respectively, for generating 3D molecules in-silico. Most recently, inspired by the huge success of diffusion model (DM) in image generation Meng et al. (2022); Ho et al. (2020)

![](images/7c2a6c4aeb39f0f540ca32e655ef94f35ddabacea279f3de65b3af0a5f344c3e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Left_Channel
        A["Receiver Distribution"] --> B["Noisy channel"]
        B --> C["EGNN"]
        C --> D["Bayesian Update"]
        D --> E["Sender Sample"]
        E --> F["Sender Distribution"]
        G["Receiver Distribution"] --> H["Noisy channel"]
        H --> I["EGNN"]
        I --> J["Bayesian Update"]
        J --> K["Sender Sample"]
        K --> L["Sender Distribution"]
    end

    subgraph Right_Channel
        M["Receiver Distribution"] --> N["Noisy channel"]
        N --> O["EGNN"]
        O --> P["Bayesian Update"]
        P --> Q["Sender Sample"]
        Q --> R["Sender Distribution"]
        S["Receiver Distribution"] --> T["Noisy channel"]
        T --> U["EGNN"]
        U --> V["Bayesian Update"]
        V --> W["Sender Sample"]
        W --> X["Sender Distribution"]
    end

    style Left_Channel fill:#f9f,stroke:#333
    style Right_Channel fill:#bbf,stroke:#333
```
</details>

Figure 1: The framework of GeoBFN

and beyond Li et al. (2022), DM incorporating geometric symmetries has been widely explored in the field of geometry generation Hoogeboom et al. (2022); Xu et al. (2023).

However, two major challenges remain in directly applying DM to molecule geometry: multi-modality and noise sensitivity. The multi-modality issue refers to the dependency on diverse data forms to effectively depict the atomic-level geometry of a molecule. For instance, the continuous variable of atom coordinates is essential for describing the spatial arrangement, while either the discretised atom charge or categorical atom types are employed to completely determine the molecule's composition. Noise sensitivity refers to the fact that applying noise or perturbing the atom coordinates will not only change the value of the variable but also have a significant impact on the relationship among different atoms as the Euclidean distances are also changed. Therefore, a small noise on atom coordinates could bring a sudden drop of the signal at the molecule level.

To alleviate these issues, Xu et al. (2023) introduces a latent space for alleviating the inconsistency of unified Gaussian diffusion on different modalities. Anand & Achim (2022) propose to use decomposed modeling of different modalities. Peng et al. (2023) use different noise schedulers for different modalities to accommodate noise sensitivity. However, these methods either depend on the sophisticated and artifact-filled design or lack of guarantee or constraint on the designed space.

In this work, we propose Geometric Bayesian Flow Networks (GeoBFN) to model 3D molecule geometry in a principally different way. Bayesian Flow Networks Graves et al. (2023) (BFN) is a novel generative model developed quite recently. Taking a unique approach by incorporating Bayesian inference to modify the parameters of a collection of independent distributions, brings a fresh perspective to geometric generative modeling. Firstly, GeoBFN uses a unified probabilistic modeling formulation for different modalities in the molecule geometry. Secondly, regarding the variable of 3D atom coordinates, the input variance for BFNs is considerably lower than DMs, leading to better compatibility with the inherent noise sensitivity. Further, we bring the geometry symmetries into the Bayesian update procedure through an equivariant inter-dependency modeling module. We also demonstrate that the density function of implied generative distribution is SE-(3) invariant and the generative process of iterative updating is roto-translational equivariant. Thirdly, with BFN's powerful probabilistic modeling capacity, 3D molecule geometry representation can be further optimized into a representation with only two similar modalities: discretised charge and continuous atom coordinates. The mode-redundancy issue on discretised variable in the original BFNs is fixed by an early mode-seeking sampling strategy in GeoBFN.

With operating on the space with less variance, GeoBFN could sample with any number of steps which provides a superior trade-off between efficiency and quality, which leads to a $20\times$ speedup with competitive performance. Besides, GeoBFN is a general framework that can be easily extended to other molecular tasks. We conduct thorough evaluations of GeoBFN on multiple benchmarks, including both unconditional and property-conditioned molecule generation tasks. Results demonstrate that GeoBFN consistently achieves state-of-the-art generation performance on molecule stability and other metrics. Empirical studies also show a significant improvement in controllable generation and demonstrate that GeoBFN enjoys a significantly higher modeling capacity and inference efficiency.

# 2 PRELIMINARIES

# 2.1 SE-(3) INVARIANT DENSITY MODELING

To distinguish geometry representation and the atomic property features, we use the tuple $\mathbf{g} = \langle \mathbf{x},\mathbf{h}\rangle$ to represent the 3D molecules. Note here $\mathbf{x} = (\mathbf{x}^1,\dots ,\mathbf{x}^N)\in \mathbb{R}^{N\times 3}$ is the atom coordinate matrix, and $\mathbf{h} = (\mathbf{h}^1,\dots ,\mathbf{h}^N)\in \mathbb{R}^{N\times d}$ is the node feature matrix, e.g., atomic types and charges. Density estimation on the 3D molecules should satisfy specific symmetry conditions of the geometry. In this work, we focus on the transformations $T_{g}$ in the Special Euclidean group (SE-(3)), i.e., the group of rotation and translation in 3D space, where transformations $T_{g}$ can be represented by a translation t and an orthogonal matrix rotation R. Note for a generative model on molecule geometry with underlying density function $p_{\theta}(\langle \mathbf{x},\mathbf{h}\rangle)$ , the likelihood should not be influenced by the rotation or translation of the entire molecule, which means the likelihood function should be SE-(3) invariant on the input coordinates, i.e., $p_{\theta}(\langle \mathbf{x},\mathbf{h}\rangle) = p_{\theta}(\langle \mathbf{Rx} + \mathbf{t},\mathbf{h}\rangle)$ .

# 2.2 BAYESIAN FLOW NETWORKS

The Bayesian Flow Networks (BFNs) are based on the following latent variable models: for learning the probability distribution $p_{\theta}$ over g, a series of noisy versions $\langle y_{1}, \cdots, y_{n} \rangle$ of g are introduced as

latent variables. And then the variational lower bound of likelihood is optimized:

$$
\begin{array}{l} \log p _ {\boldsymbol {\theta}} (\mathbf {g}) \geq \underset {\mathbf {y} _ {1}, \dots , \mathbf {y} _ {n} \sim q} {\mathbb {E}} \left[ \log \frac {p _ {\phi} \left(\mathbf {g} \mid \mathbf {y} _ {1} , \dots , \mathbf {y} _ {n}\right) p _ {\phi} \left(\mathbf {y} _ {1} , \dots , \mathbf {y} _ {n}\right)}{q \left(\mathbf {y} _ {1} , \dots , \mathbf {y} _ {n} \mid \mathbf {g}\right)} \right] \\ = - D _ {K L} \left(q \| p _ {\phi} \left(\mathbf {y} _ {1}, \dots , \mathbf {y} _ {n}\right)\right) + \underset {\mathbf {y} _ {1}, \dots , \mathbf {y} _ {n} \sim q} {\mathbb {E}} \log \left[ p _ {\phi} \left(\mathbf {g} \mid \mathbf {y} _ {1}, \dots , \mathbf {y} _ {n}\right) \right] \tag {1} \\ \end{array}
$$

And q is namely the variational distribution. The prior distribution of latent variables is usually organized autoregressively, i.e., $p_{\phi}(\mathbf{y}_{1}, \cdots, \mathbf{y}_{n}) = p_{\phi}(\mathbf{y}_{1}) p_{\phi}(\mathbf{y}_{2} \mid \mathbf{y}_{1}) p_{\phi}(\mathbf{y}_{n} \mid \mathbf{y}_{n-1} \cdots \mathbf{y}_{1})$ which also implies the data generation procedure, i.e., $y_{1} \to \cdots \to y_{n} \to g$ (Note: this procedure only demonstrates the generation order, yet does NOT imply Markov property for the following derivative).

One widely adopted intuition for the generation process is that the information of the data samples should progressively increase along with the above Markov chain, e.g., noisier images to cleaner images. The key motivation of BFNs is that the information along the latent variables should change as smoothly as possible for all modalities including discretized and discrete variables. To this end, BFNs operate on the distributions in the parameter space, in contrast to the sample space.

We introduce components of BFNs one by one (Fig.2a). Firstly, the variational distribution $q$ is defined by the following form:

$$
q \left(\mathbf {y} _ {1}, \dots , \mathbf {y} _ {n} \mid \mathbf {g}\right) = \prod_ {i = 1} ^ {n} p _ {S} \left(\mathbf {y} _ {i} \mid \mathbf {g}; \alpha_ {i}\right) \tag {2}
$$

$p_{S}\left(\mathbf{y}_{i} \mid \mathbf{g}; \alpha_{i}\right)$ is termed as the sender distribution, which could be seen as adding noise to the data according to a predefined accuracy $\alpha_{i}$ .

Secondly, for the definition of $p_{\phi}$ , BFNs will first transfer the noisy sample $\mathbf{y}$ to the parameter space, obtaining $\theta$ , then apply Bayesian update in the parameters space and transfer back to the noisy sample space at last. To clarify, $\theta$ refers to the parameter of distributions in the sample space, e.g., the mean/variance for Gaussian distribution or probabilities for categorical distribution. In the scope of BFNs, the distributions on the sample space are factorized by default, e.g., $p(\mathbf{g} \mid \boldsymbol{\theta}) = \prod_{d=1}^{D} p(g^{(d)} \mid \theta^{(d)})$ .

Thirdly, a neural network $\Phi$ takes $\pmb{\theta}$ as input and aims to model the dependency among different dimensions hence to recover the distribution of the original sample $\mathbf{g}$ . The output of neural network $\Phi(\pmb{\theta})$ still lies in the parameter space, and we termed it as the parameter of output distribution $p_O$ , where $p_O(\mathbf{y}|\pmb{\theta};\phi) = \prod_{d=1}^{D} p_O(\mathbf{y}^{(d)} \mid \Phi(\pmb{\theta})^{(d)})$ .

To map the noisy sample y to the input space, Bayesian update is applied to $\theta$ :

$$
\boldsymbol {\theta} _ {i} \leftarrow h (\boldsymbol {\theta} _ {i - 1}, \mathbf {y} _ {i}, \alpha_ {i}), \tag {3}
$$

$h$ is called Bayesian update function. The distribution over $(\theta_0, \ldots, \theta_{n-1})$ is then defined by the Bayesian update distribution via marginalizing out $\mathbf{y}$ :

$$
p _ {\phi} \left(\boldsymbol {\theta} _ {0}, \dots , \boldsymbol {\theta} _ {n - 1}\right) = p (\boldsymbol {\theta} _ {0}) \prod_ {i = 1} ^ {n} p _ {U} \left(\boldsymbol {\theta} _ {i} \mid \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right), \tag {4}
$$

where $p(\pmb{\theta}_0)$ is a simple prior for ease of generation, e.g., standard normal, and $p_U$ could be obtained from Eq. 3:

$$
p _ {U} \left(\boldsymbol {\theta} _ {i} \mid \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) = \underset {p _ {R} (\mathbf {y} _ {i} | \boldsymbol {\theta} _ {i - 1}; \alpha_ {i})} {\mathbb {E}} \delta \left(\boldsymbol {\theta} _ {i} - h \left(\boldsymbol {\theta} _ {i - 1}, \mathbf {y} _ {i}, \alpha_ {i}\right)\right), \tag {5}
$$

$\delta$ being the Dirac delta distribution. $p_{R}(\mathbf{y}_{i}|\boldsymbol{\theta}_{i - 1},\alpha_{i}) = \underset {p_{O}(\mathbf{x}^{\prime}|\boldsymbol{\theta}_{i - 1};\phi)}{\mathbb{E}} p_{S}(\mathbf{y}_{i}|\mathbf{x}^{\prime};\alpha_{i})$ and is also called the as receiver distribution.

At last we map $\Phi (\pmb {\theta})$ back to the noisy sample space by combining the known form, accuracy of $P_{S}$ and marginalizing out $\mathbf{y}$ :

$$
\begin{array}{l} p _ {\phi} \left(\mathbf {y} _ {1}, \dots , \mathbf {y} _ {n}\right) = p _ {\phi} \left(\mathbf {y} _ {1}\right) \prod_ {i = 2} ^ {n} p _ {\phi} \left(\mathbf {y} _ {i} \mid \mathbf {y} _ {\{1: i - 1 \}} \right. = \prod_ {i = 1} ^ {n} p _ {\phi} \left(\mathbf {y} _ {i} \mid \boldsymbol {\theta} _ {i - 1}\right) \\ = \prod_ {i = 1} ^ {n} p _ {O} (\mathbf {x} _ {i} ^ {\prime} | \boldsymbol {\theta} _ {i - 1}; \phi) \left[ p _ {S} (\mathbf {y} _ {i} | \mathbf {x} _ {i} ^ {\prime}; \alpha_ {i}) \right], \tag {6} \\ \end{array}
$$

![](images/cc525fb3b451e03df87832a5ee15d2723227530a5c8030a78b16b2a1295f3f2f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    θ0["θ₀"] --> θ1["θ₁"]
    θ1 --> θ2["θ₂"]
    θ2 --> θ3["θᵢ"]
    θ3 --> θn["θₙ"]
    θn --> g["g"]
    θ1 --> y1["y₁"]
    θ2 --> y2["y₂"]
    θ3 --> yi["yᵢ"]
    θn --> yn["yₙ"]
    y1 -.-> y2
    y2 -.-> yi
    yi -.-> yn
    y3 -.-> yn
```
</details>

a Graphical Model of BFN

![](images/be3adafc65c3772700e386f0458ed18ff9244a13c8f8d99f80207998737c7018.jpg)  
b Graphical Model of Diffusion   
Figure 2: Graphical View of Comparison between BFN and Diffusion

where we use $\theta_{0:n-1}$ to abbreviate $(\theta_{0},\ldots,\theta_{n-1})$ , and y similar. . Till now, we have defined q, $p_{\phi}(\mathbf{y}_{1},\ldots,\mathbf{y}_{n})$ , and $p_{\phi}(\mathbf{g}\mid\mathbf{y}_{1},\ldots,\mathbf{y}_{n})$ is simply $p_{O}(\mathbf{g}\mid\boldsymbol{\theta}_{n})$ on each sample, thus Eq.1 can be estimated.

# 3 METHODOLOGY

# 3.1 SE-(3) INVARIANT GEOMETRY DENSITY MODELING

As discussed in Sec. 2.1, for a generative model on the 3D molecule geometry, it is crucial to hold the SE-(3) invariant conditions. Recall the mathematics formula of the geometries $g = \langle x, h \rangle$ , we denote the latent variable, e.g., noisy samples, of g as $y^{g}$ . We are interested in applying the SE-(3) invariant conditions to the probabilistic model $p_{\phi}$ . To this end, we need to first reformulate the likelihood function:

$$
p _ {\phi} (\mathbf {g}) = p _ {\phi} \left(\langle \mathbf {x}, \mathbf {h} \rangle\right) = \int_ {\mathbf {y} _ {1} ^ {g}, \dots , \mathbf {y} _ {n} ^ {g}} p _ {\phi} \left(\mathbf {g} \mid \mathbf {y} _ {1} ^ {g}, \dots , \mathbf {y} _ {n} ^ {g}\right) p _ {\phi} \left(\mathbf {y} _ {1} ^ {g}, \dots , \mathbf {y} _ {n} ^ {g}\right) d \mathbf {y} _ {1} ^ {g} \dots d \mathbf {y} _ {n} ^ {g}. \tag {7}
$$

With $\theta^{x}$ and $y^{x}$ and all the distributions defined in the same way as above, we focus on the variables corresponds to x in the geometry $y^{g}$ . Then we have the following theorem:

# Theorem 3.1. (SE-(3) Invariant Condition)

- With the $\theta^x$ , $\mathbf{y}^x$ , $\mathbf{x}$ constrained in the zero Center of Mass(CoM) space (Köhler et al., 2020; Xu et al., 2022), the likelihood function $p_\phi$ is translational invariant.   
- When the following properties are satisfied, the likelihood function $p_{\phi}$ is roto-invariant:

$$
p _ {O} \left(\mathbf {x} ^ {\prime} \mid \boldsymbol {\theta} _ {i - 1} ^ {x}; \phi\right) = p _ {O} \left(\mathbf {R} (\mathbf {x} ^ {\prime}) \mid \boldsymbol {R} (\boldsymbol {\theta} _ {i - 1} ^ {x}); \phi\right); p _ {S} \left(\mathbf {y} ^ {x} \mid \mathbf {x} ^ {\prime}; \alpha\right) = p _ {S} \left(\mathbf {R} (\mathbf {y} ^ {x}) \mid \mathbf {R} (\mathbf {x} ^ {\prime}); \alpha\right);
$$

$$
h (\mathbf {R} (\pmb {\theta} _ {i - 1} ^ {x}), \mathbf {R} (\mathbf {y} _ {i} ^ {x}), \alpha_ {i}) = \mathbf {R} h (\pmb {\theta} _ {i - 1} ^ {x}, \mathbf {y} _ {i} ^ {x}, \alpha_ {i}); p (\mathbf {x} ^ {\prime} | \pmb {\theta} _ {0} ^ {x}) = p (\mathbf {R} (\mathbf {x} ^ {\prime}) | \pmb {\theta} _ {0} ^ {x}), \forall o r t h o g o n a l \mathbf {R}
$$

Proposition 3.2. With the condition in Theorem. 3.1 satisfied, the evidence lower bound objective in Eq. 1, i.e.,

$$
\mathcal {L} _ {V L B} (\mathbf {x}) = \underset {p _ {\phi} \left(\boldsymbol {\theta} _ {0} ^ {x}, \dots , \boldsymbol {\theta} _ {n} ^ {x}\right)} {\mathbb {E}} \left[ \sum_ {i = 1} ^ {n} D _ {K L} \left(p _ {S} (\cdot | \mathbf {x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i - 1} ^ {x}; \alpha_ {i})\right) - \log p _ {\phi} (\mathbf {x} | \boldsymbol {\theta} _ {n} ^ {x}) \right], \tag {8}
$$

with the Bayesian update distribution $p_{\phi} \left( \boldsymbol{\theta}_{0}^{x}, \ldots, \boldsymbol{\theta}_{n-1}^{x} \right) = \prod_{i=1}^{n} p_{U} \left( \boldsymbol{\theta}_{i} \mid \boldsymbol{\theta}_{i-1}, \mathbf{x}; \alpha_{i} \right)$ similar to Eq. 4. And $p_{R}(\cdot \mid \boldsymbol{\theta}_{i-1}^{x}; \alpha_{i}) = \underset{p_{O}(\mathbf{x}_{i}^{\prime} | \boldsymbol{\theta}_{i-1}^{x}; \phi)}{\mathbb{E}} \left[ p_{S}(\mathbf{y}_{i} | \mathbf{x}_{i}^{\prime}; \alpha_{i}) \right]$ , $p_{\phi}(\mathbf{x} | \boldsymbol{\theta}_{n}^{x}) = p_{O}(\mathbf{x} | \boldsymbol{\theta}_{n}^{x}, \phi)$ . Derivation from Eq. 1 to equation 8 is at Appendix C.4. if $p_{U} \left( \boldsymbol{\theta}_{i} \mid \boldsymbol{\theta}_{i-1}, \mathbf{x}; \alpha_{i} \right) = p_{U} \left( \mathbf{R}\boldsymbol{\theta}_{i} \mid \mathbf{R}\boldsymbol{\theta}_{i-1}, \mathbf{R}\mathbf{x}; \alpha_{i} \right)$ , then $\mathcal{L}_{VLB}(\mathbf{x})$ is also SE-(3) invariant.

We leave the formal proof of Theorem. 3.1 and Proposition. 3.2 in Appendix C.

# 3.2 GEOMETRIC BAYESIAN FLOW NETWORKS

Then we introduce the detailed formulation of geometric Bayesian flow networks (GeoBFN) based on the analysis in Sec. 3.1. For describing a 3D molecule geometry $g = \langle x, h \rangle$ , various representations can be utilized for the node features. The atom types $h_{t}$ and atomic charges $h_{c}$ are commonly employed, with the former being discrete (categorical) and the latter being discretized (integer). Together with the continuous variable, e.g., atom coordinates x, the network module in the modeling of the output distribution of GeoBFN could be parameterized with an equivariant graph neural network (EGNN) (Satorras et al., 2021b) $\Phi$ :

$$
\Phi \left(\mathbf {R} \boldsymbol {\theta} ^ {x} + \mathbf {t}, \left[ \boldsymbol {\theta} ^ {h _ {t}}, \boldsymbol {\theta} ^ {h _ {c}} \right]\right) = \left[ \mathbf {R} \boldsymbol {\theta} ^ {x ^ {\prime}} + \mathbf {t}, \boldsymbol {\theta} ^ {h _ {t} ^ {\prime}}, \boldsymbol {\theta} ^ {h _ {c} ^ {\prime}} \right], \quad \forall \mathbf {R}, \mathbf {t} \tag {9}
$$

where $\Phi (\pmb{\theta}^x,[\pmb{\theta}^{h_t},\pmb{\theta}^{h_c})] = [\pmb{\theta}^{x'}\pmb{\theta}^{h_t'}\pmb{\theta}^{h_c'}]$ . And then we introduce the necessary components to derive the objective in Eq. 8

Atom Coordinates x and Charge $h_{c}$ : For the continuous and discretized variables, the input distribution is set as the factorized Gaussian distributions, where $\theta \stackrel{\operatorname{def}}{:=}\{\mu,\rho\}$ the parameter of $\mathcal{N}\left(\cdot\mid\mu,\rho^{-1}\mathbf{I}\right)$ . For simplicity, we take x as an example to illustrate the common parts of the two variables. And $\theta_{0}^{x}$ is set as $\{0,1\}$ . The sender distribution $p_{S}$ is also an isotropic Gaussian distribution:

$$
p _ {S} (\cdot \mid \mathbf {x}; \alpha \mathbf {I}) = \mathcal {N} (\mathbf {x}, \alpha^ {- 1} \mathbf {I}) \tag {10}
$$

Given the nice property of isotropic Gaussian (proof given by Graves et al. (2023)), the simple form of Bayesian update function could be derived as:

$$
h \left(\left\{\boldsymbol {\mu} _ {i - 1}, \rho_ {i - 1} \right\}, \mathbf {y}, \alpha\right) = \left\{\boldsymbol {\mu} _ {i}, \rho_ {i} \right\}, \quad \text { Here } \quad \rho_ {i} = \rho_ {i - 1} + \alpha , \boldsymbol {\mu} _ {i} = \frac {\boldsymbol {\mu} _ {i - 1} \rho_ {i - 1} + \mathbf {y} \alpha}{\rho_ {i}} \tag {11}
$$

As shown in Eq. 11, the randomness only exists in $\mu$ , and the corresponding Bayesian update distribution in Eq. 8 is as:

$$
p _ {U} \left(\boldsymbol {\theta} _ {i} \mid \boldsymbol {\theta} _ {i - 1}, \mathbf {x}; \alpha\right) = \mathcal {N} \left(\boldsymbol {\mu} _ {i} \mid \frac {\alpha \mathbf {x} + \boldsymbol {\mu} _ {i - 1} \rho_ {i - 1}}{\rho_ {i}}, \frac {\alpha}{\rho_ {i} ^ {2}} \mathbf {I}\right) \tag {12}
$$

The above discrete-time Bayesian update could be easily extended to continuous-time, with an accuracy scheduler defined as $\beta(t) = \int_{t' = 0}^{t}\alpha\left(t'\right)dt', t \in [0,1]$ . Given the accuracy additive property of $p_U$ (proof given by Graves et al. (2023)), $\mathbb{E}_{p_U(\boldsymbol{\theta}_{i-1}|\boldsymbol{\theta}_{i-2},\mathbf{x};\alpha_a)} p_U(\boldsymbol{\theta}_i \mid \boldsymbol{\theta}_{i-1},\mathbf{x};\alpha_b) = p_U(\boldsymbol{\theta}_i \mid \boldsymbol{\theta}_{i-2},\mathbf{x};\alpha_a + \alpha_b)$ , the Bayesian flow distribution could be obtained as:

$$
p _ {F} (\boldsymbol {\theta} ^ {x} \mid \mathbf {x}; t) = p _ {U} \left(\boldsymbol {\theta} ^ {x} \mid \boldsymbol {\theta} _ {0}, \mathbf {x}; \beta (t)\right) \tag {13}
$$

The key difference of atom coordinates x and charges $h_{c}$ lies in the design of the output distribution. For continuous variable x, the network module $\Phi$ directly outputs an estimated $\hat{x} = \Phi(\theta^{g}, t)$ . Hence for timestep t, the output distribution is

$$
p _ {O} \left(\mathbf {x} ^ {\prime} \mid \boldsymbol {\theta} ^ {g}, t; \phi\right) = \delta (\mathbf {x} - \Phi (\boldsymbol {\theta} ^ {g}, t)) \tag {14}
$$

While for discretized variable $h_{c}$ , the network module will output two variables, $\mu_{h_{c}}$ and $\ln \sigma_{h_{c}}$ with dimension equivalent to $h_{c}$ which implies a distribution $\mathcal{N}(\boldsymbol{\mu}_{h_{c}}, \sigma_{h_{c}}^{2} \mathbf{I})$ . With a K-bins discretized variable, the support is split into K buckets with each bucket k centered as $k_{c} = \frac{2k-1}{K} - 1$ and left boundary as $k_{l} = k_{c} - \frac{1}{K}$ and right boundary as $k_{r} = k_{c} + \frac{1}{K}$ . Then for each k, the probability is the mass from $k_{l}$ to $k_{r}$ , i.e., $\int_{k_{l}}^{k_{r}} \mathcal{N}(\mu_{h_{c}}, \sigma_{h_{c}}^{2} \mathbf{I})$ . And the first and last bins are curated by making sure the sum of the probability mass is 1. Then the output distribution is:

$$
p _ {O} (\mathbf {h} _ {c} \mid \boldsymbol {\theta} ^ {g}, t; \phi) = \prod_ {d = 1} ^ {D} p _ {O} ^ {(d)} \left(k \left(h _ {c} ^ {(d)}\right) \mid \boldsymbol {\theta} ^ {g}, t; \phi\right), \tag {15}
$$

where the function $k(\cdot)$ maps the variable to the corresponding bucket.

Atom Types $h_{t}$ : The atom types $h_{t}$ are discrete variables with K categories, where the corresponding parameter space lies in probability simplex thus the procedure is slightly different from the others. The input distribution for $\mathbf{h}_{t}$ is $p_{I}(\mathbf{h}_{t} \mid \boldsymbol{\theta}) = \prod_{d=1}^{D} \boldsymbol{\theta}^{h_{t}(d)}$ , where D is number if variables. And the input prior $\theta_{0}^{h_{t}} = \frac{1}{K}$ , where $\frac{1}{K}$ is the length KD vector whose entries are all $\frac{1}{K}$ . The sender distribution, could be derived with the central limit theorem, lies in the form of

$$
p _ {S} (\mathbf {y} \mid \mathbf {h} _ {t}; \alpha) = \mathcal {N} (\mathbf {y} \mid \alpha (K \mathbf {e} _ {\mathbf {h} _ {t}} - \mathbf {1}), \alpha K \mathbf {I}) \tag {16}
$$

where 1 is a vector of ones, I is the identity matrix, and $\mathbf{e}_j\in \mathbb{R}^K$ is a vector defined as the projection from the class index $j$ to a length $K$ one-hot vector (proof given by Graves et al. (2023)). In other words, each element of $\mathbf{e}_j$ is defined as $(\mathbf{e}_j)_k = \delta_{jk}$ , where $\delta_{jk}$ is the Kronecker delta function. And $\mathbf{e}_{\mathbf{h}_t}\stackrel {\mathrm{def}}{=}\left(\mathbf{e}_{\mathbf{h}_t^{(1)}},\dots ,\mathbf{e}_{\mathbf{h}_t^{(D)}}\right)\in \mathbb{R}^{KD}$ .

![](images/28a6b6ee342b9fd1edcbb2ce734dc6414c58657adc99b72e8c698bbdfb50364b.jpg)

<details>
<summary>chemical</summary>

Molecular structures of GeoBFN and EDM showing atom arrangements and spatial arrangements
</details>

Figure 3: The Bayesian Flow and Diffusion Process of GeoBFN and EDM.

The Bayesian update function could be derived as $h(\boldsymbol{\theta}_{i-1}, \mathbf{y}, \alpha) = \frac{e^{\mathbf{y}}\boldsymbol{\theta}_{i-1}}{\sum_{k=1}^{K} e^{\mathbf{y}_k(\boldsymbol{\theta}_{i-1})_k}}$ (proof given by Graves et al. (2023)). And similar to Eq. 13, the Bayesian flow distribution for $\mathbf{h}_t$ is as:

$$
p _ {F} \left(\boldsymbol {\theta} ^ {h _ {t}} \mid \mathbf {h} _ {t}; t\right) = \underset {\mathcal {N} \left(\mathbf {y} ^ {\mathbf {h} _ {t}} \mid \beta (t) \left(K \mathbf {e} _ {\mathbf {h} _ {t}} - \mathbf {1}\right), \beta (t) K \mathbf {I}\right)} {\mathbb {E}} \delta \left(\boldsymbol {\theta} ^ {h _ {t}} - \operatorname{softmax} \left(\mathbf {y} ^ {\mathbf {h} _ {t}}\right)\right) \tag {17}
$$

With the network module $\Phi$ , the output distribution could be obtained as

$$
p _ {O} ^ {(d)} (k \mid \boldsymbol {\theta} ^ {g}; t) = \left(\operatorname{softmax} \left(\Phi^ {(d)} (\boldsymbol {\theta} ^ {g}, t)\right)\right) _ {k}, p _ {O} (\mathbf {h} _ {t} \mid \boldsymbol {\theta}; t) = \prod_ {d = 1} ^ {D} p _ {O} ^ {(d)} \left(h _ {t} ^ {(d)} \mid \boldsymbol {\theta} ^ {g}; t\right) \tag {18}
$$

Training Objective: By combining the different variables together, we could obtain the unified continuous-time loss for GeoBFN based on Eq. 25 to Eq. 41 in (Graves et al., 2023) as:

$$
\begin{array}{l} L ^ {\infty} (\mathbf {g}) = L ^ {\infty} (\langle \mathbf {x}, \mathbf {h} _ {c}, \mathbf {h} _ {t} \rangle) = \underset {t \sim U (0, 1), p _ {F} (\boldsymbol {\theta} ^ {g} | \mathbf {g}; t)} {\mathbb {E}} \left[ \frac {\alpha^ {g} (t)}{2} \| \mathbf {g} - \Phi (\boldsymbol {\theta} ^ {g}, t) \| ^ {2} \right] \\ = \underset { \begin{array}{c} t \sim U (0, 1), \\ \boldsymbol {\theta} ^ {g} \sim p _ {F} (\cdot | \mathbf {g}; t) \end{array} } {\mathbb {E}} \left[ \frac {\alpha^ {x} (t)}{2} \| \mathbf {x} - \Phi_ {\mathbf {x}} \| ^ {2} + \frac {\alpha^ {\mathbf {h} _ {c}} (t)}{2} \| \mathbf {h} _ {c} - \Phi_ {\mathbf {h} _ {c}} \| ^ {2} + \frac {\alpha^ {\mathbf {h} _ {t}} (t)}{2} \| \mathbf {h} _ {t} - \Phi_ {\mathbf {h} _ {t}} \| ^ {2} \right] \tag {19} \\ \end{array}
$$

Where $\Phi$ . is short for $\Phi.\left(\boldsymbol{\theta}^{g},t\right)$ . The joint Bayesian flow distribution is decomposed as:

$$
p _ {F} (\boldsymbol {\theta} ^ {g} \mid \mathbf {g}; t) = p _ {F} (\boldsymbol {\theta} ^ {x} \mid \mathbf {x}; t) p _ {F} (\boldsymbol {\theta} ^ {h _ {c}} \mid \mathbf {h} _ {c}; t) p _ {F} (\boldsymbol {\theta} ^ {h _ {t}} \mid \mathbf {h} _ {t}; t), \tag {20}
$$

with $\alpha^x$ , $\alpha^{h_c}$ and $\alpha^{h_t}$ refer to the corresponding accuracy scheduler (details provided by Graves et al. (2023)). And $\Phi_x$ is defined the same as in Eq. 14; while $\Phi_{h_c}$ is defined by the weighted average of different bucket centers with the output distribution in Eq. 15 as $\left(\sum_{k=1}^{K} p_O^{(1)}(k \mid \boldsymbol{\theta}, t) k_c, \ldots, \sum_{k=1}^{K} p_O^{(D)}(k \mid \boldsymbol{\theta}, t) k_c\right)$ ; And for $\Phi_{h_t}$ , it is defined as the $\sum_{k=1}^{K} p_O^{(d)}(k \mid \boldsymbol{\theta}; t) \mathbf{e}_k$ based on Eq. 18.

Remark 3.3. The GeoBFN defined in the above formulation satisfied the SE(3)-invariant condition in Theorem. 3.1.

Sampling GeoBFN will generate samples follow the graphical model in the recursive procedure as illustrated in Fig. 2a: e.g., $g' \sim p_O(\cdot|\boldsymbol{\theta}_{i-1}) \to y \sim p_S(\cdot|g', \alpha) \to \boldsymbol{\theta}_i = h(\boldsymbol{\theta}_{i-1}, y, \alpha)$ .

# 3.3 OVERCOME NOISE SENSITIVITY IN MOLECULE GEOMETRY

One key obstacle of applying diffusion models to 3D molecule generation is the noise sensitivity property of the molecule geometry. The property of noise sensitivity seeks to state the fact: When noise is incorporated into the coordinates and displaces them significantly from their original positions, the bond distance between certain connected atoms may exceed the bond length range[1]. Under these circumstances, the point cloud could potentially lose the critical chemical information inherently encoded in the bonded structures; Another perspective stems from the reality that when noise is added to the coordinates, the relationships (distance) between different atoms could alter at a more rapid pace, e.g. modifying the coordinates of one atom results in altering its distance to all other atoms. Thus, the intermediate steps' structure in the generation procedure of diffusion models the intermediate steps' structure might be uninformative. And the majority of the information being acquired in the final few steps of generation (as depicted in Fig. 3).

A fundamental belief underpinning GeoBFN is that a smoother transformation during the generative process could result in a more favorable inductive bias according to (Graves et al., 2023). This process occurs within the parameter space of GeoBFN, which is regulated through the Bayesian update procedure. Specifically, samples exhibiting higher degrees of noise are assigned lesser weight during this update (refer to Eq. 11). This approach consequently leads to a significant reduction in variance within the parameter space as (Graves et al., 2023), which in turn facilitates the smooth transformation of molecular geometries. As illustrated in Fig. 3, this is evidenced by the gradual convergence of the structure of the intermediary steps towards the final structure, thus underscoring the effectiveness of smoother transformation.

# 3.4 OPTIMIZED DISCRETISED VARIABLE SAMPLING

Previous research (Hoogeboom et al., 2022; Xu et al., 2023; Wu et al., 2022) utilizes both the atom types $h_{t}$ and charges $h_{c}$ to represent the atomic properties. The $h_{c}$ usually serves as an auxiliary loss for improving training which is not involved in determining the molecule graph during generation due to the insufficient modeling. However, there is redundant information between these two variables, since the $h_{t}$ and $h_{c}$ variables have a one-to-one mapping, e.g. the charge value 4 could be uniquely determined as the Carbon atom. We found that with advanced probabilistic modeling on discretized data, GeoBFN could conduct training and sampling only with x and $h_{c}$ . However, there exists a counterexample for the objective in Eq. 19 and the output distribution during sampling as in Eq. 15. As shown in Fig 5, the boundary condition for clamping the cumulative probability function in the bucket could cause the mismatch, e.g., the true density should be centered in the center bucket while the output distribution instead put the most density in the first and last buckets which cause the mode-redundancy as shown in upper-left in Fig. 5. Though the weighted sum in Eq. 19 is optimized, the sampling procedure will rarely sample the center buckets. And such cases could be non-negligible in our scenarios, especially when the number of bins is small for low dimensional data. To alleviate this issue, we instead update the output distribution in the sampling procedure to:

$$
\hat {\boldsymbol {k}} _ {c} (\boldsymbol {\theta}, t) = \text { NEAREST\_CENTER } \left(\left[ \sum_ {k = 1} ^ {K} p _ {O} ^ {(1)} (k \mid \boldsymbol {\theta}, t) k _ {c}, \dots , \sum_ {k = 1} ^ {K} p _ {O} ^ {(D)} (k \mid \boldsymbol {\theta}, t) k _ {c} \right]\right) \tag {21}
$$

Function NEAREST\_CENTER compares inputs to the center bins $\vec{k}_c = \left(k_c^{(1)},\dots ,k_c^{(D)}\right)$ , and return the nearest center for each input value. The updated distribution is unbiased towards the training objective and also reduce the variance during generation which could be found in the trajectory of Fig.5.

# 4 EXPERIMENTS

# 4.1 EXPERIMENT SETUP

Task and Datasets We focus on the 3D molecule generation task following the setting of prior works (Gebauer et al., 2019; Luo & Ji, 2021; Satorras et al., 2021a; Hoogeboom et al., 2022; Wu et al., 2022). We consider both Unconditional Molecular Generation which assesses the capability to learn the underlying molecular data distribution and generate chemically valid and structurally diverse molecules and the Conditional Molecule Generation tasks which evaluate the capacity of generating molecules with desired properties. For Conditional Molecule Generation, we implement a conditional version GeoBFN with the details in the Appendix. The widely adapted QM9 (Ramakrishnan et al., 2014) and the GEOM-DRUG (Gebauer et al., 2019; 2021) with large molecules are used for the experiments. And the data configurations directly follow previous work(Anderson et al., 2019; Hoogeboom et al., 2022; Xu et al., 2023) $^{2}$ .

Evaluation Metrics The evaluation configuration follows the prior works (Hoogeboom et al., 2022; Wu et al., 2022; Xu et al., 2023). For the Unconditional Molecular Generation, the bond types are first predicted (single, double, triple, or none) based on pair-wise atomic distance and atom types in the 10000 generated molecular geometries (Hoogeboom et al., 2022). With the obtained molecular graph, we evaluate the quality by calculating both atom stability and molecule stability metrics. Besides, the validity (based on RDKIT) and uniqueness are also reported. Regarding the

Table 1: Results of atom stability, molecule stability, validity, validity×uniqueness (V×U), and novelty. A higher number indicates a better generation quality. The results marked with an asterisk were obtained from our own tests. And GeoBFN $_{k}$ denote the results of sampling the molecules with a specific number of steps k 

<table><tr><td rowspan="2"># Metrics</td><td colspan="5">QM9</td><td colspan="2">DRUG</td></tr><tr><td>Atom Sta (%)</td><td>Mol Sta (%)</td><td>Valid (%)</td><td>V×U (%)</td><td>Novelty (%)</td><td>Atom Sta (%)</td><td>Valid (%)</td></tr><tr><td>Data</td><td>99.0</td><td>95.2</td><td>97.7</td><td>97.7</td><td>-</td><td>86.5</td><td>99.9</td></tr><tr><td>ENF</td><td>85.0</td><td>4.9</td><td>40.2</td><td>39.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>G-Schnet</td><td>95.7</td><td>68.1</td><td>85.5</td><td>80.3</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GDM-AUG</td><td>97.6</td><td>71.6</td><td>90.4</td><td>89.5</td><td>74.6</td><td>77.7</td><td>91.8</td></tr><tr><td>EDM</td><td>98.7</td><td>82.0</td><td>91.9</td><td>90.7</td><td>58.0</td><td>81.3</td><td>92.6</td></tr><tr><td>EDM-Bridge</td><td>98.8</td><td>84.6</td><td>92.0</td><td>90.7</td><td>-</td><td>82.4</td><td>92.8</td></tr><tr><td>GEOLDM</td><td> $98.9 \pm 0.1$ </td><td> $89.4 \pm 0.5$ </td><td> $93.8 \pm 0.4$ </td><td> $92.7 \pm 0.5$ </td><td>57.0</td><td>84.4</td><td>99.3</td></tr><tr><td> $GEOBFN_{50}$ </td><td> $98.28 \pm 0.1$ </td><td> $85.11 \pm 0.5$ </td><td> $92.27 \pm 0.4$ </td><td> $90.72 \pm 0.3$ </td><td>72.9</td><td>75.11</td><td>91.66</td></tr><tr><td> $GEOBFN_{100}$ </td><td> $98.64 \pm 0.1$ </td><td> $87.21 \pm 0.3$ </td><td> $93.03 \pm 0.3$ </td><td> $91.53 \pm 0.3$ </td><td>70.3</td><td>78.89</td><td>93.05</td></tr><tr><td> $GEOBFN_{500}$ </td><td> $98.78 \pm 0.8$ </td><td> $88.42 \pm 0.2$ </td><td> $93.35 \pm 0.2$ </td><td> $91.78 \pm 0.2$ </td><td>67.7</td><td>81.39</td><td>93.47</td></tr><tr><td> $GEOBFN_{1k}$ </td><td> $99.08 \pm 0.06$ </td><td> $90.87 \pm 0.2$ </td><td> $95.31 \pm 0.1$ </td><td> $92.96 \pm 0.1$ </td><td>66.4</td><td>85.60</td><td>92.08</td></tr><tr><td> $GEOBFN_{2k}$ </td><td> $99.31 \pm 0.03$ </td><td> $93.32 \pm 0.1$ </td><td> $96.88 \pm 0.1$ </td><td> $92.41 \pm 0.1$ </td><td>65.3</td><td>86.17</td><td>91.66</td></tr></table>

Table 2: Mean Absolute Error for molecular property prediction with 500 sampling steps. A lower number indicates a better controllable generation result. 

<table><tr><td>Property Units</td><td> $\alpha$ Bohr $^{3}$ </td><td> $\Delta \varepsilon$ meV</td><td> $\varepsilon_{\text{HOMO}}$ meV</td><td> $\varepsilon_{\text{LUMO}}$ meV</td><td> $\mu$ D</td><td> $C_v$  $\frac{\text{cal}}{\text{mol}}$ K</td></tr><tr><td>QM9*</td><td>0.10</td><td>64</td><td>39</td><td>36</td><td>0.043</td><td>0.040</td></tr><tr><td>Random*</td><td>9.01</td><td>1470</td><td>645</td><td>1457</td><td>1.616</td><td>6.857</td></tr><tr><td> $N_{\text{atoms}}$ </td><td>3.86</td><td>866</td><td>426</td><td>813</td><td>1.053</td><td>1.971</td></tr><tr><td>EDM</td><td>2.76</td><td>655</td><td>356</td><td>584</td><td>1.111</td><td>1.101</td></tr><tr><td>GEOLDM</td><td>2.37</td><td>587</td><td>340</td><td>522</td><td>1.108</td><td>1.025</td></tr><tr><td>GEOBFN</td><td>2.34</td><td>577</td><td>328</td><td>516</td><td>0.998</td><td>0.949</td></tr></table>

Table 3: Ablation study, GeoBFN models molecule charge settings, the sampling step is set to 1,000.

<table><tr><td>Charge Feature</td><td>Atom Stable (%)</td><td>Mol Stable (%)</td></tr><tr><td>discretised_basis</td><td>99.08</td><td>90.87</td></tr><tr><td>continuous_basis</td><td>98.97</td><td>89.94</td></tr><tr><td>discrete</td><td>98.93</td><td>88.93</td></tr><tr><td>discrete + continuous</td><td>98.96</td><td>89.33</td></tr><tr><td>discrete + discretised</td><td>98.91</td><td>88.65</td></tr></table>

Conditional Molecule Generation, we evaluate our conditional version of GeoBFN on QM9 with 6 properties: polarizability $\alpha$ , orbital energies $\varepsilon_{\mathrm{HOMO}}$ , $\varepsilon_{\mathrm{LUMO}}$ and their gap $\Delta \varepsilon$ , Dipole moment $\mu$ , and heat capacity $C_v$ . Following previous work Hoogeboom et al. (2022); Xu et al. (2023), the conditional GeoBFN is fed with a range of property $s$ to generate samples and the same pre-trained classifier $w$ is utilized to measure the property of generated molecule as $\hat{s}$ . The Mean Absolute Error (MAE) between $s$ and $\hat{s}$ is calculated to measure whether the generated molecules is related to the conditioned property.

Baselines GeoBFN is compared with several advanced baselines including G-Schnet (Gebauer et al., 2019), Equivariant Normalizing Flows (ENF) (Satorras et al., 2021a) and Equivariant Graph Diffusion Models (EDM) with its non-equivariant variant (GDM) (Hoogeboom et al., 2022). Also with recent advancements, EDM-Bridge (Wu et al., 2022) which improves upon the performance of EDM by incorporating well-designed informative prior bridges and also GeoLDM (Xu et al., 2023) where a latent space diffusion model is applied are both included. To yield a fair comparison, all the method-agnostic configurations are set as the same. The implementation details could be found in Appendix. B.

![](images/6344cfb7f4c6bd0e3e65cc29ab3874cdaf8cfe377d9059f8a8119d8224a97127.jpg)

<details>
<summary>line</summary>

| Sampling Step | GeoBFN | EDM   | EDM-Bridge | GEOLDM | upper bound |
| ------------- | ------ | ----- | ---------- | ------ | ----------- |
| 0             | 0.85   | 0.67  | 0.70       | 0.85   | 0.95        |
| 1000          | 0.90   | 0.82  | 0.85       | 0.90   | 0.95        |
| 2000          | 0.93   | -     | -          | -      | 0.95        |
| 3000          | 0.94   | -     | -          | -      | 0.95        |
| 4000          | 0.94   | -     | -          | -      | 0.95        |
| 4500          | 0.94   | -     | -          | -      | 0.95        |
</details>

Figure 4: QM9 Molecule Stability wrt. Sampling Steps

![](images/71e8a7c47f4bb26256ea3548a71a44b26a3dec0ac7587901a62ae8416a0d4c48.jpg)

<details>
<summary>scatter</summary>

| Algorithm Type | Metric | Value |
| --- | --- | --- |
| Original sampling algorithm | Distribution | 1.00 |
| Original sampling algorithm | n_0 Generated | 0.75 |
| Original sampling algorithm | n_1 Generated | 0.50 |
| Original sampling algorithm | n_2 Generated | 0.25 |
| Original sampling algorithm | n_3 Generated | 0.00 |
| Original sampling algorithm | n_4 Generated | -0.25 |
| Original sampling algorithm | n_5 Generated | -0.50 |
| Original sampling algorithm | n_6 Generated | -0.75 |
| Original sampling algorithm | n_7 Generated | -1.00 |
| Improved sampling algorithm | Distribution | 1.00 |
| Improved sampling algorithm | n_0 Generated | 0.75 |
| Improved sampling algorithm | n_1 Generated | 0.50 |
| Improved sampling algorithm | n_2 Generated | 0.25 |
| Improved sampling algorithm | n_3 Generated | 0.00 |
| Improved sampling algorithm | n_4 Generated | -0.25 |
| Improved sampling algorithm | n_5 Generated | -0.50 |
| Improved sampling algorithm | n_6 Generated | -0.75 |
| Improved sampling algorithm | n_7 Generated | -1.00 |
| Transport trajectory | Distribution | 1.00 |
| Transport trajectory | n_0 Generated | 0.75 |
| Transport trajectory | n_1 Generated | 0.50 |
| Transport trajectory | n_2 Generated | 0.25 |
| Transport trajectory | n_3 Generated | 0.00 |
| Transport trajectory | n_4 Generated | -0.25 |
| Transport trajectory | n_5 Generated | -0.50 |
| Transport trajectory | n_6 Generated | -0.75 |
| Transport trajectory | n_7 Generated | -1.00 |
| Transport trajectory | n_8 Generated | 1.00 |
| Transport trajectory | n_9 Generated | 0.75 |
| Transport trajectory | n_10 Generated | 0.50 |
| Transport trajectory | n_11 Generated | 0.25 |
| Transport trajectory | n_12 Generated | 0.00 |
| Transport trajectory | n_13 Generated | -0.25 |
| Transport trajectory | n_14 Generated | -0.50 |
| Transport trajectory | n_15 Generated | -0.75 |
| Transport trajectory | n_16 Generated | -1.00 |
| Transport trajectory | n_17 Generated | 1.00 |
| Transport trajectory | n_18 Generated | 0.75 |
| Transport trajectory | n_19 Generated | 0.50 |
| Transport trajectory | n_20 Generated | 0.25 |
| Transport trajectory | n_21 Generated | 0.00 |
| Transport trajectory | n_22 Generated | -0.25 |
| Transport trajectory | n_23 Generated | -0.50 |
| Transport trajectory | n_24 Generated | -0.75 |
| Transport trajectory | n_25 Generated | -1.00 |
| Transport trajectory | n_26 Generated | 1.00 |
| Transport trajectory | n_27 Generated | 0.75 |
| Transport trajectory | n_28 Generated | 0.50 |
| Transport trajectory | n_29 Generated | 0.25 |
| Transport trajectory | n_30 Generated | 0.00 |
| Transport trajectory | n_31 Generated | -0.25 |
| Transport trajectory | n_32 Generated | -0.50 |
| Transport trajectory | n_33 Generated | -0.75 |
| Transport trajectory | n_34 Generated | -1.00 |
| Transport trajectory | n_35 Generated | 1.00 |
| Transport trajectory | n_36 Generated | 0.75 |
| Transport trajectory | n_37 Generated | 0.50 |
| Transport trajectory | n_38 Generated | 0.25 |
| Transport trajectory | n_39 Generated | 0.00 |
| Transport trajectory | n_40 Generated | -0.25 |
| Transport trajectory | n_41 Generated | -0.50 |
| Transport trajectory | n_42 Generated | -0.75 |
| Transport trajectory | n_43 Generated | -1.00 |
| Transport trajectory | n_44 Generated | 1.00 |
| Transport trajectory | n_45 Generated | 0.75 |
| Transport trajectory | n_46 Generated | 0.50 |
| Transport trajectory | n_47 Generated | 0.25 |
| Transport trajectory | n_48 Generated | 0.00 |
| Transport trajectory | n_49 Generated | -0.25 |
| Transport trajectory | n_50 Generated | -0.50 |
| Transport trajectory | n_51 Generated | -0.75 |
| Transport trajectory | n_52 Generated | -1.00 |
| Improved sampling algorithm (n) vs. Transport trajectory (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) (n) = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100
</details>

Figure 5: 2D Synthetic case of optimized synthetic example. In the left columns, generated samples are in orange, and data points are in blue.

# 4.2 MAIN RESULTS

The results of Unconditional Molecular Generation can be found in Tab. 1. We could observe that in both the QM9 and GEOM-DRUG datasets, GeoBFN achieves a new state-of-the-art performance regarding both the quality and diversity of the generated molecules which demonstrates the huge potential of GeoBFN on geometry generative modeling. The phenomenon demonstrates that the GeoBFN does not hold the tendency to collapse to the subset of training data which could imply a probabilistic generalization ability and could be useful for several application scenarios; The Conditional Molecule Generation results can be found in Tab. 2. GeoBFN consistently outperforms other baseline models by an obvious margin in all conditional generation tasks. This clearly highlights the effectiveness and generalization capability of the proposed methods.

# 4.3 ANY-STEP SAMPLING

One notable property of GeoBFN is that training with the continuous-time loss, e.g., Eq. 19, the sampling could be conducted with any steps without incurring additional training overhead. As shown in Tab. 1, GeoBFN could get superior performance compared to several advanced models with only 50 steps during sampling which brings $20 \times$ speed-up during sampling due to the benefit of low variance parameter space. As we could find in Fig 4, with the sampling steps increasing from 50 to 4600, the molecule stability could be further boosted to approach the upper bound, e.g., $94.25\%$ with 4000 steps.

# 4.4 ABLATION STUDIES

We conduct ablation studies on the effect of input modalities in Tab. 3. We try different compositions and losses to represent the atom types, discretised basis refers to the case where the charge feature is used with discretised and the Gaussian basis, i.e., $\phi_j(x) = \exp\left(-\frac{(x - \mu_j)^2}{2\sigma^2}\right)$ is used as functional embedding for charge; continuous basis only differ in that the continuous loss is utilized. The discrete refers to including the one-hot type representation; discrete+continuous refers to both the one-hot type and charge are included while continuous loss is included; Similar is the discrete+continuous. With only discretised variable utilized, the performance is superior to including the discrete variable which implies powerful probabilistic modeling capacity and the benefits of applying similar modality.

# 5 RELATED WORK

Previous molecule generation studies have primarily focused on generating molecules as 2D graphs (Jin et al., 2018; Liu et al., 2018; Shi et al., 2020), but there has been increasing interest in 3D molecule generation. With the increasing interest in 3D molecule generation, G-Schnet and G-SphereNet (Gebauer et al., 2019; Luo & Ji, 2021) respectively, employ autoregressive techniques to create molecules in a step-by-step manner by progressively connecting atoms or molecular fragments. These frameworks have also been extended to structure-based drug design (Li et al., 2021; Peng et al., 2022; Powers et al., 2022). There are approaches use atomic density grids that generate the entire molecule in a single step by producing a density over the voxelized 3D space (Masuda et al., 2020). Most recently, the attention has shifted towards using DMs for 3D molecule generation (Hoogeboom et al., 2022; Wu et al., 2022; Peng et al., 2023; Xu et al., 2023), with successful applications in target drug generation (Lin et al., 2022), antibody design (Luo et al., 2022), and protein design (Anand & Achim, 2022; Trippe et al., 2022). However, our method is based on the Bayesian Flow Network (Graves et al., 2023) objective and hence lies in a different model family which fundamentally differs from this line of research in both training and generation.

# 6 CONCLUSION

We introduce GeoBFN, a new generative framework for molecular geometry. GeoBFN operates in a differentiable parameter space for variables from different modalities. Also, the less variance in parameter space is naturally compatible with the noise sensitivity of molecule geometry. Given the appealing property, the GeoBFN achieves state-of-the-art performance on several 3D molecule generation benchmarks. Besides, GeoBFN can also conduct sampling with an arbitrary number of steps to reach an optimal trade-off between efficiency and quality (e.g., $20 \times$ speedup without sacrificing performance).

# ACKNOWLEDGMENTS

The authors thank the anonymous reviewers for reviewing the draft. This work is supported by the National Science and Technology Major Project (2022ZD0117502), Natural Science Foundation of China (62376133) and Guoqiang Research Institute General Project, Tsinghua University (No. 2021GQG1012).

# REFERENCES

Namrata Anand and Tudor Achim. Protein structure and sequence generation with equivariant denoising diffusion probabilistic models. arXiv preprint arXiv:2205.15019, 2022.   
Brandon Anderson, Truong Son Hy, and Risi Kondor. Cormorant: Covariant molecular neural networks. Advances in neural information processing systems, 32, 2019.   
Niklas Gebauer, Michael Gastegger, and Kristof Schütt. Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules. Advances in neural information processing systems, 32, 2019.   
Niklas WA Gebauer, Michael Gastegger, Stefaan SP Hessmann, Klaus-Robert Müller, and Kristof T Schütt. Inverse design of 3d molecular structures with conditional generative neural networks. arXiv preprint arXiv:2109.04824, 2021.   
Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson, and Faustino Gomez. Bayesian flow networks. arXiv preprint arXiv:2308.07037, 2023.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arXiv:2006.11239, 2020.   
Emiel Hoogeboom, Victor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In International Conference on Machine Learning, pp. 8867–8887. PMLR, 2022.   
Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In International conference on machine learning, pp. 2323–2332. PMLR, 2018.   
Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael John Lamarre Townshend, and Ron Dror. Learning from protein structure with geometric vector perceptrons. In International Conference on Learning Representations, 2021.   
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In 3nd International Conference on Learning Representations, 2014.   
Jonas Köhler, Leon Klein, and Frank Noe. Equivariant flows: Exact likelihood generative learning for symmetric densities. In Proceedings of the 37th International Conference on Machine Learning, 2020.   
Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori Hashimoto. Diffusion-LM improves controllable text generation. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=3s9IrEsjLyk.   
Yibo Li, Jianfeng Pei, and Luhua Lai. Structure-based de novo drug design using 3d deep generative models. Chemical science, 12(41):13664–13675, 2021.   
Haitao Lin, Yufei Huang, Meng Liu, Xuanjing Li, Shuiwang Ji, and Stan Z Li. Diffbp: Generative diffusion of 3d molecules for target protein binding. arXiv preprint arXiv:2211.11214, 2022.   
Qi Liu, Miltiadis Allamanis, Marc Brockschmidt, and Alexander Gaunt. Constrained graph variational autoencoders for molecule design. In Advances in neural information processing systems, 2018.

Shitong Luo, Yufeng Su, Xingang Peng, Sheng Wang, Jian Peng, and Jianzhu Ma. Antigen-specific antibody design and optimization with diffusion-based generative models for protein structures. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=jSorGn2Tjg.   
Youzhi Luo and Shuiwang Ji. An autoregressive flow model for 3d molecular geometry generation from scratch. In International Conference on Learning Representations, 2021.   
Tomohide Masuda, Matthew Ragoza, and David Ryan Koes. Generating 3d molecular structures conditional on a receptor binding site with deep generative models. arXiv preprint arXiv:2010.14442, 2020.   
Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=aBsCcjcPu\_tE.   
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. In NIPS-W, 2017.   
Xingang Peng, Shitong Luo, Jiaqi Guan, Qi Xie, Jian Peng, and Jianzhu Ma. Pocket2mol: Efficient molecular sampling based on 3d protein pockets. In International Conference on Machine Learning, 2022.   
Xingang Peng, Jiaqi Guan, Qiang Liu, and Jianzhu Ma. Moldiff: Addressing the atom-bond inconsistency problem in 3d molecule diffusion generation. arXiv preprint arXiv:2305.07508, 2023.   
Alexander S. Powers, Helen H. Yu, Patricia Suriana, and Ron O. Dror. Fragment-based ligand generation guided by geometric deep learning on protein-ligand structure. bioRxiv, 2022. doi: 10.1101/2022.03.17.484653. URL https://www.biorxiv.org/content/early/2022/03/21/2022.03.17.484653.   
Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. Scientific data, 1(1):1–7, 2014.   
Victor Garcia Satorras, Emiel Hoogeboom, Fabian B Fuchs, Ingmar Posner, and Max Welling. E (n) equivariant normalizing flows for molecule generation in 3d. arXiv preprint arXiv:2105.09016, 2021a.   
Victor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks. In International conference on machine learning, pp. 9323–9332. PMLR, 2021b.   
Kristof T Schütt, Farhad Arbabzadah, Stefan Chmiela, Klaus R Müller, and Alexandre Tkatchenko. Quantum-chemical insights from deep tensor neural networks. Nature communications, 8:13890, 2017.   
Chence Shi, Minkai Xu, Zhaocheng Zhu, Weinan Zhang, Ming Zhang, and Jian Tang. Graphaf: a flow-based autoregressive model for molecular graph generation. arXiv preprint arXiv:2001.09382, 2020.   
Brian L Trippe, Jason Yim, Doug Tischer, Tamara Broderick, David Baker, Regina Barzilay, and Tommi Jaakkola. Diffusion probabilistic modeling of protein backbones in 3d for the motif-scaffolding problem. arXiv preprint arXiv:2206.04119, 2022.   
Lemeng Wu, Chengyue Gong, Xingchao Liu, Mao Ye, and qiang liu. Diffusion-based molecule generation with informative prior bridges. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=TJUNtiZiTKE.   
Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, and Jian Tang. Geodiff: A geometric diffusion model for molecular conformation generation. arXiv preprint arXiv:2203.02923, 2022.

Minkai Xu, Alexander Powers, Ron Dror, Stefano Ermon, and Jure Leskovec. Geometric latent diffusion models for 3d molecule generation. arXiv preprint arXiv:2305.01140, 2023.

# A EXPLANATION OF THE DATA EXCHANGE PERSPECTIVE OF BAYESIAN FLOW NETWORKS

In this section, we provide a brief overview of the Bayesian Flow Networks Graves et al. (2023) from a data exchange perspective. Bayesian Flow Networks (BFNs) is a new class of generative model that operates on the parameters of a set of independent distributions with Bayesian inference. The meta elements of BFNs are the input distributions, sender distributions, and output distributions. To start with, we denote the D-dimensional variable as $\mathbf{m} = (m^{(1)}, \ldots, m^{(D)}) \in \mathcal{M}^{D}$ , and $\boldsymbol{\theta} = (\theta^{(1)}, \ldots, \theta^{(D)})$ represent the parameters of a D-dimensional factorised distribution, i.e., $p(\boldsymbol{m} \mid \boldsymbol{\theta}) = \prod_{d=1}^{D} p(m^{(d)} \mid \theta^{(d)})$ .

The basic logic of BFN could be better explained by the following communication example between the sender, referred to as Alice, and the receiver Bob. Alice aims to transfer some data to Bod in a progressive fashion, i.e., at each timestep, Alice corrupts the data according to some channel noise, and then the noisy sample is transferred. The sender distribution is then defined to describe the noise-adding procedure, which is also a factorized distribution, $p_{S}(\mathbf{y} \mid \mathbf{m}; \alpha) = \prod_{d=1}^{D} p_{S}(y^{(d)} \mid m^{(d)}; \alpha)$ . $\alpha$ refers to the accuracy parameter, $\alpha = 0$ refers to the information of the sample being totally destroyed by the noise, and with $\alpha$ increase the noisy sample will contain more information of the original sample. Intuitively, the sender distribution could be approximately understood as adding noise to each dimension of the data independently.

After receiving the noisy sample y, Bob will first update an initial “guess” of what is the original sample behind the noisy sample, i.e. input distribution. Note that except for the noisy sample, Bob also knows the accuracy parameter $\alpha$ and noise formulation while not aware of the original sample m, i.e., the noise level to create such a sample. The input distribution is initially a simple prior on the data space, e.g., a standard Gaussian, lies in the mean-field family, $p_{I}(\mathbf{m} \mid \boldsymbol{\theta}) = \prod_{d=1}^{D} p_{I} \left( x^{(d)} \mid \theta^{(d)} \right)$ . The parameter of input distribution will be updated through Bayesian inference, noted as $\boldsymbol{\theta}_{i} = h \left( \boldsymbol{\theta}_{i-1}, \mathbf{y}, \alpha_{i} \right)$ . This update usually lies in a simple form, e.g. additive or weighted average.

After updating the parameter of input distribution, Bob has an “assistant” which will help to provide a better guess on the original data which generates the observed noisy sample. The assistant aims to exploit more context information between different dimensions, e.g., the relationship between different pixels in an image, in contrast to updating each dimension independently as in the input distribution. Empirically, the assistant could be implemented by a neural network $\Psi$ which takes all parameters of input distribution for the prediction of parameters of each dimension, i.e., $\Psi(\theta) = (\Psi^{(1)}(\boldsymbol{\theta}, t), \ldots, \Psi^{(D)}(\boldsymbol{\theta}, t))$ .

The output distribution is then implied by the predicted parameter which lies in the formulation of $p_{O}(\mathbf{m} \mid \boldsymbol{\theta}, t) = \prod_{d=1}^{D} p_{O}(m^{(d)} \mid \Psi^{(d)}(\boldsymbol{\theta}, t))$ . Then Bob could construct a distribution to approximate the sender distribution at accuracy $\alpha$ by combining the output distribution with the known noise form, accuracy, i.e., $p_{R}(\cdot \mid \boldsymbol{\theta}; t, \alpha) = \mathbb{E}_{p_{O}(\mathbf{x}' \mid \boldsymbol{\theta}; t)} p_{S}(\mathbf{y} \mid \mathbf{x}'; \alpha)$ . Such distribution is called receiver distribution. The "assistant" of BFNs $\Psi$ is to minimize the KL divergence with a defined accuracy scheduler under different timesteps, i.e., $D_{KL}(p_{S}(\cdot \mid \mathbf{m}; \alpha_{i}) \| p_{R}(\cdot \mid \boldsymbol{\theta}_{i-1}; t_{i-1}, \alpha_{i}))$ , which could also be interpreted as transmission cost under the bits-back coding scheme.

# B IMPLEMENTATION DETAILS

The bayesian flow network is implemented with EGNNs Satorras et al. (2021b) by PyTorch (Paszke et al., 2017) package. We set the dimension of latent invariant features k to 1 for QM9 and 2 for DRUG, which extremely reduces the atomic feature dimension. For the training of vector field network $v_{\theta}$ : on QM9, we train EGNNs with 9 layers and 256 hidden features with a batch size 64; and on DRUG, we train EGNNs with 4 layers and 256 hidden features, with batch size 64. The model uses SiLU activations. We train all the modules until convergence. For all the experiments, we choose the Adam optimizer (Kingma & Ba, 2014) with a constant learning rate of $10^{-4}$ as our

default training configuration. The training on QM9 takes approximately 2000 epochs, and on DRUG takes 20 epochs.

# C PROOF OF THEOREMS

In this Section, we provide the formal proof of the Theorem. 3.1 and Proposition. 3.2, as well as the detailed derivations for Equations.

# C.1 DISCUSSION ON THE TRANSLATIONAL INVARIANCE

Remark C.1. It is important to distinguish it from the rotation invariant. The rotational invariant is defined as $p(x) = p(\mathbf{R}x)$ , while the translational is not as $p(x) = p(x + t)$ as such distribution can not integrate into one and hence does not exist. Fortunately, the freedom of translation could be eliminated by only focusing on learning distribution on the linear subspace where the center of gravity is always zero. This is, for all configurations on $R^{n \times 3}$ space, the density on the zero CoM space is utilized to represent their density; It's important to note that the distribution is not defined for configurations outside the zero CoM space. However, it remains possible to leverage the distribution to provide a density-evaluation (not probability density) on the configurations outside the zero CoM space. This is achieved by projecting them back into the subspace. The evaluation procedure for configurations out of zero CoM space could only get a quantity defined artificially instead of the true density of some real distribution, e.g. It is referred to as "CoM-free density" in (Xu et al., 2022). Thus, there does not exist correctness issues.

# C.1.1 ZERO CENTER OF MASS(COM) IN THE GEOBFN

Here we provide detailed discussions on optimizing the distribution in the zero CoM space. Recall the training objective in equation 8,

$$
\mathcal {L} _ {\mathrm{VLB}} (\mathbf {x}) = \underset {p _ {\phi} \left(\boldsymbol {\theta} _ {0} ^ {x}, \dots , \boldsymbol {\theta} _ {n} ^ {x}\right)} {\mathbb {E}} \left[ \sum_ {i = 1} ^ {n} D _ {K L} \left(p _ {S} (\cdot | \mathbf {x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i - 1} ^ {x}; \alpha_ {i})\right) - \log p _ {\phi} (\mathbf {x} | \boldsymbol {\theta} _ {n} ^ {x}) \right], \tag {22}
$$

where $p_{\phi}(\mathbf{x}|\boldsymbol{\theta}_{n}^{x}) = p_{O}(\mathbf{x}|\boldsymbol{\theta}_{n}^{x}, \phi)$ . For learning a distribution on the zero CoM space of $R^{n \times 3}$ , the $p_{S}(\cdot \mid \mathbf{x}; \alpha_{i}), p_{R}(\cdot \mid \boldsymbol{\theta}_{i-1}^{x}; \alpha_{i})$ and $p_{\phi}(\mathbf{x} \mid \boldsymbol{\theta}_{n}^{x})$ are all defined and supported on the zero CoM space, here $\sum_{i=1}^{n} x_{i} = 0$ and $\sum_{i=1}^{n} \boldsymbol{\theta}^{x} = 0$ . In other words, such distribution has no definition for variable $v \in R^{n \times 3}$ if $\sum_{i=1}^{n} v_{i} \neq 0$ . We then express the likelihood function of an isotropic diagonal Gaussian distribution, which is originally defined on the zero CoM space $((n - 1) \times 3\text{-dimensional})$ , in the ambient space $(n \times 3\text{-dimensional})$ as (Hoogeboom et al., 2022):

$$
\mathcal {N} _ {x} (\boldsymbol {x} \mid \boldsymbol {\mu}, \sigma^ {2} \mathbf {I}) = (\sqrt {2 \pi} \sigma) ^ {- (n - 1) \times 3} \exp \left(- \frac {1}{2 \sigma^ {2}} \| \boldsymbol {x} - \boldsymbol {\mu} \| ^ {2}\right) \tag {23}
$$

Here $\sigma^2$ is the variance which is equivalent for each dimension. Recall the Eq. 35 in the (Graves et al., 2023), which shows that $D_{KL}(p_S(\cdot | \mathbf{x}; \alpha_i) \| p_R(\cdot | \boldsymbol{\theta}_{i-1}^x; \alpha_i))$ takes the form of $D_{KL}\left(\mathcal{N}\left(\mathbf{x}, \alpha_i^{-1}\mathbf{I}\right) \| \mathcal{N}\left(\Phi_{\mathbf{x}}(\boldsymbol{\theta}^g, t), \alpha_i^{-1}\mathbf{I}\right)\right)$ which is the KL divergence between to diagonal Gaussian, then we derive the KL divergence for isotropic diagonal normal distributions of zero CoM with means represent on the ambient space. If $p_S = \mathcal{N}\left(\hat{\mu}_1, \sigma^2\mathbf{I}\right)$ and $p_R = \mathcal{N}\left(\hat{\mu}_2, \sigma^2\mathbf{I}\right)$ on subspace, where $\hat{\mu}_1$ and $\hat{\mu}_2$ is $(n-1) \times 3$ -dimension. Then the KL between them could be represented as:

$$
D _ {K L} (q \| p) = \frac {1}{2} \left[ \frac {\left\| \boldsymbol {\hat {\mu}} _ {1} - \boldsymbol {\hat {\mu}} _ {2} \right\| ^ {2}}{\sigma^ {2}} \right] \tag {24}
$$

There is an orthogonal transformation $Q$ which transforms the ambient space $\boldsymbol{\mu}_i\in \mathbb{R}^{n\times 3}$ where $\sum_{i}\boldsymbol{\mu}_{i} = \mathbf{0}$ to the subspace in the way that $\left[ \begin{array}{c}\hat{\boldsymbol{\mu}}\\ \mathbf{0} \end{array} \right] = \mathbf{Q}\boldsymbol{\mu}$ . With $\| \hat{\boldsymbol{\mu}}\| = \| \left[ \begin{array}{c}\boldsymbol{\mu}\\ \mathbf{0} \end{array} \right]\| = \| \boldsymbol{\mu}\|$ , there is $\| \hat{\boldsymbol{\mu}}_1 - \hat{\boldsymbol{\mu}}_2\|^2 = \| \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|^2$ . Hence we have:

$$
D _ {K L} \left(\mathcal {N} \left(\mathbf {x}, \alpha_ {i} ^ {- 1} \boldsymbol {I}\right) \| \mathcal {N} (\Phi_ {\mathbf {x}} (\boldsymbol {\theta} ^ {g}, t), \alpha_ {i} ^ {- 1} \boldsymbol {I})\right) = \frac {\alpha_ {i}}{2} \| \mathbf {x} - \Phi_ {\mathbf {x}} (\boldsymbol {\theta} ^ {g}, t) \| ^ {2} \tag {25}
$$

Hence we demonstrate the correctness of our objective in Eq. 19.

# C.1.2 PROOF OF THE TRANSLATIONAL INVARIANT DENSITY EVALUATION PROCEDURE.

Proof. For an n-atom molecule $g = \langle x, h \rangle$ , the coordinate variable x has the dimension of $n \times 3$ . Note that with the zero Center of Mass mapping (Xu et al., 2022; Satorras et al., 2021a), where we constrain the center of gravity as zero ( $\sum_{i=1}^{n} x_i = 0$ ), then variable x essentially lies in the $(n - 1) \times 3$ -dimensional linear subspace. The generative distributions $p_X$ mentioned in all of the related literature (Satorras et al., 2021a; Hoogeboom et al., 2022; Xu et al., 2022; 2023) is constrained in the zero Center of Mass space. This is, for samples in the ambient space, if $\sum_{i=1}^{n} x_i \neq 0$ , $p_X$ is not defined. The translational invariant property of distribution $p_X$ mentioned is not referred to the fact that $p_X(x) = p_X(x + t)$ for all translation vector t is satisfied in the ambient space with dimension $n \times 3$ . Actually, such conditions could not be satisfied in any space (Satorras et al., 2021a). The translational invariant condition actually refers to the invariant function f which could evaluate the density of all the ambient space based on $p_X$ , the evaluated density by f also referred to as "CoM-free standard density" in (Xu et al., 2022). The function f is defined as

$$
f (\mathbf {x}) = p _ {\mathrm{X}} (Q \mathbf {x}) \tag {26}
$$

where Q refers to the operation which maps the x to the zero Center of Mass space, e.g. in our work Q is defined as

$$
Q = I _ {3} \otimes \left(I _ {N} - \frac {1}{N} \mathbf {1} _ {N} \mathbf {1} _ {N} ^ {T}\right), \quad \text { s.t. } \quad Q \mathbf {x} = \left[ \begin{array}{c} \mathbf {x} _ {1} - \frac {\sum_ {i = 1} ^ {n} \mathbf {x} _ {i}}{n} \\ \dots \\ \mathbf {x} _ {n} - \frac {\sum_ {i = 1} ^ {n} \mathbf {x} _ {i}}{n} \end{array} \right] \tag {27}
$$

that subtracting mean $\frac{\sum_{i=1}^{n} x_{i}}{n}$ from each $x_{i}$ , where $I_{k}$ denotes the $k \times k$ identity matrix and $1_{k}$ denotes the k-dimensional vector filled with 1s. Then the density evaluation function f is translational invariant in the ambient space:

$$
f (\mathbf {x} + \mathbf {t}) = p _ {\mathrm{X}} (\hat {Q} (\mathbf {x} + \mathbf {t})) = p _ {\mathrm{X}} \left(\left[ \begin{array}{c} \mathbf {x} _ {1} + \mathbf {t} _ {i} - \frac {\sum_ {i = 1} ^ {n} \left(\mathbf {x} _ {i} + \mathbf {t} _ {i}\right)}{n} \\ \dots \\ \mathbf {x} _ {n} + \mathbf {t} _ {n} - \frac {\sum_ {i = 1} ^ {n} \left(\mathbf {x} _ {i} + \mathbf {t} _ {i}\right)}{n} \end{array} \right]\right) \tag {28}
$$

Note t stands for a translation vector, which implies that $t_{1} = \cdots = t_{i} = t_{n} = C \in R^{3}$ . Then we have:

$$
f (\mathbf {x} + \mathbf {t}) = p _ {\mathrm{X}} \left(\left[ \begin{array}{c} \mathbf {x} _ {1} + \mathbf {C} - \frac {\sum_ {i = 1} ^ {n} \left(\mathbf {x} _ {i} + \mathbf {C}\right)}{n} \\ \dots \\ \mathbf {x} _ {n} + \mathbf {C} - \frac {\sum_ {i = 1} ^ {n} \left(\mathbf {x} _ {i} + \mathbf {C}\right)}{n} \end{array} \right]\right) = p _ {\mathrm{X}} \left(\left[ \begin{array}{c} \mathbf {x} _ {1} - \frac {\sum_ {i = 1} ^ {n} \mathbf {x} _ {i}}{n} \\ \dots \\ \mathbf {x} _ {n} - \frac {\sum_ {i = 1} ^ {n} \mathbf {x} _ {i}}{n} \end{array} \right]\right) = p _ {\mathrm{X}} (Q \mathbf {x}) = f (\mathbf {x}) \tag {29}
$$

![](images/849b9a8c0acf8a68d8b497c6357c81baf77e2003fd5742740b7f677d1fe1c247.jpg)

Furthermore, the above proof has no constraint on the distribution $p_{X}$ . This is, for any distribution on the zero Center of Mass space, the corresponding evaluation function defined in Eq. 26 is translational invariant.

# C.2 PROOF OF THEOREM. 3.1.

Given the above discussion on the translational invariance, for simplicity, we could only focus on the rotation transformation.

Proof. Recall the graphical model in Fig. 2, we could reformulate the density function in Eq. 7 as:

$$
\begin{array}{l} p _ {\phi} (\mathbf {x}) = \int p _ {\phi} (\mathbf {x} \mid \boldsymbol {\theta} _ {1} ^ {x}, \dots , \boldsymbol {\theta} _ {n} ^ {x}) p _ {\phi} (\boldsymbol {\theta} _ {1} ^ {x}, \dots , \boldsymbol {\theta} _ {n} ^ {x}) d \boldsymbol {\theta} _ {1: n} ^ {x} \quad (\text { definition   of   marginal }) \\ = \int p _ {\phi} (\mathbf {x} \mid \boldsymbol {\theta} _ {n} ^ {x}) p (\boldsymbol {\theta} _ {0}) \prod_ {i = 1} ^ {n} p _ {U} \left(\boldsymbol {\theta} _ {i} \mid \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) d \boldsymbol {\theta} _ {1: n} ^ {x}. \tag {30} \\ \end{array}
$$

Note that $p_{\phi}(\mathbf{x} \mid \boldsymbol{\theta}_{n}^{x}) = p_{\phi}(\mathbf{R}\mathbf{x} \mid \mathbf{R}\boldsymbol{\theta}_{n}^{x}) = p_{O}(\mathbf{R}(\mathbf{x}) \mid \mathbf{R}(\boldsymbol{\theta}_{n}^{x}); \phi)$ due to the property of EGNN, and $p(\boldsymbol{\theta}_{0}) = p(\mathbf{R}\boldsymbol{\theta}_{0})$ since $\theta_{0} = 0$ . Then we prove that $p_{U}(\boldsymbol{\theta}_{i} \mid \boldsymbol{\theta}_{i-1}; \alpha_{i})$ satisfies the equivariant condition that $p_{U}(\boldsymbol{\theta}_{i} \mid \boldsymbol{\theta}_{i-1}; \alpha_{i}) = p_{U}(\mathbf{R}\boldsymbol{\theta}_{i} \mid \mathbf{R}\boldsymbol{\theta}_{i-1}; \alpha_{i})$ . Recall that $p_{U}(\boldsymbol{\theta}_{i} \mid \boldsymbol{\theta}_{i-1}; \alpha_{i}) =$

$\mathbb{E}_{p_{O}(\mathbf{y}_{i}|\boldsymbol{\theta}_{i-1};\alpha_{i})}\delta\left(\boldsymbol{\theta}_{i}-h\left(\boldsymbol{\theta}_{i-1},\mathbf{y}_{i},\alpha_{i}\right)\right)$ , then we have:

$$
\begin{array}{l} p _ {U} \left(\mathbf {R} \boldsymbol {\theta} _ {i} \mid \mathbf {R} \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) = \underset {p _ {O} (\mathbf {y} _ {i} | \mathbf {R} \boldsymbol {\theta} _ {i - 1}; \alpha_ {i})} {\mathbb {E}} \delta \left(\mathbf {R} \boldsymbol {\theta} _ {i} - h \left(\mathbf {R} \boldsymbol {\theta} _ {i - 1}, \mathbf {y} _ {i}, \alpha_ {i}\right)\right) \\ = \int p _ {O} \left(\mathbf {y} _ {i} \mid \mathbf {R} \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) \delta \left(\mathbf {R} \boldsymbol {\theta} _ {i} - h \left(\mathbf {R} \boldsymbol {\theta} _ {i - 1}, \mathbf {y} _ {i}, \alpha_ {i}\right)\right) d \mathbf {y} _ {i} \tag {31} \\ \end{array}
$$

Then we apply integration-by-substitution and replace the variable $\mathbf{y}_i$ with a new variable $\mathbf{y}_i'$ , i.e. $\mathbf{y}_i = \mathbf{R}\mathbf{y}_i'$ , into the Eq. 31:

$$
\begin{array}{l} \int p _ {O} \left(\mathbf {y} _ {i} \mid \mathbf {R} \pmb {\theta} _ {i - 1}; \alpha_ {i}\right) \delta \left(\mathbf {R} \pmb {\theta} _ {i} - h \left(\mathbf {R} \pmb {\theta} _ {i - 1}, \mathbf {y} _ {i}, \alpha_ {i}\right)\right) d \mathbf {y} _ {i} \\ = \int p _ {O} \left(\mathbf {R} \mathbf {y} _ {i} ^ {\prime} \mid \mathbf {R} \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) \delta \left(\mathbf {R} \boldsymbol {\theta} _ {i} - h \left(\mathbf {R} \boldsymbol {\theta} _ {i - 1}, \mathbf {R} \mathbf {y} _ {i} ^ {\prime}, \alpha_ {i}\right)\right) d \mathbf {R} \mathbf {y} _ {i} ^ {\prime} \\ = \int p _ {O} \left(\mathbf {R} \mathbf {y} _ {i} ^ {\prime} \mid \mathbf {R} \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) \delta \left(\mathbf {R} \boldsymbol {\theta} _ {i} - h \left(\mathbf {R} \boldsymbol {\theta} _ {i - 1}, \mathbf {R} \mathbf {y} _ {i} ^ {\prime}, \alpha_ {i}\right)\right) | \det (\mathbf {R}) | d \mathbf {y} _ {i} ^ {\prime} \tag {32} \\ \end{array}
$$

The rotation matrix $\mathbf{R}$ is a SO(3) matrix, thus the $|\operatorname{det}(\mathbf{R})| = 1$ . And for the continuous coordinate variable, the update function $h$ defined in Eq. 11 is also equivariant:

$$
h \left(\mathbf {R} \boldsymbol {\theta} _ {i - 1}, \mathbf {R} \mathbf {y} _ {i}, \alpha_ {i}\right) = \frac {\mathbf {R} \boldsymbol {\theta} _ {i - 1} \rho_ {i - 1} + \mathbf {R} \mathbf {y} _ {i} \alpha_ {i}}{\rho_ {i}} = \mathbf {R} h \left(\boldsymbol {\theta} _ {i - 1}, \mathbf {y} _ {i}, \alpha_ {i}\right) \tag {33}
$$

Putting these conditions back to the Eq. 32, we have that

$$
\begin{array}{l} p _ {U} \left(\mathbf {R} \boldsymbol {\theta} _ {i} \mid \mathbf {R} \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) = \int p _ {O} \left(\mathbf {y} _ {i} \mid \mathbf {R} \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) \delta \left(\mathbf {R} \boldsymbol {\theta} _ {i} - h \left(\mathbf {R} \boldsymbol {\theta} _ {i - 1}, \mathbf {y} _ {i}, \alpha_ {i}\right)\right) d \mathbf {y} _ {i} \\ = \int p _ {O} \left(\mathbf {R} \mathbf {y} _ {i} ^ {\prime} \mid \mathbf {R} \pmb {\theta} _ {i - 1}; \alpha_ {i}\right) \delta \left(\mathbf {R} \pmb {\theta} _ {i} - \mathbf {R} h \left(\pmb {\theta} _ {i - 1}, \mathbf {y} _ {i} ^ {\prime}, \alpha_ {i}\right)\right) | \det (\mathbf {R}) | d \mathbf {y} _ {i} ^ {\prime} \\ = \int p _ {O} \left(\mathbf {y} _ {i} ^ {\prime} \mid \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) \delta \left(\boldsymbol {\theta} _ {i} - h \left(\boldsymbol {\theta} _ {i - 1}, \mathbf {y} _ {i} ^ {\prime}, \alpha_ {i}\right)\right) d \mathbf {y} _ {i} ^ {\prime} \\ = p _ {U} \left(\boldsymbol {\theta} _ {i} \mid \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) \tag {34} \\ \end{array}
$$

Hence the transitions on the $\theta$ space are the Markov and equivariant to rotation as shown in Eq. 30. The initial state $\theta_{0}$ is a zero vector 0 which is rotation invariant. To derive the rotation-invariant property of $p_{\phi}$ , we will use the following Lemma, which is the direct application of Proposition 1 in (Xu et al., 2022). We changed the notation to make it consistent with our literature.

Lemma C.2. (Xu et al., 2022) Let $p(\boldsymbol{\theta}_0)$ be an SE(3)-invariant density function, i.e., $p(\boldsymbol{\theta}_0) = p(T_g(\boldsymbol{\theta}_0))$ . If Markov transitions $p(\boldsymbol{\theta}_i \mid \boldsymbol{\theta}_{i-1})$ are SE(3)-equivariant, i.e., $p(\boldsymbol{\theta}_i \mid \boldsymbol{\theta}_{i-1}) = p(T_g(\boldsymbol{\theta}_i) \mid T_g(\boldsymbol{\theta}_{i-1}))$ , then we have that the density $p(\boldsymbol{\theta}_n) = \int p(\boldsymbol{\theta}_0)p(\boldsymbol{\theta}_{1:n} \mid \boldsymbol{\theta}_0)\mathrm{d}\boldsymbol{\theta}_{0:n}$ is also SE(3)-invariant. ( $T_g$ stands for the SE(3) transformations.)

For completeness, we also include the derivation of the lemma from (Xu et al., 2022):

$$
\begin{array}{l} p \left(T _ {g} (\boldsymbol {\theta} _ {n})\right) = \int p \left(T _ {g} (\boldsymbol {\theta} _ {0})\right) p \left(T _ {g} (\boldsymbol {\theta} _ {1: n}) \mid T _ {g} (\boldsymbol {\theta} _ {0})\right) \mathrm{d} \boldsymbol {\theta} _ {0: n} \\ = \int p \left(T _ {g} \left(\boldsymbol {\theta} _ {0}\right)\right) \Pi_ {i = 1} ^ {n} p \left(T _ {g} \left(\boldsymbol {\theta} _ {i}\right) \mid T _ {g} \left(\boldsymbol {\theta} _ {i - 1}\right)\right) \mathrm{d} \boldsymbol {\theta} _ {0: n} \\ = \int p \left(\boldsymbol {\theta} _ {0}\right) \Pi_ {i = 1} ^ {n} p _ {\theta} \left(T _ {g} \left(\boldsymbol {\theta} _ {i}\right) \mid T _ {g} \left(\boldsymbol {\theta} _ {i - 1}\right)\right) \mathrm{d} \boldsymbol {\theta} _ {0: n} \quad (\text { invariant   prior } p \left(\boldsymbol {\theta} _ {0}\right)) \tag {35} \\ = \int p \left(\boldsymbol {\theta} _ {0}\right) \Pi_ {i = 1} ^ {n} p _ {\theta} \left(\boldsymbol {\theta} _ {i} \mid \boldsymbol {\theta} _ {i - 1}\right) \mathrm{d} \boldsymbol {\theta} _ {0: n} \quad (\text { equivariant   kernels } p \left(\boldsymbol {\theta} _ {i} \mid \boldsymbol {\theta} _ {i - 1}\right)) \\ = \int p (\boldsymbol {\theta} _ {0}) p (\boldsymbol {\theta} _ {1: n} \mid \boldsymbol {\theta} _ {0}) d \boldsymbol {\theta} _ {0: n} \\ = p \left(\boldsymbol {\theta} _ {n}\right) \\ \end{array}
$$

Given the invariant property of $\pmb{\theta}_0$ and equivariant property of the transition $p_U(\pmb{\theta}_i\mid \pmb{\theta}_{i - 1};\alpha_i)$ and the $p_{\phi}(\mathbf{x}\mid \pmb{\theta}_n^x)$ , we could directly get the conclusion in Theorem. 3.1. Now we finish the proof.

# C.3 PROOF OF PROPOSITION. 3.2.

Proof. Then we derive the invariant property of the variational lower bound in of the variational lower bounds in equation 8:

$$
\mathcal {L} _ {V L B} (\mathbf {x}) = \underset {p _ {\phi} \left(\boldsymbol {\theta} _ {0} ^ {x}, \dots , \boldsymbol {\theta} _ {n} ^ {x}\right)} {\mathbb {E}} [ \sum_ {i = 1} ^ {n} D _ {K L} (p _ {S} (\cdot | \mathbf {x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i - 1} ^ {x}; \alpha_ {i})) - \log p _ {\phi} (\mathbf {x} | \boldsymbol {\theta} _ {n} ^ {x}) ]
$$

To start with, we consider the first term:

$$
\underset {p _ {\phi} \left(\boldsymbol {\theta} _ {0} ^ {x}, \dots , \boldsymbol {\theta} _ {n} ^ {x}\right)} {\mathbb {E}} \sum_ {i = 1} ^ {n} D _ {K L} \left(p _ {S} (\cdot | \mathbf {x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i} ^ {x}; \alpha_ {i})\right) \tag {36}
$$

$$
= \sum_ {i = 0} ^ {n - 1} \underset {p _ {\phi} (\boldsymbol {\theta} _ {i} ^ {x})} {\mathbb {E}} D _ {K L} (p _ {S} (\cdot | \mathbf {x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i} ^ {x}; \alpha_ {i}))
$$

Note a natural conclusion from the Theorem. 3.1 is that the SE(3) invariance property is not only satisfied in the marginal distribution of the last time step variable $p(\boldsymbol{\theta}_n)$ , but also for the distribution of any intermediate $p(\boldsymbol{\theta}_i)$ . Such property could be justified based on the condition in Lemma. C.2. Actually, the proof of Theorem. 3.1 in the above section does not specify the time steps, hence the marginal distribution of any time step could be proved in exactly the same way. Consider the $i$ -th term in the KL part of $\mathcal{L}_{\mathrm{VLB}}(\mathbf{Rx})$ :

$$
\underset {p _ {\phi} \left(\boldsymbol {\theta} _ {i} ^ {x}\right)} {\mathbb {E}} D _ {K L} \left(p _ {S} (\cdot | \mathbf {R x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i} ^ {x}; \alpha_ {i})\right) \tag {37}
$$

$$
= \int p _ {\phi} \left(\boldsymbol {\theta} _ {i} ^ {x}\right) D _ {K L} (p _ {S} (\cdot | \mathbf {R x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i} ^ {x}; \alpha_ {i})) d \boldsymbol {\theta} _ {i} ^ {x}
$$

we introduce the variable $\theta_{i}^{\prime}$ similar to Eq. 32, i.e. $\theta_{i} = \mathbf{R}\theta_{i}^{\prime}$ , and then we extend $i$ -th term in the Eq. 36:

$$
\int p _ {\phi} \left(\mathbf {R} \boldsymbol {\theta} _ {i} ^ {x ^ {\prime}}\right) D _ {K L} (p _ {S} (\cdot | \mathbf {R x}; \alpha_ {i}) \| p _ {R} (\cdot | \mathbf {R} \boldsymbol {\theta} _ {i} ^ {x ^ {\prime}}; \alpha_ {i})) | \det (\mathbf {R}) | d \boldsymbol {\theta} _ {i} ^ {x ^ {\prime}} \tag {38}
$$

As proved in the proof of Theorem. 3.1, $p_{\phi}$ is invariant and hence $p_{\phi}\left(\mathbf{R}\pmb{\theta}_i^{x'})\right) = p_{\phi}(\pmb{\theta}_i^{x'})$ ; Also the $|\det (\mathbf{R})| = 1$ for SO(3) rotation matrix. And then we discuss the KL divergence term:

$$
\begin{array}{l} D _ {K L} (p _ {S} (\cdot | \mathbf {R x}; \alpha_ {i}) \| p _ {R} (\cdot | \mathbf {R} \pmb {\theta} _ {i} ^ {x ^ {\prime}}; \alpha_ {i})) = \int p _ {S} (\mathbf {y} | \mathbf {R x}; \alpha_ {i}) \log \frac {p _ {S} (\mathbf {y} | \mathbf {R x} ; \alpha_ {i})}{p _ {R} (\mathbf {y} | \mathbf {R} \pmb {\theta} _ {i} ^ {x ^ {\prime}} ; \alpha_ {i})} d \mathbf {y} \\ = \int p _ {S} \left(\mathbf {R} \mathbf {y} ^ {\prime} \mid \mathbf {R} \mathbf {x}; \alpha_ {i}\right) \log \frac {p _ {S} \left(\mathbf {R} \mathbf {y} ^ {\prime} \mid \mathbf {R} \mathbf {x} ; \alpha_ {i}\right)}{p _ {R} (\mathbf {R} \mathbf {y} ^ {\prime} \mid \mathbf {R} \boldsymbol {\theta} _ {i} ^ {x ^ {\prime}} ; \alpha_ {i})} \mathrm{det} (\mathbf {R}) | d \mathbf {y} ^ {\prime} \\ = \int p _ {S} \left(\mathbf {y} ^ {\prime} \mid \mathbf {x}; \alpha_ {i}\right) \log \frac {p _ {S} \left(\mathbf {y} ^ {\prime} \mid \mathbf {x} ; \alpha_ {i}\right)}{p _ {R} \left(\mathbf {y} ^ {\prime} \mid \boldsymbol {\theta} _ {i} ^ {x ^ {\prime}} ; \alpha_ {i}\right)} d \mathbf {y} ^ {\prime} = D _ {K L} (p _ {S} (\cdot | \mathbf {x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i} ^ {x ^ {\prime}}; \alpha_ {i})) \tag {39} \\ \end{array}
$$

Note that $p_{S}\left(\mathbf{y}^{\prime}\mid\mathbf{x};\alpha_{i}\right)=p_{S}\left(\mathbf{R}\mathbf{y}^{\prime}\mid\mathbf{R}\mathbf{x};\alpha_{i}\right)$ is due to that the sender distribution is isotropic; And for receiver distribution, the equivariant property that $p_{R}\left(\mathbf{y}^{\prime}\mid\mathbf{x};\alpha_{i}\right)=p_{R}\left(\mathbf{R}\mathbf{y}^{\prime}\mid\mathbf{R}\mathbf{x};\alpha_{i}\right)$ is guaranteed by both the parameterization of $p_{O}$ with Equivariant Graph Neural Network and the isotropic $p_{S}$ . At last, we put the above conclusion back to Eq. 37, and we get that:

$$
\begin{array}{l} \underset {p _ {\phi} \left(\boldsymbol {\theta} _ {i} ^ {x}\right)} {\mathbb {E}} D _ {K L} (p _ {S} (\cdot | \mathbf {R x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i} ^ {x}; \alpha_ {i})) \\ = \int_ {f} p _ {\phi} \left(\boldsymbol {\theta} _ {i} ^ {x}\right) D _ {K L} \left(p _ {S} (\cdot | \mathbf {R x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i} ^ {x}; \alpha_ {i})\right) d \boldsymbol {\theta} _ {i} ^ {x} \tag {40} \\ = \int p _ {\phi} (\pmb {\theta} _ {i} ^ {x}) D _ {K L} (p _ {S} (\cdot | \mathbf {x}; \alpha_ {i}) \| p _ {R} (\cdot | \pmb {\theta} _ {i} ^ {x}; \alpha_ {i})) d \pmb {\theta} _ {i} ^ {x} \\ = \underset {p _ {\phi} (\boldsymbol {\theta} _ {i} ^ {x})} {\mathbb {E}} D _ {K L} (p _ {S} (\cdot | \mathbf {x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i} ^ {x}; \alpha_ {i})) \\ \end{array}
$$

And here we prove the first term in $\mathcal{L}_{VLB}(\mathbf{Rx})$ is equivalent to $\mathcal{L}_{VLB}(\mathbf{x})$ . The second term could be derived in exactly the same way, and here we finish the proof.

# C.4 DERIVATION OF EQUATION 8

Note that the equation 8:

$$
\mathcal {L} _ {\mathrm{VLB}} (\mathbf {x}) = \underset {p _ {\phi} \left(\boldsymbol {\theta} _ {0} ^ {x}, \dots , \boldsymbol {\theta} _ {n} ^ {x}\right)} {\mathbb {E}} \left[ \sum_ {i = 1} ^ {n} D _ {K L} (p _ {S} (\cdot | \mathbf {x}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i - 1} ^ {x}; \alpha_ {i})) - \log p _ {\phi} (\mathbf {x} | \boldsymbol {\theta} _ {n} ^ {x}) \right], \tag {41}
$$

is the extension formulation of Eq. 1. To align the notation of Eq. 1 and equation 8, we use x in the following derivation. We first consider the term $-D_{KL}(q\|p_{\phi}(\mathbf{y}_{1},\ldots,\mathbf{y}_{n}))$ in Eq. 1, we put the Eq. 2

$$
q = q \left(\mathbf {y} _ {1}, \dots , \mathbf {y} _ {n} \mid \mathbf {x}\right) = \prod_ {i = 1} ^ {n} p _ {S} \left(\mathbf {y} _ {i} \mid \mathbf {x}; \alpha_ {i}\right) \tag {42}
$$

And the $p_{\phi}(\mathbf{y}_1,\ldots ,\mathbf{y}_n)$ in Eq. 6 as

$$
\begin{array}{l} p _ {\phi} \left(\mathbf {y} _ {1}, \ldots , \mathbf {y} _ {n}\right) = \underset {p _ {\phi} (\boldsymbol {\theta} _ {0: n - 1})} {\mathbb {E}} \left[ \prod_ {i = 1} ^ {n} \underset {p _ {O} (\mathbf {x} _ {i} ^ {\prime} | \boldsymbol {\theta} _ {i - 1; \phi})} {\mathbb {E}} \left[ p _ {S} (\mathbf {y} _ {i} | \mathbf {x} _ {i} ^ {\prime}; \alpha_ {i}) \right] \right] \\ = \underset {p _ {\phi} (\boldsymbol {\theta} _ {0: n})} {\mathbb {E}} \prod_ {i = 1} ^ {n} p _ {R} \left(\mathbf {y} _ {i} \mid \boldsymbol {\theta} _ {i - 1}; \alpha_ {i}\right) \tag {43} \\ \end{array}
$$

Putting them together into the KL divergence term, and then we get the

$$
\begin{array}{l} D _ {K L} (q \| p _ {\phi} \left(\mathbf {y} _ {1}, \ldots , \mathbf {y} _ {n}\right)) = \underset {\prod_ {i = 1} ^ {n} p _ {S} (\mathbf {y} _ {i} | \mathbf {x}; \alpha_ {i})} {\mathbb {E}} \log \frac {\prod_ {i = 1} ^ {n} p _ {S} \left(\mathbf {y} _ {i} \mid \mathbf {x} ; \alpha_ {i}\right)}{\underset {p _ {\phi} (\boldsymbol {\theta} _ {0 : n - 1})} {\mathbb {E}} \prod_ {i = 1} ^ {n} p _ {R} \left(\mathbf {y} _ {i} \mid \boldsymbol {\theta} _ {i - 1} ; \alpha_ {i}\right)} \\ = \underset {p _ {\phi} (\boldsymbol {\theta} _ {0: n - 1}) \prod_ {i = 1} ^ {n}} {\mathbb {E}} \underset {p _ {S} (\mathbf {y} _ {i} | \mathbf {x}; \alpha_ {i})} {\mathbb {E}} \log \frac {\prod_ {i = 1} ^ {n} p _ {S} \left(\mathbf {y} _ {i} \mid \mathbf {x} ; \alpha_ {i}\right)}{\prod_ {i = 1} ^ {n} p _ {R} \left(\mathbf {y} _ {i} \mid \boldsymbol {\theta} _ {i - 1} ; \alpha_ {i}\right)} \tag {44} \\ = \underset {p _ {\phi} (\boldsymbol {\theta} _ {0: n - 1})} {\mathbb {E}} \sum_ {i = 1} ^ {n} D _ {K L} (p _ {S} (\cdot | \mathbf {g}; \alpha_ {i}) \| p _ {R} (\cdot | \boldsymbol {\theta} _ {i - 1}; \alpha_ {i})) \\ \end{array}
$$

And we have derived the first term in equation 8. And for the second term,

$$
\begin{array}{l} \log p _ {\phi} (\mathbf {x} | \mathbf {y} _ {1}, \dots , \mathbf {y} _ {n}) = \log p _ {\phi} (\mathbf {x} | \boldsymbol {\theta} _ {0}, \dots , \boldsymbol {\theta} _ {n}) \quad (\text { Graphical   Model   in   Fig.2 }) \\ = \log p _ {\phi} (\mathbf {x} | \boldsymbol {\theta} _ {n}) \quad (\text { Markov   Property   of } \boldsymbol {\theta}) \tag {45} \\ \end{array}
$$

And here we finish the derivation.

# D DETAILS ON CONDITIONAL GENERATION EXPERIMENTS

# D.1 PARAMETERIZATION AND SAMPLING

For the conditional experiments, we directly follow the conditional setting of previous literature (Hoogeboom et al., 2022). We discuss the details of the parameterization and sampling in the following. For conditional experiments, we add the property c as the extra input for the interdependency modeling module in Eq. 19. The conditional objective will be as:

$$
\begin{array}{l} L ^ {\infty} (\mathbf {g}, \mathbf {c}) \\ = \underset {t \sim U (0, 1), p _ {F} (\boldsymbol {\theta} ^ {g} | \mathbf {g}; t)} {\mathbb {E}} \left[ \frac {\alpha^ {x} (t)}{2} \| \mathbf {x} - \Phi_ {\mathbf {x}} \| ^ {2} + \frac {\alpha^ {\mathbf {h} _ {c}} (t)}{2} \| \mathbf {h} _ {c} - \Phi_ {\mathbf {h} _ {c}} \| ^ {2} + \frac {\alpha^ {\mathbf {h} _ {t}} (t)}{2} \| \mathbf {h} _ {t} - \Phi_ {\mathbf {h} _ {t}} \| ^ {2} \right] \tag {46} \\ \end{array}
$$

Where $\Phi. (\pmb{\theta}^g, t, \mathbf{c})$ is short for $\Phi. (\pmb{\theta}^g, t, \mathbf{c})$ .

For the sampling procedure, the property c and node number M will be firstly sampled from a prior $p(\mathbf{c}, M)$ defined in (Hoogeboom et al., 2022). Here $p(\mathbf{c}, M)$ is computed on the training partition as a parametrized two-dimensional categorical distribution where the continuous variable c is discretized into small uniformly distributed intervals. Then we could conduct generation as in Algorithm 3 based on the conditional output distribution $p_{O}(\cdot|\boldsymbol{\theta}, \mathbf{c}, t)$ base on $\Phi(\boldsymbol{\theta}, \mathbf{c}, t)$ .

# D.2 EXPLANATIONS ON THE PROPERTIES IN TAB. 2

$\alpha$ Polarizability: Tendency of a molecule to acquire an electric dipole moment when subjected to an external electric field.

$\varepsilon_{HOMO}$ : Highest occupied molecular orbital energy.

$\varepsilon_{\mathrm{LUMO}}$ : Lowest unoccupied molecular orbital energy.

$\Delta \varepsilon$ Gap: The energy difference between HOMO and LUMO.

$\mu$ : Dipole moment.

$C_v$ : Heat capacity at 298.15 K

# E DETAILED ALGORITHMS FOR TRAINING AND SAMPLING

For a better understanding of the whole procedure in training and sampling, we involve the detailed algorithms and implements of functions in Algorithm 1, Algorithm 2 and Algorithm 3.

Algorithm 1 Functions for GeoBFN   
function DISCRETISED_CDF( $\mu\inR,\sigma\inR^{+},x\inR$ ) $F(x)\leftarrow\frac{1}{2}\left[1+\mathrm{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$ $G(x)\leftarrow\begin{cases}0&\text{if }x\leq-1\\1&\text{if }x\geq1\\F(x)&\text{otherwise}\end{cases}$ Return $G(x)$ end function

function OUTPUT_PREDICTION( $\mu_{x}\inR^{D\times3},\mu_{h}\inR^{D},t\in[0,1],\gamma_{x},\gamma_{h}\in R^{+},t_{min}\in R^{+}$ )

# $t_{min}$ set to 0.0001 by default

if $t<t_{min}$ then $\hat{\mathbf{x}}(\boldsymbol{\theta},t)\leftarrow\mathbf{0}$ $\hat{\boldsymbol{\mu}}_{h}\leftarrow\mathbf{0}$ $\hat{\boldsymbol{\sigma}}_{h}\leftarrow\mathbf{1}$ else

Input $(\boldsymbol{\mu}_{x},\boldsymbol{\mu}_{h},t)$ to network, receive $\hat{\epsilon}(\boldsymbol{\theta},t),\hat{\boldsymbol{\mu}}_{h}^{\epsilon},\ln\hat{\boldsymbol{\sigma}}_{h}^{\epsilon}$ as output $\hat{\mathbf{x}}(\boldsymbol{\theta},t)\leftarrow\frac{\boldsymbol{\mu}_{x}}{\gamma_{x}}-\sqrt{\frac{1-\gamma_{x}}{\gamma_{x}}}\hat{\epsilon}(\boldsymbol{\theta},t)$ $\hat{\boldsymbol{\mu}}_{h}\leftarrow\frac{\hat{\boldsymbol{\mu}}_{h}}{\gamma_{h}}-\sqrt{\frac{1-\gamma_{h}}{\gamma_{h}}}\hat{\boldsymbol{\mu}}_{h}^{\epsilon}$ $\hat{\boldsymbol{\sigma}}_{h}\leftarrow\sqrt{\frac{1-\gamma_{h}}{\gamma_{h}}}\ln\hat{\boldsymbol{\sigma}}_{h}^{\epsilon}$ end if

for $d\in1,\cdots,D,k\in K$ do $p_{O}^{(d)}(k|\boldsymbol{\theta};t)\leftarrow\text{DISCRETISED\_CDF}(\hat{\mu}_{h}^{(d)},\hat{\sigma}_{h}^{(d)},k_{r})-\text{DISCRETISED\_CDF}(\hat{\mu}_{h}^{(d)},\hat{\sigma}_{h}^{(d)},k_{l})$ end for

Return $\hat{x}(\boldsymbol{\theta},t),p_{O}(\cdot|\boldsymbol{\theta};t)$ end function

Algorithm 2 Training with continuous loss   
Require: $\sigma_x, \sigma_h \in \mathbb{R}$ , number of bins $K \in \mathbb{N}$ Input: coordinates $\boldsymbol{x} \in \mathbb{R}^{D \times 3}$ , normalized charges $\boldsymbol{h} \in [\frac{1}{K} - 1, 1 - \frac{1}{K}]^D$ $t \sim U(0,1)$ $\gamma_x \leftarrow 1 - \sigma_x^{2t}, \gamma_h \leftarrow 1 - \sigma_h^{2t}$ $\boldsymbol{\mu}_x \sim \mathcal{N}(\gamma_x, \gamma_x(1 - \gamma_x)\boldsymbol{I})$ $\boldsymbol{\mu}_h \sim \mathcal{N}(\gamma_h, \gamma_h(1 - \gamma_h)\boldsymbol{I})$ $\hat{\boldsymbol{x}}(\boldsymbol{\theta}, t), \boldsymbol{p}_O(\cdot | \boldsymbol{\theta}; t) \leftarrow \text{OUTPUT\_PREDICTION}(\boldsymbol{\mu}_x, \boldsymbol{\mu}_h, t, \gamma_x, \gamma_h)$ $\hat{\boldsymbol{k}}(\boldsymbol{\theta}, t) \leftarrow \left( \sum_k \boldsymbol{p}_O^{(1)} \boldsymbol{p}_O(k | \boldsymbol{\theta}; t) k_c, \ldots, \sum_k \boldsymbol{p}_O^{(D)}(k | \boldsymbol{\theta}; t) k_c \right)$ $L^\infty(\boldsymbol{x}) \leftarrow -\ln \sigma_x \sigma_x^{-2t} \| \boldsymbol{x} - \hat{\boldsymbol{x}}(\boldsymbol{\theta}, t) \|^2$ $L^\infty(\boldsymbol{h}) \leftarrow -\ln \sigma_h \sigma_h^{-2t} \| \boldsymbol{h} - \hat{\boldsymbol{k}}(\boldsymbol{\theta}, t) \|^2$ Return $L^\infty(\boldsymbol{x}) + L^\infty(\boldsymbol{h})$

Algorithm 3 Sampling procedure   
$\# \vec{k}_c = \left(k_c^{(1)}, \ldots, k_c^{(D)}\right)$ $\#$ Function NEAREST_CENTER compares inputs to the center bins $\vec{k}_c$ , $\#$ and return the nearest center for each input value.

Require: $\sigma_x, \sigma_h \in \mathbb{R}^+$ , number of steps $N \in \mathbb{N}$ $\boldsymbol{\mu}_x, \boldsymbol{\mu}_h \gets \boldsymbol{0}$ $\rho_x, \rho_h \gets 1$ for $i = 1$ to $N$ do $t \gets \frac{i-1}{n}$ $\gamma_x \gets 1 - \sigma_x^{2t}, \gamma_h \gets 1 - \sigma_h^{2t}$ $\hat{\boldsymbol{x}}(\boldsymbol{\theta}, t), \boldsymbol{p}_O(\cdot | \boldsymbol{\theta}; t) \gets \text{OUTPUT\_PREDICTION}(\boldsymbol{\mu}_x, \boldsymbol{\mu}_h, t, \gamma_x, \gamma_h)$ $\alpha_x \gets \sigma_x^{-2i/n} \left(1 - \sigma_x^{2/n}\right)$ $\alpha_h \gets \sigma_h^{-2i/n} \left(1 - \sigma_h^{2/n}\right)$ $\hat{\boldsymbol{k}}_c(\boldsymbol{\theta}, t) \gets \text{NEAREST\_CENTER}(\left[\sum_k \boldsymbol{p}_O^{(1)}(k | \boldsymbol{\theta}; t) k_c, \ldots, \sum_k \boldsymbol{p}_O^{(D)}(k | \boldsymbol{\theta}; t) k_c\right])$ $\boldsymbol{y}_h \sim \mathcal{N}(\hat{\boldsymbol{k}}_c(\boldsymbol{\theta}, t), \alpha_h^{-1} \boldsymbol{I})$ $\boldsymbol{y}_x \sim \mathcal{N}(\hat{\boldsymbol{x}}(\boldsymbol{\theta}, t), \alpha_x^{-1} \boldsymbol{I})$ $\boldsymbol{\mu}_x, \boldsymbol{\mu}_h \gets \frac{\rho_x \boldsymbol{\mu}_x + \alpha_x \boldsymbol{y}_x}{\rho_x + \alpha_x}, \frac{\rho_h \boldsymbol{\mu}_h + \alpha_h \boldsymbol{y}_h}{\rho_h + \alpha_h}$ $\rho_x, \rho_h \gets (\rho_x + \alpha_x), (\rho_h + \alpha_h)$ end for $\hat{\boldsymbol{x}}(\boldsymbol{\theta}, 1), \boldsymbol{p}_O(\cdot | \boldsymbol{\theta}; 1) \gets \text{OUTPUT\_PREDICTION}(\boldsymbol{\mu}_x, \boldsymbol{\mu}_h, 1, 1 - \sigma_x^2, 1 - \sigma_h^2)$ $\hat{\boldsymbol{k}}_c(\boldsymbol{\theta}, 1) \gets \text{NEAREST\_CENTER}(\left[\sum_k \boldsymbol{p}_O^{(1)}(k | \boldsymbol{\theta}; 1) k_c, \ldots, \sum_k \boldsymbol{p}_O^{(D)}(k | \boldsymbol{\theta}; 1) k_c\right])$ Return $\hat{\boldsymbol{x}}(\boldsymbol{\theta}, 1), \hat{\boldsymbol{k}}_c(\boldsymbol{\theta}, 1)$