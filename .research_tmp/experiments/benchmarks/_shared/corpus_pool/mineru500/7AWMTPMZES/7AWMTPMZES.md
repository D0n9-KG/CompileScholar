# Discrete Modeling via Boundary Conditional Diffusion Processes

Yuxuan Gu $^{\dagger}$ Xiaocheng Feng $^{\dagger\ddagger}$ Lei Huang $^{\dagger}$ Yingsheng Wu $^{\dagger}$ Zekun Zhou $^{\dagger}$ Weihong Zhong $^{\dagger}$ Kun Zhu $^{\dagger}$ Bing Qin $^{\dagger\ddagger}$

$^{\dagger}$ Harbin Institute of Technology $^{\ddagger}$ Peng Cheng Laboratory

{yxgu, xcfeng, lhuang, yswu, zkzhou, whzhong, kzhu, qinb}@ir.hit.edu.cn

# Abstract

We present an novel framework for efficiently and effectively extending the powerful continuous diffusion processes to discrete modeling. Previous approaches have suffered from the discrepancy between discrete data and continuous modeling. Our study reveals that the absence of guidance from discrete boundaries in learning probability contours is one of the main reasons. To address this issue, we propose a two-step forward process that first estimates the boundary as a prior distribution and then rescales the forward trajectory to construct a boundary conditional diffusion model. The reverse process is proportionally adjusted to guarantee that the learned contours yield more precise discrete data. Experimental results indicate that our approach achieves strong performance in both language modeling and discrete image generation tasks. In language modeling, our approach surpasses previous state-of-the-art continuous diffusion language models in three translation tasks and a summarization task, while also demonstrating competitive performance compared to auto-regressive transformers. Moreover, our method achieves comparable results to continuous diffusion models when using discrete ordinal pixels and establishes a new state-of-the-art for categorical image generation on the CIFAR-10 dataset.

# 1 Introduction

Discrete modeling is essential due to the natural prevalence of discreteness in numerous domains, including proteins [Madani et al., 2020, 2023], images [Parmar et al., 2018, Dosovitskiy et al., 2021], and natural language [Sutskever et al., 2014, Brown et al., 2020]. Recent dominant framework for discrete modeling is the Transformer [Vaswani et al., 2017] with an autoregressive manner. While achieving impressive performance, it does suffer from a slow step-by-step generation process, especially for long sequences. Continuous Diffusion models [Sohl-Dickstein et al., 2015, Ho et al., 2020], on the contrary, exhibit the ability to recover high-dimensional data from noise in parallel with limited iteration steps. Although proved to be effective in continuous data generation [Rombach et al., 2022, Kong et al., 2021], they continue to encounter challenges in discrete modeling [Austin et al., 2021, Chen et al., 2023b, Li et al., 2022, Gong et al., 2023b].

In this paper, we reveal a significant discrepancy pertaining to the modeling of discrete data using continuous diffusion models. Current approaches represent a discrete sample with a vector point in the continuous space. The diffusion process learns a neural network to model the probability distributions that recovers this continuous point from Gaussian noise. However, the discrete data actually corresponds to an area in the continuous space rather than a single point, where the oversimplified assumption leads to a mismatch between learned probability contours and the boundary of the discrete area. Take language generation as an example, a word is represented with an embedding vector in the embedding space. To generate this word, it is impractical to strictly enforce the predicted vector to be an exact match to the embedding. On the contrary, vectors around this embedding can also generate

the same word, thereby defining the collective area they encompass as a discrete area of this word. As illustrated in Figure 1A, suppose the learned probability density function is $p_{\theta}(\mathbf{x})$ and two points $x^{i}$ and $x^{o}$ are sampled in the same density contour where $p_{\theta}(\mathbf{x}^{i}) = p_{\theta}(\mathbf{x}^{o})$ . It is obvious that $x^{i}$ lies in the discrete area and is able to recover the discrete data while $x^{o}$ can not. This means that the diffusion model only learns a simplified scenario that does not match the real probability distribution.

To address the issues above, we proposed to take the boundaries of discrete areas as priors, as shown in Figure 1B, where boundary curves are regarded as oracle contours. As it gradually approaches the discrete boundary, the learned density contours of diffusion models are expected to transform from Gaussian distributions to the boundary distribution. Therefore, we propose to divide the forward process into two steps. First is the boundary estimation where we precisely calculate the stopping time $t_{0}$ and position $x_{t_{0}}$ at which the forward trajectory cross the boundary. Then we rescale the trajectory for both training and inference stages to make the sampling probability of noisy point $x_{t}$ conditioned on the boundary. To make the boundary estimation tractable (appendix A) and eliminate randomness in conditional state transitions $x_{t_{0}} \rightarrow$ $x_{t}$ , we utilize the Ordinary Differential Equations (ODEs) to describe the forward trajectory.

![](images/f5504799dd81a4998f43dda5741c6b4566eedf2ad571d3188fe7b43ec1c2e161.jpg)

<details>
<summary>text_image</summary>

(A)
x^i
x^o
∇p(x|x_0)
∇pθ(x)
Discrete Area ←
x_0 ←
</details>

![](images/9bc974817200419da80750ab8a1eb351d9e6ebe754574cd8a3939ca97c95c39c.jpg)

<details>
<summary>text_image</summary>

(B)
</details>

Figure 1: (A) Blue and green curves are the learned probability density contours of the diffusion model for two data points. The red area is the discrete area of the blue data $x_{0}$ and the boundary of this area is naturally a density contour. The discrete boundary is a complex hypersurface in the high-dimensional continuous space and we simplify it into a red line for convenience of description. As observed in the magnified part, the learned contours deviate from the boundary contour, resulting in inconsistent probability densities and gradient directions. (B) We consider the discrete boundary as priors for the diffusion process to estimate a more appropriate probability distribution, where the learned contours are expected to follow the shape of the discrete boundary.

Our approach is experimented in both language modeling and discrete image generation. On three machine translation datasets (IWSLT14 DE-EN [Cettolo et al., 2012], WMT14 EN-DE, WMT16 EN-RO) and a text summarization dataset (GIGAWORD [Rush et al., 2015]) for language modeling, our proposed approach not only significantly improves existing diffusion models to at most $7.8\%$ but also achieves competitive performance to autoregressive transformers. For image generation on CIFAR-10 [Krizhevsky et al., 2009], our model realizes a comparable result to continuous diffusion models with discrete ordinal pixels and establishes a new state-of-the-art for categorical pixels.

# 2 Preliminaries

Diffusion Models To model a real distribution $q(\mathbf{x}_{0})$ , diffusion models utilize a forward process $p_{t}(\mathbf{x}|\mathbf{x}_{0})$ with T steps to gradually add Gaussian noise $\pi(\mathbf{x}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ into the data distribution, where $p_{T}(\mathbf{x}|\mathbf{x}_{0}) = \pi(\mathbf{x})$ . There are different architectures for the forward process. A common approach [Ho et al., 2020] considers the forward process as the Markovian process, where $p_{t}(\mathbf{x}|\mathbf{x}_{0}) = \prod_{s=1}^{t} p_{s}(\mathbf{x}_{s}|\mathbf{x}_{s-1})$ combines a series of Gaussian distributions. Thus the forward process follows a Gaussian distribution that $p_{t}(\mathbf{x}|\mathbf{x}_{0}) = \mathcal{N}(\sqrt{\bar{\alpha}_{t}}\mathbf{x}_{0}, (1 - \bar{\alpha}_{t})\mathbf{I})$ (Variance Preserving) or $p_{t}(\mathbf{x}|\mathbf{x}_{0}) = \mathcal{N}(\mathbf{x}_{0}, \sigma_{t}^{2}\mathbf{I})$ (Variance Exploding) [Song et al., 2021b], where noise scheduler $\bar{\alpha}_{t}$ monotonically decreases from 1 to 0 and $\sigma_{t}$ increases from sufficiently small to the maximum pairwise distance between all training data points. To recover data from noise, diffusion processes train neural networks $\mathbf{x}_{\theta}(\mathbf{x}_{t}, t)$ to predict $x_{0}$ (other equivalent targets include $\epsilon$ and $\nabla \log p(\mathbf{x}_{t})$ ) from $x_{t} \sim p_{t}(\mathbf{x}|\mathbf{x}_{0})$ :

$$
\mathcal {L} _ {\theta} = \mathbb {E} _ {t \sim \mathcal {U} _ {(1, T)}, \mathbf {x} _ {0} \sim q (\mathbf {x} _ {0}), \mathbf {x} _ {t} \sim p _ {t} (\mathbf {x} | \mathbf {x} _ {0})} \left[ \| \mathbf {x} _ {0} - \mathbf {x} _ {\theta} (\mathbf {x} _ {t}, t) \| ^ {2} \right]. \tag {1}
$$

Samples are generated with a series of reverse state transition $p(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_\theta(\mathbf{x}_t, t))$ .

Flow Matching Another architecture [Lipman et al., 2023] utilizes the ODEs and defines a time-dependent flow function $\phi_t(\mathbf{x}) = \sigma_t(\mathbf{x}_0)\mathbf{x} + \mu_t(\mathbf{x}_0)$ that maps $p_t(\mathbf{x}|\mathbf{x}_0) = [\phi_t]_*\pi (\mathbf{x}) = \pi (\phi_t^{-1}(\mathbf{x}))\left|\det \frac{\mathrm{d}\phi_t^{-1}(\mathbf{x})}{\mathrm{d}\mathbf{x}}\right| = \mathcal{N}(\mu_t(\mathbf{x}_0),\sigma_t^2 (\mathbf{x}_0)\mathbf{I})$ , where $\mu_{t}$ and $\sigma_{t}$ can be the same as in diffusion

![](images/4c1fbb13d9c7616e397a04ecf137a97bc5108c16cdca52259285a28929e6a2ad.jpg)

<details>
<summary>natural_image</summary>

Abstract diagram with concentric circles and radial lines, no text or symbols present
</details>

$t = 0$ (A) Rescaled Probability Contours

![](images/44ab6c9f7a4a01284c7923c497f44addf35f540a1dd96ae11d344da2be737c82.jpg)

<details>
<summary>natural_image</summary>

Two concentric circular field patterns with directional arrows, no text or symbols present
</details>

$t = T / 2$

![](images/fd76a13c55a2a59420fa630ec0fd2fb73277ff401a3f1d790182f6dd0a3dc006.jpg)

<details>
<summary>text_image</summary>

x₁
</details>

$t = T$

![](images/5298a8e9f0215944d317af67d4f96f9bd33f539a3cf0d7de549651adbfe09aa7.jpg)

<details>
<summary>text_image</summary>

Discrete Area of x₀
x₀ → x₁
x₂ → x̂₁
x₃ → ε → x̂_γ
Caucasian Distribution
</details>

(B) Rescaled Forward Trajectory   
Figure 2: (A) Rescaled Probability Contours. The bold curve $1\sigma$ is the density contour of one standard deviation. As the time $t$ decreases from $T$ to 0, the rescaled contours will gradually fit the discrete boundary and probability densities will also concentrate to this boundary. (B) Rescaled Forward Trajectory. Original forward trajectory $\mathbf{x}_0 \to \mathbf{x}_{t_0} \to \mathbf{x}_\tau$ is rescaled to be a boundary conditional trajectory $\tilde{\mathbf{x}}_1 \to \tilde{\mathbf{x}}_t$ that starts from $\tilde{\mathbf{x}}_1 = \mathbf{x}_{t_0}$ . The rescaled forward distribution $\tilde{p}_t(\tilde{\mathbf{x}}_t | \mathbf{x}_0)$ is transformed from the discrete boundary to Gaussian distributions.

models or a more straightforward form that $\mu_t = (1 - \frac{t}{T})\mathbf{x}_0$ and $\sigma_t = \frac{t}{T}$ . Recovering data from noises relies on the vector field $u_t(\mathbf{x}|\mathbf{x}_0)$ that generates the probability path with the ODE $\mathrm{d}\phi_{T - t}(\mathbf{x}) = u_{T - t}(\phi_{T - t}(\mathbf{x})|\mathbf{x}_0)\mathrm{d}t, t:0\to T$ . Neural networks $u_{\theta}(\mathbf{x},t)$ are trained to estimate the vector field $u_t(\mathbf{x}|\mathbf{x}_0)$ via the following objective:

$$
\mathcal {L} _ {\theta} = \mathbb {E} _ {t \sim \mathcal {U} _ {(1, T)}, \mathbf {x} _ {0} \sim q (\mathbf {x} _ {0}), \mathbf {x} _ {T} \sim \pi (\mathbf {x})} \left[ \left\| u _ {\theta} \left(\phi_ {t} \left(\mathbf {x} _ {T}\right), t\right) - \frac {\mathrm{d} \phi_ {t} \left(\mathbf {x} _ {T}\right)}{\mathrm{d} t} \right\| ^ {2} \right]. \tag {2}
$$

Besides, the vector field is proved to have the form:

$$
u _ {t} (\mathbf {x} \mid \mathbf {x} _ {0}) = \frac {\sigma_ {t} ^ {\prime} \left(\mathbf {x} _ {0}\right)}{\sigma_ {t} \left(\mathbf {x} _ {0}\right)} \left(\mathbf {x} - \mu_ {t} \left(\mathbf {x} _ {0}\right)\right) + \mu_ {t} ^ {\prime} \left(\mathbf {x} _ {0}\right), \text {   where   apostrophe   indicates   derivative   to   } t. \tag {3}
$$

# 3 Methodology

As illustrated in Figure 2, our objective is to refine the probability density contours of $p_t(\mathbf{x}|\mathbf{x}_0)$ so that they better fit the boundaries of discrete samples while still allowing for the ease of sampling. Let $\mathbf{x}_0$ denote the samples from a real distribution $q(\mathbf{x}_0)$ . Obtaining a boundary-aware corresponding noisy data $\mathbf{x}$ at time $t \in [1,T]$ is $p_t(\mathbf{x}|\mathbf{x}_0) = \int p_t(\mathbf{x},\mathbf{x}_{t_0},t_0|\mathbf{x}_0)\mathrm{d}\mathbf{x}_{t_0}\mathrm{d}t_0$ , where $t_0$ is a random variable distributed according to when the diffusion trajectory and the discrete boundary intersect, and $\mathbf{x}_{t_0}$ is the corresponding sample point at $t_0$ . Then the forward process is rescaled in two steps:

$$
\tilde {p} _ {t} (\mathbf {x} | \mathbf {x} _ {0}) = \int \underbrace {\tilde {p} _ {t} (\mathbf {x} | \mathbf {x} _ {t _ {0}} , t _ {0} , \mathbf {x} _ {0})} _ {\text { Trajectory   Rescaling }} \underbrace {p \left(\mathbf {x} _ {t _ {0}} , t _ {0} \mid \mathbf {x} _ {0}\right)} _ {\text { Boundary   Estimation }} \mathrm{d} \mathbf {x} _ {t _ {0}} \mathrm{d} t _ {0}, \tag {4}
$$

where the latter term is to calculate the discrete boundaries and the former term is to rescale the forward trajectory. In order to make the equation tractable and ensure that x and $x_{t_{0}}$ are on the same trajectory, we model the forward process with flow functions $\phi_{t}(\mathbf{x})$ and extend the notation as:

$$
\psi_ {t} (\mathbf {x}) = \mathbf {u} \left(\mathbf {x} _ {0}, t\right) \mathbf {x} _ {0} + \mathbf {v} \left(\mathbf {x} _ {0}, t\right) \mathbf {x}, \quad p _ {t} (\mathbf {x} | \mathbf {x} _ {0}) = \left[ \psi_ {t} \right] _ {*} \pi (\mathbf {x}) \tag {5}
$$

where $\mathbf{u}(\cdot)$ and $\mathbf{v}(\cdot)$ are coefficient functions and sampling $x_{t}$ from $p_{t}(\mathbf{x}|\mathbf{x}_{0})$ equals to

$$
\mathbf {x} _ {t} = \psi_ {t} (\boldsymbol {\epsilon}), \quad \boldsymbol {\epsilon} \sim \pi (\mathbf {x}) = \mathcal {N} (\mathbf {0}, \mathbf {I}). \tag {6}
$$

# 3.1 Estimate Discrete Boundaries

Before figuring out the joint distribution $p(\mathbf{x}_{t_0}, t_0 | \mathbf{x}_0)$ , let's start by discussing how to verify whether an arbitrary point $\mathbf{x}$ in the continuous space belongs to the discrete area of $\mathbf{x}_0$ . Suppose $\mathbf{x}_0$ , which exists in the continuous space $S$ , is the representation vector of a discrete random variable $\mathcal{I}$ in a discrete space with $K$ states. Besides, $\mathcal{J}$ is another discrete random variable i.i.d. with $\mathcal{I}$ . We define the discrete area of $\mathbf{x}_0$ in the continuous space $S$ as:

$$
C _ {\mathcal {I}} = \{\forall \mathbf {x} \in S | f (\mathbf {x}, \mathcal {I}) > f (\mathbf {x}, \mathcal {J}), \forall \mathcal {J} \neq \mathcal {I} \}, \tag {7}
$$

where $f(\mathbf{x},\mathcal{I})$ is a function assessing the likelihood of an arbitrary continuous point x inside the discrete area of $x_{0}$ . For instance, in language modeling, K is the vocabulary size. $I, J \in K^{n}$ are two different sequences of n tokens and $x_{0} \in R^{[n,m]}$ is a sequence of m-dimensional vector embeddings for I. $f(\mathbf{x},\mathcal{I})$ is the dot similarity function. $C_{I}$ collects all vectors in the embedding space that will be decoded to generate I and excludes vectors associated with any other token sequences J.

Given a noisy point $x_{t_{0}}$ locating at the boundary between $C_{I}$ and $C_{J}$ , we can get $|f(\mathbf{x}_{t_{0}},\mathcal{I}) - f(\mathbf{x}_{t_{0}},\mathcal{J})| = 0$ based on previous definition. Replacing $x_{t_{0}}$ with eqs. (5) and (6), there is:

$$
f (\mathbf {u} _ {t _ {0}} \mathbf {x} _ {0} + \mathbf {v} _ {t _ {0}} \boldsymbol {\epsilon}, \mathcal {I}) = f (\mathbf {u} _ {t _ {0}} \mathbf {x} _ {0} + \mathbf {v} _ {t _ {0}} \boldsymbol {\epsilon}, \mathcal {J}). \tag {8}
$$

In language modeling and categorical images, $f(\cdot)$ is a linear projection function that:

$$
\mathbf {u} _ {t _ {0}} (f (\mathbf {x} _ {0}, \mathcal {I}) - f (\mathbf {x} _ {0}, \mathcal {J})) = \mathbf {v} _ {t _ {0}} (f (\boldsymbol {\epsilon}, \mathcal {J}) - f (\boldsymbol {\epsilon}, \mathcal {I})). \tag {9}
$$

Further simplification of this equation can not be universally applied to all arbitrary forms of $\mathbf{u}_{t_0}$ and $\mathbf{v}_{t_0}$ . Therefore, we calculate separately for several commonly occurring special cases.

Diffusion Process For variance preserving, there is $u_{t}^{2} + v_{t}^{2} = 1$ and we have:

$$
\mathbf {u} _ {t _ {0}} = 1 / \sqrt {1 + \left(\frac {f (\mathbf {x} _ {0} , \mathcal {I}) - f (\mathbf {x} _ {0} , \mathcal {J})}{f (\boldsymbol {\epsilon} , \mathcal {J}) - f (\boldsymbol {\epsilon} , \mathcal {I})}\right) ^ {2}} \text {   and   } \mathbf {v} _ {t _ {0}} = 1 / \sqrt {1 + \left(\frac {f (\boldsymbol {\epsilon} , \mathcal {J}) - f (\boldsymbol {\epsilon} , \mathcal {I})}{f (\mathbf {x} _ {0} , \mathcal {I}) - f (\mathbf {x} _ {0} , \mathcal {J})}\right) ^ {2}}. \tag {10}
$$

For variance exploding, there are $u_{t}=1$ and $v_{t}=\sigma_{t}$ . We can obtain:

$$
\mathbf {u} _ {t _ {0}} = 1 \text {   and   } \mathbf {v} _ {t _ {0}} = (f (\boldsymbol {\epsilon}, \mathcal {J}) - f (\boldsymbol {\epsilon}, \mathcal {I})) / (f (\mathbf {x} _ {0}, \mathcal {I}) - f (\mathbf {x} _ {0}, \mathcal {J})). \tag {11}
$$

Flow Matching For optimal transport, there is $u_{t} + v_{t} = 1$ and similarly we get:

$$
\mathbf {u} _ {t _ {0}} = 1 / \left(1 + \frac {f (\mathbf {x} _ {0} , \mathcal {I}) - f (\mathbf {x} _ {0} , \mathcal {J})}{f (\boldsymbol {\epsilon} , \mathcal {J}) - f (\boldsymbol {\epsilon} , \mathcal {I})}\right) \text {   and   } \mathbf {v} _ {t _ {0}} = 1 / \left(1 + \frac {f (\boldsymbol {\epsilon} , \mathcal {J}) - f (\boldsymbol {\epsilon} , \mathcal {I})}{f (\mathbf {x} _ {0} , \mathcal {I}) - f (\mathbf {x} _ {0} , \mathcal {J})}\right). \tag {12}
$$

As a result, $t_{0}$ can be directly derived by inverting the coefficient function $u_{t}$ or $v_{t}$ , which depends on the choice of noise scheduling strategies. Since their differences do not affect our results, we omit the detailed calculation (appendix E) and denote this process with a function $G(\cdot)$ :

$$
t _ {0} = G \left(\mathbf {x} _ {0}, \epsilon\right), \text {   where   } \mathbf {u} \left(\mathbf {x} _ {0}, G \left(\mathbf {x} _ {0}, \epsilon\right)\right) = \mathbf {u} _ {t _ {0}} \text {   and   } \mathbf {v} \left(\mathbf {x} _ {0}, G \left(\mathbf {x} _ {0}, \epsilon\right)\right) = \mathbf {v} _ {t _ {0}}. \tag {13}
$$

It's worth noting that $t_0$ is not a scalar but a vector, where the dimension is the number of elements in $\mathbf{x}_0$ . If $\mathbf{x}_0$ is a sequence of $n$ tokens, $t_0 \in [1,T]^n$ . If $\mathbf{x}_0$ is a RGB image with 3-channel $\times h$ -height $\times w$ -width of pixels, $t_0 \in [1,T]^{3 \times h \times w}$ . Furthermore, the corresponding noisy sample $\mathbf{x}_{t_0}$ is derived as:

$$
\mathbf {x} _ {t _ {0}} = \mathbf {u} (\mathbf {x} _ {0}, G (\mathbf {x} _ {0}, \boldsymbol {\epsilon})) \mathbf {x} _ {0} + \mathbf {v} (\mathbf {x} _ {0}, G (\mathbf {x} _ {0}, \boldsymbol {\epsilon})) \boldsymbol {\epsilon} = \psi_ {G (\mathbf {x} _ {0}, \boldsymbol {\epsilon})} (\boldsymbol {\epsilon}), \tag {14}
$$

which is a time-independent function of the Gaussian noise $\epsilon$ . It's worth mentioning that both $p(t_0|\mathbf{x}_0)$ and $p(\mathbf{x}_{t_0}|\mathbf{x}_0)$ are intractable, since $G(\mathbf{x}_0,\epsilon)$ and $\psi_{G(\mathbf{x}_0,\epsilon)}(\epsilon)$ are not invertible to $\epsilon$ . Different $\epsilon$ s can be mapped to a same $t_0$ or $\mathbf{x}_{t_0}$ . Fortunately, there is an one-to-one mapping between $\epsilon$ and the $[\mathbf{x}_{t_0};t_0]$ pair. We denote the boundary flow function and the corresponding inversion as

$$
\Psi (\boldsymbol {\epsilon}) = \left[ \psi_ {G \left(\mathbf {x} _ {0}, \boldsymbol {\epsilon}\right)} (\boldsymbol {\epsilon}); G \left(\mathbf {x} _ {0}, \boldsymbol {\epsilon}\right) \right], \quad \Psi^ {- 1} \left(\left[ \mathbf {x} _ {t _ {0}}; t _ {0} \right]\right) = \left(\mathbf {x} _ {t _ {0}} - \mathbf {u} \left(\mathbf {x} _ {0}, t _ {0}\right) \mathbf {x} _ {0}\right) / \mathbf {v} \left(\mathbf {x} _ {0}, t _ {0}\right), \tag {15}
$$

and the joint boundary distribution is calculated as

$$
p (\mathbf {x} _ {t _ {0}}, t _ {0} | \mathbf {x} _ {0}) = [ \Psi ] _ {*} \pi ([ \mathbf {x} _ {t _ {0}}; t _ {0} ]). \tag {16}
$$

The support set of $x_{t_{0}}$ is restricted to the boundary contour, while other regions in the space are assigned a probability of 0. To obtain the complete boundary, it is necessary to iterate over all possible choices of J and perform pairwise comparisons with I. The complexity is $O(n \times K)$ , where n elements in $x_{0}$ is independently iterated. In practical implementation, obtaining the tightest boundary only requires one step of parallel calculation and an extra $\min(\cdot)$ function over all $t_{0}$ candidates.

Confidence Factor The discrete area defined by eq. (7) represents an ideal scenario in which the confidence of the boundary is insufficiently reliable for practical application. Due to the intractability of obtaining the probability density function across the entire discrete area and calculating its confidence interval, we employ an empirical strategy. This approach involves utilizing a confidence factor, denoted as r, ranging from 0 to 1, which is multiplied by $t_{0}$ to strike a balance between confidence and discreteness. Therefore, r = 0 implies the exclusion of discrete priors, causing the discrete area to collapse into a single point, which is the original diffusion process. As the value of r increases, the modeling of discrete boundaries improves at the expense of reliability. Empirically, when the model is conditioned with good guidance, setting a larger value for r allows us to obtain better discrete priors. However, in the case of unconditional modeling, maintaining reliability becomes more crucial to prevent oscillations and even collapses during training.

# 3.2 Rescale the Forward Trajectory

In this section, we introduce how to formulate the forward trajectory conditioned on discrete boundaries and derive the rescaled noisy sampling distribution. We start with the boundary-independent forward process $p_{t}(\mathbf{x}|\mathbf{x}_{0})$ . Let $x_{t}$ denote a noisy point at time t sampled from $p_{t}(\mathbf{x}|\mathbf{x}_{0})$ , there is $\boldsymbol{\epsilon}_{t} = (\mathbf{x}_{t} - \mathbf{u}(\mathbf{x}_{0}, t)\mathbf{x}_{0}) / \mathbf{v}(\mathbf{x}_{0}, t)$ given eq. (5). Equations (13) and (14) provide the corresponding $[x_{t_{0}}; t_{0}]$ pair on the same trajectory, which is deterministically calculated with no randomness:

$$
\left[ \mathbf {x} _ {t _ {0}}; t _ {0} \right] = \Psi (\boldsymbol {\epsilon} _ {t}), \text {   where   } \boldsymbol {\epsilon} _ {t} = \left(\mathbf {x} _ {t} - \mathbf {u} (\mathbf {x} _ {0}, t) \mathbf {x} _ {0}\right) / \mathbf {v} (\mathbf {x} _ {0}, t). \tag {17}
$$

To model the transition probability $p_t(\mathbf{x}_{t_0}, t_0 | \mathbf{x}_t, \mathbf{x}_0)$ , we utilize the Dirac delta function $\delta(\mathbf{x}) \simeq \lim_{\sigma \to 0} \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ , which can be loosely thought of as aggregating all probability densities toward the origin, assigning an infinite density at the origin and zero densities elsewhere. Therefore, we have $p_t(\mathbf{x}_{t_0}, t_0 | \mathbf{x}_t, \mathbf{x}_0) = \delta([ \mathbf{x}_{t_0}; t_0] - \Psi(\boldsymbol{\epsilon}_t))$ . Then the forward process, conditioned on the discrete boundary, is simply derived via Bayes' rule:

$$
p _ {t} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {t _ {0}}, t _ {0}, \mathbf {x} _ {0}\right) = p _ {t} \left(\mathbf {x} _ {t _ {0}}, t _ {0} \mid \mathbf {x} _ {t}, \mathbf {x} _ {0}\right) \frac {p _ {t} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right)}{p \left(\mathbf {x} _ {t _ {0}} , t _ {0} \mid \mathbf {x} _ {0}\right)} = \left\{ \begin{array}{c} 0, [ \mathbf {x} _ {t _ {0}}; t _ {0} ] \neq \Psi (\boldsymbol {\epsilon} _ {t}) \\ + \infty \times \frac {p _ {t} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right)}{p \left(\mathbf {x} _ {t _ {0}} , t _ {0} \mid \mathbf {x} _ {0}\right)}, \quad \text {otherwise}. \end{array} \right. \tag {18}
$$

Since $p_t(\mathbf{x}_t|\mathbf{x}_0) > 0$ and $p(\mathbf{x}_{t_0}, t_0|\mathbf{x}_0) > 0$ , $p_t(\mathbf{x}_t|\mathbf{x}_{t_0}, t_0, \mathbf{x}_0)$ is also a delta function that

$$
p _ {t} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {t _ {0}}, t _ {0}, \mathbf {x} _ {0}\right) = \delta \left(\mathbf {x} _ {t} - \mathbf {u} \left(\mathbf {x} _ {0}, t\right) \mathbf {x} _ {0} - \mathbf {v} \left(\mathbf {x} _ {0}, t\right) \Psi^ {- 1} \left(\left[ \mathbf {x} _ {t _ {0}}; t _ {0} \right]\right)\right). \tag {19}
$$

Based on the translation property of the Dirac delta function, i.e. $\int f(x)\delta(x-a)\mathrm{d}x=f(a)$ , the original forward process $p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})=[\psi_{t}\circ\Psi^{-1}\circ\Psi]_{*}\pi(\mathbf{x}_{t})=[\psi_{t}]_{*}\pi(\mathbf{x}_{t})$ naturally ignores the influence of discrete boundaries, even if the boundary information is explicitly added as a condition.

To enable the discrete priors, we propose a simple and intuitive approach: rescale the forward trajectory. As shown in Figure 2B, the original forward process flows from $\mathbf{x}_0$ to a random noise $\epsilon$ , and we reset the starting point to $\mathbf{x}_{t_0}$ . Accordingly, the intermediate noisy points $\mathbf{x}_t, t \in [1, T]$ will be proportionally mapped on this new path, which is

$$
\tilde {\mathbf {x}} _ {t} = \mathbf {x} _ {\tau}, \quad \tau = \mathcal {T} (t, t _ {0}) = r \times t _ {0} + t \times (T - r \times t _ {0}) / T
$$

$$
= \mathbf {u} (\mathbf {x} _ {0}, \mathcal {T} (t, t _ {0})) \mathbf {x} _ {0} + \mathbf {v} (\mathbf {x} _ {0}, \mathcal {T} (t, t _ {0})) \Psi^ {- 1} \left([ \mathbf {x} _ {t _ {0}}; t _ {0} ]\right). \tag {20}
$$

Similar to eq. (19), the rescaled conditional forward process is a Dirac delta function:

$$
\tilde {p} _ {t} \left(\tilde {\mathbf {x}} _ {t} \mid \mathbf {x} _ {t _ {0}}, t _ {0}, \mathbf {x} _ {0}\right) = \delta \left(\tilde {\mathbf {x}} _ {t} - \mathbf {u} \left(\mathbf {x} _ {0}, \mathcal {T} (t, t _ {0})\right) \mathbf {x} _ {0} - \mathbf {v} \left(\mathbf {x} _ {0}, \mathcal {T} (t, t _ {0})\right) \Psi^ {- 1} \left([ \mathbf {x} _ {t _ {0}}; t _ {0} ]\right)\right). \tag {21}
$$

However, $\tilde{p}_{t}(\tilde{\mathbf{x}}_{t}|\mathbf{x}_{0})$ faces the same problem of irreversibility as in eq. (14) and we derive it as:

$$
\tilde {p} _ {t} \left(\tilde {\mathbf {x}} _ {t} \mid \mathbf {x} _ {0}\right) = \int_ {c} \tilde {p} _ {t} \left(\tilde {\mathbf {x}} _ {t}, \tau \mid \mathbf {x} _ {0}\right) \mathrm{d} \tau = \int \tilde {p} _ {t} \left(\tilde {\mathbf {x}} _ {t}, \tau \mid \mathbf {x} _ {t _ {0}}, t _ {0}, \mathbf {x} _ {0}\right) p \left(\mathbf {x} _ {t _ {0}}, t _ {0} \mid \mathbf {x} _ {0}\right) \mathrm{d} \left[ \mathbf {x} _ {t _ {0}}; t _ {0} \right] \mathrm{d} \tau \tag {22}
$$

$$
= \int [ \psi_ {\tau} \circ \Psi^ {- 1} \circ \Psi ] _ {*} \pi ([ \tilde {\mathbf {x}} _ {t}; \tau ]) \mathrm{d} \tau = \int [ \psi_ {\tau} ] _ {*} \pi ([ \tilde {\mathbf {x}} _ {t}; \tau ]) \mathrm{d} \tau .
$$

Obtaining the probability density function requires gathering together the probability densities of the same location $\tilde{x}_{t}$ with different $\tau$ , which is intractable. Fortunately, we only need to sample noisy points from this probability distribution $\tilde{\mathbf{x}}_{t} \sim \tilde{p}_{t}(\tilde{\mathbf{x}}_{t} | \mathbf{x}_{0})$ , which is easy to implement:

$$
\tilde {\mathbf {x}} _ {t} = \mathbf {u} \left(\mathbf {x} _ {0}, \mathcal {T} (t, G (\mathbf {x} _ {0}, \boldsymbol {\epsilon}))\right) \mathbf {x} _ {0} + \mathbf {v} (\mathbf {x} _ {0}, \mathcal {T} (t, G (\mathbf {x} _ {0}, \boldsymbol {\epsilon}))) \boldsymbol {\epsilon}, \quad \boldsymbol {\epsilon} \sim \pi (\mathbf {x}). \tag {23}
$$

# 3.3 Recover Data from Noise

Training Objective Theoretically, the diffusion neural networks can be trained as in eq. (2), where the rescaled vector field is derived as $\tilde{u}_t = \frac{\mathrm{d}\tilde{\mathbf{x}}_t}{\mathrm{d}t} = \frac{\mathrm{d}\tilde{\mathbf{x}}_t}{\mathrm{d}\tau}\frac{\mathrm{d}\tau}{\mathrm{d}t}$ . However, since a low error estimation on $\mathbf{x}_0$ is of significant importance to our trajectory rescaling method, according to eqs. (10) to (13), we convert the objective to an upper bound of the eq. (2) (See appendix F for more details) and train a neural network $\mathbf{x}_{\theta}(\tilde{\mathbf{x}}_t,t)$ to predict $\mathbf{x}_0$ directly:

Algorithm 1 Training 

<table><tr><td>1:</td><td>repeat</td></tr><tr><td>2:</td><td> $\mathbf{x}_{0} \sim q(\mathbf{x}_{0}), \boldsymbol{\epsilon} \sim \pi(\mathbf{x}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ </td></tr><tr><td>3:</td><td> $t \sim \text{Uniform}(\{1, \ldots, T\})$ </td></tr><tr><td>4:</td><td> $\tau := \mathcal{T}(t, G(\mathbf{x}_{0}, \boldsymbol{\epsilon}))$  // eqs. (13) and (20)</td></tr><tr><td>5:</td><td> $\tilde{\mathbf{x}}_{t} := \mathbf{u}(\mathbf{x}_{0}, \tau)\mathbf{x}_{0} + \mathbf{v}(\mathbf{x}_{0}, \tau)\boldsymbol{\epsilon}$  // eq. (23)</td></tr><tr><td>6:</td><td>Take gradient descent step on $\nabla_{\theta}||\mathbf{x}_{0} - \mathbf{x}_{\theta}(\tilde{\mathbf{x}}_{t}, t)||^{2}$  // eq. (24)</td></tr><tr><td>7:</td><td>until converged</td></tr></table>

$$
\mathcal {L} _ {\theta} = \mathbb {E} _ {\mathbf {x} _ {0} \sim q (\mathbf {x} _ {0}), t \sim \mathcal {U} _ {(1, T)}, \tilde {\mathbf {x}} _ {t} \sim \tilde {p} _ {t} (\mathbf {x} | \mathbf {x} _ {0})} \left[ \| \mathbf {x} _ {0} - \mathbf {x} _ {\theta} (\tilde {\mathbf {x}} _ {t}, t) \| ^ {2} \right]. \tag {24}
$$

The training procedure is demonstrated in algorithm 1 and key steps are summarized in the line 4.

Reverse Process A direct approach that follows the flow matching is to solve the ODE of $\mathrm{d}\psi_{T - t}(\mathbf{x}) = \tilde{u}_{T - t}(\psi_{T - t}(\mathbf{x})|\mathbf{x}_0)\mathrm{d}t,\psi_T(\mathbf{x})\sim$ $\pi (\mathbf{x})$ . This form of transformation is inefficient with $\mathbf{x}_0$ -prediction during inference because we have to solve the equation of $\tau =$ $\mathcal{T}\left(t,G\left(\mathbf{x}_{\theta},\frac{\tilde{\mathbf{x}}_t - \mathbf{u}(\mathbf{x}_\theta,\tau)\mathbf{x}_\theta}{\mathbf{v}(\mathbf{x}_\theta,\tau)}\right)\right)$ to get the $\tau$ with respect to the change of $\tilde{\mathbf{x}}_t$ and $\mathbf{x}_{\theta}$ in real time. Therefore, we provide a deterministic reverse process as an alternative, which is a special case of DDIM [Song et al., 2021a] or the ODE with discrete timesteps. Given the time intervals $\Delta t\in [\Delta t_1,\dots \Delta t_s],\sum \Delta t = T$ , we general ize the boundary conditions $[\mathbf{x}_{t_{0}};t_{0}]$ in $\tilde{p}_{t}(\tilde{\mathbf{x}}_{t}|\mathbf{x}_{t_{0}},t_{0},\mathbf{x}_{0})$ of eq. (21) and $\Psi^{-1}([ \mathbf{x}_{t_{0}};t_{0}])$ of eq. (15) to any arbitrary condition pairs $[\tilde{\mathbf{x}}_{t};\tau]$ and obtain the reverse process:

Algorithm 2 Sampling 

<table><tr><td>1:</td><td> $t := T, \tau := T$ </td><td></td></tr><tr><td>2:</td><td> $\hat{\epsilon} \simeq \tilde{\mathbf{x}}_{t} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ </td><td>// Initialing</td></tr><tr><td>3:</td><td> $\textbf{for } \Delta t := \Delta t_{1}, \ldots, \Delta t_{s} \textbf{ do}$ </td><td> $// \sum_{\Delta t} = T$ </td></tr><tr><td>4:</td><td> $\hat{\mathbf{x}}_{0} := \mathbf{x}_{\theta}(\hat{\mathbf{x}}_{t}, t)$ </td><td>// Pseudo Target</td></tr><tr><td>5:</td><td> $t := t - \Delta t$ </td><td>// Updating</td></tr><tr><td>6:</td><td> $\tau := \mathcal{T}(t, G(\hat{\mathbf{x}}_{0}, \hat{\epsilon}))$ </td><td>// eq. (25)</td></tr><tr><td>7:</td><td> $\tilde{\mathbf{x}}_{t} := \mathbf{u}(\hat{\mathbf{x}}_{0}, \tau)\hat{\mathbf{x}}_{0} + \mathbf{v}(\hat{\mathbf{x}}_{0}, \tau)\hat{\epsilon}$ </td><td></td></tr><tr><td>8:</td><td> $\hat{\epsilon} := \Psi^{-1}([ \tilde{\mathbf{x}}_{t}; \tau ])$ </td><td>// Trajectory Alteration</td></tr><tr><td>9:</td><td> $\textbf{end for}$ </td><td></td></tr><tr><td>10:</td><td> $\mathbf{x}_{0} := \mathbf{x}_{\theta}(\tilde{\mathbf{x}}_{t}, t)$ </td><td>//  $\mathbf{x}_{1} \to \mathbf{x}_{0}$ </td></tr><tr><td>11:</td><td> $\textbf{return } \mathbf{x}_{0}$ </td><td></td></tr></table>

$$
\tilde {p} ([ \tilde {\mathbf {x}} _ {t - \Delta t}; \tau_ {\Delta} ] | [ \tilde {\mathbf {x}} _ {t}; \tau ], \hat {\mathbf {x}} _ {0}) =
$$

$$
\delta \left(\left[ \begin{array}{c} \tilde {\mathbf {x}} _ {t - \Delta t} \\ \tau_ {\Delta} \end{array} \right] - \left[ \begin{array}{c} \mathbf {u} (\hat {\mathbf {x}} _ {0}, \tau_ {\Delta}) \hat {\mathbf {x}} _ {0} + \mathbf {v} (\hat {\mathbf {x}} _ {0}, \tau_ {\Delta}) \hat {\boldsymbol {\epsilon}} \\ \mathcal {T} (t - \Delta t, G (\hat {\mathbf {x}} _ {0}, \hat {\boldsymbol {\epsilon}})) \end{array} \right]\right), \tag {25}
$$

where $\hat{\mathbf{x}}_{0}=\mathbf{x}_{\theta}(\tilde{\mathbf{x}}_{t},t)$ and $\tau_{\Delta}$ is the previous timestep of $\tau$ on the same rescaled trajectory.

Sampling from the reverse process is illustrated in algorithm 2. Similar to the sampling process of DDIM [Song et al., 2021a], it starts from the Gaussian noise, iteratively predicts the pseudo target $\hat{\mathbf{x}}_0$ , and updates the reverse trajectory. However, since the $\tau$ and $\hat{\epsilon}$ are mutually conditioned, we have to keep track of the $t$ , $\tau$ , $\tilde{\mathbf{x}}_t$ , and $\hat{\epsilon}$ during each iteration and split the update of $\hat{\epsilon}$ into an asynchronous step (line 8). Because reverse trajectory keeps changing due to different pseudo targets $\hat{\mathbf{x}}_0$ predicted by learned neural networks, which brings severe instability, sometimes simply fixing the initial path (removing the line 8) exhibits better performance in experiments.

# 4 Language Modeling

Recent diffusion language models [Li et al., 2022, Gong et al., 2023b] inherit the embedding-rounding framework that a sentence with n discrete tokens $W = [w_{1}, \ldots, w_{n}]$ is embedded to a continuous space via a trainable embedding layer $\operatorname{EMB}(W) = [\operatorname{EMB}(w_{1}), \ldots, \operatorname{EMB}(w_{n})]$ . The vocabulary set is K that $\forall w_{n} \in K$ . Besides, the token embeddings are used as the target points $x_{0} = [x_{0}^{1}, \ldots, x_{0}^{n}]$ , $x_{0}^{n} = \operatorname{EMB}(w_{n})$ , for continuous diffusion trajectories. Hence, generating tokens from embeddings is:

$$
p (W | \mathbf {x} _ {0}) = \sum_ {i = 1} ^ {n} p (w _ {i} | \mathbf {x} _ {0} ^ {i}) = \sum_ {i = 1} ^ {n} \frac {\exp (f (\mathbf {x} _ {0} ^ {i} , w _ {i}))}{\sum_ {j \in K} \exp (f (\mathbf {x} _ {0} ^ {i} , j))}, \tag {26}
$$

where $f(\mathbf{x},j) = \operatorname{EMB}(j)\cdot \mathbf{x}$ is the dot production distance. It's also the function assessing the likelihood of point $\mathbf{x}$ inside the discrete area of $j$ . The coefficient functions follow the DDPM [Ho et al., 2020], which are $\mathbf{u}(\mathbf{x}_0,t) = \sqrt{\bar{\alpha}_t}$ and $\mathbf{v}(\mathbf{x}_0,t) = \sqrt{1 - \bar{\alpha}_t}$ . Besides, the objectives are

$$
\mathcal {L} _ {\theta} = \mathbb {E} _ {W, t, \tilde {\mathbf {x}} _ {t}} \left[ \sum_ {i = 1} ^ {n} \| \mathrm{EMB} (w _ {i}) - \mathbf {x} _ {\theta} (\tilde {\mathbf {x}} _ {t} ^ {i}, t) \| ^ {2} / n \right] \tag {27}
$$

Table 1: Result of BLEU scores on machine translation and ROUGE scores on text summarization. 

<table><tr><td>Models</td><td>IWSLT14 DE-EN BLEU (BLEU-1/2/3/4)↑</td><td>WMT14 EN-DE BLEU (BLEU-1/2/3/4)↑</td><td>WMT16 EN-RO BLEU (BLEU-1/2/3/4)↑</td><td>GIGAWORD ROUGE-1/2/L↑</td></tr><tr><td colspan="5">Auto-Regressive Modeling</td></tr><tr><td>Transformers</td><td>34.31 (67.3/41.6/27.9/19.1)</td><td>28.01 (58.2/33.5/21.7/14.6)</td><td>34.05 (63.1/39.9/27.6/19.6)</td><td>37.57/18.90/34.69</td></tr><tr><td>Ours+Rerank</td><td>35.02 (68.7/43.3/29.2/20.1)</td><td>27.67 (57.9/33.2/21.4/14.3)</td><td>34.33 (63.1/40.1/27.8/19.8)</td><td>37.49/18.68/34.82</td></tr><tr><td colspan="5">Diffusion Process</td></tr><tr><td>D3PM</td><td>27.61 (65.4/37.7/22.8/14.2)</td><td>22.94 (54.9/28.8/16.9/10.4)</td><td>27.84 (59.8/34.9/22.1/14.5)</td><td>33.92/14.96/31.72</td></tr><tr><td>DiffuSeq</td><td>28.78 (- / - / - / - )</td><td>15.37 (- / - / - / - )</td><td>25.45 (- / - / - / - )</td><td>31.17/12.23/29.24</td></tr><tr><td>SeqDiffuSeq</td><td>30.03 (- / - / - / - )</td><td>17.14 (- / - / - / - )</td><td>26.17 (- / - / - / - )</td><td>31.90/12.36/29.22</td></tr><tr><td>Differmer</td><td>31.58 (68.6/41.4/26.7/17.5)</td><td>24.80 (58.7/32.0/19.7/12.5)</td><td>30.08 (64.4/39.5/26.5/18.2)</td><td>35.47/15.17/32.82</td></tr><tr><td>SEDD</td><td>31.87 (68.7/41.8/27.2/18.0)</td><td>24.98 (59.2/32.4/20.1/12.9)</td><td>29.38 (62.2/38.0/24.9/16.9)</td><td>34.33/15.22/32.06</td></tr><tr><td>Dinoiser</td><td>31.91 (67.1/40.9/26.7/17.7)</td><td>24.77 (57.2/31.0/19.0/12.0)</td><td>31.49 (62.8/38.4/25.5/17.3)</td><td>35.17/15.63/32.53</td></tr><tr><td>Ours</td><td>33.42 (68.0/42.0/27.7/18.6)</td><td>26.69 (57.7/32.3/20.4/13.4)</td><td>33.15 (63.4/39.9/27.4/19.2)</td><td>36.44/16.09/33.56</td></tr></table>

and an additional rounding objective, which is commonly used in language modeling,

$$
\mathcal {L} _ {r} = - \log p _ {\theta} (W | \mathbf {x} _ {0}) = - \log p _ {\theta} (W | \mathbf {x} _ {\theta} (\tilde {\mathbf {x}} _ {t}, t)). \tag {28}
$$

The final training target is given by $L = L_{\theta} + L_{r}$ , where the $x_{0}$ of the same token sequence W keeps changing because the embedding layer EMB is trainable, which makes the model hard to be trained. Since previous work does not model discrete areas, a large number of noisy samples inside this area will make $L_{r}$ too small to guide the training of the embedding layer, leading to a mode collapse.

Experimental Setup Datasets used for experiments include three translation tasks (IWSLT14 DE-EN [Cettolo et al., 2012], WMT14 EN-DE, and WMT16 EN-RO $^{1}$ ) and one text summarization task (GIGAWORD [Rush et al., 2015]). We mainly follow the setting of Gao et al. [2022], which is inherited from previous non-auto-regressive text generation works [Gu et al., 2018, 2019, Ghazvininejad et al., 2019], where translation datasets are distilled [Kim and Rush, 2016]. Baselines are mainly continuous diffusion language models. DiffuSeq [Gong et al., 2023b] and SeqDiffuSeq [Yuan et al., 2022] are derived from Diffusion-LM [Li et al., 2022]. Differmer [Gao et al., 2022] and Dinoiser [Ye et al., 2023] are recent empirical studies highlighting that scaling up the noise is beneficial for language modeling. We also compare with discrete diffusion language models, including D3PM [Austin et al., 2021] and SEDD [Lou et al., 2023]. Since SEDD is a pre-trained language model, we configure its framework and train it from scratch specifically for our tasks. In addition, auto-regressive transformer [Vaswani et al., 2017] is still one of the most powerful architectures for language generation.

Our boundary conditional diffusion language model is constructed from Differmer [Gao et al., 2022], where the model configuration is transformer-iwslt-de-en in FAIRSEQ framework [Ott et al., 2019] for IWSLT14 DE-EN and transformer-base for other datasets. Sentences are tokenized with Byte-Pair Encoding [Sennrich et al., 2016] and evaluated by detokenized BLEU [Papineni et al., 2002] for machine translation and ROUGE [Lin, 2004] for summarization. During training, the diffusion step is T = 2000 and the confidence factor r = 1 for translation tasks since they have strong conditions, while r = 0.5 for summarization. Sentences are generated deterministically with 20 steps.

Results Performances are demonstrated in Table 1. Our approach achieves the state-of-the-art compared with continuous diffusion language models and outperforms the two discrete baselines on three machine translation and one text summarization tasks. Our method shows advantages, with a $73.6\%$ significant improvement at most on WMT14 EN-DE, over DiffuSeq [Gong et al., 2023b] and SeqDiffuSeq [Yuan et al., 2022], which are two basic methods directly applying diffusion process to language modeling. Compared with recent strong diffusion language models like Differmer [Gao et al., 2022] and Dinoiser [Ye et al., 2023], which have deployed various effective noise scheduling strategies on diffusion processes from the empirical perspective, our model is still superior with at most 3.07 advancement of BLEU score on WMT16 EN-RO. This implies the effectiveness of modeling discrete priors. In addition, we illustrate the performance of auto-regressive modeling, where we use the transformer [Vaswani et al., 2017] to rerank the generated sentence candidates (7

Table 3: Analysis on the training objectives. 

<table><tr><td>Objectives</td><td> $\mathbb{E}_{\tilde{\mathbf{x}}_{t}}\| \mathbf{x}_{0}-\hat{\mathbf{x}}_{0}\|^{2}$ </td><td> $\mathbb{E}_{\tilde{\mathbf{x}}_{t}}\| \tilde{u}_{t}(\tilde{\mathbf{x}}_{t}|\mathbf{x}_{0})-\tilde{u}_{t}(\tilde{\mathbf{x}}_{t}|\hat{\mathbf{x}}_{0})\|^{2}$ </td><td> $\mathbb{E}_{\tilde{\mathbf{x}}_{t}}[p(\hat{\mathbf{x}}_{0}\in C_{\mathbf{x}_{0}})]$ </td><td>BLEU</td></tr><tr><td> $\mathcal{L}_{\mathbf{x}_{0}}$ (eq. 24)</td><td>8.44</td><td>1.56</td><td>51.81%</td><td>33.42</td></tr><tr><td> $\mathcal{L}_{\tilde{u}_{t}}$ </td><td>8.41</td><td>1.55</td><td>52.34%</td><td>33.49</td></tr></table>

length beam $\times$ 3 sentence beams) of our model. The reranked performance can even outperform transformers on IWSLT14 DE-EN and WMT16 EN-RO.

Ablation Our approach is a general framework applicable to almost all continuous diffusion models, providing them with discrete boundaries as priors. We choose Differmer [Gao et al., 2022] as the base model and follow the configurations. As proved in eq. (19), the original forward process will ignore the discrete pri-

Table 2: Ablation studies. 

<table><tr><td>Models</td><td>IWSLT14</td><td>WMT16</td></tr><tr><td>Base (Differmer)</td><td>31.58</td><td>30.08</td></tr><tr><td>+ forward only</td><td>33.02</td><td>32.86</td></tr><tr><td>+ forward &amp; reverse</td><td>33.42</td><td>33.15</td></tr><tr><td>Optimal Transport</td><td>32.77</td><td>33.65</td></tr></table>

ors although explicitly demonstrated. We conduct ablation experiments on the rescaling module. As illustrated in Table 2, our approach rescales the trajectory of both forward and reverse processes on Differmer. Only rescaling the forward trajectory is also effective but sub-optimal due to the inconsistent distribution during inference. Due to computational cost and fair comparison, our method leaves room for improvement. For example, replacing the forward trajectory with optimal transport in Flow Matching, $\mathbf{u}(\mathbf{x}_0,t) = 1 - t / T$ and $\mathbf{v}(\mathbf{x}_0,t) = t / T$ , achieves better performance on WMT16.

Analysis Our training objective, eq. (24), is an upper bound of the eq. (2). We demonstrate the influence of this approximation in Table 3 on IWSLT14 DE-EN to reveal the thought of our formula. On the one hand, $\mathcal{L}_{\mathbf{x}_0}$ brings theoretical errors at a constant scale. On the other hand, $\mathcal{L}_{\mathbf{x}_0}$ mitigates some experimental errors from the neural networks. The first row $\mathcal{L}_{\mathbf{x}_0}$ is the objective we used in eq. (24) and the second row $\mathcal{L}_{\tilde{u}_t} = \mathbb{E}_{\{t,\mathbf{x}_0,\tilde{\mathbf{x}}_t\}}\left[\| \tilde{u}_t(\tilde{\mathbf{x}}_t|\mathbf{x}_\theta (\tilde{\mathbf{x}}_t,t)) - \frac{\mathrm{d}\tilde{\mathbf{x}}_t}{\mathrm{d}t}\| ^2\right]$ is directly derived from the eq. (2). The first two columns represent the error expectations of $\mathbf{x}_0$ and $\tilde{u}_t$ on the test set. It is easy to observe that, with the dynamic coefficient $\frac{\mathrm{d}\tau}{\mathrm{d}t} = \frac{T - r\times G(\mathbf{x}_0,\epsilon)}{T}$ (appendix F), the value of $\mathbf{x}_0$ 's error (8.44) is much larger than the $\tilde{u}_t$ 's error (1.56). Therefore, $\mathcal{L}_{\mathbf{x}_0}$ is beneficial for reducing the impact of the prediction error from the neural network. The third column in Table 3 illustrates the one-step accuracy of predicting $\mathbf{x}_0$ and the fourth column is the BLEU score on the test set. Experimental results show that optimizing the upper bound has a negligible impact on the final performance (only a $0.2\%$ drop of the BLEU score), while can improve the efficiency of the loss calculation during the training phase.

# 5 Discrete Image Generation

Image pixels are usually treated as real numbers in continuous space since adjacent pixel values exhibit linear continuity. They are essentially discrete and quantized data with a finite state space, such as 256 states in RGB format. We utilize two discrete image representations. One is binary coding provided by Bit Diffusion [Chen et al., 2023b] that converts a sub-pixel with 256 integers to a 8-bit binary code. It is more efficient as it stores ordinal relationships, but the representation space it constructs will be sparse. Another is pixel embedding, which is a more discrete form of representation because the relationships between pixels are thoroughly broken down and reconstructed by learning the embedding representation. Each pixel is regarded as a one-hot vector and transformed with an embedding layer EMB as used in language. Furthermore, we design an intermediate state to demonstrate the correlation between discreteness and modeling difficulty, which is initializing a fixed embedding with binary coding. The optimization target for binary coding is the MSE loss, and pixel embeddings take the same objective as in language.

Experimental Setup We use CIFAR-10 [Krizhevsky et al., 2009] for discrete image generation. The evaluation metric is FID [Heusel et al., 2017], which compares 50K generated samples with the training set. Our image generation model is constructed on Bit Diffusion [Chen et al., 2023b], where the architecture is U-Net [Ronneberger et al., 2015] with 3 stages, 256 channels and 3 residual blocks

![](images/a733b4d1bd347508bfc00b4bba7980fdd4419260b81f9d94f49d0acbfffc5659.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 colorful and natural imagery including animals, birds, fish, and vehicles (no text or symbols)
</details>

(A) Bit Diffusion repro (FID 10.37)

![](images/e70fcd22c61e533eb4724c14c730c2ffc556dd0e30ea876dcb5882f224cca9a5.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 nature and animal images including cats, animals, vehicles, and construction equipment (no text or symbols)
</details>

(B) DDIM (FID 4.04)

![](images/5283e3cac95c83432926ed2b0b48cc15656a67568ab55e7b642ece84e7494d1a.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 colorful images including animals, vehicles, and landscapes (no text or symbols)
</details>

(C) Ours (FID 3.86)   
Figure 3: Generated images of Bit Diffusion repro, DDIM, and Ours on CIFAR-10.

per stage. Diffusion steps are T = 1000 for both the training and inference stages. The model is trained for 1.5M steps with the learning rate of 1e-4 and batch size of 128. Since the training script and detailed hyperparameters of Bit Diffusion are not available, we have to reproduce it by ourselves and our boundary conditional diffusion model shares exactly the same configuration. Our confidence factors are r = 0.5 for all three settings. Other baselines include D3PM [Austin et al., 2021] and $\tau$ LDR [Campbell et al., 2022] which are discrete diffusion models. SDDM [Sun et al., 2023] utilizes vector quantization from VQ-GAN [Esser et al., 2021] as a continuous space for discrete data. We also compare with DDPM [Ho et al., 2020] and DDIM [Song et al., 2021a] on continuous pixels.

Results For binary coding, as shown in Table 4, our approach outperforms the reproduced Bit Diffusion and attains competitive results to state-of-the-art models. For pixel embedding where ordinal information is deconstructed and reconstituted, our method exhibits a notable improvement of 3.81 FID score over replicated Bit Diffusion. Moreover, in the case of categorical pixels, this advantage increases to 8.25, positioning our approach with trainable embedding as a new state-of-the-art solution. Additionally, as deterministic diffusion processes, our model with binary coding can slightly exceed the performance of DDIM, where the generated samples are in Figure 3.

Analysis We analyze the influence of the confidence factor r in Table 5. The factor r is selected from [0, 0.2, 0.3, 0.5], where $r = 0$ is the reproduced Bit Diffusion that discards the discrete priors. As the confidence factor increases, the impact of discreteness gradually improves, simultaneously enhancing the model's performance across all three settings. Since there is no guidance for unconditional image generation, we do not use a larger factor to prevent mode collapses.

Table 4: FID scores on CIFAR-10. 

<table><tr><td rowspan="2">Models</td><td colspan="3">CIFAR-10 (FID ↓)</td></tr><tr><td>200K</td><td>500K</td><td>Final</td></tr><tr><td colspan="4">Continuous Pixels</td></tr><tr><td>DDPM</td><td>-</td><td>-</td><td>3.17</td></tr><tr><td>DDIM</td><td>-</td><td>-</td><td>4.04</td></tr><tr><td colspan="4">Discrete Ordinal Pixels</td></tr><tr><td>D3PM GAUSS</td><td>-</td><td>-</td><td>7.34</td></tr><tr><td>τLDR-0</td><td>-</td><td>-</td><td>8.10</td></tr><tr><td>τLDR-10</td><td>-</td><td>-</td><td>3.74</td></tr><tr><td colspan="4">BINARY CODING (UINT8):</td></tr><tr><td>Bit Diffusion</td><td>-</td><td>-</td><td>3.48</td></tr><tr><td>Bit Diffusion repro</td><td>22.12</td><td>13.23</td><td>10.37</td></tr><tr><td>Ours</td><td>8.17</td><td>5.03</td><td>3.86</td></tr><tr><td colspan="4">FIXED EMBEDDING:</td></tr><tr><td>Bit Diffusion repro</td><td>19.69</td><td>16.61</td><td>12.96</td></tr><tr><td>Ours</td><td>12.32</td><td>10.09</td><td>9.15</td></tr><tr><td colspan="4">Categorical Pixels</td></tr><tr><td>D3PM UNIFORM</td><td>-</td><td>-</td><td>51.27</td></tr><tr><td>D3PM ABSORBING</td><td>-</td><td>-</td><td>30.97</td></tr><tr><td colspan="4">VECTOR QUANTIZATION:</td></tr><tr><td>D3PM-VQ</td><td>-</td><td>-</td><td>16.47</td></tr><tr><td>τLDR-VQ</td><td>-</td><td>-</td><td>40.06</td></tr><tr><td>SDDM-VQ</td><td>-</td><td>-</td><td>12.23</td></tr><tr><td colspan="4">TRAINABLE EMBEDDING:</td></tr><tr><td>Bit Diffusion repro</td><td>33.09</td><td>27.21</td><td>19.26</td></tr><tr><td>Ours</td><td>21.17</td><td>15.32</td><td>10.99</td></tr></table>

# 6 Related Work

Discrete Modeling Auto-regressive models have demonstrated a domination over discrete modeling, especially for text generation [Vaswani et al., 2017, Brown et al., 2020, Achiam et al., 2023]. However, the computation

Table 5: Confidence factors. 

<table><tr><td>Models</td><td>r = 0</td><td>0.2</td><td>0.3</td><td>0.5</td></tr><tr><td>BINARY CODING</td><td>10.37</td><td>7.39</td><td>5.33</td><td>3.86</td></tr><tr><td>FIXED EMBEDDING</td><td>12.96</td><td>11.35</td><td>10.80</td><td>9.15</td></tr><tr><td>TRAINABLE EMBEDDING</td><td>19.26</td><td>15.32</td><td>11.56</td><td>10.99</td></tr></table>

cost increases drastically as the size of sentence length or the image resolution increases. Diffusion models [Sohl-Dickstein et al., 2015, Ho et al., 2020, Dhariwal and Nichol, 2021, Saharia et al., 2022] can generate data in parallel, but are tailored for continuous problems. To generalize diffusion models for discrete data, the most straightforward methods define discrete processes in discrete spaces [Sohl-Dickstein et al., 2015, Hoogeboom et al., 2021b, Austin et al., 2021, Campbell et al., 2022, Zhang et al., 2023, Sun et al., 2023, Lou et al., 2023], which will be bothered by large number of discrete status. Besides, a simplified version of discrete diffusion processes is recently used in language modeling [He et al., 2023, Chen et al., 2023a]. Approaches in another line argue to located discrete data in continuous spaces, which is more flexible and efficient, with the mapping functions including binary bits [Chen et al., 2023b] and embeddings [Li et al., 2022, Gong et al., 2023b,a, Yuan et al., 2022, Gulrajani and Hashimoto, 2023, Han et al., 2023]. Other generative models adapted for discrete modeling includes Variational Autoencoders [Kingma and Welling, 2014], Generative Adversarial Networks [Hjelm et al., 2018, Fedus et al., 2018], and Normalizing Flows [Lindt and Hoogeboom, 2021, Hoogeboom et al., 2021a, Tan et al., 2022].

Diffusion Models with Deterministic Trajectory Deterministic diffusion process is usually used in the inference stage to speed up sampling, where DDIM [Song et al., 2021a] derives a serial of non-Markovian diffusion processes and the deterministic one is a special case from this implicit perspective. Additionally, deterministic diffusion processes can be converted to ordinary differential equations [Song et al., 2021b], which is utilized by recent sampling acceleration approaches such as DEIS [Zhang and Chen, 2023] and DPM-Solvers [Lu et al., 2022b,a, Zheng et al., 2023]. Our approach requires a deterministic forward trajectory to eliminate the randomness between the boundary point and sampled point. Flow matching [Liu, 2022, Lipman et al., 2023, Albergo and Vanden-Eijnden, 2023, Liu et al., 2023] is a collection of generative models that employ ordinary differential equations to facilitate both forward and reverse processes. They can be regarded as generally equivalent to Diffusion models. Therefore, we extend the framework of flow matching for our method.

# 7 Conclusion

We studied the gap between discrete modeling and continuous spaces, focusing on the inconsistency between probability density contours learned by continuous diffusion models and discrete boundaries. We have proposed a novel and general approach to address this issue by enabling continuous diffusion models to be conditioned on discrete priors, which is achieved via discrete boundary estimation and trajectory rescaling. An important limitation is that our method is designed for continuous diffusion models, where discrete diffusion models constructed specially on the discrete state space would not encounter the problem. However, discrete diffusion models also possess their own shortcomings, and the practical applications of continuous diffusion models are more extensive. We believe that our method has the potential to advance the development of unified and general diffusion models. By bridging the gap between discrete and continuous modeling, we hope to inspire new possibilities for modeling complex systems and phenomena.

# Acknowledgements

Bing Qin is the corresponding author of this work, We thank the anonymous reviewers for their insightful comments. This work was supported by the National Natural Science Foundation of China (NSFC) (U22B2059, grant 62276078), the Key R&D Program of Heilongjiang via grant 2022ZX01A32, the International Cooperation Project of PCL, PCL2022D01 and the Fundamental Research Funds for the Central Universities (Grant No.HIT.OCEF.2023018).

# References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=li7qeBbCR1t.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020.   
Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.   
Mauro Cettolo, Christian Girardi, and Marcello Federico. WIT3: Web inventory of transcribed and translated talks. In Mauro Cettolo, Marcello Federico, Lucia Specia, and Andy Way, editors, Proceedings of the 16th Annual Conference of the European Association for Machine Translation, pages 261–268, Trento, Italy, May 28–30 2012. European Association for Machine Translation.   
Jiaao Chen, Aston Zhang, Mu Li, Alex Smola, and Diyi Yang. A cheaper and better diffusion language model with soft-masked noise. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4765–4775, Singapore, December 2023a. Association for Computational Linguistics.   
Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. In The Eleventh International Conference on Learning Representations, 2023b.   
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 8780–8794. Curran Associates, Inc., 2021.   
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy.   
Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021.   
William Fedus, Ian Goodfellow, and Andrew M. Dai. Maskgan: Better text generation via filling in the \_. In International Conference on Learning Representations, 2018.   
Zhujin Gao, Junliang Guo, Xu Tan, Yongxin Zhu, Fang Zhang, Jiang Bian, and Linli Xu. Difformer: Empowering diffusion model on embedding space for text generation. arXiv preprint arXiv:2212.09412, 2022.   
Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-predict: Parallel decoding of conditional masked language models. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6112–6121, Hong Kong, China, November 2019. Association for Computational Linguistics.

Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong. DiffuSeq-v2: Bridging discrete and continuous text spaces for accelerated Seq2Seq diffusion models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9868–9875, Singapore, December 2023a. Association for Computational Linguistics.   
Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. In The Eleventh International Conference on Learning Representations, 2023b.   
Jiatao Gu, James Bradbury, Caiming Xiong, Victor O.K. Li, and Richard Socher. Non-autoregressive neural machine translation. In International Conference on Learning Representations, 2018.   
Jiatao Gu, Changhan Wang, and Junbo Zhao. Levenshtein transformer. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.   
Ishaan Gulrajani and Tatsunori Hashimoto. Likelihood-based diffusion language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. SSD-LM: Semi-autoregressive simplex-based diffusion language model for text generation and modular control. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11575–11596, Toronto, Canada, July 2023. Association for Computational Linguistics.   
Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. DiffusionBERT: Improving generative masked language models with diffusion models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4521–4534, Toronto, Canada, July 2023. Association for Computational Linguistics.   
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.   
R Devon Hjelm, Athul Paul Jacob, Adam Trischler, Gerry Che, Kyunghyun Cho, and Yoshua Bengio. Boundary seeking gans. In International Conference on Learning Representations, 2018.   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 6840–6851. Curran Associates, Inc., 2020.   
Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 12454–12465. Curran Associates, Inc., 2021a.   
Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021b.   
Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. In Jian Su, Kevin Duh, and Xavier Carreras, editors, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317–1327, Austin, Texas, November 2016. Association for Computational Linguistics.   
Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In International Conference on Learning Representations, 2014.   
Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2021.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.   
Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori Hashimoto. Diffusion-LM improves controllable text generation. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022.   
Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics.   
Alexandra Lindt and Emiel Hoogeboom. Discrete denoising flows. In ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models, 2021.   
Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.   
Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.   
Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International conference on learning representations (ICLR), 2023.   
Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.   
Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022a.   
Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022b.   
Ali Madani, Bryan McCann, Nikhil Naik, Nitish Shirish Keskar, Namrata Anand, Raphael R Eguchi, Po-Ssu Huang, and Richard Socher. Progen: Language modeling for protein generation. arXiv preprint arXiv:2004.03497, 2020.   
Ali Madani, Ben Krause, Eric R Greene, Subu Subramanian, Benjamin P Mohr, James M Holton, Jose Luis Olmos, Caiming Xiong, Zachary Z Sun, Richard Socher, et al. Large language models generate functional protein sequences across diverse families. Nature Biotechnology, 41(8):1099–1106, 2023.   
Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of NAACL-HLT 2019: Demonstrations, 2019.   
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Pierre Isabelle, Eugene Charniak, and Dekang Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics.   
Niki J. Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International Conference on Machine Learning (ICML), 2018. URL http://proceedings.mlr.press/v80/parmar18a.html.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pages 234–241. Springer, 2015.   
Alexander M. Rush, Sumit Chopra, and Jason Weston. A neural attention model for abstractive sentence summarization. In Lluís Màrquez, Chris Callison-Burch, and Jian Su, editors, Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 379–389, Lisbon, Portugal, September 2015. Association for Computational Linguistics.   
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 36479–36494. Curran Associates, Inc., 2022.   
Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Katrin Erk and Noah A. Smith, editors, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany, August 2016. Association for Computational Linguistics.   
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, Lille, France, 07–09 Jul 2015. PMLR.   
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a.   
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b.   
Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. In The Eleventh International Conference on Learning Representations, 2023.   
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. Advances in neural information processing systems, 27, 2014.   
Shawn Tan, Chin-Wei Huang, Alessandro Sordoni, and Aaron Courville. Learning to dequantise with truncated flows. In International Conference on Learning Representations, 2022.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.   
Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, and Mingxuan Wang. Dinoiser: Diffused conditional sequence learning by manipulating noises. arXiv preprint arXiv:2302.10025, 2023.   
Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Fei Huang, and Songfang Huang. Seqdiffuseq: Text diffusion with encoder-decoder transformers. ArXiv, abs/2212.10325, 2022.   
Pengze Zhang, Hubery Yin, Chen Li, and Xiaohua Xie. Formulating discrete probability flow through optimal transport. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In The Eleventh International Conference on Learning Representations, 2023.   
Kaiwen Zheng, Cheng Lu, Jianfei Chen, and Jun Zhu. Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

# A Stopping Time for Forward Process

The forward diffusion process $X = \{x_{n}, n \geq 0\}$ is a markovian stochastic process with a transition probability $p(\mathbf{x}_{i} | \mathbf{x}_{i-1}) = \mathcal{N}\left(\mathbf{x}_{i}; \sqrt{\alpha_{i}} \mathbf{x}_{i-1}, (1 - \alpha_{i}) \mathbf{I}\right)$ . And a stopping time $t_{0}$ with respect to X is a random time such that for each $n \geq 0$ , the event $\{t_{0} = n\}$ is completely determined by the total information known up to time n, $\{x_{0}, \ldots, x_{n}\}$ . Suppose the random variables $\{x_{n}\}$ are in a one-dimensional space and the forward process starts with $x_{0} = 0$ . Besides, let $A, x_{0} \in A$ be the discrete area belonging to $x_{0}$ that for each points in area A will be regarded as $x_{0}$ during data generation. Our expected stopping time is defined as:

$$
t _ {0} = \min \{n \geq 0, \mathbf {x} _ {n} \notin A \},
$$

which represents the first time $x_{n}$ leaves area A. We can write the probability of stopping time as:

$$
\begin{array}{l} P \left(t _ {0} = 0\right) = P \left(\mathbf {x} _ {0} \notin A\right) = 0 \\ P (t _ {0} = 1) = P (\mathbf {x} _ {0} \in A, \mathbf {x} _ {1} \notin A) \\ = \int_ {\mathbf {x} _ {1} \notin A} \mathcal {N} \left(\mathbf {x} _ {1}; \sqrt {\alpha_ {1}} \mathbf {x} _ {0}, (1 - \alpha_ {1}) \mathbf {I}\right) d \mathbf {x} _ {1} \\ P \left(t _ {0} = 2\right) = P \left(\mathbf {x} _ {0} \in A, \mathbf {x} _ {1} \in A, \mathbf {x} _ {2} \notin A\right) \\ = P \left(\mathbf {x} _ {0} \in A, \mathbf {x} _ {1} \in A\right) \times P \left(\mathbf {x} _ {2} \notin A \mid \mathbf {x} _ {1} \in A\right) \\ = \int_ {\mathbf {x} _ {2} \notin A} \left[ \int_ {\mathbf {x} _ {1} \in A} \mathcal {N} \left(\mathbf {x} _ {1}; \sqrt {\alpha_ {1}} \mathbf {x} _ {0}, (1 - \alpha_ {1}) \mathbf {I}\right) \times \right. \\ \left. \mathcal {N} \left(\mathbf {x} _ {2}; \sqrt {\alpha_ {2}} \mathbf {x} _ {1}, (1 - \alpha_ {2}) \mathbf {I}\right) \mathrm{d} \mathbf {x} _ {1} \right] \mathrm{d} \mathbf {x} _ {2} \\ P (t _ {0} = n) = P (\mathbf {x} _ {0} \in A, \dots , \mathbf {x} _ {n - 1} \in A, \mathbf {x} _ {n} \notin A) \\ = \int_ {\mathbf {x} _ {n} \notin A} \int_ {\mathbf {x} _ {\leq n} \in A} \prod_ {i = 1} ^ {n - 1} \mathcal {N} \left(\mathbf {x} _ {i}; \sqrt {\alpha_ {i}} \mathbf {x} _ {i - 1}, (1 - \alpha_ {i}) \mathbf {I}\right) d \mathbf {x} _ {1: n}. \\ \end{array}
$$

$$
\begin{array}{l} P \left(t _ {0} = 2\right) = P \left(\mathbf {x} _ {0} \in A, \mathbf {x} _ {1} \in A, \mathbf {x} _ {2} \notin A\right) \\ = P \left(\mathbf {x} _ {0} \in A, \mathbf {x} _ {1} \in A\right) \times P \left(\mathbf {x} _ {2} \notin A \mid \mathbf {x} _ {1} \in A\right) \\ = \int_ {\mathbf {x} _ {2} \notin A} \left[ \int_ {\mathbf {x} _ {1} \in A} \mathcal {N} \left(\mathbf {x} _ {1}; \sqrt {\alpha_ {1}} \mathbf {x} _ {0}, (1 - \alpha_ {1}) \mathbf {I}\right) \times \right. \\ \end{array}
$$

• • • • • •

Since the diffusion process is established in continuous space, calculating the probability of the stopping time requires integrating over each intermediate state $x_{1:n-1}$ , rather than a simple state transfer as in the discrete space. Hence, directly obtain the stopping time is intractable. Additionally, even if we are able to get probability of the stopping time, we can only get a distribution over the time dimension, without knowing the exact time of $x_{n}$ leaving area A. Therefore, we need to eliminate randomness from the state transition $x_{i-1} \rightarrow x_{i}$ and find a deterministic forward trajectory to estimate the stopping time.

# B Properties of Dirac Delta Function

There are several useful properties of Dirac delta function:

- Symmetry Property: $\delta(-x) = \delta(x)$   
- Scaling Property: $\delta(ax) = \frac{\delta(x)}{|a|}$   
- Translation Property: $\int f(x)\delta (x - a)\mathrm{d}x = f(a)$

# C Bridging Flow Matching and DDPM

In this work, we utilize the framework of Flow Matching to model the diffusion processes, where the forward process is defined by flow functions in eq. (5). Although having different mathematical forms, it is essentially equivalent to traditional diffusion processes. Here, we provide an alternative form from the perspective of state transfer $p_{t}(\mathbf{x}_{t}|\mathbf{x}_{t-1})$ .

# C.1 Deterministic Forward Process

Equation (5) gives the definition $p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0}) = [\psi_{t}]_{*}\pi(\mathbf{x})$ , where $\psi_{t}(\mathbf{x}) = \mathbf{u}_{t}\mathbf{x}_{0} + \mathbf{v}_{t}\mathbf{x}$ . Here we provide the equivalent derivation of $p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})$ from the perspective of diffusion processes:

$$
\begin{array}{l} p _ {t} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right) = \int p _ {t} \left(\mathbf {x} _ {1: t} \mid \mathbf {x} _ {0}\right) \mathrm{d} \mathbf {x} _ {1: t - 1} \\ = \int p (\mathbf {x} _ {1} | \mathbf {x} _ {0}) \prod_ {s = 2} ^ {t} p _ {s} (\mathbf {x} _ {s} | \mathbf {x} _ {s - 1}, \mathbf {x} _ {0}) \mathrm{d} \mathbf {x} _ {1: t - 1}, \tag {29} \\ \end{array}
$$

where $p(\mathbf{x}_{1}|\mathbf{x}_{0}) = \mathcal{N}(\mathbf{u}_{1}\mathbf{x}_{0}, \mathbf{v}_{1}^{2}\mathbf{I})$ is the first step of the forward process at which the global noise is introduced into the forward trajectory. The state transfer probability of forward process $p_{s}(\mathbf{x}_{s}|\mathbf{x}_{s-1}, \mathbf{x}_{0}) = \delta(\mathbf{x}_{s} - \mathbf{u}_{s}\mathbf{x}_{0} - \mathbf{v}_{s}\psi_{s-1}^{-1}(\mathbf{x}_{s-1}))$ is a Dirac delta function. Therefore,

$$
\begin{array}{l} p _ {t} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right) = \int \prod_ {s = 3} ^ {t} p _ {s} \left(\mathbf {x} _ {s} \mid \mathbf {x} _ {s - 1}, \mathbf {x} _ {0}\right) \mathrm{d} \mathbf {x} _ {2: t - 1} \\ \times \underbrace {\int p _ {2} \left(\mathbf {x} _ {2} \mid \mathbf {x} _ {1} , \mathbf {x} _ {0}\right) p \left(\mathbf {x} _ {1} \mid \mathbf {x} _ {0}\right) \mathrm{d} \mathbf {x} _ {1}} _ {Q _ {1}}, \tag {30} \\ \end{array}
$$

where we denote the integral of $x_{1}$ as $Q_{1}$ . Based on

$$
\begin{array}{l} Q _ {0} = q \left(\mathbf {x} _ {1} \mid \mathbf {x} _ {0}\right) = \mathcal {N} \left(\mathbf {u} _ {1} \mathbf {x} _ {0}, \mathbf {v} _ {1} ^ {2} \mathbf {I}\right) \\ q _ {2} \left(\mathbf {x} _ {2} \mid \mathbf {x} _ {1}, \mathbf {x} _ {0}\right) = \delta \left(\mathbf {x} _ {2} - \mathbf {u} _ {2} \mathbf {x} _ {0} - \mathbf {v} _ {2} \psi_ {1} ^ {- 1} \left(\mathbf {x} _ {1}\right)\right) \\ = \delta \left[ \mathbf {x} _ {2} - \frac {\mathbf {v} _ {2}}{\mathbf {v} _ {1}} \mathbf {x} _ {1} - \left(\mathbf {u} _ {2} - \frac {\mathbf {v} _ {2} \mathbf {u} _ {1}}{\mathbf {v} _ {1}}\right) \mathbf {x} _ {0} \right] \tag {31} \\ = \delta \left[ \mathbf {x} _ {1} - \frac {\mathbf {v} _ {1}}{\mathbf {v} _ {2}} \mathbf {x} _ {2} - \left(\mathbf {u} _ {1} - \frac {\mathbf {v} _ {1} \mathbf {u} _ {2}}{\mathbf {v} _ {2}}\right) \mathbf {x} _ {0} \right] \\ \end{array}
$$

(Symmetry Property of Dirac Delta Function)

and the Translation Property of the Dirac delta function, we can calculate $Q_{1}$ as:

$$
\begin{array}{l} Q _ {1} = \int \underbrace {p _ {2} (\mathbf {x} _ {2} | \mathbf {x} _ {1} , \mathbf {x} _ {0})} _ {\delta (x - a)} \underbrace {p (\mathbf {x} _ {1} | \mathbf {x} _ {0})} _ {f (x)} \mathrm{d} \mathbf {x} _ {1}, \\ \text { where } \left\{ \begin{array}{l} x: \mathbf {x} _ {1} \\ a: \frac {\mathbf {v} _ {1}}{\mathbf {v} _ {2}} \mathbf {x} _ {2} + \left(\mathbf {u} _ {1} - \frac {\mathbf {v} _ {1} \mathbf {u} _ {2}}{\mathbf {v} _ {2}}\right) \mathbf {x} _ {0} \end{array} \right. \tag {32} \\ \Longrightarrow Q _ {1} = \mathcal {N} (\mathbf {u} _ {2} \mathbf {x} _ {0}, \mathbf {v} _ {2} ^ {2} \mathbf {I}.) \\ \end{array}
$$

Then we can continue the deviation of $p_{t}(\mathbf{x}_{t}|\mathbf{x}_{0})$ as:

$$
\begin{array}{l} p _ {t} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right) = \int Q _ {0} \prod_ {s = 2} ^ {t} p _ {s} \left(\mathbf {x} _ {s} \mid \mathbf {x} _ {s - 1}, \mathbf {x} _ {0}\right) \mathrm{d} \mathbf {x} _ {1: t - 1} \\ = \int Q _ {1} \prod_ {s = 3} ^ {t} p _ {s} \left(\mathbf {x} _ {s} \mid \mathbf {x} _ {s - 1}, \mathbf {x} _ {0}\right) \mathrm{d} \mathbf {x} _ {2: t - 1} \tag {33} \\ = \dots \\ = \int p _ {t} (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}) Q _ {t - 2} \mathrm{d} \mathbf {x} _ {t - 1} \\ = Q _ {t - 1} = \mathcal {N} (\mathbf {u} _ {t} \mathbf {x} _ {0}, \mathbf {v} _ {t} ^ {2} \mathbf {I}) \\ \end{array}
$$

Therefore, the probability distribution of $x_{t}$ conditioned on $x_{0}$ follows a Gaussian distribution $\mathcal{N}(\mathbf{u}_{t}\mathbf{x}_{0},\mathbf{v}_{t}^{2}\mathbf{I})$ , which is the same as in original DDPMs when the coefficient functions are defined as $u_{t}=\sqrt{\bar{\alpha}_{t}}$ and $v_{t}=\sqrt{1-\bar{\alpha}_{t}}$ . This provides an important benefit that the Flow Matching and diffusion models share the same training procedure.

# C.2 Deterministic Reverse Process

The reverse tranfer probability follows Bayes' rule:

$$
\begin{array}{l} p \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {t}, \mathbf {x} _ {0}\right) = p _ {t} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {t - 1}, \mathbf {x} _ {0}\right) \frac {p _ {t - 1} \left(\mathbf {x} _ {t - 1} \mid \mathbf {x} _ {0}\right)}{p _ {t} \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {0}\right)} (34) \\ = \frac {p _ {t - 1} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {0})}{p _ {t} (\mathbf {x} _ {t} | \mathbf {x} _ {0})} \times \delta \left[ \mathbf {x} _ {t} - \frac {\mathbf {v} _ {t}}{\mathbf {v} _ {t - 1}} \mathbf {x} _ {t - 1} - \left(\mathbf {u} _ {t} - \frac {\mathbf {v} _ {t} \mathbf {u} _ {t - 1}}{\mathbf {v} _ {t - 1}}\right) \mathbf {x} _ {0} \right]. (34) \\ \end{array}
$$

Since Dirac delta function has another form of

$$
\delta (x) = \left\{ \begin{array}{c} + \infty , x = 0 \\ 0, x \neq 0 \end{array} , \right. \tag {35}
$$

and $p_t(\mathbf{x}_t|\mathbf{x}_0) > 0$ , $p_{t-1}(\mathbf{x}_{t-1}|\mathbf{x}_t) > 0$ , we have

$$
\begin{array}{l} p (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}, \mathbf {x} _ {0}) = p _ {t} (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}, \mathbf {x} _ {0}) \frac {p _ {t - 1} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {0})}{p _ {t} (\mathbf {x} _ {t} | \mathbf {x} _ {0})} \\ = \left\{ \begin{array}{c} + \infty \times \overbrace {\frac {p _ {t - 1} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {0})}{p _ {t} (\mathbf {x} _ {t} | \mathbf {x} _ {0})}} ^ {> 0}, \quad \mathbf {x} _ {t} = \left[ \frac {\mathbf {v} _ {t}}{\mathbf {v} _ {t - 1}} \mathbf {x} _ {t - 1} + \left(\mathbf {u} _ {t} - \frac {\mathbf {v} _ {t} \mathbf {u} _ {t - 1}}{\mathbf {v} _ {t - 1}}\right) \mathbf {x} _ {0} \right] \\ 0, \quad \mathbf {x} _ {t} \neq \left[ \frac {\mathbf {v} _ {t}}{\mathbf {v} _ {t - 1}} \mathbf {x} _ {t - 1} + \left(\mathbf {u} _ {t} - \frac {\mathbf {v} _ {t} \mathbf {u} _ {t - 1}}{\mathbf {v} _ {t - 1}}\right) \mathbf x _ {0} \right] \end{array} \right. \\ \simeq \left\{ \begin{array}{c} + \infty , \mathbf {x} _ {t - 1} = \left[ \frac {\mathbf {v} _ {t - 1}}{\mathbf {v} _ {t}} \mathbf {x} _ {t} + \left(\mathbf {u} _ {t - 1} - \frac {\mathbf {u} _ {t} \mathbf {v} _ {t - 1}}{\mathbf {v} _ {t}}\right) \mathbf {x} _ {0} \right] \\ 0, \mathbf {x} _ {t - 1} \neq \left[ \frac {\mathbf {v} _ {t - 1}}{\mathbf {v} _ {t}} \mathbf {x} _ {t} + \left(\mathbf {u} _ {t - 1} - \frac {\mathbf {u} _ {t} \mathbf {v} _ {t - 1}}{\mathbf {v} _ {t}}\right) \mathbf {x} \right. \\ \end{array} \right. \tag {36} \\ = \delta \left[ \mathbf {x} _ {t - 1} - \frac {\mathbf {v} _ {t - 1}}{\mathbf {v} _ {t}} \mathbf {x} _ {t} - \left(\mathbf {u} _ {t - 1} - \frac {\mathbf {u} _ {t} \mathbf {v} _ {t - 1}}{\mathbf {v} _ {t}}\right) \mathbf {x} _ {0} \right] \\ = \lim _ {\sigma \to 0} \mathcal {N} \left(\frac {\mathbf {v} _ {t - 1}}{\mathbf {v} _ {t}} \mathbf {x} _ {t} + \left(\mathbf {u} _ {t - 1} - \frac {\mathbf {u} _ {t} \mathbf {v} _ {t - 1}}{\mathbf {v} _ {t}}\right) \mathbf {x} _ {0}, \sigma^ {2} \mathbf {I}\right). \\ \end{array}
$$

# C.3 Deterministic Optimization Objective

We first include the derivation of the variational bound for diffusion models provided by Sohl-Dickstein et al. [2015]. The probability the generative model assigns to the data is:

$$
\begin{array}{l} p (\mathbf {x} _ {0}) = \int p (\mathbf {x} _ {0: T}) \mathrm{d} \mathbf {x} _ {1: T} \\ = \int p (\mathbf {x} _ {0: T}) \frac {p _ {T} (\mathbf {x} _ {1 : T} | \mathbf {x} _ {0})}{p _ {T} (\mathbf {x} _ {1 : T} | \mathbf {x} _ {0})} \mathrm{d} \mathbf {x} _ {1: T} \\ = \int p _ {T} (\mathbf {x} _ {1: T} | \mathbf {x} _ {0}) \frac {p (\mathbf {x} _ {0 : T})}{p _ {T} (\mathbf {x} _ {1 : T} | \mathbf {x} _ {0})} \mathrm{d} \mathbf {x} _ {1: T} \tag {37} \\ = \int p _ {T} (\mathbf {x} _ {1: T} | \mathbf {x} _ {0}) p (\mathbf {x} _ {T}) \prod_ {t = 1} ^ {T} \frac {p (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t})}{p _ {t} (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1})} \mathrm{d} \mathbf {x} _ {1: T}. \\ \end{array}
$$

![](images/93df0e6fc3bcfd63b5c43d56319474bda7fa7fa836ba31e543c9a062a71ccc46.jpg)

<details>
<summary>text_image</summary>

Markovian Diffusion Process
</details>

![](images/5361606080460e469752fcc28bd645b7f268dbf7b7fcf5174132eadcceda9917.jpg)

<details>
<summary>text_image</summary>

Deterministic Diffusion Process
</details>

![](images/cd5313ce8a19ae7c9ca1dd5204f1fad4dd714ee2363c4c9f0431717ba1957794.jpg)

<details>
<summary>text_image</summary>

Flow Matching
</details>

Figure 4: We demonstrate the trajectory differences among Markovian Diffusion Process, Deterministic Diffusion and Flow Matching.

Training amounts to minimizing the negative log likelihood:

$$
\begin{array}{l} \mathcal {L} = - \int p (\mathbf {x} _ {0}) \log p (\mathbf {x} _ {0}) \mathrm{d} \mathbf {x} _ {0} \\ = - \int p (\mathbf {x} _ {0}) \log \left[ \int p _ {T} (\mathbf {x} _ {1: T} | \mathbf {x} _ {0}) p (\mathbf {x} _ {T}) \prod_ {t = 1} ^ {T} \frac {p (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t})}{p _ {t} (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1})} \mathrm{d} \mathbf {x} _ {1: T} \right] \mathrm{d} \mathbf {x} _ {0} \\ \leq - \int p _ {T} (\mathbf {x} _ {0: T}) \log \left[ p (\mathbf {x} _ {T}) \prod_ {t = 1} ^ {T} \frac {p (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t})}{p _ {t} (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1})} \right] \mathrm{d} \mathbf {x} _ {0: T} \\ = \mathbb {E} _ {p _ {T} (\mathbf {x} _ {0: T})} \left[ - \log p (\mathbf {x} _ {T}) + \sum_ {t = 1} ^ {T} \log \frac {p _ {t} (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1})}{p (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t})} \right] \\ = \mathbb {E} _ {p _ {T}} \left[ \log \frac {p _ {T} (\mathbf {x} _ {T} | \mathbf {x} _ {0})}{p (\mathbf {x} _ {T})} - \log p (\mathbf {x} _ {0} | \mathbf {x} _ {1}) + \sum_ {t = 2} ^ {T} \log \frac {p (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t} , \mathbf {x} _ {0})}{p (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t})} \right] \\ = \mathbb {E} _ {p _ {T}} \Biggl [ \underbrace {D _ {\mathrm{KL}} (p _ {T} (\mathbf {x} _ {T} | \mathbf {x} _ {0}) | | p (\mathbf {x} _ {T}))} _ {\mathcal {L} _ {T}} \underbrace {- \log p (\mathbf {x} _ {0} | \mathbf {x} _ {1})} _ {\mathcal {L} _ {0}} + \sum_ {t = 2} ^ {T} \underbrace {D _ {\mathrm{KL}} (p (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t} , \mathbf {x} _ {0}) | | p (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}))} _ {\mathcal {L} _ {t - 1}} \Biggr ] \\ \end{array}
$$

where $L_{T}$ is usually ignored as a constant and $p(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ is parameterized with a neural network $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ to approximate the conditioned probability distributions in the reverse process. Since $p(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_{0}) = \lim_{\sigma \to 0} \mathcal{N}\left(\frac{\mathbf{v}_{t-1}}{\mathbf{v}_{t}}\mathbf{x}_{t} + \left(\mathbf{u}_{t-1} - \frac{\mathbf{u}_{t}\mathbf{v}_{t-1}}{\mathbf{v}_{t}}\mathbf{x}_{0}\right), \sigma^{2}\mathbf{I}\right)$ , the parameterized $p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_{t})$ can take the same form $\mathcal{N}(\boldsymbol{\mu}_{\theta}(\mathbf{x}_{t}, t), \sigma_{t}^{2}\mathbf{I})$ because the Dirac delta function is a special case of Gaussian distribution and the KL divergence of two Gaussians can be simplified. Finally, the training objective for the deterministic diffusion process is divided as:

$$
\mathcal {L} = \left\{\begin{array}{l}\mathcal {L} _ {T}: \text { a   constant }\\\mathcal {L} _ {0}: - \log \delta \left(\mathbf {x} _ {0} - \mathbf {x} _ {\theta} (\mathbf {x} _ {1}, 1)\right)\\\mathcal {L} _ {t - 1}: c \| \mathbf {x} _ {0} - \mathbf {x} _ {\theta} (\mathbf {x} _ {t}, t) \| ^ {2} + \lim _ {\sigma \rightarrow 0} \log \frac {\sigma_ {t}}{\sigma}\\c = \frac {1}{2 \sigma_ {t} ^ {2}} \left(\mathbf {u} _ {t - 1} - \frac {\mathbf {u} _ {t} \mathbf {v} _ {t - 1}}{\mathbf {v} _ {t - 1}}\right) ^ {2},\end{array}\right. \tag {38}
$$

where the simplified version $\|\mathbf{x}_{0}-\mathbf{x}_{\theta}(\mathbf{x}_{t},t)\|^{2}$ is the same as DDPMs but with different coefficients.

# D Different Diffusion Trajectories

We illustrate the trajectories of different diffusion processes in Figure 4. The forward and reverse generation for the Markovian diffusion process is:

$$
\left\{ \begin{array}{c} \mathbf {x} _ {t} = \sqrt {\bar {\alpha} _ {t}} \mathbf {x} _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \boldsymbol {\epsilon} _ {t} \\ \mathbf {x} _ {t - 1} = \frac {\sqrt {\bar {\alpha} _ {t - 1}} (1 - \alpha_ {t})}{1 - \bar {\alpha} _ {t}} \mathbf {x} _ {0} + \frac {\sqrt {\alpha_ {t}} (1 - \bar {\alpha} _ {t - 1})}{1 - \bar {\alpha} _ {t}} \mathbf {x} _ {t} \\ + \frac {\sqrt {(1 - \bar {\alpha} _ {t - 1}) (1 - \alpha_ {t})}}{\sqrt {1 - \bar {\alpha} _ {t}}} \boldsymbol {\epsilon} _ {t - 1}. \end{array} \right. \tag {39}
$$

The deterministic diffusion process:

$$
\left\{ \begin{array}{c} \mathbf {x} _ {t} = \sqrt {\bar {\alpha} _ {t}} \mathbf {x} _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \boldsymbol {\epsilon} \\ \mathbf {x} _ {t - 1} = \left(\sqrt {\bar {\alpha} _ {t - 1}} - \frac {\sqrt {\bar {\alpha} _ {t} (1 - \bar {\alpha} _ {t - 1})}}{\sqrt {1 - \bar {\alpha} _ {t}}}\right) \mathbf {x} _ {0} \\ + \frac {\sqrt {1 - \bar {\alpha} _ {t - 1}}}{\sqrt {1 - \bar {\alpha} _ {t}}} \mathbf {x} _ {t}. \end{array} \right. \tag {40}
$$

The deterministic flow matching with optimal transport:

$$
\left\{ \begin{array}{l} \mathbf {x} _ {t} = (1 - \frac {t}{T}) \mathbf {x} _ {0} + \frac {t}{T} \boldsymbol {\epsilon} \\ \mathbf {x} _ {t - 1} = \frac {1}{t} \mathbf {x} _ {0} + \frac {t - 1}{t} \mathbf {x} _ {t}. \end{array} \right. \tag {41}
$$

# E Details of the Function G

Equation (13) defines the function $G(\mathbf{x}, \epsilon)$ as the inversion of coefficient function.

Flow Matching The coefficient is $u_{t}=1-t/T$ , where $t=T\times(1-u_{t})$ . Therefore,

$$
G (\mathbf {x} _ {0}, \boldsymbol {\epsilon}) = t _ {0} = T \times (1 - \mathbf {u} _ {t _ {0}}) = T / \left(1 + \frac {f (\boldsymbol {\epsilon} , \mathcal {J}) - f (\boldsymbol {\epsilon} , \mathcal {I})}{f (\mathbf {x} _ {0} , \mathcal {I}) - f (\mathbf {x} _ {0} , \mathcal {J})}\right) \tag {42}
$$

Diffusion The coefficient for Variance Exploding is $v_{T} = \sigma_{0}\left(\frac{\sigma_{T}}{\sigma_{0}}\right)^{\frac{t}{T}}$ , where $t = T \times \frac{\log v_{t} - \log \sigma_{0}}{\log \sigma_{T} - \log \sigma_{0}}$ .

$$
G \left(\mathbf {x} _ {0}, \boldsymbol {\epsilon}\right) = t _ {0} = \mathbf {v} _ {t _ {0}} = T \times \frac {\log \mathbf {v} _ {t} - \log \sigma_ {0}}{\log \sigma_ {T} - \log \sigma_ {0}} = T \times \frac {\log \frac {f (\boldsymbol {\epsilon} , \mathcal {I}) - f (\boldsymbol {\epsilon} , \mathcal {I})}{f \left(\mathbf {x} _ {0} , \mathcal {I}\right) - f \left(\mathbf {x} _ {0} , \mathcal {I}\right)} - \log \sigma_ {0}}{\log \sigma_ {T} - \log \sigma_ {0}}. \tag {43}
$$

For Variance Preserving, the function $G(\mathbf{x}_{0}, \epsilon)$ is more difficult to calculate since $u_{t} = \sqrt{\bar{\alpha}_{t}}$ , where $\bar{\alpha} = \prod_{i=1}^{t} \alpha_{i}$ , $\alpha_{t} = 1 - \beta_{t}$ , and $\beta_{t}$ is also influenced by noise schedulers. This makes $G(\mathbf{x}_{0}, \epsilon)$ hard to calculate. Fortunately, we can bypass this function and provide the corresponding pseudo code.

# F Details of the Training Objective

The rescaled vector field is calculated as:

$$
\begin{array}{l} \tilde {u} _ {t} = \frac {\mathrm{d} \tilde {\mathbf {x}} _ {t}}{\mathrm{d} t} = \frac {\mathrm{d} \tilde {\mathbf {x}} _ {t}}{\mathrm{d} \tau} \frac {\mathrm{d} \tau}{\mathrm{d} t} \\ = \left[ \mathbf {u} ^ {\prime} \left(\mathbf {x} _ {0}, \tau\right) \mathbf {x} _ {0} + \mathbf {v} ^ {\prime} \left(\mathbf {x} _ {0}, \tau\right) \boldsymbol {\epsilon} \right] \frac {T - r \times G \left(\mathbf {x} _ {0} , \boldsymbol {\epsilon}\right)}{T} \tag {44} \\ = u _ {\tau} \times \frac {T - r \times G (\mathbf {x} _ {0} , \boldsymbol {\epsilon})}{T}. \\ \end{array}
$$

Considering the expectation form of $\tilde{u}_t$ , there is:

$$
\begin{array}{l} \mathbb {E} _ {\tilde {\mathbf {x}} _ {t}} \left[ \tilde {u} _ {t} (\tilde {\mathbf {x}} _ {t} | \mathbf {x} _ {0}) \right] = \sum p (\tilde {\mathbf {x}} _ {t} | \mathbf {x} _ {0}) \tilde {u} _ {t} (\tilde {\mathbf {x}} _ {t} | \mathbf {x} _ {0}) \\ = \sum p (\tilde {\mathbf {x}} _ {t} | \mathbf {x} _ {0}) \left[ \mathbf {u} ^ {\prime} \left(\mathbf {x} _ {0}, \tau\right) \mathbf {x} _ {0} + \mathbf {v} ^ {\prime} (\mathbf {x} _ {0}, \tau) \boldsymbol {\epsilon} \right] \underbrace {\frac {T - r \times G (\mathbf {x} _ {0} , \boldsymbol {\epsilon})}{T}} _ {0 \leq \text { coefficient } \leq 1} \\ \leq \sum p (\tilde {\mathbf {x}} _ {t} | \mathbf {x} _ {0}) \left[ \mathbf {u} ^ {\prime} \left(\mathbf {x} _ {0}, \tau\right) \mathbf {x} _ {0} + \mathbf {v} ^ {\prime} (\mathbf {x} _ {0}, \tau) \boldsymbol {\epsilon} \right] \tag {45} \\ = \mathbf {u} ^ {\prime} (\mathbf {x} _ {0}, \tau) \left[ \sum p (\tilde {\mathbf {x}} _ {t} | \mathbf {x} _ {0}) \mathbf {x} _ {0} \right] + \mathbf {v} ^ {\prime} (\mathbf {x} _ {0}, \tau) \boldsymbol {\epsilon} \\ = \tilde {u} _ {t} (\tilde {\mathbf {x}} _ {0} | \mathbb {E} _ {\tilde {\mathbf {x}} _ {t}} [ \mathbf {x} _ {0} ]). \\ \end{array}
$$

Therefore, the training objective $E\|\tilde{u}_{t}-\tilde{u}_{\theta}\|^{2}\leq c$ $E\|x_{0}-x_{\theta}\|^{2}$ , where c is the coefficient.

# G Code Implementations

Our framework is a module constructed on current diffusion models. We demonstrate our kernel part rescale diffusion trajectory with pseudo python code as below:

```python
def rescale_diffusion_trajectory(x_0, epsilon, embedding,
    labels, alphas_cumprod, timesteps, mode):
    #embedding: embedding matrix, f(x,i)=(embedding * x)[i]
    #labels: I
    #alphas_cumprod: list of all u_t
    #timesteps: t
    #mode: noising or denoising

#1. get f(x,i):
self_dot = torch.sum(embedding * embedding, dim=-1)
f_x_i = self_dot[labels][..., None]
labels = labels[..., None]

#2. get f(x,j) and f(eps,j):
embedding = embedding.permute(1, 0)
f_x_j = torch.matmul(x_0, embedding)
f_eps_j = torch.matmul(epsilon, embedding)

#3. get f(x,i) - f(x,j): (usually >=0; smaller -> closer)
#filter out f(x,i)-f(x,i) with a large positive number 100
fxi_minus_fxj = (f_x_i - f_x_j).scatter(-1, labels, 100)

#4. get f(eps,i) and f(eps,j) - f(eps,i): (larger -> more noise)
f_eps_i = torch.gather(f_eps_j, -1, labels)
#filter out f(eps,i)-f(eps,i) with a large negative number -100
fepsj_minus_fepsi = (f_eps_j - f_eps_i).scatter(-1, labels, -100)

#5. get fraction and u_t_0
#mask results outside the support set
info_mask = (fepsj_minus_fepsi < 0) | (fxi_minus_fxj < 0)
fraction = fix_minus_fjx / fjeps_minus_fieps
fraction[info_mask] = 100
min_frac, _ = fraction.min(dim=-1) # minimum
#Diffusion Variance Preserving eq. (9)
u_t_0 = torch.sqrt(1 / (1 + min_frac ** 2))[..., None]

#6. rescale timesteps
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) 
```

```python
###!!!important trick!!!!###
#We do not need to calculate the function G(x_0,t) (eq. (12)).
#Timesteps of diffusion processes are discrete and
# we just iterate over and compare with all coefficient functions.
#Besides, function G is easy to calculate for Flow Matching.
index = torch.sum(u_t_0 < sqrt_alphas_cumprod, dim=-1)

#T is the maximum timestep, for example T=2000.
#confactor is the confidence factor
#tau is the rescaled timestep
#delta_tau is the rescaled decoding velocity
if mode == 'noising':
    tau = (timesteps + index - \
    (((timesteps + 1) / T) * index)).long().clamp(0, T)
    tau = (confactor * tau.float() + \
    (1.0 - confactor) * timesteps.float()).long().clamp(0, T)
    return tau
elif mode == 'denoising':
    delta_tau = (T - index) / T
    delta_tau = (confactor * delta_tau + \
    (1 - confactor) * 1.0).clamp(0, 1)
    return delta_tau 
```

Table 6: FID of difference sampling strategies. 

<table><tr><td></td><td>Gaussian</td><td>Deterministic</td></tr><tr><td>BINARY CODING</td><td>13.39</td><td>3.86</td></tr><tr><td>FIXED EMBEDDING</td><td>12.21</td><td>9.15</td></tr><tr><td>TRAINABLE EMBEDDING</td><td>22.24</td><td>10.99</td></tr></table>

# H Analysis

Gaussian Sampling Our framework is compatible with the Gaussian sampling in DDPM, where random noises can be added into each iteration step. Algorithm 3 demonstrates the Gaussian sampling procedure. Compared with algorithm 2, a Gaussian noise $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma_t^2\mathbf{I})$ with a decreasing variance $\sigma_t$ is injected to the estimated next state $\tilde{\mathbf{x}}_t$ . This noise $\mathbf{z}$ will be mapped as changing the initial sampling $\tilde{\mathbf{x}}_T$ through the trajectory alteration step. We illustrate the deterministic and Gaussian sampling for our model on CIFAR-10 in Table 6, where the deterministic sampling can achieve a much better performance of FID. We assume this is because our coefficient functions $\mathbf{u}(\mathbf{x}_0, t)$ and $\mathbf{v}(\mathbf{x}_{0},t)$ are dynamically calculated to rescale the deterministic trajectory in the training stage. In the inference stage, $\mathbf{x}_{0}$ is replaced by $\mathbf{x}_{\theta}(\mathbf{x}_{t},t)$ , where errors will accumulate if the predicted pseudo target changes frequently. Moreover, Gaussian sampling will further introduce random noises at each reverse step, making our rescaled timestep $\tau$ far away from the training situation. Therefore, errors in the calculations of trajectory scaling will explode over iterations.

Algorithm 3 Gaussian Sampling 

<table><tr><td>1:</td><td> $t := T, \tau := T$ </td><td></td></tr><tr><td>2:</td><td> $\tilde{\mathbf{x}}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ </td><td>// Initialing</td></tr><tr><td>3:</td><td>for  $\Delta t := \Delta t_1, \ldots, \Delta t_s$  do</td><td> $// \sum_{\Delta t} = T$ </td></tr><tr><td>4:</td><td> $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma_t^2 \mathbf{I})$ </td><td>// Gaussian Noise</td></tr><tr><td>5:</td><td> $\hat{\mathbf{x}}_0 := \mathbf{x}_\theta(\tilde{\mathbf{x}}_t, t)$ </td><td>// Pseudo Target</td></tr><tr><td>6:</td><td> $\hat{\epsilon} := \Psi^{-1}([\tilde{\mathbf{x}}_t; \tau])$ </td><td>// Trajectory Alteration</td></tr><tr><td>7:</td><td> $\tau_\Delta := \mathcal{T}(t - \Delta t, G(\hat{\mathbf{x}}_0, \hat{\epsilon}))$ </td><td>// eq. (25)</td></tr><tr><td>8:</td><td> $\tilde{\mathbf{x}}_t := \mathbf{u}(\hat{\mathbf{x}}_0, \tau_\Delta) \hat{\mathbf{x}}_0 + \mathbf{v}(\hat{\mathbf{x}}_0, \tau_\Delta) \hat{\epsilon} + \mathbf{z}$ </td><td></td></tr><tr><td>9:</td><td> $t := t - \Delta t, \tau := \tau_\Delta$ </td><td>// Updating</td></tr><tr><td>10:</td><td>end for</td><td></td></tr><tr><td>11:</td><td> $\mathbf{x}_0 := \mathbf{x}_\theta(\tilde{\mathbf{x}}_t, t)$ </td><td>//  $\mathbf{x}_1 \to \mathbf{x}_0$ </td></tr><tr><td>12:</td><td>return  $\mathbf{x}_0$ </td><td></td></tr></table>

# I Limitations

Our framework is proposed to migrate the powerful continuous diffusion models to discrete problems. There is another technical route that directly designs the diffusion process on the discrete state space and our method is not useful for this scenario. However, we believe the continuous diffusion models can be a general framework for generative modeling and our effort can advance this target.

We prefer $x_{0}$ as the training target because we highly depend on the reliability of the predicted $\hat{x}_{0}$ during inference. Although it is possible to use other targets, the modeling effect will decrease in practical use, which limits the flexibility of diffusion modeling. For example, predicting the $\hat{\epsilon}$ and recovering $\hat{x}_{0}$ with eq. (23) is inefficient, because a small error in predicting $\hat{\epsilon}$ will be amplified by eq. (23) and lead to the collapse of $G(\hat{\mathbf{x}}_{0}, \hat{\boldsymbol{\epsilon}})$ .

Our approach requires extra computational cost. But they are acceptable since our rescaling process is a series of parallel matrix computations. Considering that our approach is compatible with the Self-Conditioning [Chen et al., 2023b], our overhead is negligible when it is used.

# J Other Experimental Details

For language modeling, we utilize the model configuration transformer-iwslt-de-en in FAIRSEQ framework [Ott et al., 2019] for IWSLT14 DE-EN, which has 6 transformer layers, 4 attention heads, 512 hidden dimensions, and 1024 feed forward layer dimensions. For other datasets, the configuration is transformer-base, which has 6 transformer layers, 8 attention heads, 512 hidden dimensions, and 2048 feed forward layer dimensions. The embedding dimension is 128. The beam size is 1 length prediction beam × 5 generation beam, since the length prediction is unstable for diffusion language models. For reranking, we take 7 length prediction beam × 3 generation beam as Diformer to let the transformer choose the best one.

![](images/cddca21f3806b08ffc0415f0f379eba688299140205489eac704f93f20814ddd.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 colorful images including animals, birds, horses, and vehicles (no text or symbols)
</details>

(A) Bit Diffusion repro (FID 10.37)

![](images/c7be35ea9a23594e16fac4d10ea5bb6339eb3fb9745d33f0a91e853503b330a8.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 diverse images including animals, vehicles, and food items (no text or symbols)
</details>

(B) Ours (FID 3.86)   
Figure 5: Generated BINARY CODING images of reproduced Bit Diffusion and Ours on CIFAR-10.

For image generation, we set the scaling factor r = 0.5 for training. Besides, we find that a smaller factor for inference is sometime useful. We set r = 0.45 on binary coding and r = 0.2 on fixed embedding during inference. When the pixel embedding is learnable, the scaling factor is r = 0.5, which is the same as training.

Our experiments are performed with Nvidia 80G A100. Each language result requires about 2 days on one single A100. Each image result requires about a week on one single A100.

# K Impact Statements

This paper presents work whose goal is to advance the field of Deep Learning. The datasets we used has been widely deployed for many years and has basically no negative impact. Our approach is a framework that migrates existing diffusion models to discrete problems, which does not provide a large pre-trained model that can be used to generate fake contents.

# L Case Study

Generated sentences on IWSLT14 DE-EN and GIGAWORD are illustrated in Table 7 and Table 8. Generated images on CIFAR-10 are depicted in Figure 5, 6, and 7.

Table 7: Cases of translation on IWSLT14 DE-EN. 

<table><tr><td rowspan="2">Source: GERMAN</td><td colspan="3">Target: ENGLISH</td></tr><tr><td>Differmer</td><td>Ours</td><td>Golden</td></tr><tr><td>ich möchte ihnen erzählen, wie wir das herausgefunden haben.</td><td>i want to tell you about this.</td><td>i want to tell you how we&#x27;ve figured that out.</td><td>i want to tell you how we found that out.</td></tr><tr><td>da gingen ganz schön viele verrückte dinge vor sich.</td><td>lots of crazy things.</td><td>there were quite a lot of crazy things going on.</td><td>there was a whole lot of crazy going on in there.</td></tr><tr><td>man macht etwas, das eigentlich ein wenig anders ist.</td><td>you do something a little different.</td><td>you&#x27;re doing something that&#x27;s actually a little bit different.</td><td>you do something that&#x27;s actually a little different.</td></tr><tr><td>und die welt in der wir lebten sah so aus.</td><td>and the world we lived like this.</td><td>and the world we lived in looked like this.</td><td>and the world we used to live in looked like this.</td></tr><tr><td>man erwartet eine zusätzliche milliarde spieler im nächsten jahrzehnt.</td><td>you&#x27;ll expect an next billion players.</td><td>you expect an extra billion players in the next decade.</td><td>they expect one billion more gamers in the next decade.</td></tr><tr><td>b hat diese vorteile und risiken. was wollen sie tun?</td><td>b has risks. what do you want to do?</td><td>b has these benefits and risks. what do you want to do?</td><td>b has these benefits, and these risks. what do you want to do?</td></tr><tr><td>wir haben also so eine situation, wo, je weiter unsere wissenschaft fortschreitet, wir uns um so mehr eingestehen müssen, dass diese kategorien, die wir für stabile anatomische kategorien gehalten hatten, welche sehr einfache zuordnungen herstellten um dauerhafte identitätskategorien zu schaffen, viel unschärfer sind, als wir angenommen haben.</td><td>so we have this situation where the continuing our science continues, we need to admit the more that these categories that we thought were stable anatomical categories, which made a very simple collaborations to create permanent identity ories are much unsharers than we&#x27;ve assumed.</td><td>so we have a situation where, as the further our science goes on, we have to admit in terms, the more that these categories that we thought of be a stable anatomical categories, which made a very simple assa-ments to create permanent identity cat-egories, are much more blanky than we&#x27;ve accepted.</td><td>so what we have is a sort of situation where the farther our science goes, the more we have to admit to ourselves that these categories that we thought of as stable anatomical categories that mapped very simply to stable identity categories are a lot more fuzzy than we thought.</td></tr></table>

Table 8: Cases of summarization on GIGAWORD. 

<table><tr><td rowspan="2">Source</td><td colspan="3">Target</td></tr><tr><td>Diformer</td><td>Ours</td><td>Golden</td></tr><tr><td>the asian swimming record tumbled again at the seven-day olympic test event here on friday .</td><td>asian swimming record falls again</td><td>asian swimming tumble again at olympic test event</td><td>asian swimming record tumbles again at china&#x27;s olympic trials</td></tr><tr><td>a truck carrying illegal north african immigrants flipped over in northeastern spain , killing ## and injuring six others , police said monday .</td><td>truck carrying illegal immigrants crashes in spain killing ##</td><td>## illegal immigrants killed in truck accident in north-eastern spain</td><td>## immigrants killed in road accident in spain</td></tr><tr><td>new zealand share prices closed #.## percent lower wednesday after investors took their lead from further weakness in overseas markets , dealers said .</td><td>new zealand shares fall #.## percent</td><td>new zealand shares close #.## percent lower</td><td>new zealand shares close down #.## percent</td></tr><tr><td>the sudanese opposition said here thursday it had killed more than ### government soldiers in an ambush in the east of the country .</td><td>sudanese opposition claims over ### soldiers killed</td><td>sudanese opposition claims ### soldiers killed in ambush</td><td>sudanese opposition says ### government troops killed in ambush</td></tr><tr><td>these sports stories for release tuesday , september ## , ####, are moving today to clients of the new york times news service .</td><td>thursday &#x27;s sports budget</td><td>cox news service sports budget</td><td>cox news service tuesday sports budget</td></tr><tr><td>bangladesh and india signed a deal here thursday giving green signal to resumption of passenger train service between the two neighboring countries after ## years .</td><td>bangladesh india sign agreement on train service</td><td>bangladesh india sign agreement to resume train service</td><td>bangladesh india sign agreement for resumption of train service after ## years</td></tr></table>

![](images/e36c198b8fe2d970b7add95d449cc506868fe3eed19bec05e561c7d52cb9b8fe.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 nature and wildlife images including animals, vehicles, birds, and landscapes (no text or symbols)
</details>

(A) Bit Diffusion repro (FID 12.96)

![](images/53a9efa383cc54b06899163ed25e8e70b243f8dadcb432edf76a0f027638f76f.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 diverse images including animals, vehicles, and landscapes, each with unique color and shape (no text or symbols)
</details>

(B) Ours (FID 9.15)   
Figure 6: Generated FIXED EMBEDDING images of reproduced Bit Diffusion and Ours on CIFAR-10.

![](images/1742cc3ec93c62db7fc5e541bf44794981043cc21b428c34d3fdcade711a3066.jpg)  
Figure 7: Generated TRAINABLE EMBEDDING images of reproduced Bit Diffusion and Ours on CIFAR-10.