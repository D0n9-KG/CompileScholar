# On the Generalization Properties of Diffusion Models

Puheng Li\*||¶

Zhong Li $^{\dagger\parallel**}$

Huishuai Zhang $^{\ddagger}$

Jiang Bian $^{§}$

# Abstract

Diffusion models are a class of generative models that serve to establish a stochastic transport map between an empirically observed, yet unknown, target distribution and a known prior. Despite their remarkable success in real-world applications, a theoretical understanding of their generalization capabilities remains underdeveloped. This work embarks on a comprehensive theoretical exploration of the generalization attributes of diffusion models. We establish theoretical estimates of the generalization gap that evolves in tandem with the training dynamics of score-based diffusion models, suggesting a polynomially small generalization error $O(n^{-2/5} + m^{-4/5})$ on both the sample size n and the model capacity m, evading the curse of dimensionality (i.e., not exponentially large in the data dimension) when early-stopped. Furthermore, we extend our quantitative analysis to a data-dependent scenario, wherein target distributions are portrayed as a succession of densities with progressively increasing distances between modes. This precisely elucidates the adverse effect of “modes shift” in ground truths on the model generalization. Moreover, these estimates are not solely theoretical constructs but have also been confirmed through numerical simulations. Our findings contribute to the rigorous understanding of diffusion models’ generalization properties and provide insights that may guide practical applications.

# 1 Introduction

As an emerging family of deep generative models, diffusion models (DMs; [16, 45]) have experienced a surge in popularity, owing to their unparalleled performance in a wide range of applications ([22, 39, 68, 15, 3, 35, 4, 60, 33, 69, 64, 63, 53]). This has led to notable commercial successes, such as DALL·E ([34]), Imagen ([38]), and Stable Diffusion ([36]). Mathematically,

diffusion models learn an unknown underlying distribution through a two-stage process: (i) first, successively and gradually injecting random noises (forward process); (ii) then reversing the forward process through denoising for sampling purposes (reverse process). To achieve this, an equivalent formulation of diffusion models called score-based generative models (SGMs; [50, 52]) is employed. SGMs implement the aforementioned two-stage process via the continuous dynamics represented by a joint group of coupled stochastic differential equations (SDEs) ([54, 20]).

Despite their impressive empirical performance, the theoretical foundation of DMs/SGMs remains underexplored. Generally, fundamental theoretical questions can be categorized into several aspects. By considering machine learning models as mathematical function classes from certain spaces, one can identify three central aspects: approximation, optimization and generalization. At the forefront lies the generalization problem, which aims to characterize the learning error between the learned and ground truth distributions.

The development of generalization theory for diffusion models is pressing due to both theoretical and practical concerns:

- In theory, the generalization issues of generative modeling (or learning for distributions) may exhibit as the memorization phenomenon, if the modeled distribution is eventually trained to converge to the empirical distribution only associated with training samples. Intuitively, memorization arises from two reasons: (i) it is useful for the hypothesis space to be large enough to approximate highly complex underlying target distributions (universal convergence; [65]); (ii) the underlying distribution is unknown in practice, and one can only use a dataset with finite samples drawn from the target distribution. Rigorous mathematical characterizations of memorization are developed for bias potential models and GANs in [66] and [67], respectively. A natural question is, does a similar phenomenon occur for diffusion models? To answer this, a thorough investigation of generalization properties for DMs/SGMs is required.   
- In practice, the generalization capability of diffusion models is also an essential requirement, as the memorization can lead to potential privacy and copyright risks when models are deployed. Similar to other generative models and large language models (LLMs) [5, 70, 19, 6], diffusion models can also memorize and leak training samples [5, 46], hence can be subsequently attacked using specific procedures and algorithms [28, 18, 62]. Although there are defense methods developed to meet privacy and copyright standards ([11, 14, 58]), these approaches are often heuristic, without providing sufficient quantitative understandings particularly on diffusion models. Therefore, a comprehensive investigation of the generalization foundation of diffusion models, including both theoretical and empirical aspects, is of utmost importance in improving principled tutorial guidance in practice.

The current work develops the generalization theory of diffusion models in a mathematically rigorous manner. Our main results include the following:

- We derive an upper bound of the generalization gap for diffusion models along the training dynamics. This result suggests, with early-stopping, the generalization error of diffusion models scales polynomially small on the sample size $(O(n^{-2/5}))$ and the model capacity $(O(m^{-4/5}))$ . Notably, the generalization error also escapes from the curse of dimensionality.   
- This “uniform” bound is further extended to a data-dependent setting, where a sequence of unidimensional Gaussian mixtures distributions with an increasing modes’ distance is considered as the ground truth. This result characterizes the effect of “modes shift” quantitatively, which implies that the generalization capability of diffusion models is adversely affected by the distance between high-density regions of target distributions.   
- The theoretical findings are numerically verified in simulations.

The rest of this paper is organized as follows. In Section 2, we discuss the related work on the convergence and training fronts of diffusion models and also the generalization aspects of other generative modeling methods. Section 3 is the central part, which includes the problem formulation, main results, and consequences. Section 4 includes numerical verifications on synthetic and real-world datasets $^{1}$ . All the details of proofs and experiments are found in the appendices.

# 2 Related Work

We review the related work on diffusion models concerning the central results in this paper.

- First, on the convergence theory, [7, 9] established elaborate error estimates between the modeled and target distribution given the discretization and time-dependent score matching tolerance. Compared to the present work, they did not evolve the concrete training dynamics since the setting therein focuses on the properties of optimizers.   
- Second, on the training front, [51] proposed a set of techniques to enhance the training performance of score-based generative models, scaling diffusion models to images of higher resolution, but without any characterization of possible generalization improvements. Similar to [7, 9], [49] also provided error estimates between the modeled and target distributions in a point-wise sense and again did not evolve the detailed training dynamics.   
- Third, on the generalization and memorization side, corresponding theories are developed for bias potential models and GANs in [66] and [67], respectively, where the modeled distribution learns the ground truth with early-stopping and diverges or converges to the empirical distribution only associated with training samples after sufficiently long training

time. The current work extends the mathematical analysis to the case of diffusion models under a data-dependent setting.

\- As a supplement, we also discuss related literature regarding low-density learning. [50] illustrated the difficulty of learning from low-density regions with toy formulations and simulations, which motivates the sampling method of annealed Langevin dynamics as the predecessor of (score-based) diffusion models. [21] restudied similar problems under the pure score matching regime (without the denoising or time-dependent dynamics) and attributed the difficulty to increasing isoperimetry of distributions with modes shift. [41] defined the Hardness score and numerically justified that decreasing manifold densities leads to the increasing Hardness score, and consequently applied the Hardness score as the regularization to the sampling process to enhance synthetic images from low-density regions. As a comparison, this work establishes a mathematically rigorous estimate on the generalization gap that quantitatively depends on the range of low-density regions (between modes) in target distributions for (score-based) diffusion models that requires denoising.

# 3 Formulation and Results

In this section, we first introduce the problem setup. Next, we state the main theoretical results, subsequent consequences, and possible connections. The numerical illustration is provided at last.

# 3.1 Problem Formulation

Since DMs/SGMs have already grown into a large family of generative models with an enormous number of variants, there are various ways to define the parameterization of diffusion models. Here, we adopt (one of) the most fundamental architectures proposed in $[54]$ , where the forward perturbation and reverse sampling process are both implemented by a joint group of coupled (stochastic) differential equations. See Figure 1 for an illustration of the problem formulation.

Forward perturbation. We start with the setting of unsupervised learning. Given an unlabeled dataset $D_{x} = \{x_{i}\}_{i=1}^{n} \subset R^{d}$ with the sample $x_{i} \stackrel{i.i.d.}{\sim} p_{0}(x)$ , where $p_{0}$ denotes the underlying (ground truth or target) distribution, the forward diffusion process is defined as

$$
d \boldsymbol {x} = \boldsymbol {f} (\boldsymbol {x}, t) d t + g (t) d \boldsymbol {W} _ {t}, \quad \boldsymbol {x} (0) \sim p _ {0}. \tag {1}
$$

Here, the drift coefficient $\boldsymbol{f}(\cdot,t):\mathbb{R}^{d}\mapsto\mathbb{R}^{d}$ is a time-dependent vector-valued function, and the diffusion coefficient $g(\cdot):\mathbb{R}_{\geq0}\mapsto\mathbb{R}$ is a scalar function, and $W_{t}$ denotes the standard Wiener

![](images/da1ca592b949620562b394b6ded9ec17fd35998f38092b4bfefc84091116400b.jpg)

![](images/3e8cbdbd252900d2a895bdd5fc859611e1698e9a5f2ff9fdb4dca80b3cb6292a.jpg)

![](images/ee34aff4e9ce6f905ba14c1ffe123d460cc0edbfa6fb0fe056cf4eb3ac871e12.jpg)

![](images/3aec90c36588c894b7f7109e5ef12c986ede912e98196abffd9ce2822203d97e.jpg)

![](images/1df307a1b9644f9f9814e55d49544cb1c2c5df6aa1cbdd518bbade6edfe28b5c.jpg)

![](images/3f095fcd34cc28a936480ea1776a66f26eab68a203c2924b5e0fe772a979fa2e.jpg)

![](images/d8cd5bb74884a696c2d5b0f36a3a8f695d34fbaf2bf4c9f7aed0658082aff2da.jpg)

$$
\pmb{x}(0) \longleftarrow d\pmb{x} = \left[ \pmb{f}(\pmb{x}, t) - g^2(t) \underbrace{\nabla_{\pmb{x}} \log p_t(\pmb{x})}_{\approx s_{t,\theta}(\pmb{x}) := \frac{1}{m} A \sigma(\pmb{W}\pmb{x} + Ue(t)), \pmb{\theta} = A} \right] dt + g(t) d\bar{\pmb{W}}_t \longrightarrow \pmb{x}(T)
$$

Target: Finitely-supported prob. & Gaussian mixtures

Notations:

Loss: Time-dependent score matching (Eq. (7))

t: SDE time T: maximal SDE time

Algorithm: Gradient flow

$p_T \approx \pi$ : a known prior

$\tau$ : training time

Figure 1: Illustration of the problem formulation and important notations.

process (a.k.a., Brownian motion). The SDE (1) has a unique, strong solution under certain regularity conditions (i.e., globally Lipschitz coefficients in both state and time; see [31]). From now on, we denote by $p_{t}(\boldsymbol{x}(t))$ the marginal distribution of $\boldsymbol{x}(t)$ , and let $p_{t|s}(\boldsymbol{x}(t)|\boldsymbol{x}(s))$ be the (perturbation) transition kernel from $\boldsymbol{x}(s)$ to $\boldsymbol{x}(t)$ , $0 \leq s < t \leq T < \infty$ with T as the time horizon. By appropriately selecting f and g, one can force the SDE (1) to converge to a prior distribution (typically a Gaussian). Common examples include the (time-rescaled) Ornstein–Uhlenbeck (OU) process, $^{2}$ which is a special case of the linear version of (1):

$$
d \boldsymbol {x} = f (t) \boldsymbol {x} d t + g (t) d \boldsymbol {W} _ {t}, \quad \boldsymbol {x} (0) \sim p _ {0}. \tag {2}
$$

Reverse sampling. According to $[1]$ and $[54]$ , both the following reverse-time SDE and probability flow ODE share the same marginal distribution as the forward-time SDE (1):

$$
d \boldsymbol {x} = \left[ \boldsymbol {f} (\boldsymbol {x}, t) - g ^ {2} (t) \nabla_ {\boldsymbol {x}} \log p _ {t} (\boldsymbol {x}) \right] d t + g (t) d \bar {\boldsymbol {W}} _ {t}, \tag {3}
$$

$$
d \boldsymbol {x} = \left[ \boldsymbol {f} (\boldsymbol {x}, t) - \frac {1}{2} g ^ {2} (t) \nabla_ {\boldsymbol {x}} \log p _ {t} (\boldsymbol {x}) \right] d t, \tag {4}
$$

where $\bar{W}_{t}$ is a standard Wiener process when time flows backwards from T to 0, and dt is an infinitesimal negative time step. With the initial condition $\boldsymbol{x}(T) \sim p_{T} \approx \pi$ , where $\pi$ is a known prior distribution such as the Gaussian noise, one can (numerically) solve (3) or (4) to transform noises into samples from $p_{0}$ , which is exactly the goal of generative modeling.

Loss objectives. The only remaining task is to estimate the unknown (Stein) score function $\nabla_{\boldsymbol{x}} \log p_{t}(\boldsymbol{x})$ . This is achieved by minimizing the following weighted sum of denoising score

matching ([57]) objectives:

$$
\mathcal {L} (\boldsymbol {\theta}; \lambda (\cdot)) := \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} (0) \sim p _ {0}} \left[ \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t | 0}} \left[ \| \boldsymbol {s} _ {t, \boldsymbol {\theta}} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t | 0} (\boldsymbol {x} (t) | \boldsymbol {x} (0)) \| _ {2} ^ {2} \right] \right] \right], \tag {5}
$$

with $\theta^{*} := \arg\min_{\theta} \mathcal{L}(\theta; \lambda(\cdot))$ , where $\mathcal{U}(0, T)$ denotes the uniform distribution over $[0, T]$ , and $\lambda(t) : [0, T] \mapsto \mathbb{R}_{+}$ is a weighting function, which is typically selected as

$$
\lambda (t) \propto 1 / \sqrt {\mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t | 0}} [ \| \nabla_ {\boldsymbol {x} (t)} \log p _ {t | 0} (\boldsymbol {x} (t) | \boldsymbol {x} (0)) \| _ {2} ^ {2} ]} \tag {6}
$$

according to ([54]). The score function $s_{t,\theta}: R^{d} \mapsto R^{d}$ is time-dependent and can be parameterized as a neural network (encoded with the time information) such as the U-net([37]) architecture commonly applied in the field of image segmentation. Alternatively, one can also define the time-dependent score matching loss

$$
\tilde {\mathcal {L}} (\boldsymbol {\theta}; \lambda (\cdot)) := \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \| \boldsymbol {s} _ {t, \boldsymbol {\theta}} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t)) \| _ {2} ^ {2} \right] \right], \tag {7}
$$

which is equivalent to (5) up to a constant independent of $\theta$ by [57, 49].

In practice, expectations in the objective (5) can be respectively estimated with empirical means over time steps in $[0,T]$ , data samples from $p_{0}$ and $p_{t|0}$ , which is efficient when the drift coefficient $\boldsymbol{f}(\cdot,t)$ is linear. Specifically, if the forward-time SDE takes the form of (2), the transition kernel $p_{t|0}$ has a closed form ([40])

$$
p _ {t | 0} (\boldsymbol {x} (t) | \boldsymbol {x} (0)) = \mathcal {N} (\boldsymbol {x} (t); r (t) \boldsymbol {x} (0), r ^ {2} (t) v ^ {2} (t) \boldsymbol {I} _ {d}), \tag {8}
$$

where $\mathcal{N}(\boldsymbol{x};\boldsymbol{\mu},\boldsymbol{\Sigma})$ denotes the multivariate Gaussian distribution evaluated at x with the expectation $\mu$ and covariance $\Sigma$ , and $r(t):=e^{\int_{0}^{t}f(\zeta)d\zeta}$ , $v(t):=\sqrt{\int_{0}^{t}\frac{g^{2}(\zeta)}{r^{2}(\zeta)}d\zeta}$ .

Training. We aim to investigate the gradient flow training dynamics over the empirical loss

$$
\frac {d}{d \tau} \hat {\boldsymbol {\theta}} _ {n} (\tau) = - \nabla_ {\hat {\boldsymbol {\theta}} _ {n} (\tau)} \hat {\mathcal {L}} _ {n} (\hat {\boldsymbol {\theta}} _ {n} (\tau); \lambda (\cdot)), \quad \hat {\boldsymbol {\theta}} _ {n} (0) := \hat {\boldsymbol {\theta}} _ {n} ^ {0}, \tag {9}
$$

where $\hat{L}_{n}$ is the Monte-Carlo estimation of L defined in (5) on the training dataset, $^{3}$ with an auxiliary gradient flow over the population loss

$$
\frac {d}{d \tau} \boldsymbol {\theta} (\tau) = - \nabla_ {\boldsymbol {\theta} (\tau)} \mathcal {L} (\boldsymbol {\theta} (\tau); \lambda (\cdot)), \quad \boldsymbol {\theta} (0) := \boldsymbol {\theta} ^ {0} = \hat {\boldsymbol {\theta}} _ {n} ^ {0}. \tag {10}
$$

In both cases, the weighting function $\lambda(\cdot)$ is selected as in (6). Denote the score function learned at the training time $\tau$ evaluated at the SDE time t with respect to the empirical loss and population loss as $\boldsymbol{s}_{t,\hat{\boldsymbol{\theta}}_{n}(\tau)}(\boldsymbol{x}(t))$ and $\boldsymbol{s}_{t,\boldsymbol{\theta}(\tau)}(\boldsymbol{x}(t))$ , respectively. The corresponding density functions, denoted

by $p_{t,\hat{\pmb{\theta}}_n(\tau)}(\pmb{x}(t))$ and $p_{t,\pmb{\theta}(\tau)}(\pmb{x}(t))$ , are obtained by solving

$$
\nabla_ {\boldsymbol {x} (t)} \log p _ {t, \hat {\boldsymbol {\theta}} _ {n} (\tau)} (\boldsymbol {x} (t)) = \boldsymbol {s} _ {t, \hat {\boldsymbol {\theta}} _ {n} (\tau)} (\boldsymbol {x} (t)), \quad \nabla_ {\boldsymbol {x} (t)} \log p _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) = \boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)), \tag {11}
$$

and then normalizing, respectively.

Score networks. We parameterize the score function $\pmb{s}_{t,\theta}$ as the following random feature model

$$
\boldsymbol {s} _ {t, \boldsymbol {\theta}} (\boldsymbol {x}) := \frac {1}{m} \boldsymbol {A} \sigma (\boldsymbol {W} \boldsymbol {x} + \boldsymbol {U} \boldsymbol {e} (t)) = \frac {1}{m} \sum_ {i = 1} ^ {m} \boldsymbol {a} _ {i} \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)), \tag {12}
$$

where $\sigma$ is the ReLU activation function, $A = (a_{1},\ldots ,a_{m})\in \mathbb{R}^{d\times m}$ is the trainable parameter, while $\pmb {W} = (\pmb {w}_1,\dots ,\pmb {w}_m)^{\top}\in \mathbb{R}^{m\times d}$ and $U = (\pmb {u}_1,\dots ,\pmb {u}_m)^{\top}\in \mathbb{R}^{m\times d_e}$ are randomly initialized and frozen during training, and $\pmb {e}:\mathbb{R}_{\geq 0}\mapsto \mathbb{R}^{d_e}$ is the embedding function concerning the time information. Assume that $\pmb {a}_i$ , $\pmb{w}_i$ and $\pmb{u}_i$ are i.i.d. sampled from an underlying distribution $\rho$ . Then, as $m\to \infty$ , we get

$$
\begin{array}{l} \boldsymbol {s} _ {t, \boldsymbol {\theta}} (\boldsymbol {x}) \rightarrow \bar {\boldsymbol {s}} _ {t, \bar {\boldsymbol {\theta}}} (\boldsymbol {x}) := \mathbb {E} _ {(\boldsymbol {a}, \boldsymbol {w}, \boldsymbol {u}) \sim \rho} \left[ \boldsymbol {a} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right] \\ = \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ \boldsymbol {a} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right], \tag {13} \\ \end{array}
$$

with $\boldsymbol{a}(\boldsymbol{w},\boldsymbol{u}):=\frac{1}{\rho_{0}(\boldsymbol{w},\boldsymbol{u})}\int_{\mathbb{R}^{d}}\boldsymbol{a}\rho(\boldsymbol{a},\boldsymbol{w},\boldsymbol{u})da$ and $\rho_{0}(\boldsymbol{w},\boldsymbol{u}):=\int_{\mathbb{R}^{d}}\rho(\boldsymbol{a},\boldsymbol{w},\boldsymbol{u})da$ . By the positive homogeneity property of the ReLU activation, we can assume that $\|w\|_{1}+\|u\|_{1}\leq1$ w.l.o.g.

One can view $\bar{\boldsymbol{s}}_{t,\bar{\boldsymbol{\theta}}}(\boldsymbol{x})$ as a continuous version of the random feature model. Correspondingly, the optimal solution is denoted as $\bar{\theta}^{*}$ when replacing the parameterized score function $\boldsymbol{s}_{t,\boldsymbol{\theta}}(\boldsymbol{x})$ in the loss objective (5) or (7) by $\bar{\boldsymbol{s}}_{t,\bar{\boldsymbol{\theta}}}(\boldsymbol{x})$ . Define the kernel

$$
k _ {\rho_ {0}} (\boldsymbol {x}, \boldsymbol {x} ^ {\prime}) := \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} ^ {\prime} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right],
$$

and let $\mathcal{H}_{k_{\rho_0}}$ be the induced reproducing kernel Hilbert space (RKHS; [2]), we have $\bar{\boldsymbol{s}}_{t,\bar{\boldsymbol{\theta}}} \in \mathcal{H}_{k_{\rho_0}}$ if the RKHS norm $\| \bar{\boldsymbol{s}}_{t,\bar{\boldsymbol{\theta}}}\|_{\mathcal{H}_{k_{\rho_0}}}^2 := \mathbb{E}_{(\boldsymbol{w},\boldsymbol{u}) \sim \rho_0}[\|\boldsymbol{a}(\boldsymbol{w},\boldsymbol{u})\|_2^2] = \| \| \boldsymbol{a}\|_2\|_{L^2 (\rho_0)}^2 < \infty$ , and the corresponding discrete version can be defined by the empirical average, i.e., $\| \boldsymbol{s}_{t,\boldsymbol{\theta}}\|_{\mathcal{H}_{k_{\rho_0}}}^2 := \frac{1}{m}\| \boldsymbol{A}\|_F^2 = \frac{1}{m}\sum_{i=1}^{m}\| \boldsymbol{a}(\boldsymbol{w}_i,\boldsymbol{u}_i)\|_2^2$ .

Remark 1. There are more modern and complex mathematical tools such as neural tangent kernels (NTKs) and mean fields that can be selected as the score networks. Employing these modern tools is valuable at least for theoretical completeness and we leave these as the future work. $^{4}$

The goal is to measure and bound the generalization error evolving with the gradient flow

training dynamics (9) between the learned distribution and target distribution, using the common Kullback–Leibler (KL) divergence.

Definition 1 (KL divergence). Given two distributions $p$ and $q$ , the KL divergence from $q$ to $p$ is defined as $D_{\mathrm{KL}}(p||q) = \int_{\mathbb{R}^d} p(\boldsymbol{x}) \log \left(\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\right) d\boldsymbol{x}$ .

Based on the above definitions, the generalization gap along the gradient flow training dynamics (9) is formulated as $D_{\mathrm{KL}}\left(p_{0}\|p_{0,\hat{\boldsymbol{\theta}}_{n}(\tau)}\right)$ , which is only a function of the training time $\tau$ . The goal is to estimate $D_{\mathrm{KL}}\left(p_{0}\|p_{0,\hat{\boldsymbol{\theta}}_{n}(\tau)}\right)^{5}$ .

# 3.2 Main Results

In this section, we state the main results of the generalization capability of the (score-based) diffusion models and how it evolves as the training proceeds. Based on the formulation in Section 3.1, we theoretically derive several upper bounds to estimate $D_{\mathrm{KL}} \left( p_0 \| p_{0,\hat{\theta}_n(\tau)} \right)$ under different settings. The results cover both the positive and negative aspects: in the data-independent setting where the target distribution has finite support, the generalization gap is proved to be small; while in the data-dependent setting where the target distribution possesses shift modes, the generalization is adversely affected by the modes' distance.

# 3.2.1 Data-Independent Generalization Gap

In this section, we provide the characterization of the generalization capability for diffusion models given a target distribution defined on a finite domain. Generally, the KL divergence from the learned distribution $p_{0,\hat{\boldsymbol{\theta}}_{n}(\tau)}$ at the training time $\tau$ to the target distribution $p_{0}$ can be estimated as follows.

Theorem 1. Suppose that the target distribution $p_0$ is continuously differentiable and has a compact support set, i.e., $||\boldsymbol{x}||_{\infty}$ is uniformly bounded, and there exists a reproducing kernel Hilbert space (RKHS) $\mathcal{H}$ ( $:= \mathcal{H}_{k_{\rho_0}}$ ) such that $\bar{s}_{0,\bar{\theta}^*} \in \mathcal{H}$ . Assume that the initial loss, trainable parameters, the embedding function $\boldsymbol{e}(t)$ and weighting function $\lambda(t)$ are all bounded. Then for any $\delta > 0$ , $\delta \ll 1$ , with the probability of at least $1 - \delta$ , we have

$$
D _ {\mathrm{KL}} \left(p _ {0} \| p _ {0, \hat {\pmb {\theta}} _ {n} (\tau)}\right) \lesssim \left[ \frac {\tau^ {4}}{m n} + \frac {\tau^ {3}}{m ^ {2}} + \frac {1}{\tau} \right] + \left[ \frac {1}{m} + \bar {\tilde {\mathcal {L}}} \left(\bar {\pmb {\theta}} ^ {*}\right) + \tilde {\mathcal {L}} \left(\pmb {\theta} ^ {*}\right) \right] + D _ {\mathrm{KL}} \left(p _ {T} \| \pi\right), \quad \tau \geq 1,
$$

where $\lesssim$ hides the term $d\log(d+1)$ , the polynomials of $\log(1/\delta^{2})$ , finite RKHS norms and universal positive constants only depending on T.

Remark 2. Since $p_{0}$ is compactly supported, the target score function $\boldsymbol{s}_{0}(\boldsymbol{x}) = \nabla_{\boldsymbol{x}} \log p_{0}(\boldsymbol{x})$ is also defined on a compact domain. According to [12, 13], $s_{0}$ is contained in the Barron function space with a finite Barron norm, and hence in a certain RKHS with a finite RKHS norm. Therefore, it is reasonable to require that the global minimizer $s_{0,\theta^{*}}$ or $\bar{s}_{0,\bar{\theta}^{*}}$ is also contained in some RKHS.

Proof sketch. Theorem 1 is proved via the following procedure.

1. According to Theorem 1 in [49], the KL divergence on the left-hand side can be upper bounded by the population loss of the trained model up to a small error. That is,

$$
D _ {\mathrm{KL}} \left(p _ {0} \| p _ {0, \hat {\boldsymbol {\theta}} _ {n} (\tau)}\right) \leq \tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau); g ^ {2} (\cdot)) + D _ {\mathrm{KL}} \left(p _ {T} \| \pi\right). \tag {14}
$$

2. We use the model trained with respect to the population loss (10) to perform the decomposition:

$$
\begin{array}{l} \tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau)) = \left[ \tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau)) - \tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau)) \right] + \tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau)) \\ \lesssim \left[ \tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau)) - \tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau)) \right] + \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) + \text { Monte   Carlo } \triangleq I _ {1} + I _ {2} + I _ {3}, \tag {15} \\ \end{array}
$$

where we omit the weighting function $g^{2}(\cdot)$ for simplicity. Here, $\tilde{L}$ is the loss objective obtained by replacing $s_{t,\theta}$ in $\tilde{L}$ (defined in (7)) by $\bar{s}_{t,\bar{\theta}}$ , and $I_{3}$ summarizes the resulting Monte Carlo error.

3. $I_{3}$ can be estimated via a similar argument as in [24] (Lemma 48).   
4. $I_{2}$ can be upper bounded via a standard analysis on the gradient flow dynamics over convex objectives.   
5. $I_{1}$ can be reduced as the norm product of $\pmb{s}_{0,\hat{\pmb{\theta}}_n(\tau)}$ , $\pmb{s}_{0,\pmb{\theta}(\tau)}$ and their gap, then

(a) the former can be bounded with a square-root rate growth via a general norm estimate of parameters trained under the gradient flow dynamics;

(b) the latter can be estimated by the Rademacher complexity (see e.g., Chapter 26 in [43]).

Combining all above gives the desired result. The detailed proof is found in Appendix A.1.

Discussion on error bounds. The three error terms are further analyzed as follows.

\- The first term is the main error, which implies an early-stopping generalization gap. In fact, if one selects an early-stopping time $\tau_{\mathrm{es}}$ as $\tau_{\mathrm{es}} = \Theta\left(n^{\frac{2}{5}}\right)$ , and let $m \sim n$ , we have

$$
D _ {\mathrm{KL}} \left(p _ {0} \| p _ {0, \hat {\boldsymbol {\theta}} _ {n} (\tau_ {\mathrm{es}})}\right) \lesssim (1 / n) ^ {\frac {2}{5}} + (1 / m) ^ {\frac {4}{5}}. \tag {16}
$$

- The second term is $m$ -dependent and corresponds to the approximation error, which is $o(1)$ when $m \gg 1$ . In fact, the random feature model is a universal approximator to Lipschitz continuous functions on a compact domain (Theorem 6 in [17]). See more details in Appendix A.1 (the last paragraph).   
- The third term is exponentially small in $T$ since $\pi$ (e.g. the Gaussian density) is log-Sobolev, according to a classical result in e.g. [55] (Theorem 3.20, Theorem 3.24 and Remark 3.26).

Remark 3. In practice, it is common to use the test error to evaluate the generalization performance. For diffusion models, a straightforward approach is to compute the negative log-likelihood (averaged in bits/dim; equivalent to the KL divergence) on the test dataset during training with the instantaneous change-of-variable formula ([8]) and probability flow ODE (defined in (4)), where the true score function $\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$ is replaced by $\boldsymbol{s}_{t,\boldsymbol{\theta}(\tau)}(\boldsymbol{x})$ .

Remark 4. Previous literature has established similar bounds for bias potential models ([66]) and GANs ([67]). Theorem 1 extends the corresponding results to the setting of diffusion models. Furthermore, this upper bound is finer in the sense that it incorporates the information regarding the model capacity (the hidden dimension m in this case), which shows that more parameters benefit the learning and generalization, as expected.

# 3.2.2 Data-Dependent Generalization Gap

In Section 3.2.1, we derive estimates on the generalization error for diffusion models along the training dynamics, where the target distribution is assumed to be finitely supported. In reality, this is often not the case, where target distributions usually possess distant multi-modes, from simple Gaussian mixtures to complicated Boltzmann distributions of physical systems ([29, 30]). Under these settings, the above analysis in Section 3.2.1 can not directly apply since the data domain is unbounded. It remains a problem to quantitatively characterize the generalization behavior of diffusion models given these target distributions with distant multi-modes or modes shift.

To provide a fine-grained demonstration of the generalization capability of diffusion models when applied to learn distributions with distant multi-modes, as an illustrating example, the Gaussian mixture with two modes is selected as the target distribution.

Theorem 2. Suppose the target distribution $p_{0}$ is a one-dimensional 2-mode Gaussian mixture: $p_{0}(x)=q_{1}\mathcal{N}(x;-\mu,1)+q_{2}\mathcal{N}(x;\mu,1)$ , where $\mu>\sqrt{\log(1/\delta^{2})}$ , $q_{1}$ , $q_{2}>0$ with $q_{1}+q_{2}=1$ are all constants. Under the conditions of Theorem 1 (except the uniform boundness of inputs), we have

$$
D _ {\mathrm{KL}} \left(p _ {0} \| p _ {0, \hat {\pmb {\theta}} _ {n} (\tau)}\right) \lesssim \mathrm{Poly} (\mu) \left[ \frac {\tau^ {4}}{m n} + \frac {\tau^ {3}}{m ^ {2}} \right] + \frac {1}{\tau} + \left[ \frac {\mu^ {2}}{m} + \tilde {\mathcal {L}} \left(\bar {\pmb {\theta}} ^ {*}\right) + \tilde {\mathcal {L}} \left(\pmb {\theta} ^ {*}\right) \right] + D _ {\mathrm{KL}} \left(p _ {T} \| \pi\right),
$$

where $\tau\geq1,\lesssim$ hides the polynomials of $\log(1/\delta^{2})$ , finite RKHS norms and universal positive constants only depending on T.

![](images/fbc464bebca6431b95ba8c5350e73b61e1a2c8a4fe755bf0366c34c17006493f.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Blue Line |
| ---- | -------- | --------- |
| -40  | 0        | 0         |
| -30  | 1        | 0         |
| -20  | 0        | 1         |
| -10  | 0        | 1         |
| 0    | 0        | 1         |
| 10   | 0        | 1         |
| 20   | 0        | 0         |
| 30   | 1        | 0         |
| 40   | 0        | 0         |
</details>

Figure 2: An illustration of modes shift.

Proof sketch. Theorem 2 is proved following a similar procedure with Theorem 1, except the input data x does not have a uniform bound here. This problem mainly affects the last step (5(b)) in the proof sketch of Theorem 1, and can be handled by using the fact

$$
| x | \in [ \mu - \sqrt {\log (1 / \delta^ {2})}, \mu + \sqrt {\log (1 / \delta^ {2})} ] = \Theta (\mu) \tag {17}
$$

given the target Gaussian mixture distribution. The detailed proof is found in Appendix A.2.

Remark 5. Theorem 2 indicates that, even for a simple target distribution (e.g. a one-dimensional 2-mode Gaussian mixture), the generalization error of diffusion models can be polynomially large regarding the modes' distance. Although Theorem 2 provides only an upper bound, the modes shift effect holds due to the model-target inconsistency (the last paragraph in Appendix A.2) and the following consistent experiments (Section 4.1.2).

# 4 Numerical Verifications

In this section, we numerically verify the previous theoretical results and insights (early-stopping generalization and modes shift effect) on both synthetic datasets and real-world datasets.

# 4.1 Simulations on Synthetic Datasets

# 4.1.1 Early-Stopping Generalization

First, we illustrate the early-stopping generalization gap established in Theorem 1. We select the one-hidden-layer neural network with Swish activations as the score network, which is trained using the SGD optimizer with a fixed learning rate 0.5. The target distribution is set to be a one-dimensional 2-mode Gaussian mixture with the modes' distance equalling 6, and the number of data samples is 1000. We measure the KL divergence from the trained model to the target distribution $D_{\mathrm{KL}} \left(p_0 \| p_{0,\hat{\theta}_n(\tau)}\right)$ along with the training epochs.

From Figure 3, one can observe that the KL divergence achieves its minimum at approximately the 800-th training epoch, and it starts to increase after this turning point. The experimental

results are consistent with Theorem 1 (over multiple runs), which states that there exist early-stopping times when diffusion models can generalize well, indicating the effectiveness of our upper bound. Further, the KL divergence begins to oscillate after the minimum point, which may suggest a phase transition in the training dynamics, and the transition point is around the (optimal) early-stopping time.

![](images/b6a898660473fb12ddbc7d1502c4814102d8b6e7fc31fed59af62da8e79d188a.jpg)

<details>
<summary>line</summary>

| epoch | Repetition 1 | Repetition 2 | Repetition 3 |
| ----- | ------------ | ------------ | ------------ |
| 0     | 1.35         | 1.05         | 1.05         |
| 250   | 0.25         | 0.20         | 0.20         |
| 500   | 0.15         | 0.15         | 0.15         |
| 750   | 0.25         | 0.15         | 0.15         |
| 1000  | 0.50         | 0.20         | 0.20         |
| 1250  | 0.15         | 0.25         | 0.25         |
| 1500  | 0.30         | 0.35         | 0.35         |
| 1750  | 0.15         | 0.25         | 0.25         |
| 2000  | 0.15         | 0.25         | 0.45         |
</details>

Figure 3: The KL divergence dynamics.

# 4.1.2 Modes Shift Effect

Next, we numerically test the relationship between the modes' distance and generalization (density estimation) performance. All the configurations remain the same as Section 4.1.1, except that the target Gaussian mixtures have different modes' distances.

In Figure 4, the modeled distributions exhibit the following two-stage dynamics: (i) first gradually fitting the two modes (epoch = 100 → 1000); (ii) then diverging (epoch = 1000 → 1900). This aligns with the KL divergence dynamics (Figure 3) and again verifies the corresponding theoretical results (Theorem 1). However, Figure 5 shows that when the modes are distant from each other, there is difficulty in the learning process. In Figure 5, the optimal generalization is achieved at epoch = 100, but is still far from well generalizing. As the training proceeds, the learned model is almost always a single-mode distribution (epoch = 1000, 1900). This phenomenon is aligned with the results established in Theorem 2, which states that when there are distant modes in the target distribution, the generalization performance is relatively poor.

![](images/f8622251856aa903d71bf42b7d9b15837fac186e84af53e4cac7214ee4c3b2b7.jpg)

<details>
<summary>line</summary>

| x     | Target | Trained |
|-------|--------|---------|
| -10.0 | 0.0000 | 0.0000  |
| -7.5  | 0.0000 | 0.0000  |
| -5.0  | 0.1500 | 0.1500  |
| -2.5  | 0.1750 | 0.1750  |
| 0.0   | 0.1250 | 0.1250  |
| 2.5   | 0.1500 | 0.1250  |
| 5.0   | 0.1250 | 0.1250  |
| 7.5   | 0.0000 | 0.0000  |
| 10.0  | 0.0000 | 0.0000  |
</details>

![](images/c010ead8bd59af90f4b4e6d4ba8cc15014e628bd2eb260b01e1c7c681814aebf.jpg)

<details>
<summary>line</summary>

| x     | Target | Trained |
|-------|--------|---------|
| -10.0 | 0.0000 | 0.0000  |
| -7.5  | 0.0000 | 0.0000  |
| -5.0  | 0.1500 | 0.1800  |
| -2.5  | 0.1600 | 0.1900  |
| 0.0   | 0.0250 | 0.0500  |
| 2.5   | 0.1500 | 0.1600  |
| 5.0   | 0.1600 | 0.1700  |
| 7.5   | 0.0000 | 0.0000  |
| 10.0  | 0.0000 | 0.0000  |
</details>

![](images/bb1619092b36be812c17b8c8a442f0875deaa5da8ef16936e846f3016f380f59.jpg)

<details>
<summary>line</summary>

| x     | Target | Trained |
|-------|--------|---------|
| -10.0 | 0.0000 | 0.0000  |
| -7.5  | 0.0000 | 0.0000  |
| -5.0  | 0.1500 | 0.1750  |
| -2.5  | 0.1750 | 0.2000  |
| 0.0   | 0.1500 | 0.1750  |
| 2.5   | 0.1250 | 0.1250  |
| 5.0   | 0.1500 | 0.1000  |
| 7.5   | 0.1250 | 0.0750  |
| 10.0  | 0.0750 | 0.0500  |
</details>

Figure 4: The training dynamics when the distance between two modes is 6 ( $\mu = 3$ ).

![](images/f9677ee8f9cc1c11e5da76af6e8b24580bb51883f0cd3f6fd7fe43c1713f0d72.jpg)

<details>
<summary>line</summary>

| x    | Target | Trained |
| ---- | ------ | ------- |
| -30  | 0.00   | 0.00    |
| -25  | 0.05   | 0.28    |
| -20  | 0.06   | 0.00    |
| -15  | 0.04   | 0.00    |
| -10  | 0.02   | 0.00    |
| -5   | 0.01   | 0.00    |
| 0    | 0.00   | 0.00    |
| 5    | 0.01   | 0.00    |
| 10   | 0.03   | 0.01    |
| 15   | 0.05   | 0.02    |
| 20   | 0.04   | 0.01    |
| 25   | 0.02   | 0.00    |
| 30   | 0.01   | 0.00    |
</details>

![](images/638ec59cc8a870b5029ddff9a1c11678d651a69904816ccbd971c4d1e498a9e4.jpg)

<details>
<summary>line</summary>

| x    | Target | Trained |
| ---- | ------ | ------- |
| -30  | 0.0000 | 0.0000  |
| -20  | 0.0500 | 0.1250  |
| -10  | 0.0500 | 0.0000  |
| 0    | 0.0000 | 0.0000  |
| 10   | 0.0500 | 0.2000  |
| 20   | 0.0500 | 0.0000  |
| 30   | 0.0000 | 0.0000  |
</details>

![](images/59e218089029946ae2542938073cb1a4b905bb354448e89b44e8ee303e4b33ba.jpg)

<details>
<summary>line</summary>

| x    | Target | Trained |
| ---- | ------ | ------- |
| -30  | 0.00   | 0.00    |
| -20  | 0.05   | 0.30    |
| -10  | 0.05   | 0.05    |
| 0    | 0.00   | 0.00    |
| 10   | 0.05   | 0.00    |
| 20   | 0.05   | 0.00    |
| 30   | 0.00   | 0.00    |
</details>

Figure 5: The training dynamics when the distance between two modes is 30 ( $\mu = 15$ ).

# 4.2 Simulations on Real-World Datasets

In this subsection, we verify our results on the MNIST dataset using the standard U-net architecture as the score network, which suggests that the adverse effect of modes shift on the generalization performance of diffusion models also appears in general.

The setup is as follows. First, we perform a K-means clustering on D (D denote the MNIST dataset) to get $D = \bigcup_{k=1}^{K} D_k$ , and $\bar{x}_k$ as the center of $D_k$ , $k = 1, 2, \cdots, K$ . Let $(i^*, j^*) := \arg\max_{i \neq j} \| \bar{x}_i - \bar{x}_j \|_2$ , and $D_{farthest} := D_{i^*} \bigcup D_{j^*}$ . $D_{nearest}$ is similarly constructed by arg min indices. Then, by randomly selecting the same number of data samples and using the same configuration, we train two separate diffusion models on $D_{farthest}$ and $D_{nearest}$ , respectively, and then perform inference (sampling). The training loss curves and sampling results are shown in Figure 6 and Figure 7, respectively. One can observe a significant performance gap: the diffusion model trained on $D_{farthest}$ appears a higher learning loss and worse sampling quality compared to those of $D_{nearest}$ .

![](images/741fd5fa9f864a9262177035ed433db84c16463191976c484af47e875b582b52.jpg)

<details>
<summary>line</summary>

| Epoch | Nearest | Farthest |
|-------|---------|----------|
| 0     | 180.0   | 180.0    |
| 100   | 100.0   | 120.0    |
| 200   | 60.0    | 80.0     |
| 300   | 45.0    | 65.0     |
| 400   | 40.0    | 60.0     |
| 500   | 35.0    | 55.0     |
| 600   | 32.0    | 50.0     |
| 700   | 30.0    | 48.0     |
| 800   | 28.0    | 45.0     |
| 900   | 26.0    | 42.0     |
| 1000  | 25.0    | 40.0     |
</details>

Figure 6: The training loss dynamics.

![](images/a68f2a043cf37ada8cb1e8d3c21028315e9269e22011a5357cce08d645be0757.jpg)

<details>
<summary>text_image</summary>

Ancient Chinese oracle bone script characters arranged in a grid-like pattern
</details>

![](images/cf3737c60cd563b184b2651f38d68dc1bc6abda652bd83ae925faf265d21cfc8.jpg)

<details>
<summary>text_image</summary>

5 0 3 8 8 2 3 8
8 3 8 2 3 5 3 2
5 2 8 8 3 9 3 8
8 3 3 2 3 2 2 8
8 3 3 8 6 2 5 2
2 8 8 8 3 8 3 8
3 8 5 3 2 2 5 2
3 3 3 5 2 8 3 2
</details>

Figure 7: Sampling of the farthest (left) and nearest (right) clusters.

# 4.3 Discussion

We compare the results developed in this work with former corresponding literature as follows:

- The previous work [21] also studied the adverse effect of modes shift, which particularly reported a contrastive simulation indicating the degraded performance when modeling Gaussian mixtures with the increasing distance between modes (see Figure 2 in [21] and compare with Figure 4 and Figure 5). However, the results therein are established and tested under the “pure” score matching setting, without the denoising or time-dependent dynamics. As a comparison, Theorem 2 establishes a theoretical estimate on the generalization gap for diffusion models that requires denoising, and this upper bound directly depends on the distance between modes of target distributions, instead of a circuitous characterization in [21] to attribute the difficulty of learning modes shift to increased isoperimetry of corresponding target distributions.   
- Similar adverse effect of modes shift has also been theoretically analyzed and numerically verified on recurrent neural networks (RNNs), see e.g., [23, 24]. There, the modes shift is understood as a type of long-term memory. This is the phenomenon of the “curse of memory”: When there is long-term memory in the target, it requires a large number of parameters for the approximation. Meanwhile, the training process will suffer from severe slowdowns. Both of these effects can be exponentially more pronounced with increasing memory or modes shift.   
- The very recent work [42] also considered the problem of learning Gaussian mixtures using the denoising diffusion probabilistic model (DDPM) objective, but under a teacher-student setting. That is, given the Gaussian mixtures target, [42] parametrized the score network model in the same form of the score function target, with the goal to identify true parameters. Consequently, there are all positive convergence results developed in [42], despite that the time and sample complexity increases with the distance between modes. As a comparison,

Theorem 2 adopts a pre-selected score network model without incorporating any information from the ground truth, and hence establish negative results concerning the modes shift. In fact, if the goal is to identify only true positions of the target Gaussian mixture (this is exactly the setting of [42]), the teacher-student setup seems not necessary (see Figure 5, where the true position is also learned efficiently using the one-hidden-layer Swish neural network as the score network, but the modes are always weighed incorrectly). In addition, [42] did not take the whole denoising dynamics into account. That is, the gradient descent (GD) analysis therein was performed on the denoising score matching objective successively at only two time stages: a larger $t_1$ (“high noise”) and a smaller $t_2$ (“low noise”), which is often not the case in practice.

# 5 Conclusion

In this paper, we provide a theoretical analysis of the fundamental generalization aspect of training diffusion models under both the data-independent and data-dependent settings, and early-stopping estimates of the generalization gap along the training dynamics are derived. Quantitatively, the data-independent results indicate a polynomially small generalization error that escapes from the curse of dimensionality, while the data-dependent results suggest the adverse effect of modes shift in target distributions. Numerical simulations have illustrated and verified these theoretical analyses. This work forms a basic starting point for understanding the intricacies of modern deep generative modeling and corresponding central concerns such as memorization, privacy, and copyright arising from practical applications and business products. More broadly, the approach here may have the potential to be extended to other variants in the diffusion models family, including a general SDE-based design space ([20]), consistency models ([48]), rectified flows ([27, 26]), Schrödinger bridges ([59, 56, 10, 44, 47, 25]), etc. These are certainly worthy of future exploration.

# Acknowledgements

We would like to thank Dr. Hongkang Yang for helpful discussions, and all the reviewers' valuable feedback and insightful suggestions to improve this work.

# References

[1] Brian D. O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313-326, 1982.

[2] Nachman Aronszajn. Theory of reproducing kernels. Transactions of the American Mathematical Society, 68(3):337-404, 1950.   
[3] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981–17993, 2021.   
[4] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022.   
[5] Nicholas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In USENIX Security Symposium, pages 5253–5270, 2023.   
[6] Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. Quantifying memorization across neural language models. In International Conference on Learning Representations, 2023.   
[7] Hongrui Chen, Holden Lee, and Jianfeng Lu. Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. International Conference on Machine Learning, 202:4735–4763, 2023.   
[8] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K. Duvenaud. Neural ordinary differential equations. Advances in Neural Information Processing Systems, 31:6571–6583, 2018.   
[9] Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru Zhang. Sampling is as easy as learning the score: Theory for diffusion models with minimal data assumptions. In International Conference on Learning Representations, 2023.   
[10] Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schrödinger bridge with applications to score-based generative modeling. Advances in Neural Information Processing Systems, 34:17695–17709, 2021.   
[11] Tim Dockhorn, Tianshi Cao, Arash Vahdat, and Karsten Kreis. Differentially private diffusion models. Transactions on Machine Learning Research, 2023.   
[12] Weinan E, Chao Ma, and Lei Wu. A priori estimates of the population risk for two-layer neural networks. Communications in Mathematical Sciences, 17(5):1407-1425, 2019.   
[13] Weinan E, Chao Ma, and Lei Wu. The Barron space and the flow-induced function spaces for neural network models. Constructive Approximation, 55(1):369-406, 2022.

[14] Sahra Ghalebikesabi, Leonard Berrada, Sven Gowal, Ira Ktena, Robert Stanforth, Jamie Hayes, Soham De, Samuel L Smith, Olivia Wiles, and Borja Balle. Differentially private diffusion models generate useful synthetic images. In International Workshop on Trustworthy Federated Learning in Conjunction with IJCAI, 2023.   
[15] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and Lingpeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. In International Conference on Learning Representations, 2023.   
[16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840-6851, 2020.   
[17] Daniel Hsu, Clayton H. Sanford, Rocco Servedio, and Emmanouil Vasileios Vlatakis-Gkaragkounis. On the approximation power of two-layer networks of random ReLUs. Conference on Learning Theory, 134:2423–2461, 2021.   
[18] Hailong Hu and Jun Pang. Membership inference of diffusion models. arXiv preprint arXiv:2301.09956, 2023.   
[19] Matthew Jagielski, Om Thakkar, Florian Tramer, Daphne Ippolito, Katherine Lee, Nicholas Carlini, Eric Wallace, Shuang Song, Abhradeep Guha Thakurta, Nicolas Papernot, and Chiyuan Zhang. Measuring forgetting of memorized training examples. In International Conference on Learning Representations, 2023.   
[20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.   
[21] Frederic Koehler, Alexander Heckett, and Andrej Risteski. Statistical efficiency of score matching: The view from isoperimetry. In International Conference on Learning Representations, 2023.   
[22] Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun Feng, Zhihai Xu, Qi Li, and Yueting Chen. Srdiff: Single image super-resolution with diffusion probabilistic models. Neurocomputing, 479:47–59, 2022.   
[23] Zhong Li, Jiequn Han, Weinan E, and Qianxiao Li. On the curse of memory in recurrent neural networks: Approximation and optimization analysis. In International Conference on Learning Representations, 2021.   
[24] Zhong Li, Jiequn Han, Weinan E, and Qianxiao Li. Approximation and optimization theory for linear continuous-time recurrent neural networks. Journal of Machine Learning Research, 23(42):1–85, 2022.

[25] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos Theodorou, Weili Nie, and Anima Anandkumar. I²SB: Image-to-image Schrödinger bridge. International Conference on Machine Learning, 202:22042–22062, 2023.   
[26] Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577, 2022.   
[27] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.   
[28] Tomoya Matsumoto, Takayuki Miura, and Naoto Yanai. Membership inference attacks against diffusion models. In IEEE Security and Privacy Workshops, pages 77–83, 2023.   
[29] Laurence Illing Midgley, Vincent Stimper, Gregor N. C. Simm, Bernhard Schölkopf, and José Miguel Hernández-Lobato. Flow annealed importance sampling bootstrap. In International Conference on Learning Representations, 2023.   
[30] Frank Noé, Simon Olsson, Jonas Köhler, and Hao Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457):eaaw1147, 2019.   
[31] Bernt ∅ksendal. Stochastic Differential Equations. Springer, 2003.   
[32] Jakiw Pidstrigach. Score-based generative models detect manifolds. Advances in Neural Information Processing Systems, 35:35852-35865, 2022.   
[33] Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. Grad-tts: A diffusion probabilistic model for text-to-speech. International Conference on Machine Learning, 139:8599–8608, 2021.   
[34] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
[35] Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. International Conference on Machine Learning, 139:8857–8868, 2021.   
[36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.   
[37] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015.

[38] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo-Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.   
[39] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4713–4726, 2023.   
[40] Simo Särkkä and Arno Solin. Applied Stochastic Differential Equations, volume 10. Cambridge University Press, 2019.   
[41] Vikash Sehwag, Caner Hazirbas, Albert Gordo, Firat Ozgenel, and Cristian Canton. Generating high fidelity data from low-density regions using diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11492-11501, 2022.   
[42] Kulin Shah, Sitan Chen, and Adam Klivans. Learning mixtures of gaussians using the DDPM objective. Advances in Neural Information Processing Systems, 36, 2023.   
[43] Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, 2014.   
[44] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion Schrödinger bridge matching. Advances in Neural Information Processing Systems, 36, 2023.   
[45] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. International Conference on Machine Learning, 37:2256–2265, 2015.   
[46] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6048–6058, 2023.   
[47] Vignesh Ram Somnath, Matteo Pariset, Ya-Ping Hsieh, Maria Rodriguez Martinez, Andreas Krause, and Charlotte Bunne. Aligned diffusion Schrödinger bridges. Conference on Uncertainty in Artificial Intelligence, 216:1985–1995, 2023.   
[48] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. International Conference on Machine Learning, 202:32211-32252, 2023.

[49] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. Advances in Neural Information Processing Systems, 34:1415-1428, 2021.   
[50] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in Neural Information Processing Systems, 32, 2019.   
[51] Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. Advances in Neural Information Processing Systems, 33:12438–12448, 2020.   
[52] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. Uncertainty in Artificial Intelligence Conference, 115:574–584, 2020.   
[53] Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical imaging with score-based generative models. In International Conference on Learning Representations, 2022.   
[54] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.   
[55] Ramon Van Handel. Probability in high dimension. Lecture Notes (Princeton University), 2014.   
[56] Francisco Vargas, Pierre Thodoroff, Austen Lamacraft, and Neil Lawrence. Solving Schrödinger bridges via maximum likelihood. Entropy, 23(9):1134, 2021.   
[57] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661-1674, 2011.   
[58] Nikhil Vyas, Sham M. Kakade, and Boaz Barak. On provable copyright protection for generative models. International Conference on Machine Learning, 202:35277–35299, 2023.   
[59] Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via Schrödinger bridge. International Conference on Machine Learning, 139:10794–10804, 2021.   
[60] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.   
[61] Lei Wu and Weijie J. Su. The implicit regularization of dynamical stability in stochastic gradient descent. International Conference on Machine Learning, 202:37656-37684, 2023.

[62] Yixin Wu, Ning Yu, Zheng Li, Michael Backes, and Yang Zhang. Membership inference attacks against text-to-image generation models. arXiv preprint arXiv:2210.00968, 2022.   
[63] Tian Xie, Xiang Fu, Octavian-Eugen Ganea, Regina Barzilay, and Tommi S. Jaakkola. Crystal diffusion variational autoencoder for periodic material generation. In International Conference on Learning Representations, 2022.   
[64] Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, and Jian Tang. Geodiff: A geometric diffusion model for molecular conformation generation. In International Conference on Learning Representations, 2022.   
[65] Hongkang Yang. A mathematical framework for learning probability distributions. Journal of Machine Learning, 1(4):373-431, 2022.   
[66] Hongkang Yang and Weinan E. Generalization and memorization: The bias potential model. Mathematical and Scientific Machine Learning, 145:1013-1043, 2022.   
[67] Hongkang Yang and Weinan E. Generalization error of GAN from the discriminator's perspective. Research in the Mathematical Sciences, 9(1):1–31, 2022.   
[68] Ruihan Yang, Prakhar Srivastava, and Stephan Mandt. Diffusion probabilistic modeling for video generation. Entropy, 25(10):1469, 2023.   
[69] Jongmin Yoon, Sung Ju Hwang, and Juho Lee. Adversarial purification with score-based generative models. International Conference on Machine Learning, 139:12062–12072, 2021.   
[70] Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. Counterfactual memorization in neural language models. Advances in Neural Information Processing Systems, 36, 2023.

# A Technical Results and Proofs

# A.1 Data-Independent Generalization Gap

To derive the theorem for the generalization error of this score-based generative model, we first give the following lemmas.

Lemma 1 (Forward perturbation estimates). Consider the forward diffusion process with the linear drift coefficient (2). For any $\delta > 0$ , $\delta \ll 1$ , with the probability of at least $1 - \delta$ , we have

$$
\left\| \boldsymbol {x} (t) \right\| _ {\infty} \lesssim C _ {T} \left(\left\| \boldsymbol {x} (0) \right\| _ {\infty} + \sqrt {\log \left(1 / \delta^ {2}\right)}\right), \tag {18}
$$

where $C_{T} := \max_{t \in [0, T]} \{r(t), r(t)v(t)\}$ .

Proof. When the drift coefficient $\boldsymbol{f}(\cdot,t):\mathbb{R}^{d}\mapsto\mathbb{R}^{d}$ is linear to x, i.e., $\boldsymbol{f}(\boldsymbol{x},t)=f(t)\boldsymbol{x}$ , the transition kernel $p_{t|0}$ has a closed form (8)

$$
p _ {t \mid 0} (\boldsymbol {x} (t) \mid \boldsymbol {x} (0)) = \mathcal {N} (\boldsymbol {x} (t); r (t) \boldsymbol {x} (0), r ^ {2} (t) v ^ {2} (t) \boldsymbol {I} _ {d}) \tag {19}
$$

with $r(t):= e^{\int_0^t f(\zeta)d\zeta},v(t):=\sqrt{\int_0^t\frac{g^2(\zeta)}{r^2(\zeta)}d\zeta}$ . Hence, we have

$$
\boldsymbol {x} (t) = r (t) \boldsymbol {x} (0) + r (t) v (t) \boldsymbol {z}, \quad \boldsymbol {z} \sim \mathcal {N} (\boldsymbol {0}, \boldsymbol {I} _ {d}). \tag {20}
$$

For any $\epsilon \sim \mathcal{N}(0,1)$ , $c > 1$ , we have

$$
\begin{array}{l} \mathbb {P} \{\epsilon : | \epsilon | > c \} = 2 \int_ {c} ^ {+ \infty} \frac {1}{\sqrt {2 \pi}} e ^ {- x ^ {2} / 2} d x \\ \leq \frac {1}{\sqrt {2 \pi}} \int_ {c} ^ {+ \infty} 2 x e ^ {- x ^ {2} / 2} d x = \frac {1}{\sqrt {2 \pi}} \int_ {c ^ {2}} ^ {+ \infty} e ^ {- x / 2} d x = \sqrt {\frac {2}{\pi}} e ^ {- c ^ {2} / 2}. \\ \end{array}
$$

Let $\delta = \Theta (e^{-c^2 /2})$ , we get

$$
\mathbb {P} \{\epsilon : | \epsilon | \leq \sqrt {\log (1 / \delta^ {2})} \} \geq 1 - \delta . \tag {21}
$$

Hence, for any $\delta \in (0,1)$ with $\delta \ll 1$ , with the probability of at least $1 - \delta$ , we have

$$
\left\| \boldsymbol {x} (t) \right\| _ {\infty} \lesssim C _ {T} \left(\left\| \boldsymbol {x} (0) \right\| _ {\infty} + \sqrt {\log \left(1 / \delta^ {2}\right)}\right) \tag {22}
$$

with $C_T := \max_{t \in [0, T]} \{r(t), r(t)v(t)\}$ . The proof is completed.

Lemma 2 (Theorem 1 in [49]). We have

$$
D _ {\mathrm{KL}} \left(p _ {0} \| p _ {0, \hat {\pmb {\theta}} _ {n} (\tau)}\right) \leq \tilde {\mathcal {L}} (\hat {\pmb {\theta}} _ {n} (\tau); g ^ {2} (\cdot)) + D _ {\mathrm{KL}} \left(p _ {T} \| \pi\right).
$$

Lemma 3. Both of the loss objectives $\tilde{\mathcal{L}}(\boldsymbol{\theta};\lambda(\cdot))$ and $\tilde{\bar{\mathcal{L}}}(\bar{\boldsymbol{\theta}};\lambda(\cdot))$ are quadratic (and hence convex).

Proof. The convexity arises from the following points: (i) The score matching loss objectives $\tilde{\mathcal{L}}(\boldsymbol{\theta};\lambda(\cdot))$ (defined in (7)) and $\bar{\tilde{\mathcal{L}}}(\bar{\boldsymbol{\theta}};\lambda(\cdot))$ are $L^{2}$ -metrics between the score network model and target score function; (ii) The score networks are defined as random feature models (see (12) and (13)) that are linear to trainable parameters. Therefore, using some trace techniques and basic variational calculations, it is not hard to derive the fact that the loss objectives are quadratic and hence convex with respect to trainable parameters.

(1) For $\tilde{\mathcal{L}} (\pmb {\theta};\lambda (\cdot))$ , recall that $\pmb{s}_{t,\pmb{\theta}}(\pmb {x}(t)) = \frac{1}{m}\pmb{A}\sigma (\pmb {W}\pmb {x}(t) + \pmb {U}\pmb {e}(t))$ , and let $\pmb{s}_t(\pmb {x}(t))\coloneqq \nabla_{\pmb{x}(t)}\log p_t(\pmb {x}(t))$ , $\pmb{h}_1(\pmb {x},t)\coloneqq (\sqrt{\lambda(t)} /\sqrt{m})\sigma (\pmb {W}\pmb {x} + \pmb {U}\pmb {e}(t))$ , $\pmb{h}_2(\pmb {x},t)\coloneqq \sqrt{\lambda(t)}\pmb{s}_t(\pmb {x})$ , we have

$$
\begin{array}{l} \tilde {\mathcal {L}} (\boldsymbol {\theta}; \lambda (\cdot)) = \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \big [ \boldsymbol {h} _ {1} ^ {\top} (\boldsymbol {x} (t), t) (\boldsymbol {A} / \sqrt {m}) ^ {\top} (\boldsymbol {A} / \sqrt {m}) \boldsymbol {h} _ {1} (\boldsymbol {x} (t), t) \\ \left. \left. - 2 \boldsymbol {h} _ {2} ^ {\top} (\boldsymbol {x} (t), t) (\boldsymbol {A} / \sqrt {m}) \boldsymbol {h} _ {1} (\boldsymbol {x} (t), t) \right] + \text {constant}. \right. \\ \end{array}
$$

Since for any $h, \bar{h}, B$ , we have

$$
\mathbb {E} _ {t} \mathbb {E} _ {\boldsymbol {x} (t)} [ \boldsymbol {h} ^ {\top} (\boldsymbol {x} (t), t) \boldsymbol {B} \bar {\boldsymbol {h}} (\boldsymbol {x} (t), t) ] = \mathbb {E} _ {t} \mathbb {E} _ {\boldsymbol {x} (t)} [ \mathrm{trace} (\boldsymbol {B} \bar {\boldsymbol {h}} (\boldsymbol {x} (t), t) \boldsymbol {h} ^ {\top} (\boldsymbol {x} (t), t)) ]
$$

$$
= \mathrm{trace} (\pmb {B} \mathbb {E} _ {t} \mathbb {E} _ {\pmb {x} (t)} [ \bar {\pmb {h}} (\pmb {x} (t), t) \pmb {h} ^ {\top} (\pmb {x} (t), t) ]),
$$

we further get

$$
\tilde {\mathcal {L}} (\pmb {\theta}; \lambda (\cdot)) = \frac {1}{m} \mathrm{trace} (\pmb {A} ^ {\top} \pmb {A} \pmb {B} _ {1}) - \frac {2}{\sqrt {m}} \mathrm{trace} (\pmb {A} \pmb {B} _ {2}) + \mathrm{constant},
$$

where

$$
\pmb {B} _ {1} := \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\pmb {x} (t) \sim p _ {t}} [ \pmb {h} _ {1} (\pmb {x} (t), t) \pmb {h} _ {1} ^ {\top} (\pmb {x} (t), t) ], \quad \pmb {B} _ {2} := \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\pmb {x} (t) \sim p _ {t}} [ \pmb {h} _ {1} (\pmb {x} (t), t) \pmb {h} _ {2} ^ {\top} (\pmb {x} (t), t) ].
$$

Here, $B_{1}$ is a positive semi-definite matrix, since $v^{\top}B_{1}v = E_{t}\mathbb{E}_{\boldsymbol{x}(t)}[(\boldsymbol{v}^{\top}\boldsymbol{h}_{1}(\boldsymbol{x}(t),t))^{2}] \geq 0$ for any v. Notice that for any A, B,

$$
\mathrm{trace} (\boldsymbol {A} ^ {\top} \boldsymbol {A} \boldsymbol {B}) = \mathrm{trace} (\boldsymbol {A} \boldsymbol {B} \boldsymbol {A} ^ {\top}) = \sum_ {i, j} \boldsymbol {B} _ {i j} (\boldsymbol {A} _ {:, j}) ^ {\top} \boldsymbol {A} _ {:, i} = \mathrm{vec} (\boldsymbol {A}) ^ {\top} (\boldsymbol {B} \otimes \boldsymbol {I}) \mathrm{vec} (\boldsymbol {A}),
$$

$$
\mathrm{trace} (\boldsymbol {A} \boldsymbol {B}) = \sum_ {j} (\boldsymbol {A} _ {:, j}) ^ {\top} (\boldsymbol {B} ^ {\top}) _ {:, j} = \mathrm{vec} (\boldsymbol {A}) ^ {\top} \mathrm{vec} (\boldsymbol {B} ^ {\top}),
$$

where $\otimes$ denotes the Kronecker product. Hence

$$
\tilde {\mathcal {L}} (\boldsymbol {\theta}; \lambda (\cdot)) = \frac {1}{m} \operatorname{vec} (\boldsymbol {A}) ^ {\top} \left(\boldsymbol {B} _ {1} \otimes \boldsymbol {I}\right) \operatorname{vec} (\boldsymbol {A}) - \frac {2}{\sqrt {m}} \operatorname{vec} \left(\boldsymbol {B} _ {2} ^ {\top}\right) ^ {\top} \operatorname{vec} (\boldsymbol {A}) + \text { constant } \tag {23}
$$

is a quadratic function. It is straightforward to show that the eigenvalues of $B_{1} \otimes I$ are the same

as $B_{1}$ but with multiplicity, $^{6}$ implying that $B_{1} \otimes I$ is also positive semi-definite. Therefore,

$$
\nabla_ {\pmb {\theta}} ^ {2} \tilde {\mathcal {L}} (\pmb {\theta}; \lambda (\cdot)) = \nabla_ {\mathrm{vec} (\pmb {A})} ^ {2} \tilde {\mathcal {L}} (\pmb {\theta}; \lambda (\cdot)) = 2 (\pmb {B} _ {1} \otimes \pmb {I})
$$

is positive semi-definite, i.e., the loss is convex with respect to trainable parameters.

(2) For $\bar{\tilde{\mathcal{L}}} (\bar{\boldsymbol{\theta}};\lambda (\cdot))$ , notice that

$$
\| \bar {\boldsymbol {s}} _ {t, \bar {\boldsymbol {\theta}}} (\boldsymbol {x}) \| _ {2} ^ {2} = \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \boldsymbol {a} ^ {\top} (\boldsymbol {w}, \boldsymbol {u}) \right] \mathbb {E} _ {(\boldsymbol {w} ^ {\prime}, \boldsymbol {u} ^ {\prime}) \sim \rho_ {0}} \left[ \boldsymbol {a} (\boldsymbol {w} ^ {\prime}, \boldsymbol {u} ^ {\prime}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right]
$$

$$
= \mathbb {E} _ {\left(\boldsymbol {w}, \boldsymbol {u}\right), \left(\boldsymbol {w} ^ {\prime}, \boldsymbol {u} ^ {\prime}\right) \sim \rho_ {0}} \left[ \boldsymbol {a} ^ {\top} (\boldsymbol {w}, \boldsymbol {u}) \sigma \left(\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)\right) \sigma \left(\boldsymbol {w} ^ {\prime \top} \boldsymbol {x} + \boldsymbol {u} ^ {\prime \top} \boldsymbol {e} (t)\right) \boldsymbol {a} \left(\boldsymbol {w} ^ {\prime}, \boldsymbol {u} ^ {\prime}\right) \right].
$$

Let $\boldsymbol{v} := (\boldsymbol{w}, \boldsymbol{u})$ , $\boldsymbol{v}' := (\boldsymbol{w}', \boldsymbol{u}')$ , $\boldsymbol{z}(t) := (\boldsymbol{x}^\top(t), \boldsymbol{e}^\top(t))^{\top}$ , we get

$$
\begin{array}{l} \tilde {\mathcal {L}} (\bar {\boldsymbol {\theta}}; \lambda (\cdot)) = \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \left(\| \overline {{\boldsymbol {s}}} _ {t, \bar {\boldsymbol {\theta}}} (\boldsymbol {x} (t)) \| _ {2} ^ {2} - 2 \boldsymbol {s} _ {t} ^ {\top} (\boldsymbol {x} (t)) \overline {{\boldsymbol {s}}} _ {t, \bar {\boldsymbol {\theta}}} (\boldsymbol {x} (t)) + \| \boldsymbol {s} _ {t} (\boldsymbol {x} (t)) \| _ {2} ^ {2}\right) \right] \\ = \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \left(\mathbb {E} _ {\boldsymbol {v}, \boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ \boldsymbol {a} ^ {\top} (\boldsymbol {v}) \sigma \left(\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)\right) \sigma \left(\boldsymbol {v} ^ {\prime \top} \boldsymbol {z} (t)\right) \boldsymbol {a} \left(\boldsymbol {v} ^ {\prime}\right) \right] \right. \right. \\ \left. \left. - 2 \boldsymbol {s} _ {t} ^ {\top} (\boldsymbol {x} (t)) \mathbb {E} _ {\boldsymbol {v} \sim \rho_ {0}} \left[ \boldsymbol {a} (\boldsymbol {v}) \sigma \left(\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)\right) \right]\right) \right] + \text { constant } \\ = \mathbb {E} _ {\boldsymbol {v}, \boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ \boldsymbol {a} ^ {\top} (\boldsymbol {v}) \left(\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \sigma (\boldsymbol {v} ^ {\prime \top} \boldsymbol {z} (t)) \right]\right) \boldsymbol {a} (\boldsymbol {v} ^ {\prime}) \right] \\ - 2 \mathbb {E} _ {\boldsymbol {v} \sim \rho_ {0}} \left[ \left(\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \boldsymbol {s} _ {t} ^ {\top} (\boldsymbol {x} (t)) \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \right]\right) \boldsymbol {a} (\boldsymbol {v}) \right] + \text {constant.} \\ \end{array}
$$

Then for any $\phi \in L^2 (\rho_0)$ , by symmetry we have

$$
\begin{array}{l} \delta \bar {\tilde {\mathcal {L}}} (\cdot ; \lambda (\cdot)) [ \bar {\boldsymbol {\theta}}, \phi ] = \lim _ {\epsilon \rightarrow 0} \frac {1}{\epsilon} \left(\bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} + \epsilon \phi ; \lambda (\cdot)) - \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}}; \lambda (\cdot))\right) \\ = \mathbb {E} _ {\boldsymbol {v}, \boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ \phi^ {\top} (\boldsymbol {v}) \left(\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \sigma \left(\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)\right) \sigma \left(\boldsymbol {v} ^ {\prime \top} \boldsymbol {z} (t)\right) \right]\right) \boldsymbol {a} \left(\boldsymbol {v} ^ {\prime}\right) \right. \\ + \boldsymbol {a} ^ {\top} (\boldsymbol {v}) \left(\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \sigma \left(\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)\right) \sigma \left(\boldsymbol {v} ^ {\prime \top} \boldsymbol {z} (t)\right) \right]\right) \phi \left(\boldsymbol {v} ^ {\prime}\right) ] \\ \left. - 2 \mathbb {E} _ {\boldsymbol {v} \sim \rho_ {0}} \left[ \left(\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \boldsymbol {s} _ {t} ^ {\top} (\boldsymbol {x} (t)) \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \right]\right) \phi (\boldsymbol {v}) \right] \right. \\ = 2 \mathbb {E} _ {\boldsymbol {v}, \boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ \boldsymbol {a} ^ {\top} (\boldsymbol {v} ^ {\prime}) \left(\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \right]\right) \phi (\boldsymbol {v}) \right] \\ - 2 \mathbb {E} _ {\boldsymbol {v} \sim \rho_ {0}} \left[ \left(\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \boldsymbol {s} _ {t} ^ {\top} (\boldsymbol {x} (t)) \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \right]\right) \phi (\boldsymbol {v}) \right] \\ = 2 \left\langle \mathbb {E} _ {\boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ K (\boldsymbol {v}, \boldsymbol {v} ^ {\prime}; \lambda (\cdot)) \boldsymbol {a} (\boldsymbol {v} ^ {\prime}) \right] - \boldsymbol {k} (\boldsymbol {v}; \lambda (\cdot)), \phi (\boldsymbol {v}) \right\rangle_ {L ^ {2} \left(\rho_ {0}\right)}, \\ \end{array}
$$

where

$$
K (\boldsymbol {v}, \boldsymbol {v} ^ {\prime}; \lambda (\cdot)) := \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \right],
$$

$$
\pmb {k} (\pmb {v}; \lambda (\cdot)) := \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\pmb {x} (t) \sim p _ {t}} \left[ \lambda (t) \pmb {s} _ {t} (\pmb {x} (t)) \sigma (\pmb {v} ^ {\top} \pmb {z} (t)) \right].
$$

This yields

$$
\frac {\delta \tilde {\bar {\mathcal {L}}} (\cdot ; \lambda (\cdot))}{\delta \bar {\boldsymbol {\theta}}} = 2 \mathbb {E} _ {\boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ K (\boldsymbol {v}, \boldsymbol {v} ^ {\prime}; \lambda (\cdot)) \boldsymbol {a} (\boldsymbol {v} ^ {\prime}) \right] - 2 \boldsymbol {k} (\boldsymbol {v}; \lambda (\cdot)), \tag {24}
$$

and

$$
\begin{array}{l} \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} _ {1}; \lambda (\cdot)) + \left\langle \frac {\delta \tilde {\tilde {\mathcal {L}}} (\cdot ; \lambda (\cdot))}{\delta \bar {\boldsymbol {\theta}}} \bigg | _ {\bar {\boldsymbol {\theta}} = \bar {\boldsymbol {\theta}} _ {1}}, \bar {\boldsymbol {\theta}} _ {2} - \bar {\boldsymbol {\theta}} _ {1} \right\rangle_ {L ^ {2} (\rho_ {0})} - \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} _ {2}; \lambda (\cdot)) \\ = \mathbb {E} _ {\boldsymbol {v}, \boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ \boldsymbol {a} _ {1} ^ {\top} (\boldsymbol {v}) K (\boldsymbol {v}, \boldsymbol {v} ^ {\prime}; \lambda (\cdot)) \boldsymbol {a} _ {1} (\boldsymbol {v} ^ {\prime}) \right] - 2 \mathbb {E} _ {\boldsymbol {v} \sim \rho_ {0}} \left[ \boldsymbol {k} ^ {\top} (\boldsymbol {v}; \lambda (\cdot)) \boldsymbol {a} _ {1} (\boldsymbol {v}) \right] \\ + 2 \left\langle \mathbb {E} _ {\boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ K (\boldsymbol {v}, \boldsymbol {v} ^ {\prime}; \lambda (\cdot)) \boldsymbol {a} _ {1} (\boldsymbol {v} ^ {\prime}) \right] - \boldsymbol {k} (\boldsymbol {v}; \lambda (\cdot)), \boldsymbol {a} _ {2} (\boldsymbol {v}) - \boldsymbol {a} _ {1} (\boldsymbol {v}) \right\rangle_ {L ^ {2} \left(\rho_ {0}\right)} \\ - \mathbb {E} _ {\boldsymbol {v}, \boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ \boldsymbol {a} _ {2} ^ {\top} (\boldsymbol {v}) K (\boldsymbol {v}, \boldsymbol {v} ^ {\prime}; \lambda (\cdot)) \boldsymbol {a} _ {2} (\boldsymbol {v} ^ {\prime}) \right] + 2 \mathbb {E} _ {\boldsymbol {v} \sim \rho_ {0}} \left[ \boldsymbol {k} ^ {\top} (\boldsymbol {v}; \lambda (\cdot)) \boldsymbol {a} _ {2} (\boldsymbol {v}) \right] \\ = - \mathbb {E} _ {\boldsymbol {v}, \boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ \left(\boldsymbol {a} _ {2} (\boldsymbol {v}) - \boldsymbol {a} _ {1} (\boldsymbol {v})\right) ^ {\top} K \left(\boldsymbol {v}, \boldsymbol {v} ^ {\prime}; \lambda (\cdot)\right) \left(\boldsymbol {a} _ {2} \left(\boldsymbol {v} ^ {\prime}\right) - \boldsymbol {a} _ {1} \left(\boldsymbol {v} ^ {\prime}\right)\right) \right] \\ = - \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \mathbb {E} _ {\boldsymbol {v}, \boldsymbol {v} ^ {\prime} \sim \rho_ {0}} \left[ (\boldsymbol {a} _ {2} (\boldsymbol {v}) - \boldsymbol {a} _ {1} (\boldsymbol {v})) ^ {\top} \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \sigma (\boldsymbol {v} ^ {\prime \top} \boldsymbol {z} (t)) (\boldsymbol {a} _ {2} (\boldsymbol {v} ^ {\prime}) - \boldsymbol {a} _ {1} (\boldsymbol {v} ^ {\prime})) \right] \right] \\ = - \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \left\| \mathbb {E} _ {\boldsymbol {v} \sim \rho_ {0}} \left[ (\boldsymbol {a} _ {2} (\boldsymbol {v}) - \boldsymbol {a} _ {1} (\boldsymbol {v})) \sigma (\boldsymbol {v} ^ {\top} \boldsymbol {z} (t)) \right] \right\| _ {2} ^ {2} \right] \leq 0, \\ \end{array}
$$

hence the functional $\tilde{\mathcal{L}}(\bar{\boldsymbol{\theta}};\lambda(\cdot))$ is convex with respect to $\bar{\boldsymbol{\theta}}$ (given any positive weighting function $\lambda(\cdot)$ ). The proof is completed. ☐

Since the weighting function is fixed in our analysis, we omit the notation $\lambda(\cdot)$ without ambiguity in the following contents.

Lemma 4. For any $\tau > 0$ and $\theta, \bar{\theta}$ , we have

$$
\bar {\tilde {\mathcal {L}}} \left(\overline {{\boldsymbol {\theta}}} (\tau)\right) - \bar {\tilde {\mathcal {L}}} \left(\overline {{\boldsymbol {\theta}}}\right) \lesssim \frac {\left\| \overline {{\boldsymbol {s}}} _ {0 , \overline {{\boldsymbol {\theta}}} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \left\| \overline {{\boldsymbol {s}}} _ {0 , \overline {{\boldsymbol {\theta}}}} \right\| _ {\mathcal {H}} ^ {2}}{\tau}, \quad \tilde {\mathcal {L}} \left(\boldsymbol {\theta} (\tau)\right) - \tilde {\mathcal {L}} \left(\boldsymbol {\theta}\right) \lesssim \frac {\left\| \boldsymbol {s} _ {0 , \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \left\| \boldsymbol {s} _ {0 , \boldsymbol {\theta}} \right\| _ {\mathcal {H}} ^ {2}}{\tau}.
$$

Proof. (1) For the loss objective $\tilde{\mathcal{L}}$ , we define the Lyapunov function

$$
\bar {E} (\tau) := \tau \left(\bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) - \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}})\right) + \frac {1}{2} \| \boldsymbol {a} _ {\tau} - \boldsymbol {a} \| _ {L ^ {2} \left(\rho_ {0}\right)} ^ {2}.
$$

Then, we have

$$
\begin{array}{l} \frac {d}{d \tau} \bar {E} (\tau) = \left(\bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) - \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}})\right) + \tau \cdot \frac {d}{d \tau} \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) + \left\langle \boldsymbol {a} _ {\tau} - \boldsymbol {a}, \frac {d}{d \tau} \boldsymbol {a} _ {\tau} \right\rangle_ {L ^ {2} \left(\rho_ {0}\right)} \\ \leq \left(\bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) - \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}})\right) - \left\langle \boldsymbol {a} _ {\tau} - \boldsymbol {a}, \frac {\delta \bar {\tilde {\mathcal {L}}}}{\delta \bar {\boldsymbol {\theta}}} \Big | _ {\bar {\boldsymbol {\theta}} = \bar {\boldsymbol {\theta}} (\tau)} \right\rangle_ {L ^ {2} (\rho_ {0})}, \\ \end{array}
$$

where the last inequality holds since

$$
\frac {d}{d \tau} \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) = \left\langle \frac {\delta \bar {\tilde {\mathcal {L}}}}{\delta \bar {\boldsymbol {\theta}}} \Big | _ {\bar {\boldsymbol {\theta}} = \bar {\boldsymbol {\theta}} (\tau)}, \frac {d}{d \tau} \bar {\boldsymbol {\theta}} (\tau) \right\rangle_ {L ^ {2} (\rho_ {0})} = - \left\langle \frac {\delta \bar {\tilde {\mathcal {L}}}}{\delta \bar {\boldsymbol {\theta}}} \Big | _ {\bar {\boldsymbol {\theta}} = \bar {\boldsymbol {\theta}} (\tau)}, \frac {\delta \bar {\tilde {\mathcal {L}}}}{\delta \bar {\boldsymbol {\theta}}} \Big | _ {\bar {\boldsymbol {\theta}} = \bar {\boldsymbol {\theta}} (\tau)} \right\rangle_ {L ^ {2} (\rho_ {0})} \leq 0.
$$

By convexity, for any $\tau_{1},\tau_{2}$ , it holds that

$$
\bar {\tilde {\mathcal {L}}} \left(\bar {\boldsymbol {\theta}} (\tau_ {1})\right) + \left\langle \left. \boldsymbol {a} _ {\tau_ {2}} - \boldsymbol {a} _ {\tau_ {1}}, \frac {\delta \bar {\tilde {\mathcal {L}}}}{\delta \bar {\boldsymbol {\theta}}} \right| _ {\bar {\boldsymbol {\theta}} = \bar {\boldsymbol {\theta}} (\tau_ {1})} \right\rangle_ {L ^ {2} (\rho_ {0})} \leq \bar {\tilde {\mathcal {L}}} \left(\bar {\boldsymbol {\theta}} (\tau_ {2})\right),
$$

hence $\frac{d}{d\tau}\bar{E} (\tau)\leq 0$ . We conclude that $\bar{E} (\tau)\leq \bar{E} (0)$ , or equivalently

$$
\tau \left(\bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) - \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}})\right) + \frac {1}{2} \| \boldsymbol {a} _ {\tau} - \boldsymbol {a} \| _ {L ^ {2} \left(\rho_ {0}\right)} ^ {2} \leq \frac {1}{2} \| \boldsymbol {a} _ {0} - \boldsymbol {a} \| _ {L ^ {2} \left(\rho_ {0}\right)} ^ {2}.
$$

Therefore, note that $\left\| \bar{\boldsymbol{s}}_{0,\bar{\boldsymbol{\theta}}}\right\|_{\mathcal{H}}^2 := \mathbb{E}_{\rho_0}[\| \boldsymbol {a}\| _2^2 ]$ (let $\mathcal{H}:= \mathcal{H}_{k_{\rho_0}}$ ), we obtain

$$
\bar {\tilde {\mathcal {L}}} \left(\bar {\boldsymbol {\theta}} (\tau)\right) - \bar {\tilde {\mathcal {L}}} \left(\bar {\boldsymbol {\theta}}\right) \lesssim \frac {\left\| \boldsymbol {a} _ {0} - \boldsymbol {a} \right\| _ {L ^ {2} (\rho_ {0})} ^ {2}}{\tau} \lesssim \frac {\left\| \bar {\boldsymbol {s}} _ {0 , \bar {\boldsymbol {\theta}} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \left\| \bar {\boldsymbol {s}} _ {0 , \bar {\boldsymbol {\theta}}} \right\| _ {\mathcal {H}} ^ {2}}{\tau},
$$

which gives the desired estimate.

(2) For the loss objective $\tilde{\mathcal{L}}$ , the argument is almost the same, except replacing the Lyapunov function by

$$
E (\tau) := \tau \left(\tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau)) - \tilde {\mathcal {L}} (\boldsymbol {\theta})\right) + \frac {1}{2 m} \| \boldsymbol {A} _ {\tau} - \boldsymbol {A} \| _ {F} ^ {2},
$$

and $\langle \cdot, \cdot \rangle_{L^2(\rho_0)}$ by $\langle \cdot, \cdot \rangle$ . Note that $\boldsymbol{\theta} = \mathrm{vec}(\boldsymbol{A}) / \sqrt{m}$ and $\| \boldsymbol{s}_{0,\boldsymbol{\theta}} \|_{\mathcal{H}}^2 = \| \boldsymbol{A} \|_F^2 / m$ , we obtain

$$
\tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau)) - \tilde {\mathcal {L}} (\boldsymbol {\theta}) \lesssim \frac {\| \boldsymbol {A} _ {0} - \boldsymbol {A} \| _ {F} ^ {2}}{m \tau} \lesssim \frac {\| \boldsymbol {s} _ {0 , \boldsymbol {\theta} _ {0}} \| _ {\mathcal {H}} ^ {2} + \| \boldsymbol {s} _ {0 , \boldsymbol {\theta}} \| _ {\mathcal {H}} ^ {2}}{\tau},
$$

which completes the proof.

Lemma 5. Suppose that the loss objectives $\tilde{\mathcal{L}}$ , $\tilde{\mathcal{L}}^{(n)}$ , $\tilde{\tilde{\mathcal{L}}}$ , $\tilde{\tilde{\mathcal{L}}}^{(n)}$ are bounded at the initialization, then for any $\tau > 0$ , we have

$$
\left\| \pmb {s} _ {0, \pmb {\theta} (\tau)} \right\| _ {\mathcal {H}}, \left\| \pmb {s} _ {0, \hat {\pmb {\theta}} _ {n} (\tau)} \right\| _ {\mathcal {H}} \lesssim \left\| \pmb {s} _ {0, \pmb {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\frac {\tau}{m}}, \quad \left\| \overline {{\pmb {s}}} _ {0, \overline {{\pmb {\theta}}} (\tau)} \right\| _ {\mathcal {H}}, \left\| \overline {{\pmb {s}}} _ {0, \tilde {\pmb {\theta}} _ {n} (\tau)} \right\| _ {\mathcal {H}} \lesssim \left\| \overline {{\pmb {s}}} _ {0, \overline {{\pmb {\theta}}} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau}.
$$

Proof. It's sufficient to prove $\left\| \overline{\boldsymbol{s}}_{0,\bar{\boldsymbol{\theta}} (\tau)}\right\|_{\mathcal{H}}\lesssim \sqrt{\tau}$ , and the rest part follows similarly.

Since

$$
\| \pmb {a} _ {\tau} \| _ {L ^ {2} (\rho_ {0})} \frac {d}{d \tau} \| \pmb {a} _ {\tau} \| _ {L ^ {2} (\rho_ {0})} = \frac {1}{2} \frac {d}{d \tau} \| \pmb {a} _ {\tau} \| _ {L ^ {2} (\rho_ {0})} ^ {2} = \left\langle \pmb {a} _ {\tau}, \frac {d}{d \tau} \pmb {a} _ {\tau} \right\rangle_ {L ^ {2} (\rho_ {0})} = \left\langle \pmb {a} _ {\tau}, - \frac {\delta \bar {\tilde {\mathcal {L}}}}{\delta \bar {\pmb {\theta}}} \bigg | _ {\bar {\pmb {\theta}} = \bar {\pmb {\theta}} (\tau)} \right\rangle_ {L ^ {2} (\rho_ {0})},
$$

applying Cauchy–Schwartz inequality yields

$$
\begin{array}{l} \frac {d}{d \tau} \left\| \pmb {a} _ {\tau} \right\| _ {L ^ {2} (\rho_ {0})} = \left\langle \frac {\pmb {a} _ {\tau}}{\left\| \pmb {a} _ {\tau} \right\| _ {L ^ {2} (\rho_ {0})}}, - \frac {\delta \bar {\tilde {\mathcal {L}}}}{\delta \bar {\boldsymbol {\theta}}} \bigg | _ {\bar {\boldsymbol {\theta}} = \bar {\boldsymbol {\theta}} (\tau)} \right\rangle_ {L ^ {2} (\rho_ {0})} \\ \leq \left\| \right. \frac {\delta \tilde {\bar {\mathcal {L}}}}{\delta \bar {\boldsymbol {\theta}}} \left. \right| _ {\bar {\boldsymbol {\theta}} = \bar {\boldsymbol {\theta}} (\tau)} \left\| \right. _ {L ^ {2} (\rho_ {0})} = \sqrt {- \frac {d}{d \tau} \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau))}. \\ \end{array}
$$

Thus, again by Cauchy–Schwartz inequality, for any $\tau > \tau_{0} \geq 0$ , we have

$$
\begin{array}{l} \left\| \boldsymbol {a} _ {\tau} \right\| _ {L ^ {2} \left(\rho_ {0}\right)} - \left\| \boldsymbol {a} _ {\tau_ {0}} \right\| _ {L ^ {2} \left(\rho_ {0}\right)} \leq \int_ {\tau_ {0}} ^ {\tau} \sqrt {- \frac {d}{d s} \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (s))} d s \\ \leq \sqrt {\tau - \tau_ {0}} \sqrt {- \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) + \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau_ {0}))} \\ \leq \sqrt {\tau} \sqrt {\bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (0))}. \\ \end{array}
$$

By choosing $\tau_0 = 0$ , we have $\| \pmb{a}_{\tau}\|_{L^2 (\rho_0)}\lesssim \left\| \overline{\pmb{s}}_{0,\bar{\pmb{\theta}} (0)}\right\|_{\mathcal{H}} + \sqrt{\tau}$ , hence $\left\| \overline{\pmb{s}}_{0,\bar{\pmb{\theta}} (\tau)}\right\|_{\mathcal{H}}\lesssim \left\| \overline{\pmb{s}}_{0,\bar{\pmb{\theta}} (0)}\right\|_{\mathcal{H}} + \sqrt{\tau}$ , which completes the proof.

Lemma 6 (Monte Carlo estimates). Define the Monte Carlo error

$$
\operatorname{Err} _ {\mathrm{MC}} = \operatorname{Err} _ {\mathrm{MC}} (\boldsymbol {\theta}, \bar {\boldsymbol {\theta}}; T, \lambda (\cdot)) := \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \| \boldsymbol {s} _ {t, \boldsymbol {\theta}} (\boldsymbol {x} (t)) - \bar {\boldsymbol {s}} _ {t, \bar {\boldsymbol {\theta}}} (\boldsymbol {x} (t)) \| _ {2} ^ {2} \right] \right]. \tag {25}
$$

Suppose that $\|x(0)\|_{\infty} \leq 1$ , and the trainable parameter a and embedding function $e(\cdot)$ are both bounded. Then, given any $\bar{\theta}$ , for any $\delta > 0$ , $\delta \ll 1$ , with the probability of at least $1 - \delta$ , there exists $\theta$ such that

$$
\operatorname{Err} _ {\mathrm{MC}} \lesssim \frac {\log^ {2} \left(1 / \delta^ {2}\right)}{m} d, \tag {26}
$$

where $\lesssim$ hides universal positive constants only depending on $T$ .

Proof. Fix any $\bar{\theta}$ . According to Lemma 1, for any $\delta > 0$ , $\delta \ll 1$ , with the probability of at least $1 - \delta$ , we have

$$
\left\| \boldsymbol {x} (t) \right\| _ {\infty} \lesssim C _ {T} \left(1 + \sqrt {\log \left(1 / \delta^ {2}\right)}\right) \triangleq C _ {T, \delta}. \tag {27}
$$

Based on the representation (13), for any $\boldsymbol{W} = (\boldsymbol{w}_{1}, \ldots, \boldsymbol{w}_{m})^{\top} \in \mathbb{R}^{m \times d}$ , $\boldsymbol{U} = (\boldsymbol{u}_{1}, \ldots, \boldsymbol{u}_{m})^{\top} \in \mathbb{R}^{m \times d_{e}}$ with $(\boldsymbol{w}_{i}, \boldsymbol{u}_{i}) \sim \rho_{0}$ , $i = 1, \cdots, m$ , let $\boldsymbol{A} := (\boldsymbol{a}_{1}, \ldots, \boldsymbol{a}_{m}) \in \mathbb{R}^{d \times m}$ with $\boldsymbol{a}_{i} := \boldsymbol{a}(\boldsymbol{w}_{i}, \boldsymbol{u}_{i})$ for $i = 1, \cdots, m$ , and

$$
\boldsymbol {s} _ {t, \theta} (\boldsymbol {x}) := \frac {1}{m} \boldsymbol {A} \sigma (\boldsymbol {W} \boldsymbol {x} + \boldsymbol {U} \boldsymbol {e} (t)) = \frac {1}{m} \sum_ {i = 1} ^ {m} \boldsymbol {a} _ {i} \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)), \tag {28}
$$

then $\mathbb{E}_{\boldsymbol{W},\boldsymbol{U}}[\boldsymbol{s}_{t,\boldsymbol{\theta}}(\boldsymbol{x})]=\mathbb{E}_{(\boldsymbol{w},\boldsymbol{u})\sim\rho_{0}}\left[\boldsymbol{a}(\boldsymbol{w},\boldsymbol{u})\sigma(\boldsymbol{w}^{\top}\boldsymbol{x}+\boldsymbol{u}^{\top}\boldsymbol{e}(t))\right]=\bar{\boldsymbol{s}}_{t,\bar{\boldsymbol{\theta}}}(\boldsymbol{x})$ . For $k=1,\cdots,d$ , let

$$
\begin{array}{l} Z _ {t, k} (\boldsymbol {W}, \boldsymbol {U}) := \left\| s _ {t, \boldsymbol {\theta}, k} (\boldsymbol {x}) - \bar {s} _ {t, \bar {\boldsymbol {\theta}}, k} (\boldsymbol {x}) \right\| _ {L ^ {2} (p _ {t})} = \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} ^ {1 / 2} \left[ \left| s _ {t, \boldsymbol {\theta}, k} (\boldsymbol {x}) - \bar {s} _ {t, \bar {\boldsymbol {\theta}}, k} (\boldsymbol {x}) \right| ^ {2} \right] \\ = \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} ^ {1 / 2} \left[ \left| \frac {1}{m} \sum_ {i = 1} ^ {m} a _ {i, k} \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) - \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right] \right| ^ {2} \right]. \\ \end{array}
$$

If $(\tilde{\boldsymbol{W}},\tilde{\boldsymbol{U}})$ is different from $(\boldsymbol{W},\boldsymbol{U})$ at only one component indexed by i, we have

$$
\begin{array}{l} \left| Z _ {t, k} (\boldsymbol {W}, \boldsymbol {U}) - Z _ {t, k} (\tilde {\boldsymbol {W}}, \tilde {\boldsymbol {U}}) \right| \\ = \left| \left\| s _ {t, \boldsymbol {\theta}, k} (\boldsymbol {x}) - \bar {s} _ {t, \bar {\boldsymbol {\theta}}, k} (\boldsymbol {x}) \right\| _ {L ^ {2} (p _ {t})} - \left\| s _ {t, \tilde {\boldsymbol {\theta}}, k} (\boldsymbol {x}) - \bar {s} _ {t, \bar {\boldsymbol {\theta}}, k} (\boldsymbol {x}) \right\| _ {L ^ {2} (p _ {t})} \right| \\ \stackrel {\mathrm{(i)}} {\leq} \left\| s _ {t, \boldsymbol {\theta}, k} (\boldsymbol {x}) - s _ {t, \tilde {\boldsymbol {\theta}}, k} (\boldsymbol {x}) \right\| _ {L ^ {2} (p _ {t})} \\ = \frac {1}{m} \left\| a _ {i, k} \sigma \left(\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)\right) - \tilde {a} _ {i, k} \sigma \left(\tilde {\boldsymbol {w}} _ {i} ^ {\top} \boldsymbol {x} + \tilde {\boldsymbol {u}} _ {i} ^ {\top} \boldsymbol {e} (t)\right) \right\| _ {L ^ {2} \left(p _ {t}\right)} \\ \leq \frac {1}{m} \left(| a _ {i, k} | \left\| \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right\| _ {L ^ {2} (p _ {t})} + | \tilde {a} _ {i, k} | \left\| \sigma (\tilde {\boldsymbol {w}} _ {i} ^ {\top} \boldsymbol {x} + \tilde {\boldsymbol {u}} _ {i} ^ {\top} \boldsymbol {e} (t)) \right\| _ {L ^ {2} (p _ {t})}\right) \\ \stackrel {\text {(ii)}} {\leq} \frac {1}{m} \left( \right.\left| a _ {i, k} \right|\left(\left\| \boldsymbol {w} _ {i} \right\| _ {1} C _ {T, \delta} + \left\| \boldsymbol {u} _ {i} \right\| _ {1} \| \boldsymbol {e} (t) \| _ {\infty}\right) + \left| \tilde {a} _ {i, k} \right|\left(\left\| \tilde {\boldsymbol {w}} _ {i} \right\| _ {1} C _ {T, \delta} + \left\| \tilde {\boldsymbol {u}} _ {i} \right\| _ {1} \| \boldsymbol {e} (t) \| _ {\infty})\right) \\ \stackrel {\mathrm{(iii)}} {\leq} \frac {1}{m} \left(| a _ {i, k} | + | \tilde {a} _ {i, k} |\right) (C _ {T, \delta} + \| \boldsymbol {e} (t) \| _ {\infty}) \\ \stackrel {\mathrm{(iv)}} {\lesssim} \frac {1}{m} \left(C _ {T, \delta} + C _ {T, e}\right), \\ \end{array}
$$

where (i) is from the triangle inequality, (ii) is due to the fact that $|\sigma(y)| = |\mathrm{ReLU}(y)| \leq |y|$ for any $y \in \mathbb{R}$ , the triangle and Hölder's inequality and (27), (iii) follows from the positive homogeneity property of the ReLU activation, and (iv) is due to the boundness of the trainable parameter $\pmb{a}$ and embedding function $\pmb{e}(\cdot)$ . By McDiarmid's inequality (see e.g., Lemma 26.4 in [43]), for any $\delta > 0$ , with the probability of at least $1 - \delta$ , we have

$$
\left| Z _ {t, k} (\boldsymbol {W}, \boldsymbol {U}) - \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} \left[ Z _ {t, k} (\boldsymbol {W}, \boldsymbol {U}) \right] \right| \lesssim \frac {1}{m} \left(C _ {T, \delta} + C _ {T, \boldsymbol {e}}\right) \sqrt {m \log (2 / \delta) / 2} \tag {29}
$$

$$
\lesssim (C _ {T, \delta} + C _ {T, e}) \sqrt {\frac {\log (1 / \delta)}{m}}. \tag {30}
$$

Since

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} \left[ Z _ {t, k} ^ {2} (\boldsymbol {W}, \boldsymbol {U}) \right] \\ = \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} \left[ \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \left| s _ {t, \boldsymbol {\theta}, k} (\boldsymbol {x}) - \bar {s} _ {t, \bar {\boldsymbol {\theta}}, k} (\boldsymbol {x}) \right| ^ {2} \right] \right] \\ \stackrel {\mathrm{(v)}} {=} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} \left[ \left| s _ {t, \boldsymbol {\theta}, k} (\boldsymbol {x}) - \bar {s} _ {t, \bar {\boldsymbol {\theta}}, k} (\boldsymbol {x}) \right| ^ {2} \right] \right] \\ \end{array}
$$

$$
= \frac {1}{m ^ {2}} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \right. \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} \left[\left.\left| \sum_ {i = 1} ^ {m} \left(a _ {i, k} \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) - \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right]\right)\right| ^ {2} \right]\right]
$$

$$
= \frac {1}{m ^ {2}} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} \left[ \sum_ {i = 1} ^ {m} \left(a _ {i, k} \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) - \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right]\right) ^ {2} \right] \right]
$$

$$
+ \frac {1}{m ^ {2}} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} \left[ \sum_ {i \neq j} \left(a _ {i, k} \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) - \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right]\right) \right. \right.
$$

$$
\left. \times \left(a _ {j, k} \sigma (\boldsymbol {w} _ {j} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {e} (t)) - \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right]\right) \right]
$$

$$
= \frac {1}{m ^ {2}} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \sum_ {i = 1} ^ {m} \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ \left(a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) - \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right]\right) ^ {2} \right] \right]
$$

$$
+ \frac {1}{m ^ {2}} \mathbb {E} _ {\pmb {x} \sim p _ {t}} \Bigg [ \sum_ {i \neq j} \mathbb {E} _ {(\pmb {w} _ {i}, \pmb {u} _ {i}) \sim \rho_ {0}} \Bigg [ \left(a _ {i, k} \sigma (\pmb {w} _ {i} ^ {\top} \pmb {x} + \pmb {u} _ {i} ^ {\top} \pmb {e} (t)) - \mathbb {E} _ {(\pmb {w}, \pmb {u}) \sim \rho_ {0}} \left[ a _ {k} (\pmb {w}, \pmb {u}) \sigma (\pmb {w} ^ {\top} \pmb {x} + \pmb {u} ^ {\top} \pmb {e} (t)) \right]\right) \Bigg ]
$$

$$
\times \mathbb {E} _ {\left(\boldsymbol {w} _ {j}, \boldsymbol {u} _ {j}\right) \sim \rho_ {0}} \left[ \left(a _ {j, k} \sigma \left(\boldsymbol {w} _ {j} ^ {\top} \boldsymbol {x} + \boldsymbol {u} _ {j} ^ {\top} \boldsymbol {e} (t)\right) - \mathbb {E} _ {\left(\boldsymbol {w}, \boldsymbol {u}\right) \sim \rho_ {0}} \left[ a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma \left(\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)\right) \right]\right) \right]
$$

$$
= \frac {1}{m ^ {2}} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \sum_ {i = 1} ^ {m} \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ \left(a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) - \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right]\right) ^ {2} \right] \right]
$$

$$
\leq \frac {1}{m} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ \left(a _ {k} (\boldsymbol {w}, \boldsymbol {u}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t))\right) ^ {2} \right] \right]
$$

$$
\stackrel {\text {(ii)}} {\leq} \frac {1}{m} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ \left(| a _ {k} (\boldsymbol {w}, \boldsymbol {u}) | \left(\| \boldsymbol {w} \| _ {1} C _ {T, \delta} + \| \boldsymbol {u} \| _ {1} \| \boldsymbol {e} (t) \| _ {\infty}\right)\right) ^ {2} \right] \right]
$$

$$
\stackrel {\text {(iii)}} {\leq} \frac {1}{m} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ \left(| a _ {k} (\boldsymbol {w}, \boldsymbol {u}) | (C _ {T, \delta} + \| \boldsymbol {e} (t) \| _ {\infty})\right) ^ {2} \right] \right]
$$

$$
\stackrel {\text {(iv)}} {\lesssim} \frac {1}{m} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \mathbb {E} _ {(\boldsymbol {w}, \boldsymbol {u}) \sim \rho_ {0}} \left[ (C _ {T, \delta} + C _ {T, e}) ^ {2} \right] \right] \tag {31}
$$

$$
\lesssim \left(C _ {T, \delta} + C _ {T, \boldsymbol {e}}\right) ^ {2} \frac {1}{m}, \tag {32}
$$

where (v) is due to Fubini's theorem, and (ii), (iii), (iv) is the same as before. By the triangle inequality, Jensen's inequality, (29) and (31), we obtain

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \| \boldsymbol {s} _ {t, \boldsymbol {\theta}} (\boldsymbol {x}) - \bar {\boldsymbol {s}} _ {t, \bar {\boldsymbol {\theta}}} (\boldsymbol {x}) \| _ {2} ^ {2} \right] = \sum_ {k = 1} ^ {d} \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \left| s _ {t, \boldsymbol {\theta}, k} (\boldsymbol {x}) - \bar {s} _ {t, \bar {\boldsymbol {\theta}}, k} (\boldsymbol {x}) \right| ^ {2} \right] \\ = \sum_ {k = 1} ^ {d} Z _ {t, k} ^ {2} (\boldsymbol {W}, \boldsymbol {U}) \\ \leq \sum_ {k = 1} ^ {d} \left(| Z _ {t, k} (\boldsymbol {W}, \boldsymbol {U}) - \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} [ Z _ {t, k} (\boldsymbol {W}, \boldsymbol {U}) ] | + | \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} [ Z _ {t, k} (\boldsymbol {W}, \boldsymbol {U}) ] |\right) ^ {2} \\ \end{array}
$$

$$
\lesssim \sum_ {k = 1} ^ {d} \left(| Z _ {t, k} (\boldsymbol {W}, \boldsymbol {U}) - \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} [ Z _ {t, k} (\boldsymbol {W}, \boldsymbol {U}) ] | ^ {2} + \mathbb {E} _ {\boldsymbol {W}, \boldsymbol {U}} [ Z _ {t, k} ^ {2} (\boldsymbol {W}, \boldsymbol {U}) ]\right)
$$

$$
\lesssim \sum_ {k = 1} ^ {d} (C _ {T, \delta} + C _ {T, \boldsymbol {e}}) ^ {2} \frac {\log (1 / \delta)}{m} = (C _ {T, \delta} + C _ {T, \boldsymbol {e}}) ^ {2} \frac {\log (1 / \delta)}{m} d, \tag {33}
$$

which gives

$$
\mathrm{Err} _ {\mathrm{MC}} = \frac {1}{T} \int_ {0} ^ {T} \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} \sim p _ {t}} \left[ \| \boldsymbol {s} _ {t, \boldsymbol {\theta}} (\boldsymbol {x}) - \bar {\boldsymbol {s}} _ {t, \bar {\boldsymbol {\theta}}} (\boldsymbol {x}) \| _ {2} ^ {2} \right] d t
$$

$$
\lesssim (C _ {T, \delta} + C _ {T, \pmb {e}}) ^ {2} \frac {\log (1 / \delta)}{m} d \cdot \frac {1}{T} \int_ {0} ^ {T} \lambda (t) d t \leq \frac {\log^ {2} (1 / \delta^ {2})}{m} d.
$$

Obviously, $\frac{1}{m}\| \pmb {A}\| _F^2 = \frac{1}{m}\sum_{i = 1}^{m}\| \pmb {a}(\pmb {w}_i,\pmb {u}_i)\| _2^2 = \| \pmb{s}_{t,\pmb {\theta}}\|_{\mathcal{H}_{k_{\rho_0}}}^2 < \infty$ . The proof is completed.

Now we are ready to prove the main theorem.

Proof of Theorem 1. Based on Lemma 2, we have

$$
D _ {\mathrm{KL}} \left(p _ {0} \| p _ {0, \hat {\boldsymbol {\theta}} _ {n} (\tau)}\right) \leq \tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau); g ^ {2} (\cdot)) + D _ {\mathrm{KL}} \left(p _ {T} \| \pi\right).
$$

To bound the term $\tilde{\mathcal{L}}(\hat{\boldsymbol{\theta}}_{n}(\tau); g^{2}(\cdot))$ , we use the following decomposition:

$$
\tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau)) = \left[ \tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau)) - \tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau)) \right] + \tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau))
$$

$$
\lesssim \left[ \tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau)) - \tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau)) \right] + \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) + \text {Monte Carlo} \triangleq I _ {1} + I _ {2} + I _ {3}.
$$

According to Lemma 4, we obtain

$$
I _ {2} \lesssim \bar {\tilde {\mathcal {L}}} \left(\bar {\boldsymbol {\theta}} ^ {*}\right) + \frac {1}{\tau} \left(\left\| \bar {\boldsymbol {s}} _ {0, \bar {\boldsymbol {\theta}} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \left\| \bar {\boldsymbol {s}} _ {0, \bar {\boldsymbol {\theta}} ^ {*}} \right\| _ {\mathcal {H}} ^ {2}\right).
$$

The term $I_{3}$ can be further divided into

$$
I _ {3} := \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \| \boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \bar {\boldsymbol {s}} _ {t, \bar {\boldsymbol {\theta}} (\tau)} (\boldsymbol {x} (t)) \| _ {2} ^ {2} \right] \right]
$$

$$
\lesssim \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \| \bar {\boldsymbol {s}} _ {t, \bar {\boldsymbol {\theta}} (\tau)} (\boldsymbol {x} (t)) - \bar {\boldsymbol {s}} _ {t, \bar {\boldsymbol {\theta}} ^ {*}} (\boldsymbol {x} (t)) \| _ {2} ^ {2} \right] \right]
$$

$$
+ \mathbb {E} _ {t \sim \mathcal {U} (0, T)} [ \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} [ \| \boldsymbol {s} _ {t, \boldsymbol {\theta} ^ {*}} (\boldsymbol {x} (t)) - \bar {\boldsymbol {s}} _ {t, \bar {\boldsymbol {\theta}} ^ {*}} (\boldsymbol {x} (t)) \| _ {2} ^ {2} ] ]
$$

$$
+ \mathbb {E} _ {t \sim \mathcal {U} (0, T)} [ \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} [ \| \boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \boldsymbol {s} _ {t, \boldsymbol {\theta} ^ {*}} (\boldsymbol {x} (t)) \| _ {2} ^ {2} ] ]
$$

$$
=: I _ {3, 1} + I _ {3, 2} + I _ {3, 3},
$$

where $\pmb{\theta}^{*}$ is the Monte Carlo estimator of $\bar{\pmb{\theta}}^{*}$ . By Lemma 6, for any $\delta > 0$ , $\delta \ll 1$ , with the

probability of at least $1 - \delta$ , it holds that

$$
I _ {3, 2} = \mathrm{Err} _ {\mathrm{MC}} (\boldsymbol {\theta} ^ {*}, \bar {\boldsymbol {\theta}} ^ {*}; T, \lambda (\cdot)) \lesssim \frac {\log^ {2} (1 / \delta^ {2})}{m} d, \tag {34}
$$

while Lemma 4 gives

$$
I _ {3, 1} \lesssim \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} (\tau)) + \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} ^ {*}) = I _ {2} + \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} ^ {*}) \lesssim \bar {\tilde {\mathcal {L}}} \left(\bar {\boldsymbol {\theta}} ^ {*}\right) + \frac {1}{\tau} \left(\left\| \overline {{\boldsymbol {s}}} _ {0, \bar {\boldsymbol {\theta}} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \left\| \overline {{\boldsymbol {s}}} _ {0, \bar {\boldsymbol {\theta}} ^ {*}} \right\| _ {\mathcal {H}} ^ {2}\right),
$$

and similarly,

$$
I _ {3, 3} \lesssim \tilde {\mathcal {L}} (\pmb {\theta} ^ {*}) + \frac {1}{\tau} \left(\| \pmb {s} _ {0, \pmb {\theta} _ {0}} \| _ {\mathcal {H}} ^ {2} + \| \pmb {s} _ {0, \pmb {\theta} ^ {*}} \| _ {\mathcal {H}} ^ {2}\right).
$$

Hence, for any $\delta > 0$ , $\delta \ll 1$ , with the probability of at least $1 - \delta$ , it holds that

$$
I _ {3} \lesssim \frac {\log^ {2} (1 / \delta^ {2})}{m} d + \bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} ^ {*}) + \tilde {\mathcal {L}} (\boldsymbol {\theta} ^ {*}) + \frac {1}{\tau} \left(\left\| \bar {\boldsymbol {s}} _ {0, \bar {\boldsymbol {\theta}} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \left\| \bar {\boldsymbol {s}} _ {0, \bar {\boldsymbol {\theta}} ^ {*}} \right\| _ {\mathcal {H}} ^ {2} + \| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \| _ {\mathcal {H}} ^ {2} + \| \boldsymbol {s} _ {0, \boldsymbol {\theta} ^ {*}} \| _ {\mathcal {H}} ^ {2}\right).
$$

For the term $I_{1}$ , we have

$$
\begin{array}{l} \sqrt {\tilde {\mathcal {L}} (\hat {\pmb {\theta}} _ {n} (\tau))} - \sqrt {\tilde {\mathcal {L}} (\pmb {\theta} (\tau))} \\ = \left\{\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \| \boldsymbol {s} _ {t, \hat {\boldsymbol {\theta}} _ {n} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t)) \| _ {2} ^ {2} \right] \right] \right\} ^ {\frac {1}{2}} \\ \left. \right. - \left\{\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \cdot \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \| \boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t)) \| _ {2} ^ {2} \right]\right]\right\} ^ {\frac {1}{2}} \\ = \left\{\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\pmb {x} (t) \sim p _ {t}} \left[ \lambda (t) \| \pmb {s} _ {t, \hat {\pmb {\theta}} _ {n} (\tau)} (\pmb {x} (t)) - \nabla_ {\pmb {x} (t)} \log p _ {t} (\pmb {x} (t)) \| _ {2} ^ {2} \right] \right\} ^ {\frac {1}{2}} \\ - \left\{\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\pmb {x} (t) \sim p _ {t}} \left[ \lambda (t) \| \pmb {s} _ {t, \pmb {\theta} (\tau)} (\pmb {x} (t)) - \nabla_ {\pmb {x} (t)} \log p _ {t} (\pmb {x} (t)) \| _ {2} ^ {2} \right] \right\} ^ {\frac {1}{2}} \\ \leq \left\{\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \| \boldsymbol {s} _ {t, \hat {\boldsymbol {\theta}} _ {n} (\tau)} (\boldsymbol {x} (t)) - \boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) \| _ {2} ^ {2} \right] \right\} ^ {\frac {1}{2}} \\ = \left\{\mathbb {E} _ {t \sim \mathcal {U} (0, T)} \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \lambda (t) \left\| \frac {1}{m} \sum_ {i = 1} ^ {m} \hat {\boldsymbol {a}} _ {i} (\tau) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) - \frac {1}{m} \sum_ {i = 1} ^ {m} \boldsymbol {a} _ {i} (\tau) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right\| _ {2} ^ {2} \right] \right\} ^ {\frac {1}{2}}. \\ \end{array}
$$

Notice that

$$
\left\| \frac {1}{m} \sum_ {i = 1} ^ {m} (\hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau)) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right\| _ {2} ^ {2}
$$

$$
\leq \frac {1}{m ^ {2}} \left(\sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} \left| \sigma \left(\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)\right) \right|\right) ^ {2}
$$

$$
\leq \frac {1}{m ^ {2}} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \sum_ {i = 1} ^ {m} \left| \sigma \left(\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)\right) \right| ^ {2}
$$

$$
\lesssim \frac {1}{m ^ {2}} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \sum_ {i = 1} ^ {m} \left(\left| \boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) \right| ^ {2} + \left| \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t) \right| ^ {2}\right)
$$

$$
\leq \frac {1}{m ^ {2}} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \sum_ {i = 1} ^ {m} \left(\| \boldsymbol {w} _ {i} \| _ {1} ^ {2} \| \boldsymbol {x} (t) \| _ {\infty} ^ {2} + \| \boldsymbol {u} _ {i} \| _ {1} ^ {2} \| \boldsymbol {e} (t) \| _ {\infty} ^ {2}\right)
$$

$$
\lesssim \frac {1}{m ^ {2}} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \sum_ {i = 1} ^ {m} \left(C _ {T, \delta} ^ {2} + C _ {T, \boldsymbol {e}} ^ {2}\right),
$$

which gives

$$
\sqrt {\tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau))} - \sqrt {\tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau))} \lesssim \left\{\frac {1}{m} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \left(C _ {T, \delta} ^ {2} + C _ {T, \boldsymbol {e}} ^ {2}\right) \right\} ^ {\frac {1}{2}}
$$

$$
\leq \left(C _ {T, \delta} + C _ {T, \boldsymbol {e}}\right) \left\{\frac {1}{m} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \right\} ^ {\frac {1}{2}}.
$$

Here, the triangle inequality, Cauchy–Schwartz inequality, the fact that $|\sigma(y)| = |\text{ReLU}(y)| \leq |y|$ for any $y \in R$ , Hölder's inequality, the positive homogeneity property of the ReLU activation, and the boundness of the input data, embedding function $\boldsymbol{e}(t)$ and weighting function $\lambda(t)$ . Thus, we get

$$
\begin{array}{l} \tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau)) - \tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau)) \\ \lesssim \frac {1}{m} \left(C _ {T, \delta} ^ {2} + C _ {T, e} ^ {2}\right) \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} + \sqrt {\tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau))} \left(C _ {T, \delta} + C _ {T, e}\right) \left\{\frac {1}{m} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \right\} ^ {\frac {1}{2}} \\ \lesssim \sqrt {\tilde {\mathcal {L}} (\boldsymbol {\theta} ^ {*}) + \frac {1}{\tau} \left(\| \boldsymbol {s} _ {0 , \boldsymbol {\theta} _ {0}} \| _ {\mathcal {H}} ^ {2} + \| \boldsymbol {s} _ {0 , \boldsymbol {\theta} ^ {*}} \| _ {\mathcal {H}} ^ {2}\right)} \left(C _ {T, \delta} + C _ {T, \boldsymbol {e}}\right) \left\{\frac {1}{m} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \right\} ^ {\frac {1}{2}} \\ + \frac {1}{m} \left(C _ {T, \delta} ^ {2} + C _ {T, e} ^ {2}\right) \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2}, \\ \end{array}
$$

where the last inequality follows from Lemma 4. We further deduce that

$$
\begin{array}{l} \frac {1}{m} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \\ = \frac {1}{m} \sum_ {i = 1} ^ {m} \left\| \int_ {0} ^ {\tau} \frac {d}{d \tau_ {0}} (\hat {\boldsymbol {a}} _ {i} (\tau_ {0}) - \boldsymbol {a} _ {i} (\tau_ {0})) d \tau_ {0} \right\| _ {2} ^ {2} \\ = \frac {1}{m} \sum_ {i = 1} ^ {m} \left\| \int_ {0} ^ {\tau} \left(\nabla_ {\boldsymbol {\theta} _ {i} (\tau_ {0})} \tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau_ {0})) - \nabla_ {\hat {\boldsymbol {\theta}} _ {n, i} (\tau_ {0})} \hat {\tilde {\mathcal {L}}} _ {n} (\hat {\boldsymbol {\theta}} _ {n} (\tau_ {0}))\right) d \tau_ {0} \right\| _ {2} ^ {2} \\ = \frac {1}{m ^ {2}} \sum_ {i = 1} ^ {m} \left\| \int_ {0} ^ {\tau} \left(2 \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \left(\boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau_ {0})} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t))\right) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right] \right. \right. \right. \\ - 2 \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t} ^ {(n)}} \left[\left(\boldsymbol {s} _ {t, \hat {\boldsymbol {\theta}} _ {n} (\tau_ {0})} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t))\right) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right]\right]\left. \right) d \tau_ {0} \bigg \| _ {2} ^ {2}, \\ \end{array}
$$

where $p_{t}^{(n)}$ denotes the empirical distribution of $p_{t}$ . Note that

$$
\| \boldsymbol {\theta} \| _ {2} ^ {2} = \| \operatorname{vec} (\boldsymbol {A}) \| _ {2} ^ {2} / m = \| \boldsymbol {A} \| _ {F} ^ {2} / m = \| \boldsymbol {s} _ {0, \boldsymbol {\theta}} \| _ {\mathcal {H}} ^ {2}, \tag {35}
$$

by Lemma 5 we get $\| \pmb{\theta}(\tau)\| _2 = \| s_{0,\pmb{\theta}(\tau)}\|_{\mathcal{H}}\lesssim \| s_{0,\pmb{\theta}_0}\|_{\mathcal{H}} + \sqrt{\tau / m}$ . For any $t\in [0,T]$ , define the function space

$$
\mathcal {F} _ {t} := \left\{\boldsymbol {f} _ {1} (\boldsymbol {x} (t); \boldsymbol {\theta} _ {1} (\tau)) f _ {2} (\boldsymbol {x} (t); \boldsymbol {\theta} _ {2}): \boldsymbol {f} _ {1} \in \mathcal {F} _ {1, t}, f _ {2} \in \mathcal {F} _ {2, t} \right\},
$$

where

$$
\mathcal {F} _ {1, t} := \left\{\boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t)): \| \boldsymbol {\theta} (\tau) \| _ {2} \lesssim \left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m} \right\},
$$

$$
\mathcal {F} _ {2, t} := \left\{\sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)): \| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1 \right\}.
$$

Then, according to Theorem A.5 in [61], for any $\delta \in (0,1)$ , with the probability at least $1 - \delta$ over the choice of the dataset $\mathcal{D}_{\boldsymbol{x}} = \{\boldsymbol{x}_i\}_{i=1}^n$ , it holds that

$$
\begin{array}{l} \left| \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \left(\boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t))\right) \sigma \left(\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)\right) \right] \right. \\ - \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t} ^ {(n)}} \left[ \left(\boldsymbol {s} _ {t, \hat {\boldsymbol {\theta}} _ {n} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t))\right) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right] \Bigg | \\ \leq \left| \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t}} \left[ \left(\boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t))\right) \sigma \left(\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)\right) \right] \right. \\ - \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t} ^ {(n)}} \left[ \left(\boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t))\right) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right] \Bigg | \\ + \left| \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t} ^ {(n)}} \left[ \left(\boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t))\right) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right] \right. \\ - \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t} ^ {(n)}} \left[ \left(\boldsymbol {s} _ {t, \hat {\boldsymbol {\theta}} _ {n} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t))\right) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right] \Bigg | \\ \lesssim \widehat {\operatorname{Rad}} _ {n} (\mathcal {F} _ {t}) + \sup _ {\boldsymbol {f} \in \mathcal {F} _ {t}, \boldsymbol {x} (t) \in [ - C _ {T, \delta}, C _ {T, \delta} ] ^ {d}} | \boldsymbol {f} (\boldsymbol {x} (t)) | \sqrt {\frac {\log (2 / \delta)}{n}} \\ + \left| \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t} ^ {(n)}} \left[ \left(\boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \boldsymbol {s} _ {t, \hat {\boldsymbol {\theta}} _ {n} (\tau)} (\boldsymbol {x} (t))\right) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right] \right| =: J _ {1} + J _ {2} + J _ {3}, \\ \end{array}
$$

where $\widehat{\mathrm{Rad}}_n$ denotes the (empirical) Rademacher complexity of $\mathcal{F}_t$ on $\mathcal{D}_{\boldsymbol{x}} = \{\boldsymbol{x}_i\}_{i=1}^n$ , and all the inequalities hold in the element-wise sense.

(i) For $J_{1}$ , according to Lemma A.6 in [61], we have

$$
\begin{array}{l} \widehat {\operatorname{Rad}} _ {n} (\mathcal {F} _ {t}) \leq \left(\sup _ {\boldsymbol {f} _ {1} \in \mathcal {F} _ {1, t}, \boldsymbol {x} (t) \in [ - C _ {T, \delta}, C _ {T, \delta} ] ^ {d}} | \boldsymbol {f} _ {1} (\boldsymbol {x} (t)) | + \sup _ {f _ {2} \in \mathcal {F} _ {2, t}, \boldsymbol {x} (t) \in [ - C _ {T, \delta}, C _ {T, \delta} ] ^ {d}} | f _ {2} (\boldsymbol {x} (t)) |\right) \\ \cdot \left(\widehat {\operatorname{Rad}} _ {n} \left(\mathcal {F} _ {1, t}\right) + \widehat {\operatorname{Rad}} _ {n} \left(\mathcal {F} _ {2, t}\right)\right). \\ \end{array}
$$

Note that

$$
\begin{array}{l} \left| \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right| \leq \left| \boldsymbol {w} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t) \right| \\ \lesssim \| \boldsymbol {w} \| _ {1} \| \boldsymbol {x} (t) \| _ {\infty} + \| \boldsymbol {u} \| _ {1} \| \boldsymbol {e} (t) \| _ {\infty} \\ \leq C _ {T, \delta} + C _ {T, e}, \tag {36} \\ \end{array}
$$

which yields

$$
\begin{array}{l} \left| \boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) \right| = \left| \frac {1}{\sqrt {m}} \sum_ {i = 1} ^ {m} \boldsymbol {\theta} _ {i} (\tau) \sigma \left(\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)\right) \right| \\ \leq \frac {1}{\sqrt {m}} \sum_ {i = 1} ^ {m} | \boldsymbol {\theta} _ {i} (\tau) | \left| \sigma \left(\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)\right) \right| \\ \leq \frac {1}{\sqrt {m}} \left(C _ {T, \delta} + C _ {T, e}\right) \sum_ {i = 1} ^ {m} | \boldsymbol {\theta} _ {i} (\tau) | \\ \leq \left(C _ {T, \delta} + C _ {T, e}\right) \| \boldsymbol {\theta} (\tau) \| _ {2} \\ \lesssim \left(C _ {T, \delta} + C _ {T, e}\right) \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}\right), \tag {37} \\ \end{array}
$$

and hence

$$
\begin{array}{l} \left| \boldsymbol {f} _ {1} (\boldsymbol {x} (t)) \right| = \left| \boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t)) \right| \\ \lesssim \left(C _ {T, \delta} + C _ {T, e}\right) \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}\right) + C _ {T, \delta} ^ {\prime}, \\ \left| f _ {2} (\boldsymbol {x} (t)) \right| = \left| \sigma \left(\boldsymbol {w} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)\right) \right| \lesssim C _ {T, \delta} + C _ {T, e}, \\ \end{array}
$$

where $C_{T,\delta}^{\prime}:= \max_{\boldsymbol{x}(t)\in [-C_{T,\delta},C_{T,\delta}]^{d}}\left|\nabla_{\boldsymbol{x}(t)}\log p_{t}(\boldsymbol{x}(t))\right|$ . This gives

$$
\widehat {\mathrm{Rad}} _ {n} (\mathcal {F} _ {t}) \lesssim (C _ {T, \delta} + C _ {T, e}) \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m} + 1\right) \left(\widehat {\mathrm{Rad}} _ {n} (\mathcal {F} _ {1, t}) + \widehat {\mathrm{Rad}} _ {n} (\mathcal {F} _ {2, t})\right).
$$

Let

$$
\mathcal {F} _ {1, t} ^ {\prime} := \left\{\boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)): \| \boldsymbol {\theta} (\tau) \| _ {2} \lesssim \left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m} \right\},
$$

according to Lemma 26.6 in [43], we get $\widehat{\mathrm{Rad}}_n(\mathcal{F}_{1,t})\leq \widehat{\mathrm{Rad}}_n(\mathcal{F}_{1,t}')$ . Since

$$
\begin{array}{l} \widehat {n \mathrm{Rad}} _ {n} (\mathcal {F} _ {1, t} ^ {\prime}) = \mathbb {E} _ {\boldsymbol {\xi}} \left[ \sup _ {\| \boldsymbol {\theta} (\tau) \| _ {2} \lesssim \left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}} \sum_ {j = 1} ^ {n} \xi_ {j} \boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} _ {j} (t)) \right] \\ = \mathbb {E} _ {\boldsymbol {\xi}} \left[ \sup _ {\| \boldsymbol {\theta} (\tau) \| _ {2} \lesssim \left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}} \sum_ {j = 1} ^ {n} \xi_ {j} \frac {1}{\sqrt {m}} \sum_ {i = 1} ^ {m} \boldsymbol {\theta} _ {i} (\tau) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right] \\ \end{array}
$$

$$
\leq \frac{1}{\sqrt{m}}\mathbb{E}_{\boldsymbol {\xi}}\left[\sup_{\substack{\| \boldsymbol {\theta}(\tau)\|_{2}\lesssim \big\| \boldsymbol{s}_{0,\boldsymbol{\theta}_{0}}\big\|_{\mathcal{H}} + \sqrt{\tau / m}\\ \| \boldsymbol{w}_{i}\|_{1} + \| \boldsymbol{u}_{i}\|_{1}\leq 1,  i\in [n]}}\sum_{i = 1}^{m}\boldsymbol{\theta}_{i}(\tau)\sum_{j = 1}^{n}\xi_{j}\sigma (\boldsymbol{w}_{i}^{\top}\boldsymbol{x}_{j}(t) + \boldsymbol{u}_{i}^{\top}\boldsymbol {e}(t))\right]
$$

$$
\leq \frac {1}{\sqrt {m}} \mathbb {E} _ {\boldsymbol {\xi}} \left[ \sup _ {\| \boldsymbol {\theta} (\tau) \| _ {2} \lesssim \left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}} \sum_ {i = 1} ^ {m} | \boldsymbol {\theta} _ {i} (\tau) | \sup _ {\| \boldsymbol {w} _ {i} \| _ {1} + \| \boldsymbol {u} _ {i} \| _ {1} \leq 1, i \in [ n ]} \left| \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right| \right]
$$

$$
\leq \mathbb {E} _ {\boldsymbol {\xi}} \left[ \sup _ {\| \boldsymbol {\theta} (\tau) \| _ {2} \lesssim \left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}} \| \boldsymbol {\theta} (\tau) \| _ {2} \sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \left| \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right| \right]
$$

$$
\lesssim \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}\right) \mathbb {E} _ {\boldsymbol {\xi}} \left[ \sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \left| \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right| \right],
$$

where the $\{\xi_i\}_{i=1}^n$ are independent random variables with the distribution $\mathbb{P}(\xi_i = 1) = \mathbb{P}(\xi_i = -1) = 1/2$ , and obviously,

$$
\sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \geq \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\mathbf {0} _ {d} ^ {\top} \boldsymbol {x} _ {j} (t) + \mathbf {0} _ {d} ^ {\top} \boldsymbol {e} (t)) = 0,
$$

we have

$$
\sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \left| \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right|
$$

$$
\leq \max \left\{\sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)), \sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \sum_ {j = 1} ^ {n} (- \xi_ {j}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right\}
$$

$$
\leq \sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) + \sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \sum_ {j = 1} ^ {n} (- \xi_ {j}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)),
$$

and by symmetry

$$
\mathbb {E} _ {\boldsymbol {\xi}} \left[ \sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \left| \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right| \right]
$$

$$
\leq \mathbb {E} _ {\boldsymbol {\xi}} \left[ \sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right] + \mathbb {E} _ {\boldsymbol {\xi}} \left[ \sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \sum_ {j = 1} ^ {n} (- \xi_ {j}) \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right]
$$

$$
= 2 \mathbb {E} _ {\boldsymbol {\xi}} \left[ \sup _ {\| \boldsymbol {w} \| _ {1} + \| \boldsymbol {u} \| _ {1} \leq 1} \sum_ {j = 1} ^ {n} \xi_ {j} \sigma (\boldsymbol {w} ^ {\top} \boldsymbol {x} _ {j} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)) \right] = 2 n \widehat {\operatorname{Rad}} _ {n} (\mathcal {F} _ {2, t}),
$$

i.e., $\widehat{\mathrm{Rad}}_n(\mathcal{F}_{1,t})\leq \widehat{\mathrm{Rad}}_n(\mathcal{F}_{1,t}')\lesssim 2\left(\left\| s_{0,\theta_0}\right\|_\mathcal{H} + \sqrt{\tau / m}\right)\widehat{\mathrm{Rad}}_n(\mathcal{F}_{2,t})$ . According to Lemma 26.9

(contraction lemma) and Lemma 26.11 in [43], we have

$$
\widehat {\mathrm{Rad}} _ {n} (\mathcal {F} _ {2, t}) \leq (\| \boldsymbol {x} (t) \| _ {\infty} + \| \boldsymbol {e} (t) \| _ {\infty}) \sqrt {\frac {2 \log (4 d)}{n}} \lesssim (C _ {T, \delta} + C _ {T, \boldsymbol {e}}) \sqrt {\frac {\log (d + 1)}{n}}.
$$

Combining above, we obtain

$$
J _ {1} = \widehat {\operatorname{Rad}} _ {n} (\mathcal {F} _ {t}) \lesssim (C _ {T, \delta} + C _ {T, e}) ^ {2} \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m} + 1\right) ^ {2} \sqrt {\frac {\log (d + 1)}{n}}.
$$

(ii) For $J_{2}$ , by (36) and (37), we get

$$
\left| \boldsymbol {f} (\boldsymbol {x} (t)) \right| = \left| \boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \nabla_ {\boldsymbol {x} (t)} \log p _ {t} (\boldsymbol {x} (t)) \right| \left| \sigma \left(\boldsymbol {w} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} ^ {\top} \boldsymbol {e} (t)\right) \right|
$$

$$
\lesssim (C _ {T, \delta} + C _ {T, e}) ^ {2} \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}\right) + C _ {T, \delta} ^ {\prime} (C _ {T, \delta} + C _ {T, e}),
$$

which gives

$$
J _ {2} \lesssim (C _ {T, \delta} + C _ {T, e}) ^ {2} \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m} + 1\right) \sqrt {\frac {\log (1 / \delta)}{n}}.
$$

(iii) For $J_{3}$ , we similarly have

$$
\left| \pmb {s} _ {t, \hat {\pmb {\theta}} _ {n} (\tau)} (\pmb {x} (t)) \right| \lesssim (C _ {T, \delta} + C _ {T, \pmb {e}}) \left(\left\| \pmb {s} _ {0, \pmb {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}\right),
$$

hence by (36) and (37), we get

$$
J _ {3} \leq \mathbb {E} _ {\boldsymbol {x} (t) \sim p _ {t} ^ {(n)}} \left| \left(\boldsymbol {s} _ {t, \boldsymbol {\theta} (\tau)} (\boldsymbol {x} (t)) - \boldsymbol {s} _ {t, \hat {\boldsymbol {\theta}} _ {n} (\tau)} (\boldsymbol {x} (t))\right) \sigma (\boldsymbol {w} _ {i} ^ {\top} \boldsymbol {x} (t) + \boldsymbol {u} _ {i} ^ {\top} \boldsymbol {e} (t)) \right|
$$

$$
\lesssim (C _ {T, \delta} + C _ {T, e}) ^ {2} \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\tau / m}\right).
$$

Combining (i), (ii) and (iii), we obtain

$$
\frac {1}{m} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \lesssim \frac {1}{m ^ {2}} \sum_ {i = 1} ^ {m} \left\| \int_ {0} ^ {\tau} \mathbb {E} _ {t \sim \mathcal {U} (0, T)} [ \lambda (t) (J _ {1} + J _ {2} + J _ {3}) \mathbf {1} _ {d} ] d \tau_ {0} \right\| _ {2} ^ {2}
$$

$$
\lesssim \frac {1}{m ^ {2}} \sum_ {i = 1} ^ {m} \left\| \int_ {0} ^ {\tau} (J _ {1} + J _ {2} + J _ {3}) \mathbf {1} _ {d} d \tau_ {0} \right\| _ {2} ^ {2}
$$

$$
= (J _ {1} + J _ {2} + J _ {3}) ^ {2} \tau^ {2} \frac {d}{m}
$$

$$
\lesssim \tau^ {2} \frac {d}{m} \left(C _ {T, \delta} + C _ {T, \boldsymbol {e}}\right) ^ {4} \left[ \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \frac {\tau}{m} + 1\right) ^ {2} \right.
$$

$$
\cdot \left(\sqrt {\frac {\log (d + 1)}{n}} + \sqrt {\frac {\log (1 / \delta)}{n}}\right) ^ {2} + \left(\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \| _ {\mathcal {H}} ^ {2} + \frac {\tau}{m}\right) \Bigg ],
$$

which gives

$$
I _ {1} = \tilde {\mathcal {L}} (\hat {\boldsymbol {\theta}} _ {n} (\tau)) - \tilde {\mathcal {L}} (\boldsymbol {\theta} (\tau))
$$

$$
\lesssim (C _ {T, \delta} + C _ {T, e}) \left(\sqrt {\tilde {\mathcal {L}} (\boldsymbol {\theta} ^ {*})} + \frac {1}{\sqrt {\tau}} \left(\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \| _ {\mathcal {H}} + \| \boldsymbol {s} _ {0, \boldsymbol {\theta} ^ {*}} \| _ {\mathcal {H}}\right)\right) \left\{\frac {1}{m} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2} \right\} ^ {\frac {1}{2}}
$$

$$
+ \left(C _ {T, \delta} ^ {2} + C _ {T, e} ^ {2}\right) \frac {1}{m} \sum_ {i = 1} ^ {m} \| \hat {\boldsymbol {a}} _ {i} (\tau) - \boldsymbol {a} _ {i} (\tau) \| _ {2} ^ {2}
$$

$$
\lesssim \left(C _ {T, \delta} + C _ {T, \boldsymbol {e}}\right) ^ {3} \left(\sqrt {\tilde {\mathcal {L}} (\boldsymbol {\theta} ^ {*})} + \frac {1}{\sqrt {\tau}} \left(\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \| _ {\mathcal {H}} + \| \boldsymbol {s} _ {0, \boldsymbol {\theta} ^ {*}} \| _ {\mathcal {H}}\right)\right)
$$

$$
\cdot \tau \sqrt {\frac {d}{m}} \left[ \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \tau + 1\right) \left(\sqrt {\frac {\log (d + 1)}{n}} + \sqrt {\frac {\log (1 / \delta)}{n}}\right) + \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} + \sqrt {\frac {\tau}{m}}\right) \right]
$$

$$
+ \tau^ {2} \frac {d}{m} \left(C _ {T, \delta} + C _ {T, \boldsymbol {e}}\right) ^ {6} \left[ \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \tau + 1\right) ^ {2} \left(\sqrt {\frac {\log (d + 1)}{n}} + \sqrt {\frac {\log (1 / \delta)}{n}}\right) ^ {2} + \left(\left\| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \frac {\tau}{m}\right) \right]
$$

$$
\lesssim (C _ {T, \delta} + C _ {T, \boldsymbol {e}}) ^ {6} \tau \sqrt {\frac {d}{m}} \left[ (\tau + 1) \left(\sqrt {\frac {\log (d + 1)}{n}} + \sqrt {\frac {\log (1 / \delta)}{n}}\right) + \left(\frac {\| \boldsymbol {A} _ {0} \| _ {F}}{\sqrt {m}} + \sqrt {\frac {\tau}{m}}\right) \right]
$$

$$
\cdot \left\{\left(\sqrt {\tilde {\mathcal {L}} (\boldsymbol {\theta} ^ {*})} + \frac {1}{\sqrt {m \tau}} \left(\| \boldsymbol {A} _ {0} \| _ {F} + \| \boldsymbol {A} _ {*} \| _ {F}\right)\right) \right.
$$

$$
\left. + \tau \sqrt {\frac {d}{m}} \left[ (\tau + 1) \left(\sqrt {\frac {\log (d + 1)}{n}} + \sqrt {\frac {\log (1 / \delta)}{n}}\right) + \left(\frac {\| \boldsymbol {A} _ {0} \| _ {F}}{\sqrt {m}} + \sqrt {\frac {\tau}{m}}\right) \right] \right\} \tag {38}
$$

$$
\lesssim \log^ {3} (1 / \delta^ {2}) \tau \sqrt {\frac {d}{m}} \left[ (\tau + 1) \left(\sqrt {\frac {\log (d + 1)}{n}} + \sqrt {\frac {\log (1 / \delta)}{n}}\right) + \left(\frac {1}{\sqrt {m}} + \sqrt {\frac {\tau}{m}}\right) \right]
$$

$$
\cdot \left\{\left(\sqrt {\tilde {\mathcal {L}} (\boldsymbol {\theta} ^ {*})} + \frac {1}{\sqrt {m \tau}}\right) + \tau \sqrt {\frac {d}{m}} \left[ (\tau + 1) \left(\sqrt {\frac {\log (d + 1)}{n}} + \sqrt {\frac {\log (1 / \delta)}{n}}\right) + \left(\frac {1}{\sqrt {m}} + \sqrt {\frac {\tau}{m}}\right) \right] \right\}.
$$

where $\lesssim$ hides universal positive constants only depending on $T$ .

Combining all above estimates yields

$$
D _ {\mathrm{KL}} \left(p _ {0} \| p _ {0, \hat {\boldsymbol {\theta}} _ {n} (\tau)}\right)
$$

$$
\lesssim I _ {1} + I _ {2} + I _ {3} + D _ {\mathrm{KL}} \left(p _ {T} \| \pi\right)
$$

$$
\lesssim \log^ {3} (1 / \delta^ {2}) \tau \sqrt {\frac {d}{m}} \left[ (\tau + 1) \left(\sqrt {\frac {\log (d + 1)}{n}} + \sqrt {\frac {\log (1 / \delta)}{n}}\right) + \frac {1}{\sqrt {m}} (1 + \sqrt {\tau}) \right]
$$

$$
\cdot \left\{\left(\sqrt {\tilde {\mathcal {L}} (\boldsymbol {\theta} ^ {*})} + \frac {1}{\sqrt {m \tau}}\right) + \tau \sqrt {\frac {d}{m}} \left[ (\tau + 1) \left(\sqrt {\frac {\log (d + 1)}{n}} + \sqrt {\frac {\log (1 / \delta)}{n}}\right) + \frac {1}{\sqrt {m}} (1 + \sqrt {\tau}) \right] \right\}
$$

$$
+ \left(\tilde {\mathcal {L}} \left(\bar {\boldsymbol {\theta}} ^ {*}\right) + \tilde {\mathcal {L}} \left(\boldsymbol {\theta} ^ {*}\right)\right) + \frac {\log^ {2} (1 / \delta^ {2})}{m} d + \frac {1}{\tau} \left(\left\| \bar {\boldsymbol {s}} _ {0, \bar {\boldsymbol {\theta}} _ {0}} \right\| _ {\mathcal {H}} ^ {2} + \left\| \bar {\boldsymbol {s}} _ {0, \bar {\boldsymbol {\theta}} ^ {*}} \right\| _ {\mathcal {H}} ^ {2} + \| \boldsymbol {s} _ {0, \boldsymbol {\theta} _ {0}} \| _ {\mathcal {H}} ^ {2} + \| \boldsymbol {s} _ {0, \boldsymbol {\theta} ^ {*}} \| _ {\mathcal {H}} ^ {2}\right) + D _ {\mathrm{KL}} \left(p _ {T} \| \pi\right).
$$

If one only focuses on the $(m,n)$ -dependence, i.e. the dependence on the model capacity and

sample size, this upper bound can be further simplified as

$$
\begin{array}{l} D _ {\mathrm{KL}} \left(p _ {0} \| p _ {0, \hat {\boldsymbol {\theta}} _ {n} (\tau)}\right) \\ \stackrel {\mathrm{(vi)}} {\lesssim} \left(\frac {\tau^ {2}}{\sqrt {m n}} + \frac {\tau \sqrt {\tau}}{m}\right) \cdot \left(\sqrt {\tilde {\mathcal {L}} (\pmb {\theta} ^ {*})} + \frac {1}{\sqrt {m \tau}} + \frac {\tau^ {2}}{\sqrt {m n}} + \frac {\tau \sqrt {\tau}}{m}\right) \\ + \left(\bar {\tilde {\mathcal {L}}} (\bar {\boldsymbol {\theta}} ^ {*}) + \tilde {\mathcal {L}} (\boldsymbol {\theta} ^ {*})\right) + \frac {1}{m} + \frac {1}{\tau} + D _ {\mathrm{KL}} (p _ {T} \| \pi) \\ \leq \left(\sqrt {\tilde {\mathcal {L}} (\pmb {\theta} ^ {*})} + \frac {1}{\sqrt {m \tau}} + \frac {\tau^ {2}}{\sqrt {m n}} + \frac {\tau \sqrt {\tau}}{m}\right) ^ {2} + \left(\bar {\bar {\mathcal {L}}} \left(\bar {\pmb {\theta}} ^ {*}\right) + \tilde {\mathcal {L}} \left(\pmb {\theta} ^ {*}\right)\right) + \frac {1}{m} + \frac {1}{\tau} + D _ {\mathrm{KL}} (p _ {T} \| \pi) \\ \lesssim \left(\tilde {\mathcal {L}} (\pmb {\theta} ^ {*}) + \frac {1}{m \tau} + \frac {\tau^ {4}}{m n} + \frac {\tau^ {3}}{m ^ {2}}\right) + \left(\bar {\tilde {\mathcal {L}}} \left(\bar {\pmb {\theta}} ^ {*}\right) + \tilde {\mathcal {L}} (\pmb {\theta} ^ {*})\right) + \frac {1}{m} + \frac {1}{\tau} + D _ {\mathrm{KL}} (p _ {T} \| \pi) \\ \lesssim \frac {\tau^ {4}}{m n} + \frac {\tau^ {3}}{m ^ {2}} + \frac {1}{\tau} + \frac {1}{m} + \left(\tilde {\mathcal {L}} (\bar {\boldsymbol {\theta}} ^ {*}) + \tilde {\mathcal {L}} (\boldsymbol {\theta} ^ {*})\right) + D _ {\mathrm{KL}} (p _ {T} \| \pi), \\ \end{array}
$$

where we assume $\tau \geq 1$ for simplicity, and $\stackrel{\mathrm{(vi)}}{\lesssim}$ hides the term $d\log (d + 1)$ , the polynomials of $\log (1 / \delta^2)$ and finite RKHS norms $\| \cdot \|_{\mathcal{H}}$ . The proof is completed.

Approximation errors. For the (universal) approximation, we discuss in two points:

- First, the approximation is a separate problem that can be analyzed independently of training and generalization. While it is beyond the scope of current work, we have included the approximation error in the final estimates. When the approximation by random feature models fails, the generalization error is supposed to be significant.   
- In addition, the random feature model can approximate Lipschitz continuous functions on a compact domain (Theorem 6 in [17]). Notice that the forward diffusion process defines a random path $(\pmb{x}(t), t)_{t \in [0,T]}$ contained in a rectangular domain $R_{T,\delta} := [-C_{T,\delta}, C_{T,\delta}]^d \times [0,T] \subset \mathbb{R}^{d+1}$ with $C_{T,\delta} := C_T(C_x + \sqrt{\log(1/\delta^2)})$ (use Lemma 1 and the boundness of inputs), one can apply Theorem 6 in [17] to bound (7) on the domain $R_{T,\delta}$ in $\mathbb{R}^{d+1}$ to obtain approximation results for Lipschitz continuous target score functions.

# A.2 Data-Dependent Generalization Gap

Lemma 7 (Forward perturbation estimates, Gaussian mixtures). Suppose that x is sampled from a one-dimensional 2-mode Gaussian mixture: $p_{0}(x) = q_{1}\mathcal{N}(x; -\mu, 1) + q_{2}\mathcal{N}(x; \mu, 1)$ , where $\mu > 0$ , $q_{1}$ , $q_{2} > 0$ with $q_{1} + q_{2} = 1$ are all constants. Then for any $\delta > 0$ , $\delta \ll 1$ , $\mu > \sqrt{\log(1/\delta^{2})}$ , with the probability of at least $1 - \delta$ , we have

$$
| x (t) | \lesssim C _ {T} \left(\mu + \sqrt {\log (1 / \delta^ {2})}\right) \triangleq C _ {T, \mu , \delta}. \tag {39}
$$

Proof. It is straightforward to verify that

$$
\begin{array}{l} \mathbb {P} \left(\{x: | x - \mu | \leq \sqrt {\log (1 / \delta^ {2})} \} \cup \{x: | x + \mu | \leq \sqrt {\log (1 / \delta^ {2})} \}\right) \\ = \mathbb {P} \{x: | x - \mu | \leq \sqrt {\log (1 / \delta^ {2})} \} + \mathbb {P} \{x: | x + \mu | \leq \sqrt {\log (1 / \delta^ {2})} \} \\ = \int_ {\mu - \sqrt {\log (1 / \delta^ {2})}} ^ {\mu + \sqrt {\log (1 / \delta^ {2})}} p _ {0} (x) d x + \int_ {- \mu - \sqrt {\log (1 / \delta^ {2})}} ^ {- \mu + \sqrt {\log (1 / \delta^ {2})}} p _ {0} (x) d x \\ \geq q _ {2} \int_ {\mu - \sqrt {\log (1 / \delta^ {2})}} ^ {\mu + \sqrt {\log (1 / \delta^ {2})}} \mathcal {N} (x; \mu , 1) d x + q _ {1} \int_ {- \mu - \sqrt {\log (1 / \delta^ {2})}} ^ {- \mu + \sqrt {\log (1 / \delta^ {2})}} \mathcal {N} (x; - \mu , 1) d x \\ = q _ {2} \int_ {- \sqrt {\log (1 / \delta^ {2})}} ^ {\sqrt {\log (1 / \delta^ {2})}} \mathcal {N} (x; 0, 1) d x + q _ {1} \int_ {- \sqrt {\log (1 / \delta^ {2})}} ^ {\sqrt {\log (1 / \delta^ {2})}} \mathcal {N} (x; 0, 1) d x \\ = \left(q _ {1} + q _ {2}\right) \cdot \mathbb {P} \{\epsilon : | \epsilon | \leq \sqrt {\log (1 / \delta^ {2})} \} \geq 1 - \delta , \\ \end{array}
$$

where the last inequality applies (21). That is, for any $\delta > 0$ , $\delta \ll 1$ , with the probability of at least $1 - \delta$ , we have

$$
| x | \in [ \mu - \sqrt {\log (1 / \delta^ {2})}, \mu + \sqrt {\log (1 / \delta^ {2})} ] = \Theta (\mu). \tag {40}
$$

Hence, Lemma 1 gives

$$
| x (t) | \lesssim C _ {T} \left(\mu + \sqrt {\log (1 / \delta^ {2})}\right), \tag {41}
$$

which completes the proof.

![](images/431fd205bc8f6f13bb809fa63fc2e1a23fbe18291661ccd7f421deb5fdbe2dd6.jpg)

Lemma 8 (Monte Carlo estimates, Gaussian mixtures). Define the Monte Carlo error

$$
\operatorname{Err} _ {\mathrm{MC}} = \operatorname{Err} _ {\mathrm{MC}} (\boldsymbol {\theta}, \bar {\boldsymbol {\theta}}; T, \lambda (\cdot)) := \mathbb {E} _ {t \sim \mathcal {U} (0, T)} \left[ \lambda (t) \cdot \mathbb {E} _ {x (t) \sim p _ {t}} \left[ \| s _ {t, \boldsymbol {\theta}} (x (t)) - \bar {s} _ {t, \bar {\boldsymbol {\theta}}} (x (t)) \| _ {2} ^ {2} \right] \right]. \tag {42}
$$

Suppose that the trainable parameter $\mathbf{a}$ and embedding function $\mathbf{e}(\cdot)$ are both bounded, and $x$ is sampled from a one-dimensional 2-mode Gaussian mixture: $p_0(x) = q_1\mathcal{N}(x; - \mu ,1) + q_2\mathcal{N}(x;\mu ,1)$ , where $\mu >0$ , $q_{1},q_{2} > 0$ with $q_{1} + q_{2} = 1$ are all constants. Then, given any $\bar{\theta}$ , for any $\delta >0$ , $\delta \ll 1$ , $\mu >\sqrt{\log(1 / \delta^2)}$ , with the probability of at least $1 - \delta$ , there exists $\theta$ such that

$$
\operatorname{Err} _ {\mathrm{MC}} \lesssim \mu^ {2} \frac {\log (1 / \delta)}{m}, \tag {43}
$$

where $\lesssim$ hides universal positive constants only depending on $T$ .

Proof. According to Lemma 7, we just need to follow the proof of Lemma 6 by replacing $C_{T,\delta}$ by $C_{T,\mu,\delta}$ . Notably, based on (33) in the proof of Lemma 6, one can finally derive

$$
\mathrm{Err} _ {\mathrm{MC}} \lesssim \mu^ {2} \frac {\log (1 / \delta)}{m},
$$

which gives the desired estimates.

![](images/6d1bb7c742bf3715e65c2616df63f8eea73598cd4d05b77f2c4d50dc84b7ea16.jpg)

Proof of Theorem 2. We decompose the loss $\tilde{\mathcal{L}}(\hat{\boldsymbol{\theta}}_n(\tau))$ in the same way as in the proof of Theorem 1. In fact, Theorem 2 is similarly proved by replacing $C_{T,\delta}$ in the proof of Theorem 1 by $C_{T,\mu,\delta}$ . Note that $C_{T,\mu,\delta} = C_T\left(\mu + \sqrt{\log(1/\delta^2)}\right) \lesssim \mu$ for $\mu \gg 1$ , the proof is completed.

Remark 6. A standard variance is used here for convenience. In general, if $var = o(\mu)$ as $\mu \to +\infty$ (e.g. a bounded var), similar analysis and results are supposed to hold. However, this is different when $var = \Theta(\mu)$ as $\mu \to +\infty$ , since the modes are not separated in this case, and we can not characterize modes shift by simply varying $\mu$ .

Remark 7. Simply scaling down inputs seems not to resolve the adverse effect of modes shift, since the ground truth $\mu$ is unknown. One can use the input scale to approximate $\mu$ on toy datasets, but it is not that trivial in practice, particularly for real-world applications with multiple high-dimensional modes in varied scales.

The model-target inconsistency. Informally, there is inconsistency between the score network model and target score function. In fact, given the target Gaussian mixture $p_{0}(x) = q_{1}\mathcal{N}(x; -\mu, 1) + q_{2}\mathcal{N}(x; \mu, 1)$ , the target score function is

$$
s _ {0} (x) = - x + \frac {q _ {2} \mathcal {N} (x ; \mu , 1) - q _ {1} \mathcal {N} (x ; - \mu , 1)}{q _ {2} \mathcal {N} (x ; \mu , 1) + q _ {1} \mathcal {N} (x ; - \mu , 1)} \mu ,
$$

which gives $s_{0}(x) \approx -x + \mu$ , $x \geq 0$ and $s_{0}(x) \approx -x - \mu$ , $x \leq 0$ . While the score network model is

$$
s _ {0, \theta} (x) \approx \left(\frac {1}{m} \sum_ {i: w _ {i} > 0} a _ {i} w _ {i}\right) x + \left(\frac {1}{m} \sum_ {i: w _ {i} > 0} a _ {i} \pmb {u} _ {i} ^ {\top} \pmb {e} (0)\right),
$$

where $\approx$ holds since $|x| = \Theta (\mu)$ by (40) $(\mu \gg 1)$ . Both $s_0$ and $s_{0,\theta}$ are linear functions, but they have unmatched scales in slopes and intercepts: $O(1)$ and $O(\mu)$ for $s_0$ , but both $O(a)$ 's for $s_{0,\theta}$ . That is, modeling Gaussian mixtures with large modes distances as random feature models is inconsistent.

# B Additional Experiments

# B.1 Early-Stopping Generalization

We illustrate the early-stopping generalization gap established in Theorem 1 using the Adam optimizer. All the configurations remain the same as Section 4.1 except that the learning rate is now $10^{-3}$ .

From Figure 8, one can observe that the KL divergence achieves its minimum at the 1000th training epoch, and it starts to increase after this turning point. The plot is aligned with Theorem 1, which states that there exists an optimal early-stopping time when the model can generalize well, indicating the effectiveness of the upper bound. Further, the KL divergence begins to oscillate after the minimum point (1000th training epoch), which may suggest a phase transition in the KL divergence dynamics, and the transition point is around the optimal early-stopping time. The finding is aligned with SGD setting.

![](images/83a47492dbaa4400320a27b91a4631b4486cebd7f14ef485a2821c1d74aaad31.jpg)

<details>
<summary>line</summary>

| epoch | KL divergence |
| ----- | ------------- |
| 0     | 0.69          |
| 250   | 0.23          |
| 500   | 0.18          |
| 750   | 0.14          |
| 1000  | 0.05          |
| 1250  | 0.39          |
| 1500  | 0.25          |
| 1750  | 0.33          |
| 2000  | 0.85          |
</details>

Figure 8: The KL divergence dynamics under the Adam optimizer.

# B.2 Modes Shift Effect

We further test the relationship between the modes' distance and the generalization performance using the Adam optimizer under the same configurations.

In Figure 9 and Figure 10, it is shown that the training of modeled distributions exhibits the same two-stage dynamics as the SGD setting, indicating that the modes shift effect holds not particularly for a certain optimizer.

![](images/dccb116028be3f654d06731bf3411cf862a10b846b08bbbe0e4c518c0d21ef3c.jpg)

<details>
<summary>line</summary>

| x     | Target | Trained |
|-------|--------|---------|
| -10.0 | 0.00   | 0.00    |
| -7.5  | 0.00   | 0.00    |
| -5.0  | 0.08   | 0.06    |
| -2.5  | 0.16   | 0.12    |
| 0.0   | 0.02   | 0.10    |
| 2.5   | 0.14   | 0.12    |
| 5.0   | 0.08   | 0.06    |
| 7.5   | 0.00   | 0.00    |
| 10.0  | 0.00   | 0.00    |
</details>

![](images/26005c821105c8fe3cd65dd0ea8b11f318a2c83cbaa5664941678236b8b1edfe.jpg)

<details>
<summary>line</summary>

| x     | Target | Trained |
|-------|--------|---------|
| -10.0 | 0.00   | 0.00    |
| -7.5  | 0.00   | 0.00    |
| -5.0  | 0.16   | 0.16    |
| -2.5  | 0.16   | 0.16    |
| 0.0   | 0.02   | 0.04    |
| 2.5   | 0.14   | 0.15    |
| 5.0   | 0.14   | 0.14    |
| 7.5   | 0.00   | 0.00    |
| 10.0  | 0.00   | 0.00    |
</details>

![](images/186f6c3e04cb51f0e9393af2bdac553dc8ceabaa0fd3bdabdc18ea63e0b7c689.jpg)

<details>
<summary>line</summary>

| x     | Target | Trained |
|-------|--------|---------|
| -10.0 | 0.00   | 0.00    |
| -7.5  | 0.00   | 0.00    |
| -5.0  | 0.15   | 0.30    |
| -2.5  | 0.16   | 0.15    |
| 0.0   | 0.02   | 0.03    |
| 2.5   | 0.15   | 0.06    |
| 5.0   | 0.14   | 0.05    |
| 7.5   | 0.01   | 0.01    |
| 10.0  | 0.00   | 0.00    |
</details>

Figure 9: The Adam training dynamics when the distance between two modes is 6 ( $\mu = 3$ ).

![](images/f554ba635a69e8555701d61c153efbed2f294c6a11b8b85cf58661a04fb235cd.jpg)

<details>
<summary>line</summary>

| x    | Target | Trained |
| ---- | ------ | ------- |
| -30  | 0.00   | 0.00    |
| -20  | 0.05   | 0.30    |
| -10  | 0.05   | 0.00    |
| 0    | 0.00   | 0.00    |
| 10   | 0.05   | 0.05    |
| 20   | 0.05   | 0.00    |
| 30   | 0.00   | 0.00    |
</details>

![](images/35338270e9317b3dc1750628561556bc6a107153b205628a7ba88c6b43a59b79.jpg)

<details>
<summary>line</summary>

| x    | Target | Trained |
| ---- | ------ | ------- |
| -30  | 0.00   | 0.00    |
| -20  | 0.05   | 0.00    |
| -10  | 0.05   | 0.05    |
| 0    | 0.00   | 0.00    |
| 10   | 0.05   | 0.35    |
| 20   | 0.05   | 0.00    |
| 30   | 0.00   | 0.00    |
</details>

![](images/488fcae70695f47702401660cc9726753eefa57a8fd500b71dfd6c3c5c3d7b63.jpg)

<details>
<summary>line</summary>

| x    | Target | Trained |
| ---- | ------ | ------- |
| -30  | 0.00   | 0.00    |
| -20  | 0.05   | 0.00    |
| -10  | 0.05   | 0.00    |
| 0    | 0.00   | 0.00    |
| 10   | 0.05   | 0.35    |
| 20   | 0.05   | 0.35    |
| 30   | 0.00   | 0.00    |
</details>

Figure 10: The Adam training dynamics when the distance between two modes is 30 ( $\mu = 15$ ).

# B.3 Model Capacity Dependency

We also numerically study the dependency of generalization on the model capacity. Following the same configurations in Section 4.1.1, we conduct experiments for different hidden dimensions varying from $2^{1}$ to $2^{11}$ .

In Figure 11, the left plot shows the KL divergence from the trained distribution at the 1000th training epoch to the target distribution, and the right plot shows the time duration of the modeled distribution to generalize. Here, the generalization criterion we select is $D_{KL} \leq 10^{-1}$ , and we stop training at epoch = 10000. Both the two plots in Figure 11 indicate that increasing model capacity benefits the generalization, which also verifies the corresponding theoretical results (m-dependency in Theorem 1 and Theorem 2).

![](images/8ff5bad82343903f765d42d5e9579c6dbc5893101c84f278340344a2fb8bf941.jpg)

<details>
<summary>line</summary>

| m     | KL divergence |
|-------|---------------|
| 2^2   | 1.3           |
| 2^4   | 0.45          |
| 2^6   | 0.3           |
| 2^8   | 0.2           |
| 2^10  | 0.1           |
</details>

![](images/62e0906f2207553f70a1595c35c3c4db1efb7db8c1ef79e71d45d035ec8c30fb.jpg)

<details>
<summary>line</summary>

| m     | Training epoch |
|-------|----------------|
| 2^2   | 10000          |
| 2^4   | 4500           |
| 2^6   | 2500           |
| 2^8   | 1500           |
| 2^10  | 1000           |
</details>

Figure 11: Left: The KL divergence from the model trained after 1000 epochs to the target distribution under different hidden dimensions. Right: The earliest training epoch when $D_{KL} \leq 10^{-1}$ . Here, “×” means that the model does not generalize when the training is stopped at the maximum training epoch 10000.