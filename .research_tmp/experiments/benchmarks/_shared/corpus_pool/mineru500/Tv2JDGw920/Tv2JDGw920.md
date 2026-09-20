# One-Step Generalization Ratio Guided Optimization for Domain Generalization

Sumin Cho $^{*1}$ Dongwon Kim $^{*1}$ Kwangsu Kim $^{1}$

# Abstract

Domain Generalization (DG) aims to train models that generalize to unseen target domains but often overfit to domain-specific features, known as undesired correlations. Gradient-based DG methods typically guide gradients in a dominant direction but often inadvertently reinforce spurious correlations. Recent work has employed dropout to regularize overconfident parameters, but has not explicitly adjusted gradient alignment or ensured balanced parameter updates. We propose GENIE (Generalization-ENhancing Iterative Equalizer), a novel optimizer that leverages the One-Step Generalization Ratio (OSGR) to quantify each parameter's contribution to loss reduction and assess gradient alignment. By dynamically equalizing OSGR via a preconditioning factor, GENIE prevents a small subset of parameters from dominating optimization, thereby promoting domain-invariant feature learning. Theoretically, GENIE balances convergence contribution and gradient alignment among parameters, achieving higher OSGR while retaining SGD's convergence rate. Empirically, it outperforms existing optimizers and enhances performance when integrated with various DG and single-DG methods.

# 1. Introduction

Deep neural networks (DNNs) achieve high accuracy when training and test data share a similar distribution. However, in real-world applications, data distributions often shift, causing performance degradation(Muandet et al., 2013). Domain Generalization (DG) addresses this issue by training models to generalize to out-of-distribution data from unseen domains. The main challenge is to prevent overfitting

\*Equal contribution $^{1}$ Department of Computer Science and Engineering, University of Sungkyunkwan, Suwon, Korea. Correspondence to: Sumin Cho <jsm0707@skku.edu>, Dongwon Kim <kdwaha@skku.edu>, Kwangsu Kim <kim.kwangsu@skku.edu>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](images/10564f17d998bee31f1ffee573d1ab05fdea0fa1a089e9ceac2a64fceba79cbf.jpg)

<details>
<summary>heatmap</summary>

| Gene  | Caltech101 Parameter Index | VOC2007 Parameter Index |
|-------|-----------------------------|--------------------------|
| SGD   | 0.0                         | 0.0                      |
| Adam  | 0.0                         | 0.0                      |
| SAM   | 0.0                         | 0.0                      |
| GENIE | 0.0                         | 0.0                      |
</details>

Figure 1. Heatmaps visualizing normalized parameter update magnitudes by parameter ID for different optimizers throughout training on the VLCS dataset in the DG. Previous optimizers (SGD, Adam, SAM) exhibit an imbalanced parameter update distribution, where a subset of parameters dominates the optimization process. In contrast, GENIE uniformly adjusts parameter-wise OSGR, mitigating overfitting to specific parameters and promoting a more balanced optimization across the entire parameter space.

to domain-specific features—known as spurious correlations—while learning invariant features and causal relationships that generalize across diverse domains(Shi et al., 2022; Hemati et al., 2023; Shah et al., 2020a; Ye et al., 2024).

Several DG methods have attempted to guide the gradient toward a dominant direction during training (Parascandolo et al., 2021; Shahtalebi et al., 2021; Shi et al., 2022; Rame et al., 2022). However, this dominant direction often itself driven by spurious features, inadvertently reinforcing undesired correlations. This suggests that aligning gradients toward a single dominant direction is insufficient to fully solve the problem, highlighting the need for other perspectives.

A recent approach (Michalkiewicz et al., 2023) introduced a parameter-wise dropout mechanism based on Gradient Signal-to-Noise Ratios (GSNR) to suppress overly predictive parameters and reduce their influence on optimization. While this strategy mitigates parameter updates driven by spurious correlations, it does not adjust the magnitudes of updates based on their individual contributions to general-

ization. This raises the open question of how to design optimizers that explicitly balance parameter updates according to their principled contributions to generalization, thereby mitigating the influence of spurious correlations.

Motivated by this perspective, we propose Generalization-ENhancing Iterative Equalizer (GENIE), a novel optimizer for addressing parameter imbalance. Recent work (Liu et al., 2020) introduced the One-Step Generalization Ratio (OSGR) that measures how effectively a single gradient update reduces test loss compared to training loss, providing insight into a model's generalization potential. OSGR reflects the contributions of individual parameters to generalization, based on their convergence speed and degree of gradient alignment. To leverage this insight, GENIE integrates a preconditioning factor that dynamically balances parameter-wise OSGR throughout training. This prevents a small subset of parameters from dominating the optimization, thereby promoting more robust and domain-invariant feature learning.

Our theoretical analysis shows that existing optimizers typically focus on either convergence speed or gradient alignment, often resulting in suboptimal generalization. In contrast, GENIE explicitly balances both, achieving a higher OSGR while maintaining the convergence rate of SGD(Robbins & Monro, 1951) in non-convex settings. We empirically validated GENIE on five standard DG datasets(Li et al., 2017; Fang et al., 2013; Venkateswara et al., 2017; Beery et al., 2018; Peng et al., 2019) where it consistently outperformed established optimizers, even with extended iterations. Furthermore, using our optimizer in existing DG and Single-DG (SDG) algorithms enhances their performance. We summarize our contributions as follows:

- We propose GENIE, a novel optimizer that addresses the overlooked issue of parameter imbalance in DG. It suppresses over-predictive parameters while promoting balanced parameter updates.   
- We incorporate OSGR, previously used as a generalization metric, into the optimizer's core principle. This provides an efficient and novel perspective on generalization for addressing DG.   
- GENIE is a domain-agnostic optimizer. It is validated across multiple DG benchmarks and SDG tasks, demonstrating its broad applicability and scalability.

# 2. Related Work

# 2.1. Domain Generalization

Existing DG methods address domain shift through two main strategies: (1) Feature Alignment, which aims to align features across domains to ensure consistent optimization, including methods such as domain-invariant feature learning (Sun & Saenko, 2016; Arjovsky et al., 2019; Krueger et al., 2021), data augmentation (Xu et al., 2020; Yan et al., 2020; Wang et al., 2020), and feature disentanglement (Nam et al., 2021; Mahajan et al., 2021). (2) Gradient Alignment, which focuses on aligning gradients across domains to ensure stable learning dynamics. Representative approaches include minimizing gradient differences (Koyama & Yamaguchi, 2020), increasing gradient inner products (Shi et al., 2022), updating weights only when gradient directions align (Parascandolo et al., 2021; Shahtalebi et al., 2021), and reducing inter-domain gradient variance (Rame et al., 2022). Recently, Sharpness Aware Minima (SAM)(Foret et al., 2021) has improved in-distribution generalization, inspiring the development of optimizers specifically designed for OOD tasks (Zhang et al., 2024; Wang et al., 2023). However, most DG studies overlook imbalanced parameter updates caused by differences in convergence speed or generalization capacity during optimization.

# 2.2. Preconditioning

Preconditioning improves the efficiency of optimization algorithms by incorporating curvature information of the loss function or adjusting the magnitude and direction of parameter updates. It accelerates convergence and enhances stability during training and can be categorized into three main types (Ye, 2024; Amari et al., 2021) (1) Hessian Based Preconditioning: utilizes the inverse or approximations of the Hessian matrix to capture curvature information. (Montavon et al., 2012; Dennis & Moré, 1977) (2) Adaptive Learning Rate Based Preconditioning: dynamically adjusts learning rates based on gradient magnitudes, as seen in optimizers like AdaGrad (Duchi et al., 2011), RMSProp (Hinton et al., 2012), and Adam (Kingma, 2014). (3) Normalization-Based Preconditioning: normalizes inputs and activations, as exemplified by Batch Normalization(Ioffe & Szegedy, 2015), to improve the Hessian's condition number and enhance training stability. Previous preconditioning methods aim to optimize speed and stability. The application of preconditioning to improve model generalization remains underexplored.

# 3. Method

# 3.1. Preliminary

To address the challenge of generalization in unseen target domains, a recent study(Liu et al., 2020) introduced the concept of OSGR $R(Z,n)$ . OSGR quantifies how well model updates contribute to generalization by measuring the ratio of loss reduction between test $D'$ and training data $D$ after a single optimization step:

$$
R (Z, n) = \frac {\mathbb {E} _ {D , D ^ {\prime} \sim \mathcal {Z} ^ {n}} \Delta L _ {D ^ {\prime}}}{\mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} \Delta L _ {D}}, \tag {1}
$$

where $\Delta L_{D'}$ and $\Delta L_{D}$ represent the loss changes on test and training data, respectively. OSGR is influenced by two key factors: (1) the contribution of each parameter to loss reduction, characterized by the gradient magnitude, and (2) the alignment of parameter gradients across the data distribution. Higher OSGR indicates better generalization, reflecting consistent and balanced parameter updates.

To better understand these dynamics, the following theorem links OSGR to parameter-wise statistics:

Theorem 3.1 (From Paper(Liu et al., 2020)). The relationship between gradient updates and generalization can be expressed as follows:

$$
R (Z, n) = 1 - \frac {1}{n} \sum_ {j \in J} \frac {\mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} [ g _ {j} ^ {2} ]}{\sum_ {j ^ {\prime} \in J} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} [ g _ {j ^ {\prime}} ^ {2} ]} \cdot \frac {1}{r _ {j} + \frac {1}{n}}, \tag {2}
$$

where J denotes the set of parameter index, $g_{j}^{2}$ is the squared gradient magnitude, $\rho_{j}^{2}$ is the noise variance, and n is the number of samples. Parameters with higher Gradient Signal-to-Noise Ratios (GSNR), defined as $r_{j} = \frac{g_{j}^{2}}{\rho_{j}^{2}}$ , yield higher OSGR, contributing more significantly to generalization.

A recent study (Michalkiewicz et al., 2023) leveraged GSNR to suppress overly predictive parameters during training, aiming to prioritize robust features and reduce noisy updates. However, this approach overlooks parameter-wise imbalances in OSGR, which limits overall generalization performance.

In this context, we propose a preconditioning-based approach that dynamically balances OSGR across parameters. By incorporating parameter-specific preconditioning factors, our method ensures that updates are aligned with both gradient magnitude and noise characteristics, preventing overfitting to noisy or well-learned features. This strategy not only enhances generalization but also ensures stable convergence in diverse DG settings.

# 3.2. Proposed Method

Based on Theorem 3.1, Michalkiewicz et al. (2023) introduced a gradient-masking approach that prioritizes updates for parameters with low GSNR, aiming to enhance their contribution to generalization. They argue that boosting updates to low-GSNR parameters can increase the overall GSNR and thus improve the optimization signal-to-gradient ratio (OSGR). Inspired by this perspective, we hypothesize the following relationship:

Conjecture Uniformly distributed OSGR across parameters indicate better generalization performance.

This conjecture guides the design of our method. Rather than modifying the dropout ratio across parameters, we introduce a preconditioning term that more accurately adjusts the OSGR. Next, we inject noise into all parameters to encourage exploration toward better optima. Finally, we apply random dropout to stabilize parameter updates and reduce overfitting.

# 3.2.1. PRECONDITIONING

We propose a preconditioning factor $p_{j}$ to ensure balanced contributions of each parameter to the OSGR, thus enhancing generalization. The key idea is to maintain equitable parameter influence on the overall generalization performance throughout the optimization process. We propose the following corollary for this purpose.

Corollary 3.2 (Preconditioning and OSGR). If each parameter j applies a preconditioner $p_{j}$ , the OSGR can be expressed as:

$$
R ^ {\prime} (Z, n) = \sum_ {j \in J} \frac {p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} [ g _ {j} ^ {2} ]}{\sum_ {j ^ {\prime} \in J} p _ {j ^ {\prime}} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} [ g _ {j ^ {\prime}} ^ {2} ]} \cdot \frac {1}{\frac {1}{n \cdot r _ {j}} + 1}, \tag {3}
$$

or equivalently:

$$
R ^ {\prime} (Z, n) = 1 - \frac {1}{n} \sum_ {j i n J} \frac {p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} [ g _ {j} ^ {2} ]}{\sum_ {j ^ {\prime} \in J} p _ {j ^ {\prime}} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} [ g _ {j ^ {\prime}} ^ {2} ]} \cdot \frac {1}{r _ {j} + \frac {1}{n}}. \tag {4}
$$

From Theorem 3.2, to maintain a balanced influence of parameter $j$ on the overall OSGR, we propose:

$$
p _ {j} = \frac {1}{\mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} \left[ g _ {j} ^ {2} \right]} \left(r _ {j} + \frac {1}{n}\right). \tag {5}
$$

This leads to the OSGR:

$$
R ^ {\prime} (Z, n) = 1 - \frac {1}{n} \sum_ {j \in J} \frac {1}{\sum_ {j ^ {\prime} \in J} \left(r _ {j ^ {\prime}} + \frac {1}{n}\right)} = 1 - \frac {1}{n \mathbb {E} _ {j \in J} \left(r _ {j} + \frac {1}{n}\right)}, \tag {6}
$$

where $\mathbb{E}_{j\in J}\left(r_{j}+\frac{1}{n}\right)$ represents the average GSNR contribution across parameters. Without preconditioning, parameters with large $g_{j}^{2}$ but low GSNR may receive higher weights in the OSGR expression, inflating the subtraction term. Our preconditioning alleviates this issue and improves the OSGR. This dynamic adjustment with preconditioning mitigates parameter-wise imbalances, ensuring that well-generalized features are not overwhelmed by noisy or overly dominant parameters.

In implementation, we ignore the $\frac{1}{n}$ term as n is sufficiently large, and clipping variance by $\tanh\left(\frac{1}{\sigma^{2}}\right)$ for stability. More detailed analysis on influence of variance is described in Section 3.3.3. This preconditioner $p_{j}$ is straightforward to compute and requires only the gradient statistics $m_{t}$ and variance $\sigma_{t}$ , which can be estimated during training. This efficiency makes it suitable for a wide range of DG tasks.

# 3.2.2. NOISE INJECTION

To enhance exploration during optimization, we introduce noise injection, where a noise term scaled by the variance is added to the gradient. Specifically, the noise scale is determined by $1 - \tanh\left(\frac{1}{\sigma^{2}}\right)$ , reducing noise for high variance parameters while increasing it for low variance parameters. Motivated by (Mansilla et al., 2021), this injection boosts updates to parameters with low preconditioning value.

# 3.2.3. RANDOM MASK

To further stabilize updates and mitigate overfitting, we apply a random dropout mask. This mask, sampled from a Bernoulli distribution, selectively zeroes out gradient components. By applying random masking after the preconditioning step, all parameters are equally considered to ensure robust updates.

# 3.3. Analysis

We provide a comprehensive theoretical analysis of our method from three perspectives. First, we examine generalization through the OSGR, which highlights how our effectively balances OSGR value across parameters. Second, we formalize our approach under the PAC-Bayes framework, showing that our method explicitly minimizes a tighter generalization bound. Finally, we establish that our optimizer retains the convergence rate of standard SGD while enabling more robust generalization. Proofs are provided in Section C.

# 3.3.1. GENERALIZATION ANALYSIS WITH OSGR

We obtain the following corollary regarding the OSGR of these optimizers:

Corollary 3.3 (OSGR of Optimizers). The OSGR of our proposed optimizer is:

$$
\mathcal {R} _ {\text { Ours }} = 1 - \frac {1}{n \mathbb {E} _ {j \in J} \left(r _ {j} + \frac {1}{n}\right)}, \tag {7}
$$

Comparing the resulting OSGR across different optimizers, we have:

$$
\mathcal {R} _ {\text { Ours }} \geq \mathcal {R} _ {\text { SGD }} \approx \mathcal {R} _ {\text { Adam }}. \tag {8}
$$

This corollary demonstrates that our proposed preconditioning achieves better generalization by attaining a higher overall OSGR. The following remarks provide further context and analysis:

Remark 3.4 (Conceptual Components of Optimizers). The preconditioning applied by common optimizers can be viewed as the element-wise product of two conceptual components:

\- Convergence Term: controls the effective step size,

# Algorithm 1 Algorithm for GENIE

Input: Mini-batches $\{B_{t}\}_{t=1}^{T}$ , Learning Rate $\alpha$ , Total Steps T.

Hyperparameters: $\beta\in[0,1]$ , Dropout Probability $p$ Initialize: Parameters $\theta_{0}, m_{0}\leftarrow 0, v_{0}\leftarrow 0$ .

for t = 1 to T do

Compute Gradient:

$$
g _ {t} = \nabla \mathcal {L} (\theta_ {t}; \mathcal {B} _ {t})
$$

Update Moving Averages:

$$
m _ {t} \leftarrow \beta m _ {t - 1} + (1 - \beta) g _ {t}, v _ {t} \leftarrow \beta v _ {t - 1} + (1 - \beta) g _ {t} ^ {2}
$$

Calculate GSNR and Preconditioning:

$$
\sigma_ {t} ^ {2} = v _ {t} - m _ {t} ^ {2}, r _ {j} = \tanh (\frac {1}{\sigma_ {t} ^ {2}}) m _ {t} ^ {2}
$$

$$
\hat {g} _ {t} \leftarrow \frac {m _ {t}}{1 - \beta^ {t}} \cdot \frac {1}{v _ {t}} \cdot r _ {t}
$$

Noise Injection:

$$
N o i s e _ {t} \leftarrow \xi_ {t} \big [ 1 - \tanh (\frac {1}{\sigma_ {t} ^ {2}}) \big ], \quad \xi_ {t} \sim \mathcal {N} (0, \sigma^ {2})
$$

Random Mask:

$$
M _ {j} \sim B e r n o u l l i (p)
$$

$$
\hat {g} _ {t} \leftarrow (\hat {g} _ {t} + N o i s e _ {t}) \odot M
$$

Update Parameters:

$$
\theta_ {t + 1} \leftarrow \theta_ {t} - \alpha \tilde {g} _ {t}
$$

end for

Output: Final parameters $\theta_{T + 1}$ .

thus contributing to faster convergence. It includes terms such as $E_{D\sim Z^{n}}[g_{j}^{2}]$ or $E_{D\sim Z^{n}}[g_{j}]$ .

\- Alignment Term: adjusts gradients toward stable directions. It includes the GSNR term $r_j$ .

Table 1 summarizes the convergence term, alignment term and their resulting OSGR, including SGD, Adam, and our method.

Remark 3.5 (Optimizer-Specific Analysis). SGD maintains a baseline OSGR value with no explicit adjustment. Adam introduces a convergence component combined with a partial alignment factor. In contrast, our method effectively integrates both aspects in a balanced manner.

Overall, this analysis highlights how each optimizer's design affects generalization through gradient alignment and

Table 1. Comparison of optimizers with preconditioning split into Convergence and Alignment, and OSGR as a separate column. 

<table><tr><td rowspan="2">OPT.</td><td colspan="2">PRECONDITIONING</td><td rowspan="2">OSGR</td><td rowspan="2">WEIGHT</td></tr><tr><td>CONVERGENCE</td><td>ALIGNMENT</td></tr><tr><td>SGD</td><td>-</td><td>-</td><td> $1 - \frac{1}{n} \sum_{j \in J} W_j \cdot \frac{1}{r_j + \frac{1}{n}}$ </td><td> $W_j = \frac{\mathbb{E}_{D \sim \mathcal{Z}^n}[g_j^2]}{\sum_{j'} \mathbb{E}_{D \sim \mathcal{Z}^n}[g_{j'}^2]}$ </td></tr><tr><td>ADAM</td><td colspan="2"><img src="images/e52752dc8c620a5dd8c11285c39b835e3e818b40dd80bf973d3ea054d5c63983.jpg"/></td><td><img src="images/7917dae9b2227019ffb861d558458b4694c2d47ebacc19a9c74db073613125eb.jpg"/></td><td><img src="images/dc8f3b52a5f35cbde9af55d85780a800bf4b0a5d58a4ae76b9a2c0d504f54d67.jpg"/></td></tr><tr><td>GENIE</td><td> $\frac{1}{\mathbb{E}_{D \sim \mathcal{Z}^n}(g_j^2)}$ </td><td> $r_j + \frac{1}{n}$ </td><td> $1 - \frac{1}{n} \sum_{j \in J} W_j \cdot \frac{1}{\mathbb{E}_{j \in J}(r_j + \frac{1}{n})}$ </td><td> $W_j = \frac{1}{|J|}$ </td></tr></table>

convergence speed. Incorporating both perspectives, Our method leads to a higher OSGR and thus improves generalization performance. Furthermore, we demonstrate that the alignment term in our preconditioning achieves a higher OSGR value than those of existing preconditioning methods. Detailed justifications are provided in the Section C.

# 3.3.2. GENERALIZATION ANALYSIS WITH PAC-BAYES BOUND

While the previous analysis is based on alignment and convergence dynamics using OSGR, we now adopt a complementary perspective grounded in the PAC-Bayes framework. We formulate the generalization analysis under a one-step update setting, where the KL divergence between successive parameter distributions reveals the connection between our preconditioning and a tighter generalization bound.

Theorem 3.6 (PAC-Bayes Interpretation of Preconditioning). $R(\theta)$ is the population risk and $L(\theta)$ is empirical risk. Assume that the loss function $L(\theta)$ is bounded in $[0, C]$ . For any $\lambda > 0$ , with probability at least $1 - \delta$ over the draw of $\mathcal{D}$ , and for any data-dependent distribution $\tilde{p}$ over parameters $\theta$ , the following PAC-Bayes bound holds:

$$
\mathbb {E} _ {\theta \sim \tilde {p}} [ R (\theta) ] \leq \underbrace {\mathbb {E} _ {\theta \sim \tilde {p}} [ L (\theta) ]} _ {T _ {1}} + \frac {\lambda C ^ {2}}{8 n} + \underbrace {\frac {\mathrm{KL} (\tilde {p} \| \pi) + \log \frac {1}{\delta}}{\lambda}} _ {T _ {2}}.
$$

Assume that $\tilde{p} = \mathcal{N}(\theta_{t + 1},\Sigma_{\tilde{p}})$ and $\pi = \mathcal{N}(\theta_t,\Sigma_\pi)$ , where $\Sigma_{\tilde{p}} = \mathrm{diag}(q_j^2\cdot \rho_j^2)$ and $\Sigma_{\pi} = \mathrm{diag}(\rho_j^2)$ . Let $q_{j} = \frac{\mathbb{E}[g]^{2}}{\mathbb{E}[g_{j}^{2}]}$ be a variance adaptation factor from SVAG optimizer(Balles & Hennig, 2018) that minimizes the variance to reduce $\max T_1$ term. $(\theta_{t + 1} = \theta_t - q\odot g)$

Then, minimizing the $T_{2}$ term via gradient descent yields an update direction:

$$
\nabla_ {\theta_ {t}} \mathrm{KL} (\tilde {p} \| \pi) = \underbrace {\frac {1}{\mathbb {E} [ g _ {j} ^ {2} ]} \cdot \frac {\mathbb {E} [ g _ {j} ] ^ {2}}{\rho_ {j} ^ {2}}} _ {G E N I E} \cdot g _ {j, t},
$$

which matches the preconditioning rule of our optimizer.

Remark 3.7 (Sharpness and Generalization via KL). This result shows that our method not only improves sharpness—as done in SAM—but also directly enhances generalization by minimizing both terms in the PAC-Bayes bound. Specifically, the variance adaptation factor $q_{j}$ reduces the variability of scaled gradients, thereby tightening the empirical loss term $T_{1}$ through more stable updates. Simultaneously, the $\frac{1}{\rho^{2}}$ term minimizes the KL divergence term $T_{2}$ . This result shows our the generalization property of GENIE comes from correlation with Pac-Bayes theory.

# 3.3.3. CONVERGENCE ANALYSIS

This section analyzes the convergence properties of GENIE under non-convex settings. Specifically, we adopt three widely used assumptions in the optimization literature:

Assumption 3.8. (Bounded Gradient) There exists a constant $G > 0$ such that

$$
\| \nabla \mathcal {L} (\theta_ {t}) \| \leq G \quad \text { for   all } t. \tag {9}
$$

Assumption 3.9. (L-smooth) The loss function L is L-smooth, meaning there exists a constant L > 0 such that for all $\theta_{1}, \theta_{2}$ :

$$
\left\| \nabla \mathcal {L} (\theta_ {1}) - \nabla \mathcal {L} (\theta_ {2}) \right\| \leq L \| \theta_ {1} - \theta_ {2} \|. \tag {10}
$$

Assumption 3.10. (Lower bounded variance) The variance of the stochastic gradients have lower bound by a constant $1 / S_{u}$ :

$$
\mathbb {E} [ \| g _ {t} - \nabla \mathcal {L} (\theta_ {t}) \| ^ {2} ] \geq 1 / S _ {u}, \quad \forall t. \tag {11}
$$

Under these assumptions, we establish the following result regarding the convergence rate:

Theorem 3.11. Under Theorem 3.8 Theorem 3.9, and Theorem 3.10 the average gradient norm over T iterations can be expressed as:

$$
\mathbb {E} [ \| \nabla \mathcal {L} (\theta) \| ^ {2} ] \leq O \left(\frac {1}{P _ {l}} \left(1 + \frac {G \cdot S _ {u} ^ {2}}{2}\right) \frac {1}{\sqrt {\hat {T}}}\right). \tag {12}
$$

where $P_{l}$ is lower bound of preconditioning value.

Remark 3.12 (Convergence Rate and Intuition). Theorem 3.11 shows that the average gradient norm converges at $O(T^{-1/2})$ , the standard rate for stochastic gradient methods in non-convex optimization. This implies that GENIE retains the fundamental convergence properties of SGD.

Remark 3.13 (Influence of $G \cdot S_{u}$ and $S_{u}$ ). The term $G \cdot S_{u}^{2}$ represents a trade-off associated with the GSNR. A higher GSNR upper bound( $G \cdot S_{u}$ ) indicates a stronger gradient signal, which enhances generalization performance. However, it also acts as a multiplicative factor in the gradient norm, potentially slowing down convergence and thereby creating a trade-off. Furthermore, the variance term( $S_{u}$ ) has a significant impact on the bound, further influencing the overall convergence behavior. To address this issue, we regulate the variance term using the tanh function, which effectively balances the interplay between generalization and convergence dynamics.

# 4. Experiment

Dataset. We followed the standardized protocols of DomainBed (Gulrajani & Lopez-Paz, 2021), which include dataset splits, hyperparameter searches, and model selection using validation sets. Our approach was evaluated on five DG benchmark datasets: PACS (Li et al., 2017), VLCS (Fang et al., 2013), OfficeHome (Venkateswara et al., 2017), TerraIncognita (Beery et al., 2018), and DomainNet (Peng et al., 2019).

Evaluation. In accordance with DomainBed protocols, models were trained for 15,000 iterations on DomainNet and 5,000 iterations on the other datasets. For all DG and SDG experiments, we employed the Training-domain Validation Set approach, partitioning the source domain into training and validation subsets. The optimal model was selected based on validation performance. We followed previous DG methods by constructing 20 train-validation splits, with each split repeated 3 times.

Implementation Details. We used ResNet-50 (He et al., 2016b) pre-trained on ImageNet (He et al., 2016a) as backbone architectures. Detailed implementation details are presented in Section D. The detailed results and corresponding confidence intervals of all experiments are provided in Section E.

# 4.1. Comparison of Optimizers on DG

Experiment Setup. We examined the impact of various optimization methods on generalization performance under domain shifts using Baseline ERM (Vapnik, 1999). The evaluated methods included: Standard optimizers (SGD (Robbins & Monro, 1951)), Adaptive optimizers (Adam (Kingma, 2014), AdamW (Loshchilov & Hutter, 2019), AdaBelief (Zhuang et al., 2020), AdaHessian (Yao et al., 2021), YOGI (Zaheer et al., 2018)), Sharpness-aware optimizers (SAM (Foret et al., 2021), GAM (Zhang et al., 2023b), FAD (Zhang et al., 2023a)) and our proposed GENIE.

Results. As shown in Table 2, our optimizer achieved superior performance across most datasets, surpassing existing methods. GENIE outperformed Adam, the default optimizer in most DG algorithms(Zhang et al., 2023a), by 5.69%. Additionally, it achieved improvements of 6.36% over SGD and 4.37% over SAM. In particular, it achieved remarkable performance on VLCS, which is prone to early convergence and overfitting(Matsuura & Harada, 2020), and on TerraIncognita, a wildlife image dataset with significant challenges such as lighting variations, motion blur, occlusions, and severe class imbalance(Beery et al., 2018). These results suggest that GENIE effectively prevents overfitting and enhances the learning of causal relationships by balancing parameter contributions during training. Optimizers designed for generalization, such as SAM, GAM and FAD, outperform standard optimizers, underscoring the significant role of optimization in generalization. These results emphasize the need for developing optimizers specifically tailored for DG.

Table 2. Comparison of optimizers on DG datasets. Results denoted by \* are reproduced from (Zhang et al., 2023a) using the same protocol as our paper. The best results for each dataset are highlighted in bold. 

<table><tr><td>OPT.</td><td>PACS</td><td>VLCS</td><td>OFFICE HOME</td><td>TERRA INC</td><td>DOMAIN NET</td><td>AVG.</td></tr><tr><td>ADAM*</td><td>84.2</td><td>77.3</td><td>67.6</td><td>44.4</td><td>43.0</td><td>63.3</td></tr><tr><td>ADAMW*</td><td>83.6</td><td>77.4</td><td>68.8</td><td>45.2</td><td>43.4</td><td>63.7</td></tr><tr><td>SGD*</td><td>79.9</td><td>78.1</td><td>68.5</td><td>44.9</td><td>43.2</td><td>62.9</td></tr><tr><td>YOGI*</td><td>81.2</td><td>77.6</td><td>68.3</td><td>45.4</td><td>43.5</td><td>63.2</td></tr><tr><td>ADABELIEF*</td><td>84.6</td><td>78.4</td><td>68.0</td><td>45.2</td><td>43.5</td><td>63.9</td></tr><tr><td>ADAHESSIAN*</td><td>84.5</td><td>78.6</td><td>68.4</td><td>44.4</td><td>44.4</td><td>64.1</td></tr><tr><td>SAM*</td><td>85.3</td><td>78.2</td><td>68.0</td><td>45.7</td><td>43.4</td><td>64.1</td></tr><tr><td>GAM*</td><td>86.1</td><td>78.5</td><td>68.2</td><td>45.2</td><td>43.8</td><td>64.4</td></tr><tr><td>FAD*</td><td>88.2</td><td>78.9</td><td>69.2</td><td>45.7</td><td>44.4</td><td>65.3</td></tr><tr><td>GENIE</td><td>87.8</td><td>80.7</td><td>69.7</td><td>52.0</td><td>44.1</td><td>66.9</td></tr></table>

Experiment Setup. The computational overhead of an optimizer is a critical factor in its practical applicability. To evaluate this, we trained models on the PACS and VLCS datasets for 5,000, 10,000, and 15,000 iterations, measuring average performance and training time per iteration.

Results. As reported in Table 3, GENIE consistently outperformed other optimizers, even at 5,000 iterations, while incurring lower computational overhead than SGD and Adam. Additionally, GENIE achieved an average of $1.3 \times$ faster training compared to SAM, as SAM's update rule requires

two sequential (non-parallelizable) gradient computations per step, which doubles the training time. These results experimentally validate the theoretical convergence analysis in Section 3.3.3, confirming GENIE's ability in computational efficiency and convergence speed.

Table 3. Training time (sec) and average accuracy at different iteration levels. 

<table><tr><td rowspan="2">OPT.</td><td rowspan="2">ITER.</td><td>TRAINING</td><td colspan="3">AVG.</td></tr><tr><td>TIME(/S)</td><td>PACS</td><td>VLCS</td><td>OFFICE HOME</td></tr><tr><td rowspan="3">SGD</td><td>5000</td><td>5,273</td><td>69.8</td><td>76.7</td><td>51.3</td></tr><tr><td>10000</td><td>10,546</td><td>73.9</td><td>77</td><td>62.5</td></tr><tr><td>15000</td><td>15,783</td><td>75.8</td><td>77.7</td><td>63.9</td></tr><tr><td rowspan="3">ADAM</td><td>5000</td><td>5,443</td><td>84.2</td><td>77</td><td>63.6</td></tr><tr><td>10000</td><td>10,934</td><td>86.1</td><td>77</td><td>65.2</td></tr><tr><td>15000</td><td>16,531</td><td>84.5</td><td>77</td><td>65.2</td></tr><tr><td rowspan="3">SAM</td><td>5000</td><td>5,775</td><td>82.4</td><td>79.4</td><td>69.4</td></tr><tr><td>10000</td><td>11,500</td><td>83.5</td><td>80.3</td><td>69.6</td></tr><tr><td>15000</td><td>17,191</td><td>84.1</td><td>80.4</td><td>70</td></tr><tr><td rowspan="3">GENIE(OURS)</td><td>5000</td><td>4,292</td><td>88.4</td><td>81.3</td><td>70</td></tr><tr><td>10000</td><td>8,582</td><td>87.1</td><td>81.3</td><td>69.2</td></tr><tr><td>15000</td><td>12,876</td><td>86.9</td><td>81.3</td><td>69.1</td></tr></table>

# 4.2. Integration with Current DG Algorithms

Experiment Setup. GENIE is a versatile optimizer that integrates seamlessly with various DG algorithms without requiring changes to the training procedure or model architecture. To validate its compatibility, we combined GENIE with several well-performing DG algorithms—CORAL(Sun & Saenko, 2016) and RSC(Huang et al., 2020) using ResNet-50 as the backbone—and compared its performance against other optimization techniques.

Results. The performance evaluation results for DG are summarized in Table 5. GENIE consistently outperforms existing optimization methods, demonstrating its robustness and broad applicability. These results validate GENIE's scalability and compatibility with various DG algorithms. Unlike other DG methods, which often require multiple source domains or architecture modifications, GENIE seamlessly integrates with existing training pipelines, providing consistent performance gains without additional complexity. This establishes GENIE as an algorithm-agnostic and highly adaptable optimization framework for DG tasks.

# 4.3. Single Domain Generalization

Experiment Setup. We evaluated performance in Single Domain Generalization (SDG), which is more constrained but better reflects real-world applications. The flexibility to operate in SDG without structural modifications is an advantage of our method over certain existing methods that are limited to multi-source settings. In SDG, the model is trained and validated on a single domain and tested on the others, with results averaged across all source domains. We compared GENIE with Adam, SGD, and SAM, and applied it to existing DG methods.

Results. The SDG performance results are presented in Table 4. As in previous DG settings, our optimizer outperformed existing optimizers. When applied to DG methods, conventional optimizers reduced performance, whereas GENIE achieved the highest performance as a standalone model and also improved DG methods when used as an optimizer. These results show that our method enhances DG performance without requiring architectural modifications or multiple source domains, and performs well even as a standalone method.

Table 4. Experimental results of GENIE under the SDG setting. 

<table><tr><td>ALGORITHM</td><td>PACS</td><td>VLCS</td><td>OFFICE HOME</td><td>TERRA INC</td><td>AVG.</td></tr><tr><td>ADAM</td><td>64.3</td><td>56.2</td><td>50.7</td><td>33.5</td><td>51.2</td></tr><tr><td>SGD</td><td>49.5</td><td>60.4</td><td>45.9</td><td>22.8</td><td>44.7</td></tr><tr><td>SAM</td><td>57.7</td><td>66.7</td><td>59.2</td><td>26.8</td><td>52.6</td></tr><tr><td>GENIE (OURS)</td><td>69.5</td><td>69.9</td><td>58.6</td><td>36.0</td><td>58.5</td></tr><tr><td>RSC+ADAM</td><td>56.8</td><td>51.6</td><td>2.1</td><td>31.6</td><td>35.5</td></tr><tr><td>RSC+SGD</td><td>22.2</td><td>39.8</td><td>1.7</td><td>17.6</td><td>20.3</td></tr><tr><td>RSC+GENIE(OURS)</td><td>68.2</td><td>68.7</td><td>54.4</td><td>33.2</td><td>56.1</td></tr><tr><td>CORAL+ADAM</td><td>64.3</td><td>56.2</td><td>50.7</td><td>33.5</td><td>51.2</td></tr><tr><td>CORAL+SGD</td><td>49.5</td><td>60.4</td><td>45.9</td><td>22.8</td><td>44.7</td></tr><tr><td>CORAL+GENIE(OURS)</td><td>70.9</td><td>69.2</td><td>56.4</td><td>36.7</td><td>58.3</td></tr></table>

# 4.4. Model Analysis

Ablation. We conducted an ablation study using the PACS dataset in a DG setting to evaluate the effects of Preconditioning, Noise Injection, and Random Mask (Table 6). The version without all three components corresponds to ERM trained with Adam, while the version incorporating all three represents our proposed GENIE optimizer. Experimental results show that GENIE achieved the highest performance, improving accuracy by 4.9% compared to ERM. Even when using only Preconditioning, performance improved by 3.8%, indicating that a simple preconditioning technique can enhance generalization. Additionally, in the Cartoon and Sketch domains, where objects are placed on a white background, models trained with Noise Injection and Random Mask performed better. Here, we conclude that preconditioning alone is enough for DG, but you can optionally utilize Noise Injection and Random Mask for additional robustness.

Sensitivity Analysis. GENIE employs two key hyperparameters: the dropout probability P and the coefficient B, which is used to compute the moving average and variance of gradients. To analyze the sensitivity of these hyperparameters, we conducted a grid search while keeping all other training settings fixed. As shown in Figure 2, GENIE consistently outperformed SGD, Adam, and SAM across a wide range of Pand B values, demonstrating strong robustness to hyperparameter variation. Notably, while this experi-

Table 5. Integration with DG methods. Results obtained from the original literature and DomainBed (Gulrajani & Lopez-Paz, 2021) are denoted with $\dagger$ , while results taken from (Zhang et al., 2023a) are denoted with \*. 

<table><tr><td>ALGORITHM</td><td>PACS</td><td>VLCS</td><td>OFFICEHOME</td><td>TERRAINC</td><td>AVG.</td></tr><tr><td>ERM†(VAPNIK, 1999)</td><td>85.5</td><td>77.5</td><td>66.5</td><td>46.1</td><td>68.9</td></tr><tr><td>IRM†(ARJOVSKY ET AL., 2019)</td><td>83.5</td><td>78.6</td><td>64.3</td><td>47.6</td><td>68.5</td></tr><tr><td>GROUPDRO†(SAGAWA ET AL., 2020)</td><td>84.4</td><td>76.7</td><td>66.0</td><td>43.2</td><td>67.6</td></tr><tr><td>I-MIXUP†(XU ET AL., 2020)</td><td>84.6</td><td>77.4</td><td>68.1</td><td>47.9</td><td>69.5</td></tr><tr><td>MLDG†(LI ET AL., 2018A)</td><td>84.9</td><td>77.2</td><td>66.8</td><td>47.8</td><td>69.2</td></tr><tr><td>MMD†(LI ET AL., 2018B)</td><td>84.7</td><td>77.5</td><td>66.4</td><td>42.2</td><td>67.7</td></tr><tr><td>DANN†(GANIN ET AL., 2016)</td><td>83.7</td><td>78.6</td><td>65.9</td><td>46.7</td><td>68.7</td></tr><tr><td>CDANN†(LI ET AL., 2018C)</td><td>82.6</td><td>77.5</td><td>65.7</td><td>45.8</td><td>67.9</td></tr><tr><td>MTL†(BLANCHARD ET AL., 2021)</td><td>84.6</td><td>77.2</td><td>66.4</td><td>45.6</td><td>68.5</td></tr><tr><td>SAGNET†(NAM ET AL., 2021)</td><td>86.3</td><td>77.8</td><td>68.1</td><td>48.6</td><td>70.2</td></tr><tr><td>ARM†(ZHANG ET AL., 2021)</td><td>85.1</td><td>77.6</td><td>64.8</td><td>45.5</td><td>68.3</td></tr><tr><td>VREX†(KRUEGER ET AL., 2021)</td><td>84.9</td><td>78.3</td><td>66.4</td><td>46.4</td><td>69</td></tr><tr><td>MIXSTYLE*(ZHOU ET AL., 2021)</td><td>85.2</td><td>77.9</td><td>60.4</td><td>44</td><td>66.9</td></tr><tr><td>MIRO*(CHA ET AL., 2022)</td><td>85.4</td><td>78.9</td><td>69.5</td><td>45.4</td><td>69.8</td></tr><tr><td>GENIE (OURS)</td><td>87.8</td><td>80.7</td><td>69.7</td><td>52.0</td><td>72.6</td></tr><tr><td>RSC(HUANG ET AL., 2020)+ADAM*</td><td>84.5</td><td>77.9</td><td>65.7</td><td>44.5</td><td>68.2</td></tr><tr><td>RSC+ADAMW*</td><td>83.4</td><td>77.5</td><td>66.3</td><td>45.1</td><td>68.1</td></tr><tr><td>RSC+SGD*</td><td>82.6</td><td>78.1</td><td>67</td><td>43.9</td><td>67.9</td></tr><tr><td>RSC+GENIE(OURS)</td><td>87.3</td><td>80.6</td><td>68.1</td><td>49.5</td><td>71.4</td></tr><tr><td>CORAL(SUN &amp; SAENKO, 2016) + ADAM*</td><td>86</td><td>78.9</td><td>68.7</td><td>43.7</td><td>69.3</td></tr><tr><td>CORAL+ADAMW*</td><td>86.4</td><td>79.5</td><td>69.8</td><td>45.0</td><td>70.2</td></tr><tr><td>CORAL+SGD*</td><td>85.6</td><td>78.2</td><td>69.5</td><td>45.8</td><td>69.8</td></tr><tr><td>CORAL+GENIE(OURS)</td><td>87.9</td><td>80.7</td><td>70.6</td><td>48.4</td><td>71.9</td></tr></table>

Table 6. Ablation study on the PACS dataset. Results are reported for evaluations on four domains: Art, Cartoon, Photo, and Sketch. 

<table><tr><td rowspan="2">PRE CONDITION</td><td rowspan="2">NOISE</td><td rowspan="2">MASK</td><td colspan="4">PACS</td><td rowspan="2">AVG.</td></tr><tr><td>A</td><td>C</td><td>P</td><td>S</td></tr><tr><td>X</td><td>X</td><td>X</td><td>88.0</td><td>79.7</td><td>96.7</td><td>72.7</td><td>84.2</td></tr><tr><td>O</td><td>X</td><td>X</td><td>89.5</td><td>82.3</td><td>98.4</td><td>79.4</td><td>87.4</td></tr><tr><td>O</td><td>O</td><td>X</td><td>85.4</td><td>77.4</td><td>98.6</td><td>78.7</td><td>85.0</td></tr><tr><td>O</td><td>X</td><td>O</td><td>84.6</td><td>79.9</td><td>98.3</td><td>77.4</td><td>85.1</td></tr><tr><td>O</td><td>O</td><td>O</td><td>89.3</td><td>84.1</td><td>98.7</td><td>81.6</td><td>88.4</td></tr></table>

ment involved hyperparameter tuning via grid search, all other experiments followed the DomainBed protocol, using validation performance for hyperparameter selection.

OSGR of Network Parameters Over Time. To assess whether our approach enhances the overall OSGR of network parameters during training, we tracked the average OSGR of all parameters throughout the training process. As shown in Figure 4, the OSGR measurements on the VLCS dataset show that GENIE achieves an OSGR closer to 1 than prior optimizers. This means superior generalization performance. These findings align with the theoretical Generalization analysis in Section 3.3.1, confirming that GENIE ensures more stable and balanced parameter updates during

![](images/0dd3b68b24e3a61fa77a3b68a6401557fd7a52ed039af2295c808c5706e36511.jpg)

<details>
<summary>line</summary>

| Dropout Probability(P) | Accuracy (Blue Dashed) | Accuracy (Pink Dotted) | Accuracy (Orange Dash-Dot) |
| ----------------------- | ---------------------- | ---------------------- | -------------------------- |
| 0.0                     | 55.0                   | 49.0                   | 48.0                       |
| 0.2                     | 54.5                   | 49.0                   | 48.0                       |
| 0.4                     | 54.0                   | 49.0                   | 48.0                       |
| 0.6                     | 52.5                   | 49.0                   | 48.0                       |
| 0.8                     | 54.5                   | 49.0                   | 48.0                       |
</details>

-●- GENIE(our)

![](images/d32bf2729a5dffa709195141fd613b78d5f7ae014ed2c3d651bb150d87f34164.jpg)

<details>
<summary>line</summary>

| Moving Avg(β) | Accuracy |
| ------------- | -------- |
| 0.9           | 52.5     |
| 0.95          | 54.5     |
| 0.99          | 51.5     |
| 0.995         | 52.5     |
| 0.999         | 50.0     |
</details>

Adam   
SAM

Figure 2. Performance sensitivity of GENIE to dropout probability P and coefficient B.

training, which ultimately leads to improved generalization. Interestingly, while SAM is designed for better generalization performance, it exhibits inferior OSGR values. This suggests that the sharpness-aware regime alone is insufficient for generalization, and that the OSGR regime should also be considered when addressing generalization in DG tasks. This observation is consistent with our PAC-Bayesian analysis in Section 3.3.2, which reveals that inducing balanced OSGR values leads to tighter generalization bounds, reinforcing the role of OSGR as a necessary complement to sharpness-aware optimization.

![](images/7f951a36afb782ed628f885fab429576e90ab88db51ab5f6229a363c6bfbd1ab.jpg)  
(A) Before Training

![](images/1f4211fdf8bf93f828e1c7c716b53b0db7860a75b9dacd77df547392d131e433.jpg)

<details>
<summary>scatter</summary>

| Class   | X Range | Y Range |
|---------|---------|---------|
| Class 0 | ~0–10   | ~-2–8   |
| Class 1 | ~0–6    | ~-2–8   |
| Class 2 | ~0–6    | ~-2–8   |
| Class 3 | ~0–6    | ~-2–8   |
| Class 4 | ~0–6    | ~-2–8   |
| Class 5 | ~0–6    | ~-2–8   |
| Class 6 | ~0–6    | ~-2–8   |
</details>

(B) SGD

![](images/e90f9a500a907374e1b11c23acb2ede8166b1512d983dc69b4f832a2be3ad8cf.jpg)

<details>
<summary>scatter</summary>

| Class   | X Range     | Y Range |
|---------|-------------|---------|
| Class 0 | -10 to 0    | 0 to 10 |
| Class 1 | -10 to 0    | 15 to 20 |
| Class 2 | -10 to 0    | 5 to 10  |
| Class 3 | 10 to 15    | 15 to 20 |
| Class 4 | -5 to 5     | 5 to 10  |
| Class 5 | -5 to 5     | 5 to 10  |
| Class 6 | -5 to 5     | 10 to 15 |
</details>

(C) Adam

![](images/cfafdc6377af75f0821c1e3fc1c770fa1a225a4afb0c1a47df74bc17c1f795f6.jpg)

<details>
<summary>scatter</summary>

| x    | y    | Class   |
| ---- | ---- | ------- |
| -5   | 17   | Class 0 |
| -5   | 10   | Class 1 |
| -5   | 8    | Class 2 |
| -5   | 4    | Class 3 |
| -5   | -3   | Class 4 |
| -5   | -5   | Class 5 |
| 5    | 19   | Class 0 |
| 5    | 10   | Class 1 |
| 5    | 8    | Class 2 |
| 5    | 4    | Class 3 |
| 5    | -3   | Class 4 |
| 5    | -5   | Class 5 |
| 15   | 2    | Class 6 |
| 15   | 0    | Class 6 |
| 15   | -2   | Class 6 |
| 15   | -4   | Class 6 |
</details>

(D) GENIE   
Figure 3. UMAP visualization of learned features on the PACS dataset with Sketch as the unseen target domain. (A) Before training. (B)-(D) After training with SGD, Adam, and GENIE.

![](images/fdab46e12ef12eab6b63ba625ac5ebf585d9ebf010e578b2b8c32b9ada00ad28.jpg)

<details>
<summary>line</summary>

| Iteration | GENIE   | SGD     | Adam    | SAM     |
| --------- | ------- | ------- | ------- | ------- |
| 0         | 0.9895  | 0.9895  | 0.9895  | 0.9895  |
| 2000      | 0.9895  | 0.9895  | 0.9895  | 0.9895  |
| 4000      | 0.9900  | 0.9895  | 0.9900  | 0.9895  |
| 6000      | 0.9905  | 0.9895  | 0.9905  | 0.9895  |
| 8000      | 0.9910  | 0.9895  | 0.9910  | 0.9895  |
| 10000     | 0.9915  | 0.9895  | 0.9915  | 0.9895  |
| 12000     | 0.9920  | 0.9895  | 0.9920  | 0.9895  |
| 14000     | 0.9925  | 0.9895  | 0.9925  | 0.9895  |
| 16000     | 0.9930  | 0.9895  | 0.9930  | 0.9895  |
</details>

![](images/9af31856506b83adb74d1f28ccebe709f4fcbfde03eac7769dc71bbeac1df272.jpg)

<details>
<summary>line</summary>

| Iteration | GENIE  | SGD    | Adam   | SAM    |
| --------- | ------ | ------ | ------ | ------ |
| 0         | 0.991  | 0.990  | 0.990  | 0.990  |
| 2000      | 0.991  | 0.990  | 0.990  | 0.990  |
| 4000      | 0.991  | 0.990  | 0.990  | 0.990  |
| 6000      | 0.992  | 0.990  | 0.991  | 0.990  |
| 8000      | 0.993  | 0.990  | 0.992  | 0.990  |
| 10000     | 0.994  | 0.990  | 0.993  | 0.990  |
| 12000     | 0.995  | 0.990  | 0.994  | 0.990  |
| 14000     | 0.995  | 0.990  | 0.993  | 0.990  |
</details>

Figure 4. OSGR measurements over training iterations for different optimizers on the VLCS dataset. The OSGR values were calculated for all iterations and averaged every 200 iterations for clarity and ease of comparison.

![](images/db2cd7e7727a6ce1a191165e76f05305f431fabe4c2107b9005e261b0647e435.jpg)  
Figure 5. Optimization trajectories on a simulated loss landscape.

Loss Landscape. We analyzed the convergence paths of SGD, Adam, and GENIE in the loss landscape using the FashionMNIST dataset(Xiao et al., 2017). As shown in Figure 5, each corner represents the local minima of a specific source domain. All optimizers started at (-1,3) and were updated for 30 steps under the same conditions. SGD and Adam follow steep direction and converge quickly. However, fast convergence often causes overfitting to specific source domains in OOD scenarios. Generalizable features are learned later in training(Pérez et al., 2019; Shah et al., 2020b; Nakkiran et al., 2019), so rapid convergence can prevent the model from acquiring them sufficiently. In contrast, as demonstrated in the theoretical analysis in Section 3.3.2, GENIE leads optimization toward flatter minima by effectively reducing sharpness, thereby improving generalization(Foret et al., 2021).

Feature Visualization. To examine how the GENIE optimizer operates at the feature level, we performed UMAP visualizations(McInnes et al., 2018) on the PACS dataset, with the Sketch domain held out as the unseen target. Each color represents a different class. The results Figure 3 show that GENIE leads to clear class separation across domains, suggesting effective domain-invariant feature learning.

# 5. Conclusion

We introduce GENIE, an optimizer that leverages OSGR to guide gradients in effective directions, preventing overly predictive parameters from dominating while ensuring all parameters contribute equitably to learning. GENIE achieves a higher OSGR with improved generalization and ensures fast convergence rate comparable to SGD. Empirically, it outperforms state-of-the-art optimizers across five DG benchmarks, demonstrating robust performance under significant domain shifts and limited data. Seamlessly integrating with existing DG and SDG methods, GENIE consistently achieves performance improvements. This work highlights the potential of OSGR as a guiding principle, paving the way for its use in few-shot learning, meta-learning, and other tasks requiring solutions to source-domain overfitting.

# Acknowledgements

This work was supported by Korea Internet & Security Agency(KISA) grant funded by the Korea government(PIPC) (No.RS-2023-00231200, Development of personal video information privacy protection technology capable of AI learning in an autonomous driving environment)

# Impact Statement

Our work proposes GENIE, an optimization method that enhances domain generalization by ensuring stable and balanced updates. GENIE mitigates overfitting, promotes flatter minima, and improves OOD performance, contributing to more robust and generalizable models.

# References

Amari, S.-i., Ba, J., Grosse, R. B., Li, X., Nitanda, A., Suzuki, T., Wu, D., and Xu, J. When does preconditioning help or hurt generalization? In International Conference on Learning Representations, 2021.   
Arjovsky, M., Bottou, L., Gulrajani, I., and Lopez-Paz, D. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.   
Balles, L. and Hennig, P. Dissecting adam: The sign, magnitude and variance of stochastic gradients. In International Conference on Machine Learning, pp. 404–413. PMLR, 2018.   
Beery, S., Van Horn, G., and Perona, P. Recognition in terra incognita. In Proceedings of the European conference on computer vision (ECCV), pp. 456–473, 2018.   
Blanchard, G., Deshmukh, A. A., Dogan, U., Lee, G., and Scott, C. Domain generalization by marginal transfer learning. Journal of machine learning research, 22(2):1–55, 2021.   
Cha, J., Lee, K., Park, S., and Chun, S. Domain generalization by mutual-information regularization with pretrained models. In Avidan, S., Brostow, G., Cissé, M., Farinella, G. M., and Hassner, T. (eds.), Computer Vision – ECCV 2022, pp. 440–457, Cham, 2022. Springer Nature Switzerland.   
Dennis, Jr, J. E. and Moré, J. J. Quasi-newton methods, motivation and theory. SIAM review, 19(1):46–89, 1977.   
Duchi, J., Hazan, E., and Singer, Y. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7), 2011.   
Fang, C., Xu, Y., and Rockmore, D. N. Unbiased metric learning: On the utilization of multiple datasets and web images for softening bias. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1657–1664, 2013.   
Foret, P., Kleiner, A., Mobahi, H., and Neyshabur, B. Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning Representations, 2021.

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., March, M., and Lempitsky, V. Domain-adversarial training of neural networks. Journal of machine learning research, 17(59):1–35, 2016.

Gulrajani, I. and Lopez-Paz, D. In search of lost domain generalization. In International Conference on Learning Representations, 2021.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016a.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016b.

Hemati, S., Zhang, G., Estiri, A., and Chen, X. Understanding hessian alignment for domain generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19004–19014, 2023.

Hinton, G., Srivastava, N., and Swersky, K. Neural networks for machine learning lecture 6a overview of mini-batch gradient descent. Cited on, 14(8):2, 2012.

Huang, Z., Wang, H., Xing, E. P., and Huang, D. Self-challenging improves cross-domain generalization. In Vedaldi, A., Bischof, H., Brox, T., and Frahm, J.-M. (eds.), Computer Vision – ECCV 2020, pp. 124–140, Cham, 2020. Springer International Publishing.

Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pp. 448–456. pmlr, 2015.

Kingma, D. P. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Koyama, M. and Yamaguchi, S. When is invariance useful in an out-of-distribution generalization problem? arXiv preprint arXiv:2008.01883, 2020.

Krueger, D., Caballero, E., Jacobsen, J.-H., Zhang, A., Binas, J., Zhang, D., Priol, R. L., and Courville, A. Out-of-distribution generalization via risk extrapolation (rex). In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 5815–5826. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/krueger21a.html.

Li, D., Yang, Y., Song, Y.-Z., and Hospedales, T. M. Deeper, broader and artier domain generalization. In Proceedings

of the IEEE international conference on computer vision, pp. 5542–5550, 2017.   
Li, D., Yang, Y., Song, Y.-Z., and Hospedales, T. Learning to generalize: Meta-learning for domain generalization. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018a.   
Li, H., Pan, S. J., Wang, S., and Kot, A. C. Domain generalization with adversarial feature learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5400–5409, 2018b.   
Li, Y., Gong, M., Tian, X., Liu, T., and Tao, D. Domain generalization via conditional invariant representations. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018c.   
Liu, J., Bai, Y., Jiang, G., Chen, T., and Wang, H. Understanding why neural networks generalize well through gsnr of parameters. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=HyevIJStwH.   
Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.   
Mahajan, D., Tople, S., and Sharma, A. Domain generalization using causal matching. In International conference on machine learning, pp. 7313–7324. PMLR, 2021.   
Mansilla, L., Echeveste, R., Milone, D. H., and Ferrante, E. Domain generalization via gradient surgery. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6630–6638, 2021.   
Matsuura, T. and Harada, T. Domain generalization using a mixture of multiple latent domains. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 11749–11756, 2020.   
McInnes, L., Healy, J., and Melville, J. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.   
Michalkiewicz, M., Faraki, M., Yu, X., Chandraker, M., and Baktashmotlagh, M. Domain generalization guided by gradient signal to noise ratio of parameters. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6177–6188, 2023.   
Montavon, G., Orr, G., and Müller, K.-R. Neural networks: tricks of the trade, volume 7700. springer, 2012.

Muandet, K., Balduzzi, D., and Schölkopf, B. Domain generalization via invariant feature representation. In International conference on machine learning, pp. 10–18. PMLR, 2013.   
Nakkiran, P., Kaplun, G., Kalimeris, D., Yang, T., Edelman, B. L., Zhang, F., and Barak, B. Sgd on neural networks learns functions of increasing complexity. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 3496–3506, 2019.   
Nam, H., Lee, H., Park, J., Yoon, W., and Yoo, D. Reducing domain gap by reducing style bias. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8690–8699, June 2021.   
Parascandolo, G., Neitz, A., Orvieto, A., Gresele, L., and Schölkopf, B. Learning explanations that are hard to vary. In International Conference on Learning Representations, 2021.   
Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., and Wang, B. Moment matching for multi-source domain adaptation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1406–1415, 2019.   
Pérez, G. V., Louis, A. A., and Camargo, C. Q. Deep learning generalizes because the parameter-function map is biased towards simple functions. In 7th International Conference on Learning Representations, ICLR 2019, 2019.   
Rame, A., Dancette, C., and Cord, M. Fishr: Invariant gradient variances for out-of-distribution generalization. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 18347–18377. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/rame22a.html.   
Robbins, H. and Monro, S. A Stochastic Approximation Method. The Annals of Mathematical Statistics, 22(3):400 – 407, 1951. doi: 10.1214/aoms/1177729586. URL https://doi.org/10.1214/aoms/1177729586.   
Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P. Distributionally robust neural networks. In International Conference on Learning Representations, 2020.   
Shah, H., Tamuly, K., Raghunathan, A., Jain, P., and Netrapalli, P. The pitfalls of simplicity bias in neural networks. Advances in Neural Information Processing Systems, 33:9573–9585, 2020a.

Shah, H., Tamuly, K., Raghunathan, A., Jain, P., and Netrapalli, P. The pitfalls of simplicity bias in neural networks. Advances in Neural Information Processing Systems, 33:9573–9585, 2020b.   
Shahtalebi, S., Gagnon-Audet, J.-C., Laleh, T., Faramarzi, M., Ahuja, K., and Rish, I. Sand-mask: An enhanced gradient masking strategy for the discovery of invariances in domain generalization, 2021. URL https://arxiv.org/abs/2106.02266.   
Shi, Y., Seely, J., Torr, P., Siddharth, N., Hannun, A., Usunier, N., and Synnaeve, G. Gradient matching for domain generalization. In International Conference on Learning Representations, 2022.   
Sun, B. and Saenko, K. Deep coral: Correlation alignment for deep domain adaptation. In Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14, pp. 443–450. Springer, 2016.   
Vapnik, V. N. An overview of statistical learning theory. IEEE transactions on neural networks, 10(5):988–999, 1999.   
Venkateswara, H., Eusebio, J., Chakraborty, S., and Panchanathan, S. Deep hashing network for unsupervised domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5018–5027, 2017.   
Wang, P., Zhang, Z., Lei, Z., and Zhang, L. Sharpness-aware gradient matching for domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3769–3778, 2023.   
Wang, Y., Li, H., and Kot, A. C. Heterogeneous domain generalization via domain mixup. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 3622–3626. IEEE, 2020.   
Xiao, H., Rasul, K., and Vollgraf, R. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.   
Xu, M., Zhang, J., Ni, B., Li, T., Wang, C., Tian, Q., and Zhang, W. Adversarial domain adaptation with domain mixup. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 6502–6509, 2020.   
Yan, S., Song, H., Li, N., Zou, L., and Ren, L. Improve unsupervised domain adaptation with mixup training. arXiv preprint arXiv:2001.00677, 2020.   
Yao, Z., Gholami, A., Shen, S., Mustafa, M., Keutzer, K., and Mahoney, M. Adahessian: An adaptive second order optimizer for machine learning. In proceedings of the

AAAI conference on artificial intelligence, volume 35, pp.10665–10673, 2021.   
Ye, Q. Preconditioning for accelerated gradient descent optimization and regularization. arXiv preprint arXiv:2410.00232, 2024.   
Ye, W., Zheng, G., Cao, X., Ma, Y., and Zhang, A. Spurious correlations in machine learning: A survey. arXiv preprint arXiv:2402.12715, 2024.   
Zaheer, M., Reddi, S., Sachan, D., Kale, S., and Kumar, S. Adaptive methods for nonconvex optimization. Advances in neural information processing systems, 31, 2018.   
Zhang, M., Marklund, H., Dhawan, N., Gupta, A., Levine, S., and Finn, C. Adaptive risk minimization: learning to adapt to domain shift. In Proceedings of the 35th International Conference on Neural Information Processing Systems, pp. 23664–23678, 2021.   
Zhang, R., Fan, Z., Yao, J., Zhang, Y., and Wang, Y. Domain-inspired sharpness-aware minimization under domain shifts. In The Twelfth International Conference on Learning Representations, 2024.   
Zhang, X., Xu, R., Yu, H., Dong, Y., Tian, P., and Cui, P. Flatness-aware minimization for domain generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5189–5202, October 2023a.   
Zhang, X., Xu, R., Yu, H., Zou, H., and Cui, P. Gradient norm aware minimization seeks first-order flatness and improves generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20247–20257, 2023b.   
Zhou, K., Yang, Y., Qiao, Y., and Xiang, T. Domain generalization with mixstyle. In International Conference on Learning Representations, 2021.   
Zhuang, J., Tang, T., Ding, Y., Tatikonda, S. C., Dvornek, N., Papademetris, X., and Duncan, J. Adabelief optimizer: Adapting stepsizes by the belief in observed gradients. Advances in neural information processing systems, 33:18795–18806, 2020.

# A. Notation

Table 7. Final Revised Notation Table 

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $f$ </td><td>neural network</td></tr><tr><td> $L(\theta)$ </td><td>loss function</td></tr><tr><td> $\mathcal{Z}$ </td><td>data distribution defined over  $\mathcal{X} \times \mathcal{Y}$ </td></tr><tr><td> $n$ </td><td>number of data samples</td></tr><tr><td> $D, D'$ </td><td>training/test dataset drawn from  $\mathcal{Z}$ </td></tr><tr><td> $\theta, \theta_j$ </td><td>model parameters, parameter  $j$ </td></tr><tr><td> $\theta_{t,j}$ </td><td>parameter of index  $j$  at optimization step  $t$ </td></tr><tr><td> $g_{D,j}(\theta)$ </td><td>gradient of parameter  $j$  averaged over training set  $D$ </td></tr><tr><td> $g_t$ </td><td>gradient at step  $t$ </td></tr><tr><td> $g_j^2$ </td><td>squared gradient for parameter  $j$ </td></tr><tr><td> $\rho_j^2$ </td><td>variance of parameter  $j$ &#x27;s gradient</td></tr><tr><td> $\sigma_j^2$ </td><td>variance of gradient averaged over training set</td></tr><tr><td> $r_j$ </td><td>gradient signal-to-noise ratio (GSNR),  $r_j = \frac{g_j^2}{\rho_j^2}$ </td></tr><tr><td> $p_j$ </td><td>proposed preconditioning factor for parameter  $j$ </td></tr><tr><td> $R(Z, n)$ </td><td>one-step generalization ratio (OSGR)</td></tr><tr><td> $\xi_t \sim \mathcal{N}(0, \sigma^2)$ </td><td>Gaussian noise for noise injection</td></tr><tr><td> $J$ </td><td>set of parameter index</td></tr><tr><td> $G$ </td><td>bound of gradient  $l_2$  norm</td></tr><tr><td> $1/S_u$ </td><td>lower bound of gradient variance</td></tr><tr><td> $L$ </td><td>Lipschitz constant</td></tr><tr><td> $P_l$ </td><td>lower bound of preconditioning value</td></tr><tr><td> $W_j$ </td><td>weighting factor showing up in optimizers,  $\frac{\mathbb{E}_{D \sim \mathcal{Z}^n(g_D^2,j)}}{\sum_{j'} \mathbb{E}_{D \sim \mathcal{Z}^n(g_D',j)}^2}$  in SGD</td></tr><tr><td> $\tilde{p}, \pi$ </td><td>probability measure of posterior and prior</td></tr><tr><td> $\Sigma_{\tilde{p}}, \Sigma_\pi$ </td><td>covariance matrix of Gaussian distribution</td></tr><tr><td> $\epsilon, \epsilon'$ </td><td>random error terms in gradients</td></tr><tr><td> $KL(\tilde{p}||\pi)$ </td><td>KL divergence between distributions</td></tr></table>

# B. Details of Table 1

We start out from a reinterpretation of the widely-used ADAM optimizer, which maintains moving averages of stochastic gradients and their element-wise square,

$$
\tilde {m} _ {t} = \beta_ {1} \tilde {m} _ {t - 1} + (1 - \beta_ {1}) g _ {t}, \quad \hat {m} _ {t} = \frac {\tilde {m} _ {t}}{1 - \beta_ {1} ^ {t + 1}}, \tag {13}
$$

$$
\tilde {v} _ {t} = \beta_ {2} \tilde {v} _ {t - 1} + (1 - \beta_ {2}) g _ {t} ^ {2}, \quad \hat {v} _ {t} = \frac {\tilde {v} _ {t}}{1 - \beta_ {2} ^ {t + 1}}, \tag {14}
$$

with $\beta_{1},\beta_{2}\in(0,1)$ and updates with learning rate $\alpha$ ,

$$
\theta_ {t + 1} = \theta_ {t} - \alpha \frac {\hat {m} _ {t}}{\sqrt {\hat {v} _ {t}} + \varepsilon} \tag {15}
$$

with a small constant $\varepsilon > 0$ preventing division by zero. Ignoring $\varepsilon$ and assuming $|m_{t,i}| > 0$ for the moment, we can rewrite the update direction as

$$
\frac {m _ {t}}{\sqrt {v _ {t}}} = \operatorname{sign} (m _ {t}) \sqrt {\frac {v _ {t}}{m _ {t} ^ {2}}} = \underbrace {\frac {1}{\sqrt {1 + \frac {v _ {t} - m _ {t} ^ {2}}{m _ {t} ^ {2}}}}} _ {T _ {1}} \circ \underbrace {\operatorname{sign} (m _ {t})} _ {T _ {2}}. \tag {16}
$$

Here, we divide the preconditioning into two terms. The convergnece term $T_{2}$ which modulates update size is :

$$
\operatorname{sign} \left(m _ {t}\right) = \frac {\mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} \left(g _ {j}\right)}{\left| \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} \left(g _ {j}\right) \right|} \tag {17}
$$

The alignment term $T_{1}$ which includes GSNR is :

$$
\frac {1}{\sqrt {1 + \frac {v _ {t} - m _ {t} ^ {2}}{m _ {t} ^ {2}}}} = \sqrt {\frac {1}{\frac {1}{n \cdot r _ {j}} + 1}} \tag {18}
$$

The Equation (18) can be justified by definition of GSNR using Equation (19) and Equation (20). The variance of gradient average is:

$$
v _ {t} - m _ {t} ^ {2} = \frac {\rho_ {j} ^ {2}}{n} \tag {19}
$$

and

$$
m _ {t} ^ {2} = \widetilde {g _ {j}} ^ {2} \tag {20}
$$

# C. Proof of Theorems

# C.1. Convergence Analysis

We provide the detailed derivation and proof for Theorem 3.11. From Theorem 3.9, we start with:

$$
L (\theta_ {t + 1}) \leq L (\theta_ {t}) + \underbrace {\langle \nabla L (\theta_ {t}) , \theta_ {t + 1} - \theta_ {t} \rangle} _ {T _ {1}} + \underbrace {\frac {L}{2} \| \theta_ {t + 1} - \theta_ {t} \| ^ {2}} _ {T 2}, \tag {21}
$$

where the first term, $T_{1}$ , is given by:

$$
T _ {1} = \langle \nabla L (\theta_ {t}), \theta_ {t + 1} - \theta_ {t} \rangle . \tag {22}
$$

Using our preconditioning, which we defined as:

$$
p = \frac {1}{\widetilde {g} ^ {2} + \frac {\rho^ {2}}{n}} \cdot \frac {\widetilde {g} ^ {2}}{\rho^ {2}} = \frac {n}{n + \frac {\rho^ {2}}{\widetilde {g} ^ {2}}} \cdot \frac {1}{\rho^ {2}}, \tag {23}
$$

which satysifies, given Theorem 3.10:

$$
\frac {n}{n + \frac {\rho^ {2}}{g ^ {2}}} \leq 1, \frac {1}{\rho^ {2}} \leq S _ {u} \tag {24}
$$

For $T_{2}$ , we have:

$$
T _ {2} = \frac {L}{2} \lambda^ {2} \| p \odot g _ {t} \| ^ {2} \leq \frac {L}{2} \lambda^ {2} \| S _ {u} \cdot g _ {t} \| ^ {2}, \tag {25}
$$

and its expectation satisfies, using Theorem 3.8:

$$
\mathbb {E} [ T _ {2} ] \leq \frac {L}{2} \lambda^ {2} S _ {u} ^ {2} G ^ {2}. \tag {26}
$$

For $T_{1}$ , we decompose:

$$
T _ {1} = \left\langle \nabla L (\theta_ {t}), \theta_ {t + 1} - \theta_ {t} \right\rangle = - \lambda_ {t} \left\langle \nabla L (\theta_ {t}), p \odot g _ {t} \right\rangle , \tag {27}
$$

$$
T _ {1} \leq \underbrace {- \lambda_ {t} P _ {l} \left\langle \nabla L \left(\theta_ {t}\right) \cdot g _ {t} \right\rangle} _ {T 3} + \underbrace {\lambda_ {t} \sum_ {j} \left| \left[ \nabla L \left(\theta_ {t}\right) \right] _ {j} \right| \cdot \frac {\left| g _ {t , j} \right|}{\rho_ {t , j} ^ {2}} \cdot 1 (\operatorname{sign} \left[ \left[ \nabla L \left(\theta_ {t}\right) \right] _ {j} \right] \neq \operatorname{sign} \left[ g _ {t , j} \right])} _ {T _ {4}}. \tag {28}
$$

Now, considering $T_{3}$ , we evaluate its expectation:

$$
\mathbb {E} [ T _ {3} ] = - \lambda_ {t} P _ {l} \mathbb {E} [ \langle \nabla L (\theta_ {t}), g _ {t} \rangle ], \tag {29}
$$

$P_{l}$ is lower bound of our preconditioning value. Considering $T_{4}$ :

$$
\mathbb {E} \left[ T _ {4} \right] = \lambda_ {t} \sum_ {j} \mathbb {E} \left[ \left| \left[ \nabla L \left(\theta_ {t}\right) \right] _ {j} \right| \cdot \frac {\left| g _ {t , j} \right|}{\rho_ {t , j} ^ {2}} \cdot 1 (\operatorname{sign} [ [ \nabla L \left(\theta_ {t}\right) ] ] _ {j} \neq \operatorname{sign} [ g _ {t, j} ]) \right]. \tag {30}
$$

Thus, we obtain:

$$
\mathbb {E} \left[ T _ {4} \right] = \lambda_ {t} \sum_ {j} \mathbb {E} \left[ \left| \left[ \nabla L \left(\theta_ {t}\right) \right] _ {j} \right| \cdot \frac {\left| g _ {t , j} \right|}{\rho_ {t , j} ^ {2}} \mid P (\operatorname{sign} \left[ \left[ \nabla L \left(\theta_ {t}\right) \right] _ {j} \right] \neq \operatorname{sign} \left[ g _ {t, j} \right]) \right]. \tag {31}
$$

Next, we analyze the probability term:

$$
P (\operatorname{sign} [ \nabla L (\theta_ {t}) ] _ {j} ] \neq \operatorname{sign} [ g _ {t, j} ]) \tag {32}
$$

and bound it as follows:

$$
P (\text { sign } [ \nabla L (\theta_ {t}) ] _ {j} \neq \text { sign } [ g _ {t, j} ]) \leq P (| [ \nabla L (\theta_ {t}) ] _ {j} - g _ {t, j} | \geq | [ \nabla L (\theta_ {t}) ] _ {j} |). \tag {33}
$$

Using Chebyshev's inequality:

$$
P \left(\left| [ \nabla L (\theta_ {t}) ] _ {j} - g _ {t, j} \right| \geq \left| [ \nabla L (\theta_ {t}) ] _ {j} \right|\right) \leq \frac {\operatorname{Var} \left([ \nabla L (\theta_ {t}) ] _ {j} - g _ {t , j}\right)}{\left| [ \nabla L (\theta_ {t}) ] _ {j} \right| ^ {2}} = \frac {\sigma^ {2}}{\left| [ \nabla L (\theta_ {t}) ] _ {j} \right| ^ {2}} = \frac {\rho_ {t} ^ {2} / n}{\left| [ \nabla L (\theta_ {t}) ] _ {j} \right| ^ {2}}. \tag {34}
$$

n is the number of gradient samples.

Replacing the n to step T, we bound the expectation:

$$
\mathbb {E} \left[ T _ {4} \right] \leq \lambda_ {t} \sum_ {j} \mathbb {E} \left[ \left| \left[ \nabla L \left(\theta_ {t}\right) \right] _ {j} \right| \cdot \frac {\left| g _ {t , j} \right|}{\rho_ {t , j} ^ {2}} \cdot \frac {\rho_ {t , j} ^ {2} / T}{\left| \left[ \nabla L \left(\theta_ {t}\right) \right] _ {j} \right| ^ {2}} \right] \leq \lambda_ {t} \frac {| J |}{T} \tag {35}
$$

$|J|$ is the number of parameters indicated by the size of parameter index set. Now, summing the inequalities until step T:

$$
\mathbb {E} [ L (\theta_ {t + 1}) ] \leq \mathbb {E} [ L (\theta_ {t}) ] - \lambda_ {t} P _ {l} \| \nabla L (\theta_ {t}) \| ^ {2} + \lambda_ {t} \cdot \frac {| J |}{T} + \frac {L}{2} \lambda_ {t} ^ {2} S _ {u} ^ {2} G ^ {2}. \tag {36}
$$

Rearranging:

$$
\mathbb {E} [ L (\theta_ {t + 1}) ] \leq L (\theta_ {0}) - \lambda_ {t} P _ {l} \sum_ {t = 1} ^ {T} \| \nabla L (\theta_ {t}) \| ^ {2} + T \cdot \lambda_ {t} \left(\frac {| J |}{T} + \frac {L}{2} \lambda_ {t} S _ {u} ^ {2} G ^ {2}\right). \tag {37}
$$

This results in:

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} \| \nabla L (\theta_ {t}) \| ^ {2} \leq \frac {L (\theta_ {0}) - \mathbb {E} [ L (\theta_ {t + 1}) ]}{\lambda_ {t} P _ {l} \cdot T} + \frac {1}{P _ {l}} \left(\frac {| J |}{T} + \frac {L}{2} \lambda_ {t} S _ {u} ^ {2} G ^ {2}\right). \tag {38}
$$

$$
\frac {1}{T} \sum_ {t = 1} ^ {T} \| \nabla L (\theta_ {t}) \| ^ {2} \leq \frac {L (\theta_ {0}) - \mathbb {E} [ L (\theta_ {*}) ]}{\lambda_ {t} P _ {l} \cdot T} + \frac {1}{P _ {l}} \left(\frac {| J |}{T} + \frac {L}{2} \lambda_ {t} S _ {u} ^ {2} G ^ {2}\right). \tag {39}
$$

Taking $T \to \infty$ , the convergence rate is:

$$
\mathbb {E} [ \| \nabla L (\theta_ {t}) \| ^ {2} ] \leq \frac {L (\theta_ {0}) - L (\theta_ {*})}{\lambda_ {t} P _ {l} \cdot T} + \frac {\frac {| J |}{T} + \frac {L}{2} \lambda_ {t} S _ {u} ^ {2} G ^ {2}}{P _ {l}}. \tag {40}
$$

From the final steps of our derivation, we analyze the convergence rate of the algorithm.

Taking $\lambda_T$ as:

$$
\lambda_ {T} = \sqrt {\frac {L (\theta_ {0}) - L (\theta_ {*})}{T \cdot \ell}}, \tag {41}
$$

we have:

$$
\mathbb {E} [ \| \nabla L (\theta) \| ] \leq \frac {1}{P _ {l}} \cdot \sqrt {\ell \frac {L (\theta_ {0}) - L (\theta_ {*})}{T}} + \frac {G \cdot S _ {u} ^ {2}}{2 \cdot P _ {l}} \sqrt {\ell \frac {L (\theta_ {0}) - L (\theta_ {*})}{T}} + \frac {| J |}{P _ {l} \cdot T}. \tag {42}
$$

Denoting $\frac{1}{\sqrt{\hat{T}}}$ as:

$$
\frac {1}{\sqrt {\hat {T}}} = \sqrt {\frac {L (\theta_ {0}) - L (\theta_ {*}) \cdot \ell}{T}}, \tag {43}
$$

Finally, we rewrite the bound, concluding that:

$$
\mathbb {E} [ \| \nabla L (\theta) \| ^ {2} ] \leq O \left(\frac {1}{P _ {l}} \left(1 + \frac {G \cdot S _ {u} ^ {2}}{2}\right) \frac {1}{\sqrt {\hat {T}}}\right) \quad \blacksquare \tag {44}
$$

# C.2. Proof of Theorem 3.2

We utilized (Liu et al., 2020) for proving Theorem 3.2. In one gradient descent step, the model parameter is updated by $\Delta \theta = \theta_{t + 1} - \theta_t = -\lambda p\odot g_D(\theta)$ , where $\lambda$ is the learning rate and $p$ is preconditioning. If $\lambda$ is small enough, the one-step training and test loss decrease can be approximated by:

$$
\Delta L [ D ] \approx - \Delta \theta \cdot \frac {\partial L [ D ]}{\partial \theta} + O (\lambda^ {2}) = \lambda p \odot g _ {D} (\theta) \cdot g _ {D} (\theta) + O (\lambda^ {2}), \tag {45}
$$

$$
\Delta L [ D ^ {\prime} ] \approx - \Delta \theta \cdot \frac {\partial L [ D ^ {\prime} ]}{\partial \theta} + O (\lambda^ {2}) = \lambda p \odot g _ {D} (\theta) \cdot g _ {D ^ {\prime}} (\theta) + O (\lambda^ {2}). \tag {46}
$$

Usually, there are some differences between the directions of $g_{D}(\theta)$ and $g_{D'}(\theta)$ , so statistically $\Delta L[D]$ tends to be larger than $\Delta L[D']$ , and the generalization gap would increase during training. When $\lambda \to 0$ , in one single training step, the empirical generalization gap increases by $\Delta L[D] - \Delta L[D']$ . For simplicity, we denote this quantity as:

$$
\nabla := \Delta L [ D ] - \Delta L [ D ^ {\prime} ] \approx \lambda g _ {D} (\theta) \cdot g _ {D} (\theta) - \lambda g _ {D} (\theta) \cdot g _ {D ^ {\prime}} (\theta), \tag {47}
$$

which can be further simplified as:

$$
\nabla = \lambda (p \odot \tilde {g} (\theta) + p \odot \epsilon) (\tilde {g} (\theta) + \epsilon^ {\prime}) - \lambda (p \odot \tilde {g} (\theta) + p \odot \epsilon) (\tilde {g} (\theta) + \epsilon^ {\prime}), \tag {48}
$$

$$
\nabla = \lambda (\tilde {g} (\theta) + \epsilon) (\epsilon - \epsilon^ {\prime}). \tag {49}
$$

Here, we replaced the random variables by $g_{D}(\theta)=\tilde{g}(\theta)+\epsilon$ and $g_{D'}(\theta)=\tilde{g}(\theta)+\epsilon'$ , where $\epsilon$ and $\epsilon'$ are random variables with zero mean and variance $\sigma^{2}(\theta)$ . Since $E[\epsilon']=E[\epsilon]=0$ , $\epsilon$ and $\epsilon'$ are independent. The expectation of $\nabla$ is:

$$
\mathbb {E} _ {D, D ^ {\prime} \sim \mathcal {Z} ^ {n}} (\nabla) = \mathbb {E} (\lambda p \odot \epsilon \cdot \epsilon^ {\prime}) + O (\lambda^ {2}) = \lambda \sum_ {j} p _ {j} \cdot \sigma^ {2} (\theta_ {j}) + O (\lambda^ {2}), \tag {50}
$$

where $\sigma^{2}(\theta_{j})$ is the variance of the average gradient of the parameter $\theta_{j}$ . For simplicity, when it involves a single model parameter $\theta_{j}$ , we will use only a subscript j instead of the full notation. For example, we use $\sigma_{j}^{2}$ , $r_{j}$ , and $g_{D,j}$ to denote $\sigma^{2}(\theta_{j})$ , $r(\theta_{j})$ , and $g_{D}(\theta_{j})$ , respectively.

# Expectation Analysis

Consider the expectation of $\Delta L[D]$ and $\Delta L[D']$ when $\lambda \rightarrow 0$ :

$$
\mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (\Delta L [ D ]) \approx \lambda \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (p \odot g _ {D} (\theta) \cdot g _ {D} (\theta)) = \lambda \sum_ {j} p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D, j} ^ {2}). \tag {51}
$$

$$
\mathbb {E} _ {D, D ^ {\prime} \sim \mathcal {Z} ^ {n}} (\Delta L [ D ^ {\prime} ]) = \mathbb {E} _ {D, D ^ {\prime} \sim \mathcal {Z} ^ {n}} (\Delta L [ D ] - \nabla) \approx \lambda \sum_ {j} p _ {j} (\mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D, j} ^ {2}) - \sigma_ {j} ^ {2}), \tag {52}
$$

which simplifies further as:

$$
\mathbb {E} _ {D, D ^ {\prime} \sim \mathcal {Z} ^ {n}} (\Delta L [ D ^ {\prime} ]) \approx \lambda \sum_ {j} p _ {j} (\mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D, j} ^ {2}) - \rho_ {j} ^ {2} / n). \tag {53}
$$

# Simplification of $R(\mathcal{Z},n)$

Substituting Equation (53) and Equation (51) into $R(\mathcal{Z}, n)$ , we have:

$$
R (\mathcal {Z}, n) = 1 - \frac {\sum_ {j} p _ {j} \rho_ {j} ^ {2}}{n \sum_ {j} p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D , j} ^ {2})}. \tag {54}
$$

When $r_j = \frac{\mathbb{E}_{D\sim\mathcal{Z}^n(g_D,j)}^2}{\rho^2}$ We can rewrite Equation (53) as:

$$
R (\mathcal {Z}, n) = 1 - \frac {1}{n} \sum_ {j} \frac {p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D , j} ^ {2})}{\sum_ {j ^ {\prime}} p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D ^ {\prime} , j} ^ {2})} \cdot \frac {1}{r _ {j} + \frac {1}{n}}, \tag {55}
$$

or equivalently:

$$
R (\mathcal {Z}, n) = \sum_ {j} \frac {p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D , j} ^ {2})}{\sum_ {j ^ {\prime}} p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D ^ {\prime} , j} ^ {2})} \cdot \frac {1}{\left(1 + \frac {1}{n \cdot r _ {j}}\right)}. \tag {56}
$$

# C.3. Generalization Analysis

# C.3.1. PAC BAYES BOUND AND PRECONDITIONING

Theorem C.1 (PAC-Bayes bound). Let $D \sim Z^{n}$ be a dataset sampled i.i.d. from a data distribution $\mathcal{Z}.R(\theta)$ is the population risk and $L(\theta)$ is empirical risk. Assume that the loss function $L(\theta)$ is bounded in $[0,C]$ for some constant C > 0. For any $\lambda > 0$ , with probability at least $1 - \delta$ over the draw of D, and for any data-dependent distribution $\tilde{p}$ over parameters $\theta$ , the following PAC-Bayes bound holds:

$$
\mathbb {E} _ {\theta \sim \tilde {p}} [ R (\theta) ] \leq \underbrace {\mathbb {E} _ {\theta \sim \tilde {p}} [ L (\theta) ]} _ {T _ {1}} + \frac {\lambda C ^ {2}}{8 n} + \underbrace {\frac {\mathrm{KL} (\tilde {p} \| \pi) + \log \frac {1}{\delta}}{\lambda}} _ {T _ {2}}.
$$

Here, $R(\theta)$ denotes the expected loss over the true data distribution. The SAM algorithm primarily focuses on minimizing the $T_{1}$ term in Theorem C.1, following the inequality:

$$
L _ {\mathcal {D}} (\theta) \leq \mathbb {E} _ {\epsilon \sim \mathcal {N} (0, \rho)} [ L _ {\mathcal {D}} (\theta + \epsilon) ] \leq \max _ {\| \epsilon \| _ {2} \leq \rho} [ L _ {\mathcal {D}} (\theta + \epsilon) ]. \tag {57}
$$

In contrast, our method simultaneously minimizes both the $T_{1}$ and $T_{2}$ terms. First, we determine a preconditioning vector q to reduce $\mathbb{E}_{S}\mathbb{E}_{\theta\sim\tilde{p}}[L(\theta)]$ . To minimize variance-induced error, we adopt the variance adaptation factor introduced in the SVAG optimizer(Balles & Hennig, 2018), and solve:

$$
\mathbb {E} \big [ \| q \odot g - \mathbb {E} [ g ] \| _ {2} ^ {2} \big ] = \sum_ {j} q _ {j} ^ {2} \mathbb {E} [ g _ {j} ^ {2} ] - 2 q _ {j} \mathbb {E} [ g ] ^ {2} + \mathbb {E} [ g ] ^ {2}. \tag {58}
$$

Minimizing this expression yields the optimal preconditioning:

$$
q _ {j} = \frac {\mathbb {E} [ g ] ^ {2}}{\mathbb {E} [ g _ {j} ^ {2} ]}.
$$

Next, assume $\tilde{p} = \mathcal{N}(\theta_{t + 1},\Sigma_{\tilde{p}})$ and $\pi = \mathcal{N}(\theta_t,\Sigma_\pi)$ , with both covariances defined as:

$$
\pmb {\Sigma} _ {\tilde {p}} = \mathrm{diag} (q _ {1} ^ {2} \cdot \rho_ {1} ^ {2}, q _ {2} ^ {2} \cdot \rho_ {2} ^ {2}, \ldots , q _ {| J |} ^ {2} \cdot \rho_ {| J |} ^ {2}), \quad \pmb {\Sigma} _ {\pi} = \mathrm{diag} (\rho_ {1} ^ {2}, \rho_ {2} ^ {2}, \ldots , \rho_ {| J |} ^ {2}),
$$

Here, the prior $\pi$ can be treated as a data-driven prior which is approximated with stochastic gradient descent using all data excluding the current mini-batch. Assuming the variances do not significantly change between steps. Then the KL divergence can be written as follows:

$$
K L (\tilde {p} \| \pi) = \frac {1}{2} \left[ \sum_ {i \in J} \frac {q _ {i} ^ {2} \cdot \rho_ {i} ^ {2}}{\rho_ {i} ^ {2}} + \sum_ {i \in J} \frac {\left(\theta_ {t + 1 , i} - \theta_ {t , i}\right) ^ {2}}{\rho_ {i} ^ {2}} - | J | + \sum_ {i \in J} \log \left(\frac {\rho_ {i} ^ {2}}{q _ {i} ^ {2} \cdot \rho_ {i} ^ {2}}\right) \right] \tag {59}
$$

The gradient of this KL term with respect to $\theta_{t}$ is:

$$
[ \nabla_ {\theta_ {t}} K L (\tilde {p} \| \pi) ] _ {j} = - \frac {(\theta_ {t + 1 , j} - \theta_ {t , j})}{\rho_ {j} ^ {2}} = \frac {\mathbb {E} [ g _ {j} ] ^ {2}}{\mathbb {E} [ g _ {j} ^ {2} ]} \cdot \frac {g _ {j , t}}{\rho_ {j} ^ {2}} = \underbrace {\frac {1}{\mathbb {E} [ g _ {j} ^ {2} ]} \cdot \frac {\mathbb {E} [ g _ {j , t} ] ^ {2}}{\rho_ {j} ^ {2}}} _ {\text {GENIE}} \cdot g _ {j, t}. \tag {60}
$$

This shows that minimizing the KL divergence in the PAC-Bayes bound via gradient descent naturally leads to the same preconditioning structure. Therefore, our method considers both sharpness (through variance adaptation) and generalization (via KL divergence minimization), whereas SAM focuses solely on sharpness.

It is important to note that our gradient computation is taken with respect to the prior mean $\theta_{t}$ , rather than the posterior mean $\theta_{t+1}$ . Due to the asymmetric nature of the forward KL divergence $\mathrm{KL}(\tilde{p}\|\pi)$ , this choice induces a mode-covering behavior rather than mode-seeking. As optimization proceeds, the prior distribution $\pi$ is iteratively adapted to cover a broader region of the risk landscape, effectively reducing over-concentration around a single mode. This mechanism aligns with the argument in Risk Extrapolation (Krueger et al., 2021), where covering a wider set of hypotheses can improve out-of-distribution generalization. Consequently, although the PAC-Bayes bound is originally derived under an i.i.d. assumption, this mode-covering property enables the prior to capture more diverse risk regions, thereby enhancing robustness under distribution shift.

# C.3.2. OSGR BASED ANALYSIS

From Section 3.2.1, the OSGR of our method is given by:

$$
R _ {\text { ours }} = 1 - \frac {1}{n} \sum_ {j} \frac {1}{\sum_ {j ^ {\prime}} \left(r _ {j ^ {\prime}} + \frac {1}{n}\right)} = 1 - \frac {1}{n \mathbb {E} _ {j \sim J} \left(r _ {j} + \frac {1}{n}\right)}. \tag {61}
$$

Similarly, from Theorem 3.1, the OSGR of SGD is given by:

$$
R _ {\mathrm{sgd}} = 1 - \frac {1}{n} \sum_ {j} \frac {\mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} [ g _ {j} ^ {2} ]}{\sum_ {j ^ {\prime}} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} [ g _ {j ^ {\prime}} ^ {2} ]} \cdot \frac {1}{r _ {j} + \frac {1}{n}}. \tag {62}
$$

Replacing the term $\sum_{j}\frac{\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j}^{2}]}{\sum_{j'}\mathbb{E}_{D\sim\mathcal{Z}^{n}}[g_{j'}^{2}]}$ to $\sum W_{j}=1$ which represents a weighted average, we rewrite it as:

$$
R _ {\mathrm{sgd}} = 1 - \frac {1}{n} \sum_ {j \in J} W _ {j} \left(\frac {1}{r _ {j} + \frac {1}{n}}\right). \tag {63}
$$

If we assume uniform weight $W_{j} = \frac{1}{|J|}$ , by Jensen's inequality:

$$
0 \leq 1 - \frac {1}{n} \sum_ {j \in J} W _ {j} \left(\frac {1}{r _ {j} + \frac {1}{n}}\right) \leq 1 - \frac {1}{n \mathbb {E} _ {j \in J} \left(r _ {j} + \frac {1}{n}\right)} \leq 1. \tag {64}
$$

Thus, we conclude:

$$
0 \leq R _ {\mathrm{sgd}} \leq R _ {\text {ours}} \leq 1. \tag {65}
$$

In the same way, with any preconditioning,

$$
R (\mathcal {Z}, n) = 1 - \frac {1}{n} \sum_ {j} \frac {p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D , j} ^ {2})}{\sum_ {j ^ {\prime}} p _ {j} \mathbb {E} _ {D \sim \mathcal {Z} ^ {n}} (g _ {D ^ {\prime} , j} ^ {2})} \cdot \frac {1}{r _ {j} + \frac {1}{n}}, \tag {66}
$$

Replacing the term $\sum_{j}\frac{p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D,j}^{2})}{\sum_{j^{\prime}}p_{j}\mathbb{E}_{D\sim\mathcal{Z}^{n}}(g_{D^{\prime},j}^{2})}$ to $\sum W_{j} = 1$ with $W_{j} = \frac{1}{|J|}$ , which represents a average, we rewrite it as:

$$
R _ {\text { precondition }} = 1 - \frac {1}{n} \sum_ {j \in J} W _ {j} \left(\frac {1}{r _ {j} + \frac {1}{n}}\right). \tag {67}
$$

Also by Jensen's inequality, we obtain:

$$
0 \leq R _ {\text { precondition }} \leq R _ {\text { ours }} \leq 1. \tag {68}
$$

Consequently, this result establishes that our method achieves the highest OSGR value among preconditioning methods such as Adam, RMSprop, and SVAG(Balles & Hennig, 2018).

However, the assumption of uniform weights $W_{j} = \frac{1}{|J|}$ can be overly restrictive. In practice, preconditioning methods aim to balance the contribution of parameter updates, which becomes especially important when there exists a strong imbalance in the GSNR distribution. We consider the case where a dominant coordinate exists, denoted as $j_{max} = \arg\max_{j \in J} r_{j}$ , and define the remaining coordinates as $J' = J \setminus \{j_{max}\}$ , with $j_{max}' = \arg\max_{j \in J'} r_{j}$ , such that $r_{j_{max}'} \ll r_{j_{max}}$ .

To demonstrate that our method still yields a higher OSGR under such imbalance, we consider the difference:

$$
n (R _ {\text { ours }} - R _ {\text { precondition }}) = \sum_ {j \in J} W _ {j} \left(\frac {1}{r _ {j} + \frac {1}{n}}\right) - \frac {1}{\mathbb {E} _ {j \in J} \left(r _ {j} + \frac {1}{n}\right)}. \tag {69}
$$

For sufficiently large n, the $\frac{1}{n}$ terms can be neglected:

$$
n (R _ {\text { ours }} - R _ {\text { precondition }}) \approx \sum_ {j \in J} W _ {j} \left(\frac {1}{r _ {j}}\right) - \frac {1}{\mathbb {E} _ {j \in J} (r _ {j})}. \tag {70}
$$

Separating the contribution of the dominant coordinate $j_{max}$ , we have:

$$
n (R _ {\text { ours }} - R _ {\text { precondition }}) = \sum_ {j \in J ^ {\prime}} W _ {j} \left(\frac {1}{r _ {j}}\right) + W _ {j _ {\max}} \left(\frac {1}{r _ {j _ {\max}}}\right) - \frac {1}{\mathbb {E} _ {j \in J} (r _ {j})}. \tag {71}
$$

Since $\mathbb{E}_{j\in J}(r_j)\geq \frac{r_{j\max}}{|J|}$ , this difference is lower-bounded by:

$$
n (R _ {\text { ours }} - R _ {\text { precondition }}) \geq \sum_ {j \in J ^ {\prime}} W _ {j} \left(\frac {1}{r _ {j}}\right) + W _ {j _ {\max}} \left(\frac {1}{r _ {j _ {\max}}}\right) - \frac {| J |}{r _ {j _ {\max}}}. \tag {72}
$$

Further bounding the terms using $r_{j} \leq r_{j_{\max}^{\prime}}$ for all $j \in J^{\prime}$ , we obtain:

$$
\sum_ {j \in J ^ {\prime}} W _ {j} \left(\frac {1}{r _ {j}}\right) + W _ {j _ {\max}} \left(\frac {1}{r _ {j _ {\max}}}\right) - \frac {| J |}{r _ {j _ {\max}}} \geq (1 - W _ {j _ {\max}}) \left(\frac {1}{r _ {j _ {\max} ^ {\prime}}}\right) + W _ {j _ {\max}} \left(\frac {1}{r _ {j _ {\max}}}\right) - \frac {| J |}{r _ {j _ {\max}}}. \tag {73}
$$

Therefore, if the following condition holds:

$$
\left(1 - W _ {j _ {\max}}\right) r _ {j _ {\max}} \geq | J | r _ {j _ {\max} ^ {\prime}}, \tag {74}
$$

then $R_{ours} \geq R_{precondition}$ , and our method guarantees superior OSGR performance compared to other preconditioning strategies. This result highlights the robustness of our formulation, particularly under skewed GSNR distributions.

# D. Implementation Details

# D.1. Training details

As introduced in the experimental section, we follow the standard training, hyperparameter search methods, and evaluation protocol proposed by DomainBed (Gulrajani & Lopez-Paz, 2021) to ensure a fair comparison. For each dataset, the models were trained for 15,000 iterations on DomainNet and 5,000 iterations on the other datasets. The search space of hyperparameters is provided in Table 8. All experiments were conducted on an NVIDIA GeForce RTX 4090 under the environment of Python 3.8.10, PyTorch 1.13.1, Torchvision 0.14.1, and CUDA 11.7.

Table 8. The search space of hyperparameters. 

<table><tr><td>PARAMETER</td><td>DEFAULT VALUE</td><td>SEARCH DISTRIBUTION</td></tr><tr><td>BATCH SIZE</td><td>32</td><td> $2^{\text{UNIFORM}(3,5.5)}$ </td></tr><tr><td>LEARNING RATE</td><td>0.015</td><td> $10^{\text{UNIFORM}(-3,-1)}$ </td></tr><tr><td>RESNET DROPOUT</td><td>0.0</td><td> $[0.0, 0.1, 0.5]$ </td></tr><tr><td>WEIGHT DECAY</td><td>0.0</td><td> $10^{\text{UNIFORM}(-6,-2)}$ </td></tr></table>

# D.2. Pseudo code

Algorithm 2 Unified Algorithm for RMSProp, Adam, and GENIE   
Input: Mini-batches $\{B_{t}\}_{t=1}^{T}$ , learning rate $\alpha$ , total steps T
Preconditioning Parameters: $\beta, \beta_{1}, \beta_{2} \in [0,1]$ , noise scale $\sigma, m_{0} \leftarrow 0, v_{0} \leftarrow 0$ Initialize: Parameters $\theta_{0}$ for t = 1 to T do
▷ Compute Gradient: Sample mini-batch $B_{t}$ and compute

$$
g _ {t} = \nabla \mathcal {L} (\theta_ {t}; \mathcal {B} _ {t}).
$$

▷ Compute Preconditioned Gradient:

RMSProp:

$$
v _ {t} \leftarrow \beta v _ {t - 1} + (1 - \beta) g _ {t} ^ {2}, \quad \tilde {g} _ {t} \leftarrow \frac {g _ {t}}{\sqrt {v _ {t}} + \epsilon}.
$$

Adam:

$$
\begin{array}{l} m _ {t} \leftarrow \beta_ {1} m _ {t - 1} + (1 - \beta_ {1}) g _ {t}, v _ {t} \leftarrow \beta_ {2} v _ {t - 1} + (1 - \beta_ {2}) g _ {t} ^ {2}, \\ \hat {m} _ {t} \leftarrow \frac {m _ {t}}{1 - \beta_ {1} ^ {t}}, \quad \hat {v} _ {t} \leftarrow \frac {v _ {t}}{1 - \beta_ {2} ^ {t}}, \quad \tilde {g} _ {t} \leftarrow \frac {\hat {m} _ {t}}{\sqrt {\hat {v} _ {t}} + \epsilon}. \\ \end{array}
$$

GENIE:

$$
\begin{array}{l} m _ {t} \leftarrow \beta m _ {t - 1} + (1 - \beta) g _ {t}, v _ {t} \leftarrow \beta v _ {t - 1} + (1 - \beta) g _ {t} ^ {2}, \\ \sigma_ {t} ^ {2} = v _ {t} - m _ {t} ^ {2}, \quad \mathbf {r} _ {t} = \tanh (\frac {1}{\sigma_ {t} ^ {2}}) m _ {t} ^ {2} \\ \hat {g} _ {t} \leftarrow \frac {m _ {t}}{1 - \beta^ {t}} \cdot \frac {1}{v _ {t}} \cdot \mathbf {r} _ {t}, \\ \mathrm{Noise} _ {t} \leftarrow \xi_ {t} \big [ 1 - \tanh (\frac {1}{\sigma_ {t} ^ {2}}) \big ], \quad \xi_ {t} \sim \mathcal {N} (0, \sigma^ {2}), \\ \tilde {g} _ {t} \leftarrow \hat {g} _ {t} + \text { Noise } _ {t}. \\ \tilde {g} _ {t} \leftarrow \tilde {g} _ {t} \odot M, M _ {j} \sim \operatorname{Bernoulli} (p) \\ \end{array}
$$

▷ Update Parameters:

$$
\theta_ {t + 1} \leftarrow \theta_ {t} - \alpha \tilde {g} _ {t}.
$$

end for

Output: Final parameters $\theta_{T+1}$ for RMSProp, Adam, and GENIE.

# D.3. Code

```python
[ caption={Python implementation of preconditioning updates.}, label={lst:preconditioning}] def _initialize_preconditioning(self, current_state):
    self.prev_state = current_state
    self.gmean = {k: torch.zeros_like(param) for k, param in
    self.network.named_parameters()}
    self.ge2 = {k: torch.zeros_like(param) for k, param in
    self.network.named_parameters()}
    self.scale = 0.0

def _update_preconditioning(self, lr, moving_avg):
    grad_sgd = {}
    pgrad = {}
    pGsnr = {}

# Update scale factors 
```

```python
self.scale = (moving_avg * self.scale + 1.0)
scale1 = (1 - moving_avg) * self.scale
scale2 = 2.0 - scale1
rho = (1.0 - moving_avg) * scale2 / ((1.0 + moving_avg) * scale1)

with torch.no_grad():
    # Update gradients and variance
    for k, param in self.network.named_parameters():
    delta = param.grad.data.detach()
    self.gmean[k] = self.gmean[k] * moving_avg + delta * (1.0 - moving_avg)
    self.ge2[k] = self.ge2[k] * moving_avg + (delta ** 2) * (1.0 - moving_avg)

    gm = self.gmean[k] / scale1
    ge2 = self.ge2[k] / scale1
    var = ge2 - gm.square()
    var /= (1.0 - rho)
    var = torch.where(var > 0.0, var, torch.zeros_like(var) + 1e-8)

    invvar = torch.clamp(1 / var, min=0.0, max=10.0)
    mvar = rho * var
    mvar = torch.where(mvar > 0.0, mvar, torch.zeros_like(mvar) + 1e-8)

    # Preconditioned gradient scaling
    tanh_invvar = torch.tanh(invvar)
    pGsnr[k] = (1.0 / (1.0 + mvar / (gm.square() + 1e-8))) * tanh_invvar

    # Add noise for stochastic gradient adjustment
    noise_scale = torch.sum(tanh_invvar * torch.abs(gm) *
    (1.0 / (1.0 + mvar / (gm.square() + 1e-8)))) /
    torch.sum(tanh_invvar)
    noise = torch.normal(torch.zeros_like(delta), torch.ones_like(delta)) * noise_scale
    grad_sgd[k] = (1 - tanh_invvar) * noise

    # Compute preconditioned gradients
    for k, param in self.network.named_parameters():
    pgrad[k] = self.gmean[k] / scale1 * pGsnr[k].view_as(param)

    # Apply gradients with dropout-based masking
    for k, param in self.network.named_parameters():
    mask = (torch.rand_like(param) > self.hparams['p']).float() / (1 - self.hparams['p'])
    self.prev_state[k] -= (pgrad[k] + grad_sgd[k]) * mask * lr 
```

# E. Experimental Details and Results

# E.1. Dataset

This section introduces five representative DG datasets utilized in this paper.

- PACS (Li et al., 2017): This dataset includes 4 different domain styles—Photo, Art Painting, Cartoon, and Sketch. Each domain contains 7 categories and consists of 9,991 images. It is well-suited for evaluating generalization performance across style variations.   
- OfficeHome (Venkateswara et al., 2017): This dataset consists of 4 different domain styles—Art, Clipart, Product, and Real-world. Each domain includes 65 categories, with a total of 15,588 samples.   
- VLCS (Fang et al., 2013): Derived from four distinct datasets—Caltech101, LabelMe, VOC2007, and SUN09—this dataset includes 5 shared classes across domains, containing a total of 10,729 images. It is ideal for evaluating distributional differences between datasets.   
- Terra Incognita (Beery et al., 2018): Comprising photographs of wildlife, this dataset is collected from 4 different

locations—L100, L38, L43, and L46. It includes 10 categories and 24,788 samples and is commonly used to measure model generalization performance in real-world scenarios.

\- DomainNet (Peng et al., 2019): This large-scale dataset includes 6 domains—Clipart, Infograph, Painting, Quickdraw, Real, and Sketch. It comprises 345 categories and a total of 586,575 samples.

# E.2. Detailed Results: Comparison with Existing Optimizers

We compare the performance of existing optimizers with our proposed GENIE optimizer on five datasets from the same DG benchmark. The dataset-specific experimental results are presented in Table 9, Table 10, Table 11, Table 12, and Table 13. Additionally, we analyze the performance and training time as the number of iterations increases, as shown in Table 14, Table 15, Table 16.

Table 9. Comparison Results on the PACS Dataset. 

<table><tr><td>OPTIMIZER</td><td>ART</td><td>CARTOON</td><td>PHOTO</td><td>SKETCH</td><td>AVG.</td></tr><tr><td> $ADAM^*$ </td><td>88.0±1.2</td><td>79.7±0.5</td><td>96.7±0.4</td><td>72.7±0.9</td><td>84.3</td></tr><tr><td> $ADAMW^*$ </td><td>84.1±1.5</td><td>80.7±1.2</td><td>96.9±0.4</td><td>72.8±0.6</td><td>83.6</td></tr><tr><td> $SGD^*$ </td><td>85.1±0.4</td><td>76.0±0.3</td><td>98.3±0.4</td><td>60.3±6.1</td><td>79.9</td></tr><tr><td> $YOGI^*$ </td><td>84.4±1.7</td><td>79.7±0.6</td><td>95.8±0.3</td><td>65.1±1.5</td><td>81.2</td></tr><tr><td> $ADABELIEF^*$ </td><td>85.4±2.2</td><td>80.4±1.1</td><td>97.4±0.7</td><td>75.1±1.4</td><td>84.6</td></tr><tr><td> $ADAHESSIAN^*$ </td><td>88.4±0.6</td><td>80.0±0.9</td><td>97.7±0.4</td><td>71.7±4.1</td><td>84.5</td></tr><tr><td> $SAM^*$ </td><td>85.7±1.2</td><td>81.0±1.4</td><td>97.1±0.2</td><td>77.4±1.8</td><td>85.3</td></tr><tr><td> $GAM^*$ </td><td>85.9±0.9</td><td>81.3±1.6</td><td>98.2±0.4</td><td>79.0±2.1</td><td>86.1</td></tr><tr><td> $FAD^*$ </td><td>88.5±0.5</td><td>83.0±0.8</td><td>98.4±0.2</td><td>82.8±0.9</td><td>88.2</td></tr><tr><td>GENIE (OURS)</td><td>88.7±0.7</td><td>82.8±1.3</td><td>98.5±0.1</td><td>81.3±0.4</td><td>87.8</td></tr></table>

Table 10. Comparison Results on the VLCS Dataset. 

<table><tr><td>OPTIMIZER</td><td>CALTECH</td><td>LABELME</td><td>SUN</td><td>VOC</td><td>AVG.</td></tr><tr><td> $ADAM^*$ </td><td>98.9±0.4</td><td>65.9±1.5</td><td>71.0±1.6</td><td>74.5±2.0</td><td>77.3</td></tr><tr><td> $ADAMW^*$ </td><td>98.3±0.1</td><td>65.1±1.7</td><td>70.9±1.3</td><td>75.2±1.5</td><td>77.4</td></tr><tr><td> $SGD^*$ </td><td>98.4±0.2</td><td>64.7±0.7</td><td>72.5±0.8</td><td>76.6±0.8</td><td>78.1</td></tr><tr><td> $YOGI^*$ </td><td>98.1±0.7</td><td>63.9±1.2</td><td>72.5±1.6</td><td>75.7±1.2</td><td>77.6</td></tr><tr><td> $ADABELIEF^*$ </td><td>98.0±0.1</td><td>63.9±0.4</td><td>73.4±1.0</td><td>78.2±1.8</td><td>78.4</td></tr><tr><td> $ADAHESSIAN^*$ </td><td>99.1±0.3</td><td>65.0±1.7</td><td>72.7±1.3</td><td>77.7±1.0</td><td>78.6</td></tr><tr><td> $SAM^*$ </td><td>98.5±1.0</td><td>66.2±1.6</td><td>72.0±1.0</td><td>76.1±1.0</td><td>78.2</td></tr><tr><td> $GAM^*$ </td><td>98.8±0.6</td><td>65.1±1.2</td><td>72.9±1.0</td><td>77.2±1.9</td><td>78.5</td></tr><tr><td> $FAD^*$ </td><td>99.1±0.5</td><td>66.8±0.9</td><td>73.6±1.0</td><td>76.1±1.3</td><td>78.9</td></tr><tr><td>GENIE (OURS)</td><td>99.3±0.3</td><td>67.2±1.5</td><td>76.6±0.3</td><td>79.7±0.8</td><td>80.7</td></tr></table>

Table 11. Comparison Results on the OfficeHome Dataset. 

<table><tr><td>OPTIMIZER</td><td>ART</td><td>CLIPART</td><td>PRODUCT</td><td>REAL-WORLD</td><td>AVG.</td></tr><tr><td> $ADAM^*$ </td><td>63.9±0.8</td><td>48.1±0.6</td><td>77.0±0.9</td><td>81.8±1.6</td><td>67.6</td></tr><tr><td> $ADAMW^*$ </td><td>66.1±0.7</td><td>48.7±0.6</td><td>76.6±0.8</td><td>83.6±0.4</td><td>68.8</td></tr><tr><td> $SGD^*$ </td><td>65.3±0.8</td><td>48.8±1.4</td><td>76.7±0.3</td><td>83.0±0.7</td><td>68.5</td></tr><tr><td> $YOGI^*$ </td><td>63.5±1.0</td><td>49.2±1.2</td><td>76.2±0.5</td><td>84.5±0.6</td><td>68.3</td></tr><tr><td> $ADABELIEF^*$ </td><td>65.6±2.0</td><td>48.1±0.9</td><td>74.8±0.8</td><td>83.6±0.9</td><td>68</td></tr><tr><td> $ADAHESSIAN^*$ </td><td>63.0±2.9</td><td>50.0±1.4</td><td>77.7±0.8</td><td>83.0±0.5</td><td>68.4</td></tr><tr><td> $SAM^*$ </td><td>63.5±1.2</td><td>48.6±0.9</td><td>77.0±0.8</td><td>82.9±1.3</td><td>68</td></tr><tr><td> $GAM^*$ </td><td>63.0±1.2</td><td>49.8±0.5</td><td>77.6±0.6</td><td>82.4±1.0</td><td>68.2</td></tr><tr><td> $FAD^*$ </td><td>63.5±1.0</td><td>50.3±0.8</td><td>78.0±0.4</td><td>85.0±0.6</td><td>69.2</td></tr><tr><td>GENIE (OURS)</td><td>66.2±0.5</td><td>55.0±0.4</td><td>77.5±0.4</td><td>80.0±0.5</td><td>69.7</td></tr></table>

Table 12. Comparison Results on the TerraIncognita Dataset. 

<table><tr><td>OPTIMIZER</td><td>L100</td><td>L38</td><td>L43</td><td>L46</td><td>AVG.</td></tr><tr><td> $ADAM^*$ </td><td> $42.2±3.4$ </td><td> $40.7±1.2$ </td><td> $59.9±0.2$ </td><td> $35.0±2.8$ </td><td>44.4</td></tr><tr><td> $ADAMW^*$ </td><td> $44.2±6.8$ </td><td> $39.8±1.9$ </td><td> $60.3±2.0$ </td><td> $36.6±1.8$ </td><td>45.2</td></tr><tr><td> $SGD^*$ </td><td> $41.8±5.8$ </td><td> $39.8±3.9$ </td><td> $60.5±2.2$ </td><td> $37.5±1.1$ </td><td>44.9</td></tr><tr><td> $YOGI^*$ </td><td> $43.9±2.2$ </td><td> $42.5±2.6$ </td><td> $60.5±1.1$ </td><td> $34.8±1.6$ </td><td>45.4</td></tr><tr><td> $ADABELIEF^*$ </td><td> $42.6±6.7$ </td><td> $43.0±2.0$ </td><td> $60.2±1.3$ </td><td> $35.1±0.3$ </td><td>45.2</td></tr><tr><td> $ADAHESSIAN^*$ </td><td> $42.5±4.8$ </td><td> $39.5±1.0$ </td><td> $58.4±2.6$ </td><td> $37.3±0.8$ </td><td>44.4</td></tr><tr><td> $SAM^*$ </td><td> $42.9±3.5$ </td><td> $43.0±2.2$ </td><td> $60.5±1.6$ </td><td> $36.4±1.2$ </td><td>45.7</td></tr><tr><td> $GAM^*$ </td><td> $42.2±2.6$ </td><td> $42.9±1.7$ </td><td> $60.2±1.8$ </td><td> $35.5±0.7$ </td><td>45.2</td></tr><tr><td> $FAD^*$ </td><td> $44.3±2.2$ </td><td> $43.5±1.7$ </td><td> $60.9±2.0$ </td><td> $34.1±0.5$ </td><td>45.7</td></tr><tr><td>GENIE (OURS)</td><td> $55.2±4.8$ </td><td> $47.5±2.1$ </td><td> $59.2±0.4$ </td><td> $45.9±1.0$ </td><td>52.0</td></tr></table>

Table 13. Comparison Results on the DomainNet Dataset. 

<table><tr><td>OPTIMIZER</td><td>CLIP</td><td>INFO</td><td>PAINT</td><td>QUICK</td><td>REAL</td><td>SKETCH</td><td>AVG.</td></tr><tr><td>ADAM*</td><td> $63.0 \pm 0.3$ </td><td> $20.2 \pm 0.4$ </td><td> $49.1 \pm 0.1$ </td><td> $13.0 \pm 0.3$ </td><td> $62.0 \pm 0.4$ </td><td> $50.7 \pm 0.1$ </td><td>43.0</td></tr><tr><td>ADAMW*</td><td> $63.0 \pm 0.6$ </td><td> $20.6 \pm 0.2$ </td><td> $49.6 \pm 0.0$ </td><td> $13.0 \pm 0.2$ </td><td> $63.6 \pm 0.2$ </td><td> $50.4 \pm 0.1$ </td><td>43.4</td></tr><tr><td>SGD*</td><td> $61.3 \pm 0.2$ </td><td> $20.4 \pm 0.2$ </td><td> $49.4 \pm 0.2$ </td><td> $12.6 \pm 0.1$ </td><td> $65.7 \pm 0.0$ </td><td> $49.6 \pm 0.2$ </td><td>43.2</td></tr><tr><td>YOGI*</td><td> $63.3 \pm 0.1$ </td><td> $20.6 \pm 0.1$ </td><td> $50.1 \pm 0.3$ </td><td> $13.2 \pm 0.3$ </td><td> $62.8 \pm 0.1$ </td><td> $51.0 \pm 0.2$ </td><td>43.5</td></tr><tr><td>ADABELIEF*</td><td> $63.5 \pm 0.2$ </td><td> $20.5 \pm 0.1$ </td><td> $50.0 \pm 0.3$ </td><td> $13.2 \pm 0.3$ </td><td> $63.1 \pm 0.1$ </td><td> $50.7 \pm 0.1$ </td><td>43.5</td></tr><tr><td>ADAHESSIAN*</td><td> $63.3 \pm 0.2$ </td><td> $21.4 \pm 0.1$ </td><td> $50.8 \pm 0.3$ </td><td> $13.6 \pm 0.1$ </td><td> $65.7 \pm 0.1$ </td><td> $51.4 \pm 0.2$ </td><td>44.4</td></tr><tr><td>SAM*</td><td> $63.3 \pm 0.1$ </td><td> $20.3 \pm 0.3$ </td><td> $50.0 \pm 0.3$ </td><td> $13.6 \pm 0.2$ </td><td> $63.6 \pm 0.3$ </td><td> $49.6 \pm 0.4$ </td><td>43.4</td></tr><tr><td>GAM*</td><td> $63.0 \pm 0.5$ </td><td> $20.2 \pm 0.2$ </td><td> $50.3 \pm 0.1$ </td><td> $13.2 \pm 0.3$ </td><td> $64.5 \pm 0.2$ </td><td> $51.6 \pm 0.5$ </td><td>43.8</td></tr><tr><td>FAD*</td><td> $64.1 \pm 0.3$ </td><td> $21.9 \pm 0.2$ </td><td> $50.6 \pm 0.3$ </td><td> $14.2 \pm 0.4$ </td><td> $63.6 \pm 0.1$ </td><td> $52.2 \pm 0.2$ </td><td>44.4</td></tr><tr><td>GENIE (OURS)</td><td> $62.5 \pm 0.5$ </td><td> $21.3 \pm 0.4$ </td><td> $50.0 \pm 0.4$ </td><td> $14.0 \pm 0.4$ </td><td> $64.0 \pm 0.7$ </td><td> $52.6 \pm 0.8$ </td><td>44.1</td></tr></table>

Table 14. Comparison of Optimizers on PACS Dataset Across Iterations. 

<table><tr><td rowspan="2">OPTIMIZER</td><td rowspan="2">ITERATION</td><td colspan="4">TRAINING TIME (/S)</td><td colspan="4">ACCURACY</td><td colspan="2">AVG.</td></tr><tr><td>[0]</td><td>[1]</td><td>[2]</td><td>[3]</td><td>[0]</td><td>[1]</td><td>[2]</td><td>[3]</td><td>TIME</td><td>ACC</td></tr><tr><td rowspan="3">SGD</td><td>5000</td><td>1785</td><td>1819</td><td>1823</td><td>1848</td><td>73.4</td><td>61.2</td><td>96</td><td>48.4</td><td>1819</td><td>69.8</td></tr><tr><td>10000</td><td>3570</td><td>3643</td><td>3646</td><td>3749</td><td>76.8</td><td>65.1</td><td>97.4</td><td>56.1</td><td>3652</td><td>73.9</td></tr><tr><td>150000</td><td>5371</td><td>5478</td><td>5465</td><td>5609</td><td>77.6</td><td>67.1</td><td>97.8</td><td>60.9</td><td>5481</td><td>75.9</td></tr><tr><td rowspan="3">ADAM</td><td>5000</td><td>1672</td><td>1668</td><td>1707</td><td>1714</td><td>86.2</td><td>78.2</td><td>95.7</td><td>76.6</td><td>1690</td><td>84.2</td></tr><tr><td>10000</td><td>3321</td><td>3348</td><td>3392</td><td>3408</td><td>86.2</td><td>81.7</td><td>95.7</td><td>80.8</td><td>3367</td><td>86.1</td></tr><tr><td>150000</td><td>4989</td><td>5030</td><td>5067</td><td>5092</td><td>80.2</td><td>81.7</td><td>95.5</td><td>80.8</td><td>5045</td><td>84.6</td></tr><tr><td rowspan="3">SAM</td><td>5000</td><td>2661</td><td>2717</td><td>2729</td><td>2689</td><td>84.9</td><td>74</td><td>98.2</td><td>72.6</td><td>2699</td><td>82.4</td></tr><tr><td>10000</td><td>5378</td><td>5425</td><td>5479</td><td>5392</td><td>87.1</td><td>75.7</td><td>98.1</td><td>73.2</td><td>5419</td><td>83.5</td></tr><tr><td>150000</td><td>8113</td><td>8125</td><td>8228</td><td>8073</td><td>87.2</td><td>77.1</td><td>98.4</td><td>73.8</td><td>8135</td><td>84.1</td></tr><tr><td rowspan="3">GENIE(OURS)</td><td>5000</td><td>1862</td><td>2017</td><td>1532</td><td>1419</td><td>89.3</td><td>84.1</td><td>98.7</td><td>81.6</td><td>1708</td><td>88.4</td></tr><tr><td>10000</td><td>3732</td><td>4054</td><td>3065</td><td>2836</td><td>87.6</td><td>80.9</td><td>98.4</td><td>81.6</td><td>3422</td><td>87.1</td></tr><tr><td>150000</td><td>5607</td><td>6077</td><td>4586</td><td>4256</td><td>88.2</td><td>80.1</td><td>98.4</td><td>80.9</td><td>5132</td><td>86.9</td></tr></table>

Table 15. Comparison of Optimizers on VLCS Dataset Across Iterations. 

<table><tr><td rowspan="2">OPTIMIZER</td><td rowspan="2">ITERATION</td><td colspan="4">TRAINING TIME (/S)</td><td colspan="4">ACCURACY</td><td colspan="2">AVG.</td></tr><tr><td>[0]</td><td>[1]</td><td>[2]</td><td>[3]</td><td>[0]</td><td>[1]</td><td>[2]</td><td>[3]</td><td>TIME</td><td>ACC</td></tr><tr><td rowspan="3">SGD</td><td>5000</td><td>9154</td><td>4947</td><td>9315</td><td>9119</td><td>97</td><td>61.9</td><td>73.3</td><td>74.6</td><td>8134</td><td>76.7</td></tr><tr><td>10000</td><td>18311</td><td>10054</td><td>18641</td><td>18312</td><td>97.6</td><td>60.3</td><td>72.7</td><td>77.2</td><td>16330</td><td>77</td></tr><tr><td>150000</td><td>27732</td><td>14956</td><td>27918</td><td>27419</td><td>98.1</td><td>62.4</td><td>72.6</td><td>77.8</td><td>24506</td><td>77.7</td></tr><tr><td rowspan="3">ADAM</td><td>5000</td><td>9087</td><td>4900</td><td>9154</td><td>9210</td><td>98.1</td><td>63</td><td>73</td><td>73.8</td><td>8088</td><td>77</td></tr><tr><td>10000</td><td>18148</td><td>10024</td><td>18404</td><td>18312</td><td>98.1</td><td>63</td><td>73</td><td>73.8</td><td>16222</td><td>77</td></tr><tr><td>150000</td><td>27532</td><td>14963</td><td>27615</td><td>27387</td><td>98.1</td><td>63</td><td>73</td><td>73.8</td><td>24374</td><td>77</td></tr><tr><td rowspan="3">SAM</td><td>5000</td><td>9544</td><td>5611</td><td>9523</td><td>9555</td><td>99</td><td>63.5</td><td>74.6</td><td>80.5</td><td>8558</td><td>79.4</td></tr><tr><td>10000</td><td>19068</td><td>11203</td><td>18984</td><td>18749</td><td>98.9</td><td>64.1</td><td>76.1</td><td>81.9</td><td>17001</td><td>80.3</td></tr><tr><td>150000</td><td>28510</td><td>16817</td><td>28772</td><td>27860</td><td>99</td><td>64.6</td><td>75.3</td><td>82.6</td><td>25490</td><td>80.4</td></tr><tr><td rowspan="3">GENIE(OURS)</td><td>5000</td><td>8726</td><td>4631</td><td>7118</td><td>5485</td><td>99.5</td><td>68.6</td><td>76.9</td><td>80.4</td><td>6490</td><td>81.4</td></tr><tr><td>10000</td><td>17452</td><td>9273</td><td>14235</td><td>10885</td><td>99.5</td><td>68.6</td><td>76.9</td><td>80.4</td><td>12961</td><td>81.4</td></tr><tr><td>150000</td><td>26164</td><td>13917</td><td>21396</td><td>16314</td><td>99.5</td><td>68.6</td><td>76.9</td><td>80.4</td><td>19448</td><td>81.4</td></tr></table>

Table 16. Comparison of Optimizers on OfficeHome Dataset Across Iterations. 

<table><tr><td rowspan="2">OPTIMIZER</td><td rowspan="2">ITERATION</td><td colspan="4">TRAINING TIME (/S)</td><td colspan="4">ACCURACY</td><td colspan="2">AVG.</td></tr><tr><td>[0]</td><td>[1]</td><td>[2]</td><td>[3]</td><td>[0]</td><td>[1]</td><td>[2]</td><td>[3]</td><td>TIME</td><td>ACC</td></tr><tr><td rowspan="3">SGD</td><td>5000</td><td>6,634</td><td>6,436</td><td>5,842</td><td>4,554</td><td>46.2</td><td>40.1</td><td>56.8</td><td>61.9</td><td>5,867</td><td>51.3</td></tr><tr><td>10000</td><td>13,334</td><td>12,665</td><td>11,722</td><td>8,899</td><td>56.8</td><td>46.4</td><td>70.6</td><td>76.2</td><td>11,655</td><td>62.5</td></tr><tr><td>150000</td><td>20,031</td><td>18,565</td><td>17,676</td><td>13,175</td><td>58.5</td><td>47.1</td><td>73.4</td><td>76.6</td><td>17,362</td><td>63.9</td></tr><tr><td rowspan="3">ADAM</td><td>5000</td><td>8,000</td><td>8,147</td><td>5,799</td><td>4,255</td><td>57.8</td><td>49.4</td><td>73.8</td><td>73.4</td><td>6,550</td><td>63.6</td></tr><tr><td>10000</td><td>16,260</td><td>16,602</td><td>11,605</td><td>8,379</td><td>59.8</td><td>52.2</td><td>73.8</td><td>74.8</td><td>13,212</td><td>65.2</td></tr><tr><td>150000</td><td>24,776</td><td>25,466</td><td>17,387</td><td>13,062</td><td>59.8</td><td>52.2</td><td>73.8</td><td>74.8</td><td>20,173</td><td>65.2</td></tr><tr><td rowspan="3">SAM</td><td>5000</td><td>6,639</td><td>6,598</td><td>5,884</td><td>5,151</td><td>65.4</td><td>54.2</td><td>77.1</td><td>81.1</td><td>6,068</td><td>69.4</td></tr><tr><td>10000</td><td>13,359</td><td>12,964</td><td>11,813</td><td>10,189</td><td>65.1</td><td>54.5</td><td>77.9</td><td>80.9</td><td>12,081</td><td>69.6</td></tr><tr><td>150000</td><td>20,140</td><td>18,983</td><td>17,722</td><td>14,946</td><td>65.5</td><td>55.1</td><td>78.6</td><td>80.9</td><td>17,948</td><td>70</td></tr><tr><td rowspan="3">GENIE(OURS)</td><td>5000</td><td>4,777</td><td>5,846</td><td>4,025</td><td>4,061</td><td>66.5</td><td>55.4</td><td>77.8</td><td>80.3</td><td>4,677</td><td>70</td></tr><tr><td>10000</td><td>9,553</td><td>11,624</td><td>8,062</td><td>8,211</td><td>65.8</td><td>53.3</td><td>77.8</td><td>79.9</td><td>9,363</td><td>69.2</td></tr><tr><td>150000</td><td>14,261</td><td>17,333</td><td>12,229</td><td>12,369</td><td>65.8</td><td>53.3</td><td>77.1</td><td>80.3</td><td>14,048</td><td>69.1</td></tr></table>

# E.3. Integration with DG methods

The detailed performance of previous DG methods employed with our optimizer is presented in Table 17, Table 18, Table 19, Table 20.

Table 17. Integration with existing DG algorithms on the PACS dataset. 

<table><tr><td>ALGORITHM</td><td>ART</td><td>CARTOON</td><td>PHOTO</td><td>SKETCH</td><td>AVG.</td></tr><tr><td>ERM†(VAPNIK, 1999)</td><td>84.7±0.4</td><td>80.8±0.6</td><td>97.2±0.3</td><td>79.3±1.0</td><td>85.5</td></tr><tr><td>IRM†(ARJOVSKY ET AL., 2019)</td><td>84.8±1.3</td><td>76.4±1.1</td><td>96.7±0.6</td><td>76.1±1.0</td><td>83.5</td></tr><tr><td>GROUPDRO†(SAGAWA ET AL., 2020)</td><td>83.5±0.9</td><td>79.1±0.6</td><td>96.7±0.3</td><td>78.3±2.0</td><td>84.4</td></tr><tr><td>I-MIXUP†(XU ET AL., 2020)</td><td>86.1±0.5</td><td>78.9±0.8</td><td>97.6±0.1</td><td>75.8±1.8</td><td>84.6</td></tr><tr><td>MLDG†(LI ET AL., 2018A)</td><td>85.5±1.4</td><td>80.1±1.7</td><td>97.4±0.3</td><td>76.6±1.1</td><td>84.9</td></tr><tr><td>MMD†(LI ET AL., 2018B)</td><td>86.1±1.4</td><td>79.4±0.9</td><td>96.6±0.2</td><td>76.5±0.5</td><td>84.7</td></tr><tr><td>DANN†(GANIN ET AL., 2016)</td><td>86.4±0.8</td><td>77.4±0.8</td><td>97.3±0.4</td><td>73.5±2.3</td><td>83.7</td></tr><tr><td>CDANN†(LI ET AL., 2018C)</td><td>84.6±1.8</td><td>75.5±0.9</td><td>96.8±0.3</td><td>73.5±0.6</td><td>82.6</td></tr><tr><td>MTL†(BLANCHARD ET AL., 2021)</td><td>87.5±0.8</td><td>77.1±0.5</td><td>96.4±0.8</td><td>77.3±1.8</td><td>84.6</td></tr><tr><td>SAGNET†(NAM ET AL., 2021)</td><td>87.4±1.0</td><td>80.7±0.6</td><td>97.1±0.1</td><td>80.0±0.4</td><td>86.3</td></tr><tr><td>ARM†(ZHANG ET AL., 2021)</td><td>86.8±0.6</td><td>76.8±0.5</td><td>97.4±0.3</td><td>79.3±1.2</td><td>85.1</td></tr><tr><td>VREX†(KRUEGER ET AL., 2021)</td><td>86.0±1.6</td><td>79.1±0.6</td><td>96.9±0.5</td><td>77.7±1.7</td><td>84.9</td></tr><tr><td>MIXSTYLE(ZHOU ET AL., 2021)</td><td>86.8±0.5</td><td>79.0±1.4</td><td>96.6±0.1</td><td>78.5±2.3</td><td>85.2</td></tr><tr><td>RSC+GENIE(OURS)</td><td>87.8±0.6</td><td>82.4±0.8</td><td>97.4±0.9</td><td>81.4±2.0</td><td>87.3</td></tr><tr><td>CORAL+GENIE(OURS)</td><td>88.8±0.2</td><td>82.5±0.5</td><td>98.2±0.1</td><td>82.2±0.2</td><td>87.9</td></tr></table>

Table 18. Integration with existing DG algorithms on the VLCS dataset. 

<table><tr><td>ALGORITHM</td><td>CALTECH</td><td>LABELME</td><td>SUN</td><td>VOC</td><td>AVG.</td></tr><tr><td>ERM†(VAPNIK, 1999)</td><td> $98.0 \pm 0.3$ </td><td> $64.7 \pm 1.2$ </td><td> $71.4 \pm 1.2$ </td><td> $75.2 \pm 1.6$ </td><td>77.3</td></tr><tr><td>IRM†(ARJOVSKY ET AL., 2019)</td><td> $98.6 \pm 0.1$ </td><td> $64.9 \pm 0.9$ </td><td> $73.4 \pm 0.6$ </td><td> $77.3 \pm 0.9$ </td><td>78.6</td></tr><tr><td>GROUPDRO†(SAGAWA ET AL., 2020)</td><td> $97.3 \pm 0.3$ </td><td> $63.4 \pm 0.9$ </td><td> $69.5 \pm 0.8$ </td><td> $76.7 \pm 0.7$ </td><td>76.7</td></tr><tr><td>I-MIXUP†(XU ET AL., 2020)</td><td> $98.3 \pm 0.6$ </td><td> $64.8 \pm 1.0$ </td><td> $72.1 \pm 0.5$ </td><td> $74.3 \pm 0.8$ </td><td>77.4</td></tr><tr><td>MLDG†(LI ET AL., 2018A)</td><td> $97.4 \pm 0.2$ </td><td> $65.2 \pm 0.7$ </td><td> $71.0 \pm 1.4$ </td><td> $75.3 \pm 1.0$ </td><td>77.2</td></tr><tr><td>MMD†(LI ET AL., 2018B)</td><td> $97.7 \pm 0.1$ </td><td> $64.0 \pm 1.1$ </td><td> $72.8 \pm 0.2$ </td><td> $75.3 \pm 3.3$ </td><td>77.5</td></tr><tr><td>DANN†(GANIN ET AL., 2016)</td><td> $99.0 \pm 0.3$ </td><td> $65.1 \pm 1.4$ </td><td> $73.1 \pm 0.3$ </td><td> $77.2 \pm 0.6$ </td><td>78.6</td></tr><tr><td>CDANN†(LI ET AL., 2018C)</td><td> $97.1 \pm 0.3$ </td><td> $65.1 \pm 1.2$ </td><td> $70.7 \pm 0.8$ </td><td> $77.1 \pm 1.5$ </td><td>77.5</td></tr><tr><td>MTL†(BLANCHARD ET AL., 2021)</td><td> $97.8 \pm 0.4$ </td><td> $64.3 \pm 0.3$ </td><td> $71.5 \pm 0.7$ </td><td> $75.3 \pm 1.7$ </td><td>77.2</td></tr><tr><td>SAGNET†(NAM ET AL., 2021)</td><td> $97.9 \pm 0.4$ </td><td> $64.5 \pm 0.5$ </td><td> $71.4 \pm 1.3$ </td><td> $77.5 \pm 0.5$ </td><td>77.8</td></tr><tr><td>ARM†(ZHANG ET AL., 2021)</td><td> $98.7 \pm 0.2$ </td><td> $63.6 \pm 0.7$ </td><td> $71.3 \pm 1.2$ </td><td> $76.7 \pm 0.6$ </td><td>77.6</td></tr><tr><td>VREX†(KRUEGER ET AL., 2021)</td><td> $98.4 \pm 0.3$ </td><td> $64.4 \pm 1.4$ </td><td> $74.1 \pm 0.4$ </td><td> $76.2 \pm 1.3$ </td><td>78.3</td></tr><tr><td>MIXSTYLE(ZHOU ET AL., 2021)</td><td> $98.6 \pm 0.3$ </td><td> $64.5 \pm 1.1$ </td><td> $72.6 \pm 0.5$ </td><td> $75.7 \pm 1.7$ </td><td>77.9</td></tr><tr><td>RSC+GENIE(OURS)</td><td> $99.1 \pm 0.3$ </td><td> $68.3 \pm 1.3$ </td><td> $76.0 \pm 0.4$ </td><td> $79.1 \pm 0.6$ </td><td>80.6</td></tr><tr><td>CORAL+GENIE(OURS)</td><td> $99.0 \pm 0.3$ </td><td> $68.3 \pm 1.0$ </td><td> $76.6 \pm 1.3$ </td><td> $78.9 \pm 0.9$ </td><td>80.7</td></tr></table>

Table 19. Integration with existing DG algorithms on the OfficeHome dataset. 

<table><tr><td>ALGORITHM</td><td>ART</td><td>CLIPART</td><td>PRODUCT</td><td>REAL-WORLD</td><td>AVG.</td></tr><tr><td>ERM†(VAPNIK, 1999)</td><td>61.3±0.7</td><td>52.4±0.3</td><td>75.8±0.1</td><td>76.6±0.3</td><td>66.5</td></tr><tr><td>IRM†(ARJOVSKY ET AL., 2019)</td><td>58.9±2.3</td><td>52.2±1.6</td><td>72.1±2.9</td><td>74.0±2.5</td><td>64.3</td></tr><tr><td>GROUPDRO†(SAGAWA ET AL., 2020)</td><td>60.4±0.7</td><td>52.7±1.0</td><td>75.0±0.7</td><td>76.0±0.7</td><td>66</td></tr><tr><td>I-MIXUP†(XU ET AL., 2020)</td><td>62.4±0.8</td><td>54.8±0.6</td><td>76.9±0.3</td><td>78.3±0.2</td><td>68.1</td></tr><tr><td>MLDG†(LI ET AL., 2018A)</td><td>61.5±0.9</td><td>53.2±0.6</td><td>75.0±1.2</td><td>77.5±0.4</td><td>66.8</td></tr><tr><td>MMD†(LI ET AL., 2018B)</td><td>60.4±0.2</td><td>53.3±0.3</td><td>74.3±0.1</td><td>77.4±0.6</td><td>66.4</td></tr><tr><td>DANN†(GANIN ET AL., 2016)</td><td>59.9±1.3</td><td>53.0±0.3</td><td>73.6±0.7</td><td>76.9±0.5</td><td>65.9</td></tr><tr><td>CDANN†(LI ET AL., 2018C)</td><td>61.5±1.4</td><td>50.4±2.4</td><td>74.4±0.9</td><td>76.6±0.8</td><td>65.7</td></tr><tr><td>MTL†(BLANCHARD ET AL., 2021)</td><td>61.5±0.7</td><td>52.4±0.6</td><td>74.9±0.4</td><td>76.8±0.4</td><td>66.4</td></tr><tr><td>SAGNET†(NAM ET AL., 2021)</td><td>63.4±0.2</td><td>54.8±0.4</td><td>75.8±0.4</td><td>78.3±0.3</td><td>68.1</td></tr><tr><td>ARM†(ZHANG ET AL., 2021)</td><td>58.9±0.8</td><td>51.0±0.5</td><td>74.1±0.1</td><td>75.2±0.3</td><td>64.8</td></tr><tr><td>VREX†(KRUEGER ET AL., 2021)</td><td>60.7±0.9</td><td>53.0±0.9</td><td>75.3±0.1</td><td>76.6±0.5</td><td>66.4</td></tr><tr><td>MIXSTYLE(ZHOU ET AL., 2021)</td><td>51.1±0.3</td><td>53.2±0.4</td><td>68.2±0.7</td><td>69.2±0.6</td><td>60.4</td></tr><tr><td>RSC+GENIE(OURS)</td><td>63.2±2.5</td><td>54.8±0.3</td><td>76.0±0.7</td><td>78.3±1.9</td><td>68.1</td></tr><tr><td>CORAL+GENIE(OURS)</td><td>66.5±0.2</td><td>56.7±0.3</td><td>78.8±0.1</td><td>80.4±0.6</td><td>70.6</td></tr></table>

Table 20. Integration with existing DG algorithms on the TerraIncognita dataset. 

<table><tr><td>ALGORITHM</td><td>L100</td><td>L38</td><td>L43</td><td>L46</td><td>AVG.</td></tr><tr><td>ERM†(VAPNIK, 1999)</td><td>54.3±0.4</td><td>42.5±0.7</td><td>55.6±0.3</td><td>38.8±2.5</td><td>47.8</td></tr><tr><td>IRM†(ARJOVSKY ET AL., 2019)</td><td>54.6±1.3</td><td>39.8±1.9</td><td>56.2±1.8</td><td>39.6±0.8</td><td>47.6</td></tr><tr><td>GROUPDRO†(SAGAWA ET AL., 2020)</td><td>41.2±0.7</td><td>38.6±2.1</td><td>56.7±0.9</td><td>36.4±2.1</td><td>43.2</td></tr><tr><td>I-MIXUP†(XU ET AL., 2020)</td><td>59.6±2.0</td><td>42.2±1.4</td><td>55.9±0.8</td><td>33.9±1.4</td><td>47.9</td></tr><tr><td>MLDG†(LI ET AL., 2018A)</td><td>54.2±3.0</td><td>44.3±1.1</td><td>55.6±0.3</td><td>36.9±2.2</td><td>47.8</td></tr><tr><td>MMD†(LI ET AL., 2018B)</td><td>41.9±3.0</td><td>34.8±1.0</td><td>57.0±1.9</td><td>35.2±1.8</td><td>42.2</td></tr><tr><td>DANN†(GANIN ET AL., 2016)</td><td>51.1±3.5</td><td>40.6±0.6</td><td>57.4±0.5</td><td>37.7±1.8</td><td>46.7</td></tr><tr><td>CDANN†(LI ET AL., 2018C)</td><td>47.0±1.9</td><td>41.3±4.8</td><td>54.9±1.7</td><td>39.8±2.3</td><td>45.8</td></tr><tr><td>MTL†(BLANCHARD ET AL., 2021)</td><td>49.3±1.2</td><td>39.6±6.3</td><td>55.6±1.1</td><td>37.8±0.8</td><td>45.6</td></tr><tr><td>SAGNET†(NAM ET AL., 2021)</td><td>53.0±2.9</td><td>43.0±2.5</td><td>57.9±0.6</td><td>40.4±1.3</td><td>48.6</td></tr><tr><td>ARM†(ZHANG ET AL., 2021)</td><td>49.3±0.7</td><td>38.3±2.4</td><td>55.8±0.8</td><td>38.7±1.3</td><td>45.5</td></tr><tr><td>VREX†(KRUEGER ET AL., 2021)</td><td>48.2±4.3</td><td>41.7±1.3</td><td>56.8±0.8</td><td>38.7±3.1</td><td>46.4</td></tr><tr><td>MIXSTYLE(ZHOU ET AL., 2021)</td><td>54.3±1.1</td><td>34.1±1.1</td><td>55.9±1.1</td><td>31.7±2.1</td><td>44</td></tr><tr><td>RSC+GENIE(OURS)</td><td>56.5±3.2</td><td>44.5±3.7</td><td>55.9±1.0</td><td>40.9±0.3</td><td>49.5</td></tr><tr><td>CORAL+GENIE(OURS)</td><td>57.0±1.2</td><td>42.9±0.7</td><td>54.1±0.8</td><td>39.4±1.3</td><td>48.4</td></tr></table>

# E.4. Detailed SDG performance.

This section reports a detailed comparison of our optimizer with existing optimizers across four datasets in the SDG setting. It also provides the performance of previous methods when employed with our optimizer. Table 21, Table 22 Table 23, Table 24

# E.5. Model Analysis

Figure 6. visualizes the behavior of SGD, Adam, and GENIE on various loss landscapes, with the number of iterations fixed at 30 for consistency. The results show that GENIE does not directly converge to the minima of the training data, whereas Adam and SGD quickly converge to the minima. While this behavior might be advantageous in IID scenarios, it can lead to overfitting to the source domain in OOD settings. Figure 7 provides a more detailed analysis of the experiment. It further elaborates on the parameter update patterns observed across optimizer.

Table 21. Detailed performance on the PACS dataset in the SDG setting. 

<table><tr><td>ALGORITHM</td><td>ART</td><td>CARTOON</td><td>PHOTO</td><td>SKETCH</td><td>AVG.</td></tr><tr><td>ERM+ADAM</td><td>77.5</td><td>72.1</td><td>54.4</td><td>53.3</td><td>64.3</td></tr><tr><td>ERM+SGD</td><td>64.0</td><td>66.0</td><td>40.8</td><td>27.1</td><td>49.5</td></tr><tr><td>ERM+SAM</td><td>70.1</td><td>75.6</td><td>42.5</td><td>42.7</td><td>57.7</td></tr><tr><td>ERM+GENIE(OURS)</td><td> $78.6 \pm 0.6$ </td><td> $81.9 \pm 0.9$ </td><td> $53.8 \pm 2.2$ </td><td> $63.5 \pm 5.2$ </td><td>69.5</td></tr><tr><td>RSC+GENIE(OURS)</td><td> $79.8 \pm 2.0$ </td><td> $81.0 \pm 1.2$ </td><td> $56.7 \pm 3.3$ </td><td> $65.9 \pm 4.6$ </td><td>70.9</td></tr><tr><td>CORAL+GENIE(OURS)</td><td> $78.9 \pm 0.6$ </td><td> $78.6 \pm 2.2$ </td><td> $53.9 \pm 1.2$ </td><td> $61.3 \pm 3.4$ </td><td>68.2</td></tr></table>

Table 22. Detailed performance on the VLCS dataset in the SDG setting. 

<table><tr><td>ALGORITHM</td><td>CALTECH</td><td>LABELME</td><td>SUN</td><td>VOC</td><td>AVG.</td></tr><tr><td>ERM+ADAM</td><td>33.7</td><td>53.5</td><td>62.3</td><td>75.4</td><td>56.2</td></tr><tr><td>ERM+SGD</td><td>46.3</td><td>52.4</td><td>65.6</td><td>77.3</td><td>60.4</td></tr><tr><td>ERM+SAM</td><td>54.4</td><td>68.1</td><td>65.8</td><td>78.7</td><td>66.7</td></tr><tr><td>ERM+GENIE(OURS)</td><td> $56.6 \pm 4.3$ </td><td> $75.4 \pm 1.1$ </td><td> $67.2 \pm 1.4$ </td><td> $80.5 \pm 0.3$ </td><td>69.9</td></tr><tr><td>RSC+GENIE(OURS)</td><td> $56.3 \pm 2.5$ </td><td> $72.0 \pm 1.5$ </td><td> $68.8 \pm 1.7$ </td><td> $79.6 \pm 1.1$ </td><td>69.2</td></tr><tr><td>CORAL+GENIE(OURS)</td><td> $55.9 \pm 1.3$ </td><td> $71.7 \pm 1.8$ </td><td> $67.2 \pm 1.8$ </td><td> $79.9 \pm 1.4$ </td><td>68.7</td></tr></table>

Table 23. Detailed performance on the OfficeHome dataset in the SDG setting. 

<table><tr><td>ALGORITHM</td><td>ART</td><td>CLIPART</td><td>PRODUCT</td><td>REAL-WORLD</td><td>AVG.</td></tr><tr><td>ERM+ADAM</td><td>52.6</td><td>46.5</td><td>45.6</td><td>58</td><td>50.7</td></tr><tr><td>ERM+SGD</td><td>47.7</td><td>41.4</td><td>43.2</td><td>51.5</td><td>45.9</td></tr><tr><td>ERM+SAM</td><td>60.1</td><td>57.8</td><td>55.8</td><td>63.1</td><td>59.2</td></tr><tr><td>ERM+GENIE(OURS)</td><td> $59.4 \pm 0.7$ </td><td> $58.7 \pm 0.9$ </td><td> $54.1 \pm 0.5$ </td><td> $62.0 \pm 0.3$ </td><td>58.6</td></tr><tr><td>RSC+GENIE(OURS)</td><td> $55.8 \pm 2.6$ </td><td> $52.5 \pm 2.0$ </td><td> $49.7 \pm 3.0$ </td><td> $59.7 \pm 2.0$ </td><td>54.4</td></tr><tr><td>CORAL+GENIE(OURS)</td><td> $58.0 \pm 0.9$ </td><td> $54.8 \pm 1.5$ </td><td> $51.3 \pm 0.3$ </td><td> $61.4 \pm 0.7$ </td><td>56.4</td></tr></table>

Table 24. Detailed performance on the TerraIncognita dataset in the SDG setting. 

<table><tr><td>ALGORITHM</td><td>L100</td><td>L38</td><td>L43</td><td>L46</td><td>AVG.</td></tr><tr><td>ERM+ADAM</td><td>27.0</td><td>25.5</td><td>42.9</td><td>38.6</td><td>33.5</td></tr><tr><td>ERM+SGD</td><td>22.1</td><td>17.9</td><td>22.3</td><td>29</td><td>22.8</td></tr><tr><td>ERM+SAM</td><td>21.1</td><td>21.6</td><td>30.3</td><td>34.2</td><td>26.8</td></tr><tr><td>ERM+GENIE(OURS)</td><td> $28.5 \pm 0.5$ </td><td> $29.4 \pm 1.5$ </td><td> $41.7 \pm 1.8$ </td><td> $44.5 \pm 1.3$ </td><td>36.0</td></tr><tr><td>RSC+GENIE(OURS)</td><td> $27.1 \pm 1.5$ </td><td> $22.6 \pm 1.7$ </td><td> $39.7 \pm 3.0$ </td><td> $43.3 \pm 1.1$ </td><td>33.2</td></tr><tr><td>CORAL+GENIE(OURS)</td><td> $29.3 \pm 1.0$ </td><td> $29.8 \pm 1.8$ </td><td> $43.9 \pm 1.3$ </td><td> $43.6 \pm 1.6$ </td><td>36.7</td></tr></table>

![](images/91a04029cf211c759f1daf1a28b293cbc1fc7624204be33af240cce3c278ad10.jpg)

Figure 6. Optimization trajectories on a simulated loss landscape.   
![](images/62df26d903d3b3d708d334ae0d7dc08ce30c7276254afaebc15bc16cb447ec1e.jpg)  
Figure 7. Heatmaps visualizing normalized parameter update magnitudes by parameter ID for different optimizers throughout training on the VLCS dataset in the DG.