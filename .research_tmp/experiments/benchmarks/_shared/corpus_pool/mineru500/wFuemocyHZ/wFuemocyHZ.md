# Restart Sampling for Improving Generative Processes

Yilun Xu\*

MIT

ylxu@mit.edu

Mingyang Deng\*

MIT

dengm@mit.edu

Xiang Cheng\*

MIT

chengx@mit.edu

Yonglong Tian

Google Research

yonglong@google.com

Ziming Liu

MIT

zmliu@mit.edu

Tommi Jaakkola

MIT

tommi@csail.mit.edu

# Abstract

Generative processes that involve solving differential equations, such as diffusion models, frequently necessitate balancing speed and quality. ODE-based samplers are fast but plateau in performance while SDE-based samplers deliver higher sample quality at the cost of increased sampling time. We attribute this difference to sampling errors: ODE-samplers involve smaller discretization errors while stochasticity in SDE contracts accumulated errors. Based on these findings, we propose a novel sampling algorithm called Restart in order to better balance discretization errors and contraction. The sampling method alternates between adding substantial noise in additional forward steps and strictly following a backward ODE. Empirically, Restart sampler surpasses previous SDE and ODE samplers in both speed and accuracy. Restart not only outperforms the previous best SDE results, but also accelerates the sampling speed by 10-fold / 2-fold on CIFAR-10 / ImageNet 64×64. In addition, it attains significantly better sample quality than ODE samplers within comparable sampling times. Moreover, Restart better balances text-image alignment/visual quality versus diversity than previous samplers in the large-scale text-to-image Stable Diffusion model pre-trained on LAION 512×512. Code is available at https://github.com/Newbeeer/diffusion\_restart\_sampling

# 1 Introduction

Deep generative models based on differential equations, such as diffusion models and Poisson flow generative models, have emerged as powerful tools for modeling high-dimensional data, from image synthesis $[23, 9, 13, 27, 28]$ to biological data $[10, 26]$ . These models use iterative backward processes that gradually transform a simple distribution (e.g., Gaussian in diffusion models) into a complex data distribution by solving a differential equations. The associated vector fields (or drifts) driving the evolution of the differential equations are predicted by neural networks. The resulting sample quality can be often improved by enhanced simulation techniques but at the cost of longer sampling times.

Prior samplers for simulating these backward processes can be categorized into two groups: ODE-samplers whose evolution beyond the initial randomization is deterministic, and SDE-samplers where the generation trajectories are stochastic. Several works $[23, 12, 13]$ show that these samplers demonstrate their advantages in different regimes, as depicted in Fig. 1(b). ODE solvers $[22, 16, 13]$ result in smaller discretization errors, allowing for decent sample quality even with larger step sizes (i.e., fewer number of function evaluations (NFE)). However, their generation quality plateaus rapidly. In contrast, SDE achieves better quality in the large NFE regime, albeit at the expense of increased sampling time. To better understand these differences, we theoretically analyze SDE performance: the

stochasticity in SDE contracts accumulated error, which consists of both the discretization error along the trajectories as well as the approximation error of the learned neural network relative to the ground truth drift (e.g., score function in diffusion model $[23]$ ). The approximation error dominates when NFE is large (small discretization steps), explaining the SDE advantage in this regime. Intuitively, the stochastic nature of SDE helps "forget" accumulated errors from previous time steps.

Inspired by these findings, we propose a novel sampling algorithm called Restart, which combines the advantages of ODE and SDE. As illustrated in Fig. 1(a), the Restart sampling algorithm involves K repetitions of two subroutines in a pre-defined time interval: a Restart forward process that adds a substantial amount of noise, akin to "restarting" the original backward process, and a Restart backward process that runs the backward ODE. The Restart algorithm separates the stochasticity from the drifts, and the amount of added noise in the Restart forward process is significantly larger than the small single-step noise interleaving with drifts in previous SDEs such as $[23, 13]$ , thus amplifying the contraction effect on accumulated errors. By repeating the forward-backward cycle K times, the contraction effect introduced in each Restart iteration is further strengthened. The deterministic backward processes allow Restart to reduce discretization errors, thereby enabling step sizes comparable to ODE. To maximize the contraction effects in practice, we typically position the Restart interval towards the end of the simulation, where the accumulated error is larger. Additionally, we apply multiple Restart intervals to further reduce the initial errors in more challenging tasks.

Experimentally, Restart consistently surpasses previous ODE and SDE solvers in both quality and speed over a range of NFEs, datasets, and pre-trained models. Specifically, Restart accelerates the previous best-performing SDEs by $10\times$ fewer steps for the same FID score on CIFAR-10 using VP [23] ( $2\times$ fewer steps on ImageNet $64\times64$ with EDM [13]), and outperforms fast ODE solvers (e.g., DPM-solver [16]) even in the small NFE regime. When integrated into previous state-of-the-art pre-trained models, Restart further improves performance, achieving FID scores of 1.88 on unconditional CIFAR-10 with PFGM++ [28], and 1.36 on class-conditional ImageNet $64\times64$ with EDM. To the best of our knowledge, these are the best FID scores obtained on commonly used UNet architectures for diffusion models without additional training. We also apply Restart to the practical application of text-to-image Stable Diffusion model [19] pre-trained on LAION $512\times512$ . Restart more effectively balances text-image alignment/visual quality (measured by CLIP/Aesthetic scores) and diversity (measured by FID score) with a varying classifier-free guidance strength, compared to previous samplers.

Our contributions can be summarized as follows: (1) We investigate ODE and SDE solvers and theoretically demonstrate the contraction effect of stochasticity via an upper bound on the Wasserstein distance between generated and data distributions (Sec 3); (2) We introduce the Restart sampling, which better harnesses the contraction effect of stochasticity while allowing for fast sampling. The sampler results in a smaller Wasserstein upper bound (Sec 4); (3) Our experiments are consistent with the theoretical bounds and highlight Restart's superior performance compared to previous samplers on standard benchmarks in terms of both quality and speed. Additionally, Restart improves the trade-off between key metrics on the Stable Diffusion model (Sec 5).

# 2 Background on Generative Models with Differential Equations

Many recent successful generative models have their origin in physical processes, including diffusion models $[9, 23, 13]$ and Poisson flow generative models $[27, 28]$ . These models involve a forward process that transforms the data distribution into a chosen smooth distribution, and a backward process that iteratively reverses the forward process. For instance, in diffusion models, the forward process is the diffusion process with no learned parameters:

$$
\mathrm{d} x = \sqrt {2 \dot {\sigma} (t) \sigma (t)} \mathrm{d} W _ {t},
$$

where $\sigma(t)$ is a predefined noise schedule increasing with t, and $W_{t} \in R^{d}$ is the standard Wiener process. For simplicity, we omit an additional scaling function for other variants of diffusion models as in EDM [13]. Under this notation, the marginal distribution at time t is the convolution of data distribution $p_{0} = p_{data}$ and a Gaussian kernel, i.e., $p_{t} = p_{0} * \mathcal{N}(\mathbf{0}, \sigma^{2}(t)\mathbf{I}_{d \times d})$ . The prior distribution is set to $\mathcal{N}(\mathbf{0}, \sigma^{2}(T)\mathbf{I}_{d \times d})$ since $p_{T}$ is approximately Gaussian with a sufficiently large T. Sampling of diffusion models is done via a reverse-time SDE [1] or a marginally-equivalent ODE [23]:

$$
\mathrm{d} x = - 2 \dot {\sigma} (t) \sigma (t) \nabla_ {x} \log p _ {t} (x) d t + \sqrt {2 \dot {\sigma} (t) \sigma (t)} \mathrm{d} W _ {t} \tag {1}
$$

$$
\text { (ODE) } \quad \mathrm{d} x = - \dot {\sigma} (t) \sigma (t) \nabla_ {x} \log p _ {t} (x) d t \tag {2}
$$

![](images/bce5a3b45264bd640fe48eaecc04046ab791210d4d8c016e8bca05b4427a492e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["ODE"] --> B["t = 0"]
    C["SDE"] --> D["t = t_max"]
    E["Restart"] --> F["t = t_min"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style F fill:#ccf,stroke:#333
    subgraph Time_States
        direction TB
        G["Orange: drift"]
        H["Green: noise"]
    end
```
</details>

(a)

![](images/ccea66368b99091f78eb94941e4318d7fa5b61b0cf81c8ff814be395faddc18c.jpg)

<details>
<summary>line</summary>

| NFE Regime | ODE  | SDE  | Restart |
|------------|------|------|---------|
| Small      | High | High | High    |
| Large      | Low  | Low  | Low     |
</details>

(b)   
Figure 1: (a) Illustration of the implementation of drift and noise terms in ODE, SDE, and Restart. (b) Sample quality versus number of function evaluations (NFE) for different approaches. ODE (Green) provides fast speeds but attains only mediocre quality, even with a large NFE. SDE (Yellow) obtains good sample quality but necessitates substantial sampling time. In contrast to ODE and SDE, which have their own winning regions, Restart (Red) achieves the best quality across all NFEs.

where $\nabla_{x}\log p_{t}(x)$ in the drift term is the score of intermediate distribution at time t. W.l.o.g we set $\sigma(t)=t$ in the remaining text, as in [13]. Both processes progressively recover $p_{0}$ from the prior distribution $p_{T}$ while sharing the same time-dependent distribution $p_{t}$ . In practice, we train a neural network $s_{\theta}(x,t)$ to estimate the score field $\nabla_{x}\log p_{t}(x)$ by minimizing the denoising score-matching loss [25]. We then substitute the score $\nabla_{x}\log p_{t}(x)$ with $s_{\theta}(x,t)$ in the drift term of above backward SDE (Eq. (1))/ODE (Eq. (2)) for sampling.

Recent work inspired by electrostatics has not only challenged but also integrated diffusion models, notably PFGM/PFGM++, enhances performance in both image and antibody generation $[27, 28, 10]$ . They interpret data as electric charges in an augmented space, and the generative processes involve the simulations of differential equations defined by electric field lines. Similar to diffusion models, PFGMs train a neural network to approximate the electric field in the augmented space.

# 3 Explaining SDE and ODE performance regimes

To sample from the aforementioned generative models, a prevalent approach employs general-purpose numerical solvers to simulate the corresponding differential equations. This includes Euler and Heun's 2nd method [2] for ODEs (e.g., Eq. (2)), and Euler-Maruyama for SDEs (e.g., Eq. (1)). Sampling algorithms typically balance two critical metrics: (1) the quality and diversity of generated samples, often assessed via the Fréchet Inception Distance (FID) between generated distribution and data distribution [7] (lower is better), and (2) the sampling time, measured by the number of function evaluations (NFE). Generally, as the NFE decreases, the FID score tends to deteriorate across all samplers. This is attributed to the increased discretization error caused by using a larger step size in numerical solvers.

However, as illustrated in Fig. 1(b) and observed in previous works on diffusion models [23, 22, 13], the typical pattern of the quality vs time curves behaves differently between the two groups of samplers, ODE and SDE. When employing standard numerical solvers, ODE samplers attain a decent quality with limited NFEs, whereas SDE samplers struggle in the same small NFE regime. However, the performance of ODE samplers quickly reaches a plateau and fails to improve with an increase in NFE, whereas SDE samplers can achieve noticeably better sample quality in the high NFE regime. This dilemma raises an intriguing question: Why do ODE samplers outperform SDE samplers in the small NFE regime, yet fall short in the large NFE regime?

The first part of the question is relatively straightforward to address: given the same order of numerical solvers, simulation of ODE has significantly smaller discretization error compared to the SDE. For example, the first-order Euler method for ODE results in a local error of $O(\delta^2)$ , whereas the first-order Euler-Maruyama method for SDEs yeilds a local error of $O(\delta^{\frac{3}{2}})$ (see e.g., Theorem 1 of [4]), where $\delta$ denotes the step size. As $O(\delta^{\frac{3}{2}}) \gg O(\delta^2)$ , ODE simulations exhibit lower sampling errors than SDEs, likely causing the better sample quality with larger step sizes in the small NFE regime.

In the large NFE regime the step size $\delta$ shrinks and discretization errors become less significant for both ODEs and SDEs. In this regime it is the approximation error — error arising from an inaccurate estimation of the ground-truth vector field by the neural network $s_{\theta}$ — starts to dominate the sampling error. We denote the discretized ODE and SDE using the learned field $s_{\theta}$ as $ODE_{\theta}$ and $SDE_{\theta}$ , respectively. In the following theorem, we evaluate the total errors from simulating $ODE_{\theta}$ and $SDE_{\theta}$ within the time interval $[t_{\min}, t_{\max}] \subset [0, T]$ . This is done via an upper bound on the Wasserstein-1 distance between the generated and data distributions at time $t_{\min}$ . We characterize the accumulated initial sampling errors up until $t_{\max}$ by total variation distances. Below we show that the inherent stochasticity of SDEs aids in contracting these initial errors at the cost of larger additional sampling error in $[t_{\min}, t_{\max}]$ . Consequently, SDE results in a smaller upper bound as the step size $\delta$ nears 0 (pertaining to the high NFE regime).

Theorem 1 (Informal). Let $t_{max}$ be the initial noise level and $p_t$ denote the true distribution at noise level $t$ . Let $p_t^{ODE_\theta}, p_t^{SDE_\theta}$ denote the distributions of simulating $ODE_\theta$ , $SDE_\theta$ respectively. Assume that $\forall t \in [t_{min}, t_{max}]$ , $\| x_t \| < B/2$ for any $x_t$ in the support of $p_t$ , $p_t^{ODE_\theta}$ or $p_t^{SDE_\theta}$ . Then

$$
W _ {1} (p _ {t _ {\min}} ^ {O D E _ {\theta}}, p _ {t _ {\min}}) \leq B \cdot T V \left(p _ {t _ {\max}} ^ {O D E _ {\theta}}, p _ {t _ {\max}}\right) + O (\delta + \epsilon_ {a p p r o x}) \cdot (t _ {\max} - t _ {\min})
$$

$$
\underbrace {W _ {1} (p _ {t _ {m i n}} ^ {S D E _ {\theta}} , p _ {t _ {m i n}})} _ {\text {total error}} \leq \underbrace {\left(1 - \lambda e ^ {- U}\right) B \cdot T V (p _ {t _ {m a x}} ^ {S D E _ {\theta}} , p _ {t _ {m a x}})} _ {\text {upper bound on contracted error}} + \underbrace {O (\sqrt {\delta t _ {m a x}} + \epsilon_ {a p p r o x}) \left(t _ {m a x} - t _ {m i n}\right)} _ {\text {upper bound on additional sampling error}}
$$

In the above, $U = BL_{1} / t_{min} + L_{1}^{2}t_{max}^{2} / t_{min}^{2}$ , $\lambda < 1$ is a contraction factor, $L_{1}$ and $\epsilon_{approx}$ are uniform bounds on $\| ts_{\theta}(x_t,t)\|$ and the approximation error $\| t\nabla_x\log p_t(x) - ts_\theta (x,t)\|$ for all $x_{t},t$ , respectively. $O()$ hides polynomial dependency on various Lipschitz constants and dimension.

We defer the formal version and proof of Theorem 1 to Appendix A.1. As shown in the theorem, the upper bound on the total error can be decomposed into upper bounds on the contracted error and additional sampling error. $TV(p_{t_{\mathrm{max}}}^{\mathrm{ODE}_{\theta}}, p_{t_{\mathrm{max}}})$ and $TV(p_{t_{\mathrm{max}}}^{\mathrm{SDE}_{\theta}}, p_{t_{\mathrm{max}}})$ correspond to the initial errors accumulated from both approximation and discretization errors during the simulation of the backward process, up until time $t_{\mathrm{max}}$ . In the context of SDE, this accumulated error undergoes contraction by a factor of $1 - \lambda e^{-BL_1 / t_{\mathrm{min}} - L_1^2 t_{\mathrm{max}}^2 / t_{\mathrm{min}}^2}$ within $[t_{\mathrm{min}}, t_{\mathrm{max}}]$ , due to the effect of adding noise. Essentially, the minor additive Gaussian noise in each step can drive the generated distribution and the true distribution towards each other, thereby neutralizing a portion of the initial accumulated error.

The other term related to additional sampling error includes the accumulation of discretization and approximation errors in $[t_{\min}, t_{\max}]$ . Despite the fact that SDE incurs a higher discretization error than ODE $(O(\sqrt{\delta})$ versus $O(\delta))$ , the contraction effect on the initial error is the dominant factor impacting the upper bound in the large NFE regime where $\delta$ is small. Consequently, the upper bound for SDE is significantly lower. This provides insight into why SDE outperforms ODE in the large NFE regime, where the influence of discretization errors diminishes and the contraction effect dominates. In light of the distinct advantages of SDE and ODE, it is natural to ask whether we can combine their strengths. Specifically, can we devise a sampling algorithm that maintains a comparable level of discretization error as ODE, while also benefiting from, or even amplifying, the contraction effects induced by the stochasticity of SDE? In the next section, we introduce a novel algorithm, termed Restart, designed to achieve these two goals simultaneously.

# 4 Harnessing stochasticity with Restart

In this section, we present the Restart sampling algorithm, which incorporates stochasticity during sampling while enabling fast generation. We introduce the algorithm in Sec 4.1, followed by a theoretical analysis in Sec 4.2. Our analysis shows that Restart achieves a better Wasserstein upper bound compared to those of SDE and ODE in Theorem 1 due to greater contraction effects.

# 4.1 Method

In the Restart algorithm, simulation performs a few repeated back-and-forth steps within a pre-defined time interval $[t_{\min}, t_{\max}] \subset [0, T]$ , as depicted in Figure 1(a). This interval is embedded into the simulation of the original backward ODE referred to as the main backward process, which runs from T to 0. In addition, we refer to the backward process within the Restart interval $[t_{\min}, t_{\max}]$ as the Restart backward process, to distinguish it from the main backward process.

Starting with samples at time $t_{min}$ , which are generated by following the main backward process, the Restart algorithm adds a large noise to transit the samples from $t_{min}$ to $t_{max}$ with the help of the forward process. The forward process does not require any evaluation of the neural network $s_{\theta}(x,t)$ , as it is generally defined by an analytical perturbation kernel capable of transporting distributions from $t_{min}$ to $t_{max}$ . For instance, in the case of diffusion models, the perturbation kernel is $\mathcal{N}(\mathbf{0}, (\sigma(t_{\mathrm{max}})^2 - \sigma(t_{\mathrm{min}})^2) \mathbf{I}_{d \times d})$ . The added noise in this step induces a more significant contraction compared to the small, interleaved noise in SDE. The step acts as if partially restarting the main backward process by increasing the time. Following this step, Restart simulates the backward ODE from $t_{max}$ back to $t_{min}$ using the neural network predictions as in regular ODE. We repeat these forward-backward steps within $[t_{min}, t_{max}]$ interval K times in order to further derive the benefit from contraction. Specifically, the forward and backward processes in the $i^{th}$ iteration ( $i \in \{0, \ldots, K-1\}$ ) proceed as follows:

$$
\text {(Restart forward process)} \quad x _ {t _ {\max}} ^ {i + 1} = x _ {t _ {\min}} ^ {i} + \varepsilon_ {t _ {\min} \rightarrow t _ {\max}} \tag {3}
$$

$$
\text {(Restart backward process)} \quad x _ {t _ {\min}} ^ {i + 1} = \mathrm{ODE} _ {\theta} (x _ {t _ {\max}} ^ {i + 1}, t _ {\max} \to t _ {\min}) \tag {4}
$$

where the initial $x_{t_{min}}^{0}$ is obtained by simulating the ODE until $t_{min}$ : $x_{t_{min}}^{0} = \mathrm{ODE}_{\theta}(x_{T}, T \to t_{\min})$ , and the noise $\varepsilon_{t_{\min} \to t_{\max}}$ is sampled from the corresponding perturbation kernel from $t_{min}$ to $t_{max}$ . The Restart algorithm not only adds substantial noise in the Restart forward process (Eq. (3)), but also separates the stochasticity from the ODE, leading to a greater contraction effect, which we will demonstrate theoretically in the next subsection. For example, we set $[t_{\min}, t_{\max}] = [0.05, 0.3]$ for the VP model [13] on CIFAR-10. Repetitive use of the forward noise effectively mitigates errors accumulated from the preceding simulation up until $t_{max}$ . Furthermore, the Restart algorithm does not suffer from large discretization errors as it is mainly built from following the ODE in the Restart backward process (Eq. (4)). The effect is that the Restart algorithm is able to reduce the total sampling errors even in the small NFE regime. Detailed pseudocode for the Restart sampling process can be found in Algorithm 2, Appendix B.2.

# 4.2 Analysis

We provide a theoretical analysis of the Restart algorithm under the same setting as Theorem 1. In particular, we prove the following theorem, which shows that Restart achieves a much smaller contracted error in the Wasserstein upper bound than SDE (Theorem 1), thanks to the separation of the noise from the drift, as well as the large added noise in the Restart forward process (Eq. (3)). The repetition of the Restart cycle K times further leads to a enhanced reduction in the initial accumulated error. We denote the intermediate distribution in the $i^{th}$ Restart iteration, following the discretized trajectories and the learned field $s_{\theta}$ , as $p_{t\in[t_{\min},t_{\max}]}^{\text{Restart}_{\theta}(i)}$ .

Theorem 2 (Informal). Under the same setting of Theorem 1, assume $K \leq \frac{C}{L_2(t_{max} - t_{min})}$ for some universal constant $C$ . Then

$$
\underbrace {W _ {1} (p _ {t _ {m i n}} ^ {R e s t a r t _ {\theta} (K)} , p _ {t _ {m i n}})} _ {\text {total error}} \leq \underbrace {B \cdot (1 - \lambda) ^ {K} T V (p _ {t _ {m a x}} ^ {R e s t a r t _ {\theta} (0)} , p _ {t _ {m a x}})} _ {\text {upper bound on contracted error}} + \underbrace {(K + 1) \cdot O (\delta + \epsilon_ {a p p r o x}) (t _ {m a x} - t _ {m i n})} _ {\text {upper bound on additional sampling error}}
$$

where $\lambda < 1$ is the same contraction factor as Theorem 1. $O()$ hides polynomial dependency on various Lipschitz constants, dimension.

Proof sketch. To bound the total error, we introduce an auxiliary process $q_{t\in[t_{\min},t_{\max}]}^{\text{Restart}_{\theta}(i)}$ , which initiates from true distribution $p_{t_{\max}}$ and performs the Restart iterations. This process differs from $p_{t\in[t_{\min},t_{\max}]}^{\text{Restart}_{\theta}(i)}$ only in its initial distribution at $t_{\max}$ ( $p_{t_{\max}}$ versus $p_{t_{\max}}^{\text{Restart}_{\theta}(0)}$ ). We bound the total error by the following triangular inequality:

$$
\underbrace {W _ {1} (p _ {t _ {\min}} ^ {\text { Restart } _ {\theta} (K)} , p _ {t _ {\min}})} _ {\text { total   error }} \leq \underbrace {W _ {1} (p _ {t _ {\min}} ^ {\text { Restart } _ {\theta} (K)} , q _ {t _ {\min}} ^ {\text { Restart } _ {\theta} (K)})} _ {\text { contracted   error }} + \underbrace {W _ {1} (q _ {t _ {\min}} ^ {\text { Restart } _ {\theta} (K)} , p _ {t _ {\min}})} _ {\text { additional   sampling   error }}
$$

To bound the contracted error, we construct a careful coupling process between two individual trajectories sampled from $p_{t_{\min}}^{\text{Restart}_{\theta}(i)}$ and $q_{t_{\min}}^{\text{Restart}_{\theta}(i)}, i = 0, \ldots, K - 1$ . Before these two trajectories converge, the Gaussian noise added in each Restart iteration is chosen to maximize the probability of

the two trajectories mapping to an identical point, thereby maximizing the mixing rate in TV. After converging, the two processes evolve under the same Gaussian noise, and will stay converged as their drifts are the same. Lastly, we convert the TV bound to $W_{1}$ bound by multiplying B. The bound on the additional sampling error echoes the ODE analysis in Theorem 1: since the noise-injection and ODE-simulation stages are separate, we do not incur the higher discretization error of SDE. ☐

We defer the formal version and proof of Theorem 2 to Appendix A.1. The first term in RHS bounds the contraction on the initial error at time $t_{\mathrm{max}}$ and the second term reflects the additional sampling error of ODE accumulated across repeated Restart iterations. Comparing the Wasserstein upper bound of SDE and ODE in Theorem 1, we make the following three observations: (1) Each Restart iteration has a smaller contraction factor $1 - \lambda$ compared to the one in SDE, since Restart separates the large additive noise (Eq. (3)) from the ODE (Eq. (4)). (2) Restart backward process (Eq. (4)) has the same order of discretization error $O(\delta)$ as the ODE, compared to $O(\sqrt{\delta})$ in SDE. Hence, the Restart allows for small NFE due to ODE-level discretization error. (3) The contracted error further diminishes exponentially with the number of repetitions $K$ though the additional error increases linearly with $K$ . It suggests that there is a sweet spot of $K$ that strikes a balance between reducing the initial error and increasing additional sampling error. Ideally, one should pick a larger $K$ when the initial error at time $t_{\mathrm{max}}$ greatly outweigh the incurred error in the repetitive backward process from $t_{\mathrm{max}}$ to $t_{\mathrm{min}}$ . We provide empirical evidences in Sec 5.2.

While Theorem 1 and Theorem 2 compare the upper bounds on errors of different methods, we provide empirical validation in Section 5.1 by directly calculating these errors, showing that the Restart algorithm indeed yields a smaller total error due to its superior contraction effects. The main goal of Theorem 1 and Theorem 2 is to study how the already accumulated error changes using different samplers, and to understand their ability to self-correct the error by stochasticity. In essence, these theorems differentiate samplers based on their performance post-error accumulation. For example, by tracking the change of accumulated error, Theorem 1 shed light on the distinct "winning regions" of ODE and SDE: ODE samplers have smaller discretization error and hence excel at the small NFE regime. In contrast, SDE performs better in large NFE regime where the discretization error is negligible and its capacity to contract accumulated errors comes to the fore.

# 4.3 Practical considerations

The Restart algorithm offers several degrees of freedom, including the time interval $[t_{\min}, t_{\max}]$ and the number of restart iterations K. Here we provide a general recipe of parameter selection for practitioners, taking into account factors such as the complexity of the generative modeling tasks and the capacity of the network. Additionally, we discuss a stratified, multi-level Restart approach that further aids in reducing simulation errors along the whole trajectories for more challenging tasks.

Where to Restart? Theorem 2 shows that the Restart algorithm effectively reduces the accumulated error at time $t_{max}$ by a contraction factor in the Wasserstein upper bound. These theoretical findings inspire us to position the Restart interval $[t_{\min}, t_{\max}]$ towards the end of the main backward process, where the accumulated error is more substantial. In addition, our empirical observations suggest that a larger time interval $t_{max} - t_{min}$ is more beneficial for weaker/smaller architectures or more challenging datasets. Even though a larger time interval increases the additional sampling error, the benefits of the contraction significantly outweighs the downside, consistent with our theoretical predictions. We leave the development of principled approaches for optimal time interval selection for future works.

Multi-level Restart For challenging tasks that yield significant approximation errors, the backward trajectories may diverge substantially from the ground truth even at early stage. To prevent the ODE simulation from quickly deviating from the true trajectory, we propose implementing multiple Restart intervals in the backward process, alongside the interval placed towards the end. Empirically, we observe that a 1-level Restart is sufficient for CIFAR-10, while for more challenging datasets such as ImageNet [5], a multi-level Restart results in enhanced performance [5].

# 5 Experiments

In Sec 5.1, we first empirically verify the theoretical analysis relating to the Wasserstein upper bounds. We then evaluate the performance of different sampling algorithms on standard image generation benchmarks, including CIFAR-10 [14] and ImageNet $64 \times 64$ [5] in Sec 5.2. Lastly,

![](images/a082edb2857b4787e7184a6e7d877153c8ac648c4f6644c426200da006828296.jpg)

<details>
<summary>line</summary>

| Additional Sampling error | ODE    | SDE    | Restart |
| ------------------------- | ------ | ------ | ------- |
| 0.65                      | 0.875  | 0.875  | 0.875   |
| 0.70                      | 0.875  | 0.850  | 0.850   |
| 0.75                      | 0.875  | 0.850  | 0.810   |
| 0.80                      | 0.750  | 0.850  | 0.725   |
| 0.90                      | 0.725  | 0.850  | 0.725   |
</details>

(a)

![](images/61ee9ba3be94629947aec1c640599d9be2395f0658e7e398c52c2dd595962811.jpg)

<details>
<summary>scatter</summary>

| Additional Sampling error | Total Error | Method   |
| ------------------------- | ----------- | -------- |
| 0.65                      | 0.90        | ODE      |
| 0.70                      | 0.88        | SDE      |
| 0.75                      | 0.85        | Restart  |
| 0.80                      | 0.82        | Restart  |
| 0.85                      | 0.78        | Restart  |
| 0.90                      | 0.75        | Restart  |
| 0.95                      | 0.72        | Restart  |
| 1.00                      | 0.70        | Restart  |
</details>

(b)

![](images/0154e7862aaf86ef4cb5ab2a8d0b794db76866ecfe105ff2543f2648d0052589.jpg)

<details>
<summary>line</summary>

| NFE  | ODE    | SDE    | Restart |
| ---- | ------ | ------ | ------- |
| 20   | 0.875  | 0.875  | 0.875   |
| 40   | 0.875  | 0.875  | 0.825   |
| 60   | 0.875  | 0.875  | 0.765   |
| 80   | 0.875  | 0.875  | 0.765   |
| 100  | 0.875  | 0.875  | 0.765   |
| 120  | 0.875  | 0.825  | 0.765   |
| 140  | 0.875  | 0.825  | 0.765   |
| 160  | 0.875  | 0.825  | 0.765   |
| 180  | 0.875  | 0.825  | 0.765   |
| 200  | 0.875  | 0.825  | 0.765   |
| 220  | 0.875  | 0.825  | 0.765   |
| 240  | 0.875  | 0.825  | 0.765   |
| 260  | 0.875  | 0.825  | 0.765   |
| 280  | 0.875  | 0.825  | 0.765   |
| 300  | 0.875  | 0.825  | 0.765   |
| 320  | 0.875  | 0.825  | 0.765   |
</details>

(c)   
Figure 2: Additional sampling error versus (a) contracted error, where the Pareto frontier is plotted and (b) total error, where the scatter plot is provided. (c) Pareto frontier of NFE versus total error.

we employ Restart on text-to-image generation, using Stable Diffusion model $[19]$ pre-trained on LAION-5B $[21]$ with resolution $512 \times 512$ , in Sec 5.3.

# 5.1 Additional sampling error versus contracted error

Our proposed Restart sampling algorithm demonstrates a higher contraction effect and smaller addition sampling error compared to SDE, according to Theorem 1 and Theorem 2. Although our theoretical analysis compares the upper bounds of the total, contracted and additional sampling errors, we further verify their relative values through a synthetic experiment.

Setup We construct a 20-dimensional dataset with 2000 points sampled from a Gaussian mixture, and train a four-layer MLP to approximate the score field $\nabla_x\log p_t$ . We implement the ODE, SDE, and Restart methods within a predefined time range of $[t_{\min}, t_{\max}] = [1.0, 1.5]$ , where the process outside this range is conducted via the first-order ODE. To compute various error types, we define the distributions generated by three methods as outlined in the proof of Theorem 2 and directly gauge the errors at end of simulation $t = 0$ instead of $t = t_{\min}$ : (1) the generated distribution as $p_0^{\text{Sampler}}$ , where $\text{Sampler} \in \{\text{ODE}_\theta, \text{SDE}_\theta, \text{Restart}_\theta(K)\}$ ; (2) an auxiliary distribution $q_0^{\text{Sampler}}$ initiating from true distribution $p_{t_{\max}}$ at time $t_{\max}$ . The only difference between $p_0^{\text{Sampler}}$ and $q_0^{\text{Sampler}}$ is their initial distribution at $t_{\max}(p_{t_{\max}}^{\text{ODE}_\theta}$ versus $p_{t_{\max}}$ ); and (3) the true data distribution $p_0$ . In line with Theorem 2, we use Wasserstein-1 distance $W_1(p_0^{\text{Sampler}}, q_0^{\text{Sampler}}) / W_1(q_0^{\text{Sampler}}, p_0)$ to measure the contracted error / additional sampling error, respectively. Ultimately, the total error corresponds to $W_1(p_0^{\text{Sampler}}, p_0)$ . Detailed information about dataset, metric and model can be found in the Appendix C.5.

Results In our experiment, we adjust the parameters for all three processes and calculate the total, contracted, and additional sampling errors across all parameter settings. Figure 2(a) depicts the Pareto frontier of additional sampling error versus contracted error. We can see that Restart consistently achieves lower contracted error for a given level of additional sampling error, compared to both the ODE and SDE methods, as predicted by theory. In Figure 2(b), we observe that the Restart method obtains a smaller total error within the additional sampling error range of [0.8, 0.85]. During this range, Restart also displays a strictly reduced contracted error, as illustrated in Figure 2(a). This aligns with our theoretical analysis, suggesting that the Restart method offers a smaller total error due to its enhanced contraction effects. From Figure 2(c), Restart also strikes an better balance between efficiency and quality, as it achieves a lower total error at a given NFE.

# 5.2 Experiments on standard benchmarks

To evaluate the sample quality and inference speed, we report the FID score $[7]$ (lower is better) on 50K samplers and the number of function evaluations (NFE). We borrow the pretrained VP/EDM/PFGM++ models on CIFAR-10 or ImageNet $64 \times 64$ from $[13, 28]$ . We also use the EDM discretization scheme $[13]$ (see Appendix B.1 for details) during sampling.

For the proposed Restart sampler, the hyperparameters include the number of steps in the main/Restart backward processes, the number of Restart iteration K, as well as the time interval $[t_{\min}, t_{\max}]$ . We pick the $t_{min}$ and $t_{max}$ from the list of time steps in EDM discretization scheme with a number of steps 18. For example, for CIFAR-10 (VP) with NFE=75, we choose $t_{\min}=0.06$ , $t_{\max}=0.30$ , K=10, where 0.30/0.06 is the $12^{th}/14^{th}$ time step in the EDM scheme. We also adopt EDM scheme for the Restart backward process in $[t_{\min}, t_{\max}]$ . In addition, we apply the multi-level Restart strategy (Sec 4.3) to

![](images/f4fc7ad39395d6f7c418f77d8399fa0b4488fe3e0bf9cfe09c8a6463b79d7f46.jpg)

<details>
<summary>line</summary>

| NFE  | ODE  | Gonna Go Fast | Improved SDE | Restart |
| ---- | ---- | ------------- | ------------ | ------- |
| 32   | 3.0  | 3.4           | 3.4          | 2.7     |
| 64   | 2.9  | 2.9           | 2.9          | 2.4     |
| 128  | 2.9  | 2.6           | 2.6          | 2.2     |
| 256  | 2.9  | 2.7           | 2.4          | 2.2     |
| 512  | 2.9  | 2.5           | 2.4          | 2.1     |
| 1024 | 2.9  | 2.5           | 2.4          | 2.1     |
</details>

(a) CIFAR-10, VP

![](images/cfb9556aac37493b5495b0d288e8610ab8a368d575172fafc332a19c23167dd1.jpg)

<details>
<summary>line</summary>

| NFE  | ODE  | Improved SDE | Restart |
| ---- | ---- | ------------ | ------- |
| 32   | 2.50 | 3.00         | 2.40    |
| 64   | 2.40 | 2.75         | 2.10    |
| 128  | 2.30 | 2.00         | 1.70    |
| 256  | 2.25 | 1.75         | 1.50    |
| 512  | 2.25 | 1.50         | 1.40    |
| 1024 | 2.25 | 1.40         | 1.35    |
</details>

(b) ImageNet 64×64, EDM

Figure 3: FID versus NFE on (a) unconditional generation on CIFAR-10 with VP; (b) class-conditional generation on ImageNet with EDM.   
![](images/4b541d256dea0deedee6ee425bdacbe0a2dd9627567f05c57aa9c035e72894c5.jpg)

<details>
<summary>line</summary>

| NFE | DPM Solver | Restart |
| --- | --- | --- |
| 15 | 3.45 | 2.78 |
| 20 | 2.9 | 2.4 |
| 25 | 2.3 | 2.15 |
| 30 | 2.2 | 2.12 |
| 35 | 2.2 | 2.13 |
</details>

Figure 4: CIFAR-10, VP, in the low NFE regime. Restart consistently outperforms the DPM-solver with an NFE ranging from 16 to 36.

mitigate the error at early time steps for the more challenging ImageNet $64 \times 64$ . We provide the detailed Restart configurations in Appendix C.2.

For SDE, we compare with the previously best-performing stochastic samplers proposed by $[13]$ (Improved SDE). We use their optimal hyperparameters for each dataset. We also report the FID scores of the adaptive SDE $[12]$ (Gonna Go Fast) on CIFAR-10 (VP). Since the vanilla reverse-diffusion SDE $[23]$ has a significantly higher FID score, we omit its results from the main charts and defer them to Appendix D. For ODE samplers, we compare with the Heun's $2^{nd}$ order method $[2]$ (Heun), which arguably provides an excellent trade-off between discretization errors and NFE $[13]$ . To ensure a fair comparison, we use Heun's method as the sampler in the main/Restart backward processes in Restart.

We report the FID score versus NFE in Figure 3(a) and Table 1 on CIFAR-10, and Figure 3(b) on ImageNet $64 \times 64$ with EDM. Our main findings are: (1) Restart outperforms other SDE or ODE samplers in balancing quality and speed, across datasets and models. As demonstrated in the figures, Restart achieves a 10-fold / 2-fold acceleration compared to previous best SDE results on CIFAR-10 (VP) / ImageNet $64 \times 64$ (EDM) at the same FID score. In comparison to ODE sampler (Heun), Restart obtains a better FID score, with the gap increasing significantly with NFE. (2) For stronger models such as EDM and PFGM++, Restart further improve over the ODE baseline on CIFAR-10. In contrast, the Improved SDE negatively impacts performance of EDM, as also observed in [13]. It suggests that Restart incorporates stochasticity more effectively. (3) Restart establishes new state-of-the-art FID scores for UNet architectures without additional training. In particular, Restart achieves FID scores of 1.36 on class-cond. ImageNet $64 \times 64$ with EDM, and 1.88 on uncond. CIFAR-10 with PFGM++.

To further validate that Restart can be applied in low NFE regime, we show that one can employ faster ODE solvers such as the DPM-solver-3 [16] to further accelerate Restart. Fig. 4 shows that the Restart consistently outperforms the DPM-solver with an NFE ranging from 16 to 36. This demonstrates Restart's capability to excel over ODE samplers, even in the small NFE regime. It also suggests that Restart can consistently improve other ODE samplers, not limited to the DDIM, Heun. Surprisingly, when paired with the DPM-solver, Restart achieves an FID score of 2.11 on VP setting when NFE is 30, which is significantly lower than any previous numbers (even lower

than the SDE sampler with an NFE greater than 1000 in [23]), and make VP model on par with the performance with more advanced models (such as EDM). We include detailed Restart configuration in Table 3 in Appendix C.2.

Theorem 4 shows that each Restart iteration reduces the contracted errors while increasing the additional sampling errors in the backward process. In Fig. 5, we explore the choice of the number of Restart iterations K on CIFAR-10. We find that FID score initially improves and later worsens with increasing iterations K, with a smaller turning point for stronger EDM model. This supports the theoretical analysis that sampling errors will eventually outweigh the contraction benefits as K increases, and EDM only permits fewer Restart iterations due to smaller accumulated errors. It also suggests that, as a rule of thumb, we should apply greater Restart strength (e.g., larger K) for weaker or smaller architectures and vice v

Table 1: Uncond. CIFAR-10 with EDM and PFGM++ 

<table><tr><td></td><td>NFE</td><td>FID</td></tr><tr><td colspan="3">EDM-VP [13]</td></tr><tr><td rowspan="2">ODE (Heun)</td><td>63</td><td>1.97</td></tr><tr><td>35</td><td>1.97</td></tr><tr><td rowspan="2">Improved SDE</td><td>63</td><td>2.27</td></tr><tr><td>35</td><td>2.45</td></tr><tr><td>Restart</td><td>43</td><td>1.90</td></tr><tr><td colspan="3">PFGM++ [28]</td></tr><tr><td rowspan="2">ODE (Heun)</td><td>63</td><td>1.91</td></tr><tr><td>35</td><td>1.91</td></tr><tr><td>Restart</td><td>43</td><td>1.88</td></tr></table>

![](images/2f892ab9d5d1539c606b8d1dbfcbc4e713ee105ff7432ad71b9d925ddeef9bdf.jpg)

<details>
<summary>line</summary>

| Number of Restart iterations K | EDM (Restart) | EDM (ODE) | VP (Restart) | VP (ODE) |
| ------------------------------ | ------------- | --------- | ------------ | -------- |
| 0                              | 1.9           | 2.0       | 3.0          | 3.0      |
| 20                             | 2.1           | 2.0       | 2.2          | 2.2      |
| 40                             | 2.2           | 2.0       | 2.5          | 2.5      |
| 60                             | 2.3           | 2.0       | 2.9          | 2.9      |
</details>

Figure 5: FID score with a varying number of Restart iterations K.

# 5.3 Experiments on large-scale text-to-image model

![](images/479771c18b9df51c54995c0666049ab027ced6c86b266e05dc84c69bb1061896.jpg)  
(a) FID versus CLIP score

![](images/fc3e3cc05264feffc4ce2412522f2c410b5d14ec2d7dcd4b2f7b608cb35beb53.jpg)

<details>
<summary>line</summary>

| Aesthetic score | DDIM (Steps=50) | DDIM (Steps=100) | DDDM (Steps=100) | DDDM (Steps=200) | Restart (Steps=66) |
| --------------- | --------------- | ---------------- | ---------------- | ---------------- | ------------------ |
| 5.15            | 16.0            | 15.5             | 14.0             | 13.5             | 13.0               |
| 5.20            | 14.5            | 14.0             | 13.5             | 13.0             | 12.5               |
| 5.25            | 15.0            | 14.5             | 14.0             | 13.5             | 13.0               |
| 5.30            | 16.5            | 16.0             | 15.5             | 15.0             | 14.5               |
| 5.35            | 18.0            | 17.5             | 17.0             | 16.5             | 16.0               |
| 5.40            | 20.0            | 19.5             | 19.0             | 18.5             | 18.0               |
</details>

(b) FID versus Aesthetic score   
Figure 6: FID score versus (a) CLIP ViT-g/14 score and (b) Aesthetic score for text-to-image generation at $512 \times 512$ resolution, using Stable Diffusion v1.5 with a varying classifier-free guidance weight w = 2, 3, 5, 8.

We further apply Restart to the text-to-image Stable Diffusion v1.5 $^{2}$ pre-trained on LAION-5B [21] at a resolution of $512 \times 512$ . We employ the commonly used classifier-free guidance [8, 20] for sampling, wherein each sampling step entails two function evaluations – the conditional and unconditional predictions. Following [18, 20], we use the COCO [15] validation set for evaluation. We assess text-image alignment using the CLIP score [6] with the open-sourced ViT-g/14 [11], and measure diversity via the FID score. We also evaluate visual quality through the Aesthetic score, as rated by the LAION-Aesthetics Predictor V2 [24]. Following [17], we compute all evaluation metrics using 5K captions randomly sampled from the validation set and plot the trade-off curves between CLIP/Aesthetic scores and FID score, with the classifier-free guidance weight w in {2, 3, 5, 8}.

We compare with commonly used ODE sampler DDIM [22] and the stochastic sampler DDPM [9]. For Restart, we adopt the DDIM solver with 30 steps in the main backward process, and Heun in the Restart backward process, as we empirically find that Heun performs better than DDIM in the Restart. In addition, we select different sets of the hyperparameters for each guidance weight. For instance, when $w = 8$ , we use $[t_{\mathrm{min}}, t_{\mathrm{max}}] = [0.1, 2]$ , $K = 2$ and 10 steps in Restart backward process. We defer the detailed Restart configuration to Appendix C.2, and the results of Heun to Appendix D.1.

As illustrated in Fig. 6(a) and Fig. 6(b), Restart achieves better FID scores in most cases, given the same CLIP/Aesthetic scores, using only 132 function evaluations (i.e., 66 sampling steps). Remarkably, Restart achieves substantially lower FID scores than other samplers when CLIP/Aesthetic scores

![](images/3e62d95ab1caa882f05659dfec7c50579f74e4c2429d5d770554d44b58e87de0.jpg)

<details>
<summary>natural_image</summary>

Four-panel image showing animals: a horse on horseback, a raccoon, a red abstract bird, and a black duck on a stand (no text or symbols)
</details>

(a) Restart (Steps=66)

![](images/fec677c8a4f165f05606c54b075e2bad74bba84b38694222abb17f192406cdee.jpg)

<details>
<summary>natural_image</summary>

Collage of four nature scenes: a person on horseback, a rat, a red origami bird, and a black duck-shaped object (no text or symbols)
</details>

(b) DDIM (Steps=100)

![](images/12e85520bcdbbe972c31a01b1607bf822a7748d2b4a7404e8b7afbe499ee3cf3.jpg)

<details>
<summary>natural_image</summary>

Four-panel collage showing animals: a dog riding a horse, a rat facing away, an orange fox sculpture, and a duck-shaped model (no text or symbols)
</details>

(c) DDPM (Steps=100)   
Figure 7: Visualization of generated images with classifier-free guidance weight w = 8, using four text prompts ("A photo of an astronaut riding a horse on mars.", "A raccoon playing table tennis", "Intricate origami of a fox in a snowy forest" and "A transparent sculpture of a duck made out of glass") and the same random seeds.

are high (i.e., with larger w values). Conversely, Restart generally obtains a better text-image alignment/visual quality given the same FID. We also observe that DDPM generally obtains comparable performance with Restart in FID score when CLIP/Aesthetic scores are low, with Restart being more time-efficient. These findings suggest that Restart balances diversity (FID score) against text-image alignment (CLIP score) or visual quality (Aesthetic score) more effectively than previous samplers.

In Fig. 7, we visualize the images generated by Restart, DDIM and DDPM with w = 8. Compared to DDIM, the Restart generates images with superior details (e.g., the rendition of duck legs by DDIM is less accurate) and visual quality. Compared to DDPM, Restart yields more photo-realistic images (e.g., the astronaut). We provide extended of text-to-image generated samples in Appendix E.

# 6 Conclusion and future direction

In this paper, we introduce the Restart sampling for generative processes involving differential equations, such as diffusion models and PFGMs. By interweaving a forward process that adds a significant amount of noise with a corresponding backward ODE, Restart harnesses and even enhances the individual advantages of both ODE and SDE. Theoretically, Restart provides greater contraction effects of stochasticity while maintaining ODE-level discretization error. Empirically, Restart achieves a superior balance between quality and time, and improves the text-image alignment/visual quality and diversity trade-off in the text-to-image Stable Diffusion models.

A current limitation of the Restart algorithm is the absence of a principled way for hyperparameters selection, including the number of iterations K and the time interval $[t_{\min}, t_{\max}]$ . At present, we adjust these parameters based on the heuristic that weaker/smaller models, or more challenging tasks, necessitate a stronger Restart strength. In the future direction, we anticipate developing a more principled approach to automating the selection of optimal hyperparameters for Restart based on the error analysis of models, in order to fully unleash the potential of the Restart framework.

# Acknowledgements

YX and TJ acknowledge support from MIT-DSTA Singapore collaboration, from NSF Expeditions grant (award 1918839) "Understanding the World Through Code", and from MIT-IBM Grand Challenge project. Xiang Cheng acknowledges support from NSF CCF-2112665 (TILOS AI Research Institute).

# References

[1] Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313-326, 1982.   
[2] Uri M. Ascher and Linda R. Petzold. Computer methods for ordinary differential equations and differential-algebraic equations. In SIAM, 1998.   
[3] Andrei N Borodin and Paavo Salminen. Handbook of Brownian motion-facts and formulae. Springer Science & Business Media, 2015.   
[4] Arnak S Dalalyan and Avetik Karagulyan. User-friendly guarantees for the langevin monte carlo with inaccurate gradient. Stochastic Processes and their Applications, 129(12):5278–5311, 2019.   
[5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, K. Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.   
[6] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Joseph Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Conference on Empirical Methods in Natural Language Processing, 2021.   
[7] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NIPS, 2017.   
[8] Jonathan Ho. Classifier-free diffusion guidance. ArXiv, abs/2207.12598, 2022.   
[9] Jonathan Ho, Ajay Jain, and P. Abbeel. Denoising diffusion probabilistic models. ArXiv, abs/2006.11239, 2020.   
[10] Chutian Huang, Zijing Liu, Shengyuan Bai, Linwei Zhang, Chencheng Xu, Zhe Wang, Yang Xiang, and Yuanpeng Xiong. Pf-abgen: A reliable and efficient antibody generator via poisson flow. Machine Learning for Drug Discovery Workshop, International Conference on Learning Representations, 2023.   
[11] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip. Zenodo, 2021.   
[12] Alexia Jolicoeur-Martineau, Ke Li, Remi Piche-Taillefer, Tal Kachman, and Ioannis Mitliagkas. Gotta go fast when generating data with score-based models. ArXiv, abs/2105.14080, 2021.   
[13] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. ArXiv, abs/2206.00364, 2022.   
[14] Alex Krizhevsky. Learning multiple layers of features from tiny images. Citeseer, 2009.   
[15] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, 2014.   
[16] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. arXiv preprint arXiv:2206.00927, 2022.   
[17] Chenlin Meng, Ruiqi Gao, Diederik P. Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. ArXiv, abs/2210.03142, 2022.   
[18] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. In International Conference on Machine Learning, 2021.

[19] Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10674–10685, 2021.   
[20] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, Seyedeh Sara Mahdavi, Raphael Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. ArXiv, abs/2205.11487, 2022.   
[21] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models. ArXiv, abs/2210.08402, 2022.   
[22] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. ArXiv, abs/2010.02502, 2020.   
[23] Yang Song, Jascha Narain Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. ArXiv, abs/2011.13456, 2020.   
[24] LAION-AI Team. Laion-aesthetics predictor v2. https://github.com/christophschuhmann/improved-aesthetic-predictor, 2022.   
[25] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23:1661–1674, 2011.   
[26] Joseph L. Watson, David Juergens, Nathaniel R. Bennett, Brian L. Trippe, Jason Yim, Helen E. Eisenach, Woody Ahern, Andrew J. Borst, Robert J. Ragotte, Lukas F. Milles, Basile I. M. Wicky, Nikita Hanikel, Samuel J. Pellock, Alexis Courbet, William Sheffler, Jue Wang, Preetham Venkatesh, Isaac Sappington, Susana Vázquez Torres, Anna Lauko, Valentin De Bortoli, Emile Mathieu, Regina Barzilay, T. Jaakkola, Frank DiMaio, Minkyung Baek, and David Baker. Broadly applicable and accurate protein design by integrating structure prediction networks and diffusion generative models. bioRxiv, 2022.   
[27] Yilun Xu, Ziming Liu, Max Tegmark, and T. Jaakkola. Poisson flow generative models. ArXiv, abs/2209.11178, 2022.   
[28] Yilun Xu, Ziming Liu, Yonglong Tian, Shangyuan Tong, Max Tegmark, and T. Jaakkola. Pfgm++: Unlocking the potential of physics-inspired generative models. ArXiv, abs/2302.04265, 2023.

# Appendix

# A Proofs of Main Theoretical Results

In this section, we provide proofs of our main results. We define below some crucial notations which we will use throughout. We use ODE( $\ldots$ ) to denote the backwards ODE under exact score $\nabla \log p_{t}(x)$ . More specifically, given any $x \in R^{d}$ and s > r > 0, let $x_{t}$ denote the solution to the following ODE:

$$
d x _ {t} = - t \nabla \log p _ {t} (x _ {t}) d t. \tag {5}
$$

$\mathrm{ODE}(x, s \to r)$ is defined as "the value of $x_r$ when initialized at $x_s = x$ ". It will also be useful to consider a "time-discretized ODE with drift $ts_\theta(x, t)$ : let $\delta$ denote the discretization step size and let $k$ denote any integer. Let $\delta$ denote a step size, let $\overline{x}_t$ denote the solution to

$$
d \overline {{x}} _ {t} = - t s _ {\theta} (x _ {k \delta}, k \delta) d t, \tag {6}
$$

where for any $t$ , $k$ is the unique integer such that $t \in ((k - 1)\delta, k\delta]$ . We verify that the dynamics of Eq. (6) is equivalent to the following discrete-time dynamics for $t = k\delta$ , $k \in \mathbb{Z}$ :

$$
\overline {{x}} _ {(k - 1) \delta} = \overline {{x}} _ {k \delta} - \frac {1}{2} \left(\left((k - 1) \delta\right) ^ {2} - (k \delta) ^ {2}\right) s _ {\theta} (x _ {k \delta}, k \delta).
$$

We similarly denote the value of $\overline{x}_r$ when initialized at $\overline{x}_s = x$ as $\mathrm{ODE}_{\theta}(x, s \to r)$ . Analogously, we let $\mathrm{SDE}(x, s \to r)$ and $\mathrm{SDE}_{\theta}(x, s \to r)$ denote solutions to

$$
d y _ {t} = - 2 t \nabla \log p _ {t} (y _ {t}) d t + \sqrt {2 t} d B _ {t}
$$

$$
d \overline {{y}} _ {t} = - 2 t s _ {\theta} (\overline {{y}} _ {t}, t) d t + \sqrt {2 t} d B _ {t}
$$

respectively. Finally, we will define the $Restart_{\theta}$ process as follows:

(Restart $_{\theta}$ forward process) $x_{t_{\max}}^{i+1} = x_{t_{\min}}^{i} + \varepsilon_{t_{\min} \to t_{\max}}^{i}$

(Restart $_{\theta}$ backward process) $x_{t_{\min}}^{i+1} = \text{ODE}_{\theta}(x_{t_{\max}}^{i+1}, t_{\max} \to t_{\min})$ , (7)

where $\varepsilon_{t_{\min}\to t_{\max}}^{i}\sim\mathcal{N}\left(\mathbf{0},\left(t_{\max}^{2}-t_{\min}^{2}\right)\boldsymbol{I}\right)$ . We use $\operatorname{Restart}_{\theta}(x,K)$ to denote $x_{t_{\min}}^{K}$ in the above processes, initialized at $x_{t_{\min}}^{0}=x$ . In various theorems, we will refer to a function $Q(r):\mathbb{R}^{+}\to[0,1/2)$ , defined as the Gaussian tail probability $Q(r)=Pr(a\geq r)$ for $a\sim\mathcal{N}(0,1)$ .

# A.1 Main Result

Theorem 3. [Formal version of Theorem 1] Let $t_{max}$ be the initial noise level. Let the initial random variables $\overline{x}_{t_{max}} = \overline{y}_{t_{max}}$ , and

$$
\overline {{x}} _ {t _ {\min}} = O D E _ {\theta} \left(\overline {{x}} _ {t _ {\max}}, t _ {\max} \rightarrow t _ {\min}\right)
$$

$$
\overline {{y}} _ {t _ {m i n}} = S D E _ {\theta} (\overline {{y}} _ {t _ {m a x}}, t _ {m a x} \to t _ {m i n}),
$$

Let $p_t$ denote the true population distribution at noise level $t$ . Let $p_t^{ODE_\theta}$ , $p_t^{SDE_\theta}$ denote the distributions for $x_t, y_t$ respectively. Assume that for all $x, y, s, t$ , $s_\theta(x, t)$ satisfies $\|ts_\theta(x, t) - ts_\theta(x, s)\| \leq L_0 |s - t|$ , $\|ts_\theta(x, t)\| \leq L_1$ , $\|ts_\theta(x, t) - ts_\theta(y, t)\| \leq L_2 \|x - y\|$ , and the approximation error $\|ts_\theta(x, t) - t\nabla \log p_t(x)\| \leq \epsilon_{approx}$ . Assume in addition that $\forall t \in [t_{min}, t_{max}]$ , $\|x_t\| < B/2$ for any $x_t$ in the support of $p_t$ , $p_t^{ODE_\theta}$ or $p_t^{SDE_\theta}$ , and $K \leq \frac{C}{L_2(t_{max} - t_{min})}$ for some universal constant $C$ . Then

$$
\begin{array}{l} W _ {1} (p _ {t _ {\min}} ^ {O D E _ {\theta}}, p _ {t _ {\min}}) \leq B \cdot T V \left(p _ {t _ {\max}} ^ {O D E _ {\theta}}, p _ {t _ {\max}}\right) \\ + e ^ {L _ {2} \left(t _ {\text { max }} - t _ {\text { min }}\right)} \cdot \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {\text { approx }}\right) \left(t _ {\text { max }} - t _ {\text { min }}\right) \tag {8} \\ \end{array}
$$

$$
W _ {1} (p _ {t _ {m i n}} ^ {S D E _ {\theta}}, p _ {t _ {m i n}}) \leq B \cdot \left(1 - \lambda e ^ {- B L _ {1} / t _ {m i n} - L _ {1} ^ {2} t _ {m a x} ^ {2} / t _ {m i n} ^ {2}}\right) T V (p _ {t _ {m a x}} ^ {S D E _ {\theta}}, p _ {t _ {m a x}})
$$

$$
+ e ^ {2 L _ {2} (t _ {\max} - t _ {\min})} \left(\epsilon_ {\text { approx }} + \delta L _ {0} + L _ {2} \left(\delta L _ {1} + \sqrt {2 \delta d t _ {\max}}\right)\right) (t _ {\max} - t _ {\min}) \tag {9}
$$

where $\lambda := 2Q\left(\frac{B}{2\sqrt{t_{max}^2 - t_{min}^2}}\right)$ .

Proof. Let us define $x_{t_{\max}} \sim p_{t_{\max}}$ , and let $x_{t_{\min}} = \mathrm{ODE}(x_{t_{\max}}, t_{\max} \to t_{\min})$ . We verify that $x_{t_{\min}}$ has density $p_{t_{\min}}$ . Let us also define $\hat{x}_{t_{\min}} = \mathrm{ODE}_{\theta}(x_{t_{\max}}, t_{\max} \to t_{\min})$ . We would like to bound the Wasserstein distance between $\bar{x}_{t_{\min}}$ and $x_{t_{\min}}$ (i.e., $p_{t_{\min}}^{ODE_{\theta}}$ and $p_{t_{\min}}$ ), by the following triangular inequality:

$$
W _ {1} \left(\bar {x} _ {t _ {\min}}, x _ {t _ {\min}}\right) \leq W _ {1} \left(\bar {x} _ {t _ {\min}}, \hat {x} _ {t _ {\min}}\right) + W _ {1} \left(\hat {x} _ {t _ {\min}}, x _ {t _ {\min}}\right) \tag {10}
$$

By Lemma 2, we know that

$$
\left\| \hat {x} _ {t _ {\min}} - x _ {t _ {\min}} \right\| \leq e ^ {(t _ {\max} - t _ {\min}) L _ {2}} \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {a p p r o x}\right) (t _ {\max} - t _ {\min}),
$$

where we use the fact that $\|\hat{x}_{t_{\max}} - x_{t_{\max}}\| = 0$ . Thus we immediately have

$$
W _ {1} \left(\hat {x} _ {t _ {\min}}, x _ {t _ {\min}}\right) \leq e ^ {\left(t _ {\max} - t _ {\min}\right) L _ {2}} \left(\delta \left(L _ {2} L _ {1} + L _ {0}\right) + \epsilon_ {\text { a   p   p   r   o   x }}\right) \left(t _ {\max} - t _ {\min}\right) \tag {11}
$$

On the other hand,

$$
\begin{array}{l} W _ {1} \left(\hat {x} _ {t _ {\min}}, \overline {{x}} _ {t _ {\min}}\right) \leq B \cdot T V \left(\hat {x} _ {t _ {\min}}, \overline {{x}} _ {t _ {\min}}\right) \\ \leq B \cdot T V \left(\hat {x} _ {t _ {\max}}, \overline {{{x}}} _ {t _ {\max}}\right) \tag {12} \\ \end{array}
$$

where the last equality is due to the data-processing inequality. Combining Eq. (11), Eq. (12) and the triangular inequality Eq. (10), we arrive at the upper bound for ODE (Eq. (8)). The upper bound for SDE (Eq. (9)) shares a similar proof approach. First, let $y_{t_{\max}} \sim p_{t_{\max}}$ . Let $\hat{y}_{t_{\min}} = \text{SDE}_{\theta}(y_{t_{\max}}, t_{\max} \to t_{\min})$ . By Lemma 5,

$$
T V \left(\hat {y} _ {t _ {\min}}, \overline {{y}} _ {t _ {\min}}\right) \leq \left(1 - 2 Q \left(\frac {B}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right) \cdot e ^ {- B L _ {1} / t _ {\min} - L _ {1} ^ {2} t _ {\max} ^ {2} / t _ {\min} ^ {2}}\right) \cdot T V \left(\hat {y} _ {t _ {\max}}, \overline {{y}} _ {t _ {\max}}\right)
$$

On the other hand, by Lemma 4,

$$
\mathbb {E} \left[ \| \hat {y} _ {t _ {\min}} - y _ {t _ {\min}} \| \right] \leq e ^ {2 L _ {2} (t _ {\max} - t _ {\min})} \left(\epsilon_ {a p p r o x} + \delta L _ {0} + L _ {2} \left(\delta L _ {1} + \sqrt {2 \delta d t _ {\max}}\right)\right) (t _ {\max} - t _ {\min}).
$$

The SDE triangular upper bound on $W_{1}(\bar{y}_{t_{\mathrm{min}}}, y_{t_{\mathrm{min}}})$ follows by multiplying the first inequality by B (to bound $W_{1}(\bar{y}_{t_{\mathrm{min}}}, \hat{y}_{t_{\mathrm{min}}})$ ) and then adding the second inequality (to bound $W_{1}(y_{t_{\mathrm{min}}}, \hat{y}_{t_{\mathrm{min}}})$ ). Notice that by definition, $TV\left(\hat{y}_{t_{\mathrm{max}}}, \overline{y}_{t_{\mathrm{max}}}\right) = TV\left(y_{t_{\mathrm{max}}}, \overline{y}_{t_{\mathrm{max}}}\right)$ . Finally, because of the assumption that $K \leq \frac{C}{L_{2}(t_{\mathrm{max}} - t_{\mathrm{min}})}$ for some universal constant, we summarize the second term in the Eq. (8) and Eq. (9) into the big O in the informal version Theorem 1.

Theorem 4. [Formal version of Theorem 2] Consider the same setting as Theorem 3. Let $p_{t_{min}}^{Restart_{\theta},i}$ denote the distributions after $i^{th}$ Restart iteration, i.e., the distribution of $\overline{x}_{t_{min}}^i = Restart_{\theta}(\overline{x}_{t_{min}}^0,i)$ . Given initial $\overline{x}_{t_{max}}^0\sim p_{t_{max}}^{Restart,0}$ , let $\overline{x}_{t_{min}}^0 = ODE_{\theta}(\overline{x}_{t_{max}}^0,t_{max}\to t_{min})$ . Then

$$
\begin{array}{l} W _ {1} (p _ {t _ {m i n}} ^ {R e s t a r t _ {\theta}, K}, p _ {t _ {m i n}}) \leq \underbrace {B \cdot (1 - \lambda) ^ {K} T V (p _ {t _ {m a x}} ^ {R e s t a r t , 0} , p _ {t _ {m a x}})} _ {\text { upper   bound   on   contracted   error }} \\ + \underbrace {e ^ {(K + 1) L _ {2} (t _ {\text { max }} - t _ {\text { min }})} (K + 1) \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {\text { approx }}\right) (t _ {\text { max }} - t _ {\text { min }})} _ {\text { upper   bound   on   additional   sampling   error }} \tag {13} \\ \end{array}
$$

$$
w h e r e \lambda = 2 Q \left(\frac {B}{2 \sqrt {t _ {\text { max }} ^ {2} - t _ {\text { min }} ^ {2}}}\right).
$$

Proof. Let $x_{t_{\max}}^0 \sim p_{t_{\max}}$ . Let $x_{t_{\min}}^K = \text{Restart}(x_{t_{\min}}^0, K)$ . We verify that $x_{t_{\min}}^K$ has density $p_{t_{\min}}$ . Let us also define $\hat{x}_{t_{\min}}^0 = \text{ODE}_\theta(x_{t_{\max}}^0, t_{\max} \to t_{\min})$ and $\hat{x}_{t_{\min}}^K = \text{Restart}_\theta(\hat{x}_{t_{\min}}^0, K)$ .

By Lemma 1,

$$
\begin{array}{l} T V \left(\overline {{x}} _ {t _ {\min}} ^ {K}, \hat {x} _ {t _ {\min}} ^ {K}\right) \leq \left(1 - 2 Q \left(\frac {B}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right)\right) ^ {K} T V \left(\overline {{x}} _ {t _ {\min}} ^ {0}, \hat {x} _ {t _ {\min}} ^ {0}\right) \\ \leq \left(1 - 2 Q \left(\frac {B}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right)\right) ^ {K} T V \left(\overline {{x}} _ {t _ {\max}} ^ {0}, \hat {x} _ {t _ {\max}} ^ {0}\right) \\ = \left(1 - 2 Q \left(\frac {B}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right)\right) ^ {K} T V \left(\overline {{x}} _ {t _ {\max}} ^ {0}, x _ {t _ {\max}} ^ {0}\right) \\ \end{array}
$$

The second inequality holds by data processing inequality. The above can be used to bound the 1-Wasserstein distance as follows:

$$
W _ {1} \left(\bar {x} _ {t _ {\min}} ^ {K}, \hat {x} _ {t _ {\min}} ^ {K}\right) \leq B \cdot T V \left(\bar {x} _ {t _ {\min}} ^ {K}, \hat {x} _ {t _ {\min}} ^ {K}\right) \leq \left(1 - 2 Q \left(\frac {B}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right)\right) ^ {K} T V \left(\bar {x} _ {t _ {\max}} ^ {0}, x _ {t _ {\max}} ^ {0}\right) \tag {14}
$$

On the other hand, using Lemma 3,

$$
W _ {1} \left(x _ {t _ {\min}} ^ {K}, \hat {x} _ {t _ {\min}} ^ {K}\right) \leq \left\| x _ {t _ {\min}} ^ {K} - \hat {x} _ {t _ {\min}} ^ {K} \right\|
$$

$$
\leq e ^ {(K + 1) L _ {2} (t _ {\max} - t _ {\min})} (K + 1) \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {a p p r o x}\right) (t _ {\max} - t _ {\min}) \tag {15}
$$

We arrive at the result by combining the two bounds above (Eq. (14), Eq. (15)) with the following triangular inequality,

$$
W _ {1} (\bar {x} _ {t _ {\mathrm{min}}} ^ {K}, x _ {t _ {\mathrm{min}}} ^ {K}) \leq W _ {1} (\bar {x} _ {t _ {\mathrm{min}}} ^ {K}, \hat {x} _ {t _ {\mathrm{min}}} ^ {K}) + W _ {1} (\hat {x} _ {t _ {\mathrm{min}}} ^ {K}, x _ {t _ {\mathrm{min}}} ^ {K})
$$

![](images/6fd1c34c2d85e56a420acaf91f090b9c53f211d475377db317c983d4973e2cfb.jpg)

# A.2 Mixing under Restart with exact ODE

Lemma 1. Consider the same setup as Theorem 4. Consider the Restart $_{\theta}$ process defined in equation 7. Let

$$
x _ {t _ {m i n}} ^ {i} = R e s t a r t _ {\theta} (x _ {t _ {m i n}} ^ {0}, i)
$$

$$
y _ {t _ {m i n}} ^ {i} = R e s t a r t _ {\theta} (y _ {t _ {m i n}} ^ {0}, i).
$$

Let $p_t^{Restart_\theta(i)}$ and $q_t^{Restart_\theta(i)}$ denote the densities of $x_t^i$ and $y_t^i$ respectively. Then

$$
T V \left(p _ {t _ {\min}} ^ {R e s t a r t _ {\theta} (K)}, q _ {t _ {\min}} ^ {R e s t a r t _ {\theta} (K)}\right) \leq (1 - \lambda) ^ {K} T V \left(p _ {t _ {\min}} ^ {R e s t a r t _ {\theta} (0)}, q _ {t _ {\min}} ^ {R e s t a r t _ {\theta} (0)}\right),
$$

where $\lambda = 2Q\left(\frac{B}{2\sqrt{t_{max}^2 - t_{min}^2}}\right)$ .

Proof. Conditioned on $x_{t_{\min}}^i, y_{t_{\min}}^i$ , let $x_{t_{\max}}^{i+1} = x_{t_{\min}}^i + \sqrt{t_{\max}^2 - t_{\min}^2} \xi_i^x$ and $y_{t_{\max}}^{i+1} = y_{t_{\min}}^i + \sqrt{t_{\max}^2 - t_{\min}^2} \xi_i^y$ . We now define a coupling between $x_{t_{\min}}^{i+1}$ and $y_{t_{\min}}^{i+1}$ by specifying the joint distribution over $\xi_i^x$ and $\xi_i^y$ .

If $x_{t_{\min}}^{i} = y_{t_{\min}}^{i}$ , let $\xi_{i}^{x} = \xi_{i}^{y}$ , so that $x_{t_{\min}}^{i+1} = y_{t_{\min}}^{i+1}$ . On the other hand, if $x_{t_{\min}}^{i} \neq y_{t_{\min}}^{i}$ , let $x_{t_{\max}}^{i+1}$ and $y_{t_{\max}}^{i+1}$ be coupled as described in the proof of Lemma 7, with $x' = x_{t_{\max}}^{i+1}$ , $y' = y_{t_{\max}}^{i+1}$ , $\sigma = \sqrt{t_{\max}^{2} - t_{\min}^{2}}$ . Under this coupling, we verify that,

$$
\begin{array}{l} \mathbb {E} \left[ \mathbb {1} \left\{x _ {t _ {\min}} ^ {i + 1} \neq y _ {t _ {\min}} ^ {i + 1} \right\} \right] \\ \leq \mathbb {E} \left[ \mathbb {1} \left\{x _ {t _ {\max}} ^ {i + 1} \neq y _ {t _ {\max}} ^ {i + 1} \right\} \right] \\ \leq \mathbb {E} \left[ \left(1 - 2 Q \left(\frac {\left\| x _ {t _ {\min}} ^ {i} - y _ {t _ {\min}} ^ {i} \right\|}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right)\right) \mathbb {1} \left\{x _ {t _ {\min}} ^ {i} \neq y _ {t _ {\min}} ^ {i} \right\} \right] \\ \leq \left(1 - 2 Q \left(\frac {B}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right)\right) \mathbb {E} \left[ \mathbb {1} \left\{x _ {t _ {\min}} ^ {i} \neq y _ {t _ {\min}} ^ {i} \right\} \right]. \\ \end{array}
$$

Applying the above recursively,

$$
\mathbb {E} \left[ \mathbb {1} \left\{x _ {t _ {\min}} ^ {K} \neq y _ {t _ {\min}} ^ {K} \right\} \right] \leq \left(1 - 2 Q \left(\frac {B}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right)\right) ^ {K} \mathbb {E} \left[ \mathbb {1} \left\{x _ {t _ {\min}} ^ {0} \neq y _ {t _ {\min}} ^ {0} \right\} \right].
$$

The conclusion follows by noticing that $TV\left(p_{t_{\min}}^{\text{Restart}_{\theta}(K)}, q_{t_{\min}}^{\text{Restart}_{\theta}(K)}\right) \leq Pr\left(x_{t_{\min}}^{K} \neq y_{t_{\min}}^{K}\right) = \mathbb{E}\left[\mathbb{1}\left\{x_{t_{\min}}^{K} \neq y_{t_{\min}}^{K}\right\}\right]$ , and by selecting the initial coupling so that $Pr\left(x_{t_{\min}}^{0} \neq y_{t_{\min}}^{0}\right) = TV\left(p_{t_{\min}}^{\text{Restart}_{\theta}(0)}, q_{t_{\min}}^{\text{Restart}_{\theta}(0)}\right)$ .

# A.3 $W_{1}$ discretization bound

Lemma 2 (Discretization bound for ODE). Let $x_{t_{min}} = ODE(x_{t_{max}}, t_{max} \to t_{min})$ and let $\overline{x}_{t_{min}} = ODE_{\theta}(\overline{x}_{t_{max}}, t_{max} \to t_{min})$ . Assume that for all $x, y, s, t, s_{\theta}(x, t)$ satisfies $\|ts_{\theta}(x, t) - ts_{\theta}(x, s)\| \leq L_0 |s - t|$ , $\|ts_{\theta}(x, t)\| \leq L_1$ and $\|ts_{\theta}(x, t) - ts_{\theta}(y, t)\| \leq L_2 \|x - y\|$ . Then

$$
\left\| x _ {t _ {\min}} - \overline {{x}} _ {t _ {\min}} \right\| \leq e ^ {(t _ {\max} - t _ {\min}) L _ {2}} \left(\left\| x _ {t _ {\max}} - \overline {{x}} _ {t _ {\max}} \right\| + \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {a p p r o x}\right) (t _ {\max} - t _ {\min})\right)
$$

Proof. Consider some fixed arbitrary k, and recall that $\delta$ is the step size. Recall that by definition of ODE and $ODE_{\theta}$ , for $t \in ((k - 1)\delta, k\delta]$ ,

$$
d x _ {t} = - t \nabla \log p _ {t} (x _ {t}) d t
$$

$$
d \overline {{x}} _ {t} = - t s _ {\theta} (\overline {{x}} _ {k \delta}, k \delta) d t.
$$

For $t \in [t_{\min}, t_{\max}]$ , let us define a time-reversed process $x_t^{\leftarrow} := x_{-t}$ . Let $v(x, t) := \nabla \log p_{-t}(x)$ . Then for $t \in [-t_{\max}, -t_{\min}]$

$$
d x _ {t} ^ {\leftarrow} = t v (x _ {t} ^ {\leftarrow}, t) d s.
$$

Similarly, define $\overline{x}_t^{\leftarrow} := \overline{x}_{-t}$ and $\overline{v}(x,t) := s_{\theta}(x,-t)$ . It follows that

$$
d \overline {{x}} _ {t} ^ {\leftarrow} = t \overline {{v}} (\overline {{x}} _ {k \delta} ^ {\leftarrow}, k \delta) d s,
$$

where k is the unique (negative) integer satisfying $t \in [k\delta, (k + 1)\delta)$ . Following these definitions,

$$
\begin{array}{l} \frac {d}{d t} \left\| x _ {t} ^ {\leftarrow} - \overline {{x}} _ {t} ^ {\leftarrow} \right\| \\ \leq \left\| t v \left(x _ {t} ^ {\leftarrow}, t\right) - t \bar {v} \left(x _ {t} ^ {\leftarrow}, t\right) \right\| \\ + \| t \overline {{v}} (x _ {t} ^ {\leftarrow}, t) - t \overline {{v}} (\overline {{x}} _ {t} ^ {\leftarrow}, t) \| \\ + \| t \overline {{v}} (\overline {{x}} _ {t} ^ {\leftarrow}, t) - t \overline {{v}} (\overline {{x}} _ {t} ^ {\leftarrow}, k \delta) \| \\ + \| t \overline {{v}} (\overline {{x}} _ {t} ^ {\leftarrow}, k \delta) - t \overline {{v}} (\overline {{x}} _ {k \delta} ^ {\leftarrow}, k \delta) \| \\ \leq \epsilon_ {a p p r o x} + L _ {2} \| x _ {t} ^ {\leftarrow} - \overline {{x}} _ {t} ^ {\leftarrow} \| + \delta L _ {0} + L _ {2} \| \overline {{x}} _ {t} ^ {\leftarrow} - \overline {{x}} _ {k \delta} ^ {\leftarrow} \| \\ \leq \epsilon_ {a p p r o x} + L _ {2} \left\| x _ {t} ^ {\leftarrow} - \overline {{x}} _ {t} ^ {\leftarrow} \right\| + \delta L _ {0} + \delta L _ {2} L _ {1}. \\ \end{array}
$$

Applying Gronwall's Lemma over the interval $t \in [-t_{\max}, -t_{\min}]$ ,

$$
\begin{array}{l} \left\| x _ {t _ {\min}} - \overline {{x}} _ {t _ {\min}} \right\| \\ = \left\| x _ {- t _ {\min}} ^ {\leftarrow} - \overline {{x}} _ {- t _ {\min}} ^ {\leftarrow} \right\| \\ \leq e ^ {L _ {2} \left(t _ {\max} - t _ {\min}\right)} \left(\left\| x _ {- t _ {\max}} ^ {\leftarrow} - \overline {{x}} _ {- t _ {\max}} ^ {\leftarrow} \right\| + \left(\epsilon_ {a p p r o x} + \delta L _ {0} + \delta L _ {2} L _ {1}\right) \left(t _ {\max} - t _ {\min}\right)\right) \\ = e ^ {L _ {2} \left(t _ {\max} - t _ {\min}\right)} \left(\left\| x _ {t _ {\max}} - \overline {{x}} _ {t _ {\max}} \right\| + \left(\epsilon_ {a p p r o x} + \delta L _ {0} + \delta L _ {2} L _ {1}\right) \left(t _ {\max} - t _ {\min}\right)\right). \\ \end{array}
$$

![](images/674d95dec68d21f745b42388e12cd9807ded5fa31a05c1ae0ac83f08708c6fa1.jpg)

Lemma 3. Given initial $x_{t_{max}}^0$ , let $x_{t_{min}}^0 = ODE(x_{t_{max}}^0, t_{max} \to t_{min})$ , and let $\hat{x}_{t_{min}}^0 = ODE_\theta(x_{t_{max}}^0, t_{max} \to t_{min})$ . We further denote the variables after $K$ Restart iterations as $x_{t_{min}}^K = \text{Restart}(x_{t_{min}}^0, K)$ and $\hat{x}_{t_{min}}^K = \text{Restart}_\theta(\hat{x}_{t_{min}}^0, K)$ , with true field and learned field respectively. Then there exists a coupling between $x_{t_{min}}^K$ and $\hat{x}_{t_{min}}^K$ such that

$$
\left\| x _ {t _ {\min}} ^ {K} - \hat {x} _ {t _ {\min}} ^ {K} \right\| \leq e ^ {(K + 1) L _ {2} (t _ {\max} - t _ {\min})} (K + 1) \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {a p p r o x}\right) (t _ {\max} - t _ {\min}).
$$

Proof. We will couple $x_{t_{\min}}^{i}$ and $\hat{x}_{t_{\min}}^{i}$ by using the same noise $\varepsilon_{t_{\min}\to t_{\max}}^{i}$ in the Restart forward process for $i=0\ldots K-1$ (see Eq. (7)). For any i, let us also define $y_{t_{\min}}^{i,j}:=\operatorname{Restart}_{\theta}\left(x_{t_{\min}}^{i},j-i\right)$ , and this process uses the same noise $\varepsilon_{t_{\min}\to t_{\max}}^{i}$ as previous ones. From this definition, $y_{t_{\min}}^{K,K}=x_{t_{\min}}^{K}$ . We can thus bound

$$
\left\| x _ {t _ {\min}} ^ {K}, \hat {x} _ {t _ {\min}} ^ {K} \right\| \leq \left\| y _ {t _ {\min}} ^ {0, K} - \hat {x} _ {t _ {\min}} ^ {K} \right\| + \sum_ {i = 0} ^ {K - 1} \left\| y _ {t _ {\min}} ^ {i, K} - y _ {t _ {\min}} ^ {i + 1, K} \right\| \tag {16}
$$

Using the assumption that $ts_{\theta}(\cdot ,t)$ is $L_{2}$ Lipschitz,

$$
\begin{array}{l} \left| \left| y _ {t _ {\min}} ^ {0, i + 1} - \hat {x} _ {t _ {\min}} ^ {i + 1} \right| \right| \\ = \left\| \mathrm{ODE} _ {\theta} (y _ {t _ {\max}} ^ {0, i}, t _ {\max} \rightarrow t _ {\min}) - \mathrm{ODE} _ {\theta} (\hat {x} _ {t _ {\max}} ^ {i}, t _ {\max} \rightarrow t _ {\min}) \right\| \\ \leq e ^ {L _ {2} \left(t _ {\max} - t _ {\min}\right)} \left\| y _ {t _ {\max}} ^ {0, i} - \hat {x} _ {t _ {\max}} ^ {i} \right\| \\ = e ^ {L _ {2} (t _ {\max} - t _ {\min})} \left\| y _ {t _ {\min}} ^ {0, i} - \hat {x} _ {t _ {\min}} ^ {i} \right\|, \\ \end{array}
$$

where the last equality is because we add the same additive Gaussian noise $\varepsilon_{t_{\mathrm{min}}\to t_{\mathrm{max}}}^{i}$ to $y_{t_{\mathrm{min}}}^{0,i}$ and $\hat{x}_{t_{\mathrm{min}}}^{i}$ in the Restart forward process. Applying the above recursively, we get

$$
\begin{array}{l} \left\| y _ {t _ {\min}} ^ {0, K} - \hat {x} _ {t _ {\min}} ^ {K} \right\| \leq e ^ {K L _ {2} (t _ {\max} - t _ {\min})} \left\| y _ {t _ {\min}} ^ {0, 0} - \hat {x} _ {t _ {\min}} ^ {0} \right\| \\ \leq e ^ {K L _ {2} (t _ {\mathrm{max}} - t _ {\mathrm{min}})} \left\| x _ {t _ {\mathrm{min}}} ^ {0} - \hat {x} _ {t _ {\mathrm{min}}} ^ {0} \right\| \\ \leq e ^ {(K + 1) L _ {2} (t _ {\max} - t _ {\min})} \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {a p p r o x}\right) (t _ {\max} - t _ {\min}), \tag {17} \\ \end{array}
$$

where the last line follows by Lemma 2 when setting $x_{t_{\max}} = \bar{x}_{t_{\max}}$ . We will now bound $\left\| y_{t_{\min}}^{i,K} - y_{t_{\min}}^{i + 1,K}\right\|$ for some $i \leq K$ . It follows from definition that

$$
\begin{array}{l} y _ {t _ {\min}} ^ {i, i + 1} = \mathrm{ODE} _ {\theta} \left(x _ {t _ {\max}} ^ {i}, t _ {\max} \rightarrow t _ {\min}\right) \\ y _ {t _ {\min}} ^ {i + 1, i + 1} = x _ {t _ {\min}} ^ {i + 1} = \mathrm{ODE} \left(x _ {t _ {\max}} ^ {i}, t _ {\max} \rightarrow t _ {\min}\right). \\ \end{array}
$$

By Lemma 2,

$$
\left\| y _ {t _ {\min}} ^ {i, i + 1} - y _ {t _ {\min}} ^ {i + 1, i + 1} \right\| \leq e ^ {L _ {2} (t _ {\max} - t _ {\min})} \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {a p p r o x}\right) (t _ {\max} - t _ {\min})
$$

For the remaining steps from $i + 2 \ldots K$ , both $y^{i,\cdot}$ and $y^{i+1,\cdot}$ evolve with $ODE_{\theta}$ in each step. Again using the assumption that $ts_{\theta}(\cdot,t)$ is $L_{2}$ Lipschitz,

$$
\left\| y _ {t _ {\min}} ^ {i, K} - y _ {t _ {\min}} ^ {i + 1, K} \right\| \leq e ^ {(K - i) L _ {2} (t _ {\max} - t _ {\min})} \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {a p p r o x}\right) (t _ {\max} - t _ {\min})
$$

Summing the above for $i = 0 \ldots K - 1$ , and combining with Eq. (16) and Eq. (17) gives

$$
\left\| x _ {t _ {\min}} ^ {K} - \hat {x} _ {t _ {\min}} ^ {K} \right\| \leq e ^ {(K + 1) L _ {2} (t _ {\max} - t _ {\min})} (K + 1) \left(\delta (L _ {2} L _ {1} + L _ {0}) + \epsilon_ {a p p r o x}\right) (t _ {\max} - t _ {\min}).
$$

![](images/1d83d147a346c7d039a6b00595c9471cc7f255fccffd0f798baf754e7e392c7f.jpg)

Lemma 4. Consider the same setup as Theorem 3. Let $x_{t_{min}} = SDE(x_{t_{max}}, t_{max} \to t_{min})$ and let $\overline{x}_{t_{min}} = SDE(\overline{x}_{t_{max}}, t_{max} \to t_{min})$ . Then there exists a coupling between $x_t$ and $\overline{x}_t$ such that

$$
\begin{array}{l} \mathbb {E} \left[ \| x _ {t _ {m i n}} - \overline {{x}} _ {t _ {m i n}} \| \right] \leq e ^ {2 L _ {2} (t _ {m a x} - t _ {m i n})} \mathbb {E} \left[ \| x _ {t _ {m a x}} - \overline {{x}} _ {t _ {m a x}} \| \right] \\ + e ^ {2 L _ {2} (t _ {m a x} - t _ {m i n})} \left(\epsilon_ {a p p r o x} + \delta L _ {0} + L _ {2} \left(\delta L _ {1} + \sqrt {2 \delta d t _ {m a x}}\right)\right) (t _ {m a x} - t _ {m i n}) \\ \end{array}
$$

Proof. Consider some fixed arbitrary $k$ , and recall that $\delta$ is the stepsize. By definition of SDE and $\mathrm{SDE}_{\theta}$ , for $t \in ((k - 1)\delta, k\delta]$ ,

$$
d x _ {t} = - 2 t \nabla \log p _ {t} (x _ {t}) d t + \sqrt {2 t} d B _ {t}
$$

$$
d \overline {{x}} _ {t} = - 2 t s _ {\theta} (\overline {{x}} _ {k \delta}, k \delta) d t + \sqrt {2 t} d B _ {t}.
$$

Let us define a coupling between $x_{t}$ and $\overline{x}_{t}$ by identifying their respective Brownian motions. It will be convenient to define the time-reversed processes $x_{t}^{\leftarrow} := x_{-t}$ , and $\overline{x}_{t}^{\leftarrow} := \overline{x}_{-t}$ , along with $v(x,t) := \nabla \log p_{-t}(x)$ and $\overline{v}(x,t) := s_{\theta}(x,-t)$ . Then there exists a Brownian motion $B_{t}^{\leftarrow}$ , such that for $t \in [-t_{\max}, -t_{\min}]$ ,

$$
\begin{array}{l} d x _ {t} ^ {\leftarrow} = - 2 t v (x _ {t} ^ {\leftarrow}, t) d t + \sqrt {- 2 t} d B _ {t} ^ {\leftarrow} \\ d \overline {{x}} _ {t} ^ {\leftarrow} = - 2 t \overline {{v}} (\overline {{x}} _ {k \delta} ^ {\leftarrow}, k \delta) d t + \sqrt {- 2 t} d B _ {t} ^ {\leftarrow} \\ \Rightarrow \quad d (x _ {t} ^ {\leftarrow} - \overline {{x}} _ {t} ^ {\leftarrow}) = - 2 t \left(v (x _ {t} ^ {\leftarrow}, t) - \overline {{v}} (\overline {{x}} _ {k \delta} ^ {\leftarrow}, k \delta)\right) d t, \\ \end{array}
$$

where $k$ is the unique negative integer such that $t \in [k\delta, (k + 1)\delta)$ . Thus

$$
\begin{array}{l} \frac {d}{d t} \mathbb {E} \left[ \left\| x _ {t} ^ {\leftarrow} - \overline {{x}} _ {t} ^ {\leftarrow} \right\| \right] \\ \leq 2 \left(\mathbb {E} \left[ \| t v (x _ {t} ^ {\leftarrow}, t) - t \overline {{v}} (x _ {t} ^ {\leftarrow}, t) \| \right] + \mathbb {E} \left[ \| t \overline {{v}} (x _ {t} ^ {\leftarrow}, t) - t \overline {{v}} (\overline {{x}} _ {t} ^ {\leftarrow}, t) \| \right]\right) \\ + 2 \left(\mathbb {E} \left[ \| t \overline {{v}} (\overline {{x _ {t} ^ {\leftarrow}}}, t) - t \overline {{v}} (\overline {{x _ {t} ^ {\leftarrow}}}, k \delta) \| \right] + \mathbb {E} \left[ \| t \overline {{v}} (\overline {{x _ {t} ^ {\leftarrow}}}, k \delta) - t \overline {{v}} (\overline {{x _ {k \delta} ^ {\leftarrow}}}, k \delta) \| \right]\right) \\ \leq 2 \left(\epsilon_ {\text {approx}} + L _ {2} \mathbb {E} \left[ \| x _ {t} ^ {\leftarrow} - \overline {{x}} _ {t} ^ {\leftarrow} \| \right] + \delta L _ {0} + L _ {2} \mathbb {E} \left[ \| \overline {{x}} _ {t} ^ {\leftarrow} - \overline {{x}} _ {k \delta} ^ {\leftarrow} \| \right]\right) \\ \leq 2 \left(\epsilon_ {a p p r o x} + L _ {2} \mathbb {E} \left[ \| x _ {t} ^ {\leftarrow} - \overline {{x}} _ {t} ^ {\leftarrow} \| \right] + \delta L _ {0} + L _ {2} \left(\delta L _ {1} + \sqrt {2 \delta d t _ {\max}}\right)\right). \\ \end{array}
$$

By Gronwall's Lemma,

$$
\begin{array}{l} \mathbb {E} \left[ \left\| x _ {t _ {\min}} - \overline {{x}} _ {t _ {\min}} \right\| \right] \\ = \mathbb {E} \left[ \left\| x _ {- t _ {\min}} ^ {\leftarrow} - \overline {{x}} _ {- t _ {\min}} ^ {\leftarrow} \right\| \right] \\ \leq e ^ {2 L _ {2} \left(t _ {\max} - t _ {\min}\right)} \left(\mathbb {E} \left[ \left\| x _ {- t _ {\max}} ^ {\leftarrow} - \overline {{x}} _ {- t _ {\max}} ^ {\leftarrow} \right\| \right] + \left(\epsilon_ {\text {approx}} + \delta L _ {0} + L _ {2} \left(\delta L _ {1} + \sqrt {2 \delta d t _ {\max}}\right)\right) \left(t _ {\max} - t _ {\min}\right)\right) \\ = e ^ {2 L _ {2} \left(t _ {\max} - t _ {\min}\right)} \left(\mathbb {E} \left[ \| x _ {t _ {\max}} - \bar {x} _ {t _ {\max}} \| \right] + \left(\epsilon_ {\text {approx}} + \delta L _ {0} + L _ {2} \left(\delta L _ {1} + \sqrt {2 \delta d t _ {\max}}\right)\right) \left(t _ {\max} - t _ {\min}\right)\right) \\ \end{array}
$$

![](images/0736420e1b706bb9dd56d042d91710d0c7a9d318b8b2d81e275a22b6dff968e0.jpg)

# A.4 Mixing Bounds

Lemma 5. Consider the same setup as Theorem 3. Assume that $\delta \leq t_{min}$ . Let

$$
x _ {t _ {\min}} = S D E _ {\theta} \left(x _ {t _ {\max}}, t _ {\max} \rightarrow t _ {\min}\right)
$$

$$
y _ {t _ {\text { min }}} = S D E _ {\theta} \left(y _ {t _ {\text { max }}}, t _ {\text { max }} \rightarrow t _ {\text { min }}\right).
$$

Then there exists a coupling between $x_{s}$ and $y_{s}$ such that

$$
T V \left(x _ {t _ {\min}}, y _ {t _ {\min}}\right) \leq \left(1 - 2 Q \left(\frac {B}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right) \cdot e ^ {- B L _ {1} / t _ {\min} - L _ {1} ^ {2} t _ {\max} ^ {2} / t _ {\min} ^ {2}}\right) T V \left(x _ {t _ {\max}}, y _ {t _ {\max}}\right)
$$

Proof. We will construct a coupling between $x_{t}$ and $y_{t}$ . First, let $(x_{t_{\max}}, y_{t_{\max}})$ be sampled from the optimal TV coupling, i.e., $Pr(x_{t_{\max}} \neq y_{t_{\max}}) = TV(x_{t_{\max}}, y_{t_{\max}})$ . Recall that by definition of $\mathrm{SDE}_{\theta}$ , for $t \in ((k - 1)\delta, k\delta]$ ,

$$
d x _ {t} = - 2 t s _ {\theta} (x _ {k \delta}, k \delta) d t + \sqrt {2 t} d B _ {t}.
$$

Let us define a time-rescaled version of $x_{t}$ : $\overline{x}_{t} := x_{t^{2}}$ . We verify that

$$
d \overline {{x}} _ {t} = - s _ {\theta} (\overline {{x}} _ {(k \delta) ^ {2}}, k \delta) d t + d B _ {t},
$$

where k is the unique integer satisfying $t \in [((k-1)\delta)^{2}, k^{2}\delta^{2})$ . Next, we define the time-reversed process $\overline{x}_{t}^{\leftarrow} := \overline{x}_{-t}$ , and let $v(x,t) := s_{\theta}(x,-t)$ . We verify that there exists a Brownian motion $B_{t}^{x}$ such that, for $t \in [-t_{\max}^{2}, -t_{\min}^{2}]$ ,

$$
d \overline {{x}} _ {t} ^ {\leftarrow} = v _ {t} ^ {x} d t + d B _ {t} ^ {x},
$$

where $v_{t}^{x} = s_{\theta}(\overline{x}_{-(k\delta)^{2}}^{\leftarrow}, - k\delta)$ , where $k$ is the unique positive integer satisfying $-t \in (((k - 1)\delta)^{2}, (k\delta)^{2}]$ . Let $d\overline{y}_{t}^{\leftarrow} = v_{t}^{y}dt + dB_{t}^{y}$ , be defined analogously. For any positive integer $k$ and for any $t \in [-(k\delta)^{2}, - ((k - 1)\delta)^{2})$ , let us define

$$
z _ {t} = \overline {{x}} _ {- k ^ {2} \delta^ {2}} ^ {\leftarrow} - \overline {{y}} _ {- k ^ {2} \delta^ {2}} ^ {\leftarrow} + (2 k - 1) \delta^ {2} \left(v _ {- (k \delta) ^ {2}} ^ {x} - v _ {- (k \delta) ^ {2}} ^ {y}\right) + \left(B _ {t} ^ {x} - B _ {- (k \delta) ^ {2}} ^ {x}\right) - \left(B _ {t} ^ {y} - B _ {- (k \delta) ^ {2}} ^ {y}\right).
$$

Let $\gamma_t := \frac{z_t}{\|z_t\|}$ . We will now define a coupling between $dB_t^x$ and $dB_t^y$ as

$$
d B _ {t} ^ {y} = \left(I - 2 \mathbb {1} \{t \leq \tau \} \gamma_ {t} \gamma_ {t} ^ {T}\right) d B _ {t} ^ {x},
$$

where $\mathbb{1}\{\}$ denotes the indicator function, i.e. $\mathbb{1}\{t\leq\tau\}=1$ if $t\leq\tau$ , and $\tau$ is a stopping time given by the first hitting time of $z_{t}=0$ . Let $r_{t}:=\|z_{t}\|$ . Consider some $t\in\left(-i^{2}\delta^{2},-(i-1)^{2}\delta^{2}\right)$ , and Let $j:=\frac{t_{\max}}{\delta}$ (assume w.l.o.g that this is an integer), then

$$
\begin{array}{l} r _ {t} - r _ {- t _ {\max} ^ {2}} \leq \sum_ {k = i} ^ {j} (2 k - 1) \delta^ {2} \left\| (v _ {- (k \delta) ^ {2}} ^ {x} - v _ {- (k \delta) ^ {2}} ^ {y}) \right\| + \int_ {- t _ {\max} ^ {2}} ^ {t} \mathbb {1} \left\{t \leq \tau \right\} 2 d B _ {s} ^ {1} \\ \leq \sum_ {k = i} ^ {j} \left(k ^ {2} - (k - 1) ^ {2}\right) \delta^ {2} 2 L _ {1} / \left(t _ {\min}\right) + \int_ {- t _ {\max} ^ {2}} ^ {t} \mathbb {1} \left\{t \leq \tau \right\} 2 d B _ {t} ^ {1} \\ = \int_ {- t _ {\max} ^ {2}} ^ {- (i - 1) \delta^ {2}} \frac {2 L _ {1}}{t _ {\min}} d s + \int_ {- t _ {\max} ^ {2}} ^ {t} \mathbb {1} \left\{t \leq \tau \right\} 2 d B _ {s} ^ {1}, \\ \end{array}
$$

where $dB_{s}^{1} = \langle \gamma_{t}, dB_{s}^{x} - dB_{s}^{y} \rangle$ is a 1-dimensional Brownian motion. We also verify that

$$
\begin{array}{l} r _ {- t _ {\mathrm{max}} ^ {2}} = \left\| z _ {- t _ {\mathrm{max}} ^ {2}} \right\| \\ = \left\| \overline {{x}} _ {- t _ {\max} ^ {2}} ^ {\leftarrow} - \overline {{y}} _ {- t _ {\max} ^ {2}} ^ {\leftarrow} + (2 j - 1) \delta^ {2} \left(v _ {- t _ {\max} ^ {2}} ^ {x} - v _ {- t _ {\max} ^ {2}} ^ {y}\right) + \left(B _ {t} ^ {x} - B _ {- t _ {\max} ^ {2}} ^ {x}\right) - \left(B _ {t} ^ {y} - B _ {- t _ {\max} ^ {2}} ^ {y}\right) \right\| \\ \leq \left\| \overline {{x}} _ {- t _ {\max} ^ {2}} ^ {\leftarrow} + (2 j - 1) \delta^ {2} v _ {- t _ {\max} ^ {2}} ^ {x} + \left(B _ {- (j - 1) ^ {2} \delta^ {2}} ^ {x} - B _ {- t _ {\max} ^ {2}} ^ {x}\right) \right\| \\ + \left\| \overline {{y}} _ {- t _ {\max} ^ {2}} ^ {\leftarrow} + (2 j - 1) \delta^ {2} v _ {- t _ {\max} ^ {2}} ^ {y} + \left(B _ {- (j - 1) ^ {2} \delta^ {2}} ^ {x} - B _ {t} ^ {x} + B _ {t} ^ {y} - B _ {- t _ {\max} ^ {2}} ^ {y}\right) \right\| \leq B \\ \end{array}
$$

where the third relation is by adding and subtracting $B_{-(j-1)^{2}\delta^{2}}^{x}-B_{t}^{x}$ and using triangle inequality. The fourth relation is by noticing that $\overline{x}_{-t_{\max}^{2}}^{\leftarrow}+(2j-1)\delta^{2}v_{-t_{\max}^{2}}^{x}+\left(B_{-(j-1)^{2}\delta^{2}}^{x}-B_{-t_{\max}^{2}}^{x}\right)=\overline{x}_{-(j-1)^{2}\delta^{2}}^{\leftarrow}$ and that $\overline{y}_{-t_{\max}^{2}}^{\leftarrow}(2j-1)\delta^{2}v_{-t_{\max}^{2}}^{y}+\left(B_{-(j-1)^{2}\delta^{2}}^{x}-B_{t}^{x}+B_{t}^{y}-B_{-t_{\max}^{2}}^{y}\right)\stackrel{d}{=} \overline{y}_{-(j-1)^{2}\delta^{2}}^{\leftarrow}$ , and then using our assumption in the theorem statement that all processes are supported on a ball of radius B/2.

We now define a process $s_t$ defined by $ds_t = 2L_1 / t_{\min}dt + 2dB_t^1$ , initialized at $s_{-t_{\max}^2} = B \geq r_{-t_{\max}^2}$ . We can verify that, up to time $\tau$ , $r_t \leq s_t$ with probability 1. Let $\tau'$ denote the first-hitting time of $s_t$ to 0, then $\tau \leq \tau'$ with probability 1. Thus

$$
P r (\tau \leq - t _ {\min} ^ {2}) \geq P r (\tau^ {\prime} \leq - t _ {\min} ^ {2}) \geq 2 Q \left(\frac {B}{2 \sqrt {t _ {\max} ^ {2} - t _ {\min} ^ {2}}}\right) \cdot e ^ {- B L _ {1} / t _ {\min} - L _ {1} ^ {2} t _ {\max} ^ {2} / t _ {\min} ^ {2}}
$$

where we apply Lemma 6. The proof follows by noticing that, if $\tau \leq -t_{\min}^2$ , then $x_{t_{\min}} = y_{t_{\min}}$ . This is because if $\tau \in [-k^2\delta^2, -(k-1)^2\delta^2]$ , then $\overline{x}_{-(k-1)^2\delta^2}^{\leftarrow} = \overline{y}_{-(k-1)^2\delta^2}^{\leftarrow}$ , and thus $\overline{x}_t^{\leftarrow} = \overline{y}_t^{\leftarrow}$ for all $t \geq -(k-1)^2\delta^2$ , in particular, at $t = -t_{\min}^2$ .

![](images/2232bcd27863c79b0e9761147cd6805848969d600360cff0fd32cfa18fe1aa6a.jpg)

Lemma 6. Consider the stochastic process

$$
d r _ {t} = d B _ {t} ^ {1} + c d t.
$$

Assume that $r_{0} \leq B/2$ . Let $\tau$ denote the hitting time for $r_{t} = 0$ . Then for any $T \in R^{+}$ ,

$$
P r (\tau \leq T) \geq 2 Q \left(\frac {B}{2 \sqrt {T}}\right) \cdot e ^ {- a c - \frac {c ^ {2} T}{2}},
$$

where $Q$ is the tail probability of a standard Gaussian defined in Definition 1.

Proof. We will use he following facts in our proof:

1. For $x \sim \mathcal{N}(0, \sigma^2)$ , $Pr(x > r) = \frac{1}{2}\left(1 - erf\left(\frac{r}{\sqrt{2}\sigma}\right)\right) = \frac{1}{2}erfc\left(\frac{r}{\sqrt{2}\sigma}\right)$ .   
2. $\int_0^T\frac{a\exp\left(-\frac{a^2}{2t}\right)}{\sqrt{2\pi t^3}} dt = erfc\left(\frac{a}{\sqrt{2T}}\right) = 2Pr(\mathcal{N}(0,T) > a) = 2Q\left(\frac{a}{\sqrt{T}}\right)$ by definition of $Q$ .

Let $dr_{t} = dB_{t}^{1} + cdt$ , with $r_0 = a$ . The density of the hitting time $\tau$ is given by

$$
p (\tau = t) = f (a, c, t) = \frac {a \exp \left(- \frac {(a + c t) ^ {2}}{2 t}\right)}{\sqrt {2 \pi t ^ {3}}}. \tag {18}
$$

(see e.g. [3]). From item 2 above,

$$
\int_ {0} ^ {T} f (a, 0, t) d t = 2 Q \left(\frac {a}{\sqrt {T}}\right).
$$

In the case of a general $c \neq 0$ , we can bound $\frac{(a + ct)^2}{2t} = \frac{a^2}{2t} + ac + \frac{c^2t}{2}$ . Consequently,

$$
f (a, c, t) \geq f (a, 0, t) \cdot e ^ {- a c - \frac {c ^ {2} t}{2}}.
$$

Therefore,

$$
P r (\tau \leq T) = \int_ {0} ^ {T} f (a, c, t) d t \geq \int_ {0} ^ {T} f (a, 0, t) d t e ^ {- c} = 2 Q \left(\frac {B}{2 \sqrt {T}}\right) \cdot e ^ {- a c - \frac {c ^ {2} T}{2}}.
$$

![](images/f4ed92a67e19d80c2e32b32bbb19ed817b2ab875919ae3720b27351d4be2d484.jpg)

# A.5 TV Overlap

Definition 1. Let x be sampled from standard normal distribution N(0,1). We define the Gaussian tail probability $Q(a) := Pr(x \geq a)$ .

Lemma 7. We verify that for any two random vectors $\xi_x \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ and $\xi_y \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ , each belonging to $\mathbb{R}^d$ , the total variation distance between $x' = x + \xi_x$ and $y' = y + \xi_y$ is given by

$$
T V (x ^ {\prime}, y ^ {\prime}) = 1 - 2 Q (r) \leq 1 - \frac {2 r}{r ^ {2} + 1} \frac {1}{\sqrt {2 \pi}} e ^ {- r ^ {2} / 2},
$$

where $r = \frac{\|x - y\|}{2\sigma}$ , and $Q(r) = Pr(\xi \geq r)$ , when $\xi \sim \mathcal{N}(0,1)$ .

Proof. Let $\gamma := \frac{x - y}{\|x - y\|}$ . We decompose $x', y'$ into the subspace/orthogonal space defined by $\gamma$ :

$$
x ^ {\prime} = x ^ {\perp} + \xi_ {x} ^ {\perp} + x ^ {\parallel} + \xi_ {x} ^ {\parallel}
$$

$$
y ^ {\prime} = y ^ {\perp} + \xi_ {y} ^ {\perp} + y ^ {\parallel} + \xi_ {y} ^ {\parallel}
$$

where we define

$$
x ^ {\parallel} := \gamma \gamma^ {T} x \quad x ^ {\perp} := x - x ^ {\parallel}
$$

$$
y ^ {\parallel} := \gamma \gamma^ {T} y \qquad y ^ {\perp} := y - y ^ {\parallel}
$$

$$
\xi_ {x} ^ {\parallel} := \gamma \gamma^ {T} \xi_ {x} \quad \xi_ {x} ^ {\perp} := \xi_ {x} - \xi_ {x} ^ {\parallel}
$$

$$
\xi_ {y} ^ {\parallel} := \gamma \gamma^ {T} \xi_ {y} \quad \xi_ {y} ^ {\perp} := \xi_ {y} - \xi_ {y} ^ {\parallel}
$$

We verify the independence $\xi_x^\perp \perp \xi_x^\parallel$ and $\xi_y^\perp \perp \xi_y^\parallel$ as they are orthogonal decompositions of the standard Gaussian. We will define a coupling between $x'$ and $y'$ by setting $\xi_x^\perp = \xi_y^\perp$ . Under this coupling, we verify that

$$
\left(x ^ {\perp} + \xi_ {x} ^ {\perp}\right) - \left(y ^ {\perp} + \xi_ {y} ^ {\perp}\right) = x - y - \gamma \gamma^ {T} (x - y) = 0
$$

Therefore, $x' = y'$ if and only if $x^{\parallel} + \xi_x^{\parallel} = y^{\parallel} + \xi_y^{\parallel}$ . Next, we draw $(a, b)$ from the optimal coupling between $\mathcal{N}(0,1)$ and $\mathcal{N}\left(\frac{\|x - y\|}{\sigma}, 1\right)$ . We verify that $x^{\parallel} + \xi_x^{\parallel}$ and $y^{\parallel} + \xi_y^{\parallel}$ both lie in the span of $\gamma$ . Thus it suffices to compare $\left\langle \gamma, x^{\parallel} + \xi_x^{\parallel} \right\rangle$ and $\left\langle \gamma, y^{\parallel} + \xi_y^{\parallel} \right\rangle$ . We verify that $\left\langle \gamma, x^{\parallel} + \xi_x^{\parallel} \right\rangle =$

$\left\langle \gamma, y^{\parallel} \right\rangle + \left\langle \gamma, x^{\parallel} - y^{\parallel} \right\rangle + \left\langle \gamma, \xi_x^{\parallel} \right\rangle \sim \mathcal{N}(\left\langle \gamma, y^{\parallel} \right\rangle + \|x - y\|, \sigma^2) \stackrel{d}{=} \left\langle \gamma, y^{\parallel} \right\rangle + \sigma b$ . We similarly verify that $\left\langle \gamma, y^{\parallel} + \xi_y^{\parallel} \right\rangle = \left\langle \gamma, y^{\parallel} \right\rangle + \left\langle \gamma, \xi_y^{\parallel} \right\rangle \sim \mathcal{N}(\left\langle \gamma, y^{\parallel} \right\rangle, \sigma^2) \stackrel{d}{=} \left\langle \gamma, y^{\parallel} \right\rangle + \sigma a$ .

Thus $TV(x', y') = TV(\sigma a, \sigma b) = 1 - 2Q\left(\frac{\|x - y\|}{2\sigma}\right)$ . The last inequality follows from

$$
P r (\mathcal {N} (0, 1) \geq r) \geq \frac {r}{r ^ {2} + 1} \frac {1}{\sqrt {2 \pi}} e ^ {- r ^ {2} / 2}
$$

![](images/8ab624d92eb6b105c1692511a01a756ba0ab492708921215b12322fbf0c93700.jpg)

# B More on Restart Algorithm

# B.1 EDM Discretization Scheme

[13] proposes a discretization scheme for ODE given the starting $t_{max}$ and end time $t_{min}$ . Denote the number of steps as N, then the EDM discretization scheme is:

$$
t _ {i <   N} = \left(t _ {\max} ^ {\frac {1}{\rho}} + \frac {i}{N - 1} (t _ {\min} ^ {\frac {1}{\rho}} - t _ {\max} ^ {\frac {1}{\rho}})\right) ^ {\rho}
$$

with $t_{0}=t_{max}$ and $t_{N-1}=t_{min}$ . $\rho$ is a hyperparameter that determines the extent to which steps near $t_{min}$ are shortened. We adopt the value $\rho=7$ suggested by [13] in all of our experiments. We apply the EDM scheme to create a time discretization in each Restart interval $[t_{\max}, t_{\min}]$ in the Restart backward process, as well as the main backward process between $[0, T]$ (by additionally setting $t_{min}=0.002$ and $t_{N}=0$ as in [13]). It is important to note that $t_{min}$ should be included within the list of time steps in the main backward process to seamlessly incorporate the Restart interval into the main backward process. We summarize the scheme as a function in Algorithm 1.

Algorithm 1 EDM\_Scheme( $t_{\min}, t_{\max}, N, \rho = 7$ )

1: return $\left\{(t_{\max}^{\frac{1}{\rho}} + \frac{i}{N - 1}(t_{\min}^{\frac{1}{\rho}} - t_{\max}^{\frac{1}{\rho}}))^{\rho}\right\}_{i = 0}^{N - 1}$

# B.2 Restart Algorithm

We present the pseudocode for the Restart algorithm in Algorithm 2. In this pseudocode, we describe a more general case that applies l-level Restarting strategy. For each Restart segment, the include the number of steps in the Restart backward process $N_{Restart}$ , the Restart interval $[t_{\min}, t_{\max}]$ and the number of Restart iteration K. We further denote the number of steps in the main backward process as $N_{main}$ . We use the EDM discretization scheme (Algorithm 1) to construct time steps for the main backward process ( $t_{0} = T, t_{N_{main}} = 0$ ) as well as the Restart backward process, when given the starting/end time and the number of steps.

Although Heun's $2^{\mathrm{nd}}$ order method [2] (Algorithm 3) is the default ODE solver in the pseudocode, it can be substituted with other ODE solvers, such as Euler's method or the DPM solver [16].

The provided pseudocode in Algorithm 2 is tailored specifically for diffusion models [13]. To adapt Restart for other generative models like PFGM++ [28], we only need to modify the Gaussian perturbation kernel in the Restart forward process (line 10 in Algorithm 2) to the one used in PFGM++.

# C Experimental Details

In this section, we discuss the configurations for different samplers in details. All the experiments are conducted on eight NVIDIA A100 GPUs.

Algorithm 2 Restart sampling   
1: Input: Score network $s_{\theta}$ , time steps in main backward process $t_{i\in\{0,N_{main}\}}$ , Restart parameters $\{(N_{\text{Restart},j}, K_j, t_{\text{min},j}, t_{\text{max},j})\}_{j=1}^l$ 2: Round $t_{\text{min},j\in\{1,l\}}$ to its nearest neighbor in $t_{i\in\{0,N_{main}\}}$ 3: Sample $x_0 \sim \mathcal{N}(0, T^2 I)$ 4: for $i = 0 \ldots N_{main} - 1$ do ▷ Main backward process
5: $x_{t_{i+1}} = \text{OneStep\_Heun}(s_{\theta}, t_i, t_{i+1})$ ▷ Running single step ODE
6: if $\exists j \in \{1, \ldots, l\}, t_{i+1} = t_{\text{min},j}$ then
7: $t_{\text{min}} = t_{\text{min},j}, t_{\text{max}} = t_{\text{max},j}, K = K_j, N_{\text{Restart}} = N_{\text{Restart},j}$ 8: $x_{t_{\text{min}}}^0 = x_{t_{i+1}}$ 9: for $k = 0 \ldots K - 1$ do ▷ Restart for K iterations
10: $\varepsilon_{t_{\text{min}} \to t_{\text{max}}} \sim \mathcal{N}(0, (t_{\text{max}}^2 - t_{\text{min}}^2) I)$ 11: $x_{t_{\text{max}}}^{k+1} = x_{t_{\text{min}}}^k + \varepsilon_{t_{\text{min}} \to t_{\text{max}}}$ ▷ Restart forward process
12: $\{\bar{t}_m\}_{m=0}^{N_{\text{Restart}}-1} = \text{EDM\_Scheme}(t_{\text{min}}, t_{\text{max}}, N_{\text{Restart}})$ 13: for $m = 0 \ldots N_{\text{Restart}} - 1$ do ▷ Restart backward process
14: $x_{\bar{t}_{m+1}}^{k+1} = \text{OneStep\_Heun}(s_{\theta}, \bar{t}_m, \bar{t}_{m+1})$ 15: end for
16: end for
17: end if
18: end for
19: return $x_{t_{N_{main}}}$

Algorithm 3 OneStep\_Heun( $s_{\theta}, x_{t_i}, t_i, t_{i+1}$ )   
1: $d_{i} = t_{i}s_{\theta}(x_{t_{i}}, t_{i})$ 2: $x_{t_{i+1}} = x_{t_{i}} - (t_{i+1} - t_{i})d_{i}$ 3: if $t_{i+1} \neq 0$ then
4: $d_{i}' = t_{i+1}s_{\theta}(x_{t_{i+1}}, t_{i+1})$ 5: $x_{t_{i+1}} = x_{t_{i}} - (t_{i+1} - t_{i})(\frac{1}{2}d_{i} + \frac{1}{2}d_{i}')$ 6: end if
7: return $x_{t_{i+1}}$

# C.1 Configurations for Baselines

We select Vanilla SDE [23], Improved SDE [13], Gonna Go Fast [12] as SDE baselines and the Heun's $2^{\text{nd}}$ order method [2] (Alg 3) as ODE baseline on standard benchmarks CIFAR-10 and ImageNet $64 \times 64$ . We choose DDIM [22], Heun's $2^{\text{nd}}$ order method, and DDPM [9] for comparison on Stable Diffusion model.

Vanilla SDE denotes the reverse-time SDE sampler in $[23]$ . For Improved SDE, we use the recommended dataset-specific hyperparameters (e.g., $S_{max}$ , $S_{min}$ , $S_{churn}$ ) in Table 5 of the EDM paper $[13]$ . They obtained these hyperparameters by grid search. Gonna Go Fast $[12]$ applied an adaptive step size technique based on Vanilla SDE and we directly report the FID scores listed in $[12]$ for Gonna Go Fast on CIFAR-10 (VP). For fair comparison, we use the EDM discretization scheme $[13]$ for Vanilla SDE, Improved SDE, Heun as well as Restart.

We borrow the hyperparameters such as discretization scheme or initial noise scale on Stable Diffusion models in the diffuser ${}^{3}$ code repository. We directly use the DDIM and DDPM samplers implemented in the repo. We apply the same set of hyperparameters to Heun and Restart.

# C.2 Configurations for Restart

We report the configurations for Restart for different models and NFE on standard benchmarks CIFAR-10 and ImageNet $64 \times 64$ . The hyperparameters of Restart include the number of steps in the main backward process $N_{main}$ , the number of steps in the Restart backward process $N_{Restart}$ , the Restart interval $[t_{\min}, t_{\max}]$ and the number of Restart iteration K. In Table 3 (CIFAR-10, VP)

we provide the quintuplet $(N_{\mathrm{main}}, N_{\mathrm{Restart}}, t_{\mathrm{min}}, t_{\mathrm{max}}, K)$ for each experiment. Since we apply the multi-level Restart strategy for ImageNet $64 \times 64$ , we provide $N_{main}$ as well as a list of quadruple $\{(N_{\mathrm{Restart},i}, K_i, t_{\mathrm{min},i}, t_{\mathrm{max},i})\}_{i=1}^l$ (l is the number of Restart interval depending on experiments) in Table 5. In order to integrate the Restart time interval to the main backward process, we round $t_{min,i}$ to its nearest neighbor in the time steps of main backward process, as shown in line 2 of Algorithm 2. We apply Heun method for both main/backward process. The formula for NFE calculation is $NFE = \underbrace{2 \cdot N_{main} - 1}_{main backward process} + \sum_{i=1}^{l} \underbrace{K_i}_{number of repetitions} \cdot \underbrace{(2 \cdot (N_{\mathrm{Restart},i} - 1))}_{per iteration in i^{th} Restart interval}$ in this case. Inspired by

[13], we inflate the additive noise in the Restart forward process by multiplying $S_{noise} = 1.003$ on ImageNet $64 \times 64$ , to counteract the over-denoising tendency of neural networks. We also observe that setting $\gamma = 0.05$ in Algorithm 2 of EDM [13] would slightly boost the Restart performance on ImageNet $64 \times 64$ when $t \in [0.01, 1]$ .

We further include the configurations for Restart on Stable Diffusion models in Table 10, with a varying guidance weight $w$ . Similar to ImageNet $64 \times 64$ , we use multi-level Restart with a fixed number of steps $N_{\mathrm{main}} = 30$ in the main backward process. We utilize the Euler method for the main backward process and the Heun method for the Restart backward process, as our empirical observations indicate that the Heun method doesn't yield significant improvements over the Euler method, yet necessitates double the steps. The number of steps equals to $N_{\mathrm{main}} + \sum_{i=1}^{l} K_i \cdot (2 \cdot (N_{\mathrm{Restart},i} - 1))$ in this case. We set the total number of steps to 66, including main backward process and Restart backward process.

Given the prohibitively large search space for each Restart quadruple, a comprehensive enumeration of all possibilities is impractical due to computational limitations. Instead, we adjust the configuration manually, guided by the heuristic that weaker/smaller models or more challenging tasks necessitate a stronger Restart strength (e.g., larger K, wider Restart interval, etc). On average, we select the best configuration from 5 sets for each experiment; these few trials have empirically outperformed previous SDE/ODE samplers. We believe that developing a systematic approach for determining Restart configurations could be of significant value in the future.

# C.3 Pre-trained Models

For CIFAR-10 dataset, we use the pre-trained VP and EDM models from the EDM repository $^{4}$ , and PFGM++ (D = 2048) model from the PFGM++ repository $^{5}$ . For ImageNet 64 × 64, we borrow the pre-trained EDM model from EDM repository as well.

# C.4 Classifier-free Guidance

We follow the convention in [20], where each step in classifier-free guidance is as follows:

$$
\tilde {s} _ {\theta} (x, c, t) = w s _ {\theta} (x, c, t) + (1 - w) s _ {\theta} (x, t)
$$

where c is the conditions, and $s_{\theta}(x,c,t)/s_{\theta}(x,t)$ is the conditional/unconditional models, sharing parameters. Increasing w would strengthen the effect of guidance, usually leading to a better text-image alignment [20].

# C.5 More on the Synthetic Experiment

# C.5.1 Discrete Dataset

We generate the underlying discrete dataset S with $|S| = 2000$ as follows. Firstly, we sample 2000 points, denoted as $S_{1}$ , from a mixture of two Gaussians in $R^{4}$ . Next, we project these points onto $R^{20}$ . To ensure a variance of 1 on each dimension, we scale the coordinates accordingly. This setup aims to simulate data points that primarily reside on a lower-dimensional manifold with multiple modes.

The specific details are as follows: $S_{1} \sim 0.3N(a, s^{2}I) + 0.7(-a, s^{2}I)$ , where $a = (3, 3, 3, 3) \subset \mathbb{R}^{4}$ and $s = 1$ . Then, we randomly select a projection matrix $P \in \mathbb{R}^{20 \times 4}$ , where each entry is drawn from $N(0, 1)$ , and compute $S_{2} = PS_{1}$ . Finally, we scale each coordinate by a constant factor to ensure a variance of 1.

![](images/4bc7395703ffbdcbe95df50daedaae92dea7a2532883c6e66af007703d7c1a09.jpg)

<details>
<summary>line</summary>

| Additional Sampling error | ODE    | Vanilla SDE | Improved SDE | Restart |
| ------------------------- | ------ | ----------- | ------------ | ------- |
| 0.65                      | 0.89   | 0.89        | 0.89         | 0.89    |
| 0.70                      | 0.81   | 0.85        | 0.81         | 0.81    |
| 0.75                      | 0.81   | 0.85        | 0.81         | 0.81    |
| 0.80                      | 0.74   | 0.85        | 0.74         | 0.74    |
| 0.85                      | 0.73   | 0.85        | 0.73         | 0.73    |
| 0.90                      | 0.72   | 0.85        | 0.72         | 0.72    |
</details>

(a)

![](images/666bf6d9e825c13deb8b19863c9bc7387d0a55ccf63f113c852106cee9cfb796.jpg)  
(b)

![](images/debac691876fe98d020fcf3673b470d381dffdc04bff7d65e15de473838fb322.jpg)

<details>
<summary>line</summary>

| NFE  | ODE    | Vanilla SDE | Improved SDE | Restart |
| ---- | ------ | ----------- | ------------ | ------- |
| 20   | 0.875  | 0.875       | 0.875        | 0.875   |
| 40   | 0.875  | 0.875       | 0.850        | 0.775   |
| 60   | 0.875  | 0.875       | 0.825        | 0.750   |
| 80   | 0.875  | 0.875       | 0.800        | 0.750   |
| 100  | 0.875  | 0.825       | 0.775        | 0.750   |
| 120  | 0.875  | 0.825       | 0.750        | 0.750   |
| 140  | 0.875  | 0.825       | 0.750        | 0.750   |
| 160  | 0.875  | 0.825       | 0.750        | 0.750   |
| 180  | 0.875  | 0.825       | 0.750        | 0.750   |
| 200  | 0.875  | 0.825       | 0.750        | 0.750   |
| 220  | 0.875  | 0.825       | 0.750        | 0.750   |
| 240  | 0.875  | 0.825       | 0.750        | 0.750   |
| 260  | 0.875  | 0.825       | 0.750        | 0.750   |
| 280  | 0.875  | 0.825       | 0.750        | 0.750   |
| 300  | 0.875  | 0.825       | 0.750        | 0.750   |
| 320  | 0.875  | 0.825       | 0.750        | 0.750   |
</details>

(c)   
Figure 8: Comparison of additional sampling error versus (a) contracted error (plotting the Pareto frontier) and (b) total error (using a scatter plot). (c) Pareto frontier of NFE versus total error.

# C.5.2 Model Architecture

We employ a common MLP architecture with a latent size of 64 to learn the score function. The training method is adapted from $[13]$ , which includes the preconditioning technique and denoising score-matching objective $[25]$ .

# C.5.3 Varying Hyperparameters

To achieve the best trade-off between contracted error and additional sampling error, and optimize the NFE versus FID (Fréchet Inception Distance) performance, we explore various hyperparameters. [13] shows that the Vanilla SDE can be endowed with additional flexibility by varying the coefficient $\beta(t)$ (Eq.(6) in [13]). Hence, regarding SDE, we consider NFE values from $\{20, 40, 80, 160, 320\}$ , and multiply the original $\beta(t) = \dot{\sigma}(t)/\sigma(t)$ [13] with values from $\{0, 0.25, 0.5, 1, 1.5, 2, 4, 8\}$ . It is important to note that larger NFE values do not lead to further performance improvements. For restarts, we tried the following two settings: first we set the number of steps in Restart backward process to 40 and vary the number of Restart iterations $K$ in the range $\{0, 5, 10, 15, 20, 25, 30, 35\}$ . We also conduct a grid search with the number of Restart iterations $K$ ranging from 5 to 25 and the number of steps in Restart backward process varying from 2 to 7. For ODE, we experiment with the number of steps set to $\{20, 40, 80, 160, 320, 640\}$ .

Additionally, we conduct an experiment for Improved SDE in EDM. We try different values of $S_{churn}$ in the range of $\{0, 1, 2, 4, 8, 16, 32, 48, 64\}$ . We also perform a grid search where the number of steps ranged from 20 to 320 and $S_{churn}$ takes values of $[0.2 \times steps, 0.5 \times steps, 20, 60]$ . The plot combines the results from SDE and is displayed in Figure 8.

To mitigate the impact of randomness, we collect the data by averaging the results from five runs with the same hyperparameters. To compute the Wasserstein distance between two discrete distributions, we use minimum weight matching.

# C.5.4 Plotting the Pareto frontier

We generate the Pareto frontier plots as follows. For the additional sampling error versus contracted error plot, we first sort all the data points based on their additional sampling error and then connect the data points that represent prefix minimums of the contracted error. Similarly, for the NFE versus FID plot, we sort the data points based on their NFE values and connect the points where the FID is a prefix minimum.

# D Extra Experimental Results

# D.1 Numerical Results

In this section, we provide the corresponding numerical results of Fig. 3(a) and Fig. 3(b), in Table 2, 3 (CIFAR-10 VP, EDM, PFGM++) and Table 4, 5 (ImageNet 64 × 64 EDM), respectively. We also include the performance of Vanilla SDE in those tables. For the evaluation, we compute the Fréchet distance between 50000 generated samples and the pre-computed statistics of CIFAR-10 and ImageNet 64 × 64. We follow the evaluation protocol in EDM [13] that calculates each FID scores three times with different seeds and report the minimum.

We also provide the numerical results on the Stable Diffusion model $[19]$ , with a classifier guidance weight w = 2, 3, 5, 8 in Table 6, 7, 8, 9. As in $[17]$ , we report the zero-shot FID score on 5K random prompts sampled from the COCO validation set. We evaluate CLIP score $[6]$ with the open-sourced ViT-g/14 $[11]$ , Aesthetic score by the more recent LAION-Aesthetics Predictor V2 ${}^{6}$ . We average the CLIP and Aesthetic scores over 5K generated samples. The number of function evaluations is two times the sampling steps in Stable Diffusion model, since each sampling step involves the evaluation of the conditional and unconditional model.

Table 2: CIFAR-10 sample quality (FID score) and number of function evaluations (NFE) on VP [23] for baselines 

<table><tr><td></td><td>NFE</td><td>FID</td></tr><tr><td rowspan="6">ODE (Heun) [13]</td><td>1023</td><td>2.90</td></tr><tr><td>511</td><td>2.90</td></tr><tr><td>255</td><td>2.90</td></tr><tr><td>127</td><td>2.90</td></tr><tr><td>63</td><td>2.89</td></tr><tr><td>35</td><td>2.97</td></tr><tr><td rowspan="4">Vanilla SDE [23]</td><td>1024</td><td>2.79</td></tr><tr><td>512</td><td>4.01</td></tr><tr><td>256</td><td>4.79</td></tr><tr><td>128</td><td>12.57</td></tr><tr><td rowspan="6">Gonna Go Fast [12]</td><td>1000</td><td>2.55</td></tr><tr><td>329</td><td>2.70</td></tr><tr><td>274</td><td>2.74</td></tr><tr><td>179</td><td>2.59</td></tr><tr><td>147</td><td>2.95</td></tr><tr><td>49</td><td>72.29</td></tr><tr><td rowspan="6">Improved SDE [13]</td><td>1023</td><td>2.35</td></tr><tr><td>511</td><td>2.37</td></tr><tr><td>255</td><td>2.40</td></tr><tr><td>127</td><td>2.58</td></tr><tr><td>63</td><td>2.88</td></tr><tr><td>35</td><td>3.45</td></tr></table>

Table 3: CIFAR-10 sample quality (FID score), number of function evaluations (NFE) and Restart configurations on VP [23], VP with DPM-Solver-3 [16], EDM [13] and PFGM++ [28] 

<table><tr><td>Method</td><td>NFE</td><td>FID</td><td>Configuration $(N_{\text{main}}, N_{\text{Restart},i}, K_i, t_{\text{min},i}, t_{\text{max},i})$ </td></tr><tr><td colspan="4">VP</td></tr><tr><td></td><td>519</td><td>2.11</td><td>(20, 9, 30, 0.06, 0.20)</td></tr><tr><td></td><td>115</td><td>2.21</td><td>(18, 3, 20, 0.06, 0.30)</td></tr><tr><td></td><td>75</td><td>2.27</td><td>(18, 3, 10, 0.06, 0.30)</td></tr><tr><td></td><td>55</td><td>2.45</td><td>(18, 3, 5, 0.06, 0.30)</td></tr><tr><td></td><td>43</td><td>2.70</td><td>(18, 3, 2, 0.06, 0.30)</td></tr><tr><td colspan="4">VP w/ DPM-Solver-3</td></tr><tr><td></td><td>27</td><td>2.11</td><td>(8, 3, 1, 0.06, 0.3)</td></tr><tr><td></td><td>24</td><td>2.15</td><td>(7, 3, 1, 0.06, 1)</td></tr><tr><td></td><td>21</td><td>2.28</td><td>(6, 3, 1, 0.06, 1)</td></tr><tr><td></td><td>18</td><td>2.40</td><td>(5, 3, 1, 0.06, 1)</td></tr><tr><td colspan="4">EDM</td></tr><tr><td></td><td>43</td><td>1.90</td><td>(18, 3, 2, 0.14, 0.30)</td></tr><tr><td colspan="4">PFGM++</td></tr><tr><td></td><td>43</td><td>1.88</td><td>(18, 3, 2, 0.14, 0.30)</td></tr></table>

Table 4: ImageNet 64 × 64 sample quality (FID score) and number of function evaluations (NFE) on EDM [13] for baselines 

<table><tr><td></td><td>NFE</td><td>FID (50k)</td></tr><tr><td rowspan="6">ODE (Heun) [13]</td><td>1023</td><td>2.24</td></tr><tr><td>511</td><td>2.24</td></tr><tr><td>255</td><td>2.24</td></tr><tr><td>127</td><td>2.25</td></tr><tr><td>63</td><td>2.30</td></tr><tr><td>35</td><td>2.46</td></tr><tr><td rowspan="4">Vanilla SDE [23]</td><td>1024</td><td>1.89</td></tr><tr><td>512</td><td>3.38</td></tr><tr><td>256</td><td>11.91</td></tr><tr><td>128</td><td>59.71</td></tr><tr><td rowspan="6">Improved SDE [13]</td><td>1023</td><td>1.40</td></tr><tr><td>511</td><td>1.45</td></tr><tr><td>255</td><td>1.50</td></tr><tr><td>127</td><td>1.75</td></tr><tr><td>63</td><td>2.24</td></tr><tr><td>35</td><td>2.97</td></tr></table>

Table 5: ImageNet 64 × 64 sample quality (FID score), number of function evaluations (NFE) and Restart configurations on EDM [13] 

<table><tr><td>NFE</td><td>FID (50k)</td><td>Configuration $N_{\text{main}}, \{(N_{\text{Restart},i}, K_i, t_{\text{min},i}, t_{\text{max},i})\}_{i=1}^l$ </td></tr><tr><td>623</td><td>1.36</td><td>36, {(10, 3, 19.35, 40.79),(10, 3, 1.09, 1.92),(7, 6, 0.59, 1.09), (7, 6, 0.30, 0.59),(7, 25, 0.06, 0.30)}</td></tr><tr><td>535</td><td>1.39</td><td>36, {(6, 1, 19.35, 40.79),(6, 1, 1.09, 1.92),(7, 6, 0.59, 1.09), (7, 6, 0.30, 0.59),(7, 25, 0.06, 0.30)}</td></tr><tr><td>385</td><td>1.41</td><td>36, {(3, 1, 19.35, 40.79),(6, 1, 1.09, 1.92),(6, 5, 0.59, 1.09), (6, 5, 0.30, 0.59),(6, 20, 0.06, 0.30)}</td></tr><tr><td>203</td><td>1.46</td><td>36, {(4, 1, 19.35, 40.79),(4, 1, 1.09, 1.92),(4, 5, 0.59, 1.09), (4, 5, 0.30, 0.59),(6, 6, 0.06, 0.30)}</td></tr><tr><td>165</td><td>1.51</td><td>18, {(3, 1, 19.35, 40.79),(4, 1, 1.09, 1.92),(4, 5, 0.59, 1.09), (4, 5, 0.30, 0.59),(4, 10, 0.06, 0.30)}</td></tr><tr><td>99</td><td>1.71</td><td>18, {(3, 1, 19.35, 40.79),(4, 1, 1.09, 1.92),(4, 4, 0.59, 1.09), (4, 1, 0.30, 0.59),(4, 4, 0.06, 0.30)}</td></tr><tr><td>67</td><td>1.95</td><td>18, {(5, 1, 19.35, 40.79),(5, 1, 1.09, 1.92),(5, 1, 0.59, 1.09), (5, 1, 0.06, 0.30)}</td></tr><tr><td>39</td><td>2.38</td><td>14, {(3, 1, 19.35, 40.79),(3, 1, 1.09, 1.92), (3, 1, 0.06, 0.30)}</td></tr></table>

Table 6: Numerical results on Stable Diffusion v1.5 with a classifier-free guidance weight w = 2 

<table><tr><td></td><td>Steps</td><td>FID (5k) ↓</td><td>CLIP score ↑</td><td>Aesthetic score ↑</td></tr><tr><td rowspan="2">DDIM [22]</td><td>50</td><td>16.08</td><td>0.2905</td><td>5.13</td></tr><tr><td>100</td><td>15.35</td><td>0.2920</td><td>5.15</td></tr><tr><td rowspan="2">Heun</td><td>51</td><td>18.80</td><td>0.2865</td><td>5.14</td></tr><tr><td>101</td><td>18.21</td><td>0.2871</td><td>5.15</td></tr><tr><td rowspan="2">DDPM [9]</td><td>100</td><td>13.53</td><td>0.3012</td><td>5.20</td></tr><tr><td>200</td><td>13.22</td><td>0.2999</td><td>5.19</td></tr><tr><td>Restart</td><td>66</td><td>13.16</td><td>0.2987</td><td>5.19</td></tr></table>

Table 7: Numerical results on Stable Diffusion v1.5 with a classifier-free guidance weight w = 3 

<table><tr><td></td><td>Steps</td><td>FID (5k) ↓</td><td>CLIP score ↑</td><td>Aesthetic score ↑</td></tr><tr><td rowspan="2">DDIM [22]</td><td>50</td><td>14.28</td><td>0.3056</td><td>5.22</td></tr><tr><td>100</td><td>14.30</td><td>0.3056</td><td>5.22</td></tr><tr><td rowspan="2">Heun</td><td>51</td><td>15.63</td><td>0.3022</td><td>5.20</td></tr><tr><td>101</td><td>15.40</td><td>0.3026</td><td>5.21</td></tr><tr><td rowspan="2">DDPM [9]</td><td>100</td><td>15.72</td><td>0.3129</td><td>5.28</td></tr><tr><td>200</td><td>15.13</td><td>0.3131</td><td>5.28</td></tr><tr><td>Restart</td><td>66</td><td>14.48</td><td>0.3079</td><td>5.25</td></tr></table>

Table 8: Numerical results on Stable Diffusion v1.5 with a classifier-free guidance weight w = 5 

<table><tr><td></td><td>Steps</td><td>FID (5k) ↓</td><td>CLIP score ↑</td><td>Aesthetic score ↑</td></tr><tr><td rowspan="2">DDIM [22]</td><td>50</td><td>16.60</td><td>0.3154</td><td>5.31</td></tr><tr><td>100</td><td>16.80</td><td>0.3157</td><td>5.31</td></tr><tr><td rowspan="2">Heun</td><td>51</td><td>16.26</td><td>0.3135</td><td>5.28</td></tr><tr><td>101</td><td>16.38</td><td>0.3136</td><td>5.29</td></tr><tr><td rowspan="2">DDPM [9]</td><td>100</td><td>19.62</td><td>0.3197</td><td>5.36</td></tr><tr><td>200</td><td>18.88</td><td>0.3200</td><td>5.35</td></tr><tr><td>Restart</td><td>66</td><td>16.21</td><td>0.3179</td><td>5.33</td></tr></table>

Table 9: Numerical results on Stable Diffusion v1.5 with a classifier-free guidance weight $w = 8$ 

<table><tr><td></td><td>Steps</td><td>FID (5k) ↓</td><td>CLIP score ↑</td><td>Aesthetic score ↑</td></tr><tr><td rowspan="2">DDIM [22]</td><td>50</td><td>19.83</td><td>0.3206</td><td>5.37</td></tr><tr><td>100</td><td>19.82</td><td>0.3200</td><td>5.37</td></tr><tr><td rowspan="2">Heun</td><td>51</td><td>18.44</td><td>0.3186</td><td>5.35</td></tr><tr><td>101</td><td>18.72</td><td>0.3185</td><td>5.36</td></tr><tr><td rowspan="2">DDPM [9]</td><td>100</td><td>22.58</td><td>0.3223</td><td>5.39</td></tr><tr><td>200</td><td>21.67</td><td>0.3212</td><td>5.38</td></tr><tr><td>Restart</td><td>47</td><td>18.40</td><td>0.3228</td><td>5.41</td></tr></table>

Table 10: Restart (Steps=66) configurations on Stable Diffusion v1.5 

<table><tr><td>w</td><td>Configuration $N_{\text{main}}, \left\{ (N_{\text{Restart},i}, K_i, t_{\text{min},i}, t_{\text{max},i}) \right\}_{i=1}^l$ </td></tr><tr><td>2</td><td>30, {(5, 2, 1, 9), (5, 2, 5, 10)}</td></tr><tr><td>3</td><td>30, {(10, 2, 0.1, 3)}</td></tr><tr><td>5</td><td>30, {(10, 2 0.1, 2)}</td></tr><tr><td>8</td><td>30, {(10, 2, 0.1, 2)}</td></tr></table>

![](images/a8da77b0bef0ecc6a21a72cf7aa036477977a2a45324880fbe372c3d5672ccf0.jpg)  
(a) FID versus CLIP score

![](images/320b927d75dc96bdb39d34a9e7a53b6d5d2b8341203045e498f341806f8af4f9.jpg)

<details>
<summary>line</summary>

| Aesthetic score | Restart (Steps=66) | DDIM (Steps=50) | DDIM (Steps=100) | Heun (Steps=51) | Heun (Steps=101) | DDPM (Steps=100) | DDPM (Steps=200) |
| --------------- | ------------------ | --------------- | ---------------- | --------------- | ---------------- | ---------------- | ---------------- |
| 5.15            | -                  | 16.0            | 15.5             | 16.0            | 16.0             | 16.0             | 16.0             |
| 5.20            | -                  | 14.5            | 14.0             | 15.5            | 15.5             | 15.5             | 13.5             |
| 5.25            | -                  | 15.0            | 14.5             | 16.0            | 16.0             | 16.0             | 14.5             |
| 5.30            | -                  | 16.0            | 16.0             | 17.0            | 17.0             | 17.0             | 16.0             |
| 5.35            | -                  | 18.0            | 19.0             | 18.5            | 18.5             | 18.5             | 19.0             |
| 5.40            | -                  | -               | -                | -               | -                | -                | -                |
</details>

(b) FID versus Aesthetic score   
Figure 9: FID score versus (a) CLIP ViT-g/14 score and (b) Aesthetic score for text-to-image generation at $512 \times 512$ resolution, using Stable Diffusion v1.5 with varying classifier-free guidance weight w = 2, 3, 5, 8.

![](images/cbb98d8fcd5d6f255ae00796b4d893e500575e8b1f7baeac718fa31606e1e987.jpg)

<details>
<summary>line</summary>

| log(t_min) | EDM (Restart) | EDM (ODE) | VP (Restart) | VP (ODE) |
| ---------- | ------------- | --------- | ------------ | -------- |
| -4         | 2.3           | 1.9       | 3.2          | 3.0      |
| -2         | 1.9           | 1.9       | 2.6          | 3.0      |
| 0          | 2.0           | 1.9       | 2.9          | 3.0      |
| 2          | 1.9           | 1.9       | 3.0          | 3.0      |
</details>

(a)

![](images/b201d5ba5dd04398432c205b2b995d2ee1b2e33f7f3088ee1bd69abb1502d7ab.jpg)

<details>
<summary>line</summary>

| interval length | vp restart | vp ODE | edm restart | edm ODE |
| --------------- | ---------- | ------ | ----------- | ------- |
| 0.0             | 2.3        | 2.9    | 1.9         | 1.9     |
| 0.1             | 2.1        | 2.9    | 1.8         | 1.9     |
| 0.2             | 2.1        | 2.9    | 1.8         | 1.9     |
| 0.3             | 2.4        | 2.9    | 1.9         | 1.9     |
| 0.4             | 2.5        | 2.9    | 2.0         | 1.9     |
| 0.5             | 2.6        | 2.9    | 2.0         | 1.9     |
| 0.6             | 2.7        | 2.9    | 2.0         | 1.9     |
</details>

(b)   
Figure 10: (a): Adjusting $t_{\mathrm{min}}$ in Restart on VP/EDM; (b): Adjusting the Restart interval length when $t_{\mathrm{min}} = 0.06$ .

# D.2 Sensitivity Analysis of Hyper-parameters

We also investigate the impact of varying $t_{min}$ when $t_{max} = t_{min} + 0.3$ , and the length the restart interval when $t_{min} = 0.06$ . Fig. 10(a) reveals that FID scores achieve a minimum at a $t_{min}$ close to 0 on VP, indicating higher accumulated errors at the end of sampling and poor neural estimations at small t. Note that the Restart interval 0.3 is about twice the length of the one in Table 1 and Restart does not outperform the ODE baseline on EDM. This suggests that, as a rule of thumb, we should apply greater Restart strength (e.g., larger K, $t_{max} - t_{min}$ ) for weaker or smaller architectures and vice versa.

In theory, a longer interval enhances contraction but may add more additional sampling errors. Again, the balance between these factors results in a V-shaped trend in our plots (Fig. 10(b)). In practice, selecting $t_{\mathrm{max}}$ close to the dataset's radius usually ensures effective mixing when $t_{\mathrm{min}}$ is small.

# E Extended Generated Images

In this section, we provide extended generated images by Restart, DDIM, Heun and DDPM on text-to-image Stable Diffusion v1.5 model $[19]$ . We showcase the samples of four sets of text prompts in Fig. 11, Fig. 12, Fig. 13, Fig. 14, with a classifier-guidance weight w = 8.

# F Heun's method is DPM-Solver-2 (with $r_2 = 1$ )

The first order ODE in DPM-Solver [16] (DPM-Solver-1) is in the form of:

$$
\hat {x} _ {t _ {i - 1}} = \frac {\alpha_ {t _ {i}}}{\alpha_ {t _ {i - 1}}} \hat {x} _ {t _ {i - 1}} - (\hat {\sigma} _ {t _ {i - 1}} \frac {\alpha_ {t _ {i}}}{\alpha_ {t _ {i - 1}}} - \hat {\sigma} _ {t _ {i}}) \hat {\sigma} _ {t _ {i}} \nabla_ {x} \log p _ {\hat {\sigma} _ {t _ {i}}} (\hat {x} _ {t _ {i}}) \tag {19}
$$

The first order ODE in EDM is in the form of

$$
x _ {t _ {i - 1}} = x _ {t _ {i}} - \left(\sigma_ {t _ {i - 1}} - \sigma_ {t _ {i}}\right) \sigma_ {t _ {i}} \nabla_ {x} \log p _ {\sigma_ {t _ {i}}} \left(x _ {t _ {i}}\right) \tag {20}
$$

When $x_{t} = \frac{\hat{x}_{t}}{\alpha_{t}}$ , $\hat{\sigma}_{t} = \sigma_{t}\alpha_{t}$ , we can rewrite the DPM-Solver-1 (Eq. (19)) as:

$$
\begin{array}{l} x _ {t _ {i - 1}} = x _ {t _ {i}} - \left(\sigma_ {t _ {i - 1}} - \sigma_ {t _ {i}}\right) \hat {\sigma} _ {t _ {i}} \nabla_ {x} \log p _ {\hat {\sigma} _ {t _ {i}}} (\hat {x} _ {t _ {i}}) \\ = x _ {t _ {i}} - \left(\sigma_ {t _ {i - 1}} - \sigma_ {t _ {i}}\right) \hat {\sigma} _ {t _ {i}} \nabla_ {x} \log p _ {\sigma_ {t _ {i}}} \left(x _ {t _ {i}}\right) \frac {1}{\alpha_ {t _ {i}}} \quad (\text { change   -   of   -   variable }) \\ = x _ {t _ {i}} - (\sigma_ {t _ {i - 1}} - \sigma_ {t _ {i}}) \sigma_ {t _ {i}} \nabla_ {x} \log p _ {\sigma_ {t _ {i}}} (x _ {t _ {i}}) \\ \end{array}
$$

![](images/ee5285854c30d728c616253b2c72edbcb04de728e37c85515c46cd66b9b52fd7.jpg)

<details>
<summary>natural_image</summary>

Collage of astronauts on a desert mission scene, including oncoats, horses, and a rover (no text or symbols visible)
</details>

(a) Restart (Steps=66)

![](images/0821bdcbc8d4104bed793eda3ea5ed2ce97f4a75aeccb3c10f6700e334b1f674.jpg)

<details>
<summary>natural_image</summary>

Collage of human and horseback riders traversing a desert landscape, no text or symbols visible
</details>

(b) DDIM (Steps=100)

![](images/4c5c70f933d0582b5209914ff7d75b317a44f709e8283a810bc52024a3ba392f.jpg)

<details>
<summary>natural_image</summary>

Grid of 16 images showing people on horseback in a desert setting, including parlor and desert scenes (no text or symbols)
</details>

(c) Heun (Steps=101)

![](images/1ba0518624f0fd2d764e581afdc39039e23fdd2e2dde10f5173e684c1681122d.jpg)

<details>
<summary>natural_image</summary>

Collage of 16 images showing people on horseback in a desert environment, no text or symbols visible
</details>

(d) DDPM (Steps=100)   
Figure 11: Generated images with text prompt="A photo of an astronaut riding a horse on mars" and w = 8.

![](images/576067398a6f5946ffe3cdd97eafe3b8c7126891e4098bdf148727aafaa71849.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 identical photos of raccoas interacting with a table tennis and ball, no text or symbols visible.
</details>

(a) Restart (Steps=66)

![](images/ace93a50cc2a16cfe116658b4ec36ad455ecff79ff17a96bdb466c1eae9225bb.jpg)

<details>
<summary>natural_image</summary>

Grid of 24 identical photos of a raccoon interacting with a table tennis, showing dynamic poses and ball movements (no text or symbols)
</details>

(b) DDIM (Steps=100)

![](images/6376bb27e6292a42b63ffe78ffd0eaee7634635a0051ec5105124246591ca00d.jpg)

<details>
<summary>natural_image</summary>

Grid of black-and-white photos of raccoas interacting with tennis and table tennis, no text or symbols present
</details>

(c) Heun (Steps=101)   
(d) DDPM (Steps=100)   
Figure 12: Generated images with text prompt="A raccoon playing table tennis" and w = 8.

where the expression is exact the same as the ODE in EDM [13]. It indicates that the sampling trajectory in DPM-Solver-1 is equivalent to the one in EDM, up to a time-dependent scaling $(\alpha_{t})$ . As $\lim_{t\to 0}\alpha_t = 1$ , the two solvers will lead to the same final points when using the same time discretization. Note that the DPM-Solver-1 is also equivalent to DDIM (c.f. Section 4.1 in [16]), as also used in this paper.

With that, we can further verify that the Heun's method used in this paper corresponds to the DPM-Solver-2 when setting $r_1 = 1$ .

# G Broader Impact

The field of deep generative models incorporating differential equations is rapidly evolving and holds significant potential to shape our society. Nowadays, a multitude of photo-realistic images generated by text-to-image Stable Diffusion models populate the internet. Our work introduces Restart, a novel sampling algorithm that outperforms previous samplers for diffusion models and PFGM++. With applications extending across diverse areas, the Restart sampling algorithm is especially suitable for generation tasks demanding high quality and rapid speed. Yet, it is crucial to recognize that

![](images/201136cbbf53accb70a5f695514cb543564f32b293dc8238868ff13f98825fad.jpg)

<details>
<summary>natural_image</summary>

Collage of 12 photos of origami foxes in a snowy forest setting, no text or symbols present.
</details>

(a) Restart (Steps=66)

![](images/4a5edf66559be2b8d622a077b08347b0494ba549532d277af25b169936a01d2a.jpg)

<details>
<summary>natural_image</summary>

Grid of 16 photos of a fox in various forest and snowy environments, each composed of an origami figure (no text or symbols)
</details>

(b) DDIM (Steps=100)

![](images/e1e2bf34fae1b62f6977bf2b087477fd2c2984d0fbec4906b3b02ad146d58424.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 colorful paper origami fox sculptures in a snowy forest setting, no text or symbols visible.
</details>

(c) Heun (Steps=101)

![](images/9e4ad9531b0aa61918cac3f77ce15b7296a8dd7f6515bfc5657e0d55da60056e.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 origami fox sculptures in various poses and designs, set against a snowy forest background (no text or symbols)
</details>

(d) DDPM (Steps=100)   
Figure 13: Generated images with text prompt="Intricate origami of a fox in a snowy forest" and w = 8.

![](images/ef9cf8e3df9d9696aab0a17e64e4e99dcc9f5c624b168e65e19d36dbc2c8c35d.jpg)

<details>
<summary>natural_image</summary>

Grid of 16 modern glass duck sculptures displayed on stands, including various designs and colors (no text or labels visible)
</details>

(a) Restart (Steps=66)

![](images/d1a54effa234c5d53dabbac163d8591dc769d06fd77a1c3787d321dd07425c8a.jpg)

<details>
<summary>natural_image</summary>

Grid of 16 modern glass duck sculptures in various colors and designs, displayed on stands (no text or labels visible)
</details>

(b) DDIM (Steps=100)

![](images/187b11782924eb53afca189090c10c44ae2e507e0242ab375617d5c3a2125b3a.jpg)

<details>
<summary>natural_image</summary>

Grid of 25 glass duck models in various angles and sizes, displayed in a studio setting with no visible text or symbols.
</details>

(c) Heun (Steps=101)   
(d) DDPM (Steps=100)   
Figure 14: Generated images with text prompt="A transparent sculpture of a duck made out of glass" and w = 8.

the utilization of such algorithms can yield both positive and negative repercussions, contingent on their specific applications. On the one hand, Restart sampling can facilitate the generation of highly realistic images and audio samples, potentially advancing sectors such as entertainment, advertising, and education. On the other hand, it could also be misused in deepfake technology, potentially leading to social scams and misinformation. In light of these potential risks, further research is required to develop robustness guarantees for generative models, ensuring their use aligns with ethical guidelines and societal interests.