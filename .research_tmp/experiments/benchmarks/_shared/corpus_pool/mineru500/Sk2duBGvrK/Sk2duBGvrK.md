# Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structure

Xiang Li $^{1}$ , Yixiang Dai $^{1}$ , Qing Qu $^{1}$

$^{1}$ Department of EECS, University of Michigan,

forkobe@umich.edu, yixiang@umich.edu, qingqu@umich.edu

# Abstract

In this work, we study the generalizability of diffusion models by looking into the hidden properties of the learned score functions, which are essentially a series of deep denoisers trained on various noise levels. We observe that as diffusion models transition from memorization to generalization, their corresponding nonlinear diffusion denoisers exhibit increasing linearity. This discovery leads us to investigate the linear counterparts of the nonlinear diffusion models, which are a series of linear models trained to match the function mappings of the nonlinear diffusion denoisers. Interestingly, these linear denoisers are nearly optimal for multivariate Gaussian distributions defined by the empirical mean and covariance of the training dataset, and they effectively approximate the behavior of nonlinear diffusion models. This finding implies that diffusion models have the inductive bias towards capturing and utilizing the Gaussian structure (covariance information) of the training dataset for data generation. We empirically demonstrate that this inductive bias is a unique property of diffusion models in the generalization regime, which becomes increasingly evident when the model's capacity is relatively small compared to the training dataset size. In the case where the model is highly overparameterized, this inductive bias emerges during the initial training phases before the model fully memorizes its training data. Our study provides crucial insights into understanding the notable strong generalization phenomenon recently observed in real-world diffusion models.

# 1 Introduction

In recent years, diffusion models $[1-4]$ have become one of the leading generative models, powering the state-of-the-art image generation systems such as Stable Diffusion $[5]$ . To understand the empirical success of diffusion models, several works $[6-12]$ have focused on their sampling behavior, showing that the data distribution can be effectively estimated in the reverse sampling process, assuming that the score function is learned accurately. Meanwhile, other works $[13-18]$ investigate the learning of score functions, showing that effective approximation can be achieved with score matching loss under certain assumptions. However, these theoretical insights, grounded in simplified assumptions about data distribution and neural network architectures, do not fully capture the complex dynamics of diffusion models in practical scenarios. One significant discrepancy between theory and practice is that real-world diffusion models are trained only on a finite number of data points. As argued in $[19]$ , theoretically a perfectly learned score function over the empirical data distribution can only replicate the training data. In contrast, diffusion models trained on finite samples exhibit remarkable generalizability, producing high-quality images that significantly differ from the training examples. Therefore, a good understanding of the remarkable generative power of diffusion models is still lacking.

In this work, we aim to deepen the understanding of generalizability in diffusion models by analyzing the inherent properties of the learned score functions. Essentially, the score functions can be interpreted as a series of deep denoisers trained on various noise levels. These denoisers are then

chained together to progressively denoise a randomly sampled Gaussian noise into its corresponding clean image, thus, understanding the function mappings of these diffusion denoisers is critical to demystify the working mechanism of diffusion models. Motivated by the linearity observed in the diffusion denoisers of effectively generalized diffusion models, we propose to elucidate their function mappings with a linear distillation approach, where the resulting linear models serve as the linear approximations of their nonlinear counterparts.

Contributions of this work: Our key findings can be highlighted as follows:

- Inductive bias towards Gaussian structures (Section 3). Diffusion models in the generalization regime exhibit an inductive bias towards learning diffusion denoisers that are close (but not equal) to the optimal denoisers for a multivariate Gaussian distribution, defined by the empirical mean and covariance of the training data. This implies the diffusion models have the inductive bias towards capturing the Gaussian structure (covariance information) of the training data for image generation.   
- Model Capacity and Training Duration (Section 4) We show that this inductive bias is most pronounced when the model capacity is relatively small compared to the size of the training data. However, even if the model is highly overparameterized, such inductive bias still emerges during early training phases, before the model memorizes its training data. This implies that early stopping can prompt generalization in overparameterized diffusion models.   
- Connection between Strong Generalization and Gaussian Structure (Section 5). Lastly, we argue that the recently observed strong generalization [20] results from diffusion models learning certain common low-dimensional structural features shared across non-overlapping datasets. We show that such low-dimensional features can be partially explained through the Gaussian structure.

Relationship with Prior Arts. Recent research $[20–24]$ demonstrates that diffusion models operate in two distinct regimes: (i) a memorization regime, where models primarily reproduce training samples and (ii) a generalization regime, where models generate high-quality, novel images that extend beyond the training data. In the generalization regime, a particularly intriguing phenomenon is that diffusion models trained on non-overlapping datasets can generate nearly identical samples $[20]$ . While prior work $[20]$ attributes this “strong generalization” effect to the structural inductive bias inherent in diffusion models leading to the optimal denoising basis (geometry-adaptive harmonic basis), our research advances this understanding by demonstrating diffusion models’ inductive bias towards capturing the Gaussian structure of the training data. Our findings also corroborate with observations of earlier study $[25]$ that the learned score functions of well-trained diffusion models closely align with the optimal score functions of a multivariate Gaussian approximation of the training data.

# 2 Preliminary

Basics of Diffusion Models. Given a data distribution $p_{\mathrm{data}}(\boldsymbol{x})$ , where $x \in R^{d}$ , diffusion models [1–4] define a series of intermediate states $p(\boldsymbol{x}; \sigma(t))$ by adding Gaussian noise sampled from $\mathcal{N}(\boldsymbol{0}, \sigma(t)^{2}\boldsymbol{I})$ to the data, where $\sigma(t)$ is a predefined schedule that specifies the noise level at time $t \in [0, T]$ , such that at the end stage the noise mollified distribution $p(\boldsymbol{x}; \sigma(T))$ is indistinguishable from the pure Gaussian distribution. Subsequently, a new sample is generated by progressively denoising a random noise $x_{T} \sim \mathcal{N}(\boldsymbol{0}, \sigma(T)^{2}\boldsymbol{I})$ to its corresponding clean image $x_{0}$ .

Following [4], this forward and backward diffusion process can be expressed with a probabilistic ODE:

$$
d \boldsymbol {x} = - \dot {\sigma} (t) \sigma (t) \nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)) d t. \tag {1}
$$

In practice the score function $\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}; \sigma(t))$ can be approximated by

$$
\nabla_ {\boldsymbol {x}} \log p (\boldsymbol {x}; \sigma (t)) = \left(\mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x}; \sigma (t)) - \boldsymbol {x}\right) / \sigma (t) ^ {2}, \tag {2}
$$

where $\mathcal{D}_{\theta}(\pmb{x};\sigma (t))$ is parameterized by a deep network with parameters $\pmb{\theta}$ trained with the denoising score matching objective:

$$
\min _ {\boldsymbol {\theta}} \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text {data}}} \mathbb {E} _ {\boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}, \sigma (t) ^ {2} \boldsymbol {I})} \left[ \| \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} + \boldsymbol {\epsilon}; \sigma (t)) - \boldsymbol {x} \| _ {2} ^ {2} \right]. \tag {3}
$$

In the discrete setting, the reverse ODE in (1) takes the following form:

$$
\boldsymbol {x} _ {i + 1} \leftarrow (1 - (t _ {i} - t _ {i + 1}) \frac {\dot {\sigma} (t _ {i})}{\sigma (t _ {i})}) \boldsymbol {x} _ {i} + (t _ {i} - t _ {i + 1}) \frac {\dot {\sigma} (t _ {i})}{\sigma (t _ {i})} \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {i}; \sigma (t _ {i})), \tag {4}
$$

where $\boldsymbol{x}_{0} \sim \mathcal{N}(\boldsymbol{0}, \sigma^{2}(t_{0})\boldsymbol{I})$ . Notice that at each iteration i, the intermediate sample $x_{i+1}$ is the sum of the scaled $x_{i}$ and the denoising output $\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x}_{i}; \sigma(t_{i}))$ . Obviously, the final sampled image is largely determined by the denoiser $\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x}; \sigma(t))$ . If we can understand the function mapping of these diffusion denoisers, we can demystify the working mechanism of diffusion models.

Optimal Diffusion Denoisers under Simplified Data Assumptions. Under certain assumptions on the data distribution $p_{\mathrm{data}}(\boldsymbol{x})$ , the optimal diffusion denoisers $\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x};\sigma(t))$ that minimize the score matching objective (3) can be derived analytically in closed-forms as we discuss below.

\- Multi-delta distribution of the training data. Suppose the training dataset contains a finite number of data points $\{\boldsymbol{y}_1, \boldsymbol{y}_2, ..., \boldsymbol{y}_N\}$ , a natural way to model the data distribution is to represent it as a multi-delta distribution: $p(\boldsymbol{x}) = \frac{1}{N} \sum_{i=1}^{N} \delta(\boldsymbol{x} - \boldsymbol{y}_i)$ . In this case, the optimal denoiser is

$$
\mathcal {D} _ {\mathrm{M}} (\boldsymbol {x}; \sigma (t)) = \frac {\sum_ {i = 1} ^ {N} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma (t) ^ {2} \boldsymbol {I}) \boldsymbol {y} _ {i}}{\sum_ {i = 1} ^ {N} \mathcal {N} (\boldsymbol {x} ; \boldsymbol {y} _ {i} , \sigma (t) ^ {2} \boldsymbol {I})}, \tag {5}
$$

which is essentially a softmax-weighted combination of the finite data points. As proved in [24], such diffusion denoisers $\mathcal{D}_{\mathrm{M}}(\boldsymbol{x};\sigma(t))$ can only generate exact replicas of the training samples, therefore they have no generalizability.

\- Multivariate Gaussian distribution. Recent work [25] suggests modeling the data distribution $p_{\mathrm{data}}(\boldsymbol{x})$ as a multivariate Gaussian distribution $p(\boldsymbol{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ , where the mean $\boldsymbol{\mu}$ and the covariance $\boldsymbol{\Sigma}$ are approximated by the empirical mean $\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \boldsymbol{y}_i$ and the empirical covariance $\boldsymbol{\Sigma} = \frac{1}{N} \sum_{i=1}^{N} (\boldsymbol{y}_i - \boldsymbol{\mu})(\boldsymbol{y}_i - \boldsymbol{\mu})^T$ of the training dataset. In this case, the optimal denoiser is:

$$
\mathcal {D} _ {\mathrm{G}} (\boldsymbol {x}; \sigma (t)) = \boldsymbol {\mu} + \boldsymbol {U} \tilde {\boldsymbol {\Lambda}} _ {\sigma (t)} \boldsymbol {U} ^ {T} (\boldsymbol {x} - \boldsymbol {\mu}), \tag {6}
$$

where $\Sigma = U\Lambda U^{T}$ is the SVD of the empirical covariance matrix, with singular values $\Lambda = \text{diag}(\lambda_{1}, \cdots, \lambda_{d})$ and $\tilde{\Lambda}_{\sigma(t)} = \text{diag}\left(\frac{\lambda_{1}}{\lambda_{1} + \sigma(t)^{2}}, \cdots, \frac{\lambda_{d}}{\lambda_{d} + \sigma(t)^{2}}\right)$ . With this linear Gaussian denoiser, as proved in [25], the sampling trajectory of the probabilistic ODE (1) has close form:

$$
\boldsymbol {x} _ {t} = \boldsymbol {\mu} + \sum_ {i = 1} ^ {d} \sqrt {\frac {\sigma (t) ^ {2} + \lambda_ {i}}{\sigma (T) ^ {2} + \lambda_ {i}}} \boldsymbol {u} _ {i} ^ {T} (\boldsymbol {x} _ {T} - \boldsymbol {\mu}) \boldsymbol {u} _ {i}, \tag {7}
$$

where $u_{i}$ is the $i^{th}$ singular vector of the empirical covariance matrix. While [25] demonstrate that the Gaussian scores approximate learned scores at high noise variances, we show that they are nearly the best linear approximations of learned scores across a much wider range of noise variances.

Generalization vs. Memorization of Diffusion Models. As the training dataset size increases, diffusion models transition from the memorization regime—where they can only replicate its training images—to the generalization regime, where the they produce high-quality, novel images $[17]$ . While memorization can be interpreted as an overfitting of diffusion models to the training samples, the mechanisms underlying the generalization regime remain less well understood. This study aims to explore and elucidate the inductive bias that enables effective generalization in diffusion models.

# 3 Hidden Linear and Gaussian Structures in Diffusion Models

In this section, we study the intrinsic structures of the learned score functions of diffusion models in the generalization regime. Through various experiments and theoretical investigation, we show that

# Diffusion models in the generalization regime have inductive bias towards learning the Gaussian structures of the dataset.

Based on the linearity observed in diffusion denoisers trained in the generalization regime, we propose to investigate their intrinsic properties through a linear distillation technique, with which we train a series of linear models to approximate the nonlinear diffusion denoisers (Section 3.1). Interestingly, these linear models closely resemble the optimal denoisers for a multivariate Gaussian distribution characterized by the empirical mean and covariance of the training dataset (Section 3.2). This implies diffusion models have the inductive bias towards learning the Gaussian structure of the

training dataset. We theoretically show that the observed Gaussian structure is the optimal solution to the denoising score matching objective under the constraint that the model is linear (Section 3.3). In the subsequent sections, although we mainly demonstrate our results using the FFHQ datasets, our findings are robust and extend to various architectures and datasets, as detailed in Appendix G.

# 3.1 Diffusion Models Exhibit Linearity in the Generalization Regime

Our study is motivated by the emerging linearity observed in diffusion models in the generalization regime. Specifically, we quantify the linearity of diffusion denoisers at various noise level $\sigma(t)$ by jointly assessing their "Additivity" and "Homogeneity" with a linearity score (LS) defined by the cosine similarity between $\mathcal{D}_{\boldsymbol{\theta}}(\alpha\boldsymbol{x}_{1}+\beta\boldsymbol{x}_{2};\sigma(t))$ and $\alpha\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x}_{1};\sigma(t))+\beta\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x}_{2};\sigma(t))$ :

$$
\operatorname{LS} (t) = \mathbb {E} _ {\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \sim p (\boldsymbol {x}; \sigma (t))} \left[ \left| \left\langle \frac {\mathcal {D} _ {\boldsymbol {\theta}} (\alpha \boldsymbol {x} _ {1} + \beta \boldsymbol {x} _ {2} ; \sigma (t))}{\| \mathcal {D} _ {\boldsymbol {\theta}} (\alpha \boldsymbol {x} _ {1} + \beta \boldsymbol {x} _ {2} ; \sigma (t)) \| _ {2}}, \frac {\alpha \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {1} ; \sigma (t)) + \beta \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {2} ; \sigma (t))}{\| \alpha \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {1} ; \sigma (t)) + \beta \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {2} ; \sigma (t)) \| _ {2}} \right\rangle \right| \right],
$$

where $\pmb{x}_1, \pmb{x}_2 \sim p(\pmb{x}; \sigma(t))$ , and $\alpha \in \mathbb{R}$ and $\beta \in \mathbb{R}$ are scalars. In practice, the expectation is approximated with its empirical mean over 100 samples. A more detailed discussion on this choice of measuring linearity is deferred to Appendix A.

Following the EDM training configuration [4], we set the noise levels $\sigma(t)$ within the continuous range [0.002,80]. As shown in Figure 1, as diffusion models transition from the memorization regime to the generalization regime (increasing the training dataset size), the corresponding diffusion denoisers $D_{\theta}$ exhibit increasing linearity. This phenomenon persists across diverse datasets $^{1}$ as well as various training configurations $^{2}$ ; see Appendix B for more details. This emerging linearity motivates us to ask the following questions:

- To what extent can a diffusion model be approximated by a linear model?   
- If diffusion models can be approximated linearly, what are the underlying characteristics of this linear approximation?

![](images/34f5fe7b1de71a33eebfe86f64bc7f056eb823c91f654efe6cb2f47c1af0c031.jpg)

<details>
<summary>line</summary>

| Noise Variance | dataset size = 70000 | dataset size = 35000 | dataset size = 8750 | dataset size = 4375 | dataset size = 1094 | dataset size = 137 | dataset size = 68 |
| -------------- | --------------------- | --------------------- | ------------------- | ------------------- | ------------------- | ------------------ | ----------------- |
| 0              | 1.00                  | 1.00                  | 1.00                | 1.00                | 1.00                | 1.00               | 1.00              |
| 10             | 0.98                  | 0.97                  | 0.96                | 0.95                | 0.85                | 0.78               | 0.75              |
| 20             | 0.99                  | 0.98                  | 0.98                | 0.97                | 0.92                | 0.85               | 0.80              |
| 30             | 0.99                  | 0.99                  | 0.99                | 0.98                | 0.95                | 0.90               | 0.85              |
| 40             | 0.99                  | 0.99                  | 0.99                | 0.98                | 0.96                | 0.92               | 0.90              |
| 50             | 0.99                  | 0.99                  | 0.99                | 0.98                | 0.97                | 0.95               | 0.92              |
| 60             | 0.99                  | 0.99                  | 0.99                | 0.98                | 0.96                | 0.96               | 0.95              |
| 70             | 0.99                  | 0.99                  | 0.99                | 0.98                | 0.96                | 0.95               | 0.96              |
| 80             | 1.00                  | 1.00                  | 1.00                | 1.00                | 1.00                | 1.00               | 1.00              |
</details>

Figure 1: Linearity scores of diffusion denoisers. Solid and dashed lines depict the linearity scores across noise variances for models in the generalization and memorization regimes, respectively, where $\alpha = \beta = 1/\sqrt{2}$ .

Investigating the Linear Structures via Linear Distillation. To address these questions, we investigate the hidden linear structure of diffusion denoisers through linear distillation. Specifically, for a given diffusion denoiser $\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x};\sigma(t))$ at noise level $\sigma(t)$ , we approximate it with a linear function (with a bias term) such that:

$$
\mathcal {D} _ {\mathrm{L}} (\boldsymbol {x}; \sigma (t)) := \boldsymbol {W} _ {\sigma (t)} \boldsymbol {x} + \boldsymbol {b} _ {\sigma (t)} \approx \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x}; \sigma (t)), \forall \boldsymbol {x} \sim p (\boldsymbol {x}; \sigma (t)), \tag {8}
$$

where the weight $\boldsymbol{W}_{\sigma(t)} \in \mathbb{R}^{d \times d}$ and bias $\boldsymbol{b}_{\sigma(t)} \in \mathbb{R}^{d}$ are learned by solving the following optimization problem with gradient descent: $^{3}$

$$
\min _ {\boldsymbol {W} _ {\sigma (t)}, \boldsymbol {b} _ {\sigma (t)}} \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text {data}} (\boldsymbol {x})} \mathbb {E} _ {\boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}, \sigma (t) ^ {2} \boldsymbol {I})} | | \boldsymbol {W} _ {\sigma (t)} (\boldsymbol {x} + \boldsymbol {\epsilon}) + \boldsymbol {b} _ {\sigma (t)} - \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} + \boldsymbol {\epsilon}; \sigma (t)) | | _ {2} ^ {2}. \tag {9}
$$

If these linear models effectively approximate the nonlinear diffusion denoisers, analyzing their weights can elucidate the generation mechanism.

While diffusion models are trained on continuous noise variance levels within $[0.002,80]$ , we examine the 10 discrete sampling steps specified by the EDM schedule $[4]$ : $[80.0, 42.415, 21.108, 9.723, 4.06, 1.501, 0.469, 0.116, 0.020, 0.002]$ . These steps are considered sufficient for studying the diffusion mappings for two reasons: (i) images generated using these 10 steps closely match those generated

![](images/922a2e2b347436d80d7c94389ef4aa3919fed4b71e324e2606fcd07bb37e4f65.jpg)

<details>
<summary>line</summary>

Score Field Approximation Error
| Noise Variance | Gaussian vs. EDM | Linear vs. EDM | Multi-Delta vs. EDM | Linear vs. Gaussian |
|---|---|---|---|---|
| 0 | 0.01 | 0.04 | 0.36 | 0.02 |
| 10 | 0.13 | 0.13 | 0.27 | 0.02 |
| 20 | 0.05 | 0.05 | 0.04 | 0.03 |
| 40 | 0.03 | 0.03 | 0.03 | 0.03 |
| 80 | 0.05 | 0.02 | 0.05 | 0.05 |
</details>

![](images/546174e5f05d409d027b9c84e72c87ad170396b2ce622cafe0c62f17aee5de89.jpg)

<details>
<summary>bar</summary>

Generation Trajectories (D(x_t; σ(t))) for Various Models
| Model | 80.0 | 42.415 | 21.109 | 9.723 | 4.066 | 1.502 | 0.47 | 0.117 | 0.02 | 0.002 |
|---|---|---|---|---|---|---|---|---|---|---|
| EDM |  |  |  |  |  |  |  |  |  |  |
| Multi-Delta |  |  |  |  |  |  |  |  |  |  |
| Gaussian |  |  |  |  |  |  |  |  |  |  |
| Linear |  |  |  |  |  |  |  |  |  |  |
</details>

Figure 2: Score field approximation error and sampling Trajectory. The left and right figures demonstrate the score field approximation error and the sampling trajectories $\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ of actual diffusion model (EDM), Multi-Delta model, linear model and Gaussian model respectively. Notice that the curve corresponding to the Gaussian model almost overlaps with that of the linear model, suggesting they share similar function mappings.

with more steps, and (ii) recent research [30] demonstrates that the diffusion denoisers trained on similar noise variances exhibit analogous function mappings, implying that denoiser behavior at discrete variances represents their behavior at nearby variances.

After obtaining the linear models $D_{L}$ , we evaluate their differences with the actual nonlinear denoisers $D_{\theta}$ with the score field approximation error, calculated using the expectation over the root mean square error (RMSE):

$$
\text { Score - Difference } (t) := \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text { data }} (\boldsymbol {x}), \boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}; \sigma (t) ^ {2} \boldsymbol {I})} \underbrace {\sqrt {\frac {\| \mathcal {D} _ {\mathrm{L}} (\boldsymbol {x} + \boldsymbol {\epsilon} ; \sigma (t)) - \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} + \boldsymbol {\epsilon} ; \sigma (t)) \| _ {2} ^ {2}}{d}}} _ {\text { RMSE   of   a   pair   of   randomly   sampled   } \boldsymbol {x} \text {   and   } \boldsymbol {\epsilon}}, \tag {10}
$$

where d represents the data dimension and the expectation is approximated with its empirical mean. While we present RMSE-based results in the main text, our findings remain consistent across alternative metrics, including NMSE, as detailed in Appendix G.

We perform linear distillation on well trained diffusion models operating in the generalization regime. For comprehensive analysis, we also compute the score approximation error between $D_{\theta}$ and: (i) the optimal denoisers for the multi-delta distribution $D_{M}$ defined as (5), and (ii) the optimal denoisers for the multivariate Gaussian distribution $D_{G}$ defined as (6). As shown in Figure 2, our analysis reveals three distinct regimes:

- High-noise regime [20,80]. In this regime, only coarse image structures are generated (Figure 2(right)). Quantitatively, as shown in Figure 2(left), the distilled linear model $\mathcal{D}_{\mathrm{L}}$ closely approximates its nonlinear counterpart $\mathcal{D}_{\theta}$ with RMSE below 0.05. Both Gaussian score $\mathcal{D}_{\mathrm{G}}$ and multi-delta score $\mathcal{D}_{\mathrm{M}}$ also achieve comparable approximation accuracy.   
- Low-noise regime [0.002,0.1]. In this regime, only subtle, imperceptible details are added to the generated images. Here, both $\mathcal{D}_{\mathrm{L}}$ and $\mathcal{D}_{\mathrm{G}}$ effectively approximate $\mathcal{D}_{\theta}$ with RMSE below 0.05.   
- Intermediate-noise regime [0.1,20]: This crucial regime, where realistic image content is primarily generated, exhibits significant nonlinearity. While $\mathcal{D}_{\mathrm{M}}$ exhibits high approximation error due to rapid convergence to training samples—a memorization effect theoretically proved in [24], both $\mathcal{D}_{\mathrm{L}}$ and $\mathcal{D}_{\mathrm{G}}$ maintain relatively lower approximation errors.

Qualitatively, as shown in Figure 2(right), despite the relatively high score approximation error in the intermediate noise regime, the images generated with $D_{L}$ closely resemble those generated with $D_{\theta}$ in terms of the overall image structure and certain amount of fine details. This implies (i) the underlying linear structure within the nonlinear diffusion models plays a pivotal role in their generalization capabilities and (ii) such linear structure is effectively captured by our distilled linear models. In the next section, we will explore this linear structure by examining the linear models $D_{L}$ .

# 3.2 Inductive Bias towards Learning the Gaussian Structures

Notably, the Gaussian denoisers $D_{G}$ exhibit behavior strikingly similar to the linear denoisers $D_{L}$ . As illustrated in Figure 2(left), they achieve nearly identical score approximation errors, particularly

![](images/c8ae6f116f46b53f661c51e2f4a3a731501eb2d310d963523489159586560e07.jpg)

<details>
<summary>line</summary>

| Epochs | σ = 80.0 | σ = 42.415 | σ = 21.109 | σ = 9.723 | σ = 4.066 | σ = 1.502 | σ = 0.47 | σ = 0.117 | σ = 0.02 | σ = 0.002 |
|--------|----------|------------|------------|-----------|-----------|-----------|----------|-----------|----------|-----------|
| 0      | 1.0      | 1.0        | 1.0        | 1.0       | 1.0       | 1.0       | 1.0      | 1.0       | 1.0      | 1.0       |
| 20     | 0.5      | 0.4        | 0.3        | 0.2       | 0.15      | 0.1       | 0.08     | 0.06      | 0.04     | 0.02      |
| 40     | 0.4      | 0.3        | 0.2        | 0.15      | 0.1       | 0.08      | 0.06     | 0.04      | 0.02     | 0.01      |
| 60     | 0.3      | 0.2        | 0.15       | 0.1       | 0.08      | 0.06      | 0.04     | 0.03      | 0.01     | 0.005     |
| 80     | 0.25     | 0.15       | 0.1        | 0.08      | 0.06      | 0.04      | 0.03     | 0.02      | 0.01     | 0.005     |
| 100    | 0.2      | 0.1        | 0.08       | 0.06      | 0.04      | 0.03      | 0.02     | 0.01      | 0.01     | 0.005     |
</details>

Figure 4: Linear model shares similar function mapping with Gaussian model. The left figure shows the difference between the linear weights and the Gaussian weights w.r.t. 100 training epochs of the linear distillation process for the 10 discrete noise levels. The right figure shows the correlation matrices between the first 100 singular vectors of the linear weights and Gaussian weights.

in the critical intermediate variance region. Furthermore, their sampling trajectories are remarkably similar (Figure 2(right)), producing nearly identical generated images that closely match those from the actual diffusion denoisers (Figure 3). These observations suggest that $D_{L}$ and $D_{G}$ share similar function mappings across various noise levels, leading us to hypothesize that the intrinsic linear structure underlying diffusion models corresponds to the Gaussian structure of the training data—specifically, its empirical mean and covariance. We validate this hypothesis by empirically showing that $D_{L}$ is close to $D_{G}$ through the following three complementary experiments:

\- Similarity in weight matrices. As illustrated in Figure 4(left), $W_{\sigma(t)}$ progressively converge towards $U\tilde{\Lambda}_{\sigma(t)}U^T$ throughout the linear distillation process, achieving small normalized MSE (less than 0.2) for most of the noise levels. The less satisfactory convergence behavior at $\sigma(t) = 80.0$ is due to inadequate training of the diffusion models at this particular noise level, which is minimally sampled during the training of actual diffusion models (see Appendix G.2 for more details).

- Similarity in Score functions. Furthermore, Figure 2(left, gray line) demonstrates that $\mathcal{D}_{\mathrm{L}}$ and $\mathcal{D}_{\mathrm{G}}$ maintain small score differences (RMSE less than 0.05) across all noise levels, indicating that these denoisers exhibit similar function mappings throughout the diffusion process.   
- Similarity in principal components. As shown in Figure 4(right), for a wide noise range $(\sigma(t) \in [0.116, 80.0])$ , the leading singular vectors of the linear weights $W_{\sigma(t)}$ (denoted $U_{Linear}$ ) align well with $U$ , the singular vectors of the Gaussian weights. $^{4}$ This implies that $U$ , representing the principal components of the training data, is effectively captured by the diffusion models. In the low-noise regime $(\sigma(t) \in [0.002, 0.116])$ , however, $\mathcal{D}_{\theta}$ approximates the identity mapping, leading to ambiguous singular vectors with minimal impact on image generation. Further analy-

sis of $\mathcal{D}_{\theta}$ 's behavior in the low-noise regime is provided in Appendices D and F.1.

![](images/c94efe99244d9fbb9ec389a0836c7881bfd54c8402c7e1b581b97f8e63409f5e.jpg)

<details>
<summary>text_image</summary>

Image 1 Image 2 Image 3 Image 4 Image 5
EDM
Multi-Delta
Gaussian
Linear
</details>

Figure 3: Images sampled from various Models. The figure shows the samples generated using different models starting from the same initial noises.

Since the optimization problem (9) is convex w.r.t. $\boldsymbol{W}_{\sigma(t)}$ and $\boldsymbol{b}_{\sigma(t)}$ , the optimal solution $D_{L}$ represents the unique optimal linear approximation of $D_{\theta}$ . Our analyses demonstrate that this optimal linear approximation closely aligns with $D_{G}$ , leading to our central finding: diffusion models in the generalization regime exhibit an inductive bias (which we term as the Gaussian inductive bias) towards learning the Gaussian structure of training data. This manifests in two main ways: (i) In the high-noise variance regime, well-trained diffusion models learn $D_{\theta}$ that closely approximate the linear Gaussian denoisers $D_{G}$ ; (ii) As noise variance decreases, although $D_{\theta}$ diverges from $D_{G}$ , $D_{G}$ remains nearly identical to the optimal linear approximation $D_{L}$ , and images generated by $D_{G}$ retain structural similarity to those generated by $D_{\theta}$ .

Finally, we emphasize that the Gaussian inductive bias only emerges in the generalization regime. By contrast, in the memorization regime, Figure 5 shows that $\mathcal{D}_{\mathrm{L}}$ significantly diverges from $\mathcal{D}_{\mathrm{G}}$ , and

![](images/e7103c081b59251fafdc59e25105dc2fdd5ea0de6bee5e2300b7ed91f378aa0f.jpg)

<details>
<summary>line</summary>

Score Field Approximation Error (FFHQ)
| Noise Variance (log scale) | Gaussian vs. EDM (Memorization) | Gaussian vs. EDM (Generalization) | Linear vs. EDM (Memorization) | Linear vs. EDM (Generalization) | Linear vs. Gaussian (Memorization) | Linear vs. Gaussian (Generalization) |
|---|---|---|---|---|---|---|
| 0.00 | 0.15 | 0.00 | 0.03 | 0.03 | 0.15 | 0.03 |
| 0.02 | 0.10 | 0.02 | 0.03 | 0.03 | 0.11 | 0.03 |
| 0.12 | 0.11 | 0.04 | 0.04 | 0.06 | 0.10 | 0.02 |
| 0.47 | 0.12 | 0.05 | 0.09 | 0.12 | 0.16 | 0.02 |
| 1.50 | 0.24 | 0.13 | 0.13 | 0.21 | 0.17 | 0.03 |
| 4.07 | 0.24 | 0.13 | 0.13 | 0.21 | 0.13 | 0.03 |
| 9.72 | 0.18 | 0.17 | 0.18 | 0.18 | 0.13 | 0.05 |
| 21.11 | 0.18 | 0.17 | 0.18 | 0.18 | 0.13 | 0.05 |
| 42.42 | 0.18 | 0.17 | 0.18 | 0.18 | 0.13 | 0.06 |
| 80.00 | 0.18 | 0.17 | 0.18 | 0.18 | 0.13 | 0.06 |
The chart displays a line graph with error bars indicating the RMSE values for each method at specific noise variances (log scale). The legend defines six distinct lines: Gaussian vs. EDM (Memorization), Linear vs. EDM (Generalization), Linear vs. Gaussian (Memorization), Linear vs. Gaussian (Generalization), Gaussian vs. EDM (Generalization), and Linear vs. Gaussian (Generalization). The chart is saved as a PNG file named "FFHQ". The data is presented in a single column format.
</details>

(a)

![](images/8541b6b2b7841630d7fe9c72e3e6cf463f49ba7fb91ea1502083bdcefd19c8ca.jpg)

<details>
<summary>text_image</summary>

Denoising Outputs for σ(t) = 4
Clean Image x Noise ε ~ N(0,σ(t)²I) y = x + ε
Dc(ε;σ(t)) Db(ε;σ(t)) (70000) Db(ε;σ(t)) (35000) Db(ε;σ(t)) (1094) Db(ε;σ(t)) (68)
Dc(y;σ(t)) Db(y;σ(t)) (70000) Db(y;σ(t)) (35000) Db(y;σ(t)) (1094) Db(y;σ(t)) (68)
Generalization Memorization
</details>

(b)   
Figure 5: Comparison between the diffusion denoisers in memorization and generalization regimes. Figure(a) demonstrates that in the memorization regime (trained on small datasets of size 1094 and 68), $D_{L}$ significantly diverges from $D_{G}$ , and both provide substantially poorer approximations of $D_{\theta}$ compared to the generalization regime (trained on larger datasets of size 35000 and 1094). Figure(b) qualitatively shows that the denoising outputs of $D_{\theta}$ closely match those of $D_{G}$ only in the generalization regime—a similarity that persists even when the denoisers process pure noise inputs.

both $D_{G}$ and $D_{L}$ provide considerably poorer approximations of $D_{\theta}$ compared to the generalization regime.

# 3.3 Theoretical Analysis

In this section, we demonstrate that imposing linear constraints on diffusion models while minimizing the denoising score matching objective (3) leads to the emergence of Gaussian structure.

Theorem 1. Consider a diffusion denoiser parameterized as a single-layer linear network, defined as $\mathcal{D}(\boldsymbol{x}_t; \sigma(t)) = \boldsymbol{W}_{\sigma(t)} \boldsymbol{x}_t + \boldsymbol{b}_{\sigma(t)}$ , where $\boldsymbol{W}_{\sigma(t)} \in \mathbb{R}^{d \times d}$ is a linear weight matrix and $\boldsymbol{b}_{\sigma(t)} \in \mathbb{R}^d$ is the bias vector. When the data distribution $p_{data}(\boldsymbol{x})$ has finite mean $\boldsymbol{\mu}$ and bounded positive semidefinite covariance $\boldsymbol{\Sigma}$ , the optimal solution to the score matching objective (3) is exactly the Gaussian denoiser defined in (6):

$$
\mathcal {D} _ {\mathrm{G}} (\boldsymbol {x} _ {t}; \sigma (t)) = \boldsymbol {U} \tilde {\boldsymbol {\Lambda}} _ {\sigma (t)} \boldsymbol {U} ^ {T} (\boldsymbol {x} _ {t} - \boldsymbol {\mu}) + \boldsymbol {\mu},
$$

with $\boldsymbol{W}_{\sigma(t)} = \boldsymbol{U} \tilde{\boldsymbol{\Lambda}}_{\sigma(t)} \boldsymbol{U}^{T}$ and $\boldsymbol{b}_{\sigma(t)} = \left( \boldsymbol{I} - \boldsymbol{U} \tilde{\boldsymbol{\Lambda}}_{\sigma(t)} \boldsymbol{U}^{T} \right) \boldsymbol{\mu}.$

The detailed proof is postponed to Appendix E. This optimal solution corresponds to the classical Wiener filter [31], revealing that diffusion models naturally learn the Gaussian denoisers when constrained to linear architectures. To understand why highly nonlinear diffusion models operate near this linear regime, it is helpful to model the training data distribution as the multi-delta distribution $p(\boldsymbol{x}) = \frac{1}{N}\sum_{i=1}^{N}\delta(\boldsymbol{x} - \boldsymbol{y}_i)$ , where $\{\boldsymbol{y}_1,\boldsymbol{y}_2,\dots,\boldsymbol{y}_N\}$ is the finite training images. Notice that this formulation better reflects practical scenarios where only a finite number of training samples are available rather than the ground truth data distribution. Importantly, it is proved in [25] that the optimal denoisers $\mathcal{D}_{\mathrm{M}}$ in this case is approximately equivalent to $\mathcal{D}_{\mathrm{G}}$ for high noise variance $\sigma(t)$ and query points far from the finite training data. This equivalence explains the strong similarity between $\mathcal{D}_{\mathrm{G}}$ and $D_{\mathrm{M}}$ in the high-noise variance regime, and consequently, why $\mathcal{D}_{\theta}$ and $\mathcal{D}_{\mathrm{G}}$ exhibit high similarity in this regime—deep networks converge to the optimal denoisers for finite training datasets.

However, this equivalence between $D_{G}$ and $D_{M}$ breaks down at lower $\sigma(t)$ values. The denoising outputs of $D_{M}$ are convex combinations of training data points, weighted by a softmax function with temperature $\sigma(t)^{2}$ . As $\sigma(t)^{2}$ decreases, this softmax function increasingly approximates an argmax function, effectively retrieving the training point $y_{i}$ closest to the input x. Learning this optimal solution requires not only sufficient model capacity to memorize the entire training dataset but also, as shown in [32], an exponentially large number of training samples. Due to these learning challenges, deep networks instead converge to local minima $D_{\theta}$ that, while differing from $D_{M}$ , exhibit better generalization property. Our experiments reveal that these learned $D_{\theta}$ share similar function mappings with $D_{G}$ . The precise mechanism driving diffusion models trained with gradient descent towards this particular solution remains an open question for future research.

![](images/55a67ea260f0f20b01090423bbca0f3a526bd9eab438432e9d957f9388262bbd.jpg)

<details>
<summary>line</summary>

| Dataset Size | Noise Variance σ | RMSE |
| ------------ | ---------------- | ---- |
| 68           | ~20              | ~0.15 |
| 137          | ~20              | ~0.12 |
| 1094         | ~20              | ~0.10 |
| 8750         | ~20              | ~0.08 |
| 35000        | ~20              | ~0.07 |
| 70000        | ~20              | ~0.06 |
</details>

Figure 6: Diffusion models learn the Gaussian structure when training dataset is large. Models with a fixed scale (channel size 128) are trained across various dataset sizes. The left and right figures show the score difference and the generated images respectively. "NN" denotes the nearest neighbor in the training dataset to the images generated by the diffusion models.

Notably, modeling $p_{\mathrm{data}}(\boldsymbol{x})$ as a multi-delta distribution reveals a key insight: while unconstrained optimal denoisers (5) perfectly capture the scores of the empirical distribution, they have no generalizability. In contrast, Gaussian denoisers, despite having higher score approximation errors due to the linear constraint, can generate novel images that closely match those produced by the actual diffusion models. This suggests that the generative power of diffusion models stems from the imperfect learning of the score functions of the empirical distribution.

# 4 Conditions for the Emergence of Gaussian Structures and Generalizability

In Section 3, we demonstrate that diffusion models exhibit an inductive bias towards learning denoisers that are close to the Gaussian denoisers. In this section, we investigate the conditions under which this bias manifests. Our findings reveal that this inductive bias is linked to model generalization and is governed by (i) the model capacity relative to the dataset size and (ii) the training duration. For additional results, including experiments on CIFAR-10 dataset, see Appendix F.

# 4.1 Gaussian Structures Emerge when Model Capacity is Relatively Small

First, we find that the Gaussian inductive bias and the generalization of diffusion models are heavily influenced by the relative size of the model capacity compared to the training dataset. In particular, we demonstrate that:

# Diffusion models learn the Gaussian structures when the model capacity is relatively small compared to the size of training dataset.

This argument is supported by the following two key observations:

- Increasing dataset size prompts the emergence of Gaussian structure at fixed model scale. We train diffusion models using the EDM configuration [4] with a fixed channel size of 128 on datasets of varying sizes [68, 137, 1094, 8750, 35000, 70000] until FID convergence. Figure 6(left) demonstrates that the score approximation error between diffusion denoisers $\mathcal{D}_{\theta}$ and Gaussian denoisers $\mathcal{D}_{\mathrm{G}}$ decreases as the training dataset size grows, particularly in the crucial intermediate noise variance regime ( $\sigma(t) \in [0.116, 20]$ ). This increasing similarity between $\mathcal{D}_{\theta}$ and $\mathcal{D}_{\mathrm{G}}$ correlates with a transition in the models' behavior: from a memorization regime, where generated images are replicas of training samples, to a generalization regime, where novel images exhibiting Gaussian structure $^{5}$ are produced, as shown in Figure 6(b). This correlation underscores the critical role of Gaussian structure in the generalization capabilities of diffusion models.   
- Decreasing model capacity promotes the emergence of Gaussian structure at fixed dataset sizes. Next, we investigate the impact of model scale by training diffusion models with varying channel sizes [4, 8, 16, 32, 64, 128], corresponding to [64k, 251k, 992k, 4M, 16M, 64M] parameters, on a fixed training dataset of 1094 images. Figure 7(left) shows that in the intermediate noise variance regime

![](images/48eee81e3b1706a9a55e35db076b607ba1341b76ba698363b891ca9a95131b8b.jpg)

Figure 7: Diffusion model learns the Gaussian structure when model scale is small. Models with different scales are trained on a fixed training dataset of 1094 images. The left and right figures show the score difference and the generated images respectively.   
![](images/c1404dd2768e65ab45415abb31018455fe33959ea84ec2f684839d3e1939f762.jpg)

<details>
<summary>line</summary>

| Method | Epochs | Generalization | Memorization |
|---|---|---|---|
| sufficient training (70000 images) | 187 | 0.05 | 128 (70000 images) |
| 64210 epochs (1094 images) | 841 | 0.10 | 841 (70000 images) |
| 32105 epochs (1094 images) | 2293 | 0.15 | 9173 (70000 images) |
| 9173 epochs (1094 images) | 9173 | 0.20 | 9173 (70000 images) |
| 2293 epochs (1094 images) | 1094 | 0.25 | 1094 (70000 images) |
| 841 epochs (1094 images) | 1094 | 0.30 | 1094 (70000 images) |
| 468 epochs (1094 images) | 1094 | 0.25 | 1094 (70000 images) |
| 187 epochs (1094 images) | 64210 | 0.20 | 1094 (70000 images) |
</details>

Figure 8: Diffusion model learns the Gaussian structure in early training epochs. Diffusion model with same scale (channel size 128) is trained using 1094 images. The left and right figures show the score difference and the generated images respectively.

$(\sigma(t) \in [0.116, 20])$ , the discrepancy between $\mathcal{D}_{\theta}$ and $\mathcal{D}_{\mathrm{G}}$ decreases with decreasing model scale, indicating that Gaussian structure emerges in low-capacity models. Figure 7(right) demonstrates that this trend corresponds to a transition from data memorization to the generation of images exhibiting Gaussian structure. Here we note that smaller models lead to larger discrepancy between $\mathcal{D}_{\theta}$ and $\mathcal{D}_{\mathrm{G}}$ in the high-noise regime. This phenomenon arises because diffusion models employ a bell-shaped noise sampling distribution that prioritizes intermediate noise levels, resulting in insufficient training at high noise variances, especially when model capacity is limited (see more details in Appendix F.2).

These two experiments collectively suggest that the inductive bias of diffusion models is governed by the relative capacity of the model compared to the training dataset size.

# 4.2 Overparameterized Models Learn Gaussian Structures before Memorization

In the overparameterized regime, where model capacity significantly exceeds training dataset size, diffusion models eventually memorize the training data when trained to convergence. However, examining the learning progression reveals a key insight:

# Diffusion models learn the Gaussian structures with generalizability before they memorize.

Figure 8(a) demonstrates that during early training epochs (0-841), $D_{\theta}$ progressively converge to $D_{G}$ in the intermediate noise variance regime, indicating that the diffusion model is progressively learning the Gaussian structure in the initial stages of training. Notably, by epoch 841, the diffusion model generates images strongly resembling those produced by the Gaussian model, as shown in Figure 8(b). However, continued training beyond this point increases the difference between $D_{\theta}$ and $D_{G}$ as the model transitions toward memorization. This observation suggests that early stopping could be an effective strategy for promoting generalization in overparameterized diffusion models.

![](images/6fdde7f2d7f2472f4959aafed114882c2691349373f02c4eb3124da789c27bd8.jpg)

<details>
<summary>text_image</summary>

Generated Images from Gaussian Models (size 35000)
Gaussian of S1
Gaussian of S2
Generated Images from Gaussian Models (size 1094)
Gaussian of S1
Gaussian of S2
Non-overlapping datasets with size 35000, model scale 128
Nearest Neighbor in S1
EDM trained on S1
EDM trained on S2
Nearest Neighbor in S2
(a)
Generated Images from Gaussian Models (size 1094)
Gaussian of S1
Gaussian of S2
Non-overlapping datasets with size 1094, model scale 128
Nearest Neighbor in S1
EDM trained on S1
EDM trained on S2
Nearest Neighbor in S2
(b)
Strong generalizability under small dataset size (1094)
Early Stopping at Epoch 921
EDM (S1)
EDM (S2)
Early Stopping
Decosease Scale
(c)
Decrease the Model Scale to 8
</details>

Figure 9: Diffusion models in the strong generalization regime generate similar images as the Gaussian models. Figure(a) Top: Generated images of Gausisan models; Bottom: Generated images of diffusion models, with model scale 128; S1 and S2 each has 35000 non-overlapping images. Figure(b) Top: Generated images of Gausisan model; Bottom: Generated images of diffusion models in the memorization regime, with model scale 128; S1 and S2 each has 1094 non-overlapping images. Figure(c): Early stopping and reducing model capacity help transition diffusion models from memorization to generalization.

# 5 Connection between Strong Generalizability and Gaussian Structure

A recent study $[20]$ reveals an intriguing "strong generalization" phenomenon: diffusion models trained on large, non-overlapping image datasets generate nearly identical images from the same initial noise. While this phenomenon might be attributed to deep networks' inductive bias towards learning the "true" continuous distribution of photographic images, we propose an alternative explanation: rather than learning the complete distribution, deep networks may capture certain low-dimensional common structural features shared across these datasets and these features can be partially explained by the Gaussian structure.

To validate this hypothesis, we examine two diffusion models with channel size 128, trained on non-overlapping datasets S1 and S2 (35000 images each). Figure 9(a) shows that images generated by these models (bottom) closely match those from their corresponding Gaussian models (top), highlighting the Gaussian structure's role in strong generalization.

Comparing Figure 9(a)(top) and (b)(top), we observe that $\mathcal{D}_{\mathrm{G}}$ generates nearly identical images whether the Gaussian structure is calculated on a small dataset (1094 images) or a much larger one (35000 images). This similarity emerges because datasets of the same class can exhibit similar Gaussian structure (empirical covariance) with relatively few samples—just hundreds for FFHQ. Given the Gaussian structure's critical role in generalization, small datasets may already contain much of the information needed for generalization, contrasting previous assertions in [20] that strong generalization requires training on datasets of substantial size (more than $10^{5}$ images). However, smaller datasets increase memorization risk, as shown in Figure 9(b). To mitigate this, as discussed in Section 4, we can either reduce model capacity or implement early stopping (Figure 9(c)). Indeed, models trained on 1094 and 35000 images generate remarkably similar images, though the smaller dataset yields lower perceptual quality. This similarity further demonstrates that small datasets contain substantial generalization-relevant information closely tied to Gaussian structure. Further discussion on the connections and differences between our work and [20] are detailed in Appendix H.

# 6 Discussion

In this study, we empirically demonstrate that diffusion models in the generalization regime have the inductive bias towards learning diffusion denoisers that are close to the corresponding linear Gaussian denoisers. Although real-world image distributions are significantly different from Gaussian, our findings imply that diffusion models have the bias towards learning and utilizing low-dimensional data structures, such as the data covariance, for image generation. However, the underlying mechanism by which the nonlinear diffusion models, trained with gradient descent, exhibit such linearity remains unclear and warrants further investigation.

Moreover, the Gaussian structure only partially explains diffusion models' generalizability. While models exhibit increasing linearity as they transition from memorization to generalization, a substantial gap persists between the linear Gaussian denoisers and the actual nonlinear diffusion models, especially in the intermediate noise regime. As a result, images generated by Gaussian denoisers fall

short in perceptual quality compared to those generated by the actual diffusion models especially for complex dataset such as CIFAR-10. This disparity highlights the critical role of nonlinearity in high-quality image generation, a topic we aim to investigate further in future research.

# Data Availability Statement

The code and instructions for reproducing the experiment results will be made available in the following link: https://github.com/Morefre/Understanding-Generalizability-of-Diffusion-Models-Requires-Rethinking-the-Hidden-Gaussian-Structure.

# Acknowledgment

We acknowledge funding support from NSF CAREER CCF-2143904, NSF CCF-2212066, NSF CCF-2212326, NSF IIS 2312842, NSF IIS 2402950, ONR N00014-22-1-2529, a gift grant from KLA, an Amazon AWS AI Award, and MICDE Catalyst Grant. We also acknowledge the computing support from NCSA Delta GPU [33]. We thank Prof. Rongrong Wang (MSU) for fruitful discussions and valuable feedbacks.

# References

[1] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.   
[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
[3] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations.   
[4] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in Neural Information Processing Systems, 35:26565–26577, 2022.   
[5] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
[6] Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. In International Conference on Learning Representations, 2023.   
[7] Valentin De Bortoli. Convergence of denoising diffusion models under the manifold hypothesis. Transactions on Machine Learning Research, 2022.   
[8] Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence for score-based generative modeling with polynomial complexity. Advances in Neural Information Processing Systems, 35:22870-22882, 2022.   
[9] Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence of score-based generative modeling for general data distributions. In International Conference on Algorithmic Learning Theory, pages 946–985. PMLR, 2023.   
[10] Yuchen Wu, Yuxin Chen, and Yuting Wei. Stochastic runge-kutta methods: Provable acceleration of diffusion models. arXiv preprint arXiv:2410.04760, 2024.   
[11] Gen Li, Yuting Wei, Yuejie Chi, and Yuxin Chen. A sharp convergence theory for the probability flow odes of diffusion models. arXiv preprint arXiv:2408.02320, 2024.

[12] Zhihan Huang, Yuting Wei, and Yuxin Chen. Denoising diffusion probabilistic models are optimally adaptive to unknown low dimensionality. arXiv preprint arXiv:2410.18784, 2024.   
[13] Minshuo Chen, Kaixuan Huang, Tuo Zhao, and Mengdi Wang. Score approximation, estimation and distribution recovery of diffusion models on low-dimensional data. In International Conference on Machine Learning, pages 4672-4712. PMLR, 2023.   
[14] Kazusato Oko, Shunta Akiyama, and Taiji Suzuki. Diffusion models are minimax optimal distribution estimators. In International Conference on Machine Learning, pages 26517-26582. PMLR, 2023.   
[15] Kulin Shah, Sitan Chen, and Adam Klivans. Learning mixtures of gaussians using the ddpm objective. Advances in Neural Information Processing Systems, 36:19636–19649, 2023.   
[16] Hugo Cui, Eric Vanden-Eijnden, Florent Krzakala, and Lenka Zdeborova. Analysis of learning a flow-based generative model from limited sample complexity. In The Twelfth International Conference on Learning Representations, 2023.   
[17] Huijie Zhang, Jinfan Zhou, Yifu Lu, Minzhe Guo, Liyue Shen, and Qing Qu. The emergence of reproducibility and consistency in diffusion models. In Forty-first International Conference on Machine Learning, 2024.   
[18] Peng Wang, Huijie Zhang, Zekai Zhang, Siyi Chen, Yi Ma, and Qing Qu. Diffusion models learn low-dimensional distributions via subspace clustering. arXiv preprint arXiv:2409.02426, 2024.   
[19] Sixu Li, Shi Chen, and Qin Li. A good score does not lead to a good generative model. arXiv preprint arXiv:2401.04856, 2024.   
[20] Zahra Kadkhodaie, Florentin Guth, Eero P Simoncelli, and Stéphane Mallat. Generalization in diffusion models arises from geometry-adaptive harmonic representation. In The Twelfth International Conference on Learning Representations, 2023.   
[21] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? investigating data replication in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6048–6058, 2023.   
[22] Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Understanding and mitigating copying in diffusion models. Advances in Neural Information Processing Systems, 36:47783–47803, 2023.   
[23] TaeHo Yoon, Joo Young Choi, Sehyun Kwon, and Ernest K Ryu. Diffusion probabilistic models generalize when they fail to memorize. In ICML 2023 Workshop on Structured Probabilistic Inference { \& } Generative Modeling, 2023.   
[24] Xiangming Gu, Chao Du, Tianyu Pang, Chongxuan Li, Min Lin, and Ye Wang. On memorization in diffusion models. arXiv preprint arXiv:2310.02664, 2023.   
[25] Binxu Wang and John J Vastola. The hidden linear structure in score-based models and its application. arXiv preprint arXiv:2311.10892, 2023.   
[26] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.   
[27] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.   
[28] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8188–8197, 2020.

[29] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.   
[30] Hyojun Go, Yunsung Lee, Seunghyun Lee, Shinhyeok Oh, Hyeongdon Moon, and Seungtaek Choi. Addressing negative transfer in diffusion models. Advances in Neural Information Processing Systems, 36, 2024.   
[31] Mallat Stéphane. Chapter 11 - denoising. In Mallat Stéphane, editor, A Wavelet Tour of Signal Processing (Third Edition), pages 535–610. Academic Press, Boston, third edition edition, 2009.   
[32] Chen Zeno, Greg Ongie, Yaniv Blumenfeld, Nir Weinberger, and Daniel Soudry. How do minimum-norm shallow denoisers look in function space? Advances in Neural Information Processing Systems, 36, 2024.   
[33] Timothy J Boerner, Stephen Deems, Thomas R Furlani, Shelley L Knuth, and John Towns. Access: Advancing innovation: Nsf's advanced cyberinfrastructure coordination ecosystem: Services & support. In Practice and Experience in Advanced Research Computing, pages 173–176. 2023.   
[34] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.   
[35] DP Kingma. Adam: a method for stochastic optimization. In Int Conf Learn Represent, 2014.   
[36] Alfred O. Hero. Statistical methods for signal processing. 2005.   
[37] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning, pages 1096–1103, 2008.   
[38] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.   
[39] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015.   
[40] Sergey Ioffe. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.   
[41] Andrew L Maas, Awni Y Hannun, Andrew Y Ng, et al. Rectifier nonlinearities improve neural network acoustic models. In Proc. icml, volume 30, page 3. Atlanta, GA, 2013.   
[42] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.   
[43] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.   
[44] Sreyas Mohan, Zahra Kadkhodaie, Eero P Simoncelli, and Carlos Fernandez-Granda. Robust and interpretable blind image denoising via bias-free convolutional neural networks. In International Conference on Learning Representations.   
[45] Hila Manor and Tomer Michaeli. On the posterior distribution in denoising: Application to uncertainty quantification. In The Twelfth International Conference on Learning Representations.   
[46] Siyi Chen, Huijie Zhang, Minzhe Guo, Yifu Lu, Peng Wang, and Qing Qu. Exploring low-dimensional subspaces in diffusion models for controllable image editing. arXiv preprint arXiv:2409.02374, 2024.

# Contents

1 Introduction 1   
2 Preliminary 2   
3 Hidden Linear and Gaussian Structures in Diffusion Models 3

3.1 Diffusion Models Exhibit Linearity in the Generalization Regime ..... 4   
3.2 Inductive Bias towards Learning the Gaussian Structures 5   
3.3 Theoretical Analysis 7

4 Conditions for the Emergence of Gaussian Structures and Generalizability 8

4.1 Gaussian Structures Emerge when Model Capacity is Relatively Small ..... 8   
4.2 Overparameterized Models Learn Gaussian Structures before Memorization ..... 9

5 Connection between Strong Generalizability and Gaussian Structure 10

6 Discussion 10

A Measuring the Linearity of Diffusion Denoisers 15   
B Emerging Linearity of Diffusion Models 16

B.1 Generalization and Memorization Regimes of Diffusion Models ..... 16   
B.2 Diffusion Models Exhibit Linearity in the Generalization Regime ..... 16

C Linear Distillation 16

D Diffusion Models in Low-noise Regime are Approximately Linear Mapping 18   
E Theoretical Analysis 20

E.1 Proof of Theorem 1 20   
E.2 Two Extreme Cases 22

F More Discussion on Section 4 23

F.1 Behaviors in Low-noise Regime 23   
F.2 Behaviors in High-noise Regime 23   
F.3 Similarity between Diffusion Denoiers and Gaussian Denoisers ..... 24   
F.4 CIFAR-10 Results 25

G Additional Experiment Results 25

G.1 Gaussian Structure Emerges across Various Network Architectures ..... 26   
G.2 Gaussian Inductive Bias as a General Property of DAEs 26

G.3 Gaussian Structure Emerges across Various datasets 28   
G.4 Strong Generalization on CIFAR-10 28   
G.5 Measuring Score Approximation Error with NMSE 28

# H Discussion on Geometry-Adaptive Harmonic Bases 30

H.1 GAHB only Partially Explain the Strong Generalization ..... 30   
H.2 GAHB Emerge only in Intermediate-Noise Regime 31

# I Computing Resources 33

# A Measuring the Linearity of Diffusion Denoisers

In this section, we provide a detailed discussion on how to measure the linearity of diffusion model. For a diffusion denoiser, $\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x};\sigma(t))$ , to be considered approximately linear, it must fulfill the following conditions:

- Additivity: The function should satisfy $\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x}_1 + \boldsymbol{x}_2; \sigma(t)) \approx \mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x}_1; \sigma(t)) + \mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x}_2; \sigma(t)).$   
- Homogeneity: It should also adhere to $\mathcal{D}_{\boldsymbol{\theta}}(\alpha\boldsymbol{x};\sigma(t)) \approx \alpha\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x};\sigma(t))$ .

To jointly assess these properties, we propose to measure the difference between $\mathcal{D}_{\boldsymbol{\theta}}(\alpha\boldsymbol{x}_{1}+\beta\boldsymbol{x}_{2};\sigma(t))$ and $\alpha\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x}_{1};\sigma(t))+\beta D(\boldsymbol{x}_{2};\sigma(t))$ . While the linearity score is introduced as the cosine similarity between $\mathcal{D}_{\boldsymbol{\theta}}(\alpha\boldsymbol{x}_{1}+\beta\boldsymbol{x}_{2};\sigma(t))$ and $\alpha\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x}_{1};\sigma(t))+\beta D(\boldsymbol{x}_{2};\sigma(t))$ in the main text:

$$
\mathrm{LS} (t) = \mathbb {E} _ {\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \sim p (\boldsymbol {x}; \sigma (t))} \left[ \left| \left\langle \frac {\mathcal {D} _ {\boldsymbol {\theta}} \left(\alpha \boldsymbol {x} _ {1} + \beta \boldsymbol {x} _ {2} ; \sigma (t)\right)}{\| \mathcal {D} _ {\boldsymbol {\theta}} \left(\alpha \boldsymbol {x} _ {1} + \beta \boldsymbol {x} _ {2} ; \sigma (t)\right) \| _ {2}}, \frac {\alpha \mathcal {D} _ {\boldsymbol {\theta}} \left(\boldsymbol {x} _ {1} ; \sigma (t)\right) + \beta \mathcal {D} _ {\boldsymbol {\theta}} \left(\boldsymbol {x} _ {1} ; \sigma (t)\right)}{\| \alpha \mathcal {D} _ {\boldsymbol {\theta}} \left(\boldsymbol {x} _ {1} ; \sigma (t)\right) + \beta \mathcal {D} _ {\boldsymbol {\theta}} \left(\boldsymbol {x} _ {1} ; \sigma (t)\right) \| _ {2}} \right\rangle \right| \right], \tag {11}
$$

it can also be defined with the normalized mean square difference (NMSE):

$$
\mathbb {E} _ {\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \sim p (\boldsymbol {x}; \sigma (t))} \frac {| | \mathcal {D} _ {\boldsymbol {\theta}} (\alpha \boldsymbol {x} _ {1} + \beta \boldsymbol {x} _ {2} ; \sigma (t)) - (\alpha \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {1} ; \sigma (t)) + \beta \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {1} ; \sigma (t))) | | _ {2}}{| | \mathcal {D} _ {\boldsymbol {\theta}} (\alpha \boldsymbol {x} _ {1} + \beta \boldsymbol {x} _ {2} ; \sigma (t)) | | _ {2}}, \tag {12}
$$

where the expectation is approximated with its empirical mean over 100 randomly sampled pairs of $(\pmb{x}_1, \pmb{x}_2)$ . In the next section, we will demonstrate the linearity score with both metrics.

Since the diffusion denoisers are trained solely on inputs $\boldsymbol{x} \sim p(\boldsymbol{x}; \sigma(t))$ , their behaviors on out-of-distribution inputs can be quite irregular. To produce a denoised output with meaningful image structure, it is critical that the noise component in the input x matches the correct variance $\sigma(t)^{2}$ . Therefore, our analysis of linearity is restricted to in-distribution inputs $x_{1}$ and $x_{2}$ , which are randomly sampled images with additive Gaussian noises calibrated to noise variance $\sigma(t)^{2}$ . We also need to ensure that the values of $\alpha$ and $\beta$ are chosen such that $\alpha^{2} + \beta^{2} = 1$ , maintaining the correct variance for the noise term in the combined input $\alpha x_{1} + \beta x_{2}$ . We present the linearity scores, calculated with varying values of $\alpha$ and $\beta$ , for diffusion models trained on diverse datasets in Figure 10. These models are trained with the EDM-VE configuration proposed in [4], which ensures the resulting models are in the generalization regime. Typically, setting $\alpha = \beta = 1/\sqrt{2}$ yields the lowest linearity score; however, even in this scenario, the cosine similarity remains impressively high, exceeding 0.96. This high value underscores the presence of significant linearity within diffusion denoisers.

We would like to emphasize that for linearity to manifest in diffusion denoisers, it is crucial that they are well-trained, achieving a low denoising score matching loss as indicated in (3). As shown in Figure 11, the linearity notably reduces in a less well trained diffusion model (Baseline-VE) compared to its well-trained counterpart (EDM-VE). Although both models utilize the same 'VE' network architecture $\mathcal{F}_{\theta}(\boldsymbol{x};\sigma(t))$ [2], they differ in how the diffusion denoisers are parameterized:

$$
\mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x}; \sigma (t)) := c _ {\text { skip }} (\sigma (t)) \boldsymbol {x} + c _ {\text { out }} (\mathcal {F} _ {\boldsymbol {\theta}} (\boldsymbol {x}; \sigma (t))), \tag {13}
$$

where $c_{skip}$ is the skip connection and $c_{out}$ modulate the scale of the network output. With carefully tailored $c_{skip}$ and $c_{out}$ , the EDM-VE configuration achieves a lower score matching loss compared to Baseline-VE, resulting in samples with higher quality as illustrated in Figure 11(right).

# B Emerging Linearity of Diffusion Models

In this section we provide a detailed discussion on the observation that diffusion models exhibit increasing linearity as they transition from memorization to generalization, which is briefly described in Section 3.1.

# B.1 Generalization and Memorization Regimes of Diffusion Models

As shown in Figure 12, as the training dataset size increases, diffusion models transition from the memorization regime—where they can only replicate its training images—to the generalization regime, where the they produce high-quality, novel images. To measure the generalization capabilities of diffusion models, it is crucial to assess their ability to generate images that are not mere replications of the training dataset. This can be quantitatively evaluated by generating a large set of images from the diffusion model and measuring the average difference between these generated images and their nearest neighbors in the training set. Specifically, let $\{x_{1}, x_{2}, ..., x_{k}\}$ represent k randomly sampled images from the diffusion models (we choose k = 100 in our experiments), and let $Y := \{y_{1}, y_{2}, ..., y_{N}\}$ denote the training dataset consisting of N images. We define the generalization score as follows:

$$
\text { GL   Score } := \frac {1}{k} \sum_ {i = 1} ^ {k} \frac {\left| \left| \boldsymbol {x} _ {i} - \mathrm{NN} _ {Y} (\boldsymbol {x} _ {i}) \right| \right| _ {2}}{\left| \left| \boldsymbol {x} _ {i} \right| \right| _ {2}} \tag {14}
$$

where $\mathrm{NN}_{Y}(\boldsymbol{x}_{i})$ represents the nearest neighbor of the sample $x_{k}$ in the training dataset Y, determined by the Euclidean distance on a per-pixel basis. Empirically, a GL score exceeding 0.6 indicates that the diffusion models are effectively generalizing beyond the training dataset.

# B.2 Diffusion Models Exhibit Linearity in the Generalization Regime

As demonstrated in Figure 13(a) and (d), diffusion models transition from the memorization regime to the generalization regime as the training dataset size increases. Concurrently, as depicted in Figure 13(b), (c), (e) and (f), the corresponding diffusion denoisers exhibit increasingly linearity. This phenomenon persists across diverse datasets datasets including FFHQ [26], AFHQ [28] and LSUN-Churches [29], as well as various model architectures including EDM-VE [3], EDM-VP [2] and EDM-ADM [34]. This emerging linearity implies that the hidden linear structure plays an important role in the generalizability of diffusion model.

# C Linear Distillation

As discussed in Section 3.1, we propose to study the hidden linearity observed in diffusion denosiers with linear distillation. Specifically, for a given diffusion denoiser $\mathcal{D}_{\theta}(\boldsymbol{x};\sigma(t))$ , we aim to approxi

![](images/82a7ef0291581c9396771e1c303b7c34556ce660bc7aff5f045b0883b4896322.jpg)

<details>
<summary>line</summary>

| Noise Variance | α² = 0.99 | α² = 0.87 | α² = 0.74 | α² = 0.62 | α² = 0.50 |
| -------------- | --------- | --------- | --------- | --------- | --------- |
| 0              | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     |
| 10             | 0.998     | 0.985     | 0.980     | 0.975     | 0.970     |
| 20             | 0.999     | 0.992     | 0.988     | 0.985     | 0.982     |
| 30             | 0.999     | 0.994     | 0.991     | 0.988     | 0.986     |
| 40             | 0.999     | 0.995     | 0.993     | 0.990     | 0.988     |
| 50             | 0.999     | 0.996     | 0.994     | 0.991     | 0.989     |
| 60             | 0.999     | 0.995     | 0.993     | 0.992     | 0.988     |
| 70             | 0.999     | 0.994     | 0.992     | 0.991     | 0.987     |
| 80             | 1.000     | 0.996     | 0.995     | 0.994     | 0.988     |
</details>

![](images/c8ca5f40c5887b254839cd2b43790d40abe6080eb1ae0b5f953c539524e918e1.jpg)

<details>
<summary>line</summary>

| Noise Variance | α² = 0.99 | α² = 0.87 | α² = 0.74 | α² = 0.62 | α² = 0.50 |
| -------------- | --------- | --------- | --------- | --------- | --------- |
| 0              | 1.000     | 1.000     | 1.000     | 1.000     | 1.000     |
| 10             | 1.000     | 0.985     | 0.975     | 0.970     | 0.965     |
| 20             | 1.000     | 0.990     | 0.985     | 0.980     | 0.975     |
| 30             | 1.000     | 0.992     | 0.990     | 0.985     | 0.980     |
| 40             | 1.000     | 0.993     | 0.992     | 0.988     | 0.982     |
| 50             | 1.000     | 0.994     | 0.993     | 0.990     | 0.985     |
| 60             | 1.000     | 0.993     | 0.992     | 0.988     | 0.983     |
| 70             | 1.000     | 0.992     | 0.991     | 0.986     | 0.982     |
| 80             | 1.000     | 0.991     | 0.990     | 0.984     | 0.981     |
</details>

![](images/96f7ca7a92611d9ef58739f5621f3f364f9d474546589722fdb0f346d4cc34da.jpg)

<details>
<summary>line</summary>

| Noise Variance | α² = 0.99 | α² = 0.87 | α² = 0.74 | α² = 0.62 | α² = 0.50 |
| -------------- | --------- | --------- | --------- | --------- | --------- |
| 0              | 1.00      | 1.00      | 1.00      | 1.00      | 1.00      |
| 10             | 0.99      | 0.98      | 0.98      | 0.97      | 0.96      |
| 20             | 0.99      | 0.99      | 0.98      | 0.98      | 0.98      |
| 30             | 0.99      | 0.99      | 0.98      | 0.98      | 0.98      |
| 40             | 0.99      | 0.99      | 0.98      | 0.98      | 0.98      |
| 50             | 0.99      | 0.99      | 0.98      | 0.98      | 0.98      |
| 60             | 0.99      | 0.99      | 0.98      | 0.98      | 0.98      |
| 70             | 0.99      | 0.99      | 0.98      | 0.98      | 0.98      |
| 80             | 0.99      | 0.99      | 0.98      | 0.98      | 0.98      |
</details>

Figure 10: Linearity scores for varying $\alpha$ and $\beta$ . The diffusion models are trained with the edm-ve configuration [4], which ensures the models are in the generalization regime.

![](images/5935c9ff096c6bf126123ac25a91ccc08183b875496a57b10b4266ab916bf4f5.jpg)

<details>
<summary>line</summary>

| Noise Variance | baseline-ve | edm-ve |
| -------------- | ----------- | ------ |
| 0              | 1.00        | 1.00   |
| 10             | 0.97        | 0.98   |
| 20             | 0.94        | 0.98   |
| 30             | 0.88        | 0.98   |
| 40             | 0.82        | 0.98   |
| 50             | 0.76        | 0.98   |
| 60             | 0.71        | 0.98   |
| 70             | 0.68        | 0.98   |
| 80             | 0.65        | 0.99   |
</details>

![](images/b7bbc7f5131076f58866b1c33028e93adf8dbde81bc51cd91bfc09d9f416a6ea.jpg)

<details>
<summary>bar</summary>

Generation Trajectories (D(x₁;σ(t))) for Various Models
| Model | 80.0 | 42.415 | 21.109 | 9.723 | 4.066 | 1.502 | 0.47 | 0.117 | 0.02 | 0.002 |
|---|---|---|---|---|---|---|---|---|---|---|
| EDM-VE |  |  |  |  |  |  |  |  |  |  |
| Baseline-VE |  |  |  |  |  |  |  |  |  |  |
| Gaussian |  |  |  |  |  |  |  |  |  |  |
| Linear |  |  |  |  |  |  |  |  |  |  |
</details>

Figure 11: Linearity scores and sampling trajectory. The left and right figures demonstrate the linearity scores and the sampling trajectories $\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ of actual diffusion model (EDM-VE and Baseline-VE), Multi Delta model, linear model, and Gaussian model respectively.

![](images/6a2d07f6de24ae5d7b6702da04ee91e68d758103b034efdcc2517f09b1404c33.jpg)  
Figure 12: Memorization and generalization regimes of diffusion models. Figures(a) to (c) show the images generated by diffusion models trained on 70000, 4375, 1094 FFHQ images and their corresponding nearest neighbors in the training dataset respectively. Figures(d) to (f) show the images generated by diffusion models trained on 50000, 12500, 782 CIFAR-10 images and their corresponding nearest neighbors in the training dataset respectively. Notice that when the training dataset size is small, diffusion model can only generate images in the training dataset.

mate it with a linear function (with a bias term for more expressibility):

$$
\mathcal {D} _ {\mathrm{L}} (\boldsymbol {x}; \sigma (t)) := \boldsymbol {W} _ {\sigma (t)} \boldsymbol {x} + \boldsymbol {b} _ {\sigma (t)} \approx \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x}; \sigma (t)),
$$

for $\pmb{x} \sim p(\pmb{x}; \sigma(t))$ . Notice that for three dimensional images with size $(c, h, w)$ , $\pmb{x} \in \mathbb{R}^d$ represents their vectorized version, where $d = c \times w \times h$ . Let

$$
\mathcal {L} (\boldsymbol {W}, \boldsymbol {b}) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left\| \boldsymbol {W} _ {\sigma (t)} \{k - 1 \} (\boldsymbol {x} _ {i} + \boldsymbol {\epsilon} _ {i}) + \boldsymbol {b} _ {\sigma (t)} \{k - 1 \} - \mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {i} + \boldsymbol {\epsilon} _ {i}; \sigma (t)) \right\| _ {2} ^ {2}
$$

We train 10 independent linear models for each of the selected noise variance level $\sigma(t)$ with the procedure summarized in Algorithm 1:

In practice, the gradients on $\boldsymbol{W}_{\sigma(t)}$ and $\boldsymbol{b}_{\sigma(t)}$ are obtained through automatic differentiation. Additionally, we employ the Adam optimizer [35] for updates. Additional linear distillation results are provided in Figure 14.

![](images/5e2bb6ea9aa3ae919571bbeedeb61eb6b307356c7d37aab8a784232bb2c6687d.jpg)  
Figure 13: Diffusion model exhibit increasing linearity as they transition from memorization to generalization. Figure(a) and (d) demonstrate that for both FFHQ and CIFAR-10 datasets, the generalization score increases with the training dataset size, indicating progressive model generalization. Figure(b), (c), (e), and (f) show that this transition towards generalization is accompanied by increasing denoiser linearity. Specifically, Figure(b) and (e) display linearity scores calculated using cosine similarity (11), while Figure(c) and (f) show scores computed using NMSE (12). Both metrics reveal consistent trends.

# D Diffusion Models in Low-noise Regime are Approximately Linear Mapping

It should be noted that the low score difference between $D_{G}$ and $D_{\theta}$ within the low-noise regime ( $\sigma(t) \in [0.002, 0.116]$ ) does not imply the diffusion denoisers capture the Gaussian structure, instead, the similarity arises since both of them are converging to the identity mapping as $\sigma(t)$ decreases. As shown in Figure 15, within this regime, the differences between the noisy input x and their corresponding denoised outputs $\mathcal{D}_{\theta}(x; \sigma(t))$ quickly approach 0. This indicates that the learned denoisers $D_{\theta}$ progressively converge to the identity function. Additionally, from (6), it is evident that the difference between the Gaussian weights and the identity matrix diminishes as $\sigma(t)$ decreases, which explains why $D_{G}$ can well approximate $D_{\theta}$ in the low noise variance regime.

We hypothesize that $D_{\theta}$ learns the identity function because of the following two reasons:

(i) within the low-noise regime, since the added noise is negligible compared to the clean image, the identity function already achieves a small denoising error, thus serving as a shortcut which is exploited by the deep network.   
(ii) As discussed in Appendix A, diffusion models are typically parameterized as follows:

$$
\mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x}; \sigma (t)) := c _ {\mathrm{skip}} (\sigma (t)) \boldsymbol {x} + c _ {\mathrm{out}} (\mathcal {F} _ {\boldsymbol {\theta}} (\boldsymbol {x}; \sigma (t))),
$$

where $F_{\theta}$ represents the deep network, and $c_{\mathrm{skip}}(\sigma(t))$ and $c_{\mathrm{out}}(\sigma(t))$ are adaptive parameters for the skip connection and output scaling, respectively, which adjust according to the noise variance levels. For canonical works on diffusion models [2–4, 34], as $\sigma(t)$ approaches zero, $c_{skip}$ and $c_{out}$ converge to 1 and 0 respectively. Consequently, at low variance levels, the function forms of diffusion denoisers are approximately identity mapping: $\mathcal{D}_{\theta}(\boldsymbol{x};\sigma(t))\approx\boldsymbol{x}$ .

This convergence to identity mapping has several implications. First, the weights $\boldsymbol{W}_{\sigma(t)}$ of the distilled linear models $D_{L}$ approach the identity matrix at low variances, leading to ambiguous

# Algorithm 1 Linear Distillation

# Require:

(i) the targeted diffusion denoiser $\mathcal{D}_{\theta}(\cdot ;\sigma (t))$   
(ii) weights $\pmb{W}_{\sigma(t)}$ and biases $\pmb{b}_{\sigma(t)}$ , both initialized to zero,   
(iii) gradient step size η,   
(iv) number of training iterations $K$ ,   
(v) training batch size n,   
(vi) image dataset S.

for k = 1 to K do

Randomly sample a batch of training images $\{\pmb{x}_1, \pmb{x}_2, \dots, \pmb{x}_n\}$ from $S$ .

Randomly sample a batch of noises $\{\epsilon_1, \epsilon_2, \ldots, \epsilon_n\}$ from $\mathcal{N}(\mathbf{0}, \sigma(t)\mathbf{I})$ .

Update $\boldsymbol{W}_{\sigma(t)}$ and $\boldsymbol{b}_{\sigma(t)}$ with gradient descent:

$$
\boldsymbol {W} _ {\sigma (t)} \{k \} = \boldsymbol {W} _ {\sigma (t)} \{k - 1 \} - \eta \nabla_ {\boldsymbol {W} _ {\sigma (t)} \{k - 1 \}} \mathcal {L} (\boldsymbol {W}, \boldsymbol {b})
$$

$$
\boldsymbol {b} _ {\sigma (t)} \{k \} = \boldsymbol {b} _ {\sigma (t)} \{k - 1 \} - \eta \nabla_ {\boldsymbol {b} _ {\sigma (t)} \{k - 1 \}} \mathcal {L} (\boldsymbol {W}, \boldsymbol {b})
$$

end for

Return $W_{\sigma(t)}\{K\}, b_{\sigma(t)}\{K\}$

![](images/5d2b404167fa54923064c58b71513478150381578b11c240ac2120f8f4bb0446.jpg)

FFHQ   
![](images/64b4471b49a61dece9d875b4d1a3a207c5b533df794a05f90ed92fe8b536637d.jpg)

<details>
<summary>text_image</summary>

First 25 Singular Vectors of the Linear Model
First 25 Singular Vectors of the Gaussian Model
</details>

(c)

LSUN-Churches   
![](images/63db2a27dd8c8baecbdaf8236d7f654df36c3f6d315c83efb132c74ed72dc6e9.jpg)

<details>
<summary>text_image</summary>

First 25 Singular Vectors of the Linear Model
First 25 Singular Vectors of the Gaussian Model
</details>

(d)   
Figure 14: Additional linear distillation results. Figure(a) demonstrates the gradual symmetrization of linear weights during the distillation process. Figure(b) shows that at convergence, the singular values of the linear weights closely match those of the Gaussian weights. Figure(c) and Figure(d) display the leading singular vectors of both linear and Gaussian weights at $\sigma(t)=4$ for FFHQ and LSUN-Churches datasets, respectively, revealing a strong correlation.

singular vectors. This explains the poor recovery of singular vectors for $\sigma(t) \in [0.002, 0.116]$ shown in Figure 4. Second, the presence of the bias term in (8) makes it challenging for our linear model to learn the identity function, resulting in large errors at $\sigma(t) = 0.002$ as shown in Figure 4(a).

Finally, from (4), we observe that when $D_{\theta}$ acts as an identity mapping, $x_{i+1}$ remains unchanged from $x_{i}$ . This implies that sampling steps in low-variance regions minimally affect the generated image content, as confirmed in Figure 2, where image content shows negligible variation during these steps.

![](images/ebf72f3d1fbe392be559f8297d7f7a848059bd64b47dbf477abb81b7d3d0c529.jpg)  
Figure 15: Difference between $\mathcal{D}_{\theta}(\boldsymbol{x};\sigma(t))$ and x for various noise variance levels. Figures(a) and (c) show the differences between $\mathcal{D}_{\theta}(\boldsymbol{x};\sigma(t))$ and x across $\sigma(t)\in[0.002,80]$ , measured by normalized MSE and cosine similarity, respectively. Figures(b) and (d) provide zoomed-in views of (a) and (c). The diffusion models were trained on the FFHQ dataset. Notice that the difference between $\mathcal{D}_{\theta}(\boldsymbol{x};\sigma(t))$ and x quickly converges to near zero in the low noise variance regime. The trend is consistent for various model architectures.

# E Theoretical Analysis

# E.1 Proof of Theorem 1

In this section, we give the proof of Theorem 1 (Section 3.3). Our theorem is based on the following two assumptions:

Assumption 1. Suppose that the diffusion denoisers are parameterized as single-layer linear networks, defined as $\mathcal{D}(\boldsymbol{x};\sigma(t)) = \boldsymbol{W}_{\sigma(t)}\boldsymbol{x} + \boldsymbol{b}_{\sigma(t)}$ , where $\boldsymbol{W}_{\sigma(t)} \in \mathbb{R}^{d \times d}$ is the linear weight and $\boldsymbol{b}_{\sigma(t)} \in \mathbb{R}^d$ is the bias.

Assumption 2. The data distribution $p_{data}(\mathbf{x})$ has finite mean $\mu$ and bounded positive semidefinite covariance $\Sigma$

Theorem 1. Under Assumption 1 and Assumption 2, the optimal solution to the denoising score matching objective (3) is exactly the Gaussian denoiser: $\mathcal{D}_{\mathrm{G}}(\boldsymbol{x},\sigma(t)) = \boldsymbol{\mu} + \boldsymbol{U}\tilde{\Lambda}_{\sigma(t)}\boldsymbol{U}^{T}(\boldsymbol{x} - \boldsymbol{\mu})$ , where $\boldsymbol{\Sigma} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{T}$ represents the SVD of the covariance matrix, with singular values $\lambda_{\{k=1,\dots,d\}}$ and $\tilde{\Lambda}_{\sigma(t)} = \mathrm{diag}[\frac{\lambda_k}{\lambda_k + \sigma(t)^2}]$ . Furthermore, this optimal solution can be obtained via gradient descent with a proper learning rate.

To prove Theorem 1, we first show that the Gaussian denoiser is the optimal solution to the denoising score matching objective under the linear network constraint. Then we will show that such optimal solution can be obtained via gradient descent with a proper learning rate.

The Global Optimal Solution. Under the constraint that the diffusion denoiser is restricted to a single-layer linear network with bias:

$$
\mathcal {D} (\boldsymbol {x}; \sigma (t)) = \boldsymbol {W} _ {\sigma (t)} \boldsymbol {x} + \boldsymbol {b} _ {\sigma (t)}, \tag {15}
$$

We get the following optimization problem from Equation (3):

$$
\boldsymbol {W} ^ {\star}, \boldsymbol {b} ^ {\star} = \underset {\boldsymbol {W}, \boldsymbol {b}} {\arg \min} \mathcal {L} (\boldsymbol {W}, \boldsymbol {b}; \sigma (t)) := \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text { data }}} \mathbb {E} _ {\boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}, \sigma (t) ^ {2} \boldsymbol {I})} | | \boldsymbol {W} (\boldsymbol {x} + \boldsymbol {\epsilon}) + \boldsymbol {b} - \boldsymbol {x} | | _ {2} ^ {2}, \tag {16}
$$

where we omit the footnote $\sigma(t)$ in $W_{\sigma(t)}$ and $b_{\sigma(t)}$ for simplicity. Since expectation preserves convexity, the optimization problem Equation (16) is a convex optimization problem. To find the global optimum, we first eliminate $b$ by requiring the partial derivative $\nabla_b\mathcal{L}(W,b;\sigma(t))$ to be 0. Since

$$
\nabla_ {\boldsymbol {b}} \mathcal {L} (\boldsymbol {W}, \boldsymbol {b}; \sigma (t)) = 2 * \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text { data}}} \mathbb {E} _ {\boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}, \sigma (t) ^ {2} \boldsymbol {I})} ((\boldsymbol {W} - \boldsymbol {I}) \boldsymbol {x} + \boldsymbol {W} \boldsymbol {\epsilon} + \boldsymbol {b}) \tag {17}
$$

$$
= 2 * \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text {data}}} ((\boldsymbol {W} - \boldsymbol {I}) \boldsymbol {x} + \boldsymbol {b}) \tag {18}
$$

$$
= 2 * ((\boldsymbol {W} - \boldsymbol {I}) \boldsymbol {\mu} + \boldsymbol {b}), \tag {19}
$$

we have

$$
\boldsymbol {b} ^ {\star} = \left(\boldsymbol {I} - \boldsymbol {W} ^ {*}\right) \boldsymbol {\mu}. \tag {20}
$$

Utilizing the expression for $\pmb{b}$ , we get the following equivalent form of the optimization problem:

$$
\boldsymbol {W} ^ {\star} = \underset {\boldsymbol {W}} {\arg \min} \mathcal {L} (\boldsymbol {W}; \sigma (t)) := 2 * \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text { data}}} \mathbb {E} _ {\boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}, \sigma (t) ^ {2} I)} | | \boldsymbol {W} (\boldsymbol {x} - \boldsymbol {\mu} + \boldsymbol {\epsilon}) - (\boldsymbol {x} - \boldsymbol {\mu}) | | _ {2} ^ {2}. \tag {21}
$$

The derivative $\nabla_{\boldsymbol{W}}\mathcal{L}(\boldsymbol{W};\sigma(t))$ is:

$$
\nabla_ {\boldsymbol {W}} \mathcal {L} (\boldsymbol {W}; \sigma (t)) = 2 * \mathbb {E} _ {\boldsymbol {x}} \mathbb {E} _ {\epsilon} (\boldsymbol {W} (\boldsymbol {x} - \boldsymbol {\mu} + \epsilon) (\boldsymbol {x} - \boldsymbol {\mu} + \epsilon) ^ {T} - (\boldsymbol {x} - \boldsymbol {\mu}) (\boldsymbol {x} - \boldsymbol {\mu} + \epsilon) ^ {T}) \tag {22}
$$

$$
= 2 * \mathbb {E} _ {\boldsymbol {x}} ((\boldsymbol {W} - \boldsymbol {I}) (\boldsymbol {x} - \boldsymbol {\mu}) (\boldsymbol {x} - \boldsymbol {\mu}) ^ {T} + \sigma (t) ^ {2} \boldsymbol {W}) \tag {23}
$$

$$
= 2 * \boldsymbol {W} (\boldsymbol {\Sigma} + \sigma (t) ^ {2} \boldsymbol {I}) - 2 * \boldsymbol {\Sigma}. \tag {24}
$$

Suppose $\pmb{\Sigma} = \pmb{U}\pmb{\Lambda}\pmb{U}^{T}$ is the SVD of the empirical covariance matrix, with singular values $\lambda_{\{k=1,\ldots,n\}}$ , by setting $\nabla_{\pmb{W}}\mathcal{L}(\pmb{W};\sigma(t))$ to $\mathbf{0}$ , we get the optimal solution:

$$
\boldsymbol {W} ^ {\star} = \boldsymbol {U} \boldsymbol {\Lambda} \boldsymbol {U} ^ {T} \boldsymbol {U} (\boldsymbol {\Lambda} + \sigma (t) ^ {2} \boldsymbol {I}) ^ {- 1} \boldsymbol {U} ^ {T} \tag {25}
$$

$$
= \boldsymbol {U} \tilde {\boldsymbol {\Lambda}} _ {\sigma (t)} \boldsymbol {U} ^ {T}, \tag {26}
$$

where $\tilde{\Lambda}_{\sigma(t)}[i,i] = \frac{\lambda_i}{\lambda_i + \sigma(t)^2}$ and $\lambda_{i} = \Lambda[i,i]$ . Substitute $W^{\star}$ back to Equation (20), we have:

$$
\boldsymbol {b} ^ {\star} = (\boldsymbol {I} - \boldsymbol {U} \tilde {\boldsymbol {\Lambda}} _ {\sigma (t)} \boldsymbol {U} ^ {T}) \boldsymbol {\mu}. \tag {27}
$$

Notice that the expression for $W^{\star}$ and $b^{\star}$ is exactly the Gaussian denoiser. Next, we will show this optimal solution can be achieved with gradient descent.

Gradient Descent Recovers the Optimal Solution. Consider minimizing the population loss:

$$
\mathcal {L} (\boldsymbol {W}, \boldsymbol {b}; \sigma (t)) := \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text { data }}} \mathbb {E} _ {\boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}, \sigma (t) ^ {2} \boldsymbol {I})} | | \boldsymbol {W} (\boldsymbol {x} + \boldsymbol {\epsilon}) + \boldsymbol {b} - \boldsymbol {x} | | _ {2} ^ {2}. \tag {28}
$$

Define $\tilde{\boldsymbol{W}} := [\boldsymbol{W} \quad \boldsymbol{b}], \tilde{\boldsymbol{x}} := \begin{bmatrix} \boldsymbol{x} \\ 1 \end{bmatrix}$ and $\tilde{\epsilon} = \begin{bmatrix} \boldsymbol{\epsilon} \\ 0 \end{bmatrix}$ , then we can rewrite Equation (28) as:

$$
\mathcal {L} (\tilde {\boldsymbol {W}}; \sigma (t)) := \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text { data }}} \mathbb {E} _ {\boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}, \sigma (t) ^ {2} \boldsymbol {I})} | | \tilde {\boldsymbol {W}} (\tilde {\boldsymbol {x}} + \tilde {\boldsymbol {\epsilon}}) - \boldsymbol {x} | | _ {2} ^ {2}. \tag {29}
$$

We can compute the gradient in terms of $\tilde{W}$ as:

$$
\nabla \mathcal {L} (\tilde {\boldsymbol {W}}) = 2 * \mathbb {E} _ {\boldsymbol {x}, \epsilon} (\tilde {\boldsymbol {W}} (\tilde {\boldsymbol {x}} + \tilde {\epsilon}) (\tilde {\boldsymbol {x}} + \tilde {\epsilon}) ^ {T} - \boldsymbol {x} (\tilde {\boldsymbol {x}} + \tilde {\epsilon}) ^ {T}) \tag {30}
$$

$$
= 2 * \mathbb {E} _ {\boldsymbol {x}, \epsilon} (\tilde {\boldsymbol {W}} (\tilde {\boldsymbol {x}} \tilde {\boldsymbol {x}} ^ {T} + \tilde {\boldsymbol {x}} \tilde {\boldsymbol {\epsilon}} ^ {T} + \tilde {\boldsymbol {\epsilon}} \tilde {\boldsymbol {x}} ^ {T} + \tilde {\boldsymbol {\epsilon}} \tilde {\boldsymbol {\epsilon}} ^ {T}) - \boldsymbol {x} \tilde {\boldsymbol {x}} ^ {T} - \boldsymbol {x} \tilde {\boldsymbol {\epsilon}} ^ {T}). \tag {31}
$$

Since $\mathbb{E}_{\boldsymbol{\epsilon}}(\tilde{\boldsymbol{\epsilon}}) = \mathbf{0}$ and $\mathbb{E}_{\boldsymbol{\epsilon}}(\tilde{\boldsymbol{\epsilon}}\tilde{\boldsymbol{\epsilon}}^T) = \begin{bmatrix} \sigma(t)^2\boldsymbol{I}_{d\times d} & \boldsymbol{0}_{d\times 1}\\ \boldsymbol{0}_{1\times d} & 0 \end{bmatrix}$ , we have:

$$
\nabla \mathcal {L} (\tilde {\boldsymbol {W}}) = 2 * \mathbb {E} _ {\boldsymbol {x}} (\tilde {\boldsymbol {W}} (\tilde {\boldsymbol {x}} \tilde {\boldsymbol {x}} ^ {T} + \left[ \begin{array}{c c} \sigma (t) ^ {2} \boldsymbol {I} _ {d \times d} & \boldsymbol {0} _ {d \times 1} \\ \boldsymbol {0} _ {1 \times d} & 0 \end{array} \right]) - \boldsymbol {x} \tilde {\boldsymbol {x}} ^ {T}). \tag {32}
$$

Since $\mathbb{E}(\tilde{\boldsymbol{x}}\tilde{\boldsymbol{x}}^T) = \begin{bmatrix} \mathbb{E}(\boldsymbol{x}\boldsymbol{x}^T) & \mathbb{E}(\boldsymbol{x}) \\ \mathbb{E}(\boldsymbol{x}^T) & 1 \end{bmatrix}$ , we have:

$$
\nabla \mathcal {L} (\tilde {\boldsymbol {W}}) = 2 \tilde {\boldsymbol {W}} \left[ \begin{array}{c c} \mathbb {E} _ {\boldsymbol {x}} \left(\boldsymbol {x} \boldsymbol {x} ^ {T}\right) + \sigma (t) ^ {2} \boldsymbol {I} & \boldsymbol {\mu} \\ \boldsymbol {\mu} ^ {T} & 1 \end{array} \right] - 2 \left[ \begin{array}{l l} \mathbb {E} _ {\boldsymbol {x}} \left(\boldsymbol {x} ^ {T} \boldsymbol {x}\right) & \boldsymbol {\mu} \end{array} \right]. \tag {33}
$$

With learning rate $\eta$ , we can write the update rule as:

$$
\tilde {\boldsymbol {W}} (t + 1) = \tilde {\boldsymbol {W}} (t) (1 - 2 \eta \left[ \begin{array}{c c} \mathbb {E} _ {\boldsymbol {x}} (\boldsymbol {x x} ^ {T}) + \sigma (t) ^ {2} \boldsymbol {I} & \boldsymbol {\mu} \\ \boldsymbol {\mu} ^ {T} & 1 \end{array} \right]) + 2 \eta \left[ \begin{array}{l l} \mathbb {E} _ {\boldsymbol {x}} (\boldsymbol {x} ^ {T} \boldsymbol {x}) & \boldsymbol {\mu} \end{array} \right] \tag {34}
$$

$$
= \tilde {\boldsymbol {W}} (t) (1 - 2 \eta \boldsymbol {A}) + 2 \eta \left[ \mathbb {E} _ {\boldsymbol {x}} (\boldsymbol {x} ^ {T} \boldsymbol {x}) \quad \boldsymbol {\mu} \right], \tag {35}
$$

where we define $A := I - 2\eta \begin{bmatrix} \mathbb{E}_{x}(x x^{T}) + \sigma(t)^{2}I & \mu \\ \mu^{T} & 1 \end{bmatrix}$ for simplicity. By recursively expanding the expression for $\tilde{W}$ , we have:

$$
\tilde {\boldsymbol {W}} (t + 1) = \tilde {\boldsymbol {W}} (0) \boldsymbol {A} ^ {t + 1} + 2 \eta \left[ \mathbb {E} _ {\boldsymbol {x}} (\boldsymbol {x} ^ {T} \boldsymbol {x}) \quad \boldsymbol {\mu} \right] \sum_ {i = 0} ^ {t} \boldsymbol {A} ^ {i}. \tag {36}
$$

Notice that there exists a $\eta$ , such that every eigen value of $A$ is smaller than 1 and greater than 0. In this case, $A^{t+1} \to 0$ as $t \to \infty$ . Similarly, by the property of matrix geometric series, we have $\sum_{i=0}^{t} A^i \to (I - A)^{-1}$ . Therefore we have:

$$
\tilde {\boldsymbol {W}} \rightarrow \left[\begin{array}{c c}\mathbb {E} _ {\boldsymbol {x}} (\boldsymbol {x} ^ {T} \boldsymbol {x})&\boldsymbol {\mu}\end{array}\right]\left[\begin{array}{c c}\mathbb {E} _ {\boldsymbol {x}} (\boldsymbol {x x} ^ {T}) + \sigma (t) ^ {2} \boldsymbol {I}&\boldsymbol {\mu}\\\boldsymbol {u} ^ {T}&1\end{array}\right] ^ {- 1} \tag {37}
$$

$$
= \left[ \begin{array}{l l} \mathbb {E} _ {\boldsymbol {x}} (\boldsymbol {x} ^ {T} \boldsymbol {x}) & \boldsymbol {\mu} \end{array} \right] \left[ \begin{array}{l l} \boldsymbol {B} & \boldsymbol {\mu} \\ \boldsymbol {\mu} ^ {T} & 1 \end{array} \right] ^ {- 1}, \tag {38}
$$

where we define $B := \mathbb{E}_{\boldsymbol{x}}(\boldsymbol{x}\boldsymbol{x}^{T}) + \sigma(t)^{2}\boldsymbol{I}$ for simplicity. By the Sherman–Morrison–Woodbury formula, we have:

$$
\left[ \begin{array}{c c} \boldsymbol {B} & \boldsymbol {\mu} \\ \boldsymbol {\mu} ^ {T} & 1 \end{array} \right] ^ {- 1} = \left[ \begin{array}{c c} (\boldsymbol {B} - \boldsymbol {\mu} \boldsymbol {\mu} ^ {T}) ^ {- 1} & - (\boldsymbol {B} - \boldsymbol {\mu} \boldsymbol {\mu} ^ {T}) ^ {- 1} \boldsymbol {\mu} \\ - (1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}) ^ {- 1} \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} & (1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}) ^ {- 1} \end{array} \right]. \tag {39}
$$

Therefore, we have:

$$
\tilde {\boldsymbol {W}} \rightarrow \left[ \mathbb {E} _ {\boldsymbol {x}} [ \boldsymbol {x x} ^ {T} ] (\boldsymbol {B} - \boldsymbol {\mu} \boldsymbol {\mu} ^ {T}) ^ {- 1} - \frac {\boldsymbol {\mu} \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}} - \mathbb {E} _ {\boldsymbol {x}} [ \boldsymbol {x x} ^ {T} ] (\boldsymbol {B} - \boldsymbol {\mu} \boldsymbol {\mu} ^ {T}) ^ {- 1} \boldsymbol {\mu} + \frac {\boldsymbol {\mu}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}} \right], \tag {40}
$$

from which we have

$$
\boldsymbol {W} \rightarrow \mathbb {E} _ {\boldsymbol {x}} [ \boldsymbol {x x} ^ {T} ] (\boldsymbol {B} - \boldsymbol {\mu} \boldsymbol {\mu} ^ {T}) ^ {- 1} - \frac {\boldsymbol {\mu} \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}} \tag {41}
$$

$$
\boldsymbol {b} \rightarrow - \mathbb {E} _ {\boldsymbol {x}} [ \boldsymbol {x x} ^ {T} ] (\boldsymbol {B} - \boldsymbol {\mu} \boldsymbol {\mu} ^ {T}) ^ {- 1} \boldsymbol {\mu} + \frac {\boldsymbol {\mu}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}} \tag {42}
$$

Since $\mathbb{E}_{\boldsymbol{x}}[\boldsymbol{x}\boldsymbol{x}^{T}] = \mathbb{E}_{\boldsymbol{x}}[(\boldsymbol{x} - \boldsymbol{\mu})((\boldsymbol{x} - \boldsymbol{\mu})^{T}] + \boldsymbol{\mu}\boldsymbol{\mu}^{T}$ , we have:

$$
\boldsymbol {W} = \boldsymbol {\Sigma} (\boldsymbol {\Sigma} + \sigma (t) ^ {2} \boldsymbol {I}) ^ {- 1} + \boldsymbol {\mu} \boldsymbol {\mu} ^ {T} (\boldsymbol {B} - \boldsymbol {\mu} \boldsymbol {\mu} ^ {T}) ^ {- 1} - \frac {\boldsymbol {\mu} \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}}. \tag {43}
$$

Applying Sherman-Morrison Formula, we have:

$$
\left(\boldsymbol {B} - \boldsymbol {\mu} \boldsymbol {\mu} ^ {T}\right) ^ {- 1} = \boldsymbol {B} ^ {- 1} + \frac {\boldsymbol {B} ^ {- 1} \boldsymbol {\mu} \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}}, \tag {44}
$$

therefore

$$
\mu \boldsymbol {\mu} ^ {T} (\boldsymbol {B} - \mu \boldsymbol {\mu} ^ {T}) ^ {- 1} - \frac {\mu \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}} = \frac {\mu \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \mu \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}} - \frac {\mu \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}} \tag {45}
$$

$$
= \frac {\boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}}{1 - \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} \boldsymbol {\mu}} (\boldsymbol {\mu} \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1} - \boldsymbol {\mu} \boldsymbol {\mu} ^ {T} \boldsymbol {B} ^ {- 1}) \tag {46}
$$

$$
= 0 \tag {47}
$$

, which implies

$$
\boldsymbol {W} \rightarrow \boldsymbol {\Sigma} (\boldsymbol {\Sigma} + \sigma (t) ^ {2} \boldsymbol {I}) ^ {- 1} \tag {48}
$$

$$
= \boldsymbol {U} \tilde {\boldsymbol {\Lambda}} _ {\sigma (t)} \boldsymbol {U} ^ {T}. \tag {49}
$$

Similarly, we have:

$$
\boldsymbol {b} \rightarrow (\boldsymbol {I} - \boldsymbol {U} \tilde {\boldsymbol {\Lambda}} _ {\sigma (t)} \boldsymbol {U} ^ {T}) \boldsymbol {\mu}. \tag {50}
$$

Therefore, gradient descent with a properly chosen learning rate $\eta$ recovers the Gaussian Denoisers when time goes to infinity.

# E.2 Two Extreme Cases

Our empirical results indicate that the best linear approximation of $D_{\theta}$ is approximately equivalent to $D_{G}$ . According to the orthogonality principle [36], this requires $D_{\theta}$ to satisfy:

$$
\mathbb {E} _ {\boldsymbol {x} \sim p _ {\text { data }} (\boldsymbol {x})} \mathbb {E} _ {\boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}; \sigma (t) ^ {2} \boldsymbol {I})} \left\{\left(\mathcal {D} _ {\boldsymbol {\theta}} (\boldsymbol {x} + \boldsymbol {\epsilon}; \sigma (t)) - (\boldsymbol {x} - \boldsymbol {\mu})\right) (\boldsymbol {x} + \boldsymbol {\epsilon} - \boldsymbol {\mu}) ^ {T} \right\} \approx \boldsymbol {0}. \tag {51}
$$

Notice that (51) does not hold for general denoisers. Two extreme cases for this to hold are:

- Case 1: $\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x} + \boldsymbol{\epsilon};\sigma(t))\approx \boldsymbol{x}$ for $\forall \boldsymbol {x}\sim p_{\mathrm{data}},\boldsymbol {\epsilon}\sim \mathcal{N}(\mathbf{0},\sigma (t)^{2}\boldsymbol {I})$   
- Case 2: $\mathcal{D}_{\boldsymbol{\theta}}(\boldsymbol{x} + \boldsymbol{\epsilon};\sigma(t))\approx \mathcal{D}_{\mathrm{G}}(\boldsymbol{x} + \boldsymbol{\epsilon};\sigma(t))$ for $\forall \boldsymbol {x}\sim p_{\mathrm{data}},\boldsymbol {\epsilon}\sim \mathcal{N}(\mathbf{0},\sigma (t)^{2}\boldsymbol {I})$

Case 1 requires $\mathcal{D}_{\theta}(\boldsymbol{x}+\boldsymbol{\epsilon};\sigma(t))$ to be the oracle denoiser that perfectly recover the ground truth clean image, which never happens in practice except when $\sigma(t)$ becomes extremely small. Instead, our empirical results suggest diffusion models in the generalization regime bias towards Case 2, where deep networks learn $D_{\theta}$ that approximate (not equal) to $D_{G}$ . This is evidenced in Figure 5(b), where diffusion models trained on larger datasets (35000 and 7000 images) produce denoising outputs similar to $D_{G}$ . Notice that this similarity holds even when the denoisers take pure Gaussian noise as input. The exact mechanism driving diffusion models trained with gradient descent towards this particular solution remains an open question and we leave it as future work.

# F More Discussion on Section 4

While in Section 4 we mainly focus on the discussion of the behavior of diffusion denoisers in the intermediate-noise regime, in this section we study the denoiser dynamics in both low and high-noise regime. We also provide additional experiment results on CIFAR-10 dataset.

# F.1 Behaviors in Low-noise Regime

We visualize the score differences between $D_{G}$ and $D_{\theta}$ in low-noise regime in Figure 16. The left figure demonstrates that when the dataset size becomes smaller than a certain threshold, the score difference at $\sigma = 0$ remains persistently non-zero. Moreover, the right figure shows that this difference depends solely on dataset size rather than model capacity. This phenomenon arises from two key factors: (i) $D_{\theta}$ converges to the identity mapping at low noise levels, independent of training dataset size and model capacity, and (ii) $D_{G}$ approximates the identity mapping at low noise levels only when the empirical covariance matrix is full-rank, as can be seen from (6).

Since the rank of the covariance matrix is upper-bounded by the training dataset size, $D_{G}$ differs from the identity mapping when the dataset size is smaller than the data dimension. This creates a persistent gap between $D_{G}$ and $D_{\theta}$ , with smaller datasets leading to lower rank and consequently larger score differences. These observations align with our discussion in Appendix D.

# F.2 Behaviors in High-noise Regime

As shown in Figure 7(a), while a decreased model scale pushes $\mathcal{D}_{\theta}$ in the intermediate noise region towards $\mathcal{D}_{\mathrm{G}}$ , their differences enlarges in the high noise variance regime. This phenomenon arises because diffusion models employ a bell-shaped noise sampling distribution that prioritizes intermediate noise levels, resulting in insufficient training at high noise variances. A shown in Figure 17, for high $\sigma(t)$ , $\mathcal{D}_{\theta}$ converge to $\mathcal{D}_{\mathrm{G}}$ when trained with sufficient model capacity (Figure 17(b)) and training time (Figure 17(c)). This behavior is consistent irrespective of the training dataset sizes (Figure 17(a)). Convergence in the high-noise variance regime is less crucial in practice, since diffusion steps in

![](images/174abc0a65795b860fb278c4a5a5b5fa2350f0a7f1d233320b54d9ed67902f46.jpg)

<details>
<summary>line</summary>

| Noise Variance σ | dataset size = 68 | dataset size = 137 | dataset size = 1094 | dataset size = 8750 | dataset size = 35000 | dataset size = 70000 |
| ---------------- | ----------------- | ------------------ | ------------------- | ------------------- | -------------------- | -------------------- |
| 0.0              | 0.39              | 0.34               | 0.15                | 0.02                | 0.01                 | 0.01                 |
| 0.1              | 0.22              | 0.18               | 0.10                | 0.05                | 0.03                 | 0.03                 |
| 0.2              | 0.21              | 0.19               | 0.12                | 0.07                | 0.05                 | 0.05                 |
| 0.4              | 0.23              | 0.22               | 0.15                | 0.10                | 0.07                 | 0.07                 |
| 0.6              | 0.26              | 0.24               | 0.18                | 0.12                | 0.09                 | 0.09                 |
| 0.8              | 0.27              | 0.25               | 0.20                | 0.14                | 0.11                 | 0.11                 |
| 1.0              | 0.28              | 0.26               | 0.22                | 0.15                | 0.12                 | 0.12                 |
</details>

![](images/6801f0d71ac03806b8bea0d31fa21e2d2202fee263ec266eacfadae2dc6d85ef.jpg)

<details>
<summary>line</summary>

| Noise Variance σ | RMSE (Scale = 128, 70000 images) | RMSE (Scale = 128, 1094 images) | RMSE (Scale = 64, 1094 images) | RMSE (Scale = 32, 1094 images) | RMSE (Scale = 16, 1094 images) | RMSE (Scale = 8, 1094 images) | RMSE (Scale = 4, 1094 images) |
| ---------------- | ---------------------------------- | -------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| 0.0              | ~0.00                              | ~0.15                            | ~0.15                           | ~0.15                           | ~0.15                           | ~0.15                           | ~0.15                           |
| 0.1              | ~0.05                              | ~0.10                            | ~0.10                           | ~0.10                           | ~0.10                           | ~0.10                           | ~0.10                           |
| 0.2              | ~0.07                              | ~0.12                            | ~0.12                           | ~0.12                           | ~0.12                           | ~0.12                           | ~0.12                           |
| 0.3              | ~0.08                              | ~0.15                            | ~0.15                           | ~0.15                           | ~0.15                           | ~0.15                           | ~0.15                           |
| 0.4              | ~0.09                              | ~0.18                            | ~0.18                           | ~0.18                           | ~0.18                           | ~0.18                           | ~0.18                           |
| 0.5              | ~0.10                              | ~0.20                            | ~0.20                           | ~0.20                           | ~0.20                           | ~0.20                           | ~0.20                           |
| 0.6              | ~0.11                              | ~0.22                            | ~0.22                           | ~0.22                           | ~0.22                           | ~0.22                           | ~0.22                           |
| 0.7              | ~0.12                              | ~0.24                            | ~0.24                           | ~0.24                           | ~0.24                           | ~0.24                           | ~0.24                           |
| 0.8              | ~0.13                              | ~0.26                            | ~0.26                           | ~0.26                           | ~0.26                           | ~0.26                           | ~0.26                           |
| 0.9              | ~0.14                              | ~0.28                            | ~0.28                           | ~0.28                           | ~0.28                           | ~0.28                           | ~0.28                           |
| 1.0              | ~0.15                              | ~0.30                            | ~0.30                           | ~0.30                           | ~0.30                           | ~0.30                           | ~0.30                           |
</details>

Figure 16: Score differences for low-noise variances. The left and right figures are the zoomed-in views of Figure 6(a) and Figure 7(a) respectively. Notice that when the dataset size is smaller than the dimension of the image, the score differences are always non-zero at $\sigma = 0$ .

Denoising Outputs for $\sigma(t)=60$ (PSNR=-29.5 dB)   
![](images/db265b92782a45cda022029939801e2e76c3aca445c0fdc7b17017db441822d3.jpg)

Figure 17: $D_{\theta}$ converge to $D_{G}$ with no overfitting for high noise variances. Figure(a) shows the denoising outputs of $D_{M}$ , $D_{G}$ and well-trained (trained with sufficient model capacity till convergence) $D_{\theta}$ . Notice that at high noise variance, the three different denoisers are approximately equivalent despite the training dataset size. Figure(b) shows the denoising outputs of $D_{\theta}$ with different model scales trained until convergence. Notice that $D_{\theta}$ converges to $D_{G}$ only when the model capacity is large enough. Figure(c) shows the denoising outputs of $D_{\theta}$ with sufficient large model capacity at different training epochs. Notice that $D_{\theta}$ converges to $D_{G}$ only when the training duration is long enough.   
![](images/0b7dc396eea6d30a84a709251e3d36b7b3176ba7e86e8fae0635c4553e474fcf.jpg)

<details>
<summary>text_image</summary>

Denoising Outputs for σ(t) = 4
Clean Image r Noise ε ~ N(0,σ(t)²I) y = x + ε
</details>

(a)

![](images/c87d3f3c42e21b75b41c3fb53e524524411095fcd65b9c4a3ad313c549572057.jpg)

<details>
<summary>text_image</summary>

Varying Model Scales
Dc(ε;σ(t)) Dθ(ε;σ(t)) (4) Dθ(ε;σ(t)) (8) Dθ(ε;σ(t)) (64) Dθ(ε;σ(t)) (128)
Dc(y;σ(t)) Dθ(y;σ(t)) (4) Dθ(y;σ(t)) (8) Dθ(y;σ(t)) (64) Dθ(y;σ(t)) (128)
Generalization Memorization
(b)
Varying Training Epochs
Dc(ε;σ(t)) Dθ(ε;σ(t)) (187) Dθ(ε;σ(t)) (841) Dθ(ε;σ(t)) (9173) Dθ(ε;σ(t)) (64210)
Dc(y;σ(t)) Dθ(y;σ(t)) (187) Dθ(y;σ(t)) (841) Dθ(y;σ(t)) (9173) Dθ(y;σ(t)) (64210)
Generalization Memorization
(c)
</details>

Figure 18: Denoising outputs of $D_{G}$ and $D_{\theta}$ at $\sigma = 4$ . Figure(a) shows the clean image x (from test set), random noise $\epsilon$ and the resulting noisy image y. Figure(b) compares denoising outputs of $D_{\theta}$ across different channel sizes [4, 8, 64, 128] with those of $D_{G}$ . Figure(c) shows the evolution of $D_{\theta}$ outputs at training epochs [187, 841, 9173, 64210] alongside $D_{G}$ outputs. All models are trained on a fixed dataset of 1,094 images.

this regime contribute substantially less than those in the intermediate-noise variance regime—a phenomenon we analyze further in Appendix G.5.

# F.3 Similarity between Diffusion Denoiers and Gaussian Denoisers

In Section 4, we demonstrate that the Gaussian inductive bias is most prominent in models with limited capacity and during early training stages, a finding qualitatively validated in Figure 18. Specifically, Figure 18(b) shows that larger models (channel sizes 128 and 64) tend to memorize,

![](images/df3be7f928438ac28222f7447c16636b93df295ed6defa1a6e8b52b2c14b249c.jpg)

Figure 19: Large dataset size prompts the Gaussian structure. Models with the same scale (channel size 64) are trained on CIFAR-10 datasets with varying sizes. Figure(a) shows that larger dataset size leads to increased similarity between $D_{G}$ and $D_{\theta}$ , resulting in structurally similar generated images as shown in Figure(b).   
![](images/ec286478f9df26e9e8baf78d8a9edd18c8f596ece1912b21d2738590f891507c.jpg)  
Figure 20: Smaller model scale prompts the Gaussian structure. Models with varying scales are trained on a fixed CIFAR-10 datasets with 782 images. Figure(a) shows that smaller model scale leads to increased similarity between $D_{G}$ and $D_{\theta}$ in the intermediate noise regime ( $\sigma \in [0.1, 10]$ ), resulting in structurally similar generated images as shown in figure(b). However, smaller scale leads to larger score differences in high-noise regime due to insufficient training from limited model capacity.

directly retrieving training data as denoising outputs. In contrast, smaller models (channel sizes 8 and 4) exhibit behavior similar to $D_{G}$ , producing comparable denoising outputs. Similarly, Figure 18(c) reveals that during early training epochs (0-841), $D_{\theta}$ outputs progressively align with those of $D_{G}$ . However, extended training beyond this point leads to memorization.

# F.4 CIFAR-10 Results

The effects of model capacity and training duration on the Gaussian inductive bias, as demonstrated in Figures 19 to 21, extend to the CIFAR-10 dataset. These results confirm our findings from Section 4: the Gaussian inductive bias is most prominent when model scale and training duration are limited.

# G Additional Experiment Results

While in the main text we mainly demonstrate our findings using EDM-VE diffusion models trained on FFHQ, in this section we show our results are robust and extend to various model architectures and datasets. Furthermore, we demonstrate that the Gaussian inductive bias is not unique to diffusion models, but it is a fundamental property of denoising autoencoders $[37]$ . Lastly, we verify that our

![](images/2658758dbb7cb0dc58334762c33339e81f948e28d6e0fb8b408a0c81bc5f0fc7.jpg)

Figure 21: Diffusion model learns the Gaussian structure in early training epochs. Models with the same scale (channel size 128) are trained on a fixed CIFAR-10 datasets with 782 images. Figure(a) shows that the similarity between $D_{G}$ and $D_{\theta}$ progressively increases during early training epochs (0-921) in the intermediate noise regime ( $\sigma \in [0.1, 10]$ ), resulting in structurally similar generated images as shown in figure(b). However, continue training beyond this point results in diverged $D_{G}$ and $D_{\theta}$ , resulting in memorization.   
![](images/f73bbdb8c661eb491b02539bae06f78194da315083a4f485d813d8d140797928.jpg)

<details>
<summary>line</summary>

EDM-VE (FFHQ)
| Epochs/5 | σ = 80.0 | σ = 42.415 | σ = 21.109 | σ = 9.723 | σ = 4.066 | σ = 1.502 | σ = 0.47 | σ = 0.117 | σ = 0.02 | σ = 0.002 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2.5 | 0.8 | 0.6 | 0.5 | 0.4 | 0.3 | 0.25 | 0.2 | 0.15 | 0.1 | 0.05 |
| 5.0 | 0.6 | 0.4 | 0.3 | 0.25 | 0.2 | 0.15 | 0.15 | 0.1 | 0.05 | 0.02 |
| 7.5 | 0.5 | 0.3 | 0.25 | 0.2 | 0.15 | 0.1 | 0.1 | 0.08 | 0.05 | 0.01 |
| 10.0 | 0.45 | 0.25 | 0.2 | 0.15 | 0.12 | 0.08 | 0.08 | 0.06 | 0.03 | 0.01 |
| 12.5 | 0.42 | 0.22 | 0.18 | 0.13 | 0.1 | 0.07 | 0.07 | 0.05 | 0.02 | 0.01 |
| 15.0 | 0.4 | 0.2 | 0.15 | 0.12 | 0.08 | 0.06 | 0.06 | 0.04 | 0.02 | 0.01 |
| 17.5 | 0.38 | 0.18 | 0.13 | 0.11 | 0.07 | 0.05 | 0.05 | 0.03 | 0.02 | 0.01 |
| 20.0 | 0.35 | 0.15 | 0.12 | 0.1 | 0.06 | 0.04 | 0.04 | 0.02 | 0.02 | 0.01 |
</details>

(a)

![](images/1f4c8f81325ec26fd9665ec1a419bf5b3768021d1ff98b558223dfaa61823f06.jpg)

<details>
<summary>line</summary>

EDM-ADM (FFHQ)
| Epochs | σ = 80.0 | σ = 42.415 | σ = 21.109 | σ = 9.723 | σ = 4.066 | σ = 1.502 | σ = 0.47 | σ = 0.117 | σ = 0.02 | σ = 0.002 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 20 | 0.65 | 0.45 | 0.35 | 0.25 | 0.2 | 0.15 | 0.1 | 0.08 | 0.05 | 0.03 |
| 40 | 0.55 | 0.35 | 0.25 | 0.2 | 0.15 | 0.12 | 0.08 | 0.06 | 0.04 | 0.02 |
| 60 | 0.5 | 0.3 | 0.2 | 0.18 | 0.14 | 0.11 | 0.07 | 0.05 | 0.03 | 0.02 |
| 80 | 0.48 | 0.28 | 0.18 | 0.16 | 0.13 | 0.1 | 0.06 | 0.04 | 0.025 | 0.02 |
| 100 | 0.47 | 0.27 | 0.17 | 0.15 | 0.12 | 0.09 | 0.05 | 0.035 | 0.025 | 0.025 |
</details>

(b)

![](images/39ee1e31f95371fb99d3ee3f273d0935996a2cc923664f37b3d997e00797c3ab.jpg)

<details>
<summary>line</summary>

| Epochs | σ = 80.0 | σ = 42.415 | σ = 21.109 | σ = 9.723 | σ = 4.066 | σ = 1.502 | σ = 0.47 | σ = 0.117 | σ = 0.02 | σ = 0.002 |
| ------ | -------- | ---------- | ---------- | --------- | --------- | --------- | -------- | --------- | -------- | --------- |
| 0      | 1.0      | 1.0        | 1.0        | 1.0       | 1.0       | 1.0       | 1.0      | 1.0       | 1.0      | 1.0       |
| 20     | 0.6      | 0.4        | 0.3        | 0.2       | 0.2       | 0.2       | 0.2      | 0.2       | 0.2      | 0.2       |
| 40     | 0.6      | 0.3        | 0.2        | 0.1       | 0.1       | 0.1       | 0.1      | 0.1       | 0.1      | 0.1       |
| 60     | 0.6      | 0.2        | 0.1        | 0.1       | 0.1       | 0.1       | 0.1      | 0.1       | 0.1      | 0.1       |
| 80     | 0.6      | 0.2        | 0.1        | 0.1       | 0.1       | 0.1       | 0.1      | 0.1       | 0.1      | 0.1       |
| 100    | 0.6      | 0.2        | 0.1        | 0.1       | 0.1       | 0.1       | 0.1      | 0.1       | 0.1      | 0.1       |
</details>

(c)   
Figure 22: Linear model shares similar function mapping with Gaussian model. The figures demonstrate the evolution of normalized MSE between the linear weights $D_{L}$ and the Gaussian weights $D_{G}$ w.r.t. linear distillation training epochs. Figures(a), (b) and (c) correspond to diffusion models trained on FFHQ, with EDM-VE, EDM-ADM and EDM-VP network architectures specified in [4] respectively.

conclusions remain consistent when using alternative metrics such as NMSE instead of the RMSE used in the main text.

# G.1 Gaussian Structure Emerges across Various Network Architectures

We first demonstrate that diffusion models capture the Gaussian structure of the training dataset, irrespective of the deep network architectures used. As shown in Figure 22 (a), (b), and (c), although the actual diffusion models, $\mathcal{D}_{\theta}$ , are parameterized with different architectures, for all noise variances except $\sigma(t) \in \{0.002, 80.0\}$ , their corresponding linear models, $\mathcal{D}_{\mathrm{L}}$ , consistently converge towards the common Gaussian models, $\mathcal{D}_{\mathrm{G}}$ , determined by the training dataset. Qualitatively, as depicted in Figure 23, despite variations in network architectures, diffusion models generate nearly identical images, matching those generated from the Gaussian models.

# G.2 Gaussian Inductive Bias as a General Property of DAEs

In previous sections, we explored the properties of diffusion models by interpreting them as collections of deep denoisers, which are equivalent to the denoising autoencoders (DAEs) [37] trained on various noise variances by minimizing the denoising score matching objective (3). Although diffusion models and DAEs are equivalent in the sense that both of them are trying to learn the score function of the

![](images/ecb8f2da00134bf8c8855c476c5facdfef24eaa14ed73900702fb22ad8c91995.jpg)

<details>
<summary>text_image</summary>

image 1
image 2
image 3
image 4
image 5
image 6
Gaussian
EDM-VE
EDM-VP
EDM-ADM
</details>

Figure 23: Images sampled from various model. The figure shows the sampled images from diffusion models with different network architectures and those from their corresponding Gaussian models.

noise-mollified data distribution [38], the training objective of diffusion models is more complex [4]:

$$
\min _ {\boldsymbol {\theta}} \mathbb {E} _ {\boldsymbol {x}, \boldsymbol {\epsilon}, \sigma} [ \lambda (\sigma) c _ {\text { out }} (\sigma) ^ {2} | | \mathcal {F} _ {\boldsymbol {\theta}} (\boldsymbol {x} + \boldsymbol {\epsilon}, \sigma) - \underbrace {\frac {1}{c _ {\text { out }} (\sigma)} (\boldsymbol {x} - c _ {\text { skip }} (\sigma) (\boldsymbol {x} + \boldsymbol {\epsilon}))} _ {\text { linear   combination   of } \boldsymbol {x} \text { and } \boldsymbol {\epsilon}} | | _ {2} ^ {2} ], \tag {52}
$$

where $x \sim p_{data}$ , $\epsilon \sim \mathcal{N}(\mathbf{0}, \sigma(t)^{2}\mathbf{I})$ and $\sigma \sim p_{train}$ . Notice that the training objective of diffusion models has a few distinct characteristics:

- Diffusion models use a single deep network $\mathcal{F}_{\theta}$ to perform denoising score matching across all noise variances while DAEs are typically trained separately for each noise level.   
- Diffusion models are not trained uniformly across all noise variances. Instead, during training the probability of sampling a given noise level $\sigma$ is controlled by a predefined distribution $p_{\mathrm{train}}$ and the loss is weighted by $\lambda(\sigma)$ .   
- Diffusion models often utilize special parameterizations (13). Therefore, the deep network $\mathcal{F}_{\theta}$ is trained to predict a linear combination of the clean image $x$ and the noise $\epsilon$ , whereas DAEs typically predict the clean image directly.

Given these differences, we investigate whether the Gaussian inductive bias is unique to diffusion models or a general characteristic of DAEs. To this end, we train separate DAEs (deep denoisers) using the vanilla denoising score matching objective (3) on each of the 10 discrete noise variances specified by the EDM schedule [80.0, 42.415, 21.108, 9.723, 4.06, 1.501, 0.469, 0.116, 0.020, 0.002], and compare the score differences between them and the corresponding Gaussian denoisers $\mathcal{D}_{\mathrm{G}}$ . We use no special parameterization so that $\mathcal{D}_{\theta} = \mathcal{F}_{\theta}$ ; that is, the deep network directly predicts the clean image. Furthermore, the DAEs for each noise variance are trained till convergence, ensuring all noise levels are trained sufficiently. We consider the following architectural choices:

- $DAE-NCSN$ : In this setting, the network $\mathcal{F}_{\theta}$ uses the NCSN architecture [3], the same as that used in the EDM-VE diffusion model.   
- DAE-Skip: In this setting, $\mathcal{F}_{\theta}$ is a U-Net [39] consisting of convolutional layers, batch normalization [40], leaky ReLU activation [41] and convolutional skip connections. We refer to this network as "Skip-Net". Compared to NCSN, which adapts the state of the art architecture designs, Skip-Net is deliberately constructed to be as simple as possible to test how architectural complexity affects the Gaussian inductive bias.   
- $DAE-DiT$ : In this setting, $\mathcal{F}_{\theta}$ is a Diffusion Transformer (DiT) introduced in [42]. Vision Transformers are known to lack inductive biases such as locality and translation equivariance that are inherent to convolutional models [43]. Here we are interested in if this affects the Gaussian inductive bias.

![](images/0f319ed20d362dc47a71ba18e3c6767d272b2390edf36a555d456abc45e80016.jpg)

<details>
<summary>line</summary>

Score Field Approximation Error
| Noise Variance (log scale) | DAE-Linear vs. Gaussian | DAE-Skip vs. Gaussian | DAE-DiT vs. Gaussian | DAE-NCSN vs. Gaussian | EDM vs. Gaussian |
|---|---|---|---|---|---|
| 0.00 | 0.005 | 0.012 | 0.003 | 0.004 | 0.003 |
| 0.02 | 0.006 | 0.014 | 0.012 | 0.013 | 0.013 |
| 0.12 | 0.007 | 0.045 | 0.045 | 0.048 | 0.048 |
| 0.47 | 0.009 | 0.085 | 0.085 | 0.095 | 0.095 |
| 1.50 | 0.011 | 0.128 | 0.122 | 0.135 | 0.128 |
| 4.07 | 0.013 | 0.132 | 0.128 | 0.132 | 0.132 |
| 9.72 | 0.023 | 0.128 | 0.128 | 0.128 | 0.128 |
| 21.11 | 0.014 | 0.128 | 0.128 | 0.128 | 0.128 |
| 42.42 | 0.016 | 0.128 | 0.128 | 0.128 | 0.128 |
| 80.00 | 0.023 | 0.128 | 0.128 | 0.128 | 0.128 |
</details>

(a)

![](images/7bb5ecaaf68c531eca9bfdb55665168b80b829ad6ce0f5ab59b2105135eb5b58.jpg)

<details>
<summary>other</summary>

Generation Trajectories (D(x₁; a(t))) for Various Models
| Model | 80.0 | 42.415 | 21.109 | 9.723 | 4.066 | 1.502 | 0.47 | 0.117 | 0.02 | 0.002 |
|---|---|---|---|---|---|---|---|---|---|---|
| EDM |  |  |  |  |  |  |  |  |  |  |
| DAE-Skip |  |  |  |  |  |  |  |  |  |  |
| DAE-DIT |  |  |  |  |  |  |  |  |  |  |
| Gaussian |  |  |  |  |  |  |  |  |  |  |
</details>

(b)   
Figure 24: Comparison between DAEs and diffusion models. Figure(a) compares the score field approximation error between Gaussian models and both (i) diffusion models (EDM vs. Gaussian) and (ii) DAEs with varying architectures. Figure(b) illustrates the generation trajectories of different models initialized from the same noise input.

\- DAE-Linear: In this setting we set $\mathcal{F}_{\theta}$ to be a linear model with a bias term as in (8). According to Theorem 1, these models should converge to Gaussian denoisers.

The quantitative results are shown in Figure 24(a). First, the DAE-linear models well approximate $D_{G}$ across all 10 discrete steps (RMSE smaller than 0.04), consistent with Theorem 1. Second, despite the differences between diffusion models (EDM) and DAEs, they achieve similar score approximation errors relative to $D_{G}$ for most noise variances, meaning that they can be similarly approximated by $D_{G}$ . However, diffusion models exhibit significantly larger deviations from $D_{G}$ at higher noise variances ( $\sigma \in \{42.415, 80.0\}$ ) since they utilize a bell-shaped noise sampling distribution $p_{train}$ that emphasizes training on intermediate noise levels, leading to under-training at high noise variances. Lastly, the DAEs with different architectures achieve comparable score approximation errors, and both DAEs and diffusion models generate images matching those from the Gaussian model, as shown in Figure 24(b). These findings demonstrate that the Gaussian inductive bias is not unique to diffusion models or specific architectures but is a fundamental property of DAEs.

# G.3 Gaussian Structure Emerges across Various datasets

As illustrated in Figure 25, for diffusion models trained on the CIFAR-10, AFHQ and LSUN-Churches datasets that are in the generalization regime, their generated samples match those produced by the corresponding Gaussian models. Additionally, their linear approximations, $D_{L}$ , obtained through linear distillation, align closely with the Gaussian models, $D_{G}$ , resulting in nearly identical generated images. These findings confirm that the Gaussian structure is prevalent across various datasets.

# G.4 Strong Generalization on CIFAR-10

Figure 26 demonstrates the strong generalization effect on CIFAR-10. Similar to the observations in Section 5, reducing model capacity or early stopping the training process prompts the Gaussian inductive bias, leading to generalization.

# G.5 Measuring Score Approximation Error with NMSE

While in Section 3.1 we define the score field approximation error between denoisers $\mathcal{D}_1$ and $\mathcal{D}_2$ with RMSE ((10)), this error can also be quantified using NMSE:

$$
\text { Score - Difference } (t) := \mathbb {E} _ {\boldsymbol {x} \sim p _ {\text { data }} (\boldsymbol {x}), \boldsymbol {\epsilon} \sim \mathcal {N} (\boldsymbol {0}; \sigma (t) ^ {2} \boldsymbol {I})} \frac {| | \mathcal {D} _ {1} (\boldsymbol {x} + \boldsymbol {\epsilon}) - \mathcal {D} _ {2} (\boldsymbol {x} + \boldsymbol {\epsilon}) | | _ {2}}{| | \mathcal {D} _ {1} (\boldsymbol {x} + \boldsymbol {\epsilon}) | | _ {2}}. \tag {53}
$$

As shown in Figure 27, while the trend in intermediate-noise and low-noise regimes remains unchanged, NMSE amplifies differences in the high-noise variance regime compared to RMSE. This amplified score difference between $D_{G}$ and $D_{\theta}$ does not contradict our main finding that diffusion models in the generalization regime exhibit an inductive bias towards learning denoisers

LSUN-Churches   
![](images/9951cf0afdf1ff24ffbef366326cab6c939a86a84c3a8c2db24ff09679979852.jpg)

<details>
<summary>text_image</summary>

Final Generated Samples
EDM
Multi Delta
Gaussian
Linear
</details>

(a)

![](images/7316070c59bd690226e9cf27ae8e166a36e0af7ad3c595f26d138f55c1bce794.jpg)

<details>
<summary>bar</summary>

Generation Trajectories (D(xᵣ; σ(t))) for Various Models
| Model | 80.0 | 42.415 | 21.109 | 9.723 | 4.066 | 1.502 | 0.47 | 0.117 | 0.02 | 0.002 |
|---|---|---|---|---|---|---|---|---|---|---|
| EDM |  |  |  |  |  |  |  |  |  |  |
| Multi-Delta |  |  |  |  |  |  |  |  |  |  |
| Gaussian |  |  |  |  |  |  |  |  |  |  |
| Linear |  |  |  |  |  |  |  |  |  |  |
</details>

(b)

![](images/1dd0e2281304fd1010040dcd044393a4a786e646cf0b028c07738643198d6dc4.jpg)

<details>
<summary>text_image</summary>

Final Generated Samples
EDM
Multi Delta
Gaussian
Linear
</details>

(c)

![](images/c83ea21536c0ab75e3a47ee3db014ab38f1ba8ee1422821dc250ed5fd2156334.jpg)

<details>
<summary>bar</summary>

AFHQ
| Model | Generation Trajectories (D(xt; σ(t))) for Various Models | D(xt; σ(t)) |
| :--- | :--- | :--- |
| EDM | 80.0 | 42.415 |
| EDM | 42.415 | 21.109 |
| EDM | 21.109 | 9.723 |
| EDM | 9.723 | 4.066 |
| EDM | 4.066 | 1.502 |
| EDM | 1.502 | 0.47 |
| EDM | 0.47 | 0.117 |
| EDM | 0.117 | 0.02 |
| EDM | 0.02 | 0.002 |
| Multi-Delta | 80.0 | |
| Multi-Delta | 42.415 | |
| Multi-Delta | 21.109 | |
| Multi-Delta | 9.723 | |
| Multi-Delta | 4.066 | |
| Multi-Delta | 1.502 | |
| Multi-Delta | 0.47 | |
| Multi-Delta | 0.117 | |
| Multi-Delta | 0.02 | |
| Multi-Delta | 0.002 | |
| Gaussian | 80.0 | |
| Gaussian | 42.415 | |
| Gaussian | 21.109 | |
| Gaussian | 9.723 | |
| Gaussian | 4.066 | |
| Gaussian | 1.502 | |
| Gaussian | 0.47 | |
| Gaussian | 0.117 | |
| Gaussian | 0.02 | |
| Gaussian | 0.002 | |
| Linear | 80.0 | |
| Linear | 42.415 | |
| Linear | 21.109 | |
| Linear | 9.723 | |
| Linear | 4.066 | |
| Linear | 1.502 | |
| Linear | 0.47 | |
| Linear | 0.117 | |
| Linear | 0.02 | |
| Linear | 0.002 | |
</details>

(d)

![](images/6f37c2352e25c1c63aec4a34644971f94bac1342ea5befd67c4966f47b8f2a81.jpg)

<details>
<summary>text_image</summary>

Final Generated Samples
EDM
image 1
image 2
image 3
image 4
image 5
Multi Delta
Gaussian
Linear
</details>

(e)

![](images/55b371a246fb74c3867b6de888a30d3853e5054acbf15821b4f8f7dce8265c40.jpg)

<details>
<summary>scatter</summary>

| Model             | 80.0   | 42.415 | 21.109 | 9.723  | 4.066  | 1.502  | 0.47   | 0.117  | 0.02   | 0.002  |
| ----------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| EDM               |        |        |        |        |        |        |        |        |        |        |
| DAE               |        |        |        |        |        |        |        |        |        |        |
| Gaussian Multi-Delta |        |        |        |        |        |        |        |        |        |        |
| Linear            |        |        |        |        |        |        |        |        |        |        |
</details>

(f)   
Figure 25: Final generated images and sampling trajectories for various models. Figures(a), (c) and (e) demonstrate the images generated using different models starting from the same noises for LSUN-Churches, AFHQ and CIFAR-10 respectively. Figures(b), (d) and (f) demonstrate the corresponding sampling trajectories.

approximately equivalent to $D_{G}$ in the high-noise variance regime. As discussed in Section 3.2 and appendices F.2 and G.2, this large score difference stems from inadequate training in this regime.

Figure 27 (Gaussian vs. DAE) demonstrates that when DAEs are sufficiently trained at specific noise variances, they still converge to $D_{G}$ . Importantly, the insufficient training in the high-noise variance regime minimally affects final generation quality. Figure 25(f) shows that while the diffusion model (EDM) produces noisy trajectories at early timesteps ( $\sigma \in \{80.0, 42.415\}$ ), these artifacts quickly disappear in later stages, indicating that the Gaussian inductive bias is most influential in the intermediate-noise variance regime.

Notably, even when $D_{\theta}$ are inadequately trained in the high-noise variance regime, they remain approximable by linear functions, though these functions no longer match $D_{G}$ .

![](images/ae3af2d16257269249dee7c6bce5c253b633e328d42a26be4a4c8ed0e79db914.jpg)

<details>
<summary>text_image</summary>

Generated Images from Gaussian Models (size 25000)
Gaussian of S1
Gaussian of S2
Generated Images from Gaussian Models (size 782)
Gaussian of S1
Gaussian of S2
Non-overlapping datasets with size 25000, model scale 64
Nearest Neighbor in S1
EDM trained on S1
EDM trained on S2
Nearest Neighbor in S2
Non-overlapping datasets with size 782, model scale 128
Nearest Neighbor in S1
EDM trained on S1
EDM trained on S2
Nearest Neighbor in S2
Strong generalizability under small dataset size (782)
Early Stopping at Epoch 921
EDM (S1)
EDM (S2)
EDM (S1)
EDM (S2)
Decrease the Model Scale to 4
(a)
(b)
(c)
</details>

Figure 26: Strong generalization on CIFAR-10 dataset. Figure(a) Top: Generated images of Gausisan models; Bottom: Generated images of diffusion models, with model scale 64; S1 and S2 each has 25000 non-overlapping images. Figure(b) Top: Generated images of Gausisan model; Bottom: Generated images of diffusion models in the memorization regime, with model scale 128; S1 and S2 each has 782 non-overlapping images. Figure(c): Early stopping and reducing model capacity help transition diffusion models from memorization to generalization.

![](images/03c9c6b6135d073546e1037414569fe10300440cb73d1e9321977d80d1ac2b16.jpg)

<details>
<summary>line</summary>

Score Field Approximation Error (CIFAR-10)
| Noise Variance | Optimal vs. EDM | Gaussian vs. EDM | Linear vs. EDM | Linear vs. Gaussian | Gaussian vs. DAE |
|---|---|---|---|---|---|
| 0 | 0.32 | 0.01 | 0.06 | 0.03 | 0.04 |
| 5 | 0.27 | 0.10 | 0.12 | 0.02 | 0.09 |
| 10 | 0.16 | 0.05 | 0.05 | 0.02 | 0.04 |
| 20 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 |
| 40 | 0.02 | 0.11 | 0.02 | 0.11 | 0.02 |
| 80 | 0.01 | 0.06 | 0.01 | 0.01 | 0.01 |
</details>

![](images/8e0a95783efd28822a9a63c8368019aa3c5b79bad0d413f211ab6ae513778a10.jpg)

<details>
<summary>line</summary>

Score Field Approximation Error (CIFAR-10)
| Noise Variance | Optimal vs. EDM | Gaussian vs. EDM | Linear vs. EDM | Linear vs. Gaussian | Gaussian vs. DAE |
|---|---|---|---|---|---|
| 0 | 0.7 | 0.05 | 0.1 | 0.05 | 0.25 |
| 5 | 0.45 | 0.2 | 0.2 | 0.05 | 0.2 |
| 10 | 0.15 | 0.15 | 0.15 | 0.05 | 0.15 |
| 20 | 0.15 | 0.15 | 0.15 | 0.1 | 0.1 |
| 40 | 1.0 | 0.9 | 0.15 | 0.1 | 0.1 |
| 80 | 0.7 | 0.7 | 0.15 | 0.1 | 0.1 |
</details>

![](images/a6f0c0de8091dc55a57d4e56f73298a8a6c066c64fc6b7553e2a034f27b67dcb.jpg)

<details>
<summary>line</summary>

Score Field Approximation Error (FFHQ)
| Noise Variance | Gaussian vs. EDM | Linear vs. EDM | Multi-Delta vs. EDM | Linear vs. Gaussian | Gaussian vs. DAE |
|---|---|---|---|---|---|
| 0 | 0.03 | 0.12 | 0.36 | 0.01 | 0.14 |
| 10 | 0.13 | 0.10 | 0.27 | 0.02 | 0.13 |
| 20 | 0.05 | 0.05 | 0.04 | 0.03 | 0.04 |
| 40 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 80 | 0.04 | 0.02 | 0.05 | 0.05 | 0.02 |
</details>

![](images/588d1c21229286b9a06669681d143d3fbe7441f90d78029ecf23dddd7d61c57f.jpg)

<details>
<summary>line</summary>

Score Field Approximation Error (FFHQ)
| Noise Variance | Gaussian vs. EDM | Linear vs. EDM | Multi-Delta vs. EDM | Linear vs. Gaussian | Gaussian vs. DAE |
|---|---|---|---|---|---|
| 0 | 0.65 | 0.15 | 0.65 | 0.02 | 0.02 |
| 10 | 0.22 | 0.23 | 0.54 | 0.06 | 0.27 |
| 20 | 0.15 | 0.14 | 0.35 | 0.11 | 0.19 |
| 40 | 0.13 | 0.11 | 0.12 | 0.10 | 0.09 |
| 80 | 0.13 | 0.10 | 0.22 | 0.11 | 0.06 |
</details>

Figure 27: Comparison between RMSE and NMSE score differences. Figures(a) and (c) show the score field approximation errors measured with RMSE loss while figures(b) and (d) show these errors measured using NMSE loss. Compared to RMSE, the NMSE metric highlight the score differences in the high-noise regime, where diffusion models receive the least training.

# H Discussion on Geometry-Adaptive Harmonic Bases

# H.1 GAHB only Partially Explain the Strong Generalization

Recent work $[20]$ observes that diffusion models trained on sufficiently large non-overlapping datasets (of the same class) generate nearly identical images. They explain this "strong generalization" phenomenon by analyzing bias-free deep diffusion denoisers with piecewise linear input-output

mappings:

$$
\mathcal {D} (\boldsymbol {x} _ {t}; \sigma (t)) = \nabla \mathcal {D} (\boldsymbol {x} _ {t}; \sigma (t)) \boldsymbol {x} \tag {54}
$$

$$
= \sum_ {k} \lambda_ {k} (\boldsymbol {x} _ {t}) \boldsymbol {u} _ {k} (\boldsymbol {x} _ {t}) \boldsymbol {v} _ {k} ^ {T} (\boldsymbol {x} _ {t}) \boldsymbol {x} _ {t}, \tag {55}
$$

where $\lambda_{k}(\boldsymbol{x}_{t})$ , $\boldsymbol{u}_{k}(\boldsymbol{x}_{t})$ , and $\boldsymbol{v}_{k}(\boldsymbol{x}_{t})$ represent the input-dependent singular values, left and right singular vectors of the network Jacobian $\nabla \mathcal{D}(\boldsymbol{x}_{t};\sigma (t))$ . Under this framework, strong generalization occurs when two denoisers $\mathcal{D}_1$ and $\mathcal{D}_2$ have similar Jacobians: $\nabla \mathcal{D}_1(\boldsymbol{x}_t;\sigma (t))\approx \nabla \mathcal{D}_2(\boldsymbol{x}_t;\sigma (t))$ . The authors conjecture this similarity arises from networks' inductive bias towards learning certain optimal $\nabla \mathcal{D}(\boldsymbol{x}_t;\sigma (t))$ that has sparse singular values and the singular vectors of which are the geometry-adaptive harmonic bases (GAHB)—near-optimal denoising bases that adapt to input $\boldsymbol{x}_t$ .

While $[20]$ provides valuable insights, their bias-free assumption does not reflect real-world diffusion models, which inherently contain bias terms. For feed forward ReLU networks, the denoisers are piecewise affine:

$$
\mathcal {D} (\boldsymbol {x} _ {t}; \sigma (t)) = \nabla \mathcal {D} (\boldsymbol {x} _ {t}; \sigma (t)) \boldsymbol {x} _ {t} + \boldsymbol {b} _ {\boldsymbol {x} _ {t}}, \tag {56}
$$

where $b_{x_{t}}$ is the network bias that depends on both network parameterization and the noisy input $x_{t}$ [44]. Here, similar Jacobians alone cannot explain strong generalization, as networks may differ significantly in $b_{x_{t}}$ . For more complex network architectures where even piecewise affinity fails, we consider the local linear expansion of $\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ :

$$
\mathcal {D} (\boldsymbol {x} _ {t} + \Delta \boldsymbol {x}; \sigma (t)) = \nabla \mathcal {D} (\boldsymbol {x} _ {t}; \sigma (t)) \Delta \boldsymbol {x} _ {t} + \mathcal {D} (\boldsymbol {x} _ {t}; \sigma (t)), \tag {57}
$$

which approximately holds for small perturbation $\Delta x$ . Thus, although $\nabla\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ characterizes $\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ 's local behavior around $x_{t}$ , it does not provide sufficient information on the global properties.

Our work instead examines global behavior, demonstrating that $\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ is close to $\mathcal{D}_{\mathrm{G}}(\boldsymbol{x}_{t};\sigma(t))$ —the optimal linear denoiser under the Gaussian data assumption. This implies that strong generalization partially stems from networks learning similar Gaussian structures across non-overlapping datasets of the same class. Since our linear model captures global properties but not local characteristics, it complements the local analysis in [20].

# H.2 GAHB Emerge only in Intermediate-Noise Regime

For completeness, we study the evolution of the Jacobian matrix $\nabla\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ across various noise levels $\sigma(t)$ . The results are presented in Figures 28 and 29, which reveal three distinct regimes:

- High-noise regime [10,80]. In this regime, the leading singular vectors $^{6}$ of the Jacobian matrix $\nabla \mathcal{D}(\boldsymbol{x}_t; \sigma(t))$ well align with those of the Gaussian weights (the leading principal components of the training dataset), consistent with our finding that diffusion denoisers approximate linear Gaussian denoisers in this regime. Notice that DAEs trained sufficiently on separate noise levels (Figure 29) show stronger alignment compared to vanilla diffusion models (Figure 28), which suffer from insufficient training at high noise levels.   
- Intermediate-noise regime [0.1,10]: In this regime, GAHB emerge as singular vectors of $\nabla \mathcal{D}(\boldsymbol{x}_t; \sigma(t))$ diverge from the principal components, becoming increasingly adaptive to the geometry of input image.   
- Low-noise regime [0.002,0.1]. In this regime, the leading singular vectors of $\nabla \mathcal{D}(\boldsymbol{x}_t; \sigma(t))$ show no clear patterns, consistent with our observation that diffusion denoisers approach the identical mapping, which has unconstrained singular vectors.

Notice that the leading singular vectors of $\nabla\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ are the input directions that lead to the maximum variation in denoised outputs, thus revealing meaningful information on the local properties of $\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ at $x_{t}$ . As demonstrated in Figure 30, perturbing input $x_{t}$ along these vectors at difference noise regimes leads to distinct effects on the final generated images: (i) in the high-noise regime where the leading singular vectors align with the principal components of the training dataset,

![](images/aa5d800754c96fd84dbb3eeb0dc7e253c66f7bd0be00eee7c0178525c5e06cc9.jpg)  
Figure 28: Evolution of $\nabla \mathcal{D}(\pmb{x}_t; \sigma(t))$ across varying noise levels. Figure(a) shows the generation trajectory. Figure(b) shows the correlation matrix between Jacobian singular vectors $U(\pmb{x}_t)$ and training dataset principal components $U$ . Notice that the leading singular vectors of $U(\pmb{x}_t)$ and $U$ well align in early timesteps but diverge in later timesteps. Figure(c) shows the first three principal components of the training dataset while figures(d-f) show the evolution of Jacobian's first three singular vectors across noise levels. These singular vectors initially match the principal components but progressively adapt to input image geometry, before losing distinct patterns at very low noise levels. While we present only left singular vectors, right singular vectors exhibit nearly identical behavior and yield equivalent results.

perturbing $x_{t}$ along these directions leads to canonical changes such as image class, (ii) in the intermediate-noise regime where the GAHB emerge, perturbing $x_{t}$ along the leading singular vectors modify image details such as colors while preserving overall image structure and (iii) in the low-noise regime where the leading singular vectors have no significant pattern, perturbing $x_{t}$ along these directions yield no meaningful semantic changes.

These results collectively demonstrate that the singular vectors of the network Jacobian $\nabla\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ have distinct properties at different noise regimes, with GAHB emerging specifically in the intermediate regime. This characterization has significant implications for uncertainty quantification [45] and image editing [46].

![](images/17ba8fd114622c06534516f12db3aad7d0f5b5475dd63235addb49d7fde4b171.jpg)

Figure 29: Evolution of $\nabla\mathcal{D}(\boldsymbol{x}_{t};\sigma(t))$ across varying noise levels for DAEs. We repeat the experiments in Figure 28 on DAEs that are sufficiently trained on each discrete noise levels. Notice that with sufficient training, the Jacobian singular vectors $U(\boldsymbol{x}_{t})$ show a better alignment with principal components U in early timesteps.   
![](images/1fd8b7d82c7f1771e5077ad4eb7a54a1b76a4e9bdb208d18fd56acf01ea89a68.jpg)

<details>
<summary>text_image</summary>

σ(t) = 22.79
-u₁(xₜ) +u₁(xₜ)
</details>

(a)

![](images/a7b8ef7a94d7ccd352a24b6a93265dad03d2244f73c9de18ac8a28d5e216d3dc.jpg)

<details>
<summary>text_image</summary>

σ(t) = 1.979
-u₁(xₜ) + u₁(xₜ)
</details>

(b)

![](images/cd284d74824e11ea3f92ea860f13f8ba611eb65d9f4ecc844ca703b24d201b73.jpg)

<details>
<summary>text_image</summary>

σ(t) = 0.002
-u₁(xₜ) +u₁(xₜ)
</details>

(c)   
Figure 30: Effects of perturbing $x_{t}$ along Jacobian singular vectors. Figure(a)-(c) demonstrate the effects of perturbing input $x_{t}$ along the first singular vector of the Jacobian matrix $(\boldsymbol{x}_{t} \pm \lambda \boldsymbol{u}_{1}(\boldsymbol{x}_{t}))$ on the final generated images. Perturbing $x_{t}$ in high-noise regime (Figure (a)) leads to canonical image changes while perturbation in intermediate-noise regime (Figure (b)) leads to change in details but the overall image structure is preserved. At very low noise variances, perturbation has no significant effect (Figure (c)). Similar effects are observed in concurrent work [46].

# I Computing Resources

All the diffusion models in the experiments are trained on A100 GPUs provided by NCSA Delta GPU [33].