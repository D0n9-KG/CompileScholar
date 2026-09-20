# Multivariate Probabilistic Time Series Forecasting with Correlated Errors

Vincent Zhihao Zheng

McGill University

Montréal, QC, Canada

zhihao.zheng@mail.mcgill.ca

Lijun Sun\*

McGill University

Montréal, QC, Canada

lijun.sun@mcgill.ca

# Abstract

Accurately modeling the correlation structure of errors is critical for reliable uncertainty quantification in probabilistic time series forecasting. While recent deep learning models for multivariate time series have developed efficient parameterizations for time-varying contemporaneous covariance, but they often assume temporal independence of errors for simplicity. However, real-world data often exhibit significant error autocorrelation and cross-lag correlation due to factors such as missing covariates. In this paper, we introduce a plug-and-play method that learns the covariance structure of errors over multiple steps for autoregressive models with Gaussian-distributed errors. To ensure scalable inference and computational efficiency, we model the contemporaneous covariance using a low-rank-plus-diagonal parameterization and capture cross-covariance through a group of independent latent temporal processes. The learned covariance matrix is then used to calibrate predictions based on observed residuals. We evaluate our method on probabilistic models built on RNNs and Transformer architectures, and the results confirm the effectiveness of our approach in improving predictive accuracy and uncertainty quantification without significantly increasing the parameter size.

# 1 Introduction

Uncertainty quantification is crucial in time series forecasting, especially for applications that need more detailed insights than point forecasts. Probabilistic time series forecasting with deep learning (DL) has attracted attention for its ability to capture complex, nonlinear dependencies and provide the probability distribution of target variables [1, 2]. In multivariate time series, autoregressive models are widely used for probabilistic forecasting [3–5], modeling the joint one-step-ahead predictive distribution and generating multistep-ahead predictions in a rolling manner. To enable scalable learning, these models often assume that errors are independent over time. Typically, time series variables follow a Gaussian distribution $\mathbf{z}_{t}=m(\mathbf{h}_{t})+\boldsymbol{\eta}_{t}$ , where $m(\cdot)$ is the mean function and $\boldsymbol{\eta}_{t}\sim\mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}_{t})$ is a stochastic error process with contemporaneous covariance matrix $\Sigma_{t}$ . The assumption of time-independence implies $\operatorname{Cov}(\boldsymbol{\eta}_{s},\boldsymbol{\eta}_{t})=\mathbf{0},\forall s\neq t$ . This holds when the model can account for all correlations between successive time steps through hidden states determined by previous values. However, real-world data often violate this assumption, as residuals exhibit substantial cross-correlation due to omission of important covariates and model misspecification.

Modeling error autocorrelation (or cross-correlation) is a key area of research in statistical time series models. A common approach is to assume that the error series follows a dependent temporal process, such as an autoregressive integrated moving average (ARIMA) model $[6]$ . Deep learning models face similar challenges. Previous studies have attempted to incorporate temporally-correlated errors into the training process by modifying the loss function $[7, 8]$ . However, these methods, based on

deterministic output, are not easily applicable to probabilistic forecasting models, particularly in a multivariate setting. A notable innovation is the batch training method introduced by Zheng et al. [9], which trains a univariate probabilistic forecasting model using generalized least squares (GLS) loss over batched errors. This approach parameterizes a dynamic covariance matrix to capture error autocorrelation, which is then used to calibrate the predictive distribution of time series variables. While this method consistently improves probabilistic forecasting performance compared to naive training (i.e., without considering autocorrelated errors); however, it is only applicable to univariate models, such as DeepAR [10].

In this paper, we introduce an efficient method for learning error cross-correlation in multivariate probabilistic forecasting models. Our focus is on deep learning models that are autoregressive with Gaussian-distributed errors. Modeling cross-correlation in multivariate models presents challenges due to increased dimensionality, as the covariance matrix scales with the number of time series N. To address this computational challenge, we propose characterizing error cross-correlation through a set of independent latent temporal processes using a low-rank parameterization of the covariance matrix. This approach prevents the computational cost from growing with the number of time series. Our method offers a general-purpose approach to multivariate probabilistic forecasting models, offering significantly improved predictive accuracy.

# Contributions:

1. We introduce a plug-and-play method for training autoregressive multivariate probabilistic forecasting models using a redesigned GLS loss. (§4)   
2. We propose an efficient parameterization of the error covariance matrix across multiple steps, enabling efficient computation of its inverse and determinant through matrix inversion and determinant lemmas. (§4.1)   
3. The learned covariance matrix is used to fine-tune the predictive distribution based on observed residuals. (§4.2)   
4. We demonstrate that the proposed method effectively captures error cross-correlation and improves prediction quality. Notably, these improvements are achieved through a statistical formulation without significantly increasing the size of model parameters. (§5)

# 2 Probabilistic Time Series Forecasting

Denote $z_{t} = [z_{1,t}, \ldots, z_{N,t}]^{\top} \in R^{N}$ as the vector of time series variables at time step t, where N is the number of time series. Probabilistic time series forecasting can be formulated as estimating the joint conditional distribution $p(\mathbf{z}_{T+1:T+Q} \mid \mathbf{z}_{T-P+1:T}; \mathbf{x}_{T-P+1:T+Q})$ given the observed history $\{z_{t}\}_{t=1}^{T}$ , where $z_{t_{1}:t_{2}} = [z_{t_{1}}, \ldots, z_{t_{2}}]$ and $x_{t}$ are known time-dependent covariates (e.g., time of day, day of week) for all future time steps. In essence, the problem involves predicting the time series values for Q future time steps using all available covariates and P steps of historical time series data:

$$
p \left(\mathbf {z} _ {T + 1: T + Q} \mid \mathbf {z} _ {T - P + 1: T}; \mathbf {x} _ {T - P + 1: T + Q}\right) = \prod_ {t = T + 1} ^ {T + Q} p \left(\mathbf {z} _ {t} \mid \mathbf {z} _ {t - P: t - 1}; \mathbf {x} _ {t - P: t}\right), \tag {1}
$$

which becomes an autoregressive model that can be used for either one-step-ahead $(Q = 1)$ or multistep-ahead forecasting in a rolling manner. When performing multistep-ahead forecasting, samples are drawn in the prediction range $(t \geq T + 1)$ and fed back for the next time step until the end of the desired prediction range. In neural networks, the conditioning information is commonly encoded into a state vector $h_{t}$ . Hence, Eq. (1) can be expressed more concisely:

$$
p \left(\mathbf {z} _ {T + 1: T + Q} \mid \mathbf {z} _ {T - P + 1: T}; \mathbf {x} _ {T - P + 1: T + Q}\right) = \prod_ {t = T + 1} ^ {T + Q} p \left(\mathbf {z} _ {t} \mid \mathbf {h} _ {t}\right), \tag {2}
$$

where $h_{t}$ is mapped to the parameters of a parametric distribution (e.g., multivariate Gaussian).

Existing autoregressive models typically assume that the error at each time step is independent, meaning that $z_{t}$ follows a multivariate Gaussian distribution:

$$
\mathbf {z} _ {t} \mid \mathbf {h} _ {t} \sim \mathcal {N} (\boldsymbol {\mu} (\mathbf {h} _ {t}), \boldsymbol {\Sigma} (\mathbf {h} _ {t})), \tag {3}
$$

where $\boldsymbol{\mu}(\cdot)$ and $\boldsymbol{\Sigma}(\cdot)$ map $h_{t}$ to the mean and covariance parameters of a multivariate Gaussian distribution. This formulation can be decomposed as $z_{t} = \mu_{t} + \eta_{t}$ with $\eta_{t} \sim \mathcal{N}(\mathbf{0}, \Sigma_{t})$ . The temporally independent error assumption corresponds to $\operatorname{Cov}(\boldsymbol{\eta}_{s}, \boldsymbol{\eta}_{t}) = \mathbf{0}$ for any time points s and

![](images/a889e230cc54fba82d6309dd47fa9afdda837411a5ccfaf7d1f2719804acb0c5.jpg)

<details>
<summary>heatmap</summary>

| X Range | Y Range | Color Intensity |
|---------|---------|-----------------|
| 0       | 0       | 0.6             |
| 0       | 5       | 0.4             |
| 0       | 10      | 0.2             |
| 0       | 15      | 0.0             |
| 5       | 0       | 0.6             |
| 5       | 5       | 0.4             |
| 5       | 10      | 0.2             |
| 5       | 15      | 0.0             |
| 10      | 0       | 0.6             |
| 10      | 5       | 0.4             |
| 10      | 10      | 0.2             |
| 10      | 15      | 0.0             |
| 15      | 0       | 0.6             |
| 15      | 5       | 0.4             |
| 15      | 10      | 0.2             |
| 15      | 15      | 0.0             |
| 20      | 0       | 0.6             |
| 20      | 5       | 0.4             |
| 20      | 10      | 0.2             |
| 20      | 15      | 0.0             |
| 25      | 0       | 0.6             |
| 25      | 5       | 0.4             |
| 25      | 10      | 0.2             |
| 25      | 15      | 0.0             |
| 30      | 0       | 0.6             |
| 30      | 5       | 0.4             |
| 30      | 10      | 0.2             |
| 30      | 15      | 0.0             |
| 35      | 0       | 0.6             |
| 35      | 5       | 0.4             |
| 35      | 10      | 0.2             |
| 35      | 15      | 0.0             |
| 40      | 0       | 0.6             |
| 40      | 5       | 0.4             |
| 40      | 10      | 0.2             |
| 40      | 15      | 0.0             |
| 45      | 0       | 0.6             |
| 45      | 5       | 0.4             |
| 45      | 10      | 0.2             |
| 45      | 15      | 0.0             |
| 50      | 0       | 0.6             |
| 50      | 5       | 0.4             |
| 50      | 10      | 0.2             |
| 50      | 15      | 0.0             |
| 55      | 0       | 0.6             |
| 55      | 5       | 0.4             |
| 55      | 10      | 0.2             |
| 55      | 15      | 0.0             |
| 60      | 0       | 0.6             |
| 60      | 5       | 0.4             |
| 60      | 10      | 0.2             |
| 60      | 15      | 0.0             |
| 65      | 0       | 0.6             |
| 65      | 5       | 0.4             |
| 65      | 10      | 0.2             |
| 65      | 15      | 0.0             |
| 70      | 0       | 0.6             |
| 70      | 5       | 0.4             |
| 70      | 10      | 0.2             |
| 70      | 15      | 0.0             |
| 75      | 0       | 0.6             |
| 75      | 5       | 0.4             |
| 75      | 10      | 0.2             |
| 75      | 15      | 0.0             |
| 80      | 0       | 0.6             |
| 80      | 5       | 0.4             |
| 80      | 10      | 0.2             |
| 80      | 15      | 0.0             |
| 85      | 0       | 0.6             |
| 85      | 5       | 0.4             |
| 85      | 10      | 0.2             |
| 85      | 15      | 0.0             |
| 90      | 0       | 0.6             |
| 90      | 5       | 0.4             |
| 90      | 10      | 0.2             |
| 90      | 15      | 0.0             |
| 95      | 0       | 0.6             |
| 95      | 5       | 0.4             |
| 95      | 10      | 0.2             |
| 95      | 15      | -                |
| Note: The heatmap values are not explicitly provided in the code, so they are estimated based on the provided code format.
</details>

Figure 1: Contemporaneous covariance matrix $\mathrm{Cov}(\pmb{\eta}_t,\pmb{\eta}_t)$ and cross-covariance matrix $\mathrm{Cov}(\pmb{\eta}_{t - \Delta},\pmb{\eta}_t),\Delta = 1,2,3$ , calculated based on the one-step-ahead prediction residuals of GP-Var on a batch of time series from the m4\_hourly dataset. For visualization clarity, covariance are clipped to the range [0, 0.6].

t where $s \neq t$ . Fig. 1 provides an empirical example of the contemporaneous covariance matrix $\operatorname{Cov}(\boldsymbol{\eta}_{t}, \boldsymbol{\eta}_{t})$ and cross-covariance matrix $\operatorname{Cov}(\boldsymbol{\eta}_{t-\Delta}, \boldsymbol{\eta}_{t}), \Delta = 1, 2, 3$ . The results are calculated based on the prediction residuals of GPVar [3] on the m4\_hourly dataset. While multivariate models primarily focus on contemporaneous covariance, the residuals clearly exhibit temporal dependence, as $\operatorname{Cov}(\boldsymbol{\eta}_{t-\Delta}, \boldsymbol{\eta}_{t}) \neq \mathbf{0}$ . This non-zero cross-covariance suggests that residuals still contain valuable information, which can be leveraged to improve predictions.

# 3 Related Work

# 3.1 Probabilistic Time Series Forecasting

Probabilistic forecasting aims to model the probability distribution of target variables, unlike deterministic forecasting, which produces only point estimates. There are two main approaches: parametric probability density functions (PDFs) and quantile functions $[2]$ . For example, MQ-RNN $[11]$ generates quantile forecasts using a sequence-to-sequence (Seq2Seq) RNN architecture. In contrast, PDF-based approaches assume a specific distribution (e.g., Gaussian, Poisson) and use neural networks to generate the distribution parameters. DeepAR $[10]$ , for instance, uses an RNN to model hidden state transitions, while its multivariate version, GPVar $[3]$ , employs a Gaussian copula to transform observations into Gaussian variables, assuming a joint multivariate Gaussian distribution.

Neural networks can also generate probabilistic model parameters. The deep state space model (SSM) [12] uses an RNN to generate SSM parameters. The normalizing Kalman filter (NKF) [13] combines normalizing flows (NFs) with the linear Gaussian state space model (LGM) to model nonlinear dynamics and evaluate the PDF of observations. NKF uses RNNs to produce LGM parameters at each time step, then transforms the LGM output into observations using NFs. Wang et al. [14] proposed the deep factor model, which includes a deterministic global component parameterized by an RNN and a random component from any classical probabilistic model (e.g., Gaussian white noise) to represent random effects. Some methods improve expressive conditioning for probabilistic forecasting by using Transformer instead of RNNs to model latent state dynamics, thus breaking the Markovian assumption in RNNs [15]. Other approaches adopt more flexible distribution forms, including normalizing flows [4], diffusion models [5], and copulas [16, 17]. For a recent and comprehensive review, we refer readers to Benidis et al. [2].

# 3.2 Modeling Correlated Errors

Error correlation in time series has been extensively studied in econometrics and statistics [18, 6, 19]. In multivariate time series, correlation structure is characterized by contemporaneous covariance $\mathrm{Var}(\pmb{\eta}_t) = \mathrm{Cov}(\pmb{\eta}_t,\pmb{\eta}_t)$ and cross-covariance $\mathrm{Cov}(\pmb{\eta}_{t - \Delta},\pmb{\eta}_t)$ . Cross-covariance includes both the autocovariance of errors $\mathrm{Cov}(\eta_{i,t - \Delta},\eta_{i,t})$ and the cross-lag covariance $\mathrm{Cov}(\eta_{i,t - \Delta},\eta_{j,t})$ between pairs of components in the multivariate series. Contemporaneous covariance captures the correlation among individual time series at a specific point in time. In the univariate setting, DeepAR [10] achieves probabilistic forecasting by modeling the contemporaneous covariance, assuming that errors are independent over time. To address autocorrelation, Sun et al. [7] re-parameterized the input and output of neural networks to model first-order error autocorrelation, effectively capturing serially

correlated errors using an AR(1) process. This method improves the performance of one-step-ahead neural forecasting models, allowing joint optimization of base and error regressors, but is limited to deterministic models. In spatial modeling, Saha et al. [20] introduced the RF-GLS model, which uses random forests to estimate nonlinear covariate effects and Gaussian processes (GP) to model spatial random effects. The RF-GLS model assumes that the error process follows an AR(p) process to accommodate autocorrelated errors. Zheng et al. [9] proposed training a probabilistic forecasting model with a GLS loss that explicitly models the time-varying autocorrelation of batched error terms, extending DeepAR to incorporate autocorrelated errors.

In the multivariate setting, most existing work focuses on modeling contemporaneous covariance, assuming that $\eta_{t}$ is independently distributed, which implies $\mathrm{Cov}(\boldsymbol{\eta}_{t-\Delta},\boldsymbol{\eta}_{t})=\mathbf{0}$ . For example, GPVar [3] generalizes DeepAR [10] to account for correlations between time series by viewing the distribution of time series variables as a Gaussian process. In Seq2Seq models, correlations can span across series and forecasting steps, as predictions for future time steps are generated simultaneously. Since predictions for future time steps are generated simultaneously, we refer to these correlations as contemporaneous correlations within the scope of this study. Choi et al. [21] introduced a dynamic mixture of matrix Gaussian distributions to capture contemporaneous covariance of errors in Seq2Seq models. One exception that explicitly models error cross-correlation is [8], where the authors assume that the matrix-variate error term of a multivariate Seq2Seq model follows a matrix autoregressive (AR) process with seasonal lags. However, applying this technique to probabilistic forecasting models is not straightforward.

To the best of our knowledge, our work is the first to model cross-covariance in multivariate probabilistic time series forecasting. The closest related studies are by Zheng et al. [9] and Zheng et al. [8]. Zheng et al. [9] applies GLS loss in the temporal domain to model autocorrelated errors, but their approach is tailored for univariate time series. Zheng et al. [8] models cross-covariance in multivariate forecasting models, but their method is limited to deterministic models and requires predefined seasonal lags in the error autoregressive process. Our work extends [9] to the multivariate setting, enabling the modeling of the correlation structure of multivariate errors across multiple steps. In addition, we distinguish our approach from methods that directly model the distribution of time series variables, such as Copulas [16, 17], where no decomposition of the error term is provided.

# 4 Our Method

Our methodology builds upon the formulation outlined in Eq. (2), employing an autoregressive model as its foundational framework. Using an RNN as an example, a probabilistic forecasting model consists of two components. Firstly, it incorporates a transition model $f_{\Theta}$ to capture the dynamics of state transitions $\mathbf{h}_{t}=f_{\Theta}\left(\mathbf{h}_{t-1},\mathbf{z}_{t-1},\mathbf{x}_{t}\right)$ , thus inherently having autoregressive properties. Second, it integrates a distribution head, represented by $\theta$ , which maps $h_{t}$ to the parameters of the desired probability distribution. Following GPVar [3], our approach employs the multivariate Gaussian distribution as the distribution head. The time series variable can be decomposed into a deterministic mean component and a random error component $z_{t}=\mu_{t}+\eta_{t}$ , where $\boldsymbol{\eta}_{t}\sim\mathcal{N}(\mathbf{0},\mathbf{\Sigma}_{t})$ . To efficiently model the covariance $\Sigma_{t}$ for large N, GPVar adopts a low-rank-plus-diagonal parameterization $\Sigma_{t}=L_{t}L_{t}^{\top}+\mathrm{diag}(\mathbf{d}_{t})$ , where $L_{t}\in R^{N\times R}(R\ll N)$ and $d_{t}\in R_{+}^{N}$ . Autoregressive models based on Gaussian likelihood typically assume that $\eta_{t}$ are independently distributed following a multivariate Gaussian distribution. The log-likelihood of the distribution serves as the loss function for optimizing the model:

$$
\mathcal {L} = \sum_ {t = 1} ^ {T} \log p (\mathbf {z} _ {t} \mid \theta (\mathbf {h} _ {t})) \propto \sum_ {t = 1} ^ {T} - \frac {1}{2} [ \ln | \boldsymbol {\Sigma} _ {t} | + \boldsymbol {\eta} _ {t} ^ {\top} \boldsymbol {\Sigma} _ {t} ^ {- 1} \boldsymbol {\eta} _ {t} ]. \tag {4}
$$

The parameters $\theta(\mathbf{h}_{t})$ are parameterized as $(\boldsymbol{\mu}_{t},\boldsymbol{L}_{t},\mathbf{d}_{t})$ , where $\mu_{t}\in R^{N}$ represents the mean vector of the distribution. $L_{t}$ and $d_{t}$ correspond to the covariance factor and diagonal elements in the low-rank parameterization of the multivariate Gaussian distribution. We use shared mapping functions for all time series:

$$
\mu_ {i} (\mathbf {h} _ {i, t}) = \tilde {\mu} (\mathbf {h} _ {i, t}) = \mathbf {w} _ {\mu} ^ {\top} \mathbf {h} _ {i, t},
$$

$$
d _ {i} (\mathbf {h} _ {i, t}) = \tilde {d} (\mathbf {h} _ {i, t}) = \log (1 + \exp (\mathbf {w} _ {d} ^ {\top} \mathbf {h} _ {i, t})), \tag {5}
$$

$$
l _ {i} (\mathbf {h} _ {i, t}) = \tilde {l} (\mathbf {h} _ {i, t}) = W _ {l} \mathbf {h} _ {i, t},
$$

![](images/8dc47a745bf542317e9f8411954e3e9f6791f6b7b30344b97352bc2d14f9e431.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Cross-correlation window D"] --> B["B×P"]
    B --> C["B"]
    C --> D["z_{t-D+1}"]
    D --> E["z_t"]
    E --> F["z_t^bat"]
    F --> G["μ_t^bat"]
    G --> H["R×D"]
    H --> I["L_t^bat"]
    I --> J["ε_t^bat"]
    J --> K["r_t^bat"]
    K --> L["..."]
    L --> M["r_{t-D+1:t}"]
    N["D = Q"] --> O["B×P"]
    O --> P["B×Q"]
    P --> Q["A sample slice of time series"]
    style A fill:#cce5ff,stroke:#333
    style H fill:#ffcccc,stroke:#333
    style J fill:#ffcccc,stroke:#333
    style K fill:#ffcccc,stroke:#333
    style L fill:#ffcccc,stroke:#333
    style M fill:#ffcccc,stroke:#333
    style N fill:#cce5ff,stroke:#333
    style O fill:#cce5ff,stroke:#333
    style P fill:#cce5ff,stroke:#333
    style Q fill:#cce5ff,stroke:#333
    style R fill:#ffcccc,stroke:#333
    style S fill:#ffcccc,stroke:#333
    style T fill:#ffcccc,stroke:#333
    style U fill:#ffcccc,stroke:#333
    style V fill:#ffcccc,stroke:#333
    style W fill:#ffcccc,stroke:#333
    style X fill:#ffcccc,stroke:#333
    style Y fill:#ffcccc,stroke:#333
    style Z fill:#ffcccc,stroke:#333
```
</details>

Figure 2: Graphic illustration of Eq. (8), where B is the number of time series in a batch, R is the rank of the covariance factor, D is the time window we consider cross-correlation, P and Q are the conditioning range and prediction range. Cross-correlation is modeled by introducing correlation in each row of matrix $r_{t-D+1:t}$ .

where $h_{i,t} \in R^{H}$ , $w_{\mu} \in R^{H}$ , $w_{d} \in R^{H}$ , and $W_{l} \in R^{R \times H}$ are parameters. Since the parameters of the mapping functions are shared across all time series, we can use a random subset of time series to compute the Gaussian likelihood-based loss in each optimization step, as any subset of $z_{t}$ will still follow a multivariate Gaussian distribution. In other words, we can train the model with a substantially reduced batch size B < N.

# 4.1 Training with Correlated Errors

We build upon the approach introduced in [9] to address cross-correlated errors in a multivariate context by introducing time-dependent error terms $\eta_{t}$ into the GLS loss. In many existing deep probabilistic forecasting models, such as GPVar [3], a training batch typically consists of a sample slice of B time series spanning a temporal length of $P + Q$ , where P is the conditioning range and Q is the prediction range. The Gaussian likelihood is evaluated independently at each time step within the prediction range through one-step-ahead predictions. However, this approach overlooks the serial correlation of errors across consecutive time steps. To address this limitation, we propose modifying the likelihood function by introducing a dynamic covariance that accommodates the temporal dependence of the error term, as illustrated in Fig. 2. To achieve this, we organize D smaller slices of time series with a temporal length of $P + 1$ (i.e., Q = 1), sorted by the prediction start time in sequential order, where D represents the time horizon over which we consider cross-correlation. The new batch structure effectively reconstructs the conventional training batch, covering the same time horizon when D = Q. An example of the collection of target time series variables in a batch covering cross-correlation horizon D is given by

$$
\mathbf {z} _ {t - D + 1} = \boldsymbol {\mu} _ {t - D + 1} + \boldsymbol {\eta} _ {t - D + 1},
$$

$$
\mathbf {z} _ {t - D + 2} = \boldsymbol {\mu} _ {t - D + 2} + \boldsymbol {\eta} _ {t - D + 2}, \tag {6}
$$

...

$$
\mathbf {z} _ {t} = \boldsymbol {\mu} _ {t} + \boldsymbol {\eta} _ {t},
$$

where for time point $t'$ , $\mu_{t'}$ , $L_{t'}$ and $d_{t'}$ are the outputs of the model. The covariance parameterization in GPVar corresponds to

$$
\boldsymbol {\eta} _ {t ^ {\prime}} = \boldsymbol {L} _ {t ^ {\prime}} \mathbf {r} _ {t ^ {\prime}} + \varepsilon_ {t ^ {\prime}}, \tag {7}
$$

where $\mathbf{r}_{t'} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_R)$ is a low-dimensional latent variable, and $\varepsilon_{t'} \sim \mathcal{N}(\mathbf{0}, \mathrm{diag}(\mathbf{d}_{t'}))$ is an additional error independent of $\mathbf{r}_{t'}$ . We denote $z_t^{\mathrm{bat}} = \mathrm{vec}(\mathbf{z}_{t-D+1:t}) \in \mathbb{R}^{DB}$ as the collection of target time series variables in a batch, where $\mathrm{vec}(\cdot)$ is an operator that stacks all the columns of a matrix into a vector. Similarly, we define $\mu_t^{\mathrm{bat}} \in \mathbb{R}^{DB}$ , $r_t^{\mathrm{bat}} \in \mathbb{R}^{DR}$ , $\varepsilon_t^{\mathrm{bat}} \in \mathbb{R}^{DB}$ , $d_t^{\mathrm{bat}} \in \mathbb{R}_+^{DB}$ , and $L_t^{\mathrm{bat}} = \mathrm{blkdiag}(\{L_{t'}\}_{t'=t-D+1}^t) \in \mathbb{R}^{DB \times DR}$ , where $L_t^{\mathrm{bat}}$ has a block diagonal structure (see Fig. 2). The batch-wise decomposition is then expressed as

$$
\boldsymbol {z} _ {t} ^ {\text { bat }} = \boldsymbol {\mu} _ {t} ^ {\text { bat }} + \boldsymbol {L} _ {t} ^ {\text { bat }} \boldsymbol {r} _ {t} ^ {\text { bat }} + \varepsilon_ {t} ^ {\text { bat }}. \tag {8}
$$

![](images/b0c11e4ead0a88ed936a7b3033d3299845a040f117b1ce3336638c0fd4b3b0e2.jpg)

<details>
<summary>text_image</summary>

log p (z_t^bat_1 | μ_t^bat_1, Σ_t^bat_1)
log p (z_t^bat_2 | μ_t^bat_2, Σ_t^bat_2)
z_{1,t} → h_{1,t-D+1:t}
z_{2,t} → h_{2,t-D+1:t}
z_{4,t} → h_{4,t-D+1:t}
</details>

Figure 3: Illustration of the training process. Following [3], time series dimensions are randomly sampled, and the base model (e.g., RNNs) is unrolled for each dimension individually (e.g., 1, 2, 4, followed by 1, 3, 4 as depicted). The model parameters are shared across all time series dimensions. A batch of time series variables $z_{t}^{bat}$ contains time series vectors $z_{t}$ covering time steps from $t - D + 1$ to t. In contrast to [3], our approach explicitly models dependencies over the extended temporal window from $t - D + 1$ to t during training.

The default GPVar model assumes the latent variable $r_{t}$ is temporally independent, meaning $\operatorname{Cov}\left(\boldsymbol{r}_{s},\boldsymbol{r}_{t}\right)=\mathbf{0},\forall s\neq t$ . However, this assumption cannot capture the potential cross-correlation in the errors. To address this, we introduce temporal dependencies in the latent variable within a batch by assuming $r_{t}^{bat}\sim\mathcal{N}\left(\mathbf{0},\boldsymbol{C}_{t}\otimes\mathbf{I}_{R}\right)$ , where $C_{t}$ is a dynamic $D\times D$ correlation matrix. This approach assumes that the rows in the matrix $r_{t-D+1:t}=\left[r_{t-D+1},\ldots,r_{t}\right]$ are independent and identically distributed, following $\mathcal{N}\left(\mathbf{0},\boldsymbol{C}_{t}\right)$ . To efficiently capture dynamic patterns over time, we follow Zheng et al. [9] and express $C_{t}$ as a dynamic weighted sum base kernel matrices: $C_{t}=\sum_{m=1}^{M}w_{m,t}K_{m}$ , where $w_{m,t}\geq0$ (with $\sum_{m}w_{m,t}=1$ ) represents the weights for each component. For simplicity, we model each component $K_{m}$ using a kernel matrix generated from a squared-exponential (SE) kernel function, where the $(i,j)$ -th entry is $K_{m}^{ij}=\exp(-\frac{(i-j)^{2}}{l_{m}^{2}})$ , with different lengthscales $l_{m}$ (e.g., $l=1,2,3,\ldots$ ). In addition, we incorporate an identity matrix into the additive structure to account for the independent noise process. This parameterization ensures that $C_{t}$ is a positive definite symmetric matrix with unit diagonals, making it a valid correlation matrix. The weights for these components are derived from the hidden state $h_{t}$ at each time step through a small neural network, with the number of nodes in the output layer set to M (i.e., the number of components). A softmax layer is used to ensure that these weights are summed up to 1. Note that the parameters of this network will be learned simultaneously with those of the base model.

Marginalizing out $r_t^{\mathrm{bat}}$ in Eq. (8), we have $z_t^{\mathrm{bat}} \sim \mathcal{N}\left(\mu_t^{\mathrm{bat}}, \Sigma_t^{\mathrm{bat}}\right)$ with covariance

$$
\boldsymbol {\Sigma} _ {t} ^ {\mathrm{bat}} = \left(\boldsymbol {L} _ {t} ^ {\mathrm{bat}}\right) \left(\boldsymbol {C} _ {t} \otimes \mathbf {I} _ {R}\right) \left(\boldsymbol {L} _ {t} ^ {\mathrm{bat}}\right) ^ {\top} + \operatorname{diag} \left(\boldsymbol {d} _ {t} ^ {\mathrm{bat}}\right). \tag {9}
$$

It is straightforward to derive that for any $i, j \in \{0, 1, \ldots, D - 1\}$ and $i \neq j$ , the proposed model creates cross-covariance $\operatorname{Cov}\left(\boldsymbol{\eta}_{t-i}, \boldsymbol{\eta}_{t-j}\right) = \boldsymbol{C}_{t}^{ij} \boldsymbol{L}_{t-i} \boldsymbol{L}_{t-j}^{\top}$ between times t - i and t - j, which is no longer 0. While this parameterization results in a non-stationary multivariate process through varying coregionalization [22, 23], a key difference is that both the coregionalization coefficient matrix $L_{t}$ and the temporal correlation $C_{t}$ are generated by a deep neural network. In this sense, our model can better characterize the empirical cross-covariance matrices of the residuals (see empirical examples in Fig. 1). As $\mu_{t}^{bat}$ , $L_{t}^{bat}$ , and $d_{t}^{bat}$ are default outputs of the base probabilistic model, we can compute the overall likelihood (with overlapped data) as

$$
\mathcal {L} = \sum_ {t = D} ^ {T} \log p \left(\boldsymbol {z} _ {t} ^ {\mathrm{bat}} \mid \boldsymbol {\mu} _ {t} ^ {\mathrm{bat}}, \boldsymbol {\Sigma} _ {t} ^ {\mathrm{bat}}\right). \tag {10}
$$

Here, computing the log-likelihood involves evaluating the inverse and the determinant of $\Sigma_t^{\mathrm{bat}}$ with size $DB\times DB$ , for which a naive implementation has a prohibitive time complexity of $\mathcal{O}\left(D^3 B^3\right)$ . However, our parameterization of $\Sigma_t^{\mathrm{bat}}$ as $E + ACA^\top$ , where $E = \mathrm{diag}(d_t^{\mathrm{bat}})$ , $A = L_t^{\mathrm{bat}}$ , and $C = C_t\otimes I_R$ , allows us to leverage the Sherman-Morrison-Woodbury identity (matrix inversion lemma) and the companion matrix determinant lemma to simplify the computation:

$$
\left(\boldsymbol {E} + \boldsymbol {A} \boldsymbol {C} \boldsymbol {A} ^ {\top}\right) ^ {- 1} = \boldsymbol {E} ^ {- 1} - \boldsymbol {E} ^ {- 1} \boldsymbol {A} \left(\boldsymbol {C} ^ {- 1} + \boldsymbol {A} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {A}\right) ^ {- 1} \boldsymbol {A} ^ {\top} \boldsymbol {E} ^ {- 1}, \tag {11}
$$

$$
\det \left(\boldsymbol {E} + \boldsymbol {A C A} ^ {\top}\right) = \det \left(\boldsymbol {C} ^ {- 1} + \boldsymbol {A} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {A}\right) \det \left(\boldsymbol {C}\right) \det \left(\boldsymbol {E}\right).
$$

Then, the likelihood calculation only requires computing the inverse and determinant of a $DR \times DR$ matrix, specifically $C^{-1} + A^{\top}E^{-1}A$ . These computations can be efficiently performed using Cholesky factorization. Detailed computations are provided in Appendix §A.2.

Modeling the latent process $r_{t}$ offers several advantages. Firstly, because $r_{t}$ has a much lower dimension than $\varepsilon_{t}$ , modeling the cross-correlation of $r_{t}$ results in a significantly smaller $DR \times DR$ covariance matrix compared to the $DB \times DB$ covariance matrix of $\varepsilon_{t}$ . Secondly, since $r_{t}$ follows an isotropic Gaussian distribution, the covariance of $r_{t}^{bat}$ can be parameterized with a Kronecker structure $C_{t} \otimes I_{R}$ . This greatly simplifies the task into learning a $D \times D$ correlation matrix shared by all time series in a batch. Lastly, similar to GPVar, we can still train the model in an end-to-end manner using a subset of time series in each iteration to ensure computational efficiency (Fig. 3).

# 4.2 Multistep-ahead Rolling Prediction

Autoregressive models perform multistep-ahead forecasting in an iterative manner, where the model generates a sample at each time step during prediction, using it as input for the subsequent step, and continuing this process until the desired prediction range is reached. Our approach enhances this process, similar to Zheng et al. [9], by offering additional calibration based on the learned correlation matrix $C_{t}$ . Assuming observations are available up to time step t, the conditional distribution of $\eta_{t+1}$ given errors in the past $(D-1)$ steps, can be derived as

$$
\boldsymbol {\eta} _ {t + 1} \mid \boldsymbol {\eta} _ {t}, \boldsymbol {\eta} _ {t - 1}, \dots , \boldsymbol {\eta} _ {t - D + 2} \sim \mathcal {N} \left(\boldsymbol {\Sigma} _ {*} \boldsymbol {\Sigma} _ {\mathrm{obs}} ^ {- 1} \boldsymbol {\eta} _ {\mathrm{obs}}, \boldsymbol {\Sigma} _ {t + 1} - \boldsymbol {\Sigma} _ {*} \boldsymbol {\Sigma} _ {\mathrm{obs}} ^ {- 1} \boldsymbol {\Sigma} _ {*} ^ {\top}\right), \tag {12}
$$

where $\boldsymbol{\eta}_{\mathrm{obs}} = \mathrm{vec}\left(\left[\boldsymbol{\eta}_{t - D + 2},\dots ,\boldsymbol{\eta}_{t - 1},\boldsymbol{\eta}_t\right]\right)\in \mathbb{R}^{(D - 1)B}$ represents the set of residuals, accessible at forecasting step $t + 1$ . Here, $\Sigma_{\mathrm{obs}}$ is a $(D - 1)B\times (D - 1)B$ partition of $\Sigma_{t + 1}^{\mathrm{bat}}$ that captures the covariance of $\boldsymbol{\eta}_{\mathrm{obs}}$ , and $\Sigma_{*}$ is a $B\times (D - 1)B$ partition of $\Sigma_{t + 1}^{\mathrm{bat}}$ representing the covariance between $\boldsymbol{\eta}_{t + 1}$ and $\boldsymbol{\eta}_{\mathrm{obs}}$ , i.e., $\Sigma_{t + 1}^{\mathrm{bat}} = \begin{bmatrix} \Sigma_{\mathrm{obs}} & \Sigma_{*}^{\top}\\ \Sigma_{*} & \Sigma_{t + 1} \end{bmatrix}$ . For conciseness, we omit the time index $t$ in $\Sigma_{\mathrm{obs}}$ , $\Sigma_{*}$ and $\boldsymbol{\eta}_{\mathrm{obs}}$ . Since $\mu_{t + 1}$ is a deterministic output from the base model, a sample of the target variables $\tilde{\mathbf{z}}_{t + 1}$ can be derived by first drawing a sample $\tilde{\boldsymbol{\eta}}_{t + 1}$ from Eq. (12), then combining it with the predicted mean vector $\mu_{t + 1}$ as $\tilde{\mathbf{z}}_{t + 1} = \boldsymbol{\mu}_{t + 1} + \tilde{\boldsymbol{\eta}}_{t + 1}$ . It should be noted that we can still leverage the Sherman-Morrison-Woodbury identity when computing the inverse $\Sigma_{\mathrm{obs}}^{-1}$ .

By taking the sample $\tilde{\eta}_{t+1}$ as an observed residual, we can iteratively apply the process described in Eq. (12) to derive a trajectory of $\{\tilde{z}_{t+q}\}_{q=1}^{Q}$ . Repeating this procedure allows us to generate multiple samples, characterizing the predictive distribution at each time step.

# 5 Experiments

# 5.1 Evaluation of Predictive Performance

Datasets. We use widely recognized time series benchmarking datasets from GluonTS [24]. The prediction range $(Q)$ for each dataset follows the configurations provided by GluonTS. We applied a sequential split into training, validation, and testing sets for each dataset. Each dataset was standardized using the mean and standard deviation from the training set, and predictions were rescaled to their original values for evaluation. Further details on the datasets can be found in Appendix §A.1.

Base probabilistic models. We integrated the proposed method into two distinct autoregressive models: the RNN-based GPVar [3] and the decoder-only Transformer [25]. These models are trained to generate distribution parameters as described in §4. Our approach can be applied to other autoregressive multivariate models with minimal adjustments, provided the final prediction follows a multivariate Gaussian distribution. The implementation is based on using PyTorch Forecasting [26]. Both models use lagged time series values and additional features or covariates as inputs. Details on model training (§A.3), hyperparameter tuning (§A.5), and the base model (§A.6) are provided in Appendix §A. The code is available at https://github.com/rottenivy/mv\_pts\_correlatederr.

Dynamic correlation matrix. We introduce a limited number of additional parameters to project the state vector $h_{t}$ into component weights $w_{m,t}$ , which are used to generate the dynamic correlation matrix $C_{t}$ . The number of base kernels (M) for generating $C_{t}$ and the associated lengthscale set $\{l_{m}\}_{m=1}^{M-1}$ are treated as hyperparameters. We perform a grid search over M = 2, 3, 4 and two sets of lengthscales— $\{0.5, 1.5, \ldots\}$ and $\{1.0, 2.0, \ldots\}$ . Models with the best validation loss are selected. These different lengthscales capture varying correlation decay rates, enabling the model to account

Table 1: CRPS $_{sum}$ accuracy comparison. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method. Bold values show models with time-dependent errors performing better. Mean and standard deviation are obtained from 10 runs of each model. 

<table><tr><td rowspan="2"></td><td>VAR</td><td>GARCH</td><td colspan="2">GPVar</td><td colspan="2">Transformer</td></tr><tr><td></td><td></td><td>w/o</td><td>w/</td><td>w/o</td><td>w/</td></tr><tr><td>exchange_rate</td><td>0.0033±0.0000</td><td>0.0435±0.0001</td><td>0.0068±0.0004</td><td>0.0117±0.0004</td><td>0.0055±0.0002</td><td>0.0042±0.0002</td></tr><tr><td>solar</td><td>0.7663±0.0050</td><td>0.8752±0.0015</td><td>0.7103±0.0065</td><td>0.6929±0.0039</td><td>0.4960±0.0034</td><td>0.4132±0.0027</td></tr><tr><td>electricity</td><td>0.1264±0.0006</td><td>0.2847±0.0015</td><td>0.0430±0.0005</td><td>0.0403±0.0004</td><td>0.0494±0.0004</td><td>0.0638±0.0003</td></tr><tr><td>traffic</td><td>3.5241±0.0084</td><td>0.4459±0.0005</td><td>0.1095±0.0002</td><td>0.0649±0.0002</td><td>0.0717±0.0002</td><td>0.0981±0.0002</td></tr><tr><td>wikipedia</td><td>26.2025±0.0389</td><td>0.6699±0.0045</td><td>0.1745±0.0008</td><td>0.0743±0.0009</td><td>0.0841±0.0013</td><td>0.0500±0.0005</td></tr><tr><td>m4_hourly</td><td>0.2352±0.0008</td><td>0.2758±0.0006</td><td>0.0613±0.0004</td><td>0.0358±0.0002</td><td>0.0651±0.0004</td><td>0.0616±0.0003</td></tr><tr><td>m1_quarterly</td><td>N/A</td><td>N/A</td><td>0.3942±0.0030</td><td>0.3538±0.0017</td><td>0.4448±0.0027</td><td>0.4367±0.0028</td></tr><tr><td>pems03</td><td>0.0598±0.0002</td><td>0.3202±0.0007</td><td>0.0503±0.0001</td><td>0.0491±0.0002</td><td>0.0490±0.0001</td><td>0.0386±0.0001</td></tr><tr><td>uber_hourly</td><td>N/A</td><td>N/A</td><td>0.0342±0.0006</td><td>0.0222±0.0004</td><td>0.0632±0.0003</td><td>0.0513±0.0005</td></tr><tr><td></td><td></td><td></td><td>avg. rel. impr.</td><td>13.79%</td><td>avg. rel. impr.</td><td>6.91%</td></tr></table>

for different temporal patterns. The time-varying component weights enable dynamic adaptation to changing correlation structures over time.

Baselines. We evaluate the proposed method by comparing it with a baseline model trained without accounting for error cross-correlation (Eq. (4)). The baseline model represents a special case of our model with $C_{t} = I_{D}$ . To ensure a straightforward and fair comparison, we align the cross-correlation range (D) with the prediction range (Q), ensuring identical data sampling processes for both methods. Additionally, we set P = Q following the default configuration in GluonTS. We also include VAR and GARCH as naive baseline models (see Appendix §A.4).

Metrics. We use the Continuous Ranked Probability Score (CRPS) [27] as the main metric:

$$
\operatorname{CRPS} (F, z) = \mathbb {E} _ {F} | Z - z | - \frac {1}{2} \mathbb {E} _ {F} \left| Z - Z ^ {\prime} \right|, \tag {13}
$$

where $F$ is the cumulative distribution function (CDF) of the predicted variable, $z$ is the observation, $Z$ and $Z'$ are independent copies of the prediction samples associated with the distribution $F$ . To evaluate multivariate dependencies in the time series data, we compute CRPS $_{\text{sum}}$ by first summing both the forecast and ground-truth values across all time series and then calculating the CRPS over the resulting sums [3, 16, 17]. As CRPS $_{\text{sum}}$ may overlook model performance on individual dimensions [28], we also report additional metrics, e.g., the energy score [27, 29], in Appendix §B.1.

Training dynamics. Our approach incurs additional training costs per optimization step due to the more complex likelihood function. As shown in Appendix §B.3, the training time per epoch for models using our method is generally longer than that of baseline methods. However, our parameterization allows for scalability to large time series datasets by using a small random subset of time series at each optimization step during training.

Benchmark results. The CRPS $_{sum}$ results are presented in Table 1. Our method achieves an average improvement of 13.79% for GPVar and 6.91% for the Transformer model. It is important to note that the degree of performance enhancement varies across different base models and datasets, influenced by factors such as the inherent data characteristics and the performance of different model architectures. The alignment between the actual correlation structure and our kernel assumption also plays a crucial role in the effectiveness of our method. Additionally, our approach demonstrates consistent improvements across five different metrics, with significant gains in multivariate metrics such as the energy score (Appendix §B.2).

To provide further insights, we compare the residual autocorrelation and cross-lag correlation with and without applying our method in Appendix §B.5.1, showing that our method effectively reduces cross-correlations in many scenarios. We use ACF plot comparisons to illustrate the reduction in autocorrelation and cross-correlation plot comparisons to demonstrate the decrease in cross-lag correlation. The residuals generated by the model with our method exhibit weaker cross-correlations, which is particularly enhanced by the calibration process during prediction (§4.2).

Furthermore, Appendix §B.5.2 separates the accuracy improvement over forecast steps for each dataset. The performance improvement is shown to be related to both the absolute time across the

![](images/45ba94f48bb36573b6dcd2bfbf13318b3fc7493c823c4abf743286eb88f14ad0.jpg)  
Figure 4: (a) Component weights for generating $C_t$ for a batch of time series ( $B = 8$ ) from the m4\_hourly dataset obtained by the GPVar model. Parameters $w_0, w_1, w_2$ represent the component weights of the kernel matrices associated with lengthscales $l = 0.5, 1.5, 2.5$ , and $w_3$ is the component weight of the identity matrix. Shaded areas distinguish different days; (b) The autocorrelation function (ACF) indicated by the correlation matrix $C_t$ at 17:00. Given the rapid decay of the ACF, we only plot 12 lags to enhance visualization; (c) The corresponding covariance matrix of the associated target variables $\Sigma_t^{\mathrm{bat}}$ at 17:00. A zoom-in view of a $3B \times 3B$ region is illustrated in the plot, where the diagonal blocks represent $B \times B$ covariance matrices $\Sigma_{t'}$ of $\mathbf{z}_{t'}$ over three consecutive time steps. The off-diagonal blocks describe the cross-covariance $\mathrm{Cov}(\mathbf{z}_{t-\Delta}, \mathbf{z}_t), \forall \Delta \neq 0$ . For visualization clarity, covariance values are clipped to the range [0, 0.03].

temporal span of the dataset (especially for time series with strong periodic patterns) and the relative time over the prediction horizon.

# 5.2 Model Interpretation

Our method captures error cross-correlation through the dynamic construction of a covariance matrix, achieved by combining kernel matrices with varying lengthscales in a dynamically weighted sum. A small lengthscale corresponds to short-range positive correlations, while a large lengthscale captures positive correlations over longer lags.

In Fig. 4, we depict the dynamic component weights and the resulting autocorrelation function (the first row of the correlation matrix $C_t$ ) for a batch of time series from the m4\_hourly dataset spanning a four-day window. We also provide the covariance matrix of $z_t^{\mathrm{bat}}$ using the correlation matrix and model outputs at a specific time of day. The component weight $w_3$ , corresponding to the identity matrix, dominates throughout the observation period. This suggests that the error correlation is generally mild over time. This behavior is influenced by the Kronecker structure used to parameterize the covariance over the low-dimensional latent variables $r_t$ , which assumes all latent processes share the same autocorrelation structure. Given the Kronecker structure, the model tends to learn the mildest temporal correlation among the time series in a batch.

Moreover, we observe that the dynamic component weights adjust the correlation strengths. Specifically, when the weight assigned to the identity matrix $(w_{3})$ increases, the error process tends to be more independent. In contrast, when the weights assigned to the other kernel matrices $(w_{0}, w_{1}, \text{and } w_{2})$ are larger, the error process becomes more correlated, as the kernel matrices with different lengthscales combine to formulate a specific correlation structure. Fig. 4(a) demonstrates pronounced daily patterns in temporal correlation, particularly when errors exhibit increased correlation around 17:00 each day. The corresponding autocorrelation function is shown in Fig. 4(b). Fig. 4(c) illustrates the corresponding covariance matrix of the associated target variables within the cross-correlation horizon. The diagonal blocks represent the contemporaneous covariance $\Sigma_{t}$ of $z_{t}$ at each time step, while the off-diagonal blocks capture the cross-covariance $\operatorname{Cov}(\mathbf{z}_{t-\Delta}, \mathbf{z}_{t})$ for $\forall\Delta\neq0$ , effectively modeled by our approach. The zoomed-in view provides a $3B\times3B$ region that illustrates the cross-covariance within two lags. We observe that the cross-covariance is most pronounced at lag 1, consistent with the observation in Fig. 4(a) that the component weight $w_{0}$ , assigned to the base kernel matrix with lengthscale l=0.5, is more pronounced than $w_{1}$ and $w_{2}$ .

# 6 Discussion

In this section, we discuss factors that influence the performance of our method. Specifically, we highlight the effectiveness of our model in long-term forecasting across various scenarios. We also

discuss the effect of scaling up to larger batch sizes during prediction. Additionally, we examine the impact of non-Gaussian errors on model performance.

Long-term forecasting. The advantage of modeling error correlation can vary in long-term forecasting, especially in autoregressive predictions where errors accumulate and propagate over time. Using residuals from previous time steps to calibrate forecasts may be beneficial for non-stationary segments of the time series. However, for time series with strong periodic effects, the model may also rely on seasonal lags. As shown in Fig. 21 and Fig. 22 of the Appendix, the advantage of modeling error correlation can decrease in longer-term forecasts compared to shorter-term forecasts for some datasets with strong periodic effects (e.g., the traffic dataset in Fig. 21). It is not necessarily true that the advantage diminishes for long-horizon predictions, as the effectiveness of our method depends on the quality of predictions during inference. In cases where the model provides accurate long-term forecasts, the benefit of modeling correlated errors may be less pronounced.

Scalability. Increasing the number of time series B in a batch leads to higher training costs. Because the model requires numerous iterations over the dataset for optimization, using a large B during training is not feasible. However, during prediction, the batch size can be increased to leverage more information. This may enhance both prediction accuracy and error calibration, provided sufficient memory is available. We demonstrate the effect of increasing batch size during prediction in Appendix §B.4 through additional experiments. Both models, with and without our method, show improvement from increased batch sizes during prediction, as reflected by a decrease in CRPS $_{sum}$ .

Non-Gaussian errors. For the baseline model, assuming Gaussian errors may lead to model misspecification, resulting in more correlated residuals. To address this issue, we also trained the baseline models using the likelihood of a multivariate t-distribution; the results are presented in Table 15 of the Appendix. Although using an alternative distribution can lead to better performance on some datasets without our method, we observed that our method effectively closes the performance gap when the t-distribution outperforms the Gaussian assumption. We chose the Gaussian distribution for its beneficial properties, including its marginalization rule and well-defined conditional distributions, both essential for statistically consistent model training and reliable inference. Thus, a more effective approach could involve first transforming the original observations into Gaussian-distributed data using a Gaussian Copula [3], followed by applying our method.

# 7 Conclusion and Broader Impacts

This paper presents a novel approach for addressing error cross-correlation in multivariate probabilistic time series forecasting, specifically for models with autoregressive properties and Gaussian distribution outputs. We construct a dynamic covariance matrix using a small set of independent and identically distributed latent temporal processes. These latent processes effectively model temporal correlation and integrate seamlessly into the base model, where the contemporaneous covariance is parameterized by a low-rank-plus-diagonal structure. This approach enables the modeling and prediction of a time-varying covariance matrix for the target time series variables. The experimental results demonstrate its effectiveness in enhancing uncertainty quantification.

Our contributions are two-fold. First, our approach relaxes the time-independent error assumption during the training process for probabilistic forecasting models, addressing the reality that residuals are typically time-dependent. Second, the learned cross-correlation improves multistep-ahead predictions by refining the distribution output at each forecasting step. These enhancements to existing models have broader implications for fields such as finance, healthcare, and energy, where improved forecasts and uncertainty quantification can lead to more informed decisions.

There are several avenues for future research. First, the Kronecker structure $C_{t} \otimes I_{R}$ for the covariance matrix of the latent variable $r_{t}^{bat}$ may be too restrictive for multivariate time series problems. Exploring more flexible covariance structures, such as employing different $C_{r,t}$ matrices for each latent temporal process as in the linear model of coregionalization (LMC, [22, 23]), could be a promising direction for further investigation. Second, the parameterization of $C_{t}$ could be expanded. Instead of using SE kernels, $C_{t}$ could be parameterized as fully learnable positive definite symmetric Toeplitz matrices. For example, an AR(p) process has a covariance structure in Toeplitz form, allowing for the modeling of negative correlations. This alternative approach could offer greater flexibility in capturing complex correlation patterns in multivariate time series data.

# Acknowledgments and Disclosure of Funding

We acknowledge the support from the Natural Sciences and Engineering Research Council (NSERC) of Canada (Discovery Grant). Vincent Zhihao Zheng also acknowledges the support received from the FRQNT B2X Doctoral Scholarship Program.

# References

[1] Tilmann Gneiting and Matthias Katzfuss. Probabilistic forecasting. Annual Review of Statistics and Its Application, 1(1):125–151, 2014.   
[2] Konstantinos Benidis, Syama Sundar Rangapuram, Valentin Flunkert, Yuyang Wang, Danielle Maddix, Caner Turkmen, Jan Gasthaus, Michael Bohlke-Schneider, David Salinas, Lorenzo Stella, et al. Deep learning for time series forecasting: Tutorial and literature survey. ACM Computing Surveys, 55(6):1–36, 2022.   
[3] David Salinas, Michael Bohlke-Schneider, Laurent Callot, Roberto Medico, and Jan Gasthaus. High-dimensional multivariate forecasting with low-rank gaussian copula processes. Advances in Neural Information Processing Systems, 32, 2019.   
[4] Kashif Rasul, Abdul-Saboor Sheikh, Ingmar Schuster, Urs Bergmann, and Roland Vollgraf. Multivariate probabilistic time series forecasting via conditioned normalizing flows. In International Conference on Learning Representations, 2021.   
[5] Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In International Conference on Machine Learning, pages 8857–8868. PMLR, 2021.   
[6] Rob J Hyndman and George Athanasopoulos. Forecasting: Principles and Practice. OTexts, 2018.   
[7] Fan-Keng Sun, Chris Lang, and Duane Boning. Adjusting for autocorrelated errors in neural networks for time series. Advances in Neural Information Processing Systems, 34:29806–29819, 2021.   
[8] Vincent Zhihao Zheng, Seongjin Choi, and Lijun Sun. Enhancing deep traffic forecasting models with dynamic regression. arXiv preprint arXiv:2301.06650, 2023.   
[9] Vincent Zhihao Zheng, Seongjin Choi, and Lijun Sun. Better batch for deep probabilistic time series forecasting. In International Conference on Artificial Intelligence and Statistics, pages 91–99. PMLR, 2024.   
[10] David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal of Forecasting, 36(3):1181–1191, 2020.   
[11] Ruofeng Wen, Kari Torkkola, Balakrishnan Narayanaswamy, and Dhruv Madeka. A multi-horizon quantile recurrent forecaster. arXiv preprint arXiv:1711.11053, 2017.   
[12] Syama Sundar Rangapuram, Matthias W Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. Advances in Neural Information Processing Systems, 31, 2018.   
[13] Emmanuel de Bézenac, Syama Sundar Rangapuram, Konstantinos Benidis, Michael Bohlke-Schneider, Richard Kurle, Lorenzo Stella, Hilaf Hasson, Patrick Gallinari, and Tim Januschowski. Normalizing kalman filters for multivariate time series analysis. Advances in Neural Information Processing Systems, 33:2995–3007, 2020.   
[14] Yuyang Wang, Alex Smola, Danielle Maddix, Jan Gasthaus, Dean Foster, and Tim Januschowski. Deep factors for forecasting. In International Conference on Machine Learning, pages 6607–6617. PMLR, 2019.   
[15] Binh Tang and David S Matteson. Probabilistic transformer for time series analysis. Advances in Neural Information Processing Systems, 34:23592–23608, 2021.   
[16] Alexandre Drouin, Étienne Marcotte, and Nicolas Chapados. Tactis: Transformer-attentional copulas for time series. In International Conference on Machine Learning, pages 5447–5493. PMLR, 2022.   
[17] Arjun Ashok, Étienne Marcotte, Valentina Zantedeschi, Nicolas Chapados, and Alexandre Drouin. Tactis-2: Better, faster, simpler attentional copulas for multivariate time series. In International Conference on Learning Representations, 2024.

[18] Raquel Prado, Marco AR Ferreira, and Mike West. Time Series: Modeling, Computation, and Inference. CRC Press, 2021.   
[19] James Douglas Hamilton. Time Series Analysis. Princeton University Press, 2020.   
[20] Arkajyoti Saha, Sumanta Basu, and Abhirup Datta. Random forests for spatially dependent data. Journal of the American Statistical Association, 118(541):665–683, 2023.   
[21] Seongjin Choi, Nicolas Saunier, Vincent Zhihao Zheng, Martin Trepanier, and Lijun Sun. Scalable dynamic mixture model with full covariance for probabilistic traffic forecasting. arXiv preprint arXiv:2212.06653, 2022.   
[22] Alan E Gelfand, Alexandra M Schmidt, Sudipto Banerjee, and CF Sirmans. Nonstationary multivariate process modeling through spatially varying coregionalization. Test, 13:263–312, 2004.   
[23] Rui Meng, Braden Soper, Herbert KH Lee, Vincent X Liu, John D Greene, and Priyadip Ray. Nonstationary multivariate gaussian processes for electronic health records. Journal of Biomedical Informatics, 117:103698, 2021.   
[24] Alexander Alexandrov, Konstantinos Benidis, Michael Bohlke-Schneider, Valentin Flunkert, Jan Gasthaus, Tim Januschowski, Danielle C Maddix, Syama Rangapuram, David Salinas, Jasper Schulz, et al. Gluonts: Probabilistic and neural time series modeling in python. The Journal of Machine Learning Research, 21(1):4629–4634, 2020.   
[25] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.   
[26] Jan Beitner. Pytorch forecasting. https://pytorch-forecasting.readthedocs.io, 2020.   
[27] Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477):359–378, 2007.   
[28] Alireza Koochali, Peter Schichtel, Andreas Dengel, and Sheraz Ahmed. Random noise vs. state-of-the-art probabilistic forecasting methods: A case study on crps-sum discrimination ability. Applied Sciences, 12(10):5104, 2022.   
[29] Étienne Marcotte, Valentina Zantedeschi, Alexandre Drouin, and Nicolas Chapados. Regions of reliability in the evaluation of multivariate probabilistic forecasts. In International Conference on Machine Learning, pages 23958–24004. PMLR, 2023.   
[30] Dheeru Dua and Casey Graff. Uci machine learning repository. URL https://archive.ics.uci.edu/ml/index.php, 2017.   
[31] Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. The m4 competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1):54–74, 2020.   
[32] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pages 95–104, 2018.   
[33] Spyros Makridakis, Allan Andersen, Robert Carbone, Robert Fildes, Michele Hibon, Rudolf Lewandowski, Joseph Newton, Emanuel Parzen, and Robert Winkler. The accuracy of extrapolation (time series) methods: Results of a forecasting competition. Journal of Forecasting, 1(2):111–153, 1982.   
[34] Chao Chen, Karl Petty, Alexander Skabardonis, Pravin Varaiya, and Zhanfeng Jia. Freeway performance measurement system: mining loop detector data. Transportation Research Record, 1748(1):96–102, 2001.   
[35] Caltrans. Caltrans performance measurement system. URL https://pems.dot.ca.gov/, 2015.   
[36] NYC Taxi Limousine Commission. Uber tlc foil response, 2015.   
[37] Jan Gasthaus, Konstantinos Benidis, Yuyang Wang, Syama Sundar Rangapuram, David Salinas, Valentin Flunkert, and Tim Januschowski. Probabilistic forecasting with spline quantile function rnns. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 1901–1910. PMLR, 2019.   
[38] Helmut Lütkepohl. New Introduction to Multiple Time Series Analysis. Springer Science & Business Media, 2005.   
[39] Roy Van der Weide. Go-garch: a multivariate generalized orthogonal garch model. Journal of Applied Econometrics, 17(5):549–564, 2002.

[40] Robert Engle. Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. Journal of Business & Economic Statistics, 20(3):339–350, 2002.   
[41] Skipper Seabold and Josef Perktold. Statsmodels: econometric and statistical modeling with python. SciPy, 7:1, 2010.   
[42] Prashant Srivastava. mgarch. https://pypi.org/project/mgarch/, 2022.   
[43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.   
[44] Lei Bai, Lina Yao, Can Li, Xianzhi Wang, and Can Wang. Adaptive graph convolutional recurrent network for traffic forecasting. Advances in Neural Information Processing Systems, 33:17804–17815, 2020.   
[45] Shun-Yao Shih, Fan-Keng Sun, and Hung-yi Lee. Temporal pattern attention for multivariate time series forecasting. Machine Learning, 108:1421-1441, 2019.   
[46] Massimo Guidolin and Manuela Pedio. Essentials of Time Series for Financial Applications. Academic Press, 2018.

# Appendix

# Table of Contents

# A Experimental Details ...... 14

A.1 Datasets ...... 14   
A.2 Multivariate Likelihood with Autocorrelated Errors ..... 15   
A.3 Training Procedure .... 16   
A.4 Naive Baseline Description ...... 16   
A.5 Hyperparameter Search ...... 17   
A.6 Base Model Description and Input Features .... 17

# B Metrics and Additional Results ....19

B.1 Metric Definition ...... 19   
B.1.1 Continuous Ranked Probability Score ..... 19   
B.1.2 Quantile Loss ...... 20   
B.1.3 Energy Score 20   
B.1.4 Root Relative Mean Squared Error ...... 20

B.2 Results on Other Metrics ...... 20

B.3 Training Dynamics ...... 20

B.4 Effect of the Number of Time Series during Prediction ..... 20

B.5 Additional Model Interpretation 24

B.5.1 Comparison of Residual Correlation ..... 24

B.5.2 Performance Breakdown at Each Forecast Step 24

B.6 Alternative Parametrization of $C_t$ ......29

B.6.1 Learnable Lengthscales ..... 29

B.6.2 Using Autocorrelations of an AR(p) process ..... 30

B.7 Alternative Error Assumptions ..... 31

B.8 Qualitative Results on Forecasting 32

# A Experimental Details

# A.1 Datasets

We performed experiments on a diverse set of real-world datasets obtained from GluonTS [24]. These datasets include:

- electricity [30]: Hourly electricity consumption data collected from a total of 370 households over the period spanning from January 2012 to June 2014.   
- m4\_hourly [31]: Hourly time series data from various domains, covering microeconomics, macroeconomics, finance, industry, demographics, and various other fields, are sourced from the M4-competition.   
- exchange\_rate [32]: Daily exchange rate information for eight different countries spanning the period from 1990 to 2016.   
- m1\_quarterly [33]: Quarterly time series data spanning seven different domains.   
- pems03 [34]: Traffic flow records obtained from Caltrans District 3 and accessed through the Caltrans Performance Measurement System (PeMS). The records are aggregated at a 5-minute interval.

- solar [32]: Hourly time series representing solar power production data in the state of Alabama for the year 2006.   
- traffic [35]: Hourly traffic occupancy rates recorded by sensors installed in the San Francisco freeway system between January 2008 and June 2008.   
- uber\_hourly [36]: Hourly time series of Uber pickups in New York City spanning from February to July 2015.   
- wikipedia [37]: Daily page views for 2,000 Wikipedia pages spanning from January 2012 to March 2014.

These datasets are widely employed for benchmarking time series forecasting models, following their default configurations in GluonTS, including granularity, prediction range $(Q)$ , and the number of rolling evaluations. For each dataset, we performed a sequential split into training, validation, and testing sets, with the temporal length of the validation set matching that of the testing set. The temporal length of the testing set was determined by considering the prediction range and the required number of rolling evaluations. For instance, the testing horizon for the traffic dataset is computed as $24 + 7 - 1 = 30$ time steps. As a result, the model will predict 24 steps $(Q)$ sequentially, with 7 distinct consecutive prediction start timestamps, also known as 7 forecast instances. In our experiments, we align the conditioning range $(P)$ with the prediction range $(Q)$ , maintaining consistency with the default setting in GluonTS. For simplicity, we set the autocorrelation horizon $(D)$ to also match the prediction range $(Q)$ . Essentially, in this paper, we have P = Q = D. Each dataset was standardized using the mean and standard deviation from the training set. Predictions were rescaled to their original values for computing evaluation metrics. The statistics of all datasets are summarized in Table 2.

Table 2: Dataset summary. 

<table><tr><td>Dataset</td><td>Granularity</td><td># of time series</td><td># of time steps</td><td>Q</td><td>Rolling evaluation</td></tr><tr><td>electricity</td><td>hourly</td><td>370</td><td>5,857</td><td>24</td><td>7</td></tr><tr><td>m4_hourly</td><td>hourly</td><td>414</td><td>1,008</td><td>48</td><td>7</td></tr><tr><td>exchange_rate</td><td>workday</td><td>8</td><td>6,101</td><td>30</td><td>5</td></tr><tr><td>m1_quarterly</td><td>quarterly</td><td>281</td><td>48</td><td>8</td><td>1</td></tr><tr><td>pems03</td><td>5min</td><td>358</td><td>26,208</td><td>12</td><td>24</td></tr><tr><td>solar</td><td>hourly</td><td>137</td><td>7,033</td><td>24</td><td>7</td></tr><tr><td>traffic</td><td>hourly</td><td>963</td><td>4,025</td><td>24</td><td>7</td></tr><tr><td>uber_hourly</td><td>hourly</td><td>262</td><td>8,343</td><td>24</td><td>7</td></tr><tr><td>wikipedia</td><td>daily</td><td>2,000</td><td>792</td><td>30</td><td>5</td></tr></table>

# A.2 Multivariate Likelihood with Correlated Errors

The probability density function of a multivariate normal distribution with autocorrelated errors, as described in §4.1, is defined in Eq. (14). For simplicity, we omit the subscript $t$ and superscript bat for all notations:

$$
f (\boldsymbol {z}) = (2 \pi) ^ {- B / 2} | \boldsymbol {\Sigma} | ^ {- 1 / 2} \exp \left(- \frac {1}{2} (\boldsymbol {z} - \boldsymbol {\mu}) ^ {\top} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {z} - \boldsymbol {\mu})\right). \tag {14}
$$

We use the negative log likelihood (NLL) of an observed z as the loss function for training our model. The NLL can be calculated as the negative log of the probability density function in Eq. (14):

$$
\mathcal {L} _ {N L L} = - \ln L (\boldsymbol {z}) = \frac {1}{2} \left[ \ln | \boldsymbol {\Sigma} | + (\boldsymbol {z} - \boldsymbol {\mu}) ^ {\top} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {z} - \boldsymbol {\mu}) + B \ln (2 \pi) \right], \tag {15}
$$

where B is the number of time series in a batch. The covariance matrix is parameterized as $\Sigma = L(\boldsymbol{C} \otimes \mathbf{I}_{R})\boldsymbol{L}^{\top} + \boldsymbol{E}$ . In this parameterization, $L \in R^{DB \times DR}$ is the covariance factor, $C \in R^{D \times D}$ is the autocorrelation matrix, $\boldsymbol{E} = \text{diag}(\boldsymbol{d})$ , and $d \in R_{+}^{DB}$ are the diagonal elements. The bottleneck in evaluating this NLL lies in the calculation of the inverse and determinant of $\Sigma$ . Therefore, we can simplify the calculation using the Sherman–Morrison–Woodbury identity (matrix inversion lemma) and the companion matrix determinant lemma:

$$
\boldsymbol {\Sigma} ^ {- 1} = \left(\boldsymbol {E} + \boldsymbol {L} \left(\boldsymbol {C} \otimes \mathbf {I} _ {R}\right) \boldsymbol {L} ^ {\top}\right) ^ {- 1} \tag {16}
$$

$$
= \boldsymbol {E} ^ {- 1} - \boldsymbol {E} ^ {- 1} \boldsymbol {L} \left(\left(\boldsymbol {C} \otimes \mathbf {I} _ {R}\right) ^ {- 1} + \boldsymbol {L} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {L}\right) ^ {- 1} \boldsymbol {L} ^ {\top} \boldsymbol {E} ^ {- 1}.
$$

Consequently, the Mahalanobis term in Eq. (15) becomes:

$$
\begin{array}{l} \boldsymbol {\eta} ^ {\top} \boldsymbol {\Sigma} ^ {- 1} \boldsymbol {\eta} = \boldsymbol {\eta} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {\eta} - \boldsymbol {\eta} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {L} \left(\left(\boldsymbol {C} \otimes \mathbf {I} _ {R}\right) ^ {- 1} + \boldsymbol {L} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {L}\right) ^ {- 1} \boldsymbol {L} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {\eta} \\ = \boldsymbol {\eta} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {\eta} - \boldsymbol {\eta} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {L} \left(\boldsymbol {L} _ {\text {cap}} \boldsymbol {L} _ {\text {cap}} ^ {\top}\right) ^ {- 1} \boldsymbol {L} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {\eta} \tag {17} \\ = \boldsymbol {\eta} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {\eta} - \left(\boldsymbol {L} _ {c a p} ^ {- 1} \boldsymbol {L} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {\eta}\right) ^ {\top} \left(\boldsymbol {L} _ {c a p} ^ {- 1} \boldsymbol {L} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {\eta}\right) \\ = \boldsymbol {\eta} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {\eta} - \boldsymbol {k} ^ {\top} \boldsymbol {k}, \\ \end{array}
$$

where $k = L_{cap}^{-1}L^{\top}E^{-1}\eta$ . $L_{cap}$ is the Cholesky factor of the capacitance matrix $\left((C\otimes\mathbf{I}_{R})^{-1}+L^{\top}E^{-1}L\right)$ . The computation of k can be efficiently resolved by solving the linear system of equations $L_{cap}k=L^{\top}E^{-1}\eta$ . Since E is a diagonal matrix, the only matrix inverse we need to calculate in Eq. (17) is $(C\otimes\mathbf{I}_{R})^{-1}$ , which can be further simplified as $C^{-1}\otimes I_{R}$ . Recall that C is a $D\times D$ autocorrelation matrix. Therefore, calculating its inverse is much easier than computing the inverse of $\Sigma$ , which is a $DB\times DB$ matrix. Moreover, the computational cost does not scale with the number of time series B in a batch.

The calculation of the determinant can also be greatly simplified with our parameterization:

$$
\begin{array}{l} \ln | \boldsymbol {\Sigma} | = \ln | \boldsymbol {E} + \boldsymbol {L} (\boldsymbol {C} \otimes \mathbf {I} _ {R}) \boldsymbol {L} ^ {\top} | \\ = \ln | (\boldsymbol {C} \otimes \mathbf {I} _ {R}) ^ {- 1} + \boldsymbol {L} ^ {\top} \boldsymbol {E} ^ {- 1} \boldsymbol {L} | + \ln | \boldsymbol {C} \otimes \mathbf {I} _ {R} | + \ln | \boldsymbol {E} | \tag {18} \\ = 2 \sum_ {i} ^ {D R} \ln {[ \pmb {L} _ {c a p} ] _ {i, i}} + 2 R \sum_ {i} ^ {D} \ln {[ \pmb {L} _ {C} ] _ {i, i}} + \sum_ {i} ^ {D B} \ln {[ \pmb {E} ] _ {i, i}}, \\ \end{array}
$$

where $L_{C}$ is the Cholesky factor of the autocorrelation matrix $\pmb{C}$ .

# A.3 Training Procedure

Compute used All models in the paper were trained in an Anaconda environment with access to one AMD Ryzen Threadripper PRO 5955WX CPU and four NVIDIA RTX A5000 GPUs (each with 24 GB of memory).

Batch size We adopt the approach of GPVar [3] by using B = 20 time series in a sample slice and a batch size of 16. Because our data sampler selects one slice of time series as a batch instead of sampling 16 slices simultaneously, we set accumulate\_grad\_batches to 16 to achieve an effective batch size of 16.

Training loop Each epoch involves training the model on up to 400 batches from the training set, followed by computing the NLL on the validation set. Training stops when any of the following conditions are met:

- A total of 10,000 gradient updates have been performed during model training,   
- No improvement in the best NLL value on the validation set is observed for 10 consecutive epochs.

We select the version of the model that achieved the best NLL value on the validation set.

# A.4 Naive Baseline Description

In this paper, we employ VAR [38] (Vector Autoregression) and GARCH [39] (Generalized Autoregressive Conditionally Heteroskedasticity) as two naive baseline models. The VAR(p) model is defined as

$$
\mathbf {z} _ {t} = \mathbf {c} + A _ {1} \mathbf {z} _ {t - 1} + \dots + A _ {p} \mathbf {z} _ {t - p} + \epsilon_ {t}, \epsilon_ {t} \sim \mathcal {N} (\mathbf {0}, \boldsymbol {\Sigma} _ {\epsilon}), \tag {19}
$$

where $A_{i}$ is an $N \times N$ coefficient matrix, and $\mathbf{c}$ is the intercept. We use a VAR model of lag 1 (i.e., a VAR(1) model) in the experiments. The parameters of Eq. (19) are estimated using ordinary least squares (OLS), following the procedure in [38].

The GARCH model describes the conditional covariance matrix of the error term in a multivariate system. Suppose the model for the conditional mean is an AR(1) model:

$$
\mathbf {z} _ {t} = \mathbf {c} + A _ {1} \mathbf {z} _ {t - 1} + \boldsymbol {\epsilon} _ {t}, \tag {20}
$$

where the error term is modeled as

$$
\boldsymbol {\epsilon} _ {t} = \boldsymbol {H} _ {t} ^ {1 / 2} \mathbf {e} _ {t}, \tag {21}
$$

where $H_{t}$ is an $N \times N$ conditional covariance matrix, and $e_{t}$ is an $N \times 1$ standard normal vector, $e_{t} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{N})$ . In the experiments, we use the DCC-GARCH(1, 1) model [40], where the conditional covariance matrix $H_{t}$ is defined as

$$
\boldsymbol {H} _ {t} = \boldsymbol {D} _ {t} \boldsymbol {R} _ {t} \boldsymbol {D} _ {t}, \tag {22}
$$

where $D_{t} = \mathrm{diag}\left(\mathbf{h}_{t}\right)^{1 / 2}$ , and $\mathbf{h}_t$ contains the variances for each time series. $R_{t}$ is the conditional correlation matrix in the DCC-GARCH model. The parameters of the DCC-GARCH model are estimated with the log-likelihood function:

$$
\mathcal {L} = - \frac {1}{2} \sum_ {t = 1} ^ {T} \left[ N \ln (2 \pi) + 2 \ln | \boldsymbol {D} _ {t} | + \ln | \boldsymbol {R} _ {t} | + \mathbf {e} _ {t} ^ {\top} \boldsymbol {R} _ {t} ^ {- 1} \mathbf {e} _ {t} \right]. \tag {23}
$$

In this paper, we implement the VAR model using statsmodels [41] and the DCC-GARCH model using mgarch [42].

# A.5 Hyperparameter Search

The hyperparameters and training configuration largely align with those used in the GPVar paper $[3]$ . All DL models are trained using the Adam optimizer with l2 regularization set to 1e-8, and gradients are clipped at 10.0. For all methods, we limit the total number of gradient updates to 10,000 and decay the learning rate by a factor of 2 after 500 consecutive updates without improvement. Table 3 lists the parameters that are tuned, as well as the hyperparameters that are kept constant across all datasets and not subject to tuning.

Table 3: Hyperparameters values that are fixed or searched over a range during hyperparameter tuning. 

<table><tr><td>Hyperparameter</td><td>Value or Range Searched</td></tr><tr><td>learning rate</td><td>[1e-4, 1e-3, 1e-2]</td></tr><tr><td>LSTM cells / d_model of Transformer</td><td>[10, 20, 40]</td></tr><tr><td>LSTM layers / Transformer decoder layers</td><td>2</td></tr><tr><td>n_heads (Transformer)</td><td>2</td></tr><tr><td>rank</td><td>10</td></tr><tr><td>sampling dimension</td><td>20</td></tr><tr><td>dropout</td><td>0.01</td></tr><tr><td>batch size</td><td>16</td></tr></table>

To tune the hyperparameters of each model, we conduct a grid search over nine parameters on each dataset. The best hyperparameters for each base model–dataset combination are selected based on the lowest validation loss. Once the optimal learning rate and hidden size are determined, we apply the same hyperparameters to models both with and without our method.

The number of base kernels $(M)$ for generating $C_t$ and the associated lengthscale set $\{l_m\}_{m=1}^{M-1}$ are two additional hyperparameters when applying our method. The optimal values of $M$ and $\{l_m\}_{m=1}^{M-1}$ are selected in a similar manner via hyperparameter search. The values of $M$ and $\{l_m\}_{m=1}^{M-1}$ explored during hyperparameter tuning are shown in Table 4. There are six possible combinations. For example, if we set $M = 3$ and choose the initial lengthscale to be 1.0, the lengthscales for generating the component kernels will be $\{1.0, 2.0\}$ since the last weight corresponds to the identity matrix.

# A.6 Base Model Description and Input Features

The input to the base models consists of lagged time series values and generic features that encode time and identify each time series. The number of lagged values used is determined by the time-frequency of each dataset. Specifically, we use lags $[1, 24, 168]$ for hourly data; $[1, 7, 14]$ for daily data; and $[1, 2, 4, 12, 24, 48]$ for data with a granularity of less than one hour. For all other datasets, we only use the lag-1 values.

Table 4: Hyperparameters values of our method that are searched over a range during hyperparameter tuning. 

<table><tr><td>Hyperparameter</td><td>Value or Range Searched</td></tr><tr><td>number of kernels  $M$ </td><td> $[2, 3, 4]$ </td></tr><tr><td>possible lengthscales  $\{l_m\}_{m=1}^{M-1}$ </td><td> $[\{0.5, 1.5, \ldots\}, \{1.0, 2.0, \ldots\}]$ </td></tr></table>

We use generic features to represent time. For datasets with a granularity of one hour or less, we include features for the hour of the day and the day of the week. For daily datasets, we use the day of the week feature. Additionally, each time series is distinguished by an identifier number. All features are encoded with a single value; for example, the hour of the day feature takes values in $[0, 23]$ . These feature values are concatenated with the RNN or Transformer input at each time step to generate the model input vector $y_{t}$ .

As illustrated in §4, our method requires a state vector $h_{t}$ at each time step to generate the parameters for the predictive distribution and the dynamic weights for correlation matrix kernels. We use two different neural architectures for this purpose: RNN and Transformer, both of which preserve autoregressive properties. Specifically, we use an LSTM as our base model for the RNN and a decoder-only Transformer (i.e., the GPT model [25]) for the Transformer. Table 5 and Table 6 summarize the number of parameters for the GPVar and Transformer models across each dataset.

Table 5: Number of parameters of the GPVar model for each dataset. 

<table><tr><td></td><td>covariate embedding</td><td>rnn</td><td>distribution proj</td><td>covariance proj (our method)</td></tr><tr><td>exchange_rate</td><td>60</td><td>6.1k</td><td>252</td><td>84</td></tr><tr><td>solar</td><td>3.7k</td><td>26.6k</td><td>492</td><td>164</td></tr><tr><td>electricity</td><td>16.5k</td><td>29.6k</td><td>492</td><td>164</td></tr><tr><td>traffic</td><td>72.5k</td><td>34.6k</td><td>492</td><td>164</td></tr><tr><td>wikipedia</td><td>200k</td><td>5.7k</td><td>132</td><td>44</td></tr><tr><td>m4_hourly</td><td>19.7k</td><td>10.2k</td><td>252</td><td>84</td></tr><tr><td>m1_quarterly</td><td>6.3k</td><td>25k</td><td>492</td><td>164</td></tr><tr><td>pems03</td><td>26.4k</td><td>34.6k</td><td>492</td><td>164</td></tr><tr><td>uber_hourly</td><td>9.7k</td><td>28.3k</td><td>492</td><td>164</td></tr></table>

Table 6: Number of parameters of the Transformer model for each dataset. 

<table><tr><td></td><td>target proj</td><td>covariate proj</td><td>covariate embedding</td><td>transformer</td><td>distribution proj</td><td>covariance proj (our method)</td></tr><tr><td>exchange_rate</td><td>160</td><td>400</td><td>60</td><td>26.5k</td><td>492</td><td>164</td></tr><tr><td>solar</td><td>40</td><td>400</td><td>3.7k</td><td>1.8k</td><td>132</td><td>44</td></tr><tr><td>electricity</td><td>160</td><td>2.4k</td><td>16.5k</td><td>26.5k</td><td>492</td><td>164</td></tr><tr><td>traffic</td><td>80</td><td>1.8k</td><td>72.5k</td><td>6.8k</td><td>252</td><td>84</td></tr><tr><td>wikipedia</td><td>160</td><td>4.2k</td><td>200k</td><td>26.5k</td><td>492</td><td>164</td></tr><tr><td>m4_hourly</td><td>80</td><td>1.2k</td><td>19.7k</td><td>6.8k</td><td>252</td><td>84</td></tr><tr><td>m1_quarterly</td><td>80</td><td>1.3k</td><td>6.3k</td><td>26.5k</td><td>492</td><td>164</td></tr><tr><td>pems03</td><td>70</td><td>870</td><td>26.4k</td><td>1.8k</td><td>132</td><td>44</td></tr><tr><td>uber_hourly</td><td>160</td><td>2k</td><td>9.7k</td><td>26.5k</td><td>492</td><td>164</td></tr></table>

LSTM, a type of RNN architecture, is designed to model sequences and time series data. Unlike traditional RNNs, LSTMs can learn long-term dependencies, making them effective for tasks requiring context and memory over long sequences. A decoder-only Transformer is primarily used for sequence generation tasks, such as text generation, language modeling, and machine translation. It is a simplified version of the original Transformer model introduced by Vaswani et al. [43], consisting of only the decoder component. The LSTM model can be formulated as

$$
\mathbf {f} _ {t} = \sigma (\mathbf {W} _ {f} \cdot [ \mathbf {h} _ {t - 1}, \mathbf {y} _ {t} ] + \mathbf {b} _ {f}),
$$

$$
\mathbf {i} _ {t} = \sigma (\mathbf {W} _ {i} \cdot [ \mathbf {h} _ {t - 1}, \mathbf {y} _ {t} ] + \mathbf {b} _ {i}),
$$

$$
\tilde {\mathbf {C}} _ {t} = \tanh \left(\mathbf {W} _ {C} \cdot \left[ \mathbf {h} _ {t - 1}, \mathbf {y} _ {t} \right] + \mathbf {b} _ {C}\right), \tag {24}
$$

$$
\mathbf {C} _ {t} = \mathbf {f} _ {t} \odot \mathbf {C} _ {t - 1} + \mathbf {i} _ {t} \odot \tilde {\mathbf {C}} _ {t},
$$

$$
\mathbf {o} _ {t} = \sigma (\mathbf {W} _ {o} \cdot [ \mathbf {h} _ {t - 1}, \mathbf {y} _ {t} ] + \mathbf {b} _ {o}),
$$

$$
\mathbf {h} _ {t} = \mathbf {o} _ {t} \odot \tanh (\mathbf {C} _ {t}),
$$

where $y_{t}$ is the input at each time step. The decoder-only Transformer can be formulated as

$$
\mathbf {Q} = \mathbf {Y} _ {t} \mathbf {W} _ {Q},
$$

$$
\mathbf {K} = \mathbf {Y} _ {t} \mathbf {W} _ {K},
$$

$$
\mathbf {V} = \mathbf {Y} _ {t} \mathbf {W} _ {V},
$$

$$
\mathbf {M} = \operatorname{Mask} (\mathbf {K}),
$$

$$
\mathbf {Z} = \text { Softmax } \left(\frac {\mathbf {Q} \mathbf {K} ^ {T}}{\sqrt {d _ {k}}} + \mathbf {M}\right) \mathbf {V}, \tag {25}
$$

$$
\mathbf {H} _ {t} = \text { LayerNorm } (\mathbf {Y} _ {t} + \mathbf {Z}),
$$

$$
\mathbf {F F N} = \operatorname{ReLU} \left(\mathbf {H} _ {t} \mathbf {W} _ {1} + \mathbf {b} _ {1}\right) \mathbf {W} _ {2} + \mathbf {b} _ {2},
$$

$$
\mathbf {H} _ {t} = \text { LayerNorm } (\mathbf {H} _ {t} + \mathbf {F F N}),
$$

where $H_{t}$ is the output containing state vectors for all time steps, and M is a square causal mask for the sequence to preserve autoregressive properties.

# B Metrics and Additional Results

# B.1 Metric Definition

In this paper, we repeated the evaluation process on the testing set ten times to compute the mean and standard deviation of all metrics. Metrics calculated in each independent evaluation are based on the average results from all forecast instances in the testing set. For example, the $CRPS_{sum}$ reported for traffic is the average $CRPS_{sum}$ of seven forecast instances in its testing set. 100 prediction samples were drawn for all evaluation processes.

# B.1.1 Continuous Ranked Probability Score

The Continuous Ranked Probability Score (CRPS) is defined as:

$$
\operatorname{CRPS} (F, z) = \mathbb {E} _ {F} | Z - z | - \frac {1}{2} \mathbb {E} _ {F} | Z - Z ^ {\prime} |, \tag {26}
$$

where $F$ is the cumulative distribution function (CDF) of the predicted variable, $z$ is the observation, $Z$ and $Z'$ are independent copies of a set of prediction samples associated with the distribution $F$ . For a single forecast instance, we calculate the average CRPS across time series and over the prediction horizon:

$$
\mathbb {E} _ {i, t} \left[ \mathrm{CRPS} \left(F _ {i, t}, z _ {i, t}\right) \right], \tag {27}
$$

where we use the empirical CDF to represent $F_{i,t}$ when predicting $z_{i,t}$ . Since CRPS only compares a single ground-truth value to its predicted distribution, we also calculate the CRPS $_{sum}$ [3, 16, 17] to assess multivariate dependencies in the time series data. CRPS $_{sum}$ is computed by summing both the forecasted and ground-truth values across all time series and then calculating the CRPS over the resulting sums:

$$
\mathbb {E} _ {t} \left[ \mathrm{CRPS} \left(F _ {t}, \sum_ {i} z _ {i, t}\right) \right], \tag {28}
$$

where the empirical $F_{t}$ is obtained by summing samples across time series.

# B.1.2 Quantile Loss

The Quantile Loss $(\rho$ -risk) is another metric used in [10] to evaluate the performance of probabilistic forecasting:

$$
L _ {\rho} (z, \hat {z} ^ {\rho}) = 2 (\hat {z} ^ {\rho} - z) ((1 - \rho) I _ {\hat {z} ^ {\rho} > z} - \rho I _ {\hat {z} ^ {\rho} \leq z}), \tag {29}
$$

where I is a binary indicator function that equals 1 when the condition is met, $\hat{z}^{\rho}$ represents the predicted $\rho$ -quantile, and $z$ represents the ground truth value. The quantile loss serves as a metric to assess the accuracy of a given quantile $\rho$ from the predictive distribution. We summarize the quantile losses over the testing set across all time series segments by computing a normalized summation of these losses: $\left(\sum_{i,t} L_{\rho}(z_{i,t}, \hat{z}_{i,t}^{\rho})\right) / \left(\sum_{i,t} z_{i,t}\right)$ . In this paper, we evaluate the 0.5-risk and the 0.9-risk following Salinas et al. [10].

# B.1.3 Energy Score

The Energy Score (ES) generalizes the CRPS to evaluate distributional forecasts of a vector-valued random variable and is thus another multivariate metric used in this paper:

$$
\operatorname{ES} (P, \mathbf {z}) = \underset {\mathbf {Z} \sim P} {\mathbb {E}} \| \mathbf {Z} - \mathbf {z} \| _ {2} ^ {\beta} - \frac {1}{2} \underset {\substack {\mathbf {Z} \sim P \\ \mathbf {Z} ^ {\prime} \sim P}} {\mathbb {E}} \| \mathbf {Z} - \mathbf {Z} ^ {\prime} \| _ {2} ^ {\beta}, \tag{30}
$$

where $\| \mathbf{z}\| _2$ is the Euclidean norm. In this paper, we use $\beta = 1$ , following [17]. Since we also want to aggregate over the prediction horizon, we calculate the Frobenius norm of the matrix $\| \mathbf{z}_{t + 1:t + Q}\| _F$ in practice.

# B.1.4 Root Relative Mean Squared Error

The Root Relative Mean Squared Error (RRMSE) is a metric commonly used for point forecasts $[44, 32, 45]$ . RRMSE is defined as:

$$
\mathrm{RRMSE} = \frac {\sqrt {\sum_ {t = 1} ^ {Q} \| \mathbf {z} _ {t} - \hat {\mathbf {z}} _ {t} \| _ {2} ^ {2}}}{\sqrt {\sum_ {t = 1} ^ {Q} \| \mathbf {z} _ {t} - \bar {\mathbf {z}} \| _ {2} ^ {2}}}, \tag {31}
$$

where $\hat{\mathbf{z}}_t$ is obtained by taking the mean of our prediction samples, and $\bar{\mathbf{z}}$ is the mean value of the entire forecast instance. We use this metric to evaluate the mean prediction performance of our model.

# B.2 Results on Other Forecasting Metrics

We present the results for CRPS (Table 7), the 0.5-risk (Table 8), the 0.9-risk (Table 9), ES (Table 10), and RRMSE (Table 11). An “N/A” entry in the tables indicates that the naive baseline models could not be properly fitted to this dataset. We observe consistent performance improvements in the base models using our method across different evaluation metrics. Notably, in the multivariate metric ES, our method shows significant improvement, reducing the score by an average of 5.58% for GPVar and 3.21% for the Transformer.

# B.3 Training Dynamics

In Fig. 5 and Fig. 6, we compare the training dynamics of the base models trained with and without our method. Note that the likelihood losses of the two methods are not directly comparable, even for the same dataset and base model, due to differences in the likelihood structures. We observe that while our method introduces more complexity into the likelihood function, there is no evidence that it significantly prolongs model convergence. On the contrary, our method can speed up convergence for some datasets in terms of the training steps used. We also report the training time in Table 12.

# B.4 Effect of the Number of Time Series during Prediction

The number of time series does not impact training, as the model is trained using a random subset of B time series at a time, independent of the total number of time series N. However, during prediction, the batch size can be increased beyond the training batch size of B = 20 for multistep-ahead rolling

Table 7: Comparison of CRPS accuracy. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method. Boldface values indicate that models considering time-dependent errors have better performance. Mean and standard deviation are obtained from 10 runs of each model. 

<table><tr><td rowspan="2"></td><td>VAR</td><td>GARCH</td><td colspan="2">GPVar</td><td colspan="2">Transformer</td></tr><tr><td></td><td></td><td>w/o</td><td>w/</td><td>w/o</td><td>w/</td></tr><tr><td>exchange_rate</td><td>0.0070±0.0000</td><td>0.0438±0.0001</td><td>0.0171±0.0004</td><td>0.0141±0.0003</td><td>0.0092±0.0002</td><td>0.0081±0.0001</td></tr><tr><td>solar</td><td>0.9566±0.0022</td><td>0.9193±0.0010</td><td>0.7097±0.0047</td><td>0.7521±0.0027</td><td>0.5981±0.0021</td><td>0.5627±0.0018</td></tr><tr><td>electricity</td><td>0.1548±0.0003</td><td>0.2778±0.0010</td><td>0.0586±0.0004</td><td>0.0568±0.0002</td><td>0.0665±0.0003</td><td>0.0775±0.0001</td></tr><tr><td>traffic</td><td>19.9208±0.0495</td><td>0.4063±0.0002</td><td>0.1474±0.0001</td><td>0.1296±0.0001</td><td>0.1260±0.0001</td><td>0.1318±0.0001</td></tr><tr><td>wiki</td><td>334.6021±0.4936</td><td>3.0351±0.0048</td><td>0.3712±0.0003</td><td>0.3705±0.0004</td><td>0.3737±0.0003</td><td>0.2937±0.0002</td></tr><tr><td>m4_hourly</td><td>0.2837±0.0004</td><td>0.3567±0.0004</td><td>0.1174±0.0002</td><td>0.1237±0.0002</td><td>0.1306±0.0002</td><td>0.1189±0.0002</td></tr><tr><td>m1_quarterly</td><td>N/A</td><td>N/A</td><td>0.3942±0.0030</td><td>0.3538±0.0017</td><td>0.4448±0.0027</td><td>0.4367±0.002</td></tr><tr><td>pems03</td><td>0.1144±0.0001</td><td>0.3533±0.0002</td><td>0.0828±0.0000</td><td>0.0835±0.0001</td><td>0.0826±0.0001</td><td>0.0735±0.0000</td></tr><tr><td>uber_hourly</td><td>N/A</td><td>N/A</td><td>0.1488±0.0003</td><td>0.1468±0.0002</td><td>0.1576±0.0003</td><td>0.1762±0.0003</td></tr><tr><td></td><td></td><td></td><td>avg. rel. impr.</td><td>3.59%</td><td>avg. rel. impr.</td><td>3.13%</td></tr></table>

Table 8: Comparison of 0.5-risk accuracy. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method. Boldface values indicate that models considering time-dependent errors have better performance. Mean and standard deviation are obtained from 10 runs of each model. 

<table><tr><td rowspan="2"></td><td>VAR</td><td>GARCH</td><td colspan="2">GPVar</td><td colspan="2">Transformer</td></tr><tr><td></td><td></td><td>w/o</td><td>w/</td><td>w/o</td><td>w/</td></tr><tr><td>exchange_rate</td><td>0.0049±0.0000</td><td>0.0256±0.0001</td><td>0.0109±0.0003</td><td>0.0095±0.0004</td><td>0.0060±0.0001</td><td>0.0056±0.0001</td></tr><tr><td>solar</td><td>0.6140±0.0025</td><td>0.5621±0.0008</td><td>0.4998±0.0025</td><td>0.5246±0.0016</td><td>0.4233±0.0017</td><td>0.3958±0.0013</td></tr><tr><td>electricity</td><td>0.1113±0.0005</td><td>0.2014±0.0010</td><td>0.0405±0.0003</td><td>0.0397±0.0002</td><td>0.0449±0.0002</td><td>0.0505±0.0001</td></tr><tr><td>traffic</td><td>10.2654±0.0268</td><td>0.2722±0.0002</td><td>0.0933±0.0001</td><td>0.0859±0.0001</td><td>0.0803±0.0001</td><td>0.0794±0.0001</td></tr><tr><td>wiki</td><td>171.5009±0.2573</td><td>0.7225±0.0067</td><td>0.2231±0.0005</td><td>0.2236±0.0006</td><td>0.2030±0.0005</td><td>0.1487±0.0003</td></tr><tr><td>m4_hourly</td><td>0.1992±0.0003</td><td>0.2365±0.0005</td><td>0.0807±0.0001</td><td>0.0849±0.0001</td><td>0.0880±0.0002</td><td>0.0808±0.0001</td></tr><tr><td>m1_quarterly</td><td>N/A</td><td>N/A</td><td>0.2196±0.0023</td><td>0.1948±0.0005</td><td>0.2328±0.0008</td><td>0.2327±0.0014</td></tr><tr><td>pems03</td><td>0.0784±0.0001</td><td>0.2028±0.0002</td><td>0.0568±0.0000</td><td>0.0574±0.0001</td><td>0.0569±0.0000</td><td>0.0506±0.0000</td></tr><tr><td>uber_hourly</td><td>N/A</td><td>N/A</td><td>0.1035±0.0002</td><td>0.1013±0.0002</td><td>0.1093±0.0003</td><td>0.1234±0.0002</td></tr><tr><td></td><td></td><td></td><td>avg. rel. impr.</td><td>2.75%</td><td>avg. rel. impr.</td><td>3.88%</td></tr></table>

Table 9: Comparison of 0.9-risk accuracy. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method. Boldface values indicate that models considering time-dependent errors have better performance. Mean and standard deviation are obtained from 10 runs of each model. 

<table><tr><td rowspan="2"></td><td>VAR</td><td>GARCH</td><td colspan="2">GPVar</td><td colspan="2">Transformer</td></tr><tr><td></td><td></td><td>w/o</td><td>w/</td><td>w/o</td><td>w/</td></tr><tr><td>exchange_rate</td><td>0.0021±0.0000</td><td>0.0070±0.0000</td><td>0.0042±0.0001</td><td>0.0057±0.0001</td><td>0.0030±0.0000</td><td>0.0023±0.0001</td></tr><tr><td>solar</td><td>0.4676±0.0016</td><td>0.4393±0.0008</td><td>0.1617±0.0004</td><td>0.1597±0.0003</td><td>0.2744±0.0015</td><td>0.2710±0.0015</td></tr><tr><td>electricity</td><td>0.0414±0.0003</td><td>0.0744±0.0003</td><td>0.0211±0.0004</td><td>0.0185±0.0002</td><td>0.0281±0.0002</td><td>0.0366±0.0002</td></tr><tr><td>traffic</td><td>11.0170±0.0405</td><td>0.1689±0.0001</td><td>0.0666±0.0001</td><td>0.0580±0.0001</td><td>0.0609±0.0001</td><td>0.0698±0.0000</td></tr><tr><td>wiki</td><td>174.0756±0.3770</td><td>1.5906±0.0044</td><td>0.2136±0.0002</td><td>0.2048±0.0001</td><td>0.2117±0.0006</td><td>0.1764±0.0003</td></tr><tr><td>m4_hourly</td><td>0.1029±0.0003</td><td>0.1309±0.0003</td><td>0.0452±0.0002</td><td>0.0463±0.0001</td><td>0.0525±0.0001</td><td>0.0475±0.0002</td></tr><tr><td>m1_quarterly</td><td>N/A</td><td>N/A</td><td>0.3049±0.0044</td><td>0.2787±0.0027</td><td>0.3784±0.0031</td><td>0.3621±0.0037</td></tr><tr><td>pems03</td><td>0.0399±0.0000</td><td>0.1783±0.0001</td><td>0.0317±0.0000</td><td>0.0317±0.0001</td><td>0.0304±0.0000</td><td>0.0269±0.0000</td></tr><tr><td>uber_hourly</td><td>N/A</td><td>N/A</td><td>0.0533±0.0002</td><td>0.0528±0.0001</td><td>0.0562±0.0001</td><td>0.0638±0.0002</td></tr><tr><td></td><td></td><td></td><td>avg. rel. impr.</td><td>0.22%</td><td>avg. rel. impr.</td><td>0.91%</td></tr></table>

Table 10: Comparison of ES accuracy. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method. Boldface values indicate that models considering time-dependent errors have better performance. Mean and standard deviation are obtained from 10 runs of each model. 

<table><tr><td rowspan="2"></td><td>VAR</td><td>GARCH</td><td colspan="2">GPVar</td><td colspan="2">Transformer</td></tr><tr><td></td><td></td><td>w/o</td><td>w/</td><td>w/o</td><td>w/</td></tr><tr><td>exchange_rate</td><td>0.1301±0.0002</td><td>0.6085±0.0009</td><td>0.3674±0.0067</td><td>0.2613±0.0047</td><td>0.1798±0.0039</td><td>0.1438±0.0026</td></tr><tr><td>solar (×103)</td><td>1.7429±0.0043</td><td>1.7758±0.0015</td><td>1.6052±0.0095</td><td>1.6591±0.0050</td><td>1.5307±0.0049</td><td>1.4633±0.0044</td></tr><tr><td>electricity (×105)</td><td>1.0102±0.0052</td><td>1.9422±0.0127</td><td>0.3569±0.0050</td><td>0.3172±0.0028</td><td>0.4031±0.0040</td><td>0.4754±0.0025</td></tr><tr><td>traffic</td><td>3.3585±0.010 (×103)</td><td>4.4198±0.0020</td><td>2.4008±0.0020</td><td>2.2408±0.0015</td><td>2.2240±0.0021</td><td>2.2566±0.0014</td></tr><tr><td>wiki (×107)</td><td>970.0242±2.5944</td><td>2.8857±0.0783</td><td>0.1149±0.0027</td><td>0.1155±0.0031</td><td>0.1236±0.004</td><td>0.1075±0.0046</td></tr><tr><td>m4_hourly (×103)</td><td>4.5109±0.0084</td><td>5.1849±0.0089</td><td>2.2729±0.0062</td><td>2.3611±0.0060</td><td>2.5877±0.0098</td><td>2.3440±0.0081</td></tr><tr><td>m1_quarterly (×102)</td><td>N/A</td><td>N/A</td><td>3.7565±0.0294</td><td>3.3676±0.0147</td><td>4.2149±0.0252</td><td>4.1596±0.0248</td></tr><tr><td>pems03 (×103)</td><td>1.3951±0.0009</td><td>5.4642±0.0067</td><td>1.0535±0.0010</td><td>1.0736±0.0015</td><td>1.0673±0.0012</td><td>0.9394±0.0004</td></tr><tr><td>uber_hourly (×103)</td><td>N/A</td><td>N/A</td><td>0.9035±0.0041</td><td>0.8773±0.0027</td><td>0.9377±0.0035</td><td>1.0566±0.0033</td></tr><tr><td></td><td></td><td></td><td>avg. rel. impr.</td><td>5.58%</td><td>avg. rel. impr.</td><td>3.21%</td></tr></table>

Table 11: Comparison of RRMSE accuracy. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method. Boldface values indicate that models considering time-dependent errors have better performance. Mean and standard deviation are obtained from 10 runs of each model. 

<table><tr><td rowspan="2"></td><td>VAR</td><td>GARCH</td><td colspan="2">GPVar</td><td colspan="2">Transformer</td></tr><tr><td></td><td></td><td>w/o</td><td>w/</td><td>w/o</td><td>w/</td></tr><tr><td>exchange_rate</td><td>0.0247±0.0000</td><td>0.0983±0.0002</td><td>0.0699±0.0012</td><td>0.0501±0.0010</td><td>0.0350±0.0008</td><td>0.0265±0.0007</td></tr><tr><td>solar</td><td>0.9365±0.0025</td><td>0.9556±0.0008</td><td>0.8195±0.0038</td><td>0.8334±0.0019</td><td>0.8114±0.0023</td><td>0.7761±0.0019</td></tr><tr><td>electricity</td><td>0.2732±0.0020</td><td>0.5584±0.0036</td><td>0.1010±0.0013</td><td>0.0912±0.0009</td><td>0.1130±0.0011</td><td>0.1293±0.0007</td></tr><tr><td>traffic</td><td>0.6312±0.0017 (×103)</td><td>0.9894±0.0008</td><td>0.5383±0.0005</td><td>0.5061±0.0003</td><td>0.5025±0.0005</td><td>0.5052±0.0003</td></tr><tr><td>wiki</td><td>0.6519±0.0016 (×104)</td><td>6.3386±0.2020</td><td>1.0288±0.0029</td><td>1.0393±0.0039</td><td>0.9292±0.0057</td><td>0.8752±0.0027</td></tr><tr><td>m4_hourly</td><td>0.6163±0.0012</td><td>0.6848±0.0015</td><td>0.3072±0.0008</td><td>0.3168±0.0007</td><td>0.3420±0.0011</td><td>0.3179±0.0010</td></tr><tr><td>m1_quarterly</td><td>N/A</td><td>N/A</td><td>19.1005±0.1246</td><td>17.0277±0.0845</td><td>20.2333±0.0830</td><td>20.2708±0.0924</td></tr><tr><td>pems03</td><td>0.3727±0.0003</td><td>0.8824±0.0013</td><td>0.2796±0.0003</td><td>0.2877±0.0005</td><td>0.2841±0.0003</td><td>0.2502±0.0001</td></tr><tr><td>uber_hourly</td><td>N/A</td><td>N/A</td><td>0.2358±0.0012</td><td>0.2282±0.0008</td><td>0.2458±0.0010</td><td>0.2768±0.0009</td></tr><tr><td></td><td></td><td></td><td>avg. rel. impr.</td><td>5.48%</td><td>avg. rel. impr.</td><td>2.85%</td></tr></table>

Table 12: Training cost comparison. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method. 

<table><tr><td rowspan="2"></td><td colspan="4">GPVar</td><td colspan="4">Transformer</td></tr><tr><td>w/o sec./epoch</td><td>epochs</td><td>w/ sec./epoch</td><td>epochs</td><td>w/o sec./epoch</td><td>epochs</td><td>w/ sec./epoch</td><td>epochs</td></tr><tr><td>exchange_rate</td><td>4.60</td><td>56</td><td>200.27</td><td>39</td><td>9.73</td><td>57</td><td>206.49</td><td>41</td></tr><tr><td>solar</td><td>6.18</td><td>39</td><td>74.16</td><td>51</td><td>15.37</td><td>121</td><td>181.26</td><td>132</td></tr><tr><td>electricity</td><td>7.44</td><td>71</td><td>119.06</td><td>94</td><td>19.38</td><td>63</td><td>103.15</td><td>65</td></tr><tr><td>traffic</td><td>10.50</td><td>55</td><td>225.20</td><td>48</td><td>28.30</td><td>84</td><td>247.58</td><td>100</td></tr><tr><td>wiki</td><td>12.45</td><td>30</td><td>164.81</td><td>33</td><td>28.16</td><td>51</td><td>351.06</td><td>48</td></tr><tr><td>m4_hourly</td><td>7.42</td><td>67</td><td>189.14</td><td>43</td><td>17.81</td><td>43</td><td>355.27</td><td>65</td></tr><tr><td>m1_quarterly</td><td>4.24</td><td>51</td><td>25.82</td><td>12</td><td>9.84</td><td>29</td><td>24.99</td><td>17</td></tr><tr><td>pems03</td><td>11.75</td><td>62</td><td>143.34</td><td>57</td><td>36.31</td><td>78</td><td>88.05</td><td>53</td></tr><tr><td>uber_hourly</td><td>6.82</td><td>41</td><td>174.90</td><td>57</td><td>17.28</td><td>35</td><td>188.08</td><td>65</td></tr></table>

![](images/f9e3952e4b52d01c9d661e3ea1882c54319fe043bab70d96925aae324bd8ff40.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | -20        | -25      |
| 1000 | -35        | -35      |
</details>

![](images/983f5fc1b8412613431a346606e0f64a57eec02c60a32fe9d22252eeda9e7de3.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 80         | 60       |
| 1000 | 25         | 20       |
</details>

![](images/3803f7b9a94e0bed8b1c96446d721be631a8884e56f8fe3a3056c1fdd4641284.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 80         | 80       |
| 2000 | 70         | 70       |
</details>

![](images/f4b06df3019cc717d6c965fdd1a85011eb75c1fd920329c224661413d2c6bb1a.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | -50        | -65      |
| 1000 | -70        | -70      |
</details>

![](images/f6a69d4360dd9f35d1d32cdf54b5f04165094005b26177888720bc99b5d7c1f5.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 175        | 190      |
| 500  | 165        | 180      |
</details>

![](images/57f1692add071e093705315db7008dc035dba9b954d7ce22cd5a1d071bb6b2a7.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 85         | 40       |
| 500  | 55         | 25       |
| 1000 | 50         | 20       |
| 1500 | 48         | 18       |
| 2000 | 47         | 17       |
</details>

![](images/8f7198e4517478ab715134dd4e99fdd7b0c5d0259abc9531c5b8356289bbcf79.jpg)

<details>
<summary>line</summary>

| step | train_loss: w/o | val_loss |
| ---- | --------------- | -------- |
| 0    | 110             | 40       |
| 1000 | 85              | 25       |
</details>

![](images/543c82717a11410add01a70626c57eba6ca5757093a45e5ed67f6af271407ffb.jpg)

<details>
<summary>line</summary>

| step | train_loss: w | val_loss: |
| ---- | ------------- | --------- |
| 0    | 92.5          | 92.5      |
| 1000 | 87.5          | 87.5      |
</details>

![](images/9b2175942a7408d7325db55a3dcf4532bd4674aef1ad6c472b4d563a2a8dc2ba.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 45         | 46       |
| 500  | 40         | 44       |
| 1000 | 42         | 45       |
| 1500 | 43         | 46       |
</details>

Figure 5: Training loss/validation loss vs training time of the GPVar model. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method.   
![](images/fa668ca35454d68e72808d77bdac3d91cab77aea11ee0d521265689ab34339ed.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | -20        | -25      |
| 1000 | -35        | -35      |
</details>

![](images/bcb61f4ee92f369a6f741778bd83d5bf3e01a7e81c627965f63de96993266f78.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 80         | 75       |
| 1000 | 40         | 35       |
| 2000 | 25         | 25       |
| 3000 | 20         | 20       |
</details>

![](images/da721bbb8ca8bebca60d58729f3cb136f8c3efe03549818c2cfd4e5c0bd086ed.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 80         | 80       |
| 1000 | 70         | 70       |
</details>

![](images/c4e1fe3717a6f3c8e466d8141b0e789f0487c1ef4578782c9019b4b72da5e077.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | -50        | -60      |
| 2000 | -70        | -70      |
</details>

![](images/236f81ddcc85df150816a28a5ea3b94f22ee70dd5ce881f439de3d6a291e3ad7.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 170        | 190      |
| 1000 | 165        | 180      |
</details>

![](images/9c54b1a6b9701f6db745a9feb0e8be6470393c87c4b01581ab4154504e60a653.jpg)

<details>
<summary>line</summary>

| step | train loss | val loss |
| ---- | ---------- | -------- |
| 0    | 80         | 40       |
| 1000 | 50         | 20       |
</details>

![](images/a4394d7bedbcffec9d87bae4bcae83bfcbe9c6b7c9cacf9defe1ceb17779338a.jpg)

<details>
<summary>line</summary>

| step | train_loss: w/o | val_loss |
| ---- | --------------- | -------- |
| 0    | 100             | 100      |
| 500  | 80              | 25       |
</details>

![](images/f7e8675432e428062f0e4630631a369b0d941714c02e5fe6ea385d9d661c7557.jpg)

<details>
<summary>line</summary>

| step | train_loss: w | val_loss: w/ |
| ---- | ------------- | ------------ |
| 0    | 92.5          | 92.5         |
| 2000 | 87.5          | 87.5         |
</details>

![](images/9b138c30a30bb7b6012e1758cdbd63b55dda304c9677adcc5cb899e9e149495a.jpg)

<details>
<summary>line</summary>

| step | val_loss: w |
| ---- | ----------- |
| 0    | 42.5        |
| 1000 | 40.0        |
| 2000 | 44.0        |
| 3000 | 46.0        |
| 4000 | 48.0        |
</details>

Figure 6: Training loss/validation loss vs training time of the Transformer model. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method.

predictions. This allows more information to be utilized, potentially improving both predictions and error calibration, provided that memory capacity permits. We conducted an additional experiment to demonstrate the effect of increasing the batch size during inference (Fig. 7).

![](images/4770f1a4598503b2012a00648018058d14e0e7a78fda1dd55e51039ded169740.jpg)

<details>
<summary>line</summary>

| # of time series | w/o   | w/    |
| ---------------- | ----- | ----- |
| 20               | 0.71  | 0.69  |
| 60               | 0.65  | 0.63  |
| 100              | 0.62  | 0.60  |
| 140              | 0.60  | 0.59  |
| 180              | 0.60  | 0.59  |
</details>

![](images/2ff37865f80b136e2042ddf354c77bb8b78636744d9603c00634271a144f2207.jpg)

<details>
<summary>line</summary>

| # of time series | w/o    | w/     |
| ---------------- | ------ | ------ |
| 20               | 0.043  | 0.040  |
| 60               | 0.043  | 0.0395 |
| 100              | 0.043  | 0.039  |
| 140              | 0.0425 | 0.0385 |
| 180              | 0.0425 | 0.039  |
</details>

![](images/36e4962aaa4cb15f5b8787c490330357021f4489fa6008b985ad18312358acee.jpg)

<details>
<summary>line</summary>

| # of time series | w/o   | w/    |
| ---------------- | ----- | ----- |
| 20               | 0.11  | 0.065 |
| 60               | 0.105 | 0.06  |
| 100              | 0.10  | 0.055 |
| 140              | 0.10  | 0.055 |
| 180              | 0.10  | 0.055 |
</details>

![](images/a4f501686ffb2727f67a30ff182a27c7f8779ba0f0bce095284dd499cd9448c2.jpg)

<details>
<summary>line</summary>

| # of time series | w/o    | w/     |
| ---------------- | ------ | ------ |
| 20               | 0.175  | 0.075  |
| 60               | 0.175  | 0.075  |
| 100              | 0.175  | 0.075  |
| 140              | 0.175  | 0.075  |
| 180              | 0.175  | 0.075  |
</details>

![](images/b8ff600cf0bfed63bba4ba7ac460e77249b20df3f41242de104b24701efce2ca.jpg)

<details>
<summary>line</summary>

| # of time series | w/o   | w/    |
| ---------------- | ----- | ----- |
| 20               | 0.06  | 0.035 |
| 60               | 0.06  | 0.034 |
| 100              | 0.06  | 0.033 |
| 140              | 0.06  | 0.033 |
| 180              | 0.06  | 0.033 |
</details>

Figure 7: The influence of the number of time series in a batch on the performance of inference. "w/o" denotes methods without time-dependent errors, while "w/" indicates our method. We only show some datasets here because the remaining datasets have fewer than $B = 20$ time series in the testing set.

# B.5 Additional Model Interpretation

In this section, we provide further insights into how our method improves the base model. We illustrate these improvements by comparing the cross-correlations of the residuals from models with and without our method. Additionally, we demonstrate the performance of our method over the prediction horizon in multistep-ahead forecasting.

# B.5.1 Comparison of Residual Correlation

Recall that our method models both the autocovariance of errors $\mathrm{Cov}(\eta_{i,t-\Delta},\eta_{i,t})$ and the cross-lag covariance $\mathrm{Cov}(\eta_{i,t-\Delta},\eta_{j,t})$ between all pairs of components in the multivariate series. With the calibration process introduced in §4.2, our method is expected to reduce error cross-correlations, including autocorrelation and cross-lag correlation. Here, we compare the empirical ACF of the residuals $\eta_{i,t}$ of a single time series i, as well as the empirical cross-correlations of $\eta_{t}$ across multiple time series.

We begin by comparing the ACF of the one-step-ahead prediction residuals with and without our method. The comparisons are provided for the following datasets: solar (Fig. 8), electricity (Fig. 9), traffic (Fig. 10), wiki (Fig. 11), m4\_hourly (Fig. 12), pems03 (Fig. 13), and uber\_hourly (Fig. 14). We observe that the autocorrelation of the residuals is reduced after applying our method.

Next, we compare the cross-correlations of the one-step-ahead prediction residuals with and without our method. The comparisons are provided for the following datasets: electricity (Fig. 15), traffic (Fig. 16), wiki (Fig. 17), m4\_hourly (Fig. 18), pems03 (Fig. 19), and uber\_hourly (Fig. 20). We also observe that the cross-correlations of the residuals are reduced after applying our method.

# B.5.2 Performance Breakdown at Each Forecast Step

To investigate our performance gain at each forecast step, we calculate the CRPS $_{sum}$ for each forecast step. The results are shown in Fig. 21 for GPVar and Fig. 22 for the Transformer. Note that the CRPS $_{sum}$ reported in this section may have different scales compared to previous sections because

![](images/29637caf917e9207d3c3d7cb4a50a4d2a875c283855c05e8ee1bf44d4d382ae7.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 1.0    |
| 1    | 0.3    |
| 2    | -0.2   |
| 3    | -0.5   |
| 4    | -0.1   |
| 5    | 0.0    |
| 6    | 0.1    |
| 7    | 0.0    |
| 8    | -0.1   |
| 9    | 0.0    |
| 10   | 0.0    |
| 11   | 0.0    |
| 12   | 0.0    |
| 13   | 0.0    |
| 14   | 0.0    |
| 15   | 0.1    |
| 16   | 0.0    |
| 17   | -0.1   |
| 18   | -0.2   |
</details>

![](images/aefa2772ea5350cb1c7733c5a6687b4fe43f98c4ae043570759d8bdfd7eda725.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 1.0    |
| 1    | 0.3    |
| 2    | 0.1    |
| 3    | -0.1   |
| 4    | -0.2   |
| 5    | -0.3   |
| 6    | -0.1   |
| 7    | 0.0    |
| 8    | 0.0    |
| 9    | 0.0    |
| 10   | 0.0    |
| 11   | 0.0    |
| 12   | 0.0    |
| 13   | 0.0    |
| 14   | 0.0    |
| 15   | 0.0    |
| 16   | 0.0    |
| 17   | 0.0    |
</details>

![](images/6fe98b434c80d8743bb9a60b9fac3622b712af2eb1a9ff2abeadc064a25da123.jpg)

<details>
<summary>scatter</summary>

| lags | ACF    |
|------|--------|
| 0    | 0.5    |
| 1    | 0.3    |
| 2    | 0.2    |
| 3    | 0.1    |
| 4    | 0.0    |
| 5    | -0.1   |
| 6    | -0.2   |
| 7    | -0.1   |
| 8    | 0.0    |
| 9    | 0.0    |
| 10   | 0.0    |
| 11   | -0.1   |
| 12   | -0.2   |
| 13   | -0.1   |
| 14   | 0.0    |
| 15   | 0.0    |
| 16   | 0.1    |
| 17   | 0.2    |
| 18   | 0.1    |
</details>

![](images/d8545b9732bc00e9d9cd7d14d9e61b8d77b7a344fdd6241763a05311651b5ab0.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 1.0000 |
| 1    | 0.0500 |
| 2    | 0.0200 |
| 3    | -0.0100|
| 4    | -0.0300|
| 5    | -0.0400|
| 6    | -0.0200|
| 7    | -0.0100|
| 8    | 0.0000 |
| 9    | 0.0050 |
| 10   | 0.0100 |
| 11   | 0.0150 |
| 12   | 0.0200 |
| 13   | 0.0250 |
| 14   | 0.0300 |
| 15   | 0.0350 |
| 16   | 0.0400 |
| 17   | 0.0450 |
| 18   | 0.0500 |
</details>

Figure 8: ACF comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the solar dataset.   
![](images/2a935c7023d878c5af1a33bd18d3c579c9fd1e73127edfce88049670179cbe52.jpg)

<details>
<summary>scatter</summary>

| lags | ACF (blue) | ACF (orange) |
|------|------------|--------------|
| 0    | 0.0        | 0.0          |
| 1    | 0.2        | -0.1         |
| 2    | 0.1        | 0.0          |
| 3    | 0.0        | 0.0          |
| 4    | 0.0        | 0.0          |
| 5    | 0.0        | 0.0          |
| 6    | 0.0        | 0.0          |
| 7    | 0.0        | 0.1          |
| 8    | 0.0        | 0.0          |
| 9    | 0.0        | 0.0          |
| 10   | 0.0        | -0.1         |
| 11   | 0.0        | 0.0          |
| 12   | 0.0        | 0.1          |
| 13   | 0.0        | 0.0          |
| 14   | 0.0        | -0.1         |
| 15   | 0.0        | -0.1         |
| 16   | 0.0        | 0.0          |
| 17   | 0.0        | 0.0          |
| 18   | 0.0        | 0.0          |
</details>

![](images/b84e2ecd84e7c59bb5144d30057dc1d9f57d4d2eb05552f79fd1d779e5a74645.jpg)

<details>
<summary>scatter</summary>

| lags | ACF (blue) | ACF (orange) |
|------|------------|--------------|
| 0    | 0.0        | 0.0          |
| 1    | 0.1        | 0.05         |
| 2    | 0.05       | 0.08         |
| 3    | 0.0        | 0.1          |
| 4    | -0.05      | 0.05         |
| 5    | -0.1       | 0.0          |
| 6    | -0.05      | 0.05         |
| 7    | 0.0        | 0.0          |
| 8    | 0.05       | 0.05         |
| 9    | 0.1        | 0.0          |
| 10   | 0.0        | -0.1         |
| 11   | -0.05      | -0.15        |
| 12   | -0.1       | 0.0          |
| 13   | -0.05      | 0.05         |
| 14   | 0.0        | 0.0          |
| 15   | 0.05       | 0.0          |
| 16   | 0.1        | -0.05        |
| 17   | 0.0        | -0.1         |
| 18   | -0.05      | -0.15        |
| 19   | -0.1       | -0.2         |
</details>

![](images/769fb198182f9bcc1317b7abcabbd6d6c018f02c9787ef9533d6cead949b44d2.jpg)

<details>
<summary>scatter</summary>

| lags | ACF    |
|------|--------|
| 0    | 0.2    |
| 1    | 0.1    |
| 2    | -0.1   |
| 3    | 0.0    |
| 4    | -0.2   |
| 5    | -0.1   |
| 6    | -0.1   |
| 7    | -0.1   |
| 8    | -0.1   |
| 9    | -0.1   |
| 10   | -0.1   |
| 11   | -0.1   |
| 12   | -0.1   |
| 13   | 0.0    |
| 14   | -0.1   |
| 15   | -0.2   |
| 16   | -0.1   |
| 17   | 0.0    |
| 18   | 0.0    |
</details>

![](images/ed677f500dc020f368a9d3f298b9852c4a137fcfd2c4bddc4c6a51a3da54b32d.jpg)

<details>
<summary>line</summary>

| lags | ACF (w/)
| lags | ACF (blue) |
|------|------------|
| 0    | 1.0        |
| 1    | 0.2        |
| 2    | -0.1       |
| 3    | 0.1        |
| 4    | -0.2       |
| 5    | 0.1        |
| 6    | -0.1       |
| 7    | -0.3       |
| 8    | -0.1       |
| 9    | 0.0        |
| 10   | 0.0        |
| 11   | 0.1        |
| 12   | 0.0        |
| 13   | 0.1        |
| 14   | 0.0        |
| 15   | 0.0        |
| 16   | 0.1        |
| 17   | -0.1       |
| 18   | 0.0        |
</details>

Figure 9: ACF comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the electricity dataset.

![](images/65b4f87b30b3e92372acde0f2a451b55c0245f39b87eac40c07cb530884aa797.jpg)

<details>
<summary>scatter</summary>

| lags | ACF    |
|------|--------|
| 0    | 1.0000 |
| 1    | 0.2000 |
| 2    | -0.1000 |
| 3    | 0.0500 |
| 4    | -0.2500 |
| 5    | -0.1500 |
| 6    | 0.0800 |
| 7    | -0.0500 |
| 8    | 0.1200 |
| 9    | -0.0800 |
| 10   | 0.1500 |
| 11   | -0.1200 |
| 12   | 0.1800 |
| 13   | -0.1800 |
| 14   | 0.2000 |
| 15   | -0.1500 |
| 16   | 0.1600 |
| 17   | -0.1400 |
| 18   | 0.1700 |
</details>

![](images/0d8198950165917becfffde66c63d3935224c7d84cc9042c5ddc7480d25f36e0.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 1.0    |
| 1    | 0.2    |
| 2    | 0.1    |
| 3    | -0.1   |
| 4    | -0.2   |
| 5    | -0.1   |
| 6    | 0.1    |
| 7    | 0.2    |
| 8    | 0.1    |
| 9    | 0.0    |
| 10   | -0.1   |
| 11   | -0.2   |
| 12   | -0.1   |
| 13   | -0.2   |
| 14   | -0.1   |
| 15   | -0.2   |
| 16   | -0.1   |
| 17   | 0.2    |
| 18   | 0.1    |
</details>

![](images/c5a6fb1114d94f336f43763b6a9d4420a7b0c4cbedf89e85be0a1bc9abe7916a.jpg)

<details>
<summary>scatter</summary>

| lags | ACF    |
|------|--------|
| 0    | 0.0    |
| 1    | 0.2    |
| 2    | -0.1   |
| 3    | -0.2   |
| 4    | -0.1   |
| 5    | -0.1   |
| 6    | -0.1   |
| 7    | -0.1   |
| 8    | -0.1   |
| 9    | -0.1   |
| 10   | -0.1   |
| 11   | 0.1    |
| 12   | 0.0    |
| 13   | 0.1    |
| 14   | -0.1   |
| 15   | -0.1   |
| 16   | 0.0    |
| 17   | 0.1    |
| 18   | -0.1   |
</details>

![](images/23f3c25b3850eedf288ac55101b3779eccaa8b4a1975f9a04dfa12958066f31d.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 1.0    |
| 1    | 0.2    |
| 2    | 0.0    |
| 3    | 0.0    |
| 4    | 0.0    |
| 5    | 0.0    |
| 6    | 0.0    |
| 7    | 0.0    |
| 8    | 0.0    |
| 9    | 0.0    |
| 10   | -0.1   |
| 11   | 0.0    |
| 12   | 0.0    |
| 13   | 0.0    |
| 14   | 0.0    |
| 15   | 0.0    |
| 16   | 0.0    |
| 17   | -0.1   |
</details>

Figure 10: ACF comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the traffic dataset.

![](images/193b744692415227893ee3a671f048d1d597ed66c1bc931179b34d60bc376fa5.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
|------|--------|
| 0    | 1.0000 |
| 1    | 0.3000 |
| 2    | 0.2500 |
| 3    | 0.2000 |
| 4    | 0.1500 |
| 5    | 0.1000 |
| 6    | -0.0500 |
| 7    | -0.1000 |
| 8    | -0.1500 |
| 9    | -0.2000 |
| 10   | -0.2500 |
| 11   | -0.3000 |
| 12   | -0.3500 |
| 13   | -0.4000 |
| 14   | -0.4500 |
| 15   | -0.5000 |
| 16   | -0.5500 |
| 17   | -0.6000 |
| 18   | -0.6500 |
</details>

![](images/fc3a4a4d492bf75743bc2bd226d4d0db226e10cd12c1a44a3d112734c3fc10eb.jpg)

<details>
<summary>scatter</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 0.8    |
| 1    | 0.6    |
| 2    | 0.4    |
| 3    | 0.3    |
| 4    | 0.2    |
| 5    | 0.1    |
| 6    | 0.0    |
| 7    | -0.1   |
| 8    | -0.2   |
| 9    | -0.3   |
| 10   | -0.4   |
| 11   | -0.5   |
| 12   | -0.6   |
| 13   | -0.7   |
| 14   | -0.8   |
| 15   | -0.9   |
| 16   | -1.0   |
| 17   | -1.1   |
| 18   | -1.2   |
</details>

![](images/ebe1414aae50fa5123cb2aed59ee67dd62126f3ec98fbf950fa776c16edd6e1d.jpg)

<details>
<summary>scatter</summary>

| lags | ACF (w/o) | ACF (other) |
|------|-----------|-------------|
| 0    | 0.3       | 0.2         |
| 1    | 0.1       | 0.1         |
| 2    | 0.1       | 0.1         |
| 3    | 0.1       | 0.1         |
| 4    | 0.1       | 0.1         |
| 5    | 0.2       | 0.1         |
| 6    | 0.1       | 0.1         |
| 7    | 0.1       | 0.1         |
| 8    | 0.1       | 0.1         |
| 9    | 0.0       | 0.0         |
| 10   | -0.1      | -0.1        |
| 11   | -0.1      | -0.1        |
| 12   | -0.1      | -0.1        |
| 13   | -0.1      | -0.1        |
| 14   | -0.1      | -0.1        |
| 15   | -0.1      | -0.1        |
| 16   | -0.1      | -0.1        |
| 17   | -0.1      | -0.1        |
| 18   | -0.1      | -0.1        |
</details>

![](images/ae00bb43abefc03e93c3d55f627e3eb3ab5305b9f6c69dfcdb65ce97d3487508.jpg)

<details>
<summary>line</summary>

| lags | w/ ACF | w/ ACF |
|------|--------|--------|
| 0    | 1.0    | 0.5    |
| 1    | 0.3    | 0.2    |
| 2    | 0.4    | 0.1    |
| 3    | 0.2    | 0.0    |
| 4    | 0.1    | -0.1   |
| 5    | 0.0    | -0.2   |
| 6    | -0.1   | -0.3   |
| 7    | -0.2   | -0.4   |
| 8    | -0.3   | -0.5   |
| 9    | -0.4   | -0.6   |
| 10   | -0.5   | -0.7   |
| 11   | -0.6   | -0.8   |
| 12   | -0.7   | -0.9   |
| 13   | -0.8   | -1.0   |
| 14   | -0.9   | -1.1   |
| 15   | -1.0   | -1.2   |
| 16   | -1.1   | -1.3   |
| 17   | -1.2   | -1.4   |
| 18   | -1.3   | -1.5   |
| 19   | -1.4   | -1.6   |
| 20   | -1.5   | -1.7   |
</details>

Figure 11: ACF comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the wiki dataset.   
![](images/614180478f646ce9950051bf2468e05e9b8733ece48b4f2546de73bbbeab3b5b.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 0.7    |
| 1    | 0.4    |
| 2    | 0.3    |
| 3    | 0.2    |
| 4    | 0.1    |
| 5    | 0.05   |
| 6    | 0.03   |
| 7    | 0.02   |
| 8    | 0.01   |
| 9    | 0.005  |
| 10   | 0.003  |
| 11   | 0.002  |
| 12   | 0.001  |
| 13   | 0.0005 |
| 14   | 0.0003 |
| 15   | 0.0002 |
| 16   | 0.0001 |
| 17   | 0.00005|
| 18   | 0.00003|
| 19   | 0.00002|
| 20   | 0.00001|
</details>

![](images/0bcaae2e3d689154a3887469765db557967194666f66d31c940e1fb700246649.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 1.0000 |
| 1    | 0.2500 |
| 2    | 0.3000 |
| 3    | 0.2000 |
| 4    | 0.1500 |
| 5    | 0.1000 |
| 6    | 0.0500 |
| 7    | 0.0250 |
| 8    | 0.0100 |
| 9    | 0.0050 |
| 10   | 0.0025 |
| 11   | 0.0010 |
| 12   | 0.0005 |
| 13   | 0.0002 |
| 14   | 0.0001 |
| 15   | 0.0000 |
| 16   | -0.0050 |
| 17   | -0.0100 |
| 18   | -0.0150 |
| 19   | -0.0200 |
| 20   | -0.0250 |
</details>

![](images/4331edbf21cf1a8a95f583ffa6aa5ec4e9a490c3ba7532592a0d6efe2308b617.jpg)

<details>
<summary>scatter</summary>

| lags | ACF    |
|------|--------|
| 0    | 0.5    |
| 1    | 0.3    |
| 2    | 0.2    |
| 3    | 0.1    |
| 4    | 0.05   |
| 5    | 0.02   |
| 6    | -0.01  |
| 7    | -0.03  |
| 8    | -0.05  |
| 9    | -0.07  |
| 10   | -0.09  |
| 11   | -0.11  |
| 12   | -0.13  |
| 13   | -0.15  |
| 14   | -0.17  |
| 15   | -0.19  |
| 16   | -0.21  |
| 17   | -0.23  |
| 18   | -0.25  |
</details>

![](images/153c02158e7b07d4e2864a35c492ea946ad684017dd88605492dc1afd7c9733e.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 0.7    |
| 1    | 0.4    |
| 2    | 0.3    |
| 3    | 0.2    |
| 4    | 0.1    |
| 5    | -0.1   |
| 6    | -0.2   |
| 7    | -0.3   |
| 8    | -0.4   |
| 9    | -0.5   |
| 10   | -0.6   |
| 11   | -0.7   |
| 12   | -0.8   |
| 13   | -0.9   |
| 14   | -1.0   |
| 15   | -1.1   |
| 16   | -1.2   |
| 17   | -1.3   |
| 18   | -1.4   |
</details>

Figure 12: ACF comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the m4\_hourly dataset.   
![](images/900bf4b49aec76c109f8343b17b17e23dc89a3664512578c1e419037a4c350e3.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 1.0000 |
| 1    | 0.3000 |
| 2    | 0.2500 |
| 3    | 0.2000 |
| 4    | 0.1500 |
| 5    | 0.1000 |
| 6    | 0.0500 |
| 7    | 0.0000 |
| 8    | -0.0500|
| 9    | -0.1000|
| 10   | -0.1500|
| 11   | -0.2000|
| 12   | -0.2500|
| 13   | -0.3000|
| 14   | -0.3500|
| 15   | -0.4000|
| 16   | -0.4500|
| 17   | -0.5000|
| 18   | -0.5500|
</details>

![](images/bf1d46d6978d8703bbb48f3316a810b72f17567693adbf8f59406ed6b61df5be.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 0.5    |
| 1    | 0.3    |
| 2    | 0.4    |
| 3    | 0.2    |
| 4    | 0.3    |
| 5    | 0.2    |
| 6    | 0.3    |
| 7    | 0.2    |
| 8    | 0.3    |
| 9    | 0.2    |
| 10   | 0.3    |
| 11   | 0.2    |
| 12   | 0.3    |
| 13   | 0.2    |
| 14   | 0.1    |
| 15   | 0.2    |
| 16   | 0.1    |
| 17   | 0.2    |
| 18   | 0.1    |
</details>

![](images/6d0747ea37a6b818b00f582578bb0bed776beff3d23d7b54124aeed61c2ae66b.jpg)

<details>
<summary>bar</summary>

| lags | w/o ACF | w/o Error |
|------|---------|-----------|
| 0    | 0.8     | 0.2       |
| 1    | 0.7     | 0.2       |
| 2    | 0.6     | 0.2       |
| 3    | 0.5     | 0.2       |
| 4    | 0.4     | 0.2       |
| 5    | 0.3     | 0.2       |
| 6    | 0.2     | 0.2       |
| 7    | 0.1     | 0.2       |
| 8    | 0.0     | 0.2       |
| 9    | -0.1    | 0.2       |
| 10   | -0.2    | 0.2       |
| 11   | -0.3    | 0.2       |
| 12   | -0.4    | 0.2       |
| 13   | -0.5    | 0.2       |
| 14   | -0.6    | 0.2       |
| 15   | -0.7    | 0.2       |
| 16   | -0.8    | 0.2       |
| 17   | -0.9    | 0.2       |
| 18   | -1.0    | 0.2       |
</details>

![](images/c69bd6f0df6c8b49f4aa8a0cca26a3cc7181d60b9c29ecebb05b072026de6e03.jpg)

<details>
<summary>line</summary>

| lags | ACF    |
|------|--------|
| 0    | 0.5    |
| 1    | 0.4    |
| 2    | 0.3    |
| 3    | 0.2    |
| 4    | 0.1    |
| 5    | 0.0    |
| 6    | -0.1   |
| 7    | -0.2   |
| 8    | -0.3   |
| 9    | -0.4   |
| 10   | -0.5   |
| 11   | -0.6   |
| 12   | -0.7   |
| 13   | -0.8   |
| 14   | -0.9   |
| 15   | -1.0   |
| 16   | -1.1   |
| 17   | -1.2   |
| 18   | -1.3   |
</details>

Figure 13: ACF comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the pems03 dataset.

![](images/e91cc0ca0f502cf5e91233a485d9e0e7a734a1c628f54434ec02da9691fa0a83.jpg)

<details>
<summary>line</summary>

| lags | ACF (blue) | ACF (orange) |
| ---- | ---------- | ------------ |
| 0    | 0.0        | 0.0          |
| 1    | -0.2       | -0.1         |
| 2    | 0.0        | 0.0          |
| 3    | 0.0        | 0.0          |
| 4    | -0.1       | -0.2         |
| 5    | -0.1       | -0.1         |
| 6    | 0.0        | 0.0          |
| 7    | 0.1        | 0.0          |
| 8    | 0.0        | 0.0          |
| 9    | -0.1       | -0.1         |
| 10   | -0.1       | -0.1         |
| 11   | 0.0        | 0.0          |
| 12   | -0.1       | -0.1         |
| 13   | 0.1        | 0.0          |
| 14   | 0.0        | 0.0          |
| 15   | 0.1        | 0.0          |
| 16   | 0.0        | 0.0          |
| 17   | -0.1       | -0.1         |
| 18   | -0.1       | -0.1         |
</details>

![](images/a6ca572d79468c07e4bd1e30b0bb21887ddd20bcf363ee9912d71c9661da7eb0.jpg)

<details>
<summary>line</summary>

| lags | ACF (blue) | ACF (orange) |
|------|------------|--------------|
| 0    | 0.0        | 0.0          |
| 1    | -0.1       | 0.0          |
| 2    | -0.15      | 0.0          |
| 3    | -0.1       | 0.0          |
| 4    | -0.05      | 0.0          |
| 5    | 0.0        | 0.0          |
| 6    | 0.05       | 0.0          |
| 7    | 0.0        | 0.0          |
| 8    | -0.05      | 0.0          |
| 9    | -0.1       | 0.0          |
| 10   | -0.15      | 0.0          |
| 11   | -0.1       | 0.0          |
| 12   | -0.05      | 0.0          |
| 13   | 0.0        | 0.0          |
| 14   | 0.05       | 0.0          |
| 15   | 0.1        | 0.0          |
| 16   | 0.15       | 0.0          |
| 17   | 0.1        | 0.0          |
| 18   | 0.05       | 0.0          |
| 19   | 0.0        | 0.0          |
| 20   | -0.05      | 0.0          |
</details>

![](images/ae5ade6e7b96404c49fbb5bcd2469be7cc34e8bbc7dbca7ec8c880b26eb0b615.jpg)

<details>
<summary>scatter</summary>

| lags | ACF    |
| ---- | ------ |
| 0    | 0.0    |
| 1    | 0.0    |
| 2    | 0.0    |
| 3    | 0.0    |
| 4    | 0.0    |
| 5    | 0.0    |
| 6    | 0.0    |
| 7    | 0.0    |
| 8    | 0.0    |
| 9    | 0.0    |
| 10   | 0.0    |
| 11   | 0.0    |
| 12   | 0.0    |
| 13   | 0.0    |
| 14   | 0.0    |
| 15   | 0.0    |
| 16   | 0.0    |
| 17   | 0.0    |
| 18   | 0.0    |
</details>

![](images/b4fc66733c7fa492208f34bd5fca366fd700670d63feb204508789f409254740.jpg)

<details>
<summary>line</summary>

| lags | ACF (blue) | ACF (orange) |
| ---- | ---------- | ------------ |
| 0    | 0.0        | 0.0          |
| 1    | -0.2       | 0.1          |
| 2    | -0.1       | 0.0          |
| 3    | -0.1       | 0.0          |
| 4    | -0.1       | 0.0          |
| 5    | -0.1       | 0.0          |
| 6    | -0.1       | 0.0          |
| 7    | -0.1       | 0.0          |
| 8    | -0.1       | 0.0          |
| 9    | -0.1       | 0.0          |
| 10   | -0.1       | 0.0          |
| 11   | -0.1       | 0.1          |
| 12   | -0.1       | 0.0          |
| 13   | -0.1       | 0.1          |
| 14   | -0.1       | 0.0          |
| 15   | -0.1       | 0.1          |
| 16   | -0.1       | 0.0          |
| 17   | -0.1       | 0.0          |
| 18   | -0.1       | 0.0          |
</details>

Figure 14: ACF comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the uber\_hourly dataset.

![](images/4601d898d100e66a96ae38e151df4fed66b51f1752b5abce67c312d622842c81.jpg)

Figure 15: Cross-correlation comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the electricity dataset.   
![](images/f43f89504b201d241f4cb985e002650592f4bfd55d9d0c6074b79e67426144a0.jpg)  
Figure 16: Cross-correlation comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the traffic dataset.

![](images/8d2ef95b9077270a1ad22deeedd4845edca6eb46e23bc40ff82e92a6b6432454.jpg)

Figure 17: Cross-correlation comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the wiki dataset.   
![](images/3b77eb276bebb2cf52604db0336a984fccad8f59f0954c6af96c69bb27976395.jpg)

Figure 18: Cross-correlation comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the m4\_hourly dataset.   
![](images/fbdda6086e5877a07c079df57998d448ae63a739814ad7e4f58dbea7fb7c10db.jpg)  
Figure 19: Cross-correlation comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the pems03 dataset.

![](images/dd837740f8bc45e20c1fd370d3bc5d5b44946851b6a84b8ed4c72defb063f4fa.jpg)  
Figure 20: Cross-correlation comparison of the one-step-ahead prediction residuals with and without our method. The results depict the prediction outcomes generated by GPVar for four time series in the uber\_hourly dataset.

they are not normalized. In multistep-ahead forecasting, since the predicted values are used as inputs for subsequent predictions within the prediction range, the residuals accumulate the effects of inaccuracies from previous steps. Therefore, the performance improvement depends not only on our modeling of error correlations but also on the properties of the residuals. These properties can be influenced by the absolute and relative time of the forecast and the seasonality of the data. For data without strong seasonality, residuals tend to be larger when predicting further ahead, making error accumulation more apparent. Conversely, for data with strong seasonality, the impact of error accumulation can vary. We observe that, in most scenarios, CRPS $_{sum}$ is reduced at the early forecasting stages. As predictions extend further into the future, some datasets (e.g., traffic in Fig. 21) show decreased improvement, likely due to seasonality effects. Conversely, other datasets (e.g., wiki in Fig. 21) exhibit larger improvements further into the future, possibly because the residuals accumulate over the steps.

![](images/2c859a2ad6c2abf4a12f42f74dfcb453671f1013f13f7ecba47b9053418db038.jpg)

<details>
<summary>line</summary>

| Step | w/o  | w/   |
|------|------|------|
| 0    | 0.0  | 0.0  |
| 3    | 0.0  | 0.0  |
| 6    | 0.0  | 0.0  |
| 9    | 0.0  | 0.0  |
| 12   | 0.0  | 0.0  |
| 15   | 3.0  | 3.0  |
| 18   | 1.0  | 1.0  |
| 21   | 0.5  | 0.5  |
| 24   | 0.0  | 0.0  |
</details>

![](images/efc98132e9e7a64957c8c92f237c8fdd5b84088a90718e635504ea84f81b9071.jpg)

<details>
<summary>line</summary>

| Step | w/o    | w/     |
|------|--------|--------|
| 0    | 0.075  | 0.000  |
| 3    | 0.090  | 0.050  |
| 6    | 0.025  | 0.075  |
| 9    | 0.025  | 0.025  |
| 12   | 0.010  | 0.010  |
| 15   | 0.010  | 0.010  |
| 18   | 0.100  | 0.085  |
| 21   | 0.085  | 0.075  |
| 24   | 0.085  | 0.075  |
</details>

![](images/73e4b36da0b69c34af9777525e0ff7275248a4ca1eb00a6d15f3ce6b98badfd9.jpg)

<details>
<summary>line</summary>

| Step | w/o   | w/    |
|------|-------|-------|
| 0    | 0.05  | 0.01  |
| 3    | 0.25  | 0.03  |
| 6    | 0.22  | 0.05  |
| 9    | 0.18  | 0.07  |
| 12   | 0.15  | 0.09  |
| 15   | 0.05  | 0.03  |
| 18   | 0.02  | 0.20  |
| 21   | 0.20  | 0.22  |
| 24   | 0.00  | 0.00  |
</details>

![](images/fe494ca885f6d987e03254b39d292541ece4ec646c3dd0e1eb986659aa73c124.jpg)

<details>
<summary>line</summary>

| Step | w/o   | w/    |
|------|-------|-------|
| 0    | 0.000 | 0.000 |
| 4    | 0.150 | 0.050 |
| 8    | 0.180 | 0.120 |
| 12   | 0.250 | 0.150 |
| 16   | 0.280 | 0.130 |
| 20   | 0.260 | 0.160 |
| 24   | 0.300 | 0.140 |
| 28   | 0.270 | 0.080 |
</details>

![](images/83b9336a659b3fd07079823b245b9c3fa20ef1dcf6c9b4a09a2755b31ac8330d.jpg)

<details>
<summary>line</summary>

| Step | w/o    | w/     |
|------|--------|--------|
| 0    | 0.05   | 0.10   |
| 6    | 0.02   | 0.03   |
| 12   | 0.04   | 0.06   |
| 18   | 0.03   | 0.02   |
| 24   | 0.05   | 0.04   |
| 30   | 0.07   | 0.03   |
| 36   | 0.04   | 0.02   |
| 42   | 0.15   | 0.12   |
| 48   | 0.16   | 0.14   |
</details>

![](images/665372958f3896dcf4c7ce61c7b2c0123a3fda5930fdc9e7cdfc8b83bd343330.jpg)

<details>
<summary>line</summary>

| Step | w/o   | w/    |
| ---- | ----- | ----- |
| 0    | 0.30  | 0.28  |
| 1    | 0.30  | 0.29  |
| 2    | 0.38  | 0.36  |
| 3    | 0.40  | 0.37  |
| 4    | 0.41  | 0.37  |
| 5    | 0.42  | 0.37  |
| 6    | 0.43  | 0.38  |
| 7    | 0.45  | 0.42  |
</details>

![](images/db60b8343db41b5b7e996e09dc06ca91a426fa982adf72bf468f88e52509cb6e.jpg)

<details>
<summary>line</summary>

| Step | w/o    | w/     |
| ---- | ------ | ------ |
| 0    | 0.02   | 0.015  |
| 1    | 0.045  | 0.01   |
| 2    | 0.015  | 0.035  |
| 3    | 0.02   | 0.03   |
| 4    | 0.025  | 0.025  |
| 5    | 0.03   | 0.02   |
| 6    | 0.045  | 0.015  |
| 7    | 0.05   | 0.01   |
| 8    | 0.04   | 0.02   |
| 9    | 0.035  | 0.025  |
| 10   | 0.02   | 0.04   |
</details>

![](images/a70c271a5fe70d02f99b7ab52519cf71451e95b9190a2917396ce5d68cc82816.jpg)

<details>
<summary>line</summary>

| Step | w/o    | w/     |
|------|--------|--------|
| 0    | 0.025  | 0.025  |
| 3    | 0.040  | 0.030  |
| 6    | 0.015  | 0.010  |
| 9    | 0.020  | 0.025  |
| 12   | 0.035  | 0.055  |
| 15   | 0.045  | 0.040  |
| 18   | 0.030  | 0.015  |
| 21   | 0.100  | 0.025  |
| 24   | 0.100  | 0.030  |
</details>

Figure 21: Step-wise CRPS $_{sum}$ accuracy of GPVar. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method.

# B.6 Alternative Parametrization of $C_t$

# B.6.1 Learnable Lengthscales

In this paper, the lengthscales are fixed when generating the correlation matrix $C_{t}$ , and the flexibility of $C_{t}$ comes from dynamically generating the component weights of the kernel matrices. Making these lengthscales learnable parameters to find the optimal set of $\{l_{m}\}_{m=1}^{M-1}$ is another approach we can explore to increase modeling flexibility. Based on the best model identified in Table 1, we experiment with treating the lengthscales as learnable parameters, jointly optimized with the base

![](images/b355bc5be8aa6298e08d49475ca0b2af52bf64216a23da2ebc66bac0b6b90cdf.jpg)

<details>
<summary>line</summary>

| Step | w/o    | w/     |
| ---- | ------ | ------ |
| 0    | 0.0015 | 0.0015 |
| 4    | 0.0030 | 0.0055 |
| 8    | 0.0040 | 0.0030 |
| 12   | 0.0060 | 0.0045 |
| 16   | 0.0045 | 0.0055 |
| 20   | 0.0055 | 0.0050 |
| 24   | 0.0090 | 0.0045 |
| 28   | 0.0105 | 0.0045 |
</details>

![](images/b915f3abcf463ddb26555097caa5196d12bede3567168a91b2dec02e0717fc9e.jpg)

<details>
<summary>line</summary>

| Step | w/o  | w/   |
|------|------|------|
| 0    | 0.0  | 0.0  |
| 3    | 0.0  | 0.0  |
| 6    | 0.0  | 0.0  |
| 9    | 0.0  | 0.0  |
| 12   | 1.5  | 1.0  |
| 15   | 0.5  | 0.3  |
| 18   | 1.8  | 1.6  |
| 21   | 0.7  | 0.5  |
| 24   | 0.0  | 0.0  |
</details>

![](images/ea0c511445bf9559e3e4add3309d7a11612ea84327d7b688e9da7757e395f690.jpg)

<details>
<summary>line</summary>

| Step | w/o    | w/     |
| ---- | ------ | ------ |
| 0    | 0.01   | 0.01   |
| 4    | 0.10   | 0.15   |
| 8    | 0.03   | 0.02   |
| 12   | 0.10   | 0.09   |
| 16   | 0.14   | 0.07   |
| 20   | 0.03   | 0.04   |
| 24   | 0.22   | 0.12   |
| 28   | 0.11   | 0.03   |
</details>

![](images/51efc0a08535983f7f38d28c3695ce9bedb1eac9bb7c04014a5e000a8b8defbb.jpg)

<details>
<summary>line</summary>

| Step | w/o   | w/    |
| ---- | ----- | ----- |
| 0    | 0.32  | 0.30  |
| 1    | 0.38  | 0.36  |
| 2    | 0.45  | 0.45  |
| 3    | 0.47  | 0.47  |
| 4    | 0.47  | 0.47  |
| 5    | 0.46  | 0.46  |
| 6    | 0.48  | 0.48  |
| 7    | 0.51  | 0.51  |
</details>

![](images/fbeb821d81d9a8ee7b43dbd02cc7bbfb572075548d055ae48962e0c432920444.jpg)

<details>
<summary>line</summary>

| Step | w/o    | w/     |
| ---- | ------ | ------ |
| 0    | 0.02   | 0.01   |
| 1    | 0.07   | 0.06   |
| 2    | 0.04   | 0.03   |
| 3    | 0.04   | 0.02   |
| 4    | 0.04   | 0.01   |
| 5    | 0.05   | 0.02   |
| 6    | 0.06   | 0.02   |
| 7    | 0.07   | 0.02   |
| 8    | 0.06   | 0.01   |
| 9    | 0.05   | 0.01   |
| 10   | 0.04   | 0.01   |
</details>

![](images/70704beef6092111dd817b5d540829da4fd4d87f9aeea9bd71eca426ab80bb5b.jpg)

<details>
<summary>line</summary>

| Step | w/o    | w/     |
| ---- | ------ | ------ |
| 0    | 0.05   | 0.00   |
| 3    | 0.01   | 0.13   |
| 6    | 0.04   | 0.12   |
| 9    | 0.01   | 0.05   |
| 12   | 0.07   | 0.06   |
| 15   | 0.03   | 0.01   |
| 18   | 0.05   | 0.06   |
| 21   | 0.12   | 0.06   |
| 24   | 0.21   | 0.03   |
</details>

Figure 22: Step-wise CRPS $_{sum}$ accuracy of Transformer. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method.

model. The results are shown in Table 13. We do not observe significant improvement from learnable lengthscales.

Table 13: Comparison of CRPS $_{sum}$ accuracy. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method. “w/(l)” indicates the lengthscales are learnable parameters. Boldface values indicate that models considering time-dependent errors have better performance. Mean and standard deviation are obtained from 10 runs of each model. 

<table><tr><td rowspan="2"></td><td colspan="3">GPVar</td><td colspan="3">Transformer</td></tr><tr><td>w/o</td><td>w/</td><td>w/(l)</td><td>w/o</td><td>w/</td><td>w/(l)</td></tr><tr><td>exchange_rate</td><td>0.0068±0.0004</td><td>0.0117±0.0004</td><td>0.0045±0.0001</td><td>0.0055±0.0002</td><td>0.0042±0.0002</td><td>0.0072±0.0002</td></tr><tr><td>solar</td><td>0.7103±0.0065</td><td>0.6929±0.0039</td><td>0.7727±0.0040</td><td>0.4960±0.0034</td><td>0.4132±0.0027</td><td>0.4138±0.0023</td></tr><tr><td>electricity</td><td>0.0430±0.0005</td><td>0.0403±0.0004</td><td>0.0351±0.0003</td><td>0.0494±0.0004</td><td>0.0638±0.0003</td><td>0.0858±0.0006</td></tr><tr><td>traffic</td><td>0.1095±0.0002</td><td>0.0649±0.0002</td><td>0.1297±0.0003</td><td>0.0717±0.0002</td><td>0.0981±0.0002</td><td>0.0950±0.0002</td></tr><tr><td>wiki</td><td>0.1745±0.0008</td><td>0.0743±0.0009</td><td>0.4839±0.0021</td><td>0.0841±0.0013</td><td>0.0500±0.0005</td><td>0.0472±0.0004</td></tr><tr><td>m4_hourly</td><td>0.0613±0.0004</td><td>0.0358±0.0002</td><td>0.0527±0.0003</td><td>0.0651±0.0004</td><td>0.0616±0.0003</td><td>0.0355±0.0003</td></tr><tr><td>m1_quarterly</td><td>0.3942±0.0030</td><td>0.3538±0.0017</td><td>0.3534±0.0017</td><td>0.4448±0.0027</td><td>0.4367±0.0028</td><td>0.3709±0.0120</td></tr><tr><td>pems03</td><td>0.0503±0.0001</td><td>0.0491±0.0002</td><td>0.0456±0.0001</td><td>0.0490±0.0001</td><td>0.0386±0.0001</td><td>0.0330±0.0001</td></tr><tr><td>uber_hourly</td><td>0.0342±0.0006</td><td>0.0222±0.0004</td><td>0.0218±0.0003</td><td>0.0632±0.0003</td><td>0.0513±0.0005</td><td>0.0969±0.0005</td></tr></table>

# B.6.2 Using Autocorrelations of an AR(p) process

One could parameterize $C_{t}$ as fully learnable, positive definite symmetric Toeplitz matrices. For instance, an AR(p) process has an autocorrelation matrix with a Toeplitz structure, allowing the modeling of negative correlations. This alternative approach may offer more flexibility in capturing complex correlation patterns in multivariate time series data. The autocorrelations of an AR(p) process can be obtained by solving a set of equations known as the Yule-Walker equations [46]. For example, if we consider an AR(2) process and let $\rho_{k}$ be the autocorrelation at lag k:

$$
z _ {t} = \phi_ {1} z _ {t - 1} + \phi_ {2} z _ {t - 2} + \epsilon_ {t}. \tag {32}
$$

where $\phi_{1}$ and $\phi_{2}$ are the coefficients. We have $\rho_{0}=1$ by definition and:

$$
\rho_ {1} = \phi_ {1} \rho_ {0} + \phi_ {2} \rho_ {1},
$$

$$
\dots \tag {33}
$$

$$
\rho_ {k} = \phi_ {1} \rho_ {k - 1} + \phi_ {2} \rho_ {k - 2}, k \geq 2.
$$

Since $\rho_0 = 1$ , we can solve for $\rho_1$ :

$$
\rho_ {1} = \frac {\phi_ {1}}{1 - \phi_ {2}}, \tag {34}
$$

and for any $k \geq 2$ , we can solve $\rho_{k}$ iteratively by:

$$
\rho_ {k} = \phi_ {1} \rho_ {k - 1} + \phi_ {2} \rho_ {k - 2}, k \geq 2. \tag {35}
$$

The collection $\{\rho_{0},\rho_{1},\ldots,\rho_{k},\ldots,\rho_{D-1}\}$ forms the first row or column of a Toeplitz matrix and can be used to parameterize $C_{t}$ . We perform a hyperparameter search to find the best AR order p based on the validation loss. As shown in Table 14, while the correlation matrix $C_{t}$ parameterized by an AR process shows promise in modeling both positive and negative correlations, it does not empirically provide an overall improvement compared to the kernel method used in this paper. This may be because cross-correlations in time series are predominantly positive. However, the AR method does show significant improvements on certain datasets where the kernel method does not perform well. For example, the AR method greatly improves GPVar on exchange\_rate and the Transformer on electricity.

Table 14: Comparison of CRPS $_{sum}$ accuracy. “w/o” denotes methods without time-dependent errors, while “w/” indicates our method. “w/(AR)” indicates $C_{t}$ is parameterized by an AR process. Boldface values indicate that models considering time-dependent errors have better performance. Mean and standard deviation are obtained from 10 runs of each model. 

<table><tr><td rowspan="2"></td><td colspan="3">GPVar</td><td colspan="3">Transformer</td></tr><tr><td>w/o</td><td>w/</td><td>w/(AR)</td><td>w/o</td><td>w/</td><td>w/(AR)</td></tr><tr><td>exchange_rate</td><td>0.0068±0.0004</td><td>0.0117±0.0004</td><td>0.0051±0.0002</td><td>0.0055±0.0002</td><td>0.0042±0.0002</td><td>0.0088±0.0004</td></tr><tr><td>solar</td><td>0.7103±0.0065</td><td>0.6929±0.0039</td><td>0.5923±0.0042</td><td>0.4960±0.0034</td><td>0.4132±0.0027</td><td>0.3362±0.0025</td></tr><tr><td>electricity</td><td>0.0430±0.0005</td><td>0.0403±0.0004</td><td>0.0433±0.0007</td><td>0.0494±0.0004</td><td>0.0638±0.0003</td><td>0.0252±0.0002</td></tr><tr><td>traffic</td><td>0.1095±0.0002</td><td>0.0649±0.0002</td><td>0.1095±0.0004</td><td>0.0717±0.0002</td><td>0.0981±0.0002</td><td>0.0878±0.0003</td></tr><tr><td>wiki</td><td>0.1745±0.0008</td><td>0.0743±0.0009</td><td>0.2375±0.0013</td><td>0.0841±0.0013</td><td>0.0500±0.0005</td><td>0.0512±0.0008</td></tr><tr><td>m4_hourly</td><td>0.0613±0.0004</td><td>0.0358±0.0002</td><td>0.0298±0.0002</td><td>0.0651±0.0004</td><td>0.0616±0.0003</td><td>0.0680±0.0003</td></tr><tr><td>m1_quarterly</td><td>0.3942±0.0030</td><td>0.3538±0.0017</td><td>0.1692±0.0029</td><td>0.4448±0.0027</td><td>0.4367±0.0028</td><td>0.4348±0.0028</td></tr><tr><td>pems03</td><td>0.0503±0.0001</td><td>0.0491±0.0002</td><td>0.0787±0.0002</td><td>0.0490±0.0001</td><td>0.0386±0.0001</td><td>0.0656±0.0001</td></tr><tr><td>uber_hourly</td><td>0.0342±0.0006</td><td>0.0222±0.0004</td><td>0.0375±0.0004</td><td>0.0632±0.0003</td><td>0.0513±0.0005</td><td>0.0770±0.0007</td></tr></table>

# B.7 Alternative Error Assumptions

A more suitable likelihood function can regularize the training process, potentially reducing residual correlations. For example, assuming the errors follow a multivariate t-distribution improves the robustness of the model to outliers. Additionally, a stronger base model can help produce residuals that are more independent. Based on these considerations, we designed our approach to adapt dynamically to varying levels of error correlation. The weighted correlation matrix assigns greater weight to the identity matrix when the errors exhibit lower correlation.

We also trained the baseline models using the likelihood of the multivariate t-distribution, and the results are shown in Table 15. While using an alternative distribution can lead to better performance on certain datasets when our method is not applied, we observed that our method effectively closes the performance gap in cases where the multivariate Gaussian assumption is outperformed by the t-distribution.

An important feature of our method is the ability to use a subset of time series in each training batch for model optimization, which enhances scalability. For the multivariate t-distribution, the distribution of these subsets of $z_{t}$ should have the same degrees of freedom as the full distribution of $z_{t}$ . However, since the degrees of freedom are treated as an additional output of the model in each training batch, they are not guaranteed to be consistent across batches. While this is not problematic for deep learning, it violates the marginalization property of the t-distribution from a statistical standpoint.

We chose Gaussian noise for its beneficial properties, including its marginalization rule and well-defined conditional distribution, both essential for statistically consistent model training and reliable inference. To address model misspecification, a more effective approach could involve first transforming the original observations into Gaussian-distributed data using a Gaussian Copula $[3]$ , and then applying our method.

Table 15: CRPS $_{sum}$ accuracy comparison. "w/o" denotes methods without time-dependent errors, while "w/" indicates our method. Bold values show models with time-dependent errors performing better. Mean and standard deviation are obtained from 10 runs of each model. "N/A" indicates that the model could not be properly fitted.. 

<table><tr><td rowspan="2"></td><td colspan="3">GPVar</td><td colspan="3">Transformer</td></tr><tr><td>Gaussian (w/o)</td><td>Gaussian (w/)</td><td>t-distribution (w/o)</td><td>Gaussian (w/o)</td><td>Gaussian (w/)</td><td>t-distribution (w/o)</td></tr><tr><td>exchange_rate</td><td>0.0068±0.0004</td><td>0.0117±0.0004</td><td>0.0159±0.0005</td><td>0.0055±0.0002</td><td>0.0042±0.0002</td><td>0.0101±0.0003</td></tr><tr><td>solar</td><td>0.7103±0.0065</td><td>0.6929±0.0039</td><td>N/A</td><td>0.4960±0.0034</td><td>0.4132±0.0027</td><td>N/A</td></tr><tr><td>electricity</td><td>0.0430±0.0005</td><td>0.0403±0.0004</td><td>0.0467±0.0004</td><td>0.0494±0.0004</td><td>0.0638±0.0003</td><td>0.0466±0.0002</td></tr><tr><td>traffic</td><td>0.1095±0.0002</td><td>0.0649±0.0002</td><td>0.0679±0.0002</td><td>0.0717±0.0002</td><td>0.0981±0.0002</td><td>N/A</td></tr><tr><td>wikipedia</td><td>0.1745±0.0008</td><td>0.0743±0.0009</td><td>0.0730±0.0004</td><td>0.0841±0.0013</td><td>0.0500±0.0005</td><td>0.1979±0.0005</td></tr><tr><td>m4_hourly</td><td>0.0613±0.0004</td><td>0.0358±0.0002</td><td>0.0365±0.0003</td><td>0.0651±0.0004</td><td>0.0616±0.0003</td><td>0.0665±0.0003</td></tr><tr><td>m1_quarterly</td><td>0.3942±0.0030</td><td>0.3538±0.0017</td><td>0.3550±0.0084</td><td>0.4448±0.0027</td><td>0.4367±0.0028</td><td>0.4466±0.0044</td></tr><tr><td>pems03</td><td>0.0503±0.0001</td><td>0.0491±0.0002</td><td>0.0679±0.0002</td><td>0.0490±0.0001</td><td>0.0386±0.0001</td><td>0.0529±0.0002</td></tr><tr><td>uber_hourly</td><td>0.0342±0.0006</td><td>0.0222±0.0004</td><td>0.0666±0.0010</td><td>0.0632±0.0003</td><td>0.0513±0.0005</td><td>0.0340±0.0004</td></tr></table>

# B.8 Qualitative Results on Forecasting

In this section, we provide qualitative analysis of the actual prediction performance by visualizing the predictions.

![](images/fcdd93154d0084db80bd80a97ae7a58d7722ff28b19b75bbed41e56f97c438e5.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0.80  |
| 0         | 0.80  |
| 20        | 0.80  |
</details>

![](images/4efa411c94d3392933fa08de6dcfbb6cf321610cba34873716d413622e89e4ba.jpg)

<details>
<summary>line</summary>

| Time Step | Value (Blue Line) | Value (Orange Line) |
| --------- | ----------------- | ------------------- |
| -20       | 1.6               | 1.6                 |
| 0         | 1.6               | 1.6                 |
| 20        | 1.6               | 1.7                 |
</details>

![](images/e0eea82b60a17d9066b58bcc5379aeebcce7bbeb1c63f1b382a806b2bc2dfc2f.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -25       | 1.05  |
| 0         | 1.00  |
| 25        | 1.05  |
</details>

![](images/a4dc1f2cefd5e6dc533984499243656d08064e1caced21a3705314563ca0b4d3.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0.16  |
| 0         | 0.16  |
| 20        | 0.16  |
</details>

![](images/4beef25af8a0679752f67afa2f0503a7d370a5714cae610825eca27804aadd76.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0.8   |
| 0         | 0.8   |
| 20        | 0.8   |
</details>

![](images/f60a2aa85661560c4545b9193b6fd90204af01a1ba2ae2fbdda3f7b80455e7a3.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 1.0   |
| 0         | 1.0   |
| 20        | 1.0   |
</details>

![](images/eb1a402b67aa267ad24086f883e85a83faf14157d7e19588287a87c8ad63d7de.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -25       | 1.0   |
| 0         | 1.0   |
| 25        | 1.0   |
</details>

![](images/6844e162f4ab7f00be1652d3bd539a263ebdb887ccb61c8999718ccd21d578ef.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0.8   |
| 0         | 0.8   |
| 20        | 0.8   |
</details>

![](images/6541821a49252bd37a62a518ba7be5272393607a5d6d1e5c6b0700b3a92be856.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 1.0   |
| 0         | 1.0   |
| 20        | 1.1   |
</details>

![](images/53a3a26ac08bed12b9b7726ecd6a31af66bda24406457f2a24bb79fe909eb3bc.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0.80  |
| 0         | 0.80  |
| 20        | 0.83  |
</details>

![](images/c02fae2591ff98b31793a802f62568c354492be2e9da274089f52c2e8cdcd237.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -25       | 1.0   |
| 0         | 1.0   |
| 25        | 1.1   |
</details>

![](images/bb0d5b8b4a87301919c890b39f76cae32a4bf672a56d484b6ddf7fbd8ecb8ae0.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0.80  |
| 0         | 0.82  |
| 20        | 0.83  |
</details>

observed predicted

Figure 23: Visualization of forecasting results on exchange\_rate using GPVar with our method.

![](images/4b2815df492697dffb6c59c9a07d4d0104aea9fc0bee621b3d76cfe06a438342.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0     |
| -10       | 150   |
| 0         | 0     |
| 10        | 150   |
| 20        | 0     |
</details>

![](images/2a53968156b46df71af5f4e2c47a206ede6bb5548b9616dc4eae5b3314330b57.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0     |
| 0         | 0     |
| 20        | 0     |
</details>

![](images/62071032a90dbfb278e502cfd9968726513cf9781d7491191861e2202c292ca6.jpg)

<details>
<summary>line</summary>

| Time Step | Value (Blue Line) | Value (Orange Line) |
| --------- | ----------------- | ------------------- |
| -20       | 0                 | 0                   |
| -10       | 100               | 0                   |
| 0         | 0                 | 0                   |
| 10        | 0                 | 0                   |
| 20        | 150               | 140                 |
</details>

![](images/96f6185f683f8a48593e4b952abdb17dab13404a35e9e149b3d7e52873cfaf57.jpg)

<details>
<summary>line</summary>

| Time Step | Value (Blue) | Value (Orange) |
| --------- | ------------ | -------------- |
| -20       | 0            | 0              |
| -15       | 60           | 0              |
| -10       | 50           | 0              |
| -5        | 0            | 0              |
| 0         | 0            | 0              |
| 5         | 0            | 0              |
| 10        | 0            | 80             |
| 15        | 0            | 90             |
| 20        | 0            | 0              |
</details>

![](images/ff0e6a2e7a7eb803a77e29ccffe10277c9e40f73e8f30930b8deb06eb1e2dccb.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0     |
| 0         | 0     |
| 20        | 150   |
</details>

![](images/2aae31fce22f2e4b47c3b21155ee0475c7869f681efd5bd92c13dfd4d1e815d4.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0     |
| -10       | 50    |
| 0         | 0     |
| 10        | 0     |
| 20        | 150   |
</details>

![](images/be3b9a7d26ac7acc0ebb9756053018d1d095bfb8cdd5a9e983734ff38274c7d7.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0     |
| -10       | 120   |
| 0         | 0     |
| 10        | 130   |
| 20        | 0     |
</details>

![](images/7d61e986a92999279114b4027ccb0028a2890a161e67004d3048b3941257eacf.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0     |
| -10       | 300   |
| 0         | 0     |
| 10        | 0     |
| 20        | 400   |
</details>

![](images/2307fdaded8b28754309b8a7da69b9c3239c403ed971808cbd2bff38cdcd5e3a.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0     |
| -10       | 50    |
| 0         | 0     |
| 10        | 100   |
| 20        | 0     |
</details>

![](images/a6bb07c02df98cfbd57df14729056c6b0bf50a149c4dfed43f6f8f2ef3e00721.jpg)

<details>
<summary>line</summary>

| Time Step | Value (Blue Line) | Value (Orange Line) |
| --------- | ----------------- | ------------------- |
| -20       | 0                 | 0                   |
| -10       | 100               | 0                   |
| 0         | 0                 | 0                   |
| 10        | 150               | 150                 |
| 20        | 0                 | 0                   |
</details>

![](images/b3b9529e37a886f14c63a080cb00a43be43ac26c8f85449442e2f59e0d83f688.jpg)

<details>
<summary>line</summary>

| Time Step | Value (Blue Line) | Value (Orange Line) |
| --------- | ----------------- | ------------------- |
| -20       | ~0                | ~0                  |
| 0         | ~0                | ~0                  |
| 10        | ~30               | ~120                |
| 20        | ~0                | ~0                  |
</details>

![](images/32a41477413ba2d73c094ebdf4a0cfecd6b05c036f7fe544c5960563a1c96fc5.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0     |
| -10       | 100   |
| 0         | 0     |
| 10        | 50    |
| 20        | 150   |
</details>

observed predicted

Figure 24: Visualization of forecasting results on solar using GPVar with our method.   
![](images/d139f5dbeb04c6c2c9f06b5a8a7956cbf7e3c403e78afff28934dc1aad17637d.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 400   |
| 0         | 300   |
| 20        | 400   |
</details>

![](images/61a907583af6ac7dbfe141733231db7ef34e6bec5657505595de485c357518c7.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 50    |
| 0         | 30    |
| 20        | 60    |
</details>

![](images/b70364fa4c177a9e66b219221267832886ca0389539f70bdc0146a824769a5f3.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 25    |
| 0         | 10    |
| 20        | 15    |
</details>

![](images/8769b4684e0a974573e73ea0ca148bb883d0b2dc18ad9ce81afeccdbadfb4a5c.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 35    |
| -10       | 30    |
| 0         | 50    |
| 10        | 35    |
| 20        | 60    |
</details>

![](images/5ce18bb619d22abe455f7d48c877d8f5bac8bf5b802f8d4e676b47ba59b21b90.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 3000  |
| 0         | 3000  |
| 20        | 4000  |
</details>

![](images/9fd590b575d7ad2ca9dcff551264bb9c9fd1a33a254e03c20e54d1828c8db341.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 800   |
| -10       | 1500  |
| 0         | 1400  |
| 10        | 1000  |
| 20        | 1800  |
</details>

![](images/49a77faabb499c2b00e192c3e4ff77e82c04bcfb41225b2c2a53c957bc9e7e05.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | ~500  |
| -10       | ~1500 |
| 0         | ~500  |
| 10        | ~1500 |
| 20        | ~500  |
</details>

![](images/cf43346bd361cbe1e0914cc94e6d4d266b36b5bfa79fe3a8325c9ec0b0e86bef.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 2500  |
| 0         | 3000  |
| 20        | 3000  |
</details>

![](images/f7283580d58edc043170314a08981e3240f706e930203abe3df85e8e1aa00a42.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 250   |
| 0         | 150   |
| 20        | 350   |
</details>

![](images/ac971e81c30e551cf8139e0e03804149809d9b1b47ecb3aaeb4e4d5d33371bc7.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 0     |
| 0         | 0     |
| 20        | 40    |
</details>

![](images/81916680d71d226940a95ad7fcc9509733e4142c38625fbd424b2f33f08c4500.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 50    |
| 0         | 60    |
| 20        | 80    |
</details>

![](images/5f304b4d2bed2f47d5aee36909f0250b1640ae303f6bf416871a72fa4df1b907.jpg)

<details>
<summary>line</summary>

| Time Step | Value |
| --------- | ----- |
| -20       | 500   |
| 0         | 600   |
| 20        | 700   |
</details>

observed predicted

Figure 25: Visualization of forecasting results on electricity using GPVar with our method.

![](images/a0bbb32ec993666127ed612031e5c7ff3a41f741261d360b40156ca64fce16c3.jpg)  
observed predicted

Figure 26: Visualization of forecasting results on traffic using GPVar with our method.   
![](images/062175af18df0207e2e74bd578b5c0c63072905bbd2ea5e9b86ed7d2bc7a4b93.jpg)  
observed predicted

Figure 27: Visualization of forecasting results on wiki using GPVar with our method.

![](images/9b4ce76f0979e86a4ee28527d5cbe020a79decf8fd4da6e7eac891b05388e7c3.jpg)  
observed predicted

Figure 28: Visualization of forecasting results on m4\_hourly using GPVar with our method.   
![](images/f33a295f1723fb2388dde121b13ef73135eeba9d087f3f118be5a28e02402001.jpg)  
observed predicted

Figure 29: Visualization of forecasting results on pems03 using GPVar with our method.

![](images/1ec98583cbca29b0ec3840bb65bcb3a85b75cddd21268b6b6741c74ace114a57.jpg)  
observed predicted

Figure 30: Visualization of forecasting results on uber\_hourly using GPVar with our method.

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and/or introduction have clearly stated the claims made, including the contributions made in the paper and important assumptions and limitations.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.   
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have discussed the limitations of the work in the "Conclusion" section.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
- The authors are encouraged to create a separate "Limitations" section in their paper.   
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.   
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [NA]

Justification: We do not have theoretical result in this study.

# Guidelines:

- The answer NA means that the paper does not include theoretical results.   
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.   
- All assumptions should be clearly stated or referenced in the statement of any theorems.   
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.   
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
- Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We have fully disclosed all the information needed to reproduce the main experimental results of the paper in the Appendix.

# Guidelines:

- The answer NA means that the paper does not include experiments.   
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.   
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general, releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.   
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example   
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.   
(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.   
(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).   
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

# Answer: [No]

Justification: The code will be released after the paper is accepted. However, we have provided a sufficient amount of experimental details in the Appendix.

# Guidelines:

- The answer NA means that paper does not include experiments requiring code.   
- Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
- While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.   
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).   
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

# Answer: [Yes]

Justification: The paper specified all the training and test details necessary to understand the results.

# Guidelines:

- The answer NA means that the paper does not include experiments.   
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.   
- The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

# Answer: [Yes]

Justification: We ran all of our experiments for 10 times to calculate the standard deviation.

# Guidelines:

- The answer NA means that the paper does not include experiments.   
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.   
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).   
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
- The assumptions made should be given (e.g., Normally distributed errors).

- It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a $96\%$ CI, if the hypothesis of Normality of errors is not verified.   
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper has indicated the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

Guidelines:

- The answer NA means that the paper does not include experiments.   
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

# 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We have discussed societal impacts in the last section of this paper.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.   
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.   
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The creators or original owners of assets (e.g., code, data, models), used in the paper, have been properly credited. The license and terms of use have been explicitly mentioned and properly respected.

Guidelines:

- The answer NA means that the paper does not use existing assets.   
- The authors should cite the original paper that produced the code package or dataset.   
- The authors should state which version of the asset is used and, if possible, include a URL.   
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.   
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.   
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.   
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

# 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.   
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
- The paper should discuss whether and how consent was obtained from people whose asset is used.   
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.