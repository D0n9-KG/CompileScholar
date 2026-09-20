# Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks

Steven Adriaensen\*

Machine Learning Lab

University of Freiburg

adriaens@cs.uni-freiburg.de

Herilalaina Rakotoarison\*

Machine Learning Lab

University of Freiburg

rakotoah@cs.uni-freiburg.de

Samuel Müller

Machine Learning Lab

University of Freiburg

muellesa@cs.uni-freiburg.de

Frank Hutter

Machine Learning Lab

University of Freiburg

fh@cs.uni-freiburg.de

# Abstract

Learning curve extrapolation aims to predict model performance in later epochs of training, based on the performance in earlier epochs. In this work, we argue that, while the inherent uncertainty in the extrapolation of learning curves warrants a Bayesian approach, existing methods are (i) overly restrictive, and/or (ii) computationally expensive. We describe the first application of prior-data fitted neural networks (PFNs) in this context. A PFN is a transformer, pre-trained on data generated from a prior, to perform approximate Bayesian inference in a single forward pass. We propose LC-PFN, a PFN trained to extrapolate artificial right-censored learning curves generated from a parametric prior proposed in prior art using MCMC. We demonstrate that LC-PFN can approximate the posterior predictive distribution over learning curves more accurately than MCMC, while being over 10 000 times faster. We also show that the same LC-PFN achieves competitive performance extrapolating a total of 20 000 real learning curves from four learning curve benchmarks (LCBench, NAS-Bench-201, Taskset, and PD1) that stem from training a wide range of model architectures (MLPs, CNNs, RNNs, and Transformers) on 53 different datasets with varying input modalities (tabular, image, text, and protein data). Finally, we investigate its potential in the context of model selection and find that a simple LC-PFN based predictive early stopping criterion obtains 2 - 6× speed-ups on 45 of these datasets, at virtually no overhead.

# 1 Introduction

Learning curve extrapolation [Mohr and van Rijn, 2022] aims to predict how much a machine learning model will improve with more training, e.g., to determine how much more training data to collect [Cortes et al., 1993, Frey and Fisher, 1999, Leite and Brazdil, 2004, Kolachina et al., 2012], or to define an early stopping criterion in online learning [Yao et al., 2007]. Learning curve extrapolation has recently been widely studied to speed up automated machine learning (AutoML) and hyperparameter optimization (HPO) of deep neural networks, by discarding non-promising configurations early [Swersky et al., 2014, Domhan et al., 2015, Klein et al., 2017, Baker et al., 2017, Chandrashekaran and Lane, 2017, Gargiani et al., 2019, Wistuba et al., 2022].

Despite these efforts, learning curve extrapolation is not yet widely adopted in practice, e.g., state-of-the-art multi-fidelity hyperparameter optimization techniques, such as BOHB [Falkner et al., 2018], still rely on successive halving [Li et al., 2017], i.e., the crude heuristic that learning curves mostly do not cross each other.

One reason for this is that, while many learning curves are well-behaved, some exhibit chaotic behavior and are intrinsically difficult to predict accurately [Choi et al., 2018]. In this setting, Bayesian approaches [Swersky et al., 2014, Domhan et al., 2015, Klein et al., 2017, Wistuba et al., 2022], which also quantify the reliability of their extrapolation, show great potential. However, existing methods for Bayesian inference either (i) put strong restrictions on the prior, and are incapable of modeling the variable nature of learning curves, or (ii) are too computationally expensive, limiting their practical applicability. Furthermore, most of this related work focuses on demonstrating the potential that learning curve extrapolation has to accelerate downstream AutoML/HPO tasks, yet fails to fully investigate the quality of the extrapolations themselves, and to quantify the approach's ability to handle the heterogeneity of real-world learning curves, e.g., varying performance metrics, curve shapes, divergence, heteroscedastic noise, etc.

In this work, we investigate the potential of learning curve extrapolation using prior-data fitted networks (PFNs), a meta-learned approximate Bayesian inference method recently proposed by Müller et al. [2022]. PFNs combine great flexibility with efficient and accurate approximation of the posterior predictive distribution (PPD) in a single forward pass of a transformer [Vaswani et al., 2017] trained on artificial data from the prior only. As PFNs are a promising alternative to Markov Chain Monte Carlo (MCMC) for approximating Bayesian inference, we compare our approach (LC-PFN) to the MCMC approach for learning curve extrapolation of Domhan et al. [2015], taking into account both the quality and the cost of PPD approximation.

In summary, our contributions are as follow:

- We are the first to apply PFNs to an extrapolation task, introducing LC-PFN, the first PFN for learning curve extrapolation.   
- We demonstrate that LC-PFN can be more than $10000 \times$ faster than MCMC while still yielding better probabilistic extrapolations.   
- We show that LC-PFN does not only yield better probabilistic extrapolations on prior samples, but also on real learning curves of a wide range of architectures (MLPs, CNNs, RNNs, Transformers) on varying input modalities (tabular, image, text and protein data).   
- We demonstrate the practical usefulness of LC-PFN to construct an early stopping criterion that achieves 2 - 6× speedups over baselines.   
- To facilitate reproducibility and allow others to build on our work, we open-source all code, data, and models used in our experiments at https://github.com/automl/lcpfn.

# 2 Related work

Learning curves and how to use them for decision-making has been an active research area, as recently surveyed by Mohr and van Rijn [2022]. Most related work considers point estimates of the curve or a specific property thereof [Cortes et al., 1993, Frey and Fisher, 1999, Kolachina et al., 2012, Baker et al., 2017, Kaplan et al., 2020], or follows a non-Bayesian approach to quantify uncertainty [Chandrashekaran and Lane, 2017, Gargiani et al., 2019].

Only a few works have explored Bayesian learning curve extrapolation. For example, the Freeze-Thaw Bayesian optimization method Swersky et al. [2014] used a Gaussian process (GP) as a joint model of learning curves and hyperparameters to decide what learning run to continue for a few epochs (or whether to start a new one). The model is then dynamically updated to fit the partial learning curve data. Training data grows quickly since each performance observation of the curve is treated as a datapoint, making exact GPs intractable, and thus the work relies on approximate GPs. Furthermore, their approach makes strong (prior) assumptions. On top of the standard GP assumptions, they used a specialized kernel assuming exponential growth to improve extrapolation. Domhan et al. [2015] proposed a less restrictive parametric prior (see Section 3.2 for more details) and used the gradient-free MCMC method from Foreman-Mackey et al. [2013] as approximate inference method. While MCMC is a very general approach, it can be sensitive to its hyperparameters (e.g.,

burn-in period, chain length, etc.) and, as we will show in Section 4, generating sufficient samples to reliably approximate the PPD may impose significant overhead. Klein et al. [2017] extended this parametric prior to also capture the effect of hyperparameter settings. In particular, they used a Bayesian neural network with a specialized learning curve layer, and trained this network using gradient-based MCMC on learning curve data from previously tested hyperparameter settings. While this approach is able to predict learning curves of previously unseen configurations, conditioning on the current partial learning curve requires retraining the Bayesian neural network online, which is costly. Recently, DyHPO [Wistuba et al., 2022] followed a similar dynamic HPO setup as Swersky et al. [2014], but used deep GPs [Damianou and Lawrence, 2013]. While deep GPs relax some of the standard GP assumptions, extrapolation abilities were not thoroughly analyzed, and DyHPO only predicts one epoch into the future. Finally, it is worth noting that, except for Domhan et al. [2015], all the aforementioned probabilistic approaches [Swersky et al., 2014, Klein et al., 2017, Chandrashekaran and Lane, 2017, Gargiani et al., 2019, Wistuba et al., 2022] utilize meta-learning across the learner's hyperparameter settings. While this is an interesting line of work, it limits applicability, and introduces confounding factors. We will therefore consider a simpler and more general setting in this work (see Section 3.1). Indeed, akin to the approach of Domhan et al. [2015], we operate without the assumption of access to data from previous runs employing different hyperparameter settings, nor do we assume the ability to generalize across these settings. Our results highlight that prior-data fitted networks (PFNs) offer a significantly more efficient and practical alternative to Markov Chain Monte Carlo (MCMC) methods. As categorized by Mohr and van Rijn [2022], Domhan et al. [2015] is the only comparable prior work within this category. This underlines the novelty and importance of our approach in the context of advancing current methodologies.

While we are the first to apply PFNs [Müller et al., 2022] to learning curve extrapolation, PFNs have previously been applied in different settings: Hollmann et al. [2023] used them to meta-learn a classifier for tabular data; Müller et al. [2023] as a surrogate model for Bayesian optimization; and most recently concurrent work by Dooley et al. [2023] as a zero-shot time series forecaster.

# 3 Methods

# 3.1 Bayesian learning curve extrapolation

Let $y_{t} \in [0,1]$ represent the model performance (e.g., validation accuracy) at training step $t \in \{1, \ldots, m\}$ . The problem we consider in this paper can be formulated as follows: Given a partial learning curve $y_{1}, \ldots, y_{T}$ up to some cutoff $T$ , and a prior distribution $p(\boldsymbol{y})$ over learning curves, approximate the posterior predictive distribution (PPD) $q(y_{t'} \mid y_{1}, \ldots, y_{T})$ for $T < t' \leq m$ . We will further assume that we can calculate the relative probability density of $p(\boldsymbol{y})$ , a requirement for MCMC, and that we can generate samples from $p(\boldsymbol{y})$ , a requirement for PFNs. Figure 1 provides an illustration of Bayesian learning curve extrapolation, showcasing the posterior predictive distributions (PPDs) of the extrapolated curves generated by LC-PFN and MCMC, along with a few representative curves sampled from the prior distribution $p(\boldsymbol{y})$ .

![](images/29dea513b4fe4cf9af23d7dd5623cd43b8db6c418517b00e19ce9e4b7c62fb4e.jpg)

<details>
<summary>line</summary>

| Epochs | target | data | LC-PFN | MCMC |
| ------ | ------ | ---- | ------ | ---- |
| 0      | 0.0    | 0.0  | 0.0    | 0.0  |
| 5      | 0.8    | 0.5  | 0.8    | 0.8  |
| 10     | 0.85   | 0.45 | 0.85   | 0.85 |
| 15     | 0.85   | 0.35 | 0.85   | 0.85 |
| 20     | 0.85   | 0.45 | 0.85   | 0.85 |
| 25     | 0.85   | 0.45 | 0.85   | 0.85 |
| 30     | 0.85   | 0.45 | 0.85   | 0.85 |
| 35     | 0.85   | 0.45 | 0.85   | 0.85 |
| 40     | 0.85   | 0.45 | 0.85   | 0.85 |
| 45     | 0.85   | 0.45 | 0.85   | 0.85 |
| 50     | 0.85   | 0.45 | 0.85   | 0.85 |
</details>

![](images/e5287263158d1c0e6e3731788fa5bbc1ae67bd23378552124f0b12d3c42c3e89.jpg)

<details>
<summary>line</summary>

| t   | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 | Series 6 | Series 7 | Series 8 | Series 9 |
| --- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 0   | 0.5      | 0.4      | 0.3      | 0.2      | 0.1      | 0.1      | 0.1      | 0.1      | 0.1      |
| 20  | 0.7      | 0.5      | 0.4      | 0.3      | 0.2      | 0.2      | 0.2      | 0.2      | 0.2      |
| 40  | 0.75     | 0.55     | 0.45     | 0.35     | 0.25     | 0.25     | 0.25     | 0.25     | 0.25     |
| 60  | 0.8      | 0.6      | 0.5      | 0.4      | 0.3      | 0.3      | 0.3      | 0.3      | 0.3      |
| 80  | 0.82     | 0.62     | 0.52     | 0.42     | 0.32     | 0.32     | 0.32     | 0.32     | 0.32     |
| 100 | 0.85     | 0.65     | 0.55     | 0.45     | 0.35     | 0.35     | 0.35     | 0.35     | 0.35     |
</details>

Figure 1: (Left) Visualization of Bayesian learning curve extrapolation. The plot shows the median and the 90% confidence interval of the PPDs inferred using MCMC and LC-PFN, given two partial empirical learning curves of 10 and 20 epochs, respectively, and the prior described in Section 3.2. (Right) Example of learning curves sampled from the prior $p(\boldsymbol{y})$ .

While the fixed ranges for $y_{t}$ and t are well-suited for modeling particular learning curves (e.g., accuracy over epochs), they also are restrictive. In Appendix A, we discuss the invertible normalization procedure we apply to support extrapolating, possibly diverging, iteration-based learning curves across a broad range of performance metrics (e.g., log loss).

# 3.2 Learning curve prior

Following Domhan et al. [2015], we model y as a linear combination of K basis growth curves $f_{k}$ , each parameterized by $\theta_{k}$ , and i.i.d. additive Gaussian noise with variance $\sigma^{2}$ , i.e.,

$$
y _ {t} \sim \mathcal {N} (f _ {\mathrm{comb}} (t | \boldsymbol {\xi}), \sigma^ {2}) \quad \text { with } \quad f _ {\mathrm{comb}} (t | \boldsymbol {\xi}) = \sum_ {k = 1} ^ {K} w _ {k} \cdot f _ {k} (t | \boldsymbol {\theta} _ {k}),
$$

where we assume our model parameters

$$
\boldsymbol {\xi} = (w _ {1}, \dots , w _ {K}, \boldsymbol {\theta} _ {1}, \dots , \boldsymbol {\theta} _ {K}, \sigma^ {2})
$$

to be random variables with prior $p(\pmb{\xi})$ . Here, Domhan et al. [2015] assumed an uninformative prior (i.e., $p(\pmb{\xi}) \propto 1$ ), with the exception of some hard constraints.

We adopt a strictly more informative prior, because (i) the original prior puts almost all probability mass on parameterizations yielding invalid learning curves, e.g., $y_{t} \notin [0,1]$ ; and (ii) we cannot practically sample from this prior having unbounded support, a requirement for PFNs. $^{1}$ Specifically, to mimic realistic learning curves we use bounded uniform weight priors $w_{k} \sim \mathcal{U}(0,1)$ , a low-noise prior $\log(\sigma^{2}) \sim \mathcal{N}(-8,2)$ , only allow curves with values in [0,1] and, like Domhan et al. [2015], only accept curves whose last value is higher than its first. Putting all of these together, our prior distribution thus takes the form:

$$
p (\boldsymbol {\xi}) \propto \left(\prod_ {k = 1} ^ {K} p (w _ {k}) \cdot p (\boldsymbol {\theta} _ {k})\right) \times p (\sigma^ {2}) \times \mathbb {1} \left(f _ {\text {comb}} (1 | \boldsymbol {\xi}) <   f _ {\text {comb}} (m | \boldsymbol {\xi})\right) \times \left(\prod_ {t = 1} ^ {m} \mathbb {1} \left(f _ {\text {comb}} (t | \boldsymbol {\xi}) \in [ 0, 1 ]\right)\right).
$$

Finally, we limit ourselves to three parametric families of learning curves (K = 3, see Table 1). $^{2}$ These basis curves were chosen to capture a variety of growth trends and convergence behavior. We show examples of curves sampled from this prior in Figure 1 (right).

Table 1: Formulas of the three parametric basis curves and priors over their parameters. 

<table><tr><td>Reference name</td><td>Formula  $f_{k}(t)$ </td><td>Prior  $p(\boldsymbol{\theta}_{k})$ </td></tr><tr><td> $pow_{3}$ </td><td> $c - at^{-\alpha}$ </td><td> $c \sim \mathcal{U}(0,1.25) \quad a \sim \mathcal{U}(-0.6,0.6) \quad \log(\alpha) \sim \mathcal{N}(0,4)$ </td></tr><tr><td>Janoschek</td><td> $\alpha - (\alpha - \beta)e^{-\kappa t^{\delta}}$ </td><td> $\alpha \sim \mathcal{U}(0,1) \quad \beta \sim \mathcal{U}(0,2) \quad \log(\kappa) \sim \mathcal{N}(-2,1) \quad \log(\delta) \sim \mathcal{N}(0,0.25)$ </td></tr><tr><td> $ilog_{2}$ </td><td> $c - \frac{a}{\log(t+1)}$ </td><td> $c \sim \mathcal{U}(0,1) \quad a \sim \mathcal{U}(-0.5,0.5)$ </td></tr></table>

# 3.3 Prior-data fitted networks (PFNs)

In this paper, we propose to use prior-data fitted networks (PFNs, Müller et al., 2022) instead of MCMC for learning curve extrapolation. PFNs are neural networks trained to perform approximate Bayesian prediction for supervised learning settings. That is, PFNs are trained to predict some output $y \in \mathbb{R}$ , conditioned on an input $t$ and a training set $D_{train}$ of given input-output examples. The PFN is trained for this task with samples obtained from a prior over datasets $p(\mathcal{D})$ . The loss function for training a PFN $q_{\theta}$ with parameters $\theta$ is the cross entropy $\ell_{\theta} = \mathbb{E}_{(t,y) \cup D_{train} \sim p(\mathcal{D})} [-\log q_{\theta}(y|t, D_{train})]$ for predicting the hold-out example's label $y$ , given $t$ and $D_{train}$ . Müller et al. [2022] proved that minimizing this loss over many sampled tasks $(t,y) \cup D_{train}$ directly coincides with minimizing the KL divergence between the PFN's predictions and the true PPD. In essence, the PFN meta-learns to perform approximate posterior inference on (meta-train) synthetic tasks sampled from the prior, and at inference time also does so for a (meta-test) real task.

![](images/6ffcf8ae8043183cf7531e27c06909de1b634c37d821f4ac0cf8f3e052c3554f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["epoch: 1\nacc: 10%"] --> B["Q"]
    C["epoch: 2\nacc: 26.6%"] --> D["Q"]
    E["epoch: 3\nacc: 51.4%"] --> F["Q"]
    G["epoch: 4"] --> H["Q"]
    I["epoch: 5"] --> J["Q"]
    B --> K["q(·|4, D)"]
    D --> L["q(·|4, D)"]
    F --> M["q(·|4, D)"]
    H --> N["q(·|5, D)"]
    J --> O["q(·|5, D)"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style F fill:#ccf,stroke:#333
    style H fill:#ccf,stroke:#333
    style J fill:#ccf,stroke:#333
    style K fill:#cfc,stroke:#333
    style L fill:#cfc,stroke:#333
    style M fill:#cfc,stroke:#333
    style O fill:#fcc,stroke:#333
```
</details>

Figure 2: A visualization of our LC-PFN model on the task of predicting an accuracy over epochs curve. D represents the epoch accuracies up to epoch 3 ( $= T'$ ). Attention between test and training positions is shown using red and blue arrows, respectively. Plots based on Müller et al. [2022].

# 3.4 PFNs for Learning Curve Extrapolation (LC-PFNs)

To apply PFNs to learning curve extrapolation, we train them on learning curves sampled from a given prior over learning curves (in our case, the prior defined in Section 3.2). Specifically, the training set we condition on is the available partial learning curve up to some varying cutoff point $T'$ , i.e., $D_{train} = \{(t', y_{t'})\}_{t'=1}^{T'}$ , the input we condition on is the epoch $t \in \{T' + 1, \ldots, m\}$ to predict for, and the desired output $y$ is the value of the learning curve at epoch $t$ . During training, we randomly sample the cutoff points $T' \sim \mathcal{U}(0, m - 1)$ for every batch in order to learn to predict for initial learning curves of varying sizes. Figure 2 illustrates the information flow in a LC-PFN during learning curve extrapolation.

LC-PFN architecture and hyperparameters We use the PFN architecture proposed by Müller et al. [2022] and visualized in Figure 2 (a). That is, we use a sequence Transformer [Vaswani et al., 2017] and treat each pair $(t,y)$ (for train) and $t$ (for test) as a separate position/token. We encode these using a simple linear layer. We do not use positional encoding such that we are permutation invariant. Furthermore, the attention matrix is masked such that every position only attends to the training positions. This way training examples can attend to each other, but the test examples do not influence each other's predictions. Note that the output of the PFN with parameters $\theta$ is a distribution $q_{\theta}(y|t,D_{train})$ . Following Müller et al. [2022], we discretize $q_{\theta}$ in a finite number of bins whose probability mass is predicted by the PFN, as is shown in Figure 2 (b). The size of each bin is set such that, under the prior, $y_{t}$ is equally likely to fall in each bin. The number of bins is a hyperparameter that we set to 1000. The LC-PFN model further inherits hyperparameters from the Transformer, including the number of layers (nlayers), number of heads (nheads), embedding size (emsize), and hidden size (nhidden). We use four heads, a hidden size of 1024, and conduct a thorough ablation study to investigate the effects of the number of layers and embedding size on the final performance, exploring a grid of values (see Table 2). We use a standard training procedure for all experiments, employing the Adam optimizer [Kingma and Ba, 2015] (learning rate 0.0001, batch size 100) with cosine annealing [Loshchilov and Hutter, 2017] with a linear warmup over the first $25\%$ epochs of the training. Finally, we set $m = 100$ , implying LC-PFN is trained for extrapolating sequences of up to 100 training steps (e.g., epochs). We found that most curves are shorter in practice, and when longer sequences are encountered, we subsample them as described in Appendix B.

# 4 Experiments

Our experiments aim to test the hypothesis that PFNs present a practical Bayesian approach to learning curve extrapolation. To this end, we first compare our LC-PFN approach against the MCMC approach of Domhan et al. [2015], using the same prior on samples generated from it (Section 4.1). Then, we extend the comparison to four real-world learning curve benchmarks (Section 4.2). Finally, we look beyond the quality of individual extrapolations and evaluate the potential of LC-PFN in the context of predictive early stopping to accelerate model selection (Section 4.3).

# 4.1 Extrapolating samples of the prior

The goal of this first experiment is to assess the ability of LC-PFNs and MCMC to approximate the true posterior predictive distribution (PPD). To avoid artifacts due to out-of-distribution data, in this experiment, we use curves sampled from the prior (defined in Section 3.2). Furthermore, we modified the original implementation of MCMC [Domhan et al., 2015], to use the curve prior we proposed in Section 3.2. In the following, we refer to this MCMC variant as MCMC-PP and denote the one using the original prior [Domhan et al., 2015] as MCMC-OP (used in Section 4.2). Since the LC-PFN and MCMC-PP methods are both (approximate) Bayesian inference methods, using the same prior, they aim to approximate the same true target PPD, given a partial learning curve.

As a performance metric, we evaluate the log-likelihood (LL) of the unseen data (right-censored curve) under the inferred PPD. We use this metric, also known as logarithmic score, to assess a model's ability to infer the remaining part of the curve based on the initial observed values. A benefit of this metric is that it measures the quality of the PPD as a whole (rather than merely focusing on the error associated to a specific PPD statistic) and, assuming data is generated by the prior, the exact PPD maximizes this metric.

Importantly, we vary the cutoff, i.e., the percentage of the observed curve used as input, to better assess the model's performance across different amounts of available information. Furthermore, to allow a more comprehensive comparison, we vary the hyperparameters of both LC-PFN and MCMC-PP. For LC-PFN, we vary the embedding size (emsize), the number of layers (nlayers), and the total number of learning curves used during training (nb\_data). For MCMC-PP, we vary the number of chains generated by the emcee [Foreman-Mackey et al., 2013] ensemble sampler (nwalkers), the length of each chain (burn-in + nsamples), the part of the chain omitted to account for mixing (burn-in), and the sub-sample frequency (thin). The considered values for each hyperparameter are summarized in Table 2.

Table 2: Grid of hyperparameter values evaluated for MCMC-PP and LC-PFN. 

<table><tr><td></td><td>Hyperparameters</td></tr><tr><td>MCMC-PP</td><td>nsamples ∈ [100, 250, 500, 1000, 2000, 4000]nwalkers ∈ [26, 50, 100]burn-in ∈ [0, 50, 100, 500]thin ∈ [1, 10, 100]</td></tr><tr><td>LC-PFN</td><td>nb_data ∈ [100k, 1M, 10M]emsize ∈ [128, 256, 512]nlayers ∈ [3, 6, 12]</td></tr></table>

We conducted the comparison on 10 000 sampled curves. Figure 1 shows a few curve examples, as well as inferences using LC-PFN and MCMC-PP given the data of the first 10 - 20 epochs (cutoff). We observe that both predicted median and uncertainties are indeed similar. More inference examples, with different cutoffs can be found in Appendix C.4.

Results Figure 3 displays the average log-likelihood across MCMC-PP / LC-PFN inferences for varying hyperparameters and a 10% cutoff. The log-likelihood is shown w.r.t. runtime, which is measured as the average wall-clock time for a single inference on a single Intel(R) Xeon(R) Gold 6242 CPU. Note that this inference time includes both the fit and prediction times for MCMC variants. Table 3 provides results on higher cutoffs (20%, 40%, 80%) and corresponding runtimes for three variants of each method (M1-3, P1-3), labeled in Figure 3. Generally, the LC-PFN variants (left side of figure 3) are significantly faster than MCMC-PP variants (right side). LC-PFN always ran in less than 0.1 seconds while the best MCMC-PP (M3) took over 100 seconds. Figure 3 also offers insights into the importance of LC-PFN and MCMC-PP hyperparameters. For MCMC-PP, both the cost and quality of inference increase with longer chain lengths and higher cutoffs. For LC-PFN, the inference cost increases with the model complexity which is closely related to the number of trainable parameters. Inference quality positively correlates with model size (“larger is better”), and the number of data LC-PFN was trained on. Among the hyperparameter grid we examined (Table 2), except for the smallest model (P1), all LC-PFN variants that were trained on 10M samples produce higher log-likelihood than the best MCMC-PP variant (M3). In particular, an LC-PFN (P2) with 3 layers, embedding size 256, trained on 10M samples achieved better performance (log-likelihood of PPD) than the best MCMC-PP, but more than 15 000 times faster. We also find that while the runtime of the best MCMC-PP can be reduced (with minor loss of quality) by using thinning (M2), the better LC-PFN is still approximately 7 000 times faster. Finally, it is important to note that training the largest LC-PFN (P3, 10M samples with 26M parameters) on the prior took approximately eight hours (single CPU, single RTX2080 GPU), but this cost is incurred only once for all of our experiments.

![](images/9d523e90fbb6a6748398be6d54371b8da216934cff5538bb40b4c8213ce02691.jpg)

<details>
<summary>scatter</summary>

| LC-PFN - emsize | nSamples | Time (seconds) | Log-likelihood |
| --------------- | -------- | -------------- | ------------- |
| 128             | 100      | ~0.01          | ~1.6          |
| 256             | 250      | ~0.01          | ~1.25         |
| 512             | 1000     | ~0.01          | ~1.3          |
| 100k            | 6        | ~0.01          | ~1.4          |
| 1M              | 1M       | ~0.01          | ~1.4          |
| 10M             | 6        | ~0.01          | ~1.7          |
| 3               | 12       | ~0.01          | ~1.7          |
| 6               | 12       | ~0.01          | ~1.7          |
| M2              | 12       | ~10            | ~1.65         |
| M3              | 12       | ~10            | ~1.65         |
</details>

Figure 3: Runtime (lower is better) vs log-likelihood of the true curve under the PPD (higher is better), with 10% of the curve observed. See Figure 9 in Appendix C.1 for higher cutoffs. Blue and red markers correspond respectively to LC-PFN and MCMC-PP with varying hyperparameters values. The M1-3/P1-3 labels refer to the PFN/MCMC variants listed in Table 3. The horizontal dashed line indicates the performance of the best MCMC variant.

Table 3: Comparison of three LC-PFN and MCMC-PP variants on prior curves in terms of log-likelihood (higher is better) at 10%, 20%, 40%, and 80% cutoffs. Here, M1 corresponds to the configuration used in Domhan et al. [2015]. Please refer to Table 5 for more comprehensive results. 

<table><tr><td>Label</td><td>Method</td><td>Parameters</td><td>10%</td><td>20%</td><td>40%</td><td>80%</td><td>Avg. Runtime (s)</td></tr><tr><td>M1</td><td>MCMC</td><td>nsamples=2000, nwalkers=100, burn-in=500, thin=1</td><td>1.628</td><td>1.939</td><td>2.265</td><td>2.469</td><td>54.401</td></tr><tr><td>M2</td><td>MCMC</td><td>nsamples=4000, nwalkers=100, burn-in=100, thin=100</td><td>1.641</td><td>1.958</td><td>2.277</td><td>2.477</td><td>45.160</td></tr><tr><td>M3</td><td>MCMC</td><td>nsamples=4000, nwalkers=100, burn-in=500, thin=1</td><td>1.642</td><td>1.956</td><td>2.285</td><td>2.486</td><td>103.151</td></tr><tr><td>P1</td><td>PFN</td><td>nb_data=10M, nlayers=3, emsize=128</td><td>1.58</td><td>1.99</td><td>2.28</td><td>2.43</td><td>0.004</td></tr><tr><td>P2</td><td>PFN</td><td>nb_data=10M, nlayers=3, emsize=256</td><td>1.65</td><td>2.04</td><td>2.35</td><td>2.49</td><td>0.006</td></tr><tr><td>P3</td><td>PFN</td><td>nb_data=10M, nlayers=12, emsize=512</td><td>1.76</td><td>2.13</td><td>2.40</td><td>2.52</td><td>0.050</td></tr></table>

# 4.2 Extrapolating real-world learning curves

While evaluation on data from the prior gives us a controlled setting to analyse quality and cost of the PPD approximation, performance on real-world learning curves is essential for practical usefulness. This second experiment aims to extend the previous comparison of MCMC and LC-PFN to real-world learning curve benchmarks.

We consider the best-performing variants of LC-PFN and MCMC-PP according to the average log-likelihood they obtained in the first experiment. For LC-PFN, the optimal variant (P3) features an embedding size of 512 and 12 layers, resulting in a total of 26M trainable parameters, and is trained on 10 million prior curves. For MCMC-PP, the optimal configuration (M3) involves a chain length of 4500, 100 walkers, 500 burn-in samples, without thinning. As an additional baseline, we include MCMC-OP, the original MCMC variant proposed by Domhan et al. [2015], which uses the original hyperparameters and curve prior (11 basis curves and uninformative prior over the curve parameters).

Benchmarks To evaluate the generalization capabilities of our model, we consider a diverse set of real-world curves. Our dataset comprises 20000 learning curves, sourced from four distinct benchmarks: LCBench [Zimmer et al., 2021], NAS-Bench-201 [Dong and Yang, 2020], Taskset [Metz et al., 2020] and PD1 [Wang et al., 2022], each contributing 5000 curves, randomly selected from specific subtasks. These benchmarks and subtasks were chosen to span a broad spectrum of supervised deep learning problems, training MLP (LCBench), CNN (NAS-Bench-201), RNN (Taskset), and Transformer (PD1) architectures on input modalities ranging from tabular data (LCBench), text (Taskset and PD1), protein sequence (PD1), to vision problems (NAS-Bench-201). From LCBench and NAS-Bench-201 we use validation accuracy curves whereas from Taskset and PD1 the log loss validation curves. In terms of curve length, LCBench and Taskset cover 50 epochs, NAS-Bench-201 contains up to 200 epochs, and PD1 curves have varying lengths (22 - 1414). Further details, including sample curves, on these benchmarks are provided in Appendix B.

![](images/9c3784a4632bf6ec5b8c13bcf529cf514d3c18186766d4b769fbe3d04580706f.jpg)

<details>
<summary>line</summary>

| Dataset       | Percentage of the curve observed | LC-PFN | MCMC(original) | MCMC(our prior) |
| ------------- | -------------------------------- | ------ | -------------- | --------------- |
| LCBench       | 10                               | 1.7    | 2.5            | 1.8             |
| LCBench       | 20                               | 1.7    | 2.5            | 1.8             |
| LCBench       | 30                               | 1.7    | 2.5            | 1.8             |
| LCBench       | 40                               | 1.7    | 2.5            | 1.8             |
| LCBench       | 50                               | 1.7    | 2.5            | 1.8             |
| LCBench       | 60                               | 1.7    | 2.5            | 1.8             |
| LCBench       | 70                               | 1.7    | 2.5            | 1.8             |
| LCBench       | 80                               | 1.7    | 2.5            | 1.8             |
| LCBench       | 90                               | 1.7    | 2.5            | 1.8             |
| NAS-Bench-201| 10                               | 2.0    | 1.6            | 2.3             |
| NAS-Bench-201| 20                               | 2.1    | 1.3            | 2.4             |
| NAS-Bench-201| 30                               | 2.2    | 1.3            | 2.5             |
| NAS-Bench-201| 40                               | 2.1    | 1.3            | 2.6             |
| NAS-Bench-201| 50                               | 2.0    | 1.4            | 2.6             |
| NAS-Bench-201| 60                               | 1.8    | 1.6            | 2.6             |
| NAS-Bench-201| 70                               | 1.6    | 1.9            | 2.6             |
| NAS-Bench-201| 80                               | 1.4    | 2.1            | 2.5             |
| NAS-Bench-201| 90                               | 1.3    | 2.3            | 2.4             |
| Taskset       | 10                               | 2.1    | 1.9            | 2.0             |
| Taskset       | 20                               | 2.0    | 2.0            | 2.0             |
| Taskset       | 30                               | 1.9    | 2.1            | 2.0             |
| Taskset       | 40                               | 1.8    | 2.2            | 2.0             |
| Taskset       | 50                               | 1.7    | 2.3            | 2.0             |
| Taskset       | 60                               | 1.6    | 2.4            | 2.0             |
| Taskset       | 70                               | 1.5    | 2.5            | 2.0             |
| Taskset       | 80                               | 1.4    | 2.6            | 2.0             |
| Taskset       | 90                               | 1.3    | 2.7            | 2.0             |
| PD1           | 10                               | 1.6    | 2.5            | 1.9             |
| PD1           | 20                               | 1.6    | 2.6            | 1.8             |
| PD1           | 30                               | 1.6    | 2.6            | 1.8             |
| PD1           | 40                               | 1.6    | 2.6            | 1.8             |
| PD1           | 50                               | 1.6    | 2.6            | 1.8             |
| PD1           | 60                               | 1.6    | 2.6            | 1.8             |
| PD1           | 70                               | 1.6    | 2.6            | 1.8             |
| PD1           | 80                               | 1.6    | 2.7            | 1.7             |
| PD1           | 90                               | 1.6    | 2.8            | 1.6             |
The chart displays the average rank log-likelihood for each dataset (LCBench, NAS-Bench-201, Taskset, PD1). The data is presented in a separate table with three columns: Label (X-axis) and Label (Y-axis). Legend categories are LC-PFN (blue), MCMC(original) (orange), and MCMC(our prior) (green). The chart is grouped by dataset and labeled with the same legend and axis labels.
</details>

(a) Average rank of log-likelihood values (lower is better) vs cutoffs   
![](images/4f6b89c5d41113fd63b2e57f489169012fbe0a9995f56ed9f46a64e4127da36f.jpg)

<details>
<summary>line</summary>

| Dataset       | Percentage of the curve observed | Avg rank MSE |
| ------------- | -------------------------------- | ------------ |
| LCBench       | 10                               | 2.6          |
| LCBench       | 20                               | 2.55         |
| LCBench       | 30                               | 2.5          |
| LCBench       | 40                               | 2.45         |
| LCBench       | 50                               | 2.4          |
| LCBench       | 60                               | 2.35         |
| LCBench       | 70                               | 2.3          |
| LCBench       | 80                               | 2.25         |
| LCBench       | 90                               | 2.2          |
| NAS-Bench-201| 10                               | 1.4          |
| NAS-Bench-201| 20                               | 1.1          |
| NAS-Bench-201| 30                               | 1.3          |
| NAS-Bench-201| 40                               | 1.5          |
| NAS-Bench-201| 50                               | 1.7          |
| NAS-Bench-201| 60                               | 1.9          |
| NAS-Bench-201| 70                               | 2.0          |
| NAS-Bench-201| 80                               | 2.1          |
| NAS-Bench-201| 90                               | 2.1          |
| Taskset       | 10                               | 2.1          |
| Taskset       | 20                               | 2.2          |
| Taskset       | 30                               | 2.3          |
| Taskset       | 40                               | 2.4          |
| Taskset       | 50                               | 2.5          |
| Taskset       | 60                               | 2.6          |
| Taskset       | 70                               | 2.6          |
| Taskset       | 80                               | 2.6          |
| Taskset       | 90                               | 2.6          |
| PD1           | 10                               | 2.3          |
| PD1           | 20                               | 2.4          |
| PD1           | 30                               | 2.5          |
| PD1           | 40                               | 2.6          |
| PD1           | 50                               | 2.6          |
| PD1           | 60                               | 2.6          |
| PD1           | 70                               | 2.6          |
| PD1           | 80                               | 2.6          |
| PD1           | 90                               | 2.6          |
</details>

(b) Average rank of mean squared error (MSE) values (lower is better) vs cutoffs   
Figure 4: Comparison of LC-PFN with two MCMC variants on three real-data benchmarks.

Metrics Our focus lies on the relative performance of MCMC and LC-PFN, as absolute performance is significantly influenced by the choice of prior. We consider the log-likelihood and the mean squared error (MSE) of the predictions as metrics. Here, the MSE is calculated w.r.t. the median of the PPD.

For each benchmark, we report the average rank of these metrics to aggregate results from different curves, as supposed to the average values. While the latter better captures performance differences, it is very sensitive to outliers and scale-dependent. When computed in normalized space, it would strongly depend on our choice of normalization parameters (see Appendix A).

Results For each benchmark, Figures 4a and 4b display the average rank obtained by each method in terms of log-likelihood and MSE, respectively, where ranks are averaged across all 5000 curves. We do not include error bars as the standard errors are negligible (less than 0.02). In summary, we observe similar trends for both metrics on all benchmarks. LC-PFN is never truly outperformed by MCMC-PP. On LCBench both methods rank similarly, with LC-PFN being slightly worse at high cutoffs. LC-PFN ranks better on PD1, NAS-Bench-201, and Taskset. MCMC-OP performs clearly worse on LCBench, Taskset, and PD1. On NAS-Bench-201, MCMC-OP performs best for lower cutoffs, outperforming MCMC-PP, suggesting that NAS-Bench-201 curves are better captured by the original prior. Figure 11 and Figure 12 in Appendix C.2 show the log-likelihoods and MSEs, respectively, for all three methods, for each curve and cutoff, per benchmark, providing a more detailed perspective.

# 4.3 Application: Extrapolation-based early stopping in model selection

Thus far, we have shown that LC-PFN produces extrapolations of similar or better quality to MCMC, at a small fraction of the cost. However, these extrapolations are not perfect. In many cases, the practical relevance of any errors can only be assessed in the context of a specific application.

In this final experiment, we thus consider a model selection setting where after every epoch of training we have the choice between (i) continuing the current training, or (ii) stopping early $T < m$ and starting a new training run using a different training pipeline (e.g., model architecture, hyperparameters, etc.). Here, we assume that runs cannot be resumed once stopped and that the order in which training pipelines are to be considered is given. Our objective is to obtain high-quality models as quickly as possible, by stopping suboptimal training runs as early as possible. This setting is also known as vertical model selection [Mohr and van Rijn, 2022] and was also considered by Domhan et al. [2015].

![](images/bff048727df3c125d3db34e2f9c240b8c00175b58c95b72566d4ee54c6fc40a3.jpg)  
no-stop patience(0) patience(1) patience(5) patience(10) patience(20) patience(50) LC-PFN(0.95,coarse) LC-PFN(0.95,fine)

Figure 5: Comparison of our LC-PFN based early stopping mechanism to naive baselines no-stop (no stopping, train for the full budget m) and Patience(k) (stop training after k epochs without improvement) for vertical model selection where runs are considered in a fixed order and cannot be resumed. Shown is the anytime regret (lower is better) different approaches achieve after a total number of training epochs, averaged per benchmark and across 40 orderings per task.

We consider the extrapolation-based termination criterion proposed by Domhan et al. [2015], but use LC-PFN instead of MCMC to predict the likelihood $\Pr(y_{t^{\prime}} > y_{\mathrm{best}} \mid y_{1}, \ldots, y_{T})$ that the current run will at epoch $t^{\prime}$ obtain a model better than the best obtained by any run thus far ( $y_{best}$ ), for $T < t^{\prime} \leq m$ and decide to stop the current run if that probability does not exceed some fixed threshold $\delta$ (at any point). In our experiments, we use confidence level $1 - \delta = 0.95$ as Domhan et al. [2015]. Note that this criterion can be applied at every step or only at specific cutoffs. To simulate the effect of varying granularity, we consider a coarse-grained variant with 4 cutoffs $T \in \{\lceil 0.1m \rceil, \lceil 0.2m \rceil, \lceil 0.4m \rceil, \lceil 0.8m \rceil\}$ , and a fine-grained variant with $T \in \{T \mid 1 < T < m\}$ . We investigate alternative choices for cutoffs and confidence levels in Appendix C.3.2.

Following Domhan et al. [2015], we compare against a black box approach no-stop that does not implement early stopping. We further compare against a criterion Patience(k) that implements the popular heuristic to terminate a training run when model performance did not improve for k epochs.

For evaluation, we consider the same benchmarks as in Section 4.2 (for details, see Appendix B). We chose the total budget for model selection to correspond to 20 full runs (20m). For each task, we consider the training runs in 40 different random orderings. This totals 2120 model selection experiments per method, spread across the 53 different tasks.

Results Figure 5 shows the anytime performance of all methods in our comparison, on each of the benchmarks, in terms of regret. Here, regret is the absolute difference in performance between the best model obtained thus far and the best model attained by any run on the task. Results are averaged over the different run orderings. For LCBench, NAS-Bench-201, and Taskset results are further averaged across all tasks (results for individual tasks can be found in Appendix C.3.1). The shaded area corresponds to $\pm$ 1 standard error. On all benchmarks, except for NAS-Bench-201, we observe that LC-PFN based termination criteria clearly perform best.

Also, the fine-grained variant performs better on average, suggesting that errors in inference are compensated for by the time saved by stopping runs earlier. In terms of expected speed-up, this LC-PFN variant obtains an expected regret lower than that obtained by no-stop, approximately $3.3 \times$ faster on LCBench and Taskset. Looking at individual tasks, we note $2 - 6 \times$ speed-ups on all 3 PD1 tasks, all 12 Taskset tasks, and 30 of the 35 LCBench tasks (we obtain $1 - 2 \times$ speed-ups on the remaining 5). On NAS-Bench-201, we find that all termination criteria considered, including

standard Patience heuristics, fail on all 3 tasks; this is likely related to the particular shape of learning curves on this benchmark, having an inflection point (see Figure 8 and Figure 19), not resembling any in the prior.

Finally, in terms of overhead, the computational costs of the LC-PFN inferences per model selection experiment range from 3 seconds (coarse-grained) up to 1 minute (fine-grained), and are negligible compared to the cost of the 20 full training runs of deep neural networks.

# 5 Summary, limitations, and future research

We presented the first work using prior-data fitted networks (PFNs) for Bayesian learning curve extrapolation. We show that our LC-PFN obtains qualitatively similar extrapolations, for a wide variety of learning curves, more than $10\ 000 \times$ faster than the MCMC method proposed by Domhan et al. [2015]. These inferences are now fast enough (under 100 milliseconds on CPU, and even less on GPU), to be used in the context of online learning, at virtually no overhead. This opens up a wide variety of possible applications, e.g., to speed up automated model selection in AutoML and HPO by discarding poor configurations early. It would be interesting to integrate LC-PFN as a new termination criterion in existing deep learning libraries.

Often, we also have more data available than a single partial learning curve, e.g., other curves on the same task, their hyperparameters, and/or curves of the same method on a different task, and meta-features. Previous work [Swersky et al., 2014, Klein et al., 2017, Wistuba et al., 2022, Ruhkopf et al., 2022] has already exploited this, and we could explore the potential of using PFNs for few-shot in-context meta-learning, by feeding the model multiple curves and hyperparameters as input.

While for a fair comparison to Domhan et al. [2015], reusing the original code, our prior (Section 3.2) closely resembled the one of Domhan et al. [2015], future work could improve upon this prior and overcome its limitations (e.g., on the NAS-Bench-201 tasks) by modelling divergence, slow start, double-descent, correlated heteroscedastic noise, etc.

Finally, PFNs, unlike other Bayesian methods, must learn the prior from data, which implies that the prior must be generative. Also, it suggests that high entropy priors may present challenges. Future research should investigate these limitations and how to overcome them.

# 6 Acknowledgments and disclosure of funding

We acknowledge funding by the European Union (via ERC Consolidator Grant Deep Learning 2.0, grant no. 101045765), TAILOR, a project funded by EU Horizon 2020 research and innovation programme under GA No 952215, the state of Baden-Württemberg through bwHPC and the German Research Foundation (DFG) through grant numbers INST 39/963-1 FUGG and 417962828. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

![](images/958b596e4352e12e6dbd833410da23346769b59dd60a045561d56c21f9d9f6a1.jpg)  
Funded by the European Union

# References

B. Baker, O. Gupta, N. Naik, and R. Raskar. Designing neural network architectures using reinforcement learning. arXiv:1611.02167 [cs.LG], 2017.   
A. Chandrashekaran and I. R. Lane. Speeding up hyper-parameter optimization by extrapolation of learning curves using previous builds. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 477–492. Springer, 2017.   
D. Choi, H. Cho, and W. Rhee. On the difficulty of DNN hyperparameter optimization using learning curve prediction. In 2018 IEEE Region 10 Conference (TENCON'2018), pages 651–656. IEEE, 2018.   
C. Cortes, L. D. Jackel, S. Solla, V. Vapnik, and J. Denker. Learning curves: Asymptotic values and rate of convergence. Advances in neural information processing systems, 6, 1993.   
A. Damianou and N. D. Lawrence. Deep gaussian processes. In Artificial intelligence and statistics, pages 207–215. PMLR, 2013.   
T. Domhan, J. Springenberg, and F. Hutter. Speeding up automatic Hyperparameter Optimization of deep neural networks by extrapolation of learning curves. In Proceedings of the 24th International Joint Conference on Artificial Intelligence (IJCAI'15), pages 3460–3468, 2015.   
X. Dong and Y. Yang. NAS-Bench-201: Extending the scope of reproducible Neural Architecture Search. In Proceedings of the International Conference on Learning Representations (ICLR'20), 2020.   
S. Dooley, G. S. Khurana, C. Mohapatra, S. Naidu, and C. White. ForecastPFN: Synthetically-trained zero-shot forecasting. In Proceedings of the 37th International Conference on Advances in Neural Information Processing Systems (NeurIPS'23), 2023.   
S. Falkner, A. Klein, and F. Hutter. BOHB: Robust and efficient Hyperparameter Optimization at scale. In Proceedings of the 35th International Conference on Machine Learning (ICML'18), volume 80, pages 1437–1446. PMLR, 2018.   
D. Foreman-Mackey, D. Hogg, D. Lang, and J. Goodman. emcee: The MCMC Hammer. \*Publications of the Astronomical Society of the Pacific\*, 125(925):306, 2013.   
L. J. Frey and D. H. Fisher. Modeling decision tree performance with the power law. In Seventh International Workshop on Artificial Intelligence and Statistics. PMLR, 1999.   
M. Gargiani, A. Klein, S. Falkner, and F. Hutter. Probabilistic rollouts for learning curve extrapolation across hyperparameter settings. In ICML workshop on Automated Machine Learning (AutoML workshop 2019), 2019.   
P. Gijsbers, E. LeDell, S. Poirier, J. Thomas, B. Bischl, and J. Vanschoren. An open source automl benchmark. In ICML workshop on Automated Machine Learning (AutoML workshop 2019), 2019.   
N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. In The Eleventh International Conference on Learning Representations, 2023.   
J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv:2001.08361 [cs.LG], 2020.   
D. Kingma and J. Ba. Adam: A method for stochastic optimization. In Proceedings of the International Conference on Learning Representations (ICLR'15), 2015.   
A. Klein, S. Falkner, J. Springenberg, and F. Hutter. Learning curve prediction with Bayesian neural networks. In Proceedings of the International Conference on Learning Representations (ICLR'17), 2017.   
P. Kolachina, N. Cancedda, M. Dymetman, and S. Venkatapathy. Prediction of learning curves in machine translation. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 22–30, 2012.   
R. Leite and P. Brazdil. Improving progressive sampling via meta-learning on learning curves. In European conference on machine learning, pages 250–261. Springer, 2004.   
L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar. Hyperband: Bandit-based configuration evaluation for Hyperparameter Optimization. In Proceedings of the International Conference on Learning Representations (ICLR'17), 2017.

I. Loshchilov and F. Hutter. SGDR: Stochastic gradient descent with warm restarts. In Proceedings of the International Conference on Learning Representations (ICLR'17), 2017.   
A. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies, pages 142–150, 2011.   
L. Metz, N. Maheswaranathan, R. Sun, C. D. Freeman, B. Poole, and J. Sohl-Dickstein. Using a thousand optimization tasks to learn hyperparameter search strategies. arXiv:2002.11887 [cs.LG], 2020.   
F. Mohr and J. van Rijn. Learning curves for decision making in supervised machine learning—a survey. arXiv:2201.12150v1 [cs.LG], 2022.   
S. Müller, N. Hollmann, S. Arango, J. Grabocka, and F. Hutter. Transformers can do bayesian inference. In Proceedings of the International Conference on Learning Representations (ICLR'22), 2022.   
S. Müller, M. Feurer, N. Hollmann, and F. Hutter. PFNs4BO: In-context learning for bayesian optimization. In ICML 23, 2023.   
T. Ruhkopf, A. Mohan, D. Deng, A. Tornede, F. Hutter, and M. Lindauer. Masif: Meta-learned algorithm selection using implicit fidelity information. Transactions on Machine Learning Research, 2022.   
D. Salinas, M. Seeger, A. Klein, V. Perrone, M. Wistuba, and C. Archambeau. Syne tune: A library for large scale hyperparameter tuning and reproducible research. In International Conference on Automated Machine Learning, AutoML 2022, 2022.   
K. Swersky, J. Snoek, and R. Adams. Freeze-thaw Bayesian optimization. arXiv:1406.3896 [stats.ML], 2014.   
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In Proceedings of the 31st International Conference on Advances in Neural Information Processing Systems (NeurIPS'17), 2017.   
Z. Wang, G. E. Dahl, K. Swersky, C. Lee, Z. Mariet, Z. Nado, J. Gilmer, J. Snoek, and Z. Ghahramani. Pre-training helps bayesian optimization too, 2022.   
M. Wistuba, A. Kadra, and J. Grabocka. Supervising the multi-fidelity race of hyperparameter configurations. In Proceedings of the 36th International Conference on Advances in Neural Information Processing Systems (NeurIPS'22), 2022.   
Y. Yao, L. Rosasco, and A. Caponnetto. On early stopping in gradient descent learning. Constructive Approximation, 26(2):289–315, 2007.   
L. Zimmer, M. Lindauer, and F. Hutter. Auto-PyTorch Tabular: Multi-Fidelity MetaLearning for Efficient and Robust AutoDL. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43:3079–3090, 2021.

# A Learning curve normalization

Not all learning curves of interest may resemble those of the prior we described in Section 3.2. For example, when minimizing the negative log-likelihood loss (log loss), learning curves will mostly decrease (vs. increase), be convex (vs. concave), and could take values that exceed 1.0 (there is no hard upper bound for log loss).

One approach would be to define a prior and train a specialized PFN for every performance measure. In this work, we use a more general approach: We apply a normalization procedure that allows us to accurately extrapolate a wide variety of learning curves using a single PFN, without retraining or fine-tuning. We consistently apply this procedure to all inferences involving real learning curves (i.e., the experiments described in Section 4.2 and Section 4.3), even if observations are naturally constrained to $[0, 1]$ , e.g., accuracy curves.

Step 1: Normalize the partial learning curve: Let $y_{i}^{o}$ be the $i^{th}$ performance observation. We start by normalizing it using a generalized logistic sigmoid transform $y_{i} = g_{\lambda}(y_{i}^{o})$ , that is re-parametrized by the following five parameters ( $\lambda$ ):

min? A Boolean specifying that we expect learning to minimize the performance measure. E.g., this would be true for error rate and log loss, and false for accuracy.

$l_{hard}$ , $u_{hard}$ These are possibly infinite hard lower / upper bounds for model performance. This would, e.g., be 0 / 1 for accuracy and error rate; and 0 / +∞ for log loss.

$l_{soft}, u_{soft}$ These are finite soft lower/upper bounds for model performance. They specify the range in which we expect performance values (that we care to distinguish between) to lie in practice. For accuracy and error rate, one could set these equal to the hard bounds, whereas for log loss one could, e.g., use an estimate of the loss for the untrained network (i.e., at epoch 0) as $u_{soft}$ and $l_{soft} = l_{hard}$ , or if available, choose $l_{soft}$ to be an optimistic estimate of performance at convergence (e.g., state-of-the-art). Narrower ranges will result in more accurate inferences.

Specifically, we perform a linear transformation, followed by a logistic sigmoid, another linear transform, and a minimization conditional reflection around $\frac{1}{2}$ , i.e.,

$$
g _ {\boldsymbol {\lambda}} (y _ {i} ^ {o}) = \mathrm{cr} _ {0. 5} \left(\frac {c}{1 + e ^ {- a (x - b)}} + d\right)
$$

where

$$
\operatorname{cr} _ {0. 5} (y) = \left\{ \begin{array}{l l} 1 - y & \text { if   } \min? \\ y & \text { if   } \neg \min? \end{array} \right. \quad a = \frac {2}{u _ {\text { soft }} - l _ {\text { soft }}} \quad b = - \frac {u _ {\text { soft }} + l _ {\text { soft }}}{u _ {\text { soft }} - l _ {\text { soft }}}
$$

$$
c = \frac {1 + e ^ {- a \left(\mathrm{u} _ {\text {hard}} - b\right)} + e ^ {- a \left(\mathrm{l} _ {\text {hard}} - b\right)} + e ^ {- a \left(\mathrm{u} _ {\text {hard}} + \mathrm{l} _ {\text {hard}} - 2 b\right)}}{e ^ {- a \left(\mathrm{l} _ {\text {hard}} - b\right)} - e ^ {- a \left(\mathrm{u} _ {\text {hard}} - b\right)}} \quad d = \frac {- c}{1 + e ^ {- a \left(\mathrm{l} _ {\text {hard}} - b\right)}}
$$

Note that for $\lambda = (\text{False}, -\infty, -1, 1, \infty)$ this reduces to the canonical logistic sigmoid $y_{i} = \frac{1}{1 + e^{-y_{i}^{o}}}$ . This transform will be approximately linear (shape preserving) for $y_{i}^{o} \in [l_{soft}, u_{soft}]$ , and the tails of the sigmoid will gradually squish values outside this range to [0, 1], supporting unbounded performance measures. Figure 6 visualizes this projection in general, and provides examples for accuracy with $\lambda = (\text{False}, 0, 0, 1, 1)$ and log loss with $\lambda = (\text{True}, 0, 0, \log(10), \infty)$ .

Step 2: Infer using normalized data: Next, we perform Bayesian learning curve extrapolation on the normalized partial curve y, with our usual prior $p(\mathbf{y})$ , resulting in an approximation of the PPD in the transformed space.

Step 3: Inverse transform the PPD: Finally, we can obtain the property of interest of the PPD in the original space by applying the inverse transform, given by

$$
g _ {\pmb {\lambda}} ^ {-} (y _ {i} ^ {p}) = \frac {\log (\frac {\mathrm{cr} _ {0 . 5} (y _ {i} ^ {p}) - d}{c - (\mathrm{cr} _ {0 . 5} (y _ {i} ^ {p}) - d)}) - b}{a}
$$

We use order-based statistics (e.g., median or other percentiles) in our experiments, to which we can simply apply $g_{\lambda}^{-}$ directly since it is a monotonic transform. For other statistics (e.g., mean, variance) we may need to resort to Monte Carlo estimation, applying $g_{\lambda}^{-}$ to samples of the PPD.

![](images/3081e96d80c383dfa482b7540c1a04d0d709483e67f04a9eeec368983d718ab4.jpg)

<details>
<summary>line</summary>

| original space | inference space |
| -------------- | --------------- |
| I_hand         | 0.0             |
| I_salt         | 0.5             |
| U_salt         | 1.0             |
| U_hand         | 1.0             |
</details>

![](images/70227ed4aad76940f8c87d614c9b2d54ecf45d1ba4ed1e94b0e6a7c83088171d.jpg)

<details>
<summary>line</summary>

| accuracy | value |
| -------- | ----- |
| -0.2     | -0.2  |
| 0.0      | 0.0   |
| 0.2      | 0.2   |
| 0.4      | 0.4   |
| 0.6      | 0.6   |
| 0.8      | 0.8   |
| 1.0      | 1.0   |
| 1.2      | 1.2   |
</details>

![](images/b63bb6dfac102ef8f3f6cb142fbc9ff8fd38cc4fd4823e0cccca278928e24e8f.jpg)

<details>
<summary>line</summary>

| log loss | Black Line | Blue Line |
| -------- | ---------- | --------- |
| 0        | 1.2        | 1.0       |
| 2        | 0.4        | 0.4       |
| 4        | 0.1        | 0.1       |
| 6        | 0.0        | 0.0       |
| 8        | 0.0        | 0.0       |
| 10       | 0.0        | 0.0       |
</details>

Figure 6: left: The generic logistic sigmoid transformation described in Appendix A. and used to normalize learning curves to support a wide range of possibly unbounded performance metrics. middle/right: An example of the normalization for curves maximizing accuracy / minimizing log loss, using the transformation.   
![](images/34ea8bc6f5b4f51a64903eaf7f0d79c2c6f31e16f7f730efd34f834769a06d2f.jpg)

<details>
<summary>line</summary>

| t   | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 | Series 6 | Series 7 | Series 8 | Series 9 |
| --- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 0   | 0.1      | 0.3      | 0.4      | 0.5      | 0.6      | 0.7      | 0.8      | 0.9      | 1.0      |
| 20  | 0.15     | 0.35     | 0.45     | 0.55     | 0.65     | 0.75     | 0.85     | 0.95     | 1.05     |
| 40  | 0.18     | 0.38     | 0.48     | 0.58     | 0.68     | 0.78     | 0.88     | 0.98     | 1.1      |
| 60  | 0.2      | 0.4      | 0.5      | 0.6      | 0.7      | 0.8      | 0.9      | 1.0      | 1.15     |
| 80  | 0.22     | 0.42     | 0.52     | 0.62     | 0.72     | 0.82     | 0.92     | 1.02     | 1.2      |
| 100 | 0.25     | 0.45     | 0.55     | 0.65     | 0.75     | 0.85     | 0.95     | 1.05     | 1.25     |
</details>

![](images/1bcffa2813d5af1593b517ca59eb4f1b176b0cd54a92df8d3d722a0123be0e0d.jpg)

<details>
<summary>line</summary>

| epochs | log loss (red) | log loss (cyan) | log loss (purple) | log loss (green) | log loss (yellow) | log loss (pink) | log loss (orange) | log loss (gray) |
| ------ | -------------- | --------------- | ----------------- | ---------------- | ----------------- | --------------- | ----------------- | --------------- |
| 0      | ~6.0           | ~4.0            | ~3.5              | ~3.0             | ~2.5              | ~2.0            | ~1.8              | ~1.5            |
| 20     | ~5.5           | ~3.8            | ~3.3              | ~2.8             | ~2.3              | ~1.9            | ~1.7              | ~1.3            |
| 40     | ~5.0           | ~3.6            | ~3.1              | ~2.6             | ~2.1              | ~1.8            | ~1.6              | ~1.2            |
| 60     | ~4.8           | ~3.5            | ~3.0              | ~2.5             | ~2.0              | ~1.7            | ~1.5              | ~1.1            |
| 80     | ~4.5           | ~3.4            | ~2.9              | ~2.4             | ~1.9              | ~1.6            | ~1.4              | ~1.0            |
| 100    | ~4.2           | ~3.3            | ~2.8              | ~2.3             | ~1.8              | ~1.5            | ~1.3              | ~0.9            |
</details>

Figure 7: Sample of 10 curves taken i.i.d. from the prior (Section 3.2). (Right) The same sample after applying the inverse transformation $g_{\lambda}^{-}$ to obtain samples from the log loss prior as described in Appendix A, using the transform shown in Figure 6 (right).

By applying a given normalization, we effectively transform our prior, i.e., $p(\pmb{y}^o) = g_{\lambda}^{-}(p(\pmb{y}))$ . To get some intuition of what that prior looks like, we can sample from $p(\pmb{y}^o)$ by applying $g_{\lambda}^{-}$ to samples of our prior $p(\pmb{y})$ , which is useful for fine-tuning the parameters of this transformation. Figure 7 (right) shows samples of the log loss prior for $\lambda = (\text{True}, 0, 0, \log(10), \infty)$ .

# B Detailed benchmark description

To evaluate the generalization capabilities of LC-PFN, we consider a diverse set of real-world curves. The reduce computational cost and improve reproducibility, we source these from four existing benchmarks that provide a collection of learning curves for a wide variety of supervised learning tasks. In what follows, we describe each of these benchmarks in more detail, the subset of tasks and curves we considered, and any preprocessing we did. Table 4 provides an overview of the characteristics of each benchmark and Figure 8 (a) visualizes the curves selected from each benchmark.

LCBench provides learning curve data for 2000 configurations of AutoPytorch [Zimmer et al., 2021] trained for 50 epochs on 35 tabular classification datasets from the AutoML benchmark [Gijsbers et al., 2019]. All of these configurations use momentum SGD to train an MLP, but with varying number of layers and units per layer, and varying optimization hyperparameters (batch size, learning rate, momentum, L2 regularization, dropout rate). In Section 4.2, we consider a subset of 5000 curves selected uniformly at random from all 7000 validation accuracy curves in the benchmark. In Section 4.3, we consider all 2000 validation accuracy curves, for every task, in 40 different orderings.

NAS-Bench-201 is a benchmark for Neural Architecture Search methods [Dong and Yang, 2020]. It provides learning curve data for training 15625 different architectures for 200 epochs on three different image classification datasets (CIFAR-10, CIFAR-100, and ImageNet16-120) for three different random seeds. Other aspects of the training pipeline (e.g., optimizer, hyperparameters)

Table 4: Overview of the characteristics of the real-world learning curve benchmarks we used in Section 4.2 and Section 4.3. For each benchmark, the last columns list the normalization parameters used (see Appendix A), where $y_{0}$ corresponds to the performance of the untrained model. 

<table><tr><td>benchmark</td><td># subtasks</td><td>model architecture</td><td>input modality (datasets)</td><td>length (# epochs)</td><td>metric (split)</td><td>min?</td><td> $l_{hard}$ </td><td> $l_{soft}$ </td><td> $u_{soft}$ </td><td> $u_{hard}$ </td></tr><tr><td>LCBench</td><td>35</td><td>MLP</td><td>Tabular Classification (OpenML)</td><td>50</td><td>accuracy (val)</td><td>False</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>NAS-Bench-201</td><td>3</td><td>CNN</td><td>Image Classification (Cifar-10, Cifar-100, ImageNet16-120)</td><td>200</td><td>error rate (val)</td><td>True</td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Taskset</td><td>12</td><td>RNN</td><td>Text Classification (Sentiment Analysis - IMDB)</td><td>50</td><td>log loss (val)</td><td>True</td><td>0</td><td>0</td><td> $y_0$ </td><td>+ $\infty$ </td></tr><tr><td rowspan="3">PD1</td><td rowspan="3">3</td><td>Transformer</td><td>Text (Language Modeling - lm1b)</td><td>38</td><td>log loss (val)</td><td>True</td><td>0</td><td>3.46</td><td> $y_0$ </td><td>+ $\infty$ </td></tr><tr><td>Transformer</td><td>Text (Translation - WMT)</td><td>1414</td><td>log loss (val)</td><td>True</td><td>0</td><td>1.68</td><td> $y_0$ </td><td>+ $\infty$ </td></tr><tr><td>Transformer</td><td>Protein sequences (UniRef50)</td><td>22</td><td>log loss (val)</td><td>True</td><td>0</td><td>2.60</td><td> $y_0$ </td><td>+ $\infty$ </td></tr></table>

are fixed. We use the validation error rate curves provided through the Syne Tune [Salinas et al., 2022] interface to this benchmark. As discussed in Section 3.4, the LC-PFN we consider is trained to extrapolate curves up to length 100 (m). $^{3}$ To handle the 200 epoch curves from NAS-Bench-201, we chose to subsample them, feeding only every second observation into the LC-PFN. In Section 4.2, we consider a subset of 5 000 curves selected uniformly at random from all 140 625 error rate curves in the benchmark. In Section 4.3, we consider all 46 875 error rate curves, for each of the three datasets, in 40 different orderings.

Taskset [Metz et al., 2020] provides a total of roughly 29 million learning curves for over 1162 different deep learning tasks. Here, a task is defined as optimizing a given neural network architecture for a given supervised learning problem, and the curves correspond to using 5 different optimizers, using 1000 different hyperparameter settings, and 5 different random seeds. Following Wistuba et al. [2022], we only use the validation log loss curves of a small subset of 12 tasks, that consider training different RNN architectures, with varying architectural parameters, for 50 epochs on the IMDB sentiment analysis dataset [Maas et al., 2011], a binary text classification task. As our objective was to evaluate the robustness of our approach, we avoided excluding ill-behaved (e.g., diverging) curves. However, upon inspecting the data, we found that on some of these tasks up to $90\%$ of the curves fail to significantly improve upon the initial model performance, some of which diverge almost instantly. In combination with the normalization procedure described in Appendix A, the majority of curves are effectively mapped onto the same narrow range, producing a constant trend (at 0 in case of divergence). While this is reasonable in practice, and both LC-PFN and MCMC variants predict this constant trend with very high confidence, it does create a bias in our evaluation towards methods excelling at extrapolating poor curves. To eliminate and investigate this bias, we selected the 5000 curves used in Section 4.2, such that 2500 are taken from the top $10\%$ best $^{4}$ curves per task, and 2500 from the $90\%$ others. In Figure 4, we compared methods on the "good" curves only, an evaluation on the 2500 "bad" curves is presented in Figure 8 (b). In Section 4.3, we do not make this distinction and consider all 25000 curves per task, in 40 different random orderings.

PD1 is a recent benchmark that Wang et al. [2022] describe as collecting “a large multi-task hyperparameter tuning dataset by training tens of thousands of configurations of near-state-of-the-art models on popular image and text datasets, as well as a protein sequence dataset”. We access this benchmark through the synetune [Salinas et al., 2022] library interface, which provides learning curve data for 23 tasks, where the number of curves provided, as well as the curve length, varies per task. Here, we limit our selection to log loss curves for the three tasks that consider training Transformer architectures for 22 epochs on language modeling (lm1b), 1414 epochs on translation (WMT), and 38 epochs on protein sequences (UniRef50). To handle the very long learning curves on the translation task, we subsample these aggressively, feeding only every 14th observation into the LC-PFN. In Section 4.2, we select 5 000 curves from these three tasks uniformly at random. In Section 4.3, we consider all curves per task, in 40 different random orderings. Finally, we use the optimal loss achieved by any of the runs as $l_{soft} > 0$ in our normalization. While unknown in practice, we do not expect qualitative differences in evaluation if a rough estimate were to be used instead.

![](images/7c68b07b1e6844e4a5ef6f7dd444846fb95990263c661001bac819666b13158d.jpg)

<details>
<summary>line</summary>

| t  | y (Red) | y (Green) | y (Purple) | y (Yellow) | y (Black) |
|----|---------|-----------|------------|------------|-----------|
| 0  | 0.2     | 0.3       | 0.7        | 0.8        | 0.9       |
| 10 | 0.4     | 0.4       | 0.5        | 0.8        | 0.9       |
| 20 | 0.4     | 0.5       | 0.6        | 0.8        | 0.9       |
| 30 | 0.4     | 0.5       | 0.7        | 0.8        | 0.9       |
| 40 | 0.4     | 0.6       | 0.8        | 0.8        | 0.9       |
| 50 | 0.4     | 0.6       | 0.8        | 0.8        | 0.9       |
</details>

![](images/95b64bef8a8b6c9677f48fd62e9ca5f16013f3c6d0bd01717ef7ab10a2f83a34.jpg)

<details>
<summary>line</summary>

| t   | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 | Line 6 | Line 7 |
| --- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| 0   | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    | 0.0    |
| 20  | 0.2    | 0.3    | 0.4    | 0.5    | 0.6    | 0.7    | 0.8    |
| 40  | 0.3    | 0.4    | 0.5    | 0.6    | 0.7    | 0.8    | 0.9    |
| 60  | 0.4    | 0.5    | 0.6    | 0.7    | 0.8    | 0.9    | 1.0    |
| 80  | 0.5    | 0.6    | 0.7    | 0.8    | 0.9    | 1.0    | 1.1    |
| 100 | 0.6    | 0.7    | 0.8    | 0.9    | 1.0    | 1.1    | 1.2    |
</details>

![](images/0dc9fed89fbf5e6843442ddbcbd53687f2e4827cc70030d0b057bcce8cc0455f.jpg)

<details>
<summary>line</summary>

| t  | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 | Line 6 | Line 7 | Line 8 |
|----|--------|--------|--------|--------|--------|--------|--------|--------|
| 0  | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    |
| 10 | 0.7    | 0.7    | 0.7    | 0.7    | 0.7    | 0.7    | 0.7    | 0.7    |
| 20 | 0.75   | 0.75   | 0.75   | 0.75   | 0.75   | 0.75   | 0.75   | 0.75   |
| 30 | 0.7    | 0.7    | 0.7    | 0.7    | 0.7    | 0.7    | 0.7    | 0.7    |
| 40 | 0.65   | 0.65   | 0.65   | 0.65   | 0.65   | 0.65   | 0.65   | 0.65   |
| 50 | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    | 0.6    |
</details>

![](images/a9d6bad031a9259e7dcafa02ae8b22a92483ac75d8a6b21d301cc41aabdb5335.jpg)

<details>
<summary>line</summary>

| t   | PD1    |
| --- | ------ |
| 0   | 0.0    |
| 20  | 0.6    |
| 40  | 0.7    |
| 60  | 0.8    |
| 80  | 0.85   |
| 100 | 0.9    |
</details>

(a) Learning curves sourced from each benchmark

![](images/b9c572ca45d3e9dbc1781c35aa825f981e87cca546cc56d1ed076b5f668aa1a5.jpg)

<details>
<summary>line</summary>

| Percentage of the curve observed | Average rank Log-likelihood values (Orange) | Average rank Log-likelihood values (Blue) | Average rank Log-likelihood values (Green) |
| -------------------------------- | ------------------------------------------- | ----------------------------------------- | ------------------------------------------ |
| 5                                | 2.1                                         | 1.9                                       | 1.9                                        |
| 10                               | 2.3                                         | 1.8                                       | 1.7                                        |
| 15                               | 2.4                                         | 1.8                                       | 1.7                                        |
| 20                               | 2.4                                         | 1.8                                       | 1.7                                        |
| 25                               | 2.4                                         | 1.8                                       | 1.7                                        |
| 30                               | 2.4                                         | 1.8                                       | 1.7                                        |
| 35                               | 2.4                                         | 1.8                                       | 1.7                                        |
| 40                               | 2.4                                         | 1.8                                       | 1.7                                        |
| 45                               | 2.4                                         | 1.8                                       | 1.7                                        |
</details>

![](images/cab867ed4b58db71bb4add81abd6076727ed715b1a797ccad64f07e4bb113c6e.jpg)

<details>
<summary>line</summary>

| Percentage of the curve observed | LC-PFN | MCMG |
| -------------------------------- | ------ | ---- |
| 5                                | 1.6    | 2.3  |
| 10                               | 1.7    | 2.4  |
| 15                               | 1.7    | 2.4  |
| 20                               | 1.7    | 2.4  |
| 25                               | 1.7    | 2.4  |
| 30                               | 1.7    | 2.4  |
| 35                               | 1.7    | 2.4  |
| 40                               | 1.7    | 2.4  |
| 45                               | 1.7    | 2.4  |
</details>

![](images/c18321901f49816262c66a1f8200769870f452c9c883ea3b41c93579e609cbd8.jpg)

<details>
<summary>line</summary>

| Percentage of the curve observed | MCMC(our prior) | Average rank Log-likelihood values |
| -------------------------------- | --------------- | ------------------------------------ |
| 5                                | 2.0             | 1.8                                  |
| 10                               | 2.4             | 1.5                                  |
| 15                               | 2.5             | 1.6                                  |
| 20                               | 2.4             | 1.7                                  |
| 25                               | 2.3             | 1.7                                  |
| 30                               | 2.2             | 1.7                                  |
| 35                               | 2.3             | 1.7                                  |
| 40                               | 2.2             | 1.7                                  |
| 45                               | 2.1             | 1.7                                  |
</details>

![](images/69b443b6756225a2efff6fc7021394c51b7fb54119b268903cb1e3faae9ecc95.jpg)

<details>
<summary>line</summary>

| Percentage of the curve observed | Average rank MSE values (Orange) | Average rank MSE values (Blue) | Average rank MSE values (Green) |
| --------------------------------- | --------------------------------- | ------------------------------ | ------------------------------- |
| 5                                 | 2.4                               | 1.6                            | 1.8                             |
| 10                                | 2.5                               | 1.9                            | 1.5                             |
| 15                                | 2.6                               | 1.8                            | 1.6                             |
| 20                                | 2.4                               | 1.8                            | 1.7                             |
| 25                                | 2.3                               | 1.8                            | 1.7                             |
| 30                                | 2.3                               | 1.8                            | 1.7                             |
| 35                                | 2.3                               | 1.8                            | 1.7                             |
| 40                                | 2.3                               | 1.8                            | 1.7                             |
| 45                                | 2.2                               | 1.8                            | 1.7                             |
</details>

(b) Quality of curves vs. quality of extrapolations on Taskset (average rank, lower is better)   
Figure 8: (a) Illustration showing the 5000 curves used for each of the four benchmarks in our experiments in Section 4.2. All curves are shown after normalization and subsampling, and a random subset of individual curves is highlighted. For Taskset, “good” / “bad” curves (top 10% / bottom 90% in their task) are shown in green / red, respectively. (b) Average rank of Log-likelihood and MSE values on Taskset curves for the three methods (LC-PFN, MCMC-PP, and MCMC-OP). The two leftmost plots show results on the 5000 sampled Taskset curves. The two rightmost plots provide the same comparison, but specifically for the 2500 “bad” curves. Here, we observe that MCMC-PP performs better on the “bad” curves, which we believe can be attributed to the discretization of the PPD in 1000 bins, limiting the maximal confidence / accuracy of LC-PFN, since the majority of these curves are quasi-constant (fall in the same bin) after normalization.

# C Additional analyses

# C.1 Extrapolating samples of the prior

# C.1.1 Detailed results of section 4.1

Figure 9 provides a visual representation of the log-likelihood values on prior curves for higher cutoffs, similar to Figure 3. As expected, the difference in log-likelihood values between LC-PFN and MCMC-PP decreases as more points of the curve are observed. The detailed results can be found in Table 5. Both Figure 9 and Table 5 clearly demonstrate that certain variants of LC-PFN, particularly those trained with 10M examples, consistently outperform the best variant of MCMC-PP across all considered cutoffs.

![](images/5b7eb6064999c4dd7c5b7d31913da17605f573e2ebeb5978d09483e42bf55af5.jpg)

<details>
<summary>scatter</summary>

| Group | Sample Size | Time (seconds) | Log-likelihood |
|-------|-------------|----------------|---------------|
| 20%   | 100k        | ~0.01          | ~1.5          |
| 20%   | 1M          | ~0.01          | ~1.6          |
| 20%   | 10M         | ~0.01          | ~1.7          |
| 20%   | 128         | ~0.01          | ~1.8          |
| 20%   | 256         | ~0.01          | ~1.9          |
| 20%   | 512         | ~0.01          | ~2.0          |
| 40%   | 3           | ~0.01          | ~1.6          |
| 40%   | 6           | ~0.01          | ~1.7          |
| 40%   | 12          | ~0.01          | ~1.8          |
| 40%   | 250         | ~0.01          | ~1.9          |
| 40%   | 500         | ~0.01          | ~2.0          |
| 40%   | 1000        | ~0.01          | ~2.1          |
| 40%   | 2000        | ~0.01          | ~2.2          |
| 40%   | 4000        | ~0.01          | ~2.3          |
| 80%   | 100         | ~0.01          | ~1.7          |
| 80%   | 250         | ~0.01          | ~1.8          |
| 80%   | 500         | ~0.01          | ~1.9          |
| 80%   | 1000        | ~0.01          | ~2.0          |
| 80%   | 2000        | ~0.01          | ~2.1          |
| 80%   | 4000        | ~0.01          | ~2.2          |
</details>

Figure 9: Comparison of LC-PFN and MCMC-PP with varying hyperparameters values on prior curves in terms of average runtime (lower is better) and average log-likelihood (higher is better) for different cutoff values (20%, 40%, 80%)

Table 5: Comparison of the 25 best LC-PFN and MCMC-PP variants on prior curves in terms of log-likelihood (higher is better) at 10%, 20%, 40%, and 80% cutoffs. Values in brackets correspond to one standard error. 

<table><tr><td>Method</td><td>Parameters</td><td>10%</td><td>20%</td><td>40%</td><td>80%</td><td>Avg. Runtime (s)</td></tr><tr><td>MCMC-PP</td><td>nsamples=2000, nwalkers=100, burn-in=500, thin=10</td><td>1.628 (0.01)</td><td>1.939 (0.011)</td><td>2.265 (0.01)</td><td>2.469 (0.009)</td><td>29.944 (7.6E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=2000, nwalkers=100, burn-in=500, thin=1</td><td>1.628 (0.01)</td><td>1.939 (0.011)</td><td>2.265 (0.01)</td><td>2.469 (0.009)</td><td>54.401 (3.1E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=0, thin=10</td><td>1.629 (0.01)</td><td>1.942 (0.011)</td><td>2.266 (0.009)</td><td>2.467 (0.009)</td><td>28.783 (6.2E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=0, thin=1</td><td>1.629 (0.01)</td><td>1.943 (0.011)</td><td>2.266 (0.009)</td><td>2.467 (0.009)</td><td>53.292 (4.3E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=50, thin=100</td><td>1.629 (0.01)</td><td>1.943 (0.011)</td><td>2.266 (0.009)</td><td>2.467 (0.009)</td><td>26.672 (7.3E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=100, thin=100</td><td>1.631 (0.01)</td><td>1.943 (0.011)</td><td>2.267 (0.009)</td><td>2.469 (0.009)</td><td>26.997 (7.4E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=50, thin=10</td><td>1.631 (0.01)</td><td>1.944 (0.011)</td><td>2.267 (0.009)</td><td>2.468 (0.009)</td><td>29.109 (6.3E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=50, thin=1</td><td>1.631 (0.01)</td><td>1.944 (0.011)</td><td>2.267 (0.009)</td><td>2.469 (0.009)</td><td>53.601 (4.2E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=100, thin=10</td><td>1.632 (0.01)</td><td>1.943 (0.011)</td><td>2.268 (0.01)</td><td>2.47 (0.009)</td><td>29.433 (6.4E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=100, thin=1</td><td>1.632 (0.01)</td><td>1.943 (0.011)</td><td>2.268 (0.01)</td><td>2.47 (0.009)</td><td>53.941 (4.2E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=500, thin=100</td><td>1.632 (0.01)</td><td>1.94 (0.012)</td><td>2.274 (0.01)</td><td>2.477 (0.009)</td><td>29.601 (8.2E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=500, thin=10</td><td>1.632 (0.01)</td><td>1.942 (0.011)</td><td>2.275 (0.01)</td><td>2.477 (0.009)</td><td>32.036 (7.1E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=50, burn-in=500, thin=1</td><td>1.632 (0.01)</td><td>1.942 (0.011)</td><td>2.275 (0.01)</td><td>2.477 (0.009)</td><td>56.526 (3.4E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=0, thin=100</td><td>1.637 (0.01)</td><td>1.954 (0.011)</td><td>2.274 (0.01)</td><td>2.474 (0.009)</td><td>44.076 (1.4E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=0, thin=10</td><td>1.639 (0.01)</td><td>1.956 (0.011)</td><td>2.276 (0.01)</td><td>2.476 (0.009)</td><td>48.947 (1.2E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=0, thin=1</td><td>1.639 (0.01)</td><td>1.956 (0.011)</td><td>2.276 (0.01)</td><td>2.476 (0.009)</td><td>98.033 (9.9E-01)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=50, thin=100</td><td>1.639 (0.01)</td><td>1.956 (0.011)</td><td>2.276 (0.01)</td><td>2.476 (0.009)</td><td>44.618 (1.4E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=50, thin=10</td><td>1.64 (0.01)</td><td>1.958 (0.011)</td><td>2.277 (0.01)</td><td>2.477 (0.009)</td><td>49.475 (1.2E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=50, thin=1</td><td>1.64 (0.01)</td><td>1.958 (0.011)</td><td>2.277 (0.01)</td><td>2.477 (0.009)</td><td>98.548 (1.0E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=100, thin=100</td><td>1.641 (0.01)</td><td>1.958 (0.011)</td><td>2.277 (0.01)</td><td>2.477 (0.009)</td><td>45.16 (1.4E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=100, thin=10</td><td>1.641 (0.01)</td><td>1.957 (0.011)</td><td>2.278 (0.01)</td><td>2.478 (0.009)</td><td>50.011 (1.2E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=100, thin=1</td><td>1.641 (0.01)</td><td>1.957 (0.011)</td><td>2.278 (0.01)</td><td>2.478 (0.009)</td><td>98.98 (1.1E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=500, thin=100</td><td>1.642 (0.01)</td><td>1.955 (0.011)</td><td>2.284 (0.01)</td><td>2.485 (0.009)</td><td>49.5 (1.6E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=500, thin=10</td><td>1.642 (0.01)</td><td>1.956 (0.011)</td><td>2.285 (0.01)</td><td>2.486 (0.009)</td><td>54.328 (1.3E+00)</td></tr><tr><td>MCMC-PP</td><td>nsamples=4000, nwalkers=100, burn-in=500, thin=1</td><td>1.642 (0.01)</td><td>1.956 (0.011)</td><td>2.285 (0.01)</td><td>2.486 (0.009)</td><td>103.151 (1.1E+00)</td></tr><tr><td>LC-PFN</td><td>Nb data=100k, nlayers=6, emsize=128</td><td>1.242 (0.001)</td><td>1.508 (0.001)</td><td>1.652 (0.001)</td><td>1.709 (0.002)</td><td>0.007 (4.6E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=100k, nlayers=12, emsize=128</td><td>1.254 (0.001)</td><td>1.522 (0.001)</td><td>1.669 (0.001)</td><td>1.73 (0.002)</td><td>0.012 (8.3E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=100k, nlayers=6, emsize=256</td><td>1.267 (0.002)</td><td>1.597 (0.001)</td><td>1.764 (0.001)</td><td>1.843 (0.002)</td><td>0.011 (8.2E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=100k, nlayers=12, emsize=256</td><td>1.384 (0.001)</td><td>1.664 (0.001)</td><td>1.833 (0.001)</td><td>1.9 (0.002)</td><td>0.021 (1.6E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=100k, nlayers=3, emsize=512</td><td>1.392 (0.001)</td><td>1.679 (0.001)</td><td>1.871 (0.001)</td><td>1.959 (0.002)</td><td>0.016 (1.4E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=100k, nlayers=6, emsize=512</td><td>1.439 (0.001)</td><td>1.745 (0.001)</td><td>1.947 (0.001)</td><td>2.034 (0.002)</td><td>0.028 (2.5E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=100k, nlayers=12, emsize=512</td><td>1.504 (0.001)</td><td>1.817 (0.001)</td><td>2.013 (0.001)</td><td>2.091 (0.002)</td><td>0.048 (4.2E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=1M, nlayers=3, emsize=128</td><td>1.528 (0.001)</td><td>1.894 (0.001)</td><td>2.159 (0.001)</td><td>2.279 (0.002)</td><td>0.004 (2.8E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=1M, nlayers=6, emsize=128</td><td>1.581 (0.001)</td><td>1.949 (0.001)</td><td>2.202 (0.001)</td><td>2.308 (0.002)</td><td>0.007 (4.6E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=1M, nlayers=3, emsize=256</td><td>1.583 (0.001)</td><td>1.967 (0.001)</td><td>2.239 (0.001)</td><td>2.357 (0.002)</td><td>0.007 (5.0E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=10M, nlayers=3, emsize=128</td><td>1.585 (0.001)</td><td>1.989 (0.001)</td><td>2.282 (0.002)</td><td>2.431 (0.003)</td><td>0.004 (2.8E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=1M, nlayers=12, emsize=128</td><td>1.621 (0.001)</td><td>1.982 (0.001)</td><td>2.223 (0.001)</td><td>2.326 (0.002)</td><td>0.012 (8.2E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=1M, nlayers=3, emsize=512</td><td>1.624 (0.001)</td><td>2.011 (0.001)</td><td>2.291 (0.001)</td><td>2.417 (0.002)</td><td>0.016 (1.4E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=1M, nlayers=6, emsize=256</td><td>1.627 (0.001)</td><td>2.007 (0.001)</td><td>2.269 (0.001)</td><td>2.381 (0.002)</td><td>0.012 (8.7E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=10M, nlayers=3, emsize=256</td><td>1.654 (0.001)</td><td>2.044 (0.001)</td><td>2.347 (0.002)</td><td>2.493 (0.003)</td><td>0.006 (4.7E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=1M, nlayers=12, emsize=256</td><td>1.667 (0.001)</td><td>2.035 (0.001)</td><td>2.294 (0.001)</td><td>2.406 (0.002)</td><td>0.02 (1.5E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=1M, nlayers=6, emsize=512</td><td>1.672 (0.001)</td><td>2.051 (0.001)</td><td>2.323 (0.001)</td><td>2.441 (0.002)</td><td>0.028 (2.4E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=10M, nlayers=6, emsize=128</td><td>1.673 (0.001)</td><td>2.058 (0.001)</td><td>2.352 (0.001)</td><td>2.478 (0.003)</td><td>0.007 (4.6E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=1M, nlayers=12, emsize=512</td><td>1.699 (0.001)</td><td>2.072 (0.001)</td><td>2.344 (0.001)</td><td>2.46 (0.002)</td><td>0.051 (4.6E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=10M, nlayers=3, emsize=512</td><td>1.707 (0.001)</td><td>2.083 (0.001)</td><td>2.378 (0.002)</td><td>2.517 (0.003)</td><td>0.017 (1.4E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=10M, nlayers=12, emsize=128</td><td>1.717 (0.001)</td><td>2.094 (0.001)</td><td>2.377 (0.001)</td><td>2.498 (0.003)</td><td>0.012 (8.1E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=10M, nlayers=6, emsize=256</td><td>1.721 (0.001)</td><td>2.098 (0.001)</td><td>2.387 (0.002)</td><td>2.514 (0.003)</td><td>0.011 (8.2E-04)</td></tr><tr><td>LC-PFN</td><td>Nb data=10M, nlayers=6, emsize=512</td><td>1.744 (0.001)</td><td>2.125 (0.001)</td><td>2.404 (0.002)</td><td>2.527 (0.003)</td><td>0.026 (2.3E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=10M, nlayers=12, emsize=256</td><td>1.746 (0.001)</td><td>2.124 (0.001)</td><td>2.403 (0.001)</td><td>2.522 (0.003)</td><td>0.02 (1.5E-03)</td></tr><tr><td>LC-PFN</td><td>Nb data=10M, nlayers=12, emsize=512</td><td>1.758 (0.001)</td><td>2.133 (0.001)</td><td>2.403 (0.002)</td><td>2.519 (0.003)</td><td>0.05 (4.5E-03)</td></tr></table>

![](images/42537a0826449b44d16f2cb26c9ccd5bbd488890ca90ebc676cdda088fb6747f.jpg)

<details>
<summary>line</summary>

| Time (seconds) | LC-PFN (Nb data=10M, nlayers=12, emsize=512) | LSE | MAP | Default |
| -------------- | ------------------------------------------ | --- | --- | ------- |
| 0.1            | 1.78                                       |     |     |         |
| 10             | 1.62                                       | 1.62 | 1.49 | 1.28    |
| 100            | 1.67                                       | 1.67 | 1.63 | 1.63    |
| 1000           | 1.75                                       | 1.75 | 1.73 | 1.75    |
</details>

Figure 10: runtime vs log-likelihood values of MCMC-PP, using thinning 100, with different initialization strategies and sample sizes (per walker), averaged across 1 000 curves generated i.i.d. from the prior. The blue point represents LC-PFN, with the dotted line as a reference for its LL value. The shaded areas correspond to $\pm$ 1 standard error.

# C.1.2 Effect of MCMC chain length and initialization

In our experiments in Section 4.1 we observed that the quality of MCMC inference increases with chain length. In particular, our best MCMC-PP variant (M3 in Figure 3), uses the same hyperparameter settings as Domhan et al. [2015] (M1), but with a double as long chain (discounting the burn-in). In this section, we investigate whether the chains considered are too short and whether increasing chain lengths further will eventually cause MCMC-PP to outperform LC-PFN. To study this, we will run MCMC-PP to collect up to 100 000 samples per walker (50× more than Domhan et al. [2015]). To reduce the computational cost of this experiment, we only extrapolate 1 000 (instead of 10 000) prior curves, on cutoff 10%, and use MCMC-PP with a 100-thinning strategy (M2). This variant is more compute-efficient and the minor negative effect of thinning is expected to further decrease with increasing chain length. Concurrently, we study the effect of initialization. There are different ways of initializing MCMC chains and the chosen strategy can strongly affect performance if chains are too short. Domhan et al. [2015] initializes the chain by setting the parameters for each of the K basis curves to their Least-Squared Estimates (LSE). Weights are initialized to be $\frac{1}{K}$ . If this initial point violates the constraints imposed by the prior, a default starting point is used instead. As an ablation, we compare it to an initialization always starting at the default (default). As another ablation, we compare to a strategy using the maximum a posteriori estimate (MAP) as a starting point instead, which unlike LSE takes the likelihood under the prior into account.

Figure 10 shows the quality and cost of inferences for each of the initialization strategies when collecting up to 100 000 samples (per walker). While these results clearly show that increasing chain length continues to improve performance. The trends we observe, suggest that MCMC-PP could eventually attain or even overtake the best LC-PFN. That being said, the best MCMC we considered has $20\ 000\times$ longer runtimes than that LC-PFN requires, yet it does not quite reach the same performance, so outperforming it would require impractically long chains. When optimistically extending the trends, we estimate to need at least $10\times$ longer chains, and inference times of multiple hours. In terms of initialization, we observe that MCMC performance for shorter chains indeed more strongly depends on the initialization strategy, where the LSE strategy by Domhan et al. [2015] does best for short chains, followed by MAP and default. When increasing chain length, we find that differences get smaller and order reverses such that the fixed default initialization is best, suggesting that greedy initialization (LSE, MAP) may hurt performance on some curves.

# C.2 Extrapolating real-world learning curves

In addition to the average rank plots presented in Section 4.2, we also perform pairwise comparisons of the log-likelihood and mean squared error values among the different methods (LC-PFN and MCMC variants). This approach allows for a more detailed analysis, as it compares the absolute values of the metrics. Unlike the rank plots, these pair and curvewise plots provide insights into outliers which thus further enhance our understanding of the results.

![](images/861746c918241ab316bcd90f316853765be1dde714a0f93d1e9448e2be3dd285.jpg)

![](images/2436ec69ab1d8eafda226803aa7385d7da846d0bb349ed9c8b75593d3059ddaf.jpg)

<details>
<summary>scatter</summary>

| MCMC(our prior) | LC-PFN |
| --------------- | ------ |
| -20             | -20    |
| -10             | -10    |
| 0               | 0      |
| 10              | 10     |
| 20              | 20     |
</details>

![](images/197d1edb8d22b345accd4b9fb7b864c978b771a28a87f75d2b97b84ac0434e42.jpg)

<details>
<summary>scatter</summary>

| MCMC(our prior) | Value |
| --------------- | ----- |
| -20             | 0     |
| -10             | 0     |
| 0               | 0     |
| 10              | 0     |
| 20              | 0     |
</details>

![](images/32b74fd4a758efc1780481c14ad77fa819ae05a61d1fd08c7081fb7b04e8156a.jpg)

<details>
<summary>scatter</summary>

| MCMC(our prior) | Value |
| --------------- | ----- |
| -20             | 0     |
| -10             | 10    |
| 0               | 20    |
| 10              | 30    |
| 20              | 40    |
</details>

![](images/17edc79650913806f3ee6535c539a2a9cc3f2cc47002c2620aca5551d8c591ff.jpg)

<details>
<summary>scatter</summary>

| MCMC(our prior) | Value |
| --------------- | ----- |
| -20             | 0     |
| -10             | 0     |
| 0               | 0     |
| 10              | 0     |
| 20              | 0     |
</details>

(a) PFN vs MCMC with our prior (MCMC-PP)   
![](images/cea826e43631c16205100dc861ccc81940334cece684090c9a8d05f06f4ecf2e.jpg)

<details>
<summary>scatter</summary>

| MCMC(original) | LC-PFN |
| -------------- | ------ |
| -20            | -20    |
| -10            | -10    |
| 0              | 0      |
| 10             | 10     |
| 20             | 20     |
</details>

![](images/25a86614db6dfdc43635e17a475167949c7a8631ae27621af7a0fb794c5563d1.jpg)

<details>
<summary>scatter</summary>

| MCMC(original) | Value |
| -------------- | ----- |
| -20            | 0     |
| -10            | 0     |
| 0              | 0     |
| 10             | 0     |
| 20             | 0     |
</details>

![](images/d967ef375ba32b4928df0cf6d6b8a29bbbcd14b7ca3e06616cba340953e0db47.jpg)

<details>
<summary>scatter</summary>

| MCMC(original) | Value |
| -------------- | ----- |
| -20            | 0     |
| -10            | 0     |
| 0              | 0     |
| 10             | 0     |
| 20             | 0     |
</details>

![](images/64fa10e2c51b8dfd2aeb3d7b90c28fe1e640bd1f0b85422e4e1ace2ffa13d695.jpg)

<details>
<summary>scatter</summary>

| MCMC(original) | PD1 |
| -------------- | --- |
| -20            | -   |
| -10            | -   |
| 0              | -   |
| 10             | -   |
| 20             | -   |
</details>

(b) PFN vs MCMC as in Domhan et al. [2015] (MCMC-OP)   
Figure 11: Pairwise comparison of log-likelihood (higher is better) between PFN and MCMC.

Figure 11 presents a pairwise comparison of the log-likelihood values between our PFN method and the two variants of MCMC, per curve, considering different cutoff values on the four benchmark datasets. Figure 12 presents the same comparison based on mean squared error (MSE).

We observe that LC-PFN compares favorably to MCMC-PP on PD1 and NAS-Bench-201 tasks. Both methods demonstrate comparable performance on the Taskset dataset. However, it is worth noting that for some curves of the LCBench dataset, MCMC-PP obtains very high log-likelihood and low MSE scores, while LC-PFN falls short in this aspect. We believe this discrepancy can be attributed to the fact that the LCBench dataset contains a significant number of constant curves (see Figure 8a) that fall within the same bin of the discretized PPD, effectively limiting the maximal confidence / accuracy of LC-PFN. Looking at MSE specifically, we find that for some LCBench curves, at high cutoffs, LC-PFN obtains very high errors, while MCMC-PP does not. Upon closer inspection, we found that these curves quickly converge to a value close to the optimal accuracy, and while LC-PFN captures this trend for lower cutoffs, it suddenly fails for larger cutoffs (see 5th example in Figure 18). This can likely be explained by the fact that these curves are not adequately captured by the prior proposed in Section 3.2 and this seems to be one of the few cases where LC-PFN generalizes poorly out of distribution.

When comparing with MCMC-OP, we observe that MCMC-OP has relatively few outliers in terms of log-likelihood, both in the positive and negative sense. This can likely be explained by the prior of MCMC-OP, which is more flexible (i.e. has higher entropy) than ours and making MCMC-OP thus less confident about its predictions. However, MCMC-OP performs clearly worse in terms of MSE, suggesting that the median of the high entropy PPDs produced by this method do not provide a good point estimate.

![](images/b7a167ec13cddf8e9417ceaa2121cb6bdbf7f2f1573b1cb8dbe4fbbed722ae7f.jpg)  
Figure 12: Pairwise comparison of MSE values (lower is better) between PFN and MCMC.

# C.3 Application: Extrapolation-based early stopping in model selection

In what follows, we further analyze and discuss our experiments described in Section 4.3.

# C.3.1 Results on individual tasks

While Figure 5 shows the results of our early stopping experiments for each PD1 task, results for the three remaining benchmarks are summarized by averaging them across all tasks. Since this may hide task-dependent variation and outliers, we have a closer look at the performance on individual tasks. Figure 13 shows the results for each of 35 LCBench tasks. Here, we observe that relative performances vary per task. The aggressive Patience (k) (i.e., low k) baselines perform best on some tasks, but fail on others. The LC-PFN based termination criteria consistently perform well on each task, obtaining a 2-6× speed-up (w.r.t. the final performance of no-stop) on 30 of the 35 tasks, and no significant slow-down on the remaining five. Figure 14 shows the results for each of the 3 NAS-Bench-201 tasks. Here, we observe failure on all 3 tasks, where the degree of failure seems correlated with the complexity of the image classification task. Figure 15 shows the results for each of the 12 Taskset tasks. Here, the LC-PFN based termination criteria consistently perform well, obtaining 2-6× speed-ups on all 12 tasks. We conclude that Figure 5 accurately reflects relative performances on all four benchmarks.

![](images/c2a072fe6374f3ff2a95cd9ec2ba56d9d72cca0c92e615a6c8b208dba492ce5c.jpg)  
Figure 13: Average anytime regret (lower is better) obtained by all early stopping criteria for 40 run orderings for each of the 35 LCBench tasks.

![](images/bf6578cf38603f687560bc34f32fc51808ecc33fc60c3ff54c4ac43397b58479.jpg)

<details>
<summary>line</summary>

| Dataset | Metric | No Stop | Patience(0) | Patience(1) | Patience(5) | Patience(10) | Patience(20) | Patience(50) | LC-PFN(0.95,coarse) | LC-PFN(0.95,fine) |
|---|---|---|---|---|---|---|---|---|---|---|
| - | - | - | - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - | - | - | - |
| - | - | - | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| CIFAR-201 (cifar10) | Average Regret | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
| CIFAR-201 (cifar10) (Cifar10) | Average Regret | ~0.05 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
| CIFAR-201 (cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10/CFAR-20) (Cifar10) (Cifar10) (Cifar10/CFAR-20) (Cifar10/CFAR-20) (Cifar10/CFAR-20) (Cifar10/CFAR-20) (Cifar10/CFAR-20) (Cifar10/CFAR-20) (Cifar10/CFAR-20) (Cifar10/CFAR-20) (Cifar10/CFAR-20) (LC-PFN(0.95,fine)) (Cifar10/CFAR-20) (LC-PFN(0.95,fine)) (Cifar10/CFAR-20) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) (LC-PFN(0.95,fine)) |
| CIFAR-201 (cifar10) | Average Regret | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 | ~0.1 |
| CIFAR-201 (cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar10) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (Cifar37) (C-FIN(8,coarse)) (C-FIN(8,coarse)) (C-FIN(8,coarse)) (C-FIN(8,coarse)) (C-FIN(8,coarse)) (C-FIN(8,coarse)) (C-FIN(8,coarse)) (C-FIN(8,coarse)) (C-FIN(8,coarse)) (C-FIN(8,coarse)) (C-FIN(8,coarse)), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse). LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse); LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse), LC-PFEN(8,coarse)
</details>

Figure 14: Average anytime regret (lower is better) obtained by all early stopping criteria for 40 run orderings for each of the 3 NAS-Bench-201 tasks.

![](images/0d97316003160198aef48f462a75d7c2d4dc24823ed024152a7034a04686ea76.jpg)  
Figure 15: Average anytime regret (lower is better) obtained by all early stopping criteria for 40 run orderings for each of the 12 Taskset tasks.

# C.3.2 Parameter sensitivity analysis

In our comparison in Figure 5, we considered two variants of the LC-PFN based termination criterion, varying the frequency at which it is applied, and found the fine-grained strategy, applying it every epoch, to perform best. However, many variants of this scheme exist, and in what follows we investigate the impact of some of our other choices on performance.

Confidence level: In our experiments, we terminate a training run if we are confident it will not improve upon the best model seen so far. Following prior art [Domhan et al., 2015], we adopted a confidence threshold of 0.95. Figure 16 (a) shows the anytime performance of variants using lower (0.90) and higher (0.99, 0.999) confidence levels. We observe that all of these perform relatively well, suggesting some degree of robustness. The speed-ups obtained using higher confidence levels are mostly lower (up to $2\times$ ), but more consistent. In particular, we only observe minimal slow-down on NAS-Bench-201 at confidence level 0.999, and higher confidences are also beneficial for the PD1 protein sequence (UniRef50) task.

Minimal cutoff: While we apply our predictive termination criterion every epoch, we do not apply it “as of the first epoch”, but only after two observations are available $T \geq 2$ . Since LC-PFN’s predictions are conditioned on the partial curve only, it will predict the prior for T = 0. Beyond not allowing us to discriminate between runs, it introduces a potential failure mode for easy tasks (many training runs obtain models close to the theoretical optimum) and LC-PFN (unaware of task hardness) will terminate runs before they even began because it is very unlikely to obtain a better model under the prior. Figure 16 (b) shows the anytime performance of variants that apply the criterion as soon as one, three, four, or five observations are made. We find that a choice of one (instead of two) only negatively impacts performance on LCBench (containing some very easy tasks). A minimal cutoff of two works well on all benchmarks, with minor slow-downs for the higher minimal cutoffs.

# C.4 Qualitative plots of learning curve extrapolation

To complement the quantitative evaluation in the main paper, we provide examples of extrapolations, at the four different cutoffs, for seven curves from the prior (Figure 17), LCBench (Figure 18), NAS-Bench-201 (Figure 19), Taskset (Figure 20), and PD1 (Figure 21). The plots show the median and two-sided $90\%$ confidence interval of the PPDs inferred using the LC-PFN and MCMC-PP variant considered in Section 4.2. We observe that while the quality of the extrapolations produced by LC-PFN varies strongly, they are mostly logical, given the data observed and the prior used, and we find that the inferences of MCMC-PP are rarely preferable.

# D Breakdown computational cost

Overall, reproducing all our experiments in the main paper requires approximately 163 CPU days and 60 GPU hours on our systems (GPU: NVIDIA (R) GeForce (R) RTX 2080, CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz). These costs break down as follows:

- Section 4.1 entails 80 CPU days to run all MCMC-PP variants on the 10 000 sampled curves. The overall prediction time of LC-PFN is negligible (<1 hr for each variant). However, for LC-PFN, there is an initial training cost, which amounts to approximately 60 GPU hours to train all 27 LC-PFN variants.   
- Section 4.2 requires 80 CPU days to run the MCMC algorithm on the considered real curve benchmarks.   
- Section 4.3 involves a maximum of 3 CPU days to replicate the early stopping results on all the benchmarks for both LC-PFN and the baselines.

![](images/6431586c9e9991862ae4c0010ababde36c5d3066d4f3c1efa7bd14608e6f1a5d.jpg)  
no-stop LC-PFN(0.90,fine>=2) LC-PFN(0.95,fine>=2) LC-PFN(0.99,fine>=2) LC-PFN(0.999,fine>=2)

(a) Varying confidence levels $(1 - \delta)$   
![](images/e79690d483f1f81732bd410284ad141c1235c58556a245fec6134ffa213c2e72.jpg)  
no-stop LC-PFN(0.95,fine>=1) LC-PFN(0.95,fine>=2) LC-PFN(0.95,fine>=3) LC-PFN(0.95,fine>=4) LC-PFN(0.95,fine>=5)

Figure 16: Parameter sensitivity analysis of our fine-grained LC-PFN based termination criterion in terms of the average regret (lower is better).

![](images/a1238631dac58c69d7293e7f65e170d1166dcaa159ee21a045dfd584309519d4.jpg)  
Figure 17: Extrapolations of 7 different curves from the prior at 10, 20, 40, and 80 cutoff

![](images/114e2666f6510b17e64a59f67bd55bae0b5a396adc410b3bf0f9f7ffedf45af5.jpg)

<details>
<summary>line</summary>

| Epochs | MCRC PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.1  |
| 10     | 0.8     | 0.7    | 0.7    | 0.6  |
| 20     | 0.85    | 0.75   | 0.75   | 0.65 |
| 30     | 0.9     | 0.8    | 0.8    | 0.7  |
| 40     | 0.95    | 0.85   | 0.85   | 0.75 |
| 50     | 1.0     | 0.9    | 0.9    | 0.8  |
</details>

![](images/47dc0f5b55b1734d7615b2a7f66a830ad678e467ba0ca2a0af68e1d792db72bc.jpg)

<details>
<summary>line</summary>

| Epochs | MCHC, PP | LC PPN | target | data |
| ------ | -------- | ------ | ------ | ---- |
| 0      | 0.6      | 0.6    | 0.6    | 0.1  |
| 10     | 0.7      | 0.7    | 0.7    | 0.6  |
| 20     | 0.75     | 0.75   | 0.75   | 0.65 |
| 30     | 0.78     | 0.78   | 0.78   | 0.68 |
| 40     | 0.8      | 0.8    | 0.8    | 0.7  |
| 50     | 0.8      | 0.8    | 0.8    | 0.7  |
</details>

![](images/beb7ca824acf1eca0e190ff9cb390dcebcebd4eaf55c6cda919d6e7724429f8b.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC_PP | LC.PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.1     | 0.1    | 0.1    | 0.1  |
| 5      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.7     | 0.7    | 0.7    | 0.7  |
| 15     | 0.7     | 0.7    | 0.7    | 0.7  |
| 20     | 0.7     | 0.7    | 0.7    | 0.7  |
| 25     | 0.7     | 0.7    | 0.7    | 0.7  |
| 30     | 0.7     | 0.7    | 0.7    | 0.7  |
| 35     | 0.7     | 0.7    | 0.7    | 0.7  |
| 40     | 0.7     | 0.7    | 0.7    | 0.7  |
| 45     | 0.7     | 0.7    | 0.7    | 0.7  |
| 50     | 0.7     | 0.7    | 0.7    | 0.7  |
</details>

![](images/fa9edcf27c5c14cb20e3cc89c8375a261fa1f1059f26c72e7d473812d19b847c.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC.PP | LC.PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.1  |
| 5      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.65    | 0.65   | 0.65   | 0.65 |
| 15     | 0.68    | 0.68   | 0.68   | 0.68 |
| 20     | 0.7     | 0.7    | 0.7    | 0.7  |
| 25     | 0.7     | 0.7    | 0.7    | 0.7  |
| 30     | 0.7     | 0.7    | 0.7    | 0.7  |
| 35     | 0.7     | 0.7    | 0.7    | 0.7  |
| 40     | 0.7     | 0.7    | 0.7    | 0.7  |
| 45     | 0.7     | 0.7    | 0.7    | 0.7  |
| 50     | 0.7     | 0.7    | 0.7    | 0.7  |
</details>

![](images/0fd5270644b7a45f58521ff56c9501a1039eebdef4389f66fc8d7f34d044cac9.jpg)

<details>
<summary>line</summary>

| Epochs | MCHC PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 10     | 0.3     | 0.3    | 0.4    | 0.2  |
| 20     | 0.4     | 0.4    | 0.5    | 0.3  |
| 30     | 0.45    | 0.45   | 0.55   | 0.35 |
| 40     | 0.5     | 0.5    | 0.6    | 0.4  |
| 50     | 0.55    | 0.55   | 0.65   | 0.45 |
</details>

![](images/52ee815e718d4898fc120ec4970d83006089b755d93db883fa28eef7e09836ff.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC-PP | LC-PN | target | data |
| ------ | ------- | ----- | ------ | ---- |
| 0      | 0.0     | 0.0   | 0.0    | 0.0  |
| 10     | 0.4     | 0.4   | 0.4    | 0.4  |
| 20     | 0.5     | 0.5   | 0.5    | 0.5  |
| 30     | 0.6     | 0.6   | 0.6    | 0.6  |
| 40     | 0.7     | 0.7   | 0.7    | 0.7  |
| 50     | 0.8     | 0.8   | 0.8    | 0.8  |
</details>

![](images/652899f1f379c9c3ea4142624a581ad7afc828a6037f70b89b46559eb587d6db.jpg)

<details>
<summary>line</summary>

| Epochs | MCFC-PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 10     | 0.4     | 0.4    | 0.4    | 0.4  |
| 20     | 0.5     | 0.5    | 0.5    | 0.5  |
| 30     | 0.5     | 0.5    | 0.5    | 0.5  |
| 40     | 0.5     | 0.5    | 0.5    | 0.5  |
| 50     | 0.5     | 0.5    | 0.5    | 0.5  |
</details>

![](images/bbc7477b467e3137e683eb324de6293f034cceb72fd35a97be40aa462241d5e6.jpg)

<details>
<summary>line</summary>

| Epochs | MCHC-PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 10     | 0.4     | 0.4    | 0.4    | 0.4  |
| 20     | 0.5     | 0.5    | 0.5    | 0.5  |
| 30     | 0.5     | 0.5    | 0.5    | 0.5  |
| 40     | 0.5     | 0.5    | 0.5    | 0.5  |
| 50     | 0.5     | 0.5    | 0.5    | 0.5  |
</details>

![](images/4f3aa5379b5341531d0cd0d17d9eff24a74e15ebfcf713aa954e7b0186a5f10e.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.4     | 0.4    | 0.4    | 0.4  |
| 50     | 0.8     | 0.8    | 0.8    | 0.8  |
</details>

![](images/9df33a8e58638440364956e4dc97ee03fecb20e9c86332b04ea401fc15bad0fa.jpg)

<details>
<summary>line</summary>

| Epochs | MC3C-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.4     | 0.4    | 0.4    | 0.4  |
| 10     | 0.7     | 0.6    | 0.7    | 0.7  |
| 20     | 0.7     | 0.6    | 0.7    | 0.7  |
| 30     | 0.7     | 0.6    | 0.7    | 0.7  |
| 40     | 0.7     | 0.6    | 0.7    | 0.7  |
| 50     | 0.7     | 0.6    | 0.7    | 0.7  |
</details>

![](images/fe2a0d2f367f62c889dfb3c8c77e6b4a3b05c3bae5d15599f23f864efc398d99.jpg)

<details>
<summary>line</summary>

| Epochs | MCCHC-PP | LC-PPI | target | data |
| ------ | -------- | ------ | ------ | ---- |
| 0      | 0.4      | 0.4    | 0.4    | 0.4  |
| 5      | 0.6      | 0.6    | 0.6    | 0.6  |
| 10     | 0.7      | 0.7    | 0.7    | 0.7  |
| 15     | 0.75     | 0.75   | 0.75   | 0.75 |
| 20     | 0.75     | 0.75   | 0.75   | 0.75 |
| 25     | 0.75     | 0.75   | 0.75   | 0.75 |
| 30     | 0.75     | 0.75   | 0.75   | 0.75 |
| 35     | 0.75     | 0.75   | 0.75   | 0.75 |
| 40     | 0.75     | 0.75   | 0.75   | 0.75 |
| 45     | 0.75     | 0.75   | 0.75   | 0.75 |
| 50     | 0.75     | 0.75   | 0.75   | 0.75 |
</details>

![](images/fa6e6e6241cef821ffa2dbe17fd29f88d5c598cf7068da4c86617bca5db3a6b6.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.4     | 0.4    | 0.4    | 0.4  |
| 5      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.7     | 0.7    | 0.7    | 0.7  |
| 15     | 0.7     | 0.7    | 0.7    | 0.7  |
| 20     | 0.7     | 0.7    | 0.7    | 0.7  |
| 25     | 0.7     | 0.7    | 0.7    | 0.7  |
| 30     | 0.7     | 0.7    | 0.7    | 0.7  |
| 35     | 0.7     | 0.7    | 0.7    | 0.7  |
| 40     | 0.7     | 0.7    | 0.7    | 0.7  |
| 45     | 0.7     | 0.7    | 0.7    | 0.7  |
| 50     | 0.7     | 0.7    | 0.7    | 0.7  |
</details>

![](images/3b15085eb06200ff7e1c447ad6ccf4c67a4dbe0106524bcaedb98230ded0a029.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC_RP | LC_PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.2  |
| 10     | 0.7     | 0.7    | 0.7    | 0.6  |
| 20     | 0.75    | 0.75   | 0.75   | 0.65 |
| 30     | 0.8     | 0.8    | 0.8    | 0.7  |
| 40     | 0.85    | 0.85   | 0.85   | 0.75 |
| 50     | 0.9     | 0.9    | 0.9    | 0.8  |
</details>

![](images/e177ea683fdd7a5eb6c571d612045672b816aefd8617803ce8210742ebca3df5.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC, PP | LC PFN | target | data |
| ------ | -------- | ------ | ------ | ---- |
| 0      | 0.2      | 0.2    | 0.2    | 0.2  |
| 10     | 0.7      | 0.7    | 0.7    | 0.7  |
| 20     | 0.8      | 0.8    | 0.8    | 0.8  |
| 30     | 0.85     | 0.85   | 0.85   | 0.85 |
| 40     | 0.9      | 0.9    | 0.9    | 0.9  |
| 50     | 0.95     | 0.95   | 0.95   | 0.95 |
</details>

![](images/be8cb18c897a8739adb332f95834e6fd43fcef8561966a41cdc692e70d93a2ce.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.2     | 0.2    | 0.2    | 0.2  |
| 10     | 0.7     | 0.7    | 0.7    | 0.7  |
| 20     | 0.8     | 0.8    | 0.8    | 0.8  |
| 30     | 0.8     | 0.8    | 0.8    | 0.8  |
| 40     | 0.8     | 0.8    | 0.8    | 0.8  |
| 50     | 0.8     | 0.8    | 0.8    | 0.8  |
</details>

![](images/e72b28c11b8634ba79095ea1663b98671ab6a9301e9baa936ce3586ac0153ff2.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.2     | 0.2    | 0.2    | 0.2  |
| 10     | 0.6     | 0.6    | 0.6    | 0.6  |
| 20     | 0.7     | 0.7    | 0.7    | 0.7  |
| 30     | 0.8     | 0.8    | 0.8    | 0.8  |
| 40     | 0.8     | 0.8    | 0.8    | 0.8  |
| 50     | 0.8     | 0.8    | 0.8    | 0.8  |
</details>

![](images/a0f4bcd4d438a3395e95d818920f078638ceaf900a45f96e3956827ef5d93deb.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.8     | 0.8    | 0.8    | 0.8  |
| 10     | 0.9     | 0.9    | 0.9    | 0.9  |
| 20     | 0.95    | 0.95   | 0.95   | 0.95 |
| 30     | 0.97    | 0.97   | 0.97   | 0.97 |
| 40     | 0.98    | 0.98   | 0.98   | 0.98 |
| 50     | 0.99    | 0.99   | 0.99   | 0.99 |
</details>

![](images/b8dd948f64a3cb23c6e709aa030144af9290ce112c3e3a43bb6696d869954bd2.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | Beta |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.8     | 0.8    | 0.8    | 0.8  |
| 10     | 0.95    | 0.95   | 0.95   | 0.95 |
| 20     | 0.98    | 0.98   | 0.98   | 0.98 |
| 30     | 0.99    | 0.99   | 0.99   | 0.99 |
| 40     | 0.995   | 0.995  | 0.995  | 0.995 |
| 50     | 0.998   | 0.998  | 0.998  | 0.998 |
</details>

![](images/3bab41a28bf2d61ad1246dbcf26ae7a93249e038c989d4b10c6bc96303ec4f7a.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC.PP | LC.PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.85    | 0.85   | 0.85   | 0.85 |
| 10     | 0.95    | 0.95   | 0.95   | 0.95 |
| 20     | 0.98    | 0.98   | 0.98   | 0.98 |
| 30     | 0.99    | 0.99   | 0.99   | 0.99 |
| 40     | 0.99    | 0.99   | 0.99   | 0.99 |
| 50     | 0.99    | 0.99   | 0.99   | 0.99 |
</details>

![](images/e9d150c0036b734d0f1bb983643552ea7b276234fdef456e10108cbfc9294c69.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.8     | 0.8    | 0.8    | 0.8  |
| 10     | 0.9     | 0.9    | 0.9    | 0.9  |
| 20     | 0.95    | 0.95   | 0.95   | 0.95 |
| 30     | 0.98    | 0.98   | 0.98   | 0.98 |
| 40     | 0.99    | 0.3    | 0.99   | 0.99 |
| 50     | 1.0     | 0.3    | 1.0    | 1.0  |
</details>

![](images/c627965382d393e9314c5335b545f7b6f8fbc2e4ec183c039a8f1928370238aa.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 10     | 0.2     | 0.1    | 0.3    | 0.1  |
| 20     | 0.3     | 0.15   | 0.4    | 0.15 |
| 30     | 0.35    | 0.2    | 0.45   | 0.2  |
| 40     | 0.38    | 0.22   | 0.48   | 0.22 |
| 50     | 0.4     | 0.25   | 0.5    | 0.25 |
</details>

![](images/116c060615c8c06cea7a136c4b39923b7d4c2d50e91a08454aa7117a58e6d8c6.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 10     | 0.2     | 0.2    | 0.2    | 0.2  |
| 20     | 0.4     | 0.4    | 0.4    | 0.4  |
| 30     | 0.5     | 0.5    | 0.5    | 0.5  |
| 40     | 0.6     | 0.6    | 0.6    | 0.6  |
| 50     | 0.7     | 0.7    | 0.7    | 0.7  |
</details>

![](images/3cd6147b00e44e993d280e6dfab5a2d6be312fefea0e3bcf3f53579f56151304.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 10     | 0.3     | 0.3    | 0.3    | 0.3  |
| 20     | 0.4     | 0.4    | 0.4    | 0.4  |
| 30     | 0.5     | 0.5    | 0.5    | 0.5  |
| 40     | 0.6     | 0.6    | 0.6    | 0.6  |
| 50     | 0.7     | 0.7    | 0.7    | 0.7  |
</details>

![](images/54124fa5ea18a01f3ef300c0e68688789cdd8b18ee67ae7e4d58a34e7f3d8285.jpg)

<details>
<summary>line</summary>

| Epochs | MCHC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 10     | 0.2     | 0.2    | 0.2    | 0.2  |
| 20     | 0.4     | 0.4    | 0.4    | 0.4  |
| 30     | 0.45    | 0.45   | 0.45   | 0.45 |
| 40     | 0.45    | 0.45   | 0.45   | 0.45 |
| 50     | 0.45    | 0.45   | 0.45   | 0.45 |
</details>

![](images/8eb8363bb544416caf580934eee1b9f036ae85a5c6bd570b0fb7f8c3e38813e4.jpg)

<details>
<summary>line</summary>

| Epochs | MCRC RP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 5      | 0.2     | 0.3    | 0.4    | 0.4  |
| 10     | 0.3     | 0.4    | 0.6    | 0.6  |
| 15     | 0.35    | 0.45   | 0.65   | 0.65 |
| 20     | 0.38    | 0.5    | 0.68   | 0.68 |
| 25     | 0.4     | 0.55   | 0.7    | 0.7  |
| 30     | 0.42    | 0.6    | 0.72   | 0.72 |
| 35     | 0.43    | 0.62   | 0.73   | 0.73 |
| 40     | 0.44    | 0.65   | 0.74   | 0.74 |
| 45     | 0.45    | 0.68   | 0.75   | 0.75 |
| 50     | 0.46    | 0.7    | 0.76   | 0.76 |
</details>

![](images/9f942e5d86b498360df467de3badae168b6c48c0f702fc3af2e0e0fcc60c842c.jpg)

<details>
<summary>line</summary>

| Epochs | MCHC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 10     | 0.6     | 0.6    | 0.6    | 0.6  |
| 20     | 0.8     | 0.8    | 0.8    | 0.8  |
| 30     | 0.8     | 0.8    | 0.8    | 0.8  |
| 40     | 0.8     | 0.8    | 0.8    | 0.8  |
| 50     | 0.8     | 0.8    | 0.8    | 0.8  |
</details>

![](images/ab7a2bbc3081dde0a66c4984e45b29af58d066c89b894f44ea9901b1f3063d9a.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 5      | 0.4     | 0.4    | 0.4    | 0.4  |
| 10     | 0.6     | 0.6    | 0.6    | 0.6  |
| 15     | 0.65    | 0.65   | 0.65   | 0.65 |
| 20     | 0.68    | 0.68   | 0.68   | 0.68 |
| 25     | 0.69    | 0.69   | 0.69   | 0.69 |
| 30     | 0.7     | 0.7    | 0.7    | 0.7  |
| 35     | 0.7     | 0.7    | 0.7    | 0.7  |
| 40     | 0.7     | 0.7    | 0.7    | 0.7  |
| 45     | 0.7     | 0.7    | 0.7    | 0.7  |
| 50     | 0.7     | 0.7    | 0.7    | 0.7  |
</details>

![](images/8a12cbba31367afa813efb5047edffaf41ff2967673ac95af02b4ea3baa48081.jpg)

<details>
<summary>line</summary>

| Epochs | MCHC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.0     | 0.0    | 0.0    | 0.0  |
| 5      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.65    | 0.65   | 0.65   | 0.65 |
| 15     | 0.68    | 0.68   | 0.68   | 0.68 |
| 20     | 0.7     | 0.7    | 0.7    | 0.7  |
| 25     | 0.7     | 0.7    | 0.7    | 0.7  |
| 30     | 0.7     | 0.7    | 0.7    | 0.7  |
| 35     | 0.7     | 0.7    | 0.7    | 0.7  |
| 40     | 0.7     | 0.7    | 0.7    | 0.7  |
| 45     | 0.7     | 0.7    | 0.7    | 0.7  |
| 50     | 0.7     | 0.7    | 0.7    | 0.7  |
</details>

Figure 18: Extrapolations of 7 different curves from LCBench at 10%, 20%, 40%, and 80% cutoff

![](images/1c31dfa19ae41969e1ae9f497eb209f8d96c5859c69d71251e865d18fb73c079.jpg)  
Figure 19: Extrapolations of 7 different curves from NAS-Bench-201 at 10%, 20%, 40%, and 80% cutoff

![](images/cb99164a7279c5fde477df6a4aeba16b0bcf0c64fcf2b5f940d564f178b2e048.jpg)

<details>
<summary>line</summary>

| Epochs | MCRC, PP | LC-PPN | target | data |
| ------ | -------- | ------ | ------ | ---- |
| 0      | 0.7      | 0.7    | 0.7    | 0.7  |
| 10     | 0.4      | 0.35   | 0.3    | 0.35 |
| 20     | 0.35     | 0.3    | 0.25   | 0.3  |
| 30     | 0.3      | 0.25   | 0.2    | 0.25 |
| 40     | 0.25     | 0.2    | 0.15   | 0.2  |
| 50     | 0.2      | 0.15   | 0.1    | 0.15 |
</details>

![](images/4e200024a495a1c31c014be2a8f78d027afd80e0de17627a9786775253aac14a.jpg)

<details>
<summary>line</summary>

| Epochs | MCRC, PP | LC, PPN | target | data |
| ------ | -------- | ------- | ------ | ---- |
| 0      | 0.7      | 0.7     | 0.7    | 0.7  |
| 10     | 0.3      | 0.3     | 0.3    | 0.3  |
| 20     | 0.25     | 0.25    | 0.25   | 0.25 |
| 30     | 0.2      | 0.2     | 0.2    | 0.2  |
| 40     | 0.15     | 0.15    | 0.15   | 0.15 |
| 50     | 0.1      | 0.1     | 0.1    | 0.1  |
</details>

![](images/b48f05bb7adbd60782bfd616060f708b859274bce1c77cd35811d31ac1827e1e.jpg)

<details>
<summary>line</summary>

| Epochs | MCRC PP | LC PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.7     | 0.7    | 0.7    | 0.7  |
| 10     | 0.3     | 0.3    | 0.3    | 0.3  |
| 20     | 0.2     | 0.2    | 0.2    | 0.2  |
| 30     | 0.15    | 0.15   | 0.15   | 0.15 |
| 40     | 0.1     | 0.1    | 0.1    | 0.1  |
| 50     | 0.05    | 0.05   | 0.05   | 0.05 |
</details>

![](images/ecbe28169c14eae7dc870ed818dd0d99f433627e92ec5942f268f2b03fd24a47.jpg)

<details>
<summary>line</summary>

| Epochs | MCRC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.7     | 0.7    | 0.7    | 0.7  |
| 10     | 0.3     | 0.3    | 0.3    | 0.3  |
| 20     | 0.2     | 0.2    | 0.2    | 0.2  |
| 30     | 0.15    | 0.15   | 0.15   | 0.15 |
| 40     | 0.15    | 0.15   | 0.15   | 0.15 |
| 50     | 0.15    | 0.15   | 0.15   | 0.15 |
</details>

![](images/1af828993f3275df01c4d9b0e6c363afa810b53db3a3a63e33fe6370884303a3.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC_PP | LC_PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.4    | 0.4    | 0.6  |
| 10     | 0.4     | 0.35   | 0.45   | 0.4  |
| 20     | 0.35    | 0.3    | 0.5    | 0.35 |
| 30     | 0.3     | 0.25   | 0.55   | 0.3  |
| 40     | 0.25    | 0.2    | 0.6    | 0.25 |
| 50     | 0.2     | 0.15   | 0.65   | 0.2  |
</details>

![](images/872d9633ceccf5d2e758d0915a2634d3ebcd83ef611243b16f01dfbd129fa18a.jpg)

<details>
<summary>line</summary>

| Epochs | MC3NC PP | LC-PFN | target | data |
| ------ | -------- | ------ | ------ | ---- |
| 0      | 0.6      | 0.6    | 0.6    | 0.6  |
| 10     | 0.4      | 0.4    | 0.4    | 0.4  |
| 20     | 0.4      | 0.4    | 0.4    | 0.4  |
| 30     | 0.4      | 0.4    | 0.4    | 0.4  |
| 40     | 0.4      | 0.4    | 0.4    | 0.4  |
| 50     | 0.4      | 0.4    | 0.4    | 0.4  |
</details>

![](images/36e293171ec54fba19a28a4321bbfb99e094645334ad37b71ce8360d980aa500.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC PP | LC PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.4     | 0.4    | 0.4    | 0.4  |
| 20     | 0.5     | 0.5    | 0.5    | 0.5  |
| 30     | 0.5     | 0.5    | 0.5    | 0.5  |
| 40     | 0.5     | 0.5    | 0.5    | 0.5  |
| 50     | 0.5     | 0.5    | 0.5    | 0.5  |
</details>

![](images/be112270f0cc56352ec9fef3548b811d04d918c1e6b9cd271ef35040e099c322.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.4     | 0.4    | 0.4    | 0.4  |
| 20     | 0.5     | 0.5    | 0.5    | 0.5  |
| 30     | 0.6     | 0.6    | 0.6    | 0.6  |
| 40     | 0.6     | 0.6    | 0.6    | 0.6  |
| 50     | 0.6     | 0.6    | 0.6    | 0.6  |
</details>

![](images/61dcfc6946bb2d09c24194ffc183e37d14c839501fabe6c6db8cbf1efb558a9c.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC-PP | LC-PTN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.55    | 0.58   | 0.57   | 0.58 |
| 20     | 0.52    | 0.56   | 0.55   | 0.56 |
| 30     | 0.5     | 0.54   | 0.53   | 0.54 |
| 40     | 0.48    | 0.52   | 0.51   | 0.52 |
| 50     | 0.47    | 0.51   | 0.5    | 0.51 |
</details>

![](images/7a9ffee2d2b4bb8059ceb0faadcc72fc3adb40f6e93e7f102c8149fe0606a6cd.jpg)

<details>
<summary>line</summary>

| Epochs | MC3NC_PP | LC-PFN | target | data |
|---|---|---|---|---|
| 0 | 0.65 | 0.65 | 0.65 | 0.65 |
| 10 | 0.65 | 0.65 | 0.65 | 0.65 |
| 20 | 0.65 | 0.65 | 0.65 | 0.65 |
| 30 | 0.65 | 0.65 | 0.65 | 0.65 |
| 40 | 0.65 | 0.65 | 0.65 | 0.65 |
| 50 | 0.65 | 0.65 | 0.65 | 0.65 |
</details>

![](images/5805c2f25d54a1fa46123685ad67f0da45aad50632931c82e7e81861383df603.jpg)

<details>
<summary>line</summary>

| Epochs | MCHC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.6     | 0.6    | 0.6    | 0.6  |
| 20     | 0.6     | 0.6    | 0.6    | 0.6  |
| 30     | 0.6     | 0.6    | 0.6    | 0.6  |
| 40     | 0.6     | 0.6    | 0.6    | 0.6  |
| 50     | 0.6     | 0.6    | 0.6    | 0.6  |
</details>

![](images/62a9e21a273b3349b93d28d005e3275b0c10d4719cc4f3ce857bff5707676fbd.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.6     | 0.6    | 0.6    | 0.6  |
| 20     | 0.6     | 0.6    | 0.6    | 0.6  |
| 30     | 0.6     | 0.6    | 0.6    | 0.6  |
| 40     | 0.6     | 0.6    | 0.6    | 0.6  |
| 50     | 0.6     | 0.6    | 0.6    | 0.6  |
</details>

![](images/3f87f5b8bcd4810d0096c6c148e77ca547af6f3729e2be70c0a30d2a55fdb0db.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.7     | 0.7    | 0.7    | 0.7  |
| 10     | 0.65    | 0.65   | 0.65   | 0.65 |
| 20     | 0.6     | 0.6    | 0.6    | 0.6  |
| 30     | 0.55    | 0.55   | 0.55   | 0.55 |
| 40     | 0.5     | 0.5    | 0.5    | 0.5  |
| 50     | 0.45    | 0.45   | 0.45   | 0.45 |
</details>

![](images/ce2dab94c8e580d40f2177ee7037fcd5193baef7d4437db914ab2ff7054d3059.jpg)

<details>
<summary>line</summary>

| Epochs | MC3NC-PP | LC-PFN | target | data |
| ------ | -------- | ------ | ------ | ---- |
| 0      | 0.7      | 0.7    | 0.7    | 0.7  |
| 10     | 0.65     | 0.65   | 0.65   | 0.65 |
| 20     | 0.6      | 0.6    | 0.6    | 0.6  |
| 30     | 0.55     | 0.55   | 0.55   | 0.55 |
| 40     | 0.5      | 0.5    | 0.5    | 0.5  |
| 50     | 0.45     | 0.45   | 0.45   | 0.45 |
</details>

![](images/82eb84663e549dd46e14df1293fdbec7e00f054cf79681c3109dbd2366d6e1d0.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.7     | 0.7    | 0.7    | 0.7  |
| 10     | 0.6     | 0.6    | 0.6    | 0.6  |
| 20     | 0.5     | 0.5    | 0.5    | 0.5  |
| 30     | 0.4     | 0.4    | 0.4    | 0.4  |
| 40     | 0.3     | 0.3    | 0.3    | 0.3  |
| 50     | 0.2     | 0.2    | 0.2    | 0.2  |
</details>

![](images/51f429043bf6da018a97c000cb68d9b4cca57f8fa6e0b63e50a07183b3579c32.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.7     | 0.7    | 0.7    | 0.7  |
| 10     | 0.6     | 0.6    | 0.6    | 0.6  |
| 20     | 0.4     | 0.4    | 0.4    | 0.4  |
| 30     | 0.5     | 0.5    | 0.5    | 0.5  |
| 40     | 0.5     | 0.5    | 0.5    | 0.5  |
| 50     | 0.5     | 0.5    | 0.5    | 0.5  |
</details>

![](images/2eefa78918de5e35e20c7b4388dcbddceac2e828a373b9bce8d8d26442364c24.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.7     | 0.7    | 0.7    | 0.7  |
| 10     | 0.5     | 0.45   | 0.45   | 0.45 |
| 30     | 0.4     | 0.35   | 0.35   | 0.35 |
| 50     | 0.3     | 0.3    | 0.3    | 0.3  |
</details>

![](images/a24b9d1a5987445aadac4921982c8a44c84e35e58d931bc98d6d729b338b369c.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | Beta |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.7     | 0.7    | 0.7    | 0.7  |
| 10     | 0.5     | 0.5    | 0.5    | 0.5  |
| 20     | 0.45    | 0.45   | 0.45   | 0.45 |
| 30     | 0.4     | 0.4    | 0.4    | 0.4  |
| 40     | 0.35    | 0.35   | 0.35   | 0.35 |
| 50     | 0.3     | 0.3    | 0.3    | 0.3  |
</details>

![](images/e5360fde0df33d1594905a16b87340fc92751bc651f9cf6236d5ba893af966a7.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.7     | 0.7    | 0.7    | 0.7  |
| 10     | 0.5     | 0.5    | 0.5    | 0.5  |
| 20     | 0.45    | 0.45   | 0.45   | 0.45 |
| 30     | 0.45    | 0.45   | 0.45   | 0.45 |
| 40     | 0.45    | 0.45   | 0.45   | 0.45 |
| 50     | 0.45    | 0.45   | 0.45   | 0.45 |
</details>

![](images/73df80ad0324772625cecb08002164c4ab0462a1f1617310b9a00625c8c6f663.jpg)

<details>
<summary>line</summary>

| Epochs | MCCHC APP | LC-PFN | target | data |
| ------ | --------- | ------ | ------ | ---- |
| 0      | 0.7       | 0.7    | 0.7    | 0.7  |
| 10     | 0.5       | 0.5    | 0.5    | 0.5  |
| 20     | 0.45      | 0.45   | 0.45   | 0.45 |
| 30     | 0.4       | 0.4    | 0.4    | 0.4  |
| 40     | 0.35      | 0.35   | 0.35   | 0.35 |
| 50     | 0.3       | 0.3    | 0.3    | 0.3  |
</details>

![](images/b65d51c558b198e092f89d16b82fa6d1edb83ac83cffdfd6d7808973029c1c66.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC_PP | LC_PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.4     | 0.4    | 0.4    | 0.4  |
| 20     | 0.3     | 0.3    | 0.3    | 0.3  |
| 30     | 0.25    | 0.25   | 0.25   | 0.25 |
| 40     | 0.2     | 0.2    | 0.2    | 0.2  |
| 50     | 0.15    | 0.15   | 0.15   | 0.15 |
</details>

![](images/9566d697ddc063039b94608e62547189e02cc4803111d1e1d9bfdef5ac5a824a.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.45    | 0.45   | 0.45   | 0.45 |
| 20     | 0.35    | 0.35   | 0.35   | 0.35 |
| 30     | 0.3     | 0.3    | 0.3    | 0.3  |
| 40     | 0.25    | 0.25   | 0.25   | 0.25 |
| 50     | 0.2     | 0.2    | 0.2    | 0.2  |
</details>

![](images/a931b580ad023e466e33c89ce63239e2f6e4dca31dfe5ebc8beb94dcfd91f189.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.4     | 0.4    | 0.4    | 0.4  |
| 20     | 0.35    | 0.35   | 0.35   | 0.35 |
| 30     | 0.3     | 0.3    | 0.3    | 0.3  |
| 40     | 0.25    | 0.25   | 0.25   | 0.25 |
| 50     | 0.2     | 0.2    | 0.2    | 0.2  |
</details>

![](images/125d9da5e0a188de977851ca000c551ff0a6ca79e7948b7a972a11bead58ae00.jpg)

<details>
<summary>line</summary>

| Epochs | MCRC_PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.4     | 0.4    | 0.4    | 0.4  |
| 20     | 0.35    | 0.35   | 0.35   | 0.35 |
| 30     | 0.3     | 0.3    | 0.3    | 0.3  |
| 40     | 0.25    | 0.25   | 0.25   | 0.25 |
| 50     | 0.2     | 0.2    | 0.2    | 0.2  |
</details>

![](images/68f0fe382db01ac2b30bfcb68aa3a723635a2fc481d400b35e7f89259863e5a9.jpg)

<details>
<summary>line</summary>

| Epochs | MCRC, PP | LC-PFN | target | data |
| ------ | -------- | ------ | ------ | ---- |
| 0      | 0.4      | 0.4    | 0.4    | 0.6  |
| 10     | 0.3      | 0.3    | 0.3    | 0.3  |
| 20     | 0.25     | 0.25   | 0.25   | 0.25 |
| 30     | 0.2      | 0.2    | 0.2    | 0.2  |
| 40     | 0.15     | 0.15   | 0.15   | 0.15 |
| 50     | 0.1      | 0.1    | 0.1    | 0.1  |
</details>

![](images/7362df4fd1d0b8adf4a1a678c6049d8b0d70e491e1f9f98acf3abfbdf248778d.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.3     | 0.3    | 0.3    | 0.3  |
| 20     | 0.2     | 0.2    | 0.2    | 0.2  |
| 30     | 0.15    | 0.15   | 0.15   | 0.15 |
| 40     | 0.1     | 0.1    | 0.1    | 0.1  |
| 50     | 0.08    | 0.08   | 0.08   | 0.08 |
</details>

![](images/98496ce5f606eb81d8594272af728c4223caf52578644d124e17a826804a1dc7.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 0.6     | 0.6    | 0.6    | 0.6  |
| 10     | 0.2     | 0.2    | 0.2    | 0.2  |
| 20     | 0.15    | 0.15   | 0.15   | 0.15 |
| 30     | 0.12    | 0.12   | 0.12   | 0.12 |
| 40     | 0.1     | 0.1    | 0.1    | 0.1  |
| 50     | 0.08    | 0.08   | 0.08   | 0.08 |
</details>

![](images/9aa7f8e14d728ff703523e41f3a33d9562ffcf2984fd448332b8cfa0f1149d9b.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC_PP | LC-PFN | target data | data |
| ------ | ------- | ------ | ----------- | ---- |
| 0      | 0.6     | 0.6    | 0.6         | 0.6  |
| 10     | 0.3     | 0.3    | 0.3         | 0.3  |
| 20     | 0.2     | 0.2    | 0.2         | 0.2  |
| 30     | 0.15    | 0.15   | 0.15        | 0.15 |
| 40     | 0.1     | 0.1    | 0.1         | 0.1  |
| 50     | 0.08    | 0.08   | 0.08        | 0.08 |
</details>

Figure 20: Extrapolations of 7 different curves from Taskset at 10%, 20%, 40%, and 80% cutoff

![](images/e16d01682fc3f709b950f5a7bd3cddd9cf062e4ed7d102f14be244cfd3520729.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 4.0     | 4.0    | 4.0    | 4.0  |
| 5      | 3.8     | 3.8    | 3.8    | 3.8  |
| 10     | 3.7     | 3.7    | 3.7    | 3.7  |
| 15     | 3.65    | 3.65   | 3.65   | 3.65 |
| 20     | 3.6     | 3.6    | 3.6    | 3.6  |
| 25     | 3.55    | 3.55   | 3.55   | 3.55 |
| 30     | 3.5     | 3.5    | 3.5    | 3.5  |
| 35     | 3.45    | 3.45   | 3.45   | 3.45 |
</details>

![](images/1bf5cdfe21a24e3a661139502792ea9c21317a4dc12108291aecf88ab0f88109.jpg)

<details>
<summary>line</summary>

| Epochs | Data | MCHC, PP | LC-PPN | target |
| ------ | ---- | -------- | ------ | ------ |
| 0      | 4.3  | 4.3      | 4.3    | 4.3    |
| 5      | 3.8  | 3.8      | 3.8    | 3.8    |
| 10     | 3.7  | 3.7      | 3.7    | 3.7    |
| 15     | 3.65 | 3.65     | 3.65   | 3.65   |
| 20     | 3.6  | 3.6      | 3.6    | 3.6    |
| 25     | 3.55 | 3.55     | 3.55   | 3.55   |
| 30     | 3.5  | 3.5      | 3.5    | 3.5    |
| 35     | 3.45 | 3.45     | 3.45   | 3.45   |
</details>

![](images/e1d7bb0b2f7cf73f27c604f1196343c3169de6661698106a1b9d72055674de57.jpg)

<details>
<summary>line</summary>

| Epochs | MCFC_PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 4.3     | 4.3    | 4.3    | 4.3  |
| 5      | 3.9     | 3.9    | 3.9    | 3.9  |
| 10     | 3.7     | 3.7    | 3.7    | 3.7  |
| 15     | 3.6     | 3.6    | 3.6    | 3.6  |
| 20     | 3.55    | 3.55   | 3.55   | 3.55 |
| 25     | 3.52    | 3.52   | 3.52   | 3.52 |
| 30     | 3.51    | 3.51   | 3.51   | 3.51 |
| 35     | 3.5     | 3.5    | 3.5    | 3.5  |
</details>

![](images/28be4d1822311f4cb8fcb0256243a28c336fb0da6a29bf7d0293b7ae70fbb81e.jpg)

<details>
<summary>line</summary>

| Epochs | MCCH-PP | LC-FFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 4.3     | 4.3    | 4.3    | 4.3  |
| 5      | 3.9     | 3.9    | 3.9    | 3.9  |
| 10     | 3.7     | 3.7    | 3.7    | 3.7  |
| 15     | 3.6     | 3.6    | 3.6    | 3.6  |
| 20     | 3.6     | 3.6    | 3.6    | 3.6  |
| 25     | 3.6     | 3.6    | 3.6    | 3.6  |
| 30     | 3.6     | 3.6    | 3.6    | 3.6  |
| 35     | 3.6     | 3.6    | 3.6    | 3.6  |
</details>

![](images/e825c2f616cbc2ba0a1f9e5c4628a7d03bdc95cf12fdf3486a8e4e7e7fb109c9.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PTN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 4.6     | 4.4    | 4.4    | 4.4  |
| 5      | 4.5     | 4.2    | 4.1    | 4.1  |
| 10     | 4.4     | 4.1    | 3.9    | 3.9  |
| 15     | 4.3     | 4.0    | 3.8    | 3.8  |
| 20     | 4.2     | 3.9    | 3.7    | 3.7  |
| 25     | 4.1     | 3.8    | 3.6    | 3.6  |
| 30     | 4.0     | 3.7    | 3.5    | 3.5  |
| 35     | 3.9     | 3.6    | 3.4    | 3.4  |
</details>

![](images/54960f3dbfb2c1002dc78be2931521255a1fa7975865d881338683b564255489.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC RP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 4.6     | 4.6    | 4.6    | 4.6  |
| 5      | 3.9     | 3.9    | 3.9    | 3.9  |
| 10     | 3.8     | 3.8    | 3.8    | 3.8  |
| 15     | 3.7     | 3.7    | 3.7    | 3.7  |
| 20     | 3.65    | 3.65   | 3.65   | 3.65 |
| 25     | 3.6     | 3.6    | 3.6    | 3.6  |
| 30     | 3.55    | 3.55   | 3.55   | 3.55 |
| 35     | 3.5     | 3.5    | 3.5    | 3.5  |
</details>

![](images/2c58d565c8759f18cd362d13542d9b8f2daecf10a98790d7d3f2743e2e42701c.jpg)

<details>
<summary>line</summary>

| Epochs | MCFC_PP | LC-PPs | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 4.6     | 4.6    | 4.6    | 4.6  |
| 5      | 4.2     | 4.2    | 4.2    | 4.2  |
| 10     | 3.8     | 3.8    | 3.8    | 3.8  |
| 15     | 3.6     | 3.6    | 3.6    | 3.6  |
| 20     | 3.5     | 3.5    | 3.5    | 3.5  |
| 25     | 3.4     | 3.4    | 3.4    | 3.4  |
| 30     | 3.3     | 3.3    | 3.3    | 3.3  |
| 35     | 3.2     | 3.2    | 3.2    | 3.2  |
</details>

![](images/2f27558511624cc876bd6a7fb2f24c7b8c32e4a36c5c94d59ba01c043e71f7c4.jpg)

<details>
<summary>line</summary>

| Epochs | MCFC-PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 4.6     | 4.6    | 4.6    | 4.6  |
| 5      | 4.2     | 4.2    | 4.2    | 4.2  |
| 10     | 3.9     | 3.9    | 3.9    | 3.9  |
| 15     | 3.7     | 3.7    | 3.7    | 3.7  |
| 20     | 3.6     | 3.6    | 3.6    | 3.6  |
| 25     | 3.6     | 3.6    | 3.6    | 3.6  |
| 30     | 3.6     | 3.6    | 3.6    | 3.6  |
| 35     | 3.6     | 3.6    | 3.6    | 3.6  |
</details>

![](images/51a2dcd533e5ea4b86b400b68576256dc4d1201ea0d85f288e789da2dd9268a6.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PPI | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 7.0     | 7.0    | 7.0    | 7.0  |
| 200    | 5.8     | 5.7    | 5.9    | 5.8  |
| 400    | 5.3     | 5.2    | 5.4    | 5.3  |
| 600    | 5.0     | 4.9    | 5.1    | 5.0  |
| 800    | 4.8     | 4.7    | 4.9    | 4.8  |
| 1000   | 4.6     | 4.5    | 4.7    | 4.6  |
| 1200   | 4.4     | 4.3    | 4.5    | 4.4  |
| 1400   | 4.2     | 4.1    | 4.3    | 4.2  |
</details>

![](images/0d5f53e55778ee2ad919f739b6331a6aef2d643d48ace834aece8fc465534c02.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC_PP | LC_WN | target | data |
| ------ | ------- | ----- | ------ | ---- |
| 0      | 7.0     | 7.0   | 7.0    | 7.0  |
| 200    | 5.5     | 5.3   | 5.4    | 5.6  |
| 400    | 5.2     | 5.0   | 5.1    | 5.3  |
| 600    | 5.0     | 4.8   | 4.9    | 5.1  |
| 800    | 4.8     | 4.6   | 4.7    | 4.9  |
| 1000   | 4.6     | 4.4   | 4.5    | 4.7  |
| 1200   | 4.4     | 4.2   | 4.3    | 4.5  |
| 1400   | 4.2     | 4.0   | 4.1    | 4.3  |
</details>

![](images/dee61d6e407dfecb634ad7bb191257db31da7220f4680bbf557f759fed7efdee.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC-PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 7.0     | 7.0    | 7.0    | 7.0  |
| 200    | 5.5     | 5.5    | 5.5    | 5.5  |
| 400    | 4.8     | 4.8    | 4.8    | 4.8  |
| 600    | 4.5     | 4.5    | 4.5    | 4.5  |
| 800    | 4.3     | 4.3    | 4.3    | 4.3  |
| 1000   | 4.2     | 4.2    | 4.2    | 4.2  |
| 1200   | 4.1     | 4.1    | 4.1    | 4.1  |
| 1400   | 4.0     | 4.0    | 4.0    | 4.0  |
</details>

![](images/8c1ed84b59409160eb079fdbf549581397ccb9383300ddd5816c5a7566c9262f.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 6.5     | 6.5    | 6.5    | 6.5  |
| 200    | 5.0     | 5.0    | 5.0    | 5.0  |
| 400    | 4.5     | 4.5    | 4.5    | 4.5  |
| 600    | 4.3     | 4.3    | 4.3    | 4.3  |
| 800    | 4.2     | 4.2    | 4.2    | 4.2  |
| 1000   | 4.1     | 4.1    | 4.1    | 4.1  |
| 1200   | 4.0     | 4.0    | 4.0    | 4.0  |
| 1400   | 4.0     | 4.0    | 4.0    | 4.0  |
</details>

![](images/22dc742b5ccfe77b899d7a85b236a8f23e72f43de4d2be021e926c3ee0fce279.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 6.0     | 6.0    | 6.0    | 6.0  |
| 200    | 3.5     | 3.5    | 3.5    | 3.5  |
| 400    | 3.0     | 3.0    | 3.0    | 3.0  |
| 600    | 2.8     | 2.8    | 2.8    | 2.8  |
| 800    | 2.6     | 2.6    | 2.6    | 2.6  |
| 1000   | 2.4     | 2.4    | 2.4    | 2.4  |
| 1200   | 2.2     | 2.2    | 2.2    | 2.2  |
| 1400   | 2.0     | 2.0    | 2.0    | 2.0  |
</details>

![](images/80dc19541fd11d1fd98f1ef9377cf0e76b7570a81b59ac19142fb37181dd72d5.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC, PP | LC WN | target | data |
| ------ | -------- | ----- | ------ | ---- |
| 0      | 6.0      | 6.0   | 6.0    | 6.0  |
| 200    | 4.0      | 4.0   | 4.0    | 4.0  |
| 400    | 3.8      | 3.8   | 3.8    | 3.8  |
| 600    | 3.7      | 3.7   | 3.7    | 3.7  |
| 800    | 3.6      | 3.6   | 3.6    | 3.6  |
| 1000   | 3.5      | 3.5   | 3.5    | 3.5  |
| 1200   | 3.4      | 3.4   | 3.4    | 3.4  |
| 1400   | 3.3      | 3.3   | 3.3    | 3.3  |
</details>

![](images/92942f724b8217f3db11193c54951b6adc5930c24f638d714545cb934bb0dde3.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 6.0     | 6.0    | 6.0    | 6.0  |
| 200    | 4.0     | 4.0    | 4.0    | 4.0  |
| 400    | 4.0     | 4.0    | 4.0    | 4.0  |
| 600    | 4.0     | 4.0    | 4.0    | 4.0  |
| 800    | 4.0     | 4.0    | 4.0    | 4.0  |
| 1000   | 4.0     | 4.0    | 4.0    | 4.0  |
| 1200   | 4.0     | 4.0    | 4.0    | 4.0  |
| 1400   | 4.0     | 4.0    | 4.0    | 4.0  |
</details>

![](images/2059bd7c618c63f714d6d4019d0d261fd24a58ef263e50947edcf35296ca81e8.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC.PP | LC.PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 6.0     | 6.0    | 6.0    | 6.0  |
| 200    | 4.0     | 4.0    | 4.0    | 4.0  |
| 400    | 4.0     | 4.0    | 4.0    | 4.0  |
| 600    | 4.0     | 4.0    | 4.0    | 4.0  |
| 800    | 4.0     | 4.0    | 4.0    | 4.0  |
| 1000   | 4.0     | 4.0    | 4.0    | 4.0  |
| 1200   | 4.0     | 4.0    | 4.0    | 4.0  |
| 1400   | 4.0     | 4.0    | 4.0    | 4.0  |
</details>

![](images/a512d8ee3853b2b62f3b7d83d1ed04a400d5c742214dbd91811038af8dcd75fb.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PF | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 5.0     | 5.0    | 5.0    | 5.0  |
| 200    | 2.5     | 2.5    | 2.5    | 2.5  |
| 400    | 2.2     | 2.2    | 2.2    | 2.2  |
| 600    | 2.1     | 2.1    | 2.1    | 2.1  |
| 800    | 2.0     | 2.0    | 2.0    | 2.0  |
| 1000   | 2.0     | 2.0    | 2.0    | 2.0  |
| 1200   | 2.0     | 2.0    | 2.0    | 2.0  |
| 1400   | 2.0     | 2.0    | 2.0    | 2.0  |
</details>

![](images/b863350537e98d8f15056446539dce9e06f8b42d2a0f2ce418e291449a886716.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC_PP | LC_PN | target | data |
| ------ | ------- | ----- | ------ | ---- |
| 0      | 5.0     | 5.0   | 5.0    | 5.0  |
| 200    | 2.5     | 2.5   | 2.5    | 2.5  |
| 400    | 2.2     | 2.2   | 2.2    | 2.2  |
| 600    | 2.1     | 2.1   | 2.1    | 2.1  |
| 800    | 2.05    | 2.05  | 2.05   | 2.05 |
| 1000   | 2.02    | 2.02  | 2.02   | 2.02 |
| 1200   | 2.01    | 2.01  | 2.01   | 2.01 |
| 1400   | 2.005   | 2.005 | 2.005  | 2.005 |
</details>

![](images/cf442bd1492586859e4dbfec08681fa6ddfbb3455d34f3089c313b2dc2dad2d9.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PPN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0      | 5.5     | 5.5    | 5.5    | 5.5  |
| 200    | 3.0     | 3.0    | 3.0    | 3.0  |
| 400    | 2.2     | 2.2    | 2.2    | 2.2  |
| 600    | 2.1     | 2.1    | 2.1    | 2.1  |
| 800    | 2.05    | 2.05   | 2.05   | 2.05 |
| 1000   | 2.02    | 2.02   | 2.02   | 2.02 |
| 1200   | 2.01    | 2.01   | 2.01   | 2.01 |
| 1400   | 2.005   | 2.005  | 2.005  | 2.005 |
</details>

![](images/d0589066fc92fba0d2a2d88af7c3306430582cffef58505d63689498646f55a2.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC.PP | LC.PPN | target | delta |
| ------ | ------- | ------ | ------ | ----- |
| 0      | 5.0     | 5.0    | 5.0    | 5.0   |
| 200    | 2.5     | 2.5    | 2.5    | 2.5   |
| 400    | 2.0     | 2.0    | 2.0    | 2.0   |
| 600    | 1.8     | 1.8    | 1.8    | 1.8   |
| 800    | 1.6     | 1.6    | 1.6    | 1.6   |
| 1000   | 1.5     | 1.5    | 1.5    | 1.5   |
| 1200   | 1.4     | 1.4    | 1.4    | 1.4   |
| 1400   | 1.3     | 1.3    | 1.3    | 1.3   |
</details>

![](images/7d61f084945f61c448348dbd6e046c9ed1944ed9ed1e38fde13cc03c40ae7d1f.jpg)

<details>
<summary>line</summary>

| Epochs | HCMC-PP | LC-PFN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0.0    | 2.85    | 2.85   | 2.85   | 2.85 |
| 2.5    | 2.80    | 2.78   | 2.75   | 2.75 |
| 5.0    | 2.75    | 2.73   | 2.70   | 2.70 |
| 7.5    | 2.70    | 2.68   | 2.65   | 2.65 |
| 10.0   | 2.65    | 2.63   | 2.60   | 2.60 |
| 12.5   | 2.60    | 2.58   | 2.55   | 2.55 |
| 15.0   | 2.55    | 2.53   | 2.50   | 2.50 |
| 17.5   | 2.50    | 2.48   | 2.45   | 2.45 |
| 20.0   | 2.45    | 2.43   | 2.40   | 2.40 |
</details>

![](images/632aabe26fa4737cdaf5430cd46e35e5a40550ae737849d26544cbca1e6d2ba5.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC_PP | LC-PYN | target | data |
| ------ | ------- | ------ | ------ | ---- |
| 0.0    | 2.85    | 2.80   | 2.80   | 2.80 |
| 2.5    | 2.80    | 2.78   | 2.78   | 2.78 |
| 5.0    | 2.78    | 2.76   | 2.76   | 2.76 |
| 7.5    | 2.76    | 2.74   | 2.74   | 2.74 |
| 10.0   | 2.74    | 2.72   | 2.72   | 2.72 |
| 12.5   | 2.72    | 2.70   | 2.70   | 2.70 |
| 15.0   | 2.70    | 2.68   | 2.68   | 2.68 |
| 17.5   | 2.68    | 2.66   | 2.66   | 2.66 |
| 20.0   | 2.66    | 2.64   | 2.64   | 2.64 |
</details>

![](images/8fbdf0af36706db6152ab063f3508e72cda8ebe4d83265d434c01cad96d97193.jpg)

<details>
<summary>line</summary>

| Epoch | MCFC-PP | LC-PPN | target | data |
|-------|---------|--------|--------|------|
| 0.0   | 2.85    | 2.85   | 2.85   | 2.85 |
| 2.5   | 2.80    | 2.80   | 2.80   | 2.80 |
| 5.0   | 2.78    | 2.78   | 2.78   | 2.78 |
| 7.5   | 2.76    | 2.76   | 2.76   | 2.76 |
| 10.0  | 2.74    | 2.74   | 2.74   | 2.74 |
| 12.5  | 2.73    | 2.73   | 2.73   | 2.73 |
| 15.0  | 2.72    | 2.72   | 2.72   | 2.72 |
| 17.5  | 2.71    | 2.71   | 2.71   | 2.71 |
| 20.0  | 2.70    | 2.70   | 2.70   | 2.70 |
</details>

![](images/9537135c04e64fc967d9eb742a61d107b4214ba02944a2cb47ecb25598e71103.jpg)

<details>
<summary>line</summary>

| x    | MCFC_PP | LC-PPN | target | data  |
| ---- | ------- | ------ | ------ | ----- |
| 0.0  | 2.85    | 2.85   | 2.85   | 2.85  |
| 2.5  | 2.80    | 2.80   | 2.80   | 2.80  |
| 5.0  | 2.75    | 2.75   | 2.75   | 2.75  |
| 7.5  | 2.70    | 2.70   | 2.70   | 2.70  |
| 10.0 | 2.68    | 2.68   | 2.68   | 2.68  |
| 12.5 | 2.67    | 2.67   | 2.67   | 2.67  |
| 15.0 | 2.66    | 2.66   | 2.66   | 2.66  |
| 17.5 | 2.65    | 2.65   | 2.65   | 2.65  |
| 20.0 | 2.64    | 2.64   | 2.64   | 2.64  |
</details>

![](images/56d4b2ea5a093a1a9d95ab705600ceaf346cd2feae6dd3980092747889c78f26.jpg)

<details>
<summary>line</summary>

| Epochs | MCHC-PP | LC-PFN | target data |
| ------ | ------- | ------ | ----------- |
| 0.0    | 2.800   | 2.800  | 2.800       |
| 2.5    | 2.750   | 2.750  | 2.750       |
| 5.0    | 2.725   | 2.725  | 2.725       |
| 7.5    | 2.700   | 2.700  | 2.700       |
| 10.0   | 2.675   | 2.675  | 2.675       |
| 12.5   | 2.650   | 2.650  | 2.650       |
| 15.0   | 2.625   | 2.625  | 2.625       |
| 17.5   | 2.600   | 2.600  | 2.600       |
| 20.0   | 2.575   | 2.575  | 2.575       |
</details>

![](images/6297992943855e74075b8a7fb678f5f1f5ebee07429c46e1fb95baf1d70faf70.jpg)

<details>
<summary>line</summary>

| Epochs | MCMC-PP | LC-PFN | target data |
| ------ | ------- | ------ | ----------- |
| 0.0    | 2.800   | 2.800  | 2.800       |
| 2.5    | 2.750   | 2.750  | 2.750       |
| 5.0    | 2.725   | 2.725  | 2.725       |
| 7.5    | 2.700   | 2.700  | 2.700       |
| 10.0   | 2.675   | 2.675  | 2.675       |
| 12.5   | 2.650   | 2.650  | 2.650       |
| 15.0   | 2.625   | 2.625  | 2.625       |
| 17.5   | 2.600   | 2.600  | 2.600       |
| 20.0   | 2.575   | 2.575  | 2.575       |
</details>

![](images/986fc5460dcda26faab55d3c799f77c1753fdee9e2baa738622be0ca266b5deb.jpg)

<details>
<summary>line</summary>

| Epochs | MCNC-PP | LC-FFN | target data |
| ------ | ------- | ------ | ----------- |
| 0.0    | 2.800   | 2.800  | 2.800       |
| 2.5    | 2.775   | 2.775  | 2.775       |
| 5.0    | 2.775   | 2.775  | 2.775       |
| 7.5    | 2.775   | 2.775  | 2.775       |
| 10.0   | 2.775   | 2.775  | 2.775       |
| 12.5   | 2.775   | 2.775  | 2.775       |
| 15.0   | 2.775   | 2.775  | 2.775       |
| 17.5   | 2.775   | 2.775  | 2.775       |
| 20.0   | 2.775   | 2.775  | 2.775       |
</details>

![](images/c886524d91929dafda9c14ad16fa0fde7618e88e3abbca71de29523f069a6e29.jpg)

<details>
<summary>line</summary>

| Epochs | NCCHC-PP | LC-PPN | target |
| ------ | -------- | ------ | ------ |
| 0.0    | 2.800    | 2.800  | 2.800  |
| 2.5    | 2.775    | 2.775  | 2.775  |
| 5.0    | 2.770    | 2.770  | 2.770  |
| 7.5    | 2.765    | 2.765  | 2.765  |
| 10.0   | 2.760    | 2.760  | 2.760  |
| 12.5   | 2.755    | 2.755  | 2.755  |
| 15.0   | 2.750    | 2.750  | 2.750  |
| 17.5   | 2.745    | 2.745  | 2.745  |
| 20.0   | 2.740    | 2.740  | 2.740  |
</details>

Figure 21: Extrapolations of 7 different curves from PD1 at 10%, 20%, 40%, and 80% cutoff