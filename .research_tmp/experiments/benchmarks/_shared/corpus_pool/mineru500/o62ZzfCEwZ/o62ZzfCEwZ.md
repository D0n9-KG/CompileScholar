# Investigating the Overlooked Hessian Structure: From CNNs to LLMs

Qian-Yuan Tang $^{*1}$ Yufei Gu $^{*2}$ Yunfeng Cai $^{3}$ Mingming Sun $^{4}$ Ping Li $^{5}$ Xun Zhou $^{6}$ Zeke Xie $^{\dagger2}$

# Abstract

It is well-known that the Hessian of deep loss landscape matters to optimization and generalization of deep learning. Previous studies reported a rough Hessian structure in deep learning, which consists of two components, a small number of large eigenvalues and a large number of nearly-zero eigenvalues. To the best of our knowledge, we are the first to report that a simple but overlooked power-law Hessian structure exists in well-trained deep neural networks, including Convolutional Neural Networks (CNNs) and Large Language Models (LLMs). Moreover, we provide a maximum-entropy theoretical interpretation for the power-law Hessian structure and theoretically demonstrate the existence of a robust and low-dimensional subspace of deep neural networks. Our extensive experiments using the proposed power-law spectral method demonstrate that the power-law Hessian spectra critically relate to multiple important behaviors of deep learning, including optimization, generalization, and overparameterization. Notably, we discover that the power-law Hessian structure of a given LLM can often predict generalization during training in some occasions, while conventional sharpness-based generalization measures which often work well on CNNs largely fail as an effective generalization predictor of LLMs.

# 1. Introduction

It is well-known that the Hessian matters to optimization, generalization, and even robustness of deep learning (Li et al., 2020; Ghorbani et al., 2019; Jacot et al., 2019;

$^{1}$ Department of Physics, Hong Kong Baptist University $^{2}$ xLeaF Lab, The Hong Kong University of Science and Technology (Guangzhou) $^{3}$ BIMSA $^{4}$ AGI Lab, BIMSA $^{5}$ Rutgers University $^{6}$ Seed-Foundation-Model Team, ByteDance. \*: Equal Contributions. Correspondence to: † Zeke Xie <zekexie@hkustgz.edu.cn>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

Yao et al., 2018; Dauphin et al., 2014; Byrd et al., 2011). Deep learning usually finds flat minima that generalize well (Hochreiter & Schmidhuber, 1995; 1997). The Hessian is one of the most important measures of the minima flatness and directly relates to generalization in deep learning (Hoffer et al., 2017; Neyshabur et al., 2017; Dinh et al., 2017; Wu et al., 2017). Jiang et al. (2019) reported that minima-flatness-based generalization bound is still the most reliable metric in extensive experiments. Wu et al. (2017) reported that the low-complexity solutions that generalize well have a small norm of Hessian matrix with respect to model parameters. Yao et al. (2018) reported that the spectrum of the Hessian closely connects to large-batch training and adversarial robustness.

A number of works empirically studied the Hessian structure in Deep Neural Networks (DNNs). Some papers (Sagun et al., 2016; 2017; Wu et al., 2017) empirically reported a two-component structure that, in the context of deep learning, most Hessian eigenvalues are nearly zero, while a small number of eigenvalues are large. Sankar et al. (2021) revealed that the layer-wise Hessian spectrum is similar to the entire Hessian spectrum. Zhang et al. (2024b) demonstrated that SGD performs worse than Adam for Transformers when Hessian spectra exhibit blockwise heterogeneity. Ormaniec et al. (2024) theoretically studied one-layer Transformers' Hessian in matrix derivatives while comparing them to classical networks in deep learning.

However, quantitative or statistical analysis of the Hessian structure is still largely under-explored for modern neural networks. Does an elegant statistical structure hide behind the Hessian spectrum? Does such structure matter to CNNs and LLMs? Our work provides a novel approach to understanding and analyzing the Hessian structure of deep loss landscape from a statistical perspective. This work mainly made three contributions.

First, to the best of our knowledge, we are the first to empirically discover and statistically test the power-law Hessian structure of deep loss landscape which has been overlooked by previous studies. Such novel power-law structure widely exist in DNNs, including CNNs and LLMs.

Second, we propose a framework of power-law spectral analysis for quantitatively analyzing the Hessian structure in deep learning. We not only reveal how the power-law

![](images/bb4f295de58cea5d93d7bf64d4b787161d8190d615f8ed6c29f008aee5b617f0.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | SGD       | Random    |
| --------------- | --------- | --------- |
| 10^0            | ~10^0     | ~10^0     |
| 10^1            | ~10^-1    | ~10^-1    |
| 10^2            | ~10^-2    | ~10^-2    |
| 10^3            | ~10^-3    | ~10^-3    |
</details>

(a) MNIST

![](images/34416ade949befd9c27fdd1cfaebdc0144e81a6635fc01f689ae68a226170ad1.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | SGD     | Random  |
| --------------- | ------- | ------- |
| 10^0            | 10^1    | 10^0    |
| 10^1            | 10^0    | 10^-1   |
| 10^2            | 10^-1   | 10^-2   |
| 10^3            | 10^-2   | 10^-3   |
</details>

(b) Fashion-MNIST

![](images/b9954cddd81612384121f0492795c7a46542b1bec0ed155c1fc301daea798524.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | SGD       | Random    |
| --------------- | --------- | --------- |
| 10^0            | 10^2      | 10^-1     |
| 10^1            | 10^1      | 10^-1     |
| 10^2            | 10^0      | 10^-2     |
| 10^3            | 10^-1     | 10^-3     |
</details>

(c) CIFAR-10

![](images/7398c7106ea04f2365749d74b92bfabb28dc8972cf272b4a09417bcd202c1c06.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | SGD       | Random    |
| --------------- | --------- | --------- |
| 10^0            | 10^2      | 10^-1     |
| 10^1            | 10^1      | 10^-1     |
| 10^2            | 10^0      | 10^-2     |
| 10^3            | 10^-1     | 10^-3     |
</details>

(d) CIFAR-100   
Figure 1. The power-law structure of the Hessian spectrum in deep learning. Model: LeNet. We may clearly observe that the power-law spectra generally hold for well-trained deep networks on various natural or artificial datasets, while do not hold for random neural networks. We also report that a small number of outlier eigenvalues ( $\sim$ 10) slightly deviate from the fitted straight line.

spectra explain the theoretical origin of striking findings but also empirically demonstrate multiple novel insights on optimization, generalization, and scaling.

Third, we report that the power-law Hessian spectral analysis can sometimes predict generalization of LLMs during training, particularly when conventional sharpness-based generalization measures that often work well on CNNs become nearly useless as a generalization predictor of LLMs. This suggests that generalization measures for LLMs remain to be deeply explored from a loss landscape perspective.

# 2. The Overlooked Hessian Structure

In this section, we demonstrate that the Hessian spectra of well-trained deep neural networks have a simple power-law structure overlooked by previous studies and how to theoretically derive the power-law structure.

Notations. We denote the training dataset as $\{(x,y)\}=\{(x_{j},y_{j})\}_{j=1}^{N}$ drawn from the data distribution S, the n model parameters as $\theta$ and the loss function over one data sample $\{(x_{j},y_{j})\}$ as $l(\theta,(x_{j},y_{j}))$ . For simplicity, we further denote the training loss as $L(\theta)=\frac{1}{N}\sum_{j=1}^{N}l(\theta,(x_{j},y_{j}))$ and denote its Hessian as H. We write the descending ordered eigenvalues of the Hessian H as $\{\lambda_{1},\lambda_{2},\ldots,\lambda_{n}\}$ and denote the spectral density as $p(\lambda)$ .

# 2.1. Visualizing the Power-Law Structure

Hessian has been studied as a measure of minima flatness Dinh et al. (2017); Xie et al. (2021b) and loss curvature Achille & Soatto (2019), while these works failed to reveal its elegant statistical structure. To better understand the distribution of the Hessian spectrum, we first visualize the Hessian spectrum of a well-trained neural network and a randomly initialized neural network by using the Lanczos algorithm (Meurant & Strakoš, 2006; Yao et al., 2020) to estimate the eigenvalues and spectral densities. In Figure 1, we display the top 6000 eigenvalues and their corresponding rank order. Both axes are log-scale. And we surprisingly discover an approximately straight line fits the Hessian spectrum of the well-trained neural network surprisingly well, except that a small number of outliers ( $\sim$ 10) slightly deviate from the fitted straight line. To the best of our knowledge, these fitted power-law Hessian spectra were not empirically discovered or theoretically discussed by previous papers for neural networks in deep learning.

Table 1. The Kolmogorov-Smirnov statistics of the Hessian spectra of LeNets on various datasets. The estimated power exponent $\hat{\beta}$ and slope magnitude $\hat{s}$ are also displayed. 

<table><tr><td>Dataset</td><td>Model</td><td>Training</td><td> $d_{\text{ks}}$ </td><td> $d_{\text{c}}$ </td><td>Power-Law</td></tr><tr><td>MNIST</td><td>LeNet</td><td>Random</td><td>0.0796</td><td>0.0430</td><td>No</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>0.00900</td><td>0.0430</td><td>Yes</td></tr><tr><td>Fashion-MNIST</td><td>LeNet</td><td>Random</td><td>0.0971</td><td>0.0430</td><td>No</td></tr><tr><td>Fashion-MNIST</td><td>LeNet</td><td>SGD</td><td>0.0132</td><td>0.0430</td><td>Yes</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>Random</td><td>0.0663</td><td>0.0430</td><td>No</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>SGD</td><td>0.0279</td><td>0.0430</td><td>Yes</td></tr><tr><td>CIFAR-100</td><td>LeNet</td><td>Random</td><td>0.0944</td><td>0.0430</td><td>No</td></tr><tr><td>CIFAR-100</td><td>LeNet</td><td>SGD</td><td>0.0315</td><td>0.0430</td><td>Yes</td></tr></table>

The well-fitted straight line means that the observed distribution of the Hessian eigenvalues of trained neural networks approximately obeys a power-law distribution,

$$
p (\lambda) = Z _ {c} ^ {- 1} \lambda^ {- \beta}, \tag {1}
$$

where $Z_{c}$ is the normalization factor. The observed eigenvalues can be considered as n samples from the power-law distribution $p(\lambda)$ . We may also use a corresponding finite-sample power law for describing the observed law as

$$
f _ {k} = \frac {\lambda_ {k}}{\mathrm{Tr} (H)} = Z _ {d} ^ {- 1} k ^ {- \frac {1}{\beta - 1}}, \tag {2}
$$

where $f$ is the trace-normalized eigenvalue, $k$ is the rank order, the trace $\mathrm{Tr}(H) = \sum_{k=1}^{n} \lambda_k$ , and $Z_d = \sum_{k=1}^{n} k^{-\frac{1}{\beta-1}}$ is the normalization factor for the finite-sample power law. Note that the finite-sample power law is also called Zipf's law. This can also be approximately written as

$$
\lambda_ {k} = \lambda_ {1} k ^ {- s}, \tag {3}
$$

if we let $s = \frac{1}{\beta - 1}$ denote the power exponent of Zipf's law.

# 2.2. A Maximum-Entropy Interpretation

In this subsection, we show that the maximum entropy principle widely used in statistical physics may informally explain the power-law Hessian structure in deep learning.

The maximum entropy principle (Guiasu & Shenitzer, 1985), also named the maximum entropy prior, states that the probability distribution which best represents the current state of knowledge about a system at equilibrium is the one with the highest entropy. This principle indicates that if we have no prior knowledge for suspecting one state over any other, then all states can be considered equally likely for a system at equilibrium. The logarithmic space volume is often regarded as a kind of entropy in statistical physics (Visser, 2013). Note that flat minima have larger space volume reflected by $\det(H^{-1})$ . It means maximizing the minima space volume for better generalization may be regarded as a kind of entropy maximization principle. Following Visser (2013), we may explicitly write the volume entropy as

$$
S _ {\mathrm{vol}} = \log \det (H ^ {- 1}) = - \int p (\lambda) \log \lambda d \lambda \tag {4}
$$

and the spectral entropy as

$$
S _ {\mathrm{p}} = - \int p (\lambda) \log p (\lambda) d \lambda , \tag {5}
$$

which is the entropy of the spectral density distribution.

Theorem 1 (The Maximum-Entropy Interpretation). Suppose we have the volume entropy $S_{vol}$ as Equation (4) and the spectral entropy $S_{p}$ as Equation (5). To find the optimal distribution $p^{\star}(\lambda)$ that maximizes the total entropy, where $S_{total} = S_{p} + \beta_{vol} S_{vol}$ and $\beta_{norm}$ is a Lagrange multiplier, the optimal distribution $p^{\star}(\lambda)$ can be solved as

$$
p ^ {\star} (\lambda) = e ^ {- \beta_ {\text { norm }}} \lambda^ {- \beta_ {\text { vol }}}. \tag {6}
$$

We leave the proof in Appendix A. We note that the result in Theorem 1 has an amazingly similar form to (1) with $\beta_{\mathrm{norm}} = \log Z_c$ and $\beta = \beta_{\mathrm{vol}}$ .

We may interpret the power-law structure of the Hessian spectrum from two basic maximum entropy principles with the spectral density normalization constraint. It roughly means that simple rules can almost explain the power-law Hessian spectrum in deep learning as well as statistical physics (Visser, 2013). While previous work in the field of deep learning has also interpreted the minima flatness of neural networks from an entropy perspective (Baldassi et al., 2020), the spectra have much simpler structures as shown with our empirical results.

Interestingly, similar well-fitted power laws have been widely discussed in neuroscience and biology (Reuveni et al., 2008; Tang & Kaneko, 2020). This exactly motivates us to further verify and study the power-law structure of the Hessian spectrum in the context of deep learning. We discover that the elegant power-law structure indeed exists in well-trained deep neural networks just like bioactive proteins. In contrast, random neural networks have no such power-law structure, just like deactivated (denatured or unfolded) proteins. Random neural networks which have no functional ability on the given task, break the power-law spectra similarly to deactivated proteins.

# 2.3. Goodness-of-fit Test

In this subsection, we are the first to conduct formal statistical tests on the Hessian structure. We also note that a recent work (Xie et al., 2023a) follow our statistical spectral analysis via KS Tests and only studied the structure of stochastic gradients rather than the Hessian structure.

Following Alstott et al. (2014), we use Maximum Likelihood Estimation (MLE) for estimating the parameter $\beta$ of the fitted power-law distributions and the Kolmogorov-Smirnov Test (KS Test) (Massey Jr, 1951; Goldstein et al., 2004) for statistically testing the goodness of the fit. The KS test statistic is the KS distance $d_{ks}$ between the hypothesized (fitted) distribution and the empirical data, which measures the goodness of fitting. Mathematically, the KS distance is defined as $d_{\mathrm{ks}} = \sup_{\lambda} |F^{\star}(\lambda) - \hat{F}(\lambda)|$ , where $F^{\star}(\lambda)$ is the hypothesized cumulative distribution function and $\hat{F}(\lambda)$ is the empirical cumulative distribution function based on the sampled data (Goldstein et al., 2004).

The estimated power exponent via MLE (Clauset et al., 2009) can be written as $\hat{\beta} = 1 + K\left[\sum_{i=1}^{K}\ln\left(\frac{\lambda_i}{\lambda_{\mathrm{cutoff}}}\right)\right]^{-1}$ , where $K$ is the number of tested samples and we set $\lambda_{\mathrm{cutoff}} = \lambda_k$ . The Powerlaw library (Alstott et al., 2014) provides a convenient tool to compute the KS distance, $d_{\mathrm{ks}}$ , and estimate the power exponent.

According to the practice of KS Test, we first state the power-law hypothesis that the tested spectrum is power-law. If $d_{ks}$ is lower than the critical value $d_{c}$ at the $\alpha = 0.05$ significance level, the KS test statistically will support (not reject) the power-law hypothesis. The test results associated with Figure 1 are presented in Table 1. We leave the details and more test results (e.g., ResNet18) in Appendix F.

When we say that a spectrum is (approximately) power-law in this paper, we mean that the KS test provides positive evidence to the power-law hypothesis instead of rejecting the power-law hypothesis. Our KS test results reject the power-law hypothesis for random neural networks and do not reject the power-law hypothesis for well-trained neural networks. Moreover, when the power-law hypothesis holds, the KS distance is usually significantly smaller than the critical value $d_{c}$ . For simplicity, the default $\alpha = 0.05$ significance

![](images/4a5e6963786265424d0575bed4b1d8857a279500d09712a3a98dbce8984dd74e.jpg)

<details>
<summary>scatter</summary>

| Rank | Eigenvalue or Eigengap |
| ---- | ---------------------- |
| 1    | 1.0                    |
| 10   | 0.1                    |
| 100  | 0.01                   |
| 1000 | 0.001                  |
</details>

(a) Eigenvalue Rank

![](images/6b8fec690b9b492590937011f51e0a1b9d11c99a331952237f85c7f45bd7bba6.jpg)

<details>
<summary>line</summary>

| Rank | Eigengaps | Eigenvalues |
| ---- | --------- | ----------- |
| 1    | 1.0       | 1.0         |
| 10   | 0.01      | 0.1         |
| 100  | 0.0001    | 0.01        |
| 1000 | 0.000001  | 0.001       |
</details>

(b) Eigengap Rank   
Figure 2. The power-law Hessian eigengaps. Model: LeNet. Datsets: MNIST. Subfigure (a) displayed the eigengaps by original rank indices sorted by eigenvalues. Subfigure (b) displayed the eigengaps by rank indices re-sorted by eigengaps. We also present the results of Fashion-MNIST in Figure 21 of Appendix D and GPT2-small in Figure 30 of Appendix E.

level is abbreviated in the following.

Following related work on the Hessian of neural networks (Thomas et al., 2020), our empirical analysis and statistical tests mainly focused on the top ( $\sim$ 1000) large eigenvalues larger than some minimal cutoff value $\lambda_{cutoff}$ for three reasons. First, focusing on relatively large values is very reasonable and common in various fields' power-law studies, as real-world distributions typically follow power laws only after/larger than some cutoff values (Clauset et al., 2009) to ensure the convergence of the probability distribution. Second, researchers are usually more interested in significantly large eigenvalues which contribute more to Hessian, minima sharpness, or generalization bound (Thomas et al., 2020). Third, empirically estimating a large number of nearly zero eigenvalues is very inaccurate and expensive.

# 2.4. The Power-Law Hessian Eigengaps

In this subsection, we report that the overlooked eigengaps of Hessian are also power-law and how the eigengaps suggest a robust and low-dimensional learning subspace.

The empirical investigation of the Hessian eigengaps is missing in previous works. Our experiments have closed this gap. Our experiments show that top eigengaps dominate other tailed eigengaps in deep learning. We visualize and verify the approximate power-law eigengaps in Figure 2.

The phenomenon of low-dimensional learning subspace was empirically reported recently (Gur-Ari et al., 2018; Ghorbani et al., 2019; Xie et al., 2021b) but still lacks theoretical understanding. Does the phenomenon theoretically depend on the Hessian structure? Our answer is yes. In the following part, we will demonstrate why the eigengaps of the Hessian $H$ may naturally lead to the phenomenon that learning dynamics mainly take place in a low-dimensional space during the entire training process.

![](images/b6389856c5e9c07d24255a3629bd18970e4980d6e1e98739bfc7cdbd5bd9b273.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | SGD     | Adam    | AMSGrad | AdaBound | RAdam   | Yogi    | Adia    | PNM     | Lookahead | Diffused | Vanilla SGD |
| --------------- | ------- | ------- | ------- | -------- | ------- | ------- | ------- | ------- | ---------- | -------- | ----------- |
| 10^0            | ~10^1   | ~10^1   | ~10^1   | ~10^1    | ~10^1   | ~10^1   | ~10^1   | ~10^1   | ~10^1      | ~10^1    | ~10^1       |
| 10^1            | ~10^0   | ~10^0   | ~10^0   | ~10^0    | ~10^0   | ~10^0   | ~10^0   | ~10^0   | ~10^0      | ~10^0    | ~10^0       |
| 10^2            | ~10^-1  | ~10^-1  | ~10^-1  | ~10^-1   | ~10^-1  | ~10^-1  | ~10^-1  | ~10^-1  | ~10^-1     | ~10^-1   | ~10^-1      |
| 10^3            | ~10^-2  | ~10^-2  | ~10^-2  | ~10^-2   | ~10^-2  | ~10^-2  | ~10^-2  | ~10^-2  | ~10^-2     | ~10^-2   | ~10^-2      |
</details>

![](images/d2bcb3b006459f8ab373026216a9c5aae0b340286d2c9ac01ef5fa105e94aa74.jpg)

<details>
<summary>scatter</summary>

| Method | s | Test Error |
|---|---|---|
| Pearson 0.853 | 1.00 | 0.012 |
| SGD | 1.00 | 0.0118 |
| Adam | 1.05 | 0.0122 |
| AMSGrad | 1.10 | 0.0126 |
| AdaBound | 1.10 | 0.0127 |
| Yingi | 1.10 | 0.0129 |
| RADam | 1.10 | 0.0129 |
| Adai | 1.15 | 0.0132 |
| PFM | 1.15 | 0.0134 |
| Lookahead | 1.15 | 0.0136 |
| DiffGrad | 1.20 | 0.0148 |
| Vanilla SGD | 1.20 | 0.0148 |
</details>

Figure 3. The power-law spectra hold across optimizers. Moreover, the slope magnitude $\hat{s}$ is an indicator of minima sharpness and a predictor of test performance. Model: LeNet. Dataset: MNIST.   
![](images/b998d0d49990fd47b942cd0e97c98b536e2760fbaff36e33e0988ae417880b5c.jpg)

<details>
<summary>scatter</summary>

| s    | λ₁   |
| ---- | ---- |
| 1.00 | 0.0  |
| 1.05 | 3.0  |
| 1.10 | 6.0  |
| 1.15 | 10.0 |
| 1.20 | 15.0 |
</details>

![](images/674936b28f0439c86d601fd9777db159c6cf4393ce54542e18e61305772eae51.jpg)

<details>
<summary>scatter</summary>

| s    | Tr(H) |
| ---- | ----- |
| 1.00 | 50    |
| 1.05 | 100   |
| 1.10 | 120   |
| 1.15 | 130   |
| 1.20 | 200   |
</details>

Figure 4. The slope magnitude $\hat{s}$ closely correlates to the largest Hessian eigenvalue and the Hessian trace. Model: LeNet. Dataset: MNIST.

To quantitatively understand why learning subspace is robust $^{1}$ , we may use the angle between the original Hessian eigenvector $u_{k}$ and the perturbed Hessian eigenvector $\tilde{u}_{k}$ , namely $\langle u_{k},\tilde{u}_{k}\rangle$ . Suppose the original Hessian is H, the perturbed Hessian is $\tilde{H}=H+\epsilon M$ , the i-th eigenvector of H is $u_{i}$ , and its corresponding perturbed eigenvector is $\tilde{u}_{i}$ . Under the conditions of the Davis-Kahan Theorem and (13), we have

$$
\begin{array}{l} \sup \sin \langle u _ {k}, \tilde {u} _ {k} \rangle = \frac {2 \epsilon \| M \| _ {o p}}{\min (\lambda_ {k - 1} - \lambda_ {k} , \lambda_ {k} - \lambda_ {k + 1})} \\ = \frac {2 \epsilon \| M \| _ {o p} (k + 1) ^ {s + 1}}{\lambda_ {1}}, \tag {7} \\ \end{array}
$$

where $\|M\|_{op}$ is the operator norm of the perturbation M. In the derivation details, we applied Theorem 2, a useful variant of the Davis-Kahan Theorem (Yu et al., 2015), directly to the Hessian in deep learning and demonstrate that the eigenspace robustness (spanned by the eigenvectors) is relatively tight for the top-learning eigenspace. To the extent of our knowledge, we are the first to theoretically explain the robust and low-dimensional learning subspace using Hessian eigengaps. Formal theoretical analysis and more discussion are available in Appendix B.

# 3. Empirical Analysis of CNNs

In this section, we conduct extensive experiments for exploring the behaviors of deep learning through the lens of power-law spectral analysis.

![](images/15ac9963b6f10cf507d495035fdfed52f021a234b25361d35b9f15e9cc65c628.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Depth=1 | Depth=2 | Depth=4 |
| --------------- | ------- | ------- | ------- |
| 10^0            | ~10^0   | ~10^0   | ~10^0   |
| 10^1            | ~10^0   | ~10^0   | ~10^0   |
| 10^2            | ~10^-1  | ~10^-1  | ~10^-1  |
| 10^3            | ~10^-2  | ~10^-2  | ~10^-2  |
</details>

![](images/0dd60ce4bb9a3100bfc8569cb2f780f27c5818407fd94dcd2ff1703a6f76c08e.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | #Training=600 | #Training=800 | #Training=1000 | #Training=3000 | #Training=6000 | #Training=80000 |
| --------------- | ------------- | ------------- | -------------- | -------------- | -------------- | --------------- |
| 10^0            | ~10^2         | ~10^2         | ~10^2          | ~10^2          | ~10^2          | ~10^2           |
| 10^1            | ~10^1         | ~10^1         | ~10^1          | ~10^1          | ~10^1          | ~10^1           |
| 10^2            | ~10^0         | ~10^0         | ~10^0          | ~10^0          | ~10^0          | ~10^0           |
| 10^3            | ~10^-2        | ~10^-2        | ~10^-2         | ~10^-2         | ~10^-2         | ~10^-2          |
| 10^4            | ~10^-3        | ~10^-3        | ~10^-3         | ~10^-3         | ~10^-3         | ~10^-3          |
</details>

Figure 5. The power-law spectrum holds well in overparameterized deep models but disappears in the underparameterized single-layer FCN.   
Figure 6. The spectra of LeNet on MNIST with respect to various numbers of training samples.

Models: LeNet (LeCun et al., 1998), Fully Connected Networks (FCN), and ResNet18 (He et al., 2016).

Datasets: MNIST (LeCun, 1998), Fashion-MNIST (Xiao et al., 2017), CIFAR-10/100 (Krizhevsky & Hinton, 2009), and non-image Avila (De Stefano et al., 2018).

1. Optimization and Generalization. Figure 3 discovered that the power-law spectrum consistently holds for various popular optimizers, such as SGD, Vanilla SGD, Adam, AMSGrad, AdaBound, Yogi, RAdam, Adai, PNM, Lookahead, and DiffGrad, as long as the optimizers can train the network well. We present the KS test results in Table 3.   
It is known that sharpness-based generalization measures are considered the most predictive generalization measures in deep learning (Jiang et al., 2019). We discover that the slope magnitude $\hat{s}$ of the fitted straight line may serve as a nice predictor of minima sharpness and generalization, when the power-law Hessian structure is well fitted. Note that it is common to measure minima's sharpness by the largest Hessian eigenvalue or the Hessian trace. A smaller $\hat{s}$ highly correlates to a smaller largest eigenvalue and a smaller trace in Figure 4. The similar observation holds on CIFAR-10 displayed in Figures 4 and 22 of Appendix D.   
2. Overparameterization. Figure 5 shows that the power-law spectrum holds well in overparameterized models, but disappears in underparameterized models. Overparameterization is necessary for the power-law spectrum in deep learning. It will be interesting to study phase transition of under-parameterization to over-parameterization in future.   
3. The size of training data. We evaluate the Hessian structure over various sized training data in Figure 6. The model trained with limited training data would break the power-law structure similarly as underparameterization, and lead to many sharp directions in the loss landscape. We see that data scaling and model scaling surprisingly exhibit very similar Hessian structures.   
4. Batch Size. We discover the three different phases for

![](images/3e31a172e92317ad3996e2f0bd293862e8353cc990968ce13fd337c2e5d613a8.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | B=128 | B=512 | B=640 | B=768 | B=1024 |
| --------------- | ----- | ----- | ----- | ----- | ------ |
| 10^0            | ~10^2 | ~10^1 | ~10^1 | ~10^1 | ~10^1  |
| 10^1            | ~10^1 | ~10^0 | ~10^0 | ~10^0 | ~10^0  |
| 10^2            | ~10^0 | ~10^-1| ~10^-1| ~10^-1| ~10^-1 |
| 10^3            | ~10^-2| ~10^-2| ~10^-2| ~10^-2| ~10^-2 |
| 10^4            | ~10^-3| ~10^-3| ~10^-3| ~10^-3| ~10^-3 |
</details>

![](images/9b139e57b5d6b88b7e7b0f1a5bb6594a0400fdfcd7268b33317e7228cebd4931.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | B=16384 | B=32768 | B=50000 | B=60000 | Random |
| --------------- | ------- | ------- | ------- | ------- | ------ |
| 10^0            | ~10^1   | ~10^3   | ~10^3   | ~10^3   | ~10^1  |
| 10^1            | ~10^1   | ~10^3   | ~10^3   | ~10^3   | ~10^1  |
| 10^2            | ~10^1   | ~10^3   | ~10^3   | ~10^3   | ~10^1  |
| 10^3            | ~10^-2  | ~10^2   | ~10^2   | ~10^2   | ~10^-2 |
| 10^4            | ~10^-3  | ~10^-3  | ~10^-3  | ~10^-3  | ~10^-3 |
</details>

Figure 7. Batch size matters to the spectrum. We discover three phases of the Hessian spectra for large-batch training. Model: LeNet. Dataset: MNIST.   
![](images/20b86fec8290d8fa95cef8febda59a3d962536a44114fb936848b9d095e26463.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | GPT2-nano | GPT2-small | GPT2-medium | GPT2-large |
| --------------- | --------- | ---------- | ----------- | ---------- |
| 10^0            | ~10^3     | ~10^2      | ~10^1       | ~10^1      |
| 10^1            | ~10^2     | ~10^1      | ~10^0       | ~10^0      |
| 10^2            | ~10^1     | ~10^0      | ~10^-1      | ~10^-1     |
| 10^3            | ~10^0     | ~10^-1     | ~10^-2      | ~10^-2     |
</details>

(a) Random / Pretrained

![](images/ec6c56473f1fc8900897666879fd8f785939a6128a52662749d967b3d1f0d48a.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | GPT2-nano | GPT2-small | GPT2-medium | GPT2-large |
| --------------- | --------- | ---------- | ----------- | ---------- |
| 10^0            | ~10^3     | ~10^3      | ~10^3       | ~10^3      |
| 10^1            | ~10^2     | ~10^2      | ~10^2       | ~10^2      |
| 10^2            | ~10^1     | ~10^1      | ~10^1       | ~10^1      |
| 10^3            | ~10^0     | ~10^0      | ~10^0       | ~10^0      |
</details>

(b) Trained / Fine-tuned   
Figure 8. The power-law Hessian structure exists in well-trained LLMs but disappear in their random initializations (GPT2-nano) or pretrained checkpoints (GPT2-small, GPT2-medium, GPT2-large). The power-law Hessian structure emerged on well-trained LLMs after training / fine-tuning.

large-batch training via the curves in Figure 7 and the KS test results in Table D of the appendix. To our knowledge, we are the first to report the phases and sharp phase transition for large-batch training. When we train CNNs with the same training epochs and let the batch size increase from 640 to 768, the power-law structure suddenly breaks. We see that inadequate training due to a large batch size exhibits a Hessian structure similar to one with limited training data. However, in Table D, training CNNs with the same iterations, we observe that large-batch training can also lead to power laws at the expense of more compute.

5. Supplementary Results. In Appendix D, we further discussed various interesting empirical results and insights, including studies on linear networks, modern architectures (such as ResNet18), noisy labels, task transferability, and the heavy-tail behavior of SGD.

# 4. Empirical Analysis of LLMs

In this section, we empirically studied how the power-law Hessian structure of LLMs behaves differently.

Models: GPT-2 family (Radford et al., 2019): GPT2-nano (11M), GPT2-small (124M), GPT2-medium (355M), and GPT2-large (774M), and TinyLlama (Zhang et al., 2024a) (1.1B-Chat-v1.0) with LoRA adapter (Hu et al., 2021).

Table 2. The Kolmogorov-Smirnov statistics of the Hessian spectra of various LLMs on various datasets. 

<table><tr><td>Dataset</td><td>Model</td><td>Training</td><td> $d_{\text{ks}}$ </td><td> $d_c$ </td><td>Power-Law</td></tr><tr><td>OpenWebText</td><td>GPT2-small</td><td>Random</td><td>0.1168</td><td>0.0430</td><td>No</td></tr><tr><td>OpenWebText</td><td>GPT2-small</td><td>Trained</td><td>0.0418</td><td>0.0430</td><td>Yes</td></tr><tr><td>Shakespeare</td><td>GPT2-nano</td><td>Random</td><td>0.0969</td><td>0.0430</td><td>No</td></tr><tr><td>Shakespeare</td><td>GPT2-nano</td><td>Fine-tuned</td><td>0.0353</td><td>0.0430</td><td>Yes</td></tr><tr><td>Shakespeare</td><td>GPT2-small</td><td>Pretrained</td><td>0.1058</td><td>0.0430</td><td>No</td></tr><tr><td>Shakespeare</td><td>GPT2-small</td><td>Fine-tuned</td><td>0.0259</td><td>0.0430</td><td>Yes</td></tr><tr><td>Shakespeare</td><td>GPT2-medium</td><td>Pretrained</td><td>0.0787</td><td>0.0430</td><td>No</td></tr><tr><td>Shakespeare</td><td>GPT2-medium</td><td>Fine-tuned</td><td>0.0184</td><td>0.0430</td><td>Yes</td></tr><tr><td>Shakespeare</td><td>GPT2-large</td><td>Pretrained</td><td>0.0496</td><td>0.0430</td><td>No</td></tr><tr><td>Shakespeare</td><td>GPT2-large</td><td>Fine-tuned</td><td>0.0160</td><td>0.0430</td><td>Yes</td></tr><tr><td>MathQA</td><td>Tinyllama (LoRA)</td><td>Random</td><td>0.0552</td><td>0.0430</td><td>No</td></tr><tr><td>MathQA</td><td>Tinyllama (LoRA)</td><td>Fine-tuned</td><td>0.0249</td><td>0.0430</td><td>Yes</td></tr></table>

![](images/ba87819aeafecbd3fbf35f17fcfbf7cd6516da536ccbc429643d7bb4e9837836.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | random-init | train-50 | train-100 | train-200 |
| --------------- | ----------- | -------- | --------- | --------- |
| 10^0            | ~10^1       | ~10^1    | ~10^-1    | ~10^-1    |
| 10^1            | ~10^1       | ~10^0    | ~10^-1    | ~10^-1    |
| 10^2            | ~10^1       | ~10^0    | ~10^-1    | ~10^-1    |
| 10^3            | ~10^0       | ~10^0    | ~10^-1    | ~10^-1    |
</details>

![](images/d16c0a8ccdca8405ebbdeedbd0a11ea0609e2ca7a8efad7f9b6513697fee823e.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | train-200 | train-1000 | train-5000 | train-50000 |
| --------------- | --------- | ---------- | ---------- | ----------- |
| 10^0            | ~1        | ~1         | ~1         | ~1          |
| 10^1            | ~0.5      | ~0.5       | ~0.5       | ~1          |
| 10^2            | ~0.2      | ~0.2       | ~0.2       | ~0.5        |
| 10^3            | ~0.1      | ~0.1       | ~0.1       | ~0.2        |
</details>

Figure 9. In the pretraining experiment of GPT2, the power-law Hessian structure emerged as training progressed in a two-stage process. In the first stage, the Hessian eigenvalues decrease in magnitude, indicating the discovery of a flat minimum. In the second stage, the primal Hessian eigenvalues increase to form a power-law distribution, reflecting a transition to a sharper minimum. Model: GPT2-small. Dataset: OpenWebText.

Datasets: OpenWebText (Gokaslan et al., 2019), Shakespeare (Karpathy, 2015), and MathQA (Amini et al., 2019).

1. Pre-training and Fine-tuning of LLMs. We empirically investigate the Hessian structure of LLMs with pretraining or fine-tuning. Figure 8 shows that well-trained LLMs after training/fine-tuning can exhibit the power-law Hessian structures, similarly to CNNs, while randomly initialized or pretrained models fail. Table 2 presents the KS test results on the power-law Hessian spectra across various LLMs, including GPT-2 series and TinyLlama-1b with LoRA adapter, in the case of pretraining or fine-tuning on various tasks.

The emergence of the power-law Hessian structure is evident throughout both the pretraining experiment (in Figure 9 and 10) and the fine-tuning experiment (in Figure 11). During pretraining, the power-law structure gradually develops as we optimize model parameters to capture the underlying structure of the training data. Similarly, during fine-tuning, the power-law structure adapts to reflect the specific characteristics of the target dataset. These findings, combined with empirical results from CNNs, suggest that the presence of the hessian Power-law structure is not confined to a specific model architecture or limited to either vision (Figure 14) or text data (Figure 10). Further experiment results regarding different problem setups are provided in the Appendix E.

![](images/81278b823779febdcba40f3093252fe45af141ed1b1ab521263fcb9e2dbe76af.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Eigenvalue (Red) | Eigenvalue (Blue) | Eigenvalue (Green) | Eigenvalue (Yellow) | Eigenvalue (Orange) |
| --------------- | ---------------- | ----------------- | ------------------ | ------------------- | ------------------- |
| 10^0            | ~200             | ~150              | ~100               | ~80                 | ~60                 |
| 10^1            | ~150             | ~100              | ~70                | ~50                 | ~40                 |
| 10^2            | ~100             | ~50               | ~30                | ~20                 | ~15                 |
| 10^3            | ~50              | ~20               | ~10                | ~5                  | ~3                  |
</details>

![](images/54a5163e980756ff776a7cdd03486580b45b0ca47dba7b1e8216dfccd0a103e2.jpg)

<details>
<summary>scatter</summary>

| d ks   | Test Loss | Group        |
|--------|-----------|--------------|
| 0.050  | 2.8       | train-50     |
| 0.050  | 2.6       | train-100    |
| 0.050  | 2.4       | train-200    |
| 0.050  | 2.2       | train-500    |
| 0.050  | 2.0       | train-1000   |
| 0.100  | 4.2       | random-init  |
| 0.125  | 4.5       | critical dc   |
</details>

![](images/88c9031fb092147bf06c7d46943abaf8056881a4eb296d839b9d1b67c7cd2b3b.jpg)

<details>
<summary>scatter</summary>

| λ₁   | Test Loss |
| ---- | --------- |
| 50   | 1.7       |
| 75   | 2.3       |
| 100  | 2.5       |
| 225  | 4.3       |
| 225  | 2.8       |
</details>

![](images/93a52b30a921a06827ed69f15817625483a6d1003384ec70c3018340facb8b2e.jpg)

<details>
<summary>scatter</summary>

| Tr(H)   | Test Loss |
| ------- | --------- |
| 10^3    | 2.5       |
| 10^4    | 2.8       |
| 10^5    | 4.2       |
</details>

Figure 10. The KS distance $d_{ks}$ may serve as an effective predictor to language model's generalization abilities with a Pearson correlation coefficient up to 0.923. Model: GPT2-nano (trained from random initialization). Dataset: Shakespeare.

2. Generalization Measure. We studied how the generalization of LLMs closely relates to the power-law Hessian structure across different models and datasets.

We observed that for Figure 10 and Figure 11, minima sharpness represented by the largest eigenvalue $\lambda_{1}$ and Hessian trace $Tr(H)$ behaves poorly as a generalization measure for LLMs, contradicting conventional beliefs in deep learning (Jiang et al., 2019).

In the pretraining experiment of GPT2-nano, Figure 10 shows that pretraining can decrease the KS distance and test loss effectively. However, surprisingly, we discover that sharpness-based generalization measures become nearly useless for predicting generalization. In contrast, the KS distance can serve as an effective predictor to generalization ability with the Pearson correlation coefficient up to 0.923. Note that the KS distance $d_{ks}$ metric quantifies the adherence of a model's power-law spectral properties to those expected of a well-trained neural network.

In the fine-tuning experiment of GPT2-small in Figure 11, we notice similar observations. Again, the sharpness-based generalization measure fails, whereas the power-law goodness can work well. We can even see that the sharpness increases significantly during iterations 50-5000, whereas the test loss still drops quickly. Similar observation holds in Figure 9. In contrast, during all 5000 iterations, the test loss and the KS distance continuously drop synchronously

![](images/7e787e8ea448d63cfed41b6301624e6a8a11abe64ac63f494b91baa643cf5839.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Eigenvalue (Red) | Eigenvalue (Orange) | Eigenvalue (Green) | Eigenvalue (Blue) |
| --------------- | ---------------- | ------------------- | ------------------ | ----------------- |
| 10^0            | ~10^2            | ~10^2               | ~10^2              | ~10^1             |
| 10^1            | ~10^1            | ~10^1               | ~10^1              | ~10^0             |
| 10^2            | ~10^0            | ~10^0               | ~10^0              | ~10^-1            |
| 10^3            | ~10^-1           | ~10^-1              | ~10^-1             | ~10^-2            |
</details>

![](images/27c35997ac23099a7d194f759f7f582025f578141a18af2cb354890e98c3c585.jpg)

<details>
<summary>scatter</summary>

| Model           | d_ks   | Test Loss |
| --------------- | ------ | --------- |
| pretrain        | 0.10   | 3.5       |
| finetune-50     | 0.06   | 2.5       |
| finetune-100    | 0.05   | 2.4       |
| finetune-500    | 0.05   | 1.8       |
| finetune-1000   | 0.03   | 1.7       |
| finetune-5000   | 0.03   | 1.5       |
</details>

![](images/ca4f4829be5474fb803a0907baf20e913aa49fb43b677bb779fde57b2f100805.jpg)

<details>
<summary>scatter</summary>

| λ₁  | Test Loss |
| --- | --------- |
| 10  | 2.5       |
| 30  | 2.3       |
| 35  | 2.1       |
| 50  | 1.8       |
| 55  | 1.6       |
</details>

![](images/c8b4f4252aa3906fa66c2015ece87dd57cc1d48ceace9dca4c8a64d135212ad2.jpg)

<details>
<summary>scatter</summary>

| Tr(H) | Test Loss |
|-------|-----------|
| 100   | 2.5       |
| 100   | 2.3       |
| 100   | 2.1       |
| 100   | 1.9       |
| 100   | 1.7       |
| 100   | 1.5       |
| 1000  | 3.5       |
</details>

Figure 11. The power-law Hessian structure also emerged in fine-tuning tasks. The KS distance $d_k s$ predicts the model's generalization abilities with a strong Pearson correlation coefficient up to 0.980, while the sharpness-based generalization measures obviously fail in predicting generalization of LLMs. Model: GPT2-small. Dataset: Shakespeare.

with the Pearson correlation coefficient up to 0.98, while the Pearson correlation coefficients for sharpness-based generalization measures are even only -0.714 and 0.247, which are harmful or nearly useless for predicting generalization.

We conjecture that the power-law Hessian structure and the minima's sharpness of the loss landscape can capture a model's generalization ability at different phases. Unlike CNNs, LLMs are often extremely over-parameterized and far from well-trained (e.g., many epochs). In the phase of staying far from minima, the power-law Hessian structure can better predict the generalization of LLMs. In contrast, as people usually train CNNs for many epochs, well-trained CNNs stay very close to minima. In the phase of staying close to minima, generalization can be captured by the minima's sharpness better, following conventional generalization theory. We believe a detailed generalization analysis for LLMs remains an open area for future research.

3. Model Capacity and Scaling. Scaling law predicts that the performance of LLMs typically follows power laws as we scale model parameters, training data, or computing (Bahri et al., 2024). We further investigate how the power-law Hessian structure depends on model scaling.

Figures 12 and 13 present the results of various pretraining checkpoint (step = 5000) and pretrained GPT-2 models trained and evaluated on OpenWebText. The results demonstrate that as model parameters increase, the power-law Hessian structure also becomes significantly more pronounced for both training and pretrained checkpoints, accompanied by improved model performance. The Pearson correlation coefficients are higher than 0.99 for the intermediate checkpoint and also high for the official pretrained models. This observation further supports that a scaling law perspective is reflected in the power-law Hessian structure of the LLM.

![](images/75f679f378cbc125c1baa1f260e248374b4fff5a05ce6ca17d5689bed4c25e3c.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Eigenvalue (Red) | Eigenvalue (Blue) | Eigenvalue (Green) |
| --------------- | ---------------- | ----------------- | ------------------ |
| 10^0            | ~10^1            | ~10^0             | ~10^-1             |
| 10^1            | ~10^0            | ~10^-1            | ~10^-2             |
| 10^2            | ~10^-1           | ~10^-2            | ~10^-3             |
| 10^3            | ~10^-2           | ~10^-3            | ~10^-4             |
</details>

![](images/cee2b3a78cb6ce81513e77037ea7dd2042a3bd05560d876d211ec9d77e2bf696.jpg)

<details>
<summary>line</summary>

| d_ks   | Test Loss |
| ------ | --------- |
| 0.04   | 3.50      |
| 0.06   | 3.70      |
| 0.08   | 3.75      |
</details>

(a) GPT-2 (Pretraining with 5000 steps)

![](images/c81b4ed734c07ba623ae2ec7ee2ffa598e18dbdf21b86386d89022f4da62cb64.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Eigenvalue (Red) | Eigenvalue (Green) | Eigenvalue (Blue) |
| --------------- | ---------------- | ------------------ | ----------------- |
| 10^0            | ~10^1            | ~10^0              | ~10^-1            |
| 10^1            | ~10^0            | ~10^-1             | ~10^-2            |
| 10^2            | ~10^-1           | ~10^-2             | ~10^-3            |
| 10^3            | ~10^-2           | ~10^-3             | ~10^-4            |
</details>

![](images/3e394582a8336c7496104e5ab7be37d7cf0f11cc12fcc0cf0aba85f5b325b817.jpg)

<details>
<summary>scatter</summary>

| d_ks   | Test Loss | Model        |
|--------|-----------|--------------|
| 0.02   | 2.50      | gpt2-large   |
| 0.02   | 2.80      | gpt2-medium  |
| 0.04   | 3.50      | gpt2-small   |
| 0.06   | 3.75      | gpt2-small   |
</details>

(b) GPT-2 (Pretrained Models)   
Figure 12. The power-law spectrum holds across GPT-2 models with different model capacity. As we scale model parameters, the Hessian power-law structure goodness and test performance both improve. Subfigure (a) displays the eigenvalues of intermediate pretraining checkpoints at 5000 steps; Subfigure (b) displays the eigenvalues of the official pretrained GPT-2 models. Models: GPT2-{small, medium, large}. Dataset: OpenWebText.

4. Vision Transformer. We also investigate the power-law Hessian structure of Vision Transformer on vision datasets. We analyze the pre-trained and fine-tuned ViT-base models (Dosovitskiy et al., 2021) on the CIFAR-100 dataset. Figure 14 shows that the power-law Hessian structure is absent in random ViT, and emerges in well-trained ViT models.   
5. Supplementary Results. We present additional results and discussion, including layer-wise Hessian analysis and LoRA fine-tuning, in Appendix E. (1) In Figure 31, we observe that the power-law Hessian structure exists in various layers of LLMs. The power-law Hessian structure of the first and middle layers is often more pronounced than the last layer. (2) In Figure 33, we discover that while fine-tuning LLMs with LoRA can improve the performance of LLMs on a specific task, LoRA cannot significantly affect the Hessian structure of LLMs as full-parameter fine-tuning. The power-law Hessian structure does not emerge with LoRA fine-tuning. It suggests that full-parameter and LoRA fine-tuning may have significant generalization abilities.

![](images/472ca046088f7dff64c5e5f5c8696a74858336fb09fbfd371edeb6a44fa9f795.jpg)

<details>
<summary>line</summary>

| Parameters (million) | d1s    |
| -------------------- | ------ |
| 124M                 | 0.075  |
| 354M                 | 0.055  |
| 774M                 | 0.040  |
</details>

![](images/90a31ff004e679907982f8320518600f1bc80661f0bf0a618678600d25be5cea.jpg)

<details>
<summary>scatter</summary>

| Parameters (million) | d_ss   | Group       |
| -------------------- | ------ | ----------- |
| 124M                 | 0.05   | GPT2-small  |
| 354M                 | 0.02   | GPT2-medium |
| 774M                 | 0.02   | GPT2-large  |
</details>

Figure 13. Scaling law of the power-law goodness for the Hessian structure. Larger models have a more precise power-law Hessian structure. Left: Intermediate GPT-2 pretraining checkpoints at 5000 steps; Right: Official pretrained GPT-2 models. Models: GPT2-{small, medium, large}. Dataset: OpenWebText.

![](images/d2e97580ebb9f7c726a6078d07ab2449e51aa83aa2b92d72a6b0a3a959b7ca59.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | random-init | pretrain | well-trained |
| --------------- | ----------- | -------- | ------------ |
| 1               | 1.0         | 0.1      | 0.5          |
| 10              | 1.0         | 0.1      | 0.3          |
| 100             | 0.5         | 0.05     | 0.1          |
| 1000            | 0.2         | 0.02     | 0.05         |
</details>

Figure 14. The power-law Hessian structure is absent in randomly initialized Vision Transformers and emerges with training. Model: ViT-base. Dataset: CIFAR-100.

# 5. Related Work

The Hessian structure of DNNs. A number of related works analyzed the spectral distribution of the Hessian in deep learning. Pennington & Bahri (2017) introduced an analytical framework from random matrix theory and reported that the shape of the spectrum depends strongly on the energy and the over-parameterization parameter, $\phi$ , which measures the ratio of parameters to data points. However, Pennington & Bahri (2017) mainly evaluated single-hidden-layer networks, which limits the scope of the conclusion. A followup work (Pennington & Worah, 2018) focused on a single-hidden-layer neural network with Gaussian data and weights in the limit of infinite width. Obviously, its theoretical and empirical analysis is far from practical deep models. Jacot et al. (2019) analyzed the limiting spectrum of the Hessian in neural networks with infinite width. Fort & Scherlis (2019) analyzed the Hessian spectra of initialized neural networks. Papyan (2019) studied the three-level hierarchical structure and outliers in Hessian spectra. Singh et al. (2021) proved that the Hessian can be of very low rank for DNNs with linear activations. Liao & Mahoney (2021) studied the Hessian spectra of more realistic nonlinear models. Kaur et al. (2023) studied the maximum Hessian eigenvalue and its relation to generalization. Dauphin et al. (2024) investigated the neglected Nonlinear Modeling Error (NME) matrix part of the Hessian and its influence on gradient penalties during training. While a number of works studied the Hessian spectra, they failed to empirically or theoretically discover the simple but important power-law structure, nor reveal its connection to generalization.

Other power-law phenomena in deep learning. Hestness et al. (2017) studied the power-law relation between model performance and model size as well as data size. Mahoney & Martin (2019) reported that the elements of weight matrices may exhibit power-law heavy tails and studied the trends of spectral decay. Lee et al. (2020); Velikanov & Yarotsky (2021) reported the power-law decaying eigenvalues in kernel methods. Agrawal et al. (2022) studied the eigenspectrum decaying of feature covariance with a theoretical analysis of linear regression. Xie et al. (2023a) studied the heavy-tailed structure of stochastic gradient covariance. However, none of them reported the overlooked power-law Hessian structure of DNNs.

# 6. Conclusion

While the Hessian of the deep loss landscape matters to optimization and generalization of deep learning, the statistical structure of the Hessian is still largely overlooked by previous studies. To the best knowledge, we are the first to report the overlooked power-law Hessian structure in deep learning as well as formal statistical tests on the power-law Hessian spectra. We provide a novel maximum-entropy interpretation and explain why the learning space may be low-dimensional and robust. While the main limitation of our work is that we cannot include those state-of-the-art LLMs in our experiments due to the extremely large memory and computational cost of Hessian analysis, our work still goes much further beyond those previous qualitative studies. The power-law Hessian structure provides a useful and novel perspective to reveal and analyze multiple novel behaviors of deep learning on optimization, generalization, and over-parameterization. We particularly discovered that while conventional sharpness-based generalization measures are considered nice generalization predictors of CNNs, they often completely fail to predict the generalization of LLMs. Instead, the power-law goodness of the Hessian structure often correlates better with generalization on many deep learning occasions, while we also observe that it is not a robust generalization measure. This suggests that generalization theory and measures of LLMs lacks more exploration and rethinking. We believe that our work will further inspire theories and empirical advances toward a deeper understanding of DNNs, loss landscape, and Hessian.

# Acknowledgement

This work was sponsored by Doubao Large Model Fund of ByteDance, Natural Science Foundation of China (No. 12305052), Research Grants Council of Hong Kong (Nos. 22302723 and SRFS2324-2S05), Hong Kong Baptist University's funding support (RC-FNRA-IG/22-23/SCI/03).

# Impact Statement

This paper presents work whose goal aims at understanding the foundation of deep learning. While it may have many potential societal consequences, we think none of them must be specifically discussed here.

# References

Achille, A. and Soatto, S. Where is the information in a deep neural network? arXiv preprint arXiv:1905.12213, 2019.   
Agrawal, K. K., Mondal, A. K., Ghosh, A., and Richards, B. alpha-req: Assessing representation quality in self-supervised learning by measuring eigenspectrum decay. Advances in Neural Information Processing Systems, 35:17626–17638, 2022.   
Alstott, J., Bullmore, E., and Plenz, D. powerlaw: a python package for analysis of heavy-tailed distributions. PloS one, 9(1):e85777, 2014.   
Amini, A., Gabriel, S., Lin, S., Koncel-Kedziorski, R., Choi, Y., and Hajishirzi, H. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2357–2367, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1245. URL https://aclanthology.org/N19-1245.   
Bahri, Y., Dyer, E., Kaplan, J., Lee, J., and Sharma, U. Explaining neural scaling laws. Proceedings of the National Academy of Sciences, 121(27):e2311878121, 2024.   
Baldassi, C., Pittorino, F., and Zecchina, R. Shaping the learning landscape in neural networks around wide flat minima. Proceedings of the National Academy of Sciences, 117(1):161–170, 2020.   
Byrd, R. H., Chin, G. M., Neveitt, W., and Nocedal, J. On the use of stochastic hessian information in optimization methods for machine learning. SIAM Journal on Optimization, 21(3):977–995, 2011.

Clauset, A., Shalizi, C. R., and Newman, M. E. Power-law distributions in empirical data. SIAM review, 51(4):661–703, 2009.

Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., and Bengio, Y. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. Advances in Neural Information Processing Systems, 27:2933–2941, 2014.

Dauphin, Y. N., Agarwala, A., and Mobahi, H. Neglected hessian component explains mysteries in sharpness regularization. arXiv preprint arXiv:2401.10809, 2024.

De Stefano, C., Maniaci, M., Fontanella, F., and di Freca, A. S. Reliable writer identification in medieval manuscripts through page layout features: The “avila” bible case. Engineering Applications of Artificial Intelligence, 72:99–110, 2018.

Dinh, L., Pascanu, R., Bengio, S., and Bengio, Y. Sharp minima can generalize for deep nets. In International Conference on Machine Learning, pp. 1019–1028, 2017.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. URL https://arxiv.org/abs/2010.11929.

Dubey, S. R., Chakraborty, S., Roy, S. K., Mukherjee, S., Singh, S. K., and Chaudhuri, B. B. diffgrad: An optimization method for convolutional neural networks. IEEE transactions on neural networks and learning systems, 31(11):4500–4511, 2019.

Fort, S. and Scherlis, A. The goldilocks zone: Towards better understanding of neural network loss landscapes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 3574–3581, 2019.

Ghorbani, B., Krishnan, S., and Xiao, Y. An investigation into neural net optimization via hessian eigenvalue density. In International Conference on Machine Learning, pp. 2232–2241. PMLR, 2019.

Gokaslan, A., Cohen, V., Pavlick, E., and Tellex, S. Openwebtext corpus. http://Skylion007.github.io/OpenWebTextCorpus, 2019.

Goldstein, M. L., Morris, S. A., and Yen, G. G. Problems with fitting to the power-law distribution. The European Physical Journal B-Condensed Matter and Complex Systems, 41(2):255–258, 2004.

Guiasu, S. and Shenitzer, A. The principle of maximum entropy. The mathematical intelligencer, 7(1):42–48, 1985.

Gur-Ari, G., Roberts, D. A., and Dyer, E. Gradient descent happens in a tiny subspace. arXiv preprint arXiv:1812.04754, 2018.   
Gurbuzbalaban, M., Simsekli, U., and Zhu, L. The heavy-tail phenomenon in sgd. In International Conference on Machine Learning, pp. 3964–3975. PMLR, 2021.   
Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., and Sugiyama, M. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In Advances in neural information processing systems, pp. 8527–8537, 2018.   
Han, B., Yao, Q., Liu, T., Niu, G., Tsang, I. W., Kwok, J. T., and Sugiyama, M. A survey of label-noise representation learning: Past, present and future. arXiv preprint arXiv:2011.04406, 2020.   
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.   
He, Z., Xie, Z., Zhu, Q., and Qin, Z. Sparse double descent: Where network pruning aggravates overfitting. In International Conference on Machine Learning, pp. 8635–8659. PMLR, 2022.   
Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M., Ali, M., Yang, Y., and Zhou, Y. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.   
Hochreiter, S. and Schmidhuber, J. Simplifying neural nets by discovering flat minima. In Advances in neural information processing systems, pp. 529–536, 1995.   
Hochreiter, S. and Schmidhuber, J. Flat minima. Neural Computation, 9(1):1–42, 1997.   
Hodgkinson, L. and Mahoney, M. Multiplicative noise and heavy tails in stochastic optimization. In International Conference on Machine Learning, pp. 4262–4274. PMLR, 2021.   
Hoffer, E., Hubara, I., and Soudry, D. Train longer, generalize better: closing the generalization gap in large batch training of neural networks. In Advances in Neural Information Processing Systems, pp. 1729–1739, 2017.   
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.   
Jacot, A., Gabriel, F., and Hongler, C. The asymptotic spectrum of the hessian of dnn throughout training. In

International Conference on Learning Representations, 2019.   
Jiang, Y., Neyshabur, B., Mobahi, H., Krishnan, D., and Bengio, S. Fantastic generalization measures and where to find them. In International Conference on Learning Representations, 2019.   
Karpathy, A. char-rnn. https://github.com/karpathy/char-rnn, 2015.   
Karpathy, A. NanoGPT. https://github.com/karpathy/nanoGPT, 2022.   
Kaur, S., Cohen, J., and Lipton, Z. C. On the maximum hessian eigenvalue and generalization. In Proceedings on, pp. 51–65. PMLR, 2023.   
Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. 3rd International Conference on Learning Representations, ICLR 2015, 2015.   
Krizhevsky, A. and Hinton, G. Learning multiple layers of features from tiny images. 2009.   
LeCun, Y. The mnist database of handwritten digits. http://yann.lecun.com/exdb/mnist/, 1998.   
LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.   
Lee, J., Schoenholz, S., Pennington, J., Adlam, B., Xiao, L., Novak, R., and Sohl-Dickstein, J. Finite versus infinite neural networks: an empirical study. Advances in Neural Information Processing Systems, 33:15156–15172, 2020.   
Li, X., Gu, Q., Zhou, Y., Chen, T., and Banerjee, A. Hessian based analysis of sgd for deep nets: Dynamics and generalization. In Proceedings of the 2020 SIAM International Conference on Data Mining, pp. 190–198. SIAM, 2020.   
Li, Z., Malladi, S., and Arora, S. On the validity of modeling SGD with stochastic differential equations (SDEs). In Thirty-Fifth Conference on Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=goEdyJ\_nVQI.   
Liao, Z. and Mahoney, M. W. Hessian eigenspectra of more realistic nonlinear models. Advances in Neural Information Processing Systems, 34, 2021.   
Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., and Han, J. On the variance of the adaptive learning rate and beyond. In International Conference on Learning Representations, 2019.   
Loshchilov, I. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Luo, L., Xiong, Y., Liu, Y., and Sun, X. Adaptive gradient methods with dynamic bound of learning rate. 7th International Conference on Learning Representations, ICLR 2019, 2019.   
Mahoney, M. and Martin, C. Traditional and heavy tailed self regularization in neural network models. In International Conference on Machine Learning, pp. 4284–4293. PMLR, 2019.   
Massey Jr, F. J. The kolmogorov-smirnov test for goodness of fit. Journal of the American statistical Association, 46(253):68–78, 1951.   
Meurant, G. and Strakoš, Z. The lanczos and conjugate gradient algorithms in finite precision arithmetic. Acta Numerica, 15:471–542, 2006.   
Myung, I. J. Tutorial on maximum likelihood estimation. Journal of mathematical Psychology, 47(1):90–100, 2003.   
Neyshabur, B., Bhojanapalli, S., McAllester, D., and Srebro, N. Exploring generalization in deep learning. In Advances in Neural Information Processing Systems, pp. 5949–5958, 2017.   
Ormaniec, W., Dangel, F., and Singh, S. P. What does it mean to be a transformer? insights from a theoretical hessian analysis. arXiv preprint arXiv:2410.10986, 2024.   
Panigrahi, A., Somani, R., Goyal, N., and Netrapalli, P. Nongaussianity of stochastic gradient noise. arXiv preprint arXiv:1910.09626, 2019.   
Papyan, V. Measurements of three-level hierarchical structure in the outliers in the spectrum of deepnet hessians. In International Conference on Machine Learning, pp. 5012–5021. PMLR, 2019.   
Pennington, J. and Bahri, Y. Geometry of neural network loss surfaces via random matrix theory. In International Conference on Machine Learning, pp. 2798–2806. PMLR, 2017.   
Pennington, J. and Worah, P. The spectrum of the fisher information matrix of a single-hidden-layer neural network. In NeurIPS, pp. 5415–5424, 2018.   
Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
Reddi, S. J., Kale, S., and Kumar, S. On the convergence of adam and beyond. 6th International Conference on Learning Representations, ICLR 2018, 2019.

Reuveni, S., Granek, R., and Klafter, J. Proteins: coexistence of stability and flexibility. Physical review letters, 100(20):208101, 2008.   
Sagun, L., Bottou, L., and LeCun, Y. Eigenvalues of the hessian in deep learning: Singularity and beyond. arXiv preprint arXiv:1611.07476, 2016.   
Sagun, L., Evci, U., Guney, V. U., Dauphin, Y., and Bottou, L. Empirical analysis of the hessian of over-parametrized neural networks. arXiv preprint arXiv:1706.04454, 2017.   
Sankar, A. R., Khasbage, Y., Vigneswaran, R., and Bala-subramanian, V. N. A deeper look at the hessian eigenspectrum of deep neural networks and its applications to regularization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 9481–9488, 2021.   
Simsekli, U., Sagun, L., and Gurbuzbalaban, M. A tail-index analysis of stochastic gradient noise in deep neural networks. In International Conference on Machine Learning, pp. 5827–5837, 2019.   
Singh, S. P., Bachmann, G., and Hofmann, T. Analytic insights into structure and rank of neural network hessian maps. Advances in Neural Information Processing Systems, 34:23914–23927, 2021.   
Tang, Q.-Y. and Kaneko, K. Long-range correlation in protein dynamics: Confirmation by structural data and normal mode analysis. PLoS computational biology, 16(2):e1007670, 2020.   
Thomas, V., Pedregosa, F., Merriënboer, B., Manzagol, P.-A., Bengio, Y., and Le Roux, N. On the interplay between noise and curvature and its effect on optimization and generalization. In International Conference on Artificial Intelligence and Statistics, pp. 3503–3513. PMLR, 2020.   
Velikanov, M. and Yarotsky, D. Explicit loss asymptotics in the gradient descent training of neural networks. Advances in Neural Information Processing Systems, 34:2570–2582, 2021.   
Visser, M. Zipf's law, power laws and maximum entropy. New Journal of Physics, 15(4):043021, 2013.   
Wu, L., Zhu, Z., et al. Towards understanding generalization of deep learning: Perspective of loss landscapes. arXiv preprint arXiv:1706.10239, 2017.   
Xiao, H., Rasul, K., and Vollgraf, R. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.   
Xie, Z., He, F., Fu, S., Sato, I., Tao, D., and Sugiyama, M. Artificial neural variability for deep learning: On overfitting, noise memorization, and catastrophic forgetting. Neural Computation, 33(8), 2021a.

Xie, Z., Sato, I., and Sugiyama, M. A diffusion theory for deep learning dynamics: Stochastic gradient descent exponentially favors flat minima. In International Conference on Learning Representations, 2021b. URL https://openreview.net/forum?id=wXgk\_iCiYGo.   
Xie, Z., Yuan, L., Zhu, Z., and Sugiyama, M. Positive-negative momentum: Manipulating stochastic gradient noise to improve generalization. In International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 11448–11458. PMLR, 18–24 Jul 2021c.   
Xie, Z., Wang, X., Zhang, H., Sato, I., and Sugiyama, M. Adaptive inertia: Disentangling the effects of adaptive learning rate and momentum. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 24430–24459, 2022.   
Xie, Z., Tang, Q., Sun, M., and Li, P. On the overlooked structure of stochastic gradients. In Thirty-seventh Conference on Neural Information Processing Systems, 2023a.   
Xie, Z., Xu, Z., Zhang, J., Sato, I., and Sugiyama, M. On the overlooked pitfalls of weight decay and how to mitigate them: A gradient-norm perspective. In Thirty-seventh Conference on Neural Information Processing Systems, 2023b.   
Yao, Z., Gholami, A., Lei, Q., Keutzer, K., and Mahoney, M. W. Hessian-based analysis of large batch training and robustness to adversaries. In Advances in Neural Information Processing Systems, pp. 4949–4959, 2018.   
Yao, Z., Gholami, A., Keutzer, K., and Mahoney, M. W. Pyhessian: Neural networks through the lens of the hessian. In 2020 IEEE International Conference on Big Data (Big Data), pp. 581–590. IEEE, 2020.   
Yu, Y., Wang, T., and Samworth, R. J. A useful variant of the davis–kahan theorem for statisticians. Biometrika, 102(2):315–323, 2015.   
Zaheer, M., Reddi, S., Sachan, D., Kale, S., and Kumar, S. Adaptive methods for nonconvex optimization. In Advances in neural information processing systems, pp. 9793–9803, 2018.   
Zhang, M., Lucas, J., Ba, J., and Hinton, G. E. Lookahead optimizer: k steps forward, 1 step back. Advances in Neural Information Processing Systems, 32:9597–9608, 2019.   
Zhang, P., Zeng, G., Wang, T., and Lu, W. Tinyllama: An open-source small language model, 2024a.

Zhang, Y., Chen, C., Ding, T., Li, Z., Sun, R., and Luo, Z.-Q. Why transformers need adam: A hessian perspective. arXiv preprint arXiv:2402.16788, 2024b.

# A. Proof of Theorem 1

Proof. Considering the principle of maximum entropy with the two kinds of entropy, we need to maximize the total entropy with the spectral density normalization constraint

$$
\begin{array}{l} S _ {\text { total }} = - \int p (\lambda) \log p (\lambda) d \lambda - \beta_ {\text { vol }} \int p (\lambda) \log \lambda d \lambda \tag {8} \\ - \beta_ {\mathrm{norm}} (\int p (\lambda) d \lambda - 1), \\ \end{array}
$$

where $S_{total} = S_{p} + \beta_{vol} S_{vol}$ and $\beta_{norm}$ is a Lagrange multiplier. To find the optimal distribution $p^{\star}(\lambda)$ that maximizes the total entropy, we require the following

$$
\frac {\partial S _ {\text { total }}}{\partial p (\lambda)} = - \log p (\lambda) - \beta_ {\text { vol }} \log \lambda - \beta_ {\text { norm }} = 0. \tag {9}
$$

Thus, the optimal distribution $p^{\star}(\lambda)$ can be solved as

$$
p ^ {\star} (\lambda) = e ^ {- \beta_ {\text { norm }}} \lambda^ {- \beta_ {\text { vol }}}. \tag {10}
$$

# B. Robust and Low-Dimensional Learning Space

At first, we denote the ordered eigengap as $\delta_k = \lambda_k - \lambda_{k+1}$ , which means the difference between two neighbored eigenvalues. According to (2), we have

$$
\begin{array}{l} \delta_ {k} = \operatorname{Tr} (H) Z _ {d} ^ {- 1} \left(k ^ {- \frac {1}{\beta - 1}} - (k + 1) ^ {- \frac {1}{\beta - 1}}\right) (11) \\ = \lambda_ {k} \left[ 1 - (\frac {k}{k + 1}) ^ {s} \right]. (12) \\ \end{array}
$$

It shows that the eigengaps of Hessians also approximately exhibit a power-law distribution

$$
\delta_ {k} = \operatorname{Tr} (H) Z _ {d} ^ {- 1} (k + 1) ^ {- (s + 1)} \tag {13}
$$

under the approximation $s \approx 1$ . The power exponent $s + 1$ is larger than the one in (2) by 1.

The phenomenon of low-dimensional learning subspace was empirically reported by Gur-Ari et al. (2018). Ghorbani et al. (2019) also investigated and reported that large isolated eigenvalues quickly appear in the spectrum during the optimization process, along with a surprising concentration of the gradient in the corresponding eigenspace. However, they did not theoretically explain this phenomenon. Xie et al. (2021b) theoretically demonstrated that learning dynamics mainly happens along those principal eigenvectors of Hessian corresponding to large eigenvalues. Does the phenomenon theoretically depend on the Hessian structure? Our answer is yes. In the following part, we will demonstrate why the eigengaps of the Hessian $H$ may naturally lead to the phenomenon that learning dynamics mainly takes place in a low-dimensional space during the entire training process.

Previous papers only reported that top eigenvalues of Hessian are significantly larger than other tailed ones but did not touch how top Hessian eigengaps dominate other tailed ones in deep learning. However, top large eigenvalues do not imply their eigengaps are relatively large, too. Fortunately, we theoretically and empirically demonstrate that, as the rank index increases, both eigenvalues and eigengaps decay following power laws. Moreover, the eigengaps decay faster than eigenvalues. The theoretical implication behind the power-law eigengaps actually matters to deep learning dynamics.

We directly apply Theorem 2, a useful variant of Davis-Kahan Theorem (Yu et al., 2015), to the Hessian in deep learning.

Theorem 2 (A useful variant of Davis-Kahan Theorem (Yu et al., 2015)). Suppose the true Hessian is $H$ , the perturbed Hessian is $\tilde{H} = H + \epsilon M$ , the $i$ -th eigenvector of $H$ is $u_i$ , and its corresponding perturbed eigenvector is $\tilde{u}_i$ . Under the conditions of the Davis-Kahan Theorem, we have

$$
\sin \langle u _ {k}, \tilde {u} _ {k} \rangle \leq \frac {2 \epsilon \| M \| _ {o p}}{\min (\lambda_ {k - 1} - \lambda_ {k} , \lambda_ {k} - \lambda_ {k + 1})},
$$

where $\| M\|_{op}$ is the operator norm of $M$ .

Given the power-law eigengaps in Equation (13), the upper bound of eigenvector robustness can be written as

$$
\sup \sin \langle u _ {k}, \tilde {u} _ {k} \rangle = \frac {2 \epsilon \| M \| _ {o p} (k + 1) ^ {s + 1}}{\lambda_ {1}}, \tag {14}
$$

which is relatively tight for top dimensions and very loose for tailed dimensions. A similar conclusion also holds given Equation (11). This indicates that non-top eigenspace can be highly unstable during training, because $\delta_{k}$ can decay to nearly zero for a large k. To the best of our knowledge, we are the first to demonstrate that the robustness of low-dimensional learning space directly depends on the eigengaps of the Hessian H.

# C. Experimental Settings

Computational environment. The image classification experiments are conducted on a computing cluster with NVIDIA® V100/H800 GPUs and Intel® Xeon® CPUs.

# C.1. Image Classification:

# C.1.1. MODELS, DATASETS, AND OPTIMIZERS

Models: LeNet (LeCun et al., 1998), Fully Connected Networks (FCN), ResNet18 (He et al., 2016) and Vision Transformer (Dosovitskiy et al., 2021). Particularly, we used one-layer FCN, two-layer FCN, four-layer FCN, which have 100 neurons for each hidden layer and use ReLu activations.

Datasets: MNIST (LeCun, 1998), Fashion-MNIST (Xiao et al., 2017), CIFAR-10/100 (Krizhevsky & Hinton, 2009), and non-image Avila (De Stefano et al., 2018).

Optimizers: SGD, Vanilla SGD, Adam (Kingma & Ba, 2015), AMSGrad (Reddi et al., 2019), AdaBound (Luo et al., 2019), Yogi (Zaheer et al., 2018), RAdam (Liu et al., 2019), Adai (Xie et al., 2022), PNM (Xie et al., 2021c), Lookahead (Zhang et al., 2019), and DiffGrad (Dubey et al., 2019).

# C.1.2. IMAGE CLASSIFICATION ON MNIST AND FASHION-MNIST

Data Preprocessing For MNIST and Fashion-MNIST: We perform the common per-pixel zero-mean unit-variance normalization.

Hyperparameter Settings: We select the optimal learning rate for each experiment from $\{0.0001, 0.001, 0.01, 0.1, 1, 10\}$ for SGD and use the default learning rate for adaptive gradient methods. In the experiments on MNIST and Fashion-MNIST: $\eta = 0.1$ for SGD, Vanilla SGD, Adai, PNM, and Lookahead; $\eta = 0.1$ for Vanilla SGD; $\eta = 0.001$ for Adam, AMSGrad, AdaBound, Yogi, RAdam, and DiffGrad.

We train neural networks for 50 epochs on MNIST and 200 epochs on Fashion-MNIST. For the learning rate schedule, the learning rate is divided by 10 at the epoch of 40% and 80%. The batch size is set to 128 for MNIST and Fashion-MNIST, unless we specify it otherwise.

The strength of weight decay defaults to $\lambda = 0.0005$ as the baseline for all optimizers unless we specify it otherwise.

We set the momentum hyperparameter $\beta_{1}=0.9$ for SGD and adaptive gradient methods which involve in Momentum. As for other optimizer hyperparameters, we apply the default settings directly.

# C.1.3. IMAGE CLASSIFICATION ON CIFAR-10 AND CIFAR-100

Data Preprocessing For CIFAR-10 and CIFAR-100: We perform the common per-pixel zero-mean unit-variance normalization, horizontal random flip, and $32 \times 32$ random crops after padding with 4 pixels on each side.

Hyperparameter Settings: We select the optimal learning rate for each experiment from $\{0.0001, 0.001, 0.01, 0.1, 1, 10\}$ for SGD and use the default learning rate for adaptive gradient methods. In the experiments on CIFAF-10 and CIFAR-100: $\eta = 1$ for Vanilla SGD, Adai, and PNM; $\eta = 0.1$ for SGD (with Momentum) and Lookahead; $\eta = 0.001$ for Adam, AMSGrad, AdaBound, Yogi, RAdam, and DiffGrad. For the learning rate schedule, the learning rate is divided by 10 at the epoch of $\{80, 160\}$ for CIFAR-10 and $\{100, 150\}$ for CIFAR-100, respectively. The batch size is set to 128 for both CIFAR-10 and CIFAR-100, unless we specify it otherwise.

The strength of weight decay is default to $\lambda = 0.0005$ as the baseline for all optimizers unless we specify it otherwise. Xie et al. (2023b) found that popular optimizers with $\lambda = 0.0005$ often yields test results than $\lambda = 0.0001$ for training CNNs on CIFAR-10 and CIFAR-100.

We set the momentum hyperparameter $\beta_{1}=0.9$ for SGD with Momentum. As for other optimizer hyperparameters, we apply the default hyperparameter settings directly.

# C.1.4. LEARNING CNNs WITH NOISY LABELS

We trained LeNet via SGD (with Momentum) on corrupted MNIST with various (asymmetric) label noise. We followed the setting of Han et al. (2018) for generating noisy labels for MNIST. The symmetric label noise is generated by flipping every label to other labels with uniform flip rates $\{40\%, 80\%\}$ . In this paper, when we talk about label noise, we mean symmetric label noise.

We also randomly shuffle the labels of MNIST to produce MNIST with random labels, which has little knowledge behind the pairs of instances and labels.

# C.1.5. IMAGE CLASSIFICATION ON VISION TRANSFORMER

Data Preprocessing For CIFAR-100: We perform resizing of CIFAR-100 images to $224 \times 224$ pixels for compatibility during training. We then perform horizontal random flip and the common per-pixel zero-mean unit-variance normalization.

Hyperparameter Settings: We fine-tuned the pretrained Vision Transformer (ViT-Base) on the CIFAR-100 dataset, following the experimental setup of Dosovitskiy et al. (2021). We trained the pretrained ViT with a total batch size of 4096 for 200 steps using a learning rate of 1e-4 with linear decay. We employed the Adam optimizer with betas set to (0.9, 0.999) and applied a weight decay strength of 0.1.

Hessian Spectra Computation: We utilized the Stochastic Lanczos Quadrature (SLQ) algorithm implementation from Yao et al. (2020) for computing the Hessian eigenvalues for ViT, incorporating modifications for improved computational and memory efficiency. To further reduce the computational cost of SLQ, we sampled 1000 samples in each run and approximate the model's Hessian on the target dataset. For all Hessian experiments on ViT, 3,000 eigenvalues were computed across three repeated SLQ runs to mitigate bias, with only the top 1,000 eigenvalues displayed and used for evaluation, consistent with the experiment setup for LLMs.

# C.2. Text Generation:

# C.2.1. MODELS, DATASETS AND OPTIMIZERS

Models: GPT-2 (Radford et al., 2019), Tinyllama (Zhang et al., 2024a). We use the code base of NanoGPT (Karpathy, 2022) for reproducing all GPT-2 models with pretraining and fine-tuning experiments.

Datasets: OpenWebText (Gokaslan et al., 2019), Shakespeare (Karpathy, 2015) (processed at the character level), and MathQA (Amini et al., 2019).

Optimizer: AdamW (Loshchilov, 2017) is used for training and fine-tuning of all language models utilized in this study.

# C.2.2. IMPLEMENTATION DETAILS ON HESSIAN SPECTRA COMPUTATION

To enable efficient computation of the Hessian spectra for LLMs, we applied the Stochastic Lanczos Quadrature algorithm implementation provided by Yao et al. (2020) with alternations for computational and memory speedups. As computation cost of SLQ is unaffordable on large-scale text dataset as discussed in Zhang et al. (2024b), we applied the same batch sampling trick and used a fixed batch size of 256 in all of our text generation experiments to approximate model's Hessian evaluated on the target dataset. For all experiments conducted on LLMs, 3,000 eigenvalues were computed across 3 repeated SLQ runs to minimize bias; Only the top 1,000 eigenvalues were displayed and used for evaluation in all Figures.

# C.2.3. TRAINING CONFIGURATIONS

\- GPT2-nano trained on Shakespeare. We utilized a smaller ‘baby GPT’ model with 6 layers, 6 attention heads, an embedding size of 384, and 11M parameters provided in NanoGPT. We used the AdamW optimizer with a fixed learning rate = $6 \times 10^{-4}$ . We used a batch size = 65,536 tokens and weight decay = 0.1 for a total of 1,000 steps.

Table 3. The Kolmogorov-Smirnov statistics of various optimizers for training LeNet on CIFAR-10. 

<table><tr><td>Training</td><td> $d_{\text{ks}}$ </td><td> $d_c$ </td><td>Power-Law</td><td> $\hat{\beta} \pm \sigma$ </td><td> $\hat{s}$ </td></tr><tr><td>Random</td><td>0.0663</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>SGD</td><td>0.0279</td><td>0.0430</td><td>Yes</td><td>1.968 ± 0.031</td><td>1.033</td></tr><tr><td>Vanilla SGD</td><td>0.0276</td><td>0.0430</td><td>Yes</td><td>1.935 ± 0.030</td><td>1.069</td></tr><tr><td>Adam</td><td>0.0269</td><td>0.0430</td><td>Yes</td><td>1.806 ± 0.025</td><td>1.241</td></tr><tr><td>AMSGrad</td><td>0.0232</td><td>0.0430</td><td>Yes</td><td>1.786 ± 0.025</td><td>1.271</td></tr><tr><td>AdaBound</td><td>0.0297</td><td>0.0430</td><td>Yes</td><td>1.901 ± 0.028</td><td>1.110</td></tr><tr><td>Yogi</td><td>0.0184</td><td>0.0430</td><td>Yes</td><td>1.806 ± 0.025</td><td>1.241</td></tr><tr><td>RAdam</td><td>0.0163</td><td>0.0430</td><td>Yes</td><td>1.733 ± 0.023</td><td>1.363</td></tr><tr><td>Adai</td><td>0.0310</td><td>0.0430</td><td>Yes</td><td>1.918 ± 0.029</td><td>1.090</td></tr><tr><td>PNM</td><td>0.0347</td><td>0.0430</td><td>Yes</td><td>1.911 ± 0.029</td><td>1.098</td></tr><tr><td>Lookahead</td><td>0.0358</td><td>0.0430</td><td>Yes</td><td>1.964 ± 0.030</td><td>1.037</td></tr><tr><td>DiffGrad</td><td>0.0303</td><td>0.0430</td><td>Yes</td><td>1.803 ± 0.024</td><td>1.236</td></tr></table>

Table 4. The Kolmogorov-Smirnov statistics of the Hessian spectra for various batch sizes. 

<table><tr><td>Dataset</td><td>Model</td><td>Training</td><td>Batch Size</td><td> $d_{\text{ks}}$ </td><td> $d_{\text{c}}$ </td><td>Power-Law</td><td> $\hat{\beta} \pm \sigma$ </td><td> $\hat{s}$ </td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>B=128</td><td>0.00900</td><td>0.0430</td><td>Yes</td><td>1.991±0.031</td><td>1.009</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>B=512</td><td>0.00787</td><td>0.0430</td><td>Yes</td><td>1.894±0.028</td><td>1.119</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>B=640</td><td>0.0125</td><td>0.0430</td><td>Yes</td><td>1.838±0.027</td><td>1.194</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>B=768</td><td>0.278</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>B=1024</td><td>0.129</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>B=16384</td><td>0.249</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>B=32768</td><td>0.201</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>B=50000</td><td>0.139</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>B=60000</td><td>0.0936</td><td>0.0430</td><td>No</td><td></td><td></td></tr></table>

- GPT2-{small, medium, large} pretrained on OpenWebText. We used the AdamW optimizer with a learning rate $= 6 \times 10^{-4}$ , incorporating 2,000 warmup steps and a decay schedule down to $6 \times 10^{-5}$ , following the experiment setup in NanoGPT. We used a batch size $= 65,536$ tokens and weight decay $= 0.1$ for a total of 500,000 steps.   
- GPT2-{small, medium, large} fine-tuned on OpenwebText / Shakespeare. We used the AdamW optimizer with a learning rate $= 1 \times 10^{-5}$ , incorporating 1,000 warmup steps and a decay schedule down to $1 \times 10^{-6}$ . We used a batch size $= 32,768$ tokens and weight decay $= 0.1$ for a total of 5,000 steps.   
- TinyLlama fine-tuned on MathQA. We fine-tuned the TinyLlama model with LoRA adapters, with rank = 16, alpha = 32, dropout = 0.1 as specified in the original LoRA experiment (Hu et al., 2021). We used the AdamW optimizer with a learning rate = $1 \times 10^{-4}$ , incorporating 500 warmup steps and a decay schedule down to $1 \times 10^{-5}$ . We used a batch size = 32 and weight decay = 0.1 for 1,000 steps.

# D. Supplementary Experiment Results of Convolutional Neural Networks

Three phases in large-batch training. In this section, we will further discuss the three phase changes in large-batch image classification training. We followed the same experiment setup in Table D as stated in Appendix C.1.2. First, in Phase I ( $B \leq 640$ ), moderately large-batch (B = 512) training indeed finds sharper minima than small-batch (B = 128) training, while the power-law spectrum still holds well. Power laws may guarantee that the top eigenvalues of large-batch trained networks are all larger than the corresponding eigenvalues of small-batch trained networks. The main challenge of large-batch training in Phase I is consistent with the common belief that large-batch training suffers from sharp minima and, thus, leads to bad generalization (Hoffer et al., 2017). The minima sharpness measured by $\hat{s}$ increases with the batch size.

Second, in Phase II (768 ≤ B ≤ 50000), the spectrum of large-batch (B = 1024) trained networks does not exhibit power laws but is visually similar to the spectrum of underparameterized models in Figure 5. In Phase II, large-batch trained overparameterized models behave like underparameterized models from a spectral perspective, and, thus, can lead to bad

generalization. The phase transition from Phase I to Phase II occurs in a narrow range of 640 < B < 768, which is visually observable in Figure 7a and statistically observable in Table D.

Third, in Phase III ( $B \sim 60000$ ), extremely large-batch training (B = 60000) cannot optimize the training loss well or find the Hessian spectra similarly to random initialized neural networks. Phase III indicates that, sometimes, bad convergence rather than sharp minima can become the main performance bottleneck in large-batch training when the batch size is too large.

![](images/608c9fe75840098c49b1f08d1126d844e84fbbbe0a9fe24ca0f8d9007819bfab.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Clean Labels | Label Noise 40% | Label Noise 80% |
| --------------- | ------------ | --------------- | --------------- |
| 10^0            | ~10^0        | ~10^0           | ~10^-1          |
| 10^1            | ~10^-1       | ~10^-1          | ~10^-2          |
| 10^2            | ~10^-2       | ~10^-2          | ~10^-3          |
| 10^3            | ~10^-3       | ~10^-3          | ~10^-4          |
</details>

![](images/088167d47ac76fbae67067155f60b7d81d850cc7e22811c94d31ceac68c1dd50.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Clean Labels | Label Noise 40% | Label Noise 80% |
| --------------- | ------------ | --------------- | --------------- |
| 10^0            | ~1.5         | ~1.5            | ~1.5            |
| 10^1            | ~0.5         | ~1.0            | ~1.0            |
| 10^2            | ~0.1         | ~0.5            | ~0.5            |
| 10^3            | ~0.01        | ~0.1            | ~0.1            |
| 10^4            | ~0.001       | ~0.01           | ~0.01           |
</details>

Figure 15. The spectrum in the presence of noisy labels. Top: MNIST Trainset. Bottom: MNIST Testset.

Our Hessian spectra analysis discovery differs from traditional beliefs that different phases exist in training. In phase 2, the Hessian eigenvalue increases and breaks the power-law structure, while the model's performance is much superior compared to untrained neural networks. We may leave a deeper investigation for future work.

![](images/14dc1f277c9dc305e75eb6dcd089796414c5eea4f4dcd56d0dc704960eb96911.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | lenet_B=128-train-1000 | lenet_B=512-train-1000 | lenet_B=640-train-1000 | lenet_B=768-train-1000 | lenet_B=1024-train-1000 | lenet_B=16384-train-1000 |
| --------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- | ----------------------- | ------------------------ |
| 1               | ~10^2                  | ~10^2                  | ~10^2                  | ~10^2                  | ~10^2                   | ~10^2                    |
| 10              | ~10^1                  | ~10^1                  | ~10^1                  | ~10^1                  | ~10^1                   | ~10^1                    |
| 100             | ~10^0                  | ~10^0                  | ~10^0                  | ~10^0                  | ~10^0                   | ~10^0                    |
| 1000            | ~10^-1                 | ~10^-1                 | ~10^-1                 | ~10^-1                 | ~10^-1                  | ~10^-2                   |
</details>

Table 5. The Kolmogorov-Smirnov statistics of the Hessian spectra for various batch sizes under the same number of training iterations. 

<table><tr><td>Dataset</td><td>Model</td><td>Training</td><td>Batch Size</td><td> $d_{\text{ks}}$ </td><td> $d_{\text{c}}$ </td><td>Power-Law</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td> $B = 128$ </td><td>0.0527</td><td>0.0430</td><td>No</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td> $B = 512$ </td><td>0.0934</td><td>0.0430</td><td>No</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td> $B = 640$ </td><td>0.1138</td><td>0.0430</td><td>No</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td> $B = 768$ </td><td>0.1242</td><td>0.0430</td><td>No</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td> $B = 1024$ </td><td>0.0999</td><td>0.0430</td><td>No</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td> $B = 16384$ </td><td>0.0271</td><td>0.0430</td><td>Yes</td></tr></table>

Figure 16. The Hessian spectrum for various batch sizes under the same number of training iterations.

Clean and Random Labels. We presented the spectrum of learning with clean labels and random labels in Figure 17. The number of top outliers obviously increases, because random labels make the dataset more complex. However, even if the pairs of instances and labels have little knowledge, we still observe the power-law spectrum after the dozens of top outliers. This may suggest that, even if the labels are random, neural networks can still learn useful knowledge from the instances only.

Overfitting and Noisy Labels. As DNNs overfit noisy labels easily, previous papers choose learning with noisy labels as an important setting for evaluating overfitting and generalization (Han et al., 2020; Xie et al., 2021a; He et al., 2022). Figure 15 shows that overfitting label noise makes the Hessian spectra less power-law on both the corrupted training dataset and the clean test dataset. In contrast, in the absence of noisy labels, the power-law spectra exist on both the training dataset and the test dataset.

Transferability. In transfer learning, people believe that a model pretrained on one dataset may learn useful representations

![](images/a5815b7fb79ec58143cdc9157206b78c48d22aa55c4463de3fc1db9c6a20fe06.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Training | Test |
| --------------- | -------- | ---- |
| 10^0            | 0.1      | 0.1  |
| 10^1            | 0.05     | 0.05 |
| 10^2            | 0.01     | 0.01 |
| 10^3            | 0.001    | 0.001 |
| 10^4            | 0.0001   | 0.0001 |
</details>

![](images/82bd7514a1e100be5a4c2b9199017e20dc88d67f146fe53f37ad57d891f0aaef.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Gaussian Data | MNIST with Random Labels |
| --------------- | ------------- | ------------------------ |
| 1               | 0.1           | 0.1                      |
| 10              | 0.05          | 0.05                     |
| 100             | 0.005         | 0.005                    |
| 1000            | 0.0005        | 0.0005                   |
| 10000           | 0.0001        | 0.0001                   |
</details>

Figure 17. (a) The spectrum of LeNet on (training and test) MNIST with randomly shuffled labels. (b) The spectrum of LeNet on Gaussian data with randomly shuffled labels is highly similar to that MNIST with randomly shuffled labels.   
![](images/25f34f5469920a728b20ad81a89664cb52afad405009bd9371957417f024888e.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Pretrained on CIFAR-100 | Pretrained on CIFAR-10 |
| --------------- | ------------------------ | ----------------------- |
| 1               | 100                      | 50                      |
| 10              | 20                       | 10                      |
| 100             | 5                        | 1                       |
| 1000            | 1                        | 0.1                     |
| 10000           | 0.1                      | 0.01                    |
</details>

Figure 18. Transferability of the power-law Hessian structure. A LeNet which is pretrained on CIFAR-100 may still exhibit power-law Hessian spectra when evaluated on CIFAR-10.

for relevant datasets or downstream tasks. When we pretrain a model on CIFAR-100 and evaluate its Hessian spectrum on CIFAR-10, we surprisingly discover that the power-law Hessian structure successfully transfers. This may measures the usefulness of learned representations.

Power Iteration vs. Lanczos Algorithm. We compared the spectra computed via Power Iteration Algorithm and Lanczos Algorithm in Figure 19. It shows the top eigenvalues estimated via Power Iteration Algorithm are highly consistent with the top eigenvalues via the Lanczos Algorithm. It also demonstrates that the power-law spectrum is caused by the properties of deep learning rather than the stochasticity of Lanczos Algorithm.

We presented the power-law eigengaps on Fashion-MNIST in Figure 21. It shows that the power-law eigengaps on Fashion-MNIST are highly consistent with the power-law eigengaps on MNIST.

We presented the power-law spectrum of the covariance matrix of stochastic gradient noise of FCN on MNIST in Figure 27. As the inverses of the power-law variables are power-law, the covariance spectrum shows heavy-tail properties. It demonstrates that the heavy-tail property belongs to deep neural networks rather than SGD itself.

Avila Dataset We presented the power-law spectrum of two-layer FCN on Avila Dataset in Figure 20. It shows that the power-law spectrum of neural networks may also generally exist in non-convolution neural networks trained on a non-image dataset. Particularly, we note that the Avila Dataset has only ten attributes, including intercolumnar distance, upper margin, lower margin, exploitation, row number, modular ratio, interlinear spacing, weight, peak number, and modular ratio/interlinear spacing. These attributes are essentially different from the pixels in image datasets.

Supplementary Empirical Results on CIFAR-10. The spectra of LeNet on CIFAR-10 trained via various optimizers

![](images/7baa09e91ddc96ac480d9e3ea125dfed61aaeab46dea5d0246f541a2c1b32979.jpg)

<details>
<summary>scatter</summary>

| Eigenvalue Rank | Eigenvalue | Method             |
| --------------- | ---------- | ------------------ |
| 1               | 2.0        | Power Iteration    |
| 2               | 1.5        | Power Iteration    |
| 3               | 1.2        | Power Iteration    |
| 4               | 1.0        | Power Iteration    |
| 5               | 0.8        | Power Iteration    |
| 6               | 0.6        | Power Iteration    |
| 7               | 0.5        | Power Iteration    |
| 8               | 0.4        | Power Iteration    |
| 9               | 0.3        | Power Iteration    |
| 10              | 0.2        | Power Iteration    |
| 20              | 0.1        | Power Iteration    |
| 30              | 0.05       | Power Iteration    |
| 40              | 0.03       | Power Iteration    |
| 50              | 0.02       | Power Iteration    |
| 60              | 0.01       | Power Iteration    |
| 70              | 0.005      | Power Iteration    |
| 80              | 0.003      | Power Iteration    |
| 90              | 0.002      | Power Iteration    |
| 100             | 0.001      | Power Iteration    |
</details>

Figure 19. The spectrum via Power Iteration Algorithm is highly consistent with the spectrum via Lanczos Algorithm. It also shows that the power-law spectrum is caused by the properties of deep learning rather than the stochasticity of Lanczos Algorithm.

![](images/130f275ad3a3a5307477b60468e9db715064fb9bed0bc76b5d90d649e8006bdd.jpg)

<details>
<summary>scatter</summary>

| Eigenvalue Rank | Eigenvalue |
| --------------- | ---------- |
| 10^0            | ~5         |
| 10^1            | ~0.1       |
| 10^2            | ~0.01      |
</details>

Figure 20. The spectrum of FCN on Avila Dataset. It shows that the power-law spectrum of neural networks may also exist in non-image datasets.

![](images/b0e714594eb1096c8aa0c06fb333ddf5769953e739a85526fd48e467d0da2cfd.jpg)

<details>
<summary>scatter</summary>

| Rank | Eigenvalue or Eigengap | Type        |
|------|------------------------|-------------|
| 1    | ~100                   | Eigenvalues |
| 10   | ~1                     | Eigenvalues |
| 100  | ~0.01                  | Eigenvalues |
| 1000 | ~0.0001                | Eigenvalues |
| 1    | ~1                     | Eigengaps   |
| 10   | ~0.1                   | Eigengaps   |
| 100  | ~0.001                 | Eigengaps   |
| 1000 | ~0.00001               | Eigengaps   |
</details>

(a) Eigenvalue Rank

![](images/ca4c6331a71f6ad8090f66c3f0d8cb3d1710f2d69b14d778631fbfe1d2a68bbc.jpg)

<details>
<summary>line</summary>

| Rank | Eigengaps | Eigenvalues |
| ---- | --------- | ----------- |
| 1    | 100       | 100         |
| 10   | 1         | 1           |
| 100  | 0.01      | 0.1         |
| 1000 | 0.0001    | 0.01        |
| 10000| 0.000001  | 0.001       |
</details>

(b) Eigengap Rank   
Figure 21. The power-law Hessian eigengaps in deep learning. Model: LeNet. Datset: Fashion-MNIST. Subfigure (a) displayed the eigengaps by original rank indices sorted by eigenvalues. Subfigure (b) displayed the eigengaps by rank indices re-sorted by eigengaps.

![](images/94c042bace4a64d98cc833fb653695406e45f40ac0155d814ea93fbcfc2fa9dd.jpg)

<details>
<summary>scatter</summary>

| Algorithm     | Eigenvalue Rank | Eigenvalue |
| ------------- | --------------- | ---------- |
| SGD           | 10^0            | 10^3       |
| Adam          | 10^0            | 10^2       |
| AMSGrad       | 10^0            | 10^2       |
| AdaBound      | 10^0            | 10^2       |
| RAdam         | 10^0            | 10^2       |
| Yogi          | 10^0            | 10^2       |
| Adai          | 10^0            | 10^2       |
| PNM           | 10^0            | 10^2       |
| Lookahead     | 10^0            | 10^2       |
| DiffGrad      | 10^0            | 10^2       |
| Vanilla SGD   | 10^0            | 10^2       |
</details>

![](images/21c10598f2010a040c420c0c769f8c48d75bc3f983b2d839fa92536da2b8585d.jpg)

<details>
<summary>scatter</summary>

| Method      | s    | Test Error |
| ----------- | ---- | ---------- |
| SGD         | 1.05 | 0.245      |
| SGD         | 1.10 | 0.248      |
| SGD         | 1.25 | 0.265      |
| SGD         | 1.30 | 0.270      |
| Adam        | 1.05 | 0.242      |
| Adam        | 1.25 | 0.248      |
| AMSGrad     | 1.25 | 0.268      |
| AMSGrad     | 1.30 | 0.270      |
| AdaBound    | 1.10 | 0.245      |
| Yogi        | 1.25 | 0.265      |
| RAdam       | 1.25 | 0.272      |
| Adai        | 1.10 | 0.238      |
| PNM         | 1.10 | 0.260      |
| Lookahead   | 1.05 | 0.240      |
| DiffGrad    | 1.25 | 0.272      |
| DiffGrad    | 1.30 | 0.260      |
| Vanilla SGD | 1.10 | 0.245      |
</details>

Figure 22. The power-law spectra hold across optimizers. Moreover, the slope magnitude $\hat{s}$ is an indicator of minima sharpness and a predictor of test performance. Model:LeNet. Dataset: CIFAR-10.   
![](images/c4396fdccffbd2295064868dff1f9fe386bbd1faa2405e0d6e02ea10211c530c.jpg)

<details>
<summary>scatter</summary>

| s    | λ₁   |
| ---- | ---- |
| 1.0  | 50   |
| 1.05 | 75   |
| 1.1  | 80   |
| 1.25 | 450  |
| 1.3  | 375  |
| 1.4  | 700  |
</details>

![](images/68345dbe37f848139ae9e89e816475cb94b706ea740499f204b49ea765cfb5cb.jpg)

<details>
<summary>scatter</summary>

| s    | Tr(H) |
| ---- | ----- |
| 1.05 | 800   |
| 1.07 | 1000  |
| 1.10 | 1200  |
| 1.12 | 1400  |
| 1.15 | 1600  |
| 1.20 | 2000  |
| 1.25 | 2200  |
| 1.30 | 2500  |
| 1.35 | 2800  |
</details>

Figure 23. The slope magnitude $\hat{s}$ closely correlates with the largest Hessian eigenvalue and the Hessian trace. Model:LeNet. Dataset: CIFAR-10.

are showed in Figure 22. Figures 4 and 23 shows that the slope magnitude $\hat{s}$ closely correlates with the largest Hessian eigenvalue and the Hessian trace.

We presented the power-law spectra of ResNet18 on CIFAR-10 in Figure 24. It shows that the power-law spectra hold for ResNet, a representative of the modern neural network architectures, as well as simple CNNs/FCNs. Due to the GPU memory limit, we may only display the top 50 eigenvalues for ResNet18. However, the KS test still supports accepting the power-law hypothesis.

We report the spectra of large-batch trained ResNet18 on CIFAR-10 in Figure 25. It indicates that the phase transition behaviors of the spectra with respect to batch size generally exist. However, it seems that Phase II and Phase III merge into one phase for ResNet18 on CIFAR-10.

Figure 26 shows that the small width of neural networks may also break the power-law spectrum like small depth. This also supports that overparameterization or large model capacity is necessary for the power-law spectrum.

Rethinking the heavy-tail phenomenon in SGD. The heavy-tail property of SGD has been a hot and arguable topic recently (Simsekli et al., 2019; Panigrahi et al., 2019; Gurbuzbalaban et al., 2021; Hodgkinson & Mahoney, 2021; Xie et al., 2021b; Li et al., 2021). Note that the power-law distribution is one of the most common heavy-tail distributions in the real world. We argue that the arguable heavy-tail property of SGD may depend on the power-law Hessian spectrum rather than SGD itself, as gradient noise covariance critically depends on the Hessian. We present the power-law spectra of gradient noise covariance in Figure 27.

We specifically study the Hessian spectrum of Linear Neural Networks (LNNs) which has no ReLu and BatchNorm. Figure

![](images/103ce069d96e86a2598505b00edf6713267c7b2f20009efc32922fa07fae061e.jpg)

<details>
<summary>scatter</summary>

| Eigenvalue Rank | SGD    | Vanilla SGD | Random |
| --------------- | ------ | ----------- | ------ |
| 10^0            | 10^1   | 10^1        | 10^-1  |
| 10^1            | 10^0   | 10^0        | 10^-1  |
| 10^2            | 10^-1  | 10^-1       | 10^-1  |
</details>

Figure 24. The power-law spectra of ResNet18 on CIFAR-10. It shows that the power-law spectrum of neural networks may also exist in modern neural network architectures (ResNet) as well as simple CNNs/FCNs.

![](images/7d435847123c797e3b7024acad5afd937e12a13376495a116c50b1ad337638a1.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | B=128 | B=1024 | B=1152 | B=1280 | B=2048 | B=4096 | B=16384 | Random |
| --------------- | ----- | ------ | ------ | ------ | ------ | ------ | ------- | ------ |
| 10^0            | ~10   | ~10    | ~10    | ~10    | ~10    | ~10    | ~10     | ~10    |
| 10^1            | ~0.1  | ~0.1   | ~0.1   | ~0.1   | ~0.1   | ~0.1   | ~0.1    | ~0.1   |
| 10^2            | ~0.01 | ~0.01  | ~0.01  | ~0.01  | ~0.01  | ~0.01  | ~0.01   | ~0.01  |
</details>

Figure 25. Batch size matters to the spectrum. Model: ResNet-18. Dataset: CIFAR-10. The sharp phase transition occurs in 1152 < B < 1280.

![](images/7691f4ac6a41639b7ea1ccf808d6183f1dc1f0d354b51bdc180a4a61cae6faf9.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Width=10 | Width=20 | Width=30 | Width=50 | Width=70 | Width=100 |
| --------------- | -------- | -------- | -------- | -------- | -------- | --------- |
| 10^0            | ~10      | ~10      | ~10      | ~10      | ~10      | ~10       |
| 10^1            | ~1       | ~1       | ~1       | ~1       | ~1       | ~1        |
| 10^2            | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1      |
| 10^3            | ~0.01    | ~0.01    | ~0.01    | ~0.01    | ~0.01    | ~0.01     |
| 10^4            | ~0.001   | ~0.001   | ~0.001   | ~0.001   | ~0.001   | ~0.001    |
</details>

Figure 26. The spectra are not power-law for neural networks with a small width( $\sim$ 10), but gradually become more power-law (more straight in the log-log plot) as the width increases. This may also suggest that the power-law spectrum depends on model capacity. Model: Two-layer FCN. Dataset: MNIST.

![](images/d2484be313929f52cca09feea8674b75f6c1118a987116ad809f8360bf66b6c1.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | B=100       | B=200       | B=300       |
| --------------- | ----------- | ----------- | ----------- |
| 1               | 100         | 10          | 1           |
| 10              | 1           | 0.1         | 0.01        |
| 100             | 0.01        | 0.001       | 0.0001      |
| 1000            | 0.001       | 0.0001      | 0.00001     |
</details>

Figure 27. The power-law spectrum of gradient noise covariance exists in deep learning for various batch sizes. Model: Fully Connected Network(FCN). Dataset: MNIST.

![](images/1d1db33663893bc5e3efd5d1a9dc8a6494b662345683772fdf1c16b20c227f2a.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | LNN       | LNN+BatchNorm | LNN+ReLU |
| --------------- | --------- | ------------- | -------- |
| 1               | 100.0     | 10.0          | 1.0      |
| 10              | 100.0     | 1.0           | 0.1      |
| 100             | 100.0     | 0.1           | 0.01     |
| 1000            | 10.0      | 0.01          | 0.001    |
| 10000           | 0.01      | 0.001         | 0.0001   |
</details>

Figure 28. The spectrum of LNN with or without BatchNorm and ReLu.

![](images/bce9547064777062d6ea5c108b2c8f405a2d6d45a6aafff6fcc7fb50f20b88ff.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Random | SGD-Epoch 1 | SGD-Epoch 3 | SGD-Epoch 10 | SGD-Epoch 30 | SGD-Epoch 50 |
| --------------- | ------ | ----------- | ----------- | ------------ | ------------ | ------------ |
| 10^0            | ~10^0  | ~10^0       | ~10^0       | ~10^0        | ~10^0        | ~10^0        |
| 10^1            | ~10^-1 | ~10^-1      | ~10^-1      | ~10^-1       | ~10^-1       | ~10^-1       |
| 10^2            | ~10^-2 | ~10^-2      | ~10^-2      | ~10^-2       | ~10^-2       | ~10^-2       |
| 10^3            | ~10^-3 | ~10^-3      | ~10^-3      | ~10^-3       | ~10^-3       | ~10^-3       |
</details>

Figure 29. The spectrum of LeNets trained with various epochs.

29 shows that the spectrum of LNNs is not power-law but more like spectra of underparameterized models. It may indicate that nonlinearity and BatchNorm both help improve model capacity. LNNs equipped with ReLu or BatchNorm immediately recover the power-law spectra again.

# E. Supplementary Experiment Results of Large Language Models

# E.1. Power-law Hessian Eigengaps in LLM

In this subsection, we report the power-law spectrum in Hessian eigengaps of fine-tuned GPT2-small. As previous results of LeNet (Figure 2, 21), GPT2-small as a language model, demonstrated that learning dynamics similarly takes place in a low-dimensional learning subspace as proved in Appendix B.

![](images/a794cbdc6f85887cce3dc8f6ef28d415df6ff88708c836d472197589210f7a7f.jpg)

<details>
<summary>scatter</summary>

| Eigenvalue Rank | Eigenvalue or Eigengap |
| --------------- | ---------------------- |
| 10^0            | ~10^2                  |
| 10^1            | ~10^1                  |
| 10^2            | ~10^-1                 |
| 10^3            | ~10^-3                 |
</details>

![](images/8992724cbe9bf398e24d3c2a207745ae891e43fbe3baf5514b952e382a677a92.jpg)

<details>
<summary>line</summary>

| Eigengap Rank | Eigenvalues | Eigengap |
| ------------- | ----------- | -------- |
| 1             | 100         | 1        |
| 10            | 10          | 0.1      |
| 100           | 1           | 0.01     |
| 1000          | 0.1         | 0.001    |
</details>

Figure 30. The power-law Hessian eigengaps. Model: GPT2-small. Dataset: Shakespeare. Subfigure (a) displayed the eigengaps by original rank indices sorted by eigenvalues. Subfigure (b) displayed the eigengaps by rank indices re-sorted by eigengaps.

# E.2. Layer-wise Power-law Spectra

Due to computational constraints, we reported the Hessian spectra only using the parameters of the last layer in previous sections. In this section, we will present and discuss the power-law spectrum at different layers of the fine-tuned GPT2-small model. Figure 31 presented the power-law spectrum at 2, 5, 8, $11^{th}$ layers of GPT2-small checkpoints fine-tuned on the Shakespeare dataset. The phenomenon is clear: the power-law spectrum forms early in training, particularly in the earlier layers, and the eigenvalue magnitudes decrease across layers.

# E.3. Fine-tuning with Low-dimensional Adaptations (LoRA)

In this section, we report the Hessian eigenvalues of the low dimensional adapters (LoRA (Hu et al., 2021)) for efficient fine-tuning of the pretrained TinyLlama-1B model on the MathQA dataset. The top 1000 eigenvalues are computed for

![](images/834c6449792c1c76d613d608ca7c08db81df6c0ce758df4f1fbd00ce7b229284.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | pretrain | finetune-100 | finetune-200 | finetune-500 | finetune-1000 | finetune-2000 | finetune-5000 |
| --------------- | -------- | ------------ | ------------ | ------------ | ------------- | ------------- | ------------- |
| 10^0            | ~10^2    | ~10^2        | ~10^2        | ~10^2        | ~10^2         | ~10^2         | ~10^2         |
| 10^1            | ~10^1    | ~10^1        | ~10^1        | ~10^1        | ~10^1         | ~10^1         | ~10^1         |
| 10^2            | ~10^0    | ~10^0        | ~10^0        | ~10^0        | ~10^0         | ~10^0         | ~10^0         |
| 10^3            | ~10^-1   | ~10^-1       | ~10^-1       | ~10^-1       | ~10^-1        | ~10^-1        | ~10^-1        |
</details>

(a) $2^{th}$ Layer

![](images/32dc47215c94ce2b6e37af716a6eaae789c1551ddf16843e1b38554166da579e.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | pretrain | finetune-100 | finetune-200 | finetune-500 | finetune-1000 | finetune-2000 | finetune-5000 |
| --------------- | -------- | ------------ | ------------ | ------------ | ------------- | ------------- | ------------- |
| 1               | 100      | 100          | 100          | 100          | 100           | 100           | 100           |
| 10              | 10       | 10           | 10           | 10           | 10            | 10            | 10            |
| 100             | 1        | 1            | 1            | 1            | 1             | 1             | 1             |
| 1000            | 0.1      | 0.1          | 0.1          | 0.1          | 0.1           | 0.1           | 0.1           |
</details>

(b) $5^{th}$ Layer

![](images/3a1de0aff5f03c02f09b86584688997fd96730670449b959e9d6e558920bb6af.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | pretrain | finetune-100 | finetune-200 | finetune-500 | finetune-1000 | finetune-2000 | finetune-5000 |
| --------------- | -------- | ------------ | ------------ | ------------ | ------------- | ------------- | ------------- |
| 10^0            | ~10^2    | ~10^2        | ~10^2        | ~10^2        | ~10^2         | ~10^2         | ~10^2         |
| 10^1            | ~10^1    | ~10^1        | ~10^1        | ~10^1        | ~10^1         | ~10^1         | ~10^1         |
| 10^2            | ~10^0    | ~10^0        | ~10^0        | ~10^0        | ~10^0         | ~10^0         | ~10^0         |
| 10^3            | ~10^-1   | ~10^-1       | ~10^-1       | ~10^-1       | ~10^-1        | ~10^-1        | ~10^-1        |
</details>

(c) $8^{th}$ Layer

![](images/6dcffca71b30cc086e6a5f2d92bc87c40edd11037129fc60e59ef1bfa5d27fe9.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | pretrain | finetune-50 | finetune-100 | finetune-500 | finetune-1000 | finetune-5000 |
| --------------- | -------- | ----------- | ------------ | ------------ | ------------- | ------------- |
| 10^0            | ~10^2    | ~10^2       | ~10^2        | ~10^2        | ~10^2         | ~10^2         |
| 10^1            | ~10^1    | ~10^1       | ~10^1        | ~10^1        | ~10^1         | ~10^1         |
| 10^2            | ~10^0    | ~10^0       | ~10^0        | ~10^0        | ~10^0         | ~10^0         |
| 10^3            | ~10^-1   | ~10^-1      | ~10^-1       | ~10^-1       | ~10^-1        | ~10^-1        |
</details>

(d) $12^{th}$ Layer   
Figure 31. Power-law spectrum at different layers and checkpoints of the GPT2-small model. Dataset: Shakespeare.

the Hessian of all LoRA layers are displayed, and we may observe that a hessian Power-law structure is formed after 500 fine-tuning steps.

We then discuss how generalization of the fine-tuned LoRA adapters relate to the power-law Hessian structure. In Figure 32, the KS distance $(d_{ks})$ calculated from the Hessian spectra of the adapters, initially increases during the first 100 steps of fine-tuning before decreasing alongside the test loss. Although the largest eigenvalue $\lambda_{1}$ increases at step 50 compared to its random initialization, the Hessian trace $Tr(H)$ consistently decreases throughout fine-tuning. This behavior suggests that for pretrained models with good proficiency, the power-law spectra may diverge from expected patterns, and the Hessian trace $Tr(H)$ may be a more reliable indicator of generalization capability according to Figure 32.

![](images/bec1c455bb272e7553225823ac859d373325e678b11aaaa0623e7294768eba5d.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Eigenvalue (Red) | Eigenvalue (Blue) | Eigenvalue (Green) | Eigenvalue (Orange) | Eigenvalue (Yellow) |
| --------------- | ---------------- | ----------------- | ------------------ | ------------------- | ------------------- |
| 10^0            | ~10^3            | ~10^3             | ~10^2              | ~10^2               | ~10^2               |
| 10^1            | ~10^3            | ~10^3             | ~10^2              | ~10^2               | ~10^2               |
| 10^2            | ~10^3            | ~10^3             | ~10^2              | ~10^2               | ~10^2               |
| 10^3            | ~10^2            | ~10^2             | ~10^1              | ~10^1               | ~10^1               |
</details>

![](images/8e4d8a7fe9750b438313fd246c72a6535336e5eeac02915b9bcd036047ea41cf.jpg)

<details>
<summary>scatter</summary>

| d ks   | Test Loss |
| ------ | --------- |
| 0.03   | 0.3       |
| 0.03   | 0.4       |
| 0.05   | 3.4       |
| 0.08   | 1.9       |
| 0.06   | 0.5       |
| 0.10   | 0.9       |
</details>

![](images/35a2e678529c1be5dc221402b0f7005af50ace3b06d54e37bc88174b6b1c2051.jpg)

<details>
<summary>scatter</summary>

| λ₁    | Test Loss | Group      |
| ------ | --------- | ---------- |
| 2800   | 3.4       | lora-random|
| 4600   | 1.9       | lora-50    |
| 1800   | 0.9       | lora-100   |
| 200    | 0.5       | lora-200   |
| 200    | 0.3       | lora-1000  |
</details>

![](images/e1e5719fd92ef96ab4666cff0f66015dc72c1dfb6ff9485cedcf8d48108961ee.jpg)

<details>
<summary>scatter</summary>

| Tr(H) | Test Loss | Method     |
|-------|-----------|------------|
| 10^3  | 0.3       | lora-1000  |
| 10^4  | 0.9       | lora-100   |
| 10^4  | 0.5       | lora-200   |
| 10^4  | 0.3       | lora-50    |
| 10^6  | 3.3       | lora-random|
</details>

Figure 32. The behaviour of the Hessian spectrum of all LoRA adapters aligns similarity to the original model. LoRA adapters are randomly initialized before fine-tuning. Model: LoRA Layers (adapted on TinyLlama). Dataset: MathQA.

We further computed the Hessian spectrum of the pretrained last layer of TinyLlama-base merged with the fine-tuned LoRA Adapter in Figure 33. Due to the consistency in most parameters, there are no significant changes in behaviours of the Hessian spectrum at different checkpoints of fine-tuning, nor the minima sharpness.

![](images/8b82d920e46ca90d5d2bb3c964f85520fd32ff7edc2a70424983027e16c45d8c.jpg)

<details>
<summary>line</summary>

| Eigenvalue Rank | Eigenvalue |
| --------------- | ---------- |
| 10^0            | ~10^3      |
| 10^1            | ~10^2      |
| 10^2            | ~10^1      |
| 10^3            | ~10^0      |
</details>

![](images/062ac467314c807b29ceeb2e0db840f37bb47f1e57bebeb605e9eb46e7b9df1e.jpg)

<details>
<summary>scatter</summary>

| d_ks   | Test Loss | Group     |
|--------|-----------|-----------|
| 0.025  | 3.7       | merge-50  |
| 0.025  | 1.8       | merge-100 |
| 0.025  | 1.1       | merge-200 |
| 0.025  | 0.7       | merge-500 |
| 0.025  | 0.6       | merge-1000 |
| 0.025  | 4.0       | pretrain  |
</details>

![](images/cea30e8b4b8dd222db5930c0ca3adbe4e718a7f88babd45c226ec0b6a06e72dc.jpg)

<details>
<summary>scatter</summary>

| λ₁   | Test Loss | Method     |
| ---- | --------- | ---------- |
| 490  | 3.4       | pretrain   |
| 510  | 1.9       | merge-50   |
| 510  | 0.9       | merge-100  |
| 510  | 0.6       | merge-200  |
| 510  | 0.3       | merge-500  |
| 510  | 0.2       | merge-1000 |
</details>

![](images/6a15cbe6d14219c7c62199b1815cc204617436d402b5182255f40711835a0901.jpg)

<details>
<summary>scatter</summary>

| Model | Tr(H) | Test Loss |
| :--- | :--- | :--- |
| pretrain | 95000 | 3.4 |
| merge-50 | 95000 | 1.8 |
| merge-100 | 95000 | 0.8 |
| merge-200 | 95000 | 0.6 |
| merge-500 | 95000 | 0.5 |
| merge-1000 | 95000 | 0.3 |
</details>

Figure 33. Due to consistency in most parameters, the Hessian spectrum does not change significantly at different fine-tuning checkpoints of the merged TinyLlama with LoRA adapter. The observation demonstrated the minimal impact to the original model when tuning with Low-Rank Adaptations. Model: TinyLlama-1.1B-Chat-v1.0 (merged with LoRA adapter). Dataset: MathQA.

# F. Kolmogorov-Smirnov Goodness-of-Fit Test

In this section, we introduce how to conduct the Kolmogorov-Smirnov Goodness-of-Fit Test for the self-containedness purpose.

As we mentioned above, our work used Maximum Likelihood Estimation (MLE) (Myung, 2003; Clauset et al., 2009) for estimating the parameter $\beta$ of the fitted power-law distributions and the Kolmogorov-Smirnov Test (KS Test) (Massey Jr, 1951; Goldstein et al., 2004) for statistically testing the goodness of the fit. The KS test statistic is the KS distance $d_{ks}$ between the hypothesized (fitted) distribution and the empirical data, which measures the goodness of fit. Mathematically, the KS distance is defined as

$$
d _ {\mathrm{ks}} = \sup _ {\lambda} | F ^ {\star} (\lambda) - \hat {F} (\lambda) |, \tag {15}
$$

where $F^{\star}(\lambda)$ is the hypothesized cumulative distribution function and $\hat{F}(\lambda)$ is the empirical cumulative distribution function based on the sampled data (Goldstein et al., 2004). The estimated power exponent via MLE (Clauset et al., 2009) can be written as

$$
\hat {\beta} = 1 + K \left[ \sum_ {i = 1} ^ {K} \ln \left(\frac {\lambda_ {i}}{\lambda_ {\text { cutoff }}}\right) \right] ^ {- 1}, \tag {16}
$$

where K is the number of tested samples and we set $\lambda_{cutoff} = \lambda_{k}$ . We note that the Powerlaw library (Alstott et al., 2014) provides a convenient tool to compute the KS distance, $d_{ks}$ , and estimate the power exponent.

According to the practice of KS Test (Massey Jr, 1951), we first state the power-law hypothesis that the tested spectrum is power-law. If $d_{ks}$ is higher than the critical value $d_{c}$ at the $\alpha = 0.05$ significance level, the KS test statistically will support the power-law hypothesis (we cannot reject the power-law hypothesis). We display the critical values in Table 6.

We conducted the KS tests for all of our studied spectra. We display the KS test statistics and the estimated power exponents $\hat{\beta}$ with standard errors $\sigma$ as well as the corresponding $\hat{s}$ in Tables 7, 8, 9, and 10. In the tables, we take the base hyperparameter setting in Appendix C as the default setting. For better visualization, we color accepting the power-law hypothesis in blue and color rejecting the power-law hypothesis (and the cause) in red.

Table 6. The Table of Kolmogorov-Smirnov Test Critical Values (Significance Level), which was first reported in Massey Jr (1951). If the KS distance $d_{\mathrm{ks}}$ is lower than a critical value, such as $\frac{1.36}{\sqrt{K}}$ , we would reject the null hypothesis and accept the power-law hypothesis at the $\alpha = 0.05$ significance level. Note that $K$ is the number of tested eigenvalues. 

<table><tr><td>Sample size</td><td> $\alpha = 0.2$ </td><td> $\alpha = 0.15$ </td><td> $\alpha = 0.1$ </td><td> $\alpha = 0.05$ </td><td> $\alpha = 0.01$ </td></tr><tr><td>K &gt; 35</td><td> $\frac{1.07}{\sqrt{k}}$ </td><td> $\frac{1.14}{\sqrt{k}}$ </td><td> $\frac{1.22}{\sqrt{k}}$ </td><td> $\frac{1.36}{\sqrt{k}}$ </td><td> $\frac{1.63}{\sqrt{k}}$ </td></tr><tr><td>K = 50</td><td>0.151</td><td>0.161</td><td>0.173</td><td>0.192</td><td>0.231</td></tr><tr><td>K = 1000</td><td>0.0338</td><td>0.0360</td><td>0.0386</td><td>0.0430</td><td>0.0515</td></tr></table>

Table 7. The Kolmogorov-Smirnov statistics of LeNet on MNIST and Fashion MNIST. The estimated power exponent $\hat{\beta}$ and slope magnitude $\hat{s}$ are also displayed. 

<table><tr><td>Dataset</td><td>Model</td><td>Training</td><td>Sample size</td><td>Setting</td><td> $d_{\text{ks}}$ </td><td> $d_c$ </td><td>Power-Law</td><td> $\hat{\beta} \pm \sigma$ </td><td> $\hat{s}$ </td></tr><tr><td>MNIST</td><td>LeNet</td><td>Random</td><td>1000</td><td>-</td><td>0.0796</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>-</td><td>0.00900</td><td>0.0430</td><td>Yes</td><td>1.991 ± 0.031</td><td>1.009</td></tr><tr><td>MNIST</td><td>LeNet</td><td>Vanilla SGD</td><td>1000</td><td>-</td><td>0.0103</td><td>0.0430</td><td>Yes</td><td>1.914 ± 0.029</td><td>1.094</td></tr><tr><td>MNIST</td><td>LeNet</td><td>Adam</td><td>1000</td><td>-</td><td>0.00962</td><td>0.0430</td><td>Yes</td><td>1.873 ± 0.028</td><td>1.145</td></tr><tr><td>MNIST</td><td>LeNet</td><td>AMSGrad</td><td>1000</td><td>-</td><td>0.00987</td><td>0.0430</td><td>Yes</td><td>1.845 ± 0.027</td><td>1.184</td></tr><tr><td>MNIST</td><td>LeNet</td><td>AdaBound</td><td>1000</td><td>-</td><td>0.00889</td><td>0.0430</td><td>Yes</td><td>1.904 ± 0.029</td><td>1.106</td></tr><tr><td>MNIST</td><td>LeNet</td><td>Yogi</td><td>1000</td><td>-</td><td>0.00966</td><td>0.0430</td><td>Yes</td><td>1.834 ± 0.026</td><td>1.198</td></tr><tr><td>MNIST</td><td>LeNet</td><td>RAdam</td><td>1000</td><td>-</td><td>0.0164</td><td>0.0430</td><td>Yes</td><td>1.889 ± 0.028</td><td>1.125</td></tr><tr><td>MNIST</td><td>LeNet</td><td>Adai</td><td>1000</td><td>-</td><td>0.0101</td><td>0.0430</td><td>Yes</td><td>1.892 ± 0.028</td><td>1.122</td></tr><tr><td>MNIST</td><td>LeNet</td><td>PNM</td><td>1000</td><td>-</td><td>0.0127</td><td>0.0430</td><td>Yes</td><td>1.846 ± 0.027</td><td>1.181</td></tr><tr><td>MNIST</td><td>LeNet</td><td>Lookahead</td><td>1000</td><td>-</td><td>0.0101</td><td>0.0430</td><td>Yes</td><td>1.982 ± 0.031</td><td>1.018</td></tr><tr><td>MNIST</td><td>LeNet</td><td>DiffGrad</td><td>1000</td><td>-</td><td>0.0105</td><td>0.0430</td><td>Yes</td><td>1.834 ± 0.026</td><td>1.198</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=128</td><td>0.00900</td><td>0.0430</td><td>Yes</td><td>1.991 ± 0.031</td><td>1.009</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=512</td><td>0.00787</td><td>0.0430</td><td>Yes</td><td>1.894 ± 0.028</td><td>1.119</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=640</td><td>0.0125</td><td>0.0430</td><td>Yes</td><td>1.838 ± 0.027</td><td>1.194</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=768</td><td>0.278</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=1024</td><td>0.129</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=8192</td><td>0.240</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=16384</td><td>0.249</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=32768</td><td>0.201</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=50000</td><td>0.139</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>B=60000</td><td>0.0936</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>N=600</td><td>0.205</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>N=800</td><td>0.0399</td><td>0.0430</td><td>Yes</td><td>1.995 ± 0.031</td><td>1.004</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>N=1000</td><td>0.0198</td><td>0.0430</td><td>Yes</td><td>2.128 ± 0.036</td><td>0.886</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>N=3000</td><td>0.0159</td><td>0.0430</td><td>Yes</td><td>2.091 ± 0.034</td><td>0.917</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>N=6000</td><td>0.0151</td><td>0.0430</td><td>Yes</td><td>2.001 ± 0.032</td><td>0.999</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>40% Label Noise</td><td>0.180</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>80% Label Noise</td><td>0.157</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Random Labels</td><td>0.0482</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>Fashion-MNIST</td><td>LeNet</td><td>Random</td><td>1000</td><td>-</td><td>0.0971</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>Fashion-MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>-</td><td>0.0132</td><td>0.0430</td><td>Yes</td><td>1.939 ± 0.030</td><td>1.065</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Eigengap</td><td>0.0153</td><td>0.0430</td><td>Yes</td><td>1.550 ± 0.017</td><td>1.817</td></tr><tr><td>Fashion-MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Eigengap</td><td>0.0240</td><td>0.0430</td><td>Yes</td><td>1.520 ± 0.017</td><td>1.922</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Epoch=1</td><td>0.0321</td><td>0.0430</td><td>Yes</td><td>1.908 ± 0.029</td><td>1.102</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Epoch=2</td><td>0.0298</td><td>0.0430</td><td>Yes</td><td>1.920 ± 0.029</td><td>1.087</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Epoch=3</td><td>0.0354</td><td>0.0430</td><td>Yes</td><td>1.916 ± 0.031</td><td>1.092</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Epoch=10</td><td>0.0291</td><td>0.0430</td><td>Yes</td><td>2.029 ± 0.033</td><td>0.972</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Epoch=20</td><td>0.0268</td><td>0.0430</td><td>Yes</td><td>2.081 ± 0.034</td><td>0.925</td></tr><tr><td>MNIST</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Epoch=30</td><td>0.0068</td><td>0.0430</td><td>Yes</td><td>2.074 ± 0.034</td><td>0.930</td></tr></table>

Table 8. The Kolmogorov-Smirnov statistics of LeNet on CIFAR-10 and CIFAR-100. The estimated power exponent $\hat{\beta}$ and slope magnitude $\hat{s}$ are also displayed. 

<table><tr><td>Dataset</td><td>Model</td><td>Training</td><td>Sample size</td><td>Setting</td><td> $d_{\text{ks}}$ </td><td> $d_c$ </td><td>Power-Law</td><td> $\hat{\beta} \pm \sigma$ </td><td> $\hat{s}$ </td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>Random</td><td>1000</td><td>-</td><td>0.0663</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>SGD</td><td>1000</td><td>-</td><td>0.0279</td><td>0.0430</td><td>Yes</td><td>1.968 ± 0.031</td><td>1.033</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>Vanilla SGD</td><td>1000</td><td>-</td><td>0.0276</td><td>0.0430</td><td>Yes</td><td>1.935 ± 0.030</td><td>1.069</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>Adam</td><td>1000</td><td>-</td><td>0.0269</td><td>0.0430</td><td>Yes</td><td>1.806 ± 0.025</td><td>1.241</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>AMSGrad</td><td>1000</td><td>-</td><td>0.0232</td><td>0.0430</td><td>Yes</td><td>1.786 ± 0.025</td><td>1.271</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>AdaBound</td><td>1000</td><td>-</td><td>0.0297</td><td>0.0430</td><td>Yes</td><td>1.901 ± 0.028</td><td>1.110</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>Yogi</td><td>1000</td><td>-</td><td>0.0184</td><td>0.0430</td><td>Yes</td><td>1.806 ± 0.025</td><td>1.241</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>RAdam</td><td>1000</td><td>-</td><td>0.0163</td><td>0.0430</td><td>Yes</td><td>1.733 ± 0.023</td><td>1.363</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>Adai</td><td>1000</td><td>-</td><td>0.0310</td><td>0.0430</td><td>Yes</td><td>1.918 ± 0.029</td><td>1.090</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>PNM</td><td>1000</td><td>-</td><td>0.0347</td><td>0.0430</td><td>Yes</td><td>1.911 ± 0.029</td><td>1.098</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>Lookahead</td><td>1000</td><td>-</td><td>0.0358</td><td>0.0430</td><td>Yes</td><td>1.964 ± 0.030</td><td>1.037</td></tr><tr><td>CIFAR-10</td><td>LeNet</td><td>DiffGrad</td><td>1000</td><td>-</td><td>0.0303</td><td>0.0430</td><td>Yes</td><td>1.803 ± 0.024</td><td>1.236</td></tr><tr><td>CIFAR-100</td><td>LeNet</td><td>Random</td><td>1000</td><td>-</td><td>0.0944</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>CIFAR-100</td><td>LeNet</td><td>SGD</td><td>1000</td><td>-</td><td>0.0315</td><td>0.0430</td><td>Yes</td><td>1.908 ± 0.029</td><td>1.101</td></tr><tr><td>CIFAR-100</td><td>LeNet</td><td>Vanilla SGD</td><td>1000</td><td>-</td><td>0.0379</td><td>0.0430</td><td>Yes</td><td>1.903 ± 0.029</td><td>1.108</td></tr><tr><td>CIFAR-100</td><td>LeNet</td><td>SGD</td><td>1000</td><td>Evaluated on CIFAR-10</td><td>0.0306</td><td>0.0430</td><td>Yes</td><td>1.913 ± 0.029</td><td>1.095</td></tr></table>

Table 9. The Kolmogorov-Smirnov statistics of FCN. The estimated power exponent $\hat{\beta}$ and slope magnitude $\hat{s}$ are also displayed. 

<table><tr><td>Dataset</td><td>Model</td><td>Training</td><td>Sample size</td><td>Setting</td><td> $d_{\text{ks}}$ </td><td> $d_c$ </td><td>Power-Law</td><td> $\hat{\beta} \pm \sigma$ </td><td> $\hat{s}$ </td></tr><tr><td>Avila</td><td>2Layer-FCN</td><td>SGD</td><td>50</td><td>-</td><td>0.0683</td><td>0.176</td><td>Yes</td><td>1.604 ± 0.085</td><td>1.656</td></tr><tr><td>MNIST</td><td>1Layer-FCN</td><td>Random</td><td>1000</td><td>-</td><td>0.185</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>1Layer-FCN</td><td>SGD</td><td>1000</td><td>-</td><td>0.241</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>2Layer-FCN</td><td>Random</td><td>1000</td><td>-</td><td>0.129</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>2Layer-FCN</td><td>SGD</td><td>1000</td><td>-</td><td>0.0112</td><td>0.0430</td><td>Yes</td><td>2.209 ± 0.038</td><td>0.827</td></tr><tr><td>MNIST</td><td>4Layer-FCN</td><td>Random</td><td>1000</td><td>-</td><td>0.0628</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>4Layer-FCN</td><td>SGD</td><td>1000</td><td>-</td><td>0.0141</td><td>0.0430</td><td>Yes</td><td>2.201 ± 0.038</td><td>0.833</td></tr><tr><td>MNIST</td><td>2Layer-FCN</td><td>SGD</td><td>1000</td><td>Width=10</td><td>0.149</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>2Layer-FCN</td><td>SGD</td><td>1000</td><td>Width=20</td><td>0.185</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>2Layer-FCN</td><td>SGD</td><td>1000</td><td>Width=30</td><td>0.0656</td><td>0.0430</td><td>No</td><td></td><td></td></tr><tr><td>MNIST</td><td>2Layer-FCN</td><td>SGD</td><td>1000</td><td>Width=50</td><td>0.0187</td><td>0.0430</td><td>Yes</td><td>2.138 ± 0.028</td><td>0.879</td></tr><tr><td>MNIST</td><td>2Layer-FCN</td><td>SGD</td><td>1000</td><td>Width=70</td><td>0.0376</td><td>0.0430</td><td>Yes</td><td>2.271 ± 0.030</td><td>0.787</td></tr><tr><td>MNIST</td><td>2Layer-FCN</td><td>SGD</td><td>1000</td><td>Width=100</td><td>0.0112</td><td>0.0430</td><td>Yes</td><td>2.209 ± 0.038</td><td>0.827</td></tr></table>

Table 10. The Kolmogorov-Smirnov statistics of ResNet18. The estimated power exponent $\hat{\beta}$ and slope magnitude $\hat{s}$ are also displayed. 

<table><tr><td>Dataset</td><td>Model</td><td>Training</td><td>Sample size</td><td>Setting</td><td> $d_{\text{ks}}$ </td><td> $d_c$ </td><td>Power-Law</td><td> $\hat{\beta} \pm \sigma$ </td><td> $\hat{s}$ </td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>Random</td><td>50</td><td>-</td><td>0.334</td><td>0.176</td><td>No</td><td></td><td></td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>SGD</td><td>50</td><td>-</td><td>0.0803</td><td>0.176</td><td>Yes</td><td>2.146 ± 0.162</td><td>0.873</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>Vanilla SGD</td><td>50</td><td>-</td><td>0.0891</td><td>0.176</td><td>Yes</td><td>2.193 ± 0.169</td><td>0.838</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>Adam</td><td>50</td><td>-</td><td>0.0478</td><td>0.176</td><td>Yes</td><td>2.062 ± 0.149</td><td>0.950</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>AMSGrad</td><td>50</td><td>-</td><td>0.0542</td><td>0.176</td><td>Yes</td><td>2.041 ± 0.147</td><td>0.961</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>AdaBound</td><td>50</td><td>-</td><td>0.0588</td><td>0.176</td><td>Yes</td><td>2.029 ± 0.146</td><td>0.971</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>Yogi</td><td>50</td><td>-</td><td>0.116</td><td>0.176</td><td>Yes</td><td>1.915 ± 0.129</td><td>1.092</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>RAdam</td><td>50</td><td>-</td><td>0.168</td><td>0.176</td><td>Yes</td><td>1.794 ± 0.1112</td><td>1.259</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>Adai</td><td>50</td><td>-</td><td>0.103</td><td>0.176</td><td>Yes</td><td>2.183 ± 0.167</td><td>0.845</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>PNM</td><td>50</td><td>-</td><td>0.138</td><td>0.176</td><td>Yes</td><td>2.132 ± 0.160</td><td>0.884</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>Lookahead</td><td>50</td><td>-</td><td>0.110</td><td>0.176</td><td>Yes</td><td>2.098 ± 0.155</td><td>0.911</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>DiffGrad</td><td>50</td><td>-</td><td>0.068</td><td>0.176</td><td>Yes</td><td>2.055 ± 0.149</td><td>0.948</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>SGD</td><td>50</td><td>B=512</td><td>0.0561</td><td>0.176</td><td>Yes</td><td>2.146 ± 0.151</td><td>0.936</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>SGD</td><td>50</td><td>B=1024</td><td>0.0647</td><td>0.176</td><td>Yes</td><td>2.076 ± 0.152</td><td>0.929</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>SGD</td><td>50</td><td>B=1152</td><td>0.0598</td><td>0.176</td><td>Yes</td><td>2.060 ± 0.150</td><td>0.944</td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>SGD</td><td>50</td><td>B=1280</td><td>0.331</td><td>0.176</td><td>No</td><td></td><td></td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>SGD</td><td>50</td><td>B=2048</td><td>0.334</td><td>0.176</td><td>No</td><td></td><td></td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>SGD</td><td>50</td><td>B=4096</td><td>0.334</td><td>0.176</td><td>No</td><td></td><td></td></tr><tr><td>CIFAR-10</td><td>ResNet18</td><td>SGD</td><td>50</td><td>B=16384</td><td>0.343</td><td>0.176</td><td>No</td><td></td><td></td></tr><tr><td>CIFAR-100</td><td>ResNet18</td><td>Random</td><td>50</td><td>-</td><td>0.373</td><td>0.176</td><td>No</td><td></td><td></td></tr><tr><td>CIFAR-100</td><td>ResNet18</td><td>SGD</td><td>50</td><td>-</td><td>0.108</td><td>0.176</td><td>Yes</td><td>2.299 ± 0.184</td><td>0.770</td></tr></table>