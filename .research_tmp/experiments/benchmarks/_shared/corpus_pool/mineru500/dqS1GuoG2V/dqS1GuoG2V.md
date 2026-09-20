# The Memory Perturbation Equation: Understanding Model's Sensitivity to Data

Peter Nickl $^{\dagger}$ peter.nickl@riken.jp

Lu Xu $^{*}$ $^{\dagger}$ lu.xu.sw@riken.jp

Dharmesh Tailor\*‡
d.v.tailor@uva.nl

Thomas Möllenhoff $^{\dagger}$ thomas.moellenhoff@riken.jp

Mohammad Emtiyaz Khan $^{\dagger\S}$ emtiyaz.khan@riken.jp

# Abstract

Understanding model's sensitivity to its training data is crucial but can also be challenging and costly, especially during training. To simplify such issues, we present the Memory-Perturbation Equation (MPE) which relates model's sensitivity to perturbation in its training data. Derived using Bayesian principles, the MPE unifies existing sensitivity measures, generalizes them to a wide-variety of models and algorithms, and unravels useful properties regarding sensitivities. Our empirical results show that sensitivity estimates obtained during training can be used to faithfully predict generalization on unseen test data. The proposed equation is expected to be useful for future research on robust and adaptive learning.

# 1 Introduction

Understanding model's sensitivity to training data is important to handle issues related to quality, privacy, and security. For example, we can use it to understand (i) the effect of errors and biases in the data; (ii) model's dependence on private information to avoid data leakage; (iii) model's weakness to malicious manipulations. Despite their importance, sensitivity properties of machine learning (ML) models are not well understood in general. Sensitivity is often studied through empirical investigations, but conclusions drawn this way do not always generalize across models or algorithms. Such studies are also costly, sometimes requiring thousands of GPUs [38], which can quickly become infeasible if we need to repeat them every time the model is updated.

A cheaper solution is to use local perturbation methods $[21]$ , for instance, influence measures that study sensitivity of trained model to data removal (Fig. 1(a)) $[8, 7]$ . Such methods too fall short of providing a clear understanding of sensitivity properties for generic cases. For instance, influence measures are useful to study trained models but are not suited to analyze training trajectories $[14, 54]$ . Another challenge is in handling non-differentiable loss functions or discrete parameter spaces where a natural choice of perturbation mechanisms may not always be clear $[32]$ . The measures also do not directly reveal the causes of sensitivities for generic ML models and algorithms.

In this paper, we simplify these issues by proposing a new method to unify, generalize, and understand perturbation methods for sensitivity analysis. We present the Memory-Perturbation Equation (MPE) as a unifying equation to understand sensitivity properties of generic ML algorithms. The equation builds upon the Bayesian learning rule (BLR) [28] which unifies many popular algorithms

![](images/4ef0fa70aa913c01ba786d9518a4f4992ce0ad82feac98ca32ed53ff241e8e95.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Start"] --> B["Iterations"]
    B --> C["Training on full dataset"]
    C --> D["Current"]
    D --> E["Estimate"]
    E --> F["Truth"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
```
</details>

![](images/07f971d01c6a33c74e8f30944134a72c3e59dbab96a6cbc0823b9c5c9f55b23e.jpg)

<details>
<summary>line</summary>

| Epochs | NLL Value |
| ------ | --------- |
| 0      | 1.9       |
| 50     | 1.2       |
| 100    | 0.6       |
| 150    | 0.6       |
| 200    | 0.6       |
</details>

Figure 1: Our main goal is to estimate the sensitivity of the training trajectory when examples are perturbed or simply removed; see Panel (a). We present the MPE to estimate the sensitivity without any retraining and use them to faithfully predict the test performance from training data alone; see Panel (b). The test negative log-likelihood (gray line) for ResNet–20 on CIFAR10 shows similar trends to the leave-one-out (LOO) score computed on the training data (black line).

from various fields as specific instances of a natural-gradient descent to solve a Bayesian learning problem. The MPE uses natural-gradients to understand sensitivity of all such algorithms. We use the MPE to show several new results regarding sensitivity of generic ML algorithms:

1. We show that sensitivity to a group of examples can be estimated by simply adding their natural-gradients; see Eq. 6. Larger natural-gradients imply higher sensitivity and just a few such examples can often account for most of the sensitivity. Such examples can be used to characterize the model's memory and memory-perturbation refers to the fact that the model can forget its essential knowledge when those examples are perturbed heavily.   
2. We derive Influence Function [8, 31] as a special case of the MPE when natural-gradients with respect to Gaussian posterior are used. More importantly, we derive new measures that, unlike influence functions, can be applied during training for all algorithms covered under the BLR (such as those used in deep learning and optimization). See Table 1.   
3. Measures derived using Gaussian posteriors share a common property: sensitivity to an example depends on the product of its prediction error and variance (Eq. 12). That is, most sensitive data lies where the model makes the most mistakes and is also least confident. In many cases, such estimates are extremely cheap to compute.   
4. We show that sensitivity of the training data can be used to accurately predict model generalization, even during training (Fig. 1(b)). This agrees with similar studies which also show effectiveness of sensitivity in predicting generalization [22, 12, 19, 4].

# 2 Understanding a Model's Sensitivity to Its Training Data

Understanding a model's sensitivity to its training data is important but is often done by a costly process of retraining the model multiple times. For example, consider a model with a parameter vector $\pmb{\theta} \in \mathbb{R}^P$ trained on data $\mathcal{D} = \{\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_N\}$ by using an algorithm $\mathcal{A}_t$ that generates a sequence $\{\pmb{\theta}_t\}$ for iteration $t$ that converges to a minimizer $\pmb{\theta}_*$ . Formally, we write

$$
\boldsymbol {\theta} _ {t} \leftarrow \mathcal {A} _ {t} \left(\boldsymbol {\theta} _ {t - 1}, \mathcal {L} (\boldsymbol {\theta})\right) \text {   where   } \mathcal {L} (\boldsymbol {\theta}) = \sum_ {i = 1} ^ {N} \ell_ {i} (\boldsymbol {\theta}) + \mathcal {R} (\boldsymbol {\theta}), \tag {1}
$$

and we use the loss $\ell_{i}(\boldsymbol{\theta})$ for $D_{i}$ and a regularizer $\mathcal{R}(\boldsymbol{\theta})$ . Because $\theta_{t}$ are all functions of D or its subsets, we can analyze their sensitivity by simply ‘perturbing’ the data. For example, we can remove a subset $M \subset D$ to get a perturbed dataset, denoted by $D^{M}$ , and retrain the model to get new iterates $\theta_{t}^{M}$ , converging to a minimizer $\theta_{*}^{M}$ . If the deviation $\theta_{t}^{M} - \theta_{t}$ is large for most t, we may deem the model to be highly sensitive to the examples in M. This is a simple method for sensitive analysis but requires a costly brute-force retraining [38] which is often infeasible for long training trajectories, big models, and large datasets. More importantly, conclusions drawn from retraining are often empirical and may not hold across models or algorithms.

A cheaper alternative is to use local perturbation methods [21], for instance, influence measures that estimate the sensitivity without retraining (illustrated in Fig. 1(a) by the dashed red arrow). The simplest result of this kind is for linear regression which dates back to the 70s [7]. The method makes use of the stationarity condition to derive deviations in $\theta_{*}$ due to small perturbations to data. For linear regression, the deviations can be obtained in closed-form. Consider input-output pairs $(\mathbf{x}_i, y_i)$ and the loss $\ell_i(\boldsymbol{\theta}) = \frac{1}{2}(y_i - f_i(\boldsymbol{\theta}))^2$ for $f_i(\boldsymbol{\theta}) = \mathbf{x}_i^\top \boldsymbol{\theta}$ and a regularizer $\mathcal{R}(\boldsymbol{\theta}) = \delta \| \boldsymbol{\theta} \|^2 / 2$ . We can obtain closed-form expressions of the deviation due to the removal of the $i$ 'th example as shown below (a proof is included in App. A),

$$
\boldsymbol {\theta} _ {*} ^ {\backslash i} - \boldsymbol {\theta} _ {*} = (\mathbf {H} _ {*} ^ {\backslash i}) ^ {- 1} \mathbf {x} _ {i} e _ {i}, \quad f _ {i} (\boldsymbol {\theta} _ {*} ^ {\backslash i}) - f _ {i} (\boldsymbol {\theta} _ {*}) = v _ {i} ^ {\backslash i} e _ {i}, \tag {2}
$$

where we denote $\mathbf{H}_{*}^{\backslash i} = \mathbf{H}_{*} - \mathbf{x}_{i}\mathbf{x}_{i}^{\top}$ defined using the Hessian $\mathbf{H}_{*} = \nabla^{2}\mathcal{L}(\boldsymbol{\theta}_{*})$ . We also denote the prediction error of $\boldsymbol{\theta}_{*}$ by $e_i = \mathbf{x}_i^\top \boldsymbol{\theta}_{*} - y_i$ , and prediction variance of $\boldsymbol{\theta}_{*}^{\backslash i}$ by $v_{i}^{\backslash i} = \mathbf{x}_i^\top (\mathbf{H}_*^{\backslash i})^{-1}\mathbf{x}_i$ .

The expression shows that the influence is bi-linearly related to both prediction error and variance, that is, when examples with high error and variance are removed, the model is expected to change a lot. These ideas are generalized using infinitesimal perturbation [21]. For example, influence functions [8, 32, 31] use a perturbation model $\pmb{\theta}_{*}^{\epsilon_{i}} = \arg \min_{\pmb{\theta}}\mathcal{L}(\pmb {\theta}) - \epsilon_{i}\ell_{i}(\pmb {\theta})$ with a scalar perturbation $\epsilon_{i}\in \mathbb{R}$ . By using a quadratic approximation, we get the following influence function,

$$
\left. \frac {\partial \boldsymbol {\theta} _ {*} ^ {\epsilon_ {i}}}{\partial \epsilon_ {i}} \right| _ {\epsilon_ {i} = 0} = \mathbf {H} _ {*} ^ {- 1} \nabla \ell_ {i} (\boldsymbol {\theta} _ {*}). \tag {3}
$$

This works for a generic differentiable loss function and is closely related to Eq. 2. We can choose other perturbation models, but they often exhibit bi-linear relationships; see App. A for details.

Despite their generality, there remain many open challenges with the local perturbation methods:

1. Influence functions are valid only at a stationary point $\theta_{*}$ where the gradient is assumed to be 0, and extending them to iterates $\theta_{t}$ generated by generic algorithmic-steps $A_{t}$ is non-trivial [14]. This is even more important for deep learning where we may never reach such a stationary point, for example, due to stochastic training or early stopping [33, 53].   
2. Applying influence functions to a non-differentiable loss or discrete parameter spaces is difficult. This is because the choice of perturbation model is not always obvious [32].   
3. Finally, despite their generality, these measures do not directly reveal the causes of high influence. Does the bi-linear relationship in Eq. 2 hold more generally? If yes, under what conditions? Answers to such questions are currently unknown.

Studies to fix these issues are rare in ML, rather it is more common to simply use heuristics measures. Many such measures have been proposed in the recent years, for example, those using derivatives with respect to inputs $[23, 2, 38]$ , variations of Cook's distance $[17]$ , prediction error and/or gradients $[3, 51, 42, 40]$ , backtracking training trajectories $[16]$ , or simply by retraining $[13]$ . These works, although useful, do not directly address the issues. Many of these measures are derived without any direct connections to perturbation methods. They also appear to be unaware of bi-linear relationships such as those in Eq. 2. Our goal here is to address the issues by unifying and generalizing perturbation methods of sensitivity analysis.

# 3 The Memory-Perturbation Equation (MPE)

We propose the memory-perturbation equation (MPE) to unify, generalize, and understand sensitivity methods in machine learning. We derive the equation by using a property of conjugate Bayesian models which enables us to derive a closed-form expression for the sensitivity. In a Bayesian setting, data examples can be removed by simply dividing their likelihoods from the posterior [52]. For example, consider a model with prior $p_0 = p(\boldsymbol{\theta})$ and likelihood $\tilde{p}_j = p(\mathcal{D}_j|\boldsymbol{\theta})$ , giving rise to a posterior $q_* = p(\boldsymbol{\theta}|\mathcal{D}) \propto p_0\tilde{p}_1\tilde{p}_2 \ldots \tilde{p}_N$ . To remove $\tilde{p}_j$ , say for all $j \in \mathcal{M} \subset \mathcal{D}$ , we simply divide $q_*$ by those $\tilde{p}_j$ . This is further simplified if we assume conjugate exponential-family form for $p_0$ and $\tilde{p}_j$ . Then, the division between two distributions is equivalent to a subtraction between their natural parameters. This property yields a closed-form expression for the exact deviation, as stated below.

Theorem 1 Assuming a conjugate exponential-family model, the posterior $q_*^{\backslash \mathcal{M}}$ (with natural parameter $\pmb{\lambda}_*^{\backslash \mathcal{M}}$ ) can be written in terms of $q_*$ (with natural parameter $\pmb{\lambda}_*$ ), as shown below:

$$
q _ {*} ^ {\backslash \mathcal {M}} \propto \frac {q _ {*}}{\prod_ {j \in \mathcal {M}} \tilde {p} _ {j}} \quad \Longrightarrow e ^ {\langle \boldsymbol {\lambda} _ {*} ^ {\backslash \mathcal {M}}, \mathbf {T} (\boldsymbol {\theta}) \rangle} \propto \frac {e ^ {\langle \boldsymbol {\lambda} _ {*} , \mathbf {T} (\boldsymbol {\theta}) \rangle}}{\prod_ {j \in \mathcal {M}} e ^ {\langle \widetilde {\boldsymbol {\lambda}} _ {j} , \mathbf {T} (\boldsymbol {\theta}) \rangle}} \quad \Longrightarrow \boldsymbol {\lambda} _ {*} ^ {\backslash \mathcal {M}} = \boldsymbol {\lambda} _ {*} - \sum_ {j \in \mathcal {M}} \widetilde {\boldsymbol {\lambda}} _ {j}. (4)
$$

where all exponential families are defined by using inner-product $\langle\boldsymbol{\lambda},\mathbf{T}(\boldsymbol{\theta})\rangle$ with natural parameters $\lambda$ and sufficient statistics $\mathbf{T}(\boldsymbol{\theta})$ . The natural parameter of $\tilde{p}_{j}$ is denoted by $\lambda_{j}$ .

The deviation $\lambda_{*}^{\backslash \mathcal{M}} - \lambda_{*}$ is obtained by simply adding $\widetilde{\lambda}_j$ for all $j\in \mathcal{M}$ . Further explanations and examples are given in App. B, along with some elementary facts about exponential families. We use this result to derive an equation that enables us to estimate the sensitivity of generic algorithms.

Our derivation builds on the Bayesian learning rule (BLR) [28] which unifies many algorithms by expressing their iterations as inference in conjugate Bayesian models [26]. This is done by reformulating Eq. 1 in a Bayesian setting to find an exponential-family approximation $q_* \approx p(\boldsymbol{\theta}|\mathcal{D}) \propto e^{-\mathcal{L}(\boldsymbol{\theta})}$ . At every iteration $t$ , the BLR updates the natural parameter $\lambda_t$ of an exponential-family $q_t$ which can equivalently be expressed as the posterior of a conjugate model (shown on the right),

$$
\boldsymbol {\lambda} _ {t} \leftarrow (1 - \rho) \boldsymbol {\lambda} _ {t - 1} - \rho \sum_ {j = 0} ^ {N} \tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda} _ {t - 1}) \quad \Longleftrightarrow \quad q _ {t} \propto \underbrace {(q _ {t - 1}) ^ {1 - \rho} (p _ {0}) ^ {\rho}} _ {\text { Prior }} \prod_ {j = 1} ^ {N} e \underbrace {\langle - \rho \tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda} _ {t - 1}) , \mathbf {T} (\boldsymbol {\theta}) \rangle} _ {\text { Likelihood }} \tag {5}
$$

where $\tilde{\mathbf{g}}_j(\boldsymbol {\lambda}) = \mathbf{F}(\boldsymbol {\lambda})^{-1}\nabla_{\boldsymbol{\lambda}}\mathbb{E}_q[\ell_j(\boldsymbol {\theta})]$ is the natural gradient with respect to $\boldsymbol{\lambda}$ defined using the Fisher Information Matrix $\mathbf{F}(\boldsymbol{\lambda}_t)$ of $q_{t}$ , and $\rho >0$ is the learning rate. For simplicity, we denote $\ell_0(\pmb {\theta}) = \mathcal{R}(\pmb {\theta}) = -\log p_0$ , and assume $p_0$ to be conjugate. The conjugate model on the right uses a prior and likelihood both of which, by construction, belong to the same exponential-family as $q_{t}$ . By choosing an appropriate form for $q_{t}$ and making necessary approximations to $\tilde{\mathbf{g}}_j$ , the BLR can recover many popular algorithms as special cases. For instance, using a Gaussian $q_{t}$ , we can recover stochastic gradient descent (SGD), Newton's method, RMSprop, Adam, etc. For such cases, the conjugate model at the right is often a linear model [25]. These details, along with a summary of the BLR, are included in App. C. Our main idea is to study the sensitivity of all the algorithms covered under the BLR by using the conjugate model in Eq. 5.

Let $q_{t}^{\backslash \mathcal{M}}$ be the posterior obtained with the BLR but without the data in $\mathcal{M}$ . We can estimate its natural parameter $\pmb{\lambda}_{t}^{\backslash \mathcal{M}}$ in a similar fashion as Eq. 4, that is, by dividing $q_{t}$ by the likelihood approximation at the current $\pmb{\lambda}_{t}$ . This gives us the following estimate of the deviation obtained by simply adding the natural-gradients for all examples in $\mathcal{M}$ ,

$$
\hat {\boldsymbol {\lambda}} _ {t} ^ {\backslash \mathcal {M}} - \boldsymbol {\lambda} _ {t} = \rho \sum_ {j \in \mathcal {M}} \tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda} _ {t}) \tag {6}
$$

where $\hat{\lambda}_t^{\backslash \mathcal{M}}$ is an estimate of the true $\lambda_t^{\backslash \mathcal{M}}$ . We call this the memory-perturbation equation (MPE) due to a unique property of the equation: the deviation is estimated by a simple addition and characterized solely by the examples in $\mathcal{M}$ . Due to the additive nature of the estimate, examples with larger natural-gradients contribute more to it and so we expect most of the sensitivity to be explained by just a few examples with largest natural gradients. This is similar to the representer theorem where just a few support vectors are sufficient to characterize the decision boundary [29, 47, 10]. Here, such examples can be seen as characterizing the model's memory because perturbing them can make the model forget its essential knowledge. The phrase memory-perturbation signifies this.

The equation can be easily adopted to handle an arbitrary perturbation. For instance, consider perturbation $\mathcal{L}(\boldsymbol{\theta}) - \sum_{j \in \mathcal{M}} \epsilon_j \ell_j(\boldsymbol{\theta})$ . To estimate its effect, we divide $q_t$ by the likelihood approximations raised to $\epsilon_i$ , giving us the following variant,

$$
\hat {\boldsymbol {\lambda}} _ {t} ^ {\boldsymbol {\epsilon} _ {\mathcal {M}}} - \boldsymbol {\lambda} _ {t} = \rho \sum_ {j \in \mathcal {M}} \epsilon_ {j} \tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda} _ {t}), \quad \Longrightarrow \quad \left. \frac {\partial \hat {\boldsymbol {\lambda}} _ {t} ^ {\boldsymbol {\epsilon} _ {\mathcal {M}}}}{\partial \epsilon_ {j}} \right| _ {\epsilon_ {j} = 0} = \rho \tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda} _ {t}), \quad \forall j \in \mathcal {M}, \tag {7}
$$

where we denote all $\epsilon_{j}$ in M by $\epsilon_{M}$ . Setting $\epsilon_{j}=1$ in the left reduces to Eq. 6 which corresponds to removal. The example demonstrates how to adopt the MPE to handle arbitrary perturbations.

# 3.1 Unifying the existing sensitivity measures as special cases of the MPE

The MPE is a unifying equation from which many existing sensitivity measures can be derived as special cases. We will show three such results. The first result shows that, for conjugate models, the MPE recovers the exact deviations given in Thm. 1. Such models include textbook examples [6], such as, mixture models, linear state-space models, and PCA. Below is a formal statement.

Theorem 2 For conjugate exponential-family models, Eq. 4 is obtained as a special case of the MPE in Eq. 6 evaluated at $\lambda_{*}$ of the exact posterior $q_{*}$ when we set $\ell_j(\theta) = -\log \tilde{p}_j$ and $\rho = 1$ .

The result holds because, for conjugate models, one-step of the BLR is equivalent to Bayes' rule and therefore $\tilde{\mathbf{g}}_j(\boldsymbol{\lambda}_*) = -\widetilde{\boldsymbol{\lambda}}_j$ (see [24, Sec. 5.1]). A proof is given in App. D along with an illustrative example on the Beta-Bernoulli model. We note that a recent work in [49] also takes inspiration from Bayesian models, but their sensitivity measures lack the property discussed above. See also [15] for a different approach to sensitivity analysis of variational Bayes with a focus on the posterior mean. The result above also justifies setting $\rho$ to 1, a choice we will often resort to.

Our second result is to show that the MPE recovers the influence function by Cook [7].

Theorem 3 For linear regression, Eq. 2 is obtained as a special case of the MPE in Eq. 6 evaluated at $\lambda_{*}$ of the exact posterior $q_{*} = \mathcal{N}(\pmb{\theta}|\pmb{\theta}_{*},\mathbf{H}_{*}^{-1})$ .

The proof in App. E relies on two facts: first, the natural parameter is $\lambda_{*} = (\mathbf{H}_{*}\boldsymbol{\theta}_{*}, -\frac{1}{2}\mathbf{H}_{*})$ , and second, the natural gradients for a Gaussian $q$ with mean $\mathbf{m}$ can be written as follows,

$$
\tilde {\mathbf {g}} _ {i} (\boldsymbol {\lambda}) = \left(\hat {\mathbf {g}} _ {i} - \hat {\mathbf {H}} _ {i} \mathbf {m}, \frac {1}{2} \hat {\mathbf {H}} _ {i}\right), \tag {8}
$$

where $\hat{\mathbf{g}}_{i}=\mathbb{E}_{q}[\nabla\ell_{i}(\boldsymbol{\theta})]$ and $\hat{\mathbf{H}}_{i}=\mathbb{E}_{q}[\nabla^{2}\ell_{i}(\boldsymbol{\theta})]$ . This is due to [28, Eqs. 10-11], but a proof is given in Eq. 27 of App. C. The theorem then directly follows by plugging $\tilde{\mathbf{g}}_{i}(\boldsymbol{\lambda}_{*})$ in Eq. 6. This derivation is much shorter than the classical techniques which often require inversion lemmas (see App. A.1). The estimated deviations are exact, which is not a surprise because linear regression is a conjugate Gaussian model. However, it is interesting (and satisfying) that the deviation in $\theta_{*}$ naturally emerges from the deviation in $\lambda_{*}$ .

Our final result is to recover influence functions for deep learning, specifically Eq. 3. To do so, we use a Gaussian posterior approximation $q_{*} = \mathcal{N}(\theta |\pmb{\theta}_{*},\mathbf{H}_{*}^{-1})$ obtained by using the so-called Laplace's method [34, 50, 37]. The Laplace posterior can be seen a special case of the BLR solution when the natural gradient is approximated with the delta method [28, Table 1]. Remarkably, using the same approximation in the MPE, we recover Eq. 3.

Theorem 4 The influence function in Eq. 3 is obtained as a special case of the MPE in Eq. 7 evaluated at $\lambda_{*}$ of the posterior $q_{*} = \mathcal{N}(\pmb{\theta}|\pmb{\theta}_{*},\mathbf{H}_{*}^{-1})$ when we approximate $\tilde{\mathbf{g}}_i(\pmb {\lambda})$ of Eq. 8 with the delta method by substituting $\mathbb{E}_{q_*}[\nabla \ell_i(\pmb {\theta})]\approx \nabla \ell_i(\pmb {\theta}_*)$ and $\mathbb{E}_{q_*}[\nabla^2\ell_i(\pmb {\theta})]\approx \nabla^2\ell_i(\pmb {\theta}_*)$ .

A proof is in App. F. We note that Eq. 3 can be justified as a Newton-step over the perturbed data but in the opposite direction [32, 31]. In a similar fashion, Eqs. 6 and 7 can be seen as natural-gradient steps in the opposite direction. Using the natural-gradient descent, as we have shown, can recover a variety of existing perturbation methods as special cases.

# 3.2 Generalizing the perturbation method to estimate sensitivity during training

Influence measures discussed so far assume that the model is already trained and that the loss is differentiable. We will now present generalizations to obtain new measures that can be applied during training and do not require differentiability of the loss. We will focus on Gaussian q but the derivation can be extended to other posterior forms. The main idea is to specialize Eqs. 6 and 7 to the algorithms covered under the BLR, giving rise to new measures that estimate sensitivity by simply taking a step over the perturbed data but in the opposite direction.

We first discuss sensitivity of an iteration $t$ of the BLR yielding a Gaussian $q_{t} = \mathcal{N}(\pmb{\theta}|\mathbf{m}_{t},\mathbf{S}_{t}^{-1})$ . The natural parameter is the pair $\lambda_{t} = (\mathbf{S}_{t}\mathbf{m}_{t}, - \frac{1}{2}\mathbf{S}_{t})$ . Using Eq. 8 in Eq. 6, we get

$$
\hat {\mathbf {S}} _ {t} ^ {\backslash \mathcal {M}} \hat {\mathbf {m}} _ {t} ^ {\backslash \mathcal {M}} - \mathbf {S} _ {t} \mathbf {m} _ {t} = \rho \sum_ {j \in \mathcal {M}} \mathbb {E} _ {q _ {t}} [ \nabla \ell_ {j} (\boldsymbol {\theta}) ] - \mathbb {E} _ {q _ {t}} [ \nabla^ {2} \ell_ {j} (\boldsymbol {\theta}) ] \mathbf {m} _ {t}, \quad \mathbf {S} _ {t} - \hat {\mathbf {S}} _ {t} ^ {\backslash \mathcal {M}} = \rho \sum_ {j \in \mathcal {M}} \mathbb {E} _ {q _ {t}} [ \nabla^ {2} \ell_ {j} (\boldsymbol {\theta}) ] \tag {9}
$$

<table><tr><td>Algorithm</td><td>Update</td><td>Sensitivity</td></tr><tr><td>Newton&#x27;s method</td><td> $\boldsymbol{\theta}_{t} \leftarrow \boldsymbol{\theta}_{t-1} - \mathbf{H}_{t-1}^{-1} \nabla \mathcal{L}(\boldsymbol{\theta}_{t-1})$ </td><td> $\mathbf{H}_{t-1}^{-1} \nabla \ell_{i}(\boldsymbol{\theta}_{t})$ </td></tr><tr><td>Online Newton (ON) [28]</td><td> $\boldsymbol{\theta}_{t} \leftarrow \boldsymbol{\theta}_{t-1} - \rho \mathbf{S}_{t}^{-1} \nabla \mathcal{L}(\boldsymbol{\theta}_{t-1})$ </td><td> $\mathbf{S}_{t}^{-1} \nabla \ell_{i}(\boldsymbol{\theta}_{t})$ </td></tr><tr><td>ON (diagonal+minibatch) [28]</td><td> $\boldsymbol{\theta}_{t} \leftarrow \boldsymbol{\theta}_{t-1} - \rho \mathbf{s}_{t}^{-1} \cdot \hat{\nabla} \mathcal{L}(\boldsymbol{\theta}_{t-1})$ </td><td> $\mathbf{s}_{t}^{-1} \cdot \nabla \ell_{i}(\boldsymbol{\theta}_{t})$ </td></tr><tr><td>iBLR (diagonal+minibatch) [35]</td><td> $\mathbf{m}_{t} \leftarrow \mathbf{m}_{t-1} - \rho \mathbf{s}_{t}^{-1} \cdot \hat{\nabla} \mathcal{L}(\boldsymbol{\theta}_{t-1})$ </td><td> $\mathbf{s}_{t}^{-1} \cdot \nabla \ell_{i}(\boldsymbol{\theta}_{t})$ </td></tr><tr><td>RMSprop/Adam [30]</td><td> $\boldsymbol{\theta}_{t} \leftarrow \boldsymbol{\theta}_{t-1} - \rho \mathbf{s}_{t}^{-\frac{1}{2}} \cdot \hat{\nabla} \mathcal{L}(\boldsymbol{\theta}_{t-1})$ </td><td> $\mathbf{s}_{t}^{-\frac{1}{2}} \cdot \nabla \ell_{i}(\boldsymbol{\theta}_{t})$ </td></tr><tr><td>SGD</td><td> $\boldsymbol{\theta}_{t} \leftarrow \boldsymbol{\theta}_{t-1} - \rho \hat{\nabla} \mathcal{L}(\boldsymbol{\theta}_{t-1})$ </td><td> $\nabla \ell_{i}(\boldsymbol{\theta}_{t})$ </td></tr></table>

Table 1: A list of algorithms and their sensitivity measures derived using Eq. 10. The second column gives the update, most of which use pre-conditioners that are either matrices $(\mathbf{H}_{t}, \mathbf{S}_{t})$ or a vector $(\mathbf{s}_{t})$ ; see the full update equations in Eqs. 31 to 34 in App. C. The third column shows the associated sensitivity measure to perturbation in the i'th example which can be interpreted as a step for the i example but in the opposite direction. We denote the element-wise multiplication between vectors by “.” and the minibatch gradients by $\hat{\nabla}$ . For iBLR, $\theta_{t}$ is either $m_{t}$ or a sample from $q_{t}$ .

Plugging $S_{t}$ from the second equation into the first one, we can recover the following expressions,

$$
\hat {\mathbf {m}} _ {t} ^ {\backslash \mathcal {M}} - \mathbf {m} _ {t} = \rho \big (\hat {\mathbf {S}} _ {t} ^ {\backslash \mathcal {M}} \big) ^ {- 1} \mathbb {E} _ {q _ {t}} \Big [ \sum_ {j \in \mathcal {M}} \nabla \ell_ {j} (\boldsymbol {\theta}) \Big ], \quad \frac {\partial \hat {\mathbf {m}} _ {t} ^ {\epsilon_ {i}}}{\partial \epsilon_ {i}} \Bigg | _ {\epsilon_ {i} = 0} = \rho \mathbf {S} _ {t} ^ {- 1} \mathbb {E} _ {q _ {t}} \left[ \nabla \ell_ {i} (\boldsymbol {\theta}) \right] \tag {10}
$$

For the second equation, we omit the proof but it is similar to App. F, resulting in preconditioning with $S_{t}$ . For computational ease, we will approximate $\hat{S}_{t}^{\backslash M} \approx S_{t}$ even in the first equation. We will also approximate the expectation at a sample $\theta_{t} \sim q_{t}$ or simply at the mean $\theta_{t} = m_{t}$ . Ultimately, the suggestion is to use $\mathbf{S}_{t}^{-1} \nabla \ell_{i}(\boldsymbol{\theta}_{t})$ as the sensitivity measure, or variations of it, for example, by using a Monte-Carlo average over multiple samples.

Based on this, a list of algorithms and their corresponding measures is given in Table 1. All of the algorithms can be derived as special instances of the BLR by making specific approximations (see App. C.3). The measures are obtained by applying the exact same approximations to Eq. 10. For example, Newton's method is obtained when $\mathbf{m}_t = \boldsymbol{\theta}_t$ , $\mathbf{S}_t = \mathbf{H}_{t-1}$ , and expectations are approximated by using the delta method at $\boldsymbol{\theta}_t$ (similarly to Thm. 4). With these, we get

$$
\mathbf {S} _ {t} ^ {- 1} \mathbb {E} _ {q _ {t}} [ \nabla \ell_ {i} (\boldsymbol {\theta}) ] \approx \mathbf {H} _ {t - 1} ^ {- 1} \nabla \ell_ {i} (\boldsymbol {\theta} _ {t}), \tag {11}
$$

which is the measure shown in the first row of the table. In a similar fashion, we can derive measures for other algorithms that use a slightly different approximations leading to a different preconditioner. The exact strategy to update the preconditioners is given in Eqs. 31 to 34 of App. C.3. For all, the sensitivity measure is simply an update step for the $i$ 'th example but in the opposite direction.

Table 1 shows an interplay between the training algorithm and sensitivity measures. For instance, it suggests that the measure $\mathbf{H}_{t - 1}^{-1}\nabla \ell_i(\pmb{\theta}_t)$ is justifiable for Newton's method but might be inappropriate otherwise. In general, it is more appropriate to use the algorithm's own preconditioner (if they use one). The quality of preconditioner (and therefore the measure) is tied to the quality of the posterior approximation. For example, RMSprop's preconditioner is not a good estimator of the posterior covariance when minibatch size is large [27, Thm. 1], therefore we should not expect it to work well for large minibatches. In contrast, the ON method [28] explicitly builds a good estimate of $\mathbf{S}_t$ during training and we expect it to give better (and more faithful) sensitivity estimates.

For SGD, our approach suggests using the gradient. This goes well with many existing approaches [40, 42, 51, 3] but also gives a straightforward way to modify them when the training algorithm is changed. For instance, the TracIn approach [42] builds sensitivity estimates during SGD training by tracing $\nabla \ell_j(\pmb{\theta}_t)^\top \nabla \ell_i(\pmb{\theta}_t)$ for many examples $i$ and $j$ . When the algorithm is switched, say to the ON method, we simply need to trace $\nabla \ell_j(\pmb{\theta}_t)^\top \mathbf{S}_t^{-1} \nabla \ell_i(\pmb{\theta}_t)$ . Such a modification is speculated in [42, Sec 3.2] and the MPE provides a way to accomplish exactly that. It is also possible to mix and match algorithms with different measures but caution is required. For example, to use the measure in Eq. 11, say within a first-order method, the algorithm must be modified to build a well-conditioned estimate of the Hessian. This can be tricky and can make the sensitivity measure fragile [5].

Extensions to non-differentiable loss functions and discontinuous parameter spaces is straightforward. For example, when using a Gaussian posterior, the measures in Eq. 10 can be modified to

handle non-differentiable loss function by simply replacing $\mathbb{E}_{q_{t}}[\nabla\ell_{i}(\boldsymbol{\theta})]$ with $\nabla_{m}\mathbb{E}_{q_{t}}[\ell_{i}(\boldsymbol{\theta})]$ , which is a simple application of the Bonnet theorem [44] (see App. G). The resulting approach is more principled than [32] which uses an ad-hoc smoothing of the non-differentiable loss: the smoothing in our approach is automatically done by using the posterior distribution. Handling of discontinuous parameter spaces follows in a similar fashion. For example, binary variables can be handled by measuring the sensitivity through the parameter of the Bernoulli distribution (see App. D).

# 3.3 Understanding the causes of high sensitivity estimates for the Gaussian case

The MPE can be used to understand the causes of high sensitivity-estimates. We will demonstrate this for Gaussian q but similar analysis can be done for other distributions. We find that sensitivity measures derived using Gaussian posteriors generally have two causes of high sensitivity.

To see this, consider a loss $\ell_i(\boldsymbol{\theta}) = -\log p(y_i|\sigma(f_i(\boldsymbol{\theta})))$ where $p(y_i|\mu)$ is an exponential-family distribution with expectation parameter $\mu$ , $f_i(\boldsymbol{\theta})$ is the model output for the $i$ 'th example, and $\sigma(\cdot)$ is an activation function, for example, the softmax function. For such loss functions, the gradient takes a simple form: $\nabla \ell_i(\boldsymbol{\theta}) = \nabla f_i(\boldsymbol{\theta})[\sigma(f_i(\boldsymbol{\theta})) - y_i]$ [6, Eq. 4.124]. Using this, we can approximate the deviations in model outputs by using a first-order Taylor approximation,

$$
\underbrace {f _ {i} \left(\boldsymbol {\theta} _ {t} ^ {\backslash i}\right) - f _ {i} \left(\boldsymbol {\theta} _ {t}\right)} _ {\text { Deviation   in   the   output }} \approx \nabla f _ {i} \left(\boldsymbol {\theta} _ {t}\right) ^ {\top} \left(\boldsymbol {\theta} _ {t} ^ {\backslash i} - \boldsymbol {\theta} _ {t}\right) \approx \underbrace {\nabla f _ {i} \left(\boldsymbol {\theta} _ {t}\right) ^ {\top} \mathbf {H} _ {t - 1} ^ {- 1} \nabla f _ {i} \left(\boldsymbol {\theta} _ {t}\right)} _ {= v _ {i t}, \text { prediction   variance }} \underbrace {[ \sigma (f _ {i} (\boldsymbol {\theta} _ {t})) - y _ {i} ]} _ {= e _ {i t}, \text { prediction   error }}. \tag {12}
$$

where we used $\theta_{t}^{\backslash i}-\theta_{t}\approx\mathbf{H}_{t-1}^{-1}\nabla\ell_{i}(\boldsymbol{\theta}_{t})$ which is based on the measure in the first row of Table 1. Similarly to Eq. 2, the deviation in the model output is equal to the product of the prediction error and (linearized) prediction variance of $f_{i}(\boldsymbol{\theta}_{t})$ [25, 20]. The change in the model output is expected to be high, whenever examples with high prediction error and variance are removed.

We can write many such variants with a similar bi-linear relationship. For example, Eq. 12 can be extended to get deviations in predictions as follows:

$$
\sigma (f _ {i} (\boldsymbol {\theta} _ {t} ^ {\backslash i})) - \sigma (f _ {i} (\boldsymbol {\theta} _ {t})) \approx \sigma^ {\prime} (f _ {i} (\boldsymbol {\theta} _ {t})) \nabla f _ {i} (\boldsymbol {\theta} _ {t}) ^ {\top} (\boldsymbol {\theta} _ {t} ^ {\backslash i} - \boldsymbol {\theta} _ {t}) \approx \sigma^ {\prime} (f _ {i} (\boldsymbol {\theta} _ {t})) v _ {i t} e _ {i t}. \tag {13}
$$

Eq. 12 estimates the deviation at one example and at a location $\theta_{t}$ , but we could also write them for a group of examples and evaluate them at the mean $\mathbf{m}_{t}$ or at any sample $\theta \sim q_{t}$ . For example, to remove a group $\mathcal{M}$ of size $M$ , we can write the deviation of the model-output vector $\mathbf{f}(\boldsymbol{\theta}) \in \mathbb{R}^{M}$ ,

$$
\mathbf {f} \left(\mathbf {m} _ {t} ^ {\backslash \mathcal {M}}\right) - \mathbf {f} \left(\mathbf {m} _ {t}\right) \approx \nabla \mathbf {f} \left(\mathbf {m} _ {t}\right) ^ {\top} \mathbf {S} _ {t} ^ {- 1} \nabla \mathbf {f} \left(\mathbf {m} _ {t}\right) [ \sigma (\mathbf {f} \left(\mathbf {m} _ {t}\right) - \mathbf {y} ], \tag {14}
$$

where y is the vector of labels and we used the sensitivity measure in Eq. 10. An example for sparse Gaussian process is in App. H. The measure for SGD in Table 1 can also be used which gives $f_{i}(\boldsymbol{\theta}_{t}^{\backslash i}) - f_{i}(\boldsymbol{\theta}_{t}) \approx \|\nabla f_{i}(\boldsymbol{\theta})\|^{2}e_{it}$ which is similar to the scores used in [40]. The list in Table 1 suggests that such scores can be improved by using $H_{t}$ or $S_{t}$ , essentially, replacing the gradient norm by an estimate of the prediction variance. Additional benefit can be obtained by further employing samples from $q_{t}$ instead of using a point estimate $\theta_{t}$ or $m_{t}$ ; see an example in App. H.

It is also clear that all of the deviations above can be obtained cheaply during training by using already computed quantities. The estimation does not add significant computational overhead and can be used to efficiently predict the generalization performance during training. For example, using Eq. 12, we can approximate the leave-one-out (LOO) cross-validation (CV) error as follows,

$$
\operatorname{LOO} \left(\boldsymbol {\theta} _ {t}\right) = \sum_ {i = 1} ^ {N} \ell_ {i} \left(\boldsymbol {\theta} _ {t} ^ {\backslash i}\right) = - \sum_ {i = 1} ^ {N} \log p \left(y _ {i} \mid \sigma \left(f _ {i} \left(\boldsymbol {\theta} _ {t} ^ {\backslash i}\right)\right)\right) \approx - \sum_ {i = 1} ^ {N} \log p \left(y _ {i} \mid \sigma \left(f _ {i} \left(\boldsymbol {\theta} _ {t}\right) + v _ {i t} e _ {i t}\right)\right). \tag {15}
$$

The approximation eliminates the need to train N models to perform CV, rather just uses $e_{it}$ and $v_{it}$ which are extremely cheap to compute within algorithms such as ON, RMSprop, and SGD. Leave-group-out (LGO) estimates can also be built, for example, by using Eq. 14, which enables us to understand the effect of leaving out a big chunk of training data, for example, an entire class for classification. The LOO and LGO estimates are closely related to marginal likelihood and sharpness, both of which are useful to predict generalization performance [22, 12, 19]. Estimates similar to Eq. 15 have been proposed previously [43, 4] but none of them do so during training.

![](images/553c9cbbc5cd44adc4c11e8de1dde1fbba968d0f59491407a57a52c05ef303da.jpg)

<details>
<summary>scatter</summary>

| True Deviation | h(f(θ*) - h(f(θ⁻ⁱ))) |
| -------------- | --------------------- |
| 0.0            | 0.0                   |
| 0.1            | 0.1                   |
| 0.2            | 0.2                   |
| 0.3            | 0.3                   |
| 0.4            | 0.4                   |
| 0.5            | 0.5                   |
| 0.6            | 0.6                   |
| 0.7            | 0.7                   |
| 0.8            | 0.8                   |
| 0.9            | 0.9                   |
| 1.0            | 1.0                   |
</details>

![](images/a5cafefe190cadecaa748e699f22e6fc10ca591be7e444013e351acbef0ca7fe.jpg)

<details>
<summary>scatter</summary>

| True Deviation | h(f(θ*)) - |
| -------------- | ------------ |
| 0.0            | 0.0          |
| 0.1            | 0.1          |
| 0.2            | 0.2          |
| 0.3            | 0.3          |
| 0.4            | 0.4          |
</details>

![](images/f40e8398d7f1ed46476a551eb2c6bd2563c921025bf3c03fc10b4d7c76b1169b.jpg)

<details>
<summary>scatter</summary>

| True Deviation | h(f(θ*)) - h(f(θ⁻ⁱ)) |
| -------------- | --------------------- |
| 0.0            | 0.0                   |
| 0.1            | 0.5                   |
| 0.2            | 1.0                   |
| 0.3            | 1.5                   |
| 0.4            | 2.0                   |
| 0.5            | 2.5                   |
| 0.6            | 3.0                   |
</details>

![](images/0cf2d1beb60437fee51660b228cc1115a650d2552cf031dd7b7664988ee10bf2.jpg)

<details>
<summary>bar</summary>

| Sensitivity Level | True Deviation |
| ----------------- | -------------- |
| Low Sensitivity   | 95%            |
| High Sensitivity  | 0.4%           |
| Low Sensitivity   | 0.2%           |
| High Sensitivity  | 0.3%           |
| Low Sensitivity   | 2%             |
| High Sensitivity  | 3%             |
| Low Sensitivity   | 7%             |
| High Sensitivity  | 7%             |
| Low Sensitivity   | 3%             |
| High Sensitivity  | 7%             |
| Low Sensitivity   | 7%             |
| High Sensitivity  | 7%             |
</details>

(a) MLP on MNIST

![](images/f36a0cc5aeda2ba36a32f0b40508a74f28d77bee9bfad5da548d21950431698f.jpg)

<details>
<summary>bar</summary>

| Sensitivity | Bag | Shoe |
| --- | --- | --- |
| Low Sensitivity | 5% | 4% |
| High Sensitivity | 1% | 1% |
| Low Sensitivity (True Deviation) | 36% | 0.4% |
| High Sensitivity (True Deviation) | 0.3% | 0.1% |
</details>

(b) LeNet on FMNIST

![](images/439eed6bbb8e2f0587687085482f1b5e2506b2d1433d66c03e02feb5832e56ad.jpg)

<details>
<summary>bar</summary>

| Sensitivity Level | Dog | Bird |
| ----------------- | --- | ---- |
| Low Sensitivity   | 82% | 6%   |
| High Sensitivity  | 0%  | 0%   |
</details>

(c) CNN on CIFAR-10   
Figure 2: The estimated deviation for an example removal correlates well with the true deviations in predictions. Each marker represents an example. For each panel, the histogram at the bottom shows that the majority of examples have low sensitivity and most of the large sensitivities are attributed to a small fraction of data. We show a few images of high and low sensitivity examples from two randomly chosen classes, where we observe the high-sensitivity examples to be more interesting (possibly mislabeled or just ambiguous), while low-sensitivity examples appear more predictable.

# 4 Experiments

We show experimental results to demonstrate the usefulness of the MPE to understand the sensitivity of deep-learning models. We show the following: (1) we verify that the estimated deviations (sensitivities) for data removal correlate with the truth; (2) we predict the effect of class removal on generalization error; (3) we estimate the cross-validation curve for hyperparameter tuning; (4) we predict generalization during training; and (5) we study evolution of sensitivities during training. All details of the experimental setup are included in App. I and the code is available at https://github.com/team-approx-bayes/memory-perturbation.

Estimated deviations correlate with the truth: Fig. 2 shows a good correlation between the true deviations $\sigma(f_i(\boldsymbol{\theta}_*)^-) - \sigma(f_i(\boldsymbol{\theta}_*))$ and their estimates $\sigma'(f_i(\boldsymbol{\theta}_*))v_{i*}e_{i*}$ , as shown in Eq. 13. We show results for three datasets, each using a different architecture but all trained using SGD. To estimate the Hessian $\mathbf{H}_*$ and compute $v_{i*} = \nabla f_i(\boldsymbol{\theta}_*)^\top\mathbf{H}_*^{-1}\nabla f_i(\boldsymbol{\theta}_*)$ , we use a Kronecker-factored (K-FAC) approximation implemented in the laplace [11] and ASDL [39] packages. Each marker represents a data example. The estimate roughly maintains the ranking of examples according to their sensitivity. Below each panel, a histogram of true deviations is included to show that the majority of examples have extremely low sensitivity and most of the large sensitivities are attributed to a small fraction of data. The high-sensitivity examples often include interesting cases (possibly mislabeled or simply ambiguous), some of which are visualized in each panel along with some low-sensitivity examples to show the contrast. High-sensitivity examples characterize the model's memory because perturbing them leads to a large change in the model. Similar trends are observed for removal of groups of examples in Fig. 6 of App. I.2.

Predicting the effect of class removal on generalization: Fig. 3(a) shows that the leave-group-out estimates can be used to faithfully predict the test performance even when a whole class is removed. The x-axis shows the test negative log-likelihood (NLL) on a held-out test set, while the y-axis shows the following leave-one-class-out (LOCO) loss on the set C of a left-out class,

$$
\operatorname{LOCO} _ {\mathcal {C}} \left(\boldsymbol {\theta} _ {*}\right) = \sum_ {i \in \mathcal {C}} \ell_ {i} \left(\boldsymbol {\theta} _ {*} ^ {\backslash \mathcal {C}}\right) \approx - \sum_ {i \in \mathcal {C}} \log p \left(y _ {i} \mid \sigma \left(f _ {i} \left(\boldsymbol {\theta} _ {*}\right) + v _ {i *} e _ {i *}\right)\right).
$$

![](images/d6f5b4bfdd91a6883cc02a174b3bce25683fe48f38d625f9408f757a6a1f1a20.jpg)

<details>
<summary>scatter</summary>

| Item       | Test Neg. Log-Likelihood (NLL) | Leave-one-class-out Estimation |
|------------|----------------------------------|--------------------------------|
| Shirt      | 3.0                              | 1.02                           |
| Dress      | 1.5                              | 0.48                           |
| Coat       | 2.0                              | 0.62                           |
| Top        | 2.5                              | 0.42                           |
| Sandal     | 1.0                              | 0.22                           |
| Bag        | 1.0                              | 0.20                           |
| Trouser    | 1.0                              | 0.08                           |
| Ankle-boot | 2.5                              | 0.22                           |
| Sneaker    | 2.5                              | 0.20                           |
| Pullover   | 4.5                              | 0.42                           |
</details>

(a) Predicting the effect of class removal

![](images/2f798d5efef8f00621ebf633a1403bf4291d2a8c2c4bf3894feb8200faced1d3.jpg)  
(b) Evolution of sensitivities during training

Figure 3: Panel (a) shows, in the x-axis, the test NLL of trained models with a class removed. In the y-axis, we show the respective leave-one-class-out (LOCO) estimates. Each marker corresponds to a specific class removed (text indicates class names). Results for two models on FMNIST are shown. Both show good correlation between the test NLL and LOCO estimates; see the dashed lines. Panel (b) shows the evolution of estimated sensitivities during training of LeNet5 on FMNIST. As training progresses, the model becomes more and more sensitive to a small fraction of data.   
![](images/a5037f78c2e6d5e5f1abc64abd9f788dbfb452d68a794e65d0679bc43f86c2cc.jpg)

<details>
<summary>line</summary>

| δ     | Test NLL | LOO-CV |
|-------|----------|--------|
| 10^0  | ~0.08    | ~0.08  |
| 10^1  | ~0.06    | ~0.06  |
| 10^2  | ~0.07    | ~0.07  |
| 10^3  | ~0.2     | ~0.2   |
</details>

(a) MLP on MNIST

![](images/701b82f89645db718b5ed3fc86355cf04f30889999a56efc5caae61b0b19b8fd.jpg)

<details>
<summary>line</summary>

| δ     | Value |
|-------|-------|
| 10^1  | 0.5   |
| 10^2  | 0.3   |
| 10^3  | 0.5   |
</details>

(b) LeNet5 on FMNIST

![](images/f728661d8e9af140e77e9ce9391f19cdf83d37c010cde924001ef95f8cbd9811.jpg)

<details>
<summary>line</summary>

| δ     | Value |
|-------|-------|
| 10^1  | 1.8   |
| 10^2  | 1.2   |
| 10^3  | 1.2   |
</details>

(c) CNN on CIFAR-10   
Figure 4: The test NLL (gray) almost perfectly matches the estimated LOO-CV error of Eq. 15 (black). The x-axis shows different values of $\delta$ parameter of an $L_{2}$ -regularization $\delta \| \pmb{\theta} \|^{2}/2$ .

The estimate uses an approximation: $f_{i}(\pmb{\theta}_{*}^{\backslash \mathcal{C}}) - f_{i}(\pmb{\theta}_{*}) \approx \nabla f_{i}(\pmb{\theta}_{*})^{\top} \mathbf{H}_{*}^{-1} \sum_{j \in \mathcal{C}} \nabla \ell_{j}(\pmb{\theta}_{*}) \approx v_{i*} e_{i*}$ , which is similar to Eq. 12, but uses an additional approximation $\sum_{j \in \mathcal{C}} \nabla \ell_{j}(\pmb{\theta}_{*}) \approx \nabla \ell_{i}(\pmb{\theta}_{*})$ to reduce the computation due to matrix-vector multiplications (we rely on the same K-FAC approximation used in the previous experiment). Results might improve when this approximation is relaxed. We show results for two models: MLP and LeNet. Each marker corresponds to a specific class whose names are indicated with the text. The dashed lines indicate the general trends, showing a good correlation between the truth and estimate. The classes Shirt, Pullover are the most sensitive, while the classes Bag, Trousers are least sensitive. A similar result for MNIST is in Fig. 11(d) of App. I.3.

Predicting generalization for hyperparameter tuning: We consider the tuning of the parameter $\delta$ for the $L_{2}$ -regularizer of form $\delta\|\theta\|^{2}/2$ . Fig. 4 shows an almost perfect match between the test NLL and the estimated LOO-CV error of Eq. 15. Additional figures with the test errors visualized on top are included in Fig. 7 of App. I.4 where we again see a close match to the LOO-CV curves.

Predicting generalization during training: As discussed earlier, existing influence measures are not designed to analyze sensitivity during training and care needs to be taken when using ad-hoc strategies. We first show results for our proposed measure in Eq. 10 which gives reliable sensitivity estimates during training. We use the improved-BLR method [35] which estimates the mean $m_{t}$ and a vector preconditioner $s_{t}$ during training. We can derive an estimate for the LOO error at the mean $m_{t}$ following a derivation similar to Eqs. 14 and 15,

$$
\mathrm{LOO} (\mathbf {m} _ {t}) \approx - \sum_ {i = 1} ^ {N} \log p (y _ {i} | \sigma (f _ {i} (\mathbf {m} _ {t}) + v _ {i t} e _ {i t})) \tag {16}
$$

![](images/c169cb106489877543af7863d1d19e901e719f8017cce98a2fbb19bde6981ae8.jpg)

<details>
<summary>line</summary>

| Epochs | Value |
| ------ | ----- |
| 0      | 0.7   |
| 10     | 0.45  |
| 20     | 0.38  |
| 30     | 0.36  |
| 40     | 0.35  |
| 50     | 0.34  |
| 60     | 0.33  |
| 70     | 0.32  |
| 80     | 0.31  |
| 90     | 0.31  |
| 100    | 0.31  |
</details>

(a) iBLR & LOO of Eq. 16

![](images/8f319ac8dcd7f45a73ca3aef780fa458a5de5f21c8c8cd72e5f6364fbbe6858d.jpg)

<details>
<summary>line</summary>

| Epochs | LOO-CV | Test NLL |
| ------ | ------ | -------- |
| 0      | 0.9    | 0.75     |
| 25     | 0.8    | 0.4      |
| 50     | 0.8    | 0.35     |
| 75     | 0.8    | 0.35     |
| 100    | 0.8    | 0.35     |
</details>

(b) SGD & diagonal-GGN-LOO

![](images/8ea5af3b987f0d3ab1a7c043b244dc61edd5b2f8d755c89641486d6303ea4af7.jpg)

<details>
<summary>line</summary>

| Epochs | Value |
| ------ | ----- |
| 0      | 0.8   |
| 25     | 0.4   |
| 50     | 0.38  |
| 75     | 0.36  |
| 100    | 0.35  |
</details>

(c) SGD & K-FAC-LOO   
Figure 5: We compare faithfulness of LOO estimates during training to predict the test NLL. The first panel shows results for iBLR where a good match is obtained by using the LOO estimate of Eq. 16 which uses a diagonal preconditioner. The next two panels show results for SGD where we use the LOO estimate of Eq. 15 but with different Hessian approximations. Panel (b) uses a diagonal-GGN which does not work very well. Results are improved when K-FAC is used, but they are still not as good as the iBLR, despite using a non-diagonal Hessian approximation.

where $v_{it} = \nabla f_i(\mathbf{m}_t)^\top \mathrm{diag}(\mathbf{s}_t)^{-1}\nabla f_i(\mathbf{m}_t)$ and $e_{it} = \sigma(f_i(\mathbf{m}_t)) - y_i$ .

The first panel in Fig. 5 shows a good match between the above LOO estimate and test NLL. For comparison, in the next two panels, we show results for SGD training by using two ad-hoc measures obtained by plugging different Hessian approximations in Eq. 11. The first panel approximates $H_{t}$ with a diagonal Generalized Gauss-Newton (GGN) matrix, while the second panel uses a K-FAC approximation. We see that diagonal-GGN-LOO does not work well at all and, while K-FAC-LOO improves this, it is still not as good as the iBLR result despite using a non-diagonal Hessian approximation. Not to mention, the two measures require an additional pass through the data to compute the Hessian approximation, and also need a careful setting of a damping parameter.

A similar result for iBLR is shown in Fig. 1(b) where we use the larger ResNet-20 on CIFAR10, and more such results are included in Fig. 8 of App. I.5. We also find that both diagonal-GGN-LOO or K-FAC-LOO further deteriorate when the model overfits; see Fig. 9. Results for the Adam optimizer are included in Fig. 10, where we again see that using ad hoc measures may not always work. Overall, these results show the difficulty of estimating sensitivity during training and suggest to take caution when using measures that are not naturally suited to analyze the training algorithm.

Evolution of sensitivities during training: Fig. 3(b) shows the evolution of sensitivities of examples as the training progresses. We use the iBLR algorithm and approximate the deviation as $\sigma(f_i(\mathbf{m}_t^i)) - \sigma(f_i(\mathbf{m}_t)) \approx \sigma'(f_i(\mathbf{m}_t))v_{it}e_{it}$ where $v_{it}$ and $e_{it}$ are obtained similarly to Eq. 16. The x-axis corresponds to examples sorted from least sensitive to most sensitive examples at convergence. The y-axis shows the histogram of sensitivity estimates. We observe that, as the training progresses, the distribution concentrates around a small fraction of the data. At the top, we visualize a few examples with high and low sensitivity estimates, where the high-sensitivity examples included interesting cases (similarly to Fig. 2). The result suggests that the model concentrates more and more on a small fraction of high-sensitivity examples, and therefore such examples can be used to characterize the model's memory. Additional experiments of this kind are included in Fig. 11 of App. I.6, along with other experiment details.

# 5 Discussion

We present the memory-perturbation equation by building upon the BLR framework. The equation suggests to take a step in the direction of the natural gradient of the perturbed examples. Using the MPE framework, we unify existing influence measures, generalize them to a wide variety of problems, and unravel useful properties regarding sensitivity. We also show that sensitivity estimation can be done cheaply and use this to predict generalization performance. An interesting avenue for future research is to apply the method to larger models and real-world problems. We also need to understand how our generalization measure compares to other methods, such as those considered in $[22]$ . We would also like to understand the effect of various posterior approximations. Another interesting direction is to apply the method to non-Gaussian cases, for example, to study ensemble methods in deep learning with mixture models.

# Acknowledgements

This work is supported by the Bayes duality project, JST CREST Grant Number JPMJCR2112.

# References

[1] Vincent Adam, Paul Chang, Mohammad Emtiyaz Khan, and Arno Solin. Dual Parameterization of Sparse Variational Gaussian Processes. Advances in Neural Information Processing Systems, 2021. 20   
[2] Chirag Agarwal, Daniel D'Souza, and Sara Hooker. Estimating Example Difficulty using Variance of Gradients. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022. 3   
[3] Devansh Arpit, Stanisław Jastrzębski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A Closer Look at Memorization in Deep Networks. In International Conference on Machine Learning, 2017. 3, 6   
[4] Gregor Bachmann, Thomas Hofmann, and Aurélien Lucchi. Generalization Through The Lens of Leave-One-Out Error. In International Conference on Learning Representations, 2022. 2, 7   
[5] Samyadeep Basu, Phil Pope, and Soheil Feizi. Influence Functions in Deep Learning Are Fragile. In International Conference on Learning Representations, 2021. 6   
[6] Christopher M. Bishop. Pattern Recognition and Machine Learning. Springer, 2006. 5, 7   
[7] R Dennis Cook. Detection of Influential Observation in Linear Regression. Technometrics, 1977. 1, 3, 5   
[8] R Dennis Cook and Sanford Weisberg. Characterizations of an Empirical Influence Function for Detecting Influential Cases in Regression. Technometrics, 1980. 1, 2, 3   
[9] R Dennis Cook and Sanford Weisberg. Residuals and Influence in Regression. Chapman and Hall, 1982. 14   
[10] Corinna Cortes and Vladimir Vapnik. Support-Vector Networks. Machine learning, 1995. 4   
[11] Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, and Philipp Hennig. Laplace Redux-Effortless Bayesian Deep Learning. Advances in Neural Information Processing Systems, 2021. 8   
[12] Gintare Karolina Dziugaite and Daniel M Roy. Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters Than Training Data. In Proceedings of the Conference on Uncertainty in Artificial Intelligence, 2017. 2, 7   
[13] Vitaly Feldman and Chiyuan Zhang. What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation. In Advances in Neural Information Processing Systems, 2020. 3   
[14] Wing K Fung and CW Kwan. A Note on Local Influence Based on Normal Curvature. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 1997. 1, 3   
[15] Ryan Giordano, Tamara Broderick, and Michael I Jordan. Covariances, robustness and variational Bayes. Journal of Machine Learning Research, 19(51), 2018. 5   
[16] Satoshi Hara, Atsushi Nitanda, and Takanori Maehara. Data Cleansing for Models Trained with SGD. In Advances in Neural Information Processing Systems, 2019. 3   
[17] Hrayr Harutyunyan, Alessandro Achille, Giovanni Paolini, Orchid Majumder, Avinash Ravichandran, Rahul Bhotika, and Stefano Soatto. Estimating Informativeness of Samples with Smooth Unique Information. In International Conference on Learning Representations, 2021. 3

[18] James Hensman, Nicolo Fusi, and Neil D Lawrence. Gaussian Processes for Big Data. In Proceedings of the Conference on Uncertainty in Artificial Intelligence, 2013. 20   
[19] Alexander Immer, Matthias Bauer, Vincent Fortuin, Gunnar Rätsch, and Mohammad Emtiyaz Khan. Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning. In International Conference on Machine Learning, 2021. 2, 7   
[20] Alexander Immer, Maciej Korzepa, and Matthias Bauer. Improving Predictions of Bayesian Neural Nets via Local Linearization. International Conference on Artificial Intelligence and Statistics, 2021. 7   
[21] Louis A Jaeckel. The Infinitesimal Jackknife. Technical report, Bell Lab., 1972. 1, 3   
[22] Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, and Samy Bengio. Fantastic Generalization Measures and Where To Find Them. In International Conference on Learning Representations, 2020. 2, 7, 10   
[23] Angelos Katharopoulos and Francois Fleuret. Not All Samples Are Created Equal: Deep Learning with Importance Sampling. In International Conference on Machine Learning, 2018. 3   
[24] Mohammad Emtiyaz Khan. Variational Bayes Made Easy. Fifth Symposium on Advances in Approximate Bayesian Inference, 2023. 5   
[25] Mohammad Emtiyaz Khan, Alexander Immer, Ehsan Abedi, and Maciej Korzepa. Approximate Inference Turns Deep Networks into Gaussian Processes. Advances in Neural Information Processing Systems, 2019. 4, 7, 17   
[26] Mohammad Emtiyaz Khan and Wu Lin. Conjugate-Computation Variational Inference: Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models. In International Conference on Artificial Intelligence and Statistics, 2017. 4, 15, 20   
[27] Mohammad Emtiyaz Khan, Didrik Nielsen, Voot Tangkaratt, Wu Lin, Yarin Gal, and Akash Srivastava. Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam. In International Conference on Machine Learning, 2018. 6, 17, 18, 25   
[28] Mohammad Emtiyaz Khan and Håvard Rue. The Bayesian Learning Rule. Journal of Machine Learning Research, 2023. 1, 4, 5, 6, 16, 17, 18   
[29] George S Kimeldorf and Grace Wahba. A Correspondence Between Bayesian Estimation on Stochastic Processes and Smoothing by Splines. The Annals of Mathematical Statistics, 1970. 4   
[30] Diederik Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations, 2015. 6   
[31] Pang Wei Koh, Kai-Siang Ang, Hubert Teo, and Percy S Liang. On the Accuracy of Influence Functions for Measuring Group Effects. In Advances in Neural Information Processing Systems, 2019. 2, 3, 5   
[32] Pang Wei Koh and Percy Liang. Understanding Black-Box Predictions via Influence Functions. In International Conference on Machine Learning, 2017. 1, 3, 5, 7   
[33] Aran Komatsuzaki. One Epoch is All You Need. ArXiv e-Prints, 2019. 3   
[34] Pierre-Simon Laplace. Mémoires de Mathématique et de Physique. Tome Sixieme, 1774. 5   
[35] Wu Lin, Mark Schmidt, and Mohammad Emtiyaz Khan. Handling the Positive-Definite Constraint in the Bayesian Learning Rule. In International Conference on Machine Learning, 2020. 6, 9, 17, 18   
[36] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. International Conference on Learning Representations, 2019. 23

[37] David JC MacKay. Information Theory, Inference and Learning Algorithms. Cambridge University Press, 2003. 5   
[38] Roman Novak, Yasaman Bahri, Daniel A Abolafia, Jeffrey Pennington, and Jascha Sohl-Dickstein. Sensitivity and Generalization in Neural Networks: An Empirical Study. In International Conference on Learning Representations, 2018. 1, 2, 3   
[39] Kazuki Osawa, Satoki Ishikawa, Rio Yokota, Shigang Li, and Torsten Hoefler. ASDL: A Unified Interface for Gradient Preconditioning in PyTorch. In NeurIPS Workshop Order up! The Benefits of Higher-Order Optimization in Machine Learning, 2023. 8   
[40] Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep Learning on a Data Diet: Finding Important Examples Early in Training. In Advances in Neural Information Processing Systems, 2021. 3, 6, 7   
[41] Daryl Pregibon. Logistic Regression Diagnostics. The Annals of Statistics, 1981. 14   
[42] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating Training Data Influence by Tracing Gradient Descent. In Advances in Neural Information Processing Systems, 2020. 3, 6   
[43] Kamiar Rahnama Rad and Arian Maleki. A Scalable Estimate of the Out-of-Sample Prediction Error via Approximate Leave-One-Out Cross-Validation. Journal of the Royal Statistical Society Series B: Statistical Methodology, 2020. 7   
[44] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. In International Conference on Machine Learning, 2014. 7, 17   
[45] Hugh Salimbeni, Stefanos Eleftheriadis, and James Hensman. Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models. In International Conference on Artificial Intelligence and Statistics, 2018. 20   
[46] Frank Schneider, Lukas Balles, and Philipp Hennig. DeepOBS: A Deep Learning Optimizer Benchmark Suite. In International Conference on Learning Representations, 2019. 21   
[47] Bernhard Schölkopf, Ralf Herbrich, and Alex J Smola. A Generalized Representer Theorem. In International Conference on Computational Learning Theory, 2001. 4   
[48] Saurabh Singh and Shankar Krishnan. Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020. 21   
[49] Ryutaro Tanno, Melanie F Pradier, Aditya Nori, and Yingzhen Li. Repairing Neural Networks by Leaving the Right Past Behind. Advances in Neural Information Processing Systems, 2022. 5   
[50] Luke Tierney and Joseph B Kadane. Accurate Approximations for Posterior Moments and Marginal Densities. Journal of the American Statistical Association, 1986. 5   
[51] Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J. Gordon. An Empirical Study of Example Forgetting during Deep Neural Network Learning. In International Conference on Learning Representations, 2019. 3, 6   
[52] Robert Weiss. An Approach to Bayesian Sensitivity Analysis. Journal of the Royal Statistical Society Series B: Statistical Methodology, 1996. 3   
[53] Fuzhao Xue, Yao Fu, Wangchunshu Zhou, Zangwei Zheng, and Yang You. To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis. ArXiv e-Prints, 2023. 3   
[54] Hongtu Zhu, Joseph G. Ibrahim, Sikyum Lee, and Heping Zhang. Perturbation Selection and Influence Measures in Local Influence Analysis. The Annals of Statistics, 2007. 1

# A Influence Function for Linear Regression

We consider $N$ input-output pairs $(\mathbf{x}_i, y_i)$ . The feature matrix containing $\mathbf{x}_i^\top$ as rows is denoted by $\mathbf{X}$ and the output vector of length $N$ is denoted by $\mathbf{y}$ . The loss is $\ell_i(\boldsymbol{\theta}) = \frac{1}{2}(y_i - f_i(\boldsymbol{\theta}))^2$ for $f_i(\boldsymbol{\theta}) = \mathbf{x}_i^\top \boldsymbol{\theta}$ . The regularizer is assumed to be $\mathcal{R}(\boldsymbol{\theta}) = \delta \| \boldsymbol{\theta} \|^2 / 2$ . The minimizer is given by

$$
\boldsymbol {\theta} _ {*} = \mathbf {H} _ {*} ^ {- 1} \mathbf {X} ^ {\top} \mathbf {y}. \tag {17}
$$

We define a perturbation model as follows with $\epsilon_{i}\in\mathbb{R}$ :

$$
\boldsymbol {\theta} _ {*} ^ {\epsilon_ {i}} = \underset {\boldsymbol {\theta}} {\arg \min} \mathcal {L} (\boldsymbol {\theta}) - \epsilon_ {i} \ell_ {i} (\boldsymbol {\theta}).
$$

For $\epsilon_{i} = 1$ , it corresponds to example removal. An arbitrary $\epsilon_{i}$ simply weights the example accordingly. The solution has a closed-form expression,

$$
\boldsymbol {\theta} _ {*} ^ {\epsilon_ {i}} = \left(\mathbf {H} _ {*} - \epsilon_ {i} \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top}\right) ^ {- 1} \left(\mathbf {X} ^ {\top} \mathbf {y} - \epsilon_ {i} \mathbf {x} _ {i} y _ {i}\right). \tag {18}
$$

where $\mathbf{H}_{*} = \sum_{i=1}^{N}\mathbf{x}_{i}\mathbf{x}_{i}^{\top} + \delta\mathbf{I}_{P}$ is the Hessian of $\mathcal{L}(\boldsymbol{\theta})$ . We first derive a closed-form expressions for $\boldsymbol{\theta}_{*}^{\epsilon_{i}} - \boldsymbol{\theta}_{*}$ , and then specialize them for different $\epsilon_{i}$ .

# A.1 Derivation of the leave-one-out (LOO) deviation

We denote $\Sigma_{*} = \mathbf{H}_{*}^{-1}$ and use the Sherman-Morrison formula to write

$$
\begin{array}{l} \boldsymbol {\theta} _ {*} ^ {\epsilon_ {i}} = \left(\boldsymbol {\Sigma} _ {*} + \frac {\epsilon_ {i} \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top} \boldsymbol {\Sigma} _ {*}}{1 - \epsilon_ {i} \mathbf {x} _ {i} ^ {\top} \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i}}\right) \left(\mathbf {X} ^ {\top} \mathbf {y} - \epsilon_ {i} y _ {i} \mathbf {x} _ {i}\right) \\ = \boldsymbol {\Sigma} _ {*} \mathbf {X} ^ {\top} \mathbf {y} + \epsilon_ {i} \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i} \left[ \frac {\mathbf {x} _ {i} ^ {\top} \boldsymbol {\Sigma} _ {*} \mathbf {X} ^ {\top} \mathbf {y}}{1 - \epsilon_ {i} \mathbf {x} _ {i} ^ {\top} \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i}} - \frac {\epsilon_ {i} y _ {i} \mathbf {x} _ {i} ^ {\top} \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i}}{1 - \epsilon_ {i} \mathbf {x} _ {i} ^ {\top} \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i}} - y _ {i} \right] \tag {19} \\ = \pmb {\theta} _ {*} + \epsilon_ {i} \pmb {\Sigma} _ {*} \mathbf {x} _ {i} \left[ \frac {\mathbf {x} _ {i} ^ {\top} \pmb {\theta} _ {*}}{1 - \epsilon_ {i} v _ {i}} - \frac {\epsilon_ {i} y _ {i} v _ {i}}{1 - \epsilon_ {i} v _ {i}} - y _ {i} \right] \\ = \boldsymbol {\theta} _ {*} + \epsilon_ {i} \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i} \left[ \frac {\mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} _ {*} - y _ {i}}{1 - \epsilon_ {i} v _ {i}} \right] = \boldsymbol {\theta} _ {*} + \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i} \frac {\epsilon_ {i} e _ {i}}{1 - \epsilon_ {i} v _ {i}}. \\ \end{array}
$$

In line 3 we substitute $v_{i} = \mathbf{x}_{i}^{\top}\boldsymbol{\Sigma}_{*}\mathbf{x}_{i}$ and $\boldsymbol{\theta}_{*} = \boldsymbol{\Sigma}_{*}\mathbf{X}^{\top}\mathbf{y}$ and in the last step we use $e_{i} = \mathbf{x}_{i}^{\top}\boldsymbol{\theta}_{*} - y_{i}$ . We define $e_{i}^{\backslash i} = e_{i} / (1 - v_{i})$ which is the prediction error of $\boldsymbol{\theta}_{*}^{\backslash i}$ ,

$$
e _ {i} ^ {\backslash i} = \mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} _ {*} ^ {\backslash i} - y _ {i} = \mathbf {x} _ {i} ^ {\top} \left(\boldsymbol {\theta} _ {*} + \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i} \frac {e _ {i}}{1 - v _ {i}}\right) - y _ {i} = \mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} _ {*} + \frac {v _ {i}}{1 - v _ {i}} e _ {i} - y _ {i} = \frac {e _ {i}}{1 - v _ {i}}. \tag {20}
$$

Therefore, we get the following expressions for the deviation,

$$
\boldsymbol {\theta} _ {*} ^ {\backslash i} - \boldsymbol {\theta} _ {*} = \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i} e _ {i} ^ {\backslash i}, \quad f _ {i} (\boldsymbol {\theta} _ {*} ^ {\backslash i}) - f _ {i} (\boldsymbol {\theta} _ {*}) = v _ {i} e _ {i} ^ {\backslash i}.
$$

These expressions can be written in the form of Eq. 2 by left-multiplying with $\Sigma_{*}^{-1} = \mathbf{H}_{*}^{\backslash i} + \mathbf{x}_i\mathbf{x}_i^\top$ ,

$$
(\mathbf {H} _ {*} ^ {\backslash i} + \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top}) (\boldsymbol {\theta} _ {*} ^ {\backslash i} - \boldsymbol {\theta} _ {*}) = \mathbf {x} _ {i} (\mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} _ {*} ^ {\backslash i} - y _ {i}) \Rightarrow \boldsymbol {\theta} _ {*} ^ {\backslash i} - \boldsymbol {\theta} _ {*} = (\mathbf {H} _ {*} ^ {\backslash i}) ^ {- 1} \mathbf {x} _ {i} e _ {i}.
$$

# A.2 Derivation of the infinitesimal perturbation approach

We differentiate $\theta_{*}^{\epsilon_{i}}$ in Eq. 19 to get

$$
\frac {\partial \boldsymbol {\theta} _ {*} ^ {\epsilon_ {i}}}{\partial \epsilon_ {i}} = \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i} \frac {e _ {i}}{(1 - \epsilon_ {i} v _ {i}) ^ {2}}, \tag {21}
$$

yielding the following expressions:

$$
\left. \frac {\partial \boldsymbol {\theta} _ {*} ^ {\epsilon_ {i}}}{\partial \epsilon_ {i}} \right| _ {\epsilon_ {i} = 0} = \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i} e _ {i}, \quad \left. \frac {\partial f _ {i} \left(\boldsymbol {\theta} _ {*} ^ {\epsilon_ {i}}\right)}{\partial \epsilon_ {i}} \right| _ {\epsilon_ {i} = 0} = \mathbf {x} _ {i} ^ {\top} \left. \frac {\partial \boldsymbol {\theta} _ {*} ^ {\epsilon_ {i}}}{\partial \epsilon_ {i}} \right| _ {\epsilon_ {i} = 0} = \mathbf {x} _ {i} ^ {\top} \boldsymbol {\Sigma} _ {*} \mathbf {x} _ {i} e _ {i} = v _ {i} e _ {i}. \tag {22}
$$

The second equation in Eq. 22 follows from the chain rule. We get a bi-linear relationship of the influence measure with respect to $v_{i}$ and prediction error $e_{i}$ . It is also possible to evaluate Eq. 21 at $\epsilon_{i} = 1$ representing an infinitesimal perturbation about the LOO estimate, $\partial \pmb{\theta}_{*}^{\epsilon_i} / \partial \epsilon_i|_{\epsilon_i = 1} = \pmb{\Sigma}_{*}^{\backslash i}\mathbf{x}_ie_i^{\backslash i}$ . From this and Eq. 22, we can interpret Eq. 2 as the average derivative over the interval $\epsilon_{i}\in [0,1]$ [9] or the derivative evaluated at some $0 < \epsilon_{i} < 1$ (via an application of the mean value theorem) [41].

# B Conjugate Exponential-Family Models

Exponential-family distributions take the following form:

$$
q = h (\boldsymbol {\theta}) \exp \left[ \langle \boldsymbol {\lambda}, \mathbf {T} (\boldsymbol {\theta}) \rangle - A (\boldsymbol {\lambda}) \right].
$$

where $\lambda\in\Omega$ are the natural (or canonical) parameter for which the cumulant (or log partition) function $A(\lambda)$ is finite, strictly convex and differentiable over $\Omega$ . The quantity $\mathbf{T}(\boldsymbol{\theta})$ is the sufficient statistics, $\langle\cdot,\cdot\rangle$ is an inner product and $h(\boldsymbol{\theta})$ is some function. A popular example is the Gaussian distribution, which can be rearranged to take an exponential-family form written in terms of the precision matrix $S=\Sigma^{-1}$ ,

$$
\begin{array}{l} \mathcal {N} (\pmb {\theta} | \mathbf {m}, \pmb {\Sigma}) = | 2 \pi \pmb {\Sigma} | ^ {- \frac {1}{2}} \exp \left[ - \frac {1}{2} (\pmb {\theta} - \mathbf {m}) ^ {\top} \pmb {\Sigma} ^ {- 1} (\pmb {\theta} - \mathbf {m}) \right] \\ = \exp \left[ \boldsymbol {\theta} ^ {\top} \mathbf {S m} - \frac {1}{2} \boldsymbol {\theta} ^ {\top} \mathbf {S} \boldsymbol {\theta} - \frac {1}{2} \left(\mathbf {m} ^ {\top} \mathbf {S m} + \log | 2 \pi \mathbf {S} ^ {- 1} |\right) \right]. \\ \end{array}
$$

From this, we can read-off the quantities needed to define an exponential-form,

$$
\boldsymbol {\lambda} = (\mathbf {S m}, - \frac {1}{2} \mathbf {S}), \quad \mathbf {T} (\boldsymbol {\theta}) = (\boldsymbol {\theta}, \boldsymbol {\theta} \boldsymbol {\theta} ^ {\top}), \quad A (\boldsymbol {\lambda}) = \frac {1}{2} \left(\mathbf {m} ^ {\top} \mathbf {S m} + \log | 2 \pi \mathbf {S} ^ {- 1} |\right), \quad h (\boldsymbol {\theta}) = 1. \tag {23}
$$

Both the natural parameter and sufficient statistics consist of two elements. The inner-product for the first elements is simply a transpose to get the $\theta^{\top}Sm$ term, while for the second element it is a trace which gives $-\mathrm{Tr}(\boldsymbol{\theta}\boldsymbol{\theta}^{\top}\mathbf{S}/2)=-\frac{1}{2}\boldsymbol{\theta}^{\top}\mathbf{S}\boldsymbol{\theta}$ .

Conjugate Exponential-Family Models are those where both the likelihoods and prior can be expressed in terms of the same form of exponential-family distribution with respect to $\theta$ . For instance, in linear regression, both the likelihood and prior take a Gaussian form with respect to $\theta$ ,

$$
\tilde {p} _ {i} = p (y _ {i} | \mathbf {x} _ {i}, \boldsymbol {\theta}) = \mathcal {N} (y _ {i} | \mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta}, 1) \propto \exp \left[ \boldsymbol {\theta} ^ {\top} \mathbf {x} _ {i} y _ {i} - \frac {1}{2} \boldsymbol {\theta} ^ {\top} \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} \right]
$$

$$
p _ {0} = p (\boldsymbol {\theta}) = \mathcal {N} (\boldsymbol {\theta} | 0, \mathbf {I} / \delta) \propto \exp \left[ - \frac {1}{2} \boldsymbol {\theta} ^ {\top} (\delta \mathbf {I}) \boldsymbol {\theta} \right].
$$

Note that $\tilde{p}_{i}$ is a distribution over $y_{i}$ but it can also be expressed in an (unnormalized) Gaussian form with respect to $\theta$ . The sufficient statistics of both $\tilde{p}_{i}$ and $p_{0}$ correspond to those of a Gaussian distribution. Therefore, the posterior is also a Gaussian,

$$
\begin{array}{l} q _ {*} = p (\boldsymbol {\theta} | \mathcal {D}) \propto p _ {0} \tilde {p} _ {1} \tilde {p} _ {2} \dots \tilde {p} _ {N} \\ = \exp \left[ - \frac {1}{2} \boldsymbol {\theta} ^ {\top} (\delta \mathbf {I}) \boldsymbol {\theta} \right] \prod_ {i = 1} ^ {N} \exp \left[ \boldsymbol {\theta} ^ {\top} \mathbf {x} _ {i} y _ {i} - \frac {1}{2} \boldsymbol {\theta} ^ {\top} \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} \right] \\ = \exp \left[ \boldsymbol {\theta} ^ {\top} \sum_ {i = 1} ^ {N} \mathbf {x} _ {i} y _ {i} - \frac {1}{2} \boldsymbol {\theta} ^ {\top} \left(\delta \mathbf {I} + \sum_ {i = 1} ^ {N} \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top}\right) \boldsymbol {\theta} \right] \\ = \exp \left[ \boldsymbol {\theta} ^ {\top} \mathbf {H} _ {*} \boldsymbol {\theta} _ {*} - \frac {1}{2} \boldsymbol {\theta} ^ {\top} \mathbf {H} _ {*} \boldsymbol {\theta} \right] \\ \propto \mathcal {N} (\boldsymbol {\theta} | \boldsymbol {\theta} _ {*}, \mathbf {H} _ {*} ^ {- 1}). \\ \end{array}
$$

The third line follows because $\theta_{*} = H_{*}^{-1}X^{\top}y$ , as shown in Eq. 17.

These computations can be written as conjugate-computations [26] where we simply add the natural parameters,

$$
\begin{array}{l} \tilde {p} _ {i} \propto \exp \left[ \langle \widetilde {\boldsymbol {\lambda}} _ {i}, \mathbf {T} (\boldsymbol {\theta}) \rangle \right], \text {   where   } \widetilde {\boldsymbol {\lambda}} _ {i} = (\mathbf {x} _ {i} y _ {i}, - \frac {1}{2} \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top}) \\ p _ {0} \propto \exp \left[ \langle \boldsymbol {\lambda} _ {0}, \mathbf {T} (\boldsymbol {\theta}) \rangle \right], \text {   where   } \boldsymbol {\lambda} _ {0} = (0, - \frac {1}{2} \delta \mathbf {I}) \\ \implies q _ {*} \propto \exp \left[ \langle \boldsymbol {\lambda} _ {*}, \mathbf {T} (\boldsymbol {\theta}) \rangle \right], \text {   where   } \boldsymbol {\lambda} _ {*} = \boldsymbol {\lambda} _ {0} + \sum_ {i = 1} ^ {N} \widetilde {\boldsymbol {\lambda}} _ {i} = \left(\mathbf {H} _ {*} \boldsymbol {\theta} _ {*}, - \frac {1}{2} \mathbf {H} _ {*}\right). \\ \end{array}
$$

In the same fashion, to remove the contributions of certain likelihoods, we can simply subtract their natural parameters from $\lambda_{*}$ . These are the calculations which give rise to the following equation:

$$
q _ {*} ^ {\backslash \mathcal {M}} \propto \frac {q _ {*}}{\prod_ {j \in \mathcal {M}} \tilde {p} _ {j}} \quad \Longrightarrow e ^ {\langle \mathbf {T} (\boldsymbol {\theta}), \boldsymbol {\lambda} _ {*} ^ {\backslash \mathcal {M}} \rangle} \propto \frac {e ^ {\langle \mathbf {T} (\boldsymbol {\theta}) , \boldsymbol {\lambda} _ {*} \rangle}}{\prod_ {j \in \mathcal {M}} e ^ {\langle \mathbf {T} (\boldsymbol {\theta}) , \widetilde {\boldsymbol {\lambda}} _ {j} \rangle}} \quad \Longrightarrow \boldsymbol {\lambda} _ {*} ^ {\backslash \mathcal {M}} = \boldsymbol {\lambda} _ {*} - \sum_ {j \in \mathcal {M}} \widetilde {\boldsymbol {\lambda}} _ {j}.
$$

# C The Bayesian Learning Rule

The Bayesian learning rule (BLR) aims to find a posterior approximation $q(\boldsymbol{\theta}) \approx p(\boldsymbol{\theta}|\mathcal{D}) \propto e^{-\mathcal{L}(\boldsymbol{\theta})}$ . Often, one considers a regular, minimal exponential-family $q \in \mathcal{Q}$ , for example, the class of Gaussian distributions. The approximation is found by optimizing a generalized Bayesian objective,

$$
q _ {*} = \underset {q \in \mathcal {Q}} {\arg \min} \mathbb {E} _ {q} \left[ \mathcal {L} (\boldsymbol {\theta}) \right] - \mathcal {H} (q).
$$

where $\mathcal{H}(q)=\mathbb{E}_{q}[-\log q(\boldsymbol{\theta})]$ is the entropy of q and Q is the class of exponential family approximation. The objective is equivalent to the Evidence Lower Bound (ELBO) when $\mathcal{L}(\boldsymbol{\theta})$ corresponds to the negative log-joint probability of a Bayesian model; see [28, Sec 1.2].

The BLR uses natural-gradient descent to find $q_{*}$ , where each iteration t takes the following form,

$$
\boldsymbol {\lambda} _ {t} \leftarrow \boldsymbol {\lambda} _ {t - 1} - \rho \mathbf {F} (\boldsymbol {\lambda} _ {t - 1}) ^ {- 1} \frac {\partial}{\partial \boldsymbol {\lambda}} \left[ \mathbb {E} _ {q} [ \mathcal {L} (\boldsymbol {\theta}) ] - \mathcal {H} (q) \right] \Bigg | _ {\boldsymbol {\lambda} = \boldsymbol {\lambda} _ {t - 1}} \tag {24}
$$

where $\rho > 0$ is the learning rate. The gradient is computed with respect to $\lambda$ (through $q$ ), and we scale the gradient by the Fisher Information Matrix (FIM) defined as follows,

$$
\mathbf {F} (\pmb {\lambda}) = \mathbb {E} _ {q} \left[ (\nabla_ {\pmb {\lambda}} \log q) (\nabla_ {\pmb {\lambda}} \log q) ^ {\top} \right] = \nabla_ {\pmb {\lambda}} ^ {2} A (\pmb {\lambda}).
$$

The second equality shows that, for exponential-family distribution, the above FIM is also the second derivative of the log-partition function $A(\lambda)$ .

# C.1 The BLR of Eq. 5

The BLR in Eq. 5 is obtained by simplifying the natural-gradient using the following identity,

$$
\mathbf {F} (\boldsymbol {\lambda}) ^ {- 1} \nabla_ {\boldsymbol {\lambda}} \mathbb {E} _ {q} (\cdot) = \left. \nabla_ {\boldsymbol {\mu}} \mathbb {E} _ {q} (\cdot) \right| _ {\boldsymbol {\mu} = \nabla_ {\boldsymbol {\lambda}} A (\boldsymbol {\lambda})} \tag {25}
$$

where $\mu$ is the expectation parameter. The identity works because of the minimality of the exponential-family which ensures that there is a one-to-one mapping between $\lambda$ and $\mu$ , and also that the FIM is invertible. Using this, we can show that the natural gradient of $\mathcal{H}(q)$ is simply equal to $-\lambda$ ; see [28, App. B]. Defining $\ell_{0}(\boldsymbol{\theta}) = \mathcal{R}(\boldsymbol{\theta})$ , we get the version of the BLR shown in Eq. 5,

$$
\boldsymbol {\lambda} _ {t} \leftarrow (1 - \rho) \boldsymbol {\lambda} _ {t - 1} - \rho \sum_ {j = 0} ^ {N} \tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda} _ {t - 1}), \text {   where   } \tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda} _ {t - 1}) = \nabla_ {\boldsymbol {\mu}} \mathbb {E} _ {q} [ \ell_ {j} (\boldsymbol {\theta}) ] | _ {\boldsymbol {\mu} = \nabla_ {\boldsymbol {\lambda}} A (\boldsymbol {\lambda} _ {t - 1})}.
$$

# C.2 The conjugate-model form of the BLR given in Eq. 5

To express the update in terms of the posterior of a conjugate model, we simply take the inner product with $\mathbf{T}(\boldsymbol{\theta})$ and take the exponential to write the update as

$$
\underbrace {e ^ {\langle \boldsymbol {\lambda} _ {t} , \mathbf {T} (\boldsymbol {\theta}) \rangle}} _ {\propto q _ {t}} \leftarrow \left(\underbrace {e ^ {\langle \boldsymbol {\lambda} _ {t - 1} , \mathbf {T} (\boldsymbol {\theta}) \rangle}} _ {\propto q _ {t - 1}}\right) ^ {1 - \rho} \left(\underbrace {e ^ {\langle - \tilde {\mathbf {g}} _ {0} (\boldsymbol {\lambda} _ {t - 1}) , \mathbf {T} (\boldsymbol {\theta}) \rangle}} _ {\propto p _ {0}}\right) ^ {\rho} \prod_ {j = 1} ^ {N} e ^ {\langle - \rho \tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda} _ {t - 1}), \mathbf {T} (\boldsymbol {\theta}) \rangle}, \tag {26}
$$

The simplification of the second term on the left to $p_{0}$ happens when $p_{0}$ is a conjugate prior, that is, $p_{0} \propto \exp(\langle\boldsymbol{\lambda}_{0}, \mathbf{T}(\boldsymbol{\theta})\rangle)$ for some $\lambda_{0}$ (see an example in App. B where we show that $L_{2}$ regularizer leads to such a choice). In such cases, we can simplify,

$$
\langle - \tilde {\mathbf {g}} _ {0} (\boldsymbol {\lambda}), \mathbf {T} (\boldsymbol {\theta}) \rangle = \left\langle \nabla_ {\boldsymbol {\mu}} \mathbb {E} _ {q} [ \log p _ {0} ], \mathbf {T} (\boldsymbol {\theta}) \right\rangle = \left\langle \nabla_ {\boldsymbol {\mu}} \left\langle \boldsymbol {\lambda} _ {0}, \boldsymbol {\mu} \right\rangle , \mathbf {T} (\boldsymbol {\theta}) \right\rangle = \left\langle \boldsymbol {\lambda} _ {0}, \mathbf {T} (\boldsymbol {\theta}) \right\rangle = \log p _ {0} + \text { const }.
$$

Using this in Eq. 26, we recover the conjugate model given in Eq. 5.

# C.3 BLR for a Gaussian q and the Variational Online Newton (VON) algorithm

By choosing an appropriate form for $q_{t}$ and making necessary approximations to $\tilde{\mathbf{g}}_j$ , the BLR can recover many popular algorithms as special cases. We will now give a few examples for the case of a Gaussian $q_{t} = \mathcal{N}(\pmb{\theta}|\mathbf{m}_{t},\pmb{\Sigma}_{t})$ which enables derivation of various first and second-order optimization algorithms, such as, Newton's method, RMSprop, Adam, and SGD.

As shown in Eq. 23, for a Gaussian $\mathcal{N}(\pmb{\theta}|\mathbf{m},\pmb{\Sigma})$ , the natural parameter and sufficient statistics are shown below, along with the expectation parameters $\pmb{\mu} = \mathbb{E}_q[\mathbf{T}(\pmb{\theta})]$ .

$$
\boldsymbol {\lambda} = (\mathbf {S m}, - \frac {1}{2} \mathbf {S}), \quad \mathbf {T} (\boldsymbol {\theta}) = (\boldsymbol {\theta}, \boldsymbol {\theta} \boldsymbol {\theta} ^ {\top}), \quad \boldsymbol {\mu} = (\mathbf {m}, \mathbf {m m} ^ {\top} + \boldsymbol {\Sigma}),
$$

Using these, we can write the natural gradients as gradients with respect to $\mu$ , , and then using chain-rule to express them as gradients with respect to $\mathbf{m}$ and $\Sigma$ ,

$$
\tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda}) = \nabla_ {\boldsymbol {\mu}} \mathbb {E} _ {q} [ \ell_ {j} (\boldsymbol {\theta}) ] = \binom{\nabla_ {\mathbf {m}} \mathbb {E} _ {q} [ \ell_ {j} (\boldsymbol {\theta}) ]}{\nabla_ {\mathbf {m m} ^ {\top} + \boldsymbol {\Sigma}} \mathbb {E} _ {q} [ \ell_ {j} (\boldsymbol {\theta}) ]} = \binom{\hat {\mathbf {g}} _ {j} - \hat {\mathbf {H}} _ {j} \mathbf {m}}{\frac {1}{2} \hat {\mathbf {H}} _ {j}}, \tag {27}
$$

where in the last equation we define two quantities written in terms of $\nabla \ell_j(\pmb{\theta})$ and $\nabla^2\ell_j(\pmb{\theta})$ by using Price's and Bonnet's theorem [44],

$$
\hat {\mathbf {g}} _ {j} = \nabla_ {\mathbf {m}} \mathbb {E} _ {q} [ \ell_ {j} (\boldsymbol {\theta}) ] = \mathbb {E} _ {q} [ \nabla \ell_ {j} (\boldsymbol {\theta}) ], \quad \hat {\mathbf {H}} _ {j} = 2 \nabla_ {\boldsymbol {\Sigma}} \mathbb {E} _ {q} [ \ell_ {j} (\boldsymbol {\theta}) ] = \mathbb {E} _ {q} [ \nabla^ {2} \ell_ {j} (\boldsymbol {\theta}) ]. \tag {28}
$$

Plugging these into the BLR update gives us the following update,

$$
\mathbf {S} _ {t} \mathbf {m} _ {t} \leftarrow (1 - \rho) \mathbf {S} _ {t - 1} \mathbf {m} _ {t - 1} + \rho \sum_ {j = 0} ^ {N} \left(\hat {\mathbf {H}} _ {j, t - 1} \mathbf {m} _ {t - 1} - \hat {\mathbf {g}} _ {j, t - 1}\right), \quad \mathbf {S} _ {t} \leftarrow (1 - \rho) \mathbf {S} _ {t - 1} + \rho \sum_ {j = 0} ^ {N} \hat {\mathbf {H}} _ {j, t - 1}
$$

where $\hat{g}_{j,t-1}$ and $\hat{H}_{j,t-1}$ are quantities similar to before but now evaluated at the $q_{t-1}$ . The conjugate model can be written as follows,

$$
q _ {t} \propto e ^ {\pmb {\theta} ^ {\top} \mathbf {S} _ {t} \mathbf {m} _ {t} - \frac {1}{2} \pmb {\theta} ^ {\top} \mathbf {S} _ {t} \pmb {\theta}} \propto (q _ {t - 1}) ^ {1 - \rho} (p _ {0}) ^ {\rho} \prod_ {j = 1} ^ {N} e ^ {\pmb {\theta} ^ {\top} \hat {\mathbf {i}} _ {j, t - 1} - \frac {1}{2} \pmb {\theta} ^ {\top} \hat {\mathbf {i}} _ {j, t - 1} \pmb {\theta}}
$$

The prior above is Gaussian and defined using $q_{t-1}$ and $p_{0}$ . The model uses likelihoods that are Gaussian distribution with information vector $\hat{\mathbf{i}}_{j,t-1} = \rho(\mathbf{H}_{j,t-1}\mathbf{m}_{t-1} - \hat{\mathbf{g}}_{j,t-1})$ and information matrix $\hat{I}_{j,t-1} = \rho\hat{H}_{j,t-1}$ . The likelihood is allowed to be an improper distribution, meaning that its integral is not one. This is not a problem as long as $S_{t}$ remains positive definite. A valid $S_{t}$ can be ensured by either using a Generalized Gauss-Newton approximation to the Hessian [27] or by using the improved BLR of [35]. The former strategy is used in [25] to express BLR iterations as linear models and Gaussian processes. Ultimately, we want to ensure that perturbation in the approximate likelihoods in $q_{t}$ yields a valid posterior and, as long as this is the case, the conjugate model can be used safely. For instance, in Thm. 4, this issue poses no problem at all.

The BLR update can be rearranged and written in a Newton-like form show below,

$$
\text { VON: } \quad \mathbf {m} _ {t} \leftarrow \mathbf {m} _ {t - 1} - \rho \mathbf {S} _ {t} ^ {- 1}   \mathbb {E} _ {q _ {t - 1}} \left[ \nabla \mathcal {L} (\boldsymbol {\theta}) \right], \qquad \mathbf {S} _ {t} \leftarrow (1 - \rho) \mathbf {S} _ {t - 1} + \rho   \mathbb {E} _ {q _ {t - 1}} \left[ \nabla^ {2} \mathcal {L} (\boldsymbol {\theta}) \right]. \tag {29}
$$

This is called the Variational Online Newton (VON) algorithm. A full derivation is in [27] with details on many of its variants in [28]. The simplest variant is the Online Newton (ON) algorithm, where we use the delta method,

$$
\mathbb {E} _ {q _ {t}} \left[ \nabla \mathcal {L} (\boldsymbol {\theta}) \right] \approx \nabla \mathcal {L} (\mathbf {m} _ {t}), \quad \mathbb {E} _ {q _ {t}} \left[ \nabla^ {2} \mathcal {L} (\boldsymbol {\theta}) \right] \approx \nabla^ {2} \mathcal {L} (\mathbf {m} _ {t}). \tag {30}
$$

Then denoting $\mathbf{m}_t = \theta_t$ , we get the following ON update,

$$
\text { ON: } \quad \boldsymbol {\theta} _ {t} \leftarrow \boldsymbol {\theta} _ {t - 1} - \rho \mathbf {S} _ {t} ^ {- 1} \nabla \mathcal {L} (\boldsymbol {\theta} _ {t - 1}), \qquad \mathbf {S} _ {t} \leftarrow (1 - \rho) \mathbf {S} _ {t - 1} + \rho \nabla^ {2} \mathcal {L} (\boldsymbol {\theta} _ {t - 1}). \tag {31}
$$

To reduce the cost, we can use a diagonal approximation $\mathbf{S}_t = \mathrm{diag}(\mathbf{s}_t)$ where $\mathbf{s}_t$ is a scale vector. Additionally, we can use minibatching to estimate the gradient and hessian (denoted by $\hat{\nabla}$ and $\hat{\nabla}^2$ ),

$$
\text { ON   (diagonal + minibatch): } \boldsymbol {\theta} _ {t} \leftarrow \boldsymbol {\theta} _ {t - 1} - \rho \mathbf {s} _ {t} ^ {- 1} \cdot \hat {\nabla} \mathcal {L} (\boldsymbol {\theta} _ {t - 1}), \tag {32}
$$

$$
\mathbf {s} _ {t} \leftarrow (1 - \rho) \mathbf {s} _ {t - 1} + \rho \mathrm{diag} (\hat {\nabla} ^ {2} \mathcal {L} (\boldsymbol {\theta} _ {t - 1})),
$$

where $\cdot$ indicates element-wise product two vectors and $\text{diag}(\cdot)$ extracts the diagonal of a matrix.

Several optimization algorithms can be obtained as special cases from the above variants. For example, to get Newton's method, we set $\rho = 1$ in ON to get

$$
\boldsymbol {\theta} _ {t} \leftarrow \boldsymbol {\theta} _ {t - 1} - \left[ \nabla^ {2} \mathcal {L} (\boldsymbol {\theta} _ {t - 1}) \right] ^ {- 1} \nabla \mathcal {L} (\boldsymbol {\theta} _ {t - 1}). \tag {33}
$$

RMSprop and Adam can be derived in a similar fashion [28].

In our experiments, we use the improved BLR or iBLR optimizer [35]. We use it to implement an improved version of VON [27, Eqs. 7–8] which ensures that the covariance is always positive-definite, even when the Hessian estimates are not. We use diagonal approximation $\mathbf{S}_{t} = \mathrm{diag}(\boldsymbol{\sigma}^{2})^{-1}$ , momentum and minibatching as proposed in [27, 35]. For learning rate $\alpha_{t} > 0$ , momentum $\beta_{1}, \beta_{2} \in [0, 1)$ the iterations are written as follows:

iBLR: $\mathbf{g}_t\gets \beta_1\mathbf{g}_{t - 1} + (1 - \beta_1)\widehat{\mathbf{g}}_{t - 1},$

$$
\mathbf {h} _ {t} \leftarrow \beta_ {2} \mathbf {h} _ {t - 1} + (1 - \beta_ {2}) \widehat {\mathbf {h}} _ {t - 1} + \frac {1}{2} (1 - \beta_ {2}) ^ {2} (\mathbf {h} _ {t - 1} - \widehat {\mathbf {h}} _ {t - 1}) ^ {2} / (\mathbf {h} _ {t - 1} + \delta), \tag {34}
$$

$$
\mathbf {m} _ {t} \leftarrow \mathbf {m} _ {t - 1} - \alpha_ {t} (\mathbf {g} _ {t} + \delta \mathbf {m} _ {t - 1}) / (\mathbf {h} _ {t} + \delta),
$$

$$
\boldsymbol {\sigma} _ {t} ^ {2} \leftarrow 1 / (N (\mathbf {h} _ {t} + \delta)).
$$

Here, $\delta > 0$ is the $L_{2}$ -regularization parameter and $\widehat{\mathbf{g}}_{t-1} = \frac{1}{|B|} \sum_{i \in B} \mathbb{E}_{q_{t-1}(\boldsymbol{\theta})}[\nabla \ell_i(\boldsymbol{\theta})]$ , $\widehat{\mathbf{h}}_{t-1} = \frac{1}{|B|} \sum_{i \in B} \mathbb{E}_{q_{t-1}(\boldsymbol{\theta})}[\nabla \ell_i(\boldsymbol{\theta})(\boldsymbol{\theta} - \mathbf{m}_{t-1}) / \sigma_{t-1}^2]$ denote Monte-Carlo approximations of the expected stochastic gradient and diagonal Hessian under $q_{t-1}(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} | \mathbf{m}_{t-1}, \text{diag}(\sigma_{t-1}^2))$ and minibatch $B$ . As suggested in [27, 35], we used the reparametrization trick to estimate the diagonal Hessian via gradients only. In practice, we approximate the expectations using a single random sample. We expect multiple samples to further improve the results.

# D Proof of Thm. 2 and the Beta-Bernoulli Model

From Eq. 27, it directly follows that

$$
\tilde {\mathbf {g}} _ {j} (\boldsymbol {\lambda}) = \nabla_ {\boldsymbol {\mu}} \mathbb {E} _ {q} [ - \log \tilde {p} _ {j} ] = - \nabla_ {\boldsymbol {\mu}} \langle \widetilde {\boldsymbol {\lambda}} _ {j}, \mathbb {E} _ {q} [ \mathbf {T} (\boldsymbol {\theta}) ] \rangle = - \nabla_ {\boldsymbol {\mu}} \langle \widetilde {\boldsymbol {\lambda}} _ {j}, \boldsymbol {\mu} \rangle = - \widetilde {\boldsymbol {\lambda}} _ {j}.
$$

Using this in Eq. 6, we get the deviation given in Eq. 4.

We will now show an example on Beta-Bernoulli model, which is a conjugate model. We assume the model to be $p(\mathcal{D},\theta) \propto p(\theta)\prod_{i}p(y_{i}|\theta)$ where the prior is $p(\theta) = \text{Beta}(\theta|\alpha_{0},\beta_{0})$ and likelihoods are $p(y_{i}|\theta) = \text{Ber}(y_{i}|\theta)$ with $D_{i} = y_{i}$ . This is a conjugate model and the posterior is Beta distribution, that is, it takes the same form as the prior. An expression is given below,

$$
q _ {*} = \operatorname{Beta} (\theta | \alpha_ {*}, \beta_ {*}), \text {   where   } \alpha_ {*} = \alpha_ {0} + \sum_ {j = 1} ^ {N} y _ {j}, \quad \beta_ {*} = \beta_ {0} - \sum_ {j = 1} ^ {N} y _ {j} + N.
$$

The posterior for the perturbed dataset $\mathcal{D}^{\backslash i}$ is also available in closed-form:

$$
q_{*}^{\backslash i} = \operatorname{Beta}(\theta |\alpha_{*}^{\backslash i},\beta_{*}^{\backslash i}),\text{where}\alpha_{*}^{\backslash i} = \alpha_{0} + \sum_{\substack{j = 1,\\ j\neq i}}^{N}y_{j},\qquad \beta_{*}^{\backslash i} = \beta_{0} - \sum_{\substack{j = 1,\\ j\neq i}}^{N}y_{j} + N - 1.
$$

Therefore the deviations in the posterior parameters can be simply obtained as follows:

$$
\alpha_ {*} ^ {\backslash i} - \alpha_ {*} = - y _ {i}, \quad \beta_ {*} ^ {\backslash i} - \beta_ {*} = y _ {i} - 1 \tag {35}
$$

This result can also be straightforwardly obtained using the MPE. For the Beta distribution $q_{\boldsymbol{\lambda}}(\theta) = \text{Beta}(\theta|\alpha, \beta)$ , we have $\boldsymbol{\lambda} = (\alpha - 1, \beta - 1)$ , therefore $\boldsymbol{\lambda}_{*}^{\backslash i} - \boldsymbol{\lambda}_{*} = (\alpha_{*}^{i} - \alpha_{*}, \beta_{*}^{\backslash i} - \beta_{*})$ . For Beta distribution, $\mathbf{T}(\theta) = (\log \theta, \log(1 - \theta))$ and writing the likelihood in an exponential form, we get

$$
p (y _ {i} | \theta) = \operatorname{Ber} (y _ {i} | \theta) \propto \theta^ {y _ {i}} (1 - \theta) ^ {1 - y _ {i}} \propto e ^ {y _ {i} \log \theta + (1 - y _ {i}) \log (1 - \theta)},
$$

therefore $\widetilde{\lambda}_i = (y_i, y_i - 1)$ . Setting $\lambda_*^{\backslash i} - \lambda_* = -\widetilde{\lambda}_i$ , we recover the result given in Eq. 35.

# E Proof of Thm. 3

For linear regression, we have

$$
\nabla \ell_ {i} (\boldsymbol {\theta}) = \mathbf {x} _ {i} (\mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} - y _ {i}), \qquad \nabla^ {2} \ell_ {i} (\boldsymbol {\theta}) = \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top}.
$$

Using these in Eq. 8, we get,

$$
\tilde {\mathbf {g}} _ {i} (\boldsymbol {\lambda} _ {*}) = \mathbb {E} _ {q} \left[ \mathbf {x} _ {i} (\mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} - y _ {i}) - \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} _ {*}, \frac {1}{2} \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top} \right] = \left(- \mathbf {x} _ {i} y _ {i}, \frac {1}{2} \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top}\right),
$$

The natural parameter is $\lambda_{*} = (\mathbf{H}_{*}\boldsymbol{\theta}_{*}, -\frac{1}{2}\mathbf{H}_{*})$ . In a similar way, we can define $q_{*}^{\backslash i}$ and its natural parameter. Using these, we can write Eq. 6 as

$$
\mathbf {H} _ {*} ^ {\backslash i} \boldsymbol {\theta} _ {*} ^ {\backslash i} - \mathbf {H} _ {*} \boldsymbol {\theta} _ {*} = - \mathbf {x} _ {i} y _ {i}, \quad - \frac {1}{2} \mathbf {H} _ {*} ^ {\backslash i} + \frac {1}{2} \mathbf {H} _ {*} = \frac {1}{2} \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top}.
$$

Substituting the second equation into the first one, we get the first equation below,

$$
\mathbf {H} _ {*} ^ {\backslash i} \boldsymbol {\theta} _ {*} ^ {\backslash i} - (\mathbf {H} _ {*} ^ {\backslash i} + \mathbf {x} _ {i} \mathbf {x} _ {i} ^ {\top}) \boldsymbol {\theta} _ {*} = - \mathbf {x} _ {i} y _ {i} \quad \Longrightarrow \quad \boldsymbol {\theta} _ {*} ^ {\backslash i} - \boldsymbol {\theta} _ {*} = (\mathbf {H} _ {*} ^ {\backslash i}) ^ {- 1} \mathbf {x} _ {i} (\mathbf {x} _ {i} ^ {\top} \boldsymbol {\theta} _ {*} - y _ {i}) = (\mathbf {H} _ {*} ^ {\backslash i}) ^ {- 1} \mathbf {x} _ {i} e _ {i}.
$$

The last equality is exactly Eq. 2. Since linear regression is a conjugate model, an alternate derivation would be to directly use the parameterization $\tilde{\lambda}_j$ of $\tilde{p}_i$ (derived in App. B) and plug it in Thm. 2.

# F Proof of Thm. 4

For simplicity, we denote

$$
\partial \hat {\boldsymbol {\lambda}} _ {*} ^ {\epsilon_ {i} = 0} = \left. \frac {\partial \hat {\boldsymbol {\lambda}} _ {*} ^ {\epsilon_ {i}}}{\partial \epsilon_ {i}} \right| _ {\epsilon_ {i} = 0},
$$

with $\hat{\lambda}_{*}^{\epsilon_{i}}$ as defined in Eq. 7 in the main text. For Gaussian distributions, the natural parameter comes in a pair $\hat{\lambda}_{*}^{\epsilon_{i}} = (\mathbf{H}_{*}^{\epsilon_{i}}\boldsymbol{\theta}_{*}^{\epsilon_{i}}, - \frac{1}{2}\mathbf{H}_{*}^{\epsilon_{i}})$ . Its derivative with respect to $\epsilon_{i}$ at $\epsilon_{i} = 0$ can be written as the following by using the chain rule:

$$
\partial \hat {\boldsymbol {\lambda}} _ {*} ^ {\epsilon_ {i} = 0} = \left(\mathbf {H} _ {*} \partial \boldsymbol {\theta} _ {*} ^ {\epsilon_ {i} = 0} + \partial \mathbf {H} _ {*} ^ {\epsilon_ {i} = 0} \boldsymbol {\theta} _ {*}, - \frac {1}{2} \partial \mathbf {H} _ {*} ^ {\epsilon_ {i} = 0}\right).
$$

Here, we use the fact that, as $\epsilon_{i} \to 0$ , we have $(\pmb{\theta}_{*}^{\epsilon_{i}}, \mathbf{H}_{*}^{\epsilon_{i}}) \to (\pmb{\theta}_{*}, \mathbf{H}_{*})$ and also assumed that the limit of the product is equal to the product of the individual limits. Next, we need the expression for the natural gradient, for which we will use Eq. 8 but approximate the expectation by using the delta approximation $\mathbb{E}_{q_{*}}[g(\pmb{\theta})] \approx g(\pmb{\theta}_{*})$ for any function $g$ , as shown below to define:

$$
\hat {\mathbf {g}} _ {i} (\boldsymbol {\lambda} _ {*}) = \left[ \nabla \ell_ {i} (\boldsymbol {\theta} _ {*}) - \nabla^ {2} \ell_ {i} (\boldsymbol {\theta} _ {*}) \boldsymbol {\theta} _ {*}, \frac {1}{2} \nabla^ {2} \ell_ {i} (\boldsymbol {\theta} _ {*}) \right]
$$

The claim is that if we set the perturbed $\partial \hat{\lambda}_{*}^{\epsilon_{i} = 0} = \hat{\mathbf{g}}_{i}(\boldsymbol{\lambda}_{*})$ we recover Eq. 3, that is, we set

$$
\mathbf {H} _ {*} \partial \boldsymbol {\theta} _ {*} ^ {\epsilon_ {i} = 0} + \partial \mathbf {H} _ {*} ^ {\epsilon_ {i} = 0} \boldsymbol {\theta} _ {*} = \nabla \ell (\boldsymbol {\theta} _ {*}) - \nabla^ {2} \ell_ {i} (\boldsymbol {\theta} _ {*}) \boldsymbol {\theta} _ {*}, \quad - \frac {1}{2} \partial \mathbf {H} _ {*} ^ {\epsilon_ {i} = 0} = \frac {1}{2} \nabla^ {2} \ell_ {i} (\boldsymbol {\theta} _ {*}).
$$

Plugging the second equation into the first, the second term cancels and we recover Eq. 3.

# G Extension to Non-Differentiable Loss function

For non-differentiable cases, we can use Eq. 28 to rewrite the BLR of Eq. 29 as

$$
\mathbf {m} _ {t} \leftarrow \mathbf {m} _ {t - 1} - \rho \mathbf {S} _ {t} ^ {- 1} \nabla_ {\mathbf {m}} \mathbb {E} _ {q _ {t - 1}} [ \mathcal {L} (\boldsymbol {\theta}) ], \quad \mathbf {S} _ {t} \leftarrow (1 - \rho) \mathbf {S} _ {t - 1} + 2 \rho \nabla_ {\boldsymbol {\Sigma}} \mathbb {E} _ {q _ {t - 1}} [ \mathcal {L} (\boldsymbol {\theta}) ], \tag {36}
$$

where $\mathbf{\Sigma} = \mathbf{S}^{-1}$ . Essentially, we take derivative outside the expectation instead of inside which is valid because the expectation of a non-differentiable function is still differentiable (under some regularity conditions). The same technique can be applied to Eq. 8 to get

$$
\tilde {\mathbf {g}} _ {i} (\boldsymbol {\lambda}) = \left(\nabla_ {\mathbf {m}} \mathbb {E} _ {q} [ \ell_ {i} ] - 2 \nabla_ {\boldsymbol {\Sigma}} \mathbb {E} _ {q} [ \ell_ {i} (\boldsymbol {\theta}) ] \mathbf {m}, \quad \nabla_ {\boldsymbol {\Sigma}} \mathbb {E} _ {q} [ \ell_ {i} (\boldsymbol {\theta}) ]\right), \tag {37}
$$

and proceeding in the same fashion we can write: $\hat{\mathbf{m}}_t^{\backslash i} - \mathbf{m}_t = (\hat{\mathbf{S}}_t^{\backslash i})^{-1}\nabla_{\mathbf{m}}\mathbb{E}_{q_t}[\ell_i(\boldsymbol {\theta})]$ . This is the extension of Eq. 10 to non-differentiable loss functions.

# H Sensitivity Measures for Sparse Variational Gaussian Processes

Sparse variational GP (SVGP) methods optimize the following variational objective to find a Gaussian posterior approximation $q(\mathbf{u})$ over function values $\mathbf{u} := (f(\mathbf{z}_1), f(\mathbf{z}_2), \ldots, f(\mathbf{z}_M))$ where $\mathcal{Z} := (\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_M)$ is the set of inducing inputs with $M \ll N$ :

$$
\underline {{\mathcal {L}}} (\mathbf {m}, \boldsymbol {\Sigma}, \mathcal {Z}, \phi) := \sum_ {i = 1} ^ {N} \mathbb {E} _ {q (f _ {i})} \left[ \log p (y _ {i} | f _ {i}) \right] - \mathbb {D} _ {\mathrm{KL}} (q (\mathbf {u}) \| p (\mathbf {u}))
$$

where $p(\mathbf{u}) := \mathcal{N}(\mathbf{u} | \mathbf{0}, \mathbf{K}_{\mathbf{uu}})$ is the prior with $K_{uu}$ as the covariance function $\kappa(\cdot, \cdot')$ evaluated at $\mathcal{Z}$ , $q(f_i) = \mathcal{N}(f_i | \mathbf{a}_i^\top \mathbf{m}, \mathbf{a}_i^\top \mathbf{\Sigma} \mathbf{a}_i + \sigma_i^2)$ is the posterior marginal of $f_i = f(\mathbf{x}_i)$ with $a_i := K_{uu}^{-1} k_{ui}$ and $\sigma_i^2 := \kappa_{ii} - a_i^\top K_{uu} a_i$ as the noise variance of $f_i$ conditioned on u. The objective is also used to optimize hyperparameters $\phi$ and inducing input set Z.

We can optimize the objective using the BLR for which the resulting update is identical to the variational online-newton (VON) algorithm. We first write the natural gradients,

$$
\widetilde {\nabla} \mathbb {E} _ {q _ {t} (f _ {i})} [ - \log p (y _ {i} | f _ {i}) ] = \left((e _ {i t} - \beta_ {i t} \mathbf {a} _ {i} ^ {\top} \mathbf {m} _ {*}) \mathbf {a} _ {i}, \frac {1}{2} \beta_ {i t} \mathbf {a} _ {i} \mathbf {a} _ {i} ^ {\top}\right). \tag {38}
$$

where we define

$$
e _ {i t} = \mathbb {E} _ {q _ {t} (f _ {i})} [ - \nabla_ {f _ {i}} \log p (y _ {i} | f _ {i}) ], \quad \beta_ {i t} = \mathbb {E} _ {q _ {t} (f _ {i})} [ - \nabla_ {f _ {i}} ^ {2} \log p (y _ {i} | f _ {i}) ]
$$

We define $\mathbf{A}$ to be a matrix with $\mathbf{a}_i^\top$ as rows, and $\mathbf{e}_t, \beta_t$ to be vectors of $e_{it}, \beta_{it}$ . Using these in the VON update, we simplify as follows:

$$
\mathbf {S} _ {t + 1} = (1 - \rho) \mathbf {S} _ {t} + \rho \left[ \mathbf {A} ^ {\top} \operatorname{diag} \left(\boldsymbol {\beta} _ {t}\right) \mathbf {A} + \mathbf {K} _ {\mathbf {u u}} ^ {- 1} \right] \tag {39}
$$

$$
\begin{array}{l} \mathbf {m} _ {t + 1} = \mathbf {S} _ {t + 1} ^ {- 1} \left[ (1 - \rho) \mathbf {S} _ {t} \mathbf {m} _ {t} - \rho \left(\mathbf {A} ^ {\top} \mathbf {e} _ {t} - \mathbf {A} ^ {\top} \mathrm{diag} (\boldsymbol {\beta} _ {t}) \mathbf {A m} _ {t}\right) \right] \\ = \mathbf {S} _ {t + 1} ^ {- 1} \left[ \left((1 - \rho) \mathbf {S} _ {t} + \rho \mathbf {A} ^ {\top} \mathrm{diag} (\boldsymbol {\beta} _ {t}) \mathbf {A}\right) \mathbf {m} _ {t} - \rho \mathbf {A} ^ {\top} \mathbf {e} _ {t} \right] \\ = \mathbf {S} _ {t + 1} ^ {- 1} \left[ \left(\mathbf {S} _ {t + 1} - \rho \mathbf {K} _ {\mathbf {u u}} ^ {- 1}\right) \mathbf {m} _ {t} - \rho \mathbf {A} ^ {\top} \mathbf {e} _ {t} \right] \tag {40} \\ = \mathbf {S} _ {t + 1} ^ {- 1} \left[ \mathbf {S} _ {t + 1} \mathbf {m} _ {t} - \rho \left(\mathbf {A} ^ {\top} \mathbf {e} _ {t} + \mathbf {K} _ {\mathbf {u u}} ^ {- 1} \mathbf {m} _ {t}\right) \right] \\ = \mathbf {m} _ {t} - \rho \mathbf {S} _ {t + 1} ^ {- 1} \left[ \mathbf {A} ^ {\top} \mathbf {e} _ {t} + \mathbf {K} _ {\mathbf {u u}} ^ {- 1} \mathbf {m} _ {t} \right]. \\ \end{array}
$$

For Gaussian likelihood, the updates in Eqs. 39 and 40 coincide with the method of [18], and for non-Gaussian likelihood they are similar to the natural-gradient method by [45], but we use the specific parameterization of [26]. An alternate update rule in terms of site parameters is given by [1] (see Eqs. 22-24).

We are now ready to write the sensitivity measure essentially substituting the gradient in Eq. 10),

$$
\mathbf {S} _ {t} ^ {- 1} \nabla_ {\mathbf {m}} \mathbb {E} _ {q _ {t} (\mathbf {u})} [ - \log p (y _ {i} | f _ {i}) ] = \mathbf {S} _ {t} ^ {- 1} \mathbf {a} _ {i} \mathbb {E} _ {q _ {t} (f _ {i})} [ - \nabla \log p (y _ {i} | f _ {i}) ] = \mathbf {S} _ {t} ^ {- 1} \mathbf {a} _ {i} e _ {i t} \tag {41}
$$

We can also see the bi-linear relationship by considering the deviation in the mean of the posterior marginal $f_{i}(\mathbf{m}):=\mathbf{a}_{i}^{\top}\mathbf{m}$ ,

$$
f _ {i} \left(\mathbf {m} _ {t} ^ {\backslash i}\right) - f _ {i} \left(\mathbf {m} _ {t}\right) \approx \mathbf {a} _ {i} ^ {\top} \left(\hat {\mathbf {m}} _ {t} ^ {\backslash i} - \mathbf {m} _ {t}\right) = \mathbf {a} _ {i} ^ {\top} \boldsymbol {\Sigma} _ {t} \mathbf {a} _ {i} e _ {i t} = v _ {i t} e _ {i t} \tag {42}
$$

where $v_{it} = \mathbf{a}_i^\top \boldsymbol{\Sigma}_t\mathbf{a}_i$ is the marginal variance of $f_i$ .

# I Experimental Details

# I.1 Neural network architectures

Below, we describe different neural networks used in our experiments,

MLP (500, 300): This is a multilayer perceptron (MLP) with two hidden layers of 500 and 300 neurons and a parameter count of around 546 000 (using hyperbolic-tangent activations).

MLP (32, 16): This is also an MLP with two hidden layers of 32 and 16 neurons, which accounts for around 26 000 parameters (also using hyperbolic tangent activations).

LeNet5: This is a standard convolutional neural network (CNN) architecture with three convolution layers followed by two fully-connected layers, corresponding to around 62 000 parameters.

CNN: This network, taken from the DeepOBS suite [46], consists of three convolution layers followed by three fully-connected layers with a parameter count of 895000.

ResNet-20: This network has around 274000 parameters. We use filter response normalization (FRN) [48] as an alternative to batch normalization.

MLP for USPS: For the experiment on binary USPS in Fig. 6(a), we use an MLP with three hidden layers of 30 neurons each and a total of around 10 000 parameters.

# I.2 Details of “Do estimated deviations correlate with the truth?”

In Fig. 2, we train neural network classifiers with a cross-entropy loss to obtain $\theta_{*}$ . Due to the computational demand of per-example retraining, the removed examples are randomly subsampled from the training set. We show results over 1000 examples for MNIST and FMNIST and 100 examples for CIFAR10. In the multiclass setting, the expression yields a per-class sensitivity value. We obtain a scalar value for each example by summing over the absolute values of the per-class sensitivities. For training both the original model $\theta_{*}$ and the perturbed models $\theta_{*}^{\backslash i}$ , we use SGD with a momentum parameter of 0.9 and a cosine learning-rate scheduler. To obtain $\theta_{*}^{\backslash i}$ , we retrain a model that is warmstarted at $\theta_{*}$ . Other details regarding the training setup are given in Table 2. For all models, we do not use data augmentation during training. The resulting $\theta_{*}$ for MNIST, FMNIST, and CIFAR10 have training accuracies of $99.9\%$ , $95.0\%$ , and $99.9\%$ , respectively. The test accuracies for these models are $98.4\%$ , $91.2\%$ and $76.7\%$ .

<table><tr><td>Dataset</td><td>Model</td><td> $B$ </td><td> $\delta$ </td><td> $E^{*}$ </td><td> $LR^{*}$ </td><td> $LR_{\min}^{*}$ </td><td> $E^{\backslash i}$ </td><td> $LR^{\backslash i}$ </td><td> $LR_{\min}^{\backslash i}$ </td></tr><tr><td>MNIST</td><td>MLP (500, 300)</td><td>256</td><td>100</td><td>500</td><td> $10^{-2}$ </td><td> $10^{-3}$ </td><td>300</td><td> $10^{-3}$ </td><td> $10^{-4}$ </td></tr><tr><td>FMNIST</td><td>LeNet5</td><td>256</td><td>100</td><td>300</td><td> $10^{-1}$ </td><td> $10^{-3}$ </td><td>200</td><td> $10^{-3}$ </td><td> $10^{-4}$ </td></tr><tr><td>CIFAR10</td><td>CNN</td><td>512</td><td>250</td><td>500</td><td> $10^{-2}$ </td><td> $10^{-4}$ </td><td>300</td><td> $10^{-4}$ </td><td> $10^{-6}$ </td></tr></table>

Table 2: Hyperparameters for predicting true sensitivity in Fig. 2. B, E and LR denote batch size, training epochs and learning-rates, respectively. The superscripts \* and $^i$ indicate hyperparameters for training on all data and warmstarted leave-one-out retraining, respectively. $LR_{\min}$ is the minimum learning-rate of the cosine scheduler.

Additional group removal experiments: We also study how the deviation for removing a group of examples in a set M can be estimated using a variation of Eq. 14 for the deviation in predictions at convergence. Denoting the vector of $f_{i}(\boldsymbol{\theta})$ for $i \in M$ by $\mathbf{f}_{\mathcal{M}}(\boldsymbol{\theta})$ , we get

$$
\sigma \left( \right.\mathbf {f} _ {\mathcal {M}} \left(\boldsymbol {\theta} _ {*} ^ {\backslash \mathcal {M}}\right) - \sigma \left(\mathbf {f} _ {\mathcal {M}} \left(\boldsymbol {\theta} _ {*}\right)\right) \approx \boldsymbol {\Lambda} \left(\boldsymbol {\theta} _ {*}\right) \mathbf {V} _ {\mathcal {M}} \left(\boldsymbol {\theta} _ {*}\right) \mathbf {e} _ {\mathcal {M}} \left(\boldsymbol {\theta} _ {*}\right) \approx \sum_ {i \in \mathcal {M}} \sigma^ {\prime} \left(f _ {i *}\right) v _ {i *} e _ {i *}. \tag {43}
$$

where $\mathbf{\Lambda}(\boldsymbol{\theta}_{*})$ is a diagonal matrix containing all $\sigma'(f_{i*})$ , $\mathbf{V}_{\mathcal{M}}(\boldsymbol{\theta}_{*}) = \nabla \mathbf{f}_{\mathcal{M}}(\boldsymbol{\theta}_{*})\mathbf{S}_{*}^{-1}\nabla \mathbf{f}_{\mathcal{M}}(\boldsymbol{\theta}_{*})^{\top}$ is the prediction covariance of size $M\times M$ where $M$ is the number of examples in $\mathcal{M}$ , and $\mathbf{e}_{\mathcal{M}}(\boldsymbol{\theta}_{*})$ is the vector of prediction errors. The last approximation above is done to avoid building the covariance, where we ignore the off-diagonal entries of $\mathbf{V}_{\mathcal{M}}(\boldsymbol{\theta}_{*})$ .

In Fig. 6(a) we consider a binary USPS dataset consisting of the classes for the digits 3 and 5. Using $|\mathcal{M}| = 16$ , we show the first and second approximations in Eq. 43 both correlate well with the truth obtained by removing a group and retraining the model. In Fig. 6(b) we do the same on MNIST with $|\mathcal{M}| = 64$ , where we see similar trends. For the experiment on binary USPS in Fig. 6(a), we train a MLP with three hidden layers with 30 neurons each. The original model $\theta_{*}$ is trained for 500 epochs with a learning-rate of $10^{-3}$ , a batch size of 32 and a $L_{2}$ -regularization parameter $\delta = 5$ . It has $100\%$ training accuracy and $94.8\%$ test accuracy. For the leave-group-out retraining to obtain $\theta_{*}^{\backslash M}$ , we initialize the model at $\theta_{*}$ , use a learning-rate of $10^{-3}$ and train for 1000 epochs. For the MNIST result in Fig. 6(b) we use the MLP (500, 300) model with the same hyperparameters as for

![](images/b6e5e90a6794b5afc1d04905911b0360d1584ec17b04d8be7f60168e9dfe0e7c.jpg)

<details>
<summary>scatter</summary>

| Group     | True Deviation | Estimated Deviation |
| --------- | -------------- | ------------------- |
| group     | 0.3            | 0.4                 |
| group     | 0.5            | 0.5                 |
| group     | 0.7            | 0.9                 |
| group     | 0.9            | 1.0                 |
| individual| 0.3            | 0.3                 |
| individual| 0.6            | 0.8                 |
| individual| 0.8            | 1.0                 |
| individual| 0.9            | 1.1                 |
</details>

(a) MLP on USPS-3vs5, $|\mathcal{M}| = 16$

![](images/e371f31a581f7571bed02215d82375789dacca10c058d4f408a097bab7e69f4d.jpg)

<details>
<summary>scatter</summary>

| True Deviation | Value |
| -------------- | ----- |
| 0.0            | 0.0   |
| 0.1            | 0.1   |
| 0.2            | 0.2   |
| 0.3            | 0.3   |
| 0.4            | 0.4   |
| 0.5            | 0.5   |
| 0.6            | 0.6   |
| 0.7            | 0.7   |
| 0.8            | 0.8   |
| 0.9            | 0.9   |
| 1.0            | 1.0   |
| 1.1            | 1.1   |
| 1.2            | 1.2   |
| 1.3            | 1.3   |
| 1.4            | 1.4   |
| 1.5            | 1.5   |
| 1.6            | 1.6   |
| 1.7            | 1.7   |
| 1.8            | 1.8   |
| 1.9            | 1.9   |
| 2.0            | 2.0   |
</details>

(b) MLP on MNIST, $|\mathcal{M}| = 64$   
Figure 6: Panel (a) and Panel (b) show that the estimated deviation for removal of groups of examples correlates well with the true deviations obtained by retraining. Each marker corresponds to a removed group of examples. The red circles show the second approximation in Eq. 43. In Panel (a), we additionally show (with blue squares) the first approximation of Eq. 43. We see that the second approximation is quite accurate in this case.

$\theta_{*}$ in Table 2. For $\theta_{*}^{\backslash M}$ , we initialize the model at $\theta_{*}$ and use a cosine schedule of the learning-rate from $10^{-4}$ to $10^{-5}$ over 500 epochs. We do not use data augmentation. Similarly to the experiments on per-example removal, we use a K-FAC approximation.

<table><tr><td>Dataset</td><td>Model</td><td>E</td><td>LR*</td><td>LR*min</td><td>LR\C</td><td>LR\Cmin</td></tr><tr><td>MNIST</td><td>MLP (500, 300)</td><td>500</td><td>10-2</td><td>10-3</td><td>10-4</td><td>10-5</td></tr><tr><td>MNIST</td><td>LeNet5</td><td>300</td><td>10-1</td><td>10-3</td><td>10-5</td><td>10-6</td></tr><tr><td>FMNIST</td><td>MLP (32, 16)</td><td>300</td><td>10-2</td><td>10-3</td><td>10-5</td><td>10-6</td></tr><tr><td>FMNIST</td><td>LeNet5</td><td>300</td><td>10-1</td><td>10-3</td><td>10-4</td><td>10-5</td></tr></table>

Table 3: Hyperparameters for the class removal experiments in Fig. 3(a) and Fig. 11(d). B, E and LR denote batch size, training epochs and learning-rates. The superscripts \* and \C indicate hyperparameters for training on all data and warmstarted leave-one-class-out retraining, respectively. $LR_{min}$ is the minimum learning-rate of the cosine scheduler.

# I.3 Details of “Predicting the effect of class removal on generalization”

For the FMNIST experiment in Fig. 3(a), we use the MLP (32, 16) and LeNet5 models. For the MNIST experiment in Fig. 11(d), we use the MLP (500, 300) and LeNet5 models. The hyperparameters are given in Table 3. The MLP on MNIST has a training accuracy of $99.9\%$ and a test accuracy of $98.4\%$ . When using LeNet5, the training and test accuracies are $99.2\%$ and $99.1\%$ . On FMNIST, the LeNet5 has an accuracy of $95.0\%$ on the training set, and an accuracy of $91.2\%$ on the test set. On the same dataset, the MLP has a training accuracy of $89.9\%$ and a test accuracy of $86.2\%$ . For all models, we use a regularization parameter of 100 and a batch size of 256. The leave-one-class-out training is run for 1000 epochs and the rest of the training setup is same as the previous experiment.

# I.4 Details of “Estimating the leave-one-out cross-validation curves for hyperparameter tuning”

The details of the training setup are in Table 2. Fig. 7 is the same as Fig. 4 but additionally shows the test errors. For visualization purposes, each plot uses a moving average of the plotted lines with a smoothing window. Other training details are similar to previous experiments. All models are

![](images/27be50b74a183b34d6485b737edd89866eb14a201843dc66f4f95ae693faa387.jpg)

<details>
<summary>line</summary>

| δ     | Test NLL | LOO-CV | Test error |
|-------|----------|--------|------------|
| 10^0  | ~0.08    | ~0.08  | ~2%        |
| 10^1  | ~0.06    | ~0.06  | ~2%        |
| 10^2  | ~0.05    | ~0.05  | ~2%        |
| 10^3  | ~0.2     | ~0.2   | ~4%        |
</details>

(a) MNIST, MLP

![](images/b30f3b4b0137fa9553e45c84a7df1c1c3f03923533dbdaa1df0df88fae8be45e.jpg)

<details>
<summary>line</summary>

| δ     | Line 1 | Line 2 |
|-------|--------|--------|
| 10^1  | 0.5    | 0.4    |
| 10^2  | 0.3    | 0.3    |
| 10^3  | 0.5    | 0.5    |
</details>

(b) FMNIST, LeNet5

![](images/787606d36c142e6a9ab07da07bcbecd94c413024f209c866863fc46e56c098dd.jpg)

<details>
<summary>line</summary>

| δ     | Test error |
|-------|------------|
| 10^1  | 1.8        |
| 10^2  | 1.2        |
| 10^3  | 1.2        |
</details>

(c) CIFAR10, CNN   
Figure 7: Leave-one-out estimation with sensitivities obtained from MPE (Train-LOO-MPE) can accurately estimate the LOO-CV curve for predicting generalization and tuning of the $L_{2}$ -regularization parameter on MNIST, FMNIST and CIFAR-10.

trained from scratch where we use Adam for FMNIST, AdamW [36] for CIFAR10, and SGD with a momentum parameter of 0.9 for MNIST. We use a cosine learning-rate scheduler to anneal the learning-rate. The other hyperparameters are similar to the settings of the models trained on all data from the leave-one-out experiments in Table 2, except for the number of epochs for CIFAR10 where we train for 150 epochs. Similarly to App. I.2, we use a Kronecker-factored Laplace approximation for variance computation and do not employ data augmentation during training.

<table><tr><td>Dataset</td><td>Model</td><td>Number of  $\delta s$ </td><td>Range</td><td>Smoothing window</td></tr><tr><td>MNIST</td><td>MLP (500, 300)</td><td>96</td><td> $10^{0} - 10^{3}$ </td><td>3</td></tr><tr><td>FMNIST</td><td>LeNet5</td><td>96</td><td> $10^{1} - 10^{3}$ </td><td>5</td></tr><tr><td>CIFAR10</td><td>CNN</td><td>30</td><td> $10^{1} - 10^{3}$ </td><td>3</td></tr></table>

Table 4: Experimental settings for Fig. 4.

# I.5 Details of “Predicting generalization during the training”

Details of the training setup: The experimental details, including test accuracies at the end of training, are listed in Table 5. We use a grid search to determine the regularization parameter $\delta$ . The learning-rate is decayed according to a cosine schedule. For diagonal-GGN-LOO and K-FAC-LOO, we use the SGD optimizer with an exception on the FMNIST dataset where we use the AdamW optimizer [36]. In that experiment, we use a weight decay factor of $\delta / N$ replacing the explicit $L_{2}$ -regularization term in the loss in Eq. 1. The regularizer $\mathcal{R}(\theta)$ is set to zero. We do not use training data augmentation. For all plots, the LOO-estimate is evaluated periodically during the training, which is indicated with markers.

Additional details on hyperparameters of iBLR are as follows, where $h_{0}$ is the initialization of the Hessian:

• MNIST, MLP (32, 16): $h_{0} = 0.1$   
- MNIST, LeNet5: $h_0 = 0.1$   
• FMNIST, LeNet5: $h_{0} = 0.1$   
• CIFAR10, CNN: $h_{0} = 0.05$   
• CIFAR10, ResNet20: $h_{0} = 0.01$

We set $\beta_{1}=0.9$ and $\beta_{2}=0.99999$ in all of those experiments. The magnitude of the prediction variance can depend on $h_{0}$ , which therefore can influence the magnitude of the sensitivities that are perturbing the function outputs in the LOO estimate of Eq. 16. We choose $h_{0}$ on a grid of four values [0.01, 0.05, 0.1, 0.5] to obtain sensitivities that result in a good prediction of generalization performance.

Additional Results: In Fig. 8, we show additional results for MNIST and CIFAR10 that are not included in the main text. For MNIST, we evaluate both on a the MLP (32, 16) model and a LeNet5 architecture. For the additional CIFAR10 results, we use the CNN. In Fig. 9 we include an additional

<table><tr><td>Dataset</td><td>Model</td><td>Method</td><td>LR</td><td> $LR_{\min}$ </td><td>B</td><td>δ</td><td>Test acc.</td></tr><tr><td rowspan="3">MNIST</td><td rowspan="3">MLP (32, 16)</td><td>iBLR</td><td> $10^{-2}$ </td><td> $10^{-4}$ </td><td>256</td><td>80</td><td>95.6%</td></tr><tr><td>diag.-GGN-LOO</td><td> $10^{-3}$ </td><td> $10^{-4}$ </td><td>256</td><td>80</td><td>95.8%</td></tr><tr><td>K-FAC-LOO</td><td> $10^{-3}$ </td><td> $10^{-4}$ </td><td>256</td><td>80</td><td>95.8%</td></tr><tr><td rowspan="3">MNIST</td><td rowspan="3">LeNet5</td><td>iBLR</td><td> $10^{-2}$ </td><td> $10^{-4}$ </td><td>256</td><td>60</td><td>97.5%</td></tr><tr><td>diag.-GGN-LOO</td><td> $10^{-3}$ </td><td> $10^{-4}$ </td><td>256</td><td>60</td><td>97.4%</td></tr><tr><td>K-FAC-LOO</td><td> $10^{-3}$ </td><td> $10^{-4}$ </td><td>256</td><td>60</td><td>97.4%</td></tr><tr><td rowspan="3">FMNIST</td><td rowspan="3">LeNet5</td><td>iBLR</td><td> $10^{-1}$ </td><td>0</td><td>256</td><td>60</td><td>90.7%</td></tr><tr><td>diag.-GGN-LOO</td><td> $10^{-2}$ </td><td> $10^{-4}$ </td><td>256</td><td>60</td><td>91.0%</td></tr><tr><td>K-FAC-LOO</td><td> $10^{-2}$ </td><td> $10^{-4}$ </td><td>256</td><td>60</td><td>91.0%</td></tr><tr><td rowspan="3">CIFAR10</td><td rowspan="3">CNN</td><td>iBLR</td><td> $10^{-1}$ </td><td> $10^{-4}$ </td><td>512</td><td>250</td><td>81.0%</td></tr><tr><td>diag.-GGN-LOO</td><td> $10^{-1}$ </td><td>0</td><td>512</td><td>250</td><td>75.4%</td></tr><tr><td>K-FAC-LOO</td><td> $10^{-1}$ </td><td>0</td><td>512</td><td>250</td><td>73.6%</td></tr><tr><td>CIFAR10</td><td>ResNet-20</td><td>iBLR</td><td> $2 * 10^{-1}$ </td><td>0</td><td>50</td><td>10</td><td>83.4%</td></tr></table>

Table 5: Experimental settings for predicting generalization during the training in Fig. 1(b), Fig. 5 and Fig. 8. B and E denote the batch-size and training epochs, respectively. LR and $LR_{min}$ are the start and end learning-rates of the cosine scheduler. $\delta$ is the regularization parameter. The specification in brackets in the third column indicates the method for computing sensitivities. We use either iBLR or SGD with diagonal GGN (diag.GGN) or K-FAC.

![](images/8bfb3dc5353ff9e53cce4a8c13bc83b6537141a4d7e2334f9630b351afdeada3.jpg)  
Figure 8: These plots are similar to Fig. 5 but for different model-data pairs. The three rows correspond to MLP on MNIST, LeNet5 on MNIST, and CNN on CIFAR10, respectively. The trends are almost same as those discussed in the main text.

experiment where the model overfits. The K-FAC-LOO estimate deteriorates in this case, but we can still use the LOO as a diagnostic for detecting overfitting and as a stopping criterion. We train

![](images/e6e9e487c85f83957499262fd990390b1f27bbe87ab4e36084b2e6be89881fbb.jpg)

<details>
<summary>line</summary>

| Epochs | Black Line | Gray Line |
| ------ | ---------- | --------- |
| 0      | 0.7        | 0.4       |
| 25     | 0.6        | 0.35      |
| 50     | 0.8        | 0.4       |
| 75     | 0.85       | 0.6       |
| 100    | 0.9        | 0.7       |
</details>

(a) Diagonal-GGN-LOO

![](images/bab2c82ed96842ea18deac351cf3bf435a237b66b774f43ff8adbeddab08a20b.jpg)

<details>
<summary>line</summary>

| Epochs | Test NLL | LOO-CV |
| ------ | -------- | ------ |
| 0      | 0.4      | 0.5    |
| 25     | 0.35     | 0.35   |
| 50     | 0.4      | 0.4    |
| 75     | 0.6      | 0.4    |
| 100    | 0.8      | 0.6    |
</details>

(b) K-FAC-LOO

Figure 9: Additional results for training with AdamW where we observe overfitting. We see that K-FAC-LOO deteriorates when the model start to overfit. Both the LOO measures can still be useful tools for diagnosing overfitting. Details of training setup are given in Table 6   
![](images/d6e1bdef1d32d7e6998cd3e23dc84bf3c0a51f987869532aa450ce74ae05157a.jpg)  
Figure 10: LOO-CV estimates with Adam using the measure suggested in Table 1. The two rows correspond to a batch size of 8 and a batch size of 32, respectively. A smaller batchsize generally decreases the gap between the test NLL and the estimate. Details of the training setup are given in Table 7.

a LeNet5 on FMNIST with AdamW and predict generalization. The trend of the estimated NLL matches the trend of the test NLL in the course of training.

In Fig. 10, we include further results for sensitivity estimation with the Adam optimizer. We use the following update

$\mathbf{r}_{t} \leftarrow \beta_{1}\mathbf{r}_{t-1} + (1 - \beta_{1})\mathbf{g}_{t}, \quad \mathbf{s}_{t} \leftarrow \beta_{2}\mathbf{s}_{t-1} + (1 - \beta_{2})(\mathbf{g}_{t} \cdot \mathbf{g}_{t}), \quad \boldsymbol{\theta}_{t} \leftarrow \boldsymbol{\theta}_{t-1} - \rho\mathbf{r}_{t}/(\sqrt{\hat{\mathbf{s}}_{t}} + \epsilon),$ where $g_{t}$ is the minibatch gradient, $\beta_{1}$ and $\beta_{2}$ are coefficients for the running averages, $\rho$ is a learning-rate, and $\epsilon$ a small damping to stabilize. We construct a diagonal matrix $S_{t} = \text{diag}(N\sqrt{s_{t}})$ to estimate sensitivity with MPE as suggested in Table 1 ( $N$ is the number of training examples). Better results are expected by building better estimates of $S_{t}$ as discussed in [27]. As described in section 3.4 of [27], a smaller batch size should improve the estimate, which we also observe in the experiment.

# I.6 Details of “evolution of sensitivities during training”

We use the MPE with iBLR for neural network classification on MNIST, FMNIST and CIFAR10, as well as MPE for logistic regression on MNIST. Experiment details are in Table 8.

For the experiment in Fig. 11(a), we consider Bayesian logistic regression. We set $\delta = 0.1$ . The Hessian is always positive-definite due to the convex loss function therefore we use the VON algo-

<table><tr><td>Dataset</td><td>Model</td><td>Method</td><td>LR</td><td> $LR_{\min}$ </td><td>B</td><td>δ</td><td>Test acc.</td></tr><tr><td rowspan="2">FMNIST</td><td rowspan="2">LeNet5</td><td>diag., AdamW</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td><td>256</td><td>60</td><td>88.1%</td></tr><tr><td>K-FAC, AdamW</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td><td>256</td><td>60</td><td>87.6%</td></tr></table>

Table 6: Experimental settings for predicting generalization during the training in Fig. 9.

<table><tr><td>Dataset</td><td>Model</td><td> $LR$ </td><td> $LR_{\min}$ </td><td> $\delta$ </td><td>Test acc. ( $B = 8$ )</td><td>Test acc. ( $B = 32$ )</td></tr><tr><td>MNIST</td><td>MLP (32, 16)</td><td> $10^{-3}$ </td><td>0</td><td>80</td><td>97.3%</td><td>97.4%</td></tr><tr><td>MNIST</td><td>LeNet5</td><td> $10^{-3}$ </td><td>0</td><td>60</td><td>99.2%</td><td>99.2%</td></tr><tr><td>FMNIST</td><td>LeNet5</td><td> $10^{-3}$ </td><td>0</td><td>60</td><td>91.4%</td><td>91.2%</td></tr><tr><td>CIFAR10</td><td>CNN</td><td> $10^{-3}$ </td><td>0</td><td>50</td><td>75.2%</td><td>78.4%</td></tr></table>

Table 7: Experimental settings for Fig. 10.

![](images/20d59fd78e0d000c48b56da4938cbff77810bc753a43d8a3366748505fffb29f.jpg)

<details>
<summary>line</summary>

| Examples | Estimated Deviation (5th epoch) | Estimated Deviation (25th epoch) |
| -------- | ------------------------------- | -------------------------------- |
| Low      | ~0                              | ~0                               |
| High     | ~10                             | ~10                              |
</details>

(a) MNIST, Bayesian logistic regr.

![](images/0c315d8a914167bb8b31bec2dc93eebd7ab029581135d27a4ec9ebaa5163ca44.jpg)

<details>
<summary>line</summary>

| Examples | Low Sensitivity | High Sensitivity |
| -------- | --------------- | ---------------- |
| 1st Epoch | 0.000           | 0.000            |
| 2nd      | 0.000           | 0.000            |
| 50th     | 0.000           | 0.014            |
| 20th     | 0.000           | 0.014            |
| High     | 0.000           | 0.014            |
</details>

(b) MNIST, iBLR with MLP

![](images/05f3699caf6bb25fb73db61279c669fe7c732ff8434b7a343fdc20cc1ffb3c6e.jpg)

<details>
<summary>line</summary>

| Examples | Estimated Deviation (1st Epoch) | Estimated Deviation (20th Epoch) |
| -------- | ------------------------------ | ------------------------------- |
| Low      | ~0.00                          | ~0.00                           |
| High     | ~0.00                          | ~0.06                           |
</details>

(c) CIFAR-10, iBLR & ResNet-20

![](images/de658702d23f14d6dc137c78378f23e78070a67c3aaad54b486c10b97b5847c6.jpg)

<details>
<summary>scatter</summary>

| Test NLL | LOCO estimation | Model  |
| -------- | --------------- | ------ |
| 0.5      | 0.03            | 1      |
| 0.8      | 0.04            | 6      |
| 1.0      | 0.05            | 7      |
| 1.2      | 0.06            | 8      |
| 1.5      | 0.07            | 9      |
| 2.0      | 0.08            | 3      |
| 2.5      | 0.09            | 5      |
| 3.0      | 0.10            | 2      |
| 3.5      | 0.11            | 3      |
| 4.0      | 0.12            | 9      |
</details>

(d) Class removal result for MNIST   
Figure 11: Additional experiments similar to Fig. 3(b). In Panel (a), we show the evolution of sensitivities for Bayesian logistic regression on MNIST trained with the VON algorithm. In Panel (b) we use a MLP trained with the iBLR optimizer. In Panel (c), we use a ResNet-20 trained with iBLR on CIFAR-10. Panel (d) shows the class removal result similar to Fig. 3(a), but on MNIST

<table><tr><td>Dataset</td><td>Model</td><td>B</td><td> $\delta$ </td><td>E</td></tr><tr><td>MNIST</td><td>MLP (500, 300)</td><td>256</td><td>30</td><td>100</td></tr><tr><td>FMNIST</td><td>LeNet5</td><td>256</td><td>60</td><td>100</td></tr><tr><td>CIFAR10</td><td>ResNet–20</td><td>512</td><td>35</td><td>300</td></tr></table>

Table 8: Experimental settings for evolution of sensitivities during training in Fig. 3(b), and Fig. 3.

rithm given in Eq. 29. We use 125 updates with batch-size 200, reaching a test accuracy of around $91\%$ using the mean $\mathbf{m}_t$ . We use linear learning-rate decay from 0.005 to 0.001 for the mean $\mathbf{m}$ and a learning-rate of $10^{-5}$ for the precision $\mathbf{S}$ . The expectations are approximated using 3 samples drawn from the posterior. We plot sensitivities at iteration $t = 5, 10, 25, 125$ . For this example, we use samples from $q_t$ to compute the prediction variance and error (150 samples are used). We sort examples according to their sensitivity at iteration $t = 125$ and then plot their average sensitivities in 60 groups with 100 examples in each group.

For the experiments in Fig. 3(b), Fig. 11(b) and Fig. 11(c), we consider neural network models $\mathbf{f}(\boldsymbol{\theta}_{t})$ on FMNIST, MNIST and CIFAR10. We do not use training data augmentation. For CIFAR10 we use a ResNet–20. The expectations in the iBLR are approximated using a single sample drawn from the posterior. For prediction, we use the mean $m_{t}$ . The test accuracies are 91.3% for FMNIST, 98.5% for MNIST and 80.9% for CIFAR10. We use a cosine learning-rate scheduler with an initial learning-rate of 0.1 and anneal to zero over the course of training. Other experimental details are stated in Table 8. Similar to before, we use sampling to evaluate sensitivity (150 samples are used).

# J Author Contributions Statement

Authors list: Peter Nickl (PN), Lu Xu (LX), Dharmesh Tailor (DT), Thomas Moellenhoff (TM), Mohammad Emtiyaz Khan (MEK)

All co-authors contributed to developing the main idea. MEK and DT first discussed the idea deriving sensitivity measure based on the BLR. MEK derived the MPE and the results in Sec 3 and DT helped in connecting them to influence functions. PN derived the results in 3.3 and came up with the idea to predict generalization error with LOO-CV. LX adapted it to class-removal. PN wrote the code with help from LX. PN and LX did most of the experiments with some help from TM and regular feedback from everybody. TM did the experiment on the sensitivity evolution during training with some help from PN. All authors were involved in writing and proof-reading of the paper.

# K Differences Between Camera-Ready Version and Submitted Version

We made several changes to take the feedback of reviewers into account and improve the paper.

1. The writing and organization of the paper were modified to emphasize the generalization to a wide variety of models and algorithms and the applicability of MPE during training.   
2. The presentation was changed in Section 3 to emphasize the focus on the conjugate model. Detailed derivations were pushed to the appendices and more focus was put on big picture ideas. Arbitrary perturbations parts were made explicit. Table 1 was added and more focus was put on training algorithms.   
3. We added experiments using leave-one-out estimation to predict generalization on unseen test data during training. We also added results to study the evolution of sensitivities during training using MPE with iBLR.