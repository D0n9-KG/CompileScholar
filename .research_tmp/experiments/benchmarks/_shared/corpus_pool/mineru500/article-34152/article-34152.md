# Adaptive Sampling to Reduce Epistemic Uncertainty Using Prediction Interval-Generation Neural Networks

Giorgio Morales, John W. Sheppard

Gianforte School of Computing

Montana State University, Bozeman, MT 59717, USA

giorgiomorales@ieee.org; john.sheppard@montana.edu

# Abstract

Obtaining high certainty in predictive models is crucial for making informed and trustworthy decisions in many scientific and engineering domains. However, extensive experimentation required for model accuracy can be both costly and time-consuming. This paper presents an adaptive sampling approach designed to reduce epistemic uncertainty in predictive models. Our primary contribution is the development of a metric that estimates potential epistemic uncertainty leveraging prediction interval-generation neural networks. This estimation relies on the distance between the predicted upper and lower bounds and the observed data at the tested positions and their neighboring points. Our second contribution is the proposal of a batch sampling strategy based on Gaussian processes (GPs). A GP is used as a surrogate model of the networks trained at each iteration of the adaptive sampling process. Using this GP, we design an acquisition function that selects a combination of sampling locations to maximize the reduction of epistemic uncertainty across the domain. We test our approach on three unidimensional synthetic problems and a multi-dimensional dataset based on an agricultural field for selecting experimental fertilizer rates. The results demonstrate that our method consistently converges faster to minimum epistemic uncertainty levels compared to Normalizing Flows Ensembles, MC-Dropout, and simple GPs.

Code — https://github.com/NISL-MSU/AdaptiveSampling

# 1 Introduction

In various scientific and engineering fields, the development of accurate predictive models frequently relies on experimentation. Conducting these experiments can be costly and time-consuming, making it important to adopt strategies that extract the most valuable information from each experiment. One notable example is precision agriculture (PA) where experimental results may require an entire growing season to manifest, and only a portion of the field is allocated for such trials (Lawrence, Rew, and Maxwell 2015). This is exacerbated by the fact data can often only be collected every other year, due to crop rotation.

Adaptive sampling (AS) techniques offer a promising solution by selecting samples intelligently that contribute most to improving model accuracy and reducing uncertainty (Di Fiore, Nardelli, and Mainini 2024). This work focuses on sampling techniques designed to reduce uncertainty in the prediction models across the entire input domain. Such techniques are essential for enhancing trust in decision-making systems whose optimization processes rely on accurate prediction models. For instance, in PA, determining optimal fertilizer rates depends on the shape of estimated nitrogen-yield response (N-response) curves (Bullock and Bullock (1994), Morales and Sheppard (2023a)). These curves represent the estimated crop yield values at specific field sites in response to all admissible fertilizer rates. Uncertainty across the domain can severely affect the survey shapes, leading to unreliable recommended fertilizer rates.

We note a distinction between two types of uncertainty: epistemic and aleatoric. Epistemic uncertainty represents the portion of total uncertainty that can be reduced by gathering more information or improving the prediction model. On the other hand, aleatoric uncertainty is the inherent and irreducible component of uncertainty due to the random nature of the data itself (Hüllermeier and Waegeman 2021; Nguyen, Shaker, and Hüllermeier 2022). The total uncertainty associated with a prediction $(\sigma_{y}^{2})$ encapsulates both the aleatoric $(\sigma_{a}^{2})$ and epistemic $(\sigma_{e}^{2})$ components; i.e., $\sigma_{y}^{2} = \sigma_{a}^{2} + \sigma_{e}^{2}$ . Prediction intervals (PIs) offer a comprehensive representation of this total uncertainty by estimating the upper and lower bounds within which a prediction is expected to fall with a given probability (Khosravi et al. 2011).

Several methods have been proposed to reduce uncertainty through iterative sampling. However, the majority of these methods have been developed within the framework of active learning (AL) (Nguyen, Destercke, and Hüllermeier 2019; Berry and Meger 2023) or in contexts where the primary objective is to identify the location of local or global optima (Hennig and Schuler 2012; Nguyen et al. 2019).

It is important to note that AS and AL fields do not completely overlap (Di Fiore, Nardelli, and Mainini 2024). In AL, the objective is to select training data within a limited budget to maximize model performance. AL can be categorized into population-based AL, where the test input distribution is known, and pool-based AL, where a pool of unlabeled samples is provided. Our problem configuration does not align with those categories as it is not limited to predefined data pools or known distributions. Instead, it aims

to sample from an open domain continuously, focusing on reducing epistemic uncertainty across the entire input space.

In this paper, we propose a method to reduce epistemic uncertainty through adaptive sampling using PIs generated by neural networks (NNs). Our method, called Adaptive Sampling with Prediction-Interval Neural Networks (ASPINN), involves training a dual NN architecture comprising a target-estimation network and a PI-generation network. The objective of such NNs is to produce high-quality PIs that reflect both aleatoric and epistemic uncertainties. Our specific contributions are:

1. We introduce a novel metric based on NN-generated PIs to quantify potential levels of epistemic uncertainty.   
2. We present an AS method called ASPINN. At each iteration, it builds a Gaussian Process (GP) from calculated potential epistemic uncertainty levels. The GP, a surrogate for the NN models, estimates potential epistemic uncertainty changes across the domain after sampling specific locations. An acquisition function then uses the GP to select sampling locations, aiming to minimize global epistemic uncertainty throughout the input domain.   
3. We tackle a real-world application and present an AS benchmark problem that focuses on reducing the epistemic uncertainty of an agricultural field site.   
4. Our method is shown to converge faster to minimum epistemic uncertainty levels than the compared methods.

# 2 Related Work

The problem addressed in this work shares similarities with Bayesian Optimization (BO), where at each iteration, data points are sampled at locations expected to yield significant improvements in the objective function according to a specified acquisition function. BO methods build a probabilistic model of the objective function, often a GP, to select the most promising points for evaluation (Garnett 2023).

Traditional BO methods explore the domain space sequentially; however, Gonzalez et al. (2016) proposed a batch sampling strategy for BO that accounts for the interactions between different evaluations in the batch using a penalized acquisition function. Some BO strategies focus on maximizing information gain. For instance, Wang and Jegelka (2017) introduced an acquisition function called max-value entropy search (MES), which balances exploration of areas with higher uncertainty in the surrogate model and exploitation towards the believed optimum. In addition, Nguyen et al. (2019) presented the predictive variance reduction search (PVRS) strategy, which reduces uncertainty at perceived optimal locations, leading to convergence when uncertainty at all perceived optimal locations is minimized.

In typical BO applications, the objective is to identify a single location that corresponds to the local or global optimum of an objective function ( $\arg\max f(\mathbf{x})$ ). In contrast, the solution to our problem consists of an augmented dataset that yields minimum epistemic uncertainty across the entire input space. In the fertilizer rate optimization problem discussed in the previous section, finding the rate that produces the higher estimated yield value does not necessarily coincide with the economic optimum nitrogen rate (EONR). The EONR is the N rate beyond which there is no actual profit for the farmers and its calculation depends on the shape of the N-response curves (Bullock and Bullock 1994). Therefore, the epistemic uncertainty across all admissible N rates should be reduced to provide reliable EONR recommendations for future growing seasons.

Similarly, active learning is closely related to this work. The primary distinction is that AL, given known input distributions (population-based AL) or a set of unlabeled points (pool-based AL), aims to select the minimum number of training examples to maximize model performance (Di Fiore, Nardelli, and Mainini 2024). In contrast, our approach is agnostic of the input distribution and is not restricted to a fixed pool of training candidates. Furthermore, our focus being on reducing uncertainty only considers model prediction improvement as a side-effect. What is more, it allows for repetitive sampling at a single location.

Despite the distinction above, some AL techniques can be adapted to our problem. In particular, we are interested in methods that decompose uncertainty into its aleatoric and epistemic components. A common approach is to use Monte-Carlo Dropout (MC-Dropout) (Gal and Ghahramani 2016) to quantify epistemic uncertainty in NNs. MC-Dropout uses dropout repeatedly to select random subsamples of active nodes in the network, turning a single network into an ensemble. Hence, epistemic uncertainty is represented by the sample variance of the ensemble predictions.

Furthermore, Valdenegro-Toro and Mori (2022) used a variance attenuation (VA) loss function to disentangle the epistemic and aleatoric components from the outputs of ensemble models. However, Zhang et al. (2024) pointed out that VA-based methods overestimate aleatoric uncertainty. In response, they presented a denoising approach that involves incorporating a variance approximation module into a trained prediction model to identify the aleatoric uncertainty. Finally, Berry and Meger (2023) proposed using an ensemble of normalizing flows (NFs), created using dropout masks, to estimate both aleatoric and epistemic uncertainty. To demonstrate their results, they suggested an AL framework that compares various uncertainty estimation methods. These methods are used to sample multiple-point candidates and select those with the highest epistemic uncertainty.

# 3 Proposed Method

In this work, we examine a system defined by an input vector $\mathbf{x} \in \mathbb{R}^d$ and a scalar response $y \in \mathbb{R}$ . The system's underlying function $f: \mathcal{X} \to \mathcal{Y}$ maps the input value space and the response value space such that $y = f(\mathbf{x}) + \varepsilon_a(\mathbf{x})$ , where $\varepsilon_a(\mathbf{x})$ is a random variable representing the error term that is a function of the system's aleatoric uncertainty, $\sigma_a^2(\mathbf{x})$ .

Let $\mathcal{D}_t = (\mathbf{X}_{obs}^{(t)},\mathbf{Y}_{obs}^{(t)})$ represent the dataset available at iteration $t$ consisting of $n_t$ observations, where $\mathbf{X}_{obs}^{(t)} = \{\mathbf{x}_1,\dots ,\mathbf{x}_{n_t}\}$ and $\mathbf{Y}_{obs}^{(t)} = \{y_1,\dots ,y_{n_t}\}$ . A prediction model $\hat{f}_t:\mathcal{X}\to \mathcal{Y}$ with parameters $\theta_f$ is trained by minimizing the mean squared error of the estimation:

$$
\min _ {\boldsymbol {\theta} _ {f}} \frac {1}{n _ {t}} \sum_ {(\mathbf {x} _ {i}, y _ {i}) \in \mathcal {D} _ {t}} (\hat {f} _ {t} (\mathbf {x} _ {i}) - y _ {i}) ^ {2}.
$$

![](images/701f4913069633acb61e5e925eeeb67726a7bec9bd5f009f610d6af99e3e3113.jpg)  
Figure 3.1: Epistemic uncertainty minimization through AS.

We aim to identify a batch $\mathbf{X}_{acq}^{(t)} = \{\mathbf{x}_{t,1}, \ldots, \mathbf{x}_{t,B}\}$ of B recommended sampling locations for the next iteration. These locations are chosen to minimize the epistemic uncertainty across the entire input space given a model $\hat{f}_{t}$ trained on $D_{t}$ . The epistemic uncertainty, $\sigma_{e}^{2}(\mathbf{x}_{p})$ , arises from the lack of knowledge about f and is due to the limitations of the prediction model trained on the observed dataset.

Preferences over potential sampling locations are encoded by an acquisition function $\alpha_{t}(\mathbf{x})$ . Suppose $J(\mathcal{D}_{t})$ is a function that reflects the total potential epistemic uncertainty across the domain. Then $\alpha_{t}(\mathbf{x})$ is designed to reflect the expected decrease in epistemic uncertainty $\mathbb{E}[J(\mathcal{D}_{t}) - J(\mathcal{D}_{t} \cup (\mathbf{x}, y))]$ after making an observation at location x. Fig. 3.1 depicts an instance of our problem. Here, $x^{*}$ represents the selected sampling position at each iteration (i.e., B = 1). For the general case where B > 1, the decision on where to sample the k-th element of the batch, $x_{t,k}$ , depends on the estimated effect of the previous k - 1 samples of the same batch. This requires a batch sampling strategy, which will be explored in this paper.

In the following, we describe the components of our AS-PINN method. We lay out the steps to derive a metric that reflects the epistemic uncertainty associated with an input value based on PIs. The metric is then used to design an acquisition function that allows for the selection of a batch of sampling locations, which are expected to minimize the global epistemic uncertainty during the next AS iteration.

# 3.1 Prediction Interval Generation

We generate PIs for quantifying the total uncertainty associated with a given sample, thus accounting for both aleatoric and epistemic uncertainty. We employ an NN-based PI generation method called DualAQD (Morales and Sheppard 2023b). This method uses two companion NNs: a target-estimation NN and a PI-generation NN, whose computed functions are denoted as $\hat{f}_t(\cdot)$ and $\hat{g}_t(\cdot)$ , respectively. Network $\hat{f}_t(\cdot)$ is trained on $\mathcal{D}_t$ to minimize the target estimation error so that $\hat{y} = \hat{f}_t(\mathbf{x})$ and $\hat{y} \approx y$ . Network $\hat{g}_t(\cdot)$ produces two outputs $[\hat{y}^\ell, \hat{y}^u] = \hat{g}_t(\mathbf{x})$ , which correspond to the PI lower and upper bounds. Note that $\hat{g}_t(\mathbf{x})$ makes no assumptions about the underlying uncertainty distribution.

Network $\hat{g}_{t}(\cdot)$ is trained using the DualAQD loss function to produce high-quality PIs that are as narrow as possible while capturing some specified proportion of the predicted data points (e.g., 95%). However, the model should produce wider PIs for out-of-distribution (OOD) samples since these samples are not well-represented in the training set, leading to higher associated epistemic uncertainty. To address this, the bias weights of $\hat{g}_{t}(\cdot)$ are initialized to generate wide PIs, similar to the approach proposed by Liu et al. (2022). The rationale is that these bias weights will decrease during training for in-distribution samples, resulting in narrower PIs, but will remain high for OOD samples, ensuring appropriately wider PIs to reflect the increased uncertainty.

# 3.2 Potential Epistemic Uncertainty

Let $\sigma_e^2 (\mathbf{x}_p)$ represent the epistemic uncertainty at a certain location $\mathbf{x}_p\in \mathcal{X}$ . The PI lower and upper bounds generated by NN $\hat{g}_t(\cdot)$ at $\mathbf{x}_p$ are denoted as $\hat{y}_t^\ell (\mathbf{x}_p)$ and $\hat{y}_t^u (\mathbf{x}_p)$ , respectively. We claim that using PIs alone does not provide sufficient information to determine $\sigma_e^2 (\mathbf{x}_p)$ . Consider $\mathbf{x}_p$ as an OOD sample. We may state that the total uncertainty associated with $\mathbf{x}_p$ is primarily due to epistemic uncertainty given the lack of knowledge of the prediction model about the system's behavior in this region of the input domain.

However, we cannot estimate the aleatoric uncertainty around $x_{p}$ until we gather observations in such domain region. Alternative methods can be used but they require making assumptions about the noise distribution (Seitzer et al. 2022), training an ensemble of models (Berry and Meger 2023), or using additional trainable modules (Zhang et al. 2024). Therefore, the total uncertainty conveyed by the interval $[\hat{y}_{t}^{\ell}(\mathbf{x}_{p}), \hat{y}_{t}^{u}(\mathbf{x}_{p})]$ cannot be split effectively into its epistemic and aleatoric components without further information.

Instead of attempting to provide a metric that accurately estimates $\sigma_e^2 (\mathbf{x}_p)$ directly, we propose a metric that reflects the potential levels of epistemic uncertainty. Let $\mathcal{N}(\mathbf{x}_p) = \{\mathbf{x}\in \mathbf{X}_{obs}^{(t)}|\| \mathbf{x} - \mathbf{x}_p\| _2\leq \theta \}$ denote a neighborhood that considers all samples whose Euclidean distance to $\mathbf{x}_p$ is less than a hyperparameter threshold $\theta$ . We create the set of input-response pairs $\mathcal{R}(\mathcal{N}(\mathbf{x}_p)) = \{(\mathbf{x},y)|(\mathbf{x},y)\in \mathcal{D}_t,\mathbf{x}\in \mathcal{N}(\mathbf{x}_p),\hat{y}^\ell (\mathbf{x})\leq y\leq \hat{y}^u (\mathbf{x})\}$ using the samples in $\mathcal{N}(\mathbf{x}_p)$ whose response values fall within their corresponding PI. Thus, we present the metric $Q_{t}(\mathbf{x}_{p})$ , defined as:

$$
Q _ {t} \left(\mathbf {x} _ {p}\right) = \left\{ \begin{array}{l l} \min _ {\left(\mathbf {x}, y\right) \in \mathcal {R} \left(\mathcal {N} \left(\mathbf {x} _ {p}\right)\right)} \left(\hat {y} ^ {u} (\mathbf {x}) - y\right) + & \\ \min _ {\left(\mathbf {x}, y\right) \in \mathcal {R} \left(\mathcal {N} \left(\mathbf {x} _ {p}\right)\right)} \left(y - \hat {y} ^ {\ell} (\mathbf {x})\right) & \text { if } \mathcal {N} \left(\mathbf {x} _ {p}\right) \neq \emptyset \\ \hat {y} _ {t} ^ {u} \left(\mathbf {x} _ {p}\right) - \hat {y} _ {t} ^ {\ell} \left(\mathbf {x} _ {p}\right) & \text { if } \mathcal {N} \left(\mathbf {x} _ {p}\right) = \emptyset \end{array} \right. \tag {1}
$$

The local neighborhood of $x_{p}$ may contain important contextual information that an analysis at a single location $x_{p}$ cannot capture. For instance, Fig.3.2a illustrates an interval $\mathrm{PI}(\mathbf{x}_{p}) = [\hat{y}_{t}^{\ell}(\mathbf{x}_{p}), \hat{y}_{t}^{u}(\mathbf{x}_{p})]$ generated at a single location. Suppose $Q_{t}(\mathbf{x}_{p})$ is calculated using $\mathrm{PI}(\mathbf{x}_{p})$ only (i.e., $\theta = 0$ ). Since a single point lies within the interval, $Q_{t}(\mathbf{x}_{p})$ is equal to the PI width, indicating that the epistemic uncertainty at $x_{p}$ can potentially be completely reduced. Fig.3.2b depicts a case in which the PI shown in Fig.3.2a is located in a region of the domain with low data density. As such, there exists an epistemic component that entails that the PI width could be reduced by acquiring more data in this region.

![](images/e69c9a143eea3b1829b4b0251624a0bbf7512f56c3b087d488f8eb54ec862937.jpg)

<details>
<summary>scatter</summary>

| x_p | y     | Data Point Type |
|-----|-------|-----------------|
| X_p | ŷ^u(x_p) | Data points at location X_p |
| X_p | ŷ^ℓ(x_p) | Data points at location X_p |
| X_p | 0     | Data points at location X_p |
| X_p | 1     | Data points at location X_p |
| X_p | 2     | Data points at location X_p |
| X_p | 3     | Data points at location X_p |
| X_p | 4     | Data points at location X_p |
| X_p | 5     | Data points at location X_p |
| X_p | 6     | Data points at location X_p |
| X_p | 7     | Data points at location X_p |
| X_p | 8     | Data points at location X_p |
| X_p | 9     | Data points at location X_p |
| X_p | 10    | Data points at location X_p |
| X_p | 11    | Data points at location X_p |
| X_p | 12    | Data points at location X_p |
| X_p | 13    | Data points at location X_p |
| X_p | 14    | Data points at location X_p |
| X_p | 15    | Data points at location X_p |
| X_p | 16    | Data points at location X_p |
| X_p | 17    | Data points at location X_p |
| X_p | 18    | Data points at location X_p |
| X_p | 19    | Data points at location X_p |
| X_p | 20    | Data points at location X_p |
| X_p | 21    | Data points at location X_p |
| X_p | 22    | Data points at location X_p |
| X_p | 23    | Data points at location X_p |
| X_p | 24    | Data points at location X_p |
| X_p | 25    | Data points at location X_p |
| X_p | 26    | Data points at location X_p |
| X_p | 27    | Data points at location X_p |
| X_p | 28    | Data points at location X_p |
| X_p | 29    | Data points at location X_p |
| X_p | 30    | Data points at location X_p |
| X_p | 31    | Data points at location X_p |
| X_p | 32    | Data points at location X_p |
| X_p | 33    | Data points at location X_p |
| X_p | 34    | Data points at location X_p |
| X_p | 35    | Data points at location X_p |
| X_p | 36    | Data points at location X_p |
| X_p | 37    | Data points at location X_p |
| X_p | 38    | Data points at location X_p |
| X_p | 39    | Data points at location X_p |
| X_p | 40    | Data points at location X_p |
| X_p | 41    | Data points at location X_p |
| X_p | 42    | Data points at location X_p |
| X_p | 43    | Data points at location X_p |
| X_p | 44    | Data points at location X_p |
| X_p | 45    | Data points at location X_p |
| X_p | 46    | Data points at location X_p |
| X_p | 47    | Data points at location X_p |
| X_p | 48    | Data points at location X_p |
| X_p | 49    | Data points at location X_p |
| X_p | 50    | Data points at location X_p |
| X_p | 51    | Data points at location X_p |
| X_p | 52    | Data points at location X_p |
| X_p | 53    | Data points at location X_p |
| X_p | 54    | Data points at location X_p |
| X_p | 55    | Data points at location X_p |
| X_p | 56    | Data points at location X_p |
| X_p | 57    | Data points at location X_p |
| X_p | 58    | Data points at location X_p |
| X_p | 59    | Data points at location X_p |
| X_p | 60    | Data points at location X_p |
| X_p | 61    | Data points at location X_p |
| X_p | 62    | Data points at location X_p |
| X_p | 63    | Data points at location X_p |
| X_p | 64    | Data points at location X_p |
| X_p | 65    | Data points at location X_p |
| X_p | 66    | Data points at location X_p |
| X_p | 67    | Data points at location X_p |
| X_p | 68    | Data points at location X_p |
| X_p | 69    | Data points at location X_p |
| X_p | 70    | Data points at location X_p |
| X_p | 71    | Data points at location X_p |
| X_p | 72    | Data points at location X_p |
| X_p | 73    | Data points at location X_p |
| X_p | 74    | Data points at location X_p |
| X_p | 75    | Data points at location X_p |
| X_p | 76    | Data points at location X_p |
| X_p | 77    | Data points at location X_p |
| X_p | 78    | Data points at location X_p |
| X_p | 79    | Data points at location X_p |
| X_p | 80    | Data points at location X_p |
| Y   | -     | 95% PIs         |
| (a) - Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y   Y
(a) - (b) - (c) - (d) - (e) - (f) - (g) - (h) - (i) - (j) - (k) - (l) - (m) - (n) - (o) - (p) - (q) - (r) - (s) - (t) - (u) - (v) - (w) - (x) - (y) - (z) - (a) - (b) - (c) - (d) - (e) - (f) - (g) - (h) - (i) - (j) - (k) - (l) - (m) - (n) - (o) - (p) - (q) - (r) - (s) - (t) - (u) - (v) - (w) - (x) - (y) - (x) - (z) - (e) - (f) - (g) - (h) - (i) - (j) - (k) - (l) - (l) - (o) - (p) - (q) - (r) - (s) - (t) - (u) - (v) - (w) - (x) - (y) - (x) - (z) - (e) - (f) - (g) - (h) - (i) - (j) - (k) - (l) - (l) - (o) - (p) - (p) - (q) - (r) - (s) - (t) - (u) - (v) - (w) - (x) - (y) - (x) - (z) - (e) - (f) - (g) - (h) - (i) - (j) - (k) - (l) - (l) - (o) - (p) - (p) - (p) - (p) - (p) - (p) - (p) - (p) - (p)
</details>

Figure 3.2: PIs generated at location $x_{p}$ . (a) Data points located at $x_{p}$ only. (b) PI width is affected by epistemic uncertainty. (b) PI width is mainly due to aleatoric uncertainty.

Conversely, Fig.3.2c shows a similar PI in a high data density context. Here, a reduction in $\mathrm{PI}(\mathbf{x}_{p})$ will also lead to a decrease in the PI widths of adjacent locations, provided that the uncertainty at $x_{p}$ is not independent of its surroundings. However, model $\hat{g}_{t}(\cdot)$ is trained to produce narrow PIs while maintaining a nominal coverage (e.g., 95%). Thus, it will not reduce $\mathrm{PI}(\mathbf{x}_{p})$ if this reduction would result in several samples near the PI bounds being excluded from their intervals. Notice that if $\theta > 0$ , then $Q_{t}(\mathbf{x}_{p}) \approx 0$ , indicating minimal potential epistemic uncertainty around $x_{p}$ .

# 3.3 Batch Sampling

When multiple locations are sampled at each iteration, decisions for the entire batch are made based on the current model without observing any data from the batch until the next iteration. Hence, it is necessary to simulate the decisions that would be made under the equivalent sequential policy (i.e., when B = 1) (Gonzalez et al. 2016). In other words, the decision of selecting the k-th element of the t-th batch, $x_{t,k}$ , should incorporate the estimates of change in uncertainty after sampling at locations $x_{t,1}, \ldots, x_{t,k-1}$ (i.e., $x_{t,1:k-1}$ ). Following a greedy sampling strategy, we have:

$$
\mathbf {x} _ {t, k} = \underset {\mathbf {x} _ {p} \in \mathcal {X}} {\operatorname{argmax}} \alpha_ {t} (\mathbf {x} _ {p} \mid \mathbf {x} _ {t, 1: k - 1}). \tag {2}
$$

We consider an acquisition function that estimates the reduction in the total potential epistemic uncertainty across the domain when making an observation at a given location $x_{p}$ :

$$
\alpha_ {t} (\mathbf {x} _ {p} \mid \mathbf {x} _ {t, 1: k - 1}) = J \left(\mathcal {D} _ {t, k - 1}\right) - J \left(\mathcal {D} _ {t, k - 1} \cup \left(\mathbf {x} _ {p}, \hat {f} _ {t} (\mathbf {x} _ {p})\right)\right).
$$

$\mathcal{D}_{t,k-1}$ is the dataset $\mathcal{D}_t$ augmented with the first $k-1$ samples of the batch and their corresponding estimated response values. The potential epistemic uncertainty at x during the t-iteration after sampling the first k elements of the batch is denoted as $Q_{t,k}(\mathbf{x})$ . Thus, the total potential epistemic uncertainty is calculated as $J(\mathcal{D}_{t,k}) = \sum_{\mathbf{x} \in \mathcal{X}} Q_{t,k}(\mathbf{x})$ , where $J(\mathcal{D}_{t,0}) = J(\mathcal{D}_t)$ and $Q_{t,0}(\mathbf{x}) = Q_t(\mathbf{x})$ .

Thus, $J(\mathcal{D}_{t})$ is computed based on $Q_{t}(\mathbf{x})$ , which is derived from the outputs produced by NNs $\hat{f}_{t}(\cdot)$ and $\hat{g}_{t}(\cdot)$ (Eq. 1), trained on $D_{t}$ . To calculate $J(\mathcal{D}_{t,k-1} \cup (\mathbf{x}_{p}, \hat{f}_{t}(\mathbf{x}_{p})))$ in a similar manner, it is necessary to train both NNs on the augmented dataset $D_{t,k-1} \cup (\mathbf{x}_{p}, \hat{f}_{t}(\mathbf{x}_{p}))$ . According to Eq. 2, this operation would need to be repeated $\forall x_{p} \in X$ and $\forall k \in [1, \ldots, B]$ and, as such, becomes impractical. Therefore, motivated by most BO-based approaches, we use a GP as a surrogate model. The objective is to simulate, with low computational cost, how the potential epistemic uncertainty would be affected throughout the entire domain after observing a sample at a given position.

Let us define a GP $p(\hat{f}_{t}) = \mathcal{GP}(\mu_{t}, \mathbf{K}_{t})$ that serves as a surrogate model for $\hat{f}_{t}(\cdot)$ and its associated epistemic uncertainty during the t-th iteration. This GP is characterized by the mean function $\mu_{t}$ and the positive-definite covariance matrix $K_{t}$ . Functions $\mu_{t}$ and $K_{t}$ are initialized based on the estimations generated by $\hat{f}_{t}(\cdot)$ and $\hat{g}_{t}(\cdot)$ , trained on $D_{t}$ .

For the mean function, we consider $\mu_{t}(\mathbf{x}) = \hat{f}_{t}(\mathbf{x})$ . On the other hand, the diagonal elements of $K_{t}$ reflect the uncertainty in the predictions $\hat{f}_{t}(\mathbf{x})$ due to epistemic uncertainty. Since this uncertainty varies across the domain, it represents heteroscedastic noise. Considering that the uncertainty at a given position may be correlated with nearby positions, $K_{t}$ is structured as a matrix with non-zero off-diagonal elements. Thus, the scale of $K_{t}$ depends on location and is calculated according to the potential epistemic uncertainty:

$$
\mathbf {K} _ {t} (\mathbf {x}, \mathbf {x} ^ {\prime}) = \left\{ \begin{array}{l l} Q _ {t} (\mathbf {x}), & \text {if \mathbf {x} = \mathbf {x} ^{\prime}} \\ \rho (\mathbf {x}, \mathbf {x} ^ {\prime}) \sqrt {Q _ {t} (\mathbf {x}) Q _ {t} (\mathbf {x} ^ {\prime})}, & \text {otherwise,} \end{array} \right.
$$

where $\rho (\mathbf{x},\mathbf{x}^{\prime})$ indicates the correlation between positions $\mathbf{x}$ and $\mathbf{x}^{\prime}$ . We use the radial basis function (RBF) such that $\rho (\mathbf{x},\mathbf{x}^{\prime}) = e^{-\frac{\|\mathbf{x} - \mathbf{x}^{\prime}\|^{2}}{2r^{2}}}$ , where $r$ is a tunable hyperparameter.

Given we want to assess the impact of observing a data point at a given position $x_{p}$ , we condition the GP on the data point $(\mathbf{x}_{p},\hat{f}_{t}(\mathbf{x}_{p}))$ , resulting in a GP posterior $p(\hat{f}_{t}|(\mathbf{x}_{p},\hat{f}_{t}(\mathbf{x}_{p})))$ whose covariate matrix is denoted as $\mathbf{K}_{t}(\mathbf{x},\mathbf{x}^{\prime}|\mathbf{x}_{p})$ . In general, the covariance matrix when sampling the k-th element of the batch is denoted as $\mathbf{K}_{t}(\mathbf{x},\mathbf{x}^{\prime}|\mathbf{x}_{t,1},\ldots,\mathbf{x}_{t,k})$ and $Q_{t,k} = \text{diag}(\mathbf{K}_{t}(\mathbf{x},\mathbf{x}^{\prime}|\mathbf{x}_{t,1},\ldots,\mathbf{x}_{t,k}))$ .

Given $\mathbf{x}_p$ , the covariance matrix is updated as follows:

$$
\begin{array}{l} \mathbf {K} _ {t} (\mathbf {x}, \mathbf {x} ^ {\prime} \mid \mathbf {x} _ {p}) = \mathbf {K} _ {t} (\mathbf {x}, \mathbf {x} ^ {\prime}) - \\ \mathbf {K} _ {t} (\mathbf {x}, \mathbf {x} _ {p}) \mathbf {K} _ {t} (\mathbf {x} _ {p}, \mathbf {x} _ {p}) ^ {- 1} \mathbf {K} _ {t} (\mathbf {x} _ {p}, \mathbf {x} ^ {\prime}). \\ \end{array}
$$

Hence, the updated GP variance at $\mathbf{x}_p$ collapses to zero after observing a data point at that position. Note that this would only happen when $Q_{t}(\mathbf{x}_{p})$ reflects the level of epistemic uncertainty exclusively. In practice, this assumption may not hold. Nevertheless, it allows us to construct a heuristic that guides the search toward locations where new observations would potentially cause the greatest uncertainty reduction. The next sampling location is selected using Eq. 2 based on the total potential epistemic uncertainty after observing a data point at $\mathbf{x}_p$ , which is given by:

$$
J \left(\mathcal {D} _ {t} \cup (\mathbf {x} _ {p}, \hat {f} _ {t} (\mathbf {x} _ {p}))\right) = \sum \operatorname{diag} \left(\mathbf {K} _ {t} (\mathbf {x}, \mathbf {x} ^ {\prime} \mid \mathbf {x} _ {p})\right).
$$

# 4 Experimental Results

We compared ASPINN to three methods adapted for AS: Normalizing flows ensembles (NF-Ensemble) (Berry and Meger 2023), a standard GP (Gardner et al. 2018), and MC-Dropout (Gal and Ghahramani 2016). For our experiments,

Table 4.1: Functions and noise terms of the 1-D problems. 

<table><tr><td>Name</td><td>Function f(x)</td><td>Noise εa(x)</td></tr><tr><td>cos</td><td>10 + 5 cos(x + 2)</td><td>N(0, 2 + 2 cos(1.2x))</td></tr><tr><td>hetero</td><td>7 sin(x)</td><td>N(0, 3 cos(x/2))</td></tr><tr><td>cosqr</td><td>10 + 5 cos( $\frac{x^{2}}{5}$ )</td><td>N(0,  $\frac{1}{2}$ (1 -  $\frac{x^{2}}{100}$ ))</td></tr></table>

![](images/db1f8da1624b90b7490edeee0698c96b493760dad10ed9f3f8a1f885bf3c7d64.jpg)

<details>
<summary>line</summary>

| x    | f(x) | 95% PIs from ε(x) |
| ---- | ---- | ----------------- |
| -4   | 10   | 10                |
| -2   | 15   | 15                |
| 0    | 5    | 5                 |
| 2    | 10   | 10                |
| 4    | 15   | 15                |
| -4   | 10   | 10                |
| -2   | 15   | 15                |
| 0    | 5    | 5                 |
| 2    | 10   | 10                |
| 4    | 15   | 15                |
| -10  | 10   | 10                |
| -5   | 15   | 15                |
| 0    | 5    | 5                 |
| 5    | 10   | 10                |
| 10   | 15   | 15                |
</details>

Figure 4.1: Initial cos, hetero, and cosqr datasets and the ideal 95% PIs calculated from $\varepsilon_{a}(\mathbf{x})$ across the domain.

we considered three synthetic one-dimensional (1-D) regression problems and one multidimensional regression problem based on a real-world problem. We used synthetic problems given that, in AS, we are required to sample at locations with high uncertainty that could not have been observed previously. By utilizing problems with known underlying target and noise functions, which are unknown to the AS methods, we can simulate and evaluate accurately the performance improvements resulting from the decisions made by each method in previous iterations.

# 4.1 Experiments with One-Dimensional Data

We considered three 1-D problems: cos (Morales and Sheppard 2023b), hetero (Depeweg et al. 2018), and cosqr. All three problems are affected by heteroscedastic noise, and their function equations are shown in Table 4.1. Unlike most AL and AS approaches, we do not initiate the experiments from empty datasets. For each case, we generated incomplete datasets as initial states, as shown in Fig. 4.1. The motivation for this is to produce areas with low data density, which entails high epistemic uncertainty. Thus, methods that estimate potential epistemic uncertainty more accurately and select sampling locations designed to reduce such uncertainty should require fewer AS iterations to approximate the ground-truth distribution of the problem. Additional implementation details are provided in the Appendix.

For ASPINN, we trained feed-forward NNs with varying depths: two hidden layers with 100 units for problems cos and hetero; and three hidden layers with 500, 100, and 50 units, respectively, for cosqr. The networks $\hat{f}_t$ and $\hat{g}_t$ share the same architecture except for the last layer, as $\hat{f}_t$ uses one output, while $\hat{g}_t$ uses two outputs. Furthermore, ASPINN uses two hyperparameters: the neighbor distance threshold $\theta$ and the kernel length $r$ . We performed a grid search with the values $\theta = [0.1, 0.15, 0.2, 0.25]$ and $r = [0.1, 0.15, 0.2, 0.25]$ , and selected $\theta = 0.25$ and $r = 0.15$ for all experiments. DualAQD, the PI-generation method used by ASPINN, uses a hyperparameter $\eta$ as a scale factor to adapt the coefficient that balances the two objectives of the DualAQD loss function. We chose a scale factor $\eta =$ 0.1. Other $\eta$ values (i.e., $\{0.001, 0.005, 0.01, 0.05, 0.1\}$ ) achieved similar results but with slower convergence rates.

For MC-Dropout, we used the same architecture as the target-estimation NN in ASPINN. For NF-Ensemble, we used flows with 200 hidden units for problems cos and hetero and 300 hidden units for problem cosqr. We employed ensembles consisting of five models trained during 30,000 epochs. For the standard GP, we used the same RBF kernel used by ASPINN. We utilized an inference implementation based on black-box matrix-matrix multiplication (Gardner et al. 2018) that uses 3000 training epochs.

Our objective is to reduce the epistemic uncertainty with as few AS iterations as possible. We define the performance metric $PI_{\delta}^{(t)}$ to quantify epistemic uncertainty relative to the ground truth at the t-th iteration:

$$
P I _ {\delta} ^ {(t)} = \frac {1}{| \mathcal {X} |} \sum_ {\mathbf {x} \in \mathcal {X}} \left(| y ^ {u} (\mathbf {x}) - \hat {y} _ {t} ^ {u} (\mathbf {x}) | + | y ^ {\ell} (\mathbf {x}) - \hat {y} _ {t} ^ {\ell} (\mathbf {x}) |\right).
$$

Here, $y^{\ell}(\mathbf{x})$ and $y^{u}(\mathbf{x})$ represent the ideal lower and upper PI bounds, respectively, calculated from the aleatoric noise function: $y^{u}(\mathbf{x}) = f(\mathbf{x}) + 1.96 \varepsilon_{a}(\mathbf{x})$ and $y^{\ell}(\mathbf{x}) = f(\mathbf{x}) - 1.96 \varepsilon_{a}(\mathbf{x})$ . This metric is applicable to problems with normally distributed aleatoric noise, which is the case for the problems evaluated in this work. However, none of the tested methods make assumptions about the noise distribution. Note that if $PI_{\delta}^{(t)} = 0$ , the estimated PIs match the ideal intervals, implying that the model's epistemic uncertainty has been minimized, and the total uncertainty is purely aleatoric. A non-zero $PI_{\delta}^{(t)}$ indicates a discrepancy between the estimated and ideal PIs, suggesting the presence of epistemic uncertainty. The greater the $PI_{\delta}^{(t)}$ , the higher the epistemic uncertainty. To ensure fairness, $\hat{y}_{t}^{\ell}(\mathbf{x})$ and $\hat{y}_{t}^{u}(\mathbf{x})$ are generated by an independent NN, $\hat{g}_{t}(\cdot)$ , trained on the dataset $D_{t}$ produced by each compared method at each iteration. Regardless of the uncertainty estimation model used by each method, we trained an additional PI-generation NN using the DualAQD loss to maintain a consistent uncertainty metric across all comparisons.

It is worth mentioning that other works have used different evaluation approaches. For instance, Berry and Meger (2023) employed an approach where they sampled 50 random locations from the domain. For each location, they generated 1000 samples using the ground-truth distribution and 1000 samples using the distribution predicted by each method. They then calculated the Kullback-Leibler divergence between the ground truth and the model-generated distributions. However, we believe this approach does not provide a consistent basis for evaluation, as each method employs different mechanisms for estimating uncertainty.

For our experiments, the AS process was executed for each problem for 50 iterations. This process is repeated 10 times, initializing the problems with a different seed each time. Figure 4.2 depicts an initial state of problem $\cos$ along with the augmented datasets during iterations $t = 7$ , 40. The figure also displays the corresponding calculated potential epistemic uncertainty for all values of the input domain. Figure 4.3 shows the evolution of the mean $PI_{\delta}^{(t)}$ value and its

![](images/a2fbab124ce22bba39c17bd5ee183d47fc8a9aa30fdd6b419af0fc0cb749b8ce.jpg)  
Figure 4.2: Example of the adaptive sampling process using ASPINN on the cos problem.

![](images/ce6cd53b5e0bc5a17c280f7cb3694b5819063049ba05e021eaf186334271539c.jpg)

<details>
<summary>line</summary>

| t  | 'cos' | 'hetero' | 'cosqr' |
|----|-------|----------|---------|
| 0  | 2.5   | 2.5      | 0.6     |
| 10 | 2.3   | 2.4      | 0.5     |
| 20 | 2.1   | 2.2      | 0.4     |
| 30 | 2.0   | 2.1      | 0.3     |
| 40 | 1.9   | 2.0      | 0.3     |
| 50 | 1.8   | 1.9      | 0.3     |
</details>

Figure 4.3: Evolution of the mean $PI_{\delta}^{(t)}$ value and its corresponding standard deviation for the 1-D problems.

corresponding standard deviation, calculated across the values obtained from the 10 repetitions at each t. In addition, we calculated the area under the uncertainty curve (AUUC) for each learning curve. For each problem, Table 4.2 gives the average AUUC for the four methods and corresponding standard deviations. The bold entries indicate the method that achieved the lowest average AUUC value and that its difference with respect to the values obtained by the other methods is statistically significant according to a paired t-test performed at the 0.05 significance level.

# 4.2 Experiments with Simulated Field Data

In this section, we present a multi-dimensional problem that simulates a real-world agricultural field site. A field site is defined as a specific area within a larger field (e.g., a $10 \times 10m$ ). It is used for precise monitoring and management to address local variations in soil and crop conditions.

Note that actual real-world data cannot be considered for a comparative AS study. There are multiple reasons for this. First, a given field site receives a single experimental rate during the fertilization stage and its effects are observed during the harvest season (e.g., five months for winter wheat). Second, additional samples at the same site require collect-

Table 4.2: AUUC comparison for the 1-D problems 

<table><tr><td>Problem</td><td>MCDropout</td><td>GP</td><td>NF-Ensemble</td><td>ASPINN</td></tr><tr><td>cos</td><td>112.57±24.20</td><td>123.87±26.22</td><td>113.39±19.49</td><td>97.26±7.87</td></tr><tr><td>hetero</td><td>113.80±13.38</td><td>110.21±13.59</td><td>106.44±16.26</td><td>85.95±9.11</td></tr><tr><td>cosqr</td><td>30.39±3.56</td><td>23.12±5.55</td><td>25.60±2.67</td><td>17.13±1.42</td></tr></table>

ing data over multiple years. Third, when comparing different AS methods, they may produce different experimental rates, which cannot be implemented simultaneously in a single season. Fourth, real-world conditions, such as unforeseen environmental factors and concept drift, introduce additional complexity, making it difficult to isolate the AS strategies' effects. Therefore, simulations based on the properties of a real field provide a controlled environment where different AS methods can be evaluated under identical conditions, allowing for a fair comparison.

In previous work, we derived the functional form of N-response curves of different management zones (MZs) from an actual winter wheat field as symbolic skeleton expressions using a Multi-Set Transformer (Morales and Sheppard 2024). An MZ is defined as a distinct sub-region that encompasses sites with relative homogeneity and, thus, similar fertilizer responsivity (i.e., similar response to varying fertilizer rates). A symbolic skeleton expression is a representation of a mathematical expression that captures its structural form without setting specific numerical values. For instance, the relationship between yield, y, and N rate, $x^{Nr}$ , at a given site is given by the skeleton $y = c_{1} + c_{2} \tanh(c_{3} + c_{4} x^{Nr})$ , where $c_{1} - c_{4}$ are placeholder constants. In this work, we propose to use a simulated field site from an MZ whose underlying function is based on the previous skeleton.

In particular, we consider the following yield function:

$$
y = f (\mathbf {x}) = \frac {\mathbf {x} ^ {P}}{1 5} + \left(\frac {\mathbf {x} ^ {A}}{\pi} + 1\right) \tanh \left(\frac {0 . 1 \mathbf {x} ^ {N r}}{3 \mathbf {x} ^ {V H} + 2}\right) + \varepsilon_ {a} (\mathbf {x}),
$$

where $x = [\mathbf{x}^{P}, \mathbf{x}^{A}, \mathbf{x}^{VH}, \mathbf{x}^{Nr}]$ comprises the following site-specific covariates: annual precipitation (mm), terrain aspect (radians), Sentinel-1 backscattering coefficient from the Vertical Transmit-Horizontal Receive Polarization band, and applied N rate (lbs/ac), respectively. The aleatoric noise is modeled as $\varepsilon_{a}(\mathbf{x}) = \mathcal{N}(0, (\mathbf{x}^{P} + \mathbf{x}^{Nr}) / 150)$ . Further details on the selection of these underlying and noise functions are available in the Appendix.

While this yield regression problem considers four explanatory variables, the only one that farmers can control is $x^{Nr}$ . Therefore, the AS search is focused along the $x^{Nr}$ axis to determine the best experimental N rate for reducing epistemic uncertainty. A field site receives a single fertilizer treatment; thus, we consider B = 1. The AS process was conducted over 50 iterations, with each iteration representing a different year or growing season, corresponding to a randomly generated precipitation value $\mathbf{x}^{P} \sim U(75, 150)$ . All compared methods used the same sequence of precipitation values throughout the AS process. $x^{A}$ describes topographic information of the field so it is assumed to remain constant throughout all iterations. In contrast, $x^{VH}$ , associated with soil moisture, was modeled as a function of precipitation and topographic aspect. Additional details on data

Table 4.3: AUUC comparison for the simulated field site 

<table><tr><td>MCDropout</td><td>GP</td><td>NF-Ensemble</td><td>ASPINN</td></tr><tr><td>614.68 ± 112.48</td><td>593.54 ± 107.42</td><td>730.80 ± 74.63</td><td>496.85±71.65</td></tr></table>

![](images/2e1faa7929d82c1d17a7fc87133c8b1f835dc4466ae20642c147fce59b65a8af.jpg)

<details>
<summary>line</summary>

| t  | ASPINN | NF-Ensemble | GP  | MCDropout |
|----|--------|-------------|-----|-----------|
| 0  | 4.8    | 6.2         | 5.5 | 5.3       |
| 5  | 4.5    | 5.8         | 5.0 | 4.9       |
| 10 | 3.8    | 5.0         | 4.2 | 4.0       |
| 15 | 3.2    | 4.5         | 3.5 | 3.3       |
| 20 | 2.8    | 4.0         | 3.0 | 2.9       |
| 25 | 2.5    | 3.5         | 2.7 | 2.6       |
| 30 | 2.2    | 3.0         | 2.4 | 2.3       |
| 35 | 2.0    | 2.8         | 2.2 | 2.1       |
| 40 | 1.8    | 2.5         | 2.0 | 1.9       |
| 45 | 1.5    | 2.2         | 1.8 | 1.7       |
| 50 | 1.2    | 2.0         | 1.5 | 1.4       |
</details>

Figure 4.4: Evolution of the mean $PI_{\delta}^{(t)}$ value and its corresponding standard deviation for the simulated field site.

generation are provided in the Appendix.

We applied the AS process ten times. At each iteration, we used a unique initialization seed and evaluated the epistemic uncertainty along the allowed N rates (i.e., 0, 30, 60, 90, 120, and 150 lbs/ac) under the current field conditions. Table 4.3 presents the average AUUC values and corresponding standard deviations, highlighting the best-performing method in bold. Figure 4.4 depicts the evolution of the mean $PI_{\delta}^{(t)}$ values, calculated based on the results from the ten repetitions.

# 5 Discussion

The ASPINN method involves training a PI-generation NN, which is used to design a novel potential epistemic uncertainty metric. This metric is then used in our batch sampling strategy to determine the sequence of sampling locations most likely to reduce epistemic uncertainty the greatest across the input domain.

When evaluating ASPINN on the tested 1-D problems, as shown in Fig. 4.3, we observed that it produced learning curves with faster convergence rates and lower standard deviation than the other methods. Although the confidence bands exhibit some overlap, this is attributed to outliers with high $PI_{\delta}^{(t)}$ values generated by other methods (e.g., GP), which increase the variance. Nevertheless, it is important to note that the learning curves for ASPINN consistently remain below those of the other methods across all iterations and have narrower confidence bands. Thus, the difference in AUUC values is shown to be statistically significant according to the t-test, as shown in Table 4.2. Also from Fig. 4.3, we notice that ASPINN generated constantly decreasing and smoother learning curves. Conversely, other methods, such as MC-Dropout, tend to oversample certain regions of the input domain, leading to imbalanced datasets. This oversampling results in overfitting in those regions while causing a poor fit in others, producing unstable learning curves.

Furthermore, the experiments conducted on the simulated field data exhibit consistent behavior with the results from the 1-D problems In particular, Table 4.3 demonstrates that ASPINN achieves the lowest AUUC values, and the differences between ASPINN and the compared methods are statistically significant. Given that the precipitation values vary at each iteration, the resulting learning curves are expected to exhibit multiple peaks and valleys rather than a smooth, consistently decreasing trend, as observed in Fig 4.4. This variability arises because higher precipitation values are associated with increased uncertainty levels, leading to more pronounced fluctuations in the learning curves. Considering that the sequence of precipitation values is not the same for all AS repetitions, Fig 4.4 reports only the mean curve and not the confidence bands. This is because the $PI_{\delta}^{(t)}$ values obtained by a method across different iterations are generated from contexts that could correspond to extreme opposites, leading to high variance values that do not necessarily reflect the method's performance. Despite this behavior, we observed that ASPINN consistently produced learning curves that remained below those of the compared methods.

One limitation of our approach is that it does not handle multi-modal aleatoric noise inherently. Multi-modal noise indicates that the data variability comes from different underlying sources, each contributing to a different mode in the noise distribution. In such cases, it would be necessary to use a PI-generation method capable of producing multiple upper and lower bounds based on the identified number of modes. Note, however, that the contributions proposed in this paper are not reliant on a specific PI-generation method. In the presence of multiple PIs, we would need to adapt the epistemic uncertainty metric accordingly and execute the remaining steps similarly. Another limitation, which also applies to the compared methods, is the computational cost when dealing with high-dimensional problems due to the need to evaluate all potential locations in the input space. We plan to address this limitation in future work.

# 6 Conclusion

Accurate predictive modeling is essential in many scientific and engineering disciplines, where decisions often rely on data gathered from costly and time-consuming experiments. This is especially true in fields like precision agriculture, where data collection is limited by factors such as growing seasons and crop rotation. In such contexts, reducing uncertainty in prediction models is necessary for optimizing outcomes and ensuring reliable decision-making. Addressing this challenge, our work focuses on minimizing epistemic uncertainty through adaptive sampling techniques.

We introduced ASPINN, an adaptive sampling technique designed to reduce epistemic uncertainty across an input domain using prediction intervals generated by neural networks. The novel potential epistemic uncertainty metric, central to ASPINN, provided a robust basis for guiding the sampling process. The effectiveness of our approach was demonstrated through its consistent ability to achieve faster convergence rates with lower and more stable learning curves compared to other methods. This was observed across all tested scenarios, including 1-D synthetic problems and a multi-dimensional problem that simulates an agricultural field site based on real-world winter wheat data.

In the future, we plan on adapting ASPINN for problems

affected by both heteroskedastic and multi-modal noise. In particular, this would involve integrating PI-generation techniques capable of addressing multi-modal noise functions and refining the potential epistemic uncertainty metric to account for multiple PIs at a single location.

# 7 Acknowledgments

This research was supported by the Data Intensive Farm Management project (USDA-NIFA-AFRI 2016-68004-24769 and USDA-NRCS NR213A7500013G021). Computational efforts were performed on the Tempest HPC System, operated by University Information Technology Research Cyberinfrastructure at MSU.

# References

Berry, L.; and Meger, D. 2023. Normalizing Flow Ensembles for Rich Aleatoric and Epistemic Uncertainty Modeling. Proceedings of the AAAI Conference on Artificial Intelligence, 37(6): 6806–6814.   
Bullock, D. G.; and Bullock, D. S. 1994. Quadratic and Quadratic-Plus-Plateau Models for Predicting Optimal Nitrogen Rate of Corn: A Comparison. Agronomy Journal, 86(1): 191–195.   
Depeweg, S.; Hernandez-Lobato, J.-M.; Doshi-Velez, F.; and Udluft, S. 2018. Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, 1184–1193.   
Di Fiore, F.; Nardelli, M.; and Mainini, L. 2024. Active Learning and Bayesian Optimization: A Unified Perspective to Learn with a Goal. Archives of Computational Methods in Engineering.   
Gal, Y.; and Ghahramani, Z. 2016. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. In Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, 1050–1059. New York, USA.   
Gardner, J. R.; Pleiss, G.; Bindel, D.; Weinberger, K. Q.; and Wilson, A. G. 2018. GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, 7587–7597. Red Hook, NY, USA.   
Garnett, R. 2023. Bayesian Optimization. Cambridge University Press.   
Gonzalez, J.; Dai, Z.; Hennig, P.; and Lawrence, N. 2016. Batch Bayesian Optimization via Local Penalization. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, volume 51, 648–657. Cadiz, Spain.   
Hennig, P.; and Schuler, C. J. 2012. Entropy search for information-efficient global optimization. J. Mach. Learn. Res., 13: 1809–1837.   
Hüllermeier, E.; and Waegeman, W. 2021. Aleatoric and epistemic uncertainty in machine learning: an introduction

to concepts and methods. Machine Learning, 110(3): 457-506.   
Khosravi, A.; Nahavandi, S.; Creighton, D. C.; and Atiya, A. F. 2011. Lower Upper Bound Estimation Method for Construction of Neural Network-Based Prediction Intervals. IEEE Transactions on Neural Networks, 22(3): 337–346.   
Lawrence, P. G.; Rew, L. J.; and Maxwell, B. D. 2015. A probabilistic Bayesian framework for progressively updating site-specific recommendations. Precision Agriculture, 16(3): 275–296.   
Liu, S.; Zhang, P.; Lu, D.; and Zhang, G. 2022. PI3NN: Out-of-distribution-aware Prediction Intervals from Three Neural Networks. In International Conference on Learning Representations.   
Morales, G.; and Sheppard, J. W. 2023a. Counterfactual Explanations of Neural Network-Generated Response Curves. In International Joint Conference on Neural Networks, 01–08.   
Morales, G.; and Sheppard, J. W. 2023b. Dual Accuracy-Quality-Driven Neural Network for Prediction Interval Generation. IEEE Transactions on Neural Networks and Learning Systems, 1–11.   
Morales, G.; and Sheppard, J. W. 2024. Univariate Skeleton Prediction in Multivariate Systems Using Transformers. In Machine Learning and Knowledge Discovery in Databases: Research Track. ECML PKDD 2024.   
Nguyen, V.; Gupta, S.; Rana, S.; Thai, M.; Li, C.; and Venkatesh, S. 2019. Efficient Bayesian Optimization for Uncertainty Reduction Over Perceived Optima Locations. In 2019 IEEE International Conference on Data Mining (ICDM), 1270–1275.   
Nguyen, V.-L.; Destercke, S.; and Hüllermeier, E. 2019. Epistemic Uncertainty Sampling. In Kralj Novak, P.; Šmuc, T.; and Džeroski, S., eds., Discovery Science, 72–86. Cham: Springer International Publishing.   
Nguyen, V.-L.; Shaker, M. H.; and Hüllermeier, E. 2022. How to measure uncertainty in uncertainty sampling for active learning. Machine Learning, 111(1): 89–122.   
Seitzer, M.; Tavakoli, A.; Antic, D.; and Martius, G. 2022. On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks. In International Conference on Learning Representations.   
Valdenegro-Toro, M.; and Mori, D. 2022. A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 1508–1516. Los Alamitos, CA, USA.   
Wang, Z.; and Jegelka, S. 2017. Max-value entropy search for efficient Bayesian Optimization. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, 3627–3635. JMLR.org.   
Zhang, W.; Ma, Z. M.; Das, S.; Weng, T.-W. L.; Megretski, A.; Daniel, L.; and Nguyen, L. M. 2024. One Step Closer to Unbiased Aleatoric Uncertainty Estimation. Proceedings of the AAAI Conference on Artificial Intelligence, 38(15):16857–16864.

# Appendix

In this supplementary material, we provide pseudocode for the main functions of our method, Adaptive Sampling with Prediction-Interval Neural Networks (ASPINN). In addition, we present experimental details for the synthetic 1-D and simulated field site problems presented in the paper.

# A.1 ASPINN Algorithms

Algorithm A.1 describes the function PotEpistUnc that calculates the potential epistemic uncertainty at each candidate position for sampling. It takes as inputs the dataset $\mathcal{D}_{t} = (\mathbf{X}_{obs}^{(t)}, \mathbf{Y}_{obs}^{(t)})$ available during the t-th iteration of the adaptive sampling (AS) process, the prediction interval (PI)-generation neural network (NN) $g_{t}(\cdot)$ trained on $D_{t}$ , the set $X_{test}$ of all candidate positions for sampling (i.e., the input space), and the neighbor distance threshold $\theta$ .

For each candidate position $x_{p}$ , the algorithm constructs a neighborhood $\mathcal{N}(\mathbf{x}_{p})$ consisting of all samples in $\mathbf{X}_{obs}^{(t)}$ within a radius of $\theta$ with respect to $x_{p}$ . If $\mathcal{N}(\mathbf{x}_{p})$ is empty, the potential epistemic uncertainty is given by the PI width $\hat{y}_{t}^{u}(\mathbf{x}_{p}) - \hat{y}_{t}^{\ell}(\mathbf{x}_{p})$ . Otherwise, it builds the set of input–response pairs $\mathcal{R}(\mathcal{N}(\mathbf{x}_{p}))$ using the samples in $\mathcal{N}(\mathbf{x}_{p})$ whose response values fall within their corresponding PI. From the data points in $\mathcal{R}(\mathcal{N}(\mathbf{x}_{p}))$ , the potential epistemic uncertainty $Q_{t}(\mathbf{x}_{p})$ is calculated as the sum of the minimum distance between the predicted upper bounds and the observed values, and the minimum distance between the observed values and the predicted lower bounds.

Furthermore, ASPINN's batch sampling strategy is shown in Algorithm A.2. It takes as inputs the set $X_{test}$ of all candidate positions for sampling, their corresponding potential epistemic uncertainty values $Q_{t}$ during the $t$ -th iteration of the AS process, the batch size $B$ , and the kernel length $r$ . In Lines 3-11, the algorithm initializes the covariance matrix $\mathbf{K}_t$ of a Gaussian Process (GP) surrogate model. The diagonal of $\mathbf{K}_t$ is set to be equal to $Q_{t}$ . Since the uncertainty at a given position may be correlated with nearby positions, $\mathbf{K}_t$ is structured as a matrix with non-zero off-diagonal elements. The off-diagonal elements combine the potential epistemic uncertainty values at different positions based on their correlation value, which is calculated using a radial base function (RBF) with kernel length $r$ .

Once $K_{t}$ is initialized, we assess the potential uncertainty reduction when observing each candidate position $x_{p}$ . Specifically, we condition the GP on a data point at $x_{p}$ and update its covariance matrix as shown in Line 16. The total potential epistemic uncertainty across the domain, $J(\mathcal{D}_{t})$ , is determined by summing the diagonal elements $K_{t}$ . The estimated reduction in the total potential epistemic uncertainty, when making an observation at $x_{p}$ , is thus calculated as the difference $\delta J$ between the sum of the diagonal elements of $K_{t}$ before and after the observation (Line 18). Therefore, the k-th element of the t-th batch, $x_{t,k}$ , is selected as the position that yields the greatest $\delta J$ value.

# A.2 Experiments with One-Dimensional Data

In this section, we first provide details on the generation of the 1-D test problems. We also present additional experimental results and corresponding plots.

Algorithm A.1: ASPINN's potential epistemic uncertainty   
1: function POTEPISTUNC( $\mathcal{D}_t, g_t, X_{test}, \theta$ )
2: $\left( \mathbf{X}_{obs}^{(t)}, \mathbf{Y}_{obs}^{(t)} \right) \leftarrow \mathcal{D}_t$ 3: $Q_t \leftarrow \text{zeros}(\text{size}(X_{test}))$ 4:    for $\mathbf{x}_p \in X_{test}$ do
5: $\mathcal{N}(\mathbf{x}_p) \leftarrow \{\mathbf{x} \in \mathbf{X}_{obs}^{(t)} | \| \mathbf{x} - \mathbf{x}_p \|_2 \leq \theta\}$ 6:    if $\mathcal{N}(\mathbf{x}_p) \neq \emptyset$ then
7: $\mathcal{R}(\mathcal{N}(\mathbf{x}_p)) \leftarrow [] \quad \triangleright \mathbf{x}_p$ 's neighbors falling within the PIs
8: $Y_{sub}^u, Y_{sub}^\ell \leftarrow [], []$ 9:    for $\mathbf{x} \in \mathcal{N}(\mathbf{x}_p)$ do
10: $\hat{y}^\ell(\mathbf{x}), \hat{y}^u(\mathbf{x}) \leftarrow g_t(\mathbf{x})$ 11:    if $\hat{y}^\ell(\mathbf{x}) \leq y \leq \hat{y}^u(\mathbf{x})$ then $\triangleright (\mathbf{x}, y) \in \mathcal{D}_t$ 12: $\mathcal{R}(\mathcal{N}(\mathbf{x}_p)).append((\mathbf{x}, y))$ 13: $Y_{sub}^u.append(\hat{y}^u(\mathbf{x}))$ 14: $Y_{sub}^\ell.append(\hat{y}^\ell(\mathbf{x}))$ 15: $(X_{sub}, Y_{sub}) \leftarrow \mathcal{R}(\mathcal{N}(\mathbf{x}_p))$ 16: $Q_t(\mathbf{x}_p) \leftarrow min(Y_{sub}^u - Y_{sub}) + min(Y_{sub} - Y_{sub}^\ell)$ 17:    else
18: $\hat{y}^\ell(\mathbf{x}_p), \hat{y}^u(\mathbf{x}_p) \leftarrow g_t(\mathbf{x}_p)$ 19: $Q_t(\mathbf{x}_p) \leftarrow \hat{y}_t^u(\mathbf{x}_p) - \hat{y}_t^\ell(\mathbf{x}_p)$ 20:    return $Q_t$

Data Generation Below, we report the admissible search space for each problem:

- $\cos: \mathcal{X} = \left\{-5 + \frac{10(i - 1)}{99}, |i = 1,2,\ldots,100\right\}$   
- hetero: $\mathcal{X} = \left\{-4.5 + \frac{9(i - 1)}{299}, |i = 1,2,\ldots,300\right\}$   
- $\cos \mathrm{qr}$ : $\mathcal{X} = \left\{-10 + \frac{20(i - 1)}{499}, |i = 1,2,\ldots,500\right\}$

For the case of the $\cos$ problem, we generate the initial set of observations $\mathbf{X}_{obs}^{(t = 0)}$ by uniformly sampling 200 elements from the discrete set $\mathcal{X}$ .

The initial datasets corresponding to the hetero problem are generated as recommended by Depeweg et al. (2018). In particular, a mixture of three Gaussian is created with means $\mu_{1} = -4$ , $\mu_{2} = 0$ , and $\mu_{3} = 4$ and corresponding variances $\sigma_{1} = \frac{2}{5}$ , $\sigma_{2} = 0.9$ , and $\sigma_{1} = \frac{2}{5}$ . Each Gaussian component is equally weighted. We considered an initial dataset size of $|\mathbf{X}_{obs}^{(t=0)}| = 200$ .

For the $\cos qr$ problem, the initial dataset is generated by first sampling 2,000 elements from the discrete set X uniformly and then applying a series of masks to select specific ranges of values. The process is as follows:

- Elements in the intervals $[-10, -8)$ , $[-5, -2)$ , $[3, 6)$ , and $[7, 10]$ are included in the dataset directly.   
- Additional elements are selected from the intervals $[-8, -5)$ , $[-2, 3)$ , and $[6, 7)$ with specific sizes of $1, 10$ , and $3$ elements, respectively.

By doing so, we aim to generate a complex dataset with different areas with low data density, as depicted in Fig A.1. Note that the low-density regions correspond to distinct

Algorithm A.2: ASPINN's batch sampling method   
1: function SAMPLE( $X_{test}, Q_t, B, r$ )
2: $n_Q \leftarrow \text{size}(Q_t)$ 3: $K_t \leftarrow \text{zeros}(n_Q, n_Q)$ $\triangleright$ Init GP's covariance matrix
4: for $i \in (0, n_Q)$ do
5:    for $j \in (i, n_Q)$ do
6:    if i = j then
7: $k \leftarrow Q_t(X_{test}[i])$ 8:    else
9: $\rho \leftarrow \text{RBF}(X_{test}[i], X_{test}[j]; r) \quad \triangleright$ r: kernel size
10: $k \leftarrow \rho \sqrt{Q_t(X_{test}[i]) Q_t(X_{test}[j])}$ 11: $K_t(i, j) = K_t(k, i) = k$ 12: $\mathbf{X}_{acq}^{(t)} \leftarrow []$ 13: while size( $\mathbf{X}_{acq}^{(t)} < B$ do $\triangleright$ Batch sampling loop
14: $\Delta J_{\max} \leftarrow []$ 15: for $x_p \in X_{test}$ do
16: $K_t' \leftarrow K_t(x, x') - K_t(x, x_p) K_t(x_p, x_p)^{-1}$ 17: $K_t(x_p, x')$ $\triangleright \forall K_t(x, x') \in K_t$ 18: $\Delta J \leftarrow \sum diag(K) - \sum diag(K')$ 19: if $\Delta J > \Delta J_{\max}$ then
20: $\Delta J_{\max} \leftarrow \Delta J$ 21: $x_{t,k} \leftarrow x_p$ 22: $K_{best} \leftarrow K_t'$ 23: $\mathbf{X}_{acq}^{(t)}.append(x_{t,k})$ 24: $K_t \leftarrow K_t'$ 25: return $\mathbf{X}_{acq}^{(t)}$

behaviors in both the function $f(\mathbf{x})$ and the noise $\varepsilon_{a}(\mathbf{x})$ . This approach ensures that the AS process remains focused on capturing meaningful variations in the data, rather than merely estimating data density for selecting future sample locations. For example, the intervals $[-8,-5)$ and $[6,7)$ each contain only a single observed point. However, the former covers an entire oscillation of the function, whereas the latter spans a much smaller range. Consequently, when using a PI-generation neural network to analyze the unobserved areas, there is a greater discrepancy between the estimated and ideal PIs in the first case. All PI-generation NNs are trained using a specialized loss function called DualAQD (Morales and Sheppard 2023b).

Furthermore, the initial dataset size for problem cosqr varies according to the selected initialization seed. Specifically, the obtained sizes $|\mathbf{X}_{obs}^{(t=0)}|$ for the ten AS iterations are: 1,102, 1,106, 1,123, 1,078, 1,114, 1,163, 1,084, 1,159, 1,079, and 1,104, respectively.

Experimental Results In the paper, we reported that ASPINN achieved the lowest average area under the uncertainty curve (AUUC) for the learning curves obtained for each problem. In addition, its difference with respect to the values obtained by the compared methods was found to be statistically significant according to a paired t-test performed at the 0.05 significance level. The compared methods are Normalizing flows ensembles (NF-Ensemble) (Berry and Meger 2023), a standard GP (Gardner et al. 2018), and MC-Dropout (Gal and Ghahramani 2016).

To demonstrate these results, Table A.1 presents the p-

![](images/fad5f088ce0b5ef1938ae586d429bdd4f1e75b3011122a09021be1eae26f96c4.jpg)

Figure A.1: cosqr problem. (a) An initial generated dataset and the ideal 95% PIs calculated from $\varepsilon_{a}(\mathbf{x})$ across the domain. (b) Initial PIs estimated using DualAQD.   
Table A.1: Statistical significance tests — 1-D problems. p-values obtained comparing ASPINN to the other methods. 

<table><tr><td>Compared Method</td><td>cos</td><td>hetero</td><td>cosqr</td></tr><tr><td>NF-Ensemble</td><td>4.4E-2 (↑)</td><td>6.4E-3 (↑)</td><td>5.3E-6 (↑)</td></tr><tr><td>GP</td><td>5.4E-3 (↑)</td><td>8.6E-4 (↑)</td><td>7.7E-3 (↑)</td></tr><tr><td>MC-Dropout</td><td>3.8E-2 (↑)</td><td>5.7E-4 (↑)</td><td>3.1E-6 (↑)</td></tr></table>

values from the paired t-tests comparing ASPINN with the other methods. Here, the upward-pointing arrow ( $\uparrow$ ) indicates that ASPINN performed significantly better (i.e., p-value < 0.05).

Furthermore, we illustrate some of the results obtained by ASPINN for problems cos, hetero, and cosqr in Figures A.2, A.3, and A.4, respectively. The figures show the problems' initial state and the augmented datasets obtained during iterations t = 5, t = 20, and t = 35. They also display the corresponding calculated potential epistemic uncertainty $Q_{t}(\mathbf{x})$ for all values of the input domain.

# A.3 Experiments with Simulated Field Data

It is important to note that AS techniques are applied in contexts where experts can perform experiments in some selected locations of the input domain. Thus, candidate sampling locations are limited. For example, besides the 1-D problems presented by Berry and Meger (2023), they considered a multi-dimensional problem called Pendulum. This problem consists of four input variables with one of them, torque, representing the action variable (i.e., the agent's only control is over the torque applied). In the same fashion as the Pendulum problem, we presented experiments for a multidimensional problem (i.e., four inputs) simulating a real-world agricultural site, where the action variable is given by the fertilizer rate applied. However, our problem is more complex than Pendulum.

The behavior of this site is modeled based on a symbolic skeleton that was extracted from an actual winter wheat field (Morales and Sheppard 2024):

$$
y = c _ {1} + c _ {2} \tanh (c _ {3} + c _ {4} \mathbf {x} ^ {N r}), \tag {A.3}
$$

where $c_{1} - c_{4}$ are placeholder constants that may depend on

![](images/048047877f0d539e4715fc8da6fa527a3091056e1170ecfc045ce654bcc8a0a0.jpg)

Figure A.2: Adaptive sampling process using ASPINN on the cos problem.   
![](images/f99f0db8def809094a1fffe2145339fe55e1faebfab8561badbc9f2419f73864.jpg)

Figure A.3: Adaptive sampling process using ASPINN on the hetero problem.   
![](images/437d7c3e13dba9431e0883372e288efa519a4706e6c7cd61680dff883bf2d4a5.jpg)  
Figure A.4: Adaptive sampling process using ASPINN on the cosqr problem.

other explanatory variables.

To facilitate reference, we reproduce the yield function considered in this work:

$$
y = f (\mathbf {x}) = \frac {\mathbf {x} ^ {P}}{1 5} + \left(\frac {\mathbf {x} ^ {A}}{\pi} + 1\right) \tanh \left(\frac {0 . 1 \mathbf {x} ^ {N r}}{3 \mathbf {x} ^ {V H} + 2}\right) + \varepsilon_ {a} (\mathbf {x}), \tag {A.4}
$$

where $x = [\mathbf{x}^{P}, \mathbf{x}^{A}, \mathbf{x}^{VH}, \mathbf{x}^{Nr}]$ comprises the following site-specific covariates: annual precipitation (mm), terrain aspect (radians), Sentinel-1 backscattering coefficient from the Vertical Transmit-Horizontal Receive Polarization band, and applied N rate (lbs/ac), respectively. The aleatoric noise is modeled as $\varepsilon_{a}(\mathbf{x}) = \mathcal{N}(0, (\mathbf{x}^{P} + \mathbf{x}^{Nr}) / 1500)$ .

We considered $x^{P} \in [75, 150]$ , $x^{A} \in [\pi/4, \pi/2]$ , $x^{VH} \in [0.5, 1]$ , and $x^{Nr} \in [0, 30, 60, 90, 120, 150]$ . These values were selected to reflect realistic conditions based on past observations from the sub-region of the field used to model our simulated field site. During each iteration of the process, we sample a new precipitation value such that $\mathbf{x}_{t}^{P} \sim \mathcal{U}(75, 150)$ . Similarly, we accounted for variations in the terrain aspect by modeling $x_{t}^{A}$ as $\mathcal{U}(\pi/4, \pi/2)$ . This assumption reflects slight alterations in the landscape each growing season, influenced by factors such as weather conditions and the use of heavy machinery.

Variable $x^{VH}$ is associated with soil moisture content, where lower values correspond to drier soil conditions. Given the absence of additional variables to model soil moisture accurately and since this is beyond the scope of our study, we developed a simplified moisture function that incorporates precipitation and terrain aspect. In particular, we consider $x_{t}^{VH} = \frac{x_{t}^{P}}{150}x_{t}^{A}$ , reflecting that higher precipitation and terrain aspect values result in greater soil moisture content. Based on this parameterization, the initial dataset $\mathbf{X}_{obs}^{(t=0)}$ is generated by randomly sampling 50 data points, each representing a distinct growing season.

Here, we justify the selection of these underlying and noise functions. From comparing Equations A.3 and A.4, it is observed that the constant placeholders were assigned the following values: $c_{1} = \frac{x^{P}}{15}$ , $c_{2} = (\frac{x^{A}}{\pi} + 1)$ , $c_{3} = 0$ , and $c_{4} = (\frac{0.1}{3x^{VH} + 2})$ . Below, we analyze each of these expressions. It is important to clarify that our goal is not to derive precise functional expressions for the coefficients $c_{1}-c_{4}$ in order to model the underlying function of the field accurately. Rather, our aim is to design a yield function that exhibits behavior consistent with agronomic principles, informed by past observations of an actual field.

In a previous work (Morales and Sheppard 2023a), we utilized counterfactual explanations to analyze the influence of a set of “passive features” over the shape of the response curves generated for the response variable and a selected “active feature.” In the context of this work, the agricultural field site represents a multivariate system. We are interested in the analysis of nitrogen-yield response (N-response) curves. N-response curves are tools that allow for the analysis of the site-specific responsivity to all admissible values of the N fertilizer rate, which serves as the selected “active feature.” Nevertheless, the shape of N-response curves may be influenced not only by the relationship between the response variable and the active feature but also by other factors, termed "passive features." In this case, we consider the variables $\mathbf{x}^{P}$ , $\mathbf{x}^{A}$ , and $\mathbf{x}^{VH}$ as passive features.

In (Morales and Sheppard 2023a), we studied an early-yield prediction dataset of winter wheat. The findings indicate that, although precipitation $x^{P}$ is a critical factor for crop production, it has minimal impact on N responsivity. This suggests that $x^{P}$ is independent of the other features and only shifts the N-response curves vertically without altering their shape. In Eq. A.3, $c_{1}$ acts as an independent term responsible for vertical shifts, which is why it is modeled as a function of $x^{P}$ . In addition, variables $x^{A}$ and $x^{VH}$ were identified as having a significant impact on the shape of N-response curves, making them key factors in this study.

Furthermore, $c_{2}$ stretches the N-response curves vertically. We argue this behavior corresponds to that of the terrain aspect $x^{A}$ (i.e., the slope orientation). In terrain with varying elevations located in the Northern Hemisphere, regions that are facing north and east have limited sunlight during the day and are more prone to snow retention. These are factors that may affect the responsiveness of the fertilizer. For instance, we observed that regions facing north ( $x^{A} = 0$ ) correspond to flatter N-response curves than those facing south ( $x^{A} = \pi$ ). Our simulated field site is being modeled as a field site that is located within a subregion of an actual field whose $x^{A}$ values vary between $\pi/4$ and $\pi/2$ . Within this sub-region, we found that considering $c_{2} = (\frac{x^{A}}{\pi} + 1)$ adjusts reasonably well to the variation in vertical stretching of the estimated N-response curves.

Coefficient $c_{3}$ causes horizontal shifts, which are not observed in the estimated N-response curves obtained for the studied area. Hence, for the sake of simplicity, we select $c_{3}$ to be equal to 0. Finally, $c_{4}$ controls the horizontal stretching of the curve. A lower $c_{4}$ value causes the output of the function to increase more gradually as x increases. Conversely, a higher value of $c_{4}$ leads to a steeper increase, causing the function to reach its saturation point more rapidly. Dry soil has a lower capacity to retain and absorb nutrients and, thus, reaches the saturation point more quickly than moist soil when applying N fertilizer. Therefore, we model $c_{4}$ as an inverse function of $\mathbf{x}^{VH}: c_{4} = \left( \frac{0.1}{3\mathbf{x}^{VH} + 2} \right)$ .

On the other hand, the heteroskedastic aleatoric noise is modeled as $\varepsilon_{a}(\mathbf{x}) = \mathcal{N}(0, (\mathbf{x}^{P} + \mathbf{x}^{Nr})/1500)$ . This is based on the observations that both precipitation $x^{P}$ and N fertilizer rate $x^{Nr}$ contribute to variability and uncertainty in agricultural yield outcomes. That is, increased precipitation can lead to greater variability in soil conditions, such as runoff, which in turn affects nutrient availability and crop health. In addition, the effects of the N fertilizer do not always increase monotonically. High levels of nitrogen can lead to diminishing returns or even negative effects, such as nutrient imbalances or environmental stress on the plants. These unpredictable responses add to the uncertainty in yield, particularly at higher N fertilizer rates.

Finally, we assess the differences in AUUC values achieved by ASPINN across the ten AS iterations compared to those obtained by the other methods. Table A.2 reports the p-values from the paired t-tests comparing ASPINN with

Table A.2: Statistical significance tests — Simulated field site. p-values obtained comparing ASPINN to the other methods. 

<table><tr><td>Compared Method</td><td>p-value</td></tr><tr><td>NF-Ensemble</td><td>1.3E-4 (↑)</td></tr><tr><td>GP</td><td>4.4E-2 (↑)</td></tr><tr><td>MC-Dropout</td><td>6.7E-3 (↑)</td></tr></table>

the alternative approaches. The results indicate that the differences in AUUC values are statistically significant (i.e., p-value < 0.05).