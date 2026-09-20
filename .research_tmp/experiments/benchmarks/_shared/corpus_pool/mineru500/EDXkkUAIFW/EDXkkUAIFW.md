# One-shot Active Learning Based on Lewis Weight Sampling for Multiple Deep Models

Sheng-Jun Huang $^{1}$ , Yi Li $^{2}$ , Yiming Sun $^{2}$ , and Ying-Peng Tang $^{*1}$

$^{1}$ College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics, {huangsj, tangyp}@nuaa.edu.cn

$^{2}$ School of Physical and Mathematical Sciences, Nanyang Technological University, {yili, yiming005}@ntu.edu.sg

October 3, 2024

# Abstract

Active learning (AL) for multiple target models aims to reduce labeled data querying while effectively training multiple models concurrently. Existing AL algorithms often rely on iterative model training, which can be computationally expensive, particularly for deep models. In this paper, we propose a one-shot AL method to address this challenge, which performs all label queries without repeated model training.

Specifically, we extract different representations of the same dataset using distinct network backbones, and actively learn the linear prediction layer on each representation via an $\ell_{p}$ -regression formulation. The regression problems are solved approximately by sampling and reweighting the unlabeled instances based on their maximum Lewis weights across the representations. An upper bound on the number of samples needed is provided with a rigorous analysis for $p \in [1, +\infty)$ .

Experimental results on 11 benchmarks show that our one-shot approach achieves competitive performances with the state-of-the-art AL methods for multiple target models.

# 1 Introduction

The rapid advancements in deep learning have led to a substantial increase in demand for extensive labeled data points to effectively train high-performance models. However, data labeling remains costly due to its reliance on human labor. To address this challenge, active learning (AL) $[37, 35]$ has emerged as an effective strategy to mitigate annotation costs. This approach estimates the potential utility of different unlabeled instances in improving the performance of a target model and selectively queries the labels of the most beneficial instances from the oracle (i.e., an expert who can provide the ground-truth label). A typical practice of AL conducts label querying and model updating iteratively to exploit the insights from model decisions, i.e., selecting one or a small batch of instances based on the model predictions and updating the target model in each iteration until the labeling budget is exhausted $[11, 20]$ . This paradigm has been widely applied in real-world scenarios $[19, 39]$ .

Recently, there has been a significant surge in the demand for the deployment of machine learning systems on diverse resource-constrained devices $[13, 18, 30]$ . For example, speech recognition and face recognition systems usually need to support various types of machines with varying computing and memory resources. As a result, the task of training multiple

models with varying complexities using the same labeled dataset has arisen $[3]$ , leading to a new setting of AL where multiple target models are to be learned simultaneously $[42]$ .

Tang and Huang [42] provide both theoretical and empirical evidence showcasing the potential of AL in alleviating the substantial data labeling burden associated with training multiple target models. They propose an iterative AL algorithm DIAM and validate its effectiveness for multiple deep models. However, the use of iterative AL methods results in a significant increase in model training cost. This is due to the requirement of training multiple deep models at each query iteration. A potential solution is increasing the querying batch size of conventional batch-mode AL methods. Nevertheless, this may lead to redundant querying [48]. A more cost-effective strategy could be one-shot or single-shot querying, which selects the required number of unlabeled instances and makes all label queries within one iteration devoid of retraining the models.

Most existing one-shot AL methods query a representative set of instances using the distance between feature vectors $[48, 44, 21, 40]$ . However, this approach faces challenges when handling multiple deep models, as the same instance can exhibit different feature representations in different models. This phenomenon arises due to the intrinsic representation learning of deep models, where data representations are implicitly optimized during the training process and varied network architectures yield distinct embeddings. These embeddings may contain abundant information to facilitate data selection. However, such information has not been well exploited by existing one-shot AL methods. Therefore, they may not yield optimal performance in the setting of multiple models.

In this paper, we propose a one-shot AL method for multiple deep models, accompanied by rigorous theoretical analysis. Our method is based on the fact that a deep model can be viewed as a linear prediction layer (i.e., multiple neuron models) and a nonlinear feature extractor (i.e., the network backbone). Therefore, training multiple deep models can be described as learning linear prediction layers from the outputs of distinct network backbones. In this way, active learning from diverse data representations can be formulated as optimizing a shared sampling matrix to minimize the error of each linear predictor. To facilitate computation and analysis, we consider the learning of the prediction layer as an $\ell_{p}$ regression problem with $p \in (0, +\infty)$ . In particular, our empirical studies place particular emphasis on the case of p = 2, i.e. squared loss, which is one of the most commonly used loss functions in deep learning. Specifically, suppose that there are k models and $A^{j} \in R^{n \times d}$ ( $j = 1, \ldots, k$ ) is the feature matrix obtained by feeding the dataset into the j-th network backbone. Let $f : R \to R$ be an L-Lipschitz function with $f(0) = 0$ . Typical choices of f are activation functions such as ReLU, Sigmoid, and so on. We abuse the notation and apply f to a vector $v \in R^{n}$ coordinatewise, i.e. $f(\boldsymbol{v}) = (f(\boldsymbol{v}_{1}), \ldots, f(\boldsymbol{v}_{n}))^{T}$ . Suppose that $y^{1}, \ldots, y^{c} \in R^{n}$ are c label vectors and the task is to minimize the loss $\sum_{i=1}^{c} \|f(A^{j}\theta^{ij}) - y^{i}\|_{p}^{p}$ over $\theta^{1j}, \ldots, \theta^{cj} \in R^{d}$ for all models j simultaneously. Since the construction of S is independent of $y^{1}, \ldots, y^{c}$ , we henceforth assume that c = 1, with a single label vector $y \in R^{n}$ . Therefore, we seek a shared reweighted sampling matrix S such that we can, from the labels of the sampled instances Sy, approximately solve the regression problem $\min_{\boldsymbol{\theta}} \|f(A^{j}\boldsymbol{\theta}) - \boldsymbol{y}\|_{p}^{p}$ for all models j simultaneously.

The simplest case is when there is a single model, i.e., k = 1. In this case, Gajjar et al. [15] are the first to study the problem of actively learning a single neuron model. They cast the problem as a least-squares regression problem (i.e. p = 2) $\min_{\boldsymbol{\theta}} \|f(A\boldsymbol{\theta}) - \boldsymbol{y}\|_{2}^{2}$ and find an $\tilde{\theta}$ such that

$$
\| f (A \tilde {\pmb {\theta}} ^ {j}) - \pmb {y} \| _ {2} ^ {2} \leq C \cdot \big (\| f (A \pmb {\theta} ^ {*}) - \pmb {y} \| _ {2} ^ {2} + \epsilon L ^ {2} \| A \pmb {\theta} ^ {*} \| _ {2} ^ {2} \big),
$$

where $\boldsymbol{\theta}^{*} = \arg\min_{\boldsymbol{\theta}} \|f(A\boldsymbol{\theta}) - \boldsymbol{y}\|_{2}^{2}$ is the minimizer, C is an absolute constant and $\epsilon$ is an accuracy parameter. Recall that L is the Lipschitz constant of f. Gajjar et al. [15] also show that the additive term $\epsilon L^{2} \|A\boldsymbol{\theta}^{*}\|_{2}^{2}$ is necessary. For k > 1 and general p, we seek approximate solutions $\tilde{\theta}_{1}, \ldots, \tilde{\theta}_{k}$ with the following error guarantee of a similar form on each individual

model:

$$
\left\| f \left(A ^ {j} \tilde {\boldsymbol {\theta}} ^ {j}\right) - \boldsymbol {y} \right\| _ {p} ^ {p} \leq C \cdot \left(\left\| f \left(A ^ {j} \boldsymbol {\theta} ^ {j}\right) - \boldsymbol {y} \right\| _ {p} ^ {p} + \epsilon L ^ {p} \| A ^ {j} \boldsymbol {\theta} ^ {j} \| _ {p} ^ {p}\right), \tag {1}
$$

where $\pmb{\theta}^{j} = \arg \min_{\pmb{\theta}} \|f(A^{j}\pmb{\theta}) - \pmb{y}\|_{p}^{p}$ is the minimizer for model $j$ and $C = C(p) > 0$ is a constant depending only on $p$ . Gajjar et al.[15] construct $S$ to be a leverage score sampling matrix and solve $\tilde{\pmb{\theta}} = \arg \min_{\pmb{\theta} \in E} \|f(SA\pmb{\theta}) - S\pmb{y}\|_{2}^{2}$ with $E = \{\pmb{\theta} : \|SA\pmb{\theta}\|_{2}^{2} \leq \|Sy\|_{2}^{2}/(\epsilon L^{2})\}$ . At the core of their argument lies the classical fact that such an $S$ gives an $\ell_2$ subspace embedding for $A$ , i.e., $\|SA\pmb{\theta}\|_{2} \approx \|A\pmb{\theta}\|_{2}$ for all $\pmb{\theta}$ simultaneously. In fact, it is not necessary to sample the rows of $A$ according to the exact leverage scores $\tau_1(A), \ldots, \tau_n(A)$ ; any sampling probability proportional to $t_i \gtrsim \tau_i(A)$ for $i$ -th row will suffice, with the number of samples being proportional to $\sum_{i} t_{i}$ . This very fact motivates us to tackle the task of data selection from diverse representations by sampling the rows according to the maximum of leverage scores across $A^{j}$ 's, i.e., letting $t_{i} \sim \max_{j} \tau_{i}(A^{j})$ . Solving for each model $j$ by $\tilde{\pmb{\theta}}^{j} = \arg \min_{\pmb{\theta} \in E^{j}} \|f(SA^{j}\pmb{\theta}) - S\pmb{y}\|_{2}^{2}$ with $E^{j} = \{\pmb{\theta} : \|SA^{j}\pmb{\theta}\|_{2}^{2} \leq \|Sy\|_{2}^{2}/(\epsilon L^{2})\}$ will then achieve (1) for $p = 2$ . This indicates that the queried instances are effective in learning each of the linear predictors, which fits our problem well. A potential caveat is that the number of samples needed will be proportional to $\sum_{i} t_{i} \sim \sum_{i} \max_{j} \tau_{i}(A^{j})$ , which could be as large as $kd$ . However, empirical studies show that this is not the case for real-world datasets (see Section 3.2) and our approach will thus be efficient.

For general p, instead of leverage scores, it is natural to consider Lewis weights, which can be seen as generalizations of leverage scores for general p (see Section 3.1 for the definition). It is known that an $\ell_{p}$ Lewis weight sampling matrix S give an $\ell_{p}$ subspace embedding, i.e., $\|SA\theta\|_{p} \approx \|A\theta\|_{p}$ for all $\theta$ simultaneously [10]. The approach mentioned above extends to general p naturally, attaining (1) for general p, by sampling according to the maximum Lewis weights and solving an $\ell_{p}$ -regression problem for $\tilde{x}^{j}$ with an $\ell_{p}$ -version of $E^{j}$ .

Theoretical Results. For k = 1, Gajjar et al. devised an algorithm using $\tilde{O}(d/\epsilon^{4})$ queries [17], with an analysis specific to p = 2. We generalize the approach to the $\ell_{p}$ Lewis weight sampling for $p \geq 1$ and extend it to $k \geq 1$ , giving the following theorem.

Theorem 1.1 (Informal version of Corollary 3.7). Let $w_{1}(A^{j}), \ldots, w_{n}(A^{j})$ denote the Lewis weights of $A^{j}$ and $T = \sum_{i=1}^{n} \max_{j \in [k]} w_{i}(A^{j})$ . Suppose that $T = \text{poly}(d)$ . There exists a randomized algorithm which samples

$$
m \lesssim \epsilon^ {- 4} T d ^ {\max \{\frac {p}{2} - 1, 0 \}} \log^ {2} d \log (d / \epsilon)
$$

unlabeled instances and outputs solutions $\tilde{\pmb{\theta}}^1, \ldots, \tilde{\pmb{\theta}}^k \in \mathbb{R}^d$ such that (1) holds for $p \geq 1$ and all $j \in [k]$ with probability at least 0.9.

Note that for a single matrix $A \in R^{n \times d}$ , the sum $T = \sum_{i} w_{i}(A) = d$ and so Theorem 1.1 implies a sample complexity of $\tilde{O}(d^{\max\{p/2,1\}}/\epsilon^{4})$ , recovering the result in [17] for $p = 2 \binom{1}{1}$ .

Empirical Findings. Extensive experiments are conducted on 11 classification and regression benchmarks with 50 distinct deep models. In Section 3.2, we empirically observe that the sum of the maximum leverage scores grows very slowly as the number of models increases. This result reveals the strong correlation among the leverage scores of different deep representations, providing a direction for interpreting deep representation learning [23, 33]. In Section 4, we validate the effectiveness of our method with fine-tuning and vanilla learning scenarios of deep models for both the $\ell_2$ -regression loss and cross-entropy loss. The results show that our method outperforms other one-shot baselines. Even when comparing with the state-of-the-art iterative AL methods for multiple models, our approach achieves competitive performance.

# 2 Related Work

Active learning has been extensively studied in the past decades $[37, 35]$ . With a limited query budget, many methods try to query the labels of the most useful instances for a target model by designing effective selection criteria, which commonly depend on two notions, informativeness and representativeness. Informativeness-based criteria prefer instances where the target model has a highly uncertain prediction $[28, 47, 22]$ , while representativeness-based criteria prefer instances which can help reduce the distribution gap between the queried instances and the entire dataset $[12, 4, 36]$ . While most existing methods focus on improving the performance of a specific target model, Tang and Huang $[42]$ extend the setting of AL to multiple target models. In this scenario, the active learner seeks to enhance the performance of every target model simultaneously by selective querying. Their work demonstrates that the query complexity of AL for multiple models can be upper bounded by that of an appropriately designed single model. Based on this insight, they propose an iterative algorithm called DIAM, which queries the labels of the instances located in the joint disagreement regions among multiple models. Although the method is effective, a significant concern is the substantial cost incurred by training multiple deep models at each iteration.

To reduce the computational cost of repetitive model training, one-shot AL algorithms have been proposed to query all useful instances in a single batch, thereby avoiding the need for model updates. Yang and Loog $[48]$ employ existing AL methods with pseudo-labeling to obtain a candidate set of diverse instances and select queries based on the feature distance between unlabeled instances and candidate instances. Viering et al. $[44]$ select representative data points by the kernelized discrepancy methods, e.g., Maximum Mean Discrepancy (MMD) $[1]$ , and give error bounds under different assumptions on data distribution. Jin et al. $[21]$ propose a one-shot AL method for deep image segmentation. Their approach uses self-supervised learning to obtain more informative representations and selects diverse instances based on clustering results and feature distances. In addition, Coreset $[36]$ and Transductive Experimental Design $[49]$ are implicit one-shot AL methods. However, all the aforementioned one-shot AL methods cannot handle the distinct representations of multiple deep models.

Although most existing AL methods rely on heuristics lacking theoretical analysis, AL with Lewis weight sampling has been well studied for active $\ell_{p}$ -regression problems $\min_{\theta}\|A\theta-y\|_{p}$ , where the matrix $A\in R^{n\times d}$ is fully accessible while the label vector $y\in R^{n}$ needs to be queried [7, 6, 34, 5, 31]. Provable guarantees are obtained for $(1+\epsilon)$ -approximate solutions, i.e., $\|A\theta'-y\|_{p}\leq(1+\epsilon)\|A\theta^{*}-y\|_{p}$ , where $\theta'$ is the output of the algorithm and $\theta^{*}$ the true minimizer. For p=1, Parulekar et al. [34] show that $O(\epsilon^{-2}d\log(d/(\epsilon\delta)))$ samples suffice. For p=2, Chen and Price [7] solve the problem optimally with $O(d/\epsilon)$ queries. For $p\in(1,2)$ , Chen and Derezinski [6] propose the first algorithm to solve the problem with sublinear query complexity, i.e., $O(\epsilon^{-2}d^{2}\log d)$ . For p>2, Musco et al. [31] show that $O(\epsilon^{-p}d^{p/2}\log^{2}d\log^{p-1}(d/\epsilon))$ queries suffice. Recently, Gajjar et al. [15] extend such sampling method to the single neuron model for p=2, which inspires our work. They establish a multiplicative constant-factor error bound of the form (1) using $O(d^{2}/\epsilon^{4})$ samples. This has been further improved to $O(d/\epsilon^{4})$ in [17].

# 3 Our Approach

# 3.1 Preliminaries

Notation. Suppose that the dataset has n instances $\alpha_{1},\ldots,\alpha_{n}$ and each $\alpha_{i}$ has a ground-truth label $y_{i}$ . The given data consist of a small labeled set $\mathcal{L}=\{(\boldsymbol{\alpha}_{i},y_{i})\}_{i=1}^{n_{l}}$ , used for model initialization, and a large unlabeled set $U=\{\alpha_{n_{l}+i}\}_{i=1}^{n_{u}}$ , used for active querying. Here, $n=n_{l}+n_{u}$ and it is assumed that $n_{l}\ll n_{u}$ . A neural network can be viewed as the composition of a network backbone and a linear prediction layer $\theta\in R^{d}$ composed by an activation function $f(\cdot)$ . The prediction of the network is given by $f(A\theta)$ , where $A\in R^{n\times d}$ is the feature matrix obtained

by feeding the dataset into the network backbone. Denote by $y \in R^{n}$ the corresponding label vector that needs to be queried. In our theoretical analysis, we assume that $d \ll n$ , A has full column rank, the network backbone is fixed during the learning of $\theta$ and f is L-Lipschitz continuous with $f(0) = 0$ .

For $p \geq 1$ , the $\ell_p$ norm of a vector $\boldsymbol{\theta}$ is defined to be $\| \boldsymbol{\theta} \|_p = (\sum_{i=1}^n |\boldsymbol{\theta}_i|^p)^{\frac{1}{p}}$ , where $\boldsymbol{\theta}_i$ is the $i$ -th coordinate of $\boldsymbol{\theta}$ .

For a matrix $A$ , the operator norm of $A$ is defined as $\| A \|_2 = \sup_{\boldsymbol{\theta} \in \mathbb{R}^d \setminus \{0\}} \| A \boldsymbol{\theta} \|_2 / \| \boldsymbol{\theta} \|_2$ . For integer $n \geq 1$ , we use $[n]$ to denote the set $\{1, 2, \ldots, n\}$ . We write $a = (1 \pm \epsilon)b$ if $(1 - \epsilon)b \leq a \leq (1 + \epsilon)b$ and $a \lesssim_{t_1, t_2, \ldots} b$ if there exists a constant $C$ depending only on $t_1, t_2, \ldots$ such that $a \leq Cb$ . We also write $a \sim_{t_1, t_2, \ldots} b$ if $a \lesssim_{t_1, t_2, \ldots} b$ and $b \lesssim_{t_1, t_2, \ldots} a$ .

Lewis Weights Sampling. We shall define the Lewis weights and state a classical result that Lewis weight sampling gives subspace embeddings, which is the starting point of our algorithm.

Definition 3.1 ( $\ell_{p}$ Lewis Weights). Let p > 0. Suppose that $A \in R^{n \times d}$ and its i-th row is $a_{i} \in R^{d}$ . The Lewis weights of A are $w_{1}, \ldots, w_{n}$ such that $w_{i} = (\boldsymbol{a}_{i}^{\top}(A^{\top}W^{1-\frac{2}{p}}A)^{-1}\boldsymbol{a}_{i})^{\frac{p}{2}}$ , where W is a diagonal matrix with diagonal elements $w_{1}, w_{2}, \ldots, w_{n}$ .

We remark that Lewis weights satisfy that $w_{i}(A) \in [0,1]$ and $\sum_{i=1}^{n} w_{i}(A) = d$ . When p = 2, Lewis weights are exactly the leverage scores. Next, we define $\ell_{p}$ subspace embedding and sampling matrix. Then, we state the result that Lewis weight sampling gives subspace embeddings.

Definition 3.2 ( $\ell_p$ Subspace Embedding). Let $p > 0$ and $\epsilon \in (0,1)$ be the distortion parameter. A matrix $S \in \mathbb{R}^{m \times n}$ is said to be an $\ell_p$ $\epsilon$ -subspace-embedding matrix for $A \in \mathbb{R}^{n \times d}$ if it holds simultaneously for all vectors $\boldsymbol{\theta} \in \mathbb{R}^d$ that $(1 - \epsilon)\|A\boldsymbol{\theta}\|_p \leq \|SA\boldsymbol{\theta}\|_p \leq (1 + \epsilon)\|A\boldsymbol{\theta}\|_p$ .

Definition 3.3 (Sampling Matrix). Let p > 0. Suppose that $p_{1}, \ldots, p_{n} \geq 0$ such that $p_{1} + p_{2} + \cdots + p_{n} = 1$ and $e_{1}, \ldots, e_{n}$ are the standard basis vectors of $R^{n}$ . A matrix $S \in R^{m \times n}$ is called a reweighted sampling matrix if the rows of S are i.i.d. copies of random vector X, where $X = (mp_{j})^{-1/p} e_{j}^{T}$ with probability $p_{j}, j = 1, \ldots, n$ . The number m of rows in S is called the sample size.

Lemma 3.4 (Constant-factor Subspace Embedding, [10, Theorem 7.1]). Given $A \in \mathbb{R}^{n \times d}$ . Suppose that $t_i \geq \beta w_i$ for all $i \in [n]$ , where

$$
\beta \gtrsim_ {p} \left\{ \begin{array}{l l} \log^ {3} d + \log \frac {1}{\delta}, & 0 <   p <   2, p \neq 1 \\ \log \frac {d}{\delta}, & p = 1, 2 \\ d ^ {\frac {p}{2} - 1} (\log d + \log \frac {1}{\delta}) & 2 <   p <   \infty \end{array} \right.
$$

is a sampling parameter. Let $m = \sum_{i=1}^{n} t_i$ . If $S \in R^{m \times n}$ is a reweighted sampling matrix with sampling probability $p_i = \frac{t_i}{m}$ for all i, then S is an $\ell_p \frac{1}{2}$ -subspace-embedding matrix for A with probability at least $1 - \delta$ .

We note that our main theorem only requires constant-factor subspace embedding property of the sampling matrix S and, therefore, we can ignore the dependence on $\epsilon$ in the bounds for $\ell_{p}$ subspace embeddings. The case of $p \leq 2$ is proved by Cohen and Peng [10] and the case of p > 2 is originally due to Bourgain et al. [2].

The following are stability results of Lewis weights, due to [10].

Lemma 3.5 (Lemmata 5.3 and 5.4 of [10]). Suppose $A \in \mathbb{R}^{n \times d}$ and $\overline{w_1}, \ldots, \overline{w_n}$ are the Lewis weights of $A$ . Let $w_1, \ldots, w_n$ be weights such that

$$
\frac {1}{\alpha} w _ {i} ^ {2 / p} \leq a _ {i} ^ {\top} \left(\sum_ {i} w _ {i} ^ {1 - 2 / p} a _ {i} a _ {i} ^ {\top}\right) ^ {- 1} a _ {i} \leq \alpha w _ {i} ^ {2 / p}, \quad \forall i = 1, \dots , n.
$$

Table 1: The specifications of the datasets used in the experiments. 

<table><tr><td>Dataset</td><td>#Training</td><td>#Testing</td><td>#Label</td><td>Task</td></tr><tr><td>MNIST [25]</td><td>60,000</td><td>10,000</td><td>10</td><td>Classification</td></tr><tr><td>Fashion-MNIST [46]</td><td>60,000</td><td>10,000</td><td>10</td><td>Classification</td></tr><tr><td>Kuzushiji-MNIST [8]</td><td>60,000</td><td>10,000</td><td>10</td><td>Classification</td></tr><tr><td>SVHN [32]</td><td>73,257</td><td>26,032</td><td>10</td><td>Classification</td></tr><tr><td>EMNIST-digits [9]</td><td>240,000</td><td>40,000</td><td>10</td><td>Classification</td></tr><tr><td>EMNIST-letters [9]</td><td>88,800</td><td>14,800</td><td>26</td><td>Classification</td></tr><tr><td>CIFAR-10 [24]</td><td>50,000</td><td>10,000</td><td>10</td><td>Classification</td></tr><tr><td>CIFAR-100 [24]</td><td>50,000</td><td>10,000</td><td>100</td><td>Classification</td></tr><tr><td>Biwi [14]</td><td>10,317</td><td>5,361</td><td>2</td><td>Regression</td></tr><tr><td>FLD [41]</td><td>13,466</td><td>249</td><td>10</td><td>Regression</td></tr><tr><td>CelebA [29]</td><td>162,770</td><td>19,962</td><td>10</td><td>Regression</td></tr></table>

Table 2: Summary of the initial performances of 50 deep models on the classification datasets. We report the mean accuracy, standard deviation of the accuracies, maximum and minimum accuracy in the table. 

<table><tr><td>Dataset</td><td>mean accuracy</td><td>std. deviation</td><td>maximum</td><td>minimum</td></tr><tr><td>MNIST</td><td>94.79</td><td>4.39</td><td>97.61</td><td>72.99</td></tr><tr><td>F.MNIST</td><td>81.76</td><td>3.03</td><td>85.09</td><td>67.32</td></tr><tr><td>K.MNIST</td><td>74.79</td><td>7.94</td><td>85.61</td><td>51.31</td></tr><tr><td>CIFAR-10</td><td>48.54</td><td>3.36</td><td>53.01</td><td>36.92</td></tr><tr><td>CIFAR-100</td><td>11.34</td><td>2.14</td><td>16.85</td><td>7.63</td></tr><tr><td>SVHN</td><td>45.53</td><td>7.62</td><td>60.98</td><td>31.88</td></tr><tr><td>EMNIST-l.</td><td>70.04</td><td>14.21</td><td>83.23</td><td>25.84</td></tr><tr><td>EMNIST-d.</td><td>95.12</td><td>3.64</td><td>97.76</td><td>76.13</td></tr></table>

Then it holds for all $i$ that

$$
\alpha^ {- c (p, d)} w _ {i} \leq \overline {{w _ {i}}} \leq \alpha^ {c (p, d)} w _ {i},
$$

where $c(p,d) = (p/2)/(1 - |p/2 - 1|)$ when $p < 4$ or $c(p,d) \sim_p \sqrt{d}$ when $p \geq 4$ .

# 3.2 An Empirical Observation

To address the challenges of distinct representations from multiple deep models, one solution is to sample the unlabeled instances by their maximum Lewis weight (i.e. $\max_{j} w_{i}(A^{j})$ ) among different representations. Recall that the sample size is proportional to $\sum_{i} \max_{j} w_{i}(A^{j})$ , this strategy will not save the number of queries if this sum is k times larger than that of a single regression problem. Therefore, we would like first to examine the sum of maximum Lewis weights across representations as it will determine the potential query savings. In the following empirical studies, we mainly consider the case of p = 2 (i.e., squared loss), where the Lewis weight becomes exactly the leverage score.

Empirical Settings. We conduct experiments on 11 datasets, the details are summarized in Table 1. For each dataset, we randomly sample 3000 instances to train the models with 20 epochs, then extract features for the whole dataset and calculate their leverage scores. We use the squared loss in model training and keep all other settings the same as the OFA project. We employ 50 distinct network architectures as the target models. These architectures are published by a recent NAS method OFA [3] for accommodating diverse resource-constraint devices, ranging

![](images/eaf3c9e384c6b64902191c7c1b5c7658afb8e3a6c9454a2e4569727892bb0018.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 0           | 0           |
| 10               | 20000       | 0           |
| 20               | 40000       | 0           |
| 30               | 60000       | 0           |
| 40               | 60000       | 0           |
| 50               | 60000       | 0           |
</details>

(a) MNIST

![](images/acce4c9c8913c25de758a48d66b60ecc2a12e0df765566158d298678204cddcc.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 0           | 0           |
| 10               | 20000       | 0           |
| 20               | 40000       | 0           |
| 30               | 60000       | 0           |
| 40               | 60000       | 0           |
| 50               | 60000       | 0           |
</details>

(b) Kuzushiji-MNIST

![](images/95180d6aa70e4cf3b857f6e18ebff1e5afb358550bf7d38b4dcb4d6c264ef455.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 0           | 0           |
| 10               | 20000       | 1000        |
| 20               | 40000       | 1500        |
| 30               | 60000       | 2000        |
| 40               | 80000       | 2500        |
| 50               | 100000      | 3000        |
| 60               | 120000      | 3500        |
| 70               | 140000      | 4000        |
| 80               | 160000      | 4500        |
| 90               | 180000      | 5000        |
| 100              | 200000      | 5500        |
| 110              | 220000      | 6000        |
| 120              | 240000      | 6500        |
| 130              | 260000      | 7000        |
| 140              | 280000      | 7500        |
| 150              | 300000      | 8000        |
| 160              | 320000      | 8500        |
| 170              | 340000      | 9000        |
| 180              | 360000      | 9500        |
| 190              | 380000      | 10000       |
| 200              | 400000      | 11567       |
| 210              | 420000      | 13134       |
| 220              | 440000      | 14712       |
| 230              | 460000      | 16391       |
| 240              | 480000      | 18179       |
| 250              | 500000      | 21168       |
| 260              | 520000      | 24157       |
| 270              | 540000      | 27166       |
| 280              | 560000      | 31175       |
| 290              | 580000      | 35184       |
| 300              | 600000      | 39193       |
| 310              | 620000      | 43192       |
| 320              | 640000      | 47191       |
| 330              | 660000      | 51199       |
| 340              | 680000      | 55198       |
| 350              | 712589      | 61497       |
| 361              | 745379      | 67996       |
| 371              | 779279      | 74595       |
| 381              | 814179      | 81494       |
| 391              | 849179      | 88493       |
| 401              | 884179      | 95492       |
| 411              | 92       | nan         |
| 421              | nan         | nan         |
| 431              | nan         | nan         |
| 441              | nan         | nan         |
| 451              | nan         | nan         |
| 461              | nan         | nan         |
| 471              | nan         | nan         |
| 481              | nan         | nan         |
| 491              | nan         | nan         |
| 501              | nan         | nan         |
| Note: The actual values for the 'exact value' column are not provided in the code. The 'upper bound' and 'exact value' columns are estimated based on the y-axis label 'sum of maximum leverage scores'.
</details>

(c) Fashion-MNIST

![](images/0bc6589682d93c57c94014eee7918bcb3e0a473cf576210da0bb7e1a17d71546.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 0           | 0           |
| 10               | 20000       | 0           |
| 20               | 40000       | 0           |
| 30               | 60000       | 0           |
| 40               | 80000       | 0           |
| 50               | 100000      | 0           |
</details>

(d) EMNIST-letters

![](images/f7ba52b3a2036a411fbcae159e387557f35108dc1eb6364fd791dec49b6d08b8.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 0           | 0           |
| 10               | 20000       | 0           |
| 20               | 40000       | 0           |
| 30               | 60000       | 0           |
| 40               | 80000       | 0           |
| 50               | 100000      | 0           |
</details>

(e) EMNIST-digits

![](images/b71f08f3ad0415be1db5290c13d3d9237d82caf91406ee3cb900aecf638b4f42.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 0           | 0           |
| 20               | 30000       | 5000        |
| 40               | 60000       | 7500        |
| 60               | 90000       | 10000       |
</details>

(f) SVHN

![](images/1753276a6c46e1f3ea1c3b0e0495d3e3970a7f6b778374fa7f566784077eda8c.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 0           | 0           |
| 10               | 20000       | 3000        |
| 20               | 35000       | 4000        |
| 30               | 45000       | 5000        |
| 40               | 48000       | 6000        |
| 50               | 48000       | 6000        |
</details>

(g) CIFAR-10

![](images/6e1e62dd66d40d22bf1f767bc2db23341ded71948454a7fa3b9cfab5ba63a48e.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 0           | 0           |
| 10               | 20000       | 2000        |
| 20               | 30000       | 4000        |
| 30               | 40000       | 6000        |
| 40               | 45000       | 8000        |
| 50               | 45000       | 8000        |
</details>

(h) CIFAR-100

![](images/f37701d37359137f782f1259441fbe08dd2d88a1ed091493d8503ffeaa05a1cb.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 2500        | 2500        |
| 5                | 7500        | 3500        |
| 10               | 10000       | 4500        |
| 15               | 10000       | 5000        |
| 20               | 10000       | 5500        |
| 25               | 10000       | 5800        |
| 30               | 10000       | 6000        |
| 35               | 10000       | 6200        |
| 40               | 10000       | 6300        |
| 45               | 10000       | 6400        |
| 50               | 10000       | 6500        |
</details>

(i) Biwi

![](images/e91aaa7762483a788f4b8909e909b8df47d6e4f1618267cc9811b197b124bdd0.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 3000        | 2500        |
| 5                | 8000        | 4000        |
| 10               | 12000       | 5000        |
| 15               | 12000       | 5500        |
| 20               | 12000       | 6000        |
| 25               | 12000       | 6200        |
| 30               | 12000       | 6300        |
| 35               | 12000       | 6400        |
| 40               | 12000       | 6500        |
| 45               | 12000       | 6600        |
| 50               | 12000       | 6700        |
</details>

(j) FLD

![](images/d8bb1087c866b9ccc45ddf0e347459d71c729aed9e1b65b25bf982e39fefdb3c.jpg)

<details>
<summary>line</summary>

| number of models | upper bound | exact value |
| ---------------- | ----------- | ----------- |
| 0                | 0           | 0           |
| 20               | 30000       | 10000       |
| 40               | 60000       | 12000       |
| 50               | 70000       | 13000       |
</details>

(k) CelebA   
Figure 1: The trends of the sum of the maximum Lewis weights with p = 2 among multiple representations as the number of deep models increases.

from NVIDIA Tesla V100 GPU to mobile devices. To demonstrate the distinctness of these models, we report the initial performances of the models on the classification datasets in Table 2. It can be observed that the model performances are significantly diverse. It aligns well with our problem setting. Figure 1 shows the theoretical upper bound and the exact values of the sum of the maximum leverage score of each instance across different representations.

Results. We first plot the upper bound of the sum of the maximum leverage scores across multiple representations as the number of models increases. For matrices $A^{1},\ldots,A^{k}$ of n rows, it clearly holds that $\sum_{i}\max_{j}w_{i}(A^{j})\leq\min\{\sum_{i}\sum_{j}w_{i}(A^{j}),\sum_{i}1\}\leq\min\{\sum_{i}\operatorname{rank}(A^{j}),n\}$ . This upper bound is plotted in red color, which grows almost linearly until it reaches the number of instances n.

We examine the exact values of the sum of maximum leverage scores across multiple representations. All figures show that the exact sum grows very slowly as the number of models increases. This suggests highly consistent discrimination power of most instances across different representations, as the leverage score when p = 2 is exactly the leverage score, which measures how hard an instance can be linearly represented by others. Therefore, a small number of discriminating examples suffices to effectively train multiple models. leverage scores also provide a possible direction to interpret the behavior of deep representation learning, as prior works have not discovered any simple form of correlation among the diverse representations obtained by different model architectures $[23, 33]$ .

Algorithm 1 The Proposed Algorithm   
Input: Feature matrices of labeled and unlabeled instances $L^{j}, U^{j} (j = 1, \ldots, k)$ , query budget $\tau$ , error parameter $\epsilon$ Output: Trained linear models $\tilde{\theta}^{1}, \ldots, \tilde{\theta}^{k}$ .

Initialize: $p, \bar{y} \leftarrow zero vector of length n_{u}; Q \leftarrow an empty list; m \leftarrow 0$ 1: $p_{i} \leftarrow \max_{1 \leq j \leq k} w_{i}(U^{j})$ for $i = 1, \ldots, n_{u}$ 2: $p_{i} \leftarrow p_{i} / \|p\|_{1}$ for $i = 1, \ldots, n_{u}$ 3: while Q has fewer than $\tau$ distinct elements do

4: q $\leftarrow sample a number from [n_{u}]$ with replacement with probability $p_{1}, \ldots, p_{n_{u}}$ 5: $m \leftarrow m + 1$ 6: append q to Q

7: if the label of q-th unlabeled instance is unknown then

8: $\bar{y}_{q} \leftarrow query the label of q-th unlabeled instance$ 9: $S \leftarrow zero matrix with shape (n_{l} + m) \times (n_{l} + n_{u})$ 10: $S_{i,i} \leftarrow 1$ for $i = 1, \ldots, n_{l}$ 11: $S_{i+n_{l}, Q_{i}+n_{l}} \leftarrow (m \cdot p_{Q_{i}})^{-1/p}$ for $i = 1, \ldots, m$ 12: $y \leftarrow [y_{1}, \ldots, y_{n_{l}}, \bar{y}]^{T}$ 13: for j = 1, ..., k do

14: $A^{j} \leftarrow \begin{bmatrix} L^{j} \\ U^{j} \end{bmatrix}$ 15: $\tilde{\theta}^{j} \leftarrow \arg\min_{x \in E} \|Sf(A^{j}\theta) - S\mathbf{y}\|_{p}^{p}$ , where $E = \{\theta : \|SA^{j}\theta\|_{p}^{p} \leq \frac{1}{\epsilon L^{p}} \|S\mathbf{y}\|_{p}^{p}\}$ 16: return $\tilde{\theta}^{1}, \ldots, \tilde{\theta}^{k}$

# 3.3 The Algorithm

Based on our empirical observations, we propose to sample and reweight unlabeled instances based on their maximum Lewis weights across multiple representations. Specifically, given the feature matrices of the labeled and unlabeled instances (denoted by $\{L^{j}\}_{j=1}^{k}$ and $\{U^{j}\}_{j=1}^{k}$ , respectively), our algorithm begins with calculating the Lewis weights of the unlabeled instances based on each of their feature representations. Next, a normalized maximum Lewis weight among multiple representations for each unlabeled instance is obtained:

$$
p _ {i} = \frac {\max _ {j \in [ k ]} w _ {i} (U ^ {j})}{\sum_ {i = 1} ^ {n _ {u}} \max _ {j \in [ k ]} w _ {i} (U ^ {j})}, \quad i = 1, \ldots , n _ {u}.
$$

In the querying phase, we conduct i.i.d. sampling with replacement on the unlabeled set using a probability distribution p. The sampling process is repeated until $\tau$ distinct unlabeled instances are sampled. Let Q denote the set of indices of unlabeled instances that are selected for label query. We reweight each of the instance with index $q \in Q$ by $(m \cdot p_{q})^{-1/p}$ . Finally, both the initially labeled instances with weight 1 and the reweighted queried instances will be used to update each of the target model. Note that, although Q may contain repeated entries, each instance will be queried only once and reoccurrences will not incur additional query cost. We present our algorithm in Algorithm 1.

# 3.4 Theoretical Guarantees

Our main result is as follows, which can be seen as the guarantee for a single model.

Theorem 3.6. Let $p \geq 1$ , $f(\boldsymbol{\theta})$ be an L-Lipschitz function with $f(0) = 0$ , $A \in \mathbb{R}^{n \times d}$ be the data matrix and $\boldsymbol{y} \in \mathbb{R}^n$ be the target vector. Consider a reweighted sampling matrix $S$ with row sampling probability $p_i = \frac{t_i}{m}$ , where $t_1, \ldots, t_n$ are some quantities and $m = \sum_i t_i$ .

Suppose that $t_1, \ldots, t_n \in \mathbb{R}$ satisfy that $t_i \geq \beta w_i(A)$ , where

$$
\beta \gtrsim_ {p} \epsilon^ {- 4} d ^ {\max \left\{\frac {p}{2} - 1, 0 \right\}} \log^ {2} d \cdot \log \left(\sum_ {i = 1} ^ {n} t _ {i}\right). \tag {2}
$$

Then, if $S$ is a reweighted sampling matrix as described above and $\tilde{\pmb{\theta}} = \arg \min_{\pmb{\theta} \in E} \| Sf(Ax) - Sy \|_p$ , where $E = \{\pmb{\theta} : \| SA\pmb{\theta}\|_p^p \leq \| Sy\|_p^p / (\epsilon L^p)\}$ , it holds with probability at least 0.9 that

$$
\| f (A \tilde {\pmb {\theta}}) - \pmb {y} \| _ {p} ^ {p} \leq C \left(\| f (A \pmb {\theta} ^ {*}) - \pmb {y} \| _ {p} ^ {p} + \epsilon L ^ {p} \| A \pmb {\theta} ^ {*} \| _ {p} ^ {p}\right),
$$

where $\pmb{\theta}^{*} = \arg \min_{\pmb{\theta}}\| f(A\pmb {\theta}) - \pmb{y}\|_{p}$ and $C > 0$ is a constant depending only on $p$ .

The proof of Theorem 3.6 is deferred to the next section (Section 3.5). Our analysis also suggests that an $\ell_p$ -subspace-embedding can be obtained using $\tilde{O}(d / \epsilon^2)$ samples, removing the log $n$ factor in [45], which may be of independent interest. See Appendix A for discussions. Below we show the guarantee for multiple models, which follows easily as a corollary of Theorem 3.6.

Corollary 3.7. Let $A_1, \ldots, A_k \in \mathbb{R}^{n \times d}$ be data matrices and $T = \sum_{i=1}^{n} \max_{j \in [k]} w_i(A^j)$ . Let $f(\boldsymbol{\theta})$ be an L-Lipschitz function with $f(0) = 0$ and $\boldsymbol{y} \in \mathbb{R}^n$ be the target vector. There exists an algorithm that makes

$$
m \sim_ {p} \epsilon^ {- 4} T d ^ {\max \left\{\frac {p}{2} - 1, 0 \right\}} \log^ {2} d \log (d T / \epsilon) \tag {3}
$$

queries and outputs solutions $\tilde{\theta}^{1},\ldots,\tilde{\theta}^{k}\in R^{d}$ such that (1) holds for all $j\in[k]$ with probability at least 0.9.

Proof. Let $t_{i} = \beta \cdot \max_{j} w_{i}(A^{j})$ , then for any fixed j, it holds that $t_{i} \geq \beta w_{i}(A^{j})$ . Also, $m = \sum_{i} t_{i} = \beta T$ . The sampling probability $p_{i} = t_{i}/m = \max_{j} w_{i}(A^{j})/T$ , which is exactly our sampling scheme in Algorithm 1. Take

$$
\beta \sim \epsilon^ {- 4} d ^ {\max \{\frac {p}{2} - 1, 0 \}} \log^ {2} d \log (d T / \epsilon),
$$

then $\beta$ satisfies the condition (2) in Theorem 3.6, whence the conclusion follows.

![](images/eca0c9cb1482a4311a3f4c0a8e01ee80ef3dd1ba1e90d8c2fd1a2570ced965d3.jpg)

Remark. The proof of Corollary 3.7 implies the same guarantee for Algorithm 1 if $\tau$ is set to be the quantity for m in (3). Indeed, the proof of Corollary 3.7 shows that the guarantee holds as soon as the variable m in Algorithm 1 reaches the desired amount in (3), which allows double counting of identical sampled rows; setting $\tau$ to be the same value will only result in a larger number m of samples and the guarantee will persist.

# 3.5 Proof of Theorem 3.6

We first need a simple inequality.

Fact 3.8. Suppose that a, b > 0 and p > 0. It holds that $(a + b)^{p} \leq 2^{|p-1|}(a^{p} + b^{p})$ .

Let $\mathrm{OPT} = \min_{\boldsymbol{\theta}} \| A\boldsymbol{\theta} - \boldsymbol{y}\| _p$ . Theorem 3.6 is proved by the following chain of inequalities.

$$
\begin{array}{l} \left\| f (A \tilde {\boldsymbol {\theta}}) - \boldsymbol {y} \right\| _ {p} ^ {p} \stackrel {{(\mathrm{A})}} {{\leq}} 2 ^ {| p - 1 |} (\left\| f (A \tilde {\boldsymbol {\theta}}) - f (A \boldsymbol {\theta} ^ {*}) \right\| _ {p} ^ {p} + \mathrm{OPT} ^ {p}) \\ \stackrel {\mathrm{(B)}} {\leq} 2 ^ {| p - 1 |} (\left\| S f (A \tilde {\boldsymbol {\theta}}) - S f (A \boldsymbol {\theta} ^ {*}) \right\| _ {p} ^ {p} + \epsilon^ {2} L ^ {p} R ^ {p} + \mathrm{OPT} ^ {p}) \\ \stackrel {\mathrm{(C)}} {\leq} 2 ^ {| p - 1 |} (2 ^ {| p - 1 |} \left\| S f (A \tilde {\boldsymbol {\theta}}) - S \boldsymbol {y} \right\| _ {p} ^ {p} + C _ {1} \mathrm{OPT} ^ {p} + \epsilon^ {2} L ^ {p} R ^ {p}) \\ \stackrel {\mathrm{(D)}} {\leq} 2 ^ {| p - 1 |} \left[ C _ {2} \left(\mathrm{OPT} ^ {p} + \epsilon L ^ {p} \| A \boldsymbol {\theta} ^ {*} \| _ {p} ^ {p}\right) + C _ {1} \mathrm{OPT} ^ {p} + \epsilon^ {2} L ^ {p} R ^ {p} \right] \\ \end{array}
$$

$$
\stackrel {\mathrm{(E)}} {\leq} C (\mathrm{OPT} ^ {p} + \epsilon L ^ {p} \| A \boldsymbol {\theta} ^ {*} \| _ {p} ^ {p})
$$

where inequalities (A) and (C) use Fact 3.8, inequality (D) uses [15, Claim 1]. Inequality (E) follows from that

$$
R ^ {p} := \max (\| A \tilde {\pmb {\theta}} ^ {p} \|, \| A \pmb {\theta} ^ {*} \| ^ {p}) \leq \| A \tilde {\pmb {\theta}} \| ^ {p} + \| A \pmb {\theta} ^ {*} \| ^ {p}
$$

$$
\stackrel {\text {(EA)}} {\leq} 2 \left\| S A \tilde {\boldsymbol {\theta}} \right\| _ {p} ^ {p} + \| A \boldsymbol {\theta} ^ {*} \| _ {p} ^ {p}
$$

$$
\stackrel {\mathrm{(EB)}} {\leq} 2 \frac {\| S \pmb {y} \| _ {p} ^ {p}}{\epsilon L ^ {p}} + \| A \pmb {\theta} ^ {*} \| _ {p} ^ {p}
$$

$$
\stackrel {\text {(EC)}} {\leq} 1 0 0 \frac {\| \boldsymbol {y} \| _ {p} ^ {p}}{\epsilon L ^ {p}} + \| A \boldsymbol {\theta} ^ {*} \| _ {p} ^ {p}
$$

$$
\stackrel {\text {(ED)}} {\leq} 1 0 0 \cdot 2 ^ {| p - 1 |} \frac {\| f (A \boldsymbol {\theta} ^ {*}) - y \| _ {p} ^ {p} + L ^ {p} \| A \boldsymbol {\theta} ^ {*} \| _ {p} ^ {p}}{\epsilon L ^ {p}} + \| A \boldsymbol {\theta} ^ {*} \| _ {p} ^ {p}
$$

$$
= 1 0 0 \cdot 2 ^ {| p - 1 |} \frac {\| f (A \pmb {\theta} ^ {*}) - y \| _ {p} ^ {p}}{\epsilon L ^ {p}} + \left(\frac {1 0 0 \cdot 2 ^ {| p - 1 |}}{\epsilon} + 1\right) \| A \pmb {\theta} ^ {*} \| _ {p} ^ {p},
$$

where inequality (EA) holds because $S$ is a subspace embedding matrix for $A$ , inequality (EB) is from the constraint of our approximate solution in Line 16, inequality (EC) holds with probability at least 49/50 by Markov's inequality and inequality (ED) follows from Fact 3.8.

We shall prove inequality (B) in the following lemma. We note that the following lemma is proved in [15, Lemmata 2 and 3], but their sampling complexity is $\tilde{O}(d^2/\epsilon^4)$ with an additional $d$ factor compared with ours. We improve their result by using the reduction technique and removing the $\epsilon$ -net argument.

Lemma 3.9. Suppose that $A \in \mathbb{R}^{n \times d}$ and $t_1, \ldots, t_n \in \mathbb{R}$ such that $t_i \geq \beta w_i(A)$ for all $i$ and $p \geq 1$ . Let $m = \sum_i t_i$ and $S \in \mathbb{R}^{m \times n}$ be a reweighted sampling matrix of with row sampling probabilities $p_1, \ldots, p_n$ , where $p_i = t_i / m$ . If

$$
\beta \gtrsim \frac {d ^ {\max \{\frac {p}{2} - 1 , 0 \}}}{\epsilon^ {2}} \left(\log^ {2} d \log m + \log \frac {1}{\delta}\right),
$$

then with probability at least $1 - \delta$ and fixed constant $R > 0$ , it holds for all pairs of vectors $\pmb{\theta}_1, \pmb{\theta}_2 \in \mathbb{R}^d$ with $\| A\pmb{\theta}_1\|_p \leq R$ and $\| A\pmb{\theta}_2\|_p \leq R$ that

$$
\| S f (A \boldsymbol {\theta} _ {1}) - S f (A \boldsymbol {\theta} _ {2}) \| _ {p} ^ {p} = \| f (A \boldsymbol {\theta} _ {1}) - f (A \boldsymbol {\theta} _ {2}) \| _ {p} ^ {p} \pm \epsilon L ^ {p} R ^ {p}.
$$

Proof. Let $\boldsymbol{x}=f(A\boldsymbol{\theta}_{1})-f(A\boldsymbol{\theta}_{2})$ and $y=A\theta_{1}-A\theta_{2}$ . Denote T to be the set $\mathcal{B}(R)\times\mathcal{B}(R)=\{(\boldsymbol{\theta}_{1},\boldsymbol{\theta}_{2}):\|SA\boldsymbol{\theta}_{1}\|_{p}\leq R,\|SA\boldsymbol{\theta}_{2}\|_{p}\leq R\}$ . We shall try to upper bound

$$
\mathbb {E} _ {S} \left(\max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \| S \boldsymbol {x} \| _ {p} ^ {p} - \| \boldsymbol {x} \| _ {p} ^ {p} \right|\right) ^ {\ell}
$$

for $\ell = \log (1 / \delta)$ .

Since taking the $\ell$ -th moment of the maximum is a convex function and $\mathbb{E} \| S\boldsymbol{x}\| _p^p = \| \boldsymbol {x}\| _p^p$ , the symmetrization trick yields that

$$
\underset {S} {\mathbb {E}} \left(\max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \| S \boldsymbol {x} \| _ {p} ^ {p} - \| \boldsymbol {x} \| _ {p} ^ {p} \right|\right) ^ {\ell} \leq 2 ^ {\ell} \underset {S, \sigma} {\mathbb {E}} \left(\max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k = 1} ^ {m} \sigma_ {k} \frac {| x _ {i _ {k}} | ^ {p}}{m p _ {i _ {k}}} \right|\right) ^ {\ell},
$$

where $\sigma_{k}$ 's are Rademacher variables. It follows from Lemma 3.4 that $S$ is a $\frac{1}{2}$ -subspace embedding matrix of $A$ with probability at least $1 - \delta / 2$ . Furthermore, by Lemma 3.11, with

probability at least $1 - \delta/2$ , the Lewis weights of SA are upper bounded by $\frac{1}{\beta}$ . Let E denote the event on S that the above two conditions hold. Then $\Pr(\mathcal{E}) \geq 1 - \delta$ . We assume the following proof is conditioned on E.

Next, we prove the conditional expectation over S and $\sigma$ when conditioned on E satisfies that

$$
\underset {S, \sigma} {\mathbb {E}} \left[ \left. \left(\max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k = 1} ^ {m} \sigma_ {k} \frac {| x _ {i _ {k}} | ^ {p}}{m p _ {i _ {k}}} \right|\right) ^ {\ell} \right\rvert \mathcal {E} \right] \leq \left(\frac {\epsilon}{2} L ^ {p} R ^ {p}\right) ^ {\ell} \delta . \tag {4}
$$

Once (4) is established, it would follow Markov's inequality that

$$
\begin{array}{l} \operatorname * {P r} \left\{\max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \| S \boldsymbol {x} \| _ {p} ^ {p} - \| \boldsymbol {x} \| _ {p} ^ {p} \right| \geq \epsilon L ^ {p} R ^ {p} \Bigg |   \mathcal {E} \right\} \\ \leq \frac {\mathbb {E} _ {S , \sigma} [ (\max _ {(\boldsymbol {\theta} _ {1} , \boldsymbol {\theta} _ {2}) \in T} | \| S \boldsymbol {x} \| _ {p} ^ {p} - \| \boldsymbol {x} \| _ {p} ^ {p} |) ^ {\ell} | \mathcal {E} ]}{(\epsilon L ^ {p} R ^ {p}) ^ {\ell}} \\ \leq 2 ^ {\ell} \frac {\mathbb {E} _ {S , \sigma} \left[ \left(\max _ {(\boldsymbol {\theta} _ {1} , \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k = 1} ^ {m} \sigma_ {k} \frac {| x _ {i _ {k}} | ^ {p}}{m p _ {i _ {k}}} \right|\right) ^ {\ell} \bigg |   \mathcal {E} \right]}{(\epsilon L ^ {p} R ^ {p}) ^ {\ell}} \\ \leq 2 ^ {\ell} \frac {(\frac {\epsilon}{2} L ^ {p} R ^ {p}) ^ {\ell} \delta}{(\epsilon L ^ {p} R ^ {p}) ^ {\ell}} \quad (\text { by } (4)) \\ = \delta . \\ \end{array}
$$

and then a union bound that

$$
\operatorname * {P r} \left\{\max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \| S \boldsymbol {x} \| _ {p} ^ {p} - \| \boldsymbol {x} \| _ {p} ^ {p} \right| \geq \epsilon L ^ {p} R ^ {p} \bigg |   \mathcal {E} \right\} <   2 \delta ,
$$

which would complete the proof after rescaling $\delta$ to $\delta/2$ .

Now we focus on the proof of (4), which mostly follows the same approach of Theorem 15.13 in [26]. Let

$$
\boldsymbol {u} _ {k} = \frac {f (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {1}) - f (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {2})}{(m p _ {i _ {k}}) ^ {1 / p}}, \quad \boldsymbol {v} _ {k} = \frac {\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {1} - \boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {2}}{(m p _ {i _ {k}}) ^ {1 / p}}, \quad k \in [ m ].
$$

Then $\pmb{u} = S\pmb{x}$ and $\pmb{x} = S\pmb{y}$ . We also denote

$$
\Lambda = \max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k = 1} ^ {m} \sigma_ {k} | \boldsymbol {u} _ {k} | ^ {p} \right|,
$$

so (4) can be rewritten as

$$
\underset {S, \sigma} {\mathbb {E}} \left[ \Lambda^ {\ell} \Big | \mathcal {E} \right] \leq \left(\frac {\epsilon}{2} L ^ {p} R ^ {p}\right) ^ {\ell} \delta .
$$

We shall split the sum in $\Lambda$ into two parts: large Lewis weights and small Lewis weights. Specifically, we define $\lambda_{k} = w_{k}(SA) / d$ to be the reweighted Lewis weight of $SA$ and $J = \{k\in [m]:\lambda_k\geq 1 / m^2\}$ .

First consider those coordinates not in J (small Lewis weights).

$$
\max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k \notin J} \sigma_ {k} | \boldsymbol {u} _ {k} | ^ {p} \right| \leq \sum_ {k \notin J} | \boldsymbol {u} _ {k} | ^ {p} \leq L ^ {p} \sum_ {k \notin J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {v} _ {k} \right| ^ {p} \leq \frac {2 ^ {p}}{m} d ^ {\max (1, \frac {p}{2})} L ^ {p} R ^ {p},
$$

where the last inequality follows from the fact (see [26, Lemma 15.17]) that

$$
\max _ {k \in [ m ]} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {v} _ {k} \right| \leq d ^ {\max (\frac {1}{p}, \frac {1}{2})} \| \boldsymbol {v} \| _ {p} \tag {5}
$$

and (by the definition of $\mathcal{B}(R)$ ) that $\| \pmb{v}\| _p\leq 2R$ .

Next we consider the coordinates in $J$ (large Lewis weights). We have

$$
\begin{array}{l} \max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k \in J} \sigma_ {k} | \boldsymbol {u} _ {k} | ^ {p} \right| = \max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k \in J} \lambda_ {k} \sigma_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p} \right| \\ \leq \sqrt {\frac {1}{d \beta}} \max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k \in J} \sqrt {\lambda_ {k}} \sigma_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p} \right|, \\ \end{array}
$$

where the second line follows from the fact that reweighted Lewis weights of SA are upper bounded by $\frac{1}{d\beta}$ . By the triangle inequality, we have

$$
\begin{array}{l} \mathbb {E} _ {\sigma} [ \Lambda^ {\ell} | \mathcal {E} ] \leq \left(\frac {2 ^ {p}}{m} d ^ {\max (1, \frac {p}{2})} L ^ {p} R ^ {p}\right) ^ {\ell} + (\frac {1}{d \beta}) ^ {\frac {\ell}{2}} \mathbb {E} _ {\sigma} \left[ \max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k \in J} \sqrt {\lambda_ {k}} \sigma_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p} \right| ^ {\ell} \right| \mathcal {E} \Bigg ] \\ =: \left(\frac {2 ^ {p}}{m} d ^ {\max (1, \frac {p}{2})} L ^ {p} R ^ {p}\right) ^ {\ell} + (\frac {1}{d \beta}) ^ {\frac {\ell}{2}} \mathbb {E} _ {\sigma} [ \Xi^ {\ell} | \mathcal {E} ], \\ \end{array}
$$

where

$$
\Xi = \max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T} \left| \sum_ {k \in J} \sqrt {\lambda_ {k}} \sigma_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p} \right|.
$$

To bound $\mathbb{E}_{\sigma}[\Xi^{\ell}|\mathcal{E}]$ , we introduce the associated distance $\delta((\boldsymbol{\theta}_1,\boldsymbol{\theta}_2),(\boldsymbol{\theta}_1',\boldsymbol{\theta}_2'))$ so that it is enough to bound it by the estimated entropy of $\mathcal{B}(R)$ . We define the distance to be

$$
\begin{array}{l} \delta^ {2} ((\pmb {\theta} _ {1}, \pmb {\theta} _ {2}), (\pmb {\theta} _ {1} ^ {\prime}, \pmb {\theta} _ {2} ^ {\prime})) \\ = \sum_ {k \in J} \lambda_ {k} \left(\frac {\left| \lambda_ {k} ^ {- \frac {1}{p}} [ f (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {1}) - f (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {2}) ] \right| ^ {p}}{m p _ {i _ {k}}} - \frac {\left| \lambda_ {k} ^ {- \frac {1}{p}} [ f (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {1} ^ {\prime}) - f (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {2} ^ {\prime}) ] \right| ^ {p}}{m p _ {i _ {k}}}\right) ^ {2} \tag {6} \\ =: \sum_ {k \in J} \lambda_ {k} \left(\left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p} - \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {p}\right) ^ {2} \\ \end{array}
$$

and the norm

$$
\| \theta \| _ {J} := \max _ {k \in J} \frac {\left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {a} _ {i _ {k}} ^ {\top} \theta \right|}{\left(m p _ {i _ {k}}\right) ^ {\frac {1}{p}}}. \tag {7}
$$

By the tail bound of Dudley's integral (see e.g. [43, Theorem 8.1.6]), it holds that

$$
\operatorname * {P r} \left\{\Xi \gtrsim \int_ {0} ^ {\infty} (\log N (T, \delta , \epsilon)) ^ {\frac {1}{2}} d \epsilon + z \cdot \mathrm{diam} (T)   \bigg |   \mathcal {E} \right\} \leq \exp (- z ^ {2}).
$$

According to Lemma 3.10, it holds that

$$
\int_ {0} ^ {\infty} (\log N (T, \delta , \epsilon)) ^ {\frac {1}{2}} d \epsilon \lesssim d ^ {\max (\frac {p - 2}{4}, 0)} L ^ {p} R ^ {p - 1} \int_ {0} ^ {\infty} (\log N (\mathcal {B} (R), B _ {J}, \epsilon)) ^ {\frac {1}{2}} d \epsilon .
$$

For $p \geq 2$ , the entropy estimate in [26, Proposition 15.18] gives that

$$
\begin{array}{l} d ^ {\frac {p - 2}{4}} L ^ {p} R ^ {p - 1} \int_ {0} ^ {\infty} (\log N (\mathcal {B} (R), B _ {J}, \epsilon)) ^ {\frac {1}{2}} d \epsilon \\ = d ^ {\frac {p - 2}{4}} L ^ {p} R ^ {p - 1} \int_ {0} ^ {\infty} (\log N (\mathcal {B} (1), B _ {J}, \frac {\epsilon}{R})) ^ {\frac {1}{2}} d \epsilon \\ \end{array}
$$

$$
\lesssim d ^ {\frac {p - 2}{4}} L ^ {p} R ^ {p - 1} \left(\int_ {0} ^ {1} \left(d \log \left(1 + \frac {R \sqrt {d}}{\epsilon}\right)\right) ^ {\frac {1}{2}} d \epsilon + \int_ {1} ^ {2 \sqrt {d}} \left(\frac {R ^ {2}}{\epsilon^ {2}} d \log m\right) ^ {\frac {1}{2}} d \epsilon\right)
$$

$$
\lesssim d ^ {\frac {p}{4}} L ^ {p} R ^ {p} \log d \sqrt {\log m}.
$$

For $1 < p \leq 2$ , it follows from the entropy estimate in [26, Proposition 15.19] and a similar argument to that for $p \geq 2$ that

$$
\int_ {0} ^ {\infty} \left(\log N (T, \delta , \epsilon)\right) ^ {\frac {1}{2}} d \epsilon \lesssim d ^ {\frac {1}{2}} L ^ {p} R ^ {p} \log d \sqrt {\log m}.
$$

By the property of subgaussian variables (see e.g. [5, Proposition 4.12]), we have

$$
\underset {\sigma} {\mathbb {E}} [ \Xi^ {\ell} | \mathcal {E} ] \leq K ^ {\ell} (\sqrt {\ell} d ^ {\max \{\frac {p}{4}, \frac {1}{2} \}} L ^ {p} R ^ {p} + d ^ {\max (\frac {p}{4}, \frac {1}{2})} L ^ {p} R ^ {p} \log d \sqrt {\log m}) ^ {\ell}.
$$

Hence, given $\ell = \log (1 / \delta)$ , as long as $\beta \geq 2^{p + 1}e\cdot \epsilon^{-2}K^2 d^{\max (\frac{p}{2} -1,0)}(\log (1 / \delta) + \log^2 d\log m)$ , it follows that

$$
\begin{array}{l} \mathbb {E} _ {\sigma} [ \Lambda^ {\ell} | \mathcal {E} ] \leq \left(\frac {2 ^ {p}}{m} d ^ {\max (1, \frac {p}{2})} L ^ {p} R ^ {p}\right) ^ {\ell} + (\frac {1}{d \beta}) ^ {\frac {\ell}{2}} \mathbb {E} _ {\sigma} [ \Xi^ {\ell} | \mathcal {E} ] \\ \leq \left(\frac {2 ^ {p}}{d \beta} d ^ {\max (1, \frac {p}{2})} L ^ {p} R ^ {p}\right) ^ {\ell} + \left(\frac {K d ^ {\max (\frac {p}{4} , \frac {1}{2})} L ^ {p} R ^ {p} (\sqrt {\ell} + \log d \sqrt {\log m})}{\sqrt {d \beta}}\right) ^ {\ell} \\ \leq \left(\frac {\epsilon^ {2} L ^ {p} R ^ {p}}{\log (1 / \delta) + \log^ {2} d \log m}\right) ^ {\ell} + (\epsilon L ^ {p} R ^ {p}) ^ {\ell} \delta \\ \leq (\epsilon L ^ {p} R ^ {p}) ^ {\ell} \delta . \\ \end{array}
$$

Therefore, taking expectation over $S$ while conditioned on $\mathcal{E}$ , we have that $\mathbb{E}_{S,\sigma}[\Lambda^{\ell}|\mathcal{E}] \leq (\epsilon L^{p}R^{p})^{\ell}\delta$ . Rescaling $\epsilon = \epsilon / 2$ completes the proof of (4), as desired.

Lemma 3.10. Let $\delta((\boldsymbol{\theta}_1, \boldsymbol{\theta}_2), (\boldsymbol{\theta}_1', \boldsymbol{\theta}_2'))$ and $\|\theta\|_J$ be as defined in (6) and (7), respectively. It holds that

$$
\delta ((\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}), (\boldsymbol {\theta} _ {1} ^ {\prime}, \boldsymbol {\theta} _ {2} ^ {\prime})) \lesssim \left\{ \begin{array}{l l} d ^ {\frac {p - 2}{4}} L ^ {p} R ^ {p - 1} (\| \boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {1} ^ {\prime} \| _ {J} + \| \boldsymbol {\theta} _ {2} - \boldsymbol {\theta} _ {2} ^ {\prime} \| _ {J}) & \quad p \geq 2, \\ L ^ {p} R ^ {\frac {p}{2}} (\| \boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {1} ^ {\prime} \| _ {J} + \| \boldsymbol {\theta} _ {2} - \boldsymbol {\theta} _ {2} ^ {\prime} \| _ {J}) ^ {\frac {p}{2}} & \quad 1 \leq p \leq 2. \end{array} \right.
$$

As a consequence, the diameter of the subspace $T$ is at most $O(d^{\max \left(\frac{p}{4},\frac{1}{2}\right)}L^{p}R^{p})$ .

Proof. For $p \geq 2$ , we have

$$
\begin{array}{l} \delta^ {2} \left(\left(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}\right), \left(\boldsymbol {\theta} _ {1} ^ {\prime}, \boldsymbol {\theta} _ {2} ^ {\prime}\right)\right) \\ \stackrel {\mathrm{(A)}} {\leq} \sum_ {k \in J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} - \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {2} (\left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p - 1} + \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {p - 1}) ^ {2} \\ \stackrel {\mathrm{(B)}} {\leq} 2 L ^ {2 p} p \sum_ {k \in J} \lambda_ {k} \left(\frac {\left| \lambda_ {k} ^ {- \frac {1}{p}} (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {1} - \boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {1} ^ {\prime}) \right| + \left| \lambda_ {k} ^ {- \frac {1}{p}} (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {2} - \boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {2} ^ {\prime}) \right|}{(m p _ {i _ {k}}) ^ {\frac {1}{p}}}\right) ^ {2} \\ \cdot \left(\left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {v} _ {k} \right| ^ {2 p - 2} + \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {v} _ {k} ^ {\prime} \right| ^ {2 p - 2}\right) \\ \end{array}
$$

$$
\stackrel {\text {(C)}} {\leq} 2 ^ {p - 1} p d ^ {\frac {p - 2}{2}} L ^ {2 p} R ^ {p - 2} \sum_ {k \in J} \lambda_ {k} \left(\frac {\left| \lambda_ {k} ^ {- \frac {1}{p}} (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {1} - \boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {1} ^ {\prime}) \right| + \left| \lambda_ {k} ^ {- \frac {1}{p}} (\boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {2} - \boldsymbol {a} _ {i _ {k}} ^ {\top} \boldsymbol {\theta} _ {2} ^ {\prime}) \right|}{(m p _ {i _ {k}}) ^ {\frac {1}{p}}}\right) ^ {2}
$$

$$
\cdot \left(\left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {v} _ {k} \right| ^ {p} + \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {v} _ {k} ^ {\prime} \right| ^ {p}\right)
$$

$$
\stackrel {\mathrm{(D)}} {\leq} 2 ^ {p - 1} p d ^ {\frac {p - 2}{2}} L ^ {2 p} R ^ {p - 2} \left(\| \boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {1} ^ {\prime} \| _ {J} + \| \boldsymbol {\theta} _ {2} - \boldsymbol {\theta} _ {2} ^ {\prime} \| _ {J}\right) ^ {2} \sum_ {k \in J} \lambda_ {k} (\left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {v} _ {k} \right| ^ {p} + \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {v} _ {k} ^ {\prime} \right| ^ {p})
$$

$$
\stackrel {\mathrm{(E)}} {\leq} 2 ^ {2 p - 1} p d ^ {\frac {p - 2}{2}} L ^ {2 p} R ^ {2 p - 2} \left(\| \boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {1} ^ {\prime} \| _ {J} + \| \boldsymbol {\theta} _ {2} - \boldsymbol {\theta} _ {2} ^ {\prime} \| _ {J}\right) ^ {2},
$$

where the inequality (A) follows from the fact that $|a|^{p}-|b|^{p}\leq p(|a|^{p-1}+|b|^{p-1})|a-b|$ , (B) follows from triangle inequality and $(a+b)^{2}\leq2(a^{2}+b^{2})$ , (C) follows from (5) and (E) is obtained by $\|v\|_{p}\leq\|SA\theta_{1}\|_{p}+\|SA\theta_{2}\|_{p}\leq2R$ .

For $1 \leq p \leq 2$ , we have

$$
\begin{array}{l} \delta^ {2} ((\pmb {\theta} _ {1}, \pmb {\theta} _ {2}), (\pmb {\theta} _ {1} ^ {\prime}, \pmb {\theta} _ {2} ^ {\prime})) \\ \leq \sum_ {k \in J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} - \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {2} \left(\left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p - 1} + \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {p - 1}\right) ^ {2} \\ \leq \max _ {k \in J} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} - \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {p} \cdot \sum_ {k \in J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} - \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {2 - p} \left(\left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {2 p - 2} + \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {2 p - 2}\right) \\ \leq L ^ {p} \left(\| \boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {1} ^ {\prime} \| _ {J} + \| \boldsymbol {\theta} _ {2} - \boldsymbol {\theta} _ {2} ^ {\prime} \| _ {J}\right) ^ {p} \left(\sum_ {k \in J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} - \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {p}\right) ^ {\frac {2 - p}{p}} \\ \cdot \left[ \left(\sum_ {k \in J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p}\right) ^ {\frac {2 p - 2}{p}} + \left(\sum_ {k \in J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {p}\right) ^ {\frac {2 p - 2}{p}} \right] \\ \leq L ^ {p} \left(\| \boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {1} ^ {\prime} \| _ {J} + \| \boldsymbol {\theta} _ {2} - \boldsymbol {\theta} _ {2} ^ {\prime} \| _ {J}\right) ^ {p} \left(\sum_ {k \in J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p} + \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {p}\right) ^ {\frac {2 - p}{p}} \\ \cdot \left[ \left(\sum_ {k \in J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} \right| ^ {p}\right) ^ {\frac {2 p - 2}{p}} + \left(\sum_ {k \in J} \lambda_ {k} \left| \lambda_ {k} ^ {- \frac {1}{p}} \boldsymbol {u} _ {k} ^ {\prime} \right| ^ {p}\right) ^ {\frac {2 p - 2}{p}} \right] \\ \leq 2 ^ {p} L ^ {2 p} R ^ {p} (\| \boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {1} ^ {\prime} \| _ {J} + \| \boldsymbol {\theta} _ {2} - \boldsymbol {\theta} _ {2} ^ {\prime} \| _ {J}) ^ {p}, \\ \end{array}
$$

where we use Hölder's inequality $\|fg\|_{1} \leq \|f\|_{\alpha}\|g\|_{\beta}$ with $\alpha = \frac{p}{2-p}$ and $\beta = \frac{p}{2p-2}$ in the third line.

For $p \geq 2$ , the diameter of $T$ is upper bounded by

$$
\max _ {(\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}) \in T, (\boldsymbol {\theta} _ {1} ^ {\prime}, \boldsymbol {\theta} _ {2} ^ {\prime}) \in T} \delta ((\boldsymbol {\theta} _ {1}, \boldsymbol {\theta} _ {2}), (\boldsymbol {\theta} _ {1} ^ {\prime}, \boldsymbol {\theta} _ {2} ^ {\prime}))
$$

$$
\leq 2 ^ {\frac {2 p - 1}{2}} p d ^ {\frac {p - 2}{4}} L ^ {p} R ^ {p - 1} \left(\| \boldsymbol {\theta} _ {1} - \boldsymbol {\theta} _ {1} ^ {\prime} \| _ {J} + \| \boldsymbol {\theta} _ {2} - \boldsymbol {\theta} _ {2} ^ {\prime} \| _ {J}\right)
$$

$$
\leq 2 ^ {\frac {2 p - 1}{2}} p d ^ {\frac {p}{4}} L ^ {p} R ^ {p},
$$

where we use the fact that $\|\theta_{1}-\theta_{1}^{\prime}\|_{J}\leq d^{\frac{1}{2}}R$ from (5). For $1\leq p\leq2$ , the diameter of T is upper bounded by $L^{p}R^{\frac{p}{2}}(\|\theta_{1}-\theta_{1}^{\prime}\|_{J}+\|\theta_{2}-\theta_{2}^{\prime}\|_{J})^{\frac{p}{2}}\leq d^{\frac{1}{2}}L^{p}R^{p}$ where we obtain $\|\theta_{1}-\theta_{1}^{\prime}\|_{J}\leq d^{\frac{1}{p}}R$ from (5).

Lemma 3.11. Let p > 0. Suppose that $A \in R^{n \times d}$ and $t_{1}, \ldots, t_{n} \in R$ such that $t_{i} \geq \beta w_{i}(A)$ for all i. Let $m = \sum_{i} t_{i}$ and $S \in R^{m \times n}$ be a reweighted sampling matrix of with row sampling probabilities $p_{1}, \ldots, p_{n}$ , where $p_{i} = \frac{t_{i}}{m}$ . If $\beta \geq \epsilon^{-2} \log(d/\delta)$ , then the $\ell_{p}$ Lewis weights of SA are upper bounded by $2/\beta$ with probability at least $1 - \delta$ .

Proof. Let $a_{i} \in R^{d \times 1}$ be the i-th row of A. Without loss of generality, suppose $A^{\top}W^{1-\frac{p}{2}}A = I_{d}$ . Hence, the Lewis weights of A are $w_{i}^{\frac{2}{p}} = \boldsymbol{a}_{i}^{\top}(A^{\top}W^{1-\frac{p}{2}}A)^{-1}\boldsymbol{a}_{i} = \boldsymbol{a}_{i}^{\top}\boldsymbol{a}_{i} = \|\boldsymbol{a}_{i}\|_{2}^{2}$ . We claim that

$$
(1 - \epsilon) I _ {d} \preceq \sum_ {k = 1} ^ {m} \frac {\pmb {a} _ {i _ {k}} \pmb {a} _ {i _ {k}} ^ {\top}}{m p _ {i _ {k}}} w _ {i _ {k}} ^ {1 - \frac {2}{p}} \preceq (1 + \epsilon) I _ {d}
$$

holds with probability at least $1 - \delta$ . Let $X_{k} = \frac{\pmb{a}_{i_k}\pmb{a}_{i_k}^\top}{p_{i_k}} w_{i_k}^{1 - \frac{2}{p}}$ and then we have $\mathbb{E}X_{k} = I_{d}$ . First, we have $\mathbb{E}X_{k} = I_{d}$ and $\| X_{k} - I_{d}\|_{2}\leq 1 + \frac{\| \pmb{a}_{i_k}\|_2^2}{w_{i_k} / d} w_{i_k}^{1 - \frac{2}{p}} = 1 + \frac{m}{\beta}$ . Besides, we have that

$$
\begin{array}{l} \left\| \mathbb {E} \left(X _ {k} - I _ {d}\right) \right\| _ {2} ^ {2} = \left\| \mathbb {E} (X _ {k} - I _ {d}) ^ {\top} (X _ {k} - I _ {d}) \right\| _ {2} \\ = \left\| \mathbb {E} X _ {k} ^ {\top} X _ {k} - I _ {d} \right\| _ {2} \\ = \left\| \frac {w _ {i _ {k}}}{p _ {i _ {k}}} \cdot \mathbb {E} \frac {\boldsymbol {a} _ {i _ {k}} \boldsymbol {a} _ {i _ {k}} ^ {\top} w _ {i _ {k}} ^ {1 - \frac {2}{p}}}{p _ {i _ {k}}} - I _ {d} \right\| _ {2} \\ = \left\| \frac {w _ {i _ {k}}}{p _ {i _ {k}}} \sum_ {i = 1} ^ {n} \boldsymbol {a} _ {i} \boldsymbol {a} _ {i} ^ {\top} w _ {i _ {k}} ^ {1 - 2 / p} + I _ {d} \right\| _ {2} \\ \leq 1 + \frac {m}{\beta}. \\ \end{array}
$$

By matrix Chernoff bound, it follows that

$$
\begin{array}{l} \operatorname * {P r} \left\{\left\| \frac {1}{m} \sum_ {k = 1} ^ {m} (X _ {k} - I _ {d}) \right\| _ {2} \geq \epsilon \right\} \leq 2 d \exp \left(\frac {- m \epsilon^ {2}}{1 + d + (1 + d) \cdot \epsilon / 3}\right) \\ \leq 2 d \exp \left(- \beta \epsilon^ {2}\right) \\ \end{array}
$$

Setting $\beta = \Theta(\frac{d}{\epsilon^{2}} \log \frac{d}{\delta})$ guarantees the failure probability to be at most $\delta$ , proving the claim. Therefore, we have that

$$
(1 - \epsilon) \left(\frac {d}{m}\right) ^ {1 - \frac {2}{p}} I _ {d} \preceq \left[ \sum_ {k = 1} ^ {m} \frac {\pmb {a} _ {i _ {k}}}{(m p _ {i _ {k}}) ^ {\frac {1}{p}}} \left(\frac {w _ {i _ {k}}}{d p _ {i _ {k}}}\right) ^ {1 - \frac {2}{p}} \frac {\pmb {a} _ {i _ {k}} ^ {\top}}{(m p _ {i _ {k}}) ^ {\frac {1}{p}}} \right] ^ {- 1} \preceq (1 + 2 \epsilon) \left(\frac {d}{m}\right) ^ {1 - \frac {2}{p}} I _ {d}
$$

holds with probability at least $1 - \delta$ . Hence, it follows that

$$
\frac {\pmb {a} _ {i} ^ {\top}}{(m p _ {i}) ^ {1 / p}} \cdot \left[ \sum_ {k = 1} ^ {m} \frac {\pmb {a} _ {i _ {k}}}{(m p _ {i _ {k}}) ^ {\frac {1}{p}}} \left(\frac {d p _ {i _ {k}}}{w _ {i _ {k}}}\right) ^ {\frac {2}{p} - 1} \frac {\pmb {a} _ {i _ {k}} ^ {\top}}{(m p _ {i _ {k}}) ^ {\frac {1}{p}}} \right] ^ {- 1} \cdot \frac {\pmb {a} _ {i}}{(m p _ {i}) ^ {1 / p}} \leq (1 + 2 \epsilon) \frac {d}{m} \left(\frac {w _ {i _ {k}}}{d p _ {i _ {k}}}\right) ^ {2 / p}.
$$

Applying [5, Lemma A.2] and setting $\epsilon = \frac{1}{2}$ gives that $w_{i}(SA)\leq 2\frac{d}{m}\frac{w_{i}}{dp_{i}}\leq \frac{2}{\beta}$ .

Lemma 3.12. Let $p \geq 1$ . Suppose that $A \in \mathbb{R}^{n \times d}$ and $w_i(A) \leq 1 / \beta$ for all $i$ and $\beta > 1$ . Let $\Lambda = \max_{x: \| Ax\|_p \leq 1} |\sum_{i=1}^n \sigma_i |(Ax)_i|^p|$ , where $\sigma_1, \ldots, \sigma_n$ are independent Rademacher variables. Then the following tail bound holds:

$$
\operatorname * {P r} \left\{\Lambda \geq \left[ C \frac {d ^ {\max \{\frac {p}{2} - 1 , 0 \}}}{\beta} \right] ^ {\frac {1}{2}} \left[ \log^ {2} d \log n + z \right] \right\} \leq 2 \exp (- z ^ {2}).
$$

Proof. The tail bound is proven by Dudley's integral tail bound

$$
\operatorname * {P r} \left\{\sup _ {t \in T} X _ {t} \gtrsim \int_ {0} ^ {\infty} \sqrt {\ln N (T , d , \epsilon)} d \epsilon + z \cdot \mathrm{diam} (T) \right\} \leq 2 \exp (- z ^ {2}),
$$

where $N(T, d, \epsilon)$ is the $\epsilon$ -covering number of $T$ and $\mathrm{diam}(T)$ is the diameter of the space $T$ . In our setting, $T$ is the subspace $\{y = Ax : x \in \mathbb{R}^d\}$ . From [26, Equation (15.17) and (15.18)], the diameter is bounded by $d^{\max(\frac{p}{4}, \frac{1}{2})}$ . By Dudley's integral, we have $\mathbb{E}_{\sigma} \Lambda \lesssim \int_0^\infty \sqrt{\ln N(T, d, \epsilon)} d\epsilon$ . The upper bound of the integral was proven in [26, Theorem 15.13], assuming that $w_i(A) \leq d/n$ . The same proof can go through when the upper bound of $w_i(A) \leq 1/\beta$ , with [26, Eq. (15.17)] replaced with

$$
\mathbb {E} \Lambda \leq \frac {3 d ^ {\max (\frac {p}{2} , 1)}}{2 n} + \left(\frac {2}{d \beta}\right) ^ {\frac {1}{2}} \Xi ,
$$

where

$$
\Xi = \underset {\sigma_ {i}} {\mathbb {E}} \sup _ {x: \| W ^ {- \frac {1}{p}} A x \| _ {p} \leq 1} \left| \sum_ {i \in J} \left(\frac {w _ {i}}{d}\right) ^ {\frac {1}{2}} \sigma_ {i} | x _ {i} | ^ {p} \right|
$$

In the proof of [26], $\lambda_{i}$ is our $\frac{w_i}{d}$ , and the factor $\frac{1}{M}$ is replaced with $\frac{1}{d\beta}$ due to the change of Lewis weights' upper bound from $\frac{n}{M}$ to $\frac{1}{\beta}$ .

The main difficulty is to upper bound $\Xi$ , which is again done by using Dudley's integral. The argument to upper bound the integral in [26, Theorem 15.13] still goes through when the upper bound of Lewis weights is changed, yielding that $\Xi \leq Cd^{\max \left(\frac{p}{4},\frac{1}{2}\right)}\log d\sqrt{\log n}$ . Combining the diameter of $T$ and the inequality for $\mathbb{E}\Lambda$ gives us the result.

# 4 Experiment

In this section, we conduct experiments to validate the effectiveness of our method $^{2}$ . Due to the space limitation, some empirical settings and experimental results are presented in the appendix.

Empirical Settings. We incorporate two learning scenarios in our experiments, i.e., fine-tuning and vanilla deep learning. The first one is a common learning scenario for big models. It first pre-trains the model on preliminary tasks. Then, the weights of the network backbone are fixed, and only the prediction heads are fine-tuned on downstream tasks. This setting aligns well with our problem formulation. The second scenario is the default learning scheme, i.e., updating all the parameters of the network with the training dataset.

We employ 50 distinct network architectures as the target models. These architectures are published by a recent NAS method OFA [3] for accommodating diverse resource-constraint devices, ranging from NVIDIA Tesla V100 GPU to mobile devices. It aligns well with our problem setting. We conduct experiments on 11 datasets, including 8 classification benchmarks: MNIST [25], Fashion-MNIST [46], Kuzushiji-MNIST [8], SVHN [32], EMNIST-letters and EMNIST-digits [9], CIFAR-10 and CIFAR-100 [24]; and 3 regression benchmarks: Biwi [14], FLD [41] and CelebA [29]. The specifications of the datasets and model configurations are deferred to the Appendix C.1. The active learning settings are outlined as follows.

\- For the scenario of vanilla deep learning, we conduct performance comparisons on the classification benchmarks. Specifically, 3000 instances are sampled uniformly from the training set to initialize the models. The other compared methods will then select 3000 unlabeled instances from the remaining data points for querying at each iteration, while our method conducts one-shot querying with budgets of 9000 and 15000 instances. The cross-entropy loss is employed in model training. In this scenario, the one-shot methods also query 3000 instances per batch for better comparison. However, these methods select batches independently.

![](images/9e0d5a59cb50eb4ad67b6fbbec7378f9bc485f64fd90efb148c7bb8db3741bb4.jpg)

<details>
<summary>line</summary>

| number of queries | DIAM  | Entropy | Random | Coreset | QBC   | Our   |
| ----------------- | ----- | ------- | ------ | ------- | ----- | ----- |
| 0                 | 95.0  | 95.0    | 95.0   | 95.0    | 95.0  | 98.0  |
| 3000              | 97.5  | 97.0    | 97.0   | 96.5    | 94.0  | 98.0  |
| 6000              | 98.0  | 97.5    | 97.5   | 97.0    | 93.5  | 98.0  |
| 9000              | 98.0  | 98.0    | 98.0   | 97.5    | 95.5  | 98.0  |
| 12000             | 98.0  | 98.0    | 98.0   | 97.5    | 97.0  | 98.0  |
| 15000             | 98.0  | 98.0    | 98.0   | 97.5    | 97.5  | 98.0  |
</details>

(a) MNIST

![](images/bd733cdbf9622048c875a8698293d116b9ac827f8eaa77ef6af78182ef438964.jpg)

<details>
<summary>line</summary>

| number of queries | DIAM  | Entropy | Random | Coreset | QBC   | Our   |
| ----------------- | ----- | ------- | ------ | ------- | ----- | ----- |
| 0                 | 75.0  | 75.0    | 75.0   | 75.0    | 75.0  | 92.0  |
| 3000              | 85.0  | 83.0    | 82.0   | 81.0    | 78.0  | 92.0  |
| 6000              | 90.0  | 88.0    | 87.0   | 86.0    | 79.0  | 92.0  |
| 9000              | 92.0  | 91.0    | 89.0   | 88.0    | 81.0  | 92.0  |
| 12000             | 93.0  | 92.0    | 90.0   | 89.0    | 82.0  | 92.0  |
| 15000             | 94.0  | 93.0    | 91.0   | 90.0    | 83.0  | 92.0  |
</details>

(b) Kuzushiji-MNIST

![](images/3969dcd46a12b4d88a3fed9fb190ffc7a32ee062cf5bb8fafb00ec5081dc8dca.jpg)

<details>
<summary>line</summary>

| number of queries | DIAM  | Entropy | Random | Coreset | QBC   | Our   |
| ----------------- | ----- | ------- | ------ | ------- | ----- | ----- |
| 0                 | 82.0  | 82.0    | 82.0   | 82.0    | 82.0  | 82.0  |
| 3000              | 87.0  | 86.0    | 86.0   | 86.0    | 81.0  | 87.0  |
| 6000              | 88.0  | 87.0    | 87.0   | 87.0    | 76.0  | 88.0  |
| 9000              | 89.0  | 88.0    | 88.0   | 88.0    | 74.0  | 89.0  |
| 12000             | 90.0  | 89.0    | 89.0   | 89.0    | 73.0  | 90.0  |
| 15000             | 91.0  | 90.0    | 90.0   | 90.0    | 72.0  | 91.0  |
</details>

(c) Fashion-MNIST

![](images/af71728cfd38048ecfbd79cb3fe5c64d40420ae67ef938d02b736b64b985f6b3.jpg)

<details>
<summary>line</summary>

| number of queries | DIAM  | Entropy | Random | Coreset | QBC   | Our   |
| ----------------- | ----- | ------- | ------ | ------- | ----- | ----- |
| 0                 | 47.0  | 47.0    | 47.0   | 47.0    | 47.0  | 47.0  |
| 3000              | 60.0  | 70.0    | 70.0   | 70.0    | 65.0  | 88.0  |
| 6000              | 72.0  | 80.0    | 80.0   | 80.0    | 75.0  | 88.0  |
| 9000              | 78.0  | 85.0    | 85.0   | 85.0    | 80.0  | 88.0  |
| 12000             | 82.0  | 87.0    | 87.0   | 87.0    | 83.0  | 88.0  |
| 15000             | 85.0  | 88.0    | 88.0   | 88.0    | 85.0  | 88.0  |
</details>

(d) SVHN

![](images/b0adc1fdc84cdafb3f234b228f65307c8b732bf1d7c11164ce1a980f6c85ae93.jpg)

<details>
<summary>line</summary>

| number of queries | DIAM  | Entropy | Random | Coreset | QBC   | Our   |
| ----------------- | ----- | ------- | ------ | ------- | ----- | ----- |
| 0                 | 48.0  | 48.0    | 48.0   | 48.0    | 48.0  | 48.0  |
| 3000              | 52.0  | 56.0    | 58.0   | 56.0    | 44.0  | 52.0  |
| 6000              | 56.0  | 62.0    | 64.0   | 62.0    | 48.0  | 62.0  |
| 9000              | 60.0  | 66.0    | 68.0   | 66.0    | 52.0  | 68.0  |
| 12000             | 64.0  | 70.0    | 72.0   | 70.0    | 56.0  | 72.0  |
| 15000             | 68.0  | 74.0    | 74.0   | 74.0    | 60.0  | 74.0  |
</details>

(e) CIFAR-10

![](images/a199eae960aea423b7688e20f5500fc4b774aca43eb6b97a7c658daaa56f9c43.jpg)

<details>
<summary>line</summary>

| number of queries | DIAM  | Entropy | Random | Coreset | OBC   | Our   |
| ----------------- | ----- | ------- | ------ | ------- | ----- | ----- |
| 0                 | 12.0  | 12.0    | 12.0   | 12.0    | 12.0  | 12.0  |
| 3000              | 18.0  | 16.0    | 18.0   | 16.0    | 14.0  | 18.0  |
| 6000              | 24.0  | 20.0    | 24.0   | 20.0    | 16.0  | 24.0  |
| 9000              | 30.0  | 24.0    | 30.0   | 24.0    | 18.0  | 32.0  |
| 12000             | 36.0  | 28.0    | 36.0   | 28.0    | 22.0  | 38.0  |
| 15000             | 42.0  | 32.0    | 42.0   | 32.0    | 26.0  | 42.0  |
</details>

(f) CIFAR-100

![](images/59e082218ebe6c383bfc8954090412c0f42b44d12db8a035a2e86c8b3297e6f3.jpg)

<details>
<summary>line</summary>

| number of queries | DIAM  | Entropy | Random | Coreset | QBC   | Our   |
| ----------------- | ----- | ------- | ------ | ------- | ----- | ----- |
| 0                 | 70.0  | 70.0    | 70.0   | 70.0    | 70.0  | 87.0  |
| 3000              | 85.0  | 85.0    | 85.0   | 85.0    | 65.0  | 87.0  |
| 6000              | 85.0  | 85.0    | 85.0   | 85.0    | 65.0  | 87.0  |
| 9000              | 85.0  | 85.0    | 85.0   | 85.0    | 75.0  | 87.0  |
| 12000             | 85.0  | 85.0    | 85.0   | 85.0    | 75.0  | 87.0  |
| 15000             | 85.0  | 85.0    | 85.0   | 85.0    | 75.0  | 87.0  |
</details>

(g) EMNIST-letters

![](images/82c699905eff1276d3cbb2e7b98e180cee4ba0a66eeff30d18c518a07bc4d536.jpg)

<details>
<summary>line</summary>

| number of queries | DIAM  | Entropy | Random | Coreset | QBC   | Our   |
| ----------------- | ----- | ------- | ------ | ------- | ----- | ----- |
| 0                 | 95.0  | 95.0    | 95.0   | 95.0    | 95.0  | 98.0  |
| 3000              | 94.0  | 97.0    | 97.0   | 96.5    | 92.5  | 98.0  |
| 6000              | 94.5  | 97.5    | 97.5   | 96.5    | 92.5  | 98.0  |
| 9000              | 97.0  | 97.5    | 97.5   | 96.5    | 92.5  | 98.0  |
| 12000             | 97.5  | 97.5    | 97.5   | 96.5    | 92.5  | 98.0  |
| 15000             | 97.5  | 97.5    | 97.5   | 96.5    | 92.5  | 98.0  |
</details>

(h) EMNIST-digits  
Figure 2: Results of Performance comparison in classification datasets. The error bars indicate the standard deviation of the performances of multiple models.

\- For the fine-tuning scenario, we use the regression datasets. Initially, 500 instances are sampled uniformly from the training set to fine-tune each network. Then, we fix the backbone parameters and actively query the labels among the remaining instances. Afterwards, 50 linear prediction layers with mean squared error (MSE) loss and ReLU activation function are trained on the updated labeled dataset, utilizing the features extracted by different network backbones. In this scenario, all the compared methods have the same query budgets of 3000 and 6000 instances.

We compare our algorithm with the following methods in the vanilla deep learning scenario.

- (iterative) DIAM [42]: The state-of-the-art iterative AL method for multiple target models, which prefers the instances in the joint disagreement regions of multiple models.   
- (iterative) Entropy [27]: This strategy selects instances with the highest prediction entropy. We follow the implementation in [42] to adapt it to multiple models. It queries the instances with the highest mean prediction entropy.   
- (iterative) QBC [38]: This strategy selects the instances that the target models have the most inconsistent predictions. The inconsistency is evaluated by KL divergence.   
- (one-shot) Coreset [36]: This strategy selects the most representative instances. We follow the implementation in [42] to adapt it to multiple models. It solves the coreset problem based on the features extracted by the supernet in OFA.

![](images/97037fadb871d2d1c96c7bc568f21b8258ec4c1bd9e44fe6e0c741d26300fc23.jpg)

<details>
<summary>bar</summary>

| methods | mean MSE |
| ------- | -------- |
| Our     | 0.058    |
| Random  | 0.065    |
| QBC     | 0.061    |
| Coreset | 0.063    |
</details>

(a) Biwi (3000 instances)

![](images/15696386d8f0b7910c8567149633c3bea89896612b3594d3de07c0c515700006.jpg)

<details>
<summary>bar</summary>

| methods   | mean MSE |
| --------- | -------- |
| Our       | 0.048    |
| Random    | 0.063    |
| QBC       | 0.069    |
| Coreset   | 0.064    |
</details>

(b) FLD (3000 instances)

![](images/f7511c0ec2dd7d2d78c48c064b24c590db59092452b53b7cada7c1a2da4b3a40.jpg)

<details>
<summary>bar</summary>

| methods   | mean MSE |
| --------- | -------- |
| Our       | 0.021    |
| Random    | 0.023    |
| QBC       | 0.025    |
| Coreset   | 0.023    |
</details>

(c) CelebA (3000 instances)

![](images/5892557fadfa711834717e32b52f0b8f251c78940854ec4ace3826ca2e5b9ceb.jpg)

<details>
<summary>bar</summary>

| methods   | mean MSE |
| --------- | -------- |
| Our       | 0.051    |
| Random    | 0.057    |
| QBC       | 0.057    |
| Coreset   | 0.054    |
</details>

(d) Biwi (6000 instances)

![](images/f2947c505e88df25bd8b0e4339a42a8243968f846a56b8fc9f51a52da3ff81d9.jpg)

<details>
<summary>bar</summary>

| methods | mean MSE |
| ------- | -------- |
| Our     | 0.044    |
| Random  | 0.051    |
| QBC     | 0.052    |
| Coreset | 0.047    |
</details>

(e) FLD (6000 instances)

![](images/b97b7fd90cf63a2ab8eab28a625467ef575c47763799bad0a5aaf9ed29dae10a.jpg)

<details>
<summary>bar</summary>

| methods | mean MSE |
| ------- | -------- |
| Our     | 0.019    |
| Random  | 0.023    |
| QBC     | 0.026    |
| Coreset | 0.023    |
</details>

(f) CelebA (6000 instances)   
Figure 3: Results of performance comparisons in regression datasets with different query budgets.

\- (one-shot) Random: This strategy selects instances uniformly from the unlabeled pool.

In the fine-tuning scenario, fewer existing methods are available. Specifically, we compare our algorithm with Coreset, Random and QBC methods. Although QBC is usually implemented in an iterative fashion, we employ a large query batch size for it to unify the query settings. Our method selects and reweights the unlabeled instances based on the leverage scores (i.e., p = 2) in both scenarios. Note that, in the fine-tuning scenario, our implementations remove the constraint E in Line 15 in Algorithm 1 for better examination of the practicability. In the vanilla deep learning scenario, we use the default training scheme of deep models to replace Line 15 in Algorithm 1. The mean accuracy and the mean MSE are used to evaluate the performances of multiple target models for classification and regression tasks, respectively.

Experiment Results. We report the performance comparison results in Figure 2 and Figure 3. In the scenario of vanilla deep learning, we can observe that our one-shot method achieves comparable performances with the other iterative AL methods in most cases. This phenomenon indicates that our method can significantly reduce the costs of training multiple deep model while preserving its proficiency in the ability of query saving. QBC is the worst one. We find that it causes a severe class imbalance according to the results in Table 5 in the appendix. This may explain its inferior performances. Coreset is usually worse than Random. Note that, the problem settings of [36] and our work are different. there are 50 distinct target networks to be learned in our experiment. The Coreset implementation in [42] solves the coreset problem based on the features extracted by the supernet. A drawback of this approach is that the selected instances may not be useful for other models, because the data representations are different. We believe this is the reason that why Coreset is less effective than Random in our setting. Entropy

Table 3: Comparisons on the running time between our method and the other baselines with a query budget 15000 instances. The running time includes data querying and model training (GPU hours). 

<table><tr><td></td><td>MNIST</td><td>F.MNIST</td><td>K.MNIST</td><td>SVHN</td><td>CIF.10</td><td>CIF.100</td><td>EMN.l.</td><td>EMN.d.</td></tr><tr><td>DIAM</td><td>46.643</td><td>47.597</td><td>46.765</td><td>52.228</td><td>45.493</td><td>53.532</td><td>73.522</td><td>120.840</td></tr><tr><td>QBC</td><td>23.937</td><td>24.419</td><td>24.502</td><td>26.011</td><td>25.541</td><td>30.498</td><td>36.280</td><td>40.231</td></tr><tr><td>Entropy</td><td>24.060</td><td>24.293</td><td>24.455</td><td>25.792</td><td>25.173</td><td>28.655</td><td>34.291</td><td>42.719</td></tr><tr><td>Our</td><td>5.299</td><td>5.366</td><td>5.354</td><td>5.605</td><td>5.350</td><td>5.57</td><td>9.717</td><td>12.711</td></tr><tr><td>Coreset</td><td>5.200</td><td>5.201</td><td>5.285</td><td>5.466</td><td>5.450</td><td>5.745</td><td>8.984</td><td>11.043</td></tr><tr><td>Random</td><td>4.317</td><td>4.333</td><td>4.402</td><td>4.583</td><td>4.567</td><td>4.712</td><td>7.317</td><td>8.027</td></tr></table>

method achieves comparable performances with Random. The reason may also be evidenced by the results in Table 5 in the appendix that their class imbalance ratios are highly consistent, implies that the mean entropy scores tend to have an extremely small standard deviation. The performances of DIAM are less stable. It is effective in the datasets associated with MNIST, but fails on the others. This deficiency has not been observed in our method.

In the scenario of fine-tuning, Figure 3 shows that our approach outperforms than the other baselines with different querying budgets in terms of achieving better mean MSE. These results indicate that our method is effective and robust to different query budgets, it can effectively identify the desired number of useful unlabeled instances under diverse representations to learn linear prediction layers.

We further examine the running time of different AL methods. The results are reported in Table 3. For the one-shot methods Coreset and Random, we report their running time of one-shot querying 15000 instances. It can be observed that the cost of repeated model training is prohibitive in AL for multiple deep models, demonstrating the advantages of one-shot querying. Among the active selection methods, DIAM is the slowest approach because it selects instances based on the predictions of the unlabeled dataset in the latter half of training epochs of each target model. Generating the predictions from multiple models could be expensive, particularly with a large unlabeled pool. QBC and Entropy exhibit similar time costs. Both of them need to feed the unlabeled instances into 50 models to obtain their predictions.

In the fine-tuning scenario, all the compared methods conduct one-shot querying and linear prediction layers are trained with the same computational costs. As a result, the running time of the compared methods is comparable. The results are deferred to Table 6 in the appendix.

# 5 Conclusion

In this paper, we propose a one-shot AL algorithm for multiple deep models. The task is formulated as seeking a shared reweighted sampling matrix to approximately solve multiple $\ell_{p}$ -regression problems for neuron models on distinct deep representations. Our approach is to sample and reweight the unlabeled instances based on their maximum Lewis weights across different representations. We establish an upper bound on the number of samples needed by our algorithm to achieve constant-factor approximations for multiple models and general p. Our techniques on the one hand substantially improve the upper bound on the number of samples of [15] in the case of single model and p = 2, on the other hand remove the $\log n$ factor in [45] for Lewis weight sampling to obtain $\ell_{p}$ -subspace-embedding. Extensive experiments are conducted on 11 benchmarks and 50 deep models. We observe that the sum of the maximum Lewis weights with p = 2 grows very slowly as the number of target models increases, providing a direction for interpreting deep representation learning. The performance comparisons show that our algorithm achieves competitive performances with the state-of-the-art AL methods for multiple deep models.

# Acknowledgments

S.-J. Huang is supported in part by the National Science and Technology Major Project (2020AAA0107000), the Natural Science Foundation of Jiangsu Province of China (BK20222012, BK20211517), and NSFC (62222605). Y. Li is supported in part by the Singapore Ministry of Education (AcRF) Tier 2 grant MOE-T2EP20122-0001 and Tier 1 grant RG75/21. Y.-P. Tang was supported in part by the China Scholarship Council during his visit to Nanyang Technological University, where most of this work was done. The authors would like to thank Aarshvi Gajjar, Chinmay Hedge, Christopher Musco and Xingyu Xu for pointing out an error in an earlier proof of Theorem 3.6.

# References

[1] Karsten M. Borgwardt, Arthur Gretton, Malte J. Rasch, Hans-Peter Kriegel, Bernhard Schölkopf, and Alexander J. Smola. Integrating structured biological data by kernel maximum mean discrepancy. In Proceedings of the 14th Annual International Conference on Intelligent Systems for Molecular Biology, pages 49–57, 2006.   
[2] J. Bourgain, J. Lindenstrauss, and V. Milman. Approximation of zonoids by zonotopes. Acta Mathematica, 162:73 - 141, 1989.   
[3] Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, and Song Han. Once-for-all: Train one network and specialize it for efficient deployment. In Proceedings of the 7th International Conference on Learning Representations, 2019.   
[4] Rita Chattopadhyay, Zheng Wang, Wei Fan, Ian Davidson, Sethuraman Panchanathan, and Jieping Ye. Batch mode active sampling based on marginal probability distribution matching. In Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 741–749, 2012.   
[5] Cheng Chen, Yi Li, and Yiming Sun. Online active regression. In Proceedings of the 39th International Conference on Machine Learning, pages 3320-3335. PMLR, 2022.   
[6] Xue Chen and Michal Derezinski. Query complexity of least absolute deviation regression via robust uniform convergence. In Proceedings of the 34th Annual Conference on Learning Theory, pages 1144-1179. PMLR, 2021.   
[7] Xue Chen and Eric Price. Active regression via linear-sample sparsification. In Proceedings of the 32nd Annual Conference on Learning Theory, pages 663–695. PMLR, 2019.   
[8] Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learning for classical japanese literature. arXiv cs.CV/1812.01718, 2018.   
[9] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik. EMNIST: an extension of MNIST to handwritten letters. arXiv cs.CV//1702.05373, 2017.   
[10] Michael B Cohen and Richard Peng. $L_{p}$ row sampling by Lewis weights. In Proceedings of the 47th Annual ACM Symposium on Theory of Computing, pages 183–192, 2015.   
[11] David Cohn, Les Atlas, and Richard Ladner. Improving generalization with active learning. Machine Learning, 15(2):201-221, 1994.   
[12] Sanjoy Dasgupta and Daniel Hsu. Hierarchical sampling for active learning. In Proceedings of the 25th International conference on Machine learning, pages 208-215, 2008.   
[13] Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. Model compression and hardware acceleration for neural networks: A comprehensive survey. Proceedings of the IEEE, 108(4):485–532, 2020.   
[14] Gabriele Fanelli, Matthias Dantone, Juergen Gall, Andrea Fossati, and Luc Van Gool. Random forests for real time 3d face analysis. International journal of computer vision, 101:437–458, 2013.   
[15] Aarshvi Gajjar, Christopher Musco, and Chinmay Hegde. Active learning for single neuron models with lipschitz non-linearities. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, pages 4101–4113. PMLR, 2023.

[16] Aarshvi Gajjar, Wai Ming Tai, Xingyu Xu, Chinmay Hegde, Christopher Musco, and Yi Li. Agnostic active learning of single index models with linear sample complexity. In Proceedings of COLT, page to appear, 2024.   
[17] Aarshvi Gajjar, Xingyu Xu, Chinmay Hegde, and Christopher Musco. Improved bounds for agnostic active learning of single index models. In RealML Workshop NeurIPS, 2023.   
[18] Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal of Computer Vision, 129:1789–1819, 2021.   
[19] Steven C. H. Hoi, Rong Jin, Jianke Zhu, and Michael R. Lyu. Semi-supervised SVM batch mode active learning for image retrieval. In Proceedings of the 21st IEEE Conference on Computer Vision and Pattern Recognition, 2008.   
[20] Sheng-Jun Huang, Rong Jin, and Zhi-Hua Zhou. Active learning by querying informative and representative examples. IEEE Transactions on pattern analysis and machine intelligence, 36(10):1936–1949, 2014.   
[21] Qiuye Jin, Mingzhi Yuan, Qin Qiao, and Zhijian Song. One-shot active learning for image segmentation via contrastive learning and diversity-based sampling. Knowledge-Based Systems, 241:108278, 2022.   
[22] Andreas Kirsch, Joost van Amersfoort, and Yarin Gal. Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning. In Proceedings of the 33rd Conference on Neural Information Processing Systems, pages 7026–7037, 2019.   
[23] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In Proceedings of the 36th International conference on machine learning, pages 3519–3529. PMLR, 2019.   
[24] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.   
[25] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.   
[26] Michel Ledoux and Michel Talagrand. Probability in Banach Spaces: isoperimetry and processes, volume 23. Springer Science & Business Media, 1991.   
[27] David D Lew is and Jason Catlett. Heterogeneous uncertainty sampling for supervised learning. In Machine Learning: Proceedings of the 11th International Conference, pages 148–156. Elsevier, 1994.   
[28] David D. Lewis and William A. Gale. A sequential algorithm for training text classifiers. In W. Bruce Croft and C. J. van Rijsbergen, editors, Proceedings of the 17th Annual International ACM-SIGIR Conference on Research and Development in Information Retrieval, pages 3–12. ACM/Springer, 1994.   
[29] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In 2015 IEEE International Conference on Computer Vision, 2015.   
[30] Gaurav Menghani. Efficient deep learning: A survey on making deep learning models smaller, faster, and better. ACM Computing Surveys, 55(12):1-37, 2023.   
[31] Cameron Musco, Christopher Musco, David P Woodruff, and Taisuke Yasuda. Active linear regression for $\ell_{p}$ norms and beyond. In Proceedings of the 63rd IEEE Annual Symposium on Foundations of Computer Science, pages 744–753. IEEE, 2022.

[32] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. In NIPS 2011 Workshop on Deep Learning and Unsupervised Feature Learning, 2011.   
[33] Thao Nguyen, Maithra Raghu, and Simon Kornblith. Do wide and deep networks learn the same things? uncovering how neural network representations vary with width and depth. In Proceedings of the 8th International Conference on Learning Representations, 2020.   
[34] Aditya Parulekar, Advait Parulekar, and Eric Price. L1 regression with lewis weights subsampling. Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques, 2021.   
[35] Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Brij B Gupta, Xiao-jiang Chen, and Xin Wang. A survey of deep active learning. ACM computing surveys, 54(9):1–40, 2021.   
[36] Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In Proceedings of the 6th International Conference on Learning Representations, 2018.   
[37] Burr Settles. Active learning literature survey. Technical report, University of Wisconsin-Madison, 2009.   
[38] H Sebastian Seung, Manfred Opper, and Haim Sompolinsky. Query by committee. In Proceedings of the 5th annual workshop on Computational learning theory, pages 287-294, 1992.   
[39] Haochen Shi and Hui Zhou. Deep active sampling with self-supervised learning. Frontiers of Computer Science, 17(4):174323, 2023.   
[40] Neta Shoham and Haim Avron. Experimental design for overparameterized learning with application to single shot deep active learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.   
[41] Yi Sun, Xiaogang Wang, and Xiaoou Tang. Deep convolutional network cascade for facial point detection. In Proceedings of the 26th IEEE Conference on Computer Vision and Pattern Recognition, pages 3476-3483, 2013.   
[42] Ying-Peng Tang and Sheng-Jun Huang. Active learning for multiple target models. In Proceedings of the 36th Conference on Neural Information Processing Systems, 2022.   
[43] Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science, volume 47. Cambridge University Press, 2018.   
[44] Tom J Viering, Jesse H Krijthe, and Marco Loog. Nuclear discrepancy for single-shot batch active learning. Machine Learning, 108(8-9):1561–1599, 2019.   
[45] David P Woodruff and Taisuke Yasuda. Online lewis weight sampling. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms, pages 4622-4666. SIAM, 2023.   
[46] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv cs.LG/1708.07747, 2017.   
[47] Yifan Yan and Sheng-Jun Huang. Cost-effective active learning for hierarchical multi-label classification. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, pages 2962-2968, 2018.

[48] Yazhou Yang and Marco Loog. Single shot active learning using pseudo annotators. Pattern Recognition, 89:22-31, 2019.   
[49] Kai Yu, Jinbo Bi, and Volker Tresp. Active learning via transductive experimental design. In Proceedings of the 23rd International conference on Machine learning, pages 1081-1088, 2006.

# A Subspace Embedding

We note that there are mainly two kinds of $\ell_{p}$ Lewis weight sampling. The first kind is to retain or discard each row independently. Specifically, the i-th row of A is retained with probability $p_{i}$ and discarded with probability $1 - p_{i}$ . The resulting sampled matrix SA has a random number of rows. The second kind has a fixed, prescribed number m of sampled rows. Each sample is i.i.d. chosen to be the i-th row of A with probability $t_{i}/m$ , where $t_{1}, \ldots, t_{n}$ are weights satisfying that $\sum_{i} t_{i} = m$ . We use sampling of the second kind (recall Definition 3.3) in our algorithm. However, our main result (Theorem 3.6) still works for the first kind of sampling matrices, see Appendix B for details.

In this section, we give the sample complexity for $\ell_p$ subspace embedding with distortion $1 + \epsilon$ for $p > 2$ , using both kinds of sampling schemes.

For p > 2, Woodruff and Yasuda [45] consider the first kind of sampling and give a sample complexity of $O(\epsilon^{-2}d^{p/2}(\log^{2}d\log n + \log\frac{1}{\delta}))$ for $\ell_{p}$ -subspace embeddings. This is the first result for p > 2 that has an $\epsilon^{-2}$ dependence, as the only prior result was $O(\epsilon^{-5}d^{p/2}\log d)$ with an $\epsilon^{-5}$ dependence [2]. Still, based on the result of [2], we can improve the analysis of [45] and remove the undesired $\log n$ factor in their sample complexity. We have the following theorem.

Theorem A.1. Let $A \in R^{n \times d}$ , $2 < p < \infty$ and $0 < \epsilon, \delta < 1$ . Let $p_i = \min\{\beta w_i, 1\}$ where $w_i$ is $\ell_p$ Lewis weight of $a_i$ for A and $\beta = \Omega(\frac{d^{\frac{p}{2}-1}}{\epsilon^2}(\log d + \log \frac{1}{\delta}))$ be the oversampling parameter. Let $S \in R^{n \times n}$ be the reweighted sampling matrix in which the i-th row

$$
S _ {i} = \left\{ \begin{array}{l l} \frac {1}{(p _ {i}) ^ {1 / p}} e _ {i} ^ {\top}, & \text {with prob.} p _ {i} \\ 0, & \text {with prob.} 1 - p _ {i}. \end{array} \right.
$$

With probability at least $1 - \delta$ , $S$ has $m = \Omega \left(\frac{d^{\frac{p}{2}}}{\epsilon^{2}} (\log^{2} d \log \frac{d}{\epsilon} + \log \frac{1}{\delta})\right)$ nonzero rows and $\| SAx\|_p^p = (1 \pm \epsilon) \| Ax\|_p^p$ .

We only sketch the changes in the proof of [45]. First, in the sampling we do not use $\gamma$ -one-sided Lewis weights but the exact Lewis weights of $A$ . True Lewis weights do not affect the symmetrization trick. After the symmetrization step, we remove the part of flattening matrix $A$ in their proof. Instead, we claim: (1) By [2, Theorem 7.3], $S$ is a $\frac{1}{2}$ -subspace embedding matrix of $A$ . (2) By [5, Lemma A.1] and Lemma 3.5, Lewis weights of $SA$ are uniformly upper bounded by $\frac{2}{\beta}$ . Conditioned on (1) and (2), it suffices to prove

$$
\underset {S, \sigma} {\mathbb {E}} \max _ {x: \| S A x \| _ {p} \leq 1} \left| \sum_ {k = 1} ^ {m} \sigma_ {k} | (S A) _ {k} x | ^ {p} \right| ^ {\ell} \leq \epsilon^ {\ell}.
$$

This $\ell$ -th moment upper bound can be derived in the same fashion as the end of the proof of Lemma 3.9. Then applying the Markov inequality gives us $\| SAx\|_p^p = (1\pm \epsilon)\| Ax\|_p^p$ with probability at least $1 - \delta$ .

The next theorem gives the sample complexity for $\ell_{p}$ subspace embedding for p > 2 in which samplings are i.i.d. and the probability of every row $a_{i}$ being sampled is $w_{i}/d$ .

Theorem A.2. Let $A \in R^{n \times d}$ , $2 < p < \infty$ and $0 < \epsilon, \delta < 1$ . Suppose that the $\ell_{p}$ Lewis weights of A are $w_{1}, \ldots, w_{n}$ . Let $p_{i} = w_{i}/d$ and $S \in R^{m \times d}$ be a reweighted sampling matrix whose i-th row $S_{i} = \frac{1}{(mp_{i})^{1/p}} e_{j}^{\top}$ with probability $p_{j}$ . Set $m = \Omega(\frac{d^{2}}{\epsilon^{2}} (\log^{2} d \log \frac{d}{\epsilon} + \log \frac{1}{\delta}))$ , then with probability at least $1 - \delta$ , we have $\|SAx\|_{p}^{p} = (1 \pm \epsilon) \|Ax\|_{p}^{p}$ .

We only highlight the necessary changes in the proof of Theorem A.2.

- The symmetrization step goes through in the same fashion as the long chain of inequalities in the proof of Lemma 3.9.   
- By Theorem 7.3 of [2], $S$ is a $\frac{1}{2}$ -subspace embedding matrix of $A$ .   
- By Lemma 3.11, Lewis weights of $SA$ are uniformly upper bounded by $\frac{2}{\beta}$ . The left steps are the same as the changes mentioned for Theorem A.1.

# B Result for the Other Sampling Method

In this section, we prove that our main result Theorem 3.6 still holds if the reweighted sampling matrix S is defined to be of the first kind:

$$
S _ {i} = \left\{ \begin{array}{l l} \frac {1}{(p _ {i}) ^ {1 / p}} e _ {i} ^ {\top}, & \text { with   prob. } p _ {i} \\ 0, & \text { with   prob. } 1 - p _ {i}, \end{array} \right.
$$

where $p_{i} = \beta w_{i}$ and $\beta = \Omega(\frac{d^{\frac{p}{2}-1}}{\epsilon^{2}}(\log^{2} d \log \frac{d}{\epsilon} + \log \frac{1}{\delta}))$ . Accordingly, Lines 3–8 of Algorithm 1 are changed to the following lines.

1: for $i = 1,2,\ldots ,n$ do   
2: if $a_{i}$ is sampled with probability $p_{i} = \beta w_{i}$ then   
3: $S_{i,i} = p_i^{-1 / p}$ and query the label of $a_{i}$

Compared with the proof of Theorem 3.6, the following modifications are needed: (1) By Theorem A.1, S is a 1/2-subspace embedding matrix of A. (2) To show that Lemma 3.9 is true, we observe that by [5, Lemma A.1] and Lemma 3.5, Lewis weights of SA are uniformly upper bounded by $\frac{2}{\beta}$ .

# C Detailed Experimental Settings and Additional Results

# C.1 Detailed Settings

All experiments are run on two GPU servers, each equipped with four GeForce RTX 3090 graphic cards, an Intel Xeon Gold 5317 CPU and 128 GB Memory. The details of the datasets are presented in Table 1. The configurations of 50 deep models are specified in Table 4. Note that the network dentition and pre-trained weights of each configuration can be downloaded from the GitHub page of the OFA project [3].

We follow the training settings of $[42]$ in our experiment of the vanilla deep learning scenario. Specifically, at each query iteration, each target model will be initialized with the pre-trained weights on ImageNet and trained for 20 epochs on the labeled dataset with batch size 32. SGD optimizer is employed with learning rate $1.5 \times 10^{-3}$ , momentum coefficient 0.9 and weight decay factor $3 \times 10^{-5}$ . A dropout rate of 0.1 is used in the training process.

In our experiments of the fine-tuning scenario, we train a linear prediction layer with ReLU activation function using mean squared loss. The training specifications are introduced as follows. The SGD optimizer is employed with a learning rate of $10^{-3}$ and a weight decay coefficient of $10^{-1}$ . The layer is trained for 30 epochs with training batch size 128. We set the

Table 4: The names of the model specifications used in the experiments. The network dentition and pre-trained weights of each configuration can be downloaded from the GitHub page of the OFA project [3]. 

<table><tr><td>flops@595M_top1@80.0_finetune@75</td><td>flops@482M_top1@79.6_finetune@75</td></tr><tr><td>flops@389M_top1@79.1_finetune@75</td><td>LG-G8_lat@24ms_top1@76.4_finetune@25</td></tr><tr><td>LG-G8_lat@16ms_top1@74.7_finetune@25</td><td>LG-G8_lat@11ms_top1@73.0_finetune@25</td></tr><tr><td>LG-G8_lat@8ms_top1@71.1_finetune@25</td><td>s7edge_lat@88ms_top1@76.3_finetune@25</td></tr><tr><td>s7edge_lat@58ms_top1@74.7_finetune@25</td><td>s7edge_lat@41ms_top1@73.1_finetune@25</td></tr><tr><td>s7edge_lat@29ms_top1@70.5_finetune@25</td><td>note8_lat@65ms_top1@76.1_finetune@25</td></tr><tr><td>note8_lat@49ms_top1@74.9_finetune@25</td><td>note8_lat@31ms_top1@72.8_finetune@25</td></tr><tr><td>note8_lat@22ms_top1@70.4_finetune@25</td><td>note10_lat@64ms_top1@80.2_finetune@75</td></tr><tr><td>note10_lat@50ms_top1@79.7_finetune@75</td><td>note10_lat@41ms_top1@79.3_finetune@75</td></tr><tr><td>note10_lat@30ms_top1@78.4_finetune@75</td><td>note10_lat@22ms_top1@76.6_finetune@25</td></tr><tr><td>note10_lat@16ms_top1@75.5_finetune@25</td><td>note10_lat@11ms_top1@73.6_finetune@25</td></tr><tr><td>note10_lat@8ms_top1@71.4_finetune@25</td><td>pixel1_lat@143ms_top1@80.1_finetune@75</td></tr><tr><td>pixel1_lat@132ms_top1@79.8_finetune@75</td><td>pixel1_lat@79ms_top1@78.7_finetune@75</td></tr><tr><td>pixel1_lat@58ms_top1@76.9_finetune@75</td><td>pixel1_lat@40ms_top1@74.9_finetune@25</td></tr><tr><td>pixel1_lat@28ms_top1@73.3_finetune@25</td><td>pixel1_lat@20ms_top1@71.4_finetune@25</td></tr><tr><td>pixel2_lat@62ms_top1@75.8_finetune@25</td><td>pixel2_lat@50ms_top1@74.7_finetune@25</td></tr><tr><td>pixel2_lat@35ms_top1@73.4_finetune@25</td><td>pixel2_lat@25ms_top1@71.5_finetune@25</td></tr><tr><td>1080ti_gpu64@27ms_top1@76.4_finetune@25</td><td>1080ti_gpu64@22ms_top1@75.3_finetune@25</td></tr><tr><td>1080ti_gpu64@15ms_top1@73.8_finetune@25</td><td>1080ti_gpu64@12ms_top1@72.6_finetune@25</td></tr><tr><td>v100_gpu64@11ms_top1@76.1_finetune@25</td><td>v100_gpu64@9ms_top1@75.3_finetune@25</td></tr><tr><td>v100_gpu64@6ms_top1@73.0_finetune@25</td><td>v100_gpu64@5ms_top1@71.6_finetune@25</td></tr><tr><td>tx2_gpu16@96ms_top1@75.8_finetune@25</td><td>tx2_gpu16@80ms_top1@75.4_finetune@25</td></tr><tr><td>tx2_gpu16@47ms_top1@72.9_finetune@25</td><td>tx2_gpu16@35ms_top1@70.3_finetune@25</td></tr><tr><td>cpu_lat@17ms_top1@75.7_finetune@25</td><td>cpu_lat@15ms_top1@74.6_finetune@25</td></tr><tr><td>cpu_lat@11ms_top1@72.0_finetune@25</td><td>cpu_lat@10ms_top1@71.1_finetune@25</td></tr></table>

random seed to 0 for reproducibility. Please refer to the submitted source code to reproduce our results.

The implementations of each compared method are introduced as follows. We use the code in $[42]$ to implement DIAM, Coreset and Entropy methods. Specifically, DIAM first obtains the predictions of the unlabeled instances using the models in the latter half of training epochs of each target network. Then, it selects the batch of instances that multiple models have inconsistent predictions. Coreset selects data points based on the representation of the pre-trained super-net in OFA. Entropy calculates the entropy scores of unlabeled instances based on the predictions of each target model. Subsequently, it selects the instances with the highest mean entropy scores across multiple model predictions.

# C.2 Additional results

# C.2.1 Study on Mean Percentage of Covered Instances

We further examine how many instances with high leverage scores under the representation of a single model can be covered by the maximum leverage score sampling. The statistics are calculated as follows: We first get the intersection between the sets of instances that have the top $t\%$ highest leverage score under the representation of model $j$ (denoted by $I_j^t$ ) and top $t\%$ highest Maximum leverage score (denoted by $I^t$ ). Then, we divide the cardinality of this subset by the number of $t\%$ unlabeled instances. Finally, we calculate this value for each $j \in [k]$ and

Table 5: The class imbalance ratio of different query strategies. 

<table><tr><td></td><td>MNIST</td><td>F.MNIST</td><td>K.MNIST</td><td>SVHN</td><td>CIF.10</td><td>CIF.100</td><td>EMN.l.</td><td>EMN.d.</td></tr><tr><td>DIAM</td><td>2.007</td><td>3.250</td><td>1.472</td><td>2.121</td><td>1.525</td><td>4.390</td><td>2.216</td><td>2.811</td></tr><tr><td>QBC</td><td>1.867</td><td>6.721</td><td>1.987</td><td>4.474</td><td>5.212</td><td>13.607</td><td>9.178</td><td>10.664</td></tr><tr><td>Coreset</td><td>3.561</td><td>3.708</td><td>2.116</td><td>2.330</td><td>1.871</td><td>6.000</td><td>5.375</td><td>4.491</td></tr><tr><td>Random</td><td>1.262</td><td>1.091</td><td>1.067</td><td>3.008</td><td>1.062</td><td>1.511</td><td>1.092</td><td>1.222</td></tr><tr><td>Entropy</td><td>1.331</td><td>1.077</td><td>1.091</td><td>3.052</td><td>1.087</td><td>1.439</td><td>1.088</td><td>1.166</td></tr><tr><td>Our</td><td>1.217</td><td>1.081</td><td>1.050</td><td>2.993</td><td>1.098</td><td>1.440</td><td>1.109</td><td>1.209</td></tr></table>

compute the average to obtain the mean percentage of covered instances, i.e.,

$$
\kappa (t) = \frac {1}{k} \sum_ {j = 1} ^ {k} \frac {| I _ {j} ^ {t} \cap I ^ {t} |}{| I ^ {t} |}.
$$

We report the mean percentage of covered instances of 50 deep models in Figure 4. It can be observed that $\kappa(10)$ is about $30\%$ on most datasets (except for Biwi), that is, $I^{10}$ covers on average about $30\%$ of the instances with high leverage scores of each representation for most datasets. For all datasets, as $t$ increases, $\kappa(t)$ increases rapidly. These phenomena suggest that sampling a modest number of instances by maximum leverage scores can effectively train multiple deep models, as there are a significant fraction of instances with high leverage scores shared across different models.

# C.2.2 Study on the Class Imbalance Ratio

Another metric of interest for AL classification algorithms is the class imbalance ratio, which is defined as $\max_{c}\sum_{i\in[n_{l}]}\mathbb{I}\{y_{i}=c\}/\min_{c}\sum_{i\in[n_{l}]}\mathbb{I}\{y_{i}=c\}$ , where I is the indicator function. Some active query strategies may cause severe class imbalance, rendering them hardly generalizable to other target models and learning tasks. This issue becomes more significant for multiple target models. In this experiment, we examine the class imbalance ratios of different AL methods in the classification tasks. Specifically, we compare the ratios after the third AL iteration, where a total of 12000 labeled instances are present, including the initially labeled set. The results are reported in Table 5.

We can observe that our proposed method consistently produces a balanced labeled dataset. Its class imbalanced ratio is very close to that of the Random sampling. On the other hand, Coreset suffers from the class imbalance, suggesting that the training instances with different classes exhibit diverse intra-class distances under deep representation. This implies that the instances sampled by Coreset may have a large distribution gap with the dataset. Entropy obtains a similar class imbalance ratio to that of Random, which might appear counter-intuitive, given that Entropy prefers the instances near the decision boundary and such instances are typically less likely to be class-balanced. A possible reason is that the mean entropy scores of the unlabeled instances may have a small standard deviation, potentially diminishing the advantages of using Entropy for identifying the most informative instances, in the setting of multiple target models.

Table 6: Running time (hours) of the methods in regression benchmarks. The running time includes model training and data querying. 

<table><tr><td>Methods</td><td>Biwi</td><td>FLD</td><td>CelebA</td></tr><tr><td>Random</td><td>1.217</td><td>1.383</td><td>1.202</td></tr><tr><td>Coreset</td><td>1.483</td><td>1.883</td><td>3.667</td></tr><tr><td>Our</td><td>1.551</td><td>1.850</td><td>5.817</td></tr><tr><td>QBC</td><td>1.632</td><td>1.800</td><td>4.110</td></tr></table>

![](images/26c76a7c585f2196cff0e9040709f16061244e1d762b5238628dc2baa8e93ceb.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % Maximum Lewis weights | mean percentage of covered instances |
| ---------------------------------------------------------- | ------------------------------------ |
| 10%                                                        | 25.8                                 |
| 20%                                                        | 39.0                                 |
| 30%                                                        | 49.5                                 |
| 40%                                                        | 58.3                                 |
| 50%                                                        | 66.5                                 |
| 60%                                                        | 74.1                                 |
| 70%                                                        | 81.7                                 |
| 80%                                                        | 89.1                                 |
| 90%                                                        | 95.2                                 |
</details>

(a) MNIST

![](images/28f035092ff184532d1b14aa54220c93e88c71212fddfadca07247f256f99f35.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % Maximum Lewis weights | mean percentage of covered instances |
| ---------------------------------------------------------- | ------------------------------------ |
| 10%                                                        | 33.1                                 |
| 20%                                                        | 41.9                                 |
| 30%                                                        | 49.6                                 |
| 40%                                                        | 56.9                                 |
| 50%                                                        | 63.8                                 |
| 60%                                                        | 70.5                                 |
| 70%                                                        | 77.3                                 |
| 80%                                                        | 84.2                                 |
| 90%                                                        | 91.4                                 |
</details>

(b) Kuzushiji-MNIST

![](images/9861095010121f905d048a8d1324b216cfcf61b43ba2fbdb6df919b32c021417.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % Maximum Lewis weights | mean percentage of covered instances |
| ---------------------------------------------------------- | ------------------------------------- |
| 10%                                                        | 32.0                                  |
| 20%                                                        | 43.0                                  |
| 30%                                                        | 51.6                                  |
| 40%                                                        | 59.1                                  |
| 50%                                                        | 65.8                                  |
| 60%                                                        | 71.9                                  |
| 70%                                                        | 77.7                                  |
| 80%                                                        | 84.1                                  |
| 90%                                                        | 91.0                                  |
</details>

(c) Fashion-MNIST

![](images/2786c7a2c45d6f83c4309b00fbb7f07a706039baafa9071e52fdfd2473f6e346.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % Maximum Lewis weights | mean percentage of covered instances |
| :--- | :--- |
| 10% | 34.8 |
| 20% | 43.4 |
| 30% | 50.6 |
| 40% | 57.5 |
| 50% | 64.1 |
| 60% | 70.6 |
| 70% | 77.5 |
| 80% | 84.7 |
| 90% | 92.1 |
</details>

(d) EMNIST-letters

![](images/de1758fc8df8187973389c9d6891d2ba74e175645e55a44d8a349feb96da67cd.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % Maximum Lewis weights | mean percentage of covered instances |
| :--- | :--- |
| 10% | 25.1 |
| 20% | 37.0 |
| 30% | 47.2 |
| 40% | 56.3 |
| 50% | 65.0 |
| 60% | 73.1 |
| 70% | 80.9 |
| 80% | 88.6 |
| 90% | 95.0 |
</details>

(e) EMNIST-digits

![](images/86be3355637b6e4b01dbe022d106a1c917ec9cc74757f8885c7fc270fa4c99b9.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % | mean percentage of covered instances |
|---|---|
| 10% | 38.1 |
| 20% | 49.5 |
| 30% | 57.8 |
| 40% | 64.5 |
| 50% | 70.3 |
| 60% | 75.6 |
| 70% | 80.7 |
| 80% | 85.8 |
| 90% | 92.0 |
The chart displays a single bar for each percentage of the instances with the top % maximum Lewis weights. The values are explicitly labeled on the bars.
</details>

(f) SVHN

![](images/47ff9801db109bd2000fa0a640094d87a3a7c999903f394002b1e5b849e01428.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % Maximum Lewis weights | mean percentage of covered instances |
| :--- | :--- |
| 10% | 39.1 |
| 20% | 50.1 |
| 30% | 58.1 |
| 40% | 64.6 |
| 50% | 70.1 |
| 60% | 75.4 |
| 70% | 80.5 |
| 80% | 86.0 |
| 90% | 91.9 |
</details>

(g) CIFAR-10

![](images/0430a3a7126edad61ca7d5c700bffefc9688f11592627b95e616ee2ea4b00250.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % | mean percentage of covered instances |
|---|---|
| 10% | 45.7 |
| 20% | 55.4 |
| 30% | 62.4 |
| 40% | 67.9 |
| 50% | 72.8 |
| 60% | 77.3 |
| 70% | 81.8 |
| 80% | 86.7 |
| 90% | 92.4 |
Maximum Lewis weights
</details>

(h) CIFAR-100

![](images/95adb3d7bd936dc9e0d1952853b561fbdae1f97dc71bb98fbeea303a2e01d2ac.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % Maximum Lewis weights | mean percentage of covered instances (%) |
| :--- | :--- |
| 10% | 13.8 |
| 20% | 26.4 |
| 30% | 37.0 |
| 40% | 44.8 |
| 50% | 56.0 |
| 60% | 64.5 |
| 70% | 72.6 |
| 80% | 80.8 |
| 90% | 89.6 |
</details>

(i) Biwi

![](images/f2b9f1b42da9eed81baa6c2fd0ef5004e45c4bdac26127ac5b7740ad17b720d0.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % | mean percentage of covered instances |
| ------------------------------------- | ------------------------------------ |
| 10%                                   | 34.5                                 |
| 20%                                   | 44.1                                 |
| 30%                                   | 51.6                                 |
| 40%                                   | 58.8                                 |
| 50%                                   | 65.6                                 |
| 60%                                   | 72.1                                 |
| 70%                                   | 78.1                                 |
| 80%                                   | 84.6                                 |
| 90%                                   | 91.6                                 |
</details>

(j) FLD

![](images/8b4148d5bb3f9b9b8c7d63ce7cf25b61758e27284359b8d855a352ff754a9cf7.jpg)

<details>
<summary>bar</summary>

| percentage of the instances with top % | mean percentage of covered instances |
| :--- | :--- |
| 10% | 33.6 |
| 20% | 44.1 |
| 30% | 57.2 |
| 40% | 59.0 |
| 50% | 65.2 |
| 60% | 71.2 |
| 70% | 77.4 |
| 80% | 84.0 |
| 90% | 90.9 |
maximum Lewis weights
</details>

(k) CelebA   
Figure 4: The mean percentage of shared data between instances having the highest maximum leverage score and those having the highest leverage score under the representation of a specific deep model.