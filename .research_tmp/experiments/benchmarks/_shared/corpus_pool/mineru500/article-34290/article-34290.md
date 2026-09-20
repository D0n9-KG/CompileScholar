# Hybrid Decentralized Optimization: Leveraging Both First- and Zeroth-Order Optimizers for Faster Convergence

Matin Ansaripour $^{1*}$ , Shayan Talaei $^{2*}$ , Giorgi Nadiradze $^{3}$ , Dan Alistarh $^{3}$

$^{1}$ École Polytechnique Fédérale de Lausanne (EPFL)

$^{2}$ Stanford University

$^{3}$ Institute of Science and Technology Austria (ISTA)

stalaei@stanford.edu, matin.ansaripour@epfl.ch, {giorgi.nadiradze, dan.alistarh}@ista.ac.at

# Abstract

Distributed optimization is the standard way of speeding up machine learning training, and most of the research in the area focuses on distributed first-order, gradient-based methods. Yet, there are settings where some computationally-bounded nodes may not be able to implement first-order, gradient-based optimization, while they could still contribute to joint optimization tasks. In this paper, we initiate the study of hybrid decentralized optimization, studying settings where nodes with zeroth-order and first-order optimization capabilities co-exist in a distributed system, and attempt to jointly solve an optimization task over some data distribution. We essentially show that, under reasonable parameter settings, such a system can not only withstand noisier zeroth-order agents but can even benefit from integrating such agents into the optimization process, rather than ignoring their information. At the core of our approach is a new analysis of distributed optimization with noisy and possibly-biased gradient estimators, which may be of independent interest. Our results hold for both convex and non-convex objectives. Experimental results on standard optimization tasks confirm our analysis, showing that hybrid first-zeroth order optimization can be practical, even when training deep neural networks.

Code — https://github.com/ShayanTalaei/HDO

Extended version — https://arxiv.org/abs/2210.07703

# Introduction

One key enabler of the extremely rapid recent progress of machine learning has been distributed optimization: the ability to efficiently optimize over large quantities of data, and large parameter counts, among multiple nodes or devices, in order to share the computational load, and therefore reduce end-to-end training time. Distributed machine learning has become commonplace, and it is not unusual to encounter systems which distribute model training among tens or even hundreds of nodes.

By and large, the standard distribution strategy in the context of machine learning tasks has been data-parallel (Bottou 2010), using first-order gradient estimators. We can formalize this as follows: considering a classical empirical risk minimization setting, we have a set of samples S from a distribution, and wish to minimize the function $f : R^{d} \to R$ , which is the average of losses over samples from S. In other words, we wish to find $x^{\star} = \arg\min_{x} \sum_{s \in S} f_{s}(x)/|S|$ . Assuming that we have n compute nodes which can process samples in parallel, data-parallel SGD consists of iterations in which each node computes gradient estimator for a batch of samples, and then nodes then exchange this information, either globally, via all-to-all communication, or pair-wise. Specifically, in this paper we will focus on the highly-popular decentralized optimization case, in which nodes interact in randomly chosen pairs, exchanging model information, following each local optimization step.

There is already a vast amount of literature on decentralized optimization in the case where nodes have access to first-order, gradient-based estimators. While this setting is prevalent, it does not cover the interesting case where, among the set of nodes, a fraction only have access to weaker, zeroth-order gradient estimators, corresponding to less computationally-capable devices, but which may still possess useful local data and computation.

In this paper, we initiate the study of hybrid decentralized optimization in the latter setting. Specifically, we aim to answer the following key question:

Can zeroth-order estimators be integrated in a decentralized setting, and can they boost convergence?

Roughly, we show that the answer to this question is affirmative. To arrive at it, we must overcome a number of nontrivial technical obstacles, and the answer must be qualified by key parameters, such as the first-order/zeroth-order split in the population, and the estimator variance and bias. More precisely, a key difficulty we must overcome in the algorithm and in the analysis is the fact that, under standard implementations, zeroth-order estimators are biased, breaking one of the key analytic assumptions in existing work on decentralized optimization, e.g. (Lian et al. 2017a; Wang and Joshi 2021; Koloskova et al. 2020a,b; Nadiradze et al. 2021).

Our analysis approach overcomes this obstacle and provides the first convergence bounds for hybrid decentralized optimization via a novel potential argument. Roughly, assuming a d-dimensional and L-smooth finite-sum objective function f, and a population of n nodes, in which $n_{1}$ have first-order stochastic gradient estimators of variance $\sigma_{1}$ , and

$n_{0}$ have zeroth-order estimators of variance $\sigma_{0}$ , then our analysis shows that the “stochastic noise” in the convergence of our hybrid decentralized optimization algorithm in this population is given, up to constants, by the following three quantities:

$$
\frac {\eta (d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n ^ {2}}, \frac {\eta (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{n ^ {2}}, \eta^ {2} \left(\frac {L d n _ {0}}{n}\right) ^ {k}. \tag {1}
$$

In this expression, $\eta$ is the learning rate, and the quantities $\varsigma_{1}$ and $\varsigma_{0}$ are bounds on the average variance of first-order and zeroth-order estimators at the nodes, respectively, given by the way in which the data is split among these two types of agents. Intuitively, the first term is the variance due to the (random) data split, whereas the second term is the added variance due to noise in the two types of gradient estimators. (The zeroth-order terms are scaled by the dimension, as is common in this case.) The third term bounds the bias induced by the zeroth-order gradient estimators, where k equals 1 for the convex case and 2 for the non-convex case. Using this characterization, we show that there exist reasonable parameter settings such that, if zeroth-order nodes do not have extremely high variance, they may in fact be useful for convergence, especially since the third bias term can be controlled via the learning rate $\eta$ .

Our analysis approach should be of independent interest: first, we provide a simple and general way of characterizing convergence in a population mixing first- and zeroth-order agents, which can be easily parametrized given population and estimator properties, for both convex and nonconvex objectives. (For instance, we can directly cover the case when the zeroth-order estimators are unbiased (Chen 2020) as in this case the bias term becomes zero.) Second, we do so in a very general communication model which allows agents to interact at different rates (due to randomness), covering both the pair-wise interaction model (Angluin et al. 2006; Nadiradze et al. 2021) and the global matching interactions model (Lian et al. 2017a; Wang and Joshi 2021; Koloskova et al. 2020a,b).

A key remaining question is whether the above characterization can be validated for practical setting. For this, we implemented our algorithm and examined the convergence under various optimization tasks, population relative sizes, and estimator implementations. Specifically, we implemented three different types of zeroth-order estimators: a standard biased one, e.g. (Nesterov and Spokoiny 2017), a de-biased estimator (Chen 2020), and the novel gradient-free estimator of (Baydin et al. 2022), and examined their behavior when mixed with first-order estimators. In brief, our results show that, even for high-dimensional and complex tasks, such as fine-tuning the ResNet-18 (He et al. 2015), or a Transformer model (Vaswani et al. 2023), our approach continues to converge. Importantly, we observe that our approach allows a system to incorporate information from the zeroth-order agents in an efficient and robust, showing higher convergence speed relative to the case where only first-order information is considered for optimization.

Related Work. The study of decentralized optimization algorithms dates back to Tsitsiklis (1984), and is related to the study of gossip algorithms for information dissemination (Kempe, Dobra, and Gehrke 2003; Xiao and Boyd 2004). The distinguishing feature of this setting is that optimization occurs jointly, but in the absence of a coordinator node. Several classic first-order algorithms have been ported and analyzed in the gossip setting, such as subgradient methods for convex objectives (Nedic and Ozdaglar 2009; Johansson, Rabi, and Johansson 2009; Shamir and Srebro 2014) or ADMM (Wei and Ozdaglar 2012; Iutzeler et al. 2013). References (Lian et al. 2017a,b; Assran et al. 2018) consider SGD-type algorithms in the non-convex setting, while references (Tang et al. 2018; Koloskova et al. 2020a; Nadiradze et al. 2021) analyzed the use of quantization in the gossip setting. By contrast, zeroth-order optimization has been relatively less investigated: Sahu and Kar (2020) proposes a distributed deterministic zeroth-order Frank-Wolfe-type algorithm, whereas other works by (Yuan et al. 2024) and (Mhanna and Assaad 2023) investigated the rates which can be achieved by decentralized zeroth-order algorithms, proposing multi-stage methods which can match the rate of centralized algorithms in some parameter regimes. Relative to the latter reference, we focus on simpler decentralized algorithms, which can easily interface with first-order optimizers, and perform a significantly more in-depth experimental validation.

Stochastic zeroth-order optimization has been classically applied for gradient-free optimization of convex functions, e.g. (Nesterov and Spokoiny 2017), and has been extended to tackling high-dimensionality and saddle-point constraints, e.g. (Balasubramanian and Ghadimi 2022). (The area has tight connections to bandit online optimization, under time-varying objective functions, e.g. (Flaxman, Kalai, and McMahan 2004; Agarwal, Dekel, and Xiao 2010; Shamir 2017); however, our results are not immediately relevant to this direction, as we are interested in interactions with agents possessing first-order information as well.) In this paper, we also investigate improved single-point function evaluation for better gradient estimation (Jongeneel, Yue, and Kuhn 2021) as well as the forward-mode unbiased estimator of Baydin et al. (2022).

# Preliminaries

# The System Model

We consider a standard model for the decentralized optimization setting, which is similar to (Koloskova et al. 2020a,b; Lian et al. 2017a; Nadiradze et al. 2021). Specifically, we have $n \geq 2$ agents, of which $n_0$ agents have zeroth-order gradient oracles, and $n_1$ have first-order gradient oracles. (We describe the exact optimization setup in the next section.) Beyond their oracle type, the agents are assumed to be anonymous for the purposes of the protocol. The execution will proceed in discrete steps, or rounds, where in each step, two agents are chosen to interact, uniformly at random. Specifically, when chosen, each agent performs some local computation, e.g. obtains some gradient information from their local oracle. Then, the two agents exchange parameter information, and update their local models, after which they are ready to proceed to the next round. Notice that

this random interaction model is asynchronous, in the sense that the number of interactions taken by agents up to some point in time may be different, due to randomness. The basic unit of time used in the analysis, which we call fine-grained time, will be the total number of interactions among agents up to some given point in the execution. To express global progress, we will consider parallel time, which is the average number of interactions up to some point, and can be obtained by dividing by n the total number of interactions. This corresponds to the intuition that $\Theta(n)$ interactions may occur in parallel. In experiments, we will examine the convergence of the local model at a fixed node.

This model is an instantiation of the classic population model of distributed computing (Angluin et al. 2006), in an optimization setting. The model is similar to the one adopted by Nadiradze et al. (2021) for analyzing asynchronous decentralized SGD, and is more general than the ones adopted by Koloskova et al. (2020a,b); Lian et al. (2017a); Wang and Joshi (2021) for decentralized analysis, since the latter assume that nodes are paired via perfect global random matchings in each round. (Our analysis would easily extend to global matching, yielding virtually the same results.)

# Optimization Setup

We assume each node $i$ has a local data distribution $\mathcal{D}^i$ , and that the loss function corresponding to the samples at node $i$ , denoted by $f^i(x): \mathbb{R}^d \to \mathbb{R}$ can be approximated using its stochastic form $F^i(x, \xi^i)$ for each parameter $x \in \mathbb{R}^d$ and (randomly chosen) sample $\xi^i \sim \mathcal{D}^i$ , where $f^i(x) = \mathbb{E}_{\xi^i \sim \mathcal{D}^i} [F^i(x, \xi^i)]$ . For simplicity of notation, we assume that nodes in the set $N_0 = \{1, 2, ..., n_0\}$ are zeroth-order nodes and the nodes in the set $N_1 = [n] / N_0$ are first-order nodes. Let $n_0$ and $n_1$ be the sizes of the sets $N_0$ and $N_1$ correspondingly.

In this setup nodes communicate to solve a distributed stochastic optimization problem, i.e.

$$
f ^ {*} = \min _ {x \in \mathbb {R} ^ {d}} \left[ f (x) := \frac {1}{n _ {0}} \sum_ {i \in N _ {0}} f ^ {i} (x) + \frac {1}{n _ {1}} \sum_ {i \in N _ {1}} f ^ {i} (x) \right].
$$

This means that we wish to optimize the function f which corresponds to the loss over all data samples. Since in the analysis we will wish to throttle the ratio of zeroth-order to first-order agents, we split the entire data among zeroth-order nodes, and we do the same thing for the first-order nodes. (Our analysis can be extended to settings where this is not the case, but this will allow us for instance to study what happens when either $n_{0}$ or $n_{1}$ goes to zero, without changing our objective function.) We make the following assumptions on the optimization objectives:

Assumption 1 (Strong convexity). We assume that the function $f$ is strongly convex with parameter $\ell > 0$ , i.e. for all $x, y \in \mathbb{R}^d$ :

$$
(x - y) ^ {T} (\nabla f (x) - \nabla f (y)) \geq \ell \| x - y \| ^ {2}.
$$

Assumption 2 (Smooth gradient). All the stochastic gradients $\nabla F^i$ are $L$ -Lipschitz for some constant $L > 0$ , i.e. for all $\xi^i \sim \mathcal{D}^i$ and $x, y \in \mathbb{R}^d$ :

$$
\left\| \nabla F ^ {i} (x, \xi^ {i}) - \nabla F ^ {i} (y, \xi^ {i}) \right\| \leq L \| x - y \|. \tag {2}
$$

If in addition $F^{i}$ are convex functions, then

$$
\left\| \nabla F ^ {i} (x, \xi^ {i}) - \nabla F ^ {i} (y, \xi^ {i}) \right\| \leq
$$

$$
2 L (F ^ {i} (x, \xi^ {i})) - F ^ {i} (y, \xi^ {i}) - \langle x - y, \nabla F ^ {i} (y, \xi^ {i}) \rangle).
$$

Using Assumption 2, one can easily find that the gradients of f and $f^{i}(x) \forall i \in [n]$ are also satisfying the above inequalities. Further, we make the following assumptions about the data split and the stochastic gradient estimators:

Assumption 3 (Balanced data distribution). The average variance of $\nabla f^{i}(x)s$ for both zero and first order nodes is bounded by a global constant values, i.e. for all $x \in R^{d}$ :

$$
\frac {1}{n _ {0}} \sum_ {i \in N _ {0}} \| \nabla f ^ {i} (x) - \nabla f (x) \| ^ {2} \leq \varsigma_ {0} ^ {2};
$$

$$
\frac {1}{n _ {1}} \sum_ {i \in N _ {1}} \| \nabla f ^ {i} (x) - \nabla f (x) \| ^ {2} \leq \varsigma_ {1} ^ {2}.
$$

Assumption 4 (Unbiasedness and bounded variance). For each i, $\nabla F^{i}(x,\xi^{i})$ is an unbiased estimator of $\nabla f^{i}(x)$ and its variance is bounded by a constant $s_{i}^{2}$ , i.e. for all $x\in R^{d}$ :

$$
\mathbb {E} _ {\xi^ {i} \sim \mathcal {D} ^ {i}} [ \nabla F ^ {i} (x, \xi^ {i}) ] = \nabla f ^ {i} (x);
$$

$$
\mathbb {E} _ {\xi^ {i}} \left\| \nabla F ^ {i} (x, \xi^ {i}) - \nabla f ^ {i} (x) \right\| \leq s _ {i} ^ {2}.
$$

Each node has access to an estimator $G^{i}(x)$ that estimates the local gradient $\nabla f^{i}(x)$ at point x. For nodes which can perform the gradient computation over a batch of data, i.e. first-order nodes, $G^{i}(x)$ is $\nabla F^{i}(x, \xi^{i})$ , where $\xi^{i} \sim D^{i}$ .

Definition 1. We define the average of $s_{i}$ for the zeroth and first order populations as $\sigma_{0}^{2}$ and $\sigma_{1}^{2}$ respectively. Formally, we define

$$
\sigma_ {0} ^ {2} := \frac {1}{n _ {0}} \sum_ {i \in n _ {0}} s _ {i} ^ {2}, \quad \sigma_ {1} ^ {2} := \frac {1}{n _ {1}} \sum_ {i \in n _ {1}} s _ {i} ^ {2}.
$$

# Zeroth-order Optimization

We now provide a brief introduction relative to standard basic facts and assumptions concerning zeroth-order optimization. Let the function $f_{\nu}^{i}(x) := \mathbb{E}_{u}[f^{i}(x + \nu u)], u \sim N(0, I_{d})$ be the smoothed version of each function $f^{i}(x)$ . Then, node i can estimate the gradient of $f_{\nu}^{i}$ by only evaluating some points of $f^{i}$ .

Definition 2 (Zeroth-order estimator).

$$
G _ {\nu} ^ {i} (x, u, \xi^ {i}) = \frac {F ^ {i} (x + \nu u , \xi^ {i}) - F ^ {i} (x , \xi^ {i})}{\nu} u, \tag {3}
$$

where $u \sim N(0, I_d)$ and $\xi^i \sim \mathcal{D}^i$ .

Note that under Assumption 4, one can easily prove that $G_{\nu}^{i}(x,u,\xi^{i})$ is an unbiased estimator of $\nabla f_{\nu}^{i}$ since

$$
\begin{array}{l} \mathbb {E} _ {u, \xi^ {i}} [ G _ {\nu} ^ {i} (x, u, \xi^ {i}) ] = \mathbb {E} _ {u} [ \frac {f ^ {i} (x + \nu u) - f ^ {i} (x)}{\nu} u ] \\ = \nabla f _ {\nu} ^ {i} (x). \tag {4} \\ \end{array}
$$

As a technical note, in our analysis we will set $\nu := \frac{\eta}{c}$ , where $\eta$ is the learning rate and c is a constant to be defined later. Therefore, for simplicity we can define $G^{i}(x) :=$

$G_{\nu}^{i}(x,u,\xi^{i})$ , where $G_{\nu}^{i}(x,u,\xi)$ is as defined in Definition 2 and $\nu=\frac{\eta}{c}$ . Since zeroth-order nodes cannot perform gradient computation directly, we use this $G^{i}(x)$ as their gradient estimator. We restate the following well-known fact:

Lemma 1 ((Nesterov and Spokoiny 2017), Theorem 1.1 in (Balasubramanian and Ghadimi 2022)). For a Gaussian random vector $u \sim N(0, I_d)$ we have that

$$
\mathbb {E} [ \| u \| ^ {k} ] \leq (d + k) ^ {k / 2} \tag {5}
$$

for any $k \geq 2$ . Moreover, the following statements hold for any function $f$ whose gradient is Lipschitz continuous with constant $L$ .

a) The gradient of $f_{\nu}$ is Lipschitz continuous with constant $L_{\nu}$ such that $L_{\nu} \leq L$ .   
b) For any $x \in R^{d}$ ,

$$
| f _ {\nu} (x) - f (x) | \leq \frac {\nu^ {2}}{2} L d,
$$

$$
\| \nabla f _ {\nu} (x) - \nabla f ^ {i} (x) \| \leq \frac {\nu}{2} L (d + 3) ^ {\frac {3}{2}}.
$$

c) For any $x \in R^{n}$ ,

$$
\frac {1}{\nu^ {2}} \mathbb {E} _ {u} [ \{f (x + \nu u) - f (x) \} ^ {2} \| u \| ^ {2} ] \leq
$$

$$
\frac {\nu^ {2}}{2} L ^ {2} (d + 6) ^ {3} + 2 (d + 4) \| \nabla f (x) \| ^ {2}.
$$

# The HDO Algorithm

Algorithm Description. We now describe a decentralized optimization algorithm, designed to be executed by a population of n nodes, interacting in pairs chosen uniformly at random as per our model. We assume that $n_{1}$ of the nodes have access to first-order estimators and $n_{0}$ of them have access to zeroth-order estimators, hence $n = n_{1} + n_{0}$ . Two copies of the training data are distributed, once among the first-orders and once among the zeroth-orders. Thus, each first- and zeroth-order node has access to $\frac{1}{n_{1}}$ , $\frac{1}{n_{0}}$ of the entire training data, respectively. We assume that each node i has access to a local stochastic estimator of the gradient, which we denote by $G^{i}$ , and maintains a model estimate $X^{i}$ , as well as the global learning rate $\eta$ . Without loss of generality, we assume that the models are initialized to the same randomly-chosen point. Specifically, upon every interaction, the interacting agents i and j perform the following steps:

Algorithm 1: HDO pseudocode for each interaction between randomly chosen nodes i and j   
// Nodes perform local steps. $X^{i} \leftarrow X^{i} - \eta G^{i}(X^{i});$ $X^{j} \leftarrow X^{j} - \eta G^{j}(X^{j});$ // Nodes average their local models. $avg \leftarrow (X^{i} + X^{j})/2;$ $X^{i} \leftarrow avg;$ $X^{j} \leftarrow avg;$

In a nutshell, upon each interaction, each node first performs a local model update based on its estimator, and then nodes average their local models following the interaction. We do not distinguish between estimator types in this interaction. The nodes are then ready to proceed to the next round.

# The Convergence of the HDO Algorithm

This section is dedicated to proving that the following result

Theorem 1. Assume an objective function $f: \mathbb{R}^d \to \mathbb{R}$ , equal to the average loss over all data samples, whose optimum $x^*$ we are trying to find using Algorithm 1. Let $n_0$ be the number of zeroth-order nodes, and $n_1$ be the number of first-order agents. Given the data split described in the previous section, let $f_i$ be the local objective function of node $i$ . Assume that zeroth-order nodes use estimators with $\nu = \frac{\eta}{\sqrt{d}}$ . Let the total number of steps in the algorithm $T$ and $\mu_t = \sum_{i=1}^{n} X_t^i / n$ , then we can derive the following convergence rates.

Non-Convex: Under assumptions 2, 3 and 4, and letting $T$ be large enough such that $T = \Omega\left(\max \left\{\frac{L^2(dn_0 + n_1)^2}{dn^2}, n^2 L^2, nn_0^3 d\right\}\right)$ , we have

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} = \sqrt {\frac {d}{T}} \times O \left(\left(f (\mu_ {0}) - f ^ {*}\right) + \right.
$$

$$
L \big (\frac {d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}}{n d} \big) + L \big (\frac {d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2}}{n d} \big) + L ^ {2} \sqrt {\frac {n _ {0}}{n}} \Bigg).
$$

Adding, convexity (Assumption 1), we prove the following:

Strongly Convex: If we assume that the functions $f$ and $f_{i}$ satisfy Assumptions 1, 2, 3 and 4, and let $T$ be large enough such that $\frac{T}{\log T} = \Omega \left( \frac{n(d + n)(L + 1)\left(\frac{1}{\ell} + 1\right)}{\ell} \right)$ , and let the learning rate be $\eta = \frac{4n\log T}{T\ell}$ . For $1 \leq t \leq T$ , let the sequence of weights $w_{t}$ be given by $w_{t} = \left(1 - \frac{\eta\ell}{2n}\right)^{-t}$ and let $S_{T} = \sum_{t=1}^{T}w_{T}$ . Finally, define $y_{T} = \sum_{t=1}^{T}\frac{w_{t}\mu_{t-1}}{S_{T}}$ to be the mean over local model parameters. Then, we can show that HDO provides the following convergence rate:

$$
\begin{array}{l} \mathbb {E} \left[ f \left(y _ {T}\right) - f \left(x ^ {*}\right) \right] + \frac {\ell \mathbb {E} \| \mu_ {T} - x ^ {*} \| ^ {2}}{8} \\ = O \left(\frac {L \| \mu_ {0} - x ^ {*} \| ^ {2}}{T \log T} + \frac {\log (T) \left(d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}\right)}{T \ell n} \right. \\ \left. + \frac {\log (T) (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{T \ell n} + \frac {\log (T) d n _ {0}}{T \ell n}\right). \\ \end{array}
$$

Speedup. Here, the time T refers to the total number of interactions among agents, as opposed to parallel time, corresponding to the average number of interactions T/n. These rates are reminiscent of sequential SGD. However, there are some distinctions: we are counting the total number of gradient oracle queries by the nodes, and there are some additional trailing terms, whose meanings we discuss below.

We interpret this formula from the perspective of an arbitrary local model. For this, notice that the notion of parallel time corresponding to the number of total interactions T, which is by definition $T_{p} = T/n$ , corresponds (up to constants) to the average number of interactions and gradient oracle queries performed by each node up to time T. Therefore, in the strongly convex setup, for any single model, convergence with respect to its number of performed SGD steps $T_{p}$ would be $O(\log(nT_{p})/(nT_{p}))$ (assuming all parameters are constant), which would correspond to $\Omega\left(\frac{n}{\log(nT_{p})}\right) = \Omega\left(\frac{n}{\log(T)}\right)$ speedup compared to a variant of sequential SGD. Notice that this is quite favorable to our algorithm, since we are considering biased zeroth-order estimators for some of the nodes in the population. Hence, assuming that T is polynomial in n, we get an almost-linear speedup of $\Omega\left(\frac{n}{\log(n)}\right)$ . Similarly, in the non-convex case, we get a speedup $\Omega(\sqrt{n})$ , which shows the scalability of our algorithm.

Impact of Zeroth-Order Nodes. Notice that our convergence bounds cleanly separate in the terms which come from zeroth-order nodes and terms which come from first-order nodes. For $n_{0} = 0$ , we get asymptotically the same bound as we would get if all nodes performed pure first-order SGD steps. Similarly, when $n_{0} = n$ we should be able to achieve asymptotically-optimal convergence for biased zeroth-order estimators. Further, notice that, if the bias is negligible, then the last term in each of the upper bounds disappears, and we obtain a trade-off between two populations with different variances. We can also observe the following theoretical threshold: we asymptotically match the convergence rate in the case with all nodes performing SGD steps, as long as $dn_{0} = O(n)$ (assuming all other parameters are constant).

# Analysis

As an example, we discuss the proof overview for the strongly convex case. The notations and proof steps are closely aligned with those used in the non-convex case.

Proof Overview. The convergence proof, given in full in the Appendix, can be split conceptually into two steps. The first aims to bound the variance of the local models $X_{t}^{i}$ for each time step t and node i with respect to the mean $\mu_{t} = \sum_{i} X_{t}^{i}$ . It views this variance as a potential $\Gamma_{t}$ , which as we show has supermartingale-like behavior for small enough learning rate: specifically, this quantity tends to increase due to gradient steps, but is pushed towards the mean $\mu_{t}$ by the averaging process.

The key component here is Lemma 2, which carefully bounds the evolution of the potential at a step, by modeling optimization as a dynamic load balancing process: each interaction corresponds to a weight generation step (in which gradient estimators are generated) and a load balancing step, in which the “loads” of the two nodes (corresponding to their model values) are balanced through averaging.

In the second step, we first bound the rate at which the mean $\mu_{t}$ converges towards $x^{*}$ , where we crucially (and carefully) leverage the variance bound obtained above. The main challenge in this part is dealing with biased zeroth-order estimators. In fact, even dealing with biased first-order estimators is not trivial, since for example, they are the main reason for the usage of error feedback when stochastic gradients are compressed using biased quantization (Alistarh et al. 2018). This is our second key technical result.

With this in hand, we can complete the proof by applying a standard argument which characterizes the rate at which $\mathbb{E}[f(y_{T}) - f(x^{*})]$ and $E[\|\mu_{t} - x^{*}\|^{2}$ converge towards 0.

Notation and Preliminaries. In this section, we provide a more in-depth sketch of the analysis of the HDO protocol. We begin with some notation. Recall that n is the number of nodes, split into first-order $(n_{1})$ and zeroth-order $(n_{0})$ . We will analyze a sequence of time steps $t = 1, 2, \ldots, T$ , each corresponding to an individual interaction between two nodes, which are usually denoted by i and j.

Step 1: Parameter Concentration. Next, let $X_{t}$ be a vector of model estimates at time step t, that is $X_{t} = (X_{t}^{1}, X_{t}^{2}, ..., X_{t}^{n})$ . Also, let $\mu_{t} = \frac{1}{n} \sum_{i=1}^{n} X_{t}^{i}$ , be an average estimate at time step t. The following potential function measures the variance of the models:

$$
\Gamma_ {t} = \frac {1}{n} \sum_ {i = 1} ^ {n} \| X _ {t} ^ {i} - \mu_ {t} \| ^ {2}.
$$

With this in place, one of our key technical results is to provide a supermartingale-type bound on the evolution of the potential $\Gamma_{t}$ , in terms $\eta$ , and average second moment of estimators at step t, defined as $M_{t}^{G} := \frac{1}{n} \sum_{i} \left\| G^{i}(X_{t}^{i}) \right\|^{2}$ .

Lemma 2. For any time step $t$ :

$$
\mathbb {E} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {4}{n} \eta^ {2} \mathbb {E} \left[ M _ {t} ^ {G} \right].
$$

Notice that, if we had a universal second moment bound on the estimators, that is, for any vector X and node i $\mathbb{E}\left\|G^{i}(X)\right\|^{2}\leq M$ , for some M>0, then we would be able to unroll the recursion, and, for any $t\geq0$ upper bound $E[\Gamma_{t}]$ by $\eta^{2}M^{2}$ . In the absence of such upper bound we must derive the following upper bound on $E\left[M_{t}^{G}\right]$ :

Lemma 3. Assume $\nu := \frac{\eta}{c}$ is fixed, where $\eta$ and c are the learning rate and a constant respectively. Then, for any time step t we have:

$$
\begin{array}{l} \mathbb {E} \left[ M _ {t} ^ {G} \right] \leq 6 (d + 4) L ^ {2} \mathbb {E} [ \Gamma_ {t} ] + \frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} \\ + 6 (2 d + 9) L \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ] \\ + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} \\ + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}. \\ \end{array}
$$

First, we check how this upper bound affects the upper bound given by Lemma 2. For small enough $\eta$ , the term containing $E[\Gamma_{t}]$ (which comes from the upper bound on $E\left[M_{t}^{G}\right]$ ) can be upper bounded by $\frac{1}{4n}E[\Gamma_{t}]$ , and hence it will just change the factor in front of $E[\Gamma_{t}]$ to $(1-1/4n)$ .

Second, since the above bound contains the term with $\mathbb{E}[f(\mu_{t}) - f(x^{*})]$ we are not able to bound the potential

$\Gamma$ per step, instead, for weights $w_{t} = (1 - \frac{\eta\ell}{2n})^{-t}$ , we can upper bound $\sum_{t=1}^{T} w_{t} \mathbb{E}[\Gamma_{t-1}]$ (please see Lemma 12 in the Appendix). The crucial property is that the upper bound on the weighted sum $\sum_{t=1}^{T} w_{t} \mathbb{E}[\Gamma_{t-1}]$ , is

$$
O (\eta^ {2} \sum_ {t = 1} ^ {T} w _ {t} \mathbb {E} [ f (\mu_ {t - 1}) - f (x ^ {*}) ]) + \sum_ {t = 1} ^ {T} w _ {t} O (\eta^ {2}).
$$

(for simplicity, above we assumed that all other parameters are constant.)

Step 2: Convergence of the Mean and Risk Bound. The above result allows us to characterize how well the individual parameters are concentrated around their mean. In turn, this will allow us to provide a recurrence for how fast the parameter average is moving towards the optimum. To help with the intuition, we provide the lemma which is simplified version of the one given in the additional material:

Lemma 4. For small enough $\eta$ and $t \geq 1$ we have that:

$$
\begin{array}{l} \mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\| ^ {2} \leq (1 - \frac {\ell \eta}{2 n}) \mathbb {E} \left\| \mu_ {t - 1} - x ^ {*} \right\| ^ {2} \\ - \Omega (\frac {\eta}{n}) \mathbb {E} \left[ f (\mu_ {t - 1}) - f (x ^ {*}) \right] \\ + O \left(\frac {\eta}{n}\right) \mathbb {E} [ \Gamma_ {t - 1} ] + O (\frac {\eta^ {2}}{n ^ {2}}). \\ \end{array}
$$

Note that O and $\Omega$ hide all other parameters (we assume that all other parameters are constant). As mentioned, the main challenge in the proof of this lemma is taking care of biased zeroth-order estimators.

Recall that $w_{t} = (1 - \frac{\eta\ell}{2n})^{-t}$ , by definition. We proceed by multiplying both sides of the above inequality by $w_{t}$ and then summing it up for $1 \leq t \leq T$ . Then, once we plug the upper bound on $\sum_{t=1}^{T} w_{t} \mathbb{E}[\Gamma_{t-1}]$ , for small enough $\eta$ the term $O(\frac{\eta}{n})O(\eta^2 \sum_{t=1}^{T} w_{t} \mathbb{E}[f(\mu_{t-1} - f(x^{*})])$ vanishes as it is dominated by the term $-\sum_{t=1}^{T} \Omega(\frac{\eta}{n}) \mathbb{E}[f(\mu_{t-1}) - f(x^{*})]$ .

We get the final convergence bound after some simple calculations involving division of both sides by $S_{T} = \sum_{t=1}^{T} w_{t}$ , and using $\eta = \frac{4n\log(T)}{T\ell}$ together with the upper bound on T (in turn, this makes sure that $\eta$ is small enough, so that all upper bounds we mentioned hold).

# Experimental Results

Experimental Setup and Goals. In this section, we validate our results by simulating the HDO algorithm under different conditions, including varying the number of nodes, the ratios between first-order (FO) and zeroth-order (ZO) nodes, and the "strength" of the zeroth-order gradient estimators. Our focus is on the algorithm's convergence behavior, relative to the total number of optimization steps, which we measure by tracking the loss over time or the accuracy on the hidden validation set. Our goal is to determine whether a hybrid system, combining both FO and ZO agents, can achieve better convergence compared to a system that relies on a single type of agent. Additionally, we aim to demonstrate the convergence speed, in terms of training loss at a fixed node, for different sizes of mono-type populations, each consisting of only one type of estimator.

![](images/2d74ac61113543404656645996c214c0960cb6c9f9a4855148d7e58ce2d65f7e.jpg)

<details>
<summary>line</summary>

| Step | 1 biased ZO, rv=8 | 1 biased ZO, rv=16 | 1 unbiased ZO, rv=8 | 1 unbiased ZO, rv=16 | 1 unbiased ZO, rv=128 | 1 unbiased ZO, rv=128 | 1 FO |
|------|-------------------|--------------------|---------------------|----------------------|-----------------------|-----------------------|------|
| 0    | 0.7               | 0.75               | 0.7                 | 0.75                 | 0.75                  | 0.75                  | 0.95 |
| 400  | 0.8               | 0.85               | 0.8                 | 0.85                 | 0.85                  | 0.85                  | 0.95 |
| 600  | 0.85              | 0.9                | 0.85                | 0.9                  | 0.9                   | 0.9                   | 0.95 |
| 800  | 0.85              | 0.9                | 0.85                | 0.9                  | 0.9                   | 0.9                   | 0.95 |
| 1k   | 0.85              | 0.9                | 0.85                | 0.9                  | 0.9                   | 0.9                   | 0.95 |
</details>

Figure 1: Number of random vectors (rv) impact on the biased/unbiased ZO estimators Accuracy (Acc), using a CNN model on MNIST.

![](images/632e2d1e7c82877335c8e426af2d4124f3aecd97cba13a8656233f99ddd4c757.jpg)

<details>
<summary>line</summary>

| Step | 1 ZO | 1 FO | 16 ZO | 6 FO | 64 ZO | 12 FO | 128 ZO | 24 FO | 256 ZO | 24 FO 256 ZO |
| ---- | ---- | ---- | ----- | ---- | ----- | ----- | ------ | ----- | ------ | ------------ |
| 0    | 2.35 | 2.35 | 2.35  | 2.35 | 2.35  | 2.35  | 2.35   | 2.35  | 2.35   | 2.35         |
| 100  | 2.30 | 2.28 | 2.25  | 2.20 | 2.18  | 2.15  | 2.10   | 2.15  | 2.10   | 2.08         |
| 200  | 2.28 | 2.25 | 2.20  | 2.15 | 2.12  | 2.10  | 2.08   | 2.10  | 2.08   | 2.05         |
| 300  | 2.27 | 2.23 | 2.18  | 2.13 | 2.10  | 2.08  | 2.06   | 2.08  | 2.06   | 2.03         |
| 400  | 2.27 | 2.21 | 2.17  | 2.12 | 2.09  | 2.07  | 2.05   | 2.07  | 2.05   | 2.02         |
| 450  | 2.27 | 2.19 | 2.16  | 2.11 | 2.08  | 2.06  | 2.04   | 2.06  | 2.04   | 2.01         |
</details>

Figure 2: Validation loss vs. various population configurations for regression model on MNIST.

In the implementation of HDO, each step involves nodes updating their models, followed by the formation of $O(n)$ random disjoint pairs. Each pair exchanges their models and replaces their own with the averaged model. To demonstrate the effectiveness of zeroth-order nodes under reasonable parameter settings, we evaluate the mean validation loss and accuracy across all nodes and analyze the consensus of the models by measuring the standard deviation of losses. Additional details on the models, datasets, and the complete experimental setup are provided in the Appendix.

Results. At first, we examine the performance of individual zeroth-order gradient estimators (with $\nu = 10^{-4}$ ) over time, as a function of the number of random vectors (rv) used for the gradient estimation (Figure 1); that is the number of u's used to estimate the gradient using the equation 4. To do so, we use the MNIST classification task (Deng 2012) using a Convolutional Neural Network (CNN) (Lecun et al. 1998) model. We choose values 8, 16, and 128 for the number of random vectors, and compare against the unbi-

![](images/07cf9e635d919c0715d19acb12382f6158af55a30621207c5609a0cab067e22c.jpg)

<details>
<summary>line</summary>

| Step | 1 ZO  | 1 FO  | 5 ZO  | 1 FO 5 ZO |
|------|-------|-------|-------|-----------|
| 200  | 0.70  | 0.70  | 0.80  | 0.79      |
| 400  | 0.73  | 0.78  | 0.86  | 0.86      |
| 600  | 0.75  | 0.82  | 0.88  | 0.88      |
| 800  | 0.76  | 0.85  | 0.89  | 0.89      |
| 1k   | 0.77  | 0.86  | 0.89  | 0.89      |
</details>

Figure 3: Validation accuracy (Acc) comparison between the hybrid and mono-type estimator population for ResNet-18 on the CIFAR-10 dataset.

![](images/03b91ffe750978c8bb0db2793724f1ccaca1218c420f0312da0bcf1007b82d38.jpg)

<details>
<summary>line</summary>

| Step | 1 ZO | 16 ZO | 4 FO 16 ZO | 4 FO | 1 FO |
|------|------|-------|------------|------|------|
| 0    | 0.5  | 0.5   | 0.5        | 0.5  | 0.5  |
| 200  | 0.35 | 0.3   | 0.3        | 0.3  | 0.35 |
| 400  | 0.3  | 0.25  | 0.25       | 0.25 | 0.3  |
| 600  | 0.3  | 0.25  | 0.2        | 0.25 | 0.25 |
| 800  | 0.3  | 0.25  | 0.2        | 0.25 | 0.25 |
| 1k   | 0.3  | 0.25  | 0.2        | 0.25 | 0.25 |
</details>

Figure 4: Validation loss comparison between the hybrid and mono-type estimator population for transformer model on the synthetic Brackets dataset.

ased forward-only estimator recently proposed by (Baydin et al. 2022). The results clearly demonstrate an accuracy-versus-steps advantage for a higher number of random vectors and for unbiased zeroth-order estimators compared to biased ones. Since the computational overhead of unbiasing estimators is relatively low (Chen 2020), we will use unbiased zeroth-order estimators in the subsequent experiments. A detailed explanation and experimental evaluation of the accuracy-efficiency trade-offs associated with the number of random vectors will be provided in the Appendix.

For the next set of experiments, we evaluate different mono-type populations of ZO and FO optimizers and compare their performance with that of a hybrid population. In Figure 3, we fine-tune the ResNet-18 (He et al. 2015) model, pre-trained with ImageNet-1K (Russakovsky et al. 2015), on CIFAR-10 (Krizhevsky 2012) using populations of 1 ZO, 1 FO, 5 ZO, and a hybrid system consisting of 1 FO and 5 ZO nodes. To demonstrate the scalability of HDO in both convex and non-convex scenarios, we consider two settings: for the convex case, we assess the performance of a Logistic model on the MNIST dataset with various mono-type populations and a hybrid configuration of 24 FO and 256 ZO (Figure 2). For the non-convex case, we use a Transformer model (Vaswani et al. 2023) on “Brackets” dataset (Ebrahimi, Gelda, and Zhang 2020) (see the Appendix for more details). The populations that we study here are 1 ZO, 1 FO, 4 FO, 16 ZO, and a hybrid combination of 4 FO and 16 ZO (Figure 4).

The results presented in Figure 2 (for the convex case), and Figures 3 & 4 (for the non-convex case) confirm the intuition, as well as our analysis; first-order nodes always outperform the same or lower number of zeroth-order ones, as shown in all the figures. However, zeroth-order nodes can in fact outperform first-order ones if their number is larger. We can conclude that 1) a larger population of ZO nodes can outperform smaller populations of FO or ZO nodes; and that 2) this larger uniform population is itself outperformed by a hybrid population.

Our experiments demonstrate the desired speedup and scalability; populations with a large number of ZO nodes have faster convergence. Interestingly, aligned with the Theorem 1 that the speedup caused by the ZO nodes will appear after sufficiently large T, in Figure 2, we observe that the (24-FO) group has a lower validation loss initially but the (256-ZO) and (24-FO, 256-ZO) groups outperform the former group after step 200. A similar phenomenon can also be observed in Figure 4. These results validate the theoretical finding, showing that with a sufficient number of steps, hybrid populations achieve faster convergence compared to homogeneous populations of first-order optimizers.

# Discussion, Limitations, and Future Work

We provided a first analysis of the convergence of decentralized gradient-based methods in a population mixing first-and zeroth-order gradient estimators for both convex and non-convex objectives. Our results show that even biased or noisy zeroth-order information can enhance convergence when integrated into a protocol.

The experimental results validate our analysis and premise, demonstrating that first- and zeroth-order estimators can be effectively hybridized in a decentralized population. This is promising for environments with heterogeneous computational power, allowing agents to leverage local data even without gradient extraction capabilities.

A practical embodiment could be a decentralized learning system where computationally powerful agents perform backpropagation as first-order agents, while computationally limited nodes estimate gradients via forward passes over local data, sharing this information during pairwise interactions.

We focused on decentralized optimization, where nodes interact in randomly chosen pairs. However, our analysis can extend to more general interaction graph topologies, where convergence depends on the eigenvalue gap. Another extension we plan to explore includes additional gradient estimators and large-scale practical deployments to validate our approach. Our experimental simulations confirm the feasibility of our method, achieving their intended goal.

# Acknowledgements

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No 805223 ScaleML). The authors would like to acknowledge Eugenia Iofinova for useful discussions during the inception of this project.

# References

Agarwal, A.; Dekel, O.; and Xiao, L. 2010. Optimal Algorithms for Online Convex Optimization with Multi-Point Bandit Feedback. In Colt, 28–40. Citeseer.   
Alistarh, D.; Hoefler, T.; Johansson, M.; Konstantinov, N.; Khirirat, S.; and Renggli, C. 2018. The convergence of sparsified gradient methods. In NIPS, 5977–5987.   
Angluin, D.; Aspnes, J.; Diamadi, Z.; Fischer, M. J.; and Peralta, R. 2006. Computation in networks of passively mobile finite-state sensors. Distributed computing, 18(4): 235–253.   
Assran, M.; Loizou, N.; Ballas, N.; and Rabbat, M. 2018. Stochastic gradient push for distributed deep learning. arXiv preprint arXiv:1811.10792.   
Balasubramanian, K.; and Ghadimi, S. 2022. Zeroth-Order Nonconvex Stochastic Optimization: Handling Constraints, High Dimensionality, and Saddle Points. Found. Comput. Math., 22(1): 35–76.   
Baydin, A. G.; Pearlmutter, B. A.; Syme, D.; Wood, F.; and Torr, P. 2022. Gradients without Backpropagation.   
Bottou, L. 2010. Large-scale machine learning with stochastic gradient descent. In Proceedings of COMPSTAT'2010, 177–186. Springer.   
Chen, G. 2020. Unbiased Gradient Simulation for Zeroth-Order Optimization. In 2020 Winter Simulation Conference (WSC), 2947–2959.   
Deng, L. 2012. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6): 141–142.   
Ebrahimi, J.; Gelda, D.; and Zhang, W. 2020. How Can Self-Attention Networks Recognize Dyck-n Languages? arXiv:2010.04303.   
Flaxman, A. D.; Kalai, A. T.; and McMahan, H. B. 2004. Online convex optimization in the bandit setting: gradient descent without a gradient. arXiv preprint cs/0408007.   
Gabriel, E.; Fagg, G. E.; Bosilca, G.; Angskun, T.; Dongarra, J. J.; Squyres, J. M.; Sahay, V.; Kambadur, P.; Barrett, B.; Lumsdaine, A.; Castain, R. H.; Daniel, D. J.; Graham, R. L.; and Woodall, T. S. 2004. Open MPI: Goals, Concept, and Design of a Next Generation MPI Implementation. In Proceedings, 11th European PVM/MPI Users' Group Meeting, 97–104. Budapest, Hungary.   
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2015. Deep Residual Learning for Image Recognition. arXiv:1512.03385.   
Iutzeler, F.; Bianchi, P.; Ciblat, P.; and Hachem, W. 2013. Asynchronous distributed optimization using a randomized alternating direction method of multipliers. In 52nd IEEE conference on decision and control, 3671–3676. IEEE.

Johansson, B.; Rabi, M.; and Johansson, M. 2009. A randomized incremental subgradient method for distributed optimization in networked systems. SIAM Journal on Optimization, 20(3): 1157–1170.   
Jongeneel, W.; Yue, M.-C.; and Kuhn, D. 2021. Small errors in random zeroth-order optimization are imaginary.   
Kempe, D.; Dobra, A.; and Gehrke, J. 2003. Gossip-based computation of aggregate information. In 44th Annual IEEE Symposium on Foundations of Computer Science, 2003. Proceedings., 482–491. IEEE.   
Koloskova, A.; Lin, T.; Stich, S. U.; and Jaggi, M. 2020a. Decentralized Deep Learning with Arbitrary Communication Compression. In International Conference on Learning Representations.   
Koloskova, A.; Loizou, N.; Boreiri, S.; Jaggi, M.; and Stich, S. U. 2020b. A Unified Theory of Decentralized SGD with Changing Topology and Local Updates. In ICML, 5381–5393.   
Krizhevsky, A. 2012. Learning Multiple Layers of Features from Tiny Images. University of Toronto.   
Krizhevsky, A.; Nair, V.; and Hinton, G. 2010. CIFAR-10 (Canadian Institute for Advanced Research). Available Online.   
Lecun, Y.; Bottou, L.; Bengio, Y.; and Haffner, P. 1998. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11): 2278–2324.   
Lian, X.; Zhang, C.; Zhang, H.; Hsieh, C.-J.; Zhang, W.; and Liu, J. 2017a. Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent. arXiv preprint arXiv:1705.09056.   
Lian, X.; Zhang, W.; Zhang, C.; and Liu, J. 2017b. Asynchronous decentralized parallel stochastic gradient descent. arXiv preprint arXiv:1710.06952.   
Loshchilov, I.; and Hutter, F. 2017. SGDR: Stochastic Gradient Descent with Warm Restarts. arXiv:1608.03983.   
Mhanna, E.; and Assaad, M. 2023. Single Point-Based Distributed Zeroth-Order Optimization with a Non-Convex Stochastic Objective Function. In Krause, A.; Brunskill, E.; Cho, K.; Engelhardt, B.; Sabato, S.; and Scarlett, J., eds., Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, 24701–24719. PMLR.   
Nadiradze, G.; Sabour, A.; Davies, P.; Li, S.; and Alistarh, D. 2021. Asynchronous decentralized SGD with quantized and local updates. Advances in Neural Information Processing Systems, 34.   
Nair, V.; and Hinton, G. E. 2010. Rectified Linear Units Improve Restricted Boltzmann Machines. In International Conference on Machine Learning.   
Nedic, A.; and Ozdaglar, A. 2009. Distributed subgradient methods for multi-agent optimization. IEEE Transactions on Automatic Control, 54(1): 48.   
Nesterov, Y.; and Spokoiny, V. 2017. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17(2): 527–566.

Russakovsky, O.; Deng, J.; Su, H.; Krause, J.; Satheesh, S.; Ma, S.; Huang, Z.; Karpathy, A.; Khosla, A.; Bernstein, M.; Berg, A. C.; and Fei-Fei, L. 2015. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3): 211–252.   
Sahu, A. K.; and Kar, S. 2020. Decentralized zeroth-order constrained stochastic optimization algorithms: Frank-Wolfe and variants with applications to black-box adversarial attacks. Proceedings of the IEEE, 108(11): 1890–1905.   
Shamir, O. 2017. An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. The Journal of Machine Learning Research, 18(1): 1703–1713.   
Shamir, O.; and Srebro, N. 2014. Distributed stochastic optimization and learning. In 2014 52nd Annual Allerton Conference on Communication, Control, and Computing (Allerton), 850–857. IEEE.   
Tang, H.; Zhang, C.; Gan, S.; Zhang, T.; and Liu, J. 2018. Decentralization meets quantization. CoRR, abs/1803.06443.   
Tsitsiklis, J. N. 1984. Problems in decentralized decision making and computation. Technical report, Massachusetts Inst of Tech Cambridge Lab for Information and Decision Systems.   
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2023. Attention Is All You Need. arXiv:1706.03762.   
Wang, J.; and Joshi, G. 2021. Cooperative SGD: A Unified Framework for the Design and Analysis of Local-Update SGD Algorithms. Journal of Machine Learning Research, 22(213): 1–50.   
Wei, E.; and Ozdaglar, A. 2012. Distributed alternating direction method of multipliers. In 2012 IEEE 51st IEEE Conference on Decision and Control (CDC), 5445–5450. IEEE.   
Xiao, L.; and Boyd, S. 2004. Fast linear iterations for distributed averaging. Systems & Control Letters, 53(1): 65–78.   
Yuan, D.; Wang, L.; Proutiere, A.; and Shi, G. 2024. Distributed zeroth-order optimization: Convergence rates that match centralized counterpart. Automatica, 159: 111328.

# Appendix

# Experimental Setup

In this section, we describe our experimental setup in detail. We begin by carefully describing the way in which we simulated HybridSGD in the sequential form. Then, we proceed by explaining the different types of gradient estimators that we used in our experiments, together with their implementation methods. Finally, we detail the datasets, tasks, and models used for our experiments.

# Simulation

We attempt to simulate a realistic decentralized deployment scenario sequentially, as follows. We assume n nodes, each of which initially has a model copy. Each node has an oracle to estimate the gradient of the loss function with respect to its model. It is assumed that $n_{1}$ nodes have access to first-order oracle and $n_{0}$ nodes have access to zeroth-order oracle, which can be biased or unbiased. The training dataset is distributed among first- and zeroth-order nodes so that each node has access to $\frac{1}{n}$ of training data. At each simulation step, we select $O(n)$ disjoint pairs uniformly at random and make each pair interact with each other. During the interaction, first, each node takes an SGD step and then they share their models and adapt the averaged model as their new model. To track the performance of our algorithm, each node evaluates its model on an unseen validation dataset which is shared between all the nodes. After every 10 steps, each node computes the validation loss and accuracy, and then the averaged loss and accuracy from all the nodes will be reported.

# Estimator types

First-order Using this estimator node $i$ can estimate the gradient $\nabla f^i (X^i)$ by computing $\nabla F^i (X^i,\xi^i)$ , where $f^i$ and $F^i$ are the node's local loss function and its stochastic estimator respectively. The computation is done using Pytorch built-in .backward() method.

Unbiased Zeroth-order We implemented this estimator using forward-mode differentiation technique inspired by (). Using this method, for a randomly chosen vector $u \sim N(0, I_d)$ , node $i$ can compute $F^i(X^i, \xi^i)$ and $u. \nabla F^i(X^i, \xi^i)$ in a single forward pass. It will then use $(u. \nabla F^i(X^i, \xi^i))u$ as its gradient estimator. Note that the node does not need to compute $\nabla F^i(X^i, \xi^i)$ , hence it is a zeroth-order estimation of the gradient. Moreover, $E_{u \sim N(0, I_d)}[(u. \nabla F^i(X^i, \xi^i))u] = \nabla F^i(X^i, \xi^i)$ which means the gradient estimator is unbiased.

Biased Zeroth-order For a fixed $\nu$ and a randomly chosen $u \sim N(0, I_{d})$ , node i can estimate the gradient $\nabla F^{i}(X^{i}, \xi^{i})$ simply by computing $\frac{F^{i}(X^{i} + \nu u, \xi^{i}) - F^{i}(X^{i}, \xi^{i})}{\nu} u$ or $\frac{F^{i}(X^{i} + \nu u, \xi^{i}) - F^{i}(X^{i} - \nu u, \xi^{i})}{2\nu} u$ . The computations consist of evaluating only function values, thus they are called zeroth-order estimators. However, both of them are biased estimators as their expected values would be equal to the gradient of the smoothed-version of the function, $\nabla F_{\nu}^{i}(X^{i}, \xi^{i})$ , which is close but not necessarily equal to $\nabla F^{i}(X^{i}, \xi)$ .

Note that the approximation of zeroth-order estimators can be improved by increasing the number of randomly chosen vectors and averaging the results. For the biased zeroth-order estimators, we use the batch matrix multiplication to compute the function values for all the randomly chosen vectors using constant GPU calls. Moreover, for unbiased zeroth-order estimators, we simulate the forward-mode differentiation by computing the gradient followed by computing the dot products of the gradient and the randomly chosen vectors. For more details on the implementation, we encourage readers to look at the source code of our experiments.

# Datasets and Models

We use Pytorch to manage the training process in our algorithm, as well as Open MPI (Gabriel et al. 2004) for communication between the nodes. In the first steps, we do some warm-up steps in a way that each node just trains its model without communication. We use a linear scheduler to increase the learning rate to the desired value during these steps. Then they start to communicate and we use a Cosine Annealing (CA) scheduler (Loshchilov and Hutter 2017) to make more stable training between the nodes. We did the experiments in different random seeds and for each step, we computed the mean and standard error for the final evaluation. We use the Cross-Entropy loss function in our implementation. Since each step is training on a single batch, to decrease the noise of the computed gradient, we use a gradient momentum $g_{t + 1} = mg_t + (1 - m)\nabla G^i (x)$ where $m$ is the momentum value and $g_{t}$ is the gradient to update the model at step $t$ . We fine-tuned the hyperparameters using grid search as much as possible. To be mentioned, we individually optimized the learning rates for a single agent of each node type, FO and ZO, to ensure optimal performance for both (recognizing that optimal rates are type-specific due to differences in gradient estimations). You can find the details of hyperparameter tuning in the tables below. Complete search was not possible because of the large search space. In the next paragraphs, we explain some task-specific details.

CNN on MNIST (Table 1). The CNN model used in this study consists of three convolutional and three linear layers. ReLU activation (Nair and Hinton 2010) is applied throughout, with the convolutional layers having an output channel size of 8 and the linear layers having a width of 128. The experiments were conducted on a single A10 GPU with 24 GB of VRAM and 32 GB of system RAM.

ResNet-18 on CIFAR-10 (Table 2). ZOs have a lower memory footprint, which allows us to use a larger batch size. Since they process more data points simultaneously, it is more appropriate to consider a larger learning rate for them in certain regimes. This advantage of using a larger batch size is evident in theory, as it results in lower gradient variance for ZOs. Consequently, it would not be realistic to use the same batch size and learning rate for ZOs as for FOs, given their higher variance. By leveraging the lower memory footprint of ZOs to use larger batches, we are aligning the training process more closely with real-world scenarios and reducing variance. Failing to do so would mean not fully capitalizing on the benefits that ZOs offer. Therefore, after fine-tuning, we ended up using different learning rates and batch sizes for ZO and FO nodes. We conducted this setup on a single A10 GPU with 24 GB VRAM and 100 GB RAM.

Regression model on MNIST (Table 3). We evaluated this scenario using 96 CPU cores and 4 RTX 3090 GPUs, each with 24 GB of VRAM. To maximize the number of nodes within our resource constraints, we used a single seed for this linear model. To better assess the behavior of ZO and FO nodes at scale, we set the minibatch size to 2. This choice prevents the models from seeing all the data in the initial steps, thereby avoiding premature convergence to the optimum point. Additionally, we omitted the use of gradient momentum and the CA scheduler to obtain a more realistic observation of the model's performance.

Transformer model on Brackets (Table 4). For this task, we employ a Transformer model with 2 layers of multi-head self-attention, each with 2 heads, and an embedding size of 4. A dropout rate of 0.1 is applied to mitigate overfitting. The experiments were conducted on a single Tesla T4 GPU with 15 GB of VRAM and 51 GB of system RAM.

Brackets. This dataset consists of sequences of opening '(' and closing brackets ')''. The task is to predict the correctness of the entire sequence in terms of bracketing. A sequence is defined as correct if every opening bracket has a corresponding closing bracket. The training set consists of 25,600 samples, and the validation set consists of 2,560 samples. We use this dataset because the correct bracket sequences form a context-free language, which captures some properties of natural language (Ebrahimi, Gelda, and Zhang 2020). Therefore, it can reveal nontrivial model capabilities while still being a relatively simple dataset.

<table><tr><td>Hyperparameter</td><td>Value</td><td>Search interval</td></tr><tr><td>FO batch size</td><td>256</td><td>{128, 256, 512}</td></tr><tr><td>ZO batch size</td><td>256</td><td>{128, 256, 512}</td></tr><tr><td>FO learning rate</td><td>0.01</td><td>[0.001, 0.1]</td></tr><tr><td>ZO learning rate</td><td>0.01</td><td>[0.001, 0.1]</td></tr><tr><td>FO momentum</td><td>0.9</td><td>-</td></tr><tr><td>ZO momentum</td><td>0.9</td><td>-</td></tr><tr><td>T</td><td>1000</td><td>-</td></tr><tr><td>Number of seeds</td><td>3</td><td>-</td></tr><tr><td>Warm-up steps</td><td>50</td><td>-</td></tr><tr><td>CA scheduler</td><td>Yes</td><td>-</td></tr><tr><td>rv</td><td>-</td><td>-</td></tr></table>

Table 1: Hyperparameters-tuning details for studying the impact of the number of random vectors on the biased/un-biased ZO estimators, conducted using a CNN model on MNIST. 

<table><tr><td>Hyperparameter</td><td>Value</td><td>Search interval</td></tr><tr><td>FO batch size</td><td>2</td><td>-</td></tr><tr><td>ZO batch size</td><td>2</td><td>-</td></tr><tr><td>FO learning rate</td><td>0.01</td><td>[0.001, 0.1]</td></tr><tr><td>ZO learning rate</td><td>0.01</td><td>[0.001, 0.1]</td></tr><tr><td>FO momentum</td><td>-</td><td>-</td></tr><tr><td>ZO momentum</td><td>-</td><td>-</td></tr><tr><td>T</td><td>500</td><td>-</td></tr><tr><td>Number of seeds</td><td>1</td><td>-</td></tr><tr><td>Warm-up steps</td><td>0</td><td>-</td></tr><tr><td>CA scheduler</td><td>No</td><td>-</td></tr><tr><td>rv</td><td>128</td><td>{32, 64, 128}</td></tr></table>

Table 3: Hyperparameters-tuning details for the linear regression model on MNIST.

<table><tr><td>Hyperparameter</td><td>Value</td><td>Search interval</td></tr><tr><td>FO batch size</td><td>10</td><td>[10, 100]</td></tr><tr><td>ZO batch size</td><td>50</td><td>[50, 250]</td></tr><tr><td>FO learning rate</td><td>0.001</td><td>[0.0001, 0.1]</td></tr><tr><td>ZO learning rate</td><td>0.01</td><td>[0.0001, 0.1]</td></tr><tr><td>FO momentum</td><td>0.9</td><td>-</td></tr><tr><td>ZO momentum</td><td>0.9</td><td>-</td></tr><tr><td>T</td><td>1000</td><td>-</td></tr><tr><td>Number of seeds</td><td>3</td><td>-</td></tr><tr><td>Warm-up steps</td><td>50</td><td>-</td></tr><tr><td>CA scheduler</td><td>Yes</td><td>-</td></tr><tr><td>rv</td><td>128</td><td>{16, 32, 64, 128}</td></tr></table>

Table 2: Hyperparameters-tuning details for the ResNet-18 model on CIFAR-10.

<table><tr><td>Hyperparameter</td><td>Value</td><td>Search interval</td></tr><tr><td>FO batch size</td><td>128</td><td>{128, 256, 1024, 2048}</td></tr><tr><td>ZO batch size</td><td>256</td><td>{128, 256, 1024, 2048}</td></tr><tr><td>FO learning rate</td><td>0.05</td><td>[0.0001, 0.1]</td></tr><tr><td>ZO learning rate</td><td>0.1</td><td>[0.0001, 0.1]</td></tr><tr><td>FO momentum</td><td>0.8</td><td>[0.0, 0.95]</td></tr><tr><td>ZO momentum</td><td>0.8</td><td>[0.0, 0.95]</td></tr><tr><td>T</td><td>1000</td><td>-</td></tr><tr><td>Number of seeds</td><td>17</td><td>-</td></tr><tr><td>Warm-up steps</td><td>100</td><td>100</td></tr><tr><td>CA scheduler</td><td>Yes</td><td>-</td></tr><tr><td>rv</td><td>64</td><td>{8, 16, 32, 64, 128}</td></tr></table>

Table 4: Hyperparameters-tuning details to use the Transformer model on the Brackets dataset.

# More Ablation Studies

We pursue further ablation studies to cover more aspects of the theoretical results in our experiments.

# Learning Rate Impact

To analyze the impact of the learning rate (lr) on stochastic noise (Eq. 1), we used the experimental settings described for the experiment of Table 3, with the following modifications: the experiments were conducted using three different random seeds, and the number of estimators was fixed at 90 ZO and 3 FO for all trials. The experiments were performed on A6000 GPUs with 48 GB VRAM each. The results, presented in Figure 5, show the relationship between learning rate and convergence behavior. The plot demonstrates that smaller learning rates (e.g., 0.005 and 0.01) result in smoother convergence curves, whereas larger learning rates (e.g., 0.5) exhibit more pronounced oscillations and slower convergence.

# Effect of Random Vector Count on Convergence

While we used 64–128 random vectors (rv) to demonstrate effectiveness, using fewer random vectors can provide a balance between efficiency and accuracy. Experiments were conducted on the MNIST dataset using a multilayer perceptron (MLP) model with two hidden layers of size 128. The experiments were performed on A6000 GPUs with 48 GB VRAM, and additional details on hyperparameter tuning can be found in Table 5.

The results, shown in Figure 6, illustrate the impact of the number of random vectors on loss convergence. Configurations with fewer random vectors (e.g., 8 or 16) show slower convergence and larger fluctuations in loss compared to configurations using higher numbers of random vectors (e.g., 32 or 64). This demonstrates that increasing the number of random vectors enhances the accuracy of gradient estimation, leading to more stable and rapid convergence. However, the computational overhead introduced by using more random vectors must be carefully balanced with available resources. The plot also underscores that zeroth-order nodes can achieve competitive performance by leveraging multiple forward passes, even with reduced computational capabilities.

![](images/881aa8be7238619de3a8a2b1d574047c8fa41328c18c0c7bb592bed5dfb97870.jpg)

<details>
<summary>line</summary>

| Step | lr: 0.005 | lr: 0.05 | lr: 0.001 | lr: 0.01 | lr: 0.1 | lr: 0.5 |
|------|-----------|----------|-----------|----------|---------|---------|
| 0    | 2.3       | 2.3      | 2.3       | 2.3      | 2.3     | 2.3     |
| 100  | 2.15      | 2.18     | 2.2       | 2.17     | 2.2     | 2.2     |
| 200  | 2.1       | 2.15     | 2.18      | 2.14     | 2.18    | 2.19    |
| 300  | 2.08      | 2.13     | 2.16      | 2.12     | 2.16    | 2.17    |
| 400  | 2.07      | 2.11     | 2.14      | 2.1      | 2.14    | 2.16    |
| 500  | 2.06      | 2.1      | 2.12      | 2.09     | 2.12    | 2.15    |
</details>

Figure 5: Impact of the learning rate (lr) on the validation loss, using a regression model on MNIST with 3 FO and 90 ZO nodes.

![](images/4b0a4e33c945f456628ada60bd3b732e5a53d7070694e239746247f15d126f79.jpg)

<details>
<summary>line</summary>

| Step | 90 ZO, rv: 8 | 90 ZO, rv: 16 | 90 ZO, rv: 32 | 3 FO 90 ZO, rv: 8 | 3 FO 90 ZO, rv: 16 | 3 FO 90 ZO, rv: 32 | 3 FO |
|------|--------------|---------------|---------------|-------------------|--------------------|--------------------|------|
| 600  | ~1.75        | ~1.75         | ~1.75         | ~1.75             | ~1.75              | ~1.75              | ~1.75 |
| 800  | ~1.78        | ~1.78         | ~1.78         | ~1.78             | ~1.78              | ~1.78              | ~1.78 |
| 1k   | ~1.70        | ~1.70         | ~1.70         | ~1.70             | ~1.70              | ~1.70              | ~1.70 |
| 1.2k | ~1.68        | ~1.68         | ~1.68         | ~1.68             | ~1.68              | ~1.68              | ~1.68 |
| 1.4k | ~1.65        | ~1.65         | ~1.65         | ~1.65             | ~1.65              | ~1.65              | ~1.65 |
</details>

Figure 6: Impact of the different numbers of random vectors on the validation loss between the hybrid and mono-type estimator population, using an MLP model on MNIST. Confidence intervals are omitted for better interpretability.

# Model Consensus and ZO Population Impact

We expect that each node's model converges to a global model, likely due to each model converging toward a stable point in the final steps. This convergence leads to rapid population across models, as each step involves $O(n)$ interactions. To investigate this phenomenon, we conducted experiments with 16 nodes while varying the number of ZOs within the population to study the effect of ZOs on convergence. To assess the degree of convergence and the closeness of individual models to a global model, we measured the standard deviation of their losses.

The experiments were performed on the MNIST dataset using a model architecture comprising two convolutional layers and two linear layers. All experiments were conducted on A6000 GPUs with 48 GB of VRAM. Additional hyperparameter tuning details are provided in Table 6, and the results are shown in Figure 7.

Figures 7a and 7b demonstrate that the standard deviation of losses across models approaches zero in different settings, indicating strong convergence to a global model. Figure 7a shows that configurations with higher FO nodes (e.g., 16 FO) converge more rapidly, while configurations with more ZO nodes (e.g., 16 ZO) exhibit slower convergence but achieve a similar final loss. Meanwhile, Figure 7b highlights that the standard deviation of losses diminishes across all configurations,

showcasing consistent agreement among models regardless of the number of ZO nodes. This consistency supports the feasibility of using ZO nodes in heterogeneous systems while maintaining overall model consensus.

![](images/d12e3b3881463d04d96c69d18203dc3256f07a55caea1dc9c218beac882b8c69.jpg)

<details>
<summary>line</summary>

| Step | 16 ZO | 4 FO 12 ZO | 8 FO 8 ZO | 12 FO 4 ZO | 16 FO |
| ---- | ----- | ---------- | --------- | ---------- | ----- |
| 800  | 1.64  | 1.635      | 1.635     | 1.63       | 1.63  |
| 900  | 1.625 | 1.62       | 1.62      | 1.615      | 1.61  |
| 1k   | 1.61  | 1.605      | 1.605     | 1.60       | 1.595 |
</details>

(a) Validation loss vs. steps.

![](images/f4806c35582f8122d42242eaebc5f2abd35279371a8bb0c62a0a9972b8f628d2.jpg)

<details>
<summary>line</summary>

| Step | 16 ZO | FO 12 ZO | 8 FO 8 ZO | 12 FO 4 ZO | 16 FO |
|------|-------|----------|-----------|------------|-------|
| 0    | ~0.005 | ~0.01    | ~0.01     | ~0.005     | ~0.005 |
| 200  | ~0.003 | ~0.005   | ~0.005    | ~0.003     | ~0.002 |
| 400  | ~0.002 | ~0.003   | ~0.003    | ~0.002     | ~0.0015 |
| 600  | ~0.0015| ~0.002   | ~0.002    | ~0.0015    | ~0.001 |
| 800  | ~0.001 | ~0.0015  | ~0.0015   | ~0.001     | ~0.0008 |
| 1k   | ~0.001 | ~0.001   | ~0.001    | ~0.001     | ~0.0007 |
</details>

(b) Loss Std (log scale) vs. steps.

Figure 7: (7a) Validation loss and (7b) loss standard deviation across nodes' model loss (Loss Std), considering different populations of 16 nodes using a CNN model on MNIST. 

<table><tr><td>Hyperparameter</td><td>Value</td><td>Search interval</td></tr><tr><td>FO batch size</td><td>8</td><td>-</td></tr><tr><td>ZO batch size</td><td>8</td><td>-</td></tr><tr><td>FO learning rate</td><td>0.1</td><td>[0.001, 0.5]</td></tr><tr><td>ZO learning rate</td><td>0.5</td><td>[0.001, 0.5]</td></tr><tr><td>FO momentum</td><td>0.0</td><td>-</td></tr><tr><td>ZO momentum</td><td>0.0</td><td>-</td></tr><tr><td>T</td><td>1500</td><td>-</td></tr><tr><td>Number of seeds</td><td>3</td><td>-</td></tr><tr><td>Warm-up steps</td><td>50</td><td>-</td></tr><tr><td>CA scheduler</td><td>No</td><td>-</td></tr><tr><td>rv</td><td>-</td><td>-</td></tr></table>

Table 5: Hyperparameter tuning details for the ablation study on the impact of the number of random vectors on the convergence behavior of different populations, conducted using an MLP model on the MNIST dataset.

<table><tr><td>Hyperparameter</td><td>Value</td><td>Search interval</td></tr><tr><td>FO batch size</td><td>256</td><td>{8, 64, 256}</td></tr><tr><td>ZO batch size</td><td>256</td><td>{8, 64, 256}</td></tr><tr><td>FO learning rate</td><td>0.1</td><td>[0.001, 0.1]</td></tr><tr><td>ZO learning rate</td><td>0.1</td><td>[0.001, 0.1]</td></tr><tr><td>FO momentum</td><td>0.0</td><td>-</td></tr><tr><td>ZO momentum</td><td>0.0</td><td>-</td></tr><tr><td>T</td><td>1000</td><td>-</td></tr><tr><td>Number of seeds</td><td>3</td><td>-</td></tr><tr><td>Warm-up steps</td><td>50</td><td>-</td></tr><tr><td>CA scheduler</td><td>No</td><td>-</td></tr><tr><td>rv</td><td>128</td><td>-</td></tr></table>

Table 6: Hyperparameter tuning details for the consensus study and analysis of the impact of different ZO populations, conducted using a CNN model on the MNIST dataset.

# Zeroth-order Stochastic Gradient Properties.

Lemma 5. Let $G_{\nu}^{i}(x, u, \xi^{i})$ be computed by 2. Then, under Assumptions 2 and 4 we have:

$$
\mathbb {E} _ {u, \xi^ {i}} \| G _ {\nu} ^ {i} (x, u, \xi^ {i}) \| ^ {2} \leq \frac {1}{2} \nu^ {2} L ^ {2} (d + 6) ^ {3} + 2 (d + 4) \left[ \| \nabla f ^ {i} (x) \| ^ {2} + s _ {i} ^ {2} \right], \tag {6}
$$

$$
\mathbb {E} _ {u, \xi^ {i}} \| G _ {\nu} ^ {i} (x, u, \xi) - \nabla f ^ {i} (x) \| ^ {2} \leq \frac {3 \nu^ {2}}{2} L ^ {2} (d + 6) ^ {3} + 4 (d + 4) \left[ \| \nabla f ^ {i} (x) \| ^ {2} + s _ {i} ^ {2} \right]. \tag {7}
$$

Proof. Firstly, by plugging in $F^{i}(x,\xi^{i})$ in 6 under Assumptions 2 and 4, we obtain

$$
\mathbb {E} \big \| G _ {\nu} ^ {i} (x, u, \xi) \big \| ^ {2} \leq \frac {1}{2} \nu^ {2} L ^ {2} (d + 6) ^ {3} + 2 (d + 4) \| \nabla F ^ {i} (x, \xi^ {i}) \| ^ {2}
$$

Then by getting an expectation and eliminating the randomness of the right-hand side with respect to $\xi^{i}$ , we get

$$
\mathbb {E} _ {u, \xi^ {i}} \left\| G _ {\nu} ^ {i} (x, u, \xi^ {i}) \right\| ^ {2} \leq \frac {1}{2} \nu^ {2} L ^ {2} (d + 6) ^ {3} + 2 (d + 4) \| \nabla F ^ {i} (x, \xi^ {i}) \| ^ {2}
$$

$$
\stackrel {A s s u m p t i o n 4} {\leq} \frac {1}{2} \nu^ {2} L ^ {2} (d + 6) ^ {3} + 2 (d + 4) \left[ \| f ^ {i} (x) \| ^ {2} + s _ {i} ^ {2} \right].
$$

Secondly, using 4 we have:

$$
\begin{array}{l} \mathbb {E} \left\| G _ {\nu} (x, u, \xi) - \nabla f _ {\nu} (x) \right\| ^ {2} = \mathbb {E} \left\| G _ {\nu} (x, u, \xi) \right\| ^ {2} + \left\| \nabla f _ {\nu} (x) \right\| ^ {2} - 2 \langle \mathbb {E} \left(G _ {\nu} (x, u, \xi)\right), \nabla f _ {\nu} (x) \rangle \\ \stackrel {{4}} {{=}} \mathbb {E} \left\| G _ {\nu} (x, u, \xi) \right\| ^ {2} + \underbrace {\left\| \nabla f _ {\nu} (x) \right\| ^ {2} - 2 \left\| \nabla f _ {\nu} (x) \right\| ^ {2}} _ {- \left\| \nabla f _ {\nu} (x) \right\| ^ {2} \leq 0} \leq \mathbb {E} \left\| G _ {\nu} (x, u, \xi) \right\| ^ {2} \\ \stackrel {6} {\leq} \frac {1}{2} \nu^ {2} L ^ {2} (d + 6) ^ {3} + 2 (d + 4) \left[ \| f ^ {i} (x) \| ^ {2} + s _ {i} ^ {2} \right]. \\ \end{array}
$$

Finally, together with Lemma 1 and the inequality above we can deduce:

$$
\begin{array}{l} \mathbb {E} _ {u, \xi^ {i}} \left\| G _ {\nu} ^ {i} (x, u, \xi^ {i}) - \nabla f ^ {i} (x) \right\| ^ {2} \leq 2 \mathbb {E} _ {u, \xi^ {i}} \left\| G _ {\nu} ^ {i} (x, u, \xi^ {i}) - \nabla f _ {\nu} ^ {i} (x) \right\| ^ {2} + 2 \left\| \nabla f _ {\nu} ^ {i} (x) - \nabla f ^ {i} (x) \right\| ^ {2} \\ \leq \nu^ {2} L ^ {2} (d + 6) ^ {3} + 4 (d + 4) \left[ \| f ^ {i} (x) \| ^ {2} + s _ {i} ^ {2} \right] + 2 \left\| \nabla f _ {\nu} ^ {i} (x) - \nabla f ^ {i} (x) \right\| ^ {2} \\ \stackrel {L e m m a} {\leq} ^ {1} \nu^ {2} L ^ {2} (d + 6) ^ {3} + 4 (d + 4) \left[ \| f ^ {i} (x) \| ^ {2} + s _ {i} ^ {2} \right] + \frac {\nu^ {2}}{2} L ^ {2} (d + 3) ^ {3} \\ \leq \frac {3 \nu^ {2}}{2} L ^ {2} (d + 6) ^ {3} + 4 (d + 4) \left[ \| f ^ {i} (x) \| ^ {2} + s _ {i} ^ {2} \right]. \\ \end{array}
$$

![](images/30394decc140dc925d8eb35a2d3d9ebbb3d16641efbf8fa855e0ed78a9b42bfa.jpg)

# Definitions

For the sake of simplicity, we now define some notations for the frequently-used expressions in the proof.

Definition 3 (Gamma).

$$
\Gamma_ {t} := \frac {1}{n} \sum_ {i} \| X _ {t} ^ {i} - \mu_ {t} \| ^ {2}. \tag {8}
$$

Definition 4 (Average second-moment of estimator).

$$
M _ {t} ^ {G} := \frac {1}{n} \sum_ {i} \left\| G ^ {i} (X _ {t} ^ {i}) \right\| ^ {2}. \tag {9}
$$

Definition 5 (Expectation conditioned step).

$$
\mathbb {E} _ {t} [ Y ] := \mathbb {E} [ Y | X _ {t} ^ {1}, X _ {t} ^ {2},..., X _ {t} ^ {n} ]. \tag {10}
$$

Definition 6 (Biasedness of estimators). For node i, using $G^{i}(x)$ as its gradient estimator we define $b_{i}$ as the upper bound for its biasedness, i.e.

$$
\left\| \nabla f ^ {i} (x) - \mathbb {E} [ G ^ {i} (x) ] \right\| \leq b _ {i}. \tag {11}
$$

Note that for an unbiased estimator we have $b_{i} = 0$ . Moreover, for zeroth-order estimators $\mathbb{E}\left[G^{i}(x)\right] = \nabla f^{i}(x)$ . Hence, according to Lemma 1, $\left\| \nabla f^{i}(x) - \mathbb{E}\left[G^{i}(x)\right]\right\|$ is bounded for a fixed $\nu$ . Therefore, $b_{i}$ is well-defined in our setup.

We further define the average biasedness of estimators as

$$
B := \frac {1}{n} \sum_ {i} b _ {i}. \tag {12}
$$

Definition 7 (Variance of estimators). For node i, using $G^{i}(X_{t}^{i})$ as its gradient estimator at step t, we define $(\sigma_{t}^{i})^{2}$ as the upper-bound of its variance, i.e.

$$
E \left\| \nabla f ^ {i} (X _ {t} ^ {i}) - G ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} \leq (\sigma_ {t} ^ {i}) ^ {2}. \tag {13}
$$

Note that for the first-order nodes, i.e. $G^{i}(x) = \nabla F^{i}(x)$ , using 4 we have $(\sigma_{t}^{i})^{2} := s_{i}^{2}$ . Moreover, for the zeroth-order nodes, i.e. $G^{i}(X_{t}^{i})$ is computed using 2, according to 7 we have $(\sigma_{t}^{i})^{2} := \frac{3\nu^{2}}{2} L^{2}(d+6)^{3} + 4(d+4) \left[ \| \nabla f^{i}(X_{t}^{i}) \|^{2} + s_{i}^{2} \right]$ , which is well-defined considering that $\nu$ is fixed in our setup.

We further define the average variance of estimators as

$$
(\overline {{\sigma}} _ {t}) ^ {2} := \frac {1}{n} \sum_ {i} (\sigma_ {t} ^ {i}) ^ {2}. \tag {14}
$$

# Useful Inequalities

Lemma 6 (Young). For any pair of vectors $x, y$ and $\alpha > 0$ we have

$$
\langle x, y \rangle \leq \frac {\| x \| ^ {2}}{2 \alpha} + \frac {\alpha \| y \| ^ {2}}{2}.
$$

Lemma 7 (Cauchy-Schwarz). For any vectors $x_{1}, x_{2}, \ldots, x_{n} \in R^{d}$ we have

$$
\| \sum_ {i = 1} ^ {n} x _ {i} \| ^ {2} \leq n \sum_ {i = 1} ^ {n} \| x _ {i} \| ^ {2}.
$$

# The Complete Convergence Proof

# Proof of Convex Case of Theorem 1

In this part, we assume that there exist $n_{0}$ zeroth-order nodes and $n_{1}$ first-order nodes, all having access to a shared dataset, hence a shared objective function f that they want to minimize.

Lemma 8. For any time step $t$ and constants $\alpha_0 \geq \alpha_1 > 0$ let $M_t^f(\alpha_0, \alpha_1) = \frac{\alpha_0}{n} \sum_{i \in N_0} \| \nabla f^i(X_t^i) \|^2 + \frac{\alpha_1}{n} \sum_{i \in N_1} \| \nabla f^i(X_t^i) \|^2$ . We have that:

$$
\mathbb {E} [ M _ {t} ^ {f} (\alpha_ {0}, \alpha_ {1}) ] \leq 3 L ^ {2} \alpha_ {0} \mathbb {E} [ \Gamma_ {t} ] + \frac {3 \alpha_ {0} n _ {0} \varsigma_ {0} ^ {2} + 3 \alpha_ {1} n _ {1} \varsigma_ {1} ^ {2}}{n} + 6 (\alpha_ {0} + \alpha_ {1}) L \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ].
$$

Proof.

$$
\frac {\alpha_ {0}}{n} \sum_ {i \in N _ {0}} \mathbb {E} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} = \frac {\alpha_ {0}}{n} \sum_ {i \in N _ {0}} \mathbb {E} \left\| \nabla f (X _ {t} ^ {i}) - \nabla f ^ {i} (\mu_ {t}) + \nabla f ^ {i} (\mu_ {t}) - \nabla f (\mu_ {t}) + \nabla f (\mu_ {t}) - \nabla f (x ^ {*}) \right\| ^ {2}
$$

$$
\leq^ {\text {Assumptions 2 and 3,Cauchy - Schwarz}} 3 L ^ {2} \alpha_ {0} \sum_ {i \in N _ {0}} \mathbb {E} \| X _ {t} ^ {i} - \mu_ {t} \| ^ {2} + \frac {3 \alpha_ {0} n _ {0} \varsigma_ {0} ^ {2}}{n} + 6 \alpha_ {0} L \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ]. \tag {15}
$$

Similarly, in the case of first-order nodes we get:

$$
\frac {\alpha_ {1}}{n} \sum_ {i \in N _ {1}} \mathbb {E} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} \leq 3 L ^ {2} \alpha_ {1} \sum_ {i \in N _ {1}} \mathbb {E} \left\| X _ {t} ^ {i} - \mu_ {t} \right\| ^ {2} + \frac {3 \alpha_ {1} n _ {1} \varsigma_ {1} ^ {2}}{n} + 6 \alpha_ {1} L \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ]. \tag {16}
$$

By summing up the above inequalities and using the fact that $\alpha_0 \geq \alpha_1$ (together with the definition of $\Gamma_t$ ), we get the proof of the lemma.

Lemma 3. Assume $\nu := \frac{\eta}{c}$ is fixed, where $\eta$ and $c$ are the learning rate and a constant respectively. Then, for any time step $t$ we have:

$$
\begin{array}{l} \mathbb {E} \left[ M _ {t} ^ {G} \right] \leq 6 (d + 4) L ^ {2} \mathbb {E} [ \Gamma_ {t} ] + \frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} \\ + 6 (2 d + 9) L \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ] \\ + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} \\ + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}. \\ \end{array}
$$

Proof.

$$
\begin{array}{l} \mathbb {E} _ {t} \left[ M _ {t} ^ {G} \right] = \frac {1}{n} \sum_ {i} \mathbb {E} _ {t} \left\| G ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} \stackrel {(6)} {\leq} \frac {1}{n} \sum_ {i \in N _ {0}} (\frac {1}{2} \nu^ {2} L ^ {2} (d + 6) ^ {3} + 2 (d + 4) \left[ \mathbb {E} _ {t} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} + s _ {i} ^ {2} \right]) \\ + \frac {1}{n} \sum_ {i \in N _ {1}} (\mathbb {E} _ {t} \| \nabla f ^ {i} (X _ {t} ^ {i}) \| ^ {2} + s _ {i} ^ {2}) \tag {17} \\ \end{array}
$$

$$
\leq \mathbb {E} _ {t} [ M _ {t} ^ {f} (2 (d + 4), 1) ] + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}.
$$

Next, we take expectation with respect to $X_{t}^{1}, X_{t}^{2}, \ldots, X_{t}^{n}$ and use Lemma 8 to get:

$$
\begin{array}{l} \mathbb {E} \left[ M _ {t} ^ {G} \right] \leq 6 (d + 4) L ^ {2} \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {6 (d + 4) n _ {0} s _ {0} ^ {2} + 3 n _ {1} s _ {1} ^ {2}}{n} + 6 (2 d + 9) L \mathbb {E} \left[ f \left(\mu_ {t}\right) - f \left(x ^ {*}\right) \right] \tag {18} \\ + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}. \\ \end{array}
$$

Which finishes the proof of the lemma.

Lemma 9. Assume $\nu := \frac{\eta}{c}$ is fixed, where $\eta$ and $c$ are the learning rate and a constant respectively. Then, for any time step $t$ we have

$$
\begin{array}{l} \mathbb {E} \left[ \left(\overline {{\sigma}} _ {t}\right) ^ {2} \right] \leq 1 2 (d + 4) L ^ {2} \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {1 2 n _ {0} (d + 4) \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + 6 (4 d + 1 7) L \left(f \left(\mu_ {t}\right) - f \left(x ^ {*}\right)\right) \\ + \frac {4 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {3 n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}. \\ \end{array}
$$

Proof.

$$
\begin{array}{l} \mathbb {E} _ {t} [ (\overline {{\sigma}} _ {t}) ^ {2} ] = \frac {1}{n} \sum_ {i} \mathbb {E} _ {t} [ (\sigma_ {t} ^ {i}) ^ {2} ] \stackrel {(7)} {\leq} \frac {1}{n} \sum_ {i \in N _ {0}} (\frac {3 \nu^ {2}}{2} L ^ {2} (d + 6) ^ {3} + 4 (d + 4) \left[ \mathbb {E} _ {t} \| \nabla f (X _ {t} ^ {i}) \| ^ {2} + s _ {i} ^ {2} \right]) \\ + \frac {1}{n} \sum_ {i \in N _ {1}} (\mathbb {E} _ {t} \| \nabla f ^ {i} (X _ {t} ^ {i}) \| ^ {2} + s _ {i} ^ {2}) \tag {19} \\ = \mathbb {E} _ {t} [ M _ {t} ^ {f} (4 (d + 4), 1) ] + \frac {4 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} \sigma^ {2} + \eta^ {2} \frac {3 n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}. \\ \end{array}
$$

Next, we take expectation with respect to $X_{t}^{1}, X_{t}^{2}, \ldots, X_{t}^{n}$ and use Lemma 8 to get:

$$
\begin{array}{l} \mathbb {E} _ {t} [ (\overline {{\sigma}} _ {t}) ^ {2} ] \leq 1 2 (d + 4) L ^ {2} \mathbb {E} [ \Gamma_ {t} ] + \frac {1 2 n _ {0} (d + 4) \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + 6 (4 d + 1 7) L (f (\mu_ {t}) - f (x ^ {*})) \\ + \frac {4 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {3 n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}. \\ \end{array}
$$

Which finishes the proof of the lemma.

Lemma 10. Assume $\nu := \frac{\eta}{c}$ is fixed, where $\eta$ and $c$ are the learning rate and a constant respectively. Then, for any time step $t$ we have

$$
B \leq \eta \frac {n _ {0}}{2 c n} L (d + 3) ^ {\frac {3}{2}}.
$$

Proof.

$$
B = \frac {1}{n} \sum_ {i} b _ {i} = \frac {1}{n} \sum_ {i} \| \nabla f ^ {i} (X _ {t} ^ {i}) - \mathbb {E} [ G ^ {i} (X _ {t} ^ {i}) ] \| = \frac {1}{n} \sum_ {i \in N _ {0}} \| \nabla f ^ {i} (X _ {t} ^ {i}) - \nabla f _ {\nu} ^ {i} (X _ {t} ^ {i}) \| \tag {20}
$$

$$
\stackrel {(1)} {\leq} \frac {\nu n _ {0}}{2 n} L (d + 3) ^ {\frac {3}{2}} = \eta \frac {n _ {0}}{2 c n} L (d + 3) ^ {\frac {3}{2}}. \tag {21}
$$

Lemma 2. For any time step $t$ :

$$
\mathbb {E} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {4}{n} \eta^ {2} \mathbb {E} \left[ M _ {t} ^ {G} \right].
$$

Proof. First we can open $\mathbb{E}_t[\Gamma_{t + 1}]$ as

$$
\mathbb {E} _ {t} \left[ \Gamma_ {t + 1} \right] = \mathbb {E} _ {t} \left[ \frac {1}{n} \sum_ {i} \| X _ {t + 1} ^ {i} - \mu_ {t + 1} \| ^ {2} \right]. \tag {22}
$$

Observe that in this case $\mu_{t + 1} = \mu_t - \eta (G_t^i (X_t^i) + G_t^j (X_t^j)) / n$ and

$$
X _ {t + 1} ^ {i} = X _ {t + 1} ^ {j} = (X _ {t} ^ {i} + X _ {t} ^ {j}) / 2 - \eta (G _ {t} ^ {i} (X _ {t} ^ {i}) + G _ {t} ^ {j} (X _ {t} ^ {j})) / 2.
$$

Hence,

$$
\mathbb {E} _ {t} \left[ \Gamma_ {t + 1} \right] = \frac {1}{n ^ {2} (n - 1)} \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left[ 2 \left| \left(X _ {t} ^ {i} + X _ {t} ^ {j}\right) / 2 - \left(\frac {n - 2}{2 n}\right) \eta \left(G ^ {i} \left(X _ {t} ^ {i}\right) + G ^ {j} \left(X _ {t} ^ {j}\right)\right) - \mu_ {t} \right| \right\rVert^ {2}
$$

$$
\left. + \sum_ {k \neq i, j} \left\| X _ {t} ^ {k} - \mu_ {t} + \frac {\eta}{n} (G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j})) \right\| ^ {2} \right]
$$

$$
= \frac {1}{n ^ {2} (n - 1)} \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left[ 2 \left(\| \left(X _ {t} ^ {i} + X _ {t} ^ {j}\right) / 2 - \mu_ {t} \| ^ {2} + \left(\frac {n - 2}{2 n}\right) ^ {2} \eta^ {2} \| G ^ {i} \left(X _ {t} ^ {i}\right) + G ^ {j} \left(X _ {t} ^ {j}\right) \| ^ {2} \right. \right.
$$

$$
\left. - \left(\frac {n - 2}{n}\right) \eta \Bigl \langle G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j}), (X _ {t} ^ {i} + X _ {t} ^ {j}) / 2 - \mu_ {t} \Bigr \rangle\right)
$$

$$
+ \sum_ {k \neq i, j} \Big (\| X _ {t} ^ {k} - \mu_ {t} \| ^ {2} + \big (\frac {1}{n} \big) ^ {2} \eta^ {2} \big \| G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j}) \big \| ^ {2}
$$

$$
\left. \left. + \frac {2}{n} \eta \Bigl \langle G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j}), X _ {t} ^ {k} - \mu_ {t} \Bigr \rangle\right) \right]
$$

$$
= \frac {1}{n ^ {2} (n - 1)} \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left[ \sum_ {k} \| X _ {t} ^ {k} - \mu_ {t} \| ^ {2} - \| X _ {t} ^ {i} - \mu_ {t} \| ^ {2} / 2 - \| X _ {t} ^ {j} - \mu_ {t} \| ^ {2} / 2 + \langle X _ {t} ^ {i} - \mu_ {t}, X _ {t} ^ {j} - \mu_ {t} \rangle \right.
$$

$$
+ \underbrace {\left(\frac {(n - 2) ^ {2}}{2 n ^ {2}} + \frac {n - 2}{n ^ {2}}\right)} _ {\frac {n - 2}{2 n} \leq \frac {1}{2}} \eta^ {2} \left\| G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j}) \right\| ^ {2} \tag {23}
$$

$$
- \underbrace {\left(\frac {n - 2}{n} + \frac {2}{n}\right)} _ {1} \eta \Big \langle G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j}), X _ {t} ^ {i} + X _ {t} ^ {j} - 2 \mu_ {t} \Big \rangle \Bigg ]
$$

$$
\leq (1 - \frac {1}{n}) \mathbb {E} _ {t} [ \Gamma_ {t} ] - \frac {1}{n ^ {2} (n - 1)} \eta \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left\langle G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j}), X _ {t} ^ {i} + X _ {t} ^ {j} - 2 \mu_ {t} \right\rangle
$$

$$
+ \frac {1}{n ^ {2} (n - 1)} \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \langle X _ {t} ^ {i} - \mu_ {t}, X _ {t} ^ {j} - \mu_ {t} \rangle + \frac {1}{2 n ^ {2} (n - 1)} \eta^ {2} \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left\| G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j}) \right\| ^ {2}
$$

$$
\leq (1 - \frac {1}{n})   \mathbb {E} _ {t} \left[ \Gamma_ {t} \right] + \underbrace {\frac {1}{n ^ {2} (n - 1)} \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \langle X _ {t} ^ {i} - \mu_ {t} , X _ {t} ^ {j} - \mu_ {t} \rangle} _ {P _ {1} :=} + \underbrace {\frac {2}{n ^ {2}} \eta^ {2} \sum_ {i} \mathbb {E} _ {t} \left\| G ^ {i} (X _ {t} ^ {i}) \right\| ^ {2}} _ {\frac {2}{n} \eta   \mathbb {E} _ {t} \left[ M _ {t} ^ {G} \right]}.
$$

$$
- \underbrace {\frac {1}{n ^ {2} (n - 1)} \eta \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left\langle G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j}) , X _ {t} ^ {i} + X _ {t} ^ {j} - 2 \mu_ {t} \right\rangle} _ {P _ {2} :=}
$$

Now we upper bound each of $P_{1}$ and $P_{2}$ as following

$$
P _ {1} = \frac {1}{n ^ {2} (n - 1)} \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left\langle X _ {t} ^ {i} - \mu_ {t}, X _ {t} ^ {j} - \mu_ {t} \right\rangle = \frac {- 1}{n ^ {2} (n - 1)} \sum_ {i} \mathbb {E} _ {t} \| X _ {t} ^ {i} - \mu_ {t} \| ^ {2} = \frac {- 1}{n (n - 1)} \mathbb {E} _ {t} [ \Gamma_ {t} ] \tag {24}
$$

$$
\begin{array}{l} P _ {2} = \frac {1}{n ^ {2} (n - 1)} \eta \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left\langle G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j}), X _ {t} ^ {i} + X _ {t} ^ {j} - 2 \mu_ {t} \right\rangle \\ = \frac {2}{n ^ {2} (n - 1)} \eta \left(\sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left\langle G ^ {i} \left(X _ {t} ^ {i}\right), X _ {t} ^ {j} - \mu_ {t} \right\rangle + (n - 1) \sum_ {i} \mathbb {E} _ {t} \left\langle G ^ {i} \left(X _ {t} ^ {i}\right), X _ {t} ^ {i} - \mu_ {t} \right\rangle\right) \tag {25} \\ = \frac {2 (n - 2)}{n ^ {2} (n - 1)} \sum_ {i} \mathbb {E} _ {t} \left\langle \eta G ^ {i} (X _ {t} ^ {i}), X _ {t} ^ {i} - \mu_ {t} \right\rangle \stackrel {\text {Young}} {\leq} \frac {1}{n ^ {2}} \sum_ {i} \left(2 \eta^ {2} \mathbb {E} _ {t} \left\| G ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} + \frac {1}{2} \mathbb {E} _ {t} \left\| X _ {t} ^ {i} - \mu_ {t} \right\| ^ {2}\right) \\ = \frac {2}{n} \eta^ {2} \mathbb {E} _ {t} [ M _ {t} ^ {G} ] + \frac {1}{2 n} \mathbb {E} _ {t} [ \Gamma_ {t} ]. \\ \end{array}
$$

By using (24) and (25) in inequality (23) we get

$$
\mathbb {E} _ {t} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{n}\right) \mathbb {E} _ {t} \left[ \Gamma_ {t} \right] - \frac {1}{n (n - 1)} \mathbb {E} _ {t} \left[ \Gamma_ {t} \right] + \frac {2}{n} \eta^ {2} \mathbb {E} _ {t} \left[ M _ {t} ^ {G} \right] + \frac {2}{n} \eta^ {2} \mathbb {E} _ {t} \left[ M _ {t} ^ {G} \right] + \frac {1}{2 n} \mathbb {E} _ {t} \left[ \Gamma_ {t} \right] \tag {26}
$$

$$
\leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} _ {t} \left[ \Gamma_ {t} \right] + \frac {4}{n} \eta^ {2} \mathbb {E} _ {t} \left[ M _ {t} ^ {G} \right].
$$

Finally, by taking the expectation with respect to $X_{t}^{1}, X_{t}^{2}, \ldots, X_{t}^{n}$ we will have

$$
\mathbb {E} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {4}{n} \eta^ {2} \mathbb {E} \left[ M _ {t} ^ {G} \right] \tag {27}
$$

![](images/6c369f69cd722a4f3c193a09d241aca6d554a6e9996ea8b5dcf9b78baace6a8d.jpg)

Lemma 11. For any time step $t$ and fixed learning rate $\eta \leq \frac{1}{14L(d + 4)^{\frac{1}{2}}}$

$$
\mathbb {E} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{4 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {1 2 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n ^ {2}} + \frac {2 4 \eta^ {2} (2 d + 9) L \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ]}{n}
$$

$$
+ \frac {4 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {2}} + \frac {2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {2} c ^ {2}}.
$$

Proof. From Lemma 2 we get that:

$$
\mathbb {E} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {4}{n} \eta^ {2} \mathbb {E} \left[ M _ {t} ^ {G} \right].
$$

Now, by using Lemma 3 in the inequality above we have

$$
\begin{array}{l} \mathbb {E} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {4}{n} \eta^ {2} \mathbb {E} \left[ M _ {t} ^ {G} \right] \\ \leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {4 \eta^ {2}}{n} \left(6 (d + 4) L ^ {2} \Gamma_ {t} + \frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + 6 (2 d + 9) L \left(f \left(\mu_ {t}\right) - f \left(x ^ {*}\right)\right) \right. \\ \left. + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}\right) \\ = \left(1 - \frac {1}{2 n} + \frac {2 4 \eta^ {2} L ^ {2} (d + 4)}{n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {1 2 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n ^ {2}} + \frac {2 4 \eta^ {2} (2 d + 9) L   \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ]}{n} \\ + \frac {4 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {2}} + \frac {2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {2} c ^ {2}}. \\ \end{array}
$$

We get the proof of the lemma by using $\eta \leq \frac{1}{14L(d + 4)^{\frac{1}{2}}}$ in the above inequality.

Next, we define the following weights: for any step $t \geq 0$ , let $w_{t} = (1 - \frac{\eta\ell}{2n})^{-t}$ . This allows us to prove the following lemma: Lemma 12. for any $T \geq 0$ and $\eta \leq \frac{1}{10\ell}$ :

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} w _ {t} \mathbb {E} [ \Gamma_ {t - 1} ] \leq 1 2 0 \eta^ {2} (2 d + 9) L \sum_ {t = 1} ^ {T - 1} w _ {t} \mathbb {E} [ f (\mu_ {t - 1}) - f (x ^ {*}) ] \\ + \left(\frac {6 0 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {2 0 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {1 0 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n c ^ {2}}\right) \sum_ {t = 1} ^ {T - 1} w _ {t}. \\ \end{array}
$$

Proof. Let $P_{t} = \frac{24\eta^{2}(2d + 9)L\mathbb{E}[f(\mu_{t}) - f(x^{*})]}{n}$

and let $Q = \frac{12\eta^{2}(2(d+4)n_{0}s_{0}^{2}+n_{1}s_{1}^{2})}{n^{2}} + \frac{4\eta^{2}(2(d+4)\sum_{i\in N_{0}}s_{i}^{2}+\sum_{i\in N_{1}}s_{i}^{2})}{n^{2}} + \frac{2\eta^{4}n_{0}L^{2}(d+6)^{3}}{n^{2}c^{2}}$ . Then the above lemma gives us that for any $t \geq 0$ : $\mathbb{E}[\Gamma_{t+1}] \leq (1 - \frac{1}{4n})\mathbb{E}[\Gamma_{t}] + P_{t} + Q$ . After unrolling the recursion, we get that for any t > 1, $\mathbb{E}[\Gamma_{t}] \leq \sum_{i=0}^{t-1}(P_{i} + Q)(1 - \frac{1}{4n})^{t-1-i}$ . Hence,

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} w _ {t} \mathbb {E} [ \Gamma_ {t - 1} ] \leq \sum_ {t = 2} ^ {T} w _ {t} \left(\sum_ {i = 0} ^ {t - 2} (P _ {i} + Q) (1 - \frac {1}{4 n}) ^ {t - 2 - i}\right) = \sum_ {t = 0} ^ {T - 2} (P _ {t} + Q) \sum_ {i = t + 2} ^ {T} w _ {i} (1 - \frac {1}{4 n}) ^ {i - 2 - t} \\ = (1 - \frac {\eta \ell}{2 n}) ^ {- 1} \sum_ {t = 0} ^ {T - 2} (P _ {t} + Q) \sum_ {i = t + 2} ^ {T} w _ {t + 1} (1 - \frac {\eta \ell}{2 n}) ^ {- (i - (t + 2))} (1 - \frac {1}{4 n}) ^ {i - (t + 2)} \tag {28} \\ = (1 - \frac {\eta \ell}{2 n}) ^ {- 1} \sum_ {t = 0} ^ {T - 2} w _ {t + 1} (P _ {t} + Q) \sum_ {j = 0} ^ {T - (t + 2)} \left(\frac {1 - \frac {1}{4 n}}{1 - \frac {\eta \ell}{2 n}}\right) ^ {j}. \\ \end{array}
$$

For $\frac{1}{10\ell} \geq \eta$ , we have $r := \frac{1 - \frac{1}{4n}}{1 - \frac{\eta\ell}{2n}} \leq 1$ . Hence, we can write

$$
\sum_ {j = 0} ^ {T - (t + 2)} \left(\frac {1 - \frac {1}{4 n}}{1 - \frac {\eta \ell}{2 n}}\right) ^ {j} = \sum_ {j = 0} ^ {T - (t + 2)} r ^ {j} = \frac {1 - r ^ {T - (t + 1)}}{1 - r} \stackrel {t \leq T - 2} {\leq} \frac {1}{1 - r}. \tag {29}
$$

By using the above inequality in (28) we have

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} w _ {t} \mathbb {E} [ \Gamma_ {t - 1} ] \leq (1 - \frac {\eta \ell}{2 n}) ^ {- 1} \frac {1}{1 - \frac {1 - \frac {1}{4 n}}{1 - \frac {\eta \ell}{2 n}}} \sum_ {t = 0} ^ {T - 2} w _ {t + 1} (P _ {t} + Q) \\ = \frac {1}{\frac {1}{4 n} - \frac {\eta \ell}{2 n}} \sum_ {t = 0} ^ {T - 2} w _ {t + 1} (P _ {t} + Q) = \frac {1}{\frac {1}{4 n} - \frac {\eta \ell}{2 n}} \sum_ {t = 1} ^ {T - 1} w _ {t} (P _ {t - 1} + Q). \\ \end{array}
$$

Finally, since $\frac{1}{10\ell} \geq \eta$ we get $\frac{1}{\frac{1}{4n} - \frac{\eta\ell}{2n}} \leq 5n$ and the proof of lemma is finished.

Lemma 13. For $\eta \leq \frac{\sqrt{\ell cn}}{2\sqrt{Ln_0}(d + 3)^{\frac{3}{4}}}$ , we have that

$$
\begin{array}{l} \mathbb {E} \left\| \mu_ {t + 1} - x ^ {*} \right\| ^ {2} \leq (1 - \frac {\ell \eta}{n} + \eta^ {2} \frac {4 B}{n}) \mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\| ^ {2} - (4 \frac {\eta}{n} - \eta^ {2} \frac {1 6 L (1 2 d + 5 2)}{n ^ {2}} - \eta^ {4} \frac {6 4 B L}{n ^ {3}}) \mathbb {E} \left[ f (\mu_ {t}) - f (x ^ {*}) \right] \\ + \left(2 \frac {L + \ell}{n} \eta + \eta^ {2} L ^ {2} \frac {9 6 d + 4 5 6}{n ^ {2}} + \eta^ {4} \frac {3 2 B L ^ {2}}{n ^ {3}}\right) \mathbb {E} [ \Gamma_ {t} ] \\ + \frac {\eta^ {2} ((9 6 d + 4 4 8) \varsigma_ {0} ^ {2} n _ {0} + 8 8 \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}} + \frac {8 \eta^ {2} ((d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {3}} + \frac {1 2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {3} c ^ {2}} + \eta^ {2} \frac {2 B}{n}. \\ \end{array}
$$

Proof. Let $F_{t}$ be the amount by which $\mu_{t}$ decreases at step t. So, $F_{t}$ is a sum of $\frac{\eta}{n}G^{i}(X_{t}^{i})$ and $\frac{\eta}{n}G^{j}(X_{t}^{j})$ for agents i and j, which interact at step t. Also, let $F_{t}^{\prime}$ be the amount by which $\mu_{t}$ would decrease if all the agents were contributing at that step using their true local gradients. That is $F_{t}^{\prime} = \frac{2\eta}{n^{2}}\sum_{i}\nabla f^{i}(X_{t}^{i})$ .

To make the calculations more clear, lets define $E_{t}[Y] := E[Y|X_{t}^{1}, X_{t}^{2}, ..., X_{t}^{n}]$ .

$$
\begin{array}{l} \mathbb {E} \left\| \mu_ {t + 1} - x ^ {*} \right\| ^ {2} = \mathbb {E} \left\| \mu_ {t} - F _ {t} - x ^ {*} \right\| ^ {2} = \mathbb {E} \left\| \mu_ {t} - F _ {t} - x ^ {*} - F _ {t} ^ {\prime} + F _ {t} ^ {\prime} \right\| ^ {2} \\ = \mathbb {E} \left\| \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime} \right\| ^ {2} + \mathbb {E} \left\| F _ {t} ^ {\prime} - F _ {t} \right\| ^ {2} + 2 \mathbb {E} \left\langle \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime}, F _ {t} ^ {\prime} - F _ {t} \right\rangle \\ = \mathbb {E} \left\| \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime} \right\| ^ {2} + \mathbb {E} _ {X _ {t} ^ {1}, X _ {t} ^ {2}, \dots , X _ {t} ^ {n}} \left[ \mathbb {E} _ {t} \left\| F _ {t} ^ {\prime} - F _ {t} \right\| ^ {2} \right] \\ + 2 \mathbb {E} _ {X _ {t} ^ {1}, X _ {t} ^ {2}, \dots , X _ {t} ^ {n}} \left[ \mathbb {E} _ {t} \left\langle \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime}, F _ {t} ^ {\prime} - F _ {t} \right\rangle \right] \tag {30} \\ \end{array}
$$

This means that in order to upper bound $\mathbb{E}\left\| \mu_{t + 1} - x^{*}\right\|^{2}$ , we need to upper bound $\mathbb{E}\left\| \mu_t - x^* - F_t'\right\| ^2$ , $\mathbb{E}_t\left\| F_t' - F_t\right\| ^2$ , and $\mathbb{E}_t\left\langle \mu_t - x^* - F_t', F_t' - F_t\right\rangle$ .

For the first one, when $X_{1}, X_{2}, \ldots, X_{n}$ are fixed, we have that

$$
\begin{array}{l} \left\| \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime} \right\| ^ {2} = \left\| \mu_ {t} - x ^ {*} - \frac {2 \eta}{n ^ {2}} \sum_ {i} \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} \\ = \left\| \mu_ {t} - x ^ {*} \right\| ^ {2} + 4 \frac {\eta^ {2}}{n ^ {2}} \underbrace {\left\| \frac {1}{n} \sum_ {i} \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2}} _ {R _ {1} :=} - 4 \frac {\eta}{n} \underbrace {\left\langle \mu_ {t} - x ^ {*} , \frac {1}{n} \sum_ {i} \nabla f ^ {i} (X _ {t} ^ {i}) \right\rangle} _ {R _ {2} :=} \tag {31} \\ \end{array}
$$

$$
R _ {1} = \left\| \frac {1}{n} \sum_ {i} \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} = \left\| \frac {1}{n} \sum_ {i} \nabla f ^ {i} (X _ {t} ^ {i}) - \nabla f ^ {i} (\mu_ {t}) + \nabla f ^ {i} (\mu_ {t}) - \nabla f ^ {i} (x ^ {*}) \right\| ^ {2}
$$

$$
\leq \frac {2}{n} \sum_ {i} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) - \nabla f ^ {i} (\mu_ {t}) \right\| ^ {2} + 2 \left\| \frac {1}{n} \sum_ {i} \nabla f ^ {i} (\mu_ {t}) - \nabla f ^ {i} (x ^ {*}) \right\| ^ {2} \tag {32}
$$

$$
\leq \frac {2 L ^ {2}}{n} \sum_ {i} \left\| X _ {t} ^ {i} - \mu_ {t} \right\| ^ {2} + \frac {4 L}{n} \sum_ {i} \left(f ^ {i} (\mu_ {t}) - f ^ {i} (x ^ {*})\right)
$$

$$
= \frac {2 L ^ {2}}{n} \sum_ {i} \left\| X _ {t} ^ {i} - \mu_ {t} \right\| ^ {2} + 4 L \left[ f (\mu_ {t}) - f (x ^ {*}) \right]
$$

$$
\begin{array}{l} R _ {2} = \left\langle \mu_ {t} - x ^ {*}, \frac {1}{n} \sum_ {i} \nabla f ^ {i} (X _ {t} ^ {i}) \right\rangle = \frac {1}{n} \sum_ {i} \left\langle \mu_ {t} - X _ {t} ^ {i} + X _ {t} ^ {i} - x ^ {*}, \nabla f _ {t} ^ {i} (X _ {t} ^ {i}) \right\rangle \tag {33a} \\ = \frac {1}{n} \sum_ {i} \left[ \left\langle \mu_ {t} - X _ {t} ^ {i}, \nabla f ^ {i} (X _ {t} ^ {i}) \right\rangle + \left\langle X _ {t} ^ {i} - x ^ {*}, \nabla f ^ {i} (X _ {t} ^ {i}) \right\rangle \right] \\ \end{array}
$$

Using L-smoothness property (Assumption 2) with $y = X_{t}^{i}$ and $x = x^{*}$ we have

$$
\left\langle \mu_ {t} - X _ {t} ^ {i}, \nabla f ^ {i} (X _ {t} ^ {i}) \right\rangle \geq f ^ {i} (\mu_ {t}) - f ^ {i} (X _ {t} ^ {i}) - \frac {L}{2} \| \nabla f ^ {i} (\mu_ {t}) - \nabla f ^ {i} (X _ {t} ^ {i}) \| ^ {2}. \tag {33b}
$$

Additionally, we use the $\ell$ -strong convexity (Assumption 1), to get

$$
\left\langle X _ {t} ^ {i} - x ^ {*}, \nabla f ^ {i} (X _ {t} ^ {i}) \right\rangle \geq (f ^ {i} (X _ {t} ^ {i}) - f ^ {i} (x ^ {*})) + \frac {\ell}{2} \| X _ {t} ^ {i} - x ^ {*} \| ^ {2}. \tag {33c}
$$

Now by plugging (33b) and (33c) in inequality (33a) we get that

$$
\begin{array}{l} R _ {2} \geq \frac {1}{n} \sum_ {i} \left[ f ^ {i} \left(\mu_ {t}\right) - f ^ {i} \left(X _ {t} ^ {i}\right) - \frac {L}{2} \| \nabla f ^ {i} \left(\mu_ {t}\right) - \nabla f ^ {i} \left(X _ {t} ^ {i}\right) \| ^ {2} + f ^ {i} \left(X _ {t} ^ {i}\right) - f ^ {i} \left(x ^ {*}\right) + \frac {\ell}{2} \| X _ {t} ^ {i} - x ^ {*} \| ^ {2} \right] \\ = \left[ f \left(\mu_ {t}\right) - f \left(x ^ {*}\right) \right] - \frac {L}{2 n} \sum_ {i} \| X _ {t} ^ {i} - \mu_ {t} \| ^ {2} + \frac {\ell}{2 n} \sum_ {i} \| X _ {t} ^ {i} - x ^ {*} \| ^ {2} \tag {34} \\ \geq \left[ f (\mu_ {t}) - f (x ^ {*}) \right] - \frac {L + \ell}{2 n} \sum_ {i} \left\| X _ {t} ^ {i} - \mu_ {t} \right\| ^ {2} + \frac {\ell}{4} \| \mu_ {t} - x ^ {*} \| ^ {2}. \\ \end{array}
$$

Now we plug (32) and (34) back into (31) and take expectation into the account to get

$$
\begin{array}{l} \mathbb {E} \left\| \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime} \right\| ^ {2} \leq \mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\| ^ {2} + 4 \frac {\eta^ {2}}{n ^ {2}} \Big (\frac {2 L ^ {2}}{n} \sum_ {i} \left\| X _ {t} ^ {i} - \mu_ {t} \right\| ^ {2} + 4 L \mathbb {E} \left[ f (\mu_ {t}) - f (x ^ {*}) \right] \Big) \\ - 4 \frac {\eta}{n} \left(\left[ f \left(\mu_ {t}\right) - f \left(x ^ {*}\right) \right] - \frac {L + \ell}{2 n} \sum_ {i} \mathbb {E} \left\| X _ {t} ^ {i} - \mu_ {t} \right\| ^ {2} + \frac {\ell}{4} \mathbb {E} \| \mu_ {t} - x ^ {*} \| ^ {2}\right) \tag {35} \\ = (1 - \frac {\ell \eta}{n}) \mathbb {E} \| \mu_ {t} - x ^ {*} \| ^ {2} - (4 \frac {\eta}{n} - 1 6 L \frac {\eta^ {2}}{n ^ {2}}) \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ] + (2 \frac {L + \ell}{n} \eta + 8 \frac {L ^ {2}}{n ^ {2}} \eta^ {2}) \mathbb {E} [ \Gamma_ {t} ]. \\ \end{array}
$$

For the second one we have that:

$$
\mathbb {E} _ {t} \left\| F _ {t} ^ {\prime} - F _ {t} \right\| ^ {2} = \frac {1}{n (n - 1)} \sum_ {i} \sum_ {i \neq j} \mathbb {E} _ {t} \left\| \frac {2 \eta}{n ^ {2}} \sum_ {r} \nabla f ^ {r} (X _ {t} ^ {r}) - \frac {\eta}{n} (G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j})) \right\| ^ {2}
$$

$$
\leq \frac {4 \eta^ {2}}{n ^ {3}} \sum_ {i} \mathbb {E} _ {t} \left\| \frac {1}{n} \sum_ {r} \nabla f ^ {r} (X _ {t} ^ {r}) - G ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} = \frac {4 \eta^ {2}}{n ^ {3}} \sum_ {i} \mathbb {E} _ {t} \left\| \frac {1}{n} \sum_ {r} \nabla f ^ {r} (X _ {t} ^ {r}) - \nabla f ^ {i} (X _ {t} ^ {i}) + \nabla f ^ {i} (X _ {t} ^ {i}) - G ^ {i} (X _ {t} ^ {i}) \right\| ^ {2}
$$

$$
\leq \frac {8 \eta^ {2}}{n ^ {3}} \sum_ {i} \Big (\mathbb {E} _ {t} \left\| \frac {1}{n} \sum_ {r} \nabla f ^ {r} (X _ {t} ^ {r}) - \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} + \mathbb {E} _ {t} [ (\sigma_ {t} ^ {i}) ^ {2} ] \Big)
$$

$$
\leq \frac {8 \eta^ {2}}{n ^ {3}} \Big (\sum_ {i} \mathbb {E} _ {t} \left\| \frac {1}{n - 1} \sum_ {r \neq i} [ \nabla f ^ {r} (X _ {t} ^ {r}) - \nabla f ^ {i} (X _ {t} ^ {i}) ] \right\| ^ {2} + \mathbb {E} _ {t} [ (\sigma_ {t} ^ {i}) ^ {2} ] \Big)
$$

$$
\leq \frac {8 \eta^ {2}}{n ^ {3} (n - 1)} \sum_ {i} \sum_ {r \neq i} \mathbb {E} _ {t} \left\| \nabla f ^ {r} (X _ {t} ^ {r}) - \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} + \frac {8 \eta^ {2}}{n ^ {2}} \mathbb {E} _ {t} [ (\overline {{\sigma}} _ {t}) ^ {2} ]
$$

$$
\begin{array}{l} \leq \frac {8 \eta^ {2}}{n ^ {3} (n - 1)} \sum_ {i} \sum_ {r \neq i} \mathbb {E} _ {t} \| [ \nabla f ^ {r} \left(X _ {t} ^ {r}\right) - \nabla f ^ {r} (\mu_ {t}) ] + [ \nabla f ^ {r} (\mu_ {t}) - \nabla f (\mu_ {t}) ] \\ \left. + \left[ \nabla f (\mu_ {t}) - \nabla f ^ {i} (\mu_ {t}) \right] + \left[ \nabla f ^ {i} (\mu_ {t}) - \nabla f ^ {i} (X _ {t} ^ {i}) \right] \right\| ^ {2} + \frac {8 \eta^ {2}}{n ^ {2}} \mathbb {E} _ {t} [ (\overline {{\sigma}} _ {t}) ^ {2} ] \\ \end{array}
$$

$$
\leq \frac {8 \eta^ {2}}{n ^ {3} (n - 1)} \sum_ {i} 8 (n - 1) \Big (\mathbb {E} _ {t} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) - \nabla f ^ {i} (\mu_ {t}) \right\| ^ {2} + \mathbb {E} _ {t} \left\| \nabla f ^ {i} (\mu_ {t}) - \nabla f (\mu_ {t}) \right\| ^ {2} \Big) + \frac {8 \eta^ {2}}{n ^ {2}} \mathbb {E} _ {t} [ (\overline {{\sigma}} _ {t}) ^ {2} ]
$$

$$
\leq \frac {6 4 \eta^ {2}}{n ^ {3}} \sum_ {i} \mathbb {E} _ {t} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) - \nabla f ^ {i} (\mu_ {t}) \right\| ^ {2} + \frac {6 4 \eta^ {2}}{n ^ {3}} \sum_ {i} \mathbb {E} _ {t} \left\| \nabla f ^ {i} (\mu_ {t}) - \nabla f (\mu_ {t}) \right\| ^ {2} + \frac {8 \eta^ {2}}{n ^ {2}} \mathbb {E} _ {t} [ (\overline {{\sigma}} _ {t}) ^ {2} ]
$$

$$
\leq \frac {6 4 L ^ {2} \eta^ {2}}{n ^ {3}} \sum_ {i} \mathbb {E} _ {t} \| X _ {t} ^ {i} - \mu_ {t} \| ^ {2} + \frac {6 4 \eta^ {2} (\varsigma_ {0} ^ {2} n _ {0} + \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}} + \frac {8 \eta^ {2}}{n ^ {2}} \mathbb {E} _ {t} [ (\overline {{\sigma}} _ {t}) ^ {2} ]
$$

$$
= \frac {6 4 L ^ {2} \eta^ {2}}{n ^ {2}} \mathbb {E} _ {t} [ \Gamma_ {t} ] + \frac {6 4 \eta^ {2} (\varsigma_ {0} ^ {2} n _ {0} + \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}} + \frac {8 \eta^ {2}}{n ^ {2}} \mathbb {E} _ {t} [ (\overline {{\sigma}} _ {t}) ^ {2} ].
$$

Next, we remove conditioning and use Lemma 9 to get

$$
\mathbb {E} \left\| F _ {t} ^ {\prime} - F _ {t} \right\| ^ {2} = \mathbb {E} \left[ \mathbb {E} _ {t} \left\| F _ {t} ^ {\prime} - F _ {t} \right\| ^ {2} \right] \leq \frac {6 4 L ^ {2} \eta^ {2}}{n ^ {2}} \mathbb {E} [ \Gamma_ {t} ] + \frac {6 4 \eta^ {2} (\varsigma_ {0} ^ {2} n _ {0} + \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}} + \frac {8 \eta^ {2}}{n ^ {2}} \mathbb {E} [ (\overline {{\sigma}} _ {t}) ] ^ {2}
$$

$$
\leq \frac {6 4 L ^ {2} \eta^ {2}}{n ^ {2}} \mathbb {E} [ \Gamma_ {t} ] + \frac {6 4 \eta^ {2} (\varsigma_ {0} ^ {2} n _ {0} + \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}}
$$

$$
+ \frac {8 \eta^ {2}}{n ^ {2}} \left(1 2 (d + 4) L ^ {2} \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {1 2 n _ {0} (d + 4) \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + 6 (4 d + 1 7) L \left(f \left(\mu_ {t}\right) - f \left(x ^ {*}\right)\right) \right.
$$

$$
\left. + \frac {4 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {3 n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}\right)
$$

$$
= \frac {\eta^ {2} L ^ {2} (9 6 d + 4 4 8)}{n ^ {2}} \mathbb {E} [ \Gamma_ {t} ] + \frac {\eta^ {2} ((9 6 d + 4 4 8) \varsigma_ {0} ^ {2} n _ {0} + 8 8 \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}}
$$

$$
+ \frac {4 8 \eta^ {2} (4 d + 1 7) L \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ]}{n ^ {2}} + \frac {8 \eta^ {2} ((d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {3}} + \frac {1 2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {3} c ^ {2}}. \tag {37}
$$

Now consider the last one. We have:

$$
\mathbb {E} _ {t} \left\langle \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime}, F _ {t} ^ {\prime} - F _ {t} \right\rangle = \left\langle \mu_ {t} - F _ {t} ^ {\prime} - x ^ {*}, \mathbb {E} _ {t} (F _ {t} ^ {\prime} - F _ {t}) \right\rangle
$$

Cauchy-Schwarz

$$
\leq \left\| \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime} \right\| \cdot \left\| \mathbb {E} _ {t} (F _ {t} ^ {\prime} - F _ {t}) \right\| \tag {38}
$$

$$
\leq \left[ \| \mu_ {t} - x ^ {*} \| + \| F _ {t} ^ {\prime} \| \right] \cdot \underbrace {\left| \left| \mathbb {E} _ {t} (F _ {t} ^ {\prime} - F _ {t}) \right| \right|} _ {R _ {3} :=}
$$

$$
\begin{array}{l} R _ {3} = \left\| \mathbb {E} _ {t} (F _ {t} ^ {\prime} - F _ {t}) \right\| = \left\| \frac {2 \eta}{n ^ {2}} \sum_ {i} \nabla f ^ {i} (X _ {t} ^ {i}) - \frac {2 \eta}{n ^ {2}} \mathbb {E} _ {t} \left[ G ^ {i} (X _ {t} ^ {i}) \right] \right\| \\ \leq \frac {2 \eta}{n ^ {2}} \sum_ {i} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) - \mathbb {E} _ {t} \left[ G ^ {i} (X _ {t} ^ {i}) \right] \right\| \leq \frac {2 \eta^ {2}}{n ^ {2}} \sum_ {i} b _ {i} \stackrel {1 2} {=} \frac {2 \eta^ {2}}{n} B \tag {39} \\ \end{array}
$$

By using the inequality above in (38) and taking expectation from both sides we get

$$
\mathbb {E} \Big \langle \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime}, F _ {t} ^ {\prime} - F _ {t} \Big \rangle = \mathbb {E} _ {X _ {t} ^ {1}, X _ {t} ^ {2}, \ldots , X _ {t} ^ {n}} \left[ \mathbb {E} _ {t} \Big \langle \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime}, F _ {t} ^ {\prime} - F _ {t} \Big \rangle \right]
$$

$$
\leq \frac {2 \eta^ {2}}{n} B \big (\mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\| + \mathbb {E} \left\| F _ {t} ^ {\prime} \right\| \big) \stackrel {(*)} {\leq} 2 \frac {\eta^ {2}}{n} B \big (\mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\| ^ {2} + \frac {1}{4} + \mathbb {E} \left\| F _ {t} ^ {\prime} \right\| ^ {2} + \frac {1}{4} \big)
$$

$$
\stackrel {(3 2)} {\leq} 2 \frac {\eta^ {2}}{n} B \left(\mathbb {E} \| \mu_ {t} - x ^ {*} \| ^ {2} + \frac {4 \eta^ {2}}{n ^ {2}} \mathbb {E} \left\| \frac {1}{n} \sum_ {i} \nabla f ^ {i} \left(X _ {t} ^ {i}\right) \right\| ^ {2} + 0. 5\right) \tag {40}
$$

$$
\leq 2 \frac {\eta^ {2}}{n} B \left(\mathbb {E} \| \mu_ {t} - x ^ {*} \| ^ {2} + \frac {4 \eta^ {2}}{n ^ {2}} \mathbb {E} \left[ \frac {2 L ^ {2}}{n} \sum_ {i} \left\| X _ {t} ^ {i} - \mu_ {t} \right\| ^ {2} + 4 L [ f (\mu_ {t}) - f (x ^ {*}) ] \right] + 0. 5\right)
$$

$$
\leq \eta^ {2} \frac {2 B}{n} \mathbb {E} \| \mu_ {t} - x ^ {*} \| ^ {2} + \eta^ {4} \frac {1 6 B L ^ {2}}{n ^ {3}} \mathbb {E} [ \Gamma_ {t} ] + \eta^ {4} \frac {3 2 B L}{n ^ {3}} \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ] + \eta^ {2} \frac {B}{n}.
$$

To get the (\*) inequality, we first used Young's inequality twice with $\alpha = \frac{1}{2}$ , to get that

$$
\mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\| + \mathbb {E} \left\| F _ {t} ^ {\prime} \right\| \leq (\mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\|) ^ {2} + \frac {1}{4} + (\mathbb {E} \left\| F _ {t} ^ {\prime} \right\|) ^ {2} + \frac {1}{4}
$$

and then applied Jensen's inequality to get

$$
(\mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\|) ^ {2} + \frac {1}{4} + (\mathbb {E} \left\| F _ {t} ^ {\prime} \right\|) ^ {2} + \frac {1}{4} \leq \mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\| ^ {2} + \frac {1}{4} + \mathbb {E} \left\| F _ {t} ^ {\prime} \right\| ^ {2} + \frac {1}{4}.
$$

Then following by (35), (36) and (40), the latter inequality (30) would be:

$$
\mathbb {E} \left\| \mu_ {t + 1} - x ^ {*} \right\| ^ {2} = \mathbb {E} \left\| \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime} \right\| ^ {2} + \mathbb {E} _ {X _ {t} ^ {1}, X _ {t} ^ {2}, \dots , X _ {t} ^ {n}} \left[ \mathbb {E} _ {t} \| F _ {t} ^ {\prime} - F _ {t} \| ^ {2} \right]
$$

$$
+ 2 \mathbb {E} _ {X _ {t} ^ {1}, X _ {t} ^ {2}, \dots , X _ {t} ^ {n}} \left[ \mathbb {E} _ {t} \left\langle \mu_ {t} - x ^ {*} - F _ {t} ^ {\prime}, F _ {t} ^ {\prime} - F _ {t} \right\rangle \right]
$$

$$
\leq \left(1 - \frac {\ell \eta}{n}\right) \mathbb {E} \| \mu_ {t} - x ^ {*} \| ^ {2} - \left(4 \frac {\eta}{n} - 1 6 L \frac {\eta^ {2}}{n ^ {2}}\right) \mathbb {E} \left[ f \left(\mu_ {t}\right) - f \left(x ^ {*}\right) \right] + \left(2 \frac {L + \ell}{n} \eta + 8 \frac {L ^ {2}}{n ^ {2}} \eta^ {2}\right) \mathbb {E} \left[ \Gamma_ {t} \right]
$$

$$
+ \frac {\eta^ {2} L ^ {2} (9 6 d + 4 4 8)}{n ^ {2}} \mathbb {E} [ \Gamma_ {t} ] + \frac {\eta^ {2} ((9 6 d + 4 4 8) \varsigma_ {0} ^ {2} n _ {0} + 8 8 \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}}
$$

$$
+ \frac {4 8 \eta^ {2} (4 d + 1 7) L   \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ]}{n ^ {2}} + \frac {8 \eta^ {2} ((d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {3}} + \frac {1 2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {3} c ^ {2}}.
$$

$$
+ \eta^ {2} \frac {4 B}{n} \mathbb {E} \| \mu_ {t} - x ^ {*} \| ^ {2} + \eta^ {4} \frac {3 2 B L ^ {2}}{n ^ {3}} \mathbb {E} [ \Gamma_ {t} ] + \eta^ {4} \frac {6 4 B L}{n ^ {3}} \mathbb {E} [ f (\mu_ {t}) - f (x ^ {*}) ] + \eta^ {2} \frac {2 B}{n}.
$$

Hence:

$$
\mathbb {E} \left\| \mu_ {t + 1} - x ^ {*} \right\| ^ {2} \leq (1 - \frac {\ell \eta}{n} + \eta^ {2} \frac {4 B}{n}) \mathbb {E} \left\| \mu_ {t} - x ^ {*} \right\| ^ {2} - (4 \frac {\eta}{n} - \eta^ {2} \frac {1 6 L (1 2 d + 5 2)}{n ^ {2}} - \eta^ {4} \frac {6 4 B L}{n ^ {3}}) \mathbb {E} \left[ f (\mu_ {t}) - f (x ^ {*}) \right]
$$

$$
+ \left(2 \frac {L + \ell}{n} \eta + \eta^ {2} L ^ {2} \frac {9 6 d + 4 5 6}{n ^ {2}} + \eta^ {4} \frac {3 2 B L ^ {2}}{n ^ {3}}\right) \mathbb {E} [ \Gamma_ {t} ]
$$

$$
+ \frac {\eta^ {2} ((9 6 d + 4 4 8) \varsigma_ {0} ^ {2} n _ {0} + 8 8 \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}} + \frac {8 \eta^ {2} ((d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {3}} + \frac {1 2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {3} c ^ {2}} + \eta^ {2} \frac {2 B}{n}.
$$

We get the proof of the Lemma by plugging $\eta \leq \frac{\sqrt{\ell cn}}{2\sqrt{Ln_0}(d + 3)^{\frac{3}{4}}}$ and $B\leq \frac{\eta n_0}{2cn} L(d + 3)^{\frac{3}{2}}$ (Lemma 10) in the above inequality.

Theorem 1.1 (Convex case of Theorem 1). Assume that the functions $f$ and $f_{i}$ satisfy assumptions 1, 2, 3 and 4. Let $T$ to be large enough such that $\frac{T}{\log T} = \Omega \left( \frac{n(d + n)(L + 1)\left(\frac{1}{\ell} + 1\right)}{\ell} \right)$ , and let the learning rate be $\eta = \frac{4n\log T}{T\ell}$ . For $1 \leq t \leq T$ ,

let the sequence of weights $w_{t}$ be given by $w_{t} = \left(1 - \frac{\eta\ell}{2n}\right)^{-t}$ and let $S_{T} = \sum_{t=1}^{T} w_{T}$ . Finally, define $\mu_{t} = \sum_{i=1}^{n} X_{t}^{i}/n$ and $y_{T} = \sum_{t=1}^{T} \frac{w_{t}\mu_{t-1}}{S_{T}}$ to be the mean over local model parameters. Then, we can show that HDO provides the following convergence rate:

$$
\begin{array}{l} \mathbb {E} [ f (y _ {T}) - f (x ^ {*}) ] + \frac {\ell \mathbb {E} \| \mu_ {T} - x ^ {*} \| ^ {2}}{8} \\ = O \left(\frac {L \| \mu_ {0} - x ^ {*} \| ^ {2}}{T \log T} + \frac {\log (T) (d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{T \ell n} \right. \\ \left. + \frac {\log (T) (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{T \ell n} + \frac {\log (T) d n _ {0}}{T \ell n}\right). \\ \end{array}
$$

Proof. Let $a_{t}^{2} := E \left\| \mu_{t} - x^{*} \right\|^{2}$ , $e_{t} := E \left[ f(\mu_{t}) - f(x^{*}) \right]$ and

$$
C _ {1} := 4 \frac {\eta}{n} - \eta^ {2} \frac {1 6 L (1 2 d + 5 2)}{n ^ {2}} - \eta^ {4} \frac {6 4 B L}{n ^ {3}} \geq 4 \frac {\eta}{n} - \eta^ {2} \frac {1 6 L (1 2 d + 5 2)}{n ^ {2}} - \eta^ {5} \frac {3 2 n _ {0} L ^ {2} (d + 3) ^ {\frac {3}{2}}}{d ^ {\frac {1}{2}} n ^ {4}},
$$

$$
C _ {2} := 2 \frac {L + \ell}{n} \eta + \eta^ {2} L ^ {2} \frac {9 6 d + 4 5 6}{n ^ {2}} + \eta^ {4} \frac {3 2 B L ^ {2}}{n ^ {3}} \leq 2 \frac {L + \ell}{n} \eta + \eta^ {2} L ^ {2} \frac {9 6 d + 4 5 6}{n ^ {2}} + \eta^ {5} \frac {1 6 n _ {0} L ^ {3} (d + 3) ^ {\frac {3}{2}}}{d ^ {\frac {1}{2}} n ^ {4}}
$$

$$
C _ {3} := \frac {\eta^ {2} ((9 6 d + 4 4 8) \varsigma_ {0} ^ {2} n _ {0} + 8 8 \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}} + \frac {8 \eta^ {2} ((d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {3}} + \frac {1 2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {3} c ^ {2}} + \frac {2 B \eta^ {2}}{n}
$$

$$
\leq \frac {\eta^ {2} ((9 6 d + 4 4 8) \varsigma_ {0} ^ {2} n _ {0} + 8 8 \varsigma_ {1} ^ {2} n _ {1})}{n ^ {3}} + \frac {8 \eta^ {2} ((d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {3}} + \frac {1 2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {3} d} + \frac {L (d + 3) ^ {\frac {3}{2}} n _ {0} \eta^ {3}}{d ^ {\frac {1}{2}} n ^ {2}}.
$$

Where, in the above inequalities, we used Lemma 10. Therefore, the recursion from Lemma 13 can be written as

$$
a _ {t} ^ {2} \leq (1 - \frac {\ell \eta}{2 n}) a _ {t - 1} ^ {2} - C _ {1} e _ {t - 1} + C _ {2} \mathbb {E} [ \Gamma_ {t - 1} ] + C _ {3}.
$$

Then, we multiply the above recursion by $w_{t} = (1 - \frac{\eta\ell}{2n})^{-t}$ and

$$
w _ {t} a _ {t} ^ {2} \leq w _ {t} \Big ((1 - \frac {\ell \eta}{2 n}) a _ {t - 1} ^ {2} - C _ {1} e _ {t - 1} + C _ {2} \mathbb {E} [ \Gamma_ {t - 1} ] + C _ {3} \Big) = w _ {t - 1} a _ {t - 1} ^ {2} - w _ {t} C _ {1} e _ {t - 1} + w _ {t} C _ {2} \mathbb {E} [ \Gamma_ {t - 1} ] + w _ {t} C _ {3}.
$$

By summing the above inequality for $t \in \{1, 2, ..., T\}$ and cancelling and rearrange terms we get:

$$
w _ {T} a _ {T} ^ {2} \leq w _ {0} a _ {0} ^ {2} - C _ {1} \sum_ {t = 1} ^ {T} w _ {t} e _ {t - 1} + C _ {2} \sum_ {t = 1} ^ {T} w _ {t} \mathbb {E} [ \Gamma_ {t - 1} ] + C _ {3} \sum_ {t = 1} ^ {T} w _ {t}.
$$

By using Lemma 12 in the inequality above we get that

$$
\begin{array}{l} w _ {T} a _ {T} ^ {2} \leq w _ {0} a _ {0} ^ {2} - C _ {1} \sum_ {t = 1} ^ {T} w _ {t} e _ {t - 1} + 1 2 0 C _ {2} \eta^ {2} (2 d + 9) L \sum_ {t = 1} ^ {T - 1} w _ {t} \mathbb {E} [ f (\mu_ {t - 1}) - f (x ^ {*}) ] \\ + C _ {2} \Big (\frac {6 0 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {2 0 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {1 0 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n c ^ {2}} \Big) \sum_ {t = 1} ^ {T - 1} w _ {t} \\ + C _ {3} \sum_ {t = 1} ^ {T} w _ {t} \\ \leq w _ {0} a _ {0} ^ {2} - \underbrace {(C _ {1} - 1 2 0 C _ {2} \eta^ {2} (2 d + 9) L)} _ {D _ {1} :=} \sum_ {t = 1} ^ {T} w _ {t} e _ {t - 1} \\ + \underbrace {\left(C _ {2} \left(\frac {6 0 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {2 0 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {1 0 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n c ^ {2}}\right) + C _ {3}\right)} _ {D _ {2} :=} \sum_ {t = 1} ^ {T} w _ {t}. \tag {41} \\ \end{array}
$$

Let $S_T := \sum_{t=1}^T w_t$ and $y_T := \sum_{t=1}^T \frac{w_t \mu_{t-1}}{S_T}$ , then by convexity of $f$ we have that

$$
\mathbb {E} [ f (y _ {T}) - f (x ^ {*}) ] \leq \frac {1}{S _ {T}} \sum_ {t = 1} ^ {T} w _ {t} e _ {t - 1}. \tag {42}
$$

By choosing small enough $\eta$ , so that have $D_{1} \geq 0$ , we can combine inequalities (41) and (42) to get

$$
\mathbb {E} \left[ f \left(y _ {T}\right) - f \left(x ^ {*}\right) \right] + \frac {w _ {T} a _ {T} ^ {2}}{S _ {T} D _ {1}} \leq \frac {w _ {0} a _ {0} ^ {2}}{S _ {T} D _ {1}} + \frac {D _ {2}}{D _ {1}} \tag {43}
$$

Our first goal is to lower bound $D_{1}$ , we aim to choose upper bound on $\eta$ so that $D_{1} = \Omega(\frac{\eta}{n})$ . For this it will be enough to set $\eta = O\left(\frac{1}{d(L + \ell + 1)}\right)$ . Since $C_{1} \geq 4\frac{\eta}{n} - \eta^{2}\frac{16L(12d + 52)}{n^{2}} - \eta^{4}\frac{32n_{0}L^{2}(d + 3)^{\frac{3}{2}}}{cn^{4}}$ , we have $C_{1} = \Omega(\frac{\eta}{n})$ . Plus, $C_{2} = O(1/n)$ and hence $120C_{2}\eta^{2}(2d + 9)L = O(\frac{\eta}{n})$ , thus $D_{1} = \Omega(\frac{\eta}{n})$ , as desired. Also:

$$
S _ {t} = \sum_ {t = 1} ^ {T} w _ {t} = \sum_ {t = 1} ^ {T} (1 - \frac {\eta \ell}{2 n}) ^ {- t} \geq (1 - \frac {\eta \ell}{2 n}) ^ {- T} \geq e ^ {\frac {\eta \ell T}{2 n}}.
$$

By setting $\eta = \frac{4n\log(T)}{T\ell}$ , we get $S_{t} \geq T^{2}$ . Therefore, we have

$$
\frac {w _ {0} a _ {0} ^ {2}}{S _ {T} D _ {1}} = O \left(\frac {w _ {0} a _ {0} ^ {2} \ell}{T \log (T)}\right) = O \left(\frac {L \| \mu_ {0} - x ^ {*} \| ^ {2}}{T \log (T)}\right). \tag {44}
$$

Next, we upper bound $D_2$ . Since $\eta = O\big(\frac{1}{d(L + \ell + 1)}\big)$ we get that $C_2 = O((L + \ell)\eta / n)$ . Additionally,

$$
\begin{array}{l} \frac {6 0 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {2 0 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {1 0 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n d} \\ = O \left(\frac {\eta^ {2} (d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {\eta^ {2} (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{n} + \frac {\eta^ {2} n _ {0}}{n}\right). \\ \end{array}
$$

By using $\eta = O\left(\frac{1}{(L + \ell + 1)n}\right)$ in the equation above, we get $C_2 = O\left(\frac{1}{n^2}\right)$ and hence

$$
\begin{array}{l} C _ {2} \left(\frac {6 0 \eta^ {2} (2 (d + 4) n _ {0} s _ {0} ^ {2} + n _ {1} s _ {1} ^ {2})}{n} + \frac {2 0 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {1 0 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n d}\right) \\ = O \Bigg (\frac {\eta^ {2} (d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n ^ {3}} + \frac {\eta^ {2} (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{n ^ {3}} + \frac {\eta^ {2} n _ {0}}{n ^ {3}} \Bigg). \\ \end{array}
$$

Finally, we have that

$$
C _ {3} = O \left(\frac {\eta^ {2} (d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n ^ {3}} + \frac {\eta^ {2} (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{n ^ {3}} + \frac {\eta^ {2} n _ {0}}{n ^ {3}} + \frac {\eta^ {3} d L n _ {0}}{n ^ {2}}\right).
$$

If we put together the above inequalities we get that

$$
D _ {2} = O \Bigg (\frac {\eta^ {2} (d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n ^ {3}} + \frac {\eta^ {2} (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{n ^ {3}} + \frac {\eta^ {3} L d n _ {0}}{n ^ {2}} \Bigg).
$$

By plugging $D_{1} = \Omega \left(\frac{\eta}{n}\right)$ and $\eta = \frac{4n\log(T)}{T\ell}$ , in the equation above we get that

$$
\begin{array}{l} \frac {D _ {2}}{D _ {1}} = O \left(\frac {\eta (d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n ^ {2}} + \frac {\eta (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{n ^ {2}} + \frac {\eta^ {2} L d n _ {0}}{n}\right) \\ = O \Bigg (\frac {\log (T) (d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{T \ell n} + \frac {\log (T) (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{T \ell n} + \frac {\log (T) d n _ {0}}{T \ell n} \Bigg). \\ \end{array}
$$

We also have that $D_{1} \leq C_{1} \leq \frac{4\eta}{n}$ , and

$$
\frac {w _ {T}}{S _ {T}} = \frac {(1 - \frac {\eta \ell}{2 n}) ^ {- T}}{\sum_ {t = 1} ^ {T} (1 - \frac {\eta \ell}{2 n}) ^ {- t}} \geq (1 - \frac {\eta \ell}{2 n}) \Big (1 - \frac {\eta \ell}{2 n}) ^ {- 1} - 1 \Big) = \frac {\eta \ell}{2 n}.
$$

Hence, $\frac{w_{T}}{S_{T}D_{1}} \geq \frac{\ell}{8}$ . By putting together the above above inequalities we get the final convergence bound:

$$
\begin{array}{l} \mathbb {E} [ f (y _ {T}) - f (x ^ {*}) ] + \frac {\ell \mathbb {E} \| \mu_ {T} - x ^ {*} \| ^ {2}}{8} = \mathbb {E} [ f (y _ {T}) - f (x ^ {*}) ] + \frac {\ell a _ {T} ^ {2}}{8} \leq \mathbb {E} [ f (y _ {T}) - f (x ^ {*}) ] + \frac {w _ {T} a _ {T} ^ {2}}{S _ {T} D _ {1}} \leq \frac {w _ {0} a _ {0} ^ {2}}{S _ {T} D _ {1}} + \frac {D _ {2}}{D _ {1}} \\ = O \left(\frac {L \| \mu_ {0} - x ^ {*} \| ^ {2}}{T \log (T)}\right) + \frac {\log (T) (d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{T \ell n} + \frac {\log (T) (d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2})}{T \ell n} + \frac {\log (T) d n _ {0}}{T \ell n}\left. \right). \\ \end{array}
$$

Finally, we need to gather all upper bounds on $\eta$ and compute lower bound on $T$ so that the upper bounds are satisfied. We need $\eta = O\left(\frac{1}{d(L + \ell + 1)}\right)$ , $\eta = O\left(\frac{1}{n(L + \ell + 1)}\right)$ in the proof of this theorem, the lemmas we used require $\eta \leq \frac{1}{10\ell}$ and

$$
\eta \leq \frac {\sqrt {\ell c n}}{2 \sqrt {L n _ {0}} (d + 3) ^ {\frac {3}{4}}} = O \left(\frac {\ell}{L d}\right).
$$

Considering $\ell \leq L$ , we get that we need $\eta = O\left(\frac{1}{(d + n)(L + 1)(\frac{1}{\ell} + 1)}\right)$ . Thus, we need $\frac{T}{\log(T)} = \Omega\left(\frac{n(d + n)(L + 1)(\frac{1}{\ell} + 1)}{\ell}\right)$ .

![](images/d575e219ed61212dcc87a8d1052153b12a1343dcabc5dbb1b8ea8d5ea8a54a13.jpg)

# Proof of Non-Convex Case of Theorem 1

Lemma 14. For any time step $t$ , we have

$$
\mathbb {E} \left\| \mu_ {t + 1} - \mu_ {t} \right\| ^ {2} \leq \frac {4 \eta^ {2}}{n ^ {2}} \mathbb {E} [ M _ {t} ^ {G} ].
$$

Proof.

$$
\mathbb {E} \left\| \mu_ {t + 1} - \mu_ {t} \right\| ^ {2} = \frac {1}{n (n - 1)} \sum_ {i} \sum_ {j \neq i} \mathbb {E} \left\| \frac {\eta}{n} (G ^ {i} (X _ {t} ^ {i}) + G ^ {j} (X _ {t} ^ {j})) \right\| ^ {2}
$$

$$
\stackrel {C a u c h y - S c h w a r z} {\leq} \frac {2 \eta^ {2}}{n ^ {3} (n - 1)} \sum_ {i} \sum_ {j \neq i} \mathbb {E} (\| G ^ {i} (X _ {t} ^ {i}) \| ^ {2} + \| G ^ {j} (X _ {t} ^ {j}) \| ^ {2})
$$

$$
= \frac {4 \eta^ {2}}{n ^ {3}} \sum_ {i} \mathbb {E} \left\| G ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} = \frac {4 \eta^ {2}}{n ^ {2}} \mathbb {E} [ M _ {t} ^ {G} ]
$$

Lemma 15. For any time step $t$ and constants $\alpha_0 \geq \alpha_1 > 0$ let $M_t^f(\alpha_0, \alpha_1) = \frac{\alpha_0}{n} \sum_{i \in N_0} \| \nabla f^i(X_t^i) \|^2 + \frac{\alpha_1}{n} \sum_{i \in N_1} \| \nabla f^i(X_t^i) \|^2$ . We have that:

$$
\mathbb {E} [ M _ {t} ^ {f} (\alpha_ {0}, \alpha_ {1}) ] \leq 3 L ^ {2} \alpha_ {0} \mathbb {E} [ \Gamma_ {t} ] + \frac {3 \alpha_ {0} n _ {0} \varsigma_ {0} ^ {2} + 3 \alpha_ {1} n _ {1} \varsigma_ {1} ^ {2}}{n} + 3 \frac {\alpha_ {0} n _ {0} + \alpha_ {1} n _ {1}}{n} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}.
$$

Proof.

$$
\frac {\alpha_ {0}}{n} \sum_ {i \in N _ {0}} \mathbb {E} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} = \frac {\alpha_ {0}}{n} \sum_ {i \in N _ {0}} \mathbb {E} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) - \nabla f ^ {i} (\mu_ {t}) + \nabla f ^ {i} (\mu_ {t}) - \nabla f (\mu_ {t}) + \nabla f (\mu_ {t}) \right\| ^ {2}
$$

$$
\leq \frac {3 L ^ {2} \alpha_ {0}}{n} \sum_ {i \in N _ {0}} \mathbb {E} \| X _ {t} ^ {i} - \mu_ {t} \| ^ {2} + \frac {3 \alpha_ {0} n _ {0} \varsigma_ {0} ^ {2}}{n} + 3 \frac {\alpha_ {0} n _ {0}}{n} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}. \tag {45}
$$

Similarly, in the case of first-order nodes we get:

$$
\frac {\alpha_ {1}}{n} \sum_ {i \in N _ {1}} \mathbb {E} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} \leq \frac {3 L ^ {2} \alpha_ {1}}{n} \sum_ {i \in N _ {1}} \mathbb {E} \left\| X _ {t} ^ {i} - \mu_ {t} \right\| ^ {2} + \frac {3 \alpha_ {1} n _ {1} \varsigma_ {1} ^ {2}}{n} + 3 \frac {\alpha_ {1} n _ {1}}{n} \mathbb {E} \left\| \nabla f (\mu_ {t}) \right\| ^ {2}. \tag {46}
$$

By summing up the above inequalities and using the fact that $\alpha_0 \geq \alpha_1$ (together with the definition of $\Gamma_t$ ), we get the proof of the lemma.

Lemma 16. Assume $\nu := \frac{\eta}{c}$ is fixed, where $\eta$ and $c$ are the learning rate and a constant respectively. Then, for any time step $t$ we have:

$$
\begin{array}{l} \mathbb {E} \left[ M _ {t} ^ {G} \right] \leq 6 (d + 4) L ^ {2} \mathbb {E} [ \Gamma_ {t} ] + \frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + \frac {6 (d + 4) n _ {0} + 3 n _ {1}}{n} \mathbb {E} \left\| \nabla f (\mu_ {t}) \right\| ^ {2} \\ + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}. \\ \end{array}
$$

Proof.

$$
\mathbb {E} _ {t} \left[ M _ {t} ^ {G} \right] = \frac {1}{n} \sum_ {i} \mathbb {E} _ {t} \left\| G ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} \overset {(6)} {\leq} \frac {1}{n} \sum_ {i \in N _ {0}} (\frac {1}{2} \nu^ {2} L ^ {2} (d + 6) ^ {3} + 2 (d + 4) \left[ \mathbb {E} _ {t} \left\| \nabla f ^ {i} (X _ {t} ^ {i}) \right\| ^ {2} + s _ {i} ^ {2} \right])
$$

$$
+ \frac {1}{n} \sum_ {i \in N _ {1}} (\mathbb {E} _ {t} \| \nabla f ^ {i} (X _ {t} ^ {i}) \| ^ {2} + s _ {i} ^ {2}) \tag {47}
$$

$$
= \mathbb {E} _ {t} [ M _ {t} ^ {f} (2 (d + 4), 1) ] + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}.
$$

Next, we take expectation with respect to $X_{t}^{1}, X_{t}^{2}, \ldots, X_{t}^{n}$ and use Lemma 15 to get

$$
\mathbb {E} \left[ M _ {t} ^ {G} \right] \leq 6 (d + 4) L ^ {2} \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + \frac {6 (d + 4) n _ {0} + 3 n _ {1}}{n} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \tag {48}
$$

$$
+ \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3},
$$

which finishes the proof of the lemma.

Lemma 17. For any time step $t$ and fixed learning rate $\eta \leq \frac{1}{10L(d + 4)^{\frac{1}{2}}}$

$$
\mathbb {E} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{4 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {1 2 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n ^ {2}} + \frac {4 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n ^ {2}} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}
$$

$$
+ \frac {4 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {2}} + \frac {2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {2} c ^ {2}}.
$$

Proof. Note that Lemma 2 does not assume anything about convexity. Therefore, we have:

$$
\mathbb {E} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {4}{n} \eta^ {2} \mathbb {E} \left[ M _ {t} ^ {G} \right].
$$

Now, by using Lemma 16 in the inequality above we have

$$
\begin{array}{l} \mathbb {E} \left[ \Gamma_ {t + 1} \right] \leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {4}{n} \eta^ {2} \mathbb {E} \left[ M _ {t} ^ {G} \right] \\ \leq \left(1 - \frac {1}{2 n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {4 \eta^ {2}}{n} \Bigg (6 (d + 4) L ^ {2} \mathbb {E} [ \Gamma_ {t} ] + \frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + \frac {6 (d + 4) n _ {0} + 3 n _ {1}}{n} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \\ \left. + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}\right) \\ = \left(1 - \frac {1}{2 n} + \frac {2 4 \eta^ {2} L ^ {2} (d + 4)}{n}\right) \mathbb {E} \left[ \Gamma_ {t} \right] + \frac {1 2 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n ^ {2}} + \frac {4 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n ^ {2}} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \\ + \frac {4 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {2}} + \frac {2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {2} c ^ {2}}. \\ \end{array}
$$

We get the proof of the lemma by using $\eta \leq \frac{1}{10L(d + 4)^{\frac{1}{2}}}$ in the above inequality.

Lemma 18.

$$
\begin{array}{l} \sum_ {t = 0} ^ {T - 1} \mathbb {E} [ \Gamma_ {t} ] \leq T \bigg (\frac {4 8 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {8 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n c ^ {2}} \bigg) \\ + \frac {1 6 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n} \sum_ {t = 0} ^ {T - 2} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}. \\ \end{array}
$$

Proof. Using 17 we can write:

$$
\begin{array}{l} \mathbb {E} [ \Gamma_ {t} ] \leq \bigg (1 + (1 - \frac {1}{4 n}) + \dots + (1 - \frac {1}{4 n}) ^ {t - 1} \bigg). \bigg (\frac {1 2 \eta^ {2} (2 (d + 4) n _ {0} s _ {0} ^ {2} + n _ {1} s _ {1} ^ {2})}{n ^ {2}} \\ \left. + \frac {4 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {2}} + \frac {2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {2} c ^ {2}}\right) \\ + \frac {4 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n ^ {2}} \sum_ {i = 0} ^ {t - 1} (1 - \frac {1}{4 n}) ^ {t - 1 - i} \mathbb {E} \| \nabla f (\mu_ {i}) \| ^ {2} + (1 - \frac {1}{4 n}) ^ {t} \underbrace {\mathbb {E} [ \Gamma_ {0} ]} _ {0} \\ \leq \left(\underbrace {\sum_ {i = 0} ^ {\infty} (1 - \frac {1}{4 n}) ^ {i}} _ {4 n}\right) \cdot \left(\frac {1 2 \eta^ {2} (2 (d + 4) n _ {0} s _ {0} ^ {2} + n _ {1} s _ {1} ^ {2})}{n ^ {2}} + \frac {4 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n ^ {2}} + \frac {2 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n ^ {2} c ^ {2}}\right) \tag {49} \\ + \frac {4 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n ^ {2}} \sum_ {i = 0} ^ {t - 1} \left(1 - \frac {1}{4 n}\right) ^ {t - 1 - i} \mathbb {E} \| \nabla f (\mu_ {i}) \| ^ {2} \\ \leq \frac {4 8 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {8 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n c ^ {2}} \\ + \frac {4 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n ^ {2}} \sum_ {i = 0} ^ {t - 1} (1 - \frac {1}{4 n}) ^ {t - 1 - i} \mathbb {E} \| \nabla f (\mu_ {i}) \| ^ {2}. \\ \end{array}
$$

Now, we sum up the inequality 49 for $t \in \{1, \dots, T - 1\}$ and we have:

$$
\begin{array}{l} \sum_ {t = 1} ^ {T - 1} \mathbb {E} [ \Gamma_ {t} ] \leq T \bigg (\frac {4 8 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {8 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n c ^ {2}} \bigg) \\ + \frac {4 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n ^ {2}} \sum_ {t = 1} ^ {T - 1} \sum_ {i = 0} ^ {t - 1} \left(1 - \frac {1}{4 n}\right) ^ {t - 1 - i} \mathbb {E} \| \nabla f (\mu_ {i}) \| ^ {2} \\ = T \left(\frac {4 8 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {8 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n c ^ {2}}\right) \\ + \frac {4 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n ^ {2}} \sum_ {t = 0} ^ {T - 2} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \underbrace {\sum_ {i = 0} ^ {T - 2 - t} \left(1 - \frac {1}{4 n}\right) ^ {i}} _ {\leq 4 n} \tag {50} \\ \leq T \bigg (\frac {4 8 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} + \frac {8 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n c ^ {2}} \bigg) \\ + \frac {1 6 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n} \sum_ {t = 0} ^ {T - 2} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}. \\ \end{array}
$$

By adding $\Gamma_0 = 0$ to the left hand side, the proof of the lemma gets finished.

Lemma 19. For any time step $t$ ,

$$
\mathbb {E} \langle \nabla f (\mu_ {t}), \mu_ {t + 1} - \mu_ {t} \rangle \leq - \frac {\eta}{n} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} + \frac {2 \eta}{n} B ^ {2} + \frac {2 L ^ {2} \eta}{n} \mathbb {E} [ \Gamma_ {t} ].
$$

Proof.

$$
\begin{array}{l} \mathbb {E} _ {t} \langle \nabla f (\mu_ {t}), \mu_ {t + 1} - \mu_ {t} \rangle = \langle \nabla f (\mu_ {t}), \mathbb {E} _ {t} [ \mu_ {t + 1} ] - \mu_ {t} \rangle = \langle \nabla f (\mu_ {t}), - \frac {\eta}{n ^ {2} (n - 1)} \sum_ {i} \sum_ {j \neq i} \big (\mathbb {E} _ {t} [ G ^ {i} (X _ {t} ^ {i}) ] + \mathbb {E} _ {t} [ G ^ {i} (X _ {t} ^ {j}) ] \big) \rangle \\ = \langle \nabla f (\mu_ {t}), - \frac {2 \eta}{n ^ {2}} \sum_ {i} \mathbb {E} _ {t} [ G ^ {i} (X _ {t} ^ {i}) ] \rangle = \frac {2 \eta}{n ^ {2}} \sum_ {i} \langle \nabla f (\mu_ {t}), - \mathbb {E} _ {t} [ G ^ {i} (X _ {t} ^ {i}) ] \rangle \\ = \frac {2 \eta}{n ^ {2}} \sum_ {i} \left\langle \nabla f \left(\mu_ {t}\right), - \mathbb {E} _ {t} \left[ G ^ {i} \left(X _ {t} ^ {i}\right) \right] + \nabla f ^ {i} \left(X _ {t} ^ {i}\right) - \nabla f ^ {i} \left(X _ {t} ^ {i}\right) + \nabla f ^ {i} \left(\mu_ {t}\right) - \nabla f ^ {i} \left(\mu_ {t}\right) \right\rangle \tag {51} \\ \end{array}
$$

$$
\frac {1}{n} \sum_ {i} \langle \nabla f (\mu_ {t}), - \mathbb {E} _ {t} [ G ^ {i} (X _ {t} ^ {i}) ] + \nabla f ^ {i} (X _ {t} ^ {i}) \rangle \leq \frac {1}{n} \sum_ {i} \| \nabla f (\mu_ {t}) \|. \| \mathbb {E} _ {t} [ G ^ {i} (X _ {t} ^ {i}) ] - \nabla f ^ {i} (X _ {t} ^ {i}) \|
$$

$$
\stackrel {1 1} {\leq} \frac {1}{n} \| \nabla f (\mu_ {t}) \| \sum_ {i} b _ {i} \stackrel {1 2} {=} B \| \nabla f (\mu_ {t}) \| \stackrel {A M - G M} {\leq} \frac {1}{4} \| \nabla f (\mu_ {t}) \| ^ {2} + B ^ {2} \tag {52}
$$

$$
\frac {1}{n} \sum_ {i} \langle \nabla f (\mu_ {t}), - \nabla f ^ {i} (X _ {t} ^ {i}) + \nabla f ^ {i} (\mu_ {t}) \rangle \leq \frac {1}{n} \sum_ {i} \| \nabla f (\mu_ {t}) \|. \| \nabla f ^ {i} (X _ {t} ^ {i}) - \nabla f ^ {i} (\mu_ {t}) \|
$$

$$
\stackrel {C a u c h y - S c h w a r z} {\leq} \| \nabla f (\mu_ {t}) \| \left(\frac {1}{n} \sum_ {i} \| \nabla f ^ {i} (X _ {t} ^ {i}) - \nabla f ^ {i} (\mu_ {t}) \| ^ {2}\right) ^ {1 / 2} \stackrel {2} {\leq} \| \nabla f (\mu_ {t}) \| \left(L ^ {2} \Gamma_ {t}\right) ^ {1 / 2}
$$

$$
\stackrel {A M - G M} {\leq} \frac {1}{4} \| \nabla f (\mu_ {t}) \| ^ {2} + L ^ {2} \Gamma_ {t} \tag {53}
$$

$$
\frac {1}{n} \sum_ {i} \left\langle \nabla f \left(\mu_ {t}\right), - \nabla f ^ {i} \left(\mu_ {t}\right) \right\rangle = \left\langle \nabla f \left(\mu_ {t}\right), - \frac {1}{n} \sum_ {i} \nabla f ^ {i} \left(\mu_ {t}\right) \right\rangle = - \| \nabla f \left(\mu_ {t}\right) \| ^ {2} \tag {54}
$$

Combining 52, 53, and 54, we will get

$$
\frac {1}{n} \sum_ {i} \left\langle \nabla f \left(\mu_ {t}\right), - \mathbb {E} _ {t} \left[ G ^ {i} \left(X _ {t} ^ {i}\right) \right] + \nabla f ^ {i} \left(X _ {t} ^ {i}\right) - \nabla f ^ {i} \left(X _ {t} ^ {i}\right) + \nabla f ^ {i} \left(\mu_ {t}\right) - \nabla f ^ {i} \left(\mu_ {t}\right) \right\rangle \leq - \frac {1}{2} \| \nabla f (\mu_ {t}) \| ^ {2} + B ^ {2} + L ^ {2} \Gamma_ {t}. \tag {55}
$$

By plugging in the result above in 51 and taking the expectation with respect to $X_{t}^{1}, X_{t}^{2}, \ldots, X_{t}^{n}$ from both sides, the proof would be finished.

Lemma 20. For every time step t, we have

$$
\mathbb {E} [ f (\mu_ {t + 1}) - f (\mu_ {t}) ] \leq \left(\frac {2 L (6 (d + 4) n _ {0} + 3 n _ {1}) \eta^ {2}}{n ^ {3}} - \frac {\eta}{n}\right) \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} + \left(\frac {2 L ^ {2} \eta}{n} + \frac {1 2 L (d + 4) \eta^ {2}}{n ^ {2}}\right) \mathbb {E} [ \Gamma_ {t} ]
$$

$$
+ \eta^ {2} \frac {2 L}{n ^ {2}} \bigg (\frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} \bigg)
$$

$$
+ \eta^ {3} \frac {n _ {0} ^ {2} L ^ {2}}{2 c ^ {2} n ^ {3}} (d + 3) ^ {3} + \eta^ {4} \frac {n _ {0} L ^ {3}}{c ^ {2} n ^ {3}} (d + 6) ^ {3}.
$$

Proof. Using L-smoothness 2 we can write

$$
\mathbb {E} _ {t} [ f (\mu_ {t + 1}) ] \leq f (\mu_ {t}) + \mathbb {E} _ {t} \langle \nabla f (\mu_ {t}), \mu_ {t + 1} - \mu_ {t} \rangle + \frac {L}{2} \mathbb {E} _ {t} \| \mu_ {t + 1} - \mu_ {t} \| ^ {2}.
$$

By taking the expectation with respect to $X_{t}^{1}, X_{t}^{2}, \ldots, X_{t}^{n}$ we will get

$$
\mathbb {E} [ f (\mu_ {t + 1}) ] - \mathbb {E} [ f (\mu_ {t}) ] \leq \mathbb {E} \langle \nabla f (\mu_ {t}), \mu_ {t + 1} - \mu_ {t} \rangle + \frac {L}{2} \mathbb {E} \| \mu_ {t + 1} - \mu_ {t} \| ^ {2}. \tag {56}
$$

Then, we use Lemmas 10, 14, and 19:

$$
\mathbb {E} [ f (\mu_ {t + 1}) ] - \mathbb {E} [ f (\mu_ {t}) ] \leq - \frac {\eta}{n} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} + \frac {2 \eta}{n} B ^ {2} + \frac {2 L ^ {2} \eta}{n} \mathbb {E} [ \Gamma_ {t} ] + \frac {2 L \eta^ {2}}{n ^ {2}} \mathbb {E} [ M _ {t} ^ {G} ]
$$

$$
\stackrel {1 6} {\leq} - \frac {\eta}{n} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} + \frac {2 \eta}{n} B ^ {2} + \frac {2 L ^ {2} \eta}{n} \mathbb {E} [ \Gamma_ {t} ]
$$

$$
+ \frac {2 L \eta^ {2}}{n ^ {2}} \bigg (6 (d + 4) L ^ {2} \mathbb {E} [ \Gamma_ {t} ] + \frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + \frac {6 (d + 4) n _ {0} + 3 n _ {1}}{n} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}
$$

$$
\left. + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} + \eta^ {2} \frac {n _ {0}}{2 n c ^ {2}} L ^ {2} (d + 6) ^ {3}\right)
$$

$$
\stackrel {1 0} {\leq} \left(\frac {2 L (6 (d + 4) n _ {0} + 3 n _ {1}) \eta^ {2}}{n ^ {3}} - \frac {\eta}{n}\right) \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} + \left(\frac {2 L ^ {2} \eta}{n} + \frac {1 2 L (d + 4) \eta^ {2}}{n ^ {2}}\right) \mathbb {E} [ \Gamma_ {t} ]
$$

$$
+ \eta^ {2} \frac {2 L}{n ^ {2}} \bigg (\frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} \bigg)
$$

$$
+ \eta^ {3} \frac {n _ {0} ^ {2} L ^ {2}}{2 c ^ {2} n ^ {3}} (d + 3) ^ {3} + \eta^ {4} \frac {n _ {0} L ^ {3}}{c ^ {2} n ^ {3}} (d + 6) ^ {3}.
$$

□

Theorem 1.2 (Non-Convex case of Theorem 1). Under Assumptions 2, 3 and 4, and letting $T$ to be large enough such that $T = \Omega \left( \max \left\{ \frac{L^2(dn_0 + n_1)^2}{dn^2}, n^2 L^2, nn_0^3 d \right\} \right)$ we have

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \left\| \nabla f (\mu_ {t}) \right\| ^ {2} \leq O \Bigg ((f (\mu_ {0}) - f ^ {*}) + L \big (\frac {d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}}{n d} \big) + L \big (\frac {d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2}}{n d} \big) + L ^ {2} \sqrt {\frac {n _ {0}}{n}} \Bigg) \sqrt {\frac {d}{T}}.
$$

Proof.

$$
\begin{array}{l} \mathbb {E} [ f (\mu_ {t + 1}) ] - \mathbb {E} [ f (\mu_ {t}) ] \stackrel {{2 0}} {{\leq}} \left(\frac {2 L (6 (d + 4) n _ {0} + 3 n _ {1}) \eta^ {2}}{n ^ {3}} - \frac {\eta}{n}\right) \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} + \underbrace {\left(\frac {2 L ^ {2} \eta}{n} + \frac {1 2 L (d + 4) \eta^ {2}}{n ^ {2}}\right)} _ {J _ {0} :=} \mathbb {E} [ \Gamma_ {t} ] \\ + \eta^ {2} \underbrace {\frac {2 L}{n ^ {2}} \bigg (\frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} \bigg)} _ {J _ {1} :=} \\ + \eta^ {3} \underbrace {\frac {n _ {0} ^ {2} L ^ {2}}{2 c ^ {2} n ^ {3}} (d + 3) ^ {3}} _ {J _ {2} :=} + \eta^ {4} \underbrace {\frac {n _ {0} L ^ {3}}{c ^ {2} n ^ {3}} (d + 6) ^ {3}} _ {J _ {3} :=} \\ \end{array}
$$

Considering $\eta \leq \frac{n^2}{4L(6(d + 4)n_0 + 3n_1)}$ , we can write:

$$
\mathbb {E} [ f (\mu_ {t + 1}) ] - \mathbb {E} [ f (\mu_ {t}) ] \leq - \frac {\eta}{2 n} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} + J _ {0} \mathbb {E} [ \Gamma_ {t} ] + \eta^ {2} J _ {1} + \eta^ {3} J _ {2} + \eta^ {4} J _ {3}.
$$

Now, we sum up the inequality above for $t \in \{0, ..., T - 1\}$ to get:

$$
\begin{array}{l} \mathbb {E} [ f (\mu_ {T}) ] - \mathbb {E} [ f (\mu_ {0}) ] \leq - \frac {\eta}{2 n} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} + J _ {0} \sum_ {t = 0} ^ {T - 1} \mathbb {E} [ \Gamma_ {t} ] \\ + \eta^ {2} T J _ {1} + \eta^ {3} T J _ {2} + \eta^ {4} T J _ {3}. \tag {57} \\ \end{array}
$$

By using Lemma 18 and the fact that $f(\mu_T) \geq f^*$ we will have:

$$
\begin{array}{l} \frac {\eta}{2 n} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \leq \left(f \left(\mu_ {0}\right) - f ^ {*}\right) + \eta^ {2} T J _ {1} + \eta^ {3} T J _ {2} + \eta^ {4} T J _ {3} \\ + J _ {0} \left(T \left(\frac {4 8 \eta^ {2} (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 \eta^ {2} (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n}\right)\right) \\ \left. + \frac {8 \eta^ {4} n _ {0} L ^ {2} (d + 6) ^ {3}}{n c ^ {2}}\right) + \frac {1 6 \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n} \sum_ {t = 0} ^ {T - 2} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}). \tag {58} \\ \end{array}
$$

Then, by defining $K_{1}, K_{2}, K_{3}, K_{4}, K_{5}, K_{6}$ , and $K_{7}$ as

$$
\begin{array}{l} K _ {1} := J _ {1}, \\ K _ {2} := J _ {2} + \frac {2 L ^ {2}}{n} \bigg (\frac {4 8 (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} \bigg), \\ K _ {3} := J _ {3} + \frac {1 2 L (d + 4)}{n ^ {2}} \bigg (\frac {4 8 (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} \bigg), \\ K _ {4} := \frac {1 6 n _ {0} L ^ {4} (d + 6) ^ {3}}{n ^ {2} c ^ {2}}, \\ K _ {5} := \frac {9 6 n _ {0} L ^ {3} (d + 4) (d + 6) ^ {3}}{n ^ {3} c ^ {2}}, \\ K _ {6} := \frac {6 4 L ^ {2} \eta^ {2} (6 (d + 4) n _ {0} + 3 n _ {1})}{n}, \\ K _ {7} := \frac {3 8 4 L \eta^ {3} (d + 4) (6 (d + 4) n _ {0} + 3 n _ {1})}{n ^ {2}}, \\ \end{array}
$$

we can rewrite 58 as

$$
\frac {\eta}{2 n} \big (1 - K _ {6} - K _ {7} \big) \sum_ {t = 0} ^ {T - 1} \mathbb {E} \left\| \nabla f (\mu_ {t}) \right\| ^ {2} \leq \big (f (\mu_ {0}) - f ^ {*} \big) + \eta^ {2} T K _ {1} + \eta^ {3} T K _ {2} + \eta^ {4} T K _ {3} + \eta^ {5} T K _ {4} + \eta^ {6} T K _ {5}.
$$

By setting $\eta \leq \sqrt{\frac{n}{256L^2(6(d + 4)n_0 + 3n_1)}}$ and $\eta \leq \left(\frac{n^2}{1536L(d + 4)(6(d + 4)n_0 + 3n_1)}\right)^{\frac{1}{3}}$ , we would get $1 - K_{6} - K_{7} \geq \frac{1}{2}$ and therefore we have:

$$
\frac {1}{4 n T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \left\| \nabla f (\mu_ {t}) \right\| ^ {2} \leq \frac {1}{\eta T} \big (f (\mu_ {0}) - f ^ {*} \big) + \eta K _ {1} + \eta^ {2} K _ {2} + \eta^ {3} K _ {3} + \eta^ {4} K _ {4} + \eta^ {5} K _ {5}.
$$

In this step, we set $\eta$ equal to $\frac{en}{\sqrt{T}}$ for a parameter e that we will define later.

$$
\frac {1}{4 n T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \leq \frac {1}{e n \sqrt {T}} \left(f \left(\mu_ {0}\right) - f ^ {*}\right) + \frac {e n}{\sqrt {T}} K _ {1} + \left(\frac {e n}{\sqrt {T}}\right) ^ {2} K _ {2} + \left(\frac {e n}{\sqrt {T}}\right) ^ {3} K _ {3} + \left(\frac {e n}{\sqrt {T}}\right) ^ {4} K _ {4} + \left(\frac {e n}{\sqrt {T}}\right) ^ {5} K _ {5}. \tag {59}
$$

By putting $c = \sqrt{d}$ , we can give upper bound for $K_{1}, K_{2}, K_{3}, K_{4}$ , and $K_{5}$ as follows:

$$
\begin{array}{l} K _ {1} = \frac {2 L}{n ^ {2}} \left(\frac {6 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + 3 n _ {1} \varsigma_ {1} ^ {2}}{n} + \frac {2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n}\right) \\ = O \left(\frac {L}{n ^ {2}} \left(\left(\frac {d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}}{n}\right) + \left(\frac {d \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n}\right)\right)\right), \\ \end{array}
$$

$$
K _ {2} = \frac {n _ {0} ^ {2} L ^ {2}}{2 c ^ {2} n ^ {3}} (d + 3) ^ {3} + \frac {2 L ^ {2}}{n} \bigg (\frac {4 8 (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} \bigg)
$$

$$
= O \left(\frac {n _ {0} ^ {2} L ^ {2} d ^ {2}}{n ^ {3}} + \frac {L ^ {2}}{n} \left(\left(\frac {d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}}{n}\right) + \left(\frac {d \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n}\right)\right)\right),
$$

$$
\begin{array}{l} K _ {3} = \frac {n _ {0} L ^ {3}}{c ^ {2} n ^ {3}} (d + 6) ^ {3} + \frac {1 2 L (d + 4)}{n ^ {2}} \bigg (\frac {4 8 (2 (d + 4) n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2})}{n} + \frac {1 6 (2 (d + 4) \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2})}{n} \bigg) \\ = O \left(\frac {n _ {0} L ^ {3} d ^ {2}}{n ^ {3}} + \frac {L d}{n ^ {2}} \left(\left(\frac {d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}}{n}\right) + \left(\frac {d \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n}\right)\right)\right), \\ \end{array}
$$

$$
K _ {4} = \frac {1 6 n _ {0} L ^ {4} (d + 6) ^ {3}}{n ^ {2} c ^ {2}} = O \left(\frac {n _ {0} L ^ {4} d ^ {2}}{n ^ {2}}\right),
$$

$$
K _ {5} = \frac {9 6 n _ {0} L ^ {3} (d + 4) (d + 6) ^ {3}}{n ^ {3} c ^ {2}} = O \left(\frac {n _ {0} L ^ {3} d ^ {3}}{n ^ {3}}\right).
$$

Now we multiply both sides of 59 by 4n and use the inequalities above to get:

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \leq O \left(\frac {1}{e \sqrt {T}} (f (\mu_ {0}) - f ^ {*}) + \frac {e ^ {2} n _ {0} ^ {2} L ^ {2} d ^ {2}}{T} + \frac {e ^ {3} n n _ {0} L ^ {3} d ^ {2}}{T ^ {\frac {3}{2}}} + \frac {e ^ {4} n ^ {3} n _ {0} L ^ {4} d ^ {2}}{T ^ {2}} + \frac {e ^ {5} n ^ {3} n _ {0} L ^ {3} d ^ {3}}{T ^ {\frac {5}{2}}} + \dots\right) \\ \left. + \big (\frac {e L}{\sqrt {T}} + \frac {e ^ {2} n ^ {2} L ^ {2}}{T} + \frac {e ^ {3} n ^ {2} L d}{T ^ {\frac {3}{2}}} \big) \bigg (\big (\frac {d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}}{n} \big) + \big (\frac {d \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n} \big) \bigg)\right). \\ \end{array}
$$

Next, note that time T here counts total interactions. However, $\Theta(n)$ interactions occur simultaneously. Therefore, if we define $T_{p}$ as the parallel execution time of the process, we deduce $T = \Theta(nT_{p})$ , hence we can write:

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \leq O \left(\frac {1}{e \sqrt {n T _ {p}}} (f (\mu_ {0}) - f ^ {*}) + \frac {e ^ {2} n _ {0} ^ {2} L ^ {2} d ^ {2}}{n T _ {p}} + \frac {e ^ {3} n _ {0} L ^ {3} d ^ {2}}{\sqrt {n} T _ {p} ^ {\frac {3}{2}}} + \frac {e ^ {4} n n _ {0} L ^ {4} d ^ {2}}{T _ {p} ^ {2}} + \frac {e ^ {5} \sqrt {n} n _ {0} L ^ {3} d ^ {3}}{T _ {p} ^ {\frac {5}{2}}} + \dots\right) \\ + \left(\frac {e L}{\sqrt {n T _ {p}}} + \frac {e ^ {2} n L ^ {2}}{T _ {p}} + \frac {e ^ {3} \sqrt {n} L d}{T _ {p} ^ {\frac {3}{2}}}\right) \left((\frac {d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}}{n}) + (\frac {d \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n})\right). \\ \end{array}
$$

Considering $e \leq \frac{1}{\sqrt{d}}$ and $T_{p} \geq \max \left(\frac{n^{3}L^{2}}{d}, n\right)$ , we will have

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \leq O \left(\frac {1}{e \sqrt {n T _ {p}}} (f (\mu_ {0}) - f ^ {*}) + \frac {e ^ {2} n _ {0} ^ {2} L ^ {2} d ^ {2}}{n T _ {p}} \right. \\ + \left(\frac {e L}{\sqrt {n T _ {p}}}\right) \bigg ((\frac {d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}}{n}) + (\frac {d \sum_ {i \in N _ {0}} s _ {i} ^ {2} + \sum_ {i \in N _ {1}} s _ {i} ^ {2}}{n}) \bigg) \bigg). \\ \end{array}
$$

Finally, by putting $e = \frac{1}{\sqrt{d}}$ and $T_{p} \geq n_{0}^{3}d$ , we will conclude the final convergence

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \leq O \left((f (\mu_ {0}) - f ^ {*}) + L \left(\left(\frac {d n _ {0} \varsigma_ {0} ^ {2} + n _ {1} \varsigma_ {1} ^ {2}}{n d}\right) + \left(\frac {d n _ {0} \sigma_ {0} ^ {2} + n _ {1} \sigma_ {1} ^ {2}}{n d}\right)\right) + L ^ {2} \sqrt {\frac {n _ {0}}{n}}\right) \sqrt {\frac {d}{T}}.
$$

Gathering all the assumed upper bounds for $\eta$ , we have

$$
\eta = O \left(\min \left\{\sqrt {\frac {n}{L ^ {2} (d n _ {0} + n _ {1})}}, \left(\frac {n ^ {2}}{L (d n _ {0} + n _ {1})}\right) ^ {\frac {1}{3}}, \frac {n ^ {2}}{L (d n _ {0} + n _ {1})}, \frac {1}{\sqrt {L d}} \right\}\right),
$$

which implies $\eta = O\left(\min \left\{\frac{1}{L\sqrt{d}},\frac{n^2}{L(dn_0 + n_1)}\right\}\right)$ , resulting in $T = \Omega$ $\left(\max \left\{\frac{L^2(dn_0 + n_1)^2}{dn^2},n^2 L^2,nn_0^3 d\right\}\right)$ .

□