# Best Arm Identification with Fixed Budget: A Large Deviation Perspective

Po-An Wang

EECS

KTH, Stockholm, Sweden

wang9@kth.se

Ruo-Chun Tzeng

EECS

KTH, Stockholm, Sweden

rctzeng@kth.se

Alexandre Proutiere

EECS and Digital Futures

KTH, Stockholm, Sweden

alepro@kth.se

# Abstract

We consider the problem of identifying the best arm in stochastic Multi-Armed Bandits (MABs) using a fixed sampling budget. Characterizing the minimal instance-specific error probability for this problem constitutes one of the important remaining open problems in MABs. When arms are selected using a static sampling strategy, the error probability decays exponentially with the number of samples at a rate that can be explicitly derived via Large Deviation techniques. Analyzing the performance of algorithms with adaptive sampling strategies is however much more challenging. In this paper, we establish a connection between the Large Deviation Principle (LDP) satisfied by the empirical proportions of arm draws and that satisfied by the empirical arm rewards. This connection holds for any adaptive algorithm, and is leveraged (i) to improve error probability upper bounds of some existing algorithms, such as the celebrated SR (Successive Rejects) algorithm (Audibert et al., 2010), and (ii) to devise and analyze new algorithms. In particular, we present CR (Continuous Rejects), a truly adaptive algorithm that can reject arms in any round based on the observed empirical gaps between the rewards of various arms. Applying our Large Deviation results, we prove that CR enjoys better performance guarantees than existing algorithms, including SR. Extensive numerical experiments confirm this observation.

# 1 Introduction

We study the problem of best-arm identification in stochastic bandits in the fixed budget setting. In this problem, abbreviated by BAI-FB, a learner faces $K$ distributions or arms $\nu_{1},\ldots ,\nu_{K}$ characterized by their unknown means $\pmb{\mu} = (\mu_1,\dots ,\mu_K)$ (we restrict our attention to distributions taken from a one-parameter exponential family). She sequentially pulls arms and observes samples of the corresponding distributions. More precisely, in round $t\geq 1$ , she pulls an arm $A_{t} = k$ selected depending on previous observations and observes $X_{k}(t)$ a sample of a $\nu_{k}$ -distributed random variable. $(X_{k}(t),t\geq 1,k\in [K])$ are assumed to be independent over rounds and arms. After $T$ arm draws, the learner returns $\hat{\imath}$ , an estimate of the best arm $1(\pmb {\mu}):= \arg \max_k\mu_k$ . We assume that the best arm is unique, and denote by $\Lambda$ the set of parameters $\pmb{\mu}$ such that this assumption holds. The objective is to devise an adaptive sampling algorithm minimizing the error probability $\mathbb{P}_{\pmb{\mu}}[\hat{\imath}\neq 1(\pmb {\mu})]$ . This learning task is one of the most important problems in stochastic bandits, and despite recent research efforts, it remains largely open (Qin, 2022). In particular, researchers have so far failed

at characterizing the minimal instance-specific error probability. This contrasts with other basic learning tasks in stochastic bandits such as regret minimization (Lai and Robbins, 1985) and BAI with fixed confidence (Garivier and Kaufmann, 2016), for which indeed, asymptotic instance-specific performance limits and matching algorithms have been derived. In BAI-FB, the error probability typically decreases exponentially with the sample budget T, i.e., it scales as $\exp(-R(\boldsymbol{\mu})T)$ where the instance-specific rate $R(\boldsymbol{\mu})$ depends on the sampling algorithm. Maximizing this rate over the set of adaptive algorithms is an open problem.

Instance-specific error probability lower bound. To guess the maximal rate at which the error probability decays, one may apply the same strategy as that used in regret minimization or BAI in the fixed confidence setting: (i) derive instance-specific lower bound for the error probability for some notion of uniformly good algorithms; (ii) devise a sampling strategy mimicking the optimal proportions of arm draws identified in the lower bound. Here the notion of uniformly good algorithms is that of consistent algorithms. Under such an algorithm, for any $\mu\in\Lambda$ , $\mathbb{P}_{\mu}\left[\hat{i}=1(\boldsymbol{\mu})\right]\to1$ as $T\to\infty$ . (Garivier and Kaufmann, 2016) conjectures the following asymptotic lower bound satisfied by any consistent algorithm (refer to Appendix J for details): as $T\to\infty$ ,

$$
\frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {i} \neq 1 (\boldsymbol {\mu}) ]} \leq \max _ {\boldsymbol {\omega} \in \Sigma} \inf _ {\boldsymbol {\lambda} \in \operatorname{Alt} (\boldsymbol {\mu})} \Psi (\boldsymbol {\lambda}, \boldsymbol {\omega}), \tag {1}
$$

where $\Sigma$ is the $(K - 1)$ -dimensional simplex, $\Psi(\boldsymbol{\lambda},\boldsymbol{\omega}) = \sum_{k=1}^{K}\omega_k d(\lambda_k,\mu_k)$ , $\mathrm{Alt}(\boldsymbol{\mu}) = \{\boldsymbol{\lambda}\in\Lambda:1(\boldsymbol{\mu})\neq 1(\boldsymbol{\lambda})\}$ is the set of confusing parameters (those for which $1(\boldsymbol{\mu})$ is not the best arm), and $d(x,y)$ denotes the KL divergence between two distributions of parameters $x$ and $y$ . Interestingly, the solution $\boldsymbol{\omega}^{\star}\in\Sigma$ of the optimization problem $\max_{\boldsymbol{\omega}\in\Sigma}\inf_{\boldsymbol{\lambda}\in\mathrm{Alt}(\boldsymbol{\mu})}\Psi(\boldsymbol{\lambda},\boldsymbol{\omega})$ provides the best static proportions of arm draws. More precisely, an algorithm selecting arms according to the allocation $\boldsymbol{\omega}^{\star}$ , i.e., selecting arm $k\omega_{k}^{\star}T$ times and returning the best empirical arm after $T$ samples, has an error rate matching the lower bound (1). This is a direct consequence of the fact that, under a static algorithm with allocation $\boldsymbol{\omega}$ , the empirical reward process $\{\hat{\boldsymbol{\mu}}(t)\}_{t\geq 1}$ satisfies a LDP with rate function $\boldsymbol{\lambda}\mapsto\Psi(\boldsymbol{\lambda},\boldsymbol{\omega})$ , see (Glynn and Juneja, 2004) and refer to Section 3 for more details.

Adaptive sampling algorithms and their analysis. The optimal allocation $\omega^{*}$ depends on the instance $\mu$ and is initially unknown. We may devise an adaptive sampling algorithm that (i) estimates $\omega^{*}$ and (ii) tracks this estimated optimal allocation. In the BAI with fixed confidence, such tracking scheme exhibits asymptotically optimal performance (Garivier and Kaufmann, 2016). Here however, the error made estimating $\omega^{*}$ would inevitably impact the overall error probability of the algorithm. To quantify this impact or more generally to analyze the performance of adaptive algorithms, one would need to understand the connection between the statistical properties of the arm selection process and the asymptotic statistics of the estimated expected rewards.

To be more specific, any adaptive algorithm generates a stochastic process $\{Z(t)\}_{t\geq 1} = \{(\pmb {\omega}(t),\hat{\pmb{\mu}}(t))\}_{t\geq 1}$ . $\pmb{\omega}(t) = (\omega_1(t),\dots ,\omega_K(t))$ represents the allocation realized by the algorithm up to round $t$ $(\omega_{k}(t) = N_{k}(t) / t$ and $N_{k}(t)$ denotes the number of times arm $k$ has been selected up to round $t$ ). $\hat{\pmb{\mu}} (t) = (\hat{\mu}_1(t),\dots ,\hat{\mu}_K(t))$ denotes the empirical average rewards of the various arms up to round $t$ . Now assuming that at the end of round $T$ , the algorithm returns the arm with the highest empirical reward, the error probability is $\mathbb{P}_{\pmb{\mu}}[\hat{i}\neq 1(\pmb {\mu})] = \mathbb{P}_{\pmb{\mu}}[\hat{\pmb{\mu}} (T)\in \mathrm{Alt}(\pmb {\mu})]$ . Assessing the error probability at least asymptotically requires understanding the asymptotic behavior of $\hat{\pmb{\mu}} (t)$ as $t$ grows large. Ideally, one would wish to establish the Large Deviation properties of the process $\{Z(t)\}_{t\geq 1}$ . This task is easy for algorithms using static allocations (Glynn and Juneja, 2004), but becomes challenging and open for adaptive algorithms. Addressing this challenge is the main objective of this paper.

Contributions. In this paper, we develop and leverage tools towards the analysis of adaptive sampling algorithms for the BAI-FB problem. More precisely, our contributions are as follows.

(a) We establish a connection between the LDP satisfied by the empirical proportions of arm draws $\{\omega(t)\}_{t\geq 1}$ and that satisfied by the empirical arm rewards. This connection holds for any adaptive algorithm. Specifically, we show that if the rate function of $\{\omega(t)\}_{t\geq 1}$ is lower bounded by $\omega \mapsto I(\omega)$ , then that of $(\hat{\mu}(t))_{t\geq 1}$ is also lower bounded by $\lambda \mapsto \min_{\omega \in \Sigma} \max \{\Psi(\lambda, \omega), I(\omega)\}$ . This result has interesting interpretations and implies the following asymptotic upper bound on the error probability of the algorithm considered: as $T \to \infty$ ,

$$
\frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {i} \neq 1 (\boldsymbol {\mu}) ]} \geq \inf _ {\boldsymbol {\omega} \in \Sigma , \boldsymbol {\lambda} \in \operatorname{Alt} (\boldsymbol {\mu})} \max \left\{\Psi (\boldsymbol {\lambda}, \boldsymbol {\omega}), I (\boldsymbol {\omega}) \right\}. \tag {2}
$$

The above formula, when compared to the lower bound (1), quantifies the price of not knowing $\omega^{*}$ initially, and relates the error probability to the asymptotic statistics of the sampling process used by the algorithm.

(b) We show that by simply applying our generic Large Deviation result, we may improve the error probability upper bounds of some existing algorithms, such as the celebrated SR algorithm (Audibert et al., 2010). Our result further opens up opportunities to devise and analyze new algorithms with a higher level of adaptiveness. In particular, we present CR (Continuous Rejects), an algorithm that, unlike SR, can eliminate arms in each round. This sequential elimination process is performed by comparing the empirical rewards of the various candidate arms using continuously updated thresholds. Leveraging the LDP tools developed in (a), we establish that CR enjoys better performance guarantees than SR. Hence CR becomes the algorithm with the lowest instance-specific and guaranteed error probability. We illustrate our results via numerical experiments, and compare CR to other BAI algorithms.

# 2 Related Work

We distinguish two main classes of algorithms to solve the best arm identification problem in the fixed budget setting. Algorithms from the first class, e.g. Successive Rejects (SR) (Audibert et al., 2010) and Sequential Halving (SH) (Karnin et al., 2013), split the sampling budget into phases of fixed durations, and discard arms at the end of each phase. Algorithms from the second class, e.g. UCB-E (Audibert et al., 2010) and UGapE (Gabillon et al., 2012) sequentially sample arms based on confidence bounds of their empirical rewards. It is worth mentioning that algorithms from the second class usually require some prior knowledge about the problem, for example, an upper bound of $H = \sum_{k\neq 1(\mu)}\frac{1}{(\mu_1(\mu) - \mu_k)^2}$ . Without this knowledge, the parameters can be chosen in a heuristic way, but the performance gets worse.

Algorithms from the first class exhibit better performance numerically and are also those with the best instance-specific error probability guarantees. SR had actually the best performance guarantees so far: for example, when the reward distributions are supported on $[0,1]$ , the error probability of SR satisfies: $\liminf_{T\to\infty}\frac{1}{T}\log\frac{1}{\mathbb{P}_{\boldsymbol{\mu}}[\hat{i}\neq1(\boldsymbol{\mu})]}\geq\frac{1}{H_{2}\log K}$ , where $H_{2}=\max_{k\neq1(\boldsymbol{\mu})}\frac{k}{(\mu_{1(\boldsymbol{\mu})}-\mu_{k})^{2}}$ . In this paper, we strictly improve this guarantee (see Section 3.4). Recently, (Barrier et al., 2022) also refined and extended the analysis of (Audibert et al., 2010) by replacing, in the analysis, Hoeffding's inequality by a large deviation result involving KL-divergences. Again here, we further improve this new guarantee. We also devise CR, an algorithm with error probability provably lower than our improved guarantees for SR.

Fundamental limits on the error probability have also been investigated. In the minimax setting, (Carpentier and Locatelli, 2016) established that for any algorithm, there exists a problem within the class of instances with given complexity H such that the error probability is greater than $\exp(-\frac{400T}{H\log K}) \geq \exp(-\frac{400T}{H_{2}\log K})$ . Up to a universal constant (here 400), SR is hence minimax optimal. This lower bound was also recently revisited in (Ariu et al., 2021; Degenne, 2023; Wang et al., 2023) to prove that the instance-specific lower bound (1) cannot be achieved on all instances by a single algorithm. Deriving tight instance-specific lower bounds remains open (Qin, 2022).

We conclude this section by mentioning two interesting algorithms. In (Komiyama et al., 2022), the authors propose DOT, an algorithm trying to match minimax error probability lower bounds. To this aim, the algorithm requires to periodically call an oracle able to determine an optimal allocation, solution of an optimization problem with high and unknown complexity. DOT has minimax guarantees but is computationally challenging if not infeasible (numerically, the authors cannot go beyond simple instances with 3 arms). Finally, researchers have also looked at the best arm identification problem from a Bayesian perspective. For example, (Russo, 2016) devise variants of the celebrated Thompson Sampling algorithm, that could potentially work well in practice. Nevertheless, as discussed in (Komiyama, 2022), Bayesian algorithms cannot be analyzed nor provably perform well in the frequentist setting.

# 3 Large Deviation Analysis of Adaptive Sampling Algorithms

In this section, we first recall key concepts in Large Deviations (refer to the classical textbooks (Budhiraja and Dupuis, 2019; Dembo and Zeitouni, 2009; Dupuis and Ellis, 2011; Varadhan, 2016) for a more detailed exposition). We then apply these concepts to the performance analysis of adaptive sampling algorithms. Finally, we exemplify the analysis and apply it to improve existing performance guarantees for the SR algorithm (Audibert et al., 2010).

# 3.1 Large Deviation Principles

Consider the stochastic process $\{Y(t)\}_{t\geq 1}$ with values in a separable complete metric space (i.e., a Polish space) $\mathcal{Y}$ . Large Deviations are concerned with the probabilities of rare events related to $\{Y(t)\}_{t\geq 1}$ that decay exponentially in the parameter $t$ . The asymptotic decay rate is characterized by the rate function $I:\mathcal{Y}\to \mathbb{R}_+$ defined so that essentially $-\frac{1}{t}\log \mathbb{P}[Y(t)\in B]$ converges to $\min_{x\in B}I(x)$ for any Borel set $B$ . We provide a more rigorous definition below.

Definition 1. [Large Deviation Principle (LDP)] The stochastic process $\{Y(t)\}_{t\geq 1}$ satisfies a LDP with rate function $I$ if:

(i) I is lower semicontinuous, and $\forall s \in [0, \infty]$ , the set $\mathcal{K}_{s} = \{y \in \mathcal{Y} : I(y) \leq s\}$ is compact;
(ii) for every closed (resp. open) set $C \subset Y$ (resp. $O \subset Y$ ),

$$
\varliminf_ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} \left[ Y (t) \in C \right]} \geq \inf _ {y \in C} I (y), \tag {3}
$$

$$
\varlimsup_ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} \left[ Y (t) \in O \right]} \leq \inf _ {y \in O} I (y). \tag {4}
$$

LDPs have been derived earlier in stochastic bandit literature. (Glynn and Juneja, 2004) have used Gärtner-Ellis Theorem (Ellis, 1984; Gärtner, 1977) to establish that under a static sampling algorithm with allocation $\omega \in \Sigma$ (i.e., each arm $k$ is selected $\omega_{k}T$ times up to round $T$ ), the process $\{\hat{\mu}(T)\}_{T\geq 1}$ satisfies an LDP with rate function $\lambda \mapsto \Psi (\pmb {\lambda},\pmb {\omega}) = \sum_{k = 1}^{K}\omega_{k}d(\lambda_{k},\mu_{k})$ . Our objective in the next subsection is to investigate how to extend this result to the case of adaptive sampling algorithms.

# 3.2 Analysis of adaptive sampling algorithms

An adaptive sampling algorithm generates a stochastic process $\{Z(t)\}_{t\geq 1} = \{(\omega (t),\hat{\mu} (t))\}_{t\geq 1}$ . When the sampling budget $T$ is exhausted, should the algorithm returns the arm with the highest empirical reward, the error probability is $\mathbb{P}_{\boldsymbol{\mu}}[\hat{i}\neq 1(\boldsymbol {\mu})] = \mathbb{P}_{\boldsymbol{\mu}}[\hat{\boldsymbol{\mu}} (T)\in \mathrm{Alt}(\boldsymbol {\mu})]$ . To assess the rate at which this probability decays with the budget, we may try to establish a LDP for the empirical reward process $\{\hat{\boldsymbol{\mu}} (t)\}_{t\geq 1}$ . Due to the intricate dependence between the sampling and the empirical reward processes, deriving such an LDP is very challenging. Instead, we establish a connection between the LDPs satisfied by these processes. This connection will be enough for us to derive tight upper bounds on the error probability. We present our main result in the following theorem.

Theorem 1. Assume that under some adaptive sampling algorithm, $\{\pmb{\omega}(t)\}_{t\geq 1}$ satisfies the LDP upper bound (3) with rate function $I$ . Then $\{\hat{\pmb{\mu}} (t)\}_{t\geq 1}$ satisfies the LDP upper bound (3) with rate function $\pmb {\lambda}\mapsto \min_{\pmb {\omega}\in \Sigma}\max \{\Psi (\pmb {\lambda},\pmb {\omega}),I(\pmb {\omega})\}$ . Moreover, we have: for any bounded Borel subset $S$ of $\mathbb{R}^K$ and any Borel subset $W$ of $\Sigma$ ,

$$
\varliminf_ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} , \boldsymbol {\omega} (t) \in W \right]} \geq \inf _ {\boldsymbol {\omega} \in \operatorname{cl} (W)} \max \left\{F _ {\mathcal {S}} (\boldsymbol {\omega}), I (\boldsymbol {\omega}) \right\},
$$

where $F_{\mathcal{S}}(\omega):=\inf_{\boldsymbol{\lambda}\in\mathrm{cl}(\mathcal{S})}\Psi(\boldsymbol{\lambda},\boldsymbol{\omega})$ , and $\operatorname{cl}(\mathcal{S})$ denotes the closure of $\mathcal{S}$ .

Before proving the above theorem, we make the following remarks and provide a simple corollary that will lead to improved upper bound on the error probability of the SR algorithm.

(a) Not a complete LDP. To upper bound the error probability of a given algorithm, we do not actually need to establish that $\{\hat{\mu}(t)\}_{t\geq 1}$ satisfies a complete LDP. Instead, deriving a LDP upper bound is enough. Theorem 1 provides such an upper bound, but does not yield a complete LDP. We conjecture if $\{\omega (t)\}_{t\geq 1}$ satisfies an LDP with rate function $I$ , $\{\hat{\mu} (t)\}_{t\geq 1}$ satisfies an LDP with rate function $\lambda \mapsto \inf_{\omega \in W}\max \{\Psi (\lambda ,\omega),I(\omega)\}$ . The conjecture holds for static sampling algorithms as shown

below. If it holds for adaptive algorithms, we show, in Appendix I, that when $K > 3$ , no algorithm can attain the instance-specific lower bound (1) for all parameters.

(b) Theorem 1 is tight for static sampling algorithms. When the sampling rule is static, namely $\omega(t) = \omega \in \Sigma$ , then $\{\omega(t)\}_{t \geq 1}$ satisfies a LDP with rate function $I$ defined as $I(\omega) = 0$ and $\infty$ elsewhere. Theorem 1 with $W = \Sigma$ states that $\{\hat{\mu}(t)\}_{t \geq 1}$ satisfies the LDP upper bound (3) with rate function $F_{S}$ . In fact, as shown by (Glynn and Juneja, 2004), $\{\hat{\mu}(t)\}_{t \geq 1}$ satisfies a complete LDP with this rate function.

(c) A useful corollary. From Theorem 1, we have:

$$
\varliminf_ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} , \boldsymbol {\omega} (t) \in W \right]} \geq \inf _ {\boldsymbol {\omega} \in \operatorname{cl} (W)} F _ {\mathcal {S}} (\boldsymbol {\omega}).
$$

From there, we will be able to improve the performance guarantee for SR.

# 3.3 Proof of Theorem 1

Proof. Observe that when S or W is empty, the result holds. Now recall that $F_{\mathcal{S}}(\cdot) = \inf_{\boldsymbol{\lambda} \in \mathrm{cl}(\mathcal{S})} \Psi(\boldsymbol{\lambda}, \cdot)$ is the infimum of a family of linear functions on a compact set, $\Sigma$ , hence it is upper bounded. Denote u > 0 such an upper bound. $F_{\mathcal{S}}(\cdot)$ is also continuous (see Appendix F for details). For each integer $N \in N$ , we define a collection of closed sets:

$$
W _ {n} ^ {N} = \left\{\boldsymbol {\omega} \in \operatorname{cl} (W): \frac {u (n - 1)}{N} \leq F _ {\mathcal {S}} (\boldsymbol {\omega}) \leq \frac {u n}{N} \right\}, \quad \forall n \in [ N ]. \tag {5}
$$

We observe that:

$$
\begin{array}{l} \mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S}, \boldsymbol {\omega} (t) \in W ] \leq \sum_ {n = 1} ^ {N} \mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S}, \boldsymbol {\omega} (t) \in W _ {n} ^ {N} ] \\ \leq N \max _ {n \in [ N ]} \mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S}, \boldsymbol {\omega} (t) \in W _ {n} ^ {N} ]. \\ \end{array}
$$

Taking the logarithm on both sides and dividing them by -t yields that

$$
\begin{array}{l} \varliminf_ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} , \boldsymbol {\omega} (t) \in W ]} \geq \varliminf_ {t \to \infty} \min _ {n \in [ N ]} \frac {1}{t} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} , \boldsymbol {\omega} (t) \in W _ {n} ^ {N} ]} \\ = \min _ {n \in [ N ]} \varliminf_ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} , \boldsymbol {\omega} (t) \in W _ {n} ^ {N} \right]} \\ \geq \min _ {n \in [ N ]} \max \left\{\frac {u (n - 1)}{N}, \inf _ {\boldsymbol {\omega} \in W _ {n} ^ {N}} I (\boldsymbol {\omega}) \right\}, \tag {6} \\ \end{array}
$$

where the last inequality follows from Lemma 1. Since for all $n \in [N]$ ,

$$
\max \left\{\frac {u (n - 1)}{N}, \inf _ {\boldsymbol {\omega} \in W _ {n} ^ {N}} I (\boldsymbol {\omega}) \right\} = \inf _ {\boldsymbol {\omega} \in W _ {n} ^ {N}} \max \left\{\frac {u (n - 1)}{N}, I (\boldsymbol {\omega}) \right\},
$$

the r.h.s. of (6) is equal to

$$
\begin{array}{l} \min _ {n \in [ N ]} \inf _ {\boldsymbol {\omega} \in W _ {n} ^ {N}} \max \left\{\frac {u (n - 1)}{N}, I (\boldsymbol {\omega}) \right\} \geq \min _ {n \in [ N ]} \inf _ {\boldsymbol {\omega} \in W _ {n} ^ {N}} \max \left\{F _ {\mathcal {S}} (\boldsymbol {\omega}), I (\boldsymbol {\omega}) \right\} - u / N \\ = \inf _ {\boldsymbol {\omega} \in \operatorname{cl} (W)} \max \left\{F _ {\mathcal {S}} (\boldsymbol {\omega}), I (\boldsymbol {\omega}) \right\} - u / N, \\ \end{array}
$$

where the first inequality is due to (5). As $N$ can be taken arbitrarily large, we conclude this theorem.

![](images/4b5bbed2747d81bc81a8effe76ee3554b3ee66a1866f15e08f874167b5be8025.jpg)

Lemma 1. For any $N \in \mathbb{N}$ , $n \in [N]$ ,

$$
\varliminf_ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} , \boldsymbol {\omega} (t) \in W _ {n} ^ {N} ]} \geq \max \left\{\frac {u (n - 1)}{N}, \inf _ {\boldsymbol {\omega} \in W _ {n} ^ {N}} I (\boldsymbol {\omega}) \right\}
$$

Proof. Recall $F_{\mathcal{S}}(\cdot) = \inf_{\boldsymbol{\lambda} \in \mathrm{cl}(\mathcal{S})} \Psi(\boldsymbol{\lambda}, \cdot)$ . We deduce that

$$
\mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S}, \boldsymbol {\omega} (t) \in W _ {n} ^ {N} \right] \leq \mathbb {P} _ {\boldsymbol {\mu}} \left[ X \geq t F _ {S} (\boldsymbol {\omega} (t)), \boldsymbol {\omega} (t) \in W _ {n} ^ {N} \right],
$$

where $X$ denotes $t\Psi(\hat{\boldsymbol{\mu}}(t),\boldsymbol{\omega}(t))$ for short. Let $\alpha\in(0,1)$ , Markov's inequality implies that

$$
\begin{array}{l} \mathbb {P} _ {\boldsymbol {\mu}} \left[ X \geq t F _ {S} (\boldsymbol {\omega}), \boldsymbol {\omega} (t) \in W _ {n} ^ {N} \right] = \mathbb {P} _ {\boldsymbol {\mu}} \left[ \mathbb {1} \{\boldsymbol {\omega} (t) \in W _ {n} ^ {N} \} e ^ {\alpha (X - t F _ {S} (\boldsymbol {\omega} (t)))} \geq 1 \right] \\ \leq \mathbb {E} _ {\boldsymbol {\mu}} \left[ \mathbb {1} \{\boldsymbol {\omega} (t) \in W _ {n} ^ {N} \} e ^ {\alpha (X - t F _ {S} (\boldsymbol {\omega} (t)))} \right] \\ \leq \mathbb {E} _ {\boldsymbol {\mu}} \left[ \mathbb {1} \{\boldsymbol {\omega} (t) \in W _ {n} ^ {N} \} e ^ {\alpha X} \right] e ^ {- \frac {\alpha u (n - 1)}{N}}, \tag {7} \\ \end{array}
$$

where the last inequality uses the definition of $W_{n}^{N}$ (see (5)). By applying Hölder's inequality with $p, q$ , where $p \in [1, 1 / \alpha)$ and $q = p / (p - 1)$ on r.h.s. of (7), we deduce that $\log \mathbb{P}_{\boldsymbol{\mu}}[\hat{\boldsymbol{\mu}}(t) \in \mathcal{S}, \boldsymbol{\omega}(t) \in W_{n}^{N}]$ is at most

$$
(\log \mathbb {E} _ {\boldsymbol {\mu}} \left[ e ^ {\alpha p X} \right]) / p + (\log \mathbb {E} _ {\boldsymbol {\mu}} [ \mathbb {1} \{\boldsymbol {\omega} (t) \in W _ {n} ^ {N} \} ]) / q - \frac {\alpha u (n - 1)}{N}.
$$

As $\alpha p\in (0,1)$ , Lemma 2 in Appendix B shows that the first term above is $o(t)$ . Using definition of the rate function, (3) with $C = W_{n}^{N}$ , on the second term yields that $\varliminf_{t\to \infty}\frac{1}{t}\log \frac{1}{\mathbb{P}_{\mu}[\hat{\boldsymbol{\mu}}(t)\in\mathcal{S},\boldsymbol{\omega}(t)\in W_n^N]}$ is lower bounded by

$$
(1 / q) \inf _ {\boldsymbol {\omega} \in W _ {n} ^ {N}} I (\boldsymbol {\omega}) + \frac {\alpha u (n - 1)}{N} = (1 - 1 / p) \inf _ {\boldsymbol {\omega} \in W _ {n} ^ {N}} I (\boldsymbol {\omega}) + \frac {\alpha u (n - 1)}{N}.
$$

As $p$ can be arbitrarily close to $1 / \alpha$ , we get the lower bound $(1 - \alpha)I(\omega) + \frac{\alpha u(n - 1)}{N}$ . Further choosing $\alpha$ close to either 1 or 0, the proof is completed.

![](images/06ce8dfe7dcecc56331557fc97d6862c620d6b2e0a023af4d4c9cd591dbe2a68.jpg)

# 3.4 Improved analysis of the Successive Rejects algorithm

In SR, the set of candidate arms is initialized as $C_{K} = [K]$ . The budget of samples is partitioned into K - 1 phases, and at the end of each phase, SR discards the empirical worst arm from the candidate set. In each phase, SR uniformly samples the arms in candidate set. The lengths of phases are set as follows. Define $\overline{\log K} := \frac{1}{2} + \sum_{k=2}^{K} \frac{1}{k}$ . The candidate set is denoted by $C_{j}$ when it has j > 2 arms. In the corresponding phase, (i) each arm in $C_{j}$ is sampled until the round t when $\min_{k \in C_{j}} N_{k}(t)$ reaches $T/(j\overline{\log K})$ (recall that $N_{k}(t)$ is the number of times arm k has been sampled up to round t); (ii) the empirical worst arm, denoted by $\ell_{j}$ , is then discarded, i.e., $C_{j-1} = C_{j} \setminus \{\ell_{j}\}$ . During the last phase, the algorithm equally samples the two remaining arms and finally recommends $\hat{i}$ , the arm with higher empirical mean in $C_{2}$ . The pseudo code is presented in Algorithm 1.

Algorithm 1: SR   
initialization $C_{K} \leftarrow [K], j \leftarrow K;$ for $(t = 1, \ldots, T)$ do
    if $(j > 2 \text{ and min}_{k \in \mathcal{C}_j} N_k(t) \geq \frac{T}{j \overline{\log K}})$ then
    | $\ell_j \leftarrow \arg\min_{k \in \mathcal{C}_j} \hat{\mu}_k(t)$ (tie broken arbitrarily), $C_{j-1} \leftarrow C_j \setminus \{\ell_j\}$ , and $j \leftarrow j - 1$ ;
    end
    sample $A_t \leftarrow \arg\min_{k \in \mathcal{C}_j} N_k(t)$ (tie broken arbitrarily), update $\{N_k(t)\}_{k \in \mathcal{C}_j}$ and $\hat{\mu}(t);$ end $\ell_2 \leftarrow \arg\min_{k \in \mathcal{C}_2} \hat{\mu}_k(T)$ and return $\hat{i} \leftarrow \arg\max_{k \in \mathcal{C}_2} \hat{\mu}_k(T)$ (tie broken arbitrarily).

We apply the corollary (c) in Section 3.2 to improve the existing performance guarantees of SR. To simplify the presentation, we assume wlog that $\mu_1 > \mu_2 \geq \ldots \geq \mu_K$ . For $j = 2, \ldots, K$ , define

$$
\Gamma_ {j} = \min _ {J \in \mathcal {J}} \inf \left\{\sum_ {k \in J} d (\lambda_ {k}, \mu_ {k}): \boldsymbol {\lambda} \in \mathbb {R} ^ {K}, \lambda_ {1} \leq \min _ {k \in J} \lambda_ {k} \right\}, \tag {8}
$$

where $\mathcal{J} = \{J\subseteq [K]:|J| = j,1\in J\}$ .

Theorem 2. Let $\mu \in \Lambda$ . Under SR, we have: for $j = 2, \ldots, K$ , $\varliminf_{T \to \infty} \frac{1}{T} \log \frac{1}{\mathbb{P}_{\mu}[\ell_j = 1]} \geq \frac{\Gamma_j}{j \log K}$ . Hence, the error probability of SR is upper bounded by $\varliminf_{T \to \infty} \frac{1}{T} \log \frac{1}{\mathbb{P}_{\mu}[i \neq 1]} \geq \min_{j \neq 1} \frac{\Gamma_j}{j \log K}$ .

Proof of Theorem 2. Fix $j \in \{2, \dots, K\}$ . Observe that

$$
\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 ] = \sum_ {J \in \mathcal {J}} \mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1, \mathcal {C} _ {j} = J ] \leq | \mathcal {J} | \max _ {J \in \mathcal {J}} \mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1, \mathcal {C} _ {j} = J ],
$$

which implies that

$$
\varliminf_ {T \rightarrow \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 ]} \geq \min _ {J \in \mathcal {J}} \varliminf_ {T \rightarrow \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 , \mathcal {C} _ {j} = J ]} \tag {9}
$$

as $|\mathcal{J}| < \infty$ .

Since $\ell_j$ is selected at the $\theta T$ -th round $^1$ where $\theta = (1 + \sum_{k=j+1}^{K} \frac{1}{k}) / \overline{\log K}$ , the event $\{\ell_j = 1, \mathcal{C}_j = J\}$ implies that $\{\hat{\boldsymbol{\mu}}(\theta T) \in \mathcal{S}, \boldsymbol{\omega}(\theta T) \in W\}$ , where

$$
\mathcal {S} = \left\{\boldsymbol {\lambda} \in \mathbb {R} ^ {K}: \lambda_ {1} \leq \lambda_ {k}, \forall k \in J \right\} \text {   and   } W = \left\{\boldsymbol {\omega} \in \Sigma : \omega_ {k} = \frac {1}{\theta j \overline {{\log}} K}, \forall k \in J \right\}.
$$

In other words, $\mathbb{P}_{\boldsymbol{\mu}}[\ell_j = 1, \mathcal{C}_j = J] \leq \mathbb{P}_{\boldsymbol{\mu}}[\hat{\boldsymbol{\mu}}(\theta T) \in \mathcal{S}, \boldsymbol{\omega}(\theta T) \in W]$ . Applying (c) in Section 3.2 with the above $S$ and $W$ yields that $\varprojlim_{T \to \infty} \frac{1}{\theta T} \log \frac{1}{\mathbb{P}_{\boldsymbol{\mu}}[\hat{\boldsymbol{\mu}}(\theta T) \in \mathcal{S}, \boldsymbol{\omega}(\theta T) \in W]}$ is larger than

$$
\inf _ {\boldsymbol {\omega} \in W} \inf _ {\boldsymbol {\lambda} \in \operatorname{cl} (\mathcal {S})} \Psi (\boldsymbol {\lambda}, \boldsymbol {\omega}) \geq \frac {1}{\theta j \overline {{\log}} K} \inf \left\{\sum_ {k \in J} d \left(\lambda_ {k}, \mu_ {k}\right): \boldsymbol {\lambda} \in \mathbb {R} ^ {K}, \lambda_ {1} \leq \min _ {k \in J} \lambda_ {k} \right\} \geq \frac {\Gamma_ {j}}{\theta j \overline {{\log}} K}, \tag {10}
$$

where the first inequality uses the fact that KL-divergences and the components of $\omega$ are nonnegative, and the second one is due to the definition (8) of $\Gamma_{j}$ . Combining (9) and (10) completes the proof.

The upper bound derived in Theorem 2 is tighter than those recently derived in (Barrier et al., 2022). Indeed, for any $J \in \mathcal{J}$ , since $|J| = j$ , one can find at least one index in $J$ at least larger than $j$ , say $k_{J}$ . Hence,

$$
\Gamma_ {j} \geq \min _ {J \in \mathcal {J}} \inf _ {\boldsymbol {\lambda} \in \mathbb {R} ^ {K}, \lambda_ {1} \leq \lambda_ {k _ {J}}} d (\lambda_ {1}, \mu_ {1}) + d (\lambda_ {k _ {J}}, \mu_ {k _ {J}}) \geq \inf _ {\boldsymbol {\lambda} \in \mathbb {R} ^ {K}, \lambda_ {1} \leq \lambda_ {j}} d (\lambda_ {1}, \mu_ {1}) + d (\lambda_ {j}, \mu_ {j}).
$$

The r.h.s. in the previous inequality corresponds to the upper bounds derived by (Barrier et al., 2022).

To simplify the presentation and avoid rather intricate computations involving the KL-divergences, in the remaining of the paper, we restrict our attention to specific classes of reward distributions.

Assumption 1. The rewards are bounded with values in (0,1). The reward distributions $\nu_{1},\ldots ,\nu_{K}$ are Bernoulli distributions such that $\nu_{a}$ is of mean $a$ , and for any $a\neq b$ , $d(a,b)\geq 2(a - b)^{2}$ (this is a consequence of Pinsker's inequality as rewards are in (0,1)).

Under Assumption 1, we have $\Gamma_{j} \geq 2\xi_{j}$ (a direct consequence of Proposition 3 in Appendix D.1), where for $j = 2, \ldots, K$ ,

$$
\xi_ {j} = \inf \left\{\sum_ {k = 1} ^ {j} (\lambda_ {k} - \mu_ {k}) ^ {2}: \boldsymbol {\lambda} \in [ 0, 1 ] ^ {j}, \lambda_ {1} \leq \min _ {k = 1, \dots , j} \lambda_ {k} \right\}.
$$

We give an explicit expression of $\xi_{j}$ in Proposition 1, presented in Appendix D.1. Moreover, $2\xi_{j}$ is clearly larger than $2\inf_{\lambda_1\leq \lambda_j}\left\{(\lambda_1 - \mu_1)^2 +(\lambda_j - \mu_j)^2\right\} = (\mu_1 - \mu_j)^2$ , and hence $\min_{j\neq 1}\Gamma_j / (j\overline{\log} K)\geq \min_{j\neq 1}(\mu_1 - \mu_j)^2 /(j\overline{\log} K)$ . This implies that our error probability upper bound is better than that derived in (Audibert et al., 2010).

Example 1. To illustrate the improvement brought by Theorem 2 on the performance guarantees of SR, consider the simple example with 3 Bernoulli arms and $\mu = (0.9, 0.1, 0.1)$ . Then $\min_{j \neq 1} (\mu_1 - \mu_j)^2 / (j\overline{\log}3) = 0.16$ for the upper bound presented in (Audibert et al., 2010). From Proposition 1, instead we get $\min_{j \neq 1} 2\xi_j / (j\overline{\log}3) = 0.21$ .

# 4 Continuous Rejects Algorithms

In this section, we present CR, a truly adaptive algorithm that can discard an arm in any round. We propose two variants of the algorithm, CR-C using a conservative criterion to discard arms and CR-A discarding arms more aggressively. Using the Large Deviation results of Theorem 1, we establish error probability upper bounds for both CR-C and CR-A.

# 4.1 The CR-C and CR-A algorithms

As SR, CR initializes its candidate set as $C_{K} = [K]$ . For $j \geq 2$ , $C_{j}$ denotes the candidate set when it is reduced to j arms. When j > 2, the algorithm samples arms in the candidate set $C_{j}$ uniformly until a discarding condition is met. The algorithm then discards the empirically worst arm $\ell_{j} \in C_{j}$ , i.e., $C_{j-1} \leftarrow C_{j} \setminus \{\ell_{j}\}$ . More precisely, in round t, if there are j candidate arms remaining and if $\ell(t)$ denotes the empirically worst candidate arm, the discarding condition is $N_{\ell(t)}(t) > \max_{k \notin C_{j}} N_{k}(t)$ , ( $\forall k \in C_{j}, N_{\ell(t)}(t) = N_{k}(t)$ ) $^{2}$ , and

for $\mathbb{CR} - \mathbb{C}$ : $\min_{k\in \mathcal{C}_j,k\neq \ell (t)}\hat{\mu}_k(t) - \hat{\mu}_{\ell (t)}(t)\geq G\left(\frac{\sum_{k\in\mathcal{C}_j}N_k(t)\overline{\log}j}{T - \sum_{k\notin\mathcal{C}_j}N_k(t)}\right),$ (11)

for $\mathrm{CR - A}$ : $\frac{\sum_{k\in\mathcal{C}_j,k\neq\ell(t)}\hat{\mu}_k(t)}{j-1}-\hat{\mu}_{\ell(t)}(t)\geq G\left(\frac{\sum_{k\in\mathcal{C}_j}N_k(t)\overline{\log}j}{T-\sum_{k\notin\mathcal{C}_j}N_k(t)}\right),$ (12)

where $G(\beta) = 1/\sqrt{\beta} - 1$ for all $\beta > 0$ . The idea behind (11) is to keep the probability of discarding the best arm at most smaller than that of SR while using less budget. Note that (12) is easier to achieve than (11). CR-A is hence more aggressive than CR-C, and reduces the set of arms to $C_{2}$ faster, but at the expense of a higher risk. After discarding $\ell_{3}$ , CR will sample the arms in $C_{2}$ evenly, and recommend the empirical best arm in $C_{2}$ when the budget is exhausted. The pseudo-code of CR is presented in Algorithm 2.

Algorithm 2: CR-C and CR-A   
Input: $\theta_0 \in (0, \frac{1}{\log K}) \cap \mathbb{Q}$ independent of $T$ (can be chosen as small as one wishes)

initialization

| $\mathcal{C}_K \leftarrow [K]$ , $j \leftarrow K$ , sample each arm $k \in [K]$ once, update $\{N_k(t)\}_{k \in \mathcal{C}_K}$ and $\hat{\boldsymbol{\mu}}(t)$ ;

for $t = K + 1, \ldots, \lfloor \theta_0 T \rfloor$ do

| sample $A_t \leftarrow \operatorname{argmin}_{k \in \mathcal{C}_j} N_k(t)$ (tie broken arbitrarily), update $\{N_k(t)\}_{k \in \mathcal{C}_j}$ and $\hat{\boldsymbol{\mu}}(t)$ ;

end

for $(t = \lfloor \theta_0 T \rfloor + 1, \ldots, T)$ do

| $\ell(t) \leftarrow \operatorname{argmin}_{k \in \mathcal{C}_j} \hat{\mu}_k(t)$ (tie broken arbitrarily);

if $j > 2$ , $N_{\ell(t)}(t) > \max_{k \notin \mathcal{C}_j} N_k(t)$ , ( $\forall k \in \mathcal{C}_j$ , $N_{\ell(t)}(t) = N_k(t)$ ), and (11) (resp. (12)) holds for CR-C (resp. CR-A) then

| $\ell_j \leftarrow \ell(t)$ , $\mathcal{C}_{j-1} \leftarrow \mathcal{C}_j \setminus \{\ell_j\}$ , $j \leftarrow j - 1$ sample $A_t \leftarrow \operatorname{argmin}_{k \in \mathcal{C}_j} N_k(t)$ (tie broken arbitrarily), update $\{N_k(t)\}_{k \in \mathcal{C}_j}$ and $\hat{\boldsymbol{\mu}}(t)$ ;

end $\ell_2 \leftarrow \ell(T)$ ; return $\hat{i} \leftarrow \operatorname{argmax}_{k \in \mathcal{C}_2} \hat{\mu}_k(T)$ (tie broken arbitrarily).

# 4.2 Analysis of CR-C and CR-A

As in Section 3.4, $\mu_1 > \mu_2 \geq \ldots \geq \mu_K$ is assumed wlog and we further define $\mu_{K+1} = 0$ . We introduce the following instance-specific quantities needed to state our error probability upper bounds. For $j \in \{2, \ldots, K\}$ , define

$$
\psi_ {j} = \frac {j - 1}{j} \left(\mu_ {1} - \frac {\sum_ {k = 2} ^ {j} \mu_ {k}}{j - 1}\right) ^ {2}, \quad \bar {\psi} _ {j} = \frac {j - 1}{j} \left(\mu_ {1} - \frac {\sum_ {k = 2} ^ {j - 1} \mu_ {k} + \mu_ {j + 1}}{j - 1}\right) ^ {2}, \quad \zeta_ {j} = \mu_ {j} - \mu_ {j + 1},
$$

$$
\varphi_ {j} = \frac {\sum_ {k = 1} ^ {j} \mu_ {k}}{j} - \mu_ {j + 1}, \quad \text { and } \quad \bar {\xi} _ {j} = \inf \left\{\sum_ {k = 1} ^ {K} (\lambda_ {k} - \mu_ {k}) ^ {2}: \boldsymbol {\lambda} \in [ 0, 1 ] ^ {K},   \lambda_ {1} \leq \min _ {k = 2, \ldots , j - 1, j + 1} \lambda_ {k} \right\}.
$$

Here we remark $\bar{\xi}_j \geq \xi_j$ and $\bar{\psi}_j \geq \psi_j$ . These inequalities are proven in Proposition 3 and Proposition 5 in Appendix D.

Theorem 3. Let $\boldsymbol{\mu} \in [0,1]^K$ . Under $\mathbb{CR}-\mathbb{C}$ , $\varprojlim_{T \to \infty} \frac{1}{T} \log \frac{1}{\mathbb{P}_{\boldsymbol{\mu}}[\hat{i} \neq 1]}$ is larger than

$$
2 \min _ {j = 2, \ldots , K} \left\{\frac {\min \left\{\max \left\{\frac {\xi_ {j} \overline {{\log}} (j + 1) (1 - \alpha_ {j}) \mathbb {1} _ {\{j \neq K \}}}{\overline {{\log j}}} , \xi_ {j} \right\} , \bar {\xi} _ {j} \right\}}{j \overline {{\log}} K} \right\},
$$

where $\alpha_{j}\in\mathbb{R}$ is the real number such that $\frac{2\xi_{j}(1-\alpha_{j})}{j\overline{\log j}}=[((1+\zeta_{j})\sqrt{\alpha_{j}}-\sqrt{\frac{1}{(j+1)\overline{\log(j+1)}})+]^{2}$ .

Theorem 4. Let $\boldsymbol{\mu} \in [0,1]^K$ . Under $\mathbb{CR}-\mathbb{A}$ , $\varliminf_{T \to \infty} \frac{1}{T} \log \frac{1}{\mathbb{P}_{\boldsymbol{\mu}}[\hat{i} \neq 1]}$ is larger than

$$
2 \min _ {j = 2, \ldots , K} \left\{\frac {\min \bigl \{\max \bigl \{\frac {\psi_ {j} \overline {{\log}} (j + 1) (1 - \alpha_ {j}) \mathbb {1} _ {\{j \neq K \}}}{\overline {{\log}} j} , \psi_ {j} \bigr \} , \bar {\psi} _ {j} \bigr \}}{j \overline {{\log}} K} \right\},
$$

where $\alpha_{j}\in\mathbb{R}$ is the real number such that $\frac{\psi_{j}(1-\alpha_{j})}{j\log j}=\frac{j}{j+1}[(1+\varphi_{j})\sqrt{\alpha_{j}}-\sqrt{\frac{1}{(j+1)\overline{\log(j+1)}})+]^{2}$ .

Note that Theorem 3 implies that CR-C enjoys better performance guarantees than SR, and hence has for now the best known error probability upper bounds.

Proof sketch. The complete proof of Theorems 3 and 4 are given in Appendices C.1 and C.2. We sketch that of Theorem 3. The proof consists in upper bounding $\mathbb{P}_{\mu}[\ell_j = 1]$ for $j \in \{2, \ldots, K\}$ . We focus here on the most challenging case where $j \in \{3, \ldots, K - 1\}$ (the analysis is simpler when $j = K$ , since the only possible allocation is uniform, and when $j = 2$ , since the only possible round deciding $\ell_2$ is the last round).

To upper bound $P_{\mu}[\ell_{j}=1]$ using Theorem 1, we will show that it is enough to study the large deviations of the process $\{\omega(\theta T)\}_{T\geq1}$ for any fixed $\theta\in[\theta_{0},1]$ and to define a set $S\subseteq[0,1]^{K}$ under which $\ell_{j}=\ell(\theta T)=1$ . We first observe that $\ell_{j}=\ell(\theta T)$ restricts the possible values of $\omega(\theta T)$ : $\omega(\theta T)\in\mathcal{X}_{j}:=\left\{\boldsymbol{x}\in\Sigma:\exists\sigma\in[K]^{2}\text{s.t.}x_{\sigma(1)}=\ldots=x_{\sigma(j)}>x_{\sigma(j+1)}>\ldots>x_{\sigma(K)}>0\right\}$ . We can hence just derive the LDP satisfied by $\{\omega(\theta T)\}_{T\geq1}$ on $X_{j}$ . This is done in Appendix E, and we identify by $I_{\theta}$ a rate function leading to an LDP upper bound. By defining

$$
\mathcal {X} _ {j, i} (\theta) = \left\{\boldsymbol {x} \in \mathcal {X} _ {j}: \theta x _ {\sigma (i)} i \overline {{\log}} i > 1 - \theta \sum_ {k = i + 1} ^ {K} x _ {\sigma (k)} \right\}, \forall i \in \{j, \dots , K \},
$$

As it is shown in Appendix E.1 that $I_{\theta}(\pmb{x}) = \infty$ if $\pmb{x} \in \mathcal{X}_{j,i}(\theta)$ for $i \geq j$ , we may further restrict to $\mathcal{X}_j \setminus \cup_{i=j}^{K} \mathcal{X}_{j,i}(\theta)$ .

Next, we explain how to apply Theorem 1 to upper bound $\mathbb{P}_{\boldsymbol{\mu}}[\ell_j = 1]$ . Let $\mathcal{J} = \{J \subseteq [K] : |J| = j, 1 \in J\}$ as defined in Section 3.4. For all $\beta, \theta \in (0,1]$ and $J \in \mathcal{J}$ , we introduce the sets

$$
\mathcal {S} _ {J} (\beta) = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \min _ {k \in J, k \neq 1} \lambda_ {k} - \lambda_ {1} \geq G (\beta) \right\},
$$

$$
\mathcal {Z} _ {J} (\theta , \beta) = \left\{\boldsymbol {z} \in \mathcal {X} _ {j} \setminus \cup_ {i = j} ^ {K} \mathcal {X} _ {j, i} (\theta): (\forall k \in J, z _ {k} = \max _ {k ^ {\prime} \in [ K ]} z _ {k ^ {\prime}}), \frac {\theta \sum_ {k \in J} z _ {k} \overline {{\log}} j}{1 - \theta \sum_ {k \notin J} z _ {k}} = \beta \right\}.
$$

Assume that in round $t$ , $\ell_j = \ell(t) = 1$ , $\mathcal{C}_j = J$ and let $\tau = \sum_{k \notin J} N_k(t) \leq t$ be the number of times arms outside $J$ are pulled. While $\omega(t) \notin \mathcal{X}_{j,j}(t/T)$ , we have $\beta = \frac{(t-\tau)\overline{\log j}}{T-\tau} \in (0,1]$ . Using the criteria (11), we observe that

$$
\begin{array}{l} \sum_ {t = K + 1} ^ {T} \sum_ {J \in \mathcal {J}} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \ell_ {j} = \ell (t) = 1, \mathcal {C} _ {j} = J, \boldsymbol {\omega} (t) \in \mathcal {X} _ {j} \setminus \cup_ {i = j} ^ {K} \mathcal {X} _ {j, i} (\frac {t}{T}) \right] \\ \leq \sum_ {t = K + 1} ^ {T} \sum_ {J \in \mathcal {J}} \sum_ {\tau \leq t, \tau \in \mathbb {N}} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} _ {J} (\frac {(t - \tau) \overline {{\log j}}}{T - \tau}), \boldsymbol {\omega} (t) \in \mathcal {Z} _ {J} (\frac {t}{T}, \frac {(t - \tau) \overline {{\log j}}}{T - \tau}) \right]. \\ \end{array}
$$

To upper bound the r.h.s. in the above inequality, we combine the results of Theorem 1 and a partitioning technique (presented in Appendix G). This gives:

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (\theta T) \in \mathcal {S} _ {J} (\beta) , \boldsymbol {\omega} (\theta T) \in \mathcal {Z} _ {J} (\theta , \beta) \right]} \geq \theta \inf _ {\boldsymbol {z} \in \mathrm{cl} (\mathcal {Z} _ {J} (\theta , \beta))} \max \{F _ {\mathcal {S} _ {J} (\beta)} (\boldsymbol {z}), I _ {\theta} (\boldsymbol {z}) \}.
$$

The proof is completed by providing lower bounds of $\theta F_{S_J(\beta)}(z)$ and $\theta I_{\theta}(z)$ for a fixed $z \in \mathcal{Z}_J(\theta, \beta)$ with various $J$ . Such bounds are derived in Appendix D.1 and E.3.1, respectively.

Example 2. To conclude this section, we just illustrate through a simple example the gain in terms of performance guarantees brought by CR compared to SR. Assume we have 50 Bernoulli arms with $\mu_{1}=0.95,\mu_{2}=0.85,\mu_{3}=0.2$ , and $\mu_{k}=0$ for $k=4,\ldots,50$ . For SR, Theorem 2 states that with a budget of 5000 samples, the error probability of SR does not exceed $1.93\times10^{-3}$ . With the same budget, Theorems 3 and 4 state that the error probabilities of CR-C and CR-A do not exceed $6.40\times10^{-4}$ and $6.36\times10^{-4}$ , respectively.

We note that in general, we cannot say that one of our two algorithms, CR-C or CR-A, has better guarantees than the other. This is demonstrated in the problem instances presented in Appendix K.1 and K.4.

# 5 Numerical Experiments

We consider various problem instances to numerically evaluate the performance of CR. In these instances, we vary the number of arms from 5 to 55; we use Bernoulli distributed rewards, and vary the shape of the arm-to-reward mapping. For each instance, we compare CR to SR, SH, and UGapE.

Most of our numerical experiments are presented in Appendix K. Due to space constraints, we just provide an example of these results below. In this example, we have 55 arms with convex arm-to-reward mapping. The mapping has 10 steps, and the m-th step consists of m arms with same average reward, equal to $\frac{3}{4} \cdot 3^{-\frac{m}{10}}$ . Table 1 presents the error probabilities averaged over 40,000 independent runs. Observe that CR-A performs better than CR-C (being aggressive when discarding arms has some benefits), and both versions of CR perform better than SR and all other algorithms.

Table 1: Error probability (in %). 

<table><tr><td></td><td></td><td>T=3,000</td><td>T=4,000</td><td>T=5,000</td></tr><tr><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>24.7</td><td>21.3</td><td>18.9</td></tr><tr><td>SH</td><td>Karnin et al. (2013)</td><td>10.2</td><td>5.9</td><td>3.2</td></tr><tr><td>SR</td><td>Audibert et al. (2010)</td><td>5.5</td><td>2.8</td><td>1.3</td></tr><tr><td>CR-C</td><td>(this paper)</td><td>7.1</td><td>2.6</td><td>1.1</td></tr><tr><td>CR-A</td><td>(this paper)</td><td>4.7</td><td>1.6</td><td>0.6</td></tr></table>

# 6 Conclusion

In this paper, we have established, in MAB problems, a connection between the LDP satisfied by the sampling process (under any adaptive algorithm) and that satisfied by the empirical average rewards of the various arms. This connection has allowed us to improve the performance analysis of existing best arm identification algorithms, and to devise and analyze new algorithms with an increased level of adaptiveness. We show that one of these algorithms CR-C has better performance guarantees than existing algorithms and that it performs also better in practice in most cases.

Future research directions include: (i) developing algorithms with further improved performance guarantees – can the discarding conditions of CR be further optimized? (ii) Enhancing the Large Deviation analysis of adaptive algorithms – under which conditions, can we establish a complete LDP of the process $\{Z(t)\}_{t\geq1}=\{(\boldsymbol{\omega}(t),\hat{\boldsymbol{\mu}}(t))\}_{t\geq1}$ ? Answering this question would constitute a strong step towards characterizing the minimal instance-specific error probability for best arm identification with fixed budget. (iii) Extending our approach to other pure exploration tasks: top-m arm identification problems (Bubeck et al., 2013), best arm identification in structured bandits (Yang and Tan, 2022; Azizi et al., 2022), or best policy identification in reinforcement learning.

# Acknowledgements

The authors would like to express their gratitude to Guo-Jhen Wu for his invaluable discussion during the initial stages of this project. R.-C Tzeng's research is supported by the ERC Advanced Grant REBOUND (834862), A. Proutiere is supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation, and Digital Futures.

# References

Ariu, K., Kato, M., Komiyama, J., McAlinn, K., and Qin, C. (2021). Policy choice and best arm identification: Asymptotic analysis of exploration sampling.   
Audibert, J.-Y., Bubeck, S., and Munos, R. (2010). Best arm identification in multi-armed bandits. In COLT, pages 41–53. Citeseer.   
Azizi, M., Kveton, B., and Ghavamzadeh, M. (2022). Fixed-budget best-arm identification in structured bandits. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pages 2798–2804.   
Barrier, A., Garivier, A., and Stoltz, G. (2022). On best-arm identification with a fixed budget in non-parametric multi-armed bandits. arXiv preprint arXiv:2210.00895.   
Berge, C. (1997). Topological Spaces: including a treatment of multi-valued functions, vector spaces, and convexity. Courier Corporation.   
Boyd, S., Boyd, S. P., and Vandenberghe, L. (2004). Convex optimization. Cambridge university press.   
Bubeck, S., Wang, T., and Viswanathan, N. (2013). Multiple identifications in multi-armed bandits. In International Conference on Machine Learning, pages 258–265. PMLR.   
Budhiraja, A. and Dupuis, P. (2019). Analysis and approximation of rare events. Representations and Weak Convergence Methods. Series Prob. Theory and Stoch. Modelling, 94.   
Carpentier, A. and Locatelli, A. (2016). Tight (lower) bounds for the fixed budget best arm identification bandit problem. In Conference on Learning Theory, pages 590–604. PMLR.   
Combes, R., Magureanu, S., and Proutiere, A. (2017). Minimal exploration in structured stochastic bandits. In Proc. of NeurIPS.   
Degenne, R. (2023). On the existence of a complexity in fixed budget bandit identification. In Proc. of COLT.   
Degenne, R. and Koolen, W. M. (2019). Pure exploration with multiple correct answers. In Proc. of NeurIPS.   
Dembo, A. and Zeitouni, O. (2009). Large deviations techniques and applications, volume 38. Springer Science & Business Media.   
Dupuis, P. and Ellis, R. S. (2011). A weak convergence approach to the theory of large deviations. John Wiley & Sons.   
Ellis, R. S. (1984). Large deviations for a general class of random vectors. The Annals of Probability, 12(1):1–12.   
Gabillon, V., Ghavamzadeh, M., and Lazaric, A. (2012). Best arm identification: A unified approach to fixed budget and fixed confidence. Advances in Neural Information Processing Systems, 25.   
Garivier, A. and Kaufmann, E. (2016). Optimal best arm identification with fixed confidence. In Proc. of COLT.   
Garivier, A., Ménard, P., and Stoltz, G. (2019). Explore first, exploit next: The true shape of regret in bandit problems. Mathematics of Operations Research, 44(2):377–399.

Gärtner, J. (1977). On large deviations from the invariant measure. Theory of Probability & Its Applications, 22(1):24–39.   
Glynn, P. and Juneja, S. (2004). A large deviations perspective on ordinal optimization. In Proceedings of the 2004 Winter Simulation Conference, 2004., volume 1. IEEE.   
Karnin, Z., Koren, T., and Somekh, O. (2013). Almost optimal exploration in multi-armed bandits. In International Conference on Machine Learning, pages 1238–1246. PMLR.   
Kaufmann, E., Cappé, O., and Garivier, A. (2016). On the complexity of best-arm identification in multi-armed bandit models. JMLR.   
Kaufmann, E. and Koolen, W. (2018). Mixture martingales revisited with applications to sequential tests and confidence intervals. arXiv preprint arXiv:1811.11419.   
Komiyama, J. (2022). Bayes optimal algorithm is suboptimal in frequentist best arm identification. arXiv preprint arXiv:2202.05193.   
Komiyama, J., Tsuchiya, T., and Honda, J. (2022). Minimax optimal algorithms for fixed-budget best arm identification. In Advances in Neural Information Processing Systems.   
Lai, T. L. and Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. Advances in applied mathematics.   
Magureanu, S., Combes, R., and Proutiere, A. (2014). Lipschitz bandits: Regret lower bound and optimal algorithms. In Conference on Learning Theory, pages 975–999. PMLR.   
Qin, C. (2022). Open problem: Optimal best arm identification with fixed-budget. In Conference on Learning Theory, pages 5650–5654. PMLR.   
Russo, D. (2016). Simple bayesian algorithms for best arm identification. In Annual Conference on Learning Theory. PMLR.   
Varadhan, S. S. (2016). Large deviations, volume 27. American Mathematical Soc.   
Wang, P.-A., Ariu, K., and Proutiere, A. (2023). On uniformly optimal algorithms for best arm identification in two-armed bandits with fixed budget. arXiv preprint arXiv:2308.12000.   
Wang, P.-A., Tzeng, R.-C., and Proutiere, A. (2021). Fast pure exploration via frank-wolfe. Advances in Neural Information Processing Systems, 34.   
Yang, J. and Tan, V. (2022). Minimax optimal fixed-budget best arm identification in linear bandits. Advances in Neural Information Processing Systems, 35:12253–12266.

# Contents

1 Introduction 1   
2 Related Work 3   
3 Large Deviation Analysis of Adaptive Sampling Algorithms 4

3.1 Large Deviation Principles 4   
3.2 Analysis of adaptive sampling algorithms 4   
3.3 Proof of Theorem 1 5   
3.4 Improved analysis of the Successive Rejects algorithm 6

4 Continuous Rejects Algorithms 8

4.1 The CR-C and CR-A algorithms 8   
4.2 Analysis of CR-C and CR-A 8

5 Numerical Experiments 10

6 Conclusion 10

A Notation 15   
B Technical lemmas towards the proof of Theorem 1 16   
C Analysis of CR 17

C.1 Performance analysis of CR-C 17

C.1.1 Upper bound of $\mathbb{P}_{\mu}[\ell_K = 1]$ 17   
C.1.2 Upper bound of $\mathbb{P}_{\mu}[\ell_2 = 1]$ 18   
C.1.3 Upper bound for $\mathbb{P}_{\mu}[\ell_j = 1]$ for $j\in \{3,\dots ,K - 1\}$ . . . . . . . . . . . . . 20

C.2 Performance analysis of CR-A 22

C.2.1 Upper bound of $\mathbb{P}_{\mu}[\ell_K = 1]$ 23   
C.2.2 Upper bound of $\mathbb{P}_{\mu}[\ell_2 = 1]$ 23   
C.2.3 Upper bound for $\mathbb{P}_{\mu}[\ell_j = 1]$ for $j\in \{3,\dots ,K - 1\}$ . . . . . . . . . . . . . . 24

D Optimization Problems 26

D.1 Optimization problems for SR and CR-C 26   
D.2 Optimization problem for CR-A 28

E LDP for the sampling process under CR 30

E.1 A sufficient condition towards an LDP upper bound (3) 30   
E.2 Local LDP upper bound on $\cup_{i = j}^{K}\mathcal{X}_{j,i}(\theta)$ 32   
E.3 Local LDP upper bound on $\mathcal{X}_j\setminus \cup_{i = j}^{K}\mathcal{X}_{j,i}(\theta)$ 36

E.3.1 The allocation process under CR-C 36   
E.3.2 The allocation process under CR-A 38

# F Continuity arguments 40

F.1 Verifying the continuity in Theorem 7 40

# G A partitioning technique for large deviations 42

# H A simple min-max problem 45

# I The LDP conjecture and its consequence 46

# J Discussion on the conjectured lower bound (1) 48

# K Numerical experiments 49

K.1 One group of suboptimal arms 49   
K.2 Two groups of suboptimal arms ..... 51   
K.3 Linear arm-to-reward function 53   
K.4 Concave arm-to-reward function 55   
K.5 Convex arm-to-reward function 57   
K.6 Stair arm-to-reward function 59

# A Notation

<table><tr><td colspan="2">Problem setting</td></tr><tr><td>K</td><td>Number of arms</td></tr><tr><td>[m] for any m ∈ N</td><td>The set {1, 2... ,m}</td></tr><tr><td>νk</td><td>Reward distribution for arm k</td></tr><tr><td>Xk(t)</td><td>Random reward received from pulling arm k in round t</td></tr><tr><td>μ ∈ RK</td><td>Vector of the expected rewards of the various arms</td></tr><tr><td>Λ</td><td>Set of all possible parameters μ</td></tr><tr><td>1(μ)</td><td>Best arm under parameter μ</td></tr><tr><td>T</td><td>Given budget</td></tr><tr><td colspan="2">Quantities related to the error rate lower bound</td></tr><tr><td>ω</td><td>Vector of the proportions of arm draws</td></tr><tr><td>Σ</td><td>Simplex</td></tr><tr><td>Eμ and Pμ</td><td>The expectation and probability measure corresponding to μ</td></tr><tr><td>Alt(μ)</td><td>Set of confusing parameters for μ (whose best arm is not 1(μ))</td></tr><tr><td>d(μ, μ&#x27;)</td><td>KL divergence between the distributions parametrized by μ and μ&#x27;</td></tr><tr><td>kl(a, b)</td><td>KL divergence between two Bernoulli distributions of means a and b</td></tr><tr><td>Ψ(λ, ω)</td><td>∑Kk=1 ωkd(λk, μk)</td></tr><tr><td colspan="2">Notation used in large deviation theory</td></tr><tr><td>cl(S)</td><td>The closure of S</td></tr><tr><td>FS(ω)</td><td>infλ∈cl(S) Ψ(λ, ω)</td></tr><tr><td>I</td><td>Rate function for {ω(t)}t≥1</td></tr><tr><td>B(y, δ)</td><td>The open ball with center y and radius δ</td></tr><tr><td colspan="2">Notation used in the algorithms</td></tr><tr><td>Nk(t)</td><td>Number of pulls of arm k up to t</td></tr><tr><td>ωk(t)</td><td>Nk(t)/t</td></tr><tr><td>At</td><td>The arm pulled in round t</td></tr><tr><td>μk(t)</td><td>∑s=1t Xk(s)1{As=k}/Nk(t)</td></tr><tr><td>i</td><td>Recommended arm</td></tr><tr><td colspan="2">Notation for SR, CR (assuming μ1 &gt; μ2 ≥ ... ≥ μK)</td></tr><tr><td>Cj</td><td>Candidates set with size j</td></tr><tr><td>ℓj</td><td>The arm discarded from Cj</td></tr><tr><td>ℓ(t)</td><td>Empirical worst arm at round t</td></tr><tr><td>logj</td><td>1/2 + ∑jk=2 1/k</td></tr><tr><td>G(β)</td><td>1/√β - 1</td></tr><tr><td>J</td><td>{J⊆ [K] : |J| = j, 1 ∈ J}</td></tr><tr><td>Iθ</td><td>Rate function for {ω(θT)}T≥1</td></tr><tr><td>Γj</td><td>minJ∈J inf {∑k∈J d(λk, μk) : λ ∈ RK, λ1 ≤ mink∈J λk}</td></tr><tr><td>ξj</td><td>infλ∈[0,1]j {∑jk=1(λk - μk)2 : λ1 ≤ mink=1,...,j λk}</td></tr><tr><td>ξj</td><td>infλ∈[0,1]K {∑jk=1(λk - μk)2 : λ1 ≤ mink=2,...,j-1,j+1 λk}</td></tr><tr><td>ψj</td><td>j-1/j (μ1 - ∑jk=2 μk/j-1)2</td></tr><tr><td>ψj</td><td>j-1/j (μ1 - ∑jk=2 μk + μj+1/j-1)2</td></tr><tr><td>ζj</td><td>μj - μj+1</td></tr><tr><td>φj</td><td>∑jk=1 μk/j - μj+1</td></tr></table>

# B Technical lemmas towards the proof of Theorem 1

Lemma 2. Let $\mu \in \Lambda$ , $t > \max \{K, e\}$ and $\beta \in (0,1)$ . There is a constant $c > 0$ (that depends on $K$ and $\beta$ only) s.t.

$$
\mathbb {E} _ {\pmb {\mu}} \left[ e ^ {\beta X} \right] \leq c (\log t) ^ {K},
$$

where $X = \sum_{k=1}^{K} N_k(t) d(\hat{\mu}_k(t), \mu_k)$ .

Proof. Let $M$ be the smallest positive integer s.t. (i) $\frac{\log M}{\beta} > K + 1$ and (ii) $(\log M)^{2K} < M^{\frac{1}{\beta}}$ . We have:

$$
\begin{array}{l} \mathbb {E} _ {\boldsymbol {\mu}} \left[ e ^ {\beta X} \right] \leq \sum_ {n = 0} ^ {\infty} \mathbb {P} _ {\boldsymbol {\mu}} \left[ e ^ {\beta X} \geq n \right] \\ \leq M + \sum_ {n \geq M} \mathbb {P} _ {\boldsymbol {\mu}} \left[ X \geq \frac {\log n}{\beta} \right] \\ \leq M + \sum_ {n \geq M} \frac {\left(2 (\log n) ^ {2} \log t\right) ^ {K}}{n ^ {\frac {1}{\beta}}} \frac {e ^ {K + 1}}{\beta^ {2 K} K ^ {K}}, \tag {13} \\ \end{array}
$$

where the last inequality follows from repeatedly invoking Lemma 3 with $\delta = \frac{\log n}{\beta}$ (notice that $n \geq M$ satisfies the condition on $\delta$ of Lemma 3). Observe that the r.h.s. of (13) is a Bertrand series and it is convergent since $\beta < 1$ . Unfamiliar reader can check the convergence analysis below. (ii) implies that the sequence will decrease after $n \geq M$ , hence the sum in (13) is bounded as (up to a constant multiplicative factor):

$$
\begin{array}{l} \sum_ {n \geq M} \frac {(\log n) ^ {2 K}}{n ^ {\frac {1}{\beta}}} \leq \int_ {1} ^ {\infty} \frac {(\log x) ^ {2 K}}{x ^ {\frac {1}{\beta}}} d x \\ = \int_ {0} ^ {\infty} y ^ {2 K} e ^ {- (\frac {1}{\beta} - 1) y} d y \\ \leq \left(\frac {1}{\beta} - 1\right) ^ {- 2 K - 1} \Gamma (2 K + 1). \tag {14} \\ \end{array}
$$

The constant $c$ can be deduced from (13) and (14).

![](images/a8f1d3f7c55db3ffefdd3887ac3769a49cb6e05cabf3eb40586137536c4ce47b.jpg)

The following lemma is Theorem 2 in Magureanu et al. (2014). It was originally stated for Bernoulli distributions, but as claimed in Garivier and Kaufmann (2016); Kaufmann and Koolen (2018), it is straightforward to generalize it to one-parameter exponential distributions.

Lemma 3 (Magureanu et al. (2014)). For all $\delta > (K + 1)$ and $t \in \mathbb{N}$ , we have:

$$
\mathbb {P} _ {\boldsymbol {\mu}} [ X \geq \delta ] \leq e ^ {- \delta} \left(\frac {\lceil \delta \log t \rceil \delta}{K}\right) ^ {K} e ^ {K + 1}.
$$

# C Analysis of CR

In C.1, we give the proof for Theorem 3 and in C.2 that of Theorem 4. As mentioned in §3.4 and §4.2, in the following analysis, we will assume that $1 \geq \mu_1 > \mu_2 \geq \ldots \geq \mu_K \geq 0$ .

# C.1 Performance analysis of CR-C

Theorem 3. Let $\boldsymbol{\mu} \in [0,1]^K$ . Under $\mathbb{CR}-\mathbb{C}$ , $\varprojlim_{T \to \infty} \frac{1}{T} \log \frac{1}{\mathbb{P}_{\boldsymbol{\mu}}[\hat{i} \neq 1]}$ is larger than

$$
2 \min _ {j = 2, \ldots , K} \left\{\frac {\min \left\{\max \left\{\frac {\xi_ {j} \overline {{\log}} (j + 1) (1 - \alpha_ {j}) \mathbb {1} _ {\{j \neq K \}}}{\overline {{\log}} j} , \xi_ {j} \right\} , \bar {\xi} _ {j} \right\}}{j \overline {{\log}} K} \right\},
$$

where $\alpha_{j}\in \mathbb{R}$ is the real number such that

$$
\frac {2 \xi_ {j} (1 - \alpha_ {j})}{j \overline {{\log}} j} = \left[ \left((1 + \zeta_ {j}) \sqrt {\alpha_ {j}} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2}.
$$

We upper bound $P_{\mu}\left[\ell_{j}=1\right]$ for (i) j=K; (ii) j=2; (iii) $j\in\{3,\ldots,K\}$ . The upper bound for (i), presented in C.1.1, is the easiest to derive as the only possible allocation before one discards the first arm is uniform among all arms. The bound for (ii), presented in C.1.2, is the second easiest to derive as $\ell_{2}$ is decided only in the end, namely, in the T-th round. The upper bound for (iii), presented in C.1.3, is more involved since we have to consider all possible allocations and rounds.

# C.1.1 Upper bound of $P_{\mu}$ $[\ell_{K}=1]$

Lemma 4. Let $\mu \in [0,1]^K$ . Under CR-C,

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {K} = 1 ]} \geq \frac {2 \xi_ {K}}{K \overline {{\log}} K}.
$$

Proof. Without loss of generality, let us assume $\theta_{0}T > K$ . Observe that

$$
\mathbb {P} _ {\boldsymbol {\mu}} \left[ \ell_ {K} = 1 \right] = \sum_ {t \geq \theta_ {0} T} ^ {T} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \ell_ {K} = \ell (t) = 1 \right]. \tag {15}
$$

Since $\mathbb{CR} - \mathbb{C}$ discards $\ell_K = \ell(t)$ at the round $t$ only when $N_{\ell(t)}(t) = N_k(t)$ for all $k \in [K]$ , it suffices to consider $\omega(t) \in \mathcal{X}_K = \{(1/K, \ldots, 1/K)\}$ . We further introduce

$$
\mathcal {S} _ {\theta} = \left\{\boldsymbol {\lambda} \in \mathbb {R} ^ {K}: \lambda_ {1} \leq \min _ {k \neq 1} \lambda_ {k} - G (\theta \overline {{\log}} K) \right\}, \forall \theta \in [ \theta_ {0}, 1 ] \cap \mathbb {Q}.
$$

With this notation, we can use the criteria of discarding the arm $\ell_K = \ell(t) = 1$ (see (11)) to get that:

$$
\sum_ {t \geq \theta_ {0} T} ^ {T} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \ell_ {K} = \ell (t) = 1 \right] \leq \sum_ {t \geq \theta_ {0} T} ^ {T} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} _ {\frac {t}{T}}, \boldsymbol {\omega} (t) \in \mathcal {X} _ {K} \right]. \tag {16}
$$

Applying Theorem 10 in Appendix G with $\tilde{\theta}_{0} = \theta_{0}$

$$
\mathcal {E} = \{\ell_ {K} = 1 \}, \mathcal {S} _ {\theta , \gamma} = \mathcal {S} _ {\theta}, W _ {\theta , \gamma} = \mathcal {X} _ {K}, \forall \gamma ,
$$

yields that

$$
\begin{array}{l} \varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {K} = 1 ]} \geq \inf _ {\theta , \gamma \in [ \theta_ {0}, 1 ] \cap \mathbb {Q}} \inf _ {\boldsymbol {\omega} \in \mathrm{cl} (W _ {\theta , \gamma})} \theta \max \{F _ {\mathcal {S} _ {\theta , \gamma}} (\boldsymbol {\omega}), I _ {\theta} (\boldsymbol {\omega}) \} \\ = \inf _ {\theta \in [ \theta_ {0}, 1 ] \cap \mathbb {Q}} \theta \{F _ {\mathcal {S} _ {\theta}} (1 / K, \dots , 1 / K), I _ {\theta} (1 / K, \dots , 1 / K) \}. \tag {17} \\ \end{array}
$$

Theorem 10 can be indeed applied since, in view of Theorem 5, $\{\omega (\theta T)\}_{T\geq 1}$ satisfies LDP upper bound (3) with rate function $I_{\theta}$ . In the above derivation, by convention, we let $\inf_{\lambda \in \emptyset}f(\lambda) =$

$\infty$ . Next, from Theorem 5 (a) in Appendix E.1, we know that $I_{\theta}(1 / K,\dots ,1 / K) = \infty$ if $\theta >1 / \overline{\log} K$ . Thus, the minimization problem on r.h.s. of (17) can be further lower bounded by $\inf_{\theta \in [\theta_0,1 / \overline{\log} K]\cap \mathbb{Q}}\theta F_{\mathcal{S}_{\theta}}(1 / K,\dots ,1 / K)$ . From the definition of $F_{\mathcal{S}_{\theta}}$ , we have

$$
\begin{array}{l} \theta F _ {\mathcal {S} _ {\theta}} (1 / K, \dots , 1 / K) = \frac {\theta}{K} \inf \left\{\sum_ {k = 1} ^ {K} d \left(\lambda_ {k}, \mu_ {k}\right): \lambda_ {1} \leq \min _ {k \neq 1} \lambda_ {k} - G \left(\theta \overline {{\log}} K\right) \right\} \\ \geq \frac {2 \theta}{K} \inf \left\{\sum_ {k = 1} ^ {K} \left(\lambda_ {k} - \mu_ {k}\right) ^ {2}: \lambda_ {1} \leq \min _ {k \neq 1} \lambda_ {k} - G \left(\theta \overline {{\log}} K\right) \right\} \geq \frac {2 \xi_ {K}}{K \overline {{\log}} K}, \tag {18} \\ \end{array}
$$

where the first inequality is from Assumption 1, and the last inequality follows from Proposition 4 in Appendix D.1 with $\beta = \theta \overline{\log} K$ . The proof is completed combining (17) and (18).

![](images/724ed55ddb3e726ee6de688ee6760808cac2c409dc4474374c98a1a60fb7d68b.jpg)

# C.1.2 Upper bound of $P_{\mu}$ [ $\ell_{2}=1$ ]

Lemma 5. Let $\mu\in[0,1]^{K}$ . Under CR-C,

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {2} = 1 ]} \geq \frac {\min \{\max \{4 \xi_ {2} (1 - \alpha_ {2}) , 3 \xi_ {2} \} , 3 \bar {\xi} _ {2} \}}{3 \overline {{\log}} K},
$$

where $\alpha_{2}\in \mathbb{R}$ is the real number such that

$$
\xi_ {2} \left(1 - \alpha_ {2}\right) = \left[ \left((1 + \zeta_ {2}) \sqrt {\alpha_ {2}} - \frac {1}{2}\right) _ {+} \right] ^ {2}. \tag {19}
$$

Proof. There are two arms remaining in the last phase. Hence, it suffices to consider the estimate and allocation in the last round. The possible allocation $\omega(T)$ belongs to the set

$$
\mathcal {X} _ {2} = \left\{\boldsymbol {x} \in \Sigma : \exists \sigma : [ K ] \mapsto [ K ] \text {s.t.} x _ {\sigma (1)} = x _ {\sigma (2)} > x _ {\sigma (3)} > \dots > x _ {\sigma (K)} > 0 \right\}.
$$

As we defined $\mathcal{J}$ in Section 3.4, we introduce $\mathcal{D} = \{D\subseteq [K]:|D| = 2,1\in D\}$ , and

$$
\mathcal {X} _ {D} = \left\{\boldsymbol {x} \in \mathcal {X} _ {2}: \min _ {k \in D} x _ {k} > \max _ {k ^ {\prime} \notin D} x _ {k ^ {\prime}} \right\}.
$$

The set $\cup_{D\in \mathcal{D}}\mathcal{X}_D$ is a subset of $\mathcal{X}_2$ , and is relevant when we consider events where the best arm 1 is discarded in the last elimination phase. Since $\ell_2$ is decided as the empirical worst arm in $\mathcal{C}_2$ ,

$$
\begin{array}{l} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \ell_ {2} = 1 \right] \leq \sum_ {D \in \mathcal {D}} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (T) \in \mathcal {S} _ {D}, \boldsymbol {\omega} (T) \in \mathcal {X} _ {D} \right] \\ \leq (K - 1) \max _ {D \in \mathcal {D}} \mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (T) \in \mathcal {S} _ {D}, \boldsymbol {\omega} (T) \in \mathcal {X} _ {D} ], \tag {20} \\ \end{array}
$$

where

$$
\mathcal {S} _ {D} = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \min _ {k \in [ D ]} \lambda_ {k} \geq \lambda_ {1} \right\}.
$$

Rearranging (20) yields that

$$
\begin{array}{l} \varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {2} = 1 ]} \geq \varliminf_ {T \to \infty} \min _ {D \in \mathcal {D}} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (T) \in \mathcal {S} _ {D} , \boldsymbol {\omega} (T) \in \mathcal {X} _ {D} ]} \\ \geq \min _ {D \in \mathcal {D}} \varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (T) \in \mathcal {S} _ {D} , \boldsymbol {\omega} (T) \in \mathcal {X} _ {D} ]} \\ \geq \min _ {D \in \mathcal {D}} \inf _ {\boldsymbol {\omega} \in \operatorname{cl} (\mathcal {X} _ {D})} \max \left\{F _ {\mathcal {S} _ {D}} (\boldsymbol {\omega}), I _ {1} (\boldsymbol {\omega}) \right\}, \tag {21} \\ \end{array}
$$

where $I_{1}$ denotes the rate function for which $\{\pmb{\omega}(T)\}_{T\geq 1}$ satisfies an LDP upper bound (3), and the last inequality follows from Theorem 1 with $\mathcal{S} = \mathcal{S}_D$ , $\overline{W} = \mathcal{X}_D$ .

Further introduce for all $i \in \{2, \ldots, K\}$ ,

$$
\mathcal {X} _ {2, i} (1) = \left\{\boldsymbol {x} \in \mathcal {X} _ {2}: x _ {\sigma (i)} i \overline {{\log}} i > 1 - \sum_ {k = i + 1} ^ {K} x _ {\sigma (k)} \right\},
$$

where in this definition, $\sigma$ refers to the permutation used in the definition of $\mathcal{X}_2$ . In Theorem 5 (a) in Appendix E.1, we show that $I_1(\omega) = \infty$ for all $\omega \in \cup_{i=2}^{K} \mathcal{X}_{2,i}(1)$ . Hence any $\omega \in \cup_{i=2}^{K} \mathcal{X}_{2,i}(1)$ cannot be the minimizer on the r.h.s. of (21). In the following, we define

$$
\mathcal {Z} _ {D} (1, 1) = \mathcal {X} _ {D} \setminus \cup_ {i = 2} ^ {K} \mathcal {X} _ {2, i} (1).
$$

Here the argument $(1,1)$ in $\mathcal{Z}_{D}(1,1)$ is to be consistent with our notation in Appendix E, but we abbreviate it as $Z_{D}$ for short below. To get a lower bound of (21), we consider two cases: (a) $D \neq [2]$ ; (b) D = [2].

(a) The case where $D \neq [2]$ . Using Corollary 1 with $\beta = 1, j = 2$ in Appendix D.1 yields that:

$$
\inf _ {\boldsymbol {\omega} \in \operatorname{cl} (\mathcal {X} _ {D})} \max \left\{F _ {\mathcal {S} _ {D}} (\boldsymbol {\omega}), I _ {1} (\boldsymbol {\omega}) \right\} \geq \inf _ {z \in \operatorname{cl} (\mathcal {Z} _ {D})} F _ {\mathcal {S} _ {D}} (\boldsymbol {z})
$$

$$
\geq \left(1 - \sum_ {k \notin D} z _ {k}\right) \inf \left\{\sum_ {k \in D} (\lambda_ {k} - \mu_ {k}) ^ {2}: \boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}, \min _ {k \in D} \lambda_ {k} = \lambda_ {1} \right\} \tag {22}
$$

$$
\geq \left(1 - \sum_ {k \notin D} z _ {k}\right) \bar {\xi} _ {2}
$$

$$
\geq \frac {\xi_ {2}}{\overline {{\log K}}}, \tag {23}
$$

where the last inequality directly comes from Proposition 7 with $\theta = 1$ , $j = 2$ , $i = 3$ in Appendix E.2.

(b) The case where $D = [2]$ . Next, we will show that both $\frac{\xi_2}{\log K}$ and $\frac{4\xi_2(1 - \alpha_2)}{3\log K}$ are lower bounds for $\inf_{\omega \in \mathrm{cl}(\mathcal{X}_D)}\max \{F_{\mathcal{S}_D}(\omega), I_1(\omega)\}$ . The maximum of these hence becomes our lower bound. Together with the conclusion obtained in the case (a), we complete the proof of Lemma 5.

Lower bounding by $\frac{\xi_2}{\log K}$ . Observe that (22) holds also for $D = [2]$ , hence

$$
\inf _ {\boldsymbol {\omega} \in \operatorname{cl} (\mathcal {X} _ {[ 2 ]})} \max \left\{F _ {\mathcal {S} _ {[ 2 ]}} (\boldsymbol {\omega}), I _ {1} (\boldsymbol {\omega}) \right\} \geq \inf _ {\boldsymbol {\omega} \in \operatorname{cl} (\mathcal {Z} _ {[ 2 ]})} F _ {\mathcal {S} _ {[ 2 ]}} (\boldsymbol {\omega})
$$

$$
\geq \left(1 - \sum_ {k = 3} ^ {K} z _ {k}\right) \inf _ {\boldsymbol {\lambda} \in \mathbb {R} ^ {K}, \lambda_ {2} > \lambda_ {1}} \left\{\sum_ {k \in [ 2 ]} \left(\lambda_ {k} - \mu_ {k}\right) ^ {2} \right\}
$$

$$
\geq \left(1 - \sum_ {k = 3} ^ {K} z _ {k}\right) \xi_ {2} \tag {24}
$$

$$
\geq \frac {\xi_ {2}}{\overline {{\log K}}}, \tag {25}
$$

where the second inequality is derived by Proposition 4 with $\theta = 1, \beta = 1, j = 2$ in Appendix D.1, and the last inequality follows from Proposition 7 with j = 2, i = 3 in Appendix E.3.

Lower bounding by $\frac{4\xi_{2}(1-\alpha_{2})}{3\log K}$ . One can derive another lower bound by using $I_{1}$ . In Theorem 5 (b), j=2, $\theta=\beta=1$ in Appendix E.1, we show that $I_{1}$ is a valid lower semi-continuous rate function for an LDP upper bound (3) for the process $\{\omega(T)\}_{T\geq1}$ . In Corollary 3 in Appendix E.3.1, we further show that $I_{1}(\boldsymbol{z})\geq\underline{I_{1}}(\boldsymbol{z})$ for $z\in Z_{[2]}$ , where

$$
\underline {{I}} _ {1} (\boldsymbol {z}) = \frac {4}{3 \overline {{{\log}}} K} \left[ \left((1 + \zeta_ {2}) \sqrt {\frac {z _ {\sigma (3)}}{1 - \sum_ {k = 4} ^ {K} z _ {\sigma (k)}}} - \frac {1}{2}\right) _ {+} \right] ^ {2}, \forall \boldsymbol {z} \in \mathcal {Z} _ {[ 2 ]}. \tag {26}
$$

Instead of using (25), we lower bound $F_{S_{[2]}}(z)$ as:

$$
\begin{array}{l} F _ {S _ {[ 2 ]}} (\boldsymbol {z}) \geq \left(1 - \sum_ {k = 3} ^ {K} z _ {\sigma (k)}\right) \xi_ {2} \\ = \xi_ {2} \left(1 - \sum_ {k = 4} ^ {K} z _ {\sigma (k)}\right) \left(1 - \frac {z _ {\sigma (3)}}{1 - \sum_ {k = 4} ^ {K} z _ {\sigma (k)}}\right) \\ \geq \frac {4 \xi_ {2}}{3 \overline {{\log}} K} \left(1 - \frac {z _ {\sigma (3)}}{1 - \sum_ {k = 4} ^ {K} z _ {\sigma (k)}}\right), \tag {27} \\ \end{array}
$$

where the first inequality follows the derivation of (24) and the last inequality stems from Proposition 7 with $\theta = 1, j = 2, i = 4$ in Appendix E.3. Since $F_{S_{[2]}}(z)$ and $I_1(z)$ are lower bounded by the functions of $\alpha = \frac{z_{\sigma(3)}}{1 - \sum_{k=4}^{K} z_{\sigma(k)}}$ given in (26) and (27), we have:

$$
\begin{array}{l} \inf _ {\boldsymbol {\omega} \in \operatorname{cl} (\mathcal {X} _ {[ 2 ]})} \max \left\{F _ {\mathcal {S} _ {[ 2 ]}} (\boldsymbol {\omega}), \underline {{I}} _ {1} (\boldsymbol {\omega}) \right\} \geq \frac {4}{3 \overline {{\log}} K} \inf _ {\alpha \in \mathbb {R}} \max \left\{\xi_ {2} (1 - \alpha), \left[ ((1 + \zeta_ {2}) \sqrt {\alpha} - \frac {1}{2}) _ {+} \right] ^ {2} \right\} \\ \geq \frac {4 \xi_ {2} (1 - \alpha_ {2})}{3 \overline {{\log}} K}, \tag {28} \\ \end{array}
$$

where the last inequality is due to Lemma 23 in Appendix H and $\alpha_{2}$ is defined in (19). Hence, the maximum of the r.h.s. of (25) and (28) is a lower bound for $\inf_{\omega \in \mathrm{cl}(\mathcal{X}_{[2]})}\max \{F_{\mathcal{S}_{[2]}}(\omega),\underline{I}_1(\omega)\}$

![](images/cc5d24c099bc8c5b10a4a190928296e4019001bed9717863ba1b8f2b6a31bbf0.jpg)

# C.1.3 Upper bound for $\mathbb{P}_{\mu}[\ell_j = 1]$ for $j\in \{3,\dots ,K - 1\}$

Lemma 6. Let $\boldsymbol{\mu} \in [0,1]^K$ , $j \in \{3, \ldots, K-1\}$ . Under $\mathbb{CR}-\mathbb{C}$ ,

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 ]} \geq \frac {2 \min \left\{\max \left\{\frac {\xi_ {j} \overline {{\log (j + 1) (1 - \alpha_ {j})}}}{\overline {{\log j}}} , \xi_ {j} \right\} , \bar {\xi} _ {j} \right\}}{j \overline {{\log}} K},
$$

where $\alpha_{j}\in \mathbb{R}$ is the real number such that

$$
\frac {2 \xi_ {j} (1 - \alpha_ {j})}{j \overline {{\log}} j} = \left[ \left((1 + \zeta_ {j}) \sqrt {\alpha_ {j}} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2}. \tag {29}
$$

Proof. Without loss of generality, we assume $\theta_0T > K$ and $\frac{\theta_0T}{K} \in \mathbb{N}$ .

Observe that $\mathbb{P}_{\boldsymbol{\mu}}[\ell_j = 1] = \sum_{J \in \mathcal{J}} \mathbb{P}_{\boldsymbol{\mu}}[\ell_j = 1, \mathcal{C}_j = J]$ , which directly implies

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 ]} \geq \min _ {J \in \mathcal {J}} \varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 , \mathcal {C} _ {j} = J ]}. \tag {30}
$$

There are j arms remaining while discarding $\ell_{j}$ . The possible allocation $\omega(t)$ belongs to the set

$$
\mathcal {X} _ {j} = \left\{\boldsymbol {x} \in \Sigma : \exists \sigma : [ K ] \mapsto [ K ] \text {s.t.} x _ {\sigma (1)} = \dots = x _ {\sigma (j)} > x _ {\sigma (j + 1)} > \dots > x _ {\sigma (K)} > 0 \right\}.
$$

Suppose that in round $t$ , $\mathcal{C}_j = J$ for some $J \in \mathcal{J}$ and let $\tau = \sum_{k \notin J} N_k(t) \geq \tilde{\theta}_0 T$ , where $\tilde{\theta}_0 = (K - j)\theta_0 / K$ , be the number of times arms outside $J$ are pulled. While $\ell_j = \ell(t) = 1$ , we must have $\hat{\boldsymbol{\mu}}(t) \in \mathcal{S}_J(\frac{(t - \tau)\overline{\log j}}{T - \tau})$ , where

$$
\mathcal {S} _ {J} (\beta) = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \min _ {k \in J, k \neq 1} \lambda_ {k} - \lambda_ {1} \geq G (\beta) \right\}, \forall \beta > 0,
$$

because $\frac{(t - \tau)\overline{\log j}}{T - \tau} = \frac{\sum_{k\in\mathcal{C}_jN_k(t)\overline{\log j}}}{T - \sum_{k\notin\mathcal{C}_jN_k(t)}}$ (recall the discarding condition (11)). Now further introduce $\forall\theta, \beta \in (0,1]$ ,

$$
\mathcal {X} _ {J} (\theta , \beta) = \left\{\boldsymbol {x} \in \mathcal {X} _ {j}: (\forall k \in J, x _ {k} = \max _ {k ^ {\prime} \in [ K ]} x _ {k ^ {\prime}}), \frac {\theta \sum_ {k \in J} x _ {k} \overline {{\log}} j}{1 - \theta \sum_ {k \notin J} x _ {k}} = \beta \right\}.
$$

We then have:

$$
\mathbb {P} _ {\boldsymbol {\mu}} \left[ \ell_ {j} = 1, \mathcal {C} _ {j} = J \right] \leq \sum_ {t \geq \theta_ {0} T} ^ {T} \sum_ {\tau \geq \tilde {\theta} _ {0} T} ^ {t} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} _ {J} (\frac {(t - \tau) \overline {{\log j}}}{T - \tau}), \boldsymbol {\omega} (t) \in \mathcal {X} _ {J} (\frac {t}{T}, \frac {(t - \tau) \overline {{\log j}}}{T - \tau}) \right].
$$

Applying Theorem 10 in Appendix G with $\mathcal{E} = \{\ell_j = 1, \mathcal{C}_j = J\}$ ,

$$
\mathcal {S} _ {\theta , \gamma} = \left\{ \begin{array}{c c} \mathcal {S} _ {J} (\frac {(\theta - \gamma) \overline {{\log j}}}{1 - \gamma}), & \text {if} G (\frac {(\theta - \gamma) \overline {{\log j}}}{1 - \gamma}) \leq 1, \\ \mathcal {S} _ {J} (G ^ {- 1} (1)), & \text {otherwise}, \end{array} \right., \text {and} W _ {\theta , \gamma} = \mathcal {X} _ {J} (\theta , \frac {(\theta - \gamma) \overline {{\log j}}}{1 - \gamma}),
$$

(notice that $\beta = \frac{(t - \tau)\overline{\log j}}{T - \tau} = \frac{(\theta - \gamma)\overline{\log j}}{1 - \gamma}$ and $S_J(\beta) = \emptyset$ if $G(\beta) > 1$ ) yields that

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 , \mathcal {C} _ {j} = J ]} \geq
$$

$$
\inf _ {\theta \in [ \theta_ {0}, 1 ] \cap \mathbb {Q}} \inf _ {\gamma \in [ \bar {\theta} _ {0}, 1 ] \cap \mathbb {Q}} \inf _ {\boldsymbol {x} \in \operatorname{cl} \left(\mathcal {X} _ {J} \left(\theta , \frac {(\theta - \gamma) \overline {{\log j}}}{1 - \gamma}\right)\right)} \theta \max \left\{F _ {\mathcal {S} _ {J} \left(\frac {(\theta - \gamma) \overline {{\log j}}}{1 - \gamma}\right)} (\boldsymbol {x}), I _ {\theta} (\boldsymbol {x}) \right\}. \tag {31}
$$

Further introduce for all $i \in \{j, \ldots, K\}$ ,

$$
\mathcal {X} _ {j, i} (\theta) = \left\{\boldsymbol {x} \in \mathcal {X} _ {j}: \theta x _ {\sigma (i)} i \overline {{\log}} i > 1 - \theta \sum_ {k = i + 1} ^ {K} x _ {\sigma (k)} \right\}.
$$

In Theorem 5 with (a) in Appendix E.1, we show that $I_{\theta}(\omega) = \infty$ for all $\omega \in \cup_{i=j}^{K}\mathcal{X}_{2,i}(\theta)$ , hence any $\omega \in \cup_{i=j}^{K}\mathcal{X}_{j,i}(\theta)$ will not be the minimizer on the r.h.s. of (31). In the following, we define

$$
\mathcal {Z} _ {J} (\theta , \beta) = \mathcal {X} _ {J} (\theta , \beta) \setminus \cup_ {i = j} ^ {K} \mathcal {X} _ {j, i} (\theta).
$$

Consider any $z \in \mathcal{Z}_J(\theta, \beta)$ , as $z \notin \mathcal{X}_{j,j}(\theta)$ , we have

$$
\frac {(\theta - \gamma) \overline {{\log}} j}{1 - \gamma} = \frac {\theta z _ {\sigma (j)} j \overline {{\log}} j}{1 - \theta \sum_ {k > j} z _ {\sigma (k)}} <   1.
$$

Therefore, after excluding the points in $\cup_{i=j}^{K}\mathcal{X}_{j,i}(\theta)$ and setting $\beta=\frac{(\theta-\gamma)\overline{\log j}}{1-\gamma}$ , the r.h.s of (31) can be lower bounded by

$$
\inf _ {\theta , \beta \in (0, 1 ] \cap \mathbb {Q}} \inf _ {\boldsymbol {z} \in \operatorname{cl} (\mathcal {Z} _ {J} (\theta , \beta))} \theta \max \left\{F _ {\mathcal {S} _ {J} (\beta)} (\boldsymbol {z}), I _ {\theta} (\boldsymbol {z}) \right\}. \tag {32}
$$

For lower bounding (32), we consider two cases: (a) $J \neq [j]$ ; (b) J = [2].

(a) The case where $J \neq [j]$ . If $z \in \mathcal{Z}_J(\theta, \beta)$ ,

$$
\begin{array}{l} \theta F _ {S _ {J} (\beta)} (\boldsymbol {z}) = \theta \inf _ {\boldsymbol {\lambda} \in \operatorname{cl} (S _ {J} (\beta))} \Psi (\boldsymbol {\lambda}, \boldsymbol {z}) \\ \geq 2 \inf \left\{\sum_ {k \in J} \theta z _ {k} (\lambda_ {k} - \mu_ {k}) ^ {2}: \min _ {k \in J, k \neq 1} \lambda_ {k} - \lambda_ {1} \geq G (\beta) \right\} \\ = \frac {2}{j \log j} \left(1 - \theta \sum_ {k \notin J} z _ {k}\right) \beta \inf \left\{\sum_ {k \in J} \left(\lambda_ {k} - \mu_ {k}\right) ^ {2}: \min _ {k \in J, k \neq 1} \lambda_ {k} - \lambda_ {1} \geq G (\beta) \right\} (33) \\ \geq \frac {2 \bar {\xi} _ {j}}{j \overline {{\log j}}} (1 - \theta \sum_ {k \notin J} z _ {k}) \\ \geq \frac {2 \bar {\xi} _ {j}}{j \overline {{\log}} K}, (34) \\ \end{array}
$$

where the first inequality is due to Assumption 1; the second equality uses the fact that $\forall k\in J$ , $\theta z_{k} = \frac{(1 - \theta\sum_{k^{\prime}\notin J}z_{k^{\prime}})\beta}{j\log j}$ (as $z\in \mathcal{Z}_J(\theta ,\beta)$ ); the third follows from Corollary 1 in Appendix D.1; the last one is a consequence of Proposition 7 with $i = j + 1$ in Appendix E.3.

(b) The case where $J = [j]$ . We will show that both $\frac{2\xi_j}{j\log K}$ and $\frac{2\xi_j\overline{\log(j+1)}(1 - \alpha_j)}{j\overline{\log j\log K}}$ are lower bounds for the r.h.s. of (32). The maximum of these becomes our lower bound. Together with the conclusion obtained in the case (a), we complete the proof of Lemma 6.

Lower bounding by $\frac{2\xi_j}{j\log K}$ . By Proposition 4 and Proposition 7, we can further lower bound the r.h.s. of (32) as

$$
\begin{array}{l} \theta F _ {S _ {[ j ]} (\beta)} (\boldsymbol {z}) \geq \frac {2}{j \log j} \left(1 - \theta \sum_ {k > j} z _ {k}\right) \beta \inf \left\{\sum_ {k \in [ j ]} \left(\lambda_ {k} - \mu_ {k}\right) ^ {2}: \min _ {k \in [ j ], k \neq 1} \lambda_ {k} - \lambda_ {1} \geq G (\beta) \right\} \\ \geq \frac {2 \xi_ {j}}{j \overline {{\log j}}} \left(1 - \theta \sum_ {k > j} z _ {k}\right) (35) \\ \geq \frac {2 \xi_ {j}}{j \overline {{\log}} K}, (36) \\ \end{array}
$$

where the first inequality corresponds to (33); the second inequality is due to Proposition 4 in Appendix D.1; the last inequality uses Proposition 7 with $i = j + 1$ in Appendix E.3.

Lower bounding by $\frac{2\xi_j\overline{\log(j + 1)(1 - \alpha_j)}}{j\log j\log K}$ . One can derive another lower bound by using $I_{\theta}$ . In Theorem 5 with (b) in Appendix E.1, we show that $I_{\theta}$ is a valid lower semi-continuous rate function for an LDP upper bound (3) for the process $\{\omega (\theta T)\}_{T\geq 1}$ . And in Corollary 3 in Appendix E.3.1, $I_{\theta}(\pmb {z})\geq \underline{I}_{\theta}(\pmb {z})$ for $\pmb {z}\in \mathcal{Z}_{[j]}(\theta ,\beta)$ , where

$$
\underline {{{I}}} _ {\theta} (\boldsymbol {z}) = \frac {\overline {{\log}} (j + 1)}{\theta \overline {{\log}} K} \left[ \left((1 + \zeta_ {j}) \sqrt {\frac {\theta z _ {\sigma (j + 1)}}{1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}}} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2}. \tag {37}
$$

Instead of using (36), we lower bound $F_{S_{[J]}(\beta)}(z)$ as:

$$
\theta F _ {S _ {[ j ]} (\beta)} (\boldsymbol {z}) \geq \frac {2 \xi_ {j}}{j \overline {{\log}} j} \left(1 - \theta \sum_ {k \notin [ j ]} z _ {k}\right) = \frac {2 \xi_ {j}}{j \overline {{\log}} j} \left(1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}\right) \left(1 - \frac {\theta z _ {\sigma (j + 1)}}{1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}}\right)
$$

$$
\geq \frac {2 \xi_ {j} \overline {{\log}} (j + 1)}{j \overline {{\log}} j \overline {{\log}} K} \left(1 - \frac {\theta z _ {\sigma (j + 1)}}{1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}}\right), \tag {38}
$$

where the first inequality is from (35) and the last inequality is due to Proposition 7 with $i = j + 2$ in Appendix E.3. Since $\theta F_{S_{[j](\beta)}}(z)$ and $\theta I_{\theta}(z)$ are lower bounded by the functions of $\alpha = \frac{\theta z_{\sigma(j + 1)}}{1 - \theta \sum_{k = j + 2}^{K} z_{\sigma(k)}}$ given in (37) and (38), we have:

$$
\inf _ {\theta , \beta \in [ 0, 1 ] \cap \mathbb {Q}} \theta \inf _ {\boldsymbol {z} \in \operatorname{cl} (\mathcal {Z} _ {[ j ]} (\theta , \beta))} \max \left\{F _ {\mathcal {S} _ {[ j ]} (\beta)} (\boldsymbol {z}), \underline {{I}} _ {\theta} (\boldsymbol {z}) \right\} \geq
$$

$$
\frac {\overline {{\log}} (j + 1)}{\overline {{\log}} K} \inf _ {\alpha \in \mathbb {R}} \max \left\{\frac {2 \xi_ {j} (1 - \alpha)}{j \overline {{\log}} j}, \left[ \left((1 + \zeta_ {j}) \sqrt {\alpha} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2} \right\}. \tag {39}
$$

By Lemma 23 in Appendix H, (39) is lower bounded by $\frac{2\xi_{j}\overline{\log(j+1)}(1-\alpha_{j})}{j\overline{\log j\log K}}$ , where $\alpha_{j}$ is described in (29).

![](images/4138a66243191f2c25bc7a7e868ba19ff3489f17dd6544b7b4d41ea1c0337fc3.jpg)

# C.2 Performance analysis of CR-A

Theorem 4. Let $\boldsymbol{\mu} \in [0,1]^K$ . Under $\mathrm{CR - A}$ , $\varliminf_{T \to \infty} \frac{1}{T} \log \frac{1}{\mathbb{P}_{\boldsymbol{\mu}}[\hat{i} \neq 1]}$ is larger than

$$
2 \min _ {j = 2, \ldots , K} \left\{\frac {\min \{\max \{\frac {\psi_ {j} \overline {{\log (j + 1) (1 - \alpha_ {j}) \mathbb {1}}} _ {\{j \neq K \}}}{\overline {{\log j}}} , \psi_ {j} \} , \overline {{\psi}} _ {j} \}}{j \overline {{\log}} K} \right\},
$$

where $\alpha_{j}\in \mathbb{R}$ is the real number such that

$$
\frac {\psi_ {j} (1 - \alpha_ {j})}{j \overline {{\log}} j} = \frac {j}{j + 1} \left[ \left((1 + \varphi_ {j}) \sqrt {\alpha_ {j}} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2}.
$$

We upper bound $P_{\mu}\left[\ell_{j}=1\right]$ for (i) j=K; (ii) j=2; (iii) $j\in\{3,\ldots,K\}$ . The upper bound for (i), presented in C.2.1, is the easiest to derive as the only possible allocation before one discards the first arm is uniform among all arms. The bound for (ii), presented in C.2.2, is the second easiest to derive as $\ell_{2}$ is decided only in the end, namely, in the T-th round. The upper bound for (iii), presented in C.2.3, is more involved since we have to consider all possible allocations and rounds. Overall, the analysis is very similar to that of CR-C, and we just sketch the arguments below.

# C.2.1 Upper bound of $P_{\mu}$ $[\ell_{K}=1]$

Lemma 7. Let $\mu\in[0,1]^{K}$ . Under CR-A,

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {K} = 1 ]} \geq \frac {2 \psi_ {K}}{K \overline {{\log}} K}.
$$

Proof. The proof follows the steps similar to those in the proof in Appendix C.1.1.

Applying Theorem (10) to

$$
\sum_ {t \geq \theta_ {0} T} ^ {T} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \ell_ {K} = \ell (t) = 1 \right] \leq \sum_ {t \geq \theta_ {0} T} ^ {T} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} _ {\frac {t}{T}}, \boldsymbol {\omega} (t) \in \mathcal {X} _ {K} \right], \tag {40}
$$

with

$$
\mathcal {S} _ {\theta} = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \lambda_ {1} \leq \frac {\sum_ {k = 2} ^ {K} \lambda_ {k}}{K - 1} - G (\theta \overline {{\log}} K) \right\}, \forall \theta \in (0, \frac {1}{\overline {{\log}} K} ],
$$

we obtain

$$
\varliminf_ {T \rightarrow \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {K} = 1 ]} \geq \min _ {\theta \in [ \theta_ {0}, 1 ] \cap \mathbb {Q}} \theta \left\{F _ {\mathcal {S} _ {\theta}} (1 / K, \dots , 1 / K), I _ {\theta} (1 / K, \dots , 1 / K) \right\}. \tag {41}
$$

Next, from Theorem 6 (a) in Appendix E.1, we know that $I_{\theta}(1 / K,\dots ,1 / K) = \infty$ if $\theta >1 / \overline{\log} K$ . Finally, Assumption 1 and Proposition 6 in Appendix D.2 yields that

$$
\min _ {\theta \in [ \theta_ {0}, 1 / \overline {{\log (K)}} ] \cap \mathbb {Q}} \theta \{F _ {\mathcal {S} _ {\theta}} (1 / K, \dots , 1 / K), I _ {\theta} (1 / K, \dots , 1 / K) \} \geq \frac {2 \psi_ {K}}{K \overline {{\log K}}}.
$$

![](images/7b7dc764d0854d06f159e3de4bfba2113dc9f7fb2bbc3a19a8d056a03b4d09cd.jpg)

# C.2.2 Upper bound of $P_{\mu}$ [ $\ell_{2}=1$ ]

Lemma 8. Let $\pmb{\mu} \in [0,1]^K$ . Under $\mathbb{CR}-\mathbb{A}$ ,

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {2} = 1 ]} \geq \frac {\min \{\max \{4 \psi_ {2} (1 - \alpha_ {2}) , 3 \psi_ {2} \} , 3 \bar {\psi} _ {2} \}}{3 \overline {{\log}} K},
$$

where $\alpha_{2}\in \mathbb{R}$ is the real number such that

$$
3 \psi_ {2} (1 - \alpha_ {2}) = 4 \left[ \left((1 + \varphi_ {2}) \sqrt {\alpha_ {2}} - \frac {1}{2}\right) _ {+} \right] ^ {2}. \tag {42}
$$

Proof. The proof follows the same steps as those of the proof in Appendix C.1.2.

Applying Theorem 1 with $\mathcal{S} = \mathcal{S}_D$ , $W = \mathcal{X}_D$ , where

$$
\mathcal {S} _ {D} = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \min _ {k \in [ D ]} \lambda_ {k} \geq \lambda_ {1} \right\},
$$

we get

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {2} = 1 ]} \geq \min _ {D \in \mathcal {D}} \inf _ {\boldsymbol {\omega} \in \mathrm{cl} (\mathcal {X} _ {D})} \max \{F _ {\mathcal {S} _ {D}} (\boldsymbol {\omega}), I _ {1} (\boldsymbol {\omega}) \}, \tag {43}
$$

Then we exclude the points in $\cup_{i=2}^{K}\mathcal{X}_{2,i}(1)$ using Theorem 6 (a) in Appendix E.1 and we define

$$
\mathcal {Z} _ {D} (1, 1) = \mathcal {X} _ {D} \setminus \cup_ {i = 2} ^ {K} \mathcal {X} _ {2, i} (1).
$$

Consider two cases: (a) $D \neq [2]$ ; (b) D = [2].

(a) The case where $D \neq [2]$ . Using Corollary 2 with $\beta = 1, j = 2$ in Appendix D.2 and Proposition 7 with $\theta = 1, j = 2, i = 3$ in Appendix E.2 yields that

$$
\inf _ {\boldsymbol {\omega} \in \operatorname{cl} (\mathcal {X} _ {D})} \max \left\{F _ {\mathcal {S} _ {D}} (\boldsymbol {\omega}), I _ {1} (\boldsymbol {\omega}) \right\} \geq \frac {\bar {\psi} _ {2}}{\overline {{\log}} K}.
$$

(b) The case where $D = [2]$ . We show that both $\frac{\psi_2}{\log K}$ and $\frac{4\psi_2(1 - \alpha_2)}{3\log K}$ are lower bounds for $\varinjlim_{T\to \infty}\frac{1}{T}\log \frac{1}{\mathbb{P}_{\mu}[\hat{\boldsymbol{\mu}}(T)\in\mathcal{S}_{[2]},\boldsymbol{\omega}(T)\in\mathcal{Z}_{[2]}]}$ . The maximum of these hence becomes our lower bound. Together with the conclusion obtained in the case (a), we complete the proof of Lemma 8.

Lower bounding by $\frac{\psi_2}{\log K}$ . Applying Proposition 6 with $\theta = 1, \beta = 1, j = 2$ in Appendix D.2, and Proposition 7 with $j = 2, i = 3$ in Appendix E.3, we can obtain:

$$
\inf _ {\boldsymbol {\omega} \in \operatorname{cl} (\mathcal {X} _ {[ 2 ]})} \max \left\{F _ {\mathcal {S} _ {[ 2 ]}} (\boldsymbol {\omega}), I _ {1} (\boldsymbol {\omega}) \right\} \geq \frac {\psi_ {2}}{\overline {{\log}} K}.
$$

Lower bounding by $\frac{4\psi_2(1 - \alpha_2)}{3\log K}$ . In Theorem 6 (b), $j = 2, \theta = \beta = 1$ in Appendix E.1, we show that $\overline{I_1}$ is a valid rate function for an LDP upper bound (3) for the process $\{\omega(T)\}_{T \geq 1}$ . And Corollary 4 in Appendix E.3.2 show $I_1(z) > \underline{I}_1(z)$ for $z \in \mathcal{Z}_{[2]}$ , where

$$
\underline {{I}} _ {1} (\boldsymbol {z}) = \frac {1 6}{9 \overline {{{\log}}} K} \left[ \left((1 + \varphi_ {2}) \sqrt {\frac {z _ {\sigma (3)}}{1 - \sum_ {k = 4} ^ {K} z _ {\sigma (k)}}} - \frac {1}{2}\right) _ {+} \right] ^ {2}, \forall \boldsymbol {z} \in \mathcal {Z} _ {[ 2 ]}. \tag {44}
$$

Also, we lower bound $F_{S_{[2]}}(z)$ by Proposition 7 with $\theta = 1, j = 2, i = 4$ in Appendix E.3:

$$
F _ {S _ {[ 2 ]}} (\boldsymbol {z}) \geq \frac {4 \psi_ {2}}{3 \overline {{\log}} K} \left(1 - \frac {z _ {\sigma (3)}}{1 - \sum_ {k = 4} ^ {K} z _ {\sigma (k)}}\right). \tag {45}
$$

Observe that $F_{S_{[2]}}(z)$ and $I_{1}(z)$ are lower bounded by the functions of $\alpha = \frac{z_{\sigma(3)}}{1 - \sum_{k=4}^{K} z_{\sigma(k)}}$ given in (44) and (45). Applying Lemma 23 in Appendix H yields that

$$
\inf _ {\boldsymbol {\omega} \in \operatorname{cl} \left(\mathcal {X} _ {[ 2 ]}\right)} \max \left\{F _ {\mathcal {S} _ {[ 2 ]}} (\boldsymbol {\omega}), \underline {{I}} _ {1} (\boldsymbol {\omega}) \right\} \geq \frac {4 \psi_ {2} \left(1 - \alpha_ {2}\right)}{3 \log K}. \tag {46}
$$

# C.2.3 Upper bound for $P_{\mu}\left[\ell_{j}=1\right]$ for $j\in\{3,\ldots,K-1\}$

Lemma 9. Let $\mu \in [0,1]^K$ , $j \in \{3,\ldots,K-1\}$ . Under CR-A,

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 ]} \geq \frac {2 \min \left\{\max \left\{\frac {\psi_ {j} \overline {{\log (j + 1)}} (1 - \alpha_ {j})}{\overline {{\log j}}} , \psi_ {j} \right\} , \bar {\psi} _ {j} \right\}}{j \overline {{\log}} K},
$$

where $\alpha_{j}\in \mathbb{R}$ is the real number such that

$$
\frac {\psi_ {j} (1 - \alpha_ {j})}{j \overline {{\log}} j} = \frac {j}{j + 1} \left[ \left((1 + \varphi_ {j}) \sqrt {\alpha_ {j}} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2}. \tag {47}
$$

Proof. We proceed as in Appendix C.1.3. We have:

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 ]} \geq \min _ {J \in \mathcal {J}} \varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 , \mathcal {C} _ {j} = J ]}. \tag {48}
$$

We then introduce

$$
\mathcal {S} _ {J} (\beta) = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \frac {\sum_ {k \in J , k \neq 1} \lambda_ {k}}{j - 1} - \lambda_ {1} \geq G (\beta) \right\}
$$

$$
\mathcal {X} _ {J} (\theta , \beta) = \left\{\boldsymbol {z} \in \mathcal {X} _ {j}: (\forall k \in J, x _ {k} = \max _ {k ^ {\prime} \in [ K ]} x _ {k ^ {\prime}}), \frac {\theta \sum_ {x \in J} z _ {k} \overline {{\log}} j}{1 - \theta \sum_ {k \notin J} x _ {k}} = \beta \right\}.
$$

Theorem 10 yields that for each $J \in \mathcal{J}$ ,

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \ell_ {j} = 1 , \mathcal {C} _ {j} = J ]} \geq \inf _ {\theta , \beta \in (0, 1 ] \cap \mathbb {Q}} \theta \inf _ {\boldsymbol {z} \in \mathrm{cl} (\mathcal {Z} _ {J} (\theta , \beta))} \max \left\{F _ {\mathcal {S} _ {J} (\beta)} (\boldsymbol {z}), I _ {\theta} (\boldsymbol {z}) \right\}, \tag {49}
$$

where $\mathcal{Z}_J(\theta, \beta) = \mathcal{X}_J(\theta, \beta) \setminus \cup_{i=j}^K \mathcal{X}_{j,i}(\theta)$ .

(a) The case where $J \neq [j]$ . Corollary 2 in Appendix D.2 and Proposition 7 with $i = j + 1$ in Appendix E.3 yields:

$$
\theta F _ {\mathcal {S} _ {J} (\beta)} (z) \geq \frac {2 \bar {\psi} _ {j}}{j \overline {{\log}} K}.
$$

(b) The case where $J = [j]$ . We show both $\frac{2\psi_j}{j\log K}$ and $\frac{2\psi_j\overline{\log(j+1)}(1-\alpha_j)}{j\log j\overline{\log K}}$ are lower bounds of (49). The maximum of these becomes our lower bound.

Lower bounding $\frac{2\psi_j}{j\log K}$ . By Proposition 6 and Proposition 7, Proposition 6 in Appendix D.2 and Proposition 7 with $i = j + 1$ in Appendix E.3, we have

$$
\theta F _ {S _ {J} (\beta)} (z) \geq \frac {2 \psi_ {j}}{j \overline {{\log}} K}. \tag {50}
$$

Lower bounding by $\frac{2\psi_j\overline{\log(j + 1)(1 - \alpha_j)}}{j\overline{\log j}\overline{\log K}}$ . A similar argument as above implies

$$
(4 9) \geq \frac {2 \overline {{\log}} (j + 1)}{\overline {{\log}} K} \inf _ {\alpha \in \mathbb {R}} \max \left\{\frac {\psi_ {j} (1 - \alpha)}{j \overline {{\log}} j}, \frac {j}{j + 1} \left[ \left((1 + \varphi_ {j}) \sqrt {\alpha} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2} \right\}. \tag {51}
$$

By Lemma 23 in Appendix H, (49) is lower bounded by $\frac{2\psi_{j}\overline{\log(j+1)}(1-\alpha_{j})}{j\overline{\log j\log K}}$ .

![](images/1ea09a01fdbec0a6e444b75437fb743620bd639a933de02c07dae2a9c86ea44f.jpg)

# D Optimization Problems

This section provides results related to the various optimization problems we encounter in the paper. In D.1, we compute the $\xi_{j}$ 's appearing in the performance guarantees of SR and CR-C, and prove other useful results. In D.2, we focus on computing the $\psi_{j}$ 's, useful for the performance analysis of CR-A.

# D.1 Optimization problems for SR and CR-C

Let $j \in \{2, \ldots, K\}$ and let $\boldsymbol{\mu} \in \mathbb{R}^{K}$ such that $\mu_{1} > \mu_{2} \geq \ldots \geq \mu_{K}$ . Denote by $\xi_{j}$ the optimal value of the following optimization problem:

$$
\inf \left\{\sum_ {k = 1} ^ {j} (\lambda_ {k} - \mu_ {k}) ^ {2}: \boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}, \lambda_ {1} \leq \min _ {k \neq 1} \lambda_ {k} \right\}. \tag {52}
$$

We first show Proposition 1, restated below for convenience, and deduce some related results.

Proposition 1. We have:

$$
\xi_ {j} = \left\{ \begin{array}{l l} \sum_ {k = 1, j} \left(\mu_ {k} - \frac {\mu_ {1} + \mu_ {j}}{2}\right) ^ {2}, & i f \mu_ {j - 1} \geq \frac {\mu_ {1} + \mu_ {j}}{2}, \\ \sum_ {k = 1, j - 1, j} \left(\mu_ {k} - \frac {\mu_ {1} + \mu_ {j - 1} + \mu_ {j}}{3}\right) ^ {2}, & i f \mu_ {j - 1} <   \frac {\mu_ {1} + \mu_ {j}}{2}, \mu_ {j - 2} \geq \frac {\mu_ {1} + \mu_ {j - 1} + \mu_ {j}}{3}, \\ \vdots & \vdots \\ \sum_ {k = 1} ^ {j} \left(\mu_ {k} - \frac {\sum_ {i = 1} ^ {j} \mu_ {i}}{j}\right) ^ {2}, & i f \mu_ {j - 1} <   \frac {\mu_ {1} + \mu_ {j}}{2}, \ldots , \mu_ {2} <   \frac {\mu_ {1} + \mu_ {3} + \ldots + \mu_ {j}}{j - 1}. \end{array} \right.
$$

Proof. The objective function and the functions defining the constraints in (52) are all convex. There exists $\lambda \in R^{K}$ s.t. all the constraints are strict (Slater condition). Hence we can identify the solution of (52) by just verifying the KKT conditions. The Lagrangian of the problem is

$$
\mathcal {L} _ {\boldsymbol {\mu}} (\boldsymbol {\lambda}, \eta_ {2}, \ldots , \eta_ {j}) = \frac {1}{2} \sum_ {k = 1} ^ {j} (\lambda_ {k} - \mu_ {k}) ^ {2} + \sum_ {k = 2} ^ {j} \eta_ {k} (\lambda_ {1} - \lambda_ {k}), \mathrm{for} (\boldsymbol {\lambda}, \boldsymbol {\eta}) \in \mathbb {R} ^ {K} \times \mathbb {R} _ {\geq 0} ^ {j - 1}.
$$

Let $(\lambda^{\star},\eta^{\star})$ be a saddle point of $\mathcal{L}$ . It satisfies KKT conditions:

$$
\lambda_ {1} ^ {\star} \leq \lambda_ {k} ^ {\star}, \text {   for   } k = 2, \dots , j, \tag {PrimalFeasibility}
$$

$$
\eta_ {k} ^ {\star} \geq 0, \text {   for   } k = 2, \dots , j, \tag {DualFeasibility}
$$

$$
\lambda_ {1} ^ {\star} - \mu_ {1} + \sum_ {k = 2} ^ {j} \eta_ {k} ^ {\star} = 0; \lambda_ {k} ^ {\star} - \mu_ {k} - \eta_ {k} ^ {\star} = 0, \text {   for   } k = 2, \dots , j, \tag {Stationarity}
$$

$$
\eta_ {k} ^ {\star} (\lambda_ {1} ^ {\star} - \lambda_ {k} ^ {\star}) = 0, \text {   for   } k = 2, \ldots , j. \tag {Complementarity}
$$

Let $i \in \{2, \ldots, j\}$ be the smallest index such that

$$
\mu_ {i} <   \frac {\mu_ {1} + \sum_ {i <   k \leq j} \mu_ {k}}{j - i + 1}.
$$

One can easily see the point $(\lambda^{\star},\eta^{\star})$ defined in (53) satisfies the KKT conditions listed above.

$$
\lambda_ {k} ^ {\star} = \left\{ \begin{array}{l l} \frac {\mu_ {1} + \sum_ {k = i} ^ {j} \mu_ {k}}{j - i + 2}, & \text { if } k = 1, i, \dots , j, \\ \mu_ {k}, & \text { otherwise }, \end{array} \quad \eta_ {k} ^ {\star} = \left\{ \begin{array}{l l} \frac {\mu_ {1} + \sum_ {k = i} ^ {j} \mu_ {k}}{j - i + 2} - \mu_ {k} & \text { if } k = 1, i, \dots , j, \\ 0, & \text { otherwise }. \end{array} \right. \right. \tag {53}
$$

□

We are now interested in quantifying the impact of $\mu$ on the value of $\xi_{j}$ . We investigate this impact in the following two propositions.

Proposition 2. Assume that $\xi_{j} = \sum_{b\in B}(\mu_{b} - A)^{2}$ for some $B\subseteq [j]$ and for $A = \frac{\sum_{b\in B}\mu_b}{|B|}$ . Let $S$ be such that $S_{1} = \sum_{b\in B,b\neq 1}S_{b}$ , where $S_{b}\geq 0$ for all $b\neq 1$ . Consider another parameter $\pmb{\mu}'$ defined as $\mu_1' = \mu_1 + S_1$ and $\mu_b' = \mu_b - S_b$ for all $b\in B, b\neq 1$ . Then (i) $\frac{\sum_{b\in B}\mu_b'}{|B|} = A$ ; (ii)

$$
\inf \left\{\sum_ {k \in B} (\lambda_ {k} ^ {\prime} - \mu_ {k} ^ {\prime}) ^ {2}: \boldsymbol {\lambda} ^ {\prime} \in [ 0, 1 ] ^ {K}, \lambda_ {1} ^ {\prime} \leq \min _ {b \in B} \lambda_ {b} ^ {\prime} \right\} = \sum_ {k \in B} (\mu_ {b} ^ {\prime} - A) ^ {2}.
$$

Proof. (i) is trivial. We now prove (ii). Using Proposition 1 and the fact that $\xi_{j} = \sum_{b\in B}(\mu_{b} - A)^{2}$ , we get that

$$
\forall b \neq 1, b \in B, \mu_ {b} <   \frac {\mu_ {1} + \sum_ {k > b} \mu_ {k}}{j - b + 1}. \tag {54}
$$

Also, as $S_{b} \geq 0$ ,

$$
\begin{array}{l} \mu_ {b} ^ {\prime} = \mu_ {b} - S _ {b} \leq \mu_ {b} <   \frac {\mu_ {1} + \sum_ {k > b} \mu_ {k}}{j - b + 1} \\ = \frac {1}{j - b + 1} \left(\mu_ {1} - S _ {1} + \sum_ {k > b} \mu_ {k} + \sum_ {k > b} S _ {k}\right) \\ = \frac {\mu_ {1} ^ {\prime} + \sum_ {k > b} \mu_ {k} ^ {\prime}}{j - b + 1}, \\ \end{array}
$$

where the second inequality is due to (54). By (i) and Proposition 1 again, we conclude the proof. $\square$

Proposition 3. Consider the optimization problem (52) instantiated with another $\mu' \in [0,1]^K$ which satisfies that $\mu_1' \geq \mu_1$ and $\mu_k' \leq \mu_k$ for all $k = 2, \ldots, j$ , and denote its value by $\xi_j'$ . Then $\xi_j' \geq \xi_j$ .

Proof. Consider the Lagrangians of the two optimization problems: $L_{\mu}$ and $L_{\mu'}$ . The corresponding Lagrange dual functions are: $g_{\mu}(\eta) = \min_{\lambda \in \mathbb{R}^{K}} \mathcal{L}_{\mu}(\lambda, \eta)$ and $g_{\mu'}(\eta) = \min_{\lambda \in \mathbb{R}^{K}} \mathcal{L}_{\mu'}(\lambda, \eta)$ and one can easily verify that

$$
g _ {\boldsymbol {\mu}} (\boldsymbol {\eta}) = \frac {1}{2} \left[ \left(\sum_ {k = 2} ^ {j} \eta_ {k}\right) ^ {2} + \sum_ {k = 2} ^ {j} \eta_ {k} ^ {2} \right] + \sum_ {k = 2} ^ {j} \eta_ {k} \left(\mu_ {1} - \mu_ {k} - \sum_ {i \neq k} \eta_ {i}\right),
$$

$$
g _ {\boldsymbol {\mu} ^ {\prime}} (\boldsymbol {\eta}) = \frac {1}{2} \left[ (\sum_ {k = 2} ^ {j} \eta_ {k}) ^ {2} + \sum_ {k = 2} ^ {j} \eta_ {k} ^ {2} \right] + \sum_ {k = 2} ^ {j} \eta_ {k} (\mu_ {1} ^ {\prime} - \mu_ {k} ^ {\prime} - \sum_ {i \neq k} \eta_ {i}).
$$

Recall $\mu_{1} - \mu_{k}\leq \mu_{1}^{\prime} - \mu_{k}^{\prime}$ and $\pmb {\eta}\in \mathbb{R}_{\geq 0}^{j - 1}$ , hence $g_{\pmb {\mu}}(\pmb {\eta})\leq g_{\pmb{\mu}^{\prime}}(\pmb {\eta})$ for all $\pmb {\eta}\in \mathbb{R}_{\geq 0}^{j - 1}$ . For (52), Slater condition holds clearly, hence strong duality follows (see e.g. (Boyd et al., 2004) Chapter 5.5.3). Thus, $\xi_{j} = \max_{\pmb {\eta}\in \mathbb{R}_{+}^{j - 1}}g_{\pmb {\mu}}(\pmb {\eta})\leq \max_{\pmb {\eta}\in \mathbb{R}_{+}^{j - 1}}g_{\pmb{\mu}^{\prime}}(\pmb {\eta}) = \xi_{j}^{\prime}$ .

The following result relates the function G to $\xi_{j}$ and is instrumental in the proof of Theorem 3.

Proposition 4. $\forall\beta\in(0,1],\mu\in[0,1]^{K},\text{and}2\leq j\leq K,\text{one has}$

$$
\beta \inf \left\{\sum_ {k = 1} ^ {j} (\lambda_ {k} - \mu_ {k}) ^ {2}: \boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}, \lambda_ {1} \leq \min _ {k = 2, \dots , j} \lambda_ {k} - G (\beta) \right\} \geq \xi_ {j}. \tag {55}
$$

Proof. Let $B \subseteq [j]$ s.t. $\xi_j = \sum_{b \in B} (\mu_b - A)^2$ , where $A = \frac{\sum_{b \in B} \mu_b}{|B|}$ . Using the fact that $\sum_{k \notin B} (\mu_k - \lambda_k)^2 \geq 0$ for all $\boldsymbol{\lambda} \in \mathbb{R}^K$ , one can deduce that

l.h.s. of (55) $\geq \beta$ inf $\left\{\sum_{b\in B}(\lambda_b - \mu_b)^2:\lambda_1\leq \lambda_b - G(\beta),\forall b\in B,b\neq 1\right\}$

$$
\geq \beta \inf \left\{\sum_ {b \in B} (\lambda_ {b} - \mu_ {b}) ^ {2}: \lambda_ {1} \leq \lambda_ {b} - (\mu_ {1} - \mu_ {b}) G (\beta), \forall b \in B, b \neq 1 \right\}
$$

$$
= \beta \inf \left\{\sum_ {b \in B} (\lambda_ {b} - \mu_ {b}) ^ {2}: \lambda_ {1} + (\mu_ {1} - A) G (\beta) \leq \lambda_ {k} - (A - \mu_ {b}) G (\beta), \forall b \in B, b \neq 1 \right\} \tag {56}
$$

where the second inequality comes from $1 \geq \mu_{1} - \mu_{k}$ . Now introduce $\lambda'$ and $\mu'$ as

$$
\lambda_ {1} ^ {\prime} = \lambda_ {1} + (\mu_ {1} - A) G (\beta), \lambda_ {b} ^ {\prime} = \lambda_ {b} - (A - \mu_ {b}) G (\beta), \forall b \neq 1, b \in B;
$$

$$
\mu_ {1} ^ {\prime} = \mu_ {1} - (\mu_ {1} - A) G (\beta), \mu_ {b} ^ {\prime} = \mu_ {b} + (A - \mu_ {b}) G (\beta), \forall b \neq 1, b \in B.
$$

These allow us to write the r.h.s. of (56) as the value of the following optimization problem:

$$
\beta \inf \left\{\sum_ {b \in B} (\lambda_ {b} ^ {\prime} - \mu_ {b} ^ {\prime}) ^ {2}: \lambda_ {1} ^ {\prime} \leq \min _ {b \in B} \lambda_ {b} ^ {\prime} \right\}. \tag {57}
$$

Applying Proposition 2 with $S_{b} = (A - \mu_{b})G(\beta)$ for all $b\in B, b\neq 1$ yields that the value of (57) is

$$
\beta \sum_ {b \in B} (\mu_ {b} ^ {\prime} - A) ^ {2} = \beta \left((\mu_ {1} + (\mu_ {1} - A) G (\beta) - A) ^ {2} + \sum_ {b \in B, b \neq 1} (A - \mu_ {b} + (A - \mu_ {b}) G (\beta)) ^ {2}\right). \tag {58}
$$

Recall that $G(\beta) = 1/\sqrt{\beta} - 1$ . Hence, (58) is larger than $\sum_{b \in B} (\mu_b - A)^2 = \xi_j$ .

In Proposition 4, the top- $j$ arms only are considered. We can prove similar results for any $J \in \mathcal{J}$ , by combining Proposition 3 to the arguments of the previous proof.

Corollary 1. $\forall \beta \in (0,1],\mu \in [0,1]^K,2\leq j\leq K$ , and $J\in \mathcal{J}$ $J\neq [j]$ , one has

$$
\beta \inf \left\{\sum_ {k = 1} ^ {K} (\lambda_ {k} - \mu_ {k}) ^ {2}: \boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}, \lambda_ {1} \leq \min _ {k \in J, k \neq 1} \lambda_ {k} - G (\beta) \right\} \geq \bar {\xi} _ {j}. \tag {59}
$$

Proof. Let $J \in \mathcal{J}$ , $J \neq [j]$ be fixed, we denote the indexes in $J$ by $\{\tilde{1}, \tilde{2}, \ldots, \tilde{j}\}$ such that $\tilde{1} < \tilde{2} < \ldots < \tilde{j}$ . One can repeat the argument in the proof of Proposition 4 to obtain that the l.h.s. of (59) is larger than

$$
\inf \left\{\sum_ {k = 1} ^ {K} (\lambda_ {k} - \mu_ {k}) ^ {2}: \boldsymbol {\lambda} \in [ 0, 1 ] ^ {K},   \lambda_ {1} \leq \min _ {k \in J, k \neq 1} \lambda_ {k} \right\}. \tag {60}
$$

Since every $J \in \mathcal{J}$ includes 1, $\tilde{1} = 1$ . Also, since $J \neq [j]$ , we have $\tilde{2} \geq 2, \ldots, \tilde{j} \geq j + 1$ . Because we assume that $\mu_1 > \mu_2 \geq \ldots \geq \mu_K$ , Proposition 3 yields that the value of (60) is larger than $\bar{\xi}_j$ .

# D.2 Optimization problem for CR-A

The following proposition is the analogue of Proposition 3 for $\psi_j$ .

Proposition 5. Let $\mu' \in [0,1]^K$ such that $\mu_1' \geq \mu_1$ and $\mu_k' \leq \mu_k$ for all $k = 2, \ldots, j$ . Define $\psi_j' = \frac{j-1}{j} (\mu_1' - \frac{\sum_{k=2}^{j} \mu_k'}{j-1})^2$ . Then, $\psi_j' \geq \psi_j$ .

Proof. The result simply follows from the following inequality:

$$
\psi_ {j} ^ {\prime} = \frac {j - 1}{j} (\mu_ {1} ^ {\prime} - \frac {\sum_ {k = 2} ^ {j} \mu_ {k} ^ {\prime}}{j - 1}) ^ {2} \geq \frac {j - 1}{j} (\mu_ {1} - \frac {\sum_ {k = 2} ^ {j} \mu_ {k} ^ {\prime}}{j - 1}) ^ {2} \geq \frac {j - 1}{j} (\mu_ {1} - \frac {\sum_ {k = 2} ^ {j} \mu_ {k}}{j - 1}) ^ {2} = \psi_ {j}.
$$

□

We use the following result in the proof of Theorem 4.

Proposition 6. $\forall \beta \in (0,1],\mu \in [0,1]^K$ with $\mu_1 > \mu_2\geq \ldots \geq \mu_K$ , and $2\leq j\leq K$ , one has

$$
\beta \inf \left\{\sum_ {k = 1} ^ {j} (\lambda_ {k} - \mu_ {k}) ^ {2}: \boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}, \lambda_ {1} \leq \frac {\sum_ {k = 2 , \dots , j} \lambda_ {k}}{j - 1} - G (\beta) \right\} \geq \psi_ {j}, \tag {61}
$$

where $\psi_j = \frac{j - 1}{j} (\mu_1 - \frac{\sum_{k=2}^{j}\mu_k}{j - 1})^2$ , $\forall j \in \{2, \dots, K\}$ , as introduced in Section 4.2.

Proof. The Lagrangian of the optimization problem (61) is:

$$
\mathcal {L} (\boldsymbol {\lambda}, \eta) = \frac {\beta}{2} \sum_ {k = 1} ^ {j} \left(\lambda_ {k} - \mu_ {k}\right) ^ {2} + \eta \left(\lambda_ {1} - \frac {\sum_ {k = 2} ^ {j} \lambda_ {k}}{j - 1} + G (\beta)\right), \text {for} (\boldsymbol {\lambda}, \eta) \in \mathbb {R} ^ {j} \times \mathbb {R} _ {\geq 0}.
$$

Denote the saddle point of $\mathcal{L}$ by $(\pmb{\lambda}^{\star},\eta^{\star})$ . The KKT conditions are satisfied:

$$
\lambda_ {1} ^ {\star} \leq \frac {\sum_ {k = 2} ^ {j} \lambda_ {k} ^ {\star}}{j - 1} - G (\beta) \text {   and   } \eta^ {\star} \geq 0, \tag {Feasibility}
$$

$$
\beta (\lambda_ {1} ^ {\star} - \mu_ {1}) + \eta^ {\star} = 0, \text {   and   } \beta (\lambda_ {k} ^ {\star} - \mu_ {k}) - \frac {\eta^ {\star}}{j - 1} = 0, \forall k \neq 1, \tag {Stationarity}
$$

$$
\eta^ {\star} \left(\lambda_ {1} ^ {\star} - \frac {\sum_ {k = 2} ^ {j} \lambda_ {k} ^ {\star}}{j - 1} + G (\beta)\right) = 0. \quad \text {(Complementarity)}
$$

One can simply verify that if $\eta^{\star} = 0$ , stationarity and feasibility cannot hold simultaneously. Thus $\eta^{\star} > 0$ and complementarity yield that $\lambda_1^\star - \sum_{k=2}^{j} \lambda_k^\star / (j - 1) + G(\beta) = 0$ . In conjunction with stationarity, we have

$$
\eta^ {\star} = \frac {\beta (j - 1)}{j} \left(\mu_ {1} - \frac {\sum_ {k = 2} ^ {j} \mu_ {k}}{j - 1} + G (\beta)\right),
$$

and hence the value of (61) is

$$
\frac {(j - 1) \beta}{j} \left(\mu_ {1} - \frac {\sum_ {k = 2} ^ {j} \mu_ {k}}{j - 1} + G (\beta)\right) ^ {2}. \tag {62}
$$

Recall that $G(\beta) = 1 / \sqrt{\beta} - 1$ and $\boldsymbol{\mu} \in [0,1]^K$ . We deduce that $G(\beta) \geq (1 / \sqrt{\beta} - 1)(\mu_1 - \frac{\sum_{k=2}^j \mu_k}{j-1})$ , which is equivalent to $\mu_1 - \frac{\sum_{k=2}^j \mu_k}{j-1} + G(\beta) \geq \frac{1}{\sqrt{\beta}} (\mu_1 - \frac{\sum_{k=2}^j \mu_k}{j-1})$ and hence (62) is larger than $\frac{j-1}{j} (\mu_1 - \frac{\sum_{k=2}^j \mu_k}{j-1})^2$ .

As we obtained Corollary 1 in Appendix D.1, combining Proposition 5 and the proof of Proposition 6 yields the following corollary.

Corollary 2. $\forall \beta \in (0,1],\mu \in [0,1]^K$ with $\mu_1 > \mu_2\geq \ldots \geq \mu_K,2\leq j\leq K,$ and $J\in \mathcal{J},J\neq [j]$ , one has

$$
\beta \inf \left\{\sum_ {k \in [ K ]} (\lambda_ {k} - \mu_ {k}) ^ {2}: \boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}, \lambda_ {1} \leq \frac {\sum_ {k \in J , k \neq 1} \lambda_ {k}}{j - 1} - G (\beta) \right\} \geq \bar {\psi} _ {j}. \tag {63}
$$

Proof. Let $J \in \mathcal{J}$ , $J \neq [j]$ be fixed. We denote the indexes in $J$ by $\{\tilde{1}, \tilde{2}, \ldots, \tilde{j}\}$ such that $\tilde{1} < \tilde{2} < \ldots < \tilde{j}$ . One can repeat the arguments of the proof of Proposition 6 to obtain that the l.h.s. of (63) is larger than

$$
\frac {j - 1}{j} (\mu_ {\tilde {1}} - \frac {\sum_ {k = 2} ^ {j} \mu_ {\tilde {k}}}{j - 1}) ^ {2}. \tag {64}
$$

Note that every $J \in J$ containing 1 satisfies $\tilde{1} = 1$ , and that since $J \neq [j]$ , we have $\tilde{2} \geq 2, \ldots, \tilde{j} \geq j + 1$ . As we assume that $\mu_{1} > \mu_{2} \geq \ldots \geq \mu_{K}$ , Proposition 5 yields that the value of (64) is larger than $\bar{\xi}_{j}$ .

# E LDP for the sampling process under CR

In this section, we are interested in deriving an LDP for the process $\{\omega (\theta T)\}_{T\geq 1}$ for a fixed $\theta \in (0,1]\cap \mathbb{Q}$ under $\mathbb{CR}$ . More precisely, we look for a function $I_{\theta}(\cdot)$ which satisfies an LDP upper bound (3) on $\mathcal{X}_j$ for some fixed $j\in \{1,\ldots ,K\}$ , where

$$
\mathcal {X} _ {j} = \left\{\boldsymbol {x} \in \Sigma : \exists \sigma : [ K ] \mapsto [ K ] \text {   s.t.   } x _ {\sigma (1)} = \dots = x _ {\sigma (j)} > x _ {\sigma (j + 1)} > \dots > x _ {\sigma (K)} > 0 \right\}. \tag {65}
$$

For convenience, we define $x_{\sigma (K + 1)} = 0$ . For any $j$ , we also define

$$
\mathcal {X} _ {j, i} (\theta) = \left\{\boldsymbol {x} \in \mathcal {X} _ {j}: \theta x _ {\sigma (i)} i \overline {{\log}} i > 1 - \theta \sum_ {k = i + 1} ^ {K} x _ {\sigma (k)} \right\}, \forall i \in \{j, \dots , K \}, \tag {66}
$$

where the permutation $\sigma$ depends on $x$ as in the definition of $\mathcal{X}_j$ (65).

It is important to remark that when $\theta T$ is not an integer, $\omega (\theta T)$ is not defined. Hence in the following, when we write $\varprojlim_{T\to \infty}f(\mathbb{P}_{\mu}[\omega (\theta T)\in F])$ , we actually mean $\varprojlim_{T\to \infty :\theta T\in \mathbb{N}}f(\mathbb{P}_{\mu}[\omega (\theta T)\in F])$ .

Deriving an LDP upper bound (3) is not easy in general, and to this aim, we first introduce a useful sufficient condition in E.1.

# E.1 A sufficient condition towards an LDP upper bound (3)

The following condition will be useful in our analysis, in particular in this section. This condition is similar to those presented in Chapter 2 in Varadhan (2016). We say that $\{Y(t)\}_{t\geq 1}$ satisfies an LDP local upper bound with rate function $I$ at point $\mathbf{y}\in \mathcal{Y}$ if:

$$
\varliminf_ {\delta \rightarrow 0} \varliminf_ {t \rightarrow \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} [ Y (t) \in B (\boldsymbol {y} , \delta) ]} \geq I (\boldsymbol {y}), \tag {67}
$$

where $B(\pmb {y},\delta)$ is the open ball with center $\pmb{y}$ and radius $\delta$ .

Lemma 10. Suppose Y is compact and $\{Y(t)\}_{t\geq1}$ satisfies an LDP local upper bound (67) with a lower semi-continuous function I at all $y\in Y$ , then $\{Y(t)\}_{t\geq1}$ satisfies an LDP upper bound (3).

Proof. Let $C \subseteq \mathcal{Y}$ be a closed (and hence compact) set, and $s = \inf_{\boldsymbol{y} \in C} I(\boldsymbol{y})$ . We prove $\varprojlim_{t \to \infty} \frac{1}{t} \log \frac{1}{\mathbb{P}[Y(t) \in C]} \geq s$ if (i) $s = \infty$ and if (ii) $s < \infty$ separately.

(i) If $s = \infty$ . Let $M > 0$ and $\pmb{y} \in C$ . As $I(\pmb{y}) = \infty$ , and since $I$ is lower semi-continuous, there exists $\delta_{\pmb{y}} > 0$ s.t.

$$
\varliminf_ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} \left[ Y (t) \in B (\boldsymbol {y} , \delta_ {\boldsymbol {y}}) \right]} \geq M. \tag {68}
$$

Now observe that $C \subseteq \cup_{\boldsymbol{y} \in C} B(\boldsymbol{y}, \delta_{\boldsymbol{y}})$ . The compactness of $C$ implies that we can find $N \in \mathbb{N}$ , and $\{\boldsymbol{y}_1, \ldots, \boldsymbol{y}_N\}$ such that $C \subseteq \cup_{i=1}^{N} B(\boldsymbol{y}_i, \delta_{\boldsymbol{y}_i})$ , which directly yields that

$$
\mathbb {P} \left[ Y (t) \in C \right] \leq \sum_ {i = 1} ^ {N} \mathbb {P} \left[ Y (t) \in B (\boldsymbol {y} _ {i}, \delta_ {\boldsymbol {y} _ {i}}) \right] \leq N \max _ {i \in [ N ]} \mathbb {P} \left[ Y (t) \in B (\boldsymbol {y} _ {i}, \delta_ {\boldsymbol {y} _ {i}}) \right]. \tag {69}
$$

Using a simple rearrangement in (68) and (69), we then have $\varinjlim_{t\to\infty}\frac{1}{t}\log\frac{1}{\mathbb{P}[Y(t)\in C]}\geq M$ . As M can be taken arbitrarily large, the proof is completed.

(ii) If $s < \infty$ . Let $\epsilon \in (0, s/2)$ and $\pmb{y} \in C$ . As $I(\pmb{y}) \geq s$ and since $I$ is lower semi-continuous, there exists $\delta_{\pmb{y}} > 0$ such that

$$
\varliminf_ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} \left[ Y (t) \in B (\boldsymbol {y} , \delta_ {\boldsymbol {y}}) \right]} \geq s - \epsilon . \tag {70}
$$

Now observe that $C \subseteq \cup_{\boldsymbol{y} \in C} B(\boldsymbol{y}, \delta_{\boldsymbol{y}})$ . The compactness of $C$ implies that we can find $N \in \mathbb{N}$ , $\{\boldsymbol{y}_1, \ldots, \boldsymbol{y}_N\}$ such that $C \subseteq \cup_{i=1}^{N} B(\boldsymbol{y}_i, \delta_{\boldsymbol{y}_i})$ , which directly yields that

$$
\mathbb {P} \left[ Y (t) \in C \right] \leq \sum_ {i = 1} ^ {N} \mathbb {P} \left[ Y (t) \in B (\boldsymbol {y} _ {i}, \delta_ {\boldsymbol {y} _ {i}}) \right] \leq N \max _ {i \in [ N ]} \mathbb {P} \left[ Y (t) \in B (\boldsymbol {y} _ {i}, \delta_ {\boldsymbol {y} _ {i}}) \right]. \tag {71}
$$

Using a simple rearrangement in (70) and (71), we then have $\varinjlim_{t\to \infty}\frac{1}{t}\log \frac{1}{\mathbb{P}[Y(t)\in C]}\geq s - \epsilon$ . As $\epsilon$ can be taken arbitrarily small, the proof is completed.

We apply Lemma 10 to the process $\{\omega (\theta T)\}_{T\geq 1}$ . The latter has values in $\Sigma$ , a compact set. To derive an LDP upper bound for this process (such an LDP upper bound is required to apply Theorem 1), we just need to establish at all points in $\Sigma$ a local LDP upper bound.

The following two theorems state that $\{\omega (\theta T)\}_{T\geq 1}$ under CR-C and CR-A satisfies a local LDP upper bound.

Theorem 5. [Local LDP upper bound for $\mathbb{CR} - \mathbb{C}$ ] For $\theta \in (0,1] \cap \mathbb{Q}$ , we define $I_{\theta}$ as follows.

(a) If $\boldsymbol{x} \in \mathcal{X}_1 \cup (\cup_{j=2}^K \cup_{i=j}^K \mathcal{X}_{j,i}(\theta))$ , then $I_\theta(\boldsymbol{x}) = \infty$ ;

(b) If $\exists j\in \{2,\dots ,K\}$ such that $\pmb {x}\in \mathcal{X}_j\setminus \cup_{j = 2}^{K}\cup_{i = j}^{K}\mathcal{X}_{j,i}(\theta)$ , then

$$
I _ {\theta} (\boldsymbol {x}) = \max _ {p = j, \dots , K - 1} 2 x _ {\sigma (p + 1)} \inf _ {\boldsymbol {\lambda} \in \mathcal {S} _ {p} (\boldsymbol {x})} \sum_ {k = 1} ^ {p + 1} (\lambda_ {\sigma (k)} - \mu_ {\sigma (k)}) ^ {2},
$$

where $S_{p}(\pmb{x})$ is defined in (90);

(c) If $\mathcal{V} = \cup_{k=1}^{K}\mathcal{X}_{k}$ , and $x \in \operatorname{cl}(\mathcal{V}) \setminus \mathcal{V}$ , then

$$
I _ {\theta} (\boldsymbol {x}) = \inf \{\varliminf_ {s \to \infty} I _ {\theta} (\boldsymbol {x} ^ {(s)}): \{\boldsymbol {x} ^ {(s)} \} _ {s \in \mathbb {N}} \subset \mathcal {V}, \boldsymbol {x} ^ {(s)} \to \boldsymbol {x} a s s \to \infty \};
$$

Then the process $\{\omega (\theta T)\}_{T\geq 1}$ under $\mathbb{CR} - \mathbb{C}$ satisfies an LDP upper bound (3) with the rate function $I_{\theta}$ , and $I_{\theta}$ is lower semi-continuous.

Proof. In view of Lemma 10, the theorem holds if we are able to show that $\{\omega (\theta T)\}_{T\geq 1}$ satisfies a local LDP upper bound with $I_{\theta}$ and if $I_{\theta}$ is lower semi-continuous. The first part is established below in Lemma 12, Lemma 13, Lemma 14, and Theorem 7.

For the second part, we first verify the lower semi-continuity of $I_{\theta}$ restricted to $\cup_{j=1}^{K}\mathcal{X}_j$ , and then apply Lemma 11 with $f = I_{\theta}$ to establish the lower semi-continuity of $I_{\theta}$ in $\Sigma$ . Let $\boldsymbol{x} \in \cup_{j=1}^{K}\mathcal{X}_j$ . If $\boldsymbol{x} \in \mathcal{X}_1$ , lower semi-continuity directly follows from the fact $\mathcal{X}_1$ is open and $I_{\theta}(\boldsymbol{x}) = \infty$ for $\boldsymbol{x} \in \mathcal{X}_1$ . We then consider $\boldsymbol{x} \in \mathcal{X}_j$ for some $j = 2, \ldots, K$ . By definition, there is $\sigma : [K] \mapsto [K]$ such that $x_{\sigma(1)} = \ldots = x_{\sigma(j)} > x_{\sigma(j+1)} > \ldots > x_{\sigma(K)}$ . By taking $\delta < \min_{i \geq j} \{x_{\sigma(i)} - x_{\sigma(i+1)}\}/2$ , we have $\boldsymbol{x}' \in \cup_{q=1}^{j}\mathcal{X}_q$ if $\| \boldsymbol{x}' - \boldsymbol{x} \|_\infty < \delta$ , and $I_{\theta}(\boldsymbol{x}') \geq \max_{p=j,\ldots,K-1} 2x'_{\sigma(p+1)} \inf_{\boldsymbol{\lambda} \in \mathcal{S}_p(\boldsymbol{x}')} \sum_{k=1}^{p+1} (\lambda_{\sigma(k)} - \mu_{\sigma(k)})^2$ as a consequence. Now as verified in Lemma 17 in Appendix F, the mapping $\boldsymbol{x} \mapsto 2x_{\sigma(p+1)} \inf_{\boldsymbol{\lambda} \in \mathcal{S}_p(\boldsymbol{x})} \sum_{k=1}^{p+1} (\lambda_{\sigma(k)} - \mu_{\sigma(k)})^2$ is continuous, we hence deduce that $I_{\theta}$ is lower semi-continuity at $\boldsymbol{x}$ .

Theorem 6. [Local LDP upper bound for $\mathbb{CR} - \mathbb{A}$ ] For $\theta \in (0,1] \cap \mathbb{Q}$ , we define $I_{\theta}$ as follows.

(a) If $\boldsymbol{x} \in \mathcal{X}_1 \cup (\cup_{j=2}^K \cup_{i=j}^K \mathcal{X}_{j,i}(\theta))$ , then $I_\theta(\boldsymbol{x}) = \infty$ ;

(b) If $\exists j\in \{2,\dots ,K\}$ such that $\pmb {x}\in \mathcal{X}_j\setminus \cup_{j = 2}^{K}\cup_{i = j}^{K}\mathcal{X}_{j,i}(\theta)$ , then

$$
I _ {\theta} (\boldsymbol {x}) = \max _ {p = j, \dots , K - 1} 2 x _ {\sigma (p + 1)} \inf _ {\boldsymbol {\lambda} \in \mathcal {S} _ {p} (\boldsymbol {x})} \sum_ {k = 1} ^ {p + 1} (\lambda_ {\sigma (k)} - \mu_ {\sigma (k)}) ^ {2},
$$

where $\mathcal{S}_p(\boldsymbol{x})$ is defined in (97);

(c) If $\mathcal{V} = \cup_{k=1}^{K}\mathcal{X}_{k}$ , and $\boldsymbol{x} \in \operatorname{cl}(\mathcal{V}) \setminus \mathcal{V}$ , then

$$
I _ {\theta} (\boldsymbol {x}) = \inf \{\varliminf_ {s \to \infty} I _ {\theta} (\boldsymbol {x} ^ {(s)}): \{\boldsymbol {x} ^ {(s)} \} _ {s \in \mathbb {N}} \subset \mathcal {V}, \boldsymbol {x} ^ {(s)} \to \boldsymbol {x} a s s \to \infty \};
$$

Then the process $\{\omega (\theta T)\}_{T\geq 1}$ under CR-A satisfies an LDP upper bound (3) with the rate function $I_{\theta}$ , and $I_{\theta}$ is lower semi-continuous.

Proof. In view of Lemma 10, the theorem is deduced if we are able to show that $\{\omega (\theta T)\}_{T\geq 1}$ satisfies a local LDP upper bound with $I_{\theta}$ . This is established below in Lemma 12, Lemma 13, Lemma 14, and Theorem 8.

We then verify the lower semi-continuity of $I_{\theta}$ restricted to $\cup_{j=1}^{K} \mathcal{X}_j$ , and then applying Lemma 11 with $f = I_{\theta}$ yields the lower semi-continuity of $I_{\theta}$ in $\Sigma$ . Let $\boldsymbol{x} \in \cup_{j=1}^{K} \mathcal{X}_j$ . If $\boldsymbol{x} \in \mathcal{X}_1$ , lower semi-continuity directly follows from the fact $\mathcal{X}_1$ is open and $I_{\theta}(\boldsymbol{x}) = \infty$ for $\boldsymbol{x} \in \mathcal{X}_1$ . We then consider $\boldsymbol{x} \in \mathcal{X}_j$ for some $j = 2, \ldots, K$ . By definition, there is $\sigma : [K] \mapsto [K]$ such that $x_{\sigma(1)} = \ldots = x_{\sigma(j)} > x_{\sigma(j+1)} > \ldots > x_{\sigma(K)}$ . By taking $\delta < \min_{i \geq j} \{x_{\sigma(i)} - x_{\sigma(i+1)}\}/2$ , we have $\boldsymbol{x}' \in \cup_{q=1}^{j} \mathcal{X}_q$ if $\| \boldsymbol{x}' - \boldsymbol{x} \|_\infty < \delta$ and $I_{\theta}(\boldsymbol{x}') \geq \max_{p=j,\ldots,K-1} 2x'_{\sigma(p+1)} \inf_{\boldsymbol{\lambda} \in S_p(\boldsymbol{x}')} \sum_{k=1}^{p+1} (\lambda_{\sigma(k)} - \mu_{\sigma(k)})^2$ , as a consequence. Now as verified in Lemma 18 in Appendix F, the mapping $\boldsymbol{x} \mapsto 2x_{\sigma(p+1)} \inf_{\boldsymbol{\lambda} \in S_p(\boldsymbol{x})} \sum_{k=1}^{p+1} (\lambda_{\sigma(k)} - \mu_{\sigma(k)})^2$ is continuous, we hence deduce that $I_{\theta}$ is lower semi-continuity at $\boldsymbol{x}$ .

![](images/782688052104da0810f18ea9f6359c0351f2b6212a1d5a750767217e46121a61.jpg)

Lemma 11. Suppose $\mathcal{V} \subseteq \Sigma$ is the set such that $\operatorname{cl}(\mathcal{V}) = \Sigma$ , and $f: \mathcal{V} \to \mathbb{R}$ is a lower semi-continuous function. If we extend $f$ to $\Sigma$ by defining

$$
\bar {f} (\boldsymbol {\omega}) = \left\{ \begin{array}{l l} f (\boldsymbol {\omega}), & \text {if} \boldsymbol {\omega} \in \mathcal {V}, \\ \inf \{\varliminf_ {s \to \infty} f (\boldsymbol {\omega} ^ {(s)}): \{\boldsymbol {\omega} ^ {(s)} \} _ {s \in \mathbb {N}} \subset \mathcal {V}, \boldsymbol {\omega} ^ {(s)} \to \boldsymbol {\omega} \text {as} s \to \infty \}, & \text {otherwise}, \end{array} \right.
$$

then $\bar{f}$ is a lower semi-continuous function in $\Sigma$ .

Proof. By the definition of $\bar{f}$ and the fact $\operatorname{cl}(\mathcal{V}) = \Sigma$ ,

$$
\forall \varepsilon > 0, \forall \delta > 0, \forall \boldsymbol {\omega} \in \Sigma , \exists \boldsymbol {x} \in \mathcal {V} \text {   such   that   } f (\boldsymbol {x}) <   \bar {f} (\boldsymbol {\omega}) + \epsilon \text {   and   } \| \boldsymbol {x} - \boldsymbol {\omega} \| _ {\infty} <   \delta . \tag {72}
$$

Next suppose on the contrary, $\bar{f}$ is not lower semi-continuous at some $\omega \in \Sigma$ , that is, $\exists \{\pmb{\omega}^{(s)}\} \subset \Sigma$ such that $\pmb{\omega}^{(s)} \to \pmb{\omega}$ as $s \to \infty$ and $\lim_{s \to \infty} \bar{f}(\pmb{\omega}^{(s)}) < \bar{f}(\pmb{\omega})$ . Let $\eta = \bar{f}(\pmb{\omega}) - \lim_{s \to \infty} \bar{f}(\pmb{\omega}^{(s)}) > 0$ . For each $s \in \mathbb{N}$ , (72) implies that there is $\pmb{x}^{(s)} \in \mathcal{V}$ such that $\| \pmb{x}^{(s)} - \pmb{\omega}^{(s)} \|_{\infty} < 1/s$ and $f(\pmb{x}^{(s)}) < \bar{f}(\pmb{\omega}^{(s)}) + \eta/2$ . Hence,

$$
\varliminf_ {s \to \infty} f (\boldsymbol {x} ^ {(s)}) \leq \varliminf_ {s \to \infty} \bar {f} (\boldsymbol {\omega} ^ {(s)}) + \frac {\eta}{2} <   \bar {f} (\boldsymbol {\omega}),
$$

which contradicts the lower semi-continuity of f if $\omega \in V$ and the definition of $\bar{f}$ if $\omega \notin V$ .

![](images/51322aaa171e65134478dae9ec4e77649b54ef86d24820b7e372cab3d41b20e2.jpg)

# E.2 Local LDP upper bound on $\cup_{i=j}^{K}\mathcal{X}_{j,i}(\theta)$

Let $\theta \in (0,1] \cap \mathbb{Q}$ and $j \in \{2, \dots, K\}$ . Here, we first prove the result on $\mathcal{X}_{j,j}(\theta)$ in Lemma 13 and that on $\mathcal{X}_{j,i}(\theta)$ for any $i > j$ in Lemma 14. Note the results in this subsection are valid for both CR-C and CR-A.

Lemma 12. Let $\theta \in (0,1] \cap \mathbb{Q}$ , $\boldsymbol{x} \in \mathcal{X}_1$ , the process $\{\boldsymbol{\omega}(\theta T)\}_{T \geq 1}$ satisfies an LDP local upper bound (67) with $I_\theta(\boldsymbol{x}) = \infty$ at $\boldsymbol{x} \in \mathcal{X}_1$ .

Proof. For $x \in X_{1}$ , there exists $\sigma : [K] \mapsto [K]$ such that $x_{\sigma(1)} > x_{\sigma(2)} > \ldots > x_{\sigma(K)}$ . Let $\delta < \min_{k=1,\ldots,K-1}\{x_{\sigma(k)} - x_{\sigma(k+1)}\}$ and $T > \frac{2}{\theta\delta}$ such that $\theta T \in N$ , we show that $\mathbb{P}_{\mu}[\omega(\theta T) \in B(x, \delta)] = 0$ , which directly completes the proof.

If $\omega (\theta T)\in B(\pmb {x},\delta)$ , then we have $\omega_{\sigma (1)}(\theta T) > \omega_{\sigma (2)}(\theta T) > \dots >\omega_{\sigma (K)}(\theta T)$ and

$$
\min _ {k = 1, \dots , K - 1} \left\{N _ {\sigma (k)} (\theta T) - N _ {\sigma (k + 1)} (\theta T) \right\} = \theta T \min _ {k = 1, \dots , K - 1} \left\{\omega_ {\sigma (k)} (\theta T) - \omega_ {\sigma (k + 1)} (\theta T) \right\} > 2. \tag {73}
$$

Because CR always pulls arms in the candidate set in a round-robin manner, (73) will never happen. Hence, $\mathbb{P}_{\boldsymbol{\mu}}[\boldsymbol{\omega}(\theta T) \in B(\boldsymbol{x}, \delta)] = 0$ . ☐

Lemma 13. Let $j \in \{2, \dots, K\}$ . The process $\{\omega(\theta T)\}_{T \geq 1}$ satisfies an LDP local upper bound (67) with $I_{\theta}(\pmb{x}) = \infty$ at $\pmb{x} \in \mathcal{X}_{j,j}(\theta)$ .

Proof. Let $\pmb{x} \in \mathcal{X}_{j,j}(\theta)$ . We show that there exists $\delta_{\theta, \pmb{x}} > 0$ and $T_{\theta, \pmb{x}} \in \mathbb{N}$ s.t. if $T \geq T_{\theta, \pmb{x}}$ and $\delta < \delta_{\theta, \pmb{x}}$ , then $\omega(\theta T) \notin B(\pmb{x}, \delta)$ almost surely. As a consequence, $\mathbb{P}_{\mu}[\omega(\theta T) \in B(\pmb{x}, \delta)] = 0$ , and $I_{\theta}(\pmb{x}) = \infty$ . We decompose the proof into three steps.

1. Defining $\delta_{\theta, x}$ and $T_{\theta, x}$ . We introduce the two functions, $f_1, f_2 : [0, 1] \times \Sigma \mapsto \mathbb{R}$ :

$$
f _ {1} \left(\theta^ {\prime}, \boldsymbol {x} ^ {\prime}\right) = \theta^ {\prime} \sum_ {k = 1} ^ {j} x _ {\sigma (k)} ^ {\prime} \overline {{\log}} j - \theta^ {\prime} \sum_ {k = j + 1} ^ {K} x _ {\sigma (k)} ^ {\prime} - 1,
$$

$$
f _ {2} (\theta^ {\prime}, \pmb {x} ^ {\prime}) = \min _ {k \leq j} x _ {\sigma (k)} ^ {\prime} - \max _ {k \geq j + 1} x _ {\sigma (k)} ^ {\prime}.
$$

Since $\pmb{x} \in \mathcal{X}_{j,j}(\theta)$ , we have $c = \min\{f_1(\theta, \pmb{x}), f_2(\theta, \pmb{x})\} > 0$ . Leveraging the fact that $f_1, f_2$ are continuous, we can find $\delta_{\theta, \pmb{x}} \in (0, \frac{1}{3j})$ s.t.

if $|\theta' - \theta| < 3j\delta_{\theta, \boldsymbol{x}}$ and $\| \boldsymbol{x}' - \boldsymbol{x}\|_{\infty} < 7j\delta_{\theta, \boldsymbol{x}}$ , then $\min \{f_1(\theta', \boldsymbol{x}')$ , $f_2(\theta', \boldsymbol{x}')\} > \frac{c}{2}$ . (74)

We also define $T_{\theta, \pmb{x}} = \max \{ \lceil \frac{4}{\theta_c} \rceil, \lceil \frac{1}{\delta_{\theta, \pmb{x}}} \rceil \}$ .

2. We prove that for $\delta < \delta_{\theta, x}$ and $T > T_{\theta, x}$ , $\omega(\theta T) \notin B(x, \delta)$ a.s.. We proceed by contradiction. Assume $\omega(\theta T) \in B(x, \delta)$ . From (74), we have $f_2(\theta, \omega(\theta T)) > c/2$ . It then directly yields that

$$
\begin{array}{l} \min _ {k \leq j} N _ {\sigma (k)} (\theta T) - \max _ {k \geq j + 1} N _ {\sigma (k)} (\theta T) = \theta T \left[ \min _ {k \leq j} \omega_ {\sigma (k)} (\theta T) - \max _ {k \geq j + 1} \omega_ {\sigma (k)} (\theta T) \right] \\ = \theta T f _ {2} (\theta , \omega (\theta T)) \\ > \frac {\theta c T}{2} > \frac {\theta c T _ {\theta , x}}{2} \geq 2, \tag {75} \\ \end{array}
$$

where the last inequality follows from $T_{\theta, x} > \frac{4}{\theta c}$ . Observe that CR always pulls the arms in the candidate set in a round-robin manner (the maximal difference of pulling amounts among the candidate set is at most 1), and CR stops pulling an arm $k$ after $k$ is removed from the candidate set. Thus, from (75), we deduce several facts: (i) $C_j = \{\sigma(1), \ldots, \sigma(j)\}$ ; (ii) before the round $\tau = j \min_{k \leq j} N_{\sigma(k)}(\theta T) + \sum_{k > j} N_{\sigma(k)}(\theta T)$ , the arm $\ell_j$ to be removed has not yet been decided; (iii)

$$
N _ {\sigma (k)} (\tau - j) = \left\{ \begin{array}{l l} \min _ {k \leq j} N _ {\sigma (k)} (\theta T) - 1, & \text { if } k \leq j, \\ N _ {\sigma (k)} (\tau - j) = N _ {\sigma (k)} (\theta T), & \text { if } k > j. \end{array} \right.
$$

However, we will show in the next step that in the round $\tau - j$ , the condition for discarding an arm in $C_{j}$ is triggered. In other words, $\ell_{j} = \ell(\tau - j)$ is removed from $C_{j}$ in the round $\tau - j$ , which contradicts (ii).

3. The condition for discarding an arm in round $\tau - j$ is triggered. First, from (iii) in Step 2,

$$
N _ {\sigma (1)} (\tau - j) = N _ {\sigma (2)} (\tau - j) = \dots = N _ {\sigma (j)} (\tau - j). \tag {76}
$$

Then, using (iii) in Step 2 again yields that

$$
\min _ {k \leq j} N _ {\sigma (k)} (\tau - j) - \max _ {k \geq j + 1} N _ {\sigma (k)} (\tau - j) = \min _ {k \leq j} N _ {\sigma (k)} (\theta T) - 1 - \max _ {k \geq j + 1} N _ {\sigma (k)} (\theta T) > 1, \tag {77}
$$

where the last inequality comes from (75). Finally, observe that

$$
\begin{array}{l} \theta - \frac {\tau - j}{T} = \frac {\sum_ {k \in [ j ]} N _ {\sigma (k)} (\theta T) - j \min _ {k \in [ j ]} N _ {\sigma (k)} (\theta T)}{T} + \frac {j}{T} \\ \leq j \left[ \max _ {k \in [ j ]} \omega_ {\sigma (k)} (\theta T) - x _ {\sigma (j)} + x _ {\sigma (j)} - \min _ {k \in [ j ]} \omega_ {\sigma (k)} (\theta T) \right] + j \delta_ {\theta , \boldsymbol {x}} \\ \leq j (2 \delta + \delta_ {\theta , \boldsymbol {x}}) <   3 j \delta_ {\theta , \boldsymbol {x}}, \tag {78} \\ \end{array}
$$

where the first inequality is due to $T \geq 1 / \delta_{\theta, \pmb{x}}$ ; the second inequality follows from $\omega(\theta T) \in B(\pmb{x}, \delta)$ ; the last inequality is obtained using $\delta < \delta_{\theta, \pmb{x}}$ . Combining Lemma 15 (see below) with $\delta = 3j\delta_{\theta, \pmb{x}}$ and (78) yields that $\| \pmb{\omega}(\tau - j) - \pmb{\omega}(\theta T) \|_{\infty} \leq 6j\delta_{\theta, \pmb{x}}$ , hence

$$
\left\| \boldsymbol {\omega} (\tau - j) - \boldsymbol {x} \right\| _ {\infty} \leq \left\| \boldsymbol {\omega} (\tau - j) - \boldsymbol {\omega} (\theta T) \right\| _ {\infty} + \left\| \boldsymbol {\omega} (\theta T) - \boldsymbol {x} \right\| _ {\infty} \leq 7 j \delta_ {\theta , \boldsymbol {x}}. \tag {79}
$$

From (74)-(78)-(79), we get $f_{1}\big(\frac{\tau - j}{T},\omega (\tau -j)\big) > \frac{c}{2}$ . Thus,

$$
G \left(\frac {\sum_ {k = 1} ^ {j} N _ {\sigma (k)} (\tau - j) \overline {{\log}} j}{T - \sum_ {k = j + 1} ^ {K} N _ {\sigma (k)} (\tau - j)}\right) = G \left(\frac {\frac {\tau - j}{T} \sum_ {k = 1} ^ {j} \omega_ {\sigma (k)} (\tau - j) \overline {{\log}} j}{1 - \frac {\tau - j}{T} \sum_ {k = j + 1} ^ {K} \omega_ {\sigma (k)} (\tau - j)}\right) <   G (1) = 0, \tag {80}
$$

where the inequality is directly derived from $f_{1}(\frac{\tau-j}{T},\omega(\tau-j)) > 0$ and the fact that $G(\beta) = 1/\sqrt{\beta}-1$ is a strictly decreasing function. Note that (76)-(77)-(80) trigger the condition of discarding $\ell(\tau-j)$ in the round $\tau-j$ .

Lemma 14. Let $j \in \{2, \ldots, K - 1\}$ and $i > j$ . The process $\{\omega(\theta T)\}_{T \geq 1}$ satisfies an LDP local upper bound (67) with $I_{\theta}(\pmb{x}) = \infty$ at $\pmb{x} \in \mathcal{X}_{j,i}(\theta)$ .

Proof. Let $\pmb{x} \in \mathcal{X}_{j,i}(\theta)$ and let $\sigma$ be the permutation described in (65) for $\pmb{x}$ . We show that there exists $\delta_{\theta, \pmb{x}} > 0$ s.t. for all $\delta < \delta_{\theta, \pmb{x}}$ ,

$$
\lim _ {T \rightarrow \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \boldsymbol {\omega} (\theta T) \in B (\boldsymbol {x} , \delta) ]} = \infty . \tag {81}
$$

We decompose the proof into three steps.

1. Defining $\delta_{\theta,x}$ , $T_{x}$ , and a random time $\tau$ . We introduce two functions: for all $x' \in \Sigma$ :

$$
f _ {1} (\boldsymbol {x} ^ {\prime}) = \theta \min _ {k = 1, \dots , i} x _ {\sigma (k)} ^ {\prime} i \overline {{\log}} i - \theta \sum_ {k = i + 1} ^ {K} x _ {\sigma (k)} ^ {\prime} - 1,
$$

$$
f _ {2} (\boldsymbol {x} ^ {\prime}) = \min _ {k = 1, \dots , i} x _ {\sigma (k)} ^ {\prime} - \max _ {k = i + 1, \dots , K} x _ {\sigma (k)} ^ {\prime}.
$$

Because $f_{1}(\pmb{x}) > 0$ , $f_{2}(\pmb{x}) > (x_{\sigma(i)} - x_{\sigma(i + 1)}) / 2$ , and both $f_{1}, f_{2}$ are continuous, we can find a positive $\delta_{\theta, \pmb{x}} > 0$ s.t.

$$
\forall \boldsymbol {x} ^ {\prime} \in B (\boldsymbol {x}, \delta_ {\theta , \boldsymbol {x}}), f _ {1} (\boldsymbol {x} ^ {\prime}) > 0, f _ {2} (\boldsymbol {x} ^ {\prime}) > \frac {x _ {\sigma (i)} - x _ {\sigma (i + 1)}}{2}. \tag {82}
$$

In the following, we fix $\delta < \delta_{\theta, \boldsymbol{x}}$ . We define $T_{\boldsymbol{x}}$ as: $T_{\boldsymbol{x}} = \lceil \frac{2}{x_{\sigma(i)} - x_{\sigma(i+1)}} \rceil$ . Finally, we introduce the random time $\tau = i \min_{k \leq i} N_{\sigma(k)}(\theta T) + \sum_{k > i} N_{\sigma(k)}(\theta T)$ and two fixed rounds, $\tau_{\min} = \lfloor (ix_{\sigma(i)} + \sum_{k > i} x_{\sigma(k)} - K\delta)T \rfloor$ and $\tau_{\max} = \lceil (ix_{\sigma(i)} + \sum_{k > i} x_{\sigma(k)} + K\delta)T \rceil$ .

2. If $T > T_{\theta, \boldsymbol{x}}$ and $\omega(\theta T) \in B(\boldsymbol{x}, \delta)$ , then (i) $\tau \in \{\tau_{\min}, \ldots, \tau_{\max}\}$ and (ii) $\omega(\tau) \in \mathcal{X}_{i,i}(\frac{\tau}{T})$ . (i) is trivial based on the definition of $B(\boldsymbol{x}, \delta)$ . To show (ii), we observe that

$$
\min _ {k \leq i} N _ {\sigma (k)} (\theta T) - \max _ {k \geq i + 1} N _ {\sigma (k)} (\theta T) = T f _ {2} (\boldsymbol {\omega} (\theta T)) > \frac {2}{x _ {\sigma (i)} - x _ {\sigma (i + 1)}} \frac {x _ {\sigma (i)} - x _ {\sigma (i + 1)}}{2} = 1, \tag {83}
$$

where the inequality simply comes from (82) and $T > T_{x}$ . Since CR always pulls the arms in the candidate set in a round-robin manner (the maximal difference of pulling amounts among the candidate set is at most 1), (83) implies CR discards one arm in $\{\sigma(i+1),\ldots,\sigma(K)\}$ in the round $\tau$ , and $\omega(\tau)$ is:

$$
\omega_ {\sigma (k)} (\tau) = \left\{ \begin{array}{l l} \min _ {k ^ {\prime} \leq i} N _ {\sigma (k ^ {\prime})} (\theta T) / \tau , & \text { if   } k \leq i, \\ N _ {\sigma (k) (\theta T)} / \tau , & \text { otherwise. } \end{array} \right. \tag {84}
$$

Note that (84) yields that $\omega(\tau) \in \mathcal{X}_i$ . Moreover,

$$
\frac {\tau}{T} \omega_ {\sigma (i)} (\tau) i \overline {{\log}} i - \frac {\tau}{T} \sum_ {k = i + 1} ^ {K} \omega_ {\sigma (k)} (\tau) = \theta \min _ {k \leq i} \omega_ {\sigma (k)} (\theta T) i \overline {{\log}} i - \theta \sum_ {k = i + 1} ^ {K} \omega_ {\sigma (k)} (\theta T)
$$

$$
= f _ {1} (\pmb {\omega} (\theta T)) + 1 > 1,
$$

where the inequality directly follows from (82) and the fact that $\omega(\theta T) \in B(\boldsymbol{x}, \delta_{\theta, \boldsymbol{x}})$ . Hence $\omega(\tau) \in \mathcal{X}_{i,i}(\frac{\tau}{T})$ .

3. We show (81). Thanks to (i) and (ii) in Step 2, we have, for $T > T_{\mathbf{x}}$ ,

$$
\mathbb {P} _ {\boldsymbol {\mu}} \left[ \boldsymbol {\omega} (\theta T) \in \mathcal {B} (\boldsymbol {x}, \delta) \right] \leq \sum_ {\tau = \tau_ {\min}} ^ {\tau_ {\max}} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \boldsymbol {\omega} (\tau) \in \mathcal {X} _ {i, i} (\frac {\tau}{T}) \right] \leq 2 K \delta T \max _ {\theta^ {\prime} \in (0, 1 ]} \mathbb {P} _ {\boldsymbol {\mu}} \left[ \boldsymbol {\omega} (\theta^ {\prime} T) \in \mathcal {X} _ {i, i} (\theta^ {\prime}) \right].
$$

Thus, a simple rearrangement of the above inequality yields that

$$
\begin{array}{l} \varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \boldsymbol {\omega} (\theta T) \in B (\boldsymbol {x} , \delta) ]} \geq \inf _ {\theta^ {\prime} \in (0, 1 ]} \varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \boldsymbol {\omega} (\theta^ {\prime} T) \in \mathcal {X} _ {i , i} (\theta^ {\prime}) ]} \\ \geq \inf _ {\theta^ {\prime} \in (0, 1 ]} \inf _ {\boldsymbol {x} ^ {\prime} \in \mathcal {X} _ {i, i} (\theta^ {\prime})} I _ {\theta^ {\prime}} (\boldsymbol {x} ^ {\prime}) = \infty , \\ \end{array}
$$

where the last inequality stems from Lemma 13.

![](images/e9b7af78c6c7801ede82ed0d1643449f4bdbae4f11bc502f3f4d76b320027037.jpg)

Lemma 15. Let $\theta \in (0,1] \cap \mathbb{Q}$ , and $\delta \in (0,1)$ . If we consider $\theta' \in [\theta - \theta\delta, \theta]$ and $T \in \mathbb{N}$ such that $\theta T, \theta'T \in \mathbb{N}$ , then

$$
\left\| \boldsymbol {\omega} (\theta T) - \boldsymbol {\omega} \left(\theta^ {\prime} T\right) \right\| _ {\infty} \leq 2 \delta . \tag {85}
$$

Proof. Observe that for any $k \in [K]$ ,

$$
\begin{array}{l} \left| \omega_ {k} (\theta T) - \omega_ {k} \left(\theta^ {\prime} T\right) \right| = \left| \frac {N _ {k} (\theta T)}{\theta T} - \frac {N _ {k} \left(\theta^ {\prime} T\right)}{\theta^ {\prime} T} \right| \leq \left| \frac {N _ {k} (\theta T)}{\theta T} - \frac {N _ {k} \left(\theta^ {\prime} T\right)}{\theta T} \right| + \left| \frac {N _ {k} \left(\theta^ {\prime} T\right)}{\theta T} - \frac {N _ {k} \left(\theta^ {\prime} T\right)}{\theta^ {\prime} T} \right| \\ \leq \frac {\theta - \theta^ {\prime}}{\theta} + \theta^ {\prime} (\frac {1}{\theta^ {\prime}} - \frac {1}{\theta}) \\ = 1 - \frac {\theta^ {\prime}}{\theta} + 1 - \frac {\theta^ {\prime}}{\theta} \leq 2 \delta , \\ \end{array}
$$

where the first inequality uses the triangle inequality; the second inequality simply comes from $N_{k}(\theta T) - N_{k}(\theta^{\prime}T) \leq (\theta -\theta^{\prime})T$ and $N_{k}(\theta^{\prime}T)\leq \theta^{\prime}T$ ; the third inequality is a consequence of $\theta^{\prime}\geq \theta (1 - \delta)$ . Hence (85) follows.

Lemma 16. Let $\theta \in (0,1] \cap \mathbb{Q}$ , and $\delta \in (0,1)$ . If we consider $\theta' \in [\theta - \theta\delta, \theta]$ and $T \in \mathbb{N}$ such that $\theta T, \theta'T \in \mathbb{N}$ , then

$$
\left\| \hat {\boldsymbol {\mu}} (\theta T) - \hat {\boldsymbol {\mu}} \left(\theta^ {\prime} T\right) \right\| _ {\infty} \leq 2 \delta . \tag {86}
$$

Proof. Observe that for any $k \in [K]$ ,

$$
\begin{array}{l} \left| \hat {\mu} _ {k} (\theta T) - \hat {\mu} _ {k} \left(\theta^ {\prime} T\right) \right| = \left| \frac {\sum_ {t \leq \theta T} X _ {k} (t)}{\theta T} - \frac {\sum_ {t \leq \theta^ {\prime} T} X _ {k} (t)}{\theta^ {\prime} T} \right| \\ \leq \left| \frac {\sum_ {t \leq \theta T} X _ {k} (t)}{\theta T} - \frac {\sum_ {t \leq \theta^ {\prime} T} X _ {k} (t)}{\theta T} \right| + \left| \frac {\sum_ {t \leq \theta^ {\prime} T} X _ {k} (t)}{\theta T} - \frac {\sum_ {t \leq \theta^ {\prime} T} X _ {k} (t)}{\theta^ {\prime} T} \right| \\ \leq \frac {\theta - \theta^ {\prime}}{\theta} + \theta^ {\prime} (\frac {1}{\theta^ {\prime}} - \frac {1}{\theta}) \\ = 1 - \frac {\theta^ {\prime}}{\theta} + 1 - \frac {\theta^ {\prime}}{\theta} \leq 2 \delta , \\ \end{array}
$$

where the first inequality uses the triangle inequality; the second inequality simply comes from $\sum_{\theta^{\prime}T < t\leq \theta T}X_{k}(t)\leq (\theta -\theta^{\prime})T$ and $\sum_{t\leq \theta T}X_k(t)\leq \theta 'T$ (as Assumption 1 ensures that $X_{k}(t)\in [0,1]$ ); the third inequality is a consequence of $\theta^{\prime}\geq \theta (1 - \delta)$ . Hence (86) follows.

# E.3 Local LDP upper bound on $\mathcal{X}_j\setminus \cup_{i = j}^{K}\mathcal{X}_{j,i}(\theta)$

Fix $\theta\in(0,1]\cap\mathbb{Q}$ . We will establish local LDP upper bounds on $\mathcal{X}_{j}\setminus\cup_{i=j}^{K}\mathcal{X}_{j,i}(\theta)$ for the process $\{\boldsymbol{\omega}(\theta T)\}_{T\geq 1}$ under CR-C and CR-A. The upper bound for CR-C is presented in E.3.1 and that for CR-A in E.3.2. We first present a useful proposition repeatedly used in E.3.1, E.3.2, and the main proofs for Theorem 3 and Theorem 4 in C.

One important property for $x \in X_{j} \setminus \cup_{i=j}^{K} X_{j,i}(\theta)$ is that the remaining budget for the empirical top-j arms is lower bounded by a constant. This observation is summarized in Proposition 7.

Proposition 7. If $\pmb{x} \in \mathcal{X}_j \setminus \cup_{i=j}^K \mathcal{X}_{j,i}(\theta)$ , then $\forall i \in \{j + 1, \dots, K\}$ ,

$$
1 - \theta \sum_ {k = i} ^ {K} x _ {\sigma (k)} \geq \frac {\overline {{\log}} (i - 1)}{\overline {{\log}} K}.
$$

Proof. Notice that the statement of the proposition is equivalent to (87): $\forall i \in \{j + 1, \ldots, K\}$ ,

$$
\theta \sum_ {k = i} ^ {K} x _ {\sigma (k)} \leq \frac {1}{\overline {{{{\log K}}}}} \sum_ {k = i} ^ {K} \frac {1}{k}. \tag {87}
$$

The inequalities (87) will be proved by induction.

1. We show (87) for $i = K$ . Since $\pmb{x} \notin \mathcal{X}_{j,K}(\theta)$ ,

$$
1 \geq \theta K \overline {{\log}} K x _ {\sigma (K)}. \tag {88}
$$

Dividing by $K\overline{\log} K$ on both sides of (88) directly yields (87) with $i = K$ . Now we assume (87) is valid for some $i + 1 \in \{j + 2, \dots, K\}$ , and we show (87) for $i$ .

2. We show (87) for $i$ assuming that (87) holds for $i + 1$ . As $\pmb{x} \notin \mathcal{X}_{j,i}(\theta)$ , we get:

$$
1 - \theta \sum_ {k = i + 1} ^ {K} x _ {\sigma (k)} \geq \theta i \overline {{\log}} i x _ {\sigma (i)}.
$$

Dividing the above inequality by $i\overline{\log}i$ and adding $\theta \sum_{k = i + 1}^{K}x_{\sigma (k)}$ to the both sides, we obtain that

$$
\begin{array}{l} \theta \sum_ {k = i} ^ {K} x _ {\sigma (k)} \leq \frac {1}{i \overline {{\log i}}} + (1 - \frac {1}{i \overline {{\log i}}}) \theta \sum_ {k = i + 1} ^ {K} x _ {\sigma (k)} \\ \leq \frac {1}{i \overline {{\log}} i} + (1 - \frac {1}{i \overline {{\log}} i}) \frac {1}{\overline {{\log}} K} \sum_ {k = i + 1} ^ {K} \frac {1}{k} \\ = \frac {1}{\overline {{\log}} K} \sum_ {k = i + 1} ^ {K} \frac {1}{k} + \frac {1}{i \overline {{\log}} i} \left[ 1 - \frac {1}{\overline {{\log}} K} \sum_ {k = i + 1} ^ {K} \frac {1}{k} \right], \\ \end{array}
$$

where the second inequality stems from our inductive hypothesis (87) for $i + 1$ . As $\overline{\log K} - \overline{\log i} = \sum_{k=i+1}^{K} \frac{1}{k}$ , the r.h.s on the above inequality is equal to

$$
\frac {1}{\overline {{\log}} K} \sum_ {k = i + 1} ^ {K} \frac {1}{k} + \frac {1}{i \overline {{\log}} i} \left[ 1 - \frac {1}{\overline {{\log}} K} (\overline {{\log}} K - \overline {{\log}} i) \right] = \frac {1}{\overline {{\log}} K} \sum_ {k = i} ^ {K} \frac {1}{k},
$$

hence (87) is proved.

![](images/6790d1dde3c6daec44f3ee26596c636e0265e2c79026d50dde33bfecabff54a5.jpg)

# E.3.1 The allocation process under CR-C

We show that $I_{\theta}$ presented in Theorem 7 below is a valid rate function for a local LDP upper bound (67) for the process $\{\omega(\theta T)\}_{T\geq1}$ . This function is however too complicated to use, and we will present a simpler rate function in Corollary 3.

Theorem 7. Let $j \in \{2, \ldots, K - 1\}$ , $\theta \in (0,1]$ , $\boldsymbol{x} \in \mathcal{X}_j \setminus \cup_{i=j}^{K} \mathcal{X}_{j,i}(\theta)$ , define

$$
I _ {\theta} (\boldsymbol {x}) = \max _ {p = j, \dots , K - 1} 2 x _ {\sigma (p + 1)} \inf _ {\boldsymbol {\lambda} \in \mathcal {S} _ {p} (\boldsymbol {x})} \sum_ {k = 1} ^ {p + 1} \left(\lambda_ {\sigma (k)} - \mu_ {\sigma (k)}\right) ^ {2}, \tag {89}
$$

where $\sigma$ is the permutation described in (65) for $z$ and

$$
\mathcal {S} _ {p} (\boldsymbol {x}) = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \lambda_ {\sigma (p + 1)} \leq \min _ {k \leq p} \lambda_ {\sigma (k)} - G \left(\frac {\theta x _ {\sigma (p + 1)} (p + 1) \overline {{\log}} (p + 1)}{1 - \theta \sum_ {k = p + 2} ^ {K} x _ {\sigma (k)}}\right) \right\}. \tag {90}
$$

Then the process $\{\pmb{\omega}(\theta T)\}_{T\geq 1}$ under $\mathbb{CR} - \mathbb{C}$ satisfies a local LDP upper bound (67) with $I_{\theta}$ at $\pmb{x}$ .

Proof. For simplicity, $\sigma$ is assumed to be the identity map. Let $p \in \{j, \ldots, K - 1\}$ . As $x \in X_j$ , we have $x_p - x_{p+1} > 0$ . Let T > K, and $\delta < (x_p - x_{p+1})/2$ be some positive number. We will derive an upper bound for $\mathbb{P}_{\boldsymbol{\mu}}[\boldsymbol{\omega}(\theta T) \in B(\boldsymbol{x}, \delta)]$ , and then compute its rate by driving $T \to \infty$ and $\delta \to 0$ .

Observe that of course:

$$
\mathbb {P} _ {\boldsymbol {\mu}} \left[ \boldsymbol {\omega} (\theta T) \in B (\boldsymbol {x}, \delta) \right] \leq \mathbb {P} _ {\boldsymbol {\mu}} \left[ \cup_ {\boldsymbol {y} \in B (\boldsymbol {x}, \delta)} \{\boldsymbol {\omega} (\theta T) = \boldsymbol {y} \} \right]. \tag {91}
$$

1. Obtaining a necessary condition for $\omega(\theta T) = \boldsymbol{y}$ . For any $\boldsymbol{y} \in B(\boldsymbol{x}, \delta)$ , we introduce $\theta(\boldsymbol{y})$ and $\boldsymbol{z}(\boldsymbol{y})$ as:

$$
\theta (\boldsymbol {y}) = \theta - \theta \sum_ {k = 1} ^ {p} (y _ {k} - y _ {p + 1}), \text {   and   } z _ {k} (\boldsymbol {y}) = \left\{ \begin{array}{l l} \theta y _ {k} / \theta (\boldsymbol {y}), & \text {   if   } k \geq p + 2, \\ \theta y _ {p + 1} / \theta (\boldsymbol {y}), & \text {   if   } k \leq p + 1. \end{array} \right.
$$

Following directly from the above definitions, we obtain that

$$
\theta (\boldsymbol {y}) \sum_ {k = 1} ^ {p + 1} z _ {k} (\boldsymbol {y}) = \theta (p + 1) y _ {p + 1}, \text { and } \theta (\boldsymbol {y}) z _ {k} (\boldsymbol {y}) = \theta y _ {k}, \forall k = p + 2, \dots K. \tag {92}
$$

From the choice of $\delta$ , $y_{p+1}$ is the smallest real number in $\{y_1, \ldots, y_{p+1}\}$ , so $\ell_{p+1} = p + 1$ . Moreover, $\theta(\boldsymbol{y})T$ is the round $\mathbb{CR}-\mathbb{C}$ decides $\ell_{p+1} = \ell(\theta(\boldsymbol{y})T) = p + 1$ , and $\omega(\theta(\boldsymbol{y})T) = \boldsymbol{z}(\boldsymbol{y})$ . Due to the condition for discarding $p + 1$ (see (11)), we have $\hat{\boldsymbol{\mu}}(\theta(\boldsymbol{y})T) \in \mathcal{S}_p(\boldsymbol{y})$ . Consequently, we have:

$$
\{\boldsymbol {\omega} (\theta T) = \boldsymbol {y} \} \subset \left\{\hat {\boldsymbol {\mu}} (\theta (\boldsymbol {y}) T) \in \mathcal {S} _ {p} (\boldsymbol {y}), \boldsymbol {\omega} (\theta (\boldsymbol {y}) T) = z (\boldsymbol {y}) \right\}. \tag {93}
$$

2. Reducing $\cup_{\boldsymbol{y} \in B(\boldsymbol{x}, \delta)} \{\boldsymbol{\omega}(\theta T) = \boldsymbol{y}\}$ to a single set. To do this reduction, we use the results of Step 1 and Lemmas 15 and 16.

Let $\pmb{y}_0\in \arg \max_{\pmb {y}\in B(\pmb {x},\delta)}\theta (\pmb {y})$ . Using the continuity of the function $\theta (\pmb {y})$ , there exists $\eta (\delta)$ such that $\eta (\delta)$ tends to 0 as $\delta \to 0$ , and for all $\pmb {y}\in B(\pmb {x},\delta),\theta (\pmb {y})\in [\theta (\pmb {y}_0)(1 - \eta (\delta)),\theta (\pmb {y}_0)]$ . By Lemmas 15 and 16, we obtain that:

$$
\left\| \hat {\mu} (\theta (\boldsymbol {y})) - \hat {\mu} \left(\theta \left(\boldsymbol {y} _ {0}\right)\right) \right\| _ {\infty} \leq 2 \eta (\delta),
$$

$$
\left\| \boldsymbol {\omega} (\theta (\boldsymbol {y}) T) - \boldsymbol {\omega} (\theta (\boldsymbol {y} _ {0}) T) \right\| _ {\infty} \leq 2 \eta (\delta).
$$

Now define the following sets:

$$
\bar {S} _ {\delta , p} = \cup_ {\boldsymbol {y} \in B (\boldsymbol {x}, \delta)} \{\bar {s}: \exists s \in \mathcal {S} _ {p} (\boldsymbol {y}): \| s - \bar {s} \| _ {\infty} \leq 2 \eta (\delta) \},
$$

$$
\bar {W} _ {\delta} = \cup_ {\boldsymbol {y} \in B (\boldsymbol {x}, \delta)} \{\bar {w} \in \Sigma : \| \bar {w} - z (\boldsymbol {y}) \| _ {\infty} \leq 2 \eta (\delta) \}.
$$

By construction, for all $\boldsymbol{y} \in B(\boldsymbol{x}, \delta)$ , we have that if $\hat{\boldsymbol{\mu}}(\theta(\boldsymbol{y})T) \in \mathcal{S}_{p}(\boldsymbol{y})$ , then $\hat{\boldsymbol{\mu}}(\theta(\boldsymbol{y}_{0})T) \in \bar{S}_{\delta,p}$ , and similarly $\boldsymbol{\omega}(\theta(\boldsymbol{y})T) = z(\boldsymbol{y})$ implies that $\boldsymbol{\omega}(\theta(\boldsymbol{y}_{0})T) \in \bar{W}_{\delta}$ .

3. Using Theorem 1. Putting the results of the first two steps together, we get:

$$
\mathbb {P} _ {\boldsymbol {\mu}} \left[ \boldsymbol {\omega} (\theta T) \in B (\boldsymbol {x}, \delta) \right] \leq \mathbb {P} _ {\boldsymbol {\mu}} \left[ \hat {\boldsymbol {\mu}} (\theta (\boldsymbol {y} _ {0}) T) \in \bar {S} _ {\delta , p}, \boldsymbol {\omega} (\theta (\boldsymbol {y} _ {0}) T) \in \bar {W} _ {\delta} \right]
$$

By applying (c) in Section 3.2 with $S = \bar{S}_{\delta,p}$ and $W = \bar{W}_{\delta}$ , it follows that

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \boldsymbol {\omega} (\theta T) \in B (\boldsymbol {x} , \delta) ]} \geq \theta (\boldsymbol {y} _ {0}) \inf _ {\boldsymbol {\omega} \in \bar {W} _ {\delta}} F _ {\bar {S} _ {\delta , p}} (\boldsymbol {\omega}).
$$

When $\delta$ tends to 0, the r.h.s. simply converges to $\theta(\boldsymbol{x})F_{\mathcal{S}_p(\boldsymbol{x})}(z(\boldsymbol{x}))$ . The latter is $\inf_{\boldsymbol{\lambda}\in \mathcal{S}_p(\boldsymbol{x})}2\sum_{k = 1}^{p + 1}x_{p + 1}(\lambda_k - \mu_k)^2$ (see Lemma 17 in Appendix F.1 for continuity arguments). As the above proof holds for any $p\in \{j,\dots ,K - 1\}$ , we complete the proof.

Next, as in Appendix C, we define the subset of $\mathcal{X}_{j} \setminus \cup_{i=j}^{K} \mathcal{X}_{j,i}(\theta)$ :

$$
\mathcal {Z} _ {[ j ]} (\theta , \beta) = \left\{\boldsymbol {z} \in \mathcal {X} _ {j} \setminus \cup_ {i = j} ^ {K} \mathcal {X} _ {j, i} (\theta): \sigma (k) = k,   \forall k \leq j \text { and } \frac {\theta z _ {j} j \overline {{\log}} j}{1 - \theta \sum_ {k > j} z _ {k}} = \beta \right\}
$$

for all $\beta\in(0,1]$ . Note that the permutation $\sigma$ in the above definition corresponds to that used for point z as in the definition of $X_{j}$ (65): it is such that $z_{\sigma(1)}=\ldots=z_{\sigma(j)}>z_{\sigma(j+1)}>\ldots>z_{\sigma(K)}>0$ .

Corollary 3. Let $j \in \{2, \ldots, K - 1\}$ , $\theta, \beta \in (0, 1]$ , $z \in \mathcal{Z}_{[j]}(\theta, \beta)$ . Define

$$
\underline {{I}} _ {\theta} (\pmb {z}) = \frac {\overline {{\log}} (j + 1)}{\theta \overline {{\log}} K} \left[ \left((1 + \zeta_ {j}) \sqrt {\frac {\theta z _ {\sigma (j + 1)}}{1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}}} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2},
$$

where $\sigma$ is the permutation described in (65) for $z$ , then $\underline{I}_{\theta}(z) \leq I_{\theta}(z)$ .

Proof. Observe that $I_{\theta}(z)$ is larger than the value of the following optimization problem:

$$
\min _ {\boldsymbol {\lambda} \in \mathbb {R} ^ {K}} 2 z _ {\sigma (j + 1)} \left(\left(\lambda_ {j} - \mu_ {j}\right) ^ {2} + \left(\lambda_ {\sigma (j + 1)} - \mu_ {\sigma (j + 1)}\right) ^ {2}\right), \tag {94}
$$

subject to $\lambda_{\sigma(j + 1)}\leq \lambda_j - G(\tilde{\beta})$

where

$$
\tilde {\beta} = \frac {\theta z _ {\sigma (j + 1)} (j + 1) \overline {{\log}} (j + 1)}{1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}}.
$$

One can simply verify that the optimal value of (94) is

$$
z _ {\sigma (j + 1)} [ (\mu_ {j} - \mu_ {\sigma (j + 1)} - G (\tilde {\beta})) _ {+} ] ^ {2} \geq z _ {\sigma (j + 1)} [ (1 + \zeta_ {j} - \frac {1}{\sqrt {\tilde {\beta}}}) _ {+} ] ^ {2}, \tag {95}
$$

where the inequality follows from the fact that $\mu_{j + 1}\geq \mu_{\sigma (j + 1)}$ and $G(\tilde{\beta}) = 1 / \sqrt{\tilde{\beta}} -1$ . Substituting the value of $\tilde{\beta}$ yields that the r.h.s. of (95) is equal to

$$
\frac {1}{\theta} \left(1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}\right) \left[ \left((1 + \zeta_ {j}) \sqrt {\frac {\theta z _ {\sigma (j + 1)}}{1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}}} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2}.
$$

As $z \in \mathcal{X}_j \setminus \cup_{i=j}^K \mathcal{X}_{j,i}(\theta)$ , Proposition 7 with $i = j + 2$ directly completes the proof.

# E.3.2 The allocation process under CR-A

One can use similar arguments as those used in the proof of Theorem 7 to derive the analogous rate function for the allocation process under CR-A.

Theorem 8. Let $j \in \{2, \ldots, K - 1\}$ , $\theta \in (0,1]$ , $\boldsymbol{x} \in \mathcal{X}_j \setminus \cup_{i=j}^K \mathcal{X}_{j,i}(\theta)$ , define

$$
I _ {\theta} (\boldsymbol {x}) = \max _ {p = j, \dots , K - 1} 2 x _ {\sigma (p + 1)} \inf _ {\boldsymbol {\lambda} \in \mathcal {S} _ {p} (\boldsymbol {x})} \sum_ {k = 1} ^ {p + 1} \left(\lambda_ {\sigma (k)} - \mu_ {\sigma (k)}\right) ^ {2}, \tag {96}
$$

where $\sigma$ is the permutation described in (65) for $z$ and

$$
\mathcal {S} _ {p} (\boldsymbol {x}) = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \lambda_ {\sigma (p + 1)} \leq \frac {\sum_ {k = 1} ^ {p} \lambda_ {\sigma (k)}}{p} - G \left(\frac {\theta x _ {\sigma (p + 1)} (p + 1) \overline {{\log}} (p + 1)}{1 - \theta \sum_ {k = p + 2} ^ {K} x _ {\sigma (k)}}\right) \right\}. \tag {97}
$$

Then the process $\{\omega(\theta T)\}_{T\geq1}$ under CR-A satisfies a local LDP upper bound (67) with $\bar{I}_{\theta}$ at x.

The proof is omitted as it is almost the same as that of Theorem 7. We can also obtain the analogous version of Corollary 3 as shown below:

Corollary 4. Let $j \in \{2, \ldots, K\}$ , $\theta, \beta \in (0,1]$ , $z \in \mathcal{Z}_{[j]}(\theta, \beta)$ , we define

$$
\underline {{I}} _ {\theta} (\boldsymbol {z}) = \frac {2 j \overline {{{\log}}} (j + 1)}{\theta (j + 1) \overline {{{\log}}} K} \left[ \left((1 + \varphi_ {j}) \sqrt {\frac {\theta z _ {\sigma (j + 1)}}{1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}}} - \sqrt {\frac {1}{(j + 1) \overline {{{\log}}} (j + 1)}}\right) _ {+} \right] ^ {2}, \tag {98}
$$

where $\sigma$ is the permutation described in (65) for $z$ , then $\underline{I}_{\theta}(z) \leq I_{\theta}(z)$ .

Proof. We can simply solve the optimization problem similar to (61) as in the proof Proposition 6 in Appendix D.2 and get that $I_{\theta}(z)$ is greater than the l.h.s. of the following inequality.

$$
\frac {2 j z _ {\sigma (j + 1)}}{j + 1} \left[ \left(\frac {\sum_ {k = 1} ^ {j} \mu_ {k}}{j} - \mu_ {\sigma (j + 1)} - G (\tilde {\beta})\right) _ {+} \right] ^ {2} \geq \frac {2 j z _ {\sigma (j + 1)}}{j + 1} \left[ \left((1 + \varphi_ {j}) - \frac {1}{\sqrt {\tilde {\beta}}}\right) _ {+} \right] ^ {2}, \tag {99}
$$

where

$$
\tilde {\beta} = \frac {\theta z _ {\sigma (j + 1)} (j + 1) \overline {{\log}} (j + 1)}{1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}}.
$$

(99) is due to $\mu_{j + 1}\geq \mu_{\sigma (j + 1)}$ (hence $\sum_{k = 1}^{j}\mu_{k} / j - \mu_{\sigma (j + 1)}\geq \varphi_j$ ) and $G(\tilde{\beta}) = \frac{1}{\sqrt{\tilde{\beta}}} -1$ . Substituting the value of $\tilde{\beta}$ yields that the r.h.s. of (99) equals to

$$
\frac {2 j}{\theta (j + 1)} \left(1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}\right) \left[ \left((1 + \varphi_ {j}) \sqrt {\frac {\theta z _ {\sigma (j + 1)}}{1 - \theta \sum_ {k = j + 2} ^ {K} z _ {\sigma (k)}}} - \sqrt {\frac {1}{(j + 1) \overline {{\log}} (j + 1)}}\right) _ {+} \right] ^ {2}.
$$

As $\boldsymbol{z} \in \mathcal{X}_j \setminus \cup_{i=j}^{K} \mathcal{X}_{j,i}(\theta)$ , Proposition 7 with $i = j + 2$ directly completes the proof.

![](images/78695bccdc8d67fc93cb3b3bd98860f286d319a369658c84d0c5b5282400ef5e.jpg)

# F Continuity arguments

We introduce some definitions and results taken from (Berge, 1997), and also used recently in (Wang et al., 2021; Degenne and Koolen, 2019; Combes et al., 2017) in the bandit literature.

Suppose $\mathbb{X}$ and $\mathbb{Y}$ are Hausdorff topological spaces. Let $u: \mathbb{X} \times \mathbb{Y} \to \mathbb{R}$ be a function and $\Phi: \mathbb{X} \rightrightarrows \mathbb{S}(\mathbb{Y})$ be a set-valued function, where $\mathbb{S}(\mathbb{Y})$ is the set of non-empty subsets of $\mathbb{Y}$ . Besides, we introduce $\mathbb{K}(\mathbb{X}) = \{F \in \mathbb{S}(\mathbb{X}): F \text{ is compact}\}$ . We are interested in a minimization problem of the form: for $x \in \mathbb{X}$ ,

$$
v (x) = \inf _ {y \in \Phi (x)} u (x, y).
$$

We define the set of solutions of this problem as $\Phi^{*}(x)=\{y\in\Phi(x):u(x,y)=v(x)\}$ .

Theorem 9 ((Berge, 1997)). Assume that

- $\Phi : \mathbb{X} \rightrightarrows \mathbb{K}(\mathbb{X})$ is continuous (i.e., both lower and upper hemicontinous),   
- $u: \mathbb{X} \times \mathbb{Y} \to \mathbb{R}$ is continuous.

Then the function $v: \mathbb{X} \to \mathbb{R}$ is continuous and the solution multifunction $\Phi^{*}: \mathbb{X} \to \mathbb{S}(\mathbb{Y})$ is upper hemicontinuous, with values that are nonempty and compact.

# F.1 Verifying the continuity in Theorem 7

We verify the continuity argument used in the proof of Theorem 7.

Lemma 17. The function

$$
\inf _ {\boldsymbol {\lambda} \in \mathcal {S} _ {p} (\boldsymbol {x})} \sum_ {k = 1} ^ {p + 1} x _ {p + 1} \left(\lambda_ {k} - \mu_ {k}\right) ^ {2},
$$

$$
w h e r e \mathcal {S} _ {p} (\boldsymbol {x}) = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \lambda_ {p + 1} \leq \min _ {k \leq p} \lambda_ {k} - G \left(\frac {\theta x _ {p + 1} (p + 1) \overline {{\log}} (p + 1)}{1 - \theta \sum_ {k = j + 2} ^ {K} x _ {k}}\right) \right\},
$$

is continuous for all $\pmb{x} \in \Sigma$ .

Proof. We apply Theorem 9 with:

$$
\begin{array}{l} \bullet \mathbb {X} = \Sigma , \quad \bullet \Phi (\boldsymbol {x}) = \mathcal {S} _ {p} (\boldsymbol {x}), \\ \bullet \mathbb {Y} = [ 0, 1 ] ^ {K}, \quad \bullet u (\boldsymbol {x}, \boldsymbol {\lambda}) = \sum_ {k = 1} ^ {K} x _ {p + 1} (\lambda_ {k} - \mu_ {k}) ^ {2}. \\ \end{array}
$$

As the objective function is obviously continuous, it remains to show that $\mathcal{S}_{p}(\cdot)$ is a continuous correspondence. By simply using Lemma 19 with $f(\boldsymbol{\lambda}) = \lambda_{p+1} - \min_{k \leq p} \lambda_{k}$ and $g(\boldsymbol{x}) = -G\left(\frac{\theta x_{p+1}(p+1)\overline{\log(p+1)}}{1-\theta \sum_{k=p+2}^{K} x_{k}}\right)$ , we can complete the proof. □

It is straightforward to get a similar guarantee for the function involved in CR-A: we hence omit the proof of the following lemma.

Lemma 18. The function

$$
\inf _ {\boldsymbol {\lambda} \in \mathcal {S} _ {p} (\boldsymbol {x})} \sum_ {k = 1} ^ {p + 1} x _ {p + 1} \left(\lambda_ {k} - \mu_ {k}\right) ^ {2},
$$

$$
\text { where } \mathcal {S} _ {p} (\boldsymbol {x}) = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: \lambda_ {p + 1} \leq \frac {\sum_ {k \leq p} \lambda_ {k}}{p} - G \left(\frac {\theta x _ {p + 1} (p + 1) \overline {{\log}} (p + 1)}{1 - \theta \sum_ {k = j + 2} ^ {K} x _ {k}}\right) \right\},
$$

is continuous for all $\pmb{x} \in \Sigma$ .

Lemma 19. Let $g: \Sigma \mapsto \mathbb{R}$ be a continuous mapping and $f:[0,1]^K \mapsto \mathbb{R}$ be a lower semicontinuous mapping which further satisfies that $\forall \pmb{\lambda} \in \mathbb{R}^K, \delta > 0$ , there exists $\pmb{\lambda}' \in \mathbb{R}^K$ s.t. $\| \pmb{\lambda} - \pmb{\lambda}' \|_{\infty} \leq \delta$ and $f(\pmb{\lambda}') < f(\pmb{\lambda})$ . Then $\forall \pmb{x} \in \Sigma$ ,

$$
\mathcal {S} (\boldsymbol {x}) = \left\{\boldsymbol {\lambda} \in [ 0, 1 ] ^ {K}: f (\boldsymbol {\lambda}) \leq g (\boldsymbol {x}) \right\},
$$

is a continuous correspondence.

Proof. (i) Upper hemicontinuity: Suppose $\{\pmb{x}_n\}_{n\in \mathbb{N}}\subset \Sigma$ converges to $\pmb{x}^{\star}\in \Sigma$ and $\{\lambda_n\}_{n\in \mathbb{N}}\subset \mathbb{R}^K$ converges to $\pmb{\lambda}^{\star}$ s.t. $\pmb{\lambda}_n\in S(\pmb{x}_n)$ for all $n\in \mathbb{N}$ . We will show that $\pmb{\lambda}^{\star}\in S(\pmb{x}^{\star})$ . Since $g$ is upper semicontinuous, and $\pmb{x}_n\to \pmb{x}^\star$ as $n\to \infty$ , for any $\epsilon >0$ , $\exists N_{\epsilon}\in \mathbb{N}$ s.t. $g(\pmb {x}_n)\leq g(\pmb{x}^\star) + \epsilon$ for all $n\geq N_{\epsilon}$ . As $\pmb {\lambda}_n\in S(\pmb {x}_n)$ , we deduce that

$$
f (\boldsymbol {\lambda} _ {n}) \leq g (\boldsymbol {x} _ {n}) \leq g (\boldsymbol {x} ^ {\star}) + \epsilon , \forall n \geq N _ {\epsilon}.
$$

Now the lower semicontinuity of $f$ implies $f(\pmb{\lambda}^{\star}) \leq \varprojlim_{n\to \infty}f(\pmb{\lambda}_n)\leq g(\pmb{x}^{\star}) + \epsilon$ . As $\epsilon$ can be taken arbitrarily, $\pmb{\lambda}^{\star}\in S(\pmb{x}^{\star})$ .

(ii) Lower hemicontinuity: Suppose $\{\pmb{x}_n\}_{n\in \mathbb{N}}\subset \Sigma$ converges to $\pmb{x}^{\star}\in \Sigma$ , we aim to show that for all $\pmb{\lambda}^{\star}\in S(\pmb{x}^{\star})$ (or equivalently $f(\pmb{\lambda}^{\star})\leq g(\pmb{x}^{\star})$ ), there exist a subsequence $\{\pmb{x}_{n_k}\}_{k\in \mathbb{N}}\subseteq \{\pmb{x}_n\}_{n\in \mathbb{N}}$ and a sequence $\{\pmb{\lambda}_k\}_{k\in \mathbb{N}}$ s.t. $\pmb{\lambda}_k\in S(\pmb{x}_{n_k})$ and $\pmb{\lambda}_k\to \pmb{\lambda}^\star$ as $k\to \infty$ . By assumption on $f$ , for any integer $k$ , there exists $\pmb{\lambda}_k$ s.t. $\| \pmb{\lambda}_k - \pmb{\lambda}^\star \|_{\infty} < 1 / k$ and $f(\pmb {\lambda}_k) < f(\pmb {\lambda}^\star)$ . Also, $g(\pmb {x}_n)\rightarrow g(\pmb {x}^{\star})$ as $n\to \infty$ implies there exists a finite $n$ s.t. $g(\pmb {x}_n)\geq f(\pmb {\lambda}_k)$ due to the lower semicontinuity of $g$ . Hence we can always find a subsequence $\{n_k\}$ s.t. $g(\pmb {x}_{n_k})\geq f(\pmb {\lambda}_k)$ , which is equivalent to $\pmb{\lambda}_k\in S(\pmb {x}_{n_k})$ .

Lemma 20. When S is a bounded set in $R^{K}$ , $F_{S}(\cdot)$ is a continuous function.

Proof. This is a direct application of Berge's maximum theorem.

# G A partitioning technique for large deviations

In this section, we establish a theorem that is instrumental in the large deviation analysis of our algorithms.

Theorem 10. Let $\theta_0, \tilde{\theta}_0 \in (0,1)$ . Let $(\mathcal{S}_{\theta,\gamma})_{\theta \in [\theta_0,1],\gamma \in [\tilde{\theta}_0,1]}$ and $(W_{\theta,\gamma})_{\theta \in [\theta_0,1],\gamma \in [\tilde{\theta}_0,1]}$ two collections of Borel sets in $[0,1]^K$ and $\Sigma$ , respectively. We assume that

Suppose (i) for any $T \in N$ ,

$$
\mathbb {P} _ {\boldsymbol {\mu}} [ \mathcal {E} ] \leq \sum_ {t \geq \theta_ {0} T} ^ {T} \sum_ {\tau \geq \tilde {\theta} _ {0} T} ^ {T} \mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} _ {\frac {t}{T}, \frac {\tau}{T}}, \boldsymbol {\omega} (t) \in W _ {\frac {t}{T}, \frac {\tau}{T}} ],
$$

(ii) for any $\theta \in [\theta_0, 1] \cap \mathbb{Q}$ , $\{\omega(\theta T)\}_{T \geq 1}$ satisfies LDP upper bound (3) with $I_\theta$ , where $I_\theta$ is lower semi-continuous in $\Sigma$ .

(iii) $\forall \theta, \gamma, S_{\theta, \gamma} \neq \emptyset$ . For all $\delta > 0$ , there exists $\eta > 0$ such that if $\max \{|\theta' - \theta|, |\gamma' - \gamma|\} < \eta$ , then

$$
\mathrm{dist} (\mathcal {S} _ {\theta , \gamma}, \mathcal {S} _ {\theta^ {\prime}, \gamma^ {\prime}}) = \max \left\{\sup _ {s \in \mathcal {S} _ {\theta , \gamma}} \inf _ {s ^ {\prime} \in \mathcal {S} _ {\theta^ {\prime}, \gamma^ {\prime}}} \| s - s ^ {\prime} \| _ {\infty},   \sup _ {s ^ {\prime} \in \mathcal {S} _ {\theta^ {\prime}, \gamma^ {\prime}}} \inf _ {s \in \mathcal {S} _ {\theta , \gamma}} \| s ^ {\prime} - s \| _ {\infty} \right\}.
$$

Under Assumption 1, we have

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \mathcal {E} ]} \geq \inf _ {\theta \in [ \theta_ {0}, 1 ] \cap \mathbb {Q}} \inf _ {\gamma \in [ \hat {\theta} _ {0}, 1 ] \cap \mathbb {Q}} \inf _ {\boldsymbol {\omega} \in \mathrm{cl} (W _ {\theta , \gamma})} \theta \max \{F _ {\mathcal {S} _ {\theta , \gamma}} (\boldsymbol {\omega}), I _ {\theta} (\boldsymbol {\omega}) \}. \tag {100}
$$

Proof. Without loss of generality, we can assume $\theta_0 = \tilde{\theta}_0$ . If $\theta_0 < \tilde{\theta}_0$ , we further define $S_{\theta, \gamma} = S_{\theta, \tilde{\theta}_0}$ and $W_{\theta, \gamma} = W_{\theta, \tilde{\theta}_0}$ for $\theta_0 \leq \gamma < \tilde{\theta}_0$ . And we handle the case $\theta_0 > \tilde{\theta}_0$ similarly.

The main idea behind the proof is to partition the set of instants $t \in \{\theta_0 T, \ldots, T\}$ into a finite collection of instant sets such that if $t$ lies within one of these sets then we may bound $\mathbb{P}_{\boldsymbol{\mu}}[\hat{\boldsymbol{\mu}}(t) \in S_{\frac{t}{T}, \frac{\tau}{T}}, \boldsymbol{\omega}(t) \in W_{\frac{t}{T}, \frac{\tau}{T}}]$ uniformly. From this partition, we can rewrite the upper bound $\mathbb{P}_{\boldsymbol{\mu}}[\mathcal{E}]$ as a finite sum. This sum is further upper bounded by a maximum over each of its terms. We conclude by taking the limit as $T$ grows large - since we deal with the maximum over a finite number of terms, the limit and the maximum can be exchanged.

Step 1. Partition of $[\theta_0, 1]$ . To build this partition, we leverage the results of Lemmas 15 and 16. Let $\overline{\delta > 0}$ . We construct $N_{\delta}$ points $\theta_1, \ldots, \theta_{N_{\delta}}$ as follows: $\theta_{N_{\delta}} = 1$ and

$$
\theta_ {n} = (1 - \frac {\delta}{2}) ^ {- n} \theta_ {0}, \quad \forall n = 1, \dots , N _ {\delta} - 1.
$$

$N_{\delta}$ is chosen as $\min\{p \in \mathbb{N} : \theta_0(1 - \frac{\delta}{2})^{-p} \geq 1\}$ . Now observe by construction that:

$$
\cup_ {n = 1} ^ {N _ {\delta}} \left[ \theta_ {n - 1}, \theta_ {n} \right] = \left[ \theta_ {0}, 1 \right], \tag {101}
$$

$$
\forall n, (\theta \in [ \theta_ {n - 1}, \theta_ {n} ]) \Longrightarrow (\theta \in [ \theta_ {n} (1 - \frac {\delta}{2}), \theta_ {n} ]). \tag {102}
$$

Step 2. Uniform upper bounds of $\mathbb{P}_{\boldsymbol{\mu}}[\hat{\boldsymbol{\mu}}(t) \in \mathcal{S}_{\frac{t}{T}, \frac{\tau}{T}}, \boldsymbol{\omega}(t) \in W_{\frac{t}{T}, \frac{\tau}{T}}]$ . We define the following sets: for all $n, m = 1, \ldots, N_{\delta}$ ,

$$
\bar {S} _ {n, m} ^ {\delta} = \cup_ {\theta \in [ \theta_ {n - 1}, \theta_ {n} ] \cap \mathbb {Q}} \cup_ {\gamma \in [ \theta_ {m - 1}, \theta_ {m} ] \cap \mathbb {Q}} \left\{\bar {s} \in [ 0, 1 ] ^ {K}: \exists s \in \mathcal {S} _ {\theta , \gamma}: \| \bar {s} - s \| _ {\infty} \leq \delta \right\},
$$

$$
\bar {W} _ {n, m} ^ {\delta} = \cup_ {\theta \in [ \theta_ {n - 1}, \theta_ {n} ] \cap \mathbb {Q}} \cup_ {\gamma \in [ \theta_ {m - 1}, \theta_ {m} ] \cap \mathbb {Q}} \left\{\bar {w} \in \Sigma : \exists w \in W _ {\theta , \gamma}: \| \bar {w} - w \| _ {\infty} \leq \delta \right\}.
$$

Let $\theta = t / T$ , $\gamma = \tau / T$ , and assume that $\theta \in [\theta_{n-1}, \theta_n]$ , $\gamma \in [\theta_{m-1}, \theta_m]$ . Then $\hat{\mu}(t) \in S_{\frac{t}{T}, \frac{\tau}{T}}$ implies that $\hat{\mu}(\theta_n) \in \bar{S}_{n,m}^\delta$ from Lemma 16. Similarly, $\omega(t) \in W_{\frac{t}{T}, \frac{\tau}{T}}$ implies that $\omega(\theta_n) \in \bar{W}_{n,m}^\delta$ from Lemma 15. We conclude that: for all $t, \tau$ such that $t / T \in [\theta_{n-1}T, \theta_nT]$ and $\tau / T \in [\theta_{m-1}T, \theta_mt]$ ,

$$
\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (t) \in \mathcal {S} _ {\frac {t}{T}, \frac {\tau}{T}}, \boldsymbol {\omega} (t) \in W _ {\frac {t}{T}, \frac {\tau}{T}} ] \leq \mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (\theta_ {n} T) \in \bar {S} _ {n} ^ {\delta}, \boldsymbol {\omega} (\theta_ {n} T) \in \bar {W} _ {n, m} ^ {\delta} ].
$$

Step 3. Upper bound on $\mathbb{P}_{\mu}[\mathcal{E}]$ . We denote by $p_n$ the number of instants $t$ such that $\frac{t}{T} \in [\theta_{n-1}T, \theta_nT]$ . From the above inequality, we conclude that:

$$
\begin{array}{l} \mathbb {P} _ {\boldsymbol {\mu}} [ \mathcal {E} ] \leq \sum_ {n = 1} ^ {N _ {\delta}} \sum_ {m = 1} ^ {N _ {\delta}} p _ {n} p _ {m} \mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (\theta_ {n} T) \in \bar {S} _ {n, m} ^ {\delta}, \boldsymbol {\omega} (\theta_ {n}) \in \bar {W} _ {n, m} ^ {\delta} ], \\ \leq (\sum_ {n = 1} ^ {N _ {\delta}} p _ {n}) (\sum_ {m = 1} ^ {N _ {\delta}} p _ {m}) \max _ {n, m \in \{1, \ldots , N _ {\delta} \}} \mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (\theta_ {n} T) \in \bar {S} _ {n, m} ^ {\delta}, \boldsymbol {\omega} (\theta_ {n} T) \in \bar {W} _ {n, m} ^ {\delta} ]. \\ \end{array}
$$

We note that $\sum_{n=1}^{N_{\delta}} p_n$ is roughly equal to $T$ , and always smaller than $T + 2N_{\delta}$ . Taking the logarithm, dividing by $-T$ , and the liminf of the above inequality, we get:

$$
\begin{array}{l} \varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \mathcal {E} ]} \geq \varliminf_ {T \to \infty} \frac {1}{T} \min _ {n, m \in \{1, \dots , N _ {\delta} \}} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (\theta_ {n} T) \in \bar {S} _ {n , m} ^ {\delta} , \boldsymbol {\omega} (\theta_ {n} T) \in \bar {W} _ {n , m} ^ {\delta} ]}, \\ = \min _ {n \in \{1, \dots , N _ {\delta} \}} \varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (\theta_ {n} T) \in \bar {S} _ {n , m} ^ {\delta} , \boldsymbol {\omega} (\theta_ {n} T) \in \bar {W} _ {n , m} ^ {\delta} ]}, \\ \geq \min _ {n, m \in \{1, \dots , N _ {\delta} \}} \theta_ {n} \inf _ {\boldsymbol {\omega} \in \operatorname{cl} (\bar {W} _ {n, m} ^ {\delta})} \max \left\{F _ {\bar {S} _ {n, m} ^ {\delta}} (\boldsymbol {\omega}), I _ {\theta_ {n}} (\boldsymbol {\omega}) \right\}, \\ \end{array}
$$

where the last inequality follows from Theorem 1 with $S = \bar{S}_{n,m}^{\delta}$ and $W = \bar{W}_{n,m}^{\delta}$ .

Step 4. Continuity arguments. The last step consists in proving that:

$$
\begin{array}{l} \operatorname * {l i m s u p} _ {\delta \to 0} \min _ {n, m \in \{1, \dots , N _ {\delta} \}} \inf _ {\boldsymbol {\omega} \in \mathrm{cl} (\bar {W} _ {n, m} ^ {\delta})} \theta_ {n} \max \left\{F _ {\bar {S} _ {n, m} ^ {\delta}} (\boldsymbol {\omega}), I _ {\theta_ {n}} (\boldsymbol {\omega}) \right\} \\ \geq \inf _ {\theta , \gamma \in [ \theta_ {0}, 1 ] \cap \mathbb {Q}} \inf _ {\boldsymbol {\omega} \in \operatorname{cl} (W _ {\theta , \gamma})} \theta \max \left\{F _ {\mathcal {S} _ {\theta , \gamma}} (\boldsymbol {\omega}), I _ {\theta} (\boldsymbol {\omega}) \right\}. \\ \end{array}
$$

We first state two uniform continuity results, proved in Lemma 21: For any $\epsilon > 0$ , $\theta, \gamma \in [\theta_0, 1]$ , there exists $\delta > 0$ such that

$$
\forall \boldsymbol {\omega}, \forall n, m = 1, \dots , N _ {\delta}, \quad F _ {\bar {S} _ {n, m} ^ {\delta}} (\boldsymbol {\omega}) \geq F _ {\mathcal {S} _ {\theta_ {n}, \theta_ {m}}} (\boldsymbol {\omega}) - \epsilon , \tag {103}
$$

$$
\forall \boldsymbol {\omega}, \boldsymbol {\omega} ^ {\prime}: \| \boldsymbol {\omega} - \boldsymbol {\omega} ^ {\prime} \| _ {\infty} \leq \delta , F _ {\mathcal {S} _ {\theta , \gamma}} (\boldsymbol {\omega} ^ {\prime}) \geq F _ {\mathcal {S} _ {\theta , \gamma}} (\boldsymbol {\omega}) - \epsilon , I _ {\theta} (\boldsymbol {\omega} ^ {\prime}) \geq I _ {\theta} (\boldsymbol {\omega}) - \epsilon . \tag {104}
$$

(103) is the consequence of (iii) and Lemma 21. (104) immediately follows from the compactness of $\Sigma$ , lower semi-continuity of $I_{\theta}$ , and $F_{S_{\theta,\gamma}}$ (see Lemma 20 and (ii)). We fix such a $\delta$ , and consider any convergent sequence $(n_k, m_k, \omega_k)_k$ with values in $\{1, \ldots, N_\delta\}^2 \times \Sigma$ such that if $n_k = n, m_k = m$ then $\omega_k \in \bar{W}_{n,m}^\delta$ . Denote by $(n, m, \omega_0)$ its limit. We let:

$$
g ^ {\star} = \varliminf_ {k \to \infty} \theta_ {n _ {k}} \max (F _ {\bar {S} _ {n _ {k}, m _ {k}} ^ {\delta}} (\boldsymbol {\omega} _ {k}), I _ {\theta_ {n _ {k}}} (\boldsymbol {\omega} _ {k})).
$$

Then, we have:

$$
\begin{array}{l} g ^ {\star} \geq \theta_ {n} \max (F _ {\bar {S} _ {n, m} ^ {\delta}} (\pmb {\omega} _ {0}), I _ {\theta_ {n}} (\pmb {\omega} _ {0})) - \epsilon \\ \geq \theta_ {n} \max (F _ {\mathcal {S} _ {\theta_ {n}, \theta_ {m}}} (\boldsymbol {\omega} _ {0}), I _ {\theta_ {n}} (\boldsymbol {\omega} _ {0})) - 2 \epsilon \\ \geq \theta_ {n} \max (F _ {\mathcal {S} _ {\theta_ {n}, \theta_ {m}}} (\boldsymbol {\omega}), I _ {\theta_ {n}} (\boldsymbol {\omega})) - 3 \epsilon , \\ \end{array}
$$

for some $\theta \in [\theta_{n-1},\theta_n]$ , $\gamma \in [\theta_{m-1},\theta_m]$ , $\omega \in \mathrm{cl}(W_{\theta,\gamma})$ . The first inequality is due to (104), the second to (103), and the third to the fact that $\omega_0 \in \bar{W}_{n,m}^\delta$ and (104). We conclude that:

$$
g ^ {\star} \geq \inf _ {\theta , \gamma \in [ \theta_ {0}, 1 ] \cap \mathbb {Q}} \inf _ {\boldsymbol {\omega} \in \operatorname{cl} (W _ {\theta , \gamma})} \theta \max \left\{F _ {\mathcal {S} _ {\theta , \gamma}} (\boldsymbol {\omega}), I _ {\theta} (\boldsymbol {\omega}) \right\} - 3 \epsilon .
$$

![](images/653d2cb2aa55e57795bf0c07772735f2829b03d3af88397168e8edadd498ca05.jpg)

Lemma 21. Assume $(\mathcal{S}_{\theta, \gamma})_{\theta \in [\theta_0, 1], \gamma \in [\tilde{\theta}_0, 1]}$ is a collection of nonempty sets in $[0, 1]^K$ that satisfies $\forall \delta > 0$ , there exists $\eta > 0$ such that if $\max \{|\theta' - \theta|, |\gamma' - \gamma|\} < \eta$ , then

$$
\operatorname{dist} \left(\mathcal {S} _ {\theta , \gamma}, \mathcal {S} _ {\theta^ {\prime}, \gamma^ {\prime}}\right) = \max \left\{\sup _ {s \in \mathcal {S} _ {\theta , \gamma}} \inf _ {s ^ {\prime} \in \mathcal {S} _ {\theta^ {\prime}, \gamma^ {\prime}}} \| s - s ^ {\prime} \| _ {\infty}, \sup _ {s ^ {\prime} \in \mathcal {S} _ {\theta^ {\prime}, \gamma^ {\prime}}} \inf _ {s \in \mathcal {S} _ {\theta , \gamma}} \| s ^ {\prime} - s \| _ {\infty} \right\}.
$$

Then (103) holds.

Proof. Recall that $F_{\mathcal{S}_{\theta}}(\boldsymbol{\omega}) = \inf_{\boldsymbol{\lambda} \in \mathrm{cl}(\mathcal{S}_{\theta})} \Psi(\boldsymbol{\lambda}, \boldsymbol{\omega}) = \inf_{\boldsymbol{\lambda} \in \mathrm{cl}(\mathcal{S}_{\theta})} \sum_k \omega_k d(\lambda_k, \mu_k)$ . $\Psi$ is uniformly continuous in $[0,1]^K \times \Sigma$ , and hence:

$$
\forall \epsilon > 0, \exists \bar {\delta}: \forall \boldsymbol {\lambda}, \boldsymbol {\lambda} ^ {\prime}, \| \boldsymbol {\lambda} - \boldsymbol {\lambda} ^ {\prime} \| _ {\infty} \leq 2 \bar {\delta} \Rightarrow | \Psi (\boldsymbol {\lambda}, \boldsymbol {\omega}) - \Psi (\boldsymbol {\lambda} ^ {\prime}, \boldsymbol {\omega}) | \leq \epsilon .
$$

Based on the assumption in the lemma, there exists $\eta > 0$ such that if $\max \{|\theta' - \theta|, |\gamma' - \gamma|\} < \eta$ , then

$$
\mathrm{dist} (\mathcal {S} _ {\theta , \gamma}, \mathcal {S} _ {\theta^ {\prime}, \gamma^ {\prime}}) <   \bar {\delta}.
$$

Select $\delta < \min(\bar{\delta}, 2\eta)$ . Let $n, m \in \{1, \ldots, N_{\delta}\}$ . For $(\theta, \gamma) \in [\theta_{n-1}, \theta_n] \times [\theta_{m-1}, \theta_m]$ , we have $\max(|\theta - \theta_n|, |\gamma - \theta_m|) \leq \delta/2 < \eta$ . This implies that:

$$
\operatorname{dist} \left(\mathcal {S} _ {\theta , \gamma}, \mathcal {S} _ {\theta_ {n}, \theta_ {m}}\right) <   \bar {\delta}.
$$

And hence since $\mathcal{S}_{\theta_n,\theta_m}\subset \bar{S}_{n,m}^\delta$

$$
\operatorname{dist} \left(\mathcal {S} _ {\theta , \gamma}, \bar {S} _ {n, m} ^ {\delta}\right) <   \bar {\delta}.
$$

We conclude that $\forall\boldsymbol{\omega},\;\forall\boldsymbol{\lambda}\in\mathcal{S}_{\theta,\gamma},\;\exists\boldsymbol{\lambda}^{\prime}\in\bar{S}_{n,m}^{\delta},$

$$
\Psi (\boldsymbol {\lambda}, \omega) \geq \Psi (\boldsymbol {\lambda} ^ {\prime}, \omega) - \epsilon .
$$

This completes the proof.

□

# H A simple min-max problem

The two following results are used in Appendix C.

Lemma 22. Let $b_{1}, c_{1}, b_{2}, c_{2} > 0$ . Introduce $f(x) = -b_{1}x + c_{1}$ and $g(x) = b_{2}x + c_{2}$ , $\forall x \in \mathbb{R}$ . Then

$$
\inf _ {x \in \mathbb {R}} \max \{f (x), g (x) \} \geq f (x _ {0}),
$$

where $x_0$ is the real number s.t. $x_0 \geq 0$ , $f(x_0) = g(x_0)$

Proof. As g is an increasing function and f is a decreasing function, we deduce that for all $x \geq x_{0}$ , $\max\{f(x), g(x)\} \geq g(x) \geq g(x_{0}) = f(x_{0})$ . Similarly for all $x \leq x_{0}$ , $\max\{f(x), g(x)\} \geq f(x) \geq f(x_{0})$ .

![](images/9865c241f100c73a4821d8f6cfda7db0911c3f318ce0019ad6b95972c0f48294.jpg)

Lemma 23. Let $b_{1}, c_{1}, b_{2}, c_{2} > 0$ . Introduce $f(x) = -b_{1}x + c_{1}$ and $g(x) = [(c_{2}\sqrt{x} - b_{2})_{+}]^{2}$ for $x \in \mathbb{R}_{+}$ . Then

$$
\inf _ {x \in \mathbb {R} _ {+}} \max \{f (x), g (x) \} \geq f (x _ {0}),
$$

where $x_{0}$ is the unique real number s.t. $x_{0} > 0$ and $f(x_{0}) = g(x_{0})$ .

Proof. We first prove the uniqueness of $x_0$ . Observe that $g$ is an increasing function, whereas $f$ is a strictly decreasing function. From the definition, we can have $f(0) = c_1 > 0 = g(0)$ , hence there exists an unique point $x_0 > 0$ s.t. $f(x_0) = g(x_0)$ . Leveraging the convexity of $g$ , there exists a linear function $\underline{g}$ s.t. $g(x) \geq \underline{g}(x)$ and $g(x_0) = \underline{g}(x_0)$ . The proof then follows from the fact that

$$
\inf _ {x \in \mathbb {R} _ {+}} \max \{f (x), g (x) \} \geq \inf _ {x \in \mathbb {R}} \max \{f (x), \underline {{g}} (x) \} \geq f (x _ {0}),
$$

where the last inequality is the application of Lemma 22.

![](images/38429f5a5179dc57691d604871871b1bd5e21c08dbb240106410ca1ab92340f9.jpg)

# I The LDP conjecture and its consequence

In this section, we restate the conjecture mentioned in Section 3.2, and we discuss how it relates to the conjectured lower bound (1).

Conjecture 1. Assume that under some adaptive sampling algorithm, $\{\omega(t)\}_{t\geq1}$ satisfies an LDP with rate function L. Then we have: for any non-empty subset S of $R^{K}$ and any subset W of $\Sigma$ ,

$$
\lim _ {t \to \infty} \frac {1}{t} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\boldsymbol {\mu}} (t) \in \operatorname{cl} (\mathcal {S}) , \boldsymbol {\omega} (t) \in W ]} = \inf _ {\boldsymbol {\omega} \in W} \max \left\{F _ {\mathcal {S}} (\boldsymbol {\omega}), L (\boldsymbol {\omega}) \right\}.
$$

For simplicity, we assume that $\Lambda = \{\pmb{\mu} \in \mathbb{R}^K : \mu_{1(\pmb{\mu})} > \mu_k, \forall k \neq 1(\pmb{\mu})\}$ and all the reward distributions are Gaussian. Introduce the notation:

$$
\Psi_ {\boldsymbol {\mu}} ^ {\star} = \max _ {\boldsymbol {\omega} \in \Sigma} \inf _ {\boldsymbol {\lambda} \in \mathrm{Alt} (\boldsymbol {\mu})} \Psi_ {\boldsymbol {\mu}} (\boldsymbol {\lambda}, \boldsymbol {\omega}),
$$

$$
\text { and } \omega^ {\star} (\boldsymbol {\mu}) = \underset {\boldsymbol {\omega} \in \Sigma} {\operatorname{argmax}} \inf _ {\boldsymbol {\lambda} \in \mathrm{Alt} (\boldsymbol {\mu})} \Psi_ {\boldsymbol {\mu}} (\boldsymbol {\lambda}, \boldsymbol {\omega}),
$$

where $\Psi_{\boldsymbol{\mu}}(\boldsymbol{\lambda},\boldsymbol{\omega})=\sum_{k=1}^{K}\omega_{k}d(\lambda_{k},\mu_{k})$ . Notice that the KL-divergence is symmetric in its arguments if the distributions are Gaussian. Hence the conjectured lower bound (1) is exactly the same as that in the fixed confidence setting. The solution $\boldsymbol{\omega}^{\star}(\boldsymbol{\mu})$ to the corresponding optimization problem is unique (see Theorem 5 in Garivier and Kaufmann (2016)).

We consider the set of algorithms returning the best empirical arm $\hat{\imath} = 1(\hat{\mu}(T))$ and with error probability matching the conjectured lower bound (1). If such an algorithm exists, under the assumption that Conjecture 1 is true, the rate function for the corresponding process $\{\omega(T)\}_{T\geq 1}$ must satisfy:

$$
\inf _ {\boldsymbol {\omega} \in \Sigma} \max \left\{\left(\inf _ {\boldsymbol {\lambda} \in \operatorname{Alt} (\boldsymbol {\mu})} \Psi_ {\boldsymbol {\mu}} (\boldsymbol {\lambda}, \boldsymbol {\omega})\right), L _ {\boldsymbol {\mu}} (\boldsymbol {\omega}) \right\} \geq \Psi_ {\boldsymbol {\mu}} ^ {\star}, \tag {105}
$$

where $L_{\mu}$ is the rate function of a complete LDP under $\pmb{\mu}$ for the process $\{\pmb{\omega}(T)\}_{T\geq 1}$ . Lemma 24 below shows that (105) implies the process $\{\pmb{\omega}(T)\}_{T\geq 1}$ convergences to the optimal allocation.

Lemma 24. For $\mu \in \Lambda$ , if there is a strategy satisfying (105), then $L_{\mu}(\omega) \geq \Psi_{\mu}^{\star}, \forall \omega \neq \omega^{\star}(\mu)$ and $L_{\mu}(\omega^{\star}(\mu)) = 0$ .

Proof. Assume that, on the contrary, there is $\omega' \neq \omega^{\star}(\mu)$ s.t. $L_{\mu}(\omega') < \Psi_{\mu}^{\star}$ . Together with $\inf_{\lambda \in \mathrm{Alt}(\mu)} \Psi_{\mu}(\lambda, \omega') < \Psi_{\mu}^{\star}$ , this implies that:

$$
\inf _ {\pmb {\omega} \in \Sigma} \max \left\{\big (\inf _ {\pmb {\lambda} \in \mathrm{Alt} (\pmb {\mu})} \Psi_ {\pmb {\mu}} (\pmb {\lambda}, \pmb {\omega}) \big), L _ {\pmb {\mu}} (\pmb {\omega}) \right\} \leq \max \left\{\inf _ {\pmb {\lambda} \in \mathrm{Alt} (\pmb {\mu})} \Psi_ {\pmb {\mu}} (\pmb {\lambda}, \pmb {\omega} ^ {\prime}), L _ {\pmb {\mu}} (\pmb {\omega} ^ {\prime}) \right\} <   \Psi_ {\pmb {\mu}} ^ {\star}.
$$

This contradicts the assumption, (105), so we have $L_{\boldsymbol{\mu}}(\boldsymbol{\omega}) \geq \Psi_{\boldsymbol{\mu}}^{\star}, \forall \boldsymbol{\omega} \neq \boldsymbol{\omega}^{\star}(\boldsymbol{\mu})$ . As for the optimal allocation, $\boldsymbol{\omega}^{\star}(\boldsymbol{\mu})$ , the fact $\mathbb{P}_{\boldsymbol{\mu}}[\boldsymbol{\omega}(T) \in \Sigma] = 1$ implies that

$$
0 = \lim _ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \boldsymbol {\omega} (T) \in \Sigma ]} \geq \inf _ {\boldsymbol {\omega} \in \Sigma} L _ {\boldsymbol {\mu}} (\boldsymbol {\omega}),
$$

where the last inequality stems from (4) in Definition 1. Since $L_{\mu}(\omega) \geq \Psi_{\mu}^{\star}, \forall \omega \neq \omega^{\star}(\mu)$ , we conclude that $L_{\mu}(\omega^{\star}(\mu)) = 0$ .

So far, we have investigated the consequence of matching the lower bound (1) on a single instance (a single parameter $\mu$ ). Of course, we wish to get an algorithm matching (1) for all instances. The following theorem shows that this is impossible even for two parameters.

Theorem 11. Consider $\pmb{\mu},\pmb{\pi}\in \Lambda$ s.t. $\omega^{\star}(\pmb {\mu})\neq \omega^{\star}(\pmb {\pi})$ and $\max_{k\in [K]}d(\pi_k,\mu_k) <   \Psi_{\pmb{\mu}}^{\star}$ , then there is no strategy satisfying (i) and (ii) simultaneously:

(i) (105) holds for $\pi$   
(ii) (105) holds for $\pmb{\mu}$

Proof. Assume that, on the contrary, there is such a strategy. By the assumption on $\pmb{\mu}$ and $\pmb{\pi}$ , there will be an open set $\mathcal{O} \subset \Sigma$ s.t. $\omega^{\star}(\pi) \in \mathcal{O}$ but $\omega^{\star}(\pmb{\mu}) \notin \mathcal{O}$ . On the one hand, $L_{\pi}(\omega^{\star}(\pi)) = 0$ by Lemma 24 and (i). Recalling the LDP lower bound (4) in Definition 1, $L_{\pi}(\omega^{\star}(\pi)) = 0$ directly implies that:

$$
\lim _ {T \to \infty} \mathbb {P} _ {\pi} [ \boldsymbol {\omega} (T) \in \mathcal {O} ] = 1. \tag {106}
$$

On the other hand, (ii) and Lemma 24 imply that $L_{\mu}(\omega) \geq \Psi_{\mu}^{\star}$ if $\omega \neq \omega^{\star}(\mu)$ , hence

$$
\varliminf_ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\mu} [ \boldsymbol {\omega} (T) \in \mathcal {O} ]} \geq \Psi_ {\mu} ^ {\star}. \tag {107}
$$

Now applying a change-of-measure argument (see Lemma 1 in Kaufmann et al. (2016) or equation (6) in Garivier et al. (2019)), one can derive

$$
\sum_ {k = 1} ^ {K} \mathbb {E} _ {\boldsymbol {\pi}} \left[ \omega_ {k} (T) \right] d (\pi_ {k}, \mu_ {k}) \geq \frac {1}{T} \mathrm{kl} (\mathbb {P} _ {\boldsymbol {\pi}} [ \boldsymbol {\omega} (T) \in \mathcal {O} ], \mathbb {P} _ {\boldsymbol {\mu}} [ \boldsymbol {\omega} (T) \in \mathcal {O} ]) \tag {108}
$$

Using the assumption that $\max_{k\in[K]}d(\pi_{k},\mu_{k})<\Psi_{\mu}^{\star}$ , the left-hand side of (108) is strictly smaller $\Psi_{\mu}^{\star}$ . However, by letting $T\to\infty$ on the r.h.s. of (108), (106) and (107) implies the limitinf is larger than $\Psi_{\mu}^{\star}$ . This is a contradiction. ☐

The consequence of Theorem 11 is that either our conjecture is true and in which case, for any algorithm there are two instances for which it cannot match the error lower bound (1) or the conjecture is not true (the bound provided in Theorem 1 is not tight). We finally note that recent results presented in Degenne (2023); Wang et al. (2023) suggest that indeed the lower bound (1) cannot be achieved.

# J Discussion on the conjectured lower bound (1)

In this section, we discuss two points: (i) (1) indeed corresponds to the conjectured lower bound proposed by Garivier and Kaufmann (2016), see their Section 7; (ii) however, as far as we know, there is no proof for (1), but one can derive a lower bound by inverting max and inf in (1).

(i) Without loss of generality, assume that $\mu$ is such that 1 is the best arm. We start from (1) and show that this is equivalent to Garivier-Kaufmann's formula. First, it can be easily checked that in (1), we can replace $\Sigma$ by $\Sigma_{>0} = \{\omega \in \Sigma : \omega_k > 0, \forall k \in [K]\}$ . Then, for any $\omega \in \Sigma_{>0}$ , we have

$$
\inf _ {\lambda \in \operatorname{Alt} (\boldsymbol {\mu})} \sum_ {k} \omega_ {k} d \left(\lambda_ {k}, \mu_ {k}\right) = \min _ {m \neq 1} \inf _ {\mu_ {m} <   x <   \mu_ {1}} \omega_ {1} d \left(x, \mu_ {1}\right) + \omega_ {m} d \left(x, \mu_ {m}\right).
$$

Indeed, we can decompose $\operatorname{Alt}(\mu)$ as $\cup_{m\neq 1}\{\lambda \in \Lambda :\lambda_m > \lambda_1\}$ , and thus, we have:

$$
\inf _ {\lambda \in \operatorname{Alt} (\boldsymbol {\mu})} \sum_ {k} \omega_ {k} d \left(\lambda_ {k}, \mu_ {k}\right) = \min _ {m \neq 1} \inf _ {\lambda_ {m} > \lambda_ {1}} \sum_ {k} \omega_ {k} d \left(\lambda_ {k}, \mu_ {k}\right).
$$

We conclude by observing that

$$
\inf _ {\lambda_ {m} > \lambda_ {1}} \omega_ {1} d (\lambda_ {1}, \mu_ {1}) + \omega_ {m} d (\lambda_ {m}, \mu_ {m}) = \inf _ {\mu_ {m} <   x <   \mu_ {1}} \omega_ {1} d (x, \mu_ {1}) + \omega_ {k} d (x, \mu_ {m}),
$$

which holds for all families of distributions such that $x \mapsto d(x, y)$ is monotonic (decreasing before y and increasing after y) – this holds for Bernoulli, Gaussian, etc.

(ii) Consider a consistent algorithm, and denote by $\omega_{k}(\pmb{\lambda})$ the expected proportion of rounds where the algorithm selects arm $k$ under the probability $\mathbb{P}_{\pmb{\lambda}}$ . Using the classical change-of-measure arguments, we get:

$$
\operatorname * {l i m s u p} _ {T \to \infty} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {\imath} \neq 1 ]} \leq T \sum_ {k} \omega_ {k} (\boldsymbol {\lambda}) d (\lambda_ {k}, \mu_ {k}) \leq T \max _ {\boldsymbol {\omega} \in \Sigma} \sum_ {k} \omega_ {k} d (\lambda_ {k}, \mu_ {k}).
$$

We can only deduce that:

$$
\operatorname * {l i m s u p} _ {T \to \infty} \frac {1}{T} \log \frac {1}{\mathbb {P} _ {\boldsymbol {\mu}} [ \hat {i} \neq 1 ]} \leq \inf _ {\boldsymbol {\lambda} \in \mathrm{Alt} (\boldsymbol {\mu})} \max _ {\boldsymbol {\omega} \in \Sigma} \sum_ {k} \omega_ {k} d (\lambda_ {k}, \mu_ {k}).
$$

One cannot directly apply Sion's minimax theorem to derive (1) (as $\operatorname{Alt}(\boldsymbol{\mu})$ is not a convex domain).

# K Numerical experiments

We consider various problem instances to numerically evaluate the performance of CR. In these instances, we vary the number of arms from 5 to 55; we use Bernoulli distributed rewards, and vary the shape of the arm-to-reward mapping. For each instance, we compare CR-C and CR-A to SR (Audibert et al., 2010), SH (Karnin et al., 2013), and UGapE (Gabillon et al., 2012). As discussed in Section 2, UGapE requires prior knowledge about a parameter depending on the underlying problem. We hence implement its heuristic version which estimates the parameter on the fly, such modification was suggested in previous works e.g. (Audibert et al., 2010; Karnin et al., 2013). We implement all algorithms in Julia 1.7.3 and run all experiments on a machine with Apple M1 with 16 GB RAM. $^{3}$ The error probabilities averaged over 40,000 independent runs. In all experiments, we set $\theta_0 = 10^{-5}$ for CR.

We vary the shape of the arm-to-reward function and consider one shape in each of the subsections below. We present the error probability of all algorithms in tables and figures. In the latter, the error probability is presented using the log scale, which sometimes makes the curves for some algorithms close to each other. In the tables, we present the error probability for a few budgets only, and there, we can see a clearer separation between the performance of the various algorithms.

Observe that our algorithms, CR, perform better than the other algorithms for all arm-to-reward function shapes.

# K.1 One group of suboptimal arms

This instance is considered by (Karnin et al., 2013): $\mu_{1}=0.5$ and $\mu_{k}=0.45$ for all $k\geq2$ . We can see that the performances of SR, CR-C, and CR-A are significantly better than UGapE and SH.

![](images/411b167b61afc84b3aa434968fa60a4285b1c039d88eb005f974e4602b87294d.jpg)

<details>
<summary>line</summary>

| x  | y     |
|----|-------|
| 0  | 0.50  |
| 10 | 0.45  |
| 20 | 0.45  |
| 30 | 0.45  |
| 40 | 0.45  |
</details>

Figure 1: (One group of suboptimal arms) $\mu$ with K = 40.

Table 2: (One group of suboptimal arms) error probabilities (in %). 

<table><tr><td colspan="3">K = 10</td><td>T = 6,400</td><td>T = 7,200</td><td>T = 8,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>34.66</td><td>33.22</td><td>32.15</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>19.64</td><td>16.85</td><td>14.46</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>7.86</td><td>5.86</td><td>4.29</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>7.29</td><td>5.47</td><td>4.17</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>7.37</td><td>5.52</td><td>4.07</td></tr><tr><td colspan="3">K = 20</td><td>T = 12,000</td><td>T = 14,000</td><td>T = 16,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>42.77</td><td>39.93</td><td>38.62</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>25.17</td><td>20.86</td><td>17.94</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>9.43</td><td>6.59</td><td>4.35</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>8.39</td><td>5.92</td><td>4.08</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>8.80</td><td>6.16</td><td>4.42</td></tr><tr><td colspan="3">K = 40</td><td>T = 30,000</td><td>T = 35,000</td><td>T = 40,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>42.84</td><td>40.46</td><td>38.11</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>23.26</td><td>19.24</td><td>16.12</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>6.51</td><td>4.48</td><td>3.14</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>5.96</td><td>3.99</td><td>2.89</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>6.56</td><td>4.27</td><td>3.06</td></tr></table>

![](images/5e2021d25938ffecd4bf7346d6dfd014a334316b33293378e477a4f802745533.jpg)

<details>
<summary>line</summary>

| budget | UG       | SH       | SR       | CR-C     | CR-A     |
| ------ | -------- | -------- | -------- | -------- | -------- |
| 4000   | 1e-0.5   | 1e-0.5   | 1e-0.7   | 1e-0.7   | 1e-0.7   |
| 8000   | 1e-0.6   | 1e-0.6   | 1e-0.9   | 1e-0.9   | 1e-0.9   |
</details>

(a) K = 10

![](images/d8f991323aa09c72e9f80fb2a30c25d21e0c74569454756da4ccbd490f9a604b.jpg)

<details>
<summary>line</summary>

| budget | UG       | SH       | SR       | CR-C     | CR-A     |
| ------ | -------- | -------- | -------- | -------- | -------- |
| 7500   | 1.0e-04  | 1.0e-05  | 1.0e-06  | 1.0e-07  | 1.0e-08  |
| 10000  | 1.0e-04  | 1.0e-06  | 1.0e-07  | 1.0e-08  | 1.0e-09  |
| 12500  | 1.0e-04  | 1.0e-07  | 1.0e-08  | 1.0e-09  | 1.0e-10  |
| 15000  | 1.0e-04  | 1.0e-08  | 1.0e-09  | 1.0e-10  | 1.0e-11  |
</details>

(b) K = 20

![](images/050b2a63fed912f1d0c01d3e374159ad6d452b8447c59596657d2604eea4a0f3.jpg)

<details>
<summary>line</summary>

| budget     | UG       | SH       | SR       | CR-C     | CR-A     |
| ---------- | -------- | -------- | -------- | -------- | -------- |
| 2.00×10⁴   | 1.0e-0.5 | 1.0e-0.5 | 1.0e-1.0 | 1.0e-1.0 | 1.0e-1.0 |
| 2.50×10⁴   | 1.0e-0.6 | 1.0e-0.6 | 1.0e-1.1 | 1.0e-1.1 | 1.0e-1.1 |
| 3.00×10⁴   | 1.0e-0.7 | 1.0e-0.7 | 1.0e-1.2 | 1.0e-1.2 | 1.0e-1.2 |
| 3.50×10⁴   | 1.0e-0.8 | 1.0e-0.8 | 1.0e-1.3 | 1.0e-1.3 | 1.0e-1.3 |
| 4.00×10⁴   | 1.0e-09 | 1.0e-09 | 1.0e-1.4 | 1.0e-1.4 | 1.0e-1.4 |
</details>

(c) K = 40   
Figure 2: (One group of suboptimal arms) error probabilities averaged over 40,000 independent runs.

# K.2 Two groups of suboptimal arms

In this instance, we set $\mu_{1}=0.5$ , $\mu_{k}=0.45$ for $k=2,\cdots,\lfloor\frac{K-1}{2}\rfloor$ , and $\mu_{k}=0.4$ for $k=\lfloor\frac{K-1}{2}\rfloor+1,\cdots,K$ . Compared to K.1, where CR-C is always the best, CR-A becomes relatively better here.

![](images/b2e1d281d55d041ff0842792b3ab893844e8bc30c59ccbe22221562532fe036c.jpg)

<details>
<summary>line</summary>

| x  | y     |
|----|-------|
| 0  | 0.500 |
| 2  | 0.450 |
| 20 | 0.450 |
| 21 | 0.400 |
| 40 | 0.400 |
</details>

Figure 3: (Two groups of suboptimal arms) $\mu$ with K = 40.

Table 3: (Two groups of suboptimal arms) error probabilities (in %). 

<table><tr><td colspan="3">K = 10</td><td>T = 5,600</td><td>T = 6,800</td><td>T = 8,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>25.50</td><td>23.02</td><td>20.97</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>11.12</td><td>7.71</td><td>5.35</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>4.05</td><td>2.30</td><td>1.20</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>3.68</td><td>2.05</td><td>1.13</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>3.44</td><td>1.88</td><td>0.99</td></tr><tr><td colspan="3">K = 20</td><td>T = 4,000</td><td>T = 7,000</td><td>T = 10,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>48.49</td><td>40.68</td><td>36.12</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>42.62</td><td>25.69</td><td>16.15</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>28.26</td><td>12.03</td><td>5.03</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>26.54</td><td>10.62</td><td>4.52</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>26.00</td><td>10.61</td><td>4.27</td></tr><tr><td colspan="3">K = 40</td><td>T = 15,000</td><td>T = 20,000</td><td>T = 25,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>46.04</td><td>41.80</td><td>38.30</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>27.24</td><td>18.98</td><td>13.82</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>9.97</td><td>5.03</td><td>2.43</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>9.12</td><td>4.49</td><td>2.24</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>9.26</td><td>4.78</td><td>2.38</td></tr></table>

![](images/5918cc09f795056cf366389693b3c09cde2883924a654709e1fee2dfa7a98bc9.jpg)

<details>
<summary>line</summary>

| budget | UG     | SH     | SR     | CR-C   | CR-A   |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 2000   | 0.3    | 0.25   | 0.2    | 0.18   | 0.15   |
| 4000   | 0.25   | 0.15   | 0.1    | 0.08   | 0.06   |
| 6000   | 0.2    | 0.1    | 0.05   | 0.04   | 0.03   |
| 8000   | 0.15   | 0.05   | 0.02   | 0.015  | 0.01   |
</details>

(a) K = 10

![](images/fe216a4a35e53a3e832ad7d264b8678876e5e9174d65df71300af41163d0f4b9.jpg)

<details>
<summary>line</summary>

| budget     | UG       | SH       | SR       | CR-C     | CR-A     |
| ---------- | -------- | -------- | -------- | -------- | -------- |
| 6.00×10³   | 0.5      | 0.4      | 0.3      | 0.25     | 0.2      |
| 9.00×10³   | 0.4      | 0.3      | 0.2      | 0.15     | 0.1      |
| 1.20×10⁴   | 0.3      | 0.2      | 0.1      | 0.08     | 0.05     |
| 1.50×10⁴   | 0.25     | 0.15     | 0.05     | 0.04     | 0.03     |
| 1.80×10⁴   | 0.2      | 0.1      | 0.03     | 0.02     | 0.01     |
</details>

(b) K = 20

![](images/416f9ee81aaa8244e3cf69b289b6a7188a3f081e9f70e136830e8e0055738ba1.jpg)

<details>
<summary>line</summary>

| budget     | UG       | SH       | SR       | CR-C     | CR-A     |
| ---------- | -------- | -------- | -------- | -------- | -------- |
| 5.00×10³   | 1.00E-04 | 1.00E-04 | 1.00E-04 | 1.00E-04 | 1.00E-04 |
| 1.00×10⁴   | 1.00E-05 | 1.00E-05 | 1.00E-05 | 1.00E-05 | 1.00E-05 |
| 1.50×10⁴   | 1.00E-06 | 1.00E-06 | 1.00E-06 | 1.00E-06 | 1.00E-06 |
| 2.00×10⁴   | 1.00E-07 | 1.00E-07 | 1.00E-07 | 1.00E-07 | 1.00E-07 |
| 2.50×10⁴   | 1.00E-08 | 1.00E-08 | 1.00E-08 | 1.00E-08 | 1.00E-08 |
</details>

(c) K = 40   
Figure 4: (Two groups of suboptimal arms) error probabilities averaged over 40,000 independent runs.

# K.3 Linear arm-to-reward function

In this instance, we set $\mu_{k}=\frac{3}{4}-\frac{k-1}{2K}$ for $k=1,\cdots,K$ . Here CR-A does the best.

![](images/a99eaad5daf89f7ba0d2f7520d6aa1dda444f8fd3fc5038498623e5b62bbd7c8.jpg)

<details>
<summary>line</summary>

| x  | y    |
|----|------|
| 0  | 0.75 |
| 10 | 0.65 |
| 20 | 0.55 |
| 30 | 0.45 |
| 40 | 0.35 |
</details>

Figure 5: (Linear arm-to-reward function) $\mu$ with K = 40.

Table 4: (Linear arm-to-reward function) error probability (in %). 

<table><tr><td>K = 10</td><td></td><td></td><td>T = 3,200</td><td>T = 3,600</td><td>T = 4,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>4.96</td><td>4.22</td><td>3.24</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>4.97</td><td>4.21</td><td>3.49</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>2.09</td><td>1.53</td><td>1.03</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>1.59</td><td>1.04</td><td>0.83</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>1.20</td><td>0.81</td><td>0.55</td></tr><tr><td>K = 20</td><td></td><td></td><td>T = 6,000</td><td>T = 8,000</td><td>T = 10,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>15.49</td><td>11.20</td><td>8.78</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>16.00</td><td>12.66</td><td>9.70</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>10.76</td><td>7.57</td><td>5.24</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>10.03</td><td>6.96</td><td>4.68</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>8.78</td><td>5.72</td><td>3.73</td></tr><tr><td>K = 40</td><td></td><td></td><td>T = 15,000</td><td>T = 20,000</td><td>T = 25,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>25.09</td><td>20.21</td><td>16.44</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>25.70</td><td>21.19</td><td>17.92</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>20.29</td><td>15.86</td><td>13.07</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>19.93</td><td>15.56</td><td>12.30</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>17.99</td><td>13.74</td><td>10.67</td></tr></table>

![](images/c848c00a417c886a598899f59d5e6bcef1008b6c96e2adfdda996e138a3e7a52.jpg)

<details>
<summary>line</summary>

| budget | UG       | SH       | SR       | CR-C     | CR-A     |
| ------ | -------- | -------- | -------- | -------- | -------- |
| 2000   | 1e-1.2   | 1e-1.2   | 1e-1.5   | 1e-1.6   | 1e-1.7   |
| 2500   | 1e-1.3   | 1e-1.3   | 1e-1.6   | 1e-1.7   | 1e-1.8   |
| 3000   | 1e-1.4   | 1e-1.4   | 1e-1.7   | 1e-1.8   | 1e-1.9   |
| 3500   | 1e-1.5   | 1e-1.5   | 1e-1.8   | 1e-1.9   | 1e-2.0   |
| 4000   | 1e-1.6   | 1e-1.6   | 1e-1.9   | 1e-2.0   | 1e-2.1   |
</details>

(a) K = 10

![](images/cbd048315366efbcd7e03db5d14b5364183375c5f13a8d157685a9306de54c7b.jpg)

<details>
<summary>line</summary>

| budget | UG       | SH       | SR       | CR-C     | CR-A     |
| ------ | -------- | -------- | -------- | -------- | -------- |
| 2000   | 1e-0.6   | 1e-0.6   | 1e-0.6   | 1e-0.6   | 1e-0.6   |
| 4000   | 1e-0.8   | 1e-0.7   | 1e-0.8   | 1e-0.8   | 1e-0.9   |
| 6000   | 1e-0.9   | 1e-08    | 1e-09    | 1e-09    | 1e-1.1   |
| 8000   | 1e-1.0   | 1e-09    | 1e-10    | 1e-10    | 1e-1.3   |
| 10000  | 1e-1.1   | 1e-10    | 1e-11    | 1e-11    | 1e-1.5   |
</details>

(b) $K = 20$

![](images/f878ea3a80cac8e5bd48437eb28a0b57889446f5bf26be723f38da53f9839848.jpg)

<details>
<summary>line</summary>

| budget | UG       | SH       | SR       | CR-C     | CR-A     |
| ------ | -------- | -------- | -------- | -------- | -------- |
| 16000  | 1e-0.6   | 1e-0.6   | 1e-0.7   | 1e-0.75  | 1e-0.85  |
| 18000  | 1e-0.65  | 1e-0.65  | 1e-0.75  | 1e-0.8   | 1e-0.9   |
| 20000  | 1e-0.7   | 1e-0.7   | 1e-0.8   | 1e-0.85  | 1e-0.95  |
| 22000  | 1e-0.75  | 1e-0.75  | 1e-0.85  | 1e-0.9   | 1e-1    |
| 24000  | 1e-0.8   | 1e-0.8   | 1e-0.9   | 1e-0.95  | 1e-1.05  |
</details>

(c) K = 40   
Figure 6: (Linear arm-to-reward function) error probabilities averaged over 40,000 independent runs.

# K.4 Concave arm-to-reward function

In this instance, we set $\mu_{1}=\sin(\frac{(K-1)\pi}{2K})$ and $\mu_{k}=\sin(\frac{9\pi(K-k+1)}{20K})$ for $k=2,\cdots,K$ . CR-A does the best in this instance all the time.

![](images/4cbf0112f9513831dc4713437830c45ca419ea39b50894e5c71fcb4db1260722.jpg)

<details>
<summary>line</summary>

| x  | y    |
|----|------|
| 0  | 1.0  |
| 10 | 0.9  |
| 20 | 0.7  |
| 30 | 0.4  |
| 40 | 0.0  |
</details>

Figure 7: (Concave arm-to-reward function) $\mu$ with K = 40.

Table 5: (Concave arm-to-reward function) error probability (in %). 

<table><tr><td>K = 10</td><td></td><td></td><td>T = 900</td><td>T = 1,400</td><td>T = 1,900</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>2.23</td><td>1.36</td><td>0.94</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>4.68</td><td>2.15</td><td>0.89</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>1.85</td><td>0.54</td><td>0.20</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>1.24</td><td>0.35</td><td>0.10</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>0.94</td><td>0.21</td><td>0.04</td></tr><tr><td>K = 20</td><td></td><td></td><td>T = 900</td><td>T = 1,400</td><td>T = 1,900</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>2.44</td><td>1.85</td><td>1.59</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>6.62</td><td>2.62</td><td>1.28</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>2.81</td><td>0.86</td><td>0.31</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>1.87</td><td>0.47</td><td>0.14</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>1.36</td><td>0.36</td><td>0.09</td></tr><tr><td>K = 40</td><td></td><td></td><td>T = 2,400</td><td>T = 2,800</td><td>T = 3,200</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>1.03</td><td>0.98</td><td>0.94</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>1.26</td><td>0.60</td><td>0.35</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>0.23</td><td>0.10</td><td>0.02</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>0.18</td><td>0.08</td><td>0.04</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>0.08</td><td>0.03</td><td>0.02</td></tr></table>

![](images/39c3e287ffe6ae935e8254ccfe25d61e6dc8c16b6bcae820545bf9d09cde7737.jpg)

<details>
<summary>line</summary>

| budget | UG     | SH     | SR     | CR-C   | CR-A   |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 600    | 0.05   | 0.07   | 0.06   | 0.04   | 0.03   |
| 900    | 0.02   | 0.01   | 0.015  | 0.01   | 0.005  |
| 1200   | 0.01   | 0.005  | 0.008  | 0.005  | 0.002  |
| 1500   | 0.008  | 0.003  | 0.005  | 0.003  | 0.001  |
| 1800   | 0.005  | 0.001  | 0.003  | 0.001  | 0.0005 |
</details>

(a) K = 10

![](images/ef2f88b97985589a583d216c6e929f64bdc2de31ee49bc800ae3ad15aae50454.jpg)

<details>
<summary>line</summary>

| budget | UG     | SH     | SR     | CR-C   | CR-A   |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 600    | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    |
| 900    | 0.02   | 0.02   | 0.02   | 0.02   | 0.02   |
| 1200   | 0.015  | 0.015  | 0.015  | 0.015  | 0.015  |
| 1500   | 0.012  | 0.012  | 0.012  | 0.012  | 0.012  |
| 1800   | 0.01   | 0.01   | 0.01   | 0.01   | 0.01   |
</details>

(b) $K = 20$

![](images/51376a5ac069e968e3ca9010ee8a778ab75b366e35fdb38ba43bb13a84dab6e0.jpg)

<details>
<summary>line</summary>

| budget | UG       | SH       | SR       | CR-C     | CR-A     |
| ------ | -------- | -------- | -------- | -------- | -------- |
| 2000   | 0.01     | 0.01     | 0.005    | 0.005    | 0.002    |
| 2500   | 0.01     | 0.005    | 0.003    | 0.003    | 0.001    |
| 3000   | 0.01     | 0.003    | 0.002    | 0.002    | 0.0005   |
| 3500   | 0.01     | 0.001    | 0.001    | 0.001    | 0.0001   |
</details>

(c) K = 40   
Figure 8: (Concave arm-to-reward function) error probabilities averaged over 40,000 independent runs.

# K.5 Convex arm-to-reward function

In this instance, we set $\mu_{k}=\frac{3}{10(k+1)}$ for $k=1,\cdots,K$ . Although SR sometimes does better than CR-C, CR-C becomes better than SR when there is more budget given. This confirms our theoretical analysis for CR-C (see Theorem 7).

![](images/213d20bc24c464e3a297db607935990f9f8e25c10dec9d0483905eeeb2b566a6.jpg)

<details>
<summary>line</summary>

| x  | y     |
|----|-------|
| 0  | 0.15  |
| 5  | 0.08  |
| 10 | 0.04  |
| 15 | 0.025 |
| 20 | 0.015 |
| 25 | 0.01  |
| 30 | 0.008 |
| 35 | 0.006 |
| 40 | 0.005 |
</details>

Figure 9: (Convex arm-to-reward function) $\mu$ with $K = 40$ .

Table 6: (Convex arm-to-reward function) error probability (in %). 

<table><tr><td>K = 10</td><td></td><td></td><td>T = 1,500</td><td>T = 2,000</td><td>T = 2,500</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>13.48</td><td>10.08</td><td>7.77</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>10.20</td><td>5.94</td><td>3.84</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>3.15</td><td>1.45</td><td>0.83</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>3.12</td><td>1.47</td><td>0.65</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>2.99</td><td>1.27</td><td>0.62</td></tr><tr><td>K = 20</td><td></td><td></td><td>T = 3,000</td><td>T = 3,500</td><td>T = 4,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>10.67</td><td>8.95</td><td>7.76</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>6.60</td><td>4.56</td><td>2.78</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>0.96</td><td>0.55</td><td>0.29</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>0.93</td><td>0.56</td><td>0.28</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>0.79</td><td>0.39</td><td>0.21</td></tr><tr><td>K = 40</td><td></td><td></td><td>T = 4,400</td><td>T = 5,200</td><td>T = 6,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>12.16</td><td>9.95</td><td>8.28</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>7.51</td><td>4.76</td><td>2.96</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>1.21</td><td>0.58</td><td>0.30</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>1.28</td><td>0.49</td><td>0.25</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>1.34</td><td>0.60</td><td>0.29</td></tr></table>

![](images/c2306d80d57c4955a534a047940d70c01df13a3397cb2c8a94fcc7466c227fdb.jpg)

<details>
<summary>line</summary>

| budget | UG     | SH     | SR     | CR-C   | CR-A   |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 500    | 0.3    | 0.4    | 0.3    | 0.3    | 0.3    |
| 1000   | 0.2    | 0.25   | 0.2    | 0.2    | 0.2    |
| 1500   | 0.15   | 0.18   | 0.15   | 0.15   | 0.15   |
| 2000   | 0.1    | 0.12   | 0.1    | 0.1    | 0.1    |
| 2500   | 0.08   | 0.09   | 0.08   | 0.08   | 0.08   |
</details>

(a) K = 10

![](images/81afc79b5491aea53ad3dba725dee6ab03ce73082601865f13a45f138d30d765.jpg)

<details>
<summary>line</summary>

| budget | UG     | SH     | SR     | CR-C   | CR-A   |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 1500   | 0.2    | 0.2    | 0.2    | 0.2    | 0.2    |
| 2000   | 0.15   | 0.15   | 0.15   | 0.15   | 0.15   |
| 2500   | 0.1    | 0.1    | 0.1    | 0.1    | 0.1    |
| 3000   | 0.08   | 0.08   | 0.08   | 0.08   | 0.08   |
| 3500   | 0.06   | 0.06   | 0.06   | 0.06   | 0.06   |
| 4000   | 0.04   | 0.04   | 0.04   | 0.04   | 0.04   |
</details>

(b) $K = 20$

![](images/35ba2a0a044d2d19eaec068078e9f554627ad510c2f5a568b03410664ae21a04.jpg)

<details>
<summary>line</summary>

| budget | UG     | SH     | SR     | CR-C   | CR-A   |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 2000   | 0.2    | 0.3    | 0.15   | 0.12   | 0.1    |
| 3000   | 0.15   | 0.2    | 0.08   | 0.07   | 0.06   |
| 4000   | 0.12   | 0.15   | 0.05   | 0.04   | 0.03   |
| 5000   | 0.1    | 0.1    | 0.03   | 0.02   | 0.015  |
| 6000   | 0.08   | 0.05   | 0.015  | 0.01   | 0.008  |
</details>

(c) K = 40   
Figure 10: (Convex arm-to-reward function) error probabilities averaged over 40,000 independent runs.

# K.6 Stair arm-to-reward function

In this instance, we consider $M \in \{5, 6, 10\}$ and a $M(M + 1)/2$ -dimensional vector $\mu$ . For each M, we define $\mu$ as: for all positive integers m smaller than M, there are m arms on the same level with value, $\frac{3}{4} \cdot 3^{-\frac{m}{M}}$ . For example, we plot the values for M = 10 (hence K = 55) in Figure 11. One can see in this instance, our algorithms are by far stronger than the others.

![](images/feaf90478c47f01498bff8adcfbc9275bf4feb4c74f5a974112a5bdda65f619b.jpg)

<details>
<summary>line</summary>

| x | y |
|---|---|
| 0 | 0.68 |
| 2 | 0.60 |
| 4 | 0.54 |
| 6 | 0.49 |
| 8 | 0.43 |
| 10 | 0.43 |
| 12 | 0.39 |
| 14 | 0.35 |
| 16 | 0.35 |
| 18 | 0.31 |
| 20 | 0.31 |
| 22 | 0.28 |
| 24 | 0.28 |
| 26 | 0.27 |
| 28 | 0.26 |
| 30 | 0.26 |
| 32 | 0.26 |
| 34 | 0.26 |
| 36 | 0.26 |
| 38 | 0.26 |
| 40 | 0.26 |
| 42 | 0.26 |
| 44 | 0.26 |
| 46 | 0.26 |
| 48 | 0.26 |
| 50 | 0.26 |
| 52 | 0.26 |
The chart displays a stepwise decreasing trend with discrete drops at each step point. The x-axis ranges from 0 to over 50, and the y-axis represents the corresponding values of the step function. There is no label for the data series.
</details>

Figure 11: (Stair arm-to-reward function) $\pmb{\mu}$ with $K = 55$ .

Table 7: (Stair arm-to-reward function) error probability (in %). 

<table><tr><td>K = 15</td><td></td><td></td><td>T = 1,600</td><td>T = 2,000</td><td>T = 2,400</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>12.00</td><td>9.89</td><td>8.47</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>2.18</td><td>1.07</td><td>0.52</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>0.43</td><td>0.19</td><td>0.06</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>0.36</td><td>0.10</td><td>0.02</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>0.20</td><td>0.07</td><td>0.04</td></tr><tr><td>K = 21</td><td></td><td></td><td>T = 1,500</td><td>T = 2,000</td><td>T = 2,500</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>17.44</td><td>13.97</td><td>12.08</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>7.78</td><td>3.75</td><td>1.78</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>2.27</td><td>0.92</td><td>0.32</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>1.68</td><td>0.55</td><td>0.24</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>1.14</td><td>0.35</td><td>0.09</td></tr><tr><td>K = 55</td><td></td><td></td><td>T = 3,000</td><td>T = 4,000</td><td>T = 5,000</td></tr><tr><td></td><td>UGapE</td><td>(Gabillon et al., 2012)</td><td>24.74</td><td>21.29</td><td>18.91</td></tr><tr><td></td><td>SH</td><td>Karnin et al. (2013)</td><td>13.09</td><td>7.76</td><td>4.56</td></tr><tr><td></td><td>SR</td><td>Audibert et al. (2010)</td><td>5.55</td><td>2.80</td><td>1.26</td></tr><tr><td></td><td>CR-C</td><td>(this paper)</td><td>7.10</td><td>2.58</td><td>1.05</td></tr><tr><td></td><td>CR-A</td><td>(this paper)</td><td>4.70</td><td>1.62</td><td>0.57</td></tr></table>

![](images/49c45570a07d5b14778f4a5ac0c15322da545be6cc3eb1d9307e50c3ebf9de06.jpg)

<details>
<summary>line</summary>

| budget | UG     | SH     | SR     | CR-C   | CR-A   |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 1000   | 0.1    | 0.1    | 0.05   | 0.03   | 0.02   |
| 1250   | 0.08   | 0.07   | 0.04   | 0.025  | 0.015  |
| 1500   | 0.06   | 0.05   | 0.03   | 0.02   | 0.01   |
| 1750   | 0.05   | 0.04   | 0.025  | 0.015  | 0.008  |
| 2000   | 0.04   | 0.03   | 0.02   | 0.01   | 0.005  |
| 2250   | 0.03   | 0.02   | 0.015  | 0.005  | 0.003  |
| 2500   | 0.02   | 0.015  | 0.01   | 0.003  | 0.002  |
</details>

(a) K = 15

![](images/edc13bebfea523ff5bbcb62f6fe3dd6efd59db14d33d2b5b3a314a39dac7cc1c.jpg)

<details>
<summary>line</summary>

| budget | UG     | SH     | SR     | CR-C   | CR-A   |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 1200   | 0.1    | 0.1    | 0.05   | 0.03   | 0.02   |
| 1500   | 0.08   | 0.07   | 0.03   | 0.02   | 0.01   |
| 1800   | 0.06   | 0.05   | 0.02   | 0.01   | 0.005  |
| 2100   | 0.05   | 0.04   | 0.01   | 0.005  | 0.002  |
| 2400   | 0.04   | 0.03   | 0.005  | 0.002  | 0.001  |
</details>

(b) $K = 21$

![](images/4543b3e043569fd63fe3cb03aba5bc120e3512b27dc5b5361401de8fa5b2f82a.jpg)

<details>
<summary>line</summary>

| budget | UG     | SH     | SR     | CR-C   | CR-A   |
| ------ | ------ | ------ | ------ | ------ | ------ |
| 1000   | 0.3    | 0.3    | 0.2    | 0.3    | 0.3    |
| 2000   | 0.2    | 0.2    | 0.1    | 0.2    | 0.1    |
| 3000   | 0.15   | 0.15   | 0.05   | 0.1    | 0.05   |
| 4000   | 0.1    | 0.1    | 0.02   | 0.05   | 0.02   |
| 5000   | 0.08   | 0.05   | 0.01   | 0.02   | 0.01   |
</details>

(c) K = 55   
Figure 12: (Stair arm-to-reward function) error probabilities averaged over 40,000 independent runs.