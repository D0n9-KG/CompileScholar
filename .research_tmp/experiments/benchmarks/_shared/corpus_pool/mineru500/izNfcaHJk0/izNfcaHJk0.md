# Privacy Amplification via Compression: Achieving the Optimal Privacy-Accuracy-Communication Trade-off in Distributed Mean Estimation

Wei-Ning Chen $^{†}$

Dan Song $^{\dagger}$

Ayfer Özgür $^{†}$

Peter Kairouz $^{\ddagger}$

Stanford University $^{\dagger}$ , Google Research $^{\ddagger}$

# Abstract

Privacy and communication constraints are two major bottlenecks in federated learning (FL) and analytics (FA). We study the optimal accuracy of mean and frequency estimation (canonical models for FL and FA respectively) under joint communication and $(\varepsilon,\delta)$ -differential privacy (DP) constraints. We consider both the central and the multi-message shuffling DP models. We show that in order to achieve the optimal $\ell_{2}$ error under $(\varepsilon,\delta)$ -DP, it is sufficient for each client to send $\Theta\left(n\min\left(\varepsilon,\varepsilon^{2}\right)\right)$ bits for FL and $\Theta\left(\log\left(n\min\left(\varepsilon,\varepsilon^{2}\right)\right)\right)$ bits for FA to the server, where n is the number of participating clients. Without compression, each client needs $O(d)$ bits and $O(\log d)$ bits for the mean and frequency estimation problems respectively (where d corresponds to the number of trainable parameters in FL or the domain size in FA), meaning that we can get significant savings in the regime $n\min\left(\varepsilon,\varepsilon^{2}\right)=o(d)$ , which is often the relevant regime in practice.

We propose two different ways to leverage compression for privacy amplification and achieve the optimal privacy-communication-accuracy trade-off. In both cases, each client communicates only partial information about its sample and we show that privacy is amplified by randomly selecting the part contributed by each client. In the first method, the random selection is revealed to the server, which results in a central DP guarantee with optimal privacy-communication-accuracy trade-off. In the second method, the random data parts at each client are privatized locally and anonymized by a secure shuffler, eliminating the need for a trusted server. This results in a multi-message shuffling scheme with the same optimal trade-off. As a result, our paper establishes the optimal three-way trade-off between privacy, communication, and accuracy for both the central DP and multi-message shuffling frameworks. $^{1}$

# 1 Introduction

In the basic setting of federated learning (FL) [67, 63, 60] and analytics (FA), a server wants to execute a specific learning or analytics task on raw data that is kept on clients' devices. Consider, for example, model updates in FL or histogram estimation in FA, both of which can be modeled as a distributed mean estimation problem. This problem is solved by having the clients communicate targeted messages to the server. The privacy of the users' data is ensured (in terms of explicit differential privacy (DP) [38] guarantees) by having the server inject noise into the computed mean before releasing it to the next module (e.g., the server can compute the average model update and corrupt it with the addition of noise). This is called the trusted server or central DP model, as it entrusts the central server with privatization and is one of the most common ways in which federated learning and analytics are implemented today $^{2}$ .

In this paper, we start by asking the following question about distributed mean estimation in this central DP setting: given that the server is required to release only a noisy, or equivalently approximate, version of the mean, can the clients communicate “less information” to the server? More precisely, given a desired privacy level, can we reduce the communication load of the network without sacrificing accuracy, i.e., maintaining the

same (order-wise optimal) accuracy for the desired privacy level? In recent years, there has been significant interest in the central DP model $[1]$ as well as communication efficiency and privacy for FL and FA under different models, including local DP $[77, 62, 57, 79, 17, 6, 16, 32]$ , shuffle $[40, 43]$ and distributed DP $[9, 58, 8, 33, 34]$ ; however, this basic question appears to remain unanswered.

The communication load at the clients can be reduced by having the clients communicate partial information about their samples to the server. For example, in the case of model updates, each client can update only a subset of the model coefficients. In the case of histogram estimation, information about a client's sample can be “split” into multiple parts, and the client can communicate only one part. However, this means that the server collects less information, or effectively fewer samples to estimate the target quantity. For example, in the aforementioned case of model updates, each model coefficient is updated only by a subset of the clients. A quick calculation reveals that this increases the sensitivity of the estimate to each user's sample and therefore requires the addition of larger noise at the server to achieve the same privacy level, which leads to lower accuracy.

We circumvent this challenge with a simple but insightful observation: when each client communicates only partial information about its sample, we can amplify privacy by randomly selecting the part contributed by each client. A downstream module which has only access to the final estimate revealed by the server does not know which part was contributed by which client, which leads to privacy amplification. Privacy amplification by subsampling has been studied in the prior literature $[65, 14]$ but usually only in the case of privacy amplification when the server selects a random subset of the clients (from a larger pool of available clients). In our case, the randomness is incorporated in the compression scheme of each client and it relates to the piece of information communicated by each client and not to the choice of the participating clients. We call this gain privacy amplification via compression and use it to establish the optimal communication-privacy-accuracy trade-off for the central DP model. Note that this same type of gain cannot be leveraged in the local DP model where the server is untrusted and knows the message and the identity of each client. Indeed, in the local DP model the privacy-accuracy trade-off is known to be significantly worse than the central DP model (see Table 1).

This naturally leads to a follow-up question: can we leverage privacy amplification via compression and achieve the same three-way trade-off by using secure aggregation $[33]$ and shuffling $[40]$ type models? The answer is no for secure aggregation. The communication cost for secure aggregation has been studied in $[31]$ and is significantly larger than the communication cost for central DP we establish in this paper (see Table 1). On the other hand, the communication cost for shuffling remains an open problem as discussed in $[31]$ . We resolve this open problem by showing that the optimal central DP trade-off can be also achieved with a multi-message shuffling scheme, which also establishes the optimal communication cost for multi-message shuffling schemes. As before, our scheme leverages a similar privacy amplification gain. Each client communicates partial information about its sample; the identity of the message is erased by the secure shuffler, and hence the untrusted server does not know which part is contributed by each client. However, to achieve the optimal trade-off, it is critical for each client to split its information into multiple messages and employ multiple shuffling rounds by carefully splitting the privacy budget across different rounds. In contrast, a similar gain cannot be leveraged in secure aggregation because the linearity of secure aggregation requires all participating clients to communicate consistent information (same parts), hence precluding privacy amplification by compression. See Table 1 for a detailed comparison.

Our contributions. We consider distributed mean and frequency estimation as canonical building blocks for FL and FA. We consider both the central DP and the multi-message shuffling models. We characterize the order-optimal privacy-accuracy-communication trade-off for distributed mean estimation and provide an achievable scheme for frequency estimation in the central DP model. Our results reveal that privacy and communication efficiency can be achieved simultaneously with no additional penalty on accuracy. In particular, we show that $\tilde{O}\left(n\min\left(\varepsilon,\varepsilon^{2}\right)\right)$ and $\tilde{O}\left(\log\left(n\min\left(\varepsilon,\varepsilon^{2}\right)\right)\right)$ bits of (per-client) communication are sufficient to achieve the order-optimal error under $(\varepsilon,\delta)$ -privacy for mean and frequency estimation respectively, where n is the number of participating clients. Without compression, each client needs $O(d)$ bits and $\log d$ bits for the mean and frequency estimation problems respectively (where d is the number of trainable parameters in FL or the domain size in FA), which means that we can get significant savings in the regime $n\varepsilon^{2}=o(d)$ (assuming $\varepsilon=O(1)$ ). We note that this is often the relevant regime not only for cross-silo but also for cross-device FL/FA. For instance, in practical FL, d usually ranges from $10^{6}-10^{9}$ , and n, the per-epoch sample size, is usually much smaller (e.g., of the order of $10^{3}-10^{5}$ ). For distributed mean estimation,

<table><tr><td></td><td>Communication (bits)</td><td> $\ell_2$  error</td></tr><tr><td>Local DP [32, 42]</td><td> $\Theta (\lceil \varepsilon \rceil)$ </td><td> $\Theta \left( \frac{d}{n \min(\varepsilon^2, \varepsilon)} \right)$ </td></tr><tr><td>Distributed DP (with SecAgg) [33]</td><td> $\tilde{O} (n^2 \min(\varepsilon, \varepsilon^2))$ </td><td> $\Theta \left( \frac{d}{n^2 \min(\varepsilon^2, \varepsilon)} \right)$ </td></tr><tr><td>Central DP (Theorem 4.4)</td><td> $\tilde{O} (n \min(\varepsilon, \varepsilon^2))$ </td><td> $O \left( \frac{d \log d}{n^2 \min(\varepsilon^2, \varepsilon)} \right)$ </td></tr><tr><td>Shuffle DP (Theorem 6.4)</td><td> $\tilde{O} (n \log(d) \min(\varepsilon, \varepsilon^2))$ </td><td> $O \left( \frac{d}{n^2 \min(\varepsilon^2, \varepsilon)} \right)$ </td></tr></table>

Table 1: Comparison of the communication costs of $\ell_{2}$ mean estimation under local, distributed, central, and shuffle DP (with $\delta$ terms hidden). Compared to local DP, we see that error under central DP decays much faster (e.g., $1/n^{2}$ as opposed to 1/n); compared to distributed DP with secure aggregation, our schemes achieve similar accuracy but saves the communication cost by a factor of n.

we show that the central DP trade-off can also be achieved with a multi-message shuffling scheme (within a log d factor in communication cost). Hence our paper establishes the three-way trade-off between privacy, communication, and accuracy for both the central DP and multi-message shuffling frameworks, both of which were open problems in the prior literature. Compared with local DP where 1 bit is sufficient when $\varepsilon = O(1)$ , this shows that central/shuffling DP has a larger communication cost but can achieve much smaller error (by a factor of n) and hence is usually preferable in practical applications. Compared with distributed DP where the server aggregates local (encoded) messages with secure multi-party computation (e.g., [23, 8, 34]), we can improve the communication cost by a factor of n, therefore showing that the communication cost can be reduced with a trusted server or shuffler. We summarize the comparisons of our main results to local and distributed DP in Table 1.

Notation. Throughout this paper, we use $[m]$ to denote the set of $\{1,\ldots,m\}$ for any $m\in N$ . Random variables (vectors) $(X_{1},\ldots,X_{m})$ are denoted as $X^{m}$ . We also make use of Bachmann-Landau asymptotic notation, i.e., O,o,Ω,ω, and Θ.

# 2 Problem Formulation

We first present the distributed mean estimation (DME) [73] problem under differential privacy. Note that DME is closely related to federated learning with SGD (or similar stochastic optimization methods, such as FedAvg [67]), where in each iteration, the server updates the global model by a noisy mean of the local model updates. This noisy estimate is typically obtained by using a DME scheme, and thus one can easily build a distributed DP-SGD scheme (and hence a private FL scheme) from a differentially private DME scheme. Moreover, as shown in [49], as long as we have an unbiased estimate of the gradient at each round, the convergence rates of SGD (or DP-SGD) depend on the $\ell_2$ estimation error.

Distributed mean estimation. Consider n clients each with local data $x_{i} \in R^{d}$ that satisfies $\|x_{i}\|_{2} \leq C$ for some constant C > 0 (one can think of $x_{i}$ as a clipped local gradient). A server wants to learn an estimate $\hat{\mu}$ of the mean $\mu(x^{n}) \triangleq \frac{1}{n} \sum_{i} x_{i}$ from $x^{n} = (x_{1}, \ldots, x_{n})$ after communicating with the n clients. Toward this end, each client locally compresses $x_{i}$ into a b-bit message $Y_{i} = \mathsf{enc}_{i}(x_{i}) \in \mathcal{Y}$ through a local encoder $\mathsf{enc}_{i}: X \mapsto Y$ (where $|Y| \leq 2^{b}$ and sends it to the central server, which upon receiving $Y^{n} = (Y_{1}, \ldots, Y_{n})$ computes an estimate $\hat{\mu} = \mathsf{dec}(Y^{n})$ that satisfies the following differential privacy:

Definition 2.1 (Differential Privacy). The mechanism $\hat{\mu}$ is $(\varepsilon, \delta)$ -differentially private if for any neighboring datasets $x^n := (x_1, ..., x_i, ..., x_n)$ , $x'^n := (x_1, ..., x_i', ..., x_n)$ , and measurable $\mathcal{S} \subseteq \mathcal{Y}$ ,

$$
\operatorname * {P r} \left\{\hat {\mu} \in \mathcal {S} | x ^ {n} \right\} \leq e ^ {\varepsilon} \cdot \operatorname * {P r} \left\{\hat {\mu} \in \mathcal {S} | x ^ {\prime n} \right\} + \delta ,
$$

where the probability is taken over the randomness of $\hat{\mu}$ .

Our goal is to design schemes that minimize the $\ell_{2}^{2}$ estimation error:

$$
\min _ {(\mathsf {e n c} _ {1} (\cdot), \dots , \mathsf {e n c} _ {n} (\cdot), \mathsf {d e c} (\cdot))} \max _ {x ^ {n}} \mathbb {E} \left[ \| \hat {\mu} \left(\mathsf {e n c} _ {1} (x _ {1}), \dots , \mathsf {e n c} _ {n} (x _ {n})\right) - \mu (x ^ {n}) \| _ {2} ^ {2} \right],
$$

subject to $b$ -bit communication and $(\varepsilon, \delta)$ -DP constraints.

Distributed frequency estimation. Similarly, frequency estimation can also be formulated as a mean estimation problem but with sparse (one-hot) vectors. Let each user i hold an item $x_{i}$ in a size d domain X. The server aims to estimate the histogram of the n items. Without loss of generality, we can assume that $X := \{e_{1}, ..., e_{d}\} \in \{0, 1\}^{d}$ (where $e_{j}$ is the j-th standard basis vector in $R^{d}$ ), i.e., each item is expressed as a one-hot vector. Then, the histogram of the n items can be expressed as $\pi(x^{n}) := \sum_{i \in [n]} x_{i}$ . Similar to the mean estimation problem, clients locally compute and then send $Y_{i} = \mathsf{enc}_{i}(x_{i}) \in \mathcal{Y}$ (for some Y such that $|Y| \leq 2^{b}$ ), and the central server computes the estimate $\hat{\pi} = \mathsf{dec}(Y^{n})$ . Our goal is to design schemes that minimize the $\ell_{2}^{2}$ or $\ell_{1}$ error $^{3}$ :

$$
\min _ {(\mathsf {e n c} _ {1} (\cdot), \dots , \mathsf {e n c} _ {n} (\cdot), \mathsf {d e c} (\cdot))} \max _ {x ^ {n}} \mathbb {E} \left[ \| \hat {\pi} \left(\mathsf {e n c} _ {1} (x _ {1}), \dots , \mathsf {e n c} _ {n} (x _ {n})\right) - \pi (x ^ {n}) \| \right],
$$

subject to communication and DP constraints (where $\| \cdot \|$ can be $\ell_1$ or $\ell_2^2$ ).

# 3 Related Works

Federated learning and distributed mean estimation. Federated learning $[63, 67, 59]$ emerges as a decentralized machine learning framework that provides data confidentiality by retaining clients' raw data on edge devices. In FL, communication between clients and the central server can quickly become a bottleneck $[67]$ , so previous works have focused on compressing local model updates via gradient quantization $[67, 10, 48, 73, 78, 76, 24]$ , sparsification $[18, 55, 41]$ . To further enhance data security, FL is often combined with differential privacy $[38, 1, 9]$ . Among these works, $[55]$ also employs gradient sparsification (or gradient subsampling) to reduce the problem dimensionality. However, the sparsification takes place after the aggregation of local gradients, so the randomness introduced during sparsification cannot be leveraged to amplify the differential privacy guarantee. As a result, this approach leads to a suboptimal trade-off between privacy and communication compared to our scheme.

Note that in this work, we consider FL (or more specifically, the distributed mean estimation) under a central-DP setting where the server is trusted, which is different from the local DP model $[62, 37, 69, 75, 22, 32]$ and the distributed DP model with secure aggregation $[23, 21, 58, 8, 33, 34]$ .

A key step in our mean estimation scheme is pre-processing the local data via Kashin's representation [66]. While various compression schemes, based on quantization, sparsification, and dithering have been proposed in the recent literature, Kashin's representation has also been explored in a few works for communication efficiency [47, 72, 29, 70] and for LDP [42] and is particularly powerful in the case of joint communication and privacy constraints as it helps spread the information in a vector evenly in every dimension.

Distributed frequency estimation and heavy hitters. Distributed frequency estimation (a.k.a. histogram estimation) is another canonical task that has been heavily studied under a distributed setting with DP. Prior works either focus on 1) the local DP model with or without communication constraints, e.g., [20, 19, 25, 26, 56] (under an $\ell_{\infty}$ loss for heavy hitter estimation) and [57, 79, 75, 6, 5, 32, 46, 71, 45] (under an $\ell_1$ or $\ell_2$ loss), or 2) the central DP model without communication constraints [38, 52, 64, 27, 13, 80, 36]. As suggested in [37, 3, 2, 4, 16], compared to central DP, local DP models usually incur much larger estimation errors and can significantly decrease the utility. In this work, we consider central DP but with explicit communication constraints.

Local DP with shuffling. A recent line of works $[40, 35, 12, 43, 50, 51]$ considers shuffle-DP, showing that one can significantly boost the central DP guarantees by randomly shuffling local (privatized) messages. In this work, we show that the same shuffling technique can be used to achieve the optimal central DP error with nearly optimal communication cost. Therefore, we can obtain the same level of central DP with small communication costs while weakening the security assumption: achieving the optimal communication cost (under central DP) only requires a secure shuffler (as opposed to a fully trusted central server).

# 4 Distributed Mean Estimation

In this section, we present a mean estimation scheme that achieves the optimal $\tilde{O}_{\delta}\left(\frac{C^{2}d}{n^{2}\varepsilon^{2}}\right)$ error under $(\varepsilon,\delta)$ -DP while only using $\tilde{O}(n\varepsilon^{2})$ bits of per-client communication.

We first consider a slightly simpler, discrete setting with $\ell_{\infty}$ geometry (as opposed to the $\ell_2$ mean estimation stated in Section 2): assume each client observes $x_i \in \{-c, c\}^d$ where $c > 0$ is a constant, and a central server aims to estimate the mean $\mu(x^n) := \frac{1}{n} \sum_{i=1}^{n} x_i$ by minimizing the $\ell_2^2$ error subject to the privacy and communication constraints. We argue later that solutions to the above $\ell_{\infty}$ problem can be used for $\ell_2$ mean estimation by applying Kashin's representation.

To solve the aforementioned $\ell_{\infty}$ mean estimation problem, first observe that each client's local data can be expressed in d bits since each coordinate of $x_{i}$ can only take values in $\{c,-c\}$ . To reduce the communication load to $o(d)$ bits, each client adopts the following subsampling strategy: for each coordinate $j\in[d]$ , client i chooses to send $x_{i}(j)$ to the server with probability $\gamma$ . We assume that this subsampling step is performed with a seed shared by the client and the server $^{4}$ , hence the server knows which coordinates are communicated by each client. Therefore upon receiving the client messages, it can compute the mean of each coordinate and privatize it by adding Gaussian noise. The key observation we leverage is that the randomness in the compression algorithm can be used to amplify privacy or equivalently reduce the magnitude of the Gaussian noise that is needed for privatization. Note that such randomness needs to be kept private from an adversary as the privacy guarantee of the scheme relies on it.

Algorithm 1 Coordinate Subsampled Gaussian Mechanism (CSGM)   
Input: users' data $x_1, \ldots, x_n$ , sampling parameters $\gamma := b/d$ , DP parameters $(\varepsilon, \delta)$ .

Output: mean estimator $\hat{\mu}$ .

for user $i \in [n]$ do
    for coordinate $j \in [d]$ do
    Draw $Z_{i,j} \stackrel{\text{i.i.d.}}{\sim} \text{Bern}(\gamma)$ .
    if $Z_{i,j} = 1$ then
    Send $x_i(j)$ to the server.
    end if
    end for
end for
for coordinate $j \in [d]$ do
    Server computes the average $\hat{\mu}_j := \frac{1}{n\gamma} \sum_{i: Z_{ij}=1} x_i(j) + N(0, \sigma^2)$ , where $\sigma^2$ is computed according to (1) in Theorem 4.1.

end for
Return: $\hat{\mu} := (\hat{\mu}_1, \hat{\mu}_2, \ldots, \hat{\mu}_d)$ .

We summarize the scheme in Algorithm 1 and state its privacy and utility guarantees in the following theorem.

Theorem 4.1 ( $\ell_{\infty}$ mean estimation.). Let $x_{1},...,x_{n}\in\{-c,c\}^{d}$ and let

$$
\sigma^ {2} = O \left(\frac {c ^ {2} \log (1 / \delta)}{n ^ {2} \gamma^ {2}} + \frac {c ^ {2} d (\log (d / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right). \tag {1}
$$

Then for any $\varepsilon,\delta>0$ , Algorithm 1 is $(\varepsilon,\delta)$ -DP and yields an unbiased estimator on $\mu$ . In addition, the (average) per-client communication cost is $\gamma\cdot d=b$ bits, and the $\ell_{2}^{2}$ estimation error of $\hat{\mu}$ is at most

$$
\begin{array}{l} \mathbb {E} \left[ \| \hat {\mu} - \mu \| _ {2} ^ {2} \right] \leq \frac {d c ^ {2}}{n \gamma} + d \sigma^ {2} \\ = O \left(\frac {d ^ {2} c ^ {2}}{n b} + \frac {d ^ {3} c ^ {2} \log (d / \delta)}{n ^ {2} b ^ {2}} + \frac {c ^ {2} d ^ {2} (\log (1 / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right). \tag {2} \\ \end{array}
$$

Remark 4.2 (Unbiasedness). Note that for mean estimation, we usually want the final mean estimator to be unbiased since standard convergence analyses of SGD [49] require an unbiased estimate of the true gradient in each optimization round. Given that our proposed mean estimation schemes (Algorithm 1 and Algorithm 2 in the next section) are all unbiased, we can combine them with SGD/federated averaging and readily apply [49] to obtain a convergence guarantee for the resulting communication-efficient DP-SGD.

For the $\ell_2$ mean estimation task formulated in Section 2, we pre-process local vectors by first computing their Kashin's representations and then performing randomized rounding [61, 74, 42, 32]. Specifically, if $x_i$ has $\ell_2$ norm bounded by $C$ , then its Kashin's representation (with respect to a tight frame $K \in \mathbb{R}^{d \times D}$ where $D = \Theta(d)$ ) $\tilde{x}_i$ has bounded $\ell_\infty$ norm: $\| \tilde{x}_i \|_\infty \leq c = O\left(\frac{C}{\sqrt{d}}\right)$ and satisfies $x_i = K \cdot \tilde{x}_i$ . This allows us to convert the $\ell_2$ geometry to an $\ell_\infty$ geometry. Furthermore, by randomly rounding each coordinate of $\tilde{x}_i$ to $\{-c, c\}$ (see for example [32]), we can readily apply Algorithm 1 and obtain the following result for $\ell_2$ mean estimation as a corollary:

Corollary 4.3 ( $\ell_{2}$ mean estimation). Let $x_{1},...,x_{n}\in\mathcal{B}_{2}(C)$ (i.e., $\|x_{i}\|_{2}\leq C$ for all $i\in[n]$ ). Then for any $\varepsilon,\delta>0$ , Algorithm 1 combined with Kashin's representation and randomized rounding yields an $(\varepsilon,\delta)$ -DP unbiased estimator for $\mu$ with $\ell_{2}^{2}$ estimation error bounded by

$$
O \left(\underbrace {\frac {d C ^ {2}}{n b} + \frac {C ^ {2} d ^ {2} \log (1 / \delta)}{n ^ {2} b ^ {2}}} _ {(\alpha)} + \underbrace {\frac {C ^ {2} d (\log (d / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}} _ {(\beta)}\right). \tag {3}
$$

The first term $(\alpha)$ in the estimation error in Corollary 4.3 is the error due to compression, and the second term $(\beta)$ is the error due to privatization (which is order-optimal under $(\varepsilon, \delta)$ -DP up to an additional $\log(d/\delta)$ factor as we discuss in Section 4.2). In particular, if we ignore the poly-logarithmic terms and assume $\varepsilon = O(1)$ , the privatization error $(\beta)$ can be simplified to $\tilde{O}\left(\frac{dC^2}{n^2\varepsilon^2}\right)$ , which dominates the total $\ell_2^2$ error when $b = \tilde{\Omega}_{\delta}\left(\max\left(n\varepsilon^2, \sqrt{d}\varepsilon\right)\right)$ , i.e. in this regime the total $\ell_2^2$ error is order-wise equal to the optimal centralized DP error $(\beta)$ . This implies that no more than $b = \tilde{\Omega}_{\delta}\left(\max\left(n\varepsilon^2, \sqrt{d}\varepsilon\right)\right)$ bits per client are needed to achieve the order-optimal $\ell_2^2$ error under $(\varepsilon, \delta)$ -DP.

In the next section, we introduce a modification to Algorithm 1, which allows the removal of the $\Omega\left(\sqrt{d}\varepsilon\right)$ term in the communication cost.

# 4.1 Dimension-free communication cost

In order to remove the dependence on the dimension $d$ in the communication cost $b = \tilde{\Omega}_{\delta}\left(\max \left(n\varepsilon^{2},\sqrt{d}\varepsilon\right)\right)$ from the previous section, we need to improve the performance of our scheme in the small-sample regime $n\varepsilon^{2} = o(\sqrt{d}\varepsilon)$ . Equivalently, we want to be able to achieve the centralized DP performance by using only $b = \tilde{\Omega}_{\delta}\left(n\varepsilon^{2}\right)$ bits per client when $n\varepsilon = o(\sqrt{d})$ . Assuming $\varepsilon \approx 1$ , note that this implies that the total communication bandwidth of the system $nb = o(d)$ , i.e. the server can receive information about at most $nb = n^{2}\varepsilon^{2} = o(d)$ coordinates. We show that in this regime the performance of the scheme can be improved by a priori restricting the server's attention to a subset of the coordinates.

We make the following modification to Algorithm 1: before performing Algorithm 1, the server randomly selects $d' \approx O\left(\min(d, n^2\varepsilon^2)\right)$ coordinates and only requires clients to run Algorithm 1 on them. We present the modified scheme in Algorithm 2 and summarize its performance in Theorem 4.4.

Similarly, we can obtain the following $\ell_2$ mean estimation via Kashin's representations:

Theorem 4.4 ( $\ell_{2}$ mean estimation.). Let $x_{1},...,x_{n}\in\mathcal{B}_{2}(C)$ (i.e., $\|x_{i}\|_{2}\leq C$ for all $i\in[n]$ ), $d'=\min\left(d,nb,\frac{n^{2}\varepsilon^{2}}{(\log(1/\delta)+\varepsilon)\log(d/\delta)}\right)$ , and

$$
\sigma^ {2} = O \left(\frac {C ^ {2} \log (1 / \delta)}{d ^ {\prime} n ^ {2} \gamma^ {2}} + \frac {C ^ {2} d ^ {\prime} (\log (1 / \delta) + \varepsilon) \log (d ^ {\prime} / \delta)}{d n ^ {2} \varepsilon^ {2}}\right). \tag {4}
$$

Algorithm 2 CSGM with Coordinate Pre-selection   
Input: users' data $x_{1}, \ldots, x_{n}$ , coordinate selection $d' \leq d$ , sampling parameters $\gamma := b/d'$ , DP parameters $(\varepsilon, \delta)$ .

Output: mean estimator $\hat{\mu}$ .

Randomly select $d'$ coordinates $J := \{j_{1}, \ldots, j_{d'}\} \subset [d]$ .

for user $i \in [n]$ do

    Pre-processing $x_{i}$ by restricting it on J: $x_{i}(\mathcal{J}) := (x_{i}(j_{1}), \ldots, x_{i}(j_{|\mathcal{J}|}))$ .

end for

Apply CSGM (Algorithm 1) on $x_{i}(\mathcal{J})$ , $i \in [n]$ : $\hat{\mu}_{\mathcal{J}} \leftarrow \text{CSGM}(x_{i}(\mathcal{J}), i \in [n])$ .

for $j \in [d]$ do

    if $j \in J$ then $\hat{\mu}_{j} = \hat{\mu}_{\mathcal{J}}(j)$ .

    else $\hat{\mu}_{j} = 0$ .

    end if

end for

Return: $\hat{\mu} := \left( \frac{d}{d'} \hat{\mu}_{1}, \frac{d}{d'} \hat{\mu}_{2}, \ldots, \frac{d}{d'} \hat{\mu}_{d} \right)$ .

Then for any $\varepsilon, \delta > 0$ , Algorithm 2 is $(\varepsilon, \delta)$ -DP. In addition, the (average) per-client communication cost is $\gamma d = b$ bits, and the $\ell_2^2$ estimation error is at most

$$
O \left(\max \left(\frac {C ^ {2} d \log (d / \delta)}{n b}, \frac {C ^ {2} d \log (d / \delta) (\log (1 / \delta) + \varepsilon)}{n ^ {2} \varepsilon^ {2}}\right)\right). \tag {5}
$$

Corollary 4.5. As long as $b = \Omega \left(\frac{n\varepsilon^2}{\log(1 / \delta) + \varepsilon}\right)$ , the $\ell_2^2$ error of mean estimation is

$$
O \left(\frac {C ^ {2} d \log (d / \delta) (\log (1 / \delta) + \varepsilon)}{n ^ {2} \varepsilon^ {2}}\right).
$$

As suggested by Corollary 4.5, we see that when $\varepsilon = O(1)$ , $b = \tilde{\Omega}\left(n\varepsilon^2\right)$ bits per client are sufficient to achieve the order-optimal $\tilde{O}_{\delta}\left(\frac{c^{2}d}{n^{2}\varepsilon^{2}}\right)$ error (even in the small sample regime $n \leq \sqrt{d}$ ), i.e. the communication cost of the scheme is independent of the dimension $d$ .

# 4.2 Lower bounds

In this section, we argue that the estimation error in Theorem 4.4 is optimal up to an $\log (d / \delta)$ factor. Specifically, Theorem 5.3 of [31] shows that any $b$ -bit unbiased compression scheme will incur $\Omega \left(\frac{C^2d}{nb}\right)$ error for the $\ell_2$ mean estimation problem (even when privacy is not required). This matches the first term in (5) up to a logarithmic factor.

On the other hand, the centralized Gaussian mechanism (under a central $(\varepsilon, \delta)$ -DP) achieves $O\left(\frac{C^2 d \log(1/\delta)}{n^2 \varepsilon^2}\right)$ MSE [15] (which is order-optimal in most parameter regimes; see the lower bounds in Theorem 3.1 of [28] or Proposition 23 of [30]). Hence, we can conclude that the total communication received by the server has to be at least $\Omega(n^2 \varepsilon^2)$ bits in order to achieve the same error as the Gaussian mechanism. Therefore, the (average) per-client communication cost has to be at least $\Omega(n \varepsilon^2)$ bits. Hence we conclude that Algorithm 2 is optimal (up to a logarithmic factor).

For completeness, we state the communication lower bound in the following theorem:

Theorem 4.6 (Communication lower bound for mean estimation under central DP). Let $x_{1}, \ldots, x_{n} \in \mathcal{B}_{2}(C)$ . Let $Y_{1}, \ldots, Y_{n}$ be any b-bit local reports generated from a (possibly interactive) compressor and be unbiased in

a sense that

$$
\mathbb {E} \left[ \sum_ {i} Y _ {i} \right] = \sum_ {i} x _ {i}.
$$

Then if

$$
\mathbb {E} \left[ \left\| \frac {1}{n} \sum_ {i} Y _ {i} - \frac {1}{n} \sum_ {i} x _ {i} \right\| _ {2} ^ {2} \right] \leq O \left(\frac {C ^ {2} d \log (1 / \delta)}{n ^ {2} \varepsilon^ {2}}\right),
$$

it holds that

$$
b = \Omega \left(\frac {n \varepsilon^ {2}}{\log (1 / \delta)}\right).
$$

Finally, we remark that the logarithmic gap between the upper and lower bounds may be due to the specific composition theorem (Theorem III.3 of [39]) we use in our proof, which is simpler to work with but possibly slightly weaker. However, in our experiments, we compute and account for all privacy budgets with Rényi DP [68, 81], and hence can obtain better constants compared to our theoretical analysis.

# 5 Distributed Frequency Estimation

In this section, we consider the frequency estimation problem for federated analytics. Recall that for the frequency estimation task, each client's private data $x_{i} \in \{0,1\}^{d}$ satisfies $\| x_{i}\|_{0} = 1$ , and the goal is to estimate $\pi := \frac{1}{n}\sum_{i}x_{i}$ by minimizing the $\ell_2$ (or $\ell_1,\ell_\infty$ ) error $\mathbb{E}\left[\| \pi -\hat{\pi}(Y^n)\| _2^2\right]$ subject to communication and $(\varepsilon ,\delta)$ -DP constraints. When the context is clear, we sometimes use $x_{i}$ to denote, by abuse of notation, the index of the item, i.e., $x_{i}\in [d]$ .

To fully make use of the $\ell_{0}$ structure of the problem, a standard technique is applying a Hadamard transform to convert the $\ell_{0}$ geometry to an $\ell_{\infty}$ one and then leveraging the recursive structure of Hadamard matrices to efficiently compress local messages.

Specifically, for a given $b$ -bit constraint, we partition each local item $x_{i}$ into $2^{b - 1}$ chunks $x_{i}^{(1)},\dots,x_{i}^{(2^{b} - 1)}\in$ $\{0,1\} ^B$ , where $B:=d / 2^{b - 1}$ and $x_{i}^{(j)} = x_{i}[B\cdot (j - 1):B\cdot j - 1]$ . Note that since $x_{i}$ is one-hot, only one chunk of $x_{i}^{(j)}$ is non-zero. Then, client $i$ performs the following Hadamard transform for each chunk: $y_{i}^{(\ell)} = H_{B}\cdot x_{i}^{(\ell)}$ , where $H_{B}$ is defined recursively as follows:

$$
H _ {2 ^ {n}} = \frac {1}{\sqrt {2}} \left[ \begin{array}{c c} H _ {2 ^ {n - 1}}, & H _ {2 ^ {n - 1}} \\ H _ {2 ^ {n - 1}}, & - H _ {2 ^ {n - 1}} \end{array} \right], \text {and} H _ {0} = \left[ 1 \right].
$$

Each client then generates a sampling vector $Z_{ij} \stackrel{i.i.d.}{\sim}$ Bern $\left(\frac{1}{B}\right)$ via shared randomness that is also known by the server, and commits $(y_{i}^{(1)}(j),...,y_{i}^{(2^{b-1})}(j))$ as its local report. Since $(y_{i}^{(1)}(j),...,y_{i}^{(2^{b-1})}(j))$ only contains a single non-zero entry that can be $\frac{1}{\sqrt{B}}$ or $-\frac{1}{\sqrt{B}}$ , the local report can be represented in b bits (b-1 bits for the location of the non-zero entry and 1 bit for its sign).

From the local reports, the server can compute an unbiased estimator by summing them together (with proper normalization) and performing an inverse Hadamard transform. Moreover, with an adequate injection of Gaussian noise, the frequency estimator satisfies $(\varepsilon,\delta)$ -DP.

The idea has been used in previous literature under local DP $[19, 6, 3, 32]$ , but in order to obtain the order-optimal trade-off under central-DP, one has to combine Hadamard transform with a random subsampling step and incorporate the privacy amplification due to random compression in the analysis. In Algorithm 3, we provide a summary of the resultant scheme which builds on the Recursive Hadamard Response (RHR) mechanism from $[32]$ , which was originally designed for communication-efficient frequency estimation under local DP.

In the following theorem, we control the $\ell_{\infty}$ error of Algorithm 3.

Theorem 5.1. Let $\hat{\pi}(x^n)$ be the output of Algorithm 3. Then it holds that for all $j\in [d]$ ,

$$
\mathbb {E} \left[ | \pi (j) - \hat {\pi} (j) | \right] \leq \sqrt {\frac {\sum_ {i} \mathbb {1} _ {\{x _ {i} \in [ B \cdot (j - 1) : B \cdot j - 1 ] \}}}{n ^ {2}} + \frac {\sigma^ {2}}{B}}, \tag {6}
$$

Algorithm 3 Subsampled Recursive Hadamard Response   
Input: user data $x_{1}, \ldots, x_{n} \in \{0,1\}^{d}$ (where d is a power of two), DP parameters $(\varepsilon, \delta)$ , communication budget b.

Output: frequency estimate $\hat{\pi}$ Set $B := d/2^{b-1}$ and partition each one-hot vector $x_{i}$ into $2^{b-1}$ chunks: $x_{i}^{(1)}, \ldots, x_{i}^{(2^{b}-1)} \in \{0,1\}^{B}$ .

for user $i \in [n]$ do

    Compute the Hadamard transform of each chunk: $y_{i}^{(\ell)} = H_{B} \cdot x_{i}^{(\ell)}$ .

    for coordinate $j \in [B]$ do

    Draw $Z_{i,j} \stackrel{\text{i.i.d.}}{\sim} \operatorname{Bern}\left(\frac{1}{B}\right)$ if $Z_{i,j} = 1$ then

    Send $(y_{i}^{(1)}(j), \ldots, y_{i}^{(2^{b-1})}(j))$ to the server.

    end if

    end for

end for

Server computes the average: $\forall \ell \in [2^{b-1}], j \in [B]$ ,

$$
\hat {y} ^ {(\ell)} (j) := \frac {B}{n} \sum_ {i: Z _ {i j} = 1} y _ {i} ^ {(\ell)} (j) + N (0, \sigma^ {2}),
$$

where $\sigma^2$ is computed according to Theorem 5.2.

Server performs the inverse Hadamard transform $\hat{\pi}^{(\ell)} = H_B\cdot \hat{y}^{(\ell)}$ , for $\ell = 1,\dots,B$ .

$\mathbf{Return}:\hat{\pi} = \left(\left(\hat{\pi}^{(1)}\right)^{\intercal},\dots,\left(\hat{\pi}^{(2^{b - 1})}\right)^{\intercal}\right).$

and the $\ell_2^2$ and $\ell_1$ errors are bounded by

$$
\mathbb {E} \left[ \| \pi - \hat {\pi} \| _ {2} ^ {2} \right] \leq \frac {B}{n} + \frac {d \sigma^ {2}}{B}, a n d \tag {7}
$$

$$
\mathbb {E} \left[ \| \pi - \hat {\pi} \| _ {1} \right] \leq \sqrt {\frac {d B}{n} + \frac {d ^ {2} \sigma^ {2}}{B}}. \tag {8}
$$

Theorem 5.2. For any $\varepsilon, \delta > 0$ , Algorithm 3 is $(\varepsilon, \delta)$ -DP, if

$$
\sigma^ {2} \geq O \left(\frac {B ^ {2} \log (B / \delta)}{n ^ {2}} + \frac {B (\log (1 / \delta) + \varepsilon) \log (B / \delta)}{n ^ {2} \varepsilon^ {2}}\right).
$$

By combining Theorem 5.1 and Theorem 5.2, we conclude that Algorithm 3 achieves $(\varepsilon, \delta)$ -DP with $\ell_2^2$ error

$$
\begin{array}{l} O \left(\frac {B}{n} + \frac {d B \log (B / \delta)}{n ^ {2}} + \frac {d (\log (1 / \delta) + \varepsilon) \log (B / \delta)}{n ^ {2} \varepsilon^ {2}}\right) \\ = O \left(\frac {d}{n 2 ^ {b}} + \frac {d ^ {2} \log (d / \delta)}{n ^ {2} 2 ^ {b}} + \frac {d (\log (1 / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right). \\ \end{array}
$$

Notice that when $n = \tilde{\Omega}(d)$ , the error can be simplified to

$$
O \left(\frac {d}{n 2 ^ {b}} + \frac {d (\log (1 / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right),
$$

which matches the order-optimal estimation error (up to a log d factor) subject to a b-bit constraint $[54, 3, 2]$ and $(\varepsilon, \delta)$ -DP constraint $[15, 7]$ .

# 6 Achieving the Optimal Trade-off via Shuffling

In Section 4 and Section 5, we see that the communication cost can be reduced to $(\tilde{O}(n\varepsilon^{2})$ for mean estimation and $\tilde{O}(\log(\lceil n\varepsilon^{2}\rceil))$ for frequency estimation) while still achieving the order-wise optimal error, as long as the server is trusted. On the other hand, when the server is untrusted, [33, 31] show that optimal error under $(\varepsilon,\delta)$ -DP can be achieved with secure aggregation. However, the communication cost of these schemes is $\tilde{O}(n^{2}\varepsilon^{2})$ bits per client for mean estimation and $\tilde{O}(n\varepsilon)$ bits per client for frequency estimation. This corresponds to a factor of n increase for mean estimation and an exponential increase for frequency estimation. In this section, we investigate whether the optimal communication-accuracy-privacy trade-off from the previous sections can be achieved when the server is not fully trusted.

In this section, we show that if there exists a secure shuffler that randomly permutes clients' locally privatized messages and releases them to the server, we can achieve the nearly optimal (within a log d factor) central-DP error in mean estimation with $\tilde{O}\left(n\varepsilon^{2}\right)$ bits of communication. Specifically, we present a mean estimation scheme that combines a local-DP mechanism with privacy amplification via shuffling by building on the following recent result [40, 43]:

Lemma 6.1 ([43]). Let $\mathcal{M}_i$ be an independent $(\varepsilon_0,0)$ -LDP mechanism for each $i\in [n]$ with $\varepsilon_0\leq 1$ and $\pi$ be a random permutation of $[n]$ . Then for any $\delta \in [0,1]$ such that $\varepsilon_0\leq \log \left(\frac{n}{16\log(2 / \delta)}\right)$ , the mechanism

$$
\mathcal {S}: (x _ {1}, \dots , x _ {n}) \mapsto \left(\mathcal {M} _ {1} \left(x _ {\pi (1)}\right), \dots , \mathcal {M} _ {n} \left(x _ {\pi (n)}\right)\right)
$$

is $(\varepsilon, \delta)$ -DP for some $\varepsilon$ such that $\varepsilon = O\left(\varepsilon_0 \frac{\sqrt{\log(1/\delta)}}{\sqrt{n}}\right)$ .

Privacy analysis. With the above amplification lemma, we only need to design the local randomizers $M_{i}$ that satisfy $\varepsilon_{0}$ -LDP. Note that the above lemma is only tight when $\varepsilon_{0}=O(1)$ , thus restricting the (amplified) central $\varepsilon=O(1/\sqrt{n})$ , i.e. to be very small. To accommodate larger $\varepsilon$ , users can send different portions of their messages to the server in separate shuffling rounds. Equivalently, we repeat the shuffled LDP mechanism for $T=O\left(\lceil n\varepsilon^{2}\rceil\right)$ rounds while ensuring that in each round clients communicate an independent piece of information about their sample to the server. More precisely, within each round, each client applies the local randomizers $M_{i}$ with a per-round local privacy budget $\varepsilon_{0}=O(1)$ and sends an independent message to the server. This results in (amplified) central $O(1/\sqrt{n})$ -DP per round, which after composition over $T=O\left(\lceil n\varepsilon^{2}\rceil\right)$ rounds leads to $\varepsilon$ -DP for the overall scheme as suggested by the composition theorem [57]). We detail the algorithm in Algorithm 4 in Appendix E.

Remark 6.2. Although our analysis mainly focuses on $(\varepsilon,\delta)$ -DP, one can also obtain Rényi DP guarantees (which may facilitate practical privacy accounting) using the recent amplification result [44].

Communication costs. The communication cost of the above T-round scheme can be computed as follows. As shown in [32], the optimal communication cost of an $\varepsilon_{0}$ -LDP mean estimation is $O\left(\left\lceil\varepsilon_{0}\right\rceil\right)$ bits. In addition, the (private-coin) SQKR scheme proposed in [32] uses $O\left(\left\lceil\varepsilon_{0}\right\rceil\log d\right)$ bits of communication (we state the formal performance guarantee for this scheme in Lemma 6.3), where compression is done by subsampling coordinates and privatization is performed with Randomized Response. Therefore, since the per-round $\varepsilon_{0}=O(1)$ , the total per-client communication cost is $O\left(n\varepsilon^{2}\log d\right)$ , matching the optimal communication bounds in Section 4 within a $\log d$ factor.

Lemma 6.3 (SQKR [32]). For all $\varepsilon_0 > 0, b_0 > 0$ , there exists a $(\varepsilon_0, 0)$ -LDP mechanism $x_i \mapsto \hat{\mu}$ using $b_0 \log(d)$ bits such that $\hat{\mu}$ is unbiased and satisfies

$$
\mathbb {E} \left[ \| \mu (x ^ {n}) - \hat {\mu} (x ^ {n}) \| _ {2} ^ {2} \right] = O \left(\frac {c ^ {2} d}{n \min (\varepsilon_ {0} ^ {2} , \varepsilon_ {0} , b _ {0})}\right).
$$

Finally, we summarize the performance guarantee for the overall scheme (Algorithm 4) in the following theorem.

Theorem 6.4 ( $\ell_{2}$ mean estimation). Let $x_{1},...,x_{n}\in\mathcal{B}_{2}(C)$ (i.e., $\|x_{i}\|_{2}\leq C$ for all $i\in[n]$ ). For all $\varepsilon>0,b>0,n>30$ , and $\delta\in(\delta_{\min},1]$ where $\delta_{\min}=O\left(\frac{be^{-n}}{\log(d)}\right)$ , Algorithm 4 combined with Kashin's representation and randomized rounding is $(\varepsilon,\delta)$ -DP, uses no more than b bits of communication, and achieves

$$
\mathbb {E} \left[ \| \mu (x ^ {n}) - \hat {\mu} (x ^ {n}) \| _ {2} ^ {2} \right] = O \left(C ^ {2} d \max \left(\frac {\log (d)}{n b}, \frac {\log (b / \delta) (\log (1 / \delta) + \varepsilon)}{n ^ {2} \varepsilon^ {2}}\right)\right).
$$

Remark 6.5. As opposed to previous schemes Algorithm 1-3, the shuffled SQKR requires some condition on $\delta$ , i.e., $\delta \in [\delta_{min}, 1]$ due to the specific shuffling lemma we used. In practice, however, $\delta_{\min}$ is small due to the exponential dependence on $n$ . The order-wise optimal error of $O\left(\frac{C^2d}{n^2\min(\varepsilon^2,\varepsilon)}\right)$ is achieved, up to logarithmic factors, when $b = \Omega_{\delta}\left(n\log(d)\min\left(\varepsilon^{2},\varepsilon\right)\right)$ .

Remark 6.6. We note that similar ideas of private mean estimation based on shuffling have been studied before, see, for instance, [53]. However, these papers do not use the above privacy budget splitting trick over multiple rounds, so their result is only optimal when $\varepsilon$ is very small. The above scheme can be viewed as a multi-message shuffling scheme [35, 51], and in particular, can be regarded as a generalization of the scalar mean estimation scheme [35] to d-dim mean estimation.

# 7 Experiments

In this section, we empirically evaluate our mean estimation scheme (CSGM) from Section 4, examine its privacy-accuracy-communication trade-off, and compare it with other DP mechanisms (including the shuffling-based mechanism introduced in Section 6).

Setup. For a given dimension d, and number of samples n, we generate local vectors $X_{i} \in R^{d}$ as follows: let $X_{i}(j) \stackrel{\text{i.i.d.}}{\sim} \frac{1}{\sqrt{d}} (2 \cdot \operatorname{Ber}(0.8) - 1)$ where $\operatorname{Ber}(0.8)$ is a Bernoulli random variable with bias p = 0.8. This ensures $\|X_{i}\|_{\infty} \leq 1/\sqrt{d}$ and $\|X_{i}\|_{2} \leq 1$ , and in addition, the empirical mean $\mu(X^{n}) := \frac{1}{n} \sum_{i} X_{i}$ does not converge to 0. Note that as our goal is to construct an unbiased estimator, we did not project our final estimator back to the $\ell_{\infty}$ or $\ell_{2}$ space as the projection step may introduce bias. Therefore, the $\ell_{2}$ estimation error can be greater than 1. We account for the privacy budget with Rényi DP [68] and the privacy-amplification by subsampling lemma in [81] and convert Rényi DP to $(\varepsilon, \delta)$ -DP via [30].

Privacy-accuracy-communication trade-off of CSGM. In the first set of experiments (Figure 1), we apply Algorithm 1 with different sampling rates $\gamma$ , which leads to different communication budgets ( $b = \gamma d$ ). Note that when $\gamma = 1$ , the scheme reduces to the central Gaussian mechanism without compression. In Figure1, we see that with a fixed communication budget, CSGM approximates the central (uncompressed) Gaussian mechanism in the high privacy regime (small $\varepsilon$ ) and starts deviating from it when $\varepsilon$ exceeds a certain value. In addition, that value of $\varepsilon$ depends only on sample size $n$ and the communication budget $b$ and not the dimension $d$ as predicted by our theory: recall that the compression error dominates the total error, and hence the performance starts to deviate from the (uncompressed) Gaussian mechanism when $b = o(n\varepsilon^2)$ , a condition that is independent of $d$ . Observe, for example, that when $b = 50$ bits, the Gaussian mechanism starts outperforming CSGM at $\varepsilon \geq 0.5$ for both $d = 500$ and $d = 5000$ . Hence, for $\varepsilon \approx 0.5$ CSGM is able to provide 10X compression when $d = 500$ , but 100X compression when $d = 5000$ without impacting MSE.

Comparison with local and shuffle DP. Next, we compare the CSGM with local and shuffled DP for $d = 10^{3}$ and n = 500. For local DP, we consider the private-coin SQKR scheme introduced in Section 6 which uses $\lceil \log d \rceil = 10$ bits when $\varepsilon \leq 1$ and DJW [37] which is known to be order-optimal when $\varepsilon = O(1)$ (but is not communication-efficient). For shuffle-DP, we apply the amplification lemma in [43] to find the corresponding local $\varepsilon_{0}$ (see Section 6 for more details) and simulate both SQKR and DJW as the local randomizersWe note that all shuffle-DP mechanisms considered in this experiment are single-round (as opposed to the multi-round schemes in Section 6), which is optimal in the high privacy regime (i.e., when $\varepsilon$ is small).

![](images/b14720a731625109e3bbb19b21427475d2799ef52031d33dc69099dc5a56da01.jpg)

<details>
<summary>line</summary>

| Privacy (ε) | γ = 0.1 (50 bits) | γ = 0.2 (100 bits) | γ = 0.5 (250 bits) | γ = 0.8 (400 bits) | γ = 1.0 (500 bits) |
| ----------- | ----------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| 0.0         | ~10^1             | ~10^1              | ~10^1              | ~10^1              | ~10^1              |
| 0.5         | ~10^-1            | ~10^-1             | ~10^-1             | ~10^-1             | ~10^-1             |
| 1.0         | ~10^-2            | ~10^-2             | ~10^-2             | ~10^-2             | ~10^-2             |
| 1.5         | ~10^-2            | ~10^-2             | ~10^-2             | ~10^-3             | ~10^-3             |
| 2.0         | ~10^-2            | ~10^-2             | ~10^-2             | ~10^-3             | ~10^-4             |
| 2.5         | ~10^-2            | ~10^-2             | ~10^-2             | ~10^-3             | ~10^-4             |
| 3.0         | ~10^-2            | ~10^-2             | ~10^-2             | ~10^-3             | ~10^-4             |
| 3.5         | ~10^-2            | ~10^-2             | ~10^-2             | ~10^-3             | ~10^-4             |
| 4.0         | ~10^-2            | ~10^-2             | ~10^-2             | ~10^-3             | ~10^-5             |
</details>

![](images/bee22275fd16bb5d24fa13dc53a3fe40afe5dd074ab65be870a9dfff3affb4e5.jpg)

<details>
<summary>line</summary>

| Privacy (ε) | γ = 0.01 (50 bits) | γ = 0.02 (100 bits) | γ = 0.05 (250 bits) | γ = 0.1 (500 bits) | γ = 0.2 (1000 bits) | γ = 0.5 (2500 bits) | γ = 0.8 (4000 bits) | γ = 1.0 (5000 bits) |
| ----------- | ------------------ | ------------------- | ------------------- | ------------------ | ------------------- | ------------------- | ------------------- | ------------------- |
| 0.0         | ~10^3              | ~10^3               | ~10^3               | ~10^3              | ~10^3               | ~10^3               | ~10^3               | ~10^3               |
| 0.5         | ~10^1              | ~10^1               | ~10^1               | ~10^1              | ~10^1               | ~10^1               | ~10^1               | ~10^1               |
| 1.0         | ~10^0              | ~10^0               | ~10^0               | ~10^0              | ~10^0               | ~10^0               | ~10^0               | ~10^0               |
| 1.5         | ~10^-1             | ~10^-1              | ~10^-1              | ~10^-1             | ~10^-1              | ~10^-1              | ~10^-1              | ~10^-1              |
| 2.0         | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2              | ~10^-2              |
| 2.5         | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2              | ~10^-2              |
| 3.0         | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2              | ~10^-2              |
| 3.5         | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2              | ~10^-2              |
| 4.0         | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2             | ~10^-2              | ~10^-2              | ~10^-2              | ~10^-2              |
</details>

Figure 1: MSE of CSGM (Algorithm 1) with a lower (left) and higher (right) dimension.

The MSEs of all mechanisms are reported in Figure 2. Our results suggest that for a fixed communication budget (say, 10 bits), the practical performance of CSGM outperforms shuffled-DP mechanisms, including the shuffled SQKR and DJW. In addition, the amplification gain of shuffling diminishes fast as $\varepsilon$ increases. Indeed, when $\varepsilon \geq 0.8$ , we observe no amplification gain compared to the pure local DP.

![](images/f92fcf28e1f0419dd8892719f05e35e1474468918faa39f23adf212dfe413ba4.jpg)

<details>
<summary>line</summary>

| Privacy (ε) | CSGM (10 bits) | CSGM (20 bits) | CSGM (50 bits) | CSGM (100 bits) | CSGM (500 bits) | central Gaussian | SQKR (10 bits) | DJW | SQKR_shuffle (10 bits) | DJW_shuffle |
|-------------|----------------|----------------|----------------|-----------------|-----------------|------------------|----------------|-----|------------------------|-------------|
| 0.1         | ~10^3          | ~10^3          | ~10^3          | ~10^3           | ~10^3           | ~10^3            | ~10^3          | ~10^3 | ~10^3                  | ~10^3       |
| 0.2         | ~10^2          | ~10^2          | ~10^2          | ~10^2           | ~10^2           | ~10^2            | ~10^2          | ~10^2 | ~10^2                  | ~10^2       |
| 0.4         | ~10^1          | ~10^1          | ~10^1          | ~10^1           | ~10^1           | ~10^1            | ~10^1          | ~10^1 | ~10^1                  | ~10^1       |
| 0.6         | ~10^0          | ~10^0          | ~10^0          | ~10^0           | ~10^0           | ~10^0            | ~10^0          | ~10^0 | ~10^0                  | ~10^0       |
| 0.8         | ~10^-1         | ~10^-1         | ~10^-1         | ~10^-1          | ~10^-1          | ~10^-1           | ~10^-1         | ~10^-1 | ~10^-1                 | ~10^-1      |
| 1.0         | ~10^-2         | ~10^-2         | ~10^-2         | ~10^-2          | ~10^-2          | ~10^-2           | ~10^-2         | ~10^-2 | ~10^-2                 | ~10^-2      |
</details>

Figure 2: A comparison with local and shuffled DP.

# References

[1] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pages 308–318, 2016.   
[2] J. Acharya, C. L. Canonne, and H. Tyagi. Inference under information constraints ii: Communication constraints and shared randomness. arXiv preprint arXiv:1905.08302, 2019.   
[3] J. Acharya, C. L. Canonne, and H. Tyagi. Inference under information constraints: Lower bounds from chi-square contraction. In Conference on Learning Theory, pages 3–17. PMLR, 2019.   
[4] J. Acharya, C. L. Canonne, and H. Tyagi. General lower bounds for interactive high-dimensional estimation under information constraints. arXiv preprint arXiv:2010.06562, 2020.

[5] J. Acharya and Z. Sun. Communication complexity in locally private distribution estimation and heavy hitters. In International Conference on Machine Learning, pages 51–60, 2019.   
[6] J. Acharya, Z. Sun, and H. Zhang. Hadamard response: Estimating distributions privately, efficiently, and with little communication. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 1120–1129, 2019.   
[7] J. Acharya, Z. Sun, and H. Zhang. Differentially private assouad, fano, and le cam. In Algorithmic Learning Theory, pages 48–78. PMLR, 2021.   
[8] N. Agarwal, P. Kairouz, and Z. Liu. The skellam mechanism for differentially private federated learning. Advances in Neural Information Processing Systems, 34:5052-5064, 2021.   
[9] N. Agarwal, A. T. Suresh, F. X. X. Yu, S. Kumar, and B. McMahan. cpsgd: Communication-efficient and differentially-private distributed sgd. In Advances in Neural Information Processing Systems, pages 7564–7575, 2018.   
[10] D. Alistarh, D. Grubic, J. Li, R. Tomioka, and M. Vojnovic. Qsgd: Communication-efficient sgd via gradient quantization and encoding. In Advances in Neural Information Processing Systems 30, pages 1709–1720, 2017.   
[11] J. Allen, B. Ding, J. Kulkarni, H. Nori, O. Ohrimenko, and S. Yekhanin. An algorithmic framework for differentially private data analysis on trusted processors. Advances in Neural Information Processing Systems, 32, 2019.   
[12] V. Balcer and A. Cheu. Separating local & shuffled differential privacy via histograms. arXiv preprint arXiv:1911.06879, 2019.   
[13] V. Balcer and S. Vadhan. Differential privacy on finite computers. arXiv preprint arXiv:1709.05396, 2017.   
[14] B. Balle, G. Barthe, and M. Gaboardi. Privacy amplification by subsampling: Tight analyses via couplings and divergences. Advances in Neural Information Processing Systems, 31, 2018.   
[15] B. Balle and Y.-X. Wang. Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning, pages 394–403. PMLR, 2018.   
[16] L. P. Barnes, W.-N. Chen, and A. Ozgur. Fisher information under local differential privacy. arXiv preprint arXiv:2005.10783, 2020.   
[17] L. P. Barnes, Y. Han, and A. Ozgur. Lower bounds for learning distributions under communication constraints via fisher information, 2019.   
[18] L. P. Barnes, H. A. Inan, B. Isik, and A. Ozgur. rtop-k: A statistical estimation approach to distributed sgd, 2020.   
[19] R. Bassily, K. Nissim, U. Stemmer, and A. Thakurta. Practical locally private heavy hitters. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS'17, page 2285–2293, Red Hook, NY, USA, 2017. Curran Associates Inc.   
[20] R. Bassily and A. Smith. Local, private, efficient protocols for succinct histograms. In Proceedings of the Forty-Seventh Annual ACM Symposium on Theory of Computing, STOC '15, page 127–135, New York, NY, USA, 2015. Association for Computing Machinery.   
[21] J. H. Bell, K. A. Bonawitz, A. Gascón, T. Lepoint, and M. Raykova. Secure single-server aggregation with (poly) logarithmic overhead. In Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security, pages 1253–1269, 2020.

[22] A. Bhowmick, J. Duchi, J. Freudiger, G. Kapoor, and R. Rogers. Protection against reconstruction and its applications in private federated learning. arXiv preprint arXiv:1812.00984, 2018.   
[23] K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, and K. Seth. Practical secure aggregation for federated learning on user-held data. arXiv preprint arXiv:1611.04482, 2016.   
[24] M. Braverman, A. Garg, T. Ma, H. L. Nguyen, and D. P. Woodruff. Communication lower bounds for statistical estimation problems via a distributed data processing inequality. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, pages 1011-1020, 2016.   
[25] M. Bun, J. Nelson, and U. Stemmer. Heavy hitters and the structure of local privacy. In Proceedings of the 37th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems, SIGMOD-/PODS '18, page 435–447, New York, NY, USA, 2018. Association for Computing Machinery.   
[26] M. Bun, J. Nelson, and U. Stemmer. Heavy hitters and the structure of local privacy. ACM Transactions on Algorithms (TALG), 15(4):1-40, 2019.   
[27] M. Bun and T. Steinke. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In Theory of Cryptography Conference, pages 635-658. Springer, 2016.   
[28] T. T. Cai, Y. Wang, and L. Zhang. The cost of privacy: Optimal rates of convergence for parameter estimation with differential privacy. The Annals of Statistics, 49(5):2825-2850, 2021.   
[29] S. Caldas, J. Konečny, H. B. McMahan, and A. Talwalkar. Expanding the reach of federated learning by reducing client resource requirements. arXiv preprint arXiv:1812.07210, 2018.   
[30] C. L. Canonne, G. Kamath, and T. Steinke. The discrete gaussian for differential privacy. arXiv preprint arXiv:2004.00010, 2020.   
[31] W.-N. Chen, C. A. C. Choo, P. Kairouz, and A. T. Suresh. The fundamental price of secure aggregation in differentially private federated learning. In International Conference on Machine Learning, pages 3056-3089. PMLR, 2022.   
[32] W.-N. Chen, P. Kairouz, and A. Ozgur. Breaking the communication-privacy-accuracy trilemma. Advances in Neural Information Processing Systems, 33, 2020.   
[33] W.-N. Chen, A. Özgür, G. Cormode, and A. Baharadwaj. The communication cost of security and privacy in federated frequency estimation. in submission, 2022.   
[34] W.-N. Chen, A. Ozgur, and P. Kairouz. The poisson binomial mechanism for unbiased federated learning with secure aggregation. In International Conference on Machine Learning, pages 3490-3506. PMLR, 2022.   
[35] A. Cheu, A. Smith, J. Ullman, D. Zeber, and M. Zhilyaev. Distributed differential privacy via shuffling. In Annual International Conference on the Theory and Applications of Cryptographic Techniques, pages 375–403. Springer, 2019.   
[36] G. Cormode and A. Bharadwaj. Sample-and-threshold differential privacy: Histograms and applications. In International Conference on Artificial Intelligence and Statistics, pages 1420–1431. PMLR, 2022.   
[37] J. C. Duchi, M. I. Jordan, and M. J. Wainwright. Local privacy and statistical minimax rates. In 2013 IEEE 54th Annual Symposium on Foundations of Computer Science, pages 429–438. IEEE, 2013.   
[38] C. Dwork, F. McSherry, K. Nissim, and A. Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, pages 265-284. Springer, 2006.   
[39] C. Dwork, G. N. Rothblum, and S. Vadhan. Boosting and differential privacy. In 2010 IEEE 51st Annual Symposium on Foundations of Computer Science, pages 51–60. IEEE, 2010.

[40] Ú. Erlingsson, V. Feldman, I. Mironov, A. Raghunathan, K. Talwar, and A. Thakurta. Amplification by shuffling: From local to central differential privacy via anonymity. In Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms, pages 2468–2479. SIAM, 2019.   
[41] F. Farokhi. Gradient sparsification can improve performance of differentially-private convex machine learning. In 2021 60th IEEE Conference on Decision and Control (CDC), pages 1695–1700. IEEE, 2021.   
[42] V. Feldman, C. Guzman, and S. Vempala. Statistical query algorithms for mean vector estimation and stochastic convex optimization. In Proceedings of the Twenty-Eighth Annual ACM-SIAM Symposium on Discrete Algorithms, pages 1265-1277. SIAM, 2017.   
[43] V. Feldman, A. McMillan, and K. Talwar. Hiding among the clones: A simple and nearly optimal analysis of privacy amplification by shuffling. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pages 954–964. IEEE, 2022.   
[44] V. Feldman, A. McMillan, and K. Talwar. Stronger privacy amplification by shuffling for rényi and approximate differential privacy. In Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA), pages 4966–4981. SIAM, 2023.   
[45] V. Feldman, J. Nelson, H. Nguyen, and K. Talwar. Private frequency estimation via projective geometry. In International Conference on Machine Learning, pages 6418–6433. PMLR, 2022.   
[46] V. Feldman and K. Talwar. Lossless compression of efficient private local randomizers. arXiv preprint arXiv:2102.12099, 2021.   
[47] J.-J. Fuchs. Spread representations. In 2011 Conference Record of the Forty Fifth Asilomar Conference on Signals, Systems and Computers (ASILOMAR), pages 814–817. IEEE, 2011.   
[48] V. Gandikota, D. Kane, R. K. Maity, and A. Mazumdar. vqsgd: Vector quantized stochastic gradient descent, 2019.   
[49] S. Ghadimi and G. Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization, 23(4):2341-2368, 2013.   
[50] B. Ghazi, N. Golowich, R. Kumar, R. Pagh, and A. Velingker. On the power of multiple anonymous messages. arXiv preprint arXiv:1908.11358, 2019.   
[51] B. Ghazi, R. Kumar, P. Manurangsi, and R. Pagh. Private counting from anonymous messages: Near-optimal accuracy with vanishing communication overhead. In International Conference on Machine Learning, pages 3505-3514. PMLR, 2020.   
[52] A. Ghosh, T. Roughgarden, and M. Sundararajan. Universally utility-maximizing privacy mechanisms. SIAM Journal on Computing, 41(6):1673–1693, 2012.   
[53] A. Girgis, D. Data, S. Diggavi, P. Kairouz, and A. T. Suresh. Shuffled model of differential privacy in federated learning. In International Conference on Artificial Intelligence and Statistics, pages 2521-2529. PMLR, 2021.   
[54] Y. Han, A. Özgür, and T. Weissman. Geometric lower bounds for distributed parameter estimation under communication constraints. In Conference On Learning Theory, pages 3163-3188. PMLR, 2018.   
[55] R. Hu, Y. Gong, and Y. Guo. Federated learning with sparsification-amplified privacy and adaptive optimization. arXiv preprint arXiv:2008.01558, 2020.   
[56] Z. Huang, Y. Qiu, K. Yi, and G. Cormode. Frequency estimation under multiparty differential privacy: One-shot and streaming. Proc. VLDB Endow., 15(10):2058–2070, jun 2022.   
[57] P. Kairouz, K. Bonawitz, and D. Ramage. Discrete distribution estimation under local privacy. In Proceedings of The 33rd International Conference on Machine Learning, volume 48, pages 2436–2444, New York, New York, USA, 20–22 Jun 2016.

[58] P. Kairouz, Z. Liu, and T. Steinke. The distributed discrete gaussian mechanism for federated learning with secure aggregation. arXiv preprint arXiv:2102.06387, 2021.   
[59] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, et al. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977, 2019.   
[60] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, et al. Advances and open problems in federated learning. Foundations and Trends® in Machine Learning, 14(1–2):1–210, 2021.   
[61] B. Kashin. Section of some finite-dimensional sets and classes of smooth functions (in russian) izv. Acad. Nauk. SSSR, 41:334-351, 1977.   
[62] S. P. Kasiviswanathan, H. K. Lee, K. Nissim, S. Raskhodnikova, and A. Smith. What can we learn privately? SIAM Journal on Computing, 40(3):793–826, 2011.   
[63] J. Konečný, H. B. McMahan, F. X. Yu, P. Richtárik, A. T. Suresh, and D. Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016.   
[64] A. Korolova, K. Kenthapadi, N. Mishra, and A. Ntoulas. Releasing search queries and clicks privately. In Proceedings of the 18th international conference on World wide web, pages 171–180, 2009.   
[65] N. Li, W. Qardaji, and D. Su. On sampling, anonymization, and differential privacy or, k-anonymization meets differential privacy. In Proceedings of the 7th ACM Symposium on Information, Computer and Communications Security, pages 32–33, 2012.   
[66] Y. Lyubarskii and R. Vershynin. Uncertainty principles and vector quantization. IEEE Transactions on Information Theory, 56(7):3491-3501, 2010.   
[67] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Arcas. Communication-efficient learning of deep networks from decentralized data (2016). arXiv preprint arXiv:1602.05629, 2016.   
[68] I. Mironov. Rényi differential privacy. In 2017 IEEE 30th Computer Security Foundations Symposium (CSF), pages 263-275. IEEE, 2017.   
[69] T. T. Nguyen, X. Xiao, Y. Yang, S. C. Hui, H. Shin, and J. Shin. Collecting and analyzing data from smart device users with local differential privacy, 2016.   
[70] M. Safaryan, E. Shulgin, and P. Richtárik. Uncertainty principle for communication compression in distributed and federated learning and the search for an optimal compressor. arXiv preprint arXiv:2002.08958, 2020.   
[71] A. Shah, W.-N. Chen, J. Balle, P. Kairouz, and L. Theis. Optimal compression of locally differentially private mechanisms. In International Conference on Artificial Intelligence and Statistics, pages 7680-7723. PMLR, 2022.   
[72] C. Studer, W. Yin, and R. G. Baraniuk. Signal representations with minimum $\ell_{\infty}$ -norm. In 2012 50th Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 1270–1277. IEEE, 2012.   
[73] A. T. Suresh, F. X. Yu, S. Kumar, and H. B. McMahan. Distributed mean estimation with limited communication. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, page 3329–3337. JMLR.org, 2017.   
[74] R. Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.   
[75] T. Wang, J. Zhao, X. Yang, and X. Ren. Locally differentially private data collection and analysis. arXiv preprint arXiv:1906.01777, 2019.

[76] J. Wangni, J. Wang, J. Liu, and T. Zhang. Gradient sparsification for communication-efficient distributed optimization. In Advances in Neural Information Processing Systems, pages 1299-1309, 2018.   
[77] S. L. Warner. Randomized response: A survey technique for eliminating evasive answer bias. Journal of the American Statistical Association, 60(309):63–69, 1965.   
[78] W. Wen, C. Xu, F. Yan, C. Wu, Y. Wang, Y. Chen, and H. Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. In Advances in neural information processing systems, pages 1509-1519, 2017.   
[79] M. Ye and A. Barg. Optimal schemes for discrete distribution estimation under local differential privacy. In 2017 IEEE International Symposium on Information Theory (ISIT), pages 759–763, June 2017.   
[80] W. Zhu, P. Kairouz, B. McMahan, H. Sun, and W. Li. Federated heavy hitters discovery with differential privacy. In International Conference on Artificial Intelligence and Statistics, pages 3837-3847. PMLR, 2020.   
[81] Y. Zhu and Y.-X. Wang. Poission subsampled rényi differential privacy. In International Conference on Machine Learning, pages 7634-7642. PMLR, 2019.

# A Proof of Theorem 4.1

It is trivial to see that the average communication cost is $d \cdot \gamma = b$ bits. To compute the $\ell_2^2$ estimation error, observe that

$$
\begin{array}{l} \mathbb {E} \left[ \left\| \hat {\mu} _ {x ^ {n}} - \mu_ {x ^ {n}} \right\| _ {2} ^ {2} \right] \\ = \sum_ {j = 1} ^ {d} \mathbb {E} \left[ \left(\frac {1}{n \gamma} \sum_ {i} x _ {i} (j) \cdot Z _ {i, j} + N (0, \sigma^ {2}) - \frac {1}{n} \sum_ {i} x _ {i} (j)\right) ^ {2} \right] \\ = \sum_ {j = 1} ^ {d} \frac {1}{n ^ {2}} \mathbb {E} \left[ \left(\frac {1}{\gamma} \sum_ {i} x _ {i} (j) \cdot Z _ {i, j} - \sum_ {i} x _ {i} (j)\right) ^ {2} \right] + d \sigma^ {2} \\ = \sum_ {j = 1} ^ {d} \frac {1}{n ^ {2}} \mathbb {E} \left[ \left(\frac {1}{\gamma} \sum_ {i} x _ {i} (j) \cdot Z _ {i, j}\right) ^ {2} \right] - \frac {1}{n ^ {2}} \left(\sum_ {i} x _ {i} (j)\right) ^ {2} + d \sigma^ {2} \\ = \sum_ {j = 1} ^ {d} \frac {1}{n ^ {2}} \mathbb {E} \left[ \frac {1}{\gamma^ {2}} \sum_ {i} x _ {i} ^ {2} (j) \cdot Z _ {i, j} ^ {2} + \frac {1}{\gamma^ {2}} \sum_ {i \neq i ^ {\prime}} x _ {i} (j) x _ {i ^ {\prime}} (j) Z _ {i, j} Z _ {i ^ {\prime}, j} \right] - \frac {1}{n ^ {2}} \left(\sum_ {i} x _ {i} (j)\right) ^ {2} + d \sigma^ {2} \\ = \sum_ {j = 1} ^ {d} \frac {1}{n ^ {2}} \left(\frac {1}{\gamma} \sum_ {i} x _ {i} ^ {2} (j) + \sum_ {i \neq i ^ {\prime}} x _ {i} (j) x _ {i ^ {\prime}} (j)\right) - \frac {1}{n ^ {2}} \left(\sum_ {i} x _ {i} (j)\right) ^ {2} + d \sigma^ {2} \\ = \sum_ {j = 1} ^ {d} \frac {1}{n ^ {2}} \left(\frac {1}{\gamma} - 1\right) \left(\sum_ {i} x _ {i} ^ {2} (j)\right) + d \sigma^ {2} \\ \leq \frac {d c ^ {2}}{n \gamma} + d \sigma^ {2}, \\ \end{array}
$$

which yields the inequality of (2). Next, we analyze the privacy of Algorithm 1. We first the following two lemmas for subsampling and the Gaussian mechanism:

Lemma A.1 ([65, 81]). If $\mathcal{M}$ is $(\varepsilon, \delta)$ -DP, then $\mathcal{M}'$ that applies $\mathcal{M} \circ$ PoissonSample satisfies $(\varepsilon', \delta')$ -DP with $\varepsilon' = \log(1 + \gamma(e^{\varepsilon} - 1))$ and $\delta' = \gamma\delta$ .

Lemma A.2 ([15]). For any $\varepsilon,\delta\in(0,1)$ , the Gaussian output perturbation mechanism with $\sigma^{2}:=\frac{\Delta^{2}2\log(1.25/\delta)}{\varepsilon^{2}}$ satisfies $(\varepsilon,\delta)$ -DP, where $\Delta$ is the $\ell_{2}$ sensitivity of the target function.

Now, we use the above two lemmas to analyze the per-coordinate privacy leakage of Algorithm 1. For simplicity, we analyze the sum of $x_{i}(j)$ 's instead (and normalized it in the last step). Let $S_{j}(x^{n}) := \sum_{i=1}^{n}(x_{i}(j))$ , then clearly the sensitivity of $S_{j}(x^{n})$ is $c$ , so Lemma A.2 implies $S_{j}(x^{n}) + N(0, \sigma_{1}^{2})$ satisfies $(\varepsilon_{1}, \delta_{1})$ -DP if we set $\sigma_{1}^{2} = \frac{2c^{2}\log(1.25 / \delta_{1})}{\varepsilon_{1}^{2}}$ (assuming $\varepsilon_{1} < 1$ ). Next, if applying subsampling before computing the sum, i.e.,

$$
S _ {j} \circ \text { PoissonSample } _ {\gamma} (x ^ {n}) := \sum_ {i = 1} ^ {n} x _ {i} (j) Z _ {i, j},
$$

where $Z_{i,j}\stackrel{\mathrm{i.i.d.}}{\sim}\mathsf{Bern}(1 / \gamma)$ as defined in Algorithm 1, then by Lemma A.1,

$$
S _ {j} \circ \text { PoissonSample } _ {\gamma} (x ^ {n}) + N (0, \sigma_ {1} ^ {2})
$$

satisfies $(\varepsilon_{2},\delta_{2})$ -DP with $\varepsilon_{2}:=\log(1+\gamma(e^{\varepsilon_{1}}-1))=C_{1}\gamma\varepsilon_{1}$ (since we assume $\epsilon_{1}<1$ ) and $\delta_{2}:=\gamma\delta_{1}$ . Equivalently, we have

$$
\left\{ \begin{array}{l} \varepsilon_ {1} = \tilde {C} _ {1} \frac {1}{\gamma} \varepsilon_ {2} \\ \delta_ {1} = \frac {1}{\gamma} \delta_ {2}. \end{array} \right. \tag {9}
$$

Now, since we have established the per-coordinate privacy leakage, we apply the following composition theorem to account for the total privacy budgets.

Theorem A.3. For any $\varepsilon > 0$ , $\delta \in [0,1]$ and $\tilde{\delta} \in (0,1]$ , the class of $(\varepsilon, \delta)$ -DP mechanisms satisfies $(\tilde{\varepsilon}_{\tilde{\delta}}, d\delta + \tilde{\delta})$ -DP under $d$ -fold adaptive composition, for

$$
\tilde {\varepsilon} _ {\tilde {\delta}} = d \varepsilon (e ^ {\varepsilon} - 1) + \varepsilon \sqrt {2 d \log (1 / \tilde {\delta})}.
$$

According Theorem A.3, Algorithm 1 satisfies $(\varepsilon,\delta)$ -DP for

$$
\varepsilon = d \varepsilon_ {2} (e ^ {\varepsilon_ {2}} - 1) + \varepsilon_ {2} \sqrt {2 d \log (1 / \tilde {\delta})}, \tag {10}
$$

and $\delta = d\delta_{2} + \tilde{\delta}$ (where $\tilde{\delta}$ is a free parameter that we can optimize).

Consequently, for a pre-specified (total) privacy budget $(\varepsilon, \delta)$ , we set parameters as follows. Let $\tilde{\delta} = \frac{\delta}{2}$ and $\delta_1 = \frac{1}{\gamma} \delta_2 = \frac{1}{2d\gamma} \delta$ . Let $\varepsilon_2 \leq 1$ so that $e_2^\varepsilon - 1 \leq 2\varepsilon_2$ holds. Then (10) implies Algorithm 1 is

$$
\varepsilon = 2 d \varepsilon_ {2} ^ {2} + \varepsilon_ {2} \sqrt {2 d \log (1 / \tilde {\delta})} \geq d \varepsilon_ {2} (e ^ {\varepsilon_ {2}} - 1) + \varepsilon_ {2} \sqrt {2 d \log (1 / \tilde {\delta})}.
$$

Solving the above quadratic (in-)equality for $\varepsilon_{2}$ , it yields that

$$
\varepsilon_ {2} = \min \left(1, \frac {- \sqrt {2 d \log (2 / \delta)} + \sqrt {2 d \log (2 / \delta) + 8 \varepsilon d}}{4 d}\right) = O \left(\min \left(1, \frac {\varepsilon}{\sqrt {d (\log (1 / \delta) + \varepsilon)}}\right)\right).
$$

Consequently, we set $\varepsilon_1 = \frac{\tilde{C}_1}{\gamma}\varepsilon_2 = O\left(\min \left(1,\frac{\varepsilon}{\gamma\sqrt{d(\log(1 / \delta) + \varepsilon)}}\right)\right)$ (note that we require $\varepsilon_{1} = O(1)$ so that (9) holds).

Plug in $(\varepsilon_1,\delta_1)$ into $\sigma_1^2$ , we have

$$
\sigma_ {1} ^ {2} := \frac {2 c ^ {2} \log (1 . 2 5 / \delta_ {1})}{\varepsilon_ {1} ^ {2}} = \Omega \left(\max \left(c ^ {2} \log (d / \delta), \frac {\gamma^ {2} c ^ {2} d (\log (1 / \delta) + \varepsilon) \log (d / \delta)}{\varepsilon^ {2}}\right)\right).
$$

Finally, as we are interested in estimating the (subsampled) mean instead of the sum, we will normalize the private sum by

$$
\hat {\mu} _ {j} (x ^ {n}) = \frac {1}{n \gamma} \left(S _ {j} \circ \text { PoissonSample } _ {\gamma} (x ^ {n}) + N (0, \sigma_ {1} ^ {2})\right) = \frac {1}{n \gamma} S _ {j} \circ \text { PoissonSample } _ {\gamma} (x ^ {n}) + N (0, \sigma^ {2}),
$$

where

$$
\sigma^ {2} = O \left(\max \left(\frac {c ^ {2} \log (d / \delta)}{n ^ {2} \gamma^ {2}}, \frac {c ^ {2} d (\log (1 / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right)\right).
$$

Plugging in $\sigma^2$ above and $\gamma = d / b$ yields the desired accuracy in Theorem 4.1.

Since we will reuse the above result, we summarize it into the following lemma:

Lemma A.4. Let $f_{i}: R^{d \times m} \mapsto R^{D}$ for $i = 1, \ldots, B$ be n functions with sensitivity bounded by $\Delta$ (where the number of inputs m can be a random variable). Then

$$
\left(f _ {1} \circ \text { PoissonSample } _ {\gamma} (x ^ {n}) + N (0, \sigma^ {2}),..., f _ {B} \circ \text { PoissonSample } _ {\gamma} (x ^ {n}) + N (0, \sigma^ {2})\right)
$$

satisfies $(\varepsilon, \delta)$ -DP, if

$$
\sigma^ {2} \geq O \left(\max \left(\Delta^ {2} \log (B / \delta), \frac {\gamma^ {2} \Delta^ {2} B (\log (1 / \delta) + \varepsilon) \log (B / \delta)}{\varepsilon^ {2}}\right)\right).
$$

# B Proof of Theorem 4.4

To prove Theorem 4.4, it suffices to prove the following $\ell_{\infty}$ version:

Theorem B.1. Let $x_{1}, \ldots, x_{n} \in \{-c, c\}^{d}$ , $d' = \min \left( nb, \frac{n^{2}\varepsilon^{2}}{(\log(1/\delta) + \varepsilon)\log(d/\delta)} \right)$ , and

$$
\sigma^ {2} = O \left(\frac {c ^ {2} \log (1 / \delta)}{n ^ {2} \gamma^ {2}} + \frac {c ^ {2} d ^ {\prime} (\log (d ^ {\prime} / \delta) + \varepsilon) \log (d ^ {\prime} / \delta)}{n ^ {2} \varepsilon^ {2}}\right). \tag {11}
$$

Then Algorithm 2 is $(\varepsilon, \delta)$ -DP and yields an unbiased estimator on $\mu$ . In addition, the (average) per-client communication cost is $\gamma d' = b$ bits, and the $\ell_2^2$ estimation error is at most

$$
O \left(c ^ {2} d ^ {2} \log \left(\frac {d}{\delta}\right) \max \left(\frac {1}{n b}, \frac {\left(\log (1 / \delta) + \varepsilon\right)}{n ^ {2} \varepsilon^ {2}}\right)\right). \tag {12}
$$

With a slight abuse of notation, we let $\mu_{\mathcal{J}} \in \mathbb{R}^d$ be such that

$$
\mu_ {\mathcal {J}} (j) = \left\{ \begin{array}{l l} 0, & \text {if} j \notin \mathcal {J} \\ \frac {d \mu_ {j}}{d ^ {\prime}}, & \text {else.} \end{array} \right.
$$

Note that $\mu_{\mathcal{J}}$ is an unbiased estimate of $\mu$ if $\mathcal{J}$ is selected uniformly at random. Then the $\ell_2^2$ error can be controlled by

$$
\mathbb {E} \left[ \| \mu - \hat {\mu} \| _ {2} ^ {2} \right] \stackrel {\mathrm{(a)}} {=} \mathbb {E} \left[ \| \mu - \mu_ {\mathcal {J}} \| _ {2} ^ {2} \right] + \mathbb {E} \left[ \| \mu_ {\mathcal {J}} - \hat {\mu} \| _ {2} ^ {2} \right]
$$

$$
\stackrel {{(\mathrm{b})}} {{\leq}} \mathbb {E} \left[ \| \mu - \mu_ {\mathcal {J}} \| _ {2} ^ {2} \right] + \frac {d ^ {2}}{d ^ {\prime 2}} O \left(\max \left(\frac {d ^ {\prime 2} c ^ {2}}{n b}, \frac {d ^ {\prime 3} c ^ {2} \log (d / \delta)}{n ^ {2} b ^ {2}}, \frac {c ^ {2} d ^ {\prime 2} (\log (1 / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right)\right)
$$

$$
= \mathbb {E} \left[ \| \mu - \mu_ {\mathcal {J}} \| _ {2} ^ {2} \right] + O \left(\max \left(\frac {d ^ {2} c ^ {2}}{n b}, \frac {d ^ {2} d ^ {\prime} c ^ {2} \log (d / \delta)}{n ^ {2} b ^ {2}}, \frac {c ^ {2} d ^ {2} (\log (1 / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right)\right)
$$

$$
\stackrel {(c)} {\leq} \frac {d ^ {2} c ^ {2}}{d ^ {\prime}} + O \left(\max \left(\frac {d ^ {2} c ^ {2}}{n b}, \frac {d ^ {2} d ^ {\prime} c ^ {2} \log (d / \delta)}{n ^ {2} b ^ {2}}, \frac {c ^ {2} d ^ {2} (\log (1 / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right)\right),
$$

where (a) holds since $\mu_{\mathcal{J}}$ is an unbiased estimate of $\mu$ and conditioned on $\mathcal{J}$ , $\hat{\mu}$ is an unbaised estimate of $\mu_{\mathcal{J}}$ ; (b) follows from Theorem 4.1; (c) holds due to the following fact:

$$
\mathbb {E} \left[ \| \mu - \mu_ {\mathcal {J}} \| _ {2} ^ {2} \right] \leq \sum_ {j \in \mathcal {J}} \mu_ {\mathcal {J}} (j) ^ {2} + \sum_ {j \in [ d ]} \mu_ {j} ^ {2} \leq \frac {d ^ {2} c ^ {2}}{d ^ {\prime}} + d c ^ {2} \leq \frac {2 d ^ {2} c ^ {2}}{d ^ {\prime}}.
$$

Therefore, by setting $d' = \min \left( nb, \frac{n^2\varepsilon^2}{(\log(1/\delta) + \varepsilon)\log(d/\delta)} \right)$ we ensure the first term in (c) is always smaller than the second term, and the second term can be simplified as follows:

$$
O \left(c ^ {2} d ^ {2} \max \left(\frac {1}{n b}, \frac {d ^ {\prime} \log (d / \delta)}{n ^ {2} b ^ {2}}, \frac {(\log (1 / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right)\right)
$$

$$
\leq O \left(c ^ {2} d ^ {2} \max \left(\frac {1}{n b}, \frac {n b \log (d / \delta)}{n ^ {2} b ^ {2}}, \frac {(\log (1 / \delta) + \varepsilon) \log (d / \delta)}{n ^ {2} \varepsilon^ {2}}\right)\right)
$$

$$
\leq O \left(c ^ {2} d ^ {2} \log (d / \delta) \max \left(\frac {1}{n b}, \frac {(\log (1 / \delta) + \varepsilon)}{n ^ {2} \varepsilon^ {2}}\right)\right).
$$

Finally, applying the same trick of Kashin's representation, we can transform the $\ell_{\infty}$ geometry to $\ell_2$ (similar to Proposition 4.3), hence proving Theorem 4.4.

# C Proof of Theorem 5.1

Let $\pi := \frac{1}{n} \sum_{i} x_{i}$ and $\pi^{(\ell)}$ be defined in the same way as $x_{i}^{(\ell)}$ for $\ell \in [B]$ . Then our goal is to bound $|\pi^{(\ell)}(j) - \hat{\pi}^{(\ell)}(j)|$ , for all $\ell \in [2^{b-1}]$ and $j \in [B]$ .

To this end, let $y^{(\ell)} := H_{B} \cdot \pi^{(\ell)}$ (so it holds that $\pi^{(\ell)} = \frac{1}{B} H_{B} \cdot y^{(\ell)}$ ). Then we have

$$
\begin{array}{l} \mathbb {E} \left[ \left| \pi^ {(\ell)} (j) - \hat {\pi} ^ {(\ell)} (j) \right| \right] \stackrel {\mathrm{(a)}} {\leq} \sqrt {\mathbb {E} \left[ \left(\pi^ {(\ell)} (j) - \hat {\pi} ^ {(\ell)} (j)\right) ^ {2} \right]} \\ = \sqrt {\mathbb {E} \left[ \left(\frac {1}{B} H _ {B} \cdot \left(y ^ {(\ell)} - \hat {y} ^ {(\ell)}\right) (j)\right) ^ {2} \right]}. \tag {13} \\ \end{array}
$$

Next, observe that due to the subsampling step, for all $\ell\in[2^{b-1}]$ and $j\in[B]$ ,

$$
\hat {y} ^ {(\ell)} (j) = \frac {B}{n} \sum_ {i = 1} ^ {n} \langle (H _ {B}) _ {j}, x _ {i} ^ {(\ell)} \rangle \cdot Z _ {i j} + N (0, \sigma^ {2}),
$$

where recall that $Z_{ij} \stackrel{\text{i.i.d.}}{\sim} \mathsf{Ber}(1/B)$ . Therefore, $\hat{y}^{(\ell)}(j)$ is an unbiased estimator of $y^{(\ell)}(j)$ . In addition, since we choose $Z_{ij}$ independently in Algorithm 3, $\hat{y}^{(\ell)}(j)$ 's are independent for different j's, so we have

$$
\begin{array}{l} \mathbb {E} \left[ \left(\hat {y} ^ {(\ell)} (j) - y ^ {(\ell)} (j)\right) ^ {2} \right] = \operatorname{Var} \left(\hat {y} ^ {(\ell)} (j)\right) \\ = \sigma^ {2} + \frac {B ^ {2}}{n ^ {2}} \sum_ {i = 1} ^ {n} \left\langle \left(H _ {B}\right) _ {j}, x _ {i} ^ {(\ell)} \right\rangle^ {2} \operatorname{Var} \left(Z _ {i j}\right) \\ \leq \sigma^ {2} + \frac {B}{n ^ {2}} \sum_ {i = 1} ^ {n} \langle (H _ {B}) _ {j}, x _ {i} ^ {(\ell)} \rangle^ {2} \\ = \sigma^ {2} + \frac {B}{n ^ {2}} \underbrace {\sum_ {i = 1} ^ {n} \mathbb {1} _ {\{x _ {i} \in \ell - \text {th chunk} \}}} _ {:= C _ {\ell}}, \tag {14} \\ \end{array}
$$

and for all $j \neq j'$

$$
\mathbb {E} \left[ \left(\hat {y} ^ {(\ell)} (j) - y ^ {(\ell)} (j)\right) \cdot \left(\hat {y} ^ {(\ell)} (j ^ {\prime}) - y ^ {(\ell)} (j ^ {\prime})\right) \right] = 0. \tag {15}
$$

Therefore, we continue bounding (13) as follows:

$$
\begin{array}{l} \sqrt {\mathbb {E} \left[ \left(\frac {1}{B} H _ {B} \cdot (y ^ {(\ell)} - \hat {y} ^ {(\ell)}) (j)\right) ^ {2} \right]} = \sqrt {\frac {1}{B ^ {2}} \mathbb {E} \left[ \langle (H _ {B}) _ {j} , (\hat {y} ^ {(\ell)} - y ^ {(\ell)}) \rangle^ {2} \right]} \\ = \sqrt {\frac {1}{B ^ {2}} \mathbb {E} \left[ \left(\sum_ {k = 1} ^ {B} (H _ {B}) _ {j k} \cdot (\hat {y} ^ {(\ell)} (k) - y ^ {(\ell)} (k))\right) ^ {2} \right]} \\ \stackrel {(a)} {=} \sqrt {\frac {1}{B ^ {2}} \mathbb {E} \left[ \sum_ {k = 1} ^ {B} \left(\hat {y} ^ {(\ell)} (k) - y ^ {(\ell)} (k)\right) ^ {2} \right]} \\ \stackrel {\mathrm{(b)}} {=} \sqrt {\frac {C _ {\ell}}{n ^ {2}} + \frac {\sigma^ {2}}{B}} \\ \stackrel {\mathrm{(c)}} {\leq} \sqrt {\frac {1}{n} + \frac {\sigma^ {2}}{B}}, \\ \end{array}
$$

where (a) holds since each entry of $H_B$ takes value in $\{-1, 1\}$ and by (15), (b) holds due to (14), and (c) holds because $C_\ell \leq n$ for all $\ell$ .

Finally, to bound the $\ell_2^2$ error, observe that the above analysis ensures that

$$
\mathbb {E} \left[ \left(\pi^ {(\ell)} (j) - \hat {\pi} ^ {(\ell)} (j)\right) ^ {2} \right] \leq \frac {C _ {\ell (j)}}{n ^ {2}} + \frac {\sigma^ {2}}{B},
$$

where $\ell(j) \in [2^{b-1}]$ is the index of the chuck containing $j$ . Therefore, summing over $j \in [d]$ , we must have

$$
\mathbb {E} \left[ \left\| \pi^ {(\ell)} - \hat {\pi} ^ {(\ell)} \right\| _ {2} ^ {2} \right] \leq \sum_ {j = 1} ^ {d} \frac {C _ {\ell (j)}}{n ^ {2}} + \frac {d \sigma^ {2}}{B} = \frac {B}{n} + \frac {d \sigma^ {2}}{B},
$$

since

$$
\sum_ {j} C _ {\ell (j)} = \sum_ {\ell = 1} ^ {2 ^ {b - 1}} \sum_ {j ^ {\prime} \in \ell \text {-th chunk}} \sum_ {i = 1} ^ {n} \mathbb {1} _ {\{i \in \ell \text {-th chunk} \}} = B \sum_ {\ell = 1} ^ {2 ^ {b - 1}} \sum_ {i = 1} ^ {n} \mathbb {1} _ {\{i \in \ell \text {-th chunk} \}} = B \cdot n.
$$

![](images/e9d38bbc3eff0ebb39679542273664a9e12a70b7b55b64c6f16cda5cf440e1d0.jpg)

# D Proof of Theorem 5.2

Let $f_{j}(x^{n}):=(\pi^{(1)}(j),\ldots,\pi^{(2^{b-1})}(j))$ , for $j=1,\ldots,B$ . Then the $\ell_{2}$ sensitivity of $f_{j}$ is $\Delta=\frac{B}{n}$ . Set the sampling rate $\gamma=\frac{1}{B}$ and the proof is complete by Lemma A.4.

# E Algorithm of Shuffled SQKR

Algorithm 4 Shuffled SQKR   
Input: users' data $x_1, \ldots, x_n$ , local-DP parameter $\varepsilon_0$ , communication parameters $b_0, T$ Output: mean estimator $\hat{\mu}$ for round $k \in [T]$ do  
    for user $i \in [n]$ do  
    Sample $s(i, 1), \ldots, s(i, b_0) \stackrel{\text{i.i.d.}}{\sim} \text{Unif}[d]$ Sample $Z \sim \text{Bern}\left(\frac{e^{\varepsilon_0}}{e^{\varepsilon_0} + 2^{b_0} - 1}\right)$ if $Z = 1$ then  
    Set $Y(i, 1), \ldots, Y(i, b_0) \leftarrow x_i(s(i, 1)), \ldots, x_i(s(i, b_0))$ else  
    Sample $Y(i, 1), \ldots, Y(i, b_0) \stackrel{\text{i.i.d.}}{\sim} \text{Unif}\{-c, c\}$ end if  
    Send $Y(i, 1), \ldots, Y(i, b_0)$ and $s(i, 1), \ldots, s(i, b_0)$ to shuffler  
end for  
Shuffler samples a permutation $\pi \sim \text{Unif}\{f : [n] \to [n]$ bijective}  
for $j \in [b_0]$ do  
    Shuffler sends $Y(\pi(1), j), \ldots, Y(\pi(n), j)$ and $s(\pi(1), j), \ldots, s(\pi(n), j)$ to server  
end for $\hat{\mu}^{(k)} \leftarrow \frac{d}{nb_0} \frac{e^{\varepsilon_0 + 2^{b_0} - 1}}{e^{\varepsilon_0 - 1}} \sum_{i=1}^{n} \sum_{j=1}^{b_0} Y(\pi(i), j)e_{s(\pi(i), j)}$ end for  
Return $\hat{\mu} := \frac{1}{T} \sum_{k=1}^{T} \hat{\mu}^{(k)}$

# F Proof of Theorem 6.4

Each round $x^{n} \mapsto \hat{\mu}^{(k)}$ of Algorithm 4 implements the private-coin SQKR scheme of [32], achieving the communication cost and error as stated in Lemma 6.3.

Lemma F.1 (SQKR [32]). For all $\varepsilon_{0}>0, b_{0}>0$ , the random mapping $x_{i}\mapsto y(i,1),\ldots,y(i,b_{0}),s(i,1),\ldots,s(i,b_{0})$ in Algorithm 4 is $(\varepsilon_{0},0)$ -LDP and has output that can be communicated with $b_{0}\log(d)$ bits, and the $\hat{\mu}^{(k)}$ computed from $y(i,1),\ldots,y(i,b_{0}),s(i,1),\ldots,s(i,b_{0})$ is an unbiased estimator satisfying

$$
\max _ {x ^ {n}} \mathbb {E} \left[ \left\| \mu (x ^ {n}) - \hat {\mu} ^ {(k)} (x ^ {n}) \right\| _ {2} ^ {2} \right] = O \left(\frac {c ^ {2} d}{n \min (\varepsilon_ {0} ^ {2} , \varepsilon_ {0} , b _ {0})}\right). \tag {16}
$$

We now characterize the error performance of Algorithm 4 for general choices of parameters that satisfy privacy and communication constraints.

Proposition F.2. For all $\varepsilon > 0, b > 0, n > 0$ , with any arbitrary choice of

$$
\delta_ {1} \in (e ^ {- n}, 1 ] \tag {17}
$$

$$
\delta_ {2} \in (0, 1 ], \tag {18}
$$

there exists a choice of parameters $\varepsilon_0, b_0, T$ such that Algorithm 4 is $(\varepsilon, T\delta_1 + \delta_2)$ -DP, uses no more than $b$ bits of communication, and

$$
\max _ {x ^ {n}} \mathbb {E} \left[ \| \mu - \hat {\mu} \| _ {2} ^ {2} \right] = O \left(\max \left(\frac {c ^ {2} d \log (d) b _ {0}}{n b}, \frac {c ^ {2} d \log \left(1 / \delta_ {1}\right) (\log \left(1 / \delta_ {2}\right) + \varepsilon)}{n ^ {2} \varepsilon^ {2}}\right)\right). \tag {19}
$$

Proof. For arbitrary choice of

$$
b _ {0} <   \log \left(\frac {n}{1 6 \log (2)}\right), \tag {20}
$$

it suffices to choose

$$
T = \left\lfloor \frac {b}{(\log_ {2} (d) + 1) b _ {0}} \right\rfloor \tag {21}
$$

$$
\varepsilon_ {0} = O \left(\min \left(1, \frac {\varepsilon \sqrt {n}}{\sqrt {T \log (1 / \delta_ {1}) (\log (1 / \delta_ {2}) + \varepsilon)}}\right)\right). \tag {22}
$$

Since it takes $b_{0}$ bits to send $y(i,1),\ldots,y(i,b_{0})$ and $\log_{2}(d)$ bits to send each of $s(i,1),\ldots,s(i,b_{0})$ , and this is done T times, Algorithm 4 using less than b bits is immediate from the choice of T.

Applying Lemma F.1, by construction the mapping from each $x_{i}$ to $y(i,1),\ldots ,y(i,b_0)$ is $(\varepsilon_0,0)$ -LDP. By assumption

$$
\delta_ {1} > e ^ {- n / 1 6 e} > e ^ {- n}, \tag {23}
$$

the inequality

$$
1 <   \log \left(\frac {n}{1 6 \log (2 / \delta_ {1})}\right) \tag {24}
$$

is satisfied. Then the choice of

$$
\varepsilon_ {0} \leq 1 \tag {25}
$$

also satisfies $\varepsilon_0 \leq \log \left( \frac{n}{16\log(2/\delta)} \right)$ , so by Lemma 6.1 the mapping $x^n \mapsto \hat{\mu}^{(k)}$ is $(\varepsilon_1, \delta_1)$ -DP. where

$$
\varepsilon_ {1} = O \left(\frac {\varepsilon_ {0} \sqrt {\log (1 / \delta_ {1})}}{\sqrt {n}}\right). \tag {26}
$$

Since the output of Algorithm 4 is a function of $\left(\hat{\mu}^{(1)},\dots ,\hat{\mu}^{(T)}\right)$ , by A.3 it suffices to have

$$
\varepsilon_ {1} = O \left(\min \left(1, \frac {\varepsilon}{\sqrt {T (\log (1 / \delta_ {2}) + \varepsilon)}}\right)\right) \tag {27}
$$

for Algorithm 4 to be $(\varepsilon, T\delta_{1} + \delta_{2})$ -DP. The first inequality follows from the assumption of $\delta_{1} > e^{-n}$ and choice of $\varepsilon_{0} = O(1)$ , and the second from choice of

$$
\varepsilon_ {0} = O \left(\frac {\varepsilon \sqrt {n}}{\sqrt {T \log (1 / \delta_ {1}) (\log (1 / \delta_ {2}) + \varepsilon)}}\right). \tag {28}
$$

Since $\varepsilon_{0}\leq1\leq b$ , we have $\min(\varepsilon_{0}^{2},\varepsilon_{0},b)=\varepsilon_{0}^{2}$ . Applying Lemma F.1,

$$
\max _ {x ^ {n}} \mathbb {E} \left[ \| \mu - \hat {\mu} \| _ {2} ^ {2} \right] = \frac {1}{T} \max _ {x ^ {n}} \mathbb {E} \left[ \left\| \mu - \hat {\mu} ^ {(1)} \right\| _ {2} ^ {2} \right] \tag {29}
$$

$$
= O \left(\frac {d}{T n \varepsilon_ {0} ^ {2}}\right) \tag {30}
$$

$$
= O \left(\max \left(\frac {d}{T n}, \frac {d \log (1 / \delta_ {1}) (\log (1 / \delta_ {2}) + \varepsilon)}{n ^ {2} \varepsilon^ {2}}\right)\right). \tag {31}
$$

Substituting the choice of T gives the desired result.

![](images/1298defe1d1a7996fbbcfa50b53354a2528531e9d6b57a604ed9fa826f49a9ae.jpg)

To show Theorem 6.4, it suffices to choose

$$
b _ {0} = 1 \tag {32}
$$

$$
\delta_ {1} = \frac {\delta}{2 T} \tag {33}
$$

$$
\delta_ {2} = \frac {\delta}{2}, \tag {34}
$$

which requires $n > 16e \log(2) \approx 30.14$ due to (20), and apply the previous proposition.

# G Rényi-DP for Shuffled SQKR

We can use the following result for Rényi-DP (RDP) guarantees for Algorithm 4.

Lemma G.1 ([44] Corollary 4.3). Let $\mathcal{M}_i$ be an independent $(\varepsilon_0,0)$ -LDP mechanism for each $i\in [n]$ with $\varepsilon_0\leq 1$ and $\pi$ be a random permutation of $[n]$ . Then for any $\alpha < \frac{n}{16\varepsilon_0\exp(\varepsilon_0)}$ , the mechanism

$$
\mathcal {S}: (x _ {1}, \ldots , x _ {n}) \mapsto \left(\mathcal {M} _ {1} \left(x _ {\pi (1)}\right), \ldots , \mathcal {M} _ {n} \left(x _ {\pi (n)}\right)\right)
$$

is $(\varepsilon (\alpha),\delta)$ -RDP where

$$
\varepsilon (\alpha) = O \left(\alpha \left(1 - e ^ {- \varepsilon_ {0}}\right) ^ {2} \frac {e ^ {\varepsilon_ {0}}}{n}\right). \tag {35}
$$

Applying Lemma F.1, by construction the mapping from each $x_{i}$ to $y(i,1),\ldots,y(i,b_{0})$ is $(\varepsilon_{0},0)$ -LDP. By Lemma G.1, the mapping $x^{n}\mapsto\hat{\mu}^{(k)}$ is $(\varepsilon_{1},\alpha)$ -RDP where

$$
\varepsilon_ {1} = O \left(\alpha \left(1 - e ^ {- \varepsilon_ {0}}\right) ^ {2} \frac {e ^ {\varepsilon_ {0}}}{n}\right) \tag {36}
$$

By composition, Algorithm 4 is $(T\varepsilon_{1},\alpha)$ -RDP.