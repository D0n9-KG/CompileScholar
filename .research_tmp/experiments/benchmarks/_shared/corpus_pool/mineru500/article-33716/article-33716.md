# Accelerated Methods with Compressed Communications for Distributed Optimization Problems under Data Similarity

Dmitry Bylinkin $^{1,2}$ , Aleksandr Beznosikov $^{2, 1, 3, 4}$

$^{1}$ Moscow Institute of Physics and Technology

$^{2}$ Ivannikov Institute for System Programming of the Russian Academy of Sciences

$^{3}$ Sber AI Lab

$^{4}$ Skoltech

da.bylinkin@gmail.com, anbeznosikov@gmail.com

# Abstract

In recent years, as data and problem sizes have increased, distributed learning has become an essential tool for training high-performance models. However, the communication bottleneck, especially for high-dimensional data, is a challenge. Several techniques have been developed to overcome this problem. These include communication compression and implementation of local steps, which work particularly well when there is similarity of local data samples. In this paper, we study the synergy of these approaches for efficient distributed optimization. We propose the first theoretically grounded accelerated algorithms utilizing unbiased and biased compression under data similarity, leveraging variance reduction and error feedback frameworks. In terms of communication time our theory gives $\tilde{\mathcal{O}}\left(1+\left[M^{-1/4}+\omega^{-1/2}\right]\sqrt{\delta/\mu}\right)$ complexity for unbiased compressors and $\tilde{\mathcal{O}}\left(1+\beta^{1/4}\sqrt{\delta/\mu}\right)$ for biased ones, where M is the number of computational nodes, $\beta$ is the compression power, $\delta$ is the similarity measure and $\mu$ is the parameter of strong convexity of the objective. Our theoretical results are of record and confirmed by experiments on different average losses and datasets.

# 1 Introduction

Conventional machine/deep learning algorithms often struggle to handle the scale and complexity of modern datasets, resulting in long training times and limited scalability. Distributed optimization (Verbraeken et al. 2020) has witnessed remarkable progress, driven by rising demand for efficient methods across diverse applications such as medical image analysis (Wu et al. 2022), chemical physics (Zhu, Luo, and White 2022), predictive maintenance (Bidollahkhani and Kunkel 2024) and natural language processing (Bai 2022). Distributed learning addresses emerging challenges by spreading the training process across multiple nodes, allowing scientists and engineers to process and analyze data that would be impractical to handle on a single machine (Alqahtani and Demirbas 2019). In terms of optimization, we have the following problem statement:

$$
\min _ {x \in \mathbb {R} ^ {d}} \left[ f (x) = \frac {1}{M} \sum_ {m = 1} ^ {M} f _ {m} (x) \right] \tag {1}
$$

$$
\text { with } \quad f _ {m} (x) = \frac {1}{n _ {m}} \sum_ {j = 1} ^ {n _ {m}} \ell (x, z _ {j} ^ {m}),
$$

where M refers to the number of nodes/devices/clients/agents/machines, $n_{m}$ is the size of the local dataset on m-th machine, x is the vector representation of the model using d features, $z_{j}^{m}$ is the j-th data point on the m-th node and $\ell$ is the loss function. $z_{j}^{m}$ is an ordered pair of feature description $a_{j}^{i}$ and label $b_{j}^{m}$ . We consider an architecture with a star-network topology, that is, a server, represented by $f_{1}$ , plays a crucial role as the main computational hub. The other nodes act as users that can communicate with the server but not directly with each other. This hierarchical structure allows for easy coordination through the first node within the system. When training modern distributed models, the bottleneck is often the cost of communicating information (Konečný et al. 2016; Jordan, Lee, and Yang 2019). It is a known fact that client-to-server communication is much more resource-intensive than server-to-client (Mishchenko et al. 2019; Kairouz et al. 2021). As a result, considerable effort has been devoted to development of distributed optimization methods that would be efficient in terms of nodes to server communication. Significant success has been achieved with the development of compression techniques, that help to reduce the amount of data transferred between nodes during training. In this paper, we deal with two main classes of compressors: unbiased and biased (Beznosikov et al. 2023).

Definition 1. We call the mapping $Q: \mathbb{R}^{d} \to \mathbb{R}^{d}$ an unbiased compressor, if there exists a constant $\omega > 1$ such that

$$
\mathbb {E} _ {Q} [ Q (x) ] = x, \quad \mathbb {E} _ {Q} \left[ \| Q (x) - x \| ^ {2} \right] \leq \omega \| x \| ^ {2}, \quad \forall x \in \mathbb {R} ^ {d}.
$$

Definition 2. We call the mapping $C: \mathbb{R}^{d} \to \mathbb{R}^{d}$ a biased compressor, if there exists a constant $\beta > 1$ such that

$$
\mathbb {E} _ {C} \left[ \| C (x) - x \| ^ {2} \right] \leq \left(1 - \frac {1}{\beta}\right) \| x \| ^ {2}, \quad \forall x \in \mathbb {R} ^ {d}.
$$

Let us denote by $\gamma_{\omega}$ the value showing how much the operator $Q(z)$ compresses the input vector on average. Let b be the number of bits needed to represent a single float, and $\|\cdot\|_{bits}$ be the number of bits needed to represent the input vector. We define $\gamma_{\omega}^{-1} = \frac{1}{bd}\mathbb{E}\|Q(z)\|_{bits}$ . Similarly, we introduce the compressive force $\gamma_{\beta}$ for $C(z)$ . For practical compressors, we have $\gamma_{\omega} \geq \omega$ and $\gamma_{\beta} \geq \beta$ (Vogels, Karimireddy, and Jaggi

2019; Beznosikov et al. 2023). While methods with unbiased compressors may theoretically provide more optimistic estimates (Gorbunov, Hanzely, and Richtárik 2020; Gorbunov et al. 2021; Li et al. 2020), biased ones show a notable advantage in practical applications (Sun et al. 2019). Consequently, there is a growing interest in biased compression techniques. However, analyzing biased compressors is challenging, and there is limited understanding of their behavior (Beznosikov et al. 2023; Richtárik, Sokolov, and Fatkhullin 2021; Stich and Karimireddy 2019).

Another technique that addresses the communication bottleneck in distributed optimization is utilizing local steps under the Hessian similarity condition (Shamir, Srebro, and Zhang 2014; Hendrikx et al. 2020; Kovalev et al. 2022). In this scenario, a single node represents the average nature of the data across all ones.

Definition 3. We say that there is the Hessian similarity ( $\delta$ -relatedness) between $f_{i}$ and f, if there exists a constant $\delta > 0$ such that

$$
\| \nabla^ {2} f _ {i} (x) - \nabla^ {2} f (x) \| \leq \delta , \quad \forall x \in \mathbb {R} ^ {d}.
$$

Consider f to be L-smooth. As the size n of data, distributed uniformly across the nodes, increases, the losses become more statistically similar. In the quadratic case, it is shown that $\delta \sim L/n$ . Otherwise, we have $\delta \sim L/\sqrt{n}$ for any non-quadratic losses (Hendrikx et al. 2020). As a result, the server gets to contact the clients less often to compute the full gradient. Despite the fact that research on distributed optimization via compression or similarity has been going on for quite some time, there are still open questions:

1. How to build accelerated methods that use both compression and similarity?   
2. Would such methods be superior to existing SOTAs in terms of communication efficiency?

# 2 Notation

When we talk about the communication efficiency of an algorithm, it is important to choose the right definition of its communication complexity. This can be done in several ways.

1. Number of communication rounds (CC-1). The number of times the server initiates communication with clients is used as a complexity measure. This does not take into account the number of involved machines. Even if the server has communicated with all the devices within a communication round, this counts the same as if it only has communicated with one.   
2. Number of client-server communications (CC-2). It arises when we recognize that the number of rounds of communication is not sufficient to adequately compare distributed methods. For example, if the nodes operate asynchronously. In this case, the more appropriate metric is the total number of communications rather than the number of rounds. Using this approach, we begin to experience the superiority of methods that turn out to be bad in the sense of CC-1.

3. Communication time (CC-3). We consider a synchronous setup. Let all the devices and their communication channels to the server be equivalent. Let transmission time of one unit of information from the devices to the server take $\tau$ time units. Also, if the clients start communicating with the server at same time, the channel between them is initialized in negligible time. Then, the total time of one such round of communication is $\tau K$ , where K is the amount of information transmitted from each machine to the server. This definition allows us to see the strengths of methods with compressed communications, since they can change K.

# 3 Related Works

# 3.1 Distributed Learning via Hessian Similarity

Now that the communication complexity is suitably defined, let us move on to the survey on effective communication techniques. The first work around similarity was the Newton-type DANE method, developed for quadratic strongly convex functions (Shamir, Srebro, and Zhang 2014). For this class of problems, a CC-1 lower bound was proved in (Arjevani and Shamir 2015). DANE did not reach it. This raised the question of how to fill this gap. History of work on this problem counts many papers. Nevertheless, all of them either did not reach exactly the bound or considered special cases (Zhang and Lin 2015; Lu, Freund, and Nesterov 2018; Yuan and Li 2020; Beznosikov et al. 2021; Tian et al. 2022). Recently, Accelerated ExtraGradient enjoying optimal CC-1 communication complexity under the Hessian similarity condition was constructed in (Kovalev et al. 2022). Exploiting the similarity is not the only approach to effective communication. There are also the compression techniques, to which we devote the next subsection.

# 3.2 Distributed Learning via Compression

There has been a significant amount of research done on communication compression. Especially on unbiased operators, due to their ease of analysis. The first family of compression schemes with convergence guarantees was proposed in one-device setup by Alistarh et al. (2017). It was suggested to perform gradient descent steps using compressed gradients. An extension of the proposed approach to multiple nodes was made a year later, when the first method DQGD for the distributed setup appeared (Khirirat, Feyzmahdavian, and Johansson 2018). The following was suggested: the node gets the current point, calculates the local gradient, then compresses and sends it to the server to perform the gradient step. This scheme is quite simple and easy to analyze, but it has a number of drawbacks, which is discussed below. A comprehensive review of unbiased compression can be found in (He, Huang, and Yuan 2024). As noted before, the nature of unbiased compression is quite simple and one can design methods simply by replacing the stochastic gradient with a compressed one. At the same time, biased compression shows better empirical results (Vogels, Karimireddy, and Jaggi 2019), but its nature is non-trivial and requires a special approach. In 2014, a framework for constructing methods with biased compression was proposed (Seide et al. 2014).

The key idea is that each node remembers the "error" it made when compressing the local gradient, and then takes it into account in a special way in the next rounds. This approach is called Error Feedback. There were no theoretically proven results without unnatural assumptions until (Stich and Karimireddy 2019; Beznosikov et al. 2023). In all mentioned papers both for unbiased and biased compression a similar problem arisen – none of the named algorithms converged to the true optimum. The reason is uncontrolled variance of compressed gradient difference, which results in its approximation not tending to zero as the algorithm runs. After reaching some neighborhood of the solution, the method loses the ability to take small enough steps to avoid “overshooting” the optimum. This could be solved by the variance reduction (VR) technique.

# 3.3 Variance Reduction

Originally, variance reduction was designed to solve the convergence problem of SGD (Robbins and Monro 1951). Classical stochastic optimization methods such as SGD face the same problem as the simplest distributed methods with compression: approximation of the gradient does not tend to zero when searching for the optimum. Thus, there is convergence to its neighborhood only. The variance reduction framework allows to correct this drawback of naive methods. There are two main approaches: SAGA (Defazio, Bach, and Lacoste-Julien 2014) and SVRG (Johnson and Zhang 2013). The first stores the history of evoked gradients by mini-batches, resulting in a more accurate approximation of the full gradient. The second has no "memory", but occasionally recalculates the full gradient. The optimal stochastic optimization algorithm with variance reduction is KATYUSHA (Allen-Zhu 2018a). Its various modifications are of great interest (Kovalev, Horváth, and Richtárik 2020; Allen-Zhu 2018b). These ideas could be developed to solve problems arising in the analysis of methods utilizing unbiased compression. For example, see DIANA (Mishchenko et al. 2019) and its accelerated version ADIANA (Li et al. 2020). In the non-convex case, the current state-of-the-art MARINA is also based on the VR idea (Gorbunov et al. 2021). Variance reduction can be exploited in biased compression methods as well, see EF21 (Richtárik, Sokolov, and Fatkhullin 2021), ECLK (Qian, Richtárik, and Zhang 2021). It is also known how to construct variance reduction schemes for a more general class of problems than minimization – variational inequalities (VIs) (Alacaoglu and Malitsky 2022). This approach is useful for developing methods that combine compression with other approaches, such as local steps and similarity (Beznosikov, Takác, and Gasnikov 2024; Beznosikov et al. 2022). Since our goal is to utilize the synergy of multiple techniques to develop methods for effective communication, we devote the next subsection to a brief overview and comparison of the best existing methods.

# 3.4 Effective Communications via Combination of Techniques

One of the modern methods combining different techniques to communicate efficiently is LoCoDL (Condat, Maranjyan, and Richtárik 2024). Using compression and local steps, it is possible to construct a method that, in the case of $\omega > M$ , repeats the result of ADIANA in the sense of CC-1, and outperforms it in the sense of CC-2 and CC-3. The authors of (Condat et al. 2023) managed to add client sampling to this combination of techniques. Their TAMUNA gives stronger results in some special cases. The main weakness of the above methods is that they do not exploit data similarity. Thus, if the ratio $\delta/L$ is small enough, these methods lose significantly to SOTAs that use similarity. Accelerated Extragradient (Kovalev et al. 2022) is unbeatable in the sense of CC-1 and superior to LoCoDL and TAMUNA in the sense of CC-2 and CC-3. These results are achieved by combining similarity and local steps. AccSVRS (Lin et al. 2024), which uses client sampling in addition to these techniques, loses in the sense of CC-1. However, in the sense of CC-2, this method is optimal. One can note the current best methods in terms of CC-3 are Accelerated ExtraGradient and Three Pillars Algorithm. The first one is accelerated, but does not use compression. The second one uses unbiased compression but does not use acceleration. Thus, which of these two algorithms performs better depends on the ratio of the similarity constant to the number of machines.

# 4 Our Contributions

In this paper, we investigate whether it is possible to construct communication-efficient (in terms of CC-3) algorithms for distributed optimization problems. By taking state-of-the-art techniques for handling similarity, compression and local steps into a single method, we bridge the existing gap and design a SOTA in the sense of CC-3.

1. Combination of compression and similarity. It can be seen from Table 1 that there are methods that combine similarity with either compression or acceleration – but not both techniques at the same time. However, there was no success in combining all three approaches into a single algorithm. We propose the first accelerated method for both unbiased and biased compression under similarity condition.   
2. Best communication time. To the best of our knowledge, in the case of $\delta \ll L$ , depending on the number of machines, either Accelerated ExtraGradient or Three Pillars Algorithm (Beznosikov, Takác, and Gasnikov 2024) gives the top CC-3 result. It can be seen from Table 1 that our OLGA significantly dominates both methods.   
3. Numerical experiments. The constructed theory is verified experimentally on different problems. Experiments confirm the superiority of our method and show its robustness to changes in the number of computational nodes and different values of the problem constants.

# 5 Problem Formulation and Assumptions

Real problems arising in practice are known to behave better than in theory. Therefore, constructing methods for idealized setups can be useful (Woodworth, Mishchenko, and Bach 2023). Our work relies on the strong convexity of the mean risk and the homogeneity of the data on all nodes.

<table><tr><td>Method</td><td>Approach</td><td># of client-server communications rounds (CC-1)</td><td># of client-server communications (CC-2)</td><td>Communication (CC-3)</td><td>Weaknesses</td></tr><tr><td>ADIANA</td><td>Unbiased compressionAcceleration</td><td> $\mathcal{O}\left(\left(\omega + \left(1 + M^{-1/2}\omega\right)\sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M\omega + \left(M + M^{1/2}\omega\right)\sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(1 + \left(\omega^{-1} + M^{-1/2}\right)\sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td>Only unbiased compressorAll nodes are involved in communication roundDoes not account for similarity</td></tr><tr><td>ECLK</td><td>Biased compressionAcceleration</td><td> $\mathcal{O}\left(\left(\beta + \beta^{3/2}\sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M\beta + M\beta^{3/2}\sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(1 + \beta^{1/2}\sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td>Bad constants in estimationAll nodes are involved in communication roundDoes not account for similarity</td></tr><tr><td>LoCoDL</td><td>Unbiased compressionLocal stepsAcceleration</td><td> $\mathcal{O}\left(\left(\omega + (1 + \omega^{1/2})\sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M\omega + (M + M\omega^{1/2})\sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(1 + (\omega^{-1} + \omega^{-1/2})\sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td>Does not account for similarityAll nodes are involved in communication round</td></tr><tr><td>TAMUNA</td><td>Unbiased compressionClient samplingLocal steps</td><td> $\mathcal{O}\left(\left(M + \sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M + \sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M + \sqrt{\frac{\mu}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td>Does not account for similarity</td></tr><tr><td>AccExtraGradient</td><td>SimilarityLocal stepsAcceleration</td><td> $\mathcal{O}\left(\sqrt{\frac{\delta}{\mu}}\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(M\sqrt{\frac{\delta}{\mu}}\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\sqrt{\frac{\delta}{\mu}}\log \frac{1}{\varepsilon}\right)$ </td><td>All nodes are involved in communication roundCommunication is inefficient</td></tr><tr><td>AccSVRS</td><td>SimilarityClient samplingLocal stepsAcceleration</td><td> $\mathcal{O}\left(\left(M + M^{3/4}\sqrt{\frac{\delta}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M + M^{3/4}\sqrt{\frac{\delta}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M + M^{3/4}\sqrt{\frac{\delta}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td>Communication is inefficient</td></tr><tr><td>Optimistic MASHA</td><td>SimilarityUnbiased compression</td><td> $\mathcal{O}\left(\left(M + \frac{\underline{\mu}}{\mu} + M^{1/2}\frac{\underline{\alpha}}{\mu}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M^{2} + M\frac{\underline{\mu}}{\mu} + M^{3/2}\frac{\underline{\alpha}}{\mu}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(1 + \left(M^{-1}\frac{\underline{\mu}}{\mu} + M^{-1/2}\frac{\underline{\alpha}}{\mu}\right)\log \frac{1}{\varepsilon}\right)$ </td><td>All nodes are involved in communication roundLipschitz constant is included in the estimationNo accelerationOnly permutation compressor</td></tr><tr><td>Three Pillars Algorithm</td><td>Unbiased compressionSimilarityLocal steps</td><td> $\mathcal{O}\left(\left(M + M^{1/2}\frac{\underline{\alpha}}{\mu}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M^{2} + M^{3/2}\frac{\underline{\alpha}}{\mu}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(1 + M^{-1/2}\frac{\underline{\alpha}}{\mu}\right)\log \frac{1}{\varepsilon}\right)$ </td><td>All nodes are involved in communication roundNo accelerationOnly permutation compressor</td></tr><tr><td>OLGA</td><td>SimilarityUnbiased compressionLocal stepsAcceleration</td><td> $\mathcal{O}\left(\left(\omega + \left[\omega M^{-1/4} + \omega^{1/2}\right]\sqrt{\frac{\delta}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M\omega + \left[\omega M^{3/4} + M\omega^{1/2}\right]\sqrt{\frac{\delta}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(1 + \left[M^{-1/4} + \omega^{-1/2}\right]\sqrt{\frac{\delta}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td>Only unbiased compressorAll nodes are involved in communication round</td></tr><tr><td>EF-OLGA</td><td>SimilarityBiased compressionLocal stepsAcceleration</td><td> $\mathcal{O}\left(\left(\beta + \beta^{5/4}\sqrt{\frac{\delta}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(M\beta + M\beta^{5/4}\sqrt{\frac{\delta}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td> $\mathcal{O}\left(\left(1 + \beta^{1/4}\sqrt{\frac{\delta}{\mu}}\right)\log \frac{1}{\varepsilon}\right)$ </td><td>All nodes are involved in communication round</td></tr></table>

Table 1: Summary of different communication complexities of SOTAs for distributed optimization.   
Notation: $\omega, \beta =$ compression constants, $M =$ number of computational nodes, $\delta =$ similarity (relatedness) constant, $\mu =$ constant of strong convexity of the objective $f$ , $L =$ Lipschitz constant of the gradient of $f$ .

Assumption 1. Every $f_{m}$ is $\delta$ -related to $f$ (Definition 3) and $f: \mathbb{R}^{d} \to \mathbb{R}$ is $\mu$ -strongly convex on $\mathbb{R}^{d}$ :

$$
f (x) \geq f (y) + \langle \nabla f (y), x - y \rangle + \frac {\mu}{2} \| x - y \| ^ {2}, \tag {2}
$$

$$
\forall x, y \in \mathbb {R} ^ {d}.
$$

Note that Assumption 1 allows local functions to be non-convex. The only requirement imposed on them is the Hessian similarity. We consider the distributed optimization problem of the form (1), where the data is drawn from a single distribution. It is also worth noting that Definition 3 implies $\delta$ -smoothness of $f_{m} - f$ for all $m \in [1, M]$ :

$$
\left\| \nabla (f _ {m} - f) (x) - \nabla (f _ {m} - f) (y) \right\| ^ {2} \leq \delta^ {2} \| x - y \| ^ {2}, \tag {3}
$$

$$
\forall x, y \in \mathbb {R} ^ {d}.
$$

In fact, the Hessian similarity assumption does not dramatically reduce the generality of our analysis. Indeed, if the data is heterogeneous, it is sufficient to put $\delta = L$ in our results.

# 6 Unbiased Compression via OLGA

As mentioned above, compression can be implemented through variance reduction (Qian, Richtárik, and Zhang 2021; Gorbunov et al. 2021; Beznosikov et al. 2022). The approach proposed in (Lin et al. 2024) provides a way to construct an optimal method for the variance reduction technique under the similarity condition. The idea is to try to naturally generalize this approach to schemes with compression. We first consider an unbiased compressor (Definition 1).

# Algorithm 1

1: Input: $x_{0} \in R^{d}, p \in (0,1), \theta > 0$

2: Set $N\sim \mathrm{Geom}(p)$

3: Send $x_0$ and $\nabla f_1(x_0)$ to each device

4: Collect $\nabla f(x_0) = \frac{1}{M}\sum_{m=1}^{M}\nabla f_m(x_0)$ on server

5: for $k = 0, 1, 2, \ldots, N - 1$ do:

6: for each device m in parallel do:

7: Calculate $\hat{g}_k^m$ using formula:

$$
\hat {g} _ {k} ^ {m} = \nabla f _ {m} (x _ {k}) - \nabla f _ {1} (x _ {k}) - \nabla f _ {m} (x _ {0}) + \nabla f _ {1} (x _ {0})
$$

8: Send $g_{k}^{m} = Q(\hat{g}_{k}^{m})$ to server

9: end for

10: Collect $g_{k} = \frac{1}{M}\sum_{m = 1}^{M}g_{k}^{m}$ on server

11: Calculate $t_k = g_k - \nabla f_1(x_0) + \nabla f(x_0)$

12: Update $x_{k + 1} = \arg \min_{x\in \mathbb{R}^d}q(x)$ , where

$$
q (x) = \left\langle t _ {k}, x - x _ {k} \right\rangle + \frac {1}{2 \theta} \| x - x _ {k} \| ^ {2} + f _ {1} (x)
$$

13: Send $x_{k + 1}$ and $\nabla f_1(x_{k + 1})$ to each device

14: end for

15: Output: $x_{N}$

In Algorithm 1, the full gradient is called only once per iteration before entering the loop (Line 4). At each iteration, it is required to communicate with each node and solve the subproblem (Line 12 of Algorithm 1). The number of iterations of Algorithm 1 is a random variable (see Line 2) that is not bounded from above, and hence the effectiveness of communication is important. Algorithm 1 has no acceleration. We build it to use as an integral part of more efficient Algorithm 2. From the point of view of theory, we are only

interested in the descent lemma for Algorithm 1.

Lemma 1. Consider an epoch of Algorithm 1. Let $h(x) = f_{1}(x) - f(x) + \frac{1}{2\theta}\| x\|^2$ , where $\theta \leq \min \left\{\frac{\sqrt{p}\sqrt{M}}{8\delta\sqrt{\omega}}, \frac{1}{2\delta}\right\}$ . Then the following inequality holds for every $x \in \mathbb{R}^d$ :

$$
\begin{array}{l} \mathbb {E} \left[ f (x _ {N}) - f (x) \right] \leq \mathbb {E} \Big [ \langle x - x _ {0}, \nabla h (x _ {N}) - \nabla h (x _ {0}) \rangle \\ \left. - \frac {p}{2} D _ {h} \left(x _ {0}, x _ {N}\right) - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} \right]. \tag {4} \\ \end{array}
$$

Algorithm 2: OLGA   
1: Input: $z_{0}=y_{0}\in\mathbb{R}^{d}, p\in(0,1), \theta>0, \tau\in(0,1), \alpha>0$ 2: for $k=0,1,2,\ldots,K-1$ do:
3: Update $x_{k+1}=\tau z_{k}+(1-\tau)y_{k}$ 4: Update $y_{k+1}=Alg.1(x_{k+1},\theta,p)$ 5: Send $y_{k+1}$ to each device
6: Collect $\nabla f(y_{k+1})=\frac{1}{M}\sum_{m=1}^{M}\nabla f_{m}(y_{k+1})$ on server
7: Calculate $t_{k}=\nabla(f_{1}-f)(x_{k+1})-\nabla(f_{1}-f)(y_{k+1})$ 8: Calculate $G_{k+1}=p\left(t_{k}+\frac{x_{k+1}-y_{k+1}}{\theta}\right)$ 9: Update $z_{k+1}=\arg\min_{z\in\mathbb{R}^{d}}q(z)$ , where

$$
q (z) = \frac {1}{2 \alpha} \| z - z _ {k} \| ^ {2} + \langle G _ {k + 1}, z \rangle + \frac {\mu}{2} \| z - y _ {k + 1} \| ^ {2}
$$

10: end for   
11: Output: $y_{K}$

Next, we employ interpolation framework motivated by KatyushaX (Allen-Zhu 2018b) to obtain the final version of proposed algorithm. The key difference between our approach and KatyushaX is the choice of a suitable Bregman divergence as a metric function instead of the Euclidean distance used in the mentioned method.

Note that Line 9 of Algorithm 2 can be solved analytically and does not require expensive computations. The full gradient is computed twice per iteration, whereas without compression it could be invoked potentially infinitely often, resulting in significant communication costs. Algorithm 2 calls the full gradient a second time on the outer iteration. OLGA calls the full gradient a constant rather than potentially infinite number of times. For the sake of brevity of description, we introduce two potential functions:

$$
Y _ {k} = \frac {\alpha}{\tau} [ f (y _ {k}) - f (x _ {*}) ], \quad Z _ {k} = \frac {1 + \mu \alpha}{2} \| z _ {k} - x _ {*} \| ^ {2},
$$

where $x_{*}$ is the solution of the problem (1).

Theorem 1. Let the problem (1) be solved by Algorithm 2 with $\theta \leq \min \left\{\frac{\sqrt{p}\sqrt{M}}{8\delta\sqrt{\omega}}, \frac{1}{2\delta}\right\}$ and tuning parameters such that $4\alpha p\tau \leq \theta$ . Then the following inequality holds:

$$
\mathbb {E} \left[ Y _ {k + 1} + Z _ {k + 1} \right] \leq \mathbb {E} \left[ (1 - \tau) Y _ {k} + (1 + \mu \alpha) ^ {- 1} Z _ {k} \right].
$$

As discussed above, the iteration of Algorithm 2 invokes the full gradient twice and the compressed gradient another N times. N is a random variable depending on the parameter p. On average, we have $\mathcal{O}\left(1/\gamma_{\omega}+p\right)CC-3$ for a single iteration of OLGA. It is obvious that one should choose $p=1/\gamma_{\omega}$ . It has been discussed above that for practical compressors $\gamma_{\omega}\geq\omega$ holds. Let us select $\alpha,\tau$ values more carefully and formulate the following corollary.

Corollary 1. Let the problem (1) be solved by Algorithm 2. Choose

$$
p = \frac {1}{\gamma_ {\omega}}, \quad \theta \leq \min \left\{\frac {\sqrt {p} \sqrt {M}}{8 \delta \sqrt {\omega}}, \frac {1}{2 \delta} \right\},
$$

$$
\tau = \min \left\{\frac {\sqrt {\mu} \theta^ {1 / 2} p ^ {- 1 / 2}}{4}, \frac {1}{4} \right\}, \quad \alpha = \frac {\theta p ^ {- 1}}{8 \tau},
$$

then Algorithm 2 has

$$
\tilde {\mathcal {O}} \left(\gamma_ {\omega} + \sqrt {\frac {\delta}{\mu}} \left[ \gamma_ {\omega} M ^ {- 1 / 4} + \gamma_ {\omega} ^ {1 / 2} \right]\right) C C - 1,
$$

$$
\tilde {\mathcal {O}} \left(M \gamma_ {\omega} + \sqrt {\frac {\delta}{\mu}} \left[ \gamma_ {\omega} M ^ {3 / 4} + M \gamma_ {\omega} ^ {1 / 2} \right]\right) C C - 2,
$$

and

$$
\tilde {\mathcal {O}} \left(1 + \sqrt {\frac {\delta}{\mu}} \left[ M ^ {- 1 / 4} + \gamma_ {\omega} ^ {- 1 / 2} \right]\right) C C - 3.
$$

It is worth noting that in CC-3 if $\gamma_{\omega}$ is too large, the term with M dominates and the communication time result stops improving. If $\gamma_{\omega}$ is too small, the compression effect is not as strong as it could be. Therefore, the best effect is given by $\gamma_{\omega} = \Theta\left(\sqrt{M}\right)$ . To simplify the appearance of the obtained results, the complexities are shown in Table 1 under consideration that $\gamma_{\omega} \sim \omega$ . This holds for a number of common used compressors (Beznosikov et al. 2023; Alistarh et al. 2018).

# 6.1 Discussion

Let us compare Algorithm 2 with distributed optimization SOTAs. It makes sense to make comparisons only with methods that utilize similarity, because superiority over other ones depends mainly on how much $\delta$ is less than L. The optimal choice $\gamma_{\omega} = \Theta\left(\sqrt{M}\right)$ is assumed below.

1. Accelerated ExtraGradient outperforms our method in the sense of CC-1 and CC-2. However, OLGA has $\tilde{\mathcal{O}}\left(1+M^{-1/4}\sqrt{\delta/\mu}\right)$ CC-3 vs. $\tilde{\mathcal{O}}\left(\sqrt{\delta/\mu}\right)$ for its competitor and hence turns out to be better by a factor $M^{1/4}$ . Thus, the difference in the running time of methods with a substantially large number of machines is enormous.   
2. AccSVRS loses to our method in terms of CC-1: $\tilde{\mathcal{O}}\left(M^{1/2} + M^{1/4}\sqrt{\delta/\mu}\right)$ vs. $\tilde{\mathcal{O}}\left(M + M^{3/4}\sqrt{\delta/\mu}\right)$ ; and CC-3: $\tilde{\mathcal{O}}\left(1 + M^{-1/4}\sqrt{\delta/\mu}\right)$ vs. $\tilde{\mathcal{O}}\left(M + M^{3/4}\sqrt{\delta/\mu}\right)$ ; but wins by CC-2 due to client sampling.

3. Three Pillars Algorithm loses to our method by CC-1: $\tilde{\mathcal{O}}\left(M^{1/2} + M^{1/4}\sqrt{\delta/\mu}\right)$

vs. $\mathcal{O}\left(M + M^{1 / 2}\cdot \delta /\mu\right)$ ; and CC-2:

$$
\tilde {\mathcal {O}} \left(M ^ {3 / 2} + M ^ {5 / 4} \sqrt {\delta / \mu}\right) \quad \text { vs. } \quad \mathcal {O} \left(M ^ {2} + M ^ {3 / 2} \cdot \delta / \mu\right).
$$

In terms of CC-3, it has a better $M^{-1/2}$ factor. However, Three Pillars Algorithm is not accelerated method. Thus, it loses to OLGA in a wide range of practical problems.

# 7 Biased Compression via EF-OLGA

In this section, we consider a biased compressor (Definition 2). As noted above, biased compressors are difficult to analyze and hence compressing gradient differences without introducing additional sequences will not yield results. Here we have to add "error" terms $e_{k}^{m}$ .

Algorithm 3   
1: Input: $x_{0} \in R^{d}, p \in (0,1), \theta > 0, e_{0}^{m} = 0$ 2: Set $N \in \text{Geom}(p)$ 3: Send $x_{0}$ and $\nabla f_{1}(x_{0})$ to each device
4: Collect $\nabla f(x_{0}) = \frac{1}{M} \sum_{m=1}^{M} \nabla f_{m}(x_{0})$ on server
5: for $k = 0, 1, 2, \ldots, N - 1$ do:
6:    for each device m in parallel do:
7:    Calculate $\hat{g}_{k}^{m}$ using formula: $\hat{g}_{k}^{m} = \nabla f_{m}(x_{k}) - \nabla f_{1}(x_{k}) - \nabla f_{m}(x_{0}) + \nabla f_{1}(x_{0})$ 8:    Send $g_{k}^{m} = C\{e_{k}^{m} + \theta\hat{g}_{k}^{m}\}$ to server
9:    Update $e_{k+1}^{m} = e_{k}^{m} - g_{k}^{m} + \theta\hat{g}_{k}^{m}$ 10:    if k = N - 1 then:
11:    Send $e_{k+1}^{m} = e_{N}^{m}$ to server
12:    end if
13:    end for
14:    Collect $g_{k} = \frac{1}{M} \sum_{m=1}^{M} g_{k}^{m}$ on server
15:    Calculate $t_{k} = \frac{1}{\theta} g_{k} - \nabla f_{1}(x_{0}) + \nabla f(x_{0})$ 16:    Update $x_{k+1} = \arg\min_{x \in R^{d}} q(x)$ , where $q(x) = \langle t_{k}, x - x_{k} \rangle + \frac{1}{2\theta} \|x - x_{k}\|^{2} + f_{1}(x)$

17: Send $x_{k + 1}$ and $\nabla f_1(x_{k + 1})$ to each device   
18: end for   
19: Output: $x_{N}$ , $\frac{1}{M} \sum_{m=1}^{M} e_{N}^{m}$

Algorithm 3 is obtained by combining Algorithm 1 and error feedback framework (Seide et al. 2014). It is proposed to introduce an additional sequence $e_{k}^{m}$ at each node m that will “remember” how much the gradient sent to the server differs from the true gradient computed on the machine (Line 9). Unlike the analysis in the previous section, it is not possible to introduce the appropriate metric immediately. After extensive analysis of virtual sequences, the Bregman divergence of the smooth function on "real" arguments and the Euclidean distance on "virtual" ones appear independently. This requires some manipulation to analyze correctly.

Algorithm 4: EF-OLGA   
1: Input: $z_{0} = y_{0} \in R^{d}, p \in (0,1), \theta > 0, \tau \in (0,1), \alpha > 0$ and $e_{m}^{0} \in R^{d}$ for every $m \in [1, M]$ 2: for $k = 0, 1, 2, \ldots, K - 1$ do:
3: Update $x_{k+1} = \tau z_k + (1 - \tau)y_k$ 4: Update $y_{k+1}, e_{k+1} = \text{Alg.3}(x_{k+1}, \theta, p)$ 5: Send $y_{k+1}$ to each device
6: Collect $\nabla f(y_{k+1}) = \frac{1}{M} \sum_{m=1}^{M} \nabla f(y_{k+1})$ on server
7: Calculate $t_k = \nabla(f_1 - f)(x_{k+1}) - \nabla(f_1 - f)(y_{k+1})$ 8: Calculate $G_{k+1} = p\left(t_k + \frac{x_{k+1} - \tilde{y}_{k+1}}{\theta}\right)$ , where

$$
\tilde {y} _ {k + 1} = y _ {k + 1} - e _ {k + 1}
$$

9: Update $z_{k+1} = \arg \min_{z \in \mathbb{R}^d} q(z)$ , where

$$
q (z) = \frac {1}{2 \alpha} \| z - z _ {k} \| ^ {2} + \langle G _ {k + 1}, z \rangle + \frac {\mu}{2} \| z - y _ {k + 1} \| ^ {2}
$$

10: end for

11: Output: $y_{K}$

Note that Line 8 of Algorithm 4 utilizes "error" terms from the last iteration of Algorithm 3. Without such a modification, variance reduction can not be performed. Thus, in the biased case, a stronger connection between the inner and outer algorithms is formed. As in the case of Algorithm 2, we also do not sample the stochastic gradient at the outer iteration because it does not affect the result and complicates the analysis. Therefore, EF-OLGA calls the full gradient twice per iteration.

Theorem 2. Let the problem (1) be solved by Algorithm 4 with $\theta \leq \frac{p^{3/2}}{24\delta} \leq \frac{1}{6\delta}$ and tuning parameters such that $28\alpha p\tau \leq \theta$ . Then the following inequality holds:

$$
\mathbb {E} \left[ Y _ {k + 1} + Z _ {k + 1} \right] \leq \mathbb {E} \left[ (1 - \tau) Y _ {k} + (1 + \mu \alpha) ^ {- 1} Z _ {k} \right].
$$

Again, the CC-3 of the iteration is $\mathcal{O}\left(\frac{1}{\gamma_{\beta}}+p\right)$ . Thus, $p=1/\gamma_{\beta}$ .

Corollary 2. Let the problem (1) be solved by Algorithm 4. Choose

$$
p = \frac {1}{\gamma_ {\beta}}, \quad \tau = \min \left\{\frac {\theta^ {1 / 2} p ^ {- 1 / 2}}{1 8}, \frac {1}{1 8} \right\},
$$

$$
\theta \leq \frac {p ^ {3 / 2}}{1 2 \delta}, \quad \alpha = \frac {\theta p ^ {- 1}}{3 6 \tau},
$$

then Algorithm 4 has

$$
\tilde {\mathcal {O}} \left(\gamma_ {\beta} + \gamma_ {\beta} ^ {5 / 4} \sqrt {\frac {\delta}{\mu}}\right) C C - 1, \quad \tilde {\mathcal {O}} \left(M \gamma_ {\beta} + M \gamma_ {\beta} ^ {5 / 4} \sqrt {\frac {\delta}{\mu}}\right) C C - 2,
$$

and

$$
\tilde {\mathcal {O}} \left(1 + \gamma_ {\beta} ^ {1 / 4} \sqrt {\frac {\delta}{\mu}}\right) C C - 3.
$$

This corollary repeats the proof of Corollary 1 with other constants.

# 8 Numerical Experiments

Our theoretical findings are confirmed numerically on various tasks. In particular, we consider the ridge regression problem (McDonald 2009):

$$
f (x) = \frac {1}{M} \sum_ {m = 1} ^ {M} \frac {1}{n} \sum_ {j = 1} ^ {n} \left(\langle x, a _ {j} ^ {m} \rangle - b _ {j} ^ {m}\right) ^ {2} + \lambda \| x \| ^ {2}, \tag {5}
$$

and the logistic regression problem:

$$
f (x) = \frac {1}{M} \sum_ {m = 1} ^ {M} \frac {1}{n} \sum_ {j = 1} ^ {n} \ln \left(1 + e ^ {- b _ {j} ^ {m} \left\langle x, a _ {j} ^ {m} \right\rangle}\right) + \lambda \| x \| ^ {2}. \tag {6}
$$

We set the penalty parameter $\lambda$ to $L/100$ , where L is a Lipschitz constant of the gradient of the main objective. Also we consider different datasets from LibSVM (Chang and Lin 2011): a9a and mushrooms (Appendix). Since we consider illustrative experiments with linear models, it is not difficult to calculate $\mu$ , L, $\delta$ . Therefore, the parameters of algorithms are chosen in the same way as in the theory without any tuning. We also vary the number of workers M. As competitors we take state-of-the-art methods from Table 1: AccSVRS, Accelerated ExtraGradient, ADIANA, LoCoDL. For algorithms with compression, we use a random sparsification operators RandK, where we vary the number of coordinates K (and hence $\omega$ since $\omega = d/K$ ).

See the experiments with OLGA on a9a dataset on Figures 1 - 6. For additional experiments, see Appendix G.

![](images/0b99b894a7a676eba86b72e3cc95f939a8c9da61e6a6ac9be77d1aeb851a418e.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^1 | 10^1    | 10^1             | 10^1   | 10^1   |
| 500               | 10^-3| 10^-1   | 10^-1            | 10^-1  | 10^-1  |
| 1000              | 10^-5| 10^-2   | 10^-2            | 10^-2  | 10^-2  |
| 1500              | 10^-7| 10^-3   | 10^-3            | 10^-3  | 10^-3  |
| 2000              | 10^-8| 10^-4   | 10^-4            | 10^-4  | 10^-4  |
| 2500              | 10^-9| 10^-5   | 10^-5            | 10^-5  | 10^-5  |
| 3000              | 10^-9| 10^-6   | 10^-6            | 10^-6  | 10^-6  |
</details>

(a) $\omega = 123.00$

![](images/1e9c169e5c36dc2c8209cd9f616890bcd26eb60daa6cee1fefea248485500079.jpg)  
(b) $\omega = 10.25$

![](images/d0901cd6c9b95626928abdc17f8187b1cbfa05341c45ca05ee39a7465b6c01e2.jpg)  
(c) $\omega = 3.33$   
Figure 1: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 100 and a9a dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .

![](images/0293a27938341901edbfe62d4901c5d7da89077113ffc40b3c06367268ea7e96.jpg)  
(a) $\omega = 123.00$

![](images/dc226025afd66fb64b7dcc984150d2a201ffd342138e84b59aa77e924e76255c.jpg)  
(b) $\omega = 7.24$

![](images/b4889bdc43652ae47ad71cbb402b4b83d4f29fc7e432014257a7b3d343fe843b.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^2 | 10^2    | 10^2             | 10^2   | 10^2   |
| 200               | 10^-3| 10^-2   | 10^-2            | 10^-2  | 10^-2  |
| 400               | 10^-4| 10^-3   | 10^-3            | 10^-3  | 10^-3  |
| 600               | 10^-5| 10^-4   | 10^-4            | 10^-4  | 10^-4  |
| 800               | 10^-6| 10^-5   | 10^-5            | 10^-5  | 10^-5  |
| 1000              | 10^-7| 10^-6   | 10^-6            | 10^-6  | 10^-6  |
| 1200              | 10^-8| 10^-7   | 10^-7            | 10^-7  | 10^-7  |
| 1400              | 10^-9| 10^-8   | 10^-8            | 10^-8  | 10^-8  |
| 1600              | 10^-10| 10^-9   | 10^-9            | 10^-9  | 10^-9  |
</details>

(c) ω = 4.10   
Figure 2: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 50 and a9a dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .

# 9 Conclusion

In this paper, we pioneered two algorithms, OLGA and EF-OLGA, that bridge the gap between acceleration, similarity and compression. For unbiased compressors, our theory

![](images/b5a5fe2ed9dbeb1b435f11fdd959428a9403fc08f29040d26bc0f0ead19658b9.jpg)  
(a) $\omega = 123.00$

![](images/0d6a478cb109cbdf68d5cbf91f9564d7b184fb7cc5a5cacf1afe10b8cc2c12f8.jpg)  
(b) $\omega = 10.25$

![](images/19a4ae07be8703f61b080b64624de6331532497dec9bf0f6e4d1c5bc0dd11255.jpg)  
(c) $\omega = 4.10$

Figure 3: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 20 and a9a dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .   
![](images/d92c49d430b85c329bc12438255bdfa89a52fdb38e094e87115fdb07da1fe4cf.jpg)  
(a) $\omega = 123.00$

![](images/17055deb09eada8f0c898a41017806bafcc3f7ba034328232c583ad5aeff55f7.jpg)  
(b) $\omega = 7.23$

![](images/f9f521c639e0580af14137d6378b4e8047d1e45ab032072b9d73e95db51e570a.jpg)  
(c) $\omega = 4.10$

Figure 4: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 70 and a9a dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .   
![](images/cb550131ac764121682d408d953259effe00498d80d5904ff5785a09a93e13a9.jpg)  
(a) $\omega = 123.00$

![](images/9a613f16c52f47f48e9c78321b066ea81cc63f3f220a18a9dd8a8f8a359cd4ec.jpg)  
(b) $\omega = 7.23$

![](images/1aeddfcd19e9cc7bf5576157a2184e2d79d31ed7ffd12f5805513eb51cbe042d.jpg)  
(c) $\omega = 4.10$

Figure 5: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 50 and a9a dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .   
![](images/e191f7a101f091f26b54e63437b9ef611aa02382117726c07141ba4b24db197e.jpg)  
(a) $\omega = 123.00$

![](images/de155c61a4a4f9d05dcf4532c6df882a01c752e88d0af16e332c29685b6f98af.jpg)  
(b) $\omega = 7.23$

![](images/3d36ebb7304ff84685693caad7303afb591312f3e547e1441442daf3a8d85bf0.jpg)  
(c) $\omega = 4.10$   
Figure 6: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 20 and a9a dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .

guarantees an advantage over SOTAs in terms of communication time. Numerical experiments support our theoretical findings. Nevertheless, a number of questions remain open. For biased compressors, a lower bound on the communication complexity under the similarity condition is unknown: can we design an accelerated algorithm that enjoys a better complexity? A part of the discussion of our results that does not fit into the main text, can be found in Appendix H.

# Acknowledgments

The work was done in the Laboratory of Federated Learning Problems of the ISP RAS (Supported by Grant App. No. 2 to Agreement No. 075-03-2024-214).

# References

Alacaoglu, A.; and Malitsky, Y. 2022. Stochastic variance reduction for variational inequality methods. In Conference on Learning Theory, 778–816. PMLR.   
Alistarh, D.; Grubic, D.; Li, J.; Tomioka, R.; and Vojnovic, M. 2017. QSGD: Communication-efficient SGD via gradient quantization and encoding. Advances in neural information processing systems, 30.   
Alistarh, D.; Hoefler, T.; Johansson, M.; Konstantinov, N.; Khirirat, S.; and Renggli, C. 2018. The convergence of sparsified gradient methods. Advances in Neural Information Processing Systems, 31.   
Allen-Zhu, Z. 2018a. Katyusha: The first direct acceleration of stochastic gradient methods. Journal of Machine Learning Research, 18(221): 1–51.   
Allen-Zhu, Z. 2018b. Katyusha x: Practical momentum method for stochastic sum-of-nonconvex optimization. arXiv preprint arXiv:1802.03866.   
Alqahtani, S.; and Demirbas, M. 2019. Performance analysis and comparison of distributed machine learning systems. arXiv preprint arXiv:1909.02061.   
Arjevani, Y.; and Shamir, O. 2015. Communication complexity of distributed convex learning and optimization. Advances in neural information processing systems, 28.   
Bai, H. 2022. Modern distributed data-parallel large-scale pre-training strategies for nlp models. In Proceedings of the 6th International Conference on High Performance Compilation, Computing and Communications, 44–53.   
Beznosikov, A.; Horváth, S.; Richtárik, P.; and Safaryan, M. 2023. On biased compression for distributed learning. Journal of Machine Learning Research, 24(276): 1–50.   
Beznosikov, A.; Richtárik, P.; Diskin, M.; Ryabinin, M.; and Gasnikov, A. 2022. Distributed methods with compressed communication for solving variational inequalities, with theoretical guarantees. Advances in Neural Information Processing Systems, 35: 14013–14029.   
Beznosikov, A.; Scutari, G.; Rogozin, A.; and Gasnikov, A. 2021. Distributed saddle-point problems under data similarity. Advances in Neural Information Processing Systems, 34:8172–8184.   
Beznosikov, A.; Takác, M.; and Gasnikov, A. 2024. Similarity, compression and local steps: three pillars of efficient communications for distributed variational inequalities. Advances in Neural Information Processing Systems, 36.   
Bidollahkhani, M.; and Kunkel, J. M. 2024. Revolutionizing System Reliability: The Role of AI in Predictive Maintenance Strategies. arXiv preprint arXiv:2404.13454.   
Bylinkin, D.; Degtyarev, K.; and Beznosikov, A. 2024. Accelerated Stochastic ExtraGradient: Mixing Hessian and gradient similarity to reduce communication in distributed and federated learning. arXiv preprint arXiv:2409.14280.

Chang, C.-C.; and Lin, C.-J. 2011. LIBSVM: a library for support vector machines. ACM transactions on intelligent systems and technology (TIST), 2(3): 1–27.

Condat, L.; Agarský, I.; Malinovsky, G.; and Richtárik, P. 2023. Tamuna: Doubly accelerated federated learning with local training, compression, and partial participation. arXiv preprint arXiv:2302.09832.

Condat, L.; Maranjyan, A.; and Richtárik, P. 2024. LoCoDL: Communication-Efficient Distributed Learning with Local Training and Compression. arXiv preprint arXiv:2403.04348.

Defazio, A.; Bach, F.; and Lacoste-Julien, S. 2014. SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives. Advances in neural information processing systems, 27.

Gorbunov, E.; Burlachenko, K. P.; Li, Z.; and Richtárik, P. 2021. MARINA: Faster non-convex distributed learning with compression. In International Conference on Machine Learning, 3788–3798. PMLR.

Gorbunov, E.; Hanzely, F.; and Richtárik, P. 2020. A unified theory of SGD: Variance reduction, sampling, quantization and coordinate descent. In International Conference on Artificial Intelligence and Statistics, 680–690. PMLR.

He, Y.; Huang, X.; and Yuan, K. 2024. Unbiased compression saves communication in distributed optimization: when and how much? Advances in Neural Information Processing Systems, 36.

Hendrikx, H.; Xiao, L.; Bubeck, S.; Bach, F.; and Massoulie, L. 2020. Statistically preconditioned accelerated gradient method for distributed optimization. In International conference on machine learning, 4203–4227. PMLR.

Johnson, R.; and Zhang, T. 2013. Accelerating stochastic gradient descent using predictive variance reduction. Advances in neural information processing systems, 26.

Jordan, M. I.; Lee, J. D.; and Yang, Y. 2019. Communication-efficient distributed statistical inference. Journal of the American Statistical Association.

Kairouz, P.; McMahan, H. B.; Avent, B.; Bellet, A.; Bennis, M.; Bhagoji, A. N.; Bonawitz, K.; Charles, Z.; Cormode, G.; Cummings, R.; et al. 2021. Advances and open problems in federated learning. Foundations and trends® in machine learning, 14(1–2): 1–210.

Khirirat, S.; Feyzmahdavian, H. R.; and Johansson, M. 2018. Distributed learning with compressed gradients. arXiv preprint arXiv:1806.06573.

Konečný, J.; McMahan, H. B.; Yu, F. X.; Richtárik, P.; Suresh, A. T.; and Bacon, D. 2016. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492.

Kovalev, D.; Beznosikov, A.; Borodich, E.; Gasnikov, A.; and Scutari, G. 2022. Optimal gradient sliding and its application to optimal distributed optimization under similarity. Advances in Neural Information Processing Systems, 35:33494–33507.

Kovalev, D.; Horváth, S.; and Richtárik, P. 2020. Don’t jump through hoops and remove those loops: SVRG and Katyusha are better without the outer loop. In Algorithmic Learning Theory, 451–467. PMLR.

Li, Z.; Kovalev, D.; Qian, X.; and Richtárik, P. 2020. Acceleration for compressed gradient descent in distributed and federated optimization. arXiv preprint arXiv:2002.11364.   
Lin, D.; Han, Y.; Ye, H.; and Zhang, Z. 2024. Stochastic distributed optimization under average second-order similarity: Algorithms and analysis. Advances in Neural Information Processing Systems, 36.   
Lu, H.; Freund, R. M.; and Nesterov, Y. 2018. Relatively smooth convex optimization by first-order methods, and applications. SIAM Journal on Optimization, 28(1): 333–354.   
McDonald, G. C. 2009. Ridge regression. Wiley Interdisciplinary Reviews: Computational Statistics, 1(1): 93–100.   
Mishchenko, K.; Gorbunov, E.; Takáč, M.; and Richtárik, P. 2019. Distributed learning with compressed gradient differences. arXiv preprint arXiv:1901.09269.   
Nesterov, Y. 2013. Introductory lectures on convex optimization: A basic course, volume 87. Springer Science & Business Media.   
Nozari, E.; Tallapragada, P.; and Cortés, J. 2016. Differentially private distributed convex optimization via functional perturbation. IEEE Transactions on Control of Network Systems, 5(1): 395–408.   
Qian, X.; Richtárik, P.; and Zhang, T. 2021. Error compensated distributed SGD can be accelerated. Advances in Neural Information Processing Systems, 34: 30401–30413.   
Richtárik, P.; Sokolov, I.; and Fatkhullin, I. 2021. EF21: A new, simpler, theoretically better, and practically faster error feedback. Advances in Neural Information Processing Systems, 34: 4384–4396.   
Robbins, H.; and Monro, S. 1951. A stochastic approximation method. The annals of mathematical statistics, 400–407.   
Seide, F.; Fu, H.; Droppo, J.; Li, G.; and Yu, D. 2014. 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs. In Interspeech, volume 2014, 1058–1062. Singapore.   
Shamir, O.; Srebro, N.; and Zhang, T. 2014. Communication-efficient distributed optimization using an approximate newton-type method. In International conference on machine learning, 1000–1008. PMLR.   
Stich, S. U.; and Karimireddy, S. P. 2019. The error-feedback framework: Better rates for SGD with delayed gradients and compressed communication. arXiv preprint arXiv:1909.05350.   
Sun, H.; Shao, Y.; Jiang, J.; Cui, B.; Lei, K.; Xu, Y.; and Wang, J. 2019. Sparse gradient compression for distributed SGD. In International Conference on Database Systems for Advanced Applications, 139–155. Springer.   
Tian, Y.; Scutari, G.; Cao, T.; and Gasnikov, A. 2022. Acceleration in distributed optimization under similarity. In International Conference on Artificial Intelligence and Statistics, 5721–5756. PMLR.   
Verbraeken, J.; Wolting, M.; Katzy, J.; Kloppenburg, J.; Verbelen, T.; and Rellermeyer, J. S. 2020. A survey on distributed machine learning. Acm computing surveys (csur), 53(2): 1–33.

Vogels, T.; Karimireddy, S. P.; and Jaggi, M. 2019. PowerSGD: Practical low-rank gradient compression for distributed optimization. Advances in Neural Information Processing Systems, 32.   
Weeraddana, P.; and Fischione, C. 2017. On the privacy of optimization. IFAC-PapersOnLine, 50(1): 9502–9508.   
Woodworth, B.; Mishchenko, K.; and Bach, F. 2023. Two losses are better than one: Faster optimization using a cheaper proxy. In International Conference on Machine Learning, 37273–37292. PMLR.   
Wu, Y.; Zeng, D.; Wang, Z.; Shi, Y.; and Hu, J. 2022. Distributed contrastive learning for medical image segmentation. Medical Image Analysis, 81: 102564.   
Yuan, X.-T.; and Li, P. 2020. On convergence of distributed approximate newton methods: Globalization, sharper bounds and beyond. Journal of Machine Learning Research, 21(206):1–51.   
Zhang, Y.; and Lin, X. 2015. Disco: Distributed optimization for self-concordant empirical loss. In International conference on machine learning, 362–370. PMLR.   
Zhu, W.; Luo, J.; and White, A. D. 2022. Federated learning of molecular properties with graph neural networks in a heterogeneous setting. Patterns, 3(6).

# Appendices

# A Auxiliary Definitions and Lemmas

Definition 4. Consider a continuously differentiable $h \colon \mathbb{R}^d \to \mathbb{R}$ and such $D_h(\cdot, \cdot)$ that

$$
D _ {h} (x, y) = h (x) - h (y) - \langle \nabla h (y), x - y \rangle .
$$

$D_{h}(\cdot ,\cdot)$ is called the Bregman divergence associated with $h(\cdot)$ .

Lemma 2. (Allen-Zhu 2018b) Given sequence $D_{0}, D_{1}, \ldots, D_{N} \in \mathbb{R}^{d}$ , where $N \in \text{Geom}(p)$ . Then

$$
\mathbb {E} _ {N} [ D _ {N - 1} ] = p D _ {0} + (1 - p) \mathbb {E} _ {N} [ D _ {N} ]. \tag {7}
$$

Lemma 3. (Allen-Zhu 2018b) If $g(\cdot)$ is proper $\sigma$ -strongly convex and $z_{k+1} = \arg \min_{z\in \mathbb{R}^d}\left[\frac{1}{2\alpha}\| z - z_k\|^2 +\langle G_{k + 1},z\rangle +\frac{\mu}{2}\| z - y_{k + 1}\|^2\right]$ , then for every $x\in \mathbb{R}^d$ we have

$$
\left\langle G _ {k + 1}, z _ {k} - x \right\rangle + g \left(z _ {k + 1}\right) - g (x) \leq \frac {\alpha}{2} \| G _ {k + 1} \| ^ {2} + \frac {\| z _ {k} - x \| ^ {2}}{2 \alpha} - \frac {(1 + \sigma \alpha) \| z _ {k + 1} - x \| ^ {2}}{2 \alpha}. \tag {8}
$$

Proposition 1. (Nesterov 2013) Let $f: \mathbb{R}^d \to \mathbb{R}$ be a convex function with a $L$ -Lipschitz gradient. The following inequality holds for every $x, y \in \mathbb{R}^d$ :

$$
\left. \left\| \nabla f (x) - \nabla f (y) \right\| ^ {2} \leq 2 L (f (y) - f (x) - \langle \nabla f (x), y - x \rangle). \right.
$$

Proposition 2. (Three-point equality) (Lin et al. 2024). Given a differentiable function $h \colon \mathbb{R}^d \to \mathbb{R}$ . We have

$$
\langle x - y, \nabla h (y) - \nabla h (z) \rangle = D _ {h} (x, z) - D _ {h} (x, y) - D _ {h} (y, z). \tag {9}
$$

# B Proof of Lemma 1

Lemma 4. (Lemma 1). Consider an epoch of Algorithm 1. Let $h(x) = f_{1}(x) - f(x) + \frac{1}{2\theta}\| x\|^2$ , where $\theta \leq \min \left\{\frac{\sqrt{p}\sqrt{M}}{8\delta\sqrt{\omega}}, \frac{1}{2\delta}\right\}$ . Then the following inequality holds for every $x \in \mathbb{R}^d$ :

$$
\mathbb {E} \left[ f (x _ {N}) - f (x) \right] \leq \mathbb {E} \left[ \langle x - x _ {0}, \nabla h (x _ {N}) - \nabla h (x _ {0}) \rangle - \frac {p}{2} D _ {h} (x _ {0}, x _ {N}) - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} \right].
$$

Proof. Our goal at this point is to get an evaluation within the single epoch. Let us start with the definition of strong convexity:

$$
\mathbb {E} [ f (x _ {k + 1}) - f (x) ] \leq \mathbb {E} \left[ \langle x - x _ {k + 1}, - \nabla f (x _ {k + 1}) \rangle - \frac {\mu}{2} \| x _ {k + 1} - x \| ^ {2} \right]. \tag {10}
$$

Writing the optimality condition of subproblem in Line 12 of Algorithm 1, we obtain

$$
g _ {k} - \nabla f _ {1} (x _ {0}) + \nabla f (x _ {0}) + \frac {x _ {k + 1} - x _ {k}}{\theta} + \nabla f _ {1} (x _ {k + 1}) = 0.
$$

Let us rewrite it in the following form:

$$
\nabla f _ {1} (x _ {k + 1}) = \frac {x _ {k} - x _ {k + 1}}{\theta} + \nabla (f _ {1} - f) (x _ {0}) - g _ {k}. \tag {11}
$$

It is easy to note that

$$
\nabla f (x _ {k + 1}) = \nabla f _ {1} (x _ {k + 1}) + \nabla f (x _ {k + 1}) - \nabla f _ {1} (x _ {k + 1}).
$$

By substituting (11) to this expression, we obtain

$$
\nabla f (x _ {k + 1}) = \frac {x _ {k} - x _ {k + 1}}{\theta} + \nabla (f _ {1} - f) (x _ {0}) - \nabla (f _ {1} - f) (x _ {k + 1}) - g _ {k}.
$$

Thus, we can rewrite the scalar product of (10) in the following way:

$$
\begin{array}{l} \langle x - x _ {k + 1}, - \nabla f (x _ {k + 1}) \rangle = \frac {1}{\theta} \langle x - x _ {k + 1}, x _ {k + 1} - x _ {k} \rangle + \langle x - x _ {k + 1}, g _ {k} \rangle \\ + \langle x - x _ {k + 1}, \nabla (f _ {1} - f) (x _ {k + 1}) - \nabla (f _ {1} - f) (x _ {0}) \rangle \\ = \frac {1}{\theta} \left\langle x - x _ {k + 1}, x _ {k + 1} - x _ {k} \right\rangle \\ + \langle x - x _ {k + 1}, g _ {k} - \nabla (f _ {1} - f) (x _ {0}) + \nabla (f _ {1} - f) (x _ {k}) \rangle \\ \end{array}
$$

$$
\begin{array}{l} + \langle x - x _ {k + 1}, \nabla (f _ {1} - f) (x _ {k + 1}) - \nabla (f _ {1} - f) (x _ {k}) \rangle \\ = \left\langle x - x _ {k + 1}, \frac {x _ {k + 1} - x _ {k}}{\theta} + \nabla (f _ {1} - f) (x _ {k + 1}) - \nabla (f _ {1} - f) (x _ {k}) \right\rangle \\ + \langle x - x _ {k + 1}, g _ {k} - (\nabla (f _ {1} - f) (x _ {0}) - \nabla (f _ {1} - f) (x _ {k})) \rangle . \\ \end{array}
$$

With the notation of $g_{k}$ , representing $g_{k} - (\nabla (f_{1} - f)(x_{0}) - \nabla (f_{1} - f)(x_{k}))$ as a sum

$$
g _ {k} - (\nabla (f _ {1} - f) (x _ {0}) - \nabla (f _ {1} - f) (x _ {k})) = \frac {1}{M} \sum_ {m = 1} ^ {M} \left(g _ {k} ^ {m} - (\nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}))\right),
$$

and denoting $h(x) = f_{1}(x) - f(x) + \frac{1}{2\theta}\| x\|^2$ (as in the statement of Lemma 1), we obtain

$$
\begin{array}{l} \langle x - x _ {k + 1}, - \nabla f (x _ {k + 1}) \rangle = \langle x - x _ {k + 1}, \nabla h (x _ {k + 1}) - \nabla h (x _ {k}) \rangle \\ + \left\langle x - x _ {k + 1}, \frac {1}{M} \sum_ {m = 1} ^ {M} g _ {k} ^ {m} - \{\nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}) \} \right\rangle . \tag {12} \\ \end{array}
$$

Applying (9) to the first summand of (12), we get

$$
\begin{array}{l} \langle x - x _ {k + 1}, - \nabla f (x _ {k + 1}) \rangle = D _ {h} (x, x _ {k}) - D _ {h} (x, x _ {k + 1}) - D _ {h} (x _ {k + 1}, x _ {k}) \\ + \left\langle x - x _ {k + 1}, \frac {1}{M} \sum_ {m = 1} ^ {M} g _ {k} ^ {m} - \{\nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}) \} \right\rangle . \\ \end{array}
$$

Now we are ready to calculate the expectation from both sides of the inequality. Note that $E_{Q}\left[\frac{1}{M}\sum_{m=1}^{M}g_{k}^{m}-\{\nabla(f_{1}-f_{m})(x_{0})-\nabla(f_{1}-f_{m})(x_{k})\}\right]=0$ due to Definition 1. Let us apply tower property to the second summand of (12) and replace x with $x_{k}$ (here we introduce $t_{k}=\frac{1}{M}\sum_{m=1}^{M}g_{k}^{m}-\{\nabla(f_{1}-f_{m})(x_{0})-\nabla(f_{1}-f_{m})(x_{k})\}$ ):

$$
\mathbb {E} \left[ \langle x - x _ {k + 1}, t _ {k} \rangle \right] = \mathbb {E} \mathbb {E} _ {Q} \left[ \langle x - x _ {k + 1}, t _ {k} \rangle \right] = \mathbb {E} \mathbb {E} _ {Q} \left[ \langle x - x _ {k}, t _ {k} \rangle \right] + \mathbb {E} \mathbb {E} _ {Q} \left[ \langle x _ {k} - x _ {k + 1}, t _ {k} \rangle \right] = \mathbb {E} \left[ \langle x _ {k} - x _ {k + 1}, t _ {k} \rangle \right].
$$

In the last step, we used the independence of $x_{k}$ , x and output of Q on the k-th iteration. Thus, we have

$$
\begin{array}{l} \mathbb {E} \left[ \langle x - x _ {k + 1}, - \nabla f (x _ {k + 1}) \rangle \right] = \mathbb {E} \left[ D _ {h} (x, x _ {k}) - D _ {h} (x, x _ {k + 1}) - D _ {h} (x _ {k + 1}, x _ {k}) \right] \\ + \mathbb {E} \left[ \left\langle x _ {k} - x _ {k + 1}, \frac {1}{M} \sum_ {m = 1} ^ {M} [ g _ {k} ^ {m} - \{\nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}) \} ] \right\rangle \right]. \tag {13} \\ \end{array}
$$

Applying the Cauchy-Schwartz inequality to $\frac{1}{M}\sum_{m=1}^{M}\langle x_k - x_{k+1}, g_k - \{\nabla(f_1 - f_m)(x_0) - \nabla(f_1 - f_m)(x_k)\}\rangle$ , we get

$$
\begin{array}{l} \mathbb {E} \left[ \left\langle x _ {k} - x _ {k + 1}, \frac {1}{M} \sum_ {m = 1} ^ {M} g _ {k} ^ {m} - \{\nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}) \} \right\rangle \right] \\ \leq \frac {1 - \theta \delta}{4 \theta} \mathbb {E} \left[ \| x _ {k + 1} - x _ {k} \| ^ {2} \right] + \frac {\theta}{1 - \theta \delta} \mathbb {E} \left[ \left\| \frac {1}{M} \sum_ {m = 1} ^ {M} g _ {k} ^ {m} - \{\nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}) \} \right\| ^ {2} \right]. \\ \end{array}
$$

One can estimate the second term using Definition 1 and independence of the operators $Q$ on different devices. Writing out the expectation of the compressor action, we obtain

$$
\begin{array}{l} \mathbb {E} _ {Q} \left[ \left\| \frac {1}{M} \sum_ {m = 1} ^ {M} \left[ g _ {k} ^ {m} - \left\{\nabla \left(f _ {1} - f _ {m}\right) \left(x _ {0}\right) - \nabla \left(f _ {1} - f _ {m}\right) \left(x _ {k}\right) \right\} \right] \right\| ^ {2} \right] \\ \leq \mathbb {E} _ {Q} \left[ \frac {2}{M ^ {2}} \sum_ {m _ {i} <   m _ {j} = 1} ^ {M} \big \langle g _ {k} ^ {m _ {i}} - \nabla (f _ {1} - f _ {m _ {i}}) (x _ {0}) + \nabla (f _ {1} - f _ {m _ {i}}) (x _ {k}), g _ {k} ^ {m _ {j}} - \nabla (f _ {1} - f _ {m _ {j}}) (x _ {0}) + \nabla (f _ {1} - f _ {m _ {j}}) (x _ {k}) \big \rangle \right] \\ + \mathbb {E} _ {Q} \left[ \frac {1}{M ^ {2}} \sum_ {m = 1} ^ {M} \mathbb {E} _ {Q} \| g _ {k} ^ {m} - \{\nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}) \} \| ^ {2} \right] \\ = \frac {1}{M ^ {2}} \sum_ {m = 1} ^ {M} \mathbb {E} _ {Q} \| g _ {k} ^ {m} - \{\nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}) \} \| ^ {2}. \\ \end{array}
$$

Here the summand with scalar products is equal to zero due to independence of the operator Q actions. Definition 1 implies $\mathbb{E}_{Q}\|g_{k}^{m}-\{\nabla(f_{1}-f_{m})(x_{0})-\nabla(f_{1}-f_{m})(x_{k})\}\|^{2}\leq\omega\|\nabla(f_{1}-f_{m})(x_{0})-\nabla(f_{1}-f_{m})(x_{k})\|^{2}$ . Taking the full expectation, we obtain

$$
\mathbb {E} \left[ \left\| \frac {1}{M} \sum_ {m = 1} ^ {M} \left[ g _ {k} ^ {m} - \left\{\nabla \left(f _ {1} - f _ {m}\right) \left(x _ {0}\right) - \nabla \left(f _ {1} - f _ {m}\right) \left(x _ {k}\right) \right\} \right] \right\| ^ {2} \right] \leq \frac {\omega}{M ^ {2}} \sum_ {m = 1} ^ {M} \mathbb {E} \left[ \| \nabla \left(f _ {1} - f _ {m}\right) \left(x _ {0}\right) - \nabla \left(f _ {1} - f _ {m}\right) \left(x _ {k}\right) \| ^ {2} \right]. \tag {14}
$$

The next step is to apply similarity definition to (14): under the norm in the right-hand side we use “smart zeros” $\pm\nabla f(x_{0})$ and $\pm\nabla f(x_{k})$ . We divide the obtained expression into two summands using the properties of the norm. Then we apply similarity (Definition 3) to each one.

$$
\begin{array}{l} \frac {\omega}{M ^ {2}} \sum_ {m = 1} ^ {M} \mathbb {E} \left[ \| \nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}) \| ^ {2} \right] \leq \frac {2 \omega}{M ^ {2}} \sum_ {m = 1} ^ {M} \mathbb {E} \left[ \| \nabla (f _ {1} - f) (x _ {0}) - \nabla (f _ {1} - f) (x _ {k}) \| ^ {2} \right] \\ + \frac {2 \omega}{M ^ {2}} \sum_ {m = 1} ^ {M} \mathbb {E} \left[ \| \nabla (f - f _ {m}) (x _ {0}) - \nabla (f - f _ {m}) (x _ {k}) \| ^ {2} \right] \tag {15} \\ \leq \frac {4 \omega \delta^ {2}}{M} \mathbb {E} \left[ \| x _ {k} - x _ {0} \| ^ {2} \right]. \\ \end{array}
$$

We substitute (15) into (13):

$$
\begin{array}{l} \mathbb {E} \left[ \langle x - x _ {k + 1}, - \nabla f (x _ {k + 1}) \rangle \right] = \mathbb {E} \left[ D _ {h} (x, x _ {k}) - D _ {h} (x, x _ {k + 1}) - D _ {h} (x _ {k + 1}, x _ {k}) + \frac {1 - \theta \delta}{4 \theta} \| x _ {k + 1} - x _ {k} \| ^ {2} \right] \tag {16} \\ + \mathbb {E} \left[ \frac {4 \theta \omega \delta^ {2}}{(1 - \theta \delta) M} \| x _ {k} - x _ {0} \| ^ {2} \right]. \\ \end{array}
$$

Since $f_{m} - f$ is $\delta$ -smooth (see (3)), one can note that $h(x)$ is $\left(\frac{1}{\theta} - \delta\right)$ -strongly convex for $\theta \leq 1 / \delta$ . Our choice $\theta \leq 1 / 2\delta$ is appropriate. Moreover, $h(x)$ is $\left(\delta + \frac{1}{\theta}\right)$ -smooth. (2) gives $D_{h}(x,y) \geq \frac{1 - \theta\delta}{2\theta}\| x - y\|^2$ . Proposition 1 gives $D_{h}(x,y) \leq \frac{1 + \theta\delta}{2\theta}\| x - y\|^2$ . Let us write it down in a more convenient form:

$$
0 \leq \frac {1 - \theta \delta}{2 \theta} \| x - y \| ^ {2} \leq D _ {h} (x, y) \leq \frac {1 + \theta \delta}{2 \theta} \| x - y \| ^ {2}. \tag {17}
$$

Next, we use (17) and $\theta \leq \frac{1}{2\delta}$ to obtain:

$$
\frac {1 - \theta \delta}{4 \theta} \| x _ {k + 1} - x _ {k} \| ^ {2} \leq \frac {2 \theta}{1 - \theta \delta} \frac {1 - \theta \delta}{4 \theta} D _ {h} (x _ {k + 1}, x _ {k}) = \frac {1}{2} D _ {h} (x _ {k + 1}, x _ {k}), \tag {18}
$$

and

$$
\frac {4 \theta \omega \delta^ {2}}{(1 - \theta \delta) M} \| x _ {k} - x _ {0} \| ^ {2} \leq \frac {2 \theta}{1 - \theta \delta} \frac {4 \theta \omega \delta^ {2}}{(1 - \theta \delta) M} D _ {h} (x _ {0}, x _ {k}) \leq \frac {3 2 \theta^ {2} \omega \delta^ {2}}{M} D _ {h} (x _ {0}, x _ {k}). \tag {19}
$$

Let $k = N - 1$ . We can substitute (18), (19) into (16), and then into (10):

$$
\mathbb {E} [ f (x _ {N}) - f (x) ] \leq \mathbb {E} \left[ D _ {h} (x, x _ {N - 1}) - D _ {h} (x, x _ {N}) - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} + \frac {3 2 \theta^ {2} \omega \delta^ {2}}{M} D _ {h} (x _ {0}, x _ {N - 1}) \right].
$$

Since $D_h(x_0, x_0) = 0$ , Lemma 2 implies $\mathbb{E}[D_h(x_0, x_{N-1})] \leq \mathbb{E}[D_h(x_0, x_N)]$ and $\mathbb{E}[D_h(x, x_{N-1} - D_h(x, x_N))] = p\mathbb{E}[D_h(x, x_0) - D_h(x, x_N)]$ . Thus, we obtain the following:

$$
\mathbb {E} [ f (x _ {N}) - f (x) ] \leq \mathbb {E} \left[ p D _ {h} (x, x _ {0}) - p D _ {h} (x, x _ {N}) - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} + \frac {3 2 \theta^ {2} \omega \delta^ {2}}{M} D _ {h} (x _ {0}, x _ {N}) \right].
$$

Let us apply Proposition 9 in the form $D_h(x, x_0) + D_h(x_0, x_N) - D_h(x, x_N) = \langle x - x_0, \nabla h(x_0) - \nabla h(x_N) \rangle$ :

$$
\mathbb {E} [ f (x _ {N}) - f (x) ] \leq \mathbb {E} \left[ p \langle x - x _ {0}, \nabla h (x _ {0}) - \nabla h (x _ {N}) \rangle - p D _ {h} (x _ {0}, x _ {N}) - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} + \frac {3 2 \theta^ {2} \omega \delta^ {2}}{M} D _ {h} (x _ {0}, x _ {N}) \right].
$$

Since $\theta \leq \min \left\{\frac{\sqrt{p}\sqrt{M}}{8\delta\sqrt{\omega}},\frac{1}{2\delta}\right\}$ , we have $p - \frac{32\theta^2\omega\delta^2}{M}\leq -\frac{p}{2}$ and thus

$$
\mathbb {E} \left[ f (x _ {N}) - f (x) \right] \leq \mathbb {E} \left[ \langle x - x _ {0}, \nabla h (x _ {N}) - \nabla h (x _ {0}) \rangle - \frac {p}{2} D _ {h} (x _ {0}, x _ {N}) - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} \right].
$$

This completes the proof of the lemma.

□

# C Proof of Theorem 1

Theorem 3. (Theorem 1) Let the problem (1) be solved by Algorithm 2 with $\theta \leq \min \left\{\frac{\sqrt{p}\sqrt{M}}{8\delta\sqrt{\omega}}, \frac{1}{2\delta}\right\}$ and such tuning of parameters that $4\alpha p\tau \leq \theta$ . Then the following inequality holds:

$$
\mathbb {E} \left[ Y _ {k + 1} + Z _ {k + 1} \right] \leq \mathbb {E} \left[ (1 - \tau) Y _ {k} + (1 + \mu \alpha) ^ {- 1} Z _ {k} \right].
$$

Proof. Let us move from one epoch to the method as a whole. Since the output point of Algorithm 1 was used to calculate $y_{k + 1}$ , let us re-designate $x_0 \to x_{k + 1}$ and $x_N \to y_{k + 1}$ in Lemma 1:

$$
\mathbb {E} \left[ f (y _ {k + 1}) - f (x) \right] \leq \mathbb {E} \left[ \langle x - x _ {k + 1}, \nabla h (y _ {k + 1}) - \nabla h (x _ {k + 1}) \rangle - \frac {p}{2} D _ {h} (x _ {k + 1}, y _ {k + 1}) - \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right].
$$

One can see that $G_{k+1}$ from Line 8 is almost the same as $\nabla h(y_{k+1}) - \nabla h(x_{k+1})$ in the expression above. Namely, $\nabla h(y_{k+1}) - \nabla h(x_{k+1}) = -G_{k+1}$ . Hence, using (4) for the iteration of Algorithm 2, we get

$$
\begin{array}{l} \mathbb {E} \left[ f \left(y _ {k + 1}\right) - f (x) \right] \leq \mathbb {E} \left[ \left\langle x - x _ {k + 1}, - G _ {k + 1} \right\rangle - \frac {p}{2} D _ {h} \left(x _ {k + 1}, y _ {k + 1}\right) - \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right] \\ = \mathbb {E} \left[ \left\langle z _ {k} - x _ {k + 1}, - G _ {k + 1} \right\rangle + \left\langle x - z _ {k}, - G _ {k + 1} \right\rangle - \frac {p}{2} D _ {h} \left(x _ {k + 1}, y _ {k + 1}\right) \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right] \\ = \mathbb {E} \left[ \langle z _ {k} - x _ {k + 1}, - G _ {k + 1} \rangle + \langle z _ {k} - x, G _ {k + 1} \rangle - \frac {p}{2} D _ {h} (x _ {k + 1}, y _ {k + 1}) \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

Let us apply Line 3 of Algorithm 2 to the first term $\left(z_{k} - x_{k + 1} = \frac{1 - \tau}{\tau}\left(x_{k + 1} - y_{k}\right)\right)$ :

$$
\begin{array}{l} \mathbb {E} \left[ f \left(y _ {k + 1}\right) - f (x) \right] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} \left\langle x _ {k + 1} - y _ {k}, - G _ {k + 1} \right\rangle + \left\langle z _ {k} - x, G _ {k + 1} \right\rangle - \frac {p}{2} D _ {h} \left(x _ {k + 1}, y _ {k + 1}\right) \right] \tag {20} \\ - \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

Using Lemma 1 (see Lemma (4)) with $x = y_{k}$ :

$$
\mathbb {E} \left[ f (y _ {k + 1}) - f (y _ {k}) \right] \leq \mathbb {E} \left[ \langle y _ {k} - x _ {k + 1}, - G _ {k + 1} \rangle - \frac {p}{2} D _ {h} (x _ {k + 1}, y _ {k + 1}) - \frac {\mu}{2} \| y _ {k + 1} - y _ {k} \| ^ {2} \right].
$$

We rewrite this inequality in the following form:

$$
\mathbb {E} \left[ \langle x _ {k + 1} - y _ {k}, - G _ {k + 1} \rangle \right] \leq \mathbb {E} \left[ (f (y _ {k}) - f (y _ {k + 1})) - \frac {p}{2} D _ {h} (x _ {k + 1}, y _ {k + 1}) - \frac {\mu}{2} \| y _ {k + 1} - y _ {k} \| ^ {2} \right].
$$

and substitute it into (20):

$$
\begin{array}{l} \mathbb {E} \left[ \frac {1}{\tau} (f (y _ {k + 1}) - f (x)) \right] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} (f (y _ {k}) - f (x)) + \langle z _ {k} - x, G _ {k + 1} \rangle - \frac {p}{2 \tau} D _ {h} (x _ {k + 1}, y _ {k + 1}) \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

Applying Lemma 3 to the $\langle z_k - x, G_{k+1} \rangle - \frac{\mu}{2} \| y_{k+1} - x \|^2$ taking into account the fact that $\| y_{k+1} - z_{k+1} \|^2 \geq 0$ , we obtain:

$$
\mathbb {E} \left[ \frac {1}{\tau} (f \left(y _ {k + 1}\right) - f (x)) \right] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} [ f \left(y _ {k}\right) - f (x) ] - \frac {p}{2 \tau} D _ {h} \left(x _ {k + 1}, y _ {k + 1}\right) + \frac {\alpha}{2} \| G _ {k + 1} \| ^ {2} \right] \tag {21}
$$

$$
+ \mathbb {E} \left[ \frac {1}{2 \alpha} \| z _ {k} - x \| ^ {2} - \frac {1 + \mu \alpha}{2 \alpha} \| z _ {k + 1} - x \| ^ {2} \right].
$$

The last step left to do to complete the proof is to estimate $\|G_{k+1}\|^{2}$ . Let us use (17) and Proposition 1 and the fact that $\theta \leq 1/2\delta$ here:

$$
\begin{array}{l} \frac {\alpha}{2} \| G _ {k + 1} \| ^ {2} = \frac {\alpha p ^ {2}}{2} \| \nabla h (y _ {k + 1}) - \nabla h (x _ {k + 1}) \| ^ {2} \leq \frac {\alpha p ^ {2}}{2} \frac {2 (1 + \theta \delta)}{\theta} D _ {h} (x _ {k + 1}, y _ {k + 1}) \\ \leq \frac {3 \alpha p ^ {2}}{2 \theta} D _ {h} (x _ {k + 1}, y _ {k + 1}) \leq \frac {2 \alpha p ^ {2}}{\theta} D _ {h} (x _ {k + 1}, y _ {k + 1}). \\ \end{array}
$$

Thus, we can rewrite (21) as follows:

$$
\begin{array}{l} \mathbb {E} \left[ \frac {1}{\tau} (f (y _ {k + 1}) - f (x)) \right] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} [ f (y _ {k}) - f (x) ] + p \left(\frac {2 \alpha p}{\theta} - \frac {1}{2 \tau}\right) D _ {h} (x _ {k + 1}, y _ {k + 1}) \right] \\ + \mathbb {E} \left[ \frac {1}{2 \alpha} \| z _ {k} - x \| ^ {2} - \frac {1 + \mu \alpha}{2 \alpha} \| z _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

Since $4\alpha p\tau \leq \theta$ , we have

$$
\mathbb {E} \left[ \frac {1}{\tau} (f (y _ {k + 1}) - f (x)) + \frac {1 + \mu \alpha}{2 \alpha} \| z _ {k + 1} - x \| ^ {2} \right] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} [ f (y _ {k}) - f (x) ] + \frac {1}{2 \alpha} \| z _ {k} - x \| ^ {2} \right].
$$

# D Proof of Corollary 1

Corollary 3. (Corollary 1). Let the problem (1) be solved by Algorithm 2. Choose

$$
p = \frac {1}{\gamma_ {\omega}}, \quad \theta \leq \min \left\{\frac {\sqrt {p} \sqrt {M}}{8 \delta \sqrt {\omega}}, \frac {1}{2 \delta} \right\},
$$

$$
\tau = \min \left\{\frac {\sqrt {\mu} \theta^ {1 / 2} p ^ {- 1 / 2}}{4}, \frac {1}{4} \right\}, \quad \alpha = \frac {\theta p ^ {- 1}}{8 \tau},
$$

then Algorithm 2 has

$$
\tilde {\mathcal {O}} \left(\gamma_ {\omega} + \sqrt {\frac {\delta}{\mu}} \left[ \gamma_ {\omega} M ^ {- 1 / 4} + \gamma_ {\omega} ^ {1 / 2} \right]\right) C C - 1, \quad \tilde {\mathcal {O}} \left(M \gamma_ {\omega} + \sqrt {\frac {\delta}{\mu}} \left[ \gamma_ {\omega} M ^ {3 / 4} + M \gamma_ {\omega} ^ {1 / 2} \right]\right) C C - 2,
$$

and

$$
\tilde {\mathcal {O}} \left(1 + \sqrt {\frac {\delta}{\mu}} \left[ M ^ {- 1 / 4} + \gamma_ {\omega} ^ {- 1 / 2} \right]\right) C C - 3.
$$

Proof. Let us enumerate the cases:

1. Let $\tau = \min \left\{\frac{1}{4},\frac{\sqrt{\mu}\theta^{1 / 2}p^{-1 / 2}}{4}\right\} = \frac{1}{4}$ . In this case, we have

$$
(1 - \tau) (1 + \mu \alpha) = (1 - \tau) \left(1 + \frac {\mu \theta p ^ {- 1}}{8 \tau}\right) \geq (1 - \tau) \left(1 + \frac {1}{8 \tau}\right) = \frac {3}{4} \cdot \frac {3}{2} \geq 1.
$$

This implies $\mathbb{E}[Z_{k + 1} + Y_{k + 1}] \leq (1 - \frac{1}{4})\mathbb{E}[Z_k + Y_k]$ . Thus, $CC$ -3 of Algorithm 1 is $\tilde{\mathcal{O}}(1)$ .

2. Let $\tau = \min \left\{\frac{1}{4},\frac{\sqrt{\mu}\theta^{1 / 2}p^{-1 / 2}}{4}\right\} = \frac{\sqrt{\mu}\theta^{1 / 2}p^{-1 / 2}}{4}$ . In this case, we have

$$
\mu \alpha = \frac {\mu \theta p ^ {- 1}}{8 \tau} = \frac {\sqrt {\mu} \theta^ {1 / 2} p ^ {- 1 / 2}}{2} \leq \frac {1}{2} <   1.
$$

Thus, $(1 + \mu \alpha)^{-1} \leq 1 - \frac{\mu\alpha}{2}$ . This implies $\mathbb{E}[Z_{k+1} + Y_{k+1}] \leq (1 - \tau)\mathbb{E}[\Phi_k]$ . Or, in CC-3 terms:

$$
\tilde {\mathcal {O}} \left(1 + \frac {1}{\tau}\right) = \tilde {\mathcal {O}} \left(1 + \sqrt {\frac {\delta}{\mu}} \left[ M ^ {- 1 / 4} \omega^ {1 / 4} p ^ {1 / 4} + p ^ {1 / 2} \right]\right).
$$

Substitute $p = 1 / \gamma_{\omega}$ :

$$
\tilde {\mathcal {O}} \left(1 + \frac {1}{\tau}\right) = \tilde {\mathcal {O}} \left(1 + \sqrt {\frac {\delta}{\mu}} \left[ M ^ {- 1 / 4} \omega^ {1 / 4} \gamma_ {\omega} ^ {- 1 / 4} + \gamma_ {\omega} ^ {- 1 / 2} \right]\right).
$$

Summing both cases and using $\omega \leq \gamma_{\omega}$ , we obtain $\tilde{\mathcal{O}}\left(1 + \sqrt{\frac{\delta}{\mu}}\left[M^{-1/4} + \gamma_{\omega}^{-1/2}\right]\right)$ of $CC-3$ .

![](images/6d54e07b8871b61212ed32ea2986fe046e1d349cfe67a8b1dd0aafaaae04c0b8.jpg)

# E Proof of Lemma 5

Before proceeding to the proof of the Theorem 2, we introduce an auxiliary lemma.

Lemma 5. Consider an epoch of Algorithm 3. Consider $h(x) = f_{1}(x) - f(x) + \frac{1}{2\theta} \|x\|^{2}$ , where $\theta \leq \frac{p^{3/2}}{24\delta}$ . The following inequality holds:

$$
\mathbb {E} [ f (x _ {N}) - f (x) ] \leq \mathbb {E} \left[ p \left\langle x - x _ {0}, \nabla (f _ {1} - f) (x _ {N}) - \nabla (f _ {1} - f) (x _ {0}) + \frac {\tilde {x} _ {N} - x _ {0}}{\theta} \right\rangle - \frac {p}{7} D _ {h} (x _ {0}, x _ {N}) - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} \right].
$$

Proof. Same as in the unbiased case, our goal at the beginning is to get some sort of evaluation within a single epoch. Let us start with strong convexity definition:

$$
\mathbb {E} [ f (x _ {k + 1}) - f (x) ] \leq \mathbb {E} \left[ \langle x - x _ {k + 1}, - \nabla f (x _ {k + 1}) \rangle - \frac {\mu}{2} \| x _ {k + 1} - x \| ^ {2} \right]. \tag {22}
$$

Denote $e_k = \frac{1}{M} \sum_{m=1}^{M} e_k^m$ and introduce virtual sequence $\tilde{x}_k = x_k - e_k$ . We write the optimality condition for Line 16 of Algorithm 3:

$$
\frac {1}{\theta} g _ {k} - \nabla f _ {1} (x _ {0}) + \nabla f (x _ {0}) + \frac {x _ {k + 1} - x _ {k}}{\theta} = 0.
$$

Next we are going to use $x_{k + 1}$ , so let us express it:

$$
x _ {k + 1} = x _ {k} - g _ {k} - \theta [ \nabla f _ {1} (x _ {k + 1}) + \nabla f (x _ {0}) - \nabla f _ {1} (x _ {0}) ]. \tag {23}
$$

For virtual sequence $\tilde{x}_k$ we have $\tilde{x}_{k + 1} = x_{k + 1} - e_{k + 1}$ . Let us obtain an expression for it using (23) and Line 9 of Algorithm 3.

$$
\begin{array}{l} \tilde {x} _ {k + 1} = x _ {k + 1} - e _ {k + 1} \\ = x _ {k} - \theta [ \nabla f _ {1} (x _ {k + 1}) + \nabla f (x _ {0}) - \nabla f _ {1} (x _ {0}) ] - g _ {k} - e _ {k} + g _ {k} \\ + \theta [ \nabla f (x _ {0}) - \nabla f _ {1} (x _ {0}) + \nabla f _ {1} (x _ {k}) - \nabla f (x _ {k}) ] \\ = \tilde {x} _ {k} - \theta [ \nabla f (x _ {k}) - \nabla f _ {1} (x _ {k}) + \nabla f _ {1} (x _ {k + 1}) ]. \\ \end{array}
$$

After re-arranging terms, we write

$$
\nabla f _ {1} (x _ {k + 1}) = \frac {\tilde {x} _ {k} - \tilde {x} _ {k + 1}}{\theta} + \nabla f _ {1} (x _ {k}) - \nabla f (x _ {k}). \tag {24}
$$

It is quite easy to note that

$$
\nabla f (x _ {k + 1}) = \nabla f _ {1} (x _ {k + 1}) + \nabla f (x _ {k + 1}) - \nabla f _ {1} (x _ {k + 1}).
$$

Substitute (24) to this expression and obtain

$$
\nabla f (x _ {k + 1}) = \frac {\tilde {x} _ {k} - \tilde {x} _ {k + 1}}{\theta} + \nabla (f _ {1} - f) (x _ {k}) - \nabla (f _ {1} - f) (x _ {k + 1}). \tag {25}
$$

Now we are ready to rewrite scalar product of (22) using (25) in the following way:

$$
\begin{array}{l} \langle x - x _ {k + 1}, - \nabla f (x _ {k + 1}) \rangle = \frac {1}{\theta} \langle x - x _ {k + 1} \pm \tilde {x} _ {k + 1}, \tilde {x} _ {k + 1} - \tilde {x} _ {k} \rangle \\ + \langle x - x _ {k + 1}, \nabla (f _ {1} - f) (x _ {k + 1}) - \nabla (f _ {1} - f) (x _ {k}) \rangle \\ = \langle x - x _ {k + 1}, \nabla (f _ {1} - f) (x _ {k + 1}) - \nabla (f _ {1} - f) (x _ {k}) \rangle \\ + \frac {1}{\theta} \langle x - \tilde {x} _ {k + 1}, \tilde {x} _ {k + 1} - \tilde {x} _ {k} \rangle + \frac {1}{\theta} \langle \tilde {x} _ {k + 1} - x _ {k + 1}, \tilde {x} _ {k + 1} - \tilde {x} _ {k} \rangle \\ \end{array}
$$

Next apply Proposition 9 to the first summand, square of the norm formula to the second summand and Cauchy-Schwartz inequality to the third one.

$$
\begin{array}{l} \langle x - x _ {k + 1}, - \nabla f (x _ {k + 1}) \rangle = D _ {f _ {1} - f} (x, x _ {k}) - D _ {f _ {1} - f} (x, x _ {k + 1}) - D _ {f _ {1} - f} (x _ {k + 1}, x _ {k}) \\ + \frac {1}{2 \theta} \| \tilde {x} _ {k} - x \| ^ {2} - \frac {1}{2 \theta} \| \tilde {x} _ {k + 1} - x \| ^ {2} - \frac {1}{2 \theta} \| \tilde {x} _ {k + 1} - \tilde {x} _ {k} \| ^ {2} + \frac {1}{\theta} \| e _ {k + 1} \| ^ {2} \\ + \frac {1}{4 \theta} \| \tilde {x} _ {k + 1} - \tilde {x} _ {k} \| ^ {2}. \\ \end{array}
$$

Note that $-\|a - b\|^2 \geq -\|(a - c) + c - (b - d) - d\|^2 \geq -3\|(a - c) - (b - d)\|^2 - 3\|c\|^2 - 3\|e\|^2$ and therefore

$$
- \left\| \tilde {x} _ {k + 1} - \tilde {x} _ {k} \right\| ^ {2} \leq - \frac {1}{3} \left\| x _ {k + 1} - x _ {k} \right\| ^ {2} + \left\| e _ {k + 1} \right\| ^ {2} + \left\| e _ {k} \right\| ^ {2}.
$$

Thus, we can rewrite (22) in the following form:

$$
\begin{array}{l} \mathbb {E} \left[ f \left(x _ {k + 1}\right) - f (x) \right] \leq \mathbb {E} \left[ D _ {f _ {1} - f} \left(x, x _ {k}\right) - D _ {f _ {1} - f} \left(x, x _ {k + 1}\right) - D _ {f _ {1} - f} \left(x _ {k + 1}, x _ {k}\right) + \frac {1}{2 \theta} \| \tilde {x} _ {k} - x \| ^ {2} \right] \\ + \mathbb {E} \left[ - \frac {1}{2 \theta} \| \tilde {x} _ {k + 1} - x \| ^ {2} - \frac {1}{1 2 \theta} \| x _ {k + 1} - x _ {k} \| ^ {2} + \frac {5}{4 \theta} \| e _ {k + 1} \| ^ {2} + \frac {1}{4 \theta} \| e _ {k} \| ^ {2} \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| x _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

Let $k = N - 1$ . We can change previous inequality:

$$
\begin{array}{l} \mathbb {E} [ f (x _ {N}) - f (x) ] \leq \mathbb {E} \left[ D _ {f _ {1} - f} (x, x _ {N - 1}) - D _ {f _ {1} - f} (x, x _ {N}) - D _ {f _ {1} - f} (x _ {N}, x _ {N - 1}) + \frac {1}{2 \theta} \| \tilde {x} _ {N - 1} - x \| ^ {2} \right] \\ + \mathbb {E} \left[ - \frac {1}{2 \theta} \| \tilde {x} _ {N} - x \| ^ {2} - \frac {1}{1 2 \theta} \| x _ {N} - x _ {N - 1} \| ^ {2} + \frac {5}{4 \theta} \| e _ {N} \| ^ {2} + \frac {1}{4 \theta} \| e _ {N - 1} \| ^ {2} \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| x _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

Since $D_h(x_0, x_0) = 0$ , Lemma 2 implies $\mathbb{E}[D_h(x, x_{N-1}) - D_h(x, x_N)] = p\mathbb{E}[D_h(x, x_0) - D_h(x, x_N)], \frac{1}{2\theta}\| \tilde{x}_{N-1} - x\|^2 - \frac{1}{2\theta}\| \tilde{x}_N - x\|^2 = \frac{p}{2\theta}\| \tilde{x}_0 - x\|^2 - \frac{p}{2\theta}\| \tilde{x}_{N-1} - x\|^2$ and $\mathbb{E}_N[\| e_{N-1}\|^2] = (1-p)\mathbb{E}_N[\| e_N\|^2] \leq \mathbb{E}_N[\| e_N\|^2]$ . Take into account that $e_0 = 0$ , thus $\tilde{x}_0 = x_0$ , and obtain the following:

$$
\begin{array}{l} \mathbb {E} \left[ f \left(x _ {N}\right) - f (x) \right] \leq \mathbb {E} \left[ p D _ {f _ {1} - f} \left(x, x _ {0}\right) - p D _ {f _ {1} - f} \left(x, x _ {N}\right) + \frac {p}{2 \theta} \| x _ {0} - x \| ^ {2} - \frac {p}{2 \theta} \| \tilde {x} _ {N} - x \| ^ {2} \right] \tag {26} \\ + \mathbb {E} \left[ - D _ {f _ {1} - f} (x _ {N}, x _ {N - 1}) - \frac {1}{1 2 \theta} \| x _ {N} - x _ {N - 1} \| ^ {2} + \frac {3}{2 \theta} \| e _ {N} \| ^ {2} - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} \right]. \\ \end{array}
$$

It is not as easy to work in this setting as with the unbiased compressor, since it is not possible to generate the Bregman divergence by a strongly convex function. For this purpose, we will use a workaround. We use Proposition 9 twice: for the divergence generated by $f_{1} - f$ and $\frac{1}{2\theta}\| x\|^2$ . Let us write down the equations of interest:

$$
p D _ {f _ {1} - f} (x, x _ {0}) + p D _ {f _ {1} - f} (x _ {0}, x _ {N}) - p D _ {f _ {1} - f} (x, x _ {N}) = p \langle x - x _ {0}, \nabla (f _ {1} - f) (x _ {N}) - \nabla (f _ {1} - f) (x _ {0}) \rangle ,
$$

$$
\frac {p}{2 \theta} \| x _ {0} - x \| ^ {2} + \frac {p}{2 \theta} \| x _ {0} - \tilde {x} _ {N} \| ^ {2} - \frac {p}{2 \theta} \| \tilde {x} _ {N} - x \| ^ {2} = \frac {p}{\theta} \langle x - x _ {0}, \tilde {x} _ {N} - x _ {0} \rangle
$$

Transform (26) according to the written out expressions and get

$$
\begin{array}{l} \mathbb {E} [ f (x _ {N}) - f (x) ] \leq \mathbb {E} \left[ p \left\langle x - x _ {0}, \nabla (f _ {1} - f) (x _ {N}) - \nabla (f _ {1} - f) (x _ {0}) + \frac {\tilde {x} _ {N} - x _ {0}}{\theta} \right\rangle - p D _ {f _ {1} - f} (x _ {0}, x _ {N}) - \frac {p}{2 \theta} \| x _ {0} - \tilde {x} _ {N} \| ^ {2} \right] \\ + \mathbb {E} \left[ - D _ {f _ {1} - f} \left(x _ {N}, x _ {N - 1}\right) - \frac {1}{1 2 \theta} \| x _ {N} - x _ {N - 1} \| ^ {2} + \frac {3}{2 \theta} \| e _ {N} \| ^ {2} - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} \right]. \tag {27} \\ \end{array}
$$

It is known that the function $f_{1} - f$ δ-smooth. This means that we have an upper bound $D_{f_1 - f}(x,y) \leq \frac{\delta}{2}\| x - y\|^2$ for every $x,y \in \mathbb{R}^d$ . Thus, $\theta \leq \frac{1}{6\delta}$ is sufficient to fulfill the inequality $-D_{f_1 - f}(x_N,x_{N - 1}) - \frac{1}{12\theta}\| x_N - x_{N - 1}\|^2 \leq 0$ . For $-pD_{f_1 - f}(x_0,x_N) - \frac{p}{2\theta}\| x_0 - \tilde{x}_N\|^2$ we perform a more careful analysis.

$$
\begin{array}{l} - p D _ {f _ {1} - f} \left(x _ {0}, x _ {N}\right) - \frac {p}{2 \theta} \| x _ {0} - \tilde {x} _ {N} \| ^ {2} \leq - p D _ {f _ {1} - f} \left(x _ {0}, x _ {N}\right) - \frac {p}{4 \theta} \| x _ {0} - x _ {N} \| ^ {2} + \frac {1}{2 \theta} \| e _ {N} \| ^ {2} \\ \leq \frac {p \delta}{2} \| x _ {0} - x _ {N} \| ^ {2} - \frac {p}{4 \theta} \| x _ {0} - x _ {N} \| ^ {2} + \frac {1}{2 \theta} \| e _ {N} \| ^ {2}. \\ \end{array}
$$

$\theta \leq \frac{1}{6\delta}$ . It follows that we can estimate $\delta \leq \frac{1}{6\theta}$ . Thus,

$$
- p D _ {f _ {1} - f} \left(x _ {0}, x _ {N}\right) - \frac {p}{2 \theta} \| x _ {0} - \tilde {x} _ {N} \| ^ {2} \leq \frac {1}{2 \theta} \| e _ {N} \| ^ {2} - \frac {p}{6 \theta} \| x _ {0} - x _ {N} \| ^ {2}.
$$

Using this facts, rewrite (27):

$$
\begin{array}{l} \mathbb {E} [ f (x _ {N}) - f (x) ] \leq \mathbb {E} \left[ p \left\langle x - x _ {0}, \nabla \left(f _ {1} - f\right) \left(x _ {N}\right) - \nabla \left(f _ {1} - f\right) \left(x _ {0}\right) + \frac {\tilde {x} _ {N} - x _ {0}}{\theta} \right\rangle - \frac {p}{6 \theta} \| x _ {0} - x _ {N} \| ^ {2} \right] \tag {28} \\ + \mathbb {E} \left[ \frac {2}{\theta} \| e _ {N} \| ^ {2} - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} \right]. \\ \end{array}
$$

Now let us deal with the "error" term $\| e_{k + 1}\|^2$ . Firstly, we use Definition 2 and write

$$
\mathbb {E} [ \| e _ {k + 1} \| ^ {2} ] \leq \mathbb {E} \left[ \frac {1}{M} \sum_ {m = 1} ^ {M} \| e _ {k + 1} ^ {m} \| ^ {2} \right] \leq \frac {1}{M} \sum_ {m = 1} ^ {M} \left(1 - \frac {1}{\beta}\right) \mathbb {E} \left[ \| e _ {k} ^ {m} + \theta [ \nabla f _ {m} (x _ {k}) - \nabla f _ {1} (x _ {k}) - \nabla f _ {m} (x _ {0}) + \nabla f _ {1} (x _ {0}) ] \| ^ {2} \right].
$$

Next, we omit $(1 - \frac{1}{\beta})$ -factor and use the Cauchy-Schwarz inequality:

$$
\begin{array}{l} \mathbb {E} [ \| e _ {k + 1} \| ^ {2} ] \leq (1 + c) \mathbb {E} \left[ \frac {1}{M} \sum_ {m = 1} ^ {M} \| e _ {k} ^ {m} \| ^ {2} \right] \\ + \left(1 + \frac {1}{c}\right) \theta^ {2} \frac {1}{M} \sum_ {m = 1} ^ {M} \mathbb {E} \left[ \| \nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {k}) \| ^ {2} \right]. \\ \end{array}
$$

Next, we enroll the recursion and take into account the fact that $e_0^m = 0$ . Thus, we obtain

$$
\mathbb {E} [ \| e _ {k + 1} \| ^ {2} ] \leq 4 \theta^ {2} \left(1 + \frac {1}{c}\right) \sum_ {j = 1} ^ {k} (1 + c) ^ {k - j} \mathbb {E} \left[ \| \nabla (f _ {1} - f _ {m}) (x _ {0}) - \nabla (f _ {1} - f _ {m}) (x _ {j}) \| ^ {2} \right].
$$

$\delta$ -smoothness (see (3)) of $f_{m} - f$ gives

$$
\mathbb {E} [ \| e _ {k + 1} \| ^ {2} ] \leq 4 \theta^ {2} \delta^ {2} \left(1 + \frac {1}{c}\right) \sum_ {j = 1} ^ {k} (1 + c) ^ {k - j} \mathbb {E} \left[ \| x _ {j} - x _ {0} \| ^ {2} \right].
$$

The number of iterations of the Algorithm 3 N is a random variable. Let us calculate the expectation on the action of the compressive operator and on the random variable N at once. We are specifically interested in the “error” at the last iteration of the Algorithm 3. Since $N \in \text{Geom}(p)$ , we obtain

$$
\begin{array}{l} \mathbb {E} _ {C, N} [ \| e _ {N} \| ^ {2} ] = \mathbb {E} _ {C} \left[ \sum_ {k \geq 0} p (1 - p) ^ {k} \| e _ {k} \| ^ {2} \right] \\ \leq 4 \theta^ {2} \delta^ {2} \left(1 + \frac {1}{c}\right) \sum_ {k \geq 0} p (1 - p) ^ {k} \sum_ {j = 1} ^ {k - 1} (1 + c) ^ {k - j} \| x _ {j} - x _ {0} \| ^ {2}. \\ \end{array}
$$

Choose $c = \frac{p}{2}$ :

$$
\begin{array}{l} \sum_ {k \geq 0} p (1 - p) ^ {k} \sum_ {j = 1} ^ {k - 1} (1 + c) ^ {k - j} \| x _ {j} - x _ {0} \| ^ {2} = p [ \{(1 - p) ^ {2} (1 + c) + (1 - p) ^ {3} (1 + c) ^ {2} + \dots \} \| x _ {1} - x _ {0} \| ^ {2} \\ \begin{array}{l} + \{(1 - p) ^ {3} (1 + c) + (1 - p) ^ {4} (1 + c) ^ {2} +... \} \| x _ {2} - x _ {0} \| ^ {2} \\ +... ] \end{array} \\ \leq p \left[ \frac {2}{p} (1 - p) \| x _ {1} - x _ {0} \| ^ {2} + \frac {2}{p} (1 - p) ^ {2} \| x _ {2} - x _ {0} \| ^ {2} + \ldots \right] \\ = \frac {2}{p} \mathbb {E} _ {N} [ \| x _ {N} - x _ {0} \| ^ {2} ]. \\ \end{array}
$$

Note that $\left(1 + \frac{1}{c}\right) \leq \frac{3}{p}$ . Thus, we obtain

$$
\mathbb {E} [ \| e _ {N} \| ^ {2} ] \leq \frac {2 4 \theta^ {2} \delta^ {2}}{p ^ {2}} \mathbb {E} _ {N} [ \| x _ {N} - x _ {0} \| ^ {2} ]. \tag {29}
$$

Substitute into (28):

$$
\begin{array}{l} \mathbb {E} [ f (x _ {N}) - f (x) ] \leq \mathbb {E} \left[ p \left\langle x - x _ {0}, \nabla \left(f _ {1} - f\right) \left(x _ {N}\right) - \nabla \left(f _ {1} - f\right) \left(x _ {0}\right) + \frac {\tilde {x} _ {N} - x _ {0}}{\theta} \right\rangle - \frac {p}{6 \theta} \| x _ {0} - x _ {N} \| ^ {2} \right] \tag {30} \\ + \mathbb {E} \left[ \frac {4 8 \theta \delta^ {2}}{p ^ {2}} \| x _ {N} - x _ {0} \| ^ {2} - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} \right]. \\ \end{array}
$$

Since $f_{m} - f$ is $\delta$ -smooth (see (3)), one can note that $h(x)$ is $\left(\frac{1}{\theta} - \delta\right)$ -strongly convex for $\theta \leq 1 / \delta$ . Our choice $\theta \leq 1 / 6\delta$ is appropriate. Moreover, $h(x)$ is $(\delta + \frac{1}{\theta})$ -smooth. (2) gives $D_{h}(x,y) \geq \frac{1 - \theta\delta}{2\theta}\| x - y\|^2$ . Proposition 1 gives $D_{h}(x,y) \leq \frac{1 + \theta\delta}{2\theta}\| x - y\|^2$ . Let us write it down in a more convenient form:

$$
0 \leq \frac {1 - \theta \delta}{2 \theta} \| x - y \| ^ {2} \leq D _ {h} (x, y) \leq \frac {1 + \theta \delta}{2 \theta} \| x - y \| ^ {2}. \tag {31}
$$

Choose $\theta \leq \frac{p^{3 / 2}}{24\delta}$ . With such a choice we have

$$
\left(\frac {4 8 \theta \delta^ {2}}{p ^ {2}} - \frac {p}{6 \theta}\right) \| x _ {N} - x _ {0} \| ^ {2} \leq - \frac {p}{1 2} \| x _ {N} - x _ {0} \| ^ {2} \leq - \frac {2 \theta}{1 + \theta \delta} \frac {p}{1 2} D _ {h} (x _ {0}, x _ {N}) \leq - \frac {p}{7} D _ {h} (x _ {0}, x _ {N}).
$$

Finally, substituting it into (30) we obtain

$$
\mathbb {E} [ f (x _ {N}) - f (x) ] \leq \mathbb {E} \left[ p \left\langle x - x _ {0}, \nabla (f _ {1} - f) (x _ {N}) - \nabla (f _ {1} - f) (x _ {0}) + \frac {\tilde {x} _ {N} - x _ {0}}{\theta} \right\rangle - \frac {p}{7} D _ {h} (x _ {0}, x _ {N}) - \frac {\mu}{2} \| x _ {N} - x \| ^ {2} \right].
$$

![](images/f2ccfe5d6afd850bc07648a5a61ee2b763ced359e3f6f6b193e1eb5ae98e56ad.jpg)

# F Proof of Theorem 2

Theorem 4. (Theorem 2) Let the problem (1) be solved by Algorithm 4 with $\theta \leq \frac{p^{3/2}}{24\delta} \leq \frac{1}{6\delta}$ and such tuning of parameters that $28\alpha p\tau \leq \theta$ . Then the following inequality holds:

$$
\mathbb {E} \left[ Y _ {k + 1} + Z _ {k + 1} \right] \leq \mathbb {E} \left[ (1 - \tau) Y _ {k} + (1 + \mu \alpha) ^ {- 1} Z _ {k} \right].
$$

Proof. Let us move from one epoch to the method as a whole. Since the output point of Algorithm 3 was used to calculate $y_{k+1}$ , let us re-designate $x_{0} = x_{k+1}$ and $x_{N} = y_{k+1}$ in Lemma 5:

$$
\begin{array}{l} \mathbb {E} [ f (y _ {k + 1}) - f (x) ] \leq \mathbb {E} \left[ p \left\langle x - x _ {k + 1}, \nabla (f _ {1} - f) (y _ {k + 1}) - \nabla (f _ {1} - f) (x _ {k + 1}) + \frac {\tilde {y} _ {k + 1} - x _ {k + 1}}{\theta} \right\rangle - \frac {p}{7} D _ {h} (x _ {k + 1}, y _ {k + 1}) \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

See that $G_{k+1}$ from Line 8 is almost the same as $\nabla(f_1 - f)(y_{k+1}) - \nabla(f_1 - f)(x_{k+1}) + \frac{\tilde{y}_{k+1} - x_{k+1}}{\theta}$ in the expression above. Namely, $\nabla(f_1 - f)(y_{k+1}) - \nabla(f_1 - f)(x_{k+1}) + \frac{\tilde{y}_{k+1} - x_{k+1}}{\theta} = -G_{k+1}$ .

$$
\begin{array}{l} \mathbb {E} \left[ f \left(y _ {k + 1}\right) - f (x) \right] \leq \mathbb {E} \left[ \left\langle x - x _ {k + 1}, - G _ {k + 1} \right\rangle - \frac {p}{7} D _ {h} \left(x _ {k + 1}, y _ {k + 1}\right) - \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right] \\ = \mathbb {E} \left[ \langle z _ {k} - x _ {k + 1}, - G _ {k + 1} \rangle + \langle x - z _ {k}, - G _ {k + 1} \rangle - \frac {p}{7} D _ {h} (x _ {k + 1}, y _ {k + 1}) \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right] \\ = \mathbb {E} \left[ \langle z _ {k} - x _ {k + 1}, - G _ {k + 1} \rangle + \langle z _ {k} - x, G _ {k + 1} \rangle - \frac {p}{7} D _ {h} (x _ {k + 1}, y _ {k + 1}) \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right]. \tag {32} \\ \end{array}
$$

Let us rewrite Line 3 of Algorithm 4

$$
(1 - \tau) x _ {k + 1} = \tau (z _ {k} - x _ {k + 1}) + (1 - \tau) y _ {k} \Rightarrow z _ {k} - x _ {k + 1} = \frac {1 - \tau}{\tau} [ x _ {k + 1} - y _ {k} ].
$$

and substitute it into (32):

$$
\begin{array}{l} \mathbb {E} \left[ f \left(y _ {k + 1}\right) - f (x) \right] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} \left\langle x _ {k + 1} - y _ {k}, - G _ {k + 1} \right\rangle + \left\langle z _ {k} - x, G _ {k + 1} \right\rangle - \frac {p}{2} D _ {h} \left(x _ {k + 1}, y _ {k + 1}\right) \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

Next, apply Lemma 3:

$$
\langle z _ {k} - x, G _ {k + 1} \rangle - \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} \leq \langle z _ {k} - x, G _ {k + 1} \rangle - \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} + \frac {\mu}{2} \| y _ {k + 1} - z _ {k + 1} \| ^ {2}
$$

$$
\leq \frac {\alpha}{2} \| G _ {k + 1} \| ^ {2} + \frac {1}{2 \alpha} \| z _ {k} - x \| ^ {2} - \frac {1 + \mu \alpha}{2 \alpha} \| z _ {k + 1} - x \| ^ {2}.
$$

Combining the obtained results, we have

$$
\begin{array}{l} \mathbb {E} [ f (y _ {k + 1}) - f (x) ] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} \langle x _ {k + 1} - y _ {k}, - G _ {k + 1} \rangle - \frac {p}{2} D _ {h} (x _ {k + 1}, y _ {k + 1}) \right] \\ + \mathbb {E} \left[ - \frac {\mu}{2} \| y _ {k + 1} - x \| ^ {2} + \frac {\alpha}{2} \| \tilde {G} _ {k + 1} \| ^ {2} \right] \tag {33} \\ + \mathbb {E} \left[ \frac {1}{2 \alpha} \| z _ {k} - x \| ^ {2} - \frac {1 + \mu \alpha}{2 \alpha} \| z _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

Let us write Lemma 5 with $x = y_{k}$ :

$$
\mathbb {E} [ f (y _ {k + 1}) - f (y _ {k}) ] \leq \mathbb {E} \left[ p \langle y _ {k} - x _ {k + 1}, - G _ {k + 1} \rangle - \frac {p}{7} D _ {h} (x _ {k + 1}, y _ {k + 1}) \right]
$$

$$
- \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - y _ {k} \| ^ {2} \right].
$$

Rewrite it in the following form

$$
\begin{array}{l} \mathbb {E} [ p \left\langle x _ {k + 1} - y _ {k}, - G _ {k + 1} \right\rangle ] \leq \mathbb {E} \left[ (f (y _ {k + 1}) - f (y _ {k})) - \frac {p}{7} D _ {h} (x _ {k + 1}, y _ {k + 1}) \right] \\ - \mathbb {E} \left[ \frac {\mu}{2} \| y _ {k + 1} - y _ {k} \| ^ {2} \right]. \\ \end{array}
$$

and substitute into (33). We get

$$
\begin{array}{l} \mathbb {E} \left[ \frac {1}{\tau} (f (y _ {k + 1}) - f (x)) \right] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} (f (y _ {k}) - f (x)) - \frac {p}{7 \tau} D _ {h} (x _ {k + 1}, y _ {k + 1}) + \frac {\alpha}{2} \| G _ {k + 1} \| ^ {2} \right] \\ + \mathbb {E} \left[ \frac {1}{2 \alpha} \| z _ {k} - x \| ^ {2} - \frac {1 + \mu \alpha}{2 \alpha} \| z _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

The final hurdle to proof is the need to evaluate $\|G_{k+1}\|^{2}$ . The key idea is to put the "error" term out of $\|G_{k+1}\|^{2}$ and work with two terms.

$$
\frac {\alpha}{2} \| G _ {k + 1} \| ^ {2} \leq \frac {\alpha p ^ {2}}{\theta^ {2}} \| e _ {N} \| ^ {2} + \alpha p ^ {2} \| \nabla h (x _ {k + 1}) - \nabla h (y _ {k + 1}) \| ^ {2}.
$$

Let us estimate it by parts.

$$
\frac {\alpha p ^ {2}}{\theta^ {2}} \| e _ {N} \| ^ {2} \leq \frac {\alpha p ^ {2}}{\theta^ {2}} \frac {2 4 \theta^ {2} \delta^ {2}}{p ^ {2}} \frac {2 \theta}{1 - \theta \delta} D _ {h} (x _ {k + 1}, y _ {k + 1}) \leq \frac {2 8 8 \alpha \theta \delta^ {2}}{5} D _ {h} (x _ {k + 1}, y _ {k + 1}).
$$

Since $\theta \leq \frac{p^{3 / 2}}{24\delta}$ , we can estimate $\delta \leq \frac{p^{3 / 2}}{24\theta}$ and obtain:

$$
\frac {\alpha p ^ {2}}{\theta^ {2}} \| e _ {N} \| ^ {2} \leq \frac {\alpha p ^ {3}}{1 0 \theta} D _ {h} (x _ {k + 1}, y _ {k + 1}).
$$

Let us move on to the second term:

$$
\alpha p ^ {2} \| \nabla h (x _ {k + 1}) - \nabla h (y _ {k + 1}) \| ^ {2} \leq \alpha p ^ {2} \frac {2 (1 + \theta \delta)}{\theta} D _ {h} (x _ {k + 1}, y _ {k + 1}) \leq \frac {3 \alpha p ^ {2}}{\theta} D _ {h} (x _ {k + 1}, y _ {k + 1}).
$$

Here we used (29), (31) and smoothness of $h(x)$ . Put two terms together and get

$$
\begin{array}{l} \frac {\alpha}{2} \| G _ {k + 1} \| ^ {2} \leq \frac {4 \alpha p ^ {2}}{\theta} D _ {h} (x _ {k + 1}, y _ {k + 1}). \\ \mathbb {E} \left[ \frac {1}{\tau} (f (y _ {k + 1}) - f (x)) \right] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} (f (y _ {k}) - f (x)) + p \left(\frac {4 \alpha p}{\theta} - \frac {1}{7 \tau}\right) D _ {h} (x _ {k + 1}, y _ {k + 1}) \right] \\ + \mathbb {E} \left[ \frac {1}{2 \alpha} \| z _ {k} - x \| ^ {2} - \frac {1 + \mu \alpha}{2 \alpha} \| z _ {k + 1} - x \| ^ {2} \right]. \\ \end{array}
$$

Note that $28\alpha p\tau \leq \theta$ because of our choice of parameters. We obtain

$$
\mathbb {E} \left[ \frac {1}{\tau} (f (y _ {k + 1}) - f (x)) + \frac {1 + \mu \alpha}{2 \alpha} \| z _ {k + 1} - x \| ^ {2} \right] \leq \mathbb {E} \left[ \frac {1 - \tau}{\tau} (f (y _ {k}) - f (x) + \frac {1}{2 \alpha} \| z _ {k} - x \| ^ {2} \right].
$$

□

# G Additional Experiments

This section presents more runs of OLGA. First, we solve the problems (5) and (6) using the mushrooms dataset (Chang and Lin 2011) (Figures 7-12). This allows us to further show the robustness of the method to varying the parameters $\mu$ , L, $\delta$ over a wider range. Second, we present measurements of the training time of the distributed models (Figures 13-36). Computation time depends on system we work with. Therefore, we provide two runs of experiments: on local cluster with cable connection (fast connect), on remote CPUs with Internet connection (slow connect). Note that compression plays a more essential role in the first case.

# H Future Extensions

The methods we propose could be widely extended. Next we provide a few observations in this regard. In modern distributed learning, local data is often private and should not be compromised. As a result, one cannot use the full gradient, as data could be recovered (Weeraddana and Fischione 2017). This refers to the very popular federated learning setting. Since compression is a kind of stochasticity, we can add local additive noise to OLGA or EF-OLGA to preserve differential privacy (Nozari, Tallapragada, and Cortés 2016). We go into more detail below. Let us start with Algorithm 1. There are two options: to noise gradient before or after compression. The first means the use of

$$
\nabla f _ {m} (x, \xi^ {m}) = \nabla f _ {m} (x) + \xi^ {m}, \text { where } \xi^ {m} \sim N (0, \sigma_ {m} ^ {2})
$$

Then in Line 7, we have

$$
\widehat {g} _ {k} ^ {m} = \nabla f _ {m} (x _ {k}) - \nabla f _ {1} (x _ {k}) - \nabla f _ {m} (x _ {0}) + \nabla f _ {1} (x _ {0}) + \xi_ {k} ^ {m} - \xi_ {0} ^ {m},
$$

and Line 11 is

$$
t _ {k} = g _ {k} - \nabla f _ {1} (x _ {0}) + \nabla f (x _ {0}) + \frac {1}{M} \sum_ {m = 1} ^ {M} \xi_ {0} ^ {m}.
$$

The second option involves adding noise to compressed gradient. This means using

$$
g _ {k} ^ {m} = Q (\widehat {g} _ {k} ^ {m}) + \xi_ {k} ^ {m}
$$

in Line 8. In Algorithm 2 is an accelerating framework that does not use compression. Therefore, there is only one option of noising available:

$$
\nabla f (y _ {k + 1}) = \frac {1}{M} \sum_ {m = 1} ^ {M} \nabla f _ {m} (y _ {k + 1}) + \frac {1}{M} \sum_ {m = 1} ^ {M} \xi_ {k + 1} ^ {m}
$$

in Line 7. Algorithms 3 and 4 differ only in their handling of the compression error, and could be modified in a similar way. Proofs in the case of noisy gradients repeats our ones with addition of well-known techniques (Bylinkin, Degtyarev, and Beznosikov 2024).

We note again that variance reduction helps us to design methods with compression. The same approach is used to implement client sampling. This means that this technique could also be easily exploited in our algorithms. In Algorithm 1, Line 6 should be changed to

$$
\text { For   chosen } m _ {i _ {k}} \sim U [ 1, M ].
$$

Then in Algorithm 2 the gradient difference is approximated by

$$
t _ {k} = \nabla (f _ {1} - f _ {m _ {i _ {k}}}) (x _ {k + 1}) - \nabla (f _ {1} - f _ {m _ {i _ {k}}}) (y _ {k + 1}), m _ {i _ {k}} \sim U [ 1, M ].
$$

The same can be done in Algorithms 3 and 4. This approach opens the door to asynchronous setting, where the communication channels between devices and servers cannot be opened simultaneously.

# I Reproducibility Checklist

This paper:

- Includes a conceptual outline and/or pseudocode description of AI methods introduced. Yes   
- Clearly delineates statements that are opinions, hypothesis, and speculation from objective facts and results. Yes   
- Provides well marked pedagogical references for less-familiare readers to gain background necessary to replicate the paper.

Yes

Does this paper make theoretical contributions? Yes

- All assumptions and restrictions are stated clearly and formally. Yes   
- All novel claims are stated formally (e.g., in theorem statements). Yes

• Proofs of all novel claims are included. Yes   
- Proof sketches or intuitions are given for complex and/or novel results. Partial   
- Appropriate citations to theoretical tools used are given. Yes   
- All theoretical claims are demonstrated empirically to hold. Yes   
- All experimental code used to eliminate or disprove claims is included. No

Does this paper rely on one or more datasets? Yes

- A motivation is given for why the experiments are conducted on the selected datasets No, classical datasets for the theory validation.   
- All novel datasets introduced in this paper are included in a data appendix. NA, no new data   
- All novel datasets introduced in this paper will be made publicly available upon publication of the paper with a license that allows free usage for research purposes. NA, no new data   
- All datasets drawn from the existing literature (potentially including authors' own previously published work) are accompanied by appropriate citations. Yes   
- All datasets drawn from the existing literature (potentially including authors' own previously published work) are publicly available. Yes   
- All datasets that are not publicly available are described in detail, with explanation why publicly available alternatives are not scientifically satisficing. NA, no non-public data

Does this paper include computational experiments? Yes

- Any code required for pre-processing data is included in the appendix. No, simple experiments without pre-processing.   
- All source code required for conducting and analyzing the experiments is included in a code appendix. No, simple experiments - easy to reproduce.   
- All source code required for conducting and analyzing the experiments will be made publicly available upon publication of the paper with a license that allows free usage for research purposes. No, simple experiments - easy to reproduce.   
- All source code implementing new methods have comments detailing the implementation, with references to the paper where each step comes from. No, simple experiments - easy to reproduce.   
- If an algorithm depends on randomness, then the method used for setting seeds is described in a way sufficient to allow replication of results. No, simple experiments - easy to reproduce.   
- This paper specifies the computing infrastructure used for running experiments (hardware and software), including GPU/CPU models; amount of memory; operating system; names and versions of relevant software libraries and frameworks. No, simple experiments - easy to reproduce on laptops.   
- This paper formally describes evaluation metrics used and explains the motivation for choosing these metrics. No, simple experiments - easy to reproduce.   
- This paper states the number of algorithm runs used to compute each reported result. No, simple experiments - easy to reproduce.   
- Analysis of experiments goes beyond single-dimensional summaries of performance (e.g., average; median) to include measures of variation, confidence, or other distributional information. No, simple experiments - easy to reproduce.   
- The significance of any improvement or decrease in performance is judged using appropriate statistical tests (e.g., Wilcoxon signed-rank). No, simple experiments - easy to reproduce.   
- This paper lists all final (hyper-)parameters used for each model/algorithm in the paper's experiments. No, simple experiments - easy to reproduce.   
- This paper states the number and range of values tried per (hyper-) parameter during development of the paper, along with the criterion used for selecting the final parameter setting. No, simple experiments - easy to reproduce.

![](images/edf0e016bf8ba42d837020db3004bda658a87ab45e235918eaf8fd9d2dde499a.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 200               | 1e-6 | 1e-2    | 1e-2             | 1e-2   | 1e-2   |
| 400               | 1e-8 | 1e-2    | 1e-3             | 1e-3   | 1e-4   |
| 600               | 1e-8 | 1e-2    | 1e-4             | 1e-4   | 1e-6   |
| 800               | 1e-8 | 1e-2    | 1e-5             | 1e-5   | 1e-7   |
| 1000              | 1e-8 | 1e-2    | 1e-6             | 1e-6   | 1e-7   |
| 1200              | 1e-8 | 1e-2    | 1e-6             | 1e-6   | 1e-7   |
</details>

(a) $\omega = 112.00$

![](images/1e6c267be639b1913a16243d6f694c7a6cb34bc904be5d74aea37cad72a67ac2.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^1 | 10^1    | 10^1             | 10^1   | 10^1   |
| 200               | 10^-5| 10^-1   | 10^-1            | 10^-2  | 10^-2  |
| 400               | 10^-5| 10^-1   | 10^-2            | 10^-3  | 10^-3  |
| 600               | 10^-5| 10^-1   | 10^-3            | 10^-4  | 10^-4  |
| 800               | 10^-5| 10^-1   | 10^-4            | 10^-5  | 10^-5  |
| 1000              | 10^-5| 10^-1   | 10^-5            | 10^-5  | 10^-5  |
</details>

(b) ω = 11.20

![](images/d0b738295eba51b5f3f9c8d8eaabd5b6d3942b5b5018570bdfdf56b2aa00d0a5.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^-1 | 10^-1   | 10^-1            | 10^-1  | 10^-1  |
| 200               | 10^-3 | 10^-1   | 10^-1            | 10^-1  | 10^-2  |
| 400               | 10^-5 | 10^-1   | 10^-1            | 10^-3  | 10^-3  |
| 600               | 10^-6 | 10^-1   | 10^-1            | 10^-4  | 10^-5  |
| 800               | 10^-6 | 10^-1   | 10^-1            | 10^-5  | 10^-6  |
| 1000              | 10^-6 | 10^-1   | 10^-1            | 10^-5  | 10^-6  |
| 1200              | 10^-6 | 10^-1   | 10^-1            | 10^-5  | 10^-6  |
</details>

(c) $\omega = 5.6$   
Figure 7: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with $M = 75$ and mushrooms dataset. The criterion is the communication time ( $CC - 3$ ). For methods with compression we vary the power of compression $\omega$ .

![](images/c85a9eb10cc0d6368cca9db48c8ce49b27c2b20001c36b69de5a67eed101182e.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^2 | 10^2    | 10^2             | 10^2   | 10^2   |
| 200               | 10^-4| 10^-2   | 10^-2            | 10^-2  | 10^-2  |
| 400               | 10^-6| 10^-3   | 10^-3            | 10^-3  | 10^-3  |
| 600               | 10^-8| 10^-4   | 10^-4            | 10^-4  | 10^-4  |
| 800               | 10^-8| 10^-5   | 10^-5            | 10^-5  | 10^-5  |
| 1000              | 10^-8| 10^-6   | 10^-6            | 10^-6  | 10^-6  |
| 1200              | 10^-8| 10^-7   | 10^-7            | 10^-7  | 10^-7  |
</details>

(a) $\omega = 112.00$

![](images/1a34e8c6fc865c1e0621ee16548fd1d92dd7fba4767f937b5bd150cbafee339a.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^1 | 10^1    | 10^1             | 10^1   | 10^1   |
| 200               | 10^-3| 10^0    | 10^-1            | 10^-1  | 10^-1  |
| 400               | 10^-5| 10^-1   | 10^-2            | 10^-2  | 10^-2  |
| 600               | 10^-5| 10^-2   | 10^-3            | 10^-3  | 10^-3  |
| 800               | 10^-5| 10^-3   | 10^-4            | 10^-4  | 10^-4  |
| 1000              | 10^-5| 10^-4   | 10^-5            | 10^-5  | 10^-5  |
</details>

(b) $\omega = 11.20$

![](images/41f62e11b3db9924e0dedeea45df5b58a07465b761ec733f3bdcd34993ee1706.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 200               | 0.01 | 0.01    | 0.01             | 0.01   | 0.01   |
| 400               | 0.0001 | 0.001   | 0.001            | 0.001  | 0.001  |
| 600               | 0.00001 | 0.0001  | 0.0001           | 0.0001 | 0.0001 |
| 800               | 0.000001 | 0.00001 | 0.00001          | 0.00001 | 0.00001 |
| 1000              | 0.000001 | 0.00001 | 0.00001          | 0.00001 | 0.00001 |
| 1200              | 0.000001 | 0.00001 | 0.00001          | 0.00001 | 0.00001 |
</details>

(c) $\omega = 5.6$   
Figure 8: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 50 and mushrooms dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .

![](images/1d463920f96c3e06c959863fd5e21f34cd03bf29df9f4c7e07e98acc1e808413.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 200               | 1e-8 | 1.0     | 1e-2             | 1e-2   | 1e-4   |
| 400               | 1e-8 | 1.0     | 1e-3             | 1e-3   | 1e-5   |
| 600               | 1e-8 | 1.0     | 1e-4             | 1e-4   | 1e-6   |
| 800               | 1e-8 | 1.0     | 1e-5             | 1e-5   | 1e-7   |
| 1000              | 1e-8 | 1.0     | 1e-6             | 1e-6   | 1e-8   |
| 1200              | 1e-8 | 1.0     | 1e-7             | 1e-7   | 1e-9   |
</details>

(a) $\omega = 112.00$

![](images/b0d3ef16190c31143054d53dda074f03366a84c286c00fc37b0e6d93f5644ffc.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^0 | 10^0    | 10^0             | 10^0   | 10^0   |
| 200               | 10^-4| 10^-1   | 10^-1            | 10^-2  | 10^-2  |
| 400               | 10^-5| 10^-1   | 10^-2            | 10^-3  | 10^-3  |
| 600               | 10^-6| 10^-1   | 10^-3            | 10^-4  | 10^-4  |
| 800               | 10^-6| 10^-1   | 10^-4            | 10^-5  | 10^-5  |
| 1000              | 10^-6| 10^-1   | 10^-5            | 10^-6  | 10^-6  |
</details>

(b) $\omega = 11.20$

![](images/641eaf03659189ae27e087623d05e7aa0ff2057a0ff3b19e914a3c3e712f13d2.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^0 | 10^0    | 10^0             | 10^0   | 10^0   |
| 200               | 10^-2| 10^-1   | 10^-1            | 10^-2  | 10^-2  |
| 400               | 10^-4| 10^-1   | 10^-2            | 10^-3  | 10^-3  |
| 600               | 10^-6| 10^-1   | 10^-3            | 10^-4  | 10^-4  |
| 800               | 10^-6| 10^-1   | 10^-4            | 10^-5  | 10^-5  |
| 1000              | 10^-6| 10^-1   | 10^-5            | 10^-5  | 10^-5  |
| 1200              | 10^-6| 10^-1   | 10^-5            | 10^-6  | 10^-5  |
</details>

(c) $\omega = 5.6$   
Figure 9: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 25 and mushrooms dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .

![](images/015fc39375a6a8bc6c60eec7ba6184a96c1eb485b2a1bca7d984eece0f8c3f08.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^2 | 10^2    | 10^2             | 10^2   | 10^2   |
| 500               | 10^-4| 10^2    | 10^0             | 10^-2  | 10^-4  |
| 1000              | 10^-6| 10^2    | 10^-2            | 10^-3  | 10^-6  |
| 1500              | 10^-8| 10^2    | 10^-3            | 10^-4  | 10^-8  |
| 2000              | 10^-10| 10^2    | 10^-4            | 10^-5  | 10^-9  |
</details>

(a) $\omega = 112.00$

![](images/b9f05d0988f40e8a5588eaa34b0442a3e021bae74747dbde64a8a67a27f0ae4c.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^1 | 10^1    | 10^1             | 10^1   | 10^1   |
| 500               | 10^-3| 10^1    | 10^-1            | 10^-3  | 10^-4  |
| 1000              | 10^-5| 10^1    | 10^-2            | 10^-4  | 10^-6  |
| 1500              | 10^-6| 10^1    | 10^-3            | 10^-5  | 10^-7  |
| 2000              | 10^-7| 10^1    | 10^-4            | 10^-6  | 10^-8  |
| 2500              | 10^-8| 10^1    | 10^-5            | 10^-7  | 10^-9  |
</details>

(b) $\omega = 11.20$

![](images/a2dc35a7ffae7169fa9f030679af0e5d3be0168c9cf2146fb95a65370a6a4900.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^1 | 10^1    | 10^1             | 10^1   | 10^1   |
| 500               | ~10^-3 | ~10^1   | ~10^-1           | ~10^-3 | ~10^-4 |
| 1000              | ~10^-5 | ~10^1   | ~10^-2           | ~10^-5 | ~10^-6 |
| 1500              | ~10^-6 | ~10^1   | ~10^-3           | ~10^-6 | ~10^-7 |
| 2000              | ~10^-7 | ~10^1   | ~10^-4           | ~10^-7 | ~10^-8 |
| 2500              | ~10^-8 | ~10^1   | ~10^-5           | ~10^-8 | ~10^-9 |
</details>

(c) $\omega = 5.6$   
Figure 10: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 75 and mushrooms dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .

![](images/497ec66c37566fc423a6ff0c6327b75a780714a77c8e9b589595308cc2d368bf.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 100  | 100     | 100              | 100    | 100    |
| 500               | 1e-4 | 1       | 1                | 1e-4   | 1e-6   |
| 1000              | 1e-8 | 1       | 1                | 1e-6   | 1e-8   |
| 1500              | 1e-10| 1       | 1                | 1e-7   | 1e-9   |
| 2000              | 1e-10| 1       | 1                | 1e-8   | 1e-10  |
</details>

(a) $\omega = 112.00$

![](images/85ff7686d5071ded97fbf5e819b37ca8a1f295cb3008c3bd44008fcf132c08d8.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^1 | 10^1    | 10^1             | 10^1   | 10^1   |
| 500               | 10^-3| 10^-1   | 10^-2            | 10^-3  | 10^-5  |
| 1000              | 10^-5| 10^-1   | 10^-4            | 10^-5  | 10^-7  |
| 1500              | 10^-7| 10^-1   | 10^-6            | 10^-6  | 10^-9  |
| 2000              | 10^-9| 10^-1   | 10^-8            | 10^-7  | 10^-11 |
| 2500              | 10^-11| 10^-1   | 10^-10           | 10^-8  | 10^-11 |
</details>

(b) $\omega = 11.20$

![](images/456c4ddf8654fb4057117d3cc796e91a8f69136eec9e2795a4d733f60def4d29.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 100  | 100     | 100              | 100    | 100    |
| 500               | 0.0001 | 1       | 0.1              | 0.01   | 0.00001 |
| 1000              | 0.00001 | 1       | 0.01             | 0.001  | 0.000001 |
| 1500              | 0.000001 | 1       | 0.001            | 0.0001 | 0.0000001 |
| 2000              | 0.0000001 | 1       | 0.0001           | 0.00001 | 0.00000001 |
</details>

(c) $\omega = 5.6$   
Figure 11: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 50 and mushrooms dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .

![](images/1ebc25efc516240daf2cef5af14adf2ca1d5d8baef55571c584c66062da4aa14.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^2 | 10^2    | 10^2             | 10^2   | 10^2   |
| 250               | ~10^-3 | ~10^2   | ~10^1            | ~10^-2 | ~10^-3 |
| 500               | ~10^-6 | ~10^2   | ~10^0            | ~10^-3 | ~10^-6 |
| 750               | ~10^-8 | ~10^2   | ~10^-1           | ~10^-4 | ~10^-8 |
| 1000              | ~10^-10 | ~10^2   | ~10^-2           | ~10^-5 | ~10^-9 |
| 1250              | ~10^-11 | ~10^2   | ~10^-3           | ~10^-6 | ~10^-10 |
| 1500              | ~10^-12 | ~10^2   | ~10^-4           | ~10^-7 | ~10^-11 |
</details>

(a) $\omega = 112.00$

![](images/76bb7c1619fc977c2a49350b97203b23b237b5c025ce450ac45c6287e7a231f7.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 100  | 100     | 100              | 100    | 100    |
| 1000              | 1e-6 | 100     | 1e-6             | 1e-6   | 1e-6   |
| 2000              | 1e-10| 100     | 1e-10            | 1e-10  | 1e-10  |
| 3000              | 1e-14| 100     | 1e-14            | 1e-14  | 1e-14  |
| 4000              | 1e-18| 100     | 1e-18            | 1e-18  | 1e-18  |
</details>

(b) $\omega = 11.20$

![](images/9870c938087c810d004f0ae4ac5d502289e9dd12c9ce79891eee7987838ff082.jpg)

<details>
<summary>line</summary>

| Communicated time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ----------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0                 | 10^1 | 10^1    | 10^1             | 10^1   | 10^1   |
| 1000              | 10^-8 | 10^1    | 10^-5            | 10^-5  | 10^-8  |
| 2000              | 10^-11| 10^1    | 10^-7            | 10^-7  | 10^-11 |
| 3000              | 10^-14| 10^1    | 10^-9            | 10^-9  | 10^-14 |
| 4000              | 10^-17| 10^1    | 10^-11           | 10^-11 | 10^-17 |
</details>

(c) $\omega = 5.6$   
Figure 12: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 25 and mushrooms dataset. The criterion is the communication time (CC-3). For methods with compression we vary the power of compression $\omega$ .

![](images/30cb4b45c2a8f27d3de85dbb13f6590d03fd7e0d362d17faa6df0d6e449917ac.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.000         | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.005         | 0.1        | 1.0        | 0.1              | 0.1        | 0.1        |
| 0.010         | 0.01       | 1.0        | 0.01             | 0.01       | 0.01       |
| 0.015         | 0.001      | 1.0        | 0.001            | 0.001      | 0.001      |
| 0.020         | 0.0001     | 1.0        | 0.0001           | 0.0001     | 0.0001     |
</details>

(a) $\omega = 112.00$

![](images/d0f01b243043b43c03b93dba5d4837d8751c20cc7bd1e534cdafe8cd66db7625.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 0.005         | 0.01 | 1.0     | 0.1              | 0.1    | 0.1    |
| 0.010         | 0.001| 1.0     | 0.1              | 0.01   | 0.01   |
| 0.015         | 0.0001| 1.0    | 0.1              | 0.001  | 0.001  |
| 0.020         | 0.00001| 1.0   | 0.1              | 0.0001 | 0.0001 |
</details>

(b) $\omega = 11.20$

![](images/542ecb8c2e18b73e79412dfdd1193e5595b8af9dec65479e508fa0f32fa8a8d1.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 0.005         | 0.1  | 1.0     | 0.1              | 0.1    | 0.1    |
| 0.010         | 0.01 | 1.0     | 0.1              | 0.01   | 0.01   |
| 0.015         | 0.001| 1.0     | 0.1              | 0.001  | 0.001  |
| 0.020         | 0.0001| 1.0    | 0.1              | 0.0001 | 0.0001 |
</details>

(c) $\omega = 5.6$   
Figure 13: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 75 and mushrooms dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/407eebf47c7933e656597a9c2be183e897449fede35efa0e1938d17c885bf766.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS   | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | --------- | ---------------- | ---------- | ---------- |
| 0.0           | 10^1       | 10^1      | 10^1             | 10^1       | 10^1       |
| 0.2           | 10^-3      | 10^0      | 10^-1            | 10^-1      | 10^0       |
| 0.4           | 10^-4      | 10^0      | 10^-2            | 10^-2      | 10^1       |
| 0.6           | 10^-4      | 10^0      | 10^-2            | 10^-2      | 10^0       |
| 0.8           | 10^-4      | 10^0      | 10^-2            | 10^-2      | 10^1       |
</details>

(a) $\omega = 112.00$

![](images/b0ffea468f58930cae8cf48e1f0a84ad21336b0d29fea008270adf40b8b1ccfc.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 10^1       | 10^1       | 10^1             | 10^1       | 10^1       |
| 0.2           | 10^-3      | 10^0       | 10^-1            | 10^-1      | 10^0       |
| 0.4           | 10^-4      | 10^0       | 10^-3            | 10^-2      | 10^1       |
| 0.6           | 10^-4      | 10^0       | 10^-3            | 10^-2      | 10^0       |
| 0.8           | 10^-4      | 10^0       | 10^-3            | 10^-2      | 10^0       |
</details>

(b) $\omega = 11.20$

![](images/9a92fffa103926f38afdf5337fc3ad1438fcfc136c46967d3e36e807686c0efd.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 10^1       | 10^1       | 10^1             | 10^1       | 10^1       |
| 0.2           | 10^-3      | 10^0       | 10^-1            | 10^-1      | 10^0       |
| 0.4           | 10^-4      | 10^0       | 10^-3            | 10^-2      | 10^1       |
| 0.6           | 10^-4      | 10^0       | 10^-3            | 10^-2      | 10^0       |
| 0.8           | 10^-4      | 10^0       | 10^-3            | 10^-2      | 10^0       |
</details>

(c) $\omega = 5.6$   
Figure 14: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 75 and mushrooms dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/d1b0542d0ce71105d58f44d7bf0ec966d3bfa915bfe1a5a74243af9b18a2dec9.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 10^0 | 10^0    | 10^0             | 10^0   | 10^0   |
| 0.005         | 10^-2| 10^-1   | 10^-1            | 10^-1  | 10^-1  |
| 0.010         | 10^-4| 10^-2   | 10^-2            | 10^-2  | 10^-2  |
| 0.015         | 10^-4| 10^-3   | 10^-3            | 10^-3  | 10^-3  |
| 0.020         | 10^-4| 10^-4   | 10^-4            | 10^-4  | 10^-4  |
</details>

(a) $\omega = 112.00$

![](images/e1b11413bc1c349104ada6c890b39d684680f1f77c56d36bf36e4da404633c69.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000 | 10^0 | 10^0 | 10^0 | 10^0 | 10^0 |
| 0.005 | 10^-1 | 10^-1 | 10^-1 | 10^-1 | 10^-1 |
| 0.010 | 10^-2 | 10^-2 | 10^-2 | 10^-2 | 10^-2 |
| 0.015 | 10^-3 | 10^-3 | 10^-3 | 10^-3 | 10^-3 |
| 0.020 | 10^-4 | 10^-4 | 10^-4 | 10^-4 | 10^-4 |
</details>

(b) $\omega = 11.20$

![](images/582e5797ba699680798a582ca2cbb0d05aaf6f6281a78bd366a9b629879a4964.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 10^1 | 10^1    | 10^1             | 10^1   | 10^1   |
| 0.005         | 10^-1| 10^0    | 10^-1            | 10^-1  | 10^-1  |
| 0.010         | 10^-2| 10^-1   | 10^-2            | 10^-2  | 10^-2  |
| 0.015         | 10^-3| 10^-2   | 10^-3            | 10^-3  | 10^-3  |
| 0.020         | 10^-4| 10^-3   | 10^-4            | 10^-4  | 10^-4  |
</details>

(c) $\omega = 5.6$   
Figure 15: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 50 and mushrooms dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/657c2ab6e354d5d86918be6235d1c5d6692fe56a4aca44f7d01eea8a8d709e3a.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 10^0       | 10^0       | 10^0             | 10^0       | 10^0       |
| 0.1           | 10^-2      | 10^-1      | 10^-1            | 10^-1      | 10^-1      |
| 0.2           | 10^-3      | 10^-2      | 10^-2            | 10^-2      | 10^-2      |
| 0.3           | 10^-4      | 10^-3      | 10^-3            | 10^-3      | 10^-3      |
| 0.4           | 10^-4      | 10^-4      | 10^-4            | 10^-4      | 10^-4      |
| 0.5           | 10^-4      | 10^-4      | 10^-4            | 10^-4      | 10^-4      |
| 0.6           | 10^-4      | 10^-4      | 10^-4            | 10^-4      | 10^-4      |
</details>

(a) $\omega = 112.00$

![](images/0027d379134d605a94c6b5aef2ae2a208af1bb059184587de73514242234f91d.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.1           | 0.01       | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.2           | 0.001      | 0.01       | 0.01             | 0.01       | 0.01       |
| 0.3           | 0.0001     | 0.001      | 0.001            | 0.001      | 0.001      |
| 0.4           | 0.00001    | 0.0001     | 0.0001           | 0.0001     | 0.0001     |
| 0.5           | 0.000001   | 0.00001    | 0.00001          | 0.00001    | 0.00001    |
| 0.6           | 0.0000001  | 0.000001   | 0.000001         | 0.000001   | 0.000001   |
</details>

(b) $\omega = 11.20$

![](images/34137907fdb2de56b679afaa57cb31901965a5ef8afa41cbd96fc4f6ff452313.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.1           | 0.01       | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.2           | 0.001      | 0.01       | 0.01             | 0.01       | 0.01       |
| 0.3           | 0.0001     | 0.001      | 0.001            | 0.001      | 0.001      |
| 0.4           | 0.00001    | 0.0001     | 0.0001           | 0.0001     | 0.0001     |
| 0.5           | 0.000001   | 0.00001    | 0.00001          | 0.00001    | 0.00001    |
| 0.6           | 0.0000001  | 0.000001   | 0.000001         | 0.000001   | 0.000001   |
</details>

(c) $\omega = 5.6$   
Figure 16: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 50 and mushrooms dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/ee1d81707c02b85d37fe62b5986c40308b86d48e352ec7358606b8f788ace1d5.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.000         | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.005         | 0.01       | 1.0        | 0.1              | 0.1        | 0.1        |
| 0.010         | 0.001      | 1.0        | 0.01             | 0.01       | 0.01       |
| 0.015         | 0.0001     | 1.0        | 0.001            | 0.001      | 0.001      |
| 0.020         | 0.00001    | 1.0        | 0.0001           | 0.0001     | 0.0001     |
</details>

(a) ω = 112.00

![](images/51fd78a548f8edc9f89008a494fb94bdcc6398f4db224618c329009bef5397b2.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.000         | 1.0000     | 1.0000     | 1.0000           | 1.0000     | 1.0000     |
| 0.005         | 0.0100     | 0.1000     | 0.1000           | 0.1000     | 0.1000     |
| 0.010         | 0.0010     | 0.0100     | 0.0100           | 0.0100     | 0.0100     |
| 0.015         | 0.0001     | 0.0010     | 0.0010           | 0.0010     | 0.0010     |
| 0.020         | 0.00001    | 0.0001     | 0.0001           | 0.0001     | 0.0001     |
</details>

(b) ω = 11.20

![](images/bcbc28c078761a6cd9474efe5ddc857f93e12796b35cf08731b5b125f6ec57a7.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 0.005         | 0.01 | 1.0     | 0.1              | 0.1    | 0.1    |
| 0.010         | 0.001| 1.0     | 0.01             | 0.01   | 0.01   |
| 0.015         | 0.0001| 1.0    | 0.001            | 0.001  | 0.001  |
| 0.020         | 0.00001| 1.0    | 0.0001           | 0.0001 | 0.0001 |
</details>

(c) ω = 5.6   
Figure 17: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 25 and mushrooms dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/07c6d332458c5c962a032ab9a0bcb277b89ade49751b844fff7f610b6b2929dc.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.00          | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.05          | 0.0001     | 0.1        | 0.01             | 0.1        | 1.0        |
| 0.10          | 0.0001     | 0.1        | 0.001            | 0.1        | 1.0        |
| 0.15          | 0.0001     | 0.1        | 0.001            | 0.1        | 1.0        |
| 0.20          | 0.0001     | 0.1        | 0.001            | 0.1        | 1.0        |
| 0.25          | 0.0001     | 0.1        | 0.001            | 0.1        | 1.0        |
| 0.30          | 0.0001     | 0.1        | 0.001            | 0.1        | 1.0        |
</details>

(a) ω = 112.00

![](images/8d9c84e8cc0c6c5191425742100dcd9e73aabd095066e3d7052d7591ea954c71.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.00          | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.05          | 0.01       | 0.1        | 0.01             | 0.1        | 0.1        |
| 0.10          | 0.0001     | 0.01       | 0.0001           | 0.01       | 0.01       |
| 0.15          | 0.00001    | 0.001      | 0.00001          | 0.001      | 0.001      |
| 0.20          | 0.000001   | 0.0001     | 0.000001         | 0.0001     | 0.0001     |
| 0.25          | 0.0000001  | 0.00001    | 0.0000001        | 0.00001    | 0.00001    |
| 0.30          | 0.00000001 | 0.000001   | 0.00000001       | 0.000001   | 0.000001   |
</details>

(b) ω = 11.20

![](images/da6056af33acb1c805025ed365fee7890f8ee47a88a34b35de7a80407aaf3ce3.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.00          | 1.00E+00   | 1.00E+00   | 1.00E+00         | 1.00E+00   | 1.00E+00   |
| 0.05          | 1.00E-02   | 1.00E-01   | 1.00E-02         | 1.00E-01   | 1.00E-01   |
| 0.10          | 1.00E-03   | 1.00E-02   | 1.00E-03         | 1.00E-02   | 1.00E-02   |
| 0.15          | 1.00E-04   | 1.00E-03   | 1.00E-04         | 1.00E-03   | 1.00E-03   |
| 0.20          | 1.00E-05   | 1.00E-04   | 1.00E-05         | 1.00E-04   | 1.00E-04   |
| 0.25          | 1.00E-06   | 1.00E-05   | 1.00E-06         | 1.00E-05   | 1.00E-05   |
| 0.30          | 1.00E-07   | 1.00E-06   | 1.00E-07         | 1.00E-06   | 1.00E-06   |
</details>

(c) $\omega = 5.6$   
Figure 18: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 25 and mushrooms dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/6184606a666e783409da76f6df4b40be599b5b0c1e3dddc311efb2f46dc114c9.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 0.005         | 0.001| 0.1     | 0.1              | 0.1    | 0.1    |
| 0.010         | 0.0001| 0.1     | 0.1              | 0.01   | 0.01   |
| 0.015         | 0.00001| 0.1    | 0.1              | 0.001  | 0.001  |
| 0.020         | 0.00001| 0.1    | 0.1              | 0.001  | 0.001  |
</details>

(a) ω = 112.00

![](images/a285947f0887b1f9fe9a6e95f2fb06cd56e8694be5fdc473cf294abfb4a027f6.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 0.005         | 0.001| 0.1     | 0.1              | 0.1    | 0.1    |
| 0.010         | 0.0001| 0.05   | 0.05             | 0.05   | 0.05   |
| 0.015         | 0.00001| 0.01   | 0.01             | 0.01   | 0.01   |
| 0.020         | 0.000001| 0.01   | 0.01             | 0.01   | 0.01   |
</details>

(b) ω = 11.20

![](images/ae97e8031d1777726dc874a5fc5fd54c2ee426dc6485acb1c91a9f16fddf8f09.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.005 | 0.01 | 0.1 | 0.1 | 0.1 | 0.1 |
| 0.010 | 0.001 | 0.1 | 0.1 | 0.01 | 0.01 |
| 0.015 | 0.0001 | 0.1 | 0.1 | 0.001 | 0.001 |
| 0.020 | 0.0001 | 0.1 | 0.1 | 0.001 | 0.001 |
</details>

(c) ω = 5.6   
Figure 19: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with $M = 75$ and mushrooms dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/72f12972de09b1b40de713d184a922924369851a997de9ba484428df15bf607f.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.2           | 0.01       | 0.01       | 0.01             | 0.1        | 0.1        |
| 0.4           | 0.001      | 0.001      | 0.001            | 0.01       | 0.01       |
| 0.6           | 0.0001     | 0.0001     | 0.0001           | 0.001      | 0.001      |
| 0.8           | 0.00001    | 0.00001    | 0.00001          | 0.0001     | 0.0001     |
</details>

(a) $\omega = 112.00$

![](images/8a56fd7e260080af66b111af523f71aac77eda9f324d8a02fe4e1593e959a6da.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.2           | 0.01       | 0.01       | 0.01             | 0.1        | 0.1        |
| 0.4           | 0.001      | 0.001      | 0.001            | 0.01       | 0.01       |
| 0.6           | 0.0001     | 0.0001     | 0.0001           | 0.001      | 0.001      |
| 0.8           | 0.00001    | 0.00001    | 0.00001          | 0.0001     | 0.0001     |
</details>

(b) ω = 11.20

![](images/19e0cd8959dc75242791fffa1f2dbb460c9a412f40cb879b0c69862e4d48e0f9.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.2 | 0.01 | 0.01 | 0.01 | 0.1 | 0.1 |
| 0.4 | 0.001 | 0.001 | 0.001 | 0.01 | 0.01 |
| 0.6 | 0.0001 | 0.0001 | 0.0001 | 0.001 | 0.001 |
| 0.8 | 0.00001 | 0.00001 | 0.00001 | 0.0001 | 0.0001 |
</details>

(c) $\omega = 5.6$   
Figure 20: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 75 and mushrooms dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/c9e7803fe8892be5204713254313fd46a8da17f0ea4fa2e3bfadd213230b5b39.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.000         | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.005         | 0.0001     | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.010         | 0.0001     | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.015         | 0.0001     | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.020         | 0.0001     | 0.1        | 0.1              | 0.1        | 0.1        |
</details>

(a) $\omega = 112.00$

![](images/f75dd54d2160728ad52abe293a1776ce2adee9beeef9536a0b0f5f5ecd1855b6.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.000         | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.005         | 0.0001     | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.010         | 0.00001    | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.015         | 0.00001    | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.020         | 0.00001    | 0.1        | 0.1              | 0.1        | 0.1        |
</details>

(b) $\omega = 11.20$

![](images/7d9fa8f3df94d4720b423053393125c95d0fe7642f47266ee2948f0115002f48.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 0.005         | 0.01 | 0.1     | 0.1              | 0.1    | 0.1    |
| 0.010         | 0.001| 0.01    | 0.01             | 0.01   | 0.01   |
| 0.015         | 0.0001| 0.001  | 0.001            | 0.001  | 0.001  |
| 0.020         | 0.00001| 0.0001 | 0.0001           | 0.0001 | 0.0001 |
</details>

(c) $\omega = 5.6$   
Figure 21: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 50 and mushrooms dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/80e001ffaabbe283d1ddbbe370a41b04c73ff2daa051749c6fb7b9467f116df3.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.2           | 0.001      | 0.01       | 0.01             | 0.1        | 0.1        |
| 0.4           | 0.0001     | 0.001      | 0.001            | 0.01       | 0.01       |
| 0.6           | 0.00001    | 0.0001     | 0.0001           | 0.001      | 0.001      |
| 0.8           | 0.000001   | 0.00001    | 0.00001          | 0.0001     | 0.0001     |
</details>

(a) $\omega = 112.00$

![](images/139c437c5f37fc8f71bdd77b442a1695078915256ec070c8fd55898eb3475a23.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.2 | 0.0001 | 0.01 | 0.01 | 0.1 | 0.1 |
| 0.4 | 0.0001 | 0.0001 | 0.001 | 0.01 | 0.01 |
| 0.6 | 0.0001 | 0.0001 | 0.001 | 0.01 | 0.01 |
| 0.8 | 0.0001 | 0.0001 | 0.001 | 0.01 | 0.01 |
</details>

(b) $\omega = 11.20$

![](images/06ff6663a783201981abcc9b1839f2b5e3623295dda0e397b9832ddab130cb1e.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 0.2 | 0.01 | 0.01 | 0.01 | 0.01 | 0.1 |
| 0.4 | 0.001 | 0.001 | 0.001 | 0.001 | 0.01 |
| 0.6 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.001 |
| 0.8 | 0.00001 | 0.00001 | 0.00001 | 0.00001 | 0.0001 |
</details>

(c) $\omega = 5.6$   
Figure 22: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 50 and mushrooms dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/58160977c4fa8cfe29ae330604ed26f740def920f4001ccf969b6bce2a9dca53.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 0.005         | 0.001| 0.1     | 0.1              | 0.01   | 0.1    |
| 0.010         | 0.0001| 0.01   | 0.01             | 0.001  | 0.01   |
| 0.015         | 0.00001| 0.001 | 0.001            | 0.0001 | 0.001  |
| 0.020         | 0.00001| 0.001 | 0.001            | 0.0001 | 0.001  |
</details>

(a) $\omega = 112.00$

![](images/4442895352ac81ad81b1922f538b5236fc55a6bf64bcbdc07d6f0388b6533773.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 0.005         | 0.0001 | 0.1     | 0.1              | 0.01   | 0.1    |
| 0.010         | 0.0001 | 0.1     | 0.1              | 0.01   | 0.01   |
| 0.015         | 0.0001 | 0.1     | 0.1              | 0.001  | 0.001  |
| 0.020         | 0.0001 | 0.1     | 0.1              | 0.001  | 0.001  |
</details>

(b) $\omega = 11.20$

![](images/ad063afb80c77568031d39bbc9bfe695c791fc83d11c8dd1d9f23859a30fb4a1.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1.0  | 1.0     | 1.0              | 1.0    | 1.0    |
| 0.005         | 0.01 | 0.1     | 0.1              | 0.01   | 0.1    |
| 0.010         | 0.001| 0.01    | 0.01             | 0.001  | 0.01   |
| 0.015         | 0.0001| 0.001  | 0.001            | 0.0001 | 0.001  |
| 0.020         | 0.0001| 0.001  | 0.001            | 0.0001 | 0.001  |
</details>

(c) $\omega = 5.6$   
Figure 23: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 25 and mushrooms dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/4af820926f3800ebaccba8b6ef9dafb832d07382d35050c79785e9cd237ec015.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.1           | 0.0001     | 0.0001     | 0.01             | 0.1        | 0.1        |
| 0.2           | 0.0001     | 0.0001     | 0.001            | 0.01       | 0.01       |
| 0.3           | 0.0001     | 0.0001     | 0.0001           | 0.001      | 0.001      |
| 0.4           | 0.0001     | 0.0001     | 0.0001           | 0.001      | 0.001      |
| 0.5           | 0.0001     | 0.0001     | 0.0001           | 0.001      | 0.001      |
| 0.6           | 0.0001     | 0.0001     | 0.0001           | 0.001      | 0.001      |
| 0.7           | 0.0001     | 0.0001     | 0.0001           | 0.001      | 0.001      |
| 0.8           | 0.0001     | 0.0001     | 0.0001           | 0.001      | 0.001      |
</details>

(a) $\omega = 112.00$

![](images/ed2a2f24404ee861b45223dbf541acf2ce631053fc0b78606f752f7bff521cb6.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.1           | 0.0001     | 0.01       | 0.01             | 0.1        | 0.1        |
| 0.2           | 0.0001     | 0.0001     | 0.001            | 0.01       | 0.01       |
| 0.3           | 0.0001     | 0.0001     | 0.001            | 0.01       | 0.01       |
| 0.4           | 0.0001     | 0.0001     | 0.001            | 0.01       | 0.01       |
| 0.5           | 0.0001     | 0.0001     | 0.001            | 0.01       | 0.01       |
| 0.6           | 0.0001     | 0.0001     | 0.001            | 0.01       | 0.01       |
| 0.7           | 0.0001     | 0.0001     | 0.001            | 0.01       | 0.01       |
| 0.8           | 0.0001     | 0.0001     | 0.001            | 0.01       | 0.01       |
</details>

(b) $\omega = 11.20$

![](images/de82b916bc1b61891423e9a1f35ed6aad2e8dd2f45ff5a5e1aae07e7d7cf8dfa.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.1           | 0.0001     | 0.01       | 0.01             | 0.1        | 0.1        |
| 0.2           | 0.0001     | 0.001      | 0.001            | 0.01       | 0.01       |
| 0.3           | 0.0001     | 0.001      | 0.001            | 0.01       | 0.01       |
| 0.4           | 0.0001     | 0.001      | 0.001            | 0.01       | 0.01       |
| 0.5           | 0.0001     | 0.001      | 0.001            | 0.01       | 0.01       |
| 0.6           | 0.0001     | 0.001      | 0.001            | 0.01       | 0.01       |
| 0.7           | 0.0001     | 0.001      | 0.001            | 0.01       | 0.01       |
| 0.8           | 0.0001     | 0.001      | 0.001            | 0.01       | 0.01       |
</details>

(c) $\omega = 5.6$   
Figure 24: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 25 and mushrooms dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/917707a894fa5edc909b93716d643aef702c427c3a7b22ef36df3f59d8e04652.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 100     | 100              | 100    | 100    |
| 0.005         | 0.01 | 10      | 1                | 0.1    | 0.1    |
| 0.010         | 0.001| 1       | 0.1              | 0.01   | 0.01   |
| 0.015         | 0.0001| 0.1    | 0.01             | 0.001  | 0.001  |
| 0.020         | 0.00001| 0.01  | 0.001            | 0.0001 | 0.0001 |
| 0.025         | 0.000001| 0.001 | 0.0001           | 0.00001| 0.00001|
| 0.030         | 0.0000001| 0.0001| 0.00001          | 0.000001| 0.000001|
</details>

(a) $\omega = 123.00$

![](images/aa5c8e3c34d85666d4acf8c294f7455ffb93a51481103cf27c8be5e06adbd534.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 100     | 100              | 100    | 100    |
| 0.005         | 1    | 100     | 10               | 1      | 1      |
| 0.010         | 0.1  | 100     | 1                | 0.1    | 0.1    |
| 0.015         | 0.01 | 100     | 0.1              | 0.01   | 0.01   |
| 0.020         | 0.001| 100     | 0.01             | 0.001  | 0.001  |
| 0.025         | 0.0001| 100    | 0.001            | 0.0001 | 0.0001 |
| 0.030         | 0.00001| 100   | 0.0001           | 0.00001| 0.00001|
</details>

(b) $\omega = 10.25$

![](images/c46a446c24392ba7d55ef1113e2367a3b76593e6d5ee76143b391443e3dda8d4.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 100     | 100              | 100    | 100    |
| 0.005         | 0.1  | 10      | 1                | 0.1    | 0.1    |
| 0.010         | 0.01 | 1       | 0.1              | 0.01   | 0.01   |
| 0.015         | 0.001| 0.1     | 0.01             | 0.001  | 0.001  |
| 0.020         | 0.0001| 0.01   | 0.001            | 0.0001 | 0.0001 |
</details>

(c) $\omega = 3.33$   
Figure 25: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 100 and a9a dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/158d6039d6645edfc45415ab9084bfaece7f7ac33f05b1d779ff8a0f0a9d8d97.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.2           | 10.0       | 10.0       | 1.0              | 1.0        | 10.0       |
| 0.4           | 1.0        | 1.0        | 0.1              | 0.1        | 1.0        |
| 0.6           | 0.1        | 0.1        | 0.01             | 0.01       | 1.0        |
| 0.8           | 0.01       | 0.01       | 0.001            | 0.001      | 1.0        |
</details>

(a) $\omega = 123.00$

![](images/c2216827e73bc2b984f402f900562e7355ab57e4a324bb301f248912163d77c8.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.2           | 0.01       | 10.0       | 1.0              | 1.0        | 10.0       |
| 0.4           | 0.001      | 1.0        | 0.1              | 0.1        | 1.0        |
| 0.6           | 0.0001     | 0.1        | 0.01             | 0.01       | 0.1        |
| 0.8           | 0.00001    | 0.01       | 0.001            | 0.001      | 0.01       |
</details>

(b) $\omega = 10.25$

![](images/ba7c6186640131f0949a215b24d6fdece62228f19e07f1a576a11fb81bb3abdd.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.2           | 0.1        | 10.0       | 1.0              | 1.0        | 10.0       |
| 0.4           | 0.01       | 1.0        | 0.1              | 0.1        | 1.0        |
| 0.6           | 0.001      | 0.1        | 0.01             | 0.01       | 1.0        |
| 0.8           | 0.0001     | 0.01       | 0.001            | 0.001      | 0.1        |
</details>

(c) $\omega = 3.33$   
Figure 26: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 100 and a9a dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/74d622583da4f033c5790e722209b2d587f0ebe7bd0c840a968aecac62bedf30.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.000         | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.005         | 1.0        | 10.0       | 10.0             | 1.0        | 1.0        |
| 0.010         | 0.1        | 1.0        | 1.0              | 0.1        | 0.1        |
| 0.015         | 0.01       | 0.1        | 0.1              | 0.01       | 0.01       |
| 0.020         | 0.001      | 0.01       | 0.01             | 0.001      | 0.001      |
</details>

(a) $\omega = 123.00$

![](images/f982bf29dc1c1a3aeab119e792fe8c10bb1f5a9e637dd3f8b87b66681ccac86b.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 100     | 100              | 100    | 100    |
| 0.005         | 0.01 | 10      | 10               | 0.1    | 1      |
| 0.010         | 0.001| 1       | 1                | 0.01   | 0.1    |
| 0.015         | 0.0001| 0.1    | 0.1              | 0.001  | 0.01   |
| 0.020         | 0.00001| 0.01  | 0.01             | 0.0001 | 0.001  |
</details>

(b) $\omega = 7.24$

![](images/64a311f4be43893ba463c84a659b96b299d0f6db274a2f8baff973fe342d456f.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.000         | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.005         | 0.1        | 10.0       | 10.0             | 1.0        | 1.0        |
| 0.010         | 0.01       | 1.0        | 1.0              | 0.1        | 0.1        |
| 0.015         | 0.001      | 0.1        | 0.1              | 0.01       | 0.01       |
| 0.020         | 0.0001     | 0.01       | 0.01             | 0.001      | 0.001      |
</details>

(c) $\omega = 4.10$   
Figure 27: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 50 and a9a dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/826b95b62499c46eafcca644aa97e1118ba1b50ebb5c2ecd4f3f580dd37dfb11.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1000.0     | 1000.0     | 1000.0           | 1000.0     | 1000.0     |
| 0.1           | 0.0001     | 10.0       | 10.0             | 10.0       | 10.0       |
| 0.2           | 0.0001     | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.3           | 0.0001     | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.4           | 0.0001     | 0.01       | 0.01             | 0.01       | 0.01       |
| 0.5           | 0.0001     | 0.001      | 0.001            | 0.001      | 0.001      |
| 0.6           | 0.0001     | 0.0001     | 0.0001           | 0.0001     | 0.0001     |
| 0.7           | 0.0001     | 0.00001    | 0.00001          | 0.00001    | 0.00001    |
| 0.8           | 0.0001     | 0.00001    | 0.00001          | 0.00001    | 0.00001    |
</details>

(a) $\omega = 123.00$

![](images/0429475463318a2a28d705fb1f3f4611f2e0f4370407b7e3c3c4b1cbeda3f985.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1000.0     | 1000.0     | 1000.0           | 1000.0     | 1000.0     |
| 0.1           | 10.0       | 10.0       | 1.0              | 1.0        | 10.0       |
| 0.2           | 1.0        | 1.0        | 0.1              | 0.1        | 1.0        |
| 0.3           | 0.1        | 0.1        | 0.01             | 0.01       | 0.1        |
| 0.4           | 0.01       | 0.01       | 0.001            | 0.001      | 0.01       |
| 0.5           | 0.001      | 0.001      | 0.0001           | 0.0001     | 0.001      |
| 0.6           | 0.0001     | 0.0001     | 0.00001          | 0.00001    | 0.0001     |
| 0.7           | 0.00001    | 0.00001    | 0.000001         | 0.000001   | 0.00001    |
| 0.8           | 0.000001   | 0.000001   | 0.0000001        | 0.0000001  | 0.000001   |
</details>

(b) $\omega = 7.24$

![](images/c0e631a299f38a50e79d6c46d50fd1467ce5b574205f6d2de15f3924523f72c0.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1000.0     | 1000.0     | 1000.0           | 1000.0     | 1000.0     |
| 0.1           | 10.0       | 10.0       | 10.0             | 10.0       | 10.0       |
| 0.2           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.3           | 0.1        | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.4           | 0.01       | 0.01       | 0.01             | 0.01       | 0.01       |
| 0.5           | 0.001      | 0.001      | 0.001            | 0.001      | 0.001      |
| 0.6           | 0.0001     | 0.0001     | 0.0001           | 0.0001     | 0.0001     |
| 0.7           | 0.00001    | 0.00001    | 0.00001          | 0.00001    | 0.00001    |
| 0.8           | 0.000001   | 0.000001   | 0.000001         | 0.000001   | 0.000001   |
</details>

(c) $\omega = 4.10$   
Figure 28: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 50 and a9a dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/37604c21628676b870701c8b75f73c643469007257c3f5575ef79d35bb9c9ee5.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1000 | 1000    | 1000             | 1000   | 1000   |
| 0.005         | 0.001| 10      | 10               | 0.1    | 0.1    |
| 0.010         | 0.0001| 1       | 1                | 0.01   | 0.01   |
| 0.015         | 0.00001| 0.1    | 0.1              | 0.001  | 0.001  |
| 0.020         | 0.000001| 0.01   | 0.01             | 0.0001 | 0.0001 |
</details>

(a) ω = 123.00

![](images/7ff4ab09d38bf597252e017fad656e4c39ab48b0e802e27bd41844584ca9a710.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 1000 | 1000    | 1000             | 1000   | 1000   |
| 0.005         | 0.01 | 100     | 100              | 0.1    | 0.1    |
| 0.010         | 0.001| 10      | 1                | 0.01   | 0.01   |
| 0.015         | 0.0001| 1       | 0.1              | 0.001  | 0.001  |
| 0.020         | 0.0001| 1       | 0.01             | 0.001  | 0.001  |
</details>

(b) ω = 10.25

![](images/724a3324961afab79946d63212dce378755cee5aef2f2135f23b4987a5b34b84.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.000         | 1000.0     | 1000.0     | 1000.0           | 1000.0     | 1000.0     |
| 0.005         | 1.0        | 100.0      | 100.0            | 1.0        | 1.0        |
| 0.010         | 0.01       | 10.0       | 10.0             | 0.1        | 0.1        |
| 0.015         | 0.001      | 1.0        | 1.0              | 0.01       | 0.01       |
| 0.020         | 0.0001     | 0.1        | 0.1              | 0.001      | 0.001      |
</details>

(c) ω = 4.10   
Figure 29: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 20 and a9a dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/698e2476e986a49d7c83f75f2ed2d4f58dfa63675ea539a560567b7738ce81f5.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1000.0     | 1000.0     | 1000.0           | 1000.0     | 1000.0     |
| 0.1           | 10.0       | 10.0       | 10.0             | 10.0       | 10.0       |
| 0.2           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.3           | 0.1        | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.4           | 0.01       | 0.01       | 0.01             | 0.01       | 0.01       |
| 0.5           | 0.001      | 0.001      | 0.001            | 0.001      | 0.001      |
| 0.6           | 0.0001     | 0.0001     | 0.0001           | 0.0001     | 0.0001     |
| 0.7           | 0.00001    | 0.00001    | 0.00001          | 0.00001    | 0.00001    |
| 0.8           | 0.000001   | 0.000001   | 0.000001         | 0.000001   | 0.000001   |
</details>

(a) $\omega = 123.00$

![](images/c95ffa2b9f237a3f47a8735353209f6a61a65523eb9f575f30bc165556d3bc91.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.0 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 0.1 | 1 | 10 | 10 | 1 | 10 |
| 0.2 | 0.01 | 1 | 0.1 | 0.1 | 1 |
| 0.3 | 0.001 | 0.1 | 0.01 | 0.01 | 0.1 |
| 0.4 | 0.0001 | 0.01 | 0.001 | 0.001 | 0.01 |
| 0.5 | 0.00001 | 0.001 | 0.0001 | 0.0001 | 0.001 |
| 0.6 | 0.000001 | 0.0001 | 0.00001 | 0.00001 | 0.0001 |
| 0.7 | 0.0000001 | 0.00001 | 0.000001 | 0.000001 | 0.00001 |
| 0.8 | 0.00000001 | 0.000001 | 0.0000001 | 0.0000001 | 0.000001 |
</details>

(b) ω = 10.25

![](images/ccebb6a65ee05d70ac7dcab96af4fd8f77b59e260409afe1454d092eba1fa532.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 1000.0     | 1000.0     | 1000.0           | 1000.0     | 1000.0     |
| 0.1           | 10.0       | 10.0       | 10.0             | 10.0       | 10.0       |
| 0.2           | 1.0        | 1.0        | 1.0              | 1.0        | 1.0        |
| 0.3           | 0.1        | 0.1        | 0.1              | 0.1        | 0.1        |
| 0.4           | 0.01       | 0.01       | 0.01             | 0.01       | 0.01       |
| 0.5           | 0.001      | 0.001      | 0.001            | 0.001      | 0.001      |
| 0.6           | 0.0001     | 0.0001     | 0.0001           | 0.0001     | 0.0001     |
| 0.7           | 0.00001    | 0.00001    | 0.00001          | 0.00001    | 0.00001    |
| 0.8           | 0.000001   | 0.000001   | 0.000001         | 0.000001   | 0.000001   |
</details>

(c) ω = 4.10   
Figure 30: Comparison of state-of-the-art distributed methods. The comparison is made on (5) with M = 20 and a9a dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/2552223b6381bb25d204d589afe91865c591ad79d14922eedb6f512c556d343a.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 10      | 10               | 0.1    | 0.1    |
| 0.005         | 0.001| 1       | 0.1              | 0.01   | 0.01   |
| 0.010         | 0.0001| 0.1     | 0.01             | 0.001  | 0.001  |
| 0.015         | 0.00001| 0.01   | 0.001            | 0.0001 | 0.0001 |
| 0.020         | 0.000001| 0.001 | 0.0001           | 0.00001| 0.00001|
</details>

(a) $\omega = 123.00$

![](images/15160a0303cd826313916c181c141cee6e596e72184ea51af719ca7048a1d879.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 10^2 | 10^1    | 10^1             | 10^1   | 10^1   |
| 0.005         | 10^-3| 10^0    | 10^-1            | 10^-2  | 10^-1  |
| 0.010         | 10^-4| 10^-1   | 10^-2            | 10^-2  | 10^-2  |
| 0.015         | 10^-5| 10^-2   | 10^-3            | 10^-2  | 10^-3  |
| 0.020         | 10^-5| 10^-3   | 10^-4            | 10^-3  | 10^-4  |
</details>

(b) $\omega = 7.23$

![](images/0fbdc59eee36e4b0aaf8e3847b2ae4f8eec42a2f3b5e91a99bce616c691e9d89.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 10      | 10               | 10     | 10     |
| 0.005         | 1    | 1       | 1                | 0.1    | 0.1    |
| 0.010         | 0.1  | 0.1     | 0.1              | 0.01   | 0.01   |
| 0.015         | 0.01 | 0.01    | 0.01             | 0.001  | 0.001  |
| 0.020         | 0.001| 0.001   | 0.001            | 0.0001 | 0.0001 |
</details>

(c) ω = 4.10   
Figure 31: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 70 and a9a dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/f23cb201c41a522cfa1672beb99b3179b9329212822d1d325e536859220512f4.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.1           | 0.0001     | 0.01       | 0.0001           | 0.1        | 1.0        |
| 0.2           | 0.0001     | 0.01       | 0.0001           | 0.1        | 0.1        |
| 0.3           | 0.0001     | 0.01       | 0.0001           | 0.1        | 0.1        |
| 0.4           | 0.0001     | 0.01       | 0.0001           | 0.1        | 0.1        |
| 0.5           | 0.0001     | 0.01       | 0.0001           | 0.1        | 0.1        |
</details>

(a) ω = 123.00

![](images/2362d0bc405dd693c3a217acec7349767f4fff67afc2ddd32278a654ffa92eea.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 10^1       | 10^1       | 10^1             | 10^1       | 10^1       |
| 0.1           | 10^-3      | 10^-2      | 10^-4            | 10^-2      | 10^-1      |
| 0.2           | 10^-4      | 10^-2      | 10^-4            | 10^-2      | 10^-1      |
| 0.3           | 10^-4      | 10^-2      | 10^-4            | 10^-2      | 10^-1      |
| 0.4           | 10^-4      | 10^-2      | 10^-4            | 10^-2      | 10^-1      |
| 0.5           | 10^-4      | 10^-2      | 10^-4            | 10^-2      | 10^-1      |
</details>

(b) $\omega = 7.23$

![](images/eb1da69f7e6d019d682d3a3af00b6cba7f06d309d38c7ba30ee672c59def89c9.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 10^1       | 10^1       | 10^1             | 10^1       | 10^1       |
| 0.1           | 10^-3      | 10^-1      | 10^-4            | 10^-1      | 10^-1      |
| 0.2           | 10^-5      | 10^-2      | 10^-5            | 10^-2      | 10^-1      |
| 0.3           | 10^-5      | 10^-2      | 10^-5            | 10^-2      | 10^-1      |
| 0.4           | 10^-5      | 10^-2      | 10^-5            | 10^-2      | 10^-1      |
| 0.5           | 10^-5      | 10^-2      | 10^-5            | 10^-2      | 10^-1      |
</details>

(c) ω = 4.10   
Figure 32: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 70 and a9a dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/d129977c682d14d95cae462bb11d4c305fe4b700d35aaf3abbeffffbd89b8f32.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 10      | 10               | 10     | 10     |
| 0.005         | 0.0001 | 10      | 0.1              | 0.1    | 0.1    |
| 0.010         | 0.0001 | 10      | 0.01             | 0.01   | 0.01   |
| 0.015         | 0.0001 | 10      | 0.001            | 0.001  | 0.001  |
| 0.020         | 0.0001 | 10      | 0.0001           | 0.0001 | 0.0001 |
</details>

(a) ω = 123.00

![](images/4a8e1f4d83b5c697b7c009030f089d5ceead0f6b8c8dd042fd9de502b6b35e7f.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.000         | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.005         | 0.1        | 100.0      | 0.1              | 0.1        | 0.1        |
| 0.010         | 0.01       | 100.0      | 0.01             | 0.01       | 0.01       |
| 0.015         | 0.001      | 100.0      | 0.001            | 0.001      | 0.001      |
| 0.020         | 0.0001     | 100.0      | 0.0001           | 0.0001     | 0.0001     |
| 0.025         | 0.00001    | 100.0      | 0.00001          | 0.00001    | 0.00001    |
| 0.030         | 0.000001   | 100.0      | 0.000001         | 0.000001   | 0.000001   |
</details>

(b) ω = 7.23

![](images/965e6689b2709e3afb8a9943a7df8a5245e4a73220ef0a493c057b3465ef3cb4.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 10      | 10               | 1      | 1      |
| 0.005         | 0.01 | 10      | 0.01             | 0.1    | 0.1    |
| 0.010         | 0.001| 10      | 0.001            | 0.01   | 0.01   |
| 0.015         | 0.0001| 10    | 0.0001           | 0.001  | 0.001  |
| 0.020         | 0.00001| 10   | 0.00001          | 0.0001 | 0.0001 |
| 0.025         | 0.000001| 10   | 0.000001         | 0.00001| 0.00001|
| 0.030         | 0.0000001| 10   | 0.0000001        | 0.000001| 0.000001|
</details>

(c) ω = 4.10   
Figure 33: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 50 and a9a dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/f92aa92687a269ab527e635bd26a366e24ac53754ccc52610aadaf32ca3dbbe3.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.1           | 0.0001     | 0.1        | 0.0001           | 0.01       | 0.1        |
| 0.2           | 0.0001     | 0.01       | 0.00001          | 0.01       | 0.1        |
| 0.3           | 0.0001     | 0.001      | 0.00001          | 0.01       | 0.1        |
| 0.4           | 0.0001     | 0.001      | 0.00001          | 0.01       | 0.1        |
| 0.5           | 0.0001     | 0.001      | 0.00001          | 0.01       | 0.1        |
</details>

(a) ω = 123.00

![](images/dc2ef6eac6ccf3721b9116199b91bb42c814d38c09acc654bf13527acb5858ba.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.1           | 0.0001     | 1.0        | 0.0001           | 0.1        | 0.1        |
| 0.2           | 0.00001    | 0.1        | 0.00001          | 0.1        | 0.1        |
| 0.3           | 0.00001    | 0.01       | 0.00001          | 0.1        | 0.1        |
| 0.4           | 0.00001    | 0.001      | 0.00001          | 0.1        | 0.1        |
| 0.5           | 0.00001    | 0.0001     | 0.00001          | 0.1        | 0.1        |
</details>

(b) ω = 7.23

![](images/f7ce50c12feb74f809053eacd2672b70dacba55b2cca551b0a3117bba12e0686.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.0           | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.1           | 0.01       | 1.0        | 0.0001           | 0.01       | 0.1        |
| 0.2           | 0.0001     | 0.1        | 0.00001          | 0.01       | 0.1        |
| 0.3           | 0.0001     | 0.01       | 0.00001          | 0.01       | 0.1        |
| 0.4           | 0.0001     | 0.01       | 0.00001          | 0.01       | 0.1        |
| 0.5           | 0.0001     | 0.01       | 0.00001          | 0.01       | 0.1        |
</details>

(c) ω = 4.10   
Figure 34: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 50 and a9a dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .

![](images/747fd35fde8a9ba377faeee98d03cb4ef57901ce9519b889def683b83730a1f6.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 100     | 100              | 100    | 100    |
| 0.005         | 0.01 | 1       | 0.1              | 0.1    | 0.1    |
| 0.010         | 0.001| 0.1     | 0.01             | 0.01   | 0.01   |
| 0.015         | 0.0001| 0.01   | 0.001            | 0.001  | 0.001  |
| 0.020         | 0.00001| 0.001 | 0.0001           | 0.0001 | 0.0001 |
</details>

(a) $\omega = 123.00$

![](images/6fe6408d94bab93be468a6435ec42caefe3b9a53e6854ed165a086cd590fb581.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 100     | 100              | 100    | 100    |
| 0.005         | 0.01 | 1       | 0.1              | 0.1    | 0.1    |
| 0.010         | 0.001| 0.1     | 0.01             | 0.01   | 0.01   |
| 0.015         | 0.0001| 0.01    | 0.001            | 0.001  | 0.001  |
| 0.020         | 0.00001| 0.001  | 0.0001           | 0.0001 | 0.0001 |
</details>

(b) $\omega = 7.23$

![](images/d4302d3e27b48e9c73b553f5aa51c4e5eabe2e8e4b3f688ddf39105d3ab3dac2.jpg)

<details>
<summary>line</summary>

| Training time | OLGA | AccSVRS | AccExtraGradient | ADIANA | LoCoDL |
| ------------- | ---- | ------- | ---------------- | ------ | ------ |
| 0.000         | 100  | 100     | 100              | 100    | 100    |
| 0.005         | 0.001| 1       | 0.1              | 0.1    | 0.01   |
| 0.010         | 0.0001| 0.1     | 0.01             | 0.01   | 0.001  |
| 0.015         | 0.0001| 0.01    | 0.001            | 0.001  | 0.0001 |
| 0.020         | 0.0001| 0.001   | 0.0001           | 0.0001 | 0.0001 |
</details>

(c) $\omega = 4.10$   
Figure 35: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 20 and a9a dataset. The criterion is the training time on local cluster (fast connection). For methods with compression we vary the power of compression $\omega$ .

![](images/9c9a458fa396b6ede2d575d07e72197977e9b0ba58516498f6699496e283ae49.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.00          | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.05          | 0.1        | 0.01       | 0.0001           | 0.1        | 0.1        |
| 0.10          | 0.01       | 0.001      | 0.0001           | 0.01       | 0.01       |
| 0.15          | 0.01       | 0.0001     | 0.0001           | 0.01       | 0.01       |
| 0.20          | 0.01       | 0.00001    | 0.0001           | 0.01       | 0.01       |
</details>

(a) $\omega = 123.00$

![](images/11ff32958bcaa93700d964adadb5cb94d2fa369a5953be0e9d36e8df110e5888.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.00          | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.05          | 0.0001     | 0.001      | 0.0001           | 0.1        | 0.1        |
| 0.10          | 0.01       | 0.0001     | 0.01             | 0.1        | 0.1        |
| 0.15          | 0.01       | 0.00001    | 0.01             | 0.1        | 0.1        |
| 0.20          | 0.01       | 0.000001   | 0.01             | 0.1        | 0.1        |
</details>

(b) $\omega = 7.23$

![](images/8af427d932f3a80b781ac26c8c990423240f374201579c9d13cd1c0ceac95cb0.jpg)

<details>
<summary>line</summary>

| Training time | OLGA       | AccSVRS    | AccExtraGradient | ADIANA     | LoCoDL     |
| ------------- | ---------- | ---------- | ---------------- | ---------- | ---------- |
| 0.00          | 100.0      | 100.0      | 100.0            | 100.0      | 100.0      |
| 0.05          | 0.001      | 0.01       | 0.01             | 0.1        | 0.1        |
| 0.10          | 0.0001     | 0.001      | 0.01             | 0.1        | 0.1        |
| 0.15          | 0.0001     | 0.0001     | 0.01             | 0.1        | 0.1        |
| 0.20          | 0.0001     | 0.0001     | 0.01             | 0.1        | 0.1        |
</details>

(c) $\omega = 4.10$   
Figure 36: Comparison of state-of-the-art distributed methods. The comparison is made on (6) with M = 20 and a9a dataset. The criterion is the training time on remote CPUs (slow connection). For methods with compression we vary the power of compression $\omega$ .