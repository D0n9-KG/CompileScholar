# Advancing Bayesian Optimization via Learning Correlated Latent Space

Seunghun Lee, $^{*}$ Jaewon Chu, $^{*}$ Sihyeon Kim, $^{*}$ Juyeon Ko, Hyunwoo J. Kim $^{\dagger}$

Computer Science & Engineering

Korea University

{llsshh319, allonsy07, sh\_bs15, juyon98, hyunwoojkim}@korea.ac.kr

# Abstract

Bayesian optimization is a powerful method for optimizing black-box functions with limited function evaluations. Recent works have shown that optimization in a latent space through deep generative models such as variational autoencoders leads to effective and efficient Bayesian optimization for structured or discrete data. However, as the optimization does not take place in the input space, it leads to an inherent gap that results in potentially suboptimal solutions. To alleviate the discrepancy, we propose Correlated latent space Bayesian Optimization (CoBO), which focuses on learning correlated latent spaces characterized by a strong correlation between the distances in the latent space and the distances within the objective function. Specifically, our method introduces Lipschitz regularization, loss weighting, and trust region recoordination to minimize the inherent gap around the promising areas. We demonstrate the effectiveness of our approach on several optimization tasks in discrete data, such as molecule design and arithmetic expression fitting, and achieve high performance within a small budget.

# 1 Introduction

Bayesian optimization (BO) is a standard method for a wide range of science and engineering problems such as chemical design $[1-4]$ , reinforcement learning $[5]$ , and hyperparameter tuning $[6]$ . Relying on a surrogate model typically modeled with a Gaussian process (GP), BO estimates the computationally expensive black-box objective function to solve the problem with a minimum number of function evaluations $[7]$ . While it is known as a powerful method on continuous domains $[8, 9]$ , applying BO is often obstructed by structured or discrete data, as the objective values are in a complex combinatorial space $[10]$ . This challenge has motivated recent interest in latent space Bayesian optimization (LBO) methods $[11-14]$ , which aim to find solutions in low-dimensional, continuous embeddings of the input data. By adopting deep generative models such as variational autoencoders (VAEs) $[15]$ to map the input space to the latent space, LBO has successfully addressed the difficulties of such optimization problems.

However, the fact that optimization is not directly conducted in the input space gives rise to inherent gaps, which may lead to failures in the optimization process. First, we can think of the gap between the input space and the latent space. Seminal works have been developed to address this gap, with generative models such as $\beta$ -VAE [16] emphasizing the importance of controlling loss weights, while WGAN [17] introduces improved regularization. Both aim to learn the improved latent space that better aligns with the input data distribution. Second, considering the BO problems, an additional gap emerges between the proximity of solutions in the latent space and the similarity of their

black-box objective function values. This is observed in many prior works $[18, 19]$ that learn a latent space by minimizing only reconstruction errors without considering the surrogate model. This often leads to suboptimal optimization results. A recent study $[10]$ has highlighted the significance of joint training between VAE and the surrogate model, yet it only implicitly encourages the latent space to align with the surrogate model. This limitation is observed in Figure 5a, where the latent space's landscape, with respect to objective values, remains highly non-smooth with joint training.

To this end, we propose our method Correlated latent space Bayesian Optimization (CoBO) to address the inherent gaps in LBO. First, we aim to minimize the gap between the latent space and the objective function by increasing the correlation between the distance of latent vectors and the differences in their objective values. By calculating the lower bound of the correlation, we introduce two regularizations and demonstrate their effectiveness in enhancing the correlation. Especially, one of these regularizations, called the Lipschitz regularization, encourages a smoother latent space, allowing for a more efficient optimization process (see Figure 5b). Moreover, we suggest loss weighting with respect to objective values of each input data point to particularly minimize the gap between the input space and the latent space around promising areas of high objective values. Finally, we propose the concept of trust region recoordination to adjust the search space in line with the updated latent space. We experimentally validate our method with qualitative and quantitative analyses on nine tasks using three benchmark datasets on molecule design and arithmetic expression fitting.

To summarize, our contributions are as follows:

- We propose Correlated latent space Bayesian Optimization (CoBO) to bridge the inherent gap in latent Bayesian optimization.   
- We introduce two regularizations to align the latent space with the black-box objective function based on increasing the lower bound of the correlation between the Euclidean distance of latent vectors and the distance of their corresponding objective values.   
- We present a loss weighting scheme based on the objective values of input points, aiming to close the gap between the input space and the latent space focused on promising areas.   
- We demonstrate extensive experimental results and analyses on nine tasks using three benchmark datasets on molecule design and arithmetic expression fitting and achieve state-of-the-art in all nine tasks.

# 2 Methods

In this section, we describe the main contributions of our method. Section 2.1 introduces several preliminaries on Bayesian optimization. In Section 2.2, we propose two regularizations to align the latent space with the black-box objective function. In Section 2.3, we describe our loss weighting scheme with the objective values. Lastly, in Section 2.4, we explain the overall architecture of our method.

# 2.1 Preliminaries

Bayesian optimization. Bayesian optimization (BO) [6, 8, 9] is a classical, sample-efficient optimization method that aims to solve the problem

$$
\mathbf {x} ^ {*} = \underset {\mathbf {x} \in \mathcal {X}} {\arg \max} f (\mathbf {x}), \tag {1}
$$

where X is a feasible set and f is a black-box objective function. Since the function evaluation is assumed to be expensive, BO constructs a probabilistic model of the black-box objective function. There are two main components of BO, first is a surrogate model g that provides posterior probability distribution over $f(\mathbf{x})$ conditioned on observed dataset $\mathcal{D} = \{(\mathbf{x}_{i}, y_{i})\}_{i=1}^{n}$ based prior over objective function. Second is an acquisition function $\alpha$ for deciding the most promising next query point $x_{i+1}$ based on the posterior distribution over $f(\mathbf{x})$ . BO is a well-established method, however, applying BO to high-dimensional data can be challenging due to the exponential growth of the search space. To alleviate the challenge, recent approaches [10, 20] restrict the search space to a hyper-rectangular trust region centered on the current optimal input data point. In this paper, we adopt this trust-region-based BO for handling high-dimensional search space.

Latent space Bayesian optimization. BO over structured or discrete input space X is particularly challenging, as a search space over the objective function becomes a large combinatorial one. In an effort to reduce a large combinatorial search space to continuous space, latent space Bayesian optimization (LBO) [10, 13, 14, 19, 21, 18] suggests BO over continuous latent space Z. A pretrained VAE = $\{q_{\phi}, p_{\theta}\}$ [15] is commonly used as the mapping function, where the latent space is learned to follow the prior distribution (Gaussian distribution). Given a pretrained VAE, an encoder $q_{\phi}: X \mapsto Z$ maps the input $x_{i}$ to the latent vector $z_{i}$ and the surrogate model g takes $z_{i}$ as the input. After the acquisition function $\alpha$ suggests the next latent query point $z_{i+1}$ , a decoder $p_{\theta}: Z \mapsto X$ reconstructs $z_{i+1}$ to $x_{i+1}$ , so that it can be evaluated by the black-box objective function, i.e., $f(\mathbf{x}_{i+1})$ .

LBO is a promising optimization approach for discrete or structured inputs, yet, there are two main gaps to be considered. Firstly, there is a gap between the input space and the latent space. Our focus is on narrowing the gap, especially within the promising areas, i.e., samples with high objective function values. Secondly, a gap exists between the proximity of solutions within the latent space and the similarity of their corresponding objective function values. This arises because the objective value originates from a black-box function in the discrete input space X, distinct from the latent space Z where our surrogate model g is learned. In the previous work, [10] has suggested closing the gap by jointly optimizing VAE and the surrogate GP model to align the latent space with current top-k samples.

Here, we propose CoBO that explicitly addresses those two gaps by training a latent space with Lipschitz regularization, which increases the correlation between the distance of latent samples and the distance of objective values, and loss weighting with objective values to focus on relatively important search space.

# 2.2 Aligning the latent space with the objective function

Our primary goal is to align the latent space $\mathcal{Z}$ with the black-box objective function $f$ , which can be achieved by increasing the correlation between the Euclidean distance of latent vectors and the differences in their corresponding objective values, i.e., $\mathrm{Corr}(||\mathbf{z}_1 - \mathbf{z}_2||_2, |y_1 - y_2|)$ . Assuming the objective function $f$ is an $L$ -Lipschitz continuous function, we can establish a lower bound of $\mathrm{Corr}(||\mathbf{z}_1 - \mathbf{z}_2||_2, |y_1 - y_2|)$ . In general, if the function $f$ is $L$ -Lipschitz continuous, it is defined as

$$
\forall \mathbf {z} _ {1}, \mathbf {z} _ {2} \in \mathbb {R} ^ {n}, \quad d _ {Y} (f (\mathbf {z} _ {1}), f (\mathbf {z} _ {2})) \leq L d _ {Z} (\mathbf {z} _ {1}, \mathbf {z} _ {2}), \tag {2}
$$

where $d_{Z}$ and $d_{Y}$ are distance metrics in their respective spaces for z and y. Then, the lower bound of $\operatorname{Corr}(|\mathbf{z}_{1}-\mathbf{z}_{2}||_{2},|y_{1}-y_{2}|)$ can be obtained as Theorem 1.

Theorem 1. Let $D_{Z} = d_{Z}(Z_{1}, Z_{2})$ and $D_{Y} = d_{Y}(f(Z_{1}), f(Z_{2}))$ be random variables where $Z_{1}, Z_{2}$ are i.i.d. random variables, f is an L-Lipschitz continuous function, and $d_{Z}, d_{Y}$ are distance functions. Then, the correlation between $D_{Z}$ and $D_{Y}$ is lower bounded as

$$
D _ {Y} \leq L D _ {Z} \Rightarrow C o r r _ {D _ {Z}, D _ {Y}} \geq \frac {\frac {1}{L} (\sigma_ {D _ {Y}} ^ {2} + \mu_ {D _ {Y}} ^ {2}) - L \mu_ {D _ {Z}} ^ {2}}{\sqrt {\sigma_ {D _ {Z}} ^ {2} \sigma_ {D _ {Y}} ^ {2}}},
$$

where $\mu_{D_{Z}}$ , $\sigma_{D_{Z}}^{2}$ , $\mu_{D_{Y}}$ , and $\sigma_{D_{Y}}^{2}$ are the mean and variance of $D_{Z}$ and $D_{Y}$ respectively.

Theorem 1 implies that under the assumption of the $L$ -Lipschitz continuity of $f$ , we can increase the lower bound of $\mathrm{Corr}(|\mathbf{z}_1 - \mathbf{z}_2||_2, |y_1 - y_2|)$ by reducing Lipschitz constant $L$ while $\mu_{D_Z}^2$ , $\sigma_{D_Z}^2$ , $\mu_{D_Y}^2$ , and $\sigma_{D_Y}^2$ remain as constants. Based on Theorem 1, we propose two regularizations. The first one is Lipschitz regularization, which encourages a smooth latent space $\mathcal{Z}$ w.r.t. the objective function $f$ .

$$
\mathcal {L} _ {\text { Lip }} = \sum_ {i, j \leq N} \max \left(0, \frac {| y _ {i} - y _ {j} |}{| | \mathbf {z} _ {i} - \mathbf {z} _ {j} | | _ {2}} - L\right), \tag {3}
$$

where N is the number of training data points. Here, we set the Lipschitz constant L as the median of all possible gradients of slopes. By penalizing slopes with larger gradients than an adaptively adjusted L, we encourage a decrease in L itself, leading to learning a correlated latent space.

Next, it is beneficial to keep $\mu_{D_{Z}}^{2}$ , $\sigma_{D_{Z}}^{2}$ , $\mu_{D_{Y}}^{2}$ and $\sigma_{D_{Y}}^{2}$ as constants. Given that $\mu_{D_{Y}}^{2}$ and $\sigma_{D_{Y}}^{2}$ are function-specific constants, where the black-box function is unchanged throughout the optimization,

we can treat them as fixed values. Then, we want to constrain $\mu_{D_Z}^2$ as a constant with the second regularization $\mathcal{L}_{\mathrm{z}}$ , by penalizing the average distance between the latent vectors $\mathbf{z}$ to be a constant $c$ :

$$
\mathcal {L} _ {\mathrm{z}} = \left| \left(\frac {1}{N ^ {2}} \sum_ {i, j \leq N} | | \mathbf {z} _ {i} - \mathbf {z} _ {j} | | _ {2}\right) - c \right|. \tag {4}
$$

We set $c$ as the expected Euclidean norm between two standard normal distributions, which is the prior of the variational autoencoder. That is the mean of the noncentral chi distribution [22] which is sum of squared independent normal random variables:

$$
c = \mathbb {E} \left[ \sqrt {\Sigma_ {i} ^ {n} (U _ {i} - V _ {i}) ^ {2}} \right] = \mathbb {E} [ C ] = \frac {2 \Gamma (\frac {k + 1}{2})}{\Gamma (\frac {k}{2})}, U _ {i}, V _ {i} \sim \mathcal {N} (0, 1), C \sim N C _ {\chi_ {k}}, \tag {5}
$$

$$
\sqrt {\Sigma_ {i} ^ {n} (U _ {i} - V _ {i}) ^ {2}} = \sqrt {\Sigma_ {i} ^ {n} W _ {i} ^ {2}} = C, W _ {i} \sim \mathcal {N} \left(0, \sqrt {2} ^ {2}\right), \tag {6}
$$

where $\Gamma(\cdot)$ denotes the gamma function, C denotes the random variable with noncentral chi distribution $NC_{\chi_{k}}$ and k denotes the degrees of freedom which is the same value as dimension n of the latent vector. Then c is dependent only on the dimension of the latent vector, $z \in R^{n}$ . For $\sigma_{D_{z}}^{2}$ , preliminary observations indicate that it stays in a reasonable range as long as $L_{Lip}$ is not overly penalized. Thus, we safely conclude that there is no need to explicitly constrain $\sigma_{D_{z}}^{2}$ . We refer to the supplement for further analysis and the proof of Theorem 1.

# 2.3 Loss weighting with objective values

Our focus now shifts to addressing the gap between the input space X and the latent space Z for LBO. Especially, we aim to minimize the gap in promising areas that offer better optimization opportunities, i.e., significant points with high objective values. To achieve this, we prioritize input data points based on their respective objective values by weighting the reconstruction loss term. Following [23], we utilize the cumulative density function of the Gaussian distribution for the weighting. Specifically, the weighting function w.r.t. objective value y is:

$$
\lambda (y) = P (Y > y _ {q}), \tag {7}
$$

with $Y \sim \mathcal{N}(y, \sigma^{2})$ , where $y_{q}$ represents a specific quantile of the distribution of Y, and hyperparameter $\sigma$ denotes the standard deviation of Y. The weighted reconstruction loss is as follows:

$$
\mathcal {L} _ {\text { recon\_W }} = \lambda (y) \mathcal {L} _ {\text { recon }} = - \lambda (y) \mathbb {E} _ {\mathbf {z} \sim q _ {\phi} (\mathbf {z} | \mathbf {x})} [ \log p _ {\theta} (\mathbf {x} | \mathbf {z}) ]. \tag {8}
$$

Moreover, we also apply the weighting scheme to the Lipschitz regularization term to promote a smoother latent space when the objective value is higher. The weighted Lipschitz regularization is defined with the geometric mean of the weights of two input data points:

$$
\mathcal {L} _ {\text { Lip\_W }} = \sum_ {i, j \leq N} \sqrt {\lambda (y _ {i}) \lambda (y _ {j})} \max \left(0, \frac {| y _ {i} - y _ {j} |}{| | \mathbf {z} _ {i} - \mathbf {z} _ {j} | | _ {2}} - L\right). \tag {9}
$$

# 2.4 Overall architecture of CoBO

In this section, we explain the overall architecture of CoBO. We first introduce the training schema of latent space in CoBO to encourage a high correlation between the distance in the latent space and the distance within the objective function. Next, we describe updating strategy of the surrogate model for modeling the black-box function and further present a generating procedure of the next candidate inputs for the black-box objective function through the acquisition function in the trust region. The overall procedure of our CoBO is in Algorithm 1.

Learning the latent space. Our method learns the latent space by optimizing the encoder $q_{\phi}$ and decoder $p_{\theta}$ of the pretrained VAE and updating the surrogate model in the latent space with our final loss:

$$
\mathcal {L} _ {\mathrm{CoBO}} = \mathcal {L} _ {\text { Lip\_W }} + \mathcal {L} _ {\mathrm{z}} + \mathcal {L} _ {\text { recon\_W }} + \mathcal {L} _ {\mathrm{KL}} + \mathcal {L} _ {\text { surr }}, \tag {10}
$$

$$
\mathcal {L} _ {\mathrm{KL}} = \mathrm{KL} (q _ {\phi} (\mathbf {z} | \mathbf {x}) | | p _ {\theta} (\mathbf {z})), \tag {11}
$$

where $L_{Lip\_W}$ and $L_{z}$ is the regularization term in the latent space at Section 2.2 and 2.3, $L_{recon\_W}$ is the weighted reconstruction loss term, $L_{KL}$ is the KL divergence between the latent space distribution and the prior, and $L_{surr}$ is the loss for optimizing the surrogate model. We adopt the joint training scheme, training the surrogate model and the encoder $q_{\phi}$ of the VAE model jointly [10]. Under computational considerations, we retrain a latent space after $N_{fail}$ accumulated failure of updating the optimal objective value.

Updating the surrogate model. After jointly optimizing the latent space and the surrogate model, we freeze the parameter of VAE and train our surrogate model. Note that this update is executed only after consecutive failures of updating optimal value, we also train the surrogate model in every iteration. As exact Gaussian process (GP), i.e., $f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$ , where m is a mean function and k is a covariance function, is infeasible to handle large datasets due to cubic computational complexity $O(N^{3})$ for N data points, we employ sparse GP [24] as a surrogate model which is computationally efficient via inducing point method. To alleviate cubic complexity, sparse GP approximates black-box function with $M \ll N$ pseudo-training samples called ‘inducing points’ that reduce complexity to $O(MN^{2})$ . We select the most widely used RBF kernel as a sparse GP kernel function. Finally, we adopted deep kernel learning (DKL) [25] in conjunction with sparse GP for our final surrogate model.

Generating candidates through acquisition function. Candidate samples for the acquisition function are determined by random points in a trust region centered on the current optimal value. We take a simple and powerful method, Thompson sampling as an acquisition function within the context of Bayesian optimization. The surrogate model acts as the prior

![](images/d4326df08c7271bee373df4f008871f40043d3c76473cfa9d4eb0e94be321b20.jpg)

<details>
<summary>text_image</summary>

X*
qφ
pθ
Latent space Z
Trust region
Z*
Ẑ*
</details>

Figure 1: Trust region recoordination.

belief about our objective function. Thompson sampling uses this model to draw samples, leveraging its uncertainty to decide the next point to evaluate. In detail, we first select candidate samples in the trust region and score each candidate based on the posterior of the surrogate model to get the most promising values. Also, we recoordinate the center of the trust region to $\hat{z}^{*}$ , which is obtained by passing the current optimal latent vector $z^{*}$ into updated VAE, i.e., $q_{\phi}(p_{\theta}(\mathbf{z}^{*}))$ , as shown in Figure 1. We empirically showed that trust region recoordination helps to find a better solution within a limited budget (see Table 2). Following [20], the base side length of the trust region is halved after consecutive failures of updating optimal objective value and doubled after consecutive successes.

# 3 Experiments

In this section, we demonstrate the effectiveness and efficiency of CoBO through various optimization benchmark tasks. We first introduce tasks and baselines. Then, in Section 3.1, we present evaluations of our method and the baselines for each task. In Section 3.2, we conduct an ablation study on the components of our approach. Finally, in Section 3.3, we provide qualitative analyses on the effects of our suggested regularizations and the necessity of z regularization.

Tasks. We evaluate CoBO to nine tasks on a discrete space in three different Bayesian optimization benchmarks, which consist of arithmetic expression fitting tasks, molecule design tasks named dopamine receptor D3 (DRD3) in Therapeutics Data Commons (TDC) [26] and Guacamol benchmarks [27]. The arithmetic expression task is generating polynomial expressions that are close to specific target expressions (e.g., $1/3 + x + \sin(x \times x)$ ) [10, 11, 13, 14, 28], we set the number of initialization points $|D_{0}|$ to 40k, and max oracle calls to 500. In Guacamol benchmark, we select seven challenging tasks to achieve high objective value, Median molecules 2, Zaleplon MPO, Perindopril MPO, Osimertinib MPO, Ranolazine MPO, Aripiprazole similarity, and Valsartan SMART. The results for the last three tasks are in the supplement. The goal of each task is to find molecules that have the most required properties. For every task of Guacamol benchmark, we set the number of initialization points to 10k, and max oracle calls to 70k. DRD3 task in the TDC benchmark aims to find molecules with the largest docking scores to a target protein. In DRD3, the number of initialization points is set to 100, and the number of oracle calls is set to 3k. We use SELFIES VAE [10] in Chemical design, and Grammar VAE [11] in arithmetic expression.

Algorithm 1 Correlated Bayesian Optimization (CoBO)   
Input: Pretrained VAE encoder $q_{\phi}$ , decoder $p_{\theta}$ , black-box function f, surrogate model g, acquisition function $\alpha$ , previously evaluated dataset $D_{0} = \{(\mathbf{x}_{i}, y_{i}, \mathbf{z}_{i})\}_{i=1}^{n}$ , oracle budget T, latent update interval $N_{fail}$ , batch size $N_{b}$ , loss for surrogate model $L_{surr}$ , proposed loss for joint training $L_{CoBO}$ 1: $D \leftarrow D_{0}$ 2: $n_{fail} \leftarrow 0$ 3: for t = 1, 2, ..., T do

4: $D' \leftarrow D[-N_{b} :] \cup \text{top-k}(D)$ 5: if $n_{fail} \geq N_{fail}$ then

6: $n_{fail} \leftarrow 0$ 7: Train $q_{\phi}, p_{\theta}, g$ with $L_{CoBO}, D'$ $\triangleright Eq. 10$ 8: $Z \leftarrow \{q_{\phi}(\mathbf{x}_{i}) | (\mathbf{x}_{i}, y_{i}, \mathbf{z}_{i}) \in D'\}$ 9: $D \leftarrow D \cup \{(p_{\theta}(\mathbf{z}_{i}), f(p_{\theta}(\mathbf{z}_{i})), \mathbf{z}_{i}) | \mathbf{z}_{i} \in Z\}$ 10: end if

11: Train g with $L_{surr}, D'$ if $t \neq 1$ else $D_{0}$ 12: $(\mathbf{x}^{*}, y^{*}, \mathbf{z}^{*}) = \arg\max_{(\mathbf{x}, y, \mathbf{z}) \in D} y$ 13: $\hat{\mathbf{z}}^{*} \leftarrow q_{\phi}(p_{\theta}(\mathbf{z}^{*}))$ $\triangleright$ trust region recoordination

14: Get a candidate set $Z_{cand}$ with random points in the trust region around $\hat{z}^{*}$ 15: $z_{next} \leftarrow \arg\max_{z \in Z_{cand}} \alpha(z)$ 16: if $f(p_{\theta}(z_{next})) \leq y^{*}$ then $n_{fail} \leftarrow n_{fail} + 1$ 17: $D \leftarrow D \cup \{(p_{\theta}(z_{next}), f(p_{\theta}(z_{next})), z_{next})\}$ 18: end for

19: return $x^{*}$

Baselines. We compare our methods with four BO baselines: LOL-BO [10], W-LBO [13], TuRBO [20] and LS-BO. LOL-BO proposed the joint loss to close the gap between the latent space and the discrete input space. Additionally, W-LBO weighted data points to emphasize important samples with regard to their objective values. To handle discrete and structured data, we leverage TuRBO in latent space, TuRBO-L through encoder $q_{\phi}$ of pretrained frozen VAE and reconstruct it by decoder $p_{\theta}$ to be evaluated by the objective function. For more details about TuRBO-L, see [10]. In our LS-BO approach, we adopt the standard LBO methodology. Specifically, we use a VAE with pre-trained parameters that are frozen. For sampling, the candidates are sampled from a standard normal Gaussian distribution, $\mathcal{N}(0, I)$ .

# 3.1 Results on diverse benchmark tasks

Figure 2, 3 represent the graphs that depict the number of oracle calls, i.e., the number of the black-box function evaluations, and the corresponding mean and standard deviation of objective value. The maximum number of oracles is set to 70k with Gucamol benchmarks, 3k with the DRD3 benchmark, and 500 with the arithmetic expression task. As shown in Figure 2, our method outperformed all the baselines in all the tasks. Note that the goal of the arithmetic task in Figure 3a and the DRD3 task Figure 3b is minimizing the objective function, while others aim to maximize it. In the case of Figure 2a and Figure 2d, LS-BO and TuRBO-L markedly failed to search for the one that has the best score. Especially in Figure 2a, LS-BO and TuRBO-L, which only search with the fixed latent space, failed to update the initial best point despite searching 70k molecules. It implies that the pretrained VAE cannot generate better molecules unless the latent space is updated.

# 3.2 Ablation study

Table 1: An ablation of CoBO's components. Table 2: An ablation on trust region recoordination. 

<table><tr><td> $\mathcal{L}_{z}$ </td><td> $\lambda(y)$ </td><td> $\mathcal{L}_{\text{Lip}}$ </td><td>Score</td></tr><tr><td>√</td><td>√</td><td>√</td><td>0.7921</td></tr><tr><td>×</td><td>√</td><td>√</td><td>0.7835</td></tr><tr><td>×</td><td>×</td><td>√</td><td>0.7733</td></tr><tr><td>×</td><td>×</td><td>×</td><td>0.7504</td></tr></table>

<table><tr><td>Recoordination</td><td>Score</td></tr><tr><td> $\checkmark$ </td><td>0.7921</td></tr><tr><td> $\times$ </td><td>0.6844</td></tr></table>

![](images/6322a2b9418f9795d604dbe6ebcc306d62cb55b2f04a606cc470a1fc6d2161c7.jpg)

<details>
<summary>line</summary>

| Num Oracle | Score (Blue) | Score (Orange) | Score (Green) | Score (Red) |
| ---------- | ------------ | -------------- | ------------- | ----------- |
| 10000      | 0.31         | 0.31           | 0.31          | 0.31        |
| 20000      | 0.32         | 0.32           | 0.32          | 0.31        |
| 30000      | 0.33         | 0.33           | 0.33          | 0.31        |
| 40000      | 0.36         | 0.34           | 0.34          | 0.31        |
| 50000      | 0.37         | 0.34           | 0.34          | 0.31        |
| 60000      | 0.38         | 0.35           | 0.35          | 0.31        |
| 70000      | 0.38         | 0.35           | 0.35          | 0.31        |
| 80000      | 0.38         | 0.35           | 0.35          | 0.31        |
</details>

(a) Median molecules 2 (med2)

![](images/89ac1044ea729adbe08cc7da2210ddac61b6db28d27a52bb087c3670157760bd.jpg)

<details>
<summary>line</summary>

| Num Oracle | Blue Score | Orange Score | Green Score | Red Score | Purple Score |
| ---------- | ---------- | ------------ | ----------- | --------- | ------------ |
| 10000      | 0.75       | 0.70         | 0.65        | 0.55      | 0.50         |
| 20000      | 0.76       | 0.72         | 0.68        | 0.58      | 0.50         |
| 30000      | 0.77       | 0.73         | 0.69        | 0.59      | 0.50         |
| 40000      | 0.77       | 0.74         | 0.69        | 0.60      | 0.50         |
| 50000      | 0.77       | 0.74         | 0.69        | 0.61      | 0.50         |
| 60000      | 0.77       | 0.74         | 0.69        | 0.61      | 0.51         |
| 70000      | 0.77       | 0.74         | 0.69        | 0.61      | 0.51         |
| 80000      | 0.77       | 0.74         | 0.69        | 0.61      | 0.52         |
</details>

(b) Zaleplon MPO (zale)

![](images/b8652a1a25355e618e26b6c4cdbc1a20fb11b2263ea2f129ed1429feb4f295c1.jpg)

<details>
<summary>line</summary>

| Num Oracle | Blue Score | Orange Score | Green Score | Red Score | Purple Score |
| ---------- | ---------- | ------------ | ----------- | --------- | ------------ |
| 10000      | 0.55       | 0.55         | 0.50        | 0.55      | 0.50         |
| 20000      | 0.75       | 0.75         | 0.60        | 0.55      | 0.50         |
| 30000      | 0.82       | 0.80         | 0.65        | 0.57      | 0.52         |
| 40000      | 0.84       | 0.80         | 0.65        | 0.58      | 0.53         |
| 50000      | 0.84       | 0.80         | 0.65        | 0.58      | 0.54         |
| 60000      | 0.84       | 0.80         | 0.65        | 0.59      | 0.56         |
| 70000      | 0.84       | 0.80         | 0.65        | 0.59      | 0.57         |
| 80000      | 0.84       | 0.80         | 0.65        | 0.60      | 0.58         |
</details>

(c) Perindopril MPO (pdop)

![](images/c0f7c44dc56d28e756d9ffdd1826a34bdb6357955b002e28706a4e4bed01bf8a.jpg)

<details>
<summary>line</summary>

| Num Oracle | Score (Blue) | Score (Orange) | Score (Green) | Score (Red) |
| ---------- | ------------ | -------------- | ------------- | ----------- |
| 10000      | 0.84         | 0.84           | 0.84          | 0.84        |
| 20000      | 0.88         | 0.88           | 0.88          | 0.84        |
| 40000      | 0.90         | 0.90           | 0.90          | 0.84        |
| 60000      | 0.92         | 0.92           | 0.92          | 0.84        |
| 80000      | 0.93         | 0.93           | 0.93          | 0.84        |
</details>

(d) Osimertinib MPO (osmb)

![](images/edf35291a0222f511a2450f4e8a8cb2a6a12d7904d403ef088a9b52d48c14524.jpg)

<details>
<summary>text_image</summary>

CoBO (Ours) LOL-BO W-LBO TuRBO-L LS-BO
</details>

Figure 2: Optimization results with four different tasks on the Guacamol benchmark. The lines and range are the mean and standard deviation of three repetitions with the same parameters.

![](images/691c4a814b2f8f74a0626cc199f63d1e4031233b82f87dc1324aa7d93f05a7a4.jpg)

<details>
<summary>line</summary>

| Num Oracle | Score (Line 1) | Score (Line 2) | Score (Line 3) | Score (Line 4) | Score (Line 5) |
| ---------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| 40000      | 1.75           | 1.50           | 1.25           | 1.00           | 0.75           |
| 40100      | 1.50           | 1.25           | 1.00           | 0.75           | 0.50           |
| 40200      | 1.25           | 1.00           | 0.75           | 0.50           | 0.25           |
| 40300      | 1.00           | 0.75           | 0.50           | 0.25           | 0.10           |
| 40400      | 0.75           | 0.50           | 0.25           | 0.10           | 0.05           |
| 40500      | 0.50           | 0.25           | 0.10           | 0.05           | 0.02           |
</details>

(a) Arithmetic expression

![](images/6a833db23fcafc6b554b28fdb41907f46d1df95f74591ef34f782ef8d4232fee.jpg)

<details>
<summary>line</summary>

| Num Oracle | Score (Line 1) | Score (Line 2) | Score (Line 3) | Score (Line 4) |
| ---------- | -------------- | -------------- | -------------- | -------------- |
| 100        | -12.0          | -12.0          | -12.0          | -12.0          |
| 1000       | -13.5          | -13.0          | -13.5          | -14.0          |
| 2000       | -14.5          | -13.5          | -14.5          | -15.0          |
| 3000       | -15.5          | -14.0          | -15.0          | -15.5          |
</details>

(b) TDC DRD3

![](images/c7ad444aa020faab7dcc1cce959e376cf284739a59115b3c00c664c61b45afde.jpg)

<details>
<summary>text_image</summary>

CoBO (Ours) LOL-BO W-LBO TuRBO-L LS-BO
</details>

Figure 3: Optimization results with the arithmetic expression and TDC DRD3 benchmark. The lines and range are the mean and standard deviation of three repetitions with the same parameters.

In this section, we evaluate the main components of our model to analyze their contribution. We employ Perindopril MPO (pdop) task for our experiment and note that all scores of ablation studies are recorded when the oracle number is 20k to compare each case in limited oracle numbers as the original BO intended.

We ablate the three main components of our CoBO and report the results in Table 1. As latent space regularization $L_{z}$ comes from Lipschitz regularization and loss weighting schema aims to prioritize penalized input points, we conducted cascading experiments. Notably, we observed that performance decreased as the components were omitted, and the largest performance dropped (0.0229)

![](images/80d6b97f69c31023f441b1967663525a5da90e103d61f99e5970517e9c52ebf1.jpg)

<details>
<summary>line</summary>

| Epoch | W/o ℒ_align | W/ ℒ_align |
|-------|-------------|------------|
| 0     | 0.0         | 0.0        |
| 25    | 0.2         | 0.5        |
| 50    | 0.4         | 0.6        |
| 75    | 0.3         | 0.7        |
| 100   | 0.2         | 0.8        |
| 125   | 0.3         | 0.75       |
| 150   | 0.2         | 0.7        |
| 175   | 0.1         | 0.65       |
| 200   | 0.1         | 0.6        |
</details>

(a) Perindopril MPO (pdop)

![](images/86ad430af3f8103a25b35f2114886a416ea4d4e8f1040781617a85f5b35805f0.jpg)

<details>
<summary>line</summary>

| x  | w/o ℒ_align | w/ ℒ_align |
|----|-------------|------------|
| 0  | 0.0         | 0.0        |
| 10 | 0.0         | 0.1        |
| 20 | 0.0         | 0.3        |
| 30 | 0.0         | 0.3        |
| 40 | 0.1         | 0.4        |
| 50 | 0.0         | 0.3        |
| 60 | 0.0         | 0.5        |
| 70 | 0.0         | 0.6        |
</details>

(b) Arithmetic expression   
Figure 4: Effects of $L_{align}$ . The plot depicts the Pearson correlation between the distance of the latent vectors and the distance of the objective values over the Bayesian optimization process. Each line represents training with a different loss for the latent space during the optimization process: one with $L_{align}$ (orange) and the other without $L_{align}$ (blue), where $L_{align} = L_{Lip} + L_{z}$ . We measure the correlation after every VAE update.

when Lipschitz regularization was not employed. Table 2 reports the effectiveness of trust region recoordination. We observe that applying trust region recoordination makes a considerable difference within a small oracle budget, as the score increases 0.6844 to 0.7921 with recoordination.

# 3.3 Analysis on proposed regularizations

All the analyses were conducted on the Perindopril MPO (pdop) task from the Guacamol benchmark. We include an additional analysis with the arithmetic data. For convenience, we define the $L_{align}$ as follows:

$$
\mathcal {L} _ {\text { align }} = \mathcal {L} _ {\text { Lip }} + \mathcal {L} _ {\mathrm{z}}, \tag {12}
$$

which is the loss that aligns the latent space with the black-box objective function.

Effects of $L_{align}$ on correlation. In Section 2.2, we theoretically prove that $L_{align}$ increases the correlation between the distance in the latent space and the distance within the objective function. Here, we demonstrate our theorem with further quantitative analysis to show correlation changes during the Bayesian optimization process. Figure 4 shows Pearson correlation value between the latent space distance $\|z_{i}-z_{j}\|_{2}$ and the objective value distance $|y_{i}-y_{j}|$ . The blue and orange lines indicate the models with and without our align loss $L_{align}$ in Eq. 12, respectively. We measure the correlation with $10^{3}$ data point and every $10^{6}$ pair. The data is selected as the top $10^{3}$ points with the highest value among $10^{3}$ data, which are from training data for the VAE model and surrogate model. Over the training process, the Pearson correlation values with our align loss $L_{align}$ were overall higher compared to the baseline. Moreover, $L_{align}$ increases the Pearson correlation value high over 0.7 in Figure 4a which is normally regarded as the high correlation value. This means align loss increases the correlation between the distance of the latent vectors and the distance of the objective values effectively, leading to narrowing the gap between the latent space and the objective function.

Effects of $L_{align}$ on smoothness. To validate that our model encourages the smooth latent space, we qualitatively analyze our model with $L_{align}$ and without $L_{align}$ by visualizing the landscape of the latent space after the training finishes. In Figure 5, we visualize the top-k objective value and the corresponding latent vector in 2D space with the 2D scatter plot and the corresponding 3D plot for better understanding. We reduce the dimension of the latent vector to two dimensions using principal component analysis (PCA). The color means the normalized relative objective score value. The landscape of objective value according to the latent space with $L_{align}$ (right) is smoother than the case without $L_{align}$ (left). It demonstrates our $L_{align}$ loss encourages smoothness of the latent space with respect to the objective function. Note that due to the inherent discreteness of the black-box objective function, it's expected to observe some spaces between the clusters in the 2D plot. Still, this does not detract from our purpose of effectively aligning latent vectors with respect to their discrete objective values.

![](images/b7ec0af45f4ad34c465eaa174622ff794cef976595176d42b6f65d2826e11f60.jpg)

<details>
<summary>bubble</summary>

| x | y | Objective Value |
| --- | --- | --- |
| (data not extractable) | (data not extractable) | (data not extractable) |
</details>

![](images/6066479e59be5dd983eac296f623bc8ba66fcacede2a83499a9c190541d7331b.jpg)

<details>
<summary>area_stacked</summary>

| X | Y | Z |
|---|---|---|
| 0.0 | 0.0 | 0.0 |
| 0.1 | 0.2 | 0.1 |
| 0.2 | 0.4 | 0.2 |
| 0.3 | 0.6 | 0.3 |
| 0.4 | 0.8 | 0.4 |
| 0.5 | 1.0 | 0.5 |
| 0.6 | 1.2 | 0.6 |
| 0.7 | 1.4 | 0.7 |
| 0.8 | 1.6 | 0.8 |
| 0.9 | 1.8 | 0.9 |
| 1.0 | 2.0 | 1.0 |
| 1.1 | 1.8 | 0.9 |
| 1.2 | 1.6 | 0.8 |
| 1.3 | 1.4 | 0.7 |
| 1.4 | 1.2 | 0.6 |
| 1.5 | 1.0 | 0.5 |
| 1.6 | 0.8 | 0.4 |
| 1.7 | 0.6 | 0.3 |
| 1.8 | 0.4 | 0.2 |
| 1.9 | 0.2 | 0.1 |
| 2.0 | 0.0 | 0.0 |
</details>

(a) Without $\mathcal{L}_{\mathrm{align}}$

![](images/5cf8c7568564740086291b40f79c63e9476624148af3613902fe93051dacb616.jpg)

<details>
<summary>scatter</summary>

| x | y | Objective Value |
| --- | --- | --- |
| (various) | (various) | 0.00 to 1.00 |
</details>

![](images/aeadc7132233d8798c6c209b4ca9b0569b849ac4895e285f4eb6ed3d70512879.jpg)

<details>
<summary>natural_image</summary>

3D rendered model of a boat on a grid, colored by intensity (no text or symbols)
</details>

(b) With $\mathcal{L}_{\mathrm{align}}$   
Figure 5: Visualizations on the landscape of latent space for $L_{align}$ ablation. The scatter plot and corresponding 3D plot of the latent vectors with objective values. The landscape becomes much smoother with applying $L_{align}$ . A colorbar indicates the normalized objective value, where yellow means higher value and purple means lower value.

# 4 Related Works

# 4.1 Latent space Bayesian optimization

Latent space Bayesian optimization $[10, 12–14, 19, 21, 29, 30]$ aims to resolve the issues in optimization over high-dimensional, or structured input space by introducing Bayesian optimization over latent space. As the objective for structured inputs is usually defined over a large, complex space, the challenges can be alleviated by the lower-dimensional and continuous latent space. Variational autoencoders (VAEs) $[15]$ are commonly leveraged to learn the continuous embeddings for the latent space Bayesian optimizers. Some prior works propose novel architectures for decoders $[11, 12, 31–33]$ , while others introduce loss functions to improve the surrogate for learning the objective function $[13, 14, 18, 29]$ . Note that while the surrogate model (typically GP) is modeled based on the latent space, it is the input space which the objective value is obtained from. Although this results in an inherent gap in latent space Bayesian optimization, many prior works $[12, 14, 19, 21, 29]$ do not update the generative model for the latent space. LOL-BO $[10]$ seeks to address this gap by adapting the latent space to the GP prior, while $[13]$ suggests periodic weighted retraining to update the latent space.

# 4.2 Latent space regularization

Several previous studies in latent space have focused on learning the appropriate latent space for their tasks by incorporating additional regularizations or constraints alongside the reconstruction loss. These approaches have been applied in a wide range of areas, including molecule design $[34–36]$ , domain adaptation $[37, 38]$ , semantic segmentation $[39, 40]$ , representation learning $[41–43]$ , and reinforcement learning $[44]$ . Notably, the Lipschitz constraint is commonly employed to promote smoothness in diverse optimization problems. For example, $[42]$ introduces Lipschitz regularization in learning implicit neural functions to encourage smooth latent space for 3D shapes while $[45]$ penalizes the data pairs that violate the Lipschitz constraint in adversarial training. Additionally, CoFLO $[46]$ leverages Lipschitz-like regularization in latent space Bayesian optimization. Our method proposes regularization to close the gap inherent in the latent space Bayesian optimization. Concretely, we introduce Lipschitz regularization to increase the correlation between the distance of latent space and the distance of objective value and give more weight to the loss in the promising areas.

# 5 Conclusion and Discussion

In this paper, we addressed the problem of the inherent gap in the latent space Bayesian optimization and proposed Correlated latent space Bayesian Optimization. We introduce Lipschitz regularization which maximizes the correlation between the distance of latent space and the distance of objective value to close the gap between latent space and objective value. Also, we reduced the gap between latent space and input space with a loss weighting scheme, especially in the promising areas. Additionally, by trust region recoordination, we adjust the trust regions according to the updated latent space. Our experiments on various benchmarks with molecule generation and arithmetic fitting tasks demonstrate that our CoBO significantly improves state-of-the-art methods in LBO.

Limitations and broader impacts. Given the contribution of this work to molecular design optimization, careful consideration should be given to its potential impacts on the generation of toxic or harmful substances in the design of new chemicals. We believe that our work primarily has the positive potential of accelerating the chemical and drug development process, setting a new standard in latent space Bayesian optimization.

# Acknowledgments and Disclosure of Funding

This work was partly supported by ICT Creative Consilience program (IITP-2023-2020-0-01819) supervised by the IITP, the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (NRF-2023R1A2C2005373), and Samsung Research Funding & Incubation Center of Samsung Electronics under Project Number SRFC-IT1701-51.

# References

[1] Ksenia Korovina, Sailun Xu, Kirthevasan Kandasamy, Willie Neiswanger, Barnabas Poczos, Jeff Schneider, and Eric Xing. Chembo: Bayesian optimization of small organic molecules with synthesizable recommendations. In ICAIS, 2020.   
[2] José Miguel Hernández-Lobato, James Requeima, Edward O Pyzer-Knapp, and Alán Aspuru-Guzik. Parallel and distributed thompson sampling for large-scale accelerated exploration of chemical space. In ICML, 2017.   
[3] Ryan-Rhys Griffiths and José Miguel Hernández-Lobato. Constrained bayesian optimization for automatic chemical design using variational autoencoders. Chemical science, 2020.   
[4] Ke Wang and Alexander W Dowling. Bayesian optimization for chemical products and functional materials. Current Opinion in Chemical Engineering, 2022.   
[5] Roberto Calandra, André Seyfarth, Jan Peters, and Marc Peter Deisenroth. Bayesian optimization for learning gaits under uncertainty: An experimental comparison on a dynamic bipedal walker. Annals of Mathematics and Artificial Intelligence, 2016.   
[6] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimization of machine learning algorithms. In NeurIPS, 2012.   
[7] Eric Brochu, Vlad M Cora, and Nando De Freitas. A tutorial on bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. Arxiv, 2010.   
[8] Peter I Frazier. A tutorial on bayesian optimization. Arxiv, 2018.   
[9] Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P Adams, and Nando De Freitas. Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 2015.   
[10] Natalie Maus, Haydn Jones, Juston Moore, Matt J Kusner, John Bradshaw, and Jacob Gardner. Local latent space bayesian optimization over structured inputs. In NeurIPS, 2022.   
[11] Matt J Kusner, Brooks Paige, and José Miguel Hernández-Lobato. Grammar variational autoencoder. In ICML, 2017.

[12] Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In ICML, 2018.   
[13] Austin Tripp, Erik Daxberger, and José Miguel Hernández-Lobato. Sample-efficient optimization in the latent space of deep generative models via weighted retrainin. In NeurIPS, 2020.   
[14] Antoine Grosnit, Rasul Tutunov, Alexandre Max Maraval, Ryan-Rhys Griffiths, Alexander I Cowen-Rivers, Lin Yang, Lin Zhu, Wenlong Lyu, Zhitang Chen, Jun Wang, et al. High-dimensional bayesian optimisation with variational autoencoders and deep metric learning. Arxiv, 2021.   
[15] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014.   
[16] Irina Higgins, Loïc Matthey, Arka Pal, Christopher P. Burgess, Xavier Glorot, Matthew M. Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In ICLR, 2017.   
[17] Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein gan. Arxiv, 2017.   
[18] Aryan Deshwal and Jana Doppa. Combining latent space and structured kernels for bayesian optimization over combinatorial spaces. In NeurIPS, 2021.   
[19] Rafael Gómez-Bombarelli, Jennifer N Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D Hirzel, Ryan P Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. ACS central science, 2018.   
[20] David Eriksson, Michael Pearce, Jacob Gardner, Ryan D Turner, and Matthias Poloczek. Scalable global optimization via local bayesian optimization. In NeurIPS, 2019.   
[21] Samuel Stanton, Wesley Maddox, Nate Gruver, Phillip Maffettone, Emily Delaney, Peyton Greenside, and Andrew Gordon Wilson. Accelerating bayesian optimization for biological sequence design with denoising autoencoders. In ICLR, 2022.   
[22] John Lawrence. Moments of the noncentral chi distribution. Sankhya A, 2021.   
[23] David H Brookes and Jennifer Listgarten. Design by adaptive sampling. Arxiv, 2018.   
[24] Edward Snelson and Zoubin Ghahramani. Sparse gaussian processes using pseudo-inputs. In NeurIPS, 2005.   
[25] Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P Xing. Deep kernel learning. In AISTATS, 2016.   
[26] Kexin Huang, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec, Connor Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik. Therapeutics data commons: Machine learning datasets and tasks for drug discovery and development. In NeurIPS, 2021.   
[27] Nathan Brown, Marco Fiscato, Marwin HS Segler, and Alain C Vaucher. Guacamol: benchmarking models for de novo molecular design. Journal of chemical information and modeling, 2019.   
[28] Sungsoo Ahn, Junsu Kim, Hankook Lee, and Jinwoo Shin. Guiding deep molecular optimization with genetic exploration. In NeurIPS, 2020.   
[29] Stephan Eissman, Daniel Levy, Rui Shu, Stefan Bartzsch, and Stefano Ermon. Bayesian optimization and attribute adjustment. In UAI, 2018.   
[30] Eero Siivola, Andrei Paleyes, Javier González, and Aki Vehtari. Good practices for bayesian optimization of high dimensional structured spaces. Applied AI Letters, 2021.   
[31] Bidisha Samanta, Abir De, Gourhari Jana, Vicenç Gómez, Pratim Kumar Chattaraj, Niloy Ganguly, and Manuel Gomez-Rodriguez. Nevae: A deep generative model for molecular graphs. In AAAI, 2019.

[32] Hiroshi Kajino. Molecular hypergraph grammar with its application to molecular optimization. In ICML, 2019.   
[33] Hanjun Dai, Yingtao Tian, Bo Dai, Steven Skiena, and Le Song. Syntax-directed variational autoencoder for structured data. In ICLR, 2018.   
[34] Egbert Castro, Abhinav Godavarthi, Julian Rubinfien, Kevin Givechian, Dhananjay Bhaskar, and Smita Krishnaswamy. Transformer-based protein generation with regularized latent space optimization. Nature Machine Intelligence, 2022.   
[35] Pascal Notin, José Miguel Hernández-Lobato, and Yarin Gal. Improving black-box optimization in vae latent space using decoder uncertainty. In NeurIPS, 2021.   
[36] ANM Abeer, Nathan Urban, M Ryan Weil, Francis J Alexander, and Byung-Jun Yoon. Multi-objective latent space optimization of generative molecular design models. Arxiv, 2022.   
[37] Guoliang Kang, Lu Jiang, Yi Yang, and Alexander G Hauptmann. Contrastive adaptation network for unsupervised domain adaptation. In CVPR, 2019.   
[38] Lei Tian, Yongqiang Tang, Liangchen Hu, Zhida Ren, and Wensheng Zhang. Domain adaptation by class centroid matching and local manifold self-learning. IEEE Transactions on Image Processing, 2020.   
[39] Francesco Barbato, Marco Toldo, Umberto Michieli, and Pietro Zanuttigh. Latent space regularization for unsupervised domain adaptation in semantic segmentation. In CVPRW, 2021.   
[40] Umberto Michieli and Pietro Zanuttigh. Continual semantic segmentation via repulsion-attraction of sparse and disentangled latent representations. In CVPR, 2021.   
[41] Samarth Sinha and Adji Bousso Dieng. Consistency regularization for variational autoencoders. In NeurIPS, 2021.   
[42] Hsueh-Ti Derek Liu, Francis Williams, Alec Jacobson, Sanja Fidler, and Or Litany. Learning smooth neural functions via lipschitz regularization. In SIGGRAPH, 2022.   
[43] Ramana Subramanyam Sundararaman, Riccardo Marin, Emanuele Rodola, and Maks Ovsjanikov. Reduced representation of deformation fields for effective non-rigid shape matching. In NeurIPS, 2022.   
[44] Mete Kemertas and Tristan Aumentado-Armstrong. Towards robust bisimulation metric learning. In NeurIPS, 2021.   
[45] Dávid Terjék. Adversarial lipschitz regularization. In ICLR, 2020.   
[46] Fengxue Zhang, Yair Altas, Louise Fan, Kaustubh Vinchure, Brian Nord, and Yuxin Chen. Design of physical experiments via collision-free latent space optimization. In NeurIPSW, 2020.   
[47] Jan H Jensen. A graph-based genetic algorithm and generative model/monte carlo tree search for the exploration of chemical space. Chemical science, 2019.   
[48] Marwin HS Segler, Thierry Kogej, Christian Tyrchan, and Mark P Waller. Generating focused molecule libraries for drug discovery with recurrent neural networks. ACS central science, 2018.   
[49] Jiaxuan You, Bowen Liu, Zhitao Ying, Vijay Pande, and Jure Leskovec. Graph convolutional policy network for goal-directed molecular graph generation. In NeurIPS, 2018.   
[50] Yutong Xie, Chence Shi, Hao Zhou, Yuwei Yang, Weinan Zhang, Yong Yu, and Lei Li. Mars: Markov molecular sampling for multi-objective drug discovery. In ICLR, 2020.   
[51] Zhenpeng Zhou, Steven Kearnes, Li Li, Richard N Zare, and Patrick Riley. Optimization of molecules via deep reinforcement learning. Scientific reports, 2019.

Summary. We provide additional experimental results/details and analysis in this supplement as: (A) analysis on regularization $L_{z}$ , (B) the proof of Theorem 1, (C) additional results on Guacamol Benchmarks, (D) additional results on DRD3 task, (E) efficiency analysis, and (F) implementation details.

# A Analysis on Regularization $L_{z}$

Here, we analyze the necessity of regularization $L_{z}$ . Based on Theorem 1 in the main paper, to increase the correlation between the distance of latent vectors and the differences in their corresponding objective values, we need to keep the distance between the latent vectors z to be a constant. Figure 6 displays the box plot of distances between z at each iteration of BO. The box represents the first and third quartiles, and the whiskers represent the 10 and 90 percentiles. Each data point has a top-k score of objective value. As in Figure 6, the model only with Lipschitz regularization $L_{Lip}$ (i.e., without $L_{z}$ ) increases the distance between the latent vectors $\|z_{i}-z_{j}\|_{2}$ since it is an easy way to minimize $L_{Lip}$ given as

$$
\mathcal {L} _ {\mathrm{Lip}} = \sum_ {i, j \leq N} \max \left(0, \frac {| y _ {i} - y _ {j} |}{\| \mathbf {z} _ {i} - \mathbf {z} _ {j} \| _ {2}} - L\right). \tag {13}
$$

However, when applying both regularizations $L_{Lip}$ and $L_{z}$ , we observe that the distance is preserved within a certain range, similar to the beginning of training.

![](images/e1ab37467c85295285b740f13111868058933dfe6bb96901cc14880f9aee7448.jpg)

<details>
<summary>line</summary>

| # of BO iterations | w/o regularizer ℒz | w/ regularizer ℒz |
| ------------------ | ------------------ | ----------------- |
| 0                  | 24.0               | 22.0              |
| 10                 | 27.0               | 23.0              |
| 20                 | 30.0               | 24.0              |
| 30                 | 33.0               | 25.0              |
| 40                 | 35.0               | 26.0              |
| 50                 | 37.0               | 27.0              |
| 60                 | 39.0               | 28.0              |
| 70                 | 38.0               | 27.0              |
| 80                 | 39.0               | 28.0              |
| 90                 | 38.0               | 27.0              |
| 100                | 39.0               | 28.0              |
| 110                | 40.0               | 29.0              |
| 120                | 41.0               | 30.0              |
| 130                | 42.0               | 31.0              |
| 135                | 43.0               | 32.0              |
</details>

Figure 6: Effects on Regularization $L_{z}$ . The green and red box plots depict the distances of the latent vectors with and without regularization term $L_{z}$ , respectively.

# B Proof of Theorem 1

Theorem 1. Let $D_Z = d_Z(Z_1, Z_2)$ and $D_Y = d_Y(f(Z_1), f(Z_2))$ be random variables where $Z_1, Z_2$ are i.i.d. random variables, $f$ is an L-Lipschitz continuous function, and $d_Z, d_Y$ are distance functions. Then, the correlation between $D_Z$ and $D_Y$ is lower bounded as

$$
D _ {Y} \leq L D _ {Z} \Rightarrow \mathsf {C o r r} _ {D _ {Z}, D _ {Y}} \geq \frac {\frac {1}{L} (\sigma_ {D _ {Y}} ^ {2} + \mu_ {D _ {Y}} ^ {2}) - L \mu_ {D _ {Z}} ^ {2}}{\sqrt {\sigma_ {D _ {Z}} ^ {2} \sigma_ {D _ {Y}} ^ {2}}},
$$

where $\mu_{D_Z},\sigma_{D_Z}^2,\mu_{D_Y}$ , and $\sigma_{D_Y}^2$ are the mean and variance of $D_Z$ and $D_Y$ respectively.

Proof. The correlation between $D_Z$ and $D_Y$ is:

$$
\operatorname{Corr} _ {D _ {Z}, D _ {Y}} = \frac {\operatorname{Cov} \left(D _ {Z} , D _ {Y}\right)}{\sqrt {\operatorname{Var} \left(D _ {Z}\right) \operatorname{Var} \left(D _ {Y}\right)}} \tag {14}
$$

$$
= \frac {\mathbb {E} [ (D _ {Z} - \mathbb {E} [ D _ {Z} ]) (D _ {Y} - \mathbb {E} [ D _ {Y} ]) ]}{\sqrt {\operatorname{Var} (D _ {Z}) \operatorname{Var} (D _ {Y})}} \tag {15}
$$

$$
= \frac {\mathbb {E} [ D _ {Z} D _ {Y} ] - \mathbb {E} [ D _ {Z} ] \mathbb {E} [ D _ {Y} ]}{\sqrt {\operatorname{Var} (D _ {Z}) \operatorname{Var} (D _ {Y})}}. \tag {16}
$$

By L-Lipschitz continuity, we have:

$$
d _ {Y} (f (Z _ {1}), f (Z _ {2})) \leq L d _ {Z} (Z _ {1}, Z _ {2}) \Rightarrow D _ {Y} \leq L D _ {Z}. \tag {17}
$$

Hence, the correlation is bounded as follows:

$$
\begin{array}{l} \operatorname{Corr} _ {D _ {Z}, D _ {Y}} = \frac {\mathbb {E} [ D _ {Z} D _ {Y} ] - \mathbb {E} [ D _ {Z} ] \mathbb {E} [ D _ {Y} ]}{\sqrt {\operatorname{Var} (D _ {Z}) \operatorname{Var} (D _ {Y})}} (18) \\ \geq \frac {\mathbb {E} [ \frac {1}{L} D _ {Y} D _ {Y} ] - \mathbb {E} [ D _ {Z} ] \mathbb {E} [ L D _ {Z} ]}{\sqrt {\operatorname{Var} (D _ {Z}) \operatorname{Var} (D _ {Y})}} (19) \\ = \frac {\frac {1}{L} \mathbb {E} [ (D _ {Y}) ^ {2} ] - L \mathbb {E} [ D _ {Z} ] \mathbb {E} [ D _ {Z} ]}{\sqrt {\operatorname{Var} (D _ {Z}) \operatorname{Var} (D _ {Y})}} (20) \\ = \frac {\frac {1}{L} \left(\operatorname{Var} \left[ D _ {Y} \right] + \left(\mathbb {E} \left[ D _ {Y} \right]\right) ^ {2}\right) - L \left(\mathbb {E} \left[ D _ {Z} \right]\right) ^ {2}}{\sqrt {\operatorname{Var} \left(D _ {Z}\right) \operatorname{Var} \left(D _ {Y}\right)}} (21) \\ = \frac {\frac {1}{L} (\sigma_ {D _ {Y}} ^ {2} + \mu_ {D _ {Y}} ^ {2}) - L \mu_ {D _ {Z}} ^ {2}}{\sqrt {\sigma_ {D _ {Z}} ^ {2} \sigma_ {D _ {Y}} ^ {2}}}. (22) \\ \end{array}
$$

![](images/f8fd7fd3a4d751fa03c2b6d7b70e8b34a6f60ed950d77f2bef277297dce60059.jpg)

# C Additional Results on Guacamol Benchmarks

In addition to the four tasks of the Guacamol benchmark that we previously mentioned, we also evaluate our model on three additional tasks: Ranolazine MPO, Aripiprazole similarity, and Valsartan SMART. The experimental settings for these additional tasks are the same with the settings applied to the initial four tasks. The results of the experiments are present in Figure 7. For the Valsartan SMART task, as depicted in Figure 7c, three models find the optimal point, note that our model finds the optimal point faster than other models.

# D Additional Results on DRD3 Task

We compare our results with the leaderboard $^{3}$ of the DRD3 task in Table 3. Note that we use a random initialized dataset of 100. We specifically compare the Top-1 scores as absolute values, which are also reported in our line plot.

# E Efficiency Analysis

We conduct an efficiency analysis on every tasks: the Guacamol benchmarks, the DRD3 task, and the arithmetic fitting task. In our analysis, we compare our model with four baseline models. For a fair comparison, we set up experiments for every model in the same condition, as we use the CPU of AMD EPYC 7742 with a single NVIDIA RTX 2080 TI. Note that since these are CPU-intensive tasks, the CPU is crucial to the speed of execution. We report runtimes, the found best score, and the

![](images/b4245d986649d4192169810a14d5b4bce4d7abcd93f3b2f20f38d03d0df5a872.jpg)

<details>
<summary>line</summary>

| Num Oracle | Score (Blue) | Score (Orange) | Score (Green) | Score (Red) | Score (Purple) |
| ---------- | ------------ | -------------- | ------------- | ----------- | -------------- |
| 10000      | 0.75         | 0.75           | 0.75          | 0.75        | 0.75           |
| 20000      | 0.90         | 0.85           | 0.85          | 0.78        | 0.75           |
| 40000      | 0.95         | 0.92           | 0.90          | 0.80        | 0.78           |
| 60000      | 0.95         | 0.94           | 0.91          | 0.80        | 0.78           |
| 80000      | 0.95         | 0.95           | 0.92          | 0.80        | 0.78           |
</details>

(a) Ranolazine MPO (rano)

![](images/6ce294b4c612f8418a1f12c1f2aeb9d928bd229e140de9c62ccf6ab107954868.jpg)

<details>
<summary>line</summary>

| Num Oracle | Score (Blue) | Score (Orange) | Score (Green) | Score (Red) |
| ---------- | ------------ | -------------- | ------------- | ----------- |
| 10000      | 0.70         | 0.70           | 0.70          | 0.70        |
| 20000      | 0.74         | 0.73           | 0.72          | 0.70        |
| 30000      | 0.77         | 0.75           | 0.72          | 0.70        |
| 40000      | 0.79         | 0.76           | 0.72          | 0.70        |
| 50000      | 0.81         | 0.78           | 0.72          | 0.70        |
| 60000      | 0.81         | 0.80           | 0.72          | 0.70        |
| 70000      | 0.82         | 0.81           | 0.72          | 0.70        |
| 80000      | 0.82         | 0.81           | 0.72          | 0.70        |
</details>

(b) Aripiprazole similarity (adip)

![](images/d107b192438a53738df271ced604a8eb0716f564f5a4224d89931c98584caeee.jpg)

<details>
<summary>line</summary>

| Num Oracle | Score (Blue) | Score (Orange) | Score (Green) | Score (Red) |
| ---------- | ------------ | -------------- | ------------- | ----------- |
| 10000      | 0.0          | 0.0            | 0.0           | 0.0         |
| 20000      | 0.6          | 0.4            | 0.0           | 0.0         |
| 30000      | 1.0          | 1.0            | 0.7           | 0.0         |
| 40000      | 1.0          | 1.0            | 1.0           | 0.0         |
| 50000      | 1.0          | 1.0            | 1.0           | 0.0         |
| 60000      | 1.0          | 1.0            | 1.0           | 0.0         |
| 70000      | 1.0          | 1.0            | 1.0           | 0.0         |
| 80000      | 1.0          | 1.0            | 1.0           | 0.0         |
</details>

(c) Valsartan SMART (valt)   
![](images/cc684cd81d1b7e1a47545ee6b44a1afa7e2a1dceb1da2fadf18927695af03f4f.jpg)

<details>
<summary>text_image</summary>

CoBO (Ours) LOL-BO W-LBO TuRBO-L LS-BO
</details>

Figure 7: Optimization results with the additional three tasks on the Guacamol benchmark. The lines and range are the mean and standard deviation of three repetitions with the same parameters.

Table 3: Optimization results with best score on TDC DRD3 task. Baselines are reported on leaderboard. 

<table><tr><td>Oracle calls</td><td>CoBO (Ours)</td><td>Graph-GA[47]</td><td>SMILES-LSTM[48]</td><td>GCPN[49]</td><td>MARS[50]</td><td>MolDQN[51]</td></tr><tr><td>100</td><td>-11.80</td><td>-11.13</td><td>-11.77</td><td>-9.10</td><td>-7.02</td><td>-11.63</td></tr><tr><td>500</td><td>-13.57</td><td>-12.50</td><td>-11.37</td><td>-11.97</td><td>-9.83</td><td>-7.62</td></tr><tr><td>1000</td><td>-13.97</td><td>-13.23</td><td>-11.97</td><td>-12.03</td><td>-11.10</td><td>-7.80</td></tr><tr><td>3000</td><td>-15.37</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

number of oracles, where we measure runtimes of executing a certain number of oracles as the wall clock time. The number of oracle calls increases each time as unique inputs are passed to the black-box objective function. Table 4, 5, 6 demonstrates that CoBO achieves comparable runtime with the same number of oracle calls, while outperforming the baselines by finding superior solutions.

# F Implementation Details

In our implementation, we use PyTorch $^{4}$ , BoTorch $^{5}$ and GPyTorch $^{6}$ . Additionally, we utilize the codebase $^{7}$ of [10] for the implementation. The SELFIES VAE is pretrained with 1.27M molecules in Guacamol benchmark and DRD3 task from [27] and the Grammar VAE is pretrained 40K expression in Arithmetic data from [14]. On the DRD3 task, we modify the evaluation metric from minimization to maximization by simply changing the sign of the objective values. In our experiments, we mainly

employ NVIDIA V100 and Intel Xeon Gold 6230. In this setup, the pdop tasks with a budget of 70k oracle, took an average of 11 hours.

# F.1 Hyperparameters

We grid search coefficients of our proposed regularizations $\mathcal{L}_{\mathrm{Lip\_W}}$ and $\mathcal{L}_{\mathrm{z}}$ , in the range of [10,100,1000] for $\mathcal{L}_{\mathrm{Lip\_W}}$ and [0.1,1] for $\mathcal{L}_{\mathrm{z}}$ . For some tasks, we didn't search for these hyperparameters, and their coefficients are provided in Table 8. The selected coefficients from this search are presented in Table 7. For other hyperparameters, such as coefficients for other losses, batch size, and learning rate, we set values according to Table 9.

Table 4: Efficiency comparison on Guacamol benchmarks within 70k evaluation budget. 

<table><tr><td></td><td>Model</td><td>CoBO (Ours)</td><td>LOL-BO</td><td>W-LBO</td><td>TuRBO-L</td><td>LS-BO</td></tr><tr><td rowspan="6">med2</td><td>Oracle calls</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td></tr><tr><td>Wall clock time (min)</td><td>1057.8</td><td>1080.4</td><td>175.8</td><td>246.7</td><td>1580.3</td></tr><tr><td>Found Best Score</td><td>0.3828</td><td>0.3530</td><td>0.3118</td><td>0.3118</td><td>0.3464</td></tr><tr><td>Oracle calls</td><td>55k</td><td>33k</td><td>70k</td><td>48k</td><td>16k</td></tr><tr><td>Wall clock time (min)</td><td>175.8</td><td>175.8</td><td>175.8</td><td>175.8</td><td>175.8</td></tr><tr><td>Found Best Score</td><td>0.3828</td><td>0.3434</td><td>0.3118</td><td>0.3118</td><td>0.3295</td></tr><tr><td rowspan="6">adip</td><td>Oracle calls</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td></tr><tr><td>Wall clock time (min)</td><td>4986.3</td><td>3340.7</td><td>198.2</td><td>236.3</td><td>1320.8</td></tr><tr><td>Found Best Score</td><td>0.8133</td><td>0.8086</td><td>0.6983</td><td>0.6983</td><td>0.7186</td></tr><tr><td>Oracle calls</td><td>32k</td><td>28k</td><td>70k</td><td>58k</td><td>13k</td></tr><tr><td>Wall clock time (min)</td><td>198.2</td><td>198.2</td><td>198.2</td><td>198.2</td><td>198.2</td></tr><tr><td>Found Best Score</td><td>0.7921</td><td>0.7466</td><td>0.6983</td><td>0.6983</td><td>0.7186</td></tr><tr><td rowspan="6">pdop</td><td>Oracle calls</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td></tr><tr><td>Wall clock time (min)</td><td>1020.9</td><td>1920.4</td><td>168.4</td><td>268.1</td><td>840.6</td></tr><tr><td>Found Best Score</td><td>0.8343</td><td>0.7959</td><td>0.5855</td><td>0.5736</td><td>0.6514</td></tr><tr><td>Oracle calls</td><td>33k</td><td>28k</td><td>70k</td><td>38k</td><td>18k</td></tr><tr><td>Wall clock time (min)</td><td>168.4</td><td>168.4</td><td>168.4</td><td>168.4</td><td>168.4</td></tr><tr><td>Found Best Score</td><td>0.8343</td><td>0.7948</td><td>0.5855</td><td>0.5233</td><td>0.6312</td></tr><tr><td rowspan="6">rano</td><td>Oracle calls</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td></tr><tr><td>Wall clock time (min)</td><td>3940.5</td><td>2820.9</td><td>340.6</td><td>276.1</td><td>2320.6</td></tr><tr><td>Found Best Score</td><td>0.9550</td><td>0.9468</td><td>0.8045</td><td>0.7766</td><td>0.9226</td></tr><tr><td>Oracle calls</td><td>42k</td><td>41k</td><td>55k</td><td>70k</td><td>21k</td></tr><tr><td>Wall clock time (min)</td><td>276.1</td><td>276.1</td><td>276.1</td><td>276.1</td><td>276.1</td></tr><tr><td>Found Best Score</td><td>0.9486</td><td>0.9433</td><td>0.8045</td><td>0.7766</td><td>0.9166</td></tr><tr><td rowspan="6">valt</td><td>Oracle calls</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td></tr><tr><td>Wall clock time (min)</td><td>560.3</td><td>760.5</td><td>304.1</td><td>234.2</td><td>1940.4</td></tr><tr><td>Found Best Score</td><td>0.9982</td><td>0.9982</td><td>4e-14</td><td>4e-33</td><td>0.9917</td></tr><tr><td>Oracle calls</td><td>51k</td><td>38k</td><td>57k</td><td>70k</td><td>23k</td></tr><tr><td>Wall clock time (min)</td><td>234.2</td><td>234.2</td><td>234.2</td><td>234.2</td><td>234.2</td></tr><tr><td>Found Best Score</td><td>0.9982</td><td>0.9942</td><td>4.8532e-14</td><td>4875e-36</td><td>0.6533</td></tr><tr><td rowspan="6">zale</td><td>Oracle calls</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td></tr><tr><td>Wall clock time (min)</td><td>374.7</td><td>1320.2</td><td>366.7</td><td>150.4</td><td>840.5</td></tr><tr><td>Found Best Score</td><td>0.7733</td><td>0.7521</td><td>0.6024</td><td>0.5142</td><td>0.6366</td></tr><tr><td>Oracle calls</td><td>46k</td><td>24k</td><td>35k</td><td>70k</td><td>11k</td></tr><tr><td>Wall clock time (min)</td><td>150.4</td><td>150.4</td><td>150.4</td><td>150.4</td><td>150.4</td></tr><tr><td>Found Best Score</td><td>0.7733</td><td>0.7415</td><td>0.5633</td><td>0.5142</td><td>0.5833</td></tr><tr><td rowspan="6">osmb</td><td>Oracle calls</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td><td>70k</td></tr><tr><td>Wall clock time (min)</td><td>477.6</td><td>840.1</td><td>372.2</td><td>210.2</td><td>743.4</td></tr><tr><td>Found Best Score</td><td>0.9267</td><td>0.9233</td><td>0.8336</td><td>0.8481</td><td>0.8933</td></tr><tr><td>Oracle calls</td><td>59k</td><td>43k</td><td>46k</td><td>70k</td><td>19k</td></tr><tr><td>Wall clock time (min)</td><td>210.2</td><td>210.2</td><td>210.2</td><td>210.2</td><td>210.2</td></tr><tr><td>Found Best Score</td><td>0.9233</td><td>0.9167</td><td>0.8332</td><td>0.8481</td><td>0.8866</td></tr></table>

Table 5: Efficiency comparison on DRD3 benchmark within 3k evaluation budget. 

<table><tr><td>Model</td><td>CoBO (Ours)</td><td>LOL-BO</td><td>W-LBO</td><td>TuRBO-L</td><td>LS-BO</td></tr><tr><td>Oracle calls</td><td>3k</td><td>3k</td><td>3k</td><td>3k</td><td>3k</td></tr><tr><td>Wall clock time (hr)</td><td>86.4</td><td>65.7</td><td>40.7</td><td>34.3</td><td>67.4</td></tr><tr><td>Found Best Score</td><td>-15.4</td><td>-14.6</td><td>-12.3</td><td>-12.2</td><td>-13.9</td></tr><tr><td>Oracle calls</td><td>1.5k</td><td>1.5k</td><td>2.6k</td><td>3k</td><td>2.1k</td></tr><tr><td>Wall clock time (hr)</td><td>34.3</td><td>34.3</td><td>34.3</td><td>34.3</td><td>34.3</td></tr><tr><td>Found Best Score</td><td>-14.5</td><td>-14.2</td><td>-12.3</td><td>-12.2</td><td>-13.6</td></tr></table>

Table 6: Efficiency comparison on arithmetic expression fitting task within 500 evaluation budget. 

<table><tr><td>Model</td><td>CoBO (Ours)</td><td>LOL-BO</td><td>W-LBO</td><td>TuRBO-L</td><td>LS-BO</td></tr><tr><td>Oracle calls</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td></tr><tr><td>Wall clock time (min)</td><td>5.4</td><td>11.3</td><td>11.1</td><td>338.1</td><td>1620.7</td></tr><tr><td>Found Best Score</td><td>0.1468</td><td>0.4624</td><td>0.5848</td><td>0.7725</td><td>0.5533</td></tr><tr><td>Oracle calls</td><td>500</td><td>330</td><td>173</td><td>0</td><td>0</td></tr><tr><td>Wall clock time (min)</td><td>5.4</td><td>5.4</td><td>5.4</td><td>5.4</td><td>5.4</td></tr><tr><td>Found Best Score</td><td>0.1468</td><td>0.5467</td><td>1.0241</td><td>1.521</td><td>1.521</td></tr></table>

Table 7: Coefficients of our proposed regularizations determined by grid search. 

<table><tr><td></td><td>med2</td><td>osmb</td><td>pdop</td><td>zale</td><td>Arithmetic</td><td>DRD3</td></tr><tr><td>Coefficient of  $\mathcal{L}_{\text{Lip\_W}}$ </td><td>1e3</td><td>1e2</td><td>1e2</td><td>1e3</td><td>1e1</td><td>1e1</td></tr><tr><td>Coefficient of  $\mathcal{L}_{z}$ </td><td>1e0</td><td>1e0</td><td>1e-1</td><td>1e0</td><td>1e-1</td><td>1e0</td></tr></table>

Table 8: Coefficients of our proposed regularizations w/o search. 

<table><tr><td></td><td>rano</td><td>adip</td><td>valt</td></tr><tr><td>Coefficient of  $\mathcal{L}_{\text{Lip\_W}}$ </td><td>1e2</td><td>1e2</td><td>1e2</td></tr><tr><td>Coefficient of  $\mathcal{L}_{z}$ </td><td>1e-1</td><td>1e-1</td><td>1e-1</td></tr></table>

Table 9: Other hyperparameters used in the experiments. 

<table><tr><td>Parameter</td><td>Guacamol</td><td>Arithmetic</td><td>DRD3</td></tr><tr><td>Learning rate</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Coefficient of  $\mathcal{L}_{\text{surr}}$ </td><td>1</td><td>1</td><td>1</td></tr><tr><td>Coefficient of  $\mathcal{L}_{\text{recon\_W}}$ </td><td>1</td><td>1</td><td>1</td></tr><tr><td>Coefficient of  $\mathcal{L}_{\text{KL}}$ </td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>Quantile of objective value for loss weighting</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td>Standard deviation  $\sigma$  for loss weighting</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td># initial datapoints  $N$ </td><td>10000</td><td>40000</td><td>100</td></tr><tr><td>Latent update interval  $N_{\text{fail}}$ </td><td>10</td><td>10</td><td>10</td></tr><tr><td>Batch size</td><td>10</td><td>5</td><td>1</td></tr><tr><td># top- $k$  used training</td><td>1000</td><td>10</td><td>10</td></tr></table>