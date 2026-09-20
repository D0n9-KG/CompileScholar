# ON THE EFFECT OF BATCH SIZE IN BYZANTINE-ROBUST DISTRIBUTED LEARNING

Yi-Rui Yang Chang-Wei Shi Wu-Jun Li\*

National Key Laboratory for Novel Software Technology,

Department of Computer Science and Technology,

Nanjing University, Nanjing, China

{yangyr, shicw}@smail.nju.edu.cn, liwujun@nju.edu.cn

# ABSTRACT

Byzantine-robust distributed learning (BRDL), in which computing devices are likely to behave abnormally due to accidental failures or malicious attacks, has recently become a hot research topic. However, even in the independent and identically distributed (i.i.d.) case, existing BRDL methods will suffer a significant drop on model accuracy due to the large variance of stochastic gradients. Increasing batch size is a simple yet effective way to reduce the variance. However, when the total number of gradient computation is fixed, a too-large batch size will lead to a too-small iteration number (update number), which may also degrade the model accuracy. In view of this challenge, we mainly study the effect of batch size when the total number of gradient computation is fixed in this work. In particular, we show that when the total number of gradient computation is fixed, the optimal batch size corresponding to the tightest theoretical upper bound in BRDL increases with the fraction of Byzantine workers. Therefore, compared to the case without attacks, a larger batch size is preferred when under Byzantine attacks. Motivated by the theoretical finding, we propose a novel method called Byzantine-robust stochastic gradient descent with normalized momentum (ByzSGDnm) in order to further increase model accuracy in BRDL. We theoretically prove the convergence of ByzSGDnm for general non-convex cases under Byzantine attacks. Empirical results show that when under Byzantine attacks, using a relatively large batch size can significantly increase the model accuracy, which is consistent with our theoretical results. Moreover, ByzSGDnm can achieve higher model accuracy than existing BRDL methods when under deliberately crafted attacks. In addition, we empirically show that increasing batch size has the bonus of training acceleration.

# 1 INTRODUCTION

Distributed learning has attracted much attention (Haddadpour et al., 2019; Jaggi et al., 2014; Lee et al., 2017; Lian et al., 2017; Ma et al., 2015; Shamir et al., 2014; Sun et al., 2018; Yang, 2013; Yu et al., 2019a;b; Zhao et al., 2017; 2018; Zhou et al., 2018; Zinkevich et al., 2010) for years due to its wide application. In traditional distributed learning, it is typically assumed that there is no failure or attack. However, in some real-world applications such as edge-computing (Shi et al., 2016) and federated learning (McMahan & Ramage, 2017), the service provider (also known as the server) usually has weak control over computing nodes (also known as workers). In these cases, various software and hardware failures may happen on workers (Xie et al., 2019). Worse even, some workers may get hacked by a malicious third party and intentionally send wrong information to foil the distributed learning process (Kairouz et al., 2021). The workers under failure or attack are also called Byzantine workers. Distributed learning with the existence of Byzantine workers, which is also known as Byzantine-robust distributed learning (BRDL), has recently become a hot research topic (Bernstein et al., 2019; Bulusu et al., 2021; Chen et al., 2018; Damaskinos et al., 2018; Diakonikolas et al., 2017; Diakonikolas & Kane, 2019; Konstantinidis & Ramamoorthy, 2021; Lamport et al., 2019; Rajput et al., 2019; Sohn et al., 2020; Wu et al., 2020; Yang & Li, 2021; 2023; Yang et al., 2020; Yin et al., 2019).

A typical way to obtain Byzantine robustness is to substitute the mean aggregator with robust aggregators such as Krum (Blanchard et al., 2017), geometric median (Chen et al., 2017), coordinate-wise median (Yin et al., 2018), centered clipping (Karimireddy et al., 2021), and so on. However, when there are Byzantine workers, even if robust aggregators are used, it is inevitable that an aggregation error will be introduced, which is the difference between the aggregated result and the true mean value. Furthermore, even in the independent and identically distributed (i.i.d.) cases, the aggregation error could be large due to the large variance of stochastic gradients (Karimireddy et al., 2021) which are typical values sent from workers to the server for parameter updating. The large aggregation error would make BRDL methods fail (Xie et al., 2020).

It has been shown in existing works that the variance of the values from non-Byzantine workers can be reduced by using local momentum on workers (Allen-Zhu et al., 2020; El-Mhamdi et al., 2021; Farhadkhani et al., 2022; Karimireddy et al., 2021). However, as the empirical results in our work will show, even if local momentum has been used, existing BRDL methods will suffer a significant drop on model accuracy when under attacks. Therefore, more sophisticated techniques are required to further reduce the variance of stochastic gradients.

Increasing batch size is a simple yet effective way to reduce the variance. However, when the total number of gradient computation is fixed, a too-large batch size will lead to a too-small iteration number (update number), which may also degrade the model accuracy (Goyal et al., 2017; Hoffer et al., 2017; Keskar et al., 2017; You et al., 2020; Zhao et al., 2020; 2023). In view of this challenge, we mainly study the effect of batch size in i.i.d. cases when the total number of gradient computation is fixed. The main contributions of this work are listed as follows:

- We show that when the total number of gradient computation is fixed, the optimal batch size corresponding to the tightest theoretical upper bound in BRDL increases with the fraction of Byzantine workers.   
- Motivated by the theoretical finding, we propose a novel method called Byzantine-robust stochastic gradient descent with normalized momentum (ByzSGDnm) in order to further increase model accuracy in BRDL.   
- We theoretically prove the convergence of ByzSGDnm for non-convex cases under attacks.   
- We empirically show that when under Byzantine attacks, compared to the cases of small batch size, setting a relatively large batch size can significantly increase the model accuracy. Moreover, ByzSGDnm can achieve higher model accuracy than existing BRDL methods when under deliberately crafted attacks.   
- In addition, increasing batch size has the bonus of training acceleration, which is verified by our empirical results.

# 2 PRELIMINARY

In this paper, we mainly focus on the following optimization problem:

$$
\min _ {\mathbf {w} \in \mathbb {R} ^ {d}} F (\mathbf {w}) = \mathbb {E} _ {\xi \sim \mathcal {D}} [ f (\mathbf {w}, \xi) ], \tag {1}
$$

where $w \in R^{d}$ is the model parameter and D is the distribution of training data. In addition, we mainly focus on the widely-used parameter-server (PS) framework in this work, where there are m computing nodes (workers) that collaborate to train the learning model under the coordination of a central server. Each worker can independently draw samples $\xi$ from data distribution D. That is to say, we focus on the i.i.d. cases in this paper. Moreover, among the m workers, a fraction of $\delta$ workers are Byzantine, which may behave abnormally and send arbitrary values to the server due to accidental failure or malicious attacks. The other workers, which are called non-Byzantine workers, will faithfully conduct the training algorithm without any fault. Formally, we use $G \subseteq \{1, 2, \ldots, m\}$ to denote the index set of non-Byzantine workers where $|\mathcal{G}| = (1 - \delta)m$ . The server has no access to any training data and does not know which workers are Byzantine. In this work, we mainly consider the loss functions that satisfy the following three assumptions, which are quite common in distributed learning. For simplicity, we use the notation $\|\cdot\|$ to denote the Euclidean norm of a vector.

Assumption 1 (Bounded variance). There exists $\sigma \geq 0$ , such that $\mathbb{E}_{\xi \sim \mathcal{D}} \| \nabla f(\mathbf{w}, \xi) - \nabla F(\mathbf{w})\|^2 \leq \sigma^2$ for all $\mathbf{w} \in \mathbb{R}^d$ .

Assumption 2 (Lower bound of $F(\cdot)$ ). There exists $F^{*} \in \mathbb{R}$ such that $F(\mathbf{w}) \geq F^{*}$ for all $\mathbf{w} \in \mathbb{R}^{d}$ .

Assumption 3 (L-smoothness). The loss function $F(\cdot)$ is differentiable everywhere on $\mathbb{R}^d$ . Moreover, $\| \nabla F(\mathbf{w}) - \nabla F(\mathbf{w}') \| \leq L \| \mathbf{w} - \mathbf{w}' \|$ for all $\mathbf{w}, \mathbf{w}' \in \mathbb{R}^d$ .

A typical and widely-used algorithm to solve the optimization problem (1) with potential Byzantine workers is Byzantine-robust stochastic gradient descent with momentum (ByzSGDm) (Farhadkhani et al., 2022; Karimireddy et al., 2021). Compared with vanilla stochastic gradient descent with momentum (SGDm), the main difference in ByzSGDm is that the mean aggregator on the server is substituted by a robust aggregator. Specifically, in ByzSGDm, the server updates the model parameter at the t-th iteration by computing

$$
\mathbf {w} _ {t + 1} = \mathbf {w} _ {t} - \eta_ {t} \cdot \mathbf {A g g} (\mathbf {u} _ {t} ^ {(1)}, \dots , \mathbf {u} _ {t} ^ {(m)}),
$$

where $\eta_{t}$ is the learning rate and $\mathbf{Agg}(\cdot)$ is a robust aggregator. Local momentum $\mathbf{u}_{t}^{(k)}$ is received from the k-th worker ( $k = 1, 2, \ldots, m$ ). For each non-Byzantine worker $k \in G$ ,

$$
\mathbf {u} _ {t} ^ {(k)} = \left\{ \begin{array}{l l} \mathbf {g} _ {0} ^ {(k)}, & t = 0; \\ \beta \mathbf {u} _ {t - 1} ^ {(k)} + (1 - \beta) \mathbf {g} _ {t} ^ {(k)}, & t > 0, \end{array} \right.
$$

where $\beta$ is the momentum hyper-parameter and $\mathbf{g}_t^{(k)} = \frac{1}{B}\sum_{b = 1}^{B}\nabla f(\mathbf{w}_t,\xi_t^{(k,b)})$ is the mean value of a mini-batch of stochastic gradients with size $B$ . For each Byzantine worker $k\in [m]\setminus \mathcal{G}$ , $\mathbf{u}_t^{(k)}$ can be an arbitrary value. For space saving, more details about ByzSGDm are moved to Algorithm 2 and Algorithm 3 in Appendix A.

For a ‘good’ aggregator, the aggregated result $\mathbf{Agg}(\mathbf{u}_{t}^{(1)},\ldots,\mathbf{u}_{t}^{(m)})$ should be close to the true mean of the momentums on non-Byzantine workers, which can be written as $\frac{1}{|\mathcal{G}|}\sum_{k\in\mathcal{G}}\mathbf{u}_{t}^{(k)}$ . To quantitatively measure a robust aggregator, the definition of $(\delta_{\max},c)$ -robust aggregator has been proposed in existing works (Karimireddy et al., 2021), which we present in Definition 1 below.

Definition 1 (( $\delta_{max}$ , c)-robust aggregator (Karimireddy et al., 2021)). Let $0 \leq \delta_{max} < \frac{1}{2}$ and $c \geq 0$ . Random vectors $x_{1}, \ldots, x_{m} \in R^{d}$ satisfy that $E\|x_{k} - x_{k'}\|^2 \leq \rho^2$ for all fixed k, $k' \in G$ , where $G \subseteq \{1, \ldots, m\}$ and $|\mathcal{G}| = (1 - \delta)m$ . An aggregator $\mathbf{Agg}(\cdot)$ is called a $(\delta_{\max}, c)$ -robust aggregator if we always have that

$$
\mathbb {E} \| \mathbf {e} \| ^ {2} \leq c \delta \rho^ {2},
$$

when $\delta \leq \delta_{\mathrm{max}}$ . Here, $\mathbf{e} = \mathbf{A}\mathbf{g}\mathbf{g}(\mathbf{x}_1,\dots ,\mathbf{x}_m) - \frac{1}{|\mathcal{G}|}\sum_{k\in \mathcal{G}}\mathbf{x}_k$ is called the aggregation error.

In addition, it has been proved that for any potential robust aggregator, there is inevitably an aggregation error of $\Omega(\delta\rho^{2})$ in the worst case (Karimireddy et al., 2021). It has also been proved that some existing aggregators such as centered clipping (Karimireddy et al., 2021) satisfy Definition 1.

# 3 METHODOLOGY

# 3.1 EFFECT OF BATCH SIZE ON CONVERGENCE

As shown in existing works on Byzantine-robust distributed learning (Blanchard et al., 2017; Chen et al., 2017; Li et al., 2019; Yin et al., 2018), even if robust aggregators have been used, there is typically a drop on model accuracy under Byzantine attacks due to the aggregation error. Therefore, we attempt to alleviate the drop on model accuracy by reducing the aggregation error.

According to Definition 1, there are three variables related to the upper bound of aggregation error. The fraction of Byzantine workers $\delta$ is determined by the problem, which can hardly be reduced. The constant $c$ is mainly related to the specific robust aggregator. There have been many works (Blanchard et al., 2017; Chen et al., 2017; Karimireddy et al., 2021; Li et al., 2019; Yin et al., 2018) that propose various robust aggregators. In this work, we mainly attempt to reduce $\rho$ . Moreover, we focus on the i.i.d. setting in this work. Since $\mathbb{E}[\mathbf{x}_k] = \mathbb{E}[\mathbf{x}_{k'}]$ in this case, according to Assumption 1, we have

$$
\mathbb {E} \| \mathbf {x} _ {k} - \mathbf {x} _ {k ^ {\prime}} \| ^ {2} = \mathbb {E} \| (\mathbf {x} _ {k} - \mathbb {E} [ \mathbf {x} _ {k} ]) - (\mathbf {x} _ {k ^ {\prime}} - \mathbb {E} [ \mathbf {x} _ {k ^ {\prime}} ]) \| ^ {2} = \mathbb {E} \| \mathbf {x} _ {k} - \mathbb {E} [ \mathbf {x} _ {k} ] \| ^ {2} + \mathbb {E} \| \mathbf {x} _ {k ^ {\prime}} - \mathbb {E} [ \mathbf {x} _ {k ^ {\prime}} ] \| ^ {2} \leq 2 \sigma^ {2}, \tag {2}
$$

which implies that $\rho^{2} \leq 2\sigma^{2}$ in i.i.d. cases under Assumption 1. Therefore, we can reduce $\rho$ by reducing the variance $\sigma^{2}$ in i.i.d. cases. A simple but effective way to reduce the variance is increasing the batch size on each worker, which is denoted by B in this paper. For simplicity, we assume that all workers adopt the same batch size in this work. Compared to the case with batch size 1, the variance of stochastic gradients will be reduced to 1/B of the original if the batch size is set to B. However, to make the total number of gradient computation unchanged, the total iteration number will be reduced to 1/B of the original, leading to fewer times of model updating. Formally, we use $\mathcal{C} = TBm(1 - \delta)$ to denote the total number of gradient computation on non-Byzantine workers, where T is the total iteration number. Thus, we have $T = \frac{\mathcal{C}}{Bm(1 - \delta)}$ . It implies that a larger batch size B will lead to a smaller total iteration number T when the total number of gradient computation C is fixed. In many BRDL applications with deep learning models, C can be used to approximately evaluate the computation cost since the computation cost of robust aggregation and model updating is negligible compared to that of gradient computation.

We first recall the convergence of ByzSGDm, which has been adequately studied in existing works (Karimireddy et al., 2021). We restate the convergence results of ByzSGDm in Theorem 1 below. For space saving, the details of how Theorem 1 is obtained from existing results (Karimireddy et al., 2021) are presented in Appendix B.

Theorem 1 (Convergence of ByzSGDm (Karimireddy et al., 2021)). Suppose that $F(\mathbf{w}_0) - F^* \leq F_0$ . Under Assumptions 1, 2 and 3, when $\mathbf{Agg}(\cdot)$ is $(\delta_{\max}, c)$ -robust and $\delta \leq \delta_{\max}$ , setting $\eta_t = \eta = \min \left( \sqrt{\frac{F_0 + \frac{5c\delta\sigma^2}{16BL}}{\frac{20LT\sigma^2}{B}\left(\frac{2}{m} + c\delta\right)}}, \frac{1}{8L} \right)$ and $1 - \beta = 8L\eta$ , we have the following result for ByzSGDm:

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| ^ {2} \leq 1 6 \sqrt {\frac {\sigma^ {2} (1 + c \delta m)}{T B m}} \left(\sqrt {1 0 L F _ {0}} + \sqrt {\frac {3 c \delta \sigma^ {2}}{B}}\right) + \frac {3 2 L F _ {0}}{T} + \frac {2 0 \sigma^ {2} (1 + c \delta m)}{T B m}. \tag {3}
$$

When $\mathcal{C}$ is fixed, inequality (3) can be re-written as $\frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E} \| \nabla F(\mathbf{w}_t) \|^2 \leq \mathcal{U}(B)$ since $T = \frac{\mathcal{C}}{Bm(1-\delta)}$ , where $\mathcal{U}(B)$ is a real-valued function with respect to batch size $B$ . Specifically,

$$
\begin{array}{l} \mathcal {U} (B) = 1 6 \sqrt {\frac {\sigma^ {2} (1 + c \delta m) (1 - \delta)}{\mathcal {C}}} \left(\sqrt {1 0 L F _ {0}} + \sqrt {\frac {3 c \delta \sigma^ {2}}{B}}\right) + \frac {3 2 L F _ {0} B m (1 - \delta)}{\mathcal {C}} \\ + \frac {2 0 \sigma^ {2} (1 + c \delta m) (1 - \delta)}{\mathcal {C}}. \\ \end{array}
$$

Please note that $\mathcal{U}(B)$ is originally defined on the set of positive integers $N^{*}$ since B denotes the batch size. We here extend the definition of $\mathcal{U}(B)$ to $B \in (0, +\infty)$ for simplicity. The results will be interpreted back to $B \in N^{*}$ at the end of our analysis. Then we attempt to find the optimal batch size $B^{*}$ that minimizes the theoretical upper bound $\mathcal{U}(B)$ when $\mathcal{C} = TBm(1 - \delta)$ is fixed. Formally, $B^{*}$ is defined by the following optimization problem:

$$
B ^ {*} = \operatorname * {a r g   m i n} _ {B \in (0, + \infty)} \mathcal {U} (B).
$$

We present Proposition 1 below, which provides an explicit expression of $B^{*}$ . Please refer to Appendix B for the proof details.

Proposition 1. $\mathcal{U}(B)$ is strictly convex on $(0, +\infty)$ . Moreover, when $\delta > 0$ , we have

$$
B ^ {*} = \left(\frac {3}{1 6 L ^ {2} (F _ {0}) ^ {2} m}\right) ^ {\frac {1}{3}} \left(\frac {c \delta (1 + c \delta m)}{m (1 - \delta)}\right) ^ {\frac {1}{3}} \sigma^ {\frac {4}{3}} \mathcal {C} ^ {\frac {1}{3}}, \tag {4}
$$

and

$$
\begin{array}{l} \mathcal {U} (B ^ {*}) = \frac {1 6 \sqrt {1 0 L F _ {0} (1 + c \delta m) (1 - \delta)} \sigma}{\mathcal {C} ^ {\frac {1}{2}}} + \frac {2 4 \left[ 1 2 c \delta (1 + c \delta m) (1 - \delta) ^ {2} L F _ {0} m \right] ^ {\frac {1}{3}} \sigma^ {\frac {4}{3}}}{\mathcal {C} ^ {\frac {2}{3}}} \\ + \frac {2 0 (1 + c \delta m) (1 - \delta) \sigma^ {2}}{\mathcal {C}}. \\ \end{array}
$$

Algorithm 1 Byzantine-Robust SGD with Normalized Momentum (ByzSGDnm)   
Input: initial model parameter $w_{0}$ , worker number m, iteration number T, learning rates $\{\eta_{t}\}_{t=0}^{T-1}$ , batch size B, momentum hyper-parameter $\beta\in[0,1)$ , robust aggregator $\mathbf{Agg}(\cdot)$ ;
for t=0 to T-1 do
    Broadcast $w_{t}$ to all workers;
    on worker $k\in\{1,\ldots,m\}$ in parallel do
    Receive $w_{t}$ from the server;
    Independently draw B samples $\xi_{t}^{(k,1)},\ldots,\xi_{t}^{(k,B)}$ from distribution D;
    Compute $\mathbf{g}_{t}^{(k)}=\frac{1}{B}\sum_{b=1}^{B}\nabla f(\mathbf{w}_{t},\xi_{t}^{(k,b)})$ ;
    Update local momentum $\mathbf{u}_{t}^{(k)}=\left\{\begin{aligned}&\mathbf{g}_{0}^{(k)},&t=0;\\&\beta\mathbf{u}_{t-1}^{(k)}+(1-\beta)\mathbf{g}_{t}^{(k)},&t>0;\end{aligned}\right.$ Send $\mathbf{u}_{t}^{(k)}$ to the server (Byzantine workers may send arbitrary values at this step);
    end on worker
    Receive $\{\mathbf{u}_{t}^{(k)}\}_{k=1}^{m}$ from the m workers, and compute $\mathbf{u}_{t}=\mathbf{Agg}(\mathbf{u}_{t}^{(1)},\ldots,\mathbf{u}_{t}^{(m)})$ ;
    Update model parameter with normalized momentum: $w_{t+1}=w_{t}-\eta_{t}\frac{u_{t}}{\|u_{t}\|}$ ;
end for
Output model parameter $w_{T}$ .

Please note that $\mathcal{U}(B)$ has no more than one global minimizer due to the strict convexity. Thus, $B^{*}$ is well-defined when $\delta > 0$ . Furthermore, the term $\left(\frac{c\delta(1 + c\delta m)}{m(1 - \delta)}\right)^{\frac{1}{3}}$ in (4) is monotonically increasing with respect to $\delta$ . It implies that when the total number of gradient computation on non-Byzantine workers $\mathcal{C} = TBm(1 - \delta)$ is fixed, $B^{*}$ will increase as the fraction of Byzantine workers $\delta$ increases. Then, we interpret the above results back to $B \in \mathbb{N}^{*}$ . Due to the strict convexity, $\mathcal{U}(B)$ is monotonically decreasing when $B \in (0, B^{*})$ and monotonically increasing when $B \in (B^{*}, +\infty)$ . Thus, the optimal integer batch size that minimizes $\mathcal{U}(B)$ equals either $\lfloor B^{*} \rfloor$ or $\lfloor B^{*} \rfloor + 1$ , which also increases with $\delta$ . The notation $\lfloor B^{*} \rfloor$ represents the largest integer that is not larger than $B^{*}$ . In addition, the conclusion will be further supported by the empirical results in Section 5.

Meanwhile, although $B^{*} \rightarrow 0$ as $\delta \rightarrow 0^{+}$ , it should not be interpreted as recommending a batch size that is close to 0 when there is no attack. In fact, since C is fixed, a too-small batch size B implies a too-large iteration number T, which will lead to a large communication cost. Moreover, the computation power of some devices (e.g., GPUs) will not be effectively utilized when B is too small. Thus, the setting of B is a trade-off between model accuracy and running time when $\delta = 0$ , which has been studied for years (Goyal et al., 2017; You et al., 2020; Zhao et al., 2020; 2023). In addition, please note that although a smaller $\mathcal{U}(B)$ does not necessarily ensure a better empirical performance given the complexity and variety in real-world applications, it provides a better worst-case guarantee.

# 3.2 BYZANTINE-ROBUST SGD WITH NORMALIZED MOMENTUM

Proposition 1 shows that the optimal batch size $B^{*}$ that minimizes $\mathcal{U}(B)$ increases with the fraction of Byzantine workers. Hence, a relatively large batch size is preferred when under Byzantine attacks. In existing works on traditional large-batch training without attacks (Goyal et al., 2017; Hoffer et al., 2017; Keskar et al., 2017; Zhao et al., 2020; 2023), the normalization technique is widely used to increase model accuracy. Motivated by this, we propose a novel method called Byzantine-robust stochastic gradient descent with normalized momentum (ByzSGDnm), by introducing a simple normalization operation. Specifically, in ByzSGDnm, the model parameters are updated by:

$$
\mathbf {w} _ {t + 1} = \mathbf {w} _ {t} - \eta_ {t} \cdot \frac {\mathbf {A g g} (\mathbf {u} _ {t} ^ {(1)} , \ldots , \mathbf {u} _ {t} ^ {(m)})}{\| \mathbf {A g g} (\mathbf {u} _ {t} ^ {(1)} , \ldots , \mathbf {u} _ {t} ^ {(m)}) \|}.
$$

The details of ByzSGDnm are illustrated in Algorithm 1. Please note that there are some existing methods using layer-wise normalization (You et al., 2020). However, these methods might suffer from degradation of model accuracy without additional training tricks such as warm-up (Zhao et al., 2023). Furthermore, as shown in (Zhao et al., 2023), the layer-wise (block-wise) normalization might slow down the convergence rate. Hence, we follow the way of performing normalization on the whole momentum (Zhao et al., 2020; 2023; Cutkosky & Mehta, 2020).

Moreover, please note that the purpose of traditional large-batch training (Goyal et al., 2017; You et al., 2020; Zhao et al., 2020; 2023) is mainly to accelerate the training process by reducing communication cost and utilizing the computation power more effectively. However, in this work, the main purpose of increasing batch size and using momentum normalization is to enhance the Byzantine robustness and increase the model accuracy under Byzantine attacks. The acceleration effect of adopting large batch size is viewed as a bonus in this work. Please refer to Section 5 for the empirical results about the wall-clock time of ByzSGDm and ByzSGDnm with different batch size.

# 4 CONVERGENCE

In this section, we theoretically analyze the convergence of ByzSGDnm under Assumptions 1, 2 and 3. The assumptions are common in distributed learning. For space saving, we only present the main results here. Please refer to Appendix B for the proof details.

Theorem 2. Suppose that $F(\mathbf{w}_0) - F^* \leq F_0$ and let $\alpha = 1 - \beta$ . Under Assumptions 1, 2 and 3, when $\mathbf{Agg}(\cdot)$ is $(\delta_{\max}, c)$ -robust, $\delta \leq \delta_{\max}$ and $\eta_t = \eta$ , we have the following result for ByzSGDnm:

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq \frac {2 F _ {0}}{\eta T} + \frac {1 0 \eta L}{\alpha} + \frac {9 \sqrt {2 c m \delta (1 - \delta)} + 9}{\sqrt {B m (1 - \delta)}} \left(\frac {1}{\alpha T} + \sqrt {\alpha}\right) \sigma .
$$

Finally, we show that when the learning rate $\eta$ and the momentum hyper-parameter $\beta = 1 - \alpha$ are properly set, ByzSGDnm can achieve the convergence order of $O\left(\frac{1}{T^{\frac{1}{4}}}\right)$ by Proposition 2 below.

Proposition 2. Under Assumptions 1, 2 and 3, when $\mathbf{Agg}(\cdot)$ is $(\delta_{\max}, c)$ -robust and $\delta \leq \delta_{\max}$ , setting $1 - \beta = \alpha = \min\left(\frac{\sqrt{80LF_0Bm(1 - \delta)}}{\left[9\sqrt{2cm\delta(1 - \delta)} + 9\right]\sigma\sqrt{T}}, 1\right)$ and $\eta_t = \eta = \sqrt{\frac{\alpha F_0}{5LT}}$ , we have that

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq 6 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {1}{2}} \left(\frac {5 L F _ {0} \sigma^ {2}}{T B m (1 - \delta)}\right) ^ {\frac {1}{4}} + 1 2 \sqrt {\frac {5 L F _ {0}}{T}} \\ + \frac {2 7 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{2}}}{4 \sqrt {5 T B ^ {2} m ^ {2} (1 - \delta) ^ {2} L F _ {0}}} \sigma^ {2}. \tag {5} \\ \end{array}
$$

Moreover, when $\mathcal{C} = TBm(1 - \delta)$ is fixed, the optimal batch size $\tilde{B}^*$ that minimizes the right-hand side of (5) is $\tilde{B}^* = \frac{9\left[\sqrt{2cm\delta(1 - \delta)} + 1\right]^{\frac{3}{2}}\sigma^2}{80m(1 - \delta)LF_0}$ . In this case ( $B = \tilde{B}^*$ ), we have:

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq \frac {6 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {1}{2}} \left(5 L F _ {0} \sigma^ {2}\right) ^ {\frac {1}{4}}}{\mathcal {C} ^ {\frac {1}{4}}} + \frac {1 8 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{4}} \sigma}{\mathcal {C} ^ {\frac {1}{2}}}.
$$

Inequality (5) illustrates that after T iterations, ByzSGDnm can guarantee that

$$
\min _ {t = 0, \dots , T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq O \left(\frac {(L F _ {0}) ^ {\frac {1}{4}} \sqrt {\sigma}}{T ^ {\frac {1}{4}}} + \frac {1}{T ^ {\frac {1}{2}}}\right).
$$

Therefore, ByzSGDnm has the same convergence order as vanilla SGD with normalized momentum (Cutkosky & Mehta, 2020) without attacks. The extra factor $\left[\sqrt{2cm\delta(1-\delta)}+1\right]^{\frac{1}{2}}/(1-\delta)^{\frac{1}{4}}$ in the right-hand side (RHS) of (5) is due to the existence of Byzantine workers and increases with $\delta$ . The extra factor vanishes (equals 1) when there is no Byzantine worker ( $\delta=0$ ). Moreover, it has been shown in existing works (Arjevani et al., 2023; Cutkosky & Mehta, 2020) that under Assumptions 1, 2 and 3, the convergence order $O(1/T^{\frac{1}{4}})$ is optimal for SGD. Byz-VR-MARINA (Gorbunov et al., 2023) achieves a better convergence order by intermittently using full gradients. However, full gradients are computationally expensive, especially in real-world applications with a large number of training instances. We detailedly compare ByzSGDnm with Byz-VR-MARINA in Appendix D. The empirical results show that ByzSGDnm significantly outperforms Byz-VR-MARINA. In addition, $\tilde{B}^{*}$ also increases with $\delta$ since both $\delta(1-\delta)$ and $\frac{1}{1-\delta}$ increase with $\delta$ when $\delta\in[0,\frac{1}{2})$ .

The analysis in this paper is based on the definition of $(\delta_{\mathrm{max}}, c)$ -robust aggregator (Definition 1). There is also another criterion of robust aggregators in existing works called the $(f, \kappa)$ -robustness (Al- louah et al., 2023). Similar results can also be obtained under the $(f, \kappa)$ -robustness. Please refer to Appendix C for more details.

# 5 EXPERIMENT

Task and platform. In this section, we will empirically test the performance of ByzSGDm and ByzSGDnm on image classification tasks. Each algorithm will be used to train a ResNet-20 (He et al., 2016) deep learning model on CIFAR-10 dataset (Krizhevsky et al., 2009). All the experiments presented in this work are conducted on a distributed platform with 9 dockers. Each docker is bound to an NVIDIA TITAN Xp GPU. One docker is chosen as the server while the other 8 dockers are chosen as workers. The training instances are randomly and equally distributed to the workers.

Experimental settings. In existing works (Allouah et al., 2023; Karimireddy et al., 2021; 2022) on BRDL, the batch size is typically set to 32 or 50 on the CIFAR-10 dataset. Therefore, We set ByzSGDm (Karimireddy et al., 2021) with batch size 32 as the baseline, and compare the performance of ByzSGDm with different batch size (ranging from 64 to 1024) to the baseline under ALIE attack (Baruch et al., 2019). In our experiments, we use four widely-used robust aggregators Krum (KR) (Blanchard et al., 2017), geometric median (GM) (Chen et al., 2017), coordinate-wise median (CM) (Yin et al., 2018) and centered clipping (CC) (Karimireddy et al., 2021) for ByzSGDm. Moreover, we set the clipping radius to 0.1 for CC. We train the model for 160 epochs with cosine annealing learning rates (Loshchilov & Hutter, 2017). Specifically, the learning rate at the i-th epoch will be $\eta_{i} = \frac{\eta_{0}}{2}(1 + \cos(\frac{i}{160}\pi))$ for $i = 0, 1, \ldots, 159$ . The initial learning rate $\eta_{0}$ is selected from $\{0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0\}$ , and the best final top-1 test accuracy is used as the final metrics. The momentum hyper-parameter $\beta$ is set to 0.9. Please note that the total number of gradient computation on non-Byzantine workers C is independent of batch size. Specifically, $C = 160 \times 50000 \times (1 - \delta)$ since we train the model for 160 epochs with 50000 training instances.

Evaluation on the effect of batch size. We first evaluate the performance of ByzSGDm with different batch size when the fraction of Byzantine workers $\delta$ is 0 (no attack), $\frac{1}{8}$ and $\frac{3}{8}$ , respectively. As the results in Table 1 and Table 2 show, the batch size corresponding to the best top-1 accuracy increases with $\delta$ , which is consistent with our theoretical results. Moreover, when $\delta = \frac{3}{8}$ , using a relatively large batch size greatly increases the test accuracy. Meanwhile, the test accuracy decreases with the batch size when there is no attack ( $\delta = 0$ ), which is consistent with existing works (Goyal et al., 2017; Hoffer et al., 2017; Keskar et al., 2017; You et al., 2020; Zhao et al., 2020; 2023).

Table 1: The final top-1 test accuracy of ByzSGDm with various batch size under ALIE attack when Krum (KM) and geometric median (GM) are used as the robust aggregator 

<table><tr><td rowspan="2">Batch size</td><td colspan="3">ByzSGDm with KR</td></tr><tr><td> $\delta = 0$ </td><td> $\delta = 1/8$ </td><td> $\delta = 3/8$ </td></tr><tr><td>32×8 (baseline)</td><td>91.08%</td><td>55.84%</td><td>38.55%</td></tr><tr><td>64×8</td><td>89.98% (-1.10%)</td><td>63.22% (+7.38%)</td><td>54.15% (+15.60%)</td></tr><tr><td>128×8</td><td>89.71% (-1.37%)</td><td>75.06% (+19.22%)</td><td>55.98% (+17.43%)</td></tr><tr><td>256×8</td><td>89.15% (-1.93%)</td><td>84.47% (+28.63%)</td><td>59.28% (+20.73%)</td></tr><tr><td>512×8</td><td>86.15% (-4.93%)</td><td>85.68% (+29.84%)</td><td>83.42% (+44.87%)</td></tr><tr><td>1024×8</td><td>84.97% (-6.11%)</td><td>83.48% (+27.64%)</td><td>83.45% (+44.90%)</td></tr><tr><td rowspan="2">Batch size</td><td colspan="3">ByzSGDm with GM</td></tr><tr><td> $\delta = 0$ </td><td> $\delta = 1/8$ </td><td> $\delta = 3/8$ </td></tr><tr><td>32×8 (baseline)</td><td>92.02%</td><td>83.81%</td><td>63.11%</td></tr><tr><td>64×8</td><td>91.50% (-0.52%)</td><td>87.92% (+4.11%)</td><td>70.88% (+7.77%)</td></tr><tr><td>128×8</td><td>90.85% (-1.17%)</td><td>89.68% (+5.87%)</td><td>82.08% (+18.97%)</td></tr><tr><td>256×8</td><td>89.26% (-2.76%)</td><td>87.99% (+4.18%)</td><td>87.62% (+24.51%)</td></tr><tr><td>512×8</td><td>88.21% (-3.81%)</td><td>87.70% (+3.89%)</td><td>86.95% (+23.84%)</td></tr><tr><td>1024×8</td><td>86.52% (-5.50%)</td><td>85.94% (+2.13%)</td><td>84.75% (+21.64%)</td></tr></table>

Table 2: The final top-1 test accuracy of ByzSGDm with various batch size under ALIE attack when coordinate-wise median (CM) and centered clipping (CC) are used as the robust aggregator 

<table><tr><td rowspan="2">Batch size</td><td colspan="3">ByzSGDm with CM</td></tr><tr><td> $\delta = 0$ </td><td> $\delta = 1/8$ </td><td> $\delta = 3/8$ </td></tr><tr><td>32×8 (baseline)</td><td>92.30%</td><td>86.46%</td><td>33.11%</td></tr><tr><td>64×8</td><td>91.79% (-0.51%)</td><td>88.09% (+1.63%)</td><td>55.66% (+22.55%)</td></tr><tr><td>128×8</td><td>90.43% (-1.87%)</td><td>89.16% (+2.70%)</td><td>66.38% (+33.27%)</td></tr><tr><td>256×8</td><td>89.84% (-2.46%)</td><td>88.60% (+2.14%)</td><td>82.47% (+49.36%)</td></tr><tr><td>512×8</td><td>87.27% (-5.03%)</td><td>87.20% (+0.74%)</td><td>83.25% (+50.14%)</td></tr><tr><td>1024×8</td><td>84.06% (-8.24%)</td><td>83.71% (-2.75%)</td><td>80.94% (+47.83%)</td></tr><tr><td rowspan="2">Batch size</td><td colspan="3">ByzSGDm with CC</td></tr><tr><td> $\delta = 0$ </td><td> $\delta = 1/8$ </td><td> $\delta = 3/8$ </td></tr><tr><td>32×8 (baseline)</td><td>92.52%</td><td>86.55%</td><td>72.83%</td></tr><tr><td>64×8</td><td>91.74% (-0.78%)</td><td>88.59% (+2.04%)</td><td>79.45% (+6.62%)</td></tr><tr><td>128×8</td><td>90.63% (-1.89%)</td><td>88.94% (+2.39%)</td><td>84.94% (+12.11%)</td></tr><tr><td>256×8</td><td>89.40% (-3.12%)</td><td>88.46% (+1.91%)</td><td>87.25% (+14.42%)</td></tr><tr><td>512×8</td><td>88.78% (-3.74%)</td><td>88.29% (+1.74%)</td><td>87.46% (+14.63%)</td></tr><tr><td>1024×8</td><td>85.50% (-7.02%)</td><td>84.88% (-1.67%)</td><td>83.70% (+10.87%)</td></tr></table>

Table 3: The final top-1 test accuracy when there are 3 Byzantine workers under ALIE attack 

<table><tr><td>Method</td><td>with KR</td><td>with GM</td><td>with CM</td><td>with CC</td></tr><tr><td>ByzSGDm, batch size = 32 × 8 (baseline)</td><td>38.55%</td><td>63.11%</td><td>33.11%</td><td>72.83%</td></tr><tr><td>ByzSGDnm, batch size = 32 × 8</td><td>43.47%</td><td>69.45%</td><td>61.28%</td><td>78.50%</td></tr><tr><td>ByzSGDm, batch size = 512 × 8</td><td>83.42%</td><td>86.95%</td><td>83.25%</td><td>87.46%</td></tr><tr><td>ByzSGDnm, batch size = 512 × 8</td><td>85.12%</td><td>89.13%</td><td>86.03%</td><td>88.53%</td></tr></table>

Table 4: The final top-1 test accuracy when there are 3 Byzantine workers under FoE attack 

<table><tr><td>Method</td><td>with KR</td><td>with GM</td><td>with CM</td><td>with CC</td></tr><tr><td>ByzSGDm, batch size = 32 × 8 (baseline)</td><td>10.00%</td><td>78.36%</td><td>83.97%</td><td>83.60%</td></tr><tr><td>ByzSGDnm, batch size = 32 × 8</td><td>10.00%</td><td>88.55%</td><td>84.12%</td><td>88.99%</td></tr><tr><td>ByzSGDm, batch size = 512 × 8</td><td>10.00%</td><td>84.09%</td><td>79.16%</td><td>86.24%</td></tr><tr><td>ByzSGDnm, batch size = 512 × 8</td><td>10.00%</td><td>89.12%</td><td>84.65%</td><td>89.32%</td></tr></table>

Effectiveness of large batch size and momentum normalization. In this paper, we propose to use (i) a relatively large batch size and (ii) the momentum normalization technique. Here, we empirically evaluate the effectiveness of these two improvements. Specifically, we will compare the performance of the following four methods: (a) ByzSGDm with batch size $32 \times 8$ (baseline), (b) ByzSGDnm with batch size $32 \times 8$ , (c) ByzSGDm with batch size $512 \times 8$ , and (d) ByzSGDnm with batch size $512 \times 8$ . The performances of the methods are compared when there are 3 workers under ALIE attack (Baruch et al., 2019) and FoE attack (Xie et al., 2020), respectively. As presented in Table 3 and Table 4, among the four methods, ByzSGDnm with batch size $512 \times 8$ has the best top-1 test accuracy except for the case of using aggregator KR under FoE attack. All the methods fail when using KR under FoE attack mainly because the KR aggregator is not robust against FoE attack, as shown in existing works (Karimireddy et al., 2021; Xie et al., 2020). The empirical results of ByzSGDm and ByzSGDnm with more different batch size are deferred to Appendix E for space saving. In addition, we also compare the performance of ByzSGDm and ByzSGDnm under no attack (or failure) and under bit-flipping failure (Xie et al., 2019), respectively. ByzSGDnm has a comparable performance to ByzSGDm in these two cases. Please refer to Appendix E for the detailed results.

More evaluation when NNM is used. We also compare the empirical performance of different methods when the nearest neighbour mixing (NNM) (Allouah et al., 2023) technique is used. NNM is originally proposed to enhance the robustness of aggregators in general non-i.i.d. cases but can

Table 5: The final top-1 test accuracy when there are 3 Byzantine workers under ALIE attack and the nearest neighbour mixing (NNM) technique is used 

<table><tr><td>Method</td><td>with KR</td><td>with GM</td><td>with CM</td><td>with CC</td></tr><tr><td>ByzSGDm, batch size = 32 × 8 (baseline)</td><td>58.61%</td><td>72.58%</td><td>71.51%</td><td>76.48%</td></tr><tr><td>ByzSGDnm, batch size = 32 × 8</td><td>80.41%</td><td>79.50%</td><td>79.81%</td><td>79.91%</td></tr><tr><td>ByzSGDm, batch size = 512 × 8</td><td>85.26%</td><td>85.37%</td><td>86.95%</td><td>85.98%</td></tr><tr><td>ByzSGDnm, batch size = 512 × 8</td><td>87.68%</td><td>88.09%</td><td>87.69%</td><td>87.59%</td></tr></table>

Table 6: The wall-clock time of ByzSGDm and ByzSGDnm for 160 epochs (in second) 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td></tr><tr><td>ByzSGDm</td><td>2007.39s</td><td>985.52s(×2.04 faster)</td><td>522.27s(×3.84 faster)</td><td>366.98s(×5.47 faster)</td><td>314.80s(×6.38 faster)</td></tr><tr><td>ByzSGDnm</td><td>1985.78s</td><td>978.50s(×2.03 faster)</td><td>515.46s(×3.85 faster)</td><td>376.70s(×5.27 faster)</td><td>327.62s(×6.06 faster)</td></tr></table>

also be used in i.i.d. cases. As the results in Table 5 show, ByzSGDnm with batch size $512 \times 8$ still has the best final top-1 test accuracy under ALIE attacks when NNM is used. In addition, we find it interesting that when combined with NNM, the performance of KR and CM is improved, but the performance of GM and CC is degraded. Since NNM is originally proposed for general non-i.i.d. cases, it requires further study to understand this behavior of NNM in i.i.d. cases. However, since we mainly focus on the effect of batch size, it is beyond the scope of this work.

The bonus of training acceleration. Existing works (Goyal et al., 2017; Hoffer et al., 2017; Keskar et al., 2017; You et al., 2020; Zhao et al., 2020; 2023) have shown that increasing batch size can accelerate the training process by reducing the communication cost and utilizing the computing power of GPUs more effectively. We present the wall-clock time for 160 epochs when using CC as the robust aggregator under no attack in Table 6. Please note that whether there are attacks or not has almost no effect on the computation cost of non-Byzantine workers. For both ByzSGDm and ByzSGDnm, the running time decreases as the batch size increases. It verifies that increasing batch size has the bonus of training acceleration. In addition, ByzSGDnm has a comparable running time to ByzSGDm, which shows that the computation cost of the momentum normalization is negligible.

Comparison with Byz-VR-MARINA. Byz-VR-MARINA (Gorbunov et al., 2023) is originally proposed for non-i.i.d. cases, but can also be used in i.i.d. cases. Therefore, we also empirically compare ByzSGDnm with Byz-VR-MARINA. Empirical results show that ByzSGDnm significantly outperforms Byz-VR-MARINA in i.i.d. cases. Detailed results are deferred to Appendix D.

Although we mainly study the effect of batch size in BRDL for i.i.d. cases in this work, we also provide some empirical results under non-i.i.d. settings in Appendix E.2. The empirical results show that ByzSGDnm still outperforms existing methods in the non-i.i.d. setting. However, further work is required to detailedly discover the effect of batch size and the behavior of ByzSGDnm in non-i.i.d. cases, which we will study in the future.

# 6 CONCLUSION

In this paper, we theoretically show that when the total number of gradient computation is fixed, the optimal batch size corresponding to the tightest theoretical upper bound in BRDL increases with the fraction of Byzantine workers. The theoretical results indicate that a relatively large batch size is preferred when there are Byzantine attacks. Furthermore, we propose a novel method called ByzSGDnm and prove the convergence of ByzSGDnm. Empirical results show that when under Byzantine attacks, setting a relatively large batch size can significantly increase the model accuracy compared to the case of small batch size, which is consistent with our theoretical results. Moreover, ByzSGDnm can achieve higher model accuracy than existing BRDL methods when under attack. In addition, increasing batch size has the bonus of training acceleration, which is verified by the empirical results.

# REPRODUCIBILITY STATEMENT

For the theoretical results of our work, the assumptions are presented in Section 2 and the detailed proofs are deferred to Appendix B. For the empirical results, the experimental platform and the hyper-parameter settings are described in Section 5. The core code for our experiments can be found in the supplementary material.

# ACKNOWLEDGMENTS

This work is supported by National Key R&D Program of China (No. 2020YFA0713900), NSFC Project (No. 12326615, No. 62192783), and Fundamental Research Funds for the Central Universities (No. 020214380108).

# REFERENCES

Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. QSGD: Communication-efficient SGD via gradient quantization and encoding. In Advances in Neural Information Processing Systems, pp. 1709–1720, 2017.   
Zeyuan Allen-Zhu, Faeze Ebrahimian, Jerry Li, and Dan Alistarh. Byzantine-resilient non-convex stochastic gradient descent. arXiv preprint arXiv:2012.14368, 2020.   
Youssef Allouah, Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafaël Pinot, and John Stephan. Fixing by mixing: A recipe for optimal byzantine ml under heterogeneity. In Proceedings of the International Conference on Artificial Intelligence and Statistics, pp. 1232–1300. PMLR, 2023.   
Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. Lower bounds for non-convex stochastic optimization. Mathematical Programming, 199(1-2):165–214, 2023.   
Gilad Baruch, Moran Baruch, and Yoav Goldberg. A little is enough: Circumventing defenses for distributed learning. In Advances in Neural Information Processing Systems, pp. 8635–8645, 2019.   
Jeremy Bernstein, Jiawei Zhao, Kamyar Azizzadenesheli, and Anima Anandkumar. signSGD with majority vote is communication efficient and fault tolerant. In Proceedings of the International Conference on Learning Representations, 2019.   
Peva Blanchard, Rachid Guerraoui, Julien Stainer, et al. Machine learning with adversaries: Byzantine tolerant gradient descent. In Advances in Neural Information Processing Systems, pp. 119–129, 2017.   
Saikiran Bulusu, Prashant Khanduri, Swatantra Kafle, Pranay Sharma, and Pramod K Varshney. Byzantine resilient non-convex scsg with distributed batch gradient computations. IEEE Transactions on Signal and Information Processing over Networks, 7:754–766, 2021.   
Lingjiao Chen, Hongyi Wang, Zachary Charles, and Dimitris Papailiopoulos. Draco: Byzantine-resilient distributed training via redundant gradients. In Proceedings of the International Conference on Machine Learning, pp. 903–912, 2018.   
Yudong Chen, Lili Su, and Jiaming Xu. Distributed statistical machine learning in adversarial settings: Byzantine gradient descent. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 1(2):1–25, 2017.   
Ashok Cutkosky and Harsh Mehta. Momentum improves normalized sgd. In Proceedings of the International Conference on Machine Learning, pp. 2260–2268, 2020.   
Georgios Damaskinos, Rachid Guerraoui, Rhicheek Patra, Mahsa Taziki, et al. Asynchronous Byzantine machine learning (the case of SGD). In Proceedings of the International Conference on Machine Learning, pp. 1145–1154, 2018.   
Aaron Defazio and Léon Bottou. On the ineffectiveness of variance reduced optimization for deep learning. Advances in Neural Information Processing Systems, 32, 2019.

Ilias Diakonikolas and Daniel M Kane. Recent advances in algorithmic high-dimensional robust statistics. arXiv preprint arXiv:1911.05911, 2019.   
Ilias Diakonikolas, Gautam Kamath, Daniel M Kane, Jerry Li, Ankur Moitra, and Alistair Stewart. Being robust (in high dimensions) can be practical. In Proceedings of the International Conference on Machine Learning, pp. 999–1008, 2017.   
El-Mahdi El-Mhamdi, Rachid Guerraoui, and Sébastien Rouault. Distributed momentum for byzantine-resilient stochastic gradient descent. In Proceedings of the International Conference on Learning Representations, 2021.   
Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, and John Stephan. Byzantine machine learning made easy by resilient averaging of momentums. In Proceedings of the International Conference on Machine Learning, pp. 6246–6283, 2022.   
Eduard Gorbunov, Samuel Horváth, Peter Richtárik, and Gauthier Gidel. Variance reduction is an antidote to Byzantines: Better rates, weaker assumptions and communication compression as a cherry on the top. In Proceedings of the International Conference on Learning Representations, 2023.   
Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.   
Farzin Haddadpour, Mohammad Mahdi Kamani, Mehrdad Mahdavi, and Viveck Cadambe. Trading redundancy for communication: Speeding up distributed SGD for non-convex optimization. In Proceedings of the International Conference on Machine Learning, pp. 2545–2554, 2019.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.   
Elad Hoffer, Itay Hubara, and Daniel Soudry. Train longer, generalize better: closing the generalization gap in large batch training of neural networks. In Advances in Neural Information Processing Systems, volume 30, 2017.   
Kevin Hsieh, Amar Phanishayee, Onur Mutlu, and Phillip Gibbons. The non-iid data quagmire of decentralized machine learning. In Proceedings of the International Conference on Machine Learning, pp. 4387–4398, 2020.   
Martin Jaggi, Virginia Smith, Martin Takác, Jonathan Terhorst, Sanjay Krishnan, Thomas Hofmann, and Michael I Jordan. Communication-efficient distributed dual coordinate ascent. In Advances in Neural Information Processing Systems, pp. 3068–3076, 2014.   
Rie Johnson and Tong Zhang. Accelerating stochastic gradient descent using predictive variance reduction. In Advances in Neural Information Processing Systems, pp. 315–323, 2013.   
Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and Trends® in Machine Learning, 14(1–2):1–210, 2021.   
Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Learning from history for Byzantine robust optimization. In Proceedings of the International Conference on Machine Learning, pp. 5311–5319, 2021.   
Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Byzantine-robust learning on heterogeneous datasets via bucketing. In Proceedings of the International Conference on Learning Representations, 2022.   
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In Proceedings of the International Conference on Learning Representations, 2017.

Konstantinos Konstantinidis and Aditya Ramamoorthy. Byzshield: An efficient and robust system for distributed training. Proceedings of Machine Learning and Systems, 3:812–828, 2021.   
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, 2009.   
Leslie Lamport, Robert Shostak, and Marshall Pease. The Byzantine generals problem. In Concurrency: the works of leslie lamport, pp. 203–226, 2019.   
Jason D Lee, Qihang Lin, Tengyu Ma, and Tianbao Yang. Distributed stochastic variance reduced gradient methods by sampling extra data with replacement. Journal of Machine Learning Research, 18(1):4404–4446, 2017.   
Liping Li, Wei Xu, Tianyi Chen, Georgios B Giannakis, and Qing Ling. RSA: Byzantine-robust stochastic aggregation methods for distributed learning from heterogeneous datasets. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 1544–1551, 2019.   
Zhize Li, Hongyan Bao, Xiangliang Zhang, and Peter Richtárik. Page: A simple and optimal probabilistic gradient estimator for nonconvex optimization. In Proceedings of the International Conference on Machine Learning, pp. 6286–6295, 2021.   
Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and Ji Liu. Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent. In Advances in Neural Information Processing Systems, pp. 5330–5340, 2017.   
Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In Proceedings of the International Conference on Learning Representations, 2017.   
Chenxin Ma, Virginia Smith, Martin Jaggi, Michael Jordan, Peter Richtárik, and Martin Takác. Adding vs. averaging in distributed primal-dual optimization. In Proceedings of the International Conference on Machine Learning, pp. 1973–1982, 2015.   
Brendan McMahan and Daniel Ramage. Federated learning: Collaborative machine learning without centralized training data. Google Research Blog, 3, 2017.   
Shashank Rajput, Hongyi Wang, Zachary Charles, and Dimitris Papailiopoulos. Detox: A redundancy-based framework for faster and more robust gradient aggregation. In Advances in Neural Information Processing Systems, pp. 10320–10330, 2019.   
Ohad Shamir, Nati Srebro, and Tong Zhang. Communication-efficient distributed optimization using an approximate newton-type method. In Proceedings of the International Conference on Machine Learning, pp. 1000–1008, 2014.   
Weisong Shi, Jie Cao, Quan Zhang, Youhuizi Li, and Lanyu Xu. Edge computing: Vision and challenges. IEEE Internet of Things Journal, 3(5):637–646, 2016.   
Jy-yong Sohn, Dong-Jun Han, Beongjun Choi, and Jaekyun Moon. Election coding for distributed learning: Protecting signsgd against Byzantine attacks. In Advances in Neural Information Processing Systems, pp. 14615–14625, 2020.   
Shizhao Sun, Wei Chen, Jiang Bian, Xiaoguang Liu, and Tie-Yan Liu. Slim-dp: a multi-agent system for communication-efficient distributed deep learning. In Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems, pp. 721–729, 2018.   
Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European Conference on Computer Vision, pp. 3–19, 2018.   
Zhaoxian Wu, Qing Ling, Tianyi Chen, and Georgios B Giannakis. Federated variance-reduced stochastic gradient descent with robustness to Byzantine attacks. IEEE Transactions on Signal Processing, 68:4583–4596, 2020.   
Cong Xie, Sanmi Koyejo, and Indranil Gupta. Zeno: Distributed stochastic gradient descent with suspicion-based fault-tolerance. In Proceedings of the International Conference on Machine Learning, pp. 6893–6901, 2019.

Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. Fall of empires: Breaking Byzantine-tolerant sgd by inner product manipulation. In Proceedings of the Conference on Uncertainty in Artificial Intelligence, pp. 261–270, 2020.   
Tianbao Yang. Trading computation for communication: Distributed stochastic dual coordinate ascent. In Advances in Neural Information Processing Systems, pp. 629–637, 2013.   
Yi-Rui Yang and Wu-Jun Li. BASGD: Buffered asynchronous SGD for Byzantine learning. In Proceedings of the International Conference on Machine Learning, pp. 11751–11761, 2021.   
Yi-Rui Yang and Wu-Jun Li. Buffered asynchronous SGD for Byzantine learning. Journal of Machine Learning Research, 24(204):1–62, 2023.   
Zhixiong Yang, Arpita Gang, and Waheed U Bajwa. Adversary-resilient distributed and decentralized statistical inference and machine learning: An overview of recent advances under the Byzantine threat model. IEEE Signal Processing Magazine, 37(3):146–159, 2020.   
Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Byzantine-robust distributed learning: Towards optimal statistical rates. In Proceedings of the International Conference on Machine Learning, pp. 5650–5659, 2018.   
Dong Yin, Yudong Chen, Ramchandran Kannan, and Peter Bartlett. Defending against saddle point attack in Byzantine-robust distributed learning. In Proceedings of the International Conference on Machine Learning, pp. 7074–7084, 2019.   
Yang You, Jing Li, Sashank J. Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training BERT in 76 minutes. In Proceedings of the International Conference on Learning Representations, 2020.   
Hao Yu, Rong Jin, and Sen Yang. On the linear speedup analysis of communication efficient momentum SGD for distributed non-convex optimization. In Proceedings of the International Conference on Machine Learning, pp. 7184–7193, 2019a.   
Hao Yu, Sen Yang, and Shenghuo Zhu. Parallel restarted SGD with faster convergence and less communication: Demystifying why model averaging works for deep learning. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 5693–5700, 2019b.   
Shen-Yi Zhao, Ru Xiang, Ying-Hao Shi, Peng Gao, and Wu-Jun Li. SCOPE: scalable composite optimization for learning on spark. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 2928–2934, 2017.   
Shen-Yi Zhao, Gong-Duo Zhang, Ming-Wei Li, and Wu-Jun Li. Proximal SCOPE for distributed sparse learning. In Advances in Neural Information Processing Systems, pp. 6551–6560, 2018.   
Shen-Yi Zhao, Yin-Peng Xie, and Wu-Jun Li. Stochastic normalized gradient descent with momentum for large batch training. arXiv preprint arXiv:2007.13985, 2020.   
Shen-Yi Zhao, Chang-Wei Shi, Yin-Peng Xie, and Wu-Jun Li. Stochastic normalized gradient descent with momentum for large-batch training. Science China Information Sciences, 2023.   
Yi Zhou, Yingbin Liang, Yaoliang Yu, Wei Dai, and Eric P Xing. Distributed proximal gradient algorithm for partially asynchronous computer clusters. Journal of Machine Learning Research, 19(1):733–764, 2018.   
Martin Zinkevich, Markus Weimer, Lihong Li, and Alex J Smola. Parallelized stochastic gradient descent. In Advances in Neural Information Processing Systems, pp. 2595–2603, 2010.

# A BYZANTINE-ROBUST SGDM

The detailed algorithm of Byzantine-robust SGDm (ByzSGDm) on the server and workers are presented in Algorithm 2 and Algorithm 3, respectively.

Algorithm 2 ByzSGDm (Server)   
Input: worker number m, iteration number T, learning rates $\{\eta_{t}\}_{t=0}^{T-1}$ , robust aggregator $\mathbf{Agg}(\cdot)$ ;
Initialization: model parameter $w_{0}$ ;
Broadcast $w_{0}$ to all workers;
for t = 0 to T - 1 do
    Receive $\{\mathbf{u}_{t}^{(k)}\}_{k=1}^{m}$ from all workers, and compute $\mathbf{u}_{t} = \mathbf{Agg}(\mathbf{u}_{t}^{(1)}, \ldots, \mathbf{u}_{t}^{(m)})$ ;
    Update model parameter $w_{t+1} = w_{t} - \eta_{t} u_{t}$ ;
    Broadcast $w_{t+1}$ to all workers;
end for
Output model parameter $w_{T}$ .

Algorithm 3 ByzSGDm (Worker\_k)   
Input: iteration number $T$ , batch size $B$ , momentum hyper-parameter $\beta \in [0,1)$ ;  
Receive initial model parameter $\mathbf{w}_0$ from the server;  
for $t = 0$ to $T - 1$ do  
Independently draw $B$ samples $\xi_t^{(k,1)}, \ldots, \xi_t^{(k,B)}$ from distribution $\mathcal{D}$ ;  
Compute $\mathbf{g}_t^{(k)} = \frac{1}{B} \sum_{b=1}^{B} \nabla f(\mathbf{w}_t, \xi_t^{(k,b)})$ ;  
Update local momentum $\mathbf{u}_t^{(k)} = \begin{cases} \mathbf{g}_0^{(k)}, & t = 0; \\ \beta \mathbf{u}_{t-1}^{(k)} + (1 - \beta) \mathbf{g}_t^{(k)}, & t > 0; \end{cases}$ Send $\mathbf{u}_t^{(k)}$ to the server (Byzantine workers may send arbitrary values at this step);  
Receive the latest model parameter $\mathbf{w}_{t+1}$ from the server;  
end for

# B PROOF DETAILS

# B.1 PROOF OF THEOREM 1

Proof. It has been proved in (Karimireddy et al., 2021) that for ByzSGDm, we have

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| ^ {2} \leq 1 6 \sqrt {\frac {\tilde {\sigma} ^ {2} (1 + c \delta m)}{T m} \left(1 0 L F _ {0} + 3 c \delta \tilde {\sigma} ^ {2}\right)} + \frac {3 2 L F _ {0}}{T} + \frac {2 0 \tilde {\sigma} ^ {2} (1 + c \delta m)}{T m}, \tag {6}
$$

where $\tilde{\sigma}^2$ is the variance of stochastic gradients. In the setting of this work, we have $\tilde{\sigma}^2 = \sigma^2 / B$ where $B$ is the batch size on each worker. Therefore, we have

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| ^ {2} \leq 1 6 \sqrt {\frac {\sigma^ {2} (1 + c \delta m)}{T B m} \left(1 0 L F _ {0} + \frac {3 c \delta \sigma^ {2}}{B}\right)} + \frac {3 2 L F _ {0}}{T} + \frac {2 0 \sigma^ {2} (1 + c \delta m)}{T B m}. \tag {7}
$$

Moreover, since for all $x, y > 0$ ,

$$
\frac {1}{2} \sqrt {x} + \frac {1}{2} \sqrt {y} \leq \sqrt {x + y} = \sqrt {(\sqrt {x}) ^ {2} + (\sqrt {y}) ^ {2}} \leq \sqrt {x} + \sqrt {y}, \tag {8}
$$

we have that

$$
\frac {1}{2} \leq \frac {\sqrt {x + y}}{\sqrt {x} + \sqrt {y}} \leq 1. \tag {9}
$$

Therefore, we can replace the term $\sqrt{10LF_0 + \frac{3c\delta\sigma^2}{B}}$ with $\left(\sqrt{10LF_0} + \sqrt{\frac{3c\delta\sigma^2}{B}}\right)$ without changing the convergence order. Consequently, we have

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| ^ {2} \leq 1 6 \sqrt {\frac {\sigma^ {2} (1 + c \delta m)}{T B m}} \left(\sqrt {1 0 L F _ {0}} + \sqrt {\frac {3 c \delta \sigma^ {2}}{B}}\right) + \frac {3 2 L F _ {0}}{T} + \frac {2 0 \sigma^ {2} (1 + c \delta m)}{T B m}. \tag {10}
$$

![](images/ebce42501cf15c21ef313297eadc91131f216b0ac066bcb0d94d9061cc323ce8.jpg)

# B.2 PROOF OF PROPOSITION 1

Proof. Since

$$
\begin{array}{l} \mathcal {U} (B) = 1 6 \sqrt {\frac {\sigma^ {2} (1 + c \delta m) (1 - \delta)}{\mathcal {C}}} \left(\sqrt {1 0 L F _ {0}} + \sqrt {\frac {3 c \delta \sigma^ {2}}{B}}\right) + \frac {3 2 L F _ {0} B m (1 - \delta)}{\mathcal {C}} \\ + \frac {2 0 \sigma^ {2} (1 + c \delta m) (1 - \delta)}{\mathcal {C}}, \tag {11} \\ \end{array}
$$

it is not hard to find that $\mathcal{U}(B)$ is continuous and differentiable on $(0, +\infty)$ . Then we analyze the convexity of $\mathcal{U}(B)$ by showing that it has a positive second-order derivative. For simplicity, we define the constants $A_1, A_2$ and $A_3$ as follows:

$$
A _ {1} = 1 6 \sqrt {\frac {1 0 L F _ {0} (1 + c \delta m) (1 - \delta) \sigma^ {2}}{\mathcal {C}}} + \frac {2 0 (1 + c \delta m) (1 - \delta) \sigma^ {2}}{\mathcal {C}}, \tag {12}
$$

$$
A _ {2} = 1 6 \sqrt {\frac {3 c \delta (1 + c \delta m) (1 - \delta) \sigma^ {4}}{\mathcal {C}}}, \tag {13}
$$

$$
A _ {3} = \frac {3 2 L F _ {0} m (1 - \delta)}{\mathcal {C}}. \tag {14}
$$

According to the definition of $\mathcal{U}(B)$ , we have

$$
\mathcal {U} (B) = A _ {1} + A _ {2} B ^ {- \frac {1}{2}} + A _ {3} B, \quad B \in (0, + \infty). \tag {15}
$$

Therefore,

$$
\mathcal {U} ^ {\prime} (B) = - \frac {1}{2} A _ {2} B ^ {- \frac {3}{2}} + A _ {3}, \quad B \in (0, + \infty). \tag {16}
$$

Thus,

$$
\mathcal {U} ^ {\prime \prime} (B) = \frac {3}{4} A _ {2} B ^ {- \frac {5}{2}} > 0, \quad B \in (0, + \infty). \tag {17}
$$

which implies that $\mathcal{U}(B)$ is strictly convex. According to (16), the equation $\mathcal{U}'(B) = 0$ has the only solution

$$
B ^ {*} = \left(\frac {A _ {2}}{2 A _ {3}}\right) ^ {\frac {2}{3}} = \left(\frac {1 6 \sqrt {\frac {3 c \delta (1 + c \delta m) (1 - \delta) \sigma^ {4}}{\mathcal {C}}}}{2 \cdot \frac {3 2 L F _ {0} m (1 - \delta)}{\mathcal {C}}}\right) ^ {\frac {2}{3}} = \left(\frac {\sqrt {3 c \delta (1 + c \delta m) \sigma^ {4} \mathcal {C}}}{4 L F _ {0} m \sqrt {1 - \delta}}\right) ^ {\frac {2}{3}} \tag {18}
$$

$$
= \left(\frac {3}{1 6 L ^ {2} (F _ {0}) ^ {2} m}\right) ^ {\frac {1}{3}} \left(\frac {c \delta (1 + c \delta m)}{m (1 - \delta)}\right) ^ {\frac {1}{3}} \sigma^ {\frac {4}{3}} \mathcal {C} ^ {\frac {1}{3}}. \tag {19}
$$

Thus, $B^{*}$ is the only global minimizer of $\mathcal{U}(B)$ . The minimum value $\mathcal{U}(B^{*})$ is:

$$
\mathcal {U} (B ^ {*}) = \mathcal {U} \left(\left(\frac {A _ {2}}{2 A _ {3}}\right) ^ {\frac {2}{3}}\right) \tag {20}
$$

$$
= A _ {1} + 3 \left(\frac {(A _ {2}) ^ {2} A _ {3}}{4}\right) ^ {\frac {1}{3}} \tag {21}
$$

$$
\begin{array}{l} = \frac {1 6 \sqrt {1 0 L F _ {0} (1 + c \delta m) (1 - \delta)} \sigma}{\mathcal {C} ^ {\frac {1}{2}}} + \frac {2 0 (1 + c \delta m) (1 - \delta) \sigma^ {2}}{\mathcal {C}} \\ + 3 \left(\frac {\left(1 6 \sqrt {\frac {3 c \delta (1 + c \delta m) (1 - \delta) \sigma^ {4}}{\mathcal {C}}}\right) ^ {2} \left(\frac {3 2 L F _ {0} m (1 - \delta)}{\mathcal {C}}\right)}{4}\right) ^ {\frac {1}{3}} (22) \\ = \frac {1 6 \sqrt {1 0 L F _ {0} (1 + c \delta m) (1 - \delta)} \sigma}{\mathcal {C} ^ {\frac {1}{2}}} + \frac {2 0 (1 + c \delta m) (1 - \delta) \sigma^ {2}}{\mathcal {C}} \\ + \frac {2 4 \left[ 1 2 c \delta (1 + c \delta m) (1 - \delta) ^ {2} L F _ {0} m \right] ^ {\frac {1}{3}} \sigma^ {\frac {4}{3}}}{\mathcal {C} ^ {\frac {2}{3}}}. (23) \\ \end{array}
$$

![](images/6786aa88ebe80885bff5c3c9c1f1fba9323f19ce7b6e95bd784d25aa35710ebd.jpg)

# B.3 PROOF OF THEOREM 2

Firstly, we present Lemma 1 below, which quantitatively shows that the variance of stochastic gradients can be reduced by increasing the batch size B.

Lemma 1 (Mini-batch variance reduction). Under Assumption 1, we have that $\forall k\in \mathcal{G}$ ,

$$
\mathbb {E} \left\| \mathbf {g} _ {t} ^ {(k)} - \nabla F (\mathbf {w} _ {t}) \right\| ^ {2} \leq \frac {\sigma^ {2}}{B}, \quad \forall t \geq 0. \tag {24}
$$

Proof. By the definition of $\mathbf{g}_t^{(k)}$ ,

$$
\mathbf {g} _ {t} ^ {(k)} = \frac {1}{B} \sum_ {b = 1} ^ {B} \nabla f (\mathbf {w} _ {t}, \xi_ {t} ^ {(k, b)}). \tag {25}
$$

Since for all $k \in \mathcal{G}$ , $\mathbb{E}[\nabla f(\mathbf{w}_t, \xi_t^{(k,b)})] = \nabla F(\mathbf{w}_t)$ and $\nabla f(\mathbf{w}_t, \xi_t^{(k,b)})$ is independent to each other with bounded variance $\sigma^2$ , we have that

$$
\mathbb {E} \left\| \mathbf {g} _ {t} ^ {(k)} - \nabla F (\mathbf {w} _ {t}) \right\| ^ {2} \leq \frac {\sigma^ {2}}{B}. \tag {26}
$$

Moreover, by using Cauchy-Schwarz inequality, it is obtained that

$$
\mathbb {E} \left\| \mathbf {g} _ {t} ^ {(k)} - \nabla F (\mathbf {w} _ {t}) \right\| \leq \sqrt {\mathbb {E} \left\| \mathbf {g} _ {t} ^ {(k)} - \nabla F (\mathbf {w} _ {t}) \right\| ^ {2}} \leq \frac {\sigma}{\sqrt {B}}. \tag {27}
$$

![](images/09c69d3c6c66e67090d835b9b51d00475f7e9bc8fbf3cae25ff7cae1dac28cfd.jpg)

We define

$$
\bar {\mathbf {u}} _ {t} = \frac {1}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \mathbf {u} _ {t} ^ {(k)}, \tag {28}
$$

which represents the exact averaging local momentum of all non-faulty workers $k \in G$ . Then we provide an upper bound for the aggregation error of a $(\delta_{\max}, c)$ -robust aggregator in Lemma 2. The notation of $\bar{u}_{t}$ is only used for theoretical analysis, which does not appear in the algorithm.

The proof of Lemma 2 is inspired by the existing work (Karimireddy et al., 2021). Moreover, we have improved the analysis details, and the upper bound of aggregation error in Lemma 2 is tighter than that in (Karimireddy et al., 2021).

Lemma 2 (Aggregation bias). Let $\alpha = 1 - \beta$ . Under Assumption 1, when $\mathbf{Agg}(\cdot)$ is $(\delta_{\max}, c)$ -robust and $\delta \leq \delta_{\max}$ , we have that

$$
\mathbb {E} \| \mathbf {u} _ {t} - \bar {\mathbf {u}} _ {t} \| ^ {2} \leq \frac {2 c \delta \sigma^ {2}}{B} [ \alpha + (1 - \alpha) ^ {2 t} ], \quad \forall t \geq 0. \tag {29}
$$

Proof. Let $\alpha = 1 - \beta$ . For all $t \geq 0$ and fixed $k, k' \in \mathcal{G}$ , since $\mathbf{g}_t^{(k)}$ is independent to $\mathbf{g}_t^{(k')}$ and $\mathbf{u}_{t-1}^{(k)}$ ,

$$
\mathbb {E} \left\| \mathbf {u} _ {t} ^ {(k)} - \mathbf {u} _ {t} ^ {(k ^ {\prime})} \right\| ^ {2} = \mathbb {E} \left\| (1 - \alpha) [ \mathbf {u} _ {t - 1} ^ {(k)} - \mathbf {u} _ {t - 1} ^ {(k ^ {\prime})} ] + \alpha [ \mathbf {g} _ {t} ^ {(k)} - \mathbf {g} _ {t} ^ {(k ^ {\prime})} ] \right\| ^ {2} \tag {30}
$$

$$
= \mathbb {E} \left\| (1 - \alpha) [ \mathbf {u} _ {t - 1} ^ {(k)} - \mathbf {u} _ {t - 1} ^ {(k ^ {\prime})} ] \right\| ^ {2} + \mathbb {E} \left\| \alpha [ \mathbf {g} _ {t} ^ {(k)} - \mathbf {g} _ {t} ^ {(k ^ {\prime})} ] \right\| ^ {2} \tag {31}
$$

$$
= (1 - \alpha) ^ {2} \mathbb {E} \left\| \mathbf {u} _ {t - 1} ^ {(k)} - \mathbf {u} _ {t - 1} ^ {(k ^ {\prime})} \right\| ^ {2} + 2 \alpha^ {2} \mathbb {E} \left\| \mathbf {g} _ {t} ^ {(k)} - \nabla F (\mathbf {w} _ {t}) \right\| ^ {2} \tag {32}
$$

$$
\leq (1 - \alpha) ^ {2} \mathbb {E} \left\| \mathbf {u} _ {t - 1} ^ {(k)} - \mathbf {u} _ {t - 1} ^ {(k ^ {\prime})} \right\| ^ {2} + \frac {2 \alpha^ {2} \sigma^ {2}}{B}. \tag {33}
$$

Recursively using the inequality above, we have that

$$
\mathbb {E} \left\| \mathbf {u} _ {t} ^ {(k)} - \mathbf {u} _ {t} ^ {(k ^ {\prime})} \right\| ^ {2} \leq (1 - \alpha) ^ {2 t} \mathbb {E} \left\| \mathbf {u} _ {0} ^ {(k)} - \mathbf {u} _ {0} ^ {(k ^ {\prime})} \right\| ^ {2} + \frac {2 \alpha^ {2} \sigma^ {2}}{B} [ 1 + (1 - \alpha) ^ {2} + \dots + (1 - \alpha) ^ {2 (t - 1)} ] \tag {34}
$$

$$
= (1 - \alpha) ^ {2 t} \mathbb {E} \left\| \mathbf {g} _ {0} ^ {(k)} - \mathbf {g} _ {0} ^ {(k ^ {\prime})} \right\| ^ {2} + \frac {2 \alpha^ {2} \sigma^ {2}}{B} \cdot \frac {1 - (1 - \alpha) ^ {2 t}}{1 - (1 - \alpha) ^ {2}} \tag {35}
$$

$$
\leq (1 - \alpha) ^ {2 t} \cdot \frac {2 \sigma^ {2}}{B} + \frac {2 \alpha^ {2} \sigma^ {2}}{B} \cdot \frac {1}{\alpha (2 - \alpha)} \tag {36}
$$

$$
\leq (1 - \alpha) ^ {2 t} \cdot \frac {2 \sigma^ {2}}{B} + \frac {2 \alpha \sigma^ {2}}{B} \tag {37}
$$

$$
= \frac {2 \sigma^ {2}}{B} [ \alpha + (1 - \alpha) ^ {2 t} ]. \tag {38}
$$

According to the definition of $(\delta_{\max}, c)$ -robust aggregator,

$$
\mathbb {E} \left\| \mathbf {A g g} (\mathbf {u} _ {t} ^ {(1)}, \dots , \mathbf {u} _ {t} ^ {(m)}) - \frac {1}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \mathbf {u} _ {t} ^ {(k)} \right\| ^ {2} \leq \frac {2 c \delta \sigma^ {2}}{B} [ \alpha + (1 - \alpha) ^ {2 t} ]. \tag {39}
$$

Namely,

$$
\mathbb {E} \| \mathbf {u} _ {t} - \bar {\mathbf {u}} _ {t} \| ^ {2} \leq \frac {2 c \delta \sigma^ {2}}{B} [ \alpha + (1 - \alpha) ^ {2 t} ]. \tag {40}
$$

By using Cauchy-Schwarz inequality, it is obtained that

$$
\mathbb {E} \| \mathbf {u} _ {t} - \bar {\mathbf {u}} _ {t} \| \leq \sqrt {\mathbb {E} \| \mathbf {u} _ {t} - \bar {\mathbf {u}} _ {t} \| ^ {2}} \leq \frac {\sqrt {2 c \delta [ \alpha + (1 - \alpha) ^ {2 t} ]} \sigma}{\sqrt {B}}. \tag {41}
$$

![](images/c1ad3a8201d337d5f510dc1c693c61d77f0c1190accf0e3bdcbbea7e91231766.jpg)

Based on Lemma 2, we can further obtain Lemma 3, which provides an upper bound for the difference between the aggregated momentum $u_{t}$ and the global gradient $\nabla F(\mathbf{w}_{t})$ .

Lemma 3. Under Assumptions 1 and 3, when $\mathbf{Agg}(\cdot)$ is $(\delta_{\max}, c)$ -robust, $\delta \leq \delta_{\max}$ and $\eta_t = \eta$ , we have the following result for ByzSGDnm:

$$
\mathbb {E} \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \| \leq \frac {\eta L}{\alpha} + \frac {\sqrt {2 c m \delta (1 - \delta)} + 1}{\sqrt {B m (1 - \delta)}} [ (1 - \alpha) ^ {t} + \sqrt {\alpha} ] \sigma , \quad \forall t \geq 0. \tag {42}
$$

Proof. $\forall t\geq 0$ , we have that

$$
\bar {\mathbf {u}} _ {t} - \nabla F (\mathbf {w} _ {t}) = \left(\frac {1}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \mathbf {u} _ {t} ^ {(k)}\right) - \nabla F (\mathbf {w} _ {t}) \tag {43}
$$

$$
= \frac {1}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \left[ \beta \mathbf {u} _ {t - 1} ^ {(k)} + (1 - \beta) \mathbf {g} _ {t} ^ {(k)} \right] - \nabla F (\mathbf {w} _ {t}) \tag {44}
$$

$$
= \beta \bar {\mathbf {u}} _ {t - 1} + \left(\frac {1 - \beta}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \mathbf {g} _ {t} ^ {(k)}\right) - \nabla F (\mathbf {w} _ {t}) \tag {45}
$$

$$
= \beta [ \bar {\mathbf {u}} _ {t - 1} - \nabla F (\mathbf {w} _ {t}) ] + \frac {1 - \beta}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \left[ \mathbf {g} _ {t} ^ {(k)} - \nabla F (\mathbf {w} _ {t}) \right] \tag {46}
$$

$$
= \beta \left[ \bar {\mathbf {u}} _ {t - 1} - \nabla F \left(\mathbf {w} _ {t - 1}\right) \right] + \beta \left[ \nabla F \left(\mathbf {w} _ {t}\right) - \nabla F \left(\mathbf {w} _ {t - 1}\right) \right] + \frac {1 - \beta}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \left[ \mathbf {g} _ {t} ^ {(k)} - \nabla F \left(\mathbf {w} _ {t}\right) \right]. \tag {47}
$$

Recursively using the equation above and substituting $\beta$ with $1 - \alpha$ , we have that

$$
\bar {\mathbf {u}} _ {t} - \nabla F (\mathbf {w} _ {t}) = (1 - \alpha) ^ {t} [ \bar {\mathbf {u}} _ {0} - \nabla F (\mathbf {w} _ {0}) ] + \sum_ {t ^ {\prime} = 1} ^ {t} (1 - \alpha) ^ {t - t ^ {\prime} + 1} [ \nabla F (\mathbf {w} _ {t ^ {\prime}}) - \nabla F (\mathbf {w} _ {t ^ {\prime} - 1}) ]
$$

$$
+ \sum_ {t ^ {\prime} = 1} ^ {t} (1 - \alpha) ^ {t - t ^ {\prime}} \left\{\frac {\alpha}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \left[ \mathbf {g} _ {t ^ {\prime}} ^ {(k)} - \nabla F (\mathbf {w} _ {t ^ {\prime}}) \right] \right\}. \tag {48}
$$

Noticing that $\mathbf{u}_0^{(k)} = \mathbf{g}_0^{(k)}$ , we have that $\bar{\mathbf{u}}_0 - \nabla F(\mathbf{w}_0) = \frac{1}{|\mathcal{G}|}\sum_{k\in \mathcal{G}}[\mathbf{g}_0^{(k)} - \nabla F(\mathbf{w}_0)]$ . Therefore,

$$
\bar {\mathbf {u}} _ {t} - \nabla F (\mathbf {w} _ {t}) = \sum_ {t ^ {\prime} = 1} ^ {t} (1 - \alpha) ^ {t - t ^ {\prime} + 1} [ \nabla F (\mathbf {w} _ {t ^ {\prime}}) - \nabla F (\mathbf {w} _ {t ^ {\prime} - 1}) ]
$$

$$
+ \frac {1}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \left\{(1 - \alpha) ^ {t} \left[ \mathbf {g} _ {0} ^ {(k)} - \nabla F (\mathbf {w} _ {0}) \right] + \alpha \sum_ {t ^ {\prime} = 1} ^ {t} (1 - \alpha) ^ {t - t ^ {\prime}} \left[ \mathbf {g} _ {t ^ {\prime}} ^ {(k)} - \nabla F (\mathbf {w} _ {t ^ {\prime}}) \right] \right\}. \tag {49}
$$

According to Assumption 3,

$$
\mathbb {E} \| \nabla F (\mathbf {w} _ {t ^ {\prime}}) - \nabla F (\mathbf {w} _ {t ^ {\prime} - 1}) \| \leq L \cdot \mathbb {E} \| \mathbf {w} _ {t ^ {\prime}} - \mathbf {w} _ {t ^ {\prime} - 1} \| = L \cdot \mathbb {E} \left\| - \eta \frac {\mathbf {u} _ {t ^ {\prime} - 1}}{\| \mathbf {u} _ {t ^ {\prime} - 1} \|} \right\| = \eta L. \tag {50}
$$

Since $\mathbf{g}_t^{(k)}$ 's are independent to each other, by using Lemma 1 and $|\mathcal{G}| \geq (1 - \delta)m$ , we have that

$$
\mathbb {E} \left\| \frac {1}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \left\{(1 - \alpha) ^ {t} [ \mathbf {g} _ {0} ^ {(k)} - \nabla F (\mathbf {w} _ {0}) ] + \alpha \sum_ {t ^ {\prime} = 1} ^ {t} (1 - \alpha) ^ {t - t ^ {\prime}} [ \mathbf {g} _ {t ^ {\prime}} ^ {(k)} - \nabla F (\mathbf {w} _ {t ^ {\prime}}) ] \right\} \right\|
$$

$$
\leq \left(\mathbb {E} \left\| \frac {1}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \left\{(1 - \alpha) ^ {t} [ \mathbf {g} _ {0} ^ {(k)} - \nabla F (\mathbf {w} _ {0}) ] + \alpha \sum_ {t ^ {\prime} = 1} ^ {t} (1 - \alpha) ^ {t - t ^ {\prime}} [ \mathbf {g} _ {t ^ {\prime}} ^ {(k)} - \nabla F (\mathbf {w} _ {t ^ {\prime}}) ] \right\} \right\| ^ {2}\right) ^ {\frac {1}{2}} \tag {51}
$$

$$
= \frac {1}{| \mathcal {G} |} \left\{\sum_ {k \in \mathcal {G}} \left[ (1 - \alpha) ^ {2 t} \mathbb {E} \left\| \mathbf {g} _ {0} ^ {(k)} - \nabla F (\mathbf {w} _ {0}) \right\| ^ {2} + \alpha^ {2} \sum_ {t ^ {\prime} = 1} ^ {t} (1 - \alpha) ^ {2 t - 2 t ^ {\prime}} \mathbb {E} \left\| \mathbf {g} _ {t ^ {\prime}} ^ {(k)} - \nabla F (\mathbf {w} _ {t ^ {\prime}}) \right\| ^ {2} \right] \right\} ^ {\frac {1}{2}} \tag {52}
$$

$$
\leq \frac {1}{| \mathcal {G} |} \left\{\frac {| \mathcal {G} | \sigma^ {2}}{B} \left[ (1 - \alpha) ^ {2 t} + \alpha^ {2} \sum_ {t ^ {\prime} = 1} ^ {t} (1 - \alpha) ^ {2 t - 2 t ^ {\prime}} \right] \right\} ^ {\frac {1}{2}} \tag {53}
$$

$$
= \frac {\sigma}{\sqrt {B | \mathcal {G} |}} \sqrt {(1 - \alpha) ^ {2 t} + \alpha^ {2} \frac {1 - (1 - \alpha) ^ {2 t}}{1 - (1 - \alpha) ^ {2}}} \tag {54}
$$

$$
\leq \frac {\sigma}{\sqrt {B | \mathcal {G} |}} \left[ (1 - \alpha) ^ {t} + \alpha \sqrt {\frac {1 - (1 - \alpha) ^ {2 t}}{1 - (1 - \alpha) ^ {2}}} \right] \tag {55}
$$

$$
\leq \frac {\sigma}{\sqrt {B | \mathcal {G} |}} \left[ (1 - \alpha) ^ {t} + \alpha \sqrt {\frac {1}{\alpha (2 - \alpha)}} \right] \tag {56}
$$

$$
= \frac {\sigma}{\sqrt {B | \mathcal {G} |}} \left[ (1 - \alpha) ^ {t} + \sqrt {\frac {\alpha}{2 - \alpha}} \right] \tag {57}
$$

$$
\leq \frac {\sigma}{\sqrt {B m (1 - \delta)}} \left[ (1 - \alpha) ^ {t} + \sqrt {\alpha} \right]. \tag {58}
$$

Consequently,

$$
\mathbb {E} \| \bar {\mathbf {u}} _ {t} - \nabla F (\mathbf {w} _ {t}) \| \leq \eta L \sum_ {t ^ {\prime} = 1} ^ {t} (1 - \alpha) ^ {t - t ^ {\prime} + 1} + \frac {\sigma}{\sqrt {B m (1 - \delta)}} [ (1 - \alpha) ^ {t} + \sqrt {\alpha} ] \tag {59}
$$

$$
\leq \frac {\eta L}{\alpha} + \frac {\sigma}{\sqrt {B m (1 - \delta)}} \left[ (1 - \alpha) ^ {t} + \sqrt {\alpha} \right]. \tag {60}
$$

Since $\mathbb{E}\| \mathbf{u}_t - \bar{\mathbf{u}}_t\| \leq \sqrt{\mathbb{E}\| \mathbf{u}_t - \bar{\mathbf{u}}_t\|^2}$ , according to Lemma 2, we have that

$$
\mathbb {E} \left\| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \right\| \leq \mathbb {E} \left\| \bar {\mathbf {u}} _ {t} - \nabla F (\mathbf {w} _ {t}) \right\| + \mathbb {E} \left\| \mathbf {u} _ {t} - \bar {\mathbf {u}} _ {t} \right\| \tag {61}
$$

$$
\leq \frac {\eta L}{\alpha} + \frac {\sigma}{\sqrt {B m (1 - \delta)}} \left[ (1 - \alpha) ^ {t} + \sqrt {\alpha} \right] + \frac {\sqrt {2 c \delta [ \alpha + (1 - \alpha) ^ {2 t} ]} \sigma}{\sqrt {B}} \tag {62}
$$

$$
\leq \frac {\eta L}{\alpha} + \frac {\sigma}{\sqrt {B m (1 - \delta)}} \left[ (1 - \alpha) ^ {t} + \sqrt {\alpha} \right] + \frac {\sqrt {2 c \delta} [ \sqrt {\alpha} + (1 - \alpha) ^ {t} ] \sigma}{\sqrt {B}} \tag {63}
$$

$$
= \frac {\eta L}{\alpha} + \frac {\sqrt {2 c m \delta (1 - \delta)} + 1}{\sqrt {B m (1 - \delta)}} [ (1 - \alpha) ^ {t} + \sqrt {\alpha} ] \sigma . \tag {64}
$$

![](images/2a661c0d7acfcf0780828870ce7da76611da11dcb36fd046916758272cc6b4b8.jpg)

Then we present the descent lemma for SGD with normalized momentum. The proof of Lemma 4 is inspired by the existing work (Cutkosky & Mehta, 2020), but the result in Lemma 4 is more general than that in (Cutkosky & Mehta, 2020).

Lemma 4 (Descent lemma). Under Assumptions 1 and 3, for any constant $\gamma \in (0,1)$ , we have the following result for ByzSGDnm:

$$
F (\mathbf {w} _ {t + 1}) \leq F (\mathbf {w} _ {t}) - \eta_ {t} \frac {1 - \gamma}{1 + \gamma} \| \nabla F (\mathbf {w} _ {t}) \| + \eta_ {t} \frac {2}{\gamma (1 + \gamma)} \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \| + \frac {\left(\eta_ {t}\right) ^ {2} L}{2}. \tag {65}
$$

Proof. According to Assumption 3, we have that

$$
F (\mathbf {w} _ {t + 1}) = F (\mathbf {w} _ {t} - \eta_ {t} \frac {\mathbf {u} _ {t}}{\| \mathbf {u} _ {t} \|}) \tag {66}
$$

$$
\leq F (\mathbf {w} _ {t}) - \nabla F (\mathbf {w} _ {t}) ^ {T} \left(\eta_ {t} \frac {\mathbf {u} _ {t}}{\| \mathbf {u} _ {t} \|}\right) + \frac {L}{2} \left\| \eta_ {t} \frac {\mathbf {u} _ {t}}{\| \mathbf {u} _ {t} \|} \right\| ^ {2} \tag {67}
$$

$$
= F (\mathbf {w} _ {t}) - \eta_ {t} \frac {\nabla F (\mathbf {w} _ {t}) ^ {T} \mathbf {u} _ {t}}{\| \mathbf {u} _ {t} \|} + \frac {(\eta_ {t}) ^ {2} L}{2}. \tag {68}
$$

Let $\gamma \in (0,1)$ be an arbitrary constant. Then we consider the following two cases:

(i) When $\| \mathbf{u}_t - \nabla F(\mathbf{w}_t)\| \leq \gamma \| \nabla F(\mathbf{w}_t)\|$ , we have

$$
- \eta_ {t} \frac {\nabla F (\mathbf {w} _ {t}) ^ {T} \mathbf {u} _ {t}}{\| \mathbf {u} _ {t} \|} = - \eta_ {t} \frac {\nabla F (\mathbf {w} _ {t}) ^ {T} [ \nabla F (\mathbf {w} _ {t}) + (\mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t})) ]}{\| \nabla F (\mathbf {w} _ {t}) + (\mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t})) \|} \tag {69}
$$

$$
\leq - \eta_ {t} \frac {\| \nabla F (\mathbf {w} _ {t}) \| ^ {2} - \| \nabla F (\mathbf {w} _ {t}) \| \cdot \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \|}{\| \nabla F (\mathbf {w} _ {t}) \| + \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \|} \tag {70}
$$

$$
\leq - \eta_ {t} \frac {\| \nabla F (\mathbf {w} _ {t}) \| ^ {2} - \gamma \| \nabla F (\mathbf {w} _ {t}) \| ^ {2}}{(1 + \gamma) \| \nabla F (\mathbf {w} _ {t}) \|} \tag {71}
$$

$$
= - \eta_ {t} \frac {1 - \gamma}{1 + \gamma} \| \nabla F (\mathbf {w} _ {t}) \| \tag {72}
$$

$$
\leq - \eta_ {t} \frac {1 - \gamma}{1 + \gamma} \| \nabla F (\mathbf {w} _ {t}) \| + \eta_ {t} \frac {2}{\gamma (1 + \gamma)} \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \|. \tag {73}
$$

(ii) When $\| \mathbf{u}_t - \nabla F(\mathbf{w}_t)\| >\gamma \| \nabla F(\mathbf{w}_t)\|$ , we have

$$
- \eta_ {t} \frac {\nabla F (\mathbf {w} _ {t}) ^ {T} \mathbf {u} _ {t}}{\| \mathbf {u} _ {t} \|} \leq \eta_ {t} \| \nabla F (\mathbf {w} _ {t}) \| \tag {74}
$$

$$
= - \eta_ {t} \frac {1 - \gamma}{1 + \gamma} \| \nabla F (\mathbf {w} _ {t}) \| + \eta_ {t} \frac {2}{1 + \gamma} \| \nabla F (\mathbf {w} _ {t}) \| \tag {75}
$$

$$
\leq - \eta_ {t} \frac {1 - \gamma}{1 + \gamma} \| \nabla F (\mathbf {w} _ {t}) \| + \eta_ {t} \frac {2}{\gamma (1 + \gamma)} \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \|. \tag {76}
$$

In summary, we always have

$$
- \eta_ {t} \frac {\nabla F (\mathbf {w} _ {t}) ^ {T} \mathbf {u} _ {t}}{\| \mathbf {u} _ {t} \|} \leq - \eta_ {t} \frac {1 - \gamma}{1 + \gamma} \| \nabla F (\mathbf {w} _ {t}) \| + \eta_ {t} \frac {2}{\gamma (1 + \gamma)} \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \|. \tag {77}
$$

Therefore,

$$
F (\mathbf {w} _ {t + 1}) \leq F (\mathbf {w} _ {t}) - \eta_ {t} \frac {1 - \gamma}{1 + \gamma} \| \nabla F (\mathbf {w} _ {t}) \| + \eta_ {t} \frac {2}{\gamma (1 + \gamma)} \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \| + \frac {\left(\eta_ {t}\right) ^ {2} L}{2}. \tag {78}
$$

![](images/4c11a44fffd720ffe2607063035502cf8525f5e67ce679b3ad6b1443f068b559.jpg)

Finally, we can obtain Theorem 2 by recursively using Lemma 4, taking expectation on both sides and using Lemma 3. The proof details are presented below.

Proof. Recursively using Lemma 4 from $t = 0$ to $T - 1$ and letting $\eta_t = \eta$ , we have that

$$
F (\mathbf {w} _ {T}) \leq F (\mathbf {w} _ {0}) - \frac {(1 - \gamma) \eta}{1 + \gamma} \sum_ {t = 0} ^ {T - 1} \| \nabla F (\mathbf {w} _ {t}) \| + \frac {2 \eta}{\gamma (1 + \gamma)} \sum_ {t = 0} ^ {T - 1} \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \| + \frac {T \eta^ {2} L}{2}. \tag {79}
$$

Therefore,

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \| \nabla F (\mathbf {w} _ {t}) \| \leq \frac {(1 + \gamma) [ F (\mathbf {w} _ {0}) - F (\mathbf {w} _ {T}) ]}{(1 - \gamma) \eta T} + \frac {2}{T \gamma (1 - \gamma)} \sum_ {t = 0} ^ {T - 1} \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \| + \frac {(1 + \gamma) \eta L}{2 (1 - \gamma)}. \tag {80}
$$

According to Assumption 2, we have that $F(\mathbf{w}_T) \geq F^*$ . Furthermore, by taking expectation on both sides, letting $\gamma = \frac{1}{3}$ and using Lemma 3, it is obtained that

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq \frac {2 [ F (\mathbf {w} _ {0}) - F ^ {*} ]}{\eta T} + \frac {9}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \mathbf {u} _ {t} - \nabla F (\mathbf {w} _ {t}) \| + \eta L \tag {81}
$$

$$
\leq \frac {2 \left[ F \left(\mathbf {w} _ {0}\right) - F ^ {*} \right]}{\eta T} + \frac {9 \eta L}{\alpha} + \frac {9 \sqrt {2 c m \delta (1 - \delta)} + 9}{\sqrt {B m (1 - \delta)}} \left(\frac {1}{\alpha T} + \sqrt {\alpha}\right) \sigma + \eta L \tag {82}
$$

$$
\leq \frac {2 F _ {0}}{\eta T} + \frac {1 0 \eta L}{\alpha} + \frac {9 \sqrt {2 c m \delta (1 - \delta)} + 9}{\sqrt {B m (1 - \delta)}} \left(\frac {1}{\alpha T} + \sqrt {\alpha}\right) \sigma . \tag {83}
$$

![](images/f1360d07b23dc371f3581cefbb737c3f44603378a806c36f9591fce0074a8eff.jpg)

# B.4 PROOF OF PROPOSITION 2

Proof. Learning rate $\eta$ appears only in the first two terms of the RHS of inequality (2). Thus,

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq \frac {2 F _ {0}}{\eta T} + \frac {1 0 \eta L}{\alpha} + \frac {9 \sqrt {2 c m \delta (1 - \delta)} + 9}{\sqrt {B m (1 - \delta)}} \left(\frac {1}{\alpha T} + \sqrt {\alpha}\right) \sigma \\ \leq 2 \sqrt {\frac {2 F _ {0}}{\eta T} \times \frac {1 0 \eta L}{\alpha}} + \frac {9 \sqrt {2 c m \delta (1 - \delta)} + 9}{\sqrt {B m (1 - \delta)}} \left(\frac {1}{\alpha T} + \sqrt {\alpha}\right) \sigma (84) \\ = \sqrt {\frac {8 0 L F _ {0}}{\alpha T}} + \frac {9 \sqrt {2 c m \delta (1 - \delta)} + 9}{\sqrt {B m (1 - \delta)}} \left(\frac {1}{\alpha T} + \sqrt {\alpha}\right) \sigma . (85) \\ \end{array}
$$

The second equation holds only and only if $\frac{2F_{0}}{\eta T} = \frac{10\eta L}{\alpha}$ , which is equivalent to that $\eta = \sqrt{\frac{\alpha F_{0}}{5LT}}$ . Furthermore, since $\alpha = \min\left(\frac{\sqrt{80LF_{0}Bm(1-\delta)}}{\left[9\sqrt{2cm\delta(1-\delta)}+9\right]\sigma\sqrt{T}}, 1\right)$ , we consider the following two cases.

(i) When $\alpha = \frac{\sqrt{80LF_0Bm(1 - \delta)}}{\left[9\sqrt{2cm\delta(1 - \delta)} + 9\right]\sigma\sqrt{T}}$ , we have that

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq 6 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {1}{2}} \left(\frac {5 L F _ {0} \sigma^ {2}}{T B m (1 - \delta)}\right) ^ {\frac {1}{4}} \\ + \frac {2 7 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{2}}}{4 \sqrt {5 T B ^ {2} m ^ {2} (1 - \delta) ^ {2} L F _ {0}}} \sigma^ {2}. \tag {86} \\ \end{array}
$$

(ii) When $\alpha = 1$ , it implies $\frac{\sqrt{80LF_0Bm(1 - \delta)}}{\left[9\sqrt{2cm\delta(1 - \delta)} + 9\right]\sigma\sqrt{T}} \geq 1$ . Namely, $\frac{9\sqrt{2cm\delta(1 - \delta)} + 9}{\sqrt{Bm(1 - \delta)}}\sigma \leq \frac{\sqrt{80LF_0}}{\sqrt{T}}$ . In this case,

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq \sqrt {\frac {8 0 L F _ {0}}{\alpha T}} + \frac {9 \sqrt {2 c m \delta (1 - \delta)} + 9}{\sqrt {B m (1 - \delta)}} \left(\frac {1}{\alpha T} + \sqrt {\alpha}\right) \sigma (87) \\ \leq \sqrt {\frac {8 0 L F _ {0}}{T}} + \sqrt {\frac {8 0 L F _ {0}}{T}} \left(\frac {1}{T} + 1\right) (88) \\ \leq 1 2 \sqrt {\frac {5 L F _ {0}}{T}}. (89) \\ \end{array}
$$

In summary,

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq 6 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {1}{2}} \left(\frac {5 L F _ {0} \sigma^ {2}}{T B m (1 - \delta)}\right) ^ {\frac {1}{4}} + 1 2 \sqrt {\frac {5 L F _ {0}}{T}} \\ + \frac {2 7 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{2}}}{4 \sqrt {5 T B ^ {2} m ^ {2} (1 - \delta) ^ {2} L F _ {0}}} \sigma^ {2}. \tag {90} \\ \end{array}
$$

In addition, when $\mathcal{C} = TBm(1 - \delta)$ is fixed, we have

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq 6 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {1}{2}} \left(\frac {5 L F _ {0} \sigma^ {2}}{\mathcal {C}}\right) ^ {\frac {1}{4}} \\ + 1 2 \sqrt {\frac {5 m (1 - \delta) L F _ {0} B}{\mathcal {C}}} + \frac {2 7 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{2}}}{4 \sqrt {5 m (1 - \delta) L F _ {0} \mathcal {C} B}} \sigma^ {2}. \tag {91} \\ \end{array}
$$

Moreover,

$$
1 2 \sqrt {\frac {5 m (1 - \delta) L F _ {0} B}{\mathcal {C}}} + \frac {2 7 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{2}}}{4 \sqrt {5 m (1 - \delta) L F _ {0} \mathcal {C} B}} \sigma^ {2}
$$

$$
\geq 2 \sqrt {\left(1 2 \sqrt {\frac {5 m (1 - \delta) L F _ {0} B}{\mathcal {C}}}\right) \times \left(\frac {2 7 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{2}}}{4 \sqrt {5 m (1 - \delta) L F _ {0} \mathcal {C} B}} \sigma^ {2}\right)} \tag {92}
$$

$$
= \frac {1 8 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{4}} \sigma}{\mathcal {C} ^ {\frac {1}{2}}}. \tag {93}
$$

The equation holds if and only if

$$
1 2 \sqrt {\frac {5 m (1 - \delta) L F _ {0} B}{\mathcal {C}}} = \frac {2 7 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{2}}}{4 \sqrt {5 m (1 - \delta) L F _ {0} \mathcal {C} B}} \sigma^ {2}, \tag {94}
$$

which is equivalent to that

$$
B = \frac {9 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{2}} \sigma^ {2}}{8 0 m (1 - \delta) L F _ {0}}. \tag {95}
$$

![](images/b05e60c63352ab69dcd1a23e5dd1e0c2162e17629e78ec9e1025e66cfc5e4fe1.jpg)

# C DISCUSSION ABOUT THE $(f, \kappa)$ -ROBUSTNESS

The theoretical analysis in the main text is based on the definition of $(\delta_{\mathrm{max}}, c)$ -robust aggregator (Karimireddy et al., 2021). In this section, we show that similar results can be obtained under the definition of $(f, \kappa)$ -robustness (Allouah et al., 2023). In existing works 11, f is defined to be the number of Byzantine workers. However, we have used the notation f to denote the loss function in this paper. In order to avoid misunderstanding, we will use $m\delta$ to denote the number of Byzantine workers in the following text, where m is the total number of workers and $\delta$ is the fraction of Byzantine workers. Firstly, we present the definition of $(\delta, \kappa)$ -robust aggregator, which is equivalent to the $(f, \kappa)$ -robustness (Allouah et al., 2023) since the worker number m is deterministic in this paper.

Definition 2 ((δ, κ)-robustness). Let δ ∈ [0, ½) and κ ≥ 0. An aggregator Agg(·) is called a (δ, κ)-robust aggregator if for any vectors x₁, ..., xₘ ∈ ℝᵈ and any set G ⊆ {1, ..., m} satisfying |G| = (1 - δ)m, we have that

$$
\left\| \mathbf {A} \mathbf {g} \mathbf {g} (\mathbf {x} _ {1}, \dots , \mathbf {x} _ {m}) - \bar {\mathbf {x}} _ {\mathcal {G}} \right\| ^ {2} \leq \frac {\kappa}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \left\| \mathbf {x} _ {k} - \bar {\mathbf {x}} _ {\mathcal {G}} \right\| ^ {2}, \tag {96}
$$

where $\bar{\mathbf{x}}_{\mathcal{G}} = \frac{1}{|\mathcal{G}|}\sum_{k\in \mathcal{G}}\mathbf{x}_k.$

Thus, for a $(\delta, \kappa)$ -robust aggregator $\mathbf{Agg}(\cdot)$ , when $\{\mathbf{x}_k\}_{k \in \mathcal{G}'}$ is a set of i.i.d. random vectors where $\mathcal{G} \subseteq \{1, \ldots, m\}$ satisfying $|\mathcal{G}| = (1 - \delta)m$ and $\mathbb{E} \| \mathbf{x}_k - \mathbb{E} [\mathbf{x}_k] \|^2 = \sigma^2$ for each $k \in \mathcal{G}$ , we have:

$$
\mathbb {E} \left\| \mathbf {A g g} \left(\mathbf {x} _ {1}, \dots , \mathbf {x} _ {m}\right) - \bar {\mathbf {x}} _ {\mathcal {G}} \right\| ^ {2} \leq \mathbb {E} \left[ \frac {\kappa}{| \mathcal {G} |} \sum_ {k \in \mathcal {G}} \left\| \mathbf {x} _ {k} - \bar {\mathbf {x}} _ {\mathcal {G}} \right\| ^ {2} \right] = \frac {| \mathcal {G} | - 1}{| \mathcal {G} |} \kappa \sigma^ {2} = \left(1 - \frac {1}{(1 - \delta) m}\right) \kappa \sigma^ {2}. \tag {97}
$$

Meanwhile, for a $(\delta_{\max}, c)$ -robust aggregator $\mathbf{Agg}(\cdot)$ , when $\{\mathbf{x}_k\}_{k \in \mathcal{G}'}$ is a set of i.i.d. random vectors where $\mathcal{G} \subseteq \{1, \ldots, m\}$ satisfying $|\mathcal{G}| = (1 - \delta)m$ and $\mathbb{E} \| \mathbf{x}_k - \mathbb{E}[\mathbf{x}_k] \|^2 = \sigma^2$ for each $k \in \mathcal{G}$ , we have $\mathbb{E} \| \mathbf{x}_k - \mathbf{x}_{k'} \|^2 \leq 2\sigma^2$ according to (2), and thus

$$
\mathbb {E} \| \mathbf {A g g} (\mathbf {x} _ {1}, \dots , \mathbf {x} _ {m}) - \bar {\mathbf {x}} _ {\mathcal {G}} \| ^ {2} \leq 2 c \delta \sigma^ {2}. \tag {98}
$$

Comparing (97) and (98), we can find that in i.i.d. cases, the upper bounds of aggregation error under the two definitions are of the same order $O(\sigma^2)$ . Moreover, the other factors $|\mathcal{G}|$ , $\kappa$ , $c$ and $\delta$ will not change during the distributed learning process. Thus, to obtain the optimal batch size that minimizes the theoretical upper bound under the $(\delta, \kappa)$ -robustness, we can simply replace the factor $c\delta$ with $\tilde{c} = \frac{\kappa}{2}\left(1 - \frac{1}{(1 - \delta)m}\right)$ .

Recall that the optimal batch size that minimize the theoretical upper bound for ByzSGDm and ByzSGDnm under the definition of $(\delta_{\max}, c)$ -robustness are

$$
B ^ {*} = \left(\frac {3}{1 6 L ^ {2} (F _ {0}) ^ {2} m}\right) ^ {\frac {1}{3}} \left(\frac {c \delta (1 + c \delta m)}{m (1 - \delta)}\right) ^ {\frac {1}{3}} \sigma^ {\frac {4}{3}} \mathcal {C} ^ {\frac {1}{3}}, \tag {99}
$$

and

$$
\tilde {B} ^ {*} = \frac {9 \left[ \sqrt {2 c m \delta (1 - \delta)} + 1 \right] ^ {\frac {3}{2}} \sigma^ {2}}{8 0 m (1 - \delta) L F _ {0}} = \frac {9 \left[ \frac {\sqrt {2 c m \delta}}{(1 - \delta) ^ {\frac {1}{6}}} + \frac {1}{(1 - \delta) ^ {\frac {2}{3}}} \right] ^ {\frac {3}{2}} \sigma^ {2}}{8 0 m L F _ {0}}, \tag {100}
$$

respectively. Thus, under the definition of $(\delta,\kappa)$ -robustness, the optimal batch size that minimize the theoretical upper bound for ByzSGDm and ByzSGDnm are

$$
\left(\frac {3}{1 6 L ^ {2} (F _ {0}) ^ {2} m}\right) ^ {\frac {1}{3}} \left(\frac {\tilde {c} (1 + \tilde {c} m)}{m (1 - \delta)}\right) ^ {\frac {1}{3}} \sigma^ {\frac {4}{3}} \mathcal {C} ^ {\frac {1}{3}},
$$

and

$$
\frac {9 \left[ \frac {\sqrt {2 \tilde {c} m}}{(1 - \delta) ^ {\frac {1}{6}}} + \frac {1}{(1 - \delta) ^ {\frac {2}{3}}} \right] ^ {\frac {3}{2}} \sigma^ {2}}{8 0 m L F _ {0}},
$$

respectively. Notice that the term $\frac{1}{1 - \delta}$ is monotonically increasing w.r.t. $\delta$ . Thus, in order to prove that the optimal batch size is monotonically increasing w.r.t. $\delta$ , we only need to prove that $\tilde{c} = \frac{\kappa}{2}\left(1 - \frac{1}{(1 - \delta)m}\right)$ is monotonically increasing w.r.t. $\delta$ .

It has been shown in existing works (Allouah et al., 2023) that (i) $\kappa = 1 + \frac{\delta}{1 - 2\delta}$ for Krum, (ii) $\kappa = \frac{\delta}{1 - 2\delta} (1 + \frac{\delta}{1 - 2\delta})$ for coordinate-wise trimmed-mean, (iii) $\kappa = (1 + \frac{\delta}{1 - 2\delta})^2$ for coordinate-wise median and geometric median, and that (iv) the lower bound for $\kappa$ is $\kappa = \frac{\delta}{1 - 2\delta}$ . Then we prove that $\tilde{c}$ is monotonically increasing w.r.t. $\delta$ for these four cases separately.

Case (i). For Krum, we have $\kappa = 1 + \frac{\delta}{1 - 2\delta}$ ( $0 \leq \delta < \frac{1}{2}$ ), and thus,

$$
\tilde {c} (\delta) = \frac {1}{2} \left(1 + \frac {\delta}{1 - 2 \delta}\right) \left(1 - \frac {1}{(1 - \delta) m}\right) \tag {101}
$$

$$
= \frac {1}{2} \cdot \frac {1 - \delta}{1 - 2 \delta} \left(1 - \frac {1}{(1 - \delta) m}\right) \tag {102}
$$

$$
= \frac {1}{2} \cdot \frac {1 - \delta - \frac {1}{m}}{1 - 2 \delta}. \tag {103}
$$

Therefore, the derivative of $\tilde{c} (\delta)$ is

$$
\tilde {c} ^ {\prime} (\delta) = \frac {1}{2} \cdot \frac {- (1 - 2 \delta) + 2 (1 - \delta - \frac {1}{m})}{(1 - 2 \delta) ^ {2}} = \frac {m - 2}{2 m (1 - 2 \delta) ^ {2}}. \tag {104}
$$

Since the fraction of Byzantine workers is smaller than $\frac{1}{2}$ , Byzantine worker can appear only when the total worker number m > 2. Thus, we have m - 2 > 0. In addition, $(1 - 2\delta)^{2} > 0$ since $0 \leq \delta < \frac{1}{2}$ . Therefore, we have $\tilde{c}'(\delta) > 0$ , and $\tilde{c}(\delta)$ is monotonically increasing w.r.t. $\delta$ .

Case (ii). For coordinate-wise trimmed-mean, we have $\kappa = \frac{\delta}{1 - 2\delta} (1 + \frac{\delta}{1 - 2\delta})$ . Thus,

$$
\tilde {c} (\delta) = \frac {1}{2} \cdot \frac {\delta}{1 - 2 \delta} \left(1 + \frac {\delta}{1 - 2 \delta}\right) \left(1 - \frac {1}{(1 - \delta) m}\right) \tag {105}
$$

$$
= \left[ \frac {1}{2} \left(1 + \frac {\delta}{1 - 2 \delta}\right) \left(1 - \frac {1}{(1 - \delta) m}\right) \right] \cdot \left(\frac {\delta}{1 - 2 \delta}\right). \tag {106}
$$

It has been proven in case (i) that the first term $\left[\frac{1}{2}\left(1+\frac{\delta}{1-2\delta}\right)\left(1-\frac{1}{(1-\delta)m}\right)\right]$ is monotonically increasing w.r.t. $\delta$ . Since both $\delta$ and $\frac{1}{1-2\delta}$ increase as $\delta\in[0,\frac{1}{2})$ increases, the second term $\frac{\delta}{1-2\delta}$ is also monotonically increasing w.r.t. $\delta$ . Moreover, the two terms are both positive. Thus, we have that $\tilde{c}(\delta)$ is monotonically increasing w.r.t. $\delta$ .

Case (iii). For coordinate-wise median and geometric median, we have $\kappa = (1 + \frac{\delta}{1 - 2\delta})^2$ . Thus,

$$
\tilde {c} (\delta) = \frac {1}{2} \cdot \left(1 + \frac {\delta}{1 - 2 \delta}\right) ^ {2} \left(1 - \frac {1}{(1 - \delta) m}\right) \tag {107}
$$

$$
= \left[ \frac {1}{2} \left(1 + \frac {\delta}{1 - 2 \delta}\right) \left(1 - \frac {1}{(1 - \delta) m}\right) \right] \cdot \left(1 + \frac {\delta}{1 - 2 \delta}\right). \tag {108}
$$

It has been proven in case (i) that the first term $\left[\frac{1}{2}\left(1+\frac{\delta}{1-2\delta}\right)\left(1-\frac{1}{(1-\delta)m}\right)\right]$ is monotonically increasing w.r.t. $\delta$ . Since both $\delta$ and $\frac{1}{1-2\delta}$ increase as $\delta\in[0,\frac{1}{2})$ increases, the second term $\left(1+\frac{\delta}{1-2\delta}\right)$ is also monotonically increasing w.r.t. $\delta$ . Moreover, the two terms are both positive. Thus, we have that $\tilde{c}(\delta)$ is monotonically increasing w.r.t. $\delta$ .

Case (iv). The lower bound for $\kappa$ is $\kappa = \frac{\delta}{1-2\delta}$ . Thus, the lower bound for $\tilde{c}(\delta)$ is

$$
\tilde {c} (\delta) = \frac {1}{2} \cdot \frac {\delta}{1 - 2 \delta} \left(1 - \frac {1}{(1 - \delta) m}\right) \tag {109}
$$

$$
= \frac {1}{2} \left(1 + \frac {\delta}{1 - 2 \delta}\right) \left(1 - \frac {1}{(1 - \delta) m}\right) + \frac {1}{2} \left(\frac {1}{(1 - \delta) m} - 1\right) \tag {110}
$$

It has been proven in case (i) that the first term $\left[\frac{1}{2}\left(1+\frac{\delta}{1-2\delta}\right)\left(1-\frac{1}{(1-\delta)m}\right)\right]$ is monotonically increasing w.r.t. $\delta$ . The second term $\frac{1}{2}\left(\frac{1}{(1-\delta)m}-1\right)$ is also monotonically increasing w.r.t. $\delta$ . Thus, for the lower bound of $\kappa$ , $\tilde{c}(\delta)$ is also monotonically increasing w.r.t. $\delta$ .

In summary, under the definition of $(f,\kappa)$ -robustness, the optimal batch size that minimize the theoretical upper bound for ByzSGDm and ByzSGDnm increase with the fraction of Byzantine workers $\delta$ . The results are consistent with those under the definition of $(\delta_{\mathrm{max}},c)$ -robust aggregator.

# D COMPARISON WITH BYZ-VR-MARINA

We compare ByzSGDnm with Byz-VR-MARINA (Gorbunov et al., 2023) in this section.

# D.1 VARIANCE REDUCTION TECHNIQUES

We first compare the techniques for variance reduction in ByzSGDnm and Byz-VR-MARINA. As presented in the main text, in ByzSGDnm, the variance of stochastic gradients is reduced mainly by using (i) large batch size and (ii) local momentums. The two techniques introduce little extra computation cost but are empirically effective, as the empirical results in the main text show. However, the two techniques work only in the i.i.d. cases that we focus on in this work.

Byz-VR-MARINA adopts PAGE (Li et al., 2021) for variance reduction. In contrast to the techniques in ByzSGDnm, PAGE can reduce the bias in general non-i.i.d. cases but introduces much more computation cost (as we will shown in Appendix D.3).

# D.2 COMMUNICATION COST

ByzSGDnm and Byz-VR-MARINA take different ways to reduce the communication cost in distributed learning. ByzSGDnm uses large-batch training (Cutkosky & Mehta, 2020; Goyal et al.,

2017; You et al., 2020) to reduce the number communication round while Byz-VR-MARINA uses communication compression such as quantization (Alistarh et al., 2017) to reduce the transmitted bits in each communication round.

# D.3 COMPUTATION COST

Since the computation cost mainly comes from gradient computation in many real-world distributed learning applications, we use the number of gradient computation on non-Byzantine workers to estimate the computation cost for both methods. In ByzSGDnm, each of the $m(1-\delta)$ non-Byzantine workers will draw a mini-batch of B samples and compute the corresponding stochastic gradients at each iteration. Thus, the total number of gradient computation in each iteration for ByzSGDnm is $Bm(1-\delta)$ . In each iteration of Byz-VR-MARINA, each of the $m(1-\delta)$ non-Byzantine workers computes the full gradient with probability p and computes a mini-batch estimation of gradient differences with probability 1-p. Since the estimation with batch size B requires 2B times of gradient computation, the expectation of the total number of gradient computation in each iteration for Byz-VR-MARINA is $(1-\delta)np + 2Bm(1-\delta)(1-p)$ where n is the total number of training instances. Therefore, the number of gradient computation in Byz-VR-MARINA for each iteration is $k_{C}$ times that of ByzSGDnm in expectation, where

$$
k _ {\mathcal {C}} \triangleq \frac {(1 - \delta) n p + 2 B m (1 - \delta) (1 - p)}{B m (1 - \delta)} = \frac {n p + 2 B m (1 - p)}{B m} = 2 + p \left(\frac {n}{B m} - 2\right).
$$

Since Bm is the total batch size of each iteration and n is the total number of instances, $\frac{n}{Bm}$ is the number of iteration per epoch, which is typically much larger than 2. Therefore, the number of gradient computation for Byz-VR-MARINA is more than twice that of ByzSGDnm in expectation.

# D.4 THEORETICAL CONVERGENCE ORDER

As we have proved in Section 4 in the main text, under Assumptions 1, 2 and 3, we have

$$
\min _ {t = 0, \dots , T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| \leq O \left(\frac {1}{T ^ {\frac {1}{4}}}\right)
$$

or equivalently

$$
\min _ {t = 0, \dots , T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| ^ {2} \leq O \left(\frac {1}{T ^ {\frac {1}{2}}}\right)
$$

for ByzSGDnm with $(\delta_{\mathrm{max}}, c)$ -robust aggregators. Meanwhile, as the results in Gorbunov et al. (2023) show, under similar conditions, we have

$$
\min _ {t = 0, \dots , T - 1} \mathbb {E} \| \nabla F (\mathbf {w} _ {t}) \| ^ {2} \leq O \left(\frac {1}{T}\right)
$$

for Byz-VR-MARINA. It has been shown in existing works (Arjevani et al., 2023; Cutkosky & Mehta, 2020) that the convergence order $\min_t \mathbb{E} \| \nabla F(\mathbf{w}_t) \| \leq O(1/T^{\frac{1}{4}})$ is optimal for SGD under Assumptions 1, 2 and 3. Byz-VR-MARINA achieves the faster convergence order mainly because of intermittently using full gradients for variance reduction. The strategy of intermittently using full gradients is also adopted in some traditional methods such as SVRG (Johnson & Zhang, 2013). Although using full gradients improves the theoretical convergence order, it also increases the expected number of gradient computation as discussed in Appendix D.3 above.

In summary, compared to ByzSGDnm, Byz-VR-MARINA has a faster theoretical convergence order with respect to the iteration number, but also requires more times of gradient computation per iteration. Moreover, the number of gradient computation for Byz-VR-MARINA depends on p and n.

# D.5 EMPIRICAL PERFORMANCE

Finally, we empirically compare the performance of ByzSGDnm and Byz-VR-MARINA. The experimental settings are the same as those presented in Section 5 of the main text. Specifically, we use cosine annealing (Loshchilov & Hutter, 2017) learning rates for each method. The initial learning rate for Byz-VR-MARINA is selected from $\{0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0\}$ , and

the best final top-1 test accuracy is used as the final metrics. We set the batch size to $512 \times 8$ for ByzSGDnm. We use the top-1 test accuracy w.r.t. gradient computation number as final metrics. For fairness, we do not use communication compression for Byz-VR-MARINA. The empirical results of Byz-VR-MARINA when the batch size is $32 \times 8$ , $64 \times 8$ , $128 \times 8$ , $256 \times 8$ and $512 \times 8$ are presented in Figure 1, Figure 2, Figure 3, Figure 4 and Figure 5, respectively.

![](images/4364e1a40f9658f9084c2775f2da01cb8a32ac16d1e65e0aa0ef7b6c2cc826ea.jpg)

<details>
<summary>line</summary>

| Gradient computation number (x10^6) | ByzSGDnm with KR | Byz-VR-MARINA with KR (B=32×8, p=0.2) | Byz-VR-MARINA with KR (B=32×8, p=0.1) | Byz-VR-MARINA with KR (B=32×8, p=0.05) |
| ------------------------------------ | ---------------- | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                    | 10               | 10                                     | 10                                     | 10                                      |
| 1                                    | 45               | 35                                     | 30                                     | 25                                      |
| 2                                    | 60               | 35                                     | 35                                     | 30                                      |
| 3                                    | 70               | 35                                     | 35                                     | 35                                      |
| 4                                    | 75               | 35                                     | 35                                     | 40                                      |
| 5                                    | 80               | 35                                     | 35                                     | 45                                      |
| 6                                    | 85               | 35                                     | 40                                     | 45                                      |
| 7                                    | 85               | 35                                     | 40                                     | 45                                      |
| 8                                    | 85               | 35                                     | 40                                     | 45                                      |
</details>

![](images/2f1f6b74b9cc67d2b932e4649ab54de1e04baf3607d8e300a4b9c8c4ad5c3a3c.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with GM | Byz-VR-MARINA with GM (B=32×8, p=0.2) | Byz-VR-MARINA with GM (B=32×8, p=0.1) | Byz-VR-MARINA with GM (B=32×8, p=0.05) |
| ----------------------------------- | ---------------- | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                   | 10               | 10                                     | 10                                     | 10                                      |
| 1                                   | 70               | 30                                     | 30                                     | 25                                      |
| 2                                   | 80               | 35                                     | 35                                     | 30                                      |
| 3                                   | 85               | 35                                     | 40                                     | 35                                      |
| 4                                   | 85               | 35                                     | 40                                     | 35                                      |
| 5                                   | 85               | 35                                     | 40                                     | 35                                      |
| 6                                   | 90               | 40                                     | 40                                     | 40                                      |
| 7                                   | 90               | 35                                     | 35                                     | 40                                      |
| 8                                   | 90               | 40                                     | 40                                     | 40                                      |
</details>

![](images/2444aee2bcc98979a18091bd2180c29db9ce6b75b8388aa717b3a80452898e04.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with CM | Byz-VR-MARINA with CM (B=32×8, p=0.2) | Byz-VR-MARINA with CM (B=32×8, p=0.1) | Byz-VR-MARINA with CM (B=32×8, p=0.05) |
| ----------------------------------- | ---------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                   | 10               | 10                                     | 10                                     | 10                                     |
| 1                                   | ~45              | ~25                                    | ~30                                    | ~25                                    |
| 2                                   | ~60              | ~30                                    | ~35                                    | ~30                                    |
| 3                                   | ~70              | ~35                                    | ~40                                    | ~35                                    |
| 4                                   | ~75              | ~35                                    | ~40                                    | ~35                                    |
| 5                                   | ~80              | ~35                                    | ~40                                    | ~35                                    |
| 6                                   | ~85              | ~35                                    | ~45                                    | ~40                                    |
| 7                                   | ~85              | ~35                                    | ~40                                    | ~40                                    |
| 8                                   | ~85              | ~35                                    | ~40                                    | ~40                                    |
</details>

![](images/f379decf54bbe9a89deb9607c1eedbe913a01ea4290cbe3380b45a220ee4d4c1.jpg)

<details>
<summary>line</summary>

| Gradient computation number | ByzSGDnm with CC | Byz-VR-MARINA with CC (B=32×8, p=0.2) | Byz-VR-MARINA with CC (B=32×8, p=0.1) | Byz-VR-MARINA with CC (B=32×8, p=0.05) |
| --------------------------- | ---------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                           | 10               | 10                                     | 10                                     | 10                                     |
| 1                           | 70               | 35                                     | 30                                     | 25                                     |
| 2                           | 80               | 40                                     | 35                                     | 30                                     |
| 3                           | 85               | 45                                     | 40                                     | 35                                     |
| 4                           | 88               | 48                                     | 42                                     | 38                                     |
| 5                           | 90               | 50                                     | 45                                     | 40                                     |
| 6                           | 92               | 52                                     | 48                                     | 42                                     |
| 7                           | 93               | 55                                     | 50                                     | 45                                     |
| 8                           | 94               | 58                                     | 52                                     | 48                                     |
</details>

Figure 1: Top-1 test accuracy w.r.t. gradient computation number of different methods when there are 3 workers under ALIE attack. The batch size for Byz-VR-MARINA is set to $32 \times 8$ .

![](images/8a3e553f17a51528244fad6c92cac0ddd8612ee175126a3b4343017fda5ee922.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with KR | Byz-VR-MARINA with KR (B=64×8, p=0.2) | Byz-VR-MARINA with KR (B=64×8, p=0.1) | Byz-VR-MARINA with KR (B=64×8, p=0.05) |
| ----------------------------------- | ---------------- | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                   | 10               | 10                                     | 10                                     | 10                                      |
| 1                                   | ~50              | ~30                                    | ~30                                    | ~25                                     |
| 2                                   | ~60              | ~35                                    | ~35                                    | ~30                                     |
| 3                                   | ~70              | ~35                                    | ~35                                    | ~35                                     |
| 4                                   | ~75              | ~35                                    | ~35                                    | ~35                                     |
| 5                                   | ~80              | ~35                                    | ~35                                    | ~35                                     |
| 6                                   | ~85              | ~35                                    | ~35                                    | ~35                                     |
| 7                                   | ~85              | ~35                                    | ~35                                    | ~35                                     |
| 8                                   | ~85              | ~35                                    | ~35                                    | ~35                                     |
</details>

![](images/52b49234463e9acdea9b907e372736473ad5d8aa4705f2e5cd866f149cf37606.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with GM | Byz-VR-MARINA with GM (B=64×8, p=0.2) | Byz-VR-MARINA with GM (B=64×8, p=0.1) | Byz-VR-MARINA with GM (B=64×8, p=0.05) |
| ---------------------------------- | ---------------- | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                  | 10               | 10                                     | 10                                     | 10                                      |
| 1                                  | 70               | 25                                     | 25                                     | 25                                      |
| 2                                  | 80               | 30                                     | 30                                     | 30                                      |
| 3                                  | 85               | 35                                     | 35                                     | 35                                      |
| 4                                  | 85               | 35                                     | 35                                     | 35                                      |
| 5                                  | 85               | 35                                     | 35                                     | 35                                      |
| 6                                  | 85               | 35                                     | 35                                     | 35                                      |
| 7                                  | 85               | 35                                     | 35                                     | 35                                      |
| 8                                  | 85               | 35                                     | 35                                     | 35                                      |
</details>

![](images/77f2f51da7f647d516c57b59f23972dfe15c49222191a3f52d68857e8beb4c73.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with CM | Byz-VR-MARINA with CM (B=64×8, p=0.2) | Byz-VR-MARINA with CM (B=64×8, p=0.1) | Byz-VR-MARINA with CM (B=64×8, p=0.05) |
| ---------------------------------- | ---------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                  | 10               | 10                                     | 10                                     | 10                                     |
| 1                                  | ~45              | ~25                                    | ~30                                    | ~20                                    |
| 2                                  | ~60              | ~30                                    | ~35                                    | ~25                                    |
| 3                                  | ~70              | ~35                                    | ~40                                    | ~30                                    |
| 4                                  | ~75              | ~40                                    | ~45                                    | ~35                                    |
| 5                                  | ~80              | ~45                                    | ~50                                    | ~40                                    |
| 6                                  | ~85              | ~50                                    | ~55                                    | ~45                                    |
| 7                                  | ~88              | ~55                                    | ~60                                    | ~50                                    |
| 8                                  | ~90              | ~60                                    | ~65                                    | ~55                                    |
</details>

![](images/131879a7940416eaaf5844a62ef795b47ccdcc9d0f89b1f97b2759c323cef7a9.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with CC | Byz-VR-MARINA with CC (B=64×8, p=0.2) | Byz-VR-MARINA with CC (B=64×8, p=0.1) | Byz-VR-MARINA with CC (B=64×8, p=0.05) |
| ---------------------------------- | ---------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                  | 10               | 10                                     | 10                                     | 10                                     |
| 1                                  | 70               | 25                                     | 25                                     | 25                                     |
| 2                                  | 80               | 35                                     | 35                                     | 35                                     |
| 3                                  | 85               | 38                                     | 38                                     | 38                                     |
| 4                                  | 88               | 40                                     | 40                                     | 40                                     |
| 5                                  | 90               | 42                                     | 42                                     | 42                                     |
| 6                                  | 92               | 45                                     | 45                                     | 45                                     |
| 7                                  | 93               | 48                                     | 48                                     | 48                                     |
| 8                                  | 94               | 50                                     | 50                                     | 50                                     |
</details>

Figure 2: Top-1 test accuracy w.r.t. gradient computation number of different methods when there are 3 workers under ALIE attack. The batch size for Byz-VR-MARINA is set to $64 \times 8$ .

![](images/708b05fc94a777c4db8f9e29e57ab002d2225fa00b2f68d8f5a25fb277850ee9.jpg)

<details>
<summary>line</summary>

| Gradient computation number (x10^6) | Byz-SGDnm with KR | Byz-VR-MARINA with KR (B=128×8, p=0.2) | Byz-VR-MARINA with KR (B=128×8, p=0.1) | Byz-VR-MARINA with KR (B=128×8, p=0.05) |
| ------------------------------------ | ----------------- | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                    | 10                | 10                                     | 10                                     | 10                                      |
| 1                                    | 45                | 30                                     | 25                                     | 20                                      |
| 2                                    | 60                | 35                                     | 30                                     | 25                                      |
| 3                                    | 70                | 40                                     | 35                                     | 30                                      |
| 4                                    | 75                | 45                                     | 40                                     | 35                                      |
| 5                                    | 80                | 45                                     | 40                                     | 35                                      |
| 6                                    | 85                | 45                                     | 40                                     | 35                                      |
| 7                                    | 85                | 45                                     | 40                                     | 35                                      |
| 8                                    | 85                | 45                                     | 40                                     | 35                                      |
</details>

![](images/eb234eaa0804328014e6301033058bdb12d17d55b358a2f7c0f4b70160bf352d.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with GM | Byz-VR-MARINA with GM (B=128×8, p=0.2) | Byz-VR-MARINA with GM (B=128×8, p=0.1) | Byz-VR-MARINA with GM (B=128×8, p=0.05) |
| ---------------------------------- | ---------------- | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                  | 10               | 10                                     | 10                                     | 10                                      |
| 1                                  | 60               | 30                                     | 25                                     | 20                                      |
| 2                                  | 75               | 40                                     | 35                                     | 30                                      |
| 3                                  | 80               | 45                                     | 40                                     | 35                                      |
| 4                                  | 85               | 48                                     | 42                                     | 38                                      |
| 5                                  | 88               | 50                                     | 45                                     | 40                                      |
| 6                                  | 90               | 52                                     | 48                                     | 42                                      |
| 7                                  | 92               | 55                                     | 50                                     | 45                                      |
| 8                                  | 95               | 58                                     | 52                                     | 48                                      |
</details>

![](images/ebf18a191af4a122905ca6874fa846c0fb3efe7c3cb222f163600346d9e572c4.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | Byz-SGDnm with CM | Byz-VR-MARINA with CM (B=128×8, p=0.2) | Byz-VR-MARINA with CM (B=128×8, p=0.1) | Byz-VR-MARINA with CM (B=128×8, p=0.05) |
| ---------------------------------- | ----------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                  | 10                | 10                                     | 10                                     | 10                                     |
| 1                                  | ~60               | ~25                                    | ~25                                    | ~30                                    |
| 2                                  | ~75               | ~35                                    | ~35                                    | ~35                                    |
| 3                                  | ~80               | ~40                                    | ~40                                    | ~40                                    |
| 4                                  | ~85               | ~40                                    | ~40                                    | ~40                                    |
| 5                                  | ~85               | ~40                                    | ~40                                    | ~40                                    |
| 6                                  | ~85               | ~40                                    | ~40                                    | ~40                                    |
| 7                                  | ~85               | ~40                                    | ~40                                    | ~40                                    |
| 8                                  | ~85               | ~40                                    | ~40                                    | ~40                                    |
</details>

![](images/21a3587ee3ad0f24a4a7d05be34d9c59ce2bbef19d6c8e1ad03e7ebf7c44404e.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | Byz-SGDnm with CC | Byz-VR-MARINA with CC (B=128×8, p=0.2) | Byz-VR-MARINA with CC (B=128×8, p=0.1) | Byz-VR-MARINA with CC (B=128×8, p=0.05) |
| ----------------------------------- | ----------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                   | 10                | 10                                     | 10                                     | 10                                     |
| 1                                   | 60                | 30                                     | 25                                     | 20                                     |
| 2                                   | 75                | 35                                     | 30                                     | 25                                     |
| 3                                   | 80                | 40                                     | 35                                     | 30                                     |
| 4                                   | 85                | 45                                     | 40                                     | 35                                     |
| 5                                   | 88                | 48                                     | 42                                     | 38                                     |
| 6                                   | 90                | 50                                     | 45                                     | 40                                     |
| 7                                   | 92                | 52                                     | 48                                     | 42                                     |
| 8                                   | 93                | 53                                     | 50                                     | 43                                     |
</details>

Figure 3: Top-1 test accuracy w.r.t. gradient computation number of different methods when there are 3 workers under ALIE attack. The batch size for Byz-VR-MARINA is set to $128 \times 8$ .

![](images/a2850907fa919efc593b11344405dd392baf8af408b1826c13037a4e27baf6b2.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | Byz-SGDnm with KR | Byz-VR-MARINA with KR (B=256×8, p=0.2) | Byz-VR-MARINA with KR (B=256×8, p=0.1) | Byz-VR-MARINA with KR (B=256×8, p=0.05) |
| ----------------------------------- | ----------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                   | 10                | 10                                     | 10                                     | 10                                     |
| 1                                   | ~45               | ~30                                    | ~25                                    | ~20                                    |
| 2                                   | ~60               | ~35                                    | ~30                                    | ~25                                    |
| 3                                   | ~70               | ~40                                    | ~35                                    | ~30                                    |
| 4                                   | ~75               | ~45                                    | ~40                                    | ~35                                    |
| 5                                   | ~80               | ~45                                    | ~40                                    | ~35                                    |
| 6                                   | ~85               | ~45                                    | ~40                                    | ~35                                    |
| 7                                   | ~85               | ~45                                    | ~40                                    | ~35                                    |
| 8                                   | ~85               | ~45                                    | ~40                                    | ~35                                    |
</details>

![](images/0036287dd73d56f10f24c30c5eb52d8664a81c5744ce62553524bc70a5728fbe.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | Byz-SGDnm with GM | Byz-VR-MARINA with GM (B=256×8, p=0.2) | Byz-VR-MARINA with GM (B=256×8, p=0.1) | Byz-VR-MARINA with GM (B=256×8, p=0.05) |
| ---------------------------------- | ----------------- | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                  | 10                | 10                                     | 10                                     | 10                                      |
| 1                                  | 70                | 30                                     | 30                                     | 30                                      |
| 2                                  | 80                | 35                                     | 35                                     | 35                                      |
| 3                                  | 85                | 38                                     | 38                                     | 38                                      |
| 4                                  | 88                | 40                                     | 40                                     | 40                                      |
| 5                                  | 90                | 42                                     | 42                                     | 42                                      |
| 6                                  | 92                | 43                                     | 43                                     | 43                                      |
| 7                                  | 93                | 44                                     | 44                                     | 44                                      |
| 8                                  | 94                | 45                                     | 45                                     | 45                                      |
</details>

![](images/9e40a43d75a0394414e85c6bc3ade064bcd0c62c2f81f19a9fab5da538fb813d.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | Byz-SGDnm with CM | Byz-VR-MARINA with CM (B=256 × 8, p=0.2) | Byz-VR-MARINA with CM (B=256 × 8, p=0.1) | Byz-VR-MARINA with CM (B=256 × 8, p=0.05) |
| ---------------------------------- | ----------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| 0                                  | ~15               | ~15                                      | ~15                                      | ~10                                      |
| 1                                  | ~45               | ~25                                      | ~25                                      | ~20                                      |
| 2                                  | ~60               | ~30                                      | ~30                                      | ~25                                      |
| 3                                  | ~70               | ~35                                      | ~35                                      | ~30                                      |
| 4                                  | ~75               | ~40                                      | ~35                                      | ~35                                      |
| 5                                  | ~80               | ~45                                      | ~35                                      | ~35                                      |
| 6                                  | ~85               | ~45                                      | ~35                                      | ~35                                      |
| 7                                  | ~85               | ~45                                      | ~35                                      | ~35                                      |
| 8                                  | ~85               | ~45                                      | ~35                                      | ~35                                      |
</details>

![](images/5a3f171a309169983b7fb0591b1f59113534d28a43791e2fa5ab81b0d97618eb.jpg)  
Figure 4: Top-1 test accuracy w.r.t. gradient computation number of different methods when there are 3 workers under ALIE attack. The batch size for Byz-VR-MARINA is set to $256 \times 8$ .

As illustrated in Figure 1, Figure 2, Figure 3, Figure 4 and Figure 5, Byz-VR-MARINA converges much more slowly and has a much lower final top-1 accuracy than ByzSGDnm. There are mainly two reasons. Firstly, the full gradients in Byz-VR-MARINA is used to alleviate the bias in non-i.i.d. cases and is computation expensive. However, in i.i.d. cases that we focus on in this work, the full gradient is unnecessary and requires a large number times of gradient computation. Secondly, it

![](images/731c2fa911f7ed21a0db5e1a4357532d96b46f727740cf56a09458dfae6d36c1.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | Byz-SGDnrm with KR | Byz-VR-MARINA with KR (B=512×8, p=0.2) | Byz-VR-MARINA with KR (B=512×8, p=0.1) | Byz-VR-MARINA with KR (B=512×8, p=0.05) |
| ----------------------------------- | ------------------ | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                   | 10                 | 10                                     | 10                                     | 10                                      |
| 1                                   | 45                 | 30                                     | 25                                     | 20                                      |
| 2                                   | 60                 | 35                                     | 30                                     | 25                                      |
| 3                                   | 70                 | 40                                     | 35                                     | 30                                      |
| 4                                   | 75                 | 45                                     | 40                                     | 35                                      |
| 5                                   | 80                 | 50                                     | 45                                     | 40                                      |
| 6                                   | 85                 | 55                                     | 50                                     | 45                                      |
| 7                                   | 90                 | 60                                     | 55                                     | 50                                      |
| 8                                   | 95                 | 65                                     | 60                                     | 55                                      |
| 9                                   | 100                | 70                                     | 65                                     | 60                                      |
</details>

![](images/17aa53236fdf22b756cc41be94c124e09deda2dcee08b04933efb1d7cd864b0c.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with GM | Byz-VR-MARINA with GM (B=512×8, p=0.2) | Byz-VR-MARINA with GM (B=512×8, p=0.1) | Byz-VR-MARINA with GM (B=512×8, p=0.05) |
| ---------------------------------- | ---------------- | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                  | 10               | 10                                     | 10                                     | 10                                      |
| 1                                  | 60               | 25                                     | 20                                     | 20                                      |
| 2                                  | 75               | 30                                     | 25                                     | 25                                      |
| 3                                  | 80               | 35                                     | 30                                     | 30                                      |
| 4                                  | 85               | 40                                     | 35                                     | 35                                      |
| 5                                  | 90               | 45                                     | 40                                     | 40                                      |
| 6                                  | 90               | 45                                     | 40                                     | 40                                      |
| 7                                  | 90               | 45                                     | 40                                     | 40                                      |
| 8                                  | 90               | 45                                     | 40                                     | 40                                      |
</details>

![](images/99cef26a54815bfe86c497008818dac0fa04b86d69db669a6a352535bfa5fa15.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | Byz-SGDnm with CM | Byz-VR-MARINA with CM (B=512×8, p=0.2) | Byz-VR-MARINA with CM (B=512×8, p=0.1) | Byz-VR-MARINA with CM (B=512×8, p=0.05) |
| ----------------------------------- | ----------------- | -------------------------------------- | -------------------------------------- | --------------------------------------- |
| 0                                   | 10                | 10                                     | 10                                     | 10                                      |
| 1                                   | ~30               | ~25                                    | ~25                                    | ~20                                     |
| 2                                   | ~50               | ~30                                    | ~30                                    | ~25                                     |
| 3                                   | ~70               | ~35                                    | ~35                                    | ~30                                     |
| 4                                   | ~80               | ~40                                    | ~40                                    | ~35                                     |
| 5                                   | ~85               | ~45                                    | ~45                                    | ~40                                     |
| 6                                   | ~90               | ~50                                    | ~50                                    | ~45                                     |
| 7                                   | ~95               | ~55                                    | ~55                                    | ~50                                     |
| 8                                   | ~98               | ~60                                    | ~60                                    | ~55                                     |
</details>

![](images/02ec894e0884edf0c53b0b4868e0b9cab710f830ca42701e802e8a8c350568cd.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | Byz-SGDnrm with CC | Byz-VR-MARINA with CC (B=512×8, p=0.2) | Byz-VR-MARINA with CC (B=512×8, p=0.1) | Byz-VR-MARINA with CC (B=512×8, p=0.05) |
| ----------------------------------- | ------------------ | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                   | 10                 | 10                                     | 10                                     | 10                                     |
| 1                                   | 60                 | 25                                     | 20                                     | 20                                     |
| 2                                   | 75                 | 30                                     | 25                                     | 25                                     |
| 3                                   | 80                 | 35                                     | 30                                     | 30                                     |
| 4                                   | 85                 | 40                                     | 35                                     | 35                                     |
| 5                                   | 88                 | 42                                     | 38                                     | 38                                     |
| 6                                   | 90                 | 43                                     | 40                                     | 40                                     |
| 7                                   | 92                 | 44                                     | 42                                     | 42                                     |
| 8                                   | 93                 | 45                                     | 43                                     | 43                                     |
| 9                                   | 94                 | 46                                     | 44                                     | 44                                     |
| 10                                  | 95                 | 47                                     | 45                                     | 45                                     |
</details>

Figure 5: Top-1 test accuracy w.r.t. gradient computation number of different methods when there are 3 workers under ALIE attack. The batch size for Byz-VR-MARINA is set to $512 \times 8$ .

has been shown in existing works (Defazio & Bottou, 2019) that using full gradients for variance reduction does not improve over SGD on deep learning models.

# E MORE EXPERIMENTAL RESULTS

# E.1 INDEPENDENT AND IDENTICALLY DISTRIBUTED (I.I.D.) CASE

Final top-1 test accuracy when using different batch size. We present the empirical results of ByzSGDm and ByzSGDnm with different batch size (ranging from $32 \times 8$ to $1024 \times 8$ ) under no attack or failure, under bit-flipping failure (Xie et al., 2019), ALIE attack and FoE attack in Table 7, Table 8, Table 9 and Table 10, respectively. Specifically, workers under bit-flipping failure will send the vectors that are -10 times the true values. The final top-1 test accuracy of the two methods under ALIE attack when NNM technique (Allouah et al., 2023) is used is presented in Table 11 below.

Table 7: The final top-1 test accuracy of ByzSGDm and ByzSGDnm with different batch size when there is no attack or failure 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td><td>1024×8</td></tr><tr><td>ByzSGDm + KR</td><td>91.08%</td><td>89.98%</td><td>89.71%</td><td>89.15%</td><td>86.15%</td><td>84.97%</td></tr><tr><td>ByzSGDnm + KR</td><td>91.00%</td><td>90.15%</td><td>89.23%</td><td>88.76%</td><td>87.83%</td><td>84.71%</td></tr><tr><td>ByzSGDm + GM</td><td>92.02%</td><td>91.50%</td><td>90.85%</td><td>89.26%</td><td>88.21%</td><td>86.52%</td></tr><tr><td>ByzSGDnm + GM</td><td>92.18%</td><td>91.81%</td><td>91.22%</td><td>89.93%</td><td>90.01%</td><td>88.08%</td></tr><tr><td>ByzSGDm + CM</td><td>92.30%</td><td>91.79%</td><td>90.43%</td><td>89.84%</td><td>87.27%</td><td>84.06%</td></tr><tr><td>ByzSGDnm + CM</td><td>92.29%</td><td>91.70%</td><td>91.15%</td><td>90.20%</td><td>89.06%</td><td>88.11%</td></tr><tr><td>ByzSGDm + CC</td><td>92.52%</td><td>91.74%</td><td>90.63%</td><td>89.40%</td><td>88.78%</td><td>85.50%</td></tr><tr><td>ByzSGDnm + CC</td><td>92.51%</td><td>91.91%</td><td>91.50%</td><td>90.00%</td><td>89.33%</td><td>88.47%</td></tr></table>

Table 8: The final top-1 test accuracy of ByzSGDm and ByzSGDnm with different batch size when there are 3 Byzantine workers under bit-flipping failure 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td><td>1024×8</td></tr><tr><td>ByzSGDm + KR</td><td>91.09%</td><td>90.30%</td><td>89.55%</td><td>88.37%</td><td>87.52%</td><td>85.68%</td></tr><tr><td>ByzSGDnm + KR</td><td>90.71%</td><td>90.14%</td><td>89.59%</td><td>88.89%</td><td>85.76%</td><td>82.22%</td></tr><tr><td>ByzSGDm + GM</td><td>88.97%</td><td>89.18%</td><td>88.61%</td><td>87.43%</td><td>85.72%</td><td>83.56%</td></tr><tr><td>ByzSGDnm + GM</td><td>88.64%</td><td>89.16%</td><td>88.89%</td><td>88.22%</td><td>87.78%</td><td>86.21%</td></tr><tr><td>ByzSGDm + CM</td><td>86.80%</td><td>87.11%</td><td>87.40%</td><td>86.37%</td><td>85.40%</td><td>81.23%</td></tr><tr><td>ByzSGDnm + CM</td><td>87.39%</td><td>88.12%</td><td>87.66%</td><td>86.76%</td><td>86.33%</td><td>85.52%</td></tr><tr><td>ByzSGDm + CC</td><td>88.92%</td><td>88.97%</td><td>88.78%</td><td>88.02%</td><td>86.54%</td><td>83.93%</td></tr><tr><td>ByzSGDnm + CC</td><td>88.81%</td><td>88.89%</td><td>88.96%</td><td>88.45%</td><td>87.56%</td><td>85.53%</td></tr></table>

Table 9: The final top-1 test accuracy of ByzSGDm and ByzSGDnm with different batch size when there are 3 Byzantine workers under ALIE attack 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td><td>1024×8</td></tr><tr><td>ByzSGDm + KR</td><td>38.55%</td><td>54.15%</td><td>55.98%</td><td>59.28%</td><td>83.42%</td><td>83.45%</td></tr><tr><td>ByzSGDnm + KR</td><td>43.47%</td><td>70.88%</td><td>80.20%</td><td>82.83%</td><td>85.12%</td><td>85.93%</td></tr><tr><td>ByzSGDm + GM</td><td>63.11%</td><td>70.88%</td><td>82.08%</td><td>87.62%</td><td>86.95%</td><td>84.75%</td></tr><tr><td>ByzSGDnm + GM</td><td>69.45%</td><td>83.23%</td><td>86.63%</td><td>88.66%</td><td>89.13%</td><td>88.16%</td></tr><tr><td>ByzSGDm + CM</td><td>33.11%</td><td>55.66%</td><td>66.38%</td><td>82.47%</td><td>83.25%</td><td>80.94%</td></tr><tr><td>ByzSGDnm + CM</td><td>61.28%</td><td>71.46%</td><td>80.24%</td><td>83.55%</td><td>86.03%</td><td>85.74%</td></tr><tr><td>ByzSGDm + CC</td><td>72.83%</td><td>79.45%</td><td>84.94%</td><td>87.25%</td><td>87.46%</td><td>83.70%</td></tr><tr><td>ByzSGDnm + CC</td><td>78.50%</td><td>83.91%</td><td>86.56%</td><td>88.32%</td><td>88.53%</td><td>87.89%</td></tr></table>

Table 10: The final top-1 test accuracy of ByzSGDm and ByzSGDnm with different batch size when there are 3 Byzantine workers under FoE attack 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td><td>1024×8</td></tr><tr><td>ByzSGDm + KR</td><td>10.00%</td><td>10.00%</td><td>10.00%</td><td>10.00%</td><td>10.00%</td><td>10.00%</td></tr><tr><td>ByzSGDnm + KR</td><td>10.00%</td><td>10.00%</td><td>10.00%</td><td>10.00%</td><td>10.00%</td><td>10.00%</td></tr><tr><td>ByzSGDm + GM</td><td>78.36%</td><td>81.98%</td><td>82.69%</td><td>82.20%</td><td>84.09%</td><td>78.90%</td></tr><tr><td>ByzSGDnm + GM</td><td>88.55%</td><td>88.75%</td><td>90.99%</td><td>90.23%</td><td>89.12%</td><td>88.38%</td></tr><tr><td>ByzSGDm + CM</td><td>83.97%</td><td>84.28%</td><td>84.01%</td><td>83.48%</td><td>79.16%</td><td>78.76%</td></tr><tr><td>ByzSGDnm + CM</td><td>84.12%</td><td>84.77%</td><td>85.23%</td><td>85.74%</td><td>84.65%</td><td>83.36%</td></tr><tr><td>ByzSGDm + CC</td><td>83.60%</td><td>84.26%</td><td>87.45%</td><td>88.48%</td><td>86.24%</td><td>81.36%</td></tr><tr><td>ByzSGDnm + CC</td><td>88.99%</td><td>90.07%</td><td>90.69%</td><td>90.54%</td><td>89.32%</td><td>88.20%</td></tr></table>

Table 11: The final top-1 test accuracy of ByzSGDm and ByzSGDnm with different batch size when there are 3 Byzantine workers under ALIE attack and NNM technique is used 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td><td>1024×8</td></tr><tr><td>ByzSGDm + KR</td><td>58.61%</td><td>65.96%</td><td>78.37%</td><td>85.71%</td><td>85.26%</td><td>83.97%</td></tr><tr><td>ByzSGDnm + KR</td><td>80.41%</td><td>83.85%</td><td>85.88%</td><td>87.15%</td><td>87.68%</td><td>87.09%</td></tr><tr><td>ByzSGDm + GM</td><td>72.58%</td><td>73.64%</td><td>78.02%</td><td>85.73%</td><td>85.37%</td><td>85.32%</td></tr><tr><td>ByzSGDnm + GM</td><td>79.50%</td><td>83.96%</td><td>86.42%</td><td>86.91%</td><td>88.09%</td><td>87.23%</td></tr><tr><td>ByzSGDm + CM</td><td>71.51%</td><td>78.15%</td><td>82.95%</td><td>86.06%</td><td>86.95%</td><td>85.24%</td></tr><tr><td>ByzSGDnm + CM</td><td>79.81%</td><td>84.14%</td><td>85.69%</td><td>87.65%</td><td>87.69%</td><td>87.11%</td></tr><tr><td>ByzSGDm + CC</td><td>76.48%</td><td>81.18%</td><td>84.63%</td><td>86.65%</td><td>85.98%</td><td>85.36%</td></tr><tr><td>ByzSGDnm + CC</td><td>79.91%</td><td>83.50%</td><td>87.00%</td><td>87.48%</td><td>87.59%</td><td>87.78%</td></tr></table>

More results about wall-clock time. In Section 5 in the main text, we have reported the wall-clock time of different methods for 160 epochs, which empirically shows that using large batch size has the bonus of training acceleration. To further support the conclusion, we also present the wall-clock time that different methods require to reach 50%, 75% and 85% top-1 test accuracy in Table 12, Table 13 and Table 14, respectively. As we can see from the results, ByzSGDnm requires less time to reach the target accuracy than ByzSGDm in most cases. Moreover, setting a relatively large batch size can significantly accelerate the training process.

Table 12: The wall-clock time for different methods to reach 50% top-1 test accuracy when there are 3 Byzantine workers under ALIE attack. The missing values mean that the target test accuracy is not reached in 160 epochs for the corresponding methods. 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td><td>1024×8</td></tr><tr><td>ByzSGDm + KR</td><td>-</td><td>157.10s</td><td>34.04s</td><td>33.93s</td><td>31.32s</td><td>70.77s</td></tr><tr><td>ByzSGDnm + KR</td><td>408.89s</td><td>31.46s</td><td>26.27s</td><td>27.66s</td><td>33.35s</td><td>44.31s</td></tr><tr><td>ByzSGDm + GM</td><td>74.21s</td><td>42.70s</td><td>26.27s</td><td>28.90s</td><td>38.77s</td><td>55.85s</td></tr><tr><td>ByzSGDnm + GM</td><td>61.09s</td><td>24.92s</td><td>17.43s</td><td>24.18s</td><td>23.24s</td><td>38.24s</td></tr><tr><td>ByzSGDm + CM</td><td>303.55s</td><td>110.69s</td><td>28.52s</td><td>37.20s</td><td>36.52s</td><td>73.44s</td></tr><tr><td>ByzSGDnm + CM</td><td>475.48s</td><td>41.17s</td><td>22.32s</td><td>27.35s</td><td>27.16s</td><td>45.66s</td></tr><tr><td>ByzSGDm + CC</td><td>102.17s</td><td>37.60s</td><td>17.89s</td><td>29.08s</td><td>27.11s</td><td>67.76s</td></tr><tr><td>ByzSGDnm + CC</td><td>63.56s</td><td>25.20s</td><td>20.75s</td><td>10.86s</td><td>21.45s</td><td>27.22s</td></tr></table>

Table 13: The wall-clock time for different methods to reach 75% top-1 test accuracy when there are 3 Byzantine workers under ALIE attack. The missing values mean that the target test accuracy is not reached in 160 epochs for the corresponding methods. 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td><td>1024×8</td></tr><tr><td>ByzSGDm + KR</td><td>-</td><td>-</td><td>-</td><td>-</td><td>133.69s</td><td>148.70s</td></tr><tr><td>ByzSGDnm + KR</td><td>-</td><td>-</td><td>328.49s</td><td>123.12s</td><td>130.45s</td><td>134.09s</td></tr><tr><td>ByzSGDm + GM</td><td>-</td><td>-</td><td>114.38s</td><td>59.19s</td><td>90.74s</td><td>124.03s</td></tr><tr><td>ByzSGDnm + GM</td><td>-</td><td>252.19s</td><td>66.13s</td><td>54.12s</td><td>45.07s</td><td>76.49s</td></tr><tr><td>ByzSGDm + CM</td><td>-</td><td>-</td><td>-</td><td>202.97s</td><td>110.25s</td><td>204.42s</td></tr><tr><td>ByzSGDnm + CM</td><td>-</td><td>-</td><td>322.83s</td><td>150.25s</td><td>63.85s</td><td>109.23s</td></tr><tr><td>ByzSGDm + CC</td><td>-</td><td>727.07s</td><td>77.19s</td><td>87.42s</td><td>86.53s</td><td>150.59s</td></tr><tr><td>ByzSGDnm + CC</td><td>1465.07s</td><td>112.49s</td><td>109.18s</td><td>56.20s</td><td>61.30s</td><td>81.28s</td></tr></table>

Table 14: The wall-clock time for different methods to reach 85% top-1 test accuracy when there are 3 Byzantine workers under ALIE attack. The missing values mean that the target test accuracy is not reached in 160 epochs for the corresponding methods. 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td><td>1024×8</td></tr><tr><td>ByzSGDm + KR</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ByzSGDnm + KR</td><td>-</td><td>-</td><td>-</td><td>-</td><td>279.88s</td><td>248.95s</td></tr><tr><td>ByzSGDm + GM</td><td>-</td><td>-</td><td>-</td><td>269.31s</td><td>218.80s</td><td>-</td></tr><tr><td>ByzSGDnm + GM</td><td>-</td><td>-</td><td>408.89s</td><td>201.76s</td><td>176.06s</td><td>171.29s</td></tr><tr><td>ByzSGDm + CM</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ByzSGDnm + CM</td><td>-</td><td>-</td><td>-</td><td>-</td><td>240.10s</td><td>257.50s</td></tr><tr><td>ByzSGDm + CC</td><td>-</td><td>-</td><td>-</td><td>292.38s</td><td>207.57s</td><td>-</td></tr><tr><td>ByzSGDnm + CC</td><td>-</td><td>-</td><td>426.95s</td><td>216.13s</td><td>176.83s</td><td>188.12s</td></tr></table>

Top-1 test accuracy w.r.t. gradient computation number / wall-clock time. We present the top-1 test accuracy w.r.t. gradient computation number and wall-clock time for different methods when there are 3 workers under ALIE attack in Figure 6 and Figure 7, respectively. The empirical results show that using a relatively large batch size $B = 512 \times 8$ can lead to a stabler training process and higher final top-1 test accuracy. Moreover, under each setting of aggregators and batch size in the experiment, ByzSGDnm outperforms ByzSGDm in final top-1 test accuracy.

![](images/f9e00ec76bdc5d1bedd6510daa0d97c2463bfc0c99d2b309d95a36481f0e64f3.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with KR (B=512×8) | ByzSGDnm with KR (B=512×8) | ByzSGDnm with KR (B=128×8) | ByzSGDnm with KR (B=128×8) | ByzSGDnm with KR (B=32×8) | ByzSGDnm with KR (B=32×8) |
| ----------------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| 0                                   | 10                           | 10                           | 10                           | 10                           | 10                           | 10                           |
| 1                                   | 40                           | 40                           | 40                           | 40                           | 40                           | 40                           |
| 2                                   | 60                           | 60                           | 60                           | 60                           | 60                           | 60                           |
| 3                                   | 70                           | 70                           | 70                           | 70                           | 70                           | 70                           |
| 4                                   | 75                           | 75                           | 75                           | 75                           | 75                           | 75                           |
| 5                                   | 80                           | 80                           | 80                           | 80                           | 80                           | 80                           |
| 6                                   | 85                           | 85                           | 85                           | 85                           | 85                           | 85                           |
| 7                                   | 90                           | 90                           | 90                           | 90                           | 90                           | 90                           |
| 8                                   | 95                           | 95                           | 95                           | 95                           | 95                           | 95                           |
| 9                                   | 98                           | 98                           | 98                           | 98                           | 98                           | 98                           |
| 10                                  | 100                          | 100                          | 100                          | 100                          | 100                          | 100                          |
</details>

![](images/78f81c913b348f6b3aa469685d31b1fb7dec5b4ef04fca13d33256550a41f748.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDm with GM (B=512×8) | ByzSGDm with GM (B=512×8) | ByzSGDm with GM (B=128×8) | ByzSGDm with GM (B=128×8) | ByzSGDm with GM (B=32×8) | ByzSGDm with GM (B=32×8) |
| ---------------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | ------------------------- | ------------------------- |
| 0                                  | ~10                        | ~10                        | ~10                        | ~10                        | ~10                       | ~10                       |
| 1                                  | ~60                        | ~60                        | ~60                        | ~60                        | ~60                       | ~60                       |
| 2                                  | ~70                        | ~70                        | ~70                        | ~70                        | ~70                       | ~70                       |
| 3                                  | ~75                        | ~75                        | ~75                        | ~75                        | ~75                       | ~75                       |
| 4                                  | ~80                        | ~80                        | ~80                        | ~80                        | ~80                       | ~80                       |
| 5                                  | ~85                        | ~85                        | ~85                        | ~85                        | ~85                       | ~85                       |
| 6                                  | ~90                        | ~90                        | ~90                        | ~90                        | ~90                       | ~90                       |
| 7                                  | ~90                        | ~90                        | ~90                        | ~90                        | ~90                       | ~90                       |
| 8                                  | ~90                        | ~90                        | ~90                        | ~90                        | ~90                       | ~90                       |
</details>

![](images/a24f3541446142ba5a9f838a1672ec8d56a31cc4b76986616d98bb5b36d4aa7f.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with CM (B=512×8) | ByzSGDnm with CM (B=512×8) | ByzSGDnm with CM (B=128×8) | ByzSGDnm with CM (B=128×8) | ByzSGDnm with CM (B=32×8) | ByzSGDnm with CM (B=32×8) |
| ---------------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| 0                                  | 10                           | 10                           | 10                           | 10                           | 10                           | 10                           |
| 1                                  | ~60                          | ~65                          | ~55                          | ~60                          | ~45                          | ~50                          |
| 2                                  | ~70                          | ~75                          | ~65                          | ~70                          | ~55                          | ~60                          |
| 3                                  | ~75                          | ~80                          | ~70                          | ~75                          | ~60                          | ~65                          |
| 4                                  | ~80                          | ~85                          | ~75                          | ~80                          | ~65                          | ~70                          |
| 5                                  | ~85                          | ~90                          | ~80                          | ~85                          | ~70                          | ~75                          |
| 6                                  | ~90                          | ~95                          | ~85                          | ~90                          | ~75                          | ~80                          |
| 7                                  | ~95                          | ~98                          | ~90                          | ~95                          | ~80                          | ~85                          |
| 8                                  | ~98                          | ~99                          | ~95                          | ~98                          | ~85                          | ~90                          |
</details>

![](images/e4a43c9b72dce7501dc31b551fefedca74d49a08e1abf2ff19845a333c06b97b.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with CC (B=512×8) | ByzSGDnm with CC (B=512×8) | ByzSGDnm with CC (B=128×8) | ByzSGDnm with CC (B=128×8) | ByzSGDnm with CC (B=32×8) | ByzSGDnm with CC (B=32×8) |
| ----------------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| 0                                   | 10                           | 10                           | 10                           | 10                           | 10                           | 10                           |
| 1                                   | 60                           | 65                           | 60                           | 65                           | 60                           | 65                           |
| 2                                   | 70                           | 75                           | 70                           | 75                           | 70                           | 75                           |
| 3                                   | 75                           | 80                           | 75                           | 80                           | 75                           | 80                           |
| 4                                   | 80                           | 85                           | 80                           | 85                           | 80                           | 85                           |
| 5                                   | 85                           | 90                           | 85                           | 90                           | 85                           | 90                           |
| 6                                   | 90                           | 95                           | 90                           | 95                           | 90                           | 95                           |
| 7                                   | 95                           | 98                           | 95                           | 98                           | 95                           | 98                           |
| 8                                   | 98                           | 99                           | 98                           | 99                           | 98                           | 99                           |
</details>

Figure 6: Top-1 test accuracy w.r.t. gradient computation number of ByzSGDnm and ByzSGDm with different batch size when there are 3 workers under ALIE attack

![](images/96b091fff65c005976377ad5061b3c51a7e7ca642ee07b8556e9d03b8ce303ae.jpg)

<details>
<summary>line</summary>

| Wall-clock time (second) | BYzSGDnm with KR (B=512×8) | BYzSGDnm with KR (B=512×8) | BYzSGDnm with KR (B=128×8) | BYzSGDnm with KR (B=128×8) | BYzSGDnm with KR (B=32×8) | BYzSGDnm with KR (B=32×8) |
| ------------------------ | --------------------------- | --------------------------- | --------------------------- | --------------------------- | -------------------------- | -------------------------- |
| 0                        | 10                          | 10                          | 10                          | 10                          | 10                         | 10                         |
| 500                      | 80                          | 75                          | 60                          | 55                          | 40                         | 35                         |
</details>

![](images/6428c6d57a85f3972e7084175100be3bafdec8b26d1a956a31c909b60e5aeee7.jpg)

<details>
<summary>line</summary>

| Wall-clock time (second) | ByzSGDnm with GM (B=512×8) | ByzSGDnm with GM (B=512×8) | ByzSGDnm with GM (B=128×8) | ByzSGDnm with GM (B=128×8) | ByzSGDnm with GM (B=32×8) | ByzSGDnm with GM (B=32×8) |
| ------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- | ---------------------------- |
| 0                         | 10                           | 10                           | 10                           | 10                           | 10                           | 10                           |
| 500                       | 90                           | 85                           | 80                           | 75                           | 60                           | 55                           |
</details>

![](images/332b447516338a9fb37f10b312a3a34162f4252ba44e02c9ddb75df69d748c66.jpg)

<details>
<summary>line</summary>

| Wall-clock time (second) | ByzSGDnm with CM (B=512×8) | ByzSGDnm with CM (B=512×8) | ByzSGDnm with CM (B=128×8) | ByzSGDnm with CM (B=128×8) | ByzSGDnm with CM (B=32×8) | ByzSGDnm with CM (B=32×8) |
| ------------------------ | --------------------------- | --------------------------- | --------------------------- | --------------------------- | -------------------------- | -------------------------- |
| 0                        | 10                          | 10                          | 10                          | 10                          | 10                         | 10                         |
| 50                       | 40                          | 40                          | 40                          | 40                          | 40                         | 40                         |
| 100                      | 60                          | 60                          | 60                          | 60                          | 60                         | 60                         |
| 150                      | 70                          | 70                          | 70                          | 70                          | 70                         | 70                         |
| 200                      | 80                          | 80                          | 80                          | 80                          | 80                         | 80                         |
| 250                      | 85                          | 85                          | 85                          | 85                          | 85                         | 85                         |
| 300                      | 90                          | 90                          | 90                          | 90                          | 90                         | 90                         |
| 350                      | 95                          | 95                          | 95                          | 95                          | 95                         | 95                         |
| 400                      | 98                          | 98                          | 98                          | 98                          | 98                         | 98                         |
| 450                      | 99                          | 99                          | 99                          | 99                          | 99                         | 99                         |
| 500                      | 100                         | 100                         | 100                         | 100                         | 100                        | 100                        |
</details>

![](images/a2b340366d93a748b5ce55583b4359becb1d2ce9025323ac6503c27221233959.jpg)

<details>
<summary>line</summary>

| Wall-clock time (second) | ByzSGDnm with CC (B=512×8) | ByzSGDnm with CC (B=512×8) | ByzSGDnm with CC (B=128×8) | ByzSGDnm with CC (B=128×8) | ByzSGDnm with CC (B=32×8) | ByzSGDnm with CC (B=32×8) |
| ------------------------ | --------------------------- | --------------------------- | --------------------------- | --------------------------- | -------------------------- | -------------------------- |
| 0                        | 10                          | 10                          | 10                          | 10                          | 10                         | 10                         |
| 50                       | 60                          | 65                          | 55                          | 60                          | 40                         | 45                         |
| 100                      | 70                          | 75                          | 65                          | 70                          | 50                         | 55                         |
| 150                      | 75                          | 80                          | 70                          | 75                          | 55                         | 60                         |
| 200                      | 80                          | 85                          | 75                          | 80                          | 60                         | 65                         |
| 250                      | 85                          | 90                          | 80                          | 85                          | 65                         | 70                         |
| 300                      | 90                          | 95                          | 85                          | 90                          | 70                         | 75                         |
| 350                      | 95                          | 98                          | 90                          | 95                          | 75                         | 80                         |
| 400                      | 98                          | 99                          | 95                          | 98                          | 80                         | 85                         |
| 450                      | 99                          | 99.5                        | 98                          | 99                          | 85                         | 90                         |
| 500                      | 100                         | 100                         | 100                         | 100                         | 90                         | 95                         |
</details>

Figure 7: Top-1 test accuracy w.r.t. wall-clock time of ByzSGDnm and ByzSGDm with different batch size when there are 3 workers under ALIE attack

![](images/4c3966ce80913bcaca19b2a0c7c2374e280fa7074a1407612de11b30cb148687.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | F(w) - ByzSGDm with KR (B=512×8) | F(w) - ByzSGDm with KR (B=128×8) | F(w) - ByzSGDm with KR (B=32×8) | F(w) - ByzSGDm with KR (B=32×8) |
| ----------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- |
| 0                                   | 5.0                                | 5.0                                | 5.0                                | 5.0                                |
| 1                                   | ~2.5                               | ~2.5                               | ~2.5                               | ~2.5                               |
| 2                                   | ~1.5                               | ~1.5                               | ~1.5                               | ~1.5                               |
| 3                                   | ~1.0                               | ~1.0                               | ~1.0                               | ~1.0                               |
| 4                                   | ~0.8                               | ~0.8                               | ~0.8                               | ~0.8                               |
| 5                                   | ~0.6                               | ~0.6                               | ~0.6                               | ~0.6                               |
| 6                                   | ~0.5                               | ~0.5                               | ~0.5                               | ~0.5                               |
| 7                                   | ~0.4                               | ~0.4                               | ~0.4                               | ~0.4                               |
| 8                                   | ~0.3                               | ~0.3                               | ~0.3                               | ~0.3                               |
</details>

![](images/f57b1663cdb8fa1011897345e32b55ec16b4cd33454d0b417e3644da4ecb1b4c.jpg)

<details>
<summary>line</summary>

| Gradient computation number (x10^6) | F(w) - ByzSGDnm with GM (B=512×8) | F(w) - ByzSGDnm with GM (B=512×8) | F(w) - ByzSGDnm with GM (B=128×8) | F(w) - ByzSGDnm with GM (B=128×8) | F(w) - ByzSGDnm with GM (B=32×8) | F(w) - ByzSGDnm with GM (B=32×8) |
| ------------------------------------ | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- |
| 0                                    | 5.0                                | 5.0                                | 5.0                                | 5.0                                | 5.0                                | 5.0                                |
| 1                                    | 1.5                                | 1.5                                | 1.5                                | 1.5                                | 1.5                                | 1.5                                |
| 2                                    | 0.5                                | 0.5                                | 0.5                                | 0.5                                | 0.5                                | 0.5                                |
| 3                                    | 0.2                                | 0.2                                | 0.2                                | 0.2                                | 0.2                                | 0.2                                |
| 4                                    | 0.1                                | 0.1                                | 0.1                                | 0.1                                | 0.1                                | 0.1                                |
| 5                                    | 0.05                               | 0.05                               | 0.05                               | 0.05                               | 0.05                               | 0.05                               |
| 6                                    | 0.02                               | 0.02                               | 0.02                               | 0.02                               | 0.02                               | 0.02                               |
| 7                                    | 0.01                               | 0.01                               | 0.01                               | 0.01                               | 0.01                               | 0.01                               |
| 8                                    | 0.005                              | 0.005                              | 0.005                              | 0.005                              | 0.005                              | 0.005                              |
</details>

![](images/c6ad3f3a032ac68fa0911be57ba6963f509b317f63dc6a762a0c253b50b45816.jpg)

<details>
<summary>line</summary>

| Gradient computation number | F(w) for B=512×8 | F(w) for B=512×8 | F(w) for B=128×8 | F(w) for B=128×8 | F(w) for B=32×8 | F(w) for B=32×8 |
| ---------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 0                            | 5.0              | 5.0              | 3.0              | 3.0              | 2.0              | 2.0              |
| 100000                       | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              |
| 200000                       | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              |
| 300000                       | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              |
| 400000                       | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              |
| 500000                       | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              |
| 600000                       | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              |
| 700000                       | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              |
| 800000                       | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              | 0.5              |
</details>

![](images/83585caa05080cfa03a32ea817fcaf83cf54aecb2903172273b02d22c73535e4.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | F(w) - ByzSGDnm with CC (B=512×8) | F(w) - ByzSGDm with CC (B=512×8) | F(w) - ByzSGDnm with CC (B=128×8) | F(w) - ByzSGDnm with CC (B=128×8) | F(w) - ByzSGDnm with CC (B=32×8) | F(w) - ByzSGDnm with CC (B=32×8) |
| ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- |
| 0                                  | 5.0                                | 5.0                                | 5.0                                | 5.0                                | 5.0                                | 5.0                                |
| 1                                  | 4.5                                | 4.5                                | 4.5                                | 4.5                                | 4.5                                | 4.5                                |
| 2                                  | 4.0                                | 4.0                                | 4.0                                | 4.0                                | 4.0                                | 4.0                                |
| 3                                  | 3.5                                | 3.5                                | 3.5                                | 3.5                                | 3.5                                | 3.5                                |
| 4                                  | 3.0                                | 3.0                                | 3.0                                | 3.0                                | 3.0                                | 3.0                                |
| 5                                  | 2.5                                | 2.5                                | 2.5                                | 2.5                                | 2.5                                | 2.5                                |
| 6                                  | 2.0                                | 2.0                                | 2.0                                | 2.0                                | 2.0                                | 2.0                                |
| 7                                  | 1.5                                | 1.5                                | 1.5                                | 1.5                                | 1.5                                | 1.5                                |
| 8                                  | 1.0                                | 1.0                                | 1.0                                | 1.0                                | 1.0                                | 1.0                                |
</details>

Figure 8: Training loss $F(\mathbf{w})$ w.r.t. gradient computation number of ByzSGDnm and ByzSGDm with different batch size when there are 3 workers under ALIE attack

Meanwhile, we also present the training loss w.r.t. gradient computation number in Figure 8. The results further verify the effectiveness of large batch size and ByzSGDnm.

# E.2 NON-I.I.D. CASE

Although we mainly focus on the i.i.d. case in this paper, we also provide some empirical results in non-i.i.d. cases in this section. Specifically, we randomly sample from the training set of CIFAR-10 dataset according to the Dirichlet distribution with hyper-parameter 1.0. The number of training instances for each class on each worker is presented in Table E.1 below.

Table 15: The number of training instances for each class on each worker 

<table><tr><td>Class label</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>Total</td></tr><tr><td>Worker 0</td><td>523</td><td>334</td><td>62</td><td>582</td><td>491</td><td>2502</td><td>721</td><td>148</td><td>568</td><td>319</td><td>6250</td></tr><tr><td>Worker 1</td><td>492</td><td>898</td><td>697</td><td>159</td><td>83</td><td>92</td><td>787</td><td>2415</td><td>67</td><td>560</td><td>6250</td></tr><tr><td>Worker 2</td><td>754</td><td>465</td><td>459</td><td>426</td><td>2365</td><td>167</td><td>121</td><td>815</td><td>678</td><td>0</td><td>6250</td></tr><tr><td>Worker 3</td><td>0</td><td>0</td><td>61</td><td>159</td><td>749</td><td>304</td><td>364</td><td>671</td><td>688</td><td>3254</td><td>6250</td></tr><tr><td>Worker 4</td><td>1106</td><td>515</td><td>1692</td><td>652</td><td>593</td><td>611</td><td>553</td><td>70</td><td>0</td><td>458</td><td>6250</td></tr><tr><td>Worker 5</td><td>2004</td><td>105</td><td>105</td><td>2608</td><td>29</td><td>0</td><td>0</td><td>345</td><td>1054</td><td>0</td><td>6250</td></tr><tr><td>Worker 6</td><td>59</td><td>2146</td><td>1561</td><td>314</td><td>564</td><td>395</td><td>324</td><td>135</td><td>752</td><td>0</td><td>6250</td></tr><tr><td>Worker 7</td><td>62</td><td>537</td><td>363</td><td>100</td><td>126</td><td>929</td><td>2130</td><td>401</td><td>1193</td><td>409</td><td>6250</td></tr><tr><td>Total</td><td>5000</td><td>5000</td><td>5000</td><td>5000</td><td>5000</td><td>5000</td><td>5000</td><td>5000</td><td>5000</td><td>5000</td><td>50000</td></tr></table>

Moreover, we replace the batch normalization layers in the ResNet-20 deep learning model (He et al., 2016) with group normalization layers (Wu & He, 2018) as suggested in existing works (Hsieh et al., 2020). We empirically compare ByzSGDnm with ByzSGDm and Byz-VR-MARINA (Gorbunov et al., 2023). We try the batch size $32 \times 8$ , $128 \times 8$ and $512 \times 8$ for each method. Among the 8 workers, one worker (worker 0) is under ALIE attack. The other hyper-parameter settings for ByzSGDm and ByzSGDnm are the same as those in Section 5 of the main text. The other hyper-parameter settings for Byz-VR-MARINA are the same as those in Appendix D.5.

Comparison among the methods in the non-i.i.d. case. The top-1 test accuracy w.r.t. gradient computation number is illustrated in Figure 9. We also present the final top-1 test accuracy of ByzSGDm and ByzSGDnm with different batch size in Table 16. As the empirical results show, ByzSGDnm still outperforms ByzSGDm and Byz-VR-MARINA under this non-i.i.d. setting. Moreover, for ByzSGDnm and ByzSGDm, increasing batch size can still increase top-1 test accuracy under this non-i.i.d. setting.

Further comparison when NNM technique is used. We also test the empirical performance of the three methods (ByzSGDnm, ByzSGDm and Byz-VR-MARINA) under the non-i.i.d. setting when nearest neighbour mixing (NNM) (Allouah et al., 2023) technique is used. The top-1 test accuracy w.r.t. gradient computation number when there is 1 worker under ALIE attack is illustrated in Figure 10. We also present the final top-1 test accuracy of ByzSGDm and ByzSGDnm with different batch size in Table 17. Compared to the case without NNM, the final top-1 accuracy of ByzSGDnm and ByzSGDm significantly increases (from about 35% to about 80%) when using NNM technique. On the contrary, the final top-1 test accuracy of Byz-VR-MARINA is still around 20% when NNM is used. In addition, for ByzSGDm and ByzSGDnm, the batch size that leads to the best top-1 test accuracy decreases when combined with NNM. A possible reason is that when combined with NNM, the term c in equation (4) decreases, leading to the decrease of $B^{*}$ . However, since the bias should also be taken into consideration, it requires further work to study the effect of batch size in non-i.i.d. cases.

Meanwhile, as the results in Table 17 and in Figure 10 show, ByzSGDnm still outperforms ByzSGDm when combined with NNM in non-i.i.d. cases. The empirical results show that in non-i.i.d. cases, ByzSGDnm is still a promising choice. Since we mainly focus on the i.i.d. case in this work, we will further study the behavior of ByzSGDnm in non-i.i.d. cases in future work.

Table 16: The final top-1 test accuracy of ByzSGDm and ByzSGDnm with GM in the non-i.i.d. setting when there is 1 Byzantine worker under ALIE attack. 

<table><tr><td>Batch size</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td><td>1024×8</td></tr><tr><td>ByzSGDm</td><td>10.16%</td><td>23.33%</td><td>27.78%</td><td>29.63%</td><td>32.90%</td><td>32.53%</td></tr><tr><td>ByzSGDnm</td><td>23.87%</td><td>26.78%</td><td>28.42%</td><td>30.04%</td><td>35.24%</td><td>36.25%</td></tr></table>

![](images/ebee594243865107764eb6e7621e9369db239f65d1661675e7c07824221bebd3.jpg)

<details>
<summary>line</summary>

| Gradient computation number | ByzSGDm with GM (B=512×8) | ByzSGDm with GM (B=512×8) | ByzSGDm with GM (B=128×8) | ByzSGDm with GM (B=128×8) | ByzSGDm with GM (B=32×8) | ByzSGDm with GM (B=32×8) |
| ---------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| 0                            | 10                         | 10                         | 10                         | 10                         | 10                         | 10                         |
| 1                            | 30                         | 30                         | 25                         | 25                         | 20                         | 20                         |
| 2                            | 35                         | 35                         | 28                         | 28                         | 25                         | 25                         |
| 3                            | 36                         | 36                         | 29                         | 29                         | 26                         | 26                         |
| 4                            | 37                         | 37                         | 30                         | 30                         | 27                         | 27                         |
| 5                            | 36                         | 36                         | 29                         | 29                         | 26                         | 26                         |
| 6                            | 35                         | 35                         | 28                         | 28                         | 25                         | 25                         |
| 7                            | 34                         | 34                         | 27                         | 27                         | 24                         | 24                         |
| 8                            | 33                         | 33                         | 26                         | 26                         | 23                         | 23                         |
</details>

(a) ByzSGDnm v.s. ByzSGDm

![](images/f17165f7c9234ae58cc662fd338c179f5bf831b5f4e0989d057861a8f71878ec.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | Top-1 test accuracy (B=512×8) | Top-1 test accuracy (B=32×8, p=0.05) | Top-1 test accuracy (B=128×8, p=0.05) | Top-1 test accuracy (B=512×8, p=0.05) |
| ----------------------------------- | ------------------------------ | ------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                   | 10                             | 10                                    | 10                                     | 10                                     |
| 1                                   | 30                             | 12                                    | 12                                     | 12                                     |
| 2                                   | 35                             | 16                                    | 16                                     | 16                                     |
| 3                                   | 37                             | 18                                    | 18                                     | 18                                     |
| 4                                   | 36                             | 22                                    | 20                                     | 20                                     |
| 5                                   | 35                             | 24                                    | 22                                     | 22                                     |
| 6                                   | 36                             | 25                                    | 23                                     | 23                                     |
| 7                                   | 35                             | 26                                    | 24                                     | 24                                     |
| 8                                   | 35                             | 26                                    | 24                                     | 24                                     |
</details>

(b) ByzSGDnm v.s. Byz-VR-MARINA (p = 0.05)

![](images/00e336748c7ecad0db77559d4307b1fce4fab4b66a446e4339c8be3e06d5b662.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with GM (B=512×8) | Byz-VR-MARINA with GM (B=32×8, p=0.1) | Byz-VR-MARINA with GM (B=128×8, p=0.1) | Byz-VR-MARINA with GM (B=512×8, p=0.1) |
| ----------------------------------- | ---------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                   | 10                           | 10                                     | 10                                     | 10                                     |
| 1                                   | 30                           | 15                                     | 12                                     | 10                                     |
| 2                                   | 35                           | 18                                     | 15                                     | 10                                     |
| 3                                   | 36                           | 17                                     | 16                                     | 10                                     |
| 4                                   | 35                           | 16                                     | 17                                     | 10                                     |
| 5                                   | 36                           | 15                                     | 18                                     | 10                                     |
| 6                                   | 35                           | 20                                     | 22                                     | 10                                     |
| 7                                   | 35                           | 22                                     | 23                                     | 10                                     |
| 8                                   | 35                           | 25                                     | 24                                     | 10                                     |
</details>

(c) ByzSGDnm v.s. Byz-VR-MARINA (p = 0.1)

![](images/8e763e1bb5a58f7172a532c95714c818a7163e3b56a213fd7de18cc95d342f21.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDnm with GM (B=512×8) | Byz-VR-MARINA with GM (B=32×8, p=0.2) | Byz-VR-MARINA with GM (B=128×8, p=0.2) | Byz-VR-MARINA with GM (B=512×8, p=0.2) |
| ----------------------------------- | --------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| 0                                   | 10                          | 10                                     | 10                                     | 10                                     |
| 1                                   | ~35                         | ~15                                    | ~10                                    | ~10                                    |
| 2                                   | ~35                         | ~17                                    | ~10                                    | ~10                                    |
| 3                                   | ~35                         | ~18                                    | ~18                                    | ~10                                    |
| 4                                   | ~35                         | ~19                                    | ~20                                    | ~10                                    |
| 5                                   | ~35                         | ~19                                    | ~22                                    | ~10                                    |
| 6                                   | ~35                         | ~18                                    | ~24                                    | ~10                                    |
| 7                                   | ~35                         | ~16                                    | ~25                                    | ~10                                    |
| 8                                   | ~35                         | ~16                                    | ~25                                    | ~10                                    |
</details>

(d) ByzSGDnm v.s. Byz-VR-MARINA (p = 0.2)   
Figure 9: Top-1 test accuracy w.r.t. gradient computation number when there is 1 worker under ALIE attack under the non-i.i.d. setting

Table 17: The final top-1 test accuracy of ByzSGDm and ByzSGDnm with GM in the non-i.i.d. setting when there is 1 Byzantine worker under ALIE attack and NNM technique is used. 

<table><tr><td>Batch size</td><td>16×8</td><td>32×8</td><td>64×8</td><td>128×8</td><td>256×8</td><td>512×8</td></tr><tr><td>ByzSGDm</td><td>80.46%</td><td>80.56%</td><td>78.29%</td><td>74.10%</td><td>66.71%</td><td>58.78%</td></tr><tr><td>ByzSGDnm</td><td>81.11%</td><td>81.74%</td><td>80.08%</td><td>77.98%</td><td>74.65%</td><td>69.05%</td></tr></table>

![](images/8edcaa53429afe49dd66ee3c4222c68346e9379989d369c3c146ba877ade78ca.jpg)

<details>
<summary>line</summary>

| Gradient computation number (×10⁶) | ByzSGDm with GM (B=512×8) | ByzSGDm with GM (B=512×8) | ByzSGDm with GM (B=128×8) | ByzSGDm with GM (B=128×8) | ByzSGDm with GM (B=32×8) | ByzSGDm with GM (B=32×8) |
| ---------------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| 0                                  | 10                         | 10                         | 10                         | 10                         | 10                         | 10                         |
| 1                                  | 30                         | 35                         | 40                         | 45                         | 50                         | 55                         |
| 2                                  | 45                         | 50                         | 60                         | 65                         | 70                         | 75                         |
| 3                                  | 55                         | 60                         | 70                         | 75                         | 80                         | 85                         |
| 4                                  | 60                         | 65                         | 75                         | 80                         | 85                         | 90                         |
| 5                                  | 65                         | 70                         | 80                         | 85                         | 90                         | 95                         |
| 6                                  | 70                         | 75                         | 85                         | 90                         | 95                         | 98                         |
| 7                                  | 75                         | 80                         | 90                         | 95                         | 98                         | 99                         |
| 8                                  | 80                         | 85                         | 95                         | 98                         | 99                         | 100                        |
</details>

(a) ByzSGDnm v.s. ByzSGDm

![](images/dbef3571db632985beecab6d7b38b2169b2af845efe45f4949e78e2e46183278.jpg)

<details>
<summary>line</summary>

| Gradient computation number | ByzSGDnm with GM (B=32×8) | Byz-VR-MARINA with GM (B=32×8, p=0.05) | Byz-VR-MARINA with GM (B=128×8, p=0.05) | Byz-VR-MARINA with GM (B=512×8, p=0.05) |
| ---------------------------- | -------------------------- | -------------------------------------- | ---------------------------------------- | --------------------------------------- |
| 0                            | 10                         | 10                                     | 10                                       | 10                                      |
| 1e6                          | 70                         | 15                                     | 15                                       | 10                                      |
| 2e6                          | 75                         | 20                                     | 15                                       | 10                                      |
| 3e6                          | 78                         | 22                                     | 18                                       | 10                                      |
| 4e6                          | 79                         | 20                                     | 18                                       | 10                                      |
| 5e6                          | 80                         | 18                                     | 18                                       | 10                                      |
| 6e6                          | 80                         | 18                                     | 18                                       | 10                                      |
| 7e6                          | 80                         | 18                                     | 18                                       | 10                                      |
| 8e6                          | 80                         | 18                                     | 18                                       | 10                                      |
</details>

(b) ByzSGDnm v.s. Byz-VR-MARINA (p = 0.05)

![](images/8e7bad0ecd2faa6914ab1ee8e634c27c4953eeea9b24b9489b796a15de05d70b.jpg)

<details>
<summary>line</summary>

| Gradient computation number | ByzSGDnm with GM (B=32×8) | Byz-VR-MARINA with GM (B=32×8, p=0.1) | Byz-VR-MARINA with GM (B=128×8, p=0.1) | Byz-VR-MARINA with GM (B=512×8, p=0.1) |
| ---------------------------- | -------------------------- | -------------------------------------- | --------------------------------------- | --------------------------------------- |
| 0                            | 10                         | 10                                     | 10                                      | 10                                      |
| 1                            | 60                         | 15                                     | 15                                      | 15                                      |
| 2                            | 70                         | 15                                     | 15                                      | 20                                      |
| 3                            | 75                         | 15                                     | 15                                      | 15                                      |
| 4                            | 78                         | 15                                     | 15                                      | 15                                      |
| 5                            | 80                         | 15                                     | 15                                      | 15                                      |
| 6                            | 80                         | 15                                     | 20                                      | 20                                      |
| 7                            | 80                         | 15                                     | 20                                      | 20                                      |
| 8                            | 80                         | 15                                     | 20                                      | 20                                      |
</details>

(c) ByzSGDnm v.s. Byz-VR-MARINA $(p = 0.1)$

![](images/a1a95d73af6c75846065d68f926bbe47a67447c6d793a49ea09ed99a7bf06c51.jpg)

<details>
<summary>line</summary>

| Gradient computation number | ByzSGDnm with GM (B=32×8) | Byz-VR-MARINA with GM (B=32×8, p=0.2) | Byz-VR-MARINA with GM (B=128×8, p=0.2) | Byz-VR-MARINA with GM (B=512×8, p=0.2) |
| ---------------------------- | -------------------------- | -------------------------------------- | --------------------------------------- | --------------------------------------- |
| 0                            | 10                         | 10                                     | 10                                      | 10                                      |
| 1                            | 65                         | 12                                     | 12                                      | 15                                      |
| 2                            | 70                         | 15                                     | 15                                      | 18                                      |
| 3                            | 75                         | 18                                     | 18                                      | 20                                      |
| 4                            | 78                         | 20                                     | 20                                      | 22                                      |
| 5                            | 79                         | 20                                     | 20                                      | 20                                      |
| 6                            | 80                         | 20                                     | 20                                      | 20                                      |
| 7                            | 80                         | 20                                     | 20                                      | 20                                      |
| 8                            | 80                         | 20                                     | 20                                      | 20                                      |
</details>

(d) ByzSGDnm v.s. Byz-VR-MARINA $(p = 0.2)$   
Figure 10: Top-1 test accuracy w.r.t. gradient computation number when there is 1 worker under ALIE attack under the non-i.i.d. setting and NNM technique is used