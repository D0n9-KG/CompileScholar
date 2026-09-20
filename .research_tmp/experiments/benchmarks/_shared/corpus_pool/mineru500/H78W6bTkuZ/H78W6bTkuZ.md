# MetaOptimize: A Framework for Optimizing Step Sizes and Other Meta-parameters

Arsalan Sharifnassab $^{12}$ Saber Salehkaleybar $^{3}$ Richard Sutton $^{2}$

# Abstract

We address the challenge of optimizing meta-parameters (hyperparameters) in machine learning, a key factor for efficient training and high model performance. Rather than relying on expensive meta-parameter search methods, we introduce MetaOptimize: a dynamic approach that adjusts meta-parameters, particularly step sizes (also known as learning rates), during training. More specifically, MetaOptimize can wrap around any first-order optimization algorithm, tuning step sizes on the fly to minimize a specific form of regret that considers the long-term impact of step sizes on training, through a discounted sum of future losses. We also introduce lower-complexity variants of MetaOptimize that, in conjunction with its adaptability to various optimization algorithms, achieve performance comparable to those of the best hand-crafted learning rate schedules across diverse machine learning tasks.

# 1. Introduction

Optimization algorithms used in machine learning involve meta-parameters (i.e., hyperparameters) that substantially influence their performance. These meta-parameters are typically identified through a search process, such as grid search or other trial-and-error methods, prior to training. However, the computational cost of this meta-parameter search is significantly larger than that of training with optimal meta-parameters (Dahl et al., 2023; Jin, 2022). Meta-parameter optimization seeks to streamline this process by concurrently adjusting meta-parameters during training, moving away from the computationally expensive and often sub-

$^{1}$ Openmind Research Institute, Canada $^{2}$ Department of Computing Science, University of Alberta, Edmonton, Canada $^{3}$ Leiden Institute of Advanced Computer Science, Leiden University, Leiden, Netherlands. Correspondence to: Arsalan Sharifnassab <arsalan.sharifnassab@openmindresearch.org>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

optimal trial and error search methods.

Meta-parameter optimization is particularly important in continual learning (De Lange et al., 2021), where continually changing environments or evolving loss functions necessitate adaptation of meta-parameters, such as step sizes, to track time-varying optima rather than settling on a static value.

In this work, we propose MetaOptimize, a general framework for optimizing meta-parameters to minimize a form of regret that explicitly accounts for the long-term influence of step sizes on future loss. Although this framework can handle various meta-parameters, we concentrate on step sizes as they are ubiquitous and crucial in practice.

MetaOptimize offers additional advantages beyond reducing search overhead. First, it enables dynamic step-size updates during training, potentially speeding the learning process. Traditional methods typically rely on manually designed learning rate schedules (e.g., initial increase followed by decay (Amid et al., 2022)), whereas MetaOptimize automatically discovers similar patterns.

Second, adapting step sizes across different network blocks (e.g., layers or neurons) can improve performance (Singh et al., 2015; Howard & Ruder, 2018), yet manually tuning such blockwise step sizes is impractical for large networks. By design, MetaOptimize handles these blockwise adjustments.

The concept of meta step-size optimization dates back to (Kesten, 1958), Delta-bar-Delta (Sutton, 1981; Jacobs, 1988), and its incremental variant, IDBD (Sutton, 1992). Numerous methods have emerged over the years (Section 8). This work distinguishes itself from prior efforts through the following key aspects:

\- We introduce a formal approach to step-size optimization by minimizing a specific form of regret, essentially a discounted sum of future losses, and demonstrate how to do this causally via the MetaOptimize framework.

\- MetaOptimize is general and can wrap around any first-order optimization algorithm (the base update), such as SGD, RMSProp (Hinton, 2012), Adam (Kingma & Ba,

2014), or Lion (Chen et al., 2023), while optimizing step sizes via a separate first-order method (the meta update), such as SGD, Adam, RMSProp, or Lion.

- We develop approximation methods (Section 6) that, when incorporated into MetaOptimize, yield computationally efficient algorithms outperforming state-of-the-art automatic hyperparameter optimization methods on various stationary and continual (non-stationary) benchmarks (see Section 7).   
- We show that some existing methods (like IDBD and its extensions, and hypergradient descent (Baydin et al., 2017)) are specific instances or approximations within the MetaOptimize framework (Section 5).

# 2. Problem Setting

We introduce a general continual optimization setting that, for a given sequence of loss functions $f_{t}(\cdot):\mathbb{R}^{n}\to\mathbb{R}, t=0,1,2,\ldots$ , aims to find a sequence of weight vectors $w_{1},w_{2},w_{3},\ldots$ that minimize a discounted sum of future losses:

$$
F _ {t} ^ {\gamma} \stackrel {\text { def }} {=} (1 - \gamma) \sum_ {\tau > t} \gamma^ {\tau - t - 1} f _ {\tau} (\boldsymbol {w} _ {\tau}), \tag {1}
$$

where $\gamma\in[0,1)$ is a fixed discount factor, typically close to 1, called the discount factor. For stationary supervised learning, $f_{t}$ are i.i.d. samples from the same distribution, so minimizing $F_{t}^{\gamma}$ promotes rapid reduction of the expected loss.

Consider an arbitrary first-order optimization algorithm (e.g., SGD, RMSProp, Adam, or Lion) for updating $w_{t}$ . At time t, it takes the gradient $\nabla f_{t}(\boldsymbol{w}_{t})$ of the current loss, along with an m-dimensional meta-parameter vector $\beta_{t}$ , to update $w_{t}$ and possibly some internal variables $\tilde{x}_{t}$ (e.g., momentum in Adam). Denoting $x_{t} \stackrel{\operatorname{def}}{=} \operatorname{Stack}(\boldsymbol{w}_{t}, \tilde{\boldsymbol{x}}_{t})$ and calling this update rule $Alg_{base}$ , we have

$$
\boldsymbol {x} _ {t + 1} = \operatorname{Alg} _ {\text { base }} (\boldsymbol {x} _ {t}, \nabla f _ {t} (\boldsymbol {w} _ {t}), \boldsymbol {\beta} _ {t}). \tag {2}
$$

The goal of MetaOptimize is to determine a sequence $\beta_{t}$ such that plugging them into the above base update yields a trajectory $\{w_{t}\}$ minimizing $F_{t}^{\gamma}$ .

Step-size adaptation is a natural special case: at each step t, the m-dimensional $\beta_{t}$ defines an n-dimensional step-size vector $\alpha_{t}$ via some fixed function $\sigma : R^{m} \to R^{n}$ , i.e.,

$$
\boldsymbol {\alpha} _ {t} = \sigma (\boldsymbol {\beta} _ {t}). \tag {3}
$$

A good choice for $\sigma(\cdot)$ is exponential, ensuring $\alpha_{t}$ is always positive and making multiplicative changes in $\alpha_{t}$ correspond to additive changes in $\beta_{t}$ (Sutton, 1992). By partitioning the network weights into blocks, we can learn a shared scalar step-size per block, or even a unique step-size per weight, all handled automatically by MetaOptimize.

# 3. Forward and Backward Views

Because $F_{t}^{\gamma}$ depends on future losses, minimizing it causally requires an alternative view. Suppose hypothetically we had oracle access to future information (i.e., future loss values and weights). We could update

$$
\begin{array}{l} \boldsymbol {\beta} _ {t + 1} = \boldsymbol {\beta} _ {t} - \eta \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {\beta} _ {t}} F _ {t} ^ {\gamma} \tag {4} \\ = \boldsymbol {\beta} _ {t} - \eta (1 - \gamma) \sum_ {\tau > t} \gamma^ {\tau - t - 1} \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {\beta} _ {t}} f _ {\tau} (\boldsymbol {w} _ {\tau}), \\ \end{array}
$$

where $\eta$ is a meta step-size. This forward-view update, however, is not causal because we do not have the required future information at time t.

To address this, we adopt an eligibility-trace-style approach from reinforcement learning (Sutton, 1988; Sutton & Barto, 2018), introducing a backward-view update:

$$
\boldsymbol {\beta} _ {\tau + 1} \leftarrow \boldsymbol {\beta} _ {\tau} - \eta (1 - \gamma) \sum_ {t <   \tau} \gamma^ {\tau - t - 1} \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {\beta} _ {t}} f _ {\tau} (\boldsymbol {w} _ {\tau}), \tag {5}
$$

so that terms involving $f_{\tau}$ (and $w_{\tau}$ ) appear at time $\tau$ (instead of t), which is the earliest time that these quantities become available. In the small- $\eta$ limit, the backward view closely approximates the forward view. $^{1}$

Accordingly, we define a causal gradient estimate

$$
\widehat {\nabla_ {\boldsymbol {\beta}} F} _ {\tau} \stackrel {{\text { def }}} {{=}} (1 - \gamma) \sum_ {t = 0} ^ {\tau - 1} \gamma^ {\tau - t - 1} \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {\beta} _ {t}} f _ {\tau} (\boldsymbol {w} _ {\tau}).
$$

It follows from chain rule that

$$
\widehat {\nabla_ {\boldsymbol {\beta}} F} _ {\tau} = \mathcal {H} _ {\tau} ^ {T} \nabla f _ {\tau} (\boldsymbol {w} _ {\tau}), \tag {6}
$$

where

$$
\mathcal {H} _ {\tau} \stackrel {\text { def }} {=} (1 - \gamma) \sum_ {t = 0} ^ {\tau - 1} \gamma^ {\tau - t - 1} \frac {d \boldsymbol {w} _ {\tau}}{\mathrm{d} \boldsymbol {\beta} _ {t}}. \tag {7}
$$

Hence, $H_{\tau}$ encodes how past $\beta_{t}$ values cumulatively affect $w_{\tau}$ under discounting by $\gamma$ .

# 4. MetaOptimize

Algorithm 1 presents the general MetaOptimize framework for learning the meta-parameters $\beta_{t}$ . At each step

t, we replace the intractable gradient $\nabla_{\beta}F_{t}^{\gamma}$ with the causal surrogate $\widehat{\nabla_{\beta}F_{t}}$ to ensure the update is feasible in real time, as discussed in Section 3. Specifically, we feed $\widehat{\nabla_{\beta}F_{t}} = \mathcal{H}_{t}^{T}\nabla f_{t}(\boldsymbol{w}_{t})$ (from (6)) into any first-order meta-update rule $Alg_{meta}$ , just like a conventional gradient.

Formally, define $\pmb{y}_t \stackrel{\mathrm{def}}{=} \mathrm{Stack}(\beta_t, \tilde{\pmb{y}}_t)$ as the stack of metaparameters $\beta_t$ and any internal states $\tilde{\pmb{y}}_t$ of $\mathrm{Alg}_{\mathrm{meta}}$ (e.g., momentum). The meta-update is then:

$$
\boldsymbol {y} _ {t + 1} = \operatorname{Alg} _ {\text { meta }} \left(\boldsymbol {y} _ {t}, \mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t})\right). \tag {8}
$$

After applying the base update (2) to produce $x_{t+1}$ , we compute $\mathcal{H}_{t}^{T}\nabla f_{t}(\boldsymbol{w}_{t})$ and plug it into (8) to update $y_{t+1}$ (and thus $\beta_{t+1}$ ). The remaining question is how to maintain $H_{t}$ defined in (7), through application of the chain rule.

To compute $H_{t}$ incrementally, let us stack the columns of the $n \times m$ matrix $H_{t}$ into a single vector $h_{t}$ , and let

$$
G _ {t} \stackrel {\text { def }} {=} \left[ \begin{array}{c c c} \frac {\mathrm{d}   \boldsymbol {y} _ {t + 1}}{\mathrm{d}   \boldsymbol {y} _ {t}} & \frac {\mathrm{d}   \boldsymbol {y} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & \frac {\mathrm{d}   \boldsymbol {y} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \\ \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {y} _ {t}} & \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \\ \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {y} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right]. \tag {9}
$$

Applying the chain rule then implies

$$
\left[ \begin{array}{c} \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \\ \frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \\ \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \end{array} \right] = G _ {t} \left[ \begin{array}{c} \frac {\mathrm{d} \boldsymbol {y} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \\ \frac {\mathrm{d} \boldsymbol {x} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \\ \frac {\mathrm{d} \boldsymbol {h} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \end{array} \right],
$$

which, when summed over $\tau$ , turns into

$$
\sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \left[ \begin{array}{l} \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \\ \frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \\ \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \end{array} \right] = G _ {t} \left[ \begin{array}{l} \frac {\mathrm{d} \boldsymbol {y} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {t}} \\ \frac {\mathrm{d} \boldsymbol {x} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {t}} \\ \frac {\mathrm{d} \boldsymbol {h} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {t}} \end{array} \right] + G _ {t} \sum_ {\tau = 0} ^ {t - 1} \gamma^ {t - \tau} \left[ \begin{array}{l} \frac {\mathrm{d} \boldsymbol {y} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \\ \frac {\mathrm{d} \boldsymbol {x} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \\ \frac {\mathrm{d} \boldsymbol {h} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \end{array} \right]. \tag {10}
$$

Defining

$$
Y _ {t} \stackrel {\text { def }} {=} (1 - \gamma) \sum_ {\tau = 0} ^ {t - 1} \gamma^ {t - \tau - 1} \frac {\mathrm{d}   \boldsymbol {y} _ {t}}{\mathrm{d}   \boldsymbol {\beta} _ {\tau}} \tag {11}
$$

$$
X _ {t} \stackrel {\text { def }} {=} (1 - \gamma) \sum_ {\tau = 0} ^ {t - 1} \gamma^ {t - \tau - 1} \frac {\mathrm{d}   \boldsymbol {x} _ {t}}{\mathrm{d}   \boldsymbol {\beta} _ {\tau}}, \tag {12}
$$

$$
Q _ {t} \stackrel {\text { def }} {=} (1 - \gamma) \sum_ {\tau = 0} ^ {t - 1} \gamma^ {t - \tau - 1} \frac {\mathrm{d}   \boldsymbol {h} _ {t}}{\mathrm{d}   \boldsymbol {\beta} _ {\tau}}, \tag {13}
$$

and noting that $y_{t} = \text{Stack}(\beta_{t}, \tilde{y}_{t})$ , one obtains a compact update of the form

$$
\left[ \begin{array}{l} Y _ {t + 1} \\ X _ {t + 1} \\ Q _ {t + 1} \end{array} \right] = G _ {t} \left(\gamma \left[ \begin{array}{l} Y _ {t} \\ X _ {t} \\ Q _ {t} \end{array} \right] + (1 - \gamma) \left[ \begin{array}{c} I \\ 0 \\ 0 \\ 0 \end{array} \right]\right), \tag {14}
$$

Algorithm 1 MetaOptimize Framework (for general meta-parameters)

Given: A base-update $\mathrm{Alg}_{\mathrm{base}}$ , a meta-update $\mathrm{Alg}_{\mathrm{meta}}$ , and a discount-factor $\gamma \leq 1$ .

Initialize:

$$
\begin{array}{l} \text { Initialize:} \\ X _ {0} = 0 _ {(n + \tilde {n}) \times m}, Y _ {0} = \left[ \begin{array}{c} I _ {m \times m} \\ 0 _ {\tilde {m} \times m} \end{array} \right], Q _ {0} = 0 _ {n m \times m}. \\ \text { or } t = 0, 1, 2, \quad \text { do } \end{array}
$$

$$
\boldsymbol {x} _ {t + 1} \leftarrow \operatorname{Alg} _ {\text { base }} (\boldsymbol {x} _ {t}, \nabla f _ {t} (\boldsymbol {w} _ {t}), \boldsymbol {\beta} _ {t}).
$$

$$
\mathcal {H} _ {t} = \text { first   } n \text {   rows   of   } X _ {t}.
$$

$$
\boldsymbol {y} _ {t + 1} \leftarrow \operatorname{Alg} _ {\text { meta }} \left(\boldsymbol {y} _ {t}, \mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t})\right).
$$

$$
\begin{array}{l} \text { Update } \left[ \begin{array}{c} Y _ {t + 1} \\ X _ {t + 1} \\ Q _ {t + 1} \end{array} \right] \text { from   (14),   using   G_{t}   in   (9).} \end{array}
$$

and then extract $H_{t}$ by taking the top n rows of $X_{t}$ (since $x_{t} = \text{Stack}(w_{t}, \tilde{x}_{t})$ ). The blocks of $G_{t}$ can be found for standard algorithms (SGD, Adam, Lion, etc.) as detailed in Appendix A. Notably, the first row of $G_{t}$ blocks depends only on $Alg_{meta}$ , and the rest of $G_{t}$ blocks depend only on $Alg_{base}$ . Algorithm 1 summarizes the procedure.

Remark 4.1. A distinction of MetaOptimize from existing meta-parameter optimization methods is that it explicitly captures dynamics of the meta-parameters $\beta$ , and how changes in the current $\beta$ affect $\beta$ in future. The term $Y_{t}$ in (11) links changes in past $\beta_{t}$ to future $\beta$ values, which then influences $H_{t}$ . Intuitively, if $\beta_{t}$ has been changing consistently in one direction (e.g., steadily increasing), it amplifies $Y_{t}$ and thus $H_{t}$ , accelerating ongoing updates. Conversely, if $\beta_{t}$ stays nearly constant (indicating it may be close to optimal), $Y_{t}$ shrinks and so do the subsequent updates to $\beta_{t}$ , stabilizing around the optimum.

# 5. Reducing Complexity

The matrix $G_{t}$ can be large and may involve Hessian terms, increasing the computational burden. We discuss two practical approximations:

2×2 approximation. In (9), we zero out all blocks in the last row and column, effectively removing $Q_{t}$ . Empirically, this simplification often has negligible impact on performance. Intuitively, $H_{t}$ does not affect the base update directly ( $d x_{t+1}/d h_{t} = 0$ ), so the extra blocks in $G_{t}$ involving $h_{t}$ often have minor influence on the final metaparameter trajectory.

L-approximation. We go one step further, also zeroing out the block in the first row and second column of $G_{t}$ .

Algorithm 2 MetaOptimize with $2 \times 2$ approx., $(\text{Alg}_{\text{base}}, \text{Alg}_{\text{meta}}) = (\text{SGD}, \text{SGD})$ , and scalar step-size   
Initialize: $\mathcal{H}_0 = \mathbf{0}_{n\times 1},Y_0 = 1$

for $t = 1,2,\ldots \mathbf{do}$

$\alpha_{t} = e^{\beta_{t}}$   
Base update:

$$
\begin{array}{l} \boldsymbol {w} _ {t + 1} = \boldsymbol {w} _ {t} - \alpha_ {t} \nabla f _ {t} (\boldsymbol {w} _ {t}) \\ \mathcal {H} _ {t + 1} = \gamma \left(I - \alpha_ {t} \nabla^ {2} f _ {t} (\boldsymbol {w} _ {t})\right) \mathcal {H} _ {t} - Y _ {t} \alpha_ {t} \nabla f _ {t} (\boldsymbol {w} _ {t}) \\ Y _ {t + 1} = \gamma Y _ {t} + (1 - \gamma) - \gamma \eta \mathcal {H} _ {t} ^ {T} \nabla^ {2} f _ {t} (\pmb {w} _ {t}) \mathcal {H} _ {t} \\ \# \text { For   L - approximation   let } Y _ {t + 1} = 1 \\ \end{array}
$$

Meta update:   
$\beta_{t + 1} = \beta_t - \eta \mathcal{H}_t^T\nabla f_t(\pmb {w}_t)$   
end for

Formally,

$$
G _ {t} ^ {L} \stackrel {\text { def }} {=} \left[ \begin{array}{c c} \frac {\mathrm{d}   \boldsymbol {y} _ {t + 1}}{\mathrm{d}   \boldsymbol {y} _ {t}} & 0 \\ \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {y} _ {t}} & \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} \end{array} \right], \tag {15}
$$

and the update in (14) simplifies to

$$
\left[ \begin{array}{l} Y _ {t + 1} \\ X _ {t + 1} \end{array} \right] = G _ {t} ^ {L} \left(\gamma \left[ \begin{array}{l} Y _ {t} \\ X _ {t} \end{array} \right] + (1 - \gamma) \left[ \begin{array}{c} I \\ 0 \\ \hline 0 \end{array} \right]\right). \tag {16}
$$

This again discards $Q_{t}$ , but also certain cross-terms in $Y_{t}$ 's update. Empirically, L-approximation often matches the performance and sometimes improves the stability of the $2 \times 2$ approach.

Intuition. Algorithm 2 illustrates the $2 \times 2$ approximation for the case of SGD base/meta updates with a single scalar step size. Observe how $H_{t}$ effectively accumulates (decayed) past gradients to decide whether to increase or decrease $\alpha_{t}$ . If current and past gradients align, $\alpha_{t}$ is raised for faster learning; if they oppose each other, $\alpha_{t}$ shrinks. The decay $\gamma(I - [\alpha] \nabla^{2} f_{t})$ of $H_{t}$ ensures that if past gradients poorly approximate future ones due to large $\nabla^{2} f_{t}$ or $\alpha$ , their influence fades more rapidly. Meanwhile, $Y_{t}$ reflects how changing past $\beta$ influences the current $\beta$ ; large swings in $\beta$ amplify $H_{t+1}$ , while near-constant $\beta$ dampens updates. Under the L-approximation, $Y_{t}$ becomes constant in this particular setup, further simplifying the algorithm.

Containing some prior methods as special cases. Under L-approximation, and restricting both base and meta updates to plain SGD, MetaOptimize reduces to IDBD (Sutton, 1982) and its extension (Xu et al., 2018); see Appendix B.1 for derivations. Another notable special case is $\gamma = 0$ , which recovers the Hypergradient-descent approach (Baydin et al., 2017), updating step sizes to minimize the immediate loss $f_{t}(\boldsymbol{w}_{t})$ rather than the discounted sum $F_{t}^{\gamma}$ , ignoring long-term effects of step size on future loss.

# 6. Hessian-Free MetaOptimize

The $G_{t}$ matrix typically involves Hessian, $\nabla^{2}f_{t}(\boldsymbol{w}_{t})$ , of the loss function, e.g., in the $d\boldsymbol{w}_{t+1}/d\boldsymbol{w}_{t}$ block where $w_{t+1} = w_{t} - \alpha_{t}\nabla f_{t}(\boldsymbol{w}_{t})$ . Including second-order information in $G_{t}$ can be costly. Interestingly, for certain base and meta algorithms, we can eliminate the Hessian without much compromising the performance.

For example, Lion (Chen et al., 2023) updates weights by taking the sign of the gradient (plus momentum). Since the derivative of the sign function is zero almost everywhere, $\mathrm{d}\pmb{w}_{t + 1} / \mathrm{d}\pmb{w}_t$ and related partials do not involve $\nabla^2 f_t(\pmb{w}_t)$ . Hence, if both base and meta updates use Lion, $G_{t}$ becomes Hessian-free throughout, avoiding second-order computations entirely (Appendix A.1.3, A.3.2).

For other algorithms, we may consider their Hessian-free approximation by zeroing out any Hessian term in $G_{t}$ . The Hessian-free approximation turns out to be a good approximation, especially for base and meta algorithms that involve gradient normalization, like RMSProp and Adam. Note that, the sign function used in the Lion algorithm is an extreme form of normalization that divides a vector by its absolute value. We could instead use softer forms of normalization, such as normalizing to square root of a trace of squared vector, $v_{t}$ , as in RMSProp. Such normalizations typically result in two opposing Hessian-based terms in $H_{t}$ 's update (stemming from $\frac{d w_{t+1}}{d w_{t}}$ and $\frac{d w_{t+1}}{d v_{t}}$ blocks of matrix $G_{t}$ ), aiming to cancel out, particularly when consecutive gradients are positively correlated.

When Hessian terms are removed in the $2 \times 2$ approximation, $X_{t}$ and $Y_{t}$ become diagonal or simply vectorized, drastically reducing matrix-multiplication overhead. The overall complexity per step thus becomes similar to that of regular base and meta updates, requiring only a few extra vector operations. Algorithm 3 in Appendix A illustrates these Hessian-free variants (SGDm, AdamW, Lion) under $2 \times 2$ approximation.

In summary, Hessian-free and $2 \times 2$ or L-approximations yield a range of practical MetaOptimize instantiations that maintain strong performance at low additional cost.

# 7. Experiments

We evaluate MetaOptimize on image-classification and language-modeling benchmarks. Out of many possible base/meta-algorithm combinations and approximations (Algorithm 3), we showcase a few Hessian-free variants that performed well in practice. In the experiments, MetaOptimize starts with step-sizes set one or two orders of magnitude below typical good fixed step-sizes, with no specific tuning. We compare MetaOptimize against some popular baselines whose meta-parameters are well-tuned for each

task separately. See Appendix C for more details. Codes are available at https://github.com/sabersalehk/MetaOptimize.

# 7.1. CIFAR10 dataset

The first set of experiments involve training ResNet-18 with batch size of 100 on the CIFAR10 (Krizhevsky et al., 2009) dataset. Fig. 1 depicts the learning curves of four combinations of (base, meta) algorithms for Hessian-free MetaOptimize, along with the corresponding baselines with well-tuned fixed step sizes. Besides using a single scalar step-size, we also test a blockwise variant that partitions the ResNet18 parameters into six blocks (one for each linear layer and four blocks for the ResNet modules). In every tested combination, MetaOptimize outperforms its corresponding fixed-step-size baseline.

![](images/7f14400d6b7946e74419e693024e14510f217c3b47dba05f9ade6c8bbed5da00.jpg)

Figure 1. Learning curves for selected (base, meta) combinations in CIFAR10.   
![](images/806685a953f8db56efdb838bc93bf7b65cf146d0620d2c5ed1de9feda2f2af05.jpg)

<details>
<summary>line</summary>

| Iteration | Initial alpha = 10e-6 | Initial alpha = 10e-7 | Initial alpha = 10e-8 | Initial alpha = 10e-9 |
| --------- | --------------------- | --------------------- | --------------------- | --------------------- |
| 0.00      | ~1.0                  | ~0.0                  | ~0.0                  | ~0.0                  |
| 0.25      | ~9.0                  | ~8.0                  | ~3.0                  | ~0.5                  |
| 0.50      | ~3.0                  | ~3.5                  | ~8.5                  | ~1.0                  |
| 0.75      | ~2.0                  | ~2.5                  | ~3.0                  | ~6.0                  |
| 1.00      | ~1.5                  | ~1.5                  | ~2.0                  | ~8.5                  |
| 1.25      | ~1.0                  | ~1.0                  | ~1.5                  | ~6.0                  |
| 1.50      | ~0.8                  | ~0.8                  | ~1.0                  | ~4.0                  |
| 1.75      | ~0.6                  | ~0.6                  | ~0.8                  | ~2.0                  |
| 2.00      | ~0.5                  | ~0.5                  | ~0.6                  | ~1.0                  |
</details>

Figure 2. Robustness to initial step-sizes, for (Lion, Lion) as (base, meta) update in CIFAR10.

Interestingly, as demonstrated in Fig. 2, the MetaOptimize algorithms show remarkable robustness to initial step-size choices, even for initial step sizes that are several orders of magnitude smaller than the optimal fixed step-size.

# 7.2. Non-stationary CIFAR100

We evaluated MetaOptimize in a non-stationary setting with 10 sequential tasks, each containing 10 classes from CIFAR100. After training for one epoch on a task, it abruptly switches without explicit notification to the optimizer, and without resetting weights. We use a batch size of one, meaning each data point is seen exactly once. The model is based on a simple CNN network consisting of two convolution (and max pooling) layers followed by a fully connected layer. Each curve is averaged over 5 random seeds.

![](images/ec9e4774473a8a797cf0bb71cf092d3716680be822cd579b5a562007740afc23.jpg)

<details>
<summary>line</summary>

| Epoch | AdamW, Fixed stepsize | MetaOptimize (AdamW, Adam), Scalar | MetaOptimize (AdamW, Adam), Blockwise |
|-------|------------------------|-------------------------------------|----------------------------------------|
| 0     | 30.0                   | 30.0                                | 30.0                                   |
| 5000  | 37.5                   | 37.8                                | 37.6                                   |
| 10000 | 36.0                   | 36.5                                | 36.8                                   |
| 15000 | 35.5                   | 36.2                                | 36.5                                   |
| 20000 | 35.8                   | 36.8                                | 37.0                                   |
| 25000 | 36.2                   | 37.2                                | 37.5                                   |
| 30000 | 36.5                   | 37.5                                | 38.0                                   |
| 35000 | 36.8                   | 37.8                                | 38.5                                   |
| 40000 | 37.0                   | 38.0                                | 39.0                                   |
| 45000 | 37.2                   | 38.2                                | 39.2                                   |
| 50000 | 37.5                   | 38.5                                | 39.5                                   |
</details>

Figure 3. Learning curves for non-stationary CIFAR100.   
![](images/7aadf071daea6d7cc4645eb44561c8305c3e8b15cb7b1405b70f3b0f2e6dd60e.jpg)

<details>
<summary>line</summary>

| Iteration | Step-size of first block (x10⁻⁵) | Step-size of second block (x10⁻⁴) |
| --------- | -------------------------------- | --------------------------------- |
| 0         | 10.0                             | 0.0                               |
| 5000      | 6.0                              | 2.0                               |
| 10000     | 1.0                              | 4.0                               |
| 15000     | 6.0                              | 2.0                               |
| 20000     | 1.0                              | 4.0                               |
| 25000     | 4.0                              | 2.0                               |
| 30000     | 1.0                              | 4.0                               |
| 35000     | 6.0                              | 2.0                               |
| 40000     | 1.0                              | 4.0                               |
| 45000     | 9.0                              | 2.0                               |
| 50000     | 1.0                              | 4.0                               |
</details>

Figure 4. Blockwise stepsizes learned by MetaOptimize (AdamW, Adam) on non-stationary CIFAR100. Note that the scale of the y-axis for the two curves differ by an order of magnitude. Step-sizes of both blocks are initialized at $\alpha_{0} = 10^{-4}$ .

Figure 3 presents cumulative top-1 accuracy—averaged over all past training times—for AdamW (with the best fixed step-size), MetaOptimize with a scalar step-size, and MetaOptimize with blockwise step-sizes (two blocks: one for the first three layers and one for the last layer). MetaOptimize consistently outperforms the baseline. See Appendix D for additional plots.

The learned step-sizes reveal an interesting pattern (Fig. 4):

- Task adaptation: MetaOptimize increases step-sizes immediately after task switches to enhance adaptation.   
- Layer-wise behavior: In blockwise case, early-layer step-sizes decrease over time, indicating convergence

Algorithm 3 Hessian-free MetaOptimize algorithms with $2 \times 2$ approximation used in experiments   
Parameters: \(\eta > 0\) (default \(10^{-3}\)), \(\gamma \in [0,1]\) (default \(\simeq 1\))
Initialize: \(h_0 = 0_{n \times 1}\).
for \(t = 1,2,\ldots\) do
    Base update \(\begin{bmatrix} \boldsymbol{\alpha}_t = \sigma(\boldsymbol{\beta}_t) & \# \text{ exponential scalar/blockwise} \\ m_{t+1} = \rho m_t + (1 - \rho)\nabla f_t(w_t) \\ \text{if Alg}_{\text{base}} \text{ is SGDm then} & \Delta w = -\boldsymbol{\alpha}_t m_t - \kappa \boldsymbol{\alpha}_t w_t \\ \text{if Alg}_{\text{base}} \text{ is Lion then} & \Delta w = -\boldsymbol{\alpha}_t \text{ Sign } (c m_t + (1 - c)\nabla f_t) - \kappa \boldsymbol{\alpha}_t w_t \\ \text{if Alg}_{\text{base}} \text{ is AdamW then} & v_{t+1} = \lambda v_t + (1 - \lambda)\nabla f_t(w_t)^2 \\ & \Delta w = -\boldsymbol{\alpha}_t \mu_t m_t / \sqrt{v_t} - \kappa \boldsymbol{\alpha}_t w_t & \# \text{ where } \mu_t = \sqrt{1 - \lambda^t} / (1 - \rho^t) \\ w_{t+1} = w_t + \Delta w \\ h_{t+1} = \gamma (1 - \kappa \boldsymbol{\alpha}_t) h_t + \Delta w \end{bmatrix}\)
Meta update \(\begin{bmatrix} z = h_t^\top \nabla f_t(w_t) & \# \text{ This is for scalar step-sizes.} \\ & \# \text{ For blockwise, should compute sum of } h_t\nabla f_t(w_t) \text{ over each block.} \\ \bar{m}_{t+1} = \bar{\rho} \bar{m}_t + (1 - \bar{\rho}) z \\ \text{if Alg}_{\text{meta}} \text{ is Lion then} & \beta_{t+1} = \beta_t - \eta \text{ Sign } (\bar{c} \bar{m}_t + (1 - \bar{c}) z) \\ \text{if Alg}_{\text{meta}} \text{ is Adam then} & \bar{v}_{t+1} = \bar{\lambda} \bar{v}_t + (1 - \bar{\lambda}) z^2 \\ & \beta_{t+1} = \beta_t - \eta \bar{\mu}_t \bar{m}_t / \sqrt{\bar{v}_t} & \# \text{ where } \bar{\mu}_t = \sqrt{1 - \bar{\lambda}^t} / (1 - \bar{\rho}^t) \\ end for

to globally useful features, while last-layer step-sizes increase, reflecting the need to adapt to changing labels.

# 7.3. ImageNet dataset

We trained ResNet-18 with batch-size 256 on ImageNet (Deng et al., 2009). We compared MetaOptimize with scalar step-size against four state-of-the-art hyperparameter optimization algorithms, namely DoG (Ivgi et al., 2023), gdtuo (Chandra et al., 2022), Prodigy (Mishchenko & Defazio, 2023), and mechanic (Cutkosky et al., 2024), as well as AdamW and Lion baselines with fixed step-sizes, and AdamW with a well-tuned cosine decay learning rate scheduler with a 10k iterations warmup. Learning curves and complexity overheads are shown respectively in Fig. 5 and Table 1, showcasing the advantage of MetaOptimize algorithms (learning curve of DoG is not depicted due to its relatively poor performance). Unlike CIFAR10, here the blockwise versions of MetaOptimize showed no improvement over the scalar versions. Refer to Appendix D for further details.

# 7.4. Language modeling

For language model experiments, we used the TinyStories dataset (Eldan & Li, 2023), a synthetic collection of brief stories designed for children aged 3 to 4. This dataset proves effective for training and evaluating language models that are significantly smaller than the current state-of-the-art, and capable of crafting stories that are not only fluent and coherent but also diverse.

We used the implementation in (Karpathy, 2024) for training 15M parameter model with a batch size of 128 on the TinyStories dataset. Two combinations of Hessian-free MetaOptimize with scalar step sizes were tested against Lion and AdamW with well-tuned fixed step sizes, AdamW with a well-tuned cosine decay learning rate scheduler with 1k warmup iterations, and the four state-of-the-art step-size adaptation algorithms mentioned in the previous subsection. According to the learning curves, shown in Fig. 6, MetaOptimize outperforms all baselines (with an initial delay due to small initial step-sizes), except for the well-tuned learning rate scheduler within 30k iterations.

# 7.5. Sensitivity analysis

Here, we briefly discussion the sensitivity of MetaOptimize to its meta-meta-parameters.

For the meta-stepsize $\eta$ in MetaOptimize, there is generally no need for tuning, and the default value $\eta = 10^{-3}$ works universally well in stationary supervised learning. All experiments in this section used this default value with no sweeping required. The rationale for this choice is that when using Adam, Lion, or RMSProp for meta-updates, the absolute change in $\beta$ per iteration is approximately $\eta \times O(1) \simeq 10^{-3}$ . Unless the current stepsize $\alpha$ is already near its optimal value, most $\beta$ updates will consistently move toward the optimal $\beta$ . Within 1,000 steps, $\beta$ can change by $O(1)$ , nearly doubling or halving $\alpha = \exp(\beta)$ . Over 10,000 iterations, $\alpha$ can adjust to stepsizes that are $e^{10} > 20,000$ times larger or smaller, allowing $\eta \simeq 10^{-3}$ to efficiently track optimal stepsizes while minimizing unnecessary fluctuations in $\alpha$ .

Regarding the discount factor $\gamma$ , we used the $\gamma = 1$ in all stationary experiments and observed minimal sensitivity to

Table 1. Per-iteration wall-clock-time and GPU-memory overhead (compared to AdamW). 

<table><tr><td rowspan="2">Algorithm</td><td colspan="2">ImageNet</td><td colspan="2">TinyStories</td></tr><tr><td>Time</td><td>Space</td><td>Time</td><td>Space</td></tr><tr><td>AdamW (fixed stepsize)</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>DoG (Ivgi et al., 2023)</td><td>+45%</td><td>1.4%</td><td>+268%</td><td>0%</td></tr><tr><td>gdtuo (Chandra et al., 2022)</td><td>+85%</td><td>64%</td><td>+150%</td><td>21%</td></tr><tr><td>mechanic (Cutkosky et al., 2024)</td><td>+42%</td><td>88%</td><td>+9%</td><td>0%</td></tr><tr><td>Prodigy (Mishchenko &amp; Defazio, 2023)</td><td>+42%</td><td>13%</td><td>+9%</td><td>0%</td></tr><tr><td>MetaOptimize (AdamW, Lion)</td><td>+44%</td><td>33%</td><td>+13%</td><td>0%</td></tr></table>

![](images/5ee387ac6bae20e1593fb5a8d00d1f89013f24903f26def4a989e98d4a17e033.jpg)

<details>
<summary>line</summary>

| Epoch | AdamW, LR scheduler | AdamW, Fixed stepsize | Lion, Fixed stepsize | MetaOptimize (AdamW, Lion), Scalar | MetaOptimize (Lion, Lion), Scalar | MetaOptimize (SGDm, Lion) | gdtuo | mechanic | Prodigy |
|-------|----------------------|------------------------|----------------------|-------------------------------------|------------------------------------|----------------------------|-------|----------|---------|
| 0     | 65.0                 | 65.0                   | 65.0                 | 65.0                                | 65.0                               | 65.0                       | 65.0  | 65.0     | 65.0    |
| 20    | 78.0                 | 77.0                   | 76.0                 | 80.0                                | 79.0                               | 81.0                       | 82.0  | 79.0     | 83.0    |
| 40    | 83.0                 | 82.0                   | 81.0                 | 85.0                                | 84.0                               | 86.0                       | 87.0  | 84.0     | 88.0    |
| 60    | 86.0                 | 85.0                   | 84.0                 | 87.0                                | 86.0                               | 88.0                       | 89.0  | 86.0     | 90.0    |
| 80    | 87.0                 | 86.0                   | 85.0                 | 88.0                                | 87.0                               | 89.0                       | 90.0  | 87.0     | 91.0    |
| 100   | 87.5                 | 86.5                   | 85.5                 | 88.5                                | 87.5                               | 89.5                       | 90.5  | 87.5     | 91.5    |
</details>

Figure 5. ImageNet learning curves.

![](images/cca9ddafd70ab06e86eb728602f40aa52f535e522005d8d5b0df97cbe4cf98f5.jpg)

<details>
<summary>line</summary>

| Iteration | AdamW, LR scheduler | AdamW, Fixed stepsize | Lion, Fixed stepsize | MetaOptimize (AdamW, Lion) | MetaOptimize (Lion, Lion) | DoG | gdtuo | mechanic | Prodigy |
| --------- | ------------------- | --------------------- | -------------------- | -------------------------- | ------------------------- | --- | ----- | -------- | ------- |
| 0         | 2.2                 | 2.2                   | 2.2                  | 2.2                        | 2.2                       | 2.2 | 2.2   | 2.2      | 2.2     |
| 5000      | 1.4                 | 1.5                   | 1.5                  | 1.5                        | 1.5                       | 1.6 | 1.6   | 1.6      | 1.7     |
| 10000     | 1.3                 | 1.4                   | 1.4                  | 1.4                        | 1.4                       | 1.5 | 1.5   | 1.5      | 1.6     |
| 15000     | 1.25                | 1.35                  | 1.35                 | 1.35                       | 1.35                      | 1.45| 1.45  | 1.45     | 1.5     |
| 20000     | 1.2                 | 1.3                   | 1.3                  | 1.3                        | 1.3                       | 1.4 | 1.4   | 1.4      | 1.45    |
| 25000     | 1.2                 | 1.3                   | 1.3                  | 1.3                        | 1.3                       | 1.4 | 1.4   | 1.4      | 1.4     |
| 30000     | 1.2                 | 1.3                   | 1.3                  | 1.3                        | 1.3                       | 1.4 | 1.4   | 1.4      | 1.4     |
</details>

Figure 6. TinyStories (language model) learning curves.

$\gamma$ for values $\gamma \geq 0.999$ in a series of preliminary tests. However, performance begins to degrade with smaller values of $\gamma$ . In the non-stationary CIFAR100, $\gamma = 0.999$ performed slightly better than 1.

# 8. Related Works

Automatic adaptation of step sizes, has been an important research topic in the literature of stochastic optimization. Several works aimed to remove the manual tuning of learning rates via adaptations of classical line search (Rolinek & Martius, 2018; Vaswani et al., 2019; Paquette & Scheinberg, 2020; Kunstner et al., 2023) and Polyak step size (Berrada et al., 2020; Loizou et al., 2021), stochastic proximal methods (Asi & Duchi, 2019), stochastic quadratic approximation (Schaul et al., 2013), hyper-gradient descent (Baydin et al., 2017), nested hyper-gradient descent (Chandra et al., 2022), distance to a solution adaptation (Ivgi et al., 2023; Defazio & Mishchenko, 2023; Mishchenko & Defazio, 2023), and online convex learning (Cutkosky et al., 2024). A limitation of most of these methods is their potential underperformance when their meta-parameters are not optimally configured for specific problems (Ivgi et al., 2023). Moreover, the primary focus of most of these methods is on minimizing immediate loss rather than considering the long-term effects of step sizes on future loss.

Normalization techniques proposed over past few years, such as AdaGrad (Duchi et al., 2011), RMSProp, and Adam have significantly enhanced the training process. While these algorithms show promise in the stationary problems, these normalization techniques do not optimize effective step sizes and are prone to have sub-optimal performance especially in the continual learning settings (Degris et al., 2024).

An early practical step-size optimization method was the Incremental-Delta-Bar-Delta (IDBD) algorithm, introduced in (Sutton, 1992), which aimed to optimize the step-size vector to minimize a specific form of quadratic loss functions in a continual setting. This algorithm was later extended for neural networks in (Xu et al., 2018; Donini et al., 2019), and further adapted in (Mahmood et al., 2012; Javed, 2020; Micaelli & Storkey, 2021) for different meta or base updates beyond SGD. However, the development of IDBD and its extensions included some implicit assumptions, notably overlooking the impact of step-size dynamics on the formulation of step-size update rules. These extensions are, in essence, special cases of the L-approximation within the MetaOptimize framework. The current work extends the IDBD research, significantly broadening the framework and establishing a solid basis for the derivations. IDBD and its extensions have been used in various machine learning tasks including independent component analysis (Schraudolph &

Giannakopoulos, 1999), human motion tracking (Kehl & Van Gool, 2006), classification (Koop, 2007), and reinforcement learning (Xu et al., 2018; Young et al., 2018; Javed et al., 2024). Refer to (Sutton, 2022) for a comprehensive history of step-size optimization.

Hypergradient Descent (HD) (Baydin et al., 2017) adapts learning rates using immediate loss gradients. MADA (Ozkara et al., 2024) extends HD by parameterizing a space of optimizers and navigating it via hypergradient descent. Both focus on short-term effects, whereas MetaOptimize introduces a discount factor $\gamma$ to model long-term influences, generalizing HD as a special case when $\gamma = 0$ .

A related line of work is gradient-based bilevel optimization, initially introduced by (Bengio, 2000) and later expanded in (Maclaurin et al., 2015; Pedregosa, 2016; Franceschi et al., 2018; Gao et al., 2022). Recent advances, such as (Lorraine et al., 2020), enable the optimization of millions of hyperparameters. While bilevel optimization focuses on tuning hyperparameters to minimize validation loss through repeated full training runs of the base algorithm, MetaOptimize diverges significantly. Designed for continual learning, MetaOptimize optimizes meta-parameters on-the-fly during a single streaming run, without relying on validation loss. Instead, it minimizes online loss (or regret) directly, aligning with the continual learning framework where no validation or test sets exist, and data arrives sequentially.

Another relevant literature is learn to optimize (L2O), which aim to learn optimization strategies from data. Classical L2O methods such as (Andrychowicz et al., 2016) train optimizers offline and deploy them unchanged. Later works, including (Metz et al., 2020; 2022), develop more effective or scalable architectures, often using neural networks to modulate optimizer behavior. While powerful, these methods typically lack the ability to adapt online. In contrast, MetaOptimize updates its meta-parameters continuously during training, which is advantageous in nonstationary or continual learning scenarios.

There is also a line of research on the so-called parameter-free optimization that aims to remove the need for step-size tuning with almost no knowledge of the problem properties. Most of these methods are primarily designed for stochastic convex optimization (Luo & Schapire, 2015; Orabona & Pál, 2016), while more recent ones (Orabona & Tommasi, 2017; Ivgi et al., 2023) were applied to supervised learning tasks with small or moderate sample sizes.

# 9. Limitations and Future Works

Our work represents a step toward unlocking the potential of meta-parameter optimization, with substantial room for further exploration, some of which we outline here: Hessian: We confined our experiments to Hessian-free methods for practicality, though Hessian-based algorithms could offer superior performance. These methods, however, face challenges requiring additional research. The Hessian matrix is notably noisy, impacting $H_{t+1}$ multiplicatively, necessitating smoothing and clipping techniques. Additionally, the Hessian approximates the loss landscape's curvature but fails to account for non-differentiable curvatures, such as those from ReLU unit breakpoints, significant at training's end. From a computational perspective, developing low-complexity methods for approximate Hessian matrix products, especially for adjusting step-sizes at the layer and weight levels, is essential.

More accurate traces: As discussed in Section 3, accuracy of the backward approximation (5) may degrade for larger values of the meta-stepsize $\eta$ . Eligibility traces in RL suffer from a similar problem, to resolve which more-sophisticated traces (e.g., Dutch traces) have been developed (see Chapter 11 of (Sutton & Barto, 2018)). Developing more accurate backward approximations for meta-parameter optimization can result in considerable improvements in performance and stability.

Blockwise step-sizes: While step sizes can vary much in granularity, our experiments focused on scalar and blockwise step-sizes. While increasing the number of step sizes is anticipated to enhance performance, our experimental findings in Section 7 reveal that this improvement is not consistent across the MetaOptimize approximations evaluated. Further investigation is needed in future research.

Other approximations: We explored a limited set of MetaOptimize's possible approximations, leaving a comprehensive analysis of various approximations for future research.

Other meta-parameters: Our study was limited to differentiable meta-parameters, not covering discrete ones like batch size or network layer count. We also did not investigate several significant differentiable meta-parameters beyond step-sizes, deferring such exploration to future work.

Automatic Differentiation: While certain versions of MetaOptimize, such as the L-Approximation, could be implemented using standard automatic differentiation software, its applicability to the general case of MetaOptimize remains unclear. Unlike updates for w and $\beta$ (base and meta parameters), the H matrix lacks an explicit incremental formula that can be easily handled by automatic differentiation. For some versions of MetaOptimize, including the Hessian-free approximations used in our experiments, automatic differentiation is unnecessary, as meta updates do not require additional differentiation. Exploring the scope and applicability of automatic differentiation across different MetaOptimize instances is an interesting direction for future research.

Discount factor $\gamma = 1$ : Our backward formulation (Eq. (14)) formally assumes $\gamma < 1$ due to a normalization factor used in the definition of the surrogate gradient. For $\gamma = 1$ , a simple workaround is to remove this scaling factor, which makes the derivation fully valid and consistent—matching our actual implementation and experiments. That said, the case $\gamma = 1$ presents subtle theoretical differences, much like in reinforcement learning and dynamic programming where additional centering is often helpful. Adapting similar techniques for meta-optimization may yield benefits and is a promising direction for future work.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Ehsan Amid, Rohan Anil, Christopher Fifty, and Manfred K Warmuth. Step-size adaptation using exponentiated gradient updates. arXiv preprint arXiv:2202.00145, 2022.   
Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. Advances in neural information processing systems, 29, 2016.   
Hilal Asi and John C Duchi. The importance of better models in stochastic optimization. Proceedings of the National Academy of Sciences, 116(46):22924–22930, 2019.   
Atilim Gunes Baydin, Robert Cornish, David Martinez Rubio, Mark Schmidt, and Frank Wood. Online learning rate adaptation with hypergradient descent. arXiv preprint arXiv:1703.04782, 2017.   
Yoshua Bengio. Gradient-based optimization of hyperparameters. Neural Computation, 12(8):1889–1900, 2000. doi: 10.1162/089976600300015187.   
Leonard Berrada, Andrew Zisserman, and M Pawan Kumar. Training neural networks for and by interpolation. In International Conference on Machine Learning, pp. 799–809. PMLR, 2020.   
Kartik Chandra, Audrey Xie, Jonathan Ragan-Kelley, and Erik Meijer. Gradient descent: The ultimate optimizer. Advances in Neural Information Processing Systems, 35:8214–8225, 2022.

X Chen, C Liang, D Huang, E Real, K Wang, Y Liu, H Pham, X Dong, T Luong, CJ Hsieh, et al. Symbolic discovery of optimization algorithms. arxiv 2023. arXiv preprint arXiv:2302.06675, 2023.

Ashok Cutkosky, Aaron Defazio, and Harsh Mehta. Mechanic: A learning rate tuner. Advances in Neural Information Processing Systems, 36, 2024.

George E Dahl, Frank Schneider, Zachary Nado, Naman Agarwal, Chandramouli Shama Sastry, Philipp Hennig, Sourabh Medapati, Runa Eschenhagen, Priya Kasimbeg, Daniel Suo, et al. Benchmarking neural network training algorithms. arXiv preprint arXiv:2306.07179, 2023.

Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(7):3366–3385, 2021.

Aaron Defazio and Konstantin Mishchenko. Learning-rate-free learning by d-adaptation. In International Conference on Machine Learning, pp. 7449–7479. PMLR, 2023.

Thomas Degris, Khurram Javed, Arsalan Sharifnassab, Yuxin Liu, and Richard Sutton. Step-size optimization for continual learning. arXiv preprint arXiv:2401.17401, 2024.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255. IEEE, 2009. doi: 10.1109/CVPR.2009.5206848.

Michele Donini, Luca Franceschi, Massimiliano Pontil, Orchid Majumder, and Paolo Frasconi. Marthe: Scheduling the learning rate via online hypergradients. arXiv preprint arXiv:1910.08525, 2019.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12(7), 2011.

Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak coherent english? arXiv preprint arXiv:2305.07759, 2023.

Luca Franceschi, Michele Donini, Paolo Frasconi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International Conference on Machine Learning, volume 80, pp. 1568–1577, 2018.

Boyan Gao, Henry Gouk, Hae Beom Lee, and Timothy M Hospedales. Meta mirror descent: Optimiser learning for fast convergence. arXiv preprint arXiv:2203.02711, 2022.   
Geoffrey Hinton. Neural networks for machine learning, lecture 6.5 - rmsprop, 2012. URL https://www.cs.toronto.edu/\~tijmen/csc321/slides/lecture\_slides\_lec6.pdf. Coursera Lecture.   
Jeremy Howard and Sebastian Ruder. Universal language model fine-tuning for text classification. arXiv preprint arXiv:1801.06146, 2018.   
Maor Ivgi, Oliver Hinder, and Yair Carmon. DoG is SGD's best friend: A parameter-free dynamic step size schedule. In International Conference on Machine Learning, pp. 14465–14499. PMLR, 2023.   
Robert A Jacobs. Increased rates of convergence through learning rate adaptation. Neural networks, 1(4):295–307, 1988.   
Khurram Javed. Step-size adaptation for rmsprop. Technical Report, 2020. URL https://khurramjaved.com/reports/idbd\_rmsprop.pdf.   
Khurram Javed, Arsalan Sharifnassab, and Richard S Sutton. Swifttd: A fast and robust algorithm for temporal difference learning. In Reinforcement Learning Conference, 2024.   
Honghe Jin. Hyperparameter importance for machine learning algorithms. arXiv preprint arXiv:2201.05132, 2022.   
Andrej Karpathy. llama2.c: Inference llama 2 in one file of pure c, 2024. URL https://github.com/karpathy/llama2.c. GitHub repository.   
Roland Kehl and Luc Van Gool. Markerless tracking of complex human motions from multiple views. Computer Vision and Image Understanding, 104(2-3):190–209, 2006.   
Harry Kesten. Accelerated stochastic approximation. The Annals of Mathematical Statistics, pp. 41–59, 1958.   
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
A Koop. Investigating Experience: Temporal Coherence and Empirical Knowledge Representation. University of Alberta MSc. PhD thesis, thesis, 2007.   
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

Frederik Kunstner, Victor S Portella, Mark Schmidt, and Nick Harvey. Searching for optimal per-coordinate step-sizes with multidimensional backtracking. arXiv preprint arXiv:2306.02527, 2023.   
Nicolas Loizou, Sharan Vaswani, Issam Hadj Laradji, and Simon Lacoste-Julien. Stochastic polyak step-size for sgd: An adaptive learning rate for fast convergence. In International Conference on Artificial Intelligence and Statistics, pp. 1306–1314. PMLR, 2021.   
Jonathan Lorraine, Paul Vicol, and David Duvenaud. Optimizing millions of hyperparameters by implicit differentiation. In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics, volume 108, pp. 1540–1552. PMLR, 2020.   
Haipeng Luo and Robert E Schapire. Achieving all with no parameters: Adanormalhedge. In Conference on Learning Theory, pp. 1286–1304. PMLR, 2015.   
Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In International Conference on Machine Learning, pp. 2113–2122. PMLR, 2015.   
Ashique Rupam Mahmood, Richard S Sutton, Thomas Degris, and Patrick M Pilarski. Tuning-free step-size adaptation. In International Conference on Acoustics, Speech and Signal Processing, pp. 2121–2124. IEEE, 2012.   
Luke Metz, Niru Maheswaranathan, C Daniel Freeman, Ben Poole, and Jascha Sohl-Dickstein. Tasks, stability, architecture, and compute: Training more effective learned optimizers, and using them to train themselves. arXiv preprint arXiv:2009.11243, 2020. URL https://arxiv.org/abs/2009.11243.   
Luke Metz, C Daniel Freeman, James Harrison, Niru Maheswaranathan, and Jascha Sohl-Dickstein. Practical tradeoffs between memory, compute, and performance in learned optimizers. In Conference on Lifelong Learning Agents, pp. 142–164, 2022. URL https://arxiv.org/abs/2203.11860.   
Paul Micaelli and Amos J Storkey. Gradient-based hyperparameter optimization over long horizons. In Advances in Neural Information Processing Systems, pp. 10798-10809, 2021.   
Konstantin Mishchenko and Aaron Defazio. Prodigy: An expeditiously adaptive parameter-free learner. arXiv preprint arXiv:2306.06101, 2023.   
Francesco Orabona and Dávid Pál. Coin betting and parameter-free online learning. In Advances in Neural Information Processing Systems, volume 29, 2016.

Francesco Orabona and Tatiana Tommasi. Training deep networks without learning rates through coin betting. In Advances in Neural Information Processing Systems, volume 30, 2017.   
Kaan Ozkara, Can Karakus, Parameswaran Raman, Mingyi Hong, Shoham Sabach, Branislav Kveton, and Volkan Cevher. Mada: Meta-adaptive optimizers through hyper-gradient descent. arXiv preprint arXiv:2401.08893, 2024. URL https://arxiv.org/abs/2401.08893.   
Courtney Paquette and Katya Scheinberg. A stochastic line search method with expected complexity analysis. SIAM Journal on Optimization, 30(1):349–376, 2020.   
Fabian Pedregosa. Hyperparameter optimization with approximate gradient. In International Conference on Machine Learning, pp. 737–746. PMLR, 2016.   
Michal Rolinek and Georg Martius. L4: Practical loss-based stepsize adaptation for deep learning. In Advances in Neural Information Processing Systems, 2018.   
Saber Salehkaleybar. Metaoptimize. https://github.com/sabersalehk/MetaOptimize, 2025.   
Tom Schaul, Sixin Zhang, and Yann LeCun. No more pesky learning rates. In International conference on machine learning, pp. 343–351. PMLR, 2013.   
Nicol Schraudolph and Xavier Giannakopoulos. Online independent component analysis with local learning rate adaptation. Advances in neural information processing systems, 12, 1999.   
Bharat Singh, Soham De, Yangmuzi Zhang, Thomas Goldstein, and Gavin Taylor. Layer-specific adaptive learning rates for deep networks. In International Conference on Machine Learning and Applications, pp. 364–368. IEEE, 2015.   
Richard S. Sutton. Adaptation of learning rate parameters. Wright-Patterson Air Force Base, Ohio, 1981. Technical Report AFWAL-TR-81-1070.   
Richard S Sutton. A theory of salience change dependent on the relationship between discrepancies on successive trials on which the stimulus is present. Unpublished working paper, 1982.   
Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44, 1988.   
Richard S Sutton. Adapting bias by gradient descent: An incremental version of delta-bar-delta. In AAAI, volume 92, pp. 171–176. San Jose, CA, 1992.

Richard S Sutton. A history of meta-gradient: Gradient methods for meta-learning. arXiv preprint arXiv:2202.09701, 2022.   
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.   
Sharan Vaswani, Aaron Mishkin, Issam Laradji, Mark Schmidt, Gauthier Gidel, and Simon Lacoste-Julien. Painless stochastic gradient: Interpolation, line-search, and convergence rates. Advances in Neural Information Processing Systems, 2019.   
Zhongwen Xu, Hado P van Hasselt, and David Silver. Meta-gradient reinforcement learning. In Advances in neural Information Processing Systems, 2018.   
Kenny Young, Baoxiang Wang, and Matthew E Taylor. Metatrace: Online step-size tuning by meta-gradient descent for reinforcement learning control. arXiv preprint arXiv:1805.04514, 2018.

# Appendices

# A. Step-size Optimization for Different Choices of Base and Meta updates

In this appendix, we derive $G_{t}$ defined in (9) for different choices of algorithms for base and meta updates, and propose corresponding step-size optimization algorithms.

Consider the following partitions of $G_{t}$ ,

$$
G _ {t} ^ {\text { meta }} \stackrel {\text { def }} {=} \left[ \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}} \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {x} _ {t}} \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \right], \tag {17}
$$

$$
G _ {t} ^ {\text { base }} \stackrel {\text { def }} {=} \left[ \begin{array}{l l l} \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {y} _ {t}} & \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \\ \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {y} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right]. \tag {18}
$$

Then,

$$
G _ {t} = \left[ \begin{array}{l l l} \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}} & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {x} _ {t}} & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \\ \frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}} & \frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {x} _ {t}} & \frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \\ \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}} & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {x} _ {t}} & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \end{array} \right] = \left[ \begin{array}{l} G _ {t} ^ {\text {meta}} \\ \frac {}{} G _ {t} ^ {\text {base}} \end{array} \right]. \tag {19}
$$

In the sequel, we study base and meta updates separately, because $Alg_{base}$ and $Alg_{meta}$ impact disjoint sets of blocks in $G_{t}$ . In particular, as we will see, the choice of $Alg_{base}$ only affects $G^{base}$ while the choice of $Alg_{meta}$ only affects $G^{meta}$ .

Notation conventions in all Appendices: For any vector v, we denote by $[v]$ a diagonal matrix with diagonal entries derived from v. We denote by $\sigma'(\beta_{t})$ the Jacobian of $\alpha_{t}$ with respect to $\beta_{t}$ .

Before delving into computing $G_{t}^{base}$ and $G_{t}^{meta}$ for different base and meta algorithms, we further simplify these matrices.

# A.1. Derivation of $G^{meta}$ for Different Meta Updates

We start by simplifying $G^{meta}$ , and introducing some notations.

Note that the meta update has no dependence on internal variables, $\tilde{x}$ , of the base algorithm. As a result,

$$
\frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \tilde {\boldsymbol {x}} _ {t}} = 0. \tag {20}
$$

Then,

$$
G _ {t} ^ {\text { meta }} = \left[ \begin{array}{l l l} \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}} & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {x} _ {t}} & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \end{array} \right] = \left[ \begin{array}{l l l l} \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}} & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \tilde {\boldsymbol {x}} _ {t}} & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \end{array} \right] = \left[ \begin{array}{l l l l} \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}} & \sum_ {\boldsymbol {w} _ {t}} ^ {\boldsymbol {y} _ {t + 1}} & 0 & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \end{array} \right], \tag {21}
$$

where the third equality is due to (20). Let

$$
L _ {t} \stackrel {\text { def }} {=} \left[ \begin{array}{c c c c} \nabla f _ {t} (\boldsymbol {w} _ {t}) ^ {T} & 0 & 0 & 0 \\ \hline 0 & \nabla f _ {t} (\boldsymbol {w} _ {t}) ^ {T} & 0 & 0 \\ \hline 0 & 0 & \ddots & 0 \\ \hline 0 & 0 & 0 & \nabla f _ {t} (\boldsymbol {w} _ {t}) ^ {T} \end{array} \right] \begin{array}{l l l l} & \leftarrow 1 \\ & \leftarrow 2 \\ & \vdots \\ & \leftarrow m \end{array} \tag {22}
$$

and recall that $h_t$ is a vectorization of $\mathcal{H}_t$ . Then,

$$
\mathcal {H} _ {t} \nabla f _ {t} (\boldsymbol {w} _ {t}) = L _ {t} \boldsymbol {h} _ {t}. \tag {23}
$$

We now proceed to derivation of $G^{meta}$ for different choices of $Alg_{meta}$ .

# A.1.1. Meta SGD

Here, we consider SGD for the meta update (8),

$$
\boldsymbol {\beta} _ {t + 1} = \boldsymbol {\beta} _ {t} - \eta \widehat {\nabla_ {\boldsymbol {\beta}} F} _ {t} = \boldsymbol {\beta} _ {t} - \eta \mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t}), \tag {24}
$$

where $\eta$ is a scalar, called the meta step size. In this case, $y_{t} = \beta_{t}$ . It then follows from (24) that

$$
\frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} = - \eta \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {h} _ {t}} \left(\mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t})\right) = - \eta \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {h} _ {t}} \left(L _ {t} \boldsymbol {h} _ {t}\right) = - \eta L _ {t}, \tag {25}
$$

where the second equality is due to (23). Consequently, from (21), we obtain

$$
\begin{array}{l} G _ {t} ^ {\mathrm{meta}} = \left[ \begin{array}{l l l l} \frac {\mathrm{d} \pmb {y} _ {t + 1}}{\mathrm{d} \pmb {y} _ {t}} & \frac {\mathrm{d} \pmb {y} _ {t + 1}}{\mathrm{d} \pmb {w} _ {t}} & 0 & \frac {\mathrm{d} \pmb {y} _ {t + 1}}{\mathrm{d} \pmb {h} _ {t}} \end{array} \right] \\ = \left[ \begin{array}{c c c c} \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \end{array} \right] \tag {26} \\ = \left[ \begin{array}{l l l l} I & - \eta \mathcal {H} _ {t} ^ {T} \nabla^ {2} f _ {t} (\pmb {w} _ {t}) & 0 & - \eta L _ {t} \end{array} \right], \\ \end{array}
$$

where the last equality follows from (25) and simple differentiations of (24). Here, $\nabla^{2}f_{t}(\boldsymbol{w}_{t})$ denotes the Hessian of $f_{t}$ at $w_{t}$ .

# A.1.2. Meta Adam

The meta update based on the Adam algorithm is as follows,

$$
\begin{array}{l} \bar {\boldsymbol {m}} _ {t + 1} = \bar {\rho} \bar {\boldsymbol {m}} _ {t} + \mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t}), \\ \bar {\boldsymbol {v}} _ {t + 1} = \bar {\lambda} \boldsymbol {v} _ {t} + \left(\mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t})\right) ^ {2}, \\ \bar {\mu} _ {t} = \left(\frac {1 - \bar {\rho}}{1 - \bar {\rho} ^ {t}}\right) / \sqrt {\frac {1 - \bar {\lambda}}{1 - \bar {\lambda} ^ {t}}}, \tag {27} \\ \boldsymbol {\beta} _ {t + 1} = \boldsymbol {\beta} _ {t} - \eta \bar {\mu} _ {t} \frac {\bar {\boldsymbol {m}} _ {t}}{\sqrt {\bar {\boldsymbol {v}} _ {t}}} \\ \end{array}
$$

where $\bar{m}_{t}$ is the momentum vector, $\bar{v}_{t}$ is the trace of squared surrogate-meta-gradient. Since Adam algorithm needs to keep track of $\beta_{t}$ , $\bar{m}_{t}$ , and $\bar{v}_{t}$ , we have

$$
\boldsymbol {y} _ {t} = \left[ \begin{array}{l} \boldsymbol {\beta} _ {t} \\ \bar {\boldsymbol {m}} _ {t} \\ \bar {\boldsymbol {v}} _ {t} \end{array} \right]. \tag {28}
$$

Recall the following notation convention at the end of the Introduction section: for any $k \geq 1$ , and any k-dimensional vector $v = [v_{1}, \ldots, v_{k}]$ , we denote the corresponding diagonal matrix by $[v]$ :

$$
[ \boldsymbol {v} ] \stackrel {\text { def }} {=} \left[ \begin{array}{c c c} v _ {1} & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & v _ {k} \end{array} \right]. \tag {29}
$$

Consequently, from (21), we obtain

$$
\begin{array}{l} G _ {t} ^ {\mathrm{meta}} = \left[ \begin{array}{c c} \frac {\mathrm{d}   \boldsymbol {y} _ {t + 1}}{\mathrm{d}   \boldsymbol {y} _ {t}} & \frac {\mathrm{d}   \boldsymbol {y} _ {t + 1}}{\mathrm{d}   \boldsymbol {w} _ {t}} \\ 0 & \frac {\mathrm{d}   \boldsymbol {y} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right] \\ = \left[ \begin{array}{c c c c c} \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \bar {\boldsymbol {m}} _ {t}} & \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \bar {\boldsymbol {v}} _ {t}} & \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & 0 \\ \frac {\mathrm{d} \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & \frac {\mathrm{d} \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d} \bar {\boldsymbol {m}} _ {t}} & \frac {\mathrm{d} \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d} \bar {\boldsymbol {v}} _ {t}} & \frac {\mathrm{d} \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & 0 \\ \frac {\mathrm{d} \bar {\boldsymbol {v}} t + 1}{\mathrm{d} \boldsymbol {\beta} _ {t}} & \frac {\mathrm{d} \bar {\boldsymbol {v}} t + 1}{\mathrm{d} \bar {\boldsymbol {m}} _ {t}} & \frac {\mathrm{d} \bar {\boldsymbol {v}} t + 1}{\mathrm{d} \bar {\boldsymbol {v}} _ {t}} & \frac {\mathrm{d} \bar {\boldsymbol {v}} t + 1}{\mathrm{d} \boldsymbol {w} _ {t}} & 0 \end{array} \right] \tag {30} \\ = \left[ \begin{array}{c c c c c c} I & - \eta \bar {\mu} _ {t} \left[ \frac {1}{\sqrt {\bar {\boldsymbol {v}} _ {t}}} \right] & \frac {\eta \bar {\mu} _ {t}}{2} \left[ \frac {\bar {\boldsymbol {m}} _ {t}}{\bar {\boldsymbol {v}} _ {t} ^ {1 . 5}} \right] & 0 & 0 & 0 \\ 0 & \bar {\rho} I & 0 & \mathcal {H} _ {t} ^ {T} \nabla^ {2} f _ {t} & 0 & \frac {\mathrm{d}   \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \\ 0 & 0 & \bar {\lambda} I & 2 \big [ \mathcal {H} _ {t} ^ {T} \nabla f _ {t} \big ] \mathcal {H} _ {t} ^ {T} \nabla^ {2} f _ {t} & 0 & \frac {\mathrm{d}   \bar {\boldsymbol {v}} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right], \\ \end{array}
$$

where the last equality follows by calculating derivatives of (27). For the two remaining terms in the last column of $G_{t}$ , we have

$$
\frac {\mathrm{d} \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} = \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {h} _ {t}} \left(\mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t})\right) = \eta \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {h} _ {t}} \left(L _ {t} \boldsymbol {h} _ {t}\right) = \eta L _ {t}. \tag {31}
$$

where the first equality follows from the update of $\bar{\pmb{m}}_{t + 1}$ in (27), and the second equality is due to (23). In the same vein,

$$
\frac {\mathrm{d} \bar {\boldsymbol {v}} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} = \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {h} _ {t}} \left(\mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t})\right) ^ {2} = \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {h} _ {t}} \left(L _ {t} \boldsymbol {h} _ {t}\right) ^ {2} = 2 \left[ L _ {t} \boldsymbol {h} _ {t} \right] \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {h} _ {t}} \left(L _ {t} \boldsymbol {h} _ {t}\right) = 2 \left[ L _ {t} \boldsymbol {h} _ {t} \right] L _ {t} = 2 \left[ \mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t}) \right] L _ {t}, \tag {32}
$$

where the first equality follows from the update of $\bar{v}_{t+1}$ in (27), the second equality is due to (23), and the last equality is again from (23).

Plugging (31) and (32) into (30), we obtain

$$
G _ {t} ^ {\text { meta }} = \left[ \begin{array}{c c c c c c} I & - \eta \bar {\mu} _ {t} \left[ \frac {1}{\sqrt {\bar {\boldsymbol {v}} _ {t}}} \right] & \frac {\eta \bar {\mu} _ {t}}{2} \left[ \frac {\bar {\boldsymbol {m}} _ {t}}{\bar {\boldsymbol {v}} _ {t} ^ {1 . 5}} \right] & 0 & 0 & 0 \\ 0 & \bar {\rho} I & 0 & \mathcal {H} _ {t} ^ {T} \nabla^ {2} f _ {t} & 0 & \eta L _ {t} \\ 0 & 0 & \bar {\lambda} I & 2 \left[ \mathcal {H} _ {t} ^ {T} \nabla f _ {t} \right] \mathcal {H} _ {t} ^ {T} \nabla^ {2} f _ {t} & 0 & 2 \left[ \mathcal {H} _ {t} ^ {T} \nabla f _ {t} \right] L _ {t} \end{array} \right]. \tag {33}
$$

# A.1.3. Meta Lion

The meta update based on the lion algorithm is as follows

$$
\bar {\boldsymbol {m}} _ {t + 1} = \rho \bar {\boldsymbol {m}} _ {t} + (1 - \rho) \widehat {\nabla_ {\beta} F _ {t}}, \tag {34}
$$

$$
\boldsymbol {\beta} _ {t + 1} = \boldsymbol {\beta} _ {t} - \eta \operatorname{Sign} \left(c \bar {\boldsymbol {m}} _ {t} + (1 - c) \widehat {\nabla_ {\boldsymbol {\beta}} F} _ {t}\right), \tag {35}
$$

where $\eta$ is a scalar, called the meta step size, and $\rho, c \in [0,1)$ . Note that the meta algorithm operates on a low dimensional space. Therefore, we drop the regularizers like weight-decay in the meta updates, as they are primarily aimed to resolve the overfitting problem in high dimensional problems. Substituting $\widehat{\nabla_{\beta}F_{t}}$ with $\mathcal{H}_{t}^{T}\nabla f_{t}(\boldsymbol{w}_{t})$ we obtain the following meta updates

$$
\bar {\boldsymbol {m}} _ {t + 1} = \rho \bar {\boldsymbol {m}} _ {t} + (1 - \rho) \mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t}), \tag {36}
$$

$$
\boldsymbol {\beta} _ {t + 1} = \boldsymbol {\beta} _ {t} - \eta \operatorname{Sign} \left(c \bar {\boldsymbol {m}} _ {t} + (1 - c) \mathcal {H} _ {t} ^ {T} \nabla f _ {t} (\boldsymbol {w} _ {t})\right). \tag {37}
$$

In this case,

$$
\boldsymbol {y} _ {t} = \left[ \begin{array}{c} \boldsymbol {\beta} _ {t} \\ \bar {\boldsymbol {m}} _ {t} \end{array} \right],
$$

and

$$
\begin{array}{l} G _ {t} ^ {\mathrm{meta}} = \left[ \begin{array}{l l l l} \frac {\mathrm{d} \pmb {y} _ {t + 1}}{\mathrm{d} \pmb {y} _ {t}} & \frac {\mathrm{d} \pmb {y} _ {t + 1}}{\mathrm{d} \pmb {w} _ {t}} & 0 & \frac {\mathrm{d} \pmb {y} _ {t + 1}}{\mathrm{d} \pmb {h} _ {t}} \end{array} \right] \\ = \left[ \begin{array}{c c c c c} \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \bar {\boldsymbol {m}} _ {t}} & \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {\beta} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \\ \frac {\mathrm{d} \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & \frac {\mathrm{d} \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d} \bar {\boldsymbol {m}} _ {t}} & \frac {\mathrm{d} \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & 0 & \frac {\mathrm{d} \bar {\boldsymbol {m}} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \end{array} \right] \tag {38} \\ = \left[ \begin{array}{c c c c c} I & 0 & 0 & 0 & 0 \\ \frac {\mathrm{d}   \bar {m} _ {t + 1}}{\mathrm{d}   \beta_ {t}} & \frac {\mathrm{d}   \bar {m} _ {t + 1}}{\mathrm{d}   \bar {m} _ {t}} & \frac {\mathrm{d}   \bar {m} _ {t + 1}}{\mathrm{d}   w _ {t}} & 0 & \frac {\mathrm{d}   \bar {m} _ {t + 1}}{\mathrm{d}   h _ {t}} \end{array} \right], \\ \end{array}
$$

where the last equality follows from (37). Consider the following block representation of $Y_{t}$ :

$$
Y _ {t} = \left[ \begin{array}{c} B _ {t} \\ Y _ {t} ^ {\bar {m}} \end{array} \right]. \tag {39}
$$

Since the base algorithm, does not take $\bar{m}$ as input, as we will see in (41) and (42) of next subsection (Appendix A.2), $\frac{d\bar{m}_{t+1}}{d\bar{m}_{t}}$ is the only non-zero block of $G_{t}$ in its column of blocks (i.e., $\frac{ds_{t+1}}{d\bar{m}_{t}} = 0$ for every variable s other than $\bar{m}$ ). Consequently, it follows from (14) that $Y_{t}^{\bar{m}}$ as defined in (39), has no impact on the update of $X_{t+1}$ , $B_{t+1}$ , and $Q_{t+1}$ . Therefore, we can zero-out the rows and columns of $G^{meta}$ that correspond to derivative of $\bar{m}$ . As such we obtain the following equivalent of $G^{meta}$ in (38) from an algorithmic perspective:

$$
G _ {t} ^ {\text { meta }} \equiv \left[ \begin{array}{c c} I _ {m \times m} & 0 \\ 0 & 0 \end{array} \right]. \tag {40}
$$

As a result, we get $B_{t} = I$ for all times $t$ .

# A.2. Derivation of $G^{base}$ for Different Base Updates

We now turn our focus to computation of $G^{\mathrm{base}}$ . Let us start by simplifying $G^{\mathrm{base}}$ , and introducing some notations.

Note that the base update has no dependence on internal variables, $\tilde{y}$ , of the meta update. As a result,

$$
\frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \tilde {\boldsymbol {y}} _ {t}} = 0. \tag {41}
$$

Moreover, it follows from the definition of $\mathcal{H}_t$ in (7) that

$$
\frac {\mathrm{d} \mathcal {H} _ {t + 1}}{\mathrm{d} \tilde {\boldsymbol {y}} _ {t}} = (1 - \gamma) \sum_ {t = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d}}{d \tilde {\boldsymbol {y}} _ {t}} \left(\frac {d \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}}\right) = (1 - \gamma) \sum_ {t = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d}}{d \boldsymbol {\beta} _ {\tau}} \left(\frac {d \boldsymbol {w} _ {t + 1}}{\mathrm{d} \tilde {\boldsymbol {y}} _ {t}}\right) = (1 - \gamma) \sum_ {t = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d}}{d \boldsymbol {\beta} _ {\tau}} (0) = 0,
$$

where the third equality follows from (41). Therefore,

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \tilde {\boldsymbol {y}} _ {t}} = 0. \tag {42}
$$

Note also that $Alg_{base}$ does not take $H_{t}$ as input, and therefore,

$$
\frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} = 0. \tag {43}
$$

Consequently, we can simplify $G_{t}^{base}$ as follows,

$$
G _ {t} ^ {\text {base}} = \left[\begin{array}{l l l}\frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}}&\frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {x} _ {t}}&\frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}}\\\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}}&\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {x} _ {t}}&\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}}\end{array}\right] = \left[\begin{array}{l l l l}\frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}}&\frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \tilde {\boldsymbol {y}} _ {t}}&\frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {x} _ {t}}&\frac {\mathrm{d} \boldsymbol {x} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}}\\\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm {d\beta_ {t}}}&\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \tilde {\boldsymbol {y}} _ {t}}&\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {x} _ {t}}&\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}}\end{array}\right] = \left[ \right.\begin{array}{l l l l}\end{array}
$$

where the last equality is due to (41), (42), and (43).

On an independent note, consider the following block representation of $Y_{t}$ ,

$$
Y _ {t} = \left[ \begin{array}{c} B _ {t} - \frac {1 - \gamma}{\gamma} I \\ \tilde {Y} _ {t} \end{array} \right], \tag {45}
$$

Therefore,

$$
\gamma Y _ {t} + (1 - \gamma) \left[ \begin{array}{l} I \\ 0 \end{array} \right] = \gamma \left[ \begin{array}{l} B _ {t} \\ \tilde {Y} _ {t} \end{array} \right]
$$

It then follows from (19) and (14) that

$$
\left[ \begin{array}{l} X _ {t + 1} \\ Q _ {t + 1} \end{array} \right] = \gamma G _ {t} ^ {\text { base }} \left[ \begin{array}{c} \left[ \begin{array}{l} B _ {t} \\ \tilde {Y} _ {t} \end{array} \right] \\ X _ {t} \\ Q _ {t} \end{array} \right]. \tag {46}
$$

Moreover, from the definition of $Y_{t}$ in (11), we have

$$
\frac {\mathrm{d} B _ {t}}{\mathrm{d} \pmb {x} _ {t}} = (1 - \gamma) \frac {\mathrm{d}}{\mathrm{d} \pmb {x} _ {t}} \sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d} \pmb {\beta} _ {t}}{\mathrm{d} \pmb {\beta} _ {\tau}} = (1 - \gamma) \sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d}}{\mathrm{d} \pmb {\beta} _ {\tau}} \left(\frac {\mathrm{d} \pmb {\beta} _ {t}}{\mathrm{d} \pmb {x} _ {t}}\right) = (1 - \gamma) \sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d}}{\mathrm{d} \pmb {\beta} _ {\tau}} (0) = 0,
$$

$$
\frac {\mathrm{d} B _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {t}} = (1 - \gamma) \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {\beta} _ {t}} \sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d} \boldsymbol {\beta} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} = (1 - \gamma) \sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} \left(\frac {\mathrm{d} \boldsymbol {\beta} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {t}}\right) = (1 - \gamma) \sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d}}{\mathrm{d} \boldsymbol {\beta} _ {\tau}} (I) = 0, \tag {47}
$$

$$
\frac {\mathrm{d} B _ {t}}{\mathrm{d} \pmb {h} _ {t}} = (1 - \gamma) \frac {\mathrm{d}}{\mathrm{d} \pmb {h} _ {t}} \sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d} \pmb {\beta} _ {t}}{\mathrm{d} \pmb {\beta} _ {\tau}} = (1 - \gamma) \sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d}}{\mathrm{d} \pmb {\beta} _ {\tau}} \left(\frac {\mathrm{d} \pmb {\beta} _ {t}}{\mathrm{d} \pmb {h} _ {t}}\right) = (1 - \gamma) \sum_ {\tau = 0} ^ {t} \gamma^ {t - \tau} \frac {\mathrm{d}}{\mathrm{d} \pmb {\beta} _ {\tau}} (0) = 0.
$$

Finally, recall the definition

$$
\sigma^ {\prime} \left(\boldsymbol {\beta} _ {t}\right) \stackrel {\text { def }} {=} \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \boldsymbol {\beta} _ {t}} \tag {48}
$$

as the Jacobian of $\alpha_{t}$ with respect to $\beta_{t}$ .

We now proceed to derivation of $G^{\mathrm{base}}$ for different choices of $\mathrm{Alg}_{\mathrm{base}}$ .

# A.3. Base SGD

Base SGD algorithm makes the following base update in each iteration:

$$
\boldsymbol {w} _ {t + 1} = \boldsymbol {w} _ {t} - \boldsymbol {\alpha} _ {t} \nabla f _ {t} (\boldsymbol {w} _ {t}). \tag {49}
$$

In this case, $\pmb{x}_t = \pmb{w}_t$ and $X_{t} = \mathcal{H}_{t}$ . Then, $G_{t}^{\mathrm{base}}$ in (44) can be simplified to

$$
\begin{array}{l} G _ {t} ^ {\mathrm{base}} = \left[ \begin{array}{c c c c} \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & 0 \\ \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right] \\ = \left[ \begin{array}{c c c c} \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & 0 \\ \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \end{array} \right] \tag {50} \\ = \left[ \begin{array}{c c c c} - [ \nabla f _ {t} (\boldsymbol {w} _ {t}) ]   \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) & 0 & I - [ \boldsymbol {\alpha} _ {t} ]   \nabla^ {2} f _ {t} (\boldsymbol {w} _ {t}) & 0 \\ \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {w} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right], \\ \end{array}
$$

where the last equality follows by computing simple derivatives of $w_{t+1}$ in (49).

We proceed to compute the three remaining entries of $G_{t}^{\mathrm{base}}$ , i.e., $\mathrm{d}\pmb{h}_{t + 1} / \mathrm{d}\beta_t$ , $\mathrm{d}\pmb{h}_{t + 1} / \mathrm{d}\pmb{w}_t$ , and $\mathrm{d}\pmb{h}_{t + 1} / \mathrm{d}\pmb{h}_t$ . Note that by plugging the first row of $G_{t}^{\mathrm{base}}$ , given in (50), into (46), and noting that $\mathcal{H}_t = X_t$ , we obtain

$$
\mathcal {H} _ {t + 1} = \gamma (I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t} (\boldsymbol {w} _ {t})) \mathcal {H} _ {t} - \gamma [ \nabla f _ {t} (\boldsymbol {w} _ {t}) ] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t}, \tag {51}
$$

for all $t \geq 0$ . By vectorizing both sides of (51) we obtain

$$
\boldsymbol {h} _ {t + 1} = \gamma \left[ \begin{array}{c} \frac {\left(I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t}\right) \mathcal {H} _ {t} ^ {[ 1 ]} - [ \nabla f _ {t} ] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 1 ]}}{\left(I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t}\right) \mathcal {H} _ {t} ^ {[ 2 ]} - [ \nabla f _ {t} ] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 2 ]}} \\ \vdots \\ \frac {\left(I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t}\right) \mathcal {H} _ {t} ^ {[ m ]} - [ \nabla f _ {t} ] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ m ]}}{\left(I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t}\right) \mathcal {H} _ {t} ^ {[ m ]}} \end{array} \right]. \tag {52}
$$

Note that for any pair of same-size vectors a and b, we have $[a]$ $b = [b]$ a where $[a]$ and $[b]$ are diagonal matrices of a and b, respectively. Therefore, (52) can be equivalently written in the following form

$$
\boldsymbol {h} _ {t + 1} = \gamma \left[ \begin{array}{c} \frac {\left(I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t}\right) \mathcal {H} _ {t} ^ {[ 1 ]} - \left[ \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 1 ]} \right] \nabla f _ {t}}{\vdots} \\ \hline \left(I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t}\right) \mathcal {H} _ {t} ^ {[ m ]} - \left[ \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ m ]} \right] \nabla f _ {t} \end{array} \right]. \tag {53}
$$

By taking the derivative of (52) with respect to $h_t$ , we obtain

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} = \gamma \left[ \begin{array}{c c c c} I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t} (\boldsymbol {w} _ {t}) & 0 & 0 & 0 \\ \hline 0 & I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t} (\boldsymbol {w} _ {t}) & 0 & 0 \\ \hline 0 & 0 & \ddots & 0 \\ \hline 0 & 0 & 0 & I - [ \boldsymbol {\alpha} _ {t} ] \nabla^ {2} f _ {t} (\boldsymbol {w} _ {t}) \end{array} \right] \quad \begin{array}{l l l l} & \leftarrow 1 \mathrm{st} \\ & \leftarrow 2 \mathrm{nd} \\ & \vdots \\ & \leftarrow m \mathrm{th} \end{array} . \tag {54}
$$

In the above equation, note that $d B_{t}/d h_{t}=0$ due to (47). Let $\beta_{t}[i]$ and $w_{t}[j]$ denote the ith and jth entries of $\beta_{t}$ and $w_{t}$ , for $i=1,\ldots,m$ and $j=1,\ldots,n$ , respectively. It then follows from (52) and (47) that

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} = - \gamma \left[ \begin{array}{c c c} \left[ \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ 1 ]} \right] \nabla^ {2} f _ {t} \mathcal {H} _ {t} ^ {[ 1 ]} + [ \nabla f _ {t} ] \frac {\partial \sigma^ {\prime} (\boldsymbol {\beta} _ {t})}{\partial \beta_ {t} [ 1 ]} B _ {t} ^ {[ 1 ]} & \dots & \left[ \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ m ]} \right] \nabla^ {2} f _ {t} \mathcal {H} _ {t} ^ {[ 1 ]} + [ \nabla f _ {t} ] \frac {\partial \sigma^ {\prime} (\boldsymbol {\beta} _ {t})}{\partial \beta_ {t} [ m ]} B _ {t} ^ {[ 1 ]} \\ \hline \vdots & \ddots & \vdots \\ \hline \left[ \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ 1 ]} \right] \nabla^ {2} f _ {t} \mathcal {H} _ {t} ^ {[ m ]} + [ \nabla f _ {t} ] \frac {\partial \sigma^ {\prime} (\boldsymbol {\beta} _ {t})}{\partial \beta_ {t} [ 1 ]} B _ {t} ^ {[ m ]} & \dots & \left[ \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ m ]} \right] \nabla^ {2} f _ {t} \mathcal {H} _ {t} ^ {[ m ]} + [ \nabla f _ {t} ] \frac {\partial \sigma^ {\prime} (\boldsymbol {\beta} _ {t})}{\partial \beta_ {t} [ m ]} B _ {t} ^ {[ m ]} \end{array} \right], \tag {55}
$$

where $\frac{\partial}{\partial\beta}$ stands for the entry-wise partial derivative of a matrix with respect to a scalar variable $\beta$ . In the same vein, (53) and (47) imply that

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} = - \gamma \left[ \begin{array}{c} \left[ \boldsymbol {\alpha} _ {t} \right] \frac {\mathrm{d} \left(\nabla^ {2} f _ {t} (\boldsymbol {w} _ {t}) \mathcal {H} _ {t} ^ {[ 1 ]}\right)}{\mathrm{d} \boldsymbol {w} _ {t}} + \left[ \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 1 ]} \right] \nabla^ {2} f _ {t} (\boldsymbol {w} _ {t}) \\ \hline \vdots \\ \hline \left[ \boldsymbol {\alpha} _ {t} \right] \frac {\mathrm{d} \left(\nabla^ {2} f _ {t} (\boldsymbol {w} _ {t}) \mathcal {H} _ {t} ^ {[ m ]}\right)}{\mathrm{d} \boldsymbol {w} _ {t}} + \left[ \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ m ]} \right] \nabla^ {2} f _ {t} (\boldsymbol {w} _ {t}) \end{array} \right]. \tag {56}
$$

Finally, $G_{t}^{\mathrm{base}}$ is obtained by plugging (54), (55), and (56) into (50).

In the special case that $\beta$ is a scalar (equivalently m = 1), and furthermore $\alpha = \sigma(\beta) = e^{\beta}$ , matrix $G_{t}^{base}$ would be simplified to

$$
G _ {t} ^ {\mathrm{base(scalar)}} = \left[ \begin{array}{c c c} 1 & - \eta   \pmb {h} _ {t} ^ {T} \nabla^ {2} f _ {t} (\pmb {w} _ {t}) & - \eta   \nabla f _ {t} (\pmb {w} _ {t}) ^ {T} \\ - \alpha \nabla f _ {t} (\pmb {w} _ {t}) & I - \alpha \nabla^ {2} f _ {t} (\pmb {w} _ {t}) & 0 \\ - \gamma \alpha \nabla^ {2} f _ {t} (\pmb {w} _ {t}) \pmb {h} _ {t} - B _ {t}   \alpha \nabla f _ {t} (\pmb {w} _ {t}) & - \gamma \alpha \frac {\mathrm{d}   \big (\nabla^ {2} f _ {t} (\pmb {w} _ {t}) \pmb {h} _ {t} \big)}{d \pmb {w} _ {t}} - B _ {t}   \alpha \nabla^ {2} f _ {t} (\pmb {w} _ {t}) & \gamma \big (I - \alpha \nabla^ {2} f _ {t} (\pmb {w} _ {t}) \big) \end{array} \right].
$$

# A.3.1. Base AdamW

The base update according to the AdamW algorithm (Loizou et al., 2021) is as follows,

$$
\boldsymbol {m} _ {t + 1} = \rho \boldsymbol {m} _ {t} + \nabla f _ {t} (\boldsymbol {w} _ {t}),
$$

$$
\boldsymbol {v} _ {t + 1} = \lambda \boldsymbol {v} _ {t} + \nabla f _ {t} (\boldsymbol {w} _ {t}) ^ {2},
$$

$$
\mu_ {t} = \left(\frac {1 - \rho}{1 - \rho^ {t}}\right) / \sqrt {\frac {1 - \lambda}{1 - \lambda^ {t}}}, \tag {57}
$$

$$
\boldsymbol {w} _ {t + 1} = \boldsymbol {w} _ {t} - \alpha_ {t} \mu_ {t} \frac {\boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}}} - \kappa \alpha_ {t} \boldsymbol {w} _ {t},
$$

where $m_{t}$ is the momentum vector, $v_{t}$ is the trace of gradient square used for normalization, and $\kappa > 0$ is a weight-decay parameter. Therefore the base algorithm needs to keep track of $w_{t}, m_{t}, v_{t}$ , i.e.,

$$
\boldsymbol {x} _ {t} = \left[ \begin{array}{l} \boldsymbol {w} _ {t} \\ \boldsymbol {m} _ {t} \\ \boldsymbol {v} _ {t} \end{array} \right]. \tag {58}
$$

It then follows from (44) that

$$
G _ {t} ^ {\mathrm{base}} = \left[ \begin{array}{c c c c} \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & 0 \\ \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right]
$$

$$
= \left[ \begin{array}{c c c c c} \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {m} _ {t}} & \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {v} _ {t}} \\ \frac {\mathrm{d} \boldsymbol {m} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {m} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {m} _ {t + 1}}{\mathrm{d} \boldsymbol {m} _ {t}} & \frac {\mathrm{d} \boldsymbol {m} _ {t + 1}}{\mathrm{d} \boldsymbol {v} _ {t}} \\ \frac {\mathrm{d} \boldsymbol {v} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {v} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {v} _ {t + 1}}{\mathrm{d} \boldsymbol {m} _ {t}} & \frac {\mathrm{d} \boldsymbol {v} _ {t + 1}}{\mathrm{d} \boldsymbol {v} _ {t}} \\ \hline \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {m} _ {t}} & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {v} _ {t}} \\ \hline \end{array} \right] \tag {59}
$$

$$
= \left[ \begin{array}{c c c c c} - \mu_ {t} \left[ \frac {\boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}}} + \kappa \boldsymbol {w} _ {t} \right] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) & 0 & I - \kappa [ \boldsymbol {\alpha} _ {t} ] & - \mu_ {t} \left[ \frac {\boldsymbol {\alpha} _ {t}}{\sqrt {\boldsymbol {v} _ {t}}} \right] & \frac {\mu_ {t}}{2} \left[ \frac {\boldsymbol {\alpha} _ {t} \boldsymbol {m} _ {t}}{\boldsymbol {v} _ {t} ^ {1 . 5}} \right] & 0 \\ 0 & 0 & \nabla^ {2} f _ {t} & \rho I & 0 & 0 \\ 0 & 0 & 2 [ \nabla f _ {t} ]   \nabla^ {2} f _ {t} & 0 & \lambda I & 0 \\ \hline \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {w} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {m} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {v} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right]
$$

where the last equality follows from simple derivative computations in (57).

We proceed to compute the terms in the last row of the $G_{t}^{base}$ above. Consider the following block representation of $X_{t}$ ,

$$
X _ {t} = \left[ \begin{array}{c} \mathcal {H} _ {t} \\ X _ {t} ^ {m} \\ X _ {t} ^ {v} \end{array} \right], \tag {60}
$$

Plugging the first row of $G_{t}^{\mathrm{base}}$ , given in (59), into (46), implies that

$$
\mathcal {H} _ {t + 1} = - \gamma \mu_ {t} \left[ \frac {\boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}}} + \kappa \boldsymbol {w} _ {t} \right] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} + \gamma (I - \kappa [ \boldsymbol {\alpha} _ {t} ]) \mathcal {H} _ {t} - \gamma \mu_ {t} \left[ \frac {\boldsymbol {\alpha} _ {t}}{\sqrt {\boldsymbol {v} _ {t}}} \right] X _ {t} ^ {m} + \gamma \frac {\mu_ {t}}{2} \left[ \frac {\boldsymbol {\alpha} _ {t} \boldsymbol {m} _ {t}}{\boldsymbol {v} _ {t} ^ {1 . 5}} \right] X _ {t} ^ {v}. \tag {61}
$$

for all $t \geq 0$ . Note that for any pair of same-size vectors $\pmb{a}$ and $\pmb{b}$ , we have $[\pmb{a}] \pmb{b} = [\pmb{b}] \pmb{a}$ where $[\pmb{a}]$ and $[\pmb{b}]$ are diagonal matrices of $\pmb{a}$ and $\pmb{b}$ , respectively. Therefore, the $i$ th column in the matrix equation (61) can be equivalently written as

$$
\mathcal {H} _ {t + 1} ^ {[ i ]} = - \gamma \mu_ {t} \left[ \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ i ]} \right] \frac {\boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}}} + \kappa \boldsymbol {w} _ {t} + \gamma (I - \kappa [ \boldsymbol {\alpha} _ {t} ]) \mathcal {H} _ {t} ^ {[ i ]} - \gamma \mu_ {t} \left[ X _ {t} ^ {m [ i ]} \right] \frac {\boldsymbol {\alpha} _ {t}}{\sqrt {\boldsymbol {v} _ {t}}} + \gamma \frac {\mu_ {t}}{2} \left[ X _ {t} ^ {v [ i ]} \right] \frac {\boldsymbol {\alpha} _ {t} \boldsymbol {m} _ {t}}{\boldsymbol {v} _ {t} ^ {1 . 5}}, \tag {62}
$$

where $B_{t}^{[i]}, \mathcal{H}_{t}^{[i]}, X_{t}^{m[i]}$ , and $X_{t}^{v[i]}$ stand for the $i$ th columns of $B_{t}, \mathcal{H}_{t}, X_{t}^{m}$ , and $X_{t}^{v}$ , respectively. Following similar arguments as in (47), it is easy to show that

$$
\begin{array}{l} \frac {\mathrm{d} X _ {t} ^ {m}}{\mathrm{d} \boldsymbol {\beta} _ {t}} = \frac {\mathrm{d} X _ {t} ^ {v}}{\mathrm{d} \boldsymbol {\beta} _ {t}} = 0, \\ \frac {\mathrm{d} X _ {t} ^ {m}}{\mathrm{d} \boldsymbol {w} _ {t}} = \frac {\mathrm{d} X _ {t} ^ {v}}{\mathrm{d} \boldsymbol {w} _ {t}} = 0, \\ \frac {\mathrm{d} X _ {t} ^ {m}}{\mathrm{d} \boldsymbol {m} _ {t}} = \frac {\mathrm{d} X _ {t} ^ {v}}{\mathrm{d} \boldsymbol {m} _ {t}} = 0, \tag {63} \\ \frac {\mathrm{d} X _ {t} ^ {m}}{\mathrm{d} \boldsymbol {v} _ {t}} = \frac {\mathrm{d} X _ {t} ^ {v}}{\mathrm{d} \boldsymbol {v} _ {t}} = 0, \\ \frac {\mathrm{d} X _ {t} ^ {m}}{\mathrm{d} \boldsymbol {h} _ {t}} = \frac {\mathrm{d} X _ {t} ^ {v}}{\mathrm{d} \boldsymbol {h} _ {t}} = 0. \\ \end{array}
$$

Note that $h_{t}$ is an nm-dimensional vector derived from stacking the columns of $H_{t}$ . Therefore, we consider a block representation of $h_{t}$ consisting of m blocks, each of which corresponds to a column of $H_{t}$ . By taking the derivative of (61) with respect to $h_{t}$ , and using (63), we obtain

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} = \gamma \left[ \begin{array}{c c c c} I - \kappa [ \boldsymbol {\alpha} _ {t} ] & 0 & 0 & 0 \\ \hline 0 & I - \kappa [ \boldsymbol {\alpha} _ {t} ] & 0 & 0 \\ \hline 0 & 0 & \ddots & 0 \\ \hline 0 & 0 & 0 & I - \kappa [ \boldsymbol {\alpha} _ {t} ] \end{array} \right] \quad \begin{array}{l} \leftarrow 1 \text {st} \\ \leftarrow 2 \text {nd} \\ \vdots \\ \leftarrow m \text {th} \end{array} . \tag {64}
$$

Let $\beta_t[i]$ and $w_{t}[j]$ denote the $i$ th and $j$ th entries of $\beta_{t}$ and $\boldsymbol{w}_{t}$ , for $i = 1, \ldots, m$ and $j = 1, \ldots, n$ , respectively. Note that $\mathrm{d} \boldsymbol{h}_{t+1}/\mathrm{d} \boldsymbol{\beta}_{t}$ is a block matrix, in the form of an $m \times m$ array of $n \times 1$ blocks, $\frac{\mathrm{d} \boldsymbol{h}_{t+1}}{\mathrm{d} \boldsymbol{\beta}_{t}}[i, j] \stackrel{\text{def}}{=} \frac{\mathrm{d} \mathcal{H}_{t+1}^{[i]}}{\mathrm{d} \beta_{t}[j]}$ , for $i, j = 1, \ldots, m$ . It then follows from (61) and (63) that, for $i, j = 1, \ldots, m$ ,

$$
\begin{array}{l} \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \beta_ {t}} [ i, j ] = \frac {\mathrm{d} \mathcal {H} _ {t + 1} ^ {[ i ]}}{\mathrm{d} \beta_ {t} [ j ]} \\ = - \gamma \mu_ {t} \left[ \frac {\boldsymbol {m} _ {t}}{\sqrt {\boldsymbol {v} _ {t}}} + \kappa \boldsymbol {w} _ {t} \right] \left(\frac {\partial \sigma^ {\prime} (\boldsymbol {\beta} _ {t})}{\partial \beta_ {t} [ j ]}\right) B _ {t} ^ {[ i ]} + \gamma \left(I - \kappa \left[ \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ j ]} \right]\right) \mathcal {H} _ {t} ^ {[ i ]} \tag {65} \\ - \gamma \mu_ {t} \Big [ \frac {1}{\sqrt {\pmb {v} _ {t}}} \Big ] \Big [ \frac {\mathrm{d} \pmb {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ j ]} \Big ] X _ {t} ^ {m [ i ]} + \gamma \frac {\mu_ {t}}{2} \Big [ \frac {\pmb {m} _ {t}}{\pmb {v} _ {t} ^ {1 . 5}} \Big ] \Big [ \frac {\mathrm{d} \pmb {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ j ]} \Big ] X _ {t} ^ {v [ i ]}, \\ \end{array}
$$

where $\frac{\partial}{\partial\beta}$ stands for the entry-wise partial derivative of a matrix with respect to a scalar variable $\beta$ .

In the same vein, it follows from (62) and (63) that

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} = - \gamma \mu_ {t} \kappa \left[ \begin{array}{c} \left[ \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 1 ]} \right] \\ \hline \vdots \\ \hline \left[ \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ m ]} \right] \end{array} \right], \tag {66}
$$

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {m} _ {t}} = \gamma \mu_ {t} \left[ \begin{array}{c} \left[ \frac {\boldsymbol {\alpha} _ {t} X _ {t} ^ {v [ 1 ]}}{2 \boldsymbol {v} _ {t} ^ {1 . 5}} - \frac {\sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 1 ]}}{\sqrt {\boldsymbol {v} _ {t}}} \right] \\ \hline \vdots \\ \hline \left[ \frac {\boldsymbol {\alpha} _ {t} X _ {t} ^ {v [ m ]}}{2 \boldsymbol {v} _ {t} ^ {1 . 5}} - \frac {\sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ m ]}}{\sqrt {\boldsymbol {v} _ {t}}} \right] \end{array} \right], \tag {67}
$$

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {v} _ {t}} = \frac {\gamma \mu_ {t}}{2} \left[ \begin{array}{c} \left[ \frac {1}{\boldsymbol {v} _ {t} ^ {1 . 5}} \right] \left[ \left(\sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 1 ]}\right) \boldsymbol {m} _ {t} + \boldsymbol {\alpha} _ {t} X _ {t} ^ {m [ 1 ]} - \frac {3 \boldsymbol {\alpha} _ {t} \boldsymbol {m} _ {t} X _ {t} ^ {v [ 1 ]}}{2 \boldsymbol {v} _ {t}} \right] \\ \hline \vdots \\ \hline \left[ \frac {1}{\boldsymbol {v} _ {t} ^ {1 . 5}} \right] \left[ \left(\sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ m ]}\right) \boldsymbol {m} _ {t} + \boldsymbol {\alpha} _ {t} X _ {t} ^ {m [ m ]} - \frac {3 \boldsymbol {\alpha} _ {t} \boldsymbol {m} _ {t} X _ {t} ^ {v [ m ]}}{2 \boldsymbol {v} _ {t}} \right] \end{array} \right]. \tag {68}
$$

Finally, $G_{t}^{\mathrm{base}}$ is obtained by plugging (64), (65), (66), (67), and (68) into (59).

# A.3.2. Base Lion

The lion algorithm, when used for base update, is as follows

$$
\boldsymbol {m} _ {t + 1} = \rho \boldsymbol {m} _ {t} + (1 - \rho) \nabla f _ {t} (\boldsymbol {w} _ {t}), \tag {69}
$$

$$
\boldsymbol {w} _ {t + 1} = \boldsymbol {w} _ {t} - \boldsymbol {\alpha} _ {t} \operatorname{Sign} (c \boldsymbol {m} _ {t} + (1 - c) \nabla f _ {t}) - \kappa \boldsymbol {\alpha} _ {t} \boldsymbol {w} _ {t}, \tag {70}
$$

where $m_{t}$ is called the momentum, $\kappa > 0$ is the weight-decay parameter, $\rho, c \in [0,1)$ are constants, and $\operatorname{Sign}(\cdot)$ is a function that computes entry-wise sign of a vector. Let

$$
\boldsymbol {x} _ {t} = \left[ \begin{array}{l} \boldsymbol {w} _ {t} \\ \boldsymbol {m} _ {t} \end{array} \right]. \tag {71}
$$

It then follows from (44) that

$$
\begin{array}{l} G _ {t} ^ {\text {base}} = \left[ \begin{array}{c c c c} \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {x} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & 0 \\ \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {x} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right] \\ = \left[ \begin{array}{c c c c c} \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \boldsymbol {m} _ {t}} & 0 \\ \frac {\mathrm{d} \boldsymbol {m} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {m} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {m} _ {t + 1}}{\mathrm{d} \boldsymbol {m} _ {t}} & 0 \\ \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {m} _ {t}} & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \end{array} \right] \tag {72} \\ = \left[ \begin{array}{c c c c c} - \big [ \operatorname{Sign} \big (c   \boldsymbol {m} _ {t} + (1 - c) \nabla f _ {t} \big) + \kappa \boldsymbol {w} _ {t} \big ] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) & 0 & I - \kappa \left[ \boldsymbol {\alpha} _ {t} \right] & 0 & 0 \\ \frac {\mathrm{d}   \boldsymbol {m} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {m} _ {t + 1}}{\mathrm{d}   \boldsymbol {w} _ {t}} & \frac {\mathrm{d}   \boldsymbol {m} _ {t + 1}}{\mathrm{d}   \boldsymbol {m} _ {t}} & 0 \\ \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {w} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {m} _ {t}} & \frac {\mathrm{d}   \boldsymbol {h} _ {t + 1}}{\mathrm{d}   \boldsymbol {h} _ {t}} \end{array} \right] \\ \end{array}
$$

where the second equality is due to (71) and the last equality follows from (70). Consider the following block representation of $X_{t}$ ,

$$
X _ {t} = \left[ \begin{array}{c} \mathcal {H} _ {t} \\ X _ {t} ^ {m} \end{array} \right]. \tag {73}
$$

Plugging the first row of $G_{t}^{\mathrm{base}}$ , given in (72), into (46), implies that

$$
\mathcal {H} _ {t + 1} = - \gamma \left[ \operatorname{Sign} \left(c \boldsymbol {m} _ {t} + (1 - c) \nabla f _ {t}\right) + \kappa \boldsymbol {w} _ {t} \right] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} + \gamma (I - \kappa [ \boldsymbol {\alpha} _ {t} ]) \mathcal {H} _ {t} \tag {74}
$$

For simplicity of notation, we define the diagonal matrix $S_{t}$ as

$$
S _ {t} \stackrel {\text { def }} {=} \left[ \operatorname{Sign} \left(c   \boldsymbol {m} _ {t} + (1 - c) \nabla f _ {t}\right) + \kappa \boldsymbol {w} _ {t} \right]. \tag {75}
$$

Then,

$$
\boldsymbol {h} _ {t + 1} = \gamma \left[ \begin{array}{c} - S _ {t} \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 1 ]} + \gamma (I - \kappa [ \boldsymbol {\alpha} _ {t} ]) \mathcal {H} _ {t} ^ {[ 1 ]} \\ \hline \vdots \\ \hline - S _ {t} \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ m ]} + \gamma (I - \kappa [ \boldsymbol {\alpha} _ {t} ]) \mathcal {H} _ {t} ^ {[ m ]}. \end{array} \right] \tag {76}
$$

It follows that

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {m} _ {t}} = 0, \tag {77}
$$

and

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} = - \gamma \left[ \begin{array}{c c c} \left[ \boldsymbol {e} _ {1} \right] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 1 ]} & \dots & \left[ \boldsymbol {e} _ {n} \right] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ 1 ]} \\ \hline \vdots & \ddots & \vdots \\ \hline \left[ \boldsymbol {e} _ {1} \right] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ m ]} & \dots & \left[ \boldsymbol {e} _ {n} \right] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) B _ {t} ^ {[ m ]} \end{array} \right], \tag {78}
$$

where $e_{i}$ is the ith unit vector (i.e., an n-dimensional vector whose ith entry is 1 and all other entries are zero). Let $\beta_{t}[i]$ and $H_{t}^{[i]}$ be the ith entry of $\beta_{t}$ and ith column of $H_{t}$ , respectively, for $i = 1, \ldots, m$ . Then,

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} = - \gamma \left[ \begin{array}{c c c} \gamma \kappa \left[ \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ 1 ]} \right] \mathcal {H} _ {t} ^ {[ 1 ]} + S _ {t} \frac {\partial \sigma^ {\prime} (\boldsymbol {\beta} _ {t})}{\partial \beta_ {t} [ 1 ]} B _ {t} ^ {[ 1 ]} & \dots & \gamma \kappa \left[ \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ m ]} \right] \mathcal {H} _ {t} ^ {[ 1 ]} + S _ {t} \frac {\partial \sigma^ {\prime} (\boldsymbol {\beta} _ {t})}{\partial \beta_ {t} [ m ]} B _ {t} ^ {[ 1 ]} \\ \hline \vdots & \ddots & \vdots \\ \hline \gamma \kappa \left[ \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ 1 ]} \right] \mathcal {H} _ {t} ^ {[ m ]} + S _ {t} \frac {\partial \sigma^ {\prime} (\boldsymbol {\beta} _ {t})}{\partial \beta_ {t} [ 1 ]} B _ {t} ^ {[ m ]} & \dots & \gamma \kappa \left[ \frac {\mathrm{d} \boldsymbol {\alpha} _ {t}}{\mathrm{d} \beta_ {t} [ m ]} \right] \mathcal {H} _ {t} ^ {[ m ]} + S _ {t} \frac {\partial \sigma^ {\prime} (\boldsymbol {\beta} _ {t})}{\partial \beta_ {t} [ m ]} B _ {t} ^ {[ m ]} \end{array} \right], \tag {79}
$$

and

$$
\frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} = \gamma \left[ \begin{array}{c c c c} I - \kappa [ \boldsymbol {\alpha} _ {t} ] & 0 & 0 & 0 \\ \hline 0 & I - \kappa [ \boldsymbol {\alpha} _ {t} ] & 0 & 0 \\ \hline 0 & 0 & \ddots & 0 \\ \hline 0 & 0 & 0 & I - \kappa [ \boldsymbol {\alpha} _ {t} ] \end{array} \right] \quad \begin{array}{l} \leftarrow 1 \text {st} \\ \leftarrow 2 \text {nd} \\ \vdots \\ \leftarrow m \text {th}, \end{array} . \tag {80}
$$

It follows from (21), (72), and (77) that in the $G_{t}$ matrix, $\frac{d m_{t+1}}{d m_{t}}$ is the only non-zero block in its corresponding column of blocks. Consequently, it follows from (14) that $X_{t}^{m}$ , as defined in (73), has no impact on the update of $H_{t+1}$ , $Y_{t+1}$ , and $Q_{t+1}$ . Therefore, the rows and columns of $G^{base}$ that correspond to derivative of m can be completely removed from $G^{base}$ . By removing these rows and columns from $G^{t}$ , the matrix update (14) simplifies to

$$
\left[ \begin{array}{c} Y _ {t + 1} \\ \mathcal {H} _ {t + 1} \\ Q _ {t + 1} \end{array} \right] = \gamma \left[ \begin{array}{c c c} \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {y} _ {t}} & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {y} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \\ \left[ - \left[ \operatorname{Sign} \left(c \boldsymbol {m} _ {t} + (1 - c) \nabla f _ {t}\right) \right] \sigma^ {\prime} (\boldsymbol {\beta} _ {t}) \quad 0 \right] & I - \kappa [ \boldsymbol {\alpha} _ {t} ] & 0 \\ \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {\beta} _ {t}} & 0 & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {w} _ {t}} & \frac {\mathrm{d} \boldsymbol {h} _ {t + 1}}{\mathrm{d} \boldsymbol {h} _ {t}} \end{array} \right] \left(\left[ \begin{array}{c} Y _ {t} \\ \mathcal {H} _ {t} \\ Q _ {t} \end{array} \right] + (1 - \gamma) \left[ \begin{array}{c} I \\ 0 \\ 0 \\ 0 \end{array} \right]\right), \tag {81}
$$

where $d h_{t+1}/d \beta_{t}$ , $d h_{t+1}/d w_{t}$ , and $d h_{t+1}/d h_{t}$ are given in (79), (78), and (80), respectively; and the blocks in the first row depend on the meta update.

# B. Exiting Step-size Optimization Algorithms as Special Cases of MetaOptimize

In this appendix we show that some of the existing step-size optimization algorithms are special cases of the MetaOptimize framework. In particular, we first consider the IDBD algorithm (Sutton, 1992) and its extension (Xu et al., 2018), and then discuss about the HyperGradient algorithm (Baydin et al., 2017).

# B.1. IDBD and Its Extensions

(Sutton, 1992) proposed the IDBD algorithm for step-size optimization of a class of quadratic loss functions. In particular, it considers loss functions of the form

$$
f _ {t} (\boldsymbol {w} _ {t}) = \frac {1}{2} \left(\boldsymbol {a} _ {t} ^ {T} \boldsymbol {w} _ {t} - b _ {t}\right) ^ {2}, \tag {82}
$$

for a given sequence of feature vectors $a_{t}$ and target values $b_{t}$ , for $t = 1, 2, \ldots$ . Moreover, Sutton (1992) assumes weight-wise step sizes, in which case $\beta_{t}$ has the same dimension as $w_{t}$ . The update rule of IDBD is as follows:

$$
\boldsymbol {g} _ {t} \leftarrow (\boldsymbol {a} _ {t} ^ {T} \boldsymbol {w} _ {t} - b _ {t}) \boldsymbol {a} _ {t}, \tag {83}
$$

$$
\boldsymbol {\beta} _ {t + 1} \leftarrow \boldsymbol {\beta} _ {t} - \eta \boldsymbol {h} _ {t} \boldsymbol {g} _ {t}, \tag {84}
$$

$$
\boldsymbol {\alpha} _ {t + 1} \leftarrow \exp (\boldsymbol {\beta} _ {t + 1}), \tag {85}
$$

$$
\boldsymbol {w} _ {t + 1} \leftarrow \boldsymbol {w} _ {t} - \boldsymbol {\alpha} _ {t + 1} \boldsymbol {g} _ {t}, \tag {86}
$$

$$
\boldsymbol {h} _ {t + 1} \leftarrow \left(1 - \boldsymbol {\alpha} _ {t + 1} \boldsymbol {a} _ {t} ^ {2}\right) ^ {+} \boldsymbol {h} _ {t} - \boldsymbol {\alpha} _ {t + 1} \boldsymbol {g} _ {t}, \tag {87}
$$

where $(\cdot)^{+}$ clips the entries at zero to make them non-negative, aimed to improve stability. Here, $g_{t}$ is the gradient of $f_{t}(\boldsymbol{w}_{t})$ and $a_{t}^{2}$ in the last line is a vector that contains diagonal entries of the Hessian of $f_{t}$ . The updated values of $\beta$ and w would remain unchanged, if instead of the vector $h_{t}$ , we use a diagonal matrix $H_{t}$ and replace (84) and (87) by

$$
\boldsymbol {\beta} _ {t + 1} \leftarrow \boldsymbol {\beta} _ {t} - \eta \mathcal {H} _ {t} \boldsymbol {g} _ {t}, \tag {88}
$$

$$
\mathcal {H} _ {t + 1} \leftarrow \left(1 - \left[ \boldsymbol {\alpha} _ {t + 1} \boldsymbol {a} ^ {2} \right]\right) ^ {+} \mathcal {H} _ {t} - \left[ \boldsymbol {\alpha} _ {t + 1} \boldsymbol {g} _ {t} \right].
$$

Note that $\left[a^{2}\right]$ is a matrix that is obtained from zeroing-out all non-diagonal entries of the Hessian matrix of $f_{t}$ . It is easy to see that the above formulation of IDBD, equals the L-approximation of MetaOptimize framework when we use SGD for both base and meta updates, and further use a diagonal approximation of the Hessian matrix along with a rectifier in the update of $H_{t}$ .

An extension of IDBD beyond quadratic case has been derived in (Xu et al., 2018). Similar to IDBD, they also consider weight-wise step sizes, i.e., m = n. The update of step sizes in this method is as follows:

$$
\boldsymbol {\beta} _ {t + 1} \leftarrow \boldsymbol {\beta} _ {t} - \eta \mathcal {H} _ {t} ^ {\top} \nabla f _ {t} (\boldsymbol {w} _ {t})
$$

$$
\boldsymbol {\alpha} _ {t + 1} \leftarrow \exp (\boldsymbol {\beta} _ {t + 1}),
$$

$$
\pmb {w} _ {t + 1} \leftarrow \pmb {w} _ {t} - \pmb {\alpha} _ {t + 1} \nabla f _ {t} (\pmb {w} _ {t}),
$$

$$
\mathcal {H} _ {t + 1} \leftarrow \left(I - \left[ \boldsymbol {\alpha} _ {t + 1} \right] \nabla^ {2} f _ {t} (\boldsymbol {w} _ {t})\right) \mathcal {H} _ {t} - \left[ \boldsymbol {\alpha} _ {t + 1} \nabla f _ {t} (\boldsymbol {w} _ {t}) \right].
$$

Similar to IDBD, it is straightforward to check that the above set of updates is equivalent to the L-approximation of MetaOptimize framework that uses SGD for both base and meta updates, except for the fact that the above algorithm uses $\alpha_{t+1}$ in $w_{t+1}$ and $H_{t+1}$ updates whereas MetaOptimize uses $\alpha_{t}$ . This however has no considerable impact since $\alpha_{t}$ varies slowly.

# B.2. Hyper-gradient Descent

HyperGradient descent was proposed in (Baydin et al., 2017) as a step-size optimization method. It considers scalar step size with straightforward extensions to weight-wise step sizes, and at each time t, updates the step size in a direction to minimize the immediate next loss function. In particular, they propose the following additive update for step sizes, that can wrap around an arbitrary base update:

$$
\boldsymbol {\alpha} _ {t} = \beta_ {t} \mathbf {1} _ {n \times 1},
$$

$$
\beta_ {t + 1} = \beta_ {t} - \eta \frac {\mathrm{d} f _ {t} (\boldsymbol {w} _ {t})}{\mathrm{d} \beta_ {t - 1}} = \beta_ {t} - \eta \nabla f _ {t} (\boldsymbol {w} _ {t}) ^ {T} \frac {\mathrm{d} \boldsymbol {w} _ {t}}{\mathrm{d} \beta_ {t - 1}}. \tag {89}
$$

The last update can be equivalently written as

$$
\beta_ {t + 1} = \beta_ {t} - \eta   \mathcal {H} _ {t} ^ {T}   \nabla f _ {t} (\boldsymbol {w} _ {t}),
$$

$$
\mathcal {H} _ {t + 1} = 0 \times \mathcal {H} _ {t} + \frac {\mathrm{d} \boldsymbol {w} _ {t + 1}}{\mathrm{d} \beta_ {t}}. \tag {90}
$$

The step-size update in (90) can be perceived as a special case of MetaOptimize in two different ways. First, as a MetaOptimize algorithm that uses SGD as its meta update and approximate the $G_{t}$ matrix in (9) by zeroing out all of its blocks except for the top two blocks in the first column. From another perspective, the additive HyperGradient descent in (90) is also equivalent to a MetaOptimize algorithm that uses SGD as its meta update and sets $\gamma = 0$ . Note that setting $\gamma$ equal to zero would eliminate the dependence of $H_{t+1}$ on $X_{t}$ and $Q_{t}$ , as can be verified from (14). This would also render the $\beta$ updates ignorant about the long-term impact of step size on future losses.

# C. Experiment Details

In the appendix, we describe the details of experiments performed throughout the paper. In our experiments on CIFAR10, non-stationary CIFAR100, and ImageNet dataset, we used a machine with four Intel Xeon Gold 5120 Skylake @ 2.2GHz CPUs and a single NVIDIA V100 Volta (16GB HBM2 memory) GPU. For TinyStories dataset, we used a machine with four AMD Milan 7413 @ 2.65 GHz 128M cache L3 CPUs and a single NVIDIA A100SXM4 (40 GB memory) GPU. In all experiments, the meta step size $\eta$ is set to $10^{-3}$ . The meta-parameters used in the considered optimization algorithm for CIFAR10, non-stationary CIFAR100, ImageNet, and TinyStories are given in Table 2, Table 4, and Table 5, respectively. In the experiments, we performed a grid search for $\rho$ , $\bar{\rho} \in \{0.9, 0.99, 0.999\}$ , $\lambda$ , $\bar{\lambda} \in \{0.99, 0.999\}$ , and $c$ , $\bar{c} \in \{0.9, 0.99\}$ . Regarding baselines with fixed step sizes, we did a grid search for the learning rate in the set $\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ . We set $\gamma$ equal to one in all experiments. Moreover, in ImageNet (respectively TinyStories) dataset, for AdamW with the learning rate scheduler, we considered a cosine decay with 10k (respectively 1k) steps warmup (according to extensive experimental studies in (Chen et al., 2023) (respectively (Karpathy, 2024))) and did a grid search for the maximum learning rate in the set $\{10^{-5}, 10^{-4}, 10^{-3}\}$ .

Regarding other baseline algorithms, for DoG, although it is a parameter-free algorithm, its performance is still sensitive to the initial step movement. We did a grid search for the initial step movement in the set $\{10^{-9}, 10^{-8}, 10^{-7}, 10^{-6}\}$ and reported the performance for the best value. In all experiments of DoG, we considered the polynomial decay averaging. For Prodigy, we used the default values of parameters as suggested by the authors in github repository. For gdtuo, we considered the following (base, meta) combinations: (RMSprop, Adam), (Adam, Adam), and (SGD with momentum, Adam) and chose the best combination. For mechanic, we did experiments for the base updates of SGDm, Lion, and Adam and considered the best update. In order to have a fair comparison, in mechanic and gdtuo, we used the same initial step size as MetaOptimize.

Regarding the complexity overheads reported in Table 1, for AdamW with fixed step-size we used the Pytroch implementation of AdamW. For all other baselines, we used the implementation from the Github repository provided along with (and cited in) the corresponding paper. For MetaOptimize, we used the implementation in (Salehkaleybar, 2025). Note that the implementation of MetaOptimize in (Salehkaleybar, 2025) is not optimized for time or space efficiency, and smaller complexity overheads might be achieved with more efficient codes. For each algorithm, the wall-clock time overhead and GPU space overhead are computed by $(T_{\mathrm{Alg}} - T_{\mathrm{AdamW}}) / T_{\mathrm{AdamW}}$ and $\left(B_{\mathrm{AdamW}}^{\max} / B_{\mathrm{Alg}}^{\max}\right) - 1$ , respectively; where $T_{\mathrm{Alg}}$ and $T_{\mathrm{AdamW}}$ are per-iteration runtimes of the algorithm and AdamW, and $B_{\mathrm{Alg}}^{\max}$ and $B_{\mathrm{AdamW}}^{\max}$ are the maximum batch-sizes that did not cause GPU-memory outage for the algorithm and AdamW.

<table><tr><td>Base Update</td><td>Meta Update (if any)</td><td> $\rho$ </td><td> $\lambda$ </td><td> $\kappa$ </td><td>c</td><td> $\bar{\rho}$ </td><td> $\bar{\lambda}$ </td><td> $\bar{c}$ </td><td> $\alpha_0$ </td><td> $\eta$ </td><td> $\gamma$ </td></tr><tr><td rowspan="3">AdamW</td><td>Fixed step size</td><td>0.9</td><td>0.999</td><td>0.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td> $10^{-5}$ </td><td>-</td><td>1</td></tr><tr><td>Adam, Scalar</td><td>0.9</td><td>0.999</td><td>0.1</td><td>-</td><td>0.9</td><td>0.999</td><td>-</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td>Adam, Blockwise</td><td>0.9</td><td>0.999</td><td>0.1</td><td>-</td><td>0.9</td><td>0.999</td><td>-</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td rowspan="3">Lion</td><td>Fixed step size</td><td>0.99</td><td>-</td><td>0.1</td><td>0.9</td><td>-</td><td>-</td><td>-</td><td> $10^{-4}$ </td><td>-</td><td>1</td></tr><tr><td>Lion, Scalar</td><td>0.99</td><td>-</td><td>0.1</td><td>0.9</td><td>0.99</td><td>-</td><td>0.9</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td>Lion, Blockwise</td><td>0.99</td><td>-</td><td>0.1</td><td>0.9</td><td>0.99</td><td>-</td><td>0.9</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td rowspan="3">RMSprop</td><td>Fixed step size</td><td>-</td><td>0.999</td><td>0.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td> $10^{-5}$ </td><td>-</td><td>1</td></tr><tr><td>Adam, Scalar</td><td>-</td><td>0.999</td><td>0.1</td><td>-</td><td>0.9</td><td>0.999</td><td>-</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td>Adam, Blockwise</td><td>-</td><td>0.999</td><td>0.1</td><td>-</td><td>0.9</td><td>0.999</td><td>-</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td rowspan="3">SGDm</td><td>Fixed step size</td><td>0.9</td><td>-</td><td>0.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td> $10^{-3}$ </td><td>-</td><td>1</td></tr><tr><td>Adam, Scalar</td><td>0.9</td><td>-</td><td>0.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td>Adam, Blockwise</td><td>0.9</td><td>-</td><td>0.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr></table>

Table 2. The values of meta-parameters used in CIFAR10 dataset.

<table><tr><td>Base Update</td><td>Meta Update (if any)</td><td> $\rho$ </td><td> $\lambda$ </td><td> $\kappa$ </td><td> $\bar{\rho}$ </td><td> $\bar{\lambda}$ </td><td> $\alpha_0$ </td><td> $\eta$ </td><td> $\gamma$ </td></tr><tr><td rowspan="3">AdamW</td><td>Fixed step size</td><td>0.9</td><td>0.999</td><td>0.1</td><td>-</td><td>-</td><td> $10^{-4}$ </td><td>-</td><td>-</td></tr><tr><td>Adam, Scalar</td><td>0.9</td><td>0.999</td><td>0.1</td><td>0.9</td><td>0.999</td><td> $10^{-4}$ </td><td> $10^{-3}$ </td><td>0.999</td></tr><tr><td>Adam, Blockwise</td><td>0.9</td><td>0.999</td><td>0.1</td><td>0.9</td><td>0.999</td><td> $10^{-4}$ </td><td> $10^{-3}$ </td><td>0.999</td></tr></table>

Table 3. The values of meta-parameters used in non-stationary CIFAR100 experiment. 

<table><tr><td>Base Update</td><td>Meta Update</td><td> $\rho$ </td><td> $\lambda$ </td><td> $\kappa$ </td><td> $c$ </td><td> $\bar{\rho}$ </td><td> $\bar{\lambda}$ </td><td> $\bar{c}$ </td><td> $\alpha_0$ </td><td> $\eta$ </td><td> $\gamma$ </td></tr><tr><td rowspan="3">AdamW</td><td>Fixed step size</td><td>0.9</td><td>0.999</td><td>0.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td> $10^{-5}$ </td><td>-</td><td>1</td></tr><tr><td>Lion, Scalar</td><td>0.9</td><td>0.999</td><td>0.1</td><td>-</td><td>0.99</td><td>-</td><td>0.9</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td>Lion, Blockwise</td><td>0.9</td><td>0.999</td><td>0.1</td><td>-</td><td>0.99</td><td>-</td><td>0.9</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td rowspan="3">Lion</td><td>Fixed step size</td><td>0.99</td><td>-</td><td>0.1</td><td>0.9</td><td>-</td><td>-</td><td>-</td><td> $10^{-5}$ </td><td>-</td><td>1</td></tr><tr><td>Lion, Scalar</td><td>0.99</td><td>-</td><td>0.1</td><td>0.9</td><td>0.99</td><td>-</td><td>0.9</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td>Lion, Blockwise</td><td>0.99</td><td>-</td><td>0.1</td><td>0.9</td><td>0.99</td><td>-</td><td>0.9</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td>SGDm</td><td>Lion, Scalar</td><td>0.9</td><td>-</td><td>0.1</td><td>0.9</td><td>-</td><td>-</td><td>-</td><td> $10^{-5}$ </td><td> $10^{-3}$ </td><td>1</td></tr></table>

Table 4. The values of meta-parameters used in ImageNet dataset.

# D. Further Experimental Results

Continual CIFAR100 experiment: In Figure 3, we plotted the average top-1 accuracy—averaged over all past training times. This metric is used in continual learning mainly because it summarizes algorithmic performance across multiple tasks, avoiding difficult/misleading interpretations from task-specific accuracy variations. Here, we also include the accuracy curves to reveal such variations. Figure 7 depicts test accuracy at the end of each task.

Moreover, Fig.8 presents block-wise step sizes similar to those in Fig.4, but displayed in separate subfigures for each block to enhance the visibility of step-size ranges. Note that step size of both blocks start at $10^{-4}$ .

![](images/6b05bd71e373c76114b3665041f954f19dbc1fb61041f50b8c98dc4b63949873.jpg)

<details>
<summary>line</summary>

| Task number | AdamW, Fixed stepsize | MetaOptimize (AdamW,Adam), Scalar | MetaOptimize (AdamW,Adam), Blockwise |
| ----------- | --------------------- | ---------------------------------- | ------------------------------------ |
| 1           | 45.5                  | 48.7                               | 48.5                                 |
| 2           | 48.8                  | 47.0                               | 46.8                                 |
| 3           | 47.8                  | 47.5                               | 46.5                                 |
| 4           | 43.5                  | 44.8                               | 45.5                                 |
| 5           | 50.8                  | 51.5                               | 52.0                                 |
| 6           | 50.2                  | 52.3                               | 52.2                                 |
| 7           | 47.9                  | 51.3                               | 50.8                                 |
| 8           | 49.9                  | 52.0                               | 51.5                                 |
| 9           | 46.0                  | 47.9                               | 48.0                                 |
| 10          | 45.2                  | 48.3                               | 46.8                                 |
</details>

Figure 7. Top 1 test accuracy of each task in the non-stationary CIFAR100 experiment, computed at the end of the task.

ImageNet dataset: In Figure 9, we depict the train accuracy (top 1) and test accuracy (top 1) of the considered algorithms in ImageNet dataset. As can be seen, in the train accuracy (top 1), MetaOptimize (SGDm, Lion) and MetaOptimize (AdamW, Lion) have the best performance. Moreover, in the test accuracy (top1), these two combinations of MetaOptimize outperform other hyperparameter optimization methods and only AdamW with a handcrafted learning rate scheduler has a slightly better performance at the end of the training process.

Figure 10 shows the results for the blockwise version of MetaOptimize for two combinations of (AdamW, Lion) and (Lion,

<table><tr><td>Base Update</td><td>Meta Update (if any)</td><td> $\rho$ </td><td> $\lambda$ </td><td> $\kappa$ </td><td> $c$ </td><td> $\bar{\rho}$ </td><td> $\bar{\lambda}$ </td><td> $\bar{c}$ </td><td> $\alpha_0$ </td><td> $\eta$ </td><td> $\gamma$ </td></tr><tr><td rowspan="2">AdamW</td><td>Fixed stepsize</td><td>0.9</td><td>0.999</td><td>0.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td> $10^{-5}$ </td><td>-</td><td>1</td></tr><tr><td>Adam, Scalar</td><td>0.9</td><td>0.999</td><td>0.1</td><td>-</td><td>0.9</td><td>0.999</td><td>-</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr><tr><td rowspan="2">Lion</td><td>Fixed stepsize</td><td>0.99</td><td>-</td><td>0.1</td><td>0.9</td><td>-</td><td>-</td><td>-</td><td> $10^{-4}$ </td><td>-</td><td>1</td></tr><tr><td>Lion, Scalar</td><td>0.99</td><td>-</td><td>0.1</td><td>0.9</td><td>0.99</td><td>-</td><td>0.9</td><td> $10^{-6}$ </td><td> $10^{-3}$ </td><td>1</td></tr></table>

Table 5. The values of meta-parameters used in TinyStories dataset.

![](images/75a292daea4966f9700eb6f8e6d0666a65661ae798283a8014cf8e02c228d083.jpg)

<details>
<summary>line</summary>

| Iteration | Step size of 1st Block |
| --------- | ---------------------- |
| 0         | 0.00010                |
| 5000      | 0.00006                |
| 10000     | 0.00006                |
| 15000     | 0.00005                |
| 20000     | 0.00004                |
| 25000     | 0.00005                |
| 30000     | 0.00004                |
| 35000     | 0.00003                |
| 40000     | 0.00003                |
| 45000     | 0.00002                |
| 50000     | 0.00001                |
</details>

![](images/5dab0047d725cb32c9f11be5f61833f952680a75e6e47bd422f188e41874a8a2.jpg)

<details>
<summary>line</summary>

| Iteration | Step size of 2nd Block |
| --------- | ---------------------- |
| 0         | 0.0001                 |
| 5000      | 0.0001                 |
| 10000     | 0.0003                 |
| 15000     | 0.0002                 |
| 20000     | 0.0006                 |
| 25000     | 0.0010                 |
| 30000     | 0.0011                 |
| 35000     | 0.0011                 |
| 40000     | 0.0020                 |
| 45000     | 0.0019                 |
| 50000     | 0.0022                 |
</details>

Figure 8. Blockwise stepsizes learned by MetaOptimize (AdamW, Adam) on non-stationary CIFAR100. Note that the scale of the y-axis for the two curves differ by an order of magnitude. Step-sizes of both blocks are initialized at $\alpha_{0} = 10^{-4}$ .

Lion). As can be seen, they showed no improvement over the scalar version.

TinyStories experiment: In Figure 11, we provide the test loss of considered algorithms for the TinyStories datasets. As can be seen, the learning curves have the same trends as the training loss in Figure 6.

![](images/f68d69480fc2338452e026d36798f3200e1672219eafb6530f65fb566242ea87.jpg)

<details>
<summary>line</summary>

| Epoch | AdamW, LR scheduler | AdamW, Fixed stepsize | Lion, Fixed stepsize | MetaOptimize (AdamW, Lion), Scalar | MetaOptimize (Lion, Lion), Scalar | MetaOptimize (SGDm, Lion) | gdtuo | mechanic | Prodigy |
|-------|----------------------|------------------------|----------------------|-------------------------------------|------------------------------------|----------------------------|-------|----------|---------|
| 0     | 0                    | 0                      | 0                    | 0                                   | 0                                  | 0                          | 0     | 0        | 0       |
| 20    | 55                   | 58                     | 52                   | 60                                  | 57                                 | 62                         | 59    | 54       | 53      |
| 40    | 62                   | 65                     | 58                   | 65                                  | 62                                 | 68                         | 64    | 60       | 59      |
| 60    | 66                   | 68                     | 60                   | 68                                  | 65                                 | 70                         | 67    | 63       | 62      |
| 80    | 68                   | 70                     | 62                   | 70                                  | 67                                 | 72                         | 69    | 65       | 64      |
| 100   | 70                   | 72                     | 64                   | 72                                  | 69                                 | 74                         | 71    | 67       | 66      |
</details>

(a) Train Accuracy (Top 1)

![](images/7970bbe1bf28702850d6a69975b293203eecf8e221661ef4a2d6d8d3d1065295.jpg)

<details>
<summary>line</summary>

| Epoch | AdamW, LR scheduler | AdamW, Fixed stepsize | Lion, Fixed stepsize | MetaOptimize (AdamW, Lion), Scalar | MetaOptimize (Lion, Lion), Scalar | MetaOptimize (SGDm, Lion) | gdtuo | mechanic | Prodigy |
|-------|----------------------|------------------------|----------------------|-------------------------------------|------------------------------------|----------------------------|-------|----------|---------|
| 0     | ~15                  | ~15                    | ~15                  | ~15                                 | ~15                                | ~15                        | ~15   | ~15      | ~15     |
| 20    | ~60                  | ~60                    | ~55                  | ~60                                 | ~55                                | ~60                        | ~55   | ~45      | ~55     |
| 40    | ~65                  | ~65                    | ~60                  | ~65                                 | ~60                                | ~65                        | ~60   | ~60      | ~65     |
| 60    | ~67                  | ~67                    | ~63                  | ~67                                 | ~63                                | ~67                        | ~63   | ~63      | ~67     |
| 80    | ~68                  | ~68                    | ~64                  | ~68                                 | ~64                                | ~68                        | ~64   | ~64      | ~68     |
| 90    | ~68                  | ~68                    | ~64                  | ~68                                 | ~64                                | ~68                        | ~64   | ~64      | ~68     |
</details>

(b) Test Accuracy (Top 1)   
Figure 9. ImageNet learning curves.

![](images/ada6e195d9171ebe0858620b8107ecf247a7303d0f9410bed1563eaa82281c7d.jpg)

<details>
<summary>line</summary>

| Epoch | MetaOptimize (AdamW, Lion), Scalar | MetaOptimize (Lion, Lion), Scalar | MetaOptimize (AdamW, Lion), Blockwise | MetaOptimize (Lion, Lion), Blockwise |
|-------|-------------------------------------|------------------------------------|----------------------------------------|--------------------------------------|
| 0     | 65.0                                | 65.0                               | 65.0                                   | 65.0                                 |
| 20    | 82.0                                | 80.0                               | 78.0                                   | 79.0                                 |
| 40    | 85.0                                | 83.0                               | 80.0                                   | 82.0                                 |
| 60    | 86.0                                | 84.0                               | 81.0                                   | 83.0                                 |
| 80    | 87.0                                | 85.0                               | 81.5                                   | 84.0                                 |
| 90    | 87.5                                | 85.5                               | 81.5                                   | 84.5                                 |
</details>

Figure 10. Comparison of blockwise version of MetaOptimize with the scalar version in ImageNet dataset.

![](images/db999afd883e521828ca5039524e402fb14842b405005f9a5e1903f7a34c54df.jpg)

<details>
<summary>line</summary>

| Iteration | AdamW, LR scheduler | AdamW, Fixed stepsize | Lion, Fixed stepsize | MetaOptimize (AdamW, Lion) | MetaOptimize (Lion, Lion) | DoG | gdtuo | mechanic | Prodigy |
| --------- | ------------------- | --------------------- | -------------------- | -------------------------- | ------------------------- | --- | ----- | -------- | ------- |
| 0         | 2.2                 | 2.2                   | 2.2                  | 2.2                        | 2.2                       | 2.2 | 2.2   | 2.2      | 2.2     |
| 5000      | 1.3                 | 1.4                   | 1.5                  | 1.4                        | 1.5                       | 1.8 | 1.6   | 1.6      | 1.9     |
| 10000     | 1.25                | 1.35                  | 1.4                  | 1.35                       | 1.4                       | 1.7 | 1.5   | 1.5      | 1.7     |
| 15000     | 1.2                 | 1.3                   | 1.35                 | 1.3                        | 1.35                      | 1.6 | 1.4   | 1.4      | 1.6     |
| 20000     | 1.2                 | 1.3                   | 1.3                  | 1.25                       | 1.3                       | 1.55| 1.35  | 1.35     | 1.5     |
| 25000     | 1.2                 | 1.25                  | 1.25                 | 1.25                       | 1.25                      | 1.5 | 1.3   | 1.3      | 1.4     |
| 30000     | 1.2                 | 1.25                  | 1.25                 | 1.25                       | 1.25                      | 1.45| 1.25  | 1.25     | 1.35    |
</details>

Figure 11. TinyStories learning curves.