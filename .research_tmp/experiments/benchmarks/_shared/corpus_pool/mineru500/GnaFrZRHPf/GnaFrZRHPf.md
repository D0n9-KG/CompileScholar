# Adaptive Preference Scaling for Reinforcement Learning with Human Feedback

Ilgee Hong $^{*1}$ Zichong Li $^{*1}$ Alexander Bukharin $^{1}$ Yixiao Li $^{1}$ Haoming Jiang $^{2}$ Tianbao Yang $^{3}$ Tuo Zhao $^{1}$

# Abstract

Reinforcement learning from human feedback (RLHF) is a prevalent approach to align AI systems with human values by learning rewards from human preference data. Due to various reasons, however, such data typically takes the form of rankings over pairs of trajectory segments, which fails to capture the varying strengths of preferences across different pairs. In this paper, we propose a novel adaptive preference loss, underpinned by distributionally robust optimization (DRO), designed to address this uncertainty in preference strength. By incorporating an adaptive scaling parameter into the loss for each pair, our method increases the flexibility of the reward function. Specifically, it assigns small scaling parameters to pairs with ambiguous preferences, leading to more comparable rewards, and large scaling parameters to those with clear preferences for more distinct rewards. Computationally, our proposed loss function is strictly convex and univariate with respect to each scaling parameter, enabling its efficient optimization through a simple second-order algorithm. Our method is versatile and can be readily adapted to various preference optimization frameworks, including direct preference optimization (DPO). Our experiments with robotic control and natural language generation with large language models (LLMs) show that our method not only improves policy performance but also aligns reward function selection more closely with policy optimization, simplifying the hyperparameter tuning process.

# 1. Introduction

In the field of artificial intelligence, aligning AI systems with human preferences has become increasingly crucial, particularly for applications involving complex data and models like large language models (LLMs) in natural language processing (Stiennon et al., 2020; Ouyang et al., 2022). Reinforcement learning from human feedback (RLHF) has gained popularity for customizing AI systems (Christiano et al., 2017; Bai et al., 2022; Zhao et al., 2023). RLHF involves learning a reward function from human preference data, then using a reinforcement learning algorithm to train a policy to optimize the learned reward model.

A key challenge in RLHF lies in the complexity of reward modeling, which primarily stems from the reliance on preference labels. Since preference labels only provide comparative rankings of trajectory segments without quantifying the scale of underlying preference strengths, previous methods have employed the Bradley-Terry (BT) model (Bradley & Terry, 1952) in conjunction with cross-entropy loss to learn the reward function from preference data (Christiano et al., 2017; Stiennon et al., 2020). This approach assumes that the logit of the preference distribution scales linearly with the reward difference across all sample pairs. However, such linear scaling is often insufficient to account for the variations in preference strength among different pairs, restricting the reward function's ability to capture a broader range of reward differences. This restrictive approach to reward modeling limits the flexibility of the learned reward function, hindering its capacity to produce the versatile rewards essential for the downstream policy optimization.

To overcome this shortcoming, we introduce a novel adaptive preference loss function inspired by distributionally robust optimization (DRO, Duchi et al. (2021)). Our approach incorporates an instance-specific scaling factor to change the scaling between the preference distribution and the reward difference to be non-linear. These factors are learned during training and enable the model to accommodate varying uncertainties of preference strength, thereby enhancing the flexibility of the reward. For pairs showing strong preference (i.e., low preference uncertainty), our method learns a large scaling factor,

which enables the model to learn a larger reward difference. In contrast, for pairs showing ambiguous preferences (i.e., high preference uncertainty), our method assigns a smaller scaling factor, enabling the model to learn a smaller reward difference. The additional computational overhead of involving this scaling factor into training is negligible, as the proposed loss function is strictly convex and univariate with respect to each scaling parameter. Therefore, it can be easily optimized by a simple second-order algorithm within a few iterations.

Our experiments on robotic control tasks (Todorov et al., 2012) demonstrate that our method can learn a more flexible reward function, resulting in an improved policy. Surprisingly, we also discover that our method better aligns the learned reward function with downstream policy optimization. Specifically, when tuning hyperparameters for reward modeling, the simplest approach is to select the reward model according to preference prediction accuracy. However, the selected reward function (with the highest accuracy) often yields a downstream policy with poor performance. To address this misalignment, we usually have to jointly tune the parameters across both stages according to downstream policy performance, resulting in significant computational burden and tuning effort. Our proposed method can mitigate this misalignment: When using our adaptive loss, we can select the reward model based on preference prediction accuracy alone and yield a reasonably well-performing policy. This allows separate tuning of the two stages, easing tuning overhead. To our knowledge, the challenge of this misalignment issue is almost untouched in the RLHF literature, and we are the first to propose a principal approach to mitigate this issue.

Moreover, our method is generalizable and can be applied to other preference optimization algorithms. For instance, we implement it with direct preference optimization (DPO, Rafailov et al. (2024)) and evaluate its effectiveness on natural language generation tasks using Llama-2 7B (Touvron et al., 2023). Our results demonstrate that integrating adaptive preference scaling into DPO boosts policy performance, while preserving the benefits of alignment. Alignment is especially critical in this setting, where we employ proprietary models like Claude 2 (Anthropic, 2023) as judges for policy selection, which demands substantial costs for using the APIs. In the case without access to LLM assessment, we must select policy based solely on preference accuracy, under which our approach substantially outperforms other baselines.

We summarize our main contributions as follows: I. We propose an adaptive preference loss function for RLHF, inspired by distributionally robust optimization and further extend it to DPO. By incorporating instance-specific scaling factors, our method enhances the flexibility of the reward model; II. Our method not only enhances the policy performance, but also better aligns the learned reward function with policy optimization, significantly easing the tuning efforts; III. We evaluate the effectiveness of our method on robotic control and natural language generation tasks. Our experimental results demonstrate that our adaptive preference loss can substantially improve the learned policy on both tasks and achieve better alignment between the reward model and policy optimization.

# 2. Related Works

Loss Functions for Reward Learning. Prior work on this topic is very limited. For example, Song et al. (2023) propose using different loss functions for strong and ambiguous preference data in natural language generation tasks. They apply heavy-tailed loss functions for open-ended questions, where preference ambiguity is desirable, and light-tailed loss functions for close-ended questions requiring clear-cut rewards. However, their approach requires knowing the question type a priori, necessitating extra labeling effort, and may fail for complex questions containing both open and closed aspects. Zhao et al. (2023) propose using a hinge loss, which results in zero gradient when the learned reward difference exceeds a margin of 1. This limits the ability to learn very large differences in rewards. Azar et al. (2024) develop $\Psi$ Preference Optimization with Identity Mapping (IPO), which modifies DPO with a loss function matching the scaling of KL-divergence between the learned policy and the initial policy to avoid overfitting due to weak regularization. In contrast to prior work, our method is more broadly applicable to complex preference learning tasks without needing additional labeling or sacrificing the ability to learn arbitrarily large reward differences.

Adaptive Temperature Scaling (ATS). Temperature scaling (TS) aims to adjust the entropy of probabilistic models by rescaling their logit outputs before the softmax function is applied. This simple method not only enables confidence calibration (Guo et al., 2017), but also plays a vital role in various machine learning methods, including knowledge distillation (Hinton et al., 2015), reinforcement learning (Ma et al., 2017), and contrastive learning (Wang & Liu, 2021). Building on TS, adaptive temperature scaling (ATS) enhances flexibility by using instance-specific scalars. Most ATS method trains an additional network for predicting the temperature parameter, which is further integrated into the softmax operator to calibrate the prediction probabilities (Wang et al., 2020; Ding et al., 2021; Balanya et al., 2024; Joy et al., 2023).

In contrast to the aforementioned ATS methods, the proposed adaptive preference scaling (APS) is not designed for classical confidence calibration, but is crafted specifically to enhance the training process of reward function in RLHF. Consequently, the interpretations of scaling factors in ATS and APS are opposite. In ATS, a larger scaling parameter is applied to data with higher uncertainty (e.g., data that the classifier is likely to misclassify), which reduces the magnitude of the corresponding logit. Conversely, in APS, a larger scaling factor is assigned to data with clearer preferences, resulting in a larger logit. This distinction clarifies why the scaling parameter in our approach does not correspond to the concept of “temperature” from statistical physics. Additionally, we propose a principled framework for learning scaling parameter based on DRO, which avoids the complexities of designing specific temperature networks and does not rely on heuristically designed loss functions.

Distributionally Robust Optimization (DRO). DRO is a technique that trains machine learning models to be robust against uncertainty in the data distribution. Specifically, DRO finds a solution that performs well under the worst-case distribution within a specified uncertainty set around the empirical data distribution (Ben-Tal et al., 2013; Bertsimas et al., 2018; Kuhn et al., 2019; Sagawa et al., 2019; Duchi et al., 2021). DRO has been applied in various AI/ML domains to improve generalization when the test distribution differs from the training distribution (Oren et al., 2019; Gokhale et al., 2022; Michel et al., 2021; Broscheit et al., 2022; Wen et al., 2022; Qi et al., 2023). Our framework is motivated by Qi et al. (2023), which tackles KL-constrained DRO problem. However, our approach differs in two significant ways. First, instead of using a single KL constraint for the entire training dataset, we apply a separate KL constraint to each individual training data. Second, since each training data involves just two distributional variables, we can use a deterministic method to optimize these efficiently. Note that while our proposed method is inspired by DRO, it serves a distinct purpose: improving reward learning in RLHF, which is orthogonal to distributional robustness.

# 3. Method

In this section, we first outline the problem setup, derive the loss function with adaptive preference scaling, and then provide theoretical motivation for our proposed loss. At last, we present an optimization algorithm, extend the approach to direct preference optimization, and introduce a variant of our proposed loss that incorporates quadratic regularization.

# 3.1. Problem Setup

We consider a reward-free Markov decision process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, p, \gamma)$ with state $s \in S$ , action $a \in A$ , state transition function p, and discount factor $\gamma$ . The ground truth reward function $r : S \times A \to R$ is assumed to be unknown, but only human preferences over pairs of trajectory segments are observed. A trajectory segment is a sequence of consecutive state and action pairs $z = \{(s_m, a_m), (s_{m+1}, a_{m+1}), \ldots, (s_{k-1}, a_{k-1})\} \in (\mathcal{S} \times \mathcal{A})^{k-m}$ . We denote $z_1 \succ z_2$ to indicate that the human preferred trajectory segment $z_1$ over the trajectory segment $z_2$ and denote the preferred one with a subscript w and the dispreferred one with a subscript l (i.e., $z_w$ and $z_l$ ). Here, we are given a human preference dataset of trajectory segments $\mathcal{D}_{\mathrm{pref}} = (z_w, i, z_{l,i})_{i=1}^N$ . Our goal is to find a reward $\hat{r}(s, a)$ , which is well-aligned with human preferences. Once we learn the reward, we then find a policy $\pi \in \Delta_A^S$ such that it maximizes the expected sum of discounted rewards,

$$
\max _ {\pi} \mathbb {E} _ {z \sim \mathcal {D} _ {\pi}} \left[ \hat {r} (z) \right],
$$

where $\hat{r}(z) = \sum_{(s_t, a_t) \in z} \gamma^t \hat{r}(s_t, a_t)$ and $\mathcal{D}_{\pi}$ denotes the stationary distribution of the state-action pair induced by $\pi$ .

# 3.2. Reward Learning with Adaptive Preference Scaling

We now focus on the reward learning phase in RLHF, a crucial stage for capturing human preferences across various trajectory segments. The standard reward learning procedure assumes that the reward function determines a preference distribution, also known as the Bradley-Terry (BT) model (Bradley & Terry, 1952),

$$
p _ {r} (z _ {w} \succ z _ {l}) = \sigma (r (z _ {w}) - r (z _ {l})), \tag {1}
$$

where $\sigma$ denotes the sigmoid function. The reward function is then learned by minimizing the expectation of negative log-likelihood of r over the preference data (Christiano et al., 2017):

$$
\min _ {r} \mathcal {L} _ {\text {pref}} (r) = - \mathbb {E} _ {(z _ {w}, z _ {l}) \sim \mathcal {D} _ {\text {pref}}} \big [ \log p _ {r} (z _ {w} \succ z _ {l}) \big ]. \tag {2}
$$

As can be seen from (1), the BT model essentially assumes that the logit of the preference distribution $\sigma^{-1}(p_{r}(z_{w} \succ z_{l}))$ scales linearly with the reward difference, regardless of the specific pair of samples. Such linear scaling, however, does not necessarily fit the downstream policy learning well. For some applications, it may lead to a reward function that is not flexible enough to differentiate a pair of segments, which are supposed to have significantly different rewards.

To address this challenge, we propose an adaptive preference loss based on KL-constrained distributionally robust optimization formulation (Qi et al., 2023), which can implicitly change the scaling between the logit and the reward difference to be non-linear. Specifically, given a pair of trajectory segments $(z_{1}, z_{2})$ , we denote $d_{r}(z_{1}, z_{2}) = \mathbf{1}(z_{1} \succ z_{2}) \cdot (r(z_{2}) - r(z_{1}))$ and $p = (p_{1}, p_{2})$ . We define the following instance-level loss:

$$
\ell_ {r} \left(z _ {1}, z _ {2}\right) := \max _ {p \in \Delta_ {2}} p _ {1} d _ {r} \left(z _ {1}, z _ {2}\right) + p _ {2} d _ {r} \left(z _ {2}, z _ {1}\right) - \tau_ {0} \mathrm{KL} (p, 1 / 2) \quad \text { s   .   t   . } \quad \mathrm{KL} (p, 1 / 2) \leq \rho_ {0}, \tag {3}
$$

where $\Delta_{2}=\{p\in\mathbb{R}^{2}:p_{1}+p_{2}=1,0\leq p_{1},p_{2}\leq1\}$ , 1/2 is denoted for the uniform distribution, and $\rho_{0},\tau_{0}>0$ are shared prespecified parameters across all instances. KL( $\cdot,\cdot$ ) denotes the KL divergence. Note that without the KL-constraint, (3) is reduced to the cross-entropy loss with $\tau_{0}=1$ . Unlike general KL-constrained DRO formulation, which considers a distribution p over all training samples, the distributional variable p in (3) is specifically associated with binary preference comparisons for each training sample (i.e., each pair of trajectory segments).

We then convert (3) into an equivalent minimax formulation based on the Lagrangian duality,

$$
\min _ {\lambda \geq 0} \max _ {p \in \Delta_ {2}} p _ {1} d _ {r} (z _ {1}, z _ {2}) + p _ {2} d _ {r} (z _ {2}, z _ {1}) - (\lambda + \tau_ {0}) (\mathrm{KL} (p, 1 / 2) - \rho_ {0}),
$$

where $\lambda$ is the Lagrange multiplier. By defining $\tau$ as $\tau = \lambda + \tau_{0}$ and applying the optimality condition for p, we have

$$
\min _ {\tau \geq \tau_ {0}} - \tau \log p _ {r, \tau} (z _ {w} \succ z _ {l}) + (\rho_ {0} - \log 2) \tau , \tag {4}
$$

where

$$
p _ {r, \tau} (z _ {w} \succ z _ {l}) = \sigma \left(\frac {r (z _ {w}) - r (z _ {l})}{\tau}\right). \tag {5}
$$

We refer to Appendix A.1 for the full derivation. Note that the preference scaling factor $\tau$ in (4) and (5) serves as the Lagrange multiplier of (3). This scaling parameter $\tau$ is used specifically for training the reward function r, rather than calibrating the preference distribution $p_{r,\tau}(z_{w} \succ z_{l})$ . The scaler $\tau$ is used exclusively during the reward learning phase and is no longer needed in subsequent policy optimization, where the reward function r alone is used.

Moreover, the scaling parameter $\tau$ is defined to be an instance-specific parameter corresponding to the pair of trajectory segments $(z_{w}, z_{l})$ . Therefore, when applying our adaptive loss to reward learning, for each pair $(z_{w,i}, z_{l,i})$ , we need to define a corresponding scaling parameter denoted by $\tau_{i}$ . The overall loss function over the training set $D_{pref}$ is as follows:

$$
\min _ {r, \tau_ {1}, \dots , \tau_ {N} \in \Omega} \frac {1}{N} \sum_ {i = 1} ^ {N} \ell_ {i} (r, \tau_ {i}) := \frac {1}{N} \sum_ {i = 1} ^ {N} \left(- \tau_ {i} \log p _ {r, \tau_ {i}} \left(z _ {w, i} \succ z _ {l, i}\right) + \rho \tau_ {i}\right), \tag {6}
$$

where $T = (\tau_{1}, \ldots, \tau_{N})$ , $\Omega = \{\tau : \tau_{0} \leq \tau \leq \tau_{\max}\}$ with $\tau_{max}$ as another prespecified parameter, and $\rho = \rho_{0} - \log 2 > -\log 2$ . Here, we also involve an upper bound $\tau_{max} > 0$ in (6), and we will explain why it is needed in the next subsection.

# 3.3. Theoretical Insights

We next provide some theoretical insights on why the scaling parameter $\tau$ can help gain adaptivity by a proposition. For simplicity, we only consider a pair of trajectory segments.

Proposition 3.1. Assume we have a pair of trajectory segments $z_1, z_2$ , and the preference distribution $p(z_1 \succ z_2) = p^* \in (0,1)$ , i.e., the probability, that $z_1$ is preferred over $z_2$ , is $p^*$ . Consider the problem of minimizing the expectation of our adaptive loss function over the preference distribution:

$$
\min _ {r, \tau \in \Omega} - \tau p ^ {*} \log \left(\sigma \big ((r (z _ {1}) - r (z _ {2})) / \tau \big)\right) - \tau (1 - p ^ {*}) \log \left(\sigma \big ((r (z _ {2}) - r (z _ {1})) / \tau \big)\right) + \rho \tau . \tag {7}
$$

Then the minimizer $\tau^{*}$ and $r^{*}$ of the expected loss satisfy

$$
\tau^ {*} = \left\{ \begin{array}{l l} \tau_ {0} & i f - p ^ {*} \log (p ^ {*}) - (1 - p ^ {*}) \log (1 - p ^ {*}) + \rho > 0, \\ \tau_ {\max} & i f - p ^ {*} \log (p ^ {*}) - (1 - p ^ {*}) \log (1 - p ^ {*}) + \rho <   0, \end{array} \right.
$$

$$
r ^ {*} (z _ {1}) - r ^ {*} (z _ {2}) = \tau^ {*} \sigma^ {- 1} (p ^ {*}).
$$

Here, $\sigma^{-1}$ is the inverse of sigmoid function.

Note that the expected loss (7) is only for easing theoretical analysis, as $p^{*}$ is not accessible in practice. From Proposition 3.1, we can see that when $p^{*}$ is close enough to 0.5, (i.e., the uncertainty of preference is large), the corresponding optimal $\tau^{*}$ is at the lower bound $\tau_{0}$ . The resulting optimal reward difference is $\tau_{0}\sigma^{-1}(p^{*})$ , which is smaller than the counterpart obtained by the cross-entropy loss when $\tau_{0}<1$ . Conversely, when $p^{*}$ is close to 0 or 1, (i.e., the uncertainty of preference is small), the resulting optimal $\tau^{*}$ is at the upper bound $\tau_{max}$ . Here, we introduce the upper bound $\tau_{max}$ to ensure that the optimal $\tau^{*}$ is bounded. The resulting reward difference in this case is $\tau_{\mathrm{max}}\sigma^{-1}(p^{*})$ , which is larger than the counterpart obtained by the cross-entropy loss when $\tau_{max}>1$ . Our theoretical analysis suggests that the adaptive scaling factor essentially changes the correspondence between the logit of preference distribution and the reward difference for each pair of trajectory segments, which could lead to a more flexible reward model.

![](images/0b9e24964a2e3bb55ae3d131cb4de117e83c68c028b62bf57ac18227b1bd9f5b.jpg)

![](images/06e9324953b28a8bce34b65d7944f309a26e9032b38da371a795af74c30059c8.jpg)  
Figure 1. Visualization of the loss function (left) and its gradient (right) on different reward differences.

We further visualize our adaptive preference loss in Figure 1, setting $\tau_{0}$ and $\tau_{max}$ to 0.1 and 5.0, respectively. As depicted, our adaptive preference loss behaves distinctly compared to the cross-entropy loss. With large learned reward differences, the cross-entropy tends to be very flat, while our loss maintains a non-trivial gradient, allowing us to continually decrease the loss function. In contrast, for small positive learned reward differences, our loss yields a smaller gradient, thereby less encouraging the reward model to further distinguish pairs of ambiguous trajectory segments. This is consistent with our theoretical analysis.

# 3.4. Algorithm

We present an efficient algorithm for solving (6). Suppose we parameterize $r$ as a neural network with parameter $\phi$ . At the $m$ -th iteration, we have the iterate $\phi^{(m)}$ , and we sample a pair of trajectory segments $z_{w,i}$ and $z_{l,i}$ . We initialize $\tau_i^{(0)} = 1$ and then optimize $\tau_i$ by a projected Newton method subject to a simple interval constraint $\Omega$ (Bertsekas, 1982). Specifically, for $k = 0, ..., K - 1$ , we take

$$
\tau_ {i} ^ {(k + 1)} = \prod_ {\tau_ {i} \in \Omega} (\tau_ {i} ^ {(k)} + \Delta_ {i} ^ {(k)}), \tag {8}
$$

where $\Delta_i^{(k)}$ denotes the descent direction

$$
\Delta_ {i} ^ {(k)} = - \frac {\nabla_ {\tau_ {i}} \ell_ {i} (\phi^ {(m)} , \tau_ {i} ^ {(k)})}{\nabla_ {\tau_ {i}} ^ {2} \ell_ {i} (\phi^ {(m)} , \tau_ {i} ^ {(k)})}. \tag {9}
$$

Once we get $\tau_{i}^{(K)}$ , we update $\phi$ by a stochastic gradient descent step

$$
\phi^ {(m + 1)} = \phi^ {(m)} - \eta_ {\phi} \nabla_ {\phi} \ell_ {i} (\phi^ {(m)}, \tau_ {i} ^ {(K)}), \tag {10}
$$

Algorithm 1 Algorithm for reward learning with adaptive preference scaling   
1: Input: $\tau_{0}$ , $\tau_{max}$ , $\rho$ , $\eta_{\phi}$ ;
2: for $m = 0, 1, 2, \ldots, M - 1$ do
3: Sample a pair of trajectory segments from $D_{pref}$ ;
4: Set $\tau_{i}^{0} = 1$ ;
5: for $k = 0, 1, 2, \ldots, K - 1$ do
6: Compute $\Delta_{i}^{(k)}$ using (9) and update $\tau_{i}^{(k)}$ using (8);
7: end for
8: Update $\phi^{(m)}$ using (10) or Adam-style step;
9: end for

where $\eta_{\phi}$ is the learning rate. We summarized our proposed algorithm in Algorithm 1.

Remark 3.1. Note that since $\ell_i(\phi, \tau_i)$ is strictly convex and univariate with respect to $\tau_i$ , in each iteration $m$ , $\tau_i^{(K)}$ is guaranteed to be near-optimal (i.e., $\tau_i^{(K)} \approx \tau_i^\star$ ). Therefore, the convergence of Algorithm 1 can be guaranteed by the convergence of stochastic gradient descent on the reward model parameter $\phi$ .

# 3.5. Extension to Direct Preference Optimization (DPO)

Our adaptive preference scaling approach is generic and can be extended to DPO (Rafailov et al., 2024), which is another popular method for policy learning from human preferences. DPO directly learns the policy in supervised manner using the preference data of state-action pairs $\mathcal{D}_{\mathrm{pref}} = (s_i, a_{w,i}, a_{l,i})_{i=1}^N$ . This approach forgoes the need to learn the reward function explicitly by the reparameterization of reward function r with respect to its optimal policy $\pi_r$ ,

$$
r (s, a) = \beta \log \frac {\pi_ {r} (a | s)}{\pi_ {\text { ref }} (a | s)} + \beta \log Z (s), \tag {11}
$$

where $Z(s) = \sum_{a}\pi_{\mathrm{ref}}(a|s)\exp \left(r(s,a) / \beta\right)$ and $\pi_{\mathrm{ref}}$ denotes the reference policy. By plugging in (11) back into (2), we have the policy optimization problem

$$
\min _ {\pi} \mathcal {L} _ {\mathrm{DPO}} (\pi) = - \mathbb {E} _ {(s, a _ {w}, a _ {l}) \sim \mathcal {D} _ {\mathrm{pref}}} \log \sigma \big (\beta r _ {\pi} (a _ {w} | s) - \beta r _ {\pi} (a _ {l} | s) \big),
$$

where $r_{\pi}(a|s) = \log (\pi (a|s) / \pi_{\mathrm{ref}}(a|s))$ denotes the log-probability ratio.

Similarly, we can integrate adaptive preference scaling into DPO by plugging in (11) into (6). By merging $\beta$ with the $\tau_{i}$ and $\rho$ , we can further obtain the adaptive DPO (Ada-DPO) formulation as

$$
\min _ {\pi , \tau_ {1}, \dots , \tau_ {N} \in \Omega} \mathcal {L} _ {\mathrm{Ada-DPO}} (\pi , \tau_ {1}, \dots , \tau_ {N}) := \frac {1}{N} \sum_ {i = 1} ^ {N} \left[ - \tau_ {i} \log \sigma \left(\frac {r _ {\pi} (a _ {w , i} | s _ {i}) - r _ {\pi} (a _ {l , i} | s _ {i})}{\tau_ {i}}\right) + \rho \tau_ {i} \right].
$$

Remark 3.2. Note that the proposed adaptive preference loss can be further combined with other RLHF approaches, such as PEBBLE (Lee et al., 2021) and SURF (Park et al., 2022), which still optimize the standard cross-entropy loss (see Lee et al. (2021, Eq. (4)) and Park et al. (2022, Eq. (3))).

# 3.6. Extension to Quadratic Regularization

We now introduce a variant of our adaptive preference loss that uses quadratic regularization for $\tau$ . This modification removes the need for the hyperparameter $\tau_{max}$ in $\Omega$ , easing the tuning effort. We define the following instance-level adaptive preference loss with quadratic regularization:

$$
\min _ {\tau \geq \tau_ {0}} \ell_ {\text { quad }} (r, \tau) := - \tau \log p _ {r, \tau} (z _ {w} \succ z _ {l}) + \rho_ {0} \tau^ {2} - \log 2 \tau . \tag {12}
$$

Compared to (4), which includes a linear regularization term of $(\rho_{0}-\log2)\tau$ , (12) modifies the regularization term with coefficient $\rho_{0}$ to be quadratic while keeping the term $\log2\tau$ linear. Additionally, in (12), the constraint on $\tau$ only specifies a lower bound $\tau_{0}$ and no longer includes an upper bound $\tau_{max}$ . The following proposition provides theoretical insights for this modification.

Proposition 3.2. Assume we have a pair of trajectory segments $z_{1}, z_{2}$ , and the preference distribution $p(z_{1} \succ z_{2}) = p^{*} \in (0, 1)$ . Consider the problem of minimizing the expectation of our adaptive loss function with quadratic regularization over the preference distribution:

$$
\min _ {r, \tau \geq \tau_ {0}} - \tau p ^ {*} \log \left(\sigma \big ((r (z _ {1}) - r (z _ {2})) / \tau \big)\right) - \tau (1 - p ^ {*}) \log \left(\sigma \big ((r (z _ {2}) - r (z _ {1})) / \tau \big)\right) + \rho_ {0} \tau^ {2} - \log 2 \tau .
$$

Then the minimizer $\tau^{*}$ and $r^{*}$ of the expected loss satisfy

$$
\tau^ {\star} = \max \left\{\tau_ {0}, \left(p ^ {*} \log \left(p ^ {*}\right) + \left(1 - p ^ {*}\right) \log \left(1 - p ^ {*}\right) + \log 2\right) / \left(2 \rho_ {0}\right) \right\},
$$

$$
r ^ {*} (z _ {1}) - r ^ {*} (z _ {2}) = \tau^ {*} \sigma^ {- 1} (p ^ {*}).
$$

Here, $\sigma^{-1}$ is the inverse of sigmoid function.

Note that unlike the adaptive preference loss with linear regularization described in Proposition 3.1, the optimal value $\tau^{\star}$ for quadratic regularization does not involve the upper bound $\tau_{\mathrm{max}}$ .

# 4. Experiments

In this section, we examine the effectiveness of our adaptive preference loss based on robotic control and natural language generation tasks.

# 4.1. Robotic Control

Experiment Setup. We apply our proposed reward learning method on 3 robotic control tasks from the PyBullet (Coumans & Bai, 2016-2019) environments: HalfCheetah, Ant, and Hopper. These environments are similar to those available in OpenAI Gym (Brockman et al., 2016) but they are known to be much harder to solve (Raffin et al., 2022). Similarly to Gao et al. (2023), our setting is synthetic, where we use the ground truth rewards to provide preference labels on each pair of samples due to high expense of collecting human preferences. For the reward function, we use two-hidden-layer MLPs, each

![](images/15e787c77ca5729d423f7cbea017af806418d8bbf715dc4f0e38c89b27fd8d3b.jpg)

<details>
<summary>line</summary>

| Timesteps (1e6) | Pref  | Ada-Pref |
| --------------- | ----- | -------- |
| 0.0             | -1200 | -1200    |
| 0.5             | 1500  | 1500     |
| 1.0             | 2200  | 2200     |
| 1.5             | 2600  | 2600     |
| 2.0             | 2700  | 2700     |
| 2.5             | 2800  | 2800     |
| 3.0             | 2900  | 2900     |
</details>

![](images/c89f1c3824c06fa301fcdfb4f4ad12881259394b29d97a22cfc4e04c7bf61eb5.jpg)

<details>
<summary>line</summary>

| Timesteps (1e6) | Pref  | Ada-Pref |
| --------------- | ----- | -------- |
| 0.0             | 0     | 0        |
| 0.5             | 1000  | 1000     |
| 1.0             | 2000  | 2000     |
| 1.5             | 2500  | 2500     |
| 2.0             | 2800  | 2900     |
| 2.5             | 2900  | 3000     |
| 3.0             | 2800  | 3100     |
</details>

![](images/d011168a4a6f5856ed927e5422a201525c8887411564644cec6dee6e15401aeb.jpg)

<details>
<summary>line</summary>

| Timesteps (1e6) | Pref  | Ada-Pref |
| --------------- | ----- | -------- |
| 0.0             | 0     | 0        |
| 0.5             | 700   | 1000     |
| 1.0             | 900   | 1300     |
| 1.5             | 1000  | 1400     |
| 2.0             | 1100  | 1500     |
| 2.5             | 1150  | 1600     |
| 3.0             | 1200  | 1700     |
</details>

![](images/f23d3e0cf2c51ff19ffb53183ca23eceff5517c5ab71538011a66935c388635e.jpg)

<details>
<summary>line</summary>

| Percentile | Pref  | Ada-Pref |
| ---------- | ----- | -------- |
| 0          | 2400  | 2550     |
| 20         | 2500  | 2700     |
| 40         | 2550  | 2800     |
| 60         | 2850  | 2900     |
| 80         | 3000  | 3100     |
| 100        | 3200  | 3300     |
</details>

(a) HalfCheetah

![](images/07b7fbe7470baa5cea6358baa44518f23c07b50ea2d9bc600074e45771374ae8.jpg)

<details>
<summary>line</summary>

| Percentile | Pref  | Ada-Pref |
| ---------- | ----- | -------- |
| 0          | 2850  | 3050     |
| 20         | 2950  | 3100     |
| 40         | 3000  | 3200     |
| 60         | 3050  | 3250     |
| 80         | 3150  | 3350     |
| 100        | 3250  | 3600     |
</details>

(b) Ant

![](images/179e4f03753ec28956e2e19c40af2a55efb156add5dea6cd2ca234418b7165b0.jpg)

<details>
<summary>line</summary>

| Percentile | Pref  | Ada-Pref |
| ---------- | ----- | -------- |
| 0          | 0     | 0        |
| 20         | 100   | 2000     |
| 40         | 1300  | 2000     |
| 60         | 2000  | 2200     |
| 80         | 2300  | 2500     |
| 100        | 2500  | 2500     |
</details>

(c) Hopper   
Figure 2. Learning curve plots (top) and percentile plots (bottom) for Pref and Ada-Pref. For the learning curve plots, returns at each timestep are averaged across 10 different seeds, then smoothed over timesteps using an exponential moving average (EMA) with a smoothing factor of $\alpha = 0.1$ . For the percentile plots, returns from 10 different seeds are sorted in ascending order.

containing 64 hidden units. This configuration is aligned with the designs of both the policy and value networks. Following Christiano et al. (2017), we repeat the following three steps for each stage: (i) We sample a set of trajectories by the policy $\pi$ , and update the policy with proximal policy optimization (PPO, Schulman et al. (2017)) alongside a reward function $\hat{r}$ . (ii) We split the segments (the sequence of state-action pairs) into a training set and a testing set. Then, we randomly sample pairs of segments from the training set, and generate $\mathcal{D}_{\mathrm{pref}}$ with preference labels. We do the same to the testing set, and generate $\mathcal{D}_{\mathrm{pref}}^{\prime}$ . (iii) We train the reward function $\hat{r}$ on $\mathcal{D}_{\mathrm{pref}}$ , and use $\mathcal{D}_{\mathrm{pref}}^{\prime}$ for evaluating the preference prediction of $\hat{r}$ .

For notational simplicity, we name our adaptive preference scaling method for reward learning as “Ada-Pref”. We compare Ada-Pref with the baseline method “Pref”, which uses the standard cross-entropy loss for reward learning. For every 10000 timesteps the policy $\pi$ runs, we evaluate the learned policy based on 20 test episodes. We also compute the average preference prediction accuracy of the learned reward function across stages. We set the budget to 3 million timesteps and perform training over 10 different seeds. For hyperparameter tuning in both reward learning and policy optimization, we apply two different criteria: 1) We identify the best policy based on its performance (the one with the highest return) and subsequently select the corresponding reward function. 2) We choose the best reward function based on its performance (the one with the highest average preference prediction accuracy) and then select the corresponding policy. Details of the implementations and hyperparameter tuning procedures are in Appendix B.1.

# Results. We summarize the results on three PyBullet tasks as follows:

Table 1 and Figure 2 illustrate the results for Pref and Ada-Pref on the PyBullet tasks, based on the first hyperparameter tuning criterion. In Table 1, we report the highest return of the best policy and the average preference accuracy of the corresponding reward function. We can see that Ada-Pref consistently outperforms Pref in terms of return on all three tasks and achieves comparable preference accuracy. The upper panel of Figure 2 shows the learning curve plots. We can see that Ada-Pref surpasses Pref at nearly every timestep and reaches a higher plateau across all tasks. The lower panel of Figure 2 presents percentile plots from different seeds to demonstrate individual run behaviors. As shown, we confirm that Ada-Pref consistently outperforms Pref at every percentile across all tasks.

Table 2 presents the results for Pref and Ada-Pref based on the second hyperparameter tuning criterion. In Table 2, we report the average preference accuracy of the best learned reward function and the highest return of the corresponding policy. From Table 2, we can see that both methods show a decrease in performance compared to Table 1, while Ada-Pref still outperforms Pref in terms of both preference accuracy and return on all three tasks. Furthermore, Ada-Pref demonstrates greater resistance to performance degradation than Pref, indicating its superior ability to align the learned reward function with policy optimization. This alignment allows for effective policy selection based on preference accuracy without the need to evaluate the policy using ground truth rewards.

Table 1. Table for the highest return of the best policy and the average preference prediction accuracy of the corresponding reward function. 

<table><tr><td>Task</td><td>Method</td><td>Return</td><td>Preference Accuracy (%)</td></tr><tr><td rowspan="2">HalfCheetah</td><td>Pref</td><td>2724.42</td><td>89.09</td></tr><tr><td>Ada-Pref</td><td>2875.45</td><td>89.46</td></tr><tr><td rowspan="2">Ant</td><td>Pref</td><td>2917.81</td><td>85.57</td></tr><tr><td>Ada-Pref</td><td>3177.11</td><td>85.48</td></tr><tr><td rowspan="2">Hopper</td><td>Pref</td><td>1324.91</td><td>92.08</td></tr><tr><td>Ada-Pref</td><td>1692.1</td><td>91.36</td></tr></table>

Table 2. Table for the average preference prediction accuracy of the best reward function and the highest return of the corresponding policy. 

<table><tr><td>Task</td><td>Method</td><td>Return</td><td>Preference Accuracy (%)</td></tr><tr><td rowspan="2">HalfCheetah</td><td>Pref</td><td>2620.83</td><td>89.41</td></tr><tr><td>Ada-Pref</td><td>2865.07</td><td>90.75</td></tr><tr><td rowspan="2">Ant</td><td>Pref</td><td>2750.99</td><td>87.93</td></tr><tr><td>Ada-Pref</td><td>3008.69</td><td>89.23</td></tr><tr><td rowspan="2">Hopper</td><td>Pref</td><td>744.66</td><td>93.18</td></tr><tr><td>Ada-Pref</td><td>1134.73</td><td>93.26</td></tr></table>

# 4.2. Natural Language Generation

Experiment Setup. We apply DPO with our proposed adaptive preference loss (Ada-DPO) to two open-ended text generation tasks: summarization and single-turn dialogue. We adopt the Llama-2 7B model (Touvron et al., 2023) as the backbone and conduct instruction tuning on each task to obtain the initial reference models. For summarization, the policy generates summaries given posts collected from Reddit. We use the filtered TL;DR summarization dataset (Völske et al., 2017) for instruction tuning, which contains more than 117K Reddit posts, each with a human-written summary. We apply

the human preferences collected by Stiennon et al. (2020) for preference optimization, where each transcript contains a pair of responses along with a preference label. For single-turn dialogue, the policy responds to various human queries ranging from simple questions to complex demands. We utilize the Anthropic Helpful and Harmless dialogue preferences dataset (Bai et al., 2022) for both instruction tuning and preference optimization. This dataset contains 170k human-AI dialogues, with each dialogue containing two AI responses and a human preference label. We use the preferred responses for instruction tuning and the full set of preferences for optimization. For instruction tuning stage, we fine-tune the entire Llama-2 model. For the alignment stage using Ada-DPO and different baselines, we apply LoRA fine-tuning for computational efficiency concerns, as we need to simultaneously tune multiple hyperparameters. The rank of the LoRA adaptor is 64. We consider three baseline methods: DPO (Rafailov et al., 2024), $\Psi$ Preference Optimization with Identity Mapping (IPO, Azar et al. (2024)) and Sequence Likelihood Calibration with Human Feedback (SLiC-HF, Zhao et al. (2023)).

As human evaluation is prohibitively expensive, we use Claude 2 (Anthropic, 2023), a proprietary large language model, to automatically evaluate responses based on summary quality and helpfulness/harmlessness for the summarization and dialogue tasks, respectively. Prior work has shown that Claude 2 and GPT-4 can effectively measure a quantitative improvement over the instruction-tuned model (Dubois et al., 2024). We split a small subset from each instruction tuning dataset for testing and calculate the win rate against the instruction-tuned reference model as the evaluation metric. The percentage of instances where the response generated by policy A is preferred over policy B is referred to as the win rate of A against B. We also split a subset from each preference optimization dataset to validate the preference prediction accuracy. Details of the implementations and hyperparameter selections are in Appendix B.2.

Results. We summarize the results on the two natural language generation tasks as follows:

In Figure 3, we select the model with the highest win rate and present the win rate and its preference accuracy for all baselines. We observe that Ada-DPO outperforms the other baselines on both tasks in terms of win rate and achieves comparable preference accuracy. In Figure 4, we display the performance of the model selected with the highest accuracy (not win rate). As shown, Ada-DPO achieves a significant improvement beyond the DPO baseline in terms of win rate and obtains a comparable preference accuracy. This again indicates that Ada-DPO yields better alignment between the learned reward function and policy optimization, allowing good policy selection based on preference accuracy without a proprietary LLM judge.

![](images/01176627ee93084938c2ffafd5b9c17611c58888983f86a02fb66e113fa87cd9.jpg)

<details>
<summary>bar</summary>

| Model | Method | Win Rate (%) | Preference Accuracy (%) |
| :--- | :--- | :--- | :--- |
| SLiC-HF | Summarization | 53.5 | 68.5 |
| SLiC-HF | Dialogue | 53.8 | 69.0 |
| DPO | Summarization | 53.7 | 69.2 |
| DPO | Dialogue | 53.4 | 54.5 |
| IPO | Summarization | 52.2 | 71.0 |
| IPO | Dialogue | 58.5 | 70.5 |
| Ada-DPO | Summarization | 55.0 | 69.8 |
| Ada-DPO | Dialogue | 60.5 | 56.5 |
</details>

Figure 3. The best win rate and the preference prediction accuracy of the corresponding model.

![](images/c818f30f9c587effe707afc3b21ddb0226b835ca824c9408be2053c79de2dd6d.jpg)

<details>
<summary>bar</summary>

| Method | Win Rate (%) | Preference Accuracy (%) |
| :--- | :--- | :--- |
| Summarization | 10 | 76 |
| Dialogue | 50 | 58 |
| Ada-DPO | 45 | 76 |
| Ada-DPO | 56 | 58 |
</details>

Figure 4. The best preference prediction accuracy and the win rate of the corresponding model.

# 4.3. Detailed Analysis

We present detailed analyses of Ada-Pref and Ada-DPO for both the Ant and summarization tasks. Figure 5(a) presents a histogram of the learned scaling factors $\tau$ for the Ant task. We can see that around $60\%$ of these scaling factors reach the upper bound, while about $10\%$ converge to the lower bound, and the rest are distributed across the region. In Figure 5(b), we explore the relationship between preference strength and the learned scaling factors $\tau$ , and in Figure 5(c), we investigate the relationship between preference strength and the learned reward difference for Pref and Ada-Pref. We measure preference strength using the true reward difference, categorize it into five percentile bins, and then bin the scaling factors and the learned reward differences accordingly to compute the average. As can be seen, the learned scaling factor increases monotonically with preference strength, demonstrating that the our method successfully adapts the loss scaling to the varying degrees of preference in the data. Furthermore, Ada-Pref learns smaller reward differences for pairs with ambiguous preferences and learns larger reward differences for those with strong preferences, compared to Pref. This indicates that our method leads to a more flexible reward function.

![](images/47a4082aefdbc0b2f89895efc60891c9958e84a81a798c75ec70bd3eff569e87.jpg)

<details>
<summary>bar</summary>

| τ    | Percentage (%) |
| ---- | -------------- |
| 0.2  | 7              |
| 0.4  | 3              |
| 0.6  | 2              |
| 0.8  | 1              |
| 1.0  | 63             |
</details>

(a) Histogram of $\tau$

![](images/b7d528b49be5804f7a4446e7a44cffcf021f2ab7679565049eceae9d2fdfe37f.jpg)

<details>
<summary>line</summary>

| Preference Strength | Average τ |
| ------------------- | --------- |
| 1                   | 0.68      |
| 2                   | 0.70      |
| 3                   | 0.79      |
| 4                   | 0.87      |
| 5                   | 0.95      |
</details>

(b) Preference strength and $\tau$

![](images/d4ce8312e1aa5bb6328bb339f07f8ba26eefb64f11c7e71ba7bbf873e511bf3e.jpg)

<details>
<summary>line</summary>

| Preference Strength | Pref    | Ada-Pref |
| ------------------- | ------- | -------- |
| 1                   | 0.0000  | 0.0000   |
| 2                   | 0.0035  | 0.0050   |
| 3                   | 0.0065  | 0.0130   |
| 4                   | 0.0120  | 0.0190   |
| 5                   | 0.0115  | 0.0245   |
</details>

(c) Preference strength and learned reward difference   
Figure 5. Histogram of the learned scaling factors, relationship between preference strength and the learned scaling factors, and relationship between preference strength and the learned reward difference. All plots are from the Ant task.

In Figure 6(a), we plot a histogram of the learned scaling factors $\tau$ for the summarization task. We can see that around $40\%$ of the scaling factors converge to the upper bound, with the rest distributed across the region. We also display the relationship between the confidence scores and the scaling factors in Figure 6(b). The confidence score is an integer from 1 to 4 included in the dataset, and a higher score denotes a stronger preference. We bin the scaling factors based on confidence scores and compute the average. As shown, the scaling factors positively correlate with confidence scores, justifying that we learn larger $\tau$ for strong preferences and smaller $\tau$ for ambiguous ones.

![](images/013584f2d93ae582ca427e26c31c1a71a65a3f5f66a2a719cbe0f3dee9332a28.jpg)

<details>
<summary>bar</summary>

| τ   | Percentage (%) |
| --- | -------------- |
| 0   | 3              |
| 1   | 2              |
| 2   | 1              |
| 3   | 1              |
| 4   | 45             |
</details>

(a) Histogram of $\tau$

![](images/4071bdecaf0d2fd9523ee001b258ac6497342424c3925d9b469d026063cffd60.jpg)

<details>
<summary>line</summary>

| Confidence Score | Average τ |
| ---------------- | --------- |
| 1                | 3.0       |
| 2                | 3.1       |
| 3                | 3.2       |
| 4                | 3.6       |
</details>

(b) Confidence score and $\tau$   
Figure 6. Histogram of the learned scaling factors and relationship between the confidence scores and the learned scaling factors. Both plots are from the summarization task.

# Sample pair with large $\tau$ ( $\tau = 4.0$ )

# Prompt

POST: This morning, I was walking between terminals in the Minneapolis airport. A bunch of us got stuck behind a group of 8 teenage girls slowly walking ...

Chosen: 8 spoiled brat girls got stuck at the airport due to their immaturity and obliviousness, and I'd like to know how I can help them.

Rejected: Teenage girls, who we had no idea were teenage girls, were being obnoxiously obnoxiously obnoxiously obnoxiously obnoxiously obnoxiously obnoxiously

![](images/51f5006748fb5f3cdc2cc352d0cd762399ed0876e7ee9d566e55d281fc9f9501.jpg)

<details>
<summary>bar</summary>

| Method   | Learned Reward Dif. |
| -------- | ------------------- |
| DPO      | 1.13                |
| Ada-DPO  | 30.4                |
</details>

# Sample pair with small $\tau(\tau=0.1)$

# Prompt

POST: On our second date I told her I had Bipolar 2 and she was fine with it. She borrowed a book I had on the disorder that helped her understand. Everything ...

Chosen: I have bipolar 2, and recently my depression and moodiness is getting to my girlfriend. She is afraid to come over because she thinks im losing interest in her. How do I help her?

Rejected: I have bipolar 2, and recently my depression and moodiness is getting to my girlfriend. She is afraid to come over because she is afraid I will lose interest in her.

![](images/adacec08552eaa1c7d26cc33f69fc29c89daf1726f943b3bc458a2a9d4fd79c8.jpg)

<details>
<summary>bar</summary>

| Method   | Learned Reward Dif. |
| -------- | ------------------- |
| DPO      | 0.385               |
| Ada-DPO  | 0.031               |
</details>

Figure 7. Examples of preference sample pairs with large (left) and small (right) scaling factors $\tau$ , and the comparison of the learned reward difference. The preferred (chosen) responses are colored by green and the rejected responses are colored by red.

We further present two pairs of preference samples where Ada-DPO assigns large or small scaling factors in Figure 7. We observe that the sample pair with a large scaling factor shows a strong preference, as the rejected response is nonsensical while the chosen one is clear. Ada-DPO learns a larger reward difference for such data, while it is much smaller with DPO. Conversely, for the sample pair with a small scaling factor, the two responses are very similar, indicating its ambiguity. Ada-DPO learns a small reward difference on this pair, while DPO gets a large reward difference.

# 4.4. Experiments with Quadratic Regularization

![](images/245e6574c5799314a8f827cc0c6a1f1f63c124f4699c71467269d7d04ea9d618.jpg)

<details>
<summary>line</summary>

| Timesteps (1e6) | Pref  | Ada-Pref-Quad |
| --------------- | ----- | ------------- |
| 0.0             | 0     | 0             |
| 0.5             | 1500  | 1600          |
| 1.0             | 2200  | 2400          |
| 1.5             | 2600  | 2800          |
| 2.0             | 2700  | 2900          |
| 2.5             | 2750  | 3000          |
| 3.0             | 2700  | 3000          |
</details>

(a) Learning curve

![](images/add1229c0ed108cebcf40d884ccb1a75b70e1509f93133f631166dcab66ecd30.jpg)

<details>
<summary>line</summary>

| Percentile | Pref   | Ada-Pref-Quad |
| ---------- | ------ | ------------- |
| 0          | 2850   | 2900          |
| 20         | 2950   | 3000          |
| 40         | 3000   | 3050          |
| 60         | 3050   | 3100          |
| 80         | 3150   | 3300          |
| 100        | 3250   | 3700          |
</details>

(b) Percentile plot   
Figure 8. Learning curve (left) and percentile plot (right) for Pref and Ada-Pref-Quad. Both plots are from the Ant task.

We provide the experiment results for our adaptive preference loss with quadratic regularization. Here, we name the method as “Ada-Pref-Quad” and the one applied to DPO as “Ada-DPO-Quad”. Table 3 and Figure 8 show the results for Pref and Ada-Pref-Quad on the Ant task, and DPO and Ada-DPO-Quad on the single-turn dialogue. In Table 3, we report the performance of the best policy and the preference accuracy of the corresponding reward function. From Table 3, we can see that Ada-Pref-Quad outperforms Pref on the Ant task, and Ada-DPO-Quad surpasses DPO on the single-turn dialogue in terms of return and win rate, respectively. Figures 8 presents the learning curve and the percentile plot for the Ant task. As shown, Ada-Pref-Quad surpasses Pref at every timestep and across all percentiles.

Table 3. Table for the highest return (left) and the best win rate (right) of the best policy and the preference prediction accuracy of the corresponding reward function. 

<table><tr><td>Task</td><td>Method</td><td>Return</td><td>Preference Accuracy (%)</td><td>Task</td><td>Method</td><td>Win Rate (%)</td><td>Preference Accuracy (%)</td></tr><tr><td rowspan="2">Ant</td><td>Pref</td><td>2917.81</td><td>90.08</td><td rowspan="2">Dialogue</td><td>DPO</td><td>53.38</td><td>54.39</td></tr><tr><td>Ada-Pref-Quad</td><td>3116.57</td><td>90.66</td><td>Ada-DPO-Quad</td><td>56.00</td><td>53.56</td></tr></table>

Figure 9(a) shows a histogram of the scaling factors $\tau$ learned by Ada-Pref-Quad for the Ant task. Compared to Figure 5(a), we can see much smoother distribution of $\tau$ due to the quadratic regularization. Figures 9(b) and 9(c) illustrate the relationship between preference strength and the learned scaling factors $\tau$ , and the relationship between preference strength and the learned reward difference for Pref and Ada-Pref-Quad. As can be seen, the learned scaling factor for Ada-Pref-Quad increases monotonically with preference strength, indicating that the quadratic regularization maintains the adaptability of loss scaling to the varying preference levels in the data. Moreover, Ada-Pref-Quad learns smaller reward differences for pairs with ambiguous preferences and learns larger reward differences for those with strong preferences. This demonstrates that Ada-Pref-Quad also leads to a more flexible reward function compared to Pref.

# 4.5. Discussions on Hyperparameter Tuning

Compared to the cross-entropy loss, our method needs three additional hyperparameters: the bounds on the scaling factors $\tau_{0}$ and $\tau_{max}$ , and the regularization parameter $\rho$ . In our experiments, we fixed $\tau_{0}$ at 0.1 without tuning it, as this value worked well for all five tasks. We did tune $\tau_{max}$ to adjust the scale of $\tau$ , but this can be avoided by using the quadratic

![](images/81d9926c4631c43103b599fb297709a14babc26aa15b18d5193a45c46202bc12.jpg)

<details>
<summary>histogram</summary>

| τ | Percentage (%) |
| --- | --- |
| 0.0-0.1 | 18 |
| 0.1-0.2 | 12.5 |
| 0.2-0.3 | 14 |
| 0.3-0.4 | 14 |
| 0.4-0.5 | 12.5 |
| 0.5-0.6 | 9.5 |
| 0.6-0.7 | 5.5 |
| 0.7-0.8 | 2.5 |
| 0.8-0.9 | 0 |
| 0.9-1.0 | 0 |
</details>

(a) Histogram of $\tau$

![](images/cb89aa968fee6d486d9efe1694c190e72b8db6056a3a34ddac2625b58a47148d.jpg)

<details>
<summary>line</summary>

| Preference Strength | Average τ |
| ------------------- | --------- |
| 1                   | 0.23      |
| 2                   | 0.25      |
| 3                   | 0.30      |
| 4                   | 0.37      |
| 5                   | 0.50      |
</details>

(b) Preference strength and $\tau$

![](images/992fc8e84327bfbe40751e3e609247ce82865ab25f03a65e8b4b671d5998c9e9.jpg)

<details>
<summary>line</summary>

| Preference Strength | Pref    | Ada-Pref-Quad |
| ------------------- | ------- | ------------- |
| 1                   | 0.0000  | 0.0000        |
| 2                   | 0.0035  | 0.0065        |
| 3                   | 0.0060  | 0.0130        |
| 4                   | 0.0120  | 0.0205        |
| 5                   | 0.0115  | 0.0235        |
</details>

(c) Preference strength and learned reward difference   
Figure 9. Histogram of the learned scaling factors, relationship between preference strength and the learned scaling factors, and relationship between preference strength and the learned reward difference. All plots are for Ada-Pref-Quad on the Ant task.

regularization formulation described in Section 3.6. The parameter $\rho$ turns out to be more important, because it controls the distribution of the scaling factors. We performed a careful grid search to tune $\rho$ in our experiments. Figure 10 shows the hyperparameter sensitivity of $\rho$ on the Ant and summarization tasks. Overall, we found that smaller values of $\rho$ often lead to better performance.

![](images/c446bf4524b1bde907b5c5332571e57ac109d628cf9bd38cfcd4b97425951ea6.jpg)

<details>
<summary>line</summary>

| p     | Return | Preference Acc. |
|-------|--------|-----------------|
| -0.6  | 3180   | 86              |
| -0.4  | 3050   | 87              |
| -0.2  | 2950   | 88              |
</details>

![](images/56cf8c1ae9073b5b345d281e99cd65a6d9f918fbe5d785e8a83f2d907bd43d8a.jpg)

<details>
<summary>line</summary>

| ρ     | Win rate | Accuracy |
| ------ | -------- | -------- |
| -0.7   | 52       | 68       |
| -0.6   | 51       | 70       |
| -0.5   | 53       | 69       |
| -0.4   | 23       | 64       |
| -0.3   | 23       | 66       |
| -0.2   | 24       | 65       |
</details>

Figure 10. Hyperparameter sensitivity of $\rho$ .

# 5. Conclusion

RLHF is an emerging challenge in machine learning. Prior to the popularity of models like ChatGPT, research on designing proper loss functions for reward learning was limited. To bridge this gap, we explore uncertainties in underlying preference strengths and propose an adaptive preference loss function. This loss function incorporates instance-specific scaling factors to modulate the correspondence between reward differences and preference distributions. Taking the result in this paper as an initial start, we expect more sophisticated and stronger follow-up work that applies to RLHF with similar structures. All of these efforts may ultimately assist in developing more principled RLHF methods to better control risks associated with advanced AI systems.

# References

Anthropic. Claude, 2023. URL https://www.anthropic.com.   
Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., and Calandriello, D. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pp. 4447–4455. PMLR, 2024.   
Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., Joseph, N., Kadavath, S., Kernion, J., Conerly, T., Showk, S. E., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Hume, T., Johnston, S., Kravec, S., Lovitt, L., Nanda, N., Olsson, C., Amodei, D., Brown, T. B., Clark, J., McCandlish, S., Olah, C., Mann, B., and Kaplan, J. Training a helpful and harmless assistant with reinforcement learning from human feedback. CoRR, abs/2204.05862, 2022.   
Balanya, S. A., Maroñas, J., and Ramos, D. Adaptive temperature scaling for robust calibration of deep neural networks. Neural Computing and Applications, pp. 1–23, 2024.   
Ben-Tal, A., Den Hertog, D., De Waegenaere, A., Melenberg, B., and Rennen, G. Robust solutions of optimization problems affected by uncertain probabilities. Management Science, 59(2):341–357, 2013.   
Bertsekas, D. P. Projected newton methods for optimization problems with simple constraints. SIAM Journal on control and Optimization, 20(2):221–246, 1982.   
Bertsimas, D., Gupta, V., and Kallus, N. Data-driven robust optimization. Mathematical Programming, 167:235–292, 2018.   
Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.   
Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J., and Zaremba, W. Openai gym. arXiv preprint arXiv:1606.01540, 2016.   
Broscheit, S., Do, Q., and Gaspers, J. Distributionally robust finetuning bert for covariate drift in spoken language understanding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, pp. 1970–1985, 2022.   
Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.   
Coumans, E. and Bai, Y. Pybullet, a python module for physics simulation for games, robotics and machine learning, 2016-2019. URL https://pybullet.org/.   
Ding, Z., Han, X., Liu, P., and Niethammer, M. Local temperature scaling for probability calibration. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6889–6899, 2021.   
Dubois, Y., Li, C. X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., Guestrin, C., Liang, P. S., and Hashimoto, T. B. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems, 36, 2024.   
Duchi, J. C., Glynn, P. W., and Namkoong, H. Statistics of robust optimization: A generalized empirical likelihood approach. Mathematics of Operations Research, 46(3):946–969, 2021.   
Gao, L., Schulman, J., and Hilton, J. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pp. 10835–10866. PMLR, 2023.   
Gokhale, T., Chaudhary, A., Banerjee, P., Baral, C., and Yang, Y. Semantically distributed robust optimization for vision-and-language inference. In 60th Annual Meeting of the Association for Computational Linguistics, pp. 1493–1513, 2022.   
Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. In International conference on machine learning, pp. 1321–1330. PMLR, 2017.

Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.   
Joy, T., Pinto, F., Lim, S.-N., Torr, P. H., and Dokania, P. K. Sample-dependent adaptive temperature scaling for improved calibration. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 14919–14926, 2023.   
Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. International Conference on Learning Representations, 2015.   
Kuhn, D., Esfahani, P. M., Nguyen, V. A., and Shafieezadeh-Abadeh, S. Wasserstein distributionally robust optimization: Theory and applications in machine learning. In Operations research & management science in the age of analytics, pp. 130–166. Informs, 2019.   
Lee, K., Smith, L., and Abbeel, P. Pebble: Feedback-efficient interactive reinforcement learning via relabeling experience and unsupervised pre-training. In International Conference on Machine Learning, 2021.   
Ma, X., Yin, P., Liu, J., Neubig, G., and Hovy, E. Softmax q-distribution estimation for structured prediction: A theoretical interpretation for raml. arXiv preprint arXiv:1705.07136, 2017.   
Michel, P., Hashimoto, T., and Neubig, G. Modeling the second player in distributionally robust optimization. International Conference on Learning Representations, 2021.   
Oren, Y., Sagawa, S., Hashimoto, T. B., and Liang, P. Distributionally robust language modeling. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4227–4237, 2019.   
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.   
Park, J., Seo, Y., Shin, J., Lee, H., Abbeel, P., and Lee, K. SURF: Semi-supervised reward learning with data augmentation for feedback-efficient preference-based reinforcement learning. In International Conference on Learning Representations, 2022.   
Qi, Q., Lyu, J., Chan, K.-S., Bai, E.-W., and Yang, T. Stochastic constrained dro with a complexity independent of sample size. Transactions on Machine Learning Research, 2023.   
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.   
Raffin, A. Rl baselines3 zoo, 2020. URL https://github.com/DLR-RM/rl-baselines3-zoo.   
Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., and Dormann, N. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1–8, 2021.   
Raffin, A., Kober, J., and Stulp, F. Smooth exploration for robotic reinforcement learning. In Conference on Robot Learning, pp. 1634–1644. PMLR, 2022.   
Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019.   
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.   
Song, Z., Cai, T., Lee, J. D., and Su, W. J. Reward collapse in aligning large language models. CoRR, abs/2305.17608, 2023.   
Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize from human feedback. Advances in Neural Information Processing Systems, 2020.   
Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pp. 5026–5033. IEEE, 2012.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
Völske, M., Potthast, M., Syed, S., and Stein, B. Tl; dr: Mining reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization, pp. 59–63, 2017.   
von Werra, L., Belkada, Y., Tunstall, L., Beeching, E., Thrush, T., Lambert, N., and Huang, S. Trl: Transformer reinforcement learning, 2020. URL https://github.com/huggingface/trl.   
Wang, F. and Liu, H. Understanding the behaviour of contrastive loss. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2495–2504, 2021.   
Wang, P.-H., Hsieh, S.-I., Chang, S.-C., Chen, Y.-T., Pan, J.-Y., Wei, W., and Juan, D.-C. Contextual temperature for language modeling. arXiv preprint arXiv:2012.13575, 2020.   
Wen, H., Yi, X., Yao, T., Tang, J., Hong, L., and Chi, E. H. Distributionally-robust recommendations for improving worst-case user experience. In Proceedings of the ACM Web Conference 2022, pp. 3606–3610, 2022.   
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations, 2020.   
Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., and Liu, P. J. Slic-hf: Sequence likelihood calibration with human feedback. CoRR, abs/2305.10425, 2023.

# A. Derivation and Proofs of Section 3

# A.1. Derivation of Equation (4)

In this subsection, we present the full derivation of Equation (4). Recall the following loss:

$$
\ell_ {r} (z _ {1}, z _ {2}) = \max _ {p \in \Delta_ {2}} p _ {1} d _ {r} (z _ {1}, z _ {2}) + p _ {2} d _ {r} (z _ {2}, z _ {1}) - \tau_ {0} \mathrm{KL} (p, 1 / 2) \qquad \text {s.t.} \quad \mathrm{KL} (p, 1 / 2) \leq \rho_ {0}.
$$

Using the Lagrangian duality, we have

$$
\max _ {p \in \Delta_ {2}} \min _ {\lambda \geq 0} p _ {1} d _ {r} (z _ {1}, z _ {2}) + p _ {2} d _ {r} (z _ {2}, z _ {1}) - \tau_ {0} \mathrm{KL} (p, 1 / 2) - \lambda (\mathrm{KL} (p, 1 / 2) - \rho_ {0}).
$$

By strong duality theorem, we have

$$
\min _ {\lambda \geq 0} \max _ {p \in \Delta_ {2}} p _ {1} d _ {r} (z _ {1}, z _ {2}) + p _ {2} d _ {r} (z _ {2}, z _ {1}) - \tau_ {0} \mathrm{KL} (p, 1 / 2) - \lambda (\mathrm{KL} (p, 1 / 2) - \rho_ {0}),
$$

which is equivalent to

$$
\min _ {\lambda \geq 0} \max _ {p \in \Delta_ {2}} p _ {1} d _ {r} (z _ {1}, z _ {2}) + p _ {2} d _ {r} (z _ {2}, z _ {1}) - (\lambda + \tau_ {0}) (\mathrm{KL} (p, 1 / 2) - \rho_ {0}) - \tau_ {0} \rho_ {0}.
$$

We let $\tau = \lambda +\tau_0$ and obtain

$$
\min _ {\tau \geq \tau_ {0}} \max _ {p \in \Delta_ {2}} p _ {1} d _ {r} (z _ {1}, z _ {2}) + p _ {2} d _ {r} (z _ {2}, z _ {1}) - \tau (\mathrm{KL} (p, 1 / 2) - \rho_ {0}) - \tau_ {0} \rho_ {0}.
$$

Now, we consider the optimality conditions for the inner constrained maximization problem by defining the following Lagrangian function:

$$
L _ {r, \tau} (p, \mu) = p _ {1} d _ {r} (z _ {1}, z _ {2}) + p _ {2} d _ {r} (z _ {2}, z _ {1}) - \tau (\mathrm{KL} (p, 1 / 2) - \rho_ {0}) - \mu \bigg (\sum_ {k = 1} ^ {2} p _ {k} - 1 \bigg),
$$

where $\mu$ is the Lagrange multiplier. The optimal solutions $p^{r,\tau}$ to the inner maximization problem satisfy the following KKT conditions:

$$
d _ {r} (z _ {1}, z _ {2}) - \tau (\log (p ^ {r, \tau}) + 1) - \mu = 0,
$$

$$
d _ {r} (z _ {2}, z _ {1}) - \tau (\log (p ^ {r, \tau}) + 1) - \mu = 0,
$$

$$
\text { and } \sum_ {k = 1} ^ {2} p _ {k} ^ {r, \tau} = 1.
$$

Then we have

$$
p _ {1} ^ {r, \tau} = \frac {\exp \left(d _ {r} (z _ {1} , z _ {2}) / \tau\right)}{\exp \left(d _ {r} (z _ {1} , z _ {2}) / \tau\right) + \exp \left(d _ {r} (z _ {2} , z _ {1}) / \tau\right)} \quad \text {and} \quad p _ {2} ^ {r, \tau} = \frac {\exp \left(d _ {r} (z _ {2} , z _ {1}) / \tau\right)}{\exp \left(d _ {r} (z _ {1} , z _ {2}) / \tau\right) + \exp \left(d _ {r} (z _ {2} , z _ {1}) / \tau\right)}.
$$

Plugging in $p_{1}^{r,\tau}$ and $p_{2}^{r,\tau}$ back into the inner maximization problem, we have

$$
\min _ {\tau \geq \tau_ {0}} \tau \log \left(\exp \left(d _ {r} (z _ {1}, z _ {2}) / \tau\right) + \exp \left(d _ {r} (z _ {2}, z _ {1}) / \tau\right)\right) - \tau \log 2 + (\tau - \tau_ {0}) \rho_ {0}.
$$

Without loss of generality, we let $z_{1} = z_{w}$ and $z_{2} = z_{l}$ , and obtain

$$
\min _ {\tau \geq \tau_ {0}} - \tau \log \sigma \left(\frac {r (z _ {w}) - r (z _ {l})}{\tau}\right) + (\rho_ {0} - \log 2) \tau ,
$$

where $\sigma$ is the logistic function. This completes the derivation.

# A.2. Proof of Proposition 3.1

We first derive the expectation of the adaptive loss. By taking expectation of (4) with $\rho = \rho_{0} - \log 2$ , we have:

$$
\begin{array}{l} \underset {z _ {1}, z _ {2}} {\mathbb {E}} \left\{\mathbf {1} (z _ {1} \succ z _ {2}) \Big [ - \tau \log \left(\sigma \big ((r (z _ {1}) - r (z _ {2})) / \tau \big)\right) + \rho \tau \right] \\ \left. + \mathbf {1} \left(z _ {2} \succ z _ {1}\right) \left[ - \tau \log \left(\sigma \left(\left(r \left(z _ {2}\right) - r \left(z _ {1}\right)\right) / \tau\right)\right) + \rho \tau \right] \right\} \\ = p ^ {*} \left[ - \tau \log \left(\sigma \left((r (z _ {1}) - r (z _ {2})) / \tau\right)\right) + \rho \tau \right] + (1 - p ^ {*}) \left[ - \tau \log \left(\sigma \left((r (z _ {2}) - r (z _ {1})) / \tau\right)\right) + \rho \tau \right] \\ = - \tau p ^ {*} \log \left(\sigma \big ((r (z _ {1}) - r (z _ {2})) / \tau \big)\right) - \tau (1 - p ^ {*}) \log \left(\sigma \big ((r (z _ {2}) - r (z _ {1})) / \tau \big)\right) + \rho \tau . \\ \end{array}
$$

By the optimality condition of $r(z_{1}) - r(z_{2})$ and $\tau$ , we have

$$
r (z _ {1}) - r (z _ {2}) = \tau \sigma^ {- 1} (p ^ {*}), \tag {13}
$$

where $\sigma^{-1}$ is the inverse of sigmoid function. Plugging (13) into the objective in (7), we obtain

$$
\min _ {\tau \in \Omega} \bigl [ - p ^ {*} \log (p ^ {*}) - (1 - p ^ {*}) \log (1 - p ^ {*}) + \rho \bigr ] \tau ,
$$

whose objective is essentially linear in $\tau$ . Hence, when $-p^{*}\log(p^{*})-(1-p^{*})\log(1-p^{*})+\rho>0$ , the corresponding optimal $\tau^{*}$ is at the lower bound $\tau_{0}$ and the optimal reward difference $r^{*}(z_{1})-r^{*}(z_{2})=\tau_{0}\sigma^{-1}(p^{*})$ given the optimality condition. Conversely, when $-p^{*}\log(p^{*})-(1-p^{*})\log(1-p^{*})+\rho<0$ , we have $\tau^{*}=\tau_{\max}$ and $r^{*}(z_{1})-r^{*}(z_{2})=\tau_{\max}\sigma^{-1}(p^{*})$ . This completes the proof.

# A.3. Proof of Proposition 3.2

We first derive the expectation of the adaptive loss with quadratic regularization. By taking expectation of (12), we have:

$$
\begin{array}{l} \underset {z _ {1}, z _ {2}} {\mathbb {E}} \left\{\mathbf {1} (z _ {1} \succ z _ {2}) \big [ - \tau \log \left(\sigma \big ((r (z _ {1}) - r (z _ {2})) / \tau \big)\right) + \rho_ {0} \tau^ {2} - \log 2 \tau \right] \\ \left. + \mathbf {1} \left(z _ {2} \succ z _ {1}\right) \left[ - \tau \log \left(\sigma \left(\left(r \left(z _ {2}\right) - r \left(z _ {1}\right)\right) / \tau\right)\right) + \rho_ {0} \tau^ {2} - \log 2 \tau \right] \right\} \\ = p ^ {*} \big [ - \tau \log \big (\sigma \big ((r (z _ {1}) - r (z _ {2})) / \tau \big) \big) + \rho_ {0} \tau^ {2} - \log 2 \tau \big ] \\ + \left(1 - p ^ {*}\right) \left[ - \tau \log \left(\sigma \left(\left(r (z _ {2}) - r (z _ {1})\right) / \tau\right)\right) + \rho_ {0} \tau^ {2} - \log 2 \tau \right] \\ = - \tau p ^ {*} \log \left(\sigma \left((r (z _ {1}) - r (z _ {2})) / \tau\right)\right) - \tau (1 - p ^ {*}) \log \left(\sigma \left((r (z _ {2}) - r (z _ {1})) / \tau\right)\right) + \rho_ {0} \tau^ {2} - \log 2 \tau . \tag {14} \\ \end{array}
$$

By the optimality condition of $r(z_{1}) - r(z_{2})$ and $\tau$ , we have

$$
r (z _ {1}) - r (z _ {2}) = \tau \sigma^ {- 1} (p ^ {*}), \tag {15}
$$

where $\sigma^{-1}$ is the inverse of sigmoid function. Plugging (15) into the objective in (14), we obtain

$$
\min _ {\tau \geq \tau_ {0}} \left[ - p ^ {*} \log (p ^ {*}) - (1 - p ^ {*}) \log (1 - p ^ {*}) - \log 2 \right] \tau + \rho_ {0} \tau^ {2}. \tag {16}
$$

Note that (16) is always bounded without the need of an upper bound of $\tau$ . Specifically, with any $p^{\star} \in (0,1)$ , we have $-p^{*}\log(p^{*}) - (1 - p^{*})\log(1 - p^{*}) - \log 2 \leq 0$ and $\tau^{\star} = \max\{\tau_{0}, (p^{*}\log(p^{*}) + (1 - p^{*})\log(1 - p^{*}) + \log 2)/(2\rho_{0})\}$ . This completes the proof.

# B. Implementation Details

# B.1. Robotic Control

Our implementations of robotic control tasks are based on Stable-Baselines3 (Raffin et al., 2021) and RL Zoo training framework (Raffin, 2020). For both Ada-Pref and Pref, we set the segment length to 1 as it is the most basic unit that the gold reward model is able to provide preference for. Additional experiments with a segment size of 25 for the Ant, HalfCheetah, and Hopper are in Appendix C. We calculate the average preference prediction accuracy over the first 1 million timesteps. At each training step, we assign preference labels to every possible pair of trajectory segments within a mini-batch based on their ranking from the gold reward model. We set the batch size to 64 for the HalfCheetah and Ant tasks and 4 for

the Hopper task. We tune the number of epochs in $\{1,3,5\}$ . We use Adam optimizer (Kingma & Ba, 2015) and tune the learning rate in $\{5e-3,1e-3,5e-4,1e-4\}$ for the HalfCheetah and Ant, and set the learning rate to 1e-2 for the Hopper. For Ada-Pref, we tune the $\tau_{max}$ in $\{1.0,3.0\}$ and the $\rho_{0}$ in $\{0.1,0.3,0.5\}$ . We fix $\tau_{0}=0.1$ and the number of Newton iterations to 3 for all experiments. Details of the chosen hyperparameters for reward learning for all three tasks are summarized in Tables 4 and 5. For PPO, we reused all hyperparameters from the original paper (Schulman et al., 2017) optimized for the Mujoco benchmark (Todorov et al., 2012). Details of the hyperparameters for PPO are in Table 6.

Table 4. Chosen hyperparameters for reward learning used for Table 1. 

<table><tr><td>Task</td><td>Method</td><td># epochs</td><td>Learning rate</td><td> $\tau_{\text{max}}$ </td><td> $\rho_0$ </td></tr><tr><td rowspan="2">HalfCheetah</td><td>Pref</td><td>5</td><td>5e-3</td><td>-</td><td>-</td></tr><tr><td>Ada-Pref</td><td>3</td><td>1e-3</td><td>3.0</td><td>0.5</td></tr><tr><td rowspan="2">Ant</td><td>Pref</td><td>1</td><td>5e-4</td><td>-</td><td>-</td></tr><tr><td>Ada-Pref</td><td>5</td><td>1e-4</td><td>1.0</td><td>0.1</td></tr><tr><td rowspan="2">Hopper</td><td>Pref</td><td>5</td><td>1e-2</td><td>-</td><td>-</td></tr><tr><td>Ada-Pref</td><td>5</td><td>1e-2</td><td>1.0</td><td>0.1</td></tr></table>

Table 5. Chosen hyperparameters for reward learning used for Table 2. 

<table><tr><td>Task</td><td>Method</td><td># epochs</td><td>Learning rate</td><td> $\tau_{\text{max}}$ </td><td> $\rho_0$ </td></tr><tr><td rowspan="2">HalfCheetah</td><td>Pref</td><td>3</td><td>5e-3</td><td>-</td><td>-</td></tr><tr><td>Ada-Pref</td><td>5</td><td>1e-3</td><td>3.0</td><td>0.5</td></tr><tr><td rowspan="2">Ant</td><td>Pref</td><td>5</td><td>5e-4</td><td>-</td><td>-</td></tr><tr><td>Ada-Pref</td><td>5</td><td>1e-3</td><td>1.0</td><td>0.5</td></tr><tr><td rowspan="2">Hopper</td><td>Pref</td><td>3</td><td>1e-2</td><td>-</td><td>-</td></tr><tr><td>Ada-Pref</td><td>3</td><td>1e-2</td><td>3.0</td><td>0.3</td></tr></table>

Table 6. Chosen hyperparameters for PPO. 

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>optimizer</td><td>Adam</td></tr><tr><td>discount (γ)</td><td>0.99</td></tr><tr><td>value function coefficient</td><td>0.5</td></tr><tr><td>entropy coefficient</td><td>0.0</td></tr><tr><td>shared network between actor and critic</td><td>False</td></tr><tr><td>max gradient norm</td><td>0.5</td></tr><tr><td>learning rate schedule</td><td>constant</td></tr><tr><td>advantage normalization</td><td>True</td></tr><tr><td>clip range value function</td><td>no</td></tr><tr><td>number of steps per rollout</td><td>2048</td></tr><tr><td>initial log σ</td><td>0.0</td></tr><tr><td>learning rate</td><td>3·10-4</td></tr><tr><td>number of epochs</td><td>10</td></tr><tr><td>number of samples per mini-batch</td><td>64</td></tr><tr><td>non-linearity</td><td>Tanh</td></tr><tr><td>GAE coefficient (λ)</td><td>0.95</td></tr><tr><td>clip range</td><td>0.2</td></tr><tr><td>orthogonal initialization</td><td>yes</td></tr></table>

# B.2. Natural Language Generation

Our implementations of natural language generation tasks are based on transformers (Wolf et al., 2020) and trl training framework (von Werra et al., 2020). We provide more details on each task as follows:

# B.2.1. SUMMARIZATION

For the instruction tuning stage, we randomly select 800 data from the filtered TL;DR summarization dataset (Völske et al., 2017) for testing the policy and leave the rest for supervised tuning. In the preference optimization stage, we split the preference dataset (Stiennon et al., 2020) into a training and testing set to evaluate the preference accuracy. For both stages, we omit the title and only use the post content as the prompt. The prompt format follows: "POST: post content.\n\nTL;DR:"

For Ada-DPO and all baselines, we set the batch size to 32 and train 1 epoch for both instruction tuning and preference optimization. We set the $\alpha$ parameters of LoRA fine-tuning to 16, and tune the other parameters by grid search. The learning rate is tuned in $\{5e - 6, 5e - 5, 1e - 4, 5e - 4\}$ . SLiC-HF, IPO and DPO include parameter $\beta$ , which is tuned in a range of $\{0.01, 0.1, 0.3, 0.5\}$ . For Ada-DPO, we tune the $\rho_0$ in $\{0.05, 0.1, 0.3, 0.5\}$ and the $\tau_{\mathrm{max}}$ in $\{1.0, 4.0, 5.0, 10.0\}$ . We fix $\tau_0 = 0.1$ and the number of Newton iterations to 5 for all experiments. The best Ada-DPO is achieved with $lr = 5e - 5$ , $\rho_0 = 0.1$ , and $\tau_{\mathrm{max}} = 4.0$ .

# B.2.2. SINGLE-TURN DIALOGUE

We use the original training split in the Anthropic Helpful and Harmless dialogue preferences dataset (Bai et al., 2022) for training in both stages. We randomly select 800 samples from its testing split to calculate the win rate, and use the rest of the data in the testing split for validation during preference optimization. We use the original data format.

In the dialogue task, we use the same batch size of 32 and 1 epoch for training. The learning rate is tuned in $\{5e - 6, 5e - 5, 1e - 4\}$ . The parameter $\beta$ for baselines is tuned in a range of $\{0.01, 0.1, 0.3\}$ . For Ada-DPO, we tune the $\rho_0$ in $\{0.05, 0.1, 0.3, 0.5\}$ , $\tau_{\max}$ in $\{1, 5, 10\}$ and fix $\tau_0 = 0.1$ . The best Ada-DPO is achieved with $lr = 5e - 5$ , $\rho_0 = 0.05$ , and $\tau_{\max} = 5.0$ .

# B.2.3. EVALUATION PROMPT.

We apply Claude 2 for win rate judgments. Here, we include the used prompt for generating win rates for both generation tasks. The order of responses is switched and a response only counts for win if it wins in both orders.

# Summarization win rate prompt.

Human: Which of the following summaries does a better job of summarizing the most important points in the given forum post, without including unimportant or irrelevant details? A good summary is both precise and concise.

Post: <prompt>
Summary A: <response A>
Summary B: <response B>

FIRST provide a one-sentence comparison of the two summaries, explaining which you prefer and why. SECOND, on a new line, state only "A" or "B" to indicate your choice. Your response should use the format:

Comparison: <one-sentence comparison and explanation>
Preferred: <"A" or "B">

Assistant:

# Dialogue win rate prompt.

Human: For the following query to a chatbot, which response is more helpful and harmless?

Query: <prompt>

Response A: <response A>

Response B: <response B>

FIRST provide a one-sentence comparison of the two responses and explain which you feel is more helpful and harmless. SECOND, on a new line, state only "A" or "B" to indicate which response is more helpful. Your response should use the format:

Comparison: <one-sentence comparison and explanation>

More helpful: < "A" or "B">

Assistant:

# C. Additional Experiments

We present the results for three PyBullet tasks, using a segment size of 25. Table 7 and Figure 11 show the performance of Pref and Ada-Pref on the PyBullet tasks, based on the first hyperparameter tuning criterion. Table 8 displays the results for Pref and Ada-Pref according to the second hyperparameter tuning criterion. These results reconfirm the effectiveness of our adaptive preference loss.

![](images/801b4e009746e9d3ed5b22f0912e3015a15839c9ba76a0c2ef6334b8ee020e4f.jpg)

<details>
<summary>line</summary>

| Timesteps (1e6) | Pref   | Ada-Pref |
| --------------- | ------ | -------- |
| 0.0             | -1200  | -1200    |
| 0.5             | 1500   | 1600     |
| 1.0             | 2200   | 2400     |
| 1.5             | 2500   | 2700     |
| 2.0             | 2600   | 2800     |
</details>

![](images/cb832f2d5fc66f0a28bbbc4ecbce3e27b4e3ee2b9caa10a51a7c0ec9576d2f0c.jpg)

<details>
<summary>line</summary>

| Timesteps (1e6) | Pref  | Ada-Pref |
| --------------- | ----- | -------- |
| 0.0             | 0     | 0        |
| 0.5             | 1500  | 1600     |
| 1.0             | 2500  | 2600     |
| 1.5             | 2700  | 2800     |
| 2.0             | 2800  | 2900     |
</details>

![](images/6c3d2189d7c1f67383705a07244be7d58caa80c6c3cf32194a6dcf538a705b0b.jpg)

<details>
<summary>line</summary>

| Timesteps (1e6) | Pref  | Ada-Pref |
| --------------- | ----- | -------- |
| 0.0             | 0     | 0        |
| 0.5             | 750   | 750      |
| 1.0             | 875   | 900      |
| 1.5             | 875   | 900      |
| 2.0             | 875   | 900      |
</details>

![](images/8ab069eadf0cae3b556d3ee75c7ac5e37e1ed772802577f96df7a2456f228646.jpg)

<details>
<summary>line</summary>

| Percentile | Pref  | Ada-Pref |
| ---------- | ----- | -------- |
| 0          | 2000  | 2250     |
| 20         | 2250  | 2400     |
| 40         | 2450  | 2600     |
| 60         | 2750  | 2850     |
| 80         | 3050  | 3100     |
| 100        | 3100  | 3150     |
</details>

(a) HalfCheetah

![](images/396c4225d869980aaf66bd3877b6d9fca5016a68485a4f429b404bbab485df62.jpg)

<details>
<summary>line</summary>

| Percentile | Pref  | Ada-Pref |
| ---------- | ----- | -------- |
| 0          | 2650  | 2780     |
| 20         | 2750  | 2900     |
| 40         | 2800  | 2920     |
| 60         | 2850  | 2980     |
| 80         | 2950  | 3200     |
| 100        | 3150  | 3350     |
</details>

(b) Ant

![](images/2740ac635c0faddf2d71b3da558ffde8bcb19ec9366a4cc1b4f6f1b8c5813300.jpg)

<details>
<summary>line</summary>

| Percentile | Pref  | Ada-Pref |
| ---------- | ----- | -------- |
| 0          | 0     | 0        |
| 20         | 0     | 0        |
| 40         | 750   | 0        |
| 60         | 1000  | 1900     |
| 80         | 1900  | 2100     |
| 100        | 2200  | 2150     |
</details>

(c) Hopper   
Figure 11. Learning curve plots (top) and percentile plots (bottom) for Pref and Ada-Pref. For the learning curve plots, returns at each timestep are averaged across 10 different seeds, then smoothed over timesteps using an exponential moving average (EMA) with a smoothing factor of $\alpha = 0.1$ . For the percentile plots, returns from 10 different seeds are sorted in ascending order.

Table 7. Table for the highest return of the best policy and the average preference prediction accuracy of the corresponding reward function. 

<table><tr><td>Task</td><td>Method</td><td>Return</td><td>Preference Accuracy (%)</td></tr><tr><td rowspan="2">HalfCheetah</td><td>Pref</td><td>2575.69</td><td>90.82</td></tr><tr><td>Ada-Pref</td><td>2689.9</td><td>90.35</td></tr><tr><td rowspan="2">Ant</td><td>Pref</td><td>2832.87</td><td>84.88</td></tr><tr><td>Ada-Pref</td><td>2960.47</td><td>84.09</td></tr><tr><td rowspan="2">Hopper</td><td>Pref</td><td>883.49</td><td>85.0</td></tr><tr><td>Ada-Pref</td><td>1025.74</td><td>85.15</td></tr></table>

Table 8. Table for the average preference prediction accuracy of the best reward function and the highest return of the corresponding policy. 

<table><tr><td>Task</td><td>Method</td><td>Return</td><td>Preference Accuracy (%)</td></tr><tr><td rowspan="2">HalfCheetah</td><td>Pref</td><td>2564.49</td><td>91.38</td></tr><tr><td>Ada-Pref</td><td>2609.03</td><td>90.79</td></tr><tr><td rowspan="2">Ant</td><td>Pref</td><td>2738.4</td><td>86.21</td></tr><tr><td>Ada-Pref</td><td>2917.22</td><td>85.15</td></tr><tr><td rowspan="2">Hopper</td><td>Pref</td><td>796.52</td><td>85.79</td></tr><tr><td>Ada-Pref</td><td>1025.74</td><td>85.15</td></tr></table>