# RAT: Adversarial Attacks on Deep Reinforcement Agents for Targeted Behaviors

Fengshuo Bai $^{1,2}$ , Runze Liu $^{6}$ ,
Yali Du $^{3}$ , Ying Wen $^{1,*}$ , Yaodong Yang $^{4,5,*}$ ,

$^{1}$ Shanghai Jiao Tong University $^{2}$ Zhongguancun Academy $^{3}$ King's College London

$^{4}$ Center for AI Safety and Governance, Institute for AI, Peking University

$^{5}$ State Key Laboratory of General Artificial Intelligence, Peking University

$^{6}$ Tsinghua Shenzhen International Graduate School, Tsinghua University

# Abstract

Evaluating deep reinforcement learning (DRL) agents against targeted behavior attacks is critical for assessing their robustness. These attacks aim to manipulate the victim into specific behaviors that align with the attacker's objectives, often bypassing traditional reward-based defenses. Prior methods have primarily focused on reducing cumulative rewards; however, rewards are typically too generic to capture complex safety requirements effectively. As a result, focusing solely on reward reduction can lead to suboptimal attack strategies, particularly in safety-critical scenarios where more precise behavior manipulation is needed. To address these challenges, we propose RAT, a method designed for universal, targeted behavior attacks. RAT trains an intention policy that is explicitly aligned with human preferences, serving as a precise behavioral target for the adversary. Concurrently, an adversary manipulates the victim's policy to follow this target behavior. To enhance the effectiveness of these attacks, RAT dynamically adjusts the state occupancy measure within the replay buffer, allowing for more controlled and effective behavior manipulation. Our empirical results on robotic simulation tasks demonstrate that RAT outperforms existing adversarial attack algorithms in inducing specific behaviors. Additionally, RAT shows promise in improving agent robustness, leading to more resilient policies. We further validate RAT by guiding Decision Transformer agents to adopt behaviors aligned with human preferences in various MuJoCo tasks, demonstrating its effectiveness across diverse tasks. The supplementary videos are available at https://sites.google.com/view/jj9uxjgmba5lr3g.

# 1 Introduction

Reinforcement learning (RL) (Sutton and Barto 2018) combined with deep neural networks (DNN) (LeCun, Bengio, and Hinton 2015) shows extraordinary capabilities of allowing agents to master complex behaviors in various domains, including robotic manipulation (Wang et al. 2023; Bai et al. 2023), video games (Zhang et al. 2023, 2024b; Wang\* et al. 2024; Wen et al. 2024), industrial applications (Xu and Yu 2023; Shi et al. 2024; Jia et al. 2024). However, recent findings (Huang et al. 2017; Pattanaik et al. 2018; Zhang et al. 2020, 2024a) show that even well-trained DRL agents suffer

Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

\*Corresponding authors. Email: yaodong.yang@pku.edu.cn, ying.wen@sjtu.edu.cn

![](images/3ea6cce2026de0d3753a57eede90c6d51e1fce1bed5ddf42ed69bf2ebd640ea2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["victim $"] --> B["targeted behavior attack (Ours)"]
    B --> C["reduce expected return"]
    C --> D["wp symbol"]
```
</details>

Figure 1: An example illustrating the distinction between our approach and generic attacks.

from vulnerability against test-time attacks, raising concerns in high-risk or safety-critical situations. To understand adversarial attacks on learning algorithms and enhance the robustness of DRL agents, it is crucial to evaluate the performance of the agents under any potential adversarial attacks with certain constraints. In other words, identifying a universal and strong adversary is essential.

Existing methods pay little attention to devising universal, efficient, targeted behavior attacks. Firstly, several methods primarily focused on reducing the cumulative reward often lack specified attack targets. Prior research (Zhang et al. 2020, 2021; Sun et al. 2022) considers training strong adversaries by perturbing state observations of victims to achieve the worst-case expected return. However, rewards lack the expressiveness to adequately encode complex safety requirements (Vamplew et al. 2022; Hasanbeig, Kroening, and Abate 2020). Additionally, requiring the victim's training rewards to craft such attacks is generally impractical. Therefore, only quantifying the decrease in cumulative reward can be too generic and result in suboptimal attack performance, particularly when adversaries are intended to execute specific safety-related attacks. Consider the scenario depicted in Figure 1, where a robot's objective is to collect coins. Previous attack methods aim at inducing the robot away from the coins by minimizing its expected return. However, this approach overlooks specific unsafe behaviors, such as manipulating the robot to collide with a bomb. Secondly, the previous targeted attack only considered predefined targets, which resulted in rigidity and inefficiency. (Hussenot, Geist, and Pietquin 2019a; Lin et al. 2017a) mainly focuses on mislead-

ing the agent towards a predetermined state or target policy, overlooking specific behaviors. Additionally, the difficulty of providing a well-designed targeted policy makes these methods hard to apply. In a broader context, these adversarial attacks are incapable of controlling the behaviors of agents as a form of universal attack.

In this paper, we present a novel adversarial attack method, RAT, which focuses on AdveRsarial Attacks against deep reinforcement learning agents for Targeted behavior. RAT consists of three core components: an intention policy, an adversary, and a weighting function, all trained simultaneously. Unlike previous methods that rely on predefined target policies, RAT dynamically trains an intention policy that aligns with human preferences, providing a flexible and adaptive behavioral target for the adversary. By leveraging advances in preference-based reinforcement learning (PbRL) (Lee, Smith, and Abbeel 2021; Park et al. 2022; Liu et al. 2022; Bai et al. 2024), the intention policy efficiently captures human intent during the training process. RAT employs the adversary to perturb the victim agent's observations, guiding the agent towards the behaviors specified by the intention policy. To further enhance attack effectiveness, we introduce a weighting function that adjusts the state occupancy measure, optimizing the distribution of states visited during training. This adjustment improves both the performance and efficiency of the attack. Through iterative refinement, RAT steers the victim agent toward specific human-desired behaviors with greater precision than existing adversarial attack methods.

Our contributions are summarized as follows: (1) We propose a universal targeted behavior attack method against DRL agents, designed to induce specific behaviors in a victim agent across a wide range of tasks. (2) We provide a theoretical analysis of RAT, offering a convergence guarantee under clearly defined conditions, which enhances the understanding of its effectiveness. (3) Through extensive experiments across various domains, we demonstrate that RAT significantly outperforms existing adversarial attack methods, showing that both online and offline RL agents, including Decision Transformer, are susceptible to our approach. (4) We introduce two variants, RAT-ATLA and RAT-WocaR, which demonstrate how RAT can be effectively employed to enhance the robustness of DRL agents through adversarial training, showing its versatility in both attack and defense.

# 2 Related Work

Adversarial Attacks on State Observations in DRL. Huang et al. (2017) applies the Fast Gradient Sign Method (FGSM) (Goodfellow, Shlens, and Szegedy 2015) to compute adversarial perturbations, directing the victim policy towards suboptimal actions. Pattanaik et al. (2018) introduces a strategy to make the victim choose the worst action based on its Q-function. Gleave et al. (2020) focuses on adversarial attacks within the context of a two-player Markov game rather than altering the agent's observation. Zhang et al. (2020) proposes the state-adversarial MDP (SA-MDP) and develops two adversarial attack methods, Robust Sarsa (RS) and Maximal Action Difference (MAD). SA-RL (Zhang et al. 2021) optimizes an adversary to perturb states using end-to-end RL. PA-AD (Sun et al. 2022) utilizes an RL-based "director" to determine the best policy perturbation direction and an optimization-based "actor" to generate perturbed states accordingly. Another line of work focuses on steering DRL agents toward specific states or policies. Lin et al. (2017b); Buddareddygari et al. (2022) propose targeted adversarial attack methods against DRL agents, aimed at directing the agent to a specific state. Hussenot, Geist, and Pietquin (2019b) offer a novel approach by attacking the agent to mimic a target policy. However, these methods often require access to the victim's training reward or a predetermined target state or policy, which may be impractical. Our method differs from these methods by emphasizing the manipulation of the victim's behaviors without needing access to the victim's training reward or a pre-defined target state or policy.

Robustness for State Observations in DRL. Training DRL agents with perturbed state observations from adversaries has been explored in various studies. Shen et al. (2020); Oikarinen et al. (2021) focus on a strategy, ensuring that the policy produces similar outputs for similar inputs, which has demonstrated certifiable performance in video games. Another research direction, as presented in Pinto et al. (2017); Mandlekar et al. (2017); Pattanaik et al. (2018), aims to enhance an agent's robustness by training it under adversarial attacks. Zhang et al. (2021) proposes ATLA, a method that alternates between training an RL agent and an RL adversary, significantly enhancing policy robustness. Building on this concept, Sun et al. (2022) proposed PA-ATLT, which employs a similar approach but utilizes a more advanced RL attacker. And several methods proposed by Fischer et al. (2019); Lütjens, Everett, and How (2020), concentrate on the lower bounds of the Q-function to certify an agent's robustness at every step. WocaR-RL (Liang et al. 2022b) is an efficient method that directly estimates and optimizes the worst-case reward of a policy under attacks without requiring extra samples for learning an attacker.

Preference-based RL. PbRL provides an effective way to incorporate human preferences into agent learning. Christiano et al. (2017) proposes a foundational framework for PbRL. Ibarz et al. (2018) utilizes expert demonstrations to initialize the policy, besides learning the reward model from human preferences. Nonetheless, these earlier methods often require extensive human feedback, which is typically not feasible in practical scenarios. Recent studies have addressed this limitation: Lee, Smith, and Abbeel (2021) develops a feedback-efficient PbRL algorithm, leveraging unsupervised exploration and reward relabeling. Park et al. (2022) furthers feedback efficiency through semi-supervised reward learning and data augmentation. Meanwhile, Liang et al. (2022a) proposes an intrinsic reward to enhance exploration. Continuing this trend, Liu et al. (2022) improves feedback efficiency by aligning the Q-function with human preferences. Additionally, several works (Bai et al. 2024; Liu et al. 2024) have been dedicated to improving feedback efficiency by providing diverse insights. In our research, we employ PbRL to capture human intent and train an intention policy, which serves as the learning target for training adversaries.

# 3 Problem Setup and Notations

The Victim Policy. In RL, agent learning can be modeled as a finite-horizon Markov Decision Process (MDP) defined as a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \gamma)$ . S and A denote state and action space, respectively. $R : S \times A \times S \to R$ is the reward function, and $\gamma \in (0,1)$ is the discount factor. $P : S \times A \times S \to [0,1]$ denotes the transition dynamics, which determines the probability of transferring to $s'$ given state s and action a. We denote the stationary policy $\pi_{\nu} : S \to \mathcal{P}(A)$ , where $\nu$ are parameters of the victim. We suppose the victim policy is fixed and uses the approximator.

Threat Model. To study targeted behavior attack with human preferences, we formulate it as rewarded state-adversarial Markov Decision Process (RSA-MDP). Formally, a RSA-MDP is a tuple $(\mathcal{S},\mathcal{A},\mathcal{B},\widehat{\mathcal{R}},\mathcal{P},\gamma)$ . The adversary $\pi_{\alpha}:\mathcal{S}\to \mathcal{P}(\mathcal{S})$ perturbs the states before the victim observes them, where $\alpha$ are parameters of the adversary. The adversary perturbs the state $\mathbf{s}$ into $\tilde{\mathbf{s}}$ restricted by $\mathcal{B}(\mathbf{s})$ (i.e., $\tilde{\mathbf{s}}\in \mathcal{B}(\mathbf{s})$ ). $\mathcal{B}(\mathbf{s})$ is defined as a small set $\{\tilde{\mathbf{s}}\in \mathcal{S}:\| \mathbf{s} - \tilde{\mathbf{s}}\| _p\leq \epsilon \}$ , which limits the attack power of the adversary, and $\epsilon$ is the attack budget. Since directly generating $\tilde{\mathbf{s}}\in \mathcal{B}(\mathbf{s})$ is hard, the adversary learns to produce a Gaussian noise $\Delta$ with $\ell_{\infty}(\Delta)$ less than 1, and we obtain the perturbed state through $\tilde{\mathbf{s}} = \mathbf{s} + \Delta *\epsilon$ . The victim takes action according to the observed $\tilde{\mathbf{s}}$ , while true states in the environment are not changed. Recall that $\pi_{\nu \circ \alpha}$ denotes the perturbed policy caused by adversary $\pi_{\alpha}$ , i.e., $\pi_{\nu \circ \alpha}(\cdot |\mathbf{s}) = \pi_{\nu}\left(\cdot |\pi_{\alpha}(\mathbf{s})\right),\forall \mathbf{s}\in \mathcal{S}$ .

Unlike SA-MDP (Zhang et al. 2020), RSA-MDP introduces $\widehat{\mathcal{R}}$ , which learns from human preferences. The target of RSA-MDP is to solve the optimal adversary $\pi_{\alpha}^{*}$ , which enables the victim to achieve the maximum cumulative reward (i.e., from $\widehat{\mathcal{R}}$ ) over all states. Lemma C.1 shows that solving the optimal adversary in RSA-MDP is equivalent to finding the optimal policy in MDP $\hat{\mathcal{M}} = (\mathcal{S}, \hat{\mathcal{A}}, \widehat{\mathcal{R}}, \widehat{\mathcal{P}}, \gamma)$ , where $\hat{\mathcal{A}} = \mathcal{S}$ and $\widehat{\mathcal{P}}$ is the transition dynamics of the adversary.

Lemma 3.1. Given a RSA-MDP $\mathcal{M} = (\mathcal{S},\mathcal{A},\mathcal{B},\widehat{\mathcal{R}},\mathcal{P},\gamma)$ and a fixed victim policy $\pi_{\nu}$ , there exists a MDP $\hat{\mathcal{M}} = (\mathcal{S},\hat{\mathcal{A}},\widehat{\mathcal{R}},\widehat{\mathcal{P}},\gamma)$ such that the optimal policy of $\hat{\mathcal{M}}$ is equivalent to the optimal adversary $\pi_{\alpha}$ in RSA-MDP given a fixed victim, where $\hat{\mathcal{A}} = \mathcal{S}$ and

$$
\widehat {\mathcal {P}} (\mathbf {s} ^ {\prime} | \mathbf {s}, \mathbf {a}) = \sum_ {\mathbf {a} \in \mathcal {A}} \pi_ {\nu} (\mathbf {a} | \widehat {\mathbf {a}}) \mathcal {P} (\mathbf {s} ^ {\prime} | \mathbf {s}, \mathbf {a}) \quad f o r \mathbf {s}, \mathbf {s} ^ {\prime} \in \mathcal {S} a n d \widehat {\mathbf {a}} \in \widehat {\mathcal {A}}.
$$

# 4 Method

In this section, we introduce RAT, a generic framework adaptable to any RL algorithm for conducting targeted behavior attack against DRL learners. RAT is composed of three integral components: an intention policy $\pi_{\theta}$ , the adversary $\pi_{\alpha}$ , and the weighting function $h_{\omega}$ , all of which are trained in tandem. The fundamental concept behind RAT is twofold: (1) It develops an intention policy to serve as the learning objective for the adversary. (2) A weighting function is trained to adjust the state occupancy measure of replay buffer, and the training of $\pi_{\alpha}$ and $h_{\omega}$ is formulated as a bi-level optimization problem. The framework of RAT is depicted in Figure 3, with a comprehensive procedure outlined in Appendix A.

![](images/17f8d30bb3b5f6e767772f14d5616fc36814438a27ea80e33583ae4273612956.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Earth"] --> B["\hat{r}_ψ"]
    B --> C["replay buffer"]
    C --> D["learning from human preferences"]
    D --> E["Human preference with checkmarks"]
    D --> F["Human preference with X marks"]
    D --> G["Human preference with checkmarks"]
    C --> H["π_θ(a|s)"]
    H --> C
    I["loop"] --> A
```
</details>

Figure 2: Diagram of PbRL. The reward model $\widehat{r}_{\psi}$ is trained to align with human intention, providing estimations of rewards for policy learning. The policy is optimized by using transitions relabeled by the up-to-date reward model.

# 4.1 Learning Intention Policy

RAT is designed to find an optimal adversary capable of manipulating the victim's behaviors in alignment with human intentions. To achieve this, we consider capturing human intentions and training an intention policy $\pi_{\theta}$ , which translates these abstract intentions into action-level behaviors. A practical approach to realizing this concept is through PbRL, a method that aligns the intention policy with human intent without the need for reward engineering. As depicted in Figure 2, within the PbRL framework, the agent does not rely on a ground-truth reward function. Instead, humans provide preference labels comparing two agent trajectories, and the reward model $\widehat{r}_{\psi}$ is trained to match human preferences (Christiano et al. 2017).

Formally, we denote a state-action sequence of length k, $\{s_{t+1}, a_{t+1}, \cdots, s_{t+k}, a_{t+k}\}$ as a segment $\sigma$ . Given a pair of segments $(\sigma^{0}, \sigma^{1})$ , humans provide a preference label y indicating which segment is preferred. Here, y represents a distribution, specifically $y \in \{(0,1), (1,0), (0.5,0.5)\}$ . In accordance with the Bradley-Terry model (Bradley and Terry 1952), we construct a preference predictor as shown in (1):

$$
P _ {\psi} \left[ \sigma^ {0} \succ \sigma^ {1} \right] = \frac {\exp \sum_ {t} \widehat {r} _ {\psi} \left(\mathbf {s} _ {t} ^ {0} , \mathbf {a} _ {t} ^ {0}\right)}{\sum_ {i \in \{0 , 1 \}} \exp \sum_ {t} \widehat {r} _ {\psi} \left(\mathbf {s} _ {t} ^ {i} , \mathbf {a} _ {t} ^ {i}\right)}, \tag {1}
$$

where $\sigma^{0} \succ \sigma^{1}$ indicates a preference for $\sigma^{0}$ over $\sigma^{1}$ . This predictor determines the probability of a segment being preferred, proportional to its exponential return.

The reward model is optimized to align the predicted preference labels with human preferences using a cross-entropy loss, as expressed in the following equation:

$$
\mathcal {L} (\psi) = - \underset {(\sigma^ {0}, \sigma^ {1}, y) \sim \mathcal {D}} {\mathbb {E}} \left[ \sum_ {i = 0} ^ {1} y (i) \log P _ {\psi} [ \sigma^ {i} \succ \sigma^ {1 - i} ] \right], \tag {2}
$$

where D represents a dataset of triplets $(\sigma^{0}, \sigma^{1}, y)$ that consist of segment pairs and corresponding human preference labels. By minimizing the cross-entropy loss as defined in (2), we derive an estimated reward function $\widehat{r}_{\psi}$ . This function is then utilized to provide reward estimations for policy learning using any RL algorithm. Following PEBBLE (Lee,

![](images/cc3c14d60b78a24eddcd27a943f13631ceefac64d71c1860bf5177b23c3e24bd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["learn from preference"] --> B["reward model"]
    B <--> C["(s, a, s')"]
    C --> D["replay buffer"]
    D --> E["behavior policy"]
    E --> F["πθ(a|s)"]
    E --> G["πνα(a|s)"]
    F --> H["outer loss Jπ"]
    G --> I["inner loss ℒ"]
    H --> J["hω(s)"]
    I --> J
    J --> K["outer level: optimize ω"]
    K --> L["inner level: optimize α"]
    L --> M["πθ(a|s)"]
    L --> N["πνα(a|s)"]
    M --> O["bi-level optimization"]
    N --> O
```
</details>

Figure 3: Overview of RAT. During training, it learns the intention policy $\pi_{\theta}$ and the reward model $\widehat{r}_{\psi}$ , following the principles of PbRL. Simultaneously, it trains an adversary $\pi_{\alpha}$ and a weighting function $h_{\omega}$ within a bi-level optimization framework. In the inner-level, the adversary is optimized such that the perturbed policy aligns with the intention policy. A validation loss $J_{\pi}$ is introduced, serving as a metric to assess the adversary's performance. In the outer-level, the weighting function is updated to improve the performance of the adversary by minimizing the outer loss $J_{\pi}$ .

Smith, and Abbeel 2021), we employ the Soft Actor-Critic (SAC) (Haarnoja et al. 2018) algorithm to train the intention policy $\pi_{\theta}$ . The Q-function $Q_{\phi}$ is optimized by reducing the Bellman residual, as defined below:

$$
J _ {Q} (\phi) = \underset {\tau_ {t} \sim \mathcal {B}} {\mathbb {E}} \left[ \left(Q _ {\phi} (\mathbf {s} _ {t}, \mathbf {a} _ {t}) - \widehat {r} _ {t} - \gamma \bar {V} (\mathbf {s} _ {t + 1})\right) ^ {2} \right], \tag {3}
$$

where $\bar{V} (\mathbf{s}_t) = \mathbb{E}_{\mathbf{a}_t\sim \pi_\theta}\bigl [Q_{\bar{\phi}}(\mathbf{s}_t,\mathbf{a}_t) - \mu \log \pi_\theta (\mathbf{a}_t|\mathbf{s}_t)\bigr ],\tau_t =$ $(\mathbf{s}_t,\mathbf{a}_t,\widehat{r}_t,\mathbf{s}_{t + 1})$ represents the transition at time $t$ , with $\bar{\phi}$ being the parameter of the target soft Q-function. The intention policy $\pi_{\theta}$ is updated to minimize the following loss:

$$
J _ {\pi} (\theta) = \mathbb {E} _ {\mathbf {s} _ {t} \sim \mathcal {B}, \mathbf {a} _ {t} \sim \pi_ {\theta}} \left[ \mu \log \pi_ {\theta} (\mathbf {a} _ {t} | \mathbf {s} _ {t}) - Q _ {\phi} (\mathbf {s} _ {t}, \mathbf {a} _ {t}) \right], (4)
$$

where $\mu$ is the temperature parameter.

In this way, RAT effectively captures human intent via the reward model $\widehat{r}_{\psi}$ and leverages $\pi_{\theta}$ to provide behavior-level guidance for the training of the adversary.

# 4.2 Learning Adversary and Weighting Function

To steer the victim policy towards behaviors desired by humans, RAT trains the adversary by minimizing the Kullback-Leibler (KL) divergence between the perturbed policy $\pi_{\nu\circ\alpha}$ and the intention policy $\pi_{\theta}$ . Additionally, certain pivotal moments during adversary training can significantly influence the success rate of attacks. To ensure a stable training process and enhance the adversary's performance, a weighting function $h_{\omega}$ is introduced to re-weight the state occupancy measure of dataset.

Formally, our method is formulated as a bi-level optimization algorithm. It alternates between updating the adversary $\pi_{\alpha}$ and the weighting function $h_{\omega}$ through inner and outer optimization processes. In the inner level, the adversary's parameters $\alpha$ are optimized by minimizing the re-weighted KL divergence between $\pi_{\nu\circ\alpha}$ and $\pi_{\theta}$ , as specified in (6). At the outer level, the weighting function is developed to identify crucial states and improve the adversary's performance, as guided by a performance metric of the adversary. This metric is represented as a meta-level loss $J_{\pi}$ , detailed in (7). The whole objective of RAT is formulated as:

$$
\min _ {\omega} J _ {\pi} (\alpha (\omega)), \tag {5}
$$

$$
\text { s.t. } \quad \alpha (\omega) = \arg \min _ {\alpha} \mathcal {L} (\alpha ; \omega , \theta).
$$

Inner Loop: Training Adversary $\pi_{\alpha}$ . In inner-level optimization, with the given intention policy $\pi_{\theta}$ and the weighting function $h_{\omega}$ , the goal is to identify the optimal adversary. This is achieved by minimizing the re-weighted KL divergence between $\pi_{\nu\circ\alpha}$ and $\pi_{\theta}$ , as shown in equation (6):

$$
\mathcal {L} (\alpha ; \omega , \theta) = \underset {\mathbf {s} \sim \mathcal {B}} {\mathbb {E}} \left[ h _ {\omega} (\mathbf {s}) D _ {\mathrm{KL}} \big (\pi_ {\nu \circ \alpha} (\cdot | \mathbf {s}) \| \pi_ {\theta} (\cdot | \mathbf {s}) \big) \right], \tag {6}
$$

where $h_{\omega}(\mathbf{s})$ represents the importance weights determined by the weighting function $h_{\omega}$ .

Intuitively, the adversary is optimized to ensure that the perturbed policy $\pi_{\nu\circ\alpha}$ aligns behaviorally with the intention policy. Concurrently, $h_{\omega}$ allocates varying weights to states, reflecting their differing levels of importance. Through the synergistic effort of the intention policy and the weighting function, our method effectively trains an optimal adversary. Outer Loop: Training Weighting Function $h_{\omega}$ . In outer-level optimization, the goal is to develop a precise weighting function that can identify significant moments and refine the state occupancy measure of the replay buffer to enhance adversary learning. As the intention policy is the target for the perturbed policy, it becomes simpler to establish a validation loss. This loss measures the perturbed policy's performance and simultaneously reflects the adversary's effectiveness. Consequently, the weighting function is trained to differentiate the importance of states by optimizing this validation loss. The perturbed policy $\pi_{\nu\circ\alpha}$ is assessed using a policy loss in (7), adapted from the policy loss in (4):

$$
J _ {\pi} (\alpha (\omega)) = \underset { \begin{array}{c} \mathbf {s} _ {t} \sim \mathcal {B}, \\ \mathbf {a} _ {t} \sim \pi_ {\nu \circ \alpha (\omega)} \end{array} } {\mathbb {E}} \left[ \mu \log \pi_ {\nu \circ \alpha (\omega)} \left(\mathbf {a} _ {t} \mid \mathbf {s} _ {t}\right) - Q _ {\phi} \left(\mathbf {s} _ {t}, \mathbf {a} _ {t}\right) \right], \tag {7}
$$

where $\alpha (\omega)$ denotes $\alpha$ implicitly depends on $\omega$ . The optimization process involves calculating the implicit derivative of $J_{\pi}(\alpha (\omega))$ with respect to $\omega$ and finding the optimal $\omega^{*}$ through optimization.

Practical Implementation. A one-step gradient update is used to approximate $\arg\min_{\alpha}$ , as shown in (8), thus establishing a connection between $\alpha$ and $\omega$ :

$$
\hat {\alpha} (\omega) \approx \alpha_ {t} - \eta_ {t} \left. \nabla_ {\alpha} \mathcal {L} (\alpha ; \omega , \theta) \right| _ {\alpha_ {t}}. \tag {8}
$$

The gradient of the outer loss with respect to $\omega$ is then determined using the chain rule:

$$
\begin{array}{l} \nabla_ {\omega} J _ {\pi} (\alpha (\omega)) | _ {\omega_ {t}} = \left. \nabla_ {\hat {\alpha}} J _ {\pi} (\hat {\alpha} (\omega)) \right| _ {\hat {\alpha} _ {t}} \left. \nabla_ {\omega} \hat {\alpha} _ {t} (\omega) \right| _ {\omega_ {t}} \\ = \sum_ {\mathbf {s}} f (\mathbf {s}) \cdot \left. \nabla_ {\omega} h (\mathbf {s}) \right| _ {\omega_ {t}}, \tag {9} \\ \end{array}
$$

where $f(\mathbf{s}) = -\eta_{t} \cdot (\nabla_{\hat{\alpha}} J_{\pi}(\alpha(\omega)))^{\top} \nabla_{\alpha} D_{\mathrm{KL}}(\pi_{\nu \circ \alpha}(\cdot | \mathbf{s}) \parallel \pi_{\theta}(\cdot | \mathbf{s}))$ . The detailed derivation is provided in Appendix B. The essence of this step is to establish and compute the relationship between $\alpha$ and $\omega$ . By obtaining the implicit derivative, RAT updates the parameters of the weighting function using gradient descent with an outer learning rate.

# 4.3 Theoretical Analysis

We provide convergence guarantee of RAT. In Theorem 4.1, we demonstrate the convergence rate of the outer loss. We demonstrate that the gradient of the outer loss with respect to $\omega$ will converge to zero. Consequently, RAT learns a more effective adversary by leveraging the importance of the weights generated by the optimal weighting function. Theorem 4.2 addresses the convergence of the inner loss. We prove that the inner loss of RAT algorithm converges to critical points under certain reasonable conditions, thereby ensuring that the parameters of the adversary can converge towards the optimal parameters. Detailed theorems and their proofs are available in Appendix D.

Theorem 4.1. Suppose $J_{\pi}$ is Lipschitz-smooth with constant $L$ , the gradient of $J_{\pi}$ and $\mathcal{L}$ is bounded by $\rho$ . Let the training iterations be $T$ , the inner-level optimization learning rate $\eta_t = \min \{1, \frac{c_1}{T}\}$ for some constant $c_1 > 0$ where $\frac{c_1}{T} < 1$ . Let the outer-level optimization learning rate $\beta_t = \min \{\frac{1}{L}, \frac{c_2}{\sqrt{T}}\}$ for some constant $c_2 > 0$ where $c_2 \leq \frac{\sqrt{T}}{L}$ , and $\sum_{t=1}^{\infty} \beta_t \leq \infty, \sum_{t=1}^{\infty} \beta_t^2 \leq \infty$ . The convergence rate of $J_{\pi}$ achieves

$$
\min _ {1 \leq t \leq T} \mathbb {E} \left[ \| \nabla_ {\omega} J _ {\pi} (\alpha_ {t + 1} (\omega_ {t})) \| ^ {2} \right] \leq \mathcal {O} \left(\frac {1}{\sqrt {T}}\right). \tag {10}
$$

Theorem 4.2. Suppose $J_{\pi}$ is Lipschitz-smooth with constant $L$ , the gradient of $J_{\pi}$ and $\mathcal{L}$ is bounded by $\rho$ . Let the training iterations be $T$ , the inner-level optimization learning rate $\eta_t = \min \{1, \frac{c_1}{T}\}$ for some constant $c_1 > 0$ where $\frac{c_1}{T} < 1$ . Let the outer-level optimization learning rate $\beta_t = \min \{\frac{1}{L}, \frac{c_2}{\sqrt{T}}\}$ for some constant $c_2 > 0$ where $c_2 \leq \frac{\sqrt{T}}{L}$ , and $\sum_{t=1}^{\infty} \beta_t \leq \infty, \sum_{t=1}^{\infty} \beta_t^2 \leq \infty$ . $\mathcal{L}$ achieves

$$
\lim _ {t \to \infty} \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| ^ {2} \right] = 0. \tag {11}
$$

# 5 Experiments

In this section, we evaluate our method using a range of robotic simulation manipulation tasks from Meta-world (Yu et al. 2020) and continuous locomotion tasks from MuJoCo (Todorov, Erez, and Tassa 2012). Our objective is to address the following key questions: (1) Does our method have the capacity to implement universal targeted behavior attack against DRL learners? (2) Can our approach successfully deceive a commonly used offline RL method, such as the Decision Transformer (Chen et al. 2021), to execute specific behaviors? (3) Does our method contribute to enhancing an agent's robustness through adversarial training? (4) Are the individual components within our approach effective? The responses to problems (1) – (4) are addressed in Sections 5.2 through 5.5, respectively. A detailed description of the experimental setup is available in Appendix E.

# 5.1 Setup

Compared Methods. We compare our algorithm with Random attack and two state-of-the-art evasion attack methods, including (1) Random: a basic baseline that samples random perturbed observations via a uniform distribution. (2) SA-RL (Zhang et al. 2021): learning an adversary in the form of end-to-end RL formulation. (3) PA-AD (Sun et al. 2022): combining RL-based “director” and non-RL “actor” to find state perturbations. (4) RAT: our proposed method, which collaboratively learns adversarial policy and weighting function with the guidance of intention policy.

Implementation Settings. In our experiments, all methods follow PEBBLE (Lee, Smith, and Abbeel 2021) to learn the reward model using the same number of preference labels. The key modification in employing PbRL is that the rewards in transitions are derived from the reward model $\widehat{r}_{\psi}$ , rather than ground-truth rewards, and this model is trained by minimizing equation 2. Specifically, in the original versions of SA-RL (Zhang et al. 2021) and PA-AD (Sun et al. 2022), the negative value of the reward obtained by the victim is used to train adversaries. We adapt this by using estimated rewards from $\widehat{r}_{\psi}$ . To evaluate performance effectively and expedite the training, we follow the foundational settings in PbRL (Lee, Smith, and Abbeel 2021; Park et al. 2022; Liu et al. 2022), considering the use of a scripted teacher that always provides accurate preference labels. For the manipulation scenario, we employ 9000 labels across all tasks. In the opposite behavior scenario, the label usage varies: 1000 for Window Close, 3000 for Drawer Close, 5000 for Faucet Open, Faucet Close, and Window Open, and 7000 for Drawer Open, Door Lock, and Door Unlock. More information about the scripted teacher and preference collection is detailed in Appendix A.2. Moreover, to minimize the influence of PbRL, we include oracle versions of SA-RL and PA-AD, which utilize the ground-truth rewards of the targeted task. For implementing SA-RL\* and PA-AD\*, the official repositories are employed. As in most existing research (Zhang et al. 2020, 2021; Sun et al. 2022), we also use state attacks with $L^{\infty}$ norm in our experiments.

Table 1: The average attack success rate, along with the standard deviation, is calculated for various evasion attacks against victim agents in both scenarios. The results are averaged over 30 episodes. Full results are available at Appendix F.1. 

<table><tr><td></td><td>Task</td><td>PA-AD (oracle)</td><td>PA-AD</td><td>SA-RL (oracle)</td><td>SA-RL</td><td>Random</td><td>RAT (ours)</td></tr><tr><td rowspan="8">Manipulation</td><td>Door Lock</td><td>4.50 ± 4.00</td><td>3.50 ± 6.63</td><td>76.50 ± 14.97</td><td>39.50 ± 30.48</td><td>0.00 ± 0.00</td><td>87.00 ± 10.00</td></tr><tr><td>Door Unlock</td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td>11.11 ± 13.43</td><td>0.56 ± 0.00</td><td>0.00 ± 0.00</td><td>97.00 ± 6.63</td></tr><tr><td>Window Open</td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td>30.00 ± 21.19</td><td>8.00 ± 15.13</td><td>0.00 ± 0.00</td><td>72.50 ± 19.62</td></tr><tr><td>Window Close</td><td>0.00 ± 0.00</td><td>0.50 ± 0.00</td><td>99.00 ± 3.00</td><td>23.50 ± 37.22</td><td>0.00 ± 0.00</td><td>72.50 ± 40.01</td></tr><tr><td>Drawer Open</td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td>100.00 ± 0.00</td><td>26.00 ± 27.28</td><td>0.00 ± 0.00</td><td>97.50 ± 4.00</td></tr><tr><td>Drawer Close</td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td>57.50 ± 18.00</td><td>4.00 ± 8.00</td><td>0.00 ± 0.00</td><td>76.00 ± 24.98</td></tr><tr><td>Faucet Open</td><td>1.50 ± 4.00</td><td>2.50 ± 4.00</td><td>63.50 ± 20.52</td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td>84.00 ± 21.19</td></tr><tr><td>Faucet Close</td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td>66.50 ± 16.85</td><td>4.50 ± 9.22</td><td>0.00 ± 0.00</td><td>91.00 ± 6.71</td></tr><tr><td rowspan="8">Opposite</td><td>Door Lock</td><td>9.50 ± 7.48</td><td>10.00 ± 9.17</td><td>8.00 ± 13.42</td><td>2.00 ± 0.00</td><td>1.00 ± 3.00</td><td>99.00 ± 3.00</td></tr><tr><td>Door Unlock</td><td>3.00 ± 5.00</td><td>4.00 ± 4.58</td><td>8.00 ± 18.33</td><td>6.00 ± 12.00</td><td>0.00 ± 0.00</td><td>98.50 ± 4.00</td></tr><tr><td>Window Open</td><td>15.50 ± 12.21</td><td>17.00 ± 11.14</td><td>15.00 ± 16.61</td><td>7.00 ± 16.12</td><td>1.00 ± 3.00</td><td>77.50 ± 33.41</td></tr><tr><td>Window Close</td><td>38.50 ± 23.69</td><td>55.00 ± 14.70</td><td>63.00 ± 34.70</td><td>20.00 ± 39.80</td><td>5.50 ± 5.00</td><td>99.00 ± 0.00</td></tr><tr><td>Drawer Open</td><td>1.50 ± 4.00</td><td>0.50 ± 3.00</td><td>1.11 ± 0.00</td><td>3.00 ± 0.00</td><td>0.00 ± 0.00</td><td>85.50 ± 29.34</td></tr><tr><td>Drawer Close</td><td>88.50 ± 7.81</td><td>79.00 ± 18.44</td><td>81.00 ± 20.88</td><td>63.00 ± 32.50</td><td>0.00 ± 0.00</td><td>92.00 ± 17.32</td></tr><tr><td>Faucet Open</td><td>6.50 ± 9.00</td><td>10.00 ± 13.75</td><td>6.00 ± 18.00</td><td>0.00 ± 0.00</td><td>0.00 ± 0.00</td><td>81.50 ± 29.68</td></tr><tr><td>Faucet Close</td><td>19.00 ± 13.27</td><td>32.00 ± 11.00</td><td>7.00 ± 12.81</td><td>8.00 ± 16.00</td><td>0.50 ± 0.00</td><td>96.00 ± 12.81</td></tr></table>

![](images/f3444a0db65c10528c4162a44ef4eeb92afde144a25b59734af4e90535fc3b56.jpg)

![](images/b3f2df8786216f1d8ee594fbe5508b4af03da4aa10614a17b0c5ac1be0423613.jpg)

![](images/05314e652e81b4306065b6623a4da80bc1c0bcaa2f296e1de7694dbdc1937d03.jpg)

![](images/a47465f216134ecb328adebae2e126030c0031cd2adbad65e5e9af1ad974aa74.jpg)

![](images/deb0ab2d89c26cb7516b7776d95a34a06f8ad2bb28c510c7f29dcea57a71b28a.jpg)  
(a) Cheetah-Run Backwards

![](images/9053cccc390b1a99c5993a1fc8c583a40d547c325a9df8ac9a78b45012950443.jpg)

![](images/0f8d7aba56818b7a122952a4be5937f6cac0bfab5e971bcfa2a79c98a8165709.jpg)

![](images/629fabf40935fe3262d9e7dc5f75a192bd12241cc779872de3e32ffffc45ef8a.jpg)

![](images/100d58346808146306a6c509a1a913a31bc17cd779e70bd574589e6b63b01bed.jpg)

![](images/c711957474d492fc5eae67cb6dc288e66d7e29f5c140a005025e77f1d0ae9ed0.jpg)  
(b) Walker-Stand on One Foot

![](images/c10f9b6c6c624873460d07151e38bc787090b5ecddfe059e88ddb684e56124c9.jpg)

![](images/85cbc8926ee7ef23b97ea77e2961c08945b2e034f646c5f7099361cd6267dc48.jpg)

![](images/5ca45b4259cd03f321bffa340ffe0b985ab85342b1c84a902542744087cbd46c.jpg)

![](images/ff69c78059356315f3d1d85c67b0c7e20831360305f75b2c2a4ff192c5088e99.jpg)

![](images/8d8b654afad2515fbd69fb74c2e07ac9451348f810a2409251367374ad008631.jpg)  
(c) Cheetah-90 Degree Push-up

![](images/91df29227388a1d934b50f1ed5e9bdff3d95992e914ae340471ef148f5bd7ee7.jpg)

![](images/a2c54c5cc3bfbb9fe1f0a8108c915ea2bd4849862c566b8f470431d082228cab.jpg)

![](images/97f4600f486549fd0e2c1a2f8bf933690b2eb74501ed29a03de5d5f2903a2a70.jpg)

![](images/33699cb90628e61aa750e8e6cbc5fa7a0fedf04c015d26af7a0b495394157bac.jpg)

![](images/3cb988d87ec16ab8a62107fb07590a6e351770db90991dd91f653b4120968f3a.jpg)  
(d) Walker-Dance   
Figure 4: Human desired behaviors behaved by the Decision Transformer under the attack of RAT.

To ensure a fair comparison, identical experimental settings (including hyper-parameters and neural networks) for reward learning are applied across all methods. We conduct a quantitative evaluation of all methods by comparing their average attack success rates. Comprehensive details on hyperparameter settings, implementation details, and scenario designs are available in Appendix E.

Evaluation Metrics. The success metric for adversarial attacks revolves around the proximity between the task-relevant object and its final goal position, denoted as $I_{\|o-t\|_{2}<\epsilon}$ , where $\epsilon$ is a minimal distance threshold. In the manipulation scenario, we set $\epsilon = 0.05$ (5cm). For the opposite behaviors scenario, we apply the success metrics and thresholds specified for each task by Meta-world (Yu et al. 2020). We summarize the all success metrics in our experiments in the Table 7 in the Appendix E.4.

# 5.2 Case I: Manipulation on DRL Agents

We first conduct an evaluation of our method and various other adversarial attack algorithms across two different scenarios, applying them to a range of simulated robotic manipulation tasks. Each victim agent is a well-trained SAC (Haarnoja et al. 2018) agent, specialized for a specific manipulation task and trained for $10^{6}$ timesteps using the open-source code \* available. Details on hyperparameter settings are provided in Appendix E.3.

Scenarios on Manipulation. In this scenario, our objective was to manipulate the victim (robotic arm) to grasp objects at locations distant from the originally intended target, rather than completing its initial task. Table 1 presents the average attack success rates of both baseline methods and our approach across four manipulation tasks. The results indicate that the performance of RAT significantly exceeds that of the baselines by a large margin. To reduce the influence of PbRL and further highlight the advantages of RAT, we also trained

baseline methods using the ground-truth reward function, labeling these as “oracle” versions. Notably, the performance of SA-RL (oracle) shows considerable improvement on several tasks compared to its preference-based counterpart. Nonetheless, RAT still outperformed SA-RL with oracle rewards in most scenarios. These findings underscore the ability of RAT to enable agents to effectively learn adversary based on human preferences. Additionally, it was observed that PA-AD struggles to perform effectively in manipulation tasks, even when trained with ground-truth rewards.

Scenarios on Opposite Behaviors. Robotic manipulation holds significant practical value in real-world applications. Therefore, we craft this scenario to quantitatively assess the vulnerability of agents proficient in various manipulation skills. In this setup, each victim agent is expected to perform the opposite of its mastered task when subjected to the manipulator's targeted attack. For instance, a victim trained to open windows would be manipulated to close them instead. As demonstrated in Table 1, RAT exhibits exceptional performance, consistently demonstrating clear advantages over the baseline methods across all tasks. This outcome reaffirms that RAT is not only effective across a broad spectrum of tasks but also capable of efficiently learning adversaries aligned with human preferences.

We observe that SA-RL and PA-AD exhibit relatively low attack success rates across numerous tasks, which can be attributed to the issue of distribution drift. This drift arises due to discrepancies between the data distribution sampled by the perturbed policy and the distribution corresponding to human-desired behaviors, leading to suboptimal performance.

# 5.3 Case II: Manipulation on Sequence Model Agents

In this experiment, we show the vulnerability of offline RL agents and the capability of RAT to fool them into acting human desired behaviors. As for the implementation, we choose some online models \* as victims, which are well-trained by official implementation with D4RL. We choose two tasks, Cheetah and Walker, using expert-level Decision Transformer agents as the victims. As illustrated in Figure 4, Decision Transformer reveals weaknesses that can be exploited, leading it to execute human-preferred behaviors rather than its intended tasks. Under adversarial manipulation, the Cheetah agent is shown to run backwards rapidly in Figure 4a and perform a 90-degree push-up in Figure 4c. Meanwhile, the Walker agent maintains superior balance on one foot in Figure 4b and appears to dance with one leg raised in Figure 4d. These outcomes indicate that RAT is effective in manipulating these victim agents towards behaviors consistent with human preferences, highlighting the significant vulnerability of embodied agents to strong adversaries. This experiment is expected to spur further research into improving the robustness of offline RL agents and embodied AI systems.

# 5.4 Robust Agents Training and Evaluation

A practical application of RAT is in assessing the robustness of established models or in enhancing an agent's robustness via adversarial training. ATLA-PPO (Zhang et al. 2021) presents a generic training framework aimed at improving robustness, which involves alternating training between an agent and an SA-RL attacker. PA-ATLA (Sun et al. 2022) follows a similar approach but employs a more advanced RL attacker, PA-AD. Drawing inspiration from previous works (Zhang et al. 2021; Liang et al. 2022b), we introduce two novel robust training methods: RAT-ATLA and RAT-WocaR. RAT-ATLA's central strategy is to alternately train an agent and a RAT attacker, whereas RAT-WocaR focuses on directly estimating and minimizing the reward of the intention policy, obviating the need for extra samples to learn an attacker. Table 2 compares the effectiveness of RAT-ATLA and RAT-WocaR for SAC agents on robotic simulation manipulation tasks against leading robust training methods. The experimental findings highlight two key points: first, RAT-ATLA and RAT-WocaR substantially improve agent robustness; and second, RAT is capable of executing stronger attacks on robust agents, showcasing its effectiveness in challenging environments.

Table 2: Average episode rewards $\pm$ standard deviation of robust agents under different attack methods, and results are averaged across 100 episodes. 

<table><tr><td>Task</td><td>Model</td><td>RAT</td><td>PA-AD</td><td>SA-RL</td><td>Avg R</td></tr><tr><td rowspan="4">Door Lock</td><td>RAT-ATLA</td><td> $874 \pm 444$ </td><td> $628 \pm 486$ </td><td> $503 \pm 120$ </td><td>668</td></tr><tr><td>RAT-WocaR</td><td> $774 \pm 241$ </td><td> $527 \pm 512$ </td><td> $520 \pm 236$ </td><td>607</td></tr><tr><td>PA-ATLA</td><td> $491 \pm 133$ </td><td> $483 \pm 15$ </td><td> $517 \pm 129$ </td><td>497</td></tr><tr><td>ATLA-PPO</td><td> $469 \pm 11$ </td><td> $629 \pm 455$ </td><td> $583 \pm 173$ </td><td>545</td></tr><tr><td rowspan="4">Door Unlock</td><td>RAT-ATLA</td><td> $477 \pm 203$ </td><td> $745 \pm 75$ </td><td> $623 \pm 60$ </td><td>615</td></tr><tr><td>RAT-WocaR</td><td> $525 \pm 78$ </td><td> $647 \pm 502$ </td><td> $506 \pm 39$ </td><td>559</td></tr><tr><td>PA-ATLA</td><td> $398 \pm 12$ </td><td> $381 \pm 11$ </td><td> $398 \pm 79$ </td><td>389</td></tr><tr><td>ATLA-PPO</td><td> $393 \pm 36$ </td><td> $377 \pm 8$ </td><td> $385 \pm 26$ </td><td>385</td></tr><tr><td rowspan="4">Faucet Open</td><td>RAT-ATLA</td><td> $442 \pm 167$ </td><td> $451 \pm 96$ </td><td> $504 \pm 55$ </td><td>465</td></tr><tr><td>RAT-WocaR</td><td> $1223 \pm 102$ </td><td> $1824 \pm 413$ </td><td> $1575 \pm 389$ </td><td>1541</td></tr><tr><td>PA-ATLA</td><td> $438 \pm 53$ </td><td> $588 \pm 222$ </td><td> $373 \pm 32$ </td><td>466</td></tr><tr><td>ATLA-PPO</td><td> $610 \pm 293$ </td><td> $523 \pm 137$ </td><td> $495 \pm 305$ </td><td>522</td></tr><tr><td rowspan="4">Faucet Close</td><td>RAT-ATLA</td><td> $1048 \pm 343$ </td><td> $1223 \pm 348$ </td><td> $570 \pm 453$ </td><td>947</td></tr><tr><td>RAT-WocaR</td><td> $1369 \pm 158$ </td><td> $1416 \pm 208$ </td><td> $3372 \pm 1311$ </td><td>2052</td></tr><tr><td>PA-ATLA</td><td> $661 \pm 279$ </td><td> $371 \pm 65$ </td><td> $704 \pm 239$ </td><td>538</td></tr><tr><td>ATLA-PPO</td><td> $1362 \pm 149$ </td><td> $688 \pm 196$ </td><td> $426 \pm 120$ </td><td>825</td></tr></table>

# 5.5 Ablation Studies

Contribution of Each Component. In our further experiments, we investigate the effect of each component in RAT on Drawer Open and Drawer Close for the manipulation scenario and on Faucet Open, Faucet Close for the opposite behavior scenario. RAT incorporates three essential components or techniques: the intention policy $\pi_{\theta}$ , the weighting function $h_{\omega}$ and the combined behavior policy. As detailed in Table 3, $\pi_{\theta}$ emerges as a pivotal component in RAT, significantly boosting the attack success rate. This enhancement is largely due to its capability to mitigate distribution drift between the victim's behavior and the desired behavior.

Effects of the Weighting Function. To further understand the weighting function proposed in Section 4.2, we conduct comprehensive experimental data analysis and visualization from multiple perspectives. We sample five perturbed policies

Table 3: Effects of each component in RAT is evaluated based on the average attack success rate on four simulated robotic manipulation tasks. These results represent the mean success rate across five runs. 

<table><tr><td>Scenario</td><td>Task</td><td>RAT</td><td> $\text{RAT w/o } h_{\omega}$ </td><td> $\text{RAT w/o } \pi_{\theta}$ </td><td> $\text{RAT w/o combined policy}$ </td></tr><tr><td rowspan="2">Manipulation</td><td>Drawer Open</td><td>99.1%</td><td>91.3%</td><td>21.7%</td><td>68.0%</td></tr><tr><td>Drawer Close</td><td>80.9%</td><td>70.2%</td><td>8.0%</td><td>26.0%</td></tr><tr><td rowspan="2">Opposite</td><td>Faucet Open</td><td>84.4%</td><td>89.8%</td><td>0.0%</td><td>57.0%</td></tr><tr><td>Faucet Close</td><td>95.1%</td><td>94.1%</td><td>13.0%</td><td>59.1%</td></tr></table>

![](images/93d3579d4a050745350f6bbe86bb09eaec52f97cc4bab699b6b285f26631b534.jpg)

<details>
<summary>scatter</summary>

| x    | y    | cluster |
| ---- | ---- | ------- |
| -80  | 20   | 1       |
| -60  | 40   | 2       |
| -40  | 60   | 3       |
| -20  | 80   | 4       |
| 0    | 100  | 5       |
| 20   | 80   | 6       |
| 40   | 60   | 7       |
| 60   | 40   | 8       |
| 80   | 20   | 9       |
| 100  | 0    | 10      |
| -70  | -20  | 1       |
| -50  | -40  | 2       |
| -30  | -60  | 3       |
| -10  | -80  | 4       |
| 10   | -60  | 5       |
| 30   | -40  | 6       |
| 50   | -20  | 7       |
| 70   | 0    | 8       |
| 90   | 20   | 9       |
| -90  | -25  | 1       |
| -70  | -50  | 2       |
| -50  | -75  | 3       |
| -30  | -90  | 4       |
| -10  | -75  | 5       |
| 10   | -60  | 6       |
| 30   | -40  | 7       |
| 50   | -20  | 8       |
| 70   | 0    | 9       |
| 90   | 20   | 10      |
| -85  | -35  | 1       |
| -65  | -65  | 2       |
| -45  | -95  | 3       |
| -25  | -85  | 4       |
| -5   | -75  | 5       |
| -15  | -65  | 6       |
| -35  | -55  | 7       |
| -55  | -45  | 8       |
| -75  | -35  | 9       |
| -95  | -25  | 10      |
| -85  | -15  | 11      |
| -65  | -35  | 12      |
| -45  | -55  | 13      |
| -25  | -75  | 14      |
| -5   | -95  | 15      |
| -15  | -85  | 16      |
| -35  | -75  | 17      |
| -55  | -65  | 18      |
| -75  | -55  | 19      |
| -95  | -45  | 20      |
| -85  | -35  |          |
| -75  | -25  |          |
| -65  | -15  |          |
| -55  | -35  |          |
| -45  | -55  |          |
| -25  | -75  |          |
| -5   | -95  |          |
| -15  | -85  |          |
| -35  | -75  |          |
| -55  | -65  |          |
| -75  | -55  |          |
| -95  | -45  |          |
| -85  | -35  |          |
| -75  | -25  |          |
| -65  | -15  |          |
| -55  | -35  |          |
| -45  | -55  |          |
| -25  | -75  |          |
| -5   | -95  |          |
| -15  | -8.5 |          |
| -3   | -7   |          |
| -1   | -5   |          |
| -1.5 | -3.5 |          |
| -3.5 | -2   |          |
| -6.5 | -1   |          |
| -9.5 | -3.5 |          |
| -8.5 | -6   |          |
| -7.5 | -8.5 |          |
| -6.5 | -1    |          |
| -4.5 | -3.5 |          |
| -2.5 | -6   |          |
| -4.5 | -8.5 |          |
| -6.5 | -1.21|          |
| -9.5 | -3.81|         |
| -8.5 | -6.41|        |
| -7.5 | -9.41|        |
| -6.5 | -1.61|        |
| -4.5 | -3.81|        |
| -2.5 | -6.41|        |
| -4.5 | -9.41|        |
| -6.5 | -2    |        |
| -9.5 |     nan   |         |
|           |      nan   |             |
The data is a scatter plot with 'x' and 'y' as the same variables for each cluster. The data points are grouped by cluster label '1' to '12'. The values in the scatter plot are explicitly labeled on the chart.
</details>

(a) t-SNE Visualization

![](images/88a2f74c6e2a6e24679e395391d72fcce24cf94517b5765acd5a2e0bd11fc62b.jpg)

<details>
<summary>bar</summary>

| Average Return | Normalized Weight |
| -------------- | ----------------- |
| 0              | -0.6              |
| 1              | -0.4              |
| 2              | 0.25              |
| 3              | 0.3               |
| 4              | 0.4               |
</details>

(b) Weight Visualization   
Figure 5: Effects of the Weighting Function. (a) Trajectory weights generated by the weighting function from various policies are visualized with t-SNE. (b) A visualization of the weights of trajectories of different qualities by five different policies.

uniformly, each representing a progressive stage of performance improvement before the convergence of RAT. For each of these policies, 100 trajectories were rolled out, and their corresponding trajectory weight vectors were obtained via the weighting function. Utilizing t-SNE (van der Maaten and Hinton 2008) for visualization, Figure 5a showcases the weight vectors of different policies. This illustration reveals distinct boundaries between the trajectory weights of various policies, indicating the weighting function's ability to differentiate trajectories based on their quality. In Figure 5b, trajectories with higher success rates in manipulation are represented in darker colors. The visualization suggests that the weighting function assigns higher weights to more successful trajectories, thereby facilitating the improvement of the adversary's performance.

To thoroughly assess the impact of feedback amounts and attack budgets on the performance of RAT, as well as the quality of the learned reward functions, we conducted extensive experiments. The detailed analyses and discussions of these aspects are provided in Appendix F.

# 6 Conclusion

In this paper, we propose RAT, a targeted behavior attack approach against DRL learners, which manipulates the victim to perform human-desired behaviors. RAT involves an adversary adding imperceptible perturbations on the observations of the victim, an intention policy learned through PbRL as a flexible behavior target, and a weighting function to identify essential states for the efficient adversarial attack. We analyze the convergence of RAT and prove that RAT converges to critical points under some mild conditions. Empirically, we design two scenarios on several manipulation tasks in Meta-world, and the results demonstrate that RAT outperforms the baselines in the targeted adversarial setting. Additionally, RAT can enhance the robustness of agents via adversarial training. We further show embodied agents' vulnerability by attacking Decision Transformer on some MuJoCo tasks.

# References

Bai, F.; Zhang, H.; Tao, T.; Wu, Z.; Wang, Y.; and Xu, B. 2023. PiCor: Multi-Task Deep Reinforcement Learning with Policy Correction. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 37(6): 6728–6736.   
Bai, F.; Zhao, R.; Zhang, H.; Cui, S.; Wen, Y.; Yang, Y.; Xu, B.; and Han, L. 2024. Efficient Preference-based Reinforcement Learning via Aligned Experience Estimation. arXiv preprint arXiv:2405.18688.   
Bradley, R. A.; and Terry, M. E. 1952. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika, 39(3/4): 324–345.   
Buddareddygari, P.; Zhang, T.; Yang, Y.; and Ren, Y. 2022. Targeted Attack on Deep RL-based Autonomous Driving with Learned Visual Patterns. In International Conference on Robotics and Automation (ICRA), 10571–10577.   
Chen, L.; Lu, K.; Rajeswaran, A.; Lee, K.; Grover, A.; Laskin, M.; Abbeel, P.; Srinivas, A.; and Mordatch, I. 2021. Decision Transformer: Reinforcement Learning via Sequence Modeling. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, 15084–15097. Curran Associates, Inc. Christiano, P. F.; Leike, J.; Brown, T.; Martic, M.; Legg, S.; and Amodei, D. 2017. Deep Reinforcement Learning from Human Preferences. In Advances in Neural Information Processing Systems (NeurIPS), volume 30. Curran Associates, Inc.   
Fischer, M.; Mirman, M.; Stalder, S.; and Vechev, M. 2019. Online robustness training for deep reinforcement learning. arXiv preprint arXiv:1911.00887.   
Gleave, A.; Dennis, M.; Wild, C.; Kant, N.; Levine, S.; and Russell, S. 2020. Adversarial Policies: Attacking Deep Reinforcement Learning. In International Conference on Learning Representations (ICLR).   
Goodfellow, I. J.; Shlens, J.; and Szegedy, C. 2015. Explaining and Harnessing Adversarial Examples. In International Conference on Learning Representations (ICLR).

Haarnoja, T.; Zhou, A.; Abbeel, P.; and Levine, S. 2018. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. In International Conference on Machine Learning (ICML), volume 80, 1861–1870.   
Hasanbeig, M.; Kroening, D.; and Abate, A. 2020. Deep Reinforcement Learning with Temporal Logics. In Formal Modeling and Analysis of Timed Systems (FORMATS), 1–22.   
Huang, S. H.; Papernot, N.; Goodfellow, I. J.; Duan, Y.; and Abbeel, P. 2017. Adversarial Attacks on Neural Network Policies. In International Conference on Learning Representations (ICLR).   
Hussenot, L.; Geist, M.; and Pietquin, O. 2019a. Targeted Attacks on Deep Reinforcement Learning Agents through Adversarial Observations. abs/1905.12282.   
Hussenot, L.; Geist, M.; and Pietquin, O. 2019b. Targeted Attacks on Deep Reinforcement Learning Agents through Adversarial Observations. CoRR, abs/1905.12282.   
Ibarz, B.; Leike, J.; Pohlen, T.; Irving, G.; Legg, S.; and Amodei, D. 2018. Reward learning from human preferences and demonstrations in Atari. In Advances in Neural Information Processing Systems (NeurIPS), volume 31. Curran Associates, Inc.   
Janner, M.; Fu, J.; Zhang, M.; and Levine, S. 2019. When to Trust Your Model: Model-Based Policy Optimization. In Advances in Neural Information Processing Systems (NeurIPS), volume 32.   
Jia, X.; Yang, Z.; Li, Q.; Zhang, Z.; and Yan, J. 2024. Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving. arXiv preprint arXiv:2406.03877.   
Jin, F.; Liu, Y.; and Tan, Y. 2024. Derivative-Free Optimization for Low-Rank Adaptation in Large Language Models. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32: 4607–4616.   
Jin, F.; Zhang, J.; and Zong, C. 2023. Parameter-efficient Tuning for Large Language Model without Calculating Its Gradients. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 321–330.   
LeCun, Y.; Bengio, Y.; and Hinton, G. 2015. Deep learning. Nature, 521(7553): 436–444.   
Lee, K.; Smith, L. M.; and Abbeel, P. 2021. PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training. In International Conference on Machine Learning (ICML), volume 139, 6152–6163.   
Liang, X.; Shu, K.; Lee, K.; and Abbeel, P. 2022a. Reward Uncertainty for Exploration in Preference-based Reinforcement Learning. In International Conference on Learning Representations (ICLR).   
Liang, Y.; Sun, Y.; Zheng, R.; and Huang, F. 2022b. Efficient Adversarial Training without Attacking: Worst-Case-Aware Robust Reinforcement Learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, 22547–22561.

Lin, Y.-C.; Hong, Z.-W.; Liao, Y.-H.; Shih, M.-L.; Liu, M.-Y.; and Sun, M. 2017a. Tactics of Adversarial Attack on Deep Reinforcement Learning Agents. In International Joint Conference on Artificial Intelligence (IJCAI), 3756–3762.   
Lin, Y.-C.; Hong, Z.-W.; Liao, Y.-H.; Shih, M.-L.; Liu, M.-Y.; and Sun, M. 2017b. Tactics of Adversarial Attack on Deep Reinforcement Learning Agents. In IJCAI, 3756–3762.   
Liu, R.; Bai, F.; Du, Y.; and Yang, Y. 2022. Meta-Reward-Net: Implicitly Differentiable Reward Learning for Preference-based Reinforcement Learning. In Oh, A. H.; Agarwal, A.; Belgrave, D.; and Cho, K., eds., Advances in Neural Information Processing Systems (NeurIPS).   
Liu, R.; Du, Y.; Bai, F.; Lyu, J.; and Li, X. 2024. PEARL: Zero-shot Cross-task Preference Alignment and Robust Reward Learning for Robotic Manipulation. In International Conference on Machine Learning (ICML).   
Lütjens, B.; Everett, M.; and How, J. P. 2020. Certified Adversarial Robustness for Deep Reinforcement Learning. In Conference on Robot Learning (CoRL), volume 100, 1328–1337.   
Mairal, J. 2013. Stochastic majorization-minimization algorithms for large-scale optimization. In Advances in Neural Information Processing Systems (NeurIPS), volume 26.   
Mandlekar, A.; Zhu, Y.; Garg, A.; Fei-Fei, L.; and Savarese, S. 2017. Adversarially Robust Policy Learning: Active construction of physically-plausible perturbations. In International Conference on Intelligent Robots and Systems (IROS), 3932–3939.   
Nesterov, Y. 1998. Introductory lectures on convex programming.   
Oikarinen, T.; Zhang, W.; Megretski, A.; Daniel, L.; and Weng, T.-W. 2021. Robust Deep Reinforcement Learning through Adversarial Loss. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, 26156–26167.   
Park, J.; Seo, Y.; Shin, J.; Lee, H.; Abbeel, P.; and Lee, K. 2022. SURF: Semi-supervised Reward Learning with Data Augmentation for Feedback-efficient Preference-based Reinforcement Learning. In International Conference on Learning Representations (ICLR).   
Pattanaik, A.; Tang, Z.; Liu, S.; Bommannan, G.; and Chowdhary, G. 2018. Robust Deep Reinforcement Learning with Adversarial Attacks. In International Conference on Autonomous Agents and MultiAgent Systems (AAMAS). International Foundation for Autonomous Agents and Multiagent Systems.   
Pinto, L.; Davidson, J.; Sukthankar, R.; and Gupta, A. 2017. Robust Adversarial Reinforcement Learning. In International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine Learning Research, 2817–2826. PMLR.   
Shen, Q.; Li, Y.; Jiang, H.; Wang, Z.; and Zhao, T. 2020. Deep Reinforcement Learning with Robust and Smooth Policy. In International Conference on Machine Learning (ICML), volume 119, 8707–8718.   
Shi, Y.; Wen, M.; Zhang, Q.; Zhang, W.; Liu, C.; and Liu, W. 2024. Autonomous Goal Detection and Cessation in

Reinforcement Learning: A Case Study on Source Term Estimation. arXiv preprint arXiv:2409.09541.   
Sun, Y.; Zheng, R.; Liang, Y.; and Huang, F. 2022. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL. In International Conference on Learning Representations (ICLR).   
Sutton, R. S.; and Barto, A. G. 2018. Reinforcement learning: An introduction. MIT press.   
Todorov, E.; Erez, T.; and Tassa, Y. 2012. MuJoCo: A physics engine for model-based control. In International Conference on Intelligent Robots and Systems (IROS), 5026–5033.   
Vamplew, P.; Smith, B. J.; Källström, J.; Ramos, G.; Rădulescu, R.; Roijers, D. M.; Hayes, C. F.; Heintz, F.; Mannion, P.; Libin, P. J.; et al. 2022. Scalar reward is not enough: A response to silver, singh, precup and sutton (2021). Autonomous Agents and Multi-Agent Systems (AAMAS), 36(2):41.   
van der Maaten, L.; and Hinton, G. 2008. Visualizing Data using t-SNE. Journal of Machine Learning Research, 9(86):2579–2605.   
Wang, X.; Tian, Z.; Wan, Z.; Wen, Y.; Wang, J.; and Zhang, W. 2023. Order Matters: Agent-by-agent Policy Optimization. 11th ICLR.   
Wang\*, X.; Zhang\*, S.; Zhang, W.; Dong, W.; Chen, J.; Wen, Y.; and Zhang, W. 2024. ZSC-Eval: An Evaluation Toolkit and Benchmark for Multi-agent Zero-shot Coordination. Advances in neural information processing systems (NeurIPS) Track on Datasets and Benchmarks.   
Wen, M.; Wan, Z.; Wang, J.; Zhang, W.; and Wen, Y. 2024. Reinforcing LLM Agents via Policy Optimization with Action Decomposition. In Advances in neural information processing systems (NeurIPS).   
Xu, Y.; and Yu, L. 2023. DRL-Based Trajectory Tracking for Motion-Related Modules in Autonomous Driving. arXiv preprint arXiv:2308.15991.   
Yu, T.; Quillen, D.; He, Z.; Julian, R.; Hausman, K.; Finn, C.; and Levine, S. 2020. Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning. In Conference on Robot Learning (CoRL), volume 100 of Proceedings of Machine Learning Research, 1094–1100. PMLR.   
Zhang, H.; Chen, H.; Boning, D. S.; and Hsieh, C.-J. 2021. Robust Reinforcement Learning on State Observations with Learned Optimal Adversary. In International Conference on Learning Representations (ICLR).   
Zhang, H.; Chen, H.; Xiao, C.; Li, B.; Liu, M.; Boning, D.; and Hsieh, C.-J. 2020. Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, 21024–21037. Curran Associates, Inc.   
Zhang, H.; Sun, K.; bo xu; Kong, L.; and Müller, M. 2024a. A Distance-based Anomaly Detection Framework for Deep Reinforcement Learning. Transactions on Machine Learning Research.   
Zhang, H.; Xiao, C.; Gao, C.; Wang, H.; bo xu; and Müller, M. 2024b. Exploiting the Replay Memory Before Exploring the

Environment: Enhancing Reinforcement Learning Through Empirical MDP Iteration. In Advances in Neural Information Processing Systems (NeurIPS).

Zhang, H.; Xiao, C.; Wang, H.; Jin, J.; bo xu; and Müller, M. 2023. Replay Memory as An Empirical MDP: Combining Conservative Estimation with Experience Replay. In International Conference on Learning Representations (ICLR).

Zhu, R.; Ma, Z.; Wu, J.; Gao, J.; Wang, J.; Lin, D.; and He, C. 2024a. Utilize the Flow before Stepping into the Same River Twice: Certainty Represented Knowledge Flow for Refusal-Aware Instruction Tuning. arXiv preprint arXiv:2410.06913.

Zhu, X.; Fu, Y.; Zhou, B.; and Lin, Z. 2024b. Critical data size of language models from a grokking perspective. arXiv preprint arXiv:2401.10463.

Zhu, X.; Qi, B.; Zhang, K.; Long, X.; Lin, Z.; and Zhou, B. 2024c. PaD: Program-aided Distillation Can Teach Small Models Reasoning Better than Chain-of-thought Fine-tuning. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2571–2597.

# A Full Procedure of RAT

In this section, we provide a detailed explanation of our method, including the pseudo code, the design of the Combined Behavior Policy, and the basic setup for preference-based reinforcement learning (PbRL). The procedures for our method are fully outlined in Algorithm 1, which describes the reward learning process and adversary updates in RAT.

Algorithm 1: RAT   
Input: a fixed victim policy $\pi_{\nu}$ , frequency of human feedback K, outer loss updating frequency M, task horizon H

1: Initialize parameters of $Q_{\phi}, \pi_{\theta}, \widehat{r}_{\psi}, \pi_{\alpha}$ and $h_{\omega}$ 2: Initialize B and $\pi_{\theta}$ with unsupervised exploration

3: Initialize preference data set $D \leftarrow \emptyset$ 4: for each iteration do

5: // Construct the combined behavior policy $\pi$ 6: if episode is done then

7: $h \sim U(0, H)$ 8: $\pi^{1:h} = \pi_{\nu \circ \alpha}^{1:h}$ and $\pi^{h+1:H} = \pi_{\theta}^{h+1:H}$ 9: end if

10: Take action $a_{t} \sim \pi$ and collect $s_{t+1}$ 11: Store transition into dataset $B \leftarrow B \cup \{(s_{t}, a_{t}, \widehat{r}_{\psi}(s_{t}), s_{t+1})\}$ 12: // Query preference and Reward learning

13: if iteration % K == 0 then

14: Sample pair of trajectories ( $\sigma^{0}, \sigma^{1}$ )

15: Query preference y from manipulator

16: Store preference data into dataset $D \leftarrow D \cup \{(\sigma^{0}, \sigma^{1}, y)\}$ 17: Sample batch $\{(\sigma^{0}, \sigma^{1}, y)_{i}\}_{i=1}^{n}$ from D

18: Optimize (2) to update $\widehat{r}_{\psi}$ 19: end if

20: // Inner loss optimization

21: for each gradient step do

22: Sample random mini-batch transitions from B

23: Optimize $\pi_{\alpha}$ : minimize (6) with respect to $\alpha$ 24: end for

25: // Outer loss optimization

26: if iteration % M == 0 then

27: Sample random mini-batch transitions from B

28: Optimize $h_{\omega}$ : minimize (7) with respect to $\omega$ 29: end if

30: // Intention policy learning

31: Update $Q_{\phi}$ and $\pi_{\theta}$ according to (3) and (4), respectively.

32: end for

Output: adversary $\pi_{\alpha}$

# A.1 The Combined Behavior Policy

To address the inefficiencies caused by the distribution discrepancy between the learned policy $\pi_{\theta}$ and the perturbed policy $\pi_{\nu\circ\alpha}$ , we developed a behavior policy $\pi$ for data collection inspired by Branched Rollout (Janner et al. 2019). Our approach combines the intention policy $\pi_{\theta}$ with the perturbed policy $\pi_{\nu\circ\alpha}$ to balance exploration and exploitation during data collection. Specifically, we define the behavior policy $\pi$ as a combination of $\pi_{\nu\circ\alpha}$ and $\pi_{\theta}$ , where $\pi^{1:h} = \pi_{\nu\circ\alpha}^{1:h}$ and $\pi^{h+1:H} = \pi_{\theta}^{h+1:H}$ . Here, h is sampled from a uniform distribution $U(0,H)$ , where H represents the task horizon. This combined policy is used to collect data, which is then stored in the replay buffer for training. By varying the point h at which the policy switches from $\pi_{\nu\circ\alpha}$ to $\pi_{\theta}$ , we maintain a balance between exploration of new behaviors and reinforcement of learned behaviors, effectively mitigating the distribution discrepancy.

# A.2 Details of PbRL

In this section, we present details of the scripted teacher and the preference collection process, both of which are crucial components of PbRL. All methods in our paper follow the reward learning settings outlined in Lee, Smith, and Abbeel (2021). Scripted Teacher. To systematically evaluate the performance of our methods, we utilize a scripted teacher that provides preferences between pairs of trajectory segments based on the oracle reward function for online settings, like prior methods (Lee, Smith, and Abbeel 2021; Park et al. 2022; Liu et al. 2022). While leveraging human preference labels would be ideal, it is often impractical for quick and quantitative evaluations. The scripted teacher approximates human intentions by mapping states s and

actions a to ground truth rewards, providing immediate feedback. This function is designed to simulate the decision-making process of a human teacher by approximating their preferences based on cumulative rewards.

Preference Collection. During training, we query the scripted teacher for preference labels at regular intervals. A batch of segment pairs is sampled, and the cumulative rewards for each segment are calculated based on the rewards provided by the scripted teacher. The segment with the higher cumulative reward is assigned a label of 1, while the other is labeled 0. The computational cost of this process is proportional to the number of preference labels M and the segment length N, resulting in a time complexity of $\mathcal{O}(MN)$ . However, this cost is negligible compared to adversary training, which involves more computationally expensive gradient calculations.

# B Derivation of the Gradient of the Outer-level Loss

In this section, we present detailed derivation of the gradient of the outer loss $J_{\pi}$ with respect to the parameters of the weighting function $\omega$ . According to the chain rule, we can derive that

$$
\begin{array}{l} \nabla_ {\omega} J _ {\pi} (\hat {\alpha} (\omega)) | _ {\omega_ {t}} \\ = \frac {\partial J _ {\pi} (\hat {\alpha} (\omega))}{\partial \hat {\alpha} (\omega)} \Big | _ {\hat {\alpha} _ {t}} \frac {\partial \hat {\alpha} _ {t} (\omega)}{\partial \omega} \Big | _ {\omega_ {t}} \\ = \frac {\partial J _ {\pi} (\hat {\alpha} (\omega))}{\partial \hat {\alpha} (\omega)} \left| _ {\hat {\alpha} _ {t}} \frac {\partial \hat {\alpha} _ {t} (\omega)}{\partial h (\mathbf {s} ; \omega)} \right| _ {\omega_ {t}} \frac {\partial h (\mathbf {s} ; \omega)}{\partial \omega} \Big | _ {\omega_ {t}} \tag {12} \\ = - \left. \eta_ {t} \frac {\partial J _ {\pi} (\hat {\alpha} (\omega))}{\partial \hat {\alpha} (\omega)} \right| _ {\hat {\alpha} _ {t}} \sum_ {\mathbf {s} \sim \mathcal {B}} \frac {\partial D _ {\mathrm{KL}} \left(\pi_ {\nu \circ \alpha} (\mathbf {s}) \parallel \pi_ {\theta} (\mathbf {s})\right)}{\partial \alpha} \Big | _ {\alpha_ {t}} \frac {\partial h (\mathbf {s} ; \omega)}{\partial \omega} \Big | _ {\omega_ {t}} \\ = - \left. \right. \eta_ {t} \sum_ {\mathbf {s} \sim \mathcal {B}} \left( \right.\frac {\partial J _ {\pi} (\hat {\alpha} (\omega))}{\partial \hat {\alpha} (\omega)} \left. \right| _ {\hat {\alpha} _ {t}} ^ {\top} \frac {\partial D _ {\mathrm{KL}} (\pi_ {\nu \circ \alpha} (\mathbf {s}) \| \pi_ {\theta} (\mathbf {s}))}{\partial \alpha} \left. \right| _ {\alpha_ {t}}\left. \right)\left. \frac {\partial h (\mathbf {s} ; \omega)}{\partial \omega} \right| _ {\omega_ {t}}. \\ \end{array}
$$

For brevity of expression, we let:

$$
f (\mathbf {s}) = \frac {\partial J _ {\pi} (\hat {\alpha} (\omega))}{\partial \hat {\alpha} (\omega)} \bigg | _ {\hat {\alpha} _ {t}} ^ {\top} \frac {\partial D _ {\mathrm{KL}} \left(\pi_ {\nu \circ \alpha} (\mathbf {s}) \| \pi_ {\theta} (\mathbf {s})\right)}{\partial \hat {\alpha}} \bigg | _ {\alpha_ {t}}. \tag {13}
$$

The gradient of outer-level optimization loss with respect to parameters $\omega$ is:

$$
\left. \nabla_ {\omega} J _ {\pi} (\hat {\alpha} (\omega)) \right| _ {\omega_ {t}} = - \eta_ {t} \sum_ {\mathbf {s} \sim \mathcal {B}} f (\mathbf {s}) \cdot \left. \frac {\partial h (\mathbf {s} ; \omega)}{\partial \omega} \right| _ {\omega_ {t}}. \tag {14}
$$

# C Connection between RSA-MDP and MDP

Lemma C.1. Given a RSA-MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{B}, \widehat{\mathcal{R}}, \mathcal{P}, \gamma)$ and a fixed victim policy $\pi_{\nu}$ , there exists a MDP $\hat{\mathcal{M}} = (\mathcal{S}, \hat{\mathcal{A}}, \widehat{\mathcal{R}}, \widehat{\mathcal{P}}, \gamma)$ such that the optimal policy of $\hat{M}$ is equivalent to the optimal adversary $\pi_{\alpha}$ in RSA-MDP given a fixed victim, where $\hat{A} = S$ and

$$
\widehat {\mathcal {P}} (\mathbf {s} ^ {\prime} | \mathbf {s}, \mathbf {a}) = \sum_ {\mathbf {a} \in \mathcal {A}} \pi_ {\nu} (\mathbf {a} | \widehat {\mathbf {a}}) \mathcal {P} (\mathbf {s} ^ {\prime} | \mathbf {s}, \mathbf {a}) \quad f o r \mathbf {s}, \mathbf {s} ^ {\prime} \in \mathcal {S} a n d \widehat {\mathbf {a}} \in \widehat {\mathcal {A}}.
$$

# D Theoretical Analysis and Proofs

# D.1 Theorem 1: Convergence Rate of the Outer Loss

Lemma D.1. (Lemma 1.2.3 in Nesterov (1998)) If function $f(x)$ is Lipschitz smooth on $\mathbb{R}^n$ with constant $L$ , then $\forall x, y \in \mathbb{R}^n$ , we have

$$
\left| f (y) - f (x) - f ^ {\prime} (x) ^ {\top} (y - x) \right| \leq \frac {L}{2} \| y - x \| ^ {2}. \tag {15}
$$

Proof. $\forall x,y\in \mathbb{R}^n$ , we have

$$
\begin{array}{l} f (y) = f (x) + \int_ {0} ^ {1} f ^ {\prime} (x + \tau (y - x)) ^ {\top} (y - x) d \tau \tag {16} \\ = f (x) + f ^ {\prime} (x) ^ {\top} (y - x) + \int_ {0} ^ {1} [ f ^ {\prime} (x + \tau (y - x)) - f ^ {\prime} (x) ] ^ {\top} (y - x) d \tau . \\ \end{array}
$$

Then we can derive that

$$
\begin{array}{l} \left| f (y) - f (x) - f ^ {\prime} (x) ^ {\top} (y - x) \right| = \left| \int_ {0} ^ {1} \left[ f ^ {\prime} (x + \tau (y - x)) - f ^ {\prime} (x) \right] ^ {\top} (y - x) d \tau \right| \\ \leq \int_ {0} ^ {1} \left| \left[ f ^ {\prime} (x + \tau (y - x)) - f ^ {\prime} (x) \right] ^ {\top} (y - x) \right| d \tau \tag {17} \\ \leq \int_ {0} ^ {1} \| f ^ {\prime} (x + \tau (y - x)) - f ^ {\prime} (x) \| \cdot \| y - x \| d \tau \\ \leq \int_ {0} ^ {1} \tau L \left\| y - x \right\| ^ {2} d \tau = \frac {L}{2} \left\| y - x \right\| ^ {2}, \\ \end{array}
$$

where the first inequality holds for $\left|\int_{a}^{b}f(x)dx\right|\leq\int_{a}^{b}|f(x)|dx$ , the second inequality holds for Cauchy-Schwarz inequality, and the last inequality holds for the definition of Lipschitz smoothness.

Theorem D.2. Suppose $J_{\pi}$ is Lipschitz-smooth with constant $L$ , the gradient of $J_{\pi}$ and $\mathcal{L}$ is bounded by $\rho$ . Let the training iterations be $T$ , the inner-level optimization learning rate $\eta_t = \min \{1, \frac{c_1}{T}\}$ for some constant $c_1 > 0$ where $\frac{c_1}{T} < 1$ . Let the outer-level optimization learning rate $\beta_t = \min \{\frac{1}{L}, \frac{c_2}{\sqrt{T}}\}$ for some constant $c_2 > 0$ where $c_2 \leq \frac{\sqrt{T}}{L}$ , and $\sum_{t=1}^{\infty} \beta_t \leq \infty, \sum_{t=1}^{\infty} \beta_t^2 \leq \infty$ . The convergence rate of $J_{\pi}$ achieves

$$
\min _ {1 \leq t \leq T} \mathbb {E} \left[ \| \nabla_ {\omega} J _ {\pi} (\alpha_ {t + 1} (\omega_ {t})) \| ^ {2} \right] \leq \mathcal {O} \left(\frac {1}{\sqrt {T}}\right). \tag {18}
$$

Proof. First,

$$
\begin{array}{l} J _ {\pi} (\hat {\alpha} _ {t + 2} (\omega_ {t + 1})) - J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) \\ \begin{array}{l} \left. \delta_ {\pi} \left(\alpha_ {t + 2} \left(\omega_ {t + 1}\right)\right) - \delta_ {\pi} \left(\alpha_ {t + 1} \left(\omega_ {t}\right)\right) \right. \\ = \left\{J _ {\pi} \left(\hat {\alpha} _ {t + 2} \left(\omega_ {t + 1}\right)\right) - J _ {\pi} \left(\hat {\alpha} _ {t + 1} \left(\omega_ {t + 1}\right)\right) \right\} + \left\{J _ {\pi} \left(\hat {\alpha} _ {t + 1} \left(\omega_ {t + 1}\right)\right) - J _ {\pi} \left(\hat {\alpha} _ {t + 1} \left(\omega_ {t}\right)\right) \right\}. \end{array} \tag {19} \\ \end{array}
$$

Then we separately derive the two terms of (19). For the first term,

$$
\begin{array}{l} J _ {\pi} (\hat {\alpha} _ {t + 2} (\omega_ {t + 1})) - J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t + 1})) \\ \leq \nabla_ {\hat {\alpha}} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t + 1})) ^ {\top} (\hat {\alpha} _ {t + 2} (\omega_ {t + 1}) - \hat {\alpha} _ {t + 1} (\omega_ {t + 1})) + \frac {L}{2} \left\| \hat {\alpha} _ {t + 2} (\omega_ {t + 1}) - \hat {\alpha} _ {t + 1} (\omega_ {t + 1}) \right\| ^ {2} \\ \leq \left\| \nabla_ {\hat {\alpha}} J _ {\pi} \left(\hat {\alpha} _ {t + 1} \left(\omega_ {t + 1}\right)\right) \right\| \cdot \left\| \hat {\alpha} _ {t + 2} \left(\omega_ {t + 1}\right) - \hat {\alpha} _ {t + 1} \left(\omega_ {t + 1}\right) \right\| + \frac {L}{2} \left\| \hat {\alpha} _ {t + 2} \left(\omega_ {t + 1}\right) - \hat {\alpha} _ {t + 1} \left(\omega_ {t + 1}\right) \right\| ^ {2} \tag {20} \\ \leq \rho \cdot \| - \eta_ {t + 1} \nabla_ {\hat {\alpha}} \mathcal {L} (\hat {\alpha} _ {t + 1}) \| + \frac {L}{2} \| - \eta_ {t + 1} \nabla_ {\hat {\alpha}} \mathcal {L} (\hat {\alpha} _ {t + 1}) \| ^ {2} \\ \leq \eta_ {t + 1} \rho^ {2} + \frac {L}{2} \eta_ {t + 1} ^ {2} \rho^ {2}, \\ \end{array}
$$

where $\hat{\alpha}_{t+2}(\omega_{t+1}) - \hat{\alpha}_{t+1}(\omega_{t+1}) = -\eta_{t+1}\nabla_{\hat{\alpha}}\mathcal{L}(\hat{\alpha}_{t+1})$ , the first inequality holds for Lemma D.1, the second inequality holds for Cauchy-Schwarz inequality, the third inequality holds for $\|\nabla_{\hat{\alpha}}J_{\pi}(\hat{\alpha}_{t+1}(\omega_{t+1}))\| \leq \rho$ , and the last inequality holds for $\|\nabla_{\hat{\alpha}}\mathcal{L}(\hat{\alpha}_{t+1})\| \leq \rho$ . It can be proved that the gradient of $\omega$ with respect to $J_{\pi}$ is Lipschitz continuous and we assume the Lipschitz constant is L. Therefore, for the second term,

$$
\begin{array}{l} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t + 1})) - J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) \\ \leq \nabla_ {\omega} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) ^ {\top} (\omega_ {t + 1} - \omega_ {t}) + \frac {L}{2} \left\| \omega_ {t + 1} - \omega_ {t} \right\| ^ {2} \\ = - \beta_ {t} \nabla_ {\omega} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) ^ {\top} \nabla_ {\omega} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) + \frac {L \beta_ {t} ^ {2}}{2} \| \nabla_ {\omega} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) \| ^ {2} \tag {21} \\ = - \left(\beta_ {t} - \frac {L \beta_ {t} ^ {2}}{2}\right) \left\| \nabla_ {\omega} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) \right\| ^ {2}, \\ \end{array}
$$

where $\omega_{t + 1} - \omega_t = -\beta_t\nabla_\omega J_\pi (\hat{\alpha}_{t + 1}(\omega_t))$ , and the first inequality holds for Lemma D.1. Therefore, (19) becomes

$$
J _ {\pi} (\hat {\alpha} _ {t + 2} (\omega_ {t + 1})) - J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) \leq \eta_ {t + 1} \rho^ {2} + \frac {L}{2} \eta_ {t + 1} ^ {2} \rho^ {2} - (\beta_ {t} - \frac {L \beta_ {t} ^ {2}}{2}) \| \nabla_ {\omega} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) \| ^ {2}. \tag {22}
$$

Rearranging the terms of (22), we obtain

$$
\left(\beta_ {t} - \frac {L \beta_ {t} ^ {2}}{2}\right) \| \nabla_ {\omega} J _ {\pi} \left(\hat {\alpha} _ {t + 1} \left(\omega_ {t}\right)\right) \| ^ {2} \leq J _ {\pi} \left(\hat {\alpha} _ {t + 1} \left(\omega_ {t}\right)\right) - J _ {\pi} \left(\hat {\alpha} _ {t + 2} \left(\omega_ {t + 1}\right)\right) + \eta_ {t + 1} \rho^ {2} + \frac {L}{2} \eta_ {t + 1} ^ {2} \rho^ {2}. \tag {23}
$$

Then, we sum up both sides of (23),

$$
\sum_ {t = 1} ^ {T} (\beta_ {t} - \frac {L \beta_ {t} ^ {2}}{2}) \left\| \nabla_ {\omega} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) \right\| ^ {2}
$$

$$
\leq J _ {\pi} \left(\hat {\alpha} _ {2} \left(\omega_ {1}\right)\right) - J _ {\pi} \left(\hat {\alpha} _ {T + 2} \left(\omega_ {T + 1}\right)\right) + \sum_ {t = 1} ^ {T} \left(\eta_ {t + 1} \rho^ {2} + \frac {L}{2} \eta_ {t + 1} ^ {2} \rho^ {2}\right) \tag {24}
$$

$$
\leq J _ {\pi} (\hat {\alpha} _ {2} (\omega_ {1})) + \sum_ {t = 1} ^ {T} (\eta_ {t + 1} \rho^ {2} + \frac {L}{2} \eta_ {t + 1} ^ {2} \rho^ {2}).
$$

Therefore,

$$
\min _ {1 \leq t \leq T} \mathbb {E} \left[ \left\| \nabla_ {\omega} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) \right\| ^ {2} \right]
$$

$$
\leq \frac {\sum_ {t = 1} ^ {T} (\beta_ {t} - \frac {L \beta_ {t} ^ {2}}{2}) \left\| \nabla_ {\omega} J _ {\pi} (\hat {\alpha} _ {t + 1} (\omega_ {t})) \right\| ^ {2}}{\sum_ {t = 1} ^ {T} (\beta_ {t} - \frac {L \beta_ {t} ^ {2}}{2})}
$$

$$
\leq \frac {1}{\sum_ {t = 1} ^ {T} (2 \beta_ {t} - L \beta_ {t} ^ {2})} \left[ 2 J _ {\pi} (\hat {\alpha} _ {2} (\omega_ {1})) + \sum_ {t = 1} ^ {T} (2 \eta_ {t + 1} \rho^ {2} + L \eta_ {t + 1} ^ {2} \rho^ {2}) \right]
$$

$$
\leq \frac {1}{\sum_ {t = 1} ^ {T} \beta_ {t}} \left[ 2 J _ {\pi} \left(\hat {\alpha} _ {2} \left(\omega_ {1}\right)\right) + \sum_ {t = 1} ^ {T} \eta_ {t + 1} \rho^ {2} \left(2 + L \eta_ {t + 1}\right) \right] \tag {25}
$$

$$
\leq \frac {1}{T \beta_ {t}} \left[ 2 J _ {\pi} \left(\hat {\alpha} _ {2} \left(\omega_ {1}\right)\right) + T \eta_ {t + 1} \rho^ {2} (2 + L) \right] \tag {25}
$$

$$
= \frac {2 J _ {\pi} (\hat {\alpha} _ {2} (\omega_ {1}))}{T \beta_ {t}} + \frac {\eta_ {t + 1} \rho^ {2} (2 + L)}{\beta_ {t}}
$$

$$
= \frac {2 J _ {\pi} \left(\hat {\alpha} _ {2} \left(\omega_ {1}\right)\right)}{T} \max \left\{L, \frac {\sqrt {T}}{c _ {2}} \right\} + \min \left\{1, \frac {c _ {1}}{T} \right\} \max \left\{L, \frac {\sqrt {T}}{c _ {2}} \right\} \rho^ {2} (2 + L)
$$

$$
\leq \frac {2 J _ {\pi} (\hat {\alpha} _ {2} (\omega_ {1}))}{c _ {2} \sqrt {T}} + \frac {c _ {1} \rho^ {2} (2 + L)}{c _ {2} \sqrt {T}}
$$

$$
= \mathcal {O} \left(\frac {1}{\sqrt {T}}\right),
$$

where the second inequality holds according to (24), the third inequality holds for $\sum_{t=1}^{T}\left(2\beta_t - L\beta_t^2\right) \geq \sum_{t=1}^{T}\beta_t$ .

# D.2 Theorem 2: Convergence of the Inner Loss

Lemma D.3. (Lemma A.5 in Mairal (2013)) Let $(a_{n})_{n\geq 1},(b_{n})_{n\geq 1}$ be two non-negative real sequences such that the series $\sum_{n = 1}^{\infty}a_n$ diverges, the series $\sum_{n = 1}^{\infty}a_nb_{n}$ converges, and there exists $C > 0$ such that $|b_{n + 1} - b_n|\leq Ca_n$ . Then, the sequence $(b_{n})_{n\geq 1}$ converges to 0.

Theorem D.4. Suppose $J_{\pi}$ is Lipschitz-smooth with constant $L$ , the gradient of $J_{\pi}$ and $\mathcal{L}$ is bounded by $\rho$ . Let the training iterations be $T$ , the inner-level optimization learning rate $\eta_t = \min \{1, \frac{c_1}{T}\}$ for some constant $c_1 > 0$ where $\frac{c_1}{T} < 1$ . Let the outer-level optimization learning rate $\beta_t = \min \{\frac{1}{L}, \frac{c_2}{\sqrt{T}}\}$ for some constant $c_2 > 0$ where $c_2 \leq \frac{\sqrt{T}}{L}$ , and $\sum_{t=1}^{\infty} \beta_t \leq \infty, \sum_{t=1}^{\infty} \beta_t^2 \leq \infty$ . $\mathcal{L}$ achieves

$$
\lim _ {t \rightarrow \infty} \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right) \| ^ {2} \right] = 0. \tag {26}
$$

Proof. First,

$$
\mathcal {L} (\alpha_ {t + 1}; \omega_ {t + 1}) - \mathcal {L} (\alpha_ {t}; \omega_ {t})
$$

$$
\begin{array}{l} \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t + 1}\right) - \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right) \\ = \left\{\mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t + 1}\right) - \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t}\right) \right\} + \left\{\mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t}\right) - \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right) \right\}. \end{array} \tag {27}
$$

For the first term in (27),

$$
\mathcal {L} (\alpha_ {t + 1}; \omega_ {t + 1}) - \mathcal {L} (\alpha_ {t + 1}; \omega_ {t})
$$

$$
\leq \nabla_ {\omega} \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t}\right) ^ {\top} \left(\omega_ {t + 1} - \omega_ {t}\right) + \frac {L}{2} \left\| \omega_ {t + 1} - \omega_ {t} \right\| ^ {2} \tag {28}
$$

$$
= - \beta_ {t} \nabla_ {\omega} \mathcal {L} (\alpha_ {t + 1}; \omega_ {t}) ^ {\top} \nabla_ {\omega} J _ {\pi} (\alpha_ {t + 1} (\omega_ {t})) + \frac {L \beta_ {t} ^ {2}}{2} \left\| \nabla_ {\omega} J _ {\pi} (\alpha_ {t + 1} (\omega_ {t})) \right\| ^ {2}.
$$

where $\omega_{t+1}-\omega_{t}=-\beta_{t}\nabla_{\omega}J_{\pi}(\alpha_{t+1}(\omega_{t}))$ , and the first inequality holds according to Lemma D.1. For the second term in (27),

$$
\mathcal {L} (\alpha_ {t + 1}; \omega_ {t}) - \mathcal {L} (\alpha_ {t}; \omega_ {t})
$$

$$
\leq \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) ^ {\top} (\alpha_ {t + 1} - \alpha_ {t}) + \frac {L}{2} \| \alpha_ {t + 1} - \alpha_ {t} \| ^ {2}
$$

$$
= - \eta_ {t} \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) ^ {\top} \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) + \frac {L \eta_ {t} ^ {2}}{2} \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| ^ {2} \tag {29}
$$

$$
= - \left(\eta_ {t} - \frac {L \eta_ {t} ^ {2}}{2}\right) \left\| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \right\| ^ {2}.
$$

where $\alpha_{t+1}-\alpha_{t}=-\eta_{t}\nabla_{\alpha}\mathcal{L}(\alpha_{t};\omega_{t})$ , and the first inequality holds according to Lemma (D.1). Therefore, (27) becomes

$$
\mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t + 1}\right) - \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right)
$$

$$
\leq - \beta_ {t} \nabla_ {\omega} \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t}\right) ^ {\top} \nabla_ {\omega} J _ {\pi} \left(\alpha_ {t + 1} \left(\omega_ {t}\right)\right) + \frac {L \beta_ {t} ^ {2}}{2} \| \nabla_ {\omega} J _ {\pi} \left(\alpha_ {t + 1} \left(\omega_ {t}\right)\right) \| ^ {2} \tag {30}
$$

$$
- \left(\eta_ {t} - \frac {L \eta_ {t} ^ {2}}{2}\right) \left\| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \right\| ^ {2}.
$$

Taking expectation of both sides of (30) and rearranging the terms, we obtain

$$
\eta_ {t} \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| ^ {2} \right] + \beta_ {t} \mathbb {E} \left[ \| \nabla_ {\omega} \mathcal {L} (\alpha_ {t + 1}; \omega_ {t}) \| \cdot \| \nabla_ {\omega} J _ {\pi} (\alpha_ {t + 1} (\omega_ {t})) \| \right]
$$

$$
\leq \mathbb {E} \left[ \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right) \right] - \mathbb {E} \left[ \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t + 1}\right) \right] + \frac {L \beta_ {t} ^ {2}}{2} \mathbb {E} \left[ \| \nabla_ {\omega} J _ {\pi} \left(\alpha_ {t + 1} \left(\omega_ {t}\right)\right) \| ^ {2} \right] \tag {31}
$$

$$
+ \frac {L \eta_ {t} ^ {2}}{2} \mathbb {E} \left[ \left\| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \right\| ^ {2} \right].
$$

Summing up both sides of (31) from $t = 1$ to $\infty$ ,

$$
\sum_ {t = 1} ^ {\infty} \eta_ {t} \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| ^ {2} \right] + \sum_ {t = 1} ^ {\infty} \beta_ {t} \mathbb {E} \left[ \| \nabla_ {\omega} \mathcal {L} (\alpha_ {t + 1}; \omega_ {t}) \| \cdot \| \nabla_ {\omega} J _ {\pi} (\alpha_ {t + 1} (\omega_ {t})) \| \right]
$$

$$
\leq \mathbb {E} \left[ \mathcal {L} \left(\alpha_ {1}; \omega_ {1}\right)\right] - \lim _ {t \rightarrow \infty} \mathbb {E} \left[ \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t + 1}\right)\right] + \sum_ {t = 1} ^ {\infty} \frac {L \beta_ {t} ^ {2}}{2} \mathbb {E} \left[ \| \nabla_ {\omega} J _ {\pi} \left(\alpha_ {t + 1} \left(\omega_ {t}\right)\right) \| ^ {2} \right] \tag {32}
$$

$$
+ \sum_ {t = 1} ^ {\infty} \frac {L \eta_ {t} ^ {2}}{2} \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| ^ {2} \right]
$$

$$
\leq \sum_ {t = 1} ^ {\infty} \frac {L (\eta_ {t} ^ {2} + \beta_ {t} ^ {2}) \rho^ {2}}{2} + \mathbb {E} \left[ \mathcal {L} (\alpha_ {1}; \omega_ {1}) \right] \leq \infty ,
$$

where the second inequality holds for $\sum_{t=1}^{\infty} \eta_t^2 \leq \infty$ , $\sum_{t=1}^{\infty} \beta_t^2 \leq \infty$ , $\|\nabla_\alpha \mathcal{L}(\alpha_t; \omega_t)\| \leq \rho$ , $\|\nabla_\omega J_\pi (\alpha_{t+1}(\omega_t))\| \leq \rho$ . Since

$$
\sum_ {t = 1} ^ {\infty} \beta_ {t} \mathbb {E} \left[ \| \nabla_ {\omega} \mathcal {L} (\alpha_ {t + 1}; \omega_ {t}) \| \cdot \| \nabla_ {\omega} J _ {\pi} (\alpha_ {t + 1} (\omega_ {t})) \| \right] \leq L \rho \sum_ {t = 1} ^ {\infty} \beta_ {t} \leq \infty . \tag {33}
$$

Therefore, we have

$$
\sum_ {t = 1} ^ {\infty} \eta_ {t} \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| ^ {2} \right] <   \infty . \tag {34}
$$

Since $|(\| a\| + \| b\|)(\| a\| - \| b\|)| \leq \| a + b\| \| a - b\|$ , we can derive that

$$
\begin{array}{l} \left| \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t + 1}; \omega_ {t + 1}) \| ^ {2} \right] - \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| ^ {2} \right] \right| \\ = \left| \mathbb {E} \left[ \left(\| \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t + 1}\right) \| + \| \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right) \|\right) + \left(\| \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t + 1}\right) \| - \| \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right) \|\right) \right] \right| \\ \leq \mathbb {E} \Big [ \Big | \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t + 1}; \omega_ {t + 1}) \| + \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| \Big | \Big | \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t + 1}; \omega_ {t + 1}) \| - \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| \Big | \Big ] \\ \leq \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t + 1}; \omega_ {t + 1}) + \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| \cdot \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t + 1}; \omega_ {t + 1}) - \nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}) \| \right] \\ \leq \mathbb {E} \left[ \left(\| \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t + 1}\right) \| + \| \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right) \|\right) \| \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t + 1}; \omega_ {t + 1}\right) - \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right) \| \right] \tag {35} \\ \leq 2 L \rho \mathbb {E} \Big [ \left\| (\alpha_ {t + 1}, \omega_ {t + 1}) - (\alpha_ {t}, \omega_ {t}) \right\| \Big ] \\ \leq 2 L \rho \eta_ {t} \beta_ {t} \mathbb {E} \left[ \| (\nabla_ {\alpha} \mathcal {L} (\alpha_ {t}; \omega_ {t}), \nabla_ {\omega} J _ {\pi} (\alpha_ {t + 1} (\omega_ {t}))) \| \right] \\ \leq 2 L \rho \eta_ {t} \beta_ {t} \sqrt {\mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} (\alpha_ {t} ; \omega_ {t}) \| ^ {2} \right] + \mathbb {E} \left[ \| \nabla_ {\omega} J _ {\pi} (\alpha_ {t + 1} (\omega_ {t})) \| ^ {2} \right]} \\ \leq 2 L \rho \eta_ {t} \beta_ {t} \sqrt {2 \rho^ {2}} \\ \leq 2 \sqrt {2} L \rho^ {2} \eta_ {t} \beta_ {t}. \\ \end{array}
$$

Since $\sum_{t=1}^{\infty} \eta_t = \infty$ , according to Lemma D.3, we have

$$
\lim _ {t \rightarrow \infty} \mathbb {E} \left[ \| \nabla_ {\alpha} \mathcal {L} \left(\alpha_ {t}; \omega_ {t}\right) \| ^ {2} \right] = 0. \tag {36}
$$

![](images/3d8e0ca50ca471b57930712fa8ef25be25091a9fbfde4d217d43c74f42e75545.jpg)

# E Experimental Details

In this section, we provide a concrete description of our experiments and detailed hyper-parameters of RAT. For each run of experiments, we run on a single Nvidia Tesla V100 GPUs and 16 CPU cores (Intel Xeon Gold 6230 CPU @ 2.10GHz) for training.

# E.1 Tasks

In phase one of our experiments, we evaluate our method on eight robotic manipulation tasks obtained from Meta-world (Yu et al. 2020). These tasks serve as a representative set for testing the effectiveness of our approach. In phase two, we further assess our method on two locomotion tasks sourced from Mujoco (Todorov, Erez, and Tassa 2012). By including tasks from both domains, we aim to demonstrate the versatility and generalizability of our approach across different task types. The specific tasks we utilize in our experiments are as follows:

# Meta-world

- Door Lock: An agent controls a simulated Sawyer arm to lock the door.   
- Door Unlock: An agent controls a simulated Sawyer arm to unlock the door.   
- Drawer Open: An agent controls a simulated Sawyer arm to open the drawer to a target position.   
- Drawer Close: An agent controls a simulated Sawyer arm to close the drawer to a target position.   
- Faucet Open: An agent controls a simulated Sawyer arm to open the faucet to a target position.   
- Faucet Close: An agent controls a simulated Sawyer arm to close the faucet to a target position.   
- Window Open: An agent controls a simulated Sawyer arm to open the window to a target position.   
- Window Close: An agent controls a simulated Sawyer arm to close the window to a target position.

# Mujoco

- Half Cheetah: A 2d robot with nine links and eight joints aims to learn to run forward (right) as fast as possible.   
- Walker: A 2d two-legged robot aims to move in the forward (right).

# E.2 Hyper-parameters Setting

In our experiments, we adopt the PEBBLE (Lee, Smith, and Abbeel 2021) as our baseline approach for reward learning from human feedback. It is worth to emphasize that the PA-AD (oracle) (Zhang et al. 2021) and SA-RL (oracle) (Sun et al. 2022) use the truth victim reward function. To ensure a fair comparison, All methods employ the same neural network structure and keep the same parameter settings as described in their work. The specific hyper-parameters for SA-RL are provided in Table 5.

Table 4: Hyper-parameters of RAT for adversary training. 

<table><tr><td>Hyper-parameter</td><td>Value</td><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Number of layers</td><td>3</td><td>Hidden units of each layer</td><td>256</td></tr><tr><td>Learning rate</td><td>0.0003</td><td>Batch size</td><td>1024</td></tr><tr><td>Length of segment</td><td>50</td><td>Number of reward functions</td><td>3</td></tr><tr><td>Frequency of feedback</td><td>5000</td><td>Feedback batch size</td><td>128</td></tr><tr><td>Adversarial budget</td><td>0.1</td><td> $(\beta_1, \beta_2)$ </td><td>(0.9, 0.999)</td></tr></table>

Table 5: Hyper-parameters of SA-RL for adversary training. 

<table><tr><td>Hyper-parameter</td><td>Value</td><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Number of layers</td><td>3</td><td>Hidden units of each layer</td><td>256</td></tr><tr><td>Learning rate</td><td>0.00005</td><td>Mini-Batch size</td><td>32</td></tr><tr><td>Length of segment</td><td>50</td><td>Number of reward functions</td><td>3</td></tr><tr><td>Frequency of feedback</td><td>5000</td><td>Feedback batch size</td><td>128</td></tr><tr><td>Adversarial budget</td><td>0.1</td><td>Entropy coefficient</td><td>0.0</td></tr><tr><td>Clipping parameter</td><td>0.2</td><td>Discount  $\gamma$ </td><td>0.99</td></tr><tr><td>GAE lambda</td><td>0.95</td><td>KL divergence target</td><td>0.01</td></tr></table>

# E.3 Victim Agents Settings

Our experiment is divided into two phases. In the first phase, we conduct experiments using a variety of simulated robotic manipulation tasks from the Meta-world environment. In the second phase, we shift our focus to two continuous control environments from the OpenAI Gym MuJoCo suite.

![](images/8b19e9df8061e9422ece96b49da331ebc28fae2450e161622421d8030d91d3f6.jpg)

<details>
<summary>line</summary>

| Env. Steps (x10^6) | door-lock-v2 | door-unlock-v2 | window-open-v2 | window-close-v2 | drawer-open-v2 | drawer-close-v2 | faucet-open-v2 | faucet-close-v2 |
| ------------------ | ------------ | -------------- | -------------- | --------------- | --------------- | ---------------- | --------------- | ---------------- |
| 0.0                | ~45          | ~35            | ~15            | ~10             | ~30             | ~40              | ~25             | ~35              |
| 0.2                | ~95          | ~90            | ~95            | ~95             | ~95             | ~95              | ~95             | ~95              |
| 0.4                | ~98          | ~95            | ~98            | ~98             | ~98             | ~98              | ~98             | ~98              |
| 0.6                | ~98          | ~95            | ~98            | ~98             | ~98             | ~98              | ~98             | ~98              |
| 0.8                | ~98          | ~95            | ~98            | ~98             | ~98             | ~98              | ~98             | ~98              |
| 1.0                | ~98          | ~95            | ~98            | ~98             | ~98             | ~98              | ~98             | ~98              |
</details>

Figure 6: The evaluation curves for the training of victim agents are measured based on their success rate well-designed in Meta-world (Yu et al. 2020).

Meta-world. The victim models for Meta-world tasks are trained using the Soft Actor-Critic (SAC) algorithm, as introduced by Haarnoja et al. (2018). Implementation is based on the open-source repository available at \*. In each agent's training, fully connected neural networks are utilized both as the policy network and for the double Q networks. The specific hyperparameters employed in our experiments are detailed in Table 6. As depicted in Figure 6, each victim agent has been thoroughly trained to master a specific manipulation skill.

Mujoco. To demonstrate the vulnerability of the Decision Transformer, we employ well-trained models of expert-level proficiency. Specifically, we utilize the Cheetah agent\* and the Walker agent\*, both of which are based on Decision Transformer (Chen et al. 2021) models. These models have been trained on expert trajectories sampled from the Gym environment.

Table 6: Hyper-parameters of SAC for victim training. 

<table><tr><td>Hyper-parameter</td><td>Value</td><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Total training steps</td><td> $10^6$ </td><td>Replay buffer capacity</td><td> $10^6$ </td></tr><tr><td>Number of layers</td><td>3</td><td>Initial temperature</td><td>0.1</td></tr><tr><td>Hidden units of each layer</td><td>256</td><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate</td><td>0.0001</td><td>Critic target update freq</td><td>2</td></tr><tr><td>Discount γ</td><td>0.99</td><td>Critic EMA τ</td><td>0.005</td></tr><tr><td>Batch size</td><td>1024</td><td> $(\beta_1, \beta_2)$ </td><td>(0.9, 0.999)</td></tr><tr><td>Random steps</td><td>5000</td><td>Agent update frequency</td><td>1</td></tr></table>

# E.4 Scenario Design

To assess the efficacy of our method, we meticulously crafted two experimental setups: the Manipulation Scenario and the Opposite Behavior Scenario.

Scenario Description. In both scenarios, the victim agent is a proficiently trained policy in robotic tasks, as detailed in E.3. In the Manipulation Scenario, the adversary's aim is to alter the agent's behavior via targeted adversarial attacks, compelling the agent to grasp objects distant from the initially intended target location. The successful completion of these grasping actions signifies the effectiveness of the adversarial attack. Conversely, in the Opposite Behavior Scenario, the victim policy is a well-established policy in simulated robotic manipulation tasks. Here, the adversary's objective is to manipulate the agent's behavior to perform actions contrary to its original purpose. For example, if the policy is originally designed to open windows, the attacker endeavors to deceive the agent into closing them instead.

Table 7: Success metrics for the Meta-world tasks are quantified in meters. The metrics for the first four rows are sourced from (Yu et al. 2020), and we utilize the built-in functions provided therein without any alterations. The Manipulation Scenario metric, devised by us, is applied across all tasks within the Manipulation Scenario. 

<table><tr><td>Task</td><td>Success Metric</td><td>Task</td><td>Success Metric</td></tr><tr><td>door-lock</td><td> $\mathbb{I}_{\|o-t\|_{2}<0.02}$ </td><td>door-unlock</td><td> $\mathbb{I}_{\|o-t\|_{2}<0.02}$ </td></tr><tr><td>drawer-open</td><td> $\mathbb{I}_{\|o-t\|_{2}<0.03}$ </td><td>drawer-close</td><td> $\mathbb{I}_{\|o-t\|_{2}<0.055}$ </td></tr><tr><td>faucet-open</td><td> $\mathbb{I}_{\|o-t\|_{2}<0.07}$ </td><td>faucet-close</td><td> $\mathbb{I}_{\|o-t\|_{2}<0.07}$ </td></tr><tr><td>window-open</td><td> $\mathbb{I}_{\|o-t\|_{2}<0.05}$ </td><td>window-close</td><td> $\mathbb{I}_{\|o-t\|_{2}<0.05}$ </td></tr><tr><td>Manipulation Scenario</td><td> $\mathbb{I}_{\|o-t\|_{2}<0.05}$ </td><td colspan="2"></td></tr></table>

Evaluation Metric. The success metric for all our tasks revolves around the proximity between the task-relevant object and its final goal position, denoted as $I_{\|o-t\|_{2}<\epsilon}$ , where $\epsilon$ is a minimal distance threshold, such as 5 cm. In the Manipulation Scenario, we set $\epsilon = 0.05$ (5cm). For the Opposite Behavior Scenario, we apply the success metrics and thresholds specified for each task by Meta-world (Yu et al. 2020). We summarize the all success metrics in our experiments in the Table 7.

# F Full Experiments

F.1 Full Experiment Results   
![](images/cee535f7dc2707abf0f225779e5b2f4e21db3ebedd4ba7217e44c724d4531cf7.jpg)

Figure 7: Training curves of different methods on various tasks in the manipulation scenario. The solid line and shaded area denote the mean and the standard deviation of success rate, respectively, over ten runs. The red line (our method) outperforms all the baselines in PbRL setting and even exceeds most baselines in oracle setting.   
![](images/3356c15a7d1f26415a56243097de7ff73c44373aea4dadeb67a4c0597c1d557b.jpg)  
Figure 8: Training curves of all methods on various tasks in the opposite behaviors scenario. The solid line and shaded area denote the mean and the standard deviation of success rate over ten runs. In this scenario, the red line (our method) outperforms all the baselines in both PbRL setting and oracle setting, which demonstrates the effectiveness of RAT.

# F.2 Ablation studies

Impact of Feedback Amount. We evaluate the performance of RAT using different numbers of preference labels. Table 8 presents the results across varying numbers of labels: 3000, 5000, 7000, 9000 for the Drawer Open task in the manipulation scenario and 1000, 3000, 5000, 7000 for the Faucet Close task in the opposite behavior scenario. The experimental results demonstrate that increasing the number of human feedback labels significantly improves the performance of RAT, leading to a stronger adversary and a more stable attack success rate. For instance, in the Drawer Open task, the attack success rate increases by 47.6% when the number of labels rises from 3000 to 9000, demonstrating the importance of adequate feedback for effective adversary learning. In contrast, SA-RL and PA-AD exhibit poor performance even with sufficient feedback, with PA-AD failing entirely in the manipulation scenario. This is likely due to the limited exploration space in these methods, constrained by the fixed victim policy. In contrast, RAT enables better exploration by incorporating an intention policy, allowing for more dynamic interactions and improved performance in complex tasks.

Table 8: Success rate of different methods with varying numbers of preference labels on the Drawer Open task in the manipulation scenario and the Faucet Close task in the opposite behavior scenario. The success rate is reported as the mean and standard deviation over 30 episodes. 

<table><tr><td>Environment</td><td>Feedback</td><td>RAT (ours)</td><td>PA-AD</td><td>SA-RL</td></tr><tr><td rowspan="4">Drawer Open(manipulation)</td><td>3000</td><td>65.7% ± 37.1%</td><td>0.0% ± 0.0%</td><td>8.3% ± 13.2%</td></tr><tr><td>5000</td><td>86.7% ± 18.1%</td><td>0.0% ± 0.0%</td><td>21.3% ± 18.9%</td></tr><tr><td>7000</td><td>95.7% ± 13.6%</td><td>0.0% ± 0.0%</td><td>28.0% ± 28.1%</td></tr><tr><td>9000</td><td>97.0% ± 6.9%</td><td>0.0% ± 0.0%</td><td>13.0% ± 18.5%</td></tr><tr><td rowspan="4">Faucet Close(opposite behavior)</td><td>1000</td><td>69.7% ± 35.2%</td><td>16.7% ± 9.4%</td><td>2.0% ± 6.0%</td></tr><tr><td>3000</td><td>79.0% ± 16.2%</td><td>29.0% ± 14.0%</td><td>6.0% ± 11.7%</td></tr><tr><td>5000</td><td>95.3% ± 9.2%</td><td>21.3% ± 12.8%</td><td>3.3% ± 12.7%</td></tr><tr><td>7000</td><td>95.3% ± 7.6%</td><td>22.7% ± 12.4%</td><td>4.0% ± 7.1%</td></tr></table>

![](images/ee27cee6372a7a64e88219ddee2a4735d9c4a19a37809b50d3df6dc4070ef867.jpg)  
Figure 9: Training curves of success rate with different adversarial budgets on Drawer Open for the manipulation scenario and Faucet Close for the opposite behavior scenario. The solid line and shaded area denote the mean and the standard deviation of the success rate across five runs.

Impact of Different Attack Budgets. We also investigate the impact of the attack budget on the performance. To gain further insights, we conduct additional experiments with different attack budgets: 0.05, 0.075, 0.1, 0.15 for the Drawer Open task and 0.02, 0.05, 0.075, 0.1 for the Faucet Close task in the respective scenarios. In Figure 9, we present the performance of the baseline method and RAT with different attack budgets. The experimental results demonstrate that the performance of all methods improves with an increase in the attack budget.

![](images/6dbd3ee2133a3575ce77a718e3da71fa4b77846608add6024e4fcaade339e75a.jpg)

<details>
<summary>line</summary>

| Env. Steps | Learned Reward | True Reward |
| ---------- | -------------- | ----------- |
| 0          | -5.0           | -4.0        |
| 50         | -1.0           | -3.0        |
| 100        | 0.0            | 0.0         |
| 200        | 0.0            | 0.0         |
| 300        | 0.0            | 0.0         |
| 400        | 0.0            | 0.0         |
| 500        | 0.0            | 0.0         |
</details>

(a) Faucet Open

![](images/4966c3c911109bd5bc0911ec8c8027b6e738f1e8d5ee4736e49a583889b26944.jpg)

<details>
<summary>line</summary>

| Env. Steps | Learned Reward | True Reward |
| ---------- | -------------- | ----------- |
| 0          | -5.0           | -3.5        |
| 50         | -1.0           | -2.5        |
| 100        | 0.5            | 0.0         |
| 150        | 0.0            | 0.0         |
| 200        | -0.5           | 0.0         |
| 250        | 0.5            | 0.0         |
| 300        | 0.0            | 0.0         |
| 350        | 0.5            | 0.0         |
| 400        | 0.0            | 0.0         |
| 450        | -0.5           | 0.0         |
| 500        | -1.0           | 0.0         |
</details>

(b) Faucet Close

![](images/ad57cc43e225078a7db39ba0123d254b0cf3da246c5809a59eea73a0abcf87a5.jpg)

<details>
<summary>line</summary>

| Env. Steps | Learned Reward | True Reward |
| ---------- | -------------- | ----------- |
| 0          | 1.0            | 4.0         |
| 50         | -1.0           | -3.0        |
| 100        | 0.5            | 0.0         |
| 150        | 0.0            | -0.5        |
| 200        | 0.5            | 0.0         |
| 250        | 0.0            | -0.5        |
| 300        | 0.5            | 0.0         |
| 350        | 0.0            | -0.5        |
| 400        | 0.5            | 0.0         |
| 450        | 0.0            | -0.5        |
| 500        | 0.5            | 0.0         |
</details>

(c) Drawer Open

![](images/bcc5d1c612992db411053ac8427407cc53ef6ebb0cb93ef2700d697d4e2c9b61.jpg)

<details>
<summary>line</summary>

| Env. Steps | Learned Reward | True Reward |
| ---------- | -------------- | ----------- |
| 0          | -1.5           | -1.8        |
| 50         | -2.0           | -1.9        |
| 100        | -2.5           | -1.8        |
| 150        | 0.5            | 0.6         |
| 200        | 0.4            | 0.5         |
| 250        | 0.3            | 0.4         |
| 300        | 0.4            | 0.5         |
| 350        | 0.3            | 0.4         |
| 400        | 0.4            | 0.5         |
| 450        | 0.3            | 0.4         |
| 500        | 0.4            | 0.5         |
</details>

(d) Drawer Close   
Figure 10: Quality of learned reward. Time series of the normalized learned reward (blue) and the ground truth reward (orange). These rewards are obtained from rollouts generated by a policy optimized using RAT.

Quality of learned reward functions. We further analyze the quality of the reward functions learned by RAT compared to the true reward function. In Figure 10, we present four time series plots that depict the normalized learned reward (blue) and the ground truth reward (orange). These plots represent two scenarios: opposite behaviors and manipulation tasks. The results indicate that the learned reward function aligns well with the true reward function derived from human feedback. This alignment is achieved by capturing various human intentions through the preference data.

# G Discussion

In this work, we propose RAT, a novel adversarial attack framework targeting deep reinforcement learning (DRL) agents for inducing specific behaviors. RAT integrates three core components: an intention policy, an adversary, and a weighting function, all trained simultaneously. Unlike prior approaches that rely on predefined target policies, RAT dynamically trains an intention policy aligned with human preferences, offering a flexible and adaptive behavioral target for the adversary. Leveraging advancements in preference-based reinforcement learning (PbRL), the intention policy effectively captures human intent during training. The adversary perturbs the victim agent's observations, steering the agent toward behaviors specified by the intention policy. To enhance attack efficacy, the weighting function adjusts the state occupancy measure, optimizing the distribution of states encountered during training. This optimization improves both the effectiveness and efficiency of the attack. Through iterative refinement, RAT achieves superior precision in directing the victim agent toward human-desired behaviors compared to existing adversarial attack methods.

An important future direction is the extension of targeted adversarial attacks to LLM-based agents and Vision-Language-Action (VLA) models, which have become increasingly impactful in various practical applications driven by advancements in large-scale models (Zhu et al. 2024b,c; Jin, Zhang, and Zong 2023; Zhu et al. 2024a; Jin, Liu, and Tan 2024). Investigating the vulnerabilities of these models to targeted adversarial attacks is crucial for identifying security risks and improving their robustness. These studies can provide valuable insights for designing more resilient architectures and effective defense mechanisms. Additionally, analyzing the behavior of these models under adversarial perturbations in complex real-world scenarios is critical to ensuring their reliability and safety in practical deployments.